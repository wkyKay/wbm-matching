"""Label-free adapters for the frozen gallery and the separate exact evaluator.

The source run is read-only. All newly computed features and train-fitted Yu
encoders belong to a fresh evaluation destination, never to the source run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .common import get_device, hash_file, save_json, signature
from .exact_metrics import order_scores


@dataclass
class RankingResult:
    ids: np.ndarray
    scores: np.ndarray
    coarse_distances: np.ndarray | None = None
    scored_count: int = 0
    eligible_count: int = 0
    coarse_scored_count: int = 0


def _ids(values, name):
    result = np.asarray(values)
    if result.size == 0 and result.ndim == 1:
        return result.astype(np.int64)
    if (result.ndim != 1 or result.dtype.kind not in "iu" or np.any(result < 0)
            or len(np.unique(result)) != len(result)):
        raise ValueError(f"{name} must contain unique nonnegative integer sample IDs")
    return result.astype(np.int64, copy=False)


def _save_arrays(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _extract_one(task):
    method, sample_id, wafer, settings = task
    if method == "wu":
        from methods.wu.wu import encode_wafer
        result = encode_wafer(wafer, settings)
    elif method == "wbbs":
        from methods.wbbs.wbbs import encode_wafer
        result = encode_wafer(wafer, settings)
    else:
        from methods.wmhd.wmhd import extract_features
        result = extract_features(wafer, settings)
    return sample_id, result


def _extract_gallery(method, bundle, ids, settings, workers):
    tasks = ((method, int(i), np.asarray(bundle.maps[int(i)]), settings) for i in ids)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = executor.map(_extract_one, tasks, chunksize=8) if executor else map(_extract_one, tasks)
    features = []
    last_report = time.monotonic()
    try:
        for index, (sample_id, feature) in enumerate(results):
            if sample_id != int(ids[index]):
                raise RuntimeError("Parallel feature extraction changed sample order")
            features.append(feature)
            now = time.monotonic()
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(ids) or now - last_report >= 30:
                print(f"{method} evaluation features: {index + 1}/{len(ids)}", flush=True)
                last_report = now
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return features


def _yu_infer(model, maps, sample_ids, config, device, *, features):
    """Use Yu's exact inference preprocessing without reading evaluation labels."""
    import torch
    from methods.yu import yu

    batch_size = config.get("inference_batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("inference_batch_size must be a positive integer")
    ids = _ids(sample_ids, "Yu inference IDs")
    predictions, embeddings = [], []
    reported, last_report = 0, time.monotonic()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            images = torch.stack([yu.preprocess(maps[int(i)]) for i in ids[start:start + batch_size]]).to(device)
            if features:
                h = model.forward_features(images)
                logits = model.fc3(h)
                embeddings.append(h.cpu().numpy())
            else:
                logits = model(images)
            predictions.append(logits.argmax(1).cpu().numpy())
            done, now = start + len(images), time.monotonic()
            if done - reported >= 100 or done == len(ids) or now - last_report >= 30:
                print(f"Yu evaluation inference: {done}/{len(ids)}", flush=True)
                reported, last_report = done, now
    return {"ids": ids,
            "predictions": np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64),
            "features": np.concatenate(embeddings) if embeddings else np.empty((0, 1024), dtype=np.float32)}


class PreparedMethod:
    """One gallery's features; rank one explicit candidate list at a time."""

    def __init__(self, method, ids, data, settings, metadata, *, tie_seed, variant="global"):
        self.method, self.ids, self.data, self.settings = method, ids, data, settings
        self.metadata, self.tie_seed, self.variant = metadata, tie_seed, variant
        self.rows = {int(sample_id): row for row, sample_id in enumerate(ids)}

    def rank(self, query_id, candidate_ids):
        if isinstance(query_id, (bool, np.bool_)) or not isinstance(query_id, (int, np.integer)):
            raise ValueError("query_id must be an integer sample ID")
        query_id = int(query_id)
        ids = _ids(candidate_ids, "Candidate IDs")
        if query_id in ids:
            raise ValueError("Candidate IDs must exclude the query itself")
        if query_id not in self.rows or any(int(i) not in self.rows for i in ids):
            raise ValueError("Query and candidates must belong to the prepared gallery")
        qrow = self.rows[query_id]
        rows = np.asarray([self.rows[int(i)] for i in ids], dtype=np.int64)
        if self.method == "wu":
            return self._rank_wu(query_id, qrow, ids, rows)
        if self.method == "wbbs":
            from methods.wbbs.wbbs import score_candidates
            scores = score_candidates(self.data[qrow], (self.data[i] for i in rows), self.settings)
        elif self.method == "wmhd":
            from methods.wmhd.wmhd import score_candidates
            scores = score_candidates(self.data[qrow], (self.data[i] for i in rows), self.settings)
        elif self.method == "nakazawa":
            from methods.nakazawa.nakazawa import score_candidates
            scores = score_candidates(self.data["codes"][qrow], self.data["codes"][rows])
        else:
            from methods.yu.yu import categorical_hamming
            scores = -categorical_hamming(self.data["codes"][qrow], self.data["codes"][rows])
        scores = np.asarray(scores, dtype=np.float64)
        scored_count = len(scores)
        if self.variant == "predicted_class":
            keep = (self.data["predicted_defect"][qrow]
                    & self.data["predicted_defect"][rows]
                    & (self.data["predicted_classes"][rows] == self.data["predicted_classes"][qrow]))
            ids, scores = ids[keep], scores[keep]
        order = order_scores(ids, scores, query_id, seed=self.tie_seed)
        return RankingResult(ids[order], scores[order], scored_count=scored_count,
                             eligible_count=len(ids))

    def _rank_wu(self, query_id, qrow, ids, rows):
        features, full, regions = self.data
        distances = np.linalg.norm(np.asarray(features[rows], dtype=np.float64)
                                   - np.asarray(features[qrow], dtype=np.float64), axis=1)
        if not np.isfinite(distances).all():
            raise ValueError("Wu coarse distances contain non-finite values")
        # Apply the shared tie rule BEFORE the shortlist boundary is cut.
        coarse = order_scores(ids, -distances, query_id, seed=self.tie_seed)
        coarse = coarse[:min(int(self.settings["coarse_candidates"]), len(ids))]
        beta = float(self.settings["beta"])
        scores = np.clip(beta * (full[rows[coarse]] @ full[qrow])
                         + (1.0 - beta) * (regions[rows[coarse]] @ regions[qrow]), -1.0, 1.0)
        order = order_scores(ids[coarse], scores, query_id, seed=self.tie_seed,
                             secondary=distances[coarse])
        selected = coarse[order]
        return RankingResult(ids[selected], np.asarray(scores[order], dtype=np.float64),
                             distances[selected], scored_count=len(coarse),
                             eligible_count=len(coarse), coarse_scored_count=len(ids))


def prepare_method(bundle, config, pool, destination, source_run_signature=None,
                   device=None, workers=1, variant="global", tie_seed=2026):
    """Prepare only pool.gallery; Yu additionally fits its encoder on train.

    The runner must validate the frozen pool and source-run provenance first.
    No existing artifacts are reused or overwritten by this function.
    """
    method = config["method"]
    if method not in ("wu", "yu", "wbbs", "nakazawa", "wmhd"):
        raise ValueError(f"Unsupported method: {method}")
    if variant not in ("global", "predicted_class") or (variant != "global" and method != "yu"):
        raise ValueError("predicted_class is only available for Yu")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if isinstance(tie_seed, bool) or not isinstance(tie_seed, int) or tie_seed < 0:
        raise ValueError("tie_seed must be a nonnegative integer")
    ids = _ids(pool.gallery, "Gallery IDs")
    if not len(ids):
        raise ValueError("Cannot prepare an empty gallery")
    gallery_set = set(map(int, ids))
    if any(int(q) not in gallery_set for q in pool.queries):
        raise ValueError("Every query must belong to the frozen gallery")
    destination = Path(destination)
    source = Path(config["output_dir"])
    if destination.resolve() == source.resolve():
        raise ValueError("Evaluation destination must differ from the source run")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Evaluation feature destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {"method": method, "variant": variant, "split": pool.split_name,
                "gallery_count": len(ids), "query_count": len(pool.queries),
                "protocol_signature": pool.protocol_signature,
                "source_run": str(source.resolve()), "source_run_signature": source_run_signature,
                "score_direction": "higher_is_better", "tie_seed": tie_seed,
                "labels_used_for_ranking": False, "normal_holdout_used": False,
                "scored_count_definition": "final similarity scores computed; Wu counts only NCC shortlist",
                "eligible_count_definition": "results eligible after method shortlist or prediction gate",
                "coarse_scored_count_definition": "Wu full-gallery Euclidean scores; zero for other methods",
                "files": {}}
    settings = config.get("parameters", {})
    feature_path = destination / "features.npz"
    if method in ("wu", "wbbs", "wmhd"):
        if method == "wu":
            from methods.wu.wu import parameters
        elif method == "wbbs":
            from methods.wbbs.wbbs import parameters
        else:
            from methods.wmhd.wmhd import parameters
        settings = parameters(settings)
        extracted = _extract_gallery(method, bundle, ids, settings, workers)
        if method == "wu":
            # Match Wu's existing float32 feature/NCC caches before ranking.
            data = tuple(np.stack([feature[j] for feature in extracted]).astype(np.float32, copy=False)
                         for j in range(3))
            _save_arrays(feature_path, sample_ids=ids, features=data[0], full=data[1], regions=data[2])
            metadata["coarse_candidates"] = int(settings["coarse_candidates"])
            metadata["ranking_secondary_key"] = "Euclidean distance ascending"
        elif method == "wbbs":
            data = extracted
            lengths = [len(feature.points) for feature in data]
            _save_arrays(feature_path, sample_ids=ids, offsets=np.r_[0, np.cumsum(lengths)],
                         points=np.concatenate([f.points for f in data]),
                         descriptors=np.concatenate([f.descriptors for f in data]),
                         original_counts=np.asarray([f.original_count for f in data], dtype=np.int64))
            metadata["empty_features"] = sum(n == 0 for n in lengths)
        else:
            data = extracted
            lengths = [len(feature.points) for feature in data]
            _save_arrays(feature_path, sample_ids=ids, offsets=np.r_[0, np.cumsum(lengths)],
                         points=np.concatenate([f.points for f in data]),
                         neighborhoods=np.concatenate([f.neighborhoods for f in data]),
                         scalars=np.asarray([[f.radius, f.beta, f.raw_defects, f.support_dies] for f in data]))
            metadata.update(empty_after_esrn=sum(n == 0 for n in lengths),
                            distance_direction="query_to_candidate", empty_query_negative_infinity_allowed=True)
        metadata["parameters"] = settings
    else:
        execution_device = get_device(str(device if device is not None else config.get("device", "auto")))
        metadata["device"] = str(execution_device)
        if method == "nakazawa":
            from methods.nakazawa import nakazawa
            model, execution_device = nakazawa.load_trained_model(bundle, config, device=execution_device)
            features, codes = nakazawa.encode_maps(model, bundle.maps, ids,
                batch_size=config.get("inference_batch_size", 8), device=execution_device)
            data = {"codes": codes}
            _save_arrays(feature_path, sample_ids=ids, sigmoid_features=features, codes=codes)
            metadata.update(checkpoint_sha256=hash_file(source / "checkpoint_last.pt"),
                            model_selection="last_epoch", distance="binary Hamming count")
            del model
        else:
            from methods.yu import yu
            if execution_device.type not in ("cpu", "cuda"):
                raise ValueError("Yu supports CPU or CUDA execution")
            if not source_run_signature:
                raise ValueError("Yu requires its validated checkpoint run signature")
            model, provenance = yu._selected_model("classifier", source, source_run_signature, execution_device)
            train_ids = _ids(bundle.train_ids, "Yu PCA train IDs")
            if np.intersect1d(train_ids, ids).size:
                raise ValueError("Yu PCA train IDs must be disjoint from evaluation gallery")
            training = _yu_infer(model, bundle.maps, train_ids, config, execution_device, features=True)
            encoder = yu.TenValueEncoder(seed=int(config["seed"])).fit(training["features"], training["ids"])
            classifier_hash = hash_file(source / "classifier" / "best.pt")
            encoder_signature = signature({"run": source_run_signature, "classifier": classifier_hash})
            encoder_path = destination / "encoder.npz"
            encoder.save(encoder_path, encoder_signature)
            metadata["files"][encoder_path.name] = hash_file(encoder_path)
            del training
            evaluated = _yu_infer(model, bundle.maps, ids, config, execution_device, features=True)
            codes = encoder.transform(evaluated["features"])
            data = {"codes": codes, "predicted_classes": evaluated["predictions"]}
            del model
            metadata.update(classifier=provenance, classifier_sha256=classifier_hash,
                            pca_fit_split="train", pca_fit_count=len(train_ids),
                            pca_dimensions=256, pca_scaling="train min-max; clip [0,1]",
                            distance="categorical Hamming count")
            if variant == "predicted_class":
                detector, provenance = yu._selected_model("detector", source, source_run_signature, execution_device)
                detected = _yu_infer(detector, bundle.maps, ids, config, execution_device, features=False)
                data["predicted_defect"] = np.asarray(detected["predictions"], dtype=bool)
                metadata.update(detector=provenance, detector_sha256=hash_file(source / "detector" / "best.pt"))
                del detector
            _save_arrays(feature_path, sample_ids=ids, **data)
    metadata["files"][feature_path.name] = hash_file(feature_path)
    metadata["prepare_seconds"] = time.perf_counter() - started
    save_json(destination / "prepared_metadata.json", metadata)
    return PreparedMethod(method, ids, data, settings, metadata, tie_seed=tie_seed, variant=variant)

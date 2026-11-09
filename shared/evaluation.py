"""Independent unified evaluation, without mutating training runs.

The original method entry points and their manifests remain unchanged. This
runner has its own version/signature and always requires a fresh output folder.
"""
from __future__ import annotations

import csv
from collections import deque
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from . import common
from . import coverage_metrics
from .common import hash_file, read_json, save_json, signature
from .retrieval_protocol import protocol_file_hashes
from .exact_metrics import DEFAULT_KS, METRIC_VERSION, evaluate_rankings
from .evaluation_adapters import prepare_method


METHODS = ("wu", "yu", "wbbs", "nakazawa", "wmhd")
TRAINED_METHODS = ("yu", "nakazawa")
SMOKE_KS = (1, 5, 10, 20)
RUNNER_VERSION = "unified-evaluation-v1"
TIE_POLICY = "SHA256(exact_v1:tie:seed:query_id:candidate_id), first 8 bytes big-endian; ID collision fallback"


_RANK_WORKER = None


def _initialize_rank_worker(ranker):
    global _RANK_WORKER
    _RANK_WORKER = ranker


def _rank_query(task):
    query_id, candidates = task
    started = time.perf_counter()
    ranked = _RANK_WORKER.rank(int(query_id), candidates)
    return ranked, time.perf_counter() - started


def iter_ranked_queries(ranker, items, workers=1):
    """Ordered, bounded query execution; each worker sees the full candidate list.

    Spawn avoids inheriting active BLAS/CUDA thread state. At most 2*workers
    queries are pending; never materialize the test query/candidate Cartesian
    product. Only immutable prepared features are sent to each worker once.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("rank_workers must be a positive integer")
    if workers == 1:
        for query_id, candidates in items:
            started = time.perf_counter()
            ranked = ranker.rank(int(query_id), candidates)
            yield query_id, candidates, ranked, time.perf_counter() - started
        return
    iterator, pending = iter(items), deque()
    executor = ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_rank_worker, initargs=(ranker,))
    try:
        def submit_next():
            try:
                query_id, candidates = next(iterator)
            except StopIteration:
                return False
            pending.append((query_id, candidates,
                            executor.submit(_rank_query, (query_id, candidates))))
            return True

        for _ in range(2 * workers):
            if not submit_next():
                break
        while pending:
            query_id, candidates, future = pending.popleft()
            ranked, elapsed = future.result()
            yield query_id, candidates, ranked, elapsed
            submit_next()
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _settings(config):
    # Match the existing ensure_run contract; evaluation options never enter it.
    ignored = {"output_dir", "device", "workers", "num_workers", "cpu_threads", "torch_threads",
               "data_file", "split_file", "retrieval_protocol_dir", "cache_dir", "resume"}
    return {k: v for k, v in config.items() if k not in ignored}


def _method_path(method):
    return common.ROOT / "methods" / method / f"{method}.py"


def _source_hashes(config):
    return {"data_file": hash_file(config["data_file"]),
            "split_file": hash_file(config["split_file"]),
            "retrieval_protocol": protocol_file_hashes(config["retrieval_protocol_dir"])}


def verify_source_run(config):
    """Reproduce the original manifest signature using read-only operations."""
    folder = Path(config["output_dir"])
    marker = folder / "run_manifest.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Train {config['method']} first; missing source manifest: {marker}")
    files = [Path(common.__file__).resolve(), common.PROTOCOL_SOURCE,
             common.RETRIEVAL_SOURCE, _method_path(config["method"])]
    code = {str(p.relative_to(common.ROOT)): hash_file(p) for p in files}
    payload = {"format_version": 3, "settings": _settings(config),
               "sources": _source_hashes(config), "code": code}
    record = read_json(marker)
    expected = signature(payload)
    if record != {**payload, "signature": expected}:
        raise ValueError("Training run code/config/data/protocol signature mismatch; "
                         "use the matching config and source run, never edit its manifest")
    return {"path": str(folder.resolve()), "base_signature": expected,
            "manifest_sha256": hash_file(marker), "sources": payload["sources"]}


def validate_training_source(bundle, config, source_record, variant="global"):
    """Check completion separately from best-checkpoint selection, read-only."""
    folder = Path(config["output_dir"])
    data_summary = read_json(folder / "protocol" / "data_summary.json")
    if data_summary.get("signature") != bundle.signature:
        raise ValueError("Training data identity differs from evaluation data identity")
    smoke = bool(config.get("smoke", False))
    provenance = {**source_record, "smoke_training": smoke, "files": {}}
    if config["method"] == "yu":
        from methods.yu import yu
        checkpoint_signature = signature({"run": source_record["base_signature"],
                                          "data": bundle.signature, "method": yu.METHOD_VERSION})
        tasks = ("classifier", "detector") if variant == "predicted_class" else ("classifier",)
        provenance["tasks"] = {}
        for task in tasks:
            summary_path = folder / task / "summary.json"
            summary = read_json(summary_path)
            options = config["training"][task]
            epochs = int(options["epochs"])
            if summary.get("signature") != checkpoint_signature or summary.get("task") != task:
                raise ValueError(f"Yu {task} training summary signature mismatch")
            if summary.get("configured_epochs") != epochs:
                raise ValueError(f"Yu {task} configured epochs differ")
            last_path = folder / task / "last.pt"
            last = yu._load_checkpoint(last_path)
            if (last.get("signature") != checkpoint_signature or last.get("task") != task
                    or last.get("method_version") != yu.METHOD_VERSION):
                raise ValueError(f"Yu {task} last checkpoint signature mismatch")
            state = last["state"]
            if any(summary.get(summary_key) != state.get(state_key) for summary_key, state_key in
                   (("completed_epochs", "epoch"), ("next_batch", "next_batch"),
                    ("truncated_epoch", "truncated_epoch"), ("steps", "steps"))):
                raise ValueError(f"Yu {task} checkpoint/summary progress mismatch")
            train_size = len(yu._task_ids(bundle, task, "train"))
            expected_steps = epochs * math.ceil(train_size / int(options["batch_size"]))
            complete = (state.get("epoch") == epochs and state.get("next_batch") == 0
                        and state.get("truncated_epoch") is False
                        and state.get("steps") == expected_steps)
            if not smoke and (not complete or summary.get("training_complete") is not True):
                raise ValueError(f"Formal Yu {task} evaluation requires completed training; resume training first")
            del last
            best_path = folder / task / "best.pt"
            best = yu._load_checkpoint(best_path)
            if (best.get("signature") != checkpoint_signature or best.get("task") != task
                    or best.get("method_version") != yu.METHOD_VERSION):
                raise ValueError(f"Yu {task} selected checkpoint signature mismatch")
            # An earlier best epoch is valid; completion belongs to last.pt.
            provenance["tasks"][task] = {"training_complete": complete,
                "selected_steps": best["state"]["steps"],
                "selected_completed_epochs": best["state"]["epoch"],
                "selected_truncated_epoch": best["state"]["truncated_epoch"],
                "selection": "minimum valid CE; original method convention"}
            del best
            for path in (summary_path, last_path, best_path):
                provenance["files"][str(path.relative_to(folder))] = hash_file(path)
        provenance["detector_used"] = variant == "predicted_class"
    else:
        from methods.nakazawa import nakazawa
        params = nakazawa.training_parameters(config.get("training"), smoke=smoke)
        marker = folder / "training_complete.json"
        record = read_json(marker)
        checkpoint_path = folder / "checkpoint_last.pt"
        if (record.get("checkpoint_sha256") != hash_file(checkpoint_path)
                or record.get("epochs") != params["epochs"]
                or record.get("selection") != "last_epoch"
                or record.get("validation_used_to_select") is not False
                or bool(record.get("smoke")) != smoke):
            raise ValueError("Nakazawa completion record/hash/selection mismatch")
        checkpoint = nakazawa._load_checkpoint(checkpoint_path, bundle, config)
        if checkpoint.get("epoch") != params["epochs"]:
            raise ValueError("Nakazawa evaluation requires the configured final epoch")
        del checkpoint
        if not smoke:
            last_epoch = record.get("final_epoch", {})
            if (last_epoch.get("train_samples") != len(bundle.train_ids)
                    or last_epoch.get("valid_samples") != len(bundle.valid_ids)):
                raise ValueError("Nakazawa formal training did not cover the complete train/valid sets")
        checkpoint_signature = source_record["base_signature"]
        provenance.update({"selection": "last_epoch", "training_complete": True})
        provenance["files"] = {path.name: hash_file(path) for path in (marker, checkpoint_path)}
    return checkpoint_signature, provenance


def _fresh_output(path, config, source_run=None):
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else common.ROOT / path).resolve()
    # Protect both repository inputs and any existing method run.
    protected = [common.ROOT / name for name in
                 ("methods", "shared", "scripts", "tests", "protocol", "docs", ".git", ".venv")]
    protected.extend(Path(config[key]).resolve() for key in ("data_file", "split_file", "retrieval_protocol_dir"))
    protected.append(Path(config["output_dir"]).resolve())
    if source_run is not None:
        protected.append(Path(source_run).resolve())
    if path == common.ROOT or path in common.ROOT.parents:
        raise ValueError("Evaluation output cannot contain the repository")
    if any(path == item or path in item.parents or item in path.parents for item in protected):
        raise ValueError("Evaluation output overlaps source data/code/protocol or an existing method run")
    if path.exists():
        raise FileExistsError(f"Evaluation output already exists; choose a fresh directory: {path}")
    return path


def _check_request(config, split_name, ks, variant, workers, tie_seed):
    method = config.get("method")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    if split_name not in ("valid", "test"):
        raise ValueError("split_name must be valid or test")
    if variant not in ("global", "predicted_class") or (method != "yu" and variant != "global"):
        raise ValueError("predicted_class is an auxiliary Yu-only variant")
    if type(workers) is not int or workers < 1 or type(tie_seed) is not int or tie_seed < 0:
        raise ValueError("workers must be positive and tie_seed nonnegative integers")
    subset = any(value is not None for value in config.get("subset", {}).values())
    if split_name == "test" and (config.get("smoke", False) or subset):
        raise ValueError("Test requires a full formal configuration; smoke/subset runs cannot evaluate test")
    if subset and not config.get("smoke", False):
        raise ValueError("Subset configurations must be explicitly marked smoke")
    ks = list(SMOKE_KS if config.get("smoke", False) else DEFAULT_KS) if ks is None else list(ks)
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must contain distinct positive integers")
    return sorted(ks)


def run_evaluation(config, output_dir, *, split_name="valid", ks=None, source_run=None,
                   variant="global", device=None, workers=1, tie_seed=2026, rank_workers=1):
    """Evaluate one method using the frozen pool; never train or select on test."""
    config = dict(config)
    original_run = config["output_dir"]
    ks = _check_request(config, split_name, ks, variant, workers, tie_seed)
    if type(rank_workers) is not int or rank_workers < 1:
        raise ValueError("rank_workers must be a positive integer")
    if source_run is not None:
        if config["method"] not in TRAINED_METHODS:
            raise ValueError("source_run is only needed for trained methods Yu/Nakazawa")
        source = Path(source_run).expanduser()
        config["output_dir"] = str((source if source.is_absolute() else common.ROOT / source).resolve())
    if device is not None:
        config["device"] = device
    output = _fresh_output(output_dir, {**config, "output_dir": original_run},
                           source_run=config["output_dir"])
    source_record = verify_source_run(config) if config["method"] in TRAINED_METHODS else None
    # Create exclusively, including for empty existing folders; never overwrite.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        data_config = {**config, "output_dir": str(output), "cache_dir": str(output / "raw_cache")}
        bundle = common.prepare_data(data_config)
        pool = common.make_retrieval_pool(bundle, data_config, split_name)
        if max(ks) > pool.candidates_per_query:
            raise ValueError("K exceeds the candidate count; select explicit smaller --ks for synthetic/debug data")
        checkpoint_signature, source_provenance = None, None
        if source_record is not None:
            checkpoint_signature, source_provenance = validate_training_source(bundle, config, source_record, variant)
        files = [Path(__file__), Path(__file__).with_name("exact_metrics.py"),
                 Path(__file__).with_name("coverage_metrics.py"),
                 Path(__file__).with_name("evaluation_adapters.py"),
                 common.ROOT / "scripts" / "evaluate.py", Path(common.__file__),
                 common.PROTOCOL_SOURCE, common.RETRIEVAL_SOURCE, _method_path(config["method"])]
        payload = {"runner_version": RUNNER_VERSION, "metric_version": METRIC_VERSION,
            "coverage_metric_version": coverage_metrics.METRIC_VERSION,
            "method": config["method"], "variant": variant, "split": split_name,
            "ks": ks, "tie_seed": tie_seed, "tie_policy": TIE_POLICY,
            "settings": _settings(config), "data_signature": bundle.signature,
            "sources": _source_hashes(config), "training_source": source_provenance,
            "code": {str(p.relative_to(common.ROOT)): hash_file(p) for p in files},
            "smoke": bool(config.get("smoke", False)), "query_count": len(pool),
            "gallery_count": len(pool.gallery), "candidate_rule": "all_gallery_except_query",
            "execution": {"feature_workers": workers, "rank_workers": rank_workers,
                          "rank_start_method": "spawn" if rank_workers > 1 else "serial",
                          "max_pending_queries": 2 * rank_workers if rank_workers > 1 else 1},
            "candidates_per_query": pool.candidates_per_query,
            "relevance": "exact equality of all eight defect labels",
            "ranking_limit": max(ks), "full_mAP_reported": False,
            "coverage_reports": True}
        eval_signature = signature(payload)
        save_json(output / "evaluation_manifest.json", {**payload, "signature": eval_signature})
        save_json(output / "resolved_config.json", config)
        save_json(output / "environment.json", common.environment_info())
        destination = output / split_name
        destination.mkdir()
        if config["method"] in TRAINED_METHODS:
            import torch
            common.seed_everything(int(config["seed"]))
            threads = int(config.get("torch_threads", 0))
            if threads > 0:
                torch.set_num_threads(threads)
            save_json(destination / "inference_environment.json", {
                "seed": int(config["seed"]), "torch_threads": torch.get_num_threads(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "settings_origin": "shared.common.seed_everything; original method convention",
            })
        preparation_start = time.perf_counter()
        ranker = prepare_method(bundle, config, pool, destination / "prepared",
                                source_run_signature=checkpoint_signature, device=config.get("device", "auto"),
                                workers=workers, variant=variant, tie_seed=tie_seed)
        preparation_seconds = time.perf_counter() - preparation_start
        save_json(destination / "method.json", ranker.metadata)
        ranking_path = destination / "rankings.csv"
        counts_path = destination / "query_diagnostics.csv"
        rank_seconds = []
        ranking_started = time.perf_counter()
        max_k = max(ks)
        with ranking_path.with_suffix(".csv.tmp").open("x", newline="", encoding="utf-8") as ranks_file, \
                counts_path.with_suffix(".csv.tmp").open("x", newline="", encoding="utf-8") as counts_file:
            writer = csv.DictWriter(ranks_file, fieldnames=["query_id", "rank", "candidate_id", "score", "coarse_distance"])
            diagnostic = csv.DictWriter(counts_file, fieldnames=["query_id", "input_candidates", "scored_count",
                "coarse_scored_count", "eligible_count", "returned", "ranking_seconds"])
            writer.writeheader()
            diagnostic.writeheader()

            def rankings():
                with closing(iter_ranked_queries(ranker, pool.items(), rank_workers)) as stream:
                    for number, (qid, candidates, ranked, elapsed) in enumerate(stream, 1):
                        rank_seconds.append(elapsed)
                        ids, scores = np.asarray(ranked.ids), np.asarray(ranked.scores)
                        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or scores.shape != ids.shape
                                or len(np.unique(ids)) != len(ids) or not np.isin(ids, candidates).all()
                                or np.isnan(scores).any() or np.isposinf(scores).any()):
                            raise ValueError("Method returned invalid IDs/scores")
                        distances = ranked.coarse_distances
                        if distances is not None and np.asarray(distances).shape != ids.shape:
                            raise ValueError("Method returned invalid coarse distances")
                        for position in range(min(max_k, len(ids))):
                            writer.writerow({"query_id": int(qid), "rank": position + 1,
                                "candidate_id": int(ids[position]), "score": float(scores[position]),
                                "coarse_distance": "" if distances is None else float(distances[position])})
                        diagnostic.writerow({"query_id": int(qid), "input_candidates": len(candidates),
                            "scored_count": ranked.scored_count,
                            "coarse_scored_count": getattr(ranked, "coarse_scored_count", 0),
                            "eligible_count": ranked.eligible_count, "returned": min(max_k, len(ids)),
                            "ranking_seconds": elapsed})
                        yield int(qid), ids[:max_k]
                        if number % 100 == 0 or number == len(pool):
                            print(f"{config['method']} {split_name}: {number}/{len(pool)} queries", flush=True)
            result = evaluate_rankings(bundle, pool, rankings(), ks, destination, split_name=split_name,
                metadata={"evaluation_signature": eval_signature, "method": config["method"],
                          "variant": variant, "tie_seed": tie_seed, "tie_policy": TIE_POLICY,
                          "smoke": bool(config.get("smoke", False)), "ranking_limit": max_k})
        ranking_path.with_suffix(".csv.tmp").replace(ranking_path)
        counts_path.with_suffix(".csv.tmp").replace(counts_path)
        coverage_metrics.evaluate_rankings_file(
            bundle, pool, ranking_path, ks, destination, split_name=split_name,
            metadata={"method": config["method"],
                      "variant": variant, "tie_seed": tie_seed, "tie_policy": TIE_POLICY,
                      "smoke": bool(config.get("smoke", False)), "ranking_limit": max_k,
                      "exact_metric_version": METRIC_VERSION})
        # Reject a concurrent source change rather than publishing mixed artifacts.
        if _source_hashes(config) != payload["sources"] or any(
                hash_file(common.ROOT / name) != digest for name, digest in payload["code"].items()):
            raise ValueError("Source data/protocol/code changed during evaluation")
        if source_provenance is not None:
            if hash_file(Path(config["output_dir"]) / "run_manifest.json") != source_provenance["manifest_sha256"]:
                raise ValueError("Training source changed during evaluation")
            for name, digest in source_provenance["files"].items():
                if hash_file(Path(config["output_dir"]) / name) != digest:
                    raise ValueError("Training checkpoint changed during evaluation")
        save_json(destination / "timing.json", {"preparation_seconds": preparation_seconds,
            "ranking_total_seconds": float(sum(rank_seconds)),
            "ranking_wall_seconds": time.perf_counter() - ranking_started,
            "ranking_mean_seconds": float(np.mean(rank_seconds)),
            "ranking_median_seconds": float(np.median(rank_seconds)),
            "ranking_p95_seconds": float(np.percentile(rank_seconds, 95)),
            "ranking_scope": "sum of per-query worker times; excludes metric calculation and CSV writes",
            "ranking_wall_scope": "ranking stage elapsed time including worker startup, metrics, CSV writes and output checks",
            "total_seconds": time.perf_counter() - started, "workers": workers, "rank_workers": rank_workers,
            "device": config.get("device", "cpu")})
        output_files = {str(path.relative_to(output)): hash_file(path)
                        for path in destination.rglob("*") if path.is_file()}
        save_json(output / "completed.json", {"evaluation_signature": eval_signature,
            "split": split_name, "num_queries": len(pool), "test_evaluated": split_name == "test",
            "smoke": bool(config.get("smoke", False)), "files": output_files})
        return result
    except Exception as error:
        save_json(output / "failed.json", {"error_type": type(error).__name__, "error": str(error),
            "complete": False, "note": "Partial artifacts are not benchmark results; use a fresh output directory"})
        raise

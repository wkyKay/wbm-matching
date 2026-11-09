#!/usr/bin/env python3
"""Wang & Wang (2023) WBBS: closed-form tensor voting and weighted BB pairs.

This module supplies descriptors and scores only. Retrieval evaluation is left
for the shared experiment requested by the user. See README for parameters not
specified in the paper; no learned model or labels enter the score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAMETERS = {
    "sigma_d": 9.0,
    "noise_threshold": 0.32,
    "alpha": 5.0,
    "mu": 1.0,
    "total_dies": 2061,
    "block_size": 256,
}


def parameters(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    unknown = set(overrides or {}) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown WBBS parameters: {sorted(unknown)}")
    result = {**DEFAULT_PARAMETERS, **(overrides or {})}
    for name in ("sigma_d", "noise_threshold", "alpha", "mu"):
        if not np.isscalar(result[name]) or not np.isfinite(result[name]):
            raise ValueError(f"{name} must be finite")
    if result["sigma_d"] <= 0:
        raise ValueError("sigma_d must be positive")
    if not 0 <= result["noise_threshold"] <= 1:
        raise ValueError("noise_threshold must lie in [0, 1]")
    if result["alpha"] < 0 or result["mu"] < 0:
        raise ValueError("alpha and mu must be nonnegative")
    for name in ("total_dies", "block_size"):
        if isinstance(result[name], bool) or not isinstance(result[name], (int, np.integer)) or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    return result


def _points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite (N, 2) array")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Voting sites must be unique")
    return points


def tensor_vote(points: np.ndarray, sigma_d: float = 9.0, block_size: int = 256) -> np.ndarray:
    """Exact Eq. (2), (6)-(8), with each voter initialized to I and no self-vote.

    For K_j=I, R(I-0.5 rr^T)R^T = I-0.5 rr^T, so this is the closed-form
    matrix expression, not an approximation by a density feature. All other
    defective points vote once; there is no radius cutoff or point sampling.
    Units in exp(-||xi-xj||^2/sigma_d) are raw pixel coordinates.
    """
    points = _points(points)
    if not np.isfinite(sigma_d) or sigma_d <= 0:
        raise ValueError("sigma_d must be finite and positive")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    tensors = np.zeros((len(points), 2, 2), dtype=np.float64)
    for start in range(0, len(points), block_size):
        delta = points[start:start + block_size, None, :] - points[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        nonzero = distance2 > 0
        decay = np.exp(-distance2 / sigma_d) * nonzero
        # c * (I - .5 rr^T); zero-distance terms are excluded completely.
        scale = np.divide(0.5 * decay, distance2, out=np.zeros_like(decay), where=nonzero)
        tensors[start:start + len(delta)] = -np.einsum("ij,ija,ijb->iab", scale, delta, delta)
        total = decay.sum(axis=1)
        tensors[start:start + len(delta), 0, 0] += total
        tensors[start:start + len(delta), 1, 1] += total
    return tensors


def tensor_saliencies(tensors: np.ndarray) -> np.ndarray:
    """Eq. (9)-(10): columns are lambda_1-lambda_2 and lambda_2."""
    tensors = np.asarray(tensors, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[1:] != (2, 2) or not np.isfinite(tensors).all():
        raise ValueError("Expected finite (N, 2, 2) tensors")
    if not np.allclose(tensors, tensors.swapaxes(1, 2), atol=1e-12, rtol=1e-12):
        raise ValueError("Tensor voting output must be symmetric")
    eigenvalues = np.linalg.eigvalsh(tensors)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Tensor voting output must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0)
    return np.column_stack((eigenvalues[:, 1] - eigenvalues[:, 0], eigenvalues[:, 0]))


def retained_points(saliencies: np.ndarray, threshold: float = 0.32) -> np.ndarray:
    """Section 2.1: remove only when BOTH stick and ball are strictly below t*max."""
    saliencies = np.asarray(saliencies, dtype=np.float64)
    if saliencies.ndim != 2 or saliencies.shape[1] != 2 or not np.isfinite(saliencies).all() or np.any(saliencies < 0):
        raise ValueError("Expected nonnegative finite (N, 2) saliencies")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if len(saliencies) == 0:
        return np.empty(0, dtype=bool)
    return ~np.all(saliencies < threshold * saliencies.max(axis=0), axis=1)


@dataclass(frozen=True)
class WaferFeatures:
    """Retained sites in raster order and [normalized stick, ball, radius]."""
    points: np.ndarray
    descriptors: np.ndarray
    original_count: int

    def __post_init__(self):
        points = _points(self.points)
        descriptors = np.asarray(self.descriptors, dtype=np.float64)
        if descriptors.shape != (len(points), 3) or not np.isfinite(descriptors).all():
            raise ValueError("Descriptors must have shape (retained points, 3) and be finite")
        if np.any(descriptors < 0) or np.any(descriptors > 1 + 1e-12):
            raise ValueError("Descriptors must be normalized to [0, 1]")
        if not isinstance(self.original_count, (int, np.integer)) or self.original_count < len(points):
            raise ValueError("original_count must cover all retained sites")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "descriptors", descriptors)


def encode_wafer(wafer: np.ndarray, settings: dict[str, Any] | None = None) -> WaferFeatures:
    """Decode MixedWM38, vote, threshold, and build the Eq. (15)-(16) descriptors.

    Code 2 is defective, 1/2 define wafer geometry, and 0/3 are ignored.
    Per-image saliency maxima BEFORE denoising normalize the two channels;
    rho is divided by the greatest support radius about its bounding-box center.
    These normalization details are explicit implementation conventions.
    """
    p = parameters(settings)
    wafer = np.asarray(wafer)
    if wafer.ndim != 2 or min(wafer.shape) == 0 or not np.isin(wafer, [0, 1, 2, 3]).all():
        raise ValueError("Expected a nonempty 2D wafer with values 0, 1, 2, 3")
    points = np.argwhere(wafer == 2).astype(np.float64)
    if not len(points):
        return WaferFeatures(points, np.empty((0, 3)), 0)
    saliencies = tensor_saliencies(tensor_vote(points, float(p["sigma_d"]), int(p["block_size"])))
    keep = retained_points(saliencies, float(p["noise_threshold"]))
    maxima = saliencies.max(axis=0)
    normalized = np.divide(saliencies, maxima, out=np.zeros_like(saliencies), where=maxima > 0)
    support = np.argwhere(np.isin(wafer, [1, 2])).astype(np.float64)
    center = (support.min(axis=0) + support.max(axis=0)) / 2
    radius = float(np.linalg.norm(support - center, axis=1).max())
    rho = np.linalg.norm(points - center, axis=1) / radius if radius > 0 else np.zeros(len(points))
    descriptors = np.column_stack((normalized, rho))
    return WaferFeatures(points[keep], descriptors[keep], len(points))


def best_buddies(query_descriptors: np.ndarray, candidate_descriptors: np.ndarray,
                 mu: float = 1.0, block_size: int = 256) -> np.ndarray:
    """Eq. (11), (15): return index pairs that are mutual nearest neighbors.

    Distances are exact squared Euclidean in (stick, ball, sqrt(mu)*rho).
    Exact ties select the lowest index in the other set. Raster-order features
    make this deterministic without injecting a different similarity metric.
    """
    query = np.asarray(query_descriptors, dtype=np.float64)
    candidate = np.asarray(candidate_descriptors, dtype=np.float64)
    if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() for a in (query, candidate)):
        raise ValueError("Descriptors must be finite (N, 3) arrays")
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and nonnegative")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if not len(query) or not len(candidate):
        return np.empty((0, 2), dtype=np.int64)
    scale = np.array([1.0, 1.0, np.sqrt(mu)])
    q, c = query * scale, candidate * scale
    q_to_c = np.empty(len(q), dtype=np.int64)
    c_to_q = np.zeros(len(c), dtype=np.int64)
    c_distance = np.full(len(c), np.inf)
    for start in range(0, len(q), block_size):
        distance = cdist(q[start:start + block_size], c, metric="sqeuclidean")
        q_to_c[start:start + len(distance)] = np.argmin(distance, axis=1)
        local = np.argmin(distance, axis=0)
        minimum = distance[local, np.arange(len(c))]
        # Strict comparison preserves the first query row across block ties.
        better = minimum < c_distance
        c_distance[better] = minimum[better]
        c_to_q[better] = start + local[better]
    rows = np.flatnonzero(c_to_q[q_to_c] == np.arange(len(q)))
    return np.column_stack((rows, q_to_c[rows]))


def score_pair(query: WaferFeatures, candidate: WaferFeatures,
               settings: dict[str, Any] | None = None) -> float:
    """Original symmetric Eq. (12)-(14): BBPs/min(N,M) * exp(-alpha*|N-M|/l).

    N and M count retained defective sites. An empty point set yields 0 by an
    explicit extension of the otherwise undefined paper formula.
    """
    p = parameters(settings)
    n, m = len(query.descriptors), len(candidate.descriptors)
    if min(n, m) == 0:
        return 0.0
    pairs = best_buddies(query.descriptors, candidate.descriptors, float(p["mu"]), int(p["block_size"]))
    return float(len(pairs) / min(n, m) * np.exp(-float(p["alpha"]) * abs(n - m) / int(p["total_dies"])))


def score_candidates(query: WaferFeatures, candidates: Iterable[WaferFeatures],
                     settings: dict[str, Any] | None = None) -> np.ndarray:
    """One score per supplied candidate, in the same order; higher is better."""
    p = parameters(settings)
    return np.asarray([score_pair(query, candidate, p) for candidate in candidates], dtype=np.float64)


def rank_candidates(query: WaferFeatures, candidates: Iterable[WaferFeatures],
                    candidate_ids: np.ndarray, settings: dict[str, Any] | None = None,
                    *, query_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rank all supplied candidates; ties use original IDs, optional self exclusion."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("Candidate IDs must be a unique vector")
    scores = score_candidates(query, candidates, settings)
    if scores.shape != ids.shape:
        raise ValueError("Candidate features and IDs must have the same length")
    keep = ids != query_id if query_id is not None else np.ones(len(ids), dtype=bool)
    ids, scores = ids[keep], scores[keep]
    order = np.lexsort((ids, -scores))
    return ids[order], scores[order]


@dataclass
class FeatureStore:
    ids: np.ndarray
    offsets: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    original_counts: np.ndarray
    settings: dict[str, Any]

    def get(self, sample_id: int) -> WaferFeatures:
        row = int(np.searchsorted(self.ids, sample_id))
        if row == len(self.ids) or self.ids[row] != sample_id:
            raise KeyError(f"No prepared WBBS features for sample_id={sample_id}")
        start, end = self.offsets[row:row + 2]
        return WaferFeatures(self.points[start:end], self.descriptors[start:end], int(self.original_counts[row]))


def load_prepared(output: str | Path) -> FeatureStore:
    """Load and validate a prepared cache; sample_id always means the raw NPZ row."""
    from shared.common import hash_file, read_json
    output = Path(output)
    meta = read_json(output / "prepare.json")
    cache = output / "features.npz"
    if hash_file(cache) != meta["features_sha256"]:
        raise ValueError("WBBS feature cache hash mismatch")
    with np.load(cache, allow_pickle=False) as data:
        store = FeatureStore(**{name: data[name].copy() for name in
                                ("ids", "offsets", "points", "descriptors", "original_counts")},
                             settings=parameters(meta["parameters"]))
    if (store.ids.ndim != 1 or not np.array_equal(store.ids, np.unique(store.ids)) or
            store.offsets.shape != (len(store.ids) + 1,) or store.offsets[0] != 0 or
            np.any(np.diff(store.offsets) < 0) or store.offsets[-1] != len(store.descriptors) or
            len(store.points) != len(store.descriptors) or store.original_counts.shape != store.ids.shape):
        raise ValueError("Invalid WBBS cache indexing")
    return store


def rank_from_pool(store: FeatureStore, pool: Any, query_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Score the exact candidates supplied by a frozen FullGalleryPool; no metrics."""
    candidate_ids = np.asarray(pool[query_id], dtype=np.int64)
    if query_id in candidate_ids:
        raise ValueError("Frozen candidates must exclude the query itself")
    return rank_candidates(store.get(query_id), (store.get(int(i)) for i in candidate_ids),
                           candidate_ids, store.settings, query_id=query_id)


def _encode_task(task: tuple[int, np.ndarray, dict[str, Any]]) -> tuple[int, WaferFeatures]:
    sample_id, wafer, settings = task
    return sample_id, encode_wafer(wafer, settings)


def prepare(bundle: Any, config: dict[str, Any], output: Path, workers: int = 1) -> dict[str, Any]:
    """Prepare the complete valid gallery (or the marked smoke subset); no test IO."""
    from shared.common import file_lock, hash_file, read_json, save_json, signature
    p = parameters(config.get("parameters"))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    ids = np.unique(np.asarray(bundle.valid_ids, dtype=np.int64))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache_signature = signature({"bundle": bundle.signature, "parameters": p, "split": "valid", "ids": ids.tolist()})
    manifest, cache = output / "prepare.json", output / "features.npz"
    with file_lock(output / ".prepare.lock"):
        if manifest.exists():
            previous = read_json(manifest)
            if previous.get("cache_signature") != cache_signature:
                raise ValueError("WBBS cache signature mismatch; choose a fresh output directory")
            load_prepared(output)
            return previous
        if cache.exists():
            raise ValueError("Unmanifested WBBS cache exists; choose a fresh output directory")
        started = time.perf_counter()
        tasks = ((int(i), np.asarray(bundle.maps[i]), p) for i in ids)
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        results = executor.map(_encode_task, tasks, chunksize=8) if executor else map(_encode_task, tasks)
        offsets, points, descriptors, original_counts = [0], [], [], []
        try:
            for row, (sample_id, feature) in enumerate(results):
                if sample_id != int(ids[row]):
                    raise RuntimeError("WBBS feature order changed")
                offsets.append(offsets[-1] + len(feature.points))
                points.append(feature.points)
                descriptors.append(feature.descriptors)
                original_counts.append(feature.original_count)
                if (row + 1) % 100 == 0:
                    print(f"WBBS prepare: {row + 1}/{len(ids)}", flush=True)
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        arrays = {"ids": ids, "offsets": np.asarray(offsets, dtype=np.int64),
                  "points": np.concatenate(points) if points else np.empty((0, 2)),
                  "descriptors": np.concatenate(descriptors) if descriptors else np.empty((0, 3)),
                  "original_counts": np.asarray(original_counts, dtype=np.int64)}
        temporary = output / "features.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(cache)
        record = {"method": "wbbs", "cache_signature": cache_signature,
                  "data_signature": bundle.signature, "parameters": p, "rows": len(ids),
                  "split": "valid", "smoke": bool(config.get("smoke", False)),
                  "feature_columns": ["stick", "ball", "rho"], "features_sha256": hash_file(cache),
                  "retained_points": int(offsets[-1]), "original_points": int(sum(original_counts)),
                  "empty_features": int(np.count_nonzero(np.diff(offsets) == 0)),
                  "training_required": False, "test_accessed": False, "metrics_computed": False,
                  "workers": workers, "elapsed_seconds": time.perf_counter() - started}
        save_json(manifest, record)
        return record


def main(argv: list[str] | None = None) -> int:
    from shared.common import ensure_run, load_config, make_retrieval_pool, prepare_data, save_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--stage", choices=["audit", "prepare"], default="audit")
    parser.add_argument("--output-dir", help="Override output directory, relative to compared_methods")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = load_config(args.config, overrides={"output_dir": args.output_dir}, require_ks=False)
    if config.get("method") != "wbbs":
        raise ValueError("Expected method=wbbs")
    p = parameters(config.get("parameters"))
    ensure_run(config, [Path(__file__)])
    bundle = prepare_data(config)
    pool = make_retrieval_pool(bundle, config, split_name="valid")
    output = Path(config["output_dir"])
    audit = {"method": "wbbs", "training_required": False, "feature_split": "valid",
             "rows": len(bundle.valid_ids), "queries": len(pool), "data_signature": bundle.signature,
             "parameters": p, "test_accessed": False, "metrics_computed": False,
             "paper_parameters": ["noise_threshold", "alpha", "mu"],
             "implementation_conventions": ["sigma_d", "total_dies", "single_identity_vote_pass",
                 "exclude_self_vote", "saliency_max_normalization_before_filter", "support_bbox_radial_normalization",
                 "lowest_point_index_on_nearest_neighbor_tie", "empty_set_score_zero"]}
    save_json(output / "audit.json", audit)
    result = {"audit": audit}
    if args.stage == "prepare":
        result["prepare"] = prepare(bundle, config, output, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

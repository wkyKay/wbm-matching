#!/usr/bin/env python3
"""Hsu 2020 WMHD equations 1–6; feature preparation and scoring, no evaluation.

The paper does not specify its rotation alignment algorithm or numerical
preprocessing parameters. This implementation uses fixed Cartesian alignment;
see README.md for the explicit reproduction limits and parameter provenance.
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
from scipy import ndimage
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAMETERS: dict[str, Any] = {
    "esrn_iterations": 1,
    "esrn_good_threshold": 0.625,
    "esrn_defect_threshold": 0.625,
    "mountain_m": 0.1,
    "matching_tolerance": 0.1,
    "weight_design": "W2",
    "outlier_penalty": True,
    "point_chunk_size": 256,
    "numerical_epsilon": 1e-12,
}


def parameters(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    supplied = overrides or {}
    unknown = set(supplied) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown WMHD parameters: {sorted(unknown)}")
    p = {**DEFAULT_PARAMETERS, **supplied}
    for name in ("esrn_iterations", "point_chunk_size"):
        if isinstance(p[name], bool) or not isinstance(p[name], int):
            raise ValueError(f"{name} must be an integer")
    if p["esrn_iterations"] < 0 or p["point_chunk_size"] < 1:
        raise ValueError("ESRN iterations must be nonnegative and chunks positive")
    for name in ("esrn_good_threshold", "esrn_defect_threshold"):
        if not 0.5 < float(p[name]) <= 1:
            raise ValueError(f"{name} must lie in (0.5, 1]")
    for name in ("mountain_m", "matching_tolerance", "numerical_epsilon"):
        if not np.isfinite(p[name]) or float(p[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if p["weight_design"] not in ("W0", "W1", "W2"):
        raise ValueError("weight_design must be W0, W1 or W2")
    if not isinstance(p["outlier_penalty"], bool):
        raise ValueError("outlier_penalty must be boolean")
    return p


def wafer_masks(wafer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MixedWM38 adapter: code 2 is defective, codes 1/2 are real dies."""
    wafer = np.asarray(wafer)
    if wafer.ndim != 2 or min(wafer.shape) == 0:
        raise ValueError("Expected a nonempty 2D wafer map")
    if not np.isin(wafer, [0, 1, 2, 3]).all():
        raise ValueError("MixedWM38 pixel values must be 0/1/2/3")
    return wafer == 2, np.isin(wafer, [1, 2])


def esrn(defects: np.ndarray, support: np.ndarray,
         settings: dict[str, Any] | None = None) -> np.ndarray:
    """Section 3.2's King-move state updates with documented fixed thresholds.

    Eight adjacent real dies have equal weight; the center and background are
    excluded from the denominator. Updates in each pass use the previous mask.
    Exact weights, thresholds, iteration count and boundary handling are not
    provided in Hsu 2020, so these are explicit implementation conventions.
    """
    p = parameters(settings)
    support = np.asarray(support, dtype=bool)
    mask = np.asarray(defects, dtype=bool)
    if mask.ndim != 2 or mask.shape != support.shape or min(mask.shape) == 0:
        raise ValueError("Defect and support masks must be equal nonempty 2D arrays")
    mask = mask.copy() & support
    kernel = np.ones((3, 3), dtype=np.int64)
    kernel[1, 1] = 0
    denominator = ndimage.convolve(support.astype(np.int64), kernel, mode="constant", cval=0)
    for _ in range(p["esrn_iterations"]):
        bad = ndimage.convolve(mask.astype(np.int64), kernel, mode="constant", cval=0)
        good = denominator - bad
        connected = support & (denominator > 0)
        enhance = connected & ~mask & (bad >= float(p["esrn_defect_threshold"]) * denominator)
        remove = connected & mask & (good >= float(p["esrn_good_threshold"]) * denominator)
        mask = (mask | enhance) & ~remove
    return mask


def wafer_coordinates(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Real-die row/column coordinates, centered and scaled to maximum radius 1.

    The wafer support, not the defect extent or another split, determines the
    center and scale. Normalization is a MixedWM38 implementation convention.
    """
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or min(support.shape) == 0:
        raise ValueError("Expected a nonempty 2D support mask")
    pixels = np.argwhere(support)
    if not len(pixels):
        return np.empty((0, 2), dtype=np.float64), (np.asarray(support.shape) - 1) / 2, 1.0
    center = (pixels.min(axis=0) + pixels.max(axis=0)) / 2.0
    centered = pixels.astype(np.float64) - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    radius = radius if radius > 0 else 1.0
    return centered / radius, center, radius


def mountain_field(defects: np.ndarray, support: np.ndarray,
                   settings: dict[str, Any] | None = None) -> tuple[np.ndarray, float]:
    """Equation (1), beta = N_all_dies / sum_defect distance_to_center.

    Distances for the density kernel are Euclidean (an explicitly documented
    convention); city-block distance is used for the WMHD matching itself.
    No normalization by the number of defects or by maximum mountain height.
    """
    p = parameters(settings)
    support = np.asarray(support, dtype=bool)
    defects = np.asarray(defects, dtype=bool)
    if defects.shape != support.shape or defects.ndim != 2:
        raise ValueError("Defect and support masks must have equal 2D shape")
    coordinates, center, radius = wafer_coordinates(support)
    points = (np.argwhere(defects & support) - center) / radius
    field = np.zeros(support.shape, dtype=np.float64)
    if not len(points):
        return field, 0.0
    radial_sum = float(np.linalg.norm(points, axis=1).sum())
    beta = len(coordinates) / max(radial_sum, float(p["numerical_epsilon"]))
    values = np.empty(len(coordinates), dtype=np.float64)
    chunk = p["point_chunk_size"]
    for start in range(0, len(coordinates), chunk):
        distances = cdist(coordinates[start:start + chunk], points, metric="euclidean")
        values[start:start + chunk] = np.exp(-float(p["mountain_m"]) * beta * distances).sum(axis=1)
    field[support] = values
    return field, float(beta)


@dataclass(frozen=True)
class WMHDFeatures:
    """One wafer's retained points and center-inclusive 3x3 mountain patches."""
    points: np.ndarray
    neighborhoods: np.ndarray
    radius: float
    beta: float
    raw_defects: int
    support_dies: int

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        neighborhoods = np.asarray(self.neighborhoods, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or neighborhoods.shape != (len(points), 9):
            raise ValueError("Expected points (N,2) and mountain neighborhoods (N,9)")
        if not np.isfinite(points).all() or not np.isfinite(neighborhoods).all() or (neighborhoods < 0).any():
            raise ValueError("Feature coordinates and nonnegative mountain values must be finite")
        if not np.isfinite(self.radius) or self.radius <= 0 or not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Feature radius must be positive and beta nonnegative")
        if self.raw_defects < 0 or self.support_dies < 0:
            raise ValueError("Die counts must be nonnegative")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "neighborhoods", neighborhoods)


def extract_features(wafer: np.ndarray,
                     settings: dict[str, Any] | None = None) -> WMHDFeatures:
    p = parameters(settings)
    defects, support = wafer_masks(wafer)
    cleaned = esrn(defects, support, p)
    _, center, radius = wafer_coordinates(support)
    pixels = np.argwhere(cleaned)
    field, beta = mountain_field(cleaned, support, p)
    padded = np.pad(field, 1)
    patches = np.empty((len(pixels), 9), dtype=np.float64)
    for k, (dy, dx) in enumerate((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
        patches[:, k] = padded[pixels[:, 0] + 1 + dy, pixels[:, 1] + 1 + dx]
    return WMHDFeatures((pixels - center) / radius, patches, radius, beta,
                        int(defects.sum()), int(support.sum()))


def weight_matrix(query: np.ndarray, candidate: np.ndarray,
                  design: str = "W2") -> np.ndarray:
    """Equations (3–5): no weight, center ratio, or SUM of nine ratios."""
    query, candidate = np.asarray(query, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if query.ndim != 2 or candidate.ndim != 2 or query.shape[1] != 9 or candidate.shape[1] != 9:
        raise ValueError("Expected two (N,9) mountain patch arrays")
    if design not in ("W0", "W1", "W2"):
        raise ValueError("weight design must be W0, W1 or W2")
    if not np.isfinite(query).all() or not np.isfinite(candidate).all() or (query < 0).any() or (candidate < 0).any():
        raise ValueError("Mountain values must be finite and nonnegative")
    shape = (len(query), len(candidate))
    if design == "W0":
        return np.ones(shape, dtype=np.float64)
    result = np.zeros(shape, dtype=np.float64)
    for k in ([4] if design == "W1" else range(9)):
        a, b = query[:, k, None], candidate[None, :, k]
        denominator = np.maximum(a, b)
        # Both zero is an agreeing empty neighborhood, with contribution zero.
        result += np.divide(np.abs(a - b), denominator, out=np.zeros(shape), where=denominator > 0)
    return result


def distance(query: WMHDFeatures, candidate: WMHDFeatures,
             settings: dict[str, Any] | None = None) -> float:
    """Directional W2-O1 distance, smaller means more similar.

    O1 limits possible point matches to L1 distance < Sout. Unmatched query
    points contribute No*Sout; the matching sum still divides by original NA.
    O0 uses every pair. No symmetric max/mean is added to the paper's formula.
    """
    p = parameters(settings)
    n_query, n_candidate = len(query.points), len(candidate.points)
    if n_query == 0:
        return 0.0 if n_candidate == 0 else float("inf")
    tolerance = float(p["matching_tolerance"])
    if n_candidate == 0:
        return float(n_query * tolerance) if p["outlier_penalty"] else float("inf")
    summed, outliers = 0.0, 0
    chunk = p["point_chunk_size"]
    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        spatial = cdist(query.points[start:stop], candidate.points, metric="cityblock")
        if p["outlier_penalty"]:
            allowed = spatial < tolerance
            matched = allowed.any(axis=1)
            # O1 excludes all other point pairs from the minimum. Compute the
            # identical W0/W1/W2 terms only for allowed pairs, retaining the
            # original component accumulation and matched-row summation order.
            rows, columns = np.nonzero(allowed)
            weights = np.ones(len(rows)) if p["weight_design"] == "W0" else np.zeros(len(rows))
            for k in ([4] if p["weight_design"] == "W1" else
                      range(9) if p["weight_design"] == "W2" else ()):
                a = query.neighborhoods[start:stop, k][rows]
                b = candidate.neighborhoods[columns, k]
                denominator = np.maximum(a, b)
                weights += np.divide(np.abs(a - b), denominator,
                                     out=np.zeros(len(rows)), where=denominator > 0)
            minima = np.full(stop - start, np.inf)
            np.minimum.at(minima, rows, spatial[rows, columns] * weights)
            summed += float(minima[matched].sum())
            outliers += int((~matched).sum())
        else:
            weighted = spatial * weight_matrix(query.neighborhoods[start:stop],
                                              candidate.neighborhoods, p["weight_design"])
            summed += float(weighted.min(axis=1).sum())
    return summed / n_query + (outliers * tolerance if p["outlier_penalty"] else 0.0)


def score_candidates(query: WMHDFeatures, candidates: Iterable[WMHDFeatures],
                     settings: dict[str, Any] | None = None) -> np.ndarray:
    """Negative distances in input order; consumes all supplied candidates."""
    p = parameters(settings)
    return np.asarray([-distance(query, candidate, p) for candidate in candidates], dtype=np.float64)


def rank_candidates(query: WMHDFeatures, candidates: Iterable[WMHDFeatures],
                    candidate_ids: np.ndarray, settings: dict[str, Any] | None = None,
                    *, query_id: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Rank all supplied candidates, descending score, then original ID on ties.

    The caller supplies the frozen pool's candidates (already excludes self).
    Supplying query_id also enforces self-exclusion when using a full gallery.
    This API does not inspect class labels or calculate evaluation metrics.
    """
    ids = np.asarray(candidate_ids)
    if ids.ndim == 1 and ids.size == 0:
        ids = ids.astype(np.int64)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("Candidate IDs must be a unique nonnegative one-dimensional integer array")
    if query_id is not None and (isinstance(query_id, bool) or not isinstance(query_id, (int, np.integer))):
        raise ValueError("query_id must be an integer")
    pairs = zip(ids, candidates, strict=True)
    selected = (candidate for sample_id, candidate in pairs if query_id is None or sample_id != query_id)
    scores = score_candidates(query, selected, settings)
    if query_id is not None:
        ids = ids[ids != query_id]
    order = np.lexsort((ids, -scores))
    return ids[order], scores[order]


def _encode_task(task):
    sample_id, wafer, p = task
    return sample_id, extract_features(wafer, p)


def load_features(output: str | Path) -> dict[int, WMHDFeatures]:
    """Read and hash-check a prepared valid feature cache, keyed by original ID."""
    from shared.common import hash_file, read_json
    output = Path(output)
    record = read_json(output / "prepare.json")
    path = output / "features.npz"
    if record.get("features_sha256") != hash_file(path):
        raise ValueError("Feature cache file hash mismatch")
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        ids, offsets = archive["sample_ids"], archive["offsets"]
        points, neighborhoods, scalars = archive["points"], archive["neighborhoods"], archive["scalars"]
        if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or offsets.shape != (len(ids) + 1,)
                or offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(points)
                or neighborhoods.shape != (len(points), 9) or scalars.shape != (len(ids), 4)):
            raise ValueError("Invalid packed feature cache")
        for row, sample_id in enumerate(ids):
            sl = slice(int(offsets[row]), int(offsets[row + 1]))
            radius, beta, raw, support = scalars[row]
            result[int(sample_id)] = WMHDFeatures(points[sl], neighborhoods[sl], float(radius),
                                                 float(beta), int(raw), int(support))
    return result


def prepare(bundle: Any, config: dict[str, Any], workers: int = 1) -> dict[str, Any]:
    """Prepare the frozen valid gallery only; no train/test features or metrics."""
    from shared.common import file_lock
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(config["output_dir"])
    with file_lock(output / ".features.lock"):
        return _prepare_locked(bundle, config, workers)


def _prepare_locked(bundle: Any, config: dict[str, Any], workers: int) -> dict[str, Any]:
    """Call only while holding .features.lock through the complete cache write."""
    from shared.common import hash_file, make_retrieval_pool, read_json, save_json, signature
    if workers < 1:
        raise ValueError("workers must be positive")
    p = parameters(config.get("parameters"))
    pool = make_retrieval_pool(bundle, config, split_name="valid")
    ids = np.asarray(pool.gallery, dtype=np.int64)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache_signature = signature({"bundle": bundle.signature, "parameters": p,
                                 "sample_ids": ids, "split": "valid", "alignment": "fixed_cartesian"})
    manifest, cache = output / "prepare.json", output / "features.npz"
    if manifest.exists():
        record = read_json(manifest)
        if record.get("cache_signature") != cache_signature:
            raise ValueError("Feature cache signature mismatch; choose a fresh output directory")
        load_features(output)
        return record
    if cache.exists():
        raise FileExistsError("Existing cache has no completion manifest; choose a new output directory")
    started = time.perf_counter()
    tasks = ((int(i), np.asarray(bundle.maps[i]), p) for i in ids)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = executor.map(_encode_task, tasks, chunksize=8) if executor else map(_encode_task, tasks)
    features = []
    try:
        for row, (sample_id, feature) in enumerate(results):
            if sample_id != int(ids[row]):
                raise RuntimeError("Unexpected feature extraction order")
            features.append(feature)
            if (row + 1) % 100 == 0:
                print(f"WMHD prepare: {row + 1}/{len(ids)}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    lengths = [len(f.points) for f in features]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    points = np.concatenate([f.points for f in features]) if features else np.empty((0, 2))
    neighborhoods = np.concatenate([f.neighborhoods for f in features]) if features else np.empty((0, 9))
    scalars = np.asarray([[f.radius, f.beta, f.raw_defects, f.support_dies] for f in features], dtype=np.float64).reshape(-1, 4)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, sample_ids=ids, offsets=offsets, points=points,
                            neighborhoods=neighborhoods, scalars=scalars)
    temporary.replace(cache)
    record = {"method": "wmhd", "cache_signature": cache_signature, "split": "valid",
              "rows": len(ids), "queries": len(pool), "empty_after_esrn": int(sum(n == 0 for n in lengths)),
              "parameters": p, "coordinate_alignment": "fixed_cartesian",
              "reproduction_limit": "Paper's Cartesian-Polar rotation alignment is unspecified and not reproduced",
              "features_sha256": hash_file(cache), "data_signature": bundle.signature,
              "protocol_signature": pool.protocol_signature, "test_accessed": False,
              "training_required": False, "metrics_computed": False,
              "elapsed_seconds": time.perf_counter() - started}
    save_json(manifest, record)
    return record


def main(argv: list[str] | None = None) -> int:
    from shared.common import ensure_run, load_config, make_retrieval_pool, prepare_data, save_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--stage", choices=["audit", "prepare"], default="audit")
    parser.add_argument("--output-dir", help="Fresh output path, relative to compared_methods")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = load_config(args.config, overrides={"output_dir": args.output_dir}, require_ks=False)
    if config.get("method") != "wmhd":
        raise ValueError("Expected method=wmhd")
    p = parameters(config.get("parameters"))
    ensure_run(config, [Path(__file__)])
    bundle = prepare_data(config)
    pool = make_retrieval_pool(bundle, config, split_name="valid")
    record = {"method": "wmhd", "training_required": False, "parameters": p,
              "valid_queries": len(pool), "valid_gallery": len(pool.gallery),
              "data_signature": bundle.signature, "protocol_signature": pool.protocol_signature,
              "coordinate_alignment": "fixed_cartesian", "metrics_computed": False,
              "test_accessed": False,
              "reproduction_limit": "Cartesian-Polar rotation alignment unspecified by paper; not reproduced"}
    save_json(Path(config["output_dir"]) / "audit.json", record)
    if args.stage == "prepare":
        record = prepare(bundle, config, workers=args.workers)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

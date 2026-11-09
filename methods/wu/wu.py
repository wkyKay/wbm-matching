#!/usr/bin/env python3
"""Paper-based implementation of Wu et al. (2015) WMSR; explicit valid/test evaluation.

The public functions operate on arrays and have no dependency on the project
retriever.  Implementation conventions missing from the paper are configurable.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.measure import label, perimeter, regionprops
from skimage.transform import hough_line, hough_line_peaks, radon, resize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAMETERS: dict[str, Any] = {
    "coarse_candidates": 500,
    "beta": 0.5,
    "median_size": 3,
    "connectivity": 2,
    "perimeter_neighborhood": 4,
    "radon_angles": 180,
    "radon_bins": 20,
    "radon_circle": False,
    "std_ddof": 1,
    "hough_angles": 180,
    "hough_peak_relative_threshold": 0.3,
    "hough_max_peaks": 20,
    "hough_min_distance": 2,
    "hough_min_angle": 5,
    "hough_band_half_width": 0.5,
    "hough_min_length_fraction": 0.10,
    "hough_max_gap_fraction": 0.03,
    "rerank_shape": [52, 52],
    "numerical_epsilon": 1e-12,
}


def parameters(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge and validate the documented implementation parameters."""
    result = {**DEFAULT_PARAMETERS, **(overrides or {})}
    if int(result["coarse_candidates"]) < 1:
        raise ValueError("coarse_candidates must be positive")
    if not 0.0 <= float(result["beta"]) <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    if result["radon_bins"] != 20 or result["radon_angles"] != 180:
        raise ValueError("This reproduction fixes 20 Radon bins and 180 angles")
    if result["radon_circle"] is not False:
        raise ValueError("radon_circle must be false to preserve boundary defects")
    if int(result["median_size"]) < 1 or int(result["median_size"]) % 2 != 1:
        raise ValueError("median_size must be a positive odd number")
    if int(result["connectivity"]) not in (1, 2):
        raise ValueError("2D connectivity must be 1 or 2")
    if int(result["perimeter_neighborhood"]) not in (4, 8):
        raise ValueError("perimeter_neighborhood must be 4 or 8")
    if len(result["rerank_shape"]) != 2 or min(result["rerank_shape"]) < 1:
        raise ValueError("rerank_shape must contain two positive sizes")
    if int(result["hough_angles"]) < 2 or int(result["hough_max_peaks"]) < 1:
        raise ValueError("Hough angles and peak cap must be positive")
    if not 0.0 < float(result["hough_peak_relative_threshold"]) <= 1.0:
        raise ValueError("Hough relative threshold must lie in (0, 1]")
    if float(result["numerical_epsilon"]) <= 0:
        raise ValueError("numerical_epsilon must be positive")
    return result


def wafer_masks(wafer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode MixedWM38: 1=normal, 2=defect; 0/3 are ignored background."""
    array = np.asarray(wafer)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError("Expected a nonempty 2D wafer map")
    if not np.isin(array, [0, 1, 2, 3]).all():
        raise ValueError("Wafer map values must be 0, 1, 2 or 3")
    return array == 2, np.isin(array, [1, 2])


def wafer_geometry(support: np.ndarray) -> tuple[np.ndarray, float]:
    """BBox center and half its shorter diameter; empty support is well-defined."""
    coords = np.argwhere(support)
    if coords.size:
        lower, upper = coords.min(axis=0), coords.max(axis=0)
        return (lower + upper) / 2.0, float(np.min(upper - lower + 1)) / 2.0
    return (np.asarray(support.shape, dtype=float) - 1) / 2.0, max(0.5, min(support.shape) / 2.0)


def _regions(mask: np.ndarray, settings: dict[str, Any]) -> list[Any]:
    return regionprops(label(mask, connectivity=int(settings["connectivity"])))


def largest_region(mask: np.ndarray, settings: dict[str, Any] | None = None) -> np.ndarray:
    """Largest-area connected component in its original full-canvas location."""
    p = parameters(settings)
    labels = label(np.asarray(mask, dtype=bool), connectivity=int(p["connectivity"]))
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros_like(labels, dtype=bool)
    counts[0] = 0
    # Labels follow raster scan order; argmax picks the first on an area tie.
    return labels == int(np.argmax(counts))


def radon_features(mask: np.ndarray, settings: dict[str, Any] | None = None) -> np.ndarray:
    """40 features: normalized sinogram -> rho-wise mean/std -> cubic 20+20."""
    p = parameters(settings)
    image = np.asarray(mask, dtype=np.float64)
    if not image.any():
        return np.zeros(40, dtype=np.float64)
    theta = np.linspace(0.0, 180.0, int(p["radon_angles"]), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False, preserve_range=True)
    span = float(np.ptp(sinogram))
    if span <= float(p["numerical_epsilon"]):
        return np.zeros(40, dtype=np.float64)
    sinogram = (sinogram - sinogram.min()) / span
    profiles = (sinogram.mean(axis=1), sinogram.std(axis=1, ddof=int(p["std_ddof"])))
    source = np.linspace(0.0, 1.0, sinogram.shape[0])
    target = np.linspace(0.0, 1.0, 20)
    kind = "cubic" if len(source) >= 4 else "linear"
    values = [interp1d(source, row, kind=kind, assume_sorted=True)(target) for row in profiles]
    return np.concatenate(values).astype(np.float64)


def regional_features(mask: np.ndarray, support: np.ndarray,
                      settings: dict[str, Any] | None = None) -> np.ndarray:
    """Table III's 14 features, including both Sa and Sp where specified."""
    p = parameters(settings)
    regions = _regions(np.asarray(mask, dtype=bool), p)
    if not regions:
        return np.zeros(14, dtype=np.float64)
    center, radius = wafer_geometry(support)
    perimeters = [float(perimeter(r.image, neighborhood=int(p["perimeter_neighborhood"]))) for r in regions]
    area_index = max(range(len(regions)), key=lambda i: (regions[i].area, -regions[i].label))
    perimeter_index = max(range(len(regions)), key=lambda i: (perimeters[i], -regions[i].label))
    sa, sp = regions[area_index], regions[perimeter_index]
    values = [float(sa.area) / (np.pi * radius**2), perimeters[perimeter_index] / radius]
    pairs = []
    for region in (sa, sp):
        distances = np.linalg.norm(region.coords.astype(float) - center, axis=1) / radius
        pairs.append((float(distances.max()), float(distances.min()),
                      float(region.axis_major_length) / radius,
                      float(region.axis_minor_length) / radius,
                      float(region.solidity), float(region.eccentricity)))
    # Order: max-distance Sa/Sp, min-distance Sa/Sp, major Sa/Sp, ... .
    values.extend(value for attribute in zip(*pairs) for value in attribute)
    return np.asarray(values, dtype=np.float64)


def statistical_features(mask: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Global defect ratio and separate outermost/second-outermost ring ratios."""
    support = np.asarray(support, dtype=bool)
    defects = np.asarray(mask, dtype=bool) & support
    first_inner = ndimage.binary_erosion(support, structure=np.ones((3, 3)), border_value=0)
    second_inner = ndimage.binary_erosion(first_inner, structure=np.ones((3, 3)), border_value=0)
    zones = (support, support & ~first_inner, first_inner & ~second_inner)
    return np.asarray([float(np.count_nonzero(defects & zone)) / np.count_nonzero(zone)
                       if zone.any() else 0.0 for zone in zones], dtype=np.float64)


def hough_line_count(mask: np.ndarray, support: np.ndarray,
                     settings: dict[str, Any] | None = None) -> int:
    """Deterministic peak/segment counting with paper's 10%/3% length rules.

    A peak owns defective pixels within a configurable half-width of its line.
    Their projected intervals have unit pixel width; empty gaps must be strictly
    below 3% of the diameter. Distinct surviving intervals are counted as lines.
    Peak extraction parameters are conventions, not specified by Wu et al.
    """
    p = parameters(settings)
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0
    _, radius = wafer_geometry(support)
    diameter = 2.0 * radius
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, int(p["hough_angles"]), endpoint=False)
    accumulator, angles, distances = hough_line(np.asarray(mask, dtype=bool), theta=theta)
    peaks, peak_angles, peak_distances = hough_line_peaks(
        accumulator, angles, distances,
        min_distance=int(p["hough_min_distance"]), min_angle=int(p["hough_min_angle"]),
        threshold=float(accumulator.max()) * float(p["hough_peak_relative_threshold"]),
        num_peaks=int(p["hough_max_peaks"]),
    )
    min_length = float(p["hough_min_length_fraction"]) * diameter
    max_gap = float(p["hough_max_gap_fraction"]) * diameter
    y, x = coords[:, 0].astype(float), coords[:, 1].astype(float)
    count = 0
    for _, angle, distance in sorted(zip(peaks, peak_angles, peak_distances),
                                    key=lambda item: (-int(item[0]), float(item[1]), float(item[2]))):
        on_line = np.abs(x * np.cos(angle) + y * np.sin(angle) - distance) <= float(p["hough_band_half_width"])
        projected = np.sort(-x[on_line] * np.sin(angle) + y[on_line] * np.cos(angle))
        if len(projected) < 2:
            continue
        # Adjacent projected pixel intervals have no positive empty gap.
        gaps = np.maximum(np.diff(projected) - 1.0, 0.0)
        splits = np.flatnonzero((gaps > 0.0) & (gaps >= max_gap)) + 1
        for segment in np.split(projected, splits):
            if len(segment) >= 2 and float(segment[-1] - segment[0]) > min_length:
                count += 1
    return count


def geometry_features(mask: np.ndarray, support: np.ndarray,
                      settings: dict[str, Any] | None = None) -> np.ndarray:
    """14 regional + 3 statistical + 1 Hough = 18 geometry features."""
    return np.concatenate((regional_features(mask, support, settings),
                           statistical_features(mask, support),
                           np.asarray([hough_line_count(mask, support, settings)], dtype=float)))


def extract_features(wafer: np.ndarray, settings: dict[str, Any] | None = None) -> np.ndarray:
    """Return exactly 116 features; no learned or additional feature scaling."""
    p = parameters(settings)
    defects, support = wafer_masks(wafer)
    denoised = ndimage.median_filter(defects.astype(np.uint8), size=int(p["median_size"]),
                                     mode="constant", cval=0).astype(bool) & support
    features = np.concatenate([part for mask in (defects, denoised)
                               for part in (radon_features(mask, p), geometry_features(mask, support, p))])
    if features.shape != (116,) or not np.isfinite(features).all():
        raise FloatingPointError("Feature extraction must produce 116 finite values")
    return features.astype(np.float32)


def ncc_vector(mask: np.ndarray, shape: Iterable[int] | None = None,
               epsilon: float = 1e-12) -> np.ndarray:
    """Centered unit vector; zero variance maps to zero, not NaN or perfect match."""
    image = np.asarray(mask, dtype=np.float64)
    if shape is not None and image.shape != tuple(shape):
        image = resize(image, tuple(shape), order=0, mode="constant", cval=0,
                       anti_aliasing=False, preserve_range=True)
    vector = image.ravel() - float(image.mean())
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        return np.zeros(vector.shape, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def ncc_score(query: np.ndarray, candidate: np.ndarray, epsilon: float = 1e-12) -> float:
    """Equation (24) for aligned arrays; no sliding or transform maximization."""
    if np.asarray(query).shape != np.asarray(candidate).shape:
        raise ValueError("NCC requires equal, already aligned array shapes")
    return float(np.clip(np.dot(ncc_vector(query, epsilon=epsilon),
                                ncc_vector(candidate, epsilon=epsilon)), -1.0, 1.0))


def encode_wafer(wafer: np.ndarray, settings: dict[str, Any] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = parameters(settings)
    defects, _ = wafer_masks(wafer)
    salient = largest_region(defects, p)
    return (extract_features(wafer, p),
            ncc_vector(defects, p["rerank_shape"], float(p["numerical_epsilon"])),
            ncc_vector(salient, p["rerank_shape"], float(p["numerical_epsilon"])))


def rank_candidates(query_feature: np.ndarray, candidate_features: np.ndarray,
                    candidate_ids: np.ndarray, query_full: np.ndarray, candidate_full: np.ndarray,
                    query_region: np.ndarray, candidate_regions: np.ndarray,
                    settings: dict[str, Any] | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reranked IDs, NCC scores, and Euclidean distances for top-n only."""
    p = parameters(settings)
    ids = np.asarray(candidate_ids, dtype=np.int64)
    if len(ids) == 0:
        return ids, np.empty(0), np.empty(0)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique")
    features = np.asarray(candidate_features, dtype=np.float64)
    distances = np.linalg.norm(features - np.asarray(query_feature, dtype=np.float64), axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Coarse distances contain non-finite values")
    coarse = np.lexsort((ids, distances))[:min(int(p["coarse_candidates"]), len(ids))]
    beta = float(p["beta"])
    scores = (beta * (np.asarray(candidate_full)[coarse] @ query_full)
              + (1.0 - beta) * (np.asarray(candidate_regions)[coarse] @ query_region))
    scores = np.clip(scores, -1.0, 1.0)
    if not np.isfinite(scores).all():
        raise ValueError("NCC scores contain non-finite values")
    order = np.lexsort((ids[coarse], distances[coarse], -scores))
    selected = coarse[order]
    return ids[selected], scores[order], distances[selected]


def _encode_task(task: tuple[int, np.ndarray, dict[str, Any]]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    sample_id, wafer, settings = task
    return (sample_id, *encode_wafer(wafer, settings))


def selected_ids(bundle: Any, split_name: str = "valid") -> np.ndarray:
    """WMSR needs no fit: cache only rows belonging to the requested split."""
    if split_name not in ("valid", "test"):
        raise ValueError("Retrieval split must be valid or test")
    ids = bundle.valid_ids if split_name == "valid" else bundle.reserved_ids
    return np.unique(np.asarray(ids, dtype=np.int64))


def audit(bundle: Any, config: dict[str, Any], output: Path,
          split_name: str = "valid") -> dict[str, Any]:
    from shared.common import save_json
    p = parameters(config.get("parameters"))
    ids = selected_ids(bundle, split_name)
    record = {"method": "wu", "feature_dimension": 116, "training_required": False,
              "allowed_splits": ["valid", "test (explicit --stage test only)"], "feature_splits": [split_name],
              "normal_features_required": False, "data_signature": bundle.signature,
              "selected_rows": len(ids),
              "counts": {key: len(getattr(bundle, key)) for key in
                         ("train_ids", "valid_ids", "normal_train_ids", "normal_valid_ids")},
              "parameters": p, "test_requested": split_name == "test", "test_accessed": False}
    save_json(output / "audit.json", record)
    return record


def prepare(bundle: Any, config: dict[str, Any], output: Path, workers: int,
            split_name: str = "valid") -> dict[str, Any]:
    from shared.common import read_json, save_json, signature
    p = parameters(config.get("parameters"))
    ids = selected_ids(bundle, split_name)
    output.mkdir(parents=True, exist_ok=True)
    cache_signature = signature({"bundle": bundle.signature, "parameters": p,
                                 "split": split_name, "ids": ids.tolist()})
    manifest = output / "prepare.json"
    filenames = ["feature_ids.npy", "features.npy", "ncc_full.npy", "ncc_region.npy"]
    if manifest.exists():
        previous = read_json(manifest)
        if previous.get("cache_signature") != cache_signature:
            raise ValueError("Feature cache signature mismatch; use a fresh output directory")
        if all((output / name).exists() for name in filenames):
            return previous
    started = time.perf_counter()
    np.save(output / "feature_ids.npy", ids)
    size = int(np.prod(p["rerank_shape"]))
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32, shape=(len(ids), 116))
    full = np.lib.format.open_memmap(output / "ncc_full.npy", mode="w+", dtype=np.float32, shape=(len(ids), size))
    region = np.lib.format.open_memmap(output / "ncc_region.npy", mode="w+", dtype=np.float32, shape=(len(ids), size))
    tasks = ((int(sample_id), np.asarray(bundle.maps[sample_id]), p) for sample_id in ids)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results = executor.map(_encode_task, tasks, chunksize=16) if executor else map(_encode_task, tasks)
    empty_full = empty_region = 0
    try:
        for row, (sample_id, feat, whole, salient) in enumerate(results):
            if sample_id != int(ids[row]):
                raise RuntimeError("Feature extraction order changed unexpectedly")
            features[row], full[row], region[row] = feat, whole, salient
            empty_full += int(not np.any(whole))
            empty_region += int(not np.any(salient))
            if (row + 1) % 1000 == 0:
                print(f"Wu prepare: {row + 1}/{len(ids)}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    features.flush()
    full.flush()
    region.flush()
    record = {"cache_signature": cache_signature, "data_signature": bundle.signature,
              "rows": len(ids), "dimension": 116, "zero_variance_full": empty_full,
              "zero_variance_region": empty_region, "workers": workers,
              "elapsed_seconds": time.perf_counter() - started,
              "files": filenames, "parameters": p, "split": split_name,
              "test_accessed": split_name == "test"}
    save_json(manifest, record)
    return record


def validate(bundle: Any, config: dict[str, Any], output: Path,
             split_name: str = "valid") -> dict[str, Any]:
    from shared.common import evaluate_rankings, make_retrieval_pool, read_json, save_json, signature
    p = parameters(config.get("parameters"))
    manifest = output / "prepare.json"
    if not manifest.exists():
        raise FileNotFoundError("Run --stage prepare before --stage valid")
    ids = np.load(output / "feature_ids.npy", allow_pickle=False)
    selected = selected_ids(bundle, split_name)
    expected = signature({"bundle": bundle.signature, "parameters": p,
                          "split": split_name, "ids": selected.tolist()})
    if read_json(manifest).get("cache_signature") != expected or not np.array_equal(ids, selected):
        raise ValueError("Prepared cache does not match this data/configuration")
    features = np.load(output / "features.npy", mmap_mode="r", allow_pickle=False)
    full = np.load(output / "ncc_full.npy", mmap_mode="r", allow_pickle=False)
    region = np.load(output / "ncc_region.npy", mmap_mode="r", allow_pickle=False)
    row_for_id = {int(sample_id): row for row, sample_id in enumerate(ids)}
    pool = make_retrieval_pool(bundle, config, split_name=split_name)
    maximum = max(config["ks"])
    csv_path, json_path = output / "rankings.csv", output / "rankings.json"
    csv_temporary, json_temporary = csv_path.with_suffix(".csv.tmp"), json_path.with_suffix(".json.tmp")
    started = time.perf_counter()
    with csv_temporary.open("w", newline="", encoding="utf-8") as csv_handle, \
            json_temporary.open("w", encoding="utf-8") as json_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=["query_id", "rank", "candidate_id", "score", "coarse_distance"])
        writer.writeheader()
        json_handle.write("{\n")

        def rankings():
            for number, (query_id, raw_candidates) in enumerate(pool.items()):
                query_id = int(query_id)
                candidates = np.asarray(raw_candidates, dtype=np.int64)
                if query_id in candidates:
                    raise ValueError("A query cannot be its own candidate")
                query_row = row_for_id[query_id]
                candidate_rows = np.asarray([row_for_id[int(value)] for value in candidates], dtype=np.int64)
                ranked, scores, distances = rank_candidates(
                    features[query_row], features[candidate_rows], candidates,
                    full[query_row], full[candidate_rows], region[query_row], region[candidate_rows], p)
                ranked, scores, distances = ranked[:maximum], scores[:maximum], distances[:maximum]
                writer.writerows({"query_id": query_id, "rank": rank, "candidate_id": int(candidate),
                                  "score": float(score), "coarse_distance": float(distance)}
                                 for rank, (candidate, score, distance) in enumerate(zip(ranked, scores, distances), start=1))
                if number:
                    json_handle.write(",\n")
                json_handle.write(f"  {json.dumps(str(query_id))}: {json.dumps(ranked.tolist())}")
                yield query_id, ranked

        metrics = evaluate_rankings(bundle, pool, rankings(), config["ks"], output, split_name=split_name)
        json_handle.write("\n}\n")
    csv_temporary.replace(csv_path)
    json_temporary.replace(json_path)
    elapsed = time.perf_counter() - started
    record = {"method": "wu", "split": split_name, "queries": len(pool), "coarse_candidates": p["coarse_candidates"],
              "beta": p["beta"], "ranking_seconds": elapsed, "metrics": metrics,
              "ranking_storage": "top max(ks), streamed; metric denominators use the full gallery",
              "stored_ranking_limit": maximum,
              "training_required": False, "test_accessed": split_name == "test"}
    save_json(output / f"{split_name}_summary.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    from shared.common import ensure_run, load_config, make_retrieval_pool, prepare_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--stage", choices=["audit", "prepare", "valid", "test", "all"], default="audit")
    parser.add_argument("--output-dir", help="Override output directory, relative to compared_methods")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    overrides = {"output_dir": args.output_dir} if args.output_dir else None
    config = load_config(args.config, overrides=overrides)
    if config.get("method") != "wu":
        raise ValueError("Expected method=wu")
    parameters(config.get("parameters"))
    ensure_run(config, [Path(__file__)])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    bundle = prepare_data(config)
    if args.stage == "test" and config.get("smoke", False):
        raise ValueError("Test requires the full frozen protocol; smoke configurations cannot run test")
    selected_split = "test" if args.stage == "test" else "valid"
    audit_output = output / "test" if args.stage == "test" else output
    result: dict[str, Any] = {"audit": audit(bundle, config, audit_output, split_name=selected_split)}
    if args.stage in ("prepare", "all"):
        result["prepare"] = prepare(bundle, config, output, args.workers)
    if args.stage in ("valid", "all"):
        result["valid"] = validate(bundle, config, output)
    if args.stage == "test":
        # Validate the frozen full-test protocol before any test feature work.
        make_retrieval_pool(bundle, config, split_name="test")
        result["prepare_test"] = prepare(bundle, config, output / "test", args.workers, split_name="test")
        result["test"] = validate(bundle, config, output / "test", split_name="test")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

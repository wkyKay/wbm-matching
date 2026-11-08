"""Shared data/provenance utilities; algorithms live in wu/ and yu/.

The frozen test rows are used only to check split integrity. No test feature,
prediction or evaluation stage exists here. All image-processing callers must
use the explicit train/valid ID arrays in DataBundle.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
_HASH_CACHE = {}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def hash_file(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        result = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                result.update(chunk)
        _HASH_CACHE[key] = result.hexdigest()
    return _HASH_CACHE[key]


def signature(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         default=_json_default, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path, overrides=None):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    config = read_json(path)
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    config.setdefault("seed", 2026)
    config.setdefault("expected_classes", 37)
    config.setdefault("normal_split", {"train": 700, "valid": 100, "holdout": 200})
    config.setdefault("subset", {})
    for key in ("data_file", "split_file", "output_dir", "cache_dir"):
        if key not in config:
            continue
        value = Path(config[key]).expanduser()
        config[key] = str(value.resolve() if value.is_absolute() else (ROOT / value).resolve())
    if config.get("method") not in ("wu", "yu"):
        raise ValueError("method must be wu or yu")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not config.get("ks") or any(not isinstance(k, int) or k < 1 for k in config["ks"]):
        raise ValueError("ks must contain positive integers")
    return config


@contextmanager
def file_lock(path):
    """Advisory file lock supported by the target macOS/Linux environments."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def environment_info():
    packages = {}
    for package in ("numpy", "scipy", "scikit-image", "scikit-learn", "torch", "Pillow"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def ensure_run(config, method_files):
    """Reject mixing different scientific settings, inputs or source code."""
    output = Path(config["output_dir"]).resolve()
    files = [Path(__file__).resolve()] + [Path(p).resolve() for p in method_files]
    protected = [ROOT, Path(config["data_file"]), Path(config["split_file"])] + files
    if any(output == p or output in p.parents for p in protected):
        raise ValueError("Output directory would contain code or source inputs")
    ignored = {"output_dir", "device", "workers", "num_workers", "cpu_threads", "torch_threads",
               "data_file", "split_file", "cache_dir", "resume"}
    settings = {k: v for k, v in config.items() if k not in ignored}
    sources = {key: hash_file(config[key]) for key in ("data_file", "split_file")}
    code = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: hash_file(p)
            for p in files}
    payload = {"format_version": 1, "settings": settings, "sources": sources, "code": code}
    run_signature = signature(payload)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "run_manifest.json"
    with file_lock(output / ".manifest.lock"):
        if marker.exists():
            if read_json(marker).get("signature") != run_signature:
                raise ValueError("Code/config/data changed: choose a new output_dir; old run is preserved")
        elif any(p.name != ".manifest.lock" for p in output.iterdir()):
            raise FileExistsError("Refusing a nonempty output directory without a matching run manifest")
        else:
            save_json(marker, {**payload, "signature": run_signature})
            save_json(output / "environment.json", environment_info())
        save_json(output / "resolved_config.json", config)
    return run_signature


def _raw_arrays(config):
    """Cache source arrays once as uint8; no feature fitting occurs here."""
    digest = hash_file(config["data_file"])
    cache = Path(config.get("cache_dir", Path(config["output_dir"]).parent / "_data")) / digest[:20]
    with file_lock(cache / ".lock"):
        marker = cache / "ready.json"
        if marker.exists():
            record = read_json(marker)
            if record["source_sha256"] != digest:
                raise ValueError("Raw-cache source mismatch")
            for name in ("maps.npy", "labels.npy"):
                if not (cache / name).exists() or hash_file(cache / name) != record["files"][name]:
                    raise ValueError("Raw-cache file missing or corrupted; choose/remove only this generated cache")
        else:
            with np.load(config["data_file"], allow_pickle=False) as archive:
                labels = archive["arr_1"]
                maps = archive["arr_0"]
                if maps.ndim != 3 or labels.shape != (len(maps), 8):
                    raise ValueError("Expected arr_0=(N,H,W), arr_1=(N,8)")
                if not np.isin(labels, [0, 1]).all():
                    raise ValueError("Expected eight binary base-defect labels")
                temporary = cache / "maps.tmp.npy"
                destination = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=maps.shape)
                for start in range(0, len(maps), 1024):
                    chunk = maps[start:start + 1024]
                    if not np.isin(chunk, [0, 1, 2, 3]).all():
                        raise ValueError("Unsupported WBM pixel code; expected 0/1/2/3")
                    destination[start:start + len(chunk)] = chunk
                destination.flush()
                del destination
                temporary.replace(cache / "maps.npy")
                np.save(cache / "labels.npy", labels.astype(np.uint8))
                shape = list(maps.shape)
                del maps
            save_json(marker, {"source_sha256": digest, "shape": shape,
                               "files": {name: hash_file(cache / name) for name in ("maps.npy", "labels.npy")}})
    return np.load(cache / "maps.npy", mmap_mode="r"), np.load(cache / "labels.npy", mmap_mode="r")


@dataclass
class DataBundle:
    maps: np.ndarray
    labels: np.ndarray
    class_ids: np.ndarray
    train_ids: np.ndarray
    valid_ids: np.ndarray
    normal_train_ids: np.ndarray
    normal_valid_ids: np.ndarray
    reserved_ids: np.ndarray
    normal_reserved_ids: np.ndarray
    class_signatures: list
    signature: str
    full_counts: dict

    def summary(self):
        return {"source_shape": list(self.maps.shape), "num_defect_classes": len(self.class_signatures),
                "used_defect_train": len(self.train_ids), "used_defect_valid": len(self.valid_ids),
                "used_normal_train": len(self.normal_train_ids), "used_normal_valid": len(self.normal_valid_ids),
                "reserved_defect_count": len(self.reserved_ids),
                "reserved_normal_count": len(self.normal_reserved_ids), "full_counts": self.full_counts,
                "test_processed": False, "signature": self.signature,
                "pixel_policy": "defect=2; valid die=1 or 2; code 3 ignored (same as source project)"}


def _subset(ids, count, seed, class_ids=None):
    ids = np.asarray(sorted(ids), dtype=np.int64)
    if count is None:
        return ids
    if not isinstance(count, int) or count < 1:
        raise ValueError("Subset counts must be positive integers or null")
    rng = np.random.default_rng(seed)
    groups = [ids] if class_ids is None else [ids[class_ids[ids] == cls] for cls in sorted(set(class_ids[ids]))]
    selected = []
    for group in groups:
        if count > len(group):
            raise ValueError(f"Requested {count} samples from a group containing {len(group)}")
        selected.extend(rng.choice(group, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def prepare_data(config):
    maps, labels = _raw_arrays(config)
    with Path(config["split_file"]).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = {key: [] for key in ("train", "valid", "test")}
    seen = set()
    nonempty = np.flatnonzero(np.asarray(labels).sum(axis=1) > 0)
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id in seen or row["split"] not in ids or not 0 <= sample_id < len(labels):
            raise ValueError("Duplicate/out-of-range sample ID or unknown split")
        if sample_id != int(row["original_index"]):
            raise ValueError("sample_id must be the original NPZ row index")
        expected = np.array([int(row[f"label_{i}"]) for i in range(8)], dtype=np.uint8)
        if not expected.any() or not np.array_equal(expected, labels[sample_id]):
            raise ValueError(f"Source/manifest label mismatch at {sample_id}")
        if "valid_index" in row and nonempty[int(row["valid_index"])] != sample_id:
            raise ValueError("Frozen valid_index mismatch")
        seen.add(sample_id)
        ids[row["split"]].append(sample_id)
    if seen != set(nonempty.tolist()) or any(not values for values in ids.values()):
        raise ValueError("Frozen split must partition all nonempty-label wafers")
    signatures = sorted({tuple(int(v) for v in labels[i]) for i in ids["train"]})
    if len(signatures) != config.get("expected_classes", 37):
        raise ValueError("Unexpected train combination-class count")
    mapping = {value: index for index, value in enumerate(signatures)}
    class_ids = np.full(len(labels), -1, dtype=np.int16)
    for i in nonempty:
        key = tuple(int(v) for v in labels[i])
        if key not in mapping:
            raise ValueError("A non-train combination is absent from the train taxonomy")
        class_ids[i] = mapping[key]
    normal_ids = np.flatnonzero(np.asarray(labels).sum(axis=1) == 0)
    normal_settings = config.get("normal_split", {"train": 700, "valid": 100, "holdout": 200})
    if set(normal_settings) != {"train", "valid", "holdout"} or any(
            not isinstance(v, int) or v < 0 for v in normal_settings.values()):
        raise ValueError("normal_split must contain nonnegative train/valid/holdout counts")
    if sum(normal_settings.values()) != len(normal_ids):
        raise ValueError("Normal split counts must partition every empty-label source sample")
    shuffled = np.random.default_rng(config["seed"]).permutation(normal_ids)
    nt, nv = normal_settings["train"], normal_settings["valid"]
    normal = {"train": np.sort(shuffled[:nt]), "valid": np.sort(shuffled[nt:nt + nv]),
              "holdout": np.sort(shuffled[nt + nv:])}
    subset = config.get("subset", {})
    chosen = {split: _subset(ids[split], subset.get(f"{split}_per_class"),
                            config["seed"] + offset, class_ids)
              for offset, split in enumerate(("train", "valid"))}
    chosen_normal = {split: _subset(normal[split], subset.get(f"normal_{split}"), config["seed"] + 2 + offset)
                     for offset, split in enumerate(("train", "valid"))}
    specification = {"source": hash_file(config["data_file"]), "split": hash_file(config["split_file"]),
                     "seed": config["seed"], "subset": subset, "normal_split": normal_settings,
                     "class_signatures": signatures, "train_ids": chosen["train"], "valid_ids": chosen["valid"],
                     "normal_train_ids": chosen_normal["train"], "normal_valid_ids": chosen_normal["valid"]}
    bundle = DataBundle(maps, labels, class_ids, chosen["train"], chosen["valid"],
                        chosen_normal["train"], chosen_normal["valid"],
                        np.asarray(sorted(ids["test"]), dtype=np.int64), normal["holdout"],
                        [list(s) for s in signatures], signature(specification),
                        {**{s: len(v) for s, v in ids.items()}, "normal": len(normal_ids)})
    output = Path(config["output_dir"]) / "protocol"
    save_json(output / "data_summary.json", bundle.summary())
    save_json(output / "class_mapping.json", {"derived_from": "train only", "normal_id": -1,
                                               "classes": bundle.class_signatures})
    save_csv(output / "normal_split.csv", ({"sample_id": int(i), "split": split}
               for split in ("train", "valid", "holdout") for i in normal[split]), ["sample_id", "split"])
    for name in ("train_ids", "valid_ids", "normal_train_ids", "normal_valid_ids"):
        save_csv(output / f"{name}.csv", ({"sample_id": int(i)} for i in getattr(bundle, name)), ["sample_id"])
    return bundle


def make_valid_pool(bundle, config):
    settings = config["valid_pool"]
    ids = np.asarray(sorted(bundle.valid_ids), dtype=np.int64)
    nq, nc = int(settings["queries"]), int(settings["candidates"])
    if not 0 < nq <= len(ids) or not 0 < nc < len(ids):
        raise ValueError("Requested valid pool exceeds available IDs; set an explicit smaller smoke pool")
    rng = np.random.default_rng(settings.get("seed", config["seed"]))
    queries = sorted(rng.choice(ids, nq, replace=False).tolist())
    pool = {q: sorted(rng.choice(ids[ids != q], nc, replace=False).tolist()) for q in queries}
    path = Path(config["output_dir"]) / "protocol" / "valid_pool.json"
    document = {"settings": settings, "queries": queries, "candidates": {str(k): v for k, v in pool.items()}}
    if path.exists() and read_json(path) != document:
        raise ValueError("Stored valid pool does not match deterministic protocol")
    save_json(path, document)
    return pool


def evaluate_rankings(bundle, pool, rankings, ks, output_dir):
    """Exact-combination relevance; omitted/routed-out candidates are misses.

    AP@K uses ALL relevant candidates in the specified pool as its denominator,
    not the number retrieved in Top-K. Queries with R=0 are counted and excluded
    from mean AP/recall; precision still includes them. No coverage metric.
    """
    if set(rankings) != set(pool):
        raise ValueError("Rankings must contain exactly the query IDs in the frozen pool")
    allowed = set(int(i) for i in bundle.valid_ids)
    rows = []
    for q, candidates in pool.items():
        if q not in allowed or not set(candidates) <= allowed or q in candidates:
            raise ValueError("Validation evaluation accepts only held-out valid IDs, excluding self")
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("Candidate pool must be nonempty and contain unique IDs")
        ranking = [int(i) for i in rankings[q]]
        if len(ranking) != len(set(ranking)) or not set(ranking) <= set(candidates) or q in ranking:
            raise ValueError("Ranking contains duplicate, unknown or self candidate")
        if bundle.class_ids[q] < 0 or any(bundle.class_ids[c] < 0 for c in candidates):
            raise ValueError("This retrieval protocol is for the 37 nonempty combination classes")
        if max(ks) > len(candidates):
            raise ValueError("K cannot exceed the specified candidate pool size")
        total = int(np.sum(bundle.class_ids[candidates] == bundle.class_ids[q]))
        relevant = (bundle.class_ids[ranking] == bundle.class_ids[q]).astype(np.float64)
        cumulative = np.cumsum(relevant)
        precisions = cumulative / np.arange(1, len(ranking) + 1)
        row = {"query_id": int(q), "class_id": int(bundle.class_ids[q]),
               "pool_size": len(candidates), "relevant_in_pool": total, "returned": len(ranking)}
        for k in ks:
            hits = float(relevant[:k].sum())
            row[f"Precision@{k}"] = hits / k
            row[f"Recall@{k}"] = hits / total if total else None
            row[f"AP@{k}"] = float(np.sum(precisions[:k] * relevant[:k]) / total) if total else None
        rows.append(row)
    metrics = {}
    for k in ks:
        for prefix in ("Precision", "Recall", "AP"):
            values = [r[f"{prefix}@{k}"] for r in rows if r[f"{prefix}@{k}"] is not None]
            key = f"mAP@{k}" if prefix == "AP" else f"{prefix}@{k}"
            metrics[key] = float(np.mean(values)) if values else None
    result = {"num_queries": len(rows), "queries_without_relevant_candidates": sum(r["relevant_in_pool"] == 0 for r in rows),
              "metrics": metrics, "relevance": "exact equality of the entire nonempty defect-label combination",
              "ap_denominator": "all relevant candidates in the fixed pool (including screened/routed-out candidates)",
              "zero_relevant_policy": "exclude from mAP/Recall; include in Precision", "test_evaluated": False}
    output_dir = Path(output_dir)
    save_json(output_dir / "metrics.json", result)
    save_csv(output_dir / "per_query.csv", rows, list(rows[0]))
    return result


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for local checks")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N, mps, auto")
    return device

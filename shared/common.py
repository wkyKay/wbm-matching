"""Shared data/provenance utilities; algorithms live in wu/ and yu/.

Training and model selection use explicit train/valid IDs. Test retrieval is
available only through an explicit test stage, using a shared frozen protocol.
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
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from .data_protocol import deduplicate_maps
from .retrieval_protocol import FullGalleryPool, load_retrieval_protocol, protocol_file_hashes

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCE = Path(__file__).resolve().with_name("data_protocol.py")
RETRIEVAL_SOURCE = Path(__file__).resolve().with_name("retrieval_protocol.py")
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


def load_config(path, overrides=None, *, require_ks=True):
    """Load settings; feature-only methods need not configure retrieval metrics."""
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
    for key in ("data_file", "split_file", "retrieval_protocol_dir", "output_dir", "cache_dir"):
        if key not in config:
            continue
        value = Path(config[key]).expanduser()
        config[key] = str(value.resolve() if value.is_absolute() else (ROOT / value).resolve())
    if not isinstance(config.get("method"), str) or not config["method"].strip():
        raise ValueError("method must be a nonempty method name")
    if "valid_pool" in config or "retrieval_protocol_dir" not in config:
        raise ValueError("Configure retrieval_protocol_dir; runtime valid_pool sampling is no longer supported")
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if require_ks or "ks" in config:
        ks = config.get("ks")
        if (not isinstance(ks, list) or not ks or
                any(type(k) is not int or k < 1 for k in ks)):
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
    files = [Path(__file__).resolve(), PROTOCOL_SOURCE, RETRIEVAL_SOURCE] + [Path(p).resolve() for p in method_files]
    protocol_dir = Path(config["retrieval_protocol_dir"]).resolve()
    protected = [ROOT, Path(config["data_file"]), Path(config["split_file"]), protocol_dir] + files
    if any(output == p or output in p.parents for p in protected):
        raise ValueError("Output directory would contain code or source inputs")
    if protocol_dir in output.parents:
        raise ValueError("Output directory must not be inside the frozen retrieval protocol")
    ignored = {"output_dir", "device", "workers", "num_workers", "cpu_threads", "torch_threads",
               "data_file", "split_file", "retrieval_protocol_dir", "cache_dir", "resume"}
    settings = {k: v for k, v in config.items() if k not in ignored}
    sources = {key: hash_file(config[key]) for key in ("data_file", "split_file")}
    sources["retrieval_protocol"] = protocol_file_hashes(protocol_dir)
    frozen_metadata = read_json(protocol_dir / "metadata.json")
    if (frozen_metadata.get("source_sha256") != sources["data_file"] or
            frozen_metadata.get("split_sha256") != sources["split_file"]):
        raise ValueError("Frozen retrieval protocol belongs to different source data/split")
    code = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: hash_file(p)
            for p in files}
    payload = {"format_version": 3, "settings": settings, "sources": sources, "code": code}
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
    deduplication: dict
    full_valid_ids: np.ndarray
    retrieval_protocol: object = None

    def summary(self):
        return {"source_shape": list(self.maps.shape), "num_defect_classes": len(self.class_signatures),
                "used_defect_train": len(self.train_ids), "used_defect_valid": len(self.valid_ids),
                "used_normal_train": len(self.normal_train_ids), "used_normal_valid": len(self.normal_valid_ids),
                "reserved_defect_count": len(self.reserved_ids),
                "reserved_normal_count": len(self.normal_reserved_ids), "full_counts": self.full_counts,
                "deduplication": self.deduplication,
                "retrieval_protocol_signature": (self.retrieval_protocol.signature
                                                  if self.retrieval_protocol else None),
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
    # Audit raw pixels globally, before any class or split filtering. This checks
    # held-out source identity only; no test features or predictions are computed.
    deduplication = deduplicate_maps(maps, labels)
    representatives = deduplication.representative_ids
    unique_nonempty = representatives[np.asarray(labels[representatives]).sum(axis=1) > 0]
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
        if deduplication.representative_of[sample_id] != sample_id:
            raise ValueError(
                f"Split contains duplicate sample {sample_id}; canonical representative is "
                f"{deduplication.representative_of[sample_id]}. Use a new deduplicated protocol "
                "from scripts/build_protocol.py; the legacy split is preserved, not filtered silently.")
        expected = np.array([int(row[f"label_{i}"]) for i in range(8)], dtype=np.uint8)
        if not expected.any() or not np.array_equal(expected, labels[sample_id]):
            raise ValueError(f"Source/manifest label mismatch at {sample_id}")
        if "valid_index" in row:
            valid_index = int(row["valid_index"])
            if not 0 <= valid_index < len(nonempty) or nonempty[valid_index] != sample_id:
                raise ValueError("Frozen valid_index mismatch")
        seen.add(sample_id)
        ids[row["split"]].append(sample_id)
    if seen != set(unique_nonempty.tolist()) or any(not values for values in ids.values()):
        raise ValueError("Deduplicated split must partition all canonical nonempty-label wafers exactly once")
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
    normal_ids = representatives[np.asarray(labels[representatives]).sum(axis=1) == 0]
    normal_settings = config.get("normal_split", {"train": 700, "valid": 100, "holdout": 200})
    if set(normal_settings) != {"train", "valid", "holdout"} or any(
            not isinstance(v, int) or v < 0 for v in normal_settings.values()):
        raise ValueError("normal_split must contain nonnegative train/valid/holdout counts")
    if sum(normal_settings.values()) != len(normal_ids):
        raise ValueError("Normal split counts must partition the deduplicated empty-label samples")
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
    full_valid = np.asarray(sorted(ids["valid"]), dtype=np.int64)
    full_test = np.asarray(sorted(ids["test"]), dtype=np.int64)
    frozen = None
    if "retrieval_protocol_dir" in config:
        frozen = load_retrieval_protocol(
            config["retrieval_protocol_dir"], data_file=config["data_file"], split_file=config["split_file"],
            valid_ids=full_valid, test_ids=full_test, labels=labels)
    specification = {"source": hash_file(config["data_file"]), "split": hash_file(config["split_file"]),
                     "protocol_code": hash_file(PROTOCOL_SOURCE),
                     "retrieval_protocol": frozen.signature if frozen else None,
                     "deduplication": deduplication.summary(),
                     "seed": config["seed"], "subset": subset, "normal_split": normal_settings,
                     "class_signatures": signatures, "train_ids": chosen["train"], "valid_ids": chosen["valid"],
                     "normal_train_ids": chosen_normal["train"], "normal_valid_ids": chosen_normal["valid"]}
    bundle = DataBundle(maps, labels, class_ids, chosen["train"], chosen["valid"],
                        chosen_normal["train"], chosen_normal["valid"],
                        full_test, normal["holdout"],
                        [list(s) for s in signatures], signature(specification),
                        {**{s: len(v) for s, v in ids.items()}, "normal": len(normal_ids)},
                        deduplication.summary(), full_valid, frozen)
    output = Path(config["output_dir"]) / "protocol"
    previous = output / "data_summary.json"
    if previous.exists() and read_json(previous).get("signature") != bundle.signature:
        raise ValueError("Data protocol changed: choose a new output_dir; old protocol records are preserved")
    save_json(output / "data_summary.json", bundle.summary())
    save_json(output / "deduplication.json", {**deduplication.summary(),
              "source_sha256": hash_file(config["data_file"]),
              "split_sha256": hash_file(config["split_file"]),
              "representative_rule": "smallest original NPZ row ID; all raw pixel values compared",
              "audit_only_heldout_pixels": True})
    save_csv(output / "removed_duplicates.csv",
             ({"removed_id": int(i), "representative_id": int(group[0])}
              for group in deduplication.duplicate_groups for i in group[1:]),
             ["removed_id", "representative_id"])
    save_json(output / "class_mapping.json", {"derived_from": "train only", "normal_id": -1,
                                               "classes": bundle.class_signatures})
    save_csv(output / "normal_split.csv", ({"sample_id": int(i), "split": split}
               for split in ("train", "valid", "holdout") for i in normal[split]), ["sample_id", "split"])
    for name in ("train_ids", "valid_ids", "normal_train_ids", "normal_valid_ids"):
        save_csv(output / f"{name}.csv", ({"sample_id": int(i)} for i in getattr(bundle, name)), ["sample_id"])
    return bundle


def make_retrieval_pool(bundle, config, split_name="valid"):
    """Read shared frozen queries/gallery; never draw a new evaluation pool.

    Smoke validation uses the intersection with its explicit small valid subset.
    It is a debug-only derivative of the frozen lists, not a benchmark result.
    """
    if split_name not in ("valid", "test"):
        raise ValueError("Retrieval split must be valid or test")
    if "valid_pool" in config:
        raise ValueError("Runtime valid_pool sampling is no longer supported")
    if not config.get("retrieval_protocol_dir"):
        raise ValueError("Set retrieval_protocol_dir to a generated, frozen protocol")
    frozen = bundle.retrieval_protocol
    if frozen is None:
        raise ValueError("Prepare data with retrieval_protocol_dir before retrieval")
    # Prevent changing a protocol after data preparation or substituting a path.
    if signature(protocol_file_hashes(config["retrieval_protocol_dir"])) != frozen.signature:
        raise ValueError("Retrieval protocol changed after data preparation")
    pool = getattr(frozen, split_name)
    smoke = bool(config.get("smoke", False))
    subset = any(value is not None for value in config.get("subset", {}).values())
    if split_name == "test" and (smoke or subset):
        raise ValueError("Test evaluation requires a full formal configuration, not a smoke/subset run")
    if split_name == "valid" and smoke:
        gallery = np.asarray(sorted(bundle.valid_ids), dtype=np.int64)
        queries = np.intersect1d(pool.queries, gallery)
        if not len(queries):
            raise ValueError("Smoke valid subset contains no frozen queries; use a larger smoke subset")
        pool = FullGalleryPool(queries, gallery, "valid", frozen.signature)
    elif split_name == "valid" and not np.array_equal(bundle.valid_ids, pool.gallery):
        raise ValueError("Formal validation requires the complete frozen valid gallery; use smoke only for debugging")
    path = Path(config["output_dir"]) / "protocol" / f"{split_name}_pool.json"
    document = {"format_version": 2, "split": split_name,
                "protocol_signature": frozen.signature,
                "derived_smoke": smoke,
                "queries": pool.queries, "gallery": pool.gallery,
                "candidate_rule": "all_gallery_except_query", "candidates_per_query": pool.candidates_per_query}
    if path.exists() and signature(read_json(path)) != signature(document):
        raise ValueError("Stored retrieval pool differs from the frozen protocol; use a new output_dir")
    save_json(path, document)
    return pool


def make_valid_pool(bundle, config):
    """Compatibility wrapper for methods using the shared validation protocol."""
    return make_retrieval_pool(bundle, config, "valid")


def evaluate_rankings(bundle, pool, rankings, ks, output_dir, split_name="valid"):
    """Exact-combination relevance; omitted/routed-out candidates are misses.

    AP@K uses ALL relevant candidates in the specified pool as its denominator,
    not the number retrieved in Top-K. Queries with R=0 are counted and excluded
    from mean AP/recall; precision still includes them. No coverage metric.
    """
    if split_name not in ("valid", "test"):
        raise ValueError("Retrieval split must be valid or test")
    if hasattr(pool, "split_name") and pool.split_name != split_name:
        raise ValueError("Retrieval pool split does not match the requested evaluation")
    allowed_ids = bundle.valid_ids if split_name == "valid" else bundle.reserved_ids
    allowed = set(int(i) for i in allowed_ids)
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integer cutoffs")
    query_ids = set(pool)
    if not query_ids:
        raise ValueError("Retrieval pool must contain at least one query")
    items = rankings.items() if isinstance(rankings, Mapping) else iter(rankings)
    rows = []
    seen = set()
    for q, ranking in items:
        if q not in query_ids or q in seen:
            raise ValueError("Rankings must contain exactly one result per frozen query")
        seen.add(q)
        candidates = pool[q]
        candidate_set = set(candidates)
        if q not in allowed or not candidate_set <= allowed or q in candidate_set:
            raise ValueError(f"{split_name} evaluation accepts only its held-out IDs, excluding self")
        if len(candidates) == 0 or len(candidates) != len(candidate_set):
            raise ValueError("Candidate pool must be nonempty and contain unique IDs")
        ranking = [int(i) for i in ranking]
        if len(ranking) != len(set(ranking)) or not set(ranking) <= candidate_set or q in ranking:
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
    if seen != query_ids:
        raise ValueError("Rankings must contain exactly the query IDs in the frozen pool")
    metrics = {}
    for k in ks:
        for prefix in ("Precision", "Recall", "AP"):
            values = [r[f"{prefix}@{k}"] for r in rows if r[f"{prefix}@{k}"] is not None]
            key = f"mAP@{k}" if prefix == "AP" else f"{prefix}@{k}"
            metrics[key] = float(np.mean(values)) if values else None
    result = {"num_queries": len(rows), "queries_without_relevant_candidates": sum(r["relevant_in_pool"] == 0 for r in rows),
              "metrics": metrics, "relevance": "exact equality of the entire nonempty defect-label combination",
              "ap_denominator": "all relevant candidates in the fixed pool (including screened/routed-out candidates)",
              "zero_relevant_policy": "exclude from mAP/Recall; include in Precision",
              "split": split_name, "test_evaluated": split_name == "test",
              "retrieval_protocol_signature": getattr(pool, "protocol_signature", None)}
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

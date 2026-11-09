"""Build a new deduplicated protocol without overwriting source or old splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.data_protocol import BASE_CLASS_NAMES, DEDUPLICATION_POLICY, deduplicate_maps, stratified_split


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_protocol(data_file, output_dir, seed=2026):
    """Create a new protocol directory; explicit relative paths use the cwd."""
    data_file = Path(data_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing protocol directory: {output_dir}")
    source_hash = _sha256(data_file)
    with np.load(data_file, allow_pickle=False) as archive:
        maps, labels = archive["arr_0"], archive["arr_1"]
    deduplication = deduplicate_maps(maps, labels)
    splits = stratified_split(labels, deduplication.representative_ids, seed=seed)
    # valid_index retains its historical meaning: index among ALL nonempty rows.
    nonempty = np.flatnonzero(labels.sum(axis=1) > 0)
    valid_index = {int(sample_id): index for index, sample_id in enumerate(nonempty)}
    rows = []
    for split, ids in splits.items():
        for sample_id in ids:
            sample_id = int(sample_id)
            row = {"sample_id": sample_id, "original_index": sample_id,
                   "valid_index": valid_index[sample_id], "split": split,
                   "label_signature": "|".join(map(str, np.flatnonzero(labels[sample_id])))}
            row.update({f"label_{i}": int(labels[sample_id, i]) for i in range(8)})
            rows.append(row)
    removed = [{"removed_id": int(sample_id), "representative_id": int(representative)}
               for sample_id, representative in enumerate(deduplication.representative_of)
               if sample_id != representative]
    if _sha256(data_file) != source_hash:
        raise RuntimeError("Source NPZ changed while preparing the protocol")
    # Exclusive directory creation and exclusive file writes preserve all old files.
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "split.csv", rows,
               ["sample_id", "original_index", "valid_index", "split", "label_signature"]
               + [f"label_{i}" for i in range(8)])
    _write_csv(output_dir / "duplicates.csv", removed, ["removed_id", "representative_id"])
    metadata = {
        "format_version": 1,
        "data_file": str(data_file),
        "source_sha256": source_hash,
        "source_shape": list(maps.shape),
        "deduplication_policy": DEDUPLICATION_POLICY,
        "representative_policy": "smallest original NPZ row ID",
        "identity": "all raw pixel codes 0/1/2/3; hash candidates confirmed by array_equal",
        "label_conflicts": "rejected",
        "seed": int(seed),
        "split_ratios": {"train": 0.7, "valid": 0.1, "test": 0.2},
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "class_names": list(BASE_CLASS_NAMES),
        "summary": deduplication.summary(),
        "normal_policy": "unique Normal IDs excluded from split.csv; runtime uses explicit normal_split counts",
        "valid_index_policy": "index in original nonempty-label source rows, before deduplication",
        "files": {name: _sha256(output_dir / name) for name in ("split.csv", "duplicates.csv")},
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=str(ROOT.parent / "data/wm38k/Wafer_Map_Datasets.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "protocol/deduplicated"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = build_protocol(args.data_file, args.output_dir, seed=args.seed)
    print(json.dumps({"output_dir": str(Path(args.output_dir).expanduser().resolve()),
                      "summary": result["summary"], "split_counts": result["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()

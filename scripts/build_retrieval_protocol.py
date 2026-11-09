"""Freeze stratified valid queries and exhaustive test queries in a new directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.retrieval_protocol import build_retrieval_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=str(ROOT.parent / "data/wm38k/Wafer_Map_Datasets.npz"))
    parser.add_argument("--split-file", default=str(ROOT / "protocol/deduplicated/split.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "protocol/retrieval_v1"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--valid-queries", type=int, default=500)
    parser.add_argument("--min-per-class", type=int, default=5)
    args = parser.parse_args()
    result = build_retrieval_protocol(args.data_file, args.split_file, args.output_dir,
                                      args.seed, args.valid_queries, args.min_per_class)
    print(json.dumps({"output_dir": str(Path(args.output_dir).expanduser().resolve()),
                      "valid": result["valid"], "test": result["test"],
                      "candidate_rule": result["candidate_rule"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate any of the five methods with the frozen exact-combination protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Method config; relative to the command's working directory")
    parser.add_argument("--split", choices=("valid", "test"), required=True,
                        help="Explicit held-out split; test rejects smoke/subset configurations")
    parser.add_argument("--output-dir", required=True, help="New evaluation directory, relative to compared_methods")
    parser.add_argument("--source-run", help="Read-only Yu/Nakazawa trained run; defaults to config output_dir")
    parser.add_argument("--ks", type=int, nargs="+", help="Cutoffs; formal default 1 5 10 20 50 100 500, smoke 1 5 10 20")
    parser.add_argument("--variant", choices=("global", "predicted_class"), default="global",
                        help="Yu global is the main comparison; predicted_class is a separate auxiliary result")
    parser.add_argument("--device", help="CPU/CUDA inference device; no training is performed")
    parser.add_argument("--workers", type=int, default=1, help="Workers for handcrafted feature preparation")
    parser.add_argument("--rank-workers", type=int, default=1,
                        help="Processes for full-candidate query ranking; bounded queue, preserves query order")
    args = parser.parse_args(argv)
    from shared.common import load_config
    from shared.evaluation import run_evaluation
    config = load_config(args.config, require_ks=False)
    result = run_evaluation(config, args.output_dir, split_name=args.split, ks=args.ks,
                            source_run=args.source_run, variant=args.variant,
                            device=args.device, workers=args.workers, rank_workers=args.rank_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only F checkpoint/cache reuse for cross-combination coverage evaluation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-project", type=Path, default=ROOT.parent / "wbm-region-retrieval")
    parser.add_argument("--config", type=Path, help="F suite config; default: project's canonical F config")
    parser.add_argument("--source-run", type=Path, required=True, help="Existing F suite root containing cache/ and seed_2026/")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory, relative to compared_methods")
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    project = args.f_project.resolve()
    sys.path.insert(0, str(project))
    import numpy as np
    import torch
    from shared import common, coverage_metrics
    from shared.cross_combination import CrossCombinationPool
    from shared.exact_metrics import order_scores
    from shared.evaluation import _fresh_output
    from experiments.proposed._shared.ef_suite import load_suite, code_hashes, build_model, embeddings
    from experiments.proposed._shared.ef_training import run_signature
    from wbm_retrieval.common import digest, get_device, seed_everything
    from wbm_retrieval.data import Protocol, source_fingerprint
    from wbm_retrieval.diagnostics import batched_map_scores
    from wbm_retrieval.training import load_checkpoint

    source = args.source_run.resolve()
    suite, config = load_suite(args.config or project / "experiments/proposed/f/config.json",
                               output_root=str(source), seed=args.seed, device=args.device)
    if config.get("data_protocol") != "deduplicated_v1" or config["evaluation"]["metric_version"] != "exact_v1":
        raise ValueError("F source must use the frozen deduplicated exact_v1 protocol")
    output = _fresh_output(args.output_dir, {**config, "split_file": config["split_manifest"]}, source)
    for name in ("wbm_retrieval", "experiments", "protocol", "scripts", "tests", "doc", ".git"):
        protected = project / name
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Output overlaps F project source/protocol")
    ks = sorted(set(args.ks))
    if not ks or min(ks) < 1 or len(ks) != len(args.ks):
        raise ValueError("K must be positive and unique")
    manifest = common.read_json(source / "suite_manifest.json")
    identity = {k: v for k, v in manifest.items() if k != "signature"}
    settings = {k: v for k, v in config.items() if k not in (
        "device", "num_workers", "seed", "output_dir", "data_file", "split_manifest", "query_manifest", "candidate_manifest",
        "deduplication_metadata", "retrieval_protocol_dir")}
    expected = {"sources": source_fingerprint(config), "settings": settings,
                "suite": {k: suite[k] for k in ("suite_id", "variants", "valid_pool", "selection_metric", "evaluate_every")},
                "code": code_hashes()}
    if identity != expected or digest(identity) != manifest["signature"]:
        raise ValueError("F source code/config/data signature mismatch; do not rewrite the source manifest")
    cache = source / "cache"
    prepared = common.read_json(cache / "prepared.json")
    if prepared["signature"] != manifest["signature"]:
        raise ValueError("F cache signature mismatch")
    checks = {source / name: sha for name, sha in prepared["files"].items()}
    model_dir = source / f"seed_{args.seed}" / "F"
    checkpoint_path = model_dir / "best_nce.pt"
    for path in (source / "suite_manifest.json", cache / "prepared.json",
                 model_dir / "training_summary.json", checkpoint_path):
        checks[path] = common.hash_file(path)
    for path, sha in checks.items():
        if common.hash_file(path) != sha:
            raise ValueError(f"Missing/changed source artifact: {path}")
    summary = common.read_json(model_dir / "training_summary.json")
    if not summary["completed"] or summary["epochs_completed"] != config["local"]["epochs"]:
        raise ValueError("F training is incomplete")
    checkpoint = load_checkpoint(checkpoint_path)
    if (checkpoint["signature"] != run_signature(manifest["signature"], args.seed, "F")
            or checkpoint["epoch"] != summary["best_nce_epoch"]):
        raise ValueError("F best_nce checkpoint provenance mismatch")
    labels = np.load(cache / "labels.npy", mmap_mode="r")
    protocol = Protocol(config)
    base_pool = getattr(protocol.retrieval, args.split)
    pool = CrossCombinationPool(base_pool, labels)
    if max(ks) > pool.candidates_per_query:
        raise ValueError("K exceeds the smallest filtered candidate pool")
    # Ensure F and the five methods use the same actual frozen ID files.
    for relative in ("deduplicated/split.csv", "retrieval_v1/valid_queries.csv", "retrieval_v1/valid_gallery.csv",
                     "retrieval_v1/test_queries.csv", "retrieval_v1/test_gallery.csv"):
        if common.hash_file(project / "protocol" / relative) != common.hash_file(ROOT / "protocol" / relative):
            raise ValueError(f"Cross-project frozen protocol mismatch: {relative}")
    with np.load(cache / f"{args.split}_regions.npz") as archive:
        meta = dict(archive)
    if set(meta["map_ids"].tolist()) != set(pool.gallery.tolist()):
        raise ValueError("F region cache does not cover the frozen gallery exactly")
    code_paths = [Path(__file__), ROOT / "shared/cross_combination.py", ROOT / "shared/coverage_metrics.py",
                  ROOT / "shared/exact_metrics.py", ROOT / "shared/evaluation.py", ROOT / "shared/common.py"]
    checks.update({p: common.hash_file(p) for p in code_paths})
    payload = {"method": "F", "checkpoint": "best_nce", "epoch": checkpoint["epoch"],
               "source_run": str(source), "source_signature": manifest["signature"],
               "source_files": {str(p): sha for p, sha in checks.items()},
               "derived_protocol": pool.metadata, "ks": ks, "split": args.split,
               "tie_seed": config["evaluation"]["tie_seed"], "ranking_limit": max(ks)}
    evaluation_signature = common.signature(payload)
    output.mkdir(parents=True, exist_ok=False)
    try:
        common.save_json(output / "evaluation_manifest.json", {**payload, "signature": evaluation_signature})
        common.save_json(output / "environment.json", common.environment_info())
        common.save_csv(output / "cross_combination_candidates.csv", pool.audit, list(pool.audit[0]))
        destination = output / args.split
        destination.mkdir()
        seed_everything(args.seed)
        device = get_device(args.device)
        model = build_model(config, suite["variants"]["F"]).to(device)
        model.load_state_dict(checkpoint["model"])
        with torch.inference_mode():
            features = embeddings(model, cache / f"{args.split}_F.npy", config, device)
        indices = {int(i): np.arange(a, b) for i, a, b in zip(meta["map_ids"], meta["offsets"][:-1], meta["offsets"][1:])}
        padded = np.full((len(meta["map_ids"]), int(np.diff(meta["offsets"]).max())), -1, dtype=np.int64)
        positions = {int(i): n for n, i in enumerate(meta["map_ids"])}
        for sample_id, tokens in indices.items():
            padded[positions[sample_id], :len(tokens)] = tokens
        ranking_path = destination / "rankings.csv"
        with ranking_path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])
            for number, (q, candidates) in enumerate(pool.items(), 1):
                tokens = indices[q]
                similarities = features[tokens] @ features.T
                scores = batched_map_scores(similarities, padded[[positions[int(c)] for c in candidates]],
                                           meta["area"][tokens], config["matching"])
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite F score")
                order = order_scores(candidates, scores, q, config["evaluation"]["tie_seed"])[:max(ks)]
                writer.writerows((q, r, int(candidates[i]), float(scores[i])) for r, i in enumerate(order, 1))
                if number % 100 == 0 or number == len(pool):
                    print(f"F {args.split}: {number}/{len(pool)} queries", flush=True)
        taxonomy = {label: i for i, label in enumerate(sorted({tuple(labels[q]) for q in pool.gallery}))}
        classes = np.full(len(labels), -1, dtype=np.int64)
        for q in pool.gallery:
            classes[q] = taxonomy[tuple(labels[q])]
        result = coverage_metrics.evaluate_rankings_file(
            SimpleNamespace(labels=labels, class_ids=classes), pool, ranking_path, ks, destination,
            split_name=args.split, metadata={"evaluation_signature": evaluation_signature,
                                            "method": "F", "derived_protocol": pool.metadata})
        if source_fingerprint(config) != expected["sources"] or code_hashes() != expected["code"]:
            raise ValueError("F sources changed during evaluation")
        if any(common.hash_file(p) != sha for p, sha in checks.items()):
            raise ValueError("Source artifact or evaluator changed during evaluation")
        common.save_json(output / "completed.json", {"evaluation_signature": evaluation_signature,
            "num_queries": len(pool), "split": args.split, "test_evaluated": args.split == "test",
            "files": {str(p.relative_to(output)): common.hash_file(p) for p in destination.iterdir() if p.is_file()}})
        print(result["metrics"])
        return 0
    except Exception as error:
        common.save_json(output / "failed.json", {"complete": False, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Optional real-data DataLoader spawn check; uses only four train IDs."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import common
from methods.yu import yu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", help="New audit output directory; relative paths use compared_methods")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(2)
    output = args.output_dir
    if output is None:
        audits = ROOT / "artifacts/validation"
        audits.mkdir(parents=True, exist_ok=True)
        output = tempfile.mkdtemp(prefix="dataloader_workers_", dir=audits)
    config = common.load_config(ROOT / "methods/yu/smoke.json",
                                {"output_dir": output, "cache_dir": str(ROOT / "artifacts/_data")})
    common.ensure_run(config, [Path(__file__), Path(yu.__file__)])
    bundle = common.prepare_data(config)
    ids = bundle.train_ids[:4]
    dataset = yu.WaferDataset(bundle, ids, task="classifier", training=True, seed=config["seed"], epoch=3)
    expected = [dataset[i] for i in range(len(dataset))]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=args.workers,
                                        multiprocessing_context="spawn", shuffle=False)
    seen = 0
    for sample_ids, images, labels in loader:
        for sample_id, image, label in zip(sample_ids, images, labels):
            e_id, e_image, e_label = expected[seen]
            assert sample_id.item() == e_id and label.item() == e_label
            assert torch.equal(image, e_image), "Worker preprocessing differs from serial execution"
            seen += 1
    assert seen == 4
    report = {"workers": args.workers, "samples": seen, "identical_to_serial": True,
              "device": "cpu", "multiprocessing_start": "spawn", "test_processed": False,
              "output_dir": config["output_dir"]}
    common.save_json(Path(config["output_dir"]) / "dataloader_workers.json", report)
    print(report)


if __name__ == "__main__":
    main()

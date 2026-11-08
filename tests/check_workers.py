"""Optional real-data DataLoader spawn check; uses only four train IDs."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "yu"))
import common
import yu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    torch.set_num_threads(2)
    config = common.load_config(ROOT / "yu/smoke.json")
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
              "device": "cpu", "multiprocessing_start": "spawn", "test_processed": False}
    common.save_json(ROOT / "artifacts/validation/dataloader_workers.json", report)
    print(report)


if __name__ == "__main__":
    main()

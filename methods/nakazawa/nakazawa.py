#!/usr/bin/env python3
"""Nakazawa & Kulkarni (2018), adapted to the frozen MixedWM38 protocol.

Paper: doi:10.1109/TSM.2018.2795466, II-B/C, Table I and III-B.
This module implements classification training, sigmoid features, binary codes
and Hamming scoring. It does not define retrieval relevance or test metrics.
Unreported training details are explicit implementation conventions; see README.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.common import (ensure_run, file_lock, get_device, hash_file, load_config,
                           make_retrieval_pool, prepare_data, read_json,
                           save_csv, save_json, seed_everything, signature)

INPUT_SHAPE = (286, 400)
CODE_BITS = 256
DEFAULT_PARAMETERS = {"input_shape": [286, 400], "input_channels": 1,
                      "conv_padding": "valid", "dropout": 0.5,
                      "resize": "nearest", "initialization": "xavier_uniform"}
DEFAULT_TRAINING = {"epochs": 10, "batch_size": 385, "micro_batch_size": 8,
                    "optimizer": "adam", "learning_rate": 0.001,
                    "betas": [0.9, 0.999], "epsilon": 1e-8,
                    "weight_decay": 0.0, "selection": "last_epoch",
                    "max_train_batches": None, "max_valid_batches": None}


def parameters(overrides=None):
    unknown = set(overrides or {}) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown model parameters: {sorted(unknown)}")
    p = {**DEFAULT_PARAMETERS, **(overrides or {})}
    if tuple(p["input_shape"]) != INPUT_SHAPE or p["input_channels"] != 1:
        raise ValueError("MixedWM38 adaptation uses one channel at the paper's 286 x 400 size")
    if p["conv_padding"] not in ("valid", "same"):
        raise ValueError("conv_padding must be valid or same (paper does not specify padding)")
    if not isinstance(p["dropout"], (int, float)) or not 0 <= p["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if p["resize"] != "nearest" or p["initialization"] != "xavier_uniform":
        raise ValueError("This implementation fixes nearest resize and Xavier uniform initialization")
    return p


def training_parameters(overrides=None, *, smoke=False):
    unknown = set(overrides or {}) - set(DEFAULT_TRAINING)
    if unknown:
        raise ValueError(f"Unknown training parameters: {sorted(unknown)}")
    p = {**DEFAULT_TRAINING, **(overrides or {})}
    for key in ("epochs", "batch_size", "micro_batch_size"):
        if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if p["micro_batch_size"] > p["batch_size"]:
        raise ValueError("micro_batch_size cannot exceed batch_size")
    if p["optimizer"] != "adam" or p["selection"] != "last_epoch":
        raise ValueError("Configured convention is Adam and last_epoch, with no validation-based selection")
    if not math.isfinite(p["learning_rate"]) or p["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if len(p["betas"]) != 2 or any(not 0 <= value < 1 for value in p["betas"]):
        raise ValueError("Adam betas must be in [0, 1)")
    if not math.isfinite(p["epsilon"]) or p["epsilon"] <= 0:
        raise ValueError("Adam epsilon must be positive and finite")
    if not math.isfinite(p["weight_decay"]) or p["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for key in ("max_train_batches", "max_valid_batches"):
        if p[key] is not None and (not smoke or isinstance(p[key], bool)
                                   or not isinstance(p[key], int) or p[key] < 1):
            raise ValueError(f"{key} is a positive debug-only limit, allowed only with smoke=true")
    return p


def preprocess_wafer(wafer, settings=None):
    """Use observed failed-die occupancy; do not invent unavailable defect counts.

    Original density maps contain count per die / maximum count. MixedWM38 only
    supplies pass/fail, so a failed die contributes one and all other codes zero.
    Nearest resizing preserves those values; no filtering, color map or augment.
    """
    parameters(settings)
    wafer = np.asarray(wafer)
    if wafer.ndim != 2 or min(wafer.shape) < 1 or not np.isin(wafer, [0, 1, 2, 3]).all():
        raise ValueError("Expected a nonempty 2D MixedWM38 map with codes 0/1/2/3")
    image = torch.from_numpy(np.asarray(wafer == 2, dtype=np.float32))
    return F.interpolate(image[None, None], size=INPUT_SHAPE, mode="nearest")[0]


class NakazawaCNN(nn.Module):
    """Table I: [conv, ReLU, maxpool] x3 -> FC256 sigmoid -> dropout -> FC.

    The output FC has 37 classes for MixedWM38 instead of the paper's 22.
    Forward returns logits for numerically stable softmax cross entropy; use
    predict_proba for the paper's final softmax probabilities.
    """
    def __init__(self, num_classes=37, settings=None):
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        self.settings = parameters(settings)
        self.num_classes = num_classes
        padding = 0 if self.settings["conv_padding"] == "valid" else 1
        layers = []
        channels = 1
        height, width = INPUT_SHAPE
        for output_channels in (32, 32, 64):
            layers.extend([nn.Conv2d(channels, output_channels, 3, stride=1, padding=padding),
                           nn.ReLU(), nn.MaxPool2d(2, stride=2)])
            height, width = (height + 2 * padding - 2) // 2, (width + 2 * padding - 2) // 2
            channels = output_channels
        self.convolutions = nn.Sequential(*layers)
        self.feature_shape = (channels, height, width)
        self.fc = nn.Linear(channels * height * width, CODE_BITS)
        self.dropout = nn.Dropout(float(self.settings["dropout"]))
        self.classifier = nn.Linear(CODE_BITS, num_classes)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def features(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, *INPUT_SHAPE):
            raise ValueError("CNN inputs must have shape (N, 1, 286, 400)")
        return torch.sigmoid(self.fc(self.convolutions(inputs).flatten(1)))

    def forward(self, inputs):
        return self.classifier(self.dropout(self.features(inputs)))

    def predict_proba(self, inputs):
        return self(inputs).softmax(dim=1)


def binary_codes(features):
    """Section II-C uses >= 0.5, including the equality boundary."""
    values = np.asarray(features)
    if values.ndim not in (1, 2) or values.shape[-1] != CODE_BITS:
        raise ValueError("Expected 256 sigmoid features per wafer")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Sigmoid features must be finite values in [0, 1]")
    return (values >= 0.5).astype(np.uint8)


def hamming_distances(query_code, candidate_codes):
    """Return integer bit differences, not a distance between real-valued features."""
    query, candidates = np.asarray(query_code), np.asarray(candidate_codes)
    if query.shape != (CODE_BITS,) or candidates.ndim != 2 or candidates.shape[1] != CODE_BITS:
        raise ValueError("Expected a 256-bit query and (N, 256) candidate codes")
    if not np.isin(query, [0, 1]).all() or not np.isin(candidates, [0, 1]).all():
        raise ValueError("Hamming inputs must be binary")
    return np.count_nonzero(candidates != query, axis=1).astype(np.int16)


def score_candidates(query_code, candidate_codes):
    """Descending-compatible score: negative integer Hamming distance."""
    return -hamming_distances(query_code, candidate_codes)


def rank_candidates(query_code, candidate_codes, candidate_ids, *, query_id=None):
    """Full ranking; ties use ascending original sample_id, without label access."""
    ids = np.asarray(candidate_ids)
    distances = hamming_distances(query_code, candidate_codes)
    if ids.ndim != 1 or len(ids) != len(distances) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("candidate_ids must be a one-dimensional integer array matching codes")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("candidate_ids must be unique and nonnegative")
    if query_id is not None:
        keep = ids != int(query_id)
        ids, distances = ids[keep], distances[keep]
    order = np.lexsort((ids, distances))
    return ids[order], distances[order]


class WaferDataset(Dataset):
    def __init__(self, maps, sample_ids, class_ids, settings=None):
        self.maps = maps
        self.sample_ids = np.asarray(sample_ids, dtype=np.int64)
        self.class_ids = class_ids
        self.settings = parameters(settings)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = int(self.sample_ids[index])
        target = int(self.class_ids[sample_id])
        if target < 0:
            raise ValueError("Normal wafers are not training classes in this MixedWM38 adaptation")
        return preprocess_wafer(self.maps[sample_id], self.settings), target


def _loader(bundle, ids, config, batch_size, *, shuffle=False, epoch=0):
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    dataset = WaferDataset(bundle.maps, ids, bundle.class_ids, config.get("parameters"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      generator=generator, num_workers=int(config.get("num_workers", 0)))


def classification_epoch(model, loader, device, *, optimizer=None, micro_batch_size=8, max_batches=None):
    """Train or monitor classification; microbatches preserve effective batch size.

    Loss gradients are summed / effective batch count, with one optimizer update
    per loader batch. This is not a smaller-batch training substitution.
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = seen = updates = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        effective_count = len(labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
        for start in range(0, effective_count, micro_batch_size):
            x = images[start:start + micro_batch_size].to(device)
            y = labels[start:start + micro_batch_size].to(device)
            with torch.set_grad_enabled(training):
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
                if training:
                    (loss / effective_count).backward()
            loss_sum += float(loss.detach().cpu())
            correct += int((logits.detach().argmax(1) == y).sum().cpu())
            seen += len(y)
        if training:
            optimizer.step()
            updates += 1
    if not seen:
        raise ValueError("Classification loader contains no usable samples")
    return {"loss": loss_sum / seen, "accuracy": correct / seen,
            "samples": seen, "optimizer_updates": updates}


def _torch_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _run_signature(config):
    return read_json(Path(config["output_dir"]) / "run_manifest.json")["signature"]


def audit(bundle, config):
    pool = make_retrieval_pool(bundle, config, "valid")
    record = {"method": "nakazawa2018", "data": bundle.summary(),
              "parameters": parameters(config.get("parameters")),
              "training": training_parameters(config.get("training"), smoke=bool(config.get("smoke"))),
              "input_representation": "one-channel failed-die occupancy; nearest resize to 286x400",
              "input_limitation": "MixedWM38 has no per-die defect counts; original density cannot be reconstructed",
              "model_selection": "last epoch; paper reports 10 epochs but does not define checkpoint selection",
              "validation_queries": len(pool), "validation_gallery": len(pool.gallery),
              "validation_role": "classification monitoring only; no retrieval metric or selection",
              "test_features_computed": False, "retrieval_metrics_defined": False}
    save_json(Path(config["output_dir"]) / "audit.json", record)
    return record


def prepare(bundle, config):
    """Record exact training/validation IDs; resize lazily to avoid an 11 GB cache."""
    pool = make_retrieval_pool(bundle, config, "valid")
    result = {"train_samples": len(bundle.train_ids), "valid_samples": len(bundle.valid_ids),
              "queries": len(pool), "gallery": len(pool.gallery),
              "input_preprocessing": "lazy per image, no image-level fitting",
              "normal_used_for_training": False, "data_signature": bundle.signature}
    save_json(Path(config["output_dir"]) / "prepared.json", result)
    return result


def _load_checkpoint(path, bundle, config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("run_signature") != _run_signature(config) or checkpoint.get("data_signature") != bundle.signature:
        raise ValueError("Checkpoint code/config/data signature mismatch")
    if checkpoint.get("class_signatures") != bundle.class_signatures:
        raise ValueError("Checkpoint class taxonomy mismatch")
    return checkpoint


def train(bundle, config, *, resume=False):
    # Matching signatures do not make two writers safe. Serialize each run's
    # training, and re-check completion/existing files only after acquiring it.
    with file_lock(Path(config["output_dir"]) / ".training.lock"):
        return _train(bundle, config, resume=resume)


def _train(bundle, config, *, resume=False):
    p = training_parameters(config.get("training"), smoke=bool(config.get("smoke")))
    output = Path(config["output_dir"])
    checkpoint_path = output / "checkpoint_last.pt"
    completed = output / "training_complete.json"
    if completed.exists():
        result = read_json(completed)
        if not checkpoint_path.exists() or result["checkpoint_sha256"] != hash_file(checkpoint_path):
            raise ValueError("Completed checkpoint is missing or changed")
        _load_checkpoint(checkpoint_path, bundle, config)
        return result
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("Training already started; pass --resume or choose a new output directory")
    seed_everything(config["seed"])
    device = get_device(config.get("device", "auto"))
    model = NakazawaCNN(len(bundle.class_signatures), config.get("parameters")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"],
                                 betas=tuple(p["betas"]), eps=p["epsilon"], weight_decay=p["weight_decay"])
    first_epoch, history = 1, []
    if checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, bundle, config)
        if checkpoint.get("training_device_type") != device.type:
            raise ValueError("Resume requires the original device type to restore dropout RNG; encoding may use another device")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if device.type == "mps" and checkpoint.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    valid_loader = _loader(bundle, bundle.valid_ids, config, p["micro_batch_size"])
    started = time.perf_counter()
    for epoch in range(first_epoch, p["epochs"] + 1):
        loader = _loader(bundle, bundle.train_ids, config, p["batch_size"], shuffle=True, epoch=epoch)
        training = classification_epoch(model, loader, device, optimizer=optimizer,
                                        micro_batch_size=p["micro_batch_size"], max_batches=p["max_train_batches"])
        validation = classification_epoch(model, valid_loader, device,
                                          micro_batch_size=p["micro_batch_size"], max_batches=p["max_valid_batches"])
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in training.items()},
                        **{f"valid_{k}": v for k, v in validation.items() if k != "optimizer_updates"}})
        _torch_save(checkpoint_path, {"format_version": 1, "epoch": epoch,
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "mps_rng_state": torch.mps.get_rng_state() if device.type == "mps" else None,
                    "training_device_type": device.type,
                    "history": history, "run_signature": _run_signature(config),
                    "data_signature": bundle.signature, "class_signatures": bundle.class_signatures})
        save_csv(output / "training_history.csv", history, list(history[0]))
        print(json.dumps({"training_epoch": history[-1]}, ensure_ascii=False), flush=True)
    result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": hash_file(checkpoint_path),
              "epochs": p["epochs"], "selection": "last_epoch", "validation_used_to_select": False,
              "final_epoch": history[-1], "seconds_this_invocation": time.perf_counter() - started,
              "smoke": bool(config.get("smoke")), "test_processed": False}
    save_json(completed, result)
    return result


def load_trained_model(bundle, config, device=None):
    output = Path(config["output_dir"])
    if not (output / "training_complete.json").exists():
        raise FileNotFoundError("Run --stage train to completion before encoding")
    record = read_json(output / "training_complete.json")
    path = output / "checkpoint_last.pt"
    if hash_file(path) != record["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint hash mismatch")
    checkpoint = _load_checkpoint(path, bundle, config)
    if checkpoint["epoch"] != training_parameters(config.get("training"), smoke=bool(config.get("smoke")))["epochs"]:
        raise ValueError("Encoding requires the configured final epoch")
    device = get_device(config.get("device", "auto")) if device is None else device
    model = NakazawaCNN(len(bundle.class_signatures), config.get("parameters")).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), device


def encode_maps(model, maps, sample_ids, *, batch_size=8, device=None, settings=None):
    """Public, label-free API returning (sigmoid_features, binary_codes).

    IDs refer to original source rows. The caller chooses an already fixed pool;
    this function does no sampling, fitting, labeling or metric calculation.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    ids = np.asarray(sample_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= len(maps))):
        raise ValueError("sample_ids must be original source row IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if settings is None:
        settings = model.settings
    if parameters(settings) != model.settings:
        raise ValueError("Encoding preprocessing must match model settings")
    device = next(model.parameters()).device if device is None else torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    result = np.empty((len(ids), CODE_BITS), dtype=np.float32)
    try:
        with torch.inference_mode():
            for start in range(0, len(ids), batch_size):
                inputs = torch.stack([preprocess_wafer(maps[int(i)], settings)
                                      for i in ids[start:start + batch_size]]).to(device)
                result[start:start + len(inputs)] = model.features(inputs).cpu().numpy()
    finally:
        model.train(was_training)
    return result, binary_codes(result)


def encode_valid(bundle, config):
    with file_lock(Path(config["output_dir"]) / ".encoding.lock"):
        return _encode_valid(bundle, config)


def _encode_valid(bundle, config):
    pool = make_retrieval_pool(bundle, config, "valid")
    output = Path(config["output_dir"])
    checkpoint = output / "checkpoint_last.pt"
    # Check completion and provenance before accepting/reusing encoded results.
    model, device = load_trained_model(bundle, config)
    directory = output / "valid_codes"
    fingerprint = signature({"run": _run_signature(config), "data": bundle.signature,
                             "checkpoint": hash_file(checkpoint), "sample_ids": pool.gallery})
    marker = directory / "metadata.json"
    if marker.exists():
        record = read_json(marker)
        if record.get("signature") != fingerprint:
            raise ValueError("Encoded feature cache belongs to another model/data/protocol")
        for name, digest in record["files"].items():
            if hash_file(directory / name) != digest:
                raise ValueError("Encoded feature cache is missing or corrupt")
        return record
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Incomplete existing code cache; preserve it and use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    features, codes = encode_maps(model, bundle.maps, pool.gallery,
                                  batch_size=int(config.get("inference_batch_size", 8)), device=device)
    arrays = {"sample_ids.npy": pool.gallery, "sigmoid_features.npy": features, "binary_codes.npy": codes}
    for name, values in arrays.items():
        path = directory / name
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
    record = {"signature": fingerprint, "split": "valid", "sample_count": len(pool.gallery),
              "code_bits": CODE_BITS, "threshold": ">= 0.5", "distance": "integer Hamming",
              "row_identity": "sample_ids.npy stores original NPZ row IDs",
              "query_ids": pool.queries, "candidate_rule": "all_gallery_except_query",
              "protocol_signature": pool.protocol_signature, "checkpoint_sha256": hash_file(checkpoint),
              "files": {name: hash_file(directory / name) for name in arrays},
              "smoke": bool(config.get("smoke")), "retrieval_metrics_defined": False,
              "test_processed": False}
    save_json(marker, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--stage", choices=("audit", "prepare", "train", "encode", "all"), default="audit")
    parser.add_argument("--output-dir", help="Override output directory, relative to compared_methods")
    parser.add_argument("--device", help="cpu, cuda, cuda:N, mps, or auto")
    parser.add_argument("--resume", action="store_true", help="Resume the last completed epoch of this exact run")
    args = parser.parse_args(argv)
    config = load_config(args.config, {"output_dir": args.output_dir, "device": args.device}, require_ks=False)
    if config["method"] != "nakazawa":
        raise ValueError("Expected method=nakazawa")
    parameters(config.get("parameters"))
    training_parameters(config.get("training"), smoke=bool(config.get("smoke")))
    if int(config.get("num_workers", 0)) < 0 or int(config.get("torch_threads", 4)) < 1:
        raise ValueError("num_workers must be nonnegative and torch_threads positive")
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    ensure_run(config, [Path(__file__)])
    bundle = prepare_data(config)
    result = {"audit": audit(bundle, config)}
    if args.stage in ("prepare", "train", "encode", "all"):
        result["prepare"] = prepare(bundle, config)
    if args.stage in ("train", "all"):
        result["training"] = train(bundle, config, resume=args.resume)
    if args.stage in ("encode", "all"):
        result["encoding"] = encode_valid(bundle, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

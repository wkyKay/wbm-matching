"""Yu (2019): two CNNs, train-fitted PCA, ten-valued categorical retrieval.

Paper-specified choices and explicit reproduction choices are documented in README.md.
This module never reads test samples; the shared protocol owns split validation.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F


METHOD_VERSION = "yu2019-mixedwm38-v1"
PALETTE = np.asarray([[255, 255, 255], [0, 255, 255], [255, 0, 255]], dtype=np.uint8)


def _conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), nn.ReLU())


class DetectionCNN(nn.Module):
    """Fig. 3: 224 -> 74 -> 24 -> 8; two fully connected layers."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv(3, 32), nn.MaxPool2d(3, 3), _conv(32, 64), nn.MaxPool2d(3, 3),
            _conv(64, 64), nn.MaxPool2d(3, 3))
        self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.fc2(self.dropout(torch.sigmoid(self.fc1(h))))


class ClassificationCNN(nn.Module):
    """Fig. 4 with the final eight-class head replaced by 37 combinations."""

    def __init__(self, num_classes=37):
        super().__init__()
        if num_classes != 37:
            raise ValueError("This protocol preserves exactly 37 non-Normal combination classes")
        self.features = nn.Sequential(
            _conv(3, 32), _conv(32, 32), nn.MaxPool2d(3, 3),
            _conv(32, 64), _conv(64, 64), nn.MaxPool2d(3, 3),
            _conv(64, 128), _conv(128, 128), _conv(128, 128), nn.MaxPool2d(3, 3))
        self.fc1 = nn.Linear(8 * 8 * 128, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward_features(self, x):
        h = self.features(x).flatten(1)
        h = self.dropout1(torch.sigmoid(self.fc1(h)))
        return torch.sigmoid(self.fc2(h))

    def forward(self, x):
        return self.fc3(self.dropout2(self.forward_features(x)))


def filtered_map(raw_map):
    """Filter the original grid before any resizing, preserving valid die support."""
    raw = np.asarray(raw_map)
    if raw.ndim != 2 or raw.size == 0 or not np.isin(raw, [0, 1, 2, 3]).all():
        raise ValueError("Expected a nonempty 2-D map with values 0/1/2/3")
    raw = np.where(raw == 3, 0, raw)  # Shared source-project policy: code 3 is not a valid die.
    wafer = raw != 0
    filtered = median_filter(raw.astype(np.uint8), size=9, mode="nearest")
    return np.where(wafer, np.maximum(filtered, 1), 0).astype(np.uint8)


def preprocess(raw_map, *, training=False, rng=None, crop_scale=(0.95, 1.0), rotation=True):
    """Median-9 on original categorical grid, resize, RGB, then [-1, 1].

0/3 = outside/ignored, 1 = passing die, 2 = failing die. Nearest-neighbor
resizing/rotation preserve the palette. Crop bounds are reproduction choices.
"""
    filtered = filtered_map(raw_map)
    im = Image.fromarray(filtered).resize((224, 224), Image.Resampling.NEAREST)
    if training:
        rng = rng if rng is not None else np.random.default_rng()
        if rotation:
            im = im.rotate(float(rng.uniform(0, 360)), resample=Image.Resampling.NEAREST, fillcolor=0)
        low, high = crop_scale
        if not (0 < low <= high <= 1):
            raise ValueError("crop_scale must be ordered in (0,1]")
        side = max(1, int(round(224 * math.sqrt(float(rng.uniform(low, high))))))
        left, top = rng.integers(0, 225 - side, size=2)
        im = im.crop((int(left), int(top), int(left + side), int(top + side)))
        im = im.resize((224, 224), Image.Resampling.NEAREST)
    rgb = PALETTE[np.asarray(im, dtype=np.uint8)].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray((rgb / 127.5 - 1).transpose(2, 0, 1)))


class TenValueEncoder:
    """PCA/min-max fitted only on defect training Fc2 features."""

    def __init__(self, dimensions=256, seed=0):
        if dimensions != 256:
            raise ValueError("Yu retrieval must retain 256 PCA dimensions")
        self.dimensions = dimensions
        self.seed = seed
        self.pca = PCA(n_components=dimensions, whiten=False, svd_solver="randomized", random_state=seed)
        self.minimum = None
        self.maximum = None
        self.fit_ids = None

    def fit(self, features, sample_ids):
        features = np.asarray(features, dtype=np.float64)
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
        if features.ndim != 2 or features.shape[1] != 1024 or len(features) < 257:
            raise ValueError("PCA256 requires at least 257 real train rows of 1024-D Fc2 features")
        if len(sample_ids) != len(features) or len(np.unique(sample_ids)) != len(sample_ids):
            raise ValueError("PCA fitting IDs must be unique and match the feature rows")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite Fc2 training features")
        self.pca.fit(features)
        projected = self.pca.transform(features)
        self.minimum, self.maximum = projected.min(0), projected.max(0)
        self.fit_ids = sample_ids.copy()
        return self

    def transform(self, features):
        if self.minimum is None:
            raise RuntimeError("Fit/load the train-only PCA encoder first")
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != 1024 or not np.isfinite(features).all():
            raise ValueError("Expected finite N x 1024 Fc2 features")
        if len(features) == 0:
            return np.empty((0, 256), dtype=np.uint8)
        projected = self.pca.transform(features)
        span = self.maximum - self.minimum
        normalized = np.zeros_like(projected)
        np.divide(projected - self.minimum, span, out=normalized, where=span > 0)
        normalized = np.clip(normalized, 0, 1)
        return np.minimum(np.floor(10 * normalized), 9).astype(np.uint8)

    def save(self, path, signature):
        if self.fit_ids is None:
            raise RuntimeError("Cannot save unfitted PCA")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, components=self.pca.components_, mean=self.pca.mean_,
                                minimum=self.minimum, maximum=self.maximum, fit_ids=self.fit_ids,
                                signature=np.asarray(signature), seed=np.asarray(self.seed))

    @classmethod
    def load(cls, path, signature):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["signature"]) != signature:
                raise ValueError("PCA signature differs from the current configuration/protocol")
            encoder = cls(seed=int(archive["seed"]))
            encoder.pca.components_ = archive["components"].copy()
            encoder.pca.mean_ = archive["mean"].copy()
            encoder.pca.n_features_in_ = 1024
            encoder.pca.n_components_ = 256
            encoder.minimum = archive["minimum"].copy()
            encoder.maximum = archive["maximum"].copy()
            encoder.fit_ids = archive["fit_ids"].copy()
        return encoder


def categorical_hamming(query_code, candidate_codes):
    q, c = np.asarray(query_code), np.asarray(candidate_codes)
    if q.shape != (256,) or c.ndim != 2 or c.shape[1] != 256:
        raise ValueError("Expected one 256-D query and N x 256 candidate codes")
    if not (np.isin(q, np.arange(10)).all() and np.isin(c, np.arange(10)).all()):
        raise ValueError("Ten-value codes must contain only integers 0 through 9")
    return np.count_nonzero(c != q, axis=1)


def explicit_l2(model):
    """Paper Eq.(4): sum of weight squares, excluding biases by stated convention."""
    return sum(parameter.square().sum() for name, parameter in model.named_parameters()
               if parameter.requires_grad and name.endswith("weight"))


def preprocess_audit(bundle, output_dir, run_signature):
    """Record filter effects on allowed train/valid records without adapting the filter."""
    from common import read_json, save_json, save_csv

    folder = Path(output_dir) / "preprocessing"
    marker = folder / "summary.json"
    if marker.exists() and read_json(marker).get("signature") == run_signature:
        return read_json(marker)
    rows, groups = [], {}
    sets = {"defect_train": bundle.train_ids, "defect_valid": bundle.valid_ids,
            "normal_train": bundle.normal_train_ids, "normal_valid": bundle.normal_valid_ids}
    for group, ids in sets.items():
        stat = {"maps": len(ids), "original_defect_dies": 0, "filtered_defect_dies": 0,
                "original_zero_defect_maps": 0, "filtered_zero_defect_maps": 0,
                "nonempty_to_zero_defect_maps": 0, "zero_valid_die_maps": 0, "ignored_code3_dies": 0}
        for sample_id in ids:
            raw = np.asarray(bundle.maps[int(sample_id)])
            filtered = filtered_map(raw)
            before, after = int(np.count_nonzero(raw == 2)), int(np.count_nonzero(filtered == 2))
            valid_dies = int(np.count_nonzero(np.isin(raw, [1, 2])))
            ignored = int(np.count_nonzero(raw == 3))
            rows.append({"sample_id": int(sample_id), "group": group,
                         "combination_class": int(bundle.class_ids[sample_id]),
                         "valid_dies": valid_dies, "ignored_code3_dies": ignored,
                         "original_defect_dies": before, "filtered_defect_dies": after,
                         "defect_retention_ratio": after / before if before else None,
                         "filtered_zero_defects": after == 0})
            stat["original_defect_dies"] += before
            stat["filtered_defect_dies"] += after
            stat["original_zero_defect_maps"] += int(before == 0)
            stat["filtered_zero_defect_maps"] += int(after == 0)
            stat["nonempty_to_zero_defect_maps"] += int(before > 0 and after == 0)
            stat["zero_valid_die_maps"] += int(valid_dies == 0)
            stat["ignored_code3_dies"] += ignored
        stat["aggregate_defect_retention_ratio"] = (stat["filtered_defect_dies"] / stat["original_defect_dies"]
                                                    if stat["original_defect_dies"] else None)
        groups[group] = stat
    summary = {"signature": run_signature, "order": "original grid median9 -> nearest224 -> RGB",
               "groups": groups, "test_processed": False, "holdout_processed": False,
               "interpretation": "Fixed-filter diagnostics, not retrieval performance or a tuning criterion"}
    if rows:
        save_csv(folder / "per_sample.csv", rows, list(rows[0]))
    save_json(marker, summary)
    return summary


class WaferDataset(torch.utils.data.Dataset):
    def __init__(self, bundle, sample_ids, *, task="classifier", training=False, epoch=0, seed=0,
                 crop_scale=(0.95, 1.0), rotation=True):
        self.bundle, self.ids, self.task = bundle, np.asarray(sample_ids, dtype=np.int64), task
        self.training, self.epoch, self.seed = training, epoch, seed
        self.crop_scale, self.rotation = crop_scale, rotation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = int(self.ids[index])
        combination = int(self.bundle.class_ids[sample_id])
        target = int(combination >= 0) if self.task == "detector" else combination
        if self.task == "classifier" and target < 0:
            raise ValueError("Normal cannot enter the 37-class classifier training/evaluation")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, sample_id]))
        image = preprocess(self.bundle.maps[sample_id], training=self.training, rng=rng,
                           crop_scale=self.crop_scale, rotation=self.rotation)
        return sample_id, image, target


def _loader(dataset, batch_size, config, *, epoch=0):
    # This private generator prevents iterator creation from advancing dropout's RNG.
    generator = torch.Generator().manual_seed(int(config["seed"]) + int(epoch))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=int(config.get("num_workers", 0)),
                                      pin_memory=str(config.get("device", "cpu")).startswith("cuda"),
                                      generator=generator, drop_last=False)


def _capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; cannot restore all CUDA RNG states faithfully")
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path):
    # These are our own local checkpoints and include Python/NumPy RNG state.
    return torch.load(path, map_location="cpu", weights_only=False)


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _model(task):
    return DetectionCNN() if task == "detector" else ClassificationCNN()


def _task_ids(bundle, task, split):
    if split not in ("train", "valid"):
        raise ValueError("Only train/valid are permitted")
    defects = getattr(bundle, split + "_ids")
    return (np.concatenate([defects, getattr(bundle, "normal_" + split + "_ids")])
            if task == "detector" else np.asarray(defects))


def _classification_metrics(targets, predictions, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    correct, support, predicted = confusion.diagonal(), confusion.sum(1), confusion.sum(0)
    precision = np.divide(correct, predicted, out=np.zeros(num_classes, dtype=float), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(num_classes, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(num_classes, dtype=float), where=precision + recall > 0)
    return {"accuracy": float(correct.sum() / max(1, confusion.sum())), "macro_f1": float(f1.mean()),
            "balanced_accuracy": float(recall[support > 0].mean()) if np.any(support) else 0.,
            "samples": int(confusion.sum()), "confusion_matrix": confusion.tolist(),
            "per_class_support": support.tolist(), "per_class_recall": recall.tolist()}


@torch.inference_mode()
def predict(model, bundle, sample_ids, config, device, *, task, features=False):
    model.eval()
    ids, targets, predictions, embeddings = [], [], [], []
    total_loss, count = 0., 0
    dataset = WaferDataset(bundle, sample_ids, task=task)
    for sample_id, images, target in _loader(dataset, config["inference_batch_size"], config):
        images = images.to(device)
        if features:
            h = model.forward_features(images)
            logits = model.fc3(h)
            embeddings.append(h.cpu().numpy())
        else:
            logits = model(images)
        total_loss += F.cross_entropy(logits, target.to(device), reduction="sum").item()
        count += len(target)
        ids.extend(sample_id.tolist())
        targets.extend(target.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = _classification_metrics(np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int),
                                      2 if task == "detector" else 37)
    metrics["ce"] = total_loss / count if count else None
    return {"ids": np.asarray(ids, dtype=np.int64), "targets": np.asarray(targets),
            "predictions": np.asarray(predictions), "metrics": metrics,
            "features": np.concatenate(embeddings) if embeddings else np.empty((0, 1024), dtype=np.float32)}


def train_network(task, bundle, config, output_dir, run_signature, device, *, resume=False, max_steps=None):
    from common import save_json, seed_everything

    task_seed = int(config["seed"]) + (0 if task == "detector" else 10000)
    seed_everything(task_seed)
    model = _model(task).to(device)
    options = config["training"][task]
    batch_size, epochs = int(options["batch_size"]), int(options["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
    folder = Path(output_dir) / task
    folder.mkdir(parents=True, exist_ok=True)
    last_path, best_path = folder / "last.pt", folder / "best.pt"
    state = {"epoch": 0, "next_batch": 0, "steps": 0, "best_valid_ce": float("inf"),
             "history": [], "epoch_loss_sum": 0., "epoch_samples": 0, "truncated_epoch": False}
    if last_path.exists():
        if not resume:
            raise FileExistsError(f"{last_path} exists; use --resume or a new output directory")
        checkpoint = _load_checkpoint(last_path)
        if checkpoint["signature"] != run_signature or checkpoint["task"] != task:
            raise ValueError("Checkpoint configuration/protocol/method signature mismatch")
        if checkpoint["torch_version"] != str(torch.__version__):
            raise ValueError("PyTorch version changed; deterministic resume requires the original version")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to(optimizer, device)
        state = checkpoint["state"]
        _restore_rng(checkpoint["rng"])
        del checkpoint
    elif resume and best_path.exists():
        raise ValueError("Cannot resume without the matching last.pt checkpoint")

    train_ids = _task_ids(bundle, task, "train")
    valid_ids = _task_ids(bundle, task, "valid")
    if len(train_ids) == 0 or len(valid_ids) == 0:
        raise ValueError("Both train and valid must be nonempty")
    l2_coefficient = float(config["training"]["classifier_l2"]) if task == "classifier" else 0.

    def save_checkpoint(best=False):
        payload = {"signature": run_signature, "task": task, "method_version": METHOD_VERSION,
                   "config": config, "torch_version": str(torch.__version__), "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "state": state, "rng": _capture_rng()}
        _atomic_torch_save(last_path, payload)
        if best:
            _atomic_torch_save(best_path, payload)

    while state["epoch"] < epochs and (max_steps is None or state["steps"] < max_steps):
        started = time.monotonic()
        epoch = int(state["epoch"])
        order = np.random.default_rng(task_seed + epoch).permutation(train_ids)
        total_batches = math.ceil(len(order) / batch_size)
        start_batch = int(state["next_batch"])
        remaining = order[start_batch * batch_size:]
        dataset = WaferDataset(bundle, remaining, task=task, training=True, epoch=epoch, seed=task_seed,
                               crop_scale=config["augmentation"]["crop_scale"],
                               rotation=config["augmentation"]["rotation"])
        model.train()
        for offset, (_, images, targets) in enumerate(_loader(dataset, batch_size, config, epoch=epoch)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = F.cross_entropy(logits, targets.to(device))
            if l2_coefficient:
                loss = loss + l2_coefficient * explicit_l2(model)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {task} loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            state["steps"] += 1
            state["next_batch"] = start_batch + offset + 1
            state["epoch_loss_sum"] += loss.item() * len(targets)
            state["epoch_samples"] += len(targets)
            state["truncated_epoch"] = state["next_batch"] < total_batches
            if state["steps"] % int(config["training"]["checkpoint_every_steps"]) == 0:
                save_checkpoint()
            if max_steps is not None and state["steps"] >= max_steps:
                break
        complete = state["next_batch"] == total_batches
        result = predict(model, bundle, valid_ids, config, device, task=task)
        valid_ce = result["metrics"]["ce"]
        best = valid_ce < state["best_valid_ce"]
        if best:
            state["best_valid_ce"] = valid_ce
        state["history"].append({"epoch": epoch + 1, "epoch_complete": complete,
                                 "steps": state["steps"], "next_batch": state["next_batch"],
                                 "train_loss": state["epoch_loss_sum"] / max(1, state["epoch_samples"]),
                                 "valid": result["metrics"], "seconds": time.monotonic() - started})
        if complete:
            state["epoch"] += 1
            state["next_batch"] = 0
            state["epoch_loss_sum"], state["epoch_samples"] = 0., 0
        state["truncated_epoch"] = not complete
        save_checkpoint(best=best)
        save_json(folder / "history.json", state["history"])
        print(f"{task}: epoch={epoch + 1}, complete={complete}, steps={state['steps']}, "
              f"valid_CE={valid_ce:.6f}, valid_macro_F1={result['metrics']['macro_f1']:.6f}", flush=True)
    if not best_path.exists():
        # A process may have stopped just after a periodic checkpoint at its step ceiling.
        result = predict(model, bundle, valid_ids, config, device, task=task)
        state["best_valid_ce"] = result["metrics"]["ce"]
        save_checkpoint(best=True)
    summary = {"task": task, "steps": state["steps"], "completed_epochs": state["epoch"],
               "next_batch": state["next_batch"], "truncated_epoch": state["truncated_epoch"],
               "training_complete": state["epoch"] >= epochs, "configured_epochs": epochs,
               "max_steps_this_invocation": max_steps, "best_valid_ce": state["best_valid_ce"],
               "train_samples": len(train_ids), "valid_samples": len(valid_ids),
               "selection": "minimum valid CE; exact ties retain earlier checkpoint",
               "l2": l2_coefficient, "batch_size": batch_size, "signature": run_signature}
    save_json(folder / "summary.json", summary)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _selected_model(task, output_dir, run_signature, device):
    path = Path(output_dir) / task / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Train {task} before validation: {path}")
    checkpoint = _load_checkpoint(path)
    if checkpoint["signature"] != run_signature or checkpoint["task"] != task:
        raise ValueError("Selected checkpoint signature mismatch")
    model = _model(task).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = {"checkpoint": str(path), "steps": checkpoint["state"]["steps"],
                "completed_epochs": checkpoint["state"]["epoch"],
                "truncated_epoch": checkpoint["state"]["truncated_epoch"]}
    del checkpoint
    return model, metadata


def validate(bundle, config, output_dir, run_signature, device):
    from common import make_valid_pool, evaluate_rankings, save_json, save_csv, hash_file, signature

    folder = Path(output_dir) / "valid"
    folder.mkdir(parents=True, exist_ok=True)
    pool = make_valid_pool(bundle, config)
    pool = {int(q): [int(c) for c in cs] for q, cs in pool.items()}
    model, classifier_provenance = _selected_model("classifier", output_dir, run_signature, device)
    encoder_signature = signature({"run": run_signature, "classifier": hash_file(Path(output_dir) / "classifier/best.pt")})
    encoder_path, encoder = Path(output_dir) / "encoder.npz", None
    if encoder_path.exists():
        try:
            cached = TenValueEncoder.load(encoder_path, encoder_signature)
            if np.array_equal(cached.fit_ids, np.asarray(bundle.train_ids)):
                encoder = cached
        except ValueError:
            pass  # A newer selected classifier requires refitting on train, never valid.
    encoder_reused = encoder is not None
    if encoder is None:
        train = predict(model, bundle, bundle.train_ids, config, device, task="classifier", features=True)
        encoder = TenValueEncoder(seed=int(config["seed"])).fit(train["features"], train["ids"])
        encoder.save(encoder_path, encoder_signature)
        del train
    valid = predict(model, bundle, bundle.valid_ids, config, device, task="classifier", features=True)
    codes = encoder.transform(valid["features"])
    indices = {int(sample_id): i for i, sample_id in enumerate(valid["ids"])}
    predicted_class = {int(sample_id): int(p) for sample_id, p in zip(valid["ids"], valid["predictions"])}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    detector, detector_provenance = _selected_model("detector", output_dir, run_signature, device)
    detected = predict(detector, bundle, _task_ids(bundle, "detector", "valid"), config, device, task="detector")
    predicted_defect = {int(sample_id): bool(p) for sample_id, p in zip(detected["ids"], detected["predictions"])}
    del detector
    rankings, gated, gate_counts = {}, {}, []
    global_rows, gated_rows = [], []
    for query_id, candidate_ids in pool.items():
        if query_id not in indices or any(c not in indices for c in candidate_ids):
            raise ValueError("Validation retrieval must use only explicit defect valid IDs")
        if query_id in candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Candidate pool contains self or duplicate records")
        distances = categorical_hamming(codes[indices[query_id]], codes[[indices[c] for c in candidate_ids]])
        ordered = np.asarray(candidate_ids, dtype=np.int64)[np.lexsort((candidate_ids, distances))].tolist()
        rankings[query_id] = ordered
        # Both sides use predictions, never valid truth labels, for class gating.
        gated[query_id] = ([c for c in ordered if predicted_defect[c]
                            and predicted_class[c] == predicted_class[query_id]]
                           if predicted_defect[query_id] else [])
        distance_by_id = dict(zip(candidate_ids, distances.tolist()))
        for target_rows, ranking in ((global_rows, ordered), (gated_rows, gated[query_id])):
            for rank, candidate_id in enumerate(ranking, 1):
                target_rows.append({"query_id": query_id, "rank": rank, "candidate_id": candidate_id,
                                    "categorical_hamming": int(distance_by_id[candidate_id]),
                                    "query_predicted_class": predicted_class[query_id],
                                    "candidate_predicted_class": predicted_class[candidate_id]})
        gate_counts.append({"query_id": query_id, "input_candidates": len(ordered),
                            "remaining_candidates": len(gated[query_id]),
                            "detector_rejected_query": not predicted_defect[query_id]})
    results = {
        "global": evaluate_rankings(bundle, pool, rankings, config["ks"], folder / "global"),
        "predicted_class": evaluate_rankings(bundle, pool, gated, config["ks"], folder / "predicted_class")}
    fields = ["query_id", "rank", "candidate_id", "categorical_hamming", "query_predicted_class",
              "candidate_predicted_class"]
    save_csv(folder / "global/rankings.csv", global_rows, fields)
    save_csv(folder / "predicted_class/rankings.csv", gated_rows, fields)
    save_json(folder / "classification.json", {"classifier": valid["metrics"], "detector": detected["metrics"]})
    save_json(folder / "gate_counts.json", gate_counts)
    save_json(folder / "provenance.json", {"signature": run_signature, "classifier": classifier_provenance,
              "detector": detector_provenance, "pca_fit_split": "train", "pca_rows": len(bundle.train_ids),
              "pca_dimensions": 256, "pca_whiten": False, "pca_cache_reused": encoder_reused,
              "scaling": "train per-dimension min-max; clip [0,1]",
              "distance": "categorical Hamming count", "tie_break": "sample_id ascending",
              "main_result": "global (no detection/classification gate)",
              "auxiliary": "predicted-class cascade; both query and candidates gated by model predictions",
              "labels_used_for_ranking": False, "normal_holdout_used": False, "test_used": False})
    save_json(folder / "summary.json", results)
    np.savez_compressed(folder / "features_codes.npz", ids=valid["ids"], codes=codes,
                        predicted_classes=valid["predictions"])
    return results


def main(argv=None):
    import argparse
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from common import load_config, ensure_run, prepare_data, save_json, get_device, signature, seed_everything

    parser = argparse.ArgumentParser(description="Yu 2019 MixedWM38; train/valid only")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--stage", choices=["audit", "prepare", "train", "valid", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Total optimizer-step ceiling per network; resume continues toward this ceiling")
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in {"device": args.device, "output_dir": args.output_dir}.items()
                 if value is not None}
    config = load_config(args.config, overrides=overrides)
    if config.get("method") != "yu":
        raise ValueError("Expected method=yu")
    ceiling = args.max_steps if args.max_steps is not None else config["training"].get("max_steps")
    if ceiling is not None and ceiling < 1:
        raise ValueError("--max-steps must be positive")
    if config.get("smoke", False) and (ceiling is None or ceiling > 8):
        raise ValueError("Smoke is capped at eight optimizer steps per complete network")
    if int(config.get("torch_threads", 0)) > 0:
        torch.set_num_threads(int(config["torch_threads"]))
    base_signature = ensure_run(config, [Path(__file__)])
    bundle = prepare_data(config)
    run_signature = signature({"run": base_signature, "data": bundle.signature, "method": METHOD_VERSION})
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "yu_protocol.json", {"signature": run_signature, "bundle": bundle.summary(),
               "method": METHOD_VERSION, "train_labels": "all 37 complete combination classes",
               "normal": "only separate detector train/valid; holdout remains unused"})
    preprocess_audit(bundle, output_dir, run_signature)
    if args.stage in ("audit", "prepare"):
        print(json.dumps(bundle.summary(), ensure_ascii=False, indent=2))
        return
    seed_everything(int(config["seed"]))
    device = get_device(config.get("device", "auto"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This implementation supports explicit CPU or CUDA execution")
    if args.stage in ("train", "all"):
        for task in ("detector", "classifier"):
            train_network(task, bundle, config, output_dir, run_signature, device,
                          resume=args.resume, max_steps=ceiling)
    if args.stage in ("valid", "all"):
        validate(bundle, config, output_dir, run_signature, device)


if __name__ == "__main__":
    main()

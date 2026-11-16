"""F-aligned whole-wafer WaPIRL adaptation; no region extraction.

Model/loss/optimization/checkpoint primitives frozen from the current F sources.
See README.md and f_reference.json for provenance and deliberate differences.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import math
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.common import (load_config, ensure_run, prepare_data, seed_everything,
    get_device, save_json, read_json, signature, hash_file, file_lock)

METHOD_VERSION = "wapirl-f-aligned-whole-v1"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, image):
        value = F.relu(self.bn1(self.conv1(image)), inplace=True)
        value = self.bn2(self.conv2(value))
        return F.relu(value + self.shortcut(image), inplace=True)


class Encoder(nn.Module):
    """WaPIRL's [2,2,2,2] backbone followed by GAP, without a learned bottleneck.

    Return raw 512-D features. Cosine normalization belongs to the loss or
    downstream scoring, so it does not change the original memory-bank EMA.
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        if embedding_dim != 512:
            raise ValueError("The WaPIRL-aligned encoder has a fixed 512-D GAP output")
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        stages, previous = [], 64
        for channels, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            stages.append(nn.Sequential(BasicBlock(previous, channels, stride), BasicBlock(channels, channels)))
            previous = channels
        self.stages = nn.Sequential(*stages)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                # Match WaPIRL/utils/initialization.py, including its BN bias.
                nn.init.ones_(module.weight)
                nn.init.ones_(module.bias)

    def forward(self, image):
        value = self.stages(self.stem(image))
        return F.adaptive_avg_pool2d(value, 1).flatten(1)


class Projector(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.net = nn.Linear(embedding_dim, projection_dim)
        nn.init.normal_(self.net.weight, 0, 0.02)
        nn.init.zeros_(self.net.bias)

    def forward(self, embedding):
        return self.net(embedding)


def build_model(config, spec):
    # Construct shared layers in exactly the A-D RNG order before adapting F's stem.
    encoder = Encoder(config["embedding_dim"])
    encoder.stem[0].stride = (spec["stem_stride"],) * 2
    projector = Projector(config["embedding_dim"], config["local"]["projector_dim"])
    if spec["input_channels"] == 3:
        with torch.random.fork_rng(devices=[]):
            # The replacement is initialized on CPU; do not reseed CUDA/MPS sampling.
            torch.random.default_generator.manual_seed(config["seed"] + 52)
            conv = nn.Conv2d(3, 64, 7, spec["stem_stride"], 3, bias=False)
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        encoder.stem[0] = conv
    return nn.ModuleDict({"encoder": encoder, "projector": projector})


class MemoryBank:
    def __init__(self, count, dim, device, momentum=0.5):
        self.buffer = torch.zeros(count, dim, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def initialize(self, encoder, projector, data_loader, device):
        encoder.eval()
        projector.eval()
        for indices, image, _, _info in data_loader:
            self.buffer[indices.to(device)] = projector(encoder(image.to(device)))

    def negative_indices(self, excluded, count):
        probabilities = torch.ones(len(self.buffer), device=self.buffer.device)
        probabilities[excluded] = 0
        available = int((probabilities > 0).sum())
        if available == 0:
            raise ValueError("No negatives left outside the current batch")
        return torch.multinomial(probabilities, count, replacement=True)

    @torch.no_grad()
    def update(self, indices, values):
        self.buffer[indices] = self.momentum * self.buffer[indices] + (1 - self.momentum) * values


def nce_loss(anchors, positives, negatives, temperature):
    """Each row contrasts one positive against a shared, detached bank sample."""
    positives, anchors = F.normalize(positives, dim=1), F.normalize(anchors.detach(), dim=1)
    negatives = F.normalize(negatives.detach(), dim=1)
    positive = (anchors * positives).sum(1, keepdim=True)
    logits = torch.cat([positive, positives @ negatives.T], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def optimization(model, settings):
    """Use the same WaPIRL SGD/cosine recipe for both learned methods."""
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"],
                                momentum=settings["momentum"], weight_decay=settings["weight_decay"])
    # WaPIRL's zero-warmup schedule uses this multiplier, including its floor.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1e-4 + max(0., 0.5 * (1 + math.cos(math.pi * epoch / settings["epochs"]))))
    return optimizer, scheduler


def loader(dataset, config, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle,
                      num_workers=config["num_workers"], pin_memory=config["device"].startswith("cuda"),
                      generator=generator, persistent_workers=False)


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints generated by this project from a trusted local run.
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, history, signature, **extra):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "epoch": epoch, "best_loss": best_loss, "history": history, "signature": signature,
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    if next(model.parameters()).device.type == "mps":
        payload["mps_rng"] = torch.mps.get_rng_state()
    payload.update(extra)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, model, optimizer, scheduler, signature, resume):
    if not path.exists():
        return 0, float("inf"), [], None
    if not resume:
        raise FileExistsError(f"Existing checkpoint {path}; pass --resume or choose a new output_dir")
    checkpoint = load_checkpoint(path)
    if checkpoint["signature"] != signature:
        raise ValueError("Checkpoint/config mismatch; choose a new output_dir")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if torch.cuda.is_available() and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if "mps_rng" in checkpoint and next(model.parameters()).device.type == "mps":
        torch.mps.set_rng_state(checkpoint["mps_rng"])
    return checkpoint["epoch"], checkpoint["best_loss"], checkpoint["history"], checkpoint


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            # Non-capturable AdamW keeps its scalar step counter on CPU.
            if torch.is_tensor(value) and key != "step":
                state[key] = value.to(device)


@contextmanager
def validation_rng(device, seed):
    """Keep validation sampling deterministic without altering future MPS draws."""
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def resize_nearest(array, size):
    h, w = array.shape
    rr = np.minimum((np.arange(size) * h / size).astype(int), h - 1)
    cc = np.minimum((np.arange(size) * w / size).astype(int), w - 1)
    return array[np.ix_(rr, cc)]


def preprocess(raw, size=96):
    raw = np.asarray(raw)
    if raw.ndim != 2 or not all(raw.shape):
        raise ValueError('Expected a nonempty 2D wafer')
    defect = resize_nearest((raw == 2).astype(np.uint8), size)
    support = resize_nearest(((raw == 1) | (raw == 2)).astype(np.uint8), size)
    return np.stack([defect, defect, support]).astype(np.float32)


class WholeViews(Dataset):
    def __init__(self, maps, ids, seed, size=96):
        self.maps, self.ids, self.seed, self.size = maps, np.asarray(ids), seed, size
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        original = preprocess(self.maps[int(self.ids[index])], self.size)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        view = np.ascontiguousarray(np.rot90(original, int(rng.integers(4)), axes=(-2, -1)))
        return index, torch.from_numpy(original), torch.from_numpy(view), {}


def model_for(config):
    return build_model(config, config['parameters'])


def checkpoint_signature(bundle, run_signature):
    return signature({'run': run_signature, 'data': bundle.signature, 'method': METHOD_VERSION})


def train(bundle, config, run_signature, resume=False):
    seed_everything(config['seed'])
    device = get_device(config['device'])
    config = {**config, 'device': str(device)}
    settings = config['local']
    if any(v is not None for v in config.get('subset', {}).values()) and not config.get('smoke'):
        raise ValueError('Subsets require an explicit smoke config')
    if settings['epochs'] < 1 or config['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be positive')
    model = model_for(config).to(device)
    optimizer, scheduler = optimization(model, settings)
    output = Path(config['output_dir'])
    sig = checkpoint_signature(bundle, run_signature)
    start, best, history, checkpoint = restore(output/'last.pt', model, optimizer, scheduler, sig, resume)
    optimizer_to_device(optimizer, device)
    training = WholeViews(bundle.maps, bundle.train_ids, config['seed'], config['input_size'])
    valid = WholeViews(bundle.maps, bundle.valid_ids, config['seed']+1, config['input_size'])
    if len(training) < 2 or not len(valid):
        raise ValueError('Need >=2 train and >=1 valid images')
    bank = MemoryBank(len(training), settings['projector_dim'], device, settings['memory_momentum'])
    best_epoch = checkpoint['best_epoch'] if checkpoint else 0
    steps = checkpoint['steps'] if checkpoint else 0
    if checkpoint:
        bank.buffer.copy_(checkpoint['memory'].to(device))
    else:
        bank.initialize(model['encoder'], model['projector'], loader(training, config, config['seed'], False), device)
    del checkpoint
    for epoch in range(start, settings['epochs']):
        started = time.monotonic()
        training.epoch = epoch
        model.train()
        total, count = 0., 0
        for batch, (indices, original, view, _) in enumerate(loader(training, config, config['seed']+epoch, True), 1):
            indices = indices.to(device)
            z, zt = model['projector'](model['encoder'](torch.cat([original.to(device), view.to(device)]))).chunk(2)
            anchors = bank.buffer[indices].detach().clone()
            if len(indices) == len(training):
                losses = []
                for i, index in enumerate(indices):
                    negatives = bank.buffer[bank.negative_indices(index.reshape(1), settings['num_negatives'])]
                    losses.append(.5*(nce_loss(anchors[i:i+1], z[i:i+1], negatives, settings['temperature']) +
                        nce_loss(anchors[i:i+1], zt[i:i+1], negatives, settings['temperature'])))
                loss = torch.stack(losses).mean()
            else:
                negatives = bank.buffer[bank.negative_indices(indices, settings['num_negatives'])]
                loss = .5*(nce_loss(anchors, z, negatives, settings['temperature']) +
                           nce_loss(anchors, zt, negatives, settings['temperature']))
            value = loss.item()
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite training NCE')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bank.update(indices, z.detach())
            steps += 1
            total += value*len(indices)
            count += len(indices)
            if batch % 20 == 0:
                print(f'epoch={epoch+1} batch={batch} loss={value:.6f}', flush=True)
        model.eval()
        val_sum, val_count = 0., 0
        with torch.no_grad(), validation_rng(device, config['seed']+999):
            for _, original, view, _ in loader(valid, config, config['seed'], False):
                z, zt = model['projector'](model['encoder'](torch.cat([original.to(device), view.to(device)]))).chunk(2)
                chosen = torch.randint(len(bank.buffer), (settings['num_negatives'],), device=device)
                value = nce_loss(z, zt, bank.buffer[chosen], settings['temperature']).item()
                val_sum += value*len(original)
                val_count += len(original)
        val_loss = val_sum/val_count
        if not math.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation NCE')
        improved = val_loss < best
        if improved:
            best, best_epoch = val_loss, epoch+1
        history.append({'epoch': epoch+1, 'train_loss': total/count, 'valid_nce': val_loss,
            'train_samples': count, 'valid_samples': val_count, 'steps': steps,
            'learning_rate': optimizer.param_groups[0]['lr'], 'seconds': time.monotonic()-started})
        scheduler.step()
        extra = dict(memory=bank.buffer.detach().cpu(), best_epoch=best_epoch, steps=steps)
        if improved:
            save_checkpoint(output/'best_nce.pt', model, optimizer, scheduler, epoch+1, best, history, sig, **extra)
        save_checkpoint(output/'last.pt', model, optimizer, scheduler, epoch+1, best, history, sig, **extra)
        save_json(output/'history.json', history)
        print(f'epoch={epoch+1}/{settings["epochs"]} train={total/count:.6f} valid_nce={val_loss:.6f} best_epoch={best_epoch}', flush=True)
    save_json(output/'training_complete.json', {'signature': sig, 'epochs': settings['epochs'],
        'steps': steps, 'best_epoch': best_epoch, 'selection': 'minimum_valid_nce',
        'smoke': bool(config.get('smoke')), 'train_count': len(training), 'valid_count': len(valid),
        'last_sha256': hash_file(output/'last.pt'), 'best_sha256': hash_file(output/'best_nce.pt')})


def validate_source(bundle, config, run_signature):
    output = Path(config['output_dir'])
    record = read_json(output/'training_complete.json')
    expected = checkpoint_signature(bundle, run_signature)
    epochs = config['local']['epochs']
    steps = epochs*math.ceil(len(bundle.train_ids)/config['batch_size'])
    if (record.get('signature') != expected or record.get('epochs') != epochs or record.get('steps') != steps
        or record.get('selection') != 'minimum_valid_nce' or record.get('smoke') != bool(config.get('smoke'))
        or record.get('train_count') != len(bundle.train_ids) or record.get('valid_count') != len(bundle.valid_ids)):
        raise ValueError('WaPIRL training incomplete or provenance mismatch')
    for name, key in [('last.pt','last_sha256'), ('best_nce.pt','best_sha256')]:
        if hash_file(output/name) != record[key]:
            raise ValueError('WaPIRL checkpoint hash mismatch')
        ck = load_checkpoint(output/name)
        if ck['signature'] != expected or ck['steps'] != ck['epoch']*math.ceil(len(bundle.train_ids)/config['batch_size']):
            raise ValueError('WaPIRL checkpoint signature/progress mismatch')
        if ck['epoch'] != (epochs if name == 'last.pt' else record['best_epoch']):
            raise ValueError('WaPIRL checkpoint epoch mismatch')
        if ck['history'][-1]['train_samples'] != len(bundle.train_ids) or ck['history'][-1]['valid_samples'] != len(bundle.valid_ids):
            raise ValueError('WaPIRL incomplete epoch')
    return record


def encode_maps(bundle, config, ids, device):
    model = model_for(config).to(device)
    checkpoint = load_checkpoint(Path(config['output_dir'])/'best_nce.pt')
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.eval()
    data = WholeViews(bundle.maps, ids, config['seed'], config['input_size'])
    features = []
    with torch.inference_mode():
        for _, original, _, _ in loader(data, {**config, 'device':str(device), 'batch_size':config['inference_batch_size']}, config['seed'], False):
            features.append(F.normalize(model['encoder'](original.to(device)), dim=1).cpu().numpy())
    return np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--output-dir')
    parser.add_argument('--device')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, {'output_dir':args.output_dir,'device':args.device}, require_ks=False)
    torch.set_num_threads(config.get('torch_threads',4))
    run = ensure_run(config, [__file__])
    with file_lock(Path(config['output_dir'])/'.training.lock'):
        bundle = prepare_data(config)
        train(bundle, config, run, args.resume)


if __name__ == '__main__':
    main()

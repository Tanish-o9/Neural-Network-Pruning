"""Baseline training and fine-tuning (same loop for both)."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

import config as cfg


def seed_everything(seed):
    """Seed python, numpy and torch. (cuDNN is not forced to be deterministic, to keep it fast.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_resnet18(num_classes=10):
    """ResNet18 adapted to 32x32 images: 3x3 stem convolution, no max-pool."""
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Top-1 accuracy (%) in full FP32. The model must already be on `device`."""
    device = device or cfg.DEVICE
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def fit(model, train_loader, test_loader, epochs, lr, schedule, desc, log_test=False):
    """Train `model` in place.

    schedule: "onecycle" (baseline) or "cosine" (fine-tuning).
    Test accuracy is only logged for monitoring (log_test); it is never used to pick
    checkpoints or hyper-parameters.
    """
    device = cfg.DEVICE
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY, nesterov=True)
    total_steps = epochs * len(train_loader)
    if schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.25)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.USE_AMP)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"{desc} epoch {epoch}/{epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.USE_AMP):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
            bar.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{100 * correct / seen:.1f}")

        msg = f"[{desc}] epoch {epoch}/{epochs} | train loss {loss_sum / seen:.4f} | train acc {100 * correct / seen:.2f}%"
        if log_test and test_loader is not None:
            msg += f" | test acc {evaluate(model, test_loader):.2f}%"
        tqdm.write(msg)
    return model


def train_baseline(train_loader, test_loader, force=False):
    """Train the baseline ResNet18, or load it from the cached checkpoint."""
    ckpt = cfg.results_path(cfg.BASELINE_CKPT)
    model = build_resnet18(cfg.NUM_CLASSES)
    if os.path.exists(ckpt) and not force:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        print(f"Loaded cached baseline from {ckpt} (use --force-retrain to train again)")
        return model.to(cfg.DEVICE)

    seed_everything(cfg.SEED)
    fit(model, train_loader, test_loader, cfg.BASELINE_EPOCHS, cfg.BASELINE_MAX_LR,
        "onecycle", "baseline", log_test=True)
    torch.save(model.state_dict(), ckpt)
    print(f"Saved baseline to {ckpt}")
    return model


def fine_tune(model, train_loader, desc):
    """Fine-tune a pruned model. Identical budget (epochs, LR, seed) for every run."""
    seed_everything(cfg.SEED)
    return fit(model, train_loader, None, cfg.FT_EPOCHS, cfg.FT_LR, "cosine", desc)

"""CIFAR-10 dataloaders (with simple augmentation for training)."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config as cfg

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _seed_worker(worker_id):
    """Give every dataloader worker a reproducible seed."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _maybe_subset(dataset, n, seed):
    """Random subset of n items (n = 0 means keep everything)."""
    if not n or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


def get_dataloaders(seed=None):
    """Return (train_loader, test_loader).

    A fresh call with the same seed gives the same shuffling order, so every
    fine-tuning run sees exactly the same data in the same order.
    """
    seed = cfg.SEED if seed is None else seed

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.CIFAR10(cfg.DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(cfg.DATA_DIR, train=False, download=True, transform=test_tf)
    train_ds = _maybe_subset(train_ds, cfg.TRAIN_SUBSET, seed)
    test_ds = _maybe_subset(test_ds, cfg.TEST_SUBSET, seed)

    g = torch.Generator()
    g.manual_seed(seed)
    common = dict(num_workers=cfg.NUM_WORKERS, pin_memory=cfg.HAS_CUDA, worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              drop_last=True, generator=g, **common)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **common)
    return train_loader, test_loader

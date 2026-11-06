"""MNIST loading and the shared stratified test subsample.

Usage (from repo root):  python -m src.data [--config configs/default.yaml] [--force]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src import DEFAULT_CONFIG, ROOT, load_config

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def get_transform() -> transforms.Compose:
    """ToTensor (scales to [0, 1]) followed by MNIST mean/std normalization."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])


def get_datasets(root: Path) -> tuple[datasets.MNIST, datasets.MNIST]:
    """MNIST train (60k) and test (10k) sets, downloaded to root if missing."""
    tf = get_transform()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return train, test


def get_loaders(cfg: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, train_eval_loader, test_loader).

    The shuffled training loader owns a seeded generator, so batch order does not
    depend on global RNG consumed elsewhere (e.g. by evaluation loaders).
    """
    train, test = get_datasets(ROOT / cfg["data"]["root"])
    workers = cfg["data"]["num_workers"]
    batch_size = cfg["train"]["batch_size"]
    eval_batch_size = cfg["train"]["eval_batch_size"]

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=gen, num_workers=workers)
    train_eval_loader = DataLoader(train, batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch_size=eval_batch_size, shuffle=False, num_workers=workers)
    return train_loader, train_eval_loader, test_loader


def make_subsample(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Sorted indices of a class-proportional stratified subsample of the given labels."""
    idx, _ = train_test_split(np.arange(len(labels)), train_size=size, stratify=labels, random_state=seed)
    return np.sort(idx)


def load_or_create_subsample(cfg: dict, force: bool = False) -> np.ndarray:
    """Load outputs/subsample_idx.npy, creating it from the MNIST test set if missing or forced."""
    path = ROOT / cfg["paths"]["subsample_idx"]
    if path.exists() and not force:
        print(f"subsample: cached at {path}")
        return np.load(path)

    _, test = get_datasets(ROOT / cfg["data"]["root"])
    idx = make_subsample(test.targets.numpy(), cfg["data"]["subsample_size"], cfg["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, idx)
    print(f"subsample: saved {len(idx)} indices to {path}")
    return idx


def main(cfg: dict, force: bool = False) -> np.ndarray:
    """Data stage: download MNIST and create the shared test subsample."""
    return load_or_create_subsample(cfg, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

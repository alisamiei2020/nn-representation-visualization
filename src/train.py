"""Train SmallCNN, saving untrained and trained checkpoints plus their accuracies.

Usage (from repo root):  python -m src.train [--config configs/default.yaml] [--force]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import DEFAULT_CONFIG, ROOT, load_config
from src.data import get_loaders
from src.model import build_model

OPTIMIZERS = {"adam": torch.optim.Adam}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def output_paths(cfg: dict) -> dict[str, Path]:
    """Files produced by this stage."""
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    return {
        "untrained": ckpt_dir / "untrained.pt",
        "trained": ckpt_dir / "trained.pt",
        "metrics": ROOT / cfg["paths"]["metrics"] / "training.json",
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Classification accuracy of model over loader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def accuracies(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> dict:
    """Train and test accuracy over the full sets."""
    return {
        "train_acc": evaluate(model, train_loader, device),
        "test_acc": evaluate(model, test_loader, device),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over loader with cross-entropy loss; returns mean loss per sample."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        n += y.numel()
    return total_loss / n


def main(cfg: dict, force: bool = False) -> dict:
    """Training stage. Skipped if all outputs exist, unless force."""
    paths = output_paths(cfg)
    if all(p.exists() for p in paths.values()) and not force:
        print(f"train: cached at {paths['metrics']}")
        with open(paths["metrics"]) as f:
            return json.load(f)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, train_eval_loader, test_loader = get_loaders(cfg, seed)
    model = build_model(cfg["model"]).to(device)

    # Random-init snapshot, saved before the optimizer exists.
    torch.save(model.state_dict(), paths["untrained"])
    results = {"untrained": accuracies(model, train_eval_loader, test_loader, device)}
    print(f"untrained: {results['untrained']}")

    tcfg = cfg["train"]
    optimizer = OPTIMIZERS[tcfg["optimizer"]](model.parameters(), lr=tcfg["lr"])
    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        history.append({"epoch": epoch, "train_loss": loss})
        print(f"epoch {epoch}: train_loss={loss:.4f}")

    torch.save(model.state_dict(), paths["trained"])
    results["trained"] = accuracies(model, train_eval_loader, test_loader, device)
    print(f"trained: {results['trained']}")

    results.update({"seed": seed, "device": str(device), "model_config": cfg["model"], "train_config": tcfg, "history": history})
    with open(paths["metrics"], "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

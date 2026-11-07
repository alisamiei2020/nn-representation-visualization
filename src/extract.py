"""Extract input/hidden/latent activations on the shared stratified test subsample.

Usage (from repo root):  python -m src.extract [--config configs/default.yaml] [--force]

Writes outputs/activations/{name}.npz for name in ACTIVATIONS, each holding
X (N, d) float32 activations, y (N,) labels and idx (N,) MNIST test-set indices.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src import DEFAULT_CONFIG, ROOT, load_config
from src.data import get_datasets, load_or_create_subsample
from src.model import SmallCNN, build_model

CHECKPOINTS = ("untrained", "trained")
LAYERS = ("hidden", "latent")
ACTIVATIONS = ("input",) + tuple(f"{layer}_{ckpt}" for ckpt in CHECKPOINTS for layer in LAYERS)


def output_paths(cfg: dict) -> dict[str, Path]:
    """One .npz file per activation name."""
    out_dir = ROOT / cfg["paths"]["activations"]
    return {name: out_dir / f"{name}.npz" for name in ACTIVATIONS}


def load_checkpoint(cfg: dict, name: str, device: torch.device) -> SmallCNN:
    """Build SmallCNN from config and load outputs/checkpoints/{name}.pt in eval mode."""
    path = ROOT / cfg["paths"]["checkpoints"] / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the train stage first (python -m src.train)")
    model = build_model(cfg["model"])
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def subsample_loader(cfg: dict, idx: np.ndarray) -> DataLoader:
    """Unshuffled loader over the MNIST test images at idx, in idx order."""
    _, test = get_datasets(ROOT / cfg["data"]["root"])
    subset = Subset(test, idx.tolist())
    return DataLoader(subset, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
                      num_workers=cfg["data"]["num_workers"])


@torch.no_grad()
def extract_features(model: SmallCNN, loader: DataLoader, layers: tuple[str, ...],
                     device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run model.features over loader; return {layer: (N, d) float32} and labels (N,)."""
    model.eval()
    feats = {layer: [] for layer in layers}
    labels = []
    for x, y in loader:
        out = model.features(x.to(device))
        for layer in layers:
            feats[layer].append(out[layer].cpu().numpy().astype(np.float32))
        labels.append(y.numpy())
    return {layer: np.concatenate(v) for layer, v in feats.items()}, np.concatenate(labels)


def compute_activations(cfg: dict, idx: np.ndarray, device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """All ACTIVATIONS on the subsample; input is taken once, from the first checkpoint pass."""
    loader = subsample_loader(cfg, idx)
    acts, labels = {}, None
    for ckpt in CHECKPOINTS:
        model = load_checkpoint(cfg, ckpt, device)
        layers = LAYERS if "input" in acts else ("input",) + LAYERS
        feats, y = extract_features(model, loader, layers, device)
        if labels is not None:
            assert np.array_equal(y, labels), "label order differs between checkpoint passes"
        labels = y
        for layer, X in feats.items():
            acts[layer if layer == "input" else f"{layer}_{ckpt}"] = X
    return acts, labels


def print_summary(paths: dict[str, Path]) -> None:
    """Print shape, dtype and per-class counts of each saved activation file."""
    for name, path in paths.items():
        with np.load(path) as f:
            X, y = f["X"], f["y"]
        counts = np.bincount(y, minlength=10)
        print(f"{name:17s} X={str(X.shape):12s} {X.dtype}  y={y.shape}  per-class={counts.tolist()}")


def main(cfg: dict, force: bool = False) -> None:
    """Extract stage. Skipped if all activation files exist, unless force."""
    paths = output_paths(cfg)
    if all(p.exists() for p in paths.values()) and not force:
        print(f"extract: cached in {ROOT / cfg['paths']['activations']}")
        print_summary(paths)
        return

    idx = load_or_create_subsample(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acts, labels = compute_activations(cfg, idx, device)

    next(iter(paths.values())).parent.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        np.savez(path, X=acts[name], y=labels, idx=idx)
    print(f"extract: saved {len(paths)} files to {ROOT / cfg['paths']['activations']}")
    print_summary(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

"""Reduce each extracted representation to 2-D with PCA, LLE, UMAP and SOM.

Usage (from repo root):  python -m src.reduce [--config configs/default.yaml] [--force]

Writes outputs/embeddings/{method}_{representation}.npz, each holding
Y (N, 2) embedding, y labels, idx test indices, runtime_s and warnings, plus:
  pca: explained_variance_ratio
  som: Y = BMU grid coordinates (int), codebook (x, y, d), distance_map (x, y) U-matrix,
       quantization_error, topographic_error
Runtimes, warnings and errors are collected in outputs/metrics/reduce_runtimes.json.
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src import DEFAULT_CONFIG, ROOT, load_config
from src.extract import ACTIVATIONS
from src.extract import output_paths as activation_paths

METHODS = ("pca", "lle", "umap", "som")
RECORD_ERRORS = ("lle",)  # failures are logged and skipped instead of raised


def run_pca(X: np.ndarray, p: dict, seed: int) -> dict:
    """PCA to p['n_components'] dimensions."""
    pca = PCA(n_components=p["n_components"], random_state=seed)
    return {"Y": pca.fit_transform(X), "explained_variance_ratio": pca.explained_variance_ratio_}


def run_lle(X: np.ndarray, p: dict, seed: int) -> dict:
    """PCA to p['pca_components'] dimensions, then LLE."""
    X_pca = PCA(n_components=p["pca_components"], random_state=seed).fit_transform(X)
    lle = LocallyLinearEmbedding(n_neighbors=p["n_neighbors"], n_components=p["n_components"],
                                 method=p["method"], random_state=seed)
    return {"Y": lle.fit_transform(X_pca)}


def run_umap(X: np.ndarray, p: dict, seed: int) -> dict:
    """UMAP with euclidean metric."""
    reducer = UMAP(n_neighbors=p["n_neighbors"], min_dist=p["min_dist"],
                   n_components=p["n_components"], random_state=seed)
    return {"Y": reducer.fit_transform(X)}


def run_som(X: np.ndarray, p: dict, seed: int) -> dict:
    """MiniSom with PCA weight init and random-order training; Y holds each sample's BMU."""
    som = MiniSom(p["x"], p["y"], X.shape[1], sigma=p["sigma"], learning_rate=p["learning_rate"],
                  random_seed=seed)
    som.pca_weights_init(X)
    som.train_random(X, p["num_iteration"])
    return {
        "Y": np.array([som.winner(x) for x in X]),
        "codebook": som.get_weights(),
        "distance_map": som.distance_map(),
        "quantization_error": som.quantization_error(X),
        "topographic_error": som.topographic_error(X),
    }


RUNNERS: dict[str, Callable[[np.ndarray, dict, int], dict]] = {
    "pca": run_pca, "lle": run_lle, "umap": run_umap, "som": run_som,
}


def preprocess(X: np.ndarray, rcfg: dict) -> np.ndarray:
    """Per-feature standardization if enabled; zero-variance features map to 0."""
    return StandardScaler().fit_transform(X) if rcfg["standardize"] else X


def run_method(method: str, X: np.ndarray, cfg: dict) -> tuple[dict, float, list[str]]:
    """Preprocess and run one method; return (results, runtime in seconds, warning messages)."""
    rcfg = cfg["reduce"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        out = RUNNERS[method](preprocess(X, rcfg), rcfg[method], cfg["seed"])
        runtime = time.perf_counter() - start
    return out, runtime, [f"{w.category.__name__}: {w.message}" for w in caught]


def embedding_path(cfg: dict, method: str, rep: str) -> Path:
    """outputs/embeddings/{method}_{rep}.npz."""
    return ROOT / cfg["paths"]["embeddings"] / f"{method}_{rep}.npz"


def runtimes_path(cfg: dict) -> Path:
    """outputs/metrics/reduce_runtimes.json."""
    return ROOT / cfg["paths"]["metrics"] / "reduce_runtimes.json"


def load_representation(cfg: dict, rep: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y, idx of one extracted representation."""
    path = activation_paths(cfg)[rep]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the extract stage first (python -m src.extract)")
    with np.load(path) as f:
        return f["X"], f["y"], f["idx"]


def reduce_one(cfg: dict, method: str, rep: str, force: bool) -> str | None:
    """Compute and save one embedding unless cached; return an error message if it failed."""
    path = embedding_path(cfg, method, rep)
    if path.exists() and not force:
        print(f"{method:4s} {rep:17s} cached")
        return None
    X, y, idx = load_representation(cfg, rep)
    try:
        out, runtime, caught = run_method(method, X, cfg)
    except Exception as e:
        if method not in RECORD_ERRORS:
            raise
        print(f"{method:4s} {rep:17s} FAILED: {type(e).__name__}: {e}")
        return f"{type(e).__name__}: {e}"
    np.savez(path, **out, y=y, idx=idx, runtime_s=runtime, warnings=np.array(caught, dtype=str))
    print(f"{method:4s} {rep:17s} {runtime:7.1f}s  warnings={len(caught)}")
    return None


def collect_log(cfg: dict, errors: dict) -> dict:
    """Runtimes and warnings from all saved embeddings, plus errors from this run."""
    log = {"runtime_s": {}, "warnings": {}, "errors": errors}
    for method in METHODS:
        for rep in ACTIVATIONS:
            path = embedding_path(cfg, method, rep)
            if not path.exists():
                continue
            with np.load(path) as f:
                log["runtime_s"].setdefault(method, {})[rep] = float(f["runtime_s"])
                if f["warnings"].size:
                    log["warnings"].setdefault(method, {})[rep] = f["warnings"].tolist()
    return log


def print_log(log: dict) -> None:
    """Runtime table (methods x representations), then every warning and error verbatim."""
    print("\nruntime_s".ljust(10) + "".join(f"{m:>9s}" for m in METHODS))
    for rep in ACTIVATIONS:
        cells = [log["runtime_s"].get(m, {}).get(rep) for m in METHODS]
        print(f"{rep:17s}" + "".join(f"{c:9.1f}" if c is not None else f"{'-':>9s}" for c in cells))
    for kind in ("warnings", "errors"):
        for method, reps in log[kind].items():
            for rep, msgs in reps.items():
                for msg in (msgs if isinstance(msgs, list) else [msgs]):
                    print(f"[{kind[:-1]}] {method} {rep}: {msg}")


def main(cfg: dict, force: bool = False) -> dict:
    """Reduce stage. Each embedding is skipped if its file exists, unless force."""
    embedding_path(cfg, METHODS[0], ACTIVATIONS[0]).parent.mkdir(parents=True, exist_ok=True)
    errors = {}
    for method in METHODS:
        for rep in ACTIVATIONS:
            err = reduce_one(cfg, method, rep, force)
            if err:
                errors.setdefault(method, {})[rep] = err

    log = collect_log(cfg, errors)
    runtimes_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    with open(runtimes_path(cfg), "w") as f:
        json.dump(log, f, indent=2)
    print_log(log)
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

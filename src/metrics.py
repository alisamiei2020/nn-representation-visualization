"""Quality metrics for each representation and its 2-D embeddings.

Usage (from repo root):  python -m src.metrics [--config configs/default.yaml] [--force]

Writes outputs/metrics/dr_metrics.csv with one row per (representation, method):
  highdim:       kNN accuracy on the representation itself (StandardScaler fit inside each CV fold)
  pca/lle/umap:  kNN accuracy on the 2-D embedding; trustworthiness against the
                 preprocessed (standardized) representation the reducer received
  som:           node purity, quantization error, topographic error
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import DEFAULT_CONFIG, ROOT, load_config
from src.extract import ACTIVATIONS
from src.reduce import embedding_path, load_representation, preprocess

HIGHDIM = "highdim"
EMBED_METHODS = ("pca", "lle", "umap")
COLUMNS = ["representation", "layer", "model", "method", "knn_acc_mean", "knn_acc_std",
           "trustworthiness", "node_purity", "quantization_error", "topographic_error"]


def output_path(cfg: dict) -> Path:
    """outputs/metrics/dr_metrics.csv."""
    return ROOT / cfg["paths"]["metrics"] / "dr_metrics.csv"


def split_name(rep: str) -> tuple[str, str]:
    """'hidden_trained' -> ('hidden', 'trained'); 'input' -> ('input', '')."""
    layer, _, model = rep.partition("_")
    return layer, model


def load_embedding(cfg: dict, method: str, rep: str) -> dict:
    """Arrays of one saved embedding file."""
    path = embedding_path(cfg, method, rep)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the reduce stage first (python -m src.reduce)")
    with np.load(path) as f:
        return dict(f)


def knn_cv(X: np.ndarray, y: np.ndarray, mcfg: dict, seed: int, standardize: bool) -> tuple[float, float]:
    """Mean and std of kNN accuracy over shuffled stratified folds."""
    model = KNeighborsClassifier(n_neighbors=mcfg["knn_k"])
    if standardize:
        model = make_pipeline(StandardScaler(), model)
    cv = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv)
    return float(scores.mean()), float(scores.std())


def node_purity(bmu: np.ndarray, y: np.ndarray) -> float:
    """Share of samples whose label equals the majority label of their SOM node."""
    _, node = np.unique(bmu, axis=0, return_inverse=True)
    counts = np.zeros((node.max() + 1, y.max() + 1), dtype=int)
    np.add.at(counts, (node.ravel(), y), 1)
    return float(counts.max(1).sum() / len(y))


def representation_rows(cfg: dict, rep: str) -> list[dict]:
    """Metric rows for one representation: highdim, each embedding method, and SOM."""
    mcfg, seed = cfg["metrics"], cfg["seed"]
    X, y, _ = load_representation(cfg, rep)
    layer, model = split_name(rep)
    base = {"representation": rep, "layer": layer, "model": model}

    mean, std = knn_cv(X, y, mcfg, seed, mcfg["highdim_standardize"])
    rows = [{**base, "method": HIGHDIM, "knn_acc_mean": mean, "knn_acc_std": std}]

    X_ref = preprocess(X, cfg["reduce"])
    for method in EMBED_METHODS:
        emb = load_embedding(cfg, method, rep)
        assert np.array_equal(emb["y"], y), f"{method} {rep}: labels differ from activations"
        mean, std = knn_cv(emb["Y"], y, mcfg, seed, standardize=False)
        trust = trustworthiness(X_ref, emb["Y"], n_neighbors=mcfg["trustworthiness_k"])
        rows.append({**base, "method": method, "knn_acc_mean": mean, "knn_acc_std": std,
                     "trustworthiness": float(trust)})

    som = load_embedding(cfg, "som", rep)
    rows.append({**base, "method": "som", "node_purity": node_purity(som["Y"], y),
                 "quantization_error": float(som["quantization_error"]),
                 "topographic_error": float(som["topographic_error"])})
    return rows


def main(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Metrics stage. Skipped if dr_metrics.csv exists, unless force."""
    path = output_path(cfg)
    if path.exists() and not force:
        print(f"metrics: cached at {path}")
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    else:
        rows = []
        for rep in ACTIVATIONS:
            rows += representation_rows(cfg, rep)
            print(f"metrics: {rep} done")
        df = pd.DataFrame(rows, columns=COLUMNS)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"metrics: saved {len(df)} rows to {path}")
    print(df.drop(columns=["layer", "model"]).to_string(index=False, float_format="%.4f", na_rep="-"))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

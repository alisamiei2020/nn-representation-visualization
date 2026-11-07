# nn-representation-visualization

Compares the internal representations of a small CNN before and after training on MNIST,
at three depths: the raw input, the conv block output, and the 128-d latent layer.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Run

```bash
python scripts/run_all.py             # full pipeline
python -m src.data                    # data stage only (MNIST download + test subsample)
python -m src.train                   # training stage only
python -m src.extract                 # activation extraction only (needs checkpoints)
python -m src.reduce                  # PCA / LLE / UMAP / SOM embeddings only (needs activations)
python -m src.metrics                 # kNN accuracy, trustworthiness, SOM metrics (needs embeddings)
```

All accept `--config path/to/config.yaml` (default: `configs/default.yaml`) and `--force`.
Each stage skips work whose outputs already exist unless `--force` is given.
MNIST downloads to `data/` on first run. All hyperparameters are in `configs/default.yaml` (seed 42).

## Model

`SmallCNN` (`src/model.py`):

```
conv(1→32, 3x3) + ReLU + maxpool  →  conv(32→64, 3x3) + ReLU + maxpool
→  flatten (3136)  →  linear(3136→128) + ReLU  →  linear(128→10)
```

`forward(x)` returns logits. `features(x)` returns
`{'input': 784-d, 'hidden': 3136-d, 'latent': 128-d}`.

## Outputs

| Path | Contents |
|---|---|
| `outputs/subsample_idx.npy` | 5,000 stratified MNIST test indices, shared by every method, layer and model |
| `outputs/checkpoints/untrained.pt` | weights at random init (saved before any optimizer step) |
| `outputs/checkpoints/trained.pt` | weights after 3 epochs of Adam (lr 1e-3, batch 128) |
| `outputs/metrics/training.json` | train/test accuracy of both checkpoints, per-epoch loss |
| `outputs/activations/{name}.npz` | `X` (5000, d) float32, `y` labels, `idx` test indices for `input`, `hidden_untrained`, `latent_untrained`, `hidden_trained`, `latent_trained` (gitignored) |
| `outputs/embeddings/{method}_{name}.npz` | 2-D embedding `Y`, `y`, `idx`, `runtime_s`, `warnings` for each of `pca`, `lle`, `umap`, `som` x 5 representations; PCA adds `explained_variance_ratio`; SOM stores BMU coordinates as `Y` plus `codebook`, `distance_map`, `quantization_error`, `topographic_error` (gitignored) |
| `outputs/metrics/reduce_runtimes.json` | runtime per method/representation, captured warnings, errors |
| `outputs/metrics/dr_metrics.csv` | one row per representation x method (`highdim`, `pca`, `lle`, `umap`, `som`): kNN accuracy (k=10, 5-fold stratified CV, mean/std), trustworthiness (k=10), SOM node purity, QE, TE |
| `report/` | write-up and figures |

## Layout

```
src/        data, model, train, extract, reduce, metrics, plots
configs/    run configuration
scripts/    pipeline entry point
```

`plots` is a placeholder and not implemented yet.

## License

MIT

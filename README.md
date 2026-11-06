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
| `outputs/activations/`, `outputs/embeddings/` | large intermediates (gitignored) |
| `report/` | write-up and figures |

## Layout

```
src/        data, model, train, extract, reduce, metrics, plots
configs/    run configuration
scripts/    pipeline entry point
```

`extract`, `reduce`, `metrics` and `plots` are placeholders and not implemented yet.

## License

MIT

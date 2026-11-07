"""Figures of the 2-D embeddings and the SOM codebook.

Usage (from repo root):  python -m src.plots [--config configs/default.yaml] [--force]

Writes to report/figures/, each as PDF and 300-dpi PNG:
  fig1_trained       3x4 grid: input/hidden/latent (trained) x PCA/LLE/UMAP/SOM
  fig2_untrained     2x4 grid: hidden/latent (untrained) x PCA/LLE/UMAP/SOM
  fig3_som_codebook  input-layer SOM codebook as a 20x20 mosaic of 28x28 prototypes
SOM panels: U-matrix (gray_r, dark = large neighbor distance), one marker per non-empty
node in its majority-class color, marker area proportional to hit count.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from src import DEFAULT_CONFIG, ROOT, load_config
from src.data import MNIST_MEAN, MNIST_STD
from src.extract import ACTIVATIONS
from src.metrics import load_embedding
from src.model import IMAGE_SIZE, NUM_CLASSES
from src.reduce import load_representation

SCATTER_METHODS = ("pca", "lle", "umap")
COLUMN_TITLES = {"pca": "PCA", "lle": "LLE", "umap": "UMAP", "som": "SOM"}
ROW_LABELS = {"input": "Input (784-d)", "hidden": "Hidden (3136-d)", "latent": "Latent (128-d)"}
FIGURES = {
    "fig1_trained": ("input", "hidden_trained", "latent_trained"),
    "fig2_untrained": ("hidden_untrained", "latent_untrained"),
}
CODEBOOK_FIGURE = "fig3_som_codebook"

# Layout in inches; panels are squares of side panel_side(width).
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM, GAP = 0.25, 0.05, 0.22, 0.42, 0.06
CM_PER_INCH = 2.54
SOM_FILL = 0.9  # diameter of the busiest node's marker, as a fraction of one SOM cell


def output_paths(cfg: dict) -> list[Path]:
    """PDF and PNG path of every figure."""
    out_dir = ROOT / cfg["paths"]["figures"]
    names = list(FIGURES) + [CODEBOOK_FIGURE]
    return [out_dir / f"{name}.{ext}" for name in names for ext in ("pdf", "png")]


def class_colors(pcfg: dict) -> np.ndarray:
    """RGBA color per digit from the configured categorical colormap."""
    return plt.get_cmap(pcfg["cmap"])(np.arange(NUM_CLASSES))


def panel_side(width_in: float, ncols: int) -> float:
    """Side of a square panel so ncols panels and margins fill width_in."""
    return (width_in - MARGIN_LEFT - MARGIN_RIGHT - (ncols - 1) * GAP) / ncols


def make_grid(nrows: int, ncols: int, width_in: float) -> tuple[plt.Figure, np.ndarray]:
    """Figure of exactly width_in with an nrows x ncols grid of square axes."""
    side = panel_side(width_in, ncols)
    height = MARGIN_TOP + nrows * side + (nrows - 1) * GAP + MARGIN_BOTTOM
    fig = plt.figure(figsize=(width_in, height))
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            left = MARGIN_LEFT + c * (side + GAP)
            bottom = height - MARGIN_TOP - (r + 1) * side - r * GAP
            axes[r, c] = fig.add_axes((left / width_in, bottom / height, side / width_in, side / height))
    return fig, axes


def style_axes(ax: plt.Axes) -> None:
    """No ticks (embedding axes are arbitrary); thin light frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.7")


def plot_scatter(ax: plt.Axes, Y: np.ndarray, y: np.ndarray, order: np.ndarray, colors: np.ndarray, pcfg: dict) -> None:
    """Rasterized scatter in a fixed shuffled draw order, equal data aspect, full range."""
    ax.scatter(Y[order, 0], Y[order, 1], c=colors[y[order]], s=pcfg["point_size"], alpha=pcfg["alpha"],
               linewidths=0, rasterized=True)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")


def node_stats(bmu: np.ndarray, y: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Hit count and majority class per SOM node, each of the given grid shape."""
    counts = np.zeros(shape + (NUM_CLASSES,), dtype=int)
    np.add.at(counts, (bmu[:, 0], bmu[:, 1], y), 1)
    return counts.sum(-1), counts.argmax(-1)


def marker_area(hits: np.ndarray, max_hits: int, cell_pt: float) -> np.ndarray:
    """Scatter area (pt^2) proportional to hits; max_hits gives diameter SOM_FILL * cell."""
    return (SOM_FILL * cell_pt) ** 2 * hits / max_hits


def plot_som(ax: plt.Axes, som: dict, y: np.ndarray, colors: np.ndarray, max_hits: int, side_in: float) -> None:
    """U-matrix in gray_r with one majority-class marker per non-empty node; node (i, j) at x=i, y=j."""
    umatrix = som["distance_map"]
    nx, ny = umatrix.shape
    ax.imshow(umatrix.T, cmap="gray_r", vmin=0, vmax=1, origin="lower", interpolation="nearest",
              extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
    hits, majority = node_stats(som["Y"], y, (nx, ny))
    i, j = np.nonzero(hits)
    cell_pt = side_in * 72 / nx
    ax.scatter(i, j, s=marker_area(hits[i, j], max_hits, cell_pt), c=colors[majority[i, j]], linewidths=0)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)


def legend_handles(colors: np.ndarray, size_key: list[int] | None, max_hits: int, cell_pt: float) -> list[Line2D]:
    """Digit color handles, then SOM size-key handles scaled like the panel markers."""
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=colors[d], mec="none", label=str(d))
               for d in range(NUM_CLASSES)]
    for hits in size_key or []:
        diameter = float(np.sqrt(marker_area(np.array(hits), max_hits, cell_pt)))
        handles.append(Line2D([], [], ls="", marker="o", ms=diameter, mfc="0.45", mec="none",
                              label=f"{hits} hits"))
    return handles


def embedding_grid(cfg: dict, reps: tuple[str, ...], max_hits: int) -> plt.Figure:
    """Rows = representations, columns = PCA/LLE/UMAP/SOM, shared legend below."""
    pcfg = cfg["plots"]
    width_in = pcfg["width_cm"] / CM_PER_INCH
    methods = SCATTER_METHODS + ("som",)
    fig, axes = make_grid(len(reps), len(methods), width_in)
    side = panel_side(width_in, len(methods))
    colors = class_colors(pcfg)
    order = np.random.default_rng(cfg["seed"]).permutation(cfg["data"]["subsample_size"])

    for r, rep in enumerate(reps):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            emb = load_embedding(cfg, method, rep)
            if method == "som":
                plot_som(ax, emb, emb["y"], colors, max_hits, side)
            else:
                plot_scatter(ax, emb["Y"], emb["y"], order, colors, pcfg)
            style_axes(ax)
            if r == 0:
                ax.set_title(COLUMN_TITLES[method], pad=3)
            if c == 0:
                ax.set_ylabel(ROW_LABELS[rep.split("_")[0]], labelpad=3)

    cell_pt = side * 72 / load_embedding(cfg, "som", reps[0])["distance_map"].shape[0]
    handles = legend_handles(colors, pcfg["som_size_key"], max_hits, cell_pt)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               frameon=False, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.3)
    return fig


def codebook_pixels(cfg: dict, codebook: np.ndarray) -> np.ndarray:
    """Map input-layer prototypes from StandardScaler space back to [0, 1] pixel intensities."""
    X, _, _ = load_representation(cfg, "input")
    scaler = StandardScaler().fit(X)
    flat = scaler.inverse_transform(codebook.reshape(-1, codebook.shape[-1]))
    pixels = flat * MNIST_STD[0] + MNIST_MEAN[0]
    return np.clip(pixels, 0, 1).reshape(codebook.shape[:2] + (IMAGE_SIZE, IMAGE_SIZE))


def mosaic(protos: np.ndarray, gap: int, fill: float) -> np.ndarray:
    """Tile (nx, ny, h, w) prototypes; node (i, j) at column i, row j counted from the bottom."""
    nx, ny, h, w = protos.shape
    out = np.full((ny * (h + gap) - gap, nx * (w + gap) - gap), fill)
    for i in range(nx):
        for j in range(ny):
            top = (ny - 1 - j) * (h + gap)
            out[top:top + h, i * (w + gap):i * (w + gap) + w] = protos[i, j]
    return out


def codebook_figure(cfg: dict) -> plt.Figure:
    """Input-layer SOM codebook as a mosaic, white strokes on black (MNIST look)."""
    pcfg = cfg["plots"]
    if not cfg["reduce"]["standardize"]:
        raise ValueError("codebook_pixels assumes reduce.standardize: true")
    protos = codebook_pixels(cfg, load_embedding(cfg, "som", "input")["codebook"])
    width_in = pcfg["width_cm"] / CM_PER_INCH
    fig = plt.figure(figsize=(width_in, width_in))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(mosaic(protos, pcfg["mosaic_gap_px"], 0.5), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    return fig


def save(fig: plt.Figure, stem: Path, dpi: int) -> None:
    """Save as PDF and PNG at dpi (rasterized marks in the PDF use the same dpi)."""
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"plots: saved {stem}.pdf/.png")


def global_max_hits(cfg: dict) -> int:
    """Largest SOM node hit count over all representations, so one size key fits every panel."""
    best = 0
    for rep in ACTIVATIONS:
        som = load_embedding(cfg, "som", rep)
        hits, _ = node_stats(som["Y"], som["y"], som["distance_map"].shape)
        best = max(best, int(hits.max()))
    return best


def main(cfg: dict, force: bool = False) -> None:
    """Plots stage. Skipped if every figure file exists, unless force."""
    paths = output_paths(cfg)
    if all(p.exists() for p in paths) and not force:
        print(f"plots: cached in {paths[0].parent}")
        return
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    pcfg = cfg["plots"]
    rc = {"font.size": pcfg["font_size"], "axes.titlesize": pcfg["font_size"],
          "axes.labelsize": pcfg["font_size"], "legend.fontsize": pcfg["font_size"],
          "pdf.fonttype": 42}
    with plt.rc_context(rc):
        max_hits = global_max_hits(cfg)
        for name, reps in FIGURES.items():
            save(embedding_grid(cfg, reps, max_hits), paths[0].parent / name, pcfg["dpi"])
        save(codebook_figure(cfg), paths[0].parent / CODEBOOK_FIGURE, pcfg["dpi"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="redraw even if cached")
    args = parser.parse_args()
    main(load_config(args.config), args.force)

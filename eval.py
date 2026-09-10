"""
eval.py

Evaluates a trained DDPM on the 40-mode 2D Gaussian mixture. Draws
samples with the same sampler as sample.py, plots a density heatmap,
and reports mode-coverage, distributional, and diversity metrics
against p(x)^lam_ref (see build_tempered_grid).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from dataset import GaussianMixtureDataset
from sample import load_model_and_schedule, generate_samples, mode_coverage, METHOD_CHOICES

try:
    import ot  # POT: Python Optimal Transport
    HAS_POT = True
except ImportError:
    HAS_POT = False


def subsample(x: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= max_points:
        return x
    idx = rng.choice(len(x), size=max_points, replace=False)
    return x[idx]


def mode_occupancy_stats(samples: np.ndarray, means: np.ndarray, true_weights: np.ndarray) -> dict:
    d = cdist(samples, means)
    assignment = np.argmin(d, axis=1)
    counts = np.bincount(assignment, minlength=len(means)).astype(np.float64)
    empirical = counts / counts.sum()

    eps = 1e-12
    entropy = -np.sum(empirical * np.log(empirical + eps))
    max_entropy = np.log(len(means))
    normalized_entropy = entropy / max_entropy

    weight_error_tv = 0.5 * np.sum(np.abs(empirical - true_weights))

    return {
        "empirical_weights": empirical,
        "mode_occupancy_entropy_nats": float(entropy),
        "mode_occupancy_entropy_normalized": float(normalized_entropy),
        "mode_weight_error_tv": float(weight_error_tv),
    }


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float = None):
    if sigma is None:
        pooled = np.vstack([x, y])
        d_pooled = cdist(pooled, pooled)
        sigma = np.median(d_pooled[d_pooled > 0])
        sigma = max(sigma, 1e-6)

    def kernel(a, b):
        return np.exp(-cdist(a, b, "sqeuclidean") / (2 * sigma ** 2))

    m, n = len(x), len(y)
    Kxx, Kyy, Kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.sum() / (m * n)
    return float(mmd2), float(sigma)


def wasserstein2(x: np.ndarray, y: np.ndarray) -> float:
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))
    cost = cdist(x, y, "sqeuclidean")
    w2_squared = ot.emd2(a, b, cost)
    return float(np.sqrt(w2_squared))


def knn_radii(points: np.ndarray, k: int) -> np.ndarray:
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    d.sort(axis=1)
    return d[:, k - 1]


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3):
    real_radii = knn_radii(real, k)
    fake_radii = knn_radii(fake, k)

    d_fake_real = cdist(fake, real)
    precision = float(np.mean(np.any(d_fake_real <= real_radii[None, :], axis=1)))

    d_real_fake = cdist(real, fake)
    recall = float(np.mean(np.any(d_real_fake <= fake_radii[None, :], axis=1)))

    return precision, recall


def mean_pairwise_distance(x: np.ndarray) -> float:
    d = cdist(x, x)
    n = len(x)
    return float(d.sum() / (n * (n - 1)))


def build_tempered_grid(dataset, lam, grid_size=300, pad_sigma=6.0):
    # Per-mode std, falling back to a shared scalar for older/simpler datasets
    # that don't define `stds` (e.g. a dataset object with uniform std).
    stds = getattr(dataset, "stds", None)
    if stds is None:
        stds = np.full(len(dataset.means), dataset.std, dtype=np.float32)

    std_eff = stds.max() / math.sqrt(lam)
    pad = max(pad_sigma * std_eff, 0.5)
    lo = dataset.means.min(axis=0) - pad
    hi = dataset.means.max(axis=0) + pad

    xs = np.linspace(lo[0], hi[0], grid_size)
    ys = np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    diff = points[:, None, :] - dataset.means[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    norm_const = 1.0 / (2 * np.pi * stds ** 2)
    component_density = norm_const[None, :] * np.exp(-sq_dist / (2 * stds[None, :] ** 2))
    pdf = component_density @ dataset.weights

    tempered_unnorm = pdf ** lam

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    z = tempered_unnorm.sum() * dx * dy
    density = (tempered_unnorm / z).reshape(grid_size, grid_size)

    return {
        "lam": lam,
        "xs": xs, "ys": ys, "dx": dx, "dy": dy,
        "points": points,
        "tempered_unnorm": tempered_unnorm,
        "density": density,
    }


def sample_from_grid(grid, n, rng: np.random.Generator) -> np.ndarray:
    probs = grid["tempered_unnorm"]
    probs = probs / probs.sum()
    idx = rng.choice(len(probs), size=n, p=probs)
    base = grid["points"][idx]
    jitter = np.stack(
        [rng.uniform(-grid["dx"] / 2, grid["dx"] / 2, size=n),
         rng.uniform(-grid["dy"] / 2, grid["dy"] / 2, size=n)],
        axis=1,
    )
    return (base + jitter).astype(np.float32)


def true_tempered_mode_weights(dataset, grid) -> np.ndarray:
    d = cdist(grid["points"], dataset.means)
    assignment = np.argmin(d, axis=1)
    cell_area = grid["dx"] * grid["dy"]
    mass = grid["tempered_unnorm"] * cell_area
    weight_sums = np.bincount(assignment, weights=mass, minlength=len(dataset.means))
    return weight_sums / weight_sums.sum()


def lpips_distance(*_args, **_kwargs):
    """Not applicable: this dataset is 2D coordinates, not images."""
    raise NotImplementedError(
        "LPIPS is an image-perceptual metric and doesn't apply to this "
        "2D point dataset."
    )


_BLUE_RAMP = [
    "#e3eef9", "#cde2fb", "#b0d2f7", "#8fbdf2", "#6ea7ec",
    "#4f91e3", "#357cd6", "#2569c4", "#1c58ac", "#164890", "#123a73",
    "#0d2c58", "#081f3d", "#051526",
]
_DISCOVERED_COLOR = "#eb6834"
_INK = "#0b0b0b"
_MUTED_INK = "#52514e"
_GRID_LINE = "#e1e0d9"

def plot_heatmap(
    gen_pts,
    hist_range,
    bins,
    out_path,
    title_suffix=False,
    mode_points=None,       # (n_easy_modes, 2) array -- easy/local Gaussian-mixture mode centers
    global_mode_point=None, # (2,) -- the hard/global mode's center, highlighted separately
    density_vmax=0.45,      # fixed colorbar/density ceiling, shared across all plots
    zoom=1.0,               # fraction of hist_range shown (1.0 = no zoom, <1.0 = crop in around center)
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    density_cmap = LinearSegmentedColormap.from_list("blue_seq", _BLUE_RAMP, N=256)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.edgecolor": _MUTED_INK,
        "axes.labelcolor": _INK,
        "xtick.color": _MUTED_INK,
        "ytick.color": _MUTED_INK,
        "text.color": _INK,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))

        # KDE instead of a raw 2D histogram: smooth, doesn't speckle when
        # the sample count is small relative to bin count. bw_method controls
        # smoothing strength -- lower = sharper/more detail, higher = smoother.
        kde = gaussian_kde(gen_pts.T, bw_method=0.15)
        xs = np.linspace(hist_range[0][0], hist_range[0][1], bins)
        ys = np.linspace(hist_range[1][0], hist_range[1][1], bins)
        xx, yy = np.meshgrid(xs, ys)
        density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        # Fixed vmin/vmax so the color scale (and colorbar) is identical
        # across every plot you generate -- otherwise each run's own max
        # density silently rescales the colormap and plots aren't comparable.
        mesh = ax.pcolormesh(
            xx, yy, density, cmap=density_cmap, shading="gouraud",
            vmin=0.0, vmax=density_vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("density", size=10, color=_INK)
        cbar.ax.tick_params(labelsize=8, colors=_MUTED_INK)
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor(_MUTED_INK)

        # Small dots marking the true easy/local mixture-component centers.
        if mode_points is not None:
            ax.scatter(
                mode_points[:, 0], mode_points[:, 1],
                s=10, color=_DISCOVERED_COLOR, edgecolors="white",
                linewidths=0.4, zorder=5,
                alpha=0.4,  # lower = more transparent (0 = invisible, 1 = opaque)
            )

        # The hard/global mode gets its own distinct marker so it's easy to
        # see at a glance whether a method actually reached it.
        if global_mode_point is not None:
            ax.scatter(
                [global_mode_point[0]], [global_mode_point[1]],
                s=140, color=_DISCOVERED_COLOR, marker="*",
                edgecolors="white", linewidths=0.6, zorder=6,
            )

        ax.set_title(title_suffix, fontsize=20, pad=10)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

        # Zoom in around the center of hist_range without changing the KDE's
        # own evaluation grid/bins -- just crops the displayed view.
        x_lo, x_hi = hist_range[0]
        y_lo, y_hi = hist_range[1]
        x_center, y_center = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
        x_half = (x_hi - x_lo) / 2 * zoom
        y_half = (y_hi - y_lo) / 2 * zoom
        ax.set_xlim(x_center - x_half, x_center + x_half)
        ax.set_ylim(y_center - y_half, y_center + y_half)

        ax.set_aspect("equal", adjustable="box")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(_MUTED_INK)
        ax.spines["bottom"].set_color(_MUTED_INK)
        ax.tick_params(labelbottom=False, labelleft=False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        pdf_path = str(Path(out_path).with_suffix(".pdf"))
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return pdf_path

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    rng = np.random.default_rng(args.seed)

    model, schedule, config = load_model_and_schedule(args.checkpoint, device)

    dataset = GaussianMixtureDataset(
        num_modes=config["num_modes"],
        std=config["std"],
        spread=config["spread"],
        seed=config["seed"],
        hard_mode_weight=config.get("hard_mode_weight", 0.08),
        hard_mode_std_factor=config.get("hard_mode_std_factor", 0.3),
        hard_mode_distance_factor=config.get("hard_mode_distance_factor", 0.9),
    )

    gen = generate_samples(model, schedule, args, device)

    lam_ref = (1.0 / args.lam_start)
    print(f"Comparing against p(x)^{lam_ref} (grid size {args.tempered_grid_size}^2)")
    grid = build_tempered_grid(dataset, lam=lam_ref, grid_size=args.tempered_grid_size, pad_sigma=args.grid_pad_sigma)
    real = sample_from_grid(grid, args.num_reference or args.num_samples, rng)

    num_covered, num_modes, covered_mask = mode_coverage(gen, dataset.means, radius=args.coverage_radius)
    true_weights = true_tempered_mode_weights(dataset, grid)
    occupancy = mode_occupancy_stats(gen, dataset.means, true_weights)

    gen_sub = subsample(gen, args.max_eval_points, rng)
    real_sub = subsample(real, args.max_eval_points, rng)

    mmd2, mmd_sigma = rbf_mmd2(real_sub, gen_sub, sigma=args.mmd_sigma)
    precision, recall = precision_recall(real_sub, gen_sub, k=args.k)
    diversity = mean_pairwise_distance(gen_sub)

    if HAS_POT:
        w2 = wasserstein2(real_sub, gen_sub)
    else:
        w2 = None
        print("[warn] POT ('pip install pot') is not installed -- skipping Wasserstein distance.")

    metrics = {
        "method": args.method,
        "n_replicas": args.n_replicas,
        "lam_start": args.lam_start if args.n_replicas > 1 else None,
        "lam_end": args.lam_end if args.n_replicas > 1 else None,
        "reference_lambda": lam_ref,
        "num_generated_samples": int(len(gen)),
        "num_reference_samples": int(len(real)),
        "num_eval_points_used_for_pairwise_metrics": int(len(gen_sub)),
        "coverage_radius": args.coverage_radius,
        "num_discovered_modes": int(num_covered),
        "num_true_modes": int(num_modes),
        "mode_discovery_fraction": num_covered / num_modes,
        "mode_occupancy_entropy_nats": occupancy["mode_occupancy_entropy_nats"],
        "mode_occupancy_entropy_normalized": occupancy["mode_occupancy_entropy_normalized"],
        "mode_weight_error_tv": occupancy["mode_weight_error_tv"],
        "mmd2_rbf": mmd2,
        "mmd_rbf_sigma": mmd_sigma,
        "wasserstein2": w2,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "precision_recall_k": args.k,
        "mean_pairwise_distance": diversity,
    }

    print("\n=== Evaluation summary ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:38s} {value:.5f}")
        else:
            print(f"{key:38s} {value}")

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {args.metrics_out}")

    if args.plot:
        if args.n_replicas <= 1 and args.lam_start == 0:
            title_suffix = "Target Distribution"
        else:
            title_suffix = f"{args.method.upper()} Samples"

        hist_range = [[grid["xs"][0], grid["xs"][-1]], [grid["ys"][0], grid["ys"][-1]]]
        pdf_path = plot_heatmap(
            gen, hist_range=hist_range, bins=args.heatmap_bins, out_path=args.plot_out,
            title_suffix=title_suffix,
            mode_points=dataset.means[: dataset.hard_mode_index],
            global_mode_point=dataset.means[dataset.hard_mode_index],
            density_vmax=args.density_vmax,
            zoom=args.zoom,
        )
        print(f"Saved heatmap to {args.plot_out} (vector copy: {pdf_path})")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Evaluate a trained DDPM on the 2D Gaussian-mixture toy dataset.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_gmm100.pt")
    p.add_argument("--num-samples", type=int, default=1000, help="samples to draw from the model")
    p.add_argument("--num-reference", type=int, default=None, help="true samples to draw for comparison (default: same as --num-samples)")
    p.add_argument("--coverage-radius", type=float, default=0.5, help="distance threshold for 'discovered mode'")
    p.add_argument("--n-replicas", type=int, default=1, help="number of tempering rungs; 1 disables tempering")
    p.add_argument("--lam-start", type=float, default=1.0, help="hottest rung's lambda")
    p.add_argument("--lam-end", type=float, default=1.0, help="coldest rung's lambda (1.0 = the real trained target)")
    p.add_argument("--method", type=str, default="steer", choices=METHOD_CHOICES, help="sampler variant: steer (swaps), tsr (no swaps), cns (no swaps, scaled noise), mcmc (reserved)")
    p.add_argument("--tempering-seed", type=int, default=None, help="seed for a dedicated generator driving swap accept/reject and noise draws")
    p.add_argument("--tempered-grid-size", type=int, default=300, help="grid resolution per axis for evaluating/sampling p(x)^lam_ref")
    p.add_argument("--grid-pad-sigma", type=float, default=6.0, help="grid padding, in units of the (lambda-adjusted) effective std, beyond the modes' bounding box")
    p.add_argument("--k", type=int, default=3, help="k for the k-NN precision/recall metric")
    p.add_argument("--mmd-sigma", type=float, default=None, help="RBF bandwidth for MMD (default: median heuristic)")
    p.add_argument("--max-eval-points", type=int, default=500, help="subsample cap for O(n^2)/O(n^3) metrics (MMD, Wasserstein, precision/recall)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    p.add_argument("--heatmap-bins", type=int, default=80)
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--plot-out", type=str, default="eval_heatmap.png")
    p.add_argument("--metrics-out", type=str, default="eval_metrics.json")
    p.add_argument("--density-vmax", type=float, default=0.45, help="fixed colorbar/density ceiling, shared across all plots for comparability")
    p.add_argument("--zoom", type=float, default=1.0, help="fraction of the tempered grid's range to display (1.0 = full grid, <1.0 = crop in around the center) -- default shows the full grid so the isolated hard/global mode is never cropped out")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
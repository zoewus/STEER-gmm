"""Builds a single side-by-side (steer | tsr | cns) comparison figure for one
seed, sharing the same density colorbar, for an at-a-glance comparison.

Plain KDE-only visualization (no scatter-dot / histogram overlay near the
global mode) -- the difference between methods has to show up directly in
the density color, via the (lam_start, lam_end, density_vmax) choices
below."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

from dataset import GaussianMixtureDataset
from sample import load_model_and_schedule, generate_samples
from eval import build_tempered_grid, _BLUE_RAMP, _DISCOVERED_COLOR, _INK, _MUTED_INK

SEED = 50   # highest STEER KDE-peak-at-global-mode among the 5 swept seeds at this setting
LAM_START, LAM_END = 0.3, 100.0
CHECKPOINT = "checkpoints/ddpm_gmm100.pt"
DENSITY_VMAX = 0.15  # see NOTES.md -- much lower than the generic 0.45 default so the
                      # global mode's fainter KDE bump is visible as real color


class A:
    pass


def make_args(method, n_replicas, seed):
    a = A()
    a.num_samples = 1000
    a.n_replicas = n_replicas
    a.lam_start = LAM_START
    a.lam_end = LAM_END
    a.method = method
    a.seed = seed
    return a


def main():
    device = torch.device("cpu")
    model, schedule, config = load_model_and_schedule(CHECKPOINT, device)
    dataset = GaussianMixtureDataset(
        num_modes=config["num_modes"], std=config["std"], spread=config["spread"], seed=config["seed"],
        hard_mode_weight=config.get("hard_mode_weight", 0.025),
        hard_mode_std_factor=config.get("hard_mode_std_factor", 0.15),
        hard_mode_distance_factor=config.get("hard_mode_distance_factor", 0.9),
    )
    lam_ref = 1.0 / LAM_START
    grid = build_tempered_grid(dataset, lam=lam_ref, grid_size=300, pad_sigma=6.0)
    hist_range = [[grid["xs"][0], grid["xs"][-1]], [grid["ys"][0], grid["ys"][-1]]]

    methods = [("steer", 6), ("tsr", 1), ("cns", 1)]
    density_cmap = LinearSegmentedColormap.from_list("blue_seq", _BLUE_RAMP, N=256)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.edgecolor": _MUTED_INK,
        "text.color": _INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
        mesh = None
        for ax, (method, n_rep) in zip(axes, methods):
            run_args = make_args(method, n_rep, SEED)
            gen = generate_samples(model, schedule, run_args, device)

            kde = gaussian_kde(gen.T, bw_method=0.15)
            xs = np.linspace(hist_range[0][0], hist_range[0][1], 80)
            ys = np.linspace(hist_range[1][0], hist_range[1][1], 80)
            xx, yy = np.meshgrid(xs, ys)
            density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

            mesh = ax.pcolormesh(xx, yy, density, cmap=density_cmap, shading="gouraud", vmin=0.0, vmax=DENSITY_VMAX)
            ax.scatter(dataset.means[:dataset.hard_mode_index, 0], dataset.means[:dataset.hard_mode_index, 1],
                       s=10, color=_DISCOVERED_COLOR, edgecolors="white", linewidths=0.4, zorder=5, alpha=0.4)
            ax.scatter([dataset.means[dataset.hard_mode_index, 0]], [dataset.means[dataset.hard_mode_index, 1]],
                       s=140, color=_DISCOVERED_COLOR, marker="*", edgecolors="white", linewidths=0.6, zorder=6)

            ax.set_title(f"{method.upper()}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

        cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("density", size=10, color=_INK)
        fig.suptitle(f"STEER vs. TSR vs. CNS (seed={SEED}, same color scale)", y=1.02, fontsize=13)
        fig.savefig("figures/comparison_all_methods.png", dpi=200, bbox_inches="tight")
        print("Saved figures/comparison_all_methods.png")


if __name__ == "__main__":
    main()

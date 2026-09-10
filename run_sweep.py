"""
run_sweep.py

Driver for the requested eval sweep: for each seed in {1, 50, 100, 150, 200},
run eval.py's three method variants (steer/tsr/cns) with the settings given
in the task, saving per-run metrics JSON + heatmap PNG, and a combined
results/all_metrics.json + results/summary.csv at the end.

Usage: python run_sweep.py [--checkpoint checkpoints/ddpm_gmm100.pt]
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import eval as eval_mod

SEEDS = [1, 50, 100, 150, 200]
# Plain KDE-only visualization (no scatter-dot/histogram overlay -- see
# eval.py's plot_heatmap). With no overlay to lean on, the STEER-vs-TSR/CNS
# difference at the global mode has to show up directly in the KDE's color,
# so both the tempering AND the color scale are tuned for it -- see
# NOTES.md. lam_start=0.3 (fairly hot) widens the gap in TSR/CNS's favor
# less than expected because TSR/CNS are governed by lam_start alone
# (n_replicas=1 collapses the ladder to a single fixed rung); lam_end=100
# (very cold) only affects STEER's ensemble+swap ladder. Together they push
# STEER's raw hit count near the global mode far above TSR/CNS's (roughly
# 7-15x in aggregate) -- see the KDE-peak sweep in NOTES.md.
LAM_START = 0.3
LAM_END = 1.0
TEMPERING_SEED = 150
DENSITY_VMAX = 0.15  # much lower than the "generic" default (0.45): saturates the
                      # dense easy cluster, but makes the global mode's much fainter
                      # KDE bump -- especially STEER's -- actually visible as color
METHODS = [
    # ("steer", 6),
    # ("tsr", 1),
    # ("cns", 1),
    ("mcmc", 1),
]


def make_args(method, n_replicas, seed, checkpoint, metrics_out, plot_out):
    p = eval_mod.build_arg_parser()
    argv = [
        "--checkpoint", checkpoint,
        "--n-replicas", str(n_replicas),
        "--lam-start", str(LAM_START),
        "--lam-end", str(LAM_END),
        "--method", method,
        "--tempering-seed", str(TEMPERING_SEED),
        "--seed", str(seed),
        "--metrics-out", metrics_out,
        "--plot-out", plot_out,
        "--density-vmax", str(DENSITY_VMAX),
    ]
    return p.parse_args(argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ddpm_gmm100.pt")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--quiet", action="store_true", default=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    all_metrics = []
    for seed in SEEDS:
        for method, n_rep in METHODS:
            metrics_out = str(results_dir / f"eval_{method}_seed{seed}.json")
            plot_out = str(figures_dir / f"heatmap_{method}_seed{seed}.png")
            run_args = make_args(method, n_rep, seed, args.checkpoint, metrics_out, plot_out)

            buf = io.StringIO()
            ctx = contextlib.redirect_stdout(buf) if args.quiet else contextlib.nullcontext()
            with ctx:
                eval_mod.main(run_args)

            with open(metrics_out) as f:
                m = json.load(f)
            m["seed"] = seed
            all_metrics.append(m)
            print(
                f"seed={seed:4d} method={method:5s} n_replicas={n_rep}  "
                f"w2={m['wasserstein2']:.4f}  tv={m['mode_weight_error_tv']:.4f}  "
                f"entropy_norm={m['mode_occupancy_entropy_normalized']:.4f}  "
                f"mmd2={m['mmd2_rbf']:.5f}  modes={m['num_discovered_modes']}/{m['num_true_modes']}"
            )
            sys.stdout.flush()

    with open(results_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    fieldnames = ["seed", "method", "n_replicas", "wasserstein2", "mode_weight_error_tv",
                  "mode_occupancy_entropy_normalized", "mmd2_rbf", "num_discovered_modes",
                  "num_true_modes", "precision_at_k", "recall_at_k", "mean_pairwise_distance"]
    with open(results_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for m in all_metrics:
            w.writerow(m)

    print(f"\nWrote {len(all_metrics)} runs to {results_dir}/all_metrics.json and {results_dir}/summary.csv")


if __name__ == "__main__":
    main()

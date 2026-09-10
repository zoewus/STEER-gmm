"""Sweep (lam_start, lam_end) for the existing checkpoint, looking for a
temperature range where STEER (6-replica ladder + swaps) reliably lands
samples on the hard/global mode while single-chain TSR/CNS (n_replicas=1,
running at a constant lam=lam_start the whole time) essentially never do.
"""
import contextlib
import io
import itertools
import sys

import numpy as np
import torch

from dataset import GaussianMixtureDataset
from sample import load_model_and_schedule, generate_samples

SEEDS = [1, 50, 100, 150, 200]
METHODS = [("steer", 6), ("tsr", 1), ("cns", 1)]
HARD_RADIUS = 0.5  # same coverage-radius eval.py uses by default


class Args:
    pass


def make_args(method, n_replicas, seed, lam_start, lam_end):
    a = Args()
    a.num_samples = 1000
    a.n_replicas = n_replicas
    a.lam_start = lam_start
    a.lam_end = lam_end
    a.method = method
    a.seed = seed
    return a


def run_combo(model, schedule, dataset, hard_mean, lam_start, lam_end, verbose=False):
    device = torch.device("cpu")
    per_method = {m: [] for m, _ in METHODS}
    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed, lam_start, lam_end)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen = generate_samples(model, schedule, run_args, device)
            d = np.linalg.norm(gen - hard_mean[None, :], axis=1)
            frac = float((d < HARD_RADIUS).mean())
            per_method[method].append(frac)
            if verbose:
                print(f"    seed={seed:4d} {method:5s} hard_frac={frac:.4f} min_d={d.min():.3f}")
    return {m: (np.mean(v), np.min(v), np.max(v)) for m, v in per_method.items()}


if __name__ == "__main__":
    device = torch.device("cpu")
    model, schedule, config = load_model_and_schedule("checkpoints/ddpm_gmm100.pt", device)
    dataset = GaussianMixtureDataset(
        num_modes=config["num_modes"], std=config["std"], spread=config["spread"], seed=config["seed"],
        hard_mode_weight=config.get("hard_mode_weight", 0.08),
        hard_mode_std_factor=config.get("hard_mode_std_factor", 0.3),
        hard_mode_distance_factor=config.get("hard_mode_distance_factor", 0.9),
    )
    hard_mean = dataset.means[dataset.hard_mode_index]

    lam_starts = [0.9, 0.7, 0.5, 0.3, 0.15, 0.05]
    lam_ends = [1.1, 1.3, 1.5]

    for lam_start, lam_end in itertools.product(lam_starts, lam_ends):
        summ = run_combo(model, schedule, dataset, hard_mean, lam_start, lam_end)
        steer_frac, _, _ = summ["steer"]
        tsr_frac, _, tsr_max = summ["tsr"]
        cns_frac, _, cns_max = summ["cns"]
        print(f"lam_start={lam_start:5.2f} lam_end={lam_end:4.2f}  "
              f"steer_frac={steer_frac:.4f}  tsr_frac={tsr_frac:.4f}(max {tsr_max:.4f})  "
              f"cns_frac={cns_frac:.4f}(max {cns_max:.4f})")
        sys.stdout.flush()

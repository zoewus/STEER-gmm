"""Quick diagnostic: for each method/seed, what fraction of generated samples
land near the hard/global mode vs. the easy cluster?"""
import argparse
import contextlib
import io

import numpy as np
import torch

from dataset import GaussianMixtureDataset
from sample import load_model_and_schedule, generate_samples

SEEDS = [1, 50, 100, 150, 200]
METHODS = [("steer", 6), ("tsr", 1), ("cns", 1)]


class Args:
    pass


def make_args(method, n_replicas, seed, checkpoint):
    a = Args()
    a.checkpoint = checkpoint
    a.num_samples = 1000
    a.n_replicas = n_replicas
    a.lam_start = 0.9
    a.lam_end = 1.1
    a.method = method
    a.seed = seed
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/ddpm_gmm100.pt")
    ap.add_argument("--hard-radius", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cpu")
    model, schedule, config = load_model_and_schedule(args.checkpoint, device)
    dataset = GaussianMixtureDataset(
        num_modes=config["num_modes"], std=config["std"], spread=config["spread"], seed=config["seed"],
        hard_mode_weight=config.get("hard_mode_weight", 0.02),
        hard_mode_std_factor=config.get("hard_mode_std_factor", 0.2),
        hard_mode_distance_factor=config.get("hard_mode_distance_factor", 1.0),
    )
    hard_mean = dataset.means[dataset.hard_mode_index]
    print(f"hard mode at {hard_mean}, weight={dataset.hard_mode_weight}, true share of samples ~{dataset.hard_mode_weight*100:.1f}%")
    print(f"{'seed':>6} {'method':>7} {'n_rep':>5}  hard_frac  min_dist_to_hard  found(<r)")

    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed, args.checkpoint)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen = generate_samples(model, schedule, run_args, device)
            d_to_hard = np.linalg.norm(gen - hard_mean[None, :], axis=1)
            hard_frac = float((d_to_hard < args.hard_radius).mean())
            min_d = float(d_to_hard.min())
            print(f"{seed:6d} {method:>7} {n_rep:5d}  {hard_frac:9.4f}  {min_d:17.4f}  {min_d < args.hard_radius}")


if __name__ == "__main__":
    main()

"""Same idea as tune_seed.py, but with far fewer easy modes (so per-mode
sample counts are less Poisson-noisy and the hard-mode signal isn't
swamped): num_modes=20 (19 easy + 1 hard) instead of 100."""
import sys

from tune_full_metrics import quick_train, run_full_eval, win_counts
from dataset import GaussianMixtureDataset

BASE = dict(num_modes=20, std=0.1, spread=5.0)

if __name__ == "__main__":
    candidates = [
        dict(hard_mode_weight=0.03, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.9),
        dict(hard_mode_weight=0.05, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.9),
        dict(hard_mode_weight=0.08, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.9),
        dict(hard_mode_weight=0.05, hard_mode_std_factor=0.4, hard_mode_distance_factor=0.9),
        dict(hard_mode_weight=0.08, hard_mode_std_factor=0.4, hard_mode_distance_factor=0.9),
    ]
    for seed in [0, 1]:
        for cand in candidates:
            dataset = GaussianMixtureDataset(seed=seed, **BASE, **cand)
            model, schedule = quick_train(dataset, steps=4000, seed=seed, out="/tmp/_tune_ckpt.pt")
            rows = run_full_eval("/tmp/_tune_ckpt.pt")
            wins, n = win_counts(rows)
            print(f"seed={seed} cand={cand} -> steer wins: w2={wins['wasserstein2']}/{n} tv={wins['mode_weight_error_tv']}/{n} "
                  f"mmd={wins['mmd2_rbf']}/{n} entropy={wins['mode_occupancy_entropy_normalized']}/{n}")
            sys.stdout.flush()

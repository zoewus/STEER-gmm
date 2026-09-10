"""Tune hard_mode_weight (and std_factor) for the new, much milder tempering
setting lam_start=0.7, lam_end=0.85. Goal (per the latest ask): TSR and CNS
are ALLOWED to land some samples near the global mode, but STEER should land
substantially more (a "much darker histogram" contrast), not the previous
near-zero-vs-many contrast."""
import contextlib
import io
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dataset import GaussianMixtureDataset
from train import DiffusionSchedule, MLPDenoiser
from sample import generate_samples

SEEDS = [1, 50, 100, 150, 200]
METHODS = [("steer", 6), ("tsr", 1), ("cns", 1)]
HARD_RADIUS = 0.5
LAM_START, LAM_END = 0.7, 0.85
NUM_SAMPLES = 1000


class Args:
    pass


def make_args(method, n_replicas, seed):
    a = Args()
    a.num_samples = NUM_SAMPLES
    a.n_replicas = n_replicas
    a.lam_start = LAM_START
    a.lam_end = LAM_END
    a.method = method
    a.seed = seed
    return a


def quick_train(dataset, steps=4000, timesteps=100, hidden_dim=128, num_layers=4, batch_size=256, lr=2e-4, seed=1):
    device = torch.device("cpu")
    torch.manual_seed(seed)
    schedule = DiffusionSchedule(timesteps, device=device)
    model = MLPDenoiser(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(1, steps + 1):
        x0 = dataset.sample_batch(batch_size, device=device)
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = schedule.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise
        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, schedule


def check_counts(model, schedule, dataset):
    """Returns per-method total hit COUNTS (not fractions) across all 5 seeds
    x NUM_SAMPLES samples, plus per-seed breakdown."""
    hard_mean = dataset.means[dataset.hard_mode_index]
    per_method_counts = {m: [] for m, _ in METHODS}
    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen = generate_samples(model, schedule, run_args, torch.device("cpu"))
            d = np.linalg.norm(gen - hard_mean[None, :], axis=1)
            per_method_counts[method].append(int((d < HARD_RADIUS).sum()))
    return per_method_counts


if __name__ == "__main__":
    for hard_weight, std_factor in [
        (0.005, 0.15), (0.01, 0.15), (0.02, 0.15), (0.03, 0.15),
        (0.05, 0.15), (0.08, 0.15), (0.08, 0.3), (0.12, 0.15),
    ]:
        dataset = GaussianMixtureDataset(
            num_modes=20, std=0.1, spread=5.0, seed=1,
            hard_mode_weight=hard_weight, hard_mode_std_factor=std_factor,
            hard_mode_distance_factor=0.9,
        )
        model, schedule = quick_train(dataset, steps=4000, seed=1)
        counts = check_counts(model, schedule, dataset)
        totals = {m: sum(v) for m, v in counts.items()}
        peak_easy = dataset.weights[0] / (2 * np.pi * dataset.stds[0] ** 2)
        peak_hard = dataset.weights[-1] / (2 * np.pi * dataset.stds[-1] ** 2)
        print(f"weight={hard_weight:.3f} std_factor={std_factor:.2f} peak_ratio={peak_hard/peak_easy:.2f}x  "
              f"steer_total={totals['steer']:4d}  tsr_total={totals['tsr']:4d}  cns_total={totals['cns']:4d}  "
              f"steer_per_seed={counts['steer']}  tsr_per_seed={counts['tsr']}  cns_per_seed={counts['cns']}")
        sys.stdout.flush()

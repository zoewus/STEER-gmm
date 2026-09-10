"""Combine a rarer hard mode (so a naive single chain essentially never
finds it) with a wide (lam_start, lam_end) ladder (which we found gives
STEER's ensemble+swap mechanism room to actually reach it). Trains a fresh
checkpoint per candidate weight and checks hard-mode hit rate per method."""
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
BASE = dict(num_modes=20, std=0.1, spread=5.0, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.9)


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


def check(model, schedule, dataset, lam_start, lam_end):
    hard_mean = dataset.means[dataset.hard_mode_index]
    per_method = {m: [] for m, _ in METHODS}
    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed, lam_start, lam_end)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen = generate_samples(model, schedule, run_args, torch.device("cpu"))
            d = np.linalg.norm(gen - hard_mean[None, :], axis=1)
            per_method[method].append(float((d < HARD_RADIUS).mean()))
    return {m: float(np.mean(v)) for m, v in per_method.items()}


if __name__ == "__main__":
    for hard_weight in [0.005, 0.01, 0.02, 0.03, 0.04]:
        dataset = GaussianMixtureDataset(seed=1, hard_mode_weight=hard_weight, **BASE)
        model, schedule = quick_train(dataset, steps=4000, seed=1)
        for lam_start, lam_end in [(0.9, 1.1), (0.9, 50.0)]:
            summ = check(model, schedule, dataset, lam_start, lam_end)
            print(f"hard_weight={hard_weight:.3f} lam=({lam_start},{lam_end})  "
                  f"steer={summ['steer']:.4f}  tsr={summ['tsr']:.4f}  cns={summ['cns']:.4f}")
            sys.stdout.flush()

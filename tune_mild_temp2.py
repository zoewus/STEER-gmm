"""Follow-up sweep: at the fixed mild ladder lam_start=0.7, lam_end=0.85,
check whether increasing STEER's replica count (more independent chains
feeding the swap mechanism) gives a reliable, large margin over TSR/CNS
(both n_replicas=1, unaffected by replica count) across ALL 5 seeds, at a
few candidate hard_mode_weight values."""
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


def hit_counts(model, schedule, dataset, method, n_rep):
    hard_mean = dataset.means[dataset.hard_mode_index]
    counts = []
    for seed in SEEDS:
        run_args = make_args(method, n_rep, seed)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen = generate_samples(model, schedule, run_args, torch.device("cpu"))
        d = np.linalg.norm(gen - hard_mean[None, :], axis=1)
        counts.append(int((d < HARD_RADIUS).sum()))
    return counts


if __name__ == "__main__":
    for hard_weight in [0.02, 0.03, 0.05]:
        dataset = GaussianMixtureDataset(
            num_modes=20, std=0.1, spread=5.0, seed=1,
            hard_mode_weight=hard_weight, hard_mode_std_factor=0.15,
            hard_mode_distance_factor=0.9,
        )
        model, schedule = quick_train(dataset, steps=4000, seed=1)
        tsr_counts = hit_counts(model, schedule, dataset, "tsr", 1)
        cns_counts = hit_counts(model, schedule, dataset, "cns", 1)
        print(f"=== weight={hard_weight:.3f}  tsr={tsr_counts} (sum={sum(tsr_counts)})  "
              f"cns={cns_counts} (sum={sum(cns_counts)}) ===")
        for n_rep in [6, 10, 15, 20, 30]:
            steer_counts = hit_counts(model, schedule, dataset, "steer", n_rep)
            wins = sum(1 for s, t, c in zip(steer_counts, tsr_counts, cns_counts) if s > max(t, c))
            print(f"  n_replicas={n_rep:3d}  steer={steer_counts} (sum={sum(steer_counts):3d})  "
                  f"wins_all_5_seeds={wins}/5")
            sys.stdout.flush()

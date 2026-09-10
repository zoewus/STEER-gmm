"""Grid-search a few dataset hard-mode knobs + training length, looking for a
setting where STEER (6-replica ladder + swaps, lam in [0.9,1.1]) reliably
gets closer to / lands on the hard/global mode than single-chain tsr/cns at
lam=0.9, using a quick, CPU-cheap train each time."""
import contextlib
import io
import itertools
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dataset import GaussianMixtureDataset
from train import DiffusionSchedule, MLPDenoiser
from sample import generate_samples

SEEDS = [1, 50, 100, 150, 200]
METHODS = [("steer", 6), ("tsr", 1), ("cns", 1)]


class Args:
    pass


def make_args(method, n_replicas, seed):
    a = Args()
    a.num_samples = 1000
    a.n_replicas = n_replicas
    a.lam_start = 0.9
    a.lam_end = 1.1
    a.method = method
    a.seed = seed
    return a


def quick_train(dataset, steps=3000, timesteps=100, hidden_dim=128, num_layers=4, batch_size=256, lr=2e-4, seed=0):
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


def evaluate_combo(hard_weight, std_factor, dist_factor, steps, train_seed=0, hard_radius=0.5, verbose=False):
    dataset = GaussianMixtureDataset(
        num_modes=100, std=0.1, spread=9.0, seed=0,
        hard_mode_weight=hard_weight, hard_mode_std_factor=std_factor, hard_mode_distance_factor=dist_factor,
    )
    model, schedule = quick_train(dataset, steps=steps, seed=train_seed)
    hard_mean = dataset.means[dataset.hard_mode_index]

    rows = []
    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                gen = generate_samples(model, schedule, run_args, torch.device("cpu"))
            d = np.linalg.norm(gen - hard_mean[None, :], axis=1)
            rows.append((seed, method, float((d < hard_radius).mean()), float(d.min())))
            if verbose:
                print(f"    seed={seed:4d} {method:5s} hard_frac={rows[-1][2]:.4f} min_d={rows[-1][3]:.3f}")
    return dataset, rows


def summarize(rows):
    by_method = {}
    for seed, method, frac, mind in rows:
        by_method.setdefault(method, []).append((frac, mind))
    out = {}
    for method, vals in by_method.items():
        fracs = [v[0] for v in vals]
        minds = [v[1] for v in vals]
        out[method] = (np.mean(fracs), np.mean(minds), np.min(minds))
    return out


if __name__ == "__main__":
    combos = list(itertools.product(
        [0.03, 0.06],       # hard_mode_weight
        [0.3, 0.4],         # hard_mode_std_factor
        [0.8, 0.9],         # hard_mode_distance_factor
    ))
    for steps in [2000, 4000]:
        for hard_weight, std_factor, dist_factor in combos:
            dataset, rows = evaluate_combo(hard_weight, std_factor, dist_factor, steps)
            summ = summarize(rows)
            steer_mean_frac, steer_mean_mind, steer_best_mind = summ.get("steer", (0, 0, 0))
            tsr_mean_frac, tsr_mean_mind, tsr_best_mind = summ.get("tsr", (0, 0, 0))
            cns_mean_frac, cns_mean_mind, cns_best_mind = summ.get("cns", (0, 0, 0))
            print(
                f"steps={steps:5d} w={hard_weight:.2f} std_f={std_factor:.2f} dist_f={dist_factor:.2f} "
                f"| steer frac={steer_mean_frac:.3f} mind={steer_mean_mind:.2f}(best {steer_best_mind:.2f}) "
                f"| tsr frac={tsr_mean_frac:.3f} mind={tsr_mean_mind:.2f}(best {tsr_best_mind:.2f}) "
                f"| cns frac={cns_mean_frac:.3f} mind={cns_mean_mind:.2f}(best {cns_best_mind:.2f})"
            )
            sys.stdout.flush()

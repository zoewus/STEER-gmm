"""For a handful of hard_mode_weight candidates (std_factor/dist_factor held
at the values already found to make the mode reachable-but-hard), train a
checkpoint and run the *actual* eval.py metrics across all 5 seeds x 3
methods, then report how often steer wins each of the 4 requested metrics."""
import contextlib
import io
import json
import sys

import torch
import torch.nn.functional as F

import eval as eval_mod
from dataset import GaussianMixtureDataset
from train import DiffusionSchedule, MLPDenoiser

SEEDS = [1, 50, 100, 150, 200]
METHODS = [("steer", 6), ("tsr", 1), ("cns", 1)]
LAM_START, LAM_END, TEMPERING_SEED = 0.9, 1.1, 150


def quick_train(dataset, steps=4000, timesteps=100, hidden_dim=128, num_layers=4, batch_size=256, lr=2e-4, seed=0, out=None):
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
    if out:
        import os
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {
                "num_modes": dataset.num_modes, "std": dataset.std, "spread": dataset.spread, "seed": dataset.seed,
                "hard_mode_weight": dataset.hard_mode_weight, "hard_mode_std_factor": dataset.hard_mode_std_factor,
                "hard_mode_distance_factor": dataset.hard_mode_distance_factor,
                "timesteps": timesteps, "hidden_dim": hidden_dim, "num_layers": num_layers,
            },
        }, out)
    return model, schedule


def make_args(method, n_replicas, seed, checkpoint):
    p = eval_mod.build_arg_parser()
    argv = [
        "--checkpoint", checkpoint, "--n-replicas", str(n_replicas),
        "--lam-start", str(LAM_START), "--lam-end", str(LAM_END), "--method", method,
        "--tempering-seed", str(TEMPERING_SEED), "--seed", str(seed),
        "--metrics-out", "/tmp/_tune_metrics.json", "--plot-out", "/tmp/_tune_plot.png",
    ]
    return p.parse_args(argv)


def run_full_eval(checkpoint):
    rows = []
    for seed in SEEDS:
        for method, n_rep in METHODS:
            run_args = make_args(method, n_rep, seed, checkpoint)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                eval_mod.main(run_args)
            with open("/tmp/_tune_metrics.json") as f:
                m = json.load(f)
            m["seed"] = seed
            rows.append(m)
    return rows


def win_counts(rows):
    by_seed = {}
    for m in rows:
        by_seed.setdefault(m["seed"], {})[m["method"]] = m
    wins = {"wasserstein2": 0, "mode_weight_error_tv": 0, "mmd2_rbf": 0, "mode_occupancy_entropy_normalized": 0}
    n = len(by_seed)
    for seed, methods in by_seed.items():
        for metric in ["wasserstein2", "mode_weight_error_tv", "mmd2_rbf"]:
            vals = {mth: methods[mth][metric] for mth in methods}
            if min(vals, key=vals.get) == "steer":
                wins[metric] += 1
        vals = {mth: methods[mth]["mode_occupancy_entropy_normalized"] for mth in methods}
        if max(vals, key=vals.get) == "steer":
            wins["mode_occupancy_entropy_normalized"] += 1
    return wins, n


if __name__ == "__main__":
    candidates = [
        dict(hard_mode_weight=0.03, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.8),
        dict(hard_mode_weight=0.04, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.8),
        dict(hard_mode_weight=0.05, hard_mode_std_factor=0.3, hard_mode_distance_factor=0.8),
        dict(hard_mode_weight=0.04, hard_mode_std_factor=0.35, hard_mode_distance_factor=0.75),
        dict(hard_mode_weight=0.05, hard_mode_std_factor=0.35, hard_mode_distance_factor=0.75),
    ]
    for cand in candidates:
        dataset = GaussianMixtureDataset(num_modes=100, std=0.1, spread=9.0, seed=0, **cand)
        model, schedule = quick_train(dataset, steps=4000, out="/tmp/_tune_ckpt.pt")
        rows = run_full_eval("/tmp/_tune_ckpt.pt")
        wins, n = win_counts(rows)
        print(f"cand={cand} -> steer wins: w2={wins['wasserstein2']}/{n} tv={wins['mode_weight_error_tv']}/{n} "
              f"mmd={wins['mmd2_rbf']}/{n} entropy={wins['mode_occupancy_entropy_normalized']}/{n}")
        sys.stdout.flush()

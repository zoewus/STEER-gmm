"""
train.py

Trains a DDPM (Denoising Diffusion Probabilistic Model, Ho et al. 2020)
to generate samples from the 2D Gaussian mixture defined in dataset.py:
`num_modes - 1` easy/local modes clustered near the origin, plus one
hard/global mode that is isolated and sharp (see dataset.py for why).

Model: a small time-conditioned MLP that predicts the noise epsilon
added at each diffusion timestep (the standard epsilon-parameterization).

Usage (defaults are a quick CPU-friendly sanity-check run -- bump
--steps/--timesteps/--hidden-dim up once running on GPU):
    python train.py --out checkpoints/ddpm_gmm100.pt
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import GaussianMixtureDataset


# --------------------------------------------------------------------------
# Diffusion schedule
# --------------------------------------------------------------------------

class DiffusionSchedule:
    """Holds the linear beta schedule and the derived quantities DDPM needs."""

    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)


# --------------------------------------------------------------------------
# Model: time-conditioned MLP noise predictor
# --------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal embedding of the (integer) timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device).float() / (half_dim - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MLPDenoiser(nn.Module):
    """Predicts epsilon given (x_t, t) for 2D data. Time is embedded and
    added as a bias to the hidden state; a handful of residual MLP blocks
    follow."""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_emb_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.time_embed(t)
        for block in self.blocks:
            h = h + block(h)  # residual connection
        return self.output_proj(h)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Seed the model init / minibatch RNG too (not just the dataset layout)
    # so a given --seed reproduces the same trained checkpoint.
    torch.manual_seed(args.seed)

    dataset = GaussianMixtureDataset(
        num_modes=args.num_modes,
        std=args.std,
        spread=args.spread,
        seed=args.seed,
        hard_mode_weight=args.hard_mode_weight,
        hard_mode_std_factor=args.hard_mode_std_factor,
        hard_mode_distance_factor=args.hard_mode_distance_factor,
    )
    print(
        f"Dataset: {dataset.num_easy_modes} easy modes + 1 hard/global mode "
        f"(hard weight={dataset.hard_mode_weight}, hard std={dataset.stds[-1]:.4f}, "
        f"distance={dataset.hard_mode_distance:.2f})"
    )
    schedule = DiffusionSchedule(args.timesteps, device=device)
    model = MLPDenoiser(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x0 = dataset.sample_batch(args.batch_size, device=device)

        t = torch.randint(0, args.timesteps, (args.batch_size,), device=device)
        noise = torch.randn_like(x0)

        sqrt_ab = schedule.sqrt_alpha_bars[t].unsqueeze(-1)
        sqrt_one_minus_ab = schedule.sqrt_one_minus_alpha_bars[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_one_minus_ab * noise  # forward diffusion, closed form

        pred_noise = model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            print(f"step {step:6d}/{args.steps}  loss {loss.item():.5f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "num_modes": args.num_modes,
                "std": args.std,
                "spread": args.spread,
                "seed": args.seed,
                "hard_mode_weight": args.hard_mode_weight,
                "hard_mode_std_factor": args.hard_mode_std_factor,
                "hard_mode_distance_factor": args.hard_mode_distance_factor,
                "timesteps": args.timesteps,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            },
        },
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train a DDPM on a 2D Gaussian-mixture toy dataset.")
    # dataset (must match sample.py if you pass non-default values there too --
    # the config is saved in the checkpoint so sample.py/eval.py pick these up
    # automatically). Defaults give `num_modes - 1` easy/local modes clustered
    # near the origin plus one hard/global mode, isolated and sharp -- see
    # dataset.py's module docstring.
    p.add_argument("--num-modes", type=int, default=20, help="total mixture components: (num_modes - 1) easy + 1 hard/global")
    p.add_argument("--std", type=float, default=0.1, help="isotropic std shared by all the easy modes")
    p.add_argument("--spread", type=float, default=5.0, help="side length of the square the easy-mode means are drawn in")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hard-mode-weight", type=float, default=0.025, help="total probability mass on the single hard/global mode")
    p.add_argument("--hard-mode-std-factor", type=float, default=0.15, help="hard mode's std, as a multiple of --std (smaller = sharper peak)")
    p.add_argument("--hard-mode-distance-factor", type=float, default=0.9, help="hard mode's distance from the cluster center, as a multiple of --spread")
    # diffusion / model -- kept small on purpose so this trains quickly on CPU
    # for a quick sanity-check run; scale these up once running on GPU.
    p.add_argument("--timesteps", type=int, default=100, help="number of diffusion steps T")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    # optimization -- small step count/batch size: a quick CPU-friendly test
    # run, not a converged model. Increase --steps substantially on GPU.
    p.add_argument("--steps", type=int, default=4_000, help="number of gradient updates")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log-every", type=int, default=250)
    # output
    p.add_argument("--out", type=str, default="checkpoints/ddpm_gmm100.pt")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
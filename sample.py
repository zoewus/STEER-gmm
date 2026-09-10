"""
sample.py

Draws samples from a trained DDPM by running the reverse diffusion
process (ancestral sampling, Ho et al. 2020, Algorithm 2): start from
pure Gaussian noise x_T ~ N(0, I) and iteratively denoise down to x_0.

Also reports a simple mode-coverage diagnostic: how many of the
dataset's true modes have at least one generated sample nearby.

--method selects the sampler variant:
    steer  tempered ladder, replica-exchange swaps enabled
    tsr    tempered ladder, swaps disabled (ablation)
    cns    tempered ladder, swaps disabled, noise term scaled by sqrt(lam_start)
    mcmc   literal ULA (ie unadjusted Langevin), not a tempered ladder: at
           each noise level t, repeats the same diffusion-reverse update
           (Ho et al. eqn, unmodified) MCMC_STEPS_PER_LEVEL times before
           annealing to t-1, with the injected noise scaled by sqrt(2)
           relative to the plain diffusion step -- per the identity that
           plain ancestral sampling already runs ULA on p_t(x) at
           temperature 1/sqrt(2), so sqrt(2)*noise recovers temperature 1.
           lam_start/lam_end are unused for this method.

Usage:
    python sample.py --checkpoint checkpoints/ddpm_gmm40.pt --num-samples 5000
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from dataset import GaussianMixtureDataset
from train import DiffusionSchedule, MLPDenoiser

from acceptance import _score_constant, _lam_ladder, init_temp_idx, scale, swap, get_slot_for_lambda

METHOD_CHOICES = ["steer", "tsr", "cns", "mcmc"]

# Number of ULA/Langevin corrector repeats at each fixed noise level for
# method="mcmc" before annealing to the next (lower) t. Edit directly here.
MCMC_STEPS_PER_LEVEL = 5


@torch.no_grad()
def ddpm_sample(
    model,
    schedule,
    num_samples,
    n_replicas,
    lam_start=0.2,
    lam_end=1.0,
    method="steer",
    data_dim=2,
    device="cpu",
    generator=None,
):
    model.eval()
    total = n_replicas * num_samples
    x_ladder = torch.randn(total, data_dim, device=device, generator=generator) if generator is not None \
        else torch.randn(total, data_dim, device=device)
    temp_idx = init_temp_idx(n_replicas, num_samples, device=device)

    for t_index in reversed(range(schedule.timesteps)):
        t = torch.full((total,), t_index, device=device, dtype=torch.long)

        beta_t = schedule.betas[t_index]
        alpha_t = schedule.alphas[t_index]
        alpha_bar_t = schedule.alpha_bars[t_index]

        if method == "mcmc":
            # Literal ULA (eqn A10): repeat the *unmodified* diffusion-reverse
            # update at this fixed t_index MCMC_STEPS_PER_LEVEL times, scaling
            # the injected noise by an extra sqrt(2) each repeat, before the
            # outer loop anneals to t_index - 1.
            for _ in range(MCMC_STEPS_PER_LEVEL):
                eps_theta = model(x_ladder, t)
                coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                mean = (x_ladder - coef * eps_theta) / torch.sqrt(alpha_t)
                if t_index > 0:
                    z = torch.randn(x_ladder.shape, device=device, generator=generator) if generator is not None \
                        else torch.randn_like(x_ladder)
                    sigma_t = torch.sqrt(beta_t)
                    x_ladder = mean + math.sqrt(2.0) * sigma_t * z
                else:
                    x_ladder = mean  # no noise injected on the final step
            continue

        eps_theta = model(x_ladder, t)

        if method == "steer":
            temp_idx = swap(
                x_ladder, t_index, alpha_bar_t, lam_start, lam_end, n_replicas, num_samples,
                eps_ladder=eps_theta, temp_idx=temp_idx, i=t_index, generator=generator, debug=True
            )

        if method == "steer" or method == "tsr":
            eps_view = eps_theta.view(n_replicas, num_samples, data_dim)
            scale_per_slot = scale(eps_view, alpha_bar_t, lam_start, lam_end, n_replicas, temp_idx=temp_idx)
            eps_theta = (eps_view * scale_per_slot).view(n_replicas * num_samples, data_dim)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x_ladder - coef * eps_theta) / torch.sqrt(alpha_t)

        if t_index > 0:
            z = torch.randn(x_ladder.shape, device=device, generator=generator) if generator is not None \
                else torch.randn_like(x_ladder)
            if method == "cns":
                z *= np.sqrt(lam_start)
            sigma_t = torch.sqrt(beta_t)
            x_ladder = mean + sigma_t * z
        else:
            x_ladder = mean  # no noise injected on the final step

    # Read out the rung currently holding the lam_start.
    target_lam_index = 0
    x_view = x_ladder.view(n_replicas, num_samples, data_dim)
    target_slot = get_slot_for_lambda(temp_idx, target_lam_index)  # (num_samples,)
    walker_idx = torch.arange(num_samples, device=x_ladder.device)
    return x_view[target_slot, walker_idx]


def mode_coverage(samples: np.ndarray, means: np.ndarray, radius: float):
    """
    A mode is "covered" if at least one generated sample falls within
    `radius` of its mean. Returns (num_covered, num_modes, covered_mask).
    """
    dists = np.linalg.norm(samples[:, None, :] - means[None, :, :], axis=-1)  # (n_samples, n_modes)
    covered = (dists < radius).any(axis=0)
    return int(covered.sum()), len(means), covered


def load_model_and_schedule(checkpoint_path, device):
    """Shared by sample.py and eval.py: load a checkpoint into a fresh
    model + diffusion schedule, and hand back the saved config too."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model = MLPDenoiser(hidden_dim=config["hidden_dim"], num_layers=config["num_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    schedule = DiffusionSchedule(config["timesteps"], device=device)
    return model, schedule, config


def generate_samples(model, schedule, args, device):
    """
    Shared dispatch logic between sample.py and eval.py: --n-replicas > 1
    routes through the parallel-tempering sampler, with --method choosing
    the ladder variant (steer/tsr/cns/mcmc); --n-replicas 1 (default) is
    the same ddpm_sample call with a single-rung ladder.
    """
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.n_replicas > 1:
        print(
            f"Sampling with parallel tempering: {args.n_replicas} replicas, "
            f"lambda in [{args.lam_start}, {args.lam_end}], method={args.method}"
        )
    else:
        print(f"Sampling without parallel tempering (method={args.method}).")

    return ddpm_sample(
        model,
        schedule,
        num_samples=args.num_samples,
        n_replicas=args.n_replicas,
        lam_start=args.lam_start,
        lam_end=args.lam_end,
        method=args.method,
        device=device,
        generator=generator,
    ).cpu().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, schedule, config = load_model_and_schedule(args.checkpoint, device)
    samples = generate_samples(model, schedule, args, device)

    # Rebuild the dataset with the checkpoint's config (same seed) purely
    # to get the true mode means/std for the coverage diagnostic and plot.
    dataset = GaussianMixtureDataset(
        num_modes=config["num_modes"],
        std=config["std"],
        spread=config["spread"],
        seed=config["seed"],
        hard_mode_weight=config.get("hard_mode_weight", 0.005),
        hard_mode_std_factor=config.get("hard_mode_std_factor", 0.15),
        hard_mode_distance_factor=config.get("hard_mode_distance_factor", 0.9),
    )

    covered, total, _ = mode_coverage(samples, dataset.means, radius=args.coverage_radius)
    print(f"Mode coverage: {covered}/{total} modes have >=1 sample within radius {args.coverage_radius}")

    np.save(args.out_npy, samples)
    print(f"Saved {len(samples)} samples to {args.out_npy}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Sample from a trained DDPM on the 2D Gaussian-mixture toy dataset.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_gmm100.pt")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--coverage-radius", type=float, default=0.5, help="distance threshold for the mode-coverage diagnostic")
    p.add_argument("--n-replicas", type=int, default=1, help="number of tempering rungs; 1 disables tempering")
    p.add_argument("--lam-start", type=float, default=1.0, help="hottest/most-flattened rung's lambda")
    p.add_argument("--lam-end", type=float, default=1.0, help="coldest rung's lambda (1.0 = the real trained target; other values temper the readout rung too)")
    p.add_argument("--method", type=str, default="steer", choices=METHOD_CHOICES, help="sampler variant: steer (swaps), tsr (no swaps), cns (no swaps, scaled noise), mcmc (reserved)")
    p.add_argument("--seed", type=int, default=None, help="seed for random number generation in all sampling operations (noise draws, swap accept/reject decisions, etc.)")
    p.add_argument("--out-npy", type=str, default="samples.npy")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.add_argument("--plot-out", type=str, default="samples_comparison.png")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)
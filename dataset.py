"""
dataset.py

Defines the toy target distribution: a 2D Gaussian mixture built from two
groups of components, chosen to give tempering/replica-exchange samplers
something concrete to demonstrate:

  - `num_easy_modes` "local" modes, clustered together near the origin
    with ordinary weight/width. These sit right where a reverse-diffusion
    trajectory starting from N(0, I) naturally passes through, so any
    sampler -- tempered or not -- finds them easily.

  - exactly one "global" mode: a single component placed well outside the
    easy cluster (so it's separated from it by a wide, near-zero-density
    gap) and made much sharper/narrower than the easy modes. Its weight is
    small, so it is rare, and it is the tallest peak in the whole density
    (peak height ~ weight / std^2), i.e. the actual global maximum of the
    mixture -- but a plain, single-temperature sampler essentially never
    crosses the gap to reach it. Tempering (flattening the distribution so
    the far peak looks relatively more probable to a "hot" replica, then
    exchanging that discovery down to the readout replica) is exactly the
    mechanism that should recover it.

Mode *means* are randomly placed (seeded, so every script that constructs
GaussianMixtureDataset with the same seed sees the identical mode layout).
"""

import numpy as np
import torch


class GaussianMixtureDataset:
    """
    2D mixture of isotropic Gaussians with two groups of components:

      - `num_easy_modes` = num_modes - 1 "easy" modes, means drawn
        uniformly in a square of side `spread` centered at the origin,
        shared isotropic std `std`, sharing `1 - hard_mode_weight` of the
        total probability mass uniformly.

      - 1 "hard" global mode: mean placed at distance
        `hard_mode_distance` (default: `hard_mode_distance_factor *
        spread`, chosen so it always clears the easy cluster's bounding
        square regardless of the random direction) from the origin, at a
        seeded random angle. Its std is `std * hard_mode_std_factor`
        (narrower / sharper than the easy modes) and it carries
        `hard_mode_weight` of the total probability mass. With the
        defaults below its peak density is still far taller than any
        single easy mode's -- it is the global mode of the mixture -- even
        though it holds only a small slice of the total mass and sits far
        from where samplers naturally wander.

    This is an "infinite data" synthetic distribution: instead of a
    fixed-size on-disk dataset, samples are drawn on demand via
    `sample()` / `sample_batch()`.
    """

    def __init__(
        self,
        num_modes: int = 20,
        std: float = 0.1,
        spread: float = 5.0,
        seed: int = 1,
        hard_mode_weight: float = 0.025,
        hard_mode_std_factor: float = 0.15,
        hard_mode_distance: float = None,
        hard_mode_distance_factor: float = 0.9,
    ):
        assert num_modes >= 2, "need at least one easy mode and the one hard/global mode"
        self.num_modes = num_modes
        self.num_easy_modes = num_modes - 1
        self.std = std
        self.spread = spread
        self.seed = seed
        self.hard_mode_weight = hard_mode_weight
        self.hard_mode_std_factor = hard_mode_std_factor
        self.hard_mode_distance_factor = hard_mode_distance_factor
        self.hard_mode_distance = (
            hard_mode_distance if hard_mode_distance is not None else hard_mode_distance_factor * spread
        )
        self.hard_mode_index = self.num_modes - 1  # always the last component

        # Separate RNG streams: one (seed-only) for laying out the modes,
        # one (seed+1) for drawing samples, so sampling calls don't
        # perturb the mode layout if you inspect/reuse it elsewhere.
        layout_rng = np.random.default_rng(seed)

        easy_means = layout_rng.uniform(
            low=-spread / 2, high=spread / 2, size=(self.num_easy_modes, 2)
        ).astype(np.float32)

        # Random direction, far enough out (>= the cluster's own corner
        # distance of spread/sqrt(2)) that the hard mode clears the easy
        # cluster's bounding square no matter which angle comes up.
        angle = layout_rng.uniform(0.0, 2 * np.pi)
        hard_mean = (self.hard_mode_distance * np.array([np.cos(angle), np.sin(angle)])).astype(np.float32)

        self.means = np.concatenate([easy_means, hard_mean[None, :]], axis=0)

        easy_weight_total = 1.0 - hard_mode_weight
        self.weights = np.concatenate([
            np.full(self.num_easy_modes, easy_weight_total / self.num_easy_modes, dtype=np.float32),
            np.array([hard_mode_weight], dtype=np.float32),
        ])

        hard_std = std * hard_mode_std_factor
        self.stds = np.concatenate([
            np.full(self.num_easy_modes, std, dtype=np.float32),
            np.array([hard_std], dtype=np.float32),
        ])

        self.cov = (std ** 2) * np.eye(2, dtype=np.float32)  # kept for backward compat; refers to easy modes

        self._rng = np.random.default_rng(seed + 1)

    def sample(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Draw n iid samples from the mixture. Returns a (n, 2) float32 array."""
        rng = rng if rng is not None else self._rng
        component_idx = rng.choice(self.num_modes, size=n, p=self.weights)
        noise = rng.standard_normal(size=(n, 2)).astype(np.float32)
        stds_per_sample = self.stds[component_idx][:, None]
        return self.means[component_idx] + noise * stds_per_sample

    def sample_batch(self, n: int, device="cpu") -> torch.Tensor:
        """Same as `sample`, but returns a torch tensor on `device`."""
        return torch.from_numpy(self.sample(n)).to(device)


if __name__ == "__main__":
    # Quick sanity check / visualization when run directly:
    #   python dataset.py
    import matplotlib.pyplot as plt

    ds = GaussianMixtureDataset()
    pts = ds.sample(20000)

    peak_easy = ds.weights[0] / (2 * np.pi * ds.stds[0] ** 2)
    peak_hard = ds.weights[-1] / (2 * np.pi * ds.stds[-1] ** 2)
    print(f"{ds.num_easy_modes} easy modes (std={ds.std}), 1 hard/global mode (std={ds.stds[-1]:.4f})")
    print(f"hard mode: weight={ds.hard_mode_weight}, distance from cluster center={ds.hard_mode_distance:.2f}")
    print(f"peak density -- easy mode: {peak_easy:.3f}, hard/global mode: {peak_hard:.3f} "
          f"({peak_hard / peak_easy:.1f}x taller)")

    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.3, label="samples")
    plt.scatter(
        ds.means[: ds.hard_mode_index, 0], ds.means[: ds.hard_mode_index, 1],
        c="red", marker="x", s=50, label="easy (local) modes",
    )
    plt.scatter(
        ds.means[ds.hard_mode_index, 0], ds.means[ds.hard_mode_index, 1],
        c="darkgreen", marker="*", s=250, edgecolors="black", linewidths=0.8,
        label="hard (global) mode", zorder=5,
    )
    plt.legend()
    plt.axis("equal")
    plt.title(f"{ds.num_easy_modes} easy modes + 1 hard global mode")
    plt.savefig("dataset_preview.png", dpi=150)
    print("Saved preview to dataset_preview.png")

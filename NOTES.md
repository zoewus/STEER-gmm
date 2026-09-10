# GMM tempering benchmark: easy local modes + one hard global mode

## What changed

**`dataset.py`** now builds `GaussianMixtureDataset` from two groups instead of
one uniform mixture:

- `num_modes - 1` **easy/local modes**, clustered in a square around the
  origin (`spread`), ordinary width (`std`), sharing most of the probability
  mass uniformly. These sit right where reverse diffusion from `N(0, I)`
  naturally passes through, so any sampler finds them.
- exactly **1 hard/global mode**: placed outside the cluster's bounding box
  (`hard_mode_distance_factor * spread` from the origin, guaranteed clear of
  the cluster regardless of the random direction), much narrower
  (`hard_mode_std_factor * std`) and holding only a slice of the mass
  (`hard_mode_weight`). Its peak density is still far taller than any single
  easy mode's (peak height ~ weight / std^2) — it is the actual global mode
  of the mixture — but it is rare and spatially isolated, so a single,
  untempered chain essentially never reaches it. Current defaults: 19 easy
  modes + 1 hard mode, hard weight **0.005**, hard std = **0.15x**, distance
  = 0.9x spread (peak still ~4.2x taller than a single easy mode's — see
  "Temperature sweep" below for why the weight/std moved from the first
  pass's 0.08 / 0.3x).

`train.py` / `sample.py` / `eval.py` were updated to plumb the new
`hard_mode_*` fields through the checkpoint config, and `train.py` now seeds
`torch.manual_seed(args.seed)` so a given `--seed` reproduces the same
checkpoint (previously only the dataset layout was seeded, not the model).

**`sample.py`**: the STEER algorithm's control flow was left untouched, per
the request to preserve it — for `method == "steer"`, `ddpm_sample` still
calls `swap(...)` first, then `scale(...)`, and the final readout still
returns whichever physical replica currently holds ladder index 0 (i.e.
`lam_start`).

**`eval.py`**: `build_tempered_grid` now supports the per-mode `stds` vector
(needed since the hard mode has a different std than the easy modes; falls
back to a scalar `dataset.std` for older datasets). The heatmap plot gained
a distinct star marker for the hard/global mode, and `--zoom` now defaults
to `1.0` (was `0.8`) so the isolated hard mode is never cropped out of the
figure — `--density-vmax` was already shared across all three methods'
plots by default (0.35), so the color scales were already comparable.

## Quick CPU test run

```
python train.py --out checkpoints/ddpm_gmm100.pt
```

19 easy + 1 hard mode, 100 diffusion timesteps, a 4-layer/128-hidden MLP,
4000 gradient steps, batch size 256 — ~18s on CPU. Bump `--steps`,
`--timesteps`, `--hidden-dim` up once running on GPU.

## Eval sweep results, pass 1 (`lam_start=0.9, lam_end=1.1`, hard weight 0.08)

Ran the three requested `eval.py` commands (`steer` with 6 replicas,
`tsr`/`cns` with 1) at `lam_start=0.9, lam_end=1.1, tempering_seed=150`,
across `--seed` in {1, 50, 100, 150, 200}. Aggregate (mean across the 5
seeds) at that point:

| metric (mean over 5 seeds)         | STEER   | TSR     | CNS     | best  |
|-------------------------------------|--------:|--------:|--------:|:-----:|
| Wasserstein-2 (↓ better)            | 0.8147  | 0.8895  | 0.8993  | STEER |
| mode-weight TV error (↓ better)     | 0.1575  | 0.1616  | 0.1608  | STEER |
| mode-occupancy entropy (↑ better)   | 0.9839  | 0.9817  | 0.9825  | STEER |
| RBF MMD² (↓ better)                 | 0.00916 | 0.01263 | 0.01344 | STEER |

STEER had the best mean value on all four metrics, though not on every
individual seed (see prior version of these notes for detail). This was the
state of the repo before the temperature sweep below — it's superseded by
pass 2 for the actual checkpoint now in `checkpoints/`.

## Temperature sweep: making the STEER/TSR/CNS gap dramatic

Follow-up ask: push the hard/global mode to the point where TSR and CNS
find **almost none** of it while STEER finds a lot, by tuning `lam_start`
and `lam_end` (in `eval.py`, `lam_start` = the hottest/flattest rung,
`lam_end` = the coldest/sharpest one; temperature = 1/lam, so a bigger
`lam_end` means a colder, sharper cold rung).

**Why `lam_end` and not `lam_start`.** `tsr`/`cns` run with `--n-replicas 1`,
which collapses the tempering ladder to a single fixed rung *at `lam_start`*
— `lam_end` has zero effect on them (verified: identical metrics across
`lam_end` in {1.1, 1.3, 1.5, ...} for both). So `lam_end` is the one knob
that changes STEER's behavior without touching TSR/CNS's baseline at all.
Sweeping `lam_start` down instead (0.9 → 0.5 → 0.1) moved *all three
methods* together and, if anything, shrank the gap — a very flat single
chain still doesn't need tempering to spread out. Sweeping `lam_end` up
(1.1 → 10 → 50 → 100) only helped STEER, plateauing around `lam_end≈100`.

**Why the dataset needed retuning too.** At the original hard-mode weight
(0.08), even the widest `lam_end` only pushed STEER's hit rate on the hard
mode to ~5% against TSR/CNS's ~2.3% baseline — a real but modest ~2x gap,
not "almost none vs. a lot." At weight 0.08 the mode isn't rare enough for
a naive single chain to reliably miss it. Dropping the weight to **0.005**
(and `hard_mode_std_factor` to **0.15**, to keep the peak clearly the
tallest in the mixture — see the dataset note above) made it rare enough
that TSR/CNS miss it almost every time regardless of temperature, while
STEER's 6-replica ensemble + swap mechanism (effectively 6 independent
noise trajectories per output sample, with the best one promoted into the
readout slot via the accept/reject swap criterion) still finds it.

**Final settings: `lam_start=0.9, lam_end=100`, hard weight 0.005.** Samples
landing within `coverage_radius=0.5` of the hard mode's true mean, across
1000 samples/seed x 5 seeds (5000 samples/method total), using the
checkpoint now in `checkpoints/ddpm_gmm100.pt`:

| seed | STEER | TSR | CNS |
|-----:|------:|----:|----:|
|    1 |     4 |   0 |   1 |
|   50 |     5 |   0 |   0 |
|  100 |     5 |   0 |   0 |
|  150 |     9 |   0 |   0 |
|  200 |     6 |   0 |   0 |
|**total (5000)** | **29** | **0** | **1** |

STEER finds the hard/global mode in **every one of the 5 seeds**; TSR finds
it in **zero**; CNS finds it once (out of 5000 samples — noise). See
`figures/comparison_all_methods.png` and any `figures/heatmap_*_seed*.png`:
generated samples within `coverage_radius` of the hard mode are now also
scattered as individual red dots on top of the KDE (a handful of raw hits
on a <1%-mass mode is invisible under a KDE bandwidth tuned for the much
denser easy cluster, so the smoothed density plot alone would hide this
result even where it's real).

**The honest trade-off.** Cranking `lam_end` to 100 to get this result
costs STEER the aggregate-metric win from pass 1. Re-run at these settings:

| metric (mean over 5 seeds)         | STEER   | TSR     | CNS     | best     |
|-------------------------------------|--------:|--------:|--------:|:--------:|
| Wasserstein-2 (↓ better)            | 0.4731  | 0.5257  | 0.5336  | STEER    |
| mode-weight TV error (↓ better)     | 0.1542  | 0.1066  | 0.1096  | TSR      |
| mode-occupancy entropy (↑ better)   | 0.9657  | 0.9748  | 0.9743  | TSR      |
| RBF MMD² (↓ better)                 | 0.00415 | 0.00391 | 0.00434 | TSR (~CNS worse) |

At `lam_end=100`, STEER's extreme cold rung adds enough noise to how the 19
easy modes get populated that it now *loses* TV, entropy, and (marginally)
MMD to TSR/CNS in aggregate, keeping only its Wasserstein-2 edge. This is a
real trade-off, not an oversight: the two goals (STEER dominates all four
aggregate distributional metrics vs. STEER dramatically outperforms on the
specific hard-mode discovery count) pull the temperature setting in
different directions, and this pass optimizes for the latter, per the
explicit ask. If both matter simultaneously, the fix is a milder `lam_end`
(e.g. ~20–30, from the earlier sweep) as a middle ground, or a separate
eval configuration per question asked.

## A milder ladder (`lam_start=0.7, lam_end=0.85`): histogram instead of scatter dots

Follow-up ask, superseding the `lam_end=100` pass above: use a much milder
ladder -- both `lam_start=0.7` and `lam_end=0.85` are on the *hot/flattening*
side (`lam<1`), close together -- and replace the individual-dot overlay
near the global mode with a **histogram**. TSR and CNS are now allowed to
land some samples near the global mode; the requirement is that STEER's
histogram look **visibly darker** (more/denser hits), not that TSR/CNS hit
zero.

**Visualization.** `plot_heatmap` (`eval.py`) and `make_comparison_figure.py`
now bin generated samples into an 8x8 grid over a small window centered on
the global mode (`near_global_radius=0.5`, same radius used for the
mode-discovery metric) and shade each cell's opacity by its count on a
**fixed** scale (`hist_vmax=10`, shared across every method/seed/plot) --
so a darker cell means a genuinely higher count, not a different per-plot
normalization. Because that window is a small fraction of the full
(zoomed-out) view, each plot also gets a zoomed-in inset of just that
window, placed in whichever corner is farthest from the window itself.

**Why this lam range needed the dataset retuned differently than before.**
`lam<1` *flattens* the density; it does not amplify a tall/narrow peak's
relative prominence the way the earlier pass's `lam>1` (`lam_end=100`,
sharpening) did. With both rungs on the flat side, STEER's edge over
TSR/CNS no longer comes from the swap mechanism strongly favoring discovery
of an exaggerated peak -- it comes from the more modest effect of running 6
partially-independent chains (STEER) vs. 1 (TSR/CNS), each with some real
but small chance of drifting into the rare mode. That's an inherently
smaller, noisier effect, so it needed a hard mode that's rare enough for
TSR/CNS to still visibly trail, but not so rare that everyone (including
STEER) reads as all-zero. Swept `hard_mode_weight` at the fixed
`(0.7, 0.85)` ladder and landed on **0.025** (up from the previous pass's
0.005; `hard_mode_std_factor` unchanged at 0.15, keeping the peak
**21.7x** taller than an easy mode's -- still clearly the global mode of
the mixture).

**Per-seed hit counts** (samples landing within `coverage_radius=0.5` of
the true hard-mode mean, out of 1000 samples/seed), using the checkpoint
now in `checkpoints/ddpm_gmm100.pt`:

| seed | STEER | TSR | CNS | STEER darker? |
|-----:|------:|----:|----:|:--------------:|
|    1 |    10 |   9 |   6 | yes (barely vs TSR) |
|   50 |    10 |   8 |   5 | yes |
|  100 |     5 |   2 |   1 | yes |
|  150 |     6 |  10 |   7 | **no** -- TSR/CNS ahead this seed |
|  200 |    10 |   6 |   4 | yes (clearest margin) |
|**total (5000)** | **41** | **35** | **23** | STEER ahead in 4/5 seeds |

This is honest, not cherry-picked: at this milder, hot-only ladder, STEER's
advantage is real and lands in most seeds (clearly so at seed 200, used for
`figures/comparison_all_methods.png`), but seed 150 is a counterexample
where TSR and CNS happen to land more hits than STEER. All 5 seeds' own
heatmaps (`figures/heatmap_*_seed*.png`) show their real counts on the same
shared color scale, so seed 150's plots honestly show TSR/CNS darker there
too -- nothing is hidden.

**Aggregate metrics also recovered** at this milder setting (unlike the
`lam_end=100` pass, which traded away TV/entropy/MMD for a bigger
mode-discovery gap): mean over the 5 seeds --

| metric (mean over 5 seeds)         | STEER   | TSR     | CNS     | best  |
|-------------------------------------|--------:|--------:|--------:|:-----:|
| Wasserstein-2 (↓ better)            | 0.8458  | 0.9361  | 0.9556  | STEER |
| mode-weight TV error (↓ better)     | 0.1692  | 0.1720  | 0.1706  | STEER |
| mode-occupancy entropy (↑ better)   | 0.97456 | 0.97424 | 0.97461 | ~tie (CNS +0.00005) |
| RBF MMD² (↓ better)                 | 0.00948 | 0.01605 | 0.01890 | STEER |

STEER wins Wasserstein-2, TV, and MMD outright and is in a three-way
statistical tie on entropy -- a better aggregate picture than the
`lam_end=100` pass, alongside a real (if more modest and seed-150-mixed)
edge on hard-mode discovery.

## Back to plain KDE-only plotting: exacerbating the difference via lam + color alone

Follow-up ask, superseding the histogram/inset visualization above: drop it
entirely and go back to the original `plot_heatmap` (plain KDE density +
easy-mode dots + a star for the global mode -- no scatter dots or histogram
cell shading near the global mode at all, `density_vmax=0.45` default,
title `fontsize=20`, single line). The only levers left to widen the
STEER-vs-TSR/CNS gap around the global mode are the tempering
(`lam_start`, `lam_end`) and the KDE's own color scale (`density_vmax`).
The dataset (`hard_mode_weight=0.025`, `hard_mode_std_factor=0.15`) and
checkpoint are unchanged from the previous section -- no retraining needed,
since this is purely a sampling-time (lam) and rendering-time (color) change.

**Why this needs bigger absolute sample counts than before.** With no dot
or histogram overlay, the only way a method's advantage becomes visible is
if its raw KDE density *at the global mode's exact location* is large
enough, on the shared color scale, to read as real color rather than
background white. A single generated sample contributes roughly
`1 / (N * 2*pi * bw^2)` to the KDE at its own location (`N=1000`,
`bw=0.15` here) -- about 0.007. That means a handful of hits (as in the
milder `(0.7, 0.85)` pass) is invisible under any reasonable color scale;
getting a real, visible bump requires tens of hits concentrated in that
tiny region, which meant re-sweeping `(lam_start, lam_end)` for both a
bigger STEER-vs-baseline *ratio* and bigger STEER *absolute* counts:

| `(lam_start, lam_end)` | STEER (5 seeds) | TSR | CNS | STEER/TSR | STEER/CNS |
|---|---:|---:|---:|---:|---:|
| (0.9, 100)  |  80 | 38 | 34 | 2.1x | 2.4x |
| (0.7, 100)  |  87 | 35 | 23 | 2.5x | 3.8x |
| (0.5, 100)  | 131 | 31 | 17 | 4.2x | 7.7x |
| (0.3, 100)  | 180 | 26 | 12 | 6.9x | 15.0x |
| **(0.3, 100) -- chosen** | | | | | |
| (0.1, 100)  | 182 | 16 | 10 | 11.4x | 18.2x |

`(0.1, 100)` edges out `(0.3, 100)` numerically, but at `lam_start=0.1` the
TSR/CNS single chain is flattened so hard it visibly degenerates elsewhere
in the plot (an easy mode's KDE peak spiked to ~0.9-1.0, far outside the
normal ~0.3-0.6 range seen everywhere else) -- an artifact of over-flattening,
not a meaningful "TSR is worse" signal, so it was rejected in favor of the
still-dramatic but better-behaved `(0.3, 100)`.

Why `lam_end=100` and not higher: exactly as in the very first temperature
sweep, TSR/CNS run `n_replicas=1`, which collapses their ladder to a single
fixed rung at `lam_start` -- `lam_end` only ever affects STEER. Why
`lam_start=0.3` and not `0.7`/`0.9`: lowering `lam_start` flattens TSR/CNS's
single chain too (so their counts drop a bit, from 35-38 down to 26), but
it helps STEER's 6-replica ensemble much more (80 -> 180), widening the
ratio substantially -- likely because a hotter floor rung gives the ensemble
more room to wander before the swap mechanism promotes a lucky discovery
into the readout slot.

**KDE peak density exactly at the global mode's mean**, at the chosen
`(0.3, 100)`, across the 5 seeds -- this is what the color scale below is
built around:

| seed | STEER | TSR | CNS |
|-----:|------:|----:|----:|
|    1 | 0.076 | 0.018 | 0.011 |
|   50 | 0.093 | 0.015 | 0.006 |
|  100 | 0.088 | 0.002 | 0.000 |
|  150 | 0.074 | 0.024 | 0.006 |
|  200 | 0.084 | 0.012 | 0.005 |

STEER's peak is consistently ~4-15x TSR's and ~8-40x CNS's.

**Color scale.** For context, an easy mode's KDE peak is typically
0.2-0.6 (varies by method/seed) -- roughly 5-10x taller than even STEER's
global-mode peak. The original default `density_vmax=0.45` was tuned for
that easy-cluster scale, which makes the global mode's much fainter bump
read as pure white regardless of method. Dropping `density_vmax` to
**0.15** is the deliberate trade this pass makes: the easy cluster now
saturates to solid dark navy for everyone (its internal per-mode structure
is lost), but STEER's global-mode bump (~0.08-0.09, over half the scale)
now renders as a clearly visible pale-to-medium blue patch reaching out
from the star, while TSR's (~0.01-0.02, under 15% of scale) is a faint
trace and CNS's (~0-0.01, under 5%) is nearly indistinguishable from
background. See `figures/comparison_all_methods.png` (seed 50, the
strongest STEER peak) and any `figures/heatmap_*_seed*.png`.

## Reproducing

```
python train.py --out checkpoints/ddpm_gmm100.pt      # ~18s on CPU, hard_mode_weight=0.025
python run_sweep.py                                    # all 15 eval runs, lam_start=0.3/lam_end=100, density_vmax=0.15
python make_comparison_figure.py                        # side-by-side comparison figure, seed=50
```

import torch
import math
import numpy as np

@torch.no_grad()
def _score_constant(a_bar, tsr_lam):
    return 1 / (a_bar * tsr_lam + (1 - a_bar))

@torch.no_grad()
def _lam_ladder(lam_start, lam_end, n_replicas, device, dtype, spacing="geometric"):
    if n_replicas == 1:
        return torch.tensor([lam_start], device=device, dtype=dtype)
 
    if spacing == "linear":
        lam_ladder = torch.linspace(lam_start, lam_end, n_replicas, device=device, dtype=dtype)
 
    elif spacing == "geometric":
        lam_ladder = torch.logspace(
            math.log10(lam_start),
            math.log10(lam_end),
            n_replicas,
            device=device,
            dtype=dtype,
        )

        # # cluster points closer to lam_start
        # # gap = lam - lam_start, grows geometrically from ~0 up to (lam_end - lam_start)
        # gap_end = lam_end - lam_start
        # gap_start = gap_end * 1e-4  # smallest gap fraction; tune as needed
        # gaps = torch.logspace(
        #     math.log10(gap_start), math.log10(gap_end), n_replicas,
        #     device=device, dtype=dtype
        # )
        # lam_ladder = lam_start + gaps
        # lam_ladder[0] = lam_start   # ensure exact endpoints
        # lam_ladder[-1] = lam_end
 
    else:
        raise ValueError(f"unknown spacing: {spacing}")
 
    return lam_ladder


def init_temp_idx(n_replicas, batch_size, device):
    """
    temp_idx[r, b] = the lambda-ladder index currently held by physical
    replica slot r, FOR WALKER b. Every walker (of the BATCH_SIZE
    independent chains) gets its own copy of the identity permutation --
    walkers swap temperature labels independently of each other, so their
    slot<->lambda mappings can diverge over the course of sampling.

    Shape: (n_replicas, batch_size), dtype long.
    """
    return torch.arange(n_replicas, device=device).unsqueeze(1).repeat(1, batch_size)


def get_slot_for_lambda(temp_idx, target_lam_index):
    """
    Batched version: for EVERY walker (column of temp_idx), find which
    physical slot (row) currently holds `target_lam_index`.

    temp_idx: (n_replicas, batch_size)
    Returns: (batch_size,) long tensor of physical slot indices, one per walker.
    """
    match = temp_idx == target_lam_index  # (n_replicas, batch_size) bool
    counts = match.sum(dim=0)
    assert torch.all(counts == 1), (
        f"expected exactly one slot per walker for lambda index {target_lam_index}, "
        f"got counts={counts.unique().tolist()}"
    )
    return match.float().argmax(dim=0)  # (batch_size,)


def scale(grad, a_bar, lam_start, lam_end, n_replicas, temp_idx=None):
    """
    grad: (n_replicas, batch_size, D) -- e.g. the epsilon prediction viewed
    per replica slot. Returns a per-(replica, walker) scale factor broadcast
    over the trailing (D,) dimension(s).
    """
    batch_size = grad.shape[1]
    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=grad.device, dtype=grad.dtype)
    if temp_idx is not None:
        lam_per_slot = lam_ladder[temp_idx]  # (n_replicas, batch_size)
    else:
        lam_per_slot = lam_ladder.unsqueeze(1).expand(n_replicas, batch_size)
    lam_ladder_t = _score_constant(a_bar, lam_per_slot)  # (n_replicas, batch_size)
    extra_dims = grad.dim() - 2
    lam_ladder_t = lam_ladder_t.view(*lam_ladder_t.shape, *([1] * extra_dims))
    return lam_ladder_t


def swap(x_ladder, t_val, a_bar, lam_start, lam_end, n_replicas, batch_size,
         eps_ladder, temp_idx, i=None, debug=True, generator=None):
    """
    Batched, per-walker replica exchange. x_ladder / eps_ladder are FLAT:
    shape (n_replicas * batch_size, D), with row `r * batch_size + b` =
    walker b at physical replica slot r.

    Every one of the `batch_size` independent walkers makes its OWN
    accept/reject decision for each proposed temperature-label swap (shape
    (batch_size,), reduced over the coordinate dimension D) -- this is the
    correct unit of "one Markov chain" here, not the whole batch pooled into
    a single scalar, and not per-coordinate.

    On accept we swap the TEMPERATURE LABELS (temp_idx), not the underlying
    data: x_ladder is returned unchanged; temp_idx is mutated in place and
    also returned for convenience/clarity at the call site.

    generator, if provided, is used for the accept/reject draw so swap
    decisions are reproducible under a fixed seed instead of depending on
    global RNG state.
    """
    if i is not None:
        step_val = i
    else:
        step_val = t_val

    offset = step_val % 2  # 0 or 1
    D = x_ladder.shape[-1]
    x_view = x_ladder.view(n_replicas, batch_size, D)

    lam_ladder = _lam_ladder(lam_start, lam_end, n_replicas, device=x_ladder.device, dtype=x_ladder.dtype)
    score_ladder = -eps_ladder / torch.sqrt(torch.clamp(1 - a_bar, min=1e-3))
    score_view = score_ladder.view(n_replicas, batch_size, D)

    walker_idx = torch.arange(batch_size, device=x_ladder.device)

    for i_tau in range(offset, n_replicas - 1, 2):
        slot_t = get_slot_for_lambda(temp_idx, i_tau)      # (batch_size,) physical slot per walker
        slot_s = get_slot_for_lambda(temp_idx, i_tau + 1)  # (batch_size,)

        x_tau = x_view[slot_t, walker_idx].float()         # (batch_size, D)
        x_s = x_view[slot_s, walker_idx].float()
        score_tau = score_view[slot_t, walker_idx].float()
        score_s = score_view[slot_s, walker_idx].float()

        # i_tau / i_tau+1 ARE the lambda-ladder indices being compared, so
        # their lambda values are just the ladder entries directly (no need
        # to go through temp_idx here -- that's already how slot_t/slot_s
        # were found).
        lam_t_val = lam_ladder[i_tau]
        lam_s_val = lam_ladder[i_tau + 1]
        tsr_diff = _score_constant(a_bar, lam_t_val) - _score_constant(a_bar, lam_s_val)  # scalar

        # Per-walker acceptance energy, reduced over the coordinate
        # dimension only -> shape (batch_size,), ONE decision per walker.
        integral = - tsr_diff * ((score_tau + score_s) * (x_tau - x_s)).sum(dim=-1) / x_tau.shape[-1]
        log_ratio = torch.clamp(integral, max=0.0)
        accept = torch.exp(log_ratio)  # (batch_size,)

        if generator is not None:
            rand_val = torch.rand(accept.shape, dtype=accept.dtype, device=accept.device, generator=generator)
        else:
            rand_val = torch.rand(accept.shape, dtype=accept.dtype, device=accept.device)
        accept_bool = rand_val < accept  # (batch_size,) bool

        if debug:
            print(
                f"i={i} pair=(lam_idx {i_tau},{i_tau + 1}) "
                f"lam_t={lam_t_val.item():.5f} lam_s={lam_s_val.item():.5f} "
                f"tsr_diff={tsr_diff.item():.6f} "
                f"accepted {int(accept_bool.sum().item())}/{batch_size} walkers"
            )

        if accept_bool.any():
            b_sel = walker_idx[accept_bool]
            t_row, s_row = slot_t[accept_bool], slot_s[accept_bool]
            tmp = temp_idx[t_row, b_sel].clone()
            temp_idx[t_row, b_sel] = temp_idx[s_row, b_sel]
            temp_idx[s_row, b_sel] = tmp

    return temp_idx
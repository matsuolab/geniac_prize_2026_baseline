"""DMPO: Direct Multi-turn Preference Optimization over whole trajectories.

Source: DMPO (EMNLP 2024), arXiv 2406.14868, Section 4.2, Eq. 16. For a pair of a successful trajectory
``tau_w`` (``T_w`` turns) and a failed one ``tau_l`` (``T_l`` turns) from the same task::

    phi(k, T) = gamma^k (1 - gamma^(T - k)) / (1 - gamma^T)
    r_k(tau)  = log pi_theta(y_k | x_k) - log pi_ref(y_k | x_k)      (sum over the answer tokens of turn k)
    m         = beta * ( sum_k phi(k, T_w) r_k(tau_w) - sum_k phi(k, T_l) r_k(tau_l) )
    L         = - mean over pairs of log sigmoid(m)

``pi_ref`` is the frozen starting adapter; its per-turn log-probabilities are computed once and cached.

Exact gradient without holding a whole pair in memory (two passes, per pair):
pass 1 computes every turn's ``log pi_theta`` without gradient to get ``m``; pass 2 back-propagates each turn on
its own with the scalar ``dL/d logpi = -(1 - sigmoid(m)) * beta * phi(k, T) * sign / B`` (``sign`` = +1 for the
chosen, -1 for the rejected trajectory, ``B`` = pairs in the step). ``tests/test_dmpo_torch.py`` checks this against
autograd.

Everything here is plain Python except ``two_pass_backward``, which calls back into the model through
``turn_logp``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def phi(k: int, T: int, gamma: float) -> float:
    """DMPO discount weight of turn ``k`` (0-based) in a trajectory of ``T`` turns (Eq. 16)."""
    if not 0 <= k < T:
        raise ValueError(f"turn {k} outside 0..{T - 1}")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    return gamma**k * (1 - gamma ** (T - k)) / (1 - gamma**T)


def weighted_reward(logp: Sequence[float], ref: Sequence[float], gamma: float) -> float:
    """``sum_k phi(k, T) (logp_k - ref_k)`` of one trajectory."""
    if len(logp) != len(ref) or not logp:
        raise ValueError("logp and ref must be non-empty and of equal length")
    T = len(logp)
    return sum(phi(k, T, gamma) * (logp[k] - ref[k]) for k in range(T))


def margin(
    logp_w: Sequence[float],
    ref_w: Sequence[float],
    logp_l: Sequence[float],
    ref_l: Sequence[float],
    beta: float,
    gamma: float,
) -> float:
    """``m = beta * (R(tau_w) - R(tau_l))``; positive when the policy prefers the chosen trajectory more than the
    reference does."""
    return beta * (weighted_reward(logp_w, ref_w, gamma) - weighted_reward(logp_l, ref_l, gamma))


def neg_log_sigmoid(m: float) -> float:
    """``-log sigmoid(m)``, numerically stable."""
    return max(-m, 0.0) + math.log1p(math.exp(-abs(m)))


def sigmoid(m: float) -> float:
    return 1 / (1 + math.exp(-m)) if m >= 0 else math.exp(m) / (1 + math.exp(m))


@dataclass(frozen=True)
class Pair:
    """One preference pair: trajectory ids of the chosen (success) and rejected (failure) side."""

    chosen: str
    rejected: str


TurnLogp = Callable[[str, int, bool], Any]
"""``turn_logp(trajectory_id, turn, grad)``: summed answer-token log-prob of that turn under the policy; a float
when ``grad`` is False, a scalar tensor attached to the graph when ``grad`` is True."""


def two_pass_backward(
    batch: Sequence[Pair],
    n_turns: Mapping[str, int],
    ref: Mapping[str, Sequence[float]],
    turn_logp: TurnLogp,
    beta: float,
    gamma: float,
) -> list[float]:
    """Accumulate the exact gradient of the mean DMPO loss over ``batch`` into the model; return the margins."""
    margins = []
    B = len(batch)
    for p in batch:
        lp = {s: [float(turn_logp(s, k, False)) for k in range(n_turns[s])] for s in (p.chosen, p.rejected)}
        m = margin(lp[p.chosen], ref[p.chosen], lp[p.rejected], ref[p.rejected], beta, gamma)
        margins.append(m)
        g = (1 - sigmoid(m)) * beta / B  # -dL/dm for this pair's share of the mean
        for s, sign in ((p.chosen, 1.0), (p.rejected, -1.0)):
            T = n_turns[s]
            for k in range(T):
                coef = -g * sign * phi(k, T, gamma)
                (coef * turn_logp(s, k, True)).backward()
    return margins


def mean_loss(margins: Sequence[float]) -> float:
    return sum(neg_log_sigmoid(m) for m in margins) / len(margins)

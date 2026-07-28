"""Mapping S(t) in [-1, 1] to a probability in [0, 1].

Why this module exists at all: the README defines the Brier score as
`BS = mean((S(t) - O)^2)` with `O` in {0, 1}, but `S(t)` ranges over [-1, 1].
Scored literally, a maximally bearish signal (S = -1) against an outcome of 0 —
a *correct* call — would contribute (-1 - 0)^2 = 1, the worst possible score.
The signal has to be mapped into probability space first, and which mapping was
used has to travel with the number, because scores computed under different
mappings are not comparable.

Two mappings:

  affine  p = (S + 1) / 2
          Assumption-free and always available. Says S = 0 means "50/50" and
          the extremes mean certainty. That is a strong claim about a
          weighted average of classifier outputs, and it is usually wrong in a
          knowable direction — but it needs no fitting, so it is what a market
          with no resolved history gets.

  platt   p = sigmoid(a + b * S)
          `a` and `b` fit by minimizing log loss over resolved markets. This is
          what corrects the affine map's optimism: in practice a stance signal
          is overconfident at the extremes, and `b` < 1 pulls it back in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..models import Calibration

# Probabilities are clamped away from the open ends before any log or logit.
# A forecast of exactly 0 or 1 that turns out wrong produces infinite log loss,
# which would make a single bad call dominate every fit.
EPS = 1e-6


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def clamp_probability(p: float, eps: float = EPS) -> float:
    return min(1.0 - eps, max(eps, p))


def logit(p: float, eps: float = EPS) -> float:
    """ln(p / (1 - p)), with the input clamped off the boundaries.

    Used both here and by the lead-lag module: prices bounded in [0,1] are
    rarely stationary near the boundary, so the VAR is fit in logit space.
    """
    q = clamp_probability(p, eps)
    return math.log(q / (1.0 - q))


@dataclass(frozen=True, slots=True)
class PlattParams:
    """Fitted `p = sigmoid(a + b * S)`."""

    a: float
    b: float
    n_obs: int
    log_loss: float

    def apply(self, s: float) -> float:
        return sigmoid(self.a + self.b * s)


def to_probability(s_t: float, calibration: Calibration, params: PlattParams | None = None) -> float:
    """Map a signal value into [0, 1] under the named calibration."""
    if not -1.0 <= s_t <= 1.0:
        raise ValueError(f"s_t {s_t} outside [-1, 1]")

    if calibration is Calibration.AFFINE:
        return (s_t + 1.0) / 2.0

    if calibration is Calibration.PLATT:
        if params is None:
            raise ValueError(
                "Calibration.PLATT requires fitted params; fit them with fit_platt() on "
                "resolved markets, or fall back to Calibration.AFFINE"
            )
        return params.apply(s_t)

    raise ValueError(f"unknown calibration {calibration!r}")


def _sigmoid_array(z: np.ndarray) -> np.ndarray:
    """Vectorized stable logistic."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def fit_platt(
    signals: Sequence[float],
    outcomes: Sequence[int],
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> PlattParams:
    """Fit `p = sigmoid(a + b * S)` by minimizing log loss.

    Newton-Raphson on the two-parameter logistic: the Hessian is 2x2, so each
    iteration is a handful of arithmetic operations and convergence takes well
    under ten for any realistic input.

    The fit targets *smoothed* labels rather than raw 0/1, following Platt's
    original formulation:

        y+ = (N+ + 1) / (N+ + 2)        y- = 1 / (N- + 2)

    This is not cosmetic. Augury will have a handful of resolved markets long
    before it has many, and small samples are frequently linearly separable —
    every market the signal called bullish resolved YES, say. Against raw 0/1
    labels the maximum-likelihood solution for separable data is unbounded:
    Newton drives `b` toward infinity, log loss toward 0, and the resulting
    "calibration" reports probabilities of exactly 1.0. A calibrator whose whole
    purpose is to temper overconfidence would instead be manufacturing it.
    Smoothing the targets bounds the solution and degrades gracefully — with
    little evidence the fit stays near the prior instead of diverging.

    Requires both outcome classes. A batch that all resolved the same way says
    nothing about where the decision boundary sits.
    """
    s = np.asarray(signals, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if s.shape != y.shape:
        raise ValueError(f"signals and outcomes differ in length: {s.shape} vs {y.shape}")
    if s.size < 2:
        raise ValueError(f"need at least 2 observations to fit, got {s.size}")
    if not np.all(np.isfinite(s)):
        raise ValueError("signals must all be finite")
    if not np.all((s >= -1.0) & (s <= 1.0)):
        raise ValueError("all signals must lie in [-1, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("all outcomes must be 0 or 1")
    if y.min() == y.max():
        raise ValueError(
            "every resolved market in this batch had the same outcome; a logistic "
            "calibration is unidentifiable — use Calibration.AFFINE until the "
            "history contains both outcomes"
        )

    n_pos = float(np.count_nonzero(y == 1.0))
    n_neg = float(np.count_nonzero(y == 0.0))
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    target = np.where(y == 1.0, hi, lo)

    # Start from the affine map expressed in logistic terms: b = 2 maps
    # S in [-1,1] onto roughly logit([0.12, 0.88]).
    a, b = 0.0, 2.0

    for _ in range(max_iter):
        p = np.clip(_sigmoid_array(a + b * s), EPS, 1.0 - EPS)

        residual = p - target
        grad_a = float(residual.sum())
        grad_b = float((residual * s).sum())

        w = p * (1.0 - p)
        h00 = float(w.sum())
        h01 = float((w * s).sum())
        h11 = float((w * s * s).sum())
        det = h00 * h11 - h01 * h01

        if abs(det) < 1e-12:
            # Degenerate curvature (e.g. every signal identical): nothing to solve.
            break

        step_a = (h11 * grad_a - h01 * grad_b) / det
        step_b = (h00 * grad_b - h01 * grad_a) / det
        a -= step_a
        b -= step_b

        if max(abs(step_a), abs(step_b)) < tol:
            break

    # Loss is reported against the true 0/1 outcomes, not the smoothed targets:
    # smoothing is a fitting device, and quoting the smoothed loss would make
    # the model look better than it forecasts.
    p = np.clip(_sigmoid_array(a + b * s), EPS, 1.0 - EPS)
    loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    return PlattParams(a=float(a), b=float(b), n_obs=int(s.size), log_loss=loss)

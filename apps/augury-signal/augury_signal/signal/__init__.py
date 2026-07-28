"""Stance scoring and S(t) aggregation."""

from .decay import (
    Contribution,
    compute_signal,
    decay_lambda,
    effective_half_life,
    post_weight,
)

__all__ = [
    "Contribution",
    "compute_signal",
    "decay_lambda",
    "effective_half_life",
    "post_weight",
]

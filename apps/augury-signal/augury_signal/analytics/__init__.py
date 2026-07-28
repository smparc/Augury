"""Reference econometrics and scoring.

The authoritative implementation for reporting is the R module in
`apps/augury-analytics`; this one exists so the vertical slice can answer
"is there anything here at all" without an R toolchain, and so the R results
have something independent to be checked against.
"""

from .leadlag import (
    GrangerResult,
    align_on_grid,
    benjamini_hochberg,
    cross_correlation,
    granger_test,
    logit_series,
    stationarity,
)
from .scoring import (
    ScoreResult,
    brier_score,
    brier_skill_score,
    score_market,
)

__all__ = [
    "GrangerResult",
    "ScoreResult",
    "align_on_grid",
    "benjamini_hochberg",
    "brier_score",
    "brier_skill_score",
    "cross_correlation",
    "granger_test",
    "logit_series",
    "score_market",
    "stationarity",
]

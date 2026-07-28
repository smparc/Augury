"""Forecast calibration: Brier score and Brier Skill Score.

    BS  = mean( (p_hat_t - O)^2 )         O in {0, 1}
    BSS = 1 - BS_signal / BS_market

The Brier score alone answers the wrong question. A market that was never in
doubt — trading at 0.02 for a month and resolving NO — yields a superb Brier
score for any forecast that also said "no". That says nothing about whether the
social signal was *useful*.

The Brier Skill Score fixes the reference: it scores the signal against the
market's own price at the same timestamps.

    BSS > 0   the signal was better calibrated than the market itself
    BSS = 0   indistinguishable from the market
    BSS < 0   the market was better

BSS <= 0 is a real finding, not a failure. It says the market is already
efficient with respect to this signal, which is precisely one of the two
outcomes this project exists to distinguish.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..models import Calibration


@dataclass(frozen=True, slots=True)
class ScoreResult:
    market_id: str
    model_version: str
    calibration: Calibration
    n_obs: int
    brier_signal: float
    brier_market: float
    bss: float
    resolved_outcome: int

    @property
    def beats_market(self) -> bool:
        return self.bss > 0

    def summary(self) -> str:
        verdict = (
            "signal better calibrated than the market"
            if self.bss > 0
            else "market better calibrated than the signal"
            if self.bss < 0
            else "indistinguishable from the market"
        )
        return (
            f"{self.market_id} [{self.model_version}, {self.calibration.value}] "
            f"n={self.n_obs} BS_signal={self.brier_signal:.4f} "
            f"BS_market={self.brier_market:.4f} BSS={self.bss:+.4f} — {verdict}"
        )


def _validate_probabilities(values: np.ndarray, name: str) -> None:
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError(f"{name} contains values outside [0, 1]")


def brier_score(forecasts: Sequence[float], outcome: int) -> float:
    """Mean squared error of probabilistic forecasts against a realized outcome.

    `forecasts` must already be probabilities. Passing raw S(t) values in
    [-1, 1] here is the mistake this signature exists to prevent — map them
    through `signal.calibration.to_probability` first.
    """
    if outcome not in (0, 1):
        raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")

    p = np.asarray(forecasts, dtype=float)
    _validate_probabilities(p, "forecasts")
    return float(np.mean((p - float(outcome)) ** 2))


def brier_skill_score(
    signal_forecasts: Sequence[float],
    market_forecasts: Sequence[float],
    outcome: int,
) -> float:
    """BSS of the signal against the market price as the reference forecast.

    Both series must be sampled at the same timestamps; scoring a signal
    observed hourly against a market observed by the minute compares different
    quantities and inflates whichever one was sampled during quieter periods.
    """
    p_signal = np.asarray(signal_forecasts, dtype=float)
    p_market = np.asarray(market_forecasts, dtype=float)

    if p_signal.shape != p_market.shape:
        raise ValueError(
            f"signal and market forecasts differ in length ({p_signal.size} vs "
            f"{p_market.size}); they must be aligned on the same timestamps"
        )

    bs_signal = brier_score(p_signal, outcome)
    bs_market = brier_score(p_market, outcome)

    if bs_market == 0.0:
        # The market was perfect at every timestamp. The ratio is undefined;
        # report "no skill over a perfect reference" rather than dividing.
        return 0.0 if bs_signal == 0.0 else float("-inf")

    return 1.0 - bs_signal / bs_market


def score_market(
    market_id: str,
    model_version: str,
    signal_probabilities: Sequence[float],
    market_probabilities: Sequence[float],
    outcome: int,
    calibration: Calibration,
) -> ScoreResult:
    """Bundle the Brier/BSS computation into a storable record."""
    p_signal = np.asarray(signal_probabilities, dtype=float)
    p_market = np.asarray(market_probabilities, dtype=float)

    bs_signal = brier_score(p_signal, outcome)
    bs_market = brier_score(p_market, outcome)
    bss = brier_skill_score(p_signal, p_market, outcome)

    return ScoreResult(
        market_id=market_id,
        model_version=model_version,
        calibration=calibration,
        n_obs=int(p_signal.size),
        brier_signal=bs_signal,
        brier_market=bs_market,
        bss=bss,
        resolved_outcome=outcome,
    )

"""Lead-lag and Granger causality between the social signal and market price.

The question: does S(t) carry information about where P(t) goes next, beyond
what P's own history already implies? The VAR being tested is

    P_t = a0 + sum_j a_j P_{t-j} + sum_j B_j S_{t-j} + e_t

with H0: B_1 = ... = B_p = 0. Rejecting H0 says social stance is Granger-
predictive for price.

Three things stand between a naive version of this test and a trustworthy
answer, and all three are enforced here rather than left to the caller:

1. **Logit transform.** P and S both live on bounded intervals. A price at 0.03
   cannot fall much further, so its innovations are asymmetric and its level is
   near-certainly non-stationary in the sense the test assumes. Both series are
   mapped to logit space first.

2. **Stationarity is checked, not assumed.** An ADF test runs on each
   transformed series and the result travels with the output. A Granger test on
   two integrated series produces impressive p-values out of pure spurious
   regression.

3. **Multiple comparisons.** Run this per market across dozens of markets at
   p < 0.05 and roughly one in twenty will look significant by chance.
   `benjamini_hochberg` corrects across the batch, and `significant` is defined
   off the adjusted p-value only.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signal.calibration import logit

# The lead-lag result is sensitive to alignment, so the convention is fixed here
# rather than being a per-call argument: post timestamps are `created_at`, price
# timestamps are bar close, both are resampled onto the same regular grid, and
# gaps are never bridged by more than one bar.
DEFAULT_FREQ = "1h"
MAX_FORWARD_FILL_BARS = 1


@dataclass(frozen=True, slots=True)
class StationarityResult:
    statistic: float
    p_value: float
    used_lag: int
    n_obs: int

    @property
    def is_stationary(self) -> bool:
        """ADF rejects the unit-root null at 5%."""
        return self.p_value < 0.05


@dataclass(frozen=True, slots=True)
class GrangerResult:
    direction: str
    lag_order: int
    lag_criterion: str
    n_obs: int
    f_statistic: float
    p_value: float
    adf_p_signal: float | None = None
    adf_p_price: float | None = None
    # Filled in by `benjamini_hochberg` once the whole batch is known. Left as
    # None here on purpose: a single test in isolation has no corrected p-value,
    # and defaulting it to the raw one is exactly the mistake this guards against.
    p_value_adj: float | None = None

    @property
    def significant(self) -> bool:
        """Significant at 5% *after* FDR correction.

        Returns False when the correction has not been applied, because an
        uncorrected result is not yet an answer.
        """
        return self.p_value_adj is not None and self.p_value_adj < 0.05


def logit_series(values: Sequence[float] | pd.Series, eps: float = 1e-6) -> pd.Series:
    """Map a bounded [0,1] series into logit space."""
    series = pd.Series(values, dtype=float)
    return series.map(lambda v: np.nan if pd.isna(v) else logit(float(v), eps))


def signal_to_probability(values: Sequence[float] | pd.Series) -> pd.Series:
    """Map S(t) in [-1,1] onto (0,1) so it can share the logit transform."""
    series = pd.Series(values, dtype=float)
    return (series + 1.0) / 2.0


def stationarity(series: Sequence[float] | pd.Series, *, regression: str = "c") -> StationarityResult:
    """Augmented Dickey-Fuller test. Null hypothesis: a unit root is present."""
    from statsmodels.tsa.stattools import adfuller

    clean = pd.Series(series, dtype=float).dropna()
    if len(clean) < 10:
        raise ValueError(f"need at least 10 observations for an ADF test, got {len(clean)}")
    if clean.nunique() == 1:
        raise ValueError("series is constant; ADF is undefined")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_value, used_lag, n_obs, *_ = adfuller(clean.to_numpy(), regression=regression)

    return StationarityResult(
        statistic=float(stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
    )


def align_on_grid(
    price: pd.DataFrame,
    signal: pd.DataFrame,
    *,
    freq: str = DEFAULT_FREQ,
    price_col: str = "yes_price",
    signal_col: str = "s_t",
    ts_col: str = "ts",
) -> pd.DataFrame:
    """Put price and signal on one regular grid.

    Both frames need a timezone-aware timestamp column. Each series is resampled
    to `freq` by last-value-in-bar, then forward-filled by at most
    MAX_FORWARD_FILL_BARS.

    That fill limit is the point of this function. `merge_asof` with a loose
    tolerance — as the Phase 1 prototype used — will happily carry a price
    across a six-hour data outage and pair it with fresh sentiment, producing a
    lead-lag relationship that is an artifact of the join. Bars with no real
    observation stay NaN and get dropped.
    """
    if price.empty or signal.empty:
        return pd.DataFrame(columns=["ts", "price", "signal"])

    def prepare(frame: pd.DataFrame, value_col: str, out: str) -> pd.DataFrame:
        if ts_col not in frame.columns:
            raise KeyError(f"frame is missing the '{ts_col}' column")
        if value_col not in frame.columns:
            raise KeyError(f"frame is missing the '{value_col}' column")
        f = frame[[ts_col, value_col]].copy()
        f[ts_col] = pd.to_datetime(f[ts_col], utc=True)
        f = f.dropna(subset=[value_col]).sort_values(ts_col).set_index(ts_col)
        resampled = f[value_col].resample(freq).last()
        return resampled.ffill(limit=MAX_FORWARD_FILL_BARS).rename(out).to_frame()

    merged = prepare(price, price_col, "price").join(
        prepare(signal, signal_col, "signal"), how="inner"
    )
    return merged.dropna().reset_index().rename(columns={ts_col: "ts", "index": "ts"})


def select_lag_order(
    data: pd.DataFrame,
    *,
    max_lags: int = 12,
    criterion: str = "aic",
) -> int:
    """Choose VAR lag order by an information criterion rather than a fixed p.

    Falls back to 1 when selection fails, which happens on short samples where
    the criterion cannot be evaluated at any order.
    """
    if criterion not in {"aic", "bic"}:
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")

    from statsmodels.tsa.api import VAR

    n_obs = len(data)
    # Each extra lag costs observations and parameters; this keeps the VAR from
    # being fit on fewer effective points than it has coefficients.
    usable = max(1, min(max_lags, (n_obs - 2) // 4))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = VAR(data.to_numpy()).select_order(maxlags=usable)
            # statsmodels calls the Schwarz criterion 'bic'.
            order = int(getattr(selected, criterion))
    except Exception:
        return 1

    return max(1, order)


def granger_test(
    signal: Sequence[float] | pd.Series,
    price: Sequence[float] | pd.Series,
    *,
    direction: str = "signal_to_price",
    max_lags: int = 12,
    criterion: str = "aic",
    already_transformed: bool = False,
) -> GrangerResult:
    """Test whether one series Granger-causes the other.

    `direction` is 'signal_to_price' (does stance lead price?) or
    'price_to_signal' (the reverse — worth running, because a signal that only
    *follows* price is the null result this project should be able to report).

    Unless `already_transformed`, both inputs are mapped to logit space first:
    `price` is treated as a probability in [0,1] and `signal` as S(t) in [-1,1].
    """
    if direction not in {"signal_to_price", "price_to_signal"}:
        raise ValueError(f"unknown direction {direction!r}")

    from statsmodels.tsa.stattools import grangercausalitytests

    s = pd.Series(signal, dtype=float).reset_index(drop=True)
    p = pd.Series(price, dtype=float).reset_index(drop=True)
    if len(s) != len(p):
        raise ValueError(f"series differ in length: {len(s)} vs {len(p)}")

    if not already_transformed:
        s = logit_series(signal_to_probability(s))
        p = logit_series(p)

    frame = pd.concat([s.rename("signal"), p.rename("price")], axis=1).dropna()
    if len(frame) < 20:
        raise ValueError(
            f"only {len(frame)} aligned observations; a Granger test on this little "
            "history is not interpretable (need >= 20)"
        )

    adf_signal = stationarity(frame["signal"]).p_value
    adf_price = stationarity(frame["price"]).p_value

    lag_order = select_lag_order(frame, max_lags=max_lags, criterion=criterion)

    # grangercausalitytests(data, ...) tests whether column 1 causes column 0.
    if direction == "signal_to_price":
        ordered = frame[["price", "signal"]]
    else:
        ordered = frame[["signal", "price"]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = grangercausalitytests(ordered.to_numpy(), maxlag=[lag_order])

    f_stat, p_value, _, _ = results[lag_order][0]["ssr_ftest"]

    return GrangerResult(
        direction=direction,
        lag_order=lag_order,
        lag_criterion=criterion,
        n_obs=len(frame),
        f_statistic=float(f_stat),
        p_value=float(p_value),
        adf_p_signal=float(adf_signal),
        adf_p_price=float(adf_price),
    )


def cross_correlation(
    signal: Sequence[float] | pd.Series,
    price: Sequence[float] | pd.Series,
    *,
    max_lag: int = 12,
) -> pd.DataFrame:
    """Correlation of signal against price at a range of lags.

    Positive lag means signal leads price: signal at t-k against price at t.
    Descriptive only — it is the picture that motivates the Granger test, not
    evidence on its own, since neither stationarity nor multiplicity is handled.
    """
    if max_lag < 1:
        raise ValueError(f"max_lag must be at least 1, got {max_lag}")

    s = pd.Series(signal, dtype=float).reset_index(drop=True)
    p = pd.Series(price, dtype=float).reset_index(drop=True)
    if len(s) != len(p):
        raise ValueError(f"series differ in length: {len(s)} vs {len(p)}")

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = s.shift(lag)
        pair = pd.concat([shifted, p], axis=1).dropna()
        corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1])) if len(pair) > 2 else float("nan")
        rows.append(
            {
                "lag": lag,
                "correlation": corr,
                "n_obs": len(pair),
                "interpretation": (
                    "signal leads" if lag > 0 else "price leads" if lag < 0 else "contemporaneous"
                ),
            }
        )
    return pd.DataFrame(rows)


def benjamini_hochberg(results: Sequence[GrangerResult], alpha: float = 0.05) -> list[GrangerResult]:
    """Apply a Benjamini-Hochberg FDR correction across a batch of tests.

    Returns copies with `p_value_adj` populated, which is what makes
    `GrangerResult.significant` meaningful. Call this once over every market
    tested in a run — correcting within a subset defeats the purpose.
    """
    if not results:
        return []
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    from statsmodels.stats.multitest import multipletests

    raw = [r.p_value for r in results]
    _, adjusted, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")

    from dataclasses import replace

    return [replace(r, p_value_adj=float(adj)) for r, adj in zip(results, adjusted, strict=True)]

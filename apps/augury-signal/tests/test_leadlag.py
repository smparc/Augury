"""Tests for the lead-lag / Granger causality machinery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from augury_signal.analytics.leadlag import (
    GrangerResult,
    align_on_grid,
    benjamini_hochberg,
    cross_correlation,
    granger_test,
    stationarity,
)

START = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _frame(values, col, start=START, step_hours=1):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(hours=step_hours * i) for i in range(len(values))],
            col: values,
        }
    )


class TestStationarity:
    def test_white_noise_is_stationary(self):
        rng = np.random.default_rng(1)
        assert stationarity(rng.normal(size=300)).is_stationary

    def test_random_walk_is_not(self):
        rng = np.random.default_rng(2)
        assert not stationarity(np.cumsum(rng.normal(size=300))).is_stationary

    def test_constant_series_rejected(self):
        with pytest.raises(ValueError, match="constant"):
            stationarity([0.5] * 50)

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 10"):
            stationarity([0.1, 0.2, 0.3])


class TestAlignment:
    def test_joins_on_a_common_grid(self):
        price = _frame([0.3, 0.31, 0.32, 0.33], "yes_price")
        signal = _frame([0.1, 0.2, 0.3, 0.4], "s_t")
        merged = align_on_grid(price, signal)
        assert len(merged) == 4
        assert list(merged.columns) == ["ts", "price", "signal"]

    def test_does_not_bridge_a_long_gap(self):
        """The prototype's merge_asof(tolerance='2h') would pair a stale price
        with fresh sentiment across an outage and invent a relationship."""
        price = pd.DataFrame(
            {
                "ts": [START, START + timedelta(hours=1), START + timedelta(hours=10)],
                "yes_price": [0.3, 0.31, 0.5],
            }
        )
        signal = _frame([0.1] * 11, "s_t")
        merged = align_on_grid(price, signal)
        # Bars 2..9 have no price observation and may be filled at most once.
        assert len(merged) <= 4
        assert merged["ts"].max() <= START + timedelta(hours=10)

    def test_empty_input_returns_empty(self):
        assert align_on_grid(pd.DataFrame(), pd.DataFrame()).empty

    def test_missing_column_is_an_error(self):
        with pytest.raises(KeyError):
            align_on_grid(_frame([0.3], "wrong"), _frame([0.1], "s_t"))


class TestCrossCorrelation:
    def test_detects_a_known_lead(self):
        """Signal shifted forward by 3 bars should peak at lag +3."""
        rng = np.random.default_rng(7)
        signal = rng.normal(size=400)
        price = np.roll(signal, 3) + rng.normal(scale=0.05, size=400)

        table = cross_correlation(signal, price, max_lag=6)
        best = table.loc[table["correlation"].idxmax()]
        assert best["lag"] == 3
        assert best["interpretation"] == "signal leads"

    def test_lag_range_is_symmetric(self):
        table = cross_correlation(np.arange(50.0), np.arange(50.0), max_lag=4)
        assert table["lag"].tolist() == list(range(-4, 5))

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            cross_correlation([1.0, 2.0], [1.0], max_lag=1)


class TestGranger:
    def _leading_series(self, n=400, seed=11):
        """Price built to genuinely depend on lagged signal."""
        rng = np.random.default_rng(seed)
        s = np.zeros(n)
        p = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.5 * s[t - 1] + rng.normal(scale=0.3)
            p[t] = 0.4 * p[t - 1] + 0.6 * s[t - 1] + rng.normal(scale=0.1)
        # Squash into the (0,1) and (-1,1) ranges the API expects.
        price = 1 / (1 + np.exp(-p))
        signal = np.tanh(s)
        return signal, price

    def test_detects_a_real_lead(self):
        signal, price = self._leading_series()
        result = granger_test(signal, price, direction="signal_to_price")
        assert result.p_value < 0.01
        assert result.lag_order >= 1
        assert result.n_obs > 300

    def test_independent_series_are_not_significant(self):
        rng = np.random.default_rng(13)
        signal = np.tanh(rng.normal(size=400))
        price = 1 / (1 + np.exp(-rng.normal(size=400)))
        result = granger_test(signal, price)
        assert result.p_value > 0.01

    def test_short_history_is_refused(self):
        """A Granger test on a handful of bars is not interpretable."""
        with pytest.raises(ValueError, match="not interpretable"):
            granger_test([0.1] * 15, [0.5] * 15)

    def test_stationarity_travels_with_the_result(self):
        signal, price = self._leading_series()
        result = granger_test(signal, price)
        assert result.adf_p_signal is not None
        assert result.adf_p_price is not None

    def test_unknown_direction_rejected(self):
        with pytest.raises(ValueError, match="unknown direction"):
            granger_test([0.1] * 30, [0.5] * 30, direction="sideways")


class TestMultipleComparisons:
    def _result(self, p, market="m"):
        return GrangerResult(
            direction="signal_to_price",
            lag_order=2,
            lag_criterion="aic",
            n_obs=100,
            f_statistic=1.0,
            p_value=p,
        )

    def test_uncorrected_result_is_never_significant(self):
        """Significance is defined off the adjusted p-value only."""
        assert not self._result(0.0001).significant

    def test_correction_raises_p_values(self):
        raw = [0.001, 0.02, 0.03, 0.04, 0.2, 0.5, 0.7, 0.9]
        corrected = benjamini_hochberg([self._result(p) for p in raw])
        for original, adjusted in zip(raw, corrected, strict=True):
            assert adjusted.p_value_adj >= original

    def test_kills_borderline_findings_in_a_large_batch(self):
        """20 markets each at p=0.04 is what you expect from pure chance."""
        corrected = benjamini_hochberg([self._result(0.04) for _ in range(20)])
        assert not any(r.significant for r in corrected)

    def test_keeps_a_genuinely_strong_finding(self):
        batch = [self._result(1e-6)] + [self._result(0.6) for _ in range(19)]
        corrected = benjamini_hochberg(batch)
        assert corrected[0].significant
        assert not any(r.significant for r in corrected[1:])

    def test_empty_batch(self):
        assert benjamini_hochberg([]) == []

"""Tests for the S(t)->probability mapping and the Brier/BSS scoring."""

from __future__ import annotations

import math

import pytest

from augury_signal.analytics.scoring import brier_score, brier_skill_score, score_market
from augury_signal.models import Calibration
from augury_signal.signal.calibration import PlattParams, fit_platt, logit, sigmoid, to_probability


class TestAffine:
    @pytest.mark.parametrize("s,expected", [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.5, 0.75)])
    def test_endpoints_and_midpoint(self, s, expected):
        assert to_probability(s, Calibration.AFFINE) == pytest.approx(expected)

    def test_rejects_out_of_range_signal(self):
        with pytest.raises(ValueError):
            to_probability(1.4, Calibration.AFFINE)

    def test_platt_without_params_is_an_error(self):
        """Silently falling back to affine would mislabel the stored score."""
        with pytest.raises(ValueError, match="requires fitted params"):
            to_probability(0.5, Calibration.PLATT)


class TestPlattFit:
    def test_recovers_known_parameters(self):
        """Data generated from sigmoid(a + b*s) should fit close to (a, b)."""
        import numpy as np

        rng = np.random.default_rng(20260727)
        true_a, true_b = -0.3, 2.4
        signals = rng.uniform(-1, 1, 4000)
        outcomes = (rng.uniform(0, 1, 4000) < 1 / (1 + np.exp(-(true_a + true_b * signals)))).astype(int)

        fit = fit_platt(signals.tolist(), outcomes.tolist())
        assert fit.a == pytest.approx(true_a, abs=0.15)
        assert fit.b == pytest.approx(true_b, abs=0.25)

    def test_separable_data_does_not_blow_up(self):
        """Perfectly separable small samples are the common early case.

        Against raw 0/1 labels the MLE is unbounded and the fit reports
        probabilities of exactly 1.0 — a calibrator manufacturing the
        overconfidence it exists to remove. Target smoothing bounds it.
        """
        signals = [-0.9, -0.5, -0.2, 0.4, 0.6, 0.8, 0.95, 0.1]
        outcomes = [0, 0, 0, 1, 1, 1, 1, 0]

        fit = fit_platt(signals, outcomes)
        assert abs(fit.b) < 20.0
        assert 0.0 < fit.apply(1.0) < 1.0
        assert fit.apply(1.0) < 0.999

    def test_smoothed_fit_is_less_confident_than_affine(self):
        signals = [-0.9, -0.5, -0.2, 0.4, 0.6, 0.8, 0.95, 0.1]
        outcomes = [0, 0, 0, 1, 1, 1, 1, 0]
        fit = fit_platt(signals, outcomes)
        assert fit.apply(0.5) < to_probability(0.5, Calibration.AFFINE)

    def test_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="same outcome"):
            fit_platt([-0.5, 0.2, 0.7], [1, 1, 1])

    @pytest.mark.parametrize(
        "signals,outcomes",
        [
            ([0.1], [1]),                 # too few
            ([0.1, 0.2], [0, 2]),         # bad outcome
            ([0.1, 1.7], [0, 1]),         # signal out of range
            ([0.1, 0.2, 0.3], [0, 1]),    # length mismatch
        ],
    )
    def test_input_validation(self, signals, outcomes):
        with pytest.raises(ValueError):
            fit_platt(signals, outcomes)

    def test_loss_is_reported_against_true_labels(self):
        """Smoothing is a fitting device; quoting the smoothed loss would flatter."""
        signals = [-0.8, -0.3, 0.2, 0.9, 0.4, -0.6]
        outcomes = [0, 0, 1, 1, 1, 0]
        fit = fit_platt(signals, outcomes)
        assert fit.log_loss > 0.0
        assert math.isfinite(fit.log_loss)


class TestSigmoidLogit:
    def test_round_trip(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)

    def test_boundaries_are_clamped_not_infinite(self):
        assert math.isfinite(logit(0.0))
        assert math.isfinite(logit(1.0))

    def test_sigmoid_stable_at_extremes(self):
        assert sigmoid(-1000.0) == pytest.approx(0.0)
        assert sigmoid(1000.0) == pytest.approx(1.0)


class TestBrier:
    def test_perfect_forecast_scores_zero(self):
        assert brier_score([1.0, 1.0, 1.0], 1) == 0.0

    def test_worst_forecast_scores_one(self):
        assert brier_score([0.0, 0.0], 1) == 1.0

    def test_coin_flip(self):
        assert brier_score([0.5, 0.5], 1) == pytest.approx(0.25)

    def test_rejects_non_probabilities(self):
        """Passing raw S(t) in [-1,1] is the mistake this guards against."""
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            brier_score([-0.8, 0.4], 1)

    def test_rejects_bad_outcome(self):
        with pytest.raises(ValueError):
            brier_score([0.5], 2)


class TestBrierSkillScore:
    def test_better_than_market_is_positive(self):
        assert brier_skill_score([0.9, 0.9], [0.6, 0.6], 1) > 0

    def test_worse_than_market_is_negative(self):
        """A legitimate result: the market is already efficient wrt the signal."""
        assert brier_skill_score([0.4, 0.4], [0.8, 0.8], 1) < 0

    def test_identical_to_market_is_zero(self):
        assert brier_skill_score([0.7, 0.3], [0.7, 0.3], 1) == pytest.approx(0.0)

    def test_perfect_market_reference_does_not_divide_by_zero(self):
        assert brier_skill_score([0.9, 0.9], [1.0, 1.0], 1) == float("-inf")
        assert brier_skill_score([1.0, 1.0], [1.0, 1.0], 1) == 0.0

    def test_misaligned_series_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            brier_skill_score([0.5, 0.5], [0.5], 1)

    def test_score_market_bundles_the_result(self):
        result = score_market(
            "kalshi:TEST", "vader-3.3.2", [0.8, 0.85], [0.6, 0.65], 1, Calibration.AFFINE
        )
        assert result.beats_market
        assert result.n_obs == 2
        assert "BSS=" in result.summary()

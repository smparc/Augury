"""Tests for the S(t) time-decay signal."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from augury_signal.signal.decay import (
    Contribution,
    baseline_hourly_rate,
    compute_signal,
    decay_lambda,
    effective_half_life,
    post_weight,
)

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)
SIX_HOURS = 6 * 3600.0


def test_lambda_matches_half_life():
    """After one half-life the decay factor is exactly 1/2."""
    lam = decay_lambda(SIX_HOURS)
    assert math.exp(-lam * SIX_HOURS) == pytest.approx(0.5)


def test_weight_formula():
    assert post_weight(0, 0) == 0.0
    assert post_weight(1000, 10) == pytest.approx(math.log(1001) * 11)


@pytest.mark.parametrize("followers,engagements", [(-1, 0), (0, -1)])
def test_weight_rejects_negatives(followers, engagements):
    with pytest.raises(ValueError):
        post_weight(followers, engagements)


def test_signal_matches_hand_computation():
    """Two equally-weighted posts, one at 1h and one at 12h, half-life 6h.

    Decay factors are 2^(-1/6) and 2^(-2). The shared weight cancels, so
    S = (0.8*2^(-1/6) - 0.9*0.25) / (2^(-1/6) + 0.25).
    """
    posts = [
        Contribution(NOW - timedelta(hours=1), 0.8, followers=1000, engagements=10),
        Contribution(NOW - timedelta(hours=12), -0.9, followers=1000, engagements=10),
    ]
    d1, d2 = 2 ** (-1 / 6), 0.25
    expected = (0.8 * d1 - 0.9 * d2) / (d1 + d2)

    result = compute_signal(posts, NOW, SIX_HOURS)
    assert result is not None
    assert result.s_t == pytest.approx(expected, abs=1e-12)
    assert result.n_posts == 2


def test_single_post_returns_its_own_stance():
    """With one contributor the decay cancels entirely in the ratio."""
    result = compute_signal([Contribution(NOW - timedelta(hours=3), 0.42, followers=50)], NOW, SIX_HOURS)
    assert result is not None
    assert result.s_t == pytest.approx(0.42)


def test_extreme_age_does_not_produce_nan():
    """A post 30 days old at a 30-minute half-life underflows exp() to zero.

    Computed naively this is 0/0. The log-space form must still return the
    post's stance, because it is the only contributor.
    """
    ancient = [Contribution(NOW - timedelta(days=30), 0.5, followers=100, engagements=1)]
    result = compute_signal(ancient, NOW, 1800.0)
    assert result is not None
    assert not math.isnan(result.s_t)
    assert result.s_t == pytest.approx(0.5)


def test_future_posts_are_excluded():
    """Posts after the evaluation time would leak lookahead into a backtest."""
    posts = [
        Contribution(NOW - timedelta(hours=1), -1.0, followers=1000),
        Contribution(NOW + timedelta(hours=1), 1.0, followers=1000),
    ]
    result = compute_signal(posts, NOW, SIX_HOURS)
    assert result is not None
    assert result.n_posts == 1
    assert result.s_t == pytest.approx(-1.0)


def test_undefined_signal_returns_none_not_zero():
    """No posts, and zero-weight-only posts, are undefined — not neutral.

    Returning 0.0 would tell the LMSR the crowd is neutral when in fact nobody
    has said anything, which is a materially different claim.
    """
    assert compute_signal([], NOW, SIX_HOURS) is None
    zero_weight = [Contribution(NOW, 0.9, followers=0, engagements=99)]
    assert compute_signal(zero_weight, NOW, SIX_HOURS) is None


def test_recency_dominates():
    """Equal weights, opposite stances: the newer post wins."""
    posts = [
        Contribution(NOW - timedelta(minutes=5), 1.0, followers=500, engagements=5),
        Contribution(NOW - timedelta(hours=24), -1.0, followers=500, engagements=5),
    ]
    result = compute_signal(posts, NOW, SIX_HOURS)
    assert result is not None
    assert result.s_t > 0.8


def test_max_age_window():
    posts = [
        Contribution(NOW - timedelta(hours=1), 1.0, followers=100),
        Contribution(NOW - timedelta(hours=50), -1.0, followers=100),
    ]
    result = compute_signal(posts, NOW, SIX_HOURS, max_age=timedelta(hours=24))
    assert result is not None
    assert result.n_posts == 1


def test_signal_stays_within_bounds():
    posts = [
        Contribution(NOW - timedelta(minutes=i), 1.0 if i % 2 else -1.0, followers=10_000, engagements=100)
        for i in range(1, 200)
    ]
    result = compute_signal(posts, NOW, SIX_HOURS)
    assert result is not None
    assert -1.0 <= result.s_t <= 1.0


def test_stance_out_of_range_rejected():
    with pytest.raises(ValueError):
        compute_signal([Contribution(NOW, 1.5, followers=10)], NOW, SIX_HOURS)


class TestAdaptiveHalfLife:
    def test_quiet_market_keeps_baseline(self):
        assert effective_half_life(
            SIX_HOURS,
            min_half_life_seconds=1800,
            recent_post_count=5,
            baseline_post_rate=5,
            recent_price_move=0.0,
        ) == SIX_HOURS

    def test_volume_surge_shortens(self):
        # 90 posts against a baseline of 5/hr and a 3x trigger => factor 6.
        assert effective_half_life(
            SIX_HOURS,
            min_half_life_seconds=1800,
            recent_post_count=90,
            baseline_post_rate=5,
            recent_price_move=0.0,
        ) == pytest.approx(SIX_HOURS / 6)

    def test_price_move_shortens(self):
        # A 0.20 move against a 0.05 threshold => factor 4.
        assert effective_half_life(
            SIX_HOURS,
            min_half_life_seconds=1800,
            recent_post_count=0,
            baseline_post_rate=5,
            recent_price_move=0.20,
        ) == pytest.approx(SIX_HOURS / 4)

    def test_floor_is_respected(self):
        assert effective_half_life(
            SIX_HOURS,
            min_half_life_seconds=1800,
            recent_post_count=100_000,
            baseline_post_rate=1,
            recent_price_move=0.9,
        ) == 1800

    def test_disabled_ignores_triggers(self):
        assert effective_half_life(
            SIX_HOURS,
            min_half_life_seconds=1800,
            recent_post_count=100_000,
            baseline_post_rate=1,
            recent_price_move=0.9,
            enabled=False,
        ) == SIX_HOURS


def test_baseline_rate_uses_median_not_mean():
    """One viral hour must not raise the baseline enough to mask the next surge."""
    times = [NOW - timedelta(hours=3, minutes=m) for m in range(500)]
    times += [NOW - timedelta(hours=h) for h in range(5, 24)]
    rate = baseline_hourly_rate(times, NOW, lookback_hours=24)
    assert rate <= 2.0

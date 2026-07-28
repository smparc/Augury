"""Tests for the reference LMSR implementation.

The invariants asserted here are the same ones the C++ engine's Catch2 suite
checks against the shared golden vectors.
"""

from __future__ import annotations

import math

import pytest

from augury_signal.engine.lmsr import (
    LMSRMarket,
    calibrate_b_from_depth,
    cost,
    price_yes,
    prices,
    shares_to_move_price,
    trade_cost,
    worst_case_subsidy,
)


class TestCostFunction:
    def test_uniform_state_is_b_ln_k(self):
        """C(0) = b*ln(K) for a K-outcome market."""
        assert cost([0.0, 0.0], 100.0) == pytest.approx(100.0 * math.log(2))
        assert cost([0.0] * 4, 50.0) == pytest.approx(50.0 * math.log(4))

    def test_does_not_overflow_at_realistic_scale(self):
        """q/b of 500 overflows exp() in float64; log-sum-exp must not.

        The naive formula returns inf here, and every price derived from it
        becomes nan.
        """
        value = cost([50_000.0, 0.0], 100.0)
        assert math.isfinite(value)
        # C(q) -> max(q) as the leading term dominates.
        assert value == pytest.approx(50_000.0, rel=1e-9)

    def test_translation_invariance(self):
        """Adding a constant to every q raises C by exactly that constant."""
        base = cost([10.0, -5.0], 25.0)
        assert cost([110.0, 95.0], 25.0) == pytest.approx(base + 100.0)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            cost([1.0], 10.0)
        with pytest.raises(ValueError):
            cost([1.0, 2.0], 0.0)
        with pytest.raises(ValueError):
            cost([1.0, math.inf], 10.0)


class TestPrices:
    def test_prices_form_a_distribution(self):
        p = prices([120.0, -30.0, 45.0], 40.0)
        assert math.fsum(p) == pytest.approx(1.0)
        assert all(0.0 < x < 1.0 for x in p)

    def test_symmetric_state_is_even(self):
        assert prices([0.0, 0.0], 100.0) == pytest.approx([0.5, 0.5])

    def test_binary_helper_agrees_with_vector_form(self):
        vector = prices([37.0, -12.0], 55.0)[0]
        assert price_yes(37.0, -12.0, 55.0) == pytest.approx(vector)

    def test_logit_is_linear_in_share_imbalance(self):
        """logit(p_yes) = (q_yes - q_no)/b — the identity everything else uses."""
        b = 80.0
        for q_yes, q_no in [(0.0, 0.0), (100.0, 20.0), (-60.0, 15.0)]:
            p = price_yes(q_yes, q_no, b)
            assert math.log(p / (1 - p)) == pytest.approx((q_yes - q_no) / b)


class TestTrades:
    def test_zero_trade_is_free(self):
        assert trade_cost([10.0, 4.0], [0.0, 0.0], 30.0) == pytest.approx(0.0)

    def test_buying_raises_the_price(self):
        market = LMSRMarket(b=100.0)
        before = market.price
        market.buy_yes(50.0)
        assert market.price > before

    def test_round_trip_is_free_up_to_float_error(self):
        """Buy then sell the same size returns to the original state."""
        market = LMSRMarket(b=100.0)
        paid = market.buy_yes(75.0)
        refund = market.buy_yes(-75.0)
        assert paid + refund == pytest.approx(0.0, abs=1e-9)
        assert market.price == pytest.approx(0.5)

    def test_cost_bounded_by_shares(self):
        """A YES share can never cost more than 1: it pays at most 1 on resolution."""
        market = LMSRMarket(b=100.0)
        shares = 250.0
        assert market.buy_yes(shares) < shares

    def test_shares_to_move_price_is_exact(self):
        market = LMSRMarket(b=140.0)
        needed = shares_to_move_price(market.price, 0.8, market.b)
        market.buy_yes(needed)
        assert market.price == pytest.approx(0.8, abs=1e-12)

    def test_move_to_price_hits_target(self):
        market = LMSRMarket(b=60.0, q_yes=30.0)
        for target in (0.1, 0.5, 0.97):
            market.move_to_price(target)
            assert market.price == pytest.approx(target, abs=1e-9)

    def test_worst_case_subsidy(self):
        assert worst_case_subsidy(2, 100.0) == pytest.approx(100.0 * math.log(2))


class TestRecalibration:
    def test_price_is_preserved(self):
        """Changing b is a liquidity update, not new information about the outcome.

        If it moved the price, the simulation would be injecting signal that no
        trader ever expressed.
        """
        market = LMSRMarket(b=100.0)
        market.move_to_price(0.73)
        before = market.price
        market.recalibrate(925.0)
        assert market.price == pytest.approx(before, abs=1e-12)
        assert market.b == 925.0

    def test_higher_b_means_less_price_impact(self):
        thin, thick = LMSRMarket(b=10.0), LMSRMarket(b=10_000.0)
        thin.buy_yes(100.0)
        thick.buy_yes(100.0)
        assert abs(thin.price - 0.5) > abs(thick.price - 0.5)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            LMSRMarket(b=100.0).recalibrate(0.0)


class TestDepthCalibration:
    def test_matches_the_defining_equation(self):
        """b = depth / (logit(ask) - logit(bid))."""
        bid, ask, depth = 0.30, 0.35, 500.0
        width = math.log(ask / (1 - ask)) - math.log(bid / (1 - bid))
        got = calibrate_b_from_depth(depth, depth, bid, ask)
        assert got == pytest.approx(depth / width, rel=1e-9)

    def test_calibrated_b_reproduces_the_observed_move(self):
        """The whole point: consuming `depth` shares walks bid -> ask."""
        bid, ask, depth = 0.20, 0.26, 800.0
        b = calibrate_b_from_depth(depth, depth, bid, ask)
        market = LMSRMarket(b=b)
        market.move_to_price(bid)
        market.buy_yes(depth)
        assert market.price == pytest.approx(ask, abs=1e-9)

    def test_deeper_book_gives_larger_b(self):
        thin = calibrate_b_from_depth(10.0, 10.0, 0.4, 0.45)
        thick = calibrate_b_from_depth(10_000.0, 10_000.0, 0.4, 0.45)
        assert thick > thin

    def test_side_selection_uses_the_named_depth(self):
        """Real books are lopsided, so 'bid' and 'ask' genuinely differ."""
        bid_only = calibrate_b_from_depth(40.0, 69_298.0, 0.03, 0.04, side="bid")
        ask_only = calibrate_b_from_depth(40.0, 69_298.0, 0.03, 0.04, side="ask")
        both = calibrate_b_from_depth(40.0, 69_298.0, 0.03, 0.04, side="mean")
        assert bid_only < both < ask_only

    @pytest.mark.parametrize(
        "depth_bid,depth_ask,bid,ask",
        [
            (100.0, 100.0, 0.6, 0.4),   # crossed
            (100.0, 100.0, 0.5, 0.5),   # locked
            (None, None, 0.3, 0.4),     # no depth
            (0.0, 0.0, 0.3, 0.4),       # zero depth
            (100.0, 100.0, None, 0.4),  # one-sided quote
        ],
    )
    def test_unusable_books_fall_back(self, depth_bid, depth_ask, bid, ask):
        """Thin prediction markets produce these constantly; they must not raise."""
        assert calibrate_b_from_depth(depth_bid, depth_ask, bid, ask, fallback=42.0) == 42.0

    def test_result_is_clamped(self):
        huge = calibrate_b_from_depth(1e12, 1e12, 0.499, 0.501)
        assert huge <= 1.0e6


class TestSignalStep:
    def test_full_responsiveness_snaps_to_target(self):
        market = LMSRMarket(b=100.0)
        market.step_from_signal(0.8, responsiveness=1.0)
        assert market.price == pytest.approx(0.8, abs=1e-9)

    def test_partial_responsiveness_undershoots(self):
        """Below 1.0 the market integrates the signal instead of restating it."""
        market = LMSRMarket(b=100.0)
        market.step_from_signal(0.9, responsiveness=0.25)
        assert 0.5 < market.price < 0.9

    def test_repeated_steps_converge(self):
        market = LMSRMarket(b=100.0)
        for _ in range(100):
            market.step_from_signal(0.7, responsiveness=0.3)
        assert market.price == pytest.approx(0.7, abs=1e-6)

    def test_max_shares_caps_a_single_step(self):
        market = LMSRMarket(b=100.0)
        shares, _ = market.step_from_signal(0.999, responsiveness=1.0, max_shares=10.0)
        assert shares == pytest.approx(10.0)

    def test_rejects_bad_responsiveness(self):
        with pytest.raises(ValueError):
            LMSRMarket(b=100.0).step_from_signal(0.6, responsiveness=1.5)

"""Guard the committed cross-language golden vectors.

This is a regression tripwire, not a proof of correctness: it asserts that the
reference implementation still reproduces the vectors checked into
`schemas/testdata/`. Changing the math without regenerating them fails here
*before* the C++ and R suites start failing for reasons that look unrelated.

If this test fails and the change was intended, run
`python -m augury_signal.golden` and re-verify the other languages.
"""

from __future__ import annotations

import json

import pytest

from augury_signal.config import repo_root
from augury_signal.engine.lmsr import (
    calibrate_b_from_depth,
    cost,
    price_yes,
    prices,
    shares_to_move_price,
    trade_cost,
)
from augury_signal.golden import EPOCH
from augury_signal.signal.decay import Contribution, compute_signal, effective_half_life

TOL = 1e-9


@pytest.fixture(scope="module")
def vectors():
    path = repo_root() / "schemas" / "testdata" / "golden_vectors.json"
    if not path.exists():
        pytest.fail(f"{path} is missing; run `python -m augury_signal.golden`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_file_is_well_formed(vectors):
    assert vectors["tolerance"] == TOL
    for section in ("signal", "lmsr", "adaptive_decay", "calibration"):
        assert vectors[section], f"{section} section is empty"


def test_signal_vectors(vectors):
    from datetime import timedelta

    for case in vectors["signal"]:
        contributions = [
            Contribution(
                created_at=EPOCH - timedelta(seconds=post["age_seconds"]),
                stance=post["stance"],
                followers=post["followers"],
                engagements=post["engagements"],
            )
            for post in case["posts"]
        ]
        result = compute_signal(contributions, EPOCH, case["half_life_seconds"])
        assert result is not None, case["name"]
        assert result.s_t == pytest.approx(case["expected"]["s_t"], abs=TOL), case["name"]
        assert result.n_posts == case["expected"]["n_posts"], case["name"]


def test_lmsr_cost_and_price_vectors(vectors):
    for case in vectors["lmsr"]["cost"]:
        assert cost(case["q"], case["b"]) == pytest.approx(case["expected_cost"], abs=TOL)
        assert prices(case["q"], case["b"]) == pytest.approx(case["expected_prices"], abs=TOL)


def test_lmsr_trade_vectors(vectors):
    for case in vectors["lmsr"]["trade"]:
        got = trade_cost(case["q"], case["dq"], case["b"])
        assert got == pytest.approx(case["expected_cost"], abs=TOL)


def test_lmsr_binary_price_vectors(vectors):
    for case in vectors["lmsr"]["binary_price"]:
        got = price_yes(case["q_yes"], case["q_no"], case["b"])
        assert got == pytest.approx(case["expected_price_yes"], abs=TOL)


def test_shares_to_move_vectors(vectors):
    for case in vectors["lmsr"]["shares_to_move"]:
        got = shares_to_move_price(case["p_from"], case["p_to"], case["b"])
        assert got == pytest.approx(case["expected_shares"], abs=TOL)


def test_b_calibration_vectors(vectors):
    for case in vectors["lmsr"]["b_calibration"]:
        got = calibrate_b_from_depth(
            case["depth_bid"],
            case["depth_ask"],
            case["yes_bid"],
            case["yes_ask"],
            side=case["side"],
            fallback=100.0,
        )
        assert got == pytest.approx(case["expected_b"], abs=TOL)


def test_adaptive_decay_vectors(vectors):
    for case in vectors["adaptive_decay"]:
        got = effective_half_life(
            case["base_half_life_seconds"],
            min_half_life_seconds=case["min_half_life_seconds"],
            recent_post_count=case["recent_post_count"],
            baseline_post_rate=case["baseline_post_rate"],
            recent_price_move=case["recent_price_move"],
        )
        assert got == pytest.approx(case["expected_half_life_seconds"], abs=TOL), case["name"]

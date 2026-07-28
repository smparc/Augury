"""Generate the cross-language golden vectors in `schemas/testdata/`.

Augury implements the same equations in three languages: this package (the
reference), the C++ engine, and the R analytics module. Independent
implementations of a formula agree only by luck unless something forces the
issue, so each one is tested against a shared set of fixed inputs and expected
outputs produced here.

Regenerate with `python -m augury_signal.golden` whenever the reference math
changes, and expect the C++ and R suites to start failing until they are
brought back into line. That failure is the mechanism working.

Values are emitted at full float64 repr precision; the consuming suites compare
to 1e-9, loose enough to absorb differences in summation order and tight enough
that a genuine formula divergence cannot hide.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import repo_root
from .engine.lmsr import (
    calibrate_b_from_depth,
    cost,
    price_yes,
    prices,
    shares_to_move_price,
    trade_cost,
)
from .models import Calibration
from .signal.calibration import logit, sigmoid, to_probability
from .signal.decay import Contribution, compute_signal, effective_half_life, post_weight

# Fixed epoch so the vectors are byte-identical between runs. Anything
# time-dependent here is expressed as an offset from this instant.
EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _signal_vectors() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    scenarios: list[tuple[str, list[tuple[float, float, int, int]], float]] = [
        # (name, [(age_hours, stance, followers, engagements)], half_life_seconds)
        ("single_post", [(3.0, 0.42, 50, 0)], 21600.0),
        ("two_posts_opposing", [(1.0, 0.8, 1000, 10), (12.0, -0.9, 1000, 10)], 21600.0),
        ("recency_dominates", [(0.0833, 1.0, 500, 5), (24.0, -1.0, 500, 5)], 21600.0),
        ("heavy_tail", [(0.5, 0.6, 250000, 5000), (2.0, -0.4, 100, 1)], 21600.0),
        # Underflows exp() naively: 30 days at a 30-minute half-life.
        ("extreme_age", [(720.0, 0.5, 100, 1)], 1800.0),
        ("short_half_life", [(0.25, 0.9, 8000, 120), (1.5, -0.7, 12000, 40)], 3600.0),
        (
            "many_posts",
            [(float(i) * 0.5, (1.0 if i % 2 == 0 else -1.0) * (0.5 + i / 40.0), 1000 + 100 * i, i)
             for i in range(20)],
            21600.0,
        ),
    ]

    for name, posts, half_life in scenarios:
        contributions = [
            Contribution(
                created_at=EPOCH - timedelta(hours=age),
                stance=stance,
                followers=followers,
                engagements=engagements,
            )
            for age, stance, followers, engagements in posts
        ]
        result = compute_signal(contributions, EPOCH, half_life)
        assert result is not None, name

        cases.append(
            {
                "name": name,
                "half_life_seconds": half_life,
                "posts": [
                    {
                        "age_seconds": age * 3600.0,
                        "stance": stance,
                        "followers": followers,
                        "engagements": engagements,
                        "weight": post_weight(followers, engagements),
                    }
                    for age, stance, followers, engagements in posts
                ],
                "expected": {
                    "s_t": result.s_t,
                    "n_posts": result.n_posts,
                    "weight_sum": result.weight_sum,
                },
            }
        )

    return cases


def _lmsr_vectors() -> dict[str, Any]:
    cost_cases = [
        {"q": [0.0, 0.0], "b": 100.0},
        {"q": [100.0, 0.0], "b": 100.0},
        {"q": [-250.0, 125.0], "b": 40.0},
        # q/b = 500: overflows exp() in float64 without max-subtraction.
        {"q": [50000.0, 0.0], "b": 100.0},
        {"q": [0.0, 0.0, 0.0, 0.0], "b": 50.0},
        {"q": [10.0, -5.0, 30.0], "b": 12.5},
    ]
    for case in cost_cases:
        case["expected_cost"] = cost(case["q"], case["b"])
        case["expected_prices"] = prices(case["q"], case["b"])

    trade_cases = [
        {"q": [0.0, 0.0], "dq": [50.0, 0.0], "b": 100.0},
        {"q": [120.0, 30.0], "dq": [-60.0, 0.0], "b": 75.0},
        {"q": [0.0, 0.0], "dq": [250.0, 0.0], "b": 100.0},
    ]
    for case in trade_cases:
        case["expected_cost"] = trade_cost(case["q"], case["dq"], case["b"])

    binary_cases = [
        {"q_yes": 0.0, "q_no": 0.0, "b": 100.0},
        {"q_yes": 137.0, "q_no": -12.0, "b": 55.0},
        {"q_yes": -900.0, "q_no": 100.0, "b": 250.0},
    ]
    for case in binary_cases:
        case["expected_price_yes"] = price_yes(case["q_yes"], case["q_no"], case["b"])

    move_cases = [
        {"p_from": 0.5, "p_to": 0.8, "b": 140.0},
        {"p_from": 0.2, "p_to": 0.26, "b": 800.0},
        {"p_from": 0.97, "p_to": 0.03, "b": 60.0},
    ]
    for case in move_cases:
        case["expected_shares"] = shares_to_move_price(case["p_from"], case["p_to"], case["b"])

    # Calibration inputs include a real Kalshi book observed 2026-07-27 on
    # KXFEDDECISION-26SEP-C25: bid 0.03 x 40, ask 0.04 x 69298.61.
    calibration_cases = [
        {"depth_bid": 500.0, "depth_ask": 500.0, "yes_bid": 0.30, "yes_ask": 0.35, "side": "mean"},
        {"depth_bid": 40.0, "depth_ask": 69298.61, "yes_bid": 0.03, "yes_ask": 0.04, "side": "mean"},
        {"depth_bid": 40.0, "depth_ask": 69298.61, "yes_bid": 0.03, "yes_ask": 0.04, "side": "bid"},
        {"depth_bid": 40.0, "depth_ask": 69298.61, "yes_bid": 0.03, "yes_ask": 0.04, "side": "ask"},
        # Crossed book: must fall back rather than raise.
        {"depth_bid": 100.0, "depth_ask": 100.0, "yes_bid": 0.6, "yes_ask": 0.4, "side": "mean"},
    ]
    for case in calibration_cases:
        case["expected_b"] = calibrate_b_from_depth(
            case["depth_bid"],
            case["depth_ask"],
            case["yes_bid"],
            case["yes_ask"],
            side=case["side"],
            fallback=100.0,
        )

    return {
        "cost": cost_cases,
        "trade": trade_cases,
        "binary_price": binary_cases,
        "shares_to_move": move_cases,
        "b_calibration": calibration_cases,
    }


def _adaptive_vectors() -> list[dict[str, Any]]:
    # (name, recent_post_count, baseline_post_rate, recent_price_move)
    scenarios: list[tuple[str, float, float, float]] = [
        ("quiet", 5.0, 5.0, 0.0),
        ("volume_surge", 90.0, 5.0, 0.0),
        ("price_shock", 0.0, 5.0, 0.20),
        ("both", 60.0, 5.0, 0.30),
        ("floored", 100_000.0, 1.0, 0.9),
    ]

    cases: list[dict[str, Any]] = []
    for name, post_count, baseline, price_move in scenarios:
        cases.append(
            {
                "name": name,
                "recent_post_count": post_count,
                "baseline_post_rate": baseline,
                "recent_price_move": price_move,
                "base_half_life_seconds": 21600.0,
                "min_half_life_seconds": 1800.0,
                "expected_half_life_seconds": effective_half_life(
                    21600.0,
                    min_half_life_seconds=1800.0,
                    recent_post_count=post_count,
                    baseline_post_rate=baseline,
                    recent_price_move=price_move,
                ),
            }
        )
    return cases


def _calibration_vectors() -> dict[str, Any]:
    signals = [-1.0, -0.75, -0.3, 0.0, 0.25, 0.6, 1.0]
    return {
        "affine": [
            {"s_t": s, "expected_p": to_probability(s, Calibration.AFFINE)} for s in signals
        ],
        "logit": [
            {"p": p, "expected_logit": logit(p)}
            for p in [0.001, 0.03, 0.25, 0.5, 0.75, 0.97, 0.999]
        ],
        "sigmoid": [
            {"x": x, "expected_p": sigmoid(x)} for x in [-30.0, -3.0, -0.5, 0.0, 0.5, 3.0, 30.0]
        ],
    }


def build() -> dict[str, Any]:
    return {
        "_comment": (
            "Cross-language golden vectors for Augury. Generated by "
            "`python -m augury_signal.golden` from the Python reference "
            "implementation. The C++ (Catch2) and R (testthat) suites assert "
            "their own implementations reproduce these to within 1e-9. Do not "
            "hand-edit."
        ),
        "epoch": EPOCH.isoformat().replace("+00:00", "Z"),
        "tolerance": 1e-9,
        "signal": _signal_vectors(),
        "lmsr": _lmsr_vectors(),
        "adaptive_decay": _adaptive_vectors(),
        "calibration": _calibration_vectors(),
    }


def write(path: Path | None = None) -> Path:
    target = path or repo_root() / "schemas" / "testdata" / "golden_vectors.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    target = write()
    data = json.loads(target.read_text(encoding="utf-8"))
    counts = {
        "signal": len(data["signal"]),
        "lmsr.cost": len(data["lmsr"]["cost"]),
        "lmsr.trade": len(data["lmsr"]["trade"]),
        "lmsr.binary_price": len(data["lmsr"]["binary_price"]),
        "lmsr.shares_to_move": len(data["lmsr"]["shares_to_move"]),
        "lmsr.b_calibration": len(data["lmsr"]["b_calibration"]),
        "adaptive_decay": len(data["adaptive_decay"]),
    }
    print(f"wrote {target}")
    for name, count in counts.items():
        print(f"  {name}: {count} vectors")


if __name__ == "__main__":
    main()

"""Reference LMSR implementation.

Mirrored by the C++ engine in `apps/augury-engine`. The two are checked against
the same golden vectors in `schemas/testdata/`; if this changes, regenerate them.
"""

from .lmsr import (
    LMSRMarket,
    calibrate_b_from_depth,
    cost,
    price_yes,
    prices,
    shares_to_move_price,
    trade_cost,
    worst_case_subsidy,
)

__all__ = [
    "LMSRMarket",
    "calibrate_b_from_depth",
    "cost",
    "price_yes",
    "prices",
    "shares_to_move_price",
    "trade_cost",
    "worst_case_subsidy",
]

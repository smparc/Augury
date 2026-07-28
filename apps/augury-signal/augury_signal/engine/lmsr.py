"""Hanson's Logarithmic Market Scoring Rule.

    C(q) = b * ln( sum_j exp(q_j / b) )
    p_k  = exp(q_k / b) / sum_j exp(q_j / b)

A trade that moves the outstanding-share vector from q to q + dq costs
C(q + dq) - C(q).

Every exponential here goes through a max-subtracted log-sum-exp. That is not
defensive style, it is required: with the b values this system actually
calibrates (single digits to low hundreds, from real book depth) and share
counts in the thousands, q_j / b reaches several hundred and `exp` overflows to
inf in float64. The naive formula returns inf or nan on ordinary inputs.

Binary-market identity used throughout: with q = (q_yes, q_no),

    p_yes = sigmoid( (q_yes - q_no) / b )   =>   logit(p_yes) = (q_yes - q_no) / b

so buying `n` YES shares moves logit(p) by exactly n / b. That linearity in
logit space is what makes `b` interpretable as liquidity and is the basis of
both `shares_to_move_price` and `calibrate_b_from_depth`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..signal.calibration import clamp_probability, logit

# Below this the market is so thin that a single share moves the price across
# the whole range; above it, no realistic trade moves it at all. Both extremes
# make the simulation useless, so calibration is clamped into this band.
B_MIN = 1.0
B_MAX = 1.0e6


def _log_sum_exp(values: Sequence[float]) -> float:
    """ln(sum(exp(v))), max-subtracted."""
    if not values:
        raise ValueError("log_sum_exp of an empty sequence")
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    total = math.fsum(math.exp(v - peak) for v in values)
    return peak + math.log(total)


def _validate(q: Sequence[float], b: float) -> None:
    if len(q) < 2:
        raise ValueError(f"LMSR needs at least 2 outcomes, got {len(q)}")
    if b <= 0:
        raise ValueError(f"liquidity parameter b must be positive, got {b}")
    if any(not math.isfinite(x) for x in q):
        raise ValueError(f"outstanding shares must all be finite, got {list(q)}")


def cost(q: Sequence[float], b: float) -> float:
    """C(q) = b * ln(sum_j exp(q_j / b))."""
    _validate(q, b)
    return b * _log_sum_exp([x / b for x in q])


def prices(q: Sequence[float], b: float) -> list[float]:
    """Price vector p, a proper probability distribution summing to 1."""
    _validate(q, b)
    scaled = [x / b for x in q]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


def price_yes(q_yes: float, q_no: float, b: float) -> float:
    """Binary-market YES price, via the sigmoid identity."""
    _validate((q_yes, q_no), b)
    z = (q_yes - q_no) / b
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def trade_cost(q: Sequence[float], dq: Sequence[float], b: float) -> float:
    """Cost of moving the market from q to q + dq.

    Positive means the trader pays. Computed as a difference of two cost
    functions, each already log-sum-exp stable.
    """
    if len(q) != len(dq):
        raise ValueError(f"q and dq differ in length: {len(q)} vs {len(dq)}")
    after = [a + d for a, d in zip(q, dq, strict=True)]
    return cost(after, b) - cost(q, b)


def shares_to_move_price(p_from: float, p_to: float, b: float) -> float:
    """YES shares that move the binary price from `p_from` to `p_to`.

        n = b * ( logit(p_to) - logit(p_from) )

    Positive means buying YES. Both prices are clamped off {0, 1}: reaching
    either boundary exactly requires infinite shares, which is a property of
    LMSR, not an error to raise on.
    """
    if b <= 0:
        raise ValueError(f"liquidity parameter b must be positive, got {b}")
    return b * (logit(p_to) - logit(p_from))


def worst_case_subsidy(n_outcomes: int, b: float) -> float:
    """Maximum the market maker can lose: b * ln(K).

    The other half of what `b` controls. Raising b to make the synthetic market
    less jumpy directly raises the subsidy being committed, which is why b is
    calibrated from observed depth rather than turned up until the output looks
    smooth.
    """
    if n_outcomes < 2:
        raise ValueError(f"need at least 2 outcomes, got {n_outcomes}")
    if b <= 0:
        raise ValueError(f"liquidity parameter b must be positive, got {b}")
    return b * math.log(n_outcomes)


def calibrate_b_from_depth(
    depth_bid: float | None,
    depth_ask: float | None,
    yes_bid: float | None,
    yes_ask: float | None,
    *,
    side: str = "mean",
    fallback: float = 100.0,
) -> float:
    """Fit `b` so the synthetic market is as hard to move as the real one.

    In the real book, consuming the size resting between the bid and the ask
    walks the price across the quoted range. In LMSR, moving the price the same
    distance takes `b * (logit(ask) - logit(bid))` shares. Setting those equal:

        b = depth / (logit(ask) - logit(bid))

    `side` selects which depth is used: 'ask' (cost of pushing the price up),
    'bid' (pushing it down), or 'mean' of the two. Real books are often lopsided
    — the sample this was built against had 40 shares on the bid against ~69,000
    on the ask — so the choice genuinely changes the answer, and 'mean' is the
    symmetric default rather than an obviously correct one.

    Returns `fallback` when the book is unusable: crossed or locked quotes, a
    missing side, or zero depth. Those happen routinely on thin prediction
    markets and are not worth raising on mid-simulation.
    """
    if side not in {"bid", "ask", "mean"}:
        raise ValueError(f"side must be 'bid', 'ask', or 'mean', got {side!r}")
    if fallback <= 0:
        raise ValueError(f"fallback must be positive, got {fallback}")

    if yes_bid is None or yes_ask is None:
        return fallback
    if not (0.0 <= yes_bid < yes_ask <= 1.0):
        # Crossed, locked, or out of range: no usable width to calibrate against.
        return fallback

    width = logit(yes_ask) - logit(yes_bid)
    if width <= 0:
        return fallback

    candidates: list[float] = []
    if side in {"ask", "mean"} and depth_ask:
        candidates.append(depth_ask)
    if side in {"bid", "mean"} and depth_bid:
        candidates.append(depth_bid)

    if not candidates:
        return fallback

    depth = math.fsum(candidates) / len(candidates)
    if depth <= 0:
        return fallback

    return min(B_MAX, max(B_MIN, depth / width))


@dataclass(slots=True)
class LMSRMarket:
    """A binary LMSR market maker with mutable state.

    `b` is settable between trades: as the real book thickens or thins, the
    synthetic market's sensitivity should follow it. Re-calibrating b does not
    change the current price — `q` is rescaled to hold `logit(p) = (q_yes -
    q_no) / b` fixed — because a liquidity update is not new information about
    the outcome, and letting it jump the price would inject signal that no one
    traded on.
    """

    b: float = 100.0
    q_yes: float = 0.0
    q_no: float = 0.0
    realized_cost: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        _validate((self.q_yes, self.q_no), self.b)

    @property
    def q(self) -> tuple[float, float]:
        return (self.q_yes, self.q_no)

    @property
    def price(self) -> float:
        """Current YES price."""
        return price_yes(self.q_yes, self.q_no, self.b)

    @property
    def cost_value(self) -> float:
        return cost(self.q, self.b)

    def buy_yes(self, shares: float) -> float:
        """Buy `shares` of YES (negative sells). Returns the cost paid."""
        if not math.isfinite(shares):
            raise ValueError(f"shares must be finite, got {shares}")
        paid = trade_cost(self.q, (shares, 0.0), self.b)
        self.q_yes += shares
        self.realized_cost += paid
        self.history.append((shares, paid))
        return paid

    def move_to_price(self, target: float) -> tuple[float, float]:
        """Trade exactly enough to set the YES price to `target`.

        Returns (shares_traded, cost). The target is clamped off {0, 1}, since
        LMSR reaches the boundaries only in the limit.
        """
        target = clamp_probability(target)
        shares = shares_to_move_price(self.price, target, self.b)
        return shares, self.buy_yes(shares)

    def recalibrate(self, new_b: float) -> None:
        """Update `b` while holding the current price fixed.

        logit(p) = (q_yes - q_no) / b, so preserving p under a change of b means
        scaling the share imbalance by the same ratio.
        """
        if new_b <= 0:
            raise ValueError(f"liquidity parameter b must be positive, got {new_b}")
        ratio = new_b / self.b
        imbalance = (self.q_yes - self.q_no) * ratio
        # Keep the representation canonical: all imbalance on the YES leg.
        self.q_yes, self.q_no = imbalance, 0.0
        self.b = new_b

    def step_from_signal(
        self,
        p_target: float,
        *,
        responsiveness: float = 1.0,
        max_shares: float | None = None,
    ) -> tuple[float, float]:
        """Move the simulated price part of the way toward the signal's forecast.

        `responsiveness` in (0, 1] is the fraction of the logit-space gap closed
        in one step. 1.0 snaps straight to the target, which makes the synthetic
        price a redundant restatement of the signal; below 1.0 the market
        integrates the signal over time, which is the behavior actually worth
        comparing against a real order book.

        `max_shares` caps a single step, so one outlier signal cannot move the
        synthetic market further than any real trader plausibly would.
        """
        if not 0.0 < responsiveness <= 1.0:
            raise ValueError(f"responsiveness must be in (0, 1], got {responsiveness}")

        target = clamp_probability(p_target)
        gap = logit(target) - logit(self.price)
        shares = self.b * gap * responsiveness

        if max_shares is not None:
            if max_shares < 0:
                raise ValueError(f"max_shares must be non-negative, got {max_shares}")
            shares = max(-max_shares, min(max_shares, shares))

        return shares, self.buy_yes(shares)

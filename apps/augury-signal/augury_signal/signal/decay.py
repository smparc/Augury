"""The stance-weighted, time-decayed social signal S(t).

                sum_i  w_i * stance_i * exp(-lambda * (t - t_i))
    S(t)  =    -------------------------------------------------
                sum_i  w_i * exp(-lambda * (t - t_i))

    w_i     = ln(1 + followers_i) * (1 + engagements_i)
    lambda  = ln(2) / t_half

Implementation note that matters for correctness, not just style: the sums are
evaluated in log space with the maximum factored out. A 6-hour half-life gives
lambda ~ 3.2e-5/s, so a week-old post carries exp(-19.4); at the 30-minute
half-life the adaptive path can reach, that same post carries exp(-233), which
underflows to exactly 0.0 in float64. Computing the ratio directly would then
be 0/0 for any window whose posts are all old. Factoring out the max makes the
dominant term exp(0) = 1 and the ratio well-defined regardless of scale.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..models import Calibration, SignalPoint

LN2 = math.log(2.0)

# Floor for the adaptive-decay volume baseline. Most tracked markets are quiet
# most of the time, so the median hourly post count is frequently 0.
MIN_BASELINE_POST_RATE = 1.0


def decay_lambda(half_life_seconds: float) -> float:
    """lambda = ln(2) / t_half."""
    if half_life_seconds <= 0:
        raise ValueError(f"half_life_seconds must be positive, got {half_life_seconds}")
    return LN2 / half_life_seconds


def post_weight(followers: int, engagements: int) -> float:
    """w = ln(1 + followers) * (1 + engagements).

    Note the degenerate case this formula has by construction: an account with
    zero followers gets ln(1) = 0 and therefore contributes *nothing*, no matter
    how much engagement the post drew. That is what the specification says, so
    it is what this implements — but it means a window containing only
    zero-follower posts has no defined signal, and `compute_signal` returns None
    there rather than inventing one.
    """
    if followers < 0:
        raise ValueError(f"followers must be non-negative, got {followers}")
    if engagements < 0:
        raise ValueError(f"engagements must be non-negative, got {engagements}")
    return math.log1p(followers) * (1.0 + engagements)


@dataclass(frozen=True, slots=True)
class Contribution:
    """One post's input to S(t): when it was posted, its weight, and its stance."""

    created_at: datetime
    stance: float
    followers: int = 0
    engagements: int = 0
    weight_override: float | None = None

    @property
    def weight(self) -> float:
        if self.weight_override is not None:
            return self.weight_override
        return post_weight(self.followers, self.engagements)


def compute_signal(
    contributions: Iterable[Contribution],
    at: datetime,
    half_life_seconds: float,
    *,
    market_id: str = "",
    model_version: str = "",
    max_age: timedelta | None = None,
    calibrate: Calibration | None = None,
) -> SignalPoint | None:
    """Evaluate S(t) at `at`.

    Returns None when the signal is undefined — no posts in the window, or every
    post carries zero weight. A caller must not substitute 0.0 for None: zero is
    a real, neutral signal value and conflating the two would tell the LMSR that
    the crowd is neutral when in fact nobody has said anything.

    Posts timestamped after `at` are excluded. Allowing them would leak future
    information into a backtest, which is exactly the bug that makes a lead-lag
    result look spectacular and be worthless.
    """
    lam = decay_lambda(half_life_seconds)
    cutoff = at - max_age if max_age is not None else None

    # log_terms[i] = ln(w_i) - lambda * (t - t_i)
    log_terms: list[float] = []
    stances: list[float] = []

    for c in contributions:
        if c.created_at > at:
            continue
        if cutoff is not None and c.created_at < cutoff:
            continue
        weight = c.weight
        if weight <= 0.0:
            # ln(0) is undefined and the term is identically zero anyway.
            continue
        if not -1.0 <= c.stance <= 1.0:
            raise ValueError(f"stance {c.stance} outside [-1, 1]")

        age_seconds = (at - c.created_at).total_seconds()
        log_terms.append(math.log(weight) - lam * age_seconds)
        stances.append(c.stance)

    if not log_terms:
        return None

    peak = max(log_terms)
    # Both sums are relative to the peak term, so the common exp(peak) factor
    # cancels in the ratio and never has to be representable.
    relative = [math.exp(lt - peak) for lt in log_terms]
    denominator = math.fsum(relative)
    numerator = math.fsum(s * r for s, r in zip(stances, relative, strict=True))

    if denominator == 0.0:  # unreachable: relative contains exp(0) == 1.0
        return None

    s_t = numerator / denominator
    # Guard against float error pushing a weighted mean of values in [-1,1]
    # a hair outside the interval, which would trip the SignalPoint validator.
    s_t = min(1.0, max(-1.0, s_t))

    # The true weight sum, restored to absolute scale. This one *may* underflow
    # to 0.0 when every contributing post has decayed away, and that is
    # meaningful: it says s_t is being carried by numerically negligible weight.
    try:
        weight_sum = math.exp(peak) * denominator
    except OverflowError:
        weight_sum = math.inf

    p_hat: float | None = None
    calibration: Calibration | None = None
    if calibrate is not None:
        from .calibration import to_probability

        p_hat = to_probability(s_t, calibrate)
        calibration = calibrate

    return SignalPoint(
        market_id=market_id,
        ts=at,
        model_version=model_version,
        s_t=s_t,
        n_posts=len(log_terms),
        weight_sum=weight_sum,
        half_life_seconds=half_life_seconds,
        p_hat=p_hat,
        calibration=calibration,
    )


def effective_half_life(
    base_half_life_seconds: float,
    *,
    min_half_life_seconds: float,
    recent_post_count: float,
    baseline_post_rate: float,
    recent_price_move: float,
    volume_surge_multiple: float = 3.0,
    price_move_threshold: float = 0.05,
    enabled: bool = True,
) -> float:
    """Enhancement 3 — adaptive decay.

    During a debate, an FOMC statement, or an election night, a 6-hour half-life
    means posts from before the news still dominate the average. This shortens
    the half-life in proportion to how unusual the current window is, so the
    signal tracks what is being said now.

    Two independent triggers, whichever is stronger:

      * volume surge — posts in the trailing window versus the market's own
        baseline rate, in units of `volume_surge_multiple`;
      * price move — realized movement in the trailing window, in units of
        `price_move_threshold`.

    Both are expressed as "how many times past the trigger are we", the larger
    of the two divides the half-life, and the result is floored at
    `min_half_life_seconds`. Below the triggers the factor is 1.0, so a quiet
    market keeps the baseline half-life exactly.
    """
    if base_half_life_seconds <= 0:
        raise ValueError("base_half_life_seconds must be positive")
    if min_half_life_seconds <= 0:
        raise ValueError("min_half_life_seconds must be positive")

    if not enabled:
        return base_half_life_seconds

    volume_factor = 1.0
    if volume_surge_multiple > 0:
        # A sparse market has a median hourly post count of zero, which would
        # make the surge ratio undefined and silently disable the volume
        # trigger exactly where it is most useful — a quiet market that
        # suddenly gets 30 posts in an hour is the clearest possible surge.
        # Floor the baseline at one post per hour so the comparison stays
        # meaningful.
        baseline = max(baseline_post_rate, MIN_BASELINE_POST_RATE)
        surge = recent_post_count / (baseline * volume_surge_multiple)
        volume_factor = max(1.0, surge)

    price_factor = 1.0
    if price_move_threshold > 0:
        price_factor = max(1.0, abs(recent_price_move) / price_move_threshold)

    factor = max(volume_factor, price_factor)
    return max(min_half_life_seconds, base_half_life_seconds / factor)


def baseline_hourly_rate(post_times: Sequence[datetime], at: datetime, lookback_hours: int = 24) -> float:
    """Median posts-per-hour over the trailing window, used as the surge baseline.

    Median rather than mean: one viral hour would otherwise raise the baseline
    enough to mask the next surge, which is precisely when the shortened
    half-life is wanted.
    """
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive")

    start = at - timedelta(hours=lookback_hours)
    buckets = [0] * lookback_hours
    for t in post_times:
        if not (start <= t <= at):
            continue
        index = int((t - start).total_seconds() // 3600)
        # A post exactly at `at` lands one past the last bucket.
        buckets[min(index, lookback_hours - 1)] += 1

    ordered = sorted(buckets)
    mid = lookback_hours // 2
    if lookback_hours % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0

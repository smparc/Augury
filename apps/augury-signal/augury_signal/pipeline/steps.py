"""Individual pipeline steps.

Each function is deliberately independent and re-runnable: Dagster wires them
into an asset graph, the CLI calls them directly, and both must be able to
re-run a step over an overlapping window without corrupting anything. That is
why every step ends in an idempotent upsert rather than an append.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ..clients.kalshi import KalshiClient
from ..clients.polymarket import PolymarketClient
from ..clients.x import XClient
from ..config import MarketConfig, Settings, TrackedMarket
from ..db import Database
from ..engine.lmsr import LMSRMarket, calibrate_b_from_depth
from ..models import (
    Calibration,
    Market,
    MarketStatus,
    Post,
    PriceSource,
    PriceTick,
    SignalPoint,
    SimTick,
    Stance,
    Venue,
)
from ..signal.calibration import PlattParams, to_probability
from ..signal.decay import (
    Contribution,
    baseline_hourly_rate,
    compute_signal,
    effective_half_life,
)
from ..signal.stance import EnsembleStanceModel, StanceModel

log = logging.getLogger(__name__)


def venue_client(venue: Venue, settings: Settings) -> KalshiClient | PolymarketClient:
    if venue is Venue.KALSHI:
        return KalshiClient(settings.kalshi_base_url)
    return PolymarketClient(settings.polymarket_gamma_url, settings.polymarket_clob_url)


# ---------------------------------------------------------------------------
# Config -> database
# ---------------------------------------------------------------------------


def sync_config(db: Database, config: MarketConfig, settings: Settings) -> dict[str, int]:
    """Mirror `markets.yaml` into `markets` and `market_queries`.

    Market metadata is refreshed from the venue at the same time, so a market
    that closed since the config was written is recorded as closed rather than
    being polled forever.
    """
    markets = 0
    terms = 0

    for tracked in config.markets:
        market = tracked.market
        try:
            client = venue_client(market.venue, settings)
            fetched = client.get_market(market.ticker, target=market.target)
            # The YAML is authoritative for the stance target; the venue is
            # authoritative for status, timing, and resolution.
            market = Market(
                market_id=fetched.market_id,
                venue=fetched.venue,
                ticker=fetched.ticker,
                title=fetched.title,
                target=tracked.market.target,
                series_ticker=tracked.market.series_ticker or fetched.series_ticker,
                open_time=fetched.open_time,
                close_time=fetched.close_time,
                status=fetched.status,
                resolved_outcome=fetched.resolved_outcome,
                resolution_time=fetched.resolution_time,
                tracked=True,
            )
        except Exception as exc:
            log.warning("could not refresh %s from venue (%s); storing config values", market.market_id, exc)

        db.upsert_market(market)
        markets += 1
        terms += db.replace_market_queries(tracked.query)

    return {"markets": markets, "query_terms": terms}


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def poll_prices(
    tracked: TrackedMarket,
    settings: Settings,
    *,
    lookback_days: int = 14,
    include_history: bool = True,
) -> list[PriceTick]:
    """Fetch a market's price history and current book.

    Returns candle ticks plus one live book snapshot. The book snapshot is what
    carries depth, and depth is what `b` is calibrated from.
    """
    market = tracked.market
    client = venue_client(market.venue, settings)
    ticks: list[PriceTick] = []

    if include_history:
        try:
            if isinstance(client, KalshiClient):
                series = market.series_ticker or market.ticker.split("-")[0]
                ticks.extend(
                    client.get_candlesticks(
                        series, market.ticker, period_minutes=60, lookback_days=lookback_days
                    )
                )
            else:
                interval = "1m" if lookback_days > 7 else "1w"
                ticks.extend(
                    client.get_price_history(market.ticker, interval=interval, fidelity_minutes=60)
                )
        except Exception as exc:
            log.warning("price history unavailable for %s: %s", market.market_id, exc)

    try:
        ticks.append(client.get_orderbook(market.ticker))
    except Exception as exc:
        log.warning("order book unavailable for %s: %s", market.market_id, exc)

    return ticks


def latest_book(ticks: Sequence[PriceTick]) -> PriceTick | None:
    books = [t for t in ticks if t.source is PriceSource.BOOK]
    return max(books, key=lambda t: t.ts) if books else None


# ---------------------------------------------------------------------------
# Posts
# ---------------------------------------------------------------------------


def ingest_posts(
    tracked: TrackedMarket,
    config: MarketConfig,
    x_client: XClient,
    *,
    max_results: int = 100,
    lookback_days: int = 7,
) -> list[Post]:
    query = tracked.query.to_x_query(config.language, config.global_exclude)
    start = datetime.now(UTC) - timedelta(days=lookback_days)
    return x_client.search(
        query, tracked.market.market_id, max_results=max_results, start_time=start
    )


def score_posts(posts: Sequence[Post], model: StanceModel, target: str) -> list[Stance]:
    """Score posts against the market's target claim.

    When the model is an ensemble, the per-post disagreement between the primary
    and the lexicon baseline is captured as a data-quality flag.
    """
    if not posts:
        return []

    texts = [p.text for p in posts]

    if isinstance(model, EnsembleStanceModel):
        pairs = model.score_with_disagreement(texts, target)
        return [
            Stance(
                post_id=post.post_id,
                market_id=post.market_id,
                created_at=post.created_at,
                model_version=model.model_version,
                stance=score.stance,
                confidence=score.confidence,
                disagreement=disagreement,
            )
            for post, (score, disagreement) in zip(posts, pairs, strict=True)
        ]

    scores = model.score(texts, target)
    return [
        Stance(
            post_id=post.post_id,
            market_id=post.market_id,
            created_at=post.created_at,
            model_version=model.model_version,
            stance=score.stance,
            confidence=score.confidence,
        )
        for post, score in zip(posts, scores, strict=True)
    ]


# ---------------------------------------------------------------------------
# Signal series
# ---------------------------------------------------------------------------


def _price_at(
    ticks: Sequence[PriceTick], times: Sequence[datetime], at: datetime
) -> float | None:
    """Most recent price at or before `at`, within one bar.

    `ticks` must be sorted ascending by `ts`, with `times` its timestamps. The
    linear scan this replaces ran once per grid bar over the whole price series,
    making the signal build quadratic in history length — an 8,000-bar market
    (about a year of hourly candles) did ~64M comparisons to produce a series
    that is O(n log n) work.
    """
    if not ticks:
        return None
    index = bisect_right(times, at)
    if index == 0:
        return None
    best = ticks[index - 1]
    if at - best.ts > timedelta(hours=1):
        return None
    return best.mid


def build_signal_series(
    tracked: TrackedMarket,
    posts: Sequence[Post],
    stances: Sequence[Stance],
    price_ticks: Sequence[PriceTick],
    *,
    model_version: str,
    settings: Settings,
    freq_hours: int = 1,
    calibration: Calibration = Calibration.AFFINE,
    platt: PlattParams | None = None,
) -> list[SignalPoint]:
    """Evaluate S(t) on a regular grid across the posts' time span.

    The half-life at each bar is computed from that bar's own conditions —
    trailing post volume against the market's baseline, and trailing realized
    price movement — so a debate hour decays faster than a quiet overnight one.
    """
    if not posts or not stances:
        return []

    stance_by_post = {(s.post_id, s.created_at): s.stance for s in stances}
    contributions = [
        Contribution(
            created_at=p.created_at,
            stance=stance_by_post[(p.post_id, p.created_at)],
            followers=p.followers,
            engagements=p.engagements,
        )
        for p in posts
        if (p.post_id, p.created_at) in stance_by_post and p.is_accepted
    ]
    if not contributions:
        return []

    post_times = sorted(c.created_at for c in contributions)
    start = post_times[0]
    end = max(post_times[-1], datetime.now(UTC))

    step = timedelta(hours=freq_hours)
    base_half_life = tracked.half_life_hours * 3600.0
    adaptive = tracked.adaptive
    ordered_prices = sorted(price_ticks, key=lambda t: t.ts)
    price_times = [t.ts for t in ordered_prices]

    points: list[SignalPoint] = []
    cursor = start.replace(minute=0, second=0, microsecond=0) + step

    while cursor <= end:
        window_start = cursor - step
        # Both bounds by binary search on the sorted post times: counting with a
        # generator scanned every post once per bar, which is quadratic in a
        # window that can hold thousands of each.
        recent = bisect_right(post_times, cursor) - bisect_right(post_times, window_start)
        baseline = baseline_hourly_rate(post_times, cursor, lookback_hours=24)

        price_now = _price_at(ordered_prices, price_times, cursor)
        price_prev = _price_at(ordered_prices, price_times, window_start)
        move = 0.0 if price_now is None or price_prev is None else price_now - price_prev

        half_life = effective_half_life(
            base_half_life,
            min_half_life_seconds=adaptive.min_half_life_hours * 3600.0,
            recent_post_count=float(recent),
            baseline_post_rate=float(baseline),
            recent_price_move=move,
            volume_surge_multiple=adaptive.volume_surge_multiple,
            price_move_threshold=adaptive.price_move_threshold,
            enabled=adaptive.enabled,
        )

        point = compute_signal(
            contributions,
            cursor,
            half_life,
            market_id=tracked.market.market_id,
            model_version=model_version,
            max_age=timedelta(days=7),
        )

        if point is not None:
            p_hat = to_probability(point.s_t, calibration, platt)
            points.append(
                SignalPoint(
                    market_id=point.market_id,
                    ts=point.ts,
                    model_version=point.model_version,
                    s_t=point.s_t,
                    n_posts=point.n_posts,
                    weight_sum=point.weight_sum,
                    half_life_seconds=point.half_life_seconds,
                    p_hat=p_hat,
                    calibration=calibration,
                    computed_at=datetime.now(UTC),
                )
            )

        cursor += step

    return points


# ---------------------------------------------------------------------------
# LMSR simulation
# ---------------------------------------------------------------------------


def _book_snapshots(price_ticks: Sequence[PriceTick]) -> list[PriceTick]:
    """Book snapshots, oldest first. Only these carry depth.

    Candle rows carry bid/ask but no resting size, and `b` is calibrated from
    size. Calibrating against a candle would silently fall back to the default
    on every bar while looking like it had used real liquidity.
    """
    return sorted(
        (t for t in price_ticks if t.source is PriceSource.BOOK), key=lambda t: t.ts
    )


def simulate_lmsr(
    signals: Sequence[SignalPoint],
    price_ticks: Sequence[PriceTick],
    *,
    run_id: str = "live",
    responsiveness: float = 0.35,
    initial_price: float | None = None,
) -> list[SimTick]:
    """Drive a synthetic LMSR market from the signal series.

    `b` is re-calibrated at each step from the most recent book snapshot *at or
    before* that step, so the synthetic market's sensitivity tracks the real
    market's liquidity instead of being a constant chosen to make the output look
    good. Recalibration deliberately holds the current price fixed: a liquidity
    update is not new information about the outcome, and letting it move the
    price would inject signal nobody traded on.

    Two properties this depends on, both enforced here rather than assumed:

    * **Only book rows are eligible.** Candles carry quotes but no depth.
    * **No lookahead.** A step at time t sees only books at or before t. Using
      the *latest* book for every step — which is what an earlier version did —
      hands the whole simulation liquidity information from its own future.

    When `poll_prices` has supplied just one snapshot (the common case today,
    since history comes from candles), this degrades to a single constant `b`,
    which is the honest behaviour: there is no depth series to track.
    """
    if not signals:
        return []

    books = _book_snapshots(price_ticks)
    book_times = [b.ts for b in books]
    fallback_book = books[-1] if books else None

    def calibrate(book: PriceTick | None) -> float:
        return calibrate_b_from_depth(
            book.depth_bid if book else None,
            book.depth_ask if book else None,
            book.yes_bid if book else None,
            book.yes_ask if book else None,
        )

    ordered_signals = sorted(signals, key=lambda s: s.ts)

    # Opening liquidity: the last book at or before the first signal point, or
    # failing that the earliest book we have. A simulation that opened on the
    # *final* snapshot's depth would be using the future to set its own inertia.
    opening_book = _book_at(books, book_times, ordered_signals[0].ts) or (
        books[0] if books else None
    )
    market = LMSRMarket(b=calibrate(opening_book))

    anchor = initial_price
    if anchor is None:
        anchor = fallback_book.mid if fallback_book and fallback_book.mid is not None else 0.5
    market.move_to_price(anchor)
    # Anchoring is a setup move, not a trade the simulation should be charged for.
    market.realized_cost = 0.0

    ticks: list[SimTick] = []

    for point in ordered_signals:
        book = _book_at(books, book_times, point.ts)
        if book is not None:
            new_b = calibrate(book)
            if new_b != market.b:
                market.recalibrate(new_b)

        target = point.p_hat if point.p_hat is not None else (point.s_t + 1.0) / 2.0
        shares, _ = market.step_from_signal(target, responsiveness=responsiveness)

        ticks.append(
            SimTick(
                market_id=point.market_id,
                ts=point.ts,
                sim_price=market.price,
                b=market.b,
                q_yes=market.q_yes,
                q_no=market.q_no,
                run_id=run_id,
                cost=market.cost_value,
                trade_size=shares,
                signal=point.s_t,
            )
        )

    return ticks


def _book_at(
    books: Sequence[PriceTick], times: Sequence[datetime], at: datetime
) -> PriceTick | None:
    """Most recent book snapshot at or before `at`, by binary search.

    `books` must be sorted ascending by `ts` and `times` must be its timestamps,
    hoisted out so the search stays O(log n) instead of rebuilding the key list
    on every one of potentially thousands of bars.
    """
    if not books:
        return None
    index = bisect_right(times, at)
    return books[index - 1] if index else None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_markets(db: Database, settings: Settings) -> list[dict[str, Any]]:
    """Phase 7 watcher: settle markets whose close time has passed.

    For each newly resolved market this freezes a final Brier/BSS into `scores`
    and closes out any open paper position at the realized outcome.
    """
    outcomes: list[dict[str, Any]] = []

    for market in db.open_markets_past_close():
        try:
            client = venue_client(market.venue, settings)
            fresh = client.get_market(market.ticker, target=market.target)
        except Exception as exc:
            log.warning("resolution check failed for %s: %s", market.market_id, exc)
            continue

        db.upsert_market(fresh)

        if fresh.status is not MarketStatus.SETTLED or fresh.resolved_outcome is None:
            outcomes.append({"market_id": market.market_id, "status": fresh.status.value, "settled": False})
            continue

        result = {
            "market_id": market.market_id,
            "status": "settled",
            "settled": True,
            "outcome": fresh.resolved_outcome,
        }
        result.update(_freeze_scores(db, fresh))
        result.update(_settle_position(db, fresh))
        outcomes.append(result)

    return outcomes


def _freeze_scores(db: Database, market: Market) -> dict[str, Any]:
    from ..analytics.scoring import score_market as compute_score

    outcome = market.resolved_outcome
    assert outcome is not None

    frozen: dict[str, Any] = {"scored_models": []}

    with db.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_version FROM signals WHERE market_id = %s",
            (market.market_id,),
        )
        versions = [row["model_version"] for row in cur.fetchall()]

    for version in versions:
        signal_rows = db.signal_series(market.market_id, version)
        price_rows = db.price_series(market.market_id)
        if not signal_rows or not price_rows:
            continue

        prices_by_hour = {
            row["ts"].replace(minute=0, second=0, microsecond=0): row for row in price_rows
        }

        signal_probs: list[float] = []
        market_probs: list[float] = []
        calibration = Calibration.AFFINE

        for row in signal_rows:
            if row["p_hat"] is None:
                continue
            bucket = row["ts"].replace(minute=0, second=0, microsecond=0)
            price_row = prices_by_hour.get(bucket)
            if price_row is None:
                continue
            reference = price_row["yes_price"]
            if reference is None:
                bid, ask = price_row["yes_bid"], price_row["yes_ask"]
                reference = (bid + ask) / 2.0 if bid is not None and ask is not None else None
            if reference is None:
                continue
            signal_probs.append(float(row["p_hat"]))
            market_probs.append(float(reference))
            if row["calibration"]:
                calibration = Calibration(row["calibration"])

        if not signal_probs:
            continue

        score = compute_score(
            market.market_id, version, signal_probs, market_probs, outcome, calibration
        )
        db.record_score(
            market.market_id,
            version,
            n_obs=score.n_obs,
            calibration=calibration,
            brier_signal=score.brier_signal,
            brier_market=score.brier_market,
            bss=score.bss,
            resolved_outcome=outcome,
            window_start=signal_rows[0]["ts"],
            window_end=signal_rows[-1]["ts"],
            final=True,
        )
        frozen["scored_models"].append({"model_version": version, "bss": score.bss})

    return frozen


def _settle_position(db: Database, market: Market, strategy: str = "signal_v1") -> dict[str, Any]:
    """Close an open paper position at the realized outcome.

    A YES share pays 1 if the outcome is YES and 0 otherwise; NO is the mirror.
    """
    position = db.position(market.market_id, strategy)
    if position is None or position["settled"]:
        return {"position_settled": False}

    qty_yes = float(position["qty_yes"])
    qty_no = float(position["qty_no"])
    if qty_yes == 0.0 and qty_no == 0.0:
        return {"position_settled": False}

    outcome = market.resolved_outcome
    assert outcome is not None

    payout = qty_yes * float(outcome) + qty_no * float(1 - outcome)
    ts = market.resolution_time or datetime.now(UTC)

    if qty_yes:
        db.record_fill(
            market.market_id, ts,
            side="yes", action="settle", quantity=abs(qty_yes),
            price=float(outcome), cash_flow=qty_yes * float(outcome),
            strategy=strategy, note="market resolution",
        )
    if qty_no:
        db.record_fill(
            market.market_id, ts,
            side="no", action="settle", quantity=abs(qty_no),
            price=float(1 - outcome), cash_flow=qty_no * float(1 - outcome),
            strategy=strategy, note="market resolution",
        )

    db.apply_fill_to_position(
        market.market_id,
        strategy=strategy,
        d_qty_yes=-qty_yes,
        d_qty_no=-qty_no,
        d_cash=payout,
        d_realized=payout,
        settled=True,
    )

    return {"position_settled": True, "payout": payout}

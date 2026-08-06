"""The end-to-end vertical slice for a single market.

Runs the whole pipeline in memory — poll, ingest, score, aggregate, simulate,
analyze — and prints a report. Persistence is opt-in, so the slice works before
TimescaleDB exists, which is what makes it useful as the first thing to run on
a fresh checkout.

What it will not do is pretend. If the posts came from fixtures, the report says
so; if there is too little history for a Granger test, it says that rather than
printing a p-value nobody should believe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..analytics.leadlag import cross_correlation, granger_test
from ..analytics.scoring import ScoreResult, score_market
from ..config import MarketConfig, Settings, TrackedMarket
from ..models import Calibration, MarketStatus, Post, PriceTick, SignalPoint, SimTick, Stance
from ..signal.stance import EnsembleStanceModel, VaderStanceModel, build_stance_model
from .steps import (
    build_signal_series,
    ingest_posts,
    latest_book,
    poll_prices,
    score_posts,
    simulate_lmsr,
)

# A Granger test needs enough aligned bars to be interpretable at all; below
# this the slice reports "insufficient history" instead of a number.
MIN_GRANGER_OBS = 20

# Price range below which the series is treated as constant. Prediction-market
# prices are quoted in cents, so any *real* move is at least 0.01; anything
# smaller is arithmetic residue from deriving one side of the book as `1 - x`.
# The threshold is far above float64 epsilon and far below one tick, so it
# cannot mistake a genuine move for noise or vice versa.
DEGENERATE_PRICE_RANGE = 1e-9


@dataclass(slots=True)
class SliceResult:
    market: TrackedMarket
    prices: list[PriceTick] = field(default_factory=list)
    posts: list[Post] = field(default_factory=list)
    stances: list[Stance] = field(default_factory=list)
    signals: list[SignalPoint] = field(default_factory=list)
    sim: list[SimTick] = field(default_factory=list)
    model_version: str = ""
    live_posts: bool = False
    cross_correlation: Any = None
    granger: Any = None
    granger_note: str = ""
    score: ScoreResult | None = None
    score_note: str = ""

    @property
    def calibrated_b(self) -> float | None:
        return self.sim[-1].b if self.sim else None

    def report(self) -> str:
        market = self.market.market
        book = latest_book(self.prices)
        lines: list[str] = []

        def rule(title: str) -> None:
            lines.append("")
            lines.append(f"── {title} " + "─" * max(0, 62 - len(title)))

        lines.append("═" * 68)
        lines.append(f"AUGURY VERTICAL SLICE — {market.market_id}")
        lines.append("═" * 68)
        lines.append(f"  {market.title}")
        lines.append(f"  target: {market.target}")
        lines.append(f"  status: {market.status.value}" + (
            f"  outcome: {market.resolved_outcome}" if market.resolved_outcome is not None else ""
        ))
        if market.close_time:
            lines.append(f"  closes: {market.close_time.isoformat()}")

        rule("market data")
        candles = [p for p in self.prices if p.source.value == "candle"]
        lines.append(f"  price bars fetched   : {len(candles)}")
        if book:
            spread = f"  spread {book.spread:.4f}" if book.spread is not None else ""
            lines.append(f"  book                 : bid {book.yes_bid} / ask {book.yes_ask}{spread}")
            lines.append(f"  depth (bid/ask)      : {book.depth_bid} / {book.depth_ask}")
            if book.yes_bid is None or book.yes_ask is None:
                lines.append(
                    "  NOTE: the book is one-sided, so there is no quoted width to"
                )
                lines.append(
                    "        calibrate b against — the simulation used the fallback."
                )
        if self.calibrated_b is not None:
            lines.append(f"  LMSR b (from depth)  : {self.calibrated_b:,.2f}")

        rule("social signal")
        source = "LIVE X API" if self.live_posts else "FIXTURE REPLAY (no live X reads, no cost)"
        lines.append(f"  post source          : {source}")
        lines.append(f"  posts                : {len(self.posts)}")
        lines.append(f"  stance model         : {self.model_version}")
        if self.stances:
            mean_stance = sum(s.stance for s in self.stances) / len(self.stances)
            lines.append(f"  mean raw stance      : {mean_stance:+.4f}")
            flagged = [s for s in self.stances if s.disagreement is not None and s.disagreement > 0.5]
            if flagged:
                lines.append(f"  high-disagreement    : {len(flagged)} posts flagged")
        if self.signals:
            latest = self.signals[-1]
            lines.append(f"  S(t) points          : {len(self.signals)}")
            lines.append(
                f"  latest S(t)          : {latest.s_t:+.4f}"
                f"  (p_hat {latest.p_hat:.4f}, {latest.n_posts} posts,"
                f" half-life {latest.half_life_seconds / 3600:.2f}h)"
            )
            half_lives = {round(s.half_life_seconds / 3600, 2) for s in self.signals}
            if len(half_lives) > 1:
                lines.append(
                    f"  adaptive decay       : half-life varied {min(half_lives):.2f}h"
                    f" – {max(half_lives):.2f}h across the window"
                )

        rule("synthetic LMSR market")
        if self.sim:
            first, last = self.sim[0], self.sim[-1]
            lines.append(f"  sim ticks            : {len(self.sim)}")
            lines.append(f"  sim price            : {first.sim_price:.4f} → {last.sim_price:.4f}")
            real = book.mid if book and book.mid is not None else None
            if real is not None:
                lines.append(f"  real market mid      : {real:.4f}  (gap {last.sim_price - real:+.4f})")
        else:
            lines.append("  (no signal points, nothing to simulate)")

        rule("lead-lag")
        if self.cross_correlation is not None and not self.cross_correlation.empty:
            table = self.cross_correlation.dropna(subset=["correlation"])
            if not table.empty:
                best = table.loc[table["correlation"].abs().idxmax()]
                lines.append(
                    f"  strongest |r|        : {best['correlation']:+.4f} at lag "
                    f"{int(best['lag']):+d}h ({best['interpretation']}, n={int(best['n_obs'])})"
                )
                for _, row in table.iterrows():
                    if abs(int(row["lag"])) <= 3:
                        lines.append(
                            f"    lag {int(row['lag']):+d}h  r = {row['correlation']:+.4f}"
                            f"  ({row['interpretation']})"
                        )
        else:
            lines.append("  (not enough overlapping observations)")

        if self.granger is not None:
            g = self.granger
            lines.append(
                f"  Granger S→P          : F={g.f_statistic:.3f} p={g.p_value:.4f} "
                f"lag={g.lag_order} ({g.lag_criterion.upper()}) n={g.n_obs}"
            )
            lines.append(
                f"  ADF p-values         : signal {g.adf_p_signal:.4f}, price {g.adf_p_price:.4f}"
            )
            lines.append(
                "  NOTE: uncorrected. Significance requires a Benjamini-Hochberg"
            )
            lines.append(
                "        correction across every market tested — see augury-analytics."
            )
        elif self.granger_note:
            lines.append(f"  Granger              : {self.granger_note}")

        rule("calibration")
        if self.score is not None:
            lines.append(f"  {self.score.summary()}")
        else:
            lines.append(f"  {self.score_note or 'market unresolved — no Brier score yet'}")

        lines.append("")
        lines.append("═" * 68)
        if not self.live_posts:
            lines.append(
                "Posts were replayed from fixtures. These numbers exercise the pipeline;"
            )
            lines.append(
                "they are not evidence about the real relationship between X and this market."
            )
            lines.append("═" * 68)
        return "\n".join(lines)


def run_slice(
    tracked: TrackedMarket,
    config: MarketConfig,
    settings: Settings,
    *,
    lookback_days: int = 14,
    max_posts: int = 100,
    responsiveness: float = 0.35,
    persist: bool = False,
) -> SliceResult:
    """Run the full pipeline for one market and return everything it produced."""
    from ..clients.x import XClient

    result = SliceResult(market=tracked)

    # 1. Prices: history plus a live book snapshot for depth.
    result.prices = poll_prices(tracked, settings, lookback_days=lookback_days)

    # 2. Posts: fixtures unless live mode is explicitly enabled.
    x_client = XClient.from_settings(settings)
    result.live_posts = x_client.is_live
    result.posts = ingest_posts(
        tracked, config, x_client, max_results=max_posts, lookback_days=min(lookback_days, 7)
    )

    # 3. Stance. VADER doubles as the ensemble's second opinion when the
    #    primary model is target-conditioned.
    primary = build_stance_model(settings.stance_model)
    model = (
        EnsembleStanceModel(primary, VaderStanceModel())
        if settings.stance_model.lower() != "vader"
        else primary
    )
    result.model_version = model.model_version
    result.stances = score_posts(result.posts, model, tracked.market.target)

    # 4. S(t) on an hourly grid, with the adaptive half-life.
    result.signals = build_signal_series(
        tracked,
        result.posts,
        result.stances,
        result.prices,
        model_version=result.model_version,
        settings=settings,
        calibration=Calibration.AFFINE,
    )

    # 5. Synthetic LMSR market driven by the signal.
    result.sim = simulate_lmsr(result.signals, result.prices, responsiveness=responsiveness)

    # 6. Lead-lag analysis on the aligned series.
    _analyze(result)

    # 7. Brier / BSS, only if the market has actually resolved.
    _score(result)

    if persist:
        _persist(result, settings)

    return result


def _aligned_pairs(result: SliceResult) -> tuple[list[float], list[float]]:
    """Signal and market price on a shared hourly grid."""
    prices_by_hour: dict[datetime, float] = {}
    for tick in sorted(result.prices, key=lambda t: t.ts):
        value = tick.mid
        if value is not None:
            prices_by_hour[tick.ts.replace(minute=0, second=0, microsecond=0)] = value

    signal_values: list[float] = []
    price_values: list[float] = []
    for point in result.signals:
        bucket = point.ts.replace(minute=0, second=0, microsecond=0)
        if bucket in prices_by_hour:
            signal_values.append(point.s_t)
            price_values.append(prices_by_hour[bucket])

    return signal_values, price_values


def _analyze(result: SliceResult) -> None:
    signal_values, price_values = _aligned_pairs(result)

    if len(signal_values) < 5:
        result.granger_note = (
            f"only {len(signal_values)} aligned observations — not enough to analyze"
        )
        return

    # A correlation against a constant series is 0/0. Reporting the resulting
    # zeros as "no relationship" would be wrong — the truth is that this market
    # did not move at all in the window, so the data cannot answer the question.
    #
    # Tested on the price *range*, not on set cardinality. Distinct-value counting
    # was the bug: observed on KXFEDDECISION-26SEP-C25, a window held exactly two
    # distinct floats — 0.015 and 0.015000000000000001, six ULPs apart, because
    # the YES ask is derived as 1 - 0.98 and neither is representable in binary64.
    # Cardinality said "two values, go ahead"; the cross-correlation then ran
    # against float noise and reported a smooth gradient to -0.398 at lag +6h
    # labelled "signal leads". That is precisely the artifact this project exists
    # not to produce.
    price_range = max(price_values) - min(price_values)
    if price_range < DEGENERATE_PRICE_RANGE:
        result.granger_note = (
            f"market price moved by {price_range:.3g} across all {len(price_values)} aligned "
            f"bars (below the {DEGENERATE_PRICE_RANGE:g} floating-point noise floor) — "
            "correlation and causality are both undefined here"
        )
        return

    result.cross_correlation = cross_correlation(
        signal_values, price_values, max_lag=min(6, len(signal_values) // 3)
    )

    if len(signal_values) < MIN_GRANGER_OBS:
        result.granger_note = (
            f"{len(signal_values)} aligned observations; a Granger test needs at least "
            f"{MIN_GRANGER_OBS} to be interpretable"
        )
        return

    try:
        result.granger = granger_test(signal_values, price_values, direction="signal_to_price")
    except Exception as exc:
        result.granger_note = f"test could not be run: {exc}"


def _score(result: SliceResult) -> None:
    market = result.market.market

    if market.status is not MarketStatus.SETTLED or market.resolved_outcome is None:
        result.score_note = "market has not resolved yet — Brier/BSS deferred to the watcher"
        return

    signal_probs: list[float] = []
    market_probs: list[float] = []

    prices_by_hour: dict[datetime, float] = {}
    for tick in sorted(result.prices, key=lambda t: t.ts):
        value = tick.mid
        if value is not None:
            prices_by_hour[tick.ts.replace(minute=0, second=0, microsecond=0)] = value

    for point in result.signals:
        if point.p_hat is None:
            continue
        bucket = point.ts.replace(minute=0, second=0, microsecond=0)
        if bucket in prices_by_hour:
            signal_probs.append(point.p_hat)
            market_probs.append(prices_by_hour[bucket])

    if not signal_probs:
        result.score_note = "market resolved but no aligned signal/price pairs to score"
        return

    result.score = score_market(
        market.market_id,
        result.model_version,
        signal_probs,
        market_probs,
        market.resolved_outcome,
        Calibration.AFFINE,
    )


def _persist(result: SliceResult, settings: Settings) -> None:
    from ..bus import Bus
    from ..db import Database

    with Database(settings.database_url) as db:
        db.upsert_market(result.market.market)
        db.replace_market_queries(result.market.query)
        db.insert_price_ticks(result.prices)
        db.insert_posts(result.posts)
        db.insert_stances(result.stances)
        for point in result.signals:
            db.insert_signal(point)
        db.insert_sim_ticks(result.sim)

    bus = Bus(settings.redis_url)
    if bus.available():
        for point in result.signals[-1:]:
            bus.publish_signal(point)
        for tick in result.sim[-1:]:
            bus.publish_sim(tick)
        bus.close()


def now_utc() -> datetime:
    return datetime.now(UTC)

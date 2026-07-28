"""Dagster asset graph for the Augury pipeline.

    market_config -> market_prices -> posts -> stances -> signals -> sim_prices
                                                            \\-> calibration_fit
                                                            \\-> resolutions

Run with:

    dagster dev -f augury_signal/pipeline/orchestrator.py

Each asset is partitioned by market, so one venue outage fails a single
partition rather than the whole run, and a re-materialization only re-does the
market that failed.

Every asset writes through the idempotent upserts in `augury_signal.db`, which
is what makes re-materializing an overlapping window safe — Dagster will
happily re-run a partition, and doing so must not double-count posts in S(t).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from ..config import load_market_config, load_settings
from ..models import Calibration

log = logging.getLogger(__name__)

try:
    from dagster import (
        AssetExecutionContext,
        Definitions,
        MetadataValue,
        Output,
        ScheduleDefinition,
        StaticPartitionsDefinition,
        asset,
        define_asset_job,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Dagster is an optional extra — install with "
        "`uv pip install -e '.[orchestration]'` inside apps/augury-signal"
    ) from exc


def _market_partitions() -> StaticPartitionsDefinition:
    """One partition per tracked market, from markets.yaml."""
    config = load_market_config()
    return StaticPartitionsDefinition([t.market.market_id for t in config.markets])


MARKET_PARTITIONS = _market_partitions()


def _tracked(market_id: str):
    return load_market_config().by_id(market_id)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


@asset(
    description="Mirror config/markets.yaml into the markets and market_queries tables.",
    compute_kind="postgres",
)
def market_config(context: AssetExecutionContext) -> Output[int]:
    from ..db import Database
    from .steps import sync_config

    settings = load_settings()
    config = load_market_config()

    with Database(settings.database_url) as db:
        counts = sync_config(db, config, settings)

    context.log.info("synced %s markets", counts["markets"])
    return Output(
        counts["markets"],
        metadata={
            "markets": counts["markets"],
            "query_terms": counts["query_terms"],
        },
    )


@asset(
    partitions_def=MARKET_PARTITIONS,
    deps=[market_config],
    description="Poll venue price history and the current order book (depth feeds LMSR b).",
    compute_kind="http",
)
def market_prices(context: AssetExecutionContext) -> Output[int]:
    from ..db import Database
    from .steps import latest_book, poll_prices

    settings = load_settings()
    tracked = _tracked(context.partition_key)

    ticks = poll_prices(tracked, settings, lookback_days=7)
    with Database(settings.database_url) as db:
        db.insert_price_ticks(ticks)

    book = latest_book(ticks)
    return Output(
        len(ticks),
        metadata={
            "ticks": len(ticks),
            "yes_bid": MetadataValue.text(str(book.yes_bid if book else "n/a")),
            "yes_ask": MetadataValue.text(str(book.yes_ask if book else "n/a")),
            "depth_ask": MetadataValue.text(str(book.depth_ask if book else "n/a")),
        },
    )


@asset(
    partitions_def=MARKET_PARTITIONS,
    deps=[market_config],
    description="Ingest posts for the market (fixture replay unless AUGURY_X_LIVE=1).",
    compute_kind="http",
)
def posts(context: AssetExecutionContext) -> Output[int]:
    from ..clients.x import XClient
    from ..db import Database
    from .steps import ingest_posts

    settings = load_settings()
    config = load_market_config()
    tracked = _tracked(context.partition_key)

    client = XClient.from_settings(settings)
    fetched = ingest_posts(tracked, config, client, max_results=100)

    with Database(settings.database_url) as db:
        stored = db.insert_posts(fetched)

    if not client.is_live:
        context.log.warning(
            "posts came from fixture replay — results are not real measurements"
        )

    return Output(
        stored,
        metadata={
            "fetched": len(fetched),
            "stored_new": stored,
            "source": "x" if client.is_live else "fixture",
        },
    )


@asset(
    partitions_def=MARKET_PARTITIONS,
    deps=[posts],
    description="Score unscored posts against the market's target claim.",
    compute_kind="pytorch",
)
def stances(context: AssetExecutionContext) -> Output[int]:
    from ..db import Database
    from ..signal.stance import EnsembleStanceModel, VaderStanceModel, build_stance_model
    from .steps import score_posts

    settings = load_settings()
    tracked = _tracked(context.partition_key)

    primary = build_stance_model(settings.stance_model)
    model = (
        EnsembleStanceModel(primary, VaderStanceModel())
        if settings.stance_model.lower() != "vader"
        else primary
    )

    with Database(settings.database_url) as db:
        # Only posts this model version has not seen, so swapping in DeBERTa
        # picks up the entire backlog instead of finding everything done.
        pending = db.unscored_posts(tracked.market.market_id, model.model_version, limit=1000)
        scored = score_posts(pending, model, tracked.market.target)
        db.insert_stances(scored)

    return Output(
        len(scored),
        metadata={"scored": len(scored), "model_version": model.model_version},
    )


@asset(
    partitions_def=MARKET_PARTITIONS,
    deps=[stances, market_prices],
    description="Aggregate stances into the S(t) series with adaptive time decay.",
    compute_kind="python",
)
def signals(context: AssetExecutionContext) -> Output[int]:
    from ..bus import Bus
    from ..db import Database
    from ..models import FilterVerdict, Post, Stance
    from .steps import build_signal_series

    settings = load_settings()
    tracked = _tracked(context.partition_key)
    market_id = tracked.market.market_id

    from ..signal.stance import EnsembleStanceModel, VaderStanceModel, build_stance_model

    primary = build_stance_model(settings.stance_model)
    model = (
        EnsembleStanceModel(primary, VaderStanceModel())
        if settings.stance_model.lower() != "vader"
        else primary
    )
    version = model.model_version

    with Database(settings.database_url) as db:
        rows = db.scored_posts(market_id, version, since=datetime.now(UTC) - timedelta(days=7))
        price_rows = db.price_series(market_id, since=datetime.now(UTC) - timedelta(days=7))

        # `build_signal_series` works on Post/Stance records; rebuild the
        # minimum needed from the joined query rather than re-reading text.
        synthetic_posts = [
            Post(
                post_id=f"{market_id}:{index}",
                market_id=market_id,
                created_at=created_at,
                text="",
                followers=followers,
                engagements=engagements,
                filter_verdict=FilterVerdict.ACCEPTED,
            )
            for index, (created_at, _stance, followers, engagements) in enumerate(rows)
        ]
        synthetic_stances = [
            Stance(
                post_id=f"{market_id}:{index}",
                market_id=market_id,
                created_at=created_at,
                model_version=version,
                stance=stance,
                confidence=abs(stance),
            )
            for index, (created_at, stance, _followers, _engagements) in enumerate(rows)
        ]

        from ..models import PriceSource, PriceTick

        price_ticks = [
            PriceTick(
                market_id=market_id,
                ts=row["ts"],
                source=PriceSource(row["source"]),
                yes_price=row["yes_price"],
                yes_bid=row["yes_bid"],
                yes_ask=row["yes_ask"],
                spread=row["spread"],
                depth_bid=row["depth_bid"],
                depth_ask=row["depth_ask"],
            )
            for row in price_rows
        ]

        points = build_signal_series(
            tracked,
            synthetic_posts,
            synthetic_stances,
            price_ticks,
            model_version=version,
            settings=settings,
            calibration=Calibration.AFFINE,
        )
        for point in points:
            db.insert_signal(point)

    bus = Bus(settings.redis_url)
    if points and bus.available():
        bus.publish_signal(points[-1])
        bus.close()

    latest = points[-1].s_t if points else None
    return Output(
        len(points),
        metadata={
            "points": len(points),
            "latest_s_t": MetadataValue.text(f"{latest:+.4f}" if latest is not None else "n/a"),
            "model_version": version,
        },
    )


@asset(
    partitions_def=MARKET_PARTITIONS,
    deps=[signals],
    description="Drive the synthetic LMSR market from the signal series.",
    compute_kind="python",
)
def sim_prices(context: AssetExecutionContext) -> Output[int]:
    from ..bus import Bus
    from ..db import Database
    from ..models import PriceSource, PriceTick, SignalPoint
    from .steps import simulate_lmsr

    settings = load_settings()
    market_id = context.partition_key

    with Database(settings.database_url) as db:
        signal_rows = db.signal_series(market_id, _current_model_version(settings))
        price_rows = db.price_series(market_id)

        points = [
            SignalPoint(
                market_id=market_id,
                ts=row["ts"],
                model_version=_current_model_version(settings),
                s_t=float(row["s_t"]),
                n_posts=int(row["n_posts"]),
                weight_sum=float(row["weight_sum"]),
                half_life_seconds=float(row["half_life_seconds"]),
                p_hat=float(row["p_hat"]) if row["p_hat"] is not None else None,
                calibration=Calibration(row["calibration"]) if row["calibration"] else None,
            )
            for row in signal_rows
        ]
        ticks = [
            PriceTick(
                market_id=market_id,
                ts=row["ts"],
                source=PriceSource(row["source"]),
                yes_price=row["yes_price"],
                yes_bid=row["yes_bid"],
                yes_ask=row["yes_ask"],
                spread=row["spread"],
                depth_bid=row["depth_bid"],
                depth_ask=row["depth_ask"],
            )
            for row in price_rows
        ]

        sim = simulate_lmsr(points, ticks)
        db.insert_sim_ticks(sim)

    bus = Bus(settings.redis_url)
    if sim and bus.available():
        bus.publish_sim(sim[-1])
        bus.close()

    return Output(
        len(sim),
        metadata={
            "ticks": len(sim),
            "b": MetadataValue.float(sim[-1].b) if sim else MetadataValue.text("n/a"),
            "sim_price": MetadataValue.float(sim[-1].sim_price) if sim else MetadataValue.text("n/a"),
        },
    )


@asset(
    deps=[signals],
    description="Settle closed markets: freeze final Brier/BSS and close paper positions.",
    compute_kind="postgres",
)
def resolutions(context: AssetExecutionContext) -> Output[int]:
    from ..db import Database
    from .steps import resolve_markets

    settings = load_settings()
    with Database(settings.database_url) as db:
        outcomes = resolve_markets(db, settings)

    settled = [o for o in outcomes if o.get("settled")]
    for entry in settled:
        context.log.info("settled %s -> %s", entry["market_id"], entry["outcome"])

    return Output(len(settled), metadata={"checked": len(outcomes), "settled": len(settled)})


@asset(
    deps=[resolutions],
    description="Refit the Platt calibration once enough markets have resolved.",
    compute_kind="python",
)
def calibration_fit(context: AssetExecutionContext) -> Output[str]:
    from ..db import Database
    from ..signal.calibration import fit_platt

    settings = load_settings()
    version = _current_model_version(settings)

    with Database(settings.database_url) as db:
        pairs = db.resolved_signal_pairs(version)

    if len(pairs) < 2:
        message = f"only {len(pairs)} resolved market(s); staying on affine calibration"
        context.log.info(message)
        return Output(message, metadata={"n_resolved": len(pairs), "fitted": False})

    try:
        fit = fit_platt([s for s, _ in pairs], [o for _, o in pairs])
    except ValueError as exc:
        # Single-class history is expected early on and is not a failure.
        context.log.info("calibration not identifiable yet: %s", exc)
        return Output(str(exc), metadata={"n_resolved": len(pairs), "fitted": False})

    message = f"p_hat = sigmoid({fit.a:+.4f} + {fit.b:+.4f} * S)"
    return Output(
        message,
        metadata={
            "n_resolved": fit.n_obs,
            "fitted": True,
            "a": fit.a,
            "b": fit.b,
            "log_loss": fit.log_loss,
        },
    )


def _current_model_version(settings) -> str:
    from ..signal.stance import EnsembleStanceModel, VaderStanceModel, build_stance_model

    primary = build_stance_model(settings.stance_model)
    if settings.stance_model.lower() == "vader":
        return primary.model_version
    return EnsembleStanceModel(primary, VaderStanceModel()).model_version


# ---------------------------------------------------------------------------
# Jobs and schedules
# ---------------------------------------------------------------------------

refresh_job = define_asset_job(
    name="augury_refresh",
    selection=[market_prices, posts, stances, signals, sim_prices],
    description="Full ingest-to-simulation cycle for every tracked market.",
)

resolution_job = define_asset_job(
    name="augury_resolution",
    selection=[resolutions, calibration_fit],
    description="Settle closed markets and refit calibration.",
)

# Every 15 minutes: fast enough to catch intraday moves, slow enough that a
# live X budget of a couple of thousand reads lasts the whole day.
refresh_schedule = ScheduleDefinition(
    job=refresh_job,
    cron_schedule="*/15 * * * *",
    name="augury_refresh_schedule",
)

# Hourly is ample: a market resolves once, and nothing downstream is urgent.
resolution_schedule = ScheduleDefinition(
    job=resolution_job,
    cron_schedule="0 * * * *",
    name="augury_resolution_schedule",
)

defs = Definitions(
    assets=[
        market_config,
        market_prices,
        posts,
        stances,
        signals,
        sim_prices,
        resolutions,
        calibration_fit,
    ],
    jobs=[refresh_job, resolution_job],
    schedules=[refresh_schedule, resolution_schedule],
)

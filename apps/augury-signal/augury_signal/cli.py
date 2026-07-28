"""Command-line entry point: `augury <command>`.

Commands that touch the database say so; `slice` and `markets` deliberately do
not, so the pipeline can be exercised before TimescaleDB is running.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_market_config, load_settings
from .models import Calibration


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_markets(args: argparse.Namespace) -> int:
    """List the tracked markets and their query terms."""
    config = load_market_config()
    for tracked in config.markets:
        market = tracked.market
        print(f"\n{market.market_id}")
        print(f"  title    : {market.title}")
        print(f"  target   : {market.target}")
        print(f"  half-life: {tracked.half_life_hours}h"
              f" (adaptive floor {tracked.adaptive.min_half_life_hours}h)")
        if args.verbose:
            print(f"  query    : {tracked.query.to_x_query(config.language, config.global_exclude)}")
        else:
            print(f"  terms    : {len(tracked.query.all_terms())} positive,"
                  f" {len(tracked.query.exclude)} excluded")
    print(f"\n{len(config.markets)} tracked market(s)")
    return 0


def cmd_slice(args: argparse.Namespace) -> int:
    """Run the end-to-end vertical slice for one market."""
    from .pipeline.slice import run_slice

    settings = load_settings()
    config = load_market_config()

    try:
        tracked = config.resolve(args.market)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = run_slice(
        tracked,
        config,
        settings,
        lookback_days=args.lookback_days,
        max_posts=args.max_posts,
        responsiveness=args.responsiveness,
        persist=args.persist,
    )
    print(result.report())
    return 0


def cmd_sync_config(args: argparse.Namespace) -> int:
    """Mirror markets.yaml into the database."""
    from .db import Database
    from .pipeline.steps import sync_config

    settings = load_settings()
    config = load_market_config()

    with Database(settings.database_url) as db:
        if not db.ping():
            print(f"error: cannot reach the database at {settings.database_url}", file=sys.stderr)
            print("hint: `make up && make db-migrate`", file=sys.stderr)
            return 1
        counts = sync_config(db, config, settings)

    print(f"synced {counts['markets']} market(s), {counts['query_terms']} query term(s)")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """One full refresh cycle for every tracked market, persisted."""
    from .bus import Bus
    from .clients.x import XClient
    from .db import Database
    from .pipeline.steps import (
        build_signal_series,
        ingest_posts,
        poll_prices,
        score_posts,
        simulate_lmsr,
    )
    from .signal.stance import EnsembleStanceModel, VaderStanceModel, build_stance_model

    settings = load_settings()
    config = load_market_config()

    primary = build_stance_model(settings.stance_model)
    model = (
        EnsembleStanceModel(primary, VaderStanceModel())
        if settings.stance_model.lower() != "vader"
        else primary
    )

    x_client = XClient.from_settings(settings)
    bus = Bus(settings.redis_url)
    bus_up = bus.available()

    with Database(settings.database_url) as db:
        if not db.ping():
            print(f"error: cannot reach the database at {settings.database_url}", file=sys.stderr)
            return 1

        for tracked in config.markets:
            market_id = tracked.market.market_id

            ticks = poll_prices(tracked, settings, lookback_days=args.lookback_days)
            db.insert_price_ticks(ticks)

            posts = ingest_posts(tracked, config, x_client, max_results=args.max_posts)
            stored = db.insert_posts(posts)

            stances = score_posts(posts, model, tracked.market.target)
            db.insert_stances(stances)

            signals = build_signal_series(
                tracked, posts, stances, ticks,
                model_version=model.model_version,
                settings=settings,
                calibration=Calibration.AFFINE,
            )
            for point in signals:
                db.insert_signal(point)

            sim = simulate_lmsr(signals, ticks)
            db.insert_sim_ticks(sim)

            if bus_up:
                if signals:
                    bus.publish_signal(signals[-1])
                if sim:
                    bus.publish_sim(sim[-1])

            latest = f"{signals[-1].s_t:+.4f}" if signals else "n/a"
            print(
                f"{market_id}: {len(ticks)} ticks, {len(posts)} posts ({stored} new), "
                f"{len(signals)} signal points, S(t)={latest}"
            )

    bus.close()
    return 0


def cmd_watch_resolutions(args: argparse.Namespace) -> int:
    """Settle markets that have closed, freezing final scores and positions."""
    from .db import Database
    from .pipeline.steps import resolve_markets

    settings = load_settings()

    with Database(settings.database_url) as db:
        if not db.ping():
            print(f"error: cannot reach the database at {settings.database_url}", file=sys.stderr)
            return 1
        outcomes = resolve_markets(db, settings)

    if not outcomes:
        print("no markets past their close time")
        return 0

    for entry in outcomes:
        if entry.get("settled"):
            scored = ", ".join(
                f"{s['model_version']} BSS={s['bss']:+.4f}" for s in entry.get("scored_models", [])
            )
            print(
                f"{entry['market_id']}: SETTLED outcome={entry['outcome']}"
                + (f" | {scored}" if scored else " | no scoreable signal history")
                + (f" | position settled, payout {entry['payout']:.2f}"
                   if entry.get("position_settled") else "")
            )
        else:
            print(f"{entry['market_id']}: past close, still {entry['status']} — not settled yet")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit Platt calibration from resolved markets."""
    from .db import Database
    from .signal.calibration import fit_platt

    settings = load_settings()

    with Database(settings.database_url) as db:
        if not db.ping():
            print("error: cannot reach the database", file=sys.stderr)
            return 1
        pairs = db.resolved_signal_pairs(args.model_version)

    if len(pairs) < 2:
        print(
            f"only {len(pairs)} resolved market(s) with signal history; "
            "affine calibration remains in use until there are more"
        )
        return 0

    signals = [s for s, _ in pairs]
    outcomes = [o for _, o in pairs]

    try:
        fit = fit_platt(signals, outcomes)
    except ValueError as exc:
        print(f"cannot fit: {exc}")
        return 0

    print(f"Platt fit over {fit.n_obs} resolved markets:")
    print(f"  p_hat = sigmoid({fit.a:+.4f} + {fit.b:+.4f} * S)")
    print(f"  log loss vs. true outcomes: {fit.log_loss:.4f}")
    for s in (-1.0, -0.5, 0.0, 0.5, 1.0):
        affine = (s + 1) / 2
        print(f"    S={s:+.1f}  affine={affine:.4f}  platt={fit.apply(s):.4f}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    """Show the X API read budget."""
    from pathlib import Path

    from .clients.budget import ReadBudget
    from .config import repo_root

    settings = load_settings()
    path = Path(repo_root()) / "apps" / "augury-signal" / ".augury" / "read_budget.json"
    budget = ReadBudget(path=path, max_daily_reads=settings.max_daily_reads)

    print(budget.summary())
    print(f"  live mode : {'ON — reads cost money' if settings.x_live else 'OFF — fixture replay'}")
    print(f"  token     : {'present' if settings.x_bearer_token else 'not set'}")
    print(f"  ledger    : {path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Row counts per table."""
    from .db import Database

    settings = load_settings()
    with Database(settings.database_url) as db:
        if not db.ping():
            print("error: cannot reach the database", file=sys.stderr)
            return 1
        counts = db.stats()
        migrations = db.applied_migrations()

    print(f"migrations applied: {', '.join(migrations) or '(none)'}")
    width = max(len(name) for name in counts)
    for table, count in counts.items():
        print(f"  {table:<{width}}  {count:>10,}")
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    """Regenerate the cross-language golden vectors."""
    from .golden import main as generate

    generate()
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="augury",
        description="Augury signal service — market polling, stance scoring, S(t), LMSR.",
    )
    parser.add_argument("--log-level", default="info", help="debug, info, warning, error")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("markets", help="list tracked markets")
    p.add_argument("-v", "--verbose", action="store_true", help="show full X query strings")
    p.set_defaults(func=cmd_markets)

    p = sub.add_parser("slice", help="end-to-end run for one market (no database required)")
    p.add_argument("--market", required=True, help="market id or bare ticker")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--max-posts", type=int, default=100)
    p.add_argument(
        "--responsiveness", type=float, default=0.35,
        help="fraction of the logit gap the LMSR closes per step (0,1]",
    )
    p.add_argument("--persist", action="store_true", help="also write results to TimescaleDB")
    p.set_defaults(func=cmd_slice)

    p = sub.add_parser("sync-config", help="mirror markets.yaml into the database")
    p.set_defaults(func=cmd_sync_config)

    p = sub.add_parser("refresh", help="one full refresh cycle for every tracked market")
    p.add_argument("--lookback-days", type=int, default=7)
    p.add_argument("--max-posts", type=int, default=100)
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("watch-resolutions", help="settle closed markets and freeze final scores")
    p.set_defaults(func=cmd_watch_resolutions)

    p = sub.add_parser("calibrate", help="fit Platt calibration from resolved markets")
    p.add_argument("--model-version", default="vader-3.3.2")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("budget", help="show the X API read budget")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("stats", help="row counts per table")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("golden", help="regenerate cross-language golden vectors")
    p.set_defaults(func=cmd_golden)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

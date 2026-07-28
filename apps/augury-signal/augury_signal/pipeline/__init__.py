"""Pipeline orchestration: the steps that turn raw data into signals and scores."""

from .slice import SliceResult, run_slice
from .steps import (
    build_signal_series,
    ingest_posts,
    poll_prices,
    resolve_markets,
    score_posts,
    simulate_lmsr,
    sync_config,
)

__all__ = [
    "SliceResult",
    "build_signal_series",
    "ingest_posts",
    "poll_prices",
    "resolve_markets",
    "run_slice",
    "score_posts",
    "simulate_lmsr",
    "sync_config",
]

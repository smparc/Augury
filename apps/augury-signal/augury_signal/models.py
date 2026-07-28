"""Typed records shared across the service, mirroring the JSON Schemas in `schemas/`.

These are the Python side of the cross-language contract. When a field changes
here it changes in `schemas/*.schema.json` and in the SQL migrations too — the
`test_schema_contract.py` suite fails if they drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class Venue(StrEnum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class MarketStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    UNKNOWN = "unknown"


class FilterVerdict(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    BOT = "bot"
    LOW_QUALITY = "low_quality"
    OFF_TOPIC = "off_topic"


class PriceSource(StrEnum):
    CANDLE = "candle"
    BOOK = "book"


class Calibration(StrEnum):
    """How S(t) in [-1,1] was mapped to a probability in [0,1].

    Recorded alongside every probability because a Brier score computed under
    `AFFINE` and one computed under `PLATT` are not comparable quantities.
    """

    AFFINE = "affine"
    PLATT = "platt"


def market_id(venue: Venue | str, ticker: str) -> str:
    """Canonical id: '<venue>:<ticker>'.

    Kalshi tickers and Polymarket condition ids share no namespace, so the venue
    prefix is what keeps cross-venue joins unambiguous.
    """
    return f"{Venue(venue).value}:{ticker}"


@dataclass(frozen=True, slots=True)
class Market:
    market_id: str
    venue: Venue
    ticker: str
    title: str
    # Declarative claim the stance model is conditioned on. Not the title:
    # a question ("Will the Fed cut?") and a claim ("The Fed will cut") score
    # differently, and the model needs the claim form.
    target: str
    series_ticker: str | None = None
    open_time: datetime | None = None
    close_time: datetime | None = None
    status: MarketStatus = MarketStatus.OPEN
    resolved_outcome: int | None = None
    resolution_time: datetime | None = None
    tracked: bool = True

    def __post_init__(self) -> None:
        settled = self.status is MarketStatus.SETTLED
        if settled != (self.resolved_outcome is not None):
            raise ValueError(
                f"{self.market_id}: status={self.status} but resolved_outcome="
                f"{self.resolved_outcome}; a market is settled exactly when it has an outcome"
            )
        if self.resolved_outcome not in (None, 0, 1):
            raise ValueError(f"{self.market_id}: resolved_outcome must be 0, 1, or None")


@dataclass(frozen=True, slots=True)
class Post:
    post_id: str
    market_id: str
    created_at: datetime
    text: str
    author_id: str | None = None
    author_created_at: datetime | None = None
    followers: int = 0
    engagements: int = 0
    lang: str | None = None
    ingested_at: datetime | None = None
    minhash: list[int] | None = None
    lsh_bucket: str | None = None
    filter_verdict: FilterVerdict = FilterVerdict.ACCEPTED
    filter_reason: str | None = None
    source: str = "x"

    @property
    def is_accepted(self) -> bool:
        return self.filter_verdict is FilterVerdict.ACCEPTED


@dataclass(frozen=True, slots=True)
class Stance:
    """One model's read on one post, relative to one market's target claim."""

    post_id: str
    market_id: str
    created_at: datetime
    model_version: str
    stance: float
    confidence: float
    # |stance_model_a - stance_model_b| when two models scored the post. High
    # disagreement is a data-quality flag, not an error.
    disagreement: float | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.stance <= 1.0:
            raise ValueError(f"stance {self.stance} outside [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@dataclass(frozen=True, slots=True)
class SignalPoint:
    """One evaluation of S(t) for one market."""

    market_id: str
    ts: datetime
    model_version: str
    s_t: float
    n_posts: int
    weight_sum: float
    half_life_seconds: float
    p_hat: float | None = None
    calibration: Calibration | None = None
    computed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.s_t <= 1.0:
            raise ValueError(f"s_t {self.s_t} outside [-1, 1]")
        if (self.p_hat is None) != (self.calibration is None):
            raise ValueError("p_hat and calibration must be set together or not at all")


@dataclass(frozen=True, slots=True)
class PriceTick:
    market_id: str
    ts: datetime
    source: PriceSource
    yes_price: float | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    spread: float | None = None
    depth_bid: float | None = None
    depth_ask: float | None = None
    volume: float | None = None
    fetched_at: datetime | None = None

    @property
    def mid(self) -> float | None:
        """Midpoint of the quoted range, or the last trade price if unquoted.

        Preferred over `yes_price` for lead-lag work: a last-trade price can be
        hours stale on a thin market while the book has moved.
        """
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.yes_price


@dataclass(frozen=True, slots=True)
class SimTick:
    market_id: str
    ts: datetime
    sim_price: float
    b: float
    q_yes: float
    q_no: float
    run_id: str = "live"
    cost: float | None = None
    trade_size: float | None = None
    signal: float | None = None


@dataclass(slots=True)
class MarketQuery:
    """Enhancement 5: the terms that bind posts to one market."""

    market_id: str
    keywords: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    def all_terms(self) -> list[str]:
        """Every positive term, deduplicated, preserving first-seen order."""
        seen: dict[str, None] = {}
        for term in (*self.keywords, *self.tickers, *self.entities, *self.aliases):
            seen.setdefault(term, None)
        return list(seen)

    def to_x_query(self, lang: str = "en", extra_exclude: list[str] | None = None) -> str:
        """Build an X recent-search query string.

        Only plain keywords and phrases: the endpoint silently ignores the rich
        operators from the x.com search UI (`min_faves:`, `since:`, `filter:`),
        so quality filtering is the Rust ingester's job, not the query's.
        """
        terms = self.all_terms()
        if not terms:
            raise ValueError(f"{self.market_id}: market query has no positive terms")

        def quote(term: str) -> str:
            return f'"{term}"' if " " in term else term

        clause = " OR ".join(quote(t) for t in terms)
        parts = [f"({clause})"]
        for term in [*self.exclude, *(extra_exclude or [])]:
            parts.append(f"-{quote(term)}")
        parts.append("-is:retweet")
        if lang:
            parts.append(f"lang:{lang}")
        return " ".join(parts)


def as_json_dict(obj: Any) -> dict[str, Any]:
    """Serialize a record for Redis publication / JSON Schema validation.

    Datetimes become RFC 3339 strings and enums become their values, which is
    what the schemas in `schemas/` expect.
    """
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(obj):
        raise TypeError(f"{type(obj).__name__} is not a dataclass")

    def convert(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        if isinstance(value, StrEnum):
            return value.value
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return {k: convert(v) for k, v in asdict(obj).items()}

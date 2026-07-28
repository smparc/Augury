"""Configuration: environment variables and `config/markets.yaml`.

One rule enforced here — nothing else in the package reads `os.environ`
directly. Config arrives as an explicit argument so tests can construct a
`Settings` without touching the developer's real `.env` (and, more importantly,
without a test accidentally picking up `AUGURY_X_LIVE=1` and spending money).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .models import Market, MarketQuery, MarketStatus, Venue, market_id


def repo_root() -> Path:
    """Walk up from this file to the repository root.

    Anchored on `config/markets.yaml` rather than `.git`, so the package still
    resolves its config when installed from a source tree without git metadata.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "markets.yaml").exists():
            return parent
    raise FileNotFoundError(
        "could not locate the repository root (no config/markets.yaml in any parent of "
        f"{here}); run from a checkout or set AUGURY_ROOT"
    )


def _root() -> Path:
    override = os.environ.get("AUGURY_ROOT")
    return Path(override).resolve() if override else repo_root()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    redis_url: str
    kalshi_base_url: str
    polymarket_gamma_url: str
    polymarket_clob_url: str
    x_bearer_token: str | None
    # False means the X client replays fixtures and makes no paid calls. This is
    # the default everywhere, including tests, and has to be turned on
    # deliberately.
    x_live: bool
    max_daily_reads: int
    half_life_hours: float
    min_half_life_hours: float
    stance_model: str
    stance_model_name: str
    stance_device: str
    log_level: str

    @property
    def half_life_seconds(self) -> float:
        return self.half_life_hours * 3600.0

    @property
    def min_half_life_seconds(self) -> float:
        return self.min_half_life_hours * 3600.0


def load_settings(env_file: Path | None = None, *, override: bool = False) -> Settings:
    """Read settings from the environment, optionally seeding from a .env file."""
    path = env_file if env_file is not None else _root() / ".env"
    if path.exists():
        load_dotenv(path, override=override)

    user = os.environ.get("POSTGRES_USER", "augury")
    password = os.environ.get("POSTGRES_PASSWORD", "augury_local_dev")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    database = os.environ.get("POSTGRES_DB", "augury")
    default_dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")

    token = os.environ.get("X_BEARER_TOKEN") or None
    # The shipped template value is not a credential; treating it as one would
    # let a live run start and fail with a confusing 401 instead of a clear error.
    if token == "your_bearer_token_here":
        token = None

    return Settings(
        database_url=os.environ.get("DATABASE_URL") or default_dsn,
        redis_url=os.environ.get("REDIS_URL") or f"redis://{redis_host}:{redis_port}/0",
        kalshi_base_url=os.environ.get(
            "KALSHI_BASE_URL", "https://external-api.kalshi.com/trade-api/v2"
        ).rstrip("/"),
        polymarket_gamma_url=os.environ.get(
            "POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"
        ).rstrip("/"),
        polymarket_clob_url=os.environ.get(
            "POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
        ).rstrip("/"),
        x_bearer_token=token,
        x_live=_env_bool("AUGURY_X_LIVE", False),
        max_daily_reads=_env_int("MAX_DAILY_READS", 2000),
        half_life_hours=_env_float("SIGNAL_HALF_LIFE_HOURS", 6.0),
        min_half_life_hours=_env_float("SIGNAL_HALF_LIFE_MIN_HOURS", 0.5),
        stance_model=os.environ.get("STANCE_MODEL", "vader"),
        stance_model_name=os.environ.get("STANCE_MODEL_NAME", "microsoft/deberta-v3-base"),
        stance_device=os.environ.get("STANCE_DEVICE", "cpu"),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveDecayConfig:
    """Enhancement 3 — how aggressively the half-life shortens under load."""

    enabled: bool = True
    min_half_life_hours: float = 0.5
    volume_surge_multiple: float = 3.0
    price_move_threshold: float = 0.05


@dataclass(slots=True)
class TrackedMarket:
    """A market from `markets.yaml`, paired with its query terms."""

    market: Market
    query: MarketQuery
    half_life_hours: float
    adaptive: AdaptiveDecayConfig


@dataclass(slots=True)
class MarketConfig:
    language: str = "en"
    half_life_hours: float = 6.0
    adaptive: AdaptiveDecayConfig = field(default_factory=AdaptiveDecayConfig)
    global_exclude: list[str] = field(default_factory=list)
    markets: list[TrackedMarket] = field(default_factory=list)

    def by_id(self, mid: str) -> TrackedMarket:
        for tracked in self.markets:
            if tracked.market.market_id == mid:
                return tracked
        known = ", ".join(t.market.market_id for t in self.markets) or "(none)"
        raise KeyError(f"market {mid!r} is not in markets.yaml; tracked markets are: {known}")

    def resolve(self, needle: str) -> TrackedMarket:
        """Look up by full id or by bare ticker, so the CLI can take either."""
        try:
            return self.by_id(needle)
        except KeyError:
            matches = [t for t in self.markets if t.market.ticker == needle]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                venues = ", ".join(t.market.venue.value for t in matches)
                raise KeyError(
                    f"ticker {needle!r} is tracked on multiple venues ({venues}); "
                    "use the full '<venue>:<ticker>' id"
                ) from None
            raise


def load_market_config(path: Path | None = None) -> MarketConfig:
    """Parse `config/markets.yaml` into typed records."""
    config_path = path if path is not None else _root() / "config" / "markets.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping at the top level")

    version = raw.get("version")
    if version != 1:
        raise ValueError(f"{config_path}: unsupported config version {version!r} (expected 1)")

    defaults = raw.get("defaults") or {}
    adaptive_raw = defaults.get("adaptive_decay") or {}
    adaptive = AdaptiveDecayConfig(
        enabled=bool(adaptive_raw.get("enabled", True)),
        min_half_life_hours=float(adaptive_raw.get("min_half_life_hours", 0.5)),
        volume_surge_multiple=float(adaptive_raw.get("volume_surge_multiple", 3.0)),
        price_move_threshold=float(adaptive_raw.get("price_move_threshold", 0.05)),
    )
    default_half_life = float(defaults.get("half_life_hours", 6.0))
    global_exclude = list(defaults.get("global_exclude") or [])

    entries = raw.get("markets") or []
    if not entries:
        raise ValueError(f"{config_path}: no markets defined")

    tracked: list[TrackedMarket] = []
    seen: set[str] = set()

    for index, entry in enumerate(entries):
        where = f"{config_path} markets[{index}]"
        try:
            venue = Venue(entry["venue"])
            ticker = entry["ticker"]
            title = entry["title"]
            target = entry["target"]
        except KeyError as exc:
            raise ValueError(f"{where}: missing required field {exc.args[0]!r}") from exc
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

        mid = entry.get("id") or market_id(venue, ticker)
        expected = market_id(venue, ticker)
        if mid != expected:
            raise ValueError(f"{where}: id {mid!r} does not match venue/ticker (expected {expected!r})")
        if mid in seen:
            raise ValueError(f"{where}: duplicate market id {mid!r}")
        seen.add(mid)

        query_raw = entry.get("query") or {}
        query = MarketQuery(
            market_id=mid,
            keywords=list(query_raw.get("keywords") or []),
            tickers=list(query_raw.get("tickers") or []),
            entities=list(query_raw.get("entities") or []),
            aliases=list(query_raw.get("aliases") or []),
            exclude=list(query_raw.get("exclude") or []),
        )
        if not query.all_terms():
            raise ValueError(f"{where}: market {mid!r} has no query terms; it would match nothing")

        market_entry_adaptive = entry.get("adaptive_decay")
        entry_adaptive = adaptive
        if market_entry_adaptive:
            entry_adaptive = AdaptiveDecayConfig(
                enabled=bool(market_entry_adaptive.get("enabled", adaptive.enabled)),
                min_half_life_hours=float(
                    market_entry_adaptive.get("min_half_life_hours", adaptive.min_half_life_hours)
                ),
                volume_surge_multiple=float(
                    market_entry_adaptive.get(
                        "volume_surge_multiple", adaptive.volume_surge_multiple
                    )
                ),
                price_move_threshold=float(
                    market_entry_adaptive.get("price_move_threshold", adaptive.price_move_threshold)
                ),
            )

        tracked.append(
            TrackedMarket(
                market=Market(
                    market_id=mid,
                    venue=venue,
                    ticker=ticker,
                    title=title,
                    target=target,
                    series_ticker=entry.get("series_ticker"),
                    status=MarketStatus.OPEN,
                ),
                query=query,
                half_life_hours=float(entry.get("half_life_hours", default_half_life)),
                adaptive=entry_adaptive,
            )
        )

    return MarketConfig(
        language=str(defaults.get("language", "en")),
        half_life_hours=default_half_life,
        adaptive=adaptive,
        global_exclude=global_exclude,
        markets=tracked,
    )


@lru_cache(maxsize=1)
def cached_market_config() -> MarketConfig:
    return load_market_config()

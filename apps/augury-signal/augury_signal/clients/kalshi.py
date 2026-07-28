"""Kalshi market data client.

Public, unauthenticated, and free — the opposite of the X side, so this can poll
as often as is useful.

Field shapes below were verified against the live API rather than taken from
docs, because the response format has changed: prices now arrive as
dollar-denominated *strings* ('0.0300'), not integer cents, and size fields
carry an `_fp` suffix. The Phase 1 prototype's assumption of `price.close_dollars`
still holds; its `yes_bid`/`yes_ask` top-level names do not.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

from ..models import (
    Market,
    MarketStatus,
    PriceSource,
    PriceTick,
    Venue,
    market_id,
)
from ._http import DEFAULT_TIMEOUT, build_session, parse_money

# Kalshi's candlestick endpoint accepts only these period intervals.
VALID_PERIODS = (1, 60, 1440)

# Kalshi's own status vocabulary -> ours. 'active' is what an open market
# actually reports from the single-market endpoint, even though the list
# endpoint filters on status=open.
_STATUS_MAP = {
    "active": MarketStatus.OPEN,
    "open": MarketStatus.OPEN,
    "initialized": MarketStatus.OPEN,
    "closed": MarketStatus.CLOSED,
    "determined": MarketStatus.CLOSED,
    "settled": MarketStatus.SETTLED,
    "finalized": MarketStatus.SETTLED,
}


def _parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class KalshiClient:
    def __init__(
        self,
        base_url: str = "https://external-api.kalshi.com/trade-api/v2",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else build_session()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    # -- markets ------------------------------------------------------------

    def list_markets(
        self,
        *,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Raw market listings, for browsing when choosing what to track."""
        params: dict[str, Any] = {"status": status, "limit": min(limit, 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._get("/markets", params).get("markets", [])

    def get_market(self, ticker: str, *, target: str | None = None) -> Market:
        """Fetch one market as a typed record.

        `target` is the stance claim from `markets.yaml`. When absent the title
        is used, which is a question rather than a claim and scores worse — so
        it is a fallback, not the intended path.
        """
        raw = self._get(f"/markets/{ticker}").get("market", {})
        if not raw:
            raise ValueError(f"Kalshi returned no market for ticker {ticker!r}")

        status = _STATUS_MAP.get(str(raw.get("status", "")).lower(), MarketStatus.UNKNOWN)

        # `result` is '' until the market settles, then 'yes' or 'no'.
        result = str(raw.get("result", "")).strip().lower()
        outcome: int | None = {"yes": 1, "no": 0}.get(result)

        # The Market invariant is that settled and has-an-outcome coincide. A
        # market can report 'settled' before `result` is populated, so trust the
        # outcome and downgrade the status rather than constructing an invalid
        # record.
        if outcome is None and status is MarketStatus.SETTLED:
            status = MarketStatus.CLOSED
        elif outcome is not None and status is not MarketStatus.SETTLED:
            status = MarketStatus.SETTLED

        return Market(
            market_id=market_id(Venue.KALSHI, ticker),
            venue=Venue.KALSHI,
            ticker=ticker,
            title=raw.get("title", ticker),
            target=target or raw.get("title", ticker),
            series_ticker=(raw.get("event_ticker") or "").split("-")[0] or None,
            open_time=_parse_ts(raw.get("open_time")),
            close_time=_parse_ts(raw.get("close_time")),
            status=status,
            resolved_outcome=outcome,
            resolution_time=_parse_ts(raw.get("expected_expiration_time")) if outcome is not None else None,
        )

    # -- prices -------------------------------------------------------------

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        *,
        period_minutes: int = 60,
        lookback_days: int = 14,
        end: datetime | None = None,
    ) -> list[PriceTick]:
        """Historical bars.

        Each bar carries bid and ask alongside the trade price, so historical
        spread comes free — which matters because a last-trade price on a thin
        market can be hours stale while the quote has moved.
        """
        if period_minutes not in VALID_PERIODS:
            raise ValueError(
                f"period_minutes must be one of {VALID_PERIODS}, got {period_minutes}"
            )
        if lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {lookback_days}")

        end_dt = end or datetime.now(UTC)
        end_ts = int(end_dt.timestamp())
        start_ts = int((end_dt - timedelta(days=lookback_days)).timestamp())

        payload = self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_minutes},
        )

        mid = market_id(Venue.KALSHI, ticker)
        ticks: list[PriceTick] = []

        for candle in payload.get("candlesticks", []):
            ts = candle.get("end_period_ts")
            if ts is None:
                continue

            price_block = candle.get("price") or {}
            bid_block = candle.get("yes_bid") or {}
            ask_block = candle.get("yes_ask") or {}

            yes_price = parse_money(price_block.get("close_dollars"))
            yes_bid = parse_money(bid_block.get("close_dollars"))
            yes_ask = parse_money(ask_block.get("close_dollars"))

            spread = None
            if yes_bid is not None and yes_ask is not None and yes_ask >= yes_bid:
                spread = yes_ask - yes_bid

            ticks.append(
                PriceTick(
                    market_id=mid,
                    ts=datetime.fromtimestamp(int(ts), tz=UTC),
                    source=PriceSource.CANDLE,
                    yes_price=yes_price,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    spread=spread,
                    volume=parse_money(candle.get("volume_fp")),
                    fetched_at=datetime.now(UTC),
                )
            )

        ticks.sort(key=lambda t: t.ts)
        return ticks

    def get_orderbook(self, ticker: str, *, depth: int = 10) -> PriceTick:
        """Top-of-book snapshot with resting size on each side (enhancement 4).

        Kalshi expresses the book as resting *bids on both sides*: `yes_dollars`
        are bids to buy YES, `no_dollars` are bids to buy NO. A NO bid at 0.96
        is economically a YES ask at 0.04, so the ask side is derived as
        `1 - best_no_bid` rather than read directly.

        `depth_bid` / `depth_ask` are top-of-book sizes — the quantity resting
        at the quote — which is the "shares within the quoted range" that the
        LMSR `b` calibration consumes.
        """
        payload = self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        book = payload.get("orderbook_fp") or payload.get("orderbook") or {}

        def best(levels: Any) -> tuple[float | None, float | None]:
            """Highest-priced level and its size, from [[price, size], ...]."""
            if not isinstance(levels, list) or not levels:
                return None, None
            parsed: list[tuple[float, float]] = []
            for level in levels:
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    continue
                price = parse_money(level[0])
                size = parse_money(level[1])
                if price is not None and size is not None:
                    parsed.append((price, size))
            if not parsed:
                return None, None
            return max(parsed, key=lambda pair: pair[0])

        yes_bid, yes_bid_size = best(book.get("yes_dollars") or book.get("yes"))
        no_bid, no_bid_size = best(book.get("no_dollars") or book.get("no"))

        yes_ask = None if no_bid is None else 1.0 - no_bid
        depth_ask = no_bid_size

        spread = None
        if yes_bid is not None and yes_ask is not None and yes_ask >= yes_bid:
            spread = yes_ask - yes_bid

        mid_price = None
        if yes_bid is not None and yes_ask is not None:
            mid_price = (yes_bid + yes_ask) / 2.0

        return PriceTick(
            market_id=market_id(Venue.KALSHI, ticker),
            ts=datetime.now(UTC),
            source=PriceSource.BOOK,
            yes_price=mid_price,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            spread=spread,
            depth_bid=yes_bid_size,
            depth_ask=depth_ask,
            fetched_at=datetime.now(UTC),
        )

    def poll_orderbooks(
        self, tickers: list[str], *, pause_seconds: float = 0.25
    ) -> list[PriceTick]:
        """Snapshot several books, pausing between calls to stay polite."""
        ticks: list[PriceTick] = []
        for index, ticker in enumerate(tickers):
            if index and pause_seconds > 0:
                time.sleep(pause_seconds)
            ticks.append(self.get_orderbook(ticker))
        return ticks

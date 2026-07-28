"""Polymarket market data client.

Two public, unauthenticated APIs are used together because neither is
sufficient alone:

  Gamma (`gamma-api.polymarket.com`)  — market metadata, resolution state, and
      top-of-book quotes (`bestBid`, `bestAsk`, `spread`). One call covers a
      whole market list, so this is the cheap path for polling many markets.

  CLOB (`clob.polymarket.com`)        — the actual order book (resting size at
      every level) and price history. Keyed by *token id*, not condition id.

A Polymarket binary market has two CLOB tokens, one per outcome. Augury tracks
the YES side, which is `clobTokenIds[0]` — the token whose index matches
`outcomes[0] == "Yes"`. That ordering is verified rather than assumed, because
silently tracking the NO token would invert every price in the pipeline and
produce a beautifully wrong negative correlation.

Several Gamma fields arrive as JSON-encoded *strings* rather than arrays
(`outcomes`, `outcomePrices`, `clobTokenIds`), which is why `_json_field` exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


def _parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _json_field(raw: dict[str, Any], key: str) -> list[Any]:
    """Decode a Gamma field that is a JSON string containing an array."""
    value = raw.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


class PolymarketClient:
    def __init__(
        self,
        gamma_url: str = "https://gamma-api.polymarket.com",
        clob_url: str = "https://clob.polymarket.com",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Any = None,
    ) -> None:
        self.gamma_url = gamma_url.rstrip("/")
        self.clob_url = clob_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else build_session()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    # -- markets ------------------------------------------------------------

    def list_markets(
        self, *, closed: bool = False, limit: int = 50, order: str = "volumeNum"
    ) -> list[dict[str, Any]]:
        payload = self._get(
            f"{self.gamma_url}/markets",
            {"closed": str(closed).lower(), "limit": limit, "order": order, "ascending": "false"},
        )
        return payload if isinstance(payload, list) else []

    def get_market_raw(self, condition_id: str) -> dict[str, Any]:
        payload = self._get(f"{self.gamma_url}/markets", {"condition_ids": condition_id})
        markets = payload if isinstance(payload, list) else []
        if not markets:
            raise ValueError(f"Polymarket returned no market for condition id {condition_id!r}")
        return markets[0]

    def yes_token_id(self, raw: dict[str, Any]) -> str:
        """CLOB token id for the YES outcome.

        Matched by name against `outcomes` rather than positionally. The YES
        token is conventionally first, but tracking the wrong token would
        invert every price downstream, so the convention is checked.
        """
        outcomes = [str(o).strip().lower() for o in _json_field(raw, "outcomes")]
        token_ids = [str(t) for t in _json_field(raw, "clobTokenIds")]

        if not token_ids:
            raise ValueError(
                f"market {raw.get('conditionId')!r} exposes no clobTokenIds; it is "
                "probably not order-book enabled"
            )
        if len(outcomes) != len(token_ids):
            raise ValueError(
                f"market {raw.get('conditionId')!r}: {len(outcomes)} outcomes but "
                f"{len(token_ids)} token ids — cannot identify the YES token"
            )

        for index, name in enumerate(outcomes):
            if name == "yes":
                return token_ids[index]

        raise ValueError(
            f"market {raw.get('conditionId')!r} has outcomes {outcomes} with no 'Yes'; "
            "Augury tracks binary YES/NO markets only"
        )

    def get_market(self, condition_id: str, *, target: str | None = None) -> Market:
        raw = self.get_market_raw(condition_id)

        closed = bool(raw.get("closed"))
        outcome: int | None = None

        if closed:
            # A resolved market prints its outcome prices at the corners.
            prices = [parse_money(p) for p in _json_field(raw, "outcomePrices")]
            outcomes = [str(o).strip().lower() for o in _json_field(raw, "outcomes")]
            for name, price in zip(outcomes, prices, strict=False):
                if name == "yes" and price is not None:
                    if price >= 0.99:
                        outcome = 1
                    elif price <= 0.01:
                        outcome = 0
                    break

        # Closed but not yet settled at the corners means resolution is still
        # pending; the Market invariant requires an outcome for SETTLED.
        status = MarketStatus.OPEN
        if closed:
            status = MarketStatus.SETTLED if outcome is not None else MarketStatus.CLOSED

        question = raw.get("question") or condition_id

        return Market(
            market_id=market_id(Venue.POLYMARKET, condition_id),
            venue=Venue.POLYMARKET,
            ticker=condition_id,
            title=question,
            target=target or question,
            series_ticker=raw.get("slug"),
            open_time=_parse_ts(raw.get("startDate")),
            close_time=_parse_ts(raw.get("endDate")),
            status=status,
            resolved_outcome=outcome,
            resolution_time=_parse_ts(raw.get("updatedAt")) if outcome is not None else None,
        )

    # -- prices -------------------------------------------------------------

    def get_quote(self, condition_id: str) -> PriceTick:
        """Top-of-book quote from Gamma, without depth.

        Cheaper than the CLOB book and enough for a price series; use
        `get_orderbook` when the LMSR needs depth to calibrate against.
        """
        raw = self.get_market_raw(condition_id)

        bid = parse_money(raw.get("bestBid"))
        ask = parse_money(raw.get("bestAsk"))
        spread = parse_money(raw.get("spread"))
        if spread is None and bid is not None and ask is not None and ask >= bid:
            spread = ask - bid

        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else parse_money(
            raw.get("lastTradePrice")
        )

        return PriceTick(
            market_id=market_id(Venue.POLYMARKET, condition_id),
            ts=datetime.now(timezone.utc),
            source=PriceSource.BOOK,
            yes_price=mid,
            yes_bid=bid,
            yes_ask=ask,
            spread=spread,
            volume=parse_money(raw.get("volumeNum")),
            fetched_at=datetime.now(timezone.utc),
        )

    def get_orderbook(self, condition_id: str, *, token_id: str | None = None) -> PriceTick:
        """Full CLOB book snapshot, reduced to top-of-book price and size.

        The CLOB returns bids ascending and asks descending, so the best bid is
        the *maximum* bid price and the best ask the *minimum* ask price. Taking
        the first element of either array — the obvious reading — gives the
        worst level on each side.
        """
        if token_id is None:
            token_id = self.yes_token_id(self.get_market_raw(condition_id))

        book = self._get(f"{self.clob_url}/book", {"token_id": token_id})

        def levels(key: str) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for level in book.get(key) or []:
                price = parse_money(level.get("price") if isinstance(level, dict) else None)
                size = parse_money(level.get("size") if isinstance(level, dict) else None)
                if price is not None and size is not None:
                    out.append((price, size))
            return out

        bids = levels("bids")
        asks = levels("asks")

        best_bid = max(bids, key=lambda pair: pair[0]) if bids else (None, None)
        best_ask = min(asks, key=lambda pair: pair[0]) if asks else (None, None)

        bid_price, bid_size = best_bid
        ask_price, ask_size = best_ask

        spread = None
        if bid_price is not None and ask_price is not None and ask_price >= bid_price:
            spread = ask_price - bid_price

        mid = None
        if bid_price is not None and ask_price is not None:
            mid = (bid_price + ask_price) / 2.0
        elif book.get("last_trade_price") is not None:
            mid = parse_money(book.get("last_trade_price"))

        return PriceTick(
            market_id=market_id(Venue.POLYMARKET, condition_id),
            ts=datetime.now(timezone.utc),
            source=PriceSource.BOOK,
            yes_price=mid,
            yes_bid=bid_price,
            yes_ask=ask_price,
            spread=spread,
            depth_bid=bid_size,
            depth_ask=ask_size,
            fetched_at=datetime.now(timezone.utc),
        )

    def get_price_history(
        self,
        condition_id: str,
        *,
        token_id: str | None = None,
        interval: str = "1w",
        fidelity_minutes: int = 60,
    ) -> list[PriceTick]:
        """Historical price series for the YES token.

        `interval` is Polymarket's window shorthand ('1d', '1w', '1m', 'max');
        `fidelity_minutes` is the bar width. Only a price is returned per point
        — no bid/ask — so these ticks carry no spread, unlike Kalshi candles.
        """
        if token_id is None:
            token_id = self.yes_token_id(self.get_market_raw(condition_id))

        payload = self._get(
            f"{self.clob_url}/prices-history",
            {"market": token_id, "interval": interval, "fidelity": fidelity_minutes},
        )

        mid = market_id(Venue.POLYMARKET, condition_id)
        now = datetime.now(timezone.utc)
        ticks: list[PriceTick] = []

        for point in payload.get("history", []):
            ts = point.get("t")
            price = parse_money(point.get("p"))
            if ts is None or price is None:
                continue
            ticks.append(
                PriceTick(
                    market_id=mid,
                    ts=datetime.fromtimestamp(int(ts), tz=timezone.utc),
                    source=PriceSource.CANDLE,
                    yes_price=price,
                    fetched_at=now,
                )
            )

        ticks.sort(key=lambda t: t.ts)
        return ticks

"""Redis pub/sub bus for cross-service messages.

Channels, matching the JSON Schemas in `schemas/`:

    augury.signal.<market_id>   StanceSignal   Python  -> C++, Java
    augury.price.<market_id>    PriceTick      Python  -> C++, Java
    augury.sim.<market_id>      SimTick        C++     -> Java, analytics

Published payloads are validated against their schema before going out when
`jsonschema` is available. The C++ and Java consumers are separately compiled
and cannot be type-checked against Python, so the schema is the only thing
holding the contract together — catching a violation at the publisher is much
cheaper than debugging a deserialization failure three services downstream.

Every publish is best-effort: a Redis outage degrades the live stream but must
not stop ingestion or scoring, since TimescaleDB is the durable record and the
bus is only how services learn about changes promptly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import repo_root
from .models import PriceTick, SignalPoint, SimTick, as_json_dict

log = logging.getLogger(__name__)

SIGNAL_CHANNEL = "augury.signal"
PRICE_CHANNEL = "augury.price"
SIM_CHANNEL = "augury.sim"


def channel_for(prefix: str, market_id: str) -> str:
    return f"{prefix}.{market_id}"


@lru_cache(maxsize=8)
def _load_schema(name: str) -> dict[str, Any] | None:
    path = repo_root() / "schemas" / f"{name}.schema.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate(payload: dict[str, Any], schema_name: str) -> None:
    """Validate a payload against its JSON Schema, if jsonschema is installed.

    Raises `ValueError` with the offending field on failure.
    """
    schema = _load_schema(schema_name)
    if schema is None:
        return
    try:
        import jsonschema
    except ImportError:
        return

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        location = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise ValueError(
            f"message does not satisfy {schema_name}.schema.json at {location}: {exc.message}"
        ) from exc


def signal_payload(point: SignalPoint) -> dict[str, Any]:
    payload = as_json_dict(point)
    if payload.get("computed_at") is None:
        payload.pop("computed_at", None)
    return payload


def price_payload(tick: PriceTick) -> dict[str, Any]:
    payload = as_json_dict(tick)
    if payload.get("fetched_at") is None:
        payload.pop("fetched_at", None)
    return payload


def sim_payload(tick: SimTick) -> dict[str, Any]:
    return as_json_dict(tick)


@dataclass(slots=True)
class Bus:
    """Publisher for the three cross-service channels."""

    url: str
    strict: bool = False
    _client: Any = None

    def client(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url, decode_responses=True)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def available(self) -> bool:
        try:
            return bool(self.client().ping())
        except Exception:
            return False

    def _publish(self, channel: str, payload: dict[str, Any], schema_name: str) -> bool:
        try:
            validate(payload, schema_name)
        except ValueError:
            # A contract violation is a bug in this service, not a transient
            # fault, so it is never swallowed regardless of `strict`.
            raise

        try:
            self.client().publish(channel, json.dumps(payload))
            return True
        except Exception as exc:
            if self.strict:
                raise
            log.warning("redis publish to %s failed (continuing): %s", channel, exc)
            return False

    def publish_signal(self, point: SignalPoint) -> bool:
        return self._publish(
            channel_for(SIGNAL_CHANNEL, point.market_id), signal_payload(point), "stance_signal"
        )

    def publish_price(self, tick: PriceTick) -> bool:
        return self._publish(
            channel_for(PRICE_CHANNEL, tick.market_id), price_payload(tick), "price_tick"
        )

    def publish_sim(self, tick: SimTick) -> bool:
        return self._publish(channel_for(SIM_CHANNEL, tick.market_id), sim_payload(tick), "sim_tick")

    def subscribe(self, *channels: str) -> Any:
        """Subscribe to channels; supports glob patterns like 'augury.signal.*'."""
        pubsub = self.client().pubsub(ignore_subscribe_messages=True)
        patterns = [c for c in channels if "*" in c or "?" in c]
        exact = [c for c in channels if c not in patterns]
        if exact:
            pubsub.subscribe(*exact)
        if patterns:
            pubsub.psubscribe(*patterns)
        return pubsub


def schema_path(name: str) -> Path:
    return repo_root() / "schemas" / f"{name}.schema.json"

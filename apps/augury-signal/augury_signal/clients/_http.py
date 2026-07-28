"""Shared HTTP session with retry and backoff.

Venue APIs are public, unauthenticated, and occasionally flaky. Every retry
here is on an idempotent GET, so replaying one is always safe.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_TIMEOUT = 15.0


def build_session(
    *,
    total_retries: int = 4,
    backoff_factor: float = 0.5,
    user_agent: str = "augury/0.2 (research; https://github.com/smparc/augury)",
) -> requests.Session:
    """Session that retries transient failures with exponential backoff.

    429 is included in the retry set and `Retry-After` is honored, so a polling
    loop that gets rate-limited backs off instead of hammering.
    """
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def parse_money(value: object, default: float | None = None) -> float | None:
    """Parse a venue price/size field into a float.

    Kalshi returns these as decimal strings ('0.0300', '69298.61'), not numbers,
    and mixes in empty strings for absent values. Everything downstream expects
    floats, so the conversion happens once, here.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default

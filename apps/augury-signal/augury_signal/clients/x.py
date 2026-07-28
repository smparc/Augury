"""X (Twitter) post ingestion, with fixtures as the default backend.

Every read from X costs money. The design consequence is that `XClient`
resolves to `FixtureXBackend` unless `AUGURY_X_LIVE=1` *and* a bearer token is
present — so a test run, a CI job, or a `make slice` on a laptop can exercise
the entire pipeline end to end at zero cost, and going live is a deliberate act
rather than an accident of configuration.

Anything derived from fixture posts carries `source='fixture'` all the way
through to the database, so a result computed from replayed data can never be
mistaken for a real one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from ..models import FilterVerdict, Post
from ._http import DEFAULT_TIMEOUT, build_session
from .budget import ReadBudget, ReadBudgetExceeded

__all__ = [
    "FixtureXBackend",
    "LiveXBackend",
    "ReadBudgetExceeded",
    "XBackend",
    "XClient",
]

X_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

# X caps a single recent-search page at 100 results.
MAX_PAGE_SIZE = 100


class XBackend(Protocol):
    """Source of posts for a market query."""

    @property
    def source_name(self) -> str: ...

    def search(
        self,
        query: str,
        market_id: str,
        *,
        max_results: int = MAX_PAGE_SIZE,
        start_time: datetime | None = None,
    ) -> list[Post]: ...


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class FixtureXBackend:
    """Replays posts from JSONL files on disk. The default backend.

    Each line is one object matching `schemas/post.schema.json`. Files are
    matched by market id, so `fixtures/kalshi_KXFEDDECISION-26JUL-C25.jsonl`
    serves that market; `fixtures/default.jsonl` serves anything unmatched.

    `anchor_to_now` shifts replayed timestamps so the newest fixture post lands
    at the current time, preserving all relative spacing. Without it, a fixture
    recorded last month decays to nothing the moment S(t) is evaluated, and
    every downstream test sees an empty signal.
    """

    def __init__(
        self,
        fixtures_dir: Path,
        *,
        anchor_to_now: bool = True,
    ) -> None:
        self.fixtures_dir = Path(fixtures_dir)
        self.anchor_to_now = anchor_to_now

    @property
    def source_name(self) -> str:
        return "fixture"

    def _fixture_path(self, market_id: str) -> Path | None:
        safe = market_id.replace(":", "_").replace("/", "_")
        candidate = self.fixtures_dir / f"{safe}.jsonl"
        if candidate.exists():
            return candidate
        default = self.fixtures_dir / "default.jsonl"
        return default if default.exists() else None

    def _read_records(self, path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text or text.startswith("//"):
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON — {exc}") from exc
                if isinstance(record, dict):
                    yield record

    def search(
        self,
        query: str,  # noqa: ARG002 — fixtures are pre-scoped to a market
        market_id: str,
        *,
        max_results: int = MAX_PAGE_SIZE,
        start_time: datetime | None = None,
    ) -> list[Post]:
        path = self._fixture_path(market_id)
        if path is None:
            return []

        records = list(self._read_records(path))
        if not records:
            return []

        parsed: list[tuple[datetime, dict[str, Any]]] = []
        for record in records:
            created = _parse_ts(record.get("created_at"))
            if created is not None:
                parsed.append((created, record))

        if not parsed:
            return []

        offset = timedelta(0)
        if self.anchor_to_now:
            newest = max(created for created, _ in parsed)
            offset = datetime.now(timezone.utc) - newest

        posts: list[Post] = []
        for created, record in parsed:
            shifted = created + offset
            if start_time is not None and shifted < start_time:
                continue
            posts.append(
                Post(
                    post_id=str(record.get("post_id") or record.get("id") or ""),
                    market_id=market_id,
                    created_at=shifted,
                    text=str(record.get("text", "")),
                    author_id=record.get("author_id"),
                    author_created_at=_parse_ts(record.get("author_created_at")),
                    followers=int(record.get("followers") or 0),
                    engagements=int(record.get("engagements") or 0),
                    lang=record.get("lang"),
                    ingested_at=datetime.now(timezone.utc),
                    filter_verdict=FilterVerdict(record.get("filter_verdict", "accepted")),
                    filter_reason=record.get("filter_reason"),
                    source="fixture",
                )
            )

        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts[:max_results]


class LiveXBackend:
    """Real X API recent-search, guarded by the persisted daily read budget."""

    def __init__(
        self,
        bearer_token: str,
        budget: ReadBudget,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        session: Any = None,
    ) -> None:
        if not bearer_token:
            raise ValueError("LiveXBackend requires a bearer token")
        self.budget = budget
        self.timeout = timeout
        self.session = session if session is not None else build_session()
        self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})

    @property
    def source_name(self) -> str:
        return "x"

    def search(
        self,
        query: str,
        market_id: str,
        *,
        max_results: int = MAX_PAGE_SIZE,
        start_time: datetime | None = None,
    ) -> list[Post]:
        page_size = max(10, min(max_results, MAX_PAGE_SIZE))

        # Checked against the page size before the call, since that is the most
        # the request could return. Reconciled to the actual count afterwards.
        self.budget.check(page_size)

        params: dict[str, Any] = {
            "query": query,
            "max_results": page_size,
            "tweet.fields": "created_at,public_metrics,lang,author_id",
            "expansions": "author_id",
            "user.fields": "public_metrics,created_at",
        }
        if start_time is not None:
            params["start_time"] = start_time.astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        response = self.session.get(X_SEARCH_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        raw_posts = payload.get("data", []) or []
        self.budget.record(len(raw_posts))

        users = {
            user["id"]: user
            for user in (payload.get("includes", {}) or {}).get("users", []) or []
            if isinstance(user, dict) and "id" in user
        }

        posts: list[Post] = []
        now = datetime.now(timezone.utc)

        for raw in raw_posts:
            created = _parse_ts(raw.get("created_at"))
            if created is None:
                continue

            metrics = raw.get("public_metrics") or {}
            engagements = sum(
                int(metrics.get(key) or 0)
                for key in ("like_count", "retweet_count", "reply_count", "quote_count")
            )

            author_id = raw.get("author_id")
            author = users.get(author_id, {})
            followers = int((author.get("public_metrics") or {}).get("followers_count") or 0)

            posts.append(
                Post(
                    post_id=str(raw.get("id")),
                    market_id=market_id,
                    created_at=created,
                    text=str(raw.get("text", "")),
                    author_id=author_id,
                    author_created_at=_parse_ts(author.get("created_at")),
                    followers=followers,
                    engagements=engagements,
                    lang=raw.get("lang"),
                    ingested_at=now,
                    source="x",
                )
            )

        return posts


class XClient:
    """Chooses a backend and exposes one search interface.

    Live mode requires both `AUGURY_X_LIVE=1` and a real bearer token. Missing
    either one falls back to fixtures with a warning rather than raising: a
    misconfigured token should degrade the run to free replay, not crash it.
    """

    def __init__(self, backend: XBackend) -> None:
        self.backend = backend

    @property
    def is_live(self) -> bool:
        return self.backend.source_name == "x"

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        fixtures_dir: Path | None = None,
        budget_path: Path | None = None,
    ) -> XClient:
        fixtures = fixtures_dir or Path(__file__).resolve().parents[2] / "fixtures"

        if not settings.x_live:
            return cls(FixtureXBackend(fixtures))

        if not settings.x_bearer_token:
            import warnings

            warnings.warn(
                "AUGURY_X_LIVE=1 but X_BEARER_TOKEN is unset (or still the template "
                "placeholder) — falling back to fixture replay, so no live posts will "
                "be ingested",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls(FixtureXBackend(fixtures))

        path = budget_path or fixtures.parent / ".augury" / "read_budget.json"
        budget = ReadBudget(path=path, max_daily_reads=settings.max_daily_reads)
        return cls(LiveXBackend(settings.x_bearer_token, budget))

    def search(
        self,
        query: str,
        market_id: str,
        *,
        max_results: int = MAX_PAGE_SIZE,
        start_time: datetime | None = None,
    ) -> list[Post]:
        return self.backend.search(
            query, market_id, max_results=max_results, start_time=start_time
        )

    def search_markets(
        self,
        queries: Sequence[tuple[str, str]],
        *,
        max_results: int = MAX_PAGE_SIZE,
        start_time: datetime | None = None,
    ) -> dict[str, list[Post]]:
        """Search several (market_id, query) pairs.

        Stops at the first budget exhaustion rather than continuing to attempt
        the rest, so partial results are returned instead of a raised error —
        a half-filled window is still usable, an exception loses everything.
        """
        results: dict[str, list[Post]] = {}
        for market_id, query in queries:
            try:
                results[market_id] = self.search(
                    query, market_id, max_results=max_results, start_time=start_time
                )
            except ReadBudgetExceeded:
                break
        return results

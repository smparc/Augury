"""Persisted daily read budget for the X API.

X is pay-per-use (~$0.005 per post read as of 2026), so the budget is the only
thing standing between a polling loop and a real bill. Two properties matter:

  * It persists. An in-memory counter resets every time the process restarts,
    which turns a crash loop into unbounded spending. The Phase 1 prototype's
    session-local `_reads_used_this_session` had exactly this hole.
  * It is checked *before* the call, against the maximum the call could return,
    not after. Reconciling to the actual count afterwards keeps the estimate
    from drifting pessimistic.

The Rust ingester enforces the same budget against the `read_budget` table in
TimescaleDB. This file-backed tracker is the Python side's equivalent for when
the database is not up; both are keyed by (UTC day, service name), so they
count separately. That is a deliberate simplification, not an oversight: each
service stays individually bounded, and running both live at once means the
real ceiling is the sum. Point them at one store before that matters.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


class ReadBudgetExceeded(RuntimeError):
    """Raised instead of making a call that would exceed the daily budget."""


@dataclass(slots=True)
class ReadBudget:
    """Tracks reads used per UTC day, persisted to a JSON file."""

    path: Path
    max_daily_reads: int
    service: str = "augury-signal"
    cost_per_read: float = 0.005

    def __post_init__(self) -> None:
        if self.max_daily_reads < 0:
            raise ValueError(f"max_daily_reads must be non-negative, got {self.max_daily_reads}")

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _load(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt ledger must not read as "zero reads used" — that would
            # silently restore the full budget. Treat it as fully spent and make
            # the operator look.
            raise RuntimeError(
                f"read budget file {self.path} is unreadable; refusing to assume the "
                "budget is unspent — inspect or delete it deliberately"
            ) from None
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-and-rename so an interrupted write cannot truncate the ledger.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _key(self, day: str | None = None) -> str:
        return f"{day or self._today()}:{self.service}"

    def used_today(self) -> int:
        return int(self._load().get(self._key(), 0))

    def remaining(self) -> int:
        return max(0, self.max_daily_reads - self.used_today())

    def check(self, requested: int) -> None:
        """Raise if `requested` more reads would exceed today's budget."""
        if requested < 0:
            raise ValueError(f"requested must be non-negative, got {requested}")
        used = self.used_today()
        if used + requested > self.max_daily_reads:
            raise ReadBudgetExceeded(
                f"{self.service}: {used} of {self.max_daily_reads} reads used today; "
                f"{requested} more would exceed the budget "
                f"(~${requested * self.cost_per_read:.2f}). Raise MAX_DAILY_READS or wait "
                f"for the UTC day to roll over."
            )

    def record(self, actual: int) -> int:
        """Record reads actually consumed. Returns the new running total."""
        if actual < 0:
            raise ValueError(f"actual must be non-negative, got {actual}")
        data = self._load()
        key = self._key()
        data[key] = int(data.get(key, 0)) + actual
        # Keep the ledger from growing without bound; 30 days is plenty of
        # history to answer "what did last week cost".
        if len(data) > 90:
            for stale in sorted(data)[:-90]:
                data.pop(stale, None)
        self._save(data)
        return data[key]

    def estimated_cost(self, reads: int | None = None) -> float:
        return (self.used_today() if reads is None else reads) * self.cost_per_read

    def summary(self) -> str:
        used = self.used_today()
        return (
            f"{self.service} {date.today().isoformat()}: {used}/{self.max_daily_reads} reads "
            f"(~${used * self.cost_per_read:.2f}), {self.remaining()} remaining"
        )

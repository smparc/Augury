"""TimescaleDB persistence.

Every write is idempotent — `ON CONFLICT DO NOTHING` or `DO UPDATE` — because
the pipeline is expected to be re-run over overlapping windows. Polling the
same market twice, or re-scoring a corpus, must not create duplicate rows or
double-count anything in S(t).

All timestamps cross the boundary as timezone-aware UTC. `psycopg` maps those
to `TIMESTAMPTZ` directly; naive datetimes would be silently interpreted in the
server's timezone, which is exactly the kind of hour-scale error that a
lead-lag study cannot survive.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .models import (
    Calibration,
    Market,
    MarketQuery,
    MarketStatus,
    Post,
    PriceTick,
    SignalPoint,
    SimTick,
    Stance,
    Venue,
)


class Database:
    """Thin wrapper over a psycopg connection.

    Deliberately not an ORM. The schema is shared with four other languages, so
    the SQL being visible at the call site is a feature: when a column changes,
    the places that must change are greppable.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def cursor(self):
        conn = self.connect()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def ping(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None
        except psycopg.Error:
            return False

    def applied_migrations(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row["version"] for row in cur.fetchall()]

    # -- markets ------------------------------------------------------------

    def upsert_market(self, market: Market) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO markets (market_id, venue, ticker, series_ticker, title, target,
                                     open_time, close_time, status, resolved_outcome,
                                     resolution_time, tracked)
                VALUES (%(market_id)s, %(venue)s, %(ticker)s, %(series_ticker)s, %(title)s,
                        %(target)s, %(open_time)s, %(close_time)s, %(status)s,
                        %(resolved_outcome)s, %(resolution_time)s, %(tracked)s)
                ON CONFLICT (market_id) DO UPDATE SET
                    title            = EXCLUDED.title,
                    target           = EXCLUDED.target,
                    close_time       = EXCLUDED.close_time,
                    status           = EXCLUDED.status,
                    resolved_outcome = EXCLUDED.resolved_outcome,
                    resolution_time  = EXCLUDED.resolution_time,
                    tracked          = EXCLUDED.tracked,
                    updated_at       = now()
                """,
                {
                    "market_id": market.market_id,
                    "venue": market.venue.value,
                    "ticker": market.ticker,
                    "series_ticker": market.series_ticker,
                    "title": market.title,
                    "target": market.target,
                    "open_time": market.open_time,
                    "close_time": market.close_time,
                    "status": market.status.value,
                    "resolved_outcome": market.resolved_outcome,
                    "resolution_time": market.resolution_time,
                    "tracked": market.tracked,
                },
            )

    def replace_market_queries(self, query: MarketQuery) -> int:
        """Rewrite a market's query terms to match the YAML config.

        Delete-then-insert rather than upsert: a term removed from the config
        must disappear from the database, or the ingester keeps paying for
        searches on keywords that were deliberately dropped.
        """
        rows = [
            (query.market_id, term_type, term)
            for term_type, terms in (
                ("keyword", query.keywords),
                ("ticker", query.tickers),
                ("entity", query.entities),
                ("alias", query.aliases),
                ("exclude", query.exclude),
            )
            for term in terms
        ]
        with self.cursor() as cur:
            cur.execute("DELETE FROM market_queries WHERE market_id = %s", (query.market_id,))
            if rows:
                cur.executemany(
                    "INSERT INTO market_queries (market_id, term_type, term) VALUES (%s, %s, %s)"
                    " ON CONFLICT DO NOTHING",
                    rows,
                )
        return len(rows)

    def get_market(self, market_id: str) -> Market | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM markets WHERE market_id = %s", (market_id,))
            row = cur.fetchone()
        return None if row is None else self._market_from_row(row)

    def tracked_markets(self) -> list[Market]:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM markets WHERE tracked ORDER BY market_id")
            return [self._market_from_row(row) for row in cur.fetchall()]

    def open_markets_past_close(self, now: datetime | None = None) -> list[Market]:
        """Markets whose close time has passed but which are not settled yet.

        This is the resolution watcher's work queue.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM markets
                WHERE tracked
                  AND status <> 'settled'
                  AND close_time IS NOT NULL
                  AND close_time <= %s
                ORDER BY close_time
                """,
                (now or datetime.now(UTC),),
            )
            return [self._market_from_row(row) for row in cur.fetchall()]

    @staticmethod
    def _market_from_row(row: dict[str, Any]) -> Market:
        return Market(
            market_id=row["market_id"],
            venue=Venue(row["venue"]),
            ticker=row["ticker"],
            title=row["title"],
            target=row["target"],
            series_ticker=row["series_ticker"],
            open_time=row["open_time"],
            close_time=row["close_time"],
            status=MarketStatus(row["status"]),
            resolved_outcome=row["resolved_outcome"],
            resolution_time=row["resolution_time"],
            tracked=row["tracked"],
        )

    # -- posts and stances --------------------------------------------------

    def insert_posts(self, posts: Sequence[Post]) -> int:
        """Insert posts, skipping any already stored.

        Returns rows actually inserted, which is what the ingest log should
        report — "fetched 100, stored 12" is the number that tells you the
        polling interval is too tight.
        """
        if not posts:
            return 0
        rows = [
            (
                p.post_id,
                p.market_id,
                p.created_at,
                p.ingested_at or datetime.now(UTC),
                p.author_id,
                p.author_created_at,
                p.followers,
                p.engagements,
                p.lang,
                p.text,
                p.minhash,
                p.lsh_bucket,
                p.filter_verdict.value,
                p.filter_reason,
                p.source,
            )
            for p in posts
        ]
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO posts (post_id, market_id, created_at, ingested_at, author_id,
                                   author_created_at, followers, engagements, lang, text,
                                   minhash, lsh_bucket, filter_verdict, filter_reason, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_id, market_id, created_at) DO NOTHING
                """,
                rows,
            )
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def insert_stances(self, stances: Sequence[Stance]) -> int:
        if not stances:
            return 0
        rows = [
            (s.post_id, s.market_id, s.created_at, s.model_version, s.stance, s.confidence, s.disagreement)
            for s in stances
        ]
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO stances (post_id, market_id, created_at, model_version,
                                     stance, confidence, disagreement)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (post_id, market_id, model_version, created_at) DO UPDATE SET
                    stance       = EXCLUDED.stance,
                    confidence   = EXCLUDED.confidence,
                    disagreement = EXCLUDED.disagreement,
                    scored_at    = now()
                """,
                rows,
            )
            return len(rows)

    def unscored_posts(
        self, market_id: str, model_version: str, *, limit: int = 500
    ) -> list[Post]:
        """Accepted posts with no stance yet under this model version.

        The `model_version` predicate is what makes re-scoring with a new model
        pick up the whole backlog instead of finding everything already done.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT p.* FROM posts p
                LEFT JOIN stances s
                       ON s.post_id = p.post_id
                      AND s.market_id = p.market_id
                      AND s.created_at = p.created_at
                      AND s.model_version = %s
                WHERE p.market_id = %s
                  AND p.filter_verdict = 'accepted'
                  AND s.post_id IS NULL
                ORDER BY p.created_at DESC
                LIMIT %s
                """,
                (model_version, market_id, limit),
            )
            return [self._post_from_row(row) for row in cur.fetchall()]

    @staticmethod
    def _post_from_row(row: dict[str, Any]) -> Post:
        from .models import FilterVerdict

        return Post(
            post_id=row["post_id"],
            market_id=row["market_id"],
            created_at=row["created_at"],
            text=row["text"],
            author_id=row["author_id"],
            author_created_at=row["author_created_at"],
            followers=row["followers"],
            engagements=row["engagements"],
            lang=row["lang"],
            ingested_at=row["ingested_at"],
            minhash=row["minhash"],
            lsh_bucket=row["lsh_bucket"],
            filter_verdict=FilterVerdict(row["filter_verdict"]),
            filter_reason=row["filter_reason"],
            source=row["source"],
        )

    def scored_posts(
        self,
        market_id: str,
        model_version: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[tuple[datetime, float, int, int]]:
        """(created_at, stance, followers, engagements) for accepted, scored posts.

        Exactly the four columns S(t) needs, so the aggregation does not pull
        post text it will never look at.
        """
        clauses = ["p.market_id = %s", "s.model_version = %s", "p.filter_verdict = 'accepted'"]
        params: list[Any] = [market_id, model_version]
        if since is not None:
            clauses.append("p.created_at >= %s")
            params.append(since)
        if until is not None:
            clauses.append("p.created_at <= %s")
            params.append(until)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT p.created_at, s.stance, p.followers, p.engagements
                FROM posts p
                JOIN stances s
                  ON s.post_id = p.post_id
                 AND s.market_id = p.market_id
                 AND s.created_at = p.created_at
                WHERE {" AND ".join(clauses)}
                ORDER BY p.created_at
                """,
                params,
            )
            return [
                (row["created_at"], float(row["stance"]), int(row["followers"]), int(row["engagements"]))
                for row in cur.fetchall()
            ]

    def post_times(self, market_id: str, since: datetime) -> list[datetime]:
        """Timestamps of accepted posts, for the adaptive-decay volume baseline."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT created_at FROM posts
                WHERE market_id = %s AND created_at >= %s AND filter_verdict = 'accepted'
                ORDER BY created_at
                """,
                (market_id, since),
            )
            return [row["created_at"] for row in cur.fetchall()]

    # -- prices and signals -------------------------------------------------

    def insert_price_ticks(self, ticks: Sequence[PriceTick]) -> int:
        if not ticks:
            return 0
        rows = [
            (
                t.market_id,
                t.ts,
                t.source.value,
                t.yes_price,
                t.yes_bid,
                t.yes_ask,
                t.spread,
                t.depth_bid,
                t.depth_ask,
                t.volume,
                t.fetched_at or datetime.now(UTC),
            )
            for t in ticks
        ]
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO market_prices (market_id, ts, source, yes_price, yes_bid, yes_ask,
                                           spread, depth_bid, depth_ask, volume, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_id, ts, source) DO UPDATE SET
                    yes_price = EXCLUDED.yes_price,
                    yes_bid   = EXCLUDED.yes_bid,
                    yes_ask   = EXCLUDED.yes_ask,
                    spread    = EXCLUDED.spread,
                    depth_bid = EXCLUDED.depth_bid,
                    depth_ask = EXCLUDED.depth_ask,
                    volume    = EXCLUDED.volume
                """,
                rows,
            )
        return len(rows)

    def insert_signal(self, point: SignalPoint) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO signals (market_id, ts, model_version, s_t, p_hat, calibration,
                                     n_posts, weight_sum, half_life_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_id, ts, model_version) DO UPDATE SET
                    s_t               = EXCLUDED.s_t,
                    p_hat             = EXCLUDED.p_hat,
                    calibration       = EXCLUDED.calibration,
                    n_posts           = EXCLUDED.n_posts,
                    weight_sum        = EXCLUDED.weight_sum,
                    half_life_seconds = EXCLUDED.half_life_seconds,
                    computed_at       = now()
                """,
                (
                    point.market_id,
                    point.ts,
                    point.model_version,
                    point.s_t,
                    point.p_hat,
                    point.calibration.value if point.calibration else None,
                    point.n_posts,
                    point.weight_sum,
                    point.half_life_seconds,
                ),
            )

    def insert_sim_ticks(self, ticks: Sequence[SimTick]) -> int:
        if not ticks:
            return 0
        rows = [
            (t.market_id, t.ts, t.run_id, t.sim_price, t.b, t.q_yes, t.q_no, t.cost) for t in ticks
        ]
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO sim_prices (market_id, ts, run_id, sim_price, b, q_yes, q_no, cost)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_id, ts, run_id) DO UPDATE SET
                    sim_price = EXCLUDED.sim_price,
                    b         = EXCLUDED.b,
                    q_yes     = EXCLUDED.q_yes,
                    q_no      = EXCLUDED.q_no,
                    cost      = EXCLUDED.cost
                """,
                rows,
            )
        return len(rows)

    def price_series(
        self, market_id: str, *, since: datetime | None = None, source: str | None = None
    ) -> list[dict[str, Any]]:
        clauses = ["market_id = %s"]
        params: list[Any] = [market_id]
        if since is not None:
            clauses.append("ts >= %s")
            params.append(since)
        if source is not None:
            clauses.append("source = %s")
            params.append(source)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT ts, source, yes_price, yes_bid, yes_ask, spread, depth_bid, depth_ask, volume
                FROM market_prices
                WHERE {" AND ".join(clauses)}
                ORDER BY ts
                """,
                params,
            )
            return list(cur.fetchall())

    def signal_series(
        self, market_id: str, model_version: str, *, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        clauses = ["market_id = %s", "model_version = %s"]
        params: list[Any] = [market_id, model_version]
        if since is not None:
            clauses.append("ts >= %s")
            params.append(since)

        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT ts, s_t, p_hat, calibration, n_posts, weight_sum, half_life_seconds
                FROM signals
                WHERE {" AND ".join(clauses)}
                ORDER BY ts
                """,
                params,
            )
            return list(cur.fetchall())

    def latest_signal(self, market_id: str, model_version: str) -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT ts, s_t, p_hat, calibration, n_posts, weight_sum, half_life_seconds
                FROM signals
                WHERE market_id = %s AND model_version = %s
                ORDER BY ts DESC LIMIT 1
                """,
                (market_id, model_version),
            )
            return cur.fetchone()

    # -- scores and results -------------------------------------------------

    def record_score(
        self,
        market_id: str,
        model_version: str,
        *,
        n_obs: int,
        calibration: Calibration,
        brier_signal: float,
        brier_market: float,
        bss: float,
        resolved_outcome: int | None,
        window_start: datetime | None = None,
        window_end: datetime | None = None,
        final: bool = False,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scores (market_id, model_version, window_start, window_end, n_obs,
                                    calibration, brier_signal, brier_market, bss,
                                    resolved_outcome, final)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    market_id,
                    model_version,
                    window_start,
                    window_end,
                    n_obs,
                    calibration.value,
                    brier_signal,
                    brier_market,
                    bss,
                    resolved_outcome,
                    final,
                ),
            )

    def resolved_signal_pairs(self, model_version: str) -> list[tuple[float, int]]:
        """(s_t, outcome) across every settled market, for fitting Platt calibration.

        One observation per market — the last signal before close — rather than
        every point. Using all points would weight a market with months of
        history hundreds of times more than one that resolved in a week, and
        would feed the fit thousands of near-identical correlated rows.
        """
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (m.market_id) s.s_t, m.resolved_outcome
                FROM markets m
                JOIN signals s ON s.market_id = m.market_id
                WHERE m.status = 'settled'
                  AND m.resolved_outcome IS NOT NULL
                  AND s.model_version = %s
                  AND (m.close_time IS NULL OR s.ts <= m.close_time)
                ORDER BY m.market_id, s.ts DESC
                """,
                (model_version,),
            )
            return [(float(row["s_t"]), int(row["resolved_outcome"])) for row in cur.fetchall()]

    def record_granger_results(self, run_id: str, results: Iterable[dict[str, Any]]) -> int:
        rows = [
            (
                run_id,
                r["market_id"],
                r["model_version"],
                r["direction"],
                r["lag_order"],
                r.get("lag_criterion", "aic"),
                r["n_obs"],
                r["f_statistic"],
                r["p_value"],
                r["p_value_adj"],
                r["significant"],
                r.get("adf_p_signal"),
                r.get("adf_p_price"),
                r.get("liquidity_control", False),
            )
            for r in results
        ]
        if not rows:
            return 0
        with self.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO granger_results (run_id, market_id, model_version, direction,
                                             lag_order, lag_criterion, n_obs, f_statistic,
                                             p_value, p_value_adj, significant,
                                             adf_p_signal, adf_p_price, liquidity_control)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, market_id, model_version, direction) DO UPDATE SET
                    p_value     = EXCLUDED.p_value,
                    p_value_adj = EXCLUDED.p_value_adj,
                    significant = EXCLUDED.significant
                """,
                rows,
            )
        return len(rows)

    # -- ledger -------------------------------------------------------------

    def record_fill(
        self,
        market_id: str,
        ts: datetime,
        *,
        side: str,
        action: str,
        quantity: float,
        price: float,
        cash_flow: float,
        strategy: str = "signal_v1",
        signal: float | None = None,
        note: str | None = None,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ledger_fills (market_id, ts, strategy, side, action, quantity,
                                          price, cash_flow, signal, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (market_id, ts, strategy, side, action, quantity, price, cash_flow, signal, note),
            )

    def apply_fill_to_position(
        self,
        market_id: str,
        *,
        strategy: str,
        d_qty_yes: float = 0.0,
        d_qty_no: float = 0.0,
        d_cash: float = 0.0,
        d_realized: float = 0.0,
        settled: bool = False,
    ) -> None:
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO positions (market_id, strategy, qty_yes, qty_no, cash,
                                       realized_pnl, settled)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (market_id, strategy) DO UPDATE SET
                    qty_yes      = positions.qty_yes + EXCLUDED.qty_yes,
                    qty_no       = positions.qty_no + EXCLUDED.qty_no,
                    cash         = positions.cash + EXCLUDED.cash,
                    realized_pnl = positions.realized_pnl + EXCLUDED.realized_pnl,
                    settled      = positions.settled OR EXCLUDED.settled,
                    updated_at   = now()
                """,
                (market_id, strategy, d_qty_yes, d_qty_no, d_cash, d_realized, settled),
            )

    def open_positions(self, strategy: str = "signal_v1") -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM positions
                WHERE strategy = %s AND NOT settled
                  AND (qty_yes <> 0 OR qty_no <> 0)
                ORDER BY market_id
                """,
                (strategy,),
            )
            return list(cur.fetchall())

    def position(self, market_id: str, strategy: str = "signal_v1") -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM positions WHERE market_id = %s AND strategy = %s",
                (market_id, strategy),
            )
            return cur.fetchone()

    # -- read budget --------------------------------------------------------

    def budget_used_today(self, service: str) -> int:
        with self.cursor() as cur:
            cur.execute(
                "SELECT reads_used FROM read_budget WHERE day = CURRENT_DATE AND service = %s",
                (service,),
            )
            row = cur.fetchone()
            return int(row["reads_used"]) if row else 0

    def budget_record(self, service: str, reads: int) -> int:
        """Add to today's counter and return the new total, atomically."""
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO read_budget (day, service, reads_used)
                VALUES (CURRENT_DATE, %s, %s)
                ON CONFLICT (day, service) DO UPDATE SET
                    reads_used = read_budget.reads_used + EXCLUDED.reads_used,
                    updated_at = now()
                RETURNING reads_used
                """,
                (service, reads),
            )
            return int(cur.fetchone()["reads_used"])

    # -- maintenance --------------------------------------------------------

    def purge_before(self, cutoff: datetime) -> dict[str, int]:
        """Drop time-series rows older than `cutoff`. Markets are never purged."""
        counts: dict[str, int] = {}
        with self.cursor() as cur:
            for table, column in (
                ("posts", "created_at"),
                ("stances", "created_at"),
                ("signals", "ts"),
                ("market_prices", "ts"),
                ("sim_prices", "ts"),
            ):
                cur.execute(f"DELETE FROM {table} WHERE {column} < %s", (cutoff,))
                counts[table] = cur.rowcount or 0
        return counts

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.cursor() as cur:
            for table in (
                "markets",
                "market_queries",
                "posts",
                "stances",
                "signals",
                "market_prices",
                "sim_prices",
                "ledger_fills",
                "positions",
                "scores",
                "granger_results",
            ):
                cur.execute(f"SELECT count(*) AS n FROM {table}")
                counts[table] = int(cur.fetchone()["n"])
        return counts


def default_lookback(days: int = 14) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)

-- 001 — extensions, migration bookkeeping, and the market reference tables.
--
-- Design note: every table keyed on a market uses the canonical text id
-- "<venue>:<ticker>" (e.g. "kalshi:KXFED-26MAR19"). Kalshi tickers and
-- Polymarket condition ids share no namespace, so a composite text key keeps
-- cross-venue joins unambiguous without a surrogate integer everywhere.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS markets (
    market_id        TEXT PRIMARY KEY,
    venue            TEXT        NOT NULL CHECK (venue IN ('kalshi', 'polymarket')),
    ticker           TEXT        NOT NULL,
    series_ticker    TEXT,
    title            TEXT        NOT NULL,
    -- The sentence the stance model is conditioned on, e.g. "The Federal
    -- Reserve will cut rates at the March 2026 meeting." Enhancement 1 needs a
    -- fixed target per post rather than inferring one.
    target           TEXT        NOT NULL,
    open_time        TIMESTAMPTZ,
    close_time       TIMESTAMPTZ,
    status           TEXT        NOT NULL DEFAULT 'open'
                                 CHECK (status IN ('open', 'closed', 'settled', 'unknown')),
    resolved_outcome SMALLINT    CHECK (resolved_outcome IN (0, 1)),
    resolution_time  TIMESTAMPTZ,
    tracked          BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (venue, ticker),
    -- A market is settled exactly when it has an outcome. Guards against the
    -- resolution watcher freezing a Brier score against a NULL outcome.
    CONSTRAINT settled_iff_outcome
        CHECK ((status = 'settled') = (resolved_outcome IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS markets_tracked_status_idx
    ON markets (tracked, status)
    WHERE tracked;

-- Enhancement 5: market-to-query mapping. Source of truth is config/markets.yaml;
-- this table is the mirror the Rust ingester and the NLP pipeline read from so
-- they do not each need a YAML parser pointed at the same file.
CREATE TABLE IF NOT EXISTS market_queries (
    market_id TEXT NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    term_type TEXT NOT NULL CHECK (term_type IN ('keyword', 'ticker', 'entity', 'alias', 'exclude')),
    term      TEXT NOT NULL,
    PRIMARY KEY (market_id, term_type, term)
);

CREATE INDEX IF NOT EXISTS market_queries_term_idx ON market_queries (term);

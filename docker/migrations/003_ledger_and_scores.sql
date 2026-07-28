-- 003 — paper-trading ledger, calibration scores, and econometric results.
--
-- Nothing in here touches real money or a real venue. The ledger is a
-- simulation of what a strategy driven by S(t) would have done.

-- ---------------------------------------------------------------------------
-- ledger_fills — append-only record of simulated fills.
-- ---------------------------------------------------------------------------
-- Append-only on purpose: positions are derivable from fills, so the fill log
-- is the source of truth and `positions` below is a materialized convenience
-- the API can read without replaying history on every request.
CREATE TABLE IF NOT EXISTS ledger_fills (
    fill_id   BIGSERIAL,
    market_id TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    ts        TIMESTAMPTZ      NOT NULL,
    strategy  TEXT             NOT NULL DEFAULT 'signal_v1',
    side      TEXT             NOT NULL CHECK (side IN ('yes', 'no')),
    action    TEXT             NOT NULL CHECK (action IN ('buy', 'sell', 'settle')),
    quantity  DOUBLE PRECISION NOT NULL CHECK (quantity > 0),
    price     DOUBLE PRECISION NOT NULL CHECK (price BETWEEN 0 AND 1),
    -- Signed cash flow: negative when buying, positive when selling or settling.
    cash_flow DOUBLE PRECISION NOT NULL,
    -- The S(t) value that triggered this fill, for post-hoc attribution.
    signal    DOUBLE PRECISION CHECK (signal BETWEEN -1 AND 1),
    note      TEXT,

    PRIMARY KEY (fill_id, ts)
);

SELECT create_hypertable('ledger_fills', 'ts',
                         chunk_time_interval => INTERVAL '30 days',
                         if_not_exists       => TRUE);

CREATE INDEX IF NOT EXISTS ledger_fills_market_strategy_idx
    ON ledger_fills (market_id, strategy, ts DESC);

-- ---------------------------------------------------------------------------
-- positions — current simulated position per (market, strategy).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positions (
    market_id    TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    strategy     TEXT             NOT NULL DEFAULT 'signal_v1',
    qty_yes      DOUBLE PRECISION NOT NULL DEFAULT 0,
    qty_no       DOUBLE PRECISION NOT NULL DEFAULT 0,
    cash         DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Set when the market settles and the position is closed out.
    settled      BOOLEAN          NOT NULL DEFAULT FALSE,
    updated_at   TIMESTAMPTZ      NOT NULL DEFAULT now(),

    PRIMARY KEY (market_id, strategy)
);

-- ---------------------------------------------------------------------------
-- scores — Brier / Brier Skill Score per market per model version.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scores (
    market_id        TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    model_version    TEXT             NOT NULL,
    computed_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    window_start     TIMESTAMPTZ,
    window_end       TIMESTAMPTZ,
    n_obs            INTEGER          NOT NULL CHECK (n_obs > 0),
    -- Which s_t -> probability mapping the signal Brier score was computed
    -- under. Comparing a Platt-calibrated score against an affine one is
    -- meaningless, so it is recorded rather than assumed.
    calibration      TEXT             NOT NULL CHECK (calibration IN ('affine', 'platt')),
    brier_signal     DOUBLE PRECISION NOT NULL CHECK (brier_signal BETWEEN 0 AND 1),
    -- Reference forecast: the market's own price at the same timestamps.
    brier_market     DOUBLE PRECISION NOT NULL CHECK (brier_market BETWEEN 0 AND 1),
    -- BSS = 1 - brier_signal / brier_market. Positive means the social signal
    -- beat the market; <= 0 is a legitimate result, not a failure.
    bss              DOUBLE PRECISION NOT NULL,
    resolved_outcome SMALLINT         CHECK (resolved_outcome IN (0, 1)),
    -- TRUE once the market has settled and the score will not change again.
    final            BOOLEAN          NOT NULL DEFAULT FALSE,

    PRIMARY KEY (market_id, model_version, computed_at)
);

CREATE INDEX IF NOT EXISTS scores_final_idx
    ON scores (market_id, model_version, computed_at DESC)
    WHERE final;

-- ---------------------------------------------------------------------------
-- granger_results — written by augury-analytics, served by augury-api.
-- ---------------------------------------------------------------------------
-- p_value_adj is the Benjamini-Hochberg FDR-corrected p-value across the whole
-- batch of markets tested in one run. `significant` is derived from the
-- adjusted value, never the raw one: testing dozens of markets at p < 0.05
-- manufactures false "sentiment leads price" findings by chance alone.
CREATE TABLE IF NOT EXISTS granger_results (
    run_id            TEXT             NOT NULL,
    market_id         TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    model_version     TEXT             NOT NULL,
    computed_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    direction         TEXT             NOT NULL CHECK (direction IN ('signal_to_price', 'price_to_signal')),
    lag_order         INTEGER          NOT NULL CHECK (lag_order > 0),
    lag_criterion     TEXT             NOT NULL CHECK (lag_criterion IN ('aic', 'bic')),
    n_obs             INTEGER          NOT NULL CHECK (n_obs > 0),
    f_statistic       DOUBLE PRECISION NOT NULL,
    p_value           DOUBLE PRECISION NOT NULL CHECK (p_value BETWEEN 0 AND 1),
    p_value_adj       DOUBLE PRECISION NOT NULL CHECK (p_value_adj BETWEEN 0 AND 1),
    significant       BOOLEAN          NOT NULL,
    -- Stationarity evidence for the logit-transformed series that were tested.
    adf_p_signal      DOUBLE PRECISION CHECK (adf_p_signal BETWEEN 0 AND 1),
    adf_p_price       DOUBLE PRECISION CHECK (adf_p_price BETWEEN 0 AND 1),
    -- TRUE when spread/depth were included as exogenous controls.
    liquidity_control BOOLEAN          NOT NULL DEFAULT FALSE,

    PRIMARY KEY (run_id, market_id, model_version, direction)
);

-- ---------------------------------------------------------------------------
-- read_budget — persisted daily X API read counter.
-- ---------------------------------------------------------------------------
-- Lives in the database rather than in service memory so that restarting
-- augury-ingest cannot reset the counter and overspend the daily budget.
CREATE TABLE IF NOT EXISTS read_budget (
    day        DATE    NOT NULL,
    service    TEXT    NOT NULL,
    reads_used INTEGER NOT NULL DEFAULT 0 CHECK (reads_used >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (day, service)
);

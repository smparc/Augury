-- 002 — the time-series core, as TimescaleDB hypertables.
--
-- Hypertable rule that shapes every primary key below: any UNIQUE/PRIMARY KEY
-- constraint must include the partitioning column. That is why the post
-- timestamp appears in the key of `posts` and `stances` rather than post_id
-- alone.

-- ---------------------------------------------------------------------------
-- posts — raw ingested social posts, including the ones the filter rejected.
-- ---------------------------------------------------------------------------
-- A single post can be relevant to more than one tracked market, so the key is
-- (post_id, market_id, created_at) and the same text may legitimately appear
-- twice under different market_ids.
CREATE TABLE IF NOT EXISTS posts (
    post_id           TEXT        NOT NULL,
    market_id         TEXT        NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    created_at        TIMESTAMPTZ NOT NULL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    author_id         TEXT,
    -- Account age is a bot-filter input (enhancement 2), so it is stored per
    -- post rather than looked up again later.
    author_created_at TIMESTAMPTZ,
    followers         INTEGER     NOT NULL DEFAULT 0 CHECK (followers >= 0),
    engagements       INTEGER     NOT NULL DEFAULT 0 CHECK (engagements >= 0),
    lang              TEXT,
    text              TEXT        NOT NULL,
    -- MinHash signature + the LSH band bucket the Rust filter assigned. Kept so
    -- near-duplicate decisions can be re-audited without re-shingling the text.
    minhash           BIGINT[],
    lsh_bucket        TEXT,
    -- Rejected posts are stored, not dropped, so the filter's own precision is
    -- measurable after the fact.
    filter_verdict    TEXT        NOT NULL DEFAULT 'accepted'
                                  CHECK (filter_verdict IN ('accepted', 'duplicate', 'bot', 'low_quality', 'off_topic')),
    filter_reason     TEXT,
    source            TEXT        NOT NULL DEFAULT 'x' CHECK (source IN ('x', 'fixture')),

    PRIMARY KEY (post_id, market_id, created_at)
);

SELECT create_hypertable('posts', 'created_at',
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists       => TRUE);

CREATE INDEX IF NOT EXISTS posts_market_time_idx
    ON posts (market_id, created_at DESC);
-- The signal query only ever aggregates accepted posts; a partial index keeps
-- rejected spam out of the hot path.
CREATE INDEX IF NOT EXISTS posts_accepted_idx
    ON posts (market_id, created_at DESC)
    WHERE filter_verdict = 'accepted';
CREATE INDEX IF NOT EXISTS posts_lsh_idx
    ON posts (lsh_bucket)
    WHERE lsh_bucket IS NOT NULL;

-- ---------------------------------------------------------------------------
-- stances — model output, one row per (post, market, model version).
-- ---------------------------------------------------------------------------
-- Deliberately separate from `posts`: re-scoring the corpus with a new model
-- must never overwrite the scores an earlier model produced, because the Brier
-- comparison between model versions depends on both surviving.
CREATE TABLE IF NOT EXISTS stances (
    post_id       TEXT             NOT NULL,
    market_id     TEXT             NOT NULL,
    created_at    TIMESTAMPTZ      NOT NULL,
    model_version TEXT             NOT NULL,
    stance        DOUBLE PRECISION NOT NULL CHECK (stance BETWEEN -1 AND 1),
    confidence    DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    -- Ensemble disagreement flag: |stance_deberta - stance_vader|, NULL when
    -- only one model has scored the post.
    disagreement  DOUBLE PRECISION CHECK (disagreement >= 0),
    scored_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),

    PRIMARY KEY (post_id, market_id, model_version, created_at)
);

SELECT create_hypertable('stances', 'created_at',
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists       => TRUE);

CREATE INDEX IF NOT EXISTS stances_market_model_time_idx
    ON stances (market_id, model_version, created_at DESC);

-- ---------------------------------------------------------------------------
-- signals — the aggregated S(t) series.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    market_id         TEXT             NOT NULL,
    ts                TIMESTAMPTZ      NOT NULL,
    model_version     TEXT             NOT NULL,
    s_t               DOUBLE PRECISION NOT NULL CHECK (s_t BETWEEN -1 AND 1),
    -- S(t) lives in [-1,1] but the Brier score needs a probability. p_hat is
    -- the calibrated mapping of s_t into [0,1]; `calibration` records which
    -- mapping produced it so scores computed under different mappings are
    -- never silently compared.
    p_hat             DOUBLE PRECISION CHECK (p_hat BETWEEN 0 AND 1),
    calibration       TEXT             CHECK (calibration IN ('affine', 'platt')),
    n_posts           INTEGER          NOT NULL CHECK (n_posts >= 0),
    weight_sum        DOUBLE PRECISION NOT NULL CHECK (weight_sum >= 0),
    -- Effective half-life at this timestamp; varies with enhancement 3's
    -- adaptive decay, so it is recorded per point rather than assumed constant.
    half_life_seconds DOUBLE PRECISION NOT NULL CHECK (half_life_seconds > 0),
    computed_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),

    PRIMARY KEY (market_id, ts, model_version)
);

SELECT create_hypertable('signals', 'ts',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists       => TRUE);

-- ---------------------------------------------------------------------------
-- market_prices — real venue prices, plus book depth/spread (enhancement 4).
-- ---------------------------------------------------------------------------
-- `source` distinguishes a historical candle close from a live order-book
-- snapshot. Both are price observations but only the book rows carry depth,
-- and mixing them without a discriminator would corrupt the lead-lag grid.
CREATE TABLE IF NOT EXISTS market_prices (
    market_id  TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    ts         TIMESTAMPTZ      NOT NULL,
    source     TEXT             NOT NULL CHECK (source IN ('candle', 'book')),
    yes_price  DOUBLE PRECISION CHECK (yes_price BETWEEN 0 AND 1),
    yes_bid    DOUBLE PRECISION CHECK (yes_bid BETWEEN 0 AND 1),
    yes_ask    DOUBLE PRECISION CHECK (yes_ask BETWEEN 0 AND 1),
    spread     DOUBLE PRECISION CHECK (spread >= 0),
    -- Shares resting within the quoted range on each side. This is what the
    -- LMSR liquidity parameter b is calibrated from.
    depth_bid  DOUBLE PRECISION CHECK (depth_bid >= 0),
    depth_ask  DOUBLE PRECISION CHECK (depth_ask >= 0),
    volume     DOUBLE PRECISION CHECK (volume >= 0),
    fetched_at TIMESTAMPTZ      NOT NULL DEFAULT now(),

    PRIMARY KEY (market_id, ts, source)
);

SELECT create_hypertable('market_prices', 'ts',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists       => TRUE);

-- ---------------------------------------------------------------------------
-- sim_prices — LMSR synthetic market path.
-- ---------------------------------------------------------------------------
-- run_id separates the live simulation ('live') from named backtest runs, so a
-- walk-forward sweep can write many paths for the same market without
-- colliding with the live one.
CREATE TABLE IF NOT EXISTS sim_prices (
    market_id TEXT             NOT NULL REFERENCES markets (market_id) ON DELETE CASCADE,
    ts        TIMESTAMPTZ      NOT NULL,
    run_id    TEXT             NOT NULL DEFAULT 'live',
    sim_price DOUBLE PRECISION NOT NULL CHECK (sim_price BETWEEN 0 AND 1),
    b         DOUBLE PRECISION NOT NULL CHECK (b > 0),
    q_yes     DOUBLE PRECISION NOT NULL,
    q_no      DOUBLE PRECISION NOT NULL,
    cost      DOUBLE PRECISION,

    PRIMARY KEY (market_id, ts, run_id)
);

SELECT create_hypertable('sim_prices', 'ts',
                         chunk_time_interval => INTERVAL '7 days',
                         if_not_exists       => TRUE);

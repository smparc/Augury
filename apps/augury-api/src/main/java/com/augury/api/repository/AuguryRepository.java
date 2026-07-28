package com.augury.api.repository;

import com.augury.api.model.Records.Fill;
import com.augury.api.model.Records.GrangerResult;
import com.augury.api.model.Records.Market;
import com.augury.api.model.Records.Position;
import com.augury.api.model.Records.PriceTick;
import com.augury.api.model.Records.Score;
import com.augury.api.model.Records.SimTick;
import com.augury.api.model.Records.StanceSignal;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

/**
 * Read access to the shared TimescaleDB schema, plus ledger writes.
 *
 * <p>Plain SQL rather than an ORM. The schema is shared with the Python, Rust,
 * C++, and R components, and every query here is analytical — the SQL is the
 * thing worth reading, so it stays visible at the call site.
 */
@Repository
public class AuguryRepository {

    private final JdbcTemplate jdbc;

    public AuguryRepository(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** Nullable-safe timestamp read: {@code getTimestamp} returns null, not an epoch. */
    private static Instant instant(ResultSet rs, String column) throws SQLException {
        Timestamp value = rs.getTimestamp(column);
        return value == null ? null : value.toInstant();
    }

    /**
     * Nullable-safe double read.
     *
     * <p>{@code ResultSet.getDouble} returns 0.0 for SQL NULL, which for a
     * price or a depth is a meaningful and wrong value — 0.0 means "certain
     * NO", not "unknown". Every optional numeric goes through here.
     */
    private static Double nullableDouble(ResultSet rs, String column) throws SQLException {
        double value = rs.getDouble(column);
        return rs.wasNull() ? null : value;
    }

    private static Integer nullableInt(ResultSet rs, String column) throws SQLException {
        int value = rs.getInt(column);
        return rs.wasNull() ? null : value;
    }

    private static final RowMapper<Market> MARKET = (rs, i) -> new Market(
            rs.getString("market_id"),
            rs.getString("venue"),
            rs.getString("ticker"),
            rs.getString("title"),
            rs.getString("target"),
            instant(rs, "close_time"),
            rs.getString("status"),
            nullableInt(rs, "resolved_outcome"),
            instant(rs, "resolution_time")
    );

    private static final RowMapper<StanceSignal> SIGNAL = (rs, i) -> new StanceSignal(
            rs.getString("market_id"),
            instant(rs, "ts"),
            rs.getString("model_version"),
            rs.getDouble("s_t"),
            nullableDouble(rs, "p_hat"),
            rs.getString("calibration"),
            rs.getInt("n_posts"),
            rs.getDouble("weight_sum"),
            rs.getDouble("half_life_seconds")
    );

    private static final RowMapper<PriceTick> PRICE = (rs, i) -> new PriceTick(
            rs.getString("market_id"),
            instant(rs, "ts"),
            rs.getString("source"),
            nullableDouble(rs, "yes_price"),
            nullableDouble(rs, "yes_bid"),
            nullableDouble(rs, "yes_ask"),
            nullableDouble(rs, "spread"),
            nullableDouble(rs, "depth_bid"),
            nullableDouble(rs, "depth_ask"),
            nullableDouble(rs, "volume")
    );

    private static final RowMapper<SimTick> SIM = (rs, i) -> new SimTick(
            rs.getString("market_id"),
            instant(rs, "ts"),
            rs.getString("run_id"),
            rs.getDouble("sim_price"),
            rs.getDouble("b"),
            rs.getDouble("q_yes"),
            rs.getDouble("q_no"),
            nullableDouble(rs, "cost"),
            null
    );

    private static final RowMapper<Score> SCORE = (rs, i) -> new Score(
            rs.getString("market_id"),
            rs.getString("model_version"),
            instant(rs, "computed_at"),
            rs.getInt("n_obs"),
            rs.getString("calibration"),
            rs.getDouble("brier_signal"),
            rs.getDouble("brier_market"),
            rs.getDouble("bss"),
            nullableInt(rs, "resolved_outcome"),
            rs.getBoolean("final")
    );

    private static final RowMapper<Position> POSITION = (rs, i) -> new Position(
            rs.getString("market_id"),
            rs.getString("strategy"),
            rs.getDouble("qty_yes"),
            rs.getDouble("qty_no"),
            rs.getDouble("cash"),
            rs.getDouble("realized_pnl"),
            rs.getBoolean("settled"),
            instant(rs, "updated_at")
    );

    private static final RowMapper<Fill> FILL = (rs, i) -> new Fill(
            rs.getLong("fill_id"),
            rs.getString("market_id"),
            instant(rs, "ts"),
            rs.getString("strategy"),
            rs.getString("side"),
            rs.getString("action"),
            rs.getDouble("quantity"),
            rs.getDouble("price"),
            rs.getDouble("cash_flow"),
            nullableDouble(rs, "signal"),
            rs.getString("note")
    );

    private static final RowMapper<GrangerResult> GRANGER = (rs, i) -> new GrangerResult(
            rs.getString("run_id"),
            rs.getString("market_id"),
            rs.getString("model_version"),
            rs.getString("direction"),
            rs.getInt("lag_order"),
            rs.getString("lag_criterion"),
            rs.getInt("n_obs"),
            rs.getDouble("f_statistic"),
            rs.getDouble("p_value"),
            rs.getDouble("p_value_adj"),
            rs.getBoolean("significant"),
            nullableDouble(rs, "adf_p_signal"),
            nullableDouble(rs, "adf_p_price"),
            rs.getBoolean("liquidity_control")
    );

    // -- markets ------------------------------------------------------------

    public List<Market> findMarkets(boolean trackedOnly) {
        String sql = "SELECT * FROM markets"
                + (trackedOnly ? " WHERE tracked" : "")
                + " ORDER BY market_id";
        return jdbc.query(sql, MARKET);
    }

    public Optional<Market> findMarket(String marketId) {
        return jdbc.query("SELECT * FROM markets WHERE market_id = ?", MARKET, marketId)
                .stream()
                .findFirst();
    }

    // -- signals ------------------------------------------------------------

    public List<StanceSignal> signalSeries(String marketId, String modelVersion, Instant since, int limit) {
        return jdbc.query(
                """
                SELECT * FROM signals
                 WHERE market_id = ?
                   AND (? IS NULL OR model_version = ?)
                   AND (? IS NULL OR ts >= ?)
                 ORDER BY ts
                 LIMIT ?
                """,
                SIGNAL,
                marketId,
                modelVersion, modelVersion,
                since == null ? null : Timestamp.from(since),
                since == null ? null : Timestamp.from(since),
                limit);
    }

    public Optional<StanceSignal> latestSignal(String marketId) {
        return jdbc.query(
                        "SELECT * FROM signals WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
                        SIGNAL, marketId)
                .stream()
                .findFirst();
    }

    // -- prices -------------------------------------------------------------

    public List<PriceTick> priceSeries(String marketId, Instant since, int limit) {
        return jdbc.query(
                """
                SELECT * FROM market_prices
                 WHERE market_id = ?
                   AND (? IS NULL OR ts >= ?)
                 ORDER BY ts
                 LIMIT ?
                """,
                PRICE,
                marketId,
                since == null ? null : Timestamp.from(since),
                since == null ? null : Timestamp.from(since),
                limit);
    }

    public Optional<PriceTick> latestPrice(String marketId) {
        return jdbc.query(
                        "SELECT * FROM market_prices WHERE market_id = ? ORDER BY ts DESC LIMIT 1",
                        PRICE, marketId)
                .stream()
                .findFirst();
    }

    // -- simulation ---------------------------------------------------------

    public List<SimTick> simSeries(String marketId, String runId, Instant since, int limit) {
        return jdbc.query(
                """
                SELECT * FROM sim_prices
                 WHERE market_id = ?
                   AND run_id = ?
                   AND (? IS NULL OR ts >= ?)
                 ORDER BY ts
                 LIMIT ?
                """,
                SIM,
                marketId,
                runId,
                since == null ? null : Timestamp.from(since),
                since == null ? null : Timestamp.from(since),
                limit);
    }

    // -- scores and econometrics --------------------------------------------

    public List<Score> scores(String marketId) {
        return jdbc.query(
                """
                SELECT * FROM scores
                 WHERE (? IS NULL OR market_id = ?)
                 ORDER BY computed_at DESC
                 LIMIT 500
                """,
                SCORE, marketId, marketId);
    }

    /**
     * FDR-corrected Granger results from the latest analytics run.
     *
     * <p>Only the newest {@code run_id} is returned. Mixing runs would mean
     * mixing p-values corrected against different batches, and the correction
     * is only meaningful within the batch it was computed over.
     */
    public List<GrangerResult> latestGrangerResults() {
        return jdbc.query(
                """
                SELECT * FROM granger_results
                 WHERE run_id = (SELECT run_id FROM granger_results
                                  ORDER BY computed_at DESC LIMIT 1)
                 ORDER BY p_value_adj
                """,
                GRANGER);
    }

    // -- ledger -------------------------------------------------------------

    public List<Position> positions(String strategy, boolean openOnly) {
        String sql = """
                SELECT * FROM positions
                 WHERE strategy = ?
                """
                + (openOnly ? " AND NOT settled AND (qty_yes <> 0 OR qty_no <> 0)" : "")
                + " ORDER BY market_id";
        return jdbc.query(sql, POSITION, strategy);
    }

    public Optional<Position> position(String marketId, String strategy) {
        return jdbc.query(
                        "SELECT * FROM positions WHERE market_id = ? AND strategy = ?",
                        POSITION, marketId, strategy)
                .stream()
                .findFirst();
    }

    public List<Fill> fills(String marketId, String strategy, int limit) {
        return jdbc.query(
                """
                SELECT * FROM ledger_fills
                 WHERE (? IS NULL OR market_id = ?)
                   AND strategy = ?
                 ORDER BY ts DESC
                 LIMIT ?
                """,
                FILL, marketId, marketId, strategy, limit);
    }

    public void recordFill(String marketId, Instant ts, String strategy, String side,
                           String action, double quantity, double price, double cashFlow,
                           Double signal, String note) {
        jdbc.update(
                """
                INSERT INTO ledger_fills (market_id, ts, strategy, side, action, quantity,
                                          price, cash_flow, signal, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                marketId, Timestamp.from(ts), strategy, side, action,
                quantity, price, cashFlow, signal, note);
    }

    /** Apply a fill's deltas to the materialized position, creating it if absent. */
    public void applyToPosition(String marketId, String strategy, double dQtyYes, double dQtyNo,
                                double dCash, double dRealized, boolean settled) {
        jdbc.update(
                """
                INSERT INTO positions (market_id, strategy, qty_yes, qty_no, cash, realized_pnl, settled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (market_id, strategy) DO UPDATE SET
                    qty_yes      = positions.qty_yes + EXCLUDED.qty_yes,
                    qty_no       = positions.qty_no + EXCLUDED.qty_no,
                    cash         = positions.cash + EXCLUDED.cash,
                    realized_pnl = positions.realized_pnl + EXCLUDED.realized_pnl,
                    settled      = positions.settled OR EXCLUDED.settled,
                    updated_at   = now()
                """,
                marketId, strategy, dQtyYes, dQtyNo, dCash, dRealized, settled);
    }
}

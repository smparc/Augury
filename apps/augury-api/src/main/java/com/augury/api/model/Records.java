package com.augury.api.model;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.time.Instant;
import java.util.List;

/**
 * API response records.
 *
 * <p>Java records mirroring the JSON Schemas in {@code schemas/}. Field names
 * match the schema exactly so a payload produced here validates against the same
 * contract the Python and C++ services publish on Redis.
 */
public final class Records {

    private Records() {
    }

    /** A tracked market and its current state. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Market(
            String marketId,
            String venue,
            String ticker,
            String title,
            String target,
            Instant closeTime,
            String status,
            Integer resolvedOutcome,
            Instant resolutionTime
    ) {
        public boolean isSettled() {
            return "settled".equals(status);
        }
    }

    /**
     * One point of the aggregated social signal.
     *
     * <p>{@code sT} lives in [-1, 1] and is <em>not</em> a probability;
     * {@code pHat} is its calibrated mapping into [0, 1], and
     * {@code calibration} names the mapping used. Scores computed under
     * different calibrations are not comparable, which is why the label travels
     * with the number rather than being assumed by the consumer.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record StanceSignal(
            String marketId,
            Instant ts,
            String modelVersion,
            double sT,
            Double pHat,
            String calibration,
            int nPosts,
            double weightSum,
            double halfLifeSeconds
    ) {
    }

    /** One real market price observation. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record PriceTick(
            String marketId,
            Instant ts,
            String source,
            Double yesPrice,
            Double yesBid,
            Double yesAsk,
            Double spread,
            Double depthBid,
            Double depthAsk,
            Double volume
    ) {
        /**
         * Midpoint of the quoted range, falling back to the last trade.
         *
         * <p>Preferred over {@code yesPrice} for comparison work: a last-trade
         * price on a thin market can be hours stale while the quote has moved.
         */
        public Double mid() {
            if (yesBid != null && yesAsk != null) {
                return (yesBid + yesAsk) / 2.0;
            }
            return yesPrice;
        }
    }

    /** One step of the synthetic LMSR market. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record SimTick(
            String marketId,
            Instant ts,
            String runId,
            double simPrice,
            double b,
            double qYes,
            double qNo,
            Double cost,
            Double signal
    ) {
    }

    /** Calibration result for a resolved market. */
    public record Score(
            String marketId,
            String modelVersion,
            Instant computedAt,
            int nObs,
            String calibration,
            double brierSignal,
            double brierMarket,
            double bss,
            Integer resolvedOutcome,
            boolean isFinal
    ) {
        /** Whether the signal was better calibrated than the market itself. */
        public boolean beatsMarket() {
            return bss > 0;
        }
    }

    /** A simulated position. */
    public record Position(
            String marketId,
            String strategy,
            double qtyYes,
            double qtyNo,
            double cash,
            double realizedPnl,
            boolean settled,
            Instant updatedAt
    ) {
    }

    /** A simulated position marked to the current market price. */
    public record PositionValuation(
            Position position,
            Double markPrice,
            double marketValue,
            double unrealizedPnl,
            double totalPnl
    ) {
    }

    /** One simulated fill. */
    public record Fill(
            long fillId,
            String marketId,
            Instant ts,
            String strategy,
            String side,
            String action,
            double quantity,
            double price,
            double cashFlow,
            Double signal,
            String note
    ) {
    }

    /** Granger causality result, already FDR-corrected across the batch. */
    public record GrangerResult(
            String runId,
            String marketId,
            String modelVersion,
            String direction,
            int lagOrder,
            String lagCriterion,
            int nObs,
            double fStatistic,
            double pValue,
            double pValueAdj,
            boolean significant,
            Double adfPSignal,
            Double adfPPrice,
            boolean liquidityControl
    ) {
    }

    /** Aligned signal/price/simulation series for charting. */
    public record MarketSeries(
            String marketId,
            List<StanceSignal> signals,
            List<PriceTick> prices,
            List<SimTick> simulation
    ) {
    }

    /** Portfolio-level summary across every open position. */
    public record PortfolioSummary(
            String strategy,
            int openPositions,
            double realizedPnl,
            double unrealizedPnl,
            double totalPnl,
            List<PositionValuation> positions
    ) {
    }
}

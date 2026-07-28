package com.augury.api.service;

import com.augury.api.model.Records.Position;
import com.augury.api.model.Records.PositionValuation;
import com.augury.api.model.Records.PortfolioSummary;
import com.augury.api.model.Records.PriceTick;
import com.augury.api.model.Records.StanceSignal;
import com.augury.api.repository.AuguryRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Simulated portfolio driven by the social signal.
 *
 * <p>Everything here is paper. No venue credentials exist in this service and
 * no order ever leaves it — the ledger measures what a strategy <em>would</em>
 * have done, which is the only claim the rest of the system is set up to
 * support.
 *
 * <p>Sizing rule: the target position is proportional to the signal's edge over
 * the market's own price, {@code pHat - marketPrice}, not to the signal level
 * itself. A signal saying "80% likely" is worth nothing when the market already
 * says 80%; it is only informative when it disagrees. Sizing on the level alone
 * would pile into consensus positions and mistake agreement for insight.
 */
@Service
public class PaperTradingService {

    private static final Logger log = LoggerFactory.getLogger(PaperTradingService.class);

    /** Below this edge the signal is treated as agreeing with the market. */
    private static final double MIN_EDGE = 0.05;

    private final AuguryRepository repository;
    private final double maxPositionShares;
    private final String strategy;

    public PaperTradingService(
            AuguryRepository repository,
            @Value("${augury.paper.max-position-shares:100.0}") double maxPositionShares,
            @Value("${augury.paper.strategy:signal_v1}") String strategy) {
        this.repository = repository;
        this.maxPositionShares = maxPositionShares;
        this.strategy = strategy;
    }

    public String strategy() {
        return strategy;
    }

    /**
     * Target YES share count implied by a signal and the current market price.
     *
     * @return signed target: positive is long YES, negative is long NO.
     */
    public double targetPosition(double pHat, double marketPrice) {
        double edge = pHat - marketPrice;
        if (Math.abs(edge) < MIN_EDGE) {
            return 0.0;
        }
        // Linear in edge, capped. Edge is bounded by [-1, 1], so scaling by the
        // cap keeps a full-disagreement signal at exactly the position limit.
        return Math.max(-maxPositionShares, Math.min(maxPositionShares, edge * maxPositionShares));
    }

    /**
     * Rebalance a market's paper position toward what the latest signal implies.
     *
     * @return shares traded, or 0 when nothing was required.
     */
    public double rebalance(StanceSignal signal, PriceTick price) {
        if (signal == null || signal.pHat() == null || price == null) {
            return 0.0;
        }
        Double mark = price.mid();
        if (mark == null) {
            // No usable quote means no defensible fill price. Skipping is
            // correct; inventing one would fabricate P&L.
            return 0.0;
        }

        double target = targetPosition(signal.pHat(), mark);
        Position current = repository.position(signal.marketId(), strategy).orElse(null);
        double held = current == null ? 0.0 : current.qtyYes() - current.qtyNo();
        double delta = target - held;

        if (Math.abs(delta) < 1e-9) {
            return 0.0;
        }

        Instant now = Instant.now();
        String side = delta > 0 ? "yes" : "no";
        String action = delta > 0 ? "buy" : "sell";
        double quantity = Math.abs(delta);
        // Cash out when buying, in when selling.
        double cashFlow = -delta * mark;

        repository.recordFill(signal.marketId(), now, strategy, side, action,
                quantity, mark, cashFlow, signal.sT(), "signal rebalance");
        repository.applyToPosition(signal.marketId(), strategy, delta, 0.0, cashFlow, 0.0, false);

        log.info("paper rebalance {}: {} {} @ {} (target {}, held {})",
                signal.marketId(), action, quantity, mark, target, held);
        return delta;
    }

    /** Mark a position to the current price. */
    public PositionValuation value(Position position, Double markPrice) {
        double net = position.qtyYes() - position.qtyNo();

        if (markPrice == null) {
            // Without a mark the unrealized figure is unknowable. Reporting 0
            // would read as "flat", so the price is surfaced as null and the
            // caller can see the valuation is incomplete.
            return new PositionValuation(position, null, 0.0, 0.0, position.realizedPnl());
        }

        double marketValue = net * markPrice;
        double unrealized = marketValue + position.cash();
        return new PositionValuation(
                position, markPrice, marketValue, unrealized, position.realizedPnl() + unrealized);
    }

    /** Portfolio summary across every open position. */
    public PortfolioSummary portfolio() {
        List<Position> positions = repository.positions(strategy, true);
        List<PositionValuation> valuations = new ArrayList<>(positions.size());

        double realized = 0.0;
        double unrealized = 0.0;

        for (Position position : positions) {
            Optional<PriceTick> latest = repository.latestPrice(position.marketId());
            PositionValuation valuation = value(position, latest.map(PriceTick::mid).orElse(null));
            valuations.add(valuation);
            realized += position.realizedPnl();
            unrealized += valuation.unrealizedPnl();
        }

        return new PortfolioSummary(
                strategy, positions.size(), realized, unrealized, realized + unrealized, valuations);
    }
}

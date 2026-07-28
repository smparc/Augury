package com.augury.api;

import com.augury.api.model.Records.Position;
import com.augury.api.model.Records.PositionValuation;
import com.augury.api.model.Records.PriceTick;
import com.augury.api.repository.AuguryRepository;
import com.augury.api.service.PaperTradingService;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for the paper-trading arithmetic.
 *
 * <p>These cover the accounting invariants — that sizing keys off edge rather
 * than signal level, and that an unknown mark is reported as unknown instead of
 * being silently valued at zero.
 */
@ExtendWith(MockitoExtension.class)
class PaperTradingServiceTest {

    @Mock
    private AuguryRepository repository;

    private PaperTradingService service;

    @BeforeEach
    void setUp() {
        service = new PaperTradingService(repository, 100.0, "signal_v1");
    }

    @Test
    @DisplayName("agreeing with the market implies no position")
    void noEdgeMeansNoPosition() {
        // The signal says 70% and so does the market: no information, no trade.
        assertThat(service.targetPosition(0.70, 0.70)).isZero();
        assertThat(service.targetPosition(0.72, 0.70)).isZero();
    }

    @Test
    @DisplayName("position size scales with the edge, not the signal level")
    void sizeScalesWithEdge() {
        // A very bullish signal against an equally bullish market: no edge.
        double consensus = service.targetPosition(0.95, 0.95);
        // A mildly bullish signal against a bearish market: real edge.
        double contrarian = service.targetPosition(0.55, 0.20);

        assertThat(consensus).isZero();
        assertThat(contrarian).isGreaterThan(0.0);
    }

    @Test
    @DisplayName("a negative edge goes short")
    void negativeEdgeGoesShort() {
        assertThat(service.targetPosition(0.20, 0.80)).isNegative();
    }

    @Test
    @DisplayName("position size is capped")
    void positionIsCapped() {
        assertThat(service.targetPosition(1.0, 0.0)).isLessThanOrEqualTo(100.0);
        assertThat(service.targetPosition(0.0, 1.0)).isGreaterThanOrEqualTo(-100.0);
    }

    @Test
    @DisplayName("an unknown mark price is reported as unknown, not as zero P&L")
    void unknownMarkIsNotZero() {
        Position position = new Position(
                "kalshi:TEST", "signal_v1", 50.0, 0.0, -20.0, 5.0, false, Instant.now());

        PositionValuation valuation = service.value(position, null);

        assertThat(valuation.markPrice()).isNull();
        assertThat(valuation.unrealizedPnl()).isZero();
        // Realized P&L is still known and must survive.
        assertThat(valuation.totalPnl()).isEqualTo(5.0);
    }

    @Test
    @DisplayName("valuation reconciles market value against cash")
    void valuationReconciles() {
        // Bought 50 YES at 0.40: cash out 20, now marked at 0.60.
        Position position = new Position(
                "kalshi:TEST", "signal_v1", 50.0, 0.0, -20.0, 0.0, false, Instant.now());

        PositionValuation valuation = service.value(position, 0.60);

        assertThat(valuation.marketValue()).isEqualTo(30.0);
        // 30 of value against 20 of cash spent is a 10 gain.
        assertThat(valuation.unrealizedPnl()).isEqualTo(10.0);
        assertThat(valuation.totalPnl()).isEqualTo(10.0);
    }

    @Test
    @DisplayName("a missing quote is skipped rather than filled at an invented price")
    void missingQuoteSkipsTrade() {
        var signal = new com.augury.api.model.Records.StanceSignal(
                "kalshi:TEST", Instant.now(), "vader-3.3.2", 0.8, 0.9, "affine", 12, 100.0, 21600.0);
        // No bid, no ask, no last trade: nothing defensible to fill against.
        var price = new PriceTick(
                "kalshi:TEST", Instant.now(), "book", null, null, null, null, null, null, null);

        double traded = service.rebalance(signal, price);

        assertThat(traded).isZero();
        verify(repository, never()).recordFill(
                anyString(), any(), anyString(), anyString(), anyString(),
                anyDouble(), anyDouble(), anyDouble(), any(), anyString());
    }

    @Test
    @DisplayName("a signal without a calibrated probability does not trade")
    void uncalibratedSignalDoesNotTrade() {
        var signal = new com.augury.api.model.Records.StanceSignal(
                "kalshi:TEST", Instant.now(), "vader-3.3.2", 0.8, null, null, 12, 100.0, 21600.0);
        var price = new PriceTick(
                "kalshi:TEST", Instant.now(), "book", 0.3, 0.29, 0.31, 0.02, 100.0, 100.0, null);

        assertThat(service.rebalance(signal, price)).isZero();
    }

    @Test
    @DisplayName("rebalancing from flat opens a position and records the fill")
    void rebalanceOpensPosition() {
        var signal = new com.augury.api.model.Records.StanceSignal(
                "kalshi:TEST", Instant.now(), "vader-3.3.2", 0.6, 0.80, "affine", 12, 100.0, 21600.0);
        var price = new PriceTick(
                "kalshi:TEST", Instant.now(), "book", 0.30, 0.29, 0.31, 0.02, 100.0, 100.0, null);

        when(repository.position("kalshi:TEST", "signal_v1")).thenReturn(java.util.Optional.empty());

        double traded = service.rebalance(signal, price);

        // Edge is 0.80 - 0.30 = 0.50, so the target is +50 shares.
        assertThat(traded).isEqualTo(50.0);
        verify(repository).recordFill(
                anyString(), any(), anyString(), anyString(), anyString(),
                anyDouble(), anyDouble(), anyDouble(), any(), anyString());
        verify(repository).applyToPosition(
                anyString(), anyString(), anyDouble(), anyDouble(), anyDouble(), anyDouble(), anyBoolean());
    }
}

package com.augury.api.web;

import com.augury.api.model.Records.Fill;
import com.augury.api.model.Records.GrangerResult;
import com.augury.api.model.Records.Market;
import com.augury.api.model.Records.MarketSeries;
import com.augury.api.model.Records.PortfolioSummary;
import com.augury.api.model.Records.PriceTick;
import com.augury.api.model.Records.Score;
import com.augury.api.model.Records.StanceSignal;
import com.augury.api.repository.AuguryRepository;
import com.augury.api.service.PaperTradingService;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/**
 * REST surface over the Augury data.
 *
 * <p>Blocking JDBC calls are pushed onto the bounded-elastic scheduler rather
 * than run on the event loop. Blocking a Netty worker thread stalls every other
 * request it is multiplexing, and with a small worker pool that is a
 * service-wide outage rather than one slow response.
 */
@RestController
@RequestMapping("/api")
public class AuguryController {

    private static final int MAX_LIMIT = 10_000;

    private final AuguryRepository repository;
    private final PaperTradingService paperTrading;

    public AuguryController(AuguryRepository repository, PaperTradingService paperTrading) {
        this.repository = repository;
        this.paperTrading = paperTrading;
    }

    private static <T> Mono<T> offload(java.util.concurrent.Callable<T> work) {
        return Mono.fromCallable(work).subscribeOn(reactor.core.scheduler.Schedulers.boundedElastic());
    }

    private static Instant since(Integer hours) {
        return hours == null ? null : Instant.now().minus(Duration.ofHours(hours));
    }

    private static int clampLimit(Integer limit) {
        if (limit == null) {
            return 5_000;
        }
        if (limit < 1) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "limit must be positive");
        }
        return Math.min(limit, MAX_LIMIT);
    }

    // -- markets ------------------------------------------------------------

    @GetMapping("/markets")
    public Mono<List<Market>> markets(
            @RequestParam(defaultValue = "true") boolean trackedOnly) {
        return offload(() -> repository.findMarkets(trackedOnly));
    }

    @GetMapping("/markets/{marketId}")
    public Mono<Market> market(@PathVariable String marketId) {
        return offload(() -> repository.findMarket(marketId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "unknown market: " + marketId)));
    }

    // -- signals and prices -------------------------------------------------

    @GetMapping("/markets/{marketId}/signal")
    public Mono<StanceSignal> latestSignal(@PathVariable String marketId) {
        return offload(() -> repository.latestSignal(marketId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.NOT_FOUND, "no signal computed yet for " + marketId)));
    }

    @GetMapping("/markets/{marketId}/signals")
    public Mono<List<StanceSignal>> signals(
            @PathVariable String marketId,
            @RequestParam(required = false) String modelVersion,
            @RequestParam(required = false) Integer hours,
            @RequestParam(required = false) Integer limit) {
        return offload(() ->
                repository.signalSeries(marketId, modelVersion, since(hours), clampLimit(limit)));
    }

    @GetMapping("/markets/{marketId}/prices")
    public Mono<List<PriceTick>> prices(
            @PathVariable String marketId,
            @RequestParam(required = false) Integer hours,
            @RequestParam(required = false) Integer limit) {
        return offload(() -> repository.priceSeries(marketId, since(hours), clampLimit(limit)));
    }

    /**
     * Everything needed to chart one market: signal, real price, and simulation.
     *
     * <p>One round trip rather than three, because a client plotting these
     * against each other would otherwise risk fetching them at slightly
     * different times and drawing series that do not line up.
     */
    @GetMapping("/markets/{marketId}/series")
    public Mono<MarketSeries> series(
            @PathVariable String marketId,
            @RequestParam(required = false) String modelVersion,
            @RequestParam(required = false) Integer hours,
            @RequestParam(defaultValue = "live") String runId,
            @RequestParam(required = false) Integer limit) {
        return offload(() -> {
            Instant from = since(hours);
            int cap = clampLimit(limit);
            return new MarketSeries(
                    marketId,
                    repository.signalSeries(marketId, modelVersion, from, cap),
                    repository.priceSeries(marketId, from, cap),
                    repository.simSeries(marketId, runId, from, cap));
        });
    }

    // -- calibration and econometrics ---------------------------------------

    @GetMapping("/scores")
    public Mono<List<Score>> scores(@RequestParam(required = false) String marketId) {
        return offload(() -> repository.scores(marketId));
    }

    /**
     * Granger results from the most recent analytics run.
     *
     * <p>{@code significant} is defined off the FDR-adjusted p-value only. The
     * raw {@code pValue} is included for transparency, but filtering on it
     * across many markets is exactly the multiple-comparisons error the
     * correction exists to prevent.
     */
    @GetMapping("/granger")
    public Mono<List<GrangerResult>> granger() {
        return offload(repository::latestGrangerResults);
    }

    // -- paper trading ------------------------------------------------------

    @GetMapping("/portfolio")
    public Mono<PortfolioSummary> portfolio() {
        return offload(paperTrading::portfolio);
    }

    @GetMapping("/fills")
    public Mono<List<Fill>> fills(
            @RequestParam(required = false) String marketId,
            @RequestParam(required = false) Integer limit) {
        return offload(() ->
                repository.fills(marketId, paperTrading.strategy(), clampLimit(limit)));
    }

    // -- server-sent events -------------------------------------------------

    /**
     * Signal updates as SSE, for clients that would rather not hold a WebSocket.
     *
     * <p>Polls the database on an interval instead of subscribing to Redis:
     * SSE consumers are typically dashboards where a few seconds of latency is
     * irrelevant, and this keeps them off the pub/sub path entirely.
     */
    @GetMapping(value = "/markets/{marketId}/signal/stream", produces = "text/event-stream")
    public Flux<StanceSignal> signalStream(@PathVariable String marketId) {
        return Flux.interval(Duration.ZERO, Duration.ofSeconds(5))
                .flatMap(tick -> offload(() -> repository.latestSignal(marketId).orElse(null)))
                .distinctUntilChanged(signal -> signal == null ? null : signal.ts());
    }
}

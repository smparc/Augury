package com.augury.api.web;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.WebSocketSession;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;

/**
 * Fans the shared Redis stream out to WebSocket clients.
 *
 * <p>Connect to {@code /ws/stream} for every tracked market, or
 * {@code /ws/stream/{marketId}} for one. Messages arrive as
 * {@code <channel>|<json>}, where the channel names the message type
 * ({@code augury.signal.*}, {@code augury.sim.*}, {@code augury.price.*}).
 */
@Component
public class AuguryWebSocketHandler implements WebSocketHandler {

    private static final Logger log = LoggerFactory.getLogger(AuguryWebSocketHandler.class);

    /**
     * Keepalive interval. Idle WebSockets are routinely closed by proxies after
     * a minute or so, and a market can easily go that long without a tick, so
     * silence has to be distinguishable from a dead connection.
     */
    private static final Duration HEARTBEAT = Duration.ofSeconds(30);

    private final Flux<String> auguryStream;

    public AuguryWebSocketHandler(Flux<String> auguryStream) {
        this.auguryStream = auguryStream;
    }

    /** Trailing path segment, when the client subscribed to a single market. */
    private static String marketFilter(WebSocketSession session) {
        URI uri = session.getHandshakeInfo().getUri();
        String path = uri.getPath();
        String prefix = "/ws/stream/";
        if (path.startsWith(prefix) && path.length() > prefix.length()) {
            return path.substring(prefix.length());
        }
        return null;
    }

    @Override
    public Mono<Void> handle(WebSocketSession session) {
        String marketId = marketFilter(session);
        log.info("websocket open: {} (filter: {})", session.getId(),
                marketId == null ? "all markets" : marketId);

        Flux<String> payloads = auguryStream
                .filter(message -> marketId == null || message.contains(marketId));

        Flux<String> heartbeats = Flux.interval(HEARTBEAT, HEARTBEAT)
                .map(tick -> "augury.heartbeat|{\"ts\":\"" + java.time.Instant.now() + "\"}");

        return session
                .send(Flux.merge(payloads, heartbeats).map(session::textMessage))
                .doFinally(signal -> log.info("websocket closed: {} ({})", session.getId(), signal));
    }
}

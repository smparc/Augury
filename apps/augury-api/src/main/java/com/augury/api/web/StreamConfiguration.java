package com.augury.api.web;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.ReactiveRedisConnectionFactory;
import org.springframework.data.redis.core.ReactiveStringRedisTemplate;
import org.springframework.data.redis.listener.PatternTopic;
import org.springframework.web.reactive.HandlerMapping;
import org.springframework.web.reactive.handler.SimpleUrlHandlerMapping;
import org.springframework.web.reactive.socket.WebSocketHandler;
import org.springframework.web.reactive.socket.server.support.WebSocketHandlerAdapter;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Sinks;

import java.util.Map;

/**
 * Live streaming: Redis pub/sub fanned out to WebSocket subscribers.
 *
 * <p>The pipeline publishes to {@code augury.signal.*} and {@code augury.sim.*}.
 * A single Redis subscription per pattern feeds a multicast sink, and every
 * WebSocket client reads from that sink. One subscription serving N clients
 * rather than N subscriptions is the difference between a dashboard with fifty
 * tabs open being free and being a load generator.
 */
@Configuration
public class StreamConfiguration {

    private static final Logger log = LoggerFactory.getLogger(StreamConfiguration.class);

    /**
     * Shared broadcast of every message on the Augury channels.
     *
     * <p>{@code onBackpressureBuffer} with {@code replay().limit(0)} means a new
     * subscriber gets messages from the moment it connects and never a backlog.
     * Replaying history to a late joiner would show it stale prices as though
     * they were live.
     */
    @Bean
    public Sinks.Many<String> streamSink() {
        return Sinks.many().multicast().onBackpressureBuffer(256, false);
    }

    @Bean
    public Flux<String> auguryStream(
            ReactiveRedisConnectionFactory connectionFactory,
            Sinks.Many<String> streamSink) {

        ReactiveStringRedisTemplate template = new ReactiveStringRedisTemplate(connectionFactory);

        template.listenTo(PatternTopic.of("augury.*"))
                .map(message -> message.getChannel() + "|" + message.getMessage())
                // A Redis outage must not take the API down: the REST surface
                // reads from TimescaleDB and stays fully functional.
                .doOnError(error -> log.warn("redis subscription error: {}", error.getMessage()))
                .retryWhen(reactor.util.retry.Retry.backoff(
                        Long.MAX_VALUE, java.time.Duration.ofSeconds(2)))
                .subscribe(
                        payload -> streamSink.tryEmitNext(payload),
                        error -> log.error("redis stream terminated", error));

        return streamSink.asFlux();
    }

    @Bean
    public HandlerMapping webSocketHandlerMapping(AuguryWebSocketHandler handler) {
        SimpleUrlHandlerMapping mapping = new SimpleUrlHandlerMapping();
        mapping.setUrlMap(Map.of(
                "/ws/stream", (WebSocketHandler) handler,
                "/ws/stream/{marketId}", (WebSocketHandler) handler));
        // Ahead of the annotated controllers so the WebSocket upgrade is not
        // swallowed by request mapping.
        mapping.setOrder(-1);
        return mapping;
    }

    @Bean
    public WebSocketHandlerAdapter webSocketHandlerAdapter() {
        return new WebSocketHandlerAdapter();
    }
}

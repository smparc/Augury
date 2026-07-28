package com.augury.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Augury API — reactive REST and WebSocket access to the signal pipeline.
 *
 * <p>Read-only with respect to market data. The one thing this service writes
 * is the simulated ledger, and it holds no venue credentials of any kind: every
 * position here is paper. That is a deliberate boundary, not an unfinished
 * feature — the system is a research instrument, and giving the API the ability
 * to place real orders would change what it is.
 */
@SpringBootApplication
public class AuguryApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(AuguryApiApplication.class, args);
    }
}

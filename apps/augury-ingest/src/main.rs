//! augury-ingest — fault-tolerant X post ingestion.
//!
//! One task per tracked market, each polling on its own interval, filtering
//! through MinHash near-duplicate detection and account heuristics, and writing
//! to TimescaleDB. Fixture replay is the default; live reads require
//! `AUGURY_X_LIVE=1`, a bearer token, and headroom in the persisted daily
//! budget.
//!
//! Failure policy: a market whose poll fails logs and retries on the next tick
//! rather than taking the process down. One venue hiccup must not stop
//! ingestion for every other market — the whole point of the per-market task
//! split.

mod budget;
mod config;
mod db;
mod filter;
mod minhash;
mod models;
mod x_client;

use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{Duration as ChronoDuration, Utc};
use clap::{Parser, Subcommand};
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

use crate::budget::ReadBudget;
use crate::config::Config;
use crate::filter::{FilterConfig, PostFilter};
use crate::x_client::{FixtureSource, LiveSource, PostSource};

#[derive(Parser)]
#[command(name = "augury-ingest", version, about = "Augury X ingestion service")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Poll every tracked market once and exit.
    Once,
    /// Poll continuously until interrupted.
    Run {
        /// Seconds between polls; overrides AUGURY_POLL_INTERVAL_SECS.
        #[arg(long)]
        interval_secs: Option<u64>,
    },
    /// Show configuration and connectivity without ingesting.
    Status,
    /// Run the filter over fixtures and print the verdicts. No database needed.
    DryRun {
        /// Market id to replay; defaults to the Fed fixture.
        #[arg(long, default_value = "kalshi:KXFEDDECISION-26JUL-C25")]
        market: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.clone().into()),
        )
        .with_target(false)
        .init();

    match cli.command {
        Command::Once => poll_all(&config).await,
        Command::Run { interval_secs } => {
            let secs = interval_secs.unwrap_or(config.poll_interval_secs);
            run_forever(&config, secs).await
        }
        Command::Status => status(&config).await,
        Command::DryRun { market } => dry_run(&config, &market).await,
    }
}

/// Build the post source implied by the configuration.
fn build_source(config: &Config, pool: Option<&sqlx::PgPool>) -> Arc<dyn PostSource> {
    if !config.x_live {
        info!("post source: fixture replay (AUGURY_X_LIVE is off — no reads will be billed)");
        return Arc::new(FixtureSource::new(&config.fixtures_dir));
    }

    let Some(token) = config.x_bearer_token.clone() else {
        warn!(
            "AUGURY_X_LIVE=1 but X_BEARER_TOKEN is unset or still the template placeholder — \
             falling back to fixture replay"
        );
        return Arc::new(FixtureSource::new(&config.fixtures_dir));
    };

    let Some(pool) = pool else {
        warn!("live mode needs a database for the read budget — falling back to fixtures");
        return Arc::new(FixtureSource::new(&config.fixtures_dir));
    };

    let budget = ReadBudget::new(pool.clone(), config.max_daily_reads);
    match LiveSource::new(token, budget) {
        Ok(source) => {
            warn!(
                "post source: LIVE X API — up to {} reads today (~${:.2})",
                config.max_daily_reads,
                config.max_daily_reads as f64 * budget::COST_PER_READ_USD
            );
            Arc::new(source)
        }
        Err(err) => {
            warn!("could not build the live source ({err}) — falling back to fixtures");
            Arc::new(FixtureSource::new(&config.fixtures_dir))
        }
    }
}

async fn status(config: &Config) -> Result<()> {
    println!("augury-ingest status");
    println!("  database   : {}", db::redact(&config.database_url));
    println!(
        "  X mode     : {}",
        if config.live_enabled() {
            "LIVE (reads cost money)"
        } else if config.x_live {
            "live requested but no token — will replay fixtures"
        } else {
            "fixture replay"
        }
    );
    println!("  daily cap  : {} reads", config.max_daily_reads);
    println!("  fixtures   : {}", config.fixtures_dir.display());
    println!("  poll every : {}s", config.poll_interval_secs);

    match db::connect(&config.database_url, 2).await {
        Ok(pool) => {
            match db::check_schema(&pool).await {
                Ok(versions) => println!("  migrations : {}", versions.join(", ")),
                Err(err) => println!("  migrations : NOT APPLIED ({err})"),
            }
            match db::tracked_markets(&pool).await {
                Ok(markets) => {
                    println!("  markets    : {}", markets.len());
                    for market in &markets {
                        println!("      {} — {}", market.market_id, market.title);
                    }
                }
                Err(err) => println!("  markets    : unavailable ({err})"),
            }
            let budget = ReadBudget::new(pool.clone(), config.max_daily_reads);
            if let Ok(summary) = budget.summary().await {
                println!("  budget     : {summary}");
            }
            pool.close().await;
        }
        Err(err) => {
            println!("  database   : UNREACHABLE — {err}");
            println!("  hint       : `make up && make db-migrate`");
        }
    }

    Ok(())
}

/// Replay fixtures through the filter without touching the database.
///
/// The fastest way to see the near-duplicate detection working: the Fed fixture
/// contains a deliberate three-post spam ring and a promotional post.
async fn dry_run(config: &Config, market_id: &str) -> Result<()> {
    let source = FixtureSource::new(&config.fixtures_dir);
    let posts = source
        .search("", market_id, config.max_posts_per_poll, None)
        .await?;

    if posts.is_empty() {
        println!("no fixture posts for {market_id}");
        return Ok(());
    }

    let mut post_filter = PostFilter::new(FilterConfig::default());
    let filtered = post_filter.apply_batch(posts);

    println!("filter dry run — {market_id}\n");
    for post in &filtered {
        let marker = if post.filter_verdict.is_accepted() { " " } else { "x" };
        let text: String = post.text.chars().take(64).collect();
        println!(
            "  [{marker}] {:<12} {:<11} {}",
            post.post_id,
            post.filter_verdict.as_str(),
            text
        );
        if let Some(reason) = &post.filter_reason {
            println!("               reason: {reason}");
        }
    }

    let stats = post_filter.stats;
    println!(
        "\n  {} posts: {} accepted, {} duplicate, {} bot, {} low quality",
        stats.total(),
        stats.accepted,
        stats.duplicate,
        stats.bot,
        stats.low_quality
    );
    println!("\n  Rejected posts are still written to the database with their reason,");
    println!("  so the filter's own precision stays measurable after the fact.");
    Ok(())
}

/// One polling pass over every tracked market.
async fn poll_all(config: &Config) -> Result<()> {
    let pool = db::connect(&config.database_url, 8).await?;
    db::check_schema(&pool)
        .await
        .context("schema check failed; run `make db-migrate`")?;

    let markets = db::tracked_markets(&pool).await?;
    if markets.is_empty() {
        warn!("no tracked markets — run `augury sync-config` to mirror markets.yaml");
        return Ok(());
    }

    let source = build_source(config, Some(&pool));
    let start_time = Utc::now() - ChronoDuration::days(config.lookback_days);

    for market in markets {
        let query = match market.query.to_x_query(&config.language, &[]) {
            Ok(q) => q,
            Err(err) => {
                warn!("{}: {err}", market.market_id);
                continue;
            }
        };

        let fetched = match source
            .search(
                &query,
                &market.market_id,
                config.max_posts_per_poll,
                Some(start_time),
            )
            .await
        {
            Ok(posts) => posts,
            Err(err) => {
                // Budget exhaustion and transient API faults both land here.
                // Neither should stop the other markets from being polled.
                error!("{}: fetch failed — {err:#}", market.market_id);
                continue;
            }
        };

        if fetched.is_empty() {
            info!("{}: no posts returned", market.market_id);
            continue;
        }

        let fetched_count = fetched.len();
        let mut post_filter = PostFilter::new(FilterConfig::default());
        let filtered = post_filter.apply_batch(fetched);
        let stats = post_filter.stats;

        match db::insert_posts(&pool, &filtered).await {
            Ok(written) => info!(
                "{}: fetched {fetched_count}, stored {written} new \
                 ({} accepted, {} duplicate, {} bot, {} low quality)",
                market.market_id, stats.accepted, stats.duplicate, stats.bot, stats.low_quality
            ),
            Err(err) => error!("{}: insert failed — {err:#}", market.market_id),
        }
    }

    pool.close().await;
    Ok(())
}

/// Poll on a fixed interval until interrupted.
async fn run_forever(config: &Config, interval_secs: u64) -> Result<()> {
    let mut ticker = interval(Duration::from_secs(interval_secs.max(30)));
    info!("polling every {interval_secs}s; Ctrl-C to stop");

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                if let Err(err) = poll_all(config).await {
                    // Never fatal: the database may be restarting, the venue may
                    // be down. Log and try again next tick.
                    error!("poll cycle failed — {err:#}");
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("shutting down");
                return Ok(());
            }
        }
    }
}

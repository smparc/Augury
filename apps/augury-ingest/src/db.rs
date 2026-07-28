//! TimescaleDB sink and market-query source.
//!
//! Uses runtime-checked queries (`sqlx::query`) rather than the compile-time
//! `query!` macros, so the crate builds without a live database to introspect.
//! That matters for CI and for a fresh checkout where Docker is not up yet.
//!
//! Market query terms are read from `market_queries` rather than parsed from
//! `config/markets.yaml`: the Python `sync-config` command is what mirrors the
//! YAML into the table, and having exactly one parser for that file avoids two
//! services drifting apart in how they interpret it.

use anyhow::{Context, Result};
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{PgPool, Row};

use crate::models::{MarketQuery, Post};

/// Connect with a small pool sized for a handful of per-market tasks.
pub async fn connect(database_url: &str, max_connections: u32) -> Result<PgPool> {
    PgPoolOptions::new()
        .max_connections(max_connections)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect(database_url)
        .await
        .with_context(|| {
            format!(
                "connecting to Postgres (is `make up && make db-migrate` done?): {}",
                redact(database_url)
            )
        })
}

/// Hide the password before a DSN reaches a log line.
pub fn redact(url: &str) -> String {
    match (url.find("://"), url.rfind('@')) {
        (Some(scheme_end), Some(at)) if at > scheme_end => {
            format!("{}://***@{}", &url[..scheme_end], &url[at + 1..])
        }
        _ => url.to_string(),
    }
}

/// A market to ingest for, with its query terms.
#[derive(Debug, Clone)]
pub struct TrackedMarket {
    pub market_id: String,
    pub ticker: String,
    pub title: String,
    pub query: MarketQuery,
}

/// Load every tracked, unsettled market and its query terms.
///
/// Settled markets are excluded: paying to read posts about a question that has
/// already been answered is pure waste.
pub async fn tracked_markets(pool: &PgPool) -> Result<Vec<TrackedMarket>> {
    let market_rows = sqlx::query(
        r#"
        SELECT market_id, ticker, title
          FROM markets
         WHERE tracked AND status <> 'settled'
         ORDER BY market_id
        "#,
    )
    .fetch_all(pool)
    .await
    .context("loading tracked markets")?;

    let mut markets = Vec::with_capacity(market_rows.len());

    for row in market_rows {
        let market_id: String = row.get("market_id");

        let term_rows = sqlx::query(
            "SELECT term_type, term FROM market_queries WHERE market_id = $1 ORDER BY term_type, term",
        )
        .bind(&market_id)
        .fetch_all(pool)
        .await
        .with_context(|| format!("loading query terms for {market_id}"))?;

        let mut query = MarketQuery {
            market_id: market_id.clone(),
            ..Default::default()
        };

        for term_row in term_rows {
            let term_type: String = term_row.get("term_type");
            let term: String = term_row.get("term");
            match term_type.as_str() {
                "keyword" => query.keywords.push(term),
                "ticker" => query.tickers.push(term),
                "entity" => query.entities.push(term),
                "alias" => query.aliases.push(term),
                "exclude" => query.exclude.push(term),
                other => tracing::warn!("ignoring unknown term_type {other:?} for {market_id}"),
            }
        }

        if query.all_terms().is_empty() {
            tracing::warn!("{market_id} has no query terms; skipping (run `augury sync-config`)");
            continue;
        }

        markets.push(TrackedMarket {
            market_id,
            ticker: row.get("ticker"),
            title: row.get("title"),
            query,
        });
    }

    Ok(markets)
}

/// Insert posts, skipping any already stored.
///
/// Returns the number of rows actually written. Fetched-minus-written is the
/// number that tells you the polling interval is tighter than the rate at which
/// new posts appear.
pub async fn insert_posts(pool: &PgPool, posts: &[Post]) -> Result<u64> {
    if posts.is_empty() {
        return Ok(0);
    }

    let mut written = 0u64;
    let mut tx = pool.begin().await.context("beginning post insert")?;

    for post in posts {
        let result = sqlx::query(
            r#"
            INSERT INTO posts (post_id, market_id, created_at, ingested_at, author_id,
                               author_created_at, followers, engagements, lang, text,
                               minhash, lsh_bucket, filter_verdict, filter_reason, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (post_id, market_id, created_at) DO NOTHING
            "#,
        )
        .bind(&post.post_id)
        .bind(&post.market_id)
        .bind(post.created_at)
        .bind(post.ingested_at.unwrap_or_else(chrono::Utc::now))
        .bind(&post.author_id)
        .bind(post.author_created_at)
        .bind(post.followers)
        .bind(post.engagements)
        .bind(&post.lang)
        .bind(&post.text)
        .bind(post.minhash.as_deref())
        .bind(&post.lsh_bucket)
        .bind(post.filter_verdict.as_str())
        .bind(&post.filter_reason)
        .bind(&post.source)
        .execute(&mut *tx)
        .await
        .with_context(|| format!("inserting post {}", post.post_id))?;

        written += result.rows_affected();
    }

    tx.commit().await.context("committing post insert")?;
    Ok(written)
}

/// Post ids already stored for a market, so a re-poll can skip re-filtering.
pub async fn known_post_ids(pool: &PgPool, market_id: &str, limit: i64) -> Result<Vec<String>> {
    let rows: Vec<PgRow> = sqlx::query(
        r#"
        SELECT post_id FROM posts
         WHERE market_id = $1
         ORDER BY created_at DESC
         LIMIT $2
        "#,
    )
    .bind(market_id)
    .bind(limit)
    .fetch_all(pool)
    .await
    .context("loading known post ids")?;

    Ok(rows.into_iter().map(|r| r.get("post_id")).collect())
}

/// Counts by filter verdict, for the ingest summary log.
pub async fn verdict_counts(pool: &PgPool, market_id: &str) -> Result<Vec<(String, i64)>> {
    let rows = sqlx::query(
        r#"
        SELECT filter_verdict, count(*) AS n
          FROM posts
         WHERE market_id = $1
         GROUP BY filter_verdict
         ORDER BY n DESC
        "#,
    )
    .bind(market_id)
    .fetch_all(pool)
    .await
    .context("counting verdicts")?;

    Ok(rows
        .into_iter()
        .map(|r| (r.get("filter_verdict"), r.get("n")))
        .collect())
}

/// Verify the schema is present before doing any work.
pub async fn check_schema(pool: &PgPool) -> Result<Vec<String>> {
    let rows = sqlx::query("SELECT version FROM schema_migrations ORDER BY version")
        .fetch_all(pool)
        .await
        .context("reading schema_migrations (has `make db-migrate` been run?)")?;

    Ok(rows.into_iter().map(|r| r.get("version")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_hides_the_password() {
        assert_eq!(
            redact("postgresql://augury:hunter2@localhost:5432/augury"),
            "postgresql://***@localhost:5432/augury"
        );
    }

    #[test]
    fn redact_passes_through_urls_without_credentials() {
        assert_eq!(
            redact("postgresql://localhost:5432/augury"),
            "postgresql://localhost:5432/augury"
        );
    }
}

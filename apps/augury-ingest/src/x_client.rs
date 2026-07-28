//! X (Twitter) post sources.
//!
//! Two backends behind one trait. [`FixtureSource`] replays JSONL from disk and
//! is the default; [`LiveSource`] calls the real API and is reachable only when
//! `AUGURY_X_LIVE=1` *and* a bearer token is present. Going live is therefore a
//! deliberate act rather than something a misconfigured environment does by
//! accident.
//!
//! Posts from the fixture backend carry `source = "fixture"` all the way into
//! the database, so a number computed from replayed data can never be mistaken
//! for a real measurement.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::Deserialize;

use crate::budget::ReadBudget;
use crate::models::{FilterVerdict, Post};

/// X caps a single recent-search page at 100 results.
pub const MAX_PAGE_SIZE: usize = 100;

const SEARCH_URL: &str = "https://api.x.com/2/tweets/search/recent";

#[async_trait::async_trait]
pub trait PostSource: Send + Sync {
    fn source_name(&self) -> &'static str;

    async fn search(
        &self,
        query: &str,
        market_id: &str,
        max_results: usize,
        start_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<Post>>;
}

// ---------------------------------------------------------------------------
// Fixture replay
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureRecord {
    #[serde(alias = "id")]
    post_id: String,
    created_at: DateTime<Utc>,
    text: String,
    #[serde(default)]
    author_id: Option<String>,
    #[serde(default)]
    author_created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    followers: i32,
    #[serde(default)]
    engagements: i32,
    #[serde(default)]
    lang: Option<String>,
}

/// Replays recorded posts from `fixtures/*.jsonl`.
#[derive(Debug, Clone)]
pub struct FixtureSource {
    dir: PathBuf,
    /// Shift replayed timestamps so the newest post lands at "now", preserving
    /// relative spacing. Without it a fixture recorded weeks ago has already
    /// decayed to nothing by the time `S(t)` is evaluated, and every run looks
    /// like an empty signal.
    anchor_to_now: bool,
}

impl FixtureSource {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            anchor_to_now: true,
        }
    }

    pub fn without_anchoring(mut self) -> Self {
        self.anchor_to_now = false;
        self
    }

    fn path_for(&self, market_id: &str) -> Option<PathBuf> {
        let safe = market_id.replace([':', '/'], "_");
        let specific = self.dir.join(format!("{safe}.jsonl"));
        if specific.exists() {
            return Some(specific);
        }
        let fallback = self.dir.join("default.jsonl");
        fallback.exists().then_some(fallback)
    }
}

#[async_trait::async_trait]
impl PostSource for FixtureSource {
    fn source_name(&self) -> &'static str {
        "fixture"
    }

    async fn search(
        &self,
        _query: &str,
        market_id: &str,
        max_results: usize,
        start_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<Post>> {
        let Some(path) = self.path_for(market_id) else {
            return Ok(Vec::new());
        };

        let contents = std::fs::read_to_string(&path)
            .with_context(|| format!("reading fixture {}", path.display()))?;

        let mut records = Vec::new();
        for (index, line) in contents.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            let record: FixtureRecord = serde_json::from_str(trimmed)
                .with_context(|| format!("{}:{}: invalid JSON", path.display(), index + 1))?;
            records.push(record);
        }

        if records.is_empty() {
            return Ok(Vec::new());
        }

        let offset = if self.anchor_to_now {
            let newest = records
                .iter()
                .map(|r| r.created_at)
                .max()
                .expect("records is non-empty");
            Utc::now() - newest
        } else {
            Duration::zero()
        };

        let now = Utc::now();
        let mut posts: Vec<Post> = records
            .into_iter()
            .map(|r| {
                let created_at = r.created_at + offset;
                Post {
                    post_id: r.post_id,
                    market_id: market_id.to_string(),
                    created_at,
                    ingested_at: Some(now),
                    author_id: r.author_id,
                    author_created_at: r.author_created_at,
                    followers: r.followers,
                    engagements: r.engagements,
                    lang: r.lang,
                    text: r.text,
                    minhash: None,
                    lsh_bucket: None,
                    filter_verdict: FilterVerdict::Accepted,
                    filter_reason: None,
                    source: "fixture".to_string(),
                }
            })
            .filter(|p| start_time.map_or(true, |start| p.created_at >= start))
            .collect();

        // Oldest first: the duplicate filter keeps whichever member of a
        // cluster it sees first, and that should be the original.
        posts.sort_by_key(|p| p.created_at);
        if posts.len() > max_results {
            posts.drain(..posts.len() - max_results);
        }
        Ok(posts)
    }
}

// ---------------------------------------------------------------------------
// Live API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SearchResponse {
    #[serde(default)]
    data: Vec<RawTweet>,
    #[serde(default)]
    includes: Includes,
}

#[derive(Debug, Default, Deserialize)]
struct Includes {
    #[serde(default)]
    users: Vec<RawUser>,
}

#[derive(Debug, Deserialize)]
struct RawTweet {
    id: String,
    text: String,
    created_at: DateTime<Utc>,
    #[serde(default)]
    author_id: Option<String>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    public_metrics: TweetMetrics,
}

#[derive(Debug, Default, Deserialize)]
struct TweetMetrics {
    #[serde(default)]
    like_count: i32,
    #[serde(default)]
    retweet_count: i32,
    #[serde(default)]
    reply_count: i32,
    #[serde(default)]
    quote_count: i32,
}

impl TweetMetrics {
    fn total(&self) -> i32 {
        self.like_count + self.retweet_count + self.reply_count + self.quote_count
    }
}

#[derive(Debug, Deserialize)]
struct RawUser {
    id: String,
    #[serde(default)]
    created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    public_metrics: UserMetrics,
}

#[derive(Debug, Default, Deserialize)]
struct UserMetrics {
    #[serde(default)]
    followers_count: i32,
}

pub struct LiveSource {
    client: reqwest::Client,
    bearer_token: String,
    budget: ReadBudget,
}

impl LiveSource {
    pub fn new(bearer_token: String, budget: ReadBudget) -> Result<Self> {
        anyhow::ensure!(!bearer_token.is_empty(), "X_BEARER_TOKEN is empty");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .user_agent("augury-ingest/0.2")
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            client,
            bearer_token,
            budget,
        })
    }
}

#[async_trait::async_trait]
impl PostSource for LiveSource {
    fn source_name(&self) -> &'static str {
        "x"
    }

    async fn search(
        &self,
        query: &str,
        market_id: &str,
        max_results: usize,
        start_time: Option<DateTime<Utc>>,
    ) -> Result<Vec<Post>> {
        let page_size = max_results.clamp(10, MAX_PAGE_SIZE) as i32;

        // Reserved before the request, because a request that the server
        // counted but that failed on our side must still be paid for.
        self.budget.reserve(page_size).await?;

        let mut params: Vec<(&str, String)> = vec![
            ("query", query.to_string()),
            ("max_results", page_size.to_string()),
            ("tweet.fields", "created_at,public_metrics,lang,author_id".into()),
            ("expansions", "author_id".into()),
            ("user.fields", "public_metrics,created_at".into()),
        ];
        if let Some(start) = start_time {
            params.push(("start_time", start.format("%Y-%m-%dT%H:%M:%SZ").to_string()));
        }

        let response = self
            .client
            .get(SEARCH_URL)
            .bearer_auth(&self.bearer_token)
            .query(&params)
            .send()
            .await
            .context("X search request failed")?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // The reservation stands: a rejected request may still have been
            // metered upstream, and over-counting is the safe direction.
            anyhow::bail!("X search returned {status}: {body}");
        }

        let payload: SearchResponse = response.json().await.context("decoding X response")?;
        let returned = payload.data.len() as i32;
        self.budget.release_unused(page_size, returned).await?;

        let now = Utc::now();
        let posts = payload
            .data
            .into_iter()
            .map(|tweet| {
                let author = tweet
                    .author_id
                    .as_ref()
                    .and_then(|id| payload.includes.users.iter().find(|u| &u.id == id));

                Post {
                    post_id: tweet.id,
                    market_id: market_id.to_string(),
                    created_at: tweet.created_at,
                    ingested_at: Some(now),
                    author_id: tweet.author_id,
                    author_created_at: author.and_then(|u| u.created_at),
                    followers: author.map(|u| u.public_metrics.followers_count).unwrap_or(0),
                    engagements: tweet.public_metrics.total(),
                    lang: tweet.lang,
                    text: tweet.text,
                    minhash: None,
                    lsh_bucket: None,
                    filter_verdict: FilterVerdict::Accepted,
                    filter_reason: None,
                    source: "x".to_string(),
                }
            })
            .collect::<Vec<_>>();

        Ok(posts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("augury-signal")
            .join("fixtures")
    }

    #[tokio::test]
    async fn replays_the_fed_fixture() {
        let source = FixtureSource::new(fixtures_dir());
        let posts = source
            .search("ignored", "kalshi:KXFEDDECISION-26JUL-C25", 100, None)
            .await
            .unwrap();

        assert!(posts.len() >= 20, "got {} posts", posts.len());
        assert!(posts.iter().all(|p| p.source == "fixture"));
        assert!(posts
            .iter()
            .all(|p| p.market_id == "kalshi:KXFEDDECISION-26JUL-C25"));
    }

    #[tokio::test]
    async fn posts_are_oldest_first() {
        let source = FixtureSource::new(fixtures_dir());
        let posts = source
            .search("q", "kalshi:KXFEDDECISION-26JUL-C25", 100, None)
            .await
            .unwrap();
        for pair in posts.windows(2) {
            assert!(pair[0].created_at <= pair[1].created_at);
        }
    }

    #[tokio::test]
    async fn anchoring_keeps_posts_recent() {
        let source = FixtureSource::new(fixtures_dir());
        let posts = source
            .search("q", "kalshi:KXFEDDECISION-26JUL-C25", 100, None)
            .await
            .unwrap();
        let newest = posts.iter().map(|p| p.created_at).max().unwrap();
        assert!((Utc::now() - newest).num_minutes() < 2);
    }

    #[tokio::test]
    async fn unknown_market_falls_back_to_default() {
        let source = FixtureSource::new(fixtures_dir());
        let posts = source
            .search("q", "kalshi:SOMETHING-ELSE", 100, None)
            .await
            .unwrap();
        assert!(!posts.is_empty(), "default.jsonl should serve this");
    }

    #[tokio::test]
    async fn missing_fixture_dir_returns_empty() {
        let source = FixtureSource::new(PathBuf::from("/nonexistent/fixtures"));
        let posts = source.search("q", "kalshi:X", 10, None).await.unwrap();
        assert!(posts.is_empty());
    }

    #[tokio::test]
    async fn max_results_keeps_the_newest() {
        let source = FixtureSource::new(fixtures_dir());
        let all = source
            .search("q", "kalshi:KXFEDDECISION-26JUL-C25", 100, None)
            .await
            .unwrap();
        let limited = source
            .search("q", "kalshi:KXFEDDECISION-26JUL-C25", 5, None)
            .await
            .unwrap();
        assert_eq!(limited.len(), 5);
        assert_eq!(limited.last().unwrap().post_id, all.last().unwrap().post_id);
    }
}

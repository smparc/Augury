//! Records shared with the rest of the system.
//!
//! These mirror `schemas/post.schema.json` and the `posts` table. Field names
//! match the JSON Schema exactly so a serialized `Post` validates against it
//! without a translation layer.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Why a post was or was not accepted into the signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterVerdict {
    Accepted,
    Duplicate,
    Bot,
    LowQuality,
    OffTopic,
}

impl FilterVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            FilterVerdict::Accepted => "accepted",
            FilterVerdict::Duplicate => "duplicate",
            FilterVerdict::Bot => "bot",
            FilterVerdict::LowQuality => "low_quality",
            FilterVerdict::OffTopic => "off_topic",
        }
    }

    pub fn is_accepted(&self) -> bool {
        matches!(self, FilterVerdict::Accepted)
    }
}

/// One ingested post, bound to exactly one market.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Post {
    pub post_id: String,
    pub market_id: String,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ingested_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub followers: i32,
    #[serde(default)]
    pub engagements: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minhash: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lsh_bucket: Option<String>,
    pub filter_verdict: FilterVerdict,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_reason: Option<String>,
    pub source: String,
}

impl Post {
    /// Age of the author's account at the time of posting.
    ///
    /// `None` when the account creation date is unknown, which must be treated
    /// as "unknown", never as "old enough" — a missing field is exactly what a
    /// scraped-together bot account looks like.
    pub fn account_age_days(&self) -> Option<i64> {
        self.author_created_at
            .map(|created| (self.created_at - created).num_days())
    }
}

/// The terms binding posts to a market, mirrored from `market_queries`.
#[derive(Debug, Clone, Default)]
pub struct MarketQuery {
    pub market_id: String,
    pub keywords: Vec<String>,
    pub tickers: Vec<String>,
    pub entities: Vec<String>,
    pub aliases: Vec<String>,
    pub exclude: Vec<String>,
}

impl MarketQuery {
    /// Every positive term, deduplicated, preserving insertion order.
    pub fn all_terms(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for term in self
            .keywords
            .iter()
            .chain(&self.tickers)
            .chain(&self.entities)
            .chain(&self.aliases)
        {
            if !seen.contains(&term.as_str()) {
                seen.push(term.as_str());
            }
        }
        seen
    }

    /// Build an X recent-search query.
    ///
    /// Plain keywords only. The endpoint silently ignores the rich operators
    /// from the x.com search UI (`min_faves:`, `since:`, `filter:`) — they do
    /// not error, they simply have no effect — so quality filtering happens in
    /// [`crate::filter`] rather than in the query string.
    pub fn to_x_query(&self, lang: &str, extra_exclude: &[String]) -> anyhow::Result<String> {
        let terms = self.all_terms();
        if terms.is_empty() {
            anyhow::bail!("market {} has no positive query terms", self.market_id);
        }

        let quote = |term: &str| -> String {
            if term.contains(' ') {
                format!("\"{term}\"")
            } else {
                term.to_string()
            }
        };

        let clause = terms
            .iter()
            .map(|t| quote(t))
            .collect::<Vec<_>>()
            .join(" OR ");

        let mut parts = vec![format!("({clause})")];
        for term in self.exclude.iter().chain(extra_exclude) {
            parts.push(format!("-{}", quote(term)));
        }
        parts.push("-is:retweet".to_string());
        if !lang.is_empty() {
            parts.push(format!("lang:{lang}"));
        }

        Ok(parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn query() -> MarketQuery {
        MarketQuery {
            market_id: "kalshi:TEST".into(),
            keywords: vec!["rate cut".into(), "FOMC".into()],
            aliases: vec!["FOMC".into()],
            exclude: vec!["fed up".into()],
            ..Default::default()
        }
    }

    #[test]
    fn query_deduplicates_terms() {
        assert_eq!(query().all_terms(), vec!["rate cut", "FOMC"]);
    }

    #[test]
    fn query_quotes_phrases_and_negates_exclusions() {
        let built = query().to_x_query("en", &["giveaway".into()]).unwrap();
        assert!(built.contains("\"rate cut\""));
        assert!(built.contains("-\"fed up\""));
        assert!(built.contains("-giveaway"));
        assert!(built.contains("-is:retweet"));
        assert!(built.contains("lang:en"));
    }

    #[test]
    fn empty_query_is_rejected() {
        let empty = MarketQuery {
            market_id: "kalshi:EMPTY".into(),
            ..Default::default()
        };
        assert!(empty.to_x_query("en", &[]).is_err());
    }

    #[test]
    fn account_age_is_none_when_unknown() {
        let post = Post {
            post_id: "1".into(),
            market_id: "kalshi:T".into(),
            created_at: Utc.with_ymd_and_hms(2026, 7, 27, 0, 0, 0).unwrap(),
            ingested_at: None,
            author_id: None,
            author_created_at: None,
            followers: 0,
            engagements: 0,
            lang: None,
            text: "x".into(),
            minhash: None,
            lsh_bucket: None,
            filter_verdict: FilterVerdict::Accepted,
            filter_reason: None,
            source: "fixture".into(),
        };
        assert_eq!(post.account_age_days(), None);

        let aged = Post {
            author_created_at: Some(Utc.with_ymd_and_hms(2026, 7, 20, 0, 0, 0).unwrap()),
            ..post
        };
        assert_eq!(aged.account_age_days(), Some(7));
    }
}

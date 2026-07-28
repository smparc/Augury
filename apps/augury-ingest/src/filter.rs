//! Bot, spam, and near-duplicate filtering (README enhancement 2).
//!
//! Runs before anything is written to storage so a spam ring cannot distort
//! `S(t)`. Rejected posts are *stored with a reason*, never dropped: without
//! the rejects there is no way to measure how often the filter is wrong, and a
//! filter nobody can audit is a filter nobody should trust.
//!
//! Ordering matters. Cheap field checks run before the MinHash comparison, so
//! obvious junk never reaches the more expensive path — but every post is still
//! fed to the duplicate detector afterward, because a spam ring whose members
//! are individually plausible is exactly what the detector exists to catch.

use crate::minhash::{DuplicateCheck, DuplicateDetector};
use crate::models::{FilterVerdict, Post};

/// Thresholds for the heuristic checks.
///
/// These are heuristics, not truths. They are deliberately loose: wrongly
/// dropping a real opinion silently biases the signal, while letting a little
/// junk through only adds noise that the follower weighting already discounts.
#[derive(Debug, Clone)]
pub struct FilterConfig {
    /// Accounts younger than this are treated as unproven.
    pub min_account_age_days: i64,
    /// A young account also needs at least this many followers to count.
    pub min_followers_for_young_account: i32,
    /// Posts shorter than this carry no stance worth scoring.
    pub min_text_length: usize,
    /// Above this fraction of non-alphanumeric characters, the post is noise.
    pub max_symbol_ratio: f64,
    /// Promotional markers that indicate paid-signal spam.
    pub spam_markers: Vec<String>,
    /// Jaccard similarity at or above which a post duplicates an earlier one.
    pub duplicate_threshold: f64,
    /// Posts remembered per market for duplicate comparison.
    pub window_capacity: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            // Two weeks: long enough to exclude accounts spun up for a single
            // event, short enough to keep genuine new users who joined to
            // discuss it.
            min_account_age_days: 14,
            min_followers_for_young_account: 50,
            min_text_length: 12,
            max_symbol_ratio: 0.6,
            spam_markers: vec![
                "promo code".into(),
                "giveaway".into(),
                "airdrop".into(),
                "free signals".into(),
                "guaranteed profits".into(),
                "join now".into(),
                "sign up bonus".into(),
                "dm me".into(),
            ],
            duplicate_threshold: crate::minhash::DEFAULT_DUPLICATE_THRESHOLD,
            window_capacity: crate::minhash::DEFAULT_WINDOW_CAPACITY,
        }
    }
}

/// Counts of what the filter did, for logging and later precision auditing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FilterStats {
    pub accepted: usize,
    pub duplicate: usize,
    pub bot: usize,
    pub low_quality: usize,
}

impl FilterStats {
    pub fn total(&self) -> usize {
        self.accepted + self.duplicate + self.bot + self.low_quality
    }

    pub fn record(&mut self, verdict: FilterVerdict) {
        match verdict {
            FilterVerdict::Accepted => self.accepted += 1,
            FilterVerdict::Duplicate => self.duplicate += 1,
            FilterVerdict::Bot => self.bot += 1,
            FilterVerdict::LowQuality | FilterVerdict::OffTopic => self.low_quality += 1,
        }
    }
}

/// Fraction of characters that are neither alphanumeric nor whitespace.
fn symbol_ratio(text: &str) -> f64 {
    let total = text.chars().count();
    if total == 0 {
        return 1.0;
    }
    let symbols = text
        .chars()
        .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
        .count();
    symbols as f64 / total as f64
}

/// The stateful filter. One instance per market, so the duplicate window is
/// scoped to posts about the same claim.
#[derive(Debug)]
pub struct PostFilter {
    config: FilterConfig,
    detector: DuplicateDetector,
    pub stats: FilterStats,
}

impl PostFilter {
    pub fn new(config: FilterConfig) -> Self {
        let detector = DuplicateDetector::new(
            crate::minhash::MinHasher::default(),
            config.duplicate_threshold,
            config.window_capacity,
        );
        Self {
            config,
            detector,
            stats: FilterStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(FilterConfig::default())
    }

    /// Cheap field checks. Returns a rejection reason, or `None` to continue.
    fn heuristic_reject(&self, post: &Post) -> Option<(FilterVerdict, String)> {
        let trimmed = post.text.trim();

        if trimmed.chars().count() < self.config.min_text_length {
            return Some((
                FilterVerdict::LowQuality,
                format!("text is {} chars, below the {}-char floor", trimmed.chars().count(), self.config.min_text_length),
            ));
        }

        let ratio = symbol_ratio(trimmed);
        if ratio > self.config.max_symbol_ratio {
            return Some((
                FilterVerdict::LowQuality,
                format!("symbol ratio {ratio:.2} exceeds {:.2}", self.config.max_symbol_ratio),
            ));
        }

        let lowered = trimmed.to_lowercase();
        if let Some(marker) = self
            .config
            .spam_markers
            .iter()
            .find(|m| lowered.contains(m.as_str()))
        {
            return Some((
                FilterVerdict::LowQuality,
                format!("contains promotional marker {marker:?}"),
            ));
        }

        // A young account is not disqualifying on its own — plenty of real
        // people join to discuss an event. A young account with almost no
        // followers is the pattern worth excluding.
        if let Some(age) = post.account_age_days() {
            if age < self.config.min_account_age_days
                && post.followers < self.config.min_followers_for_young_account
            {
                return Some((
                    FilterVerdict::Bot,
                    format!(
                        "account is {age}d old with {} followers (thresholds: {}d, {})",
                        post.followers,
                        self.config.min_account_age_days,
                        self.config.min_followers_for_young_account
                    ),
                ));
            }
        }

        None
    }

    /// Classify one post and update the duplicate window.
    ///
    /// Necessarily `&mut`: duplicate detection is stateful, and a variant that
    /// took `&self` could only work by throwing that state away — which would
    /// silently disable the filter's whole reason for existing.
    pub fn apply_mut(&mut self, mut post: Post) -> Post {
        // Always compute the signature, even for posts rejected on heuristics,
        // so a rejected post can still be compared against later.
        let (duplicate, signature) = self.detector.check(&post.post_id, &post.text);

        if let Some(sig) = signature {
            post.lsh_bucket = self.detector.hasher().primary_bucket(&sig);
            post.minhash = Some(sig);
        }

        let (verdict, reason) = match self.heuristic_reject(&post) {
            Some((verdict, reason)) => (verdict, Some(reason)),
            None => match duplicate {
                DuplicateCheck::Duplicate {
                    original_id,
                    similarity,
                } => (
                    FilterVerdict::Duplicate,
                    Some(format!(
                        "jaccard {similarity:.2} against post {original_id}"
                    )),
                ),
                DuplicateCheck::Unique => (FilterVerdict::Accepted, None),
            },
        };

        self.stats.record(verdict);
        post.filter_verdict = verdict;
        post.filter_reason = reason;
        post
    }

    /// Classify a batch in arrival order.
    ///
    /// Order matters: whichever member of a duplicate cluster arrives first is
    /// the one kept, so the batch should be sorted oldest-first to keep the
    /// original rather than an arbitrary copy.
    pub fn apply_batch(&mut self, posts: Vec<Post>) -> Vec<Post> {
        posts.into_iter().map(|p| self.apply_mut(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn post(id: &str, text: &str) -> Post {
        Post {
            post_id: id.into(),
            market_id: "kalshi:TEST".into(),
            created_at: Utc::now(),
            ingested_at: None,
            author_id: Some(format!("author_{id}")),
            author_created_at: Some(Utc::now() - Duration::days(900)),
            followers: 5_000,
            engagements: 40,
            lang: Some("en".into()),
            text: text.into(),
            minhash: None,
            lsh_bucket: None,
            filter_verdict: FilterVerdict::Accepted,
            filter_reason: None,
            source: "fixture".into(),
        }
    }

    #[test]
    fn accepts_an_ordinary_post() {
        let mut filter = PostFilter::with_defaults();
        let got = filter.apply_mut(post(
            "1",
            "The Fed looks set to cut rates by 25bps at the July meeting.",
        ));
        assert_eq!(got.filter_verdict, FilterVerdict::Accepted);
        assert!(got.filter_reason.is_none());
        assert!(got.minhash.is_some());
        assert!(got.lsh_bucket.is_some());
    }

    #[test]
    fn rejects_short_text() {
        let mut filter = PostFilter::with_defaults();
        let got = filter.apply_mut(post("1", "cut"));
        assert_eq!(got.filter_verdict, FilterVerdict::LowQuality);
    }

    #[test]
    fn rejects_promotional_spam() {
        let mut filter = PostFilter::with_defaults();
        let got = filter.apply_mut(post(
            "1",
            "FREE signals on the FOMC decision! promo code RATES50 join now",
        ));
        assert_eq!(got.filter_verdict, FilterVerdict::LowQuality);
        assert!(got.filter_reason.unwrap().contains("promo code"));
    }

    #[test]
    fn rejects_young_low_follower_accounts() {
        let mut filter = PostFilter::with_defaults();
        let mut p = post("1", "FOMC rate cut incoming, this is a real opinion honestly");
        p.author_created_at = Some(Utc::now() - Duration::days(3));
        p.followers = 12;
        let got = filter.apply_mut(p);
        assert_eq!(got.filter_verdict, FilterVerdict::Bot);
    }

    #[test]
    fn keeps_young_accounts_with_real_audiences() {
        // A new account with a following is a person, not a spam ring.
        let mut filter = PostFilter::with_defaults();
        let mut p = post("1", "FOMC rate cut incoming, this is a real opinion honestly");
        p.author_created_at = Some(Utc::now() - Duration::days(3));
        p.followers = 25_000;
        assert_eq!(filter.apply_mut(p).filter_verdict, FilterVerdict::Accepted);
    }

    #[test]
    fn unknown_account_age_is_not_treated_as_old() {
        let mut filter = PostFilter::with_defaults();
        let mut p = post("1", "A perfectly reasonable take about interest rates today");
        p.author_created_at = None;
        // Missing age cannot be checked, so the post passes on age but must not
        // be credited with having passed.
        let got = filter.apply_mut(p);
        assert_eq!(got.filter_verdict, FilterVerdict::Accepted);
        assert_eq!(got.account_age_days(), None);
    }

    #[test]
    fn flags_a_near_duplicate_ring() {
        let mut filter = PostFilter::with_defaults();
        let a = filter.apply_mut(post(
            "a",
            "Powell is cutting rates in July, the data leaves no other option",
        ));
        let b = filter.apply_mut(post(
            "b",
            "Powell is cutting rates in July, the data leaves no other option!",
        ));
        assert_eq!(a.filter_verdict, FilterVerdict::Accepted);
        assert_eq!(b.filter_verdict, FilterVerdict::Duplicate);
        assert!(b.filter_reason.unwrap().contains("jaccard"));
    }

    #[test]
    fn rejected_posts_keep_their_signature_for_auditing() {
        let mut filter = PostFilter::with_defaults();
        let got = filter.apply_mut(post(
            "1",
            "join now for guaranteed profits on every single fomc trade",
        ));
        assert!(!got.filter_verdict.is_accepted());
        assert!(got.minhash.is_some(), "signature must survive rejection");
    }

    #[test]
    fn stats_account_for_every_post() {
        let mut filter = PostFilter::with_defaults();
        filter.apply_batch(vec![
            post("1", "The Fed will cut rates at the July meeting, clearly"),
            post("2", "The Fed will cut rates at the July meeting, clearly!"),
            post("3", "short"),
            post("4", "Bitcoin is going to break one hundred fifty thousand"),
        ]);
        assert_eq!(filter.stats.total(), 4);
        assert_eq!(filter.stats.accepted, 2);
        assert_eq!(filter.stats.duplicate, 1);
        assert_eq!(filter.stats.low_quality, 1);
    }

    #[test]
    fn symbol_ratio_detects_noise() {
        assert!(symbol_ratio("!!!! ????? ***** ####") > 0.6);
        assert!(symbol_ratio("the fed will cut rates") < 0.1);
    }
}

//! Streaming MinHash near-duplicate detection (README enhancement 2).
//!
//! Spam rings post the same claim from dozens of accounts with small
//! variations — a changed emoji, a reordered clause, a different cashtag. Exact
//! hashing misses all of it, and every copy that gets through is counted again
//! by `S(t)`, so a ring with fifty accounts can move the signal as much as fifty
//! genuine opinions.
//!
//! MinHash estimates the Jaccard similarity of two documents' shingle sets
//! without storing the sets. For a random permutation of the shingle universe,
//! the probability that two sets share a minimum element is exactly their
//! Jaccard similarity; averaging that indicator over `num_hashes` independent
//! permutations gives an unbiased estimate. Comparing two signatures is then a
//! fixed-size integer comparison rather than a set intersection.
//!
//! Comparing every new post against every stored post is quadratic, so LSH does
//! the pre-filtering: the signature is split into bands, each band is hashed to
//! a bucket, and two posts are compared only if they collide in at least one
//! band. Two documents with Jaccard `s` collide in a given band with
//! probability `s^rows`, so the chance of being considered at all is
//! `1 - (1 - s^rows)^bands` — a sigmoid in `s` whose steep region is tuned by
//! the band/row split.

use std::collections::VecDeque;

/// Mersenne prime (2^61 - 1). Modulus for the universal hash family; large
/// enough that collisions between distinct shingles are negligible and small
/// enough that the multiply-add stays inside u128 without overflow.
const MERSENNE_PRIME: u64 = (1 << 61) - 1;

/// Word-level shingle width. Three words is the usual choice for short social
/// text: single words collide across unrelated posts, and five-word shingles
/// make near-duplicates with any edit look distinct.
pub const SHINGLE_SIZE: usize = 3;

/// Signature length. 128 permutations put the standard error of the Jaccard
/// estimate near 1/sqrt(128) ~= 0.088, which is precise enough to separate a
/// 0.9-similar reword from a 0.3-similar coincidence.
pub const DEFAULT_NUM_HASHES: usize = 128;

/// LSH band count. With 128 hashes this gives 16 rows per band, so the
/// collision probability crosses 50% around Jaccard 0.80 — near-duplicates get
/// compared, genuinely different posts almost never do.
pub const DEFAULT_NUM_BANDS: usize = 8;

/// Similarity at or above which two posts are considered the same claim.
pub const DEFAULT_DUPLICATE_THRESHOLD: f64 = 0.80;

/// A 64-bit FNV-1a hash. Deterministic across runs and platforms, which matters
/// because signatures are persisted to the database and compared later.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Normalize text before shingling.
///
/// URLs, mentions, and cashtags are stripped rather than kept: a spam ring
/// rotates its links and handles while reusing the body, so keeping them would
/// make otherwise identical posts look distinct — the exact evasion this filter
/// exists to catch.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for raw_token in text.split_whitespace() {
        let lowered = raw_token.to_lowercase();
        if lowered.starts_with("http://")
            || lowered.starts_with("https://")
            || lowered.starts_with('@')
            || lowered.starts_with('$')
            || lowered.starts_with('#')
        {
            continue;
        }
        let cleaned: String = lowered
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'')
            .collect();
        if cleaned.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&cleaned);
    }
    out
}

/// Word-level shingles of the normalized text.
///
/// Text shorter than one shingle yields a single shingle of the whole thing, so
/// very short posts still compare against each other instead of silently
/// producing an empty set that matches nothing.
pub fn shingles(text: &str) -> Vec<String> {
    let normalized = normalize(text);
    let words: Vec<&str> = normalized.split_whitespace().collect();

    if words.is_empty() {
        return Vec::new();
    }
    if words.len() < SHINGLE_SIZE {
        return vec![words.join(" ")];
    }

    words
        .windows(SHINGLE_SIZE)
        .map(|window| window.join(" "))
        .collect()
}

/// A family of `num_hashes` pairwise-independent hash functions of the form
/// `h_i(x) = (a_i * x + b_i) mod p`.
#[derive(Debug, Clone)]
pub struct MinHasher {
    coefficients: Vec<(u64, u64)>,
    num_bands: usize,
}

impl Default for MinHasher {
    fn default() -> Self {
        Self::new(DEFAULT_NUM_HASHES, DEFAULT_NUM_BANDS)
    }
}

impl MinHasher {
    /// Build a hasher. `num_hashes` must divide evenly into `num_bands`.
    ///
    /// The coefficients come from a fixed seed rather than an RNG: signatures
    /// are stored in the database and compared against posts hashed by a later
    /// process run, so the permutations must be identical across restarts and
    /// across machines.
    pub fn new(num_hashes: usize, num_bands: usize) -> Self {
        assert!(num_hashes > 0, "num_hashes must be positive");
        assert!(num_bands > 0, "num_bands must be positive");
        assert!(
            num_hashes % num_bands == 0,
            "num_hashes ({num_hashes}) must be divisible by num_bands ({num_bands})"
        );

        let mut coefficients = Vec::with_capacity(num_hashes);
        // Deterministic splitmix64 stream seeded by a constant.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || -> u64 {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        for _ in 0..num_hashes {
            // `a` must be non-zero for the map to be a permutation.
            let a = (next() % (MERSENNE_PRIME - 1)) + 1;
            let b = next() % MERSENNE_PRIME;
            coefficients.push((a, b));
        }

        Self {
            coefficients,
            num_bands,
        }
    }

    pub fn num_hashes(&self) -> usize {
        self.coefficients.len()
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    pub fn rows_per_band(&self) -> usize {
        self.num_hashes() / self.num_bands
    }

    /// Compute the MinHash signature of a document.
    ///
    /// Returns `None` for text that yields no shingles at all (empty, or only
    /// URLs and handles). Such a post cannot be compared to anything, and
    /// treating it as a duplicate of the last empty post would be wrong.
    pub fn signature(&self, text: &str) -> Option<Vec<i64>> {
        let shingle_set = shingles(text);
        if shingle_set.is_empty() {
            return None;
        }

        let hashed: Vec<u64> = shingle_set
            .iter()
            .map(|s| fnv1a(s.as_bytes()) % MERSENNE_PRIME)
            .collect();

        let signature = self
            .coefficients
            .iter()
            .map(|(a, b)| {
                hashed
                    .iter()
                    .map(|h| {
                        let product = (*a as u128) * (*h as u128) + (*b as u128);
                        (product % (MERSENNE_PRIME as u128)) as u64
                    })
                    .min()
                    .expect("shingle set is non-empty")
                    // Stored as BIGINT: the modulus is below 2^61 so this
                    // never wraps into a negative value.
                    as i64
            })
            .collect();

        Some(signature)
    }

    /// LSH bucket keys, one per band.
    pub fn buckets(&self, signature: &[i64]) -> Vec<String> {
        let rows = self.rows_per_band();
        signature
            .chunks(rows)
            .enumerate()
            .map(|(band_index, band)| {
                let mut bytes = Vec::with_capacity(band.len() * 8 + 1);
                bytes.push(band_index as u8);
                for value in band {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                format!("{band_index}:{:016x}", fnv1a(&bytes))
            })
            .collect()
    }

    /// Primary bucket key, stored on the post row for later auditing.
    pub fn primary_bucket(&self, signature: &[i64]) -> Option<String> {
        self.buckets(signature).into_iter().next()
    }
}

/// Estimated Jaccard similarity: the fraction of signature positions that agree.
pub fn similarity(left: &[i64], right: &[i64]) -> f64 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let matches = left
        .iter()
        .zip(right.iter())
        .filter(|(a, b)| a == b)
        .count();
    matches as f64 / left.len() as f64
}

/// One remembered post, kept only long enough to compare newcomers against.
#[derive(Debug, Clone)]
struct Seen {
    post_id: String,
    signature: Vec<i64>,
    buckets: Vec<String>,
}

/// The verdict for a candidate post.
#[derive(Debug, Clone, PartialEq)]
pub enum DuplicateCheck {
    Unique,
    Duplicate {
        /// The earlier post this one duplicates.
        original_id: String,
        similarity: f64,
    },
}

/// A bounded sliding window of recently seen signatures.
///
/// Bounded on purpose. An unbounded index would grow without limit over a long
/// ingest run, and comparing against months-old posts is not useful anyway:
/// the same phrasing recurring next week is a separate event, not a duplicate.
/// The window is what makes this a *streaming* filter.
#[derive(Debug)]
pub struct DuplicateDetector {
    hasher: MinHasher,
    threshold: f64,
    capacity: usize,
    window: VecDeque<Seen>,
}

impl DuplicateDetector {
    pub fn new(hasher: MinHasher, threshold: f64, capacity: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in [0, 1]"
        );
        assert!(capacity > 0, "capacity must be positive");
        Self {
            hasher,
            threshold,
            capacity,
            window: VecDeque::with_capacity(capacity),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(
            MinHasher::default(),
            DEFAULT_DUPLICATE_THRESHOLD,
            DEFAULT_WINDOW_CAPACITY,
        )
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn hasher(&self) -> &MinHasher {
        &self.hasher
    }

    /// Check a post and remember it.
    ///
    /// Returns the verdict alongside the signature, so the caller can persist
    /// the signature even for rejected posts — which is what makes the filter's
    /// own precision auditable after the fact.
    pub fn check(&mut self, post_id: &str, text: &str) -> (DuplicateCheck, Option<Vec<i64>>) {
        let Some(signature) = self.hasher.signature(text) else {
            return (DuplicateCheck::Unique, None);
        };

        let buckets = self.hasher.buckets(&signature);

        // Only posts sharing at least one band bucket are compared, which is
        // what keeps this linear in practice rather than quadratic.
        let mut best: Option<(&Seen, f64)> = None;
        for seen in &self.window {
            if !seen.buckets.iter().any(|b| buckets.contains(b)) {
                continue;
            }
            let score = similarity(&signature, &seen.signature);
            if score >= self.threshold && best.map_or(true, |(_, current)| score > current) {
                best = Some((seen, score));
            }
        }

        let verdict = match best {
            Some((seen, score)) => DuplicateCheck::Duplicate {
                original_id: seen.post_id.clone(),
                similarity: score,
            },
            None => DuplicateCheck::Unique,
        };

        // Duplicates are remembered too. A ring posting twenty variants should
        // have every one of them match against the whole cluster, not just the
        // first member.
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(Seen {
            post_id: post_id.to_string(),
            signature: signature.clone(),
            buckets,
        });

        (verdict, Some(signature))
    }
}

/// Posts remembered per market. A busy market produces a few thousand posts a
/// day, so this covers roughly a day of history at a bounded memory cost.
pub const DEFAULT_WINDOW_CAPACITY: usize = 4096;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_urls_handles_and_tags() {
        let got = normalize("Fed CUTS rates!! https://t.co/abc @jpow #FOMC $SPY");
        assert_eq!(got, "fed cuts rates");
    }

    #[test]
    fn shingles_are_word_trigrams() {
        let got = shingles("the fed will cut rates");
        assert_eq!(
            got,
            vec!["the fed will", "fed will cut", "will cut rates"]
        );
    }

    #[test]
    fn short_text_still_yields_a_shingle() {
        assert_eq!(shingles("rate cut"), vec!["rate cut"]);
        assert!(shingles("").is_empty());
        assert!(shingles("https://t.co/x @spam").is_empty());
    }

    #[test]
    fn signature_is_deterministic_across_hashers() {
        // Signatures are persisted and compared by later process runs, so two
        // independently constructed hashers must agree.
        let a = MinHasher::default().signature("the fed will cut rates in july");
        let b = MinHasher::default().signature("the fed will cut rates in july");
        assert_eq!(a, b);
        assert_eq!(a.unwrap().len(), DEFAULT_NUM_HASHES);
    }

    #[test]
    fn identical_text_is_perfectly_similar() {
        let hasher = MinHasher::default();
        let text = "the federal reserve will cut rates by 25 basis points";
        let a = hasher.signature(text).unwrap();
        let b = hasher.signature(text).unwrap();
        assert_eq!(similarity(&a, &b), 1.0);
    }

    #[test]
    fn minor_edits_stay_similar() {
        let hasher = MinHasher::default();
        let a = hasher
            .signature("FOMC FOMC FOMC rate cut incoming!!! follow for more alpha")
            .unwrap();
        let b = hasher
            .signature("FOMC FOMC FOMC rate cut incoming!! follow for more alpha")
            .unwrap();
        let score = similarity(&a, &b);
        assert!(score > 0.8, "expected near-duplicate, got {score}");
    }

    #[test]
    fn unrelated_text_is_dissimilar() {
        let hasher = MinHasher::default();
        let a = hasher
            .signature("the federal reserve will cut rates by 25 basis points in july")
            .unwrap();
        let b = hasher
            .signature("bitcoin just broke through one hundred fifty thousand dollars")
            .unwrap();
        let score = similarity(&a, &b);
        assert!(score < 0.3, "expected dissimilar, got {score}");
    }

    #[test]
    fn estimate_approximates_true_jaccard() {
        // Two documents sharing most of their trigrams should estimate high.
        let hasher = MinHasher::default();
        let a = hasher
            .signature("alpha beta gamma delta epsilon zeta eta theta")
            .unwrap();
        let b = hasher
            .signature("alpha beta gamma delta epsilon zeta eta iota")
            .unwrap();

        let left: std::collections::HashSet<_> =
            shingles("alpha beta gamma delta epsilon zeta eta theta")
                .into_iter()
                .collect();
        let right: std::collections::HashSet<_> =
            shingles("alpha beta gamma delta epsilon zeta eta iota")
                .into_iter()
                .collect();
        let true_jaccard =
            left.intersection(&right).count() as f64 / left.union(&right).count() as f64;

        let estimate = similarity(&a, &b);
        assert!(
            (estimate - true_jaccard).abs() < 0.15,
            "estimate {estimate} too far from true jaccard {true_jaccard}"
        );
    }

    #[test]
    fn buckets_split_evenly_into_bands() {
        let hasher = MinHasher::default();
        let signature = hasher.signature("some ordinary post about the fed").unwrap();
        assert_eq!(hasher.buckets(&signature).len(), DEFAULT_NUM_BANDS);
        assert_eq!(hasher.rows_per_band(), DEFAULT_NUM_HASHES / DEFAULT_NUM_BANDS);
    }

    #[test]
    fn detector_flags_a_spam_ring() {
        // The three near-identical posts in the Fed fixture.
        let mut detector = DuplicateDetector::with_defaults();

        let (first, sig) = detector.check("f0005", "FOMC FOMC FOMC rate cut incoming!!! follow for more alpha");
        assert_eq!(first, DuplicateCheck::Unique);
        assert!(sig.is_some());

        let (second, _) = detector.check("f0006", "FOMC FOMC FOMC rate cut incoming!! follow for more alpha");
        match second {
            DuplicateCheck::Duplicate {
                ref original_id,
                similarity,
            } => {
                assert_eq!(original_id, "f0005");
                assert!(similarity >= DEFAULT_DUPLICATE_THRESHOLD);
            }
            other => panic!("expected duplicate, got {other:?}"),
        }

        let (third, _) = detector.check("f0007", "FOMC FOMC FOMC rate cut incoming!!! follow for more alpha.");
        assert!(matches!(third, DuplicateCheck::Duplicate { .. }));
    }

    #[test]
    fn detector_passes_genuine_posts() {
        let mut detector = DuplicateDetector::with_defaults();
        detector.check("a", "the fed is going to cut rates this month for sure");
        let (verdict, _) = detector.check(
            "b",
            "bitcoin looks strong heading into the end of the year",
        );
        assert_eq!(verdict, DuplicateCheck::Unique);
    }

    #[test]
    fn window_is_bounded() {
        let mut detector = DuplicateDetector::new(MinHasher::default(), 0.8, 4);
        for i in 0..20 {
            detector.check(&format!("p{i}"), &format!("unique post number {i} about markets"));
        }
        assert_eq!(detector.len(), 4);
    }

    #[test]
    fn empty_text_is_never_duplicate() {
        let mut detector = DuplicateDetector::with_defaults();
        let (a, sig_a) = detector.check("x", "https://t.co/aaa");
        let (b, sig_b) = detector.check("y", "@someone @another");
        assert_eq!(a, DuplicateCheck::Unique);
        assert_eq!(b, DuplicateCheck::Unique);
        assert!(sig_a.is_none() && sig_b.is_none());
    }

    #[test]
    fn signatures_are_non_negative() {
        // They are stored in a BIGINT column; a wrap to negative would be a
        // silent corruption of the persisted signature.
        let hasher = MinHasher::default();
        let signature = hasher.signature("checking the sign of every entry").unwrap();
        assert!(signature.iter().all(|v| *v >= 0));
    }
}

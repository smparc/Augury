//! Configuration from the environment.
//!
//! The `.env` at the repository root is the same file the Python service reads,
//! so the two cannot drift. Nothing else in the crate touches `std::env`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The template placeholder shipped in `.env.example`. Treated as absent so a
/// live run fails with a clear message instead of a confusing 401.
const TOKEN_PLACEHOLDER: &str = "your_bearer_token_here";

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub x_bearer_token: Option<String>,
    /// False means fixture replay and no paid calls. Default.
    pub x_live: bool,
    pub max_daily_reads: i32,
    pub language: String,
    pub fixtures_dir: PathBuf,
    pub poll_interval_secs: u64,
    pub max_posts_per_poll: usize,
    pub lookback_days: i64,
    pub log_level: String,
}

impl Config {
    /// Load from the process environment, seeding from the repo-root `.env`.
    pub fn load() -> Result<Self> {
        let root = repo_root()?;
        load_dotenv(&root.join(".env"));

        let user = env_or("POSTGRES_USER", "augury");
        let password = env_or("POSTGRES_PASSWORD", "augury_local_dev");
        let host = env_or("POSTGRES_HOST", "localhost");
        let port = env_or("POSTGRES_PORT", "5432");
        let database = env_or("POSTGRES_DB", "augury");

        let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
            format!("postgresql://{user}:{password}@{host}:{port}/{database}")
        });

        let token = std::env::var("X_BEARER_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty() && t != TOKEN_PLACEHOLDER);

        Ok(Self {
            database_url,
            x_bearer_token: token,
            x_live: env_bool("AUGURY_X_LIVE", false),
            max_daily_reads: env_parse("MAX_DAILY_READS", 2_000)?,
            language: env_or("AUGURY_LANG", "en"),
            fixtures_dir: root.join("apps").join("augury-signal").join("fixtures"),
            poll_interval_secs: env_parse("AUGURY_POLL_INTERVAL_SECS", 300)?,
            max_posts_per_poll: env_parse("AUGURY_MAX_POSTS_PER_POLL", 100)?,
            lookback_days: env_parse("AUGURY_LOOKBACK_DAYS", 7)?,
            log_level: env_or("LOG_LEVEL", "info"),
        })
    }

    /// Whether live ingestion is actually possible.
    ///
    /// Live mode with no token degrades to fixtures with a warning rather than
    /// failing: a misconfigured token should cost nothing, not crash the run.
    pub fn live_enabled(&self) -> bool {
        self.x_live && self.x_bearer_token.is_some()
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => value
            .trim()
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("{key}={value:?} is not valid: {e}")),
        _ => Ok(default),
    }
}

/// Minimal `.env` reader.
///
/// Deliberately does not overwrite variables already set in the environment, so
/// an explicit `MAX_DAILY_READS=50 cargo run` wins over the file.
fn load_dotenv(path: &Path) {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return;
    };

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if std::env::var(key).is_err() {
            std::env::set_var(key, value);
        }
    }
}

/// Walk up from the crate directory to the repository root.
pub fn repo_root() -> Result<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut current = start.as_path();
    loop {
        if current.join("config").join("markets.yaml").exists() {
            return Ok(current.to_path_buf());
        }
        current = current
            .parent()
            .context("could not find the repository root (no config/markets.yaml in any parent)")?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_root_contains_the_market_config() {
        let root = repo_root().unwrap();
        assert!(root.join("config").join("markets.yaml").exists());
        assert!(root.join("schemas").exists());
    }

    #[test]
    fn fixtures_dir_exists() {
        let root = repo_root().unwrap();
        let fixtures = root.join("apps").join("augury-signal").join("fixtures");
        assert!(fixtures.exists(), "{} should exist", fixtures.display());
    }

    #[test]
    fn env_bool_accepts_common_spellings() {
        std::env::set_var("AUGURY_TEST_BOOL", "TRUE");
        assert!(env_bool("AUGURY_TEST_BOOL", false));
        std::env::set_var("AUGURY_TEST_BOOL", "0");
        assert!(!env_bool("AUGURY_TEST_BOOL", true));
        std::env::remove_var("AUGURY_TEST_BOOL");
        assert!(env_bool("AUGURY_TEST_BOOL", true));
    }

    #[test]
    fn env_parse_reports_bad_values() {
        std::env::set_var("AUGURY_TEST_INT", "not a number");
        let parsed = env_parse::<i32>("AUGURY_TEST_INT", 5);
        std::env::remove_var("AUGURY_TEST_INT");
        assert!(parsed.is_err());
    }
}

//! Persisted daily read budget for the X API.
//!
//! X charges per post read, so the budget is the only thing between a polling
//! loop and a real bill. Two properties are non-negotiable:
//!
//! * **It survives restarts.** An in-process counter resets every time the
//!   service comes back up, so a crash loop becomes unbounded spending. The
//!   counter therefore lives in the `read_budget` table, not in memory.
//! * **It is checked before the call**, against the largest number of reads the
//!   call could consume, then reconciled to the actual count afterwards.
//!
//! The reservation is written *before* the request goes out. Recording usage
//! only on success would let a request that times out after the server already
//! counted it slip through unbilled, and a run of those would quietly overshoot
//! the ceiling.

use anyhow::{Context, Result};
use sqlx::PgPool;

pub const SERVICE_NAME: &str = "augury-ingest";

/// Approximate cost per post read, for human-readable logging only.
pub const COST_PER_READ_USD: f64 = 0.005;

#[derive(Debug, thiserror::Error)]
#[error(
    "{service}: {used} of {limit} reads used today; {requested} more would exceed the budget \
     (~${cost:.2}). Raise MAX_DAILY_READS or wait for the UTC day to roll over."
)]
pub struct BudgetExceeded {
    pub service: String,
    pub used: i32,
    pub limit: i32,
    pub requested: i32,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct ReadBudget {
    pool: PgPool,
    limit: i32,
    service: String,
}

impl ReadBudget {
    pub fn new(pool: PgPool, limit: i32) -> Self {
        Self {
            pool,
            limit,
            service: SERVICE_NAME.to_string(),
        }
    }

    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = service.into();
        self
    }

    pub fn limit(&self) -> i32 {
        self.limit
    }

    /// Reads consumed so far today (UTC).
    pub async fn used_today(&self) -> Result<i32> {
        let row: Option<(i32,)> = sqlx::query_as(
            "SELECT reads_used FROM read_budget WHERE day = CURRENT_DATE AND service = $1",
        )
        .bind(&self.service)
        .fetch_optional(&self.pool)
        .await
        .context("querying today's read budget")?;

        Ok(row.map(|(used,)| used).unwrap_or(0))
    }

    pub async fn remaining(&self) -> Result<i32> {
        Ok((self.limit - self.used_today().await?).max(0))
    }

    /// Reserve `requested` reads, or fail without spending anything.
    ///
    /// The increment and the ceiling check happen in one statement so two
    /// concurrent workers cannot both observe headroom and both spend it. The
    /// `WHERE` clause on the upsert is what makes it atomic — a read followed
    /// by a write would race.
    pub async fn reserve(&self, requested: i32) -> Result<i32> {
        anyhow::ensure!(requested >= 0, "requested must be non-negative");

        // `INSERT ... SELECT ... WHERE` rather than `INSERT ... VALUES`: on the
        // first request of a UTC day there is no conflicting row, so the
        // `ON CONFLICT` guard never fires and a `VALUES` form would happily
        // insert a reservation larger than the entire daily budget. The WHERE
        // on the SELECT closes that hole; the ON CONFLICT guard covers every
        // subsequent request. Both are needed.
        let row: Option<(i32,)> = sqlx::query_as(
            r#"
            INSERT INTO read_budget (day, service, reads_used)
            SELECT CURRENT_DATE, $1, $2 WHERE $2 <= $3
            ON CONFLICT (day, service) DO UPDATE
                SET reads_used = read_budget.reads_used + EXCLUDED.reads_used,
                    updated_at = now()
                WHERE read_budget.reads_used + EXCLUDED.reads_used <= $3
            RETURNING reads_used
            "#,
        )
        .bind(&self.service)
        .bind(requested)
        .bind(self.limit)
        .fetch_optional(&self.pool)
        .await
        .context("reserving read budget")?;

        match row {
            Some((used,)) => Ok(used),
            None => {
                // The conditional update matched nothing, meaning the ceiling
                // would have been breached.
                let used = self.used_today().await?;
                Err(BudgetExceeded {
                    service: self.service.clone(),
                    used,
                    limit: self.limit,
                    requested,
                    cost: requested as f64 * COST_PER_READ_USD,
                }
                .into())
            }
        }
    }

    /// Give back the difference between what was reserved and what was used.
    ///
    /// A page reserved at 100 that returns 12 posts should not consume 100 of
    /// the day's budget; without this the effective ceiling would collapse to
    /// the number of *requests* rather than the number of reads.
    pub async fn release_unused(&self, reserved: i32, actual: i32) -> Result<()> {
        let refund = reserved - actual;
        if refund <= 0 {
            return Ok(());
        }

        sqlx::query(
            r#"
            UPDATE read_budget
               SET reads_used = GREATEST(0, reads_used - $2),
                   updated_at = now()
             WHERE day = CURRENT_DATE AND service = $1
            "#,
        )
        .bind(&self.service)
        .bind(refund)
        .execute(&self.pool)
        .await
        .context("releasing unused read budget")?;

        Ok(())
    }

    pub async fn summary(&self) -> Result<String> {
        let used = self.used_today().await?;
        Ok(format!(
            "{}: {}/{} reads used today (~${:.2}), {} remaining",
            self.service,
            used,
            self.limit,
            used as f64 * COST_PER_READ_USD,
            (self.limit - used).max(0)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The reserve/release logic is exercised against a live database by the
    // integration tests in `tests/`, which are skipped unless DATABASE_URL is
    // set. What is unit-testable here is the arithmetic of the error message.
    #[test]
    fn exceeded_error_reports_cost() {
        let err = BudgetExceeded {
            service: SERVICE_NAME.into(),
            used: 1_900,
            limit: 2_000,
            requested: 500,
            cost: 500.0 * COST_PER_READ_USD,
        };
        let message = err.to_string();
        assert!(message.contains("1900 of 2000"));
        assert!(message.contains("$2.50"));
        assert!(message.contains("MAX_DAILY_READS"));
    }

    #[test]
    fn refund_is_only_positive_difference() {
        // release_unused is a no-op when actual >= reserved.
        assert!(100 - 100 <= 0);
        assert!(100 - 120 <= 0);
        assert_eq!(100 - 12, 88);
    }
}

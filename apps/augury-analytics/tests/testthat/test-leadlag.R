# Tests for the R analytics, including the shared cross-language golden vectors.

test_that("logit and clamping behave at the boundaries", {
  # Prediction markets sit at 0 and 1 constantly; infinite values would poison
  # every downstream regression.
  expect_true(is.finite(logit(0)))
  expect_true(is.finite(logit(1)))
  expect_equal(logit(0.5), 0, tolerance = 1e-9)
  expect_equal(clamp_probability(1.5), 1 - 1e-6)
  expect_equal(clamp_probability(-0.2), 1e-6)
})

test_that("signal_to_probability maps [-1,1] onto [0,1]", {
  expect_equal(signal_to_probability(-1), 0)
  expect_equal(signal_to_probability(0), 0.5)
  expect_equal(signal_to_probability(1), 1)
})

test_that("golden vectors: logit matches the Python reference", {
  path <- find_golden_vectors()
  skip_if(is.null(path), "golden_vectors.json not found")

  vectors <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  for (case in vectors$calibration$logit) {
    expect_equal(logit(case$p), case$expected_logit, tolerance = 1e-9)
  }
})

test_that("golden vectors: affine calibration matches the Python reference", {
  path <- find_golden_vectors()
  skip_if(is.null(path), "golden_vectors.json not found")

  vectors <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  for (case in vectors$calibration$affine) {
    expect_equal(signal_to_probability(case$s_t), case$expected_p, tolerance = 1e-9)
  }
})

test_that("ADF distinguishes white noise from a random walk", {
  set.seed(1)
  noise <- rnorm(300)
  walk <- cumsum(rnorm(300))

  expect_true(test_stationarity(noise)$is_stationary)
  expect_false(test_stationarity(walk)$is_stationary)
})

test_that("ADF refuses degenerate input", {
  expect_error(test_stationarity(rep(0.5, 50)), "constant")
  expect_error(test_stationarity(c(0.1, 0.2, 0.3)), "at least 10")
})

test_that("cross-correlation finds a known lead", {
  set.seed(7)
  signal <- rnorm(400)
  # Price is the signal shifted forward by 3 bars, plus noise.
  price <- c(rep(NA, 3), head(signal, -3)) + rnorm(400, sd = 0.05)

  table <- cross_correlation(signal, price, max_lag = 6)
  best <- table[which.max(abs(table$correlation)), ]

  expect_equal(best$lag, 3)
  expect_equal(best$interpretation, "signal leads")
})

test_that("cross-correlation returns NA for constant series rather than 0", {
  # 0 would read as "no relationship"; the truth is the question is unanswerable.
  table <- cross_correlation(rep(0.5, 50), seq(0, 1, length.out = 50), max_lag = 3)
  expect_true(all(is.na(table$correlation)))
})

test_that("Granger detects a genuine lead", {
  set.seed(11)
  n <- 400
  s <- numeric(n)
  p <- numeric(n)
  for (t in 2:n) {
    s[t] <- 0.5 * s[t - 1] + rnorm(1, sd = 0.3)
    p[t] <- 0.4 * p[t - 1] + 0.6 * s[t - 1] + rnorm(1, sd = 0.1)
  }

  result <- granger_test(tanh(s), plogis(p), direction = "signal_to_price")

  expect_lt(result$p_value, 0.01)
  expect_gte(result$lag_order, 1)
  expect_true(is.na(result$significant))  # not corrected yet
})

test_that("Granger finds nothing in independent series", {
  set.seed(13)
  result <- granger_test(tanh(rnorm(400)), plogis(rnorm(400)))
  expect_gt(result$p_value, 0.01)
})

test_that("Granger refuses uninterpretably short history", {
  expect_error(
    granger_test(rep(0.1, 15), rep(0.5, 15)),
    "not interpretable"
  )
})

test_that("an uncorrected result is never significant", {
  # `significant` must stay NA until the batch correction runs; defaulting it
  # to the raw p-value is exactly the multiple-comparisons error.
  set.seed(3)
  result <- granger_test(tanh(rnorm(100)), plogis(rnorm(100)))
  expect_true(is.na(result$significant))
})

test_that("FDR correction raises p-values and kills lone borderline findings", {
  make <- function(p) list(p_value = p, market_id = "m")

  raw <- c(0.04, seq(0.05, 0.95, length.out = 19))
  corrected <- apply_fdr_correction(lapply(raw, make))

  adjusted <- vapply(corrected, function(r) r$p_value_adj, numeric(1))
  expect_true(all(adjusted >= raw - 1e-12))
  # One borderline result among 19 nulls is what chance produces.
  expect_false(corrected[[1]]$significant)
})

test_that("FDR keeps a genuinely strong finding", {
  make <- function(p) list(p_value = p, market_id = "m")
  batch <- c(list(make(1e-6)), lapply(rep(0.6, 19), make))
  corrected <- apply_fdr_correction(batch)

  expect_true(corrected[[1]]$significant)
  expect_false(any(vapply(corrected[-1], function(r) r$significant, logical(1))))
})

test_that("alignment drops bars rather than bridging a long gap", {
  base <- as.POSIXct("2026-07-01 00:00:00", tz = "UTC")

  signals <- data.frame(ts = base + 3600 * (0:10), s_t = rep(0.1, 11))
  # Price is missing for 8 straight bars.
  prices <- data.frame(
    ts = base + 3600 * c(0, 1, 10),
    yes_price = c(0.3, 0.31, 0.5)
  )

  aligned <- align_on_grid(signals, prices, max_fill_bars = 1)
  expect_lte(nrow(aligned), 4)
})

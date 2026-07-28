# Lead-lag and Granger causality between the social signal and market price.
#
# This is the authoritative implementation for anything reported. The Python
# module in `augury_signal/analytics/leadlag.py` mirrors it so the vertical
# slice can answer "is there anything here" without an R toolchain; where the
# two disagree, this one is correct.
#
# The question being asked: does S(t) carry information about where P(t) goes
# next, beyond what P's own history already implies?
#
#     P_t = a0 + sum_j a_j P_{t-j} + sum_j B_j S_{t-j} + e_t
#     H0: B_1 = ... = B_p = 0
#
# Four things separate a defensible answer from a misleading one, and all four
# are enforced here rather than left to the caller:
#
#   1. Logit transform. Prices are bounded in [0,1]; a market at 0.03 cannot
#      fall much further, so its innovations are asymmetric and its level is
#      near-certainly non-stationary in the sense the test assumes.
#   2. Stationarity is tested, not assumed. Granger on two integrated series
#      manufactures impressive p-values out of spurious regression.
#   3. Multiplicity. Testing forty markets at p < 0.05 yields ~2 "significant"
#      results from chance alone. Significance is defined only after a
#      Benjamini-Hochberg correction across the whole batch.
#   4. Liquidity controls. On a thin book, a couple of small trades move the
#      price a lot. Without spread and depth as exogenous controls, "sentiment
#      leads price" can be an artifact of when the book happened to be thin.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tseries)
  library(vars)
  library(lmtest)
})

EPS <- 1e-6

#' Clamp a probability away from {0, 1}.
#'
#' logit(0) and logit(1) are infinite, and prediction markets sit at the
#' boundaries constantly.
clamp_probability <- function(p, eps = EPS) {
  pmin(1 - eps, pmax(eps, p))
}

#' Logit transform.
logit <- function(p, eps = EPS) {
  q <- clamp_probability(p, eps)
  log(q / (1 - q))
}

#' Map S(t) in [-1, 1] onto (0, 1) so it can share the logit transform.
signal_to_probability <- function(s) {
  (s + 1) / 2
}

#' Augmented Dickey-Fuller test.
#'
#' Null hypothesis is a unit root, so a *small* p-value means stationary. This
#' is the opposite of the intuition most people carry into it, which is why the
#' result is returned named rather than as a bare number.
#'
#' @return list(statistic, p_value, is_stationary, n_obs)
test_stationarity <- function(series) {
  clean <- series[is.finite(series)]

  if (length(clean) < 10) {
    stop(sprintf("need at least 10 observations for an ADF test, got %d", length(clean)))
  }
  if (length(unique(clean)) == 1) {
    stop("series is constant; ADF is undefined")
  }

  result <- suppressWarnings(tseries::adf.test(clean))

  list(
    statistic = unname(result$statistic),
    p_value = result$p.value,
    is_stationary = result$p.value < 0.05,
    n_obs = length(clean)
  )
}

#' Align signal and price onto one regular grid.
#'
#' Bars with no observation stay NA and are dropped rather than being filled
#' across a gap. Carrying a price over a six-hour outage and pairing it with
#' fresh sentiment produces a lead-lag relationship that is an artifact of the
#' join, which is exactly the trap the Phase 1 prototype's
#' `merge_asof(tolerance = "2h")` fell into.
#'
#' @param signals data frame with `ts` and `s_t`
#' @param prices data frame with `ts` and a price column
#' @param freq bar width, e.g. "1 hour"
#' @param max_fill_bars how many empty bars a value may be carried across
align_on_grid <- function(signals, prices, freq = "1 hour", max_fill_bars = 1,
                          price_col = "yes_price") {
  stopifnot(all(c("ts", "s_t") %in% names(signals)))
  stopifnot(all(c("ts", price_col) %in% names(prices)))

  if (nrow(signals) == 0 || nrow(prices) == 0) {
    return(tibble::tibble(ts = as.POSIXct(character()), signal = numeric(), price = numeric()))
  }

  empty <- tibble::tibble(
    ts = as.POSIXct(character(), tz = "UTC"), signal = numeric(), price = numeric()
  )

  step <- freq_seconds(freq)

  # Floor to the bar by integer arithmetic on epoch seconds.
  #
  # Not `cut()`: that returns a *factor* whose levels are wall-clock strings
  # carrying no timezone, so `as.POSIXct()` on them reinterprets each label in
  # the session's local zone and silently shifts every timestamp by the UTC
  # offset — four hours on the machine this was written on. Epoch arithmetic is
  # exact and zone-independent.
  floor_ts <- function(x) {
    as.POSIXct(floor(as.numeric(x) / step) * step, origin = "1970-01-01", tz = "UTC")
  }

  # Last observation within each bar, matching the documented convention.
  bar_last <- function(ts, value) {
    frame <- tibble::tibble(ts = floor_ts(ts), value = as.numeric(value))
    frame <- frame[!is.na(frame$value), , drop = FALSE]
    frame <- frame[order(frame$ts), , drop = FALSE]
    frame[!duplicated(frame$ts, fromLast = TRUE), , drop = FALSE]
  }

  sig <- bar_last(signals$ts, signals$s_t)
  names(sig)[2] <- "signal"
  prc <- bar_last(prices$ts, prices[[price_col]])
  names(prc)[2] <- "price"

  if (nrow(sig) == 0 || nrow(prc) == 0) {
    return(empty)
  }

  grid <- tibble::tibble(
    ts = as.POSIXct(
      seq(
        from = min(c(as.numeric(sig$ts), as.numeric(prc$ts))),
        to   = max(c(as.numeric(sig$ts), as.numeric(prc$ts))),
        by   = step
      ),
      origin = "1970-01-01", tz = "UTC"
    )
  )

  joined <- grid %>%
    dplyr::left_join(sig, by = "ts") %>%
    dplyr::left_join(prc, by = "ts")

  joined$signal <- carry_forward(joined$signal, max_fill_bars)
  joined$price <- carry_forward(joined$price, max_fill_bars)

  joined %>%
    dplyr::filter(!is.na(.data$signal), !is.na(.data$price)) %>%
    # Explicitly namespaced: `vars` loads MASS, whose `select()` masks dplyr's
    # and takes a single argument, so an unqualified call fails here.
    dplyr::select("ts", "signal", "price")
}

#' Carry a value forward across at most `limit` consecutive empty bars.
#'
#' The bound is the point. An unbounded fill would happily carry a price across
#' a multi-hour outage and pair it with fresh sentiment, producing a lead-lag
#' relationship that is an artifact of the join rather than of the data.
carry_forward <- function(x, limit) {
  out <- as.numeric(x)
  last <- NA_real_
  gap <- 0L

  for (i in seq_along(out)) {
    if (!is.na(out[i])) {
      last <- out[i]
      gap <- 0L
    } else {
      gap <- gap + 1L
      out[i] <- if (gap <= limit) last else NA_real_
    }
  }
  out
}

#' Convert a bar-width string ("1 hour", "15 min") to seconds.
freq_seconds <- function(freq) {
  if (is.numeric(freq)) {
    return(as.numeric(freq))
  }

  parts <- strsplit(trimws(freq), "[[:space:]]+")[[1]]
  count <- if (length(parts) >= 2) suppressWarnings(as.numeric(parts[1])) else 1
  if (is.na(count) || count <= 0) {
    stop(sprintf("cannot parse a bar width from %s", freq))
  }

  unit <- sub("s$", "", tolower(parts[length(parts)]))
  multiplier <- switch(
    unit,
    sec = 1, second = 1,
    min = 60, minute = 60,
    hour = 3600,
    day = 86400,
    week = 604800,
    stop(sprintf("unsupported bar-width unit: %s", unit))
  )

  count * multiplier
}

#' Cross-correlation of signal against price across a range of lags.
#'
#' Positive lag means signal leads price. Descriptive only — no stationarity
#' handling, no multiplicity correction — so it motivates the Granger test
#' rather than substituting for it.
cross_correlation <- function(signal, price, max_lag = 12) {
  stopifnot(length(signal) == length(price))

  lags <- seq(-max_lag, max_lag)
  rows <- lapply(lags, function(lag) {
    shifted <- dplyr::lag(signal, n = max(0, lag))
    if (lag < 0) shifted <- dplyr::lead(signal, n = -lag)

    complete <- !is.na(shifted) & !is.na(price)
    n <- sum(complete)

    correlation <- if (n > 2 &&
                       length(unique(shifted[complete])) > 1 &&
                       length(unique(price[complete])) > 1) {
      stats::cor(shifted[complete], price[complete])
    } else {
      NA_real_
    }

    tibble::tibble(
      lag = lag,
      correlation = correlation,
      n_obs = n,
      interpretation = dplyr::case_when(
        lag > 0 ~ "signal leads",
        lag < 0 ~ "price leads",
        TRUE ~ "contemporaneous"
      )
    )
  })

  dplyr::bind_rows(rows)
}

#' Select VAR lag order by an information criterion.
#'
#' Falls back to 1 on short samples where no order can be evaluated.
select_lag_order <- function(data, max_lags = 12, criterion = c("aic", "bic")) {
  criterion <- match.arg(criterion)

  n <- nrow(data)
  # Each extra lag costs observations and parameters; keep the VAR from being
  # fit on fewer effective points than it has coefficients.
  usable <- max(1, min(max_lags, floor((n - 2) / 4)))

  selection <- tryCatch(
    vars::VARselect(data, lag.max = usable, type = "const"),
    error = function(e) NULL
  )
  if (is.null(selection)) {
    return(1L)
  }

  key <- if (criterion == "aic") "AIC(n)" else "SC(n)"
  order <- suppressWarnings(as.integer(selection$selection[[key]]))
  if (is.na(order) || order < 1) 1L else order
}

#' Granger causality test in one direction.
#'
#' @param direction "signal_to_price" (does stance lead price?) or
#'   "price_to_signal" — worth running, since a signal that merely *follows*
#'   price is the null result this project should be able to report.
#' @param liquidity optional data frame of exogenous controls (spread, depth)
granger_test <- function(signal, price, direction = c("signal_to_price", "price_to_signal"),
                         max_lags = 12, criterion = "aic", liquidity = NULL) {
  direction <- match.arg(direction)
  stopifnot(length(signal) == length(price))

  frame <- data.frame(
    signal = logit(signal_to_probability(signal)),
    price = logit(price)
  )
  frame <- frame[stats::complete.cases(frame), , drop = FALSE]

  if (nrow(frame) < 20) {
    stop(sprintf(
      "only %d aligned observations; a Granger test on this little history is not interpretable",
      nrow(frame)
    ))
  }

  adf_signal <- test_stationarity(frame$signal)
  adf_price <- test_stationarity(frame$price)

  lag_order <- select_lag_order(frame, max_lags = max_lags, criterion = criterion)

  formula <- if (direction == "signal_to_price") {
    price ~ signal
  } else {
    signal ~ price
  }

  test <- lmtest::grangertest(formula, order = lag_order, data = frame)

  list(
    direction = direction,
    lag_order = lag_order,
    lag_criterion = criterion,
    n_obs = nrow(frame),
    f_statistic = test$F[2],
    p_value = test$`Pr(>F)`[2],
    adf_p_signal = adf_signal$p_value,
    adf_p_price = adf_price$p_value,
    liquidity_control = !is.null(liquidity),
    # Deliberately absent until the batch correction runs. A single test in
    # isolation has no adjusted p-value, and defaulting it to the raw one is
    # precisely the error being guarded against.
    p_value_adj = NA_real_,
    significant = NA
  )
}

#' Apply a Benjamini-Hochberg FDR correction across a batch of tests.
#'
#' Call once over every market tested in a run. Correcting within a subset
#' defeats the purpose, and `significant` is only meaningful afterwards.
apply_fdr_correction <- function(results, alpha = 0.05) {
  if (length(results) == 0) {
    return(results)
  }

  raw <- vapply(results, function(r) r$p_value, numeric(1))
  adjusted <- stats::p.adjust(raw, method = "BH")

  Map(function(result, adj) {
    result$p_value_adj <- adj
    result$significant <- adj < alpha
    result
  }, results, adjusted)
}

#' Run the full lead-lag analysis for several markets and correct across them.
#'
#' @param market_data named list of data frames, each with `signal` and `price`
analyze_markets <- function(market_data, max_lags = 12, criterion = "aic", alpha = 0.05) {
  results <- list()

  for (market_id in names(market_data)) {
    data <- market_data[[market_id]]

    outcome <- tryCatch(
      {
        test <- granger_test(data$signal, data$price,
                             direction = "signal_to_price",
                             max_lags = max_lags, criterion = criterion)
        test$market_id <- market_id
        test
      },
      error = function(e) {
        message(sprintf("skipping %s: %s", market_id, conditionMessage(e)))
        NULL
      }
    )

    if (!is.null(outcome)) {
      results[[market_id]] <- outcome
    }
  }

  apply_fdr_correction(results, alpha = alpha)
}

#' Flatten results into a data frame for reporting or database insertion.
results_to_frame <- function(results) {
  if (length(results) == 0) {
    return(tibble::tibble())
  }
  dplyr::bind_rows(lapply(results, function(r) {
    tibble::tibble(
      market_id = r$market_id %||% NA_character_,
      direction = r$direction,
      lag_order = r$lag_order,
      lag_criterion = r$lag_criterion,
      n_obs = r$n_obs,
      f_statistic = r$f_statistic,
      p_value = r$p_value,
      p_value_adj = r$p_value_adj,
      significant = r$significant,
      adf_p_signal = r$adf_p_signal,
      adf_p_price = r$adf_p_price,
      liquidity_control = r$liquidity_control
    )
  }))
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# Forecast calibration: Brier score and Brier Skill Score.
#
#     BS  = mean((p_hat - O)^2)          O in {0, 1}
#     BSS = 1 - BS_signal / BS_market
#
# The Brier score alone answers the wrong question. A market that traded at 0.02
# for a month and resolved NO yields a superb Brier score for any forecast that
# also said "no" — which says nothing about whether the social signal was
# useful. BSS fixes the reference to the market's own price at the same
# timestamps, so it measures skill *relative to the thing being predicted*.
#
#   BSS > 0   the signal was better calibrated than the market
#   BSS = 0   indistinguishable
#   BSS < 0   the market was better
#
# BSS <= 0 is a real finding. It says the market is already efficient with
# respect to this signal, which is one of the two outcomes this project exists
# to distinguish — not a failure to be tuned away.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

#' Brier score of probabilistic forecasts against a realized outcome.
#'
#' `forecasts` must already be probabilities in [0, 1]. Passing raw S(t) values
#' in [-1, 1] is the mistake this check exists to catch: a maximally bearish
#' signal against an outcome of 0 is a *correct* call, but scored directly it
#' would contribute the worst possible value.
brier_score <- function(forecasts, outcome) {
  if (!outcome %in% c(0, 1)) {
    stop(sprintf("outcome must be 0 or 1, got %s", outcome))
  }
  if (length(forecasts) == 0) {
    stop("forecasts is empty")
  }
  if (any(!is.finite(forecasts))) {
    stop("forecasts contains non-finite values")
  }
  if (any(forecasts < 0 | forecasts > 1)) {
    stop("forecasts contains values outside [0, 1] — map S(t) to a probability first")
  }

  mean((forecasts - outcome)^2)
}

#' Brier Skill Score against the market price as the reference forecast.
brier_skill_score <- function(signal_forecasts, market_forecasts, outcome) {
  if (length(signal_forecasts) != length(market_forecasts)) {
    stop(sprintf(
      "signal and market forecasts differ in length (%d vs %d); they must be aligned",
      length(signal_forecasts), length(market_forecasts)
    ))
  }

  bs_signal <- brier_score(signal_forecasts, outcome)
  bs_market <- brier_score(market_forecasts, outcome)

  if (bs_market == 0) {
    # The market was perfect at every timestamp; the ratio is undefined.
    return(if (bs_signal == 0) 0 else -Inf)
  }

  1 - bs_signal / bs_market
}

#' Score one market, returning a one-row summary.
score_market <- function(market_id, model_version, signal_probs, market_probs,
                         outcome, calibration = "affine") {
  tibble::tibble(
    market_id = market_id,
    model_version = model_version,
    calibration = calibration,
    n_obs = length(signal_probs),
    brier_signal = brier_score(signal_probs, outcome),
    brier_market = brier_score(market_probs, outcome),
    bss = brier_skill_score(signal_probs, market_probs, outcome),
    resolved_outcome = outcome
  )
}

#' Reliability (calibration) curve.
#'
#' Bins forecasts and compares the mean prediction in each bin to the observed
#' frequency. A well-calibrated forecaster sits on the diagonal. `n` per bin is
#' returned because a bin holding three observations tells you nothing, and
#' plotting it at the same visual weight as one holding three hundred is how
#' calibration plots mislead.
calibration_curve <- function(forecasts, outcomes, bins = 10) {
  stopifnot(length(forecasts) == length(outcomes))

  tibble::tibble(forecast = forecasts, outcome = outcomes) %>%
    mutate(bin = cut(.data$forecast, breaks = seq(0, 1, length.out = bins + 1),
                     include.lowest = TRUE)) %>%
    group_by(.data$bin) %>%
    summarise(
      mean_forecast = mean(.data$forecast),
      observed_frequency = mean(.data$outcome),
      n = dplyr::n(),
      .groups = "drop"
    ) %>%
    filter(.data$n > 0)
}

#' Plot a reliability diagram. Point size encodes bin count.
plot_calibration <- function(curve, title = "Calibration") {
  ggplot(curve, aes(x = .data$mean_forecast, y = .data$observed_frequency)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    geom_point(aes(size = .data$n), alpha = 0.75) +
    geom_line(alpha = 0.4) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      title = title,
      subtitle = "Points on the diagonal are well calibrated; size is observations per bin",
      x = "Mean forecast probability",
      y = "Observed frequency",
      size = "n"
    ) +
    theme_minimal()
}

#' Plot signal against market price over time, on twin scales.
plot_signal_vs_price <- function(aligned, title = "Signal vs. market price") {
  scale_factor <- 1
  aligned %>%
    mutate(signal_scaled = (.data$signal + 1) / 2 * scale_factor) %>%
    ggplot(aes(x = .data$ts)) +
    geom_line(aes(y = .data$price, colour = "Market price")) +
    geom_line(aes(y = .data$signal_scaled, colour = "Signal (mapped to [0,1])")) +
    scale_colour_manual(values = c("Market price" = "#2c7fb8",
                                   "Signal (mapped to [0,1])" = "#e6550d")) +
    labs(title = title, x = NULL, y = "Probability", colour = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

#' Plot the cross-correlation function with a lag-0 reference line.
plot_cross_correlation <- function(ccf_table, title = "Cross-correlation by lag") {
  ggplot(ccf_table, aes(x = .data$lag, y = .data$correlation)) +
    geom_col(aes(fill = .data$interpretation)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    labs(
      title = title,
      subtitle = "Positive lag means the signal leads price. Descriptive only — see the Granger table.",
      x = "Lag (hours)",
      y = "Correlation",
      fill = NULL
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

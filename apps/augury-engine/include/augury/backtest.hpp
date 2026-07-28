// Walk-forward backtesting for the synthetic LMSR market.
//
// A single train/test split answers one question about one arbitrary cut point.
// Walk-forward asks it repeatedly: fit on window N, evaluate on window N+1,
// roll, repeat. What comes out is not one number but a distribution — and the
// stability of the calibrated `b` across windows is itself a result. A `b` that
// swings by an order of magnitude between adjacent windows means the depth
// calibration is tracking noise, and any backtest built on it is not
// measuring what it claims to.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace augury {

/// One aligned observation: the signal, the real market, and the book.
struct Observation {
  std::int64_t timestamp = 0;  // Unix seconds.
  double signal = 0.0;         // S(t) in [-1, 1].
  double market_price = 0.0;   // Real YES price in [0, 1].
  std::optional<double> yes_bid;
  std::optional<double> yes_ask;
  std::optional<double> depth_bid;
  std::optional<double> depth_ask;
};

/// Result of evaluating one walk-forward window.
struct WindowResult {
  std::size_t index = 0;
  std::size_t train_start = 0;
  std::size_t train_end = 0;  // Exclusive.
  std::size_t test_start = 0;
  std::size_t test_end = 0;  // Exclusive.

  /// `b` fitted on the training window and held fixed through the test window.
  double calibrated_b = 0.0;
  /// Responsiveness fitted on the training window.
  double responsiveness = 0.0;

  /// Mean squared error of the simulated price against the real price over the
  /// test window. Lower is better, but see `mse_baseline`.
  double mse = 0.0;
  /// MSE of a random-walk baseline: predict the last observed price. A
  /// simulation that cannot beat this is adding nothing.
  double mse_baseline = 0.0;
  /// Correlation between simulated and real price changes over the test window.
  double correlation = 0.0;
  std::size_t n_test = 0;

  /// Whether the simulation beat the naive baseline on this window.
  [[nodiscard]] bool BeatsBaseline() const { return mse < mse_baseline; }
};

/// Summary across every window.
struct BacktestSummary {
  std::vector<WindowResult> windows;

  double mean_mse = 0.0;
  double mean_baseline_mse = 0.0;
  std::size_t windows_beating_baseline = 0;

  /// Spread of the calibrated `b` across windows. The headline stability
  /// number: `b_max / b_min` far above 1 says the liquidity calibration is not
  /// reproducible, which undermines every other figure here.
  double b_min = 0.0;
  double b_max = 0.0;
  double b_mean = 0.0;
  double b_coefficient_of_variation = 0.0;

  [[nodiscard]] std::string Report() const;
};

/// Parameters for a walk-forward run.
struct BacktestConfig {
  /// Observations per training window.
  std::size_t train_size = 72;
  /// Observations per test window.
  std::size_t test_size = 24;
  /// How far the window advances each step. Defaults to `test_size`, i.e.
  /// non-overlapping test windows — overlapping ones would reuse the same
  /// observations across windows and make the results look more consistent
  /// than they are.
  std::optional<std::size_t> step;
  /// Candidate responsiveness values searched on each training window.
  std::vector<double> responsiveness_grid{0.1, 0.2, 0.35, 0.5, 0.75, 1.0};
  /// Fallback `b` when the training window has no usable book.
  double fallback_b = 100.0;
};

/// Simulate the LMSR over a span of observations with fixed parameters.
///
/// Returns the simulated price at each observation. The market is anchored to
/// the first real price so the first step is not dominated by an arbitrary
/// starting point.
[[nodiscard]] std::vector<double> SimulatePath(const std::vector<Observation>& observations,
                                               std::size_t start, std::size_t end, double b,
                                               double responsiveness);

/// Mean squared error between two equal-length series.
[[nodiscard]] double MeanSquaredError(const std::vector<double>& predicted,
                                      const std::vector<double>& actual);

/// Pearson correlation of first differences. Returns 0 when either series is
/// constant, since the correlation is then undefined rather than zero.
[[nodiscard]] double DifferenceCorrelation(const std::vector<double>& left,
                                           const std::vector<double>& right);

/// Run the walk-forward backtest.
[[nodiscard]] BacktestSummary WalkForward(const std::vector<Observation>& observations,
                                          const BacktestConfig& config = {});

}  // namespace augury

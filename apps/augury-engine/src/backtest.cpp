#include "augury/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

#include "augury/lmsr.hpp"

namespace augury {
namespace {

/// Map S(t) in [-1, 1] onto a probability. The assumption-free affine mapping;
/// the Python service uses a Platt fit once enough markets have resolved, and
/// the backtest deliberately does not, because fitting a calibration on the
/// same history being backtested is lookahead.
double SignalToProbability(double signal) { return (signal + 1.0) / 2.0; }

}  // namespace

std::vector<double> SimulatePath(const std::vector<Observation>& observations, std::size_t start,
                                 std::size_t end, double b, double responsiveness) {
  std::vector<double> path;
  if (start >= end || end > observations.size()) {
    return path;
  }

  LmsrMarket market(b);
  // Anchor to the first real price so the opening step is not dominated by an
  // arbitrary 0.5 starting point.
  market.MoveToPrice(ClampProbability(observations[start].market_price));

  path.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) {
    market.StepFromSignal(SignalToProbability(observations[i].signal), responsiveness);
    path.push_back(market.Price());
  }
  return path;
}

double MeanSquaredError(const std::vector<double>& predicted, const std::vector<double>& actual) {
  if (predicted.size() != actual.size() || predicted.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double total = 0.0;
  for (std::size_t i = 0; i < predicted.size(); ++i) {
    const double error = predicted[i] - actual[i];
    total += error * error;
  }
  return total / static_cast<double>(predicted.size());
}

double DifferenceCorrelation(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size() || left.size() < 3) {
    return 0.0;
  }

  const std::size_t n = left.size() - 1;
  std::vector<double> dl(n);
  std::vector<double> dr(n);
  for (std::size_t i = 0; i < n; ++i) {
    dl[i] = left[i + 1] - left[i];
    dr[i] = right[i + 1] - right[i];
  }

  const double mean_l = std::accumulate(dl.begin(), dl.end(), 0.0) / static_cast<double>(n);
  const double mean_r = std::accumulate(dr.begin(), dr.end(), 0.0) / static_cast<double>(n);

  double cov = 0.0;
  double var_l = 0.0;
  double var_r = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = dl[i] - mean_l;
    const double b = dr[i] - mean_r;
    cov += a * b;
    var_l += a * a;
    var_r += b * b;
  }

  // A constant series has zero variance, making the correlation 0/0. Undefined
  // is reported as 0 rather than NaN so the summary arithmetic stays finite,
  // but it means "cannot tell", not "no relationship".
  if (var_l <= 0.0 || var_r <= 0.0) {
    return 0.0;
  }
  return cov / std::sqrt(var_l * var_r);
}

BacktestSummary WalkForward(const std::vector<Observation>& observations,
                            const BacktestConfig& config) {
  BacktestSummary summary;

  if (config.train_size == 0 || config.test_size == 0) {
    throw std::invalid_argument("train_size and test_size must be positive");
  }
  if (config.responsiveness_grid.empty()) {
    throw std::invalid_argument("responsiveness_grid must not be empty");
  }

  const std::size_t step = config.step.value_or(config.test_size);
  if (step == 0) {
    throw std::invalid_argument("step must be positive");
  }

  const std::size_t window = config.train_size + config.test_size;
  if (observations.size() < window) {
    // Not enough history for even one window. An empty summary is the honest
    // answer; inventing a single degenerate window would not be.
    return summary;
  }

  std::vector<double> b_values;

  for (std::size_t origin = 0; origin + window <= observations.size(); origin += step) {
    const std::size_t train_start = origin;
    const std::size_t train_end = origin + config.train_size;
    const std::size_t test_start = train_end;
    const std::size_t test_end = train_end + config.test_size;

    // Calibrate b from the *training* window's book only. Using the test
    // window's depth would leak information the simulation could not have had.
    double b = config.fallback_b;
    for (std::size_t i = train_end; i-- > train_start;) {
      const Observation& obs = observations[i];
      if (obs.yes_bid.has_value() && obs.yes_ask.has_value()) {
        b = CalibrateBFromDepth(obs.depth_bid, obs.depth_ask, obs.yes_bid, obs.yes_ask,
                                DepthSide::kMean, config.fallback_b);
        break;
      }
    }

    // Grid-search responsiveness on the training window.
    double best_responsiveness = config.responsiveness_grid.front();
    double best_train_mse = std::numeric_limits<double>::infinity();

    std::vector<double> train_actual;
    train_actual.reserve(config.train_size);
    for (std::size_t i = train_start; i < train_end; ++i) {
      train_actual.push_back(observations[i].market_price);
    }

    for (double candidate : config.responsiveness_grid) {
      const std::vector<double> path =
          SimulatePath(observations, train_start, train_end, b, candidate);
      const double mse = MeanSquaredError(path, train_actual);
      if (std::isfinite(mse) && mse < best_train_mse) {
        best_train_mse = mse;
        best_responsiveness = candidate;
      }
    }

    // Evaluate on the untouched test window with the fitted parameters.
    std::vector<double> test_actual;
    test_actual.reserve(config.test_size);
    for (std::size_t i = test_start; i < test_end; ++i) {
      test_actual.push_back(observations[i].market_price);
    }

    const std::vector<double> test_path =
        SimulatePath(observations, test_start, test_end, b, best_responsiveness);

    // Baseline: predict the last price seen before the test window began.
    const double last_train_price = observations[train_end - 1].market_price;
    const std::vector<double> baseline(test_actual.size(), last_train_price);

    WindowResult result;
    result.index = summary.windows.size();
    result.train_start = train_start;
    result.train_end = train_end;
    result.test_start = test_start;
    result.test_end = test_end;
    result.calibrated_b = b;
    result.responsiveness = best_responsiveness;
    result.mse = MeanSquaredError(test_path, test_actual);
    result.mse_baseline = MeanSquaredError(baseline, test_actual);
    result.correlation = DifferenceCorrelation(test_path, test_actual);
    result.n_test = test_actual.size();

    b_values.push_back(b);
    summary.windows.push_back(result);
  }

  if (summary.windows.empty()) {
    return summary;
  }

  double total_mse = 0.0;
  double total_baseline = 0.0;
  for (const WindowResult& window_result : summary.windows) {
    total_mse += window_result.mse;
    total_baseline += window_result.mse_baseline;
    if (window_result.BeatsBaseline()) {
      ++summary.windows_beating_baseline;
    }
  }

  const double count = static_cast<double>(summary.windows.size());
  summary.mean_mse = total_mse / count;
  summary.mean_baseline_mse = total_baseline / count;

  const auto [min_it, max_it] = std::minmax_element(b_values.begin(), b_values.end());
  summary.b_min = *min_it;
  summary.b_max = *max_it;
  summary.b_mean = std::accumulate(b_values.begin(), b_values.end(), 0.0) / count;

  double variance = 0.0;
  for (double value : b_values) {
    const double d = value - summary.b_mean;
    variance += d * d;
  }
  variance /= count;
  summary.b_coefficient_of_variation =
      summary.b_mean > 0.0 ? std::sqrt(variance) / summary.b_mean : 0.0;

  return summary;
}

std::string BacktestSummary::Report() const {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(6);

  if (windows.empty()) {
    out << "no complete walk-forward windows — not enough aligned history\n";
    return out.str();
  }

  out << "walk-forward backtest: " << windows.size() << " windows\n";
  out << "  mean MSE            : " << mean_mse << "\n";
  out << "  mean baseline MSE   : " << mean_baseline_mse << "\n";
  out << "  windows beating it  : " << windows_beating_baseline << " / " << windows.size() << "\n";
  out.precision(2);
  out << "  b range             : " << b_min << " .. " << b_max << " (mean " << b_mean << ")\n";
  out.precision(4);
  out << "  b coeff. of var.    : " << b_coefficient_of_variation << "\n";

  if (b_coefficient_of_variation > 0.5) {
    out << "\n  WARNING: b varies by more than 50% across windows. The depth\n";
    out << "  calibration is tracking noise rather than a stable liquidity\n";
    out << "  level, so the MSE figures above are not reproducible.\n";
  }
  if (windows_beating_baseline * 2 <= windows.size()) {
    out << "\n  NOTE: the simulation failed to beat a last-price baseline on at\n";
    out << "  least half the windows. That is a real result, not a bug — it says\n";
    out << "  the signal adds nothing over the market's own last print.\n";
  }

  return out.str();
}

}  // namespace augury

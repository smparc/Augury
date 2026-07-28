// Timing harness for the LMSR core.
//
// Separate from the Catch2 suite deliberately: a benchmark has no pass/fail
// condition, and wiring one into `ctest` would make CI fail whenever a build
// agent happened to be busy. This target is run by hand (`make bench`) when
// deciding whether an optimization was worth it.
//
// The numbers that matter here are per-operation costs. The backtester runs the
// LMSR step function once per observation per candidate responsiveness per
// walk-forward window, so a 300-observation series with a 6-value grid and 10
// windows is ~18,000 step calls — cheap individually, but the reason the
// simulation core is not in Python.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "augury/backtest.hpp"
#include "augury/lmsr.hpp"

namespace {

using Clock = std::chrono::steady_clock;

/// Run `iterations` of `body` and report nanoseconds per operation.
template <typename Fn>
void Benchmark(const std::string& name, std::size_t iterations, Fn&& body) {
  // One untimed pass so the first iteration's cold caches and lazy page faults
  // do not land in the measurement.
  body();

  const auto start = Clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    body();
  }
  const auto elapsed = Clock::now() - start;

  const double total_ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
  const double per_op = total_ns / static_cast<double>(iterations);

  std::printf("  %-38s %10.1f ns/op  (%zu iterations)\n", name.c_str(), per_op, iterations);
}

std::vector<augury::Observation> SyntheticSeries(std::size_t n) {
  std::vector<augury::Observation> series;
  series.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double price = 0.5 + 0.2 * std::sin(static_cast<double>(i) / 12.0);
    augury::Observation obs;
    obs.timestamp = static_cast<std::int64_t>(1'785'000'000 + i * 3600);
    obs.signal = std::sin((static_cast<double>(i) + 3.0) / 12.0);
    obs.market_price = price;
    obs.yes_bid = price - 0.01;
    obs.yes_ask = price + 0.01;
    obs.depth_bid = 500.0;
    obs.depth_ask = 500.0;
    series.push_back(obs);
  }
  return series;
}

// `volatile` sinks keep the optimizer from deleting the work being measured.
volatile double sink = 0.0;

}  // namespace

int main() {
  std::printf("augury engine benchmarks\n\n");

  const std::vector<double> q_binary{1234.0, -567.0};
  const std::vector<double> q_multi{120.0, -30.0, 45.0, 88.0, -12.0, 7.0, 300.0, -450.0};

  std::printf("cost function\n");
  Benchmark("Cost, 2 outcomes", 2'000'000,
            [&] { sink = augury::Cost(q_binary, 100.0); });
  Benchmark("Cost, 8 outcomes", 1'000'000,
            [&] { sink = augury::Cost(q_multi, 100.0); });
  // Exercises the max-subtraction path that keeps exp() from overflowing.
  Benchmark("Cost, extreme q/b (500)", 2'000'000, [&] {
    sink = augury::Cost(std::vector<double>{50000.0, 0.0}, 100.0);
  });

  std::printf("\nprices\n");
  Benchmark("PriceYes (sigmoid identity)", 5'000'000,
            [&] { sink = augury::PriceYes(1234.0, -567.0, 100.0); });
  Benchmark("Prices, 8 outcomes", 1'000'000, [&] {
    const std::vector<double> p = augury::Prices(q_multi, 100.0);
    sink = p[0];
  });

  std::printf("\nmarket operations\n");
  Benchmark("BuyYes", 1'000'000, [&] {
    augury::LmsrMarket market(100.0);
    sink = market.BuyYes(25.0);
  });
  Benchmark("StepFromSignal", 1'000'000, [&] {
    augury::LmsrMarket market(100.0);
    sink = market.StepFromSignal(0.7, 0.35);
  });
  Benchmark("CalibrateBFromDepth", 2'000'000, [&] {
    sink = augury::CalibrateBFromDepth(500.0, 500.0, 0.30, 0.35);
  });

  std::printf("\nbacktest\n");
  const auto series_300 = SyntheticSeries(300);
  const auto series_5000 = SyntheticSeries(5000);

  Benchmark("SimulatePath, 300 observations", 2'000, [&] {
    const std::vector<double> path =
        augury::SimulatePath(series_300, 0, series_300.size(), 500.0, 0.35);
    sink = path.empty() ? 0.0 : path.back();
  });

  Benchmark("WalkForward, 300 observations", 200, [&] {
    const augury::BacktestSummary summary = augury::WalkForward(series_300);
    sink = summary.mean_mse;
  });

  Benchmark("WalkForward, 5000 observations", 10, [&] {
    const augury::BacktestSummary summary = augury::WalkForward(series_5000);
    sink = summary.mean_mse;
  });

  std::printf("\n(sink = %g — printed only to keep the optimizer honest)\n",
              static_cast<double>(sink));
  return 0;
}

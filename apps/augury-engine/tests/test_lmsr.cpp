// Correctness gate for the C++ engine.
//
// The golden-vector cases here are the point of this file: they assert that
// this implementation reproduces the Python reference in
// `apps/augury-signal/augury_signal/engine/lmsr.py` to 1e-9 on a fixed set of
// inputs. Two independent implementations of the same equations agreeing to
// nine digits is meaningful evidence that both are right; either one agreeing
// with itself is not.
//
// If these fail after an intentional change to the Python side, regenerate with
// `python -m augury_signal.golden` and re-read the diff before accepting it.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

#include "augury/backtest.hpp"
#include "augury/lmsr.hpp"

using augury::Cost;
using augury::LmsrMarket;
using augury::PriceYes;
using Catch::Approx;

namespace {

constexpr double kTolerance = 1e-9;

/// Locate `schemas/testdata/golden_vectors.json` by walking up from the build
/// directory, so the tests work regardless of where CMake puts the binary.
std::filesystem::path GoldenVectorPath() {
  std::filesystem::path current = std::filesystem::current_path();
  for (int depth = 0; depth < 8; ++depth) {
    const std::filesystem::path candidate =
        current / "schemas" / "testdata" / "golden_vectors.json";
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
    if (!current.has_parent_path() || current.parent_path() == current) {
      break;
    }
    current = current.parent_path();
  }
  return {};
}

const nlohmann::json& GoldenVectors() {
  static const nlohmann::json data = [] {
    const std::filesystem::path path = GoldenVectorPath();
    REQUIRE_FALSE(path.empty());
    std::ifstream stream(path);
    REQUIRE(stream.is_open());
    nlohmann::json parsed;
    stream >> parsed;
    return parsed;
  }();
  return data;
}

std::optional<double> OptionalDouble(const nlohmann::json& value) {
  if (value.is_null()) {
    return std::nullopt;
  }
  return value.get<double>();
}

augury::DepthSide ParseSide(const std::string& name) {
  if (name == "bid") return augury::DepthSide::kBid;
  if (name == "ask") return augury::DepthSide::kAsk;
  return augury::DepthSide::kMean;
}

}  // namespace

// ---------------------------------------------------------------------------
// Golden vectors — cross-language agreement with the Python reference
// ---------------------------------------------------------------------------

TEST_CASE("golden vectors: cost and prices", "[golden][lmsr]") {
  for (const auto& entry : GoldenVectors()["lmsr"]["cost"]) {
    const auto q = entry["q"].get<std::vector<double>>();
    const double b = entry["b"].get<double>();

    CHECK(Cost(q, b) == Approx(entry["expected_cost"].get<double>()).margin(kTolerance));

    const auto expected_prices = entry["expected_prices"].get<std::vector<double>>();
    const std::vector<double> got = augury::Prices(q, b);
    REQUIRE(got.size() == expected_prices.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i] == Approx(expected_prices[i]).margin(kTolerance));
    }
  }
}

TEST_CASE("golden vectors: trade cost", "[golden][lmsr]") {
  for (const auto& entry : GoldenVectors()["lmsr"]["trade"]) {
    const auto q = entry["q"].get<std::vector<double>>();
    const auto dq = entry["dq"].get<std::vector<double>>();
    const double b = entry["b"].get<double>();
    CHECK(augury::TradeCost(q, dq, b) ==
          Approx(entry["expected_cost"].get<double>()).margin(kTolerance));
  }
}

TEST_CASE("golden vectors: binary price", "[golden][lmsr]") {
  for (const auto& entry : GoldenVectors()["lmsr"]["binary_price"]) {
    CHECK(PriceYes(entry["q_yes"].get<double>(), entry["q_no"].get<double>(),
                   entry["b"].get<double>()) ==
          Approx(entry["expected_price_yes"].get<double>()).margin(kTolerance));
  }
}

TEST_CASE("golden vectors: shares to move price", "[golden][lmsr]") {
  for (const auto& entry : GoldenVectors()["lmsr"]["shares_to_move"]) {
    CHECK(augury::SharesToMovePrice(entry["p_from"].get<double>(), entry["p_to"].get<double>(),
                                    entry["b"].get<double>()) ==
          Approx(entry["expected_shares"].get<double>()).margin(kTolerance));
  }
}

TEST_CASE("golden vectors: b calibration from book depth", "[golden][lmsr]") {
  for (const auto& entry : GoldenVectors()["lmsr"]["b_calibration"]) {
    const double got = augury::CalibrateBFromDepth(
        OptionalDouble(entry["depth_bid"]), OptionalDouble(entry["depth_ask"]),
        OptionalDouble(entry["yes_bid"]), OptionalDouble(entry["yes_ask"]),
        ParseSide(entry["side"].get<std::string>()), 100.0);
    CHECK(got == Approx(entry["expected_b"].get<double>()).margin(kTolerance));
  }
}

// ---------------------------------------------------------------------------
// Invariants — the same properties asserted by the Python suite
// ---------------------------------------------------------------------------

TEST_CASE("cost of a uniform state is b*ln(K)", "[lmsr]") {
  CHECK(Cost(std::vector<double>{0.0, 0.0}, 100.0) == Approx(100.0 * std::log(2.0)));
  CHECK(Cost(std::vector<double>{0.0, 0.0, 0.0, 0.0}, 50.0) == Approx(50.0 * std::log(4.0)));
}

TEST_CASE("cost does not overflow at realistic scale", "[lmsr]") {
  // q/b = 500. The naive formula evaluates exp(500), which is inf in IEEE
  // double, and every price derived from it becomes NaN.
  const double value = Cost(std::vector<double>{50000.0, 0.0}, 100.0);
  REQUIRE(std::isfinite(value));
  CHECK(value == Approx(50000.0).epsilon(1e-9));
}

TEST_CASE("cost is translation invariant", "[lmsr]") {
  const double base = Cost(std::vector<double>{10.0, -5.0}, 25.0);
  CHECK(Cost(std::vector<double>{110.0, 95.0}, 25.0) == Approx(base + 100.0));
}

TEST_CASE("prices form a probability distribution", "[lmsr]") {
  const std::vector<double> p = augury::Prices(std::vector<double>{120.0, -30.0, 45.0}, 40.0);
  double total = 0.0;
  for (double value : p) {
    CHECK(value > 0.0);
    CHECK(value < 1.0);
    total += value;
  }
  CHECK(total == Approx(1.0));
}

TEST_CASE("logit is linear in share imbalance", "[lmsr]") {
  const double b = 80.0;
  for (const auto& [q_yes, q_no] : std::vector<std::pair<double, double>>{
           {0.0, 0.0}, {100.0, 20.0}, {-60.0, 15.0}}) {
    const double p = PriceYes(q_yes, q_no, b);
    CHECK(std::log(p / (1.0 - p)) == Approx((q_yes - q_no) / b).margin(1e-9));
  }
}

TEST_CASE("a round trip is free up to float error", "[lmsr]") {
  LmsrMarket market(100.0);
  const double paid = market.BuyYes(75.0);
  const double refund = market.BuyYes(-75.0);
  CHECK(paid + refund == Approx(0.0).margin(1e-9));
  CHECK(market.Price() == Approx(0.5));
}

TEST_CASE("a YES share never costs more than 1", "[lmsr]") {
  // It pays at most 1 on resolution, so paying more would be arbitrage.
  LmsrMarket market(100.0);
  const double shares = 250.0;
  CHECK(market.BuyYes(shares) < shares);
}

TEST_CASE("moving to a target price is exact", "[lmsr]") {
  LmsrMarket market(60.0, 30.0, 0.0);
  for (double target : {0.1, 0.5, 0.97}) {
    market.MoveToPrice(target);
    CHECK(market.Price() == Approx(target).margin(1e-9));
  }
}

TEST_CASE("recalibrating b preserves the price", "[lmsr]") {
  // A liquidity update is not new information about the outcome; if it moved
  // the price it would inject signal nobody traded on.
  LmsrMarket market(100.0);
  market.MoveToPrice(0.73);
  const double before = market.Price();
  market.Recalibrate(925.0);
  CHECK(market.Price() == Approx(before).margin(1e-12));
  CHECK(market.b() == Approx(925.0));
}

TEST_CASE("higher b means less price impact", "[lmsr]") {
  LmsrMarket thin(10.0);
  LmsrMarket thick(10000.0);
  thin.BuyYes(100.0);
  thick.BuyYes(100.0);
  CHECK(std::abs(thin.Price() - 0.5) > std::abs(thick.Price() - 0.5));
}

TEST_CASE("calibrated b reproduces the observed price move", "[lmsr]") {
  // The whole justification for the formula: consuming `depth` shares should
  // walk the price from bid to ask.
  const double bid = 0.20;
  const double ask = 0.26;
  const double depth = 800.0;
  const double b = augury::CalibrateBFromDepth(depth, depth, bid, ask);

  LmsrMarket market(b);
  market.MoveToPrice(bid);
  market.BuyYes(depth);
  CHECK(market.Price() == Approx(ask).margin(1e-9));
}

TEST_CASE("unusable books fall back rather than throwing", "[lmsr]") {
  // Thin prediction markets produce these constantly.
  CHECK(augury::CalibrateBFromDepth(100.0, 100.0, 0.6, 0.4, augury::DepthSide::kMean, 42.0) == 42.0);
  CHECK(augury::CalibrateBFromDepth(100.0, 100.0, 0.5, 0.5, augury::DepthSide::kMean, 42.0) == 42.0);
  CHECK(augury::CalibrateBFromDepth(std::nullopt, std::nullopt, 0.3, 0.4,
                                    augury::DepthSide::kMean, 42.0) == 42.0);
  CHECK(augury::CalibrateBFromDepth(100.0, 100.0, std::nullopt, 0.4, augury::DepthSide::kMean,
                                    42.0) == 42.0);
}

TEST_CASE("invalid inputs are rejected", "[lmsr]") {
  CHECK_THROWS_AS(Cost(std::vector<double>{1.0}, 10.0), std::invalid_argument);
  CHECK_THROWS_AS(Cost(std::vector<double>{1.0, 2.0}, 0.0), std::invalid_argument);
  CHECK_THROWS_AS(LmsrMarket(100.0).StepFromSignal(0.6, 1.5), std::invalid_argument);
  CHECK_THROWS_AS(LmsrMarket(100.0).Recalibrate(0.0), std::invalid_argument);
}

TEST_CASE("signal steps converge on the target", "[lmsr]") {
  LmsrMarket market(100.0);
  for (int i = 0; i < 100; ++i) {
    market.StepFromSignal(0.7, 0.3);
  }
  CHECK(market.Price() == Approx(0.7).margin(1e-6));
}

TEST_CASE("max_shares caps a single step", "[lmsr]") {
  LmsrMarket market(100.0);
  const double shares = market.StepFromSignal(0.999, 1.0, 10.0);
  CHECK(shares == Approx(10.0));
}

// ---------------------------------------------------------------------------
// Backtester
// ---------------------------------------------------------------------------

namespace {

std::vector<augury::Observation> SyntheticSeries(std::size_t n) {
  std::vector<augury::Observation> series;
  series.reserve(n);
  double price = 0.5;
  for (std::size_t i = 0; i < n; ++i) {
    // Price follows a slow sine; the signal leads it by a few steps.
    price = 0.5 + 0.2 * std::sin(static_cast<double>(i) / 12.0);
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

}  // namespace

TEST_CASE("walk-forward produces multiple windows", "[backtest]") {
  const auto series = SyntheticSeries(300);
  augury::BacktestConfig config;
  config.train_size = 72;
  config.test_size = 24;

  const augury::BacktestSummary summary = augury::WalkForward(series, config);
  REQUIRE(summary.windows.size() > 1);
  CHECK(summary.b_min > 0.0);
  CHECK(summary.b_max >= summary.b_min);
  CHECK_FALSE(summary.Report().empty());
}

TEST_CASE("walk-forward test windows do not overlap by default", "[backtest]") {
  const auto series = SyntheticSeries(300);
  const augury::BacktestSummary summary = augury::WalkForward(series);
  for (std::size_t i = 1; i < summary.windows.size(); ++i) {
    CHECK(summary.windows[i].test_start >= summary.windows[i - 1].test_end);
  }
}

TEST_CASE("too little history yields no windows rather than a fake one", "[backtest]") {
  const auto series = SyntheticSeries(10);
  const augury::BacktestSummary summary = augury::WalkForward(series);
  CHECK(summary.windows.empty());
  CHECK(summary.Report().find("not enough") != std::string::npos);
}

TEST_CASE("mean squared error basics", "[backtest]") {
  CHECK(augury::MeanSquaredError({0.5, 0.5}, {0.5, 0.5}) == Approx(0.0));
  CHECK(augury::MeanSquaredError({1.0, 0.0}, {0.0, 1.0}) == Approx(1.0));
  CHECK(std::isnan(augury::MeanSquaredError({0.5}, {0.5, 0.5})));
}

TEST_CASE("difference correlation is undefined for constant series", "[backtest]") {
  CHECK(augury::DifferenceCorrelation({0.5, 0.5, 0.5, 0.5}, {0.1, 0.2, 0.3, 0.4}) == Approx(0.0));
}

TEST_CASE("difference correlation detects a perfect match", "[backtest]") {
  const std::vector<double> series{0.1, 0.2, 0.35, 0.5, 0.62, 0.7};
  CHECK(augury::DifferenceCorrelation(series, series) == Approx(1.0).margin(1e-9));
}

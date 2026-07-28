// augury_engine — LMSR simulation and walk-forward backtesting.
//
// Reads aligned observations as JSON (produced by the Python service or dumped
// from TimescaleDB) and either simulates a price path or runs a walk-forward
// backtest. Deliberately file-in/file-out rather than embedding a database or
// Redis client: the numerical core is the part worth writing in C++, and
// keeping the I/O boundary at JSON means this binary is trivially testable and
// has no transport dependencies to break.
//
// Input format — a JSON array of objects:
//
//   [{"timestamp": 1785000000, "signal": 0.42, "market_price": 0.31,
//     "yes_bid": 0.30, "yes_ask": 0.32, "depth_bid": 500, "depth_ask": 480}, ...]

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "augury/backtest.hpp"
#include "augury/lmsr.hpp"

namespace {

void PrintUsage() {
  std::cerr << R"(augury_engine — LMSR simulation and backtesting

usage:
  augury_engine simulate   <observations.json> [--responsiveness R] [--b B]
  augury_engine backtest   <observations.json> [--train N] [--test N]
  augury_engine price      --q-yes Q --q-no Q --b B
  augury_engine calibrate  --bid P --ask P [--depth-bid D] [--depth-ask D]

Observations JSON is an array of objects with at least `signal` and
`market_price`; `yes_bid`/`yes_ask`/`depth_bid`/`depth_ask` enable
depth-calibrated liquidity.
)";
}

std::optional<double> OptionalField(const nlohmann::json& obj, const char* key) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return std::nullopt;
  }
  return obj.at(key).get<double>();
}

std::vector<augury::Observation> LoadObservations(const std::string& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("cannot open " + path);
  }

  nlohmann::json parsed;
  stream >> parsed;
  if (!parsed.is_array()) {
    throw std::runtime_error(path + ": expected a JSON array of observations");
  }

  std::vector<augury::Observation> observations;
  observations.reserve(parsed.size());

  for (const auto& entry : parsed) {
    augury::Observation obs;
    obs.timestamp = entry.value("timestamp", static_cast<std::int64_t>(0));
    obs.signal = entry.at("signal").get<double>();
    obs.market_price = entry.at("market_price").get<double>();
    obs.yes_bid = OptionalField(entry, "yes_bid");
    obs.yes_ask = OptionalField(entry, "yes_ask");
    obs.depth_bid = OptionalField(entry, "depth_bid");
    obs.depth_ask = OptionalField(entry, "depth_ask");
    observations.push_back(obs);
  }
  return observations;
}

std::optional<std::string> FlagValue(const std::vector<std::string>& args, const std::string& flag) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == flag) {
      return args[i + 1];
    }
  }
  return std::nullopt;
}

double FlagDouble(const std::vector<std::string>& args, const std::string& flag, double fallback) {
  const auto value = FlagValue(args, flag);
  return value.has_value() ? std::stod(*value) : fallback;
}

int RunSimulate(const std::vector<std::string>& args) {
  if (args.size() < 2) {
    PrintUsage();
    return 2;
  }

  const std::vector<augury::Observation> observations = LoadObservations(args[1]);
  if (observations.empty()) {
    std::cerr << "no observations\n";
    return 1;
  }

  const double responsiveness = FlagDouble(args, "--responsiveness", 0.35);

  double b = FlagDouble(args, "--b", -1.0);
  if (b <= 0.0) {
    // Calibrate from the most recent usable book rather than guessing.
    b = 100.0;
    for (std::size_t i = observations.size(); i-- > 0;) {
      if (observations[i].yes_bid.has_value() && observations[i].yes_ask.has_value()) {
        b = augury::CalibrateBFromDepth(observations[i].depth_bid, observations[i].depth_ask,
                                        observations[i].yes_bid, observations[i].yes_ask);
        break;
      }
    }
  }

  const std::vector<double> path =
      augury::SimulatePath(observations, 0, observations.size(), b, responsiveness);

  nlohmann::json out = nlohmann::json::array();
  for (std::size_t i = 0; i < path.size(); ++i) {
    out.push_back({{"timestamp", observations[i].timestamp},
                   {"sim_price", path[i]},
                   {"market_price", observations[i].market_price},
                   {"signal", observations[i].signal},
                   {"b", b}});
  }

  std::cout << out.dump(2) << "\n";
  std::cerr << "simulated " << path.size() << " steps with b=" << b
            << " responsiveness=" << responsiveness << "\n";
  return 0;
}

int RunBacktest(const std::vector<std::string>& args) {
  if (args.size() < 2) {
    PrintUsage();
    return 2;
  }

  const std::vector<augury::Observation> observations = LoadObservations(args[1]);

  augury::BacktestConfig config;
  config.train_size = static_cast<std::size_t>(FlagDouble(args, "--train", 72));
  config.test_size = static_cast<std::size_t>(FlagDouble(args, "--test", 24));

  const augury::BacktestSummary summary = augury::WalkForward(observations, config);
  std::cout << summary.Report();

  if (!summary.windows.empty()) {
    std::cout << "\nper-window:\n";
    for (const augury::WindowResult& window : summary.windows) {
      std::cout << "  [" << window.index << "] b=" << window.calibrated_b
                << " resp=" << window.responsiveness << " mse=" << window.mse
                << " baseline=" << window.mse_baseline << " corr=" << window.correlation
                << (window.BeatsBaseline() ? "  (beats baseline)" : "") << "\n";
    }
  }
  return 0;
}

int RunPrice(const std::vector<std::string>& args) {
  const double q_yes = FlagDouble(args, "--q-yes", 0.0);
  const double q_no = FlagDouble(args, "--q-no", 0.0);
  const double b = FlagDouble(args, "--b", 100.0);

  const std::vector<double> q{q_yes, q_no};
  std::cout << nlohmann::json{{"price_yes", augury::PriceYes(q_yes, q_no, b)},
                              {"cost", augury::Cost(q, b)},
                              {"worst_case_subsidy", augury::WorstCaseSubsidy(2, b)},
                              {"b", b}}
                   .dump(2)
            << "\n";
  return 0;
}

int RunCalibrate(const std::vector<std::string>& args) {
  const auto bid = FlagValue(args, "--bid");
  const auto ask = FlagValue(args, "--ask");
  if (!bid || !ask) {
    PrintUsage();
    return 2;
  }

  const auto depth_bid = FlagValue(args, "--depth-bid");
  const auto depth_ask = FlagValue(args, "--depth-ask");

  const double b = augury::CalibrateBFromDepth(
      depth_bid ? std::optional<double>(std::stod(*depth_bid)) : std::nullopt,
      depth_ask ? std::optional<double>(std::stod(*depth_ask)) : std::nullopt, std::stod(*bid),
      std::stod(*ask));

  std::cout << nlohmann::json{{"b", b},
                              {"worst_case_subsidy", augury::WorstCaseSubsidy(2, b)}}
                   .dump(2)
            << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) {
    PrintUsage();
    return 2;
  }

  try {
    const std::string& command = args[0];
    if (command == "simulate") return RunSimulate(args);
    if (command == "backtest") return RunBacktest(args);
    if (command == "price") return RunPrice(args);
    if (command == "calibrate") return RunCalibrate(args);

    std::cerr << "unknown command: " << command << "\n\n";
    PrintUsage();
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}

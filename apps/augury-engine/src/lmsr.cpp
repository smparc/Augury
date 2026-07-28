#include "augury/lmsr.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace augury {
namespace {

void ValidateState(std::span<const double> q, double b) {
  if (q.size() < 2) {
    throw std::invalid_argument("LMSR needs at least 2 outcomes, got " +
                                std::to_string(q.size()));
  }
  if (b <= 0.0) {
    throw std::invalid_argument("liquidity parameter b must be positive, got " +
                                std::to_string(b));
  }
  for (double value : q) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("outstanding shares must all be finite");
    }
  }
}

}  // namespace

double LogSumExp(std::span<const double> values) {
  if (values.empty()) {
    throw std::invalid_argument("log_sum_exp of an empty sequence");
  }

  const double peak = *std::max_element(values.begin(), values.end());
  if (peak == -std::numeric_limits<double>::infinity()) {
    return peak;
  }

  double total = 0.0;
  for (double value : values) {
    total += std::exp(value - peak);
  }
  return peak + std::log(total);
}

double Sigmoid(double x) {
  // Branching on the sign keeps exp() away from overflow: exp(+710) is inf in
  // IEEE double, while exp(-710) merely underflows to 0, which is harmless.
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

double ClampProbability(double p) { return std::min(1.0 - kEps, std::max(kEps, p)); }

double Logit(double p) {
  const double q = ClampProbability(p);
  return std::log(q / (1.0 - q));
}

double Cost(std::span<const double> q, double b) {
  ValidateState(q, b);

  std::vector<double> scaled;
  scaled.reserve(q.size());
  for (double value : q) {
    scaled.push_back(value / b);
  }
  return b * LogSumExp(scaled);
}

std::vector<double> Prices(std::span<const double> q, double b) {
  ValidateState(q, b);

  std::vector<double> scaled;
  scaled.reserve(q.size());
  for (double value : q) {
    scaled.push_back(value / b);
  }

  const double peak = *std::max_element(scaled.begin(), scaled.end());

  std::vector<double> exps;
  exps.reserve(scaled.size());
  double total = 0.0;
  for (double value : scaled) {
    const double e = std::exp(value - peak);
    exps.push_back(e);
    total += e;
  }

  for (double& value : exps) {
    value /= total;
  }
  return exps;
}

double PriceYes(double q_yes, double q_no, double b) {
  const std::array<double, 2> q{q_yes, q_no};
  ValidateState(q, b);
  return Sigmoid((q_yes - q_no) / b);
}

double TradeCost(std::span<const double> q, std::span<const double> dq, double b) {
  if (q.size() != dq.size()) {
    throw std::invalid_argument("q and dq differ in length");
  }

  std::vector<double> after;
  after.reserve(q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    after.push_back(q[i] + dq[i]);
  }
  return Cost(after, b) - Cost(q, b);
}

double SharesToMovePrice(double p_from, double p_to, double b) {
  if (b <= 0.0) {
    throw std::invalid_argument("liquidity parameter b must be positive");
  }
  return b * (Logit(p_to) - Logit(p_from));
}

double WorstCaseSubsidy(std::size_t n_outcomes, double b) {
  if (n_outcomes < 2) {
    throw std::invalid_argument("need at least 2 outcomes");
  }
  if (b <= 0.0) {
    throw std::invalid_argument("liquidity parameter b must be positive");
  }
  return b * std::log(static_cast<double>(n_outcomes));
}

double CalibrateBFromDepth(std::optional<double> depth_bid, std::optional<double> depth_ask,
                           std::optional<double> yes_bid, std::optional<double> yes_ask,
                           DepthSide side, double fallback) {
  if (fallback <= 0.0) {
    throw std::invalid_argument("fallback must be positive");
  }
  if (!yes_bid.has_value() || !yes_ask.has_value()) {
    return fallback;
  }

  const double bid = *yes_bid;
  const double ask = *yes_ask;
  // Crossed, locked, or out of range: no usable width to calibrate against.
  if (!(bid >= 0.0 && bid < ask && ask <= 1.0)) {
    return fallback;
  }

  const double width = Logit(ask) - Logit(bid);
  if (width <= 0.0) {
    return fallback;
  }

  double sum = 0.0;
  int count = 0;
  if ((side == DepthSide::kAsk || side == DepthSide::kMean) && depth_ask.has_value() &&
      *depth_ask > 0.0) {
    sum += *depth_ask;
    ++count;
  }
  if ((side == DepthSide::kBid || side == DepthSide::kMean) && depth_bid.has_value() &&
      *depth_bid > 0.0) {
    sum += *depth_bid;
    ++count;
  }

  if (count == 0) {
    return fallback;
  }

  const double depth = sum / static_cast<double>(count);
  if (depth <= 0.0) {
    return fallback;
  }

  return std::min(kBMax, std::max(kBMin, depth / width));
}

LmsrMarket::LmsrMarket(double b, double q_yes, double q_no)
    : b_(b), q_yes_(q_yes), q_no_(q_no) {
  const std::array<double, 2> q{q_yes, q_no};
  ValidateState(q, b);
}

double LmsrMarket::Price() const { return PriceYes(q_yes_, q_no_, b_); }

double LmsrMarket::CostValue() const {
  const std::array<double, 2> q{q_yes_, q_no_};
  return Cost(q, b_);
}

double LmsrMarket::BuyYes(double shares) {
  if (!std::isfinite(shares)) {
    throw std::invalid_argument("shares must be finite");
  }

  const std::array<double, 2> q{q_yes_, q_no_};
  const std::array<double, 2> dq{shares, 0.0};
  const double paid = TradeCost(q, dq, b_);

  q_yes_ += shares;
  realized_cost_ += paid;
  return paid;
}

double LmsrMarket::MoveToPrice(double target) {
  const double clamped = ClampProbability(target);
  const double shares = SharesToMovePrice(Price(), clamped, b_);
  BuyYes(shares);
  return shares;
}

void LmsrMarket::Recalibrate(double new_b) {
  if (new_b <= 0.0) {
    throw std::invalid_argument("liquidity parameter b must be positive");
  }
  // logit(p) = (q_yes - q_no) / b, so preserving p under a change of b means
  // scaling the imbalance by the same ratio.
  const double ratio = new_b / b_;
  const double imbalance = (q_yes_ - q_no_) * ratio;
  // Keep the representation canonical: all imbalance on the YES leg.
  q_yes_ = imbalance;
  q_no_ = 0.0;
  b_ = new_b;
}

double LmsrMarket::StepFromSignal(double p_target, double responsiveness,
                                  std::optional<double> max_shares) {
  if (!(responsiveness > 0.0 && responsiveness <= 1.0)) {
    throw std::invalid_argument("responsiveness must be in (0, 1]");
  }

  const double target = ClampProbability(p_target);
  const double gap = Logit(target) - Logit(Price());
  double shares = b_ * gap * responsiveness;

  if (max_shares.has_value()) {
    if (*max_shares < 0.0) {
      throw std::invalid_argument("max_shares must be non-negative");
    }
    shares = std::max(-*max_shares, std::min(*max_shares, shares));
  }

  BuyYes(shares);
  return shares;
}

}  // namespace augury

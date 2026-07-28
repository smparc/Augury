// Hanson's Logarithmic Market Scoring Rule.
//
//     C(q) = b * ln( sum_j exp(q_j / b) )
//     p_k  = exp(q_k / b) / sum_j exp(q_j / b)
//
// This is a second, independent implementation of the equations in
// `apps/augury-signal/augury_signal/engine/lmsr.py`. The two are held together
// by the golden vectors in `schemas/testdata/golden_vectors.json`, which the
// Catch2 suite asserts this code reproduces to 1e-9. Independent
// implementations of a formula agree only by luck unless something forces the
// issue; that file is the forcing function.
//
// Every exponential goes through a max-subtracted log-sum-exp. That is a
// correctness requirement, not a style preference: with `b` calibrated from
// real book depth and share counts in the thousands, `q_j / b` reaches several
// hundred, and `std::exp` overflows to infinity in IEEE double well before
// that. The naive formula returns inf or NaN on entirely ordinary inputs.

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace augury {

/// Below this the market is so thin a single share crosses the whole price
/// range; above it, no realistic trade moves the price at all. Both extremes
/// make the simulation useless, so calibration is clamped into this band.
inline constexpr double kBMin = 1.0;
inline constexpr double kBMax = 1.0e6;

/// Probabilities are held off {0, 1} before any log or logit. LMSR reaches the
/// boundaries only in the limit, so an exact 0 or 1 means infinite shares.
inline constexpr double kEps = 1e-6;

/// Numerically stable ln(sum(exp(v))).
[[nodiscard]] double LogSumExp(std::span<const double> values);

/// Stable logistic function.
[[nodiscard]] double Sigmoid(double x);

/// ln(p / (1 - p)), with p clamped off the boundaries.
[[nodiscard]] double Logit(double p);

/// Clamp a probability into [kEps, 1 - kEps].
[[nodiscard]] double ClampProbability(double p);

/// Cost function C(q) = b * ln(sum_j exp(q_j / b)).
[[nodiscard]] double Cost(std::span<const double> q, double b);

/// Price vector: a proper probability distribution summing to 1.
[[nodiscard]] std::vector<double> Prices(std::span<const double> q, double b);

/// Binary YES price via the sigmoid identity: p = sigmoid((q_yes - q_no) / b).
[[nodiscard]] double PriceYes(double q_yes, double q_no, double b);

/// Cost of moving the market from q to q + dq. Positive means the trader pays.
[[nodiscard]] double TradeCost(std::span<const double> q, std::span<const double> dq, double b);

/// YES shares needed to move a binary price: n = b * (logit(to) - logit(from)).
[[nodiscard]] double SharesToMovePrice(double p_from, double p_to, double b);

/// Maximum the market maker can lose: b * ln(K).
///
/// The other half of what `b` controls. Raising b to smooth the synthetic
/// market directly raises the subsidy being committed, which is why b is
/// calibrated from observed depth rather than turned up until the plot looks
/// nice.
[[nodiscard]] double WorstCaseSubsidy(std::size_t n_outcomes, double b);

/// Which side of the book to calibrate `b` against.
enum class DepthSide { kBid, kAsk, kMean };

/// Fit `b` so the synthetic market is as hard to move as the real one.
///
/// In the real book, consuming the size resting between bid and ask walks the
/// price across the quoted range. In LMSR the same move costs
/// `b * (logit(ask) - logit(bid))` shares. Setting those equal gives
///
///     b = depth / (logit(ask) - logit(bid))
///
/// Real books are severely lopsided — the Kalshi sample this was built against
/// had 40 shares on the bid against ~69,000 on the ask — so `side` genuinely
/// changes the answer, and kMean is a symmetric default rather than an
/// obviously correct one.
///
/// Returns `fallback` for an unusable book (crossed, locked, one-sided, or zero
/// depth). Those occur routinely on thin prediction markets and are not worth
/// throwing over mid-simulation.
[[nodiscard]] double CalibrateBFromDepth(std::optional<double> depth_bid,
                                         std::optional<double> depth_ask,
                                         std::optional<double> yes_bid,
                                         std::optional<double> yes_ask,
                                         DepthSide side = DepthSide::kMean,
                                         double fallback = 100.0);

/// A binary LMSR market maker with mutable state.
class LmsrMarket {
 public:
  explicit LmsrMarket(double b = 100.0, double q_yes = 0.0, double q_no = 0.0);

  [[nodiscard]] double b() const noexcept { return b_; }
  [[nodiscard]] double q_yes() const noexcept { return q_yes_; }
  [[nodiscard]] double q_no() const noexcept { return q_no_; }
  [[nodiscard]] double realized_cost() const noexcept { return realized_cost_; }

  /// Current YES price.
  [[nodiscard]] double Price() const;

  /// Value of C(q) at the current state.
  [[nodiscard]] double CostValue() const;

  /// Buy `shares` of YES (negative sells). Returns the cost paid.
  double BuyYes(double shares);

  /// Trade exactly enough to set the YES price to `target`.
  /// Returns the shares traded.
  double MoveToPrice(double target);

  /// Update `b` while holding the current price fixed.
  ///
  /// A liquidity update is not new information about the outcome. Letting it
  /// move the price would inject signal nobody traded on, so the share
  /// imbalance is rescaled to preserve logit(p) = (q_yes - q_no) / b.
  void Recalibrate(double new_b);

  /// Move the price part of the way toward the signal's forecast.
  ///
  /// `responsiveness` in (0, 1] is the fraction of the logit-space gap closed
  /// in one step. At 1.0 the synthetic price is just a restatement of the
  /// signal; below 1.0 the market integrates it over time, which is the
  /// behavior actually worth comparing against a real order book.
  double StepFromSignal(double p_target, double responsiveness = 1.0,
                        std::optional<double> max_shares = std::nullopt);

 private:
  double b_;
  double q_yes_;
  double q_no_;
  double realized_cost_ = 0.0;
};

}  // namespace augury

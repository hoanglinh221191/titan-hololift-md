#include "temporal_path.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

using Gram = std::array<double, 6>;
using Real3 = std::array<double, 3>;

double gram_value(const Gram &gram, std::size_t row, std::size_t column) {
  if (row > column)
    std::swap(row, column);
  if (row == 0)
    return column == 0 ? gram[0] : (column == 1 ? gram[1] : gram[2]);
  if (row == 1)
    return column == 1 ? gram[3] : gram[4];
  return gram[5];
}

double metric_norm2(const Real3 &value, const Gram &gram) {
  return value[0] * value[0] * gram[0] +
         2.0 * value[0] * value[1] * gram[1] +
         2.0 * value[0] * value[2] * gram[2] +
         value[1] * value[1] * gram[3] +
         2.0 * value[1] * value[2] * gram[4] +
         value[2] * value[2] * gram[5];
}

bool cholesky_upper(const Gram &gram, double upper[3][3]) {
  double lower[3][3]{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = gram_value(gram, row, column);
      for (std::size_t k = 0; k < column; ++k)
        value -= lower[row][k] * lower[column][k];
      if (row == column) {
        if (!std::isfinite(value) || value <= 1.0e-12)
          return false;
        lower[row][column] = std::sqrt(value);
      } else {
        lower[row][column] = value / lower[column][column];
      }
    }
  }
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t column = row; column < 3; ++column)
      upper[row][column] = lower[column][row];
  return true;
}

std::array<long long, 3> closest_metric_integer(const Real3 &value,
                                                const Gram &gram) {
  std::array<long long, 3> best{
      std::llround(value[0]), std::llround(value[1]),
      std::llround(value[2])};
  auto residual_for = [&](const std::array<long long, 3> &integer) {
    return Real3{value[0] - static_cast<double>(integer[0]),
                 value[1] - static_cast<double>(integer[1]),
                 value[2] - static_cast<double>(integer[2])};
  };
  double best_cost = metric_norm2(residual_for(best), gram);
  double upper[3][3]{};
  if (!std::isfinite(best_cost) || !cholesky_upper(gram, upper))
    return best;

  std::array<long long, 3> trial = best;
  std::function<void(int, double)> visit = [&](int axis, double partial_cost) {
    if (axis < 0) {
      if (partial_cost < best_cost) {
        best_cost = partial_cost;
        best = trial;
      }
      return;
    }
    double tail = 0.0;
    for (int column = axis + 1; column < 3; ++column) {
      tail += upper[axis][column] *
              (static_cast<double>(trial[column]) - value[column]);
    }
    const double diagonal = upper[axis][axis];
    const double center = value[axis] - tail / diagonal;
    const double remaining = std::max(0.0, best_cost - partial_cost);
    const double radius = std::sqrt(remaining) / std::fabs(diagonal) + 1.0e-12;
    const long long first = static_cast<long long>(std::ceil(center - radius));
    const long long last = static_cast<long long>(std::floor(center + radius));
    for (long long integer = first; integer <= last; ++integer) {
      trial[axis] = integer;
      const double term = diagonal * (static_cast<double>(integer) - center);
      const double next_cost = partial_cost + term * term;
      if (next_cost <= best_cost + 1.0e-12)
        visit(axis - 1, next_cost);
    }
  };
  visit(2, 0.0);
  return best;
}

Gram average_metric(const PbctopoTemporalCandidate &lhs,
                    const PbctopoTemporalCandidate &rhs) {
  if (lhs.metric_available && rhs.metric_available) {
    Gram gram{};
    for (std::size_t idx = 0; idx < gram.size(); ++idx)
      gram[idx] = 0.5 * (lhs.normalized_gram[idx] + rhs.normalized_gram[idx]);
    return gram;
  }
  if (lhs.metric_available)
    return lhs.normalized_gram;
  if (rhs.metric_available)
    return rhs.normalized_gram;
  return {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
}

double center_jump2_metric(const PbctopoTemporalCandidate &lhs,
                           const PbctopoTemporalCandidate &rhs) {
  const Gram gram = average_metric(lhs, rhs);
  const Real3 delta{rhs.center[0] - lhs.center[0],
                    rhs.center[1] - lhs.center[1],
                    rhs.center[2] - lhs.center[2]};
  const auto image = closest_metric_integer(delta, gram);
  return metric_norm2(
      {delta[0] - static_cast<double>(image[0]),
       delta[1] - static_cast<double>(image[1]),
       delta[2] - static_cast<double>(image[2])},
      gram);
}

int compare_path_cost(double lhs, double rhs) {
  const bool lhs_finite = std::isfinite(lhs);
  const bool rhs_finite = std::isfinite(rhs);
  if (lhs_finite != rhs_finite)
    return lhs_finite ? -1 : 1;
  if (!lhs_finite)
    return 0;
  const double tolerance =
      1.0e-12 * std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
  if (lhs < rhs - tolerance)
    return -1;
  if (lhs > rhs + tolerance)
    return 1;
  return 0;
}

} // namespace

double pbctopo_temporal_center_jump2(const std::array<double, 3> &lhs,
                                     const std::array<double, 3> &rhs) {
  PbctopoTemporalCandidate left;
  PbctopoTemporalCandidate right;
  left.center = lhs;
  right.center = rhs;
  return center_jump2_metric(left, right);
}

std::array<double, 6> pbctopo_temporal_normalized_gram(
    const std::array<double, 9> &basis) {
  auto dot_rows = [&](std::size_t lhs, std::size_t rhs) {
    return basis[lhs * 3] * basis[rhs * 3] +
           basis[lhs * 3 + 1] * basis[rhs * 3 + 1] +
           basis[lhs * 3 + 2] * basis[rhs * 3 + 2];
  };
  Gram gram{dot_rows(0, 0), dot_rows(0, 1), dot_rows(0, 2),
            dot_rows(1, 1), dot_rows(1, 2), dot_rows(2, 2)};
  const double scale = (gram[0] + gram[3] + gram[5]) / 3.0;
  if (!std::isfinite(scale) || scale <= 1.0e-12)
    return {1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  for (double &value : gram)
    value /= scale;
  return gram;
}

double pbctopo_temporal_assignment_jump2(
    const PbctopoTemporalCandidate &lhs,
    const PbctopoTemporalCandidate &rhs) {
  if (lhs.certified && rhs.certified && !lhs.component_offsets.empty() &&
      !rhs.component_offsets.empty()) {
    struct Match {
      std::size_t lhs = 0;
      std::size_t rhs = 0;
      double weight = 1.0;
    };
    std::vector<Match> matches;
    if (lhs.component_ids.size() == lhs.component_offsets.size() &&
        rhs.component_ids.size() == rhs.component_offsets.size()) {
      std::unordered_map<std::uint64_t, std::size_t> rhs_by_id;
      rhs_by_id.reserve(rhs.component_ids.size());
      for (std::size_t idx = 0; idx < rhs.component_ids.size(); ++idx)
        rhs_by_id.emplace(rhs.component_ids[idx], idx);
      for (std::size_t lhs_idx = 0; lhs_idx < lhs.component_ids.size();
           ++lhs_idx) {
        const auto found = rhs_by_id.find(lhs.component_ids[lhs_idx]);
        if (found == rhs_by_id.end())
          continue;
        const double lhs_weight =
            lhs_idx < lhs.component_weights.size()
                ? lhs.component_weights[lhs_idx]
                : 1.0;
        const double rhs_weight =
            found->second < rhs.component_weights.size()
                ? rhs.component_weights[found->second]
                : lhs_weight;
        matches.push_back(
            {lhs_idx, found->second,
             std::max(1.0, 0.5 * (lhs_weight + rhs_weight))});
      }
    } else if (lhs.component_offsets.size() == rhs.component_offsets.size()) {
      matches.reserve(lhs.component_offsets.size());
      for (std::size_t idx = 0; idx < lhs.component_offsets.size(); ++idx)
        matches.push_back({idx, idx, 1.0});
    }
    if (matches.size() >= 2) {
      double total_weight = 0.0;
      Real3 weighted_mean{};
      std::vector<Real3> deltas;
      deltas.reserve(matches.size());
      for (const Match &match : matches) {
        Int3 delta{};
        if (!checked_int3_subtract(rhs.component_offsets[match.rhs],
                                   lhs.component_offsets[match.lhs], delta)) {
          return std::numeric_limits<double>::infinity();
        }
        deltas.push_back({static_cast<double>(delta.x),
                          static_cast<double>(delta.y),
                          static_cast<double>(delta.z)});
        total_weight += match.weight;
        for (std::size_t axis = 0; axis < 3; ++axis)
          weighted_mean[axis] += match.weight * deltas.back()[axis];
      }
      for (double &value : weighted_mean)
        value /= total_weight;
      const Gram gram = average_metric(lhs, rhs);
      const auto gauge = closest_metric_integer(weighted_mean, gram);
      double jump2 = 0.0;
      for (std::size_t idx = 0; idx < matches.size(); ++idx) {
        const Real3 residual{
            deltas[idx][0] - static_cast<double>(gauge[0]),
            deltas[idx][1] - static_cast<double>(gauge[1]),
            deltas[idx][2] - static_cast<double>(gauge[2])};
        jump2 += matches[idx].weight * metric_norm2(residual, gram);
      }
      return jump2 / total_weight;
    }
  }
  return center_jump2_metric(lhs, rhs);
}

PbctopoTemporalPathResult solve_pbctopo_temporal_viterbi(
    const std::vector<std::vector<PbctopoTemporalCandidate>> &frames,
    double transition_penalty) {
  PbctopoTemporalPathResult result;
  const std::size_t invalid = std::numeric_limits<std::size_t>::max();
  result.selected.assign(frames.size(), invalid);
  transition_penalty = std::max(0.0, transition_penalty);

  std::size_t segment_begin = 0;
  while (segment_begin < frames.size()) {
    while (segment_begin < frames.size() && frames[segment_begin].empty())
      ++segment_begin;
    if (segment_begin == frames.size())
      break;
    std::size_t segment_end = segment_begin;
    while (segment_end + 1 < frames.size() &&
           !frames[segment_end + 1].empty())
      ++segment_end;

    std::vector<double> previous(frames[segment_begin].size(), 0.0);
    for (std::size_t state = 0; state < previous.size(); ++state)
      previous[state] = frames[segment_begin][state].emission_cost;
    std::vector<std::vector<std::size_t>> backtrace(segment_end - segment_begin + 1);

    for (std::size_t frame = segment_begin + 1; frame <= segment_end; ++frame) {
      std::vector<double> current(frames[frame].size(),
                                  std::numeric_limits<double>::infinity());
      backtrace[frame - segment_begin].assign(frames[frame].size(), 0);
      for (std::size_t state = 0; state < frames[frame].size(); ++state) {
        for (std::size_t prev = 0; prev < frames[frame - 1].size(); ++prev) {
          const double cost =
              previous[prev] + frames[frame][state].emission_cost +
              transition_penalty * pbctopo_temporal_assignment_jump2(
                                       frames[frame - 1][prev],
                                       frames[frame][state]);
          const int cost_cmp = compare_path_cost(cost, current[state]);
          if (cost_cmp < 0 ||
              (cost_cmp == 0 &&
               prev < backtrace[frame - segment_begin][state])) {
            current[state] = cost;
            backtrace[frame - segment_begin][state] = prev;
          }
        }
      }
      previous = std::move(current);
    }

    const auto best_it = std::min_element(previous.begin(), previous.end());
    std::size_t state = static_cast<std::size_t>(best_it - previous.begin());
    result.total_cost += *best_it;
    for (std::size_t frame = segment_end;; --frame) {
      result.selected[frame] = state;
      if (frame == segment_begin)
        break;
      const std::size_t prior = backtrace[frame - segment_begin][state];
      if (pbctopo_temporal_assignment_jump2(
              frames[frame - 1][prior], frames[frame][state]) > 1.0e-18)
        ++result.transition_count;
      state = prior;
    }
    segment_begin = segment_end + 1;
  }
  result.valid = std::any_of(result.selected.begin(), result.selected.end(),
                             [&](std::size_t value) { return value != invalid; });
  return result;
}

} // namespace titan_pbctopo

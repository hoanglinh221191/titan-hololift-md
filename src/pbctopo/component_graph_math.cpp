#include "component_graph_math.h"

#include "component_metric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>

namespace titan_pbctopo {

void finalize_normalized_pair_shift_score(PbctopoPairShiftScore &score,
                                          std::size_t lhs_atom_count,
                                          std::size_t rhs_atom_count,
                                          double contact_cutoff) {
  (void)contact_cutoff;
  const double lhs_count =
      static_cast<double>(std::max<std::size_t>(1, lhs_atom_count));
  const double rhs_count =
      static_cast<double>(std::max<std::size_t>(1, rhs_atom_count));
  score.contact_atoms = score.lhs_contact_atoms + score.rhs_contact_atoms;
  score.contact_fraction =
      0.5 * (static_cast<double>(score.lhs_contact_atoms) / lhs_count +
             static_cast<double>(score.rhs_contact_atoms) / rhs_count);
  score.soft_penalty_mean =
      score.contact_pairs > 0
          ? score.soft_penalty / static_cast<double>(score.contact_pairs)
          : 0.0;
  score.quality_score = score.contact_fraction * 1000.0 -
                        score.soft_penalty_mean * 50.0 -
                        score.shift_norm2;
  score.complete = true;
}

double pbctopo_pair_shift_confidence(
    const PbctopoPairShiftScore &best,
    const PbctopoPairShiftScore &second) noexcept {
  if (!best.valid || !best.complete || best.ultra_clash ||
      !std::isfinite(best.quality_score))
    return 0.0;
  if (second.valid && !second.complete)
    return 0.0;
  if (!second.valid || second.ultra_clash ||
      !std::isfinite(second.quality_score)) {
    return 1.0;
  }
  constexpr double kQualityScale = 100.0;
  const double gap = std::max(0.0, best.quality_score - second.quality_score);
  return std::tanh(gap / kQualityScale);
}

double pbctopo_pair_shift_candidate_weight(
    const PbctopoPairShiftScore &score, double confidence) noexcept {
  if (!score.valid || !score.complete || score.ultra_clash)
    return 0.0;
  const double bounded_contact =
      std::clamp(score.contact_fraction, 0.0, 1.0);
  const double bounded_confidence = std::clamp(confidence, 0.0, 1.0);
  return 1.0 + 4.0 * bounded_contact + 4.0 * bounded_confidence;
}

namespace {

struct Dsu {
  std::vector<std::size_t> parent;
  std::vector<unsigned char> rank;

  explicit Dsu(std::size_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  std::size_t find(std::size_t node) {
    if (parent[node] != node)
      parent[node] = find(parent[node]);
    return parent[node];
  }

  bool unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return false;
    if (rank[lhs] < rank[rhs])
      std::swap(lhs, rhs);
    parent[rhs] = lhs;
    if (rank[lhs] == rank[rhs])
      ++rank[lhs];
    return true;
  }
};

bool negate(Int3 value, Int3 &result) {
  return checked_int3_negate(value, result);
}

double residual_norm2(const Int3 &value, box Box) {
  return pbctopo_translation_norm2(Box, value);
}

} // namespace

PbctopoCycleSyncMetrics compute_fundamental_cycle_residuals(
    std::size_t component_count,
    const std::vector<PbctopoPairShiftCandidateSet> &pair_candidates,
    box Box) {
  PbctopoCycleSyncMetrics metrics;
  if (component_count < 2 || pair_candidates.empty())
    return metrics;

  std::vector<std::size_t> order(pair_candidates.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs,
                                                   std::size_t rhs) {
    const double lhs_weight = pair_candidates[lhs].candidates.empty()
                                  ? 0.0
                                  : pair_candidates[lhs].candidates.front().weight;
    const double rhs_weight = pair_candidates[rhs].candidates.empty()
                                  ? 0.0
                                  : pair_candidates[rhs].candidates.front().weight;
    return lhs_weight > rhs_weight;
  });

  struct TreeEdge {
    std::size_t node = 0;
    Int3 delta;
  };
  std::vector<std::vector<TreeEdge>> tree(component_count);
  std::vector<char> is_tree_edge(pair_candidates.size(), 0);
  Dsu dsu(component_count);
  for (std::size_t edge_idx : order) {
    const auto &set = pair_candidates[edge_idx];
    if (set.a >= component_count || set.b >= component_count ||
        set.candidates.empty())
      continue;
    if (!dsu.unite(set.a, set.b))
      continue;
    const Int3 delta = set.candidates.front().delta;
    Int3 reverse_delta{};
    if (!negate(delta, reverse_delta)) {
      ++metrics.inconsistent_edge_count;
      return metrics;
    }
    tree[set.a].push_back({set.b, delta});
    tree[set.b].push_back({set.a, reverse_delta});
    is_tree_edge[edge_idx] = 1;
  }

  std::vector<Int3> potential(component_count);
  std::vector<char> visited(component_count, 0);
  for (std::size_t root = 0; root < component_count; ++root) {
    if (visited[root])
      continue;
    std::queue<std::size_t> queue;
    queue.push(root);
    visited[root] = 1;
    while (!queue.empty()) {
      const std::size_t node = queue.front();
      queue.pop();
      for (const auto &edge : tree[node]) {
        if (visited[edge.node])
          continue;
        if (!checked_int3_add(potential[node], edge.delta,
                              potential[edge.node])) {
          ++metrics.inconsistent_edge_count;
          return metrics;
        }
        visited[edge.node] = 1;
        queue.push(edge.node);
      }
    }
  }

  double residual_sum2 = 0.0;
  for (std::size_t edge_idx = 0; edge_idx < pair_candidates.size(); ++edge_idx) {
    if (is_tree_edge[edge_idx])
      continue;
    const auto &set = pair_candidates[edge_idx];
    if (set.a >= component_count || set.b >= component_count ||
        set.candidates.empty() || dsu.find(set.a) != dsu.find(set.b))
      continue;
    Int3 tree_delta{};
    if (!checked_int3_subtract(potential[set.b], potential[set.a],
                               tree_delta)) {
      ++metrics.inconsistent_edge_count;
      continue;
    }
    double best_residual2 = std::numeric_limits<double>::infinity();
    for (const auto &candidate : set.candidates) {
      Int3 residual{};
      if (!checked_int3_subtract(tree_delta, candidate.delta, residual))
        continue;
      best_residual2 =
          std::min(best_residual2, residual_norm2(residual, Box));
    }
    if (!std::isfinite(best_residual2))
      continue;
    ++metrics.fundamental_cycle_count;
    residual_sum2 += best_residual2;
    metrics.fundamental_cycle_residual_max =
        std::max(metrics.fundamental_cycle_residual_max,
                 std::sqrt(best_residual2));
  }
  if (metrics.fundamental_cycle_count > 0) {
    metrics.fundamental_cycle_residual_rms =
        std::sqrt(residual_sum2 /
                  static_cast<double>(metrics.fundamental_cycle_count));
  }
  return metrics;
}

} // namespace titan_pbctopo

#include "titan_graph_community.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace titan_graph_community {
namespace {

struct WorkingGraph {
  int node_count{0};
  std::vector<int> row_offsets;
  std::vector<int> col_indices;
  std::vector<double> values;
  std::vector<double> loops;
  std::vector<double> degree;
  double total_weight_twice{0.0};
};

struct LevelResult {
  std::vector<int> membership;
  int community_count{0};
  bool moved{false};
};

double modularity_term(double internal_sum, double total_incident_weight,
                       double total_weight_twice) noexcept {
  if (total_weight_twice <= 0.0 || total_incident_weight <= 0.0)
    return 0.0;
  const double inv_total = 1.0 / total_weight_twice;
  const double frac = total_incident_weight * inv_total;
  return internal_sum * inv_total - frac * frac;
}

void clamp_near_zero(double &value) noexcept {
  if (std::abs(value) <= 1e-15)
    value = 0.0;
}

WorkingGraph build_working_graph(
    int node_count, std::vector<double> loops,
    std::vector<std::tuple<int, int, double>> undirected_edges) {
  if (node_count < 0) {
    throw std::invalid_argument("Community: node_count must be >= 0");
  }
  if (loops.size() != static_cast<std::size_t>(node_count)) {
    throw std::invalid_argument("Community: loop vector size mismatch");
  }

  std::sort(undirected_edges.begin(), undirected_edges.end(),
            [](const auto &lhs, const auto &rhs) {
              if (std::get<0>(lhs) != std::get<0>(rhs))
                return std::get<0>(lhs) < std::get<0>(rhs);
              if (std::get<1>(lhs) != std::get<1>(rhs))
                return std::get<1>(lhs) < std::get<1>(rhs);
              return std::get<2>(lhs) < std::get<2>(rhs);
            });

  std::vector<int> row_offsets(static_cast<std::size_t>(node_count + 1), 0);
  for (const auto &[lhs, rhs, weight] : undirected_edges) {
    if (lhs < 0 || rhs < 0 || lhs >= node_count || rhs >= node_count) {
      throw std::invalid_argument(
          "Community: edge endpoint out of range while building graph");
    }
    if (weight <= 0.0)
      continue;
    if (lhs == rhs) {
      loops[static_cast<std::size_t>(lhs)] += weight;
      continue;
    }
    ++row_offsets[static_cast<std::size_t>(lhs + 1)];
    ++row_offsets[static_cast<std::size_t>(rhs + 1)];
  }

  for (int node = 0; node < node_count; ++node) {
    row_offsets[static_cast<std::size_t>(node + 1)] +=
        row_offsets[static_cast<std::size_t>(node)];
  }

  std::vector<int> col_indices(
      static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(node_count)]),
      -1);
  std::vector<double> values(col_indices.size(), 0.0);
  std::vector<int> cursor = row_offsets;

  for (const auto &[lhs, rhs, weight] : undirected_edges) {
    if (weight <= 0.0 || lhs == rhs)
      continue;

    const int lhs_slot = cursor[static_cast<std::size_t>(lhs)]++;
    col_indices[static_cast<std::size_t>(lhs_slot)] = rhs;
    values[static_cast<std::size_t>(lhs_slot)] = weight;

    const int rhs_slot = cursor[static_cast<std::size_t>(rhs)]++;
    col_indices[static_cast<std::size_t>(rhs_slot)] = lhs;
    values[static_cast<std::size_t>(rhs_slot)] = weight;
  }

  WorkingGraph graph;
  graph.node_count = node_count;
  graph.row_offsets = std::move(row_offsets);
  graph.col_indices = std::move(col_indices);
  graph.values = std::move(values);
  graph.loops = std::move(loops);
  graph.degree.assign(static_cast<std::size_t>(node_count), 0.0);
  graph.total_weight_twice = 0.0;

  for (int node = 0; node < node_count; ++node) {
    double degree = graph.loops[static_cast<std::size_t>(node)];
    for (int edge_idx = graph.row_offsets[static_cast<std::size_t>(node)];
         edge_idx < graph.row_offsets[static_cast<std::size_t>(node + 1)];
         ++edge_idx) {
      degree += graph.values[static_cast<std::size_t>(edge_idx)];
    }
    graph.degree[static_cast<std::size_t>(node)] = degree;
    graph.total_weight_twice += degree;
  }

  return graph;
}

WorkingGraph build_working_graph(
    const titan_sparse::WeightedCSRGraph<double> &graph) {
  std::vector<std::tuple<int, int, double>> undirected_edges;
  undirected_edges.reserve(graph.values().size() / 2);
  for (int node = 0; node < graph.node_count(); ++node) {
    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      const int neighbor = graph.col_index(edge_idx);
      if (node < neighbor) {
        undirected_edges.emplace_back(node, neighbor, graph.value(edge_idx));
      }
    }
  }
  return build_working_graph(graph.node_count(),
                             std::vector<double>(
                                 static_cast<std::size_t>(graph.node_count()),
                                 0.0),
                             std::move(undirected_edges));
}

int compress_membership(std::vector<int> &membership,
                        const std::vector<int> &community_sizes) {
  std::vector<int> remap(community_sizes.size(), -1);
  int next_id = 0;
  for (std::size_t community = 0; community < community_sizes.size();
       ++community) {
    if (community_sizes[community] > 0) {
      remap[community] = next_id++;
    }
  }
  for (int &community : membership) {
    community = remap[static_cast<std::size_t>(community)];
  }
  return next_id;
}

LevelResult optimize_level(const WorkingGraph &graph, double gain_tolerance) {
  LevelResult result;
  const int node_count = graph.node_count;
  result.membership.resize(static_cast<std::size_t>(node_count));
  std::iota(result.membership.begin(), result.membership.end(), 0);
  if (node_count == 0) {
    result.community_count = 0;
    return result;
  }

  std::vector<double> community_internal = graph.loops;
  std::vector<double> community_total = graph.degree;
  std::vector<int> community_sizes(static_cast<std::size_t>(node_count), 1);
  std::set<int> empty_communities;
  std::vector<double> weights_to_community(static_cast<std::size_t>(node_count),
                                           0.0);
  std::vector<int> marks(static_cast<std::size_t>(node_count), -1);
  std::vector<int> touched;
  touched.reserve(32);
  int current_mark = 0;

  while (true) {
    bool moved_in_sweep = false;
    for (int node = 0; node < node_count; ++node) {
      const int current_community =
          result.membership[static_cast<std::size_t>(node)];
      const double node_loop = graph.loops[static_cast<std::size_t>(node)];
      const double node_degree = graph.degree[static_cast<std::size_t>(node)];

      touched.clear();
      const int mark = current_mark++;
      for (int edge_idx = graph.row_offsets[static_cast<std::size_t>(node)];
           edge_idx < graph.row_offsets[static_cast<std::size_t>(node + 1)];
           ++edge_idx) {
        const int neighbor = graph.col_indices[static_cast<std::size_t>(edge_idx)];
        const int neighbor_community =
            result.membership[static_cast<std::size_t>(neighbor)];
        if (marks[static_cast<std::size_t>(neighbor_community)] != mark) {
          marks[static_cast<std::size_t>(neighbor_community)] = mark;
          weights_to_community[static_cast<std::size_t>(neighbor_community)] =
              0.0;
          touched.push_back(neighbor_community);
        }
        weights_to_community[static_cast<std::size_t>(neighbor_community)] +=
            graph.values[static_cast<std::size_t>(edge_idx)];
      }
      std::sort(touched.begin(), touched.end());

      const double weight_to_current =
          marks[static_cast<std::size_t>(current_community)] == mark
              ? weights_to_community[static_cast<std::size_t>(current_community)]
              : 0.0;

      const double current_before =
          modularity_term(
              community_internal[static_cast<std::size_t>(current_community)],
              community_total[static_cast<std::size_t>(current_community)],
              graph.total_weight_twice);
      const double current_after =
          modularity_term(
              community_internal[static_cast<std::size_t>(current_community)] -
                  (node_loop + 2.0 * weight_to_current),
              community_total[static_cast<std::size_t>(current_community)] -
                  node_degree,
              graph.total_weight_twice);

      double best_gain = 0.0;
      int best_community = current_community;
      double best_weight_to_target = 0.0;

      auto consider_target = [&](int target_community,
                                 double weight_to_target) {
        if (target_community == current_community)
          return;
        const double before =
            current_before +
            modularity_term(
                community_internal[static_cast<std::size_t>(target_community)],
                community_total[static_cast<std::size_t>(target_community)],
                graph.total_weight_twice);
        const double after =
            current_after +
            modularity_term(
                community_internal[static_cast<std::size_t>(target_community)] +
                    (node_loop + 2.0 * weight_to_target),
                community_total[static_cast<std::size_t>(target_community)] +
                    node_degree,
                graph.total_weight_twice);
        const double gain = after - before;
        if (gain > best_gain + gain_tolerance ||
            (std::abs(gain - best_gain) <= gain_tolerance &&
             gain > gain_tolerance && target_community < best_community)) {
          best_gain = gain;
          best_community = target_community;
          best_weight_to_target = weight_to_target;
        }
      };

      for (int target_community : touched) {
        consider_target(
            target_community,
            weights_to_community[static_cast<std::size_t>(target_community)]);
      }

      if (community_sizes[static_cast<std::size_t>(current_community)] > 1 &&
          !empty_communities.empty()) {
        consider_target(*empty_communities.begin(), 0.0);
      }

      if (best_community == current_community)
        continue;

      community_internal[static_cast<std::size_t>(current_community)] -=
          (node_loop + 2.0 * weight_to_current);
      community_total[static_cast<std::size_t>(current_community)] -= node_degree;
      --community_sizes[static_cast<std::size_t>(current_community)];
      clamp_near_zero(
          community_internal[static_cast<std::size_t>(current_community)]);
      clamp_near_zero(
          community_total[static_cast<std::size_t>(current_community)]);
      if (community_sizes[static_cast<std::size_t>(current_community)] == 0) {
        empty_communities.insert(current_community);
      }

      if (community_sizes[static_cast<std::size_t>(best_community)] == 0) {
        empty_communities.erase(best_community);
      }
      community_internal[static_cast<std::size_t>(best_community)] +=
          (node_loop + 2.0 * best_weight_to_target);
      community_total[static_cast<std::size_t>(best_community)] += node_degree;
      ++community_sizes[static_cast<std::size_t>(best_community)];
      clamp_near_zero(
          community_internal[static_cast<std::size_t>(best_community)]);
      clamp_near_zero(community_total[static_cast<std::size_t>(best_community)]);

      result.membership[static_cast<std::size_t>(node)] = best_community;
      moved_in_sweep = true;
      result.moved = true;
    }

    if (!moved_in_sweep)
      break;
  }

  result.community_count =
      compress_membership(result.membership, community_sizes);
  return result;
}

WorkingGraph induce_graph(const WorkingGraph &graph,
                          const std::vector<int> &membership,
                          int community_count) {
  std::vector<double> induced_loops(static_cast<std::size_t>(community_count),
                                    0.0);
  std::map<std::pair<int, int>, double> induced_edges;

  for (int node = 0; node < graph.node_count; ++node) {
    const int node_community = membership[static_cast<std::size_t>(node)];
    induced_loops[static_cast<std::size_t>(node_community)] +=
        graph.loops[static_cast<std::size_t>(node)];

    for (int edge_idx = graph.row_offsets[static_cast<std::size_t>(node)];
         edge_idx < graph.row_offsets[static_cast<std::size_t>(node + 1)];
         ++edge_idx) {
      const int neighbor = graph.col_indices[static_cast<std::size_t>(edge_idx)];
      if (node >= neighbor)
        continue;
      const double weight = graph.values[static_cast<std::size_t>(edge_idx)];
      const int neighbor_community =
          membership[static_cast<std::size_t>(neighbor)];
      if (node_community == neighbor_community) {
        induced_loops[static_cast<std::size_t>(node_community)] += 2.0 * weight;
      } else {
        induced_edges[{std::min(node_community, neighbor_community),
                       std::max(node_community, neighbor_community)}] += weight;
      }
    }
  }

  std::vector<std::tuple<int, int, double>> undirected_edges;
  undirected_edges.reserve(induced_edges.size());
  for (const auto &[edge, weight] : induced_edges) {
    if (weight > 0.0) {
      undirected_edges.emplace_back(edge.first, edge.second, weight);
    }
  }

  return build_working_graph(community_count, std::move(induced_loops),
                             std::move(undirected_edges));
}

std::vector<int> normalize_membership_copy(std::span<const int> membership) {
  std::vector<int> normalized(membership.begin(), membership.end());
  if (normalized.empty())
    return normalized;

  const int max_community =
      *std::max_element(normalized.begin(), normalized.end());
  if (max_community < 0) {
    throw std::invalid_argument("Community: membership contains negative ids");
  }

  std::vector<int> counts(static_cast<std::size_t>(max_community + 1), 0);
  for (int community : normalized) {
    if (community < 0) {
      throw std::invalid_argument("Community: membership contains negative ids");
    }
    ++counts[static_cast<std::size_t>(community)];
  }
  compress_membership(normalized, counts);
  return normalized;
}

} // namespace

PartitionStats
summarize_partition(const titan_sparse::WeightedCSRGraph<double> &graph,
                    std::span<const int> membership) {
  if (membership.size() != static_cast<std::size_t>(graph.node_count())) {
    throw std::invalid_argument(
        "Community: membership size must match graph node count");
  }

  PartitionStats stats;
  if (graph.node_count() == 0)
    return stats;

  const std::vector<int> normalized_membership =
      normalize_membership_copy(membership);
  const int community_count = normalized_membership.empty()
                                  ? 0
                                  : (*std::max_element(normalized_membership.begin(),
                                                       normalized_membership.end()) +
                                     1);

  stats.community_sizes.assign(static_cast<std::size_t>(community_count), 0);
  stats.internal_weights.assign(static_cast<std::size_t>(community_count), 0.0);
  stats.total_incident_weights.assign(static_cast<std::size_t>(community_count),
                                      0.0);
  stats.modularity_contributions.assign(
      static_cast<std::size_t>(community_count), 0.0);

  std::vector<double> internal_sum(static_cast<std::size_t>(community_count),
                                   0.0);
  double total_weight_twice = 0.0;

  for (int node = 0; node < graph.node_count(); ++node) {
    const int community = normalized_membership[static_cast<std::size_t>(node)];
    ++stats.community_sizes[static_cast<std::size_t>(community)];
    double degree = 0.0;
    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      degree += graph.value(edge_idx);
    }
    stats.total_incident_weights[static_cast<std::size_t>(community)] += degree;
    total_weight_twice += degree;

    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      const int neighbor = graph.col_index(edge_idx);
      if (node < neighbor &&
          community == normalized_membership[static_cast<std::size_t>(neighbor)]) {
        internal_sum[static_cast<std::size_t>(community)] +=
            2.0 * graph.value(edge_idx);
      }
    }
  }

  for (int community = 0; community < community_count; ++community) {
    stats.internal_weights[static_cast<std::size_t>(community)] =
        internal_sum[static_cast<std::size_t>(community)] * 0.5;
    stats.modularity_contributions[static_cast<std::size_t>(community)] =
        modularity_term(internal_sum[static_cast<std::size_t>(community)],
                        stats.total_incident_weights[static_cast<std::size_t>(
                            community)],
                        total_weight_twice);
    stats.modularity +=
        stats.modularity_contributions[static_cast<std::size_t>(community)];
  }

  return stats;
}

LouvainResult run_louvain(const titan_sparse::WeightedCSRGraph<double> &graph,
                          double gain_tolerance) {
  LouvainResult result;
  const int original_node_count = graph.node_count();
  result.membership.resize(static_cast<std::size_t>(original_node_count));
  std::iota(result.membership.begin(), result.membership.end(), 0);
  if (original_node_count == 0) {
    return result;
  }

  WorkingGraph working_graph = build_working_graph(graph);
  std::vector<int> original_to_working(result.membership.begin(),
                                       result.membership.end());

  while (true) {
    LevelResult level = optimize_level(working_graph, gain_tolerance);
    ++result.level_count;

    for (std::size_t node = 0; node < original_to_working.size(); ++node) {
      original_to_working[node] =
          level.membership[static_cast<std::size_t>(original_to_working[node])];
    }

    result.membership = original_to_working;

    if (!level.moved || level.community_count >= working_graph.node_count) {
      break;
    }

    working_graph =
        induce_graph(working_graph, level.membership, level.community_count);
  }

  result.stats = summarize_partition(graph, result.membership);
  return result;
}

} // namespace titan_graph_community

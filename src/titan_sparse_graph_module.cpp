#include "titan_sparse_graph_module.h"

#include "titan_executor.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace titan_sparse_graph_module {
namespace {

constexpr double kEps = 1e-12;

std::string residue_node_id(const NodeMetadata &node) {
  if (!node.node_id.empty())
    return node.node_id;
  std::ostringstream oss;
  oss << (node.resname.empty() ? "UNK" : node.resname) << node.resnr << "_"
      << (node.chain.empty() ? "?" : node.chain);
  return oss.str();
}

std::string residue_node_id(const std::string &resname, int resnr,
                            const std::string &chain) {
  NodeMetadata node;
  node.resname = resname;
  node.resnr = resnr;
  node.chain = chain;
  return residue_node_id(node);
}

double safe_ratio(double value, double denom) {
  return (denom > 0.0) ? value / denom : 0.0;
}

double max_or_one(const std::vector<double> &values) {
  double max_value = 0.0;
  for (double value : values) {
    if (std::isfinite(value))
      max_value = std::max(max_value, value);
  }
  return max_value > 0.0 ? max_value : 1.0;
}

int max_or_zero(const std::vector<int> &values) {
  int max_value = 0;
  for (int value : values)
    max_value = std::max(max_value, value);
  return max_value;
}

double edge_distance_from_weight(double weight) {
  const double clamped = std::max(weight, 1e-15);
  if (clamped <= 1.0)
    return -std::log(clamped);
  return 1.0 / clamped;
}

bool is_community_source(SourceKind source) {
  return source == SourceKind::Dccm || source == SourceKind::Lspmd ||
         source == SourceKind::AllosteryGc;
}

struct DisjointSet {
  explicit DisjointSet(std::size_t n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    int root = x;
    while (parent[static_cast<std::size_t>(root)] != root)
      root = parent[static_cast<std::size_t>(root)];
    while (parent[static_cast<std::size_t>(x)] != x) {
      const int next = parent[static_cast<std::size_t>(x)];
      parent[static_cast<std::size_t>(x)] = root;
      x = next;
    }
    return root;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;
    if (size[static_cast<std::size_t>(a)] < size[static_cast<std::size_t>(b)])
      std::swap(a, b);
    parent[static_cast<std::size_t>(b)] = a;
    size[static_cast<std::size_t>(a)] += size[static_cast<std::size_t>(b)];
  }

  std::vector<int> parent;
  std::vector<int> size;
};

struct EdgeAdjacency {
  int node_count{0};
  bool directed{false};
  std::vector<int> row_offsets{};
  std::vector<int> col_indices{};
  std::vector<int> edge_ids{};
  std::vector<double> distances{};
};

std::vector<NodeMetadata> build_residue_nodes(const DCCMNetwork &dccm) {
  const auto &resnames = dccm.get_resnames();
  const auto &resnrs = dccm.get_resnrs();
  const auto &chains = dccm.get_chains();
  const auto &is_lig = dccm.get_is_lig();
  const std::size_t node_count =
      std::min({resnames.size(), resnrs.size(), chains.size(), is_lig.size()});
  std::vector<NodeMetadata> nodes(node_count);
  for (std::size_t idx = 0; idx < node_count; ++idx) {
    auto &node = nodes[idx];
    node.resname = resnames[idx];
    node.resnr = resnrs[idx];
    node.chain = chains[idx];
    node.is_lig = is_lig[idx];
    node.type = is_lig[idx] ? "Ligand" : "Receptor";
    node.label = residue_node_id(node.resname, node.resnr, node.chain);
    node.node_id = node.label;
  }
  return nodes;
}

std::vector<int> compute_component_ids(const SourceGraph &graph) {
  const std::size_t node_count = graph.nodes.size();
  if (node_count == 0)
    return {};

  DisjointSet dsu(node_count);
  for (const auto &edge : graph.edges) {
    if (edge.source < 0 || edge.target < 0 ||
        static_cast<std::size_t>(edge.source) >= node_count ||
        static_cast<std::size_t>(edge.target) >= node_count) {
      continue;
    }
    dsu.unite(edge.source, edge.target);
  }

  std::vector<int> remap(node_count, -1);
  std::vector<int> component_ids(node_count, -1);
  int next_id = 0;
  for (std::size_t node = 0; node < node_count; ++node) {
    const int root = dsu.find(static_cast<int>(node));
    int &mapped = remap[static_cast<std::size_t>(root)];
    if (mapped < 0)
      mapped = next_id++;
    component_ids[node] = mapped;
  }
  return component_ids;
}

SourceGraph apply_topk_filter(const SourceGraph &graph, int topk) {
  if (topk <= 0 || graph.edges.empty())
    return graph;

  std::vector<unsigned char> keep(graph.edges.size(), 0);
  std::vector<std::vector<int>> incident(graph.nodes.size());
  for (std::size_t idx = 0; idx < graph.edges.size(); ++idx) {
    const auto &edge = graph.edges[idx];
    if (edge.source < 0 || edge.target < 0 ||
        static_cast<std::size_t>(edge.source) >= graph.nodes.size() ||
        static_cast<std::size_t>(edge.target) >= graph.nodes.size()) {
      continue;
    }
    incident[static_cast<std::size_t>(edge.source)].push_back(
        static_cast<int>(idx));
    if (!graph.directed) {
      incident[static_cast<std::size_t>(edge.target)].push_back(
          static_cast<int>(idx));
    }
  }

  auto edge_cmp = [&](int lhs_idx, int rhs_idx) {
    const auto &lhs = graph.edges[static_cast<std::size_t>(lhs_idx)];
    const auto &rhs = graph.edges[static_cast<std::size_t>(rhs_idx)];
    if (lhs.weight != rhs.weight)
      return lhs.weight > rhs.weight;
    if (lhs.source != rhs.source)
      return lhs.source < rhs.source;
    return lhs.target < rhs.target;
  };

  for (auto &edges : incident) {
    std::stable_sort(edges.begin(), edges.end(), edge_cmp);
    const std::size_t limit =
        std::min<std::size_t>(edges.size(), static_cast<std::size_t>(topk));
    for (std::size_t idx = 0; idx < limit; ++idx) {
      keep[static_cast<std::size_t>(edges[idx])] = 1;
    }
  }

  SourceGraph filtered = graph;
  filtered.edges.clear();
  filtered.edges.reserve(graph.edges.size());
  for (std::size_t idx = 0; idx < graph.edges.size(); ++idx) {
    if (keep[idx] != 0)
      filtered.edges.push_back(graph.edges[idx]);
  }
  return filtered;
}

SourceGraph apply_component_filter(const SourceGraph &graph,
                                   const std::string &mode) {
  if (mode != "largest" || graph.nodes.empty())
    return graph;

  const auto component_ids = compute_component_ids(graph);
  if (component_ids.empty())
    return graph;

  const int component_count = 1 + max_or_zero(component_ids);
  std::vector<int> component_sizes(static_cast<std::size_t>(component_count), 0);
  for (int component : component_ids)
    ++component_sizes[static_cast<std::size_t>(component)];

  int best_component = 0;
  int best_size = component_sizes.front();
  for (int component = 1; component < component_count; ++component) {
    const int size = component_sizes[static_cast<std::size_t>(component)];
    if (size > best_size) {
      best_component = component;
      best_size = size;
    }
  }

  std::vector<int> reindex(graph.nodes.size(), -1);
  SourceGraph filtered = graph;
  filtered.nodes.clear();
  filtered.edges.clear();
  filtered.nodes.reserve(static_cast<std::size_t>(best_size));
  for (std::size_t node = 0; node < graph.nodes.size(); ++node) {
    if (component_ids[node] != best_component)
      continue;
    reindex[node] = static_cast<int>(filtered.nodes.size());
    filtered.nodes.push_back(graph.nodes[node]);
  }
  for (const auto &edge : graph.edges) {
    if (edge.source < 0 || edge.target < 0 ||
        static_cast<std::size_t>(edge.source) >= reindex.size() ||
        static_cast<std::size_t>(edge.target) >= reindex.size()) {
      continue;
    }
    const int source = reindex[static_cast<std::size_t>(edge.source)];
    const int target = reindex[static_cast<std::size_t>(edge.target)];
    if (source < 0 || target < 0)
      continue;
    EdgeMetadata remapped = edge;
    remapped.source = source;
    remapped.target = target;
    filtered.edges.push_back(remapped);
  }
  return filtered;
}

EdgeAdjacency build_adjacency(const SourceGraph &graph) {
  EdgeAdjacency adjacency;
  adjacency.node_count = static_cast<int>(graph.nodes.size());
  adjacency.directed = graph.directed;
  adjacency.row_offsets.assign(static_cast<std::size_t>(adjacency.node_count + 1),
                               0);

  const int edge_factor = graph.directed ? 1 : 2;
  for (const auto &edge : graph.edges) {
    if (edge.source < 0 || edge.target < 0 ||
        edge.source >= adjacency.node_count || edge.target >= adjacency.node_count ||
        edge.source == edge.target) {
      continue;
    }
    ++adjacency.row_offsets[static_cast<std::size_t>(edge.source + 1)];
    if (!graph.directed)
      ++adjacency.row_offsets[static_cast<std::size_t>(edge.target + 1)];
  }
  for (int node = 0; node < adjacency.node_count; ++node) {
    adjacency.row_offsets[static_cast<std::size_t>(node + 1)] +=
        adjacency.row_offsets[static_cast<std::size_t>(node)];
  }

  const std::size_t total_slots =
      static_cast<std::size_t>(adjacency.row_offsets.back());
  adjacency.col_indices.assign(total_slots, -1);
  adjacency.edge_ids.assign(total_slots, -1);
  adjacency.distances.assign(total_slots, 0.0);
  std::vector<int> cursor = adjacency.row_offsets;

  for (std::size_t edge_idx = 0; edge_idx < graph.edges.size(); ++edge_idx) {
    const auto &edge = graph.edges[edge_idx];
    if (edge.source < 0 || edge.target < 0 ||
        edge.source >= adjacency.node_count || edge.target >= adjacency.node_count ||
        edge.source == edge.target) {
      continue;
    }
    const double distance = edge_distance_from_weight(edge.weight);
    const int lhs_slot = cursor[static_cast<std::size_t>(edge.source)]++;
    adjacency.col_indices[static_cast<std::size_t>(lhs_slot)] = edge.target;
    adjacency.edge_ids[static_cast<std::size_t>(lhs_slot)] =
        static_cast<int>(edge_idx);
    adjacency.distances[static_cast<std::size_t>(lhs_slot)] = distance;
    if (!graph.directed) {
      const int rhs_slot = cursor[static_cast<std::size_t>(edge.target)]++;
      adjacency.col_indices[static_cast<std::size_t>(rhs_slot)] = edge.source;
      adjacency.edge_ids[static_cast<std::size_t>(rhs_slot)] =
          static_cast<int>(edge_idx);
      adjacency.distances[static_cast<std::size_t>(rhs_slot)] = distance;
    }
  }
  (void)edge_factor;
  return adjacency;
}

void compute_strength_degree(const SourceGraph &graph, std::vector<int> &degree_out,
                             std::vector<double> &strength_out) {
  degree_out.assign(graph.nodes.size(), 0);
  strength_out.assign(graph.nodes.size(), 0.0);
  for (const auto &edge : graph.edges) {
    if (edge.source < 0 || edge.target < 0 ||
        static_cast<std::size_t>(edge.source) >= graph.nodes.size() ||
        static_cast<std::size_t>(edge.target) >= graph.nodes.size()) {
      continue;
    }
    ++degree_out[static_cast<std::size_t>(edge.source)];
    strength_out[static_cast<std::size_t>(edge.source)] += edge.weight;
    if (!graph.directed) {
      ++degree_out[static_cast<std::size_t>(edge.target)];
      strength_out[static_cast<std::size_t>(edge.target)] += edge.weight;
    } else {
      ++degree_out[static_cast<std::size_t>(edge.target)];
      strength_out[static_cast<std::size_t>(edge.target)] += edge.weight;
    }
  }
}

void compute_brandes_betweenness(const SourceGraph &graph,
                                 const EdgeAdjacency &adjacency,
                                 std::vector<double> &node_bc_out,
                                 std::vector<double> &edge_bc_out,
                                 TitanExecutor &executor) {
  const std::size_t n = graph.nodes.size();
  const std::size_t m = graph.edges.size();
  node_bc_out.assign(n, 0.0);
  edge_bc_out.assign(m, 0.0);
  if (n == 0 || m == 0)
    return;

  const std::size_t worker_count = std::max<std::size_t>(1, executor.capacity());
  std::vector<std::vector<double>> partial_nodes(
      worker_count, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> partial_edges(
      worker_count, std::vector<double>(m, 0.0));

  const std::size_t chunk = (n + worker_count - 1) / worker_count;
  executor.parallel_do(worker_count, [&](std::size_t worker) {
    const std::size_t begin = worker * chunk;
    const std::size_t end = std::min(n, begin + chunk);
    if (begin >= end)
      return;

    auto &node_local = partial_nodes[worker];
    auto &edge_local = partial_edges[worker];
    std::vector<double> dist(n), sigma(n), delta(n);
    std::vector<std::vector<int>> pred_nodes(n), pred_edges(n);
    for (std::size_t node = 0; node < n; ++node) {
      const int degree = adjacency.row_offsets[node + 1] - adjacency.row_offsets[node];
      pred_nodes[node].reserve(static_cast<std::size_t>(std::max(1, degree)));
      pred_edges[node].reserve(static_cast<std::size_t>(std::max(1, degree)));
    }
    std::vector<int> order;
    order.reserve(n);

    for (std::size_t source = begin; source < end; ++source) {
      std::fill(dist.begin(), dist.end(),
                std::numeric_limits<double>::infinity());
      std::fill(sigma.begin(), sigma.end(), 0.0);
      std::fill(delta.begin(), delta.end(), 0.0);
      for (std::size_t node = 0; node < n; ++node) {
        pred_nodes[node].clear();
        pred_edges[node].clear();
      }
      order.clear();

      using Entry = std::pair<double, int>;
      std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
      dist[source] = 0.0;
      sigma[source] = 1.0;
      pq.push({0.0, static_cast<int>(source)});

      while (!pq.empty()) {
        const auto [dist_u, u] = pq.top();
        pq.pop();
        if (dist_u > dist[static_cast<std::size_t>(u)] + kEps)
          continue;
        order.push_back(u);
        const int row_begin = adjacency.row_offsets[static_cast<std::size_t>(u)];
        const int row_end =
            adjacency.row_offsets[static_cast<std::size_t>(u + 1)];
        for (int edge_ptr = row_begin; edge_ptr < row_end; ++edge_ptr) {
          const int v = adjacency.col_indices[static_cast<std::size_t>(edge_ptr)];
          const double alt =
              dist[static_cast<std::size_t>(u)] +
              adjacency.distances[static_cast<std::size_t>(edge_ptr)];
          if (alt + kEps < dist[static_cast<std::size_t>(v)]) {
            dist[static_cast<std::size_t>(v)] = alt;
            sigma[static_cast<std::size_t>(v)] =
                sigma[static_cast<std::size_t>(u)];
            pred_nodes[static_cast<std::size_t>(v)].clear();
            pred_edges[static_cast<std::size_t>(v)].clear();
            pred_nodes[static_cast<std::size_t>(v)].push_back(u);
            pred_edges[static_cast<std::size_t>(v)].push_back(
                adjacency.edge_ids[static_cast<std::size_t>(edge_ptr)]);
            pq.push({alt, v});
          } else if (std::abs(alt - dist[static_cast<std::size_t>(v)]) <= kEps) {
            sigma[static_cast<std::size_t>(v)] +=
                sigma[static_cast<std::size_t>(u)];
            pred_nodes[static_cast<std::size_t>(v)].push_back(u);
            pred_edges[static_cast<std::size_t>(v)].push_back(
                adjacency.edge_ids[static_cast<std::size_t>(edge_ptr)]);
          }
        }
      }

      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int w = *it;
        const auto &preds = pred_nodes[static_cast<std::size_t>(w)];
        const auto &pred_edges_w = pred_edges[static_cast<std::size_t>(w)];
        for (std::size_t idx = 0; idx < preds.size(); ++idx) {
          const int v = preds[idx];
          const int edge_id = pred_edges_w[idx];
          if (edge_id < 0 || sigma[static_cast<std::size_t>(w)] <= 0.0)
            continue;
          const double coeff =
              (sigma[static_cast<std::size_t>(v)] /
               sigma[static_cast<std::size_t>(w)]) *
              (1.0 + delta[static_cast<std::size_t>(w)]);
          delta[static_cast<std::size_t>(v)] += coeff;
          edge_local[static_cast<std::size_t>(edge_id)] += coeff;
        }
        if (w != static_cast<int>(source)) {
          node_local[static_cast<std::size_t>(w)] +=
              delta[static_cast<std::size_t>(w)];
        }
      }
    }
  });

  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    for (std::size_t node = 0; node < n; ++node)
      node_bc_out[node] += partial_nodes[worker][node];
    for (std::size_t edge = 0; edge < m; ++edge)
      edge_bc_out[edge] += partial_edges[worker][edge];
  }

  if (!graph.directed) {
    for (double &value : node_bc_out)
      value *= 0.5;
    for (double &value : edge_bc_out)
      value *= 0.5;
  }

  if (n > 2) {
    const double norm = graph.directed
                            ? static_cast<double>((n - 1) * (n - 2))
                            : static_cast<double>((n - 1) * (n - 2)) / 2.0;
    if (norm > 0.0) {
      for (double &value : node_bc_out)
        value /= norm;
    }
  }
}

std::vector<double> compute_intra_community_strengths(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const std::vector<int> &membership) {
  std::vector<double> strengths(membership.size(), 0.0);
  for (int node = 0; node < graph.node_count(); ++node) {
    if (static_cast<std::size_t>(node) >= membership.size())
      break;
    const int community = membership[static_cast<std::size_t>(node)];
    double total = 0.0;
    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      const int neighbor = graph.col_index(edge_idx);
      if (neighbor >= 0 && static_cast<std::size_t>(neighbor) < membership.size() &&
          membership[static_cast<std::size_t>(neighbor)] == community) {
        total += graph.value(edge_idx);
      }
    }
    strengths[static_cast<std::size_t>(node)] = total;
  }
  return strengths;
}

titan_sparse::WeightedCSRGraph<double>
build_undirected_weighted_graph(const SourceGraph &graph) {
  std::vector<std::tuple<int, int, double>> edges;
  edges.reserve(graph.edges.size());
  for (const auto &edge : graph.edges) {
    if (edge.source == edge.target)
      continue;
    if (edge.weight <= 0.0)
      continue;
    edges.emplace_back(edge.source, edge.target, edge.weight);
  }
  return titan_sparse::WeightedCSRGraph<double>::from_undirected_edges(
      static_cast<int>(graph.nodes.size()), edges);
}

void write_empty_community_files(const info &info_output,
                                 const std::string &source_name) {
  {
    std::ofstream out(info_output.get_graph_communities_filename(source_name));
    out << "NodeIndex,NodeId,Label,Type,ResName,ResNr,Chain,IsLig,CommunityId,"
           "CommunitySize,IntraCommunityStrength\n";
  }
  {
    std::ofstream out(
        info_output.get_graph_community_summary_filename(source_name));
    out << "CommunityId,Size,InternalWeight,TotalIncidentWeight,"
           "ModularityContribution\n";
  }
  {
    std::ofstream out(info_output.get_graph_supergraph_filename(source_name));
    out << "SourceCommunity,TargetCommunity,EdgeWeight\n";
  }
}

void write_community_files(const std::string &source_name,
                           const AnalysisResult &result,
                           const info &info_output) {
  if (!result.community_available || result.filtered_graph.directed) {
    write_empty_community_files(info_output, source_name);
    return;
  }

  const auto weighted_graph = build_undirected_weighted_graph(result.filtered_graph);
  const auto intra_strengths = compute_intra_community_strengths(
      weighted_graph, result.community_membership);

  {
    std::ofstream out(info_output.get_graph_communities_filename(source_name));
    out << "NodeIndex,NodeId,Label,Type,ResName,ResNr,Chain,IsLig,CommunityId,"
           "CommunitySize,IntraCommunityStrength\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t node = 0; node < result.filtered_graph.nodes.size(); ++node) {
      const auto &meta = result.filtered_graph.nodes[node];
      const int community = result.community_membership[node];
      const int size =
          (community >= 0 &&
           static_cast<std::size_t>(community) <
               result.community_stats.community_sizes.size())
              ? result.community_stats.community_sizes[static_cast<std::size_t>(
                    community)]
              : 0;
      out << node << ',' << residue_node_id(meta) << ',' << meta.label << ','
          << meta.type << ',' << meta.resname << ',' << meta.resnr << ','
          << meta.chain << ',' << meta.is_lig << ',' << (community + 1) << ','
          << size << ',' << intra_strengths[node] << "\n";
    }
  }

  {
    std::ofstream out(
        info_output.get_graph_community_summary_filename(source_name));
    out << "CommunityId,Size,InternalWeight,TotalIncidentWeight,"
           "ModularityContribution\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t community = 0;
         community < result.community_stats.community_sizes.size(); ++community) {
      out << (community + 1) << ','
          << result.community_stats.community_sizes[community] << ','
          << result.community_stats.internal_weights[community] << ','
          << result.community_stats.total_incident_weights[community] << ','
          << result.community_stats.modularity_contributions[community] << "\n";
    }
  }

  {
    std::map<std::pair<int, int>, double> super_edges;
    for (const auto &edge : result.filtered_graph.edges) {
      const int source_comm =
          result.community_membership[static_cast<std::size_t>(edge.source)];
      const int target_comm =
          result.community_membership[static_cast<std::size_t>(edge.target)];
      if (source_comm == target_comm)
        continue;
      const auto key =
          std::make_pair(std::min(source_comm, target_comm),
                         std::max(source_comm, target_comm));
      super_edges[key] += edge.weight;
    }
    std::ofstream out(info_output.get_graph_supergraph_filename(source_name));
    out << "SourceCommunity,TargetCommunity,EdgeWeight\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &[key, weight] : super_edges) {
      out << (key.first + 1) << ',' << (key.second + 1) << ',' << weight
          << "\n";
    }
  }
}

void write_edges_file(const std::string &source_name, const AnalysisResult &result,
                      const info &info_output) {
  std::ofstream out(info_output.get_graph_edges_filename(source_name));
  out << "Rank,Source,Target,Directed,Weight,RawScore,SourceComponent,"
         "TargetComponent,EdgeBetweenness,EdgeBetweennessNorm\n";
  out << std::setprecision(15);
  std::vector<std::size_t> order(result.filtered_graph.edges.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const auto &a = result.filtered_graph.edges[lhs];
                     const auto &b = result.filtered_graph.edges[rhs];
                     if (a.weight != b.weight)
                       return a.weight > b.weight;
                     if (a.source != b.source)
                       return a.source < b.source;
                     return a.target < b.target;
                   });
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const std::size_t edge_idx = order[rank];
    const auto &edge = result.filtered_graph.edges[edge_idx];
    const auto &source = result.filtered_graph.nodes[static_cast<std::size_t>(edge.source)];
    const auto &target = result.filtered_graph.nodes[static_cast<std::size_t>(edge.target)];
    const int source_component =
        static_cast<std::size_t>(edge.source) < result.component_ids.size()
            ? result.component_ids[static_cast<std::size_t>(edge.source)] + 1
            : 0;
    const int target_component =
        static_cast<std::size_t>(edge.target) < result.component_ids.size()
            ? result.component_ids[static_cast<std::size_t>(edge.target)] + 1
            : 0;
    const double edge_bc =
        edge_idx < result.edge_betweenness.size() ? result.edge_betweenness[edge_idx]
                                                  : 0.0;
    const double edge_bc_norm =
        edge_idx < result.edge_betweenness_norm.size()
            ? result.edge_betweenness_norm[edge_idx]
            : 0.0;
    out << (rank + 1) << ',' << residue_node_id(source) << ','
        << residue_node_id(target) << ',' << (result.filtered_graph.directed ? 1 : 0)
        << ',' << edge.weight << ',' << edge.raw_score << ','
        << source_component << ',' << target_component << ',' << edge_bc
        << ',' << edge_bc_norm << "\n";
  }
}

void write_nodes_file(const std::string &source_name, const AnalysisResult &result,
                      const info &info_output) {
  std::ofstream out(info_output.get_graph_nodes_filename(source_name));
  out << "NodeIndex,NodeId,Label,Type,ResName,ResNr,Chain,IsLig,State,"
         "Macrostate,Set,ComponentId,Degree,Strength,StrengthNorm,Betweenness,"
         "BetweennessNorm,CommunityId,CommunitySize\n";
  out << std::setprecision(15);
  for (std::size_t node = 0; node < result.filtered_graph.nodes.size(); ++node) {
    const auto &meta = result.filtered_graph.nodes[node];
    const int component =
        node < result.component_ids.size() ? result.component_ids[node] + 1 : 0;
    const int degree =
        node < result.node_degree.size() ? result.node_degree[node] : 0;
    const double strength =
        node < result.node_strength.size() ? result.node_strength[node] : 0.0;
    const double strength_norm = node < result.node_strength_norm.size()
                                     ? result.node_strength_norm[node]
                                     : 0.0;
    const double betweenness = node < result.node_betweenness.size()
                                   ? result.node_betweenness[node]
                                   : 0.0;
    const double betweenness_norm = node < result.node_betweenness_norm.size()
                                        ? result.node_betweenness_norm[node]
                                        : 0.0;
    const int community =
        (result.community_available && node < result.community_membership.size())
            ? (result.community_membership[node] + 1)
            : 0;
    const int community_size =
        (result.community_available && node < result.community_membership.size() &&
         result.community_membership[node] >= 0 &&
         static_cast<std::size_t>(result.community_membership[node]) <
             result.community_stats.community_sizes.size())
            ? result.community_stats.community_sizes[static_cast<std::size_t>(
                  result.community_membership[node])]
            : 0;
    out << node << ',' << residue_node_id(meta) << ',' << meta.label << ','
        << meta.type << ',' << meta.resname << ',' << meta.resnr << ','
        << meta.chain << ',' << meta.is_lig << ',' << meta.state << ','
        << meta.macrostate << ',' << meta.set_label << ',' << component << ','
        << degree << ',' << strength << ',' << strength_norm << ','
        << betweenness << ',' << betweenness_norm << ',' << community << ','
        << community_size << "\n";
  }
}

template <typename ScoreFn>
void write_ranked_edge_metric_file(const std::string &filename,
                                   const AnalysisResult &result,
                                   ScoreFn score_fn,
                                   const std::string &metric_name) {
  std::ofstream out(filename);
  out << "Rank,Source,Target,Weight,RawScore," << metric_name << "\n";
  out << std::setprecision(15);
  std::vector<std::size_t> order(result.filtered_graph.edges.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const double lhs_score = score_fn(lhs);
                     const double rhs_score = score_fn(rhs);
                     if (lhs_score != rhs_score)
                       return lhs_score > rhs_score;
                     const auto &a = result.filtered_graph.edges[lhs];
                     const auto &b = result.filtered_graph.edges[rhs];
                     if (a.weight != b.weight)
                       return a.weight > b.weight;
                     if (a.source != b.source)
                       return a.source < b.source;
                     return a.target < b.target;
                   });
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const std::size_t edge_idx = order[rank];
    const auto &edge = result.filtered_graph.edges[edge_idx];
    const auto &source = result.filtered_graph.nodes[static_cast<std::size_t>(edge.source)];
    const auto &target = result.filtered_graph.nodes[static_cast<std::size_t>(edge.target)];
    out << (rank + 1) << ',' << residue_node_id(source) << ','
        << residue_node_id(target) << ',' << edge.weight << ','
        << edge.raw_score << ',' << score_fn(edge_idx) << "\n";
  }
}

template <typename ScoreFn>
void write_ranked_node_metric_file(const std::string &filename,
                                   const AnalysisResult &result,
                                   ScoreFn score_fn,
                                   const std::string &metric_name) {
  std::ofstream out(filename);
  out << "Rank,NodeId,Label,Type,ResName,ResNr,Chain,IsLig,State,Macrostate,Set,"
         "Strength,"
      << metric_name << "\n";
  out << std::setprecision(15);
  std::vector<std::size_t> order(result.filtered_graph.nodes.size(), 0);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     const double lhs_score = score_fn(lhs);
                     const double rhs_score = score_fn(rhs);
                     if (lhs_score != rhs_score)
                       return lhs_score > rhs_score;
                     const double lhs_strength =
                         lhs < result.node_strength.size() ? result.node_strength[lhs] : 0.0;
                     const double rhs_strength =
                         rhs < result.node_strength.size() ? result.node_strength[rhs] : 0.0;
                     if (lhs_strength != rhs_strength)
                       return lhs_strength > rhs_strength;
                     return residue_node_id(result.filtered_graph.nodes[lhs]) <
                            residue_node_id(result.filtered_graph.nodes[rhs]);
                   });
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    const std::size_t node_idx = order[rank];
    const auto &meta = result.filtered_graph.nodes[node_idx];
    const double strength =
        node_idx < result.node_strength.size() ? result.node_strength[node_idx]
                                               : 0.0;
    out << (rank + 1) << ',' << residue_node_id(meta) << ',' << meta.label << ','
        << meta.type << ',' << meta.resname << ',' << meta.resnr << ','
        << meta.chain << ',' << meta.is_lig << ',' << meta.state << ','
        << meta.macrostate << ',' << meta.set_label << ',' << strength << ','
        << score_fn(node_idx) << "\n";
  }
}

void write_tpt_bottleneck_edges_csv(const std::string &filename,
                                    const std::vector<TptCapturedEdgeRow> &rows) {
  std::ofstream out(filename);
  out << "Rank,From,To,FromMacro,ToMacro,GrossFlux,NetFlux,RelativeNetFlux,"
         "FromQ,ToQ,MidpointQ,TransitionProximity,BottleneckScore,"
         "DominantPathSupportFlux,DominantPathSupportFraction,"
         "DominantPathCount\n";
  out << std::setprecision(15);
  for (std::size_t idx = 0; idx < rows.size(); ++idx) {
    const auto &row = rows[idx];
    out << (idx + 1) << ',' << row.from << ',' << row.to << ','
        << row.from_macro << ',' << row.to_macro << ',' << row.gross_flux << ','
        << row.net_flux << ',' << row.relative_net_flux << ',' << row.from_q
        << ',' << row.to_q << ',' << row.midpoint_q << ','
        << row.transition_proximity << ',' << row.bottleneck_score << ','
        << row.dominant_path_support_flux << ','
        << row.dominant_path_support_fraction << ','
        << row.dominant_path_count << "\n";
  }
}

void write_tpt_bottleneck_nodes_csv(const std::string &filename,
                                    const std::vector<TptCapturedNodeRow> &rows) {
  std::ofstream out(filename);
  out << "Rank,State,Set,Macrostate,Pi,QPlus,QMinus,ReactiveDensity,NetInFlux,"
         "NetOutFlux,ThroughFlux,TransitionProximity,BottleneckScore,"
         "DominantPathSupportFlux,DominantPathSupportFraction,"
         "DominantPathCount\n";
  out << std::setprecision(15);
  for (std::size_t idx = 0; idx < rows.size(); ++idx) {
    const auto &row = rows[idx];
    out << (idx + 1) << ',' << row.state << ',' << row.label << ','
        << row.macro << ',' << row.pi << ',' << row.q_plus << ','
        << row.q_minus << ',' << row.reactive_density << ',' << row.net_in_flux
        << ',' << row.net_out_flux << ',' << row.through_flux << ','
        << row.transition_proximity << ',' << row.bottleneck_score << ','
        << row.dominant_path_support_flux << ','
        << row.dominant_path_support_fraction << ','
        << row.dominant_path_count << "\n";
  }
}

void write_legacy_tpt_bottleneck_edges_dat(
    const std::string &filename, const std::vector<TptCapturedEdgeRow> &rows) {
  std::ofstream out(filename);
  out << "# rank from to from_macro to_macro gross_flux net_flux "
         "relative_net_flux from_q to_q midpoint_q transition_proximity "
         "bottleneck_score dominant_path_support_flux "
         "dominant_path_support_fraction dominant_path_count\n";
  out << std::setprecision(15);
  for (std::size_t idx = 0; idx < rows.size(); ++idx) {
    const auto &row = rows[idx];
    out << (idx + 1) << ' ' << row.from << ' ' << row.to << ' '
        << row.from_macro << ' ' << row.to_macro << ' ' << row.gross_flux
        << ' ' << row.net_flux << ' ' << row.relative_net_flux << ' '
        << row.from_q << ' ' << row.to_q << ' ' << row.midpoint_q << ' '
        << row.transition_proximity << ' ' << row.bottleneck_score << ' '
        << row.dominant_path_support_flux << ' '
        << row.dominant_path_support_fraction << ' '
        << row.dominant_path_count << '\n';
  }
}

void write_legacy_tpt_bottleneck_states_dat(
    const std::string &filename, const std::vector<TptCapturedNodeRow> &rows) {
  std::ofstream out(filename);
  out << "# rank state set macrostate pi q_plus q_minus reactive_density "
         "net_in_flux net_out_flux through_flux transition_proximity "
         "bottleneck_score dominant_path_support_flux "
         "dominant_path_support_fraction dominant_path_count\n";
  out << std::setprecision(15);
  for (std::size_t idx = 0; idx < rows.size(); ++idx) {
    const auto &row = rows[idx];
    out << (idx + 1) << ' ' << row.state << ' ' << row.label << ' '
        << row.macro << ' ' << row.pi << ' ' << row.q_plus << ' '
        << row.q_minus << ' ' << row.reactive_density << ' ' << row.net_in_flux
        << ' ' << row.net_out_flux << ' ' << row.through_flux << ' '
        << row.transition_proximity << ' ' << row.bottleneck_score << ' '
        << row.dominant_path_support_flux << ' '
        << row.dominant_path_support_fraction << ' '
        << row.dominant_path_count << '\n';
  }
}

void write_legacy_lspmd_outputs(const DCCMNetwork &dccm,
                                const AnalysisResult &result) {
  const auto &nodes = result.filtered_graph.nodes;
  {
    std::ofstream out("lspmd_edges.csv");
    out << "Source,Target,Z_avg,Weight,D_ij\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &edge : result.filtered_graph.edges) {
      const auto &source = nodes[static_cast<std::size_t>(edge.source)];
      const auto &target = nodes[static_cast<std::size_t>(edge.target)];
      const double z_avg = -std::log(std::max(edge.weight, 1e-15));
      out << residue_node_id(source) << ',' << residue_node_id(target) << ','
          << z_avg << ',' << edge.weight << ',' << z_avg << "\n";
    }
  }
  {
    const double max_strength = max_or_one(result.node_strength);
    std::ofstream out("lspmd_degree_centrality.csv");
    out << "Residue,ResNr,Chain,IsLig,DC,DC_norm\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
      out << nodes[node].resname << ',' << nodes[node].resnr << ','
          << nodes[node].chain << ',' << nodes[node].is_lig << ','
          << (node < result.node_strength.size() ? result.node_strength[node] : 0.0)
          << ','
          << safe_ratio(node < result.node_strength.size()
                            ? result.node_strength[node]
                            : 0.0,
                        max_strength)
          << "\n";
    }
  }
  {
    const double max_bc = max_or_one(result.node_betweenness);
    std::ofstream out("lspmd_betweenness_centrality.csv");
    out << "Residue,ResNr,Chain,IsLig,BC,BC_norm\n";
    out << std::fixed << std::setprecision(6);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
      out << nodes[node].resname << ',' << nodes[node].resnr << ','
          << nodes[node].chain << ',' << nodes[node].is_lig << ','
          << (node < result.node_betweenness.size()
                  ? result.node_betweenness[node]
                  : 0.0)
          << ','
          << safe_ratio(node < result.node_betweenness.size()
                            ? result.node_betweenness[node]
                            : 0.0,
                        max_bc)
          << "\n";
    }
  }
  (void)dccm;
}

} // namespace

std::string source_kind_name(SourceKind source) {
  switch (source) {
  case SourceKind::Dccm:
    return "dccm";
  case SourceKind::Lspmd:
    return "lspmd";
  case SourceKind::AllosteryGc:
    return "allostery-gc";
  case SourceKind::AllosteryTe:
    return "allostery-te";
  case SourceKind::Tpt:
    return "tpt";
  }
  return "unknown";
}

AnalysisResult analyze_graph(const SourceGraph &graph, const AnalysisConfig &cfg,
                             TitanExecutor &executor) {
  AnalysisResult result;
  result.filtered_graph = apply_topk_filter(graph, cfg.filter_topk);
  result.filtered_graph =
      apply_component_filter(result.filtered_graph, cfg.filter_component);
  result.component_ids = compute_component_ids(result.filtered_graph);
  result.component_count =
      result.component_ids.empty() ? 0 : (1 + max_or_zero(result.component_ids));

  compute_strength_degree(result.filtered_graph, result.node_degree,
                          result.node_strength);
  result.node_strength_norm.resize(result.node_strength.size(), 0.0);
  const double max_strength = max_or_one(result.node_strength);
  for (std::size_t idx = 0; idx < result.node_strength.size(); ++idx) {
    result.node_strength_norm[idx] =
        safe_ratio(result.node_strength[idx], max_strength);
  }

  const auto adjacency = build_adjacency(result.filtered_graph);
  compute_brandes_betweenness(result.filtered_graph, adjacency,
                              result.node_betweenness, result.edge_betweenness,
                              executor);

  result.node_betweenness_norm.resize(result.node_betweenness.size(), 0.0);
  result.edge_betweenness_norm.resize(result.edge_betweenness.size(), 0.0);
  const double max_node_bc = max_or_one(result.node_betweenness);
  const double max_edge_bc = max_or_one(result.edge_betweenness);
  for (std::size_t idx = 0; idx < result.node_betweenness.size(); ++idx) {
    result.node_betweenness_norm[idx] =
        safe_ratio(result.node_betweenness[idx], max_node_bc);
  }
  for (std::size_t idx = 0; idx < result.edge_betweenness.size(); ++idx) {
    result.edge_betweenness_norm[idx] =
        safe_ratio(result.edge_betweenness[idx], max_edge_bc);
  }

  const bool want_community =
      !result.filtered_graph.directed &&
      is_community_source(result.filtered_graph.source) &&
      (cfg.community_method == "auto" || cfg.community_method == "louvain") &&
      !result.filtered_graph.edges.empty();
  if (want_community) {
    const auto weighted_graph = build_undirected_weighted_graph(result.filtered_graph);
    const auto louvain_result = titan_graph_community::run_louvain(weighted_graph);
    result.community_available = true;
    result.community_membership = louvain_result.membership;
    result.community_stats = louvain_result.stats;
  }

  return result;
}

SourceGraph build_dccm_source_graph(const DCCMNetwork &dccm) {
  SourceGraph graph;
  graph.source = SourceKind::Dccm;
  graph.directed = false;
  graph.nodes = build_residue_nodes(dccm);
  const auto &edges = dccm.get_last_network_graph_edges();
  graph.edges.reserve(edges.size());
  for (const auto &edge : edges) {
    if (edge.source == edge.target || edge.weight <= 0.0)
      continue;
    graph.edges.push_back({edge.source, edge.target, edge.weight,
                           edge.raw_score});
  }
  return graph;
}

SourceGraph build_allostery_source_graph(const DCCMNetwork &dccm,
                                         SourceKind source) {
  SourceGraph graph;
  graph.source = source;
  graph.directed = (source == SourceKind::AllosteryTe);
  graph.nodes = build_residue_nodes(dccm);
  const auto &stored_edges =
      (source == SourceKind::AllosteryTe) ? dccm.get_last_allostery_te_edges()
                                          : dccm.get_last_allostery_gc_edges();
  graph.edges.reserve(stored_edges.size());
  for (const auto &edge : stored_edges) {
    if (edge.source == edge.target || edge.weight <= 0.0)
      continue;
    graph.edges.push_back({edge.source, edge.target, edge.weight,
                           edge.raw_score});
  }
  return graph;
}

SourceGraph build_lspmd_source_graph(const DCCMNetwork &dccm,
                                     const DCCMNetwork::LSPMDResult &result) {
  SourceGraph graph;
  graph.source = SourceKind::Lspmd;
  graph.directed = false;
  graph.nodes = build_residue_nodes(dccm);
  graph.edges.reserve(result.edges.size());
  for (const auto &[source, target, weight] : result.edges) {
    if (source == target || weight <= 0.0)
      continue;
    graph.edges.push_back({source, target, weight, weight});
  }
  return graph;
}

SourceGraph build_tpt_source_graph(const TitanTPT &analyzer,
                                   const std::vector<int> &macro_assignments) {
  SourceGraph graph;
  graph.source = SourceKind::Tpt;
  graph.directed = true;

  const int n_states = std::max(0, analyzer.n_states());
  graph.nodes.resize(static_cast<std::size_t>(n_states));
  std::vector<unsigned char> is_source(graph.nodes.size(), 0);
  std::vector<unsigned char> is_sink(graph.nodes.size(), 0);
  for (int state : analyzer.source_states()) {
    if (state >= 0 && static_cast<std::size_t>(state) < is_source.size())
      is_source[static_cast<std::size_t>(state)] = 1;
  }
  for (int state : analyzer.sink_states()) {
    if (state >= 0 && static_cast<std::size_t>(state) < is_sink.size())
      is_sink[static_cast<std::size_t>(state)] = 1;
  }

  for (int state = 0; state < n_states; ++state) {
    auto &node = graph.nodes[static_cast<std::size_t>(state)];
    node.state = state;
    node.type = "Microstate";
    node.label = "state_" + std::to_string(state);
    node.node_id = node.label;
    node.macrostate =
        (static_cast<std::size_t>(state) < macro_assignments.size())
            ? macro_assignments[static_cast<std::size_t>(state)]
            : -1;
    node.set_label = is_source[static_cast<std::size_t>(state)]
                         ? 'A'
                         : (is_sink[static_cast<std::size_t>(state)] ? 'B' : 'I');
  }

  graph.edges.reserve(analyzer.edges().size());
  for (const auto &edge : analyzer.edges()) {
    if (edge.from == edge.to || edge.net_flux <= 0.0)
      continue;
    graph.edges.push_back(
        {edge.from, edge.to, std::max(0.0, edge.net_flux), edge.net_flux});
  }
  return graph;
}

void TitanUnifiedGraphRuntime::configure(const arguments &args) {
  enabled_ = args.graph.flag_graph || args.community.flag_community;
  auto_sources_ = true;
  requested_sources_.clear();
  analysis_cfg_.community_method = args.graph.community_method;
  analysis_cfg_.filter_topk = args.graph.filter_topk;
  analysis_cfg_.filter_component = args.graph.filter_component;
  have_lspmd_capture_ = false;
  lspmd_graph_ = {};
  have_tpt_capture_ = false;
  tpt_graph_ = {};
  tpt_edge_rows_.clear();
  tpt_node_rows_.clear();
  if (!enabled_)
    return;

  for (const std::string &source_name : args.graph.sources) {
    if (source_name == "auto") {
      auto_sources_ = true;
      requested_sources_.clear();
      break;
    }
    auto_sources_ = false;
    if (source_name == "dccm")
      requested_sources_.push_back(SourceKind::Dccm);
    else if (source_name == "lspmd")
      requested_sources_.push_back(SourceKind::Lspmd);
    else if (source_name == "allostery-gc")
      requested_sources_.push_back(SourceKind::AllosteryGc);
    else if (source_name == "allostery-te")
      requested_sources_.push_back(SourceKind::AllosteryTe);
    else if (source_name == "tpt")
      requested_sources_.push_back(SourceKind::Tpt);
  }
  if (requested_sources_.empty())
    auto_sources_ = true;
}

bool TitanUnifiedGraphRuntime::wants_source(SourceKind source) const noexcept {
  if (!enabled_)
    return false;
  if (auto_sources_)
    return true;
  return std::find(requested_sources_.begin(), requested_sources_.end(), source) !=
         requested_sources_.end();
}

void TitanUnifiedGraphRuntime::capture_lspmd(
    const DCCMNetwork &dccm, const DCCMNetwork::LSPMDResult &result) {
  if (!wants_source(SourceKind::Lspmd))
    return;
  lspmd_graph_ = build_lspmd_source_graph(dccm, result);
  have_lspmd_capture_ = true;
}

void TitanUnifiedGraphRuntime::capture_tpt(
    const TitanTPT &analyzer, const std::vector<int> &macro_assignments,
    const std::vector<TptCapturedEdgeRow> &edge_rows,
    const std::vector<TptCapturedNodeRow> &node_rows) {
  if (!wants_source(SourceKind::Tpt))
    return;
  tpt_graph_ = build_tpt_source_graph(analyzer, macro_assignments);
  tpt_edge_rows_ = edge_rows;
  tpt_node_rows_ = node_rows;
  have_tpt_capture_ = true;
}

void TitanUnifiedGraphRuntime::export_all(const arguments &args,
                                          DCCMNetwork &dccm,
                                          TitanExecutor &executor,
                                          info &info_output,
                                          std::fstream &log) const {
  if (!enabled_)
    return;

  struct SourceExportBundle {
    SourceKind source{SourceKind::Dccm};
    SourceGraph graph{};
    bool available{false};
    AnalysisResult analysis{};
  };

  std::vector<SourceExportBundle> bundles;
  auto add_source = [&](SourceKind source, SourceGraph graph, bool available) {
    if (!wants_source(source))
      return;
    SourceExportBundle bundle;
    bundle.source = source;
    bundle.graph = std::move(graph);
    bundle.available = available;
    if (available)
      bundle.analysis = analyze_graph(bundle.graph, analysis_cfg_, executor);
    bundles.push_back(std::move(bundle));
  };

  if (wants_source(SourceKind::Dccm)) {
    SourceGraph graph = build_dccm_source_graph(dccm);
    const bool available = !graph.nodes.empty();
    add_source(SourceKind::Dccm, std::move(graph), available);
  }
  if (wants_source(SourceKind::AllosteryGc)) {
    SourceGraph graph = build_allostery_source_graph(dccm, SourceKind::AllosteryGc);
    const bool available = !graph.nodes.empty();
    add_source(SourceKind::AllosteryGc, std::move(graph), available);
  }
  if (wants_source(SourceKind::AllosteryTe)) {
    SourceGraph graph = build_allostery_source_graph(dccm, SourceKind::AllosteryTe);
    const bool available = !graph.nodes.empty();
    add_source(SourceKind::AllosteryTe, std::move(graph), available);
  }
  if (wants_source(SourceKind::Lspmd)) {
    add_source(SourceKind::Lspmd, lspmd_graph_, have_lspmd_capture_);
  }
  if (wants_source(SourceKind::Tpt)) {
    add_source(SourceKind::Tpt, tpt_graph_, have_tpt_capture_);
  }

  std::ofstream summary_out(info_output.get_graph_summary_filename());
  summary_out << "module: titan_sparse_graph\n";
  summary_out << "community_method: " << analysis_cfg_.community_method << "\n";
  summary_out << "filter_topk: " << analysis_cfg_.filter_topk << "\n";
  summary_out << "filter_component: " << analysis_cfg_.filter_component << "\n";
  summary_out << "requested_sources: ";
  if (auto_sources_) {
    summary_out << "auto\n";
  } else {
    for (std::size_t idx = 0; idx < requested_sources_.size(); ++idx) {
      if (idx > 0)
        summary_out << ",";
      summary_out << source_kind_name(requested_sources_[idx]);
    }
    summary_out << "\n";
  }

  std::size_t exported_sources = 0;
  for (const auto &bundle : bundles) {
    const std::string source_name = source_kind_name(bundle.source);
    summary_out << "\n[source." << source_name << "]\n";
    summary_out << "available: " << (bundle.available ? "yes" : "no") << "\n";
    summary_out << "directed: " << (bundle.graph.directed ? "yes" : "no")
                << "\n";
    summary_out << "nodes: " << bundle.graph.nodes.size() << "\n";
    summary_out << "input_edges: " << bundle.graph.edges.size() << "\n";
    if (bundle.available) {
      summary_out << "filtered_nodes: "
                  << bundle.analysis.filtered_graph.nodes.size() << "\n";
      summary_out << "filtered_edges: "
                  << bundle.analysis.filtered_graph.edges.size() << "\n";
      summary_out << "components: " << bundle.analysis.component_count << "\n";
      summary_out << "community_available: "
                  << (bundle.analysis.community_available ? "yes" : "no")
                  << "\n";
      write_edges_file(source_name, bundle.analysis, info_output);
      write_nodes_file(source_name, bundle.analysis, info_output);
      write_community_files(source_name, bundle.analysis, info_output);
      if (bundle.source == SourceKind::Tpt && !tpt_edge_rows_.empty()) {
        write_tpt_bottleneck_edges_csv(
            info_output.get_graph_bottleneck_edges_filename(source_name),
            tpt_edge_rows_);
        write_tpt_bottleneck_nodes_csv(
            info_output.get_graph_bottleneck_nodes_filename(source_name),
            tpt_node_rows_);
        write_legacy_tpt_bottleneck_edges_dat(
            info_output.get_tpt_bottleneck_edges_filename(), tpt_edge_rows_);
        write_legacy_tpt_bottleneck_states_dat(
            info_output.get_tpt_bottleneck_states_filename(), tpt_node_rows_);
      } else {
        write_ranked_edge_metric_file(
            info_output.get_graph_bottleneck_edges_filename(source_name),
            bundle.analysis,
            [&](std::size_t edge_idx) {
              return edge_idx < bundle.analysis.edge_betweenness.size()
                         ? bundle.analysis.edge_betweenness[edge_idx]
                         : 0.0;
            },
            "EdgeBetweenness");
        write_ranked_node_metric_file(
            info_output.get_graph_bottleneck_nodes_filename(source_name),
            bundle.analysis,
            [&](std::size_t node_idx) {
              return node_idx < bundle.analysis.node_betweenness.size()
                         ? bundle.analysis.node_betweenness[node_idx]
                         : 0.0;
            },
            "NodeBetweenness");
      }

      if (bundle.source == SourceKind::Dccm &&
          (analysis_cfg_.community_method == "auto" ||
           analysis_cfg_.community_method == "louvain")) {
        if (bundle.analysis.community_available) {
          dccm.export_community_outputs_from_graph(
              build_undirected_weighted_graph(bundle.analysis.filtered_graph),
              bundle.analysis.community_membership, info_output);
        } else {
          dccm.export_community_outputs_from_graph(
              titan_sparse::WeightedCSRGraph<double>{}, {}, info_output);
        }
      }
      if (bundle.source == SourceKind::Lspmd) {
        write_legacy_lspmd_outputs(dccm, bundle.analysis);
      }
      ++exported_sources;
    } else {
      write_edges_file(source_name, AnalysisResult{}, info_output);
      write_nodes_file(source_name, AnalysisResult{}, info_output);
      write_empty_community_files(info_output, source_name);
      {
        std::ofstream out(
            info_output.get_graph_bottleneck_edges_filename(source_name));
        out << "Rank,Source,Target,Weight,RawScore,EdgeBetweenness\n";
      }
      {
        std::ofstream out(
            info_output.get_graph_bottleneck_nodes_filename(source_name));
        out << "Rank,NodeId,Label,Type,ResName,ResNr,Chain,IsLig,State,"
               "Macrostate,Set,Strength,NodeBetweenness\n";
      }
    }
  }

  TitanUI::log_kv(log, "Graph Summary File",
                  info_output.get_graph_summary_filename());
  TitanUI::log_kv(log, "Graph Exported Sources",
                  std::to_string(exported_sources));
  TitanUI::print_info("Graph: Exported unified graph analytics to " +
                      info_output.get_graph_summary_filename());
  (void)args;
}

} // namespace titan_sparse_graph_module

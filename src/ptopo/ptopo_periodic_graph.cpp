#include "ptopo_periodic_graph.h"
#include "ptopo_potential_dsu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace titan_ptopo {
namespace {

constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// Edge indices in edge-ID order; the sort is skipped when the input is sorted.
std::vector<std::uint32_t> edge_order(const Graph &g) {
  std::vector<std::uint32_t> order(g.edges.size());
  std::iota(order.begin(), order.end(), 0u);
  const bool sorted = std::is_sorted(
      g.edges.begin(), g.edges.end(),
      [](const Edge &a, const Edge &b) { return a.id < b.id; });
  if (!sorted)
    std::sort(order.begin(), order.end(),
              [&](std::uint32_t a, std::uint32_t b) {
                return g.edges[a].id < g.edges[b].id;
              });
  return order;
}

struct ResidualRecord {
  std::uint32_t root = 0;
  Z3 residual{};
};

// Shared by audit() and audit_reference(): component partition from the
// root of each node, per-component edge counts, subgroup of the nonzero
// cycle residuals, Smith factors and the summary ranks.
Result<GraphAudit> assemble(const Graph &g,
                            const std::vector<std::uint32_t> &root,
                            const std::vector<ResidualRecord> &residuals) {
  const std::size_t n = g.nodes.size();
  GraphAudit out;
  std::vector<std::uint32_t> component_of_root(n, kNone);
  std::vector<ComponentAudit> components;
  std::vector<std::uint32_t> node_component(n, kNone);
  for (std::uint32_t i = 0; i < n; ++i) {
    std::uint32_t &slot = component_of_root[root[i]];
    if (slot == kNone) {
      slot = static_cast<std::uint32_t>(components.size());
      components.emplace_back();
      components.back().canonical_node_id = g.nodes[i].id;
    }
    ComponentAudit &c = components[slot];
    c.node_indices.push_back(i);
    c.canonical_node_id = std::min(c.canonical_node_id, g.nodes[i].id);
    node_component[i] = slot;
  }
  for (const Edge &e : g.edges)
    ++components[component_of_root[root[e.u]]].edge_count;

  std::vector<Hnf3> hnfs(components.size());
  for (const ResidualRecord &r : residuals) {
    auto step = hnf_insert(hnfs[component_of_root[r.root]], r.residual);
    if (!step)
      return std::unexpected(step.error());
  }

  Hnf3 aggregate;
  for (std::size_t k = 0; k < components.size(); ++k) {
    ComponentAudit &c = components[k];
    // A connected component has E >= V - 1.
    c.beta1 = c.edge_count + 1 - c.node_indices.size();
    c.winding_hnf = hnfs[k].to_basis();
    auto smith = smith_invariants(c.winding_hnf);
    if (!smith)
      return std::unexpected(smith.error());
    c.cokernel = std::move(*smith);
    c.ambient_null_rank = c.beta1 - c.winding_hnf.size();
    out.beta1 += c.beta1;
    out.componentwise_ambient_null_rank += c.ambient_null_rank;
    out.maximum_component_winding_rank =
        std::max(out.maximum_component_winding_rank, c.winding_hnf.size());
    auto merged = hnf_merge(aggregate, hnfs[k]);
    if (!merged)
      return std::unexpected(merged.error());
  }

  // Deterministic order: canonical (minimum stable) node ID.
  std::vector<std::uint32_t> order(components.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
    return components[a].canonical_node_id < components[b].canonical_node_id;
  });
  std::vector<std::uint32_t> rank_of(components.size());
  out.components.reserve(components.size());
  for (std::uint32_t pos = 0; pos < order.size(); ++pos) {
    rank_of[order[pos]] = pos;
    out.components.push_back(std::move(components[order[pos]]));
  }
  for (std::uint32_t &c : node_component)
    c = rank_of[c];
  out.node_component = std::move(node_component);
  out.beta0 = out.components.size();
  out.aggregate_winding_hnf = aggregate.to_basis();
  out.global_ambient_null_rank =
      out.beta1 - out.aggregate_winding_hnf.size();
  return out;
}

// Rooted spanning forest over the union edges, for cycle witnesses.
struct Forest {
  std::vector<std::uint32_t> parent;      // kNone at a tree root
  std::vector<std::uint32_t> parent_edge; // edge index to the parent
  std::vector<std::uint32_t> depth;
};

Forest build_forest(const Graph &g, const std::vector<std::uint32_t> &tree) {
  const std::size_t n = g.nodes.size();
  std::vector<std::uint32_t> offsets(n + 1, 0);
  for (const std::uint32_t t : tree) {
    ++offsets[g.edges[t].u + 1];
    ++offsets[g.edges[t].v + 1];
  }
  for (std::size_t i = 1; i <= n; ++i)
    offsets[i] += offsets[i - 1];
  std::vector<std::uint32_t> adjacency(offsets[n]);
  std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end() - 1);
  for (const std::uint32_t t : tree) {
    adjacency[cursor[g.edges[t].u]++] = t;
    adjacency[cursor[g.edges[t].v]++] = t;
  }
  Forest f;
  f.parent.assign(n, kNone);
  f.parent_edge.assign(n, kNone);
  f.depth.assign(n, 0);
  std::vector<unsigned char> seen(n, 0);
  std::vector<std::uint32_t> queue;
  queue.reserve(n);
  for (std::uint32_t seed = 0; seed < n; ++seed) {
    if (seen[seed])
      continue;
    seen[seed] = 1;
    queue.clear();
    queue.push_back(seed);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::uint32_t x = queue[head];
      for (std::uint32_t k = offsets[x]; k < offsets[x + 1]; ++k) {
        const Edge &e = g.edges[adjacency[k]];
        const std::uint32_t y = e.u == x ? e.v : e.u;
        if (seen[y])
          continue;
        seen[y] = 1;
        f.parent[y] = x;
        f.parent_edge[y] = adjacency[k];
        f.depth[y] = f.depth[x] + 1;
        queue.push_back(y);
      }
    }
  }
  return f;
}

// Orientation of tree edge `t` when walked from child x to its parent.
std::int8_t up_orientation(const Graph &g, std::uint32_t t, std::uint32_t x) {
  return g.edges[t].u == x ? std::int8_t{1} : std::int8_t{-1};
}

Result<std::vector<CycleWitness>>
collect_witnesses(const Graph &g, const std::vector<std::uint32_t> &tree,
                  const std::vector<std::pair<std::uint32_t, Z3>> &cycles,
                  std::size_t max_steps) {
  const Forest f = build_forest(g, tree);
  std::vector<CycleWitness> out;
  out.reserve(cycles.size());
  std::size_t total = 0;
  std::vector<OrientedEdge> from_v, from_u;
  for (const auto &[index, winding] : cycles) {
    const Edge &e = g.edges[index];
    CycleWitness w;
    w.winding = winding;
    w.walk.push_back({e.id, 1});
    // Tree path v -> u: climb both ends to their lowest common ancestor.
    from_v.clear();
    from_u.clear();
    std::uint32_t a = e.v, b = e.u;
    while (f.depth[a] > f.depth[b]) {
      from_v.push_back({g.edges[f.parent_edge[a]].id,
                        up_orientation(g, f.parent_edge[a], a)});
      a = f.parent[a];
    }
    while (f.depth[b] > f.depth[a]) {
      from_u.push_back({g.edges[f.parent_edge[b]].id,
                        up_orientation(g, f.parent_edge[b], b)});
      b = f.parent[b];
    }
    while (a != b) {
      from_v.push_back({g.edges[f.parent_edge[a]].id,
                        up_orientation(g, f.parent_edge[a], a)});
      a = f.parent[a];
      from_u.push_back({g.edges[f.parent_edge[b]].id,
                        up_orientation(g, f.parent_edge[b], b)});
      b = f.parent[b];
    }
    const std::size_t steps = 1 + from_v.size() + from_u.size();
    if (steps > max_steps - std::min(total, max_steps))
      return make_error(ErrorCode::ResourceLimit,
                        "cycle-witness step budget exceeded; no partial valid "
                        "audit returned");
    total += steps;
    w.walk.insert(w.walk.end(), from_v.begin(), from_v.end());
    // Down from the ancestor to u: reverse order, reverse orientation.
    for (auto it = from_u.rbegin(); it != from_u.rend(); ++it)
      w.walk.push_back({it->edge_id, static_cast<std::int8_t>(-it->orientation)});
    out.push_back(std::move(w));
  }
  return out;
}

} // namespace

Result<void> validate(const Graph &g) {
  if (g.identity_epoch.empty() || g.policy_id.empty() ||
      g.reference_basis_id.empty())
    return make_error(ErrorCode::InvalidInput,
                      "identity_epoch, policy_id and reference_basis_id are "
                      "required");
  if (g.nodes.size() >= kNone || g.edges.size() >= kNone)
    return make_error(ErrorCode::ResourceLimit,
                      "graph exceeds 32-bit node/edge indexing");
  std::vector<std::uint64_t> ids;
  ids.reserve(std::max(g.nodes.size(), g.edges.size()));
  for (const Node &node : g.nodes)
    ids.push_back(node.id);
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    return make_error(ErrorCode::InvalidInput, "duplicate node ID");
  ids.clear();
  for (const Edge &e : g.edges) {
    if (e.u >= g.nodes.size() || e.v >= g.nodes.size() ||
        !std::isfinite(e.activation) || e.activation < 0.0)
      return make_error(ErrorCode::InvalidInput,
                        "invalid edge endpoint/activation");
    ids.push_back(e.id);
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
    return make_error(ErrorCode::InvalidInput, "duplicate edge ID");
  return {};
}

Result<GraphAudit> audit(const Graph &g, const AuditOptions &options) {
  if (auto ok = validate(g); !ok)
    return std::unexpected(ok.error());
  const std::size_t n = g.nodes.size();
  detail::PotentialDsu dsu(n);
  std::vector<std::uint32_t> tree;
  std::vector<std::pair<std::uint32_t, Z3>> cycles; // edge index, residual
  std::vector<std::pair<std::uint32_t, Z3>> nonzero;
  if (options.collect_cycle_witnesses)
    tree.reserve(n);
  for (const std::uint32_t index : edge_order(g)) {
    const Edge &e = g.edges[index];
    detail::DsuInsert step;
    if (!dsu.insert(e.u, e.v, e.shift, step))
      return overflow_error("audit union-find potentials");
    if (step.merged) {
      if (options.collect_cycle_witnesses)
        tree.push_back(index);
      continue;
    }
    // The residual is a cycle sum, so later re-rooting cannot change it.
    const bool winding = !is_zero(step.residual);
    if (winding)
      nonzero.emplace_back(index, step.residual);
    if (options.collect_cycle_witnesses &&
        (winding || !options.witnesses_for_nonzero_winding_only))
      cycles.emplace_back(index, step.residual);
  }
  std::vector<std::uint32_t> root(n);
  Z3 unused{};
  for (std::uint32_t i = 0; i < n; ++i)
    if (!dsu.find(i, root[i], unused))
      return overflow_error("audit union-find potentials");
  std::vector<ResidualRecord> records;
  records.reserve(nonzero.size());
  for (const auto &[index, residual] : nonzero)
    records.push_back({root[g.edges[index].u], residual});
  auto out = assemble(g, root, records);
  if (!out)
    return out;
  if (options.collect_cycle_witnesses) {
    auto witnesses =
        collect_witnesses(g, tree, cycles, options.max_witness_steps);
    if (!witnesses)
      return std::unexpected(witnesses.error());
    out->witnesses = std::move(*witnesses);
  }
  return out;
}

Result<GraphAudit> audit_reference(const Graph &g) {
  if (auto ok = validate(g); !ok)
    return std::unexpected(ok.error());
  const std::size_t n = g.nodes.size();
  const auto order = edge_order(g);
  std::vector<std::uint32_t> offsets(n + 1, 0);
  for (const Edge &e : g.edges) {
    ++offsets[e.u + 1];
    ++offsets[e.v + 1];
  }
  for (std::size_t i = 1; i <= n; ++i)
    offsets[i] += offsets[i - 1];
  struct Arc {
    std::uint32_t to = 0, edge = 0;
    bool forward = true;
  };
  std::vector<Arc> arcs(offsets[n]);
  std::vector<std::uint32_t> cursor(offsets.begin(), offsets.end() - 1);
  for (const std::uint32_t index : order) {
    const Edge &e = g.edges[index];
    arcs[cursor[e.u]++] = {e.v, index, true};
    arcs[cursor[e.v]++] = {e.u, index, false};
  }
  std::vector<unsigned char> seen(n, 0);
  std::vector<std::uint32_t> root(n);
  std::vector<Z3> potential(n);
  std::vector<std::uint32_t> queue;
  queue.reserve(n);
  for (std::uint32_t seed = 0; seed < n; ++seed) {
    if (seen[seed])
      continue;
    seen[seed] = 1;
    root[seed] = seed;
    potential[seed] = Z3{};
    queue.clear();
    queue.push_back(seed);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::uint32_t x = queue[head];
      for (std::uint32_t k = offsets[x]; k < offsets[x + 1]; ++k) {
        const Arc &arc = arcs[k];
        if (seen[arc.to])
          continue;
        seen[arc.to] = 1;
        root[arc.to] = seed;
        const Z3 &shift = g.edges[arc.edge].shift;
        const bool ok =
            arc.forward ? checked::add3(potential[x], shift, potential[arc.to])
                        : checked::sub3(potential[x], shift, potential[arc.to]);
        if (!ok)
          return overflow_error("audit_reference potentials");
        queue.push_back(arc.to);
      }
    }
  }
  std::vector<ResidualRecord> records;
  for (const Edge &e : g.edges) {
    Z3 t{}, w{};
    if (!checked::add3(potential[e.u], e.shift, t) ||
        !checked::sub3(t, potential[e.v], w))
      return overflow_error("audit_reference residual");
    if (!is_zero(w))
      records.push_back({root[e.u], w});
  }
  return assemble(g, root, records);
}

Result<void> verify_witness(const Graph &g, const CycleWitness &w) {
  if (auto ok = validate(g); !ok)
    return ok;
  if (w.walk.empty())
    return make_error(ErrorCode::InvalidInput, "empty witness");
  std::vector<std::pair<std::uint64_t, std::uint32_t>> index(g.edges.size());
  for (std::uint32_t i = 0; i < g.edges.size(); ++i)
    index[i] = {g.edges[i].id, i};
  std::sort(index.begin(), index.end());
  std::uint32_t start = 0, current = 0;
  bool first = true;
  Z3 sum{};
  for (const OrientedEdge &step : w.walk) {
    const auto it = std::lower_bound(
        index.begin(), index.end(), step.edge_id,
        [](const auto &entry, std::uint64_t id) { return entry.first < id; });
    if (it == index.end() || it->first != step.edge_id ||
        (step.orientation != 1 && step.orientation != -1))
      return make_error(ErrorCode::InvalidInput,
                        "unknown edge or invalid witness orientation");
    const Edge &e = g.edges[it->second];
    const std::uint32_t from = step.orientation == 1 ? e.u : e.v;
    const std::uint32_t to = step.orientation == 1 ? e.v : e.u;
    if (first) {
      start = current = from;
      first = false;
    }
    if (from != current)
      return make_error(ErrorCode::InvalidInput,
                        "witness is not a continuous walk");
    current = to;
    const bool ok = step.orientation == 1 ? checked::add3(sum, e.shift, sum)
                                          : checked::sub3(sum, e.shift, sum);
    if (!ok)
      return overflow_error("verify_witness");
  }
  if (current != start || sum != w.winding)
    return make_error(ErrorCode::InvalidInput,
                      "witness closure/winding mismatch");
  return {};
}

bool equivalent_invariants(const GraphAudit &a, const GraphAudit &b) {
  if (a.beta0 != b.beta0 || a.beta1 != b.beta1 ||
      a.aggregate_winding_hnf != b.aggregate_winding_hnf ||
      a.components.size() != b.components.size())
    return false;
  for (std::size_t i = 0; i < a.components.size(); ++i) {
    const ComponentAudit &x = a.components[i];
    const ComponentAudit &y = b.components[i];
    if (x.canonical_node_id != y.canonical_node_id ||
        x.edge_count != y.edge_count || x.beta1 != y.beta1 ||
        x.winding_hnf != y.winding_hnf)
      return false;
  }
  return true;
}

Result<Graph> gauge_transform(const Graph &g, std::span<const Z3> h) {
  if (auto ok = validate(g); !ok)
    return std::unexpected(ok.error());
  if (h.size() != g.nodes.size())
    return make_error(ErrorCode::InvalidInput, "node-gauge size mismatch");
  Graph out = g;
  for (Edge &e : out.edges) {
    Z3 d{};
    if (!checked::sub3(h[e.u], h[e.v], d) ||
        !checked::add3(e.shift, d, e.shift))
      return overflow_error("gauge_transform");
  }
  return out;
}

} // namespace titan_ptopo

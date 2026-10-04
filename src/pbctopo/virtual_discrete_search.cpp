#include "virtual_discrete_search.h"

#include "lattice_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>

namespace titan_pbctopo {
namespace {

constexpr double kEventEps = 1.0e-12;

int compare_double_total(double lhs, double rhs) {
  const bool lhs_nan = std::isnan(lhs);
  const bool rhs_nan = std::isnan(rhs);
  if (lhs_nan != rhs_nan)
    return lhs_nan ? 1 : -1;
  if (lhs_nan)
    return 0;
  if (lhs < rhs)
    return -1;
  if (lhs > rhs)
    return 1;
  return 0;
}

double atom_axis_coord(const PbctopoAtom &atom, int axis) {
  if (axis == 0)
    return atom.sx;
  if (axis == 1)
    return atom.sy;
  return atom.sz;
}

int delta_axis(const Int3 &delta, int axis) {
  if (axis == 0)
    return delta.x;
  if (axis == 1)
    return delta.y;
  return delta.z;
}

double edge_weight(PbctopoLocalEdgeKind kind) {
  switch (kind) {
  case PbctopoLocalEdgeKind::TopologyBond:
    return 1.0;
  case PbctopoLocalEdgeKind::MetadataSkeleton:
    return 0.5;
  case PbctopoLocalEdgeKind::GeometryHeuristic:
  default:
    return 0.25;
  }
}

void adjust_kind_count(PbctopoLocalEdgeKind kind, bool add,
                       PbctopoVirtualAxisState &score) {
  switch (kind) {
  case PbctopoLocalEdgeKind::TopologyBond:
    if (add)
      ++score.topology_broken;
    else
      --score.topology_broken;
    break;
  case PbctopoLocalEdgeKind::MetadataSkeleton:
    if (add)
      ++score.metadata_broken;
    else
      --score.metadata_broken;
    break;
  case PbctopoLocalEdgeKind::GeometryHeuristic:
  default:
    if (add)
      ++score.geometry_broken;
    else
      --score.geometry_broken;
    break;
  }
}

void update_edge_mismatch(std::size_t edge_idx,
                          const std::vector<PbctopoVirtualSweepEdge> &edges,
                          const std::vector<int> &atom_shifts, int axis,
                          std::vector<unsigned char> &mismatched,
                          PbctopoVirtualAxisState &score) {
  const auto &edge = edges[edge_idx];
  const bool now_mismatched =
      atom_shifts[edge.b] - atom_shifts[edge.a] !=
      delta_axis(edge.desired_delta, axis);
  const bool was_mismatched = mismatched[edge_idx] != 0;
  if (now_mismatched == was_mismatched)
    return;

  mismatched[edge_idx] = now_mismatched ? 1 : 0;
  if (now_mismatched) {
    ++score.broken_edges;
    score.seam_weight += edge.weight;
    adjust_kind_count(edge.kind, true, score);
  } else {
    --score.broken_edges;
    score.seam_weight -= edge.weight;
    adjust_kind_count(edge.kind, false, score);
  }
}

bool better_axis_state(const PbctopoVirtualAxisState &lhs,
                       const PbctopoVirtualAxisState &rhs) {
  int cmp = compare_double_total(lhs.seam_weight, rhs.seam_weight);
  if (cmp != 0)
    return cmp < 0;
  if (lhs.topology_broken != rhs.topology_broken)
    return lhs.topology_broken < rhs.topology_broken;
  if (lhs.metadata_broken != rhs.metadata_broken)
    return lhs.metadata_broken < rhs.metadata_broken;
  if (lhs.geometry_broken != rhs.geometry_broken)
    return lhs.geometry_broken < rhs.geometry_broken;
  if (lhs.broken_edges != rhs.broken_edges)
    return lhs.broken_edges < rhs.broken_edges;
  const double lhs_width = lhs.interval_end - lhs.interval_begin;
  const double rhs_width = rhs.interval_end - rhs.interval_begin;
  cmp = compare_double_total(lhs_width, rhs_width);
  if (cmp != 0)
    return cmp > 0;
  return compare_double_total(lhs.center, rhs.center) < 0;
}

void build_axis_states(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoVirtualSweepEdge> &edges,
    const std::vector<std::size_t> &incident_offsets,
    const std::vector<std::size_t> &incident_edges, int axis,
    PbctopoVirtualDiscreteSearchWorkspace &workspace,
    std::vector<PbctopoVirtualAxisState> &states,
    std::size_t &event_count) {
  auto &events = workspace.events;
  events.clear();
  events.reserve(atoms.size() * 2);
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const double coord = atom_axis_coord(atoms[atom_idx], axis);
    if (!std::isfinite(coord))
      continue;
    const std::array<double, 2> atom_events{coord + 0.5, coord + 1.5};
    for (double center : atom_events) {
      if (center + kEventEps < 0.5 || center - kEventEps > 2.5)
        continue;
      events.push_back({std::clamp(center, 0.5, 2.5), atom_idx});
    }
  }
  std::sort(events.begin(), events.end(), [](const auto &lhs, const auto &rhs) {
    const int cmp = compare_double_total(lhs.center, rhs.center);
    if (cmp != 0)
      return cmp < 0;
    return lhs.atom < rhs.atom;
  });

  event_count = events.size();
  auto &atom_shifts = workspace.atom_shifts;
  auto &mismatched = workspace.mismatched;
  atom_shifts.assign(atoms.size(), 0);
  mismatched.assign(edges.size(), 0);
  PbctopoVirtualAxisState current;
  for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    const auto &edge = edges[edge_idx];
    if (delta_axis(edge.desired_delta, axis) == 0)
      continue;
    mismatched[edge_idx] = 1;
    ++current.broken_edges;
    current.seam_weight += edge.weight;
    adjust_kind_count(edge.kind, true, current);
  }

  states.clear();
  states.reserve(events.size() + 1);
  double interval_begin = 0.5;
  std::size_t interval_index = 0;
  auto append_interval = [&](double interval_end) {
    if (!(interval_end > interval_begin + kEventEps))
      return;
    PbctopoVirtualAxisState state = current;
    state.interval_begin = interval_begin;
    state.interval_end = interval_end;
    state.center = 0.5 * (interval_begin + interval_end);
    state.interval_index = interval_index++;
    if (std::fabs(state.seam_weight) <= kEventEps)
      state.seam_weight = 0.0;
    states.push_back(state);
  };

  for (std::size_t event_idx = 0; event_idx < events.size();) {
    const double event_center = events[event_idx].center;
    append_interval(event_center);

    std::size_t group_end = event_idx + 1;
    while (group_end < events.size() &&
           std::fabs(events[group_end].center - event_center) <= kEventEps) {
      ++group_end;
    }
    for (std::size_t idx = event_idx; idx < group_end; ++idx) {
      const std::size_t atom_idx = events[idx].atom;
      ++atom_shifts[atom_idx];
      for (std::size_t pos = incident_offsets[atom_idx];
           pos < incident_offsets[atom_idx + 1]; ++pos) {
        update_edge_mismatch(incident_edges[pos], edges, atom_shifts, axis,
                             mismatched, current);
      }
    }
    interval_begin = event_center;
    event_idx = group_end;
  }
  append_interval(2.5);

  if (states.empty()) {
    current.interval_begin = 0.5;
    current.interval_end = 2.5;
    current.center = 1.5;
    current.interval_index = 0;
    states.push_back(current);
  }
  std::stable_sort(states.begin(), states.end(), better_axis_state);
}

std::uint64_t saturating_product(std::uint64_t lhs, std::uint64_t rhs) {
  if (lhs == 0 || rhs == 0)
    return 0;
  if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
    return std::numeric_limits<std::uint64_t>::max();
  return lhs * rhs;
}

int source_rank(PbctopoVirtualSeedSource source) {
  switch (source) {
  case PbctopoVirtualSeedSource::Previous:
    return 0;
  case PbctopoVirtualSeedSource::Kinematic:
    return 1;
  case PbctopoVirtualSeedSource::Density:
    return 2;
  case PbctopoVirtualSeedSource::None:
  default:
    return 3;
  }
}

void assign_nearest_hint(
    PbctopoVirtualDiscreteState &state,
    const std::vector<PbctopoVirtualDiscreteSeedHint> &seed_hints,
    const PbctopoLatticeMetric &lattice) {
  for (const auto &hint : seed_hints) {
    if (!std::isfinite(hint.cx) || !std::isfinite(hint.cy) ||
        !std::isfinite(hint.cz)) {
      continue;
    }
    const auto nearest = lattice.nearest_image_scaled(
        state.cx - hint.cx, state.cy - hint.cy, state.cz - hint.cz);
    if (!nearest.valid)
      continue;
    const double priority =
        std::isfinite(hint.priority) ? hint.priority : 0.0;
    const bool closer = nearest.d2 + kEventEps < state.hint_distance2;
    const bool same_distance =
        std::fabs(nearest.d2 - state.hint_distance2) <= kEventEps;
    const bool higher_priority = priority > state.hint_priority + kEventEps;
    const bool same_priority =
        std::fabs(priority - state.hint_priority) <= kEventEps;
    if (closer ||
        (same_distance &&
         (higher_priority ||
          (same_priority && source_rank(hint.source) < source_rank(state.source))))) {
      state.hint_distance2 = nearest.d2;
      state.hint_priority = priority;
      state.source = hint.source;
    }
  }
}

bool better_discrete_state(const PbctopoVirtualDiscreteState &lhs,
                           const PbctopoVirtualDiscreteState &rhs) {
  int cmp = compare_double_total(lhs.seam_lower_bound, rhs.seam_lower_bound);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_double_total(lhs.seam_sum, rhs.seam_sum);
  if (cmp != 0)
    return cmp < 0;
  if (lhs.topology_broken_sum != rhs.topology_broken_sum)
    return lhs.topology_broken_sum < rhs.topology_broken_sum;
  if (lhs.metadata_broken_sum != rhs.metadata_broken_sum)
    return lhs.metadata_broken_sum < rhs.metadata_broken_sum;
  if (lhs.geometry_broken_sum != rhs.geometry_broken_sum)
    return lhs.geometry_broken_sum < rhs.geometry_broken_sum;
  cmp = compare_double_total(lhs.hint_distance2, rhs.hint_distance2);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_double_total(lhs.hint_priority, rhs.hint_priority);
  if (cmp != 0)
    return cmp > 0;
  if (source_rank(lhs.source) != source_rank(rhs.source))
    return source_rank(lhs.source) < source_rank(rhs.source);
  cmp = compare_double_total(lhs.cx, rhs.cx);
  if (cmp != 0)
    return cmp < 0;
  cmp = compare_double_total(lhs.cy, rhs.cy);
  if (cmp != 0)
    return cmp < 0;
  return compare_double_total(lhs.cz, rhs.cz) < 0;
}

} // namespace

std::size_t PbctopoVirtualDiscreteSearchWorkspace::capacity_bytes_estimate()
    const noexcept {
  return edges.capacity() * sizeof(PbctopoVirtualSweepEdge) +
         events.capacity() * sizeof(PbctopoVirtualSweepEvent) +
         incident_offsets.capacity() * sizeof(std::size_t) +
         incident_edges.capacity() * sizeof(std::size_t) +
         cursor.capacity() * sizeof(std::size_t) +
         atom_shifts.capacity() * sizeof(int) +
         mismatched.capacity() * sizeof(unsigned char);
}

void build_pbctopo_virtual_discrete_search_plan(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &input_edges, const box &Box,
    const std::vector<PbctopoVirtualDiscreteSeedHint> &seed_hints,
    PbctopoVirtualDiscreteSearchOptions options,
    PbctopoVirtualDiscreteSearchWorkspace &workspace,
    PbctopoVirtualDiscreteSearchPlan &plan) {
  const bool workspace_ready =
      workspace.edges.capacity() >= input_edges.size() &&
      workspace.events.capacity() >= atoms.size() * 2 &&
      workspace.incident_offsets.capacity() >= atoms.size() + 1 &&
      workspace.incident_edges.capacity() >= input_edges.size() * 2 &&
      workspace.cursor.capacity() >= atoms.size() + 1 &&
      workspace.atom_shifts.capacity() >= atoms.size() &&
      workspace.mismatched.capacity() >= input_edges.size();

  plan.x_states.clear();
  plan.y_states.clear();
  plan.z_states.clear();
  plan.states.clear();
  plan.total_state_space = 0;
  plan.omitted_seam_lower_bound = std::numeric_limits<double>::quiet_NaN();
  plan.lower_bound_gap = std::numeric_limits<double>::quiet_NaN();
  plan.search_exhaustive = false;
  plan.branch_and_bound_used = false;
  plan.workspace_reused = workspace_ready;
  plan.event_count_x = 0;
  plan.event_count_y = 0;
  plan.event_count_z = 0;
  plan.workspace_bytes_estimate = 0;

  const PbctopoLatticeMetric lattice(Box);

  auto &edges = workspace.edges;
  edges.clear();
  edges.reserve(input_edges.size());
  for (const auto &edge : input_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b)
      continue;
    const auto &a = atoms[edge.a];
    const auto &b = atoms[edge.b];
    const auto nearest = lattice.nearest_image_scaled(
        b.sx - a.sx, b.sy - a.sy, b.sz - a.sz);
    if (!nearest.valid)
      continue;
    edges.push_back({edge.a, edge.b,
                     {nearest.image_x, nearest.image_y, nearest.image_z},
                     edge.kind, edge_weight(edge.kind)});
  }

  auto &incident_offsets = workspace.incident_offsets;
  incident_offsets.assign(atoms.size() + 1, 0);
  for (const auto &edge : edges) {
    ++incident_offsets[edge.a + 1];
    ++incident_offsets[edge.b + 1];
  }
  for (std::size_t idx = 1; idx < incident_offsets.size(); ++idx)
    incident_offsets[idx] += incident_offsets[idx - 1];
  auto &incident_edges = workspace.incident_edges;
  auto &cursor = workspace.cursor;
  incident_edges.resize(incident_offsets.back());
  cursor.assign(incident_offsets.begin(), incident_offsets.end());
  for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    incident_edges[cursor[edges[edge_idx].a]++] = edge_idx;
    incident_edges[cursor[edges[edge_idx].b]++] = edge_idx;
  }

  build_axis_states(atoms, edges, incident_offsets, incident_edges, 0,
                    workspace, plan.x_states, plan.event_count_x);
  build_axis_states(atoms, edges, incident_offsets, incident_edges, 1,
                    workspace, plan.y_states, plan.event_count_y);
  build_axis_states(atoms, edges, incident_offsets, incident_edges, 2,
                    workspace, plan.z_states, plan.event_count_z);
  plan.total_state_space = saturating_product(
      saturating_product(static_cast<std::uint64_t>(plan.x_states.size()),
                         static_cast<std::uint64_t>(plan.y_states.size())),
      static_cast<std::uint64_t>(plan.z_states.size()));

  const bool small_enough =
      plan.total_state_space <= options.max_exhaustive_states;
  const std::size_t axis_keep = std::max<std::size_t>(1, options.top_k_per_axis);
  const std::size_t keep_x =
      small_enough ? plan.x_states.size()
                   : std::min(axis_keep, plan.x_states.size());
  const std::size_t keep_y =
      small_enough ? plan.y_states.size()
                   : std::min(axis_keep, plan.y_states.size());
  const std::size_t keep_z =
      small_enough ? plan.z_states.size()
                   : std::min(axis_keep, plan.z_states.size());
  plan.search_exhaustive = keep_x == plan.x_states.size() &&
                           keep_y == plan.y_states.size() &&
                           keep_z == plan.z_states.size();

  if (!plan.search_exhaustive) {
    double lower_bound = std::numeric_limits<double>::infinity();
    if (keep_x < plan.x_states.size())
      lower_bound = std::min(lower_bound, plan.x_states[keep_x].seam_weight);
    if (keep_y < plan.y_states.size())
      lower_bound = std::min(lower_bound, plan.y_states[keep_y].seam_weight);
    if (keep_z < plan.z_states.size())
      lower_bound = std::min(lower_bound, plan.z_states[keep_z].seam_weight);
    if (std::isfinite(lower_bound))
      plan.omitted_seam_lower_bound = lower_bound;
  }

  const std::uint64_t selected_state_count = saturating_product(
      saturating_product(static_cast<std::uint64_t>(keep_x),
                         static_cast<std::uint64_t>(keep_y)),
      static_cast<std::uint64_t>(keep_z));
  if (selected_state_count <=
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    plan.states.reserve(static_cast<std::size_t>(selected_state_count));
  }
  for (std::size_t ix = 0; ix < keep_x; ++ix) {
    const auto &x = plan.x_states[ix];
    for (std::size_t iy = 0; iy < keep_y; ++iy) {
      const auto &y = plan.y_states[iy];
      for (std::size_t iz = 0; iz < keep_z; ++iz) {
        const auto &z = plan.z_states[iz];
        PbctopoVirtualDiscreteState state;
        state.cx = x.center;
        state.cy = y.center;
        state.cz = z.center;
        state.seam_lower_bound =
            std::max({x.seam_weight, y.seam_weight, z.seam_weight});
        state.seam_sum = x.seam_weight + y.seam_weight + z.seam_weight;
        state.topology_broken_sum = x.topology_broken + y.topology_broken +
                                    z.topology_broken;
        state.metadata_broken_sum = x.metadata_broken + y.metadata_broken +
                                    z.metadata_broken;
        state.geometry_broken_sum = x.geometry_broken + y.geometry_broken +
                                    z.geometry_broken;
        state.ix = x.interval_index;
        state.iy = y.interval_index;
        state.iz = z.interval_index;
        assign_nearest_hint(state, seed_hints, lattice);
        plan.states.push_back(state);
      }
    }
  }
  std::stable_sort(plan.states.begin(), plan.states.end(),
                   better_discrete_state);
  plan.workspace_bytes_estimate = workspace.capacity_bytes_estimate();
}

PbctopoVirtualDiscreteSearchPlan build_pbctopo_virtual_discrete_search_plan(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &input_edges, const box &Box,
    const std::vector<PbctopoVirtualDiscreteSeedHint> &seed_hints,
    PbctopoVirtualDiscreteSearchOptions options) {
  PbctopoVirtualDiscreteSearchWorkspace workspace;
  PbctopoVirtualDiscreteSearchPlan plan;
  build_pbctopo_virtual_discrete_search_plan(
      atoms, input_edges, Box, seed_hints, options, workspace, plan);
  return plan;
}

} // namespace titan_pbctopo

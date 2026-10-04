#ifndef TITAN_PBCTOPO_VIRTUAL_DISCRETE_SEARCH_H
#define TITAN_PBCTOPO_VIRTUAL_DISCRETE_SEARCH_H

#include "types.h"

#include "../box_type.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace titan_pbctopo {

struct PbctopoVirtualDiscreteSeedHint {
  double cx = 1.5;
  double cy = 1.5;
  double cz = 1.5;
  double priority = 0.0;
  PbctopoVirtualSeedSource source = PbctopoVirtualSeedSource::Density;
};

struct PbctopoVirtualAxisState {
  double interval_begin = 0.5;
  double interval_end = 2.5;
  double center = 1.5;
  double seam_weight = 0.0;
  std::size_t broken_edges = 0;
  std::size_t topology_broken = 0;
  std::size_t metadata_broken = 0;
  std::size_t geometry_broken = 0;
  std::size_t interval_index = 0;
};

struct PbctopoVirtualDiscreteState {
  double cx = 1.5;
  double cy = 1.5;
  double cz = 1.5;
  double seam_lower_bound = 0.0;
  double seam_sum = 0.0;
  double hint_distance2 = std::numeric_limits<double>::infinity();
  double hint_priority = 0.0;
  std::size_t topology_broken_sum = 0;
  std::size_t metadata_broken_sum = 0;
  std::size_t geometry_broken_sum = 0;
  std::size_t ix = 0;
  std::size_t iy = 0;
  std::size_t iz = 0;
  PbctopoVirtualSeedSource source = PbctopoVirtualSeedSource::None;
};

struct PbctopoVirtualDiscreteSearchOptions {
  std::size_t top_k_per_axis = 4;
  std::uint64_t max_exhaustive_states = 4096;
};

struct PbctopoVirtualSweepEdge {
  std::size_t a = 0;
  std::size_t b = 0;
  Int3 desired_delta{};
  PbctopoLocalEdgeKind kind = PbctopoLocalEdgeKind::GeometryHeuristic;
  double weight = 0.25;
};

struct PbctopoVirtualSweepEvent {
  double center = 0.5;
  std::size_t atom = 0;
};

// Reused across frames and axes. These arrays contain only transient sweep
// state; the returned plan owns the axis states and selected 3D states.
struct PbctopoVirtualDiscreteSearchWorkspace {
  std::vector<PbctopoVirtualSweepEdge> edges;
  std::vector<PbctopoVirtualSweepEvent> events;
  std::vector<std::size_t> incident_offsets;
  std::vector<std::size_t> incident_edges;
  std::vector<std::size_t> cursor;
  std::vector<int> atom_shifts;
  std::vector<unsigned char> mismatched;

  std::size_t capacity_bytes_estimate() const noexcept;
};

struct PbctopoVirtualDiscreteSearchPlan {
  std::vector<PbctopoVirtualAxisState> x_states;
  std::vector<PbctopoVirtualAxisState> y_states;
  std::vector<PbctopoVirtualAxisState> z_states;
  std::vector<PbctopoVirtualDiscreteState> states;
  std::uint64_t total_state_space = 0;
  double omitted_seam_lower_bound =
      std::numeric_limits<double>::quiet_NaN();
  double lower_bound_gap = std::numeric_limits<double>::quiet_NaN();
  bool search_exhaustive = false;
  bool branch_and_bound_used = false;
  bool workspace_reused = false;
  std::size_t event_count_x = 0;
  std::size_t event_count_y = 0;
  std::size_t event_count_z = 0;
  std::size_t workspace_bytes_estimate = 0;
};

void build_pbctopo_virtual_discrete_search_plan(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &edges, const box &Box,
    const std::vector<PbctopoVirtualDiscreteSeedHint> &seed_hints,
    PbctopoVirtualDiscreteSearchOptions options,
    PbctopoVirtualDiscreteSearchWorkspace &workspace,
    PbctopoVirtualDiscreteSearchPlan &plan);

PbctopoVirtualDiscreteSearchPlan build_pbctopo_virtual_discrete_search_plan(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<PbctopoLocalEdge> &edges, const box &Box,
    const std::vector<PbctopoVirtualDiscreteSeedHint> &seed_hints,
    PbctopoVirtualDiscreteSearchOptions options = {});

} // namespace titan_pbctopo

#endif // TITAN_PBCTOPO_VIRTUAL_DISCRETE_SEARCH_H

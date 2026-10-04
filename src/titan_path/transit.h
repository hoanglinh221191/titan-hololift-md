#ifndef TITAN_PATH_TRANSIT_H
#define TITAN_PATH_TRANSIT_H

#include "titan_path/geometry.h"
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace titan_path {

// Deliberately separate from RouteMode and native controller/observation state.
enum class TransitPhase {
  Approach,
  CornerTransit,
  HandoffAdvance,
  NotApplicable
};

struct TransitPolicy {
  double clearance_angstrom{1.5};
  double turn_cap_deg{20.0};
  std::array<double, 3> probe_lengths_angstrom{1.0, 0.5, 0.25};
  double lookahead_angstrom{0.5};
  double exit_progress_angstrom{0.05};
  double merit_decrease_angstrom2{1e-6};
  double continuity_half_window_angstrom{1.0};
  double ambiguity_distance_angstrom{0.05};
  double ambiguity_progress_separation_angstrom{0.25};
  double max_total_travel_angstrom{3.0};
  double max_seed_displacement_angstrom{2.5};
  std::size_t max_steps{6};
  std::size_t beam_width{4};
  std::size_t candidate_budget{2048};
};

struct TransitSnapshot {
  std::string route_sha256;
  std::string snapshot_sha256;
  bool geometry_validated{false};
  std::vector<Vec3> registered_waypoints;
  std::vector<double> waypoint_radii;
  std::vector<Vec3> receptor;
  std::vector<Vec3> ligand;
  std::vector<double> ligand_masses;
  Vec3 old_direction{};
};

struct TransitModelStep {
  TransitPhase phase{TransitPhase::NotApplicable};
  Vec3 com{};
  Vec3 displacement{};
  Vec3 direction{};
  double progress_angstrom{0.0};
  double distance_angstrom{0.0};
  double merit_angstrom2{0.0};
  double travel_angstrom{0.0};
  double turn_deg{0.0};
  double sweep_clearance_angstrom{0.0};
};

struct TransitCounters {
  std::size_t proposals{0};
  std::size_t evaluations{0};
  std::size_t bound_rejections{0};
  std::size_t continuity_rejections{0};
  std::size_t ambiguity_rejections{0};
  std::size_t backtrack_rejections{0};
  std::size_t corridor_rejections{0};
  std::size_t merit_rejections{0};
  std::size_t clearance_rejections{0};
  std::size_t accepted_edges{0};
  std::size_t swept_pair_checks{0};
};

struct TransitPlan {
  bool found{false};
  TransitPhase initial_phase{TransitPhase::NotApplicable};
  std::string reason;
  std::size_t vertex{0};
  Vec3 fixed_goal{};
  std::vector<TransitModelStep> model_steps;
  TransitCounters counters;
};

// Every returned state is MODEL_ONLY; no native simulation state is updated.
[[nodiscard]] TransitPlan
plan_bounded_transit(const TransitSnapshot &snapshot,
                     const TransitPolicy &policy = {});
[[nodiscard]] const char *transit_phase_name(TransitPhase phase) noexcept;
} // namespace titan_path
#endif

#ifndef TITAN_PBCTOPO_COMPONENT_SEARCH_H
#define TITAN_PBCTOPO_COMPONENT_SEARCH_H

#include "config.h"
#include "types.h"
#include "../box_type.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace titan_pbctopo {

inline constexpr std::size_t kPbctopoExactAssignmentBudget = 100000;

enum class PbctopoSearchStopReason {
  None,
  StateLimit,
  WallTimeLimit,
  MemoryLimit,
};

const char *pbctopo_search_stop_reason_name(PbctopoSearchStopReason reason);

struct PbctopoComponentSearchLimits {
  std::size_t max_states = 0;
  double max_wall_seconds = 0.0;
  std::size_t max_memory_bytes = 0;
  bool enumerate_feasible_assignments = false;
};

std::size_t pbctopo_component_assignment_upper_bound(
    std::size_t component_count, std::size_t shift_count,
    std::size_t saturation_limit = std::numeric_limits<std::size_t>::max());

struct PbctopoComponentSearchCallbacks {
  using BuildSeedFn = PbctopoComponentCandidate (*)(
      const PbctopoCutCandidate &, const std::vector<std::vector<std::size_t>> &,
      const std::vector<Vec3> &);
  using ScoreAssignmentFn = PbctopoComponentCandidate (*)(
      const std::vector<PbctopoAtom> &, const std::vector<Vec3> &,
      const std::vector<std::vector<std::size_t>> &, double, std::size_t,
      std::size_t, const std::vector<PbctopoContinuityPair> &,
      const std::vector<std::size_t> &, double, double,
      const PbctopoComponentCandidate *, PbctopoTimingStats *, AnchorBackend,
      const PbctopoAnchorGraphContext *);
  using BetterCandidateFn = bool (*)(const PbctopoComponentCandidate &,
                                     const PbctopoComponentCandidate &);

  BuildSeedFn build_seed = nullptr;
  ScoreAssignmentFn score_assignment = nullptr;
  BetterCandidateFn better_candidate = nullptr;
  bool input_atoms_cover_certificate_scope = false;
  bool hard_certificate_domain_complete = false;

  bool ready() const {
    return build_seed != nullptr && score_assignment != nullptr &&
           better_candidate != nullptr;
  }
};

struct PbctopoLeafEvaluation {
  PbctopoComponentCandidate score;
  bool hard_domain_feasible = false;
  bool soft_score_valid = false;
};

struct PbctopoComponentSearchDiagnostics {
  bool attempted = false;
  // Legacy alias for offset_enumeration_exhaustive.
  bool exhaustive = false;
  bool search_completed_under_admissible_bounds = false;
  bool incumbent_bound_pruning_used = false;
  bool best_objective_proven = false;
  bool equivalent_optima_complete = false;
  // True only for the explicit unpruned feasible-set enumeration mode.
  bool feasible_assignment_set_enumerated = false;
  bool bounded_domain_feasibility_known = false;
  bool bounded_domain_feasible = false;
  bool resource_limit_hit = false;
  PbctopoSearchStopReason stop_reason = PbctopoSearchStopReason::None;
  std::size_t states_visited = 0;
  std::size_t memory_bytes_estimate = 0;
  bool memory_budget_is_estimate = true;
  double wall_seconds = 0.0;
  bool offset_enumeration_exhaustive = false;
  bool all_atom_feasibility_exhaustive = false;
  bool certificate_search_exhaustive = false;
  bool beam_used = false;
  std::size_t beam_complete_assignments = 0;
  std::size_t beam_survivors_exactly_rescored = 0;
  std::size_t equivalent_best_assignments_within_domain = 0;
  // Hard-constraint feasible leaves. Defined only when
  // feasible_assignment_set_enumerated is true.
  std::size_t feasible_assignments_within_domain = 0;
  // Scorer-valid subset of feasible_assignments_within_domain.
  std::size_t soft_score_valid_assignments_within_domain = 0;
  std::size_t component_search_shell_radius = 0;
  bool selected_assignment_boundary_known = false;
  bool selected_assignment_touches_shell_boundary = false;
  bool domain_expansion_attempted = false;
  bool domain_expansion_exhausted = false;
  bool bounded_domain_only = false;
  bool optimality_gap_known = false;
  double optimality_gap = std::numeric_limits<double>::quiet_NaN();
};

struct PbctopoComponentSearchSurvivor {
  PbctopoComponentCandidate candidate;
  std::vector<Int3> component_offsets;
};

PbctopoComponentCandidate search_component_shift_candidate(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, std::vector<Vec3> &best_points_out,
    const PbctopoComponentSearchCallbacks &callbacks,
    const std::vector<Int3> *warm_component_offsets = nullptr,
    const PbctopoCutCandidate *incumbent_cut = nullptr,
    const std::vector<Vec3> *incumbent_points = nullptr,
    PbctopoTimingStats *timing = nullptr,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    PbctopoComponentSearchDiagnostics *diagnostics = nullptr,
    std::vector<PbctopoComponentSearchSurvivor> *survivors_out = nullptr,
    std::size_t survivor_limit = 0,
    const std::vector<PbctopoRequiredContact> *required_contacts = nullptr,
    const PbctopoComponentSearchLimits *limits = nullptr);

PbctopoComponentCandidate search_component_shift_candidate_beam(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, std::vector<Vec3> &best_points_out,
    std::size_t beam_width, const PbctopoComponentSearchCallbacks &callbacks,
    const std::vector<Int3> *warm_component_offsets = nullptr,
    const PbctopoCutCandidate *incumbent_cut = nullptr,
    const std::vector<Vec3> *incumbent_points = nullptr,
    PbctopoTimingStats *timing = nullptr,
    AnchorBackend anchor_backend = AnchorBackend::Auto,
    const PbctopoAnchorGraphContext *anchor_context = nullptr,
    std::size_t exact_recovery_budget = 0,
    PbctopoComponentSearchDiagnostics *diagnostics = nullptr,
    std::vector<PbctopoComponentSearchSurvivor> *survivors_out = nullptr,
    std::size_t survivor_limit = 0,
    const std::vector<PbctopoRequiredContact> *required_contacts = nullptr,
    bool allow_adaptive_retry = true);

} // namespace titan_pbctopo

#endif

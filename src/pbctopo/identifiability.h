#ifndef TITAN_PBCTOPO_IDENTIFIABILITY_H
#define TITAN_PBCTOPO_IDENTIFIABILITY_H

#include "evidence_support_policy.h"

#include <cstddef>
#include <span>

namespace titan_pbctopo {

inline constexpr double kPbctopoEvidenceMinConfidence =
    PBCTOPO_SCORED_PAIR_SUPPORT_THRESHOLD;

// Hard feasibility only proves that an image assignment is internally
// consistent and nonclashing.  Relative component images may bypass the
// virtual search or seed later frames only when the evidence graph identifies
// every relative lattice gauge.
inline bool pbctopo_relative_images_identified(
    const PbctopoIdentifiability &identifiability) noexcept {
  return identifiability.identified &&
         identifiability.evidence_graph_connected &&
         identifiability.unconstrained_relative_components == 0;
}

void apply_pbctopo_search_identifiability(
    PbctopoIdentifiability &identifiability,
    bool certificate_search_exhaustive, bool equivalent_optima_complete,
    std::size_t equivalent_best_assignments_within_domain,
    bool feasible_assignment_set_enumerated,
    std::size_t feasible_assignments_within_domain);

void apply_pbctopo_final_selected_evidence(
    PbctopoIdentifiability &identifiability, bool evidence_audit_evaluated,
    bool evidence_graph_connected, std::size_t evidence_component_count,
    std::size_t evidence_edge_count,
    std::size_t explicit_contact_evidence_edges,
    std::size_t scored_contact_evidence_edges,
    std::size_t evidence_connected_component_count);

PbctopoIdentifiability evaluate_pbctopo_identifiability(
    std::size_t component_count,
    std::span<const PbctopoPairShiftCandidateSet> pair_candidates,
    std::span<const PbctopoRequiredContact> required_contacts,
    std::span<const std::size_t> atom_component_index,
    bool uniqueness_search_exhaustive = false,
    std::size_t equivalent_best_assignments_within_domain = 0,
    bool feasible_assignment_set_enumerated = false,
    std::size_t feasible_assignments_within_domain = 0);

} // namespace titan_pbctopo

#endif

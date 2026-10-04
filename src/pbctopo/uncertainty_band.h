#pragma once

#include <cstddef>

namespace titan_pbctopo {

inline constexpr bool pbctopo_should_certify_complete_uncertainty_band(
    bool exact_feasible_band_retained, bool feasible_set_enumerated) {
  return exact_feasible_band_retained && feasible_set_enumerated;
}

inline constexpr bool pbctopo_complete_uncertainty_band_exported(
    bool evidence_audit_complete, std::size_t rejected_candidate_count,
    std::size_t certified_candidate_count,
    std::size_t feasible_assignment_count) {
  return evidence_audit_complete && rejected_candidate_count == 0 &&
         certified_candidate_count == feasible_assignment_count;
}

} // namespace titan_pbctopo

#ifndef TITAN_PBCTOPO_ASSIGNMENT_SET_REPORT_H
#define TITAN_PBCTOPO_ASSIGNMENT_SET_REPORT_H

#include "report.h"
#include "types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>

namespace titan_pbctopo {

using PbctopoCutCandidateBetterFn = bool (*)(const PbctopoCutCandidate &,
                                             const PbctopoCutCandidate &);

void write_pbctopo_assignment_set_header(std::ostream &out);

// Audit-only records, never members of retained_certified_band.  Invalid
// assignments and rejected evaluations of a certified assignment are omitted.
void write_pbctopo_rejected_assignment_header(std::ostream &out);
void write_pbctopo_rejected_assignment_rows(
    std::ostream &out, std::size_t frame, double time_ps,
    std::span<const PbctopoCutCandidate> evaluated_candidates,
    const std::array<double, 6> &normalized_gram,
    std::uint64_t topology_epoch_id = 0);

void write_pbctopo_assignment_set_rows(
    std::ostream &out, std::size_t frame, double time_ps,
    std::span<const PbctopoCutCandidate> certified_candidates,
    const PbctopoCutCandidate *framewise_selected,
    const PbctopoIdentifiability &identifiability,
    const std::array<double, 6> &normalized_gram,
    PbctopoCutCandidateBetterFn better_candidate,
    std::uint64_t topology_epoch_id = 0,
    std::size_t soft_score_valid_assignments_within_domain = 0,
    std::size_t spatial_lift_ambiguities = 0,
    std::size_t spatial_lift_cycle_residuals = 0,
    PbctopoFrameStatus frame_status = PbctopoFrameStatus::WeakObservation);

} // namespace titan_pbctopo

#endif

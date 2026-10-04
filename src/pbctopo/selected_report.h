#ifndef TITAN_PBCTOPO_SELECTED_REPORT_H
#define TITAN_PBCTOPO_SELECTED_REPORT_H

#include "report.h"
#include "types.h"

#include <span>

namespace titan_pbctopo {

bool pbctopo_candidate_anchor_ambiguous(
    const PbctopoCutCandidate &candidate,
    std::span<const PbctopoCutCandidate> certified_pool);

bool populate_pbctopo_selected_candidate_report(
    PbctopoFrameReport &report, const PbctopoCutCandidate &candidate,
    std::span<const PbctopoAtom> atoms, double fallback_clash_cutoff,
    double continuity_cutoff,
    std::span<const PbctopoCutCandidate> certified_pool = {});

} // namespace titan_pbctopo

#endif

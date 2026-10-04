#ifndef TITAN_PBCTOPO_EVIDENCE_RELATION_REPORT_H
#define TITAN_PBCTOPO_EVIDENCE_RELATION_REPORT_H

#include "evidence_audit.h"
#include "report.h"
#include "types.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace titan_pbctopo {

inline constexpr std::uint32_t PBCTOPO_EVIDENCE_RELATION_SCHEMA_VERSION = 1;
inline constexpr std::string_view PBCTOPO_EVIDENCE_RELATION_SOURCE_KIND =
    "selected_compatibility";
inline constexpr std::string_view PBCTOPO_EVIDENCE_RELATION_SUPPORT_UNIT =
    "decision_score";
inline constexpr std::string_view
    PBCTOPO_EVIDENCE_RELATION_SUPPORT_SECONDARY_UNIT = "lattice_linf_steps";
inline constexpr std::string_view PBCTOPO_EVIDENCE_ASSIGNMENT_HASH_DOMAIN =
    "titan-pbctopo-evidence-assignment-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_DOMAIN_HASH_DOMAIN =
    "titan-pbctopo-evidence-candidate-domain-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_GROUP_HASH_DOMAIN =
    "titan-pbctopo-evidence-group-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_RELATION_SET_HASH_DOMAIN =
    "titan-pbctopo-evidence-relation-set-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_RELATION_ID_HASH_DOMAIN =
    "titan-pbctopo-evidence-relation-id-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_SUMMARY_ROW_HASH_DOMAIN =
    "titan-pbctopo-evidence-summary-row-v1";
inline constexpr std::string_view PBCTOPO_EVIDENCE_RELATION_ROW_HASH_DOMAIN =
    "titan-pbctopo-evidence-relation-row-v1";

struct PbctopoCandidateEvidenceAuditSnapshot {
  PbctopoImageAssignment assignment;
  PbctopoSelectedEvidenceAudit audit;
  std::string candidate_relation_set_sha256;
};

struct PbctopoEvidenceRelationWriteStats {
  std::size_t candidate_groups = 0;
  std::size_t relation_rows = 0;
  std::size_t duplicate_candidates_skipped = 0;
};

[[nodiscard]] std::expected<bool, std::string>
upsert_pbctopo_candidate_evidence_audit_snapshot(
    std::vector<PbctopoCandidateEvidenceAuditSnapshot> &snapshots,
    const PbctopoImageAssignment &assignment,
    const PbctopoSelectedEvidenceAudit &audit);

void write_pbctopo_candidate_evidence_summary_header(std::ostream &out);
void write_pbctopo_candidate_evidence_relation_header(std::ostream &out);

[[nodiscard]] std::expected<PbctopoEvidenceRelationWriteStats, std::string>
write_pbctopo_candidate_evidence_frame(
    std::ostream &summary_out, std::ostream &relation_out,
    const PbctopoFrameReport &frame_report,
    std::span<const PbctopoCutCandidate> retained_candidates,
    const PbctopoCutCandidate *framewise_selected,
    bool retained_candidates_are_temporal,
    std::span<const PbctopoCandidateEvidenceAuditSnapshot> snapshots);

} // namespace titan_pbctopo

#endif

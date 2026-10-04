#include "evidence_relation_report.h"

#include "evidence_connectivity_margin.h"
#include "evidence_support_policy.h"
#include "observation_schema.h"

#include "../titan_sha256.h"
#include "../titan_version_info.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace titan_pbctopo {
namespace {

constexpr std::string_view kSummaryHeader =
    "schema_version,producer_version,git_commit,git_dirty,"
    "support_policy_version,support_policy_id,relation_source_kind,frame,"
    "trajectory_frame_index,time_ps,topology_epoch_id,"
    "component_layout_signature,box_hash,atom_selection_hash,"
    "canonical_atom_universe_hash,wrapped_coordinate_hash,"
    "frame_binding_valid,candidate_index,candidate_rank,retained_candidate,"
    "framewise_selected,temporal_candidate,assignment_signature,"
    "assignment_layout_signature,assignment_canonical,"
    "assignment_identity_sha256,candidate_domain_canonical,"
    "candidate_domain_identity_sha256,"
    "group_identity_sha256,domain_completeness,"
    "component_search_shell_radius,search_domain_audit_known,"
    "search_domain_evidence_complete,"
    "selected_assignment_touches_shell_boundary,hard_feasible,"
    "evidence_state,evidence_consistent,evidence_component_count,"
    "relation_count,active_relation_count,explicit_edge_count,"
    "compatible_edge_count,connected_component_count,"
    "evidence_graph_connected,relative_lattice_dof,margin_defined,"
    "maximum_bottleneck_support,signed_margin,decision_threshold,"
    "candidate_relation_set_sha256,summary_payload_sha256";

constexpr std::string_view kRelationHeader =
    "schema_version,producer_version,git_commit,git_dirty,"
    "support_policy_version,support_policy_id,relation_source_kind,frame,"
    "trajectory_frame_index,time_ps,topology_epoch_id,"
    "component_layout_signature,box_hash,atom_selection_hash,"
    "canonical_atom_universe_hash,wrapped_coordinate_hash,"
    "frame_binding_valid,candidate_index,candidate_rank,retained_candidate,"
    "framewise_selected,temporal_candidate,assignment_signature,"
    "assignment_layout_signature,assignment_canonical,"
    "assignment_identity_sha256,candidate_domain_canonical,"
    "candidate_domain_identity_sha256,"
    "group_identity_sha256,relation_ordinal,component_a_index,"
    "component_b_index,component_a_id,component_b_id,relation_evaluated,"
    "explicit_relation,observed_hypothesis_relation,compatibility_match,"
    "observed_hypothesis_count,compatible_hypothesis_count,"
    "compatible_contact_count,alternative_hypothesis_contact_count,"
    "compatible_contact_loss_count,support_primary,support_unit,"
    "support_secondary,support_secondary_unit,support_secondary_defined,"
    "decision_threshold,active_at_threshold,support_reason,"
    "candidate_relation_count,candidate_relation_set_sha256,"
    "relation_identity_sha256,row_payload_sha256";

std::string bool_string(bool value) { return value ? "1" : "0"; }

template <class Integer>
std::string integer_string(Integer value) {
  return std::to_string(value);
}

std::string format_double(double value) {
  if (!std::isfinite(value))
    return "na";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << (value == 0.0 ? 0.0 : value);
  return out.str();
}

std::string format_hash128(PbctopoHash128 hash) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::hex << std::setfill('0') << std::setw(16) << hash.hi
      << std::setw(16) << hash.lo;
  return out.str();
}

std::string format_sha256(std::span<const std::byte, 32> digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (const std::byte value : digest) {
    const auto byte = std::to_integer<unsigned char>(value);
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

void hash_field(titan_hash::Sha256Builder &hash, std::string_view field) {
  hash.word(static_cast<std::uint64_t>(field.size()));
  hash.update(field);
}

std::string hash_fields(std::string_view domain,
                        std::span<const std::string> fields) {
  titan_hash::Sha256Builder hash;
  hash.update(domain);
  hash.word(static_cast<std::uint64_t>(fields.size()));
  for (const auto &field : fields)
    hash_field(hash, field);
  return format_sha256(hash.finish256());
}

bool csv_safe(std::string_view value) {
  return value.find_first_of(",\r\n\"") == std::string_view::npos;
}

std::expected<void, std::string>
write_csv_row(std::ostream &out, std::span<const std::string> fields) {
  for (const auto &field : fields) {
    if (!csv_safe(field))
      return std::unexpected("evidence relation CSV field requires quoting");
  }
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0)
      out << ',';
    out << fields[index];
  }
  out << '\n';
  if (!out.good())
    return std::unexpected("evidence relation CSV write failed");
  return {};
}

std::string evidence_state_name(PbctopoEvidenceState state) {
  switch (state) {
  case PbctopoEvidenceState::NotEvaluated:
    return "not_evaluated";
  case PbctopoEvidenceState::NotApplicable:
    return "not_applicable";
  case PbctopoEvidenceState::SupportedConnected:
    return "supported_connected";
  case PbctopoEvidenceState::WeakDisconnected:
    return "weak_disconnected";
  case PbctopoEvidenceState::Contradicted:
    return "contradicted";
  case PbctopoEvidenceState::WeakAmbiguous:
    return "weak_ambiguous";
  }
  return "unknown";
}

std::expected<std::string, std::string>
canonical_assignment(const PbctopoImageAssignment &assignment) {
  if (!assignment.valid || assignment.component_ids.empty() ||
      assignment.component_ids.size() != assignment.component_offsets.size()) {
    return std::unexpected("evidence relation assignment is invalid");
  }
  struct Entry {
    std::uint64_t id = 0;
    Int3 offset{};
  };
  std::vector<Entry> entries;
  entries.reserve(assignment.component_ids.size());
  for (std::size_t index = 0; index < assignment.component_ids.size(); ++index)
    entries.push_back(
        {assignment.component_ids[index], assignment.component_offsets[index]});
  std::sort(entries.begin(), entries.end(),
            [](const Entry &lhs, const Entry &rhs) { return lhs.id < rhs.id; });
  if (std::adjacent_find(entries.begin(), entries.end(),
                         [](const Entry &lhs, const Entry &rhs) {
                           return lhs.id == rhs.id;
                         }) != entries.end()) {
    return std::unexpected(
        "evidence relation assignment contains duplicate component IDs");
  }
  std::ostringstream out;
  out.imbue(std::locale::classic());
  for (std::size_t index = 0; index < entries.size(); ++index) {
    if (index != 0)
      out << '|';
    out << entries[index].id << ':' << entries[index].offset.x << ':'
        << entries[index].offset.y << ':' << entries[index].offset.z;
  }
  return out.str();
}

std::vector<std::string>
candidate_domain_fields(const PbctopoCutCandidate &candidate) {
  return {
      integer_string(candidate.assignment.layout_signature),
      bool_string(candidate.search_equivalence_known),
      bool_string(candidate.equivalent_to_search_best),
      integer_string(candidate.component_search_shell_radius),
      bool_string(candidate.search_domain_audit_known),
      bool_string(candidate.search_domain_evidence_complete),
      bool_string(candidate.selected_assignment_touches_shell_boundary),
      bool_string(candidate.domain_expansion_attempted),
      bool_string(candidate.domain_expansion_exhausted),
      bool_string(candidate.bounded_domain_only),
      pbctopo_domain_completeness_name(candidate.domain_completeness),
  };
}

std::string join_fields(std::span<const std::string> fields,
                        char separator) {
  std::string result;
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0)
      result.push_back(separator);
    result.append(fields[index]);
  }
  return result;
}

std::expected<std::vector<std::string>, std::string>
relation_core_fields(const PbctopoSelectedEvidenceRelationAudit &relation,
                     std::size_t component_count) {
  if (!relation.evaluated || relation.component_a >= relation.component_b ||
      relation.component_b >= component_count ||
      relation.component_id_a >= relation.component_id_b ||
      !std::isfinite(relation.support.support_primary) ||
      !std::isfinite(relation.support.decision_threshold) ||
      relation.support.decision_threshold !=
          PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_THRESHOLD ||
      relation.support.active_at_threshold !=
          (relation.support.support_primary >=
           relation.support.decision_threshold)) {
    return std::unexpected("evidence relation record violates G3 invariants");
  }
  return std::vector<std::string>{
      integer_string(relation.component_a),
      integer_string(relation.component_b),
      integer_string(relation.component_id_a),
      integer_string(relation.component_id_b),
      bool_string(relation.evaluated),
      bool_string(relation.support.explicit_relation),
      bool_string(relation.observed_hypothesis_relation),
      bool_string(relation.support.compatibility_match),
      integer_string(relation.support.observed_hypothesis_count),
      integer_string(relation.support.compatible_hypothesis_count),
      integer_string(relation.compatible_contact_count),
      integer_string(relation.alternative_hypothesis_contact_count),
      integer_string(relation.compatible_contact_loss_count),
      format_double(relation.support.support_primary),
      relation.support.closest_shift_linf_distance_defined
          ? integer_string(relation.support.closest_shift_linf_distance)
          : "na",
      bool_string(relation.support.closest_shift_linf_distance_defined),
      format_double(relation.support.decision_threshold),
      bool_string(relation.support.active_at_threshold),
      std::string(pbctopo_evidence_support_reason_name(
          relation.support.reason)),
  };
}

std::expected<std::string, std::string>
relation_set_hash(const PbctopoSelectedEvidenceAudit &audit) {
  if (!audit.evaluated || audit.evidence_component_count == 0)
    return std::unexpected("evidence relation audit is not exportable");

  titan_hash::Sha256Builder hash;
  hash.update(PBCTOPO_EVIDENCE_RELATION_SET_HASH_DOMAIN);
  hash.word(static_cast<std::uint64_t>(audit.relation_audits.size()));
  std::pair<std::size_t, std::size_t> previous{};
  bool has_previous = false;
  for (const auto &relation : audit.relation_audits) {
    const auto fields =
        relation_core_fields(relation, audit.evidence_component_count);
    if (!fields)
      return std::unexpected(fields.error());
    const std::pair current{relation.component_a, relation.component_b};
    if (has_previous && !(previous < current))
      return std::unexpected(
          "evidence relation records are not strictly sorted");
    hash.word(static_cast<std::uint64_t>(fields->size()));
    for (const auto &field : *fields)
      hash_field(hash, field);
    previous = current;
    has_previous = true;
  }
  return format_sha256(hash.finish256());
}

bool audit_aggregates_equal(const PbctopoSelectedEvidenceAudit &lhs,
                            const PbctopoSelectedEvidenceAudit &rhs) {
  return lhs.state == rhs.state && lhs.evaluated == rhs.evaluated &&
         lhs.consistent == rhs.consistent &&
         lhs.evidence_graph_connected == rhs.evidence_graph_connected &&
         lhs.evidence_component_count == rhs.evidence_component_count &&
         lhs.evidence_edge_count == rhs.evidence_edge_count &&
         lhs.explicit_contact_evidence_edges ==
             rhs.explicit_contact_evidence_edges &&
         lhs.scored_contact_evidence_edges ==
             rhs.scored_contact_evidence_edges &&
         lhs.evidence_connected_component_count ==
             rhs.evidence_connected_component_count &&
         lhs.relative_lattice_degrees_of_freedom ==
             rhs.relative_lattice_degrees_of_freedom &&
         lhs.contacts_checked == rhs.contacts_checked &&
         lhs.contacts_lost == rhs.contacts_lost;
}

std::expected<const PbctopoCandidateEvidenceAuditSnapshot *, std::string>
find_snapshot(
    const PbctopoImageAssignment &assignment,
    std::span<const PbctopoCandidateEvidenceAuditSnapshot> snapshots) {
  const auto assignment_text = canonical_assignment(assignment);
  if (!assignment_text)
    return std::unexpected(assignment_text.error());
  const PbctopoCandidateEvidenceAuditSnapshot *match = nullptr;
  for (const auto &snapshot : snapshots) {
    const auto snapshot_text = canonical_assignment(snapshot.assignment);
    if (!snapshot_text)
      return std::unexpected(snapshot_text.error());
    const bool same_signature =
        snapshot.assignment.signature == assignment.signature &&
        snapshot.assignment.layout_signature == assignment.layout_signature;
    const bool same_canonical = *snapshot_text == *assignment_text;
    if (same_signature && !same_canonical)
      return std::unexpected("evidence assignment signature collision");
    if (same_canonical && !same_signature)
      return std::unexpected(
          "canonical evidence assignment has inconsistent signatures");
    if (!same_canonical)
      continue;
    if (match != nullptr)
      return std::unexpected("duplicate evidence audit snapshots");
    match = &snapshot;
  }
  if (match == nullptr)
    return std::unexpected("candidate has no cached evidence relation audit");
  return match;
}

struct CandidateExportEntry {
  const PbctopoCutCandidate *candidate = nullptr;
  std::size_t candidate_index = 0;
  std::size_t candidate_rank = 0;
  bool retained = false;
  bool framewise_selected = false;
  bool temporal_candidate = false;
  std::string assignment_canonical;
  std::vector<std::string> domain_fields;
};

std::expected<std::vector<CandidateExportEntry>, std::string>
build_candidate_entries(
    std::span<const PbctopoCutCandidate> retained_candidates,
    const PbctopoCutCandidate *framewise_selected,
    bool retained_candidates_are_temporal,
    PbctopoEvidenceRelationWriteStats &stats) {
  std::vector<CandidateExportEntry> entries;
  entries.reserve(retained_candidates.size() +
                   (framewise_selected == nullptr ? 0 : 1));
  if (framewise_selected != nullptr &&
      (!framewise_selected->valid || !framewise_selected->assignment.valid)) {
    return std::unexpected("invalid framewise selected evidence candidate");
  }

  auto add_entry = [&](const PbctopoCutCandidate &candidate,
                       std::size_t candidate_index,
                       std::size_t candidate_rank, bool retained,
                       bool selected,
                       bool temporal) -> std::expected<bool, std::string> {
    const auto assignment_text = canonical_assignment(candidate.assignment);
    if (!assignment_text)
      return std::unexpected(assignment_text.error());
    const auto domain = candidate_domain_fields(candidate);
    for (auto &entry : entries) {
      const bool same_signature =
          entry.candidate->assignment.signature ==
              candidate.assignment.signature &&
          entry.candidate->assignment.layout_signature ==
              candidate.assignment.layout_signature;
      const bool same_canonical =
          entry.assignment_canonical == *assignment_text;
      if (same_signature && !same_canonical)
        return std::unexpected("candidate assignment signature collision");
      if (!same_canonical)
        continue;
      if (!same_signature)
        return std::unexpected(
            "canonical candidate assignment has inconsistent signatures");
      if (entry.domain_fields != domain)
        return std::unexpected(
            "duplicate candidate changed search-domain provenance");
      entry.framewise_selected = entry.framewise_selected || selected;
      return false;
    }
    entries.push_back({&candidate, candidate_index, candidate_rank, retained,
                       selected, temporal, *assignment_text, domain});
    return true;
  };

  for (std::size_t index = 0; index < retained_candidates.size(); ++index) {
    const auto &candidate = retained_candidates[index];
    if (!candidate.valid || !candidate.assignment.valid)
      return std::unexpected("invalid retained evidence candidate");
    const auto added = add_entry(candidate, index + 1, index + 1, true, false,
                                 retained_candidates_are_temporal);
    if (!added)
      return std::unexpected(added.error());
    if (!*added)
      ++stats.duplicate_candidates_skipped;
  }

  if (framewise_selected != nullptr) {
    const auto added = add_entry(*framewise_selected, 0, 0, false, true, false);
    if (!added)
      return std::unexpected(added.error());
  }

  std::sort(entries.begin(), entries.end(),
            [](const CandidateExportEntry &lhs,
               const CandidateExportEntry &rhs) {
              return std::tie(lhs.assignment_canonical,
                              lhs.candidate->assignment.signature,
                              lhs.candidate_index) <
                     std::tie(rhs.assignment_canonical,
                              rhs.candidate->assignment.signature,
                              rhs.candidate_index);
            });
  return entries;
}

} // namespace

std::expected<bool, std::string>
upsert_pbctopo_candidate_evidence_audit_snapshot(
    std::vector<PbctopoCandidateEvidenceAuditSnapshot> &snapshots,
    const PbctopoImageAssignment &assignment,
    const PbctopoSelectedEvidenceAudit &audit) {
  if (!assignment.valid || !audit.evaluated ||
      assignment.component_ids.size() != audit.evidence_component_count)
    return std::unexpected("candidate evidence snapshot binding is invalid");
  const auto set_hash = relation_set_hash(audit);
  if (!set_hash)
    return std::unexpected(set_hash.error());
  const auto assignment_text = canonical_assignment(assignment);
  if (!assignment_text)
    return std::unexpected(assignment_text.error());

  for (const auto &snapshot : snapshots) {
    const auto snapshot_text = canonical_assignment(snapshot.assignment);
    if (!snapshot_text)
      return std::unexpected(snapshot_text.error());
    const bool same_signature =
        snapshot.assignment.signature == assignment.signature &&
        snapshot.assignment.layout_signature == assignment.layout_signature;
    const bool same_canonical = *snapshot_text == *assignment_text;
    if (same_signature && !same_canonical)
      return std::unexpected("candidate evidence snapshot signature collision");
    if (same_canonical && !same_signature)
      return std::unexpected(
          "canonical evidence snapshot has inconsistent signatures");
    if (!same_canonical)
      continue;
    if (snapshot.candidate_relation_set_sha256 != *set_hash ||
        !audit_aggregates_equal(snapshot.audit, audit)) {
      return std::unexpected(
          "repeated candidate evidence audit changed for one assignment");
    }
    return false;
  }
  snapshots.push_back({assignment, audit, *set_hash});
  return true;
}

void write_pbctopo_candidate_evidence_summary_header(std::ostream &out) {
  out << kSummaryHeader << '\n';
}

void write_pbctopo_candidate_evidence_relation_header(std::ostream &out) {
  out << kRelationHeader << '\n';
}

std::expected<PbctopoEvidenceRelationWriteStats, std::string>
write_pbctopo_candidate_evidence_frame(
    std::ostream &summary_out, std::ostream &relation_out,
    const PbctopoFrameReport &frame_report,
    std::span<const PbctopoCutCandidate> retained_candidates,
    const PbctopoCutCandidate *framewise_selected,
    bool retained_candidates_are_temporal,
    std::span<const PbctopoCandidateEvidenceAuditSnapshot> snapshots) {
  PbctopoEvidenceRelationWriteStats stats;
  const auto entries = build_candidate_entries(
      retained_candidates, framewise_selected,
      retained_candidates_are_temporal, stats);
  if (!entries)
    return std::unexpected(entries.error());
  if (entries->empty())
    return stats;
  if (!summary_out.good() || !relation_out.good())
    return std::unexpected("evidence relation report stream is unavailable");
  if (!frame_report.frame_binding_valid || frame_report.box_hash.empty() ||
      frame_report.atom_selection_hash.empty() ||
      frame_report.canonical_atom_universe_hash.empty() ||
      frame_report.wrapped_coordinate_hash.empty()) {
    return std::unexpected("evidence relation frame binding is invalid");
  }

  std::ostringstream summary_buffer;
  std::ostringstream relation_buffer;
  summary_buffer.imbue(std::locale::classic());
  relation_buffer.imbue(std::locale::classic());

  for (const auto &entry : *entries) {
    const auto &candidate = *entry.candidate;
    const auto snapshot = find_snapshot(candidate.assignment, snapshots);
    if (!snapshot)
      return std::unexpected(snapshot.error());
    const auto &audit = (*snapshot)->audit;
    if (audit.evidence_component_count !=
        candidate.assignment.component_ids.size()) {
      return std::unexpected("candidate relation component count mismatch");
    }
    if (frame_report.component_layout_signature != 0 &&
        frame_report.component_layout_signature !=
            candidate.assignment.layout_signature) {
      return std::unexpected("candidate relation layout signature mismatch");
    }

    const std::vector<std::string> assignment_fields{
        entry.assignment_canonical};
    const std::string assignment_identity = hash_fields(
        PBCTOPO_EVIDENCE_ASSIGNMENT_HASH_DOMAIN, assignment_fields);
    const std::string domain_canonical = join_fields(entry.domain_fields, '|');
    const std::string domain_identity =
        hash_fields(PBCTOPO_EVIDENCE_DOMAIN_HASH_DOMAIN, entry.domain_fields);

    const std::vector<std::string> group_fields{
        integer_string(frame_report.frame),
        integer_string(frame_report.trajectory_frame_index),
        format_double(frame_report.time_ps),
        integer_string(frame_report.topology_epoch_id),
        integer_string(candidate.assignment.layout_signature),
        format_hash128(frame_report.box_hash),
        format_hash128(frame_report.atom_selection_hash),
        format_hash128(frame_report.canonical_atom_universe_hash),
        format_hash128(frame_report.wrapped_coordinate_hash),
        assignment_identity,
        domain_identity,
        std::string(PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_POLICY),
    };
    const std::string group_identity =
        hash_fields(PBCTOPO_EVIDENCE_GROUP_HASH_DOMAIN, group_fields);

    std::vector<PbctopoEvidenceSupportEdge> margin_edges;
    margin_edges.reserve(audit.relation_audits.size());
    for (const auto &relation : audit.relation_audits) {
      margin_edges.push_back({relation.component_a, relation.component_b,
                              relation.support.support_primary});
    }
    PbctopoEvidenceConnectivityMargin margin;
    try {
      margin = compute_pbctopo_evidence_connectivity_margin(
          audit.evidence_component_count, margin_edges,
          PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_THRESHOLD);
    } catch (const std::exception &error) {
      return std::unexpected(std::string("candidate margin replay failed: ") +
                             error.what());
    }
    if (margin.canonical_edge_count != audit.relation_audits.size() ||
        margin.thresholded_edge_count != audit.evidence_edge_count ||
        margin.thresholded_component_count !=
            audit.evidence_connected_component_count ||
        margin.thresholded_connected != audit.evidence_graph_connected) {
      return std::unexpected(
          "candidate margin replay disagrees with evidence audit");
    }

    const std::string relation_set_identity =
        (*snapshot)->candidate_relation_set_sha256;
    std::vector<std::string> common_fields{
        integer_string(PBCTOPO_EVIDENCE_RELATION_SCHEMA_VERSION),
        vibe_observation_producer_version(),
        titan_build::git_commit(),
        bool_string(titan_build::git_dirty()),
        integer_string(PBCTOPO_EVIDENCE_SUPPORT_POLICY_VERSION),
        std::string(PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_POLICY),
        std::string(PBCTOPO_EVIDENCE_RELATION_SOURCE_KIND),
        integer_string(frame_report.frame),
        integer_string(frame_report.trajectory_frame_index),
        format_double(frame_report.time_ps),
        integer_string(frame_report.topology_epoch_id),
        integer_string(candidate.assignment.layout_signature),
        format_hash128(frame_report.box_hash),
        format_hash128(frame_report.atom_selection_hash),
        format_hash128(frame_report.canonical_atom_universe_hash),
        format_hash128(frame_report.wrapped_coordinate_hash),
        bool_string(frame_report.frame_binding_valid),
        integer_string(entry.candidate_index),
        integer_string(entry.candidate_rank),
        bool_string(entry.retained),
        bool_string(entry.framewise_selected),
        bool_string(entry.temporal_candidate),
        integer_string(candidate.assignment.signature),
        integer_string(candidate.assignment.layout_signature),
        entry.assignment_canonical,
        assignment_identity,
        domain_canonical,
        domain_identity,
        group_identity,
    };

    for (std::size_t relation_index = 0;
         relation_index < audit.relation_audits.size(); ++relation_index) {
      const auto &relation = audit.relation_audits[relation_index];
      const auto core =
          relation_core_fields(relation, audit.evidence_component_count);
      if (!core)
        return std::unexpected(core.error());
      const std::vector<std::string> relation_identity_fields{
          group_identity, integer_string(relation.component_id_a),
          integer_string(relation.component_id_b),
          std::string(PBCTOPO_EVIDENCE_RELATION_SOURCE_KIND),
      };
      const std::string relation_identity = hash_fields(
          PBCTOPO_EVIDENCE_RELATION_ID_HASH_DOMAIN,
          relation_identity_fields);

      std::vector<std::string> row = common_fields;
      row.insert(row.end(), {
          integer_string(relation_index + 1),
          integer_string(relation.component_a),
          integer_string(relation.component_b),
          integer_string(relation.component_id_a),
          integer_string(relation.component_id_b),
          bool_string(relation.evaluated),
          bool_string(relation.support.explicit_relation),
          bool_string(relation.observed_hypothesis_relation),
          bool_string(relation.support.compatibility_match),
          integer_string(relation.support.observed_hypothesis_count),
          integer_string(relation.support.compatible_hypothesis_count),
          integer_string(relation.compatible_contact_count),
          integer_string(relation.alternative_hypothesis_contact_count),
          integer_string(relation.compatible_contact_loss_count),
          format_double(relation.support.support_primary),
          std::string(PBCTOPO_EVIDENCE_RELATION_SUPPORT_UNIT),
          relation.support.closest_shift_linf_distance_defined
              ? integer_string(relation.support.closest_shift_linf_distance)
              : "na",
          std::string(PBCTOPO_EVIDENCE_RELATION_SUPPORT_SECONDARY_UNIT),
          bool_string(relation.support.closest_shift_linf_distance_defined),
          format_double(relation.support.decision_threshold),
          bool_string(relation.support.active_at_threshold),
          std::string(pbctopo_evidence_support_reason_name(
              relation.support.reason)),
          integer_string(audit.relation_audits.size()),
          relation_set_identity,
          relation_identity,
      });
      row.push_back(hash_fields(PBCTOPO_EVIDENCE_RELATION_ROW_HASH_DOMAIN,
                                row));
      const auto written = write_csv_row(relation_buffer, row);
      if (!written)
        return std::unexpected(written.error());
      ++stats.relation_rows;
    }

    std::vector<std::string> summary = common_fields;
    summary.insert(summary.end(), {
        pbctopo_domain_completeness_name(candidate.domain_completeness),
        integer_string(candidate.component_search_shell_radius),
        bool_string(candidate.search_domain_audit_known),
        bool_string(candidate.search_domain_evidence_complete),
        bool_string(candidate.selected_assignment_touches_shell_boundary),
        bool_string(candidate.hard_feasibility.certified()),
        evidence_state_name(audit.state),
        bool_string(audit.consistent),
        integer_string(audit.evidence_component_count),
        integer_string(audit.relation_audits.size()),
        integer_string(audit.evidence_edge_count),
        integer_string(audit.explicit_contact_evidence_edges),
        integer_string(audit.scored_contact_evidence_edges),
        integer_string(audit.evidence_connected_component_count),
        bool_string(audit.evidence_graph_connected),
        integer_string(audit.relative_lattice_degrees_of_freedom),
        bool_string(margin.margin_defined),
        margin.margin_defined ? format_double(margin.maximum_bottleneck_support)
                              : "na",
        margin.margin_defined ? format_double(margin.signed_margin) : "na",
        format_double(PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_THRESHOLD),
        relation_set_identity,
    });
    summary.push_back(hash_fields(PBCTOPO_EVIDENCE_SUMMARY_ROW_HASH_DOMAIN,
                                  summary));
    const auto summary_written = write_csv_row(summary_buffer, summary);
    if (!summary_written)
      return std::unexpected(summary_written.error());
    ++stats.candidate_groups;
  }

  relation_out << relation_buffer.str();
  summary_out << summary_buffer.str();
  if (!relation_out.good() || !summary_out.good())
    return std::unexpected("evidence relation frame flush failed");
  return stats;
}

} // namespace titan_pbctopo

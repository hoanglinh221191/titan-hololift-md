#include "assignment_set_report.h"

#include "image_assignment.h"
#include "observation_binding.h"
#include "observation_schema.h"
#include "report.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace titan_pbctopo {
namespace {

std::string format_double(double value) {
  if (!std::isfinite(value))
    return "na";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10)
      << (value == 0.0 ? 0.0 : value);
  return out.str();
}

std::string observation_class(const PbctopoCutCandidate &candidate,
                               const PbctopoIdentifiability &identifiability,
                               bool selected, bool equivalent_set_complete,
                               bool credible_alternative_set_complete,
                               std::size_t spatial_lift_ambiguities,
                               std::size_t spatial_lift_cycle_residuals,
                               PbctopoFrameStatus frame_status) {
  if (!candidate.hard_feasibility.certified())
    return "not_hard_feasible";
  if (!identifiability.identified)
    return "weak_observation";
  if ((frame_status == PbctopoFrameStatus::Certified ||
       frame_status == PbctopoFrameStatus::Rescued) &&
      selected && candidate.search_equivalence_known &&
      candidate.equivalent_to_search_best && equivalent_set_complete &&
      credible_alternative_set_complete &&
      identifiability.objective_uniqueness_known &&
      identifiability.objective_unique_within_search_domain &&
      identifiability.evidence_uniqueness_known &&
      identifiability.evidence_unique_within_search_domain &&
      candidate.search_domain_audit_known &&
      candidate.search_domain_evidence_complete &&
      candidate.component_search_shell_radius > 0 &&
      !candidate.selected_assignment_touches_shell_boundary &&
      spatial_lift_ambiguities == 0 &&
      spatial_lift_cycle_residuals == 0) {
    if (candidate.domain_completeness ==
        PbctopoDomainCompleteness::BoundedExhaustive)
      return "strong_bounded_anchor";
  }
  return "ambiguous_anchor";
}

std::string serialize_component_assignment(
    const PbctopoImageAssignment &assignment) {
  if (!assignment.valid ||
      assignment.component_ids.size() != assignment.component_offsets.size()) {
    return "na";
  }
  struct Entry {
    std::uint64_t component_id = 0;
    Int3 offset{};
  };
  std::vector<Entry> entries;
  entries.reserve(assignment.component_offsets.size());
  for (std::size_t idx = 0; idx < assignment.component_offsets.size(); ++idx) {
    entries.push_back(
        {assignment.component_ids[idx], assignment.component_offsets[idx]});
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry &lhs, const Entry &rhs) {
              if (lhs.component_id != rhs.component_id)
                return lhs.component_id < rhs.component_id;
              if (lhs.offset.x != rhs.offset.x)
                return lhs.offset.x < rhs.offset.x;
              if (lhs.offset.y != rhs.offset.y)
                return lhs.offset.y < rhs.offset.y;
              return lhs.offset.z < rhs.offset.z;
            });
  std::ostringstream out;
  out.imbue(std::locale::classic());
  for (std::size_t idx = 0; idx < entries.size(); ++idx) {
    if (idx != 0)
      out << '|';
    out << entries[idx].component_id << ':' << entries[idx].offset.x << ':'
        << entries[idx].offset.y << ':' << entries[idx].offset.z;
  }
  return out.str();
}

bool equivalent_under_order(const PbctopoCutCandidate &lhs,
                            const PbctopoCutCandidate &rhs,
                            PbctopoCutCandidateBetterFn better_candidate) {
  return better_candidate != nullptr && !better_candidate(lhs, rhs) &&
         !better_candidate(rhs, lhs);
}

std::string rejection_reasons(const PbctopoHardFeasibility &audit) {
  std::string result;
  const auto add = [&](bool failed, const char *reason) {
    if (!failed)
      return;
    if (!result.empty())
      result += ';';
    result += reason;
  };
  add(!audit.evaluated, "hard_certificate_not_evaluated");
  add(!audit.assignment_available, "assignment_unavailable");
  add(audit.topology_edges_checked < audit.topology_edges_available,
      "topology_edge_audit_incomplete");
  add(audit.metadata_edges_checked < audit.metadata_edges_available,
      "metadata_edge_audit_incomplete");
  add(audit.hard_edge_residuals != 0, "hard_edge_residual");
  add(audit.atom_image_residuals != 0, "atom_image_residual");
  add(audit.assignment_point_residuals != 0, "assignment_point_residual");
  add(audit.topology_cycle_residuals != 0, "topology_cycle_residual");
  add(audit.unsupported_winding_rank != 0, "unsupported_winding");
  add(!audit.steric_audit_available, "steric_audit_unavailable");
  add(!audit.steric_decision_complete, "steric_decision_incomplete");
  add(!audit.steric_pair_enumeration_complete,
      "steric_pair_enumeration_incomplete");
  add(!audit.steric_scan_complete, "steric_scan_incomplete");
  add(audit.hard_clashes != 0, "hard_steric_clash");
  add(audit.metadata_internal_clashes != 0, "metadata_internal_clash");
  add(audit.required_contacts_lost != 0, "required_contact_lost");
  return result;
}

} // namespace

void write_pbctopo_rejected_assignment_header(std::ostream &out) {
  out << "schema_version,producer_version,record_scope,record_source,"
         "provider_status,coordinate_unit,time_unit,topology_epoch_id,"
         "frame,time_ps,rejection_record_index,selected_framewise,hard_feasible,"
         "layout_signature,assignment_signature,component_count,"
         "global_gauge_x,global_gauge_y,global_gauge_z,component_assignment,"
         "normalized_gram_g00,normalized_gram_g01,normalized_gram_g02,"
         "normalized_gram_g11,normalized_gram_g12,normalized_gram_g22,"
         "rejection_reasons,hard_certificate_evaluated,assignment_available,"
         "topology_edges_available,topology_edges_checked,"
         "metadata_edges_available,metadata_edges_checked,hard_edge_residuals,"
         "atom_image_residuals,assignment_point_residuals,"
         "topology_cycle_residuals,unsupported_winding_rank,"
         "steric_audit_available,steric_decision_complete,"
         "steric_pair_enumeration_complete,steric_scan_complete,"
         "steric_termination_reason,steric_pairs_checked,hard_clashes,"
         "metadata_internal_clashes,required_contacts_checked,"
         "required_contacts_lost,min_steric_margin_A\n";
}

void write_pbctopo_rejected_assignment_rows(
    std::ostream &out, std::size_t frame, double time_ps,
    std::span<const PbctopoCutCandidate> evaluated_candidates,
    const std::array<double, 6> &normalized_gram,
    std::uint64_t topology_epoch_id) {
  std::vector<const PbctopoCutCandidate *> rejected;
  for (const auto &candidate : evaluated_candidates) {
    if (!candidate.valid || !candidate.assignment.valid ||
        candidate.hard_feasibility.certified())
      continue;
    const bool certified_elsewhere = std::any_of(
        evaluated_candidates.begin(), evaluated_candidates.end(),
        [&](const auto &other) {
          return other.valid && other.assignment.valid &&
                 other.hard_feasibility.certified() &&
                 same_pbctopo_image_assignment(candidate.assignment,
                                                other.assignment);
        });
    const bool already = std::any_of(
        rejected.begin(), rejected.end(), [&](const auto *other) {
          return same_pbctopo_image_assignment(candidate.assignment,
                                                other->assignment);
        });
    if (!certified_elsewhere && !already)
      rejected.push_back(&candidate);
  }
  for (std::size_t idx = 0; idx < rejected.size(); ++idx) {
    const auto &candidate = *rejected[idx];
    const auto &assignment = candidate.assignment;
    const auto &audit = candidate.hard_feasibility;
    out << "1," << vibe_observation_producer_version()
        << ",provider_rejected_audit_ledger,hard_certificate_evaluation,"
           "provider_rejected,"
        << VIBE_OBSERVATION_COORDINATE_UNIT << ','
        << VIBE_OBSERVATION_TIME_UNIT << ',' << topology_epoch_id << ','
        << frame << ',' << format_double(time_ps) << ',' << idx + 1
        << ",0,0," << assignment.layout_signature << ','
        << assignment.signature << ',' << assignment.component_offsets.size()
        << ',' << assignment.global_gauge.x << ',' << assignment.global_gauge.y
        << ',' << assignment.global_gauge.z << ','
        << serialize_component_assignment(assignment);
    for (double value : normalized_gram)
      out << ',' << format_double(value);
    out << ',' << rejection_reasons(audit) << ',' << (audit.evaluated ? 1 : 0)
        << ',' << (audit.assignment_available ? 1 : 0)
        << ',' << audit.topology_edges_available << ',' << audit.topology_edges_checked
        << ',' << audit.metadata_edges_available << ',' << audit.metadata_edges_checked
        << ',' << audit.hard_edge_residuals << ',' << audit.atom_image_residuals
        << ',' << audit.assignment_point_residuals << ',' << audit.topology_cycle_residuals
        << ',' << audit.unsupported_winding_rank
        << ',' << (audit.steric_audit_available ? 1 : 0)
        << ',' << (audit.steric_decision_complete ? 1 : 0)
        << ',' << (audit.steric_pair_enumeration_complete ? 1 : 0)
        << ',' << (audit.steric_scan_complete ? 1 : 0)
        << ',' << static_cast<unsigned>(audit.steric_termination_reason)
        << ',' << audit.steric_pairs_checked << ',' << audit.hard_clashes
        << ',' << audit.metadata_internal_clashes << ',' << audit.required_contacts_checked
        << ',' << audit.required_contacts_lost << ','
        << format_double(audit.min_steric_margin_A) << '\n';
  }
}

void write_pbctopo_assignment_set_header(std::ostream &out) {
  out << "schema_version,producer_version,coordinate_unit,time_unit,"
         "endianness,floating_format,identity_hash_algorithm,"
         "binding_hash_algorithm,payload_checksum_algorithm,binding_mode,"
         "negative_zero_policy,"
         "nonfinite_policy,observation_scope,topology_epoch_id,"
         "frame,time_ps,candidate_rank,selected_framewise,"
         "candidate_set_scope,observation_class,search_equivalence_known,"
         "equivalent_to_search_best,equivalent_under_output_order,"
         "equivalent_set_complete,retained_certified_assignment_count,"
         "search_equivalent_assignments_exported,"
         "equivalent_best_assignments_within_domain,identified,"
         "unique_within_search_domain,uniqueness_search_exhaustive,"
         "objective_uniqueness_known,"
         "objective_unique_within_search_domain,"
         "evidence_uniqueness_known,"
         "evidence_unique_within_search_domain,"
         "feasible_assignment_set_enumerated,"
         "feasible_assignment_set_semantics,"
         "feasible_assignments_within_domain,"
         "soft_score_valid_assignments_within_domain,"
         "search_domain_assignments_exported,"
         "credible_alternative_set_complete,"
         "component_search_shell_radius,search_domain_audit_known,"
         "search_domain_evidence_complete,"
         "selected_assignment_touches_shell_boundary,"
         "domain_expansion_attempted,domain_expansion_exhausted,"
         "bounded_domain_only,domain_completeness,"
         "hard_feasible,layout_signature,assignment_signature,"
         "evidence_relative_relation_signature,"
         "evidence_compatible_hypothesis_hash,"
         "evidence_compatible_pair_hash,"
         "evidence_compatible_hypothesis_count,"
         "evidence_supported_relation_count,"
         "evidence_compatible_contact_count,"
         "evidence_no_support_relation_count,"
         "component_count,global_gauge_x,global_gauge_y,global_gauge_z,"
         "component_assignment,normalized_gram_g00,normalized_gram_g01,"
         "normalized_gram_g02,normalized_gram_g11,normalized_gram_g12,"
         "normalized_gram_g22,broken_edges,continuity_max_d2,"
         "continuity_path,anchor_max_d2,anchor_mst2,min_pair_d2,"
         "shift_norm2\n";
}

void write_pbctopo_assignment_set_rows(
    std::ostream &out, std::size_t frame, double time_ps,
    std::span<const PbctopoCutCandidate> certified_candidates,
    const PbctopoCutCandidate *framewise_selected,
    const PbctopoIdentifiability &identifiability,
    const std::array<double, 6> &normalized_gram,
    PbctopoCutCandidateBetterFn better_candidate,
    std::uint64_t topology_epoch_id,
    std::size_t soft_score_valid_assignments_within_domain,
    std::size_t spatial_lift_ambiguities,
    std::size_t spatial_lift_cycle_residuals,
    PbctopoFrameStatus frame_status) {
  // Keep the retained_certified_band contract: rejected evaluations belong
  // only in the separate audit ledger and do not contribute to band counts.
  // Their presence still conservatively lowers domain evidence completeness.
  std::vector<const PbctopoCutCandidate *> ordered;
  std::vector<const PbctopoCutCandidate *> rejected;
  ordered.reserve(certified_candidates.size());
  for (const auto &candidate : certified_candidates) {
    if (!candidate.valid || !candidate.assignment.valid)
      continue;
    if (candidate.hard_feasibility.certified())
      ordered.push_back(&candidate);
    else
      rejected.push_back(&candidate);
  }
  if (better_candidate != nullptr) {
    std::stable_sort(ordered.begin(), ordered.end(),
                     [better_candidate](const auto *lhs, const auto *rhs) {
                       return better_candidate(*lhs, *rhs);
                     });
    std::stable_sort(rejected.begin(), rejected.end(),
                     [better_candidate](const auto *lhs, const auto *rhs) {
                       return better_candidate(*lhs, *rhs);
                     });
  }
  if (ordered.empty())
    return;
  // A rejected evaluation of an assignment that is also certified (a warm
  // start that failed point consistency, say) is not a missing image; only
  // assignments absent from the certified list are exported as rejected.
  std::vector<const PbctopoCutCandidate *> distinct_rejected;
  for (const auto *candidate : rejected) {
    const bool certified_elsewhere = std::any_of(
        ordered.begin(), ordered.end(), [&](const auto *other) {
          return same_pbctopo_image_assignment(other->assignment,
                                                candidate->assignment);
        });
    const bool already = std::any_of(
        distinct_rejected.begin(), distinct_rejected.end(),
        [&](const auto *other) {
          return same_pbctopo_image_assignment(other->assignment,
                                                candidate->assignment);
        });
    if (!certified_elsewhere && !already)
      distinct_rejected.push_back(candidate);
  }
  const bool exports_rejected = !distinct_rejected.empty();

  const PbctopoCutCandidate &pool_best = *ordered.front();
  std::size_t search_equivalent_exported = 0;
  std::size_t search_domain_assignments_exported = 0;
  for (const auto *candidate : ordered) {
    if (candidate->search_equivalence_known) {
      ++search_domain_assignments_exported;
    }
    if (candidate->search_equivalence_known &&
        candidate->equivalent_to_search_best) {
      ++search_equivalent_exported;
    }
  }
  const bool equivalent_set_complete =
      identifiability.uniqueness_search_exhaustive &&
      identifiability.equivalent_best_assignments_within_domain > 0 &&
      search_equivalent_exported ==
          identifiability.equivalent_best_assignments_within_domain;
  const bool all_retained_assignments_have_search_provenance =
      search_domain_assignments_exported == ordered.size();
  const bool credible_alternative_set_complete =
      identifiability.feasible_assignment_set_enumerated &&
      identifiability.feasible_assignments_within_domain > 0 &&
      all_retained_assignments_have_search_provenance &&
      search_domain_assignments_exported ==
          identifiability.feasible_assignments_within_domain;

  for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
    PbctopoCutCandidate candidate = *ordered[rank];
    if (exports_rejected)
      candidate.search_domain_evidence_complete = false;
    const bool selected =
        candidate.hard_feasibility.certified() &&
        framewise_selected != nullptr && framewise_selected->assignment.valid &&
        same_pbctopo_image_assignment(candidate.assignment,
                                       framewise_selected->assignment);
    const bool equivalent_under_output_order =
        equivalent_under_order(pool_best, candidate, better_candidate);
    out << VIBE_OBSERVATION_SCHEMA_VERSION << ','
        << vibe_observation_producer_version() << ','
        << VIBE_OBSERVATION_COORDINATE_UNIT << ','
        << VIBE_OBSERVATION_TIME_UNIT << ','
        << VIBE_OBSERVATION_ENDIANNESS << ','
        << VIBE_OBSERVATION_FLOATING_FORMAT << ','
        << VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_HASH_ALGORITHM << ','
        << VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM << ','
        << VIBE_OBSERVATION_BINDING_MODE << ','
        << VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY << ','
        << VIBE_OBSERVATION_NONFINITE_POLICY << ','
        << VIBE_OBSERVATION_SCOPE << ','
        << topology_epoch_id << ',' << frame << ',' << format_double(time_ps)
        << ',' << rank + 1 << ','
        << (selected ? 1 : 0)
        << ",retained_certified_band,"
        << observation_class(candidate, identifiability, selected,
                             equivalent_set_complete,
                              credible_alternative_set_complete,
                              spatial_lift_ambiguities,
                              spatial_lift_cycle_residuals, frame_status)
        << ','
        << (candidate.search_equivalence_known ? 1 : 0) << ',';
    if (candidate.search_equivalence_known) {
      out << (candidate.equivalent_to_search_best ? 1 : 0);
    } else {
      out << "na";
    }
    out << ',' << (equivalent_under_output_order ? 1 : 0) << ','
        << (equivalent_set_complete ? 1 : 0) << ',' << ordered.size() << ','
        << search_equivalent_exported << ','
        << identifiability.equivalent_best_assignments_within_domain << ','
        << (identifiability.identified ? 1 : 0) << ','
        << (identifiability.unique_within_search_domain ? 1 : 0) << ','
        << (identifiability.uniqueness_search_exhaustive ? 1 : 0) << ','
        << (identifiability.objective_uniqueness_known ? 1 : 0) << ','
        << (identifiability.objective_unique_within_search_domain ? 1 : 0)
        << ',' << (identifiability.evidence_uniqueness_known ? 1 : 0) << ','
        << (identifiability.evidence_unique_within_search_domain ? 1 : 0)
        << ','
        << (identifiability.feasible_assignment_set_enumerated ? 1 : 0)
        << ','
        << (identifiability.feasible_assignment_set_enumerated
                ? "hard_constraints_within_search_domain"
                : "not_enumerated")
        << ',' << identifiability.feasible_assignments_within_domain << ','
        << soft_score_valid_assignments_within_domain << ','
        << search_domain_assignments_exported << ','
        << (credible_alternative_set_complete ? 1 : 0) << ','
        << candidate.component_search_shell_radius << ','
        << (candidate.search_domain_audit_known ? 1 : 0) << ','
        << ((candidate.search_domain_evidence_complete && !exports_rejected)
                ? 1
                : 0)
        << ',';
    if (candidate.search_domain_audit_known)
      out << (candidate.selected_assignment_touches_shell_boundary ? 1 : 0);
    else
      out << "na";
    out << ',' << (candidate.domain_expansion_attempted ? 1 : 0) << ','
        << (candidate.domain_expansion_exhausted ? 1 : 0) << ','
        << (candidate.bounded_domain_only ? 1 : 0) << ','
        << pbctopo_domain_completeness_name(candidate.domain_completeness)
        << ','
        << (candidate.hard_feasibility.certified() ? 1 : 0) << ','
        << candidate.assignment.layout_signature << ','
        << candidate.assignment.signature << ','
        << candidate.evidence_relative_relation_signature << ','
        << format_pbctopo_hash128(
               candidate.evidence_compatible_hypothesis_hash) << ','
        << format_pbctopo_hash128(candidate.evidence_compatible_pair_hash)
        << ',' << candidate.evidence_compatible_hypothesis_count << ','
        << candidate.evidence_supported_relation_count << ','
        << candidate.evidence_compatible_contact_count << ','
        << candidate.evidence_no_support_relation_count << ','
        << candidate.assignment.component_offsets.size() << ','
        << candidate.assignment.global_gauge.x << ','
        << candidate.assignment.global_gauge.y << ','
        << candidate.assignment.global_gauge.z << ','
        << serialize_component_assignment(candidate.assignment);
    for (double value : normalized_gram)
      out << ',' << format_double(value);
    out << ',' << candidate.broken_edges << ','
        << format_double(candidate.continuity_max_d2) << ','
        << format_double(candidate.continuity_path) << ','
        << format_double(candidate.anchor_max_d2) << ','
        << format_double(candidate.anchor_mst2) << ','
        << format_double(candidate.min_pair_d2) << ','
        << format_double(candidate.shift_norm2) << '\n';
  }
}

} // namespace titan_pbctopo

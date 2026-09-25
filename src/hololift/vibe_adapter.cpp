#include "vibe_adapter.h"

#include "frame_binding.h"
#include "identity.h"
#include "validation.h"
#include "vibe_contract.h"

#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace titan_hololift {
namespace {

HoloLiftFrameStatus
convert_status(titan_pbctopo::PbctopoFrameStatus value) noexcept {
  using Source = titan_pbctopo::PbctopoFrameStatus;
  switch (value) {
  case Source::Certified:
    return HoloLiftFrameStatus::Certified;
  case Source::WeakObservation:
    return HoloLiftFrameStatus::WeakObservation;
  case Source::LocalUnwrap:
    return HoloLiftFrameStatus::LocalUnwrap;
  case Source::Rescued:
    return HoloLiftFrameStatus::Rescued;
  case Source::Fallback:
    return HoloLiftFrameStatus::Fallback;
  case Source::Contradicted:
    return HoloLiftFrameStatus::Contradicted;
  }
  return HoloLiftFrameStatus::Fallback;
}

HoloLiftEvidenceState
convert_evidence_state(titan_pbctopo::PbctopoEvidenceState value) noexcept {
  using Source = titan_pbctopo::PbctopoEvidenceState;
  switch (value) {
  case Source::NotEvaluated:
    return HoloLiftEvidenceState::NotEvaluated;
  case Source::NotApplicable:
    return HoloLiftEvidenceState::NotApplicable;
  case Source::SupportedConnected:
    return HoloLiftEvidenceState::SupportedConnected;
  case Source::WeakDisconnected:
    return HoloLiftEvidenceState::WeakDisconnected;
  case Source::Contradicted:
    return HoloLiftEvidenceState::Contradicted;
  case Source::WeakAmbiguous:
    return HoloLiftEvidenceState::WeakAmbiguous;
  }
  return HoloLiftEvidenceState::NotEvaluated;
}

HoloLiftCertificateScope
convert_scope(titan_pbctopo::PbctopoCertificateScope value) noexcept {
  using Source = titan_pbctopo::PbctopoCertificateScope;
  switch (value) {
  case Source::None:
    return HoloLiftCertificateScope::None;
  case Source::LocalUnwrap:
    return HoloLiftCertificateScope::LocalUnwrap;
  case Source::ImageIntercomponentSteric:
    return HoloLiftCertificateScope::ImageIntercomponentSteric;
  case Source::ImageIntercomponentStericInterface:
    return HoloLiftCertificateScope::ImageIntercomponentStericInterface;
  }
  return HoloLiftCertificateScope::None;
}

HoloLiftCertificateGraphSource convert_certificate_graph_source(
    titan_pbctopo::PbctopoCertificateGraphSource value) noexcept {
  using Source = titan_pbctopo::PbctopoCertificateGraphSource;
  switch (value) {
  case Source::GromacsTopology:
    return HoloLiftCertificateGraphSource::GromacsTopology;
  case Source::MetadataChainResidue:
    return HoloLiftCertificateGraphSource::MetadataChainResidue;
  case Source::GeometryCutoff:
    return HoloLiftCertificateGraphSource::GeometryCutoff;
  case Source::Mixed:
    return HoloLiftCertificateGraphSource::Mixed;
  case Source::None:
    return HoloLiftCertificateGraphSource::None;
  }
  return HoloLiftCertificateGraphSource::None;
}

HoloLiftHardGraphSource
convert_hard_graph_source(
    titan_pbctopo::PbctopoHardGraphSource value) noexcept {
  using Source = titan_pbctopo::PbctopoHardGraphSource;
  switch (value) {
  case Source::None:
    return HoloLiftHardGraphSource::None;
  case Source::ExplicitTopology:
    return HoloLiftHardGraphSource::ExplicitTopology;
  case Source::ValidatedMetadata:
    return HoloLiftHardGraphSource::ValidatedMetadata;
  case Source::MixedHard:
    return HoloLiftHardGraphSource::MixedHard;
  }
  return HoloLiftHardGraphSource::None;
}

std::expected<HoloLiftHardEdgeKind, std::string> convert_hard_edge_kind(
    titan_pbctopo::PbctopoObservationHardEdgeKind value) {
  using Source = titan_pbctopo::PbctopoObservationHardEdgeKind;
  switch (value) {
  case Source::TopologyBond:
    return HoloLiftHardEdgeKind::TopologyBond;
  case Source::ValidatedMetadata:
    return HoloLiftHardEdgeKind::ValidatedMetadata;
  }
  return std::unexpected("unknown VIBE hard-edge kind");
}

HoloLiftSearchEvidenceSource convert_search_evidence_source(
    titan_pbctopo::PbctopoSearchEvidenceSource value) noexcept {
  using Source = titan_pbctopo::PbctopoSearchEvidenceSource;
  switch (value) {
  case Source::None:
    return HoloLiftSearchEvidenceSource::None;
  case Source::GeometryCutoff:
    return HoloLiftSearchEvidenceSource::GeometryCutoff;
  case Source::ScoredContacts:
    return HoloLiftSearchEvidenceSource::ScoredContacts;
  case Source::ExplicitContacts:
    return HoloLiftSearchEvidenceSource::ExplicitContacts;
  case Source::Mixed:
    return HoloLiftSearchEvidenceSource::Mixed;
  }
  return HoloLiftSearchEvidenceSource::None;
}

HoloLiftHash128
convert_hash(titan_pbctopo::PbctopoHash128 value) noexcept {
  return {value.lo, value.hi};
}

} // namespace

std::expected<std::uint32_t, std::string>
append_hololift_vibe_topology_epoch(
    HoloLiftObservationStoreBuilder &builder,
    const titan_pbctopo::PbctopoComponentLayoutObservation &layout) {
  const auto contract =
      validate_hololift_vibe_contract(builder.source_contract);
  if (!contract)
    return std::unexpected(contract.error());
  if (!layout.valid || layout.topology_epoch_id == 0 ||
      layout.layout_signature == 0 || layout.components.empty()) {
    return std::unexpected("VIBE topology-epoch layout is incomplete");
  }
  if (builder.topology_epochs.size() >=
      static_cast<std::size_t>(
          std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected("HoloLift topology-epoch index overflow");
  }

  std::vector<HoloLiftSourceAtomKey> local_atoms;
  std::vector<double> local_atom_masses;
  std::vector<HoloLiftComponentRecord> local_components;
  local_components.reserve(layout.components.size());
  for (const auto &source_component : layout.components) {
    if (source_component.component_id == 0 ||
        source_component.exact_atom_membership.empty()) {
      return std::unexpected("VIBE component membership is incomplete");
    }
    HoloLiftComponentRecord component;
    component.component_id = source_component.component_id;
    component.owner = source_component.owner;
    component.mixed_owner = source_component.mixed_owner;
    component.exact_atom_membership.begin = local_atoms.size();
    component.exact_atom_membership.count =
        source_component.exact_atom_membership.size();
    if (!source_component.exact_atom_masses.empty() &&
        source_component.exact_atom_masses.size() !=
            source_component.exact_atom_membership.size()) {
      return std::unexpected(
          "VIBE component atom masses do not match exact membership");
    }
    for (std::size_t atom_index = 0;
         atom_index < source_component.exact_atom_membership.size();
         ++atom_index) {
      const auto &atom = source_component.exact_atom_membership[atom_index];
      local_atoms.push_back(
          {atom.owner, static_cast<std::uint64_t>(atom.source_atom_id)});
      local_atom_masses.push_back(
          source_component.exact_atom_masses.empty()
              ? 0.0
              : source_component.exact_atom_masses[atom_index]);
    }
    component.root_atom = local_atoms[component.exact_atom_membership.begin];
    if (source_component.root_atom_id !=
        component.root_atom.source_atom_id) {
      return std::unexpected(
          "VIBE component root does not match exact membership");
    }
    const auto component_identity = hololift_component_identity128(
        std::span<const HoloLiftSourceAtomKey>(local_atoms)
            .subspan(component.exact_atom_membership.begin,
                     component.exact_atom_membership.count));
    if (!component_identity)
      return std::unexpected(component_identity.error());
    component.component_identity = *component_identity;
    local_components.push_back(component);
  }
  for (std::size_t idx = 1; idx < local_components.size(); ++idx) {
    if (local_components[idx - 1].component_id >=
        local_components[idx].component_id) {
      return std::unexpected(
          "VIBE topology-epoch components are not strictly ordered");
    }
  }
  std::vector<HoloLiftHardEdgeRecord> local_hard_edges;
  local_hard_edges.reserve(layout.hard_edges.size());
  for (const auto &source_edge : layout.hard_edges) {
    const auto kind = convert_hard_edge_kind(source_edge.kind);
    if (!kind)
      return std::unexpected(kind.error());
    local_hard_edges.push_back(
        {source_edge.atom_a, source_edge.atom_b, *kind});
  }
  const auto canonical_atom_universe_hash =
      hash_hololift_canonical_atom_universe(local_atoms);
  if (!canonical_atom_universe_hash)
    return std::unexpected(canonical_atom_universe_hash.error());
  if (layout.canonical_atom_universe_hash.empty() ||
      convert_hash(layout.canonical_atom_universe_hash) !=
          *canonical_atom_universe_hash) {
    return std::unexpected(
        "VIBE topology epoch canonical atom universe mismatch");
  }
  const std::size_t component_forest_edge_count =
      local_atoms.size() - local_components.size();
  if (local_hard_edges.size() < component_forest_edge_count) {
    return std::unexpected(
        "VIBE topology epoch hard graph cannot connect every component");
  }

  const auto layout_signature =
      hololift_layout_signature(local_components);
  if (!layout_signature ||
      *layout_signature != layout.layout_signature) {
    return std::unexpected(
        layout_signature ? "VIBE/HoloLift layout identity mismatch"
                          : layout_signature.error());
  }
  const auto layout_identity =
      hololift_layout_identity128(local_components);
  if (!layout_identity)
    return std::unexpected(layout_identity.error());
  const auto topology_epoch_id = hololift_topology_epoch_id(
      builder.source_contract.schema_version,
      convert_hard_graph_source(layout.hard_graph_source),
      layout.layout_signature, local_components, local_atoms,
      local_hard_edges);
  if (!topology_epoch_id ||
      *topology_epoch_id != layout.topology_epoch_id) {
    return std::unexpected(
        topology_epoch_id ? "VIBE/HoloLift topology epoch mismatch"
                          : topology_epoch_id.error());
  }
  const auto topology_epoch_identity = hololift_topology_epoch_identity128(
      builder.source_contract.schema_version,
      convert_hard_graph_source(layout.hard_graph_source), *layout_identity,
      local_components, local_hard_edges);
  if (!topology_epoch_identity)
    return std::unexpected(topology_epoch_identity.error());

  const std::size_t atom_base = builder.exact_atom_memberships.size();
  const std::size_t component_base = builder.components.size();
  const std::size_t hard_edge_base = builder.hard_edges.size();
  for (auto &component : local_components)
    component.exact_atom_membership.begin += atom_base;
  if (builder.exact_atom_masses.empty() && atom_base != 0)
    builder.exact_atom_masses.assign(atom_base, 0.0);
  if (builder.exact_atom_masses.size() != atom_base)
    return std::unexpected("HoloLift atom-mass table is not append-packed");
  builder.exact_atom_memberships.insert(
      builder.exact_atom_memberships.end(), local_atoms.begin(),
      local_atoms.end());
  builder.exact_atom_masses.insert(builder.exact_atom_masses.end(),
                                   local_atom_masses.begin(),
                                   local_atom_masses.end());
  builder.components.insert(builder.components.end(),
                            local_components.begin(),
                            local_components.end());
  builder.hard_edges.insert(builder.hard_edges.end(),
                            local_hard_edges.begin(),
                            local_hard_edges.end());
  HoloLiftTopologyEpochRecord epoch;
  epoch.topology_epoch_id = layout.topology_epoch_id;
  epoch.layout_signature = layout.layout_signature;
  epoch.components = {component_base, local_components.size()};
  epoch.hard_edges = {hard_edge_base, local_hard_edges.size()};
  epoch.hard_graph_source =
      convert_hard_graph_source(layout.hard_graph_source);
  epoch.layout_identity = *layout_identity;
  epoch.topology_epoch_identity = *topology_epoch_identity;
  epoch.canonical_atom_universe_hash = *canonical_atom_universe_hash;
  epoch.atom_count = local_atoms.size();
  epoch.hard_graph_cycle_rank =
      local_hard_edges.size() - component_forest_edge_count;
  builder.topology_epochs.push_back(epoch);
  return static_cast<std::uint32_t>(
      builder.topology_epochs.size() - 1);
}

std::expected<HoloLiftFrameProvenance, std::string>
hololift_vibe_frame_provenance(
    const titan_pbctopo::PbctopoFrameReport &report,
    std::uint64_t framewise_assignment_signature) {
  const bool hard_feasible_output =
      report.status == titan_pbctopo::PbctopoFrameStatus::Certified ||
      report.status == titan_pbctopo::PbctopoFrameStatus::Rescued ||
      report.status == titan_pbctopo::PbctopoFrameStatus::WeakObservation ||
      report.status == titan_pbctopo::PbctopoFrameStatus::Contradicted;
  if (hard_feasible_output &&
      (!report.hard_feasible || !report.hard_feasibility.certified())) {
    return std::unexpected(
        "VIBE hard-feasible output lacks a matching all-atom certificate");
  }
  const bool evaluated_evidence =
      report.evidence_state ==
          titan_pbctopo::PbctopoEvidenceState::SupportedConnected ||
      report.evidence_state ==
          titan_pbctopo::PbctopoEvidenceState::WeakDisconnected ||
      report.evidence_state ==
          titan_pbctopo::PbctopoEvidenceState::WeakAmbiguous ||
      report.evidence_state == titan_pbctopo::PbctopoEvidenceState::Contradicted;
  if (evaluated_evidence &&
      (!report.soft_observed_contact_construction_attempted ||
       !report.soft_observed_contact_construction_complete ||
       report.soft_observed_contact_cell_setup_failures != 0 ||
       report.soft_observed_contact_atom_index_failures != 0 ||
       report.soft_observed_contact_mic_query_failures != 0)) {
    return std::unexpected(
        "VIBE evaluated evidence depends on incomplete E_obs construction");
  }
  if (!report.temporal_inference_experimental &&
      report.assignment_signature != framewise_assignment_signature) {
    return std::unexpected(
        "production VIBE frame/report assignment signatures disagree");
  }
  HoloLiftFrameProvenance provenance;
  provenance.status = convert_status(report.status);
  provenance.certificate_scope =
      convert_scope(report.certificate_scope);
  provenance.certificate_graph_source =
      convert_certificate_graph_source(report.certificate_graph_source);
  provenance.hard_graph_source =
      convert_hard_graph_source(report.hard_graph_source);
  provenance.search_evidence_source =
      convert_search_evidence_source(report.search_evidence_source);
  provenance.temporal_policy =
      report.temporal_inference_experimental
          ? HoloLiftTemporalPolicyStatus::ExperimentalViterbiDiagnostic
          : HoloLiftTemporalPolicyStatus::FramewiseProduction;
  provenance.evidence_state = convert_evidence_state(report.evidence_state);
  provenance.evidence_consistency_evaluated =
      report.evidence_consistency_evaluated;
  provenance.evidence_consistent = report.evidence_consistent;
  provenance.evidence_graph_connected = report.evidence_graph_connected;
  provenance.soft_observed_contact_pair_hash =
      convert_hash(report.soft_observed_contact_pair_hash);
  provenance.soft_observed_lost_pair_hash =
      convert_hash(report.soft_observed_lost_pair_hash);
  provenance.soft_observed_all_hypotheses_hash =
      convert_hash(report.soft_observed_all_hypotheses_hash);
  provenance.soft_observed_selected_hypothesis_hash =
      convert_hash(report.soft_observed_selected_hypothesis_hash);
  provenance.soft_observed_selected_compatible_pair_hash =
      convert_hash(report.soft_observed_selected_compatible_pair_hash);
  provenance.soft_observed_hypothesis_count =
      report.soft_observed_hypothesis_count;
  provenance.soft_observed_component_relation_count =
      report.soft_observed_component_relation_count;
  provenance.soft_observed_ambiguous_relation_count =
      report.soft_observed_ambiguous_relation_count;
  provenance.soft_observed_compatible_hypothesis_count =
      report.soft_observed_compatible_hypothesis_count;
  provenance.soft_observed_selected_compatible_contact_count =
      report.soft_observed_selected_compatible_contact_count;
  provenance.soft_observed_alternative_hypothesis_contact_count =
      report.soft_observed_alternative_hypothesis_contact_count;
  provenance.soft_observed_selected_compatible_contact_loss_count =
      report.soft_observed_selected_compatible_contact_loss_count;
  provenance.soft_observed_no_support_relation_count =
      report.soft_observed_no_support_relation_count;
  provenance.soft_observed_selected_relation_has_support =
      report.soft_observed_selected_relation_has_support;
  provenance.soft_observed_hypothesis_construction_attempted =
      report.soft_observed_hypothesis_construction_attempted;
  provenance.soft_observed_hypothesis_construction_complete =
      report.soft_observed_hypothesis_construction_complete;
  provenance.soft_observed_hypothesis_mic_ambiguity_failures =
      report.soft_observed_hypothesis_mic_ambiguity_failures;
  provenance.soft_observed_hypothesis_component_mapping_failures =
      report.soft_observed_hypothesis_component_mapping_failures;
  provenance.soft_observed_hypothesis_internal_edges_ignored =
      report.soft_observed_hypothesis_internal_edges_ignored;
  provenance.soft_observed_contacts_checked =
      report.soft_observed_contacts_checked;
  provenance.soft_observed_contacts_lost = report.soft_observed_contacts_lost;
  provenance.soft_observed_contact_cutoff_A =
      report.soft_observed_contact_cutoff_A;
  provenance.soft_observed_contact_construction_attempted =
      report.soft_observed_contact_construction_attempted;
  provenance.soft_observed_contact_construction_complete =
      report.soft_observed_contact_construction_complete;
  provenance.soft_observed_contact_cell_setup_failures =
      report.soft_observed_contact_cell_setup_failures;
  provenance.soft_observed_contact_atom_index_failures =
      report.soft_observed_contact_atom_index_failures;
  provenance.soft_observed_contact_mic_query_failures =
      report.soft_observed_contact_mic_query_failures;
  provenance.framewise_assignment_signature =
      framewise_assignment_signature;
  provenance.output_assignment_signature =
      report.assignment_signature;
  provenance.temporal_assignment_signature =
      report.temporal_assignment_signature;
  provenance.temporal_selected = report.temporal_selected;
  provenance.temporal_changed_from_framewise =
      report.temporal_changed_from_greedy;
  provenance.source_frame_report_present = true;
  return provenance;
}

std::expected<std::size_t, std::string> append_hololift_vibe_frame(
    HoloLiftObservationStoreBuilder &builder,
    HoloLiftVibeFrameAppendInput input) {
  const auto contract =
      validate_hololift_vibe_contract(builder.source_contract);
  if (!contract)
    return std::unexpected(contract.error());
  if (input.frame.topology_epoch_index >= builder.topology_epochs.size())
    return std::unexpected("VIBE frame references an invalid topology epoch");

  const auto &source_epoch =
      builder.topology_epochs[input.frame.topology_epoch_index];
  if (input.frame.topology_epoch_id != source_epoch.topology_epoch_id ||
      input.frame.layout_signature != source_epoch.layout_signature ||
      !source_epoch.components.valid_for(builder.components.size())) {
    return std::unexpected(
        "VIBE frame topology epoch/layout cross-link is invalid");
  }
  const auto components =
      std::span<const HoloLiftComponentRecord>(builder.components)
          .subspan(source_epoch.components.begin,
                   source_epoch.components.count);
  HoloLiftFrameRecord local_frame = std::move(input.frame);
  if (local_frame.canonical_atom_universe_hash.empty() ||
      local_frame.canonical_atom_universe_hash !=
          source_epoch.canonical_atom_universe_hash) {
    return std::unexpected(
        "VIBE frame canonical atom universe does not match topology epoch");
  }
  local_frame.topology_epoch_identity =
      source_epoch.topology_epoch_identity;
  local_frame.layout_identity = source_epoch.layout_identity;
  if (input.candidates.empty()) {
    if (local_frame.provenance.status != HoloLiftFrameStatus::Contradicted ||
        local_frame.provenance.evidence_state !=
            HoloLiftEvidenceState::Contradicted) {
      return std::unexpected(
          "only contradicted VIBE frames may omit a retained candidate");
    }
    if (local_frame.evidence.retained_certified_assignment_count != 0 ||
        local_frame.evidence.search_equivalent_assignments_exported != 0 ||
        local_frame.evidence.search_domain_assignments_exported != 0 ||
        local_frame.evidence.equivalent_set_complete ||
        local_frame.evidence.credible_alternative_set_complete) {
      return std::unexpected(
          "contradicted VIBE gap carries nonzero candidate/export counters");
    }
    local_frame.candidates = {builder.candidates.size(), 0};
    local_frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
    local_frame.candidate_set_scope = HoloLiftCandidateSetScope::Unknown;
    HoloLiftObservationStoreBuilder scratch;
    scratch.topology_epochs.push_back(source_epoch);
    scratch.topology_epochs.front().components = {0, components.size()};
    scratch.components.assign(components.begin(), components.end());
    scratch.frames.push_back(local_frame);
    scratch.frames.front().topology_epoch_index = 0;
    scratch.frames.front().candidates = {0, 0};
    const auto frame_validation = validate_hololift_frame(scratch, 0);
    if (!frame_validation) {
      return std::unexpected(
          std::string(hololift_validation_code_name(
              frame_validation.error().code)) +
          ": " + frame_validation.error().message);
    }
    builder.frames.push_back(std::move(local_frame));
    return builder.frames.size() - 1;
  }
  local_frame.candidates = {0, input.candidates.size()};
  local_frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
  local_frame.candidate_set_scope =
      HoloLiftCandidateSetScope::RetainedCertifiedBand;

  std::vector<HoloLiftCandidateRecord> local_candidates;
  std::vector<HoloLiftLatticeImage> local_images;
  local_candidates.reserve(input.candidates.size());
  local_images.reserve(input.candidates.size() * components.size());
  std::size_t selected_count = 0;
  std::size_t search_equivalent_exported = 0;
  std::size_t search_domain_exported = 0;
  std::uint64_t selected_signature = 0;
  HoloLiftHash128 selected_identity;

  for (std::size_t idx = 0; idx < input.candidates.size(); ++idx) {
    auto &source = input.candidates[idx];
    if (source.record.rank != idx + 1 ||
        source.raw_component_images.size() != components.size()) {
      return std::unexpected(
          "VIBE candidate rank or component count is invalid");
    }
    auto canonical = canonicalize_hololift_assignment(
        source_epoch.layout_signature, components,
        source.raw_component_images);
    if (!canonical)
      return std::unexpected(canonical.error());
    if (canonical->global_gauge != HoloLiftLatticeImage{}) {
      return std::unexpected(
          "VIBE candidate assignment is not in the canonical global gauge");
    }
    if ((source.record.layout_signature != 0 &&
         source.record.layout_signature != source_epoch.layout_signature) ||
        (source.record.assignment_signature != 0 &&
         source.record.assignment_signature !=
             canonical->assignment_signature)) {
      return std::unexpected(
          "VIBE candidate identity differs from its canonical assignment");
    }

    source.record.layout_signature = source_epoch.layout_signature;
    source.record.assignment_signature = canonical->assignment_signature;
    source.record.layout_identity = canonical->layout_identity;
    source.record.assignment_identity = canonical->assignment_identity;
    source.record.component_images =
        {local_images.size(), canonical->component_images.size()};
    for (const auto &component_image : canonical->component_images)
      local_images.push_back(component_image.image);
    if (source.record.evidence.selected_framewise) {
      ++selected_count;
      selected_signature = source.record.assignment_signature;
      selected_identity = source.record.assignment_identity;
      local_frame.selected_candidate_index = idx;
    }
    if (source.record.evidence.search_equivalence_known) {
      ++search_domain_exported;
      if (source.record.evidence.equivalent_to_search_best)
        ++search_equivalent_exported;
    }
    local_candidates.push_back(std::move(source.record));
  }
  if (selected_count != 1)
    return std::unexpected(
        "VIBE retained band must contain exactly one framewise selection");

  local_frame.evidence.retained_certified_assignment_count =
      local_candidates.size();
  local_frame.evidence.search_equivalent_assignments_exported =
      search_equivalent_exported;
  local_frame.evidence.search_domain_assignments_exported =
      search_domain_exported;
  local_frame.evidence.equivalent_set_complete =
      local_frame.evidence.uniqueness_search_exhaustive &&
      local_frame.evidence.equivalent_best_assignments_within_domain > 0 &&
      search_equivalent_exported ==
          local_frame.evidence.equivalent_best_assignments_within_domain;
  local_frame.evidence.credible_alternative_set_complete =
      local_frame.evidence.feasible_assignment_set_enumerated &&
      local_frame.evidence.feasible_assignments_within_domain > 0 &&
      search_domain_exported == local_candidates.size() &&
      search_domain_exported ==
          local_frame.evidence.feasible_assignments_within_domain;
  local_frame.provenance.framewise_assignment_signature =
      selected_signature;
  local_frame.provenance.framewise_assignment_identity = selected_identity;
  if (local_frame.provenance.temporal_policy ==
      HoloLiftTemporalPolicyStatus::FramewiseProduction) {
    if (local_frame.provenance.output_assignment_signature != 0 &&
        local_frame.provenance.output_assignment_signature !=
            selected_signature) {
      return std::unexpected(
          "VIBE output assignment differs from the framewise selection");
    }
    local_frame.provenance.output_assignment_signature = selected_signature;
    local_frame.provenance.output_assignment_identity = selected_identity;
    local_frame.provenance.temporal_assignment_identity = {};
  } else if (local_frame.provenance.temporal_selected) {
    for (const auto &candidate : local_candidates) {
      if (candidate.assignment_signature ==
          local_frame.provenance.temporal_assignment_signature) {
        local_frame.provenance.temporal_assignment_identity =
            candidate.assignment_identity;
        break;
      }
    }
    local_frame.provenance.output_assignment_identity =
        local_frame.provenance.temporal_assignment_identity;
  } else {
    local_frame.provenance.output_assignment_identity = selected_identity;
    local_frame.provenance.temporal_assignment_identity = {};
  }

  HoloLiftObservationStoreBuilder scratch;
  scratch.topology_epochs.push_back(source_epoch);
  scratch.topology_epochs.front().components = {0, components.size()};
  scratch.components.assign(components.begin(), components.end());
  scratch.frames.push_back(local_frame);
  scratch.frames.front().topology_epoch_index = 0;
  scratch.candidates = local_candidates;
  scratch.component_images = local_images;
  const auto frame_validation = validate_hololift_frame(scratch, 0);
  if (!frame_validation) {
    return std::unexpected(
        std::string(hololift_validation_code_name(
            frame_validation.error().code)) +
        ": " + frame_validation.error().message);
  }

  const std::size_t frame_index = builder.frames.size();
  const std::size_t candidate_base = builder.candidates.size();
  const std::size_t image_base = builder.component_images.size();
  builder.frames.reserve(frame_index + 1);
  builder.candidates.reserve(candidate_base + local_candidates.size());
  builder.component_images.reserve(image_base + local_images.size());
  for (auto &candidate : local_candidates)
    candidate.component_images.begin += image_base;
  local_frame.candidates.begin = candidate_base;
  local_frame.selected_candidate_index += candidate_base;
  builder.component_images.insert(builder.component_images.end(),
                                  local_images.begin(), local_images.end());
  builder.candidates.insert(builder.candidates.end(), local_candidates.begin(),
                            local_candidates.end());
  builder.frames.push_back(std::move(local_frame));
  return frame_index;
}

} // namespace titan_hololift

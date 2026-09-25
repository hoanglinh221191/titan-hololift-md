param(
    [string]$Compiler = 'g++',
    [string]$WorkRoot = (Join-Path $PSScriptRoot 'tmp\hololift_phase0_smoke'),
    [string[]]$ExtraCompilerFlags = @(),
    [string]$LegacyArtifact = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$src = Join-Path $WorkRoot 'hololift_phase0_smoke.cpp'
$exe = Join-Path $WorkRoot 'hololift_phase0_smoke.exe'

@'
#include "hololift/durable_artifact.h"
#include "hololift/phase0.h"
#include "hololift/phase1.h"
#include "hololift/serialization.h"
#include "hololift/vibe_adapter.h"
#include "pbctopo/image_assignment.h"
#include "pbctopo/observation_binding.h"
#include "pbctopo/observation_layout.h"
#include "pbctopo/spatial_lift.h"
#include "titan_sha256.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace titan_hololift;

void expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL hololift_phase0_smoke: " << message << "\n";
    std::exit(1);
  }
}

struct ComponentFixture {
  std::vector<HoloLiftSourceAtomKey> atoms;
  std::uint64_t component_id = 0;
};

HoloLiftObservationStoreBuilder make_valid_builder() {
  HoloLiftObservationStoreBuilder store;
  store.source_contract =
      make_hololift_vibe_v1_contract(
          "hololift-phase0-smoke", "synthetic-package",
          "synthetic-trajectory");

  std::vector<ComponentFixture> fixtures{
      {{{0, 10}, {0, 12}}, 0},
      {{{1, 20}, {1, 21}}, 0},
  };
  for (auto &fixture : fixtures) {
    const auto component_id = hololift_component_id(fixture.atoms);
    expect(component_id.has_value(), "component identity was not built");
    fixture.component_id = *component_id;
  }
  std::sort(fixtures.begin(), fixtures.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.component_id < rhs.component_id;
            });

  for (const auto &fixture : fixtures) {
    const std::size_t atom_begin = store.exact_atom_memberships.size();
    store.exact_atom_memberships.insert(
        store.exact_atom_memberships.end(), fixture.atoms.begin(),
        fixture.atoms.end());
    HoloLiftComponentRecord component;
    component.component_id = fixture.component_id;
    component.owner = fixture.atoms.front().owner;
    component.root_atom = fixture.atoms.front();
    component.exact_atom_membership =
        {atom_begin, fixture.atoms.size()};
    const auto component_identity =
        hololift_component_identity128(fixture.atoms);
    expect(component_identity.has_value(),
           "128-bit component identity was not built");
    component.component_identity = *component_identity;
    store.components.push_back(component);
  }
  store.exact_atom_masses.reserve(store.exact_atom_memberships.size());
  for (const auto &atom : store.exact_atom_memberships) {
    store.exact_atom_masses.push_back(
        1.0 + static_cast<double>(atom.source_atom_id % 17));
  }

  const auto components =
      std::span<const HoloLiftComponentRecord>(store.components);
  const auto layout = hololift_layout_signature(components);
  expect(layout.has_value(), "layout identity was not built");
  const auto layout_identity = hololift_layout_identity128(components);
  expect(layout_identity.has_value(), "128-bit layout identity was not built");
  store.hard_edges = {
      {0, 1, HoloLiftHardEdgeKind::TopologyBond},
      {2, 3, HoloLiftHardEdgeKind::TopologyBond},
  };
  const auto epoch_id = hololift_topology_epoch_id(
      store.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout, components,
      std::span<const HoloLiftSourceAtomKey>(
          store.exact_atom_memberships),
      std::span<const HoloLiftHardEdgeRecord>(store.hard_edges));
  expect(epoch_id.has_value(), "topology epoch identity was not built");
  const auto epoch_identity = hololift_topology_epoch_identity128(
      store.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout_identity, components,
      store.hard_edges);
  expect(epoch_identity.has_value(),
         "128-bit topology epoch identity was not built");
  HoloLiftTopologyEpochRecord topology_epoch;
  topology_epoch.topology_epoch_id = *epoch_id;
  topology_epoch.layout_signature = *layout;
  topology_epoch.components = {0, store.components.size()};
  topology_epoch.hard_edges = {0, store.hard_edges.size()};
  topology_epoch.hard_graph_source =
      HoloLiftHardGraphSource::ExplicitTopology;
  topology_epoch.layout_identity = *layout_identity;
  topology_epoch.topology_epoch_identity = *epoch_identity;
  const auto canonical_atom_universe_hash =
      hash_hololift_canonical_atom_universe(store.exact_atom_memberships);
  expect(canonical_atom_universe_hash.has_value(),
         "canonical atom-universe hash was not built");
  topology_epoch.canonical_atom_universe_hash =
      *canonical_atom_universe_hash;
  topology_epoch.atom_count = store.exact_atom_memberships.size();
  topology_epoch.hard_graph_cycle_rank =
      store.hard_edges.size() - store.exact_atom_memberships.size() +
      store.components.size();
  store.topology_epochs.push_back(topology_epoch);

  std::vector<titan_pbctopo::PbctopoAtom> pbctopo_atoms(4);
  pbctopo_atoms[0].owner = 0;
  pbctopo_atoms[0].source_atom_id = 12;
  pbctopo_atoms[1].owner = 0;
  pbctopo_atoms[1].source_atom_id = 10;
  pbctopo_atoms[2].owner = 1;
  pbctopo_atoms[2].source_atom_id = 21;
  pbctopo_atoms[3].owner = 1;
  pbctopo_atoms[3].source_atom_id = 20;
  const std::vector<std::vector<std::size_t>> pbctopo_components{
      {0, 1}, {2, 3}};
  const std::vector<titan_pbctopo::PbctopoLocalEdge> pbctopo_hard_edges{
      {0, 1, 1.0, 1.0, 4.0,
       titan_pbctopo::PbctopoLocalEdgeKind::TopologyBond},
      {2, 3, 1.0, 1.0, 4.0,
       titan_pbctopo::PbctopoLocalEdgeKind::TopologyBond}};
  const auto pbctopo_layout =
      titan_pbctopo::build_pbctopo_component_layout_observation(
          pbctopo_components, pbctopo_atoms, pbctopo_hard_edges,
          titan_pbctopo::PbctopoHardGraphSource::ExplicitTopology);
  expect(pbctopo_layout.valid &&
             pbctopo_layout.layout_signature == *layout &&
             pbctopo_layout.topology_epoch_id == *epoch_id,
         "HoloLift identity does not match the VIBE v1 producer");
  HoloLiftObservationStoreBuilder adapter_probe;
  adapter_probe.source_contract = store.source_contract;
  const auto adapted_epoch =
      append_hololift_vibe_v1_topology_epoch(
          adapter_probe, pbctopo_layout);
  expect(adapted_epoch.has_value() && *adapted_epoch == 0 &&
             adapter_probe.topology_epochs.size() == 1 &&
             adapter_probe.components.size() == store.components.size() &&
             adapter_probe.exact_atom_memberships ==
                  store.exact_atom_memberships &&
              adapter_probe.hard_edges == store.hard_edges &&
              adapter_probe.topology_epochs.front().topology_epoch_identity ==
                  *epoch_identity,
         "VIBE v1 topology adapter lost exact membership");
  auto unknown_edge_layout = pbctopo_layout;
  unknown_edge_layout.hard_edges[0].kind =
      static_cast<titan_pbctopo::PbctopoObservationHardEdgeKind>(255);
  HoloLiftObservationStoreBuilder unknown_edge_builder;
  unknown_edge_builder.source_contract = store.source_contract;
  expect(!append_hololift_vibe_topology_epoch(
              unknown_edge_builder, unknown_edge_layout)
              .has_value(),
         "unknown VIBE hard-edge enum was silently converted");
  const std::vector<titan_pbctopo::Int3> pbctopo_offsets{
      {0, 0, 0}, {1, -1, 0}};
  const auto pbctopo_assignment =
      titan_pbctopo::make_pbctopo_image_assignment(
          pbctopo_offsets, pbctopo_components, pbctopo_atoms);
  expect(pbctopo_assignment.valid,
         "VIBE reference assignment was not built");
  std::vector<HoloLiftComponentImage> raw_a;
  for (std::size_t idx = 0;
       idx < pbctopo_assignment.component_ids.size(); ++idx) {
    const auto &offset = pbctopo_assignment.component_offsets[idx];
    raw_a.push_back({pbctopo_assignment.component_ids[idx],
                     {offset.x, offset.y, offset.z}});
  }
  std::vector<HoloLiftComponentImage> raw_b = raw_a;
  for (auto &entry : raw_b) {
    entry.image.x += 3;
    entry.image.y += 4;
    entry.image.z -= 2;
  }
  const auto assignment_a =
      canonicalize_hololift_assignment(*layout, components, raw_a);
  const auto assignment_b =
      canonicalize_hololift_assignment(*layout, components, raw_b);
  expect(assignment_a.has_value() && assignment_b.has_value(),
         "assignment gauge normalization failed");
  expect(assignment_a->assignment_signature ==
             assignment_b->assignment_signature &&
             assignment_a->component_images ==
                 assignment_b->component_images,
         "global lattice gauge changed canonical assignment identity");
  expect(assignment_a->assignment_signature ==
              pbctopo_assignment.signature,
         "HoloLift assignment signature does not match VIBE v1");
  std::vector<HoloLiftLatticeImage> canonical_images;
  for (const auto &entry : assignment_a->component_images)
    canonical_images.push_back(entry.image);
  const auto fast_signature = hololift_assignment_signature_canonical(
      *layout, components, canonical_images);
  expect(fast_signature.has_value() &&
             *fast_signature == assignment_a->assignment_signature &&
             !assignment_a->assignment_identity.empty() &&
             assignment_a->assignment_identity ==
                 assignment_b->assignment_identity,
         "canonical O(C) or 128-bit assignment identity is inconsistent");

  for (const auto &entry : assignment_a->component_images)
    store.component_images.push_back(entry.image);
  HoloLiftCandidateRecord candidate;
  candidate.rank = 1;
  candidate.layout_signature = *layout;
  candidate.assignment_signature =
      assignment_a->assignment_signature;
  candidate.evidence_relative_relation_signature =
      assignment_a->assignment_signature;
  candidate.evidence_compatible_hypothesis_hash = {0x47, 0x48};
  candidate.evidence_compatible_pair_hash = {0x49, 0x4a};
  candidate.evidence_compatible_hypothesis_count = 1;
  candidate.evidence_supported_relation_count = 1;
  candidate.evidence_compatible_contact_count = 1;
  candidate.layout_identity = assignment_a->layout_identity;
  candidate.assignment_identity = assignment_a->assignment_identity;
  candidate.component_images =
      {0, store.component_images.size()};
  candidate.observation_class =
      HoloLiftObservationClass::StrongBoundedAnchor;
  candidate.evidence.selected_framewise = true;
  candidate.evidence.hard_feasible = true;
  candidate.evidence.search_equivalence_known = true;
  candidate.evidence.equivalent_to_search_best = true;
  candidate.evidence.equivalent_under_output_order = true;
  candidate.domain_evidence.boundary_known = true;
  store.candidates.push_back(candidate);

  HoloLiftFrameRecord frame;
  frame.frame = 7;
  frame.time_ps = 3.5;
  frame.trajectory_frame_index = 7;
  frame.topology_epoch_index = 0;
  frame.topology_epoch_id = *epoch_id;
  frame.layout_signature = *layout;
  frame.topology_epoch_identity = *epoch_identity;
  frame.layout_identity = *layout_identity;
  frame.box_matrix = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                      0.0, 0.0, 10.0};
  frame.box_hash = hash_hololift_box_matrix(frame.box_matrix);
  expect(frame.box_hash ==
             HoloLiftHash128{0xc345ccc5e36c28bbULL,
                             0x42769feb663a2497ULL},
         "box SHA-256 identity changed");
  const std::array<HoloLiftSourceAtomKey, 4> atom_order{
      HoloLiftSourceAtomKey{0, 12}, HoloLiftSourceAtomKey{0, 10},
      HoloLiftSourceAtomKey{1, 21}, HoloLiftSourceAtomKey{1, 20}};
  frame.atom_selection_hash = hash_hololift_atom_selection(atom_order);
  frame.canonical_atom_universe_hash = *canonical_atom_universe_hash;
  expect(frame.atom_selection_hash ==
             HoloLiftHash128{0x92ade3194e64b50cULL,
                             0x2d0721d43d1c78a7ULL},
         "atom-selection SHA-256 identity changed");
  const std::array<std::array<double, 3>, 4> wrapped_coordinates{
      std::array<double, 3>{1.0, 1.0, 1.0},
      std::array<double, 3>{2.0, 1.0, 1.0},
      std::array<double, 3>{5.0, 5.0, 5.0},
      std::array<double, 3>{6.0, 5.0, 5.0}};
  frame.wrapped_coordinate_hash =
      hash_hololift_wrapped_coordinates(wrapped_coordinates);
  expect(frame.wrapped_coordinate_hash ==
             HoloLiftHash128{0x7bb5135b5b3e4a82ULL,
                             0x6d982a17a04e0766ULL},
         "wrapped-coordinate SHA-256 identity changed");
  frame.frame_binding_valid = true;
  frame.spatial_lift_policy_version =
      titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
  frame.spatial_lift_valid = true;
  const auto spatial_lift = replay_hololift_spatial_lift(
      store.topology_epochs.front(), frame, store.hard_edges, atom_order,
      wrapped_coordinates);
  expect(spatial_lift.has_value(), "spatial lift was not replayed");
  frame.spatial_lift_identity = spatial_lift->identity;
  frame.spatial_lift_ambiguous_hard_edges =
      spatial_lift->ambiguous_hard_edges;
  frame.spatial_lift_cycle_residuals = spatial_lift->cycle_residuals;
  expect(validate_hololift_spatial_lift_replay(
             store.topology_epochs.front(), frame, store.hard_edges,
             atom_order, wrapped_coordinates)
             .has_value(),
         "spatial-lift identity did not replay exactly");
  frame.candidates = {0, 1};
  frame.selected_candidate_index = 0;
  frame.candidate_set_scope =
      HoloLiftCandidateSetScope::RetainedCertifiedBand;
  frame.provenance.status = HoloLiftFrameStatus::Certified;
  frame.provenance.certificate_scope =
      HoloLiftCertificateScope::ImageIntercomponentSteric;
  frame.provenance.certificate_graph_source =
      HoloLiftCertificateGraphSource::GromacsTopology;
  frame.provenance.hard_graph_source =
      HoloLiftHardGraphSource::ExplicitTopology;
  frame.provenance.search_evidence_source =
      HoloLiftSearchEvidenceSource::ScoredContacts;
  frame.provenance.evidence_state =
      HoloLiftEvidenceState::SupportedConnected;
  frame.provenance.evidence_consistency_evaluated = true;
  frame.provenance.evidence_consistent = true;
  frame.provenance.evidence_graph_connected = true;
  frame.provenance.soft_observed_contact_pair_hash = {0x11, 0x22};
  frame.provenance.soft_observed_lost_pair_hash = {0x33, 0x44};
  frame.provenance.soft_observed_all_hypotheses_hash = {0x45, 0x46};
  frame.provenance.soft_observed_selected_hypothesis_hash = {0x47, 0x48};
  frame.provenance.soft_observed_selected_compatible_pair_hash = {0x49, 0x4a};
  frame.provenance.soft_observed_hypothesis_count = 1;
  frame.provenance.soft_observed_component_relation_count = 1;
  frame.provenance.soft_observed_compatible_hypothesis_count = 1;
  frame.provenance.soft_observed_selected_compatible_contact_count = 1;
  frame.provenance.soft_observed_alternative_hypothesis_contact_count = 0;
  frame.provenance.soft_observed_selected_compatible_contact_loss_count = 0;
  frame.provenance.soft_observed_selected_relation_has_support = true;
  frame.provenance.soft_observed_hypothesis_construction_attempted = true;
  frame.provenance.soft_observed_hypothesis_construction_complete = true;
  frame.provenance.soft_observed_contact_construction_attempted = true;
  frame.provenance.soft_observed_contact_construction_complete = true;
  frame.provenance.soft_observed_contacts_checked = 1;
  frame.provenance.soft_observed_contact_cutoff_A = 4.5;
  frame.provenance.framewise_assignment_signature =
      assignment_a->assignment_signature;
  frame.provenance.output_assignment_signature =
      assignment_a->assignment_signature;
  frame.provenance.framewise_assignment_identity =
      assignment_a->assignment_identity;
  frame.provenance.output_assignment_identity =
      assignment_a->assignment_identity;
  frame.provenance.source_frame_report_present = true;
  frame.evidence.identified = true;
  frame.evidence.uniqueness_search_exhaustive = true;
  frame.evidence.objective_uniqueness_known = true;
  frame.evidence.objective_unique_within_search_domain = true;
  frame.evidence.evidence_uniqueness_known = true;
  frame.evidence.evidence_unique_within_search_domain = true;
  frame.evidence.feasible_assignment_set_enumerated = true;
  frame.evidence.feasible_set_semantics =
      HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain;
  frame.evidence.equivalent_set_complete = true;
  frame.evidence.credible_alternative_set_complete = true;
  frame.evidence.retained_certified_assignment_count = 1;
  frame.evidence.search_equivalent_assignments_exported = 1;
  frame.evidence.equivalent_best_assignments_within_domain = 1;
  frame.evidence.feasible_assignments_within_domain = 1;
  frame.evidence.soft_score_valid_assignments_within_domain = 1;
  frame.evidence.search_domain_assignments_exported = 1;
  frame.search_domain.shell_radius = 2;
  frame.search_domain.audit_known = true;
  frame.search_domain.evidence_complete = true;
  frame.search_domain.bounded_domain_only = true;
  frame.search_domain.completeness =
      HoloLiftDomainCompleteness::BoundedExhaustive;
  store.frames.push_back(frame);
  store.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest, 1, 1, 0, 0};
  return store;
}

int main(int argc, char **argv) {
  expect(argc == 2 || argc == 3, "binary round-trip path was not provided");
  if (argc == 3) {
    auto legacy = read_hololift_observation_store_binary(argv[2]);
    if (!legacy) std::cerr << legacy.error() << '\n';
    expect(legacy.has_value(), "version 9 artifact was not readable");
    for (const auto &candidate : legacy->candidates())
      expect(!candidate.carried &&
                 candidate.policy_admission == HoloLiftPolicyAdmission::None &&
                 candidate.provider_evaluation == HoloLiftProviderEvaluation::Recorded &&
                 candidate.policy_source_frame_index == HOLOLIFT_NO_INDEX &&
                 candidate.policy_source_assignment_identity.empty(),
             "legacy binary unexpectedly acquired policy admission");
    std::cout << "PASS hololift_legacy_v9_read\n";
  }
  titan_hash::Sha256Builder sha256;
  sha256.update("abc");
  const auto sha128 = sha256.finish128();
  expect(sha128[0] == 0x414140de5dae2223ULL &&
             sha128[1] == 0xba7816bf8f01cfeaULL,
         "SHA-256 known vector failed");
  titan_hash::Sha256Builder sha256_multiblock;
  sha256_multiblock.update("abcdbcdecdefdefgefghfghighij");
  sha256_multiblock.update("hijkijkljklmklmnlmnomnopnopq");
  const auto sha128_multiblock = sha256_multiblock.finish128();
  expect(sha128_multiblock[0] == 0xe5c026930c3e6039ULL &&
             sha128_multiblock[1] == 0x248d6a61d20638b8ULL,
         "incremental multi-block SHA-256 known vector failed");
  static_assert(
      !std::is_copy_constructible_v<HoloLiftObservationStore>,
      "finalized HoloLift store must be immutable/move-only");
  const auto strong =
      parse_hololift_vibe_observation_class("strong_bounded_anchor");
  expect(strong.has_value() &&
             *strong == HoloLiftObservationClass::StrongBoundedAnchor &&
             hololift_vibe_observation_class_name(*strong) ==
                 "strong_bounded_anchor",
         "VIBE observation class mapping failed");
  expect(!parse_hololift_vibe_observation_class("confident").has_value(),
         "unknown VIBE observation class was accepted");

  HoloLiftIdentityRegistry identity_registry;
  const std::array<HoloLiftSourceAtomKey, 2> membership_a{
      HoloLiftSourceAtomKey{0, 1}, HoloLiftSourceAtomKey{0, 2}};
  const std::array<HoloLiftSourceAtomKey, 2> membership_b{
      HoloLiftSourceAtomKey{0, 1}, HoloLiftSourceAtomKey{0, 3}};
  expect(identity_registry.register_component(101, membership_a)
             .has_value() &&
             identity_registry.register_component(101, membership_a)
                 .has_value() &&
             !identity_registry.register_component(101, membership_b)
                  .has_value(),
         "component-id collision guard did not compare exact membership");
  const std::array<std::uint64_t, 2> layout_a{101, 202};
  const std::array<std::uint64_t, 2> layout_b{101, 303};
  expect(identity_registry.register_layout(404, layout_a).has_value() &&
             identity_registry.register_layout(404, layout_a)
                 .has_value() &&
             !identity_registry.register_layout(404, layout_b)
                  .has_value(),
         "layout-id collision guard did not compare exact components");

  titan_pbctopo::PbctopoFrameReport source_report;
  source_report.status =
      titan_pbctopo::PbctopoFrameStatus::Certified;
  source_report.certificate_scope =
      titan_pbctopo::PbctopoCertificateScope::
          ImageIntercomponentSteric;
  source_report.certificate_graph_source =
      titan_pbctopo::PbctopoCertificateGraphSource::
          GromacsTopology;
  source_report.hard_graph_source =
      titan_pbctopo::PbctopoHardGraphSource::ExplicitTopology;
  source_report.search_evidence_source =
      titan_pbctopo::PbctopoSearchEvidenceSource::ScoredContacts;
  source_report.hard_feasible = true;
  source_report.hard_feasibility.evaluated = true;
  source_report.hard_feasibility.assignment_available = true;
  source_report.hard_feasibility.steric_audit_available = true;
  source_report.hard_feasibility.steric_decision_complete = true;
  source_report.hard_feasibility.steric_pair_enumeration_complete = true;
  source_report.hard_feasibility.steric_termination_reason = 0;
  source_report.hard_feasibility.steric_scan_complete = true;
  source_report.assignment_signature = 17;
  const auto mapped_provenance =
      hololift_vibe_v1_frame_provenance(source_report, 17);
  expect(mapped_provenance.has_value() &&
             mapped_provenance->source_frame_report_present &&
             mapped_provenance->status ==
                 HoloLiftFrameStatus::Certified &&
             mapped_provenance->hard_graph_source ==
                 HoloLiftHardGraphSource::ExplicitTopology &&
             mapped_provenance->framewise_assignment_signature == 17 &&
             mapped_provenance->output_assignment_signature == 17,
         "VIBE frame provenance adapter lost certificate fields");
  expect(!hololift_vibe_v1_frame_provenance(source_report, 18)
              .has_value(),
         "production framewise signature disagreement was accepted");
  source_report.status = titan_pbctopo::PbctopoFrameStatus::Contradicted;
  source_report.evidence_state =
      titan_pbctopo::PbctopoEvidenceState::Contradicted;
  source_report.evidence_consistency_evaluated = true;
  source_report.evidence_consistent = false;
  source_report.evidence_graph_connected = true;
  source_report.soft_observed_contact_pair_hash = {0x55, 0x66};
  source_report.soft_observed_lost_pair_hash = {0x77, 0x88};
  source_report.soft_observed_all_hypotheses_hash = {0x91, 0x92};
  source_report.soft_observed_selected_hypothesis_hash = {0x93, 0x94};
  source_report.soft_observed_selected_compatible_pair_hash = {0x95, 0x96};
  source_report.soft_observed_hypothesis_count = 1;
  source_report.soft_observed_component_relation_count = 1;
  source_report.soft_observed_compatible_hypothesis_count = 1;
  source_report.soft_observed_selected_compatible_contact_count = 2;
  source_report.soft_observed_selected_compatible_contact_loss_count = 1;
  source_report.soft_observed_selected_relation_has_support = true;
  source_report.soft_observed_hypothesis_construction_attempted = true;
  source_report.soft_observed_hypothesis_construction_complete = true;
  source_report.soft_observed_contacts_checked = 2;
  source_report.soft_observed_contacts_lost = 1;
  source_report.soft_observed_contact_cutoff_A = 4.5;
  source_report.soft_observed_contact_construction_attempted = true;
  source_report.soft_observed_contact_construction_complete = true;
  const auto contradicted_provenance =
      hololift_vibe_v1_frame_provenance(source_report, 17);
  expect(contradicted_provenance.has_value() &&
             contradicted_provenance->status ==
                 HoloLiftFrameStatus::Contradicted &&
             contradicted_provenance->framewise_assignment_signature == 17 &&
             contradicted_provenance->output_assignment_signature == 17 &&
             contradicted_provenance->evidence_graph_connected &&
             contradicted_provenance->evidence_state ==
                 HoloLiftEvidenceState::Contradicted,
         "contradicted provenance lost diagnostic assignment identity");
  expect(!hololift_vibe_v1_frame_provenance(source_report, 18)
              .has_value(),
         "contradicted provenance bypassed assignment agreement");
  auto incomplete_evidence_report = source_report;
  incomplete_evidence_report.soft_observed_contact_construction_complete =
      false;
  incomplete_evidence_report.soft_observed_contact_mic_query_failures = 1;
  expect(!hololift_vibe_v1_frame_provenance(incomplete_evidence_report, 17)
              .has_value(),
         "incomplete E_obs construction entered the direct adapter");
  source_report.status = titan_pbctopo::PbctopoFrameStatus::Certified;
  source_report.evidence_state =
      titan_pbctopo::PbctopoEvidenceState::SupportedConnected;
  source_report.evidence_consistency_evaluated = true;
  source_report.evidence_consistent = true;
  source_report.soft_observed_contacts_lost = 0;
  source_report.temporal_inference_experimental = true;
  source_report.temporal_selected = true;
  source_report.temporal_changed_from_greedy = true;
  source_report.assignment_signature = 22;
  source_report.temporal_assignment_signature = 22;
  const auto diagnostic_provenance =
      hololift_vibe_v1_frame_provenance(source_report, 17);
  expect(diagnostic_provenance.has_value() &&
             diagnostic_provenance->framewise_assignment_signature == 17 &&
             diagnostic_provenance->output_assignment_signature == 22 &&
             diagnostic_provenance->temporal_assignment_signature == 22 &&
             diagnostic_provenance->temporal_changed_from_framewise,
         "Viterbi replay overwrote framewise provenance");

  const HoloLiftObservationStoreBuilder valid = make_valid_builder();
  const auto validation =
      validate_hololift_observation_store(valid);
  expect(validation.has_value(), "valid Phase 0 store was rejected");
  auto no_contact = valid;
  auto &no_contact_frame = no_contact.frames[0];
  auto &no_contact_candidate = no_contact.candidates[0];
  no_contact_frame.provenance.status = HoloLiftFrameStatus::WeakObservation;
  no_contact_frame.provenance.evidence_state =
      HoloLiftEvidenceState::WeakDisconnected;
  no_contact_frame.provenance.evidence_graph_connected = false;
  no_contact_frame.provenance.soft_observed_selected_hypothesis_hash = {};
  no_contact_frame.provenance.soft_observed_selected_compatible_pair_hash = {};
  no_contact_frame.provenance.soft_observed_hypothesis_count = 0;
  no_contact_frame.provenance.soft_observed_component_relation_count = 0;
  no_contact_frame.provenance.soft_observed_compatible_hypothesis_count = 0;
  no_contact_frame.provenance.soft_observed_selected_compatible_contact_count = 0;
  no_contact_frame.provenance.soft_observed_selected_relation_has_support = false;
  no_contact_frame.provenance.soft_observed_contacts_checked = 0;
  no_contact_frame.evidence.identified = false;
  no_contact_frame.evidence.objective_unique_within_search_domain = false;
  no_contact_frame.evidence.evidence_unique_within_search_domain = false;
  no_contact_candidate.observation_class = HoloLiftObservationClass::WeakObservation;
  no_contact_candidate.evidence_compatible_hypothesis_hash = {};
  no_contact_candidate.evidence_compatible_pair_hash = {};
  no_contact_candidate.evidence_compatible_hypothesis_count = 0;
  no_contact_candidate.evidence_supported_relation_count = 0;
  no_contact_candidate.evidence_compatible_contact_count = 0;
  no_contact_candidate.evidence_no_support_relation_count = 0;
  const auto no_contact_validation =
      validate_hololift_observation_store(no_contact);
  if (!no_contact_validation) {
    std::cerr << "zero-relation validation detail: "
              << no_contact_validation.error().message << "\n";
  }
  expect(no_contact_validation.has_value(),
         "zero-relation no-contact candidate was rejected");
  auto stale_no_contact = no_contact;
  stale_no_contact.frames[0].evidence.objective_unique_within_search_domain =
      true;
  stale_no_contact.frames[0].evidence.evidence_unique_within_search_domain =
      true;
  expect(!validate_hololift_observation_store(stale_no_contact).has_value(),
         "stale selected-evidence uniqueness row was accepted");

  auto weak_supported = valid;
  auto &weak_supported_frame = weak_supported.frames[0];
  auto &weak_supported_candidate = weak_supported.candidates[0];
  weak_supported_frame.provenance.status = HoloLiftFrameStatus::WeakObservation;
  weak_supported_candidate.observation_class =
      HoloLiftObservationClass::AmbiguousAnchor;
  const auto weak_supported_validation =
      validate_hololift_observation_store(weak_supported);
  if (!weak_supported_validation) {
    std::cerr << "weak-supported validation detail: "
              << weak_supported_validation.error().message << "\n";
  }
  expect(weak_supported_validation.has_value(),
         "weak-observation supported-connected candidate was rejected");
  auto weak_supported_strong = weak_supported;
  weak_supported_strong.candidates[0].observation_class =
      HoloLiftObservationClass::StrongBoundedAnchor;
  expect(!validate_hololift_observation_store(weak_supported_strong).has_value(),
         "weak-observation supported-connected candidate became a strong anchor");

  auto mixed_support = valid;
  auto &mixed_frame = mixed_support.frames[0];
  auto &mixed_candidate = mixed_support.candidates[0];
  mixed_frame.provenance.status = HoloLiftFrameStatus::WeakObservation;
  mixed_frame.provenance.evidence_state = HoloLiftEvidenceState::WeakAmbiguous;
  mixed_frame.provenance.soft_observed_hypothesis_count = 2;
  mixed_frame.provenance.soft_observed_component_relation_count = 2;
  mixed_frame.provenance.soft_observed_alternative_hypothesis_contact_count = 1;
  mixed_frame.provenance.soft_observed_no_support_relation_count = 1;
  mixed_frame.provenance.soft_observed_contacts_checked = 2;
  mixed_candidate.observation_class = HoloLiftObservationClass::AmbiguousAnchor;
  mixed_candidate.evidence_no_support_relation_count = 1;
  const auto mixed_support_validation =
      validate_hololift_observation_store(mixed_support);
  if (!mixed_support_validation) {
    std::cerr << "mixed-support validation detail: "
              << mixed_support_validation.error().message << "\n";
  }
  expect(mixed_support_validation.has_value(),
         "mixed supported/H0 candidate was rejected");
  auto tampered_candidate_evidence = valid;
  tampered_candidate_evidence.candidates[0]
      .evidence_compatible_hypothesis_hash = {0xdead, 0xbeef};
  expect(!validate_hololift_observation_store(tampered_candidate_evidence)
              .has_value(),
         "candidate evidence hash was not bound to frame provenance");
  const std::array<HoloLiftSourceAtomKey, 4> bound_atom_order{
      HoloLiftSourceAtomKey{0, 12}, HoloLiftSourceAtomKey{0, 10},
      HoloLiftSourceAtomKey{1, 21}, HoloLiftSourceAtomKey{1, 20}};
  const std::array<std::array<double, 3>, 4> bound_coordinates{
      std::array<double, 3>{1.0, 1.0, 1.0},
      std::array<double, 3>{2.0, 1.0, 1.0},
      std::array<double, 3>{5.0, 5.0, 5.0},
      std::array<double, 3>{6.0, 5.0, 5.0}};
  expect(validate_hololift_frame_binding(
             valid.topology_epochs[0], valid.frames[0], 7,
             valid.frames[0].box_matrix,
             bound_atom_order, bound_coordinates)
             .has_value(),
         "matching external trajectory frame was rejected");
  HoloLiftFrameRecord tie_frame = valid.frames[0];
  tie_frame.spatial_lift_policy_version =
      titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
  tie_frame.spatial_lift_valid = true;
  const std::array<HoloLiftSourceAtomKey, 2> tie_atom_order{
      HoloLiftSourceAtomKey{0, 2}, HoloLiftSourceAtomKey{0, 1}};
  const std::array<std::array<double, 3>, 2> tie_coordinates{
      std::array<double, 3>{5.0, 0.0, 0.0},
      std::array<double, 3>{0.0, 0.0, 0.0}};
  const std::array<HoloLiftHardEdgeRecord, 1> tie_edges{
      HoloLiftHardEdgeRecord{0, 1, HoloLiftHardEdgeKind::TopologyBond}};
  HoloLiftTopologyEpochRecord tie_epoch = valid.topology_epochs[0];
  const auto tie_universe_hash =
      hash_hololift_canonical_atom_universe(tie_atom_order);
  expect(tie_universe_hash.has_value(),
         "tie fixture atom-universe hash was not built");
  tie_epoch.canonical_atom_universe_hash = *tie_universe_hash;
  tie_epoch.atom_count = tie_atom_order.size();
  tie_epoch.hard_edges.count = tie_edges.size();
  tie_epoch.hard_graph_cycle_rank = 0;
  tie_frame.canonical_atom_universe_hash = *tie_universe_hash;
  const auto tie_lift = replay_hololift_spatial_lift(
      tie_epoch, tie_frame, tie_edges, tie_atom_order, tie_coordinates);
  expect(tie_lift.has_value() && tie_lift->ambiguous_hard_edges == 1,
         "half-cell MIC ambiguity was not recorded");
  tie_frame.spatial_lift_identity = tie_lift->identity;
  tie_frame.spatial_lift_ambiguous_hard_edges =
      tie_lift->ambiguous_hard_edges;
  tie_frame.spatial_lift_cycle_residuals = tie_lift->cycle_residuals;
  const std::array<HoloLiftSourceAtomKey, 2> reordered_tie_atoms{
      tie_atom_order[1], tie_atom_order[0]};
  const std::array<std::array<double, 3>, 2> reordered_tie_coordinates{
      tie_coordinates[1], tie_coordinates[0]};
  const auto reordered_tie_lift = replay_hololift_spatial_lift(
      tie_epoch, tie_frame, tie_edges, reordered_tie_atoms,
      reordered_tie_coordinates);
  expect(reordered_tie_lift.has_value() &&
             reordered_tie_lift->identity == tie_lift->identity &&
             reordered_tie_lift->atom_images.size() == 2 &&
             reordered_tie_lift->atom_images[0] == tie_lift->atom_images[1] &&
             reordered_tie_lift->atom_images[1] == tie_lift->atom_images[0],
         "canonical spatial lift did not remap images to trajectory order");
  expect(validate_hololift_spatial_lift_replay(
             tie_epoch, tie_frame, tie_edges, reordered_tie_atoms,
             reordered_tie_coordinates)
             .has_value(),
         "canonical spatial lift changed when atom input order changed");
  std::vector<titan_pbctopo::PbctopoAtom> producer_atoms(
      bound_coordinates.size());
  for (std::size_t idx = 0; idx < producer_atoms.size(); ++idx) {
    producer_atoms[idx].owner = bound_atom_order[idx].owner;
    producer_atoms[idx].source_atom_id =
        bound_atom_order[idx].source_atom_id;
    producer_atoms[idx].x = bound_coordinates[idx][0];
    producer_atoms[idx].y = bound_coordinates[idx][1];
    producer_atoms[idx].z = bound_coordinates[idx][2];
  }
  box producer_box(10.0, 10.0, 10.0, 90.0, 90.0, 90.0);
  const auto producer_binding =
      titan_pbctopo::build_pbctopo_frame_binding_observation(
          7, producer_box, producer_atoms);
  expect(producer_binding.valid &&
             HoloLiftHash128{producer_binding.box_hash.lo,
                             producer_binding.box_hash.hi} ==
                 valid.frames[0].box_hash &&
              HoloLiftHash128{producer_binding.atom_selection_hash.lo,
                              producer_binding.atom_selection_hash.hi} ==
                  valid.frames[0].atom_selection_hash &&
              HoloLiftHash128{
                  producer_binding.canonical_atom_universe_hash.lo,
                  producer_binding.canonical_atom_universe_hash.hi} ==
                  valid.frames[0].canonical_atom_universe_hash &&
              HoloLiftHash128{producer_binding.wrapped_coordinate_hash.lo,
                             producer_binding.wrapped_coordinate_hash.hi} ==
                 valid.frames[0].wrapped_coordinate_hash,
         "VIBE producer and HoloLift consumer frame hashes diverged");
  auto wrong_coordinates = bound_coordinates;
  wrong_coordinates[0][0] += 0.125;
  expect(!validate_hololift_frame_binding(
               valid.topology_epochs[0], valid.frames[0], 7,
               valid.frames[0].box_matrix,
               bound_atom_order, wrong_coordinates)
              .has_value(),
         "different wrapped trajectory coordinates matched the observation");
  auto wrong_atom_order = bound_atom_order;
  std::swap(wrong_atom_order[0], wrong_atom_order[1]);
  expect(!validate_hololift_frame_binding(
               valid.topology_epochs[0], valid.frames[0], 7,
               valid.frames[0].box_matrix,
               wrong_atom_order, bound_coordinates)
              .has_value(),
         "different trajectory atom order matched the observation");
  expect(!validate_hololift_frame_binding(
               valid.topology_epochs[0], valid.frames[0], 8,
               valid.frames[0].box_matrix,
               bound_atom_order, bound_coordinates)
              .has_value(),
          "different trajectory frame index matched the observation");
  auto wrong_universe_order = bound_atom_order;
  wrong_universe_order[0].source_atom_id = 999;
  HoloLiftFrameRecord rebound_wrong_universe = valid.frames[0];
  rebound_wrong_universe.atom_selection_hash =
      hash_hololift_atom_selection(wrong_universe_order);
  expect(!validate_hololift_frame_binding(
               valid.topology_epochs[0], rebound_wrong_universe, 7,
               rebound_wrong_universe.box_matrix, wrong_universe_order,
               bound_coordinates)
              .has_value(),
          "rebound frame with a different atom universe was accepted");
  expect(hololift_phase1_uses_temporal_observation(valid.frames[0]),
         "unique spatial lift was not admitted to Phase 1");
  HoloLiftFrameRecord phase1_ambiguous = valid.frames[0];
  phase1_ambiguous.spatial_lift_ambiguous_hard_edges = 1;
  expect(hololift_phase1_frame_use(phase1_ambiguous) ==
             HoloLiftPhase1FrameUse::ObservationGapAmbiguousInternalLift,
         "ambiguous internal lift was not represented as a Phase 1 gap");
  HoloLiftFrameRecord phase1_cycle = valid.frames[0];
  phase1_cycle.spatial_lift_cycle_residuals = 1;
  expect(hololift_phase1_frame_use(phase1_cycle) ==
             HoloLiftPhase1FrameUse::ObservationGapCycleResidual,
         "cycle residual was not represented as a Phase 1 gap");
  auto make_contradicted_append = []() {
    auto builder = make_valid_builder();
    HoloLiftVibeFrameAppendInput input;
    input.frame = builder.frames.front();
    input.frame.frame = 1;
    input.frame.time_ps = 1.0;
    input.frame.source_sequence_index = 1;
    input.frame.trajectory_frame_index = 1;
    input.frame.source_relation =
        HoloLiftSourceFrameRelation::ContiguousSourceFrame;
    input.frame.candidates = {0, 0};
    input.frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
    input.frame.candidate_set_scope = HoloLiftCandidateSetScope::Unknown;
    input.frame.provenance.status = HoloLiftFrameStatus::Contradicted;
    input.frame.provenance.evidence_state =
        HoloLiftEvidenceState::Contradicted;
    input.frame.provenance.evidence_consistency_evaluated = true;
    input.frame.provenance.evidence_consistent = false;
    input.frame.provenance.evidence_graph_connected = true;
    input.frame.provenance.soft_observed_contact_pair_hash = {0x55, 0x66};
    input.frame.provenance.soft_observed_lost_pair_hash = {0x77, 0x88};
    input.frame.provenance.soft_observed_all_hypotheses_hash = {0x91, 0x92};
    input.frame.provenance.soft_observed_selected_hypothesis_hash = {0x93, 0x94};
    input.frame.provenance.soft_observed_selected_compatible_pair_hash = {0x95, 0x96};
    input.frame.provenance.soft_observed_hypothesis_count = 1;
    input.frame.provenance.soft_observed_component_relation_count = 1;
    input.frame.provenance.soft_observed_compatible_hypothesis_count = 1;
    input.frame.provenance.soft_observed_selected_compatible_contact_count = 2;
    input.frame.provenance.soft_observed_selected_compatible_contact_loss_count = 1;
    input.frame.provenance.soft_observed_selected_relation_has_support = true;
    input.frame.provenance.soft_observed_hypothesis_construction_attempted = true;
    input.frame.provenance.soft_observed_hypothesis_construction_complete = true;
    input.frame.provenance.soft_observed_contact_construction_attempted = true;
    input.frame.provenance.soft_observed_contact_construction_complete = true;
    input.frame.provenance.soft_observed_contacts_checked = 2;
    input.frame.provenance.soft_observed_contacts_lost = 1;
    input.frame.provenance.framewise_assignment_identity = {};
    input.frame.provenance.output_assignment_identity = {};
    input.frame.evidence.retained_certified_assignment_count = 0;
    input.frame.evidence.search_equivalent_assignments_exported = 0;
    input.frame.evidence.search_domain_assignments_exported = 0;
    input.frame.evidence.equivalent_set_complete = false;
    input.frame.evidence.credible_alternative_set_complete = false;
    return std::pair{std::move(builder), std::move(input)};
  };
  auto [contradicted_builder, contradicted_input] = make_contradicted_append();
  const auto contradicted_append = append_hololift_vibe_frame(
      contradicted_builder, std::move(contradicted_input));
  if (!contradicted_append)
    std::cerr << "contradicted append error: " << contradicted_append.error() << '\n';
  expect(contradicted_append.has_value(),
         "direct adapter rejected a valid contradicted gap");
  auto [malformed_builder, malformed_input] = make_contradicted_append();
  malformed_input.frame.provenance.evidence_consistent = true;
  expect(!append_hololift_vibe_frame(malformed_builder,
                                     std::move(malformed_input)),
         "direct adapter accepted malformed contradicted evidence");
  auto [bad_cutoff_builder, bad_cutoff_input] = make_contradicted_append();
  bad_cutoff_input.frame.provenance.soft_observed_contact_cutoff_A =
      std::numeric_limits<double>::quiet_NaN();
  expect(!append_hololift_vibe_frame(bad_cutoff_builder,
                                     std::move(bad_cutoff_input)),
         "direct adapter accepted a non-finite evaluated-evidence cutoff");
  auto finalized =
      finalize_hololift_observation_store(make_valid_builder());
  expect(finalized.has_value() &&
             finalized->frames().size() == 1 &&
             finalized->candidates().size() == 1 &&
             finalized->component_images().size() == 2,
         "validated Phase 0 store was not finalized immutably");
  const std::filesystem::path binary_path = argv[1];
  const std::filesystem::path roundtrip_path =
      binary_path.string() + ".roundtrip";
  const auto initial_write =
      write_hololift_observation_store_binary(*finalized, binary_path);
  expect(initial_write.has_value() &&
             *initial_write == HoloLiftArtifactCommitState::
                                   CommittedDurabilityConfirmed,
         "HoloLift binary artifact was not written");
  auto restored = read_hololift_observation_store_binary(binary_path);
  expect(restored.has_value() &&
             restored->frames().size() == finalized->frames().size() &&
             restored->topology_epochs()[0].topology_epoch_identity ==
                 finalized->topology_epochs()[0].topology_epoch_identity &&
             restored->candidates()[0].assignment_identity ==
                 finalized->candidates()[0].assignment_identity &&
             restored->exact_atom_masses().size() ==
                 finalized->exact_atom_masses().size() &&
             std::equal(restored->exact_atom_masses().begin(),
                        restored->exact_atom_masses().end(),
                        finalized->exact_atom_masses().begin()) &&
             restored->source_coverage() == finalized->source_coverage() &&
             restored->source_coverage().complete(),
         "HoloLift binary artifact did not round-trip");
  HoloLiftBinaryReadLimits restrictive_limits;
  restrictive_limits.max_frames = 0;
  expect(!read_hololift_observation_store_binary(
              binary_path, restrictive_limits)
              .has_value(),
         "binary table limit did not reject before allocation");
  expect(write_hololift_observation_store_binary(*restored, roundtrip_path)
             .has_value(),
         "round-tripped HoloLift binary artifact was not written");
  const auto read_bytes = [](const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), {});
  };
  const auto original_bytes = read_bytes(binary_path);
  const auto roundtrip_bytes = read_bytes(roundtrip_path);
  expect(original_bytes == roundtrip_bytes,
         "HoloLift binary round-trip was not byte-exact");

  auto replacement_builder = make_valid_builder();
  replacement_builder.source_contract.package_id =
      "hololift-durable-replacement";
  auto replacement_store =
      finalize_hololift_observation_store(std::move(replacement_builder));
  expect(replacement_store.has_value(),
         "durable replacement fixture was not finalized");
  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::BeforeCommit);
  const auto precommit_failure = write_hololift_observation_store_binary(
      *replacement_store, binary_path);
  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::None);
  expect(!precommit_failure.has_value() &&
             precommit_failure.error().commit_state ==
                 HoloLiftArtifactCommitState::NotCommitted &&
             !precommit_failure.error().committed() &&
             read_bytes(binary_path) == original_bytes,
         "pre-commit failure changed the final Phase 0 artifact");

  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::
          AfterCommitBeforeDurabilityConfirmation);
  const auto postcommit_failure = write_hololift_observation_store_binary(
      *replacement_store, binary_path);
  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::None);
  const auto postcommit_store =
      read_hololift_observation_store_binary(binary_path);
  expect(!postcommit_failure.has_value() &&
             postcommit_failure.error().commit_state ==
                 HoloLiftArtifactCommitState::
                     CommittedDurabilityUnconfirmed &&
             postcommit_failure.error().committed() &&
             postcommit_store.has_value() &&
             postcommit_store->source_contract().package_id ==
                 "hololift-durable-replacement",
         "post-commit durability failure lost or misclassified Phase 0");

  const auto confirmed_restore =
      write_hololift_observation_store_binary(*finalized, binary_path);
  expect(confirmed_restore.has_value() &&
             *confirmed_restore == HoloLiftArtifactCommitState::
                                       CommittedDurabilityConfirmed &&
             read_bytes(binary_path) == original_bytes,
         "confirmed Phase 0 rewrite did not restore the original artifact");
  auto inconsistent_coverage = make_valid_builder();
  inconsistent_coverage.source_coverage.source_frame_count = 2;
  const auto inconsistent_coverage_error =
      validate_hololift_observation_store(inconsistent_coverage);
  expect(!inconsistent_coverage_error.has_value() &&
             inconsistent_coverage_error.error().code ==
                 HoloLiftValidationCode::InvalidSourceContract,
         "inconsistent source-trajectory coverage was accepted");
  auto unknown_coverage_with_counts = make_valid_builder();
  unknown_coverage_with_counts.source_coverage.scope =
      HoloLiftSourceCoverageScope::Unknown;
  const auto unknown_coverage_error =
      validate_hololift_observation_store(unknown_coverage_with_counts);
  expect(!unknown_coverage_error.has_value() &&
             unknown_coverage_error.error().code ==
                 HoloLiftValidationCode::InvalidSourceContract,
         "unknown source coverage retained nonzero counts");
  auto large_builder = make_valid_builder();
  large_builder.source_contract.package_id = std::string(70000, 'x');
  auto large_store =
      finalize_hololift_observation_store(std::move(large_builder));
  const std::filesystem::path large_path = binary_path.string() + ".large";
  expect(large_store.has_value() &&
             write_hololift_observation_store_binary(*large_store, large_path)
                 .has_value(),
         "large HoloLift streaming fixture was not written");
  auto restored_large = read_hololift_observation_store_binary(large_path);
  expect(restored_large.has_value() &&
             restored_large->source_contract().package_id.size() == 70000,
         "multi-buffer HoloLift payload did not stream correctly");
  HoloLiftBinaryReadLimits small_string_limit;
  small_string_limit.max_string_bytes = 1024;
  expect(!read_hololift_observation_store_binary(
              large_path, small_string_limit)
              .has_value(),
         "binary string limit did not reject the large fixture");
  auto tampered_bytes = original_bytes;
  tampered_bytes.back() = static_cast<char>(tampered_bytes.back() ^ 1);
  {
    std::ofstream output(binary_path, std::ios::binary | std::ios::trunc);
    output.write(tampered_bytes.data(),
                 static_cast<std::streamsize>(tampered_bytes.size()));
  }
  expect(!read_hololift_observation_store_binary(binary_path).has_value(),
         "tampered HoloLift binary checksum was accepted");
  auto truncated_bytes = original_bytes;
  truncated_bytes.pop_back();
  {
    std::ofstream output(binary_path, std::ios::binary | std::ios::trunc);
    output.write(truncated_bytes.data(),
                 static_cast<std::streamsize>(truncated_bytes.size()));
  }
  expect(!read_hololift_observation_store_binary(binary_path).has_value(),
         "truncated HoloLift binary artifact was accepted");

  auto transaction_builder = make_valid_builder();
  transaction_builder.frames.clear();
  transaction_builder.candidates.clear();
  transaction_builder.component_images.clear();
  HoloLiftVibeFrameAppendInput transaction;
  transaction.frame = valid.frames.front();
  transaction.frame.candidates = {};
  transaction.frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
  HoloLiftVibeCandidateAppendInput transaction_candidate;
  transaction_candidate.record = valid.candidates.front();
  transaction_candidate.record.component_images = {};
  for (std::size_t idx = 0; idx < valid.components.size(); ++idx) {
    transaction_candidate.raw_component_images.push_back(
        {valid.components[idx].component_id, valid.component_images[idx]});
  }
  transaction.candidates.push_back(transaction_candidate);
  auto tampered_transaction = transaction;
  tampered_transaction.candidates.front().record.assignment_signature ^= 1;
  const auto rejected_transaction = append_hololift_vibe_frame(
      transaction_builder, std::move(tampered_transaction));
  expect(!rejected_transaction.has_value() &&
             transaction_builder.frames.empty() &&
             transaction_builder.candidates.empty() &&
             transaction_builder.component_images.empty(),
         "failed VIBE frame transaction partially mutated the store");
  const auto committed_transaction =
      append_hololift_vibe_frame(transaction_builder, std::move(transaction));
  expect(committed_transaction.has_value() && *committed_transaction == 0 &&
             validate_hololift_observation_store(transaction_builder)
                 .has_value(),
         "valid VIBE frame transaction was not committed atomically");

  auto bad_signature = valid;
  bad_signature.candidates[0].assignment_signature ^= 1;
  const auto signature_error =
      validate_hololift_observation_store(bad_signature);
  expect(!signature_error.has_value() &&
             signature_error.error().code ==
                 HoloLiftValidationCode::AssignmentIdentityMismatch,
         "assignment identity tamper was not rejected");
  auto bad_identity128 = valid;
  bad_identity128.candidates[0].assignment_identity.lo ^= 1;
  const auto identity128_error =
      validate_hololift_observation_store(bad_identity128);
  expect(!identity128_error.has_value() &&
             identity128_error.error().code ==
                 HoloLiftValidationCode::AssignmentIdentityMismatch,
         "128-bit assignment identity tamper was not rejected");

  auto boundary_strong = valid;
  boundary_strong.candidates[0].domain_evidence.touches_shell_boundary =
      true;
  const auto boundary_error =
      validate_hololift_observation_store(boundary_strong);
  expect(!boundary_error.has_value() &&
             boundary_error.error().code ==
                 HoloLiftValidationCode::ObservationClassMismatch,
         "boundary candidate remained a strong anchor");

  auto ambiguous_spatial_strong = valid;
  ambiguous_spatial_strong.frames[0].spatial_lift_ambiguous_hard_edges = 1;
  const auto ambiguous_spatial_error =
      validate_hololift_observation_store(ambiguous_spatial_strong);
  expect(!ambiguous_spatial_error.has_value() &&
             ambiguous_spatial_error.error().code ==
                 HoloLiftValidationCode::ObservationClassMismatch,
          "ambiguous internal spatial lift remained a strong anchor");

  auto impossible_ambiguity = valid;
  impossible_ambiguity.frames[0].spatial_lift_ambiguous_hard_edges =
      valid.topology_epochs[0].hard_edges.count + 1;
  const auto impossible_ambiguity_error =
      validate_hololift_observation_store(impossible_ambiguity);
  expect(!impossible_ambiguity_error.has_value() &&
             impossible_ambiguity_error.error().code ==
                 HoloLiftValidationCode::InvalidSpatialLift,
         "ambiguity count above the hard-edge count was accepted");

  auto impossible_cycle_residual = valid;
  impossible_cycle_residual.frames[0].spatial_lift_cycle_residuals =
      valid.topology_epochs[0].hard_graph_cycle_rank + 1;
  const auto impossible_cycle_error =
      validate_hololift_observation_store(impossible_cycle_residual);
  expect(!impossible_cycle_error.has_value() &&
             impossible_cycle_error.error().code ==
                 HoloLiftValidationCode::InvalidSpatialLift,
         "cycle residual count above graph beta1 was accepted");

  auto false_global_anchor = valid;
  false_global_anchor.candidates[0].observation_class =
      HoloLiftObservationClass::StrongGlobalAnchor;
  const auto false_global_error =
      validate_hololift_observation_store(false_global_anchor);
  expect(!false_global_error.has_value() &&
             false_global_error.error().code ==
                 HoloLiftValidationCode::ObservationClassMismatch,
         "bounded search domain was promoted to a global anchor");

  auto unaudited_global_anchor = valid;
  unaudited_global_anchor.frames[0].search_domain.bounded_domain_only = false;
  unaudited_global_anchor.frames[0].search_domain.completeness =
      HoloLiftDomainCompleteness::Unknown;
  unaudited_global_anchor.candidates[0].observation_class =
      HoloLiftObservationClass::StrongGlobalAnchor;
  const auto unaudited_global_error =
      validate_hololift_observation_store(unaudited_global_anchor);
  expect(!unaudited_global_error.has_value() &&
             unaudited_global_error.error().code ==
                 HoloLiftValidationCode::ObservationClassMismatch,
         "unexhausted search domain was promoted to a global anchor");
  auto audited_global_anchor = unaudited_global_anchor;
  audited_global_anchor.frames[0].search_domain.expansion_attempted = true;
  audited_global_anchor.frames[0].search_domain.expansion_exhausted = true;
  expect(!validate_hololift_observation_store(audited_global_anchor)
              .has_value(),
         "resource-exhausted search was treated as a global proof");
  audited_global_anchor.frames[0].search_domain.completeness =
      HoloLiftDomainCompleteness::GlobalCertified;
  const auto unsupported_global_error =
      validate_hololift_observation_store(audited_global_anchor);
  expect(!unsupported_global_error.has_value() &&
             unsupported_global_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
          "global certification without outside-domain proof was accepted");

  auto changed_hard_graph = valid;
  changed_hard_graph.hard_edges[1] =
      {1, 2, HoloLiftHardEdgeKind::TopologyBond};
  const auto changed_epoch_id = hololift_topology_epoch_id(
      changed_hard_graph.source_contract.schema_version,
      changed_hard_graph.topology_epochs[0].hard_graph_source,
      changed_hard_graph.topology_epochs[0].layout_signature,
      std::span<const HoloLiftComponentRecord>(changed_hard_graph.components),
      std::span<const HoloLiftSourceAtomKey>(
          changed_hard_graph.exact_atom_memberships),
      std::span<const HoloLiftHardEdgeRecord>(changed_hard_graph.hard_edges));
  expect(changed_epoch_id.has_value() &&
             *changed_epoch_id != valid.topology_epochs[0].topology_epoch_id,
         "hard-edge graph change did not alter topology epoch identity");
  const auto changed_epoch_identity = hololift_topology_epoch_identity128(
      changed_hard_graph.source_contract.schema_version,
      changed_hard_graph.topology_epochs[0].hard_graph_source,
      changed_hard_graph.topology_epochs[0].layout_identity,
      std::span<const HoloLiftComponentRecord>(changed_hard_graph.components),
      std::span<const HoloLiftHardEdgeRecord>(changed_hard_graph.hard_edges));
  expect(changed_epoch_identity.has_value() &&
             *changed_epoch_identity !=
                 valid.topology_epochs[0].topology_epoch_identity,
         "hard-edge graph change did not alter 128-bit epoch identity");
  changed_hard_graph.topology_epochs[0].topology_epoch_id = *changed_epoch_id;
  changed_hard_graph.topology_epochs[0].topology_epoch_identity =
      *changed_epoch_identity;
  const auto cross_component_edge_error =
      validate_hololift_observation_store(changed_hard_graph);
  expect(!cross_component_edge_error.has_value() &&
             cross_component_edge_error.error().code ==
                 HoloLiftValidationCode::InvalidTopologyEpoch,
         "hard edge joining declared components was accepted");

  auto disconnected_component = valid;
  disconnected_component.hard_edges.erase(
      disconnected_component.hard_edges.begin());
  disconnected_component.topology_epochs[0].hard_edges.count =
      disconnected_component.hard_edges.size();
  const auto disconnected_epoch_id = hololift_topology_epoch_id(
      disconnected_component.source_contract.schema_version,
      disconnected_component.topology_epochs[0].hard_graph_source,
      disconnected_component.topology_epochs[0].layout_signature,
      std::span<const HoloLiftComponentRecord>(
          disconnected_component.components),
      std::span<const HoloLiftSourceAtomKey>(
          disconnected_component.exact_atom_memberships),
      std::span<const HoloLiftHardEdgeRecord>(
          disconnected_component.hard_edges));
  const auto disconnected_epoch_identity =
      hololift_topology_epoch_identity128(
          disconnected_component.source_contract.schema_version,
          disconnected_component.topology_epochs[0].hard_graph_source,
          disconnected_component.topology_epochs[0].layout_identity,
          std::span<const HoloLiftComponentRecord>(
              disconnected_component.components),
          std::span<const HoloLiftHardEdgeRecord>(
              disconnected_component.hard_edges));
  expect(disconnected_epoch_id.has_value() &&
             disconnected_epoch_identity.has_value(),
         "disconnected topology fixture identity was not built");
  disconnected_component.topology_epochs[0].topology_epoch_id =
      *disconnected_epoch_id;
  disconnected_component.topology_epochs[0].topology_epoch_identity =
      *disconnected_epoch_identity;
  const auto disconnected_component_error =
      validate_hololift_observation_store(disconnected_component);
  expect(!disconnected_component_error.has_value() &&
             disconnected_component_error.error().code ==
                 HoloLiftValidationCode::InvalidTopologyEpoch,
         "hard-graph-disconnected component was accepted");

  auto tampered_box = valid;
  tampered_box.frames[0].box_matrix[0] += 0.25;
  const auto binding_error =
      validate_hololift_observation_store(tampered_box);
  expect(!binding_error.has_value() &&
             binding_error.error().code ==
                 HoloLiftValidationCode::InvalidFrameBinding,
          "box/trajectory binding tamper was accepted");

  auto tampered_epoch_universe = valid;
  tampered_epoch_universe.topology_epochs[0]
      .canonical_atom_universe_hash.lo ^= 1;
  const auto epoch_universe_error =
      validate_hololift_observation_store(tampered_epoch_universe);
  expect(!epoch_universe_error.has_value() &&
             epoch_universe_error.error().code ==
                 HoloLiftValidationCode::InvalidTopologyEpoch,
         "tampered topology-epoch atom universe was accepted");

  auto tampered_frame_universe = valid;
  tampered_frame_universe.frames[0].canonical_atom_universe_hash.lo ^= 1;
  const auto frame_universe_error =
      validate_hololift_observation_store(tampered_frame_universe);
  expect(!frame_universe_error.has_value() &&
             frame_universe_error.error().code ==
                 HoloLiftValidationCode::InvalidFrameBinding,
         "frame atom universe differing from its epoch was accepted");

  auto missing_epoch = valid;
  ++missing_epoch.frames[0].topology_epoch_id;
  const auto epoch_error =
      validate_hololift_observation_store(missing_epoch);
  expect(!epoch_error.has_value() &&
             epoch_error.error().code ==
                 HoloLiftValidationCode::MissingTopologyEpoch,
         "unknown topology epoch was accepted");

  auto duplicate_epoch = valid;
  duplicate_epoch.topology_epochs.push_back(
      duplicate_epoch.topology_epochs.front());
  const auto duplicate_epoch_error =
      validate_hololift_observation_store(duplicate_epoch);
  expect(!duplicate_epoch_error.has_value() &&
             duplicate_epoch_error.error().code ==
                 HoloLiftValidationCode::DuplicateTopologyEpoch,
         "duplicate topology epoch id was accepted");

  auto noncanonical = valid;
  for (auto &entry : noncanonical.component_images) {
    ++entry.x;
    ++entry.y;
  }
  const auto gauge_error =
      validate_hololift_observation_store(noncanonical);
  expect(!gauge_error.has_value() &&
             gauge_error.error().code ==
                 HoloLiftValidationCode::AssignmentGaugeMismatch,
         "noncanonical global lattice gauge was accepted");

  auto unsupported_schema = valid;
  unsupported_schema.source_contract.schema_version = 99;
  const auto schema_error =
      validate_hololift_observation_store(unsupported_schema);
  expect(!schema_error.has_value() &&
             schema_error.error().code ==
                 HoloLiftValidationCode::InvalidSourceContract,
          "unsupported VIBE observation schema was accepted");

  auto bad_identity_hash_contract = valid;
  bad_identity_hash_contract.source_contract.identity_hash_algorithm =
      "sha256-256";
  expect(!validate_hololift_observation_store(bad_identity_hash_contract)
              .has_value(),
         "invalid identity hash contract was accepted");
  auto bad_binding_hash_contract = valid;
  bad_binding_hash_contract.source_contract.binding_hash_algorithm =
      "sha256-256";
  expect(!validate_hololift_observation_store(bad_binding_hash_contract)
              .has_value(),
         "invalid binding hash contract was accepted");
  auto bad_payload_hash_contract = valid;
  bad_payload_hash_contract.source_contract.payload_checksum_algorithm =
      "sha256-128";
  expect(!validate_hololift_observation_store(bad_payload_hash_contract)
              .has_value(),
         "invalid payload checksum contract was accepted");

  auto bad_selected_link = valid;
  ++bad_selected_link.frames[0]
        .provenance.framewise_assignment_signature;
  const auto selected_link_error =
      validate_hololift_observation_store(bad_selected_link);
  expect(!selected_link_error.has_value() &&
             selected_link_error.error().code ==
                 HoloLiftValidationCode::InvalidFrameProvenance,
         "framewise selected-signature mismatch was accepted");

  auto duplicate_assignment = valid;
  HoloLiftCandidateRecord duplicate_candidate =
      duplicate_assignment.candidates.front();
  duplicate_candidate.rank = 2;
  duplicate_candidate.evidence.selected_framewise = false;
  duplicate_candidate.component_images.begin =
      duplicate_assignment.component_images.size();
  duplicate_assignment.component_images.insert(
      duplicate_assignment.component_images.end(),
      valid.component_images.begin(), valid.component_images.end());
  duplicate_assignment.candidates.push_back(duplicate_candidate);
  duplicate_assignment.frames[0].candidates.count = 2;
  duplicate_assignment.frames[0]
      .evidence.retained_certified_assignment_count = 2;
  const auto duplicate_error =
      validate_hololift_observation_store(duplicate_assignment);
  expect(!duplicate_error.has_value() &&
             duplicate_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
         "duplicate assignment signature was accepted");

  auto experimental = valid;
  experimental.frames[0].provenance.temporal_policy =
      HoloLiftTemporalPolicyStatus::ExperimentalViterbiDiagnostic;
  experimental.frames[0].provenance.temporal_assignment_signature =
      experimental.candidates[0].assignment_signature;
  experimental.frames[0].provenance.output_assignment_signature =
      experimental.candidates[0].assignment_signature;
  experimental.frames[0].provenance.temporal_assignment_identity =
      experimental.candidates[0].assignment_identity;
  experimental.frames[0].provenance.output_assignment_identity =
      experimental.candidates[0].assignment_identity;
  experimental.frames[0].provenance.temporal_selected = true;
  const auto experimental_strong_error =
      validate_hololift_observation_store(experimental);
  expect(!experimental_strong_error.has_value() &&
             experimental_strong_error.error().code ==
                 HoloLiftValidationCode::ObservationClassMismatch,
         "experimental Viterbi input was mixed into strong anchors");
  experimental.candidates[0].observation_class =
      HoloLiftObservationClass::AmbiguousAnchor;
  expect(validate_hololift_observation_store(experimental).has_value(),
         "diagnostic Viterbi input was not retained as ambiguous");

  auto contradictory_objective = valid;
  contradictory_objective.frames[0]
      .evidence.equivalent_best_assignments_within_domain = 2;
  const auto contradictory_objective_error =
      validate_hololift_observation_store(contradictory_objective);
  expect(!contradictory_objective_error.has_value() &&
             contradictory_objective_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
         "objective_unique with two optima was accepted");

  auto missing_objective_count = valid;
  missing_objective_count.frames[0]
      .evidence.equivalent_best_assignments_within_domain = 0;
  missing_objective_count.frames[0].evidence.equivalent_set_complete = false;
  const auto missing_objective_count_error =
      validate_hololift_observation_store(missing_objective_count);
  expect(!missing_objective_count_error.has_value() &&
             missing_objective_count_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
         "known objective uniqueness without an optimum count was accepted");

  auto missing_evidence_audit = valid;
  missing_evidence_audit.frames[0]
      .evidence.feasible_assignment_set_enumerated = false;
  missing_evidence_audit.frames[0].evidence.feasible_set_semantics =
      HoloLiftFeasibleSetSemantics::NotEnumerated;
  missing_evidence_audit.frames[0]
      .evidence.feasible_assignments_within_domain = 0;
  missing_evidence_audit.frames[0]
      .evidence.credible_alternative_set_complete = false;
  const auto missing_evidence_audit_error =
      validate_hololift_observation_store(missing_evidence_audit);
  expect(!missing_evidence_audit_error.has_value() &&
             missing_evidence_audit_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
         "known evidence uniqueness without a feasible-set audit was accepted");

  auto unknown_audit_complete = valid;
  unknown_audit_complete.frames[0].search_domain.audit_known = false;
  const auto unknown_audit_error =
      validate_hololift_observation_store(unknown_audit_complete);
  expect(!unknown_audit_error.has_value() &&
             unknown_audit_error.error().code ==
                 HoloLiftValidationCode::InvalidCandidateSet,
         "complete search evidence without a known audit was accepted");

  auto certified_without_candidate = valid;
  certified_without_candidate.candidates.clear();
  certified_without_candidate.component_images.clear();
  certified_without_candidate.frames[0].candidates = {0, 0};
  certified_without_candidate.frames[0].selected_candidate_index =
      HOLOLIFT_NO_INDEX;
  certified_without_candidate.frames[0].evidence = {};
  certified_without_candidate.frames[0]
      .provenance.framewise_assignment_signature = 0;
  certified_without_candidate.frames[0]
      .provenance.output_assignment_signature = 0;
  certified_without_candidate.frames[0]
      .provenance.framewise_assignment_identity = {};
  certified_without_candidate.frames[0]
      .provenance.output_assignment_identity = {};
  const auto certified_without_candidate_error =
      validate_hololift_observation_store(certified_without_candidate);
  expect(!certified_without_candidate_error.has_value() &&
             certified_without_candidate_error.error().code ==
                 HoloLiftValidationCode::InvalidFrameProvenance,
         "certified frame without a candidate was accepted");

  auto missing_candidate = valid;
  missing_candidate.candidates.clear();
  missing_candidate.component_images.clear();
  missing_candidate.frames[0].candidates = {0, 0};
  missing_candidate.frames[0].selected_candidate_index =
      HOLOLIFT_NO_INDEX;
  missing_candidate.frames[0].evidence = {};
  missing_candidate.frames[0].provenance.status =
      HoloLiftFrameStatus::Fallback;
  missing_candidate.frames[0].provenance.certificate_scope =
      HoloLiftCertificateScope::None;
  missing_candidate.frames[0].provenance.certificate_graph_source =
      HoloLiftCertificateGraphSource::None;
  missing_candidate.frames[0].provenance.hard_graph_source =
      HoloLiftHardGraphSource::None;
  missing_candidate.frames[0].provenance.evidence_state =
      HoloLiftEvidenceState::NotEvaluated;
  missing_candidate.frames[0].provenance.evidence_consistency_evaluated = false;
  missing_candidate.frames[0].provenance.evidence_consistent = false;
  missing_candidate.frames[0].provenance.evidence_graph_connected = false;
  missing_candidate.frames[0].provenance.soft_observed_contact_pair_hash = {};
  missing_candidate.frames[0].provenance.soft_observed_lost_pair_hash = {};
  missing_candidate.frames[0].provenance.soft_observed_all_hypotheses_hash = {};
  missing_candidate.frames[0].provenance.soft_observed_selected_hypothesis_hash = {};
  missing_candidate.frames[0].provenance.soft_observed_selected_compatible_pair_hash = {};
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_component_relation_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_ambiguous_relation_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_compatible_hypothesis_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_selected_compatible_contact_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_alternative_hypothesis_contact_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_selected_compatible_contact_loss_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_no_support_relation_count = 0;
  missing_candidate.frames[0].provenance.soft_observed_selected_relation_has_support = false;
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_construction_attempted = false;
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_construction_complete = false;
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_mic_ambiguity_failures = 0;
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_component_mapping_failures = 0;
  missing_candidate.frames[0].provenance.soft_observed_hypothesis_internal_edges_ignored = 0;
  missing_candidate.frames[0].provenance.soft_observed_contacts_checked = 0;
  missing_candidate.frames[0].provenance.soft_observed_contacts_lost = 0;
  missing_candidate.frames[0].provenance.soft_observed_contact_cutoff_A =
      std::numeric_limits<double>::quiet_NaN();
  missing_candidate.frames[0].provenance.soft_observed_contact_construction_attempted = false;
  missing_candidate.frames[0].provenance.soft_observed_contact_construction_complete = false;
  missing_candidate.frames[0].provenance.soft_observed_contact_cell_setup_failures = 0;
  missing_candidate.frames[0].provenance.soft_observed_contact_atom_index_failures = 0;
  missing_candidate.frames[0].provenance.soft_observed_contact_mic_query_failures = 0;
  missing_candidate.frames[0]
      .provenance.framewise_assignment_signature = 0;
  missing_candidate.frames[0]
      .provenance.output_assignment_signature = 0;
  missing_candidate.frames[0].provenance.framewise_assignment_identity = {};
  missing_candidate.frames[0].provenance.output_assignment_identity = {};
  expect(validate_hololift_observation_store(missing_candidate)
             .has_value(),
         "missing candidate was incorrectly treated as hard infeasible");
  auto stale_fallback_evidence = missing_candidate;
  stale_fallback_evidence.frames[0]
      .provenance.soft_observed_contacts_checked = 1;
  expect(!validate_hololift_observation_store(stale_fallback_evidence)
              .has_value(),
         "fallback frame retained stale evidence counts");
  stale_fallback_evidence = missing_candidate;
  stale_fallback_evidence.frames[0]
      .provenance.soft_observed_contact_pair_hash = {0x11, 0x22};
  expect(!validate_hololift_observation_store(stale_fallback_evidence)
              .has_value(),
         "fallback frame retained a stale evidence hash");
  stale_fallback_evidence = missing_candidate;
  stale_fallback_evidence.frames[0]
      .provenance.soft_observed_contact_cutoff_A = 4.5;
  expect(!validate_hololift_observation_store(stale_fallback_evidence)
              .has_value(),
         "fallback frame retained a stale evidence cutoff");

  auto contradicted_gap = valid;
  contradicted_gap.candidates.clear();
  contradicted_gap.component_images.clear();
  auto &contradicted_frame = contradicted_gap.frames[0];
  contradicted_frame.candidates = {0, 0};
  contradicted_frame.selected_candidate_index = HOLOLIFT_NO_INDEX;
  contradicted_frame.candidate_set_scope = HoloLiftCandidateSetScope::Unknown;
  contradicted_frame.provenance.status = HoloLiftFrameStatus::Contradicted;
  contradicted_frame.provenance.evidence_state =
      HoloLiftEvidenceState::Contradicted;
  contradicted_frame.provenance.evidence_consistency_evaluated = true;
  contradicted_frame.provenance.evidence_consistent = false;
  contradicted_frame.provenance.evidence_graph_connected = true;
  contradicted_frame.provenance.soft_observed_contact_pair_hash = {0x55, 0x66};
  contradicted_frame.provenance.soft_observed_lost_pair_hash = {0x77, 0x88};
  contradicted_frame.provenance.soft_observed_all_hypotheses_hash = {0x91, 0x92};
  contradicted_frame.provenance.soft_observed_selected_hypothesis_hash = {0x93, 0x94};
  contradicted_frame.provenance.soft_observed_selected_compatible_pair_hash = {0x95, 0x96};
  contradicted_frame.provenance.soft_observed_hypothesis_count = 1;
  contradicted_frame.provenance.soft_observed_component_relation_count = 1;
  contradicted_frame.provenance.soft_observed_compatible_hypothesis_count = 1;
  contradicted_frame.provenance.soft_observed_selected_compatible_contact_count = 2;
  contradicted_frame.provenance.soft_observed_selected_compatible_contact_loss_count = 1;
  contradicted_frame.provenance.soft_observed_selected_relation_has_support = true;
  contradicted_frame.provenance.soft_observed_hypothesis_construction_attempted = true;
  contradicted_frame.provenance.soft_observed_hypothesis_construction_complete = true;
  contradicted_frame.provenance.soft_observed_contact_construction_attempted = true;
  contradicted_frame.provenance.soft_observed_contact_construction_complete = true;
  contradicted_frame.provenance.soft_observed_contacts_checked = 2;
  contradicted_frame.provenance.soft_observed_contacts_lost = 1;
  contradicted_frame.provenance.framewise_assignment_identity = {};
  contradicted_frame.provenance.output_assignment_identity = {};
  contradicted_frame.evidence.retained_certified_assignment_count = 0;
  contradicted_frame.evidence.search_equivalent_assignments_exported = 0;
  contradicted_frame.evidence.search_domain_assignments_exported = 0;
  contradicted_frame.evidence.equivalent_set_complete = false;
  contradicted_frame.evidence.credible_alternative_set_complete = false;
  expect(validate_hololift_observation_store(contradicted_gap).has_value(),
         "framewise contradicted diagnostic gap was rejected");
  auto temporal_contradicted_gap = contradicted_gap;
  auto &temporal_contradicted_frame = temporal_contradicted_gap.frames[0];
  temporal_contradicted_frame.provenance.temporal_policy =
      HoloLiftTemporalPolicyStatus::ExperimentalViterbiDiagnostic;
  temporal_contradicted_frame.provenance.temporal_selected = true;
  temporal_contradicted_frame.provenance.temporal_assignment_signature =
      temporal_contradicted_frame.provenance.output_assignment_signature;
  expect(validate_hololift_observation_store(temporal_contradicted_gap)
             .has_value(),
         "Viterbi contradicted diagnostic gap was rejected");

  const auto components = std::span<const HoloLiftComponentRecord>(
      valid.components);
  std::vector<HoloLiftComponentImage> overflow_images{
      {components[0].component_id,
       {std::numeric_limits<std::int64_t>::min(), 0, 0}},
      {components[1].component_id,
       {std::numeric_limits<std::int64_t>::max(), 0, 0}},
  };
  expect(!canonicalize_hololift_assignment(
              valid.frames[0].layout_signature, components,
              overflow_images)
              .has_value(),
         "lattice-image subtraction overflow was accepted");

  std::cout << "PASS hololift_phase0_smoke\n";
  return 0;
}
'@ | Set-Content -LiteralPath $src -Encoding ASCII

$compilerPath = (Get-Command $Compiler -ErrorAction Stop).Source
Push-Location $repo
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $compilerPath -std=c++23 -O2 -Wall -Wextra -Wpedantic -Wconversion `
        -DTITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS `
        -Wshadow @ExtraCompilerFlags -Isrc $src `
        src\hololift\identity.cpp `
        src\hololift\frame_binding.cpp `
        src\hololift\spatial_lift.cpp `
        src\hololift\serialization.cpp `
        src\hololift\validation.cpp `
        src\hololift\vibe_adapter.cpp `
        src\hololift\vibe_contract.cpp `
        src\pbctopo\image_assignment.cpp `
        src\pbctopo\observation_binding.cpp `
        src\pbctopo\observation_layout.cpp `
        src\pbctopo\observation_schema.cpp `
        src\pbctopo\spatial_lift.cpp `
        src\pbctopo\lattice_math.cpp `
        src\pbctopo\report.cpp -o $exe
    $compileExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($compileExitCode -ne 0) {
        throw 'Failed to compile HoloLift Phase 0 smoke'
    }
    if ($LegacyArtifact) {
        & $exe (Join-Path $WorkRoot 'roundtrip.hlift') $LegacyArtifact
    } else {
        & $exe (Join-Path $WorkRoot 'roundtrip.hlift')
    }
    if ($LASTEXITCODE -ne 0) {
        throw "HoloLift Phase 0 smoke failed: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

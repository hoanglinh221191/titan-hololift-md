param(
    [string]$Compiler = 'g++',
    [string]$WorkRoot = (Join-Path $PSScriptRoot 'tmp\hololift_phase1_smoke'),
    [string[]]$ExtraCompilerFlags = @()
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$src = Join-Path $WorkRoot 'hololift_phase1_smoke.cpp'
$exe = Join-Path $WorkRoot 'hololift_phase1_smoke.exe'

@'
#include "hololift/durable_artifact.h"
#include "hololift/frame_binding.h"
#include "hololift/identity.h"
#include "hololift/phase1.h"
#include "hololift/phase1_serialization.h"
#include "hololift/serialization.h"
#include "hololift/spatial_lift.h"
#include "hololift/validation.h"
#include "hololift/vibe_contract.h"
#include "pbctopo/lattice_math.h"
#include "pbctopo/spatial_lift.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace titan_hololift;

namespace titan_hololift::testing {
void force_phase1_dp_pool_creation_failure(bool enabled) noexcept;
void force_phase1_dp_pool_creation_bad_alloc(bool enabled) noexcept;
void force_phase1_dp_run_setup_exception(bool enabled) noexcept;
void force_phase1_dp_worker_exception(bool enabled) noexcept;
}

static_assert(!std::is_copy_constructible_v<HoloLiftPhase1AuditedResult>);
static_assert(!std::is_copy_assignable_v<HoloLiftPhase1AuditedResult>);
static_assert(std::is_move_constructible_v<HoloLiftPhase1AuditedResult>);
static_assert(std::is_move_assignable_v<HoloLiftPhase1AuditedResult>);

using Coordinates = std::array<std::array<double, 3>, 4>;

constexpr std::array<HoloLiftSourceAtomKey, 4> kAtomOrder{
    HoloLiftSourceAtomKey{0, 12}, HoloLiftSourceAtomKey{0, 10},
    HoloLiftSourceAtomKey{1, 21}, HoloLiftSourceAtomKey{1, 20}};
constexpr std::array<std::size_t, 4> kInterleavedPermutation{0, 2, 1, 3};
constexpr std::array<HoloLiftSourceAtomKey, 4> kInterleavedAtomOrder{
    kAtomOrder[kInterleavedPermutation[0]],
    kAtomOrder[kInterleavedPermutation[1]],
    kAtomOrder[kInterleavedPermutation[2]],
    kAtomOrder[kInterleavedPermutation[3]]};

void expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL hololift_phase1_smoke: " << message << '\n';
    std::exit(1);
  }
}

class CommaDecimalPoint final : public std::numpunct<char> {
protected:
  char do_decimal_point() const override { return ','; }
};

struct ComponentFixture {
  std::vector<HoloLiftSourceAtomKey> atoms;
  std::uint64_t component_id = 0;
};

Coordinates permute_coordinates(const Coordinates &coordinates) {
  Coordinates permuted{};
  for (std::size_t index = 0; index < permuted.size(); ++index)
    permuted[index] = coordinates[kInterleavedPermutation[index]];
  return permuted;
}

void rebind_builder_frames(
    HoloLiftObservationStoreBuilder &builder,
    std::span<const Coordinates> coordinates,
    std::span<const HoloLiftSourceAtomKey> atom_order) {
  expect(builder.frames.size() == coordinates.size(),
         "frame rebinding fixture size mismatch");
  const auto &epoch = builder.topology_epochs.front();
  for (std::size_t frame_index = 0; frame_index < builder.frames.size();
       ++frame_index) {
    auto &frame = builder.frames[frame_index];
    frame.atom_selection_hash = hash_hololift_atom_selection(atom_order);
    frame.wrapped_coordinate_hash =
        hash_hololift_wrapped_coordinates(coordinates[frame_index]);
    const auto lift = replay_hololift_spatial_lift(
        epoch, frame, builder.hard_edges, atom_order,
        coordinates[frame_index]);
    expect(lift.has_value(), "failed to rebind fixture spatial lift");
    frame.spatial_lift_identity = lift->identity;
    frame.spatial_lift_ambiguous_hard_edges = lift->ambiguous_hard_edges;
    frame.spatial_lift_cycle_residuals = lift->cycle_residuals;
  }
}

HoloLiftCanonicalAssignment make_assignment(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    HoloLiftLatticeImage second_component_image) {
  std::vector<HoloLiftComponentImage> raw;
  raw.reserve(components.size());
  for (std::size_t idx = 0; idx < components.size(); ++idx) {
    raw.push_back({components[idx].component_id,
                   idx == 1 ? second_component_image
                            : HoloLiftLatticeImage{}});
  }
  auto assignment =
      canonicalize_hololift_assignment(layout_signature, components, raw);
  expect(assignment.has_value(), "failed to canonicalize fixture assignment");
  return std::move(*assignment);
}

HoloLiftCanonicalAssignment make_assignment(
    std::uint64_t layout_signature,
    std::span<const HoloLiftComponentRecord> components,
    std::int64_t second_component_x) {
  return make_assignment(layout_signature, components,
                         HoloLiftLatticeImage{second_component_x, 0, 0});
}

void append_frame(
    HoloLiftObservationStoreBuilder &builder, std::size_t frame_index,
    const Coordinates &coordinates,
    std::span<const HoloLiftCanonicalAssignment *const> assignments,
    std::size_t selected_local, bool request_strong_anchor,
    bool weak_observation = false) {
  expect(!assignments.empty() && selected_local < assignments.size(),
         "invalid frame assignment fixture");
  const auto &epoch = builder.topology_epochs.front();

  HoloLiftFrameRecord frame;
  frame.frame = frame_index;
  frame.time_ps = static_cast<double>(frame_index);
  frame.source_sequence_index = frame_index;
  frame.trajectory_frame_index = frame_index;
  frame.source_relation =
      frame_index == 0 ? HoloLiftSourceFrameRelation::SegmentStart
                       : HoloLiftSourceFrameRelation::ContiguousSourceFrame;
  frame.topology_epoch_index = 0;
  frame.topology_epoch_id = epoch.topology_epoch_id;
  frame.layout_signature = epoch.layout_signature;
  frame.topology_epoch_identity = epoch.topology_epoch_identity;
  frame.layout_identity = epoch.layout_identity;
  frame.box_matrix = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0,
                      0.0, 0.0, 10.0};
  frame.box_hash = hash_hololift_box_matrix(frame.box_matrix);
  frame.atom_selection_hash = hash_hololift_atom_selection(kAtomOrder);
  frame.canonical_atom_universe_hash = epoch.canonical_atom_universe_hash;
  frame.wrapped_coordinate_hash =
      hash_hololift_wrapped_coordinates(coordinates);
  frame.frame_binding_valid = true;
  frame.spatial_lift_policy_version =
      titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
  frame.spatial_lift_valid = true;
  const auto lift = replay_hololift_spatial_lift(
      epoch, frame, builder.hard_edges, kAtomOrder, coordinates);
  expect(lift.has_value(), "failed to build fixture spatial lift");
  frame.spatial_lift_identity = lift->identity;
  frame.spatial_lift_ambiguous_hard_edges = lift->ambiguous_hard_edges;
  frame.spatial_lift_cycle_residuals = lift->cycle_residuals;

  const std::size_t candidate_begin = builder.candidates.size();
  for (std::size_t local = 0; local < assignments.size(); ++local) {
    const auto &assignment = *assignments[local];
    const std::size_t image_begin = builder.component_images.size();
    for (const auto &component : assignment.component_images)
      builder.component_images.push_back(component.image);

    HoloLiftCandidateRecord candidate;
    candidate.rank = local + 1;
    candidate.layout_signature = epoch.layout_signature;
    candidate.assignment_signature = assignment.assignment_signature;
    candidate.evidence_relative_relation_signature =
        assignment.assignment_signature;
    candidate.evidence_compatible_hypothesis_hash = {0x55, 0x66};
    candidate.evidence_compatible_pair_hash = {0x77, 0x88};
    candidate.evidence_compatible_hypothesis_count = 1;
    candidate.evidence_supported_relation_count = 1;
    candidate.evidence_compatible_contact_count = 1;
    candidate.evidence_no_support_relation_count = 0;
    candidate.component_images =
        {image_begin, assignment.component_images.size()};
    if (weak_observation) {
      candidate.observation_class = HoloLiftObservationClass::WeakObservation;
    } else if (request_strong_anchor && assignments.size() == 1 &&
               lift->ambiguous_hard_edges == 0 &&
               lift->cycle_residuals == 0) {
      candidate.observation_class =
          HoloLiftObservationClass::StrongBoundedAnchor;
    } else {
      candidate.observation_class =
          HoloLiftObservationClass::AmbiguousAnchor;
    }
    candidate.evidence.selected_framewise = local == selected_local;
    candidate.evidence.hard_feasible = true;
    candidate.evidence.search_equivalence_known = true;
    candidate.evidence.equivalent_to_search_best = local == 0;
    candidate.evidence.equivalent_under_output_order = local == selected_local;
    candidate.domain_evidence.boundary_known = true;
    candidate.domain_evidence.touches_shell_boundary = false;
    candidate.layout_identity = assignment.layout_identity;
    candidate.assignment_identity = assignment.assignment_identity;
    builder.candidates.push_back(candidate);
  }

  frame.candidates = {candidate_begin, assignments.size()};
  frame.selected_candidate_index = candidate_begin + selected_local;
  frame.candidate_set_scope =
      HoloLiftCandidateSetScope::RetainedCertifiedBand;
  frame.provenance.status = weak_observation
                                ? HoloLiftFrameStatus::WeakObservation
                                : HoloLiftFrameStatus::Certified;
  frame.provenance.certificate_scope =
      HoloLiftCertificateScope::ImageIntercomponentSteric;
  frame.provenance.certificate_graph_source =
      HoloLiftCertificateGraphSource::GromacsTopology;
  frame.provenance.hard_graph_source =
      HoloLiftHardGraphSource::ExplicitTopology;
  frame.provenance.search_evidence_source =
      HoloLiftSearchEvidenceSource::ScoredContacts;
  frame.provenance.evidence_state =
      weak_observation ? HoloLiftEvidenceState::WeakDisconnected
                       : HoloLiftEvidenceState::SupportedConnected;
  frame.provenance.evidence_consistency_evaluated = true;
  frame.provenance.evidence_consistent = true;
  frame.provenance.evidence_graph_connected = !weak_observation;
  frame.provenance.soft_observed_contact_pair_hash = {0x11, 0x22};
  frame.provenance.soft_observed_lost_pair_hash = {0x33, 0x44};
  frame.provenance.soft_observed_all_hypotheses_hash = {0x99, 0xaa};
  frame.provenance.soft_observed_selected_hypothesis_hash = {0x55, 0x66};
  frame.provenance.soft_observed_selected_compatible_pair_hash = {0x77, 0x88};
  frame.provenance.soft_observed_hypothesis_count = 1;
  frame.provenance.soft_observed_component_relation_count = 1;
  frame.provenance.soft_observed_compatible_hypothesis_count = 1;
  frame.provenance.soft_observed_selected_compatible_contact_count = 1;
  frame.provenance.soft_observed_selected_relation_has_support = true;
  frame.provenance.soft_observed_hypothesis_construction_attempted = true;
  frame.provenance.soft_observed_hypothesis_construction_complete = true;
  frame.provenance.soft_observed_contacts_checked = 1;
  frame.provenance.soft_observed_contact_cutoff_A = 4.5;
  frame.provenance.soft_observed_contact_construction_attempted = true;
  frame.provenance.soft_observed_contact_construction_complete = true;
  const auto &selected = builder.candidates[frame.selected_candidate_index];
  frame.provenance.framewise_assignment_signature =
      selected.assignment_signature;
  frame.provenance.output_assignment_signature = selected.assignment_signature;
  frame.provenance.framewise_assignment_identity = selected.assignment_identity;
  frame.provenance.output_assignment_identity = selected.assignment_identity;
  frame.provenance.source_frame_report_present = true;

  frame.evidence.identified = !weak_observation;
  frame.evidence.uniqueness_search_exhaustive = true;
  frame.evidence.objective_uniqueness_known = true;
  frame.evidence.objective_unique_within_search_domain = !weak_observation;
  frame.evidence.evidence_uniqueness_known = true;
  frame.evidence.evidence_unique_within_search_domain =
      !weak_observation && assignments.size() == 1;
  frame.evidence.feasible_assignment_set_enumerated = true;
  frame.evidence.feasible_set_semantics =
      HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain;
  frame.evidence.equivalent_set_complete = true;
  frame.evidence.credible_alternative_set_complete = true;
  frame.evidence.retained_certified_assignment_count = assignments.size();
  frame.evidence.search_equivalent_assignments_exported = 1;
  frame.evidence.equivalent_best_assignments_within_domain = 1;
  frame.evidence.feasible_assignments_within_domain = assignments.size();
  frame.evidence.soft_score_valid_assignments_within_domain =
      assignments.size();
  frame.evidence.search_domain_assignments_exported = assignments.size();
  frame.search_domain.shell_radius = 2;
  frame.search_domain.audit_known = true;
  frame.search_domain.evidence_complete = true;
  frame.search_domain.bounded_domain_only = true;
  frame.search_domain.completeness =
      HoloLiftDomainCompleteness::BoundedExhaustive;
  builder.frames.push_back(frame);
}

HoloLiftObservationStoreBuilder make_builder(
    std::vector<Coordinates> &coordinates_by_frame,
    HoloLiftHash128 &assignment_a_identity,
    HoloLiftHash128 &assignment_b_identity,
    bool three_candidate_middle = false) {
  HoloLiftObservationStoreBuilder builder;
  builder.source_contract = make_hololift_vibe_v1_contract(
      "hololift-phase1-smoke", "phase1-package", "phase1-trajectory");

  std::vector<ComponentFixture> fixtures{
      {{{0, 10}, {0, 12}}, 0},
      {{{1, 20}, {1, 21}}, 0},
  };
  for (auto &fixture : fixtures) {
    auto id = hololift_component_id(fixture.atoms);
    expect(id.has_value(), "failed to build component id");
    fixture.component_id = *id;
  }
  std::sort(fixtures.begin(), fixtures.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.component_id < rhs.component_id;
            });
  for (const auto &fixture : fixtures) {
    const std::size_t begin = builder.exact_atom_memberships.size();
    builder.exact_atom_memberships.insert(
        builder.exact_atom_memberships.end(), fixture.atoms.begin(),
        fixture.atoms.end());
    HoloLiftComponentRecord component;
    component.component_id = fixture.component_id;
    component.owner = fixture.atoms.front().owner;
    component.root_atom = fixture.atoms.front();
    component.exact_atom_membership = {begin, fixture.atoms.size()};
    auto identity = hololift_component_identity128(fixture.atoms);
    expect(identity.has_value(), "failed to build component identity");
    component.component_identity = *identity;
    builder.components.push_back(component);
  }

  const auto components =
      std::span<const HoloLiftComponentRecord>(builder.components);
  const auto layout = hololift_layout_signature(components);
  const auto layout_identity = hololift_layout_identity128(components);
  expect(layout.has_value() && layout_identity.has_value(),
         "failed to build layout identity");
  builder.hard_edges = {
      {0, 1, HoloLiftHardEdgeKind::TopologyBond},
      {2, 3, HoloLiftHardEdgeKind::TopologyBond},
  };
  const auto epoch_id = hololift_topology_epoch_id(
      builder.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout, components,
      builder.exact_atom_memberships, builder.hard_edges);
  const auto epoch_identity = hololift_topology_epoch_identity128(
      builder.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout_identity, components,
      builder.hard_edges);
  const auto atom_universe =
      hash_hololift_canonical_atom_universe(builder.exact_atom_memberships);
  expect(epoch_id.has_value() && epoch_identity.has_value() &&
             atom_universe.has_value(),
         "failed to build topology epoch");
  HoloLiftTopologyEpochRecord epoch;
  epoch.topology_epoch_id = *epoch_id;
  epoch.layout_signature = *layout;
  epoch.components = {0, builder.components.size()};
  epoch.hard_edges = {0, builder.hard_edges.size()};
  epoch.hard_graph_source = HoloLiftHardGraphSource::ExplicitTopology;
  epoch.layout_identity = *layout_identity;
  epoch.topology_epoch_identity = *epoch_identity;
  epoch.canonical_atom_universe_hash = *atom_universe;
  epoch.atom_count = builder.exact_atom_memberships.size();
  epoch.hard_graph_cycle_rank = 0;
  builder.topology_epochs.push_back(epoch);

  const auto assignment_a = make_assignment(*layout, components, 0);
  const auto assignment_b = make_assignment(*layout, components, 1);
  expect(assignment_a.assignment_identity != assignment_b.assignment_identity,
         "fixture assignments are not distinct");
  assignment_a_identity = assignment_a.assignment_identity;
  assignment_b_identity = assignment_b.assignment_identity;

  const Coordinates intact{
      std::array<double, 3>{2.0, 1.0, 1.0},
      std::array<double, 3>{1.0, 1.0, 1.0},
      std::array<double, 3>{6.0, 5.0, 5.0},
      std::array<double, 3>{5.0, 5.0, 5.0}};
  const Coordinates ambiguous{
      std::array<double, 3>{5.0, 1.0, 1.0},
      std::array<double, 3>{0.0, 1.0, 1.0},
      std::array<double, 3>{6.0, 5.0, 5.0},
      std::array<double, 3>{5.0, 5.0, 5.0}};
  coordinates_by_frame = {intact, intact, intact, ambiguous, intact, intact};

  const std::array<const HoloLiftCanonicalAssignment *, 1> only_a{
      &assignment_a};
  const std::array<const HoloLiftCanonicalAssignment *, 2> b_then_a{
      &assignment_b, &assignment_a};
  const std::array<const HoloLiftCanonicalAssignment *, 1> only_b{
      &assignment_b};
  // Optional third middle candidate two cells away, for the K >= 2 test.
  const auto assignment_c = make_assignment(*layout, components, 2);
  const std::array<const HoloLiftCanonicalAssignment *, 3> b_a_c{
      &assignment_b, &assignment_a, &assignment_c};
  append_frame(builder, 0, coordinates_by_frame[0], only_a, 0, true);
  if (three_candidate_middle)
    append_frame(builder, 1, coordinates_by_frame[1], b_a_c, 0, false, true);
  else
    append_frame(builder, 1, coordinates_by_frame[1], b_then_a, 0, false,
                 true);
  append_frame(builder, 2, coordinates_by_frame[2], only_a, 0, true);
  append_frame(builder, 3, coordinates_by_frame[3], only_a, 0, false);
  append_frame(builder, 4, coordinates_by_frame[4], only_b, 0, true);
  HoloLiftFrameRecord no_band = builder.frames.back();
  no_band.frame = 5;
  no_band.time_ps = 5.0;
  no_band.source_sequence_index = 5;
  no_band.trajectory_frame_index = 5;
  no_band.source_relation =
      HoloLiftSourceFrameRelation::ContiguousSourceFrame;
  no_band.candidates = {builder.candidates.size(), 0};
  no_band.selected_candidate_index = HOLOLIFT_NO_INDEX;
  no_band.candidate_set_scope =
      HoloLiftCandidateSetScope::RetainedCertifiedBand;
  no_band.provenance = {};
  no_band.provenance.source_frame_report_present = true;
  no_band.evidence = {};
  no_band.search_domain = {};
  builder.frames.push_back(no_band);
  builder.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest,
      builder.frames.size(), builder.frames.size(), 0, 0};
  expect(builder.frames[3].spatial_lift_ambiguous_hard_edges == 1,
         "fixture did not create one half-cell ambiguity");
  return builder;
}

HoloLiftObservationStoreBuilder make_rolling_memory_builder(
    std::size_t frame_count, std::vector<Coordinates> &coordinates_by_frame,
    HoloLiftHash128 &expected_assignment_identity) {
  expect(frame_count > 1, "rolling-memory fixture needs multiple frames");
  std::vector<Coordinates> template_coordinates;
  HoloLiftHash128 assignment_a_identity;
  HoloLiftHash128 assignment_b_identity;
  auto template_builder = make_builder(
      template_coordinates, assignment_a_identity, assignment_b_identity);
  const auto &template_frame = template_builder.frames[1];
  expect(template_frame.candidates.count == 2,
         "rolling-memory fixture needs two candidates");
  expected_assignment_identity = assignment_b_identity;

  HoloLiftObservationStoreBuilder builder;
  builder.source_contract = template_builder.source_contract;
  builder.topology_epochs = template_builder.topology_epochs;
  builder.components = template_builder.components;
  builder.exact_atom_memberships = template_builder.exact_atom_memberships;
  builder.exact_atom_masses = template_builder.exact_atom_masses;
  builder.hard_edges = template_builder.hard_edges;
  coordinates_by_frame.assign(frame_count, template_coordinates[1]);

  for (std::size_t frame_index = 0; frame_index < frame_count;
       ++frame_index) {
    HoloLiftFrameRecord frame = template_frame;
    frame.frame = frame_index;
    frame.time_ps = static_cast<double>(frame_index);
    frame.source_sequence_index = frame_index;
    frame.trajectory_frame_index = frame_index;
    frame.source_relation =
        frame_index == 0
            ? HoloLiftSourceFrameRelation::SegmentStart
            : HoloLiftSourceFrameRelation::ContiguousSourceFrame;

    const std::size_t candidate_begin = builder.candidates.size();
    for (std::size_t local = 0; local < template_frame.candidates.count;
         ++local) {
      HoloLiftCandidateRecord candidate =
          template_builder
              .candidates[template_frame.candidates.begin + local];
      const auto source_images = std::span<const HoloLiftLatticeImage>(
          template_builder.component_images)
                                     .subspan(candidate.component_images.begin,
                                              candidate.component_images.count);
      const std::size_t image_begin = builder.component_images.size();
      builder.component_images.insert(builder.component_images.end(),
                                      source_images.begin(),
                                      source_images.end());
      candidate.component_images = {image_begin, source_images.size()};
      builder.candidates.push_back(candidate);
    }
    frame.candidates = {candidate_begin, template_frame.candidates.count};
    frame.selected_candidate_index = candidate_begin;
    builder.frames.push_back(frame);
  }
  builder.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest,
      builder.frames.size(), builder.frames.size(), 0, 0};
  return builder;
}

HoloLiftPhase1Result run_rolling_memory_fixture(
    std::size_t frame_count, const HoloLiftPhase1Config &config) {
  std::vector<Coordinates> coordinates;
  HoloLiftHash128 expected_assignment_identity;
  auto builder = make_rolling_memory_builder(
      frame_count, coordinates, expected_assignment_identity);
  auto finalized = finalize_hololift_observation_store(std::move(builder));
  expect(finalized.has_value(),
         "rolling-memory fixture failed store validation");
  auto store = std::move(*finalized);
  auto preparation = make_hololift_phase1_preparation_context(store);
  expect(preparation.has_value(),
         "rolling-memory fixture failed Phase 1 preparation");
  std::vector<HoloLiftPhase1PreparedFrame> prepared;
  prepared.reserve(frame_count);
  for (std::size_t frame_index = 0; frame_index < frame_count;
       ++frame_index) {
    auto frame = prepare_hololift_phase1_frame(
        *preparation, store, frame_index, frame_index,
        store.frames()[frame_index].box_matrix, kAtomOrder,
        coordinates[frame_index]);
    expect(frame.has_value(),
           "rolling-memory fixture frame preparation failed");
    prepared.push_back(std::move(*frame));
  }
  auto result =
      solve_hololift_phase1_temporal_path(store, prepared, config);
  expect(result.has_value(), "rolling-memory fixture solve failed");
  expect(result->segments.size() == 1 &&
             result->segments[0].frame_count == frame_count &&
             std::fabs(result->objective) < 1.0e-12,
         "rolling-memory fixture objective or segment changed");
  for (const auto &selection : result->frames) {
    expect(selection.use == HoloLiftPhase1FrameUse::TemporalObservation &&
               selection.candidate_index < store.candidates().size() &&
               store.candidates()[selection.candidate_index]
                       .assignment_identity == expected_assignment_identity &&
               selection.cumulative_global_gauge == HoloLiftLatticeImage{},
           "rolling-memory fixture path or gauge changed");
  }
  return std::move(*result);
}

HoloLiftObservationStoreBuilder make_parallel_executor_builder(
    std::size_t frame_count, std::vector<Coordinates> &coordinates_by_frame) {
  constexpr std::size_t kCandidateCount = 64;
  expect(frame_count > 1, "parallel executor fixture needs multiple frames");
  std::vector<Coordinates> template_coordinates;
  HoloLiftHash128 assignment_a_identity;
  HoloLiftHash128 assignment_b_identity;
  auto template_builder = make_builder(
      template_coordinates, assignment_a_identity, assignment_b_identity);

  HoloLiftObservationStoreBuilder builder;
  builder.source_contract = template_builder.source_contract;
  builder.topology_epochs = template_builder.topology_epochs;
  builder.components = template_builder.components;
  builder.exact_atom_memberships = template_builder.exact_atom_memberships;
  builder.exact_atom_masses = template_builder.exact_atom_masses;
  builder.hard_edges = template_builder.hard_edges;
  coordinates_by_frame.assign(frame_count, template_coordinates[1]);

  const auto components =
      std::span<const HoloLiftComponentRecord>(builder.components);
  const std::uint64_t layout_signature =
      builder.topology_epochs.front().layout_signature;
  std::vector<HoloLiftCanonicalAssignment> assignments;
  assignments.reserve(kCandidateCount);
  for (std::int64_t x = -2;
       x <= 2 && assignments.size() < kCandidateCount; ++x) {
    for (std::int64_t y = -2;
         y <= 2 && assignments.size() < kCandidateCount; ++y) {
      for (std::int64_t z = -2;
           z <= 2 && assignments.size() < kCandidateCount; ++z) {
        assignments.push_back(make_assignment(
            layout_signature, components, HoloLiftLatticeImage{x, y, z}));
      }
    }
  }
  expect(assignments.size() == kCandidateCount,
         "parallel executor fixture candidate domain is incomplete");
  std::vector<const HoloLiftCanonicalAssignment *> assignment_views;
  assignment_views.reserve(assignments.size());
  for (const auto &assignment : assignments)
    assignment_views.push_back(&assignment);

  for (std::size_t frame_index = 0; frame_index < frame_count;
       ++frame_index) {
    append_frame(builder, frame_index, coordinates_by_frame[frame_index],
                 assignment_views, 0, false, true);
  }
  builder.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest,
      builder.frames.size(), builder.frames.size(), 0, 0};
  return builder;
}

std::expected<HoloLiftPhase1Result, std::string>
try_run_parallel_executor_fixture(
    std::size_t frame_count, const HoloLiftPhase1Config &config) {
  std::vector<Coordinates> coordinates;
  auto store = finalize_hololift_observation_store(
      make_parallel_executor_builder(frame_count, coordinates));
  expect(store.has_value(), "parallel executor fixture failed validation");
  auto preparation = make_hololift_phase1_preparation_context(*store);
  expect(preparation.has_value(),
         "parallel executor fixture failed Phase 1 preparation");
  std::vector<HoloLiftPhase1PreparedFrame> prepared;
  prepared.reserve(frame_count);
  for (std::size_t frame_index = 0; frame_index < frame_count;
       ++frame_index) {
    auto frame = prepare_hololift_phase1_frame(
        *preparation, *store, frame_index, frame_index,
        store->frames()[frame_index].box_matrix, kAtomOrder,
        coordinates[frame_index]);
    expect(frame.has_value(),
           "parallel executor fixture frame preparation failed");
    prepared.push_back(std::move(*frame));
  }
  auto result =
      solve_hololift_phase1_temporal_path(*store, prepared, config);
  if (!result)
    return std::unexpected(result.error());
  expect(result->segments.size() == 1 &&
             result->segments.front().frame_count == frame_count,
         "parallel executor fixture segment layout changed");
  return std::move(*result);
}

HoloLiftPhase1Result run_parallel_executor_fixture(
    std::size_t frame_count, const HoloLiftPhase1Config &config) {
  auto result = try_run_parallel_executor_fixture(frame_count, config);
  expect(result.has_value(), "parallel executor fixture solve failed");
  return std::move(*result);
}

void expect_same_parallel_dp_solution(const HoloLiftPhase1Result &reference,
                                      const HoloLiftPhase1Result &candidate) {
  expect(reference.frames.size() == candidate.frames.size() &&
             reference.segments.size() == candidate.segments.size() &&
             reference.objective == candidate.objective,
         "parallel executor changed objective or result layout");
  for (std::size_t index = 0; index < reference.frames.size(); ++index) {
    const auto &lhs = reference.frames[index];
    const auto &rhs = candidate.frames[index];
    expect(lhs.candidate_index == rhs.candidate_index &&
               lhs.emission_cost == rhs.emission_cost &&
               lhs.transition_cost == rhs.transition_cost &&
               lhs.cumulative_segment_cost == rhs.cumulative_segment_cost &&
               lhs.selected_global_gauge_increment ==
                   rhs.selected_global_gauge_increment &&
               lhs.cumulative_global_gauge == rhs.cumulative_global_gauge &&
               lhs.global_gauge_increment_ambiguous ==
                   rhs.global_gauge_increment_ambiguous,
           "parallel executor changed path, gauge, or frame ambiguity");
  }
  for (std::size_t index = 0; index < reference.segments.size(); ++index) {
    const auto &lhs = reference.segments[index];
    const auto &rhs = candidate.segments[index];
    expect(lhs.objective == rhs.objective &&
               lhs.second_best_objective == rhs.second_best_objective &&
               lhs.absolute_path_gap == rhs.absolute_path_gap &&
               lhs.relative_path_gap == rhs.relative_path_gap &&
               lhs.optimal_path_count_capped ==
                   rhs.optimal_path_count_capped &&
               lhs.ambiguous_gauge_transition_count ==
                   rhs.ambiguous_gauge_transition_count &&
               lhs.unique_on_configured_retained_state_graph ==
                   rhs.unique_on_configured_retained_state_graph,
           "parallel executor changed tie or segment ambiguity semantics");
  }
}

using SingleCoordinates = std::array<std::array<double, 3>, 2>;
constexpr std::array<HoloLiftSourceAtomKey, 2> kSingleAtomOrder{
    HoloLiftSourceAtomKey{0, 1}, HoloLiftSourceAtomKey{0, 2}};

std::array<double, 6>
normalized_gram_for_box(const std::array<double, 9> &box) {
  auto dot_rows = [&](std::size_t lhs, std::size_t rhs) {
    return box[lhs * 3] * box[rhs * 3] +
           box[lhs * 3 + 1] * box[rhs * 3 + 1] +
           box[lhs * 3 + 2] * box[rhs * 3 + 2];
  };
  std::array<double, 6> gram{
      dot_rows(0, 0), dot_rows(0, 1), dot_rows(0, 2),
      dot_rows(1, 1), dot_rows(1, 2), dot_rows(2, 2)};
  const double scale = (gram[0] + gram[3] + gram[5]) / 3.0;
  for (double &value : gram)
    value /= scale;
  return gram;
}

HoloLiftObservationStoreBuilder make_single_component_builder(
    std::span<const SingleCoordinates> coordinates,
    std::span<const std::array<double, 9>> boxes,
    std::span<const double> times,
    std::span<const HoloLiftLatticeImage> component_images = {}) {
  expect(!coordinates.empty() && coordinates.size() == boxes.size() &&
             boxes.size() == times.size() &&
             (component_images.empty() ||
              component_images.size() == coordinates.size()),
         "invalid single-component trajectory fixture");
  HoloLiftObservationStoreBuilder builder;
  builder.source_contract = make_hololift_vibe_v1_contract(
      "hololift-phase1-physical-smoke", "single-component-package",
      "single-component-trajectory");

  builder.exact_atom_memberships.insert(builder.exact_atom_memberships.end(),
                                        kSingleAtomOrder.begin(),
                                        kSingleAtomOrder.end());
  builder.hard_edges.push_back(
      {0, 1, HoloLiftHardEdgeKind::TopologyBond});
  HoloLiftComponentRecord component;
  const auto component_id = hololift_component_id(kSingleAtomOrder);
  const auto component_identity =
      hololift_component_identity128(kSingleAtomOrder);
  expect(component_id.has_value() && component_identity.has_value(),
         "failed to identify single component");
  component.component_id = *component_id;
  component.owner = 0;
  component.root_atom = kSingleAtomOrder[0];
  component.exact_atom_membership = {0, 2};
  component.component_identity = *component_identity;
  builder.components.push_back(component);

  const auto components =
      std::span<const HoloLiftComponentRecord>(builder.components);
  const auto layout = hololift_layout_signature(components);
  const auto layout_identity = hololift_layout_identity128(components);
  const auto atom_universe =
      hash_hololift_canonical_atom_universe(kSingleAtomOrder);
  expect(layout.has_value() && layout_identity.has_value() &&
             atom_universe.has_value(),
         "failed to identify single-component layout");
  const auto epoch_id = hololift_topology_epoch_id(
      builder.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout, components,
      builder.exact_atom_memberships, builder.hard_edges);
  const auto epoch_identity = hololift_topology_epoch_identity128(
      builder.source_contract.schema_version,
      HoloLiftHardGraphSource::ExplicitTopology, *layout_identity, components,
      builder.hard_edges);
  expect(epoch_id.has_value() && epoch_identity.has_value(),
         "failed to identify single-component epoch");
  HoloLiftTopologyEpochRecord epoch;
  epoch.topology_epoch_id = *epoch_id;
  epoch.layout_signature = *layout;
  epoch.components = {0, 1};
  epoch.hard_edges = {0, 1};
  epoch.hard_graph_source = HoloLiftHardGraphSource::ExplicitTopology;
  epoch.layout_identity = *layout_identity;
  epoch.topology_epoch_identity = *epoch_identity;
  epoch.canonical_atom_universe_hash = *atom_universe;
  epoch.atom_count = 2;
  epoch.hard_graph_cycle_rank = 0;
  builder.topology_epochs.push_back(epoch);

  for (std::size_t frame_index = 0; frame_index < coordinates.size();
       ++frame_index) {
    const HoloLiftLatticeImage selected_image =
        component_images.empty() ? HoloLiftLatticeImage{}
                                 : component_images[frame_index];
    const std::array<HoloLiftComponentImage, 1> raw_images{
        HoloLiftComponentImage{component.component_id, selected_image}};
    const auto assignment =
        canonicalize_hololift_assignment(*layout, components, raw_images);
    expect(assignment.has_value(),
           "failed to canonicalize single-component assignment");
    const std::size_t image_begin = builder.component_images.size();
    builder.component_images.push_back(selected_image);
    HoloLiftCandidateRecord candidate;
    candidate.rank = 1;
    candidate.observation_class =
        HoloLiftObservationClass::StrongBoundedAnchor;
    candidate.layout_signature = *layout;
    candidate.assignment_signature = assignment->assignment_signature;
    candidate.evidence_relative_relation_signature =
        assignment->assignment_signature;
    candidate.component_images = {image_begin, 1};
    candidate.evidence.selected_framewise = true;
    candidate.evidence.hard_feasible = true;
    candidate.evidence.search_equivalence_known = true;
    candidate.evidence.equivalent_to_search_best = true;
    candidate.evidence.equivalent_under_output_order = true;
    candidate.domain_evidence.boundary_known = true;
    candidate.domain_evidence.touches_shell_boundary = false;
    candidate.layout_identity = assignment->layout_identity;
    candidate.assignment_identity = assignment->assignment_identity;
    const std::size_t candidate_index = builder.candidates.size();
    builder.candidates.push_back(candidate);

    HoloLiftFrameRecord frame;
    frame.frame = frame_index;
    frame.time_ps = times[frame_index];
    frame.source_sequence_index = frame_index;
    frame.trajectory_frame_index = frame_index;
    frame.source_relation =
        frame_index == 0
            ? HoloLiftSourceFrameRelation::SegmentStart
            : HoloLiftSourceFrameRelation::ContiguousSourceFrame;
    frame.topology_epoch_index = 0;
    frame.topology_epoch_id = *epoch_id;
    frame.layout_signature = *layout;
    frame.topology_epoch_identity = *epoch_identity;
    frame.layout_identity = *layout_identity;
    frame.box_matrix = boxes[frame_index];
    frame.normalized_gram = normalized_gram_for_box(frame.box_matrix);
    frame.box_hash = hash_hololift_box_matrix(frame.box_matrix);
    frame.atom_selection_hash =
        hash_hololift_atom_selection(kSingleAtomOrder);
    frame.canonical_atom_universe_hash = *atom_universe;
    frame.wrapped_coordinate_hash =
        hash_hololift_wrapped_coordinates(coordinates[frame_index]);
    frame.frame_binding_valid = true;
    frame.spatial_lift_policy_version =
        titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
    frame.spatial_lift_valid = true;
    const auto lift = replay_hololift_spatial_lift(
        epoch, frame, builder.hard_edges, kSingleAtomOrder,
        coordinates[frame_index]);
    expect(lift.has_value(), "single-component spatial lift failed");
    frame.spatial_lift_identity = lift->identity;
    frame.spatial_lift_ambiguous_hard_edges = lift->ambiguous_hard_edges;
    frame.spatial_lift_cycle_residuals = lift->cycle_residuals;
    frame.candidates = {candidate_index, 1};
    frame.selected_candidate_index = candidate_index;
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
    frame.provenance.soft_observed_all_hypotheses_hash = {0x99, 0xaa};
    frame.provenance.soft_observed_hypothesis_construction_attempted = true;
    frame.provenance.soft_observed_hypothesis_construction_complete = true;
    frame.provenance.soft_observed_contact_cutoff_A = 4.5;
    frame.provenance.soft_observed_contact_construction_attempted = true;
    frame.provenance.soft_observed_contact_construction_complete = true;
    frame.provenance.framewise_assignment_signature =
        assignment->assignment_signature;
    frame.provenance.output_assignment_signature =
        assignment->assignment_signature;
    frame.provenance.framewise_assignment_identity =
        assignment->assignment_identity;
    frame.provenance.output_assignment_identity =
        assignment->assignment_identity;
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
    builder.frames.push_back(frame);
  }
  builder.source_coverage = {
      HoloLiftSourceCoverageScope::CompleteSourceManifest,
      builder.frames.size(), builder.frames.size(), 0, 0};
  return builder;
}

std::expected<HoloLiftPhase1Result, std::string> solve_single_component(
    const HoloLiftObservationStore &store,
    std::span<const SingleCoordinates> coordinates,
    const HoloLiftPhase1Config &config,
    HoloLiftPhase1PreparationContext &context,
    std::vector<HoloLiftPhase1PreparedFrame> &prepared) {
  prepared.clear();
  for (std::size_t frame_index = 0; frame_index < coordinates.size();
       ++frame_index) {
    auto frame = prepare_hololift_phase1_frame(
        context, store, frame_index, frame_index,
        store.frames()[frame_index].box_matrix, kSingleAtomOrder,
        coordinates[frame_index]);
    if (!frame)
      return std::unexpected(frame.error());
    prepared.push_back(std::move(*frame));
  }
  return solve_hololift_phase1_temporal_path(store, prepared, config);
}

double brute_force_transition_cost(
    const HoloLiftObservationStore &store,
    const HoloLiftPhase1PreparedFrame &previous_frame,
    const HoloLiftPhase1PreparedFrame &current_frame,
    std::size_t previous_candidate_index,
    std::size_t current_candidate_index,
    const HoloLiftPhase1Config &config) {
  std::array<double, 6> gram{};
  for (std::size_t index = 0; index < gram.size(); ++index) {
    gram[index] = 0.5 * (previous_frame.physical_gram()[index] +
                         current_frame.physical_gram()[index]);
  }
  const titan_pbctopo::PbctopoLatticeMetric metric(gram);
  expect(metric.valid(), "brute-force transition metric is invalid");
  const auto &previous_candidate =
      store.candidates()[previous_candidate_index];
  const auto &current_candidate = store.candidates()[current_candidate_index];
  const auto previous_images = store.component_images().subspan(
      previous_candidate.component_images.begin,
      previous_candidate.component_images.count);
  const auto current_images = store.component_images().subspan(
      current_candidate.component_images.begin,
      current_candidate.component_images.count);
  const auto previous_base =
      previous_frame.component_fractional_representatives();
  const auto current_base =
      current_frame.component_fractional_representatives();
  const auto &epoch = store.topology_epochs()[
      store.frames()[previous_frame.frame_index()].topology_epoch_index];
  const auto components = store.components().subspan(
      epoch.components.begin, epoch.components.count);
  expect(previous_images.size() == components.size() &&
             current_images.size() == components.size(),
         "brute-force transition component count mismatch");

  double best_weighted_residual2 = std::numeric_limits<double>::infinity();
  double total_weight = 0.0;
  for (const auto &component : components) {
    total_weight +=
        static_cast<double>(component.exact_atom_membership.count);
  }
  for (std::int64_t qx = -3; qx <= 3; ++qx) {
    for (std::int64_t qy = -3; qy <= 3; ++qy) {
      for (std::int64_t qz = -3; qz <= 3; ++qz) {
        double weighted_residual2 = 0.0;
        for (std::size_t component = 0; component < components.size();
             ++component) {
          const std::array<double, 3> delta{
              current_base[component][0] - previous_base[component][0] +
                  static_cast<double>(current_images[component].x -
                                      previous_images[component].x + qx),
              current_base[component][1] - previous_base[component][1] +
                  static_cast<double>(current_images[component].y -
                                      previous_images[component].y + qy),
              current_base[component][2] - previous_base[component][2] +
                  static_cast<double>(current_images[component].z -
                                      previous_images[component].z + qz)};
          const auto cartesian = metric.scaled_to_cartesian(
              delta[0], delta[1], delta[2]);
          const double residual2 =
              cartesian[0] * cartesian[0] +
              cartesian[1] * cartesian[1] +
              cartesian[2] * cartesian[2];
          weighted_residual2 +=
              static_cast<double>(
                  components[component].exact_atom_membership.count) *
              residual2;
        }
        best_weighted_residual2 =
            std::min(best_weighted_residual2, weighted_residual2);
      }
    }
  }
  const double delta_time =
      std::fabs(current_frame.time_ps() - previous_frame.time_ps());
  double time_factor = 1.0;
  if (config.time_scaling == HoloLiftPhase1TimeScaling::Diffusive)
    time_factor = delta_time;
  else if (config.time_scaling == HoloLiftPhase1TimeScaling::Ballistic)
    time_factor = delta_time * delta_time;
  const double variance = config.variance_floor_A2 +
                          config.temporal_scale * config.temporal_scale *
                              time_factor;
  return config.physical_transition_weight * best_weighted_residual2 /
         (total_weight * variance);
}

double brute_force_emission(const HoloLiftObservationStore &store,
                            std::size_t frame_index,
                            std::size_t candidate_index,
                            const HoloLiftPhase1Config &config) {
  const auto &frame = store.frames()[frame_index];
  std::size_t feasible_count = 0;
  std::size_t strong_anchor = HOLOLIFT_NO_INDEX;
  for (std::size_t local = 0; local < frame.candidates.count; ++local) {
    const std::size_t index = frame.candidates.begin + local;
    const auto &candidate = store.candidates()[index];
    if (!candidate.evidence.hard_feasible)
      continue;
    ++feasible_count;
    if (candidate.observation_class ==
        HoloLiftObservationClass::StrongBoundedAnchor) {
      strong_anchor = index;
    }
  }
  const auto &candidate = store.candidates()[candidate_index];
  double emission = config.ordinal_rank_weight *
                    static_cast<double>(candidate.rank - 1) /
                    static_cast<double>(std::max<std::size_t>(
                        1, feasible_count - 1));
  if (strong_anchor != HOLOLIFT_NO_INDEX &&
      strong_anchor != candidate_index) {
    emission += config.bounded_anchor_competitor_penalty;
  }
  return emission;
}

std::array<double, 9> integer_left_multiply_box(
    const std::array<std::int64_t, 9> &matrix,
    const std::array<double, 9> &box) {
  std::array<double, 9> result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        result[3 * row + column] +=
            static_cast<double>(matrix[3 * row + inner]) *
            box[3 * inner + column];
      }
    }
  }
  return result;
}

void check_policy_features(const std::filesystem::path &artifact_base) {
  expect(HoloLiftPhase1Config{}.supported_evidence_mismatch_weight == 1000.0,
         "supported-evidence weight default is not 1000");
  auto clear_support = [](HoloLiftCandidateRecord &candidate) {
    candidate.evidence_compatible_hypothesis_hash = {};
    candidate.evidence_compatible_pair_hash = {};
    candidate.evidence_compatible_hypothesis_count = 0;
    candidate.evidence_supported_relation_count = 0;
    candidate.evidence_compatible_contact_count = 0;
    candidate.evidence_no_support_relation_count = 1;
  };
  for (int supported_count : {0, 1, 2}) {
    std::vector<Coordinates> coords;
    HoloLiftHash128 a, b;
    auto builder = make_builder(coords, a, b);
    auto &middle = builder.frames[1];
    if (supported_count < 2)
      clear_support(builder.candidates[middle.candidates.begin + 1]);
    if (supported_count == 0) {
      clear_support(builder.candidates[middle.candidates.begin]);
      middle.provenance.soft_observed_selected_hypothesis_hash = {};
      middle.provenance.soft_observed_selected_compatible_pair_hash = {};
      middle.provenance.soft_observed_compatible_hypothesis_count = 0;
      middle.provenance.soft_observed_selected_compatible_contact_count = 0;
      middle.provenance.soft_observed_alternative_hypothesis_contact_count = 1;
      middle.provenance.soft_observed_no_support_relation_count = 1;
      middle.provenance.soft_observed_selected_relation_has_support = false;
    }
    auto store = finalize_hololift_observation_store(std::move(builder));
    if (!store) std::cerr << store.error().message << '\n';
    expect(store.has_value(), "evidence-preference fixture invalid");
    auto context = make_hololift_phase1_preparation_context(*store);
    expect(context.has_value(), "policy context failed");
    std::vector<HoloLiftPhase1PreparedFrame> prepared;
    for (std::size_t i = 0; i < coords.size(); ++i) {
      auto token = prepare_hololift_phase1_frame(
          *context, *store, i, i, store->frames()[i].box_matrix, kAtomOrder, coords[i]);
      expect(token.has_value(), "policy preparation failed");
      prepared.push_back(std::move(*token));
    }
    HoloLiftPhase1Config config;
    config.supported_evidence_mismatch_weight = 0;  // baseline without evidence term
    config.physical_transition_weight = 4;
    auto baseline = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(baseline.has_value(), "policy baseline failed");
    config.unique_evidence_mismatch_weight = 1000;
    auto preferred = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(preferred.has_value(), "evidence-preference solve failed");
    if (supported_count == 1) {
      expect(preferred->frames[1].candidate_index == store->frames()[1].candidates.begin &&
                 baseline->frames[1].candidate_index != preferred->frames[1].candidate_index,
             "unique evidence preference did not affect the conflicting optimum");
    } else {
      expect(preferred->frames[1].candidate_index == baseline->frames[1].candidate_index &&
                 preferred->objective == baseline->objective,
             "unique evidence preference affected K=0 or K>=2");
    }
    // The supported-evidence term is the unique one at K=1 and does nothing
    // where every candidate or none carries evidence.
    config.unique_evidence_mismatch_weight = 0;
    config.supported_evidence_mismatch_weight = 1000;
    auto supported = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(supported.has_value(), "supported-evidence preference solve failed");
    if (supported_count == 1) {
      expect(supported->frames[1].candidate_index == preferred->frames[1].candidate_index,
             "supported evidence preference differs from the unique one at K=1");
    } else {
      expect(supported->frames[1].candidate_index == baseline->frames[1].candidate_index &&
                 supported->objective == baseline->objective,
             "supported evidence preference acted where all or no candidates carry evidence");
    }
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
      config.supported_evidence_mismatch_weight = invalid;
      expect(!solve_hololift_phase1_temporal_path(*store, prepared, config),
             "invalid supported evidence mismatch weight accepted");
    }
    config.supported_evidence_mismatch_weight = 0;
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
      config.unique_evidence_mismatch_weight = invalid;
      expect(!solve_hololift_phase1_temporal_path(*store, prepared, config),
             "invalid unique evidence mismatch weight accepted");
    }
  }

  // K = 2 with an unsupported candidate: b (rank 1) and c (rank 3) carry
  // evidence, a (rank 2) does not but continues frames 0 and 2.  The unique
  // term cannot act; the supported term must move the path off a.
  {
    std::vector<Coordinates> coords;
    HoloLiftHash128 a, b;
    auto builder = make_builder(coords, a, b, true);
    auto &middle = builder.frames[1];
    const std::size_t unsupported_index = middle.candidates.begin + 1;
    clear_support(builder.candidates[unsupported_index]);
    auto store = finalize_hololift_observation_store(std::move(builder));
    if (!store) std::cerr << store.error().message << '\n';
    expect(store.has_value(), "K=2 evidence fixture invalid");
    auto context = make_hololift_phase1_preparation_context(*store);
    expect(context.has_value(), "K=2 context failed");
    std::vector<HoloLiftPhase1PreparedFrame> prepared;
    for (std::size_t i = 0; i < coords.size(); ++i) {
      auto token = prepare_hololift_phase1_frame(
          *context, *store, i, i, store->frames()[i].box_matrix, kAtomOrder, coords[i]);
      expect(token.has_value(), "K=2 preparation failed");
      prepared.push_back(std::move(*token));
    }
    HoloLiftPhase1Config config;
    config.supported_evidence_mismatch_weight = 0;  // baseline without evidence term
    config.physical_transition_weight = 4;
    auto baseline = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(baseline.has_value() && baseline->frames[1].candidate_index == unsupported_index,
           "K=2 baseline did not hold the unsupported continuity candidate");
    config.unique_evidence_mismatch_weight = 1000;
    auto unique = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(unique.has_value() &&
               unique->frames[1].candidate_index == unsupported_index &&
               unique->objective == baseline->objective,
           "unique evidence preference acted at K=2");
    config.unique_evidence_mismatch_weight = 0;
    config.supported_evidence_mismatch_weight = 1000;
    auto supported = solve_hololift_phase1_temporal_path(*store, prepared, config);
    expect(supported.has_value(), "K=2 supported-evidence solve failed");
    const std::size_t chosen = supported->frames[1].candidate_index;
    expect(chosen != unsupported_index &&
               store->candidates()[chosen].evidence_compatible_hypothesis_count > 0,
           "supported evidence preference left the path on the unsupported candidate at K=2");
  }

  std::vector<Coordinates> coords;
  HoloLiftHash128 a, b;
  auto builder = make_builder(coords, a, b);
  auto &middle = builder.frames[1];
  const std::size_t carried_index = middle.candidates.begin + 1;
  auto &candidate = builder.candidates[carried_index];
  clear_support(candidate);
  candidate.evidence.hard_feasible = false;
  candidate.evidence.search_equivalence_known = false;
  candidate.evidence.equivalent_to_search_best = false;
  candidate.evidence.equivalent_under_output_order = false;
  candidate.observation_class = HoloLiftObservationClass::NotHardFeasible;
  middle.evidence.search_domain_assignments_exported = 1;
  middle.evidence.credible_alternative_set_complete = false;
  middle.search_domain.completeness = HoloLiftDomainCompleteness::Unknown;
  expect(!validate_hololift_observation_store(builder),
         "unflagged provider-infeasible candidate was accepted");
  candidate.carried = true;
  candidate.policy_admission = HoloLiftPolicyAdmission::CarryPreviousFrame;
  candidate.provider_evaluation = HoloLiftProviderEvaluation::NotEvaluated;
  candidate.policy_source_frame_index = 0;
  candidate.policy_source_assignment_identity = a;
  auto broken_source = builder;
  broken_source.candidates[carried_index].policy_source_frame_index = 1;
  expect(!validate_hololift_observation_store(broken_source),
         "carry with invalid source frame accepted");
  auto fake_hard = builder;
  fake_hard.candidates[carried_index].evidence.hard_feasible = true;
  expect(!validate_hololift_observation_store(fake_hard),
         "policy admission promoted itself to provider hard feasibility");
  auto store = finalize_hololift_observation_store(std::move(builder));
  if (!store) std::cerr << store.error().message << '\n';
  expect(store.has_value(), "explicit carry admission failed validation");
  const auto binary_path = artifact_base.string() + ".policy.hlift";
  expect(write_hololift_observation_store_binary(*store, binary_path).has_value(),
         "policy binary write failed");
  auto restored = read_hololift_observation_store_binary(binary_path);
  expect(restored.has_value(), "policy binary read failed");
  const auto &roundtrip = restored->candidates()[carried_index];
  expect(roundtrip.carried && !roundtrip.evidence.hard_feasible &&
             roundtrip.policy_admission == HoloLiftPolicyAdmission::CarryPreviousFrame &&
             roundtrip.provider_evaluation == HoloLiftProviderEvaluation::NotEvaluated &&
             roundtrip.policy_source_frame_index == 0 &&
             roundtrip.policy_source_assignment_identity == a,
         "policy admission/provenance did not survive binary roundtrip");
  auto context = make_hololift_phase1_preparation_context(*restored);
  expect(context.has_value(), "policy restored context failed");
  std::vector<HoloLiftPhase1PreparedFrame> prepared;
  std::vector<HoloLiftPhase1TrajectoryFrameView> views;
  for (std::size_t i = 0; i < coords.size(); ++i) {
    auto token = prepare_hololift_phase1_frame(*context, *restored, i, i,
        restored->frames()[i].box_matrix, kAtomOrder, coords[i]);
    expect(token.has_value(), "policy restored preparation failed");
    prepared.push_back(std::move(*token));
    views.push_back({i, kAtomOrder, coords[i]});
  }
  HoloLiftPhase1Config config;
  config.supported_evidence_mismatch_weight = 0;  // admission mechanics only
  config.physical_transition_weight = 4;
  auto strict = solve_hololift_phase1_temporal_path(*restored, prepared, config);
  expect(strict.has_value() && strict->frames[1].candidate_index != carried_index,
         "strict default admitted a policy-only candidate");
  config.allow_policy_admitted_candidates = true;
  config.ordinal_rank_saturated = true;
  config.evidence_absent_zero_rank = true;
  auto policy = solve_hololift_phase1_temporal_path(*restored, prepared, config);
  expect(policy.has_value() && policy->frames[1].candidate_index == carried_index &&
             policy->frames[1].selected_candidate_provider_unsupported &&
             policy->frames[1].preceding_transition_provider_unsupported &&
             policy->frames[2].preceding_transition_provider_unsupported &&
             !policy->frames[0].preceding_transition_provider_unsupported &&
             policy->segments[0].provider_unsupported_selected_candidate_count == 1 &&
             policy->segments[0].provider_unsupported_transition_count == 2 &&
             !policy->certificate.all_selected_candidates_hard_feasible,
         "diagnostic carry admission or adjacent support flags are wrong");
  auto audited = audit_hololift_phase1_all_atom_reconstruction(
      *context, *restored, prepared, *policy, views);
  if (!audited) std::cerr << audited.error() << '\n';
  expect(audited.has_value() &&
             audited->result_if_valid()->segments[0].atom_temporal_injectivity_audit_passed &&
             !audited->result_if_valid()->segments[0].temporal_lift_locally_certified,
         "geometry-only pass incorrectly certified unsupported carry");
  const auto json_path = artifact_base.string() + ".policy.json";
  auto written = write_hololift_phase1_result_json(*restored, prepared, *audited, json_path);
  if (!written) std::cerr << written.error().message << '\n';
  expect(written.has_value(), "policy sealed result was not replayable");
  std::ifstream stream(json_path);
  const std::string json((std::istreambuf_iterator<char>(stream)), {});
  expect(json.find("\"allow_policy_admitted_candidates\":true") != std::string::npos &&
             json.find("\"ordinal_rank_saturated\":true") != std::string::npos &&
             json.find("\"evidence_absent_zero_rank\":true") != std::string::npos &&
             json.find("unique_evidence_mismatch_weight") != std::string::npos &&
             json.find("supported_evidence_mismatch_weight") == std::string::npos &&
             json.find("\"provider_unsupported_transition_count\":2") != std::string::npos,
         "policy result omitted flags/objective/provider support provenance, or wrote an unset supported-evidence weight");
  std::cout << "PASS hololift_policy_features (strict admission, carry provenance, binary roundtrip, K0/K1/K2 objective, supported-evidence K=2, certification)\n";
}

int main(int argc, char **argv) {
  expect(argc > 1, "result artifact path required");
  check_policy_features(argv[1]);
  HoloLiftFrameRecord contradicted_gap;
  contradicted_gap.provenance.status = HoloLiftFrameStatus::Contradicted;
  contradicted_gap.provenance.evidence_state =
      HoloLiftEvidenceState::Contradicted;
  expect(hololift_phase1_frame_use(contradicted_gap) ==
             HoloLiftPhase1FrameUse::ObservationGapContradictedEvidence,
         "contradicted evidence did not take precedence over generic gaps");
  expect(argc == 2, "Phase 1 result artifact path was not provided");
  const std::array<double, 9> transport_reference_box{
      10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const auto identity_transport = find_hololift_lattice_basis_transport(
      transport_reference_box, transport_reference_box,
      HoloLiftLatticeBasisPolicy::RequireBasisContinuous, 0.35, 1.0e-12);
  expect(identity_transport.has_value() &&
             identity_transport->identity_fast_path &&
             !identity_transport->local_search_completed &&
             !identity_transport->applied,
         "identity lattice transport did not use the constant-time fast path");
  const std::array<std::array<std::int64_t, 9>, 4> basis_changes{
      std::array<std::int64_t, 9>{1, 1, 0, 0, 1, 0, 0, 0, 1},
      std::array<std::int64_t, 9>{1, 0, 0, 0, 1, -2, 0, 0, 1},
      std::array<std::int64_t, 9>{0, 1, 0, 1, 0, 0, 0, 0, 1},
      std::array<std::int64_t, 9>{-1, 0, 0, 0, 1, 0, 0, 0, 1}};
  for (const auto &basis_change : basis_changes) {
    const auto input_box =
        integer_left_multiply_box(basis_change, transport_reference_box);
    const auto transport = find_hololift_lattice_basis_transport(
        transport_reference_box, input_box,
        HoloLiftLatticeBasisPolicy::TransportUnimodularBasis, 0.35, 1.0e-12);
    expect(transport.has_value() && transport->applied,
           "exact GL(3,Z) fixture transport was not recovered");
    for (std::size_t index = 0; index < input_box.size(); ++index) {
      expect(std::fabs(transport->transported_box[index] -
                       transport_reference_box[index]) < 1.0e-12,
             "basis transport did not preserve the Cartesian lattice");
    }
    const HoloLiftLatticeImage input_image{3, -2, 5};
    const auto reference_image =
        hololift_lattice_image_to_reference(input_image, *transport);
    const auto replayed_image =
        reference_image
            ? hololift_lattice_image_from_reference(*reference_image,
                                                    *transport)
            : std::expected<HoloLiftLatticeImage, std::string>(
                  std::unexpected(reference_image.error()));
    expect(replayed_image.has_value() && *replayed_image == input_image,
           "basis image transport is not exactly invertible");
  }
  const std::array<double, 9> ambiguous_reference_box{
      10.0, 5.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  expect(!find_hololift_lattice_basis_transport(
              ambiguous_reference_box, transport_reference_box,
              HoloLiftLatticeBasisPolicy::TransportUnimodularBasis, 0.5,
              1.0e-12)
              .has_value(),
         "ambiguous lattice-basis transport did not fail closed");
  const std::array<double, 9> large_physical_shear_box{
      10.0, 6.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  expect(!find_hololift_lattice_basis_transport(
              transport_reference_box, large_physical_shear_box,
              HoloLiftLatticeBasisPolicy::TransportUnimodularBasis, 0.35,
              1.0e-12)
              .has_value(),
         "automatic basis transport misclassified large physical shear");

  std::vector<Coordinates> coordinates;
  HoloLiftHash128 assignment_a_identity;
  HoloLiftHash128 assignment_b_identity;
  auto builder =
      make_builder(coordinates, assignment_a_identity, assignment_b_identity);
  auto finalized = finalize_hololift_observation_store(std::move(builder));
  expect(finalized.has_value(), "Phase 1 fixture failed store validation");
  auto store = std::move(*finalized);

  auto preparation = make_hololift_phase1_preparation_context(store);
  expect(preparation.has_value(), "failed to prepare Phase 1 epoch cache");
  std::vector<HoloLiftPhase1VerifiedFrame> verified;
  for (std::size_t idx = 0; idx < store.frames().size(); ++idx) {
    auto token = prepare_hololift_phase1_frame(
        *preparation, store, idx, idx, store.frames()[idx].box_matrix,
        kAtomOrder, coordinates[idx]);
    expect(token.has_value(), "valid Phase 1 frame verification failed");
    verified.push_back(std::move(*token));
  }

  auto tampered = coordinates[0];
  tampered[0][0] += 0.125;
  expect(!verify_hololift_phase1_frame(
              store, 0, 0, store.frames()[0].box_matrix, kAtomOrder, tampered)
              .has_value(),
         "tampered trajectory coordinates passed Phase 1 verification");

  HoloLiftPhase1Config config;
  config.supported_evidence_mismatch_weight = 0;  // rank and transition terms only
  config.ordinal_rank_weight = 1.0;
  config.physical_transition_weight = 4.0;
  auto result = solve_hololift_phase1_temporal_path(store, verified, config);
  expect(result.has_value(), "Phase 1 temporal solve failed");
  expect(result->segments.size() == 2 &&
             result->temporal_observation_frames == 4 &&
             result->observation_gap_frames == 2 &&
             result->changed_from_framewise_frames == 1,
         "Phase 1 segment or frame counters are wrong");
  expect(result->source_trajectory_coverage_known &&
             !result->imported_observation_coverage_complete &&
             !result->source_trajectory_coverage_complete &&
             !result->temporal_coverage_complete &&
             result->all_segment_objectives_unique &&
             result->all_observation_bands_complete,
         "Phase 1 result scope flags are wrong");
  expect(std::fabs(result->objective - 1.0) < 1.0e-12,
         "Phase 1 objective is wrong");
  const auto selected_middle = result->frames[1].candidate_index;
  expect(selected_middle != HOLOLIFT_NO_INDEX &&
             store.candidates()[store.frames()[1].selected_candidate_index]
                     .observation_class ==
                 HoloLiftObservationClass::WeakObservation &&
             store.candidates()[selected_middle].assignment_identity ==
                 assignment_a_identity &&
             result->frames[1].changed_from_framewise,
         "Phase 1 did not suppress the framewise image flicker");
  expect(!result->frames[0].pinned_strong_bounded_anchor &&
             !result->frames[2].pinned_strong_bounded_anchor &&
             result->frames[3].use ==
                 HoloLiftPhase1FrameUse::ObservationGapAmbiguousInternalLift &&
             result->frames[4].segment_index == 1 &&
             store.candidates()[result->frames[4].candidate_index]
                     .assignment_identity == assignment_b_identity &&
             result->frames[5].use ==
                 HoloLiftPhase1FrameUse::ObservationGapNoCertifiedBand,
         "Phase 1 bounded-anchor, gap, or segment policy is wrong");
  expect(result->segments[0].unique_on_configured_retained_state_graph &&
             !result->segments[0].anchors_hard_pinned &&
             result->segments[0].second_best_objective >
                 result->segments[0].objective &&
             result->segments[0].absolute_path_gap > 0.0 &&
             result->segments[0].peak_layer_state_count == 2 &&
             result->segments[0].compact_backpointer_bytes_estimate > 0,
         "Phase 1 conditional uniqueness or path-gap report is wrong");
  std::vector<double> exhaustive_path_costs;
  const std::size_t first_candidate = store.frames()[0].candidates.begin;
  const std::size_t last_candidate = store.frames()[2].candidates.begin;
  for (std::size_t local = 0;
       local < store.frames()[1].candidates.count; ++local) {
    const std::size_t middle_candidate =
        store.frames()[1].candidates.begin + local;
    double path_cost =
        brute_force_emission(store, 0, first_candidate, config) +
        brute_force_emission(store, 1, middle_candidate, config) +
        brute_force_emission(store, 2, last_candidate, config);
    path_cost += brute_force_transition_cost(
        store, verified[0], verified[1], first_candidate, middle_candidate,
        config);
    path_cost += brute_force_transition_cost(
        store, verified[1], verified[2], middle_candidate, last_candidate,
        config);
    exhaustive_path_costs.push_back(path_cost);
  }
  std::sort(exhaustive_path_costs.begin(), exhaustive_path_costs.end());
  expect(exhaustive_path_costs.size() == 2 &&
             std::fabs(exhaustive_path_costs[0] -
                       result->segments[0].objective) < 1.0e-10 &&
             std::fabs(exhaustive_path_costs[1] -
                       result->segments[0].second_best_objective) < 1.0e-10,
          "physical DP disagrees with exhaustive path/gauge enumeration");

  constexpr std::size_t kRollingShortFrames = 8;
  constexpr std::size_t kRollingLongFrames = 128;
  constexpr std::size_t kRollingCandidates = 2;
  constexpr std::size_t kRollingComponents = 2;
  constexpr std::size_t kExpectedRollingPeakImages =
      2 * kRollingCandidates * kRollingComponents;
  const auto rolling_short =
      run_rolling_memory_fixture(kRollingShortFrames, config);
  const auto rolling_long =
      run_rolling_memory_fixture(kRollingLongFrames, config);
  expect(rolling_short.pretransformed_candidate_component_images ==
                 kRollingShortFrames * kRollingCandidates *
                     kRollingComponents &&
             rolling_long.pretransformed_candidate_component_images ==
                 kRollingLongFrames * kRollingCandidates *
                     kRollingComponents,
         "logical pretransformed image work counter changed");
  expect(rolling_short.peak_live_pretransformed_candidate_component_images ==
                 kExpectedRollingPeakImages &&
             rolling_long.peak_live_pretransformed_candidate_component_images ==
                 kExpectedRollingPeakImages &&
             rolling_short
                     .peak_live_pretransformed_candidate_component_image_bytes_estimate ==
                 kExpectedRollingPeakImages *
                     sizeof(HoloLiftLatticeImage) &&
             rolling_long
                     .peak_live_pretransformed_candidate_component_image_bytes_estimate ==
                 rolling_short
                     .peak_live_pretransformed_candidate_component_image_bytes_estimate,
         "rolling pretransformed image peak scales with trajectory length");
  expect(rolling_short.rolling_pretransformed_candidate_layer_builds ==
                 kRollingShortFrames &&
             rolling_long.rolling_pretransformed_candidate_layer_builds ==
                 kRollingLongFrames &&
             rolling_long
                     .peak_live_pretransformed_candidate_component_images <
                 rolling_long.pretransformed_candidate_component_images,
         "rolling layer build accounting does not detect trajectory caching");

  HoloLiftPhase1Config parallel_config = config;
  parallel_config.dp_worker_count = 4;
  const auto parallel_result =
      solve_hololift_phase1_temporal_path(store, verified, parallel_config);
  expect(parallel_result.has_value() &&
             std::fabs(parallel_result->objective - result->objective) <
                 1.0e-12 &&
             parallel_result->segments.size() == result->segments.size(),
         "parallel Phase 1 DP changed the objective or segment layout");
  for (std::size_t frame_index = 0; frame_index < result->frames.size();
       ++frame_index) {
    expect(parallel_result->frames[frame_index].candidate_index ==
                   result->frames[frame_index].candidate_index &&
               parallel_result->frames[frame_index]
                       .selected_global_gauge_increment ==
                   result->frames[frame_index]
                       .selected_global_gauge_increment &&
               parallel_result->frames[frame_index]
                       .cumulative_global_gauge ==
                   result->frames[frame_index].cumulative_global_gauge,
           "parallel Phase 1 DP changed path or gauge selection");
  }
  expect(parallel_result->dp_parallel_layers == 0 &&
             parallel_result->dp_worker_pool_creations == 0 &&
             parallel_result->dp_worker_pool_creation_failures == 0 &&
             parallel_result->dp_max_active_workers == 0 &&
             parallel_result->dp_serial_layers == 2,
         "small Phase 1 bands did not remain on the serial path");

  constexpr std::size_t kParallelExecutorFrames = 6;
  constexpr std::size_t kParallelExecutorTransitionLayers =
      kParallelExecutorFrames - 1;
  HoloLiftPhase1Config executor_config = config;
  executor_config.dp_worker_count = 1;
  const auto executor_reference = run_parallel_executor_fixture(
      kParallelExecutorFrames, executor_config);
  expect(executor_reference.dp_parallel_layers == 0 &&
             executor_reference.dp_serial_layers ==
                 kParallelExecutorTransitionLayers &&
             executor_reference.dp_worker_pool_creations == 0 &&
             executor_reference.dp_worker_pool_creation_failures == 0 &&
             executor_reference.dp_max_active_workers == 0,
         "single-worker executor fixture did not remain serial");

  const unsigned reported_hardware = std::thread::hardware_concurrency();
  const std::size_t hardware_worker_limit =
      reported_hardware == 0 ? 1
                             : static_cast<std::size_t>(reported_hardware);
  for (const std::size_t configured_workers :
       std::array<std::size_t, 4>{1, 2, 4, 8}) {
    executor_config.dp_worker_count = configured_workers;
    const auto executor_result = run_parallel_executor_fixture(
        kParallelExecutorFrames, executor_config);
    expect_same_parallel_dp_solution(executor_reference, executor_result);
    const std::size_t expected_active_workers =
        std::min(configured_workers, hardware_worker_limit);
    if (expected_active_workers > 1) {
      expect(executor_result.dp_parallel_layers ==
                     kParallelExecutorTransitionLayers &&
                 executor_result.dp_serial_layers == 0 &&
                 executor_result.dp_worker_pool_creations == 1 &&
                 executor_result.dp_worker_pool_creation_failures == 0 &&
                 executor_result.dp_max_active_workers ==
                     expected_active_workers,
             "large Phase 1 band did not reuse one bounded worker pool");
    } else {
      expect(executor_result.dp_parallel_layers == 0 &&
                 executor_result.dp_serial_layers ==
                     kParallelExecutorTransitionLayers &&
                 executor_result.dp_worker_pool_creations == 0 &&
                 executor_result.dp_worker_pool_creation_failures == 0 &&
                 executor_result.dp_max_active_workers == 0,
             "hardware-conservative executor fallback is inconsistent");
    }
  }

  if (hardware_worker_limit > 1) {
    titan_hololift::testing::force_phase1_dp_pool_creation_failure(true);
    executor_config.dp_worker_count = 4;
    const auto failed_pool_result = run_parallel_executor_fixture(
        kParallelExecutorFrames, executor_config);
    titan_hololift::testing::force_phase1_dp_pool_creation_failure(false);
    expect_same_parallel_dp_solution(executor_reference, failed_pool_result);
    expect(failed_pool_result.dp_parallel_layers == 0 &&
               failed_pool_result.dp_serial_layers ==
                   kParallelExecutorTransitionLayers &&
               failed_pool_result.dp_worker_pool_creations == 0 &&
               failed_pool_result.dp_worker_pool_creation_failures == 1 &&
               failed_pool_result.dp_max_active_workers == 0,
           "worker-pool creation failure did not fall back deterministically");

    titan_hololift::testing::force_phase1_dp_pool_creation_bad_alloc(true);
    const auto bad_alloc_pool_result = run_parallel_executor_fixture(
        kParallelExecutorFrames, executor_config);
    titan_hololift::testing::force_phase1_dp_pool_creation_bad_alloc(false);
    expect_same_parallel_dp_solution(executor_reference,
                                     bad_alloc_pool_result);
    expect(bad_alloc_pool_result.dp_parallel_layers == 0 &&
               bad_alloc_pool_result.dp_serial_layers ==
                   kParallelExecutorTransitionLayers &&
               bad_alloc_pool_result.dp_worker_pool_creations == 0 &&
               bad_alloc_pool_result.dp_worker_pool_creation_failures == 1 &&
               bad_alloc_pool_result.dp_max_active_workers == 0,
           "worker-pool bad_alloc did not fall back deterministically");

    titan_hololift::testing::force_phase1_dp_run_setup_exception(true);
    const auto failed_setup_result = try_run_parallel_executor_fixture(
        kParallelExecutorFrames, executor_config);
    titan_hololift::testing::force_phase1_dp_run_setup_exception(false);
    expect(!failed_setup_result.has_value() &&
               failed_setup_result.error().find(
                   "Phase 1 DP executor setup failed: ") !=
                   std::string::npos,
           "worker task setup exception escaped the solver boundary");

    titan_hololift::testing::force_phase1_dp_worker_exception(true);
    const auto failed_worker_result = try_run_parallel_executor_fixture(
        kParallelExecutorFrames, executor_config);
    titan_hololift::testing::force_phase1_dp_worker_exception(false);
    expect(!failed_worker_result.has_value() &&
               failed_worker_result.error().find(
                   "Phase 1 DP worker task failed: injected Phase 1 DP worker "
                   "task failure") != std::string::npos,
           "worker task exception did not fail closed at the solver boundary");
  }

  HoloLiftPhase1Config tie_config;
  tie_config.supported_evidence_mismatch_weight = 0;
  tie_config.ordinal_rank_weight = 0.0;
  tie_config.physical_transition_weight = 0.0;
  auto tie_result =
      solve_hololift_phase1_temporal_path(store, verified, tie_config);
  expect(tie_result.has_value() &&
             !tie_result->segments[0]
                  .unique_on_configured_retained_state_graph &&
             tie_result->segments[0].optimal_path_count_capped == 2,
         "Phase 1 did not preserve an exactly tied temporal ambiguity");
  std::vector<HoloLiftPhase1TrajectoryFrameView> trajectory_views;
  trajectory_views.reserve(coordinates.size());
  for (std::size_t frame_index = 0; frame_index < coordinates.size();
       ++frame_index) {
    trajectory_views.push_back(
        {frame_index, std::span<const HoloLiftSourceAtomKey>(kAtomOrder),
         std::span<const std::array<double, 3>>(coordinates[frame_index])});
  }
  auto selected_path_audited =
      audit_hololift_phase1_all_atom_reconstruction(
          *preparation, store, verified, *result, trajectory_views);
  auto alternate_path_audited =
      audit_hololift_phase1_all_atom_reconstruction(
          *preparation, store, verified, *tie_result, trajectory_views);
  expect(selected_path_audited.has_value() &&
             alternate_path_audited.has_value() &&
             !std::equal(
                 selected_path_audited->trajectory_audit_digest_sha256()
                     .begin(),
                 selected_path_audited->trajectory_audit_digest_sha256()
                     .end(),
                 alternate_path_audited->trajectory_audit_digest_sha256()
                     .begin()),
         "trajectory audit digest did not bind selected assignment identity");

  auto run_atom_order_audit =
      [&](HoloLiftObservationStoreBuilder order_builder,
          std::vector<Coordinates> order_coordinates,
          std::span<const HoloLiftSourceAtomKey> atom_order) {
        rebind_builder_frames(order_builder, order_coordinates, atom_order);
        auto order_store = finalize_hololift_observation_store(
            std::move(order_builder));
        expect(order_store.has_value(),
               "atom-order regression store failed validation");
        auto order_context =
            make_hololift_phase1_preparation_context(*order_store);
        expect(order_context.has_value(),
               "atom-order regression context failed");
        std::vector<HoloLiftPhase1PreparedFrame> order_prepared;
        std::vector<HoloLiftPhase1TrajectoryFrameView> order_views;
        order_prepared.reserve(order_store->frames().size());
        order_views.reserve(order_store->frames().size());
        for (std::size_t frame_index = 0;
             frame_index < order_store->frames().size(); ++frame_index) {
          auto prepared = prepare_hololift_phase1_frame(
              *order_context, *order_store, frame_index, frame_index,
              order_store->frames()[frame_index].box_matrix, atom_order,
              order_coordinates[frame_index]);
          expect(prepared.has_value(),
                 "atom-order regression frame preparation failed");
          order_prepared.push_back(std::move(*prepared));
          order_views.push_back({frame_index, atom_order,
                                 order_coordinates[frame_index]});
        }
        auto order_result = solve_hololift_phase1_temporal_path(
            *order_store, order_prepared, config);
        expect(order_result.has_value(),
               "atom-order regression temporal solve failed");
        auto order_audited = audit_hololift_phase1_all_atom_reconstruction(
            *order_context, *order_store, order_prepared, *order_result,
            order_views);
        expect(order_audited.has_value(),
               "atom-order regression reconstruction audit failed");
        const auto *order_audited_result =
            order_audited->result_if_valid();
        expect(order_audited_result != nullptr,
               "atom-order regression audit seal is unexpectedly empty");
        return *order_audited_result;
      };

  std::vector<Coordinates> grouped_coordinates;
  HoloLiftHash128 grouped_assignment_a;
  HoloLiftHash128 grouped_assignment_b;
  auto grouped_builder = make_builder(grouped_coordinates,
                                      grouped_assignment_a,
                                      grouped_assignment_b);
  grouped_coordinates[0] =
      Coordinates{std::array<double, 3>{9.8, 1.0, 1.0},
                  std::array<double, 3>{8.8, 1.0, 1.0},
                  std::array<double, 3>{6.0, 5.0, 5.0},
                  std::array<double, 3>{5.0, 5.0, 5.0}};
  grouped_coordinates[1] =
      Coordinates{std::array<double, 3>{0.2, 1.0, 1.0},
                  std::array<double, 3>{9.2, 1.0, 1.0},
                  std::array<double, 3>{6.0, 5.0, 5.0},
                  std::array<double, 3>{5.0, 5.0, 5.0}};
  grouped_coordinates[2] =
      Coordinates{std::array<double, 3>{0.6, 1.0, 1.0},
                  std::array<double, 3>{9.6, 1.0, 1.0},
                  std::array<double, 3>{6.0, 5.0, 5.0},
                  std::array<double, 3>{5.0, 5.0, 5.0}};
  auto interleaved_coordinates = grouped_coordinates;
  for (auto &frame_coordinates : interleaved_coordinates)
    frame_coordinates = permute_coordinates(frame_coordinates);
  std::vector<Coordinates> ignored_coordinates;
  HoloLiftHash128 interleaved_assignment_a;
  HoloLiftHash128 interleaved_assignment_b;
  auto interleaved_builder =
      make_builder(ignored_coordinates, interleaved_assignment_a,
                   interleaved_assignment_b);
  const auto grouped_result = run_atom_order_audit(
      std::move(grouped_builder), grouped_coordinates, kAtomOrder);
  const auto interleaved_result = run_atom_order_audit(
      std::move(interleaved_builder), interleaved_coordinates,
      kInterleavedAtomOrder);
  expect(grouped_result.frames.size() == interleaved_result.frames.size() &&
             grouped_result.reconstruction_audit
                     .independent_space_time_curvature_residuals == 0 &&
             interleaved_result.reconstruction_audit
                     .independent_space_time_curvature_residuals == 0 &&
             grouped_result.reconstruction_audit
                     .atom_temporal_nearest_image_violations ==
                 interleaved_result.reconstruction_audit
                     .atom_temporal_nearest_image_violations,
         "canonical atom audit depends on caller input order");
  for (std::size_t frame_index = 0;
       frame_index < grouped_result.frames.size(); ++frame_index) {
    expect(grouped_result.frames[frame_index].use ==
                   interleaved_result.frames[frame_index].use &&
               grouped_result.frames[frame_index]
                       .selected_global_gauge_increment ==
                   interleaved_result.frames[frame_index]
                       .selected_global_gauge_increment,
            "temporal result changed under atom-order permutation");
  }

  std::vector<Coordinates> concurrent_coordinates;
  HoloLiftHash128 concurrent_assignment_a;
  HoloLiftHash128 concurrent_assignment_b;
  auto concurrent_builder = make_builder(
      concurrent_coordinates, concurrent_assignment_a,
      concurrent_assignment_b);
  concurrent_coordinates = grouped_coordinates;
  rebind_builder_frames(concurrent_builder, concurrent_coordinates,
                        kAtomOrder);
  auto concurrent_store = finalize_hololift_observation_store(
      std::move(concurrent_builder));
  expect(concurrent_store.has_value(),
         "concurrent preparation store failed validation");
  auto concurrent_context =
      make_hololift_phase1_preparation_context(*concurrent_store);
  expect(concurrent_context.has_value(),
         "concurrent preparation context failed");
  std::vector<std::future<
      std::expected<HoloLiftPhase1PreparedFrame, std::string>>>
      preparation_futures;
  preparation_futures.reserve(concurrent_store->frames().size());
  for (std::size_t frame_index = 0;
       frame_index < concurrent_store->frames().size(); ++frame_index) {
    preparation_futures.push_back(std::async(
        std::launch::async,
        [&, frame_index] {
          return prepare_hololift_phase1_frame(
              *concurrent_context, *concurrent_store, frame_index,
              frame_index,
              concurrent_store->frames()[frame_index].box_matrix,
              kAtomOrder, concurrent_coordinates[frame_index]);
        }));
  }
  std::vector<HoloLiftPhase1PreparedFrame> concurrent_prepared;
  concurrent_prepared.reserve(preparation_futures.size());
  for (auto &future : preparation_futures) {
    auto prepared = future.get();
    expect(prepared.has_value(), "concurrent frame preparation failed");
    concurrent_prepared.push_back(std::move(*prepared));
  }
  const auto concurrent_result = solve_hololift_phase1_temporal_path(
      *concurrent_store, concurrent_prepared, config);
  expect(concurrent_result.has_value() &&
             concurrent_result->frames.size() == grouped_result.frames.size() &&
             concurrent_result->certificate
                     .optimal_path_proven_on_complete_bounded_domain ==
                 grouped_result.certificate
                     .optimal_path_proven_on_complete_bounded_domain,
         "concurrent preparation changed the temporal certificate");
  for (std::size_t frame_index = 0;
       frame_index < grouped_result.frames.size(); ++frame_index) {
    expect(concurrent_result->frames[frame_index].candidate_index ==
                   grouped_result.frames[frame_index].candidate_index &&
               concurrent_result->frames[frame_index]
                       .selected_global_gauge_increment ==
                   grouped_result.frames[frame_index]
                       .selected_global_gauge_increment,
           "concurrent preparation changed the selected path or gauge");
  }

  expect(!solve_hololift_phase1_temporal_path(
              store,
              std::span<const HoloLiftPhase1VerifiedFrame>(
                  verified.data(), verified.size() - 1),
              config)
              .has_value(),
         "Phase 1 accepted an incomplete trajectory-binding audit");
  HoloLiftPhase1Config invalid_config = config;
  invalid_config.physical_transition_weight =
      std::numeric_limits<double>::quiet_NaN();
  expect(!solve_hololift_phase1_temporal_path(
              store, verified, invalid_config)
              .has_value(),
         "Phase 1 accepted a non-finite configuration");
  HoloLiftPhase1Config invalid_worker_config = config;
  invalid_worker_config.dp_worker_count = 0;
  expect(!solve_hololift_phase1_temporal_path(
              store, verified, invalid_worker_config)
              .has_value(),
         "Phase 1 accepted a zero DP worker count");

  const std::array<double, 9> cube_box{
      10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const std::array<SingleCoordinates, 2> crossing_coordinates{
      SingleCoordinates{std::array<double, 3>{9.0, 2.0, 3.0},
                        std::array<double, 3>{9.5, 2.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{1.0, 2.0, 3.0},
                        std::array<double, 3>{1.5, 2.0, 3.0}}};
  const std::array<std::array<double, 9>, 2> cube_boxes{cube_box, cube_box};
  const std::array<double, 2> unit_times{0.0, 1.0};
  const std::array<SingleCoordinates, 2> mass_weighted_coordinates{
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{6.0, 2.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{6.0, 2.0, 3.0}}};
  auto mass_weighted_builder = make_single_component_builder(
      mass_weighted_coordinates, cube_boxes, unit_times);
  mass_weighted_builder.exact_atom_masses = {1.0, 3.0};
  auto mass_weighted_store = finalize_hololift_observation_store(
      std::move(mass_weighted_builder));
  expect(mass_weighted_store.has_value(),
         "mass-weighted store failed validation");
  auto mass_weighted_context =
      make_hololift_phase1_preparation_context(*mass_weighted_store);
  expect(mass_weighted_context.has_value(),
         "mass-weighted context failed");
  const auto mass_weighted_frame = prepare_hololift_phase1_frame(
      *mass_weighted_context, *mass_weighted_store, 0, 0, cube_box,
      kSingleAtomOrder, mass_weighted_coordinates[0]);
  expect(mass_weighted_frame.has_value() &&
             mass_weighted_frame->component_weighting() ==
                 HoloLiftPhase1ComponentWeighting::AtomicMass &&
             mass_weighted_frame->component_weights().size() == 1 &&
             std::fabs(mass_weighted_frame->component_weights()[0] - 4.0) <
                 1.0e-12 &&
             std::fabs(mass_weighted_frame
                           ->component_fractional_representatives()[0][0] -
                       0.5) < 1.0e-12,
         "mass-weighted component representative is incorrect");
  auto partial_mass_builder = make_single_component_builder(
      mass_weighted_coordinates, cube_boxes, unit_times);
  partial_mass_builder.exact_atom_masses = {1.0, 0.0};
  auto partial_mass_store = finalize_hololift_observation_store(
      std::move(partial_mass_builder));
  expect(partial_mass_store.has_value(),
         "partial-mass fallback store failed validation");
  auto partial_mass_context =
      make_hololift_phase1_preparation_context(*partial_mass_store);
  expect(partial_mass_context.has_value(),
         "partial-mass fallback context failed");
  const auto partial_mass_frame = prepare_hololift_phase1_frame(
      *partial_mass_context, *partial_mass_store, 0, 0, cube_box,
      kSingleAtomOrder, mass_weighted_coordinates[0]);
  expect(partial_mass_frame.has_value() &&
             partial_mass_frame->component_weighting() ==
                 HoloLiftPhase1ComponentWeighting::AtomCount &&
             std::fabs(partial_mass_frame
                           ->component_fractional_representatives()[0][0] -
                       0.4) < 1.0e-12,
         "incomplete mass metadata did not fall back to atom-count weighting");
  auto crossing_store = finalize_hololift_observation_store(
      make_single_component_builder(crossing_coordinates, cube_boxes,
                                    unit_times));
  if (!crossing_store) {
    std::cerr << "FAIL hololift_phase1_smoke: single-component crossing "
                 "store failed validation: "
              << crossing_store.error().message << '\n';
    return 1;
  }
  auto crossing_context =
      make_hololift_phase1_preparation_context(*crossing_store);
  expect(crossing_context.has_value(),
         "single-component crossing context failed");
  HoloLiftPhase1Config physical_config;
  physical_config.supported_evidence_mismatch_weight = 0;
  physical_config.ordinal_rank_weight = 0.0;
  physical_config.physical_transition_weight = 1.0;
  physical_config.temporal_scale = 1.0;
  physical_config.variance_floor_A2 = 0.0;
  physical_config.time_scaling = HoloLiftPhase1TimeScaling::Diffusive;
  std::vector<HoloLiftPhase1PreparedFrame> crossing_prepared;
  auto crossing_result = solve_single_component(
      *crossing_store, crossing_coordinates, physical_config,
      *crossing_context, crossing_prepared);
  expect(crossing_result.has_value() && crossing_result->segments.size() == 1,
         "single-component crossing solve failed");
  expect(crossing_result->frames[1].selected_global_gauge_increment ==
             HoloLiftLatticeImage{1, 0, 0} &&
             crossing_result->frames[1].cumulative_global_gauge ==
                 HoloLiftLatticeImage{1, 0, 0} &&
             std::fabs(crossing_result->frames[1]
                           .component_displacement_rms_A -
                       2.0) < 1.0e-10,
         "single-component PBC crossing did not recover the global gauge");
  auto reconstructed_crossing = reconstruct_hololift_phase1_frame(
      *crossing_context, *crossing_store, crossing_prepared[1],
      crossing_result->frames[1], kSingleAtomOrder,
      crossing_coordinates[1]);
  expect(reconstructed_crossing.has_value() &&
             std::fabs((*reconstructed_crossing)[0][0] - 11.0) < 1.0e-10,
         "single-component universal-cover reconstruction is discontinuous");
  expect(!crossing_result->certificate.all_atom_reconstruction_audited &&
             !crossing_result->certificate
                  .coordinates_materialized_and_replay_consistent &&
             !crossing_result->certificate.temporal_lift_locally_certified,
         "solver certified all-atom reconstruction before replay");
  auto tampered_crossing_coordinates = crossing_coordinates;
  tampered_crossing_coordinates[1][0][0] += 0.01;
  const std::array<HoloLiftPhase1TrajectoryFrameView, 2>
      tampered_crossing_views{
          HoloLiftPhase1TrajectoryFrameView{
              0, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
              std::span<const std::array<double, 3>>(
                  tampered_crossing_coordinates[0])},
          HoloLiftPhase1TrajectoryFrameView{
              1, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
              std::span<const std::array<double, 3>>(
                  tampered_crossing_coordinates[1])}};
  const auto tampered_crossing_audit =
      audit_hololift_phase1_all_atom_reconstruction(
          *crossing_context, *crossing_store, crossing_prepared,
          *crossing_result, tampered_crossing_views);
  expect(!tampered_crossing_audit.has_value() &&
             !crossing_result->certificate.all_atom_reconstruction_audited,
         "all-atom audit accepted tampered trajectory coordinates");
  const std::array<HoloLiftPhase1TrajectoryFrameView, 2> crossing_views{
      HoloLiftPhase1TrajectoryFrameView{
          0, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(crossing_coordinates[0])},
      HoloLiftPhase1TrajectoryFrameView{
          1, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(crossing_coordinates[1])}};
  auto crossing_audited = audit_hololift_phase1_all_atom_reconstruction(
      *crossing_context, *crossing_store, crossing_prepared,
      *crossing_result, crossing_views);
  expect(crossing_audited.has_value(),
         "all-atom universal-cover reconstruction audit failed");
  const auto *crossing_audited_result_ptr =
      crossing_audited->result_if_valid();
  expect(crossing_audited_result_ptr != nullptr &&
             crossing_audited_result_ptr->reconstruction_audit
                     .temporal_frames_audited == 2 &&
             crossing_audited_result_ptr->reconstruction_audit
                     .atoms_audited == 4,
         "all-atom universal-cover reconstruction audit failed");
  const auto &crossing_audited_result = *crossing_audited_result_ptr;
  expect(crossing_audited_result.temporal_coverage_complete &&
             crossing_audited_result.certificate
                 .lattice_basis_transport_accepted_under_policy &&
             crossing_audited_result.certificate
                 .lattice_basis_transport_provenance_sufficient &&
             crossing_audited_result.certificate
                 .gauge_replay_arithmetic_closed &&
             crossing_audited_result.certificate
                 .atom_temporal_injectivity_audited &&
             crossing_audited_result.certificate
                 .atom_temporal_injectivity_audit_passed &&
             crossing_audited_result.certificate
                 .independent_space_time_cochain_audited &&
             crossing_audited_result.certificate
                 .independent_space_time_cochain_closed &&
             crossing_audited_result.certificate
                 .component_representative_injectivity_audit_passed &&
             crossing_audited_result.certificate
                 .selected_transition_time_reversal_audited &&
             crossing_audited_result.certificate
                 .selected_transition_time_reversal_consistent &&
             crossing_audited_result.certificate
                 .all_atom_reconstruction_audited &&
             crossing_audited_result.certificate
                 .coordinates_materialized_and_replay_consistent &&
             crossing_audited_result.certificate
                 .temporal_lift_locally_certified &&
             crossing_audited_result.certificate
                 .optimal_path_exact_on_retained_graph &&
             crossing_audited_result.certificate
                 .optimal_path_proven_on_complete_bounded_domain,
         "single-component temporal certificate is incomplete");

  constexpr std::int64_t kCancellationImage = 100000000;
  const std::array<SingleCoordinates, 2> cancellation_coordinates{
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{2.5, 2.0, 3.0}},
      SingleCoordinates{
          std::array<double, 3>{
              10.0 * static_cast<double>(kCancellationImage) + 2.0,
              2.0, 3.0},
          std::array<double, 3>{
              10.0 * static_cast<double>(kCancellationImage) + 2.5,
              2.0, 3.0}}};
  auto cancellation_store = finalize_hololift_observation_store(
      make_single_component_builder(cancellation_coordinates, cube_boxes,
                                    unit_times));
  expect(cancellation_store.has_value(),
         "quadratic-cancellation store failed validation");
  auto cancellation_context =
      make_hololift_phase1_preparation_context(*cancellation_store);
  expect(cancellation_context.has_value(),
         "quadratic-cancellation context failed");
  std::vector<HoloLiftPhase1PreparedFrame> cancellation_prepared;
  auto cancellation_result = solve_single_component(
      *cancellation_store, cancellation_coordinates, physical_config,
      *cancellation_context, cancellation_prepared);
  expect(cancellation_result.has_value() &&
             cancellation_result->frames[1]
                     .selected_global_gauge_increment ==
                 HoloLiftLatticeImage{-kCancellationImage, 0, 0} &&
             cancellation_result->frames[1]
                 .quadratic_direct_residual_fallback &&
             cancellation_result->quadratic_direct_residual_fallbacks == 1 &&
             std::fabs(cancellation_result->objective) < 1.0e-30,
         "quadratic cancellation did not use the direct residual fallback");
  const auto first_crossing_audit =
      crossing_audited_result.reconstruction_audit;
  auto repeated_crossing_audited =
      audit_hololift_phase1_all_atom_reconstruction(
          *crossing_context, *crossing_store, crossing_prepared,
          *crossing_result, crossing_views);
  expect(repeated_crossing_audited.has_value(),
         "repeated all-atom selected-path audit failed");
  const auto *repeated_crossing_result =
      repeated_crossing_audited->result_if_valid();
  expect(repeated_crossing_result != nullptr &&
             repeated_crossing_result->reconstruction_audit
                     .atom_temporal_transitions_audited ==
                 first_crossing_audit.atom_temporal_transitions_audited &&
             repeated_crossing_result->reconstruction_audit
                     .independent_space_time_edges_audited ==
                 first_crossing_audit.independent_space_time_edges_audited &&
             std::equal(
                 crossing_audited->trajectory_audit_digest_sha256().begin(),
                 crossing_audited->trajectory_audit_digest_sha256().end(),
                 repeated_crossing_audited
                     ->trajectory_audit_digest_sha256()
                     .begin()),
         "all-atom selected-path audit is not idempotent");

  const std::array<SingleCoordinates, 2> half_cell_coordinates{
      SingleCoordinates{std::array<double, 3>{1.0, 2.0, 3.0},
                        std::array<double, 3>{1.5, 2.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{6.0, 2.0, 3.0},
                        std::array<double, 3>{6.5, 2.0, 3.0}}};
  auto half_cell_store = finalize_hololift_observation_store(
      make_single_component_builder(half_cell_coordinates, cube_boxes,
                                    unit_times));
  expect(half_cell_store.has_value(), "half-cell store failed validation");
  auto half_cell_context =
      make_hololift_phase1_preparation_context(*half_cell_store);
  expect(half_cell_context.has_value(), "half-cell context failed");
  std::vector<HoloLiftPhase1PreparedFrame> half_cell_prepared;
  auto half_cell_result = solve_single_component(
      *half_cell_store, half_cell_coordinates, physical_config,
      *half_cell_context, half_cell_prepared);
  expect(half_cell_result.has_value() &&
             half_cell_result->frames[1]
                 .global_gauge_increment_ambiguous &&
             half_cell_result->segments[0]
                     .component_injectivity_uncertified > 0,
         "half-cell fixture did not expose temporal ambiguity");
  const std::array<HoloLiftPhase1TrajectoryFrameView, 2> half_cell_views{
      HoloLiftPhase1TrajectoryFrameView{
          0, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(half_cell_coordinates[0])},
      HoloLiftPhase1TrajectoryFrameView{
          1, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(half_cell_coordinates[1])}};
  auto half_cell_audited = audit_hololift_phase1_all_atom_reconstruction(
      *half_cell_context, *half_cell_store, half_cell_prepared,
      *half_cell_result, half_cell_views);
  expect(half_cell_audited.has_value(), "half-cell audit failed");
  const auto *half_cell_audited_result =
      half_cell_audited->result_if_valid();
  expect(half_cell_audited_result != nullptr &&
             half_cell_audited_result->certificate
                 .coordinates_materialized_and_replay_consistent &&
             half_cell_audited_result->certificate
                 .atom_temporal_injectivity_audited &&
             !half_cell_audited_result->certificate
                  .atom_temporal_injectivity_audit_passed &&
             !half_cell_audited_result->certificate
                  .temporal_lift_locally_certified,
         "half-cell ambiguity was over-certified as a temporal lift");

  const std::array<SingleCoordinates, 2> reverse_coordinates{
      crossing_coordinates[1], crossing_coordinates[0]};
  auto reverse_store = finalize_hololift_observation_store(
      make_single_component_builder(reverse_coordinates, cube_boxes,
                                    unit_times));
  expect(reverse_store.has_value(), "reverse crossing store failed");
  auto reverse_context =
      make_hololift_phase1_preparation_context(*reverse_store);
  expect(reverse_context.has_value(), "reverse crossing context failed");
  std::vector<HoloLiftPhase1PreparedFrame> reverse_prepared;
  auto reverse_result = solve_single_component(
      *reverse_store, reverse_coordinates, physical_config, *reverse_context,
      reverse_prepared);
  expect(reverse_result.has_value() &&
             reverse_result->frames[1].selected_global_gauge_increment ==
                 HoloLiftLatticeImage{-1, 0, 0} &&
             std::fabs(reverse_result->objective - crossing_result->objective) <
                 1.0e-10,
         "forward/reverse crossing is not gauge antisymmetric");

  const std::array<double, 9> deformed_box{
      12.0, 0.0, 0.0, 2.0, 9.0, 0.0, 0.0, 0.0, 11.0};
  const std::array<std::array<double, 9>, 2> npt_boxes{cube_box,
                                                       deformed_box};
  const std::array<SingleCoordinates, 2> npt_coordinates{
      SingleCoordinates{std::array<double, 3>{2.5, 3.0, 4.0},
                        std::array<double, 3>{3.0, 3.0, 4.0}},
      SingleCoordinates{std::array<double, 3>{3.6, 2.7, 4.4},
                        std::array<double, 3>{4.2, 2.7, 4.4}}};
  auto npt_store = finalize_hololift_observation_store(
      make_single_component_builder(npt_coordinates, npt_boxes, unit_times));
  if (!npt_store) {
    std::cerr << "FAIL hololift_phase1_smoke: NPT affine fixture failed "
                 "validation: "
              << npt_store.error().message << '\n';
    return 1;
  }
  auto npt_context = make_hololift_phase1_preparation_context(*npt_store);
  expect(npt_context.has_value(), "NPT affine context failed");
  std::vector<HoloLiftPhase1PreparedFrame> npt_prepared;
  auto npt_result = solve_single_component(
      *npt_store, npt_coordinates, physical_config, *npt_context,
      npt_prepared);
  expect(npt_result.has_value() && std::fabs(npt_result->objective) < 1.0e-20 &&
             npt_result->frames[1].selected_global_gauge_increment ==
                 HoloLiftLatticeImage{0, 0, 0},
         "pure NPT scale/shear generated a spurious peculiar displacement");

  const std::array<double, 9> basis_remapped_box{
      10.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const std::array<std::array<double, 9>, 2> frame_remapped_boxes{
      cube_box, basis_remapped_box};
  const std::array<SingleCoordinates, 2> stationary_coordinates{
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{2.5, 2.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{2.5, 2.0, 3.0}}};
  auto frame_remapped_store = finalize_hololift_observation_store(
      make_single_component_builder(stationary_coordinates,
                                    frame_remapped_boxes, unit_times));
  expect(frame_remapped_store.has_value(),
         "frame-dependent basis-remap fixture failed validation");
  auto frame_remapped_context =
      make_hololift_phase1_preparation_context(*frame_remapped_store);
  expect(frame_remapped_context.has_value(),
         "frame-dependent basis-remap context failed");
  std::vector<HoloLiftPhase1PreparedFrame> frame_remapped_prepared;
  auto rejected_basis_remap = solve_single_component(
      *frame_remapped_store, stationary_coordinates, physical_config,
      *frame_remapped_context, frame_remapped_prepared);
  expect(rejected_basis_remap.has_value() &&
             rejected_basis_remap->segments.size() == 2 &&
             rejected_basis_remap->basis_transport_transition_gaps == 1 &&
             rejected_basis_remap->basis_discontinuity_transition_gaps == 1 &&
             rejected_basis_remap->frames[1].preceding_transition_gap ==
                 HoloLiftPhase1TransitionGap::BasisDiscontinuity &&
             rejected_basis_remap->frames[1].cumulative_global_gauge ==
                 HoloLiftLatticeImage{},
         "basis discontinuity did not create an explicit gauge-reset gap");

  HoloLiftPhase1Config transported_basis_config = physical_config;
  transported_basis_config.lattice_basis_policy =
      HoloLiftLatticeBasisPolicy::TransportUnimodularBasis;
  auto transported_basis_result = solve_single_component(
      *frame_remapped_store, stationary_coordinates, transported_basis_config,
      *frame_remapped_context, frame_remapped_prepared);
  const std::array<std::int64_t, 9> expected_basis_transport{
      1, -1, 0, 0, 1, 0, 0, 0, 1};
  expect(transported_basis_result.has_value() &&
             std::fabs(transported_basis_result->objective) < 1.0e-20 &&
             transported_basis_result->frames[1]
                     .lattice_basis_transport_to_segment_reference ==
                 expected_basis_transport &&
             transported_basis_result->frames[1]
                 .lattice_basis_transport_applied &&
             transported_basis_result->frames[1]
                     .cumulative_global_gauge_input_basis ==
                 HoloLiftLatticeImage{0, 0, 0},
         "unimodular basis transport did not remove false physical motion");
  const std::array<HoloLiftPhase1TrajectoryFrameView, 2>
      frame_remapped_views{
          HoloLiftPhase1TrajectoryFrameView{
              0, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
              std::span<const std::array<double, 3>>(stationary_coordinates[0])},
          HoloLiftPhase1TrajectoryFrameView{
              1, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
              std::span<const std::array<double, 3>>(stationary_coordinates[1])}};
  auto transported_basis_audited =
      audit_hololift_phase1_all_atom_reconstruction(
          *frame_remapped_context, *frame_remapped_store,
          frame_remapped_prepared, *transported_basis_result,
          frame_remapped_views);
  expect(transported_basis_audited.has_value(),
         "experimental basis transport audit failed");
  const auto *transported_basis_audited_result =
      transported_basis_audited->result_if_valid();
  expect(transported_basis_audited_result != nullptr &&
             transported_basis_audited_result->certificate
                 .lattice_basis_transport_accepted_under_policy &&
             !transported_basis_audited_result->certificate
                  .lattice_basis_transport_provenance_sufficient &&
             transported_basis_audited_result->certificate
                 .atom_temporal_injectivity_audit_passed &&
             transported_basis_audited_result->certificate
                 .independent_space_time_cochain_closed &&
             !transported_basis_audited_result->certificate
                  .temporal_lift_locally_certified,
         "experimental basis transport received incorrect certificate scope");

  const std::array<SingleCoordinates, 2> drift_coordinates{
      SingleCoordinates{std::array<double, 3>{1.0, 2.0, 3.0},
                        std::array<double, 3>{1.5, 2.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{2.0, 2.0, 3.0},
                        std::array<double, 3>{2.5, 2.0, 3.0}}};
  const std::array<double, 2> long_times{0.0, 4.0};
  auto short_dt_store = finalize_hololift_observation_store(
      make_single_component_builder(drift_coordinates, cube_boxes,
                                    unit_times));
  auto long_dt_store = finalize_hololift_observation_store(
      make_single_component_builder(drift_coordinates, cube_boxes,
                                    long_times));
  expect(short_dt_store.has_value() && long_dt_store.has_value(),
         "variable-timestep fixtures failed validation");
  auto short_dt_context =
      make_hololift_phase1_preparation_context(*short_dt_store);
  auto long_dt_context =
      make_hololift_phase1_preparation_context(*long_dt_store);
  expect(short_dt_context.has_value() && long_dt_context.has_value(),
         "variable-timestep contexts failed");
  std::vector<HoloLiftPhase1PreparedFrame> short_dt_prepared;
  std::vector<HoloLiftPhase1PreparedFrame> long_dt_prepared;
  auto short_dt_result = solve_single_component(
      *short_dt_store, drift_coordinates, physical_config, *short_dt_context,
      short_dt_prepared);
  auto long_dt_result = solve_single_component(
      *long_dt_store, drift_coordinates, physical_config, *long_dt_context,
      long_dt_prepared);
  expect(short_dt_result.has_value() && long_dt_result.has_value() &&
             std::fabs(short_dt_result->objective /
                           long_dt_result->objective -
                       4.0) < 1.0e-10,
         "diffusive transition did not scale with the actual timestep");

  const std::array<double, 9> transformed_box{
      10.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  const std::array<std::array<double, 9>, 2> transformed_boxes{
      transformed_box, transformed_box};
  const std::array<SingleCoordinates, 2> transformed_coordinates{
      SingleCoordinates{std::array<double, 3>{9.0, 12.0, 3.0},
                        std::array<double, 3>{9.5, 12.0, 3.0}},
      SingleCoordinates{std::array<double, 3>{1.0, 2.0, 3.0},
                        std::array<double, 3>{1.5, 2.0, 3.0}}};
  auto transformed_store = finalize_hololift_observation_store(
      make_single_component_builder(transformed_coordinates,
                                    transformed_boxes, unit_times));
  expect(transformed_store.has_value(),
         "unimodular basis fixture failed validation");
  auto transformed_context =
      make_hololift_phase1_preparation_context(*transformed_store);
  expect(transformed_context.has_value(),
         "unimodular basis context failed");
  std::vector<HoloLiftPhase1PreparedFrame> transformed_prepared;
  auto transformed_result = solve_single_component(
      *transformed_store, transformed_coordinates, physical_config,
      *transformed_context, transformed_prepared);
  expect(transformed_result.has_value() &&
             std::fabs(transformed_result->objective -
                       crossing_result->objective) < 1.0e-10 &&
             transformed_result->frames[1].selected_global_gauge_increment ==
                 HoloLiftLatticeImage{1, 0, 0},
         "physical transition changed under a unimodular lattice basis");
  const std::array<HoloLiftPhase1TrajectoryFrameView, 2> transformed_views{
      HoloLiftPhase1TrajectoryFrameView{
          0, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(transformed_coordinates[0])},
      HoloLiftPhase1TrajectoryFrameView{
          1, std::span<const HoloLiftSourceAtomKey>(kSingleAtomOrder),
          std::span<const std::array<double, 3>>(transformed_coordinates[1])}};
  auto transformed_audited = audit_hololift_phase1_all_atom_reconstruction(
      *transformed_context, *transformed_store, transformed_prepared,
      *transformed_result, transformed_views);
  expect(transformed_audited.has_value(),
         "unimodular transformed-path audit failed");
  const auto *transformed_audited_result =
      transformed_audited->result_if_valid();
  expect(transformed_audited_result != nullptr &&
             transformed_store
                     ->candidates()[transformed_audited_result->frames[0]
                                        .candidate_index]
                     .assignment_identity ==
                 crossing_store
                     ->candidates()[crossing_audited_result.frames[0]
                                        .candidate_index]
                     .assignment_identity &&
             !std::equal(
                 transformed_audited->trajectory_audit_digest_sha256()
                     .begin(),
                 transformed_audited->trajectory_audit_digest_sha256().end(),
                 crossing_audited->trajectory_audit_digest_sha256().begin()),
         "trajectory audit digest did not bind exact frame bindings");

  const std::filesystem::path result_path = argv[1];
  const auto fresh_crossing_audit = [&]() {
    auto sealed = audit_hololift_phase1_all_atom_reconstruction(
        *crossing_context, *crossing_store, crossing_prepared,
        *crossing_result, crossing_views);
    expect(sealed.has_value(), "failed to create a fresh audit seal");
    return sealed;
  };

  auto move_construct_source = fresh_crossing_audit();
  expect(move_construct_source->valid() &&
             move_construct_source->result_if_valid() != nullptr,
         "fresh audit seal is unexpectedly invalid before move construction");
  HoloLiftPhase1AuditedResult move_construct_target(
      std::move(*move_construct_source));
  expect(!move_construct_source->valid() &&
             move_construct_source->result_if_valid() == nullptr &&
             move_construct_target.valid() &&
             move_construct_target.result_if_valid() != nullptr,
         "audit seal move construction did not leave a safe empty source");
  const auto moved_construct_validation =
      validate_hololift_phase1_audited_result(
          *crossing_store, crossing_prepared, *move_construct_source);
  expect(!moved_construct_validation.has_value() &&
             moved_construct_validation.error().find("moved-from") !=
                 std::string::npos,
         "validator accepted a move-constructed-from audit seal");
  std::filesystem::path moved_construct_path = result_path;
  moved_construct_path += ".moved-construct";
  std::filesystem::remove(moved_construct_path);
  const auto moved_construct_write = write_hololift_phase1_result_json(
      *crossing_store, crossing_prepared, *move_construct_source,
      moved_construct_path);
  expect(!moved_construct_write.has_value() &&
             moved_construct_write.error().commit_state ==
                 HoloLiftArtifactCommitState::NotCommitted &&
             !std::filesystem::exists(moved_construct_path),
         "writer did not reject a move-constructed-from audit seal");

  auto move_assign_source = fresh_crossing_audit();
  auto move_assign_target = fresh_crossing_audit();
  *move_assign_target = std::move(*move_assign_source);
  expect(!move_assign_source->valid() &&
             move_assign_source->result_if_valid() == nullptr &&
             move_assign_target->valid() &&
             move_assign_target->result_if_valid() != nullptr,
         "audit seal move assignment did not leave a safe empty source");
  const auto moved_assign_validation =
      validate_hololift_phase1_audited_result(
          *crossing_store, crossing_prepared, *move_assign_source);
  expect(!moved_assign_validation.has_value() &&
             moved_assign_validation.error().find("moved-from") !=
                 std::string::npos,
         "validator accepted a move-assigned-from audit seal");
  std::filesystem::path moved_assign_path = result_path;
  moved_assign_path += ".moved-assign";
  std::filesystem::remove(moved_assign_path);
  const auto moved_assign_write = write_hololift_phase1_result_json(
      *crossing_store, crossing_prepared, *move_assign_source,
      moved_assign_path);
  expect(!moved_assign_write.has_value() &&
             moved_assign_write.error().commit_state ==
                 HoloLiftArtifactCommitState::NotCommitted &&
             !std::filesystem::exists(moved_assign_path),
         "writer did not reject a move-assigned-from audit seal");

  const auto reject_result_mutation =
      [&](auto mutate, const char *failure_message) {
        auto sealed = fresh_crossing_audit();
        const auto *sealed_result = sealed->result_if_valid();
        expect(sealed_result != nullptr,
               "fresh audit seal became invalid before mutation regression");
        auto &mutable_result =
            const_cast<HoloLiftPhase1Result &>(*sealed_result);
        mutate(mutable_result);
        const auto rejected = write_hololift_phase1_result_json(
            *crossing_store, crossing_prepared, *sealed, result_path);
        expect(!rejected.has_value() &&
                   rejected.error().commit_state ==
                       HoloLiftArtifactCommitState::NotCommitted,
               failure_message);
      };

  reject_result_mutation(
      [](auto &mutated) {
        mutated.certificate.all_atom_reconstruction_audited = false;
      },
      "Phase 1 serializer accepted a contradictory audit certificate");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].atom_temporal_nearest_image_violations += 1;
      },
      "Phase 1 serializer accepted a frame audit-counter mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.segments[0].atom_temporal_transitions_audited += 1;
      },
      "Phase 1 serializer accepted a segment audit-counter mutation");
  reject_result_mutation(
      [](auto &mutated) { mutated.reconstruction_audit.atoms_audited += 1; },
      "Phase 1 serializer accepted a global audit-counter mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].maximum_atom_peculiar_displacement_A += 0.25;
        mutated.segments[0].maximum_atom_peculiar_displacement_A =
            mutated.frames[1].maximum_atom_peculiar_displacement_A;
        mutated.reconstruction_audit.maximum_atom_peculiar_displacement_A =
            mutated.frames[1].maximum_atom_peculiar_displacement_A;
      },
      "Phase 1 serializer accepted a self-consistent audit-maximum mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].maximum_component_representative_error_A += 0.25;
        mutated.segments[0].maximum_component_representative_error_A =
            mutated.frames[1].maximum_component_representative_error_A;
        mutated.reconstruction_audit
            .maximum_component_representative_error_A =
            mutated.frames[1].maximum_component_representative_error_A;
      },
      "Phase 1 serializer accepted a component-audit maximum mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].atom_temporal_nearest_image_violations += 1;
        mutated.segments[0].atom_temporal_nearest_image_violations += 1;
        mutated.reconstruction_audit
            .atom_temporal_nearest_image_violations += 1;
        mutated.segments[0].atom_temporal_injectivity_audit_passed = false;
        mutated.segments[0].temporal_lift_locally_certified = false;
        mutated.certificate.atom_temporal_injectivity_audit_passed = false;
        mutated.certificate.temporal_lift_locally_certified = false;
      },
      "Phase 1 serializer accepted a self-consistent audit hierarchy mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[0].atom_temporal_transitions_audited = 2;
        mutated.segments[0].atom_temporal_transitions_audited += 2;
        mutated.reconstruction_audit.atom_temporal_transitions_audited += 2;
      },
      "Phase 1 serializer accepted transition diagnostics at a segment start");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1]
            .lattice_basis_transport_to_segment_reference[0] =
            std::numeric_limits<std::int64_t>::max();
      },
      "Phase 1 serializer accepted an unbounded basis transform");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].lattice_basis_transport_to_segment_reference =
            {1, 1, 0, 0, 1, 0, 0, 0, 1};
        mutated.frames[1].lattice_basis_transport_applied = true;
        mutated.frames[1].cumulative_global_gauge_input_basis =
            HoloLiftLatticeImage{1, 1, 0};
      },
      "Phase 1 serializer accepted a solver-inconsistent basis transport");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].transition_cost += 0.125;
        mutated.frames[1].cumulative_segment_cost += 0.125;
        mutated.segments[0].objective += 0.125;
        mutated.objective += 0.125;
      },
      "Phase 1 serializer accepted a self-consistent cost mutation");
  reject_result_mutation(
      [](auto &mutated) {
        mutated.frames[1].selected_global_gauge_increment =
            HoloLiftLatticeImage{2, 0, 0};
        mutated.frames[1].cumulative_global_gauge =
            HoloLiftLatticeImage{2, 0, 0};
        mutated.frames[1].cumulative_global_gauge_input_basis =
            HoloLiftLatticeImage{2, 0, 0};
      },
      "Phase 1 serializer accepted a replayable but wrong gauge path");

  auto tampered_digest_audit = fresh_crossing_audit();
  auto digest_bytes =
      tampered_digest_audit->trajectory_audit_digest_sha256();
  auto *mutable_digest = const_cast<std::byte *>(digest_bytes.data());
  mutable_digest[0] ^= std::byte{0x01};
  const auto rejected_digest = write_hololift_phase1_result_json(
      *crossing_store, crossing_prepared, *tampered_digest_audit,
      result_path);
  expect(!rejected_digest.has_value() &&
             rejected_digest.error().commit_state ==
                 HoloLiftArtifactCommitState::NotCommitted,
         "Phase 1 serializer accepted a mutated trajectory audit digest");
  const std::locale original_locale = std::locale();
  std::locale::global(
      std::locale(original_locale, new CommaDecimalPoint));
  const auto locale_independent_write = write_hololift_phase1_result_json(
      *crossing_store, crossing_prepared, *crossing_audited, result_path);
  std::locale::global(original_locale);
  expect(locale_independent_write.has_value() &&
             *locale_independent_write ==
                 HoloLiftArtifactCommitState::
                     CommittedDurabilityConfirmed,
         "Phase 1 result artifact was not written");
  std::ifstream result_input(result_path, std::ios::binary);
  const std::string result_json(std::istreambuf_iterator<char>(result_input),
                                {});
  std::filesystem::path durability_unconfirmed_path = result_path;
  durability_unconfirmed_path += ".durability-unconfirmed";
  std::filesystem::remove(durability_unconfirmed_path);
  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::
          AfterCommitBeforeDurabilityConfirmation);
  const auto durability_unconfirmed_write =
      write_hololift_phase1_result_json(
          *crossing_store, crossing_prepared, *crossing_audited,
          durability_unconfirmed_path);
  detail::set_durable_artifact_test_fault(
      detail::HoloLiftDurableArtifactTestFault::None);
  std::ifstream durability_unconfirmed_input(durability_unconfirmed_path,
                                             std::ios::binary);
  const std::string durability_unconfirmed_json(
      std::istreambuf_iterator<char>(durability_unconfirmed_input), {});
  expect(!durability_unconfirmed_write.has_value() &&
             durability_unconfirmed_write.error().commit_state ==
                 HoloLiftArtifactCommitState::
                     CommittedDurabilityUnconfirmed &&
             durability_unconfirmed_write.error().committed() &&
             durability_unconfirmed_json == result_json,
         "Phase 1 post-commit durability failure lost its final artifact");
  durability_unconfirmed_input.close();
  std::filesystem::remove(durability_unconfirmed_path);
  std::filesystem::path repeated_result_path = result_path;
  repeated_result_path += ".repeat";
  const auto repeated_write = write_hololift_phase1_result_json(
      *crossing_store, crossing_prepared, *repeated_crossing_audited,
      repeated_result_path);
  expect(repeated_write.has_value() &&
             *repeated_write == HoloLiftArtifactCommitState::
                                    CommittedDurabilityConfirmed,
         "repeated sealed Phase 1 artifact was not written");
  std::ifstream repeated_result_input(repeated_result_path, std::ios::binary);
  const std::string repeated_result_json(
      std::istreambuf_iterator<char>(repeated_result_input), {});
  expect(result_json == repeated_result_json,
         "sealed Phase 1 artifact is not byte-deterministic");
  repeated_result_input.close();
  std::filesystem::remove(repeated_result_path);
  expect(result_json.find("titan-hololift-phase1-result") !=
                 std::string::npos &&
             result_json.find("\"audit_seal_version\":1") !=
                 std::string::npos &&
             result_json.find("trajectory_audit_digest_sha256") !=
                 std::string::npos &&
             result_json.find("selected_global_gauge_increment\":[1,0,0]") !=
                 std::string::npos &&
             result_json.find("independent_space_time_cochain_closed\":true") !=
                 std::string::npos &&
             result_json.find("all_atom_reconstruction_audited\":true") !=
                 std::string::npos &&
             result_json.find("relative_cost_tolerance\":9.") !=
                 std::string::npos &&
             result_json.find("global_gauge_increment_ambiguous") !=
                 std::string::npos &&
             result_json.find("ambiguous_gauge_transition_count") !=
                 std::string::npos &&
             result_json.find("phase0_payload_sha256") !=
                 std::string::npos &&
             result_json.find("\"atoms_audited\":2") !=
                 std::string::npos &&
             result_json.find("independent_space_time_edges_audited") !=
                 std::string::npos &&
             result_json.find("wrapped_coordinate_hash") != std::string::npos,
         "Phase 1 result artifact omitted replay-critical fields");
  std::cout << "PASS hololift_phase1_smoke\n";
  return 0;
}
'@ | Set-Content -LiteralPath $src -Encoding ASCII

$compilerPath = (Get-Command $Compiler -ErrorAction Stop).Source
Push-Location $repo
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $compilerPath -std=c++23 -O2 -pthread -Wall -Wextra -Wpedantic -Wconversion `
        -DTITAN_HOLOLIFT_PHASE1_TEST_HOOKS `
        -DTITAN_HOLOLIFT_DURABLE_ARTIFACT_TEST_HOOKS `
        -Wshadow @ExtraCompilerFlags -Isrc $src `
        src\hololift\identity.cpp `
        src\hololift\frame_binding.cpp `
        src\hololift\lattice_transport.cpp `
        src\hololift\phase1.cpp `
        src\hololift\phase1_serialization.cpp `
        src\hololift\serialization.cpp `
        src\hololift\spatial_lift.cpp `
        src\hololift\validation.cpp `
        src\hololift\vibe_contract.cpp `
        src\pbctopo\observation_schema.cpp `
        src\pbctopo\spatial_lift.cpp `
        src\pbctopo\lattice_math.cpp -o $exe
    $compileExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($compileExitCode -ne 0) {
        throw 'Failed to compile HoloLift Phase 1 smoke'
    }
    $artifact = Join-Path $WorkRoot 'hololift_phase1_result.json'
    & $exe $artifact
    if ($LASTEXITCODE -ne 0) {
        throw "HoloLift Phase 1 smoke failed: $LASTEXITCODE"
    }
    try {
        $parsedArtifact = Get-Content -LiteralPath $artifact -Raw |
            ConvertFrom-Json
        if ($parsedArtifact.format -ne 'titan-hololift-phase1-result' -or
            $parsedArtifact.format_version -ne 7 -or
            $parsedArtifact.audit_seal_version -ne 1 -or
            $parsedArtifact.trajectory_audit_digest_sha256.Length -ne 64 -or
            -not $parsedArtifact.certificate.independent_space_time_cochain_closed) {
            throw 'Parsed Phase 1 artifact omitted the v7 sealed-audit contract'
        }
    }
    finally {
        Remove-Item -LiteralPath $artifact -Force -ErrorAction SilentlyContinue
    }
}
finally {
    Pop-Location
}

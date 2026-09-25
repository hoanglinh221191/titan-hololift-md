#include "validation.h"

#include "identity.h"
#include "frame_binding.h"
#include "vibe_contract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace titan_hololift {
namespace {

HoloLiftValidationError make_error(
    HoloLiftValidationCode code, std::string message,
    std::size_t epoch_index = HOLOLIFT_NO_INDEX,
    std::size_t frame_index = HOLOLIFT_NO_INDEX,
    std::size_t candidate_index = HOLOLIFT_NO_INDEX) {
  return {code, epoch_index, frame_index, candidate_index,
          std::move(message)};
}

bool atom_key_less(const HoloLiftSourceAtomKey &lhs,
                   const HoloLiftSourceAtomKey &rhs) noexcept {
  if (lhs.owner != rhs.owner)
    return lhs.owner < rhs.owner;
  return lhs.source_atom_id < rhs.source_atom_id;
}

bool hard_edge_less(const HoloLiftHardEdgeRecord &lhs,
                    const HoloLiftHardEdgeRecord &rhs) noexcept {
  if (lhs.atom_a != rhs.atom_a)
    return lhs.atom_a < rhs.atom_a;
  if (lhs.atom_b != rhs.atom_b)
    return lhs.atom_b < rhs.atom_b;
  return static_cast<std::uint8_t>(lhs.kind) <
         static_cast<std::uint8_t>(rhs.kind);
}

bool valid_hard_graph_source(HoloLiftHardGraphSource source) noexcept {
  switch (source) {
  case HoloLiftHardGraphSource::None:
  case HoloLiftHardGraphSource::ExplicitTopology:
  case HoloLiftHardGraphSource::ValidatedMetadata:
  case HoloLiftHardGraphSource::MixedHard:
    return true;
  }
  return false;
}

bool hard_graph_source_matches_edges(
    HoloLiftHardGraphSource source,
    std::span<const HoloLiftHardEdgeRecord> edges) noexcept {
  bool has_topology = false;
  bool has_metadata = false;
  for (const auto &edge : edges) {
    has_topology = has_topology ||
                   edge.kind == HoloLiftHardEdgeKind::TopologyBond;
    has_metadata = has_metadata ||
                   edge.kind == HoloLiftHardEdgeKind::ValidatedMetadata;
  }
  switch (source) {
  case HoloLiftHardGraphSource::None:
    return !has_topology && !has_metadata;
  case HoloLiftHardGraphSource::ExplicitTopology:
    return has_topology && !has_metadata;
  case HoloLiftHardGraphSource::ValidatedMetadata:
    return !has_topology && has_metadata;
  case HoloLiftHardGraphSource::MixedHard:
    return has_topology && has_metadata;
  }
  return false;
}

bool zero_image(const HoloLiftLatticeImage &image) noexcept {
  return image.x == 0 && image.y == 0 && image.z == 0;
}

bool normalized_gram_is_spd(const std::array<double, 6> &gram) noexcept {
  for (const double value : gram) {
    if (!std::isfinite(value))
      return false;
  }
  const double scale =
      std::max({1.0, std::fabs(gram[0]), std::fabs(gram[1]),
                std::fabs(gram[2]), std::fabs(gram[3]),
                std::fabs(gram[4]), std::fabs(gram[5])});
  const double minor1 = gram[0];
  const double minor2 = gram[0] * gram[3] - gram[1] * gram[1];
  const double determinant =
      gram[0] * (gram[3] * gram[5] - gram[4] * gram[4]) -
      gram[1] * (gram[1] * gram[5] - gram[2] * gram[4]) +
      gram[2] * (gram[1] * gram[4] - gram[2] * gram[3]);
  if (minor1 <= 1.0e-12 * scale ||
      minor2 <= 1.0e-12 * scale * scale ||
      determinant <= 1.0e-12 * scale * scale * scale) {
    return false;
  }
  const double trace = gram[0] + gram[3] + gram[5];
  return std::fabs(trace - 3.0) <= 1.0e-8;
}

std::array<double, 6>
normalized_gram_from_box(const std::array<double, 9> &box_matrix) noexcept {
  auto dot_rows = [&](std::size_t lhs, std::size_t rhs) {
    return box_matrix[lhs * 3] * box_matrix[rhs * 3] +
           box_matrix[lhs * 3 + 1] * box_matrix[rhs * 3 + 1] +
           box_matrix[lhs * 3 + 2] * box_matrix[rhs * 3 + 2];
  };
  std::array<double, 6> gram{
      dot_rows(0, 0), dot_rows(0, 1), dot_rows(0, 2),
      dot_rows(1, 1), dot_rows(1, 2), dot_rows(2, 2)};
  const double scale = (gram[0] + gram[3] + gram[5]) / 3.0;
  if (!std::isfinite(scale) || scale <= 1.0e-18)
    return {};
  for (double &value : gram)
    value /= scale;
  return gram;
}

bool frame_binding_box_ok(const std::array<double, 9> &box_matrix) noexcept {
  for (double value : box_matrix) {
    if (!std::isfinite(value))
      return false;
  }
  const double determinant =
      box_matrix[0] *
          (box_matrix[4] * box_matrix[8] -
           box_matrix[5] * box_matrix[7]) -
      box_matrix[1] *
          (box_matrix[3] * box_matrix[8] -
           box_matrix[5] * box_matrix[6]) +
      box_matrix[2] *
          (box_matrix[3] * box_matrix[7] -
           box_matrix[4] * box_matrix[6]);
  return std::isfinite(determinant) && determinant > 1.0e-18;
}

HoloLiftObservationClass expected_observation_class(
    const HoloLiftCandidateRecord &candidate,
    const HoloLiftFrameEvidence &frame_evidence,
    const HoloLiftFrameProvenance &provenance,
    const HoloLiftFrameSearchDomain &search_domain,
    std::size_t spatial_lift_ambiguities,
    std::size_t spatial_lift_cycle_residuals) noexcept {
  if (!candidate.evidence.hard_feasible)
    return HoloLiftObservationClass::NotHardFeasible;
  if (!frame_evidence.identified)
    return HoloLiftObservationClass::WeakObservation;
  if (candidate.evidence.selected_framewise &&
      candidate.evidence.search_equivalence_known &&
      candidate.evidence.equivalent_to_search_best &&
      frame_evidence.equivalent_set_complete &&
      frame_evidence.credible_alternative_set_complete &&
      frame_evidence.objective_uniqueness_known &&
      frame_evidence.objective_unique_within_search_domain &&
      frame_evidence.evidence_uniqueness_known &&
      frame_evidence.evidence_unique_within_search_domain &&
      provenance.source_frame_report_present &&
      (provenance.status == HoloLiftFrameStatus::Certified ||
       provenance.status == HoloLiftFrameStatus::Rescued) &&
      (provenance.certificate_scope ==
           HoloLiftCertificateScope::ImageIntercomponentSteric ||
       provenance.certificate_scope ==
           HoloLiftCertificateScope::ImageIntercomponentStericInterface) &&
      provenance.certificate_graph_source !=
          HoloLiftCertificateGraphSource::None &&
      provenance.hard_graph_source != HoloLiftHardGraphSource::None &&
      provenance.temporal_policy ==
          HoloLiftTemporalPolicyStatus::FramewiseProduction &&
       search_domain.audit_known && search_domain.evidence_complete &&
       search_domain.shell_radius > 0 &&
       candidate.domain_evidence.boundary_known &&
       !candidate.domain_evidence.touches_shell_boundary &&
       spatial_lift_ambiguities == 0 &&
       spatial_lift_cycle_residuals == 0) {
    if (search_domain.completeness ==
        HoloLiftDomainCompleteness::BoundedExhaustive)
      return HoloLiftObservationClass::StrongBoundedAnchor;
  }
  return HoloLiftObservationClass::AmbiguousAnchor;
}

} // namespace

std::string_view
hololift_validation_code_name(HoloLiftValidationCode code) noexcept {
  switch (code) {
  case HoloLiftValidationCode::UnsupportedApiVersion:
    return "unsupported_api_version";
  case HoloLiftValidationCode::InvalidSourceContract:
    return "invalid_source_contract";
  case HoloLiftValidationCode::EmptyObservationStore:
    return "empty_observation_store";
  case HoloLiftValidationCode::InvalidRange:
    return "invalid_range";
  case HoloLiftValidationCode::InvalidTopologyEpoch:
    return "invalid_topology_epoch";
  case HoloLiftValidationCode::DuplicateTopologyEpoch:
    return "duplicate_topology_epoch";
  case HoloLiftValidationCode::InvalidComponentMembership:
    return "invalid_component_membership";
  case HoloLiftValidationCode::DuplicateAtomMembership:
    return "duplicate_atom_membership";
  case HoloLiftValidationCode::ComponentIdentityMismatch:
    return "component_identity_mismatch";
  case HoloLiftValidationCode::ComponentIdentityCollision:
    return "component_identity_collision";
  case HoloLiftValidationCode::LayoutIdentityMismatch:
    return "layout_identity_mismatch";
  case HoloLiftValidationCode::LayoutIdentityCollision:
    return "layout_identity_collision";
  case HoloLiftValidationCode::TopologyEpochIdentityMismatch:
    return "topology_epoch_identity_mismatch";
  case HoloLiftValidationCode::InvalidFrameOrder:
    return "invalid_frame_order";
  case HoloLiftValidationCode::InvalidFrameTime:
    return "invalid_frame_time";
  case HoloLiftValidationCode::InvalidFrameProvenance:
    return "invalid_frame_provenance";
  case HoloLiftValidationCode::InvalidFrameBinding:
    return "invalid_frame_binding";
  case HoloLiftValidationCode::InvalidSpatialLift:
    return "invalid_spatial_lift";
  case HoloLiftValidationCode::MissingTopologyEpoch:
    return "missing_topology_epoch";
  case HoloLiftValidationCode::InvalidGramMatrix:
    return "invalid_gram_matrix";
  case HoloLiftValidationCode::InvalidCandidateSet:
    return "invalid_candidate_set";
  case HoloLiftValidationCode::AssignmentLayoutMismatch:
    return "assignment_layout_mismatch";
  case HoloLiftValidationCode::AssignmentComponentMismatch:
    return "assignment_component_mismatch";
  case HoloLiftValidationCode::AssignmentGaugeMismatch:
    return "assignment_gauge_mismatch";
  case HoloLiftValidationCode::AssignmentIdentityMismatch:
    return "assignment_identity_mismatch";
  case HoloLiftValidationCode::ObservationClassMismatch:
    return "observation_class_mismatch";
  }
  return "invalid_source_contract";
}

std::expected<void, HoloLiftValidationError>
validate_hololift_source_contract(
    const HoloLiftObservationStoreBuilder &store) {
  if (store.api_version != HOLOLIFT_PHASE0_API_VERSION) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::UnsupportedApiVersion,
        "unsupported HoloLift Phase 0 API version " +
            std::to_string(store.api_version)));
  }
  const auto source =
      validate_hololift_vibe_contract(store.source_contract);
  if (!source) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidSourceContract, source.error()));
  }
  const auto &coverage = store.source_coverage;
  if (!coverage.known()) {
    if (coverage.source_frame_count != 0 ||
        coverage.imported_observation_frame_count != 0 ||
        coverage.skipped_local_unwrap_frame_count != 0 ||
        coverage.skipped_fallback_frame_count != 0) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidSourceContract,
          "unknown HoloLift source coverage carries frame counts"));
    }
  } else {
    const bool counts_fit =
        coverage.source_frame_count != 0 &&
        coverage.imported_observation_frame_count == store.frames.size() &&
        coverage.imported_observation_frame_count <=
            coverage.source_frame_count &&
        coverage.skipped_local_unwrap_frame_count <=
            coverage.source_frame_count -
                coverage.imported_observation_frame_count &&
        coverage.skipped_fallback_frame_count ==
            coverage.source_frame_count -
                coverage.imported_observation_frame_count -
                coverage.skipped_local_unwrap_frame_count;
    if (!counts_fit) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidSourceContract,
          "HoloLift source coverage counts are inconsistent"));
    }
  }
  return {};
}

std::expected<void, HoloLiftValidationError>
validate_hololift_topology_epoch(
    const HoloLiftObservationStoreBuilder &store,
                                 std::size_t epoch_index) {
  if (epoch_index >= store.topology_epochs.size()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift topology epoch index is out of range", epoch_index));
  }
  const auto &epoch = store.topology_epochs[epoch_index];
  if (epoch.topology_epoch_id == 0 || epoch.layout_signature == 0 ||
      !valid_hard_graph_source(epoch.hard_graph_source) ||
      epoch.components.count == 0) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        "HoloLift topology epoch metadata is incomplete", epoch_index));
  }
  if (!epoch.components.valid_for(store.components.size())) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift topology epoch component range is invalid", epoch_index));
  }
  if (!epoch.hard_edges.valid_for(store.hard_edges.size())) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift topology epoch hard-edge range is invalid", epoch_index));
  }

  const auto components =
      std::span<const HoloLiftComponentRecord>(store.components)
          .subspan(epoch.components.begin, epoch.components.count);
  std::size_t epoch_atom_count = 0;
  for (std::size_t local_idx = 0; local_idx < components.size();
       ++local_idx) {
    const auto &component = components[local_idx];
    if (local_idx != 0 &&
        components[local_idx - 1].component_id >= component.component_id) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidTopologyEpoch,
          "HoloLift topology-epoch components are not strictly ordered",
          epoch_index));
    }
    if (component.component_id == 0 ||
        component.exact_atom_membership.count == 0 ||
        !component.exact_atom_membership.valid_for(
            store.exact_atom_memberships.size())) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidComponentMembership,
          "HoloLift component exact membership range is invalid",
          epoch_index));
    }
    const auto membership =
        std::span<const HoloLiftSourceAtomKey>(
            store.exact_atom_memberships)
            .subspan(component.exact_atom_membership.begin,
                     component.exact_atom_membership.count);
    for (std::size_t atom_idx = 1; atom_idx < membership.size();
         ++atom_idx) {
      if (!atom_key_less(membership[atom_idx - 1],
                         membership[atom_idx])) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidComponentMembership,
            "HoloLift exact atom membership is not strictly ordered",
            epoch_index));
      }
    }
    const HoloLiftSourceAtomKey canonical_root = membership.front();
    bool mixed_owner = false;
    for (const auto &atom : membership) {
      mixed_owner = mixed_owner || atom.owner != canonical_root.owner;
    }
    if (membership.size() >
        std::numeric_limits<std::size_t>::max() - epoch_atom_count) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidTopologyEpoch,
          "HoloLift topology epoch atom count overflows size_t",
          epoch_index));
    }
    epoch_atom_count += membership.size();
    if (component.root_atom != canonical_root ||
        component.owner != canonical_root.owner ||
        component.mixed_owner != mixed_owner) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidComponentMembership,
          "HoloLift component root/owner metadata is not canonical",
          epoch_index));
    }
    const auto component_id = hololift_component_id(membership);
    if (!component_id || *component_id != component.component_id) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::ComponentIdentityMismatch,
          component_id ? "HoloLift component id mismatch"
                       : component_id.error(),
          epoch_index));
    }
    const auto component_identity =
        hololift_component_identity128(membership);
    if (!component_identity ||
        *component_identity != component.component_identity) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::ComponentIdentityMismatch,
          component_identity
              ? "HoloLift 128-bit component identity mismatch"
              : component_identity.error(),
          epoch_index));
    }
  }
  struct MembershipCursor {
    HoloLiftSourceAtomKey atom;
    std::size_t component_index = 0;
    std::size_t atom_index = 0;
  };
  const auto cursor_after = [](const MembershipCursor &lhs,
                               const MembershipCursor &rhs) {
    if (atom_key_less(rhs.atom, lhs.atom))
      return true;
    if (atom_key_less(lhs.atom, rhs.atom))
      return false;
    if (lhs.component_index != rhs.component_index)
      return lhs.component_index > rhs.component_index;
    return lhs.atom_index > rhs.atom_index;
  };
  std::priority_queue<MembershipCursor, std::vector<MembershipCursor>,
                      decltype(cursor_after)>
      membership_merge(cursor_after);
  for (std::size_t component_index = 0;
       component_index < components.size(); ++component_index) {
    const auto &component = components[component_index];
    const auto membership =
        std::span<const HoloLiftSourceAtomKey>(store.exact_atom_memberships)
            .subspan(component.exact_atom_membership.begin,
                     component.exact_atom_membership.count);
    membership_merge.push({membership.front(), component_index, 0});
  }
  bool have_previous_atom = false;
  HoloLiftSourceAtomKey previous_atom;
  std::vector<std::size_t> dense_atom_component;
  std::vector<HoloLiftSourceAtomKey> canonical_atoms;
  dense_atom_component.reserve(epoch_atom_count);
  canonical_atoms.reserve(epoch_atom_count);
  while (!membership_merge.empty()) {
    const MembershipCursor cursor = membership_merge.top();
    membership_merge.pop();
    if (have_previous_atom && cursor.atom == previous_atom) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::DuplicateAtomMembership,
          "one source atom belongs to multiple HoloLift components",
          epoch_index));
    }
    previous_atom = cursor.atom;
    have_previous_atom = true;
    dense_atom_component.push_back(cursor.component_index);
    canonical_atoms.push_back(cursor.atom);
    const auto &component = components[cursor.component_index];
    const auto membership =
        std::span<const HoloLiftSourceAtomKey>(store.exact_atom_memberships)
            .subspan(component.exact_atom_membership.begin,
                     component.exact_atom_membership.count);
    const std::size_t next_atom_index = cursor.atom_index + 1;
    if (next_atom_index < membership.size()) {
      membership_merge.push({membership[next_atom_index],
                             cursor.component_index, next_atom_index});
    }
  }
  if (epoch_atom_count >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        "HoloLift topology epoch exceeds the hard-edge index domain",
        epoch_index));
  }
  const auto canonical_atom_universe_hash =
      hash_hololift_canonical_atom_universe(canonical_atoms);
  if (!canonical_atom_universe_hash ||
      epoch.atom_count != epoch_atom_count ||
      epoch.canonical_atom_universe_hash.empty() ||
      epoch.canonical_atom_universe_hash != *canonical_atom_universe_hash) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        canonical_atom_universe_hash
            ? "HoloLift topology epoch atom universe metadata mismatch"
            : canonical_atom_universe_hash.error(),
        epoch_index));
  }
  const auto hard_edges =
      std::span<const HoloLiftHardEdgeRecord>(store.hard_edges)
          .subspan(epoch.hard_edges.begin, epoch.hard_edges.count);
  std::vector<std::size_t> hard_parent(epoch_atom_count);
  std::iota(hard_parent.begin(), hard_parent.end(), 0);
  const auto find_root = [&](std::size_t atom) {
    while (hard_parent[atom] != atom) {
      hard_parent[atom] = hard_parent[hard_parent[atom]];
      atom = hard_parent[atom];
    }
    return atom;
  };
  for (std::size_t edge_idx = 0; edge_idx < hard_edges.size(); ++edge_idx) {
    const auto &edge = hard_edges[edge_idx];
    const bool valid_kind =
        edge.kind == HoloLiftHardEdgeKind::TopologyBond ||
        edge.kind == HoloLiftHardEdgeKind::ValidatedMetadata;
    if (edge.atom_a >= edge.atom_b || edge.atom_b >= epoch_atom_count ||
        !valid_kind ||
        (edge_idx != 0 &&
         !hard_edge_less(hard_edges[edge_idx - 1], edge))) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidTopologyEpoch,
          "HoloLift hard-edge table is not canonical", epoch_index));
    }
    if (dense_atom_component[edge.atom_a] !=
        dense_atom_component[edge.atom_b]) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidTopologyEpoch,
          "HoloLift hard edge connects two declared components",
          epoch_index));
    }
    const std::size_t root_a = find_root(edge.atom_a);
    const std::size_t root_b = find_root(edge.atom_b);
    if (root_a != root_b)
      hard_parent[root_b] = root_a;
  }
  std::vector<std::size_t> component_hard_root(components.size(),
                                               HOLOLIFT_NO_INDEX);
  for (std::size_t atom = 0; atom < epoch_atom_count; ++atom) {
    const std::size_t component_index = dense_atom_component[atom];
    const std::size_t root = find_root(atom);
    if (component_hard_root[component_index] == HOLOLIFT_NO_INDEX) {
      component_hard_root[component_index] = root;
    } else if (component_hard_root[component_index] != root) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidTopologyEpoch,
          "HoloLift component is disconnected in the declared hard graph",
          epoch_index));
    }
  }
  if (!hard_graph_source_matches_edges(epoch.hard_graph_source, hard_edges)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        "HoloLift hard graph source does not match its edge kinds",
        epoch_index));
  }
  const std::size_t component_forest_edge_count =
      epoch_atom_count - components.size();
  if (hard_edges.size() < component_forest_edge_count) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        "HoloLift hard graph has fewer edges than a component forest",
        epoch_index));
  }
  const std::size_t hard_graph_cycle_rank =
      hard_edges.size() - component_forest_edge_count;
  if (epoch.hard_graph_cycle_rank != hard_graph_cycle_rank) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidTopologyEpoch,
        "HoloLift hard-graph cycle rank metadata mismatch", epoch_index));
  }

  const auto layout_signature = hololift_layout_signature(components);
  if (!layout_signature ||
      *layout_signature != epoch.layout_signature) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::LayoutIdentityMismatch,
        layout_signature ? "HoloLift layout signature mismatch"
                         : layout_signature.error(),
          epoch_index));
  }
  const auto layout_identity = hololift_layout_identity128(components);
  if (!layout_identity || *layout_identity != epoch.layout_identity) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::LayoutIdentityMismatch,
        layout_identity ? "HoloLift 128-bit layout identity mismatch"
                        : layout_identity.error(),
        epoch_index));
  }
  const auto topology_epoch_id = hololift_topology_epoch_id(
      store.source_contract.schema_version, epoch.hard_graph_source,
      epoch.layout_signature, components,
      store.exact_atom_memberships, hard_edges);
  if (!topology_epoch_id ||
      *topology_epoch_id != epoch.topology_epoch_id) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::TopologyEpochIdentityMismatch,
        topology_epoch_id ? "HoloLift topology epoch id mismatch"
                          : topology_epoch_id.error(),
          epoch_index));
  }
  const auto topology_epoch_identity = hololift_topology_epoch_identity128(
      store.source_contract.schema_version, epoch.hard_graph_source,
      epoch.layout_identity, components, hard_edges);
  if (!topology_epoch_identity ||
      *topology_epoch_identity != epoch.topology_epoch_identity) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::TopologyEpochIdentityMismatch,
        topology_epoch_identity
            ? "HoloLift 128-bit topology epoch identity mismatch"
            : topology_epoch_identity.error(),
        epoch_index));
  }
  return {};
}

std::expected<void, HoloLiftValidationError>
validate_hololift_frame(const HoloLiftObservationStoreBuilder &store,
                        std::size_t frame_index) {
  if (frame_index >= store.frames.size()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift frame index is out of range", HOLOLIFT_NO_INDEX,
        frame_index));
  }
  const auto &frame = store.frames[frame_index];
  if (!std::isfinite(frame.time_ps)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameTime,
        "HoloLift frame time is not finite", HOLOLIFT_NO_INDEX,
        frame_index));
  }
  if (!frame.frame_binding_valid ||
      !frame_binding_box_ok(frame.box_matrix) || frame.box_hash.empty() ||
      frame.atom_selection_hash.empty() ||
      frame.canonical_atom_universe_hash.empty() ||
      frame.wrapped_coordinate_hash.empty() ||
      hash_hololift_box_matrix(frame.box_matrix) != frame.box_hash) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameBinding,
        "HoloLift frame is not bound to a valid box/coordinate selection",
        HOLOLIFT_NO_INDEX, frame_index));
  }
  if (!frame.spatial_lift_valid ||
      frame.spatial_lift_policy_version == 0 ||
      frame.spatial_lift_identity.empty()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidSpatialLift,
        "HoloLift frame lacks a replayable internal spatial lift",
        HOLOLIFT_NO_INDEX, frame_index));
  }
  if (frame.topology_epoch_index >= store.topology_epochs.size()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::MissingTopologyEpoch,
        "HoloLift frame topology-epoch index is invalid",
        HOLOLIFT_NO_INDEX, frame_index));
  }
  const std::size_t epoch_index = frame.topology_epoch_index;
  const auto &epoch = store.topology_epochs[epoch_index];
  if (frame.topology_epoch_id != epoch.topology_epoch_id) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::MissingTopologyEpoch,
        "HoloLift frame topology-epoch id/index cross-link is invalid",
        epoch_index, frame_index));
  }
  if (frame.canonical_atom_universe_hash !=
      epoch.canonical_atom_universe_hash) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameBinding,
        "HoloLift frame atom universe does not match its topology epoch",
        epoch_index, frame_index));
  }
  if (frame.spatial_lift_ambiguous_hard_edges > epoch.hard_edges.count ||
      frame.spatial_lift_cycle_residuals > epoch.hard_graph_cycle_rank) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidSpatialLift,
        "HoloLift spatial-lift diagnostics exceed hard-graph bounds",
        epoch_index, frame_index));
  }
  if (frame.layout_signature != epoch.layout_signature) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::LayoutIdentityMismatch,
        "HoloLift frame layout does not match its topology epoch",
        epoch_index, frame_index));
  }
  if (frame.topology_epoch_identity != epoch.topology_epoch_identity ||
      frame.layout_identity != epoch.layout_identity) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::LayoutIdentityMismatch,
        "HoloLift frame 128-bit identities do not match its topology epoch",
        epoch_index, frame_index));
  }
  if (!normalized_gram_is_spd(frame.normalized_gram)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidGramMatrix,
        "HoloLift normalized lattice Gram matrix is not finite SPD "
        "with trace 3",
        epoch_index, frame_index));
  }
  const auto binding_gram = normalized_gram_from_box(frame.box_matrix);
  for (std::size_t idx = 0; idx < binding_gram.size(); ++idx) {
    const double scale =
        std::max({1.0, std::fabs(binding_gram[idx]),
                  std::fabs(frame.normalized_gram[idx])});
    if (std::fabs(binding_gram[idx] - frame.normalized_gram[idx]) >
        1.0e-10 * scale) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameBinding,
          "HoloLift normalized Gram matrix does not match the bound box",
          epoch_index, frame_index));
    }
  }
  const bool contradicted_gap_candidate_scope =
      frame.provenance.status == HoloLiftFrameStatus::Contradicted &&
      frame.candidates.count == 0 &&
      frame.candidate_set_scope == HoloLiftCandidateSetScope::Unknown;
  if (!contradicted_gap_candidate_scope &&
      frame.candidate_set_scope !=
          HoloLiftCandidateSetScope::RetainedCertifiedBand) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift Phase 0 requires a retained_certified_band candidate set",
        epoch_index, frame_index));
  }
  if (!frame.candidates.valid_for(store.candidates.size())) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift frame candidate range is invalid", epoch_index,
        frame_index));
  }
  if (frame.evidence.retained_certified_assignment_count !=
      frame.candidates.count) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift retained candidate count does not match its flat range",
        epoch_index, frame_index));
  }
  if ((frame.search_domain.evidence_complete &&
       !frame.search_domain.audit_known) ||
      (frame.search_domain.audit_known &&
       frame.search_domain.shell_radius == 0) ||
      (frame.search_domain.expansion_exhausted &&
       !frame.search_domain.expansion_attempted)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift frame search-domain evidence is internally inconsistent",
        epoch_index, frame_index));
  }
  const auto domain_completeness = frame.search_domain.completeness;
  const bool bounded_complete =
      domain_completeness == HoloLiftDomainCompleteness::BoundedExhaustive;
  const bool global_complete =
      domain_completeness == HoloLiftDomainCompleteness::GlobalCertified;
  if (global_complete ||
      (bounded_complete && !frame.search_domain.bounded_domain_only) ||
      (bounded_complete &&
       (!frame.search_domain.audit_known ||
         !frame.search_domain.evidence_complete))) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        global_complete
            ? "HoloLift global certification is unsupported without an "
              "outside-domain infeasibility proof"
            : "HoloLift domain-completeness proof is inconsistent with its "
              "audit",
        epoch_index, frame_index));
  }

  const auto components =
      std::span<const HoloLiftComponentRecord>(store.components)
          .subspan(epoch.components.begin, epoch.components.count);
  std::size_t selected_count = 0;
  std::size_t search_equivalent_exported = 0;
  std::size_t search_domain_exported = 0;
  std::unordered_set<std::uint64_t> assignment_signatures;
  assignment_signatures.reserve(frame.candidates.count);
  std::unordered_set<HoloLiftHash128, HoloLiftHash128Hasher>
      assignment_identities;
  assignment_identities.reserve(frame.candidates.count);
  std::size_t root_component_index = 0;
  for (std::size_t idx = 1; idx < components.size(); ++idx) {
    if (components[idx].exact_atom_membership.count >
            components[root_component_index].exact_atom_membership.count ||
        (components[idx].exact_atom_membership.count ==
             components[root_component_index]
                 .exact_atom_membership.count &&
         components[idx].component_id <
             components[root_component_index].component_id)) {
      root_component_index = idx;
    }
  }
  for (std::size_t local_idx = 0; local_idx < frame.candidates.count;
       ++local_idx) {
    const std::size_t candidate_index =
        frame.candidates.begin + local_idx;
    const auto &candidate = store.candidates[candidate_index];
    if (candidate.rank != local_idx + 1) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift candidate ranks are not contiguous and one-based",
          epoch_index, frame_index, candidate_index));
    }
    if (candidate.layout_signature != frame.layout_signature) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentLayoutMismatch,
          "HoloLift candidate layout does not match its frame",
          epoch_index, frame_index, candidate_index));
    }
    if (candidate.layout_identity != frame.layout_identity) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentLayoutMismatch,
          "HoloLift candidate 128-bit layout identity does not match its frame",
          epoch_index, frame_index, candidate_index));
    }
    if (!candidate.component_images.valid_for(
            store.component_images.size()) ||
        candidate.component_images.count != components.size()) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentComponentMismatch,
          "HoloLift candidate component-image range is invalid",
          epoch_index, frame_index, candidate_index));
    }
    const auto images =
        std::span<const HoloLiftLatticeImage>(store.component_images)
            .subspan(candidate.component_images.begin,
                     candidate.component_images.count);
    if (!zero_image(images[root_component_index])) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentGaugeMismatch,
          "HoloLift candidate assignment is not in canonical global gauge",
          epoch_index, frame_index, candidate_index));
    }
    const auto assignment_signature = hololift_assignment_signature_canonical(
        frame.layout_signature, components, images);
    if (!assignment_signature ||
        *assignment_signature != candidate.assignment_signature) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentIdentityMismatch,
          assignment_signature
              ? "HoloLift candidate assignment signature mismatch"
              : assignment_signature.error(),
          epoch_index, frame_index, candidate_index));
    }
    const bool candidate_has_supported_relation =
        candidate.evidence_compatible_hypothesis_count != 0;
    if (candidate.evidence_relative_relation_signature !=
            candidate.assignment_signature ||
        candidate_has_supported_relation !=
            !candidate.evidence_compatible_hypothesis_hash.empty() ||
        candidate_has_supported_relation !=
            !candidate.evidence_compatible_pair_hash.empty() ||
        candidate_has_supported_relation !=
            (candidate.evidence_compatible_contact_count != 0) ||
        candidate_has_supported_relation !=
            (candidate.evidence_supported_relation_count != 0) ||
        candidate.evidence_supported_relation_count +
                candidate.evidence_no_support_relation_count !=
            frame.provenance.soft_observed_component_relation_count) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift candidate relation-hypothesis provenance is inconsistent",
          epoch_index, frame_index, candidate_index));
    }
    const auto assignment_identity =
        hololift_assignment_identity128_canonical(
            frame.layout_identity, components, images);
    if (!assignment_identity ||
        *assignment_identity != candidate.assignment_identity) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::AssignmentIdentityMismatch,
          assignment_identity
              ? "HoloLift 128-bit assignment identity mismatch"
              : assignment_identity.error(),
          epoch_index, frame_index, candidate_index));
    }
    if (!assignment_signatures
             .insert(candidate.assignment_signature)
             .second) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift frame contains a duplicate assignment signature",
          epoch_index, frame_index, candidate_index));
    }
    if (!assignment_identities.insert(candidate.assignment_identity).second) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift frame contains a duplicate 128-bit assignment identity",
          epoch_index, frame_index, candidate_index));
    }
    const bool policy_admitted =
        candidate.policy_admission != HoloLiftPolicyAdmission::None;
    const bool carry_source =
        candidate.policy_admission == HoloLiftPolicyAdmission::CarryPreviousFrame;
    const bool known_admission =
        candidate.policy_admission == HoloLiftPolicyAdmission::None ||
        carry_source ||
        candidate.policy_admission == HoloLiftPolicyAdmission::RestoreDeclaredDomain;
    const bool known_evaluation =
        candidate.provider_evaluation == HoloLiftProviderEvaluation::Recorded ||
        candidate.provider_evaluation == HoloLiftProviderEvaluation::NotEvaluated;
    if (!known_admission || !known_evaluation ||
        candidate.carried != policy_admitted ||
        (policy_admitted &&
         (candidate.evidence.hard_feasible ||
          candidate.evidence.selected_framewise ||
          candidate.evidence.search_equivalence_known ||
          candidate.evidence.equivalent_to_search_best ||
          candidate.evidence.equivalent_under_output_order)) ||
        (!policy_admitted &&
         candidate.provider_evaluation != HoloLiftProviderEvaluation::Recorded) ||
        (!carry_source &&
         (candidate.policy_source_frame_index != HOLOLIFT_NO_INDEX ||
          !candidate.policy_source_assignment_identity.empty()))) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift policy admission/provider provenance is inconsistent",
          epoch_index, frame_index, candidate_index));
    }
    if (carry_source) {
      if (frame_index == 0 ||
          candidate.policy_source_frame_index != frame_index - 1 ||
          candidate.policy_source_assignment_identity.empty()) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidCandidateSet,
            "HoloLift carry lacks an immediately preceding source frame",
            epoch_index, frame_index, candidate_index));
      }
      const auto &source_frame = store.frames[candidate.policy_source_frame_index];
      if (!source_frame.candidates.valid_for(store.candidates.size()) ||
          source_frame.topology_epoch_identity != frame.topology_epoch_identity ||
          source_frame.layout_identity != frame.layout_identity ||
          std::none_of(store.candidates.data() + source_frame.candidates.begin,
                       store.candidates.data() + source_frame.candidates.begin +
                           source_frame.candidates.count,
                       [&](const auto &source) {
                         return source.assignment_identity ==
                                candidate.policy_source_assignment_identity;
                       })) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidCandidateSet,
            "HoloLift carry source assignment is absent or incompatible",
            epoch_index, frame_index, candidate_index));
      }
    }
    if (!candidate.evidence.hard_feasible && !policy_admitted) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift band contains a non-certified assignment without policy admission",
          epoch_index, frame_index, candidate_index));
    }
    if (!policy_admitted && candidate.domain_evidence.boundary_known !=
        frame.search_domain.audit_known) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "candidate shell-boundary provenance disagrees with its frame "
          "search-domain audit",
          epoch_index, frame_index, candidate_index));
    }
    selected_count += candidate.evidence.selected_framewise ? 1 : 0;
    search_domain_exported +=
        candidate.evidence.search_equivalence_known ? 1 : 0;
    search_equivalent_exported +=
        candidate.evidence.search_equivalence_known &&
                candidate.evidence.equivalent_to_search_best
            ? 1
            : 0;
  }

  if (frame.candidates.count == 0) {
    if (frame.selected_candidate_index != HOLOLIFT_NO_INDEX ||
        selected_count != 0) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "empty HoloLift candidate set has a selected assignment",
          epoch_index, frame_index));
    }
  } else if (selected_count != 1 ||
             frame.selected_candidate_index < frame.candidates.begin ||
             frame.selected_candidate_index >=
                 frame.candidates.begin + frame.candidates.count ||
             !store.candidates[frame.selected_candidate_index]
                  .evidence.selected_framewise) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift frame must identify exactly one framewise-selected "
        "candidate",
        epoch_index, frame_index));
  }
  if (frame.candidates.count != 0) {
    const auto begin = store.candidates.begin() + frame.candidates.begin;
    const auto end = begin + frame.candidates.count;
    const auto output = std::find_if(
        begin, end, [&](const HoloLiftCandidateRecord &candidate) {
          return candidate.assignment_signature ==
                 frame.provenance.output_assignment_signature;
        });
    if (output == end ||
        output->evidence_relative_relation_signature !=
            frame.provenance.output_assignment_signature ||
        output->evidence_compatible_hypothesis_hash !=
            frame.provenance.soft_observed_selected_hypothesis_hash ||
        output->evidence_compatible_pair_hash !=
            frame.provenance.soft_observed_selected_compatible_pair_hash ||
        output->evidence_compatible_hypothesis_count !=
            frame.provenance.soft_observed_compatible_hypothesis_count ||
        output->evidence_compatible_contact_count !=
            frame.provenance.soft_observed_selected_compatible_contact_count ||
        output->evidence_no_support_relation_count !=
            frame.provenance.soft_observed_no_support_relation_count) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "selected HoloLift candidate evidence does not match frame provenance",
          epoch_index, frame_index, frame.selected_candidate_index));
    }
  }

  const bool retained_band_eligible_status =
      frame.provenance.status == HoloLiftFrameStatus::Certified ||
      frame.provenance.status == HoloLiftFrameStatus::Rescued ||
      frame.provenance.status == HoloLiftFrameStatus::WeakObservation;
  const bool hard_feasible_output_status =
      retained_band_eligible_status ||
      frame.provenance.status == HoloLiftFrameStatus::Contradicted;
  if (retained_band_eligible_status && frame.candidates.count == 0) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "hard-certified HoloLift frame has no selected candidate",
        epoch_index, frame_index));
  }
  if (!retained_band_eligible_status && frame.candidates.count != 0) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "uncertified HoloLift frame carries a certified candidate band",
        epoch_index, frame_index));
  }
  if (frame.provenance.status == HoloLiftFrameStatus::WeakObservation &&
      frame.provenance.evidence_state == HoloLiftEvidenceState::WeakDisconnected &&
      frame.evidence.identified) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift weak-observation status disagrees with identifiability",
        epoch_index, frame_index));
  }
  if (frame.evidence.identified !=
      frame.provenance.evidence_graph_connected) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "legacy identified alias disagrees with evidence graph connectivity",
        epoch_index, frame_index));
  }
  const auto evidence_state = frame.provenance.evidence_state;
  const bool evidence_hashes_present =
      !frame.provenance.soft_observed_contact_pair_hash.empty() &&
      !frame.provenance.soft_observed_lost_pair_hash.empty() &&
      !frame.provenance.soft_observed_all_hypotheses_hash.empty() &&
      ((frame.provenance.soft_observed_compatible_hypothesis_count == 0) ==
       frame.provenance.soft_observed_selected_hypothesis_hash.empty()) &&
      ((frame.provenance.soft_observed_compatible_hypothesis_count == 0) ==
       frame.provenance.soft_observed_selected_compatible_pair_hash.empty()) &&
      (frame.provenance.soft_observed_selected_relation_has_support ==
       (frame.provenance.soft_observed_compatible_hypothesis_count > 0));
  const bool evidence_cutoff_valid =
      std::isfinite(frame.provenance.soft_observed_contact_cutoff_A) &&
      frame.provenance.soft_observed_contact_cutoff_A > 0.0;
  if (frame.provenance.soft_observed_contacts_lost >
      frame.provenance.soft_observed_contacts_checked) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift evidence lost-contact count exceeds checked count",
        epoch_index, frame_index));
  }
  if (frame.provenance.soft_observed_selected_compatible_contact_loss_count >
          frame.provenance.soft_observed_selected_compatible_contact_count ||
      frame.provenance.soft_observed_contacts_lost !=
          frame.provenance
              .soft_observed_selected_compatible_contact_loss_count) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift selected-compatible evidence loss counts are inconsistent",
        epoch_index, frame_index));
  }
  const auto raw_partition =
      frame.provenance.soft_observed_selected_compatible_contact_count +
      frame.provenance.soft_observed_alternative_hypothesis_contact_count +
      frame.provenance.soft_observed_hypothesis_internal_edges_ignored;
  if (frame.provenance.evidence_consistency_evaluated &&
      raw_partition != frame.provenance.soft_observed_contacts_checked) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift raw E_obs count does not equal its hypothesis partition",
        epoch_index, frame_index));
  }
  const bool construction_complete =
      frame.provenance.soft_observed_contact_construction_attempted &&
      frame.provenance.soft_observed_contact_construction_complete &&
      frame.provenance.soft_observed_contact_cell_setup_failures == 0 &&
      frame.provenance.soft_observed_contact_atom_index_failures == 0 &&
      frame.provenance.soft_observed_contact_mic_query_failures == 0 &&
      frame.provenance.soft_observed_hypothesis_construction_attempted &&
      frame.provenance.soft_observed_hypothesis_construction_complete &&
      frame.provenance.soft_observed_hypothesis_mic_ambiguity_failures == 0 &&
      frame.provenance.soft_observed_hypothesis_component_mapping_failures == 0;
  const bool supported_consistent =
      evidence_state == HoloLiftEvidenceState::SupportedConnected &&
      frame.provenance.evidence_consistency_evaluated &&
      frame.provenance.evidence_consistent &&
      frame.provenance.evidence_graph_connected && evidence_hashes_present &&
      evidence_cutoff_valid && construction_complete &&
      frame.provenance.soft_observed_contacts_lost == 0 &&
      (frame.provenance.status == HoloLiftFrameStatus::Certified ||
       frame.provenance.status == HoloLiftFrameStatus::Rescued ||
       frame.provenance.status == HoloLiftFrameStatus::WeakObservation);
  const bool weak_consistent =
      evidence_state == HoloLiftEvidenceState::WeakDisconnected &&
      frame.provenance.evidence_consistency_evaluated &&
      frame.provenance.evidence_consistent &&
      !frame.provenance.evidence_graph_connected && evidence_hashes_present &&
      evidence_cutoff_valid && construction_complete &&
      frame.provenance.soft_observed_contacts_lost == 0 &&
      frame.provenance.status == HoloLiftFrameStatus::WeakObservation;
  const bool weak_ambiguous_consistent =
      evidence_state == HoloLiftEvidenceState::WeakAmbiguous &&
      frame.provenance.evidence_consistency_evaluated &&
      frame.provenance.evidence_consistent && evidence_hashes_present &&
      evidence_cutoff_valid && construction_complete &&
      frame.provenance.soft_observed_contacts_lost == 0 &&
      (frame.provenance.soft_observed_ambiguous_relation_count > 0 ||
       frame.provenance.soft_observed_no_support_relation_count > 0) &&
      frame.provenance.status == HoloLiftFrameStatus::WeakObservation;
  const bool contradicted_consistent =
      evidence_state == HoloLiftEvidenceState::Contradicted &&
      frame.provenance.evidence_consistency_evaluated &&
      !frame.provenance.evidence_consistent && evidence_hashes_present &&
      evidence_cutoff_valid && construction_complete &&
      frame.provenance.soft_observed_contacts_lost > 0 &&
      frame.provenance.status == HoloLiftFrameStatus::Contradicted &&
      frame.candidates.count == 0;
  const bool empty_selected_evidence_payload =
      !frame.provenance.evidence_consistency_evaluated &&
      !frame.provenance.evidence_consistent &&
      frame.provenance.soft_observed_lost_pair_hash.empty() &&
      frame.provenance.soft_observed_selected_hypothesis_hash.empty() &&
      frame.provenance.soft_observed_selected_compatible_pair_hash.empty() &&
      frame.provenance.soft_observed_compatible_hypothesis_count == 0 &&
      !frame.provenance.soft_observed_selected_relation_has_support &&
      frame.provenance.soft_observed_selected_compatible_contact_count == 0 &&
      frame.provenance.soft_observed_alternative_hypothesis_contact_count == 0 &&
      frame.provenance.soft_observed_selected_compatible_contact_loss_count == 0 &&
      frame.provenance.soft_observed_no_support_relation_count == 0 &&
      frame.provenance.soft_observed_contacts_lost == 0;
  const bool not_applicable_consistent =
      evidence_state == HoloLiftEvidenceState::NotApplicable &&
      empty_selected_evidence_payload &&
      frame.provenance.soft_observed_contacts_checked == 0 &&
      frame.provenance.soft_observed_contact_pair_hash.empty() &&
      !frame.provenance.soft_observed_contact_construction_attempted &&
      !std::isfinite(frame.provenance.soft_observed_contact_cutoff_A) &&
      frame.provenance.status == HoloLiftFrameStatus::LocalUnwrap;
  const bool raw_failure_diagnostics_coherent =
      (frame.provenance.soft_observed_contacts_checked == 0 ||
       !frame.provenance.soft_observed_contact_pair_hash.empty()) &&
      (frame.provenance.soft_observed_contact_pair_hash.empty() ||
       frame.provenance.soft_observed_contact_construction_attempted) &&
      (!std::isfinite(frame.provenance.soft_observed_contact_cutoff_A) ||
       frame.provenance.soft_observed_contact_construction_attempted);
  const bool not_evaluated_consistent =
      evidence_state == HoloLiftEvidenceState::NotEvaluated &&
      empty_selected_evidence_payload &&
      raw_failure_diagnostics_coherent &&
      frame.provenance.status == HoloLiftFrameStatus::Fallback;
  if (!(supported_consistent || weak_consistent || weak_ambiguous_consistent ||
        contradicted_consistent ||
        not_applicable_consistent || not_evaluated_consistent)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift schema-v7 evidence state/status semantics are inconsistent",
        epoch_index, frame_index));
  }

  if (!frame.provenance.source_frame_report_present) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "HoloLift frame lacks its VIBE frame-report provenance",
        epoch_index, frame_index));
  }
  const bool contradicted_diagnostic_assignment =
        frame.provenance.status == HoloLiftFrameStatus::Contradicted &&
        frame.provenance.framewise_assignment_signature != 0 &&
        frame.provenance.output_assignment_signature != 0;
  if (frame.candidates.count == 0) {
    if ((!contradicted_diagnostic_assignment &&
         (frame.provenance.framewise_assignment_signature != 0 ||
          frame.provenance.output_assignment_signature != 0 ||
          frame.provenance.temporal_assignment_signature != 0)) ||
        !frame.provenance.framewise_assignment_identity.empty() ||
        !frame.provenance.output_assignment_identity.empty() ||
        !frame.provenance.temporal_assignment_identity.empty() ||
        (!contradicted_diagnostic_assignment &&
         (frame.provenance.temporal_selected ||
          frame.provenance.temporal_changed_from_framewise))) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameProvenance,
          "candidate-free HoloLift frame has invalid assignment provenance",
          epoch_index, frame_index));
    }
  } else {
    const auto &selected =
        store.candidates[frame.selected_candidate_index];
    if (frame.provenance.framewise_assignment_signature !=
            selected.assignment_signature ||
        frame.provenance.framewise_assignment_identity !=
            selected.assignment_identity) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameProvenance,
          "HoloLift framewise assignment signature does not match the "
          "selected candidate",
          epoch_index, frame_index,
          frame.selected_candidate_index));
    }
  }
  if (frame.provenance.temporal_policy ==
      HoloLiftTemporalPolicyStatus::FramewiseProduction) {
    if (frame.provenance.temporal_assignment_signature != 0 ||
        !frame.provenance.temporal_assignment_identity.empty() ||
        frame.provenance.temporal_selected ||
        frame.provenance.temporal_changed_from_framewise ||
        frame.provenance.output_assignment_signature !=
            frame.provenance.framewise_assignment_signature ||
        frame.provenance.output_assignment_identity !=
            frame.provenance.framewise_assignment_identity) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameProvenance,
          "production framewise HoloLift input carries inconsistent "
          "temporal/output provenance",
          epoch_index, frame_index));
    }
  } else if (frame.provenance.temporal_selected) {
    const bool candidate_free_contradicted_temporal =
        contradicted_diagnostic_assignment && frame.candidates.count == 0;
    const bool retained_temporal_identity_invalid =
        !candidate_free_contradicted_temporal &&
        (frame.provenance.temporal_assignment_identity.empty() ||
         frame.provenance.output_assignment_signature !=
             frame.provenance.temporal_assignment_signature ||
         frame.provenance.output_assignment_identity !=
             frame.provenance.temporal_assignment_identity ||
         !assignment_signatures.contains(
             frame.provenance.temporal_assignment_signature) ||
         !assignment_identities.contains(
             frame.provenance.temporal_assignment_identity));
    const bool contradicted_temporal_signature_invalid =
        candidate_free_contradicted_temporal &&
        frame.provenance.output_assignment_signature !=
            frame.provenance.temporal_assignment_signature;
    if (frame.provenance.temporal_assignment_signature == 0 ||
        retained_temporal_identity_invalid ||
        contradicted_temporal_signature_invalid ||
        frame.provenance.temporal_changed_from_framewise !=
            (frame.provenance.output_assignment_signature !=
             frame.provenance.framewise_assignment_signature)) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameProvenance,
          "experimental Viterbi assignment provenance is inconsistent",
          epoch_index, frame_index));
    }
  } else if (frame.provenance.temporal_assignment_signature != 0 ||
             !frame.provenance.temporal_assignment_identity.empty() ||
             frame.provenance.temporal_changed_from_framewise ||
             frame.provenance.output_assignment_signature !=
                 frame.provenance.framewise_assignment_signature ||
             frame.provenance.output_assignment_identity !=
                 frame.provenance.framewise_assignment_identity) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "inactive Viterbi diagnostic changed assignment provenance",
        epoch_index, frame_index));
  }
  if (hard_feasible_output_status &&
      (frame.provenance.certificate_scope !=
           HoloLiftCertificateScope::ImageIntercomponentSteric &&
       frame.provenance.certificate_scope !=
           HoloLiftCertificateScope::ImageIntercomponentStericInterface)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "hard-certified HoloLift frame lacks an intercomponent certificate "
        "scope",
        epoch_index, frame_index));
  }
  if (!hard_feasible_output_status &&
      frame.provenance.status != HoloLiftFrameStatus::LocalUnwrap &&
      frame.provenance.certificate_scope != HoloLiftCertificateScope::None) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "uncertified HoloLift frame carries a certificate scope",
        epoch_index, frame_index));
  }
  if (frame.provenance.status == HoloLiftFrameStatus::LocalUnwrap &&
      frame.provenance.certificate_scope !=
          HoloLiftCertificateScope::LocalUnwrap) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "local-unwrap HoloLift frame has the wrong certificate scope",
        epoch_index, frame_index));
  }
  if (hard_feasible_output_status &&
      frame.provenance.hard_graph_source != epoch.hard_graph_source) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "frame hard-graph source does not match its topology epoch",
        epoch_index, frame_index));
  }
  if ((frame.provenance.certificate_scope ==
           HoloLiftCertificateScope::ImageIntercomponentSteric ||
       frame.provenance.certificate_scope ==
           HoloLiftCertificateScope::ImageIntercomponentStericInterface) &&
      (frame.provenance.certificate_graph_source ==
           HoloLiftCertificateGraphSource::None ||
       frame.provenance.hard_graph_source ==
           HoloLiftHardGraphSource::None)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidFrameProvenance,
        "intercomponent HoloLift certificate lacks graph provenance",
        epoch_index, frame_index));
  }

  if (frame.evidence.search_equivalent_assignments_exported !=
          search_equivalent_exported ||
      frame.evidence.search_domain_assignments_exported !=
          search_domain_exported) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift candidate provenance counters do not match the band",
        epoch_index, frame_index));
  }
  const bool equivalent_set_complete =
      frame.evidence.uniqueness_search_exhaustive &&
      frame.evidence.equivalent_best_assignments_within_domain > 0 &&
      search_equivalent_exported ==
          frame.evidence.equivalent_best_assignments_within_domain;
  if (frame.evidence.equivalent_set_complete !=
      equivalent_set_complete) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift equivalent optimum-set completeness is inconsistent",
        epoch_index, frame_index));
  }
  const bool credible_alternative_set_complete =
      frame.evidence.feasible_assignment_set_enumerated &&
      frame.evidence.feasible_assignments_within_domain > 0 &&
      search_domain_exported == frame.candidates.count &&
      search_domain_exported ==
          frame.evidence.feasible_assignments_within_domain;
  if (frame.evidence.credible_alternative_set_complete !=
      credible_alternative_set_complete) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift credible alternative-set completeness is inconsistent",
        epoch_index, frame_index));
  }
  const auto expected_feasible_set_semantics =
      frame.evidence.feasible_assignment_set_enumerated
          ? HoloLiftFeasibleSetSemantics::HardConstraintsWithinSearchDomain
          : HoloLiftFeasibleSetSemantics::NotEnumerated;
  if (frame.evidence.feasible_set_semantics !=
      expected_feasible_set_semantics) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift feasible-set semantics disagrees with enumeration status",
        epoch_index, frame_index));
  }
  if (!frame.evidence.feasible_assignment_set_enumerated &&
      frame.evidence.feasible_assignments_within_domain != 0) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift non-enumerated hard-feasible set has a nonzero count",
        epoch_index, frame_index));
  }
  if ((frame.evidence.objective_unique_within_search_domain &&
       !frame.evidence.objective_uniqueness_known) ||
      (frame.evidence.evidence_unique_within_search_domain &&
       !frame.evidence.evidence_uniqueness_known)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift uniqueness value is true while its proof status is unknown",
        epoch_index, frame_index));
  }
  if (frame.evidence.objective_uniqueness_known) {
    if (frame.evidence.equivalent_best_assignments_within_domain == 0) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "known HoloLift objective uniqueness has no optimum count",
          epoch_index, frame_index));
    }
    const bool expected_objective_unique =
        frame.evidence.identified &&
        frame.evidence.equivalent_best_assignments_within_domain == 1;
    if (frame.evidence.objective_unique_within_search_domain !=
        expected_objective_unique) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift objective uniqueness disagrees with the optimum count",
          epoch_index, frame_index));
    }
  }
  if (frame.evidence.feasible_assignment_set_enumerated &&
      frame.evidence.soft_score_valid_assignments_within_domain >
          frame.evidence.feasible_assignments_within_domain) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift soft-valid count exceeds the hard-feasible count",
        epoch_index, frame_index));
  }
  if (frame.evidence.evidence_unique_within_search_domain &&
      (!frame.evidence.feasible_assignment_set_enumerated ||
       frame.evidence.feasible_assignments_within_domain != 1)) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidCandidateSet,
        "HoloLift evidence uniqueness lacks a singleton feasible set",
        epoch_index, frame_index));
  }
  if (frame.evidence.evidence_uniqueness_known) {
    if (!frame.evidence.feasible_assignment_set_enumerated ||
        frame.evidence.feasible_assignments_within_domain == 0) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "known HoloLift evidence uniqueness lacks a feasible-set audit",
          epoch_index, frame_index));
    }
    const bool expected_evidence_unique =
        frame.evidence.identified &&
        frame.evidence.feasible_assignment_set_enumerated &&
        frame.evidence.feasible_assignments_within_domain == 1;
    if (frame.evidence.evidence_unique_within_search_domain !=
        expected_evidence_unique) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidCandidateSet,
          "HoloLift evidence uniqueness disagrees with the hard-feasible set",
          epoch_index, frame_index));
    }
  }

  for (std::size_t local_idx = 0; local_idx < frame.candidates.count;
       ++local_idx) {
    const std::size_t candidate_index =
        frame.candidates.begin + local_idx;
    const auto &candidate = store.candidates[candidate_index];
    if (candidate.observation_class !=
        expected_observation_class(candidate, frame.evidence,
                                   frame.provenance,
                                    frame.search_domain,
                                    frame.spatial_lift_ambiguous_hard_edges,
                                    frame.spatial_lift_cycle_residuals)) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::ObservationClassMismatch,
          "HoloLift observation class does not match VIBE evidence",
          epoch_index, frame_index, candidate_index));
    }
  }
  return {};
}

std::expected<void, HoloLiftValidationError>
validate_hololift_observation_store(
    const HoloLiftObservationStoreBuilder &store) {
  const auto source = validate_hololift_source_contract(store);
  if (!source)
    return source;
  if (store.topology_epochs.empty() || store.frames.empty()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::EmptyObservationStore,
        "HoloLift observation store requires at least one epoch and frame"));
  }
  if (!store.exact_atom_masses.empty()) {
    if (store.exact_atom_masses.size() !=
        store.exact_atom_memberships.size()) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidComponentMembership,
          "HoloLift atom-mass table does not match exact memberships"));
    }
    for (const double mass : store.exact_atom_masses) {
      if (!std::isfinite(mass) || mass < 0.0) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidComponentMembership,
            "HoloLift atom masses must be finite and non-negative"));
      }
    }
  }

  std::unordered_set<std::uint64_t> epoch_ids;
  epoch_ids.reserve(store.topology_epochs.size());
  std::unordered_set<HoloLiftHash128, HoloLiftHash128Hasher> epoch_identities;
  epoch_identities.reserve(store.topology_epochs.size());
  std::unordered_map<std::uint64_t, HoloLiftRange>
      component_membership_by_id;
  std::unordered_map<std::uint64_t, HoloLiftRange> layout_components_by_id;
  std::size_t next_component = 0;
  std::size_t next_atom = 0;
  std::size_t next_hard_edge = 0;
  for (std::size_t epoch_index = 0;
       epoch_index < store.topology_epochs.size(); ++epoch_index) {
    const auto &epoch = store.topology_epochs[epoch_index];
    if (!epoch_ids.insert(epoch.topology_epoch_id).second) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::DuplicateTopologyEpoch,
          "duplicate HoloLift topology epoch id", epoch_index));
    }
    if (epoch.topology_epoch_identity.empty() ||
        !epoch_identities.insert(epoch.topology_epoch_identity).second) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::DuplicateTopologyEpoch,
          "empty or duplicate HoloLift 128-bit topology epoch identity",
          epoch_index));
    }
    if (epoch.components.begin != next_component) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidRange,
          "HoloLift topology-epoch component tables are not append-packed",
          epoch_index));
    }
    if (epoch.hard_edges.begin != next_hard_edge) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidRange,
          "HoloLift topology-epoch hard-edge table is not append-packed",
          epoch_index));
    }
    const auto epoch_validation =
        validate_hololift_topology_epoch(store, epoch_index);
    if (!epoch_validation)
      return epoch_validation;
    const auto components =
        std::span<const HoloLiftComponentRecord>(store.components)
            .subspan(epoch.components.begin, epoch.components.count);
    for (std::size_t local_idx = 0; local_idx < components.size();
         ++local_idx) {
      const auto &component = components[local_idx];
      if (component.exact_atom_membership.begin != next_atom) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidRange,
            "HoloLift exact-membership table is not append-packed",
            epoch_index));
      }
      const auto membership =
          std::span<const HoloLiftSourceAtomKey>(
              store.exact_atom_memberships)
              .subspan(component.exact_atom_membership.begin,
                       component.exact_atom_membership.count);
      const auto [component_it, component_inserted] =
          component_membership_by_id.emplace(
              component.component_id, component.exact_atom_membership);
      if (!component_inserted) {
        const auto existing =
            std::span<const HoloLiftSourceAtomKey>(
                store.exact_atom_memberships)
                .subspan(component_it->second.begin,
                         component_it->second.count);
        if (existing.size() != membership.size() ||
            !std::equal(existing.begin(), existing.end(),
                        membership.begin())) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::ComponentIdentityCollision,
            "HoloLift component id collision across topology epochs",
            epoch_index));
        }
      }
      next_atom += component.exact_atom_membership.count;
    }
    const auto [layout_it, layout_inserted] =
        layout_components_by_id.emplace(epoch.layout_signature,
                                        epoch.components);
    if (!layout_inserted) {
      const auto existing_components =
          std::span<const HoloLiftComponentRecord>(store.components)
              .subspan(layout_it->second.begin, layout_it->second.count);
      const bool same_layout =
          existing_components.size() == components.size() &&
          std::equal(existing_components.begin(), existing_components.end(),
                     components.begin(), [](const auto &lhs, const auto &rhs) {
                       return lhs.component_id == rhs.component_id;
                     });
      if (!same_layout) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::LayoutIdentityCollision,
            "HoloLift layout signature collision across topology epochs",
            epoch_index));
      }
    }
    next_component += epoch.components.count;
    next_hard_edge += epoch.hard_edges.count;
  }
  if (next_component != store.components.size() ||
      next_atom != store.exact_atom_memberships.size() ||
      next_hard_edge != store.hard_edges.size()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift topology tables contain orphan records"));
  }

  std::size_t next_candidate = 0;
  std::size_t next_image = 0;
  std::vector<HoloLiftHash128> epoch_selection_hashes(
      store.topology_epochs.size());
  std::vector<unsigned char> epoch_selection_hash_known(
      store.topology_epochs.size(), 0);
  for (std::size_t frame_index = 0; frame_index < store.frames.size();
       ++frame_index) {
    const auto &frame = store.frames[frame_index];
    if (frame.candidates.begin != next_candidate) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidRange,
          "HoloLift frame candidate table is not append-packed",
          HOLOLIFT_NO_INDEX, frame_index));
    }
    if (frame_index != 0) {
      const auto &previous = store.frames[frame_index - 1];
      if (frame.frame <= previous.frame) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameOrder,
            "HoloLift frame ids are not strictly increasing",
            HOLOLIFT_NO_INDEX, frame_index));
      }
      if (frame.time_ps < previous.time_ps) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameTime,
            "HoloLift frame times are not nondecreasing",
            HOLOLIFT_NO_INDEX, frame_index));
      }
      if (frame.source_sequence_index <=
          previous.source_sequence_index) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameOrder,
            "HoloLift source sequence indices are not strictly increasing",
            HOLOLIFT_NO_INDEX, frame_index));
      }
      if (frame.trajectory_frame_index <= previous.trajectory_frame_index) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameBinding,
            "HoloLift trajectory-frame indices are not strictly increasing",
            HOLOLIFT_NO_INDEX, frame_index));
      }
      if (frame.source_relation ==
              HoloLiftSourceFrameRelation::ContiguousSourceFrame &&
          (frame.source_sequence_index !=
               previous.source_sequence_index + 1 ||
           frame.trajectory_frame_index !=
               previous.trajectory_frame_index + 1 ||
           frame.topology_epoch_id != previous.topology_epoch_id)) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameProvenance,
            "HoloLift contiguous source relation crosses a gap or topology "
            "epoch",
            HOLOLIFT_NO_INDEX, frame_index));
      }
    } else if (frame.source_relation !=
               HoloLiftSourceFrameRelation::SegmentStart) {
      return std::unexpected(make_error(
          HoloLiftValidationCode::InvalidFrameProvenance,
          "the first HoloLift frame must start an observation segment",
          HOLOLIFT_NO_INDEX, frame_index));
    }
    if (frame.topology_epoch_index < epoch_selection_hashes.size()) {
      const std::size_t epoch_index = frame.topology_epoch_index;
      if (!epoch_selection_hash_known[epoch_index]) {
        epoch_selection_hashes[epoch_index] = frame.atom_selection_hash;
        epoch_selection_hash_known[epoch_index] = 1;
      } else if (epoch_selection_hashes[epoch_index] !=
                 frame.atom_selection_hash) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidFrameBinding,
            "HoloLift atom-selection hash changed inside a topology epoch",
            epoch_index, frame_index));
      }
    }
    for (std::size_t local_idx = 0; local_idx < frame.candidates.count;
         ++local_idx) {
      const auto &candidate =
          store.candidates[frame.candidates.begin + local_idx];
      if (candidate.component_images.begin != next_image) {
        return std::unexpected(make_error(
            HoloLiftValidationCode::InvalidRange,
            "HoloLift component-image table is not append-packed",
            HOLOLIFT_NO_INDEX, frame_index,
            frame.candidates.begin + local_idx));
      }
      next_image += candidate.component_images.count;
    }
    const auto frame_validation =
        validate_hololift_frame(store, frame_index);
    if (!frame_validation)
      return frame_validation;
    next_candidate += frame.candidates.count;
  }
  if (next_candidate != store.candidates.size() ||
      next_image != store.component_images.size()) {
    return std::unexpected(make_error(
        HoloLiftValidationCode::InvalidRange,
        "HoloLift frame tables contain orphan records"));
  }
  return {};
}

struct HoloLiftObservationStoreFinalizer {
  static HoloLiftObservationStore
  make(HoloLiftObservationStoreBuilder &&builder) {
    return HoloLiftObservationStore(std::move(builder));
  }
};

std::expected<HoloLiftObservationStore, HoloLiftValidationError>
finalize_hololift_observation_store(
    HoloLiftObservationStoreBuilder &&builder) {
  const auto validation =
      validate_hololift_observation_store(builder);
  if (!validation)
    return std::unexpected(validation.error());
  return HoloLiftObservationStoreFinalizer::make(std::move(builder));
}

} // namespace titan_hololift

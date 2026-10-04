#include "tabi_ses_patch_projection.h"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string_view>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

struct ClosedInterval {
  double lower{0.0};
  double upper{0.0};
};

[[nodiscard]] bool finite(const Vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool rounded_interval(double value,
                                    ClosedInterval &result) noexcept {
  if (!std::isfinite(value))
    return false;
  result.lower =
      std::nextafter(value, -std::numeric_limits<double>::infinity());
  result.upper = std::nextafter(value, std::numeric_limits<double>::infinity());
  return std::isfinite(result.lower) && std::isfinite(result.upper);
}

[[nodiscard]] bool interval_subtract(double left, double right,
                                     ClosedInterval &result) noexcept {
  volatile double rounded = left - right;
  return rounded_interval(rounded, result);
}

[[nodiscard]] bool interval_add(const ClosedInterval &left,
                                const ClosedInterval &right,
                                ClosedInterval &result) noexcept {
  volatile double lower = left.lower + right.lower;
  volatile double upper = left.upper + right.upper;
  if (!std::isfinite(lower) || !std::isfinite(upper))
    return false;
  result.lower =
      std::nextafter(lower, -std::numeric_limits<double>::infinity());
  result.upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
  return std::isfinite(result.lower) && std::isfinite(result.upper);
}

[[nodiscard]] bool interval_subtract(const ClosedInterval &left,
                                     const ClosedInterval &right,
                                     ClosedInterval &result) noexcept {
  volatile double lower = left.lower - right.upper;
  volatile double upper = left.upper - right.lower;
  if (!std::isfinite(lower) || !std::isfinite(upper))
    return false;
  result.lower =
      std::nextafter(lower, -std::numeric_limits<double>::infinity());
  result.upper = std::nextafter(upper, std::numeric_limits<double>::infinity());
  return std::isfinite(result.lower) && std::isfinite(result.upper);
}

[[nodiscard]] bool interval_multiply(const ClosedInterval &left,
                                     const ClosedInterval &right,
                                     ClosedInterval &result) noexcept {
  std::array<double, 4> products{};
  volatile double p0 = left.lower * right.lower;
  volatile double p1 = left.lower * right.upper;
  volatile double p2 = left.upper * right.lower;
  volatile double p3 = left.upper * right.upper;
  products = {p0, p1, p2, p3};
  if (!std::all_of(products.begin(), products.end(),
                   [](double value) { return std::isfinite(value); }))
    return false;
  const auto [minimum, maximum] =
      std::minmax_element(products.begin(), products.end());
  result.lower =
      std::nextafter(*minimum, -std::numeric_limits<double>::infinity());
  result.upper =
      std::nextafter(*maximum, std::numeric_limits<double>::infinity());
  return std::isfinite(result.lower) && std::isfinite(result.upper);
}

[[nodiscard]] bool exact_interval(double value,
                                  ClosedInterval &result) noexcept {
  if (!std::isfinite(value))
    return false;
  result = {value, value};
  return true;
}

[[nodiscard]] bool bit_equal(double left, double right) noexcept {
  return std::bit_cast<std::uint64_t>(left) ==
         std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] bool bit_equal(const Vec3 &left, const Vec3 &right) noexcept {
  return bit_equal(left.x, right.x) && bit_equal(left.y, right.y) &&
         bit_equal(left.z, right.z);
}

template <class Value>
[[nodiscard]] bool safe_subspan(const std::vector<Value> &values,
                                std::size_t begin, std::size_t end,
                                std::span<const Value> &result) noexcept {
  result = {};
  if (begin > end || end > values.size())
    return false;
  if (begin == end)
    return true;
  result = {values.data() + begin, end - begin};
  return true;
}

[[nodiscard]] bool same_canonical_vertex(
    const PatchGridCanonicalInterpolationVertexLineage &left,
    const PatchGridCanonicalInterpolationVertexLineage &right) noexcept {
  return left.key == right.key &&
         bit_equal(left.canonical_interpolation_point,
                   right.canonical_interpolation_point) &&
         bit_equal(left.canonical_interpolation_parameter,
                   right.canonical_interpolation_parameter) &&
         left.exact_location_crossing == right.exact_location_crossing &&
         left.canonical_interpolation_complete ==
             right.canonical_interpolation_complete &&
         left.canonical_edge_occurrence_geometry_bitwise_consistent ==
             right.canonical_edge_occurrence_geometry_bitwise_consistent &&
         left.directed_production_geometry_bitwise_certified ==
             right.directed_production_geometry_bitwise_certified &&
         left.patch_candidate_collection_complete ==
             right.patch_candidate_collection_complete &&
         left.patch_relation_classification_complete ==
             right.patch_relation_classification_complete &&
         left.exact_analytic_location_certified ==
             right.exact_analytic_location_certified &&
         left.candidate_refs_present_without_exact_location ==
             right.candidate_refs_present_without_exact_location;
}

[[nodiscard]] bool same_relation(
    const PatchGridCanonicalVertexPatchRelationEvidence &left,
    const PatchGridCanonicalVertexPatchRelationEvidence &right) noexcept {
  return left.first == right.first && left.second == right.second &&
         left.relation == right.relation;
}

[[nodiscard]] bool same_triangle(
    const PatchGridPreOrientationTriangleOccurrenceLineage &left,
    const PatchGridPreOrientationTriangleOccurrenceLineage &right) noexcept {
  return left.tetra_position == right.tetra_position &&
         left.triangle_ordinal == right.triangle_ordinal &&
         left.source_edge_occurrence_positions ==
             right.source_edge_occurrence_positions &&
         left.canonical_vertex_positions == right.canonical_vertex_positions &&
         bit_equal(left.canonical_twice_area, right.canonical_twice_area) &&
         left.single_common_candidate_ref ==
             right.single_common_candidate_ref &&
         left.orientation_stage == right.orientation_stage &&
         left.finite_geometry_observed == right.finite_geometry_observed &&
         left.computed_zero_area_observed ==
             right.computed_zero_area_observed &&
         left.patch_candidate_summary_complete ==
             right.patch_candidate_summary_complete &&
         left.single_common_candidate_ref_available ==
             right.single_common_candidate_ref_available &&
         left.vertex_triplet_patch_identity_certified ==
             right.vertex_triplet_patch_identity_certified &&
         left.planar_face_surface_location_certified ==
             right.planar_face_surface_location_certified &&
         left.final_mesh_orientation_certified ==
             right.final_mesh_orientation_certified;
}

[[nodiscard]] bool vertex_payload_counters_consistent(
    const PreparedPatchGridVertexMagnitudeWitnesses &prepared) noexcept {
  if (prepared.vertices.empty()) {
    return prepared.candidate_offsets.empty() && prepared.candidates.empty() &&
           prepared.candidate_refs.empty() &&
           prepared.analytic_candidates == 0U &&
           prepared.certified_analytic_candidates == 0U &&
           prepared.uncertified_analytic_candidates == 0U &&
           prepared.sparse_candidates == 0U &&
           prepared.fallback_candidates == 0U &&
           prepared.clamp_floor_candidates == 0U &&
           prepared.raw_exact_owner_candidates == 0U &&
           prepared.tolerance_tied_candidates == 0U &&
           prepared.final_scalar_exact_owner_candidates == 0U &&
           prepared.raw_source_certified_vertices == 0U &&
           prepared.final_scalar_source_certified_vertices == 0U &&
           prepared.floor_clamped_vertices == 0U &&
           prepared.sparse_suppressed_yz_vertices == 0U &&
           prepared.topology_analytic_evidence_vertices == 0U &&
           prepared.topology_exact_vertex_certified_vertices == 0U &&
           prepared.topology_nonanalytic_vertices == 0U &&
           !prepared.raw_source_certificate_complete &&
           !prepared.final_scalar_source_certificate_complete &&
           !prepared.topology_source_classification_complete &&
           !prepared.topology_exact_vertex_certificate_complete &&
           !prepared.surface_location_certified &&
           prepared.status == "no_vertices";
  }
  std::size_t analytic = 0U;
  std::size_t certified_analytic = 0U;
  std::size_t uncertified_analytic = 0U;
  std::size_t sparse = 0U;
  std::size_t fallback = 0U;
  std::size_t clamp = 0U;
  std::size_t raw_exact = 0U;
  std::size_t tied = 0U;
  std::size_t final_exact = 0U;
  for (const PatchGridMagnitudeSourceCandidate &candidate :
       prepared.candidates) {
    switch (candidate.kind) {
    case PatchGridMagnitudeSourceKind::analytic_crossing:
      ++analytic;
      if (candidate.source_evidence_certified())
        ++certified_analytic;
      else
        ++uncertified_analytic;
      break;
    case PatchGridMagnitudeSourceKind::sparse_sample:
      ++sparse;
      break;
    case PatchGridMagnitudeSourceKind::grid_spacing_fallback:
      ++fallback;
      break;
    case PatchGridMagnitudeSourceKind::clamp_floor:
      ++clamp;
      break;
    }
    raw_exact += static_cast<std::size_t>(candidate.raw_exact_owner);
    tied += static_cast<std::size_t>(candidate.raw_tolerance_tied);
    final_exact += static_cast<std::size_t>(candidate.final_scalar_exact_owner);
  }

  std::size_t raw_vertices = 0U;
  std::size_t final_vertices = 0U;
  std::size_t floor_vertices = 0U;
  std::size_t sparse_suppressed = 0U;
  std::size_t topology_analytic = 0U;
  std::size_t topology_exact = 0U;
  std::size_t topology_nonanalytic = 0U;
  bool all_raw = true;
  bool all_final = true;
  bool all_topology_classified = true;
  bool all_topology_exact = true;
  for (const PatchGridVertexMagnitudeWitness &vertex : prepared.vertices) {
    const PatchGridTopologyEvidenceClassification expected_topology =
        classify_patch_grid_topology_evidence(
            vertex.topology_state, vertex.topology_source, vertex.cavity_fill);
    if (vertex.topology_evidence_status != expected_topology.status ||
        vertex.topology_analytic_evidence !=
            expected_topology.analytic_evidence ||
        vertex.topology_exact_vertex_certified !=
            expected_topology.exact_vertex_certified)
      return false;
    raw_vertices +=
        static_cast<std::size_t>(vertex.raw_source_certificate_complete);
    final_vertices += static_cast<std::size_t>(
        vertex.final_scalar_source_certificate_complete);
    floor_vertices += static_cast<std::size_t>(vertex.floor_clamp_applied);
    sparse_suppressed += static_cast<std::size_t>(vertex.sparse_suppressed_yz);
    all_raw &= vertex.raw_source_certificate_complete;
    all_final &= vertex.final_scalar_source_certificate_complete;
    if (vertex.topology_evidence_status ==
        PatchGridTopologyEvidenceStatus::unknown) {
      all_topology_classified = false;
      all_topology_exact = false;
    } else if (vertex.topology_analytic_evidence) {
      ++topology_analytic;
      if (vertex.topology_exact_vertex_certified)
        ++topology_exact;
      else
        all_topology_exact = false;
    } else {
      ++topology_nonanalytic;
      all_topology_exact = false;
    }
  }
  const bool raw_complete = all_raw &&
                            raw_vertices == prepared.vertices.size() &&
                            certified_analytic != 0U;
  const bool final_complete =
      all_final && final_vertices == prepared.vertices.size() && raw_complete;
  const bool topology_exact_complete =
      all_topology_exact && topology_exact == prepared.vertices.size();
  const bool expected_status =
      (certified_analytic == 0U &&
       prepared.status == "no_usable_analytic_evidence") ||
      (certified_analytic != 0U && !raw_complete &&
       prepared.status == "raw_source_uncertified") ||
      (raw_complete && !final_complete &&
       prepared.status == "final_scalar_source_uncertified") ||
      (final_complete && !all_topology_classified &&
       prepared.status == "topology_source_unclassified") ||
      (final_complete && all_topology_classified && !topology_exact_complete &&
       prepared.status == "topology_exact_vertex_uncertified") ||
      (final_complete && all_topology_classified && topology_exact_complete &&
       prepared.status == "pass");
  return analytic == prepared.analytic_candidates &&
         certified_analytic == prepared.certified_analytic_candidates &&
         uncertified_analytic == prepared.uncertified_analytic_candidates &&
         sparse == prepared.sparse_candidates &&
         fallback == prepared.fallback_candidates &&
         clamp == prepared.clamp_floor_candidates &&
         raw_exact == prepared.raw_exact_owner_candidates &&
         tied == prepared.tolerance_tied_candidates &&
         final_exact == prepared.final_scalar_exact_owner_candidates &&
         raw_vertices == prepared.raw_source_certified_vertices &&
         final_vertices == prepared.final_scalar_source_certified_vertices &&
         floor_vertices == prepared.floor_clamped_vertices &&
         sparse_suppressed == prepared.sparse_suppressed_yz_vertices &&
         topology_analytic == prepared.topology_analytic_evidence_vertices &&
         topology_exact == prepared.topology_exact_vertex_certified_vertices &&
         topology_nonanalytic == prepared.topology_nonanalytic_vertices &&
         prepared.raw_source_certificate_complete == raw_complete &&
         prepared.final_scalar_source_certificate_complete == final_complete &&
         prepared.topology_source_classification_complete ==
             all_topology_classified &&
         prepared.topology_exact_vertex_certificate_complete ==
             topology_exact_complete &&
         !prepared.surface_location_certified && expected_status;
}

[[nodiscard]] bool cell_payload_summaries_consistent(
    const PreparedPatchGridCellLineage &prepared) noexcept {
  if (prepared.cells.empty()) {
    return prepared.cell_tetra_offsets.empty() && prepared.tetrahedra.empty() &&
           prepared.tetra_edge_offsets.empty() && prepared.edges.empty() &&
           prepared.edge_endpoint_patch_offsets.empty() &&
           prepared.endpoint_patch_refs.empty() &&
           prepared.edge_relation_offsets.empty() &&
           prepared.relations.empty() &&
           prepared.edge_location_ref_offsets.empty() &&
           prepared.location_patch_refs.empty() &&
           prepared.relation_counts == std::array<std::size_t, 5>{} &&
           prepared.axis_edges == 0U && prepared.diagonal_edges == 0U &&
           prepared.scalar_lineage_certified_edges == 0U &&
           prepared.surface_location_certified_edges == 0U &&
           prepared.topology_sign_certified_edges == 0U &&
           !prepared.scalar_lineage_complete &&
           !prepared.patch_candidate_collection_complete &&
           !prepared.patch_relation_classification_complete &&
           !prepared.surface_location_certificate_complete &&
           !prepared.topology_sign_certificate_complete &&
           prepared.status == "empty_active_cells";
  }
  std::array<std::size_t, 5> relation_counts{};
  for (const PatchGridEdgePatchRelationEvidence &evidence :
       prepared.relations) {
    const std::size_t slot = static_cast<std::size_t>(evidence.relation);
    if (slot >= relation_counts.size())
      return false;
    ++relation_counts[slot];
  }
  std::size_t axis = 0U;
  std::size_t diagonal = 0U;
  std::size_t scalar = 0U;
  std::size_t location = 0U;
  std::size_t topology = 0U;
  bool all_scalar = true;
  bool all_candidates = true;
  bool all_relations = true;
  bool any_relation = false;
  bool all_locations = true;
  bool all_topology = true;
  if (prepared.edge_relation_offsets.size() != prepared.edges.size() + 1U)
    return false;
  for (std::size_t position = 0U; position < prepared.edges.size();
       ++position) {
    const PatchGridSignChangingEdgeLineage &edge = prepared.edges[position];
    if (edge.cartesian_axis_edge)
      ++axis;
    else
      ++diagonal;
    scalar += static_cast<std::size_t>(edge.scalar_lineage_complete);
    location += static_cast<std::size_t>(edge.surface_location_certified);
    topology += static_cast<std::size_t>(edge.topology_sign_evidence_complete);
    all_scalar &= edge.scalar_lineage_complete;
    all_candidates &= edge.patch_candidate_collection_complete;
    all_relations &= edge.patch_relation_classification_complete;
    all_locations &= edge.surface_location_certified;
    all_topology &= edge.topology_sign_evidence_complete;
    const std::size_t begin = prepared.edge_relation_offsets[position];
    const std::size_t end = prepared.edge_relation_offsets[position + 1U];
    if (begin > end || end > prepared.relations.size())
      return false;
    any_relation |= begin != end;
  }
  const bool have_edges = !prepared.edges.empty();
  const bool scalar_complete =
      have_edges && all_scalar && scalar == prepared.edges.size();
  const bool candidate_complete = have_edges && all_candidates;
  const bool relation_complete = have_edges && any_relation && all_relations;
  const bool location_complete =
      have_edges && all_locations && location == prepared.edges.size();
  const bool topology_complete =
      have_edges && all_topology && topology == prepared.edges.size();
  const bool expected_pass = scalar_complete && candidate_complete &&
                             relation_complete && location_complete &&
                             topology_complete;
  return relation_counts == prepared.relation_counts &&
         axis == prepared.axis_edges && diagonal == prepared.diagonal_edges &&
         scalar == prepared.scalar_lineage_certified_edges &&
         location == prepared.surface_location_certified_edges &&
         topology == prepared.topology_sign_certified_edges &&
         prepared.scalar_lineage_complete == scalar_complete &&
         prepared.patch_candidate_collection_complete == candidate_complete &&
         prepared.patch_relation_classification_complete == relation_complete &&
         prepared.surface_location_certificate_complete == location_complete &&
         prepared.topology_sign_certificate_complete == topology_complete &&
         prepared.status ==
             (expected_pass ? "pass" : "pass_with_uncertified_edges");
}

[[nodiscard]] bool same_c2a_semantics(
    const PreparedPatchGridTriangleOccurrenceLineage &left,
    const PreparedPatchGridTriangleOccurrenceLineage &right) noexcept {
  if (left.complex_fingerprint != right.complex_fingerprint ||
      left.lattice_identity != right.lattice_identity ||
      left.edge_occurrence_to_canonical_vertex !=
          right.edge_occurrence_to_canonical_vertex ||
      left.canonical_vertex_occurrence_offsets !=
          right.canonical_vertex_occurrence_offsets ||
      left.canonical_vertex_edge_occurrences !=
          right.canonical_vertex_edge_occurrences ||
      left.canonical_vertex_candidate_ref_offsets !=
          right.canonical_vertex_candidate_ref_offsets ||
      left.canonical_vertex_candidate_refs !=
          right.canonical_vertex_candidate_refs ||
      left.canonical_vertex_relation_offsets !=
          right.canonical_vertex_relation_offsets ||
      left.canonical_vertex_exact_location_ref_offsets !=
          right.canonical_vertex_exact_location_ref_offsets ||
      left.canonical_vertex_exact_location_refs !=
          right.canonical_vertex_exact_location_refs ||
      left.triangle_candidate_ref_offsets !=
          right.triangle_candidate_ref_offsets ||
      left.triangle_candidate_refs != right.triangle_candidate_refs ||
      left.relation_counts != right.relation_counts ||
      left.interpolation_occurrences != right.interpolation_occurrences ||
      left.canonical_interpolation_vertices !=
          right.canonical_interpolation_vertices ||
      left.bitwise_interpolation_conflicts !=
          right.bitwise_interpolation_conflicts ||
      left.exact_location_vertices != right.exact_location_vertices ||
      left.candidate_refs_without_exact_location_vertices !=
          right.candidate_refs_without_exact_location_vertices ||
      left.triangle_occurrences != right.triangle_occurrences ||
      left.finite_triangle_occurrences != right.finite_triangle_occurrences ||
      left.computed_zero_area_triangle_occurrences !=
          right.computed_zero_area_triangle_occurrences ||
      left.nonfinite_triangle_occurrences !=
          right.nonfinite_triangle_occurrences ||
      left.single_common_candidate_triangles !=
          right.single_common_candidate_triangles ||
      left.vertex_triplet_patch_identity_certified_triangles !=
          right.vertex_triplet_patch_identity_certified_triangles ||
      left.interpolation_identity_complete !=
          right.interpolation_identity_complete ||
      left.patch_candidate_attribution_complete !=
          right.patch_candidate_attribution_complete ||
      left.triangle_occurrence_lineage_complete !=
          right.triangle_occurrence_lineage_complete ||
      left.vertex_triplet_patch_identity_classification_complete !=
          right.vertex_triplet_patch_identity_classification_complete ||
      left.planar_face_surface_location_certified !=
          right.planar_face_surface_location_certified ||
      left.final_mesh_orientation_certified !=
          right.final_mesh_orientation_certified ||
      left.status != right.status ||
      left.canonical_vertices.size() != right.canonical_vertices.size() ||
      left.canonical_vertex_relations.size() !=
          right.canonical_vertex_relations.size() ||
      left.triangles.size() != right.triangles.size())
    return false;
  for (std::size_t index = 0U; index < left.canonical_vertices.size();
       ++index) {
    if (!same_canonical_vertex(left.canonical_vertices[index],
                               right.canonical_vertices[index]))
      return false;
  }
  for (std::size_t index = 0U; index < left.canonical_vertex_relations.size();
       ++index) {
    if (!same_relation(left.canonical_vertex_relations[index],
                       right.canonical_vertex_relations[index]))
      return false;
  }
  for (std::size_t index = 0U; index < left.triangles.size(); ++index) {
    if (!same_triangle(left.triangles[index], right.triangles[index]))
      return false;
  }
  return true;
}

[[nodiscard]] bool ref_in_range(const PatchComplex &complex,
                                PatchRefId ref) noexcept {
  if (!ref.valid())
    return false;
  switch (ref.kind()) {
  case PatchKind::contact:
    return ref.index() < complex.contacts.size();
  case PatchKind::toroidal:
    return ref.index() < complex.tori.size();
  case PatchKind::reentrant:
    return ref.index() < complex.reentrant.size();
  }
  return false;
}

[[nodiscard]] bool checked_add(std::size_t &value,
                               std::size_t increment) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - increment)
    return false;
  value += increment;
  return true;
}

// Audit-local consistency validation only. Geometry was already checked by
// Slice A; this pass derives every scientific summary from its immutable
// witness records and binds those summaries back to ray provenance.
[[nodiscard]] bool crossing_payload_summaries_consistent(
    const PreparedPatchCrossingWitnesses &prepared,
    const PatchRayProvenance &provenance) noexcept {
  if (prepared.axis_offsets[0] != 0U ||
      prepared.axis_offsets[3] != prepared.witnesses.size())
    return false;
  for (std::size_t axis = 0U; axis < 3U; ++axis) {
    if (prepared.axis_offsets[axis] > prepared.axis_offsets[axis + 1U])
      return false;
  }

  std::size_t refs = 0U;
  std::size_t roots = 0U;
  std::size_t identity = 0U;
  std::size_t reproduced = 0U;
  std::size_t locations = 0U;
  std::size_t transversal = 0U;
  std::size_t multi = 0U;
  std::size_t multi_uncertified = 0U;
  std::size_t simplicity_uncertified = 0U;
  std::size_t repeated = 0U;
  std::size_t intersection_uncertified = 0U;
  std::size_t roots_not_reproduced = 0U;
  std::size_t multiplicity_mismatches = 0U;
  std::size_t projector_uncertified = 0U;
  std::size_t patch_residual_failures = 0U;
  std::size_t nonfinite_bounded_residuals = 0U;
  std::size_t maximum_refs = 0U;

  for (const CertifiedPatchCrossingWitness &witness : prepared.witnesses) {
    if (witness.ref_count == 0U || witness.total_root_multiplicity == 0U ||
        witness.maximum_ref_multiplicity == 0U ||
        witness.maximum_ref_multiplicity > witness.total_root_multiplicity ||
        !checked_add(refs, witness.ref_count) ||
        !checked_add(roots, witness.total_root_multiplicity))
      return false;
    maximum_refs =
        std::max(maximum_refs, static_cast<std::size_t>(witness.ref_count));
    if (witness.ref_count > 1U && !checked_add(multi, 1U))
      return false;
    if (witness.maximum_ref_multiplicity > 1U && !checked_add(repeated, 1U))
      return false;

    switch (witness.identity_status) {
    case CrossingIdentityCertificateStatus::valid_single_ref:
      if (witness.ref_count != 1U || !checked_add(identity, 1U))
        return false;
      break;
    case CrossingIdentityCertificateStatus::valid_multi_ref:
      if (witness.ref_count <= 1U || !checked_add(identity, 1U))
        return false;
      break;
    case CrossingIdentityCertificateStatus::invalid:
      break;
    default:
      return false;
    }

    switch (witness.root_status) {
    case CrossingRootReproductionStatus::reproduced:
      if (!checked_add(reproduced, 1U))
        return false;
      break;
    case CrossingRootReproductionStatus::intersection_uncertified:
      if (!checked_add(intersection_uncertified, 1U))
        return false;
      break;
    case CrossingRootReproductionStatus::root_not_reproduced:
      if (!checked_add(roots_not_reproduced, 1U))
        return false;
      break;
    case CrossingRootReproductionStatus::root_multiplicity_mismatch:
      if (!checked_add(multiplicity_mismatches, 1U))
        return false;
      break;
    case CrossingRootReproductionStatus::nonfinite_residual:
      break;
    default:
      return false;
    }

    switch (witness.location_status) {
    case CrossingBoundedLocationStatus::certified:
      if (!witness.root_reproduced() || !checked_add(locations, 1U))
        return false;
      break;
    case CrossingBoundedLocationStatus::root_unreproduced:
      if (witness.root_reproduced())
        return false;
      break;
    case CrossingBoundedLocationStatus::projector_uncertified:
      if (!checked_add(projector_uncertified, 1U))
        return false;
      break;
    case CrossingBoundedLocationStatus::patch_residual_failed:
      if (!checked_add(patch_residual_failures, 1U))
        return false;
      break;
    case CrossingBoundedLocationStatus::nonfinite_residual:
      if (!checked_add(nonfinite_bounded_residuals, 1U))
        return false;
      break;
    default:
      return false;
    }

    switch (witness.transversality_status) {
    case CrossingTransversalityStatus::
        primitive_filtered_transversal_unoriented:
      if (!witness.location_certified() || witness.ref_count != 1U ||
          witness.maximum_ref_multiplicity != 1U ||
          !witness.primitive_simplicity_certified ||
          !checked_add(transversal, 1U))
        return false;
      break;
    case CrossingTransversalityStatus::repeated_root_ambiguous:
      if (witness.maximum_ref_multiplicity <= 1U)
        return false;
      break;
    case CrossingTransversalityStatus::multi_ref_relation_uncertified:
      if (witness.ref_count <= 1U || !checked_add(multi_uncertified, 1U))
        return false;
      break;
    case CrossingTransversalityStatus::primitive_simplicity_uncertified:
      if (!witness.location_certified() ||
          witness.primitive_simplicity_certified ||
          !checked_add(simplicity_uncertified, 1U))
        return false;
      break;
    case CrossingTransversalityStatus::location_uncertified:
      if (witness.location_certified())
        return false;
      break;
    default:
      return false;
    }
    if (witness.orientation_sign_certified)
      return false;
  }

  const bool identity_complete = identity == prepared.witnesses.size();
  const bool reproduction_complete = reproduced == prepared.witnesses.size();
  const bool location_complete = locations == prepared.witnesses.size();
  const bool transversality_complete = transversal == prepared.witnesses.size();
  std::string_view expected_status = "pass";
  if (!identity_complete)
    expected_status = "identity_uncertified";
  else if (!reproduction_complete)
    expected_status = "root_reproduction_uncertified";
  else if (!location_complete)
    expected_status = "location_uncertified";
  else if (!transversality_complete)
    expected_status =
        repeated != 0U            ? "repeated_root_transversality_ambiguous"
        : multi_uncertified != 0U ? "multi_ref_relation_uncertified"
        : simplicity_uncertified != 0U ? "primitive_simplicity_uncertified"
                                       : "transversality_uncertified";

  return prepared.witnesses.size() == provenance.unique_crossings &&
         refs == prepared.refs_audited &&
         roots == prepared.root_events_audited &&
         identity == prepared.identity_certified_crossings &&
         reproduced == prepared.root_reproduced_crossings &&
         locations == prepared.bounded_patch_location_certified_crossings &&
         transversal == prepared.transversality_certified_crossings &&
         multi == prepared.multi_ref_crossings &&
         multi_uncertified ==
             prepared.multi_ref_relation_uncertified_crossings &&
         simplicity_uncertified ==
             prepared.primitive_simplicity_uncertified_crossings &&
         repeated == prepared.repeated_root_ambiguities &&
         intersection_uncertified == prepared.intersection_uncertified &&
         roots_not_reproduced == prepared.roots_not_reproduced &&
         multiplicity_mismatches == prepared.root_multiplicity_mismatches &&
         projector_uncertified == prepared.projector_uncertified &&
         patch_residual_failures == prepared.patch_residual_failures &&
         nonfinite_bounded_residuals == prepared.nonfinite_bounded_residuals &&
         prepared.identity_certificate_complete == identity_complete &&
         prepared.root_reproduction_complete == reproduction_complete &&
         prepared.location_certificate_complete == location_complete &&
         prepared.transversality_certificate_complete ==
             transversality_complete &&
         prepared.status == expected_status && refs == provenance.refs_total &&
         roots == provenance.roots_total && multi == provenance.seam_groups &&
         repeated == provenance.repeated_root_groups &&
         repeated == provenance.same_primitive_near_pair_unclassified &&
         maximum_refs == provenance.max_refs_per_crossing;
}

[[nodiscard]] bool checked_payload(std::size_t capacity, std::size_t width,
                                   std::size_t &bytes) noexcept {
  if (capacity > std::numeric_limits<std::size_t>::max() / width)
    return false;
  return checked_add(bytes, capacity * width);
}

[[nodiscard]] bool
projection_payload_bytes(const PreparedPatchGridProjectionAudit &result,
                         std::size_t &bytes) noexcept {
  bytes = 0U;
  return checked_payload(result.vertices.capacity(),
                         sizeof(PatchGridCanonicalVertexProjectionAudit),
                         bytes) &&
         checked_payload(result.vertex_projection_offsets.capacity(),
                         sizeof(std::size_t), bytes) &&
         checked_payload(result.candidate_projections.capacity(),
                         sizeof(PatchGridBoundedCandidateProjection), bytes) &&
         checked_payload(result.triangles.capacity(),
                         sizeof(PatchGridTriangleProjectionAudit), bytes);
}

struct GridVertexIndex {
  std::size_t x{0U};
  std::size_t y{0U};
  std::size_t z{0U};
};

[[nodiscard]] bool decode_grid_vertex(const PatchMappingGridSpec &spec,
                                      std::size_t linear,
                                      GridVertexIndex &result) noexcept {
  std::size_t vertex_count = 0U;
  if (!validate_patch_mapping_grid_spec(spec, nullptr, &vertex_count) ||
      linear >= vertex_count)
    return false;
  const std::size_t original = linear;
  result.x = linear % spec.nx;
  linear /= spec.nx;
  result.y = linear % spec.ny;
  result.z = linear / spec.ny;
  std::size_t reconstructed = 0U;
  return result.z < spec.nz &&
         try_patch_grid_vertex_linear_index(spec, result.x, result.y, result.z,
                                            reconstructed) &&
         reconstructed == original;
}

[[nodiscard]] bool upward_lattice_distance(double spacing, unsigned l1_steps,
                                           double &bound) noexcept {
  bound = 0.0;
  if (!std::isfinite(spacing) || !(spacing > 0.0) || l1_steps == 0U ||
      l1_steps > 3U)
    return false;
  // L1 is a conservative upper bound for every Euclidean tetra-edge chord.
  // At most three explicitly sequenced additions are outward-enclosed, so no
  // libm square-root accuracy assumption enters the projector radius.
  bound = std::nextafter(spacing, std::numeric_limits<double>::infinity());
  for (unsigned step = 1U; step < l1_steps; ++step) {
    volatile double sum = bound + spacing;
    if (!std::isfinite(sum))
      return false;
    bound = std::nextafter(sum, std::numeric_limits<double>::infinity());
  }
  return std::isfinite(bound) && bound > 0.0;
}

[[nodiscard]] bool
canonical_edge_bound(const PatchMappingGridSpec &spec,
                     const PatchGridCanonicalInterpolationVertexKey &key,
                     double &bound) noexcept {
  GridVertexIndex first;
  GridVertexIndex second;
  if (!key.valid() ||
      !decode_grid_vertex(spec, key.first_vertex_linear_index(), first) ||
      !decode_grid_vertex(spec, key.second_vertex_linear_index(), second))
    return false;
  const auto delta = [](std::size_t left, std::size_t right) {
    return left < right ? right - left : left - right;
  };
  const std::size_t dx = delta(first.x, second.x);
  const std::size_t dy = delta(first.y, second.y);
  const std::size_t dz = delta(first.z, second.z);
  if (dx > 1U || dy > 1U || dz > 1U)
    return false;
  const unsigned l1_steps = static_cast<unsigned>(dx + dy + dz);
  return upward_lattice_distance(spec.grid_spacing, l1_steps, bound);
}

[[nodiscard]] bool canonical_occurrences_are_production_tetra_edges(
    const PreparedPatchGridCellLineage &cell_lineage,
    const PreparedPatchGridTriangleOccurrenceLineage &triangle_lineage,
    std::size_t canonical_position,
    const PatchGridCanonicalInterpolationVertexKey &key) noexcept {
  if (canonical_position + 1U >=
          triangle_lineage.canonical_vertex_occurrence_offsets.size() ||
      triangle_lineage
              .canonical_vertex_occurrence_offsets[canonical_position] >=
          triangle_lineage
              .canonical_vertex_occurrence_offsets[canonical_position + 1U] ||
      triangle_lineage
              .canonical_vertex_occurrence_offsets[canonical_position + 1U] >
          triangle_lineage.canonical_vertex_edge_occurrences.size())
    return false;
  for (std::size_t position =
           triangle_lineage
               .canonical_vertex_occurrence_offsets[canonical_position];
       position <
       triangle_lineage
           .canonical_vertex_occurrence_offsets[canonical_position + 1U];
       ++position) {
    const std::size_t edge_position =
        triangle_lineage.canonical_vertex_edge_occurrences[position];
    if (edge_position >= cell_lineage.edges.size() ||
        edge_position >=
            triangle_lineage.edge_occurrence_to_canonical_vertex.size() ||
        triangle_lineage.edge_occurrence_to_canonical_vertex[edge_position] !=
            canonical_position)
      return false;
    const PatchGridSignChangingEdgeLineage &edge =
        cell_lineage.edges[edge_position];
    if (edge.tetra_position >= cell_lineage.tetrahedra.size())
      return false;
    PatchGridCanonicalInterpolationVertexKey observed;
    if (!PatchGridCanonicalInterpolationVertexKey::try_make(
            edge.first_vertex_linear_index, edge.second_vertex_linear_index,
            observed) ||
        observed != key)
      return false;
    const auto &tetra = cell_lineage.tetrahedra[edge.tetra_position];
    bool first_present = false;
    bool second_present = false;
    std::uint8_t first_local = 0U;
    std::uint8_t second_local = 0U;
    std::uint8_t local = 0U;
    for (const std::size_t vertex : tetra.vertex_linear_indices) {
      if (vertex == key.first_vertex_linear_index()) {
        first_present = true;
        first_local = local;
      }
      if (vertex == key.second_vertex_linear_index()) {
        second_present = true;
        second_local = local;
      }
      ++local;
    }
    if (!first_present || !second_present)
      return false;
    if (second_local < first_local)
      std::swap(first_local, second_local);
    constexpr std::array<PatchGridLocalTetraEdge, 6> tetra_edges = {
        PatchGridLocalTetraEdge{0U, 1U}, PatchGridLocalTetraEdge{0U, 2U},
        PatchGridLocalTetraEdge{0U, 3U}, PatchGridLocalTetraEdge{1U, 2U},
        PatchGridLocalTetraEdge{1U, 3U}, PatchGridLocalTetraEdge{2U, 3U}};
    if (std::find(tetra_edges.begin(), tetra_edges.end(),
                  PatchGridLocalTetraEdge{first_local, second_local}) ==
        tetra_edges.end())
      return false;
  }
  return true;
}

[[nodiscard]] PatchBoundedCandidateProjectionStatus
classify_projection(const LocalProjectionHit &hit, PatchRefId expected,
                    double maximum_distance) noexcept {
  if (!hit.found)
    return PatchBoundedCandidateProjectionStatus::projector_miss;
  if (hit.kind != expected.kind() || hit.patch_index != expected.index())
    return PatchBoundedCandidateProjectionStatus::projector_ref_mismatch;
  if (!hit.bounded_closest_certified)
    return PatchBoundedCandidateProjectionStatus::bounded_location_uncertified;
  if (!hit.normal_certified)
    return PatchBoundedCandidateProjectionStatus::normal_uncertified;
  if (!finite(hit.point) || !finite(hit.normal) ||
      !std::isfinite(hit.distance) || hit.distance < 0.0)
    return PatchBoundedCandidateProjectionStatus::nonfinite_result;
  if (hit.distance > maximum_distance)
    return PatchBoundedCandidateProjectionStatus::distance_outside_edge_bound;
  return PatchBoundedCandidateProjectionStatus::certified;
}

[[nodiscard]] bool
projection_certified(PatchBoundedCandidateProjectionStatus status) noexcept {
  return status == PatchBoundedCandidateProjectionStatus::certified;
}

[[nodiscard]] bool find_projection(
    const PreparedPatchGridProjectionAudit &result,
    std::size_t canonical_vertex, PatchRefId ref,
    const PatchGridBoundedCandidateProjection *&projection) noexcept {
  projection = nullptr;
  if (canonical_vertex + 1U >= result.vertex_projection_offsets.size())
    return false;
  const std::size_t begin = result.vertex_projection_offsets[canonical_vertex];
  const std::size_t end =
      result.vertex_projection_offsets[canonical_vertex + 1U];
  if (begin > end || end > result.candidate_projections.size())
    return false;
  const auto found = std::lower_bound(
      result.candidate_projections.begin() + begin,
      result.candidate_projections.begin() + end, ref,
      [](const PatchGridBoundedCandidateProjection &candidate,
         PatchRefId requested) { return candidate.ref < requested; });
  if (found == result.candidate_projections.begin() + end || found->ref != ref)
    return true;
  projection = &*found;
  return true;
}

} // namespace

bool patch_orientation_rounding_contract_available() noexcept {
#if defined(__FAST_MATH__)
  return false;
#else
  if (!std::numeric_limits<double>::is_iec559 ||
      std::numeric_limits<double>::radix != 2 ||
      std::numeric_limits<double>::digits != 53 ||
      std::numeric_limits<double>::max_exponent != 1024 ||
      std::numeric_limits<double>::has_denorm != std::denorm_present ||
      std::fegetround() != FE_TONEAREST)
    return false;
  // A runtime probe catches x86 FTZ/DAZ (and equivalent modes) that are not
  // represented by the standard fenv rounding-mode query.
  volatile double denormal = std::numeric_limits<double>::denorm_min();
  volatile double one = 1.0;
  volatile double retained = denormal * one;
  return retained == denormal && retained != 0.0;
#endif
}

PatchOrientationSignInterval audit_patch_orientation_sign_interval(
    const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &normal) noexcept {
  PatchOrientationSignInterval result;
  result.rounding_contract_available =
      patch_orientation_rounding_contract_available();
  if (!result.rounding_contract_available) {
    result.status = PatchOrientationSignStatus::ambiguous_rounding_environment;
    return result;
  }
  if (!finite(a) || !finite(b) || !finite(c) || !finite(normal)) {
    result.status = PatchOrientationSignStatus::ambiguous_nonfinite;
    return result;
  }

  std::array<ClosedInterval, 3> ab{};
  std::array<ClosedInterval, 3> ac{};
  if (!interval_subtract(b.x, a.x, ab[0]) ||
      !interval_subtract(b.y, a.y, ab[1]) ||
      !interval_subtract(b.z, a.z, ab[2]) ||
      !interval_subtract(c.x, a.x, ac[0]) ||
      !interval_subtract(c.y, a.y, ac[1]) ||
      !interval_subtract(c.z, a.z, ac[2])) {
    result.status = PatchOrientationSignStatus::ambiguous_nonfinite;
    return result;
  }

  std::array<ClosedInterval, 3> cross{};
  ClosedInterval first;
  ClosedInterval second;
  if (!interval_multiply(ab[1], ac[2], first) ||
      !interval_multiply(ab[2], ac[1], second) ||
      !interval_subtract(first, second, cross[0]) ||
      !interval_multiply(ab[2], ac[0], first) ||
      !interval_multiply(ab[0], ac[2], second) ||
      !interval_subtract(first, second, cross[1]) ||
      !interval_multiply(ab[0], ac[1], first) ||
      !interval_multiply(ab[1], ac[0], second) ||
      !interval_subtract(first, second, cross[2])) {
    result.status = PatchOrientationSignStatus::ambiguous_nonfinite;
    return result;
  }

  std::array<ClosedInterval, 3> normal_intervals{};
  std::array<ClosedInterval, 3> terms{};
  if (!exact_interval(normal.x, normal_intervals[0]) ||
      !exact_interval(normal.y, normal_intervals[1]) ||
      !exact_interval(normal.z, normal_intervals[2]) ||
      !interval_multiply(cross[0], normal_intervals[0], terms[0]) ||
      !interval_multiply(cross[1], normal_intervals[1], terms[1]) ||
      !interval_multiply(cross[2], normal_intervals[2], terms[2]) ||
      !interval_add(terms[0], terms[1], first) ||
      !interval_add(first, terms[2], second)) {
    result.status = PatchOrientationSignStatus::ambiguous_nonfinite;
    return result;
  }

  result.lower = second.lower;
  result.upper = second.upper;
  result.interval_finite = true;
  if (result.lower > 0.0) {
    result.status = PatchOrientationSignStatus::positive;
    result.sign_certified = true;
  } else if (result.upper < 0.0) {
    result.status = PatchOrientationSignStatus::negative;
    result.sign_certified = true;
  } else {
    result.status = PatchOrientationSignStatus::ambiguous_zero_interval;
  }
  return result;
}

PreparedPatchGridProjectionAudit build_patch_grid_projection_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PreparedPatchGridCellLineage &cell_lineage,
    const PreparedPatchGridTriangleOccurrenceLineage &triangle_lineage) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  PreparedPatchGridProjectionAudit result;
  const auto elapsed_ms = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count();
  };
  const auto fail = [&](std::string_view status) {
    PreparedPatchGridProjectionAudit failed;
    failed.status.assign(status);
    failed.build_ms = elapsed_ms();
    return failed;
  };

  try {
    if (!crossing_payload_summaries_consistent(crossing_witnesses,
                                               grid.provenance))
      return fail("invalid_crossing_counters");
    if (!vertex_payload_counters_consistent(vertex_witnesses))
      return fail("invalid_vertex_counters");
    if (!cell_payload_summaries_consistent(cell_lineage))
      return fail("invalid_cell_counters");
    {
      const auto validation_started = Clock::now();
      const PreparedPatchGridTriangleOccurrenceLineage rebuilt =
          build_patch_grid_pre_orientation_triangle_lineage(
              complex, index, grid, spec, crossing_witnesses, vertex_witnesses,
              cell_lineage);
      result.validation_rebuild_ms = std::chrono::duration<double, std::milli>(
                                         Clock::now() - validation_started)
                                         .count();
      result.validation_rebuild_accounted_peak_bytes = std::max(
          rebuilt.retained_bytes, rebuilt.accounted_peak_temporary_bytes);
      if (!same_c2a_semantics(rebuilt, triangle_lineage))
        return fail("invalid_c2a_binding");
    }

    if (triangle_lineage.triangles.empty()) {
      const bool exact_empty_payload =
          triangle_lineage.canonical_vertices.empty() &&
          triangle_lineage.edge_occurrence_to_canonical_vertex.empty() &&
          triangle_lineage.canonical_vertex_occurrence_offsets.empty() &&
          triangle_lineage.canonical_vertex_edge_occurrences.empty() &&
          triangle_lineage.canonical_vertex_candidate_ref_offsets.empty() &&
          triangle_lineage.canonical_vertex_candidate_refs.empty() &&
          triangle_lineage.canonical_vertex_relation_offsets.empty() &&
          triangle_lineage.canonical_vertex_relations.empty() &&
          triangle_lineage.canonical_vertex_exact_location_ref_offsets
              .empty() &&
          triangle_lineage.canonical_vertex_exact_location_refs.empty() &&
          triangle_lineage.triangle_candidate_ref_offsets.empty() &&
          triangle_lineage.triangle_candidate_refs.empty();
      // Compatibility with the stable upstream empty contract: an exactly
      // empty Slice-B returns no_vertices, so the current C2a validator reports
      // invalid_vertex_payload before observing empty C1. This is accepted only
      // after both audit-local empty validators and semantic rebuild matched;
      // it grants no binding or geometry certificate.
      const bool exact_upstream_empty =
          vertex_witnesses.vertices.empty() && cell_lineage.cells.empty() &&
          triangle_lineage.status == "invalid_vertex_payload";
      if (!exact_empty_payload ||
          (triangle_lineage.status != "empty_input" && !exact_upstream_empty))
        return fail("invalid_empty_input");
      result.status = "empty_input";
      result.build_ms = elapsed_ms();
      return result;
    }
    if ((triangle_lineage.status != "pass" &&
         triangle_lineage.status !=
             "pass_with_candidate_attribution_uncertified") ||
        triangle_lineage.nonfinite_triangle_occurrences != 0U ||
        triangle_lineage.computed_zero_area_triangle_occurrences != 0U ||
        triangle_lineage.finite_triangle_occurrences !=
            triangle_lineage.triangles.size() ||
        triangle_lineage.planar_face_surface_location_certified ||
        triangle_lineage.final_mesh_orientation_certified)
      return fail("c2a_geometry_uncertified");

    result.complex_fingerprint = triangle_lineage.complex_fingerprint;
    result.lattice_identity = triangle_lineage.lattice_identity;
    result.input_binding_certified = true;
    result.vertices.reserve(triangle_lineage.canonical_vertices.size());
    std::size_t vertex_plus_one = triangle_lineage.canonical_vertices.size();
    if (!checked_add(vertex_plus_one, 1U))
      throw std::length_error("vertex_offsets");
    result.vertex_projection_offsets.reserve(vertex_plus_one);
    result.vertex_projection_offsets.push_back(0U);
    result.candidate_projections.reserve(
        triangle_lineage.canonical_vertex_candidate_refs.size());

    const auto vertex_projection_started = Clock::now();
    for (std::size_t vertex_position = 0U;
         vertex_position < triangle_lineage.canonical_vertices.size();
         ++vertex_position) {
      const auto &source = triangle_lineage.canonical_vertices[vertex_position];
      std::span<const PatchRefId> candidate_refs;
      std::span<const PatchRefId> exact_refs;
      if (vertex_position + 1U >=
              triangle_lineage.canonical_vertex_candidate_ref_offsets.size() ||
          vertex_position + 1U >=
              triangle_lineage.canonical_vertex_exact_location_ref_offsets
                  .size() ||
          !safe_subspan(
              triangle_lineage.canonical_vertex_candidate_refs,
              triangle_lineage
                  .canonical_vertex_candidate_ref_offsets[vertex_position],
              triangle_lineage
                  .canonical_vertex_candidate_ref_offsets[vertex_position + 1U],
              candidate_refs) ||
          !safe_subspan(
              triangle_lineage.canonical_vertex_exact_location_refs,
              triangle_lineage
                  .canonical_vertex_exact_location_ref_offsets[vertex_position],
              triangle_lineage
                  .canonical_vertex_exact_location_ref_offsets[vertex_position +
                                                               1U],
              exact_refs))
        throw std::length_error("vertex_refs");

      PatchGridCanonicalVertexProjectionAudit vertex;
      vertex.canonical_vertex_position = vertex_position;
      vertex.key = source.key;
      vertex.canonical_interpolation_point =
          source.canonical_interpolation_point;
      vertex.exact_location_crossing = source.exact_location_crossing;
      if (!finite(vertex.canonical_interpolation_point) ||
          !canonical_occurrences_are_production_tetra_edges(
              cell_lineage, triangle_lineage, vertex_position, vertex.key) ||
          !canonical_edge_bound(spec, vertex.key,
                                vertex.edge_distance_upper_bound))
        return fail("invalid_canonical_edge");
      vertex.exact_location_preserved =
          source.exact_analytic_location_certified &&
          source.exact_location_crossing.valid() && exact_refs.size() == 1U;
      if (vertex.exact_location_preserved) {
        vertex.exact_location_status =
            PatchExactLocationPreservationStatus::preserved_single_ref;
        vertex.exact_location_ref = exact_refs.front();
        if (!ref_in_range(complex, vertex.exact_location_ref))
          return fail("invalid_exact_ref");
        if (!checked_add(result.exact_locations_preserved, 1U))
          throw std::length_error("counter");
      } else if (source.exact_analytic_location_certified ||
                 !exact_refs.empty()) {
        vertex.exact_location_status = PatchExactLocationPreservationStatus::
            multiple_or_uncertified_exact_evidence;
      } else {
        vertex.exact_location_status =
            PatchExactLocationPreservationStatus::no_exact_evidence;
      }

      bool projections_complete = source.patch_candidate_collection_complete;
      for (const PatchRefId ref : candidate_refs) {
        PatchGridBoundedCandidateProjection projection;
        projection.ref = ref;
        if (!ref_in_range(complex, ref)) {
          projection.status =
              PatchBoundedCandidateProjectionStatus::invalid_patch_ref;
          projections_complete = false;
        } else {
          const std::size_t kind = static_cast<std::size_t>(ref.kind());
          if (kind >= result.vertex_projector_calls_by_kind.size() ||
              !checked_add(result.vertex_projector_calls, 1U) ||
              !checked_add(result.vertex_projector_calls_by_kind[kind], 1U))
            throw std::length_error("counter");
          const LocalProjectionHit hit = project_patch_bounded_closest(
              complex, index, ref.kind(), ref.index(),
              source.canonical_interpolation_point,
              vertex.edge_distance_upper_bound);
          projection.projected_point = hit.point;
          projection.projected_normal = hit.normal;
          projection.distance = hit.distance;
          projection.status =
              classify_projection(hit, ref, vertex.edge_distance_upper_bound);
          projection.bounded_projection_certified =
              projection_certified(projection.status);
          projection.normal_certified = projection.bounded_projection_certified;
          if (projection.bounded_projection_certified) {
            if (!checked_add(result.vertex_projector_successes, 1U) ||
                !checked_add(result.vertex_projector_successes_by_kind[kind],
                             1U))
              throw std::length_error("counter");
          } else {
            projections_complete = false;
            if (!checked_add(result.vertex_projector_ambiguous, 1U) ||
                !checked_add(result.vertex_projector_ambiguous_by_kind[kind],
                             1U))
              throw std::length_error("counter");
          }
        }
        result.candidate_projections.push_back(projection);
      }
      vertex.bounded_candidate_projection_complete =
          !candidate_refs.empty() && projections_complete;
      vertex.unique_candidate_classification_complete =
          source.patch_candidate_collection_complete;
      if (!source.patch_candidate_collection_complete) {
        vertex.unique_candidate_status = PatchUniqueCandidateProjectionStatus::
            candidate_collection_uncertified;
      } else if (candidate_refs.empty()) {
        vertex.unique_candidate_status =
            PatchUniqueCandidateProjectionStatus::no_candidate_refs;
      } else if (candidate_refs.size() > 1U) {
        vertex.unique_candidate_status =
            PatchUniqueCandidateProjectionStatus::multiple_candidate_refs;
      } else if (!projections_complete) {
        vertex.unique_candidate_status = PatchUniqueCandidateProjectionStatus::
            candidate_projection_uncertified;
      } else {
        vertex.unique_candidate_status =
            PatchUniqueCandidateProjectionStatus::certified_unique;
        vertex.unique_candidate_ref = candidate_refs.front();
        vertex.unique_candidate_projection_certified = true;
        if (!checked_add(result.unique_candidate_vertices, 1U))
          throw std::length_error("counter");
      }
      result.vertices.push_back(vertex);
      result.vertex_projection_offsets.push_back(
          result.candidate_projections.size());
    }
    result.vertex_projection_ms = std::chrono::duration<double, std::milli>(
                                      Clock::now() - vertex_projection_started)
                                      .count();
    result.vertex_projection_classification_complete =
        result.vertices.size() == triangle_lineage.canonical_vertices.size();

    result.triangles.reserve(triangle_lineage.triangles.size());
    double triangle_bound = 0.0;
    if (!upward_lattice_distance(spec.grid_spacing, 3U, triangle_bound))
      return fail("invalid_triangle_bound");
    const auto centroid_started = Clock::now();
    for (std::size_t triangle_position = 0U;
         triangle_position < triangle_lineage.triangles.size();
         ++triangle_position) {
      const auto &source = triangle_lineage.triangles[triangle_position];
      PatchGridTriangleProjectionAudit triangle;
      triangle.triangle_position = triangle_position;
      triangle.centroid_maximum_distance = triangle_bound;
      std::array<const PatchGridCanonicalVertexProjectionAudit *, 3> vertices{};
      for (std::size_t local = 0U; local < 3U; ++local) {
        const std::size_t position = source.canonical_vertex_positions[local];
        if (position >= result.vertices.size())
          throw std::length_error("triangle_vertex");
        vertices[local] = &result.vertices[position];
        if (vertices[local]->exact_location_preserved &&
            !checked_add(triangle.exact_location_vertex_count, 1U))
          throw std::length_error("counter");
      }
      const long double centroid_x =
          (static_cast<long double>(
               vertices[0]->canonical_interpolation_point.x) +
           static_cast<long double>(
               vertices[1]->canonical_interpolation_point.x) +
           static_cast<long double>(
               vertices[2]->canonical_interpolation_point.x)) /
          3.0L;
      const long double centroid_y =
          (static_cast<long double>(
               vertices[0]->canonical_interpolation_point.y) +
           static_cast<long double>(
               vertices[1]->canonical_interpolation_point.y) +
           static_cast<long double>(
               vertices[2]->canonical_interpolation_point.y)) /
          3.0L;
      const long double centroid_z =
          (static_cast<long double>(
               vertices[0]->canonical_interpolation_point.z) +
           static_cast<long double>(
               vertices[1]->canonical_interpolation_point.z) +
           static_cast<long double>(
               vertices[2]->canonical_interpolation_point.z)) /
          3.0L;
      triangle.canonical_centroid = {static_cast<double>(centroid_x),
                                     static_cast<double>(centroid_y),
                                     static_cast<double>(centroid_z)};
      if (!finite(triangle.canonical_centroid))
        return fail("nonfinite_centroid");

      std::span<const PatchRefId> common_refs;
      if (triangle_position + 1U >=
              triangle_lineage.triangle_candidate_ref_offsets.size() ||
          !safe_subspan(
              triangle_lineage.triangle_candidate_refs,
              triangle_lineage
                  .triangle_candidate_ref_offsets[triangle_position],
              triangle_lineage
                  .triangle_candidate_ref_offsets[triangle_position + 1U],
              common_refs))
        throw std::length_error("triangle_refs");
      if (!source.patch_candidate_summary_complete) {
        triangle.centroid_status =
            PatchCentroidProjectionStatus::vertex_projection_uncertified;
      } else if (common_refs.empty()) {
        triangle.centroid_status =
            PatchCentroidProjectionStatus::no_common_candidate;
      } else if (common_refs.size() > 1U) {
        triangle.centroid_status =
            PatchCentroidProjectionStatus::multiple_common_candidates;
      } else {
        triangle.common_candidate_ref = common_refs.front();
        triangle.same_candidate_vertex_projection_certified = true;
        for (std::size_t local = 0U; local < 3U; ++local) {
          if (!vertices[local]->unique_candidate_projection_certified ||
              vertices[local]->unique_candidate_ref !=
                  triangle.common_candidate_ref) {
            triangle.same_candidate_vertex_projection_certified = false;
            continue;
          }
          const PatchGridBoundedCandidateProjection *projection = nullptr;
          if (!find_projection(result, source.canonical_vertex_positions[local],
                               triangle.common_candidate_ref, projection))
            throw std::length_error("projection_csr");
          if (projection == nullptr ||
              !projection->bounded_projection_certified)
            triangle.same_candidate_vertex_projection_certified = false;
        }
        if (!triangle.same_candidate_vertex_projection_certified) {
          triangle.centroid_status =
              PatchCentroidProjectionStatus::vertex_projection_uncertified;
        } else {
          if (!checked_add(result.centroid_projector_calls, 1U))
            throw std::length_error("counter");
          const LocalProjectionHit hit = project_patch_bounded_closest(
              complex, index, triangle.common_candidate_ref.kind(),
              triangle.common_candidate_ref.index(),
              triangle.canonical_centroid, triangle_bound);
          triangle.centroid_projected_point = hit.point;
          triangle.centroid_projected_normal = hit.normal;
          triangle.centroid_projection_distance = hit.distance;
          const PatchBoundedCandidateProjectionStatus status =
              classify_projection(hit, triangle.common_candidate_ref,
                                  triangle_bound);
          if (status == PatchBoundedCandidateProjectionStatus::certified) {
            triangle.centroid_status = PatchCentroidProjectionStatus::
                certified_single_common_candidate;
            triangle.centroid_bounded_projection_certified = true;
            if (!checked_add(result.centroid_projector_successes, 1U))
              throw std::length_error("counter");
          } else if (status ==
                     PatchBoundedCandidateProjectionStatus::nonfinite_result) {
            triangle.centroid_status =
                PatchCentroidProjectionStatus::nonfinite_result;
          } else if (status == PatchBoundedCandidateProjectionStatus::
                                   distance_outside_edge_bound) {
            triangle.centroid_status =
                PatchCentroidProjectionStatus::distance_outside_triangle_bound;
          } else {
            triangle.centroid_status =
                PatchCentroidProjectionStatus::projector_uncertified;
          }
        }
      }
      result.triangles.push_back(triangle);
    }
    result.centroid_projection_ms = std::chrono::duration<double, std::milli>(
                                        Clock::now() - centroid_started)
                                        .count();

    const auto orientation_started = Clock::now();
    for (std::size_t triangle_position = 0U;
         triangle_position < result.triangles.size(); ++triangle_position) {
      auto &triangle = result.triangles[triangle_position];
      const auto &source = triangle_lineage.triangles[triangle_position];
      bool unique_candidate_identity =
          triangle.same_candidate_vertex_projection_certified;
      for (const std::size_t vertex_position :
           source.canonical_vertex_positions) {
        if (vertex_position >= result.vertices.size())
          throw std::length_error("triangle_vertex");
        const auto &vertex = result.vertices[vertex_position];
        unique_candidate_identity =
            unique_candidate_identity &&
            vertex.unique_candidate_projection_certified &&
            vertex.unique_candidate_ref == triangle.common_candidate_ref;
      }
      if (!unique_candidate_identity) {
        triangle.orientation_interval.status =
            PatchOrientationSignStatus::ambiguous_candidate_identity;
      } else if (!triangle.centroid_bounded_projection_certified) {
        triangle.orientation_interval.status =
            PatchOrientationSignStatus::ambiguous_normal_uncertified;
      } else {
        const Vec3 &a = result.vertices[source.canonical_vertex_positions[0]]
                            .canonical_interpolation_point;
        const Vec3 &b = result.vertices[source.canonical_vertex_positions[1]]
                            .canonical_interpolation_point;
        const Vec3 &c = result.vertices[source.canonical_vertex_positions[2]]
                            .canonical_interpolation_point;
        triangle.orientation_interval = audit_patch_orientation_sign_interval(
            a, b, c, triangle.centroid_projected_normal);
      }
      switch (triangle.orientation_interval.status) {
      case PatchOrientationSignStatus::positive:
        if (!checked_add(result.orientation_positive, 1U))
          throw std::length_error("counter");
        break;
      case PatchOrientationSignStatus::negative:
        triangle.suggested_swap_preorientation = true;
        if (!checked_add(result.orientation_negative, 1U))
          throw std::length_error("counter");
        break;
      default:
        if (!checked_add(result.orientation_ambiguous, 1U))
          throw std::length_error("counter");
        break;
      }
      triangle.planar_face_surface_location_certified = false;
      triangle.final_mesh_orientation_certified = false;
    }
    result.orientation_ms = std::chrono::duration<double, std::milli>(
                                Clock::now() - orientation_started)
                                .count();
    result.triangle_projection_classification_complete =
        result.triangles.size() == triangle_lineage.triangles.size();
    result.orientation_classification_complete =
        result.triangles.size() == triangle_lineage.triangles.size();
    result.planar_face_surface_location_certified = false;
    result.final_mesh_orientation_certified = false;

    if (!projection_payload_bytes(result, result.retained_bytes))
      throw std::length_error("bytes");
    result.accounted_peak_temporary_bytes = std::max(
        result.retained_bytes, result.validation_rebuild_accounted_peak_bytes);
    result.status = result.vertex_projector_ambiguous != 0U ||
                            result.orientation_ambiguous != 0U
                        ? "pass_with_ambiguous_projection"
                        : "pass";
    result.build_ms = elapsed_ms();
    return result;
  } catch (const std::bad_alloc &) {
    return fail("alloc_fail");
  } catch (const std::length_error &) {
    return fail("length_err");
  }
}

} // namespace TitanPBE::Tabi::analytic_ses

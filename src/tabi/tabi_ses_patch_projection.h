#ifndef TITAN_PBE_TABI_SES_PATCH_PROJECTION_H
#define TITAN_PBE_TABI_SES_PATCH_PROJECTION_H

#include "tabi_ses_patch_mapping.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {

// C2b is an audit-only consumer of the immutable Slice A/B/C1/C2a payloads.
// It does not alter marching-tetrahedra connectivity, vertex coordinates, or
// production orientation.

enum class PatchBoundedCandidateProjectionStatus : std::uint8_t {
  certified,
  invalid_patch_ref,
  projector_ref_mismatch,
  projector_miss,
  bounded_location_uncertified,
  normal_uncertified,
  nonfinite_result,
  distance_outside_edge_bound
};

enum class PatchUniqueCandidateProjectionStatus : std::uint8_t {
  certified_unique,
  no_candidate_refs,
  candidate_collection_uncertified,
  candidate_projection_uncertified,
  multiple_candidate_refs
};

enum class PatchExactLocationPreservationStatus : std::uint8_t {
  preserved_single_ref,
  no_exact_evidence,
  multiple_or_uncertified_exact_evidence
};

enum class PatchCentroidProjectionStatus : std::uint8_t {
  certified_single_common_candidate,
  no_common_candidate,
  multiple_common_candidates,
  vertex_projection_uncertified,
  projector_uncertified,
  nonfinite_result,
  distance_outside_triangle_bound
};

enum class PatchOrientationSignStatus : std::uint8_t {
  positive,
  negative,
  ambiguous_zero_interval,
  ambiguous_rounding_environment,
  ambiguous_nonfinite,
  ambiguous_normal_uncertified,
  ambiguous_candidate_identity
};

struct PatchOrientationSignInterval {
  double lower{0.0};
  double upper{0.0};
  PatchOrientationSignStatus status{
      PatchOrientationSignStatus::ambiguous_rounding_environment};
  bool rounding_contract_available{false};
  bool interval_finite{false};
  bool sign_certified{false};
};

// The interval proof is enabled only for IEC 60559 binary64 arithmetic under
// FE_TONEAREST and when the translation unit is not compiled with fast-math.
// It certifies only the sign of cross(b-a,c-a) dot normal for the supplied
// normal; it is not a surface-orientation or final-mesh certificate.
[[nodiscard]] bool patch_orientation_rounding_contract_available() noexcept;
[[nodiscard]] PatchOrientationSignInterval
audit_patch_orientation_sign_interval(const Vec3 &a, const Vec3 &b,
                                      const Vec3 &c,
                                      const Vec3 &normal) noexcept;

struct PatchGridBoundedCandidateProjection {
  PatchRefId ref;
  Vec3 projected_point;
  Vec3 projected_normal;
  double distance{0.0};
  PatchBoundedCandidateProjectionStatus status{
      PatchBoundedCandidateProjectionStatus::projector_miss};
  // These are full-record conclusions, not raw projector observations. They
  // become true only when status == certified.
  bool bounded_projection_certified{false};
  bool normal_certified{false};
};

struct PatchGridCanonicalVertexProjectionAudit {
  std::size_t canonical_vertex_position{0U};
  PatchGridCanonicalInterpolationVertexKey key;
  Vec3 canonical_interpolation_point;
  // Conservative L1 enclosure derived only from integer lattice deltas and
  // spacing. It intentionally avoids subtracting large world coordinates.
  double edge_distance_upper_bound{0.0};
  PatchCrossingAddress exact_location_crossing;
  PatchRefId exact_location_ref;
  PatchRefId unique_candidate_ref;
  PatchUniqueCandidateProjectionStatus unique_candidate_status{
      PatchUniqueCandidateProjectionStatus::no_candidate_refs};
  PatchExactLocationPreservationStatus exact_location_status{
      PatchExactLocationPreservationStatus::no_exact_evidence};
  // This is copied only from a validated C2a exact-location certificate.
  // Candidate-only evidence can never set it.
  bool exact_location_preserved{false};
  bool bounded_candidate_projection_complete{false};
  bool unique_candidate_classification_complete{false};
  bool unique_candidate_projection_certified{false};
};

struct PatchGridTriangleProjectionAudit {
  std::size_t triangle_position{0U};
  PatchRefId common_candidate_ref;
  Vec3 canonical_centroid;
  Vec3 centroid_projected_point;
  Vec3 centroid_projected_normal;
  double centroid_projection_distance{0.0};
  double centroid_maximum_distance{0.0};
  PatchCentroidProjectionStatus centroid_status{
      PatchCentroidProjectionStatus::no_common_candidate};
  PatchOrientationSignInterval orientation_interval;
  std::size_t exact_location_vertex_count{0U};
  bool same_candidate_vertex_projection_certified{false};
  bool centroid_bounded_projection_certified{false};
  bool suggested_swap_preorientation{false};
  // These two claims are intentionally unavailable in C2b.
  bool planar_face_surface_location_certified{false};
  bool final_mesh_orientation_certified{false};
};

struct PreparedPatchGridProjectionAudit {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::vector<PatchGridCanonicalVertexProjectionAudit> vertices;
  // Candidate projections are stored once per canonical vertex/ref.
  std::vector<std::size_t> vertex_projection_offsets;
  std::vector<PatchGridBoundedCandidateProjection> candidate_projections;
  std::vector<PatchGridTriangleProjectionAudit> triangles;
  std::array<std::size_t, 3> vertex_projector_calls_by_kind{};
  std::array<std::size_t, 3> vertex_projector_successes_by_kind{};
  std::array<std::size_t, 3> vertex_projector_ambiguous_by_kind{};
  std::size_t vertex_projector_calls{0U};
  std::size_t vertex_projector_successes{0U};
  std::size_t vertex_projector_ambiguous{0U};
  std::size_t exact_locations_preserved{0U};
  std::size_t unique_candidate_vertices{0U};
  std::size_t centroid_projector_calls{0U};
  std::size_t centroid_projector_successes{0U};
  std::size_t orientation_positive{0U};
  std::size_t orientation_negative{0U};
  std::size_t orientation_ambiguous{0U};
  // Semantic C2a rebuild used solely for fail-closed input validation. It
  // performs no bounded projection and builds no adjacency oracle.
  double validation_rebuild_ms{0.0};
  std::size_t validation_rebuild_accounted_peak_bytes{0U};
  // Exact retained vector payload capacity; fixed/status metadata and
  // allocator bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted vector payload observed at phase boundaries. This excludes
  // allocator metadata and old/new reallocation overlap; it is not RSS.
  std::size_t accounted_peak_temporary_bytes{0U};
  double vertex_projection_ms{0.0};
  double centroid_projection_ms{0.0};
  double orientation_ms{0.0};
  double build_ms{0.0};
  bool input_binding_certified{false};
  bool vertex_projection_classification_complete{false};
  bool triangle_projection_classification_complete{false};
  // Classification can be complete while every sign is conservatively
  // ambiguous; it is not an orientation certificate.
  bool orientation_classification_complete{false};
  bool planar_face_surface_location_certified{false};
  bool final_mesh_orientation_certified{false};
  std::string status{"not_built"};
};

[[nodiscard]] PreparedPatchGridProjectionAudit
build_patch_grid_projection_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PreparedPatchGridCellLineage &cell_lineage,
    const PreparedPatchGridTriangleOccurrenceLineage &triangle_lineage);

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_PROJECTION_H

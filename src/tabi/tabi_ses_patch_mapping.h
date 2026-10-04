#ifndef TITAN_PBE_TABI_SES_PATCH_MAPPING_H
#define TITAN_PBE_TABI_SES_PATCH_MAPPING_H

#include "tabi_ses_patch_intersections.h"
#include "tabi_ses_patch_ray.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {

struct PatchMappingGridSpec {
  Vec3 lower;
  double grid_spacing{0.0};
  std::size_t nx{0U};
  std::size_t ny{0U};
  std::size_t nz{0U};
};

// Stable address into one axis CSR. The top two route bits store RayAxis and
// the low 62 bits store the row. The crossing is the axis-global CSR index.
// Every size_t conversion is checked before an address can become valid.
class PatchCrossingAddress {
public:
  static constexpr unsigned axis_shift = 62U;
  static constexpr std::uint64_t row_mask =
      (std::uint64_t{1} << axis_shift) - 1U;
  static constexpr std::uint64_t invalid_route =
      std::numeric_limits<std::uint64_t>::max();

  constexpr PatchCrossingAddress() noexcept = default;

  [[nodiscard]] static bool try_make(RayAxis axis, std::size_t row,
                                     std::size_t crossing,
                                     PatchCrossingAddress &result) noexcept;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return route_ != invalid_route &&
           (route_ >> axis_shift) <= static_cast<std::uint64_t>(RayAxis::z);
  }
  [[nodiscard]] constexpr RayAxis axis() const noexcept {
    return valid() ? static_cast<RayAxis>(route_ >> axis_shift) : RayAxis::x;
  }
  [[nodiscard]] constexpr std::size_t row() const noexcept {
    return static_cast<std::size_t>(route_ & row_mask);
  }
  [[nodiscard]] constexpr std::size_t crossing() const noexcept {
    return static_cast<std::size_t>(crossing_);
  }
  [[nodiscard]] constexpr std::uint64_t packed_route() const noexcept {
    return route_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(PatchCrossingAddress, PatchCrossingAddress) noexcept = default;

  [[nodiscard]] friend constexpr bool
  operator<(PatchCrossingAddress left, PatchCrossingAddress right) noexcept {
    if (left.route_ != right.route_)
      return left.route_ < right.route_;
    return left.crossing_ < right.crossing_;
  }

private:
  std::uint64_t route_{invalid_route};
  std::uint32_t crossing_{0U};
};

static_assert(sizeof(PatchCrossingAddress) <= 16U);

struct DecodedPatchRayRow {
  RayAxis axis{RayAxis::x};
  std::size_t row{0U};
  std::size_t u_index{0U};
  std::size_t v_index{0U};
  // Coordinates passed as (u,v) to intersect_patch_axis.
  double u{0.0};
  double v{0.0};
};

[[nodiscard]] bool validate_patch_mapping_grid_spec(
    const PatchMappingGridSpec &spec,
    std::array<std::size_t, 3> *row_counts = nullptr,
    std::size_t *vertex_count = nullptr) noexcept;

[[nodiscard]] bool decode_patch_ray_row(const PatchMappingGridSpec &spec,
                                        RayAxis axis, std::size_t row,
                                        DecodedPatchRayRow &decoded) noexcept;

[[nodiscard]] bool
reconstruct_patch_crossing_point(const PatchMappingGridSpec &spec,
                                 const PatchCrossingAddress &address,
                                 double coordinate, Vec3 &point) noexcept;

enum class CrossingIdentityCertificateStatus : std::uint8_t {
  valid_single_ref,
  valid_multi_ref,
  invalid
};

enum class CrossingRootReproductionStatus : std::uint8_t {
  reproduced,
  intersection_uncertified,
  root_not_reproduced,
  root_multiplicity_mismatch,
  nonfinite_residual
};

enum class CrossingBoundedLocationStatus : std::uint8_t {
  certified,
  root_unreproduced,
  projector_uncertified,
  patch_residual_failed,
  nonfinite_residual
};

enum class CrossingTransversalityStatus : std::uint8_t {
  primitive_filtered_transversal_unoriented,
  repeated_root_ambiguous,
  multi_ref_relation_uncertified,
  primitive_simplicity_uncertified,
  location_uncertified
};

struct CrossingMultiplicityClassification {
  CrossingRootReproductionStatus root_status{
      CrossingRootReproductionStatus::root_multiplicity_mismatch};
  CrossingTransversalityStatus transversality_if_location_certified{
      CrossingTransversalityStatus::location_uncertified};
};

// Pure Slice-A classification used by the builder and synthetic gate. Patch
// identity is intentionally independent: a valid ref remains identified even
// when its retained multiplicity does not reproduce or is unusable for parity.
[[nodiscard]] CrossingMultiplicityClassification
classify_patch_crossing_multiplicity(std::uint32_t stored_multiplicity,
                                     std::size_t observed_matches) noexcept;

[[nodiscard]] CrossingTransversalityStatus
classify_patch_crossing_group_transversality(
    std::size_t ref_count, std::uint32_t maximum_ref_multiplicity,
    bool bounded_location_certified,
    bool primitive_simplicity_certified) noexcept;

struct CertifiedPatchCrossingWitness {
  PatchCrossingAddress address;
  Vec3 point;
  double coordinate{0.0};
  double root_match_tolerance{0.0};
  double bounded_patch_tolerance{0.0};
  double maximum_root_residual{0.0};
  double maximum_patch_distance{0.0};
  std::uint32_t ref_count{0U};
  // Exact multiplicity retained by provenance for this coordinate group.
  // These fields distinguish, for example, two and three merged roots even
  // though both remain unusable as parity-changing crossings in Slice A.
  std::uint32_t total_root_multiplicity{0U};
  std::uint32_t maximum_ref_multiplicity{0U};
  CrossingIdentityCertificateStatus identity_status{
      CrossingIdentityCertificateStatus::invalid};
  CrossingRootReproductionStatus root_status{
      CrossingRootReproductionStatus::root_not_reproduced};
  CrossingBoundedLocationStatus location_status{
      CrossingBoundedLocationStatus::root_unreproduced};
  CrossingTransversalityStatus transversality_status{
      CrossingTransversalityStatus::location_uncertified};
  // intersect_patch_axis filters primitive roots with |n.axis| <= 1e-10.
  // Slice A retains that primitive-filtered contract but not the signed
  // derivative/normal component needed to certify orientation sign.
  bool orientation_sign_certified{false};
  bool primitive_simplicity_certified{false};

  [[nodiscard]] bool identity_certified() const noexcept {
    return identity_status ==
               CrossingIdentityCertificateStatus::valid_single_ref ||
           identity_status ==
               CrossingIdentityCertificateStatus::valid_multi_ref;
  }
  [[nodiscard]] bool root_reproduced() const noexcept {
    return root_status == CrossingRootReproductionStatus::reproduced;
  }
  [[nodiscard]] bool location_certified() const noexcept {
    return location_status == CrossingBoundedLocationStatus::certified;
  }
  [[nodiscard]] bool transversality_certified() const noexcept {
    return transversality_status ==
           CrossingTransversalityStatus::
               primitive_filtered_transversal_unoriented;
  }
};

struct PreparedPatchCrossingWitnesses {
  // Axis-major offsets into witnesses; offsets[3] is the total count.
  std::array<std::size_t, 4> axis_offsets{};
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::vector<CertifiedPatchCrossingWitness> witnesses;
  std::size_t refs_audited{0U};
  std::size_t root_events_audited{0U};
  std::size_t identity_certified_crossings{0U};
  std::size_t root_reproduced_crossings{0U};
  std::size_t bounded_patch_location_certified_crossings{0U};
  std::size_t transversality_certified_crossings{0U};
  std::size_t multi_ref_crossings{0U};
  std::size_t multi_ref_relation_uncertified_crossings{0U};
  std::size_t primitive_simplicity_uncertified_crossings{0U};
  std::size_t repeated_root_ambiguities{0U};
  std::size_t intersection_uncertified{0U};
  std::size_t roots_not_reproduced{0U};
  std::size_t root_multiplicity_mismatches{0U};
  std::size_t projector_uncertified{0U};
  std::size_t patch_residual_failures{0U};
  std::size_t nonfinite_bounded_residuals{0U};
  // Exact owned vector payload capacity; fixed fields, allocator metadata and
  // possible status-string storage are excluded.
  std::size_t retained_bytes{0U};
  // Accounted vector payload observed at builder phase boundaries. Allocator
  // metadata and old/new allocation overlap during vector growth are excluded;
  // this is not an exact process-memory peak.
  std::size_t peak_temporary_bytes{0U};
  double build_ms{0.0};
  bool identity_certificate_complete{false};
  bool root_reproduction_complete{false};
  bool location_certificate_complete{false};
  bool transversality_certificate_complete{false};
  std::string status{"not_built"};
};

// This is not a second seam tolerance. It only checks whether the same
// PatchRef on the same decoded row reproduces its retained root. The maximum
// covers the builder's merge band and coordinate-scale floating-point forward
// error; multiple roots inside it remain ambiguous.
[[nodiscard]] double patch_crossing_root_match_tolerance(
    double coordinate_merge_tolerance, const DecodedPatchRayRow &row,
    double coordinate, double grid_spacing) noexcept;

[[nodiscard]] bool
patch_crossing_residual_within_tolerance(double residual,
                                         double tolerance) noexcept;

// Audit-only preparation. It neither changes ray coordinates/parity nor
// polygonization. Each simple provenance ref is independently re-intersected
// on its exact decoded ray before the stored coordinate becomes certified.
[[nodiscard]] PreparedPatchCrossingWitnesses
build_certified_patch_crossing_witnesses(const PatchComplex &complex,
                                         const LocalPatchProjectionIndex &index,
                                         const PatchRayGrid &grid,
                                         const PatchMappingGridSpec &spec);

enum class PatchGridTopologyState : std::uint8_t {
  unresolved,
  outside,
  inside
};

enum class PatchGridTopologySource : std::uint8_t {
  unknown,
  inside_atom_union,
  outside_probe_expanded_union,
  direct_unanimous_parity,
  analytic_local_repair,
  morphology_fallback,
  cavity_fill
};

enum class PatchGridTopologyEvidenceStatus : std::uint8_t {
  unknown,
  certified_inside_atom_union,
  certified_outside_probe_expanded_union,
  direct_unanimous_parity_not_yet_audited,
  analytic_perturbed_point_parity,
  morphology_fallback_nonanalytic,
  cavity_fill_nonanalytic
};

struct PatchGridTopologyEvidenceClassification {
  PatchGridTopologyEvidenceStatus status{
      PatchGridTopologyEvidenceStatus::unknown};
  bool analytic_evidence{false};
  // A perturbed-point parity result is analytic evidence, but it is not an
  // exact certificate for the unperturbed grid vertex.
  bool exact_vertex_certified{false};
};

[[nodiscard]] PatchGridTopologyEvidenceClassification
classify_patch_grid_topology_evidence(PatchGridTopologyState state,
                                      PatchGridTopologySource source,
                                      bool cavity_fill) noexcept;

struct PatchGridVertexMagnitudeRequest {
  std::size_t ix{0U};
  std::size_t iy{0U};
  std::size_t iz{0U};
  float sparse_magnitude{0.0F};
  PatchGridTopologyState topology_state{PatchGridTopologyState::unresolved};
  PatchGridTopologySource topology_source{PatchGridTopologySource::unknown};
  bool sparse_magnitude_present{false};
  bool cavity_fill{false};
};

enum class PatchGridMagnitudeSourceKind : std::uint8_t {
  grid_spacing_fallback,
  sparse_sample,
  analytic_crossing,
  clamp_floor
};

enum class PatchGridMagnitudeEvidenceStatus : std::uint8_t {
  certified_analytic_crossing,
  grid_spacing_fallback_nonanalytic,
  sparse_sample_nonanalytic,
  clamp_floor_nonanalytic,
  missing_crossing_witness,
  identity_uncertified,
  root_uncertified,
  bounded_location_uncertified,
  repeated_root_ambiguous,
  multi_ref_relation_uncertified,
  primitive_simplicity_uncertified,
  transversality_uncertified
};

struct PatchGridMagnitudeSourceCandidate {
  PatchGridMagnitudeSourceKind kind{
      PatchGridMagnitudeSourceKind::grid_spacing_fallback};
  RayAxis axis{RayAxis::x};
  PatchCrossingAddress crossing;
  double distance{0.0};
  std::uint32_t refs_begin{0U};
  std::uint32_t refs_end{0U};
  PatchGridMagnitudeEvidenceStatus evidence_status{
      PatchGridMagnitudeEvidenceStatus::grid_spacing_fallback_nonanalytic};
  bool raw_exact_owner{false};
  bool raw_tolerance_tied{false};
  bool final_scalar_exact_owner{false};

  [[nodiscard]] bool source_evidence_certified() const noexcept {
    return evidence_status ==
           PatchGridMagnitudeEvidenceStatus::certified_analytic_crossing;
  }
};

struct PatchGridVertexMagnitudeWitness {
  std::size_t linear_index{0U};
  std::size_t ix{0U};
  std::size_t iy{0U};
  std::size_t iz{0U};
  double raw_magnitude{0.0};
  double clamped_magnitude{0.0};
  double double_arithmetic_tie_tolerance{0.0};
  double sparse_float_tie_tolerance{0.0};
  double source_tie_tolerance{0.0};
  PatchGridTopologyState topology_state{PatchGridTopologyState::unresolved};
  PatchGridTopologySource topology_source{PatchGridTopologySource::unknown};
  // Exact production sparse payload. It is copied as float and converted to
  // double only by the scalar rule; absent vertices store canonical +0.0F.
  float sparse_magnitude{0.0F};
  bool sparse_magnitude_present{false};
  bool sparse_suppressed_yz{false};
  bool cavity_fill{false};
  bool floor_clamp_applied{false};
  PatchGridTopologyEvidenceStatus topology_evidence_status{
      PatchGridTopologyEvidenceStatus::unknown};
  bool topology_analytic_evidence{false};
  bool topology_exact_vertex_certified{false};
  bool raw_source_certificate_complete{false};
  bool final_scalar_source_certificate_complete{false};
  // A grid endpoint is not a point on the SES. Slice B certifies magnitude
  // ownership only; surface location remains false by construction.
  bool surface_location_certified{false};
};

struct PreparedPatchGridVertexMagnitudeWitnesses {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::vector<PatchGridVertexMagnitudeWitness> vertices;
  // CSR offsets into candidates, one entry per sorted vertex plus sentinel.
  std::vector<std::size_t> candidate_offsets;
  std::vector<PatchGridMagnitudeSourceCandidate> candidates;
  // Candidate ref groups are copied contiguously and addressed by refs_begin/
  // refs_end. Sparse/fallback/clamp-floor candidates have an empty range.
  std::vector<PatchRefId> candidate_refs;
  std::size_t analytic_candidates{0U};
  std::size_t certified_analytic_candidates{0U};
  std::size_t uncertified_analytic_candidates{0U};
  std::size_t sparse_candidates{0U};
  std::size_t fallback_candidates{0U};
  std::size_t clamp_floor_candidates{0U};
  std::size_t raw_exact_owner_candidates{0U};
  std::size_t tolerance_tied_candidates{0U};
  std::size_t final_scalar_exact_owner_candidates{0U};
  std::size_t raw_source_certified_vertices{0U};
  std::size_t final_scalar_source_certified_vertices{0U};
  std::size_t floor_clamped_vertices{0U};
  std::size_t sparse_suppressed_yz_vertices{0U};
  std::size_t topology_analytic_evidence_vertices{0U};
  std::size_t topology_exact_vertex_certified_vertices{0U};
  std::size_t topology_nonanalytic_vertices{0U};
  std::size_t retained_bytes{0U};
  // Accounted vector payload observed at builder phase boundaries. Allocator
  // metadata and old/new allocation overlap during vector growth are excluded;
  // this is not an exact process-memory peak.
  std::size_t peak_temporary_bytes{0U};
  double build_ms{0.0};
  bool raw_source_certificate_complete{false};
  bool final_scalar_source_certificate_complete{false};
  bool topology_source_classification_complete{false};
  bool topology_exact_vertex_certificate_complete{false};
  bool surface_location_certified{false};
  std::string status{"not_built"};
};

[[nodiscard]] bool try_patch_grid_vertex_linear_index(
    const PatchMappingGridSpec &spec, std::size_t ix, std::size_t iy,
    std::size_t iz, std::size_t &linear_index) noexcept;

// Audit-only reconstruction of the exact production magnitude rule. The
// scalar is never changed by tie handling; ties only retain extra source
// evidence. Output order is canonical by linear vertex then source kind/axis/
// crossing address.
[[nodiscard]] PreparedPatchGridVertexMagnitudeWitnesses
build_patch_grid_vertex_magnitude_witnesses(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    std::span<const PatchGridVertexMagnitudeRequest> requests);

// Convenience for focused tests and one-off diagnostics. It intentionally
// performs the same O(all retained witnesses) binding validation as the batch;
// future cell/relation hot paths must prepare one batch, never call this per
// grid vertex.
[[nodiscard]] PreparedPatchGridVertexMagnitudeWitnesses
select_patch_grid_vertex_magnitude_sources(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PatchGridVertexMagnitudeRequest &request);

inline constexpr std::array<std::array<std::uint8_t, 3>, 8>
    patch_grid_production_corners = {std::array<std::uint8_t, 3>{0U, 0U, 0U},
                                     std::array<std::uint8_t, 3>{1U, 0U, 0U},
                                     std::array<std::uint8_t, 3>{1U, 1U, 0U},
                                     std::array<std::uint8_t, 3>{0U, 1U, 0U},
                                     std::array<std::uint8_t, 3>{0U, 0U, 1U},
                                     std::array<std::uint8_t, 3>{1U, 0U, 1U},
                                     std::array<std::uint8_t, 3>{1U, 1U, 1U},
                                     std::array<std::uint8_t, 3>{0U, 1U, 1U}};

inline constexpr std::array<std::array<std::uint8_t, 4>, 6>
    patch_grid_production_tetrahedra = {
        std::array<std::uint8_t, 4>{0U, 5U, 1U, 6U},
        std::array<std::uint8_t, 4>{0U, 1U, 2U, 6U},
        std::array<std::uint8_t, 4>{0U, 2U, 3U, 6U},
        std::array<std::uint8_t, 4>{0U, 3U, 7U, 6U},
        std::array<std::uint8_t, 4>{0U, 7U, 4U, 6U},
        std::array<std::uint8_t, 4>{0U, 4U, 5U, 6U}};

class PatchGridCellAddress {
public:
  static constexpr std::uint64_t invalid_value =
      std::numeric_limits<std::uint64_t>::max();

  constexpr PatchGridCellAddress() noexcept = default;

  [[nodiscard]] static bool try_make(const PatchMappingGridSpec &spec,
                                     std::size_t ix, std::size_t iy,
                                     std::size_t iz,
                                     PatchGridCellAddress &result) noexcept;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return linear_ != invalid_value;
  }
  [[nodiscard]] constexpr std::uint64_t linear() const noexcept {
    return linear_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(PatchGridCellAddress, PatchGridCellAddress) noexcept = default;
  [[nodiscard]] friend constexpr bool
  operator<(PatchGridCellAddress left, PatchGridCellAddress right) noexcept {
    return left.linear_ < right.linear_;
  }

private:
  std::uint64_t linear_{invalid_value};
};

static_assert(sizeof(PatchGridCellAddress) == sizeof(std::uint64_t));

struct PatchGridActiveCellRequest {
  std::size_t ix{0U};
  std::size_t iy{0U};
  std::size_t iz{0U};
  // Exact production signs used by marching tetrahedra. Slice C1 never
  // recomputes or overwrites this mask from topology evidence metadata.
  std::uint8_t inside_mask{0U};
};

struct PatchGridLocalTetraEdge {
  std::uint8_t first{0U};
  std::uint8_t second{0U};

  [[nodiscard]] friend constexpr bool
  operator==(PatchGridLocalTetraEdge,
             PatchGridLocalTetraEdge) noexcept = default;
};

[[nodiscard]] std::size_t enumerate_patch_grid_tetra_sign_changing_edges(
    std::uint8_t tetra_inside_mask,
    std::array<PatchGridLocalTetraEdge, 6> &edges) noexcept;
// A mask with any high four bits set returns size_t::max() as an explicit
// invalid sentinel; valid low-four-bit masks return 0..4 edges.

enum class PatchGridEdgeScalarLineageStatus : std::uint8_t {
  certified,
  no_analytic_source,
  sparse_sample,
  grid_spacing_fallback,
  clamp_floor,
  tolerance_tied_source,
  missing_crossing_witness,
  identity_uncertified,
  root_uncertified,
  bounded_location_uncertified,
  repeated_root_ambiguous,
  multi_ref_relation_uncertified,
  primitive_simplicity_uncertified,
  transversality_uncertified
};

enum class PatchGridEdgeSurfaceLocationStatus : std::uint8_t {
  certified_axis_crossing,
  internal_diagonal_location_uncertified,
  endpoint_scalar_lineage_uncertified,
  no_common_axis_crossing,
  multiple_common_axis_crossings,
  crossing_axis_row_mismatch,
  crossing_outside_edge,
  interpolation_residual_failed
};

enum class PatchGridEdgeTopologyStatus : std::uint8_t {
  exact,
  topology_evidence_missing,
  direct_parity_not_yet_audited,
  analytic_local_repair_perturbed,
  morphology_fallback_nonanalytic,
  cavity_fill_nonanalytic
};

struct PatchGridCanonicalCellLineage {
  PatchGridCellAddress address;
  std::size_t ix{0U};
  std::size_t iy{0U};
  std::size_t iz{0U};
  std::uint8_t inside_mask{0U};
  // Canonical grid-vertex linear identities in the exact order of
  // patch_grid_production_corners. These are not positions in a witness
  // vector; any temporary witness lookup remains builder-local.
  std::array<std::size_t, 8> corner_vertex_linear_indices{};
  bool topology_sign_evidence_complete{false};
};

struct PatchGridCanonicalTetraLineage {
  std::size_t cell_position{0U};
  std::uint8_t tetra_index{0U};
  std::uint8_t inside_mask{0U};
  std::array<std::uint8_t, 4> cell_corners{};
  std::array<std::size_t, 4> vertex_linear_indices{};
  bool topology_sign_evidence_complete{false};
};

struct PatchGridSignChangingEdgeLineage {
  std::size_t tetra_position{0U};
  // Ordinal within the filtered sign-changing list produced from the fixed
  // six local-pair table, not the pair's index in that unfiltered table.
  std::uint8_t edge_ordinal{0U};
  std::uint8_t first_cell_corner{0U};
  std::uint8_t second_cell_corner{0U};
  std::size_t first_vertex_linear_index{0U};
  std::size_t second_vertex_linear_index{0U};
  RayAxis cartesian_axis{RayAxis::x};
  // Endpoint-specific evidence is authoritative. The aggregate statuses
  // below are deterministic summaries only; the builder must document its
  // precedence and must never discard either endpoint reason.
  std::array<PatchGridEdgeScalarLineageStatus, 2> endpoint_scalar_status{
      PatchGridEdgeScalarLineageStatus::no_analytic_source,
      PatchGridEdgeScalarLineageStatus::no_analytic_source};
  std::array<PatchGridTopologyEvidenceStatus, 2> endpoint_topology_status{
      PatchGridTopologyEvidenceStatus::unknown,
      PatchGridTopologyEvidenceStatus::unknown};
  PatchGridEdgeScalarLineageStatus scalar_status{
      PatchGridEdgeScalarLineageStatus::no_analytic_source};
  PatchGridEdgeSurfaceLocationStatus location_status{
      PatchGridEdgeSurfaceLocationStatus::
          internal_diagonal_location_uncertified};
  PatchGridEdgeTopologyStatus topology_status{
      PatchGridEdgeTopologyStatus::topology_evidence_missing};
  PatchCrossingAddress location_crossing;
  Vec3 interpolated_point;
  double interpolation_parameter{0.0};
  double interpolation_residual{0.0};
  double interpolation_tolerance{0.0};
  // Production marching-tetra geometry exists independently of whether its
  // scalar source or an analytic patch location can be certified.
  bool production_interpolation_complete{false};
  // Distinguishes a concrete analytic crossing residual from default zeros on
  // diagonals, missing crossings, or other fail-closed location paths.
  bool analytic_location_residual_available{false};
  bool cartesian_axis_edge{false};
  bool scalar_lineage_complete{false};
  bool patch_candidate_collection_complete{false};
  bool patch_relation_classification_complete{false};
  bool surface_location_certified{false};
  bool topology_sign_evidence_complete{false};
};

struct PatchGridEdgePatchRelationEvidence {
  PatchRefId first;
  PatchRefId second;
  PatchRelation relation{PatchRelation::uncertified};
};

enum class PatchGridRelationOracleBindingStatus : std::uint8_t {
  unavailable,
  built_once_from_bound_batch
};

struct PreparedPatchGridCellLineage {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  std::vector<PatchGridCanonicalCellLineage> cells;
  std::vector<std::size_t> cell_tetra_offsets;
  std::vector<PatchGridCanonicalTetraLineage> tetrahedra;
  std::vector<std::size_t> tetra_edge_offsets;
  std::vector<PatchGridSignChangingEdgeLineage> edges;
  // For edge e, endpoint zero uses offsets[2*e..2*e+1] and endpoint one uses
  // offsets[2*e+1..2*e+2]. Refs are sorted unique within each endpoint.
  std::vector<std::size_t> edge_endpoint_patch_offsets;
  std::vector<PatchRefId> endpoint_patch_refs;
  std::vector<std::size_t> edge_relation_offsets;
  std::vector<PatchGridEdgePatchRelationEvidence> relations;
  std::vector<std::size_t> edge_location_ref_offsets;
  std::vector<PatchRefId> location_patch_refs;
  std::array<std::size_t, 5> relation_counts{};
  std::size_t axis_edges{0U};
  std::size_t diagonal_edges{0U};
  std::size_t scalar_lineage_certified_edges{0U};
  std::size_t surface_location_certified_edges{0U};
  std::size_t topology_sign_certified_edges{0U};
  // The oracle is setup-local and is destroyed before this result returns.
  // This counts its vector payload while it was live; it is not retained by
  // PreparedPatchGridCellLineage.
  std::size_t oracle_accounted_setup_bytes{0U};
  double oracle_build_ms{0.0};
  PatchGridRelationOracleBindingStatus relation_oracle_binding{
      PatchGridRelationOracleBindingStatus::unavailable};
  // Exact retained vector payload capacity. Fixed fields/status metadata and
  // allocator bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted builder-owned vector payload observed at phase boundaries. It
  // excludes allocator metadata and possible old/new allocation overlap
  // during vector growth, so it is not an exact process peak.
  std::size_t accounted_peak_temporary_bytes{0U};
  double build_ms{0.0};
  bool scalar_lineage_complete{false};
  // Candidate collection can be complete even when no bound oracle exists.
  // A relation other than uncertified is permitted only when
  // relation_oracle_binding == built_once_from_bound_batch.
  bool patch_candidate_collection_complete{false};
  bool patch_relation_classification_complete{false};
  bool surface_location_certificate_complete{false};
  bool topology_sign_certificate_complete{false};
  std::string status{"not_built"};
};

// Slice C1 audit-only lineage. This does not emit triangles, propagate IDs to
// a mesh, or alter marching-tetrahedra connectivity.
[[nodiscard]] PreparedPatchGridCellLineage build_patch_grid_cell_lineage(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    std::span<const PatchGridActiveCellRequest> active_cells);

// Slice C2a audit-only implementation. These types describe canonical
// marching-tetra interpolation vertices and triangle occurrences before
// production's later atom-SDF b/c swap. No production call site consumes this
// payload; it does not authorize connectivity changes, mimic coordinate
// hashing, or certify planar face interiors.

class PatchGridCanonicalInterpolationVertexKey {
public:
  constexpr PatchGridCanonicalInterpolationVertexKey() noexcept = default;

  [[nodiscard]] static bool
  try_make(std::size_t first_vertex_linear_index,
           std::size_t second_vertex_linear_index,
           PatchGridCanonicalInterpolationVertexKey &result) noexcept;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return first_ < second_;
  }
  [[nodiscard]] constexpr std::size_t
  first_vertex_linear_index() const noexcept {
    return first_;
  }
  [[nodiscard]] constexpr std::size_t
  second_vertex_linear_index() const noexcept {
    return second_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(PatchGridCanonicalInterpolationVertexKey,
             PatchGridCanonicalInterpolationVertexKey) noexcept = default;
  [[nodiscard]] friend constexpr bool
  operator<(PatchGridCanonicalInterpolationVertexKey left,
            PatchGridCanonicalInterpolationVertexKey right) noexcept {
    return left.first_ != right.first_ ? left.first_ < right.first_
                                       : left.second_ < right.second_;
  }

private:
  std::size_t first_{std::numeric_limits<std::size_t>::max()};
  std::size_t second_{std::numeric_limits<std::size_t>::max()};
};

struct PatchGridPreOrientationTriangleEdges {
  // Triangle vertex order is the exact pre-add_triangle production order.
  // Each local edge stores its two tetra-local endpoints in canonical order.
  std::array<PatchGridLocalTetraEdge, 3> edges{};

  [[nodiscard]] friend constexpr bool
  operator==(PatchGridPreOrientationTriangleEdges,
             PatchGridPreOrientationTriangleEdges) noexcept = default;
};

// Pure exhaustive helper for all low-four-bit tetra sign masks. A mask with
// high bits returns size_t::max(); valid masks return zero, one, or two
// triangles with production connectivity and pre-add_triangle vertex order.
// Every returned edge key is canonicalized, so this ordering does not certify
// the bitwise directed interpolation point used by production when an edge is
// evaluated inside-to-outside versus outside-to-inside.
[[nodiscard]] std::size_t enumerate_patch_grid_pre_orientation_triangles(
    std::uint8_t tetra_inside_mask,
    std::array<PatchGridPreOrientationTriangleEdges, 2> &triangles) noexcept;

struct PatchGridCanonicalInterpolationVertexLineage {
  PatchGridCanonicalInterpolationVertexKey key;
  Vec3 canonical_interpolation_point;
  double canonical_interpolation_parameter{0.0};
  PatchCrossingAddress exact_location_crossing;
  bool canonical_interpolation_complete{false};
  // Every occurrence sharing the canonical key must reproduce canonical t and
  // point bitwise. A conflict invalidates C2a rather than selecting one.
  bool canonical_edge_occurrence_geometry_bitwise_consistent{false};
  // C2a cannot certify bitwise equivalence to production's directed edge
  // arithmetic; the audit builder leaves this fail-closed.
  bool directed_production_geometry_bitwise_certified{false};
  bool patch_candidate_collection_complete{false};
  bool patch_relation_classification_complete{false};
  bool exact_analytic_location_certified{false};
  // Conservative attribution is present without an exact patch location.
  bool candidate_refs_present_without_exact_location{false};
};

struct PatchGridCanonicalVertexPatchRelationEvidence {
  PatchRefId first;
  PatchRefId second;
  PatchRelation relation{PatchRelation::uncertified};
};

enum class PatchGridTriangleOrientationStage : std::uint8_t {
  pre_atom_sdf_orientation
};

struct PatchGridPreOrientationTriangleOccurrenceLineage {
  std::size_t tetra_position{0U};
  std::uint8_t triangle_ordinal{0U};
  std::array<std::size_t, 3> source_edge_occurrence_positions{};
  // Positions in PreparedPatchGridTriangleOccurrenceLineage::
  // canonical_vertices, in exact pre-orientation production order.
  std::array<std::size_t, 3> canonical_vertex_positions{};
  // Geometry is evaluated from canonical interpolation points. It does not
  // reproduce production's directed IEEE-754 edge arithmetic bitwise.
  double canonical_twice_area{0.0};
  // Candidate-set intersection only; this does not certify face interior.
  PatchRefId single_common_candidate_ref;
  PatchGridTriangleOrientationStage orientation_stage{
      PatchGridTriangleOrientationStage::pre_atom_sdf_orientation};
  // These observations describe canonical geometry only.
  bool finite_geometry_observed{false};
  // Exact zero in the computed canonical cross product. This is an arithmetic
  // observation, not a proof of mathematical degeneracy or good conditioning;
  // near-degenerate analysis is deferred to C2b. With a valid nondegenerate
  // lattice and the positive magnitude clamp this is normally false; the flag
  // is a fail-closed defense for arithmetic collapse/extreme underflow.
  bool computed_zero_area_observed{false};
  bool patch_candidate_summary_complete{false};
  bool single_common_candidate_ref_available{false};
  // Stronger than candidate attribution: all three canonical vertices must
  // have the exact same single certified analytic PatchRefId.
  bool vertex_triplet_patch_identity_certified{false};
  // C2a never certifies the planar triangle interior or final orientation.
  bool planar_face_surface_location_certified{false};
  bool final_mesh_orientation_certified{false};
};

struct PreparedPatchGridTriangleOccurrenceLineage {
  PatchComplexFingerprint complex_fingerprint;
  PatchRayProvenance::LatticeIdentity lattice_identity;
  // One canonical vertex position for each C1 edge occurrence.
  std::vector<std::size_t> edge_occurrence_to_canonical_vertex;
  std::vector<PatchGridCanonicalInterpolationVertexLineage> canonical_vertices;
  // For canonical vertex v, offsets[v..v+1] indexes C1 edge occurrences.
  std::vector<std::size_t> canonical_vertex_occurrence_offsets;
  std::vector<std::size_t> canonical_vertex_edge_occurrences;
  std::vector<std::size_t> canonical_vertex_candidate_ref_offsets;
  std::vector<PatchRefId> canonical_vertex_candidate_refs;
  std::vector<std::size_t> canonical_vertex_relation_offsets;
  std::vector<PatchGridCanonicalVertexPatchRelationEvidence>
      canonical_vertex_relations;
  // Exact location refs remain distinct from conservative candidate refs.
  std::vector<std::size_t> canonical_vertex_exact_location_ref_offsets;
  std::vector<PatchRefId> canonical_vertex_exact_location_refs;
  std::vector<PatchGridPreOrientationTriangleOccurrenceLineage> triangles;
  // Each triangle slice is the intersection of the three canonical-vertex
  // candidate sets. These refs remain candidate-only; even one common ref does
  // not certify the planar triangle interior.
  std::vector<std::size_t> triangle_candidate_ref_offsets;
  std::vector<PatchRefId> triangle_candidate_refs;
  std::array<std::size_t, 5> relation_counts{};
  std::size_t interpolation_occurrences{0U};
  std::size_t canonical_interpolation_vertices{0U};
  std::size_t bitwise_interpolation_conflicts{0U};
  std::size_t exact_location_vertices{0U};
  std::size_t candidate_refs_without_exact_location_vertices{0U};
  std::size_t triangle_occurrences{0U};
  std::size_t finite_triangle_occurrences{0U};
  std::size_t computed_zero_area_triangle_occurrences{0U};
  std::size_t nonfinite_triangle_occurrences{0U};
  std::size_t single_common_candidate_triangles{0U};
  std::size_t vertex_triplet_patch_identity_certified_triangles{0U};
  // Exact retained vector payload capacity; fixed/status metadata and
  // allocator bookkeeping are excluded.
  std::size_t retained_bytes{0U};
  // Accounted builder-owned vector payload observed at phase boundaries; not
  // an exact RSS/process peak and excluding reallocation overlap.
  std::size_t accounted_peak_temporary_bytes{0U};
  double build_ms{0.0};
  bool interpolation_identity_complete{false};
  bool patch_candidate_attribution_complete{false};
  bool triangle_occurrence_lineage_complete{false};
  bool vertex_triplet_patch_identity_classification_complete{false};
  // C2a intentionally cannot certify either claim and must leave both false.
  bool planar_face_surface_location_certified{false};
  bool final_mesh_orientation_certified{false};
  std::string status{"not_built"};
};

// Implemented audit-only C2a builder. It has no production call site and does
// not alter polygonization, connectivity, or orientation.
[[nodiscard]] PreparedPatchGridTriangleOccurrenceLineage
build_patch_grid_pre_orientation_triangle_lineage(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    const PatchRayGrid &grid, const PatchMappingGridSpec &spec,
    const PreparedPatchCrossingWitnesses &crossing_witnesses,
    const PreparedPatchGridVertexMagnitudeWitnesses &vertex_witnesses,
    const PreparedPatchGridCellLineage &cell_lineage);

} // namespace TitanPBE::Tabi::analytic_ses

#endif // TITAN_PBE_TABI_SES_PATCH_MAPPING_H

#ifndef TITAN_PBE_TABI_CALDERON_TRANSFER_H
#define TITAN_PBE_TABI_CALDERON_TRANSFER_H

#include "tabi_calderon_aux_surface.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace TitanPBE::Tabi::calderon {

// Sparse primary-panel to auxiliary-P1 transfer data. The reverse stencil is
// the exact mass-adjoint map under the primary panel-area and auxiliary P1
// lumped masses; it is not an independently interpolated reverse map.
struct TraceTransferWeight {
  std::size_t index{0};
  double value{0.0};

  [[nodiscard]] bool operator==(const TraceTransferWeight &) const = default;
};

using TraceTransferStencil = std::vector<TraceTransferWeight>;

enum class GeometryNormalMatching {
  FacetNearest,
  OwnerSdfAware,
};

struct GeometryNormalMismatchSample {
  std::string direction;
  std::size_t source_index{0};
  std::size_t source_atom_index{0};
  std::size_t matched_index{0};
  std::size_t matched_atom_index{0};
  double distance{0.0};
  double normal_dot{0.0};
  bool same_owner{false};
  bool owner_ambiguous{false};
  bool positive_alternative{false};
};

struct PrimaryAuxiliaryTransferConfig {
  std::size_t nearest_primary_panels{4};
  std::size_t geometry_sample_cap{0}; // Zero audits every source/target point.
  double max_primary_to_auxiliary_distance{0.75};
  double max_auxiliary_to_primary_distance{0.75};
  double min_normal_dot{0.25};
  double min_normal_alignment_fraction{1.0};
  double max_relative_area_mismatch{0.25};
  GeometryNormalMatching normal_matching{GeometryNormalMatching::FacetNearest};
  double normal_candidate_distance_slack{0.25};
  bool require_area_weighted_normal_alignment{false};
  std::size_t mismatch_sample_cap{128};
};

struct PrimaryAuxiliaryTransferResult {
  std::string status{"not_built"};
  std::size_t primary_panels{0};
  std::size_t auxiliary_vertices{0};
  std::size_t primary_geometry_samples{0};
  std::size_t auxiliary_geometry_samples{0};
  double primary_area_a2{0.0};
  double auxiliary_area_a2{0.0};
  double relative_area_mismatch{0.0};
  double primary_to_auxiliary_rms_distance{0.0};
  double primary_to_auxiliary_max_distance{0.0};
  double auxiliary_to_primary_rms_distance{0.0};
  double auxiliary_to_primary_max_distance{0.0};
  double primary_to_auxiliary_min_normal_dot{1.0};
  double auxiliary_to_primary_min_normal_dot{1.0};
  double primary_normal_alignment_fraction{0.0};
  double auxiliary_normal_alignment_fraction{0.0};
  double primary_area_weighted_normal_alignment_fraction{0.0};
  double auxiliary_area_weighted_normal_alignment_fraction{0.0};
  double primary_same_owner_fraction{0.0};
  double auxiliary_same_owner_fraction{0.0};
  double primary_owner_ambiguity_fraction{0.0};
  double auxiliary_owner_ambiguity_fraction{0.0};
  std::size_t primary_negative_normal_samples{0};
  std::size_t auxiliary_negative_normal_samples{0};
  std::size_t primary_positive_alternative_samples{0};
  std::size_t auxiliary_positive_alternative_samples{0};
  double primary_coverage_fraction{0.0};
  double auxiliary_coverage_fraction{0.0};
  double forward_constant_relative{0.0};
  double adjoint_relative{0.0};
  double adjoint_reverse_constant_relative{0.0};
  double build_ms{0.0};
  std::string normal_matching_policy{"facet-nearest"};
  std::vector<GeometryNormalMismatchSample> normal_mismatch_samples;
  std::vector<TraceTransferStencil> primary_to_auxiliary_p1;
  std::vector<TraceTransferStencil> auxiliary_to_primary_adjoint;
};

// C5C.6 balances the sparse transfer support so the auxiliary lumped masses
// and globally area-scaled primary panel masses are exact transport marginals.
// Both directions then preserve constants and are mutual Riesz adjoints.
struct MassCompatibleTransferConfig {
  PrimaryAuxiliaryTransferConfig geometry{};
  std::size_t max_balance_iterations{4000};
  double balance_relative_tolerance{1.0e-11};
  double support_floor{1.0e-12};
};

struct MassCompatibleTransferResult {
  std::string status{"not_built"};
  std::string geometry_status{"not_built"};
  std::size_t primary_panels{0};
  std::size_t auxiliary_vertices{0};
  std::size_t support_edges{0};
  std::size_t support_components{0};
  std::size_t unsupported_primary_panels_before_repair{0};
  std::size_t support_repair_edges{0};
  std::size_t balance_iterations{0};
  double primary_mass_global_scale{0.0};
  double component_mass_scale_min{0.0};
  double component_mass_scale_max{0.0};
  double balance_relative{0.0};
  double forward_constant_relative{0.0};
  double reverse_constant_relative{0.0};
  double adjoint_relative{0.0};
  double max_forward_weight{0.0};
  double max_reverse_weight{0.0};
  double build_ms{0.0};
  PrimaryAuxiliaryTransferResult geometry;
  std::vector<TraceTransferStencil> primary_to_auxiliary_p1;
  std::vector<TraceTransferStencil> auxiliary_to_primary_balanced_adjoint;
};

[[nodiscard]] PrimaryAuxiliaryTransferResult
build_primary_auxiliary_trace_transfer(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const PrimaryAuxiliaryTransferConfig &config = {});

[[nodiscard]] std::vector<double> apply_primary_to_auxiliary_trace_transfer(
    const PrimaryAuxiliaryTransferResult &transfer,
    std::span<const double> primary_values);

[[nodiscard]] std::vector<double>
apply_auxiliary_to_primary_adjoint_trace_transfer(
    const PrimaryAuxiliaryTransferResult &transfer,
    std::span<const double> auxiliary_values);

[[nodiscard]] MassCompatibleTransferResult
build_mass_compatible_primary_auxiliary_trace_transfer(
    std::span<const Panel> primary_panels,
    const AuxiliarySurfaceResult &auxiliary,
    const MassCompatibleTransferConfig &config = {});

[[nodiscard]] std::vector<double>
apply_mass_compatible_primary_to_auxiliary(
    const MassCompatibleTransferResult &transfer,
    std::span<const double> primary_values);

[[nodiscard]] std::vector<double>
apply_mass_compatible_auxiliary_to_primary(
    const MassCompatibleTransferResult &transfer,
    std::span<const double> auxiliary_values);

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_TRANSFER_H

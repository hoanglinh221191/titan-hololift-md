#ifndef TITAN_PBE_TABI_CALDERON_WEAK_FORM_H
#define TITAN_PBE_TABI_CALDERON_WEAK_FORM_H

#include "tabi_calderon_trace_space.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace TitanPBE::Tabi::calderon {

// Dense, synthetic-mesh-only Laplace Calderon calibration. This is deliberately
// separate from the JCP solver and never supplies a production preconditioner.
struct WeakCalderonConfig {
  int adjacent_pair_subdivision_depth{2};
  int duffy_order{6};
  double identity_relative_tolerance{0.25};
};

struct WeakCalderonResult {
  std::string status{"not_built"};
  std::size_t p1_dofs{0};
  std::size_t triangles{0};
  double mass_pivot_ratio{0.0};
  double single_layer_symmetry_relative{0.0};
  double hypersingular_symmetry_relative{0.0};
  double hypersingular_constant_relative{0.0};
  double strong_identity_relative{0.0};
  double weak_identity_mass_scaled_relative{0.0};
  double weak_identity_raw_frobenius_relative{0.0};
  double k_v_commutator_relative{0.0};
  double w_k_commutator_relative{0.0};
  double assembly_ms{0.0};
};

// C5C.1 uses discontinuous P0 functions on barycentric vertex-dual cells.
// This gives the P1 potential trace and normal trace a square, explicit Riesz
// pairing without changing the primary TABI panel discretisation.
struct MixedTraceCalderonConfig {
  int adjacent_pair_subdivision_depth{2};
  double eps_ratio{80.0};
  double kappa{0.125};
};

struct MixedTraceCalderonResult {
  std::string status{"not_built"};
  std::string screened_hypersingular_status{"not_built"};
  std::size_t p1_dofs{0};
  std::size_t dual_p0_dofs{0};
  std::size_t dual_p0_subtriangles{0};
  double dual_p0_partition_relative{0.0};
  double pairing_pivot_ratio{0.0};
  double pairing_inverse_relative{0.0};
  double k1_symmetry_relative{0.0};
  double k2_k3_negative_epsilon_closure_relative{0.0};
  double k2_k3_positive_epsilon_closure_relative{0.0};
  double k2_k3_separated_nonclosure_relative{0.0};
  std::string derivative_block_relation{"not_built"};
  double k4_separated_pair_symmetry_relative{0.0};
  double zero_screening_kernel_absolute{0.0};
  double assembly_ms{0.0};
};

struct ScreenedHypersingularConfig {
  int adjacent_pair_subdivision_depth{2};
  double kappa{0.125};
};

struct ScreenedHypersingularResult {
  std::string status{"not_built"};
  std::string weak_form_sign{"not_built"};
  std::string separated_pair_comparison_status{"not_built"};
  std::size_t p1_dofs{0};
  std::size_t triangles{0};
  std::size_t separated_source_evaluations{0};
  double weak_pre_symmetry_relative{0.0};
  double weak_symmetry_relative{0.0};
  double direct_pre_symmetry_relative{0.0};
  double direct_symmetry_relative{0.0};
  double global_direct_negative_sign_relative{0.0};
  double global_direct_positive_sign_relative{0.0};
  double global_direct_weak_relative{0.0};
  double negative_sign_separated_relative{0.0};
  double positive_sign_separated_relative{0.0};
  double selected_sign_separated_relative{0.0};
  double assembly_ms{0.0};
};

// C5C.4 combines the C5C.1 mixed P1/dual-P0 K1/K2/K3 calibration with
// the C5C.2/C5C.3 screened K4 weak form. It reproduces the current JCP
// sign convention on a closed synthetic mesh, but remains diagnostic-only.
struct CompleteMixedJcpConfig {
  int adjacent_pair_subdivision_depth{2};
  double eps_ratio{80.0};
  double kappa{0.125};
  double direct_action_relative_tolerance{0.025};
};

struct CompleteMixedJcpResult {
  std::string status{"not_built"};
  std::string weak_form_sign{"not_built"};
  std::size_t p1_dofs{0};
  std::size_t dual_p0_dofs{0};
  std::size_t dual_p0_subtriangles{0};
  std::size_t separated_source_evaluations{0};
  double lhs_phi{0.0};
  double lhs_phin{0.0};
  double pairing_pivot_ratio{0.0};
  double pairing_inverse_relative{0.0};
  double direct_k4_weak_relative{0.0};
  double strong_interaction_relative{0.0};
  double strong_system_relative{0.0};
  double deterministic_probe_relative{0.0};
  double assembly_ms{0.0};
};

struct CompleteMixedJcpActionResult {
  std::string status{"not_built"};
  CompleteMixedJcpResult calibration;
  std::vector<double> action;
};

[[nodiscard]] WeakCalderonResult assemble_laplace_weak_calderon_identity(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const WeakCalderonConfig &config = {});

[[nodiscard]] MixedTraceCalderonResult
assemble_jcp_mixed_trace_pairing_calibration(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const MixedTraceCalderonConfig &config = {});

[[nodiscard]] ScreenedHypersingularResult
assemble_jcp_screened_hypersingular_weak_form(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const ScreenedHypersingularConfig &config = {});

[[nodiscard]] CompleteMixedJcpResult assemble_complete_mixed_jcp_calibration(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const CompleteMixedJcpConfig &config = {});

// Small-mesh oracle for C5D lifecycle work. The returned action is the exact
// strong mixed C5C.4 operator applied to one vector; no production path calls
// this dense O(N^2)-storage calibration.
[[nodiscard]] CompleteMixedJcpActionResult
apply_complete_mixed_jcp_dense_reference(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    std::span<const double> unknowns,
    const CompleteMixedJcpConfig &config = {});

} // namespace TitanPBE::Tabi::calderon

#endif // TITAN_PBE_TABI_CALDERON_WEAK_FORM_H

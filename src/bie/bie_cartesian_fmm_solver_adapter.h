#ifndef TITAN_BIE_CARTESIAN_FMM_SOLVER_ADAPTER_H
#define TITAN_BIE_CARTESIAN_FMM_SOLVER_ADAPTER_H

#include "../tabi/tabi_fmm_m2l_fallback.h"
#include "../tabi/tabi_types.h"
#include "fmm/cartesian/bie_cartesian_fmm_operator.h"

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan::bie {

struct CartesianFullFmmPrepareOptions {
  int expansion_order{4};
  int leaf_size{64};
  int max_depth{10};
  double theta{0.45};
  double eps_ratio{80.0};
  double kappa{0.0};
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  // Narrow deterministic seams for the unit-level hardening smoke. These are
  // deliberately absent from production builds and have no CLI surface.
  bool force_scalar_direct_for_testing{false};
  bool inject_invalid_scalar_direct_pair_for_testing{false};
  bool inject_direct_coverage_mismatch_for_testing{false};
#endif
};

#if defined(TITAN_CARTESIAN_FMM_TESTING)
// Aggregated only for apply(..., count_apply=true).  It is written solely by
// the P3D0 test sidecar and is absent from normal Cartesian-full builds.
struct CartesianFullFmmDirectPhaseTiming {
  bool available{false};
  bool finite{true};
  std::size_t counted_apply_calls{0};
  std::uint64_t expected_direct_panel_pairs{0};
  std::uint64_t applied_direct_panel_pairs{0};
  double packed_unknown_cache_reorder_seconds{0.0};
  double packed_worker_scratch_prep_seconds{0.0};
  double packed_parallel_region_seconds{0.0};
  double packed_worker_metric_reduction_seconds{0.0};
  double packed_finite_scan_seconds{0.0};
  double packed_wrapper_residual_seconds{0.0};
  double packed_phase_sum_seconds{0.0};
  double packed_total_seconds{0.0};
  double packed_closure_seconds{0.0};
  double node_patch_correction_seconds{0.0};
  double adapter_direct_total_seconds{0.0};
  double adapter_direct_residual_seconds{0.0};
  double adapter_coverage_bookkeeping_seconds{0.0};
  double adapter_residual_coverage_seconds{0.0};
  double adapter_direct_phase_sum_seconds{0.0};
  double adapter_direct_closure_seconds{0.0};
};
#endif

struct CartesianFullFmmRuntimeStats {
  fmm::cartesian::CartesianFmmPlanStats plan{};
  fmm::cartesian::CartesianFmmApplyStats last_apply{};
  fmm::cartesian::CartesianFmmApplyStats total_apply{};
  std::size_t apply_calls{0};
  std::string requested_engine{"cartesian-full-experimental"};
  std::string effective_engine{"cartesian-full-experimental"};
  int expansion_order{0};
  int leaf_size{0};
  double opening_theta{0.0};
  std::string direct_kernel{"scalar"};
  std::size_t node_patch_correction_pairs{0};
  double direct_plan_seconds{0.0};
  bool finite_outputs{true};
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  CartesianFullFmmDirectPhaseTiming testing_direct_phase_timing{};
#endif
};

// A node-patch direct interaction is centroid-lumped except on the sparse
// topological one-ring.  The packed AVX2 kernel evaluates the centroid term;
// these immutable records add (dual-patch analytic - centroid) without
// revisiting topology or quadrature during each GMRES matvec.
struct NodePatchDirectCorrection {
  std::uint32_t source{0};
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
  double uncoupled_k2{0.0};
};

class CartesianFullFmmSolverOperator {
public:
  static std::expected<std::unique_ptr<CartesianFullFmmSolverOperator>,
                       std::string>
  prepare(const std::vector<TitanPBE::Tabi::Panel> &panels,
          const CartesianFullFmmPrepareOptions &options,
          TitanExecutor &executor);

  CartesianFullFmmSolverOperator(const CartesianFullFmmSolverOperator &) =
      delete;
  CartesianFullFmmSolverOperator &
  operator=(const CartesianFullFmmSolverOperator &) = delete;
  CartesianFullFmmSolverOperator(CartesianFullFmmSolverOperator &&) = delete;
  CartesianFullFmmSolverOperator &
  operator=(CartesianFullFmmSolverOperator &&) = delete;
  ~CartesianFullFmmSolverOperator() = default;

  void apply(const std::vector<double> &unknowns, double lhs_phi,
             double lhs_phin, bool coupled, TitanExecutor &executor,
             std::vector<double> &out, bool count_apply = true);

  [[nodiscard]] const CartesianFullFmmRuntimeStats &stats() const noexcept {
    return stats_;
  }

#if defined(TITAN_CARTESIAN_FMM_TESTING)
  void hold_apply_guard_for_testing();
  void release_apply_guard_for_testing() noexcept;
  [[nodiscard]] titan::tabi::fmm_m2l::PackedFallbackPlanInspection
  packed_direct_plan_inspection_for_testing() const;
#endif

private:
  CartesianFullFmmSolverOperator(
      const std::vector<TitanPBE::Tabi::Panel> &panels,
      CartesianFullFmmPrepareOptions options,
      std::shared_ptr<const fmm::cartesian::CartesianFmmPlan> plan);

  const std::vector<TitanPBE::Tabi::Panel> *panels_{nullptr};
  CartesianFullFmmPrepareOptions options_{};
  fmm::cartesian::CartesianFmmOperator operator_;
  std::vector<double> phi_{};
  std::vector<double> phin_{};
  titan::tabi::fmm_m2l::PackedJcpFallbackPlan packed_direct_plan_{};
  std::vector<std::size_t> node_patch_correction_offsets_{};
  std::vector<NodePatchDirectCorrection> node_patch_corrections_{};
  bool packed_direct_enabled_{false};
  bool node_patch_direct_enabled_{false};
  std::atomic_flag applying_ = ATOMIC_FLAG_INIT;
  CartesianFullFmmRuntimeStats stats_{};
};

} // namespace titan::bie

#endif // TITAN_BIE_CARTESIAN_FMM_SOLVER_ADAPTER_H

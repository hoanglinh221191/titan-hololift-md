#ifndef TITAN_TABI_FMM_CARTESIAN_SAMPLED_PANEL_MATVEC_H
#define TITAN_TABI_FMM_CARTESIAN_SAMPLED_PANEL_MATVEC_H

#include "tabi_fmm_kifmm_fields.h"
#include "tabi_fmm_kifmm_real_panel_cloud.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct CartesianSampledPanelMatvecConfig {
  int order{4};
  double theta{0.6};
  double tail_tol{0.002};
  double strict_geometry_theta{0.5};
  double eps_ratio{80.0};
  double kappa{0.12746};
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
  std::string fallback_kernel{"packed-avx2"};
  std::string refine_mode{"operator-tail-static"};
  std::string cache_mode{"green-jet-compressed"};
  std::string plan_builder{"metric-cache"};
  std::size_t cache_budget_mb{256};
  bool build_direct_reference{true};
  double max_aggregate_relative_rms{1.0e-6};
  double max_channel_relative_rms{5.0e-6};
};

struct CartesianSampledPanelMatvecResult {
  std::string status{"not_started"};
  std::string execution_status{"not_started"};
  std::string accuracy_status{"not_evaluated"};
  std::string accuracy_failure{"none"};
  std::string timing_scope{"sampled_rows_only"};
  std::string pair_count_scope{"sampled_rows_only"};
  std::string fallback_kernel_effective{"unknown"};
  bool finite{false};
  int order{0};
  std::size_t source_panels{0};
  std::size_t sampled_rows{0};
  std::size_t far_panel_pairs{0};
  std::size_t fallback_panel_pairs{0};
  std::size_t production_direct_reference_valid_pairs{0};
  std::size_t invalid_pairs{0};
  std::size_t nonproduction_fallback_pairs{0};
  double aggregate_relative_rms{0.0};
  double k1_relative_rms{0.0};
  double k2_relative_rms{0.0};
  double k3_relative_rms{0.0};
  double k4_relative_rms{0.0};
  double max_channel_relative_rms{0.0};
  std::string max_channel_name{"none"};
  double k4_abs_rms{0.0};
  double k4_reference_rms{0.0};
  double linf_max_abs_tail{0.0};
  double linf_max_relative_tail{0.0};
  std::size_t geometry_guard_rejected_cells{0};
  std::size_t geometry_guard_direct_fallback_pairs{0};
  double geometry_ratio_max{0.0};
  std::size_t fallback_refined_pairs{0};
  std::size_t fallback_base_near_pairs{0};
  std::size_t refined_cells{0};
  std::size_t split_children{0};
  double setup_seconds{0.0};
  double apply_seconds{0.0};
  double direct_reference_seconds{0.0};
  double total_seconds{0.0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t policy_hash{0};
  std::vector<KifmmJcpChannelRow> approximate_rows{};
  std::vector<KifmmJcpChannelRow> direct_rows{};
};

CartesianSampledPanelMatvecResult run_guarded_cartesian_p4_sampled_panel_matvec(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const CartesianSampledPanelMatvecConfig &config);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_CARTESIAN_SAMPLED_PANEL_MATVEC_H

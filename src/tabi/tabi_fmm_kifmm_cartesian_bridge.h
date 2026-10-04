#ifndef TITAN_TABI_FMM_KIFMM_CARTESIAN_BRIDGE_H
#define TITAN_TABI_FMM_KIFMM_CARTESIAN_BRIDGE_H

#include "tabi_fmm_kifmm_fields.h"
#include "tabi_fmm_m2l_local.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmCartesianBridgeConfig {
  int surface_order{8};
  int cartesian_order{5};
  double equiv_scale{1.05};
  double upward_check_scale{3.0};
  double local_equiv_scale{3.0};
  double local_check_scale{1.05};
  double kappa{0.12746};
  double eps_ratio{80.0};
  double aggregate_relative_rms_gate{1.0e-6};
  double max_channel_relative_rms_gate{5.0e-6};
  std::string solver{"qr"};
  std::string condition_policy{"qr-rank-only"};
};

struct KifmmCartesianBridgeRow {
  std::string row_id{"not_started"};
  std::string execution_status{"not_started"};
  std::string accuracy_status{"not_evaluated"};
  std::string accuracy_failure{"none"};
  double aggregate_relative_rms{0.0};
  double k1_relative_rms{0.0};
  double k2_relative_rms{0.0};
  double k3_relative_rms{0.0};
  double k4_relative_rms{0.0};
  double max_channel_relative_rms{0.0};
  std::string max_channel_name{"none"};
  double k4_abs_rms{0.0};
  double k4_reference_rms{0.0};
  double m2e_max_relative_residual{0.0};
  double e2l_coefficient_relative_rms{-1.0};
  std::string l2p_gradient_method{"not_applicable"};
  double l2p_target_gradient_relative_rms{0.0};
  int surface_order{0};
  int cartesian_order{0};
  std::uint64_t policy_hash{0};
};

struct KifmmCartesianBridgeSuiteResult {
  std::string decision{"not_started"};
  std::size_t required_rows{0};
  std::vector<KifmmCartesianBridgeRow> rows{};
};

struct KifmmCartesianSplitLocalBridgeResult {
  titan::tabi::fmm_m2l::CartesianJcpLocalP4 combined_local{};
  titan::tabi::fmm_m2l::CartesianJcpLocalP4 k4_laplace_local{};
  titan::tabi::fmm_m2l::CartesianJcpLocalP4 k4_yukawa_local{};
  bool finite_outputs{false};
  std::string status{"not_started"};
  double local_equiv_max_relative_residual{0.0};
  double e2l_coefficient_relative_rms{-1.0};
};

KifmmJcpChannels bridge_cartesian_moments_to_equivalents(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &source_box,
    const titan::tabi::fmm_m2l::CartesianJcpMoments &moments,
    const KifmmCartesianBridgeConfig &config,
    KifmmEquivalentSystemCache *system_cache = nullptr);

titan::tabi::fmm_m2l::CartesianJcpLocalP4
bridge_equivalent_channels_to_cartesian_local(
    const KifmmSurfaceRule &local_surface_rule, const KifmmBox &target_box,
    const KifmmJcpChannels &channels,
    const KifmmCartesianBridgeConfig &config);

KifmmCartesianSplitLocalBridgeResult
bridge_equivalent_channels_to_cartesian_local_split_k4(
    const KifmmSurfaceRule &local_surface_rule, const KifmmBox &target_box,
    const KifmmJcpChannels &channels,
    const KifmmCartesianBridgeConfig &config);

KifmmCartesianBridgeSuiteResult run_cartesian_kifmm_bridge_smoke_suite();

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_CARTESIAN_BRIDGE_H

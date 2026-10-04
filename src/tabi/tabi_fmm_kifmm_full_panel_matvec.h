#ifndef TITAN_TABI_FMM_KIFMM_FULL_PANEL_MATVEC_H
#define TITAN_TABI_FMM_KIFMM_FULL_PANEL_MATVEC_H

#include "tabi_fmm_kifmm_real_panel_cloud.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace titan::tabi::fmm_kifmm {

struct KifmmFullPanelPreparedState;

struct KifmmFullPanelMatvecResult {
  KifmmLiteResult lite{};
  std::vector<std::size_t> sampled_target_indices{};
  std::string source_box_policy{"tight-bbox"};
  std::string kifmm_full_apply_policy{"serial"};
  std::size_t kifmm_target_tasks{0};
  std::size_t kifmm_parallel_workers{0};
  std::size_t production_fallback_valid_pairs{0};
  std::size_t sampled_direct_reference_valid_pairs{0};
  std::size_t sampled_invalid_pairs{0};
  std::size_t nonproduction_fallback_pairs{0};
  std::uint64_t panel_geometry_hash{0};
  std::uint64_t unknown_vector_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t full_output_hash_phi{0};
  std::uint64_t full_output_hash_phin{0};
  std::uint64_t full_channel_output_hash{0};
  std::uint64_t policy_hash{0};
  std::size_t full_pair_coverage{0};
  double tree_build_seconds{0.0};
  double worklist_build_seconds{0.0};
  double equivalent_build_solve_seconds{0.0};
  double equivalent_matrix_build_seconds{0.0};
  double equivalent_condition_seconds{0.0};
  double equivalent_factor_seconds{0.0};
  double equivalent_rhs_build_seconds{0.0};
  double equivalent_solve_seconds{0.0};
  double kifmm_channel_build_seconds{0.0};
  double full_far_eval_seconds{0.0};
  double full_fallback_eval_seconds{0.0};
  double kifmm_far_eval_seconds{0.0};
  double kifmm_fallback_eval_seconds{0.0};
  double kifmm_parallel_speedup_vs_serial{0.0};
  double sampled_dense_audit_seconds{0.0};
  double setup_seconds{0.0};
  double apply_seconds{0.0};
  double first_apply_seconds{0.0};
  std::size_t equivalent_factor_cache_hits{0};
  std::size_t equivalent_factor_cache_misses{0};
  double equivalent_cache_hit_rate{0.0};
  std::size_t equivalent_system_cache_entries{0};
  std::size_t equivalent_laplace_systems{0};
  std::size_t equivalent_yukawa_systems{0};
  std::size_t unique_equivalent_half_widths{0};
  std::size_t unique_equivalent_kappa_h_values{0};
  double equivalent_half_width_min{0.0};
  double equivalent_half_width_max{0.0};
  double equivalent_systems_per_source_node{0.0};
  std::size_t source_tree_levels{0};
  std::size_t unique_source_node_half_widths{0};
  double source_root_half_width{0.0};
  bool level_cube_geometry{false};
  std::size_t unique_far_source_nodes{0};
  std::size_t cached_source_nodes{0};
  std::size_t source_tree_bytes{0};
  std::size_t target_tree_bytes{0};
  std::size_t worklist_bytes{0};
  std::size_t equivalent_charge_storage_bytes{0};
  std::size_t cached_channel_object_estimated_bytes{0};
  int max_equivalent_rank{0};
  double min_equivalent_sigma_min{0.0};
  double max_equivalent_sigma_max{0.0};
  double max_equivalent_rhs_norm{0.0};
  double max_equivalent_charge_norm{0.0};
  double max_equivalent_charge_to_rhs_norm{0.0};
  bool condition_computed{false};
  bool condition_gate_enabled{false};
  double aggregate_relative_rms{0.0};
  double k1_relative_rms{0.0};
  double k2_relative_rms{0.0};
  double k3_relative_rms{0.0};
  double k4_relative_rms{0.0};
  double aggregate_abs_rms{0.0};
  double k1_abs_rms{0.0};
  double k2_abs_rms{0.0};
  double k3_abs_rms{0.0};
  double k4_abs_rms{0.0};
  double k1_reference_rms{0.0};
  double k2_reference_rms{0.0};
  double k3_reference_rms{0.0};
  double k4_reference_rms{0.0};
  double total_seconds{0.0};
  int order{0};
  bool used_nonproduction_fallback{false};
  bool finite{false};
  std::string status{"not_started"};
};

struct KifmmFullPanelPreparedOperator {
  std::vector<TitanPBE::Tabi::Panel> panels{};
  std::vector<std::size_t> sampled_target_indices{};
  KifmmRealPanelCloudConfig config{};
  std::uint64_t panel_geometry_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::size_t source_panels{0};
  std::size_t target_panels{0};
  std::size_t prepare_build_count{0};
  std::size_t apply_count{0};
  bool frozen_worklist{false};
  bool frozen_equivalent_system_cache{false};
  std::size_t prepared_worklist_pairs{0};
  std::size_t prepared_far_source_nodes{0};
  std::size_t prepared_equivalent_system_cache_entries{0};
  double prepare_seconds{0.0};
  std::shared_ptr<KifmmFullPanelPreparedState> state{};
};

KifmmFullPanelMatvecResult run_kifmm_lite_full_panel_matvec(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmRealPanelCloudConfig &config);

KifmmFullPanelPreparedOperator prepare_kifmm_lite_full_panel_operator(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmRealPanelCloudConfig &config);

KifmmFullPanelMatvecResult apply_prepared_kifmm_lite_full_panel_operator(
    KifmmFullPanelPreparedOperator &prepared,
    std::span<const double> unknowns,
    bool compute_sampled_direct_audit = true);

KifmmFullPanelMatvecResult apply_prepared_kifmm_lite_full_panel_operator(
    KifmmFullPanelPreparedOperator &prepared,
    std::span<const double> unknowns,
    TitanExecutor *executor,
    bool compute_sampled_direct_audit = true);

std::vector<std::vector<std::size_t>>
build_kifmm_lite_fallback_panel_graph(
    const KifmmFullPanelPreparedOperator &prepared,
    std::size_t neighbor_cap);

void assemble_kifmm_lite_tabi_system_matvec(
    const KifmmFullPanelMatvecResult &applied,
    std::span<const double> unknowns,
    double lhs_phi,
    double lhs_phin,
    std::vector<double> &out);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_FULL_PANEL_MATVEC_H

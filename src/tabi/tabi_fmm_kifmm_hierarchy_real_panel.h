#ifndef TITAN_TABI_FMM_KIFMM_HIERARCHY_REAL_PANEL_H
#define TITAN_TABI_FMM_KIFMM_HIERARCHY_REAL_PANEL_H

#include "tabi_fmm_kifmm_hierarchy_jcp.h"
#include "tabi_fmm_kifmm_sampled_panel_matvec.h"
#include "tabi_types.h"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmHierarchyRealPanelResult {
  std::string status{"not_started"};
  std::string execution_status{"not_started"};
  std::string accuracy_status{"not_evaluated"};
  std::string accuracy_failure{"none"};

  std::size_t source_panels{0};
  std::size_t target_panels{0};
  std::size_t sampled_rows{0};
  std::uint64_t full_coverage_expected{0};
  std::uint64_t sampled_coverage_expected{0};
  std::string timing_scope{"sampled_rows_only"};
  std::string compression_status{"not_evaluated"};
  std::string full_apply_stage_status{"not_evaluated"};
  double kifmm_compression_ratio{
      std::numeric_limits<double>::quiet_NaN()};
  double cartesian_k4_compression_ratio{
      std::numeric_limits<double>::quiet_NaN()};
  int order{0};
  double theta{0.0};
  double eps_ratio{80.0};
  double kappa{0.12746};
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
  std::string condition_policy{"qr-rank-only"};
  std::string source_box_policy{"strict-octree"};
  std::string m2l_eval_policy{"scalar-existing"};
  std::string m2l_matrix_cache_policy{"off"};
  std::string m2l_gemm_batch_policy{"per-pair"};
  std::string m2l_parallel_policy{"serial"};
  int m2l_parallel_workers{1};
  std::string tree_stop_policy{"baseline"};
  std::string jcp_channel_policy{"all"};
  std::string hierarchy_k4_policy{"computed"};
  std::uint64_t hierarchy_k4_evaluation_count{0};
  double tree_panel_radius_factor{0.0};
  int tree_occupancy_stop{0};
  std::string real_panel_guard{"off"};
  std::string hier_debug_mode{"full"};
  double real_panel_guard_margin_factor{0.0};
  double real_panel_guard_margin_max{0.0};

  double aggregate_relative_rms_gate{1.0e-6};
  double max_channel_relative_rms_gate{5.0e-6};
  double aggregate_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k1_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k2_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k3_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k4_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double max_channel_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  std::string max_channel_name{"none"};
  double k4_abs_rms{std::numeric_limits<double>::quiet_NaN()};
  double k4_reference_rms{std::numeric_limits<double>::quiet_NaN()};

  double far_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double far_k1_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double far_k2_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double far_k3_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double far_k4_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double fallback_k4_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t far_direct_valid_pairs{0};
  double source_box_boundary_margin_min{
      std::numeric_limits<double>::quiet_NaN()};
  double target_check_excess_min{std::numeric_limits<double>::quiet_NaN()};
  double target_panel_envelope_excess_min{
      std::numeric_limits<double>::quiet_NaN()};
  double target_local_check_excess_min{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t real_guard_rejected_pairs{0};
  std::uint64_t real_guard_split_pairs{0};
  std::uint64_t real_guard_direct_pairs{0};
  std::uint64_t negative_target_check_excess_pairs{0};
  std::uint64_t negative_target_check_excess_panel_pairs{0};
  std::uint64_t negative_target_local_check_excess_pairs{0};
  std::uint64_t negative_target_local_check_excess_panel_pairs{0};
  double child_local_check_inside_parent_local_check_min_margin{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t negative_l2l_local_containment_pairs{0};
  double min_leaf_half_width_over_panel_radius{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t panels_crossing_node_boundary{0};
  int tree_node_count{0};
  int tree_leaf_count{0};
  int tree_max_depth_observed{0};
  int tree_radius_stop_leaf_count{0};
  int tree_occupancy_stop_leaf_count{0};
  int tree_leaf_size_stop_count{0};
  int tree_max_depth_stop_count{0};
  int tree_unsplittable_stop_count{0};
  double tree_min_leaf_half_width_over_panel_radius{
      std::numeric_limits<double>::quiet_NaN()};
  double tree_max_leaf_panel_radius{0.0};
  double tree_mean_leaf_source_count{0.0};
  double tree_mean_leaf_target_count{0.0};
  std::array<std::uint64_t, 5> mac_bin_cell_pairs{};
  std::array<std::uint64_t, 5> mac_bin_panel_pairs{};
  std::array<double, 5> mac_bin_min_target_check_excess{
      {std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::quiet_NaN()}};

  std::uint64_t sampled_dense_reference_valid_pairs{0};
  std::uint64_t production_fallback_pairs{0};
  std::uint64_t production_invalid_pairs{0};
  std::uint64_t nonproduction_fallback_pairs{0};
  std::uint64_t far_panel_pairs{0};
  std::uint64_t direct_panel_pairs{0};
  std::uint64_t exact_pair_coverage{0};
  double far_fraction{0.0};

  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::uint64_t panel_geometry_hash{0};
  std::uint64_t unknown_vector_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t output_hash{0};

  double hierarchy_seconds{0.0};
  double production_fallback_seconds{0.0};
  double dense_oracle_seconds{0.0};
  double total_seconds{0.0};
  std::string runtime_memory_attribution_status{"not_evaluated"};
  double apply_source_build_seconds{0.0};
  double apply_hierarchy_seconds{0.0};
  double apply_guard_audit_seconds{0.0};
  double apply_production_fallback_seconds{0.0};
  double apply_cartesian_k4_seconds{0.0};
  double apply_dense_oracle_seconds{0.0};
  double apply_finalize_seconds{0.0};
  double apply_unattributed_seconds{0.0};
  std::uint64_t apply_rss_start_bytes{0};
  std::uint64_t apply_peak_working_set_start_bytes{0};
  std::uint64_t apply_rss_after_source_build_bytes{0};
  std::uint64_t apply_peak_working_set_after_source_build_bytes{0};
  std::uint64_t apply_rss_after_hierarchy_bytes{0};
  std::uint64_t apply_peak_working_set_after_hierarchy_bytes{0};
  std::uint64_t apply_rss_after_guard_audit_bytes{0};
  std::uint64_t apply_peak_working_set_after_guard_audit_bytes{0};
  std::uint64_t apply_rss_after_fallback_bytes{0};
  std::uint64_t apply_peak_working_set_after_fallback_bytes{0};
  std::uint64_t apply_rss_after_cartesian_k4_bytes{0};
  std::uint64_t apply_peak_working_set_after_cartesian_k4_bytes{0};
  std::uint64_t apply_rss_after_dense_oracle_bytes{0};
  std::uint64_t apply_peak_working_set_after_dense_oracle_bytes{0};
  std::uint64_t apply_rss_after_finalize_bytes{0};
  std::uint64_t apply_peak_working_set_after_finalize_bytes{0};
  std::uint64_t apply_peak_working_set_delta_bytes{0};
  std::uint64_t m2l_matrix_cache_entries{0};
  std::uint64_t m2l_matrix_cache_peak_entries{0};
  std::uint64_t m2l_matrix_cache_hits{0};
  std::uint64_t m2l_matrix_cache_misses{0};
  double m2l_matrix_cache_hit_rate{0.0};
  std::uint64_t m2l_matrix_cache_limit_bytes{0};
  std::uint64_t m2l_matrix_cache_evictions{0};
  std::uint64_t m2l_matrix_cache_estimated_bytes{0};
  std::uint64_t m2l_matrix_cache_peak_estimated_bytes{0};
  std::uint64_t m2l_matrix_cache_total_built_bytes{0};
  double m2l_matrix_cache_build_seconds{0.0};
  double m2l_matrix_gemm_seconds{0.0};
  std::string m2l_matrix_cache_key_status{"not_used"};
  std::uint64_t m2l_gemm_group_count{0};
  std::uint64_t m2l_gemm_grouped_pair_count{0};
  std::uint64_t m2l_gemm_grouped_rhs_columns{0};
  std::uint64_t m2l_gemm_max_group_size{0};
  std::uint64_t m2l_parallel_task_count{0};
  std::uint64_t m2l_parallel_group_count{0};
  std::uint64_t m2l_parallel_max_task_pairs{0};
  double m2l_parallel_prepare_seconds{0.0};
  double m2l_parallel_eval_seconds{0.0};
  double m2l_parallel_merge_seconds{0.0};
  std::uint64_t m2l_target_nodes{0};
  std::uint64_t m2l_source_nodes{0};
  std::uint64_t m2l_check_points_total{0};
  std::uint64_t m2l_equivalent_points{0};
  std::uint64_t m2l_kernel_evals_estimated{0};
  double m2l_seconds_per_translation{0.0};
  double m2l_seconds_per_check_point{0.0};

  KifmmHierarchyJcpResult hierarchy{};
  KifmmHierarchyJcpResult flat{};
  double flat_vs_dense_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double flat_vs_dense_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double flat_vs_dense_max_channel_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double hierarchy_vs_dense_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double hierarchy_vs_dense_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double hierarchy_vs_flat_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double hierarchy_vs_flat_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t flat_output_hash{0};
  double local_check_rhs_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double local_check_rhs_max_abs_error{
      std::numeric_limits<double>::quiet_NaN()};
  double l2p_oracle_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double l2p_oracle_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double k4_laplace_grad_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double k4_yukawa_grad_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double k4_combined_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double split_k4_laplace_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double split_k4_yukawa_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double split_k4_combined_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double split_vs_dense_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double split_vs_flat_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  std::string hybrid_mode{"none"};
  double m2e_max_relative_residual{
      std::numeric_limits<double>::quiet_NaN()};
  double local_equiv_max_relative_residual{
      std::numeric_limits<double>::quiet_NaN()};
  double e2l_coefficient_relative_rms{-1.0};
  std::string e2l_coefficient_status{"not_applicable"};
  std::string k4_decomposition_status{"not_evaluated"};
  std::string hierarchy_inner_status{"none"};
  std::string hierarchy_inner_accuracy_failure{"none"};
  std::string l2p_gradient_method{"not_applicable"};
  std::string k1_source{"hierarchy_q8"};
  std::string k2_source{"hierarchy_q8"};
  std::string k3_source{"hierarchy_q8"};
  std::string k4_source{"hierarchy_q8"};
  std::string channel_hybrid_double_count_guard{"not_applicable"};
  double hierarchy_k123_max_channel_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  std::string hierarchy_k123_max_channel_name{"none"};
  double hybrid_recombined_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double hybrid_recombined_max_channel_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  std::string hybrid_recombined_max_channel_name{"none"};
  std::uint64_t channel_hybrid_recombined_hash{0};
  std::string cartesian_k4_status{"not_run"};
  std::string cartesian_k4_execution_status{"not_started"};
  std::string cartesian_k4_accuracy_status{"not_evaluated"};
  std::string cartesian_k4_accuracy_failure{"not_evaluated"};
  double cartesian_k4_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double cartesian_k4_abs_rms{std::numeric_limits<double>::quiet_NaN()};
  double cartesian_k4_reference_rms{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t cartesian_k4_far_panel_pairs{0};
  std::uint64_t cartesian_k4_fallback_panel_pairs{0};
  double cartesian_k4_total_seconds{0.0};
  std::uint64_t cartesian_k4_policy_hash{0};
  std::uint64_t shared_worklist_hash{0};
  double shared_theta{0.0};
  std::string shared_guard{"none"};
  std::uint64_t shared_far_pairs{0};
  std::uint64_t shared_direct_pairs{0};
  double shared_far_fraction{0.0};
  std::uint64_t cartesian_k4_shared_far_pairs{0};
  std::uint64_t cartesian_k4_shared_direct_pairs{0};
  std::uint64_t direct_avx2_all_channel_pairs{0};
  std::string worklist_consistency_status{"not_evaluated"};
  std::uint64_t kifmm_worklist_hash{0};
  std::uint64_t kifmm_far_pairs{0};
  std::uint64_t kifmm_direct_pairs{0};
  double kifmm_far_fraction{0.0};
  std::string dual_worklist_status{"not_evaluated"};
  std::string channel_recombination_status{"not_evaluated"};
  std::string worklist_independence_status{"not_evaluated"};
  std::vector<KifmmHierarchyFieldAttribution> field_attribution{};
  std::vector<KifmmJcpChannelRow> full_approximate_rows{};

  std::string lite_status{"not_run"};
  double lite_aggregate_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double lite_max_channel_relative_rms{
      std::numeric_limits<double>::quiet_NaN()};
  double lite_total_seconds{0.0};
};

KifmmHierarchyRealPanelResult run_kifmm_hierarchy_real_panel_sampled(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config);

KifmmHierarchyRealPanelResult
run_kifmm_hierarchy_real_panel_full_apply_sampled_audit(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config);

struct DualWorklistChannelHybridPreparedOperator {
  std::vector<TitanPBE::Tabi::Panel> panels{};
  std::vector<std::size_t> sampled_target_indices{};
  std::vector<Vec3> targets{};
  std::vector<Vec3> target_normals{};
  std::vector<std::size_t> all_target_indices{};
  KifmmHierarchyTree tree{};
  KifmmHierarchyWorklist worklist{};
  KifmmHierarchyJcpConfig config{};
  std::uint64_t panel_geometry_hash{0};
  std::uint64_t sampled_target_hash{0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::size_t source_panels{0};
  std::size_t target_panels{0};
  std::size_t prepare_build_count{0};
  std::size_t apply_count{0};
  bool frozen_worklist{false};
  bool frozen_equivalent_system_cache{false};
  std::size_t prepared_worklist_pairs{0};
  std::size_t prepared_equivalent_system_cache_entries{0};
  double prepare_seconds{0.0};
  std::string prepared_rebuild_status{"not_prepared"};
};

DualWorklistChannelHybridPreparedOperator
prepare_dual_worklist_channel_hybrid_operator(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const std::size_t> sampled_target_indices,
    const KifmmHierarchyJcpConfig &config);

KifmmHierarchyRealPanelResult
apply_prepared_dual_worklist_channel_hybrid_operator(
    DualWorklistChannelHybridPreparedOperator &prepared,
    std::span<const double> unknowns,
    bool compute_sampled_direct_audit = true);

void assemble_dual_worklist_channel_hybrid_tabi_system_matvec(
    const KifmmHierarchyRealPanelResult &applied,
    std::span<const double> unknowns,
    double lhs_phi,
    double lhs_phin,
    std::vector<double> &out);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_HIERARCHY_REAL_PANEL_H

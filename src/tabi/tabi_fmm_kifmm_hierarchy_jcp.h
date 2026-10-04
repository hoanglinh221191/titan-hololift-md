#ifndef TITAN_TABI_FMM_KIFMM_HIERARCHY_JCP_H
#define TITAN_TABI_FMM_KIFMM_HIERARCHY_JCP_H

#include "tabi_fmm_kifmm_fields.h"
#include "tabi_fmm_kifmm_hierarchy.h"

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmHierarchyJcpConfig {
  KifmmHierarchyPolicy hierarchy{};
  double kappa{0.12746};
  double eps_ratio{80.0};
  double aggregate_relative_rms_gate{1.0e-6};
  double max_channel_relative_rms_gate{5.0e-6};
  std::string real_panel_guard{"off"};
  double real_panel_guard_margin_factor{0.0};
  std::string hier_debug_mode{"full"};
  bool build_dense_oracle{true};
  bool include_direct_fallback{true};
};

struct KifmmHierarchyFieldAttribution {
  std::string scope{"none"};
  std::string field_name{"none"};
  std::string kernel{"none"};
  int node_id{-1};
  int node_level{-1};
  double relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double abs_rms{std::numeric_limits<double>::quiet_NaN()};
  double reference_rms{std::numeric_limits<double>::quiet_NaN()};
  double max_abs_error{std::numeric_limits<double>::quiet_NaN()};
};

struct KifmmHierarchyJcpResult {
  std::string status{"not_started"};
  std::string execution_status{"not_started"};
  std::string accuracy_status{"not_evaluated"};
  std::string accuracy_failure{"none"};
  std::string hier_debug_mode{"full"};
  KifmmHierarchyStats stats{};
  double aggregate_relative_rms_gate{1.0e-6};
  double max_channel_relative_rms_gate{5.0e-6};
  double aggregate_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k1_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k2_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k3_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double k4_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double max_channel_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  std::string max_channel_name{"none"};
  double max_abs_error{std::numeric_limits<double>::quiet_NaN()};
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
  double k4_abs_rms{std::numeric_limits<double>::quiet_NaN()};
  double k4_reference_rms{std::numeric_limits<double>::quiet_NaN()};
  std::string hybrid_mode{"none"};
  double m2e_max_relative_residual{
      std::numeric_limits<double>::quiet_NaN()};
  double local_equiv_max_relative_residual{
      std::numeric_limits<double>::quiet_NaN()};
  double e2l_coefficient_relative_rms{-1.0};
  std::string e2l_coefficient_status{"not_applicable"};
  std::string k4_decomposition_status{"not_computed"};
  std::string l2p_gradient_method{"not_applicable"};
  double p2m_seconds{0.0};
  double p2m_rhs_seconds{0.0};
  double p2m_solve_seconds{0.0};
  double m2m_seconds{0.0};
  double m2l_seconds{0.0};
  double local_solve_seconds{0.0};
  double l2l_seconds{0.0};
  double l2p_seconds{0.0};
  double direct_fallback_seconds{0.0};
  double dense_oracle_seconds{0.0};
  double total_seconds{0.0};
  std::string hierarchy_runtime_memory_attribution_status{"not_evaluated"};
  std::uint64_t hierarchy_rss_start_bytes{0};
  std::uint64_t hierarchy_peak_start_bytes{0};
  std::uint64_t hierarchy_rss_after_direct_fallback_bytes{0};
  std::uint64_t hierarchy_peak_after_direct_fallback_bytes{0};
  std::uint64_t hierarchy_rss_after_p2m_bytes{0};
  std::uint64_t hierarchy_peak_after_p2m_bytes{0};
  std::uint64_t hierarchy_rss_after_m2m_bytes{0};
  std::uint64_t hierarchy_peak_after_m2m_bytes{0};
  std::uint64_t hierarchy_rss_after_m2l_bytes{0};
  std::uint64_t hierarchy_peak_after_m2l_bytes{0};
  std::uint64_t hierarchy_rss_after_l2l_bytes{0};
  std::uint64_t hierarchy_peak_after_l2l_bytes{0};
  std::uint64_t hierarchy_rss_after_l2p_bytes{0};
  std::uint64_t hierarchy_peak_after_l2p_bytes{0};
  std::uint64_t hierarchy_rss_after_dense_oracle_bytes{0};
  std::uint64_t hierarchy_peak_after_dense_oracle_bytes{0};
  std::uint64_t hierarchy_rss_after_finalize_bytes{0};
  std::uint64_t hierarchy_peak_after_finalize_bytes{0};
  std::uint64_t hierarchy_peak_delta_bytes{0};
  std::uint64_t k123_upward_equivalent_estimated_bytes{0};
  std::uint64_t k123_local_rhs_estimated_bytes{0};
  std::uint64_t k123_dense_local_rhs_estimated_bytes{0};
  std::uint64_t k123_local_equivalent_estimated_bytes{0};
  std::uint64_t k123_rows_estimated_bytes{0};
  std::uint64_t k123_equivalent_system_cache_estimated_bytes{0};
  std::uint64_t k123_total_estimated_bytes{0};
  std::string m2l_eval_policy{"scalar-existing"};
  std::string m2l_matrix_cache_policy{"off"};
  std::string m2l_gemm_batch_policy{"per-pair"};
  std::string m2l_parallel_policy{"serial"};
  int m2l_parallel_workers{1};
  std::string jcp_channel_policy{"all"};
  std::string hierarchy_k4_policy{"computed"};
  std::uint64_t hierarchy_k4_evaluation_count{0};
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
  int equivalent_factor_cache_entries{0};
  int equivalent_factor_cache_hits{0};
  int equivalent_factor_cache_misses{0};
  std::uint64_t p2m_leaf_solves{0};
  std::uint64_t m2m_translations{0};
  std::uint64_t m2m_parent_solves{0};
  std::uint64_t m2l_translations{0};
  std::uint64_t local_node_solves{0};
  std::uint64_t l2l_translations{0};
  std::uint64_t l2p_target_count{0};
  std::uint64_t local_rhs_adds{0};
  std::uint64_t upward_rhs_adds{0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::uint64_t output_hash{0};
  std::vector<KifmmJcpChannelRow> approximate_rows{};
  std::vector<KifmmJcpChannelRow> direct_rows{};
  std::vector<KifmmHierarchyFieldAttribution> field_attribution{};
};

KifmmHierarchyJcpResult run_full_hierarchical_jcp_kifmm(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config);

KifmmHierarchyJcpResult run_full_hierarchical_jcp_kifmm_with_worklist(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config, const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist);

KifmmHierarchyJcpResult run_flat_jcp_kifmm_with_worklist(
    std::span<const KifmmPanelSource> sources, std::span<const Vec3> targets,
    std::span<const Vec3> target_normals,
    const KifmmHierarchyJcpConfig &config, const KifmmHierarchyTree &tree,
    const KifmmHierarchyWorklist &worklist);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_HIERARCHY_JCP_H

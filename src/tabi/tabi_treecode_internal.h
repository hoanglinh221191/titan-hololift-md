#pragma once

#include "tabi/tabi_types.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi::detail {

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr std::size_t kBranchFactor = 8;
inline constexpr std::size_t kDefaultLeafSize = 32;
inline constexpr double kDefaultOpeningTheta = 0.45;
inline constexpr std::uint32_t kMortonMax = (1u << 21) - 1u;
inline constexpr int kTaylorOrderDefault = 3;
inline constexpr int kTaylorOrderMax = 5;
inline constexpr std::size_t kTaylorMomentMax = 56;
inline constexpr double kFragmentationGapRatioThreshold = 0.75;
inline constexpr std::size_t kJcpK1PhiChannel = 0;
inline constexpr std::size_t kJcpK2PhiNormalChannel = 1;
inline constexpr std::size_t kJcpK3PhinChannel = 2;
inline constexpr std::size_t kJcpK4MixedChannel = 3;
inline constexpr int kSeriesSide = kTaylorOrderMax + 1;
inline constexpr std::size_t kSeriesSize = kSeriesSide * kSeriesSide * kSeriesSide;

struct MomentIndex {
  int ax{0};
  int ay{0};
  int az{0};
  double inv_factorial{1.0};
};



struct NodeMoments {
  std::array<double, kTaylorMomentMax> values{};
};



struct TaylorSeries {
  std::array<double, kSeriesSize> coeff{};
};



struct MacAuditTally {
  std::size_t unsafe_count{0};
  std::size_t unsafe_panels{0};
  double min_gap{0.0};
  double min_gap_ratio{0.0};
  double sum_gap_ratio{0.0};
  std::size_t samples{0};
  std::size_t physical_gap_rejects{0};
};



struct FragmentationTally {
  std::size_t far_count{0};
  std::size_t far_panels{0};
  std::size_t split_rejects{0};
  double gap_ratio_sum{0.0};
  double gap_ratio_max{0.0};
};



struct ExpansionAuditTally {
  std::size_t pairs{0};
  std::size_t panels{0};
  double direct_ms{0.0};
  double rel_error_sum{0.0};
  double rel_error_max{0.0};
  std::vector<double> rel_errors;
  std::array<double, kJcpExpansionChannelCount> channel_rel_error_sum{};
  std::array<double, kJcpExpansionChannelCount> channel_rel_error_max{};
  std::array<std::vector<double>, kJcpExpansionChannelCount>
      channel_rel_errors;
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_high_error_caught{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_high_error_missed{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_high_error_caught_by_split{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_phantom_high_error_count{};
  std::array<double, kJcpExpansionChannelCount>
      channel_phantom_high_error_abs_max{};
  std::size_t worst_channel{kJcpExpansionChannelCount};
  std::size_t estimator_high_error_caught{0};
  std::size_t estimator_high_error_missed{0};
  std::size_t estimator_high_error_caught_by_split{0};
  std::size_t estimator_phantom_high_error_count{0};
  double estimator_phantom_high_error_abs_max{0.0};
  std::size_t convergence_overreject_count{0};
  std::size_t convergence_overreject_panels{0};
  std::size_t convergence_overreject_caught_count{0};
  std::size_t convergence_overreject_caught_panels{0};
  double convergence_overreject_abs_error_sum{0.0};
  double convergence_overreject_abs_error_max{0.0};
  double convergence_overreject_estimated_abs_error_sum{0.0};
  double convergence_overreject_estimated_abs_error_max{0.0};
  std::vector<double> convergence_overreject_abs_error_samples;
  std::vector<double> convergence_overreject_required_scale_samples;
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_caught_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      convergence_channel_overreject_panels{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_sum{};
  std::array<double, kJcpExpansionChannelCount>
      convergence_channel_overreject_abs_error_max{};
  std::array<std::vector<double>, kJcpExpansionChannelCount>
      convergence_channel_overreject_required_scale_samples;
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      accepted_miss_channel_panels{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_rho_sum{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_rho_min{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_rho_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_sum{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_tail_ratio_max{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_sum{};
  std::array<double, kJcpExpansionChannelCount>
      accepted_miss_channel_estimated_error_max{};
  struct AcceptedMissRecord {
    std::size_t target_index{0};
    std::size_t source_node_index{0};
    std::size_t channel{kJcpExpansionChannelCount};
    double rho{0.0};
    double tail_ratio{0.0};
    double estimated_error{0.0};
    double rel_error{0.0};
    double abs_error{0.0};
    bool is_phantom{false};
  };
  std::vector<AcceptedMissRecord> accepted_miss_records;
  struct OverrejectCalibrationRecord {
    std::size_t target_index{0};
    std::size_t source_node_index{0};
    std::size_t channel{kJcpExpansionChannelCount};
    std::size_t node_panels{0};
    double rho{0.0};
    double aspect_ratio{0.0};
    double estimated_rel_error{0.0};
    double estimated_abs_error{0.0};
    double oracle_rel_error{0.0};
    double oracle_abs_error{0.0};
    double required_scale{0.0};
    bool caught{false};
  };
  std::vector<OverrejectCalibrationRecord> overreject_calibration_records;
  struct ConvergenceShapeRecord {
    std::size_t target_index{0};
    std::size_t source_node_index{0};
    std::size_t channel{kJcpExpansionChannelCount};
    std::size_t node_panels{0};
    double rho{0.0};
    double rho_power{0.0};
    double aspect_ratio{0.0};
    double node_radius{0.0};
    double target_distance{0.0};
    double source_norm{0.0};
    double tail_norm_p{0.0};
    double tail_norm_pm1{0.0};
    double tail_ratio{0.0};
    double effective_amp{0.0};
    double estimated_rel_error{0.0};
    double estimated_abs_error{0.0};
    double oracle_rel_error{0.0};
    double oracle_abs_error{0.0};
    double oracle_abs_over_source_norm{0.0};
    double oracle_abs_over_tail_norm{0.0};
    double oracle_abs_over_tail_rho{0.0};
    bool caught{false};
  };
  std::vector<ConvergenceShapeRecord> convergence_shape_records;
  struct KernelResponseRecord {
    std::size_t target_index{0};
    std::size_t source_node_index{0};
    std::size_t channel{kJcpExpansionChannelCount};
    std::size_t node_panels{0};
    double rho{0.0};
    double aspect_ratio{0.0};
    std::string source_family;
    double source_norm{0.0};
    double tail_norm_p{0.0};
    double tail_norm_pm1{0.0};
    double estimated_abs_error{0.0};
    double kernel_tail_p_abs{0.0};
    double kernel_tail_pm1_abs{0.0};
    double kernel_tail_envelope_abs{0.0};
    double oracle_rel_error{0.0};
    double oracle_abs_error{0.0};
    double oracle_abs_over_kernel_tail_envelope{0.0};
    bool caught{false};
  };
  std::vector<KernelResponseRecord> kernel_response_records;
};



struct ConvergenceAuditTally {
  std::size_t pairs{0};
  double rho_sum{0.0};
  double rho_max{0.0};
  double aspect_sum{0.0};
  double aspect_max{0.0};
};



struct ConvergenceEstimatorDecision {
  bool active{false};
  bool unstable{false};
  bool split_reject{false};
  double tail_norm_p{0.0};
  double tail_norm_pm1{0.0};
  double source_norm{0.0};
  double tail_ratio{0.0};
  double estimated_error{0.0};
  double estimated_abs_error{0.0};
  double dielectric_jcp_amp{1.0};
  double effective_jcp_amp{1.0};
  std::array<double, kJcpExpansionChannelCount> channel_estimated_error{};
  std::array<double, kJcpExpansionChannelCount> channel_estimated_abs_error{};
  std::array<double, kJcpExpansionChannelCount> channel_effective_jcp_amp{};
  std::array<double, kJcpExpansionChannelCount> channel_source_norm{};
  std::array<double, kJcpExpansionChannelCount> channel_tail_norm_p{};
  std::array<double, kJcpExpansionChannelCount> channel_tail_norm_pm1{};
  std::array<double, kJcpExpansionChannelCount> channel_tail_ratio{};
  std::array<bool, kJcpExpansionChannelCount> channel_unstable{};
  bool low_abs_reaccepted{false};
  std::array<bool, kJcpExpansionChannelCount> channel_low_abs_reaccepted{};
};



struct ConvergenceEstimatorTally {
  std::size_t samples{0};
  std::size_t unstable_count{0};
  std::size_t unstable_panels{0};
  std::size_t split_rejects{0};
  std::size_t split_reject_panels{0};
  std::size_t low_abs_reaccepted_count{0};
  std::size_t low_abs_reaccepted_panels{0};
  double tail_ratio_sum{0.0};
  double tail_ratio_max{0.0};
  double estimated_error_sum{0.0};
  double estimated_error_max{0.0};
  double estimated_abs_error_sum{0.0};
  double estimated_abs_error_max{0.0};
  double tail_norm_p_sum{0.0};
  double tail_norm_p_max{0.0};
  double tail_norm_pm1_sum{0.0};
  double tail_norm_pm1_max{0.0};
  double dielectric_jcp_amp_sum{0.0};
  double dielectric_jcp_amp_max{0.0};
  double effective_jcp_amp_sum{0.0};
  double effective_jcp_amp_max{0.0};
  std::array<double, kJcpExpansionChannelCount>
      channel_estimated_error_sum{};
  std::array<double, kJcpExpansionChannelCount>
      channel_estimated_error_max{};
  std::array<double, kJcpExpansionChannelCount>
      channel_effective_jcp_amp_sum{};
  std::array<double, kJcpExpansionChannelCount>
      channel_effective_jcp_amp_max{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_unstable_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_unstable_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_split_rejects{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_split_reject_panels{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_low_abs_reaccepted_count{};
  std::array<std::size_t, kJcpExpansionChannelCount>
      channel_low_abs_reaccepted_panels{};
};



struct LeafPairBatchTally {
  std::size_t interactions{0};
  std::size_t targets{0};
  std::size_t source_panels{0};
  std::size_t residual_panels{0};
  std::size_t fallbacks{0};
  std::size_t vectorized_sources{0};
  std::size_t vector_lanes{0};
  std::size_t soa_sources{0};
  std::size_t scalar_tasks{0};
  std::size_t batch_tasks{0};
  std::size_t source_panel_evals{0};
  std::size_t selection_candidate_scans{0};
  std::size_t selection_clean_rejects{0};
  std::size_t selection_probe_calls{0};
  std::size_t selection_probe_node_visits{0};
  std::size_t selection_probe_cache_hits{0};
  std::size_t selection_probe_cache_misses{0};
  std::size_t selection_insufficient_targets{0};
  std::size_t selection_same_target_leaf_rejects{0};
  std::size_t directness_ancestor_probe_calls{0};
  std::size_t directness_ancestor_probe_node_visits{0};
  std::size_t directness_pair_cache_hits{0};
  std::size_t directness_pair_cache_misses{0};
  std::size_t directness_all_direct_proofs{0};
  std::size_t directness_mixed_fallbacks{0};
  std::size_t directness_verify_checks{0};
  std::size_t directness_verify_mismatches{0};
  std::size_t worklist_tasks{0};
  std::size_t worklist_groups{0};
  std::size_t worklist_batched_groups{0};
  std::size_t worklist_scalar_flush_tasks{0};
  std::size_t worklist_forced_flushes{0};
  std::size_t worklist_active_groups_max{0};
  std::size_t worklist_verify_checks{0};
  std::size_t worklist_verify_mismatches{0};
  std::size_t leaf_block_target_leaves{0};
  std::size_t leaf_block_targets{0};
  std::size_t leaf_block_source_groups{0};
  std::size_t leaf_block_batched_groups{0};
  std::size_t leaf_block_scalar_tail_tasks{0};
  std::size_t leaf_block_verify_checks{0};
  std::size_t leaf_block_verify_mismatches{0};
  double directness_ancestor_probe_ms{0.0};
  double selection_probe_ms{0.0};
  double selection_ms{0.0};
  double worklist_ms{0.0};
  double leaf_block_group_ms{0.0};
  double leaf_block_flush_ms{0.0};
  double leaf_block_writeback_ms{0.0};
  double target_avx2_ms{0.0};
  double soa_ms{0.0};
  double scalar_ms{0.0};
  double tail_ms{0.0};
  double ms{0.0};
};



enum class LeafWorkReason : std::uint8_t {
  MacTerminal = 0,
  PhysicalGap,
  Fragmentation,
  Convergence,
  ConvergenceEstimator,
  ConvergenceRho,
  ConvergenceAspect,
  SurfaceMargin,
  SurfaceMarginHard,
  SurfaceMarginSoft,
  K3RhoCap,
  SingularFallback,
  ScalarTail,
};



struct TraversalItem {
  std::size_t node_index{0};
  LeafWorkReason leaf_reason{LeafWorkReason::MacTerminal};
};



struct LeafWorkTally {
  std::size_t mac_terminal_panels{0};
  std::size_t physical_gap_panels{0};
  std::size_t fragmentation_panels{0};
  std::size_t convergence_panels{0};
  std::size_t convergence_estimator_panels{0};
  std::size_t convergence_rho_panels{0};
  std::size_t convergence_aspect_panels{0};
  std::size_t surface_margin_panels{0};
  std::size_t surface_margin_hard_panels{0};
  std::size_t surface_margin_soft_panels{0};
  std::size_t k3_rho_cap_panels{0};
  std::size_t singular_fallback_panels{0};
  std::size_t scalar_tail_panels{0};
};



struct DeferredLeafPairWorkItem {
  std::size_t target_index{0};
  LeafWorkReason leaf_reason{LeafWorkReason::MacTerminal};
};



struct DeferredLeafPairGroup {
  std::size_t target_leaf_index{std::numeric_limits<std::size_t>::max()};
  std::size_t source_node_index{std::numeric_limits<std::size_t>::max()};
  std::size_t source_first_panel{std::numeric_limits<std::size_t>::max()};
  std::size_t source_panel_count{0};
  std::vector<DeferredLeafPairWorkItem> items;
};



struct MacDecision {
  bool ratio_accept{false};
  bool accepted{false};
  bool unsafe{false};
  bool physical_gap_limited{false};
  bool convergence_rho_limited{false};
  bool convergence_aspect_limited{false};
  bool surface_margin_limited{false};
  bool surface_margin_applied{false};
  bool surface_margin_skipped{false};
  bool surface_margin_adaptive_limited{false};
  bool surface_margin_hard_applied{false};
  bool surface_margin_soft_applied{false};
  bool surface_margin_hard_limited{false};
  bool surface_margin_soft_limited{false};
  double rho{0.0};
  double aspect{0.0};
  double gap{0.0};
  double gap_ratio{0.0};
};



struct KernelSeriesBase {
  TaylorSeries ux;
  TaylorSeries uy;
  TaylorSeries uz;
  TaylorSeries r;
  TaylorSeries inv_r;
  TaylorSeries inv_r2;
  TaylorSeries inv_r3;
  TaylorSeries screened;
};



struct JcpAccum {
  double phi{0.0};
  double phin{0.0};
};



struct JcpChannelAccum {
  std::array<JcpAccum, kJcpExpansionChannelCount> channel{};
};



struct TreeUnknownCache {
  std::vector<double> phi;
  std::vector<double> phin;
  AlignedDoubleVector ordered_phi;
  AlignedDoubleVector ordered_phin;
};



struct MortonEntry {
  std::uint64_t code{0};
  std::size_t index{0};
};



struct TopDownTempNode {
  std::size_t first_panel{0};
  std::size_t panel_count{0};
  std::array<std::size_t, kBranchFactor> children{};
  std::size_t child_count{0};
  std::size_t depth{0};
};



struct AdaptivePruneParams {
  double smooth_theta_cap{0.24};
  double smooth_theta_factor{0.30};
  double p4_proxy_limit{0.0035};
};



struct JcpConvergenceNorms {
  bool valid{false};
  int taylor_order{0};
  double scalar_source_norm{0.0};
  double scalar_tail_norm_p{0.0};
  double scalar_tail_norm_pm1{0.0};
  double normal_source_norm{0.0};
  double normal_tail_norm_p{0.0};
  double normal_tail_norm_pm1{0.0};
  std::array<double, kJcpExpansionChannelCount> channel_source_norm{};
  std::array<double, kJcpExpansionChannelCount> channel_tail_norm_p{};
  std::array<double, kJcpExpansionChannelCount> channel_tail_norm_pm1{};
};

struct PackedJcpMoments {
  std::vector<NodeMoments> phin;
  std::array<std::vector<NodeMoments>, 3> phi_normal;
  std::vector<JcpConvergenceNorms> convergence_norms;
};



struct MomentAuditSummary {
  std::size_t nodes{0};
  double max_rel_error{0.0};
  double mean_rel_error{0.0};
  double translation_offset_max{0.0};
  double translation_offset_mean{0.0};
  double cancellation_ratio_max{0.0};
  double cancellation_ratio_mean{0.0};
  double dynamic_range_max{0.0};
};

std::size_t effective_workers(const TitanExecutor &executor,
                              std::size_t work_items) noexcept;
double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) noexcept;
double point_distance(double ax, double ay, double az, double bx, double by,
                      double bz) noexcept;
double tree_panel_max_edge_length(const Panel &panel) noexcept;
double node_aspect_ratio(const TreeNode &node) noexcept;

void record_far_geometric_audit(MacAuditTally &tally, const TreeNode &node,
                                double gap, double gap_ratio,
                                bool unsafe) noexcept;
void merge_far_geometric_audit(MacAuditTally &dst,
                               const MacAuditTally &src) noexcept;
void copy_far_geometric_audit_to_stats(TreecodeStats &stats,
                                       const MacAuditTally &tally) noexcept;
void record_fragmentation_audit(FragmentationTally &tally,
                                const TreeNode &node,
                                bool split_reject) noexcept;
void merge_fragmentation_audit(FragmentationTally &dst,
                               const FragmentationTally &src) noexcept;
void copy_fragmentation_audit_to_stats(TreecodeStats &stats,
                                       const FragmentationTally &tally) noexcept;
MacDecision evaluate_mac(const Panel &target, const TreeNode &node,
                         const TreeWorkspace &tree, double distance) noexcept;
std::string normalize_leaf_pair_kernel(const std::string &requested);
TreePanelCache build_tree_panel_cache(const std::vector<Panel> &panels,
                                      const std::vector<std::size_t> &order);
TreeUnknownCache build_tree_unknown_cache(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, const std::vector<std::size_t> &order);

const std::vector<MomentIndex> &moment_indices(int order);
double pow_int(double value, int power) noexcept;
TaylorSeries series_sub(const TaylorSeries &a, const TaylorSeries &b,
                        int order) noexcept;
TaylorSeries series_scale(const TaylorSeries &a, double scale,
                          int order) noexcept;
TaylorSeries green_series(double tx, double ty, double tz, double sx,
                          double sy, double sz, double kappa,
                          int order) noexcept;
TaylorSeries source_normal_component_green_series(
    double tx, double ty, double tz, double sx, double sy, double sz,
    int component, double kappa, int order) noexcept;
TaylorSeries target_normal_green_series(const Panel &target, double sx,
                                        double sy, double sz, double kappa,
                                        int order) noexcept;
TaylorSeries mixed_normal_component_green_series(
    const Panel &target, double sx, double sy, double sz, int component,
    double kappa, int order) noexcept;
double taylor_sum_series(const NodeMoments &moments, int order,
                         const TaylorSeries &series) noexcept;
double point_kernel(const Panel &target, double sx, double sy, double sz,
                    double kappa) noexcept;
double panel_kernel(const Panel &target, const Panel &source,
                    double kappa) noexcept;
double green_value(double tx, double ty, double tz, double sx, double sy,
                   double sz, double kappa) noexcept;
double source_normal_component_green(double tx, double ty, double tz,
                                     double sx, double sy, double sz,
                                     int component, double kappa) noexcept;
double target_normal_green(const Panel &target, double sx, double sy,
                           double sz, double kappa) noexcept;
double mixed_normal_component_green(const Panel &target, double sx, double sy,
                                    double sz, int component,
                                    double kappa) noexcept;
double direct_sum(const Panel &target, std::size_t target_index,
                  const std::vector<Panel> &panels,
                  const std::vector<double> &density, double kappa,
                  std::size_t begin, std::size_t end) noexcept;
double direct_ordered_sum(const Panel &target, std::size_t target_index,
                          const std::vector<Panel> &panels,
                          const std::vector<double> &density, double kappa,
                          const std::vector<std::size_t> &order,
                          std::size_t begin, std::size_t count,
                          std::size_t &near_count) noexcept;
double node_kernel_taylor(const Panel &target, const TreeNode &node,
                          const NodeMoments &moments, int order, double kappa,
                          const std::string &derivative_mode) noexcept;

std::size_t child_node_index(const TreeWorkspace &tree, const TreeNode &node,
                             std::size_t child) noexcept;
void push_child_nodes(const TreeWorkspace &tree, const TreeNode &node,
                      std::vector<std::size_t> &stack);
void push_child_nodes(const TreeWorkspace &tree, const TreeNode &node,
                      std::vector<TraversalItem> &stack,
                      LeafWorkReason reason);
bool is_singular_near_pair(const Panel &target, std::size_t target_index,
                           const Panel &source,
                           std::size_t source_index) noexcept;

std::vector<NodeMoments>
compute_node_moments(const std::vector<Panel> &panels,
                     const std::vector<double> &density,
                     const TreeWorkspace &tree, int taylor_order,
                     TitanExecutor &executor);
PackedJcpMoments compute_jcp_node_moments_packed(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, const TreeWorkspace &tree, int taylor_order,
    TitanExecutor &executor);
MomentAuditSummary audit_jcp_moment_translation(
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    bool coupled, const TreeWorkspace &tree, int taylor_order,
    const PackedJcpMoments &packed);

std::string resolve_leaf_panel_kernel(const std::string &requested);
std::string resolve_leaf_pair_batch_kernel(const std::string &requested,
                                           std::size_t target_batch_size);
std::string resolve_jcp_direct_batch_kernel(const TreeWorkspace &tree,
                                            const std::string &batch_kernel);
JcpAccum direct_jcp_ordered_sum(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, const std::string &kernel,
    std::size_t &near_count) noexcept;
void direct_jcp_target_batch_tile_sum(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    const std::string &kernel, std::vector<JcpAccum> &accum,
    std::size_t &near_count, std::size_t &vectorized_sources,
    std::size_t &vector_lanes, bool clean_source_span,
    std::size_t &soa_sources) noexcept;
bool packed_jcp_avx2_available() noexcept;
void direct_jcp_packed_target_source_span_avx2(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, std::size_t source_first,
    std::size_t source_count, std::vector<JcpAccum> &accum,
    std::size_t &near_count, std::size_t &vectorized_sources,
    std::size_t &vector_lanes, bool vector_exp = false) noexcept;
JcpAccum direct_jcp_ordered_tile_sum(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    const std::string &kernel, std::size_t &near_count,
    const std::vector<std::size_t> *exclusions = nullptr,
    std::size_t exclusion_begin = 0,
    std::size_t exclusion_count = 0) noexcept;
std::string tile_fallback_status(const std::string &kernel,
                                 std::size_t tile_nodes);
JcpAccum taylor_jcp_node_sum(const Panel &target, const TreeNode &node,
                             const NodeMoments &phin_scalar,
                             const std::array<NodeMoments, 3> &phi_normal,
                             double eps_ratio, double kappa, bool coupled,
                             int order,
                             const std::string &derivative_mode) noexcept;
JcpChannelAccum taylor_jcp_node_channel_sum(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order,
    const std::string &derivative_mode) noexcept;
JcpChannelAccum taylor_jcp_node_channel_degree_sum(
    const Panel &target, const TreeNode &node, const NodeMoments &phin_scalar,
    const std::array<NodeMoments, 3> &phi_normal, double eps_ratio,
    double kappa, bool coupled, int order, const std::string &derivative_mode,
    int moment_degree) noexcept;
JcpAccum sum_jcp_channels(const JcpChannelAccum &channels) noexcept;
JcpChannelAccum direct_jcp_ordered_channel_sum_scalar(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, std::size_t &near_count) noexcept;
ConvergenceEstimatorDecision evaluate_convergence_estimator(
    const TreeWorkspace &tree, const TreeNode &node,
    const PackedJcpMoments &packed, std::size_t node_index, int taylor_order,
    bool coupled, double rho, double eps_ratio) noexcept;
double jcp_rel_error(const JcpAccum &taylor, const JcpAccum &direct,
                     bool coupled) noexcept;
bool should_audit_far_expansion(const TreeWorkspace &tree,
                                std::size_t panel_count,
                                const TreeNode &node,
                                std::atomic_size_t &sampled_candidates) noexcept;
bool should_audit_far_expansion_mode(
    const TreeWorkspace &tree, const std::string &mode,
    std::size_t panel_count, const TreeNode &node,
    std::atomic_size_t &sampled_candidates) noexcept;
void record_far_expansion_audit(
    ExpansionAuditTally &tally, const JcpAccum &taylor,
    const JcpAccum &direct, bool coupled, std::size_t panel_count,
    double direct_ms, bool estimator_active, bool estimator_unstable,
    bool estimator_split_reject, double estimator_error_cap,
    double accepted_miss_absolute_floor,
    const std::string &channel_missed_semantics) noexcept;
void record_far_expansion_channel_audit(
    ExpansionAuditTally &tally, const JcpChannelAccum &taylor,
    const JcpChannelAccum &direct,
    const ConvergenceEstimatorDecision &estimator, double estimator_error_cap,
    double accepted_miss_absolute_floor,
    const std::string &channel_missed_semantics,
    const std::string &accepted_miss_diagnostics, std::size_t target_index,
    std::size_t source_node_index, std::size_t panel_count, double rho);
void record_split_far_expansion_channel_audit(
    ExpansionAuditTally &tally, const JcpAccum &taylor,
    const JcpAccum &direct, const JcpChannelAccum &taylor_channels,
    const JcpChannelAccum &direct_channels,
    const ConvergenceEstimatorDecision &estimator, bool coupled,
    double estimator_error_cap, double accepted_miss_absolute_floor) noexcept;
void record_convergence_overreject_audit(
    ExpansionAuditTally &tally, const JcpAccum &taylor,
    const JcpAccum &direct, const JcpChannelAccum &taylor_channels,
    const JcpChannelAccum &direct_channels,
    const ConvergenceEstimatorDecision &estimator, const TreeNode &node,
    bool coupled, double estimator_error_cap,
    double accepted_miss_absolute_floor, const std::string &detail_mode,
    const std::string &shape_mode, const std::string &kernel_response_mode,
    std::size_t target_index,
    std::size_t source_node_index, double rho, double target_distance,
    int taylor_order, const JcpChannelAccum &kernel_tail_p,
    const JcpChannelAccum &kernel_tail_pm1);
void merge_far_expansion_audit(ExpansionAuditTally &dst,
                               const ExpansionAuditTally &src) noexcept;
void copy_far_expansion_audit_to_stats(TreecodeStats &stats,
                                       const ExpansionAuditTally &tally) noexcept;
void write_accepted_miss_detail_csv(const ExpansionAuditTally &tally);
void write_overreject_calibration_detail_csv(const ExpansionAuditTally &tally);
void write_convergence_shape_calibration_csv(
    const ExpansionAuditTally &tally);
void write_kernel_response_calibration_csv(const ExpansionAuditTally &tally);
void add_leaf_work(LeafWorkTally &tally, LeafWorkReason reason,
                   std::size_t panels) noexcept;
std::size_t leaf_work_tally_total(const LeafWorkTally &tally) noexcept;
void record_convergence_estimator(ConvergenceEstimatorTally &tally,
                                  const TreeNode &node,
                                  const ConvergenceEstimatorDecision &decision)
    noexcept;
void merge_convergence_estimator(ConvergenceEstimatorTally &dst,
                                 const ConvergenceEstimatorTally &src) noexcept;
void copy_convergence_estimator_to_stats(
    TreecodeStats &stats, const ConvergenceEstimatorTally &tally) noexcept;
void record_far_convergence_audit(ConvergenceAuditTally &tally,
                                  const TreeNode &node, double rho) noexcept;
void merge_far_convergence_audit(ConvergenceAuditTally &dst,
                                 const ConvergenceAuditTally &src) noexcept;
void copy_far_convergence_audit_to_stats(
    TreecodeStats &stats, const ConvergenceAuditTally &tally) noexcept;
bool leaf_pair_batching_enabled(const TreeWorkspace &tree) noexcept;
std::size_t leaf_pair_skip_key(const TreeWorkspace &tree,
                               std::size_t target_index,
                               std::size_t node_index) noexcept;
bool leaf_pair_batch_clean_for_target(const std::vector<Panel> &panels,
                                      const TreeWorkspace &tree,
                                      const TreeNode &node,
                                      std::size_t target_index) noexcept;
bool would_visit_leaf_direct_for_target(
    const std::vector<Panel> &panels, const TreeWorkspace &tree,
    const std::vector<std::size_t> &rank, const PackedJcpMoments &jcp_moments,
    std::size_t target_index, std::size_t source_node_index, int taylor_order,
    bool coupled, double eps_ratio, std::vector<std::size_t> &stack,
    std::size_t *node_visits) noexcept;
bool would_visit_leaf_direct_for_target_ancestor_path(
    const std::vector<Panel> &panels, const TreeWorkspace &tree,
    const std::vector<std::size_t> &rank, const PackedJcpMoments &jcp_moments,
    std::size_t target_index, std::size_t source_node_index, int taylor_order,
    bool coupled, double eps_ratio,
    std::size_t *node_visits) noexcept;




// Stage 3AF internal implementation entrypoints.
TreeWorkspace build_tree_workspace_impl(const std::vector<Panel> &panels,
                                   TitanExecutor &executor,
                                   std::size_t leaf_size,
                                   double opening_theta,
                                   const std::string &leaf_near_policy,
                                   const std::string &leaf_panel_kernel,
                                   const std::string &adaptive_level,
                                   const std::string &quad_buffer,
                                   const std::string &interaction_mode,
                                   const std::string &node_bounds_mode,
                                   const std::string &tree_builder,
                                   const std::string &mac_safety,
                                   double mac_gap_factor,
                                   const std::string &moment_audit,
                                   const std::string &expansion_center,
                                   const std::string &fragmentation_safety,
                                   const std::string &expansion_audit,
                                   int expansion_audit_sample_count,
                                   int expansion_audit_max_panels,
                                   const std::string &moment_construction,
                                   const std::string &moment_sum,
                                   const std::string &convergence_audit,
                                   const std::string &convergence_safety,
                                   double convergence_rho_cap,
                                   double mac_aspect_cap,
                                   const std::string &convergence_estimator,
                                   double convergence_tail_threshold,
                                   double convergence_estimator_error_cap,
                                   double convergence_unstable_rho_cap,
                                   const std::string
                                       &convergence_calibration,
                                   const std::string &cluster_policy,
                                   const std::string
                                       &convergence_dielectric_amp,
                                   int target_batch_size,
                                   int leaf_pair_min_source_panels,
                                   int leaf_pair_min_targets,
                                   const std::string
                                       &convergence_estimator_model,
                                   const std::string
                                       &expansion_audit_channel,
                                   const std::string
                                       &convergence_channel_calibration,
                                   const std::string
                                       &channel_missed_semantics,
                                   const std::string
                                       &convergence_split_policy,
                                   const std::string
                                       &accepted_miss_diagnostics,
                                   double convergence_k3_accepted_rho_cap,
                                   const std::string
                                       &mac_distance_policy,
                                   double mac_surface_margin_factor,
                                   const std::string
                                       &mac_surface_margin_policy,
                                   double mac_surface_margin_aspect_threshold,
                                   double mac_surface_margin_rho_trigger,
                                   double accepted_miss_absolute_floor,
                                   const std::string &leaf_pair_kernel,
                                   const std::string &leaf_kernel_profile,
                                   const std::string &leaf_work_profile,
                                   const std::string &jcp_direct_kernel,
                                   const std::string
                                       &leaf_pair_selection_profile,
                                   const std::string &leaf_pair_probe_cache,
                                   const std::string &leaf_pair_target_group,
                                   int leaf_pair_max_lookahead,
                                   const std::string
                                       &leaf_pair_directness_probe,
                                   const std::string
                                       &leaf_pair_directness_cache,
                                   const std::string
                                       &leaf_pair_directness_verify,
                                   const std::string &leaf_pair_scheduler,
                                   int leaf_pair_worklist_max_active,
                                   const std::string
                                       &leaf_pair_worklist_verify,
                                   const std::string &leaf_block_verify,
                                   double mac_surface_margin_soft_aspect_threshold,
                                   double mac_surface_margin_soft_rho_trigger,
                                   double mac_surface_margin_soft_factor,
                                   const std::string
                                       &convergence_split_selectivity,
                                   double convergence_split_absolute_floor,
                                   const std::string
                                       &convergence_overreject_audit,
                                   const std::string
                                       &convergence_overreject_detail,
                                   const std::string
                                       &convergence_shape_audit,
                                   const std::string
                                       &convergence_kernel_response_audit,
                                   const std::string &leaf_mac,
                                   const std::string &fmm_m2l);
void apply_direct_operator_impl(const std::vector<Panel> &panels,
                           const std::vector<double> &density,
                           double contrast, double kappa,
                           TitanExecutor &executor,
                           std::vector<double> &out,
                           TreecodeStats *stats);
void apply_treecode_operator_impl(const std::vector<Panel> &panels,
                             const std::vector<double> &density,
                             double contrast, double kappa,
                             const TreeWorkspace &tree,
                             TitanExecutor &executor,
                             std::vector<double> &out,
                             TreecodeStats *stats);
void apply_jcp_treecode_operator_impl(const std::vector<Panel> &panels,
                                 const std::vector<double> &unknowns,
                                 double eps_ratio, double kappa,
                                 double lhs_phi, double lhs_phin,
                                 bool coupled, const TreeWorkspace &tree,
                                 const std::string &near_integration,
                                 const std::string &quad_buffer,
                                 TitanExecutor &executor,
                                 std::vector<double> &out,
                                 TreecodeStats *stats);

struct ConvergenceSplitSelectivityInput {
  bool unstable{false};
  bool safety_split{false};
  double rho{0.0};
  double unstable_rho_cap{0.25};
  std::string_view split_selectivity{"legacy"};
  double split_absolute_floor{0.0};
  std::string_view split_policy{"global"};
  std::string_view estimator_model{"stage3l"};
  double estimated_abs_error{0.0};
  std::array<bool, kJcpExpansionChannelCount> channel_unstable{};
  std::array<double, kJcpExpansionChannelCount> channel_estimated_abs_error{};
};

struct ConvergenceSplitSelectivityResult {
  bool split_reject{false};
  bool low_abs_reaccepted{false};
  std::array<bool, kJcpExpansionChannelCount> channel_low_abs_reaccepted{};
};

struct ConvergenceOverrejectAuditDecision {
  bool caught_high_error{false};
  bool low_error_overreject{false};
};

inline ConvergenceSplitSelectivityResult evaluate_convergence_split_selectivity(
    const ConvergenceSplitSelectivityInput &input) noexcept {
  ConvergenceSplitSelectivityResult result;
  const bool base_split =
      input.unstable && input.safety_split &&
      std::isfinite(input.rho) && input.rho > input.unstable_rho_cap;
  if (!base_split)
    return result;

  const bool k3_hot_policy =
      input.split_policy == "k3-hot" &&
      input.estimator_model == "channel-tail-eps";
  if (k3_hot_policy && !input.channel_unstable[kJcpK3PhinChannel])
    return result;

  const double floor =
      std::isfinite(input.split_absolute_floor)
          ? std::max(0.0, input.split_absolute_floor)
          : 0.0;
  const bool absolute_mode =
      input.split_selectivity == "absolute-aware" ||
      input.split_selectivity == "channel-absolute-aware";
  if (!absolute_mode || floor <= 0.0) {
    result.split_reject = true;
    return result;
  }

  if (input.split_selectivity == "channel-absolute-aware" &&
      input.estimator_model == "channel-tail-eps") {
    bool channel_should_split = false;
    for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
         ++channel) {
      if (k3_hot_policy && channel != kJcpK3PhinChannel)
        continue;
      if (!input.channel_unstable[channel])
        continue;
      const double abs_error = input.channel_estimated_abs_error[channel];
      if (std::isfinite(abs_error) && abs_error > floor) {
        channel_should_split = true;
      } else {
        result.channel_low_abs_reaccepted[channel] = true;
      }
    }
    result.split_reject = channel_should_split;
  } else {
    result.split_reject = std::isfinite(input.estimated_abs_error) &&
                          input.estimated_abs_error > floor;
    if (!result.split_reject) {
      for (std::size_t channel = 0; channel < kJcpExpansionChannelCount;
           ++channel) {
        if (input.channel_unstable[channel])
          result.channel_low_abs_reaccepted[channel] = true;
      }
    }
  }

  result.low_abs_reaccepted = !result.split_reject;
  return result;
}

inline ConvergenceOverrejectAuditDecision
classify_convergence_overreject_audit(
    bool split_reject, double relative_error, double absolute_error,
    double relative_error_cap, double absolute_floor) noexcept {
  ConvergenceOverrejectAuditDecision result;
  if (!split_reject || relative_error_cap <= 0.0)
    return result;
  const bool high_relative =
      std::isfinite(relative_error) && relative_error > relative_error_cap;
  const double floor =
      std::isfinite(absolute_floor) ? std::max(0.0, absolute_floor) : 0.0;
  const bool high_absolute =
      std::isfinite(absolute_error) && absolute_error > floor;
  result.caught_high_error = high_relative && high_absolute;
  result.low_error_overreject = !result.caught_high_error;
  return result;
}


} // namespace TitanPBE::Tabi::detail

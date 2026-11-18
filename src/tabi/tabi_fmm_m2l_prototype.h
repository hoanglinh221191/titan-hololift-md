#ifndef TITAN_TABI_FMM_M2L_PROTOTYPE_H
#define TITAN_TABI_FMM_M2L_PROTOTYPE_H

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanPBE::Tabi {
struct Panel;
}

namespace titan::tabi::fmm_m2l {

enum class KernelKind {
  Laplace,
  Yukawa,
};

struct PrototypeConfig {
  KernelKind kernel = KernelKind::Laplace;
  double kappa = 0.0;
  int max_leaf_size = 64;
  int expansion_order = 4;
  double separation_theta = 0.6;
  int threads = 1;
};

using FmmM2lConfig = PrototypeConfig;

struct PrototypeMetrics {
  std::size_t sources = 0;
  std::size_t targets = 0;
  std::size_t cells = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  std::size_t m2l_operators = 0;
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  double direct_seconds = 0.0;
  double rms_abs_error = 0.0;
  double max_abs_error = 0.0;
  double rel_rms_error = 0.0;
  bool finite_outputs = true;
};

struct JcpChannelPrototypeMetrics {
  std::size_t panels = 0;
  std::size_t channels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  double direct_seconds = 0.0;
  double aggregate_rel_rms_error = 0.0;
  double max_channel_rel_rms_error = 0.0;
  double max_abs_error = 0.0;
  bool finite_outputs = true;
};

struct JcpK1BridgeMetrics {
  std::size_t panels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  double direct_seconds = 0.0;
  double rel_rms_error = 0.0;
  double rms_abs_error = 0.0;
  double max_abs_error = 0.0;
  bool finite_outputs = true;
};

struct JcpDerivativeBridgeMetrics {
  std::size_t panels = 0;
  std::size_t channels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  double direct_seconds = 0.0;
  double aggregate_rel_rms_error = 0.0;
  double max_channel_rel_rms_error = 0.0;
  double k2_rel_rms_error = 0.0;
  double k3_rel_rms_error = 0.0;
  double k4_rel_rms_error = 0.0;
  double k2_reference_rms = 0.0;
  double k3_reference_rms = 0.0;
  double k4_reference_rms = 0.0;
  double max_abs_error = 0.0;
  bool finite_outputs = true;
};

struct JcpDirectParityMetrics {
  std::size_t panels = 0;
  double phi_rel_rms_error = 0.0;
  double phin_rel_rms_error = 0.0;
  double aggregate_rel_rms_error = 0.0;
  double max_abs_error = 0.0;
  bool finite_outputs = true;
};

struct JcpOfflineOperatorResult {
  std::size_t panels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  std::vector<double> phi;
  std::vector<double> phin;
  bool finite_outputs = true;
};

struct JcpAnalyticOperatorResult {
  std::size_t panels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  std::size_t near_self_pairs = 0;
  std::size_t near_one_ring_pairs = 0;
  std::size_t near_duffy_pairs = 0;
  std::size_t near_centroid_pairs = 0;
  std::size_t near_fallbacks = 0;
  std::size_t duffy_subdivisions_max = 0;
  double duffy_subdivisions_mean = 0.0;
  std::string near_kernel_policy = "centroid";
  double setup_seconds = 0.0;
  double apply_seconds = 0.0;
  std::size_t context_nodes = 0;
  std::size_t context_work_items = 0;
  std::vector<double> phi;
  std::vector<double> phin;
  bool finite_outputs = true;
};

struct JcpOperatorAttributionRecord {
  std::string probe;
  std::size_t target_panel{0};
  std::string region;
  int source_node{-1};
  int source_depth{-1};
  std::size_t source_panels{0};
  double distance{0.0};
  double source_radius{0.0};
  double mac_ratio{0.0};
  double expansion_distance{0.0};
  double target_expansion_radius{0.0};
  double source_expansion_radius{0.0};
  double target_center_offset{0.0};
  double source_center_offset{0.0};
  double expansion_mac_ratio{0.0};
  std::array<double, 4> approximate{};
  std::array<double, 4> direct{};
};

struct JcpOperatorAttributionResult {
  std::string approximation_mode{"legacy_degree1"};
  std::string probe;
  std::vector<std::size_t> sampled_targets;
  std::vector<JcpOperatorAttributionRecord> records;
  int configured_expansion_order{0};
  int implemented_far_moment_degree{1};
  std::size_t near_records{0};
  std::size_t far_records{0};
  double recomposition_rel_error{0.0};
  double moment_build_seconds{0.0};
  double far_evaluation_seconds{0.0};
  double m2l_seconds{0.0};
  double l2p_seconds{0.0};
  std::size_t local_target_leaves{0};
  std::size_t local_m2l_pairs{0};
  int local_source_order{4};
  int local_target_order{4};
  int local_kernel_derivative_order{8};
  std::string local_refine_mode{"off"};
  double local_refine_tolerance{4.0e-3};
  double local_tail_ratio_before{0.0};
  double local_tail_ratio_after{0.0};
  std::size_t local_refined_cells{0};
  std::size_t local_split_children{0};
  std::size_t local_direct_fallback_panels{0};
  std::size_t local_direct_fallback_pairs{0};
  std::size_t local_far_pairs_before{0};
  std::size_t local_far_pairs_after{0};
  double local_refinement_seconds{0.0};
  std::uint64_t local_plan_hash{0};
  std::size_t local_plan_build_count{0};
  std::size_t local_static_probe_count{0};
  std::string local_refine_probe_mode{"rademacher"};
  bool local_solution_probe_used{false};
  double local_linf_max_abs_tail{0.0};
  double local_linf_max_relative_tail{0.0};
  std::size_t local_linf_outlier_panels{0};
  std::string local_geometry_guard{"off"};
  double local_geometry_theta{0.0};
  std::size_t local_geometry_guard_rejected_cells{0};
  std::size_t local_geometry_guard_direct_fallback_pairs{0};
  double local_geometry_gap_min{0.0};
  double local_geometry_ratio_max{0.0};
  std::string local_cache_mode{"off"};
  std::string local_cache_layout{"off"};
  std::size_t local_cache_operators{0};
  std::size_t local_cache_bytes{0};
  std::size_t local_cache_operator_bytes{0};
  std::size_t local_cache_uncompressed_estimated_bytes{0};
  double local_cache_compression_ratio{1.0};
  std::size_t local_cache_far_pair_offsets{0};
  std::size_t local_cache_build_count{0};
  std::size_t local_cache_hits{0};
  std::size_t local_cache_tile_bytes{0};
  std::size_t local_cache_tile_count{0};
  std::size_t local_cache_tile_builds{0};
  double local_cache_tile_seconds{0.0};
  double local_plan_seconds{0.0};
  std::string local_plan_builder{"legacy-rescan"};
  double local_plan_probe_moment_seconds{0.0};
  double local_plan_pair_metric_seconds{0.0};
  double local_plan_refinement_seconds{0.0};
  std::size_t local_plan_pair_metric_evaluations{0};
  std::size_t local_plan_pair_metric_cache_hits{0};
  std::size_t local_plan_refinement_iterations{0};
  std::size_t local_plan_metric_cache_peak_bytes{0};
  std::size_t local_plan_parallel_target_leaves{0};
  double local_cache_seconds{0.0};
  double local_candidate_apply_seconds{0.0};
  std::size_t local_candidate_repeat_count{1};
  double local_candidate_median_apply_seconds{0.0};
  double local_candidate_direct_reference_seconds{0.0};
  std::string local_fallback_requested{"scalar"};
  std::string local_fallback_effective{"scalar"};
  std::uint64_t local_fallback_worklist_hash{0};
  std::size_t local_fallback_worklist_build_count{0};
  std::size_t local_fallback_tasks{0};
  std::size_t local_fallback_metadata_bytes{0};
  std::size_t local_fallback_base_near_pairs{0};
  std::size_t local_fallback_refined_pairs{0};
  std::size_t local_fallback_vectorized_sources{0};
  std::size_t local_fallback_vectorized_lanes{0};
  std::size_t local_fallback_scalar_self_pairs{0};
  double local_fallback_seconds{0.0};
  bool finite_outputs{true};
};

struct JcpLocalP4CandidateResult {
  std::vector<double> phi;
  std::vector<double> phin;
  std::string refine_mode{"off"};
  double refine_tolerance{2.0e-3};
  std::string cache_mode{"off"};
  std::string cache_layout{"off"};
  std::string status{"ok"};
  std::uint64_t plan_hash{0};
  std::size_t plan_build_count{0};
  std::size_t static_probe_count{0};
  std::string refine_probe_mode{"rademacher"};
  bool solution_probe_used{false};
  double linf_max_abs_tail{0.0};
  double linf_max_relative_tail{0.0};
  std::size_t linf_outlier_panels{0};
  std::string geometry_guard{"off"};
  double geometry_theta{0.0};
  std::size_t geometry_guard_rejected_cells{0};
  std::size_t geometry_guard_direct_fallback_pairs{0};
  double geometry_gap_min{0.0};
  double geometry_ratio_max{0.0};
  std::size_t refined_cells{0};
  std::size_t split_children{0};
  std::size_t direct_fallback_panels{0};
  std::size_t direct_fallback_pairs{0};
  std::size_t far_pairs{0};
  std::size_t far_panel_pairs{0};
  std::size_t cache_operator_count{0};
  std::size_t cache_bytes{0};
  std::size_t cache_operator_bytes{0};
  std::size_t cache_uncompressed_estimated_bytes{0};
  double cache_compression_ratio{1.0};
  std::size_t cache_far_pair_offsets{0};
  std::size_t cache_build_count{0};
  std::size_t cache_hits{0};
  std::size_t cache_tile_bytes{0};
  std::size_t cache_tile_count{0};
  std::size_t cache_tile_builds{0};
  double cache_tile_seconds{0.0};
  double plan_seconds{0.0};
  std::string plan_builder{"legacy-rescan"};
  double plan_probe_moment_seconds{0.0};
  double plan_pair_metric_seconds{0.0};
  double plan_refinement_seconds{0.0};
  std::size_t plan_pair_metric_evaluations{0};
  std::size_t plan_pair_metric_cache_hits{0};
  std::size_t plan_refinement_iterations{0};
  std::size_t plan_metric_cache_peak_bytes{0};
  std::size_t plan_parallel_target_leaves{0};
  double cache_seconds{0.0};
  double moment_seconds{0.0};
  double m2l_seconds{0.0};
  double direct_fallback_seconds{0.0};
  double l2p_seconds{0.0};
  double apply_seconds{0.0};
  int local_source_order{4};
  int local_target_order{4};
  int local_kernel_derivative_order{8};
  std::string fallback_requested{"scalar"};
  std::string fallback_effective{"scalar"};
  std::uint64_t fallback_worklist_hash{0};
  std::size_t fallback_worklist_build_count{0};
  std::size_t fallback_tasks{0};
  std::size_t fallback_metadata_bytes{0};
  std::size_t fallback_base_near_pairs{0};
  std::size_t fallback_refined_pairs{0};
  std::size_t fallback_vectorized_sources{0};
  std::size_t fallback_vectorized_lanes{0};
  std::size_t fallback_scalar_self_pairs{0};
  double recomposition_rel_error{0.0};
  bool finite_outputs{true};
};

using JcpLocalCartesianCandidateResult = JcpLocalP4CandidateResult;

struct JcpLocalP4OperatorOptions {
  std::string near_integration{"analytic_singular"};
  std::string quad_buffer{"off"};
  std::string refine_mode{"operator-tail-static"};
  double refine_tolerance{2.0e-3};
  std::string cache_mode{"green-jet-compressed"};
  std::size_t cache_budget_mb{256};
  std::string fallback_mode{"packed-avx2"};
  std::string plan_builder{"metric-cache"};
  std::size_t refine_probe_count{4};
  std::string refine_probe_mode{"rademacher"};
  std::string geometry_guard{"off"};
  double geometry_theta{0.0};
};

using JcpLocalCartesianOperatorOptions = JcpLocalP4OperatorOptions;

void assemble_jcp_local_p4_system_matvec(
    const JcpLocalP4CandidateResult &interaction,
    const std::vector<double> &unknowns, double lhs_phi, double lhs_phin,
    bool coupled, std::vector<double> &out);

struct JcpGuardedGmresMetrics {
  std::size_t panels = 0;
  std::size_t near_pairs = 0;
  std::size_t far_pairs = 0;
  int direct_iterations = 0;
  int adapter_iterations = 0;
  double direct_initial_residual = 0.0;
  double direct_final_residual = 0.0;
  double adapter_initial_residual = 0.0;
  double adapter_final_residual = 0.0;
  double direct_true_residual = 0.0;
  double solution_rel_error = 0.0;
  double direct_matvec_ms = 0.0;
  double adapter_matvec_ms = 0.0;
  double speedup_vs_direct_matvec = 0.0;
  double adapter_setup_ms = 0.0;
  double adapter_apply_ms = 0.0;
  bool direct_converged = false;
  bool adapter_converged = false;
  bool adapter_residual_decreased = false;
  bool finite_outputs = true;
};

struct PrototypePointCloud {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> q;
  std::vector<std::size_t> panel_id;
};

struct PrototypePanel {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double nx = 0.0;
  double ny = 0.0;
  double nz = 1.0;
  double area = 1.0;
  double scalar_density = 1.0;
  double source_x = 0.0;
  double source_y = 0.0;
  double source_z = 0.0;
  // Mean normal of a node patch, which goes with its centroid (Panel).
  double source_nx = 0.0;
  double source_ny = 0.0;
  double source_nz = 1.0;
  bool has_source_quadrature_center = false;
};

// Release-facing aliases. The historical Prototype* names remain available for
// source compatibility with archived benchmark probes and reports.
using SurfacePanelView = PrototypePanel;
using JcpPanel = PrototypePanel;

struct FmmM2lOverlapPanelGroup {
  std::vector<std::size_t> owned_panels;
  std::vector<std::size_t> factor_panels;
  std::size_t halo_count{0};
};

class FmmM2lAnalyticContext {
public:
  FmmM2lAnalyticContext(const PrototypeConfig &config,
                        std::vector<PrototypePanel> panels, double eps_ratio,
                        double kappa);
  ~FmmM2lAnalyticContext();

  FmmM2lAnalyticContext(FmmM2lAnalyticContext &&) noexcept;
  FmmM2lAnalyticContext &
  operator=(FmmM2lAnalyticContext &&) noexcept;
  FmmM2lAnalyticContext(const FmmM2lAnalyticContext &) = delete;
  FmmM2lAnalyticContext &
  operator=(const FmmM2lAnalyticContext &) = delete;

  [[nodiscard]] bool matches(const PrototypeConfig &config,
                             std::size_t panel_count, double eps_ratio,
                             double kappa) const noexcept;
  [[nodiscard]] std::size_t panel_count() const noexcept;
  [[nodiscard]] std::size_t context_nodes() const noexcept;
  [[nodiscard]] std::size_t context_work_items() const noexcept;
  [[nodiscard]] double setup_seconds() const noexcept;
  [[nodiscard]] std::vector<std::vector<std::size_t>>
  block_panel_ids(std::size_t max_block_size) const;
  [[nodiscard]] std::vector<FmmM2lOverlapPanelGroup>
  overlap_panel_groups(std::size_t owned_block_size,
                       std::size_t overlap_panel_cap) const;
  [[nodiscard]] std::vector<FmmM2lOverlapPanelGroup>
  distance_knn_overlap_panel_groups(std::size_t owned_block_size,
                                    std::size_t overlap_panel_cap) const;
  // executor (optional): the rows are built in parallel, each as before.
  [[nodiscard]] std::vector<std::vector<std::size_t>>
  near_panel_graph(std::size_t neighbor_cap,
                   TitanExecutor *executor = nullptr) const;
  [[nodiscard]] std::vector<std::vector<std::size_t>>
  distance_knn_panel_graph(std::size_t neighbor_cap) const;

  JcpAnalyticOperatorResult apply(const std::vector<double> &phi,
                                  const std::vector<double> &phin,
                                  TitanExecutor &executor,
                                  bool coupled = true,
                                  const std::string &near_integration =
                                      "centroid",
                                  const std::string &quad_buffer = "off",
                                  const std::vector<TitanPBE::Tabi::Panel>
                                      *production_panels = nullptr) const;

  JcpAnalyticOperatorResult apply_into(
      const std::vector<double> &phi, const std::vector<double> &phin,
      TitanExecutor &executor, std::vector<double> &phi_out,
      std::vector<double> &phin_out, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr)
      const;

  JcpOperatorAttributionResult attribute(
      const std::vector<double> &phi, const std::vector<double> &phin,
      std::size_t sample_count, const std::string &probe,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr)
      const;

  JcpOperatorAttributionResult attribute_cartesian(
      const std::vector<double> &phi, const std::vector<double> &phin,
      std::size_t sample_count, const std::string &probe,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr)
      const;

  JcpOperatorAttributionResult attribute_local_p4(
      const std::vector<double> &phi, const std::vector<double> &phin,
      std::size_t sample_count, const std::string &probe,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr,
      const std::string &refine_mode = "off",
      double refine_tolerance = 4.0e-3)
      const;

  JcpLocalCartesianCandidateResult apply_local_cartesian_candidate(
      const std::vector<double> &phi, const std::vector<double> &phin,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr,
      const std::string &refine_mode = "operator-tail-static",
      double refine_tolerance = 2.0e-3,
      const std::string &cache_mode = "kernel-jet",
      std::size_t cache_budget_mb = 256,
      const std::string &fallback_mode = "scalar",
      const std::string &plan_builder = "legacy-rescan",
      std::size_t refine_probe_count = 4,
      const std::string &refine_probe_mode = "rademacher",
      const std::string &geometry_guard = "off",
      double geometry_theta = 0.0,
      const std::vector<std::size_t> *sampled_target_indices = nullptr) const;

  JcpLocalP4CandidateResult apply_local_p4_candidate(
      const std::vector<double> &phi, const std::vector<double> &phin,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr,
      const std::string &refine_mode = "operator-tail-static",
      double refine_tolerance = 2.0e-3,
      const std::string &cache_mode = "kernel-jet",
      std::size_t cache_budget_mb = 256,
      const std::string &fallback_mode = "scalar",
      const std::string &plan_builder = "legacy-rescan",
      std::size_t refine_probe_count = 4,
      const std::string &refine_probe_mode = "rademacher",
      const std::string &geometry_guard = "off",
      double geometry_theta = 0.0,
      const std::vector<std::size_t> *sampled_target_indices = nullptr) const;

  JcpLocalP4CandidateResult prepare_local_p4_operator(
      const std::vector<TitanPBE::Tabi::Panel> &production_panels,
      TitanExecutor &executor, const JcpLocalP4OperatorOptions &options);

  JcpLocalP4CandidateResult apply_prepared_local_p4_operator(
      const std::vector<double> &phi, const std::vector<double> &phin,
      TitanExecutor &executor, bool coupled = true) const;

  JcpOperatorAttributionResult attribute_local_p4_candidate(
      const std::vector<double> &phi, const std::vector<double> &phin,
      std::size_t sample_count, const std::string &probe,
      TitanExecutor &executor, bool coupled = true,
      const std::string &near_integration = "centroid",
      const std::string &quad_buffer = "off",
      const std::vector<TitanPBE::Tabi::Panel> *production_panels = nullptr,
      const std::string &refine_mode = "operator-tail-static",
      double refine_tolerance = 2.0e-3,
      const std::string &cache_mode = "kernel-jet",
      std::size_t cache_budget_mb = 256,
      std::size_t repeat_count = 1,
      const std::string &fallback_mode = "scalar",
      const std::string &plan_builder = "legacy-rescan",
      std::size_t refine_probe_count = 4,
      const std::string &refine_probe_mode = "rademacher",
      const std::string &geometry_guard = "off",
      double geometry_theta = 0.0) const;

private:
  JcpOperatorAttributionResult attribute_impl(
      const std::vector<double> &phi, const std::vector<double> &phin,
      std::size_t sample_count, const std::string &probe,
      TitanExecutor &executor, bool coupled,
      const std::string &near_integration, const std::string &quad_buffer,
      const std::vector<TitanPBE::Tabi::Panel> *production_panels,
      int approximation_mode, const std::string &refine_mode,
      double refine_tolerance) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using FmmM2lContext = FmmM2lAnalyticContext;

const char *kernel_kind_name(KernelKind kernel) noexcept;

PrototypeMetrics run_synthetic_probe(const PrototypeConfig &config,
                                     std::size_t panel_count, unsigned seed);

PrototypePointCloud make_mini_protein_like_point_cloud(std::size_t panel_count,
                                                       unsigned seed);

std::vector<PrototypePanel>
make_tabi_panel_like_fixture(std::size_t panel_count, unsigned seed);

PrototypePointCloud point_cloud_from_panels(const std::vector<PrototypePanel> &panels);

PrototypeMetrics run_point_cloud_probe(const PrototypeConfig &config,
                                       const PrototypePointCloud &points);

JcpChannelPrototypeMetrics
run_jcp_channel_probe(const PrototypeConfig &config,
                      const std::vector<PrototypePanel> &panels);

std::string classify_scope_d_decision(const JcpChannelPrototypeMetrics &metrics);

JcpK1BridgeMetrics
run_jcp_k1_bridge_probe(const PrototypeConfig &config,
                        const std::vector<PrototypePanel> &panels);

std::string classify_scope_e_decision(const JcpK1BridgeMetrics &metrics);

JcpDerivativeBridgeMetrics
run_jcp_derivative_bridge_probe(const PrototypeConfig &config,
                                const std::vector<PrototypePanel> &panels);

std::string
classify_scope_f_decision(const JcpDerivativeBridgeMetrics &metrics);

JcpDirectParityMetrics compare_jcp_direct_reference(
    const std::vector<PrototypePanel> &panels,
    const std::vector<double> &production_phi,
    const std::vector<double> &production_phin, double eps_ratio,
    double kappa);

JcpOfflineOperatorResult evaluate_jcp_offline_operator(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    double eps_ratio, double kappa);

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    double eps_ratio, double kappa);

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa);

JcpAnalyticOperatorResult evaluate_jcp_offline_operator_analytic(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    const std::vector<double> &phi, const std::vector<double> &phin,
    double eps_ratio, double kappa, TitanExecutor &executor);

JcpGuardedGmresMetrics run_jcp_fmm_m2l_guarded_gmres_probe(
    const PrototypeConfig &config, const std::vector<PrototypePanel> &panels,
    double eps_ratio, double kappa, int restart, int max_iterations,
    double tolerance);

std::string classify_scope_a_decision(const PrototypeMetrics &metrics,
                                      const PrototypeConfig &config);

} // namespace titan::tabi::fmm_m2l

#endif // TITAN_TABI_FMM_M2L_PROTOTYPE_H

#pragma once

// Internals of the post-loop analyses (src/postproc/): the helpers that
// engine_postproc.cpp kept in an anonymous namespace, now in
// postproc_detail and shared by the src/postproc/*.cpp files. Only those
// files include this header.

#include "titan_dham_segments.h"
#include "postproc/engine_postproc.h"
#include "titan_alignment.h"
#include "titan_corr_knn.h"
#include "titan_executor.h"
#include "titan_mbar_weights.h"
#include "titan_sparse_graph_module.h"
#include "titan_ui.h"
#include <Eigen/Dense>
#include <algorithm>
#include <mutex>
#include <thread>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <cstdint>
#include <unordered_map>
#include <type_traits>
#include <array>

// Clang 21 COMDAT workaround: force emission of Matrix destructor
namespace {
[[maybe_unused]] void _eigen_force_dtor() {
  Eigen::MatrixXd m;
  (void)m;
  Eigen::VectorXd v;
  (void)v;
}
} // namespace

namespace postproc_detail {

constexpr double kBarNm3ToKJPerMol = 0.06022140857;
constexpr double kEnergyTimeFilterTolerancePs = 1e-3;
constexpr int kLjFeatureColumns = 20;

struct LjMicrostateSummary {
  bool valid{false};
  std::vector<double> rho_bulk_nm3;
  std::vector<double> cluster_frac;
  std::vector<double> dense_frac;
  std::vector<double> q6_bar;
  std::vector<double> solid_like_frac;
  std::vector<double> fcc_frac;
  std::vector<double> hcp_frac;
};

struct ProteinMicrostateSummary {
  bool valid_rmsd{false};
  bool valid_dssp{false};
  std::vector<double> rmsd_mean;
  std::vector<double> helix_frac;
  std::vector<double> strand_frac;
  std::vector<double> coil_frac;
  std::vector<double> turn_frac;
  std::vector<double> bend_frac;
};

struct QNativeMicrostateSummary {
  bool valid{false};
  std::vector<double> mean_q;
};

enum class TptBasinLabel : unsigned char { Intermediate = 0, Source = 1, Sink = 2 };

using TptKineticDiagnostics = TitanValidation::TptKineticDiagnostics;

using TptBootstrapAggregate = TitanValidation::TptBootstrapAggregate;

struct DhamWindowAuditRow {
  int replica_id{-1};
  bool traj_present{false};
  bool energy_present{false};
  bool overlap_found{false};
  bool truncated_start{false};
  bool truncated_end{false};
  bool irregular_energy_dt{false};
  bool sparse_energy{false};
  std::size_t traj_frames{0};
  std::size_t energy_points{0};
  std::size_t kept_frames{0};
  std::size_t dropped_outside_overlap{0};
  std::size_t unmatched_in_overlap{0};
  double traj_min_ps{std::numeric_limits<double>::quiet_NaN()};
  double traj_max_ps{std::numeric_limits<double>::quiet_NaN()};
  double energy_min_ps{std::numeric_limits<double>::quiet_NaN()};
  double energy_max_ps{std::numeric_limits<double>::quiet_NaN()};
  double overlap_start_ps{std::numeric_limits<double>::quiet_NaN()};
  double overlap_end_ps{std::numeric_limits<double>::quiet_NaN()};
  double inferred_dt_energy_ps{std::numeric_limits<double>::quiet_NaN()};
  // KIN-21: energy grid coarser than the trajectory grid (one frame per
  // energy row carries the energy, the others are excluded from state means)
  bool energy_coarser_than_traj{false};
  std::size_t duplicate_energy_matches{0};
  double inferred_dt_traj_ps{std::numeric_limits<double>::quiet_NaN()};
  std::string status{"uninitialized"};
};

struct MappedEquilibriumFrameWeights {
  AlignedDoubleVector local_weights{};
  bool use_weights{false};
  std::size_t matched_frames{0};
  std::size_t invalid_frames{0};
  double raw_weight_sum{0.0};
  double ess{0.0};
  std::vector<std::size_t> ranked_local_indices{};
};

struct PairScoreRow {
  std::string source{};
  std::string target{};
  double score{0.0};
};

struct TptPathCluster {
  int representative_rank{0};
  double total_flux{0.0};
  double total_fraction{0.0};
  std::vector<int> member_ranks;
  std::vector<int> representative_nodes;
};

struct TptPathSupport {
  std::vector<double> edge_flux;
  std::vector<double> edge_fraction;
  std::vector<int> edge_count;
  std::vector<double> state_flux;
  std::vector<double> state_fraction;
  std::vector<int> state_count;

  TptPathSupport() = default;
  TptPathSupport(std::size_t n_edges, std::size_t n_states)
      : edge_flux(n_edges, 0.0), edge_fraction(n_edges, 0.0),
        edge_count(n_edges, 0), state_flux(n_states, 0.0),
        state_fraction(n_states, 0.0), state_count(n_states, 0) {}
};

struct TptBottleneckEdgeRow {
  int from{-1};
  int to{-1};
  int from_macro{-1};
  int to_macro{-1};
  double gross_flux{0.0};
  double net_flux{0.0};
  double relative_net_flux{0.0};
  double from_q{std::numeric_limits<double>::quiet_NaN()};
  double to_q{std::numeric_limits<double>::quiet_NaN()};
  double midpoint_q{std::numeric_limits<double>::quiet_NaN()};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  int dominant_path_count{0};
};

struct TptBottleneckStateRow {
  int state{-1};
  char label{'I'};
  int macro{-1};
  double pi{0.0};
  double q_plus{std::numeric_limits<double>::quiet_NaN()};
  double q_minus{std::numeric_limits<double>::quiet_NaN()};
  double reactive_density{0.0};
  double net_in_flux{0.0};
  double net_out_flux{0.0};
  double through_flux{0.0};
  double transition_proximity{0.0};
  double bottleneck_score{0.0};
  double dominant_path_support_flux{0.0};
  double dominant_path_support_fraction{0.0};
  int dominant_path_count{0};
};

struct EquilibriumPopulationSummary {
  std::vector<double> microstate_populations{};
  std::vector<double> macrostate_populations{};
  std::vector<int64_t> microstate_frame_counts{};
  std::vector<int64_t> macrostate_frame_counts{};
  std::size_t matched_frames{0};
  std::size_t invalid_frames{0};
  double raw_weight_sum{0.0};
};

void log_tica_timing_stats(std::fstream &log,
                           const TitanKinetics::TimingStats &stats);

titan_sparse_graph_module::TptCapturedEdgeRow
to_graph_tpt_edge_row(const TptBottleneckEdgeRow &row);

titan_sparse_graph_module::TptCapturedNodeRow
to_graph_tpt_node_row(const TptBottleneckStateRow &row);

void store_allostery_graph_edges_from_corr_knn(
    DCCMNetwork &dccm_analyzer, const TitanCorrKnnAnalyzer &analyzer);

double tpt_transition_proximity(double q_value);

std::uint64_t pack_directed_edge_key(int from, int to);

double percentile_sorted(std::vector<double> values, double frac);

[[nodiscard]] bool keep_postproc_energy_time(const arguments &args,
                                             double time_ps) noexcept;

std::vector<std::pair<double, double>>
filter_postproc_energy_pairs(const std::vector<std::pair<double, double>> &rows,
                             const arguments &args);

void append_if_finite(std::vector<double> &dst, double value);

std::string trim_copy(std::string value);

std::string to_lower_copy(std::string value);

bool parse_scalar_condition(const std::string &expr, std::string &feature_name,
                            std::string &op, double &threshold,
                            std::string &error_message);

bool evaluate_scalar_condition(double value, const std::string &op,
                               double threshold);

const std::vector<double> *
lookup_lj_feature_vector(const LjMicrostateSummary &summary,
                         const std::string &feature_name);

double quantile_of_finite(const std::vector<double> &values, double frac);

std::vector<int> fallback_extreme_state_indices(const std::vector<double> &values,
                                                bool select_max);

bool resolve_lj_preset(const std::string &preset_name,
                       const LjMicrostateSummary &summary,
                       std::vector<int> &states_out,
                       std::string &resolved_label,
                       std::string &error_message);

bool resolve_protein_preset(const std::string &preset_name,
                            const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const ProteinMicrostateSummary &protein_summary,
                            const Eigen::VectorXd &state_pi,
                            std::vector<int> &states_out,
                            std::string &resolved_label,
                            std::string &error_message);

bool load_lj_feature_rows(const std::vector<std::string> &feature_files,
                          size_t expected_frames,
                          std::vector<std::array<double, kLjFeatureColumns>>
                              &rows_out,
                          std::fstream &log);

LjMicrostateSummary compute_lj_microstate_summary(
    const TitanMSM &msm_analyzer,
    const std::vector<std::string> &feature_files,
    std::fstream &log);

ProteinMicrostateSummary compute_protein_microstate_summary(
    const TitanMSM &msm_analyzer, const TitanRMSAnalyzer *rms_analyzer,
    const TitanDSSPAnalyzer *dssp_analyzer, std::fstream &log);

QNativeMicrostateSummary compute_qnative_microstate_summary(
    const TitanMSM &msm_analyzer, const TitanQNativeAnalyzer *qnative_analyzer,
    std::fstream &log);

bool resolve_tpt_basin_spec(const std::string &spec, const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const LjMicrostateSummary &lj_summary,
                            const ProteinMicrostateSummary &protein_summary,
                            const QNativeMicrostateSummary &qnative_summary,
                            const Eigen::VectorXd &state_pi,
                            std::vector<int> &states_out,
                            std::string &resolved_label,
                            std::string &error_message);

void write_tpt_state_labels(const std::string &filename,
                            const TitanTPT &tpt_analyzer,
                            const TitanMSM &msm_analyzer,
                            const std::vector<int> &macro_assignments,
                            const LjMicrostateSummary &lj_summary,
                            const ProteinMicrostateSummary &protein_summary,
                            const QNativeMicrostateSummary &qnative_summary,
                            const Eigen::VectorXd &state_pi);

double estimate_tpt_projection_bandwidth(const TitanMSM &msm_analyzer);

bool write_tpt_reactive_fes_2d(const std::string &filename,
                               const TitanTPT &tpt_analyzer,
                               const TitanMSM &msm_analyzer,
                               std::fstream &log);

bool write_tpt_flux_map_2d(const std::string &filename,
                           const TitanTPT &tpt_analyzer,
                           const TitanMSM &msm_analyzer);

bool is_trajectory_boundary(const std::vector<int> &replicas,
                            const std::vector<double> &times_ps, size_t prev,
                            size_t curr);

std::vector<std::vector<int>>
build_msm_state_segments(const TitanMSM &msm_analyzer);

bool read_tpt_demux_map(const std::string &path,
                        std::vector<std::vector<int>> &rows_out,
                        std::string &error_message);

bool build_demux_state_segments(
    const TitanMSM &msm_analyzer, const std::vector<std::vector<int>> &demux_rows,
    std::vector<std::vector<int>> &segments_out, std::string &error_message);

bool build_segment_count_matrix(
    int n_states, int lag_frames, const std::vector<std::vector<int>> &segments,
    Eigen::MatrixXd &out_counts, Eigen::VectorXd &out_occupancy,
    std::int64_t &out_transition_count, std::string &error_message);

bool build_tpt_model_from_segments(
    int n_states, int lag_frames, const std::vector<std::vector<int>> &segments,
    bool symmetrize_counts, Eigen::MatrixXd &out_tpm, Eigen::VectorXd &out_pi,
    std::int64_t &out_transition_count, std::string &error_message);

[[maybe_unused]] TptKineticDiagnostics compute_tpt_kinetic_diagnostics_from_segments(
    const std::vector<std::vector<int>> &segments, int n_states,
    int lag_frames, double lag_time_ns, const TitanTPT &tpt_analyzer);

[[maybe_unused]] bool run_tpt_bootstrap(
    const std::vector<std::vector<int>> &base_segments, int n_states,
    int lag_frames, double lag_time_ns, bool symmetrize_counts,
    const std::vector<int> &source_states, const std::vector<int> &sink_states,
    int top_paths, double flux_cut, int n_threads, int n_samples,
    unsigned int seed, TitanExecutor &executor, TptBootstrapAggregate &out_bootstrap,
    std::string &error_message);

void write_tpt_bootstrap_files(const std::string &summary_filename,
                               const std::string &states_filename,
                               const TptBootstrapAggregate &bootstrap,
                               const TitanTPT &reference_tpt,
                               const std::string &model_source);

void write_tpt_kinetics_file(const std::string &filename,
                             const TptKineticDiagnostics &diag,
                             const std::string &model_source);

std::string join_int_list(const std::vector<int> &values);

double path_edge_similarity(const std::vector<int> &lhs,
                            const std::vector<int> &rhs);

std::vector<TptPathCluster>
cluster_tpt_paths(const TitanTPT &tpt_analyzer, double similarity_cut = 0.6);

std::size_t write_tpt_path_clusters_file(const std::string &filename,
                                         const TitanTPT &tpt_analyzer);

TptPathSupport accumulate_tpt_path_support(const TitanTPT &tpt_analyzer,
                                           TitanExecutor &executor);

std::vector<TptBottleneckEdgeRow>
compute_tpt_bottleneck_edges(const TitanTPT &tpt_analyzer,
                             const std::vector<int> &macro_assignments,
                             const TptPathSupport &support);

std::vector<TptBottleneckStateRow>
compute_tpt_bottleneck_states(const TitanTPT &tpt_analyzer,
                              const std::vector<int> &macro_assignments,
                              const TptPathSupport &support);

std::size_t write_tpt_bottleneck_edges_file(
    const std::string &filename,
    const std::vector<TptBottleneckEdgeRow> &rows);

std::size_t write_tpt_bottleneck_states_file(
    const std::string &filename,
    const std::vector<TptBottleneckStateRow> &rows);

void append_tpt_bottleneck_summary(
    const std::string &summary_filename,
    const std::vector<TptBottleneckEdgeRow> &edge_rows,
    const std::vector<TptBottleneckStateRow> &state_rows,
    std::size_t top_n = 5);

bool build_macro_tpt_model(const Eigen::MatrixXd &micro_tpm,
                           const Eigen::VectorXd &micro_pi,
                           const std::vector<int> &macro_assignments,
                           Eigen::MatrixXd &macro_tpm,
                           Eigen::VectorXd &macro_pi,
                           std::string &error_message);

std::size_t write_tpt_all_pairs_kinetics_file(const std::string &filename,
                                              const Eigen::MatrixXd &macro_tpm,
                                              const Eigen::VectorXd &macro_pi,
                                              double lag_time_ns);

bool detect_irregular_energy_dt(
    const std::vector<std::pair<double, double>> &xvg_data, double inferred_dt_ps);

void write_dham_window_audit_file(const std::string &filename,
                                  const std::vector<DhamWindowAuditRow> &rows);

bool is_mbar_equilibrium_weight_source(
    const PostprocEquilibriumWeights &weight_selection);

const std::vector<double> *
context_global_frame_weights(const PostprocEquilibriumWeights &weight_selection,
                             bool pca_context);

MappedEquilibriumFrameWeights map_global_weights_to_local_frame_context(
    const std::string &label, const std::vector<int64_t> &local_global_indices,
    const std::vector<double> &global_frame_weights);

std::string residue_label(const std::string &resname, int resnr,
                          const std::string &chain);

void ensure_csv_header_if_missing(const std::string &filename,
                                  const std::string &header);

void write_equilibrium_provenance(std::ofstream &out,
                                  const PostprocEquilibriumWeights &selection,
                                  const MappedEquilibriumFrameWeights &mapped,
                                  const std::string &note,
                                  int embedding_dims = -1);

void write_umap_equilibrium_summary_file(
    const std::string &filename, const TitanUMAP &umap_analyzer,
    const TitanMSM &msm_analyzer, const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection);

void write_dccm_equilibrium_context_file(
    const std::string &filename, const DCCMNetwork &dccm_analyzer,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection);

std::vector<PairScoreRow> read_pair_score_rows(const std::string &path);

void append_ranked_pair_scores(std::ofstream &out, const std::string &metric,
                               std::vector<PairScoreRow> rows);

void write_allostery_equilibrium_ranking_file(
    const std::string &filename, const info &info_output,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection);

void write_corr_knn_equilibrium_ranking_file(
    const std::string &filename, const info &info_output,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection);

void write_lspmd_equilibrium_ranking_file(
    const std::string &filename, const DCCMNetwork &dccm_analyzer,
    const DCCMNetwork::LSPMDResult &result,
    const MappedEquilibriumFrameWeights &mapped,
    const PostprocEquilibriumWeights &weight_selection);

AlignedDoubleVector map_global_weights_to_local_indices(
    const std::string &label, const std::vector<int64_t> &local_global_indices,
    const std::vector<double> &global_frame_weights, bool &use_weights);

TitanPCA::WeightContext
make_pca_weight_context(const PostprocEquilibriumWeights &weight_selection);

bool compute_equilibrium_populations(
    const TitanMSM &msm_analyzer, const std::vector<double> &global_frame_weights,
    const std::vector<int> &macro_assignments,
    EquilibriumPopulationSummary &summary);

void write_msm_equilibrium_summary_file(
    const std::string &filename, const TitanMSM &msm_analyzer,
    const std::vector<int> &macro_assignments,
    const EquilibriumPopulationSummary &summary,
    const PostprocEquilibriumWeights &weight_selection);

void write_tpt_equilibrium_overlay_file(
    const std::string &filename, const TitanTPT &tpt_analyzer,
    const std::vector<int> &macro_assignments,
    const EquilibriumPopulationSummary &summary,
    const PostprocEquilibriumWeights &weight_selection);

std::vector<std::pair<double, double>>
read_projection_csv_pairs(const std::string &path);

int select_effective_msm_clusters(const arguments &args,
                                  const TitanKinetics &tica_analyzer);

std::string describe_msm_cluster_selection(const arguments &args,
                                           int effective_k,
                                           const TitanKinetics &tica_analyzer);

void rebuild_msm_projections_from_dccm(const arguments &args,
                                       const DCCMNetwork &dccm_analyzer,
                                       const TitanKinetics &tica_analyzer,
                                       TitanMSM &msm_analyzer,
                                       std::fstream &log);

} // namespace postproc_detail

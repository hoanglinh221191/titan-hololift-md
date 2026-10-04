#pragma once

#include "titan_dham.h"
#include "titan_executor.h"
#include "titan_msm.h"
#include "titan_tpt.h"

#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace TitanValidation {

using StateSegments = std::vector<std::vector<int>>;

struct ModelConfig {
  int components{0};
  int lag_frames{0};
  int n_clusters{0};
};

struct ChapmanKolmogorovRowError {
  int multiple{0};
  int state{0};
  double row_weight{0.0};
  double row_mae{std::numeric_limits<double>::quiet_NaN()};
  double row_frobenius{std::numeric_limits<double>::quiet_NaN()};
  double row_max_abs{std::numeric_limits<double>::quiet_NaN()};
};

struct ChapmanKolmogorovMetric {
  int multiple{0};
  std::int64_t direct_transition_count{0};
  int rows_compared{0};
  double weighted_mae{std::numeric_limits<double>::quiet_NaN()};
  double weighted_frobenius{std::numeric_limits<double>::quiet_NaN()};
  double max_abs{0.0};
  std::vector<ChapmanKolmogorovRowError> row_errors;
};

struct ChapmanKolmogorovSummary {
  bool valid{false};
  int lag_frames{0};
  std::string model_source{"not_available"};
  std::vector<ChapmanKolmogorovMetric> metrics;
  std::string error_message{};
};

struct VampCandidateResult {
  ModelConfig config{};
  int folds_requested{0};
  int folds_effective{0};
  int folds_scored{0};
  double score_mean{-std::numeric_limits<double>::infinity()};
  double score_std{std::numeric_limits<double>::infinity()};
  std::string status{"not_evaluated"};
};

struct VampSweepResult {
  std::vector<VampCandidateResult> candidates;
  ModelConfig selected{};
  bool selected_is_current{true};
  bool used_auto_pick{false};
  bool valid{false};
  std::string warning_message{};
  std::unique_ptr<TitanMSM> winning_model{};
};

struct TptKineticDiagnostics {
  int n_segments{0};
  int segments_with_source{0};
  int segments_with_sink{0};
  int segments_with_both{0};
  int min_segment_length_frames{0};
  int max_segment_length_frames{0};
  double mean_segment_length_frames{0.0};
  int64_t total_frames{0};
  int64_t frames_source{0};
  int64_t frames_sink{0};
  int64_t frames_intermediate{0};
  int observed_a_to_b_events{0};
  int observed_b_to_a_events{0};
  int observed_round_trips{0};
  double observed_source_fraction{0.0};
  double observed_sink_fraction{0.0};
  double observed_intermediate_fraction{0.0};
  double lag_time_ns{0.0};
  double model_rate_ab_per_lag{0.0};
  double model_rate_ab_per_ns{0.0};
  double model_mfpt_ab_lags{std::numeric_limits<double>::infinity()};
  double model_mfpt_ab_ns{std::numeric_limits<double>::infinity()};
  int ck_rows_compared{0};
  double ck_mean_abs_error{std::numeric_limits<double>::quiet_NaN()};
  double ck_max_abs_error{std::numeric_limits<double>::quiet_NaN()};
  double ck_weighted_frobenius{std::numeric_limits<double>::quiet_NaN()};
  std::string markov_sanity{"not_available"};
  std::string suitability{
      "insufficient: kinetic diagnostics not computed"};
};

struct TptBootstrapAggregate {
  int requested_samples{0};
  int successful_samples{0};
  std::vector<double> total_reactive_flux;
  std::vector<double> rate_source_to_sink_per_lag;
  std::vector<double> mfpt_source_to_sink_lags;
  std::vector<double> source_population;
  std::vector<double> sink_population;
  Eigen::VectorXd q_forward_sum;
  Eigen::VectorXd q_forward_sq_sum;
  Eigen::VectorXd reactive_density_sum;
  Eigen::VectorXd reactive_density_sq_sum;
};

struct MsmBootstrapAggregate {
  int requested_samples{0};
  int successful_samples{0};
  Eigen::VectorXd stationary_sum;
  Eigen::VectorXd stationary_sq_sum;
  std::vector<std::vector<double>> implied_timescale_samples;
  std::vector<std::vector<double>> ck_weighted_frobenius_samples;
  std::vector<std::vector<double>> ck_weighted_mae_samples;
};

struct DhamRebuildContext {
  bool available{false};
  int n_states{0};
  int lag_frames{0};
  int n_replicas{0};
  std::vector<double> temps_k;
  double target_temp_k{300.0};
  int max_iter{5000};
  double tolerance{1e-6};
  std::string ensemble{"nvt"};
  std::unordered_map<int, std::vector<int>> per_replica_dtrajs;
  std::unordered_map<int, std::vector<double>> per_replica_timestamps_ps;
  std::unordered_map<int, std::vector<double>> per_replica_observables;
};

struct TptValidationContext {
  bool available{false};
  bool kinetic_mode{false};
  bool symmetrize_bootstrap{false};
  int n_states{0};
  int lag_frames{0};
  int top_paths{0};
  double flux_cut{0.0};
  double lag_time_ns{0.0};
  std::string model_source{"not_available"};
  std::vector<int> source_states;
  std::vector<int> sink_states;
  StateSegments bootstrap_segments;
  Eigen::MatrixXd model_tpm;
  Eigen::VectorXd model_pi;
};

StateSegments build_msm_state_segments(const TitanMSM &msm_analyzer);

bool build_segment_count_matrix(int n_states, int lag_frames,
                                const StateSegments &segments,
                                Eigen::MatrixXd &out_counts,
                                Eigen::VectorXd &out_occupancy,
                                std::int64_t &out_transition_count,
                                std::string &error_message);

bool build_markov_model_from_segments(int n_states, int lag_frames,
                                      const StateSegments &segments,
                                      bool symmetrize_counts,
                                      Eigen::MatrixXd &out_tpm,
                                      Eigen::VectorXd &out_pi,
                                      std::int64_t &out_transition_count,
                                      std::string &error_message);

bool build_dham_model_from_context(const DhamRebuildContext &context,
                                   int lag_frames, Eigen::MatrixXd &out_tpm,
                                   Eigen::VectorXd &out_pi,
                                   std::int64_t &out_transition_count,
                                   std::string &error_message,
                                   Eigen::MatrixXd *out_counts = nullptr);

ChapmanKolmogorovSummary
compute_ck_for_segments(int n_states, int lag_frames, const StateSegments &segments,
                        bool symmetrize_counts,
                        const std::vector<int> &multiples,
                        const std::string &model_source);

ChapmanKolmogorovSummary
compute_ck_for_transition_model(const Eigen::MatrixXd &p_tau,
                                int lag_frames, int n_states,
                                const StateSegments &segments,
                                const std::vector<int> &multiples,
                                const std::string &model_source);

ChapmanKolmogorovSummary
compute_ck_for_dham(const DhamRebuildContext &context,
                    const Eigen::MatrixXd &p_tau, int lag_frames,
                    const std::vector<int> &multiples,
                    const std::string &model_source);

double compute_discrete_vamp2_score(const StateSegments &segments, int n_states,
                                    int lag_frames);

VampSweepResult evaluate_vamp_candidates(
    const TitanMSM &base_msm, int base_components,
    const std::vector<ModelConfig> &candidates, int requested_folds,
    int kmeans_iter, unsigned int base_seed, TitanExecutor &executor);

TptKineticDiagnostics compute_tpt_kinetic_diagnostics_from_segments(
    const StateSegments &segments, int n_states, int lag_frames,
    double lag_time_ns, const TitanTPT &tpt_analyzer);

bool run_tpt_bootstrap(const StateSegments &base_segments, int n_states,
                       int lag_frames, double lag_time_ns,
                       bool symmetrize_counts,
                       const std::vector<int> &source_states,
                       const std::vector<int> &sink_states, int top_paths,
                       double flux_cut, int n_threads, int n_samples,
                       unsigned int seed, TitanExecutor &executor,
                       TptBootstrapAggregate &out_bootstrap,
                       std::string &error_message);

bool run_dham_tpt_bootstrap(const DhamRebuildContext &context, int top_paths,
                            double flux_cut,
                            const std::vector<int> &source_states,
                            const std::vector<int> &sink_states, int n_samples,
                            unsigned int seed, int n_threads,
                            TitanExecutor &executor,
                            TptBootstrapAggregate &out_bootstrap,
                            std::string &error_message);

bool run_msm_bootstrap(const StateSegments &base_segments, int n_states,
                       int lag_frames, double lag_time_ns,
                       const std::vector<int> &ck_multiples,
                       int n_timescales, int n_samples, unsigned int seed,
                       int n_threads, TitanExecutor &executor,
                       MsmBootstrapAggregate &out_bootstrap,
                       std::string &error_message);

std::vector<double> compute_implied_timescales_from_tpm(const Eigen::MatrixXd &tpm,
                                                        double lag_time_ns,
                                                        int n_timescales);

} // namespace TitanValidation


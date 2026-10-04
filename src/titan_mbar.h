#pragma once

#include "arguments.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanMBAR {

struct FrameMetadata {
  std::vector<int> replica_ids{};
  std::vector<double> timestamps_ps{};
  std::vector<int64_t> global_indices{};
  std::vector<double> box_volumes_nm3{};

  [[nodiscard]] bool empty() const noexcept {
    return replica_ids.empty() || timestamps_ps.empty() || global_indices.empty();
  }
};

struct WindowAuditRow {
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
  double traj_min_ps{0.0};
  double traj_max_ps{0.0};
  double energy_min_ps{0.0};
  double energy_max_ps{0.0};
  double overlap_start_ps{0.0};
  double overlap_end_ps{0.0};
  double inferred_dt_energy_ps{0.0};
  std::string status{"uninitialized"};
};

struct FrameWeightRow {
  int replica_id{-1};
  std::size_t local_frame_index{0};
  double time_ps{0.0};
  double weight{0.0};
  double sampled_temperature_k{0.0};
  int sampled_state_index{-1};
  int64_t global_frame_index{-1};
};

struct CvPoint {
  double temperature_k{0.0};
  double mean_energy{0.0};
  double std_energy{0.0};
  double heat_capacity{0.0};
  double std_heat_capacity{0.0};
};

struct CoreResult {
  bool solved{false};
  bool converged{false};
  bool used_demux_map{false};
  bool used_per_frame_temperature{false};
  bool box_volume_metadata_available{false};
  std::string abort_reason{};
  std::string solver_method{"n/a"};
  std::string convergence_note{"n/a"};
  std::string temperature_ladder_source{"file"};
  std::vector<double> sampled_temperatures_k{};
  std::vector<int> sampled_counts{};
  std::vector<WindowAuditRow> window_audit{};
  std::vector<FrameWeightRow> frame_weights{};
  std::vector<double> global_frame_weights{};
  std::vector<double> tica_global_frame_weights{};
  std::vector<double> pca_global_frame_weights{};
  std::vector<CvPoint> cv_scan{};
  std::size_t usable_replica_series{0};
  std::size_t active_sampled_states{0};
  std::size_t total_usable_samples{0};
  std::size_t mapped_global_frames{0};
  std::size_t mapped_tica_global_frames{0};
  std::size_t mapped_pca_global_frames{0};
  std::size_t solver_iterations{0};
  std::size_t solver_line_search_backtracks{0};
  double target_temp_k{300.0};
  double target_mean_energy{0.0};
  double target_heat_capacity{0.0};
  double target_std_energy{0.0};
  double target_std_heat_capacity{0.0};
  double residual_inf{0.0};
  double residual_rms{0.0};
  double target_effective_sample_size{0.0};
  double min_state_effective_sample_size{0.0};
  double min_adjacent_overlap{0.0};
};

CoreResult run_temperature_ladder_reweighting(const arguments &args,
                                              const FrameMetadata &frame_metadata,
                                              TitanExecutor &executor,
                                              std::fstream &log);

void write_window_audit_file(const std::string &filename,
                             const std::vector<WindowAuditRow> &rows);
void write_frame_weights_file(const std::string &filename,
                              const std::vector<FrameWeightRow> &rows);
void write_cv_file(const std::string &filename,
                   const std::vector<CvPoint> &points);
void write_summary_file(const std::string &filename, const CoreResult &result,
                        const arguments &args, std::fstream &log);

} // namespace TitanMBAR

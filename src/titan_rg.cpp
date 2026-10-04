#include "titan_rg.h"

#include "titan_executor.h"
#include "titan_gyration.h"
#include "titan_selection_utils.h"
#include "titan_ui.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace {

std::size_t effective_worker_count(std::size_t requested_threads,
                                   std::size_t work_items) noexcept {
  if (requested_threads == 0 || work_items < 512)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(requested_threads, work_items / 256));
}

} // namespace

std::string TitanRgAnalyzer::uppercase_copy(std::string value) {
  return titan_selection::uppercase_copy(std::move(value));
}

std::string TitanRgAnalyzer::trim_copy(std::string value) {
  return titan_selection::trim_copy(std::move(value));
}

bool TitanRgAnalyzer::is_traceable_atom(const std::string &atom_name,
                                        TraceKind trace_kind) noexcept {
  return titan_selection::is_traceable_atom(atom_name, trace_kind);
}

void TitanRgAnalyzer::configure(const RgArgs &args) {
  requested_ = args.flag_rg2;
  hist_bins_ = std::max(1, args.hist_bins);
  enabled_ = requested_ && initialized_ && total_trace_count_ > 0;
}

TitanRgAnalyzer::TraceKind
TitanRgAnalyzer::select_trace_kind(const std::string &mol_type,
                                   const std::string &force_field) const {
  return titan_selection::select_trace_kind(mol_type, force_field);
}

void TitanRgAnalyzer::build_trace_atoms(const Multi_MOL &mol,
                                        const std::string &mol_type,
                                        const std::string &force_field,
                                        std::vector<TraceAtom> &trace_atoms) {
  titan_selection::build_trace_atoms(mol, mol_type, force_field, trace_atoms);
}

void TitanRgAnalyzer::reset_runtime_buffers() {
  raw_x_.assign(total_trace_count_, 0.0);
  raw_y_.assign(total_trace_count_, 0.0);
  raw_z_.assign(total_trace_count_, 0.0);
  reset_trajectory_state();
}

void TitanRgAnalyzer::reset_trajectory_state() {
  rg2_series_.clear();
  rg_series_.clear();
  time_series_ps_.clear();
  shape_series_.clear();
}

void TitanRgAnalyzer::init(const Multi_MOL &mol1, const Multi_MOL &mol2,
                           const arguments &args, std::fstream &log) {
  nocal_mode_ = args.flag_nocal;
  force_field_ = args.force_f;

  if (initialized_) {
    enabled_ = requested_ && total_trace_count_ > 0;
    return;
  }

  build_trace_atoms(mol1, args.mol_1_type, args.force_f, mol1_trace_);
  if (!nocal_mode_)
    build_trace_atoms(mol2, args.mol_2_type, args.force_f, mol2_trace_);

  total_trace_count_ = mol1_trace_.size() + mol2_trace_.size();
  initialized_ = true;
  enabled_ = requested_ && total_trace_count_ > 0;

  if (total_trace_count_ == 0) {
    TitanUI::print_warning(
        "RG2: no supported trace atoms found for the selected molecule types.");
    if (log.is_open()) {
      TitanUI::log_kv(log, "RG2 Trace mol1", "0");
      TitanUI::log_kv(log, "RG2 Trace mol2", "0");
    }
    return;
  }

  reset_runtime_buffers();

  if (log.is_open()) {
    const TraceKind mol1_kind = select_trace_kind(args.mol_1_type, args.force_f);
    TitanUI::log_kv(log, "RG2 Trace mol1",
                    std::to_string(mol1_trace_.size()) + " (" +
                        titan_selection::trace_kind_name(mol1_kind) + ")");
    if (!nocal_mode_) {
      const TraceKind mol2_kind =
          select_trace_kind(args.mol_2_type, args.force_f);
      TitanUI::log_kv(log, "RG2 Trace mol2",
                      std::to_string(mol2_trace_.size()) + " (" +
                          titan_selection::trace_kind_name(mol2_kind) + ")");
    }
  }
}

void TitanRgAnalyzer::gather_trace_coordinates(const Multi_MOL &mol1,
                                               const Multi_MOL &mol2,
                                               std::size_t snap_idx,
                                               TitanExecutor &executor) {
  const std::size_t mol1_count = mol1_trace_.size();
  const std::size_t total_count = total_trace_count_;
  if (total_count == 0)
    return;

  const MOL &mol1_frame = mol1.m_mol[snap_idx];
  const std::size_t mol2_snap =
      mol2.m_mol.empty()
          ? 0
          : (mol2.m_mol.size() > 1
                 ? std::min<std::size_t>(snap_idx, mol2.m_mol.size() - 1)
                 : 0);
  const MOL *mol2_frame = mol2.m_mol.empty() ? nullptr : &mol2.m_mol[mol2_snap];

  auto gather_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      if (i < mol1_count) {
        const int atom_idx = mol1_trace_[i].atom_index;
        if (atom_idx >= 0 &&
            static_cast<std::size_t>(atom_idx) < mol1_frame.pos.x.size() &&
            static_cast<std::size_t>(atom_idx) < mol1_frame.pos.y.size() &&
            static_cast<std::size_t>(atom_idx) < mol1_frame.pos.z.size()) {
          raw_x_[i] = mol1_frame.pos.x[static_cast<std::size_t>(atom_idx)];
          raw_y_[i] = mol1_frame.pos.y[static_cast<std::size_t>(atom_idx)];
          raw_z_[i] = mol1_frame.pos.z[static_cast<std::size_t>(atom_idx)];
        } else {
          raw_x_[i] = raw_y_[i] = raw_z_[i] = 0.0;
        }
        continue;
      }

      const std::size_t local = i - mol1_count;
      const int atom_idx = mol2_trace_[local].atom_index;
      if (mol2_frame != nullptr && atom_idx >= 0 &&
          static_cast<std::size_t>(atom_idx) < mol2_frame->pos.x.size() &&
          static_cast<std::size_t>(atom_idx) < mol2_frame->pos.y.size() &&
          static_cast<std::size_t>(atom_idx) < mol2_frame->pos.z.size()) {
        raw_x_[i] = mol2_frame->pos.x[static_cast<std::size_t>(atom_idx)];
        raw_y_[i] = mol2_frame->pos.y[static_cast<std::size_t>(atom_idx)];
        raw_z_[i] = mol2_frame->pos.z[static_cast<std::size_t>(atom_idx)];
      } else {
        raw_x_[i] = raw_y_[i] = raw_z_[i] = 0.0;
      }
    }
  };

  const std::size_t worker_count =
      effective_worker_count(executor.capacity(), total_count);
  if (worker_count <= 1) {
    gather_range(0, total_count);
    return;
  }

  executor.parallel_for(total_count, worker_count, 1,
                        [&](std::size_t, std::size_t begin, std::size_t end) {
                          gather_range(begin, end);
                        });
}

void TitanRgAnalyzer::process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                                    std::size_t snap_idx,
                                    TitanExecutor &executor) {
  if (!enabled_ || total_trace_count_ == 0)
    return;
  if (snap_idx >= mol1.m_mol.size())
    return;

  gather_trace_coordinates(mol1, mol2, snap_idx, executor);
  const auto stats = titan_gyration::build_point_stats_soa(
      raw_x_.data(), raw_y_.data(), raw_z_.data(), total_trace_count_);
  const double rg2 = titan_gyration::compute_rg2_from_stats(stats);
  const double rg = titan_gyration::compute_rg_from_stats(stats);
  const auto principal_rg2 = titan_gyration::compute_principal_rg2(stats);
  const auto shape =
      titan_gyration::compute_gyration_shape_metrics(principal_rg2);
  const double lambda_sum =
      principal_rg2[0] + principal_rg2[1] + principal_rg2[2];
  const double lambda_mean = lambda_sum / 3.0;
  const double normalized_asphericity =
      lambda_sum > 1.0e-12 ? shape.asphericity / lambda_sum : 0.0;
  const double prolateness =
      lambda_sum > 1.0e-12
          ? 27.0 * (principal_rg2[0] - lambda_mean) *
                (principal_rg2[1] - lambda_mean) *
                (principal_rg2[2] - lambda_mean) /
                (lambda_sum * lambda_sum * lambda_sum)
          : 0.0;
  const double principal_aspect_ratio =
      std::sqrt(std::max(0.0, principal_rg2[0]) /
                std::max(1.0e-12, principal_rg2[2]));

  rg2_series_.push_back(rg2);
  rg_series_.push_back(rg);
  time_series_ps_.push_back(mol1.m_mol[snap_idx].time_of_snap);
  shape_series_.push_back(
      {principal_rg2, shape.kappa2, shape.asphericity,
       normalized_asphericity, shape.acylindricity, prolateness,
       principal_aspect_ratio});
}

TitanRgAnalyzer::OutputSummary TitanRgAnalyzer::compute_summary() const {
  return compute_summary_for_series(rg2_series_, rg_series_);
}

TitanRgAnalyzer::OutputSummary TitanRgAnalyzer::compute_summary_for_series(
    const std::vector<double> &rg2_series,
    const std::vector<double> &rg_series) {
  OutputSummary summary;
  summary.frame_count = rg2_series.size();
  if (rg2_series.empty() || rg_series.empty())
    return summary;

  summary.mean_rg2 =
      std::accumulate(rg2_series.begin(), rg2_series.end(), 0.0) /
      static_cast<double>(rg2_series.size());
  summary.mean_rg =
      std::accumulate(rg_series.begin(), rg_series.end(), 0.0) /
      static_cast<double>(rg_series.size());
  summary.min_rg2 = *std::min_element(rg2_series.begin(), rg2_series.end());
  summary.max_rg2 = *std::max_element(rg2_series.begin(), rg2_series.end());
  summary.min_rg = *std::min_element(rg_series.begin(), rg_series.end());
  summary.max_rg = *std::max_element(rg_series.begin(), rg_series.end());

  double var_rg2 = 0.0;
  for (const double value : rg2_series) {
    const double diff = value - summary.mean_rg2;
    var_rg2 += diff * diff;
  }
  double var_rg = 0.0;
  for (const double value : rg_series) {
    const double diff = value - summary.mean_rg;
    var_rg += diff * diff;
  }
  var_rg2 /= static_cast<double>(rg2_series.size());
  var_rg /= static_cast<double>(rg_series.size());
  summary.std_rg2 = std::sqrt(std::max(0.0, var_rg2));
  summary.std_rg = std::sqrt(std::max(0.0, var_rg));
  return summary;
}

void TitanRgAnalyzer::export_timeseries(const std::string &filename,
                                        const OutputSummary &summary,
                                        std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  out << "# TITAN Rg2 time series\n";
  out << "# trace_atoms_total " << total_trace_count_ << "\n";
  out << "# frames " << summary.frame_count << "\n";
  out << "# mean_rg2_A2 " << std::fixed << std::setprecision(6)
      << summary.mean_rg2 << "\n";
  out << "# mean_rg_A " << std::fixed << std::setprecision(6)
      << summary.mean_rg << "\n";
  out << "# principal_rg2_order lambda1 >= lambda2 >= lambda3\n";
  out << "# prolateness_sign negative=oblate positive=prolate\n";
  out << "# columns = frame time_ps rg2_A2 rg_A lambda1_A2 lambda2_A2 "
         "lambda3_A2 kappa2 asphericity_A2 normalized_asphericity "
         "acylindricity_A2 prolateness principal_aspect_ratio\n";
  for (std::size_t i = 0; i < rg2_series_.size(); ++i) {
    const ShapeFrame shape =
        i < shape_series_.size() ? shape_series_[i] : ShapeFrame{};
    out << std::setw(10) << std::right << (i + 1) << " "
        << std::setw(14) << std::fixed << std::setprecision(4)
        << (i < time_series_ps_.size() ? time_series_ps_[i] : 0.0) << " "
        << std::setw(14) << std::fixed << std::setprecision(6)
        << rg2_series_[i] << " " << std::setw(12) << std::fixed
        << std::setprecision(6) << (i < rg_series_.size() ? rg_series_[i] : 0.0)
        << " " << shape.principal_rg2[0] << " " << shape.principal_rg2[1]
        << " " << shape.principal_rg2[2] << " " << shape.kappa2 << " "
        << shape.asphericity << " " << shape.normalized_asphericity << " "
        << shape.acylindricity << " " << shape.prolateness << " "
        << shape.principal_aspect_ratio
        << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Output", filename);
}

void TitanRgAnalyzer::export_summary(const std::string &filename,
                                     const OutputSummary &summary,
                                     std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  out << "# columns = n_frames mean_rg2_A2 std_rg2_A2 min_rg2_A2 max_rg2_A2 "
         "mean_rg_A std_rg_A min_rg_A max_rg_A\n";
  out << summary.frame_count << " " << std::fixed << std::setprecision(6)
      << summary.mean_rg2 << " " << summary.std_rg2 << " " << summary.min_rg2
      << " " << summary.max_rg2 << " " << summary.mean_rg << " "
      << summary.std_rg << " " << summary.min_rg << " " << summary.max_rg
      << "\n";

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Summary Output", filename);
}

void TitanRgAnalyzer::export_histogram(const std::string &filename,
                                       const OutputSummary &summary,
                                       std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  const int bins = std::max(1, hist_bins_);
  const double min_rg2 = summary.min_rg2;
  const double max_rg2 = summary.max_rg2;
  const double width =
      (bins > 0 && max_rg2 > min_rg2) ? (max_rg2 - min_rg2) / bins : 1.0;
  std::vector<std::size_t> counts(static_cast<std::size_t>(bins), 0);
  for (const double value : rg2_series_) {
    std::size_t idx = 0;
    if (max_rg2 > min_rg2) {
      const double scaled = (value - min_rg2) / width;
      idx = std::min<std::size_t>(static_cast<std::size_t>(scaled),
                                  counts.size() - 1);
    }
    ++counts[idx];
  }

  out << "# columns = bin_left_A2 bin_center_A2 bin_right_A2 count probability\n";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const double left = min_rg2 + static_cast<double>(i) * width;
    const double right =
        (max_rg2 > min_rg2) ? (left + width) : min_rg2;
    const double center = (left + right) * 0.5;
    const double probability =
        rg2_series_.empty()
            ? 0.0
            : static_cast<double>(counts[i]) / static_cast<double>(rg2_series_.size());
    out << std::fixed << std::setprecision(6) << left << " " << center << " "
        << right << " " << counts[i] << " " << probability << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Histogram Output", filename);
}

void TitanRgAnalyzer::export_aggregate_timeseries(const std::string &filename,
                                                  std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  out << "# TITAN aggregated Rg2 time series\n";
  out << "# trajectories " << aggregate_summaries_.size() << "\n";
  out << "# frames " << aggregate_frames_.size() << "\n";
  out << "# principal_rg2_order lambda1 >= lambda2 >= lambda3\n";
  out << "# prolateness_sign negative=oblate positive=prolate\n";
  out << "# columns = traj_index frame time_ps rg2_A2 rg_A lambda1_A2 "
         "lambda2_A2 lambda3_A2 kappa2 asphericity_A2 "
         "normalized_asphericity acylindricity_A2 prolateness "
         "principal_aspect_ratio\n";
  for (const auto &frame : aggregate_frames_) {
    out << std::setw(10) << std::right << frame.traj_index << " "
        << std::setw(10) << std::right << frame.frame_index << " "
        << std::setw(14) << std::fixed << std::setprecision(4) << frame.time_ps
        << " " << std::setw(14) << std::fixed << std::setprecision(6)
        << frame.rg2 << " " << std::setw(12) << std::fixed
        << std::setprecision(6) << frame.rg << " " << frame.principal_rg2[0]
        << " " << frame.principal_rg2[1] << " " << frame.principal_rg2[2]
        << " " << frame.kappa2 << " " << frame.asphericity << " "
        << frame.normalized_asphericity << " " << frame.acylindricity << " "
        << frame.prolateness << " " << frame.principal_aspect_ratio << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Aggregate Output", filename);
}

void TitanRgAnalyzer::export_aggregate_summary(const std::string &filename,
                                               std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  out << "# columns = traj_index n_frames mean_rg2_A2 std_rg2_A2 min_rg2_A2 "
         "max_rg2_A2 mean_rg_A std_rg_A min_rg_A max_rg_A label\n";
  for (const auto &record : aggregate_summaries_) {
    const auto &summary = record.summary;
    out << record.traj_index << " " << summary.frame_count << " " << std::fixed
        << std::setprecision(6) << summary.mean_rg2 << " " << summary.std_rg2
        << " " << summary.min_rg2 << " " << summary.max_rg2 << " "
        << summary.mean_rg << " " << summary.std_rg << " " << summary.min_rg
        << " " << summary.max_rg << " " << record.label << "\n";
  }

  if (!aggregate_rg2_series_.empty() && !aggregate_rg_series_.empty()) {
    const OutputSummary global_summary =
        compute_summary_for_series(aggregate_rg2_series_, aggregate_rg_series_);
    out << -1 << " " << global_summary.frame_count << " " << std::fixed
        << std::setprecision(6) << global_summary.mean_rg2 << " "
        << global_summary.std_rg2 << " " << global_summary.min_rg2 << " "
        << global_summary.max_rg2 << " " << global_summary.mean_rg << " "
        << global_summary.std_rg << " " << global_summary.min_rg << " "
        << global_summary.max_rg << " ALL\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Aggregate Summary", filename);
}

void TitanRgAnalyzer::export_aggregate_histogram(const std::string &filename,
                                                 std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RG2: failed to open " + filename);
    return;
  }

  if (aggregate_rg2_series_.empty()) {
    out << "# columns = bin_left_A2 bin_center_A2 bin_right_A2 count probability\n";
    if (log.is_open())
      TitanUI::log_kv(log, "RG2 Aggregate Histogram", filename);
    return;
  }

  const OutputSummary summary =
      compute_summary_for_series(aggregate_rg2_series_, aggregate_rg_series_);
  const int bins = std::max(1, hist_bins_);
  const double min_rg2 = summary.min_rg2;
  const double max_rg2 = summary.max_rg2;
  const double width =
      (bins > 0 && max_rg2 > min_rg2) ? (max_rg2 - min_rg2) / bins : 1.0;
  std::vector<std::size_t> counts(static_cast<std::size_t>(bins), 0);
  for (const double value : aggregate_rg2_series_) {
    std::size_t idx = 0;
    if (max_rg2 > min_rg2) {
      const double scaled = (value - min_rg2) / width;
      idx = std::min<std::size_t>(static_cast<std::size_t>(scaled),
                                  counts.size() - 1);
    }
    ++counts[idx];
  }

  out << "# columns = bin_left_A2 bin_center_A2 bin_right_A2 count probability\n";
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const double left = min_rg2 + static_cast<double>(i) * width;
    const double right = (max_rg2 > min_rg2) ? (left + width) : min_rg2;
    const double center = (left + right) * 0.5;
    const double probability =
        static_cast<double>(counts[i]) /
        static_cast<double>(aggregate_rg2_series_.size());
    out << std::fixed << std::setprecision(6) << left << " " << center << " "
        << right << " " << counts[i] << " " << probability << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RG2 Aggregate Histogram", filename);
}

void TitanRgAnalyzer::finalize_trajectory(const std::string &rg2_filename,
                                          const std::string &summary_filename,
                                          const std::string &hist_filename,
                                          std::size_t traj_index,
                                          const std::string &traj_label,
                                          std::fstream &log) {
  if (!enabled_ || rg2_series_.empty() || total_trace_count_ == 0) {
    reset_trajectory_state();
    return;
  }

  const OutputSummary summary = compute_summary();
  for (std::size_t i = 0; i < rg2_series_.size(); ++i) {
    const ShapeFrame shape =
        i < shape_series_.size() ? shape_series_[i] : ShapeFrame{};
    aggregate_frames_.push_back(
        {traj_index,
         i + 1,
         i < time_series_ps_.size() ? time_series_ps_[i] : 0.0,
         rg2_series_[i],
         i < rg_series_.size() ? rg_series_[i] : 0.0,
         shape.principal_rg2,
         shape.kappa2,
         shape.asphericity,
         shape.normalized_asphericity,
         shape.acylindricity,
         shape.prolateness,
         shape.principal_aspect_ratio});
  }
  aggregate_summaries_.push_back({traj_index, traj_label, summary});
  aggregate_rg2_series_.insert(aggregate_rg2_series_.end(), rg2_series_.begin(),
                               rg2_series_.end());
  aggregate_rg_series_.insert(aggregate_rg_series_.end(), rg_series_.begin(),
                              rg_series_.end());

  export_timeseries(rg2_filename, summary, log);
  export_summary(summary_filename, summary, log);
  export_histogram(hist_filename, summary, log);

  TitanUI::print_info("[RG2] Exported Rg2 to " + rg2_filename);
  TitanUI::print_info("[RG2] Exported summary to " + summary_filename);
  TitanUI::print_info("[RG2] Exported histogram to " + hist_filename);

  reset_trajectory_state();
}

void TitanRgAnalyzer::finalize_all(const std::string &rg2_all_filename,
                                   const std::string &summary_all_filename,
                                   const std::string &hist_all_filename,
                                   std::fstream &log) {
  if (!enabled_ || aggregate_summaries_.size() <= 1 ||
      aggregate_rg2_series_.empty() || aggregate_rg_series_.empty()) {
    return;
  }

  export_aggregate_timeseries(rg2_all_filename, log);
  export_aggregate_summary(summary_all_filename, log);
  export_aggregate_histogram(hist_all_filename, log);

  TitanUI::print_info("[RG2] Exported aggregate Rg2 to " + rg2_all_filename);
  TitanUI::print_info("[RG2] Exported aggregate summary to " +
                      summary_all_filename);
  TitanUI::print_info("[RG2] Exported aggregate histogram to " +
                      hist_all_filename);
}

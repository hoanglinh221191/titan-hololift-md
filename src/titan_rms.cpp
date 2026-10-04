#include "titan_rms.h"

#include "titan_alignment.h"
#include "titan_executor.h"
#include "titan_selection_utils.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace {

double sum_avx2(const double *data, std::size_t n) {
  return titan_simd_dispatch::sum_doubles(data, n);
}

double squared_diff_sum_avx2(const double *lhs, const double *rhs,
                             std::size_t n) {
  return titan_simd_dispatch::squared_diff_sum(lhs, rhs, n);
}

std::size_t effective_worker_count(std::size_t requested_threads,
                                   std::size_t work_items) noexcept {
  if (requested_threads == 0 || work_items < 512)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(requested_threads, work_items / 256));
}

} // namespace

std::string TitanRMSAnalyzer::uppercase_copy(std::string value) {
  return titan_selection::uppercase_copy(std::move(value));
}

std::string TitanRMSAnalyzer::trim_copy(std::string value) {
  return titan_selection::trim_copy(std::move(value));
}

bool TitanRMSAnalyzer::is_traceable_atom(const std::string &atom_name,
                                         TraceKind trace_kind) noexcept {
  return titan_selection::is_traceable_atom(atom_name, trace_kind);
}

void TitanRMSAnalyzer::configure(const RmsArgs &args) {
  requested_ = args.flag_rms;
  enabled_ = requested_ && initialized_ && total_trace_count_ > 0;
}

TitanRMSAnalyzer::TraceKind
TitanRMSAnalyzer::select_trace_kind(const std::string &mol_type,
                                    const std::string &force_field) const {
  return titan_selection::select_trace_kind(mol_type, force_field);
}

void TitanRMSAnalyzer::build_trace_atoms(const Multi_MOL &mol,
                                         const std::string &mol_type,
                                         const std::string &force_field,
                                         std::vector<TraceAtom> &trace_atoms) {
  titan_selection::build_trace_atoms(mol, mol_type, force_field, trace_atoms);
}

void TitanRMSAnalyzer::reset_runtime_buffers() {
  raw_x_.assign(total_trace_count_, 0.0);
  raw_y_.assign(total_trace_count_, 0.0);
  raw_z_.assign(total_trace_count_, 0.0);
  mean_x_.assign(total_trace_count_, 0.0);
  mean_y_.assign(total_trace_count_, 0.0);
  mean_z_.assign(total_trace_count_, 0.0);
  m2_x_.assign(total_trace_count_, 0.0);
  m2_y_.assign(total_trace_count_, 0.0);
  m2_z_.assign(total_trace_count_, 0.0);
  ref_centered_.resize(total_trace_count_);
  mobile_.resize(total_trace_count_);
  reset_trajectory_state();
}

void TitanRMSAnalyzer::reset_trajectory_state() {
  std::fill(mean_x_.begin(), mean_x_.end(), 0.0);
  std::fill(mean_y_.begin(), mean_y_.end(), 0.0);
  std::fill(mean_z_.begin(), mean_z_.end(), 0.0);
  std::fill(m2_x_.begin(), m2_x_.end(), 0.0);
  std::fill(m2_y_.begin(), m2_y_.end(), 0.0);
  std::fill(m2_z_.begin(), m2_z_.end(), 0.0);
  rmsd_series_.clear();
  time_series_ps_.clear();
  rotation_ = titan::algebra::Mat3::identity();
  ref_initialized_ = false;
  current_traj_frames_ = 0;
}

void TitanRMSAnalyzer::init(const Multi_MOL &mol1, const Multi_MOL &mol2,
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
        "RMSD/RMSF: no supported trace atoms found for the selected "
        "molecule types.");
    if (log.is_open()) {
      TitanUI::log_kv(log, "RMS Trace mol1", "0");
      TitanUI::log_kv(log, "RMS Trace mol2", "0");
    }
    return;
  }

  reset_runtime_buffers();

  if (log.is_open()) {
    const TraceKind mol1_kind = select_trace_kind(args.mol_1_type, args.force_f);
    TitanUI::log_kv(log, "RMS Trace mol1",
                    std::to_string(mol1_trace_.size()) + " (" +
                        titan_selection::trace_kind_name(mol1_kind) + ")");
    if (!nocal_mode_) {
      const TraceKind mol2_kind =
          select_trace_kind(args.mol_2_type, args.force_f);
      TitanUI::log_kv(log, "RMS Trace mol2",
                      std::to_string(mol2_trace_.size()) + " (" +
                          titan_selection::trace_kind_name(mol2_kind) + ")");
    }
  }
}

void TitanRMSAnalyzer::gather_trace_coordinates(const Multi_MOL &mol1,
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

void TitanRMSAnalyzer::update_welford(TitanExecutor &executor) {
  const std::size_t total_count = total_trace_count_;
  if (total_count == 0 || current_traj_frames_ == 0)
    return;

  const double sample_count = static_cast<double>(current_traj_frames_);
  auto update_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const double x = mobile_(i, 0);
      const double y = mobile_(i, 1);
      const double z = mobile_(i, 2);

      const double dx = x - mean_x_[i];
      mean_x_[i] += dx / sample_count;
      const double dx2 = x - mean_x_[i];
      m2_x_[i] += dx * dx2;

      const double dy = y - mean_y_[i];
      mean_y_[i] += dy / sample_count;
      const double dy2 = y - mean_y_[i];
      m2_y_[i] += dy * dy2;

      const double dz = z - mean_z_[i];
      mean_z_[i] += dz / sample_count;
      const double dz2 = z - mean_z_[i];
      m2_z_[i] += dz * dz2;
    }
  };

  const std::size_t worker_count =
      effective_worker_count(executor.capacity(), total_count);
  if (worker_count <= 1) {
    update_range(0, total_count);
    return;
  }

  executor.parallel_for(total_count, worker_count, 1,
                        [&](std::size_t, std::size_t begin, std::size_t end) {
                          update_range(begin, end);
                        });
}

void TitanRMSAnalyzer::process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                                     std::size_t snap_idx,
                                     TitanExecutor &executor) {
  if (!enabled_ || total_trace_count_ == 0)
    return;
  if (snap_idx >= mol1.m_mol.size())
    return;

  gather_trace_coordinates(mol1, mol2, snap_idx, executor);

  const double inv_count = 1.0 / static_cast<double>(total_trace_count_);
  const double centroid_x = sum_avx2(raw_x_.data(), total_trace_count_) * inv_count;
  const double centroid_y = sum_avx2(raw_y_.data(), total_trace_count_) * inv_count;
  const double centroid_z = sum_avx2(raw_z_.data(), total_trace_count_) * inv_count;

  auto fill_mobile_range = [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      mobile_(i, 0) = raw_x_[i] - centroid_x;
      mobile_(i, 1) = raw_y_[i] - centroid_y;
      mobile_(i, 2) = raw_z_[i] - centroid_z;
    }
  };

  const std::size_t worker_count =
      effective_worker_count(executor.capacity(), total_trace_count_);
  if (worker_count <= 1) {
    fill_mobile_range(0, total_trace_count_);
  } else {
    executor.parallel_for(
        total_trace_count_, worker_count, 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          fill_mobile_range(begin, end);
        });
  }

  double rmsd = 0.0;
  if (!ref_initialized_) {
    ref_centered_ = mobile_;
    ref_initialized_ = true;
  } else {
    TitanAlignment::align_frame_inplace(ref_centered_.const_view(),
                                        mobile_.view(), rotation_);
    const double sq_sum =
        squared_diff_sum_avx2(mobile_.data(), ref_centered_.data(),
                              total_trace_count_ * 3);
    rmsd = std::sqrt(sq_sum / static_cast<double>(total_trace_count_));
  }

  ++current_traj_frames_;
  update_welford(executor);
  rmsd_series_.push_back(rmsd);
  time_series_ps_.push_back(mol1.m_mol[snap_idx].time_of_snap);
}

TitanRMSAnalyzer::OutputSummary
TitanRMSAnalyzer::compute_summary(std::vector<double> &rmsf_mol1,
                                  std::vector<double> &rmsf_mol2) const {
  OutputSummary summary;
  if (current_traj_frames_ == 0 || total_trace_count_ == 0)
    return summary;

  if (!rmsd_series_.empty()) {
    summary.mean_rmsd =
        std::accumulate(rmsd_series_.begin(), rmsd_series_.end(), 0.0) /
        static_cast<double>(rmsd_series_.size());
    summary.max_rmsd =
        *std::max_element(rmsd_series_.begin(), rmsd_series_.end());
  }

  const double inv_n = 1.0 / static_cast<double>(current_traj_frames_);
  rmsf_mol1.assign(mol1_trace_.size(), 0.0);
  rmsf_mol2.assign(mol2_trace_.size(), 0.0);

  for (std::size_t i = 0; i < mol1_trace_.size(); ++i) {
    const double variance =
        (m2_x_[i] + m2_y_[i] + m2_z_[i]) * inv_n;
    rmsf_mol1[i] = std::sqrt(std::max(0.0, variance));
  }
  for (std::size_t i = 0; i < mol2_trace_.size(); ++i) {
    const std::size_t idx = mol1_trace_.size() + i;
    const double variance =
        (m2_x_[idx] + m2_y_[idx] + m2_z_[idx]) * inv_n;
    rmsf_mol2[i] = std::sqrt(std::max(0.0, variance));
  }

  if (!rmsf_mol1.empty()) {
    summary.mean_rmsf_mol1 =
        std::accumulate(rmsf_mol1.begin(), rmsf_mol1.end(), 0.0) /
        static_cast<double>(rmsf_mol1.size());
  }
  if (!rmsf_mol2.empty()) {
    summary.mean_rmsf_mol2 =
        std::accumulate(rmsf_mol2.begin(), rmsf_mol2.end(), 0.0) /
        static_cast<double>(rmsf_mol2.size());
  }

  return summary;
}

void TitanRMSAnalyzer::export_rmsd(const std::string &filename,
                                   const OutputSummary &summary,
                                   std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RMSD/RMSF: failed to open " + filename);
    return;
  }

  out << "# TITAN RMSD time series\n";
  out << "# trace_atoms_total " << total_trace_count_ << "\n";
  out << "# frames " << current_traj_frames_ << "\n";
  out << "# mean_rmsd_A " << std::fixed << std::setprecision(6)
      << summary.mean_rmsd << "\n";
  out << "# max_rmsd_A " << std::fixed << std::setprecision(6)
      << summary.max_rmsd << "\n";
  out << "# columns = frame time_ps rmsd_A\n";

  for (std::size_t i = 0; i < rmsd_series_.size(); ++i) {
    out << std::setw(10) << std::right << (i + 1) << " "
        << std::setw(14) << std::fixed << std::setprecision(4)
        << (i < time_series_ps_.size() ? time_series_ps_[i] : 0.0) << " "
        << std::setw(12) << std::fixed << std::setprecision(6)
        << rmsd_series_[i] << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "RMSD Output", filename);
}

void TitanRMSAnalyzer::export_rmsf(const std::string &filename,
                                   const OutputSummary &summary,
                                   const std::vector<double> &rmsf_mol1,
                                   const std::vector<double> &rmsf_mol2,
                                   std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("RMSD/RMSF: failed to open " + filename);
    return;
  }

  out << "# TITAN RMSF profile\n";
  out << "# trace_atoms_total " << total_trace_count_ << "\n";
  out << "# frames " << current_traj_frames_ << "\n";
  out << "# fit_mode combined_trace_kabsch\n";
  out << "# mol1_trace_atoms " << mol1_trace_.size() << "\n";
  out << "# mol2_trace_atoms " << mol2_trace_.size() << "\n";
  out << "# molecule_summary columns = molecule trace_atoms mean_rmsf_A\n";
  out << "mol1 " << mol1_trace_.size() << " " << std::fixed
      << std::setprecision(6) << summary.mean_rmsf_mol1 << "\n";
  if (!nocal_mode_ && !mol2_trace_.empty()) {
    out << "mol2 " << mol2_trace_.size() << " " << std::fixed
        << std::setprecision(6) << summary.mean_rmsf_mol2 << "\n";
  }

  auto write_section = [&](const char *label, const std::vector<TraceAtom> &atoms,
                           const std::vector<double> &rmsf_values) {
    if (atoms.empty())
      return;
    out << "# residue_rmsf molecule = " << label << "\n";
    out << "# columns = ordinal chain resnr resname atom rmsf_A\n";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      out << std::setw(8) << std::right << atoms[i].ordinal << " "
          << std::setw(4) << atoms[i].chain << " " << std::setw(8)
          << atoms[i].resnr << " " << std::setw(6) << atoms[i].resname << " "
          << std::setw(4) << atoms[i].atom_name << " " << std::setw(12)
          << std::fixed << std::setprecision(6)
          << (i < rmsf_values.size() ? rmsf_values[i] : 0.0) << "\n";
    }
  };

  write_section("mol1", mol1_trace_, rmsf_mol1);
  if (!nocal_mode_)
    write_section("mol2", mol2_trace_, rmsf_mol2);

  if (log.is_open())
    TitanUI::log_kv(log, "RMSF Output", filename);
}

void TitanRMSAnalyzer::finalize_trajectory(const std::string &rmsd_filename,
                                           const std::string &rmsf_filename,
                                           std::fstream &log) {
  if (!enabled_ || current_traj_frames_ == 0 || total_trace_count_ == 0) {
    reset_trajectory_state();
    return;
  }

  std::vector<double> rmsf_mol1;
  std::vector<double> rmsf_mol2;
  const OutputSummary summary = compute_summary(rmsf_mol1, rmsf_mol2);

  export_rmsd(rmsd_filename, summary, log);
  export_rmsf(rmsf_filename, summary, rmsf_mol1, rmsf_mol2, log);

  trajectory_series_.push_back(
      TitanRMSAnalyzer::TrajectorySeries{rmsd_series_, time_series_ps_});

  TitanUI::print_info("[RMS] Exported RMSD to " + rmsd_filename);
  TitanUI::print_info("[RMS] Exported RMSF to " + rmsf_filename);

  reset_trajectory_state();
}

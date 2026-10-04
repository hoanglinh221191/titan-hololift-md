#ifndef TITAN_RMS_H
#define TITAN_RMS_H

#include "Multi_MOL.h"
#include "algebra/mat3.h"
#include "algebra/matrix3_buffer.h"
#include "aligned_allocator.h"
#include "arguments.h"
#include "titan_selection_utils.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

class TitanExecutor;

class TitanRMSAnalyzer {
public:
  struct TrajectorySeries {
    std::vector<double> rmsd_series;
    std::vector<double> time_series_ps;
  };

  using TraceAtom = titan_selection::SelectedAtom;
  using TraceKind = titan_selection::TraceKind;

  TitanRMSAnalyzer() = default;

  void configure(const RmsArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL &mol2, const arguments &args,
            std::fstream &log);
  void process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                     std::size_t snap_idx, TitanExecutor &executor);
  void finalize_trajectory(const std::string &rmsd_filename,
                           const std::string &rmsf_filename,
                           std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] std::size_t mol1_trace_count() const noexcept {
    return mol1_trace_.size();
  }
  [[nodiscard]] std::size_t mol2_trace_count() const noexcept {
    return mol2_trace_.size();
  }
  [[nodiscard]] std::size_t total_trace_count() const noexcept {
    return total_trace_count_;
  }
  [[nodiscard]] std::size_t current_frame_count() const noexcept {
    return current_traj_frames_;
  }
  [[nodiscard]] const std::vector<TrajectorySeries> &
  trajectory_series() const noexcept {
    return trajectory_series_;
  }

private:
  struct OutputSummary {
    double mean_rmsd{0.0};
    double max_rmsd{0.0};
    double mean_rmsf_mol1{0.0};
    double mean_rmsf_mol2{0.0};
  };

  static std::string uppercase_copy(std::string value);
  static std::string trim_copy(std::string value);
  static bool is_traceable_atom(const std::string &atom_name,
                                TraceKind trace_kind) noexcept;

  [[nodiscard]] TraceKind select_trace_kind(const std::string &mol_type,
                                            const std::string &force_field) const;
  void build_trace_atoms(const Multi_MOL &mol, const std::string &mol_type,
                         const std::string &force_field,
                         std::vector<TraceAtom> &trace_atoms);
  void reset_runtime_buffers();
  void reset_trajectory_state();
  void gather_trace_coordinates(const Multi_MOL &mol1, const Multi_MOL &mol2,
                                std::size_t snap_idx, TitanExecutor &executor);
  void update_welford(TitanExecutor &executor);
  [[nodiscard]] OutputSummary compute_summary(
      std::vector<double> &rmsf_mol1,
      std::vector<double> &rmsf_mol2) const;
  void export_rmsd(const std::string &filename, const OutputSummary &summary,
                   std::fstream &log) const;
  void export_rmsf(const std::string &filename, const OutputSummary &summary,
                   const std::vector<double> &rmsf_mol1,
                   const std::vector<double> &rmsf_mol2,
                   std::fstream &log) const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool nocal_mode_{false};
  std::string force_field_;

  std::vector<TraceAtom> mol1_trace_;
  std::vector<TraceAtom> mol2_trace_;
  std::size_t total_trace_count_{0};

  AlignedDoubleVector raw_x_;
  AlignedDoubleVector raw_y_;
  AlignedDoubleVector raw_z_;
  AlignedDoubleVector mean_x_;
  AlignedDoubleVector mean_y_;
  AlignedDoubleVector mean_z_;
  AlignedDoubleVector m2_x_;
  AlignedDoubleVector m2_y_;
  AlignedDoubleVector m2_z_;
  std::vector<double> rmsd_series_;
  std::vector<double> time_series_ps_;
  std::vector<TrajectorySeries> trajectory_series_;

  titan::algebra::Matrix3Buffer ref_centered_;
  titan::algebra::Matrix3Buffer mobile_;
  titan::algebra::Mat3 rotation_{titan::algebra::Mat3::identity()};
  bool ref_initialized_{false};
  std::size_t current_traj_frames_{0};
};

#endif // TITAN_RMS_H

#ifndef TITAN_RG_H
#define TITAN_RG_H

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "arguments.h"
#include "titan_selection_utils.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

class TitanRgAnalyzer {
public:
  using TraceAtom = titan_selection::SelectedAtom;
  using TraceKind = titan_selection::TraceKind;

  TitanRgAnalyzer() = default;

  void configure(const RgArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL &mol2, const arguments &args,
            std::fstream &log);
  void process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                     std::size_t snap_idx, TitanExecutor &executor);
  void finalize_trajectory(const std::string &rg2_filename,
                           const std::string &summary_filename,
                           const std::string &hist_filename,
                           std::size_t traj_index,
                           const std::string &traj_label,
                           std::fstream &log);
  void finalize_all(const std::string &rg2_all_filename,
                    const std::string &summary_all_filename,
                    const std::string &hist_all_filename, std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] bool has_multiple_trajectories() const noexcept {
    return aggregate_summaries_.size() > 1;
  }

private:
  struct OutputSummary {
    std::size_t frame_count{0};
    double mean_rg2{0.0};
    double std_rg2{0.0};
    double min_rg2{0.0};
    double max_rg2{0.0};
    double mean_rg{0.0};
    double std_rg{0.0};
    double min_rg{0.0};
    double max_rg{0.0};
  };
  struct AggregateFrame {
    std::size_t traj_index{0};
    std::size_t frame_index{0};
    double time_ps{0.0};
    double rg2{0.0};
    double rg{0.0};
    std::array<double, 3> principal_rg2{0.0, 0.0, 0.0};
    double kappa2{0.0};
    double asphericity{0.0};
    double normalized_asphericity{0.0};
    double acylindricity{0.0};
    double prolateness{0.0};
    double principal_aspect_ratio{1.0};
  };
  struct ShapeFrame {
    std::array<double, 3> principal_rg2{0.0, 0.0, 0.0};
    double kappa2{0.0};
    double asphericity{0.0};
    double normalized_asphericity{0.0};
    double acylindricity{0.0};
    double prolateness{0.0};
    double principal_aspect_ratio{1.0};
  };
  struct AggregateTrajectorySummary {
    std::size_t traj_index{0};
    std::string label;
    OutputSummary summary;
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
  [[nodiscard]] OutputSummary compute_summary() const;
  [[nodiscard]] static OutputSummary
  compute_summary_for_series(const std::vector<double> &rg2_series,
                             const std::vector<double> &rg_series);
  void export_timeseries(const std::string &filename,
                         const OutputSummary &summary, std::fstream &log) const;
  void export_summary(const std::string &filename,
                      const OutputSummary &summary, std::fstream &log) const;
  void export_histogram(const std::string &filename,
                        const OutputSummary &summary, std::fstream &log) const;
  void export_aggregate_timeseries(const std::string &filename,
                                   std::fstream &log) const;
  void export_aggregate_summary(const std::string &filename,
                                std::fstream &log) const;
  void export_aggregate_histogram(const std::string &filename,
                                  std::fstream &log) const;

  int hist_bins_{100};
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
  std::vector<double> rg2_series_;
  std::vector<double> rg_series_;
  std::vector<double> time_series_ps_;
  std::vector<ShapeFrame> shape_series_;
  std::vector<AggregateFrame> aggregate_frames_;
  std::vector<AggregateTrajectorySummary> aggregate_summaries_;
  std::vector<double> aggregate_rg2_series_;
  std::vector<double> aggregate_rg_series_;
};

#endif // TITAN_RG_H

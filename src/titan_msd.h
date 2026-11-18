#ifndef TITAN_MSD_H
#define TITAN_MSD_H

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "arguments.h"
#include "titan_rotational_diffusion.h"
#include "titan_selection_utils.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;
class info;

class TitanMSDAnalyzer {
public:
  struct LagResult {
    std::size_t lag_index{0};
    double lag_time_ps{0.0};
    double msd_a2{0.0};
    std::uint64_t samples{0};
  };

  struct ResidueRange {
    bool has_chain{false};
    std::string chain{};
    std::size_t begin{0};
    std::size_t end{0};
  };

  TitanMSDAnalyzer() = default;

  void configure(const MsdArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL *mol2,
            const arguments &args, std::fstream &log);
  void process_frame(const Multi_MOL &mol1, const Multi_MOL *mol2,
                     std::size_t snap_idx, TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir,
                           const info &info_output, const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, TitanExecutor &executor,
                           std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] std::size_t selected_atom_count() const noexcept {
    return selected_atoms_.size();
  }

private:
  struct FitResult {
    bool ok{false};
    std::string status{"insufficient_points"};
    double tmin_ps{0.0};
    double tmax_ps{0.0};
    double slope{0.0};
    double intercept{0.0};
    double r2{0.0};
    std::size_t n{0};
  };

  // D (and alpha) per contiguous TIME block of the trajectory, each from the
  // MSD of origins inside the block over the full-trajectory fit window
  // (STR-25): the sampling error of D, unlike blocks along the lag axis of
  // one MSD curve, which only measure its curvature.
  struct TimeBlocks {
    std::string status{"not_requested"};
    std::size_t count{0};
    double length_ps{0.0};
    std::vector<double> slopes;
    std::vector<double> alphas;
  };

  void reset_trajectory_state();
  // Moves the stored frames into calc_* (unwrapped in place when the policy
  // is unwrap): one copy of the coordinates at a time (STR-13).
  void prepare_coordinates(AlignedDoubleVector &calc_x,
                           AlignedDoubleVector &calc_y,
                           AlignedDoubleVector &calc_z,
                           std::string &effective_pbc,
                           std::string &pbc_warning);
  void lag_results_in_range(const AlignedDoubleVector &calc_x,
                            const AlignedDoubleVector &calc_y,
                            const AlignedDoubleVector &calc_z,
                            const std::string &effective_pbc,
                            std::size_t first_frame, std::size_t end_frame,
                            std::size_t maxlag, TitanExecutor &executor,
                            std::vector<LagResult> &results) const;
  [[nodiscard]] FitResult fit_linear_window(
      const std::vector<LagResult> &results, bool loglog, double tmin,
      double tmax) const;
  [[nodiscard]] TimeBlocks
  compute_time_blocks(const std::string &blocks_spec, const FitResult &diffusion,
                      const FitResult &alpha, const AlignedDoubleVector &calc_x,
                      const AlignedDoubleVector &calc_y,
                      const AlignedDoubleVector &calc_z,
                      const std::string &effective_pbc,
                      TitanExecutor &executor) const;
  [[nodiscard]] std::size_t resolved_maxlag(std::size_t frames) const;
  void check_memory_budget() const;
  void build_selection(const Multi_MOL &target, const arguments &args);
  void build_residue_selection(const Multi_MOL &target,
                               const std::string &selector);
  [[nodiscard]] bool residue_in_ranges(const mol_info &atom,
                                       const std::vector<ResidueRange> &ranges)
      const;
  [[nodiscard]] const Multi_MOL &target_molecule(const Multi_MOL &mol1,
                                                 const Multi_MOL *mol2) const;
  [[nodiscard]] bool valid_box(const box &frame_box) const noexcept;
  [[nodiscard]] double atom_mass(std::size_t atom_index) const noexcept;
  void append_frame_points(const MOL &frame);
  void compute_lag_results(const AlignedDoubleVector &calc_x,
                           const AlignedDoubleVector &calc_y,
                           const AlignedDoubleVector &calc_z,
                           const std::string &effective_pbc,
                           TitanExecutor &executor,
                           std::vector<LagResult> &results) const;
  [[nodiscard]] FitResult fit_linear(const std::vector<LagResult> &results,
                                     bool loglog) const;
  [[nodiscard]] std::pair<double, double> resolve_fit_window(
      const std::vector<LagResult> &results) const;
  void export_timeseries(const std::string &filename,
                         const std::vector<LagResult> &results,
                         std::fstream &log) const;
  void export_diffusion_summary(const std::string &filename,
                                const std::vector<LagResult> &results,
                                const FitResult &fit,
                                const std::string &effective_pbc,
                                const std::string &pbc_warning,
                                const TimeBlocks &blocks,
                                std::fstream &log) const;
  void export_alpha_fit(const std::string &filename, const FitResult &fit,
                        std::fstream &log) const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool use_mol2_{false};
  std::string selection_{"mol1"};
  std::string mode_{"com"};
  std::string maxlag_{"auto"};
  std::string fit_window_{"auto"};
  std::string pbc_{"auto"};
  bool rotational_requested_{false};
  std::string rotational_fit_window_{"auto"};
  std::string memory_limit_{"auto"};
  // STR-13: every frame is kept (multiple time origins, time blocks), so
  // the storage grows by bytes_per_frame_ per frame; the run stops with a
  // message when it would pass memory_budget_bytes_.
  std::size_t memory_budget_bytes_{0};
  std::size_t bytes_per_frame_{0};

  std::vector<titan_selection::SelectedAtom> selected_atoms_;
  AlignedDoubleVector selected_masses_;
  std::size_t point_count_{0};

  std::vector<double> time_ps_;
  std::vector<box> boxes_;
  std::vector<bool> box_valid_;
  AlignedDoubleVector raw_x_;
  AlignedDoubleVector raw_y_;
  AlignedDoubleVector raw_z_;
  AlignedDoubleVector reconstructed_com_x_;
  AlignedDoubleVector reconstructed_com_y_;
  AlignedDoubleVector reconstructed_com_z_;
  // STR-28: com mode, per selected atom: position unwrapped in time and the
  // raw position of the previous frame (O(atoms) memory, streaming).
  std::vector<double> unwrapped_atom_x_, unwrapped_atom_y_, unwrapped_atom_z_;
  std::vector<double> previous_atom_x_, previous_atom_y_, previous_atom_z_;
  titan_rotational_diffusion::Analyzer rotational_analyzer_;
};

#endif // TITAN_MSD_H

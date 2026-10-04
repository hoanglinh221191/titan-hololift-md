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

  void reset_trajectory_state();
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
  void compute_lag_results(TitanExecutor &executor,
                           std::vector<LagResult> &results,
                           std::string &effective_pbc,
                           std::string &pbc_warning) const;
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
  titan_rotational_diffusion::Analyzer rotational_analyzer_;
};

#endif // TITAN_MSD_H

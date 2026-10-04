#ifndef TITAN_SQ_H
#define TITAN_SQ_H

#include "Multi_MOL.h"
#include "arguments.h"
#include "titan_neighbor_grid.h"
#include "titan_selection_utils.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;
class info;

class TitanSqAnalyzer {
public:
  TitanSqAnalyzer() = default;

  void configure(const SqArgs &args);
  void init(const Multi_MOL &mol1, const arguments &args, std::fstream &log);
  void process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                     TitanExecutor &executor);
  void finalize_trajectory(const std::string &base_dir, const info &info_output,
                           const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }

private:
  struct ChannelSelection {
    std::string name;
    std::string status{"inactive"};
    bool active{false};
    std::vector<titan_selection::SelectedAtom> atoms;
    std::vector<int> atom_indices;
    std::vector<unsigned char> membership;
  };

  struct PairAccumulator {
    std::size_t lhs_channel{0};
    std::size_t rhs_channel{0};
    std::string label;
    std::size_t ordered_pair_count{0};
    std::size_t overlap_count{0};
    std::vector<double> rdf_hist;
    double density_weight_sum_a3{0.0};
  };

  void prepare_channels(const Multi_MOL &mol1, const arguments &args,
                        std::fstream &log);
  void reset_trajectory_state();
  void ensure_layout_from_frame(const MOL &frame);
  void accumulate_distance(std::size_t lhs, std::size_t rhs, double distance_a);
  void export_summary(const std::string &filename, std::fstream &log) const;
  void export_rdf_profile(const PairAccumulator &pair, const std::string &filename,
                          std::fstream &log) const;
  void export_structure_factor(const PairAccumulator &pair,
                               const std::string &filename,
                               std::fstream &log) const;
  [[nodiscard]] std::vector<double>
  build_g_profile(const PairAccumulator &pair) const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool layout_ready_{false};
  bool warned_invalid_box_{false};
  bool warned_empty_channels_{false};

  std::string mol_type_;
  std::string force_field_;
  std::string sticker_names_;

  double rdf_bin_a_{1.0};
  double rdf_max_input_a_{0.0};
  bool rdf_max_auto_{true};
  double q_bin_inv_a_{0.05};
  double q_max_input_inv_a_{0.0};
  bool q_max_auto_{true};
  double fixed_rdf_max_a_{0.0};
  double fixed_q_max_inv_a_{0.0};
  std::size_t rdf_bin_count_{0};
  std::size_t q_bin_count_{0};

  std::size_t atom_count_{0};
  std::size_t total_frames_seen_{0};
  std::size_t valid_frames_{0};
  std::size_t skipped_invalid_box_frames_{0};
  std::size_t skipped_invalid_density_frames_{0};
  double first_valid_volume_a3_{0.0};
  std::string first_valid_box_type_;

  std::vector<ChannelSelection> channels_;
  std::vector<PairAccumulator> pair_profiles_;
  std::vector<unsigned char> active_atom_mask_;
  titan_neighbor::HalfShellGrid neighbor_grid_;
};

#endif // TITAN_SQ_H

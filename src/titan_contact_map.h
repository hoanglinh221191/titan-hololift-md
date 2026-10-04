#ifndef TITAN_CONTACT_MAP_H
#define TITAN_CONTACT_MAP_H

#include "Multi_MOL.h"
#include "arguments.h"
#include "titan_contact_geometry.h"
#include "titan_contact_kernels.h"
#include "titan_contact_primitives.h"
#include "titan_neighbor_grid.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class info;

class TitanContactMapAnalyzer {
public:
  TitanContactMapAnalyzer() = default;

  void configure(const CMapArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL *mol2,
            const arguments &args, std::fstream &log);
  void init(const Multi_MOL &mol1, const arguments &args, std::fstream &log) {
    init(mol1, nullptr, args, log);
  }
  void process_frame(const Multi_MOL &mol1, const Multi_MOL *mol2,
                     std::size_t snap_idx, TitanExecutor &executor);
  void process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                     TitanExecutor &executor) {
    process_frame(mol1, nullptr, snap_idx, executor);
  }
  void finalize_trajectory(const std::string &base_dir, const info &info_output,
                           const arguments &args,
                           const std::vector<std::string> &input_line,
                           int input_name, std::fstream &log);
  void finalize_all(const std::string &base_dir, const info &info_output,
                    std::fstream &log);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] bool latest_contact_count_valid() const noexcept {
    return !contact_count_timeseries_.empty();
  }
  [[nodiscard]] double latest_contact_count() const noexcept {
    return contact_count_timeseries_.empty()
               ? 0.0
               : static_cast<double>(contact_count_timeseries_.back().second);
  }
  [[nodiscard]] double latest_contact_count_normalized() const noexcept;

private:
  struct ContactRecord {
    titan_contact::ResiduePairKey key;
    std::uint64_t contact_frames{0};
    double min_distance_a{0.0};
    double distance_sum_a{0.0};
    std::uint64_t distance_samples{0};
  };

  void reset_trajectory_state();
  void accumulate_pooled_state();
  [[nodiscard]] std::size_t residue_index(
      const std::vector<titan_contact::ResidueKey> &residues,
      const titan_contact::ResidueKey &residue) const;
  [[nodiscard]] bool pair_allowed(const titan_contact::ResidueKey &lhs,
                                  const titan_contact::ResidueKey &rhs) const;
  [[nodiscard]] bool has_valid_box(const box &frame_box) const noexcept;
  [[nodiscard]] std::string residue_label(
      const titan_contact::ResidueKey &residue) const;
  void export_summary(const std::string &filename, std::fstream &log) const;
  void export_edges(const std::string &filename, std::fstream &log) const;
  void export_matrix(const std::string &filename, std::fstream &log) const;
  void export_timeseries(const std::string &filename, std::fstream &log) const;
  void export_occupancy(const std::string &filename, std::fstream &log) const;
  [[nodiscard]] std::size_t feature_pair_count() const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool warned_invalid_box_{false};

  titan_contact::ContactMode mode_{titan_contact::ContactMode::SidechainCOM};
  std::string scope_{"mol1"};
  std::string domain_{"both"};
  double cutoff_a_{6.5};
  int min_seqsep_{3};
  std::size_t atom_count_{0};

  std::uint64_t total_frames_seen_{0};
  std::uint64_t valid_frames_{0};
  std::uint64_t skipped_invalid_box_frames_{0};
  std::uint64_t skipped_incomplete_frames_{0};

  titan_contact_geometry::ContactGeometryCache row_geometry_;
  titan_contact_geometry::ContactGeometryCache col_geometry_;
  std::vector<titan_contact::ResidueKey> row_residues_;
  std::vector<titan_contact::ResidueKey> col_residues_;
  std::vector<titan_contact_geometry::ContactPoint> row_points_scratch_;
  std::vector<titan_contact_geometry::ContactPoint> col_points_scratch_;
  titan_contact_kernel::PointSoA row_soa_scratch_;
  titan_contact_kernel::PointSoA col_soa_scratch_;
  std::vector<std::uint64_t> row_occupancy_frames_;
  std::vector<std::uint64_t> col_occupancy_frames_;
  std::vector<std::pair<double, std::uint64_t>> contact_count_timeseries_;
  std::unordered_map<titan_contact::ResiduePairKey, ContactRecord,
                     titan_contact::ResiduePairKeyHash>
      records_;
  std::uint64_t pooled_valid_frames_{0};
  std::uint64_t pooled_total_frames_seen_{0};
  std::uint64_t pooled_skipped_invalid_box_frames_{0};
  std::uint64_t pooled_skipped_incomplete_frames_{0};
  std::vector<std::uint64_t> pooled_row_occupancy_frames_;
  std::vector<std::uint64_t> pooled_col_occupancy_frames_;
  std::unordered_map<titan_contact::ResiduePairKey, ContactRecord,
                     titan_contact::ResiduePairKeyHash>
      pooled_records_;
};

#endif // TITAN_CONTACT_MAP_H

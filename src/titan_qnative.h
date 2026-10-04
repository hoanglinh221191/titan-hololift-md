#ifndef TITAN_QNATIVE_H
#define TITAN_QNATIVE_H

#include "Multi_MOL.h"
#include "BoxMatrix.h"
#include "arguments.h"
#include "titan_contact_geometry.h"
#include "titan_contact_kernels.h"
#include "titan_contact_primitives.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class TitanExecutor;
class info;

class TitanQNativeAnalyzer {
public:
  TitanQNativeAnalyzer() = default;

  void configure(const QNativeArgs &args);
  void init(const Multi_MOL &mol1, const Multi_MOL *mol2,
            const arguments &args, std::fstream &log,
            const Multi_MOL *trajectory_mol1 = nullptr,
            const Multi_MOL *trajectory_mol2 = nullptr);
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
  [[nodiscard]] const std::vector<double> &all_q_values() const noexcept {
    return all_q_values_;
  }
  [[nodiscard]] bool latest_q_valid() const noexcept {
    return !q_values_.empty();
  }
  [[nodiscard]] double latest_q() const noexcept {
    return q_values_.empty() ? 0.0 : q_values_.back();
  }

private:
  struct NativeContact {
    titan_contact::ResiduePairKey key;
    std::size_t row_index{};
    std::size_t col_index{};
    double reference_distance_a{};
  };

  struct ContactOccupancy {
    titan_contact::ResiduePairKey key;
    double formed_sum{0.0};
    std::uint64_t hard_frames{0};
  };

  struct FrameRecord {
    double time_ps{};
    double q{};
    double formed{};
  };

  void reset_trajectory_state();
  void accumulate_pooled_state();
  [[nodiscard]] bool has_valid_box(const box &frame_box) const noexcept;
  [[nodiscard]] bool pair_allowed(const titan_contact::ResidueKey &lhs,
                                  const titan_contact::ResidueKey &rhs) const;
  [[nodiscard]] double native_pair_distance(
      const std::vector<titan_contact_geometry::ContactPoint> &row_points,
      const std::vector<titan_contact_geometry::ContactPoint> &col_points,
      const titan_contact::ResiduePairKey &key, const BoxMatrix &box_matrix) const;
  void build_native_contacts_from_reference(const Multi_MOL &mol1,
                                            const Multi_MOL *mol2);
  [[nodiscard]] std::size_t residue_index(
      const std::vector<titan_contact::ResidueKey> &residues,
      const titan_contact::ResidueKey &residue) const;
  [[nodiscard]] std::string residue_label(
      const titan_contact::ResidueKey &residue) const;
  [[nodiscard]] double contact_contribution(double distance_a) const;
  [[nodiscard]] double mean_q() const;
  [[nodiscard]] double std_q() const;
  [[nodiscard]] double min_q() const;
  [[nodiscard]] double max_q() const;
  void export_summary(const std::string &filename, std::fstream &log) const;
  void export_native_contacts(const std::string &filename,
                              std::fstream &log) const;
  void export_timeseries(const std::string &filename, std::fstream &log) const;
  void export_occupancy(const std::string &filename, std::fstream &log) const;

  bool requested_{false};
  bool enabled_{false};
  bool initialized_{false};
  bool warned_invalid_box_{false};

  titan_contact::ContactMode mode_{titan_contact::ContactMode::SidechainCOM};
  std::string scope_{"mol1"};
  std::string domain_{"both"};
  std::string reference_source_{"loaded-reference"};
  double cutoff_a_{6.5};
  int min_seqsep_{3};
  bool soft_{false};
  double beta_{5.0};

  std::uint64_t total_frames_seen_{0};
  std::uint64_t valid_frames_{0};
  std::uint64_t skipped_invalid_box_frames_{0};
  std::uint64_t skipped_incomplete_frames_{0};

  titan_contact_geometry::ContactGeometryCache row_geometry_;
  titan_contact_geometry::ContactGeometryCache col_geometry_;
  titan_contact_geometry::ContactGeometryCache trajectory_row_geometry_;
  titan_contact_geometry::ContactGeometryCache trajectory_col_geometry_;
  std::vector<titan_contact::ResidueKey> row_residues_;
  std::vector<titan_contact::ResidueKey> col_residues_;
  std::vector<titan_contact_geometry::ContactPoint> row_points_scratch_;
  std::vector<titan_contact_geometry::ContactPoint> col_points_scratch_;
  titan_contact_kernel::PointSoA row_soa_scratch_;
  titan_contact_kernel::PointSoA col_soa_scratch_;
  std::vector<NativeContact> native_contacts_;
  std::vector<titan_contact_kernel::NativePairIndex> native_pair_indices_;
  std::vector<ContactOccupancy> occupancy_;
  std::vector<FrameRecord> timeseries_;
  std::vector<double> q_values_;
  std::vector<double> all_q_values_;

  std::uint64_t pooled_total_frames_seen_{0};
  std::uint64_t pooled_valid_frames_{0};
  std::uint64_t pooled_skipped_invalid_box_frames_{0};
  std::uint64_t pooled_skipped_incomplete_frames_{0};
  std::vector<ContactOccupancy> pooled_occupancy_;
  std::vector<FrameRecord> pooled_timeseries_;
  std::vector<double> pooled_q_values_;
};

#endif // TITAN_QNATIVE_H

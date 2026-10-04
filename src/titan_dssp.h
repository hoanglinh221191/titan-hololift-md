#ifndef TITAN_DSSP_H
#define TITAN_DSSP_H

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "titan_segment_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

class TitanExecutor;

class TitanDSSPAnalyzer {
public:
  static constexpr std::size_t kStateCount = 8;
  using FrameProfile = std::array<double, kStateCount>;

  enum class State : std::uint8_t {
    H = 0,
    B = 1,
    E = 2,
    G = 3,
    I = 4,
    T = 5,
    S = 6,
    C = 7
  };

  TitanDSSPAnalyzer() = default;

  static bool supports_moltype(const std::string &mol_type);
  static std::size_t
  compatible_backbone_residue_count(const std::vector<mol_info> &info);
  static bool
  has_protein_backbone_signature(const std::vector<mol_info> &info);
  static char state_to_char(State state);
  static const char *state_name(State state);

  void configure(const DsspArgs &args);
  void configure_segments(const SegmentArgs &args, std::uint8_t molecule);
  void init(const Multi_MOL &mol, const std::string &label);
  void process_frame(const Multi_MOL &mol, std::size_t snap_idx,
                     TitanExecutor &executor);
  void finalize_trajectory();
  void export_results(const std::string &filename, std::fstream &log) const;
  void export_content_profile(const std::string &filename,
                              std::fstream &log) const;
  void export_segment_content(const std::string &filename,
                              std::fstream &log) const;
  void export_bridge_registry(const std::string &filename,
                              std::fstream &log) const;
  void export_ladder_distribution(const std::string &filename,
                                  std::fstream &log) const;

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool has_results() const noexcept {
    return !trajectory_fractions_.empty();
  }
  [[nodiscard]] std::size_t residue_count() const noexcept {
    return residues_.size();
  }
  [[nodiscard]] const std::string &label() const noexcept { return label_; }
  [[nodiscard]] const std::vector<std::vector<FrameProfile>> &
  trajectory_frame_profiles() const noexcept {
    return trajectory_frame_profiles_;
  }

private:
  enum class HelixFlag : std::uint8_t { None, Start, Middle, StartAndEnd, End };
  enum class BridgeType : std::uint8_t { None, Parallel, AntiParallel };

  struct BondRecord {
    int residue{-1};
    double energy{0.0};
  };

  struct ResidueBackbone {
    std::size_t ordinal{0};
    std::size_t resnr{0};
    std::size_t internal_resnr{0};
    std::size_t chainnum{0};
    std::size_t local_index{0};
    std::string chain{" "};
    std::string resname{"UNK"};
    int atom_n{-1};
    int atom_h{-1};
    int atom_ca{-1};
    int atom_c{-1};
    int atom_o{-1};
  };

  struct Bridge {
    BridgeType type{BridgeType::None};
    std::uint32_t sheet{0};
    std::uint32_t ladder{0};
    std::vector<std::size_t> i;
    std::deque<std::size_t> j;
  };

  struct ResidueAnnotation {
    std::string entity;
    std::string segment;
  };

  struct ResolvedSegment {
    std::string entity;
    std::string segment;
    std::string coordinate;
    std::size_t begin{0};
    std::size_t end{0};
    std::vector<std::size_t> residue_indices;
  };

  struct FrameBridgePair {
    std::size_t residue_i{0};
    std::size_t residue_j{0};
    BridgeType type{BridgeType::None};
    long long registry{0};
  };

  struct FrameLadder {
    BridgeType type{BridgeType::None};
    std::size_t length{0};
    bool interchain{false};
  };

  void build_residue_map(const std::vector<mol_info> &info);
  void prepare_backbone_coordinates(const MOL &frame);
  void compute_hbond_topology(TitanExecutor &executor);
  void classify_current_frame();
  void compute_mean_sem(std::vector<double> &mean,
                        std::vector<double> &sem) const;
  [[nodiscard]] std::string
  build_consensus(const std::vector<double> &mean) const;
  void ensure_scratch_capacity(std::size_t n_res);
  void load_segment_annotations();
  [[nodiscard]] static const char *
  bridge_type_name(BridgeType type) noexcept;

  [[nodiscard]] bool no_chain_break_between(std::size_t begin,
                                            std::size_t end) const noexcept;
  [[nodiscard]] bool donor_acceptor_excluded(std::size_t donor,
                                             std::size_t acceptor) const
      noexcept;
  [[nodiscard]] bool test_bond(std::size_t donor,
                               std::size_t acceptor) const noexcept;
  [[nodiscard]] BridgeType test_bridge(std::size_t r1,
                                       std::size_t r2) const noexcept;
  [[nodiscard]] double hbond_energy_scalar(std::size_t donor,
                                           std::size_t acceptor) const noexcept;
  void update_best_bond(std::array<BondRecord, 2> &best, std::size_t acceptor,
                        double energy) const noexcept;

  [[nodiscard]] static std::size_t state_index(State state) noexcept {
    return static_cast<std::size_t>(state);
  }

  static constexpr std::size_t flat_index(std::size_t residue_index,
                                          std::size_t state_index) noexcept {
    return residue_index * kStateCount + state_index;
  }

  std::string label_;
  bool enabled_{false};
  bool requested_{true};
  bool initialized_{false};
  std::size_t total_frames_{0};
  std::size_t current_traj_frames_{0};
  double hbond_cutoff_{-0.5};
  std::string segment_config_file_;
  std::uint8_t molecule_id_{1};
  titan_segments::Config segment_config_;
  bool segment_config_loaded_{false};

  std::vector<ResidueBackbone> residues_;
  std::vector<std::array<std::uint64_t, kStateCount>> current_traj_counts_;
  std::vector<std::vector<double>> trajectory_fractions_;
  std::vector<FrameProfile> current_traj_frame_profiles_;
  std::vector<std::vector<FrameProfile>> trajectory_frame_profiles_;
  std::vector<std::size_t> trajectory_frame_counts_;
  std::vector<State> frame_states_;
  std::vector<std::array<BondRecord, 2>> best_acceptors_;
  std::vector<std::array<HelixFlag, 3>> helix_flags_;
  std::vector<ResidueAnnotation> residue_annotations_;
  std::vector<ResolvedSegment> resolved_segments_;
  std::vector<FrameBridgePair> current_frame_bridge_pairs_;
  std::vector<FrameLadder> current_frame_ladders_;
  std::vector<std::vector<FrameBridgePair>> bridge_frames_;
  std::vector<std::vector<FrameLadder>> ladder_frames_;

  AlignedDoubleVector n_x_, n_y_, n_z_;
  AlignedDoubleVector h_x_, h_y_, h_z_;
  AlignedDoubleVector ca_x_, ca_y_, ca_z_;
  AlignedDoubleVector c_x_, c_y_, c_z_;
  AlignedDoubleVector o_x_, o_y_, o_z_;
  AlignedDoubleVector kappa_deg_;

  std::vector<std::uint8_t> has_n_;
  std::vector<std::uint8_t> has_h_;
  std::vector<std::uint8_t> has_ca_;
  std::vector<std::uint8_t> has_c_;
  std::vector<std::uint8_t> has_o_;
  std::vector<std::uint8_t> donor_valid_;
  std::vector<std::uint8_t> acceptor_valid_;
  std::vector<std::uint8_t> has_kappa_;
};

#endif // TITAN_DSSP_H

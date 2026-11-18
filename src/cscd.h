#ifndef CSCD_H
#define CSCD_H
// -cscd: side-chain centre-of-mass distances between the residues of
// molecule 1 and molecule 2.
//
// Every molecule-1 residue gets one file, <prefix><chain><resnr>.dat (a
// chain+number that repeats in the molecule adds _r<internal number>). A row
// is one frame: the time, then the distance from that residue's side-chain
// centre of mass to the nearest molecule-2 side-chain centre (default), or to
// every molecule-2 residue in topology order (-act). The centres are the
// engine's per-frame residue centres (MOL::take_com: the atoms marked side
// chain, the whole residue when none is marked), so distances are plain
// Euclidean on the frame's coordinates after the engine's PBC treatment, in
// its length unit. The header says which kind of centre a run used.
//
// With several input trajectories the working-directory files hold one row
// per distinct time: rows of different trajectories at the same time are
// averaged (replicas), other times are kept in time order (segments).
// -opdbsep also writes each trajectory's own rows into its output folder.

#include "COM.h"

#include <cstddef>
#include <string>
#include <vector>

class TitanExecutor;

class all_cscd {
public:
  struct Options {
    bool nearest_only{true};       // false with -act: one column per partner
    bool skip_same_residue{false}; // -g=x-x: a residue is not its own partner
    std::string file_prefix{"out_cal_dis_com_"};
    std::string time_unit{"ps"};
    std::string length_unit{"A"};
    // What the engine's centres are, for the file header.
    std::string centre{"side chain"};
  };

  // mol1_internal / mol2_internal: each centre's internal residue number
  // (mol_info::get_internal_resnr), used to name residues whose chain and
  // number repeat; may be left empty.
  void configure(const Options &options,
                 std::vector<std::size_t> mol1_internal = {},
                 std::vector<std::size_t> mol2_internal = {});
  [[nodiscard]] bool configured() const { return configured_; }

  // Adds the distances of one frame. `frame` counts the analysed frames of
  // the current trajectory from 0. A centre is identified by its first atom
  // (com::get_respos), unique within its molecule. Throws std::runtime_error
  // when the centres differ from the first frame's or a frame number repeats.
  void add_frame(std::size_t frame, double time, const std::vector<com> &mol1,
                 const std::vector<com> &mol2, TitanExecutor *executor,
                 std::size_t threads);

  // Closes the current trajectory: writes its own rows into `directory` when
  // that is not empty, then merges them into the rows of earlier ones.
  void end_trajectory(const std::string &directory);

  // Writes the merged files into `directory` ("" = working directory).
  // Ends a trajectory left open. Returns the number of files written.
  std::size_t write(const std::string &directory);

  [[nodiscard]] std::size_t residues() const { return labels_.size(); }
  [[nodiscard]] std::size_t columns() const { return total_columns_; }
  [[nodiscard]] std::size_t trajectories() const { return trajectories_; }
  [[nodiscard]] std::size_t merged_rows() const { return count_.size(); }
  // Merged rows that average more than one trajectory.
  [[nodiscard]] std::size_t averaged_rows() const;

private:
  struct Rows {
    std::vector<std::size_t> frame; // frame number (open trajectory only)
    std::vector<double> time;
    std::vector<double> value; // rows x total_columns_
  };

  void build_layout(const std::vector<com> &mol1,
                    const std::vector<com> &mol2);
  void check_layout(const std::vector<com> &mol1,
                    const std::vector<com> &mol2) const;
  void sort_merged_by_time();
  void write_files(const std::string &directory, const Rows &rows,
                   const std::vector<std::size_t> *counts) const;

  Options options_;
  bool configured_{false};
  bool layout_ready_{false};

  std::vector<std::size_t> mol1_internal_, mol2_internal_;

  // Layout, fixed by the first frame.
  std::vector<std::size_t> mol1_atom_, mol2_atom_; // first atom of each centre
  std::vector<std::string> labels_;         // per molecule-1 residue
  std::vector<std::string> partner_labels_; // per molecule-2 residue
  std::vector<std::size_t> self_partner_;   // npos when none
  std::vector<std::size_t> offset_;         // first column of each residue
  std::size_t total_columns_{0};

  Rows current_; // rows of the open trajectory
  Rows sum_;     // closed trajectories: sums per distinct time, time-sorted
  std::vector<std::size_t> count_; // trajectories contributing to sum_ rows
  std::size_t trajectories_{0};

  std::vector<double> x2_, y2_, z2_; // molecule-2 centres of the frame
};

#endif

#ifndef TITAN_SG_STREAMING_H
#define TITAN_SG_STREAMING_H

#include "Multi_MOL.h"
#include "cache_sigma_epsilon.h"
#include "groups.h"
#include "titan_energy_decomposition.h"
#include "titan_executor.h"

#include <chrono>
#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

class SgStreamingAnalyzer {
public:
  SgStreamingAnalyzer() = default;

  bool init(groups_cal &groups, const Multi_MOL &rec, const Multi_MOL &lig,
            const cache_sigma_epsilon &cached_data, const arguments &args,
            std::size_t available_ram_bytes, std::fstream *log);

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] const std::string &resolved_mode() const noexcept {
    return resolved_mode_;
  }
  [[nodiscard]] const std::string &resolved_kernel() const noexcept {
    return resolved_kernel_;
  }

  void process_frame(const MOL &rec_frame, const MOL &lig_frame,
                     const Multi_MOL &rec, const Multi_MOL &lig,
                     const cache_sigma_epsilon &cached_data,
                     const arguments &args, TitanExecutor &executor,
                     std::size_t trajectory_index);

  void finalize(const arguments &args);

private:
  struct AtomGroup {
    int group_index{-1};
    std::string name;
    std::vector<std::size_t> atoms;
  };

  struct RunningStats {
    std::size_t n{0};
    double mean_elec{0.0};
    double mean_vdw{0.0};
    double mean_total{0.0};
    double m2_elec{0.0};
    double m2_vdw{0.0};
    double m2_total{0.0};

    void add(double elec, double vdw);
    [[nodiscard]] double sd_elec() const;
    [[nodiscard]] double sd_vdw() const;
    [[nodiscard]] double sd_total() const;
  };

  struct PairEnergy {
    double elec{0.0};
    double vdw{0.0};
    PairEnergy &operator+=(const PairEnergy &rhs) {
      elec += rhs.elec;
      vdw += rhs.vdw;
      return *this;
    }
  };

  struct PairState {
    std::string name;
    AtomGroup rec_group;
    AtomGroup lig_group;
    RunningStats all;
    std::vector<RunningStats> per_traj;
    std::ofstream elec_ts;
    std::ofstream vdw_ts;
    std::ofstream total_ts;
  };

  [[nodiscard]] bool should_use_stream_mode(
      const cache_sigma_epsilon &cached_data, const arguments &args,
      std::size_t rec_atoms, std::size_t lig_atoms,
      std::size_t available_ram_bytes, std::fstream *log);
  void write_memory_report(const cache_sigma_epsilon &cached_data,
                           const arguments &args, std::size_t rec_atoms,
                           std::size_t lig_atoms,
                           std::size_t available_ram_bytes) const;
  AtomGroup build_group(groups_cal &groups, int subgroup_index,
                        std::size_t parent_first, std::size_t mol_size) const;
  void open_timeseries_files(const arguments &args);
  PairEnergy compute_pair_energy(const PairState &pair, const MOL &rec_frame,
                                 const MOL &lig_frame, const Multi_MOL &rec,
                                 const Multi_MOL &lig,
                                 const cache_sigma_epsilon &cached_data,
                                 const arguments &args,
                                 TitanExecutor &executor);

  bool active_{false};
  bool timeseries_full_{true};
  bool cutoff_active_{true};
  bool use_cell_list_{true};
  double cutoff_nm_{-1.0};
  double memory_budget_bytes_{0.0};
  std::string resolved_mode_{"cache"};
  std::string resolved_kernel_{"scalar"};
  std::string cache_policy_note_;
  EnergyDecompositionProvider energy_provider_;
  std::vector<PairState> pairs_;
  std::chrono::steady_clock::duration process_time_{};
  std::size_t frames_processed_{0};
  std::size_t cell_list_frames_{0};
  std::size_t exact_frames_{0};
  std::size_t avx2_frames_{0};
};

#endif // TITAN_SG_STREAMING_H

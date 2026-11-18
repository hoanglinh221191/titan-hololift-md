// Replica -> thermodynamic-state map for MBAR (-mbar-demux-map), KIN-17.
//
// Each row lists, for every thermodynamic state (column), the replica that
// sampled it. A row with n_states + 1 numbers starts with its time in ps (the
// layout of GROMACS replica_index.xvg); a row with n_states numbers has no
// time column, and its rows are then the rows of the energy .xvg files, which
// must share one time axis and have exactly as many rows as the map.
//
// A sample is assigned by TIME, never by a row or frame index: the state of
// replica r at time t is read from the row nearest to t (the convention of
// gmx trjcat -demux), provided that row lies within half the median row
// spacing. Times outside the map, rows missing replica r and gaps return -1,
// and the caller refuses to run with unresolved samples.
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace titan_fe {

class DemuxStateMap {
public:
  enum class TimeSource { none, column, energy_rows };

  // n_states: thermodynamic states (columns); n_replicas: energy files.
  bool read(const std::string &path, std::size_t n_states,
            std::size_t n_replicas, std::string &error);

  // Map without a time column: take the row times from the energy files
  // (one vector of row times per replica, parsed before -b/-e/-dt; an empty
  // vector is a replica without energy data).
  bool attach_energy_row_times(
      const std::vector<std::vector<double>> &energy_row_times,
      std::string &error);

  [[nodiscard]] bool has_time_column() const noexcept {
    return source_ == TimeSource::column;
  }
  [[nodiscard]] bool timed() const noexcept {
    return source_ != TimeSource::none && times_ps_.size() == rows_.size();
  }
  [[nodiscard]] TimeSource time_source() const noexcept { return source_; }
  [[nodiscard]] const char *time_source_name() const noexcept;
  [[nodiscard]] std::size_t rows() const noexcept { return rows_.size(); }
  [[nodiscard]] bool covers_replica(int replica) const {
    return state_of_replica_.count(replica) != 0;
  }

  // Row nearest to time_ps, -1 when none lies within half the row spacing.
  [[nodiscard]] long row_at(double time_ps) const;
  // State sampled by `replica` at time_ps, -1 when it cannot be resolved.
  [[nodiscard]] int state_at(int replica, double time_ps) const;

private:
  void finish_times();

  std::vector<std::vector<int>> rows_;  // [row][state] -> replica, -1 absent
  std::vector<double> times_ps_;        // one per row once timed
  std::unordered_map<int, std::vector<int>> state_of_replica_; // [row] -> state
  TimeSource source_{TimeSource::none};
  double half_spacing_ps_{0.0};
};

} // namespace titan_fe

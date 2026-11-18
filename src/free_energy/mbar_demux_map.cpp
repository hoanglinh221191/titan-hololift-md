#include "free_energy/mbar_demux_map.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace titan_fe {
namespace {

// absolute tolerance on times printed to 1e-3 ps, plus a relative part
double time_tolerance(double t) { return 1e-3 + 1e-9 * std::abs(t); }

bool to_replica_id(double value, int &out) {
  if (!std::isfinite(value) || std::abs(value - std::round(value)) > 1e-6)
    return false;
  out = static_cast<int>(std::lround(value));
  return true;
}

} // namespace

const char *DemuxStateMap::time_source_name() const noexcept {
  switch (source_) {
  case TimeSource::column:
    return "time column";
  case TimeSource::energy_rows:
    return "energy-file rows";
  default:
    return "none";
  }
}

bool DemuxStateMap::read(const std::string &path, std::size_t n_states,
                         std::size_t n_replicas, std::string &error) {
  rows_.clear();
  times_ps_.clear();
  state_of_replica_.clear();
  source_ = TimeSource::none;
  half_spacing_ps_ = 0.0;
  error.clear();
  if (n_states == 0) {
    error = "demux map needs at least one thermodynamic state";
    return false;
  }
  std::ifstream in(path);
  if (!in.is_open()) {
    error = "cannot open demux map file: " + path;
    return false;
  }

  std::string line;
  std::size_t columns = 0;
  int line_no = 0;
  // KIN-23: every parse error names the file and the line
  const auto where = [&path](int line) {
    return "demux map " + path + ":" + std::to_string(line);
  };
  std::vector<double> times;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash = line.find('#');
    if (hash != std::string::npos)
      line.erase(hash);
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '@')
      continue; // blank line or xvg directive
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    std::vector<double> values;
    std::string token;
    while (iss >> token) {
      try {
        std::size_t used = 0;
        const double v = std::stod(token, &used);
        if (used != token.size())
          throw std::invalid_argument(token);
        values.push_back(v);
      } catch (const std::exception &) {
        error = where(line_no) + ": not a number: '" + token + "'";
        return false;
      }
    }
    if (values.empty())
      continue;
    if (columns == 0) {
      columns = values.size();
      if (columns == n_states + 1) {
        source_ = TimeSource::column;
      } else if (columns != n_states) {
        error = where(line_no) + ": " + std::to_string(columns) +
                " columns; expected " + std::to_string(n_states) +
                " (one replica per state) or " + std::to_string(n_states + 1) +
                " (time in ps, then one replica per state)";
        return false;
      }
    } else if (values.size() != columns) {
      error = where(line_no) + ": " + std::to_string(values.size()) +
              " columns, the first data row has " + std::to_string(columns);
      return false;
    }

    std::size_t offset = 0;
    if (source_ == TimeSource::column) {
      if (!std::isfinite(values[0])) {
        error = where(line_no) + ": time is not finite";
        return false;
      }
      if (!times.empty() &&
          !(values[0] > times.back() + time_tolerance(times.back()))) {
        error = where(line_no) + ": times must increase strictly";
        return false;
      }
      times.push_back(values[0]);
      offset = 1;
    }
    std::vector<int> row(n_states, -1);
    std::vector<unsigned char> seen(n_replicas, 0);
    for (std::size_t s = 0; s < n_states; ++s) {
      int replica = -1;
      if (!to_replica_id(values[offset + s], replica) || replica < -1 ||
          (replica >= 0 && static_cast<std::size_t>(replica) >= n_replicas)) {
        error = where(line_no) + ": replica index must be an integer in [0, " +
                std::to_string(n_replicas) + ") or -1";
        return false;
      }
      if (replica >= 0) {
        if (seen[static_cast<std::size_t>(replica)]) {
          error = where(line_no) + ": replica " +
                  std::to_string(replica) + " appears in two states";
          return false;
        }
        seen[static_cast<std::size_t>(replica)] = 1;
      }
      row[s] = replica;
    }
    rows_.push_back(std::move(row));
  }
  if (rows_.empty()) {
    error = "demux map " + path + " has no data rows";
    return false;
  }

  for (std::size_t r = 0; r < rows_.size(); ++r) {
    for (std::size_t s = 0; s < n_states; ++s) {
      const int replica = rows_[r][s];
      if (replica < 0)
        continue;
      auto &states = state_of_replica_[replica];
      if (states.empty())
        states.assign(rows_.size(), -1);
      states[r] = static_cast<int>(s);
    }
  }
  if (source_ == TimeSource::column) {
    times_ps_ = std::move(times);
    finish_times();
  }
  return true;
}

bool DemuxStateMap::attach_energy_row_times(
    const std::vector<std::vector<double>> &energy_row_times,
    std::string &error) {
  error.clear();
  if (source_ == TimeSource::column)
    return true;
  const std::vector<double> *reference = nullptr;
  for (std::size_t k = 0; k < energy_row_times.size(); ++k) {
    const auto &times = energy_row_times[k];
    if (times.empty())
      continue;
    if (times.size() != rows_.size()) {
      error = "demux map without a time column has " +
              std::to_string(rows_.size()) + " rows but the energy file of "
              "replica " + std::to_string(k) + " has " +
              std::to_string(times.size()) +
              "; such a map needs exactly one row per energy row. Add the "
              "time (ps) as the first column (GROMACS replica_index.xvg)";
      return false;
    }
    if (!reference) {
      reference = &times;
      continue;
    }
    for (std::size_t r = 0; r < times.size(); ++r) {
      if (std::abs(times[r] - (*reference)[r]) >
          time_tolerance((*reference)[r])) {
        error = "demux map without a time column: the energy files do not "
                "share one time axis (replica " + std::to_string(k) +
                ", row " + std::to_string(r) +
                "). Add the time (ps) as the first column of the map";
        return false;
      }
    }
  }
  if (!reference) {
    error = "demux map without a time column: no energy rows to take the "
            "times from";
    return false;
  }
  for (std::size_t r = 1; r < reference->size(); ++r) {
    if (!((*reference)[r] > (*reference)[r - 1])) {
      error = "demux map without a time column: energy times do not increase "
              "strictly (row " + std::to_string(r) + ")";
      return false;
    }
  }
  times_ps_ = *reference;
  source_ = TimeSource::energy_rows;
  finish_times();
  return true;
}

void DemuxStateMap::finish_times() {
  std::vector<double> spacing;
  spacing.reserve(times_ps_.size());
  for (std::size_t r = 1; r < times_ps_.size(); ++r)
    spacing.push_back(times_ps_[r] - times_ps_[r - 1]);
  half_spacing_ps_ = 0.0;
  if (!spacing.empty()) {
    const auto mid = spacing.begin() + static_cast<long>(spacing.size() / 2);
    std::nth_element(spacing.begin(), mid, spacing.end());
    half_spacing_ps_ = 0.5 * *mid;
  }
}

long DemuxStateMap::row_at(double time_ps) const {
  if (!timed() || !std::isfinite(time_ps))
    return -1;
  const auto it = std::lower_bound(times_ps_.begin(), times_ps_.end(), time_ps);
  long best = -1;
  double best_distance = std::numeric_limits<double>::infinity();
  if (it != times_ps_.end()) {
    best = static_cast<long>(it - times_ps_.begin());
    best_distance = *it - time_ps;
  }
  if (it != times_ps_.begin()) {
    // a tie goes to the later row, as gmx trjcat -demux rounds
    const double distance = time_ps - *(it - 1);
    if (distance < best_distance - 1e-9 * std::max(1.0, std::abs(time_ps))) {
      best = static_cast<long>(it - times_ps_.begin()) - 1;
      best_distance = distance;
    }
  }
  if (best < 0 || best_distance > half_spacing_ps_ + time_tolerance(time_ps))
    return -1;
  return best;
}

int DemuxStateMap::state_at(int replica, double time_ps) const {
  const auto it = state_of_replica_.find(replica);
  if (it == state_of_replica_.end())
    return -1;
  const long row = row_at(time_ps);
  if (row < 0)
    return -1;
  return it->second[static_cast<std::size_t>(row)];
}

} // namespace titan_fe

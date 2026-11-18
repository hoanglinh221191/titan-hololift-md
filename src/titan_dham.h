// ============================================================================
// TitanDHAM - input helpers of the -dham reweighting: GROMACS .xvg energy
// parsing, trajectory/energy window matching, file lists and temperatures.
//
// KIN-13: the estimator is TRAM (titan_tram.h); the small-bias DHAM
// approximation with state-mean energies (Rosta & Hummer, JCTC 2015) that
// this class implemented before is gone: for temperature ensembles the bias
// depends on each configuration's energy, not on the Markov state.
//
// Dependencies: Eigen/Dense, TitanUI, parse_xvg.h
// ============================================================================
#pragma once

#include "parse_xvg.h"
#include "titan_ui.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================================
// Constants
// ============================================================================
static constexpr double DHAM_kB_kJ = 0.008314463;  // kJ/(mol·K)
static constexpr double DHAM_LOG_ZERO_GUARD = 1e-300;

// Row-major layout for AVX2-friendly row scans in iterate_once()
using MatrixXdRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>;

// ============================================================================
// TitanDHAM
// ============================================================================
class TitanDHAM {
public:
    struct WindowMatchResult {
        std::vector<std::size_t> kept_indices;
        std::vector<double> kept_dtraj_times_ps;
        std::vector<double> aligned_values;
        double inferred_dt_energy_ps{0.0};
        double match_tolerance_ps{0.0};
        double traj_min_ps{0.0};
        double traj_max_ps{0.0};
        double energy_min_ps{0.0};
        double energy_max_ps{0.0};
        double overlap_start_ps{0.0};
        double overlap_end_ps{0.0};
        std::size_t original_frames{0};
        std::size_t kept_frames{0};
        std::size_t matched_frames{0};
        std::size_t dropped_outside_overlap{0};
        std::size_t unmatched_in_overlap{0};
        bool overlap_found{false};
        // KIN-21: one trajectory frame per energy row. Frames whose nearest
        // row lies within tolerance but is nearer to another frame keep the
        // frame (kept_indices) and get NaN (no energy sample); counted here.
        std::size_t duplicate_energy_matches{0};
        double inferred_dt_traj_ps{0.0};       // median trajectory spacing
        bool energy_coarser_than_traj{false};  // energy dt > trajectory dt
    };

    // -------------------------------------------------------------------------
    // Static helpers — public so postproc code can call them
    // -------------------------------------------------------------------------

    // Parse GROMACS .xvg file → {time_ps, energy_kJ} pairs
    // Uses high-perf TitanIO::parse_xvg_time_value under the hood
    static std::vector<std::pair<double,double>>
    parse_xvg(const std::string& path);

    // Median of the positive, finite differences between consecutive times
    // (ps); 0 when there is none. Robust to a repeated frame (difference 0)
    // or an atypical first spacing.
    static double median_positive_spacing(const std::vector<double>& times_ps);

    static WindowMatchResult
    window_match_energy_to_dtraj(
        const std::vector<double>& dtraj_times_ps,
        const std::vector<std::pair<double,double>>& xvg_data,
        double dt_energy_ps);

    // Map energy values to dtraj timestamps via nearest-neighbour matching.
    // Returns vector<double> of same length as dtraj_times_ps.
    // NaN is inserted where no match is found within 0.5*dt_energy_ps.
    static std::vector<double>
    align_energy_to_dtraj(const std::vector<double>& dtraj_times_ps,
                          const std::vector<std::pair<double,double>>& xvg_data,
                          double dt_energy_ps);

    // Compute per-state mean energy from dtraj + aligned energy array.
    // Returns matrix [n_states × n_replicas].
    // Frame-state pairs where energy is NaN are excluded from average.
    static Eigen::MatrixXd
    compute_state_energies(
        const std::unordered_map<int, std::vector<int>>&    per_replica_dtrajs,
        const std::unordered_map<int, std::vector<double>>& per_replica_energies,
        int n_states,
        int n_replicas,
        int num_threads = 1);

    // Read list of file paths from a text file (one path per line)
    static std::vector<std::string>
    read_file_list(const std::string& list_path);

    // Read temperatures from a text file (one temperature per line, in K)
    static std::vector<double>
    read_temps(const std::string& temps_path);

};

// ============================================================================
// Inline implementation of I/O helpers
// ============================================================================

inline std::vector<std::pair<double,double>>
TitanDHAM::parse_xvg(const std::string& path) {
    auto result = TitanIO::parse_xvg_time_value(path, 0, 1);
    if (!result) {
        TitanUI::print_error("DHAM: " + result.error());
        return {};
    }
    return std::move(result.value());
}

inline double
TitanDHAM::median_positive_spacing(const std::vector<double>& times_ps)
{
    std::vector<double> deltas;
    deltas.reserve(times_ps.size() > 1 ? times_ps.size() - 1 : 0);
    for (std::size_t i = 1; i < times_ps.size(); ++i) {
        const double dt = times_ps[i] - times_ps[i - 1];
        if (std::isfinite(dt) && dt > 0.0)
            deltas.push_back(dt);
    }
    if (deltas.empty())
        return 0.0;
    const std::size_t mid = deltas.size() / 2;
    std::nth_element(deltas.begin(),
                     deltas.begin() + static_cast<std::ptrdiff_t>(mid),
                     deltas.end());
    return deltas[mid];
}

inline TitanDHAM::WindowMatchResult
TitanDHAM::window_match_energy_to_dtraj(
    const std::vector<double>& dtraj_times_ps,
    const std::vector<std::pair<double,double>>& xvg_data,
    double dt_energy_ps)
{
    WindowMatchResult result;
    result.original_frames = dtraj_times_ps.size();
    if (dtraj_times_ps.empty() || xvg_data.empty())
        return result;

    auto infer_dt_ps = [&](double fallback) {
        if (fallback > 0.0)
            return fallback;

        std::vector<double> positive_deltas;
        positive_deltas.reserve(xvg_data.size() > 1 ? xvg_data.size() - 1 : 0);
        for (std::size_t i = 1; i < xvg_data.size(); ++i) {
            const double dt = xvg_data[i].first - xvg_data[i - 1].first;
            if (std::isfinite(dt) && dt > 0.0)
                positive_deltas.push_back(dt);
        }

        if (!positive_deltas.empty()) {
            const std::size_t mid = positive_deltas.size() / 2;
            std::nth_element(positive_deltas.begin(),
                             positive_deltas.begin() +
                                 static_cast<std::ptrdiff_t>(mid),
                             positive_deltas.end());
            return positive_deltas[mid];
        }
        return 2.0;
    };

    result.inferred_dt_energy_ps = infer_dt_ps(dt_energy_ps);
    if (!std::isfinite(result.inferred_dt_energy_ps) ||
        result.inferred_dt_energy_ps <= 0.0) {
        result.inferred_dt_energy_ps = 2.0;
    }
    result.match_tolerance_ps =
        0.5 * result.inferred_dt_energy_ps +
        1e-9 * std::max(1.0, std::abs(result.inferred_dt_energy_ps));

    const auto [traj_min_it, traj_max_it] =
        std::minmax_element(dtraj_times_ps.begin(), dtraj_times_ps.end());
    result.traj_min_ps = *traj_min_it;
    result.traj_max_ps = *traj_max_it;
    result.energy_min_ps = xvg_data.front().first;
    result.energy_max_ps = xvg_data.back().first;
    result.overlap_start_ps =
        std::max(result.traj_min_ps,
                 result.energy_min_ps - result.match_tolerance_ps);
    result.overlap_end_ps =
        std::min(result.traj_max_ps,
                 result.energy_max_ps + result.match_tolerance_ps);

    if (result.overlap_start_ps > result.overlap_end_ps)
        return result;

    result.overlap_found = true;
    result.kept_indices.reserve(dtraj_times_ps.size());
    result.kept_dtraj_times_ps.reserve(dtraj_times_ps.size());
    result.aligned_values.reserve(dtraj_times_ps.size());

    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    constexpr std::size_t no_row = static_cast<std::size_t>(-1);
    // KIN-21: kept position that holds each energy row (the nearest frame)
    std::vector<std::size_t> row_owner(xvg_data.size(), no_row);
    std::size_t j = 0;
    for (std::size_t i = 0; i < dtraj_times_ps.size(); ++i) {
        const double t_target = dtraj_times_ps[i];
        if (t_target < result.overlap_start_ps ||
            t_target > result.overlap_end_ps) {
            ++result.dropped_outside_overlap;
            continue;
        }

        if (j > 0 && xvg_data[j].first > t_target + result.match_tolerance_ps)
            j = 0;
        // KIN-19: a repeated energy row (two concatenated .xvg parts that both
        // hold t = 0) must not stop the scan: step over rows of equal time.
        while (j + 1 < xvg_data.size() &&
               (std::abs(xvg_data[j + 1].first - t_target) <
                    std::abs(xvg_data[j].first - t_target) ||
                xvg_data[j + 1].first == xvg_data[j].first)) {
            ++j;
        }

        result.kept_indices.push_back(i);
        result.kept_dtraj_times_ps.push_back(t_target);
        const double distance = std::abs(xvg_data[j].first - t_target);
        if (distance <= result.match_tolerance_ps) {
            // KIN-21: an energy row is one sample. When the energy grid is
            // coarser than the trajectory grid several frames fall within
            // tolerance of the same row; the nearest keeps it (the earlier on
            // a tie), the others get no energy.
            const std::size_t pos = result.kept_indices.size() - 1;
            const std::size_t owner = row_owner[j];
            if (owner == no_row) {
                row_owner[j] = pos;
                result.aligned_values.push_back(xvg_data[j].second);
                ++result.matched_frames;
            } else if (distance <
                       std::abs(xvg_data[j].first -
                                result.kept_dtraj_times_ps[owner])) {
                result.aligned_values[owner] = nan_val;
                row_owner[j] = pos;
                result.aligned_values.push_back(xvg_data[j].second);
                ++result.duplicate_energy_matches;
            } else {
                result.aligned_values.push_back(nan_val);
                ++result.duplicate_energy_matches;
            }
        } else {
            result.aligned_values.push_back(nan_val);
            ++result.unmatched_in_overlap;
        }
    }

    result.kept_frames = result.kept_indices.size();
    result.inferred_dt_traj_ps = median_positive_spacing(dtraj_times_ps);
    result.energy_coarser_than_traj =
        result.inferred_dt_traj_ps > 0.0 &&
        result.inferred_dt_energy_ps >
            result.inferred_dt_traj_ps * (1.0 + 1e-6);
    return result;
}

inline std::vector<double>
TitanDHAM::align_energy_to_dtraj(
    const std::vector<double>& dtraj_times_ps,
    const std::vector<std::pair<double,double>>& xvg_data,
    double dt_energy_ps)
{
    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> aligned(dtraj_times_ps.size(), nan_val);
    if (xvg_data.empty()) return aligned;

    const auto matched = window_match_energy_to_dtraj(
        dtraj_times_ps, xvg_data, dt_energy_ps);
    if (!matched.overlap_found) {
        TitanUI::print_warning(
            "DHAM: trajectory time range [" +
            std::to_string(matched.traj_min_ps) + ", " +
            std::to_string(matched.traj_max_ps) +
            "] ps has no overlap with energy range [" +
            std::to_string(matched.energy_min_ps) + ", " +
            std::to_string(matched.energy_max_ps) + "] ps.");
        return aligned;
    }

    for (std::size_t i = 0; i < matched.kept_indices.size(); ++i)
        aligned[matched.kept_indices[i]] = matched.aligned_values[i];

    if (matched.dropped_outside_overlap > 0) {
        TitanUI::print_warning(
            "DHAM: trajectory time range [" +
            std::to_string(matched.traj_min_ps) + ", " +
            std::to_string(matched.traj_max_ps) +
            "] ps extends outside energy range [" +
            std::to_string(matched.energy_min_ps) + ", " +
            std::to_string(matched.energy_max_ps) +
            "] ps. Frames outside the common overlap remain unmatched unless "
            "you enable DHAM window matching.");
    }

    std::size_t nan_count =
        matched.dropped_outside_overlap + matched.unmatched_in_overlap;
    if (nan_count > aligned.size() / 10) {
        TitanUI::print_warning(
            "DHAM: " + std::to_string(nan_count) + "/" +
            std::to_string(aligned.size()) +
            " trajectory frames could not be matched to energy values.");
    }
    return aligned;
}

inline std::vector<std::string>
TitanDHAM::read_file_list(const std::string& list_path) {
    std::vector<std::string> result;
    std::ifstream f(list_path);
    if (!f.is_open()) {
        TitanUI::print_error("DHAM: Cannot open file list: " + list_path);
        return result;
    }
    std::string line;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        result.push_back(line.substr(start, end - start + 1));
    }
    return result;
}

// KIN-23: lines that are blank or start with '#' or '@' (comments, xvg
// directives) are skipped; any other line must hold only temperatures
// (finite, > 0 K), separated by blanks or commas. Anything else stops the
// read with the file and line, and the result is empty (the callers then
// refuse to run). The old `f >> t` loop stopped silently at the first
// header or comment and returned the temperatures read so far.
inline std::vector<double>
TitanDHAM::read_temps(const std::string& temps_path) {
    std::vector<double> temps;
    std::ifstream f(temps_path);
    if (!f.is_open()) {
        TitanUI::print_error("Cannot open temperature file: " + temps_path);
        return temps;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(f, line)) {
        ++line_no;
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#' || line[first] == '@')
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            double t = 0.0;
            size_t used = 0;
            bool ok = true;
            try {
                t = std::stod(token, &used);
            } catch (const std::exception&) {
                ok = false;
            }
            if (!ok || used != token.size()) {
                TitanUI::print_error("Temperature file " + temps_path + ":" +
                                     std::to_string(line_no) + ": '" + token +
                                     "' is not a number");
                return {};
            }
            if (!std::isfinite(t) || t <= 0.0) {
                TitanUI::print_error("Temperature file " + temps_path + ":" +
                                     std::to_string(line_no) + ": temperature " +
                                     token + " must be finite and > 0 K");
                return {};
            }
            temps.push_back(t);
        }
    }
    return temps;
}

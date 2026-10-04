// ============================================================================
// TitanDHAM — Dynamic Histogram Analysis Method
// Reweights REMD/H-REMD MSM to a target temperature using energy information.
//
// Explicit small-bias DHAM approximation: Rosta & Hummer, JCTC 2015.
// State-mean energies approximate thermodynamic biases relative to target T;
// arbitrary H-REMD and cross-temperature physical rates require more information.
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
    };

    // -------------------------------------------------------------------------
    // Constructor
    //   n_states:   number of microstates (K-means clusters)
    //   n_replicas: number of REMD replicas
    //   temps:      temperature of each replica (K), length = n_replicas
    //   target_temp: temperature to reweight to (K)
    // -------------------------------------------------------------------------
    TitanDHAM(int n_states, int n_replicas,
               const std::vector<double>& temps,
               double target_temp = 300.0);

    // -------------------------------------------------------------------------
    // Data loading
    // -------------------------------------------------------------------------

    // Accumulate one transition i→j for replica k (called after K-means)
    void accumulate_counts(int replica_k, int state_i, int state_j);

    // Set the mean potential energy (kJ/mol) of microstate i at replica k
    void set_state_energy(int replica_k, int state_i, double mean_energy_kJ);

    // -------------------------------------------------------------------------
    // Solve self-consistent equations (WHAM-like iteration)
    //   max_iter:  maximum number of iterations
    //   tol:       convergence tolerance on free energy shifts f_k
    // Returns true if converged.
    // -------------------------------------------------------------------------
    bool solve(int max_iter = 5000, double tol = 1e-6);

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------

    // Stationary distribution at target temperature (length = n_states)
    const Eigen::VectorXd& get_stationary_distribution() const { return p_i_; }

    // Unbiased transition matrix at target temperature (n_states × n_states)
    Eigen::MatrixXd get_unbiased_transition_matrix() const;

    // Free energy per microstate at target temperature (kJ/mol, relative to min)
    Eigen::VectorXd get_free_energy_profile() const;

    // Per-replica free energy shifts f_k (dimensionless)
    const Eigen::VectorXd& get_free_energy_shifts() const { return f_k_; }

    // Compute per-frame weights W_t = p_i / N_i for each replica trajectory.
    std::unordered_map<int, std::vector<double>>
    get_frame_weights(
        const std::unordered_map<int, std::vector<int>>& per_replica_dtrajs) const;

    // Flatten per-replica frame weights in replica order 0..n_replicas-1.
    std::vector<double>
    get_flattened_frame_weights(
        const std::unordered_map<int, std::vector<int>>& per_replica_dtrajs,
        int n_replicas) const;

    // Export results to files
    // eu_par: energy unit divisor (1.0 = kJ/mol, 4.184 = kcal/mol)
    void write_results(const std::string& pi_file,
                       const std::string& fes_file,
                       const std::string& tpm_file,
                       std::fstream& log,
                       double eu_par = 1.0) const;

    // -------------------------------------------------------------------------
    // Static helpers — public so postproc code can call them
    // -------------------------------------------------------------------------

    // Parse GROMACS .xvg file → {time_ps, energy_kJ} pairs
    // Uses high-perf TitanIO::parse_xvg_time_value under the hood
    static std::vector<std::pair<double,double>>
    parse_xvg(const std::string& path);

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

private:
    int n_states_;
    int n_replicas_;
    double beta_target_;               // 1 / (kB * T_target)
    std::vector<double> betas_;        // 1 / (kB * T_k) for each replica

    // Sparse count maps — one per replica:
    // C_k_sparse_[k][i * n_states_ + j] = transitions i→j in replica k
    // This avoids storing dense n_states x n_states matrices for every replica.
    std::vector<std::unordered_map<std::uint64_t, double>> C_k_sparse_;

    // c_out_(i,k) = sum_j C_k(i,j) = row sums (outgoing counts)
    // Maintained online during accumulate_counts() and rebuilt from sparse maps
    // before solve() as a consistency check.
    MatrixXdRM  c_out_;              // [n_states × n_replicas]  RowMajor

    // u_ik_(i,k) = beta_k * U_k(i)  — dimensionless reduced potential
    MatrixXdRM  u_ik_;               // [n_states × n_replicas]  RowMajor

    Eigen::MatrixXd target_tpm_;      // accepted DHAM transition matrix
    Eigen::VectorXd f_k_;             // bias normalization; NaN if energies incomplete
    Eigen::VectorXd p_i_;             // stationary distribution at target T

    bool solved_ = false;

    // Rebuild c_out_ from C_k_ row sums
    void rebuild_c_out();

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
        while (j + 1 < xvg_data.size() &&
               std::abs(xvg_data[j + 1].first - t_target) <
               std::abs(xvg_data[j].first - t_target)) {
            ++j;
        }

        result.kept_indices.push_back(i);
        result.kept_dtraj_times_ps.push_back(t_target);
        if (std::abs(xvg_data[j].first - t_target) <= result.match_tolerance_ps) {
            result.aligned_values.push_back(xvg_data[j].second);
            ++result.matched_frames;
        } else {
            result.aligned_values.push_back(nan_val);
            ++result.unmatched_in_overlap;
        }
    }

    result.kept_frames = result.kept_indices.size();
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

inline std::vector<double>
TitanDHAM::read_temps(const std::string& temps_path) {
    std::vector<double> temps;
    std::ifstream f(temps_path);
    if (!f.is_open()) {
        TitanUI::print_error("DHAM: Cannot open temps file: " + temps_path);
        return temps;
    }
    double t;
    while (f >> t) {
        if (t <= 0.0) {
            TitanUI::print_error("DHAM: Invalid temperature <= 0: " + std::to_string(t));
            continue;
        }
        temps.push_back(t);
    }
    return temps;
}

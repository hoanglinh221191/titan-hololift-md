#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct LLPSLifetimeDistributionResult;

namespace titan_llps_stats {

struct LifetimeStats {
  double mean{0.0};
  double median{0.0};
  double p90{0.0};
  double max{0.0};
};

struct LineFitResult {
  bool ok{false};
  double slope{0.0};
  double intercept{0.0};
};

double percentile_sorted(const std::vector<double> &sorted, double frac);
LifetimeStats compute_lifetime_stats(const std::vector<double> &values);
LLPSLifetimeDistributionResult
build_lifetime_distribution(const std::vector<double> &values);
double fit_tau_curve(const std::vector<double> &curve,
                     const std::vector<double> &lags);
LineFitResult fit_line_window(const std::vector<double> &x,
                              const std::vector<double> &y,
                              const std::vector<uint64_t> &count,
                              double start_ns, double end_ns);
std::string join_ints(const std::vector<int> &values);
std::string join_doubles(const std::vector<double> &values);
std::string trim_copy(std::string value);
std::unordered_set<std::string> split_csv_set(const std::string &csv);
uint64_t pack_periodic_cell(int ix, int iy, int iz, int ny, int nz);
uint64_t pack_signed_cell(int ix, int iy, int iz);
uint64_t pack_u32_pair(int a, int b);
std::string join_ints_pipe(const std::vector<int> &values);
std::array<double, 3>
eigenvalues_symmetric_3x3(double xx, double xy, double xz, double yy,
                          double yz, double zz);
double pearson_corr(const std::vector<double> &x,
                    const std::vector<double> &y);
std::vector<double> average_tie_ranks(const std::vector<double> &values);
double spearman_corr(const std::vector<double> &x,
                     const std::vector<double> &y);
double entropy_bits_from_counts(const std::unordered_map<int, uint64_t> &counts,
                                uint64_t total);
double normalized_entropy_bits(double entropy_bits,
                               std::size_t observed_states);
double conditional_entropy_bits(
    const std::unordered_map<int, uint64_t> &x_counts,
    const std::unordered_map<uint64_t, uint64_t> &joint_counts,
    uint64_t total);
double mutual_information_bits(
    const std::unordered_map<int, uint64_t> &x_counts,
    const std::unordered_map<int, uint64_t> &y_counts,
    const std::unordered_map<uint64_t, uint64_t> &joint_counts,
    uint64_t total);
double partner_entropy_given_binding_bits(
    const std::unordered_map<int, std::unordered_map<int, uint64_t>>
        &partner_counts_by_binding,
    uint64_t contacting_samples);
std::vector<double> make_uniform_edges(double min_value, double max_value,
                                       int bins);
int locate_uniform_bin(double value, const std::vector<double> &edges);
std::string format_optional_csv(double value, int precision = 6);

} // namespace titan_llps_stats

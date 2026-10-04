#include "titan_llps_stats.h"
#include "titan_llps.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace titan_llps_stats {

double percentile_sorted(const std::vector<double> &sorted, double frac) {
  if (sorted.empty())
    return 0.0;
  if (sorted.size() == 1)
    return sorted.front();
  const double pos = frac * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double alpha = pos - static_cast<double>(lo);
  return sorted[lo] * (1.0 - alpha) + sorted[hi] * alpha;
}

LifetimeStats compute_lifetime_stats(const std::vector<double> &values) {
  LifetimeStats stats;
  if (values.empty())
    return stats;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  stats.max = sorted.back();
  stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
               static_cast<double>(sorted.size());
  stats.median = percentile_sorted(sorted, 0.5);
  stats.p90 = percentile_sorted(sorted, 0.9);
  return stats;
}

LLPSLifetimeDistributionResult
build_lifetime_distribution(const std::vector<double> &values) {
  LLPSLifetimeDistributionResult result;
  if (values.empty())
    return result;
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  result.total_events = static_cast<uint64_t>(sorted.size());
  const LifetimeStats stats = compute_lifetime_stats(sorted);
  result.mean_ns = stats.mean;
  result.median_ns = stats.median;
  result.p90_ns = stats.p90;
  result.max_ns = stats.max;
  constexpr double kTol = 1e-9;
  for (double value : sorted) {
    if (!result.lifetime_ns.empty() &&
        std::abs(result.lifetime_ns.back() - value) < kTol) {
      ++result.counts.back();
      continue;
    }
    result.lifetime_ns.push_back(value);
    result.counts.push_back(1);
  }
  return result;
}

double fit_tau_curve(const std::vector<double> &curve,
                     const std::vector<double> &lags) {
  if (curve.size() < 2 || curve.size() != lags.size())
    return 0.0;
  constexpr double inv_e = 1.0 / 2.71828182845904523536;
  for (std::size_t i = 1; i < curve.size(); ++i) {
    if (curve[i] <= inv_e && curve[i - 1] > curve[i]) {
      const double frac = (curve[i - 1] - inv_e) / (curve[i - 1] - curve[i]);
      return lags[i - 1] + frac * (lags[i] - lags[i - 1]);
    }
  }
  double tau = 0.0;
  for (std::size_t i = 1; i < curve.size(); ++i)
    tau += 0.5 * (curve[i - 1] + curve[i]) * (lags[i] - lags[i - 1]);
  return tau;
}

LineFitResult fit_line_window(const std::vector<double> &x,
                              const std::vector<double> &y,
                              const std::vector<uint64_t> &count,
                              double start_ns, double end_ns) {
  long double sum_x = 0.0L;
  long double sum_y = 0.0L;
  long double sum_xx = 0.0L;
  long double sum_xy = 0.0L;
  long double n = 0.0L;
  for (std::size_t i = 0; i < x.size() && i < y.size() && i < count.size();
       ++i) {
    if (count[i] == 0)
      continue;
    if (x[i] < start_ns || x[i] > end_ns)
      continue;
    sum_x += x[i];
    sum_y += y[i];
    sum_xx += x[i] * x[i];
    sum_xy += x[i] * y[i];
    n += 1.0L;
  }
  LineFitResult fit;
  const long double denom = n * sum_xx - sum_x * sum_x;
  if (n < 2.0L || std::abs(static_cast<double>(denom)) < 1e-12)
    return fit;
  fit.ok = true;
  fit.slope = static_cast<double>((n * sum_xy - sum_x * sum_y) / denom);
  fit.intercept = static_cast<double>((sum_y - fit.slope * sum_x) / n);
  return fit;
}

std::string join_ints(const std::vector<int> &values) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      oss << ';';
    oss << values[i];
  }
  return oss.str();
}

std::string join_doubles(const std::vector<double> &values) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      oss << ';';
    oss << values[i];
  }
  return oss.str();
}

std::string trim_copy(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

std::unordered_set<std::string> split_csv_set(const std::string &csv) {
  std::unordered_set<std::string> out;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (!token.empty())
      out.insert(token);
  }
  return out;
}

uint64_t pack_periodic_cell(int ix, int iy, int iz, int ny, int nz) {
  return (static_cast<uint64_t>(ix) * static_cast<uint64_t>(ny) +
          static_cast<uint64_t>(iy)) *
             static_cast<uint64_t>(nz) +
         static_cast<uint64_t>(iz);
}

uint64_t pack_signed_cell(int ix, int iy, int iz) {
  constexpr int bias = 1 << 20;
  constexpr uint64_t mask = (1ULL << 21) - 1ULL;
  return ((static_cast<uint64_t>(ix + bias) & mask) << 42) |
         ((static_cast<uint64_t>(iy + bias) & mask) << 21) |
         (static_cast<uint64_t>(iz + bias) & mask);
}

uint64_t pack_u32_pair(int a, int b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}

std::string join_ints_pipe(const std::vector<int> &values) {
  if (values.empty())
    return "unbound";
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0)
      oss << '|';
    oss << values[i];
  }
  return oss.str();
}

std::array<double, 3>
eigenvalues_symmetric_3x3(double xx, double xy, double xz, double yy,
                          double yz, double zz) {
  const double trace = xx + yy + zz;
  const double q = trace / 3.0;
  const double ax = xx - q;
  const double ay = yy - q;
  const double az = zz - q;
  const double p1 = xy * xy + xz * xz + yz * yz;
  std::array<double, 3> eig{};
  if (p1 <= 1e-24) {
    eig = {xx, yy, zz};
  } else {
    const double p2 = ax * ax + ay * ay + az * az + 2.0 * p1;
    const double p = std::sqrt(std::max(0.0, p2 / 6.0));
    const double inv_p = (p > 0.0) ? (1.0 / p) : 0.0;
    const double bxx = ax * inv_p;
    const double bxy = xy * inv_p;
    const double bxz = xz * inv_p;
    const double byy = ay * inv_p;
    const double byz = yz * inv_p;
    const double bzz = az * inv_p;
    const double det_b =
        bxx * (byy * bzz - byz * byz) -
        bxy * (bxy * bzz - byz * bxz) +
        bxz * (bxy * byz - byy * bxz);
    const double r = std::max(-1.0, std::min(1.0, det_b / 2.0));
    const double phi = std::acos(r) / 3.0;
    constexpr double two_pi = 6.28318530717958647692;
    eig[0] = q + 2.0 * p * std::cos(phi);
    eig[2] = q + 2.0 * p * std::cos(phi + two_pi / 3.0);
    eig[1] = 3.0 * q - eig[0] - eig[2];
  }
  std::sort(eig.begin(), eig.end(), std::greater<double>());
  for (double &v : eig)
    if (v < 0.0 && v > -1e-10)
      v = 0.0;
  return eig;
}

double pearson_corr(const std::vector<double> &x,
                    const std::vector<double> &y) {
  if (x.size() != y.size() || x.size() < 2)
    return std::numeric_limits<double>::quiet_NaN();
  long double sx = 0.0L;
  long double sy = 0.0L;
  long double sxx = 0.0L;
  long double syy = 0.0L;
  long double sxy = 0.0L;
  for (std::size_t i = 0; i < x.size(); ++i) {
    sx += x[i];
    sy += y[i];
    sxx += x[i] * x[i];
    syy += y[i] * y[i];
    sxy += x[i] * y[i];
  }
  const long double n = static_cast<long double>(x.size());
  const long double denom_x = n * sxx - sx * sx;
  const long double denom_y = n * syy - sy * sy;
  if (denom_x <= 0.0L || denom_y <= 0.0L)
    return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>((n * sxy - sx * sy) /
                             std::sqrt(denom_x * denom_y));
}

std::vector<double> average_tie_ranks(const std::vector<double> &values) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return values[a] < values[b];
  });
  std::vector<double> ranks(values.size(), 0.0);
  std::size_t i = 0;
  while (i < order.size()) {
    std::size_t j = i + 1;
    while (j < order.size() && values[order[j]] == values[order[i]])
      ++j;
    const double rank =
        0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
    for (std::size_t k = i; k < j; ++k)
      ranks[order[k]] = rank;
    i = j;
  }
  return ranks;
}

double spearman_corr(const std::vector<double> &x,
                     const std::vector<double> &y) {
  if (x.size() != y.size() || x.size() < 2)
    return std::numeric_limits<double>::quiet_NaN();
  return pearson_corr(average_tie_ranks(x), average_tie_ranks(y));
}

double entropy_bits_from_counts(const std::unordered_map<int, uint64_t> &counts,
                                uint64_t total) {
  if (total == 0)
    return 0.0;
  long double h = 0.0L;
  for (const auto &entry : counts) {
    if (entry.second == 0)
      continue;
    const long double p =
        static_cast<long double>(entry.second) / static_cast<long double>(total);
    h -= p * std::log2(p);
  }
  return static_cast<double>(h);
}

double normalized_entropy_bits(double entropy_bits,
                               std::size_t observed_states) {
  if (observed_states <= 1 || entropy_bits <= 0.0)
    return 0.0;
  const double denom = std::log2(static_cast<double>(observed_states));
  if (!(denom > 0.0))
    return 0.0;
  return entropy_bits / denom;
}

double conditional_entropy_bits(
    const std::unordered_map<int, uint64_t> &x_counts,
    const std::unordered_map<uint64_t, uint64_t> &joint_counts,
    uint64_t total) {
  if (total == 0)
    return 0.0;
  long double h = 0.0L;
  for (const auto &entry : joint_counts) {
    if (entry.second == 0)
      continue;
    const int x = static_cast<int>(entry.first >> 32);
    const auto x_it = x_counts.find(x);
    if (x_it == x_counts.end() || x_it->second == 0)
      continue;
    const long double p_xy =
        static_cast<long double>(entry.second) / static_cast<long double>(total);
    const long double p_x =
        static_cast<long double>(x_it->second) / static_cast<long double>(total);
    h -= p_xy * std::log2(p_xy / p_x);
  }
  return static_cast<double>(h);
}

double mutual_information_bits(
    const std::unordered_map<int, uint64_t> &x_counts,
    const std::unordered_map<int, uint64_t> &y_counts,
    const std::unordered_map<uint64_t, uint64_t> &joint_counts,
    uint64_t total) {
  if (total == 0)
    return 0.0;
  long double mi = 0.0L;
  for (const auto &entry : joint_counts) {
    if (entry.second == 0)
      continue;
    const int x = static_cast<int>(entry.first >> 32);
    const int y = static_cast<int>(entry.first & 0xffffffffULL);
    const auto x_it = x_counts.find(x);
    const auto y_it = y_counts.find(y);
    if (x_it == x_counts.end() || y_it == y_counts.end() ||
        x_it->second == 0 || y_it->second == 0)
      continue;
    const long double p_xy =
        static_cast<long double>(entry.second) / static_cast<long double>(total);
    const long double p_x =
        static_cast<long double>(x_it->second) / static_cast<long double>(total);
    const long double p_y =
        static_cast<long double>(y_it->second) / static_cast<long double>(total);
    mi += p_xy * std::log2(p_xy / (p_x * p_y));
  }
  return static_cast<double>(mi);
}

double partner_entropy_given_binding_bits(
    const std::unordered_map<int, std::unordered_map<int, uint64_t>>
        &partner_counts_by_binding,
    uint64_t contacting_samples) {
  if (contacting_samples == 0)
    return 0.0;
  long double weighted = 0.0L;
  for (const auto &binding_entry : partner_counts_by_binding) {
    uint64_t binding_total = 0;
    for (const auto &partner_entry : binding_entry.second)
      binding_total += partner_entry.second;
    if (binding_total == 0)
      continue;
    const long double weight = static_cast<long double>(binding_total) /
                               static_cast<long double>(contacting_samples);
    weighted += weight * static_cast<long double>(
                             entropy_bits_from_counts(binding_entry.second,
                                                      binding_total));
  }
  return static_cast<double>(weighted);
}

std::vector<double> make_uniform_edges(double min_value, double max_value,
                                       int bins) {
  const int safe_bins = std::max(1, bins);
  if (!std::isfinite(min_value))
    min_value = 0.0;
  if (!std::isfinite(max_value))
    max_value = min_value;
  if (!(max_value > min_value)) {
    const double delta = (std::abs(min_value) > 1e-6)
                             ? 1e-6 * std::abs(min_value)
                             : 1e-6;
    min_value -= delta;
    max_value += delta;
  }
  std::vector<double> edges(static_cast<std::size_t>(safe_bins + 1),
                            min_value);
  const double step = (max_value - min_value) / static_cast<double>(safe_bins);
  for (int i = 0; i <= safe_bins; ++i)
    edges[static_cast<std::size_t>(i)] =
        min_value + step * static_cast<double>(i);
  edges.back() = max_value;
  return edges;
}

int locate_uniform_bin(double value, const std::vector<double> &edges) {
  if (edges.size() < 2)
    return 0;
  if (value <= edges.front())
    return 0;
  if (value >= edges.back())
    return static_cast<int>(edges.size()) - 2;
  const auto upper = std::upper_bound(edges.begin(), edges.end(), value);
  const auto idx = static_cast<int>(std::distance(edges.begin(), upper)) - 1;
  return std::max(0, std::min(idx, static_cast<int>(edges.size()) - 2));
}

std::string format_optional_csv(double value, int precision) {
  if (!std::isfinite(value))
    return "NA";
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

} // namespace titan_llps_stats

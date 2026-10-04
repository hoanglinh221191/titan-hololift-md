#include "titan_mbar.h"

#include "aligned_allocator.h"
#include "titan_dham.h"
#include "titan_executor.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>

namespace {

constexpr double kBoltzmannKjPerMolK = 0.008314462618;
constexpr double kLogGuard = 1e-20;
constexpr double kTimeFilterTolerancePs = 1e-3;
constexpr int kSciIterations = 200;
constexpr int kSciIterationsFast = 12;
constexpr int kNewtonIterations = 40;
constexpr int kNewtonIterationsFast = 16;
constexpr int kBarIterations = 100;
constexpr int kLineSearchMaxBacktracks = 12;
constexpr double kResidualConvTol = 1e-8;
constexpr double kFixedPointStepTol = 1e-8;
constexpr double kNewtonStepClamp = 10.0;
constexpr double kAdjacentOverlapWarn = 0.03;
constexpr double kMinStateEssWarn = 5.0;
constexpr double kMinTargetEssFractionWarn = 0.20;
constexpr std::size_t kAvxDoubleWidth = 4;

struct ReplicaSeries {
  int replica_id{-1};
  int fixed_state_index{-1};
  double fixed_temperature_k{0.0};
  std::vector<double> energies{};
  std::vector<std::size_t> local_indices{};
  std::vector<double> times_ps{};
  std::vector<int> sampled_state_indices{};
  std::vector<double> sampled_temperatures_k{};
};

struct PreparedSeries {
  int replica_id{-1};
  std::vector<double> observables{};
  std::vector<std::size_t> local_indices{};
  std::vector<double> times_ps{};
  std::vector<int> sampled_state_indices{};
  std::vector<double> sampled_temperatures_k{};
  std::vector<int64_t> global_indices{};
};

struct SampleOrigin {
  int replica_id{-1};
  std::size_t local_frame_index{0};
  double time_ps{0.0};
  int sampled_state_index{-1};
  double sampled_temperature_k{0.0};
  int64_t global_frame_index{-1};
};

struct ObservableResult {
  double mean_energy{0.0};
  double heat_capacity{0.0};
  AlignedDoubleVector normalized_weights{};
};

struct SolverEvaluation {
  std::vector<double> residual{};
  std::vector<double> hessian{};
  std::vector<double> square_sums{};
  AlignedDoubleVector log_denom{};
  double residual_inf{0.0};
  double residual_rms{0.0};
  double objective{0.0};
};

struct SolverStats {
  std::string method{"not_run"};
  std::size_t iterations{0};
  std::size_t line_search_backtracks{0};
  double residual_inf{0.0};
  double residual_rms{0.0};
};

struct RunningStats {
  std::size_t count{0};
  double mean{0.0};
  double m2{0.0};

  void add(double value) {
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    const double delta2 = value - mean;
    m2 += delta * delta2;
  }

  [[nodiscard]] double stddev() const {
    if (count < 2)
      return 0.0;
    return std::sqrt(std::max(0.0, m2 / static_cast<double>(count - 1)));
  }
};

struct TimeSelectionSummary {
  std::size_t count{0};
  double min_ps{0.0};
  double max_ps{0.0};
};

[[nodiscard]] bool keep_mbar_energy_time(const arguments &args,
                                         double time_ps) noexcept {
  if (args.flag_begin_set && time_ps < args.begin_time - kTimeFilterTolerancePs)
    return false;
  if (args.flag_end_set && time_ps > args.end_time + kTimeFilterTolerancePs)
    return false;
  if (args.dt > 0.0) {
    const double ref_time = args.flag_begin_set ? args.begin_time : 0.0;
    const double diff = time_ps - ref_time;
    if (diff < -kTimeFilterTolerancePs)
      return false;
    const long long steps = std::lround(diff / args.dt);
    if (std::abs(diff - static_cast<double>(steps) * args.dt) >
        kTimeFilterTolerancePs) {
      return false;
    }
  }
  return true;
}

template <typename Row, typename TimeAccessor>
TimeSelectionSummary summarize_selected_times(const std::vector<Row> &rows,
                                              TimeAccessor time_accessor,
                                              const arguments &args) {
  TimeSelectionSummary summary;
  for (const auto &row : rows) {
    const double time_ps = time_accessor(row);
    if (!keep_mbar_energy_time(args, time_ps))
      continue;
    if (summary.count == 0) {
      summary.min_ps = time_ps;
      summary.max_ps = time_ps;
    } else {
      summary.min_ps = std::min(summary.min_ps, time_ps);
      summary.max_ps = std::max(summary.max_ps, time_ps);
    }
    ++summary.count;
  }
  return summary;
}

std::vector<std::pair<double, double>>
filter_time_value_pairs(const std::vector<std::pair<double, double>> &rows,
                        const arguments &args) {
  std::vector<std::pair<double, double>> filtered;
  filtered.reserve(rows.size());
  for (const auto &row : rows) {
    if (!keep_mbar_energy_time(args, row.first))
      continue;
    filtered.push_back(row);
  }
  return filtered;
}

bool read_mbar_demux_map(const std::string &path,
                         std::vector<std::vector<int>> &rows_out,
                         std::string &error_message) {
  rows_out.clear();
  std::ifstream in(path);
  if (!in.is_open()) {
    error_message = "cannot open demux map file: " + path;
    return false;
  }

  std::string line;
  std::size_t expected_cols = 0;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos)
      line.erase(hash_pos);
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream iss(line);
    std::vector<int> row;
    int value = 0;
    while (iss >> value)
      row.push_back(value);
    if (row.empty())
      continue;
    if (expected_cols == 0) {
      expected_cols = row.size();
    } else if (row.size() != expected_cols) {
      error_message = "inconsistent demux-map column count at line " +
                      std::to_string(line_no);
      return false;
    }
    rows_out.push_back(std::move(row));
  }

  if (rows_out.empty()) {
    error_message = "demux map is empty";
    return false;
  }
  return true;
}

std::unordered_map<int, std::vector<int>>
build_per_replica_state_assignments(const std::vector<std::vector<int>> &rows,
                                    std::size_t n_states) {
  std::unordered_map<int, std::vector<int>> result;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    const auto &values = rows[row];
    for (std::size_t state_idx = 0; state_idx < values.size() &&
                                   state_idx < n_states;
         ++state_idx) {
      const int replica_id = values[state_idx];
      if (replica_id < 0)
        continue;
      auto &assignment = result[replica_id];
      if (assignment.size() < rows.size())
        assignment.assign(rows.size(), -1);
      assignment[row] = static_cast<int>(state_idx);
    }
  }
  return result;
}

int map_temperature_to_state(const std::vector<double> &temps,
                             double observed_temperature_k) {
  if (temps.empty() || !std::isfinite(observed_temperature_k))
    return -1;
  std::size_t best_idx = 0;
  double best_diff = std::abs(temps.front() - observed_temperature_k);
  for (std::size_t idx = 1; idx < temps.size(); ++idx) {
    const double diff = std::abs(temps[idx] - observed_temperature_k);
    if (diff < best_diff) {
      best_diff = diff;
      best_idx = idx;
    }
  }
  return static_cast<int>(best_idx);
}

std::vector<double>
infer_ladder_temperatures_from_observed_series(
    const std::vector<std::vector<TitanIO::XvgTimeTwoValues>> &series_by_replica,
    std::size_t n_states, const arguments &args, std::string &error_message) {
  error_message.clear();
  if (n_states == 0) {
    error_message = "cannot infer MBAR ladder temperatures for zero states";
    return {};
  }

  std::vector<double> observed_temperatures;
  for (const auto &series : series_by_replica) {
    for (const auto &row : series) {
      if (!keep_mbar_energy_time(args, row.time_ps))
        continue;
      if (std::isfinite(row.value2))
        observed_temperatures.push_back(row.value2);
    }
  }
  if (observed_temperatures.size() < n_states) {
    error_message =
        "not enough finite per-frame temperatures to infer the MBAR ladder";
    return {};
  }

  std::sort(observed_temperatures.begin(), observed_temperatures.end());
  std::vector<double> centers(n_states, 0.0);
  for (std::size_t state_idx = 0; state_idx < n_states; ++state_idx) {
    const std::size_t pos = static_cast<std::size_t>(
        (static_cast<double>(state_idx) + 0.5) *
        static_cast<double>(observed_temperatures.size() - 1) /
        static_cast<double>(n_states));
    centers[state_idx] = observed_temperatures[pos];
  }

  std::vector<double> updated_centers(n_states, 0.0);
  std::vector<int> counts(n_states, 0);
  for (int iter = 0; iter < 48; ++iter) {
    std::fill(updated_centers.begin(), updated_centers.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (double temperature_k : observed_temperatures) {
      std::size_t best_idx = 0;
      double best_diff = std::abs(temperature_k - centers.front());
      for (std::size_t state_idx = 1; state_idx < centers.size(); ++state_idx) {
        const double diff = std::abs(temperature_k - centers[state_idx]);
        if (diff < best_diff) {
          best_diff = diff;
          best_idx = state_idx;
        }
      }
      updated_centers[best_idx] += temperature_k;
      counts[best_idx] += 1;
    }

    double max_shift = 0.0;
    for (std::size_t state_idx = 0; state_idx < n_states; ++state_idx) {
      double new_center = centers[state_idx];
      if (counts[state_idx] > 0) {
        new_center =
            updated_centers[state_idx] / static_cast<double>(counts[state_idx]);
      } else {
        const std::size_t pos = static_cast<std::size_t>(
            (static_cast<double>(state_idx) + 0.5) *
            static_cast<double>(observed_temperatures.size() - 1) /
            static_cast<double>(n_states));
        new_center = observed_temperatures[pos];
      }
      max_shift = std::max(max_shift, std::abs(new_center - centers[state_idx]));
      centers[state_idx] = new_center;
    }
    std::sort(centers.begin(), centers.end());
    if (max_shift < 1e-8)
      break;
  }

  if (!std::all_of(centers.begin(), centers.end(),
                   [](double value) { return std::isfinite(value); })) {
    error_message = "inferred MBAR ladder contains non-finite temperatures";
    return {};
  }
  return centers;
}

std::vector<double> infer_ladder_temperatures_from_demux(
    const std::vector<std::vector<TitanIO::XvgTimeTwoValues>> &series_by_replica,
    const std::unordered_map<int, std::vector<int>> &per_replica_assignments,
    std::size_t n_states, const arguments &args, std::string &error_message) {
  error_message.clear();
  if (n_states == 0) {
    error_message = "cannot infer MBAR ladder temperatures for zero states";
    return {};
  }

  std::vector<double> state_sums(n_states, 0.0);
  std::vector<int> state_counts(n_states, 0);
  for (std::size_t replica_idx = 0; replica_idx < series_by_replica.size();
       ++replica_idx) {
    const auto assignment_it =
        per_replica_assignments.find(static_cast<int>(replica_idx));
    if (assignment_it == per_replica_assignments.end())
      continue;
    const auto &series = series_by_replica[replica_idx];
    const auto &assignments = assignment_it->second;
    const std::size_t n_rows = std::min(series.size(), assignments.size());
    for (std::size_t row_idx = 0; row_idx < n_rows; ++row_idx) {
      if (!keep_mbar_energy_time(args, series[row_idx].time_ps))
        continue;
      const int state_index = assignments[row_idx];
      if (state_index < 0 ||
          static_cast<std::size_t>(state_index) >= n_states ||
          !std::isfinite(series[row_idx].value2)) {
        continue;
      }
      state_sums[static_cast<std::size_t>(state_index)] += series[row_idx].value2;
      state_counts[static_cast<std::size_t>(state_index)] += 1;
    }
  }

  std::vector<double> temperatures(n_states, 0.0);
  for (std::size_t state_idx = 0; state_idx < n_states; ++state_idx) {
    if (state_counts[state_idx] == 0) {
      error_message =
          "demux-aware MBAR could not infer one or more ladder temperatures "
          "from per-frame replica temperatures";
      return {};
    }
    temperatures[state_idx] =
        state_sums[state_idx] / static_cast<double>(state_counts[state_idx]);
  }
  return temperatures;
}

std::vector<std::pair<double, double>>
make_time_value_pairs(const std::vector<TitanIO::XvgTimeTwoValues> &rows,
                      bool use_second_value, const arguments &args) {
  std::vector<std::pair<double, double>> result;
  result.reserve(rows.size());
  for (const auto &row : rows) {
    if (!keep_mbar_energy_time(args, row.time_ps))
      continue;
    result.emplace_back(row.time_ps, use_second_value ? row.value2 : row.value1);
  }
  return result;
}

std::vector<double>
materialize_aligned_values(std::size_t size,
                           const TitanDHAM::WindowMatchResult &matched) {
  const double nan_val = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> aligned(size, nan_val);
  for (std::size_t i = 0; i < matched.kept_indices.size() &&
                          i < matched.aligned_values.size();
       ++i) {
    if (matched.kept_indices[i] < aligned.size())
      aligned[matched.kept_indices[i]] = matched.aligned_values[i];
  }
  return aligned;
}

[[nodiscard]] const char *observable_symbol(bool use_npt) {
  return use_npt ? "H" : "E";
}

[[nodiscard]] const char *observable_name(bool use_npt) {
  return use_npt ? "enthalpy-like" : "energy";
}

bool detect_irregular_energy_dt(
    const std::vector<std::pair<double, double>> &xvg_data,
    double inferred_dt_ps) {
  if (xvg_data.size() < 4 || !std::isfinite(inferred_dt_ps) ||
      inferred_dt_ps <= 0.0) {
    return false;
  }
  int positive_deltas = 0;
  int irregular = 0;
  const double tol = std::max(1e-6, 0.25 * inferred_dt_ps);
  for (std::size_t i = 1; i < xvg_data.size(); ++i) {
    const double dt = xvg_data[i].first - xvg_data[i - 1].first;
    if (!(dt > 0.0))
      continue;
    ++positive_deltas;
    if (std::abs(dt - inferred_dt_ps) > tol)
      ++irregular;
  }
  return positive_deltas > 0 &&
         static_cast<double>(irregular) / static_cast<double>(positive_deltas) >
             0.20;
}

std::vector<double> build_temperature_scan(const std::vector<double> &temps,
                                           double target_temp,
                                           int scan_points) {
  std::vector<double> targets;
  if (temps.empty())
    return targets;

  if (scan_points > 1) {
    const auto [tmin_it, tmax_it] =
        std::minmax_element(temps.begin(), temps.end());
    const double t_min = *tmin_it;
    const double t_max = *tmax_it;
    if (scan_points == 2 || std::abs(t_max - t_min) < 1e-12) {
      targets.push_back(t_min);
      if (t_max != t_min)
        targets.push_back(t_max);
    } else {
      const double dt =
          (t_max - t_min) / static_cast<double>(std::max(scan_points - 1, 1));
      for (int i = 0; i < scan_points; ++i)
        targets.push_back(t_min + dt * static_cast<double>(i));
    }
  }

  if (targets.empty())
    targets.push_back(target_temp);

  bool have_target = false;
  for (double value : targets) {
    if (std::abs(value - target_temp) <= 1e-9 * std::max(1.0, target_temp)) {
      have_target = true;
      break;
    }
  }
  if (!have_target)
    targets.push_back(target_temp);

  std::sort(targets.begin(), targets.end());
  targets.erase(std::unique(targets.begin(), targets.end(),
                            [](double lhs, double rhs) {
                              return std::abs(lhs - rhs) <=
                                     1e-9 * std::max({1.0, std::abs(lhs),
                                                      std::abs(rhs)});
                            }),
                targets.end());
  return targets;
}

void normalize_free_energies(std::vector<double> &f_k) {
  if (f_k.empty())
    return;
  const double offset = f_k.front();
  for (double &value : f_k)
    value -= offset;
}

#if defined(__AVX2__)
inline __m256d mbar_linear_terms_avx2(__m256d log_term, __m256d beta_term,
                                      __m256d energy_term) {
#if defined(__FMA__)
  return _mm256_fnmadd_pd(beta_term, energy_term, log_term);
#else
  return _mm256_sub_pd(log_term, _mm256_mul_pd(beta_term, energy_term));
#endif
}

double horizontal_max_avx2(__m256d value) {
  alignas(32) double lanes[kAvxDoubleWidth];
  _mm256_store_pd(lanes, value);
  return std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3]));
}
#endif

double fill_mbar_linear_terms(double energy, const std::vector<double> &beta_k,
                              const std::vector<double> &log_nk_plus_fk,
                              double *terms_out) {
  const std::size_t k_count = beta_k.size();
  double max_v = -std::numeric_limits<double>::infinity();
  std::size_t k = 0;

#if defined(__AVX2__)
  if (k_count >= kAvxDoubleWidth) {
    const __m256d v_energy = _mm256_set1_pd(energy);
    __m256d v_max = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const std::size_t simd_end = (k_count / kAvxDoubleWidth) * kAvxDoubleWidth;
    for (; k < simd_end; k += kAvxDoubleWidth) {
      const __m256d v_log = _mm256_loadu_pd(log_nk_plus_fk.data() + k);
      const __m256d v_beta = _mm256_loadu_pd(beta_k.data() + k);
      const __m256d v_terms = mbar_linear_terms_avx2(v_log, v_beta, v_energy);
      _mm256_storeu_pd(terms_out + k, v_terms);
      v_max = _mm256_max_pd(v_max, v_terms);
    }
    max_v = horizontal_max_avx2(v_max);
  }
#endif

  for (; k < k_count; ++k) {
    const double value = log_nk_plus_fk[k] - beta_k[k] * energy;
    terms_out[k] = value;
    if (value > max_v)
      max_v = value;
  }
  return max_v;
}

double sum_shifted_exponentials(const double *terms, std::size_t count,
                                double max_v, double *unnormalized_out) {
  double sum = 0.0;
  for (std::size_t k = 0; k < count; ++k) {
    const double value = std::exp(terms[k] - max_v);
    if (unnormalized_out)
      unnormalized_out[k] = value;
    sum += value;
  }
  return sum;
}

void normalize_weights_accumulate(const double *unnormalized_weights,
                                  std::size_t count, double inv_denom,
                                  double *normalized_out, double *residual_out,
                                  double *square_sum_out) {
  std::size_t k = 0;
#if defined(__AVX2__)
  if (count >= kAvxDoubleWidth) {
    const __m256d v_inv = _mm256_set1_pd(inv_denom);
    const std::size_t simd_end = (count / kAvxDoubleWidth) * kAvxDoubleWidth;
    for (; k < simd_end; k += kAvxDoubleWidth) {
      const __m256d v_unnorm = _mm256_loadu_pd(unnormalized_weights + k);
      const __m256d v_weight = _mm256_mul_pd(v_unnorm, v_inv);
      _mm256_storeu_pd(normalized_out + k, v_weight);
      if (residual_out) {
        const __m256d v_res = _mm256_loadu_pd(residual_out + k);
        _mm256_storeu_pd(residual_out + k, _mm256_sub_pd(v_res, v_weight));
      }
      if (square_sum_out) {
        const __m256d v_sq = _mm256_loadu_pd(square_sum_out + k);
#if defined(__FMA__)
        _mm256_storeu_pd(square_sum_out + k,
                         _mm256_fmadd_pd(v_weight, v_weight, v_sq));
#else
        _mm256_storeu_pd(square_sum_out + k,
                         _mm256_add_pd(v_sq, _mm256_mul_pd(v_weight, v_weight)));
#endif
      }
    }
  }
#endif
  for (; k < count; ++k) {
    const double weight = unnormalized_weights[k] * inv_denom;
    normalized_out[k] = weight;
    if (residual_out)
      residual_out[k] -= weight;
    if (square_sum_out)
      square_sum_out[k] += weight * weight;
  }
}

void guess_free_energy_ti(const std::vector<double> &beta_k,
                          const std::vector<double> &energy_means,
                          std::vector<double> &f_k, double energy_min) {
  const int k_count = static_cast<int>(beta_k.size());
  std::vector<int> order(static_cast<std::size_t>(k_count));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return beta_k[static_cast<std::size_t>(lhs)] <
           beta_k[static_cast<std::size_t>(rhs)];
  });

  if (order.empty())
    return;
  f_k[static_cast<std::size_t>(order.front())] = 0.0;
  for (int idx = 0; idx + 1 < k_count; ++idx) {
    const int current = order[static_cast<std::size_t>(idx)];
    const int next = order[static_cast<std::size_t>(idx + 1)];
    const double d_beta =
        beta_k[static_cast<std::size_t>(next)] -
        beta_k[static_cast<std::size_t>(current)];
    const double mean_e =
        0.5 * ((energy_means[static_cast<std::size_t>(current)] - energy_min) +
               (energy_means[static_cast<std::size_t>(next)] - energy_min));
    f_k[static_cast<std::size_t>(next)] =
        f_k[static_cast<std::size_t>(current)] + d_beta * mean_e;
  }
}

void compute_log_denom(const AlignedDoubleVector &shifted_samples,
                       const std::vector<double> &beta_k,
                       const std::vector<double> &log_nk_plus_fk,
                       TitanExecutor &executor,
                       AlignedDoubleVector &log_denom) {
  const std::size_t n_samples = shifted_samples.size();
  const std::size_t k_count = beta_k.size();
  log_denom.assign(n_samples, 0.0);
  if (n_samples == 0 || k_count == 0)
    return;

  const std::size_t workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), n_samples));
  executor.parallel_do(workers, [&](std::size_t worker) {
    const std::size_t begin = n_samples * worker / workers;
    const std::size_t end = n_samples * (worker + 1) / workers;
    AlignedDoubleVector exp_terms(k_count, 0.0);
    for (std::size_t n = begin; n < end; ++n) {
      const double energy = shifted_samples[n];
      const double max_v = fill_mbar_linear_terms(
          energy, beta_k, log_nk_plus_fk, exp_terms.data());
      const double sum =
          sum_shifted_exponentials(exp_terms.data(), k_count, max_v, nullptr);
      log_denom[n] = max_v + std::log(sum + kLogGuard);
    }
  });
}

SolverEvaluation evaluate_mbar_state(
    const AlignedDoubleVector &shifted_samples,
    const std::vector<double> &beta_k, const std::vector<int> &sample_counts,
    const std::vector<double> &log_nk, const std::vector<double> &f_k,
    TitanExecutor &executor, bool want_hessian) {
  SolverEvaluation evaluation;
  const std::size_t k_count = beta_k.size();
  const std::size_t n_samples = shifted_samples.size();
  evaluation.residual.assign(k_count, 0.0);
  evaluation.square_sums.assign(k_count, 0.0);
  evaluation.log_denom.assign(n_samples, 0.0);
  if (want_hessian)
    evaluation.hessian.assign(k_count * k_count, 0.0);
  for (std::size_t k = 0; k < k_count; ++k)
    evaluation.residual[k] = static_cast<double>(sample_counts[k]);
  if (k_count == 0 || n_samples == 0)
    return evaluation;

  std::vector<double> log_nk_plus_fk(k_count, 0.0);
  for (std::size_t k = 0; k < k_count; ++k)
    log_nk_plus_fk[k] = log_nk[k] + f_k[k];
  compute_log_denom(shifted_samples, beta_k, log_nk_plus_fk, executor,
                    evaluation.log_denom);

  const std::size_t workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), n_samples));
  std::vector<std::vector<double>> residual_local(
      workers, std::vector<double>(k_count, 0.0));
  std::vector<std::vector<double>> square_local(
      workers, std::vector<double>(k_count, 0.0));
  std::vector<std::vector<double>> hessian_local;
  if (want_hessian) {
    hessian_local.assign(workers,
                         std::vector<double>(k_count * k_count, 0.0));
  }

  executor.parallel_do(workers, [&](std::size_t worker) {
    const std::size_t begin = n_samples * worker / workers;
    const std::size_t end = n_samples * (worker + 1) / workers;
    auto &res_local = residual_local[worker];
    auto &sq_local = square_local[worker];
    AlignedDoubleVector weights(k_count, 0.0);
    AlignedDoubleVector exp_terms(k_count, 0.0);
    if (!want_hessian) {
      for (std::size_t n = begin; n < end; ++n) {
        const double energy = shifted_samples[n];
        const double max_v = fill_mbar_linear_terms(
            energy, beta_k, log_nk_plus_fk, exp_terms.data());
        const double denom = std::max(
            sum_shifted_exponentials(exp_terms.data(), k_count, max_v,
                                     weights.data()),
            kLogGuard);
        normalize_weights_accumulate(weights.data(), k_count, 1.0 / denom,
                                     weights.data(), res_local.data(),
                                     sq_local.data());
      }
      return;
    }

    auto &h_local = hessian_local[worker];
    for (std::size_t n = begin; n < end; ++n) {
      const double energy = shifted_samples[n];
      const double max_v =
          fill_mbar_linear_terms(energy, beta_k, log_nk_plus_fk,
                                 exp_terms.data());
      const double denom = std::max(
          sum_shifted_exponentials(exp_terms.data(), k_count, max_v,
                                   weights.data()),
          kLogGuard);
      normalize_weights_accumulate(weights.data(), k_count, 1.0 / denom,
                                   weights.data(), res_local.data(),
                                   sq_local.data());

      for (std::size_t row = 0; row < k_count; ++row) {
        const double w_row = weights[row];
        const std::size_t diag_idx = row * k_count + row;
        h_local[diag_idx] += w_row * (1.0 - w_row);
        for (std::size_t col = row + 1; col < k_count; ++col) {
          const double offdiag = -w_row * weights[col];
          const std::size_t ij = row * k_count + col;
          const std::size_t ji = col * k_count + row;
          h_local[ij] += offdiag;
          h_local[ji] += offdiag;
        }
      }
    }
  });

  for (std::size_t worker = 0; worker < workers; ++worker) {
    for (std::size_t k = 0; k < k_count; ++k) {
      evaluation.residual[k] += residual_local[worker][k];
      evaluation.square_sums[k] += square_local[worker][k];
    }
    if (!want_hessian)
      continue;
    for (std::size_t idx = 0; idx < k_count * k_count; ++idx)
      evaluation.hessian[idx] += hessian_local[worker][idx];
  }

  const double total_samples =
      static_cast<double>(std::max<std::size_t>(1, n_samples));
  double max_abs_residual = 0.0;
  double residual_sq_sum = 0.0;
  for (double value : evaluation.residual) {
    max_abs_residual = std::max(max_abs_residual, std::abs(value));
    residual_sq_sum += value * value;
  }
  evaluation.residual_inf = max_abs_residual / total_samples;
  evaluation.residual_rms =
      std::sqrt(residual_sq_sum / static_cast<double>(std::max<std::size_t>(
                     1, evaluation.residual.size()))) /
      total_samples;
  evaluation.objective = 0.5 * residual_sq_sum / (total_samples * total_samples);
  return evaluation;
}

std::vector<double> compute_self_consistent_update(
    const AlignedDoubleVector &shifted_samples, const std::vector<double> &beta_k,
    const AlignedDoubleVector &log_denom, TitanExecutor &executor) {
  const std::size_t k_count = beta_k.size();
  const std::size_t n_samples = shifted_samples.size();
  std::vector<double> update(k_count, 0.0);
  if (k_count == 0 || n_samples == 0)
    return update;

  const std::size_t workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), k_count));
  std::vector<std::vector<double>> partial(
      workers, std::vector<double>(k_count, 0.0));
  executor.parallel_do(workers, [&](std::size_t worker) {
    const std::size_t begin = k_count * worker / workers;
    const std::size_t end = k_count * (worker + 1) / workers;
    auto &local = partial[worker];
    for (std::size_t k = begin; k < end; ++k) {
      double max_v = -std::numeric_limits<double>::infinity();
      for (std::size_t n = 0; n < n_samples; ++n) {
        const double value = -beta_k[k] * shifted_samples[n] - log_denom[n];
        if (value > max_v)
          max_v = value;
      }
      double sum = 0.0;
      for (std::size_t n = 0; n < n_samples; ++n)
        sum += std::exp(-beta_k[k] * shifted_samples[n] - log_denom[n] - max_v);
      local[k] = -(max_v + std::log(sum + kLogGuard));
    }
  });

  for (std::size_t worker = 0; worker < workers; ++worker) {
    const std::size_t begin = k_count * worker / workers;
    const std::size_t end = k_count * (worker + 1) / workers;
    for (std::size_t k = begin; k < end; ++k)
      update[k] = partial[worker][k];
  }
  normalize_free_energies(update);
  return update;
}

bool solve_dense_linear_system(std::vector<double> matrix,
                               const std::vector<double> &rhs,
                               std::vector<double> &solution) {
  const std::size_t n = rhs.size();
  solution = rhs;
  if (n == 0)
    return true;

  for (std::size_t pivot = 0; pivot < n; ++pivot) {
    std::size_t best_row = pivot;
    double best_abs = std::abs(matrix[pivot * n + pivot]);
    for (std::size_t row = pivot + 1; row < n; ++row) {
      const double value = std::abs(matrix[row * n + pivot]);
      if (value > best_abs) {
        best_abs = value;
        best_row = row;
      }
    }
    if (best_abs < 1e-14)
      return false;

    if (best_row != pivot) {
      for (std::size_t col = pivot; col < n; ++col)
        std::swap(matrix[pivot * n + col], matrix[best_row * n + col]);
      std::swap(solution[pivot], solution[best_row]);
    }

    const double diag = matrix[pivot * n + pivot];
    for (std::size_t row = pivot + 1; row < n; ++row) {
      const double factor = matrix[row * n + pivot] / diag;
      if (!std::isfinite(factor))
        return false;
      matrix[row * n + pivot] = 0.0;
      for (std::size_t col = pivot + 1; col < n; ++col)
        matrix[row * n + col] -= factor * matrix[pivot * n + col];
      solution[row] -= factor * solution[pivot];
    }
  }

  for (std::size_t back = n; back-- > 0;) {
    double value = solution[back];
    for (std::size_t col = back + 1; col < n; ++col)
      value -= matrix[back * n + col] * solution[col];
    const double diag = matrix[back * n + back];
    if (std::abs(diag) < 1e-14)
      return false;
    solution[back] = value / diag;
    if (!std::isfinite(solution[back]))
      return false;
  }
  return true;
}

double compute_effective_sample_size(const AlignedDoubleVector &weights) {
  if (weights.empty())
    return 0.0;
  double sum_sq = 0.0;
  for (double weight : weights)
    sum_sq += weight * weight;
  if (!(sum_sq > 0.0))
    return 0.0;
  return 1.0 / sum_sq;
}

double compute_min_state_effective_sample_size(
    const SolverEvaluation &evaluation, const std::vector<int> &sample_counts) {
  double min_ess = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < evaluation.residual.size() &&
                          k < evaluation.square_sums.size() &&
                          k < sample_counts.size();
       ++k) {
    const double sum_weights =
        static_cast<double>(sample_counts[k]) - evaluation.residual[k];
    const double sum_sq = evaluation.square_sums[k];
    if (!(sum_weights > 0.0) || !(sum_sq > 0.0))
      continue;
    const double ess = (sum_weights * sum_weights) / sum_sq;
    if (std::isfinite(ess))
      min_ess = std::min(min_ess, ess);
  }
  return std::isfinite(min_ess) ? min_ess : 0.0;
}

double compute_min_adjacent_overlap(const SolverEvaluation &evaluation) {
  const std::size_t k_count = evaluation.residual.size();
  if (k_count < 2 || evaluation.hessian.size() != k_count * k_count ||
      evaluation.square_sums.size() != k_count)
    return 1.0;

  double min_overlap = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k + 1 < k_count; ++k) {
    const double numerator = std::max(
        0.0, -evaluation.hessian[k * k_count + (k + 1)]);
    const double denom =
        std::sqrt(std::max(0.0, evaluation.square_sums[k]) *
                  std::max(0.0, evaluation.square_sums[k + 1]));
    const double overlap = (denom > 0.0) ? (numerator / denom) : 0.0;
    min_overlap = std::min(min_overlap, overlap);
  }
  return std::isfinite(min_overlap) ? min_overlap : 0.0;
}

std::string build_convergence_note(bool converged, std::size_t total_samples,
                                   double target_ess, double min_state_ess,
                                   double min_adjacent_overlap,
                                   double residual_inf) {
  if (converged)
    return "converged";

  const double target_ess_warn =
      std::max(kMinStateEssWarn,
               kMinTargetEssFractionWarn * static_cast<double>(total_samples));
  if (min_adjacent_overlap < kAdjacentOverlapWarn)
    return "insufficient adjacent-state overlap";
  if ((min_state_ess > 0.0 && min_state_ess < kMinStateEssWarn) ||
      (target_ess > 0.0 && target_ess < target_ess_warn)) {
    return "insufficient effective samples";
  }
  if (residual_inf >= kResidualConvTol)
    return "residual stalled before reaching tolerance";
  return "not converged";
}

bool solve_two_state_bar_fallback(const AlignedDoubleVector &shifted_samples,
                                  const std::vector<double> &beta_k,
                                  const std::vector<int> &sample_counts,
                                  std::vector<double> &f_k,
                                  SolverStats &stats) {
  if (beta_k.size() != 2 || sample_counts.size() != 2 ||
      shifted_samples.empty() || sample_counts[0] <= 0 || sample_counts[1] <= 0)
    return false;

  const double log_n0 = std::log(static_cast<double>(sample_counts[0]));
  const double log_n1 = std::log(static_cast<double>(sample_counts[1]));
  const double total_samples = static_cast<double>(shifted_samples.size());
  auto residual = [&](double delta) {
    double value = static_cast<double>(sample_counts[1]);
    for (double energy : shifted_samples) {
      const double a0 = log_n0 - beta_k[0] * energy;
      const double a1 = log_n1 + delta - beta_k[1] * energy;
      const double max_v = std::max(a0, a1);
      const double denom =
          std::exp(a0 - max_v) + std::exp(a1 - max_v) + kLogGuard;
      value -= std::exp(a1 - max_v) / denom;
    }
    return value;
  };

  double center = (f_k.size() >= 2) ? (f_k[1] - f_k[0]) : 0.0;
  double width = 1.0;
  double lower = center - width;
  double upper = center + width;
  double f_lower = residual(lower);
  double f_upper = residual(upper);
  for (int iter = 0; iter < 60 && f_lower * f_upper > 0.0; ++iter) {
    width *= 2.0;
    lower = center - width;
    upper = center + width;
    f_lower = residual(lower);
    f_upper = residual(upper);
  }
  if (f_lower * f_upper > 0.0)
    return false;

  bool converged = false;
  double delta = center;
  for (int iter = 0; iter < kBarIterations; ++iter) {
    delta = 0.5 * (lower + upper);
    const double f_mid = residual(delta);
    ++stats.iterations;
    stats.residual_inf = std::abs(f_mid) / std::max(1.0, total_samples);
    stats.residual_rms = stats.residual_inf;
    if (stats.residual_inf < kResidualConvTol ||
        std::abs(upper - lower) <
            1e-12 * std::max({1.0, std::abs(lower), std::abs(upper)})) {
      converged = true;
      break;
    }
    if (f_mid > 0.0) {
      lower = delta;
    } else {
      upper = delta;
    }
  }

  f_k.assign(2, 0.0);
  f_k[1] = delta;
  normalize_free_energies(f_k);
  stats.method = "bar_2state_fallback";
  return converged;
}

bool solve_mbar(const AlignedDoubleVector &samples,
                const std::vector<double> &beta_k,
                const std::vector<int> &sample_counts,
                std::vector<double> &f_k, double energy_min,
                TitanExecutor &executor, bool fast_mode,
                SolverStats *solver_stats) {
  const std::size_t k_count = beta_k.size();
  const std::size_t n_samples = samples.size();
  if (k_count == 0 || n_samples == 0)
    return false;

  const int sci_iters = fast_mode ? kSciIterationsFast : kSciIterations;
  const int newton_iters =
      fast_mode ? kNewtonIterationsFast : kNewtonIterations;
  SolverStats local_stats;
  SolverStats &stats = solver_stats ? *solver_stats : local_stats;
  stats.method = "not_run";
  stats.iterations = 0;
  stats.line_search_backtracks = 0;
  stats.residual_inf = 0.0;
  stats.residual_rms = 0.0;

  std::vector<double> log_nk(k_count, -1e12);
  for (std::size_t k = 0; k < k_count; ++k) {
    if (sample_counts[k] > 0)
      log_nk[k] = std::log(static_cast<double>(sample_counts[k]));
  }

  AlignedDoubleVector shifted_samples(samples.size(), 0.0);
  for (std::size_t n = 0; n < n_samples; ++n)
    shifted_samples[n] = samples[n] - energy_min;

  normalize_free_energies(f_k);
  SolverEvaluation current =
      evaluate_mbar_state(shifted_samples, beta_k, sample_counts, log_nk, f_k,
                          executor, false);
  stats.residual_inf = current.residual_inf;
  stats.residual_rms = current.residual_rms;
  if (current.residual_inf < kResidualConvTol) {
    stats.method = "fixed_point";
    return true;
  }

  for (int iter = 0; iter < sci_iters; ++iter) {
    ++stats.iterations;
    const auto fp_target = compute_self_consistent_update(
        shifted_samples, beta_k, current.log_denom, executor);
    std::vector<double> delta(k_count, 0.0);
    double max_delta = 0.0;
    for (std::size_t k = 0; k < k_count; ++k) {
      delta[k] = fp_target[k] - f_k[k];
      max_delta = std::max(max_delta, std::abs(delta[k]));
    }
    if (max_delta < kFixedPointStepTol)
      break;

    bool accepted = false;
    for (int backtrack = 0; backtrack <= kLineSearchMaxBacktracks;
         ++backtrack) {
      const double alpha = std::ldexp(1.0, -backtrack);
      std::vector<double> candidate = f_k;
      for (std::size_t k = 0; k < k_count; ++k)
        candidate[k] += alpha * delta[k];
      normalize_free_energies(candidate);
      auto candidate_eval =
          evaluate_mbar_state(shifted_samples, beta_k, sample_counts, log_nk,
                              candidate, executor, false);
      if (candidate_eval.objective + 1e-14 < current.objective ||
          candidate_eval.residual_inf < current.residual_inf) {
        if (backtrack > 0)
          stats.line_search_backtracks +=
              static_cast<std::size_t>(backtrack);
        f_k = std::move(candidate);
        current = std::move(candidate_eval);
        accepted = true;
        break;
      }
    }
    if (!accepted)
      break;
    stats.residual_inf = current.residual_inf;
    stats.residual_rms = current.residual_rms;
    if (current.residual_inf < kResidualConvTol) {
      stats.method = "fixed_point";
      return true;
    }
  }

  current = evaluate_mbar_state(shifted_samples, beta_k, sample_counts, log_nk,
                                f_k, executor, true);
  stats.residual_inf = current.residual_inf;
  stats.residual_rms = current.residual_rms;
  if (current.residual_inf < kResidualConvTol) {
    stats.method = "fixed_point";
    return true;
  }

  for (int iter = 0; iter < newton_iters; ++iter) {
    ++stats.iterations;
    if (current.residual_inf < kResidualConvTol) {
      stats.method = "newton_dense";
      return true;
    }
    if (k_count <= 1)
      break;

    const std::size_t dim = k_count - 1;
    std::vector<double> residual_sub(dim, 0.0);
    std::vector<double> hessian_sub(dim * dim, 0.0);
    for (std::size_t row = 1; row < k_count; ++row) {
      residual_sub[row - 1] = current.residual[row];
      for (std::size_t col = 1; col < k_count; ++col) {
        hessian_sub[(row - 1) * dim + (col - 1)] =
            current.hessian[row * k_count + col];
      }
    }

    double trace = 0.0;
    for (std::size_t idx = 0; idx < dim; ++idx)
      trace += hessian_sub[idx * dim + idx];
    double ridge =
        std::max(1e-10, 1e-10 * trace / static_cast<double>(std::max<std::size_t>(1, dim)));
    std::vector<double> step_sub;
    bool solved_step = false;
    for (int attempt = 0; attempt < 6; ++attempt) {
      std::vector<double> regularized = hessian_sub;
      for (std::size_t idx = 0; idx < dim; ++idx)
        regularized[idx * dim + idx] += ridge;
      if (solve_dense_linear_system(regularized, residual_sub, step_sub)) {
        solved_step = true;
        break;
      }
      ridge *= 10.0;
    }
    if (!solved_step)
      break;

    double max_step = 0.0;
    for (double &value : step_sub) {
      if (!std::isfinite(value)) {
        solved_step = false;
        break;
      }
      max_step = std::max(max_step, std::abs(value));
    }
    if (!solved_step)
      break;
    if (max_step > kNewtonStepClamp) {
      const double scale = kNewtonStepClamp / max_step;
      for (double &value : step_sub)
        value *= scale;
    }

    bool accepted = false;
    for (int backtrack = 0; backtrack <= kLineSearchMaxBacktracks;
         ++backtrack) {
      const double alpha = std::ldexp(1.0, -backtrack);
      std::vector<double> candidate = f_k;
      for (std::size_t k = 1; k < k_count; ++k)
        candidate[k] += alpha * step_sub[k - 1];
      normalize_free_energies(candidate);
      auto candidate_eval =
          evaluate_mbar_state(shifted_samples, beta_k, sample_counts, log_nk,
                              candidate, executor, true);
      if (candidate_eval.objective + 1e-14 < current.objective ||
          candidate_eval.residual_inf < current.residual_inf) {
        if (backtrack > 0)
          stats.line_search_backtracks +=
              static_cast<std::size_t>(backtrack);
        f_k = std::move(candidate);
        current = std::move(candidate_eval);
        accepted = true;
        break;
      }
    }
    if (!accepted)
      break;

    stats.residual_inf = current.residual_inf;
    stats.residual_rms = current.residual_rms;
    if (current.residual_inf < kResidualConvTol) {
      stats.method = "newton_dense";
      return true;
    }
  }

  if (k_count == 2) {
    SolverStats bar_stats;
    if (solve_two_state_bar_fallback(shifted_samples, beta_k, sample_counts, f_k,
                                     bar_stats)) {
      stats.iterations += bar_stats.iterations;
      stats.residual_inf = bar_stats.residual_inf;
      stats.residual_rms = bar_stats.residual_rms;
      stats.method = bar_stats.method;
      return true;
    }
    stats.iterations += bar_stats.iterations;
  }

  current = evaluate_mbar_state(shifted_samples, beta_k, sample_counts, log_nk,
                                f_k, executor, true);
  stats.residual_inf = current.residual_inf;
  stats.residual_rms = current.residual_rms;
  stats.method = "newton_dense_stalled";
  return false;
}

ObservableResult compute_observable_at_temperature(
    const AlignedDoubleVector &samples, const std::vector<double> &beta_sampled,
    const std::vector<int> &sample_counts, const std::vector<double> &f_k,
    double target_temperature_k, double energy_min, TitanExecutor &executor,
    bool need_weights) {
  ObservableResult result;
  const std::size_t k_count = beta_sampled.size();
  const std::size_t n_samples = samples.size();
  if (k_count == 0 || n_samples == 0 || target_temperature_k <= 0.0)
    return result;

  std::vector<double> log_nk(k_count, -1e12);
  for (std::size_t k = 0; k < k_count; ++k) {
    if (sample_counts[k] > 0)
      log_nk[k] = std::log(static_cast<double>(sample_counts[k]));
  }
  std::vector<double> log_nk_plus_fk(k_count, 0.0);
  for (std::size_t k = 0; k < k_count; ++k)
    log_nk_plus_fk[k] = log_nk[k] + f_k[k];

  AlignedDoubleVector shifted_samples(n_samples, 0.0);
  for (std::size_t n = 0; n < n_samples; ++n)
    shifted_samples[n] = samples[n] - energy_min;

  AlignedDoubleVector log_denom(n_samples, 0.0);
  compute_log_denom(shifted_samples, beta_sampled, log_nk_plus_fk, executor,
                    log_denom);

  const double beta_target = 1.0 / (kBoltzmannKjPerMolK * target_temperature_k);
  double max_v = -std::numeric_limits<double>::infinity();
  for (std::size_t n = 0; n < n_samples; ++n) {
    const double value = -beta_target * shifted_samples[n] - log_denom[n];
    if (value > max_v)
      max_v = value;
  }

  double weight_sum = 0.0;
  double mean = 0.0;
  double m2 = 0.0;
  if (need_weights)
    result.normalized_weights.assign(n_samples, 0.0);

  for (std::size_t n = 0; n < n_samples; ++n) {
    const double energy = shifted_samples[n];
    const double weight =
        std::exp(-beta_target * energy - log_denom[n] - max_v);
    if (need_weights)
      result.normalized_weights[n] = weight;
    const double new_weight_sum = weight_sum + weight;
    const double delta = energy - mean;
    mean += (weight / new_weight_sum) * delta;
    const double delta2 = energy - mean;
    m2 += weight * delta * delta2;
    weight_sum = new_weight_sum;
  }

  if (weight_sum <= 0.0)
    return result;
  if (need_weights) {
    for (double &weight : result.normalized_weights)
      weight /= weight_sum;
  }

  const double variance = std::max(0.0, m2 / weight_sum);
  result.mean_energy = mean + energy_min;
  result.heat_capacity =
      variance * kBoltzmannKjPerMolK * beta_target * beta_target;
  return result;
}

} // namespace

namespace TitanMBAR {

CoreResult run_temperature_ladder_reweighting(const arguments &args,
                                              const FrameMetadata &frame_metadata,
                                              TitanExecutor &executor,
                                              std::fstream &log) {
  CoreResult result;
  result.target_temp_k = args.mbar.target_temp;
  if (!args.mbar.flag_mbar)
    return result;

  if (args.mbar.ener_list_file.empty()) {
    result.abort_reason = "-mbar-ener not set";
    TitanUI::print_error("MBAR: -mbar-ener not set. Skipping MBAR.");
    return result;
  }

  auto ener_paths = TitanDHAM::read_file_list(args.mbar.ener_list_file);
  if (ener_paths.empty()) {
    result.abort_reason = "empty energy list";
    TitanUI::print_error("MBAR: no usable energy file paths were read.");
    return result;
  }

  const bool use_npt = (args.mbar.ensemble == "npt");
  const bool use_per_frame = (args.mbar.temp_mode == "per_frame");
  const bool use_demux_map =
      use_per_frame && !args.mbar.demux_map_file.empty();
  const bool have_temp_file = !args.mbar.temps_file.empty();
  const bool infer_ladder_from_replica_input = !have_temp_file;
  result.used_per_frame_temperature = use_per_frame;
  result.used_demux_map = use_demux_map;
  result.temperature_ladder_source = have_temp_file ? "file" : "replica-inferred";

  if (infer_ladder_from_replica_input && !use_per_frame) {
    result.abort_reason =
        "auto ladder inference requires -mbar-temp-mode per_frame";
    TitanUI::print_error(
        "MBAR: when -mbar-temps is omitted, use -mbar-temp-mode per_frame so "
        "Titan can infer the temperature ladder from replica data.");
    return result;
  }

  const bool have_frame_metadata =
      !frame_metadata.replica_ids.empty() &&
      frame_metadata.replica_ids.size() == frame_metadata.timestamps_ps.size() &&
      frame_metadata.replica_ids.size() == frame_metadata.global_indices.size();
  const bool have_box_volume_metadata =
      have_frame_metadata &&
      frame_metadata.box_volumes_nm3.size() == frame_metadata.replica_ids.size();
  result.box_volume_metadata_available = have_box_volume_metadata;
  if (!frame_metadata.empty() && !have_frame_metadata) {
    TitanUI::print_warning("MBAR: frame metadata size mismatch. Proceeding with "
                           "energy-only core outputs.");
  }
  if (have_frame_metadata && !frame_metadata.box_volumes_nm3.empty() &&
      !have_box_volume_metadata) {
    TitanUI::print_warning(
        "MBAR: frame box-volume metadata size mismatch. NPT runs will treat "
        "the XVG observable as already enthalpy-like.");
  }
  if (use_per_frame) {
    if (use_demux_map) {
      TitanUI::print_info(
          "MBAR: using demux-map assignments for per-frame thermodynamic "
          "states.");
    } else {
      TitanUI::print_info(
          "MBAR: mapping per-frame temperatures to the nearest ladder state.");
    }
  }
  if (use_npt) {
    TitanUI::print_info(
        "MBAR: ensemble=npt. The XVG observable column is treated as an "
        "enthalpy-like quantity.");
  }

  std::unordered_map<int, std::vector<double>> per_replica_times;
  std::unordered_map<int, std::vector<int64_t>> per_replica_global_indices;
  std::unordered_map<int, std::vector<double>> per_replica_box_volumes;
  if (have_frame_metadata) {
    const std::size_t n_frames = frame_metadata.replica_ids.size();
    for (std::size_t idx = 0; idx < n_frames; ++idx) {
      const int replica_id = frame_metadata.replica_ids[idx];
      per_replica_times[replica_id].push_back(frame_metadata.timestamps_ps[idx]);
      per_replica_global_indices[replica_id].push_back(
          frame_metadata.global_indices[idx]);
      if (have_box_volume_metadata) {
        per_replica_box_volumes[replica_id].push_back(
            frame_metadata.box_volumes_nm3[idx]);
      }
    }
  }

  std::unordered_map<int, std::vector<int>> per_replica_state_assignments;
  std::size_t n_states = 0;
  std::vector<double> temps;
  if (have_temp_file) {
    temps = TitanDHAM::read_temps(args.mbar.temps_file);
    if (temps.empty()) {
      result.abort_reason = "empty temps file";
      TitanUI::print_error(
          "MBAR: no usable ladder temperatures were read from -mbar-temps.");
      return result;
    }
    n_states = temps.size();
  }
  if (use_demux_map) {
    std::vector<std::vector<int>> demux_rows;
    std::string demux_error;
    if (!read_mbar_demux_map(args.mbar.demux_map_file, demux_rows, demux_error)) {
      result.abort_reason = demux_error;
      TitanUI::print_error("MBAR: " + demux_error);
      return result;
    }
    if (n_states == 0)
      n_states = demux_rows.front().size();
    if (demux_rows.front().size() != n_states) {
      result.abort_reason = "demux/temperature count mismatch";
      TitanUI::print_error(
          "MBAR: demux-map column count (" +
          std::to_string(demux_rows.front().size()) +
          ") must match the number of ladder temperatures (" +
          std::to_string(n_states) + ").");
      return result;
    }
    per_replica_state_assignments =
        build_per_replica_state_assignments(demux_rows, n_states);
  }
  if (n_states == 0)
    n_states = ener_paths.size();
  if (n_states == 0) {
    result.abort_reason = "no MBAR states";
    TitanUI::print_error("MBAR: no thermodynamic states could be determined.");
    return result;
  }
  if (have_temp_file && temps.size() != ener_paths.size()) {
    result.abort_reason = "temps/energy count mismatch";
    TitanUI::print_error("MBAR: #temps (" + std::to_string(temps.size()) +
                         ") != #energy_files (" +
                         std::to_string(ener_paths.size()) + ").");
    return result;
  }

  const bool need_temperature_series =
      use_per_frame && (!use_demux_map || infer_ladder_from_replica_input);
  std::vector<std::vector<std::pair<double, double>>> parsed_xvg(
      ener_paths.size());
  std::vector<std::vector<TitanIO::XvgTimeTwoValues>> parsed_xvg_with_temp(
      ener_paths.size());
  const std::size_t workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), ener_paths.size()));
  if (need_temperature_series) {
    std::vector<std::string> parse_errors(ener_paths.size());
    executor.parallel_do(workers, [&](std::size_t worker) {
      const std::size_t begin = ener_paths.size() * worker / workers;
      const std::size_t end = ener_paths.size() * (worker + 1) / workers;
      for (std::size_t k = begin; k < end; ++k) {
        auto parsed = TitanIO::parse_xvg_time_two_values(
            ener_paths[k], 0, 1, static_cast<std::size_t>(args.mbar.temp_column));
        if (parsed) {
          parsed_xvg_with_temp[k] = std::move(parsed.value());
        } else {
          parse_errors[k] = parsed.error();
        }
      }
    });
    for (const auto &error : parse_errors) {
      if (!error.empty())
        TitanUI::print_error("MBAR: " + error);
    }
  } else {
    executor.parallel_do(workers, [&](std::size_t worker) {
      const std::size_t begin = ener_paths.size() * worker / workers;
      const std::size_t end = ener_paths.size() * (worker + 1) / workers;
      for (std::size_t k = begin; k < end; ++k)
        parsed_xvg[k] = TitanDHAM::parse_xvg(ener_paths[k]);
    });
  }

  if (infer_ladder_from_replica_input) {
    if (!need_temperature_series) {
      result.abort_reason = "missing per-frame temperatures for ladder inference";
      TitanUI::print_error(
          "MBAR: automatic ladder inference requires per-frame temperature "
          "data in the replica XVG files.");
      return result;
    }

    std::string infer_error;
    if (use_demux_map) {
      temps = infer_ladder_temperatures_from_demux(parsed_xvg_with_temp,
                                                   per_replica_state_assignments,
                                                   n_states, args, infer_error);
      result.temperature_ladder_source = "demux-inferred";
    } else {
      temps = infer_ladder_temperatures_from_observed_series(parsed_xvg_with_temp,
                                                             n_states, args,
                                                             infer_error);
      result.temperature_ladder_source = "replica-inferred";
    }
    if (!infer_error.empty() || temps.size() != n_states) {
      result.abort_reason =
          infer_error.empty() ? "failed to infer ladder temperatures"
                              : infer_error;
      TitanUI::print_error("MBAR: " + result.abort_reason);
      return result;
    }
    TitanUI::print_info("MBAR: inferred " + std::to_string(temps.size()) +
                        " ladder temperatures from replica input.");
  }

  auto apply_match_to_audit = [&](WindowAuditRow &audit,
                                  const TitanDHAM::WindowMatchResult &matched,
                                  bool sparse_energy, bool irregular_energy_dt) {
    audit.overlap_found = matched.overlap_found;
    audit.overlap_start_ps = matched.overlap_start_ps;
    audit.overlap_end_ps = matched.overlap_end_ps;
    audit.kept_frames = matched.kept_frames;
    audit.dropped_outside_overlap = matched.dropped_outside_overlap;
    audit.unmatched_in_overlap = matched.unmatched_in_overlap;
    audit.inferred_dt_energy_ps = matched.inferred_dt_energy_ps;
    audit.sparse_energy = sparse_energy;
    audit.irregular_energy_dt = irregular_energy_dt;
    const double coverage_tol =
        std::max(1e-6, 2.0 * std::max(1e-6, matched.inferred_dt_energy_ps));
    audit.truncated_start =
        std::isfinite(matched.traj_min_ps) && std::isfinite(matched.energy_min_ps) &&
        matched.energy_min_ps > matched.traj_min_ps + coverage_tol;
    audit.truncated_end =
        std::isfinite(matched.traj_max_ps) && std::isfinite(matched.energy_max_ps) &&
        matched.energy_max_ps < matched.traj_max_ps - coverage_tol;
  };

  auto overlap_status = [&](const WindowAuditRow &audit, bool has_window_loss) {
    if (audit.truncated_start && audit.truncated_end)
      return std::string("truncated_both");
    if (audit.truncated_start)
      return std::string("truncated_start");
    if (audit.truncated_end)
      return std::string("truncated_end");
    if (audit.irregular_energy_dt)
      return std::string("irregular_dt");
    if (audit.sparse_energy)
      return std::string("sparse_energy");
    if (has_window_loss)
      return std::string("trimmed_to_overlap");
    return std::string("ok");
  };

  auto push_sample = [&](PreparedSeries &series, double observable_value,
                         std::size_t local_frame_index, double time_ps,
                         int sampled_state_index, double sampled_temperature_k,
                         int64_t global_frame_index) {
    if (!std::isfinite(observable_value) ||
        !std::isfinite(sampled_temperature_k) || sampled_state_index < 0 ||
        static_cast<std::size_t>(sampled_state_index) >= temps.size()) {
      return false;
    }
    series.observables.push_back(observable_value);
    series.local_indices.push_back(local_frame_index);
    series.times_ps.push_back(time_ps);
    series.sampled_state_indices.push_back(sampled_state_index);
    series.sampled_temperatures_k.push_back(sampled_temperature_k);
    series.global_indices.push_back(global_frame_index);
    return true;
  };

  std::vector<PreparedSeries> usable_series;
  usable_series.reserve(ener_paths.size());
  bool strict_window_violation = false;

  for (std::size_t k = 0; k < ener_paths.size(); ++k) {
    WindowAuditRow audit;
    audit.replica_id = static_cast<int>(k);
    std::vector<std::pair<double, double>> filtered_energy_pairs;
    std::vector<std::pair<double, double>> filtered_observable_pairs;
    std::vector<std::pair<double, double>> filtered_temperature_pairs;
    TimeSelectionSummary selected_energy{};
    if (need_temperature_series) {
      filtered_observable_pairs =
          make_time_value_pairs(parsed_xvg_with_temp[k], false, args);
      if (!use_demux_map) {
        filtered_temperature_pairs =
            make_time_value_pairs(parsed_xvg_with_temp[k], true, args);
      }
      selected_energy = summarize_selected_times(
          parsed_xvg_with_temp[k],
          [](const TitanIO::XvgTimeTwoValues &row) { return row.time_ps; }, args);
    } else {
      filtered_energy_pairs = filter_time_value_pairs(parsed_xvg[k], args);
      selected_energy = summarize_selected_times(
          parsed_xvg[k],
          [](const std::pair<double, double> &row) { return row.first; }, args);
    }
    audit.energy_present = selected_energy.count > 0;
    audit.energy_points = selected_energy.count;
    if (selected_energy.count > 0) {
      audit.energy_min_ps = selected_energy.min_ps;
      audit.energy_max_ps = selected_energy.max_ps;
    }

    auto it_times = per_replica_times.find(static_cast<int>(k));
    auto it_globals = per_replica_global_indices.find(static_cast<int>(k));
    const bool have_replica_metadata =
        have_frame_metadata && it_times != per_replica_times.end() &&
        it_globals != per_replica_global_indices.end() &&
        it_times->second.size() == it_globals->second.size();
    audit.traj_present = have_replica_metadata;
    if (have_replica_metadata) {
      audit.traj_frames = it_times->second.size();
      if (!it_times->second.empty()) {
        audit.traj_min_ps = it_times->second.front();
        audit.traj_max_ps = it_times->second.back();
      }
    }

    if (!need_temperature_series && filtered_energy_pairs.empty()) {
      audit.status = "missing_or_empty_energy";
      result.window_audit.push_back(std::move(audit));
      continue;
    }
    if (need_temperature_series && filtered_observable_pairs.empty()) {
      audit.status = "missing_or_empty_energy";
      result.window_audit.push_back(std::move(audit));
      continue;
    }

    PreparedSeries series;
    series.replica_id = static_cast<int>(k);
    const auto demux_it = per_replica_state_assignments.find(static_cast<int>(k));
    const std::vector<int> *demux_assignment =
        (use_demux_map && demux_it != per_replica_state_assignments.end())
            ? &demux_it->second
            : nullptr;

    if (!have_replica_metadata) {
      if (use_demux_map && demux_assignment == nullptr) {
        audit.status = "missing_demux_assignment";
        result.window_audit.push_back(std::move(audit));
        continue;
      }

      if (need_temperature_series) {
        series.observables.reserve(filtered_observable_pairs.size());
        series.local_indices.reserve(filtered_observable_pairs.size());
        series.times_ps.reserve(filtered_observable_pairs.size());
        series.sampled_state_indices.reserve(filtered_observable_pairs.size());
        series.sampled_temperatures_k.reserve(filtered_observable_pairs.size());
        series.global_indices.reserve(filtered_observable_pairs.size());
        for (std::size_t idx = 0; idx < parsed_xvg_with_temp[k].size(); ++idx) {
          const auto &row = parsed_xvg_with_temp[k][idx];
          if (!keep_mbar_energy_time(args, row.time_ps))
            continue;
          int sampled_state_index = -1;
          double sampled_temperature_k =
              std::numeric_limits<double>::quiet_NaN();
          if (use_demux_map) {
            if (demux_assignment == nullptr || idx >= demux_assignment->size())
              continue;
            sampled_state_index = (*demux_assignment)[idx];
          } else {
            sampled_state_index = map_temperature_to_state(temps, row.value2);
          }
          if (sampled_state_index >= 0 &&
              static_cast<std::size_t>(sampled_state_index) < temps.size()) {
            sampled_temperature_k =
                temps[static_cast<std::size_t>(sampled_state_index)];
          }
          push_sample(series, row.value1, idx, row.time_ps, sampled_state_index,
                      sampled_temperature_k, -1);
        }
      } else {
        series.observables.reserve(filtered_energy_pairs.size());
        series.local_indices.reserve(filtered_energy_pairs.size());
        series.times_ps.reserve(filtered_energy_pairs.size());
        series.sampled_state_indices.reserve(filtered_energy_pairs.size());
        series.sampled_temperatures_k.reserve(filtered_energy_pairs.size());
        series.global_indices.reserve(filtered_energy_pairs.size());
        for (std::size_t idx = 0; idx < parsed_xvg[k].size(); ++idx) {
          if (!keep_mbar_energy_time(args, parsed_xvg[k][idx].first))
            continue;
          int sampled_state_index = static_cast<int>(k);
          double sampled_temperature_k =
              (k < temps.size()) ? temps[k]
                                 : std::numeric_limits<double>::quiet_NaN();
          if (demux_assignment != nullptr) {
            if (idx >= demux_assignment->size())
              continue;
            sampled_state_index = (*demux_assignment)[idx];
            if (sampled_state_index < 0 ||
                static_cast<std::size_t>(sampled_state_index) >= temps.size()) {
              continue;
            }
            sampled_temperature_k =
                temps[static_cast<std::size_t>(sampled_state_index)];
          }
          push_sample(series, parsed_xvg[k][idx].second, idx,
                      parsed_xvg[k][idx].first, sampled_state_index,
                      sampled_temperature_k, -1);
        }
      }
      audit.overlap_found = true;
      audit.kept_frames = series.observables.size();
      audit.inferred_dt_energy_ps =
          (args.mbar.dt_energy_ps > 0.0) ? args.mbar.dt_energy_ps : 0.0;
      if (need_temperature_series) {
        audit.sparse_energy = (filtered_observable_pairs.size() < 2);
        audit.irregular_energy_dt =
            detect_irregular_energy_dt(filtered_observable_pairs,
                                       args.mbar.dt_energy_ps);
        audit.status =
            series.observables.empty()
                ? "no_valid_temps"
                : (use_demux_map ? "demux_energy_only" : "per_frame_energy_only");
      } else {
        audit.sparse_energy = (filtered_energy_pairs.size() < 2);
        audit.irregular_energy_dt =
            detect_irregular_energy_dt(filtered_energy_pairs,
                                       args.mbar.dt_energy_ps);
        audit.status =
            series.observables.empty()
                ? "no_state_assignment"
                : (use_demux_map ? "demux_energy_only" : "energy_only");
      }
      result.window_audit.push_back(audit);
      if (!series.observables.empty())
        usable_series.push_back(std::move(series));
      continue;
    }

    if (args.mbar.window_match == "off") {
      if (use_demux_map && demux_assignment == nullptr) {
        audit.status = "missing_demux_assignment";
        result.window_audit.push_back(std::move(audit));
        continue;
      }

      std::vector<double> aligned_observables;
      std::vector<double> aligned_temperatures;
      if (need_temperature_series) {
        const auto matched = TitanDHAM::window_match_energy_to_dtraj(
            it_times->second, filtered_observable_pairs, args.mbar.dt_energy_ps);
        apply_match_to_audit(
            audit, matched, filtered_observable_pairs.size() < 2,
            detect_irregular_energy_dt(filtered_observable_pairs,
                                       matched.inferred_dt_energy_ps));
        aligned_observables =
            materialize_aligned_values(it_times->second.size(), matched);
        if (!use_demux_map) {
          const auto matched_temps = TitanDHAM::window_match_energy_to_dtraj(
              it_times->second, filtered_temperature_pairs,
              args.mbar.dt_energy_ps);
          aligned_temperatures =
              materialize_aligned_values(it_times->second.size(), matched_temps);
        }
      } else {
        const auto matched = TitanDHAM::window_match_energy_to_dtraj(
            it_times->second, filtered_energy_pairs, args.mbar.dt_energy_ps);
        apply_match_to_audit(
            audit, matched, filtered_energy_pairs.size() < 2,
            detect_irregular_energy_dt(filtered_energy_pairs,
                                       matched.inferred_dt_energy_ps));
        aligned_observables =
            materialize_aligned_values(it_times->second.size(), matched);
      }

      for (std::size_t idx = 0; idx < aligned_observables.size() &&
                              idx < it_times->second.size();
           ++idx) {
        if (!std::isfinite(aligned_observables[idx]))
          continue;
        int sampled_state_index = static_cast<int>(k);
        double sampled_temperature_k =
            (k < temps.size()) ? temps[k]
                               : std::numeric_limits<double>::quiet_NaN();
        if (need_temperature_series && !use_demux_map) {
          if (idx >= aligned_temperatures.size() ||
              !std::isfinite(aligned_temperatures[idx])) {
            continue;
          }
          sampled_state_index =
              map_temperature_to_state(temps, aligned_temperatures[idx]);
          if (sampled_state_index >= 0 &&
              static_cast<std::size_t>(sampled_state_index) < temps.size()) {
            sampled_temperature_k =
                temps[static_cast<std::size_t>(sampled_state_index)];
          }
        } else if (demux_assignment != nullptr) {
          if (idx >= demux_assignment->size())
            continue;
          sampled_state_index = (*demux_assignment)[idx];
          if (sampled_state_index < 0 ||
              static_cast<std::size_t>(sampled_state_index) >= temps.size()) {
            continue;
          }
          sampled_temperature_k =
              temps[static_cast<std::size_t>(sampled_state_index)];
        }
        const int64_t global_idx =
            (idx < it_globals->second.size()) ? it_globals->second[idx] : -1;
        push_sample(series, aligned_observables[idx], idx, it_times->second[idx],
                    sampled_state_index, sampled_temperature_k, global_idx);
      }
      audit.kept_frames = series.observables.size();
      audit.status = series.observables.empty()
                         ? (need_temperature_series && !use_demux_map
                                ? "no_valid_temps"
                                : "no_state_assignment")
                         : (need_temperature_series && !use_demux_map
                                ? "per_frame_off"
                                : (use_demux_map ? "demux_off" : "legacy_off"));
      result.window_audit.push_back(audit);
      if (!series.observables.empty())
        usable_series.push_back(std::move(series));
      continue;
    }

    if (use_demux_map && demux_assignment == nullptr) {
      audit.status = "missing_demux_assignment";
      result.window_audit.push_back(std::move(audit));
      continue;
    }

    TitanDHAM::WindowMatchResult matched;
    std::vector<double> aligned_temperatures;
    if (need_temperature_series) {
      matched = TitanDHAM::window_match_energy_to_dtraj(
          it_times->second, filtered_observable_pairs, args.mbar.dt_energy_ps);
      apply_match_to_audit(
          audit, matched, filtered_observable_pairs.size() < 2,
          detect_irregular_energy_dt(filtered_observable_pairs,
                                     matched.inferred_dt_energy_ps));
      if (!use_demux_map) {
        const auto matched_temps = TitanDHAM::window_match_energy_to_dtraj(
            it_times->second, filtered_temperature_pairs,
            args.mbar.dt_energy_ps);
        aligned_temperatures =
            materialize_aligned_values(it_times->second.size(), matched_temps);
      }
    } else {
      matched = TitanDHAM::window_match_energy_to_dtraj(
          it_times->second, filtered_energy_pairs, args.mbar.dt_energy_ps);
      apply_match_to_audit(
          audit, matched, filtered_energy_pairs.size() < 2,
          detect_irregular_energy_dt(filtered_energy_pairs,
                                     matched.inferred_dt_energy_ps));
    }

    const bool no_usable_overlap = !matched.overlap_found || matched.kept_frames == 0;
    const bool has_window_loss =
        matched.dropped_outside_overlap > 0 || matched.unmatched_in_overlap > 0;
    if (no_usable_overlap) {
      audit.status = "no_overlap";
      if (args.mbar.window_match == "strict")
        strict_window_violation = true;
      result.window_audit.push_back(std::move(audit));
      continue;
    }
    if (args.mbar.window_match == "strict" && has_window_loss) {
      audit.status = "strict_rejected";
      strict_window_violation = true;
      result.window_audit.push_back(std::move(audit));
      continue;
    }

    for (std::size_t idx = 0; idx < matched.kept_indices.size() &&
                            idx < matched.aligned_values.size() &&
                            idx < matched.kept_dtraj_times_ps.size();
         ++idx) {
      if (!std::isfinite(matched.aligned_values[idx]))
        continue;
      const std::size_t local_frame_index = matched.kept_indices[idx];
      int sampled_state_index = static_cast<int>(k);
      double sampled_temperature_k =
          (k < temps.size()) ? temps[k]
                             : std::numeric_limits<double>::quiet_NaN();
      if (need_temperature_series && !use_demux_map) {
        if (local_frame_index >= aligned_temperatures.size() ||
            !std::isfinite(aligned_temperatures[local_frame_index])) {
          continue;
        }
        sampled_state_index =
            map_temperature_to_state(temps, aligned_temperatures[local_frame_index]);
        if (sampled_state_index >= 0 &&
            static_cast<std::size_t>(sampled_state_index) < temps.size()) {
          sampled_temperature_k =
              temps[static_cast<std::size_t>(sampled_state_index)];
        }
      } else if (demux_assignment != nullptr) {
        if (local_frame_index >= demux_assignment->size())
          continue;
        sampled_state_index = (*demux_assignment)[local_frame_index];
        if (sampled_state_index < 0 ||
            static_cast<std::size_t>(sampled_state_index) >= temps.size()) {
          continue;
        }
        sampled_temperature_k =
            temps[static_cast<std::size_t>(sampled_state_index)];
      }
      const int64_t global_idx =
          (local_frame_index < it_globals->second.size())
              ? it_globals->second[local_frame_index]
              : -1;
      push_sample(series, matched.aligned_values[idx], local_frame_index,
                  matched.kept_dtraj_times_ps[idx], sampled_state_index,
                  sampled_temperature_k, global_idx);
    }
    audit.kept_frames = series.observables.size();
    audit.status =
        series.observables.empty()
            ? (need_temperature_series && !use_demux_map ? "no_valid_temps"
                                                         : "no_state_assignment")
            : overlap_status(audit, has_window_loss);
    result.window_audit.push_back(audit);
    if (!series.observables.empty())
      usable_series.push_back(std::move(series));
  }

  if (strict_window_violation) {
    result.abort_reason =
        "strict window matching rejected one or more replicas";
    TitanUI::print_warning(
        "MBAR: strict window matching failed. Skipping MBAR reweighting.");
    return result;
  }
  if (usable_series.empty()) {
    result.abort_reason = "no usable aligned energy samples";
    TitanUI::print_warning(
        "MBAR: no usable aligned energy samples remain after window matching.");
    return result;
  }

  result.usable_replica_series = usable_series.size();

  AlignedDoubleVector samples;
  std::vector<double> sampled_temperatures;
  std::vector<int> sample_counts;
  std::vector<double> observable_means;
  std::vector<SampleOrigin> sample_origins;
  std::size_t total_samples = 0;
  for (const auto &series : usable_series)
    total_samples += series.observables.size();
  samples.reserve(total_samples);
  sample_origins.reserve(total_samples);

  std::vector<int> all_state_counts(temps.size(), 0);
  std::vector<double> all_state_sums(temps.size(), 0.0);
  for (const auto &series : usable_series) {
    for (std::size_t idx = 0; idx < series.observables.size(); ++idx) {
      const int state_index = series.sampled_state_indices[idx];
      if (state_index < 0 ||
          static_cast<std::size_t>(state_index) >= temps.size()) {
        continue;
      }
      samples.push_back(series.observables[idx]);
      all_state_counts[static_cast<std::size_t>(state_index)] += 1;
      all_state_sums[static_cast<std::size_t>(state_index)] += series.observables[idx];

      SampleOrigin origin;
      origin.replica_id = series.replica_id;
      origin.local_frame_index = series.local_indices[idx];
      origin.time_ps = series.times_ps[idx];
      origin.sampled_state_index = state_index;
      origin.sampled_temperature_k = series.sampled_temperatures_k[idx];
      origin.global_frame_index = series.global_indices[idx];
      sample_origins.push_back(origin);
    }
  }

  std::vector<int> active_state_indices;
  active_state_indices.reserve(temps.size());
  for (std::size_t state_idx = 0; state_idx < temps.size(); ++state_idx) {
    if (all_state_counts[state_idx] > 0)
      active_state_indices.push_back(static_cast<int>(state_idx));
  }
  if (active_state_indices.empty() || samples.empty()) {
    result.abort_reason = "no MBAR states with usable samples";
    TitanUI::print_warning(
        "MBAR: no thermodynamic states with usable samples remain after "
        "assignment/matching.");
    return result;
  }

  sampled_temperatures.reserve(active_state_indices.size());
  sample_counts.reserve(active_state_indices.size());
  observable_means.reserve(active_state_indices.size());
  for (int state_index : active_state_indices) {
    sampled_temperatures.push_back(temps[static_cast<std::size_t>(state_index)]);
    sample_counts.push_back(all_state_counts[static_cast<std::size_t>(state_index)]);
    observable_means.push_back(
        all_state_sums[static_cast<std::size_t>(state_index)] /
        static_cast<double>(std::max(
            1, all_state_counts[static_cast<std::size_t>(state_index)])));
  }

  const double energy_min = *std::min_element(samples.begin(), samples.end());
  std::vector<double> beta_sampled(sampled_temperatures.size(), 0.0);
  for (std::size_t k = 0; k < sampled_temperatures.size(); ++k) {
    beta_sampled[k] = 1.0 / (kBoltzmannKjPerMolK *
                             std::max(1e-9, sampled_temperatures[k]));
  }

  std::vector<double> f_k(sampled_temperatures.size(), 0.0);
  SolverStats solver_stats;
  guess_free_energy_ti(beta_sampled, observable_means, f_k, energy_min);
  result.converged =
      solve_mbar(samples, beta_sampled, sample_counts, f_k, energy_min,
                 executor, false, &solver_stats);
  result.solved = true;
  result.solver_method = solver_stats.method;
  result.solver_iterations = solver_stats.iterations;
  result.solver_line_search_backtracks = solver_stats.line_search_backtracks;
  result.sampled_temperatures_k = sampled_temperatures;
  result.sampled_counts = sample_counts;
  result.active_sampled_states = sampled_temperatures.size();
  result.total_usable_samples = samples.size();
  AlignedDoubleVector shifted_samples_eval(samples.size(), 0.0);
  for (std::size_t n = 0; n < samples.size(); ++n)
    shifted_samples_eval[n] = samples[n] - energy_min;
  std::vector<double> log_nk_sampled(beta_sampled.size(), -1e12);
  for (std::size_t k = 0; k < beta_sampled.size(); ++k) {
    if (sample_counts[k] > 0)
      log_nk_sampled[k] = std::log(static_cast<double>(sample_counts[k]));
  }
  const auto final_evaluation =
      evaluate_mbar_state(shifted_samples_eval, beta_sampled, sample_counts,
                          log_nk_sampled, f_k, executor, true);
  result.residual_inf = final_evaluation.residual_inf;
  result.residual_rms = final_evaluation.residual_rms;
  result.min_state_effective_sample_size =
      compute_min_state_effective_sample_size(final_evaluation, sample_counts);
  result.min_adjacent_overlap = compute_min_adjacent_overlap(final_evaluation);

  const auto target_observable = compute_observable_at_temperature(
      samples, beta_sampled, sample_counts, f_k, args.mbar.target_temp,
      energy_min, executor, true);
  result.target_mean_energy = target_observable.mean_energy;
  result.target_heat_capacity = target_observable.heat_capacity;
  result.target_effective_sample_size =
      compute_effective_sample_size(target_observable.normalized_weights);
  result.convergence_note = build_convergence_note(
      result.converged, result.total_usable_samples,
      result.target_effective_sample_size,
      result.min_state_effective_sample_size, result.min_adjacent_overlap,
      result.residual_inf);
  if (!result.converged) {
    TitanUI::print_warning(
        "MBAR: not converged (" + result.convergence_note +
        "), residual_inf = " + std::to_string(result.residual_inf) + ".");
  }

  const auto scan_temperatures = build_temperature_scan(
      sampled_temperatures, args.mbar.target_temp, args.mbar.scan_points);
  result.cv_scan.reserve(scan_temperatures.size());
  for (double temperature : scan_temperatures) {
    auto observable = compute_observable_at_temperature(
        samples, beta_sampled, sample_counts, f_k, temperature, energy_min,
        executor, false);
    CvPoint point;
    point.temperature_k = temperature;
    point.mean_energy = observable.mean_energy;
    point.heat_capacity = observable.heat_capacity;
    result.cv_scan.push_back(point);
    if (std::abs(temperature - args.mbar.target_temp) <=
        1e-9 * std::max(1.0, args.mbar.target_temp)) {
      result.target_mean_energy = observable.mean_energy;
      result.target_heat_capacity = observable.heat_capacity;
    }
  }

  if (args.mbar.bootstrap_samples > 0) {
    std::vector<RunningStats> energy_stats(result.cv_scan.size());
    std::vector<RunningStats> cv_stats(result.cv_scan.size());
    for (int boot = 0; boot < args.mbar.bootstrap_samples; ++boot) {
      std::mt19937 rng(42u + static_cast<unsigned>(boot));
      AlignedDoubleVector boot_samples;
      boot_samples.reserve(samples.size());
      std::vector<int> boot_state_counts(temps.size(), 0);
      std::vector<double> boot_state_sums(temps.size(), 0.0);
      for (const auto &series : usable_series) {
        if (series.observables.empty())
          continue;
        std::uniform_int_distribution<std::size_t> dist(
            0, series.observables.size() - 1);
        for (std::size_t i = 0; i < series.observables.size(); ++i) {
          const std::size_t pick = dist(rng);
          const double value = series.observables[pick];
          const int state_index = series.sampled_state_indices[pick];
          if (state_index < 0 ||
              static_cast<std::size_t>(state_index) >= temps.size()) {
            continue;
          }
          boot_samples.push_back(value);
          boot_state_counts[static_cast<std::size_t>(state_index)] += 1;
          boot_state_sums[static_cast<std::size_t>(state_index)] += value;
        }
      }
      if (boot_samples.empty())
        continue;
      std::vector<double> boot_temps;
      std::vector<int> boot_counts;
      std::vector<double> boot_means;
      for (std::size_t state_idx = 0; state_idx < temps.size(); ++state_idx) {
        if (boot_state_counts[state_idx] == 0)
          continue;
        boot_temps.push_back(temps[state_idx]);
        boot_counts.push_back(boot_state_counts[state_idx]);
        boot_means.push_back(
            boot_state_sums[state_idx] /
            static_cast<double>(boot_state_counts[state_idx]));
      }
      if (boot_temps.empty())
        continue;
      const double boot_energy_min =
          *std::min_element(boot_samples.begin(), boot_samples.end());
      std::vector<double> beta_boot(boot_temps.size(), 0.0);
      for (std::size_t idx = 0; idx < boot_temps.size(); ++idx) {
        beta_boot[idx] =
            1.0 / (kBoltzmannKjPerMolK * std::max(1e-9, boot_temps[idx]));
      }
      std::vector<double> f_boot(boot_temps.size(), 0.0);
      guess_free_energy_ti(beta_boot, boot_means, f_boot, boot_energy_min);
      SolverStats boot_solver_stats;
      solve_mbar(boot_samples, beta_boot, boot_counts, f_boot,
                 boot_energy_min, executor, true, &boot_solver_stats);
      for (std::size_t idx = 0; idx < result.cv_scan.size(); ++idx) {
        const auto observable = compute_observable_at_temperature(
            boot_samples, beta_boot, boot_counts, f_boot,
            result.cv_scan[idx].temperature_k, boot_energy_min, executor, false);
        energy_stats[idx].add(observable.mean_energy);
        cv_stats[idx].add(observable.heat_capacity);
      }
    }
    for (std::size_t idx = 0; idx < result.cv_scan.size(); ++idx) {
      result.cv_scan[idx].std_energy = energy_stats[idx].stddev();
      result.cv_scan[idx].std_heat_capacity = cv_stats[idx].stddev();
      if (std::abs(result.cv_scan[idx].temperature_k - args.mbar.target_temp) <=
          1e-9 * std::max(1.0, args.mbar.target_temp)) {
        result.target_std_energy = result.cv_scan[idx].std_energy;
        result.target_std_heat_capacity =
            result.cv_scan[idx].std_heat_capacity;
      }
    }
  }

  if (have_frame_metadata) {
    int64_t max_global_index = -1;
    for (int64_t value : frame_metadata.global_indices)
      max_global_index = std::max(max_global_index, value);
    if (max_global_index >= 0) {
      result.global_frame_weights.assign(static_cast<std::size_t>(max_global_index) +
                                             1,
                                         0.0);
    }
  }

  for (std::size_t idx = 0; idx < sample_origins.size() &&
                          idx < target_observable.normalized_weights.size();
       ++idx) {
    FrameWeightRow row;
    row.replica_id = sample_origins[idx].replica_id;
    row.local_frame_index = sample_origins[idx].local_frame_index;
    row.time_ps = sample_origins[idx].time_ps;
    row.weight = target_observable.normalized_weights[idx];
    row.sampled_temperature_k = sample_origins[idx].sampled_temperature_k;
    row.sampled_state_index = sample_origins[idx].sampled_state_index;
    row.global_frame_index = sample_origins[idx].global_frame_index;
    if (row.global_frame_index >= 0 &&
        static_cast<std::size_t>(row.global_frame_index) <
            result.global_frame_weights.size()) {
      result.global_frame_weights[static_cast<std::size_t>(
          row.global_frame_index)] += row.weight;
      ++result.mapped_global_frames;
    }
    result.frame_weights.push_back(row);
  }

  TitanUI::print_info("MBAR: solved " +
                      std::to_string(result.usable_replica_series) +
                      " usable replica series across " +
                      std::to_string(result.active_sampled_states) +
                      " sampled states with " +
                      std::to_string(result.total_usable_samples) +
                      " aligned samples.");
  TitanUI::print_info("MBAR: target T = " +
                      std::to_string(args.mbar.target_temp) +
                      " K, <" + std::string(observable_symbol(use_npt)) + "> = " +
                      std::to_string(result.target_mean_energy) +
                      " kJ/mol, Cv = " +
                      std::to_string(result.target_heat_capacity) +
                      " kJ/mol/K.");
  return result;
}

void write_window_audit_file(const std::string &filename,
                             const std::vector<WindowAuditRow> &rows) {
  std::ofstream out(filename);
  out << "# replica status traj_present energy_present overlap_found "
         "truncated_start truncated_end irregular_energy_dt sparse_energy "
         "traj_frames energy_points kept_frames dropped_outside unmatched_in_overlap "
         "traj_min_ps traj_max_ps energy_min_ps energy_max_ps overlap_start_ps "
         "overlap_end_ps inferred_dt_energy_ps\n";
  out << std::setprecision(15);
  for (const auto &row : rows) {
    out << row.replica_id << ' ' << row.status << ' '
        << (row.traj_present ? 1 : 0) << ' ' << (row.energy_present ? 1 : 0)
        << ' ' << (row.overlap_found ? 1 : 0) << ' '
        << (row.truncated_start ? 1 : 0) << ' '
        << (row.truncated_end ? 1 : 0) << ' '
        << (row.irregular_energy_dt ? 1 : 0) << ' '
        << (row.sparse_energy ? 1 : 0) << ' ' << row.traj_frames << ' '
        << row.energy_points << ' ' << row.kept_frames << ' '
        << row.dropped_outside_overlap << ' ' << row.unmatched_in_overlap << ' '
        << row.traj_min_ps << ' ' << row.traj_max_ps << ' '
        << row.energy_min_ps << ' ' << row.energy_max_ps << ' '
        << row.overlap_start_ps << ' ' << row.overlap_end_ps << ' '
        << row.inferred_dt_energy_ps << '\n';
  }
}

void write_frame_weights_file(const std::string &filename,
                              const std::vector<FrameWeightRow> &rows) {
  std::ofstream out(filename);
  out << "# replica_id local_frame_index time_ps weight sampled_temperature_k "
         "sampled_state_index global_frame_index\n";
  out << std::setprecision(15);
  for (const auto &row : rows) {
    out << row.replica_id << ' ' << row.local_frame_index << ' ' << row.time_ps
        << ' ' << row.weight << ' ' << row.sampled_temperature_k << ' '
        << row.sampled_state_index << ' ' << row.global_frame_index << '\n';
  }
}

void write_cv_file(const std::string &filename,
                   const std::vector<CvPoint> &points) {
  std::ofstream out(filename);
  out << "# T_K MeanE_kJmol StdE_kJmol Cv_kJmolK StdCv_kJmolK\n";
  out << std::setprecision(15);
  for (const auto &point : points) {
    out << point.temperature_k << ' ' << point.mean_energy << ' '
        << point.std_energy << ' ' << point.heat_capacity << ' '
        << point.std_heat_capacity << '\n';
  }
}

void write_summary_file(const std::string &filename, const CoreResult &result,
                        const arguments &args, std::fstream &log) {
  const bool use_npt = (args.mbar.ensemble == "npt");
  std::ofstream out(filename);
  out << std::fixed << std::setprecision(6);
  out << "MBAR Converged                 : "
      << (result.converged ? "yes" : "no") << "\n";
  out << "MBAR Abort Reason              : "
      << (result.abort_reason.empty() ? "n/a" : result.abort_reason) << "\n";
  out << "MBAR Domain                    : " << args.mbar.domain << "\n";
  out << "MBAR Ensemble                  : " << args.mbar.ensemble << "\n";
  out << "MBAR Temp Mode                 : " << args.mbar.temp_mode << "\n";
  out << "MBAR Ladder Source             : "
      << result.temperature_ladder_source << "\n";
  out << "MBAR Window Match              : " << args.mbar.window_match << "\n";
  out << "MBAR Demux Map                 : "
      << (result.used_demux_map ? args.mbar.demux_map_file : "n/a") << "\n";
  out << "MBAR Temp Column               : " << args.mbar.temp_column << "\n";
  out << "MBAR Pressure (bar)            : " << args.mbar.target_pressure_bar
      << "\n";
  out << "MBAR Observable                : " << observable_name(use_npt) << "\n";
  out << "MBAR Target Temperature (K)    : " << result.target_temp_k << "\n";
  out << "MBAR Usable Replica Series     : "
      << result.usable_replica_series << "\n";
  out << "MBAR Active Sampled States     : " << result.active_sampled_states
      << "\n";
  out << "MBAR Usable Samples            : " << result.total_usable_samples
      << "\n";
  out << "MBAR Mapped Global Frames      : " << result.mapped_global_frames
      << "\n";
  out << "MBAR Mapped TICA Frames        : "
      << result.mapped_tica_global_frames << "\n";
  out << "MBAR Mapped PCA Frames         : "
      << result.mapped_pca_global_frames << "\n";
  out << "MBAR Box Volume Metadata       : "
      << (result.box_volume_metadata_available ? "available" : "not_used")
      << "\n";
  out << "MBAR Solver Method             : " << result.solver_method << "\n";
  out << "MBAR Solver Iterations         : " << result.solver_iterations << "\n";
  out << "MBAR Solver Backtracks         : "
      << result.solver_line_search_backtracks << "\n";
  out << "MBAR Residual Inf              : " << result.residual_inf << "\n";
  out << "MBAR Residual RMS              : " << result.residual_rms << "\n";
  out << "MBAR Target ESS               : "
      << result.target_effective_sample_size << "\n";
  out << "MBAR Min State ESS             : "
      << result.min_state_effective_sample_size << "\n";
  out << "MBAR Min Adjacent Overlap      : " << result.min_adjacent_overlap
      << "\n";
  out << "MBAR Convergence Note          : " << result.convergence_note << "\n";
  out << "MBAR Bootstrap Samples         : " << args.mbar.bootstrap_samples
      << "\n";
  out << "MBAR Scan Points Requested     : " << args.mbar.scan_points << "\n";
  out << "MBAR Scan Points Written       : " << result.cv_scan.size() << "\n";
  out << "MBAR Target <" << observable_symbol(use_npt)
      << "> (kJ/mol)       : " << result.target_mean_energy << "\n";
  out << "MBAR Target Std" << observable_symbol(use_npt)
      << " (kJ/mol)      : " << result.target_std_energy << "\n";
  out << "MBAR Target Cv (kJ/mol/K)      : " << result.target_heat_capacity
      << "\n";
  out << "MBAR Target StdCv (kJ/mol/K)   : "
      << result.target_std_heat_capacity << "\n";
  out << "MBAR Sampled Temperatures (K)  :";
  for (double value : result.sampled_temperatures_k)
    out << ' ' << value;
  out << "\n";
  out << "MBAR Sample Counts             :";
  for (int value : result.sampled_counts)
    out << ' ' << value;
  out << "\n";

  TitanUI::log_kv(log, "MBAR Converged", result.converged ? "yes" : "no");
  TitanUI::log_kv(log, "MBAR Domain", args.mbar.domain);
  TitanUI::log_kv(log, "MBAR Ensemble", args.mbar.ensemble);
  TitanUI::log_kv(log, "MBAR Temp Mode", args.mbar.temp_mode);
  TitanUI::log_kv(log, "MBAR Ladder Source",
                  result.temperature_ladder_source);
  TitanUI::log_kv(log, "MBAR Window Match", args.mbar.window_match);
  TitanUI::log_kv(log, "MBAR Solver Method", result.solver_method);
  TitanUI::log_kv(log, "MBAR Solver Iterations",
                  std::to_string(result.solver_iterations));
  TitanUI::log_kv(log, "MBAR Residual Inf",
                  std::to_string(result.residual_inf));
  TitanUI::log_kv(log, "MBAR Target ESS",
                  std::to_string(result.target_effective_sample_size));
  TitanUI::log_kv(log, "MBAR Min State ESS",
                  std::to_string(result.min_state_effective_sample_size));
  TitanUI::log_kv(log, "MBAR Min Adjacent Overlap",
                  std::to_string(result.min_adjacent_overlap));
  TitanUI::log_kv(log, "MBAR Convergence Note", result.convergence_note);
  TitanUI::log_kv(log, "MBAR Usable Samples",
                  std::to_string(result.total_usable_samples));
  TitanUI::log_kv(log, "MBAR Summary File", filename);
}

} // namespace TitanMBAR

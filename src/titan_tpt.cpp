#include "titan_tpt.h"
#include "titan_executor.h"
#include "titan_thread_compat.h"

#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {

constexpr double kTptEps = 1e-14;
constexpr double kNoiseGuardScale = 1e-12;

using SparseMatrixCM = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

template <typename T> T clamp01(T value) {
  return std::clamp(value, static_cast<T>(0), static_cast<T>(1));
}

std::string join_nodes(const std::vector<int> &nodes) {
  std::ostringstream oss;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (i > 0)
      oss << "->";
    oss << nodes[i];
  }
  return oss.str();
}

} // namespace

TitanTPT::TitanTPT(const Eigen::MatrixXd &tpm,
                   const Eigen::VectorXd &stationary_dist)
    : n_states_(static_cast<int>(tpm.rows())),
      stationary_dist_(stationary_dist),
      q_forward_(Eigen::VectorXd::Zero(tpm.rows())),
      q_backward_(Eigen::VectorXd::Zero(tpm.rows())),
      reactive_density_(Eigen::VectorXd::Zero(tpm.rows())),
      gross_out_flux_(Eigen::VectorXd::Zero(tpm.rows())),
      gross_in_flux_(Eigen::VectorXd::Zero(tpm.rows())),
      net_out_flux_(Eigen::VectorXd::Zero(tpm.rows())),
      net_in_flux_(Eigen::VectorXd::Zero(tpm.rows())) {
  if (tpm.rows() != tpm.cols()) {
    throw std::invalid_argument("TPT: transition matrix must be square");
  }
  if (stationary_dist.size() != tpm.rows()) {
    throw std::invalid_argument(
        "TPT: stationary distribution size does not match TPM");
  }

  build_sparse_transition_backend(tpm);
}

std::string TitanTPT::trim_copy(const std::string &value) {
  const auto first =
      std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
      });
  if (first == value.end())
    return {};
  const auto last =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
      }).base();
  return std::string(first, last);
}

bool TitanTPT::parse_state_spec(const std::string &spec, int n_states,
                                std::vector<int> &states_out,
                                std::string *error_message) {
  auto fail = [&](const std::string &message) {
    if (error_message)
      *error_message = message;
    states_out.clear();
    return false;
  };

  std::string body = trim_copy(spec);
  if (body.empty())
    return fail("state specification is empty");

  const auto colon_pos = body.find(':');
  if (colon_pos != std::string::npos) {
    std::string prefix = trim_copy(body.substr(0, colon_pos));
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (prefix != "micro" && prefix != "state" && prefix != "states") {
      return fail("unsupported basin prefix '" + prefix +
                  "' (expected micro)");
    }
    body = trim_copy(body.substr(colon_pos + 1));
  }

  if (body.empty())
    return fail("no microstate indices found after basin prefix");

  std::vector<int> values;
  std::stringstream ss(body);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (token.empty())
      return fail("empty token inside state specification");

    const auto dash_pos = token.find('-');
    if (dash_pos == std::string::npos) {
      char *end_ptr = nullptr;
      const long parsed = std::strtol(token.c_str(), &end_ptr, 10);
      if (end_ptr == token.c_str() || *end_ptr != '\0') {
        return fail("invalid microstate token '" + token + "'");
      }
      if (parsed < 0) {
        return fail("microstate indices must be >= 0");
      }
      if (n_states >= 0 && parsed >= n_states) {
        return fail("microstate index " + std::to_string(parsed) +
                    " is out of range for K=" + std::to_string(n_states));
      }
      values.push_back(static_cast<int>(parsed));
      continue;
    }

    if (dash_pos == 0 || dash_pos + 1 >= token.size()) {
      return fail("invalid range token '" + token + "'");
    }

    const std::string first_token = trim_copy(token.substr(0, dash_pos));
    const std::string last_token = trim_copy(token.substr(dash_pos + 1));
    char *end_first = nullptr;
    char *end_last = nullptr;
    const long first_val = std::strtol(first_token.c_str(), &end_first, 10);
    const long last_val = std::strtol(last_token.c_str(), &end_last, 10);
    if (end_first == first_token.c_str() || *end_first != '\0' ||
        end_last == last_token.c_str() || *end_last != '\0') {
      return fail("invalid range token '" + token + "'");
    }
    if (first_val < 0 || last_val < 0) {
      return fail("microstate indices must be >= 0");
    }
    if (first_val > last_val) {
      return fail("descending ranges are not supported in '" + token + "'");
    }
    if (n_states >= 0 && last_val >= n_states) {
      return fail("microstate index " + std::to_string(last_val) +
                  " is out of range for K=" + std::to_string(n_states));
    }
    for (long v = first_val; v <= last_val; ++v) {
      values.push_back(static_cast<int>(v));
    }
  }

  if (values.empty())
    return fail("no valid microstate indices were parsed");

  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  states_out = std::move(values);
  return true;
}

std::string TitanTPT::format_state_list(const std::vector<int> &states) {
  if (states.empty())
    return "[]";
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < states.size(); ++i) {
    if (i > 0)
      oss << ",";
    oss << states[i];
  }
  oss << "]";
  return oss.str();
}

void TitanTPT::build_sparse_transition_backend(
    const Eigen::MatrixXd &transition_matrix) {
  for (int i = 0; i < stationary_dist_.size(); ++i) {
    const double value = stationary_dist_[i];
    stationary_dist_[i] = (std::isfinite(value) && value > 0.0) ? value : 0.0;
  }
  const double pi_sum = stationary_dist_.sum();
  if (pi_sum <= kTptEps) {
    stationary_dist_.setConstant(1.0 / std::max(1, n_states_));
  } else {
    stationary_dist_ /= pi_sum;
  }

  std::vector<int> tpm_row_offsets(static_cast<size_t>(n_states_) + 1, 0);
  std::vector<int> tpm_col_indices;
  std::vector<double> tpm_values;

  for (int i = 0; i < n_states_; ++i) {
    tpm_row_offsets[static_cast<size_t>(i)] =
        static_cast<int>(tpm_col_indices.size());

    double row_sum = 0.0;
    std::vector<int> row_cols;
    std::vector<double> row_values;
    row_cols.reserve(static_cast<size_t>(n_states_));
    row_values.reserve(static_cast<size_t>(n_states_));

    for (int j = 0; j < n_states_; ++j) {
      double value = transition_matrix(i, j);
      if (!std::isfinite(value) || value < 0.0)
        value = 0.0;
      if (value <= 0.0)
        continue;
      row_cols.push_back(j);
      row_values.push_back(value);
      row_sum += value;
    }

    if (row_sum <= 0.0) {
      tpm_col_indices.push_back(i);
      tpm_values.push_back(1.0);
      continue;
    }

    for (size_t idx = 0; idx < row_cols.size(); ++idx) {
      tpm_col_indices.push_back(row_cols[idx]);
      tpm_values.push_back(row_values[idx] / row_sum);
    }
  }
  tpm_row_offsets[static_cast<size_t>(n_states_)] =
      static_cast<int>(tpm_col_indices.size());
  tpm_graph_.assign(n_states_, std::move(tpm_row_offsets),
                    std::move(tpm_col_indices), std::move(tpm_values));
  reverse_edge_lookup_.assign(tpm_graph_.values().size(), -1);
  auto edge_key = [](int from, int to) -> std::uint64_t {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(from)) << 32) |
           static_cast<std::uint32_t>(to);
  };
  std::unordered_map<std::uint64_t, int> edge_index_map;
  edge_index_map.reserve(tpm_graph_.values().size() * 2);
  for (int from = 0; from < n_states_; ++from) {
    for (int edge_idx = tpm_graph_.row_begin(from);
         edge_idx < tpm_graph_.row_end(from); ++edge_idx) {
      const int to = tpm_graph_.col_index(edge_idx);
      edge_index_map.emplace(edge_key(from, to), edge_idx);
    }
  }
  for (int from = 0; from < n_states_; ++from) {
    for (int edge_idx = tpm_graph_.row_begin(from);
         edge_idx < tpm_graph_.row_end(from); ++edge_idx) {
      const int to = tpm_graph_.col_index(edge_idx);
      const auto it = edge_index_map.find(edge_key(to, from));
      if (it != edge_index_map.end()) {
        reverse_edge_lookup_[static_cast<size_t>(edge_idx)] = it->second;
      }
    }
  }
  std::vector<int> reverse_counts(static_cast<size_t>(n_states_), 0);
  std::vector<double> reverse_row_sums(static_cast<size_t>(n_states_), 0.0);
  for (int from = 0; from < n_states_; ++from) {
    for (int edge_idx = tpm_graph_.row_begin(from);
         edge_idx < tpm_graph_.row_end(from);
         ++edge_idx) {
      const int to = tpm_graph_.col_index(edge_idx);
      const double pi_to = stationary_dist_[to];
      if (pi_to <= 0.0)
        continue;
      const double raw =
          (stationary_dist_[from] * tpm_graph_.value(edge_idx)) / pi_to;
      if (!std::isfinite(raw) || raw <= 0.0)
        continue;
      ++reverse_counts[static_cast<size_t>(to)];
      reverse_row_sums[static_cast<size_t>(to)] += raw;
    }
  }

  std::vector<int> reverse_row_offsets(static_cast<size_t>(n_states_) + 1, 0);
  for (int i = 0; i < n_states_; ++i) {
    const bool use_self_loop =
        reverse_row_sums[static_cast<size_t>(i)] <= 0.0;
    reverse_row_offsets[static_cast<size_t>(i + 1)] =
        reverse_row_offsets[static_cast<size_t>(i)] +
        (use_self_loop ? 1 : reverse_counts[static_cast<size_t>(i)]);
  }

  std::vector<int> reverse_col_indices(
      static_cast<size_t>(reverse_row_offsets[static_cast<size_t>(n_states_)]),
      0);
  std::vector<double> reverse_values(reverse_col_indices.size(), 0.0);
  std::vector<int> reverse_write = reverse_row_offsets;

  for (int i = 0; i < n_states_; ++i) {
    if (reverse_row_sums[static_cast<size_t>(i)] <= kTptEps) {
      const int pos = reverse_write[static_cast<size_t>(i)]++;
      reverse_col_indices[static_cast<size_t>(pos)] = i;
      reverse_values[static_cast<size_t>(pos)] = 1.0;
    }
  }

  for (int from = 0; from < n_states_; ++from) {
    for (int edge_idx = tpm_graph_.row_begin(from);
         edge_idx < tpm_graph_.row_end(from);
         ++edge_idx) {
      const int to = tpm_graph_.col_index(edge_idx);
      const double row_sum = reverse_row_sums[static_cast<size_t>(to)];
      if (row_sum <= 0.0)
        continue;
      const double pi_to = stationary_dist_[to];
      if (pi_to <= 0.0)
        continue;
      const double raw =
          (stationary_dist_[from] * tpm_graph_.value(edge_idx)) / pi_to;
      if (!std::isfinite(raw) || raw <= 0.0)
        continue;
      const int pos = reverse_write[static_cast<size_t>(to)]++;
      reverse_col_indices[static_cast<size_t>(pos)] = from;
      reverse_values[static_cast<size_t>(pos)] = raw / row_sum;
    }
  }

  reverse_graph_.assign(n_states_, std::move(reverse_row_offsets),
                        std::move(reverse_col_indices),
                        std::move(reverse_values));
}

void TitanTPT::validate_basins(const std::vector<int> &source_states,
                               const std::vector<int> &sink_states) const {
  if (source_states.empty()) {
    throw std::invalid_argument("TPT: source basin is empty");
  }
  if (sink_states.empty()) {
    throw std::invalid_argument("TPT: sink basin is empty");
  }
  std::vector<unsigned char> used(static_cast<size_t>(n_states_), 0);
  for (int state : source_states) {
    if (state < 0 || state >= n_states_) {
      throw std::invalid_argument("TPT: source state out of range");
    }
    used[static_cast<size_t>(state)] = 1;
  }
  for (int state : sink_states) {
    if (state < 0 || state >= n_states_) {
      throw std::invalid_argument("TPT: sink state out of range");
    }
    if (used[static_cast<size_t>(state)] != 0) {
      throw std::invalid_argument("TPT: source and sink basins overlap");
    }
  }
}

void TitanTPT::build_membership_masks() {
  is_source_.assign(static_cast<size_t>(n_states_), 0);
  is_sink_.assign(static_cast<size_t>(n_states_), 0);
  for (int state : source_states_) {
    is_source_[static_cast<size_t>(state)] = 1;
  }
  for (int state : sink_states_) {
    is_sink_[static_cast<size_t>(state)] = 1;
  }
}

void TitanTPT::solve_committor_system(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const std::vector<unsigned char> &one_boundary,
    const std::vector<unsigned char> &zero_boundary,
    Eigen::VectorXd &out_committor) const {
  out_committor.setZero();
  std::vector<int> intermediate_states;
  intermediate_states.reserve(static_cast<size_t>(n_states_));
  for (int i = 0; i < n_states_; ++i) {
    if (one_boundary[static_cast<size_t>(i)] != 0) {
      out_committor[i] = 1.0;
    } else if (zero_boundary[static_cast<size_t>(i)] == 0) {
      intermediate_states.push_back(i);
    }
  }

  if (intermediate_states.empty())
    return;

  const int n_mid = static_cast<int>(intermediate_states.size());
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_mid);
  std::vector<int> state_to_mid(static_cast<size_t>(n_states_), -1);
  for (int row = 0; row < n_mid; ++row) {
    state_to_mid[static_cast<size_t>(intermediate_states[static_cast<size_t>(row)])] =
        row;
  }

  auto solve_dense_fallback = [&]() {
    Eigen::MatrixXd lhs = Eigen::MatrixXd::Zero(n_mid, n_mid);
    for (int row = 0; row < n_mid; ++row) {
      const int i = intermediate_states[static_cast<size_t>(row)];
      lhs(row, row) = 1.0;
      double one_mass = 0.0;

      for (int edge_idx = graph.row_begin(i); edge_idx < graph.row_end(i);
           ++edge_idx) {
        const int j = graph.col_index(edge_idx);
        const double value = graph.value(edge_idx);
        if (value <= 0.0)
          continue;
        if (one_boundary[static_cast<size_t>(j)] != 0) {
          one_mass += value;
          continue;
        }
        if (zero_boundary[static_cast<size_t>(j)] != 0)
          continue;
        const int col = state_to_mid[static_cast<size_t>(j)];
        if (col < 0)
          continue;
        lhs(row, col) -= value;
      }
      rhs[row] = one_mass;
    }
    return lhs.colPivHouseholderQr().solve(rhs);
  };

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(n_mid) *
                   static_cast<size_t>(std::min(n_states_, 16)) +
                   static_cast<size_t>(n_mid));

  for (int row = 0; row < n_mid; ++row) {
    const int i = intermediate_states[static_cast<size_t>(row)];
    double diag = 1.0;
    double one_mass = 0.0;

    for (int edge_idx = graph.row_begin(i); edge_idx < graph.row_end(i);
         ++edge_idx) {
      const int j = graph.col_index(edge_idx);
      const double value = graph.value(edge_idx);
      if (value <= 0.0)
        continue;

      if (one_boundary[static_cast<size_t>(j)] != 0) {
        one_mass += value;
        continue;
      }
      if (zero_boundary[static_cast<size_t>(j)] != 0)
        continue;

      const int col = state_to_mid[static_cast<size_t>(j)];
      if (col < 0)
        continue;
      if (col == row) {
        diag -= value;
      } else {
        triplets.emplace_back(row, col, -value);
      }
    }

    rhs[row] = one_mass;
    triplets.emplace_back(row, row, diag);
  }

  SparseMatrixCM lhs(n_mid, n_mid);
  lhs.setFromTriplets(triplets.begin(), triplets.end());
  lhs.makeCompressed();

  Eigen::SparseLU<SparseMatrixCM, Eigen::COLAMDOrdering<int>> solver;
  Eigen::VectorXd solution;
  solver.compute(lhs);
  if (solver.info() != Eigen::Success) {
    solution = solve_dense_fallback();
  } else {
    solution = solver.solve(rhs);
    if (solver.info() != Eigen::Success || !solution.allFinite()) {
      solution = solve_dense_fallback();
    }
  }

  for (int row = 0; row < n_mid; ++row) {
    const int i = intermediate_states[static_cast<size_t>(row)];
    out_committor[i] = clamp01(solution[row]);
  }
}

void TitanTPT::compute_reactive_densities() {
  for (int i = 0; i < n_states_; ++i) {
    reactive_density_[i] =
        stationary_dist_[i] * q_forward_[i] * q_backward_[i];
  }
}


void TitanTPT::compute_flux_matrices(TitanExecutor *executor, int n_threads,
                                     double flux_cut) {
  gross_out_flux_.setZero();
  gross_in_flux_.setZero();
  net_out_flux_.setZero();
  net_in_flux_.setZero();
  edges_.clear();

  const int max_threads = (executor != nullptr)
                              ? std::max(1, static_cast<int>(executor->capacity()))
                              : std::max(1, static_cast<int>(
                                               std::thread::hardware_concurrency()));
  const int threads = std::clamp(n_threads, 1, max_threads);
  const bool use_threads = (threads > 1 && n_states_ >= 64);

  const int worker_count = use_threads ? threads : 1;
  std::vector<Eigen::VectorXd> partial_gross_out(
      static_cast<size_t>(worker_count), Eigen::VectorXd::Zero(n_states_));
  std::vector<Eigen::VectorXd> partial_gross_in(
      static_cast<size_t>(worker_count), Eigen::VectorXd::Zero(n_states_));
  std::vector<Eigen::VectorXd> partial_net_out(
      static_cast<size_t>(worker_count), Eigen::VectorXd::Zero(n_states_));
  std::vector<Eigen::VectorXd> partial_net_in(
      static_cast<size_t>(worker_count), Eigen::VectorXd::Zero(n_states_));
  std::vector<std::vector<EdgeFlux>> partial_edges(
      static_cast<size_t>(worker_count));

  auto compute_rows = [&](int worker_idx, int row_begin, int row_end) {
    auto &gross_out_local = partial_gross_out[static_cast<size_t>(worker_idx)];
    auto &gross_in_local = partial_gross_in[static_cast<size_t>(worker_idx)];
    auto &net_out_local = partial_net_out[static_cast<size_t>(worker_idx)];
    auto &net_in_local = partial_net_in[static_cast<size_t>(worker_idx)];
    auto &edge_local = partial_edges[static_cast<size_t>(worker_idx)];

    for (int i = row_begin; i < row_end; ++i) {
      const double prefactor = stationary_dist_[i] * q_backward_[i];
      for (int edge_idx = tpm_graph_.row_begin(i);
           edge_idx < tpm_graph_.row_end(i); ++edge_idx) {
        const int j = tpm_graph_.col_index(edge_idx);
        if (i == j)
          continue;

        const double probability = tpm_graph_.value(edge_idx);
        const double fij =
            (prefactor <= 0.0) ? 0.0 : prefactor * probability * q_forward_[j];
        if (fij > 0.0) {
          gross_out_local[i] += fij;
          gross_in_local[j] += fij;
        }

        const int reverse_edge_idx = reverse_edge_lookup_[static_cast<size_t>(edge_idx)];
        const bool reverse_exists = reverse_edge_idx >= 0;
        if (i > j && reverse_exists)
          continue;

        double fji = 0.0;
        if (reverse_exists) {
          const double reverse_probability = tpm_graph_.value(reverse_edge_idx);
          const double reverse_prefactor = stationary_dist_[j] * q_backward_[j];
          if (reverse_prefactor > 0.0 && reverse_probability > 0.0) {
            fji = reverse_prefactor * reverse_probability * q_forward_[i];
          }
        }
        if (fij > fji) {
          const double net = fij - fji;
          if (net <= 0.0)
            continue;
          net_out_local[i] += net;
          net_in_local[j] += net;
          edge_local.push_back(EdgeFlux{i, j, fij, net});
        } else if (fji > fij) {
          const double net = fji - fij;
          if (net <= 0.0)
            continue;
          net_out_local[j] += net;
          net_in_local[i] += net;
          edge_local.push_back(EdgeFlux{j, i, fji, net});
        } else if (!reverse_exists && fij > 0.0) {
          net_out_local[i] += fij;
          net_in_local[j] += fij;
          edge_local.push_back(EdgeFlux{i, j, fij, fij});
        }
      }
    }
  };

  if (!use_threads) {
    compute_rows(0, 0, n_states_);
  } else if (executor != nullptr) {
    const int chunk = (n_states_ + threads - 1) / threads;
    if (threads > 1) {
      executor->parallel_do(static_cast<size_t>(threads), [&](size_t tid_sz) {
        const int tid = static_cast<int>(tid_sz);
        const int row_begin = tid * chunk;
        const int row_end = std::min(row_begin + chunk, n_states_);
        compute_rows(tid, row_begin, row_end);
      });
    } else {
      compute_rows(0, 0, n_states_);
    }
  } else {
    std::vector<titan::jthread> workers;
    workers.reserve(static_cast<size_t>(threads));
    const int chunk = (n_states_ + threads - 1) / threads;
    for (int tid = 0; tid < threads; ++tid) {
      const int row_begin = tid * chunk;
      const int row_end = std::min(row_begin + chunk, n_states_);
      if (row_begin >= row_end)
        continue;
      workers.emplace_back(
          [&, tid, row_begin, row_end]() { compute_rows(tid, row_begin, row_end); });
    }
  }

  for (int tid = 0; tid < worker_count; ++tid) {
    gross_out_flux_ += partial_gross_out[static_cast<size_t>(tid)];
    gross_in_flux_ += partial_gross_in[static_cast<size_t>(tid)];
    net_out_flux_ += partial_net_out[static_cast<size_t>(tid)];
    net_in_flux_ += partial_net_in[static_cast<size_t>(tid)];
  }

  double total_flux = 0.0;
  double pi_source = 0.0;
  double pi_sink = 0.0;
  for (int state : source_states_) {
    total_flux += gross_out_flux_[state];
    pi_source += stationary_dist_[state];
  }
  for (int state : sink_states_) {
    pi_sink += stationary_dist_[state];
  }

  summary_.n_states = n_states_;
  summary_.n_source = static_cast<int>(source_states_.size());
  summary_.n_sink = static_cast<int>(sink_states_.size());
  summary_.n_intermediate = n_states_ - summary_.n_source - summary_.n_sink;
  summary_.source_population = pi_source;
  summary_.sink_population = pi_sink;
  summary_.intermediate_population =
      std::max(0.0, 1.0 - pi_source - pi_sink);
  summary_.total_reactive_flux = total_flux;
  summary_.rate_source_to_sink_per_lag = 0.0;
  summary_.mfpt_source_to_sink_lags = std::numeric_limits<double>::infinity();
  if (pi_source > kTptEps) {
    summary_.rate_source_to_sink_per_lag = total_flux / pi_source;
    if (summary_.rate_source_to_sink_per_lag > kTptEps) {
      summary_.mfpt_source_to_sink_lags =
          1.0 / summary_.rate_source_to_sink_per_lag;
    }
  }

  const double noise_floor =
      std::max(flux_cut, summary_.total_reactive_flux * kNoiseGuardScale);
  size_t total_edge_candidates = 0;
  for (const auto &worker_edges : partial_edges) {
    total_edge_candidates += worker_edges.size();
  }
  edges_.reserve(total_edge_candidates);
  double max_flux = 0.0;
  for (const auto &worker_edges : partial_edges) {
    for (const auto &edge : worker_edges) {
      if (edge.net_flux <= noise_floor)
        continue;
      max_flux = std::max(max_flux, edge.net_flux);
      edges_.push_back(edge);
    }
  }

  std::sort(edges_.begin(), edges_.end(),
            [](const EdgeFlux &lhs, const EdgeFlux &rhs) {
              if (lhs.net_flux != rhs.net_flux)
                return lhs.net_flux > rhs.net_flux;
              if (lhs.gross_flux != rhs.gross_flux)
                return lhs.gross_flux > rhs.gross_flux;
              if (lhs.from != rhs.from)
                return lhs.from < rhs.from;
              return lhs.to < rhs.to;
            });
  summary_.max_net_flux = max_flux;
}

void TitanTPT::extract_dominant_paths(int top_paths, double flux_cut) {
  paths_.clear();
  if (top_paths <= 0 || summary_.total_reactive_flux <= flux_cut ||
      edges_.empty())
    return;

  std::vector<EdgeFlux> residual = edges_;
  std::vector<std::vector<int>> outgoing(static_cast<size_t>(n_states_));
  for (int edge_idx = 0; edge_idx < static_cast<int>(residual.size()); ++edge_idx) {
    outgoing[static_cast<size_t>(residual[static_cast<size_t>(edge_idx)].from)]
        .push_back(edge_idx);
  }
  const double effective_cut =
      std::max(flux_cut, summary_.total_reactive_flux * kNoiseGuardScale);

  for (int rank = 0; rank < top_paths; ++rank) {
    std::vector<double> best(static_cast<size_t>(n_states_), 0.0);
    std::vector<int> predecessor_state(static_cast<size_t>(n_states_), -1);
    std::vector<int> predecessor_edge(static_cast<size_t>(n_states_), -1);
    using QueueItem = std::pair<double, int>;
    std::priority_queue<QueueItem> queue;

    for (int state : source_states_) {
      best[static_cast<size_t>(state)] =
          std::numeric_limits<double>::infinity();
      queue.emplace(best[static_cast<size_t>(state)], state);
    }

    while (!queue.empty()) {
      const auto [capacity, node] = queue.top();
      queue.pop();
      if (capacity + effective_cut < best[static_cast<size_t>(node)])
        continue;

      for (int edge_idx : outgoing[static_cast<size_t>(node)]) {
        const auto &edge = residual[static_cast<size_t>(edge_idx)];
        const double edge_flux = edge.net_flux;
        if (edge_flux <= effective_cut)
          continue;
        const int next = edge.to;
        const double candidate = std::min(capacity, edge_flux);
        if (candidate > best[static_cast<size_t>(next)] + effective_cut) {
          best[static_cast<size_t>(next)] = candidate;
          predecessor_state[static_cast<size_t>(next)] = node;
          predecessor_edge[static_cast<size_t>(next)] = edge_idx;
          queue.emplace(candidate, next);
        }
      }
    }

    int best_sink = -1;
    double bottleneck = 0.0;
    for (int sink_state : sink_states_) {
      const double candidate = best[static_cast<size_t>(sink_state)];
      if (candidate > bottleneck) {
        bottleneck = candidate;
        best_sink = sink_state;
      }
    }

    if (best_sink < 0 || bottleneck <= effective_cut)
      break;

    std::vector<int> nodes;
    std::vector<int> path_edge_indices;
    for (int current = best_sink; current >= 0;
         current = predecessor_state[static_cast<size_t>(current)]) {
      nodes.push_back(current);
      const int edge_idx = predecessor_edge[static_cast<size_t>(current)];
      if (edge_idx >= 0)
        path_edge_indices.push_back(edge_idx);
      if (is_source_[static_cast<size_t>(current)] != 0)
        break;
    }
    std::reverse(nodes.begin(), nodes.end());
    std::reverse(path_edge_indices.begin(), path_edge_indices.end());

    if (nodes.size() < 2 || is_source_[static_cast<size_t>(nodes.front())] == 0 ||
        is_sink_[static_cast<size_t>(nodes.back())] == 0) {
      break;
    }

    for (int edge_idx : path_edge_indices) {
      auto &edge = residual[static_cast<size_t>(edge_idx)];
      edge.net_flux = std::max(0.0, edge.net_flux - bottleneck);
    }

    Pathway path;
    path.nodes = std::move(nodes);
    path.flux = bottleneck;
    path.fraction =
        (summary_.total_reactive_flux > kTptEps)
            ? bottleneck / summary_.total_reactive_flux
            : 0.0;
    paths_.push_back(std::move(path));
  }
}

void TitanTPT::compute(const std::vector<int> &source_states,
                       const std::vector<int> &sink_states, int top_paths,
                       double flux_cut, TitanExecutor *executor, int n_threads) {
  validate_basins(source_states, sink_states);
  source_states_ = source_states;
  sink_states_ = sink_states;
  build_membership_masks();

  solve_committor_system(tpm_graph_, is_sink_, is_source_, q_forward_);
  solve_committor_system(reverse_graph_, is_source_, is_sink_, q_backward_);

  for (int state : source_states_) {
    q_forward_[state] = 0.0;
    q_backward_[state] = 1.0;
  }
  for (int state : sink_states_) {
    q_forward_[state] = 1.0;
    q_backward_[state] = 0.0;
  }

  q_forward_ = q_forward_.unaryExpr(
      [](double value) { return clamp01(value); });
  q_backward_ = q_backward_.unaryExpr(
      [](double value) { return clamp01(value); });

  compute_reactive_densities();
  compute_flux_matrices(executor, n_threads, flux_cut);
  const double noise_floor =
      std::max(flux_cut, summary_.total_reactive_flux * kNoiseGuardScale);
  extract_dominant_paths(top_paths, noise_floor);
  computed_ = true;
}

void TitanTPT::write_results(const std::string &summary_file,
                             const std::string &committor_file,
                             const std::string &flux_edges_file,
                             const std::string &paths_file, std::fstream &log,
                             const std::string &mode_label,
                             bool dham_requested,
                             const std::string &source_label,
                             const std::string &sink_label,
                             const std::string &model_source_label) const {
  if (!computed_) {
    throw std::runtime_error("TPT: write_results called before compute()");
  }

  {
    std::ofstream out(summary_file);
    out << "# TITAN TPT summary\n";
    out << "mode " << mode_label << "\n";
    if (!source_label.empty())
      out << "source_spec " << source_label << "\n";
    if (!sink_label.empty())
      out << "sink_spec " << sink_label << "\n";
    out << "source_states " << format_state_list(source_states_) << "\n";
    out << "sink_states " << format_state_list(sink_states_) << "\n";
    out << "n_states " << summary_.n_states << "\n";
    out << "n_source " << summary_.n_source << "\n";
    out << "n_sink " << summary_.n_sink << "\n";
    out << "n_intermediate " << summary_.n_intermediate << "\n";
    out << std::setprecision(15);
    out << "source_population " << summary_.source_population << "\n";
    out << "sink_population " << summary_.sink_population << "\n";
    out << "intermediate_population " << summary_.intermediate_population
        << "\n";
    out << "total_reactive_flux " << summary_.total_reactive_flux << "\n";
    out << "max_net_flux " << summary_.max_net_flux << "\n";
    out << "rate_source_to_sink_per_lag "
        << summary_.rate_source_to_sink_per_lag << "\n";
    out << "mfpt_source_to_sink_lags " << summary_.mfpt_source_to_sink_lags
        << "\n";
    out << "n_positive_flux_edges " << edges_.size() << "\n";
    out << "n_paths " << paths_.size() << "\n";
    out << "dham_requested " << (dham_requested ? "yes" : "no") << "\n";
    out << "model_source "
        << (model_source_label.empty() ? "msm_built_model" : model_source_label)
        << "\n";
  }

  {
    std::ofstream out(committor_file);
    out << "# state set pi q_plus q_minus reactive_density gross_out gross_in "
           "net_out net_in\n";
    out << std::setprecision(15);
    for (int i = 0; i < n_states_; ++i) {
      char label = 'I';
      if (is_source_[static_cast<size_t>(i)] != 0)
        label = 'A';
      else if (is_sink_[static_cast<size_t>(i)] != 0)
        label = 'B';
      out << i << ' ' << label << ' ' << stationary_dist_[i] << ' '
          << q_forward_[i] << ' ' << q_backward_[i] << ' '
          << reactive_density_[i] << ' ' << gross_out_flux_[i] << ' '
          << gross_in_flux_[i] << ' ' << net_out_flux_[i] << ' '
          << net_in_flux_[i] << '\n';
    }
  }

  {
    std::ofstream out(flux_edges_file);
    out << "# from to gross_flux net_flux net_flux_fraction\n";
    out << std::setprecision(15);
    for (const auto &edge : edges_) {
      const double fraction =
          (summary_.total_reactive_flux > kTptEps)
              ? edge.net_flux / summary_.total_reactive_flux
              : 0.0;
      out << edge.from << ' ' << edge.to << ' ' << edge.gross_flux << ' '
          << edge.net_flux << ' ' << fraction << '\n';
    }
  }

  {
    std::ofstream out(paths_file);
    out << "# rank flux flux_fraction n_nodes path\n";
    out << std::setprecision(15);
    for (size_t rank = 0; rank < paths_.size(); ++rank) {
      const auto &path = paths_[rank];
      out << (rank + 1) << ' ' << path.flux << ' ' << path.fraction << ' '
          << path.nodes.size() << ' ' << join_nodes(path.nodes) << '\n';
    }
  }

  TitanUI::log_kv(log, "TPT Source States", format_state_list(source_states_));
  TitanUI::log_kv(log, "TPT Sink States", format_state_list(sink_states_));
  TitanUI::log_kv(log, "TPT Total Reactive Flux",
                  std::to_string(summary_.total_reactive_flux));
  TitanUI::log_kv(log, "TPT Source Population",
                  std::to_string(summary_.source_population));
  TitanUI::log_kv(log, "TPT Sink Population",
                  std::to_string(summary_.sink_population));
  TitanUI::log_kv(log, "TPT Rate A->B / lag",
                  std::to_string(summary_.rate_source_to_sink_per_lag));
  TitanUI::log_kv(log, "TPT Positive Flux Edges",
                  std::to_string(edges_.size()));
  TitanUI::log_kv(log, "TPT Dominant Paths", std::to_string(paths_.size()));
}

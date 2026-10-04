#include "titan_umap_knn.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"
#include "titan_thread_compat.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace TitanUMAPKNN {
namespace {

constexpr int kExactQueryBatch = 8;
constexpr int kCandidateBlock = 128;

struct NeighborCandidate {
  int index = -1;
  double dist_sq = std::numeric_limits<double>::infinity();
};

struct InternalBuildResult {
  std::vector<std::vector<Neighbor>> knn;
  double elapsed_ms = 0.0;
  std::uint64_t distance_evals = 0;
  int iterations_completed = 0;
};

using NeighborRow = std::vector<NeighborCandidate>;
using FlagRow = std::vector<std::uint8_t>;

struct ReverseNeighborLists {
  std::vector<std::vector<int>> all;
  std::vector<std::vector<int>> newly_added;
};

bool try_insert_neighbor(NeighborRow &best, FlagRow *is_new, int index,
                         double dist_sq);

double squared_distance_avx2(const double *lhs, const double *rhs, int dim) {
  return titan_simd_dispatch::squared_distance_rowmajor(lhs, rhs, dim);
}

inline void prefetch_row_read(const double *ptr) noexcept {
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
#else
  (void)ptr;
#endif
}

bool evaluate_candidates_blocked(const MatrixRMConstRef &input,
                                 const double *lhs, int dim,
                                 const std::vector<int> &candidates,
                                 NeighborRow &best, FlagRow *is_new,
                                 std::uint64_t &distance_evals) {
  bool changed = false;
  for (size_t block_begin = 0; block_begin < candidates.size();
       block_begin += kCandidateBlock) {
    const size_t block_end =
        std::min(candidates.size(), block_begin + kCandidateBlock);
    if (block_end < candidates.size()) {
      prefetch_row_read(input.data() +
                        static_cast<ptrdiff_t>(candidates[block_end]) * dim);
    }

    for (size_t pos = block_begin; pos < block_end; ++pos) {
      const int candidate = candidates[pos];
      const double *rhs =
          input.data() + static_cast<ptrdiff_t>(candidate) * dim;
      if (pos + 1 < block_end) {
        prefetch_row_read(input.data() +
                          static_cast<ptrdiff_t>(candidates[pos + 1]) * dim);
      }
      const double dist_sq = squared_distance_avx2(lhs, rhs, dim);
      ++distance_evals;
      changed |= try_insert_neighbor(best, is_new, candidate, dist_sq);
    }
  }
  return changed;
}

bool try_insert_neighbor(NeighborRow &best, FlagRow *is_new, int index,
                         double dist_sq) {
  if (index < 0 || best.empty())
    return false;

  for (size_t pos = 0; pos < best.size(); ++pos) {
    if (best[pos].index != index)
      continue;
    if (dist_sq >= best[pos].dist_sq)
      return false;

    best[pos].dist_sq = dist_sq;
    if (is_new != nullptr && pos < is_new->size()) {
      (*is_new)[pos] = 1u;
    }
    while (pos > 0 && best[pos].dist_sq < best[pos - 1].dist_sq) {
      std::swap(best[pos], best[pos - 1]);
      if (is_new != nullptr && pos < is_new->size()) {
        std::swap((*is_new)[pos], (*is_new)[pos - 1]);
      }
      --pos;
    }
    while (pos + 1 < best.size() && best[pos].dist_sq > best[pos + 1].dist_sq) {
      std::swap(best[pos], best[pos + 1]);
      if (is_new != nullptr && pos + 1 < is_new->size()) {
        std::swap((*is_new)[pos], (*is_new)[pos + 1]);
      }
      ++pos;
    }
    return true;
  }

  if (dist_sq >= best.back().dist_sq)
    return false;

  size_t pos = best.size() - 1;
  while (pos > 0 && dist_sq < best[pos - 1].dist_sq) {
    best[pos] = best[pos - 1];
    if (is_new != nullptr && pos < is_new->size()) {
      (*is_new)[pos] = (*is_new)[pos - 1];
    }
    --pos;
  }
  best[pos].index = index;
  best[pos].dist_sq = dist_sq;
  if (is_new != nullptr && pos < is_new->size()) {
    (*is_new)[pos] = 1u;
  }
  return true;
}

ReverseNeighborLists build_reverse_neighbor_lists(
    const std::vector<NeighborRow> &current,
    const std::vector<FlagRow> &current_is_new) {
  ReverseNeighborLists reverse;
  reverse.all.resize(current.size());
  reverse.newly_added.resize(current.size());

  std::vector<size_t> all_counts(current.size(), 0);
  std::vector<size_t> new_counts(current.size(), 0);
  for (size_t row = 0; row < current.size(); ++row) {
    const NeighborRow &neighbors = current[row];
    const FlagRow &flags = current_is_new[row];
    for (size_t pos = 0; pos < neighbors.size(); ++pos) {
      const int neighbor_index = neighbors[pos].index;
      if (neighbor_index < 0)
        continue;
      const size_t target = static_cast<size_t>(neighbor_index);
      ++all_counts[target];
      if (pos < flags.size() && flags[pos] != 0u) {
        ++new_counts[target];
      }
    }
  }

  for (size_t row = 0; row < current.size(); ++row) {
    reverse.all[row].reserve(all_counts[row]);
    reverse.newly_added[row].reserve(new_counts[row]);
  }

  for (size_t row = 0; row < current.size(); ++row) {
    const NeighborRow &neighbors = current[row];
    const FlagRow &flags = current_is_new[row];
    for (size_t pos = 0; pos < neighbors.size(); ++pos) {
      const int neighbor_index = neighbors[pos].index;
      if (neighbor_index < 0)
        continue;
      const size_t target = static_cast<size_t>(neighbor_index);
      reverse.all[target].push_back(static_cast<int>(row));
      if (pos < flags.size() && flags[pos] != 0u) {
        reverse.newly_added[target].push_back(static_cast<int>(row));
      }
    }
  }

  return reverse;
}

std::vector<std::vector<Neighbor>> finalize_neighbors(
    const std::vector<std::vector<NeighborCandidate>> &rows) {
  std::vector<std::vector<Neighbor>> knn(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    knn[i].resize(rows[i].size());
    for (size_t j = 0; j < rows[i].size(); ++j) {
      knn[i][j].index = rows[i][j].index;
      knn[i][j].distance =
          std::sqrt(std::max(rows[i][j].dist_sq, 0.0));
      knn[i][j].membership = 0.0;
    }
  }
  return knn;
}

Mode select_mode(Mode requested_mode, int n_samples, int auto_exact_threshold) {
  if (requested_mode != Mode::Auto)
    return requested_mode;
  return (n_samples <= auto_exact_threshold) ? Mode::Exact : Mode::Approx;
}

InternalBuildResult build_exact_knn_graph(const MatrixRMConstRef &input,
                                          int neighbor_count,
                                          const BuildConfig &cfg) {
  const auto start = std::chrono::steady_clock::now();
  const int n_samples = static_cast<int>(input.rows());
  const int dim = static_cast<int>(input.cols());
  const int max_neighbors = std::min(neighbor_count, n_samples - 1);

  std::vector<std::vector<NeighborCandidate>> knn(
      static_cast<size_t>(n_samples),
      std::vector<NeighborCandidate>(static_cast<size_t>(max_neighbors)));

  const size_t hw_threads = static_cast<size_t>(std::max(1, cfg.n_threads));
  const size_t n_workers =
      std::min<size_t>(hw_threads, static_cast<size_t>(n_samples));
  const int block_size =
      std::max(1, (n_samples + static_cast<int>(n_workers) - 1) /
                      static_cast<int>(n_workers));

  std::vector<std::uint64_t> worker_counts(n_workers, 0);
  const auto run_worker = [&](size_t worker_id) {
    const int begin = static_cast<int>(worker_id) * block_size;
    const int end = std::min(n_samples, begin + block_size);
    std::vector<NeighborRow> batch_best(
        static_cast<size_t>(kExactQueryBatch),
        NeighborRow(static_cast<size_t>(max_neighbors)));
    std::array<const double *, kExactQueryBatch> lhs_ptrs{};
    std::array<int, kExactQueryBatch> row_ids{};
    std::uint64_t local_count = 0;

    for (int batch_begin = begin; batch_begin < end;
         batch_begin += kExactQueryBatch) {
      const int batch_size = std::min(kExactQueryBatch, end - batch_begin);
      for (int b = 0; b < batch_size; ++b) {
        const int row = batch_begin + b;
        row_ids[b] = row;
        lhs_ptrs[b] = input.data() + static_cast<ptrdiff_t>(row) * dim;
        std::fill(batch_best[static_cast<size_t>(b)].begin(),
                  batch_best[static_cast<size_t>(b)].end(), NeighborCandidate{});
      }

      for (int candidate_block = 0; candidate_block < n_samples;
           candidate_block += kCandidateBlock) {
        const int candidate_end =
            std::min(n_samples, candidate_block + kCandidateBlock);
        if (candidate_end < n_samples) {
          prefetch_row_read(input.data() +
                            static_cast<ptrdiff_t>(candidate_end) * dim);
        }
        for (int j = candidate_block; j < candidate_end; ++j) {
          const double *rhs = input.data() + static_cast<ptrdiff_t>(j) * dim;
          if (j + 1 < candidate_end) {
            prefetch_row_read(input.data() +
                              static_cast<ptrdiff_t>(j + 1) * dim);
          }
          for (int b = 0; b < batch_size; ++b) {
            if (row_ids[b] == j)
              continue;
            const double dist_sq = squared_distance_avx2(lhs_ptrs[b], rhs, dim);
            ++local_count;
            try_insert_neighbor(batch_best[static_cast<size_t>(b)], nullptr, j,
                                dist_sq);
          }
        }
      }

      for (int b = 0; b < batch_size; ++b) {
        knn[static_cast<size_t>(row_ids[b])] =
            batch_best[static_cast<size_t>(b)];
      }
    }

    worker_counts[worker_id] = local_count;
  };

  if (cfg.executor != nullptr) {
    if (n_workers > 1) {
      cfg.executor->parallel_do(n_workers, run_worker);
    } else {
      run_worker(0);
    }
  } else {
    std::vector<titan::jthread> workers;
    workers.reserve(n_workers);
    for (size_t worker_id = 0; worker_id < n_workers; ++worker_id) {
      const int begin = static_cast<int>(worker_id) * block_size;
      const int end = std::min(n_samples, begin + block_size);
      if (begin >= end)
        break;
      workers.emplace_back([&, worker_id]() { run_worker(worker_id); });
    }
  }

  std::uint64_t total_count = 0;
  for (std::uint64_t count : worker_counts) {
    total_count += count;
  }

  InternalBuildResult result;
  result.knn = finalize_neighbors(knn);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                start)
          .count();
  result.distance_evals = total_count;
  return result;
}

std::vector<NeighborCandidate> random_seed_neighbors(
    const MatrixRMConstRef &input, int row, int neighbor_count, unsigned int seed,
    int seed_neighbors, std::uint64_t &distance_evals) {
  const int n_samples = static_cast<int>(input.rows());
  const int dim = static_cast<int>(input.cols());
  std::vector<NeighborCandidate> best(static_cast<size_t>(neighbor_count));
  if (n_samples <= 1)
    return best;

  const int init_samples =
      std::min(n_samples - 1, std::max(neighbor_count, seed_neighbors));
  std::mt19937 rng(seed ^ (0x9E3779B9u * static_cast<unsigned int>(row + 1)));
  std::uniform_int_distribution<int> node_dist(0, n_samples - 1);
  std::vector<int> sampled;
  sampled.reserve(static_cast<size_t>(init_samples));
  while (static_cast<int>(sampled.size()) < init_samples) {
    const int candidate = node_dist(rng);
    if (candidate == row)
      continue;
    if (std::find(sampled.begin(), sampled.end(), candidate) != sampled.end())
      continue;
    sampled.push_back(candidate);
  }

  const double *lhs = input.data() + static_cast<ptrdiff_t>(row) * dim;
  if (sampled.size() > 1) {
    std::sort(sampled.begin(), sampled.end());
  }
  evaluate_candidates_blocked(input, lhs, dim, sampled, best, nullptr,
                              distance_evals);

  return best;
}

InternalBuildResult build_approx_knn_graph(const MatrixRMConstRef &input,
                                           int neighbor_count,
                                           const BuildConfig &cfg) {
  const auto start = std::chrono::steady_clock::now();
  const int n_samples = static_cast<int>(input.rows());
  const int dim = static_cast<int>(input.cols());
  const int max_neighbors = std::min(neighbor_count, n_samples - 1);

  std::vector<std::vector<NeighborCandidate>> current(
      static_cast<size_t>(n_samples),
      std::vector<NeighborCandidate>(static_cast<size_t>(max_neighbors)));
  std::vector<FlagRow> current_is_new(
      static_cast<size_t>(n_samples),
      FlagRow(static_cast<size_t>(max_neighbors), 0u));

  const size_t hw_threads = static_cast<size_t>(std::max(1, cfg.n_threads));
  const size_t n_workers =
      std::min<size_t>(hw_threads, static_cast<size_t>(n_samples));
  const int block_size =
      std::max(1, (n_samples + static_cast<int>(n_workers) - 1) /
                      static_cast<int>(n_workers));

  std::vector<std::uint64_t> worker_seed_counts(n_workers, 0);
  const auto seed_worker = [&](size_t worker_id) {
    const int begin = static_cast<int>(worker_id) * block_size;
    const int end = std::min(n_samples, begin + block_size);
    std::uint64_t local_count = 0;
    for (int i = begin; i < end; ++i) {
      NeighborRow seeded =
          random_seed_neighbors(input, i, max_neighbors, cfg.seed,
                                cfg.approx_seed_neighbors, local_count);
      current[static_cast<size_t>(i)] = seeded;
      FlagRow &flags = current_is_new[static_cast<size_t>(i)];
      for (size_t pos = 0; pos < seeded.size() && pos < flags.size(); ++pos) {
        flags[pos] = (seeded[pos].index >= 0) ? 1u : 0u;
      }
    }
    worker_seed_counts[worker_id] = local_count;
  };
  if (cfg.executor != nullptr) {
    if (n_workers > 1) {
      cfg.executor->parallel_do(n_workers, seed_worker);
    } else {
      seed_worker(0);
    }
  } else {
    std::vector<titan::jthread> workers;
    workers.reserve(n_workers);
    for (size_t worker_id = 0; worker_id < n_workers; ++worker_id) {
      const int begin = static_cast<int>(worker_id) * block_size;
      const int end = std::min(n_samples, begin + block_size);
      if (begin >= end)
        break;
      workers.emplace_back([&, worker_id]() { seed_worker(worker_id); });
    }
  }

  std::uint64_t total_distance_evals = 0;
  for (std::uint64_t count : worker_seed_counts) {
    total_distance_evals += count;
  }

  int iterations_completed = 0;
  for (int iter = 0; iter < cfg.approx_iterations; ++iter) {
    const ReverseNeighborLists reverse =
        build_reverse_neighbor_lists(current, current_is_new);
    std::vector<NeighborRow> pending_rows(static_cast<size_t>(n_samples));
    std::vector<FlagRow> next_is_new(
        static_cast<size_t>(n_samples),
        FlagRow(static_cast<size_t>(max_neighbors), 0u));
    std::vector<std::uint8_t> row_changed(static_cast<size_t>(n_samples), 0u);
    std::vector<std::uint64_t> worker_counts(n_workers, 0);
    std::vector<size_t> worker_updates(n_workers, 0);
    const auto refine_worker = [&](size_t worker_id) {
      const int begin = static_cast<int>(worker_id) * block_size;
      const int end = std::min(n_samples, begin + block_size);
      std::uint64_t local_count = 0;
      size_t local_updates = 0;
      std::vector<int> candidates;
      std::vector<int> new_sources;
      std::vector<std::uint32_t> visited_epoch(static_cast<size_t>(n_samples), 0u);
      std::vector<std::uint32_t> source_epoch(static_cast<size_t>(n_samples), 0u);
      std::uint32_t visit_token = 0u;
      const int reserve_guess = 4 * max_neighbors +
                                cfg.approx_extra_random_candidates;
      candidates.reserve(static_cast<size_t>(std::max(reserve_guess, 1)));
      new_sources.reserve(static_cast<size_t>(std::max(2 * max_neighbors, 1)));
      std::mt19937 rng(cfg.seed ^ (0x85EBCA6Bu *
                                  static_cast<unsigned int>(worker_id + 1)) ^
                       static_cast<unsigned int>(iter + 1));
      std::uniform_int_distribution<int> node_dist(0, n_samples - 1);

      for (int i = begin; i < end; ++i) {
            ++visit_token;
            if (visit_token == 0u) {
              std::fill(visited_epoch.begin(), visited_epoch.end(), 0u);
              std::fill(source_epoch.begin(), source_epoch.end(), 0u);
              visit_token = 1u;
            }
            candidates.clear();
            new_sources.clear();
            NeighborRow best = current[static_cast<size_t>(i)];
            FlagRow best_is_new(static_cast<size_t>(max_neighbors), 0u);

            const auto push_candidate = [&](int candidate) {
              if (candidate < 0 || candidate == i)
                return;
              const size_t idx = static_cast<size_t>(candidate);
              if (visited_epoch[idx] == visit_token)
                return;
              visited_epoch[idx] = visit_token;
              candidates.push_back(candidate);
            };
            const auto push_new_source = [&](int candidate) {
              if (candidate < 0 || candidate == i)
                return;
              const size_t idx = static_cast<size_t>(candidate);
              if (source_epoch[idx] == visit_token)
                return;
              source_epoch[idx] = visit_token;
              new_sources.push_back(candidate);
              push_candidate(candidate);
            };
            const auto queue_forward_neighbors = [&](int row_index,
                                                     bool track_new_sources) {
              const NeighborRow &row_neighbors = current[static_cast<size_t>(row_index)];
              const FlagRow &row_flags = current_is_new[static_cast<size_t>(row_index)];
              for (size_t pos = 0; pos < row_neighbors.size(); ++pos) {
                const int candidate = row_neighbors[pos].index;
                if (candidate < 0)
                  continue;
                push_candidate(candidate);
                if (track_new_sources && pos < row_flags.size() &&
                    row_flags[pos] != 0u) {
                  push_new_source(candidate);
                }
              }
            };
            const auto queue_reverse_neighbors =
                [&](const std::vector<int> &neighbors, bool track_new_sources) {
                  for (int candidate : neighbors) {
                    push_candidate(candidate);
                    if (track_new_sources) {
                      push_new_source(candidate);
                    }
                  }
                };

            queue_forward_neighbors(i, true);
            queue_reverse_neighbors(reverse.all[static_cast<size_t>(i)], false);
            queue_reverse_neighbors(reverse.newly_added[static_cast<size_t>(i)], true);

            for (int source : new_sources) {
              queue_forward_neighbors(source, false);
              queue_reverse_neighbors(reverse.all[static_cast<size_t>(source)],
                                      false);
            }

            for (int extra = 0; extra < cfg.approx_extra_random_candidates;
                 ++extra) {
              push_candidate(node_dist(rng));
            }

            bool changed = false;
            const double *lhs = input.data() + static_cast<ptrdiff_t>(i) * dim;
            if (candidates.size() > 1) {
              std::sort(candidates.begin(), candidates.end());
            }
            changed |= evaluate_candidates_blocked(
                input, lhs, dim, candidates, best, &best_is_new, local_count);

            if (changed) {
              pending_rows[static_cast<size_t>(i)] = std::move(best);
              next_is_new[static_cast<size_t>(i)] = std::move(best_is_new);
              row_changed[static_cast<size_t>(i)] = 1u;
              ++local_updates;
            }
      }

      worker_counts[worker_id] = local_count;
      worker_updates[worker_id] = local_updates;
    };
    if (cfg.executor != nullptr) {
      if (n_workers > 1) {
        cfg.executor->parallel_do(n_workers, refine_worker);
      } else {
        refine_worker(0);
      }
    } else {
    std::vector<titan::jthread> workers;
      workers.reserve(n_workers);
      for (size_t worker_id = 0; worker_id < n_workers; ++worker_id) {
        const int begin = static_cast<int>(worker_id) * block_size;
        const int end = std::min(n_samples, begin + block_size);
        if (begin >= end)
          break;
        workers.emplace_back([&, worker_id]() { refine_worker(worker_id); });
      }
    }

    size_t total_updates = 0;
    std::uint64_t iter_distance_evals = 0;
    for (size_t worker_id = 0; worker_id < n_workers; ++worker_id) {
      total_updates += worker_updates[worker_id];
      iter_distance_evals += worker_counts[worker_id];
    }
    total_distance_evals += iter_distance_evals;
    for (size_t row = 0; row < current.size(); ++row) {
      if (row_changed[row] != 0u) {
        current[row].swap(pending_rows[row]);
      }
      current_is_new[row].swap(next_is_new[row]);
    }
    iterations_completed = iter + 1;

    if (total_updates == 0)
      break;
  }

  InternalBuildResult result;
  result.knn = finalize_neighbors(current);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                start)
          .count();
  result.distance_evals = total_distance_evals;
  result.iterations_completed = iterations_completed;
  return result;
}

double compute_recall(const std::vector<std::vector<Neighbor>> &approx_knn,
                      const std::vector<std::vector<Neighbor>> &exact_knn) {
  size_t total = 0;
  size_t matched = 0;

  for (size_t row = 0; row < exact_knn.size(); ++row) {
    std::vector<int> approx_indices;
    std::vector<int> exact_indices;
    approx_indices.reserve(approx_knn[row].size());
    exact_indices.reserve(exact_knn[row].size());

    for (const Neighbor &neighbor : approx_knn[row]) {
      if (neighbor.index >= 0)
        approx_indices.push_back(neighbor.index);
    }
    for (const Neighbor &neighbor : exact_knn[row]) {
      if (neighbor.index >= 0)
        exact_indices.push_back(neighbor.index);
    }

    std::sort(approx_indices.begin(), approx_indices.end());
    std::sort(exact_indices.begin(), exact_indices.end());

    total += exact_indices.size();
    for (int neighbor : exact_indices) {
      matched +=
          std::binary_search(approx_indices.begin(), approx_indices.end(), neighbor)
              ? 1u
              : 0u;
    }
  }

  if (total == 0)
    return 1.0;
  return static_cast<double>(matched) / static_cast<double>(total);
}

} // namespace

BuildResult build_knn_graph(const MatrixRMConstRef &input, int neighbor_count,
                            const BuildConfig &cfg) {
  BuildResult result;
  result.stats.requested_mode = cfg.mode;
  result.stats.selected_mode =
      select_mode(cfg.mode, static_cast<int>(input.rows()), cfg.auto_exact_threshold);

  if (neighbor_count <= 0 || input.rows() <= 1) {
    return result;
  }

  if (cfg.benchmark) {
    const InternalBuildResult exact =
        build_exact_knn_graph(input, neighbor_count, cfg);
    const InternalBuildResult approx =
        build_approx_knn_graph(input, neighbor_count, cfg);

    result.stats.exact_elapsed_ms = exact.elapsed_ms;
    result.stats.approx_elapsed_ms = approx.elapsed_ms;
    result.stats.exact_distance_evals = exact.distance_evals;
    result.stats.approx_distance_evals = approx.distance_evals;
    result.stats.approx_iterations_completed = approx.iterations_completed;
    result.stats.benchmark_recall = compute_recall(approx.knn, exact.knn);
    result.stats.benchmark_has_baseline = true;

    if (result.stats.selected_mode == Mode::Approx) {
      result.knn = approx.knn;
      result.stats.selected_elapsed_ms = approx.elapsed_ms;
      result.stats.selected_distance_evals = approx.distance_evals;
    } else {
      result.knn = exact.knn;
      result.stats.selected_elapsed_ms = exact.elapsed_ms;
      result.stats.selected_distance_evals = exact.distance_evals;
    }
    return result;
  }

  if (result.stats.selected_mode == Mode::Approx) {
    const InternalBuildResult approx = build_approx_knn_graph(input, neighbor_count, cfg);
    result.knn = approx.knn;
    result.stats.selected_elapsed_ms = approx.elapsed_ms;
    result.stats.selected_distance_evals = approx.distance_evals;
    result.stats.approx_elapsed_ms = approx.elapsed_ms;
    result.stats.approx_distance_evals = approx.distance_evals;
    result.stats.approx_iterations_completed = approx.iterations_completed;
    return result;
  }

  const InternalBuildResult exact =
      build_exact_knn_graph(input, neighbor_count, cfg);
  result.knn = exact.knn;
  result.stats.selected_elapsed_ms = exact.elapsed_ms;
  result.stats.selected_distance_evals = exact.distance_evals;
  result.stats.exact_elapsed_ms = exact.elapsed_ms;
  result.stats.exact_distance_evals = exact.distance_evals;
  return result;
}

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::Exact:
    return "exact";
  case Mode::Approx:
    return "approx";
  case Mode::Auto:
    return "auto";
  }
  return "exact";
}

} // namespace TitanUMAPKNN

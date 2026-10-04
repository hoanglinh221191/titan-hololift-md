#ifndef TITAN_EXECUTOR_H
#define TITAN_EXECUTOR_H

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include "titan_thread_compat.h"
#include <vector>

class TitanExecutor {
public:
  explicit TitanExecutor(std::size_t nt);
  ~TitanExecutor();

  TitanExecutor(const TitanExecutor &) = delete;
  TitanExecutor &operator=(const TitanExecutor &) = delete;

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  void parallel_for(
      std::size_t total_items, std::size_t requested_workers,
      std::size_t min_grain,
      const std::function<void(std::size_t, std::size_t, std::size_t)> &fn);

  void parallel_do(std::size_t requested_workers,
                   const std::function<void(std::size_t)> &fn);

private:
  enum class BatchMode : unsigned char { None, ParallelFor, ParallelDo };

  void worker_loop(titan::stop_token stop_token, std::size_t worker_id);
  void shutdown();

  std::size_t compute_parallel_for_workers(std::size_t total_items,
                                           std::size_t requested_workers,
                                           std::size_t min_grain) const;
  std::size_t compute_parallel_do_workers(std::size_t requested_workers) const;

  std::size_t capacity_{1};
  std::vector<titan::jthread> workers_;

  mutable std::mutex mutex_;
  std::condition_variable batch_cv_;
  std::condition_variable done_cv_;

  BatchMode batch_mode_{BatchMode::None};
  std::size_t batch_generation_{0};
  std::size_t batch_worker_count_{0};
  std::size_t pending_workers_{0};
  std::size_t total_items_{0};
  bool stop_{false};
  std::exception_ptr first_exception_;
  std::function<void(std::size_t)> do_fn_;
  std::function<void(std::size_t, std::size_t, std::size_t)> for_fn_;

  static thread_local TitanExecutor *active_executor_;
};

#endif // TITAN_EXECUTOR_H

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

  // Several threads may submit batches; they run one after another. A batch
  // submitted from inside a worker of this executor (nested) runs inline on
  // that worker, serially.
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
  // With mutex_ held: waits until the executor is free and publishes a batch.
  void begin_batch(
      std::unique_lock<std::mutex> &lock, BatchMode mode,
      std::size_t active_workers, std::size_t total_items,
      const std::function<void(std::size_t, std::size_t, std::size_t)> *for_fn,
      const std::function<void(std::size_t)> *do_fn);
  // With mutex_ held: waits for the workers, frees the executor, wakes the
  // waiting submitters and rethrows the first exception of the batch.
  void end_batch(std::unique_lock<std::mutex> &lock);

  std::size_t compute_parallel_for_workers(std::size_t total_items,
                                           std::size_t requested_workers,
                                           std::size_t min_grain) const;
  std::size_t compute_parallel_do_workers(std::size_t requested_workers) const;

  std::size_t capacity_{1};
  std::vector<titan::jthread> workers_;

  mutable std::mutex mutex_;
  // CORE-08: one condition variable per kind of waiter. A single shared one
  // with notify_one lost wake-ups when two threads submitted batches.
  std::condition_variable batch_cv_; // workers: a new batch or stop
  std::condition_variable done_cv_;  // submitter of the running batch
  std::condition_variable idle_cv_;  // other submitters: executor free again

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

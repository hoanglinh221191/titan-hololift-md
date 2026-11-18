#include "titan_executor.h"

#include <algorithm>
#include <utility>

thread_local TitanExecutor *TitanExecutor::active_executor_ = nullptr;

TitanExecutor::TitanExecutor(std::size_t nt) : capacity_(std::max<std::size_t>(1, nt)) {
  workers_.reserve(capacity_);
  for (std::size_t worker_id = 0; worker_id < capacity_; ++worker_id) {
    workers_.emplace_back(
        [this, worker_id](titan::stop_token stop_token) {
          worker_loop(stop_token, worker_id);
        });
  }
}

TitanExecutor::~TitanExecutor() { shutdown(); }

void TitanExecutor::parallel_for(
    std::size_t total_items, std::size_t requested_workers,
    std::size_t min_grain,
    const std::function<void(std::size_t, std::size_t, std::size_t)> &fn) {
  if (total_items == 0)
    return;

  if (!fn) {
    throw std::invalid_argument("TitanExecutor::parallel_for received empty function");
  }

  const std::size_t active_workers =
      compute_parallel_for_workers(total_items, requested_workers, min_grain);
  // CORE-08: a nested batch (submitted from a worker of this executor) runs
  // inline on that worker; waiting for the pool from inside it would
  // deadlock. The old code asserted here, and TITAN builds without NDEBUG,
  // so a nested call aborted the run.
  if (active_workers <= 1 || active_executor_ == this) {
    fn(0, 0, total_items);
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  begin_batch(lock, BatchMode::ParallelFor, active_workers, total_items, &fn,
              nullptr);
  end_batch(lock);
}

void TitanExecutor::parallel_do(
    std::size_t requested_workers, const std::function<void(std::size_t)> &fn) {
  if (!fn) {
    throw std::invalid_argument("TitanExecutor::parallel_do received empty function");
  }

  const std::size_t active_workers =
      compute_parallel_do_workers(requested_workers);
  // nested batch: inline on the calling worker (see parallel_for)
  if (active_workers <= 1 || active_executor_ == this) {
    fn(0);
    return;
  }

  std::unique_lock<std::mutex> lock(mutex_);
  begin_batch(lock, BatchMode::ParallelDo, active_workers, 0, nullptr, &fn);
  end_batch(lock);
}

void TitanExecutor::begin_batch(
    std::unique_lock<std::mutex> &lock, BatchMode mode,
    std::size_t active_workers, std::size_t total_items,
    const std::function<void(std::size_t, std::size_t, std::size_t)> *for_fn,
    const std::function<void(std::size_t)> *do_fn) {
  // other submitters wait here, on their own condition variable
  idle_cv_.wait(lock, [this]() { return batch_mode_ == BatchMode::None; });

  batch_mode_ = mode;
  batch_worker_count_ = active_workers;
  pending_workers_ = active_workers;
  total_items_ = total_items;
  first_exception_ = nullptr;
  for_fn_ = for_fn ? *for_fn : std::function<void(std::size_t, std::size_t,
                                                  std::size_t)>{};
  do_fn_ = do_fn ? *do_fn : std::function<void(std::size_t)>{};
  ++batch_generation_;
  batch_cv_.notify_all();
}

void TitanExecutor::end_batch(std::unique_lock<std::mutex> &lock) {
  // only the submitter of the running batch waits on done_cv_
  done_cv_.wait(lock, [this]() { return pending_workers_ == 0; });

  for_fn_ = {};
  do_fn_ = {};
  batch_mode_ = BatchMode::None;
  total_items_ = 0;
  const std::exception_ptr exception = std::exchange(first_exception_, nullptr);
  lock.unlock();
  // the executor is free: every waiting submitter re-checks; one takes it
  idle_cv_.notify_all();
  if (exception)
    std::rethrow_exception(exception);
}

void TitanExecutor::worker_loop(titan::stop_token stop_token,
                                std::size_t worker_id) {
  std::size_t local_generation = 0;

  for (;;) {
    BatchMode mode = BatchMode::None;
    std::size_t active_workers = 0;
    std::size_t total_items = 0;
    std::function<void(std::size_t)> do_fn;
    std::function<void(std::size_t, std::size_t, std::size_t)> for_fn;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      batch_cv_.wait(lock, [&]() {
        return stop_ || stop_token.stop_requested() ||
               batch_generation_ != local_generation;
      });

      if (stop_ || stop_token.stop_requested())
        break;

      local_generation = batch_generation_;
      mode = batch_mode_;
      active_workers = batch_worker_count_;
      total_items = total_items_;
      do_fn = do_fn_;
      for_fn = for_fn_;
    }

    if (worker_id >= active_workers || mode == BatchMode::None)
      continue;

    try {
      active_executor_ = this;
      if (mode == BatchMode::ParallelFor) {
        const std::size_t begin = total_items * worker_id / active_workers;
        const std::size_t end = total_items * (worker_id + 1) / active_workers;
        if (begin < end)
          for_fn(worker_id, begin, end);
      } else if (mode == BatchMode::ParallelDo) {
        do_fn(worker_id);
      }
      active_executor_ = nullptr;
    } catch (...) {
      active_executor_ = nullptr;
      std::lock_guard<std::mutex> lock(mutex_);
      if (!first_exception_)
        first_exception_ = std::current_exception();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_workers_ > 0) {
      --pending_workers_;
      if (pending_workers_ == 0)
        done_cv_.notify_one();
    }
  }
}

void TitanExecutor::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_)
      return;
    stop_ = true;
    ++batch_generation_;
  }

  for (auto &worker : workers_) {
    worker.request_stop();
  }
  batch_cv_.notify_all();
  workers_.clear();
}

std::size_t TitanExecutor::compute_parallel_for_workers(
    std::size_t total_items, std::size_t requested_workers,
    std::size_t min_grain) const {
  const std::size_t resolved_grain = std::max<std::size_t>(1, min_grain);
  const std::size_t grain_limited =
      (total_items + resolved_grain - 1) / resolved_grain;
  const std::size_t requested = std::max<std::size_t>(1, requested_workers);
  return std::max<std::size_t>(
      1, std::min(capacity_, std::min(requested, grain_limited)));
}

std::size_t TitanExecutor::compute_parallel_do_workers(
    std::size_t requested_workers) const {
  return std::max<std::size_t>(
      1, std::min(capacity_, std::max<std::size_t>(1, requested_workers)));
}

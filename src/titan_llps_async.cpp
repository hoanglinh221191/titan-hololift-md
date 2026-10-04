// Split from titan_llps.cpp: async LLPS frame pipeline.
#include "titan_llps.h"
#include "MOL.h"
#include "MOL_info.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>

void TitanLLPS::worker_loop(titan::stop_token stoken) {
  while (true) {
    FrameWorkItem item;
    {
      std::unique_lock<std::mutex> lock(queue_mtx_);
      queue_cv_.wait(lock, [&] {
        return !work_queue_.empty() || stoken.stop_requested() || worker_stop_;
      });
      if (work_queue_.empty()) {
        if (stoken.stop_requested() || worker_stop_)
          return;
        continue;
      }
      item = std::move(work_queue_.front());
      work_queue_.pop();
      worker_processing_ = true;
    }
    queue_full_cv_.notify_one();

    try {
    cell_ = std::move(item.cell);
    process_frame_impl(item.pos_nm, item.rna_nm ? item.rna_nm->data() : nullptr,
                       item.n_rna, item.bx, item.by, item.bz, item.time_ns,
                       item.frame_idx);
    } catch (...) {
      { std::lock_guard<std::mutex> lock(queue_mtx_);
        worker_error_=std::current_exception(); worker_stop_=true;
        worker_processing_=false;
        while(!work_queue_.empty()) work_queue_.pop();
      }
      queue_cv_.notify_all(); queue_full_cv_.notify_all(); return;
    }

    {
      std::lock_guard<std::mutex> lock(queue_mtx_);
      item.pos_nm.clear();
      if (item.rna_nm)
        item.rna_nm->clear();
      free_work_items_.push_back(std::move(item));
      worker_processing_ = false;
    }
    queue_cv_.notify_all();
  }
}

void TitanLLPS::start_async() {
  if (async_mode_) return;
  if (worker_error_) std::rethrow_exception(worker_error_);
  async_mode_ = true;
  worker_stop_ = false;
  worker_processing_ = false;
  worker_thread_ =
      titan::jthread([this](titan::stop_token stoken) { worker_loop(stoken); });
}

void TitanLLPS::process_frame_async(
    const MOL &mol_frame, const std::vector<mol_info> & /*info*/, float bx,
    float by, float bz, float time_ns, int frame_idx, const MOL *rna_frame,
    const std::vector<mol_info> *rna_info) {
  auto frame_cell=titan_llps_geometry::Cell::from_box(mol_frame.Box,bx,by,bz);
  if (!async_mode_) start_async();
  const auto pack_begin = std::chrono::steady_clock::now();
  const auto &ax = mol_frame.get_pos_x();
  const auto &ay = mol_frame.get_pos_y();
  const auto &az = mol_frame.get_pos_z();
  const int n_atoms = static_cast<int>(ax.size());

  constexpr float kAngToNm = 0.1f;
  FrameWorkItem item;
  {
    std::lock_guard<std::mutex> lock(queue_mtx_);
    if (!free_work_items_.empty()) {
      item = std::move(free_work_items_.back());
      free_work_items_.pop_back();
    }
  }
  item.cell=std::move(frame_cell);
  item.pos_nm.resize(static_cast<std::size_t>(n_atoms) * 3);
  for (int i = 0; i < n_atoms; ++i) {
    item.pos_nm[3 * i + 0] = static_cast<float>(ax[i]) * kAngToNm;
    item.pos_nm[3 * i + 1] = static_cast<float>(ay[i]) * kAngToNm;
    item.pos_nm[3 * i + 2] = static_cast<float>(az[i]) * kAngToNm;
  }
  item.bx = bx;
  item.by = by;
  item.bz = bz;
  item.time_ns = time_ns;
  item.frame_idx = frame_idx;

  if (rna_frame) {
    if (rna_info && rna_atom_chain_.empty()) {
      flush_async();
      set_rna_metadata(*rna_info);
      start_async();
    }
    const auto &rx = rna_frame->get_pos_x();
    const auto &ry = rna_frame->get_pos_y();
    const auto &rz = rna_frame->get_pos_z();
    item.n_rna = static_cast<int>(rx.size());
    if (!item.rna_nm)
      item.rna_nm.emplace();
    item.rna_nm->resize(static_cast<std::size_t>(item.n_rna) * 3);
    for (int i = 0; i < item.n_rna; ++i) {
      (*item.rna_nm)[3 * i + 0] = static_cast<float>(rx[i]) * kAngToNm;
      (*item.rna_nm)[3 * i + 1] = static_cast<float>(ry[i]) * kAngToNm;
      (*item.rna_nm)[3 * i + 2] = static_cast<float>(rz[i]) * kAngToNm;
    }
  } else {
    item.n_rna = 0;
    if (item.rna_nm)
      item.rna_nm->clear();
  }
  perf_.pack_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - pack_begin)
                            .count();

  {
    std::unique_lock<std::mutex> lock(queue_mtx_);
    queue_full_cv_.wait(lock, [&] {
      return static_cast<int>(work_queue_.size()) < MAX_QUEUE_SIZE_ ||
             worker_stop_;
    });
    if (worker_error_) std::rethrow_exception(worker_error_);
    if (worker_stop_) throw std::runtime_error("LLPS worker stopped before accepting frame");
    work_queue_.push(std::move(item));
    perf_.max_queue_depth = std::max<std::size_t>(perf_.max_queue_depth,
                                                  work_queue_.size());
  }
  queue_cv_.notify_one();
}

void TitanLLPS::flush_async() {
  if (!async_mode_) {
    if(worker_error_) std::rethrow_exception(worker_error_);
    return;
  }
  {
    std::unique_lock<std::mutex> lock(queue_mtx_);
    queue_cv_.wait(lock, [&] {
      return work_queue_.empty() && !worker_processing_;
    });
    worker_stop_ = true;
  }
  queue_cv_.notify_all();
  if (worker_thread_.joinable())
    worker_thread_.join();
  async_mode_ = false;
  if(worker_error_) std::rethrow_exception(worker_error_);
}

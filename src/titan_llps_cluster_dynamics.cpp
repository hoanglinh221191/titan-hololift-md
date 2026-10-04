// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "titan_exact_periodic_box.h"
#include "box_type.h"
#include "titan_executor.h"
#include "titan_llps_stats.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace titan_llps_stats;

namespace {

template <typename T>
void resize_row_major(std::vector<T> &buffer, std::size_t rows,
                      std::size_t old_cols, std::size_t new_cols,
                      const T &fill_value) {
  std::vector<T> resized(rows * new_cols, fill_value);
  const std::size_t copy_cols = std::min(old_cols, new_cols);
  for (std::size_t row = 0; row < rows; ++row) {
    const auto old_begin =
        buffer.begin() + static_cast<std::ptrdiff_t>(row * old_cols);
    const auto old_end =
        old_begin + static_cast<std::ptrdiff_t>(copy_cols);
    const auto new_begin =
        resized.begin() + static_cast<std::ptrdiff_t>(row * new_cols);
    std::copy(old_begin, old_end, new_begin);
  }
  buffer.swap(resized);
}

} // namespace

void TitanLLPSCluster::resize_dynamics_ring_storage(std::size_t chain_count) {
  if (!dynamics_enabled_ || dynamics_ring_cap_ <= 0 ||
      dynamics_ring_chain_capacity_ >= chain_count)
    return;
  const std::size_t rows = static_cast<std::size_t>(dynamics_ring_cap_);
  const std::size_t old_cols = dynamics_ring_chain_capacity_;
  resize_row_major(frame_present_ring_, rows, old_cols, chain_count,
                   static_cast<uint8_t>(0));
  resize_row_major(frame_inside_ring_, rows, old_cols, chain_count,
                   static_cast<uint8_t>(0));
  resize_row_major(frame_episode_ring_, rows, old_cols, chain_count,
                   static_cast<uint64_t>(0));
  resize_row_major(frame_unwrapped_x_ring_, rows, old_cols, chain_count, 0.0);
  resize_row_major(frame_unwrapped_y_ring_, rows, old_cols, chain_count, 0.0);
  resize_row_major(frame_unwrapped_z_ring_, rows, old_cols, chain_count, 0.0);
  dynamics_ring_chain_capacity_ = chain_count;
}

void TitanLLPSCluster::resize_membership_lag_storage(std::size_t chain_count) {
  if (!dynamics_enabled_ || dynamics_max_lag_frames_ < 0 ||
      membership_lag_chain_capacity_ >= chain_count)
    return;
  const std::size_t rows =
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1);
  const std::size_t old_cols = membership_lag_chain_capacity_;
  resize_row_major(membership_joint_inside_, rows, old_cols, chain_count,
                   static_cast<uint64_t>(0));
  resize_row_major(membership_pair_counts_, rows, old_cols, chain_count,
                   static_cast<uint64_t>(0));
  membership_lag_chain_capacity_ = chain_count;
}

void TitanLLPSCluster::ensure_chain_capacity(std::size_t chain_index) {
  const std::size_t target = chain_index + 1;
  if (chain_states_.size() >= target)
    return;

  chain_states_.resize(target);
  chain_index_species_.resize(target, -1);
  chain_index_raw_id_.resize(target, -1);
  membership_inside_counts_.resize(target, 0);
  membership_frame_counts_.resize(target, 0);
  resize_membership_lag_storage(target);
  resize_dynamics_ring_storage(target);
}

void TitanLLPSCluster::update_lineage(
    double time_ns, const std::vector<std::vector<int>> &cluster_pro,
    const std::vector<std::vector<int>> &cluster_rna) {
  std::vector<LineageClusterSnapshot> current_clusters;
  current_clusters.reserve(cluster_pro.size());
  for (std::size_t cid = 1; cid < cluster_pro.size(); ++cid) {
    if (static_cast<int>(cluster_pro[cid].size()) < min_lineage_cluster_size_)
      continue;
    LineageClusterSnapshot row;
    row.frame_cluster_id = static_cast<int>(cid);
    row.protein_chains = static_cast<int>(cluster_pro[cid].size());
    row.rna_chains = static_cast<int>(cluster_rna[cid].size());
    row.chain_ids.reserve(cluster_pro[cid].size() + cluster_rna[cid].size());
    for (int chain : cluster_pro[cid])
      row.chain_ids.push_back(chain);
    for (int chain : cluster_rna[cid])
      row.chain_ids.push_back(-(chain + 1));
    std::sort(row.chain_ids.begin(), row.chain_ids.end());
    current_clusters.push_back(std::move(row));
  }

  std::vector<std::vector<int>> prev_to_curr(prev_lineage_clusters_.size());
  for (const auto &curr : current_clusters) {
    LineageClusterRow row;
    row.time_ns = time_ns;
    row.frame_cluster_id = curr.frame_cluster_id;
    row.protein_chains = curr.protein_chains;
    row.rna_chains = curr.rna_chains;

    for (std::size_t p = 0; p < prev_lineage_clusters_.size(); ++p) {
      const auto &prev = prev_lineage_clusters_[p];
      std::size_t i = 0;
      std::size_t j = 0;
      std::size_t inter = 0;
      std::size_t uni = 0;
      while (i < prev.chain_ids.size() || j < curr.chain_ids.size()) {
        if (i < prev.chain_ids.size() &&
            (j == curr.chain_ids.size() || prev.chain_ids[i] < curr.chain_ids[j])) {
          ++uni;
          ++i;
        } else if (j < curr.chain_ids.size() &&
                   (i == prev.chain_ids.size() ||
                    curr.chain_ids[j] < prev.chain_ids[i])) {
          ++uni;
          ++j;
        } else {
          ++uni;
          ++inter;
          ++i;
          ++j;
        }
      }
      const double jaccard =
          (uni > 0) ? (static_cast<double>(inter) / static_cast<double>(uni))
                    : 0.0;
      if (jaccard >= lineage_overlap_threshold_) {
        row.matched_prev_clusters.push_back(prev.frame_cluster_id);
        row.matched_jaccard.push_back(jaccard);
        prev_to_curr[p].push_back(curr.frame_cluster_id);
      }
    }

    if (row.matched_prev_clusters.empty()) {
      row.event = "born";
    } else if (row.matched_prev_clusters.size() == 1) {
      row.event = "stable";
    } else {
      row.event = "fusion";
      ++cluster_lineage_result_.fusion_events;
    }
    lineage_rows_.push_back(std::move(row));
  }

  for (const auto &matches : prev_to_curr) {
    if (matches.size() > 1)
      ++cluster_lineage_result_.fission_events;
  }
  prev_lineage_clusters_ = std::move(current_clusters);
}

void TitanLLPSCluster::update_dynamics(
    const box &Box, double time_ns, int /*giant_cluster_id*/,
    const std::vector<std::vector<int>> &cluster_pro,
    const std::vector<std::vector<int>> &cluster_rna,
    const std::vector<ChainFrameSummary> &frame_chains) {
  if (!dynamics_enabled_)
    return;

  if (!have_time_window_) {
    first_frame_time_ns_ = time_ns;
    have_time_window_ = true;
  }
  last_frame_time_ns_ = time_ns;

  update_lineage(time_ns, cluster_pro, cluster_rna);

  std::vector<std::size_t> chain_indices(frame_chains.size(), 0);
  for (std::size_t i = 0; i < frame_chains.size(); ++i) {
    const auto &summary = frame_chains[i];
    const uint64_t key = pack_chain_key(summary.species, summary.raw_chain_id);
    auto it = chain_key_to_index_.find(key);
    std::size_t chain_index = 0;
    if (it == chain_key_to_index_.end()) {
      chain_index = chain_states_.size();
      chain_key_to_index_.emplace(key, static_cast<int>(chain_index));
      ensure_chain_capacity(chain_index);
      chain_states_[chain_index].species = summary.species;
      chain_states_[chain_index].raw_chain_id = summary.raw_chain_id;
      chain_index_species_[chain_index] = summary.species;
      chain_index_raw_id_[chain_index] = summary.raw_chain_id;
    } else {
      chain_index = static_cast<std::size_t>(it->second);
    }
    chain_indices[i] = chain_index;
  }

  const std::size_t chain_count = chain_states_.size();
  current_present_buf_.assign(chain_count, 0);
  current_inside_buf_.assign(chain_count, 0);
  current_episode_buf_.assign(chain_count, 0);
  current_unwrapped_x_buf_.assign(chain_count, 0.0);
  current_unwrapped_y_buf_.assign(chain_count, 0.0);
  current_unwrapped_z_buf_.assign(chain_count, 0.0);

  const TitanExactPeriodicBox bmat(Box);
  for (std::size_t i = 0; i < frame_chains.size(); ++i) {
    const auto &summary = frame_chains[i];
    const std::size_t chain_index = chain_indices[i];
    auto &state = chain_states_[chain_index];
    current_present_buf_[chain_index] = 1;
    current_inside_buf_[chain_index] = summary.inside_dense ? 1 : 0;
    membership_frame_counts_[chain_index] += 1;
    if (summary.inside_dense)
      membership_inside_counts_[chain_index] += 1;

    if (!state.seen_any_frame) {
      state.seen_any_frame = true;
      state.first_time_ns = time_ns;
      state.prev_inside = false;
      state.has_prev_com = false;
    }
    state.last_time_ns = time_ns;

    if (!state.has_prev_com) {
      state.prev_wrapped_x = summary.wrapped_com_x;
      state.prev_wrapped_y = summary.wrapped_com_y;
      state.prev_wrapped_z = summary.wrapped_com_z;
      state.unwrapped_x = summary.wrapped_com_x;
      state.unwrapped_y = summary.wrapped_com_y;
      state.unwrapped_z = summary.wrapped_com_z;
      state.has_prev_com = true;
    } else {
      double dx = summary.wrapped_com_x - state.prev_wrapped_x;
      double dy = summary.wrapped_com_y - state.prev_wrapped_y;
      double dz = summary.wrapped_com_z - state.prev_wrapped_z;
      bmat.apply_mic(dx, dy, dz);
      state.unwrapped_x += dx;
      state.unwrapped_y += dy;
      state.unwrapped_z += dz;
      state.prev_wrapped_x = summary.wrapped_com_x;
      state.prev_wrapped_y = summary.wrapped_com_y;
      state.prev_wrapped_z = summary.wrapped_com_z;
    }

    if (!state.prev_inside && summary.inside_dense) {
      state.current_episode_id = next_episode_id_++;
      state.episode_start_ns = time_ns;
      state.episode_censored_start =
          (dynamics_total_frames_ == 0 || !state.seen_any_frame);
      if (summary.species == 0)
        ++protein_entry_events_;
      else
        ++rna_entry_events_;
      if (dynamics_detailed_) {
        if (membership_trace_streaming_ && membership_trace_out_.is_open()) {
          membership_trace_out_ << time_ns << "," << species_label(summary.species)
                                << "," << summary.raw_chain_id << ","
                                << state.current_episode_id << ",1,entry\n";
        } else {
          membership_trace_rows_.push_back(
              {time_ns, summary.species, summary.raw_chain_id,
               state.current_episode_id, 1, "entry"});
        }
      }
    } else if (state.prev_inside && !summary.inside_dense) {
      ResidenceEpisodeRow row;
      row.species = summary.species;
      row.raw_chain_id = summary.raw_chain_id;
      row.episode_id = state.current_episode_id;
      row.start_ns = state.episode_start_ns;
      row.end_ns = time_ns;
      row.duration_ns = std::max(0.0, row.end_ns - row.start_ns);
      row.censored_start = state.episode_censored_start;
      row.censored_end = false;
      residence_rows_.push_back(row);
      if (summary.species == 0)
        ++protein_exit_events_;
      else
        ++rna_exit_events_;
      if (dynamics_detailed_) {
        if (membership_trace_streaming_ && membership_trace_out_.is_open()) {
          membership_trace_out_ << time_ns << "," << species_label(summary.species)
                                << "," << summary.raw_chain_id << ","
                                << state.current_episode_id << ",0,exit\n";
        } else {
          membership_trace_rows_.push_back(
              {time_ns, summary.species, summary.raw_chain_id,
               state.current_episode_id, 0, "exit"});
        }
      }
      state.current_episode_id = 0;
      state.episode_start_ns = 0.0;
      state.episode_censored_start = false;
    }

    state.prev_inside = summary.inside_dense;
    current_episode_buf_[chain_index] = state.current_episode_id;
    current_unwrapped_x_buf_[chain_index] = state.unwrapped_x;
    current_unwrapped_y_buf_[chain_index] = state.unwrapped_y;
    current_unwrapped_z_buf_[chain_index] = state.unwrapped_z;
  }

  const int usable_lag =
      std::min(dynamics_max_lag_frames_, dynamics_ring_count_);
  constexpr std::size_t msd_batch_size = 128;
  std::array<double, msd_batch_size> protein_dx_buf{};
  std::array<double, msd_batch_size> protein_dy_buf{};
  std::array<double, msd_batch_size> protein_dz_buf{};
  std::array<double, msd_batch_size> rna_dx_buf{};
  std::array<double, msd_batch_size> rna_dy_buf{};
  std::array<double, msd_batch_size> rna_dz_buf{};
  auto flush_msd_batch = [&](int species, int lag, double lag_time,
                             std::size_t &used, auto &dx_buf, auto &dy_buf,
                             auto &dz_buf) {
    if (used == 0)
      return;
    const double msd_sum = titan_simd_dispatch::squared_displacement_sum_xyz(
        dx_buf.data(), dy_buf.data(), dz_buf.data(), used);
    const auto lag_idx = static_cast<std::size_t>(lag);
    if (species == 0) {
      msd_sum_protein_[lag_idx] += msd_sum;
      msd_count_protein_[lag_idx] += static_cast<uint64_t>(used);
      msd_time_sum_protein_[lag_idx] += lag_time * static_cast<double>(used);
      msd_time_count_protein_[lag_idx] += static_cast<uint64_t>(used);
    } else {
      msd_sum_rna_[lag_idx] += msd_sum;
      msd_count_rna_[lag_idx] += static_cast<uint64_t>(used);
      msd_time_sum_rna_[lag_idx] += lag_time * static_cast<double>(used);
      msd_time_count_rna_[lag_idx] += static_cast<uint64_t>(used);
    }
    used = 0;
  };

  for (int lag = 1; lag <= usable_lag; ++lag) {
    const int slot = (dynamics_total_frames_ - lag + dynamics_ring_cap_) %
                     dynamics_ring_cap_;
    if (dynamics_ring_chain_capacity_ == 0)
      continue;

    const double lag_time =
        time_ns - frame_time_ring_[static_cast<std::size_t>(slot)];
    const std::size_t slot_base =
        static_cast<std::size_t>(slot) * dynamics_ring_chain_capacity_;
    const std::size_t lag_base =
        static_cast<std::size_t>(lag) * membership_lag_chain_capacity_;
    std::size_t protein_used = 0;
    std::size_t rna_used = 0;
    for (std::size_t chain = 0; chain < chain_states_.size(); ++chain) {
      const std::size_t ring_idx = slot_base + chain;
      const std::size_t lag_idx = lag_base + chain;
      if (!current_present_buf_[chain] || !frame_present_ring_[ring_idx]) {
        continue;
      }

      ++membership_pair_counts_[lag_idx];
      if (current_inside_buf_[chain] && frame_inside_ring_[ring_idx]) {
        ++membership_joint_inside_[lag_idx];
      }

      const int species = chain_states_[chain].species;
      if (species == 0) {
        protein_autocorr_lag_time_sum_[static_cast<std::size_t>(lag)] += lag_time;
        ++protein_autocorr_lag_time_count_[static_cast<std::size_t>(lag)];
      } else {
        rna_autocorr_lag_time_sum_[static_cast<std::size_t>(lag)] += lag_time;
        ++rna_autocorr_lag_time_count_[static_cast<std::size_t>(lag)];
      }

      if (!(current_inside_buf_[chain] && frame_inside_ring_[ring_idx] &&
            current_episode_buf_[chain] > 0 &&
            current_episode_buf_[chain] == frame_episode_ring_[ring_idx])) {
        continue;
      }

      const double dx =
          current_unwrapped_x_buf_[chain] - frame_unwrapped_x_ring_[ring_idx];
      const double dy =
          current_unwrapped_y_buf_[chain] - frame_unwrapped_y_ring_[ring_idx];
      const double dz =
          current_unwrapped_z_buf_[chain] - frame_unwrapped_z_ring_[ring_idx];

      if (species == 0) {
        protein_dx_buf[protein_used] = dx;
        protein_dy_buf[protein_used] = dy;
        protein_dz_buf[protein_used] = dz;
        ++protein_used;
        if (protein_used == msd_batch_size) {
          flush_msd_batch(0, lag, lag_time, protein_used, protein_dx_buf,
                          protein_dy_buf, protein_dz_buf);
        }
      } else {
        rna_dx_buf[rna_used] = dx;
        rna_dy_buf[rna_used] = dy;
        rna_dz_buf[rna_used] = dz;
        ++rna_used;
        if (rna_used == msd_batch_size) {
          flush_msd_batch(1, lag, lag_time, rna_used, rna_dx_buf, rna_dy_buf,
                          rna_dz_buf);
        }
      }
    }

    flush_msd_batch(0, lag, lag_time, protein_used, protein_dx_buf,
                    protein_dy_buf, protein_dz_buf);
    flush_msd_batch(1, lag, lag_time, rna_used, rna_dx_buf, rna_dy_buf,
                    rna_dz_buf);
  }

  const int write_slot = dynamics_total_frames_ % dynamics_ring_cap_;
  frame_time_ring_[static_cast<std::size_t>(write_slot)] = time_ns;
  const std::size_t write_base =
      static_cast<std::size_t>(write_slot) * dynamics_ring_chain_capacity_;
  const std::size_t write_end = write_base + chain_states_.size();
  std::fill(frame_present_ring_.begin() + static_cast<std::ptrdiff_t>(write_base),
            frame_present_ring_.begin() + static_cast<std::ptrdiff_t>(write_end),
            0);
  std::fill(frame_inside_ring_.begin() + static_cast<std::ptrdiff_t>(write_base),
            frame_inside_ring_.begin() + static_cast<std::ptrdiff_t>(write_end),
            0);
  std::fill(frame_episode_ring_.begin() + static_cast<std::ptrdiff_t>(write_base),
            frame_episode_ring_.begin() + static_cast<std::ptrdiff_t>(write_end),
            0);
  std::fill(frame_unwrapped_x_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_base),
            frame_unwrapped_x_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_end),
            0.0);
  std::fill(frame_unwrapped_y_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_base),
            frame_unwrapped_y_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_end),
            0.0);
  std::fill(frame_unwrapped_z_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_base),
            frame_unwrapped_z_ring_.begin() +
                static_cast<std::ptrdiff_t>(write_end),
            0.0);
  for (std::size_t chain = 0; chain < chain_states_.size(); ++chain) {
    const std::size_t ring_idx = write_base + chain;
    frame_present_ring_[ring_idx] = current_present_buf_[chain];
    frame_inside_ring_[ring_idx] = current_inside_buf_[chain];
    frame_episode_ring_[ring_idx] = current_episode_buf_[chain];
    frame_unwrapped_x_ring_[ring_idx] = current_unwrapped_x_buf_[chain];
    frame_unwrapped_y_ring_[ring_idx] = current_unwrapped_y_buf_[chain];
    frame_unwrapped_z_ring_[ring_idx] = current_unwrapped_z_buf_[chain];
  }

  dynamics_ring_count_ =
      std::min(dynamics_ring_count_ + 1, dynamics_ring_cap_ - 1);
  ++dynamics_total_frames_;
}

void TitanLLPSCluster::finalize_residence_summary() {
  for (auto &state : chain_states_) {
    if (!state.prev_inside || state.current_episode_id == 0)
      continue;
    residence_rows_.push_back({state.species, state.raw_chain_id,
                               state.current_episode_id, state.episode_start_ns,
                               last_frame_time_ns_,
                               std::max(0.0, last_frame_time_ns_ -
                                                  state.episode_start_ns),
                               state.episode_censored_start, true});
    state.prev_inside = false;
    state.current_episode_id = 0;
  }

  auto build_summary = [&](int species) {
    LLPSEpisodeSummary summary;
    std::vector<double> complete;
    for (const auto &row : residence_rows_) {
      if (row.species != species)
        continue;
      ++summary.total_episodes;
      if (row.censored_start || row.censored_end) {
        ++summary.censored_episodes;
        continue;
      }
      ++summary.complete_episodes;
      complete.push_back(row.duration_ns);
    }
    if (!complete.empty()) {
      std::sort(complete.begin(), complete.end());
      summary.mean_complete_ns =
          std::accumulate(complete.begin(), complete.end(), 0.0) /
          static_cast<double>(complete.size());
      summary.median_complete_ns = percentile_sorted(complete, 0.5);
      summary.p90_complete_ns = percentile_sorted(complete, 0.9);
      summary.max_complete_ns = complete.back();
    }
    return summary;
  };

  chain_dynamics_result_.protein_residence = build_summary(0);
  chain_dynamics_result_.rna_residence = build_summary(1);
}

void TitanLLPSCluster::finalize_membership_autocorr(TitanExecutor &executor) {
  const int n_lags = std::min(dynamics_max_lag_frames_, dynamics_total_frames_ - 1);
  auto compute_species = [&](int species, LLPSAutocorrSpeciesResult &out,
                             const std::vector<double> &lag_time_sum,
                             const std::vector<uint64_t> &lag_time_count) {
    out.lag_ns.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
    out.mean.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
    out.stddev.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
    out.contributing_chains.assign(static_cast<std::size_t>(n_lags + 1), 0);
    std::vector<int> species_chains;
    for (std::size_t i = 0; i < chain_states_.size(); ++i)
      if (chain_states_[i].species == species)
        species_chains.push_back(static_cast<int>(i));

    uint64_t lag0_count = 0;
    for (int idx : species_chains) {
      const double p = (membership_frame_counts_[static_cast<std::size_t>(idx)] > 0)
                           ? static_cast<double>(
                                 membership_inside_counts_[static_cast<std::size_t>(idx)]) /
                                 static_cast<double>(
                                     membership_frame_counts_[static_cast<std::size_t>(idx)])
                           : 0.0;
      if (p > 0.0 && p < 1.0)
        ++lag0_count;
    }
    if (lag0_count > 0) {
      out.mean[0] = 1.0;
      out.contributing_chains[0] = lag0_count;
    }

    const std::size_t workers =
        std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(),
                                                       static_cast<std::size_t>(n_lags)));
    executor.parallel_do(workers, [&](std::size_t worker) {
      const int begin =
          1 + static_cast<int>(worker * static_cast<std::size_t>(n_lags) / workers);
      const int end =
          1 + static_cast<int>((worker + 1) * static_cast<std::size_t>(n_lags) /
                               workers);
      for (int lag = begin; lag < end; ++lag) {
        std::vector<double> values;
        values.reserve(species_chains.size());
        for (int idx : species_chains) {
          const auto chain = static_cast<std::size_t>(idx);
          const std::size_t lag_idx =
              static_cast<std::size_t>(lag) * membership_lag_chain_capacity_ +
              chain;
          if (membership_pair_counts_[lag_idx] == 0)
            continue;
          const double p =
              (membership_frame_counts_[chain] > 0)
                  ? static_cast<double>(membership_inside_counts_[chain]) /
                        static_cast<double>(membership_frame_counts_[chain])
                  : 0.0;
          if (p <= 0.0 || p >= 1.0)
            continue;
          const double p11 =
              static_cast<double>(membership_joint_inside_[lag_idx]) /
              static_cast<double>(membership_pair_counts_[lag_idx]);
          double value = (p11 - p * p) / (p * (1.0 - p));
          value = std::max(-1.0, std::min(1.0, value));
          values.push_back(value);
        }
        if (lag < static_cast<int>(lag_time_count.size()) &&
            lag_time_count[static_cast<std::size_t>(lag)] > 0) {
          out.lag_ns[static_cast<std::size_t>(lag)] =
              lag_time_sum[static_cast<std::size_t>(lag)] /
              static_cast<double>(lag_time_count[static_cast<std::size_t>(lag)]);
        }
        if (values.empty())
          continue;
        const double mean =
            std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());
        double var = 0.0;
        for (double value : values)
          var += (value - mean) * (value - mean);
        var /= static_cast<double>(values.size());
        out.mean[static_cast<std::size_t>(lag)] = mean;
        out.stddev[static_cast<std::size_t>(lag)] = std::sqrt(std::max(0.0, var));
        out.contributing_chains[static_cast<std::size_t>(lag)] =
            static_cast<uint64_t>(values.size());
      }
    });

    out.tau_ns = fit_tau_curve(out.mean, out.lag_ns);
  };

  compute_species(0, chain_dynamics_result_.protein_membership_autocorr,
                  protein_autocorr_lag_time_sum_,
                  protein_autocorr_lag_time_count_);
  compute_species(1, chain_dynamics_result_.rna_membership_autocorr,
                  rna_autocorr_lag_time_sum_, rna_autocorr_lag_time_count_);

  const double total_duration =
      (have_time_window_ && last_frame_time_ns_ > first_frame_time_ns_)
          ? (last_frame_time_ns_ - first_frame_time_ns_)
          : 0.0;
  double protein_obs_chain_ns = 0.0;
  double rna_obs_chain_ns = 0.0;
  for (const auto &state : chain_states_) {
    if (!state.seen_any_frame)
      continue;
    const double obs = std::max(0.0, state.last_time_ns - state.first_time_ns);
    if (state.species == 0)
      protein_obs_chain_ns += obs;
    else
      rna_obs_chain_ns += obs;
  }

  if (total_duration > 0.0) {
    chain_dynamics_result_.protein_entry_rate_per_ns =
        static_cast<double>(protein_entry_events_) / total_duration;
    chain_dynamics_result_.protein_exit_rate_per_ns =
        static_cast<double>(protein_exit_events_) / total_duration;
    chain_dynamics_result_.rna_entry_rate_per_ns =
        static_cast<double>(rna_entry_events_) / total_duration;
    chain_dynamics_result_.rna_exit_rate_per_ns =
        static_cast<double>(rna_exit_events_) / total_duration;
  }
  if (protein_obs_chain_ns > 0.0) {
    chain_dynamics_result_.protein_entry_rate_per_chain_ns =
        static_cast<double>(protein_entry_events_) / protein_obs_chain_ns;
    chain_dynamics_result_.protein_exit_rate_per_chain_ns =
        static_cast<double>(protein_exit_events_) / protein_obs_chain_ns;
  }
  if (rna_obs_chain_ns > 0.0) {
    chain_dynamics_result_.rna_entry_rate_per_chain_ns =
        static_cast<double>(rna_entry_events_) / rna_obs_chain_ns;
    chain_dynamics_result_.rna_exit_rate_per_chain_ns =
        static_cast<double>(rna_exit_events_) / rna_obs_chain_ns;
  }
}

void TitanLLPSCluster::finalize_dense_msd(TitanExecutor &executor) {
  const int n_lags =
      std::max(0, std::min(dynamics_max_lag_frames_, dynamics_total_frames_ - 1));
  dense_msd_result_.lag_ns.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
  dense_msd_result_.protein_msd_nm2.assign(static_cast<std::size_t>(n_lags + 1),
                                           0.0);
  dense_msd_result_.rna_msd_nm2.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
  dense_msd_result_.protein_counts.assign(static_cast<std::size_t>(n_lags + 1), 0);
  dense_msd_result_.rna_counts.assign(static_cast<std::size_t>(n_lags + 1), 0);

  if (n_lags > 0) {
    const std::size_t workers =
        std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(),
                                                       static_cast<std::size_t>(n_lags)));
    executor.parallel_do(workers, [&](std::size_t worker) {
      const int begin =
          1 + static_cast<int>(worker * static_cast<std::size_t>(n_lags) / workers);
      const int end =
          1 + static_cast<int>((worker + 1) * static_cast<std::size_t>(n_lags) /
                               workers);
      for (int lag = begin; lag < end; ++lag) {
        const auto lag_idx = static_cast<std::size_t>(lag);
        if (msd_count_protein_[lag_idx] > 0) {
          dense_msd_result_.protein_msd_nm2[lag_idx] =
              msd_sum_protein_[lag_idx] /
              static_cast<double>(msd_count_protein_[lag_idx]);
          dense_msd_result_.protein_counts[lag_idx] = msd_count_protein_[lag_idx];
          dense_msd_result_.lag_ns[lag_idx] =
              msd_time_sum_protein_[lag_idx] /
              static_cast<double>(
                  std::max<uint64_t>(1, msd_time_count_protein_[lag_idx]));
        } else if (msd_count_rna_[lag_idx] > 0) {
          dense_msd_result_.lag_ns[lag_idx] =
              msd_time_sum_rna_[lag_idx] /
              static_cast<double>(std::max<uint64_t>(1, msd_time_count_rna_[lag_idx]));
        }

        if (msd_count_rna_[lag_idx] > 0) {
          dense_msd_result_.rna_msd_nm2[lag_idx] =
              msd_sum_rna_[lag_idx] /
              static_cast<double>(msd_count_rna_[lag_idx]);
          dense_msd_result_.rna_counts[lag_idx] = msd_count_rna_[lag_idx];
        }
      }
    });
  }

  auto choose_window = [&](const std::vector<double> &lag_ns,
                           const std::vector<uint64_t> &counts) {
    double first_positive = 0.0;
    double max_lag = 0.0;
    for (std::size_t i = 1; i < lag_ns.size(); ++i) {
      if (counts[i] == 0)
        continue;
      if (first_positive <= 0.0)
        first_positive = lag_ns[i];
      max_lag = lag_ns[i];
    }
    if (first_positive <= 0.0)
      return std::pair<double, double>{0.0, 0.0};
    const double start =
        (msd_fit_start_ns_ >= 0.0) ? msd_fit_start_ns_ : std::max(5.0 * first_positive, first_positive);
    const double end =
        (msd_fit_end_ns_ >= 0.0) ? msd_fit_end_ns_ : 0.25 * max_lag;
    return std::pair<double, double>{start, std::max(start, end)};
  };

  const auto protein_window =
      choose_window(dense_msd_result_.lag_ns, dense_msd_result_.protein_counts);
  const auto rna_window =
      choose_window(dense_msd_result_.lag_ns, dense_msd_result_.rna_counts);
  dense_msd_result_.protein_fit_start_ns = protein_window.first;
  dense_msd_result_.protein_fit_end_ns = protein_window.second;
  dense_msd_result_.rna_fit_start_ns = rna_window.first;
  dense_msd_result_.rna_fit_end_ns = rna_window.second;

  const auto protein_fit = fit_line_window(
      dense_msd_result_.lag_ns, dense_msd_result_.protein_msd_nm2,
      dense_msd_result_.protein_counts, protein_window.first,
      protein_window.second);
  if (protein_fit.ok)
    dense_msd_result_.protein_diffusion_nm2_per_ns =
        std::max(0.0, protein_fit.slope / 6.0);

  const auto rna_fit = fit_line_window(
      dense_msd_result_.lag_ns, dense_msd_result_.rna_msd_nm2,
      dense_msd_result_.rna_counts, rna_window.first, rna_window.second);
  if (rna_fit.ok)
    dense_msd_result_.rna_diffusion_nm2_per_ns =
        std::max(0.0, rna_fit.slope / 6.0);
}

void TitanLLPSCluster::finalize_dynamics(TitanExecutor &executor) {
  if (!dynamics_enabled_)
    return;
  const auto finalize_begin = std::chrono::steady_clock::now();
  perf_.finalize_executor_workers =
      std::max<std::size_t>(1, executor.capacity());
  finalize_residence_summary();
  finalize_membership_autocorr(executor);
  finalize_dense_msd(executor);
  finalize_rna_information(executor);
  perf_.finalize_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                finalize_begin)
                                .count();
}

#ifdef TITAN_LLPS_TESTING
void TitanLLPSCluster::debug_seed_dense_msd(
    int total_frames, int max_lag_frames, double fit_start_ns,
    double fit_end_ns, const std::vector<double> &protein_sum,
    const std::vector<uint64_t> &protein_count,
    const std::vector<double> &protein_time_sum,
    const std::vector<uint64_t> &protein_time_count,
    const std::vector<double> &rna_sum, const std::vector<uint64_t> &rna_count,
    const std::vector<double> &rna_time_sum,
    const std::vector<uint64_t> &rna_time_count) {
  dynamics_enabled_ = true;
  dynamics_total_frames_ = total_frames;
  dynamics_max_lag_frames_ = max_lag_frames;
  msd_fit_start_ns_ = fit_start_ns;
  msd_fit_end_ns_ = fit_end_ns;
  msd_sum_protein_ = protein_sum;
  msd_count_protein_ = protein_count;
  msd_time_sum_protein_ = protein_time_sum;
  msd_time_count_protein_ = protein_time_count;
  msd_sum_rna_ = rna_sum;
  msd_count_rna_ = rna_count;
  msd_time_sum_rna_ = rna_time_sum;
  msd_time_count_rna_ = rna_time_count;
  dense_msd_result_ = {};
}
#endif

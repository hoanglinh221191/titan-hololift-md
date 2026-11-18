// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "titan_exact_periodic_box.h"
#include "titan_llps_geometry.h"
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

  // STR-09. Lineage by containment, not by symmetric Jaccard: a previous
  // cluster P belongs to a current cluster C when at least
  // lineage_overlap_threshold_ (default 0.5) of P's members are in C, and C
  // descends from P when that share of C's members were in P. Fusion: two or
  // more previous clusters belong to one current cluster; fission: two or
  // more current clusters descend from one previous cluster. With Jaccard
  // >= 0.30 a 5-chain droplet joining a 100-chain one had J = 5/105 and the
  // fusion (and the reverse fission) was never counted.
  std::vector<std::vector<int>> prev_to_curr(prev_lineage_clusters_.size());
  for (const auto &curr : current_clusters) {
    LineageClusterRow row;
    row.time_ns = time_ns;
    row.frame_cluster_id = curr.frame_cluster_id;
    row.protein_chains = curr.protein_chains;
    row.rna_chains = curr.rna_chains;
    std::size_t sources = 0;

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
      if (inter == 0)
        continue;
      const double jaccard =
          (uni > 0) ? (static_cast<double>(inter) / static_cast<double>(uni))
                    : 0.0;
      const double prev_share = static_cast<double>(inter) /
                                static_cast<double>(prev.chain_ids.size());
      const double curr_share = static_cast<double>(inter) /
                                static_cast<double>(curr.chain_ids.size());
      const bool prev_in_curr = prev_share >= lineage_overlap_threshold_;
      const bool curr_from_prev = curr_share >= lineage_overlap_threshold_;
      if (prev_in_curr || curr_from_prev) {
        row.matched_prev_clusters.push_back(prev.frame_cluster_id);
        row.matched_jaccard.push_back(jaccard);
        row.matched_prev_share.push_back(prev_share);
      }
      if (prev_in_curr)
        ++sources;
      if (curr_from_prev)
        prev_to_curr[p].push_back(curr.frame_cluster_id);
    }

    if (row.matched_prev_clusters.empty()) {
      row.event = "born";
    } else if (sources < 2) {
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
  {
    // STR-01: slab axis and normal of the lateral MSD from this frame's cell
    const auto slab = titan_llps_geometry::slab_normal(
        titan_llps_geometry::Cell::from_box(Box, 0.0, 0.0, 0.0));
    slab_normal_axis_ = slab.axis;
    slab_normal_ = slab.normal;
  }

  // ---- Liquidity v2, pass 1 -------------------------------------------
  // Minimum-image displacement of every unit since the previous analysed
  // frame, and from those the net translation of the dense phase. The
  // condensate is free to translate inside the box (COM removal only pins the
  // whole system, water included), and that translation is common to every
  // resident unit, so it must be removed before a single-unit MSD means
  // self-diffusion rather than drift of the droplet.
  frame_dx_buf_.assign(frame_chains.size(), 0.0);
  frame_dy_buf_.assign(frame_chains.size(), 0.0);
  frame_dz_buf_.assign(frame_chains.size(), 0.0);
  frame_drift_use_buf_.assign(frame_chains.size(), 0);
  {
    double drift_num_x = 0.0;
    double drift_num_y = 0.0;
    double drift_num_z = 0.0;
    double drift_den = 0.0;
    for (std::size_t i = 0; i < frame_chains.size(); ++i) {
      const auto &summary = frame_chains[i];
      const auto &state = chain_states_[chain_indices[i]];
      if (!state.has_prev_com)
        continue;
      double dx = summary.wrapped_com_x - state.prev_wrapped_x;
      double dy = summary.wrapped_com_y - state.prev_wrapped_y;
      double dz = summary.wrapped_com_z - state.prev_wrapped_z;
      bmat.apply_mic(dx, dy, dz);
      frame_dx_buf_[i] = dx;
      frame_dy_buf_[i] = dy;
      frame_dz_buf_[i] = dz;
      if (state.prev_inside && summary.inside_dense) {
        frame_drift_use_buf_[i] = 1;
        const double w = static_cast<double>(std::max(1, summary.n_beads));
        drift_num_x += w * dx;
        drift_num_y += w * dy;
        drift_num_z += w * dz;
        drift_den += w;
      }
    }
    if (msd_drift_correct_ && drift_den > 0.0) {
      dense_drift_x_ += drift_num_x / drift_den;
      dense_drift_y_ += drift_num_y / drift_den;
      dense_drift_z_ += drift_num_z / drift_den;
    }
  }

  for (std::size_t i = 0; i < frame_chains.size(); ++i) {
    const auto &summary = frame_chains[i];
    const std::size_t chain_index = chain_indices[i];
    auto &state = chain_states_[chain_index];
    current_present_buf_[chain_index] = 1;
    current_inside_buf_[chain_index] = summary.inside_dense ? 1 : 0;
    membership_frame_counts_[chain_index] += 1;
    if (summary.inside_dense)
      membership_inside_counts_[chain_index] += 1;
    // Pooled occupancy <h>: every unit counts, permanent residents included.
    if (summary.species == 0) {
      ++pooled_frames_protein_;
      if (summary.inside_dense)
        ++pooled_inside_protein_;
    } else {
      ++pooled_frames_rna_;
      if (summary.inside_dense)
        ++pooled_inside_rna_;
    }

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
      state.unwrapped_x += frame_dx_buf_[i];
      state.unwrapped_y += frame_dy_buf_[i];
      state.unwrapped_z += frame_dz_buf_[i];
      state.prev_wrapped_x = summary.wrapped_com_x;
      state.prev_wrapped_y = summary.wrapped_com_y;
      state.prev_wrapped_z = summary.wrapped_com_z;
    }

    if (!state.prev_inside && summary.inside_dense) {
      state.current_episode_id = next_episode_id_++;
      state.episode_start_ns = time_ns;
      state.episode_censored_start =
          (dynamics_total_frames_ == 0 || !state.seen_any_frame);
      // Liquidity v3: a unit that is already inside at its first analysed
      // frame did not "enter" (left-censored); counting it made the entry
      // rate equal (units / window) even when nothing was ever exchanged.
      if (state.first_time_ns < time_ns) {
        if (summary.species == 0)
          ++protein_entry_events_;
        else
          ++rna_entry_events_;
      }
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
    // Drift removed and converted to nanometres. TITAN carries coordinates in
    // angstrom everywhere else; the MSD outputs are labelled nm^2, so the
    // conversion has to happen before anything is accumulated.
    current_unwrapped_x_buf_[chain_index] =
        0.1 * (state.unwrapped_x - dense_drift_x_);
    current_unwrapped_y_buf_[chain_index] =
        0.1 * (state.unwrapped_y - dense_drift_y_);
    current_unwrapped_z_buf_[chain_index] =
        0.1 * (state.unwrapped_z - dense_drift_z_);
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
      const bool joint_inside =
          (current_inside_buf_[chain] && frame_inside_ring_[ring_idx]);
      if (joint_inside) {
        ++membership_joint_inside_[lag_idx];
      }

      const int species = chain_states_[chain].species;
      if (species == 0) {
        ++pooled_pairs_protein_[static_cast<std::size_t>(lag)];
        if (joint_inside)
          ++pooled_joint_protein_[static_cast<std::size_t>(lag)];
      } else {
        ++pooled_pairs_rna_[static_cast<std::size_t>(lag)];
        if (joint_inside)
          ++pooled_joint_rna_[static_cast<std::size_t>(lag)];
      }
      if (species == 0) {
        protein_autocorr_lag_time_sum_[static_cast<std::size_t>(lag)] += lag_time;
        ++protein_autocorr_lag_time_count_[static_cast<std::size_t>(lag)];
      } else {
        rna_autocorr_lag_time_sum_[static_cast<std::size_t>(lag)] += lag_time;
        ++rna_autocorr_lag_time_count_[static_cast<std::size_t>(lag)];
      }

      // Liquidity v2: pair on "inside the dense phase at both endpoints".
      // The old condition also demanded one uninterrupted residence episode,
      // which discards every pair whose membership flag flickered once in
      // between. For a unit whose dense/dilute classification sits near the
      // decision boundary no episode ever spans a single lag, so the whole
      // species silently produced zero MSD samples.
      if (!joint_inside) {
        continue;
      }

      const double dx =
          current_unwrapped_x_buf_[chain] - frame_unwrapped_x_ring_[ring_idx];
      const double dy =
          current_unwrapped_y_buf_[chain] - frame_unwrapped_y_ring_[ring_idx];
      const double dz =
          current_unwrapped_z_buf_[chain] - frame_unwrapped_z_ring_[ring_idx];

      // Liquidity v3: lateral (in-plane) squared displacement; STR-01: the
      // component along the slab normal is removed, not a Cartesian one.
      {
        const double lat =
            titan_llps_geometry::lateral_sq({dx, dy, dz}, slab_normal_);
        if (species == 0)
          msd_lat_sum_protein_[static_cast<std::size_t>(lag)] += lat;
        else
          msd_lat_sum_rna_[static_cast<std::size_t>(lag)] += lat;
      }

      // FRAP analogue: has this unit moved further than one mesh length?
      {
        const double thr = mobile_displacement_nm_;
        const bool moved = (dx * dx + dy * dy + dz * dz) > thr * thr;
        if (species == 0) {
          ++disp_all_protein_[static_cast<std::size_t>(lag)];
          if (moved)
            ++disp_gt_protein_[static_cast<std::size_t>(lag)];
        } else {
          ++disp_all_rna_[static_cast<std::size_t>(lag)];
          if (moved)
            ++disp_gt_rna_[static_cast<std::size_t>(lag)];
        }
      }

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
    } else {
      // Liquidity v3: no residence episode with both ends observed -> the
      // statistics are undefined, not zero.
      const double q = std::numeric_limits<double>::quiet_NaN();
      summary.mean_complete_ns = summary.median_complete_ns = q;
      summary.p90_complete_ns = summary.max_complete_ns = q;
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
      const auto ci = static_cast<std::size_t>(idx);
      const double p = (membership_frame_counts_[ci] > 0)
                           ? static_cast<double>(membership_inside_counts_[ci]) /
                                 static_cast<double>(membership_frame_counts_[ci])
                           : 0.0;
      if (p > 0.0 && p < 1.0)
        ++lag0_count;
      // Material-exchange census (see LLPSAutocorrSpeciesResult).
      if (membership_frame_counts_[ci] == 0)
        continue;
      if (membership_inside_counts_[ci] == 0)
        ++out.units_never;
      else if (membership_inside_counts_[ci] == membership_frame_counts_[ci])
        ++out.units_permanent;
      else
        ++out.units_exchanging;
    }
    {
      const uint64_t seen = out.units_permanent + out.units_exchanging;
      out.exchange_fraction =
          (seen > 0) ? static_cast<double>(out.units_exchanging) /
                           static_cast<double>(seen)
                     : std::numeric_limits<double>::quiet_NaN();
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

    // Liquidity v3: with no unit whose membership ever changed, the per-unit
    // correlation is 0/0 and tau is undefined (it used to come out as 0).
    out.tau_ns = (lag0_count > 0) ? fit_tau_curve(out.mean, out.lag_ns)
                                  : std::numeric_limits<double>::quiet_NaN();

    // ---- Liquidity v2 -------------------------------------------------
    // `out.mean` above is a per-unit average that drops every unit with
    // occupancy p == 0 or p == 1. Those are exactly the permanent residents,
    // i.e. the immobile population a FRAP plateau reports, so the per-unit
    // curve cannot be compared with FRAP at all. Accumulate the pooled
    // correlation instead and split it into a decaying amplitude A (mobile)
    // and a plateau 1 - A (immobile).
    out.units = static_cast<uint64_t>(species_chains.size());
    if (n_lags >= 1) {
      const std::vector<uint64_t> &pooled_joint =
          (species == 0) ? pooled_joint_protein_ : pooled_joint_rna_;
      const std::vector<uint64_t> &pooled_pairs =
          (species == 0) ? pooled_pairs_protein_ : pooled_pairs_rna_;
      const uint64_t inside_total =
          (species == 0) ? pooled_inside_protein_ : pooled_inside_rna_;
      const uint64_t frames_total =
          (species == 0) ? pooled_frames_protein_ : pooled_frames_rna_;
      const double p = (frames_total > 0)
                           ? static_cast<double>(inside_total) /
                                 static_cast<double>(frames_total)
                           : 0.0;
      out.occupancy = p;
      out.pooled.assign(static_cast<std::size_t>(n_lags + 1), 0.0);
      out.pooled_pairs.assign(static_cast<std::size_t>(n_lags + 1), 0);
      if (p > 0.0 && p < 1.0) {
        out.pooled[0] = 1.0;
        const double denom = p * (1.0 - p);
        for (int lag = 1; lag <= n_lags; ++lag) {
          const auto li = static_cast<std::size_t>(lag);
          if (li >= pooled_pairs.size() || pooled_pairs[li] == 0)
            continue;
          const double p11 = static_cast<double>(pooled_joint[li]) /
                             static_cast<double>(pooled_pairs[li]);
          out.pooled[li] =
              std::max(-1.0, std::min(1.0, (p11 - p * p) / denom));
          out.pooled_pairs[li] = pooled_pairs[li];
        }
        double tail_sum = 0.0;
        int tail_n = 0;
        const int tail_begin = std::max(1, (4 * n_lags) / 5);
        for (int lag = tail_begin; lag <= n_lags; ++lag) {
          const auto li = static_cast<std::size_t>(lag);
          if (li < out.pooled_pairs.size() && out.pooled_pairs[li] > 0) {
            tail_sum += out.pooled[li];
            ++tail_n;
          }
        }
        const double plateau =
            (tail_n > 0)
                ? std::max(0.0, std::min(1.0, tail_sum /
                                                  static_cast<double>(tail_n)))
                : 0.0;
        out.pooled_plateau = plateau;
        out.mobile_fraction = std::max(0.0, std::min(1.0, 1.0 - plateau));
        if (out.mobile_fraction > 1e-6) {
          std::vector<double> norm(out.pooled.size(), 0.0);
          for (std::size_t li = 0; li < out.pooled.size(); ++li)
            norm[li] = (out.pooled[li] - plateau) / out.mobile_fraction;
          out.pooled_tau_ns = fit_tau_curve(norm, out.lag_ns);
        }
      } else if (frames_total > 0) {
        // Occupancy pinned at 0 or 1: no exchange happened in this window.
        // Liquidity v3: the pooled correlation is then 0/0. v2 reported an
        // "immobile plateau" of 1 and amplitude 0, which reads like a
        // measurement; it is not one. Report undefined instead.
        const double q = std::numeric_limits<double>::quiet_NaN();
        out.pooled[0] = q;
        out.pooled_plateau = q;
        out.mobile_fraction = q;
        out.pooled_tau_ns = q;
      }
    }
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

  // ---- Liquidity v2 ----------------------------------------------------
  // The old window was 5 x (first lag) .. 0.25 x (max lag) with no check that
  // the curve was Fickian anywhere in it. In a condensate the short-lag part
  // of the MSD is not diffusive (correlated / cage motion), so a straight-line
  // slope taken there is not 6D and silently inflates D. Locate the longest
  // run of lags whose local log-log slope is 1 within tolerance, fit there,
  // and refuse to report a diffusion coefficient when no such run exists.
  // ---- Liquidity v3 ----------------------------------------------------
  // v2 estimated the local exponent from a +/-2 *point* central difference
  // on linearly spaced lags and accepted any run of >= 4 points. At long lags
  // that difference spans a log-time interval of ~1e-2 (pure noise), at short
  // lags 4 points span 0.4-1.6 ns, so a "Fickian window" could be declared on
  // a stretch covering less than a factor 4 of noise. v3 measures the local
  // exponent by a log-log least-squares fit over a fixed factor-2 window in
  // time, [t/sqrt2, t*sqrt2], and only accepts a Fickian run that covers at
  // least a factor kMinSpanRatio (one decade) in lag time.
  constexpr double kExpLo = 0.85;
  constexpr double kExpHi = 1.15;
  constexpr int kMinWindowPoints = 3;
  constexpr double kMinSpanRatio = 10.0;  // one decade in lag time
  const double nan_value = std::numeric_limits<double>::quiet_NaN();
  const double kSqrt2 = std::sqrt(2.0);

  auto analyse_species = [&](const std::vector<double> &msd,
                             const std::vector<uint64_t> &counts,
                             double &fit_start, double &fit_end,
                             double &diffusion, double &exponent,
                             double &intercept, bool &diffusive,
                             std::vector<double> &alpha_out,
                             double dim_factor = 6.0) {
    fit_start = 0.0;
    fit_end = 0.0;
    diffusion = nan_value;
    exponent = nan_value;
    intercept = nan_value;
    diffusive = false;

    const auto &lag_ns = dense_msd_result_.lag_ns;
    alpha_out.assign(lag_ns.size(), nan_value);
    std::vector<std::size_t> usable;
    for (std::size_t i = 1; i < lag_ns.size() && i < msd.size() &&
                            i < counts.size();
         ++i) {
      if (counts[i] == 0 || !(lag_ns[i] > 0.0) || !(msd[i] > 0.0))
        continue;
      usable.push_back(i);
    }
    if (usable.size() < static_cast<std::size_t>(kMinWindowPoints))
      return;

    // Drop the statistically thin tail: keep lags whose sample count is at
    // least a fifth of the shortest lag's.
    const uint64_t ref_count = counts[usable.front()];
    while (usable.size() > static_cast<std::size_t>(kMinWindowPoints) &&
           counts[usable.back()] * 5 < ref_count)
      usable.pop_back();

    // Local exponent: OLS slope of ln MSD on ln t over [t/sqrt2, t*sqrt2].
    const std::size_t n = usable.size();
    std::vector<double> alpha(n, nan_value);
    {
      std::size_t lo = 0;
      std::size_t hi = 0;
      for (std::size_t k = 0; k < n; ++k) {
        const double t = lag_ns[usable[k]];
        while (lo < n && lag_ns[usable[lo]] < t / kSqrt2)
          ++lo;
        if (hi < k)
          hi = k;
        while (hi + 1 < n && lag_ns[usable[hi + 1]] <= t * kSqrt2)
          ++hi;
        if (hi + 1 < kMinWindowPoints + lo)
          continue;
        long double sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
        for (std::size_t j = lo; j <= hi; ++j) {
          const long double x = std::log(lag_ns[usable[j]]);
          const long double y = std::log(msd[usable[j]]);
          sx += x;
          sy += y;
          sxx += x * x;
          sxy += x * y;
          m += 1;
        }
        const long double den = m * sxx - sx * sx;
        if (den > 0)
          alpha[k] = static_cast<double>((m * sxy - sx * sy) / den);
      }
    }
    for (std::size_t k = 0; k < n; ++k)
      alpha_out[usable[k]] = alpha[k];

    // Longest (in log time) contiguous run with exponent inside the band.
    std::size_t best_begin = 0;
    std::size_t best_end = 0;
    double best_ratio = 0.0;
    for (std::size_t k = 0; k < n;) {
      const bool in_band = std::isfinite(alpha[k]) && alpha[k] >= kExpLo &&
                           alpha[k] <= kExpHi;
      if (!in_band) {
        ++k;
        continue;
      }
      std::size_t j = k;
      while (j + 1 < n && usable[j + 1] == usable[j] + 1 &&
             std::isfinite(alpha[j + 1]) && alpha[j + 1] >= kExpLo &&
             alpha[j + 1] <= kExpHi)
        ++j;
      const double ratio = lag_ns[usable[j]] / lag_ns[usable[k]];
      if (ratio > best_ratio) {
        best_ratio = ratio;
        best_begin = k;
        best_end = j;
      }
      k = j + 1;
    }

    const bool explicit_window =
        (msd_fit_start_ns_ >= 0.0 || msd_fit_end_ns_ >= 0.0);
    if (explicit_window) {
      fit_start = (msd_fit_start_ns_ >= 0.0) ? msd_fit_start_ns_
                                             : lag_ns[usable.front()];
      fit_end = (msd_fit_end_ns_ >= 0.0) ? msd_fit_end_ns_
                                         : lag_ns[usable.back()];
      fit_end = std::max(fit_start, fit_end);
    } else if (best_ratio >= kMinSpanRatio) {
      fit_start = lag_ns[usable[best_begin]];
      fit_end = lag_ns[usable[best_end]];
    } else {
      // No Fickian stretch at all. Report the measured exponent over the
      // whole usable range so the caller can see how far from diffusive the
      // curve is, and leave the diffusion coefficient undefined.
      fit_start = lag_ns[usable.front()];
      fit_end = lag_ns[usable.back()];
      std::vector<double> log_t;
      std::vector<double> log_m;
      std::vector<uint64_t> ones;
      for (std::size_t k = 0; k < n; ++k) {
        log_t.push_back(std::log(lag_ns[usable[k]]));
        log_m.push_back(std::log(msd[usable[k]]));
        ones.push_back(1);
      }
      const auto log_fit = fit_line_window(log_t, log_m, ones,
                                           -std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max());
      if (log_fit.ok)
        exponent = log_fit.slope;
      return;
    }

    // Log-log exponent over the chosen window.
    std::vector<double> log_t;
    std::vector<double> log_m;
    std::vector<uint64_t> ones;
    for (std::size_t k = 0; k < n; ++k) {
      const double t = lag_ns[usable[k]];
      if (t < fit_start || t > fit_end)
        continue;
      log_t.push_back(std::log(t));
      log_m.push_back(std::log(msd[usable[k]]));
      ones.push_back(1);
    }
    if (log_t.size() >= 2) {
      const auto log_fit = fit_line_window(log_t, log_m, ones,
                                           -std::numeric_limits<double>::max(),
                                           std::numeric_limits<double>::max());
      if (log_fit.ok)
        exponent = log_fit.slope;
    }

    // Only the thin-tail-filtered lags enter the linear fit.
    std::vector<uint64_t> fit_counts(counts.size(), 0);
    for (std::size_t k = 0; k < n; ++k)
      fit_counts[usable[k]] = counts[usable[k]];
    const auto fit =
        fit_line_window(lag_ns, msd, fit_counts, fit_start, fit_end);
    if (!fit.ok)
      return;
    intercept = fit.intercept;
    diffusive = std::isfinite(exponent) && exponent >= kExpLo &&
                exponent <= kExpHi && fit.slope > 0.0 &&
                (explicit_window || best_ratio >= kMinSpanRatio);
    if (diffusive)
      diffusion = fit.slope / dim_factor;
  };

  analyse_species(dense_msd_result_.protein_msd_nm2,
                  dense_msd_result_.protein_counts,
                  dense_msd_result_.protein_fit_start_ns,
                  dense_msd_result_.protein_fit_end_ns,
                  dense_msd_result_.protein_diffusion_nm2_per_ns,
                  dense_msd_result_.protein_msd_exponent,
                  dense_msd_result_.protein_msd_intercept_nm2,
                  dense_msd_result_.protein_diffusive,
                  dense_msd_result_.protein_local_alpha);
  analyse_species(dense_msd_result_.rna_msd_nm2, dense_msd_result_.rna_counts,
                  dense_msd_result_.rna_fit_start_ns,
                  dense_msd_result_.rna_fit_end_ns,
                  dense_msd_result_.rna_diffusion_nm2_per_ns,
                  dense_msd_result_.rna_msd_exponent,
                  dense_msd_result_.rna_msd_intercept_nm2,
                  dense_msd_result_.rna_diffusive,
                  dense_msd_result_.rna_local_alpha);

  // Liquidity v3: lateral (in-plane) MSD, D = slope / 4.
  {
    const std::size_t nl = dense_msd_result_.lag_ns.size();
    dense_msd_result_.slab_normal_axis = slab_normal_axis_;
    dense_msd_result_.slab_normal = slab_normal_;
    dense_msd_result_.protein_msd_lat_nm2.assign(nl, 0.0);
    dense_msd_result_.rna_msd_lat_nm2.assign(nl, 0.0);
    for (std::size_t i = 1; i < nl; ++i) {
      if (i < msd_count_protein_.size() && msd_count_protein_[i] > 0)
        dense_msd_result_.protein_msd_lat_nm2[i] =
            msd_lat_sum_protein_[i] /
            static_cast<double>(msd_count_protein_[i]);
      if (i < msd_count_rna_.size() && msd_count_rna_[i] > 0)
        dense_msd_result_.rna_msd_lat_nm2[i] =
            msd_lat_sum_rna_[i] / static_cast<double>(msd_count_rna_[i]);
    }
    analyse_species(dense_msd_result_.protein_msd_lat_nm2,
                    dense_msd_result_.protein_counts,
                    dense_msd_result_.protein_lat_fit_start_ns,
                    dense_msd_result_.protein_lat_fit_end_ns,
                    dense_msd_result_.protein_lat_diffusion_nm2_per_ns,
                    dense_msd_result_.protein_lat_exponent,
                    dense_msd_result_.protein_lat_intercept_nm2,
                    dense_msd_result_.protein_lat_diffusive,
                    dense_msd_result_.protein_lat_local_alpha, 4.0);
    analyse_species(dense_msd_result_.rna_msd_lat_nm2,
                    dense_msd_result_.rna_counts,
                    dense_msd_result_.rna_lat_fit_start_ns,
                    dense_msd_result_.rna_lat_fit_end_ns,
                    dense_msd_result_.rna_lat_diffusion_nm2_per_ns,
                    dense_msd_result_.rna_lat_exponent,
                    dense_msd_result_.rna_lat_intercept_nm2,
                    dense_msd_result_.rna_lat_diffusive,
                    dense_msd_result_.rna_lat_local_alpha, 4.0);
  }

  // Liquidity v3: whole RNA molecule COM MSD, pooled over RNA molecules and
  // time origins, pairs inside the dense phase at both ends.
  {
    const int nl = std::max(0, n_lags);
    dense_msd_result_.rna_molecule_lag_ns.assign(static_cast<std::size_t>(nl + 1), 0.0);
    dense_msd_result_.rna_molecule_msd_nm2.assign(static_cast<std::size_t>(nl + 1), 0.0);
    dense_msd_result_.rna_molecule_msd_lat_nm2.assign(static_cast<std::size_t>(nl + 1), 0.0);
    dense_msd_result_.rna_molecule_counts.assign(static_cast<std::size_t>(nl + 1), 0);
    dense_msd_result_.rna_molecules = rna_molecule_tracks_.size();
    std::vector<double> tsum(static_cast<std::size_t>(nl + 1), 0.0);
    std::vector<double> s3(static_cast<std::size_t>(nl + 1), 0.0);
    std::vector<double> s2(static_cast<std::size_t>(nl + 1), 0.0);
    for (const auto &kv : rna_molecule_tracks_) {
      const auto &tr = kv.second;
      const std::size_t nf = tr.t_ns.size();
      for (int lag = 1; lag <= nl; ++lag) {
        const auto L = static_cast<std::size_t>(lag);
        for (std::size_t a = 0; a + L < nf; ++a) {
          if (!tr.inside[a] || !tr.inside[a + L])
            continue;
          const double dx = tr.x[a + L] - tr.x[a];
          const double dy = tr.y[a + L] - tr.y[a];
          const double dz = tr.z[a + L] - tr.z[a];
          s3[L] += dx * dx + dy * dy + dz * dz;
          s2[L] += titan_llps_geometry::lateral_sq({dx, dy, dz}, slab_normal_);
          tsum[L] += tr.t_ns[a + L] - tr.t_ns[a];
          ++dense_msd_result_.rna_molecule_counts[L];
        }
      }
    }
    for (int lag = 1; lag <= nl; ++lag) {
      const auto L = static_cast<std::size_t>(lag);
      const auto c = dense_msd_result_.rna_molecule_counts[L];
      if (c == 0)
        continue;
      dense_msd_result_.rna_molecule_lag_ns[L] = tsum[L] / static_cast<double>(c);
      dense_msd_result_.rna_molecule_msd_nm2[L] = s3[L] / static_cast<double>(c);
      dense_msd_result_.rna_molecule_msd_lat_nm2[L] = s2[L] / static_cast<double>(c);
    }
  }

  // Liquidity v3: MSD and apparent D = MSD/(6t) at fixed lags. These exist
  // whether or not a Fickian regime was reached and are the quantities that
  // can be compared across states and temperatures at equal lag.
  {
    const auto at_lag = [&](const std::vector<double> &msd,
                            const std::vector<uint64_t> &cnt, double t) {
      std::size_t best = 0;
      double err = std::numeric_limits<double>::max();
      for (std::size_t i = 1; i < dense_msd_result_.lag_ns.size(); ++i) {
        if (i >= cnt.size() || cnt[i] == 0)
          continue;
        const double e = std::abs(dense_msd_result_.lag_ns[i] - t);
        if (e < err) {
          err = e;
          best = i;
        }
      }
      // Accept only an exact (to half a frame) match.
      if (best == 0 || err > 0.5 * dense_msd_result_.lag_ns[1] + 1e-9)
        return nan_value;
      return msd[best];
    };
    dense_msd_result_.fixed_lags_ns = {4.0, 10.0, 40.0, 100.0, 200.0, 250.0};
    dense_msd_result_.protein_msd_fixed.clear();
    dense_msd_result_.rna_msd_fixed.clear();
    for (double t : dense_msd_result_.fixed_lags_ns) {
      dense_msd_result_.protein_msd_fixed.push_back(
          at_lag(dense_msd_result_.protein_msd_nm2,
                 dense_msd_result_.protein_counts, t));
      dense_msd_result_.rna_msd_fixed.push_back(
          at_lag(dense_msd_result_.rna_msd_nm2, dense_msd_result_.rna_counts,
                 t));
    }
  }

  // FRAP-comparable mobile fraction: how much of the dense phase translates
  // further than one mesh length within a fixed lag.
  dense_msd_result_.drift_corrected = msd_drift_correct_;
  dense_msd_result_.dense_drift_removed_nm =
      0.1 * std::sqrt(dense_drift_x_ * dense_drift_x_ +
                      dense_drift_y_ * dense_drift_y_ +
                      dense_drift_z_ * dense_drift_z_);
  dense_msd_result_.mobile_displacement_nm = mobile_displacement_nm_;
  {
    const double window_ns =
        (have_time_window_ && last_frame_time_ns_ > first_frame_time_ns_)
            ? (last_frame_time_ns_ - first_frame_time_ns_)
            : 0.0;
    double target = (mobile_lag_ns_ >= 0.0) ? mobile_lag_ns_ : 0.25 * window_ns;
    std::size_t best = 0;
    double best_err = std::numeric_limits<double>::max();
    for (std::size_t i = 1; i < dense_msd_result_.lag_ns.size(); ++i) {
      const bool has_data =
          (i < disp_all_protein_.size() && disp_all_protein_[i] > 0) ||
          (i < disp_all_rna_.size() && disp_all_rna_[i] > 0);
      if (!has_data)
        continue;
      const double err = std::abs(dense_msd_result_.lag_ns[i] - target);
      if (err < best_err) {
        best_err = err;
        best = i;
      }
    }
    if (best > 0) {
      dense_msd_result_.mobile_lag_ns = dense_msd_result_.lag_ns[best];
      if (best < disp_all_protein_.size() && disp_all_protein_[best] > 0) {
        dense_msd_result_.protein_mobile_pairs = disp_all_protein_[best];
        dense_msd_result_.protein_mobile_fraction_displacement =
            static_cast<double>(disp_gt_protein_[best]) /
            static_cast<double>(disp_all_protein_[best]);
      } else {
        dense_msd_result_.protein_mobile_fraction_displacement = nan_value;
      }
      if (best < disp_all_rna_.size() && disp_all_rna_[best] > 0) {
        dense_msd_result_.rna_mobile_pairs = disp_all_rna_[best];
        dense_msd_result_.rna_mobile_fraction_displacement =
            static_cast<double>(disp_gt_rna_[best]) /
            static_cast<double>(disp_all_rna_[best]);
      } else {
        dense_msd_result_.rna_mobile_fraction_displacement = nan_value;
      }
    } else {
      dense_msd_result_.protein_mobile_fraction_displacement = nan_value;
      dense_msd_result_.rna_mobile_fraction_displacement = nan_value;
    }
  }
  // Liquidity v3: the whole displacement mobile-fraction curve, so that the
  // number can be read at any lag instead of at one arbitrary one.
  {
    const std::size_t nl = dense_msd_result_.lag_ns.size();
    dense_msd_result_.protein_mobile_curve.assign(nl, nan_value);
    dense_msd_result_.rna_mobile_curve.assign(nl, nan_value);
    dense_msd_result_.protein_mobile_curve_pairs.assign(nl, 0);
    dense_msd_result_.rna_mobile_curve_pairs.assign(nl, 0);
    for (std::size_t i = 1; i < nl; ++i) {
      if (i < disp_all_protein_.size() && disp_all_protein_[i] > 0) {
        dense_msd_result_.protein_mobile_curve[i] =
            static_cast<double>(disp_gt_protein_[i]) /
            static_cast<double>(disp_all_protein_[i]);
        dense_msd_result_.protein_mobile_curve_pairs[i] = disp_all_protein_[i];
      }
      if (i < disp_all_rna_.size() && disp_all_rna_[i] > 0) {
        dense_msd_result_.rna_mobile_curve[i] =
            static_cast<double>(disp_gt_rna_[i]) /
            static_cast<double>(disp_all_rna_[i]);
        dense_msd_result_.rna_mobile_curve_pairs[i] = disp_all_rna_[i];
      }
    }
  }
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

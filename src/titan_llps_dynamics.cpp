// Split from titan_llps.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps.h"
#include "info.h"
#include "titan_llps_stats.h"
#include "titan_ui.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>

using namespace titan_llps_stats;

void TitanLLPS::update_contact_dynamics(double time_ns, int frame_idx,
                                        const std::vector<uint64_t> &sticker_pairs,
                                        const std::vector<uint64_t> &protein_pairs,
                                        const std::vector<uint64_t> &protein_rna_pairs) {
  if (!dynamics_enabled_)
    return;

  const bool had_previous_frame = have_contact_time_window_;
  if (!have_contact_time_window_) {
    first_contact_time_ns_ = time_ns;
    have_contact_time_window_ = true;
  }
  const double dt_ns =
      (time_ns > last_contact_time_ns_) ? (time_ns - last_contact_time_ns_) : 0.0;

  auto emit_contact_trace = [&](double row_time_ns, const std::string &type,
                                int a_id, int b_id, int state,
                                double duration_ns) {
    if (!dynamics_detailed_)
      return;
    if (contact_trace_streaming_ && contact_trace_out_.is_open()) {
      contact_trace_out_ << row_time_ns << "," << type << "," << a_id << ","
                         << b_id << "," << state << "," << duration_ns
                         << "\n";
      return;
    }
    contact_trace_rows_.push_back(
        {row_time_ns, type, 0, a_id, b_id, state, duration_ns});
  };

  auto update_lifetimes =
      [&](const std::vector<uint64_t> &current,
          std::unordered_map<uint64_t, ActiveContactRun> &active,
          std::vector<double> &lifetimes, const std::string &type,
          auto decode_ids) {
        for (uint64_t key : current) {
          auto it = active.find(key);
          if (it == active.end()) {
            active.emplace(key, ActiveContactRun{time_ns, time_ns, frame_idx,
                                                 frame_idx});
            if (dynamics_detailed_) {
              const auto ids = decode_ids(key);
              emit_contact_trace(time_ns, type, ids.first, ids.second, 1, 0.0);
            }
          } else {
            it->second.last_time_ns = time_ns;
            it->second.last_frame_idx = frame_idx;
          }
        }

        for (auto it = active.begin(); it != active.end();) {
          if (std::binary_search(current.begin(), current.end(), it->first)) {
            ++it;
            continue;
          }
          const double duration = std::max(0.0, time_ns - it->second.start_time_ns);
          lifetimes.push_back(duration);
          if (dynamics_detailed_) {
            const auto ids = decode_ids(it->first);
            emit_contact_trace(time_ns, type, ids.first, ids.second, 0,
                               duration);
          }
          it = active.erase(it);
        }
      };

  update_lifetimes(sticker_pairs, active_sticker_contacts_,
                   sticker_contact_lifetimes_ns_, "sticker_pair",
                   [](uint64_t key) { return TitanLLPS::unpack_pair(key); });
  update_lifetimes(
      protein_rna_pairs, active_prna_contacts_, prna_contact_lifetimes_ns_,
      "protein_rna_chain_pair", [&](uint64_t key) {
        const auto ids = unpack_pair(key);
        const int raw_rna =
            (ids.second >= 0 &&
             ids.second < static_cast<int>(rna_chain_raw_ids_.size()))
                ? rna_chain_raw_ids_[static_cast<std::size_t>(ids.second)]
                : ids.second;
        return std::make_pair(ids.first, raw_rna);
      });

  const std::size_t total_chains =
      static_cast<std::size_t>(n_chains_) + rna_chain_raw_ids_.size();
  neighbor_offsets_buf_.assign(total_chains + 1, 0);
  for (uint64_t key : protein_pairs) {
    const auto [a, b] = unpack_pair(key);
    if (a >= 0 && b >= 0 && a < n_chains_ && b < n_chains_) {
      ++neighbor_offsets_buf_[static_cast<std::size_t>(a) + 1];
      ++neighbor_offsets_buf_[static_cast<std::size_t>(b) + 1];
    }
  }
  for (uint64_t key : protein_rna_pairs) {
    const auto [protein_chain, rna_local] = unpack_pair(key);
    const int global_rna =
        n_chains_ + std::max(0, std::min(rna_local,
                                         static_cast<int>(rna_chain_raw_ids_.size()) - 1));
    if (protein_chain >= 0 && protein_chain < n_chains_ &&
        global_rna >= n_chains_ &&
        global_rna < static_cast<int>(total_chains)) {
      ++neighbor_offsets_buf_[static_cast<std::size_t>(protein_chain) + 1];
      ++neighbor_offsets_buf_[static_cast<std::size_t>(global_rna) + 1];
    }
  }
  for (std::size_t idx = 1; idx < neighbor_offsets_buf_.size(); ++idx)
    neighbor_offsets_buf_[idx] += neighbor_offsets_buf_[idx - 1];
  neighbor_write_pos_buf_ = neighbor_offsets_buf_;
  neighbor_flat_buf_.assign(neighbor_offsets_buf_.back(), -1);
  for (uint64_t key : protein_pairs) {
    const auto [a, b] = unpack_pair(key);
    if (a >= 0 && b >= 0 && a < n_chains_ && b < n_chains_) {
      neighbor_flat_buf_[neighbor_write_pos_buf_[static_cast<std::size_t>(a)]++] = b;
      neighbor_flat_buf_[neighbor_write_pos_buf_[static_cast<std::size_t>(b)]++] = a;
    }
  }
  for (uint64_t key : protein_rna_pairs) {
    const auto [protein_chain, rna_local] = unpack_pair(key);
    const int global_rna =
        n_chains_ + std::max(0, std::min(rna_local,
                                         static_cast<int>(rna_chain_raw_ids_.size()) - 1));
    if (protein_chain >= 0 && protein_chain < n_chains_ &&
        global_rna >= n_chains_ &&
        global_rna < static_cast<int>(total_chains)) {
      neighbor_flat_buf_[neighbor_write_pos_buf_[static_cast<std::size_t>(
          protein_chain)]++] = global_rna;
      neighbor_flat_buf_[neighbor_write_pos_buf_[static_cast<std::size_t>(
          global_rna)]++] = protein_chain;
    }
  }
  current_neighbor_offsets_buf_.assign(total_chains + 1, 0);
  current_neighbor_flat_buf_.clear();
  current_neighbor_flat_buf_.reserve(neighbor_flat_buf_.size());
  for (std::size_t idx = 0; idx < total_chains; ++idx) {
    const auto begin = neighbor_offsets_buf_[idx];
    const auto end = neighbor_offsets_buf_[idx + 1];
    if (end > begin) {
      auto span_begin =
          neighbor_flat_buf_.begin() + static_cast<std::ptrdiff_t>(begin);
      auto span_end =
          neighbor_flat_buf_.begin() + static_cast<std::ptrdiff_t>(end);
      std::sort(span_begin, span_end);
      span_end = std::unique(span_begin, span_end);
      current_neighbor_flat_buf_.insert(current_neighbor_flat_buf_.end(),
                                        span_begin, span_end);
    }
    current_neighbor_offsets_buf_[idx + 1] = current_neighbor_flat_buf_.size();
  }

  if (prev_neighbor_offsets_.size() != total_chains + 1) {
    prev_neighbor_offsets_.assign(total_chains + 1, 0);
    prev_neighbor_flat_.clear();
  }
  if (dt_ns > 0.0 && had_previous_frame) {
    NeighborFrameMetric frame_metric;
    frame_metric.time_ns = time_ns;
    frame_metric.dt_ns = dt_ns;
    double protein_sum = 0.0;
    double rna_sum = 0.0;
    double global_sum = 0.0;
    uint64_t protein_count = 0;
    uint64_t rna_count = 0;
    uint64_t global_count = 0;
    uint64_t protein_changes = 0;
    uint64_t rna_changes = 0;
    uint64_t global_changes = 0;

    for (std::size_t idx = 0; idx < total_chains; ++idx) {
      const std::size_t prev_begin = prev_neighbor_offsets_[idx];
      const std::size_t prev_end = prev_neighbor_offsets_[idx + 1];
      const std::size_t curr_begin = current_neighbor_offsets_buf_[idx];
      const std::size_t curr_end = current_neighbor_offsets_buf_[idx + 1];
      std::size_t i = prev_begin;
      std::size_t j = curr_begin;
      std::size_t inter = 0;
      std::size_t uni = 0;
      while (i < prev_end || j < curr_end) {
        if (i < prev_end &&
            (j == curr_end ||
             prev_neighbor_flat_[i] < current_neighbor_flat_buf_[j])) {
          ++uni;
          ++i;
        } else if (j < curr_end &&
            (i == prev_end ||
             current_neighbor_flat_buf_[j] < prev_neighbor_flat_[i])) {
          ++uni;
          ++j;
        } else {
          ++uni;
          ++inter;
          ++i;
          ++j;
        }
      }
      const uint64_t changes =
          static_cast<uint64_t>((prev_end - prev_begin) + (curr_end - curr_begin) -
                                2 * inter);
      const bool is_protein = idx < static_cast<std::size_t>(n_chains_);
      if (uni > 0) {
        const double jaccard = static_cast<double>(inter) / static_cast<double>(uni);
        global_sum += jaccard;
        ++global_count;
        if (is_protein) {
          protein_sum += jaccard;
          ++protein_count;
        } else {
          rna_sum += jaccard;
          ++rna_count;
        }
      }
      global_changes += changes;
      if (is_protein)
        protein_changes += changes;
      else
        rna_changes += changes;
    }

    frame_metric.protein_persistence =
        (protein_count > 0) ? (protein_sum / protein_count) : 0.0;
    frame_metric.rna_persistence =
        (rna_count > 0) ? (rna_sum / rna_count) : 0.0;
    frame_metric.global_persistence =
        (global_count > 0) ? (global_sum / global_count) : 0.0;
    frame_metric.protein_changes = protein_changes;
    frame_metric.rna_changes = rna_changes;
    frame_metric.global_changes = global_changes;
    neighbor_frame_metrics_.push_back(frame_metric);

    if (protein_count > 0) {
      sum_protein_neighbor_persistence_ += protein_sum;
      count_protein_neighbor_persistence_ += protein_count;
    }
    if (rna_count > 0) {
      sum_rna_neighbor_persistence_ += rna_sum;
      count_rna_neighbor_persistence_ += rna_count;
    }
    if (global_count > 0) {
      sum_global_neighbor_persistence_ += global_sum;
      count_global_neighbor_persistence_ += global_count;
    }
    total_protein_neighbor_changes_ += protein_changes;
    total_rna_neighbor_changes_ += rna_changes;
    total_global_neighbor_changes_ += global_changes;
    total_neighbor_dt_ns_ += dt_ns;
  }
  prev_neighbor_offsets_.swap(current_neighbor_offsets_buf_);
  prev_neighbor_flat_.swap(current_neighbor_flat_buf_);

  if (dt_ns > 0.0 && had_previous_frame) {
    InterfaceTurnoverFrame turnover;
    turnover.time_ns = time_ns;
    turnover.dt_ns = dt_ns;
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < prev_prna_contacts_.size() || j < protein_rna_pairs.size()) {
      if (i < prev_prna_contacts_.size() &&
          (j == protein_rna_pairs.size() ||
           prev_prna_contacts_[i] < protein_rna_pairs[j])) {
        ++turnover.lost_contacts;
        ++i;
      } else if (j < protein_rna_pairs.size() &&
                 (i == prev_prna_contacts_.size() ||
                  protein_rna_pairs[j] < prev_prna_contacts_[i])) {
        ++turnover.new_contacts;
        ++j;
      } else {
        ++i;
        ++j;
      }
    }
    total_new_interface_contacts_ += turnover.new_contacts;
    total_lost_interface_contacts_ += turnover.lost_contacts;
    total_interface_dt_ns_ += dt_ns;
    interface_turnover_frames_.push_back(turnover);
  }
  prev_prna_contacts_ = protein_rna_pairs;
  last_contact_time_ns_ = time_ns;
}

void TitanLLPS::finalize_contact_dynamics() {
  if (!dynamics_enabled_)
    return;
  for (const auto &entry : active_sticker_contacts_) {
    sticker_contact_lifetimes_ns_.push_back(
        std::max(0.0, last_contact_time_ns_ - entry.second.start_time_ns));
  }
  for (const auto &entry : active_prna_contacts_) {
    prna_contact_lifetimes_ns_.push_back(
        std::max(0.0, last_contact_time_ns_ - entry.second.start_time_ns));
  }
  active_sticker_contacts_.clear();
  active_prna_contacts_.clear();

  contact_dynamics_result_.sticker_contact_lifetime =
      build_lifetime_distribution(sticker_contact_lifetimes_ns_);
  contact_dynamics_result_.protein_rna_contact_lifetime =
      build_lifetime_distribution(prna_contact_lifetimes_ns_);
  if (count_protein_neighbor_persistence_ > 0) {
    contact_dynamics_result_.protein_neighbor_persistence =
        sum_protein_neighbor_persistence_ /
        static_cast<double>(count_protein_neighbor_persistence_);
  }
  if (count_rna_neighbor_persistence_ > 0) {
    contact_dynamics_result_.rna_neighbor_persistence =
        sum_rna_neighbor_persistence_ /
        static_cast<double>(count_rna_neighbor_persistence_);
  }
  if (count_global_neighbor_persistence_ > 0) {
    contact_dynamics_result_.global_neighbor_persistence =
        sum_global_neighbor_persistence_ /
        static_cast<double>(count_global_neighbor_persistence_);
  }
  if (total_neighbor_dt_ns_ > 0.0) {
    contact_dynamics_result_.protein_rewiring_rate_per_ns =
        static_cast<double>(total_protein_neighbor_changes_) / total_neighbor_dt_ns_;
    contact_dynamics_result_.rna_rewiring_rate_per_ns =
        static_cast<double>(total_rna_neighbor_changes_) / total_neighbor_dt_ns_;
    contact_dynamics_result_.global_rewiring_rate_per_ns =
        static_cast<double>(total_global_neighbor_changes_) / total_neighbor_dt_ns_;
  }
  if (total_interface_dt_ns_ > 0.0) {
    contact_dynamics_result_.interface_new_contacts_per_ns =
        static_cast<double>(total_new_interface_contacts_) / total_interface_dt_ns_;
    contact_dynamics_result_.interface_lost_contacts_per_ns =
        static_cast<double>(total_lost_interface_contacts_) / total_interface_dt_ns_;
    contact_dynamics_result_.interface_net_change_per_ns =
        static_cast<double>(total_new_interface_contacts_) /
            total_interface_dt_ns_ -
        static_cast<double>(total_lost_interface_contacts_) /
            total_interface_dt_ns_;
  }
  contact_dynamics_result_.interface_new_contacts = total_new_interface_contacts_;
  contact_dynamics_result_.interface_lost_contacts = total_lost_interface_contacts_;
}

void TitanLLPS::export_contact_dynamics(const info &info_output,
                                        std::fstream &log) const {
  if (!dynamics_enabled_)
    return;

  auto write_lifetime_file = [](const std::string &filename,
                                const LLPSLifetimeDistributionResult &dist) {
    std::ofstream out(filename);
    if (!out.is_open())
      return false;
    out << "# total_events " << dist.total_events << "\n";
    out << "# mean_ns " << std::fixed << std::setprecision(6) << dist.mean_ns
        << "\n";
    out << "# median_ns " << dist.median_ns << "\n";
    out << "# p90_ns " << dist.p90_ns << "\n";
    out << "# max_ns " << dist.max_ns << "\n";
    out << "# lifetime_ns count\n";
    for (std::size_t i = 0; i < dist.lifetime_ns.size(); ++i)
      out << dist.lifetime_ns[i] << " " << dist.counts[i] << "\n";
    return true;
  };

  if (write_lifetime_file(info_output.get_llps_sticker_contact_lifetimes_filename(),
                          contact_dynamics_result_.sticker_contact_lifetime)) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_sticker_contact_lifetimes_filename());
  }
  if (write_lifetime_file(info_output.get_llps_prna_contact_lifetimes_filename(),
                          contact_dynamics_result_.protein_rna_contact_lifetime)) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_prna_contact_lifetimes_filename());
  }

  {
    std::ofstream out(info_output.get_llps_neighbor_rewiring_filename());
    if (out.is_open()) {
      out << "protein_neighbor_persistence: "
          << contact_dynamics_result_.protein_neighbor_persistence << "\n";
      out << "rna_neighbor_persistence: "
          << contact_dynamics_result_.rna_neighbor_persistence << "\n";
      out << "global_neighbor_persistence: "
          << contact_dynamics_result_.global_neighbor_persistence << "\n";
      out << "protein_rewiring_rate_per_ns: "
          << contact_dynamics_result_.protein_rewiring_rate_per_ns << "\n";
      out << "rna_rewiring_rate_per_ns: "
          << contact_dynamics_result_.rna_rewiring_rate_per_ns << "\n";
      out << "global_rewiring_rate_per_ns: "
          << contact_dynamics_result_.global_rewiring_rate_per_ns << "\n";
      out << "# time_ns dt_ns protein_persistence rna_persistence "
             "global_persistence protein_changes rna_changes global_changes\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : neighbor_frame_metrics_) {
        out << row.time_ns << " " << row.dt_ns << " "
            << row.protein_persistence << " " << row.rna_persistence << " "
            << row.global_persistence << " " << row.protein_changes << " "
            << row.rna_changes << " " << row.global_changes << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_neighbor_rewiring_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_interface_turnover_filename());
    if (out.is_open()) {
      out << "new_contacts_per_ns: "
          << contact_dynamics_result_.interface_new_contacts_per_ns << "\n";
      out << "lost_contacts_per_ns: "
          << contact_dynamics_result_.interface_lost_contacts_per_ns << "\n";
      out << "net_change_per_ns: "
          << contact_dynamics_result_.interface_net_change_per_ns << "\n";
      out << "total_new_contacts: "
          << contact_dynamics_result_.interface_new_contacts << "\n";
      out << "total_lost_contacts: "
          << contact_dynamics_result_.interface_lost_contacts << "\n";
      out << "# time_ns dt_ns new_contacts lost_contacts new_per_ns lost_per_ns "
             "net_per_ns\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : interface_turnover_frames_) {
        const double new_rate =
            (row.dt_ns > 0.0) ? (static_cast<double>(row.new_contacts) / row.dt_ns)
                              : 0.0;
        const double lost_rate =
            (row.dt_ns > 0.0) ? (static_cast<double>(row.lost_contacts) / row.dt_ns)
                              : 0.0;
        out << row.time_ns << " " << row.dt_ns << " " << row.new_contacts
            << " " << row.lost_contacts << " " << new_rate << " " << lost_rate
            << " " << (new_rate - lost_rate) << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_interface_turnover_filename());
    }
  }

  if (dynamics_detailed_) {
    if (contact_trace_streaming_) {
      if (contact_trace_out_.is_open())
        const_cast<std::ofstream &>(contact_trace_out_).flush();
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_chain_contact_trace_filename());
    } else {
    std::ofstream out(info_output.get_llps_chain_contact_trace_filename());
    if (out.is_open()) {
      out << "Time_ns,ContactType,IdA,IdB,State,Duration_ns\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : contact_trace_rows_) {
        out << row.time_ns << "," << row.contact_type << "," << row.a_id << ","
            << row.b_id << "," << row.state << "," << row.duration_ns << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_chain_contact_trace_filename());
    }
    }
  }

  TitanUI::log_kv(log, "LLPS Sticker Lifetime Events",
                  std::to_string(
                      contact_dynamics_result_.sticker_contact_lifetime.total_events));
  TitanUI::log_kv(log, "LLPS Protein-RNA Lifetime Events",
                  std::to_string(contact_dynamics_result_
                                     .protein_rna_contact_lifetime.total_events));
}

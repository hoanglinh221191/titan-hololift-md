// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "BoxMatrix.h"
#include "info.h"
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

void TitanLLPSCluster::finalize_rna_information(TitanExecutor &executor) {
  rna_information_summary_rows_.clear();
  rna_information_per_chain_rows_.clear();
  rna_compaction_state_histogram_rows_.clear();
  rna_partner_state_histogram_rows_.clear();
  rna_transition_information_rows_.clear();
  rna_binding_compaction_information_rows_.clear();
  rna_information_rg_edges_.clear();
  rna_information_longrange_edges_.clear();
  rna_information_primary_scope_ = "none";

  if (!rna_information_enabled_ || rna_compaction_samples_.empty())
    return;

  auto scope_name = [](int scope_index) -> const char * {
    switch (scope_index) {
    case 0:
      return "all_frames";
    case 1:
      return "inside_dense_only";
    default:
      return "contacting_only";
    }
  };
  auto keep_scope = [](const RNACompactionSample &sample, int scope_index) {
    if (scope_index == 1)
      return sample.inside_dense;
    if (scope_index == 2)
      return sample.bound_protein_chains > 0.0;
    return true;
  };

  auto scope_samples = [&](int scope_index) {
    uint64_t count = 0;
    for (const auto &sample : rna_compaction_samples_)
      if (keep_scope(sample, scope_index))
        ++count;
    return count;
  };

  int primary_scope = 1;
  if (scope_samples(primary_scope) == 0) {
    primary_scope = 2;
    if (scope_samples(primary_scope) == 0)
      primary_scope = 0;
  }
  rna_information_primary_scope_ = scope_name(primary_scope);

  double rg_min = std::numeric_limits<double>::infinity();
  double rg_max = -std::numeric_limits<double>::infinity();
  double longrange_min = std::numeric_limits<double>::infinity();
  double longrange_max = -std::numeric_limits<double>::infinity();
  for (const auto &sample : rna_compaction_samples_) {
    if (!keep_scope(sample, primary_scope))
      continue;
    rg_min = std::min(rg_min, sample.rg_nm);
    rg_max = std::max(rg_max, sample.rg_nm);
    longrange_min =
        std::min(longrange_min, sample.longrange_contact_fraction);
    longrange_max =
        std::max(longrange_max, sample.longrange_contact_fraction);
  }
  if (!std::isfinite(rg_min) || !std::isfinite(rg_max)) {
    rg_min = 0.0;
    rg_max = 1.0;
  }
  if (!std::isfinite(longrange_min) || !std::isfinite(longrange_max)) {
    longrange_min = 0.0;
    longrange_max = 1.0;
  }
  rna_information_rg_edges_ =
      make_uniform_edges(rg_min, rg_max, rna_information_rg_bins_);
  rna_information_longrange_edges_ = make_uniform_edges(
      longrange_min, longrange_max, rna_information_longrange_bins_);

  for (auto &sample : rna_compaction_samples_) {
    const int rg_bin =
        locate_uniform_bin(sample.rg_nm, rna_information_rg_edges_);
    const int longrange_bin = locate_uniform_bin(
        sample.longrange_contact_fraction, rna_information_longrange_edges_);
    sample.compaction_state_id =
        rg_bin * rna_information_longrange_bins_ + longrange_bin;
  }

  std::unordered_map<int, std::vector<const RNACompactionSample *>> chain_samples_map;
  chain_samples_map.reserve(rna_chain_compaction_stats_.size() + 1);
  for (const auto &sample : rna_compaction_samples_)
    chain_samples_map[sample.rna_chain_id].push_back(&sample);

  std::vector<int> chain_ids;
  chain_ids.reserve(chain_samples_map.size());
  for (auto &entry : chain_samples_map) {
    std::sort(entry.second.begin(), entry.second.end(),
              [](const RNACompactionSample *a, const RNACompactionSample *b) {
                if (a->frame != b->frame)
                  return a->frame < b->frame;
                return a->time_ns < b->time_ns;
              });
    chain_ids.push_back(entry.first);
  }
  std::sort(chain_ids.begin(), chain_ids.end());

  struct AggregateInfoMetrics {
    uint64_t sample_count{0};
    uint64_t contacting_samples{0};
    uint64_t transition_count{0};
    double total_transition_time_ns{0.0};
    double total_transition_bits{0.0};
    std::unordered_map<int, uint64_t> compaction_counts;
    std::unordered_map<int, uint64_t> binding_counts;
    std::unordered_map<uint64_t, uint64_t> joint_counts;
    std::unordered_map<int, uint64_t> partner_counts;
    std::unordered_map<int, std::unordered_map<int, uint64_t>>
        partner_counts_by_binding;
    std::unordered_map<uint64_t, uint64_t> transition_counts;
    std::unordered_map<int, uint64_t> transition_row_totals;
  };

  auto collect_metrics =
      [&](const std::vector<const RNACompactionSample *> &samples,
          int scope_index) {
        AggregateInfoMetrics metrics;
        metrics.compaction_counts.reserve(samples.size());
        metrics.binding_counts.reserve(samples.size());
        metrics.joint_counts.reserve(samples.size() * 2 + 1);
        metrics.partner_counts.reserve(samples.size());
        metrics.transition_counts.reserve(samples.size());
        metrics.transition_row_totals.reserve(samples.size());
        std::unordered_map<int, const RNACompactionSample *> prev_by_chain;
        prev_by_chain.reserve(8);

        for (const RNACompactionSample *sample : samples) {
          if (sample == nullptr || !keep_scope(*sample, scope_index))
            continue;
          ++metrics.sample_count;
          const int binding_value =
              static_cast<int>(std::lround(sample->bound_protein_chains));
          ++metrics.compaction_counts[sample->compaction_state_id];
          ++metrics.binding_counts[binding_value];
          ++metrics.joint_counts[pack_u32_pair(binding_value,
                                               sample->compaction_state_id)];

          if (binding_value > 0) {
            ++metrics.contacting_samples;
            ++metrics.partner_counts[sample->partner_state_id];
            ++metrics.partner_counts_by_binding[binding_value]
                                      [sample->partner_state_id];
          }

          const RNACompactionSample *prev = nullptr;
          const auto prev_it = prev_by_chain.find(sample->rna_chain_id);
          if (prev_it != prev_by_chain.end())
            prev = prev_it->second;
          if (prev != nullptr && sample->frame == prev->frame + 1) {
            const uint64_t transition_key = pack_u32_pair(
                prev->compaction_state_id, sample->compaction_state_id);
            ++metrics.transition_counts[transition_key];
            ++metrics.transition_row_totals[prev->compaction_state_id];
            ++metrics.transition_count;
            const double dt_ns = sample->time_ns - prev->time_ns;
            if (dt_ns > 0.0)
              metrics.total_transition_time_ns += dt_ns;
          }
          prev_by_chain[sample->rna_chain_id] = sample;
        }

        for (const auto &entry : metrics.transition_counts) {
          const int from_state = static_cast<int>(entry.first >> 32);
          const auto row_total_it = metrics.transition_row_totals.find(from_state);
          if (row_total_it == metrics.transition_row_totals.end() ||
              row_total_it->second == 0)
            continue;
          const double p =
              static_cast<double>(entry.second) /
              static_cast<double>(row_total_it->second);
          if (p > 0.0)
            metrics.total_transition_bits +=
                static_cast<double>(entry.second) * (-std::log2(p));
        }

        return metrics;
      };

  auto append_scope_outputs = [&](const std::string &scope,
                                  const AggregateInfoMetrics &metrics) {
    RNAInformationSummaryRow row;
    row.scope = scope;
    row.samples = metrics.sample_count;
    row.transitions = metrics.transition_count;
    const double h_compaction =
        entropy_bits_from_counts(metrics.compaction_counts, metrics.sample_count);
    const double h_binding =
        entropy_bits_from_counts(metrics.binding_counts, metrics.sample_count);
    const double h_partner =
        entropy_bits_from_counts(metrics.partner_counts, metrics.contacting_samples);
    row.h_compaction_bits = h_compaction;
    row.h_compaction_norm =
        normalized_entropy_bits(h_compaction, metrics.compaction_counts.size());
    row.h_partner_bits = h_partner;
    row.h_partner_norm =
        normalized_entropy_bits(h_partner, metrics.partner_counts.size());
    row.h_partner_given_binding_bits = partner_entropy_given_binding_bits(
        metrics.partner_counts_by_binding, metrics.contacting_samples);
    row.h_compaction_given_binding_bits = conditional_entropy_bits(
        metrics.binding_counts, metrics.joint_counts, metrics.sample_count);
    row.mi_binding_compaction_bits = mutual_information_bits(
        metrics.binding_counts, metrics.compaction_counts, metrics.joint_counts,
        metrics.sample_count);
    const double mi_denom = std::min(h_binding, h_compaction);
    row.mi_binding_compaction_norm =
        (mi_denom > 0.0) ? (row.mi_binding_compaction_bits / mi_denom) : 0.0;
    row.transition_entropy_rate_bits_per_step =
        (metrics.transition_count > 0)
            ? (metrics.total_transition_bits /
               static_cast<double>(metrics.transition_count))
            : 0.0;
    row.transition_entropy_rate_bits_per_ns =
        (metrics.total_transition_time_ns > 0.0)
            ? (metrics.total_transition_bits / metrics.total_transition_time_ns)
            : 0.0;
    rna_information_summary_rows_.push_back(row);

    std::vector<int> compaction_states;
    compaction_states.reserve(metrics.compaction_counts.size());
    for (const auto &entry : metrics.compaction_counts)
      compaction_states.push_back(entry.first);
    std::sort(compaction_states.begin(), compaction_states.end());
    for (int state_id : compaction_states) {
      const int rg_bin = state_id / rna_information_longrange_bins_;
      const int longrange_bin = state_id % rna_information_longrange_bins_;
      if (rg_bin < 0 || longrange_bin < 0 ||
          static_cast<std::size_t>(rg_bin + 1) >= rna_information_rg_edges_.size() ||
          static_cast<std::size_t>(longrange_bin + 1) >=
              rna_information_longrange_edges_.size())
        continue;
      RNACompactionStateHistogramRow hist_row;
      hist_row.scope = scope;
      hist_row.state_id = state_id;
      hist_row.frame_count = metrics.compaction_counts.at(state_id);
      hist_row.probability =
          (metrics.sample_count > 0)
              ? static_cast<double>(hist_row.frame_count) /
                    static_cast<double>(metrics.sample_count)
              : 0.0;
      hist_row.rg_bin = rg_bin;
      hist_row.rg_min_nm =
          rna_information_rg_edges_[static_cast<std::size_t>(rg_bin)];
      hist_row.rg_max_nm =
          rna_information_rg_edges_[static_cast<std::size_t>(rg_bin + 1)];
      hist_row.longrange_bin = longrange_bin;
      hist_row.longrange_min =
          rna_information_longrange_edges_[static_cast<std::size_t>(longrange_bin)];
      hist_row.longrange_max = rna_information_longrange_edges_[static_cast<std::size_t>(
          longrange_bin + 1)];
      rna_compaction_state_histogram_rows_.push_back(hist_row);
    }

    std::vector<int> binding_values;
    binding_values.reserve(metrics.partner_counts_by_binding.size());
    for (const auto &entry : metrics.partner_counts_by_binding)
      binding_values.push_back(entry.first);
    std::sort(binding_values.begin(), binding_values.end());
    for (int binding_value : binding_values) {
      const auto binding_it = metrics.partner_counts_by_binding.find(binding_value);
      if (binding_it == metrics.partner_counts_by_binding.end())
        continue;
      uint64_t binding_total = 0;
      for (const auto &partner_entry : binding_it->second)
        binding_total += partner_entry.second;
      std::vector<int> partner_ids;
      partner_ids.reserve(binding_it->second.size());
      for (const auto &partner_entry : binding_it->second)
        partner_ids.push_back(partner_entry.first);
      std::sort(partner_ids.begin(), partner_ids.end());
      for (int partner_id : partner_ids) {
        RNAPartnerStateHistogramRow partner_row;
        partner_row.scope = scope;
        partner_row.bound_protein_chains = binding_value;
        partner_row.partner_state_id = partner_id;
        partner_row.partner_set =
            (partner_id >= 0 &&
             static_cast<std::size_t>(partner_id) < rna_partner_state_labels_.size())
                ? rna_partner_state_labels_[static_cast<std::size_t>(partner_id)]
                : "unknown";
        partner_row.frame_count = binding_it->second.at(partner_id);
        partner_row.probability =
            (metrics.contacting_samples > 0)
                ? static_cast<double>(partner_row.frame_count) /
                      static_cast<double>(metrics.contacting_samples)
                : 0.0;
        partner_row.conditional_probability =
            (binding_total > 0)
                ? static_cast<double>(partner_row.frame_count) /
                      static_cast<double>(binding_total)
                : 0.0;
        rna_partner_state_histogram_rows_.push_back(partner_row);
      }
    }

    std::vector<std::pair<int, int>> transitions;
    transitions.reserve(metrics.transition_counts.size());
    for (const auto &entry : metrics.transition_counts)
      transitions.emplace_back(static_cast<int>(entry.first >> 32),
                               static_cast<int>(entry.first & 0xffffffffULL));
    std::sort(transitions.begin(), transitions.end());
    for (const auto &transition : transitions) {
      const int from_state = transition.first;
      const int to_state = transition.second;
      const uint64_t transition_key = pack_u32_pair(from_state, to_state);
      const auto count_it = metrics.transition_counts.find(transition_key);
      if (count_it == metrics.transition_counts.end())
        continue;
      const auto row_total_it = metrics.transition_row_totals.find(from_state);
      RNATransitionInformationRow transition_row;
      transition_row.scope = scope;
      transition_row.from_state_id = from_state;
      transition_row.to_state_id = to_state;
      transition_row.count = count_it->second;
      transition_row.probability =
          (row_total_it != metrics.transition_row_totals.end() &&
           row_total_it->second > 0)
              ? static_cast<double>(count_it->second) /
                    static_cast<double>(row_total_it->second)
              : 0.0;
      rna_transition_information_rows_.push_back(transition_row);
    }

    std::vector<std::pair<int, int>> joint_states;
    joint_states.reserve(metrics.joint_counts.size());
    for (const auto &entry : metrics.joint_counts)
      joint_states.emplace_back(static_cast<int>(entry.first >> 32),
                                static_cast<int>(entry.first & 0xffffffffULL));
    std::sort(joint_states.begin(), joint_states.end());
    for (const auto &joint_state : joint_states) {
      const int binding_value = joint_state.first;
      const int state_id = joint_state.second;
      const uint64_t joint_key = pack_u32_pair(binding_value, state_id);
      const auto joint_it = metrics.joint_counts.find(joint_key);
      if (joint_it == metrics.joint_counts.end())
        continue;
      RNABindingCompactionInformationRow info_row;
      info_row.scope = scope;
      info_row.binding_value = binding_value;
      info_row.compaction_state_id = state_id;
      info_row.count = joint_it->second;
      info_row.joint_probability =
          (metrics.sample_count > 0)
              ? static_cast<double>(joint_it->second) /
                    static_cast<double>(metrics.sample_count)
              : 0.0;
      const auto binding_it = metrics.binding_counts.find(binding_value);
      info_row.binding_probability =
          (binding_it != metrics.binding_counts.end() && metrics.sample_count > 0)
              ? static_cast<double>(binding_it->second) /
                    static_cast<double>(metrics.sample_count)
              : 0.0;
      const auto compaction_it = metrics.compaction_counts.find(state_id);
      info_row.compaction_probability =
          (compaction_it != metrics.compaction_counts.end() &&
           metrics.sample_count > 0)
              ? static_cast<double>(compaction_it->second) /
                    static_cast<double>(metrics.sample_count)
              : 0.0;
      rna_binding_compaction_information_rows_.push_back(info_row);
    }
  };

  std::vector<const RNACompactionSample *> all_samples;
  all_samples.reserve(rna_compaction_samples_.size());
  for (const auto &sample : rna_compaction_samples_)
    all_samples.push_back(&sample);
  for (int scope_index = 0; scope_index < 3; ++scope_index) {
    append_scope_outputs(scope_name(scope_index),
                         collect_metrics(all_samples, scope_index));
  }

  const std::size_t total_items = chain_ids.size() * 3ULL;
  const std::size_t workers =
      std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(),
                                                     std::max<std::size_t>(1, total_items)));
  std::vector<std::vector<RNAInformationPerChainRow>> worker_rows(workers);
  executor.parallel_do(workers, [&](std::size_t worker) {
    const std::size_t begin = worker * total_items / workers;
    const std::size_t end = (worker + 1) * total_items / workers;
    auto &rows = worker_rows[worker];
    rows.reserve(end > begin ? end - begin : 0);
    for (std::size_t linear = begin; linear < end; ++linear) {
      const std::size_t chain_pos = linear / 3ULL;
      const int scope_index = static_cast<int>(linear % 3ULL);
      if (chain_pos >= chain_ids.size())
        continue;
      const int chain_id = chain_ids[chain_pos];
      const auto hit = chain_samples_map.find(chain_id);
      if (hit == chain_samples_map.end())
        continue;
      const auto metrics = collect_metrics(hit->second, scope_index);
      if (metrics.sample_count == 0)
        continue;
      RNAInformationPerChainRow row;
      row.scope = scope_name(scope_index);
      row.rna_chain_id = chain_id;
      row.frames = metrics.sample_count;
      row.transitions = metrics.transition_count;
      const double h_compaction = entropy_bits_from_counts(metrics.compaction_counts,
                                                           metrics.sample_count);
      row.h_compaction_bits = h_compaction;
      row.h_partner_bits = entropy_bits_from_counts(metrics.partner_counts,
                                                    metrics.contacting_samples);
      row.h_partner_given_binding_bits = partner_entropy_given_binding_bits(
          metrics.partner_counts_by_binding, metrics.contacting_samples);
      row.h_compaction_given_binding_bits = conditional_entropy_bits(
          metrics.binding_counts, metrics.joint_counts, metrics.sample_count);
      row.mi_binding_compaction_bits = mutual_information_bits(
          metrics.binding_counts, metrics.compaction_counts, metrics.joint_counts,
          metrics.sample_count);
      row.transition_entropy_rate_bits_per_step =
          (metrics.transition_count > 0)
              ? (metrics.total_transition_bits /
                 static_cast<double>(metrics.transition_count))
              : 0.0;
      row.transition_entropy_rate_bits_per_ns =
          (metrics.total_transition_time_ns > 0.0)
              ? (metrics.total_transition_bits /
                 metrics.total_transition_time_ns)
              : 0.0;
      rows.push_back(row);
    }
  });

  for (const auto &rows : worker_rows)
    rna_information_per_chain_rows_.insert(rna_information_per_chain_rows_.end(),
                                           rows.begin(), rows.end());
  std::sort(rna_information_per_chain_rows_.begin(),
            rna_information_per_chain_rows_.end(),
            [](const RNAInformationPerChainRow &a,
               const RNAInformationPerChainRow &b) {
              if (a.scope != b.scope)
                return a.scope < b.scope;
              return a.rna_chain_id < b.rna_chain_id;
            });
}

void TitanLLPSCluster::export_rna_information_outputs(
    const info &info_output) const {
  if (!rna_information_enabled_)
    return;

  auto format_value = [](double value) {
    if (!std::isfinite(value))
      return std::string("NA");
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
  };

  {
    std::ofstream out(info_output.get_llps_rna_information_summary_filename());
    if (out.is_open()) {
      out << "# RNA information-theoretic entropy proxies from coarse-grained "
             "RNA compaction/binding states.\n";
      out << "# log base is log2; these are configurational/dynamic proxies, "
             "not thermodynamic DeltaS.\n";
      out << "primary_scope: " << rna_information_primary_scope_ << "\n";
      out << "rg_bins: " << rna_information_rg_bins_ << "\n";
      out << "longrange_bins: " << rna_information_longrange_bins_ << "\n";
      if (rna_information_rg_edges_.size() >= 2) {
        out << "rg_min_nm: "
            << format_value(rna_information_rg_edges_.front()) << "\n";
        out << "rg_max_nm: "
            << format_value(rna_information_rg_edges_.back()) << "\n";
      }
      if (rna_information_longrange_edges_.size() >= 2) {
        out << "longrange_min: "
            << format_value(rna_information_longrange_edges_.front()) << "\n";
        out << "longrange_max: "
            << format_value(rna_information_longrange_edges_.back()) << "\n";
      }
      for (const auto &row : rna_information_summary_rows_) {
        out << row.scope << "_samples: " << row.samples << "\n";
        out << row.scope << "_transitions: " << row.transitions << "\n";
        out << row.scope << "_H_compaction_bits: "
            << format_value(row.h_compaction_bits) << "\n";
        out << row.scope << "_H_compaction_norm: "
            << format_value(row.h_compaction_norm) << "\n";
        out << row.scope << "_H_partner_bits: "
            << format_value(row.h_partner_bits) << "\n";
        out << row.scope << "_H_partner_norm: "
            << format_value(row.h_partner_norm) << "\n";
        out << row.scope << "_H_partner_given_binding_bits: "
            << format_value(row.h_partner_given_binding_bits) << "\n";
        out << row.scope << "_MI_binding_compaction_bits: "
            << format_value(row.mi_binding_compaction_bits) << "\n";
        out << row.scope << "_MI_binding_compaction_norm: "
            << format_value(row.mi_binding_compaction_norm) << "\n";
        out << row.scope << "_H_compaction_given_binding_bits: "
            << format_value(row.h_compaction_given_binding_bits) << "\n";
        out << row.scope << "_transition_entropy_rate_bits_per_step: "
            << format_value(row.transition_entropy_rate_bits_per_step) << "\n";
        out << row.scope << "_transition_entropy_rate_bits_per_ns: "
            << format_value(row.transition_entropy_rate_bits_per_ns) << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_information_summary_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_rna_information_per_chain_filename());
    if (out.is_open()) {
      out << "Scope,RNAChainID,Frames,Transitions,HCompaction_bits,"
             "HPartner_bits,HPartnerGivenBinding_bits,"
             "MI_BindingCompaction_bits,HCompactionGivenBinding_bits,"
             "TransitionEntropyRate_bits_per_step,"
             "TransitionEntropyRate_bits_per_ns\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_information_per_chain_rows_) {
        out << row.scope << "," << row.rna_chain_id << "," << row.frames << ","
            << row.transitions << "," << row.h_compaction_bits << ","
            << row.h_partner_bits << "," << row.h_partner_given_binding_bits
            << "," << row.mi_binding_compaction_bits << ","
            << row.h_compaction_given_binding_bits << ","
            << row.transition_entropy_rate_bits_per_step << ","
            << row.transition_entropy_rate_bits_per_ns << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_information_per_chain_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_compaction_state_histogram_filename());
    if (out.is_open()) {
      out << "Scope,StateID,FrameCount,Probability,RgBin,RgMin_nm,RgMax_nm,"
             "LongRangeBin,LongRangeMin,LongRangeMax\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_compaction_state_histogram_rows_) {
        out << row.scope << "," << row.state_id << "," << row.frame_count << ","
            << row.probability << "," << row.rg_bin << "," << row.rg_min_nm
            << "," << row.rg_max_nm << "," << row.longrange_bin << ","
            << row.longrange_min << "," << row.longrange_max << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_compaction_state_histogram_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_partner_state_histogram_filename());
    if (out.is_open()) {
      out << "Scope,BoundProteinChains,PartnerStateID,PartnerSet,FrameCount,"
             "Probability,ConditionalProbability\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_partner_state_histogram_rows_) {
        out << row.scope << "," << row.bound_protein_chains << ","
            << row.partner_state_id << "," << row.partner_set << ","
            << row.frame_count << "," << row.probability << ","
            << row.conditional_probability << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_partner_state_histogram_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_transition_information_filename());
    if (out.is_open()) {
      out << "Scope,FromStateID,ToStateID,Count,Probability\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_transition_information_rows_) {
        out << row.scope << "," << row.from_state_id << "," << row.to_state_id
            << "," << row.count << "," << row.probability << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_transition_information_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_binding_compaction_information_filename());
    if (out.is_open()) {
      out << "Scope,BindingValue,CompactionStateID,Count,JointProbability,"
             "BindingProbability,CompactionProbability\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_binding_compaction_information_rows_) {
        out << row.scope << "," << row.binding_value << ","
            << row.compaction_state_id << "," << row.count << ","
            << row.joint_probability << "," << row.binding_probability << ","
            << row.compaction_probability << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_binding_compaction_information_filename());
    }
  }

  if (rna_information_detailed_) {
    std::ofstream out(info_output.get_llps_rna_partner_state_trace_filename());
    if (out.is_open()) {
      out << "Time_ns,Frame,RNAChainID,InsideDense,ClusterID,BoundProteinChains,"
             "PartnerStateID,PartnerSet,CompactionStateID,Rg_nm,"
             "LongRangeContactFraction\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &sample : rna_compaction_samples_) {
        const std::string partner_set =
            (sample.partner_state_id >= 0 &&
             static_cast<std::size_t>(sample.partner_state_id) <
                 rna_partner_state_labels_.size())
                ? rna_partner_state_labels_[static_cast<std::size_t>(
                      sample.partner_state_id)]
                : "unknown";
        out << sample.time_ns << "," << sample.frame << ","
            << sample.rna_chain_id << "," << (sample.inside_dense ? 1 : 0)
            << "," << sample.cluster_id << ","
            << static_cast<int>(std::lround(sample.bound_protein_chains)) << ","
            << sample.partner_state_id << "," << partner_set << ","
            << sample.compaction_state_id << "," << sample.rg_nm << ","
            << sample.longrange_contact_fraction << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_partner_state_trace_filename());
    }
  }
}

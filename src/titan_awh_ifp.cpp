#include "titan_awh_ifp.h"

#include "parse_xvg.h"
#include "titan_dham.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace {

std::string format_double(double value, int precision = 4) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << value;
  return oss.str();
}

double safe_fraction(std::size_t numerator, std::size_t denominator) {
  if (denominator == 0)
    return 0.0;
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

constexpr std::size_t kTopContactsPerInteraction = 3;
constexpr double kImbalancedDominanceThreshold = 0.75;

} // namespace

bool TitanAwhIfp::initialize(const arguments &args, const info &info_output,
                             size_t expected_walkers) {
  walkers_.clear();
  walker_diagnostics_.clear();
  bins_.clear();
  bin_edges_.clear();
  matched_frames_ = 0;
  unmatched_frames_ = 0;
  initialized_ = false;

  const auto xvg_paths = TitanDHAM::read_file_list(args.awh.xvg_list_file);
  if (xvg_paths.empty()) {
    TitanUI::print_error("AWH: no RC .xvg files were read from " +
                         args.awh.xvg_list_file + ".");
    return false;
  }

  if (expected_walkers > 0 && xvg_paths.size() != expected_walkers) {
    TitanUI::print_error(
        "AWH: number of RC .xvg files (" + std::to_string(xvg_paths.size()) +
        ") does not match the number of active walkers (" +
        std::to_string(expected_walkers) + ").");
    return false;
  }

  walkers_.resize(xvg_paths.size());
  walker_diagnostics_.resize(xvg_paths.size());

  double global_min = std::numeric_limits<double>::infinity();
  double global_max = -std::numeric_limits<double>::infinity();

  for (size_t walker_idx = 0; walker_idx < xvg_paths.size(); ++walker_idx) {
    auto &diag = walker_diagnostics_[walker_idx];
    diag.xvg_path = xvg_paths[walker_idx];
    auto parsed = TitanIO::parse_xvg_time_value(
        xvg_paths[walker_idx], 0,
        static_cast<size_t>(args.awh.xvg_value_column));
    if (!parsed) {
      TitanUI::print_error("AWH: " + parsed.error());
      return false;
    }

    auto &walker = walkers_[walker_idx];
    walker.samples.reserve(parsed->size());
    for (const auto &[time_ps, rc_value] : parsed.value()) {
      if (!std::isfinite(time_ps) || !std::isfinite(rc_value))
        continue;
      if (!include_time_sample(args, time_ps))
        continue;
      walker.samples.push_back({time_ps, rc_value});
      global_min = std::min(global_min, rc_value);
      global_max = std::max(global_max, rc_value);
    }

    if (walker.samples.empty()) {
      TitanUI::print_error("AWH: walker " + std::to_string(walker_idx) +
                           " has no usable RC samples after applying the "
                           "current time window.");
      return false;
    }

    walker.tolerance_ps = infer_tolerance_ps(walker.samples);
    diag.rc_samples = walker.samples.size();
    diag.rc_min = walker.samples.front().rc_value;
    diag.rc_max = walker.samples.front().rc_value;
    diag.time_min_ps = walker.samples.front().time_ps;
    diag.time_max_ps = walker.samples.back().time_ps;
    diag.tolerance_ps = walker.tolerance_ps;
    for (const auto &sample : walker.samples) {
      diag.rc_min = std::min(diag.rc_min, sample.rc_value);
      diag.rc_max = std::max(diag.rc_max, sample.rc_value);
    }
  }

  if (!std::isfinite(global_min) || !std::isfinite(global_max)) {
    TitanUI::print_error(
        "AWH: no finite RC values were found after loading the .xvg files.");
    return false;
  }

  rc_min_ = global_min;
  rc_max_ = global_max;
  if (std::abs(rc_max_ - rc_min_) < 1e-12) {
    rc_max_ = rc_min_ + 1e-6;
  }

  bin_width_ = (rc_max_ - rc_min_) / static_cast<double>(args.awh.n_bins);
  bin_edges_.resize(static_cast<size_t>(args.awh.n_bins) + 1);
  for (size_t i = 0; i < bin_edges_.size(); ++i) {
    bin_edges_[i] = rc_min_ + static_cast<double>(i) * bin_width_;
  }
  bin_edges_.back() = rc_max_;

  bins_.assign(static_cast<size_t>(args.awh.n_bins), BinData{});
  for (auto &bin : bins_) {
    bin.walker_frame_counts.assign(walkers_.size(), 0);
  }
  representative_prefix_ = args.awh.output_prefix.empty()
                               ? info_output.get_awh_rep_pdb_prefix()
                               : args.awh.output_prefix;
  initialized_ = true;

  TitanUI::print_info("AWH: initialized RC-binned IFP with " +
                      std::to_string(walkers_.size()) + " walker(s), RC range [" +
                      format_double(rc_min_) + ", " + format_double(rc_max_) +
                      "], bins = " + std::to_string(args.awh.n_bins) + ".");
  return true;
}

TitanAwhIfp::AwhFrameContext
TitanAwhIfp::prepare_frame(int replica_id, double time_ps,
                           std::int64_t global_frame_index) {
  AwhFrameContext frame_ctx;
  frame_ctx.replica_id = replica_id;
  frame_ctx.time_ps = time_ps;
  frame_ctx.global_frame_index = global_frame_index;
  if (!initialized_)
    return frame_ctx;

  const bool valid_replica =
      replica_id >= 0 && static_cast<size_t>(replica_id) < walkers_.size();
  double rc_value = 0.0;
  if (!lookup_rc_value(replica_id, time_ps, rc_value)) {
    ++unmatched_frames_;
    if (valid_replica)
      ++walker_diagnostics_[static_cast<size_t>(replica_id)].unmatched_frames;
    return frame_ctx;
  }

  ++matched_frames_;
  if (valid_replica)
    ++walker_diagnostics_[static_cast<size_t>(replica_id)].matched_frames;
  const size_t bin_idx = assign_bin(rc_value);
  auto &bin = bins_[bin_idx];
  ++bin.frame_count;
  if (valid_replica && static_cast<size_t>(replica_id) < bin.walker_frame_counts.size())
    ++bin.walker_frame_counts[static_cast<size_t>(replica_id)];
  bin.matched_frames.push_back({replica_id, global_frame_index, time_ps, rc_value});
  frame_ctx.matched = true;
  frame_ctx.rc_value = rc_value;
  frame_ctx.bin_idx = bin_idx;
  frame_ctx.delta_to_center = std::abs(rc_value - bin_center(bin_idx));
  return frame_ctx;
}

void TitanAwhIfp::consume_frame(
    const AwhFrameContext &frame_ctx,
    const std::vector<TitanIFP::ContactCountAccumulator> &worker_accumulators,
    size_t active_workers, const MOL &rec_mol, const MOL &lig_mol) {
  if (!initialized_ || !frame_ctx.matched)
    return;

  auto &bin = bins_[frame_ctx.bin_idx];
  const size_t reduce_workers =
      std::min(active_workers, worker_accumulators.size());
  for (size_t worker_id = 0; worker_id < reduce_workers; ++worker_id) {
    for (const auto &[key, record] : worker_accumulators[worker_id]) {
      auto &bin_record = bin.contacts[key];
      if (bin_record.res_name_rec.empty()) {
        bin_record.resnr_rec = record.resnr_rec;
        bin_record.resnr_lig = record.resnr_lig;
        bin_record.chainid_rec = record.chainid_rec;
        bin_record.chainid_lig = record.chainid_lig;
        bin_record.res_name_rec = record.res_name_rec;
        bin_record.res_name_lig = record.res_name_lig;
      }
      for (size_t bit = 0; bit < bin_record.counts.size(); ++bit)
        bin_record.counts[bit] += record.counts[bit];
    }
  }

  auto &rep = bin.representative;
  constexpr double kRepresentativeTieTolerance = 1e-12;
  if (!rep.valid || frame_ctx.delta_to_center + kRepresentativeTieTolerance <
                        rep.delta_to_center) {
    rep.valid = true;
    rep.global_frame_index = frame_ctx.global_frame_index;
    rep.replica_id = frame_ctx.replica_id;
    rep.time_ps = frame_ctx.time_ps;
    rep.rc_value = frame_ctx.rc_value;
    rep.delta_to_center = frame_ctx.delta_to_center;
    rep.rec_frame = rec_mol;
    rep.lig_frame = lig_mol;
  }
}

void TitanAwhIfp::export_results(const arguments &args, const info &info_output,
                                 const std::vector<mol_info> &rec_info,
                                 const std::vector<mol_info> &lig_info) const {
  if (!initialized_)
    return;

  std::ofstream summary(info_output.get_awh_rc_summary_filename());
  std::ofstream by_bin(info_output.get_awh_ifp_by_bin_filename());
  std::ofstream heatmap(info_output.get_awh_ifp_heatmap_filename());
  std::ofstream structural_summary(
      info_output.get_awh_bin_structural_summary_filename());
  std::ofstream top_contacts(
      info_output.get_awh_bin_top_contacts_filename());
  std::ofstream stability_by_bin(
      info_output.get_awh_walker_stability_by_bin_filename());
  std::ofstream pair_overlap(
      info_output.get_awh_walker_pair_overlap_filename());
  std::ofstream frame_map(info_output.get_awh_rc_frame_map_filename());
  std::ofstream frame_assignments(info_output.get_awh_frame_assignments_filename());
  std::ofstream walker_diag(
      info_output.get_awh_walker_diagnostics_filename());
  std::ofstream overlap_diag(info_output.get_awh_rc_overlap_filename());

  if (!summary.is_open() || !by_bin.is_open() || !heatmap.is_open() ||
      !structural_summary.is_open() || !top_contacts.is_open() ||
      !stability_by_bin.is_open() || !pair_overlap.is_open() ||
      !frame_map.is_open() || !frame_assignments.is_open() || !walker_diag.is_open() || !overlap_diag.is_open()) {
    TitanUI::print_error(
        "AWH: failed to open one or more output files for RC-binned IFP.");
    return;
  }

  size_t empty_bins = 0;
  size_t occupied_bins = 0;
  size_t single_walker_bins = 0;
  size_t multiwalker_overlap_bins = 0;
  size_t multiwalker_balanced_bins = 0;
  size_t multiwalker_imbalanced_bins = 0;
  for (const auto &bin : bins_) {
    if (bin.frame_count == 0) {
      ++empty_bins;
      continue;
    }
    ++occupied_bins;
    const size_t active_walkers =
        std::count_if(bin.walker_frame_counts.begin(),
                      bin.walker_frame_counts.end(),
                      [](size_t value) { return value > 0; });
    if (active_walkers > 1) {
      ++multiwalker_overlap_bins;
      const auto dominant_it =
          std::max_element(bin.walker_frame_counts.begin(),
                           bin.walker_frame_counts.end());
      const size_t dominant_frames =
          (dominant_it != bin.walker_frame_counts.end()) ? *dominant_it : 0;
      const double dominant_fraction =
          safe_fraction(dominant_frames, bin.frame_count);
      if (dominant_fraction >= kImbalancedDominanceThreshold)
        ++multiwalker_imbalanced_bins;
      else
        ++multiwalker_balanced_bins;
    } else if (active_walkers == 1) {
      ++single_walker_bins;
    }
  }

  std::vector<size_t> occupied_bins_by_walker(walker_diagnostics_.size(), 0);
  for (size_t walker_idx = 0; walker_idx < walker_diagnostics_.size();
       ++walker_idx) {
    size_t occupied = 0;
    for (const auto &bin : bins_) {
      if (walker_idx < bin.walker_frame_counts.size() &&
          bin.walker_frame_counts[walker_idx] > 0) {
        ++occupied;
      }
    }
    occupied_bins_by_walker[walker_idx] = occupied;
  }

  summary << "# matched_frames " << matched_frames_ << "\n";
  summary << "# unmatched_frames " << unmatched_frames_ << "\n";
  summary << "# walker_count " << walkers_.size() << "\n";
  summary << "# rc_min " << format_double(rc_min_) << "\n";
  summary << "# rc_max " << format_double(rc_max_) << "\n";
  summary << "# bin_width " << format_double(bin_width_) << "\n";
  summary << "# occupied_bins " << occupied_bins << "\n";
  summary << "# empty_bins " << empty_bins << "\n";
  summary << "# single_walker_bins " << single_walker_bins << "\n";
  summary << "# multiwalker_overlap_bins " << multiwalker_overlap_bins << "\n";
  summary << "# multiwalker_balanced_bins " << multiwalker_balanced_bins
          << "\n";
  summary << "# multiwalker_imbalanced_bins " << multiwalker_imbalanced_bins
          << "\n";
  summary << "bin_id\trc_left\trc_center\trc_right\tframes\treplica_id\t"
             "global_frame_index\ttime_ps\trc_value\tdelta_to_center\n";

  structural_summary
      << "bin_id\trc_left\trc_center\trc_right\tframes\tactive_walkers\t"
         "dominant_walker\tdominant_fraction\trepresentative_replica\t"
         "representative_time_ps\trepresentative_rc_value\t"
         "residue_pairs_total\tcontact_events_total\tcontact_event_load_per_frame\t"
         "HBond_pair_count\tHBond_avg_hits_per_frame\t"
         "SaltBridge_pair_count\tSaltBridge_avg_hits_per_frame\t"
         "PiPi_pair_count\tPiPi_avg_hits_per_frame\t"
         "CationPi_pair_count\tCationPi_avg_hits_per_frame\t"
         "Hydrophobic_pair_count\tHydrophobic_avg_hits_per_frame\t"
         "Halogen_pair_count\tHalogen_avg_hits_per_frame\n";
  structural_summary << std::fixed << std::setprecision(6);

  top_contacts << "bin_id,rc_left,rc_center,rc_right,interaction,rank,"
                  "occupancy_pct,hit_count,chainid_rec,resid_rec,resname_rec,"
                  "chainid_lig,resid_lig,resname_lig,label\n";
  top_contacts << std::fixed << std::setprecision(3);

  stability_by_bin << "bin_id\trc_left\trc_center\trc_right\ttotal_frames\t"
                      "active_walkers\teffective_walkers\tdominant_walker\t"
                      "dominant_fraction\tstability_status";
  for (size_t walker_idx = 0; walker_idx < walkers_.size(); ++walker_idx)
    stability_by_bin << "\twalker_" << walker_idx << "_fraction";
  stability_by_bin << '\n';
  stability_by_bin << std::fixed << std::setprecision(6);

  pair_overlap << "walker_i\twalker_j\toccupied_bins_i\toccupied_bins_j\t"
                  "shared_bins\tunion_bins\tbin_jaccard\tshared_bins_over_i\t"
                  "shared_bins_over_j\tframes_i\tframes_j\tshared_frames\t"
                  "union_frames\tframe_overlap_fraction\tbalanced_shared_bins\t"
                  "imbalanced_shared_bins\n";
  pair_overlap << std::fixed << std::setprecision(6);

  frame_map << "bin_id\treplica_id\ttime_ps\trc_value\tdelta_to_center\tpdb_file\t"
               "rc_left\trc_center\trc_right\tbin_frames\tactive_walkers\t"
               "dominant_walker\tdominant_fraction\toverlap_status\n";

  frame_assignments << "bin_id\treplica_id\tglobal_frame_index\ttime_ps\trc_value\t"
                       "rc_left\trc_center\trc_right\toverlap_status\t"
                       "stability_status\n";

  walker_diag << "walker_id\trc_samples\trc_min\trc_max\ttime_min_ps\t"
                 "time_max_ps\ttolerance_ps\tmatched_frames\tunmatched_frames\t"
                 "matched_fraction\toccupied_bins\txvg_path\n";
  walker_diag << std::fixed << std::setprecision(6);
  for (size_t walker_idx = 0; walker_idx < walker_diagnostics_.size();
       ++walker_idx) {
    const auto &diag = walker_diagnostics_[walker_idx];
    const size_t total_streamed = diag.matched_frames + diag.unmatched_frames;
    walker_diag << walker_idx << '\t' << diag.rc_samples << '\t' << diag.rc_min
                << '\t' << diag.rc_max << '\t' << diag.time_min_ps << '\t'
                << diag.time_max_ps << '\t' << diag.tolerance_ps << '\t'
                << diag.matched_frames << '\t' << diag.unmatched_frames << '\t'
                << safe_fraction(diag.matched_frames, total_streamed) << '\t'
                << occupied_bins_by_walker[walker_idx] << '\t' << diag.xvg_path
                << '\n';
  }

  overlap_diag << "bin_id\trc_left\trc_center\trc_right\ttotal_frames\t"
                  "active_walkers\tdominant_walker\tdominant_fraction\t"
                  "overlap_status";
  for (size_t walker_idx = 0; walker_idx < walkers_.size(); ++walker_idx) {
    overlap_diag << "\twalker_" << walker_idx << "_frames";
  }
  overlap_diag << '\n';
  overlap_diag << std::fixed << std::setprecision(6);

  by_bin << "Bin,RC_Left,RC_Center,RC_Right,Frames,ChainID_Rec,ResID_Rec,"
            "ResName_Rec,ChainID_Lig,ResID_Lig,ResName_Lig,HBond(%),"
            "SaltBridge(%),PiPi(%),CationPi(%),Hydrophobic(%),Halogen(%)\n";
  by_bin << std::fixed << std::setprecision(3);

  std::map<std::string, std::vector<double>> heatmap_rows;

  for (size_t bin_idx = 0; bin_idx < bins_.size(); ++bin_idx) {
    const auto &bin = bins_[bin_idx];
    const double left = bin_left(bin_idx);
    const double center = bin_center(bin_idx);
    const double right = bin_right(bin_idx);
    const auto dominant_it = std::max_element(bin.walker_frame_counts.begin(),
                                              bin.walker_frame_counts.end());
    const size_t dominant_index =
        (dominant_it != bin.walker_frame_counts.end())
            ? static_cast<size_t>(
                  std::distance(bin.walker_frame_counts.begin(), dominant_it))
            : static_cast<size_t>(-1);
    const size_t dominant_frames =
        (dominant_it != bin.walker_frame_counts.end()) ? *dominant_it : 0;
    const size_t active_walkers =
        std::count_if(bin.walker_frame_counts.begin(),
                      bin.walker_frame_counts.end(),
                      [](size_t value) { return value > 0; });
    const std::string overlap_status =
        (bin.frame_count == 0)
            ? "empty"
            : (active_walkers > 1 ? "multiwalker" : "single_walker");
    const double dominant_fraction = safe_fraction(dominant_frames, bin.frame_count);
    double squared_fraction_sum = 0.0;
    for (size_t count : bin.walker_frame_counts) {
      const double fraction = safe_fraction(count, bin.frame_count);
      squared_fraction_sum += fraction * fraction;
    }
    const double effective_walkers =
        (bin.frame_count == 0 || squared_fraction_sum <= 0.0)
            ? 0.0
            : (1.0 / squared_fraction_sum);
    const std::string stability_status =
        (bin.frame_count == 0)
            ? "empty"
            : (active_walkers <= 1
                   ? "single_walker"
                   : (dominant_fraction >= kImbalancedDominanceThreshold
                          ? "multiwalker_imbalanced"
                          : "multiwalker_balanced"));

    summary << bin_idx << '\t' << format_double(left) << '\t'
            << format_double(center) << '\t' << format_double(right) << '\t'
            << bin.frame_count << '\t';
    if (bin.representative.valid) {
      summary << bin.representative.replica_id << '\t'
              << bin.representative.global_frame_index << '\t'
              << format_double(bin.representative.time_ps, 3) << '\t'
              << format_double(bin.representative.rc_value) << '\t'
              << format_double(bin.representative.delta_to_center) << '\n';
    } else {
      summary << "-1\tNA\tNA\tNA\n";
    }

    std::string pdb_name = "NA";
    if (bin.representative.valid) {
      std::ostringstream oss;
      oss << representative_prefix_ << '_' << std::setw(3) << std::setfill('0')
          << bin_idx << ".pdb";
      pdb_name = oss.str();
      write_representative_pdb(pdb_name, bin.representative, rec_info, lig_info);
    }
    frame_map << bin_idx << '\t'
              << (bin.representative.valid ? bin.representative.replica_id : -1)
              << '\t';
    if (bin.representative.valid) {
      frame_map << format_double(bin.representative.time_ps, 3) << '\t'
                << format_double(bin.representative.rc_value) << '\t'
                << format_double(bin.representative.delta_to_center) << '\t'
                << pdb_name << '\t' << format_double(left) << '\t'
                << format_double(center) << '\t' << format_double(right) << '\t'
                << bin.frame_count << '\t' << active_walkers << '\t'
                << static_cast<int>(dominant_index) << '\t'
                << format_double(dominant_fraction, 6) << '\t'
                << overlap_status << '\n';
    } else {
      frame_map << "NA\tNA\tNA\t" << pdb_name << '\t' << format_double(left)
                << '\t' << format_double(center) << '\t'
                << format_double(right) << '\t' << bin.frame_count << '\t'
                << active_walkers << "\t-1\t0.000000\t" << overlap_status
                << '\n';
    }
    for (const auto &matched : bin.matched_frames) {
      frame_assignments << bin_idx << '\t' << matched.replica_id << '\t'
                        << matched.global_frame_index << '\t'
                        << matched.time_ps << '\t' << matched.rc_value << '\t'
                        << left << '\t' << center << '\t' << right << '\t'
                        << overlap_status << '\t' << stability_status << '\n';
    }

    overlap_diag << bin_idx << '\t' << left << '\t' << center << '\t' << right
                  << '\t' << bin.frame_count << '\t' << active_walkers << '\t';
    if (bin.frame_count == 0) {
      overlap_diag << "-1\t0.000000\t" << overlap_status;
    } else {
      overlap_diag << static_cast<int>(dominant_index) << '\t'
                   << dominant_fraction << '\t'
                   << overlap_status;
    }
    for (size_t count : bin.walker_frame_counts)
      overlap_diag << '\t' << count;
    overlap_diag << '\n';

    stability_by_bin << bin_idx << '\t' << left << '\t' << center << '\t'
                     << right << '\t' << bin.frame_count << '\t'
                     << active_walkers << '\t' << effective_walkers << '\t';
    if (bin.frame_count == 0) {
      stability_by_bin << "-1\t0.000000\t" << stability_status;
    } else {
      stability_by_bin << static_cast<int>(dominant_index) << '\t'
                       << dominant_fraction << '\t' << stability_status;
    }
    for (size_t count : bin.walker_frame_counts)
      stability_by_bin << '\t' << safe_fraction(count, bin.frame_count);
    stability_by_bin << '\n';

    std::array<std::size_t, 6> pair_counts{};
    std::array<std::size_t, 6> event_counts{};
    std::array<std::vector<const ContactRecord *>, 6> ranked_contacts{};
    std::size_t residue_pairs_total = 0;
    std::size_t contact_events_total = 0;
    for (const auto &[key, record] : bin.contacts) {
      (void)key;
      ++residue_pairs_total;
      for (int bit = 0; bit < 6; ++bit) {
        const std::size_t count = record.counts[static_cast<std::size_t>(bit)];
        if (count == 0)
          continue;
        ++pair_counts[static_cast<std::size_t>(bit)];
        event_counts[static_cast<std::size_t>(bit)] += count;
        contact_events_total += count;
        ranked_contacts[static_cast<std::size_t>(bit)].push_back(&record);
      }
    }

    structural_summary << bin_idx << '\t' << left << '\t' << center << '\t'
                       << right << '\t' << bin.frame_count << '\t'
                       << active_walkers << '\t';
    if (bin.frame_count == 0) {
      structural_summary << "-1\t0.000000\t-1\tNA\tNA\t0\t0\t0.000000";
      for (int bit = 0; bit < 6; ++bit)
        structural_summary << '\t' << 0 << '\t' << "0.000000";
      structural_summary << '\n';
    } else {
      structural_summary << static_cast<int>(dominant_index) << '\t'
                         << dominant_fraction << '\t'
                         << (bin.representative.valid ? bin.representative.replica_id
                                                      : -1)
                         << '\t';
      if (bin.representative.valid) {
        structural_summary << bin.representative.time_ps << '\t'
                           << bin.representative.rc_value;
      } else {
        structural_summary << "NA\tNA";
      }
      structural_summary << '\t' << residue_pairs_total << '\t'
                         << contact_events_total << '\t'
                         << safe_fraction(contact_events_total, bin.frame_count);
      for (int bit = 0; bit < 6; ++bit) {
        structural_summary
            << '\t' << pair_counts[static_cast<std::size_t>(bit)] << '\t'
            << safe_fraction(event_counts[static_cast<std::size_t>(bit)],
                             bin.frame_count);
      }
      structural_summary << '\n';
    }

    for (int bit = 0; bit < 6; ++bit) {
      auto &records = ranked_contacts[static_cast<std::size_t>(bit)];
      std::sort(records.begin(), records.end(),
                [bit](const ContactRecord *lhs, const ContactRecord *rhs) {
                  const std::size_t lhs_count =
                      lhs->counts[static_cast<std::size_t>(bit)];
                  const std::size_t rhs_count =
                      rhs->counts[static_cast<std::size_t>(bit)];
                  if (lhs_count != rhs_count)
                    return lhs_count > rhs_count;
                  if (lhs->chainid_rec != rhs->chainid_rec)
                    return lhs->chainid_rec < rhs->chainid_rec;
                  if (lhs->resnr_rec != rhs->resnr_rec)
                    return lhs->resnr_rec < rhs->resnr_rec;
                  if (lhs->chainid_lig != rhs->chainid_lig)
                    return lhs->chainid_lig < rhs->chainid_lig;
                  return lhs->resnr_lig < rhs->resnr_lig;
                });
      const std::size_t limit =
          std::min<std::size_t>(records.size(), kTopContactsPerInteraction);
      for (std::size_t rank = 0; rank < limit; ++rank) {
        const ContactRecord &record = *records[rank];
        const std::size_t hit_count = record.counts[static_cast<std::size_t>(bit)];
        top_contacts << bin_idx << ',' << left << ',' << center << ',' << right
                     << ',' << interaction_label(bit) << ',' << (rank + 1) << ','
                     << (bin.frame_count == 0
                             ? 0.0
                             : (100.0 * static_cast<double>(hit_count) /
                                static_cast<double>(bin.frame_count)))
                     << ',' << hit_count << ',' << record.chainid_rec << ','
                     << record.resnr_rec << ',' << record.res_name_rec << ','
                     << record.chainid_lig << ',' << record.resnr_lig << ','
                     << record.res_name_lig << ','
                     << build_contact_label(record, bit) << '\n';
      }
    }

    if (bin.frame_count == 0)
      continue;

    const double inv_frames = 100.0 / static_cast<double>(bin.frame_count);
    for (const auto &[key, record] : bin.contacts) {
      (void)key;
      by_bin << bin_idx << ',' << left << ',' << center << ',' << right << ','
             << bin.frame_count << ',' << record.chainid_rec << ','
             << record.resnr_rec << ',' << record.res_name_rec << ','
             << record.chainid_lig << ',' << record.resnr_lig << ','
             << record.res_name_lig;
      for (int bit = 0; bit < 6; ++bit) {
        const double occupancy =
            static_cast<double>(record.counts[static_cast<size_t>(bit)]) *
            inv_frames;
        by_bin << ',' << occupancy;
        if (record.counts[static_cast<size_t>(bit)] > 0) {
          auto &row = heatmap_rows[build_contact_label(record, bit)];
          if (row.empty())
            row.assign(bins_.size(), 0.0);
          row[bin_idx] = occupancy;
        }
      }
      by_bin << '\n';
    }
  }

  for (size_t i = 0; i < walkers_.size(); ++i) {
    for (size_t j = i + 1; j < walkers_.size(); ++j) {
      size_t shared_bins = 0;
      size_t union_bins = 0;
      size_t shared_frames = 0;
      size_t union_frames = 0;
      size_t balanced_shared_bins = 0;
      size_t imbalanced_shared_bins = 0;
      for (const auto &bin : bins_) {
        const size_t count_i =
            (i < bin.walker_frame_counts.size()) ? bin.walker_frame_counts[i] : 0;
        const size_t count_j =
            (j < bin.walker_frame_counts.size()) ? bin.walker_frame_counts[j] : 0;
        if (count_i > 0 || count_j > 0)
          ++union_bins;
        if (count_i > 0 && count_j > 0) {
          ++shared_bins;
          const size_t active_walkers =
              std::count_if(bin.walker_frame_counts.begin(),
                            bin.walker_frame_counts.end(),
                            [](size_t value) { return value > 0; });
          const auto dominant_it =
              std::max_element(bin.walker_frame_counts.begin(),
                               bin.walker_frame_counts.end());
          const size_t dominant_frames =
              (dominant_it != bin.walker_frame_counts.end()) ? *dominant_it : 0;
          const double dominant_fraction =
              safe_fraction(dominant_frames, bin.frame_count);
          if (active_walkers > 1) {
            if (dominant_fraction >= kImbalancedDominanceThreshold)
              ++imbalanced_shared_bins;
            else
              ++balanced_shared_bins;
          }
        }
        shared_frames += std::min(count_i, count_j);
        union_frames += std::max(count_i, count_j);
      }

      pair_overlap << i << '\t' << j << '\t' << occupied_bins_by_walker[i]
                   << '\t' << occupied_bins_by_walker[j] << '\t' << shared_bins
                   << '\t' << union_bins << '\t'
                   << safe_fraction(shared_bins, union_bins) << '\t'
                   << safe_fraction(shared_bins, occupied_bins_by_walker[i])
                   << '\t'
                   << safe_fraction(shared_bins, occupied_bins_by_walker[j])
                   << '\t' << walker_diagnostics_[i].matched_frames << '\t'
                   << walker_diagnostics_[j].matched_frames << '\t'
                   << shared_frames << '\t' << union_frames << '\t'
                   << safe_fraction(shared_frames, union_frames) << '\t'
                   << balanced_shared_bins << '\t' << imbalanced_shared_bins
                   << '\n';
    }
  }

  heatmap << "label";
  for (size_t bin_idx = 0; bin_idx < bins_.size(); ++bin_idx) {
    heatmap << '\t' << "bin_" << bin_idx;
  }
  heatmap << '\n';
  heatmap << std::fixed << std::setprecision(3);
  for (const auto &[label, values] : heatmap_rows) {
    heatmap << label;
    for (double value : values) {
      heatmap << '\t' << value;
    }
    heatmap << '\n';
  }

  TitanUI::print_info(
      "AWH: exported RC-binned IFP outputs to " +
      info_output.get_awh_rc_summary_filename() + ", " +
      info_output.get_awh_ifp_by_bin_filename() + ", " +
      info_output.get_awh_ifp_heatmap_filename() + ", " +
      info_output.get_awh_bin_structural_summary_filename() + ", " +
      info_output.get_awh_bin_top_contacts_filename() + ", and " +
      info_output.get_awh_rc_frame_map_filename() + ".");
  TitanUI::print_info(
      "AWH: exported walker/bin diagnostics to " +
      info_output.get_awh_walker_diagnostics_filename() + " and " +
      info_output.get_awh_rc_overlap_filename() + ".");
  TitanUI::print_info(
      "AWH: exported walker-consistency diagnostics to " +
      info_output.get_awh_walker_stability_by_bin_filename() + " and " +
      info_output.get_awh_walker_pair_overlap_filename() + ".");

  if (unmatched_frames_ > 0) {
    TitanUI::print_warning("AWH: " + std::to_string(unmatched_frames_) +
                           " streamed frame(s) could not be matched to an RC "
                           "sample.");
  }
  if (empty_bins > 0) {
    TitanUI::print_warning("AWH: " + std::to_string(empty_bins) + "/" +
                           std::to_string(bins_.size()) +
                           " RC bin(s) are empty in the current biased ensemble "
                           "window.");
  }
  if (walkers_.size() > 1 && occupied_bins > 0) {
    if (multiwalker_overlap_bins == 0) {
      TitanUI::print_warning(
          "AWH: occupied RC bins show no multiwalker overlap; structural "
          "interpretation may be walker-specific.");
    } else if (single_walker_bins > 0) {
      TitanUI::print_warning(
          "AWH: " + std::to_string(single_walker_bins) + "/" +
          std::to_string(occupied_bins) +
          " occupied RC bin(s) are supported by only one walker.");
    }
    if (multiwalker_overlap_bins > 0 && multiwalker_balanced_bins == 0) {
      TitanUI::print_warning(
          "AWH: multiwalker overlap exists but all shared RC bins are strongly "
          "imbalanced across walkers.");
    }
  }
}

bool TitanAwhIfp::include_time_sample(const arguments &args, double time_ps) {
  if (args.flag_begin_set && time_ps < args.begin_time - 1e-3)
    return false;
  if (args.flag_end_set && time_ps > args.end_time + 1e-3)
    return false;

  if (args.dt > 0) {
    const double ref_time = args.flag_begin_set ? args.begin_time : 0.0;
    const double diff = time_ps - ref_time;
    if (diff < -1e-3)
      return false;
    const long long steps = std::llround(diff / args.dt);
    if (std::abs(diff - static_cast<double>(steps * args.dt)) > 1e-3) {
      return false;
    }
  }
  return true;
}

double TitanAwhIfp::infer_tolerance_ps(const std::vector<RcSample> &samples) {
  std::vector<double> deltas;
  deltas.reserve(samples.size() > 1 ? samples.size() - 1 : 0);
  for (size_t i = 1; i < samples.size(); ++i) {
    const double delta = samples[i].time_ps - samples[i - 1].time_ps;
    if (std::isfinite(delta) && delta > 0.0)
      deltas.push_back(delta);
  }

  if (deltas.empty())
    return 1e-3;

  const size_t mid = deltas.size() / 2;
  std::nth_element(deltas.begin(), deltas.begin() + static_cast<std::ptrdiff_t>(mid),
                   deltas.end());
  const double median_dt = deltas[mid];
  return 0.5 * median_dt + 1e-6 * std::max(1.0, std::abs(median_dt));
}

bool TitanAwhIfp::lookup_rc_value(int replica_id, double time_ps,
                                  double &rc_value) const {
  if (replica_id < 0 || static_cast<size_t>(replica_id) >= walkers_.size())
    return false;

  const auto &samples = walkers_[static_cast<size_t>(replica_id)].samples;
  if (samples.empty())
    return false;

  auto it = std::lower_bound(
      samples.begin(), samples.end(), time_ps,
      [](const RcSample &sample, double target_time) {
        return sample.time_ps < target_time;
      });

  const RcSample *best = nullptr;
  if (it != samples.end())
    best = &(*it);
  if (it != samples.begin()) {
    const RcSample &prev = *(it - 1);
    if (best == nullptr ||
        std::abs(prev.time_ps - time_ps) < std::abs(best->time_ps - time_ps)) {
      best = &prev;
    }
  }

  if (best == nullptr)
    return false;

  const double tolerance =
      walkers_[static_cast<size_t>(replica_id)].tolerance_ps;
  if (std::abs(best->time_ps - time_ps) > tolerance)
    return false;

  rc_value = best->rc_value;
  return true;
}

size_t TitanAwhIfp::assign_bin(double rc_value) const {
  if (rc_value <= rc_min_)
    return 0;
  if (rc_value >= rc_max_)
    return bins_.size() - 1;
  const double normalized = (rc_value - rc_min_) / bin_width_;
  const size_t bin_idx = static_cast<size_t>(normalized);
  return std::min(bin_idx, bins_.size() - 1);
}

double TitanAwhIfp::bin_left(size_t bin_idx) const { return bin_edges_[bin_idx]; }

double TitanAwhIfp::bin_right(size_t bin_idx) const {
  return bin_edges_[bin_idx + 1];
}

double TitanAwhIfp::bin_center(size_t bin_idx) const {
  return 0.5 * (bin_left(bin_idx) + bin_right(bin_idx));
}

std::string TitanAwhIfp::interaction_label(int bit) {
  switch (bit) {
  case BIT_HBOND:
    return "HBond";
  case BIT_SALT:
    return "SaltBridge";
  case BIT_PI_PI:
    return "PiPi";
  case BIT_CATION_PI:
    return "CationPi";
  case BIT_HYDROPHOBIC:
    return "Hydrophobic";
  case BIT_HALOGEN:
    return "Halogen";
  default:
    return "Unknown";
  }
}

std::string TitanAwhIfp::build_contact_label(const ContactRecord &record,
                                             int bit) {
  std::ostringstream oss;
  oss << record.chainid_rec << ':' << record.res_name_rec << record.resnr_rec
      << '-' << record.chainid_lig << ':' << record.res_name_lig
      << record.resnr_lig << ':' << interaction_label(bit);
  return oss.str();
}

void TitanAwhIfp::write_representative_pdb(
    const std::string &filename, const RepresentativeFrame &frame,
    const std::vector<mol_info> &rec_info,
    const std::vector<mol_info> &lig_info) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_error("AWH: cannot write representative PDB " + filename);
    return;
  }

  out << std::fixed << std::setprecision(3);
  out << "REMARK    TITAN AWH representative frame\n";
  out << "REMARK    replica " << frame.replica_id << " rc "
      << format_double(frame.rc_value) << " delta "
      << format_double(frame.delta_to_center) << '\n';
  out << "TITLE     frame t= " << std::setw(9) << frame.time_ps << '\n';
  out << "CRYST1" << std::setw(9) << frame.rec_frame.Box.get_a() << std::setw(9)
      << frame.rec_frame.Box.get_b() << std::setw(9)
      << frame.rec_frame.Box.get_c() << std::setw(7)
      << frame.rec_frame.Box.get_alpha() << std::setw(7)
      << frame.rec_frame.Box.get_beta() << std::setw(7)
      << frame.rec_frame.Box.get_gamma() << '\n';
  out << "MODEL" << std::setw(9) << 1 << '\n';

  size_t atom_serial = 1;
  for (size_t i = 0; i < frame.rec_frame.size(); ++i) {
    const size_t serial = std::min<size_t>(atom_serial, 99999);
    out << std::setw(4) << "ATOM" << std::setw(7) << serial << std::setw(5)
        << rec_info[i].get_atom() << std::setw(5) << std::right
        << rec_info[i].get_original_res_name() << std::setw(1)
        << rec_info[i].get_chain() << std::setw(4) << rec_info[i].get_resnr()
        << std::setw(12) << frame.rec_frame.pos.x[i] << std::setw(8)
        << frame.rec_frame.pos.y[i] << std::setw(8) << frame.rec_frame.pos.z[i]
        << " " << std::setw(7) << 1.00 << " " << std::setw(6) << 0.00 << '\n';
    ++atom_serial;
  }
  out << "TER\n";

  for (size_t i = 0; i < frame.lig_frame.size(); ++i) {
    const size_t serial = std::min<size_t>(atom_serial, 99999);
    out << std::setw(4) << "ATOM" << std::setw(7) << serial << std::setw(5)
        << lig_info[i].get_atom() << std::setw(5) << std::right
        << lig_info[i].get_original_res_name() << std::setw(1)
        << lig_info[i].get_chain() << std::setw(4) << lig_info[i].get_resnr()
        << std::setw(12) << frame.lig_frame.pos.x[i] << std::setw(8)
        << frame.lig_frame.pos.y[i] << std::setw(8) << frame.lig_frame.pos.z[i]
        << " " << std::setw(7) << 1.00 << " " << std::setw(6) << 0.00 << '\n';
    ++atom_serial;
  }
  out << "TER\nENDMDL\n";
}

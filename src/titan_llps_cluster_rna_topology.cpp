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

void TitanLLPSCluster::export_rna_topology_outputs(
    const info &info_output) const {
  if (!rna_topology_enabled_ || rna_compaction_samples_.empty())
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
  auto topology_label = [](int state) -> const char * {
    switch (state) {
    case 0:
      return "unbound";
    case 1:
      return "local_decorating";
    case 2:
      return "single_bridge";
    case 3:
      return "cooperative_clamp";
    default:
      return "unknown";
    }
  };
  auto add_if_finite = [](RunningStats &stats, double value) {
    if (std::isfinite(value))
      stats.add(value);
  };
  auto safe_ratio = [](uint64_t num, uint64_t den) {
    return (den > 0)
               ? (static_cast<double>(num) / static_cast<double>(den))
               : std::numeric_limits<double>::quiet_NaN();
  };
  auto stats_mean_or_nan = [](const RunningStats &stats) {
    return (stats.n > 0) ? stats.mean()
                         : std::numeric_limits<double>::quiet_NaN();
  };
  auto stats_std_or_nan = [](const RunningStats &stats) {
    return (stats.n > 1) ? stats.stddev()
                         : std::numeric_limits<double>::quiet_NaN();
  };

  struct TopologyAggregate {
    uint64_t frames{0};
    uint64_t bound_frames{0};
    uint64_t bridge_frames{0};
    uint64_t clamp_frames{0};
    uint64_t longrange_positive_bound{0};
    uint64_t longrange_positive_bridge{0};
    uint64_t longrange_positive_clamp{0};
    std::unordered_map<int, uint64_t> state_counts;
    RunningStats num_bridge_proteins;
    RunningStats num_clamp_pairs;
    RunningStats bridge_residue_span_max;
    RunningStats bridge_geom_span_max_nm;
    RunningStats clamp_supported_longrange_contacts;
    RunningStats bridge_attraction_kjmol;
    RunningStats nonbridge_attraction_kjmol;
    RunningStats bridge_attraction_strength_kjmol;
    RunningStats nonbridge_attraction_strength_kjmol;
    RunningStats local_pp_elec_kjmol;
    RunningStats local_pp_vdw_kjmol;
    RunningStats local_pp_total_kjmol;
    RunningStats local_pp_attraction_strength_kjmol;
    RunningStats prna_attraction_strength_kjmol;
    RunningStats local_pp_sticker_contacts;
    RunningStats local_pp_to_prna_contact_ratio;
    RunningStats local_pp_to_prna_attraction_ratio;
    RunningStats local_pp_to_bridge_attraction_ratio;
    RunningStats longrange_fraction_when_bound;
    RunningStats longrange_fraction_when_bridge;
    RunningStats longrange_fraction_when_clamp;
    std::unordered_map<int, RunningStats> local_pp_contacts_by_state;
    std::unordered_map<int, RunningStats> local_pp_contact_ratio_by_state;
    std::unordered_map<int, RunningStats> local_pp_attraction_ratio_by_state;
    std::unordered_map<int, RunningStats> local_pp_bridge_ratio_by_state;
  };

  std::array<TopologyAggregate, 3> scope_aggs;
  std::array<std::unordered_map<int, TopologyAggregate>, 3> per_chain_aggs;
  std::array<std::unordered_map<int, TopologyAggregate>, 3> binding_bin_aggs;

  uint64_t inside_samples = 0;
  uint64_t contacting_samples = 0;
  for (const auto &sample : rna_compaction_samples_) {
    if (sample.inside_dense)
      ++inside_samples;
    if (sample.bound_protein_chains > 0.0)
      ++contacting_samples;
  }
  std::string primary_scope = "inside_dense_only";
  if (inside_samples == 0) {
    primary_scope = (contacting_samples > 0) ? "contacting_only" : "all_frames";
  }

  for (const auto &sample : rna_compaction_samples_) {
    const bool longrange_positive = sample.longrange_intrachain_contacts > 0.0;
    const int binding_value =
        static_cast<int>(std::lround(sample.bound_protein_chains));
    for (int scope_index = 0; scope_index < 3; ++scope_index) {
      if (!keep_scope(sample, scope_index))
        continue;
      auto apply_sample = [&](TopologyAggregate &agg) {
        ++agg.frames;
        ++agg.state_counts[sample.topology_state];
        add_if_finite(agg.local_pp_contacts_by_state[sample.topology_state],
                      sample.local_pp_sticker_contacts);
        add_if_finite(agg.local_pp_contact_ratio_by_state[sample.topology_state],
                      sample.local_pp_to_prna_contact_ratio);
        add_if_finite(agg.local_pp_attraction_ratio_by_state[sample.topology_state],
                      sample.local_pp_to_prna_attraction_ratio);
        add_if_finite(agg.local_pp_bridge_ratio_by_state[sample.topology_state],
                      sample.local_pp_to_bridge_attraction_ratio);
        if (sample.bound_protein_chains > 0.0) {
          ++agg.bound_frames;
          if (longrange_positive)
            ++agg.longrange_positive_bound;
          agg.longrange_fraction_when_bound.add(sample.longrange_contact_fraction);
          add_if_finite(agg.local_pp_elec_kjmol, sample.local_pp_elec_kjmol);
          add_if_finite(agg.local_pp_vdw_kjmol, sample.local_pp_vdw_kjmol);
          add_if_finite(agg.local_pp_total_kjmol, sample.local_pp_total_kjmol);
          add_if_finite(agg.local_pp_attraction_strength_kjmol,
                        sample.local_pp_attraction_strength_kjmol);
          add_if_finite(agg.prna_attraction_strength_kjmol,
                        sample.prna_attraction_strength_kjmol);
          add_if_finite(agg.local_pp_sticker_contacts,
                        sample.local_pp_sticker_contacts);
          add_if_finite(agg.local_pp_to_prna_contact_ratio,
                        sample.local_pp_to_prna_contact_ratio);
          add_if_finite(agg.local_pp_to_prna_attraction_ratio,
                        sample.local_pp_to_prna_attraction_ratio);
          add_if_finite(agg.local_pp_to_bridge_attraction_ratio,
                        sample.local_pp_to_bridge_attraction_ratio);
        }
        if (sample.any_bridge) {
          ++agg.bridge_frames;
          if (longrange_positive)
            ++agg.longrange_positive_bridge;
          agg.num_bridge_proteins.add(
              static_cast<double>(sample.num_bridge_proteins));
          agg.bridge_residue_span_max.add(sample.bridge_residue_span_max);
          agg.bridge_geom_span_max_nm.add(sample.bridge_geom_span_max_nm);
          add_if_finite(agg.bridge_attraction_kjmol,
                        sample.bridge_attraction_kjmol);
          add_if_finite(agg.nonbridge_attraction_kjmol,
                        sample.nonbridge_attraction_kjmol);
          add_if_finite(agg.bridge_attraction_strength_kjmol,
                        sample.bridge_attraction_strength_kjmol);
          add_if_finite(agg.nonbridge_attraction_strength_kjmol,
                        sample.nonbridge_attraction_strength_kjmol);
          agg.longrange_fraction_when_bridge.add(sample.longrange_contact_fraction);
        }
        if (sample.any_clamp) {
          ++agg.clamp_frames;
          if (longrange_positive)
            ++agg.longrange_positive_clamp;
          agg.num_clamp_pairs.add(static_cast<double>(sample.num_clamp_pairs));
          agg.clamp_supported_longrange_contacts.add(
              sample.clamp_supported_longrange_contacts);
          agg.longrange_fraction_when_clamp.add(sample.longrange_contact_fraction);
        }
      };

      apply_sample(scope_aggs[static_cast<std::size_t>(scope_index)]);
      apply_sample(per_chain_aggs[static_cast<std::size_t>(scope_index)]
                                 [sample.rna_chain_id]);
      apply_sample(
          binding_bin_aggs[static_cast<std::size_t>(scope_index)][binding_value]);
    }
  }

  {
    std::ofstream out(info_output.get_llps_rna_topology_summary_filename());
    if (out.is_open()) {
      out << "# RNA bridge/clamp topology of productive binding.\n";
      out << "# Detailed traces are streamed during runtime; this file contains "
             "summary statistics.\n";
      out << "primary_scope: " << primary_scope << "\n";
      out << "sequence_span_min_residues: " << rna_compaction_longrange_min_
          << "\n";
      out << "bridge_geom_min_nm: "
          << format_optional_csv(0.1 * rna_topology_bridge_geom_min_angstrom_)
          << "\n";
      out << "patch_gap_max: " << rna_topology_patch_gap_max_ << "\n";
      out << "clamp_require_loop: "
          << (rna_topology_clamp_require_loop_ ? "1" : "0") << "\n";
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        const auto &agg = scope_aggs[static_cast<std::size_t>(scope_index)];
        const std::string scope = scope_name(scope_index);
        const double p_longrange_given_bound =
            safe_ratio(agg.longrange_positive_bound, agg.bound_frames);
        const double p_longrange_given_bridge =
            safe_ratio(agg.longrange_positive_bridge, agg.bridge_frames);
        const double p_longrange_given_clamp =
            safe_ratio(agg.longrange_positive_clamp, agg.clamp_frames);
        const double p_bridge_given_bound =
            safe_ratio(agg.bridge_frames, agg.bound_frames);
        const double p_clamp_given_bound =
            safe_ratio(agg.clamp_frames, agg.bound_frames);
        const double bridge_enrichment =
            (std::isfinite(p_longrange_given_bound) && p_longrange_given_bound > 0.0 &&
             std::isfinite(p_longrange_given_bridge))
                ? (p_longrange_given_bridge / p_longrange_given_bound)
                : std::numeric_limits<double>::quiet_NaN();
        const double clamp_enrichment =
            (std::isfinite(p_longrange_given_bound) && p_longrange_given_bound > 0.0 &&
             std::isfinite(p_longrange_given_clamp))
                ? (p_longrange_given_clamp / p_longrange_given_bound)
                : std::numeric_limits<double>::quiet_NaN();
        out << scope << "_frames: " << agg.frames << "\n";
        out << scope << "_bound_frames: " << agg.bound_frames << "\n";
        out << scope << "_bridge_frames: " << agg.bridge_frames << "\n";
        out << scope << "_clamp_frames: " << agg.clamp_frames << "\n";
        out << scope << "_P_longrange_given_bound: "
            << format_optional_csv(p_longrange_given_bound) << "\n";
        out << scope << "_P_longrange_given_bridge: "
            << format_optional_csv(p_longrange_given_bridge) << "\n";
        out << scope << "_P_longrange_given_clamp: "
            << format_optional_csv(p_longrange_given_clamp) << "\n";
        out << scope << "_P_any_bridge_given_bound: "
            << format_optional_csv(p_bridge_given_bound) << "\n";
        out << scope << "_P_any_clamp_given_bound: "
            << format_optional_csv(p_clamp_given_bound) << "\n";
        out << scope << "_bridge_enrichment_over_bound: "
            << format_optional_csv(bridge_enrichment) << "\n";
        out << scope << "_clamp_enrichment_over_bound: "
            << format_optional_csv(clamp_enrichment) << "\n";
        out << scope << "_bridge_productivity_index: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.longrange_fraction_when_bridge))
            << "\n";
        out << scope << "_clamp_productivity_index: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.longrange_fraction_when_clamp))
            << "\n";
        out << scope << "_mean_bridge_residue_span_max: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.bridge_residue_span_max))
            << "\n";
        out << scope << "_mean_bridge_geom_span_max_nm: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.bridge_geom_span_max_nm))
            << "\n";
        out << scope << "_mean_bridge_total_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.bridge_attraction_kjmol))
            << "\n";
        out << scope << "_mean_nonbridge_total_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.nonbridge_attraction_kjmol))
            << "\n";
        out << scope << "_mean_bridge_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.bridge_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_nonbridge_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.nonbridge_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_prna_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.prna_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_local_pp_total_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_total_kjmol))
            << "\n";
        out << scope << "_mean_local_pp_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_local_pp_sticker_contacts: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_sticker_contacts))
            << "\n";
        out << scope << "_mean_local_pp_to_prna_contact_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_prna_contact_ratio))
            << "\n";
        out << scope << "_mean_local_pp_to_prna_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_prna_attraction_ratio))
            << "\n";
        out << scope << "_mean_local_pp_to_bridge_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_bridge_attraction_ratio))
            << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_rna_topology_summary_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_rna_topology_per_chain_filename());
    if (out.is_open()) {
      out << "Scope,RNAChainID,FrameCount,BoundFrames,BridgeFrames,ClampFrames,"
             "BoundFraction,AnyBridgeFraction,AnyClampFraction,"
             "LongRangeGivenBound,LongRangeGivenBridge,LongRangeGivenClamp,"
             "MeanNumBridgeProteins,MeanNumClampPairs,"
             "MeanBridgeResidueSpanMax,MeanBridgeGeomSpanMax_nm,"
             "MeanClampSupportedLongRangeContacts,MeanBridgeTotal_kJmol,"
             "MeanNonBridgeTotal_kJmol,MeanBridgeAttractionStrength_kJmol,"
             "MeanNonBridgeAttractionStrength_kJmol,"
             "MeanPRNAAttractionStrength_kJmol,MeanLocalPPElec_kJmol,"
             "MeanLocalPPVdw_kJmol,MeanLocalPPTotal_kJmol,"
             "MeanLocalPPAttractionStrength_kJmol,MeanLocalPPStickerContacts,"
             "MeanLocalPPtoPRNAContactRatio,MeanLocalPPtoPRNAAttractionRatio,"
             "MeanLocalPPtoBridgeAttractionRatio\n";
      out << std::fixed << std::setprecision(6);
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        std::vector<int> chain_ids;
        chain_ids.reserve(per_chain_aggs[static_cast<std::size_t>(scope_index)].size());
        for (const auto &entry : per_chain_aggs[static_cast<std::size_t>(scope_index)])
          chain_ids.push_back(entry.first);
        std::sort(chain_ids.begin(), chain_ids.end());
        for (int chain_id : chain_ids) {
          const auto &agg =
              per_chain_aggs[static_cast<std::size_t>(scope_index)].at(chain_id);
          out << scope_name(scope_index) << "," << chain_id << "," << agg.frames
              << "," << agg.bound_frames << "," << agg.bridge_frames << ","
              << agg.clamp_frames << ","
              << format_optional_csv(safe_ratio(agg.bound_frames, agg.frames))
              << ","
              << format_optional_csv(safe_ratio(agg.bridge_frames, agg.frames))
              << ","
              << format_optional_csv(safe_ratio(agg.clamp_frames, agg.frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_bound, agg.bound_frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_bridge, agg.bridge_frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_clamp, agg.clamp_frames))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.num_bridge_proteins))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.num_clamp_pairs))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_residue_span_max))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_geom_span_max_nm))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.clamp_supported_longrange_contacts))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.bridge_attraction_kjmol))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.nonbridge_attraction_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.nonbridge_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.prna_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_elec_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_vdw_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_total_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_attraction_strength_kjmol))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.local_pp_sticker_contacts))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_to_prna_contact_ratio))
              << ","
              << format_optional_csv(
                     stats_mean_or_nan(agg.local_pp_to_prna_attraction_ratio))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_to_bridge_attraction_ratio))
              << "\n";
        }
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_rna_topology_per_chain_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_topology_binding_binned_filename());
    if (out.is_open()) {
      out << "Scope,BoundProteinChainsBin,FrameCount,AnyBridgeFraction,"
             "AnyClampFraction,LongRangeGivenBound,LongRangeGivenBridge,"
             "LongRangeGivenClamp,MeanNumBridgeProteins,MeanNumClampPairs,"
             "MeanBridgeResidueSpanMax,MeanBridgeGeomSpanMax_nm,"
             "MeanClampSupportedLongRangeContacts,MeanBridgeTotal_kJmol,"
             "MeanNonBridgeTotal_kJmol,MeanBridgeAttractionStrength_kJmol,"
             "MeanNonBridgeAttractionStrength_kJmol,"
             "MeanPRNAAttractionStrength_kJmol,MeanLocalPPElec_kJmol,"
             "MeanLocalPPVdw_kJmol,MeanLocalPPTotal_kJmol,"
             "MeanLocalPPAttractionStrength_kJmol,MeanLocalPPStickerContacts,"
             "MeanLocalPPtoPRNAContactRatio,MeanLocalPPtoPRNAAttractionRatio,"
             "MeanLocalPPtoBridgeAttractionRatio,"
             "MeanLongRangeContactFractionWhenBound\n";
      out << std::fixed << std::setprecision(6);
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        std::vector<int> bins;
        bins.reserve(binding_bin_aggs[static_cast<std::size_t>(scope_index)].size());
        for (const auto &entry :
             binding_bin_aggs[static_cast<std::size_t>(scope_index)])
          bins.push_back(entry.first);
        std::sort(bins.begin(), bins.end());
        for (int bin : bins) {
          const auto &agg =
              binding_bin_aggs[static_cast<std::size_t>(scope_index)].at(bin);
          out << scope_name(scope_index) << "," << bin << "," << agg.frames
              << ","
              << format_optional_csv(safe_ratio(agg.bridge_frames, agg.frames))
              << ","
              << format_optional_csv(safe_ratio(agg.clamp_frames, agg.frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_bound, agg.bound_frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_bridge, agg.bridge_frames))
              << ","
              << format_optional_csv(
                     safe_ratio(agg.longrange_positive_clamp, agg.clamp_frames))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.num_bridge_proteins))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.num_clamp_pairs))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_residue_span_max))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_geom_span_max_nm))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.clamp_supported_longrange_contacts))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.bridge_attraction_kjmol))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.nonbridge_attraction_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.bridge_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.nonbridge_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.prna_attraction_strength_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_elec_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_vdw_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_total_kjmol))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_attraction_strength_kjmol))
              << "," << format_optional_csv(
                            stats_mean_or_nan(agg.local_pp_sticker_contacts))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_to_prna_contact_ratio))
              << ","
              << format_optional_csv(
                     stats_mean_or_nan(agg.local_pp_to_prna_attraction_ratio))
              << "," << format_optional_csv(stats_mean_or_nan(
                            agg.local_pp_to_bridge_attraction_ratio))
              << ","
              << format_optional_csv(
                     stats_mean_or_nan(agg.longrange_fraction_when_bound))
              << "\n";
        }
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_topology_binding_binned_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_topology_state_histogram_filename());
    if (out.is_open()) {
      out << "Scope,TopologyState,TopologyLabel,FrameCount,Probability\n";
      out << std::fixed << std::setprecision(6);
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        std::vector<int> states;
        states.reserve(scope_aggs[static_cast<std::size_t>(scope_index)]
                           .state_counts.size());
        for (const auto &entry :
             scope_aggs[static_cast<std::size_t>(scope_index)].state_counts)
          states.push_back(entry.first);
        std::sort(states.begin(), states.end());
        for (int state : states) {
          const uint64_t count =
              scope_aggs[static_cast<std::size_t>(scope_index)].state_counts.at(
                  state);
          out << scope_name(scope_index) << "," << state << ","
              << topology_label(state) << "," << count << ","
              << format_optional_csv(
                     safe_ratio(count, scope_aggs[static_cast<std::size_t>(
                                               scope_index)]
                                              .frames))
              << "\n";
        }
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_topology_state_histogram_filename());
    }
  }

  struct LifetimeSummary {
    uint64_t total_events{0};
    uint64_t complete_events{0};
    uint64_t censored_events{0};
    double mean_ns{0.0};
    double median_ns{0.0};
    double p90_ns{0.0};
    double max_ns{0.0};
    std::vector<double> values;
    std::vector<uint64_t> counts;
  };

  auto compute_lifetimes = [&](int scope_index, auto active_predicate) {
    std::unordered_map<int, std::vector<const RNACompactionSample *>> by_chain;
    for (const auto &sample : rna_compaction_samples_) {
      if (!keep_scope(sample, scope_index))
        continue;
      by_chain[sample.rna_chain_id].push_back(&sample);
    }
    std::vector<double> complete_durations;
    uint64_t total_events = 0;
    uint64_t complete_events = 0;
    uint64_t censored_events = 0;
    for (auto &entry : by_chain) {
      auto &samples = entry.second;
      std::sort(samples.begin(), samples.end(),
                [](const RNACompactionSample *a, const RNACompactionSample *b) {
                  if (a->frame != b->frame)
                    return a->frame < b->frame;
                  return a->time_ns < b->time_ns;
                });
      bool in_episode = false;
      double episode_start_ns = 0.0;
      bool censored_start = false;
      for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto *sample = samples[i];
        if (sample == nullptr)
          continue;
        const bool active = active_predicate(*sample);
        const bool consecutive =
            (i > 0 && sample->frame == samples[i - 1]->frame + 1);
        if (in_episode && !consecutive) {
          ++total_events;
          ++censored_events;
          in_episode = false;
        }
        if (active && !in_episode) {
          in_episode = true;
          episode_start_ns = sample->time_ns;
          censored_start = (i == 0);
          continue;
        }
        if (active && in_episode) {
          continue;
        }
        if (!active && in_episode) {
          ++total_events;
          const double duration =
              std::max(0.0, sample->time_ns - episode_start_ns);
          if (censored_start) {
            ++censored_events;
          } else {
            ++complete_events;
            complete_durations.push_back(duration);
          }
          in_episode = false;
        }
      }
      if (in_episode) {
        ++total_events;
        ++censored_events;
        in_episode = false;
      }
    }

    std::sort(complete_durations.begin(), complete_durations.end());
    LifetimeSummary summary;
    summary.total_events = total_events;
    summary.complete_events = complete_events;
    summary.censored_events = censored_events;
    if (!complete_durations.empty()) {
      summary.mean_ns = std::accumulate(complete_durations.begin(),
                                        complete_durations.end(), 0.0) /
                        static_cast<double>(complete_durations.size());
      summary.median_ns = percentile_sorted(complete_durations, 0.50);
      summary.p90_ns = percentile_sorted(complete_durations, 0.90);
      summary.max_ns = complete_durations.back();
      std::unordered_map<int64_t, uint64_t> histogram;
      for (double duration : complete_durations) {
        const auto key = static_cast<int64_t>(std::llround(duration * 1.0e6));
        ++histogram[key];
      }
      std::vector<int64_t> keys;
      keys.reserve(histogram.size());
      for (const auto &entry : histogram)
        keys.push_back(entry.first);
      std::sort(keys.begin(), keys.end());
      for (int64_t key : keys) {
        summary.values.push_back(static_cast<double>(key) / 1.0e6);
        summary.counts.push_back(histogram[key]);
      }
    }
    return summary;
  };

  auto write_lifetime_file = [&](const std::string &filename, auto predicate) {
    std::ofstream out(filename);
    if (!out.is_open())
      return false;
    for (int scope_index = 0; scope_index < 3; ++scope_index) {
      const auto summary = compute_lifetimes(scope_index, predicate);
      out << "# scope " << scope_name(scope_index) << "\n";
      out << "# total_events " << summary.total_events << "\n";
      out << "# complete_events " << summary.complete_events << "\n";
      out << "# censored_events " << summary.censored_events << "\n";
      out << "# mean_ns " << std::fixed << std::setprecision(6)
          << summary.mean_ns << "\n";
      out << "# median_ns " << summary.median_ns << "\n";
      out << "# p90_ns " << summary.p90_ns << "\n";
      out << "# max_ns " << summary.max_ns << "\n";
      out << "# lifetime_ns count\n";
      for (std::size_t i = 0; i < summary.values.size(); ++i)
        out << summary.values[i] << " " << summary.counts[i] << "\n";
    }
    return true;
  };

  if (write_lifetime_file(info_output.get_llps_rna_bridge_lifetimes_filename(),
                          [](const RNACompactionSample &sample) {
                            return sample.any_bridge;
                          })) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_rna_bridge_lifetimes_filename());
  }
  if (write_lifetime_file(info_output.get_llps_rna_clamp_lifetimes_filename(),
                          [](const RNACompactionSample &sample) {
                            return sample.any_clamp;
                          })) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_rna_clamp_lifetimes_filename());
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_loop_closure_efficiency_filename());
    if (out.is_open()) {
      out << "# Productive RNA loop-closure efficiency conditioned on binding "
             "topology.\n";
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        const auto &agg = scope_aggs[static_cast<std::size_t>(scope_index)];
        const std::string scope = scope_name(scope_index);
        const double p_longrange_given_bound =
            safe_ratio(agg.longrange_positive_bound, agg.bound_frames);
        const double p_longrange_given_bridge =
            safe_ratio(agg.longrange_positive_bridge, agg.bridge_frames);
        const double p_longrange_given_clamp =
            safe_ratio(agg.longrange_positive_clamp, agg.clamp_frames);
        const double bridge_enrichment =
            (std::isfinite(p_longrange_given_bound) && p_longrange_given_bound > 0.0 &&
             std::isfinite(p_longrange_given_bridge))
                ? (p_longrange_given_bridge / p_longrange_given_bound)
                : std::numeric_limits<double>::quiet_NaN();
        const double clamp_enrichment =
            (std::isfinite(p_longrange_given_bound) && p_longrange_given_bound > 0.0 &&
             std::isfinite(p_longrange_given_clamp))
                ? (p_longrange_given_clamp / p_longrange_given_bound)
                : std::numeric_limits<double>::quiet_NaN();
        out << scope << "_P_longrange_given_bound: "
            << format_optional_csv(p_longrange_given_bound) << "\n";
        out << scope << "_P_longrange_given_bridge: "
            << format_optional_csv(p_longrange_given_bridge) << "\n";
        out << scope << "_P_longrange_given_clamp: "
            << format_optional_csv(p_longrange_given_clamp) << "\n";
        out << scope << "_bridge_enrichment_over_bound: "
            << format_optional_csv(bridge_enrichment) << "\n";
        out << scope << "_clamp_enrichment_over_bound: "
            << format_optional_csv(clamp_enrichment) << "\n";
        out << scope << "_bridge_productivity_index: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.longrange_fraction_when_bridge))
            << "\n";
        out << scope << "_clamp_productivity_index: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.longrange_fraction_when_clamp))
            << "\n";
        out << scope << "_mean_local_pp_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_local_pp_to_prna_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_prna_attraction_ratio))
            << "\n";
        out << scope << "_mean_local_pp_to_bridge_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_bridge_attraction_ratio))
            << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_loop_closure_efficiency_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_local_competition_summary_filename());
    if (out.is_open()) {
      out << "# Local protein-protein competition around RNA-bound proteins.\n";
      for (int scope_index = 0; scope_index < 3; ++scope_index) {
        const auto &agg = scope_aggs[static_cast<std::size_t>(scope_index)];
        const std::string scope = scope_name(scope_index);
        out << scope << "_mean_local_pp_sticker_contacts: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_sticker_contacts))
            << "\n";
        out << scope << "_std_local_pp_sticker_contacts: "
            << format_optional_csv(
                   stats_std_or_nan(agg.local_pp_sticker_contacts))
            << "\n";
        out << scope << "_mean_local_pp_to_prna_contact_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_prna_contact_ratio))
            << "\n";
        out << scope << "_std_local_pp_to_prna_contact_ratio: "
            << format_optional_csv(
                   stats_std_or_nan(agg.local_pp_to_prna_contact_ratio))
            << "\n";
        out << scope << "_mean_local_pp_to_prna_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_prna_attraction_ratio))
            << "\n";
        out << scope << "_std_local_pp_to_prna_attraction_ratio: "
            << format_optional_csv(
                   stats_std_or_nan(agg.local_pp_to_prna_attraction_ratio))
            << "\n";
        out << scope << "_mean_local_pp_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_attraction_strength_kjmol))
            << "\n";
        out << scope << "_std_local_pp_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_std_or_nan(agg.local_pp_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_prna_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.prna_attraction_strength_kjmol))
            << "\n";
        out << scope << "_std_prna_attraction_strength_kjmol: "
            << format_optional_csv(
                   stats_std_or_nan(agg.prna_attraction_strength_kjmol))
            << "\n";
        out << scope << "_mean_local_pp_to_bridge_attraction_ratio: "
            << format_optional_csv(
                   stats_mean_or_nan(agg.local_pp_to_bridge_attraction_ratio))
            << "\n";
        out << scope << "_std_local_pp_to_bridge_attraction_ratio: "
            << format_optional_csv(
                   stats_std_or_nan(agg.local_pp_to_bridge_attraction_ratio))
            << "\n";
        for (int state = 0; state <= 3; ++state) {
          const auto contact_it = agg.local_pp_contacts_by_state.find(state);
          const auto ratio_it = agg.local_pp_contact_ratio_by_state.find(state);
          const auto attr_it =
              agg.local_pp_attraction_ratio_by_state.find(state);
          const auto bridge_ratio_it =
              agg.local_pp_bridge_ratio_by_state.find(state);
          out << scope << "_" << topology_label(state)
              << "_mean_local_pp_contacts: "
              << format_optional_csv(contact_it != agg.local_pp_contacts_by_state.end()
                                         ? contact_it->second.mean()
                                         : std::numeric_limits<double>::quiet_NaN())
              << "\n";
          out << scope << "_" << topology_label(state)
              << "_mean_local_pp_to_prna_contact_ratio: "
              << format_optional_csv(
                     ratio_it != agg.local_pp_contact_ratio_by_state.end()
                         ? ratio_it->second.mean()
                         : std::numeric_limits<double>::quiet_NaN())
              << "\n";
          out << scope << "_" << topology_label(state)
              << "_mean_local_pp_to_prna_attraction_ratio: "
              << format_optional_csv(
                     attr_it != agg.local_pp_attraction_ratio_by_state.end()
                         ? attr_it->second.mean()
                         : std::numeric_limits<double>::quiet_NaN())
              << "\n";
          out << scope << "_" << topology_label(state)
              << "_mean_local_pp_to_bridge_attraction_ratio: "
              << format_optional_csv(
                     bridge_ratio_it != agg.local_pp_bridge_ratio_by_state.end()
                         ? bridge_ratio_it->second.mean()
                         : std::numeric_limits<double>::quiet_NaN())
              << "\n";
        }
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_local_competition_summary_filename());
    }
  }
}

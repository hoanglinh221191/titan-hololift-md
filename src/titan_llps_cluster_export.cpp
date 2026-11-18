// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "BoxMatrix.h"
#include "info.h"
#include "titan_executor.h"
#include "titan_llps.h"
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

void TitanLLPSCluster::export_timeseries(const std::string &out_csv) const {
  if (streaming_csv_) {
    if (ts_out_.is_open())
      const_cast<std::ofstream &>(ts_out_).flush();
    TitanUI::print_info("[LLPS-DBSCAN] Timeseries already streamed to CSV.");
    return;
  }
  std::ofstream out(out_csv);
  if (!out.is_open()) {
    TitanUI::print_warning("[LLPS-DBSCAN] Cannot open " + out_csv);
    return;
  }
  out << "Time_ps,TotalBeads,NumCondensates,"
         "GiantCluster_ProChains,GiantCluster_RnaChains,NoiseBeads,"
         "Rg_Pro_Dilute,Rg_Pro_Dense,Rg_RNA_Dilute,Rg_RNA_Dense\n";
  for (const auto &m : timeseries_) {
    out << std::fixed << std::setprecision(1) << m.time_ps << "," << m.num_nodes
        << "," << m.num_clusters << "," << m.largest_cluster_pro_chains << ","
        << m.largest_cluster_rna_chains << "," << m.noise_count << ","
        << std::setprecision(3) << m.rg_pro_dilute << "," << m.rg_pro_dense
        << "," << m.rg_rna_dilute << "," << m.rg_rna_dense << "\n";
  }
  TitanUI::print_info("[LLPS-DBSCAN] Written: " + out_csv);
}

void TitanLLPSCluster::export_performance_report(
    const info &info_output, const TitanExecutor &executor) const {
  if (droplet_shape_streaming_) {
    if (droplet_shape_out_.is_open())
      const_cast<std::ofstream &>(droplet_shape_out_).flush();
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_droplet_shape_timeseries_filename());
  } else {
    std::ofstream shape_out(info_output.get_llps_droplet_shape_timeseries_filename());
    if (shape_out.is_open()) {
      shape_out << "Time_ps,Frame,ClusterID,Nodes,ProteinNodes,RnaNodes,Rg_A,"
                   "Asphericity_A2,Acylindricity_A2,RelativeShapeAnisotropy,"
                   "SphericityProxy\n";
      shape_out << std::fixed << std::setprecision(6);
      for (const auto &row : droplet_shape_rows_) {
        shape_out << row.time_ps << "," << row.frame << "," << row.cluster_id
                  << "," << row.nodes << "," << row.protein_nodes << ","
                  << row.rna_nodes << "," << row.rg_angstrom << ","
                  << row.asphericity_angstrom2 << ","
                  << row.acylindricity_angstrom2 << ","
                  << row.relative_shape_anisotropy << ","
                  << row.sphericity_proxy << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_droplet_shape_timeseries_filename());
    }
  }

  const bool new_report = !static_cast<bool>(
      std::ifstream(info_output.get_llps_performance_report_filename()));
  std::ofstream report(info_output.get_llps_performance_report_filename(),
                       std::ios::app);
  if (!report.is_open())
    return;
  if (new_report) {
    report << "Module,Frames,Nodes,KernelPolicy,KernelEffective,"
              "ExecutorWorkers,Pack_s,Contacts_s,ProteinRNA_s,Graph_s,"
              "Density_s,Polymer_s,Dynamics_s,FinalizeCt_s,"
              "FinalizeContactMap_s,FinalizeTotal_s,MacroContactPairs,"
              "StickerPairs,ProteinPairs,ProteinRNAPairs,MaxQueueDepth,"
              "DensityMode\n";
  }
  const std::size_t effective_workers = std::max<std::size_t>(
      perf_.process_executor_workers,
      std::max<std::size_t>(perf_.finalize_executor_workers, executor.capacity()));
  report << "LLPS_dbscan," << perf_.frames_processed << ","
         << perf_.node_count_accum << "," << kernel_mode_ << ","
         << (kernel_mode_ == "scalar" ? "scalar" : "executor") << ","
         << effective_workers << "," << std::fixed << std::setprecision(6)
         << perf_.extract_seconds << "," << perf_.dbscan_seconds << ","
         << perf_.rna_compaction_seconds << "," << perf_.cluster_summary_seconds
         << "," << 0.0 << "," << 0.0 << "," << perf_.dynamics_seconds << ","
         << 0.0 << "," << 0.0 << "," << perf_.finalize_seconds << ","
         << perf_.dbscan_cluster_expansions << ","
         << perf_.dbscan_neighbor_checks << "," << 0 << "," << 0 << "," << 0
         << ",dbscan\n";
  TitanUI::print_info("  Written: " +
                      info_output.get_llps_performance_report_filename());
}

void TitanLLPSCluster::export_dynamics(
    const info &info_output, std::fstream &log,
    const LLPSContactDynamicsResult *contact_result) const {
  if (!dynamics_enabled_)
    return;

  TitanUI::log_section(log, "LLPS Dynamics Output Audit");
  TitanUI::log_kv(log, "LLPS DBSCAN Frames",
                  std::to_string(perf_.frames_processed));
  TitanUI::log_kv(log, "LLPS DBSCAN Nodes Accum",
                  std::to_string(perf_.node_count_accum));
  TitanUI::log_kv(log, "LLPS DBSCAN Kernel Policy", kernel_mode_);
  const std::size_t dynamics_ring_bytes =
      frame_present_ring_.size() * sizeof(uint8_t) +
      frame_inside_ring_.size() * sizeof(uint8_t) +
      frame_episode_ring_.size() * sizeof(uint64_t) +
      (frame_unwrapped_x_ring_.size() + frame_unwrapped_y_ring_.size() +
       frame_unwrapped_z_ring_.size()) *
          sizeof(double) +
      (membership_joint_inside_.size() + membership_pair_counts_.size()) *
          sizeof(uint64_t);
  std::ostringstream storage_oss;
  storage_oss << std::fixed << std::setprecision(3)
              << static_cast<double>(dynamics_ring_bytes) /
                     (1024.0 * 1024.0);
  TitanUI::log_kv(log, "LLPS Dynamics Storage", "flat row-major");
  TitanUI::log_kv(log, "LLPS Dynamics Storage MiB", storage_oss.str());
  TitanUI::log_kv(log, "LLPS Detailed Trace Mode",
                  dynamics_detailed_
                      ? (membership_trace_streaming_ ? "streaming"
                                                     : "buffered fallback")
                      : "disabled");
  TitanUI::log_time(log, "LLPS DBSCAN Extract", perf_.extract_seconds);
  TitanUI::log_time(log, "LLPS DBSCAN Cluster", perf_.dbscan_seconds);
  TitanUI::log_time(log, "LLPS RNA Compaction",
                    perf_.rna_compaction_seconds);
  TitanUI::log_time(log, "LLPS Cluster Summary",
                    perf_.cluster_summary_seconds);
  TitanUI::log_time(log, "LLPS Chain Dynamics", perf_.dynamics_seconds);
  TitanUI::log_time(log, "LLPS Cluster Finalize", perf_.finalize_seconds);

  auto write_autocorr = [](const std::string &filename,
                           const LLPSAutocorrSpeciesResult &result) {
    std::ofstream out(filename);
    if (!out.is_open())
      return false;
    out << "# tau_ns " << std::fixed << std::setprecision(6) << result.tau_ns
        << "\n";
    // Liquidity v2: the pooled curve keeps permanently resident units, so it
    // decays to a plateau instead of to zero and is the FRAP-comparable one.
    out << "# pooled_tau_ns " << result.pooled_tau_ns << "\n";
    out << "# pooled_variance_amplitude " << result.mobile_fraction << "\n";
    out << "# material_exchange_fraction " << result.exchange_fraction << "\n";
    out << "# units_permanent " << result.units_permanent
        << " units_exchanging " << result.units_exchanging
        << " units_never " << result.units_never << "\n";
    out << "# pooled_immobile_plateau " << result.pooled_plateau << "\n";
    out << "# dense_occupancy " << result.occupancy << "\n";
    out << "# units " << result.units << "\n";
    out << "# lag_ns mean_connected_autocorr std_connected_autocorr "
           "contributing_chains pooled_autocorr pooled_pairs\n";
    for (std::size_t i = 0; i < result.lag_ns.size(); ++i) {
      const double pooled =
          (i < result.pooled.size()) ? result.pooled[i] : 0.0;
      const uint64_t pooled_pairs =
          (i < result.pooled_pairs.size()) ? result.pooled_pairs[i] : 0;
      out << result.lag_ns[i] << " " << result.mean[i] << " " << result.stddev[i]
          << " " << result.contributing_chains[i] << " " << pooled << " "
          << pooled_pairs << "\n";
    }
    return true;
  };

  if (write_autocorr(info_output.get_llps_membership_autocorr_protein_filename(),
                     chain_dynamics_result_.protein_membership_autocorr)) {
    TitanUI::print_info(
        "  Written: " + info_output.get_llps_membership_autocorr_protein_filename());
  }
  if (write_autocorr(info_output.get_llps_membership_autocorr_rna_filename(),
                     chain_dynamics_result_.rna_membership_autocorr)) {
    TitanUI::print_info(
        "  Written: " + info_output.get_llps_membership_autocorr_rna_filename());
  }

  {
    std::ofstream out(info_output.get_llps_residence_times_filename());
    if (out.is_open()) {
      out << "Species,ChainID,EpisodeID,Start_ns,End_ns,Duration_ns,"
             "CensoredStart,CensoredEnd\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : residence_rows_) {
        out << species_label(row.species) << "," << row.raw_chain_id << ","
            << row.episode_id << "," << row.start_ns << "," << row.end_ns << ","
            << row.duration_ns << "," << (row.censored_start ? 1 : 0) << ","
            << (row.censored_end ? 1 : 0) << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_residence_times_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_entry_exit_rates_filename());
    if (out.is_open()) {
      out << "protein_entry_rate_per_ns: "
          << chain_dynamics_result_.protein_entry_rate_per_ns << "\n";
      out << "protein_exit_rate_per_ns: "
          << chain_dynamics_result_.protein_exit_rate_per_ns << "\n";
      out << "rna_entry_rate_per_ns: "
          << chain_dynamics_result_.rna_entry_rate_per_ns << "\n";
      out << "rna_exit_rate_per_ns: "
          << chain_dynamics_result_.rna_exit_rate_per_ns << "\n";
      out << "protein_entry_rate_per_chain_ns: "
          << chain_dynamics_result_.protein_entry_rate_per_chain_ns << "\n";
      out << "protein_exit_rate_per_chain_ns: "
          << chain_dynamics_result_.protein_exit_rate_per_chain_ns << "\n";
      out << "rna_entry_rate_per_chain_ns: "
          << chain_dynamics_result_.rna_entry_rate_per_chain_ns << "\n";
      out << "rna_exit_rate_per_chain_ns: "
          << chain_dynamics_result_.rna_exit_rate_per_chain_ns << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_entry_exit_rates_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_cluster_lineage_filename());
    if (out.is_open()) {
      out << "Time_ns,FrameClusterID,ProteinChains,RnaChains,MatchedPrevClusters,"
             "MatchedJaccard,MatchedPrevShare,Event\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : lineage_rows_) {
        out << row.time_ns << "," << row.frame_cluster_id << ","
            << row.protein_chains << "," << row.rna_chains << ","
            << join_ints(row.matched_prev_clusters) << ","
            << join_doubles(row.matched_jaccard) << ","
            << join_doubles(row.matched_prev_share) << "," << row.event << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_cluster_lineage_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_fusion_fission_filename());
    if (out.is_open()) {
      out << "fusion_events: " << cluster_lineage_result_.fusion_events << "\n";
      out << "fission_events: " << cluster_lineage_result_.fission_events << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_fusion_fission_filename());
    }
  }

  {
    uint64_t born = 0;
    uint64_t stable = 0;
    uint64_t fusion = 0;
    uint64_t other = 0;
    double jaccard_sum = 0.0;
    double jaccard_max = 0.0;
    uint64_t jaccard_count = 0;
    for (const auto &row : lineage_rows_) {
      if (row.event == "born")
        ++born;
      else if (row.event == "stable")
        ++stable;
      else if (row.event == "fusion")
        ++fusion;
      else
        ++other;
      for (double value : row.matched_jaccard) {
        jaccard_sum += value;
        jaccard_max = std::max(jaccard_max, value);
        ++jaccard_count;
      }
    }
    std::ofstream out(info_output.get_llps_droplet_genealogy_summary_filename());
    if (out.is_open()) {
      out << "frames_analyzed: " << dynamics_total_frames_ << "\n";
      out << "time_window_ns: " << first_frame_time_ns_ << " "
          << last_frame_time_ns_ << "\n";
      out << "lineage_rows: " << lineage_rows_.size() << "\n";
      out << "born_events: " << born << "\n";
      out << "stable_events: " << stable << "\n";
      out << "fusion_events: " << fusion << "\n";
      out << "fission_events: " << cluster_lineage_result_.fission_events << "\n";
      out << "other_events: " << other << "\n";
      out << "mean_matched_jaccard: "
          << (jaccard_count > 0
                  ? jaccard_sum / static_cast<double>(jaccard_count)
                  : 0.0)
          << "\n";
      out << "max_matched_jaccard: " << jaccard_max << "\n";
      out << "lineage_containment_threshold: " << lineage_overlap_threshold_
          << "\n";
      out << "lineage_rule: a previous cluster belongs to a current one when "
             "at least the threshold share of its members is in it (fusion: "
             ">= 2 such previous clusters); a current cluster descends from a "
             "previous one when that share of its members was in it (fission: "
             ">= 2 such current clusters)\n";
      out << "min_lineage_cluster_size: " << min_lineage_cluster_size_ << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_droplet_genealogy_summary_filename());
    }
  }

  {
    const auto &msd_ref = dense_msd_result_;
    auto write_species = [&msd_ref](std::ofstream &out,
                            const std::string &label,
                            const LLPSAutocorrSpeciesResult &autocorr,
                            const LLPSEpisodeSummary &residence,
                            double entry_rate, double exit_rate) {
      const double tau = autocorr.tau_ns;
      const double recovery_half_time =
          (tau > 0.0) ? tau * std::log(2.0)
                      : std::numeric_limits<double>::quiet_NaN();  // v3: undefined, not 0
      const double turnover_rate = entry_rate + exit_rate;
      // Liquidity v2. The old MobileFractionProxy was
      // complete_episodes / total_episodes, i.e. the fraction of residence
      // episodes that both started and ended inside the analysis window.
      // That is a censoring statistic, not a FRAP plateau: a species that
      // never leaves has no complete episodes at all, and a species with a
      // single one-frame excursion scores 1.0. Both FRAP-comparable columns
      // below replace it, and the old number is kept under its real name.
      const double censor_fraction =
          (residence.total_episodes > 0)
              ? static_cast<double>(residence.complete_episodes) /
                    static_cast<double>(residence.total_episodes)
              : 0.0;
      const bool is_protein = (label == "protein");
      const double disp_mobile =
          is_protein ? msd_ref.protein_mobile_fraction_displacement
                     : msd_ref.rna_mobile_fraction_displacement;
      const uint64_t disp_pairs =
          is_protein ? msd_ref.protein_mobile_pairs : msd_ref.rna_mobile_pairs;
      const double pooled_half_time =
          (autocorr.pooled_tau_ns > 0.0)
              ? autocorr.pooled_tau_ns * std::log(2.0)
              : std::numeric_limits<double>::quiet_NaN();
      out << label << "," << tau << "," << recovery_half_time << ","
          << residence.mean_complete_ns << "," << residence.median_complete_ns
          << "," << residence.p90_complete_ns << "," << entry_rate << ","
          << exit_rate << "," << turnover_rate << ","
          << autocorr.exchange_fraction << "," << disp_mobile << ","
          << autocorr.mobile_fraction << "," << autocorr.pooled_plateau << ","
          << autocorr.pooled_tau_ns << "," << pooled_half_time << ","
          << autocorr.occupancy << "," << autocorr.units << ","
          << autocorr.units_permanent << "," << autocorr.units_exchanging << ","
          << autocorr.units_never << "," << disp_pairs << ","
          << msd_ref.mobile_displacement_nm << "," << msd_ref.mobile_lag_ns
          << "," << censor_fraction << "\n";
    };
    std::ofstream out(info_output.get_llps_frap_recovery_proxy_filename());
    if (out.is_open()) {
      out << "Species,MembershipTau_ns,RecoveryHalfTimeProxy_ns,"
             "MeanResidence_ns,MedianResidence_ns,P90Residence_ns,"
             "EntryRate_per_ns,ExitRate_per_ns,TurnoverRate_per_ns,"
             "MobileFractionExchange,MobileFractionDisplacement,"
             "PooledVarianceAmplitude,ImmobilePlateau,PooledTau_ns,"
             "PooledHalfTime_ns,Occupancy,Units,UnitsPermanent,"
             "UnitsExchanging,UnitsNever,DisplacementPairs,"
             "DisplacementThreshold_nm,DisplacementLag_ns,"
             "EpisodeCompleteFraction\n";
      out << std::fixed << std::setprecision(6);
      write_species(out, "protein",
                    chain_dynamics_result_.protein_membership_autocorr,
                    chain_dynamics_result_.protein_residence,
                    chain_dynamics_result_.protein_entry_rate_per_ns,
                    chain_dynamics_result_.protein_exit_rate_per_ns);
      write_species(out, "rna",
                    chain_dynamics_result_.rna_membership_autocorr,
                    chain_dynamics_result_.rna_residence,
                    chain_dynamics_result_.rna_entry_rate_per_ns,
                    chain_dynamics_result_.rna_exit_rate_per_ns);
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_frap_recovery_proxy_filename());
    }
  }

  const auto &msd_res = dense_msd_result_;
  auto write_msd = [&msd_res](const std::string &filename,
                      const std::vector<double> &lags,
                      const std::vector<double> &msd,
                      const std::vector<uint64_t> &counts,
                      double diffusion, double fit_start, double fit_end,
                      double exponent, double intercept, bool diffusive,
                      const std::vector<double> &alpha) {
    std::ofstream out(filename);
    if (!out.is_open())
      return false;
    out << "# diffusion_nm2_per_ns " << std::fixed << std::setprecision(6)
        << diffusion << "\n";
    out << "# diffusive_regime " << (diffusive ? "yes" : "no") << "\n";
    out << "# msd_loglog_exponent " << exponent << "\n";
    out << "# fit_window_ns " << fit_start << " " << fit_end << "\n";
    out << "# fit_intercept_nm2 " << intercept << "\n";
    out << "# msd_pairing inside_dense_at_both_endpoints\n";
    out << "# coordinates chain_or_segment_com_unwrapped_mic_nm\n";
    out << "# dense_phase_drift_removed " <<
        (msd_res.drift_corrected ? "yes" : "no") << " cumulative_nm "
        << msd_res.dense_drift_removed_nm << "\n";
    out << "# local_alpha = d ln MSD / d ln t, OLS over [t/sqrt2, t*sqrt2]\n";
    out << "# lag_ns msd_nm2 count local_alpha\n";
    for (std::size_t i = 0; i < lags.size(); ++i) {
      out << lags[i] << " " << msd[i] << " " << counts[i] << " ";
      if (i < alpha.size() && std::isfinite(alpha[i]))
        out << alpha[i];
      else
        out << "nan";
      out << "\n";
    }
    return true;
  };

  if (write_msd(info_output.get_llps_dense_msd_protein_filename(),
                dense_msd_result_.lag_ns, dense_msd_result_.protein_msd_nm2,
                dense_msd_result_.protein_counts,
                dense_msd_result_.protein_diffusion_nm2_per_ns,
                dense_msd_result_.protein_fit_start_ns,
                dense_msd_result_.protein_fit_end_ns,
                dense_msd_result_.protein_msd_exponent,
                dense_msd_result_.protein_msd_intercept_nm2,
                dense_msd_result_.protein_diffusive,
                dense_msd_result_.protein_local_alpha)) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_dense_msd_protein_filename());
  }
  // Liquidity v3: displacement mobile-fraction curve, written next to the
  // MSD files (name derived from the protein MSD file name).
  {
    std::string fn = info_output.get_llps_dense_msd_protein_filename();
    const std::string key = "llps_dense_msd_protein";
    const auto pos = fn.find(key);
    if (pos != std::string::npos)
      fn.replace(pos, key.size(), "llps_mobile_fraction_curve");
    else
      fn = "llps_mobile_fraction_curve.dat";
    std::ofstream out(fn);
    if (out.is_open()) {
      out << "# fraction of (unit, origin) pairs, inside the dense phase at "
             "both ends, whose drift-corrected COM displacement exceeds "
          << std::fixed << std::setprecision(3)
          << dense_msd_result_.mobile_displacement_nm << " nm\n";
      out << "# lag_ns protein_fraction protein_pairs rna_fraction rna_pairs\n";
      out << std::setprecision(6);
      const auto &r = dense_msd_result_;
      for (std::size_t i = 1; i < r.lag_ns.size(); ++i) {
        out << r.lag_ns[i] << " ";
        if (i < r.protein_mobile_curve.size() &&
            std::isfinite(r.protein_mobile_curve[i]))
          out << r.protein_mobile_curve[i];
        else
          out << "nan";
        out << " "
            << (i < r.protein_mobile_curve_pairs.size()
                    ? r.protein_mobile_curve_pairs[i]
                    : 0)
            << " ";
        if (i < r.rna_mobile_curve.size() && std::isfinite(r.rna_mobile_curve[i]))
          out << r.rna_mobile_curve[i];
        else
          out << "nan";
        out << " "
            << (i < r.rna_mobile_curve_pairs.size() ? r.rna_mobile_curve_pairs[i]
                                                    : 0)
            << "\n";
      }
      TitanUI::print_info("  Written: " + fn);
    }
  }
  // Liquidity v3: lateral MSD and whole RNA molecule MSD files.
  {
    const auto derive = [&](const std::string &stem) {
      std::string fn = info_output.get_llps_dense_msd_protein_filename();
      const std::string key = "llps_dense_msd_protein";
      const auto pos = fn.find(key);
      if (pos != std::string::npos)
        fn.replace(pos, key.size(), stem);
      else
        fn = stem + ".dat";
      return fn;
    };
    const auto &r = dense_msd_result_;
    const auto put = [](std::ofstream &o, const std::vector<double> &v,
                        std::size_t i) {
      if (i < v.size() && std::isfinite(v[i]))
        o << v[i];
      else
        o << "nan";
    };
    {
      const std::string fn = derive("llps_dense_msd_lateral");
      std::ofstream out(fn);
      if (out.is_open()) {
        out << std::fixed << std::setprecision(6);
        out << "# lateral = displacement without its component along the slab "
               "normal; slab axis "
            << r.slab_normal_axis
            << " (0=a,1=b,2=c: lattice vector with the largest plane "
               "spacing), unit normal "
            << r.slab_normal[0] << " " << r.slab_normal[1] << " "
            << r.slab_normal[2] << "; D = slope/4\n";
        out << "# protein: D_lat " << r.protein_lat_diffusion_nm2_per_ns
            << " diffusive " << (r.protein_lat_diffusive ? "yes" : "no")
            << " window_ns " << r.protein_lat_fit_start_ns << " "
            << r.protein_lat_fit_end_ns << " exponent "
            << r.protein_lat_exponent << "\n";
        out << "# rna_segment: D_lat " << r.rna_lat_diffusion_nm2_per_ns
            << " diffusive " << (r.rna_lat_diffusive ? "yes" : "no")
            << " window_ns " << r.rna_lat_fit_start_ns << " "
            << r.rna_lat_fit_end_ns << " exponent " << r.rna_lat_exponent
            << "\n";
        out << "# lag_ns protein_msd_lat_nm2 protein_count protein_alpha "
               "rna_msd_lat_nm2 rna_count rna_alpha\n";
        for (std::size_t i = 0; i < r.lag_ns.size(); ++i) {
          out << r.lag_ns[i] << " ";
          put(out, r.protein_msd_lat_nm2, i);
          out << " " << (i < r.protein_counts.size() ? r.protein_counts[i] : 0)
              << " ";
          put(out, r.protein_lat_local_alpha, i);
          out << " ";
          put(out, r.rna_msd_lat_nm2, i);
          out << " " << (i < r.rna_counts.size() ? r.rna_counts[i] : 0) << " ";
          put(out, r.rna_lat_local_alpha, i);
          out << "\n";
        }
        TitanUI::print_info("  Written: " + fn);
      }
    }
    {
      const std::string fn = derive("llps_dense_msd_rna_molecule");
      std::ofstream out(fn);
      if (out.is_open()) {
        out << std::fixed << std::setprecision(6);
        out << "# whole RNA molecule COM (all backbone nodes), drift removed, "
               "pairs inside dense at both ends; molecules "
            << r.rna_molecules << "\n";
        out << "# lag_ns msd_nm2 msd_lateral_nm2 count\n";
        for (std::size_t i = 0; i < r.rna_molecule_lag_ns.size(); ++i)
          out << r.rna_molecule_lag_ns[i] << " " << r.rna_molecule_msd_nm2[i]
              << " " << r.rna_molecule_msd_lat_nm2[i] << " "
              << r.rna_molecule_counts[i] << "\n";
        TitanUI::print_info("  Written: " + fn);
      }
    }
  }
  if (write_msd(info_output.get_llps_dense_msd_rna_filename(),
                dense_msd_result_.lag_ns, dense_msd_result_.rna_msd_nm2,
                dense_msd_result_.rna_counts,
                dense_msd_result_.rna_diffusion_nm2_per_ns,
                dense_msd_result_.rna_fit_start_ns,
                dense_msd_result_.rna_fit_end_ns,
                dense_msd_result_.rna_msd_exponent,
                dense_msd_result_.rna_msd_intercept_nm2,
                dense_msd_result_.rna_diffusive,
                dense_msd_result_.rna_local_alpha)) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_dense_msd_rna_filename());
  }

  export_rna_compaction_outputs(info_output);
  TitanUI::log_kv(log, info_output.get_llps_rna_compaction_binding_summary_filename(),
                  rna_compaction_enabled_ ? "emitted" : "skipped: disabled");
  TitanUI::log_kv(log, info_output.get_llps_rna_information_summary_filename(),
                  rna_information_enabled_ ? "emitted" : "skipped: disabled");
  TitanUI::log_kv(log, info_output.get_llps_rna_topology_summary_filename(),
                  rna_topology_enabled_ ? "emitted" : "skipped: disabled");
  TitanUI::log_kv(log, info_output.get_llps_droplet_genealogy_summary_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_llps_frap_recovery_proxy_filename(),
                  "emitted");
  TitanUI::log_kv(log, "LLPS RNA Topology Cache",
                  rna_compaction_topology_cache_.ready ? "enabled"
                                                       : "not-used");

  if (dynamics_detailed_) {
    if (membership_trace_streaming_) {
      if (membership_trace_out_.is_open())
        const_cast<std::ofstream &>(membership_trace_out_).flush();
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_chain_membership_trace_filename());
    } else {
    std::ofstream out(info_output.get_llps_chain_membership_trace_filename());
    if (out.is_open()) {
      out << "Time_ns,Species,ChainID,EpisodeID,InsideDense,Event\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : membership_trace_rows_) {
        out << row.time_ns << "," << species_label(row.species) << ","
            << row.raw_chain_id << "," << row.episode_id << ","
            << row.inside_dense << "," << row.event << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_chain_membership_trace_filename());
    }
    }
  }

  {
    std::ofstream out(info_output.get_llps_liquidity_summary_filename());
    if (out.is_open()) {
      out << "protein_membership_tau_ns: "
          << chain_dynamics_result_.protein_membership_autocorr.tau_ns << "\n";
      out << "rna_membership_tau_ns: "
          << chain_dynamics_result_.rna_membership_autocorr.tau_ns << "\n";
      out << "protein_mean_residence_ns: "
          << chain_dynamics_result_.protein_residence.mean_complete_ns << "\n";
      out << "protein_median_residence_ns: "
          << chain_dynamics_result_.protein_residence.median_complete_ns << "\n";
      out << "rna_mean_residence_ns: "
          << chain_dynamics_result_.rna_residence.mean_complete_ns << "\n";
      out << "rna_median_residence_ns: "
          << chain_dynamics_result_.rna_residence.median_complete_ns << "\n";
      out << "protein_entry_rate_per_ns: "
          << chain_dynamics_result_.protein_entry_rate_per_ns << "\n";
      out << "protein_exit_rate_per_ns: "
          << chain_dynamics_result_.protein_exit_rate_per_ns << "\n";
      out << "rna_entry_rate_per_ns: "
          << chain_dynamics_result_.rna_entry_rate_per_ns << "\n";
      out << "rna_exit_rate_per_ns: "
          << chain_dynamics_result_.rna_exit_rate_per_ns << "\n";
      out << "fusion_events: " << cluster_lineage_result_.fusion_events << "\n";
      out << "fission_events: " << cluster_lineage_result_.fission_events << "\n";
      out << "protein_diffusion_nm2_per_ns: "
          << dense_msd_result_.protein_diffusion_nm2_per_ns << "\n";
      out << "rna_diffusion_nm2_per_ns: "
          << dense_msd_result_.rna_diffusion_nm2_per_ns << "\n";
      // ---- Liquidity v2 -------------------------------------------------
      out << "liquidity_block_version: 3\n";
      out << "slab_normal_axis: " << dense_msd_result_.slab_normal_axis
          << "\n";
      out << "slab_normal_vector: " << dense_msd_result_.slab_normal[0] << " "
          << dense_msd_result_.slab_normal[1] << " "
          << dense_msd_result_.slab_normal[2] << "\n";
      out << "protein_lateral_diffusion_nm2_per_ns: "
          << dense_msd_result_.protein_lat_diffusion_nm2_per_ns << "\n";
      out << "protein_lateral_diffusive_regime: "
          << (dense_msd_result_.protein_lat_diffusive ? "yes" : "no") << "\n";
      out << "protein_lateral_fit_window_ns: "
          << dense_msd_result_.protein_lat_fit_start_ns << " "
          << dense_msd_result_.protein_lat_fit_end_ns << "\n";
      out << "protein_lateral_msd_exponent: "
          << dense_msd_result_.protein_lat_exponent << "\n";
      out << "rna_segment_lateral_diffusion_nm2_per_ns: "
          << dense_msd_result_.rna_lat_diffusion_nm2_per_ns << "\n";
      out << "rna_segment_lateral_diffusive_regime: "
          << (dense_msd_result_.rna_lat_diffusive ? "yes" : "no") << "\n";
      out << "rna_segment_lateral_fit_window_ns: "
          << dense_msd_result_.rna_lat_fit_start_ns << " "
          << dense_msd_result_.rna_lat_fit_end_ns << "\n";
      out << "rna_segment_lateral_msd_exponent: "
          << dense_msd_result_.rna_lat_exponent << "\n";
      out << "rna_molecules_tracked: " << dense_msd_result_.rna_molecules
          << "\n";
      for (double t : dense_msd_result_.fixed_lags_ns) {
        const auto &lg = dense_msd_result_.rna_molecule_lag_ns;
        double v = std::numeric_limits<double>::quiet_NaN();
        double vl = v;
        for (std::size_t i = 1; i < lg.size(); ++i)
          if (dense_msd_result_.rna_molecule_counts[i] > 0 &&
              std::abs(lg[i] - t) < 1e-3) {
            v = dense_msd_result_.rna_molecule_msd_nm2[i];
            vl = dense_msd_result_.rna_molecule_msd_lat_nm2[i];
          }
        const auto tag = std::to_string(static_cast<int>(std::lround(t)));
        out << "rna_molecule_msd_at_" << tag << "ns_nm2: " << v << "\n";
        out << "rna_molecule_msd_lateral_at_" << tag << "ns_nm2: " << vl
            << "\n";
      }
      for (std::size_t i = 0; i < dense_msd_result_.fixed_lags_ns.size();
           ++i) {
        const double t = dense_msd_result_.fixed_lags_ns[i];
        const auto tag = std::to_string(static_cast<int>(std::lround(t)));
        const double mp = (i < dense_msd_result_.protein_msd_fixed.size())
                              ? dense_msd_result_.protein_msd_fixed[i]
                              : std::numeric_limits<double>::quiet_NaN();
        const double mr = (i < dense_msd_result_.rna_msd_fixed.size())
                              ? dense_msd_result_.rna_msd_fixed[i]
                              : std::numeric_limits<double>::quiet_NaN();
        out << "protein_msd_at_" << tag << "ns_nm2: " << mp << "\n";
        out << "rna_msd_at_" << tag << "ns_nm2: " << mr << "\n";
        out << "protein_Dapp_at_" << tag << "ns_nm2_per_ns: " << mp / (6.0 * t)
            << "\n";
        out << "rna_Dapp_at_" << tag << "ns_nm2_per_ns: " << mr / (6.0 * t)
            << "\n";
      }
      out << "msd_units: nm2 (coordinates converted from angstrom)\n";
      out << "msd_unit_of_motion: protein=chain_com rna=backbone_segment_com\n";
      out << "rna_segment_beads: " << rna_segment_beads_ << "\n";
      out << "dense_drift_removed: "
          << (dense_msd_result_.drift_corrected ? "yes" : "no") << "\n";
      out << "dense_drift_cumulative_nm: "
          << dense_msd_result_.dense_drift_removed_nm << "\n";
      out << "protein_msd_diffusive_regime: "
          << (dense_msd_result_.protein_diffusive ? "yes" : "no") << "\n";
      out << "rna_msd_diffusive_regime: "
          << (dense_msd_result_.rna_diffusive ? "yes" : "no") << "\n";
      out << "protein_msd_loglog_exponent: "
          << dense_msd_result_.protein_msd_exponent << "\n";
      out << "rna_msd_loglog_exponent: "
          << dense_msd_result_.rna_msd_exponent << "\n";
      out << "protein_msd_fit_window_ns: "
          << dense_msd_result_.protein_fit_start_ns << " "
          << dense_msd_result_.protein_fit_end_ns << "\n";
      out << "rna_msd_fit_window_ns: "
          << dense_msd_result_.rna_fit_start_ns << " "
          << dense_msd_result_.rna_fit_end_ns << "\n";
      out << "protein_mobile_fraction_exchange: "
          << chain_dynamics_result_.protein_membership_autocorr.exchange_fraction
          << "\n";
      out << "rna_mobile_fraction_exchange: "
          << chain_dynamics_result_.rna_membership_autocorr.exchange_fraction
          << "\n";
      out << "protein_units_permanent: "
          << chain_dynamics_result_.protein_membership_autocorr.units_permanent
          << "\n";
      out << "protein_units_exchanging: "
          << chain_dynamics_result_.protein_membership_autocorr.units_exchanging
          << "\n";
      out << "protein_units_never_dense: "
          << chain_dynamics_result_.protein_membership_autocorr.units_never
          << "\n";
      out << "rna_units_permanent: "
          << chain_dynamics_result_.rna_membership_autocorr.units_permanent
          << "\n";
      out << "rna_units_exchanging: "
          << chain_dynamics_result_.rna_membership_autocorr.units_exchanging
          << "\n";
      out << "rna_units_never_dense: "
          << chain_dynamics_result_.rna_membership_autocorr.units_never
          << "\n";
      out << "protein_pooled_variance_amplitude: "
          << chain_dynamics_result_.protein_membership_autocorr.mobile_fraction
          << "\n";
      out << "rna_pooled_variance_amplitude: "
          << chain_dynamics_result_.rna_membership_autocorr.mobile_fraction
          << "\n";
      out << "protein_immobile_plateau: "
          << chain_dynamics_result_.protein_membership_autocorr.pooled_plateau
          << "\n";
      out << "rna_immobile_plateau: "
          << chain_dynamics_result_.rna_membership_autocorr.pooled_plateau
          << "\n";
      out << "protein_pooled_tau_ns: "
          << chain_dynamics_result_.protein_membership_autocorr.pooled_tau_ns
          << "\n";
      out << "rna_pooled_tau_ns: "
          << chain_dynamics_result_.rna_membership_autocorr.pooled_tau_ns
          << "\n";
      out << "protein_dense_occupancy: "
          << chain_dynamics_result_.protein_membership_autocorr.occupancy
          << "\n";
      out << "rna_dense_occupancy: "
          << chain_dynamics_result_.rna_membership_autocorr.occupancy << "\n";
      out << "protein_units: "
          << chain_dynamics_result_.protein_membership_autocorr.units << "\n";
      out << "rna_units: "
          << chain_dynamics_result_.rna_membership_autocorr.units << "\n";
      out << "mobile_displacement_threshold_nm: "
          << dense_msd_result_.mobile_displacement_nm << "\n";
      out << "mobile_displacement_lag_ns: " << dense_msd_result_.mobile_lag_ns
          << "\n";
      out << "protein_mobile_fraction_displacement: "
          << dense_msd_result_.protein_mobile_fraction_displacement << "\n";
      out << "rna_mobile_fraction_displacement: "
          << dense_msd_result_.rna_mobile_fraction_displacement << "\n";
      out << "protein_residence_p90_ns: "
          << chain_dynamics_result_.protein_residence.p90_complete_ns << "\n";
      out << "rna_residence_p90_ns: "
          << chain_dynamics_result_.rna_residence.p90_complete_ns << "\n";
      out << "protein_residence_max_ns: "
          << chain_dynamics_result_.protein_residence.max_complete_ns << "\n";
      out << "rna_residence_max_ns: "
          << chain_dynamics_result_.rna_residence.max_complete_ns << "\n";
      out << "protein_residence_episodes_total: "
          << chain_dynamics_result_.protein_residence.total_episodes << "\n";
      out << "protein_residence_episodes_complete: "
          << chain_dynamics_result_.protein_residence.complete_episodes
          << "\n";
      out << "rna_residence_episodes_total: "
          << chain_dynamics_result_.rna_residence.total_episodes << "\n";
      out << "rna_residence_episodes_complete: "
          << chain_dynamics_result_.rna_residence.complete_episodes << "\n";
      if (contact_result) {
        // These distributions are extremely heavy tailed (a mean far above
        // the median means one long-lived contact dominates), so the mean
        // alone is not reportable. Emit the order statistics beside it.
        out << "sticker_contact_mean_lifetime_ns: "
            << contact_result->sticker_contact_lifetime.mean_ns << "\n";
        out << "sticker_contact_median_lifetime_ns: "
            << contact_result->sticker_contact_lifetime.median_ns << "\n";
        out << "sticker_contact_p90_lifetime_ns: "
            << contact_result->sticker_contact_lifetime.p90_ns << "\n";
        out << "sticker_contact_max_lifetime_ns: "
            << contact_result->sticker_contact_lifetime.max_ns << "\n";
        out << "sticker_contact_events: "
            << contact_result->sticker_contact_lifetime.total_events << "\n";
        out << "protein_rna_contact_mean_lifetime_ns: "
            << contact_result->protein_rna_contact_lifetime.mean_ns << "\n";
        out << "protein_rna_contact_median_lifetime_ns: "
            << contact_result->protein_rna_contact_lifetime.median_ns << "\n";
        out << "protein_rna_contact_p90_lifetime_ns: "
            << contact_result->protein_rna_contact_lifetime.p90_ns << "\n";
        out << "protein_rna_contact_max_lifetime_ns: "
            << contact_result->protein_rna_contact_lifetime.max_ns << "\n";
        out << "protein_rna_contact_events: "
            << contact_result->protein_rna_contact_lifetime.total_events
            << "\n";
        // Liquidity v3: censoring and uncensored order statistics.
        for (const auto &pr :
             {std::make_pair(std::string("sticker_contact"),
                             &contact_result->sticker_contact_lifetime),
              std::make_pair(std::string("protein_rna_contact"),
                             &contact_result->protein_rna_contact_lifetime)}) {
          const auto &d = *pr.second;
          out << pr.first << "_window_ns: " << d.window_ns << "\n";
          out << pr.first << "_right_censored_events: "
              << d.right_censored_events << "\n";
          out << pr.first << "_full_window_events: " << d.full_window_events
              << "\n";
          out << pr.first << "_uncensored_events: " << d.uncensored_events
              << "\n";
          out << pr.first << "_uncensored_mean_ns: " << d.uncensored_mean_ns
              << "\n";
          out << pr.first << "_uncensored_median_ns: "
              << d.uncensored_median_ns << "\n";
          out << pr.first << "_uncensored_p90_ns: " << d.uncensored_p90_ns
              << "\n";
          out << pr.first << "_uncensored_p99_ns: " << d.uncensored_p99_ns
              << "\n";
        }
        out << "protein_neighbor_persistence: "
            << contact_result->protein_neighbor_persistence << "\n";
        out << "rna_neighbor_persistence: "
            << contact_result->rna_neighbor_persistence << "\n";
        out << "global_neighbor_persistence: "
            << contact_result->global_neighbor_persistence << "\n";
        out << "protein_rewiring_rate_per_ns: "
            << contact_result->protein_rewiring_rate_per_ns << "\n";
        out << "rna_rewiring_rate_per_ns: "
            << contact_result->rna_rewiring_rate_per_ns << "\n";
        out << "global_rewiring_rate_per_ns: "
            << contact_result->global_rewiring_rate_per_ns << "\n";
        out << "interface_new_contacts_per_ns: "
            << contact_result->interface_new_contacts_per_ns << "\n";
        out << "interface_lost_contacts_per_ns: "
            << contact_result->interface_lost_contacts_per_ns << "\n";
        out << "interface_net_change_per_ns: "
            << contact_result->interface_net_change_per_ns << "\n";
      } else {
        out << "sticker_contact_mean_lifetime_ns: NA\n";
        out << "sticker_contact_median_lifetime_ns: NA\n";
        out << "sticker_contact_p90_lifetime_ns: NA\n";
        out << "sticker_contact_max_lifetime_ns: NA\n";
        out << "sticker_contact_events: NA\n";
        out << "protein_rna_contact_mean_lifetime_ns: NA\n";
        out << "protein_rna_contact_median_lifetime_ns: NA\n";
        out << "protein_rna_contact_p90_lifetime_ns: NA\n";
        out << "protein_rna_contact_max_lifetime_ns: NA\n";
        out << "protein_rna_contact_events: NA\n";
        out << "protein_neighbor_persistence: NA\n";
        out << "rna_neighbor_persistence: NA\n";
        out << "global_neighbor_persistence: NA\n";
        out << "protein_rewiring_rate_per_ns: NA\n";
        out << "rna_rewiring_rate_per_ns: NA\n";
        out << "global_rewiring_rate_per_ns: NA\n";
        out << "interface_new_contacts_per_ns: NA\n";
        out << "interface_lost_contacts_per_ns: NA\n";
        out << "interface_net_change_per_ns: NA\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_liquidity_summary_filename());
    }
  }

  TitanUI::log_kv(log, "LLPS Dynamics Protein tau",
                  std::to_string(
                      chain_dynamics_result_.protein_membership_autocorr.tau_ns));
  TitanUI::log_kv(log, "LLPS Dynamics RNA tau",
                  std::to_string(
                      chain_dynamics_result_.rna_membership_autocorr.tau_ns));
}

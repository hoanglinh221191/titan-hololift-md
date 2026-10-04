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
    out << "# lag_ns mean_connected_autocorr std_connected_autocorr "
           "contributing_chains\n";
    for (std::size_t i = 0; i < result.lag_ns.size(); ++i) {
      out << result.lag_ns[i] << " " << result.mean[i] << " " << result.stddev[i]
          << " " << result.contributing_chains[i] << "\n";
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
             "MatchedJaccard,Event\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : lineage_rows_) {
        out << row.time_ns << "," << row.frame_cluster_id << ","
            << row.protein_chains << "," << row.rna_chains << ","
            << join_ints(row.matched_prev_clusters) << ","
            << join_doubles(row.matched_jaccard) << "," << row.event << "\n";
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
      out << "lineage_overlap_threshold: " << lineage_overlap_threshold_ << "\n";
      out << "min_lineage_cluster_size: " << min_lineage_cluster_size_ << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_droplet_genealogy_summary_filename());
    }
  }

  {
    auto write_species = [](std::ofstream &out, const std::string &label,
                            const LLPSAutocorrSpeciesResult &autocorr,
                            const LLPSEpisodeSummary &residence,
                            double entry_rate, double exit_rate) {
      const double tau = autocorr.tau_ns;
      const double recovery_half_time =
          (tau > 0.0) ? tau * std::log(2.0) : 0.0;
      const double turnover_rate = entry_rate + exit_rate;
      const double mobile_fraction_proxy =
          (residence.total_episodes > 0)
              ? static_cast<double>(residence.complete_episodes) /
                    static_cast<double>(residence.total_episodes)
              : 0.0;
      out << label << "," << tau << "," << recovery_half_time << ","
          << residence.mean_complete_ns << "," << residence.median_complete_ns
          << "," << residence.p90_complete_ns << "," << entry_rate << ","
          << exit_rate << "," << turnover_rate << ","
          << mobile_fraction_proxy << "\n";
    };
    std::ofstream out(info_output.get_llps_frap_recovery_proxy_filename());
    if (out.is_open()) {
      out << "Species,MembershipTau_ns,RecoveryHalfTimeProxy_ns,"
             "MeanResidence_ns,MedianResidence_ns,P90Residence_ns,"
             "EntryRate_per_ns,ExitRate_per_ns,TurnoverRate_per_ns,"
             "MobileFractionProxy\n";
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

  auto write_msd = [](const std::string &filename, const std::vector<double> &lags,
                      const std::vector<double> &msd,
                      const std::vector<uint64_t> &counts,
                      double diffusion, double fit_start, double fit_end) {
    std::ofstream out(filename);
    if (!out.is_open())
      return false;
    out << "# diffusion_nm2_per_ns " << std::fixed << std::setprecision(6)
        << diffusion << "\n";
    out << "# fit_window_ns " << fit_start << " " << fit_end << "\n";
    out << "# lag_ns msd_nm2 count\n";
    for (std::size_t i = 0; i < lags.size(); ++i)
      out << lags[i] << " " << msd[i] << " " << counts[i] << "\n";
    return true;
  };

  if (write_msd(info_output.get_llps_dense_msd_protein_filename(),
                dense_msd_result_.lag_ns, dense_msd_result_.protein_msd_nm2,
                dense_msd_result_.protein_counts,
                dense_msd_result_.protein_diffusion_nm2_per_ns,
                dense_msd_result_.protein_fit_start_ns,
                dense_msd_result_.protein_fit_end_ns)) {
    TitanUI::print_info("  Written: " +
                        info_output.get_llps_dense_msd_protein_filename());
  }
  if (write_msd(info_output.get_llps_dense_msd_rna_filename(),
                dense_msd_result_.lag_ns, dense_msd_result_.rna_msd_nm2,
                dense_msd_result_.rna_counts,
                dense_msd_result_.rna_diffusion_nm2_per_ns,
                dense_msd_result_.rna_fit_start_ns,
                dense_msd_result_.rna_fit_end_ns)) {
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
      if (contact_result) {
        out << "sticker_contact_mean_lifetime_ns: "
            << contact_result->sticker_contact_lifetime.mean_ns << "\n";
        out << "protein_rna_contact_mean_lifetime_ns: "
            << contact_result->protein_rna_contact_lifetime.mean_ns << "\n";
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
        out << "protein_rna_contact_mean_lifetime_ns: NA\n";
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

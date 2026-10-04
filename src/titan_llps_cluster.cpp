// titan_llps_cluster.cpp - DBSCAN Phase Separation Analysis
#include "titan_llps_cluster.h"
#include "titan_exact_periodic_box.h"
#include "MOL.h"
#include "MOL_info.h"
#include "box_type.h"
#include "info.h"
#include "titan_executor.h"
#include "titan_energy_decomposition.h"
#include "titan_llps.h"
#include "titan_llps_stats.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
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

// ============================================================================
// encode_hybrid36
// ============================================================================
std::string TitanLLPSCluster::encode_hybrid36(int n) {
  constexpr int dec_max = 9999;
  if (n <= dec_max) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%4d", (n < 0 ? 0 : n));
    return std::string(buf);
  }
  static const char upper[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  static const char lower[] = "abcdefghijklmnopqrstuvwxyz";
  int rem = n - (dec_max + 1);
  constexpr int base26_max = 26 * 26 * 26 * 26;
  const char *alpha = (rem < base26_max) ? upper : lower;
  if (rem >= base26_max)
    rem -= base26_max;
  char result[8];
  result[3] = alpha[rem % 26];
  rem /= 26;
  result[2] = alpha[rem % 26];
  rem /= 26;
  result[1] = alpha[rem % 26];
  rem /= 26;
  result[0] = alpha[rem % 26];
  result[4] = '\0';
  return std::string(result);
}

std::string TitanLLPSCluster::species_label(int species) {
  return (species == 0) ? "protein" : "rna";
}

uint64_t TitanLLPSCluster::pack_chain_key(int species, int raw_chain_id) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(species)) << 32) |
         static_cast<uint32_t>(raw_chain_id);
}

void TitanLLPSCluster::RunningStats::add(double value) {
  ++n;
  sum += value;
  sum_sq += value * value;
}

double TitanLLPSCluster::RunningStats::mean() const {
  return (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
}

double TitanLLPSCluster::RunningStats::stddev() const {
  if (n < 2)
    return 0.0;
  const double m = mean();
  const double var = sum_sq / static_cast<double>(n) - m * m;
  return std::sqrt(std::max(0.0, var));
}

// ============================================================================
// Lifecycle
// ============================================================================
TitanLLPSCluster::~TitanLLPSCluster() {
  if (vmd_out_.is_open())
    vmd_out_.close();
  if (ts_out_.is_open())
    ts_out_.close();
  if (rna_binding_energy_trace_out_.is_open())
    rna_binding_energy_trace_out_.close();
  if (prna_pair_energy_trace_out_.is_open())
    prna_pair_energy_trace_out_.close();
  if (rna_topology_trace_out_.is_open())
    rna_topology_trace_out_.close();
  if (rna_bridge_trace_out_.is_open())
    rna_bridge_trace_out_.close();
  if (rna_clamp_trace_out_.is_open())
    rna_clamp_trace_out_.close();
  if (membership_trace_out_.is_open())
    membership_trace_out_.close();
  if (rna_compaction_trace_out_.is_open())
    rna_compaction_trace_out_.close();
  if (droplet_shape_out_.is_open())
    droplet_shape_out_.close();
}

void TitanLLPSCluster::init(double eps_angstrom, int min_pts,
                            const std::string &out_pdb_name,
                            const std::string &out_csv_name) {
  eps_ = eps_angstrom;
  eps_sq_ = eps_ * eps_;
  min_pts_ = min_pts;
  frame_count_ = 0;
  timeseries_.clear();
  streaming_csv_ = false;
  if (vmd_out_.is_open())
    vmd_out_.close();
  if (rna_binding_energy_trace_out_.is_open())
    rna_binding_energy_trace_out_.close();
  if (prna_pair_energy_trace_out_.is_open())
    prna_pair_energy_trace_out_.close();
  if (rna_topology_trace_out_.is_open())
    rna_topology_trace_out_.close();
  if (rna_bridge_trace_out_.is_open())
    rna_bridge_trace_out_.close();
  if (rna_clamp_trace_out_.is_open())
    rna_clamp_trace_out_.close();
  if (membership_trace_out_.is_open())
    membership_trace_out_.close();
  if (rna_compaction_trace_out_.is_open())
    rna_compaction_trace_out_.close();
  if (droplet_shape_out_.is_open())
    droplet_shape_out_.close();
  membership_trace_streaming_ = false;
  rna_compaction_trace_streaming_ = false;
  droplet_shape_streaming_ = false;
  if (!out_pdb_name.empty())
    vmd_out_.open(out_pdb_name);

  rna_compaction_samples_.clear();
  droplet_shape_rows_.clear();
  rna_compaction_topology_cache_ = {};
  rna_chain_compaction_stats_.clear();
  rna_binding_bin_stats_.clear();
  rna_energy_bin_stats_.clear();
  rna_compaction_trace_rows_.clear();
  rna_partner_state_to_id_.clear();
  rna_partner_state_labels_.clear();
  rna_partner_state_labels_.push_back("unbound");
  rna_information_rg_edges_.clear();
  rna_information_longrange_edges_.clear();
  rna_information_summary_rows_.clear();
  rna_information_per_chain_rows_.clear();
  rna_compaction_state_histogram_rows_.clear();
  rna_partner_state_histogram_rows_.clear();
  rna_transition_information_rows_.clear();
  rna_binding_compaction_information_rows_.clear();
  rna_information_primary_scope_ = "none";

  if (!out_csv_name.empty()) {
    if (ts_out_.is_open())
      ts_out_.close();
    ts_out_.open(out_csv_name);
    if (ts_out_.is_open()) {
      ts_out_ << "Time_ps,TotalBeads,NumCondensates,"
                 "GiantCluster_ProChains,GiantCluster_RnaChains,NoiseBeads,"
                 "Rg_Pro_Dilute,Rg_Pro_Dense,Rg_RNA_Dilute,Rg_RNA_Dense\n";
      streaming_csv_ = true;
      TitanUI::print_info("[LLPS-DBSCAN] Streaming CSV -> " + out_csv_name);
    } else {
      TitanUI::print_warning("[LLPS-DBSCAN] Cannot open CSV: " + out_csv_name);
    }
  }

  TitanUI::print_info("[LLPS-DBSCAN] eps=" + std::to_string(eps_) +
                      " A, min_pts=" + std::to_string(min_pts_));
  if (!out_pdb_name.empty())
    TitanUI::print_info("[LLPS-DBSCAN] Streaming PDB -> " + out_pdb_name);
}

void TitanLLPSCluster::configure_dynamics(bool enabled, bool detailed,
                                          int max_lag_frames,
                                          double lineage_overlap_threshold,
                                          int min_lineage_cluster_size,
                                          double msd_fit_start_ns,
                                          double msd_fit_end_ns) {
  dynamics_enabled_ = enabled;
  dynamics_detailed_ = detailed;
  dynamics_max_lag_frames_ = std::max(1, max_lag_frames);
  dynamics_ring_cap_ = dynamics_max_lag_frames_ + 1;
  lineage_overlap_threshold_ = lineage_overlap_threshold;
  min_lineage_cluster_size_ = std::max(1, min_lineage_cluster_size);
  msd_fit_start_ns_ = msd_fit_start_ns;
  msd_fit_end_ns_ = msd_fit_end_ns;

  if (!dynamics_enabled_)
    return;

  dynamics_ring_chain_capacity_ = 0;
  membership_lag_chain_capacity_ = 0;
  frame_time_ring_.assign(static_cast<std::size_t>(dynamics_ring_cap_), 0.0);
  frame_present_ring_.clear();
  frame_inside_ring_.clear();
  frame_episode_ring_.clear();
  frame_unwrapped_x_ring_.clear();
  frame_unwrapped_y_ring_.clear();
  frame_unwrapped_z_ring_.clear();

  membership_joint_inside_.clear();
  membership_pair_counts_.clear();
  protein_autocorr_lag_time_sum_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0.0);
  rna_autocorr_lag_time_sum_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0.0);
  protein_autocorr_lag_time_count_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
  rna_autocorr_lag_time_count_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
  msd_sum_protein_.assign(static_cast<std::size_t>(dynamics_max_lag_frames_ + 1),
                          0.0);
  msd_sum_rna_.assign(static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0.0);
  msd_count_protein_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
  msd_count_rna_.assign(static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
  msd_time_sum_protein_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0.0);
  msd_time_sum_rna_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0.0);
  msd_time_count_protein_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
  msd_time_count_rna_.assign(
      static_cast<std::size_t>(dynamics_max_lag_frames_ + 1), 0);
}

void TitanLLPSCluster::configure_trace_outputs(const info &info_output) {
  membership_trace_streaming_ = false;
  rna_compaction_trace_streaming_ = false;
  droplet_shape_streaming_ = false;
  if (membership_trace_out_.is_open())
    membership_trace_out_.close();
  if (rna_compaction_trace_out_.is_open())
    rna_compaction_trace_out_.close();
  if (droplet_shape_out_.is_open())
    droplet_shape_out_.close();

  droplet_shape_out_.open(info_output.get_llps_droplet_shape_timeseries_filename());
  if (droplet_shape_out_.is_open()) {
    droplet_shape_out_
        << "Time_ps,Frame,ClusterID,Nodes,ProteinNodes,RnaNodes,Rg_A,"
           "Asphericity_A2,Acylindricity_A2,RelativeShapeAnisotropy,"
           "SphericityProxy\n";
    droplet_shape_out_ << std::fixed << std::setprecision(6);
    droplet_shape_streaming_ = true;
    droplet_shape_rows_.clear();
  } else {
    TitanUI::print_warning("[LLPS-DBSCAN] Cannot open " +
                           info_output.get_llps_droplet_shape_timeseries_filename());
  }

  if (dynamics_enabled_ && dynamics_detailed_) {
    membership_trace_out_.open(
        info_output.get_llps_chain_membership_trace_filename());
    if (membership_trace_out_.is_open()) {
      membership_trace_out_
          << "Time_ns,Species,ChainID,EpisodeID,InsideDense,Event\n";
      membership_trace_out_ << std::fixed << std::setprecision(6);
      membership_trace_streaming_ = true;
      membership_trace_rows_.clear();
    } else {
      TitanUI::print_warning("[LLPS-DBSCAN] Cannot open " +
                             info_output
                                 .get_llps_chain_membership_trace_filename());
    }
  }

  if (rna_compaction_enabled_ && rna_compaction_detailed_) {
    rna_compaction_trace_out_.open(
        info_output.get_llps_rna_compaction_trace_filename());
    if (rna_compaction_trace_out_.is_open()) {
      rna_compaction_trace_out_
          << "Time_ns,Frame,RNAChainID,InsideDense,ClusterID,"
             "BoundProteinChains,ProteinRNABeadContacts,Rg_nm,EndToEnd_nm,"
             "Asphericity_nm2,NormalizedAsphericity,Acylindricity_nm2,"
             "RelativeShapeAnisotropy,AspectRatio,IntrachainContacts,"
             "LongRangeIntrachainContacts,LongRangeContactFraction\n";
      rna_compaction_trace_out_ << std::fixed << std::setprecision(6);
      rna_compaction_trace_streaming_ = true;
      rna_compaction_trace_rows_.clear();
    } else {
      TitanUI::print_warning("[LLPS-RNA] Cannot open " +
                             info_output
                                 .get_llps_rna_compaction_trace_filename());
    }
  }
}

void TitanLLPSCluster::configure_rna_compaction(
    const std::string &mode, bool dynamics_enabled, bool detailed, bool has_rna,
    const std::string &sticker_beads, float protein_rna_cutoff_nm,
    float intra_cutoff_nm, int longrange_min) {
  std::string normalized = mode;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  const bool explicitly_on =
      normalized == "true" || normalized == "yes" || normalized == "on" ||
      normalized == "1";
  const bool explicitly_off =
      normalized == "false" || normalized == "no" || normalized == "off" ||
      normalized == "0";
  rna_compaction_enabled_ =
      explicitly_on || (!explicitly_off && dynamics_enabled && has_rna);
  rna_compaction_detailed_ = rna_compaction_enabled_ && detailed;
  rna_compaction_prna_cutoff_angstrom_ =
      10.0 * static_cast<double>(std::max(0.0f, protein_rna_cutoff_nm));
  rna_compaction_intra_cutoff_angstrom_ =
      10.0 * static_cast<double>(std::max(0.0f, intra_cutoff_nm));
  rna_compaction_longrange_min_ = std::max(0, longrange_min);
  rna_compaction_sticker_names_ = split_csv_set(sticker_beads);

  if (!rna_compaction_enabled_)
    return;
  if (!has_rna) {
    rna_compaction_enabled_ = false;
    TitanUI::print_warning(
        "[LLPS-RNA] RNA compaction requested but no RNA molecule was detected; "
        "skipping RNA compaction outputs.");
    return;
  }
  if (rna_compaction_prna_cutoff_angstrom_ <= 0.0 ||
      rna_compaction_intra_cutoff_angstrom_ <= 0.0) {
    rna_compaction_enabled_ = false;
    TitanUI::print_warning(
        "[LLPS-RNA] RNA compaction cutoff is non-positive; skipping RNA "
        "compaction outputs.");
    return;
  }

  TitanUI::print_info("[LLPS-RNA] RNA compaction vs protein binding enabled.");
}

void TitanLLPSCluster::configure_rna_information(const std::string &mode,
                                                 bool dynamics_enabled,
                                                 bool detailed, bool has_rna,
                                                 int rg_bins,
                                                 int longrange_bins) {
  std::string normalized = mode;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  const bool explicitly_on =
      normalized == "true" || normalized == "yes" || normalized == "on" ||
      normalized == "1";
  const bool explicitly_off =
      normalized == "false" || normalized == "no" || normalized == "off" ||
      normalized == "0";
  const bool prerequisites_met =
      dynamics_enabled && rna_compaction_enabled_ && has_rna;

  rna_information_rg_bins_ = std::max(1, rg_bins);
  rna_information_longrange_bins_ = std::max(1, longrange_bins);
  rna_information_enabled_ = !explicitly_off && prerequisites_met;
  rna_information_detailed_ = rna_information_enabled_ && detailed;

  if (!rna_information_enabled_) {
    if (explicitly_on && !prerequisites_met) {
      TitanUI::print_warning(
          "[LLPS-RNA-Info] RNA information metrics requested but "
          "requirements are not met; skipping RNA information outputs.");
    }
    return;
  }

  TitanUI::print_info(
      "[LLPS-RNA-Info] RNA information-theoretic entropy proxies enabled.");
}

void TitanLLPSCluster::configure_rna_topology(
    const std::string &mode, bool dynamics_enabled, bool detailed, bool has_rna,
    float protein_protein_cutoff_nm, float protein_rna_cutoff_nm,
    int longrange_min, float bridge_geom_min_nm, int patch_gap_max,
    bool clamp_require_loop, const info &info_output) {
  std::string normalized = mode;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  const bool explicitly_on =
      normalized == "true" || normalized == "yes" || normalized == "on" ||
      normalized == "1";
  const bool explicitly_off =
      normalized == "false" || normalized == "no" || normalized == "off" ||
      normalized == "0";
  const bool prerequisites_met =
      dynamics_enabled && rna_compaction_enabled_ && has_rna;

  rna_topology_enabled_ = explicitly_on || (!explicitly_off && prerequisites_met);
  rna_topology_detailed_ = rna_topology_enabled_ && detailed;
  rna_topology_patch_gap_max_ = std::max(0, patch_gap_max);
  rna_topology_clamp_require_loop_ = clamp_require_loop;
  rna_topology_pp_cutoff_angstrom_ =
      10.0 * static_cast<double>(std::max(0.0f, protein_protein_cutoff_nm));
  rna_topology_bridge_geom_min_angstrom_ =
      (bridge_geom_min_nm > 0.0f)
          ? 10.0 * static_cast<double>(bridge_geom_min_nm)
          : 20.0 * static_cast<double>(std::max(0.0f, protein_rna_cutoff_nm));
  if (rna_topology_bridge_geom_min_angstrom_ <= 0.0)
    rna_topology_bridge_geom_min_angstrom_ = 2.0 * rna_compaction_prna_cutoff_angstrom_;

  if (rna_topology_trace_out_.is_open())
    rna_topology_trace_out_.close();
  if (rna_bridge_trace_out_.is_open())
    rna_bridge_trace_out_.close();
  if (rna_clamp_trace_out_.is_open())
    rna_clamp_trace_out_.close();

  if (!rna_topology_enabled_) {
    if (explicitly_on && !prerequisites_met) {
      TitanUI::print_warning(
          "[LLPS-RNA-Topology] RNA topology requested but requirements are "
          "not met; skipping topology outputs.");
    }
    return;
  }

  if (rna_topology_detailed_) {
    rna_topology_trace_out_.open(info_output.get_llps_rna_topology_trace_filename());
    if (rna_topology_trace_out_.is_open()) {
      rna_topology_trace_out_
          << "Time_ns,Frame,RNAChainID,InsideDense,ClusterID,BoundProteinChains,"
             "TopologyState,AnyBridge,NumBridgeProteins,AnyClamp,NumClampPairs,"
             "BridgeResidueSpanMax,BridgeGeomSpanMax_nm,"
             "LongRangeIntrachainContacts,ClampSupportedLongRangeContacts,"
             "BridgeTotal_kJmol,NonBridgeTotal_kJmol,"
             "BridgeAttractionStrength_kJmol,NonBridgeAttractionStrength_kJmol,"
             "PRNAAttractionStrength_kJmol,LocalPPStickerContacts,"
             "LocalPPElec_kJmol,LocalPPVdw_kJmol,LocalPPTotal_kJmol,"
             "LocalPPAttractionStrength_kJmol,LocalPPtoPRNAContactRatio,"
             "LocalPPtoPRNAAttractionRatio,LocalPPtoBridgeAttractionRatio\n";
    } else {
      TitanUI::print_warning("[LLPS-RNA-Topology] Cannot open " +
                             info_output.get_llps_rna_topology_trace_filename());
    }

    rna_bridge_trace_out_.open(info_output.get_llps_rna_bridge_trace_filename());
    if (rna_bridge_trace_out_.is_open()) {
      rna_bridge_trace_out_
          << "Time_ns,Frame,RNAChainID,ProteinChainID,PatchCount,BridgeCount,"
             "BridgeResidueSpanMax,BridgeGeomSpanMax_nm,BridgeContactCount,"
             "BridgeTotal_kJmol,BridgeAttractionStrength_kJmol,"
             "SupportsLongRangeContact\n";
    } else {
      TitanUI::print_warning("[LLPS-RNA-Topology] Cannot open " +
                             info_output.get_llps_rna_bridge_trace_filename());
    }

    rna_clamp_trace_out_.open(info_output.get_llps_rna_clamp_trace_filename());
    if (rna_clamp_trace_out_.is_open()) {
      rna_clamp_trace_out_
          << "Time_ns,Frame,RNAChainID,ProteinChainID_A,ProteinChainID_B,"
             "ClampCount,ClampResidueSpanMax,ClampGeomSpanMax_nm,"
             "ClampSupportedLongRangeContacts,ClampTotal_kJmol,"
             "ClampAttractionStrength_kJmol\n";
    } else {
      TitanUI::print_warning("[LLPS-RNA-Topology] Cannot open " +
                             info_output.get_llps_rna_clamp_trace_filename());
    }
  }

  TitanUI::print_info(
      "[LLPS-RNA-Topology] RNA bridge/clamp topology analysis enabled.");
}

void TitanLLPSCluster::configure_energy_correlation(
    const std::string &mode, bool dynamics_enabled, bool has_rna, bool nocal,
    const EnergyDecompositionProvider *energy_provider, bool detailed,
    const info &info_output) {
  std::string normalized = mode;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  const bool explicitly_on =
      normalized == "true" || normalized == "yes" || normalized == "on" ||
      normalized == "1";
  const bool explicitly_off =
      normalized == "false" || normalized == "no" || normalized == "off" ||
      normalized == "0";

  energy_provider_ = energy_provider;
  energy_correlation_enabled_ =
      explicitly_on || (!explicitly_off && dynamics_enabled && has_rna &&
                        rna_compaction_enabled_ && !nocal &&
                        energy_provider_ != nullptr &&
                        energy_provider_->ready());
  energy_correlation_detailed_ = energy_correlation_enabled_ && detailed;

  if (!energy_correlation_enabled_) {
    if (explicitly_on) {
      TitanUI::print_warning(
          "[LLPS-Energy] Protein-RNA energy correlation requested but "
          "requirements are not met; skipping energy-correlation outputs.");
    }
    return;
  }

  if (rna_binding_energy_trace_out_.is_open())
    rna_binding_energy_trace_out_.close();
  rna_binding_energy_trace_out_.open(
      info_output.get_llps_rna_binding_energy_trace_filename());
  if (rna_binding_energy_trace_out_.is_open()) {
    rna_binding_energy_trace_out_
        << "Time_ns,Frame,RNAChainID,InsideDense,ClusterID,"
           "BoundProteinChains,ProteinRNABeadContacts,Rg_nm,EndToEnd_nm,"
           "RelativeShapeAnisotropy,LongRangeContactFraction,"
           "Eelec_ProteinRNA_kJmol,Evdw_ProteinRNA_kJmol,"
           "Etotal_ProteinRNA_kJmol\n";
  } else {
    TitanUI::print_warning("[LLPS-Energy] Cannot open " +
                           info_output.get_llps_rna_binding_energy_trace_filename());
  }

  if (energy_correlation_detailed_) {
    if (prna_pair_energy_trace_out_.is_open())
      prna_pair_energy_trace_out_.close();
    prna_pair_energy_trace_out_.open(
        info_output.get_llps_prna_pair_energy_trace_filename());
    if (prna_pair_energy_trace_out_.is_open()) {
      prna_pair_energy_trace_out_
          << "Time_ns,Frame,ProteinChainID,RNAChainID,ContactCount,"
             "Eelec_kJmol,Evdw_kJmol,Etotal_kJmol,RNA_Rg_nm,"
             "RNA_LongRangeContactFraction\n";
    } else {
      TitanUI::print_warning("[LLPS-Energy] Cannot open " +
                             info_output.get_llps_prna_pair_energy_trace_filename());
    }
  }

  TitanUI::print_info(
      "[LLPS-Energy] Protein-RNA energy vs RNA compaction enabled.");
}

int TitanLLPSCluster::register_rna_partner_state(
    const std::vector<int> &protein_chain_ids) {
  if (protein_chain_ids.empty())
    return 0;
  const std::string key = join_ints_pipe(protein_chain_ids);
  const auto hit = rna_partner_state_to_id_.find(key);
  if (hit != rna_partner_state_to_id_.end())
    return hit->second;
  const int id = static_cast<int>(rna_partner_state_labels_.size());
  rna_partner_state_to_id_[key] = id;
  rna_partner_state_labels_.push_back(key);
  return id;
}

// ============================================================================
// Extract / DBSCAN
// ============================================================================


// ============================================================================
// Process frame
// ============================================================================
void TitanLLPSCluster::process_frame(TitanExecutor &executor, const MOL &pro,
                                     const std::vector<mol_info> &pro_info,
                                     const MOL &lig,
                                     const std::vector<mol_info> &lig_info,
                                     double current_time_ps, int frame_idx) {
  const auto extract_begin = std::chrono::steady_clock::now();
  auto nodes = extract_nodes(pro, pro_info, lig, lig_info);
  perf_.extract_seconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - extract_begin)
                               .count();
  if (nodes.empty())
    return;
  perf_.frames_processed += 1;
  perf_.node_count_accum += static_cast<uint64_t>(nodes.size());

  const auto dbscan_begin = std::chrono::steady_clock::now();
  run_dbscan(executor, nodes, pro.Box);
  perf_.dbscan_seconds += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - dbscan_begin)
                              .count();

  const auto summary_begin = std::chrono::steady_clock::now();
  int total_pro_chains = 0;
  {
    std::vector<int> all_protein;
    all_protein.reserve(nodes.size());
    for (const auto &node : nodes)
      if (node.type == 1)
        all_protein.push_back(node.chain_idx);
    std::sort(all_protein.begin(), all_protein.end());
    all_protein.erase(std::unique(all_protein.begin(), all_protein.end()),
                      all_protein.end());
    total_pro_chains = static_cast<int>(all_protein.size());
  }

  int max_cid = 0;
  for (const auto &node : nodes)
    if (node.cluster_id > max_cid)
      max_cid = node.cluster_id;

  std::vector<std::vector<int>> cluster_pro(static_cast<std::size_t>(max_cid + 1));
  std::vector<std::vector<int>> cluster_rna(static_cast<std::size_t>(max_cid + 1));
  int noise_cnt = 0;
  for (const auto &node : nodes) {
    if (node.cluster_id > 0) {
      if (node.type == 1)
        cluster_pro[static_cast<std::size_t>(node.cluster_id)].push_back(
            node.chain_idx);
      else
        cluster_rna[static_cast<std::size_t>(node.cluster_id)].push_back(
            node.chain_idx);
    } else {
      ++noise_cnt;
    }
  }

  int largest_pro = 0;
  int largest_rna = 0;
  int n_valid = 0;
  int giant_cluster_id = -1;
  for (int cid = 1; cid <= max_cid; ++cid) {
    auto &pro_chains = cluster_pro[static_cast<std::size_t>(cid)];
    auto &rna_chains = cluster_rna[static_cast<std::size_t>(cid)];
    std::sort(pro_chains.begin(), pro_chains.end());
    pro_chains.erase(std::unique(pro_chains.begin(), pro_chains.end()),
                     pro_chains.end());
    std::sort(rna_chains.begin(), rna_chains.end());
    rna_chains.erase(std::unique(rna_chains.begin(), rna_chains.end()),
                     rna_chains.end());
    const int np = static_cast<int>(pro_chains.size());
    if (np > 1) {
      ++n_valid;
      if (np > largest_pro) {
        largest_pro = np;
        largest_rna = static_cast<int>(rna_chains.size());
        giant_cluster_id = cid;
      }
    }
  }
  if (largest_pro < std::max(2, total_pro_chains / 20))
    giant_cluster_id = -1;
  perf_.cluster_summary_seconds += std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() -
                                       summary_begin)
                                       .count();

  const int N = static_cast<int>(nodes.size());
  std::vector<bool> bead_is_dense(static_cast<std::size_t>(N), false);
  std::vector<double> bfactor(static_cast<std::size_t>(N), 0.0);
  std::vector<double> occupancy(static_cast<std::size_t>(N), 0.0);
  if (fuzzy_enabled_ && giant_cluster_id > 0) {
    const auto fuzzy_begin = std::chrono::steady_clock::now();
    fuzzy_refine(executor, nodes, pro.Box, giant_cluster_id, bead_is_dense,
                 bfactor, occupancy);
    perf_.fuzzy_seconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - fuzzy_begin)
                               .count();
  } else {
    for (int i = 0; i < N; ++i) {
      const bool dense = (giant_cluster_id > 0 &&
                          nodes[static_cast<std::size_t>(i)].cluster_id ==
                              giant_cluster_id);
      bead_is_dense[static_cast<std::size_t>(i)] = dense;
      bfactor[static_cast<std::size_t>(i)] =
          (nodes[static_cast<std::size_t>(i)].cluster_id > 0) ? 1.0 : 0.0;
      occupancy[static_cast<std::size_t>(i)] = dense ? 1.0 : 0.0;
    }
  }

  auto nodes_pdb = nodes;
  std::vector<int> order(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i)
    order[static_cast<std::size_t>(i)] = i;
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (nodes[static_cast<std::size_t>(a)].type !=
        nodes[static_cast<std::size_t>(b)].type)
      return nodes[static_cast<std::size_t>(a)].type <
             nodes[static_cast<std::size_t>(b)].type;
    if (nodes[static_cast<std::size_t>(a)].chain_idx !=
        nodes[static_cast<std::size_t>(b)].chain_idx)
      return nodes[static_cast<std::size_t>(a)].chain_idx <
             nodes[static_cast<std::size_t>(b)].chain_idx;
    return nodes[static_cast<std::size_t>(a)].id <
           nodes[static_cast<std::size_t>(b)].id;
  });

  {
    std::vector<LLPSNode> sorted_nodes(static_cast<std::size_t>(N));
    std::vector<bool> sorted_dense(static_cast<std::size_t>(N), false);
    for (int i = 0; i < N; ++i) {
      sorted_nodes[static_cast<std::size_t>(i)] =
          nodes[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
      sorted_dense[static_cast<std::size_t>(i)] =
          bead_is_dense[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
    }
    nodes = std::move(sorted_nodes);
    bead_is_dense = std::move(sorted_dense);
  }

  if (paper_accumulator_.enabled()) {
    paper_accumulator_.process_frame(current_time_ps, frame_idx, nodes,
                                     bead_is_dense, giant_cluster_id, pro.Box);
  }

  const TitanExactPeriodicBox bmat(pro.Box);
  auto emit_droplet_shape = [&](const DropletShapeFrame &row) {
    if (droplet_shape_streaming_ && droplet_shape_out_.is_open()) {
      droplet_shape_out_ << row.time_ps << "," << row.frame << ","
                         << row.cluster_id << "," << row.nodes << ","
                         << row.protein_nodes << "," << row.rna_nodes << ","
                         << row.rg_angstrom << "," << row.asphericity_angstrom2
                         << "," << row.acylindricity_angstrom2 << ","
                         << row.relative_shape_anisotropy << ","
                         << row.sphericity_proxy << "\n";
      if (frame_count_ % 100 == 0)
        droplet_shape_out_.flush();
    } else {
      droplet_shape_rows_.push_back(row);
    }
  };

  DropletShapeFrame shape_row;
  shape_row.time_ps = current_time_ps;
  shape_row.frame = frame_idx;
  shape_row.cluster_id = giant_cluster_id;
  if (giant_cluster_id > 0) {
    int ref_idx = -1;
    for (int i = 0; i < N; ++i) {
      if (nodes[static_cast<std::size_t>(i)].cluster_id == giant_cluster_id) {
        ref_idx = i;
        break;
      }
    }
    if (ref_idx >= 0) {
      const auto &ref = nodes[static_cast<std::size_t>(ref_idx)];
      std::vector<std::array<double, 3>> xyz;
      xyz.reserve(static_cast<std::size_t>(N));
      for (const auto &node : nodes) {
        if (node.cluster_id != giant_cluster_id)
          continue;
        double dx = node.x - ref.x;
        double dy = node.y - ref.y;
        double dz = node.z - ref.z;
        bmat.apply_mic(dx, dy, dz);
        xyz.push_back({ref.x + dx, ref.y + dy, ref.z + dz});
        ++shape_row.nodes;
        if (node.type == 1)
          ++shape_row.protein_nodes;
        else
          ++shape_row.rna_nodes;
      }
      std::vector<std::array<double,3>> lifted;
      const bool finite_shape=titan_lift_radius_graph(bmat,xyz,eps_,lifted);
      if(!finite_shape) {
        shape_row.rg_angstrom=shape_row.asphericity_angstrom2=shape_row.acylindricity_angstrom2=shape_row.relative_shape_anisotropy=shape_row.sphericity_proxy=std::numeric_limits<double>::quiet_NaN();
      }
      xyz=std::move(lifted);
      if (finite_shape && !xyz.empty()) {
        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;
        for (const auto &p : xyz) {
          cx += p[0];
          cy += p[1];
          cz += p[2];
        }
        const double inv_n = 1.0 / static_cast<double>(xyz.size());
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;
        double xx = 0.0;
        double xy = 0.0;
        double xz = 0.0;
        double yy = 0.0;
        double yz = 0.0;
        double zz = 0.0;
        for (const auto &p : xyz) {
          const double dx = p[0] - cx;
          const double dy = p[1] - cy;
          const double dz = p[2] - cz;
          xx += dx * dx;
          xy += dx * dy;
          xz += dx * dz;
          yy += dy * dy;
          yz += dy * dz;
          zz += dz * dz;
        }
        xx *= inv_n;
        xy *= inv_n;
        xz *= inv_n;
        yy *= inv_n;
        yz *= inv_n;
        zz *= inv_n;
        const auto eig =
            eigenvalues_symmetric_3x3(xx, xy, xz, yy, yz, zz);
        const double l1 = std::max(0.0, eig[0]);
        const double l2 = std::max(0.0, eig[1]);
        const double l3 = std::max(0.0, eig[2]);
        const double rg2 = std::max(0.0, l1 + l2 + l3);
        shape_row.rg_angstrom = std::sqrt(rg2);
        shape_row.asphericity_angstrom2 = l1 - 0.5 * (l2 + l3);
        shape_row.acylindricity_angstrom2 = l2 - l3;
        const double mean_l = rg2 / 3.0;
        shape_row.relative_shape_anisotropy =
            (rg2 > 0.0)
                ? (1.5 * ((l1 - mean_l) * (l1 - mean_l) +
                          (l2 - mean_l) * (l2 - mean_l) +
                          (l3 - mean_l) * (l3 - mean_l)) /
                   (rg2 * rg2))
                : 0.0;
        shape_row.sphericity_proxy =
            std::max(0.0, std::min(1.0,
                                   1.0 - shape_row.relative_shape_anisotropy));
      }
    }
  }
  emit_droplet_shape(shape_row);

  double sum_rg_pro_dilute = 0.0;
  double sum_rg_pro_dense = 0.0;
  double sum_rg_rna_dilute = 0.0;
  double sum_rg_rna_dense = 0.0;
  int cnt_pro_dilute = 0;
  int cnt_pro_dense = 0;
  int cnt_rna_dilute = 0;
  int cnt_rna_dense = 0;
  std::vector<ChainFrameSummary> frame_chains;
  frame_chains.reserve(static_cast<std::size_t>(N));

  std::size_t ci = 0;
  while (ci < static_cast<std::size_t>(N)) {
    std::size_t cj = ci;
    const int cur_type = nodes[ci].type;
    const int cur_chain = nodes[ci].chain_idx;
    while (cj < static_cast<std::size_t>(N) && nodes[cj].type == cur_type &&
           nodes[cj].chain_idx == cur_chain)
      ++cj;
    const int n_beads = static_cast<int>(cj - ci);

    for (std::size_t k = ci + 1; k < cj; ++k) {
      double dx = nodes[k].x - nodes[k - 1].x;
      double dy = nodes[k].y - nodes[k - 1].y;
      double dz = nodes[k].z - nodes[k - 1].z;
      bmat.apply_mic(dx, dy, dz);
      nodes[k].x = nodes[k - 1].x + dx;
      nodes[k].y = nodes[k - 1].y + dy;
      nodes[k].z = nodes[k - 1].z + dz;
    }

    int dense_beads = 0;
    std::unordered_map<int, int> cluster_counts;
    for (std::size_t k = ci; k < cj; ++k) {
      if (bead_is_dense[k])
        ++dense_beads;
      ++cluster_counts[nodes[k].cluster_id];
    }
    bool chain_dense = (dense_beads > n_beads / 2);

    double ccx = 0.0;
    double ccy = 0.0;
    double ccz = 0.0;
    for (std::size_t k = ci; k < cj; ++k) {
      ccx += nodes[k].x;
      ccy += nodes[k].y;
      ccz += nodes[k].z;
    }
    ccx /= n_beads;
    ccy /= n_beads;
    ccz /= n_beads;

    double rg_sq = 0.0;
    for (std::size_t k = ci; k < cj; ++k) {
      const double dx = nodes[k].x - ccx;
      const double dy = nodes[k].y - ccy;
      const double dz = nodes[k].z - ccz;
      rg_sq += dx * dx + dy * dy + dz * dz;
    }
    const double rg = std::sqrt(rg_sq / n_beads);

    if (cur_type == 1) {
      if (chain_dense) {
        sum_rg_pro_dense += rg;
        ++cnt_pro_dense;
      } else {
        sum_rg_pro_dilute += rg;
        ++cnt_pro_dilute;
      }
    } else {
      if (chain_dense) {
        sum_rg_rna_dense += rg;
        ++cnt_rna_dense;
      } else {
        sum_rg_rna_dilute += rg;
        ++cnt_rna_dilute;
      }
    }

    int dominant_cluster = -1;
    int dominant_count = -1;
    for (const auto &entry : cluster_counts) {
      if (entry.second > dominant_count) {
        dominant_count = entry.second;
        dominant_cluster = entry.first;
      }
    }
    frame_chains.push_back(
        {cur_type == 1 ? 0 : 1, cur_chain, chain_dense, dominant_cluster, ccx,
         ccy, ccz});
    ci = cj;
  }

  const double rg_pd =
      (cnt_pro_dilute > 0) ? (sum_rg_pro_dilute / cnt_pro_dilute) : 0.0;
  const double rg_pD =
      (cnt_pro_dense > 0) ? (sum_rg_pro_dense / cnt_pro_dense) : 0.0;
  const double rg_rd =
      (cnt_rna_dilute > 0) ? (sum_rg_rna_dilute / cnt_rna_dilute) : 0.0;
  const double rg_rD =
      (cnt_rna_dense > 0) ? (sum_rg_rna_dense / cnt_rna_dense) : 0.0;

  if (streaming_csv_ && ts_out_.is_open()) {
    ts_out_ << std::fixed << std::setprecision(1) << current_time_ps << "," << N
            << "," << n_valid << "," << largest_pro << "," << largest_rna << ","
            << noise_cnt << "," << std::setprecision(3) << rg_pd << "," << rg_pD
            << "," << rg_rd << "," << rg_rD << "\n";
    if (frame_count_ % 100 == 0)
      ts_out_.flush();
  } else {
    ClusterMetrics m;
    m.time_ps = current_time_ps;
    m.num_nodes = N;
    m.num_clusters = n_valid;
    m.largest_cluster_pro_chains = largest_pro;
    m.largest_cluster_rna_chains = largest_rna;
    m.noise_count = noise_cnt;
    m.rg_pro_dilute = rg_pd;
    m.rg_pro_dense = rg_pD;
    m.rg_rna_dilute = rg_rd;
    m.rg_rna_dense = rg_rD;
    timeseries_.push_back(m);
  }

  if (dynamics_enabled_) {
    const auto dynamics_begin = std::chrono::steady_clock::now();
    update_dynamics(pro.Box, current_time_ps / 1000.0, giant_cluster_id,
                    cluster_pro, cluster_rna, frame_chains);
    perf_.dynamics_seconds += std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() -
                                  dynamics_begin)
                                  .count();
  }
  if (rna_compaction_enabled_) {
    const auto rna_begin = std::chrono::steady_clock::now();
    update_rna_compaction(pro, pro_info, lig, lig_info, pro.Box,
                          current_time_ps / 1000.0, frame_idx, frame_chains);
    perf_.rna_compaction_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      rna_begin)
            .count();
  }

  if (frame_count_ % write_pdb_stride_ == 0)
    write_frame_to_pdb(nodes_pdb, frame_idx, bfactor, occupancy);
  ++frame_count_;
}


// ============================================================================
// Dynamics helpers
// ============================================================================








// ============================================================================
// Export
// ============================================================================






// ============================================================================
// Fuzzy refine / PDB
// ============================================================================


#ifdef TITAN_LLPS_TESTING
#endif

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

namespace {

std::string csv_field(const std::string &value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos)
    return value;
  std::string escaped = "\"";
  for (const char ch : value) {
    if (ch == '\"')
      escaped += '\"';
    escaped += ch;
  }
  escaped += '\"';
  return escaped;
}

std::string resolved_chain_label(const std::string &label, int chain_index) {
  if (label.find_first_not_of(" \t") != std::string::npos)
    return label;
  return std::to_string(chain_index);
}

} // namespace

void TitanLLPS::export_sticker_hotspots(const info &info_output,
                                        std::fstream &log) const {
  const bool available =
      contacts_on_ && !result_.contact_freq.empty() && n_stickers_ > 0;
  if (!available)
    return;

  std::vector<double> residue_scores(static_cast<std::size_t>(n_stickers_), 0.0);
  for (int i = 0; i < n_stickers_; ++i) {
    const auto ii = static_cast<std::size_t>(i);
    for (int j = 0; j < n_stickers_; ++j) {
      if (i == j)
        continue;
      residue_scores[ii] += static_cast<double>(
          result_.contact_freq[ii][static_cast<std::size_t>(j)]);
    }
  }

  const double mean_score =
      residue_scores.empty()
          ? 0.0
          : std::accumulate(residue_scores.begin(), residue_scores.end(), 0.0) /
                static_cast<double>(residue_scores.size());
  std::vector<int> order(static_cast<std::size_t>(n_stickers_));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const double sa = residue_scores[static_cast<std::size_t>(a)];
    const double sb = residue_scores[static_cast<std::size_t>(b)];
    if (sa != sb)
      return sa > sb;
    return result_.sticker_residue_indices[static_cast<std::size_t>(a)] <
           result_.sticker_residue_indices[static_cast<std::size_t>(b)];
  });

  std::ofstream out(info_output.get_llps_sticker_hotspots_filename());
  if (!out.is_open()) {
    TitanUI::log_kv(log, info_output.get_llps_sticker_hotspots_filename(),
                    "skipped: cannot open");
    return;
  }
  out << "Rank,Residue,ContactFrequencySum,MeanNormalizedEnrichment\n";
  out << std::fixed << std::setprecision(6);
  int rank = 1;
  for (int idx : order) {
    const auto ii = static_cast<std::size_t>(idx);
    const double score = residue_scores[ii];
    const double enrichment = (mean_score > 0.0) ? (score / mean_score) : 0.0;
    out << rank++ << ","
        << result_.sticker_residue_indices[ii] << "," << score << ","
        << enrichment << "\n";
  }
  TitanUI::log_kv(log, "LLPS Sticker Hotspot Residues",
                  std::to_string(n_stickers_));
  TitanUI::print_info("  Written: " +
                      info_output.get_llps_sticker_hotspots_filename());
}

void TitanLLPS::export_results(const info &info_output, std::fstream &log) const {
  TitanUI::print_module_header("LLPS Analysis - Export");
  const int N = static_cast<int>(result_.frames.size());
  const char *policy = "auto";
  if (kernel_mode_ == LlpsKernelMode::Scalar)
    policy = "scalar";
  else if (kernel_mode_ == LlpsKernelMode::Avx2)
    policy = "avx2";
  const double density_mib =
      static_cast<double>((vox_pro_.size() + vox_rna_.size()) *
                          sizeof(double)) /
      (1024.0 * 1024.0);

  TitanUI::log_section(log, "LLPS Macro Summary");
  TitanUI::log_kv(log, "LLPS Frames", std::to_string(N));
  TitanUI::log_kv(log, "LLPS Macro-Nodes",
                  std::to_string(n_macro_) +
                      " (dimer_size=" + std::to_string(dimer_size_) + ")");
  if (!result_.frames.empty()) {
    TitanUI::log_kv(log, "LLPS Time Window (ns)",
                    std::to_string(result_.frames.front().time_ns) + " -> " +
                        std::to_string(result_.frames.back().time_ns));
  }
  TitanUI::log_kv(log, "LLPS Kernel Policy", policy);
  TitanUI::log_kv(log, "LLPS Kernel Effective",
                  kernel_allows_avx2() ? "avx2" : "scalar");
  TitanUI::log_kv(log, "LLPS Density Policy", density_mode_);
  TitanUI::log_kv(log, "LLPS Density Effective",
                  density_disabled_by_budget_ ? "skipped" : density_mode_);
  TitanUI::log_kv(log, "LLPS Density Memory MiB",
                  std::to_string(density_mib));
  TitanUI::log_kv(log, "LLPS Queue Max Depth",
                  std::to_string(perf_.max_queue_depth));
  TitanUI::log_kv(log, "LLPS Percolated", result_.is_percolated ? "Yes" : "No");
  if (result_.is_percolated)
    TitanUI::log_kv(log, "Percolation Time (ns)",
                    std::to_string(result_.percolation_time_ns));
  TitanUI::log_kv(log, "tau (ns)", std::to_string(result_.tau_ns));
  TitanUI::log_time(log, "LLPS Pack", perf_.pack_seconds);
  TitanUI::log_time(log, "LLPS Macro Contacts", perf_.contact_seconds);
  TitanUI::log_time(log, "LLPS Protein-RNA Contacts",
                    perf_.protein_rna_seconds);
  TitanUI::log_time(log, "LLPS Graph", perf_.graph_seconds);
  TitanUI::log_time(log, "LLPS Density", perf_.density_seconds);
  TitanUI::log_time(log, "LLPS Polymer Scaling",
                    perf_.polymer_scaling_seconds);
  TitanUI::log_time(log, "LLPS Dynamics", perf_.dynamics_seconds);
  TitanUI::log_time(log, "LLPS Finalize C(t)", perf_.finalize_ct_seconds);
  TitanUI::log_time(log, "LLPS Finalize Contact Map",
                    perf_.finalize_contact_map_seconds);
  TitanUI::log_time(log, "LLPS Finalize Total", perf_.finalize_seconds);
  {
    const bool new_report =
        !static_cast<bool>(std::ifstream(info_output.get_memory_report_filename()));
    std::ofstream report(info_output.get_memory_report_filename(), std::ios::app);
    if (report.is_open()) {
      if (new_report) {
        report << "Module,Mode,Nodes,Frames,EstimatedExtraGiB,BudgetGiB,Pairs,"
                  "Reason\n";
      }
      const std::size_t density_bytes =
          (vox_pro_.size() + vox_rna_.size()) * sizeof(double);
      const long double gib = 1024.0L * 1024.0L * 1024.0L;
      report << "LLPS/density,"
             << (density_disabled_by_budget_ ? "skipped" : density_mode_)
             << "," << n_macro_ << "," << N << "," << std::fixed
             << std::setprecision(3)
             << static_cast<double>(density_bytes / gib) << ",0,"
             << density_frame_count_ << ",density policy\n";
    }
  }
  {
    const bool new_report = !static_cast<bool>(
        std::ifstream(info_output.get_llps_performance_report_filename()));
    std::ofstream report(info_output.get_llps_performance_report_filename(),
                         std::ios::app);
    if (report.is_open()) {
      if (new_report) {
        report << "Module,Frames,Nodes,KernelPolicy,KernelEffective,"
                  "ExecutorWorkers,Pack_s,Contacts_s,ProteinRNA_s,Graph_s,"
                  "Density_s,Polymer_s,Dynamics_s,FinalizeCt_s,"
                  "FinalizeContactMap_s,FinalizeTotal_s,MacroContactPairs,"
                  "StickerPairs,ProteinPairs,ProteinRNAPairs,MaxQueueDepth,"
                  "DensityMode\n";
      }
      report << "LLPS_macro," << perf_.frames_processed << "," << n_macro_ << ","
             << policy << "," << (kernel_allows_avx2() ? "avx2" : "scalar")
             << "," << perf_.finalize_executor_workers << "," << std::fixed
             << std::setprecision(6) << perf_.pack_seconds << ","
             << perf_.contact_seconds << "," << perf_.protein_rna_seconds << ","
             << perf_.graph_seconds << "," << perf_.density_seconds << ","
             << perf_.polymer_scaling_seconds << "," << perf_.dynamics_seconds
             << "," << perf_.finalize_ct_seconds << ","
             << perf_.finalize_contact_map_seconds << ","
             << perf_.finalize_seconds << "," << perf_.macro_contact_pairs << ","
             << perf_.sticker_contact_pairs << "," << perf_.protein_chain_pairs
             << "," << perf_.protein_rna_pairs << "," << perf_.max_queue_depth
             << "," << (density_disabled_by_budget_ ? "skipped" : density_mode_)
             << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_performance_report_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_timeseries_filename());
    if (out.is_open()) {
      out << "# frame  time_ns  gcc_fraction  gcc_size  gcc_rg_nm  mean_degree  "
             "rho_dense  rho_dilute  dense_count  dilute_count  dense_volume_nm3  dilute_volume_nm3  phase_status\n";
      out << "# rho_dense/rho_dilute: counts of macro-node representative beads in declared observation windows / physical window volume; not automatic coexistence inference.\n";
      out << std::fixed << std::setprecision(6);
      out << "# phase_geometry=" << phase_config_.geometry << " axis_abc=" << phase_config_.axis << " dense_nm=" << phase_config_.dense << " dilute_inner_nm=" << phase_config_.dilute_inner << " dilute_outer_nm=" << phase_config_.dilute_outer << "\n";
      for (const auto &frame : result_.frames) {
        out << frame.frame_idx << "  " << frame.time_ns << "  "
            << frame.gcc_fraction << "  " << frame.gcc_size << "  "
            << frame.gcc_rg_nm << "  " << frame.mean_degree << "  "
            << frame.density_dense << "  " << frame.density_dilute << "  " << frame.phase.dense_count << "  " << frame.phase.dilute_count << "  " << frame.phase.dense_volume << "  " << frame.phase.dilute_volume << "  " << frame.phase.status << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_timeseries_filename());
    }
  }

  if (!result_.ct_lags_ns.empty()) {
    std::ofstream out(info_output.get_llps_ct_filename());
    if (out.is_open()) {
      out << "# lag_ns  C(t)\n";
      out << std::fixed << std::setprecision(8);
      for (std::size_t i = 0; i < result_.ct_lags_ns.size(); ++i)
        out << result_.ct_lags_ns[i] << "  " << result_.ct_values[i] << "\n";
      TitanUI::print_info("  Written: " + info_output.get_llps_ct_filename());
    }
  }

  {
    std::ofstream out(info_output.get_llps_degree_dist_filename());
    if (out.is_open()) {
      out << "# k  P(k)_mean  P(k)_std\n";
      out << std::fixed << std::setprecision(6);
      for (std::size_t k = 0; k < result_.pk_mean.size(); ++k)
        out << k << "  " << result_.pk_mean[k] << "  " << result_.pk_std[k]
            << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_degree_dist_filename());
    }
  }

  if (contacts_on_ && !result_.contact_freq.empty()) {
    std::ofstream out(info_output.get_llps_contact_map_filename());
    if (out.is_open()) {
      out << "# residue_i  residue_j  contact_frequency\n";
      out << "# n_residues=" << n_stickers_ << "\n";
      out << "# contact map is aggregated by residue number, not bead index\n";
      out << std::fixed << std::setprecision(6);
      for (int i = 0; i < n_stickers_; ++i) {
        for (int j = i + 1; j < n_stickers_; ++j) {
          if (result_.contact_freq[static_cast<std::size_t>(i)]
                                  [static_cast<std::size_t>(j)] > 1e-6f) {
            out << result_.sticker_residue_indices[static_cast<std::size_t>(i)]
                << "  "
                << result_.sticker_residue_indices[static_cast<std::size_t>(j)]
                << "  "
                << result_.contact_freq[static_cast<std::size_t>(i)]
                                       [static_cast<std::size_t>(j)]
                << "\n";
          }
        }
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_contact_map_filename());
    }
  }

  if (contacts_on_) {
    std::ofstream out(
        info_output.get_llps_chain_residue_contact_map_filename());
    if (out.is_open()) {
      out << "SpeciesA,ChainIndexA,ChainRawIDA,ChainLabelA,ResidueA,"
             "ResidueNameA,SpeciesB,ChainIndexB,ChainRawIDB,ChainLabelB,"
             "ResidueB,ResidueNameB,ContactClass,SameDimer,"
             "SequenceSeparation,ContactFrames,ContactFrequency\n";
      out << std::fixed << std::setprecision(6);

      std::vector<std::pair<uint64_t, uint64_t>> ordered_edges(
          detailed_contact_frame_counts_.begin(),
          detailed_contact_frame_counts_.end());
      std::sort(ordered_edges.begin(), ordered_edges.end(),
                [](const auto &lhs, const auto &rhs) {
                  return lhs.first < rhs.first;
                });
      for (const auto &[key, contact_frames] : ordered_edges) {
        const auto [node_a, node_b] = unpack_pair(key);
        if (node_a < 0 || node_b < 0 ||
            node_a >= static_cast<int>(contact_residue_nodes_.size()) ||
            node_b >= static_cast<int>(contact_residue_nodes_.size())) {
          continue;
        }
        const ContactResidueMeta &a =
            contact_residue_nodes_[static_cast<std::size_t>(node_a)];
        const ContactResidueMeta &b =
            contact_residue_nodes_[static_cast<std::size_t>(node_b)];

        std::string contact_class = "other";
        std::string same_dimer = "NA";
        int sequence_separation = -1;
        if (a.species == "protein" && b.species == "protein") {
          const bool same_chain = a.chain_index == b.chain_index;
          const bool same_macro =
              a.chain_index >= 0 && b.chain_index >= 0 &&
              a.chain_index / dimer_size_ == b.chain_index / dimer_size_;
          same_dimer = same_macro ? "1" : "0";
          if (same_chain) {
            contact_class = "protein_intra_chain";
            sequence_separation =
                std::abs(a.residue_index - b.residue_index);
          } else if (same_macro) {
            contact_class = "protein_inter_chain_same_dimer";
          } else {
            contact_class = "protein_inter_chain_other_dimer";
          }
        } else if (a.species != b.species) {
          contact_class = "protein_rna";
        } else if (a.chain_index == b.chain_index) {
          contact_class = "rna_intra_chain";
          sequence_separation =
              std::abs(a.residue_index - b.residue_index);
        } else {
          contact_class = "rna_inter_chain";
        }

        const double frequency =
            N > 0 ? static_cast<double>(contact_frames) /
                        static_cast<double>(N)
                  : 0.0;
        out << a.species << "," << a.chain_index << "," << a.chain_raw_id
            << ","
            << csv_field(resolved_chain_label(a.chain_label, a.chain_index))
            << "," << a.residue_index << "," << csv_field(a.residue_name)
            << "," << b.species << "," << b.chain_index << ","
            << b.chain_raw_id << ","
            << csv_field(resolved_chain_label(b.chain_label, b.chain_index))
            << "," << b.residue_index << "," << csv_field(b.residue_name)
            << "," << contact_class << "," << same_dimer << ","
            << sequence_separation << "," << contact_frames << ","
            << frequency << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_chain_residue_contact_map_filename());
    }
  }
  export_sticker_hotspots(info_output, log);

  {
    std::ofstream out(info_output.get_llps_summary_filename());
    if (out.is_open()) {
      out << "percolated: " << (result_.is_percolated ? "yes" : "no") << "\n";
      out << "percolation_time_ns: " << result_.percolation_time_ns << "\n";
      out << "tau_ns: " << result_.tau_ns << "\n";
      out << "n_macronodes: " << n_macro_ << "\n";
      out << "dimer_size: " << dimer_size_ << "\n";
      double mean_rg = 0.0;
      double mean_fraction = 0.0;
      for (const auto &frame : result_.frames) {
        mean_rg += frame.gcc_rg_nm;
        mean_fraction += frame.gcc_fraction;
      }
      if (N > 0) {
        mean_rg /= static_cast<double>(N);
        mean_fraction /= static_cast<double>(N);
      }
      out << "mean_gcc_rg_nm: " << mean_rg << "\n";
      out << "mean_gcc_fraction: " << mean_fraction << "\n";
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_summary_filename());
    }
  }

  if (density_frame_count_ > 0 && grid_nx_ > 0) {

    const std::size_t total = static_cast<std::size_t>(grid_nx_) * grid_ny_ *
                              static_cast<std::size_t>(grid_nz_);
    auto write_dx = [&](const std::string &filename,
                        const std::vector<double> &vox,
                        const std::string &label) {
      std::ofstream out(filename);
      if (!out.is_open())
        return;
      out << "object 1 class gridpositions counts " << grid_nx_ << " "
          << grid_ny_ << " " << grid_nz_ << "\n";
      out << "# Mean per-frame number density (nm^-3) on fixed fractional bins, displayed in first accepted cell; variable cells are not a fixed Cartesian density map.\n";
      int dims[3]={grid_nx_,grid_ny_,grid_nz_};
      const auto& b=density_reference_cell_;
      out << "origin " << .5*(b[0]/dims[0]+b[1]/dims[1]+b[2]/dims[2]) << " " << .5*(b[3]/dims[0]+b[4]/dims[1]+b[5]/dims[2]) << " " << .5*(b[6]/dims[0]+b[7]/dims[1]+b[8]/dims[2]) << "\n";
      for(int k=0;k<3;++k) out << "delta " << b[k]/dims[k] << " " << b[3+k]/dims[k] << " " << b[6+k]/dims[k] << "\n";
      out << "object 2 class gridconnections counts " << grid_nx_ << " "
          << grid_ny_ << " " << grid_nz_ << "\n";
      out << "object 3 class array type double rank 0 items " << total
          << " data follows\n";
      out << std::fixed << std::setprecision(6);
      int col = 0;
      for (std::size_t i = 0; i < total; ++i) {
        const double density =
            static_cast<double>(vox[i]) /
            static_cast<double>(density_frame_count_);
        if (col > 0)
          out << ' ';
        out << density;
        if (++col == 3) {
          out << "\n";
          col = 0;
        }
      }
      if (col > 0)
        out << "\n";
      out << "object \"" << label << "\" class field\n";
      TitanUI::print_info("  Written: " + filename);
    };
    write_dx(info_output.get_llps_density_pro_filename(), vox_pro_, "density_pro");
    write_dx(info_output.get_llps_density_rna_filename(), vox_rna_, "density_rna");
  }

  if (max_rna_chain_len_ > 1) {
    std::ofstream out(info_output.get_llps_polymer_scaling_filename());
    if (out.is_open()) {
      out << "# s  mean_R2_nm2  R_rms_nm  count\n";
      out << std::fixed << std::setprecision(6);
      for (int s = 1; s < max_rna_chain_len_; ++s) {
        if (count_R2_s_[static_cast<std::size_t>(s)] == 0)
          continue;
        const double mean_R2 =
            sum_R2_s_[static_cast<std::size_t>(s)] /
            static_cast<double>(count_R2_s_[static_cast<std::size_t>(s)]);
        out << s << "  " << mean_R2 << "  " << std::sqrt(mean_R2) << "  "
            << count_R2_s_[static_cast<std::size_t>(s)] << "\n";
      }
      TitanUI::print_info("  Written: " +
                          info_output.get_llps_polymer_scaling_filename());
    }
  }

  TitanUI::log_section(log, "LLPS Output Audit");
  TitanUI::log_kv(log, info_output.get_llps_timeseries_filename(),
                  N > 0 ? "emitted" : "skipped: no frames");
  TitanUI::log_kv(log, info_output.get_llps_ct_filename(),
                  result_.ct_lags_ns.empty() ? "skipped: no C(t) samples"
                                             : "emitted");
  TitanUI::log_kv(log, info_output.get_llps_degree_dist_filename(),
                  result_.pk_mean.empty() ? "skipped: no degree histogram"
                                          : "emitted");
  TitanUI::log_kv(log, info_output.get_llps_contact_map_filename(),
                  contacts_on_ ? "emitted" : "skipped: -llps-contacts off");
  TitanUI::log_kv(
      log, info_output.get_llps_chain_residue_contact_map_filename(),
      contacts_on_ ? "emitted" : "skipped: -llps-contacts off");
  TitanUI::log_kv(log, info_output.get_llps_sticker_hotspots_filename(),
                  (contacts_on_ && !result_.contact_freq.empty())
                      ? "emitted"
                      : "skipped: no contact map");
  TitanUI::log_kv(log, info_output.get_llps_density_pro_filename(),
                  density_disabled_by_budget_ ? "skipped: density policy/budget"
                                              : "emitted");
  TitanUI::log_kv(log, info_output.get_llps_density_rna_filename(),
                  density_disabled_by_budget_ ? "skipped: density policy/budget"
                                              : "emitted");
  TitanUI::log_kv(log, info_output.get_llps_polymer_scaling_filename(),
                  max_rna_chain_len_ > 1 ? "emitted"
                                         : "skipped: no RNA chain scaling");
  TitanUI::log_kv(log, info_output.get_llps_performance_report_filename(),
                  "emitted");
  TitanUI::log_kv(log, info_output.get_memory_report_filename(), "emitted");

  export_contact_dynamics(info_output, log);
}

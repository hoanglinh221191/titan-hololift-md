#include "aligned_allocator.h"
#include "BoxMatrix.h"
#include "cas_analysis.h"
#include "chunk_streamer.h"
#include "dccm_network.h"
#include "differential_evolution.h"
#include "functions.h"
#include "gbsa_intrin.h"
#include "groups.h"
#include "martini_nonbonded.h"
#include "memory_utils.h"
#include "miscellaneous.h" // Added as per instruction's Code Edit
#include "ml_extractor.h"
#include "nucleic_profiler.h"
#include "polymer_classification.h"
#include "tda.h"
#include "titan_ui.h"
#include "titan_version_info.h"

#include "engine_postproc.h"
#include "titan_alignment.h"
#include "titan_build_info.h"
#include "titan_config_schema.h"
#include "titan_contact_map.h"
#include "titan_dihedral.h"
#include "titan_qnative.h"
#include "titan_gist.h"
#include "titan_kinetics.h"
#include "titan_llps.h"
#include "titan_llps_cluster.h"
#include "titan_msm.h"
#include "titan_pbe.h"
#include "titan_executor.h"
#include "titan_awh_ifp.h"
#include "titan_ifp.h"
#include "titan_interface_analysis.h"
#include "titan_pca.h" // Ensured inclusion as per instruction
#include "titan_dssp.h"
#include "titan_energy_decomp_engine.h"
#include "titan_energy_validation.h"
#include "titan_topology_energy.h"
#include "titan_fflib_charge_policy.h"
#include "titan_parameter_audit.h"
#include "titan_hbond_kinetics.h"
#include "titan_hotspot_report.h"
#include "titan_lj.h"
#include "titan_msd.h"
#include "titan_membrane.h"
#include "titan_observable_features.h"
#include "titan_output_manifest.h"
#include "titan_performance_observer.h"
#include "titan_rg.h"
#include "titan_rms.h"
#include "titan_s2.h"
#include "titan_sparse_graph_module.h"
#include "titan_sg_streaming.h"
#include "titan_sq.h"
#include "titan_tica_backend_policy.h"
#include "titan_tica_estimator.h"
#include "titan_uncertainty.h"
#include "titan_umap.h"
#include "water_analysis.h"


#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#endif

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#pragma warning(disable : 4996)
using namespace std;
using namespace chrono;

namespace {

struct PullDirectSnapshotSelection {
  std::string source_file{};
  std::string input_mode{};
  std::string selection_rule{};
  double time_ps{0.0};
  std::size_t frame_index{0};
};

void keep_only_selected_frame(Multi_MOL &mol, std::size_t frame_index) {
  if (mol.m_mol.empty())
    return;
  if (frame_index >= mol.m_mol.size())
    frame_index = mol.m_mol.size() - 1;
  MOL selected_frame = std::move(mol.m_mol[frame_index]);
  mol.m_mol.clear();
  mol.m_mol.push_back(std::move(selected_frame));
}

std::string format_time_ps(double time_ps) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << time_ps << " ps";
  return oss.str();
}

std::string progress_source_label(const arguments &Arguments,
                                  const std::string &name_pro) {
  if (Arguments.flag_xdr && !Arguments.traj_file.empty()) {
    return TitanUI::compact_source_label(Arguments.traj_file, 18);
  } else if (Arguments.flag_1_pdb) {
    return TitanUI::compact_source_label(Arguments.pdb_name, 18);
  }
  return TitanUI::compact_source_label(name_pro, 18);
}

std::string pull_direct_selection_rule(const arguments &Arguments) {
  if (Arguments.flag_end_set) {
    return "last snapshot of first input with time <= " +
           format_time_ps(Arguments.end_time);
  }
  return "last available snapshot of first input";
}

void set_pull_snapshot_metadata(const PullDirectSnapshotSelection &selection,
                                de_arg &DE_arguments) {
  DE_arguments.snapshot_source_file = selection.source_file;
  DE_arguments.snapshot_input_mode = selection.input_mode;
  DE_arguments.snapshot_selection_rule = selection.selection_rule;
  DE_arguments.snapshot_time_ps = selection.time_ps;
  DE_arguments.snapshot_frame_index = selection.frame_index;
  DE_arguments.snapshot_metadata_valid = true;
}

std::expected<PullDirectSnapshotSelection, std::string>
prepare_pull_direct_snapshot(arguments &Arguments, de_arg &DE_arguments,
                             const std::string &force_f,
                             const std::string &name_pro,
                             const std::string &name_lig, Multi_MOL &pro_in,
                             Multi_MOL &lig, Multi_MOL &water_traj,
                             groups_cal &Groups_from_index_file,
                             const amino_acid &amine, info &output_info,
                             TitanExecutor &executor, std::size_t num_threads,
                             std::fstream &log) {
  arguments snapshot_args = Arguments;
  snapshot_args.flag_begin_set = false;
  snapshot_args.begin_time = 0.0;
  snapshot_args.dt = 0.0;

  PullDirectSnapshotSelection selection;
  selection.selection_rule = pull_direct_selection_rule(Arguments);

  if (snapshot_args.flag_xdr) {
    ChunkStreamer snapshot_streamer(snapshot_args, output_info,
                                    Groups_from_index_file, force_f, executor,
                                    std::max<std::size_t>(1, num_threads));
    Multi_MOL snapshot_membrane(force_f);
    auto stream_ok = snapshot_streamer.open_streams(
        snapshot_args.pdb_name, name_lig, pro_in, lig, snapshot_membrane);
    if (!stream_ok)
      return std::unexpected(stream_ok.error());

    const std::size_t chunk_size =
        std::max<std::size_t>(32, std::min<std::size_t>(256, num_threads * 32));

    Multi_MOL chunk_pro(force_f), chunk_lig(force_f), chunk_wat(force_f),
        chunk_membrane(force_f);
    MOL last_pro_frame, last_lig_frame;
    bool have_last_frame = false;
    bool have_last_lig_frame = snapshot_args.flag_nocal;
    std::size_t selected_frames = 0;

    while (true) {
      auto chunk_status = snapshot_streamer.read_next_chunk(
          chunk_size, chunk_pro, chunk_lig, chunk_wat, chunk_membrane);
      if (!chunk_status)
        return std::unexpected(chunk_status.error());
      if (chunk_status.value() == 0)
        break;

      selected_frames += chunk_status.value();
      if (!chunk_pro.m_mol.empty()) {
        last_pro_frame = std::move(chunk_pro.m_mol.back());
        have_last_frame = true;
      }
      if (!snapshot_args.flag_nocal && !chunk_lig.m_mol.empty()) {
        last_lig_frame = std::move(chunk_lig.m_mol.back());
        have_last_lig_frame = true;
      }
      chunk_pro.m_mol.clear();
      chunk_lig.m_mol.clear();
      chunk_wat.m_mol.clear();
      chunk_membrane.m_mol.clear();
    }

    if (!have_last_frame) {
      return std::unexpected(
          "Pull-direction snapshot selector: no frame in the first trajectory "
          "satisfied the requested -e window.");
    }
    if (!have_last_lig_frame) {
      return std::unexpected(
          "Pull-direction snapshot selector: ligand coordinates were not found "
          "in the selected trajectory frame.");
    }

    pro_in.m_mol.clear();
    pro_in.m_mol.push_back(std::move(last_pro_frame));
    if (!snapshot_args.flag_nocal) {
      lig.m_mol.clear();
      lig.m_mol.push_back(std::move(last_lig_frame));
    }
    water_traj.m_mol.clear();

    selection.input_mode = "xdr-first-trajectory";
    selection.source_file = snapshot_args.traj_file.empty() ? snapshot_args.pdb_name
                                                            : snapshot_args.traj_file;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames > 0 ? selected_frames - 1 : 0;
  } else if (snapshot_args.flag_1_pdb) {
    Multi_MOL snapshot_pro(force_f), snapshot_lig(force_f), snapshot_wat(force_f);
    take_coordinate(snapshot_pro, snapshot_lig, snapshot_wat, Groups_from_index_file,
                    snapshot_args.pdb_name, force_f, snapshot_args, amine,
                    output_info, 1, 0);
    if (snapshot_pro.m_mol.empty()) {
      return std::unexpected(
          "Pull-direction snapshot selector: no snapshot was found in the first "
          "PDB input.");
    }

    const std::size_t selected_frames = snapshot_pro.m_mol.size();
    keep_only_selected_frame(snapshot_pro, selected_frames - 1);
    if (!snapshot_args.flag_nocal && !snapshot_lig.m_mol.empty())
      keep_only_selected_frame(snapshot_lig, snapshot_lig.m_mol.size() - 1);
    pro_in.m_mol = std::move(snapshot_pro.m_mol);
    if (!snapshot_args.flag_nocal)
      lig.m_mol = std::move(snapshot_lig.m_mol);
    water_traj.m_mol.clear();

    selection.input_mode = "single-pdb-first-input";
    selection.source_file = snapshot_args.pdb_name;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames - 1;
  } else {
    Multi_MOL snapshot_pro(force_f), snapshot_lig(force_f);
    if (snapshot_pro.take_coordinate(name_pro, force_f, snapshot_args, amine,
                                     output_info, 1, 0) != 0) {
      return std::unexpected("Pull-direction snapshot selector: failed to read " +
                             name_pro);
    }
    if (!snapshot_args.flag_nocal &&
        snapshot_lig.take_coordinate(name_lig, force_f, snapshot_args, amine,
                                     output_info, 1, 0) != 0) {
      return std::unexpected("Pull-direction snapshot selector: failed to read " +
                             name_lig);
    }

    const std::size_t selected_frames =
        snapshot_args.flag_nocal
            ? snapshot_pro.m_mol.size()
            : std::min(snapshot_pro.m_mol.size(), snapshot_lig.m_mol.size());
    if (selected_frames == 0) {
      return std::unexpected(
          "Pull-direction snapshot selector: no aligned receptor/ligand frame "
          "was found in the first PDB pair.");
    }

    keep_only_selected_frame(snapshot_pro, selected_frames - 1);
    if (!snapshot_args.flag_nocal)
      keep_only_selected_frame(snapshot_lig, selected_frames - 1);
    pro_in.m_mol = std::move(snapshot_pro.m_mol);
    if (!snapshot_args.flag_nocal)
      lig.m_mol = std::move(snapshot_lig.m_mol);
    water_traj.m_mol.clear();

    selection.input_mode = "paired-pdb-first-input";
    selection.source_file = name_pro + " + " + name_lig;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames - 1;
  }

  TitanUI::log_section(log, "PULL-DIRECT SNAPSHOT SELECTION");
  TitanUI::log_kv(log, "Pull Snapshot Mode", selection.input_mode);
  TitanUI::log_kv(log, "Pull Snapshot Source", selection.source_file);
  TitanUI::log_kv(log, "Pull Snapshot Rule", selection.selection_rule);
  TitanUI::log_kv(log, "Pull Snapshot Time", format_time_ps(selection.time_ps));
  TitanUI::log_kv(log, "Pull Snapshot Frame",
                  std::to_string(selection.frame_index));
  TitanUI::log_message(
      log, "Pull-direct stops after the first input; remaining files are not read.");

  set_pull_snapshot_metadata(selection, DE_arguments);
  return selection;
}

double determinant3x3(const BoxMatrix &box_matrix) {
  return box_matrix.m[0][0] *
             (box_matrix.m[1][1] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][1]) -
         box_matrix.m[0][1] *
             (box_matrix.m[1][0] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][0]) +
         box_matrix.m[0][2] *
             (box_matrix.m[1][0] * box_matrix.m[2][1] -
              box_matrix.m[1][1] * box_matrix.m[2][0]);
}

double frame_box_volume_nm3(const box &frame_box) {
  if (frame_box.get_a() <= 0.0 || frame_box.get_b() <= 0.0 ||
      frame_box.get_c() <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const BoxMatrix box_matrix(frame_box);
  const double volume_a3 = std::abs(determinant3x3(box_matrix));
  if (!(volume_a3 > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return volume_a3 / 1000.0;
}

bool is_generic_autotype_candidate(const std::string &mol_type) {
  return mol_type == "lig" || mol_type == "ligand" ||
         mol_type == "protein" || mol_type == "protein-user" ||
         mol_type == "complex";
}

void append_module_label(std::string &modules, std::string_view label) {
  if (!modules.empty())
    modules += ' ';
  modules += label;
}

bool feature_source_is_observable_tica(const arguments &Arguments) {
  const std::string &source = Arguments.feature_export.source;
  return Arguments.feature_export.source_explicit && source != "dccm" &&
         source != "lj";
}

bool feature_source_needs_dccm(const arguments &Arguments,
                               bool legacy_tica_mode) {
  if (legacy_tica_mode)
    return true;
  const std::string &source = Arguments.feature_export.source;
  return source == "auto" || source == "dccm" || source == "mixed";
}

void add_dccm_feature_columns(std::vector<FeatureColumn> &columns, int n_ca) {
  for (int a = 0; a < n_ca; ++a) {
    for (int b = a + 1; b < n_ca; ++b) {
      columns.push_back(FeatureColumn{
          "dccm", "ca_distance_" + std::to_string(a) + "_" + std::to_string(b),
          "A", false, "", "mol1_mol2_trace"});
    }
  }
}

bool append_dccm_pairwise_features(std::vector<double> &values,
                                   DCCMNetwork &dccm_analyzer, int n_ca,
                                   Eigen::MatrixXd &mobile,
                                   Eigen::MatrixXd &ref_centered,
                                   Eigen::Matrix3d &rotation,
                                   bool &ref_initialized) {
  if (n_ca <= 0 || dccm_analyzer.get_total_frames() == 0)
    return false;
  const auto &tx = dccm_analyzer.get_latest_traj_x();
  const auto &ty = dccm_analyzer.get_latest_traj_y();
  const auto &tz = dccm_analyzer.get_latest_traj_z();
  if (tx.size() < static_cast<std::size_t>(n_ca) ||
      ty.size() < static_cast<std::size_t>(n_ca) ||
      tz.size() < static_cast<std::size_t>(n_ca))
    return false;
  for (int a = 0; a < n_ca; ++a) {
    mobile(a, 0) = tx[static_cast<std::size_t>(a)];
    mobile(a, 1) = ty[static_cast<std::size_t>(a)];
    mobile(a, 2) = tz[static_cast<std::size_t>(a)];
  }
  if (!ref_initialized) {
    ref_centered = mobile;
    Eigen::RowVectorXd col_mean = ref_centered.colwise().mean();
    ref_centered.rowwise() -= col_mean;
    mobile = ref_centered;
    ref_initialized = true;
  } else {
    TitanAlignment::align_frame_inplace(ref_centered, mobile, rotation);
  }
  for (int a = 0; a < n_ca; ++a) {
    for (int b = a + 1; b < n_ca; ++b) {
      const double dx = mobile(a, 0) - mobile(b, 0);
      const double dy = mobile(a, 1) - mobile(b, 1);
      const double dz = mobile(a, 2) - mobile(b, 2);
      values.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  }
  return true;
}

std::vector<FeatureColumn> build_observable_feature_columns(
    const arguments &Arguments, int n_ca, const TitanQNativeAnalyzer &qnative,
    const TitanDihedralAnalyzer &dihedral, const TitanContactMapAnalyzer &cmap,
    const TitanLJAnalyzer &lj_analyzer, bool tica_use_lj_features) {
  const std::string &source = Arguments.feature_export.source;
  std::vector<FeatureColumn> columns;
  const bool include_dccm =
      source == "dccm" || source == "mixed" ||
      (source == "auto" && Arguments.msm.flag_tica && !tica_use_lj_features);
  const bool include_lj =
      source == "lj" || (source == "auto" && tica_use_lj_features) ||
      (source == "auto" && !Arguments.msm.flag_tica && lj_analyzer.enabled());
  const bool include_qnative =
      source == "qnative" || source == "mixed" ||
      (source == "auto" && qnative.enabled());
  const bool include_dihedral =
      source == "dihedral" || source == "mixed" ||
      (source == "auto" && dihedral.enabled());
  const bool include_cmap =
      source == "cmap" || (source == "mixed" && cmap.enabled()) ||
      (source == "auto" && cmap.enabled());

  if (include_dccm)
    add_dccm_feature_columns(columns, n_ca);
  if (include_lj) {
    for (int i = 0; i < lj_analyzer.feature_dim(); ++i) {
      columns.push_back(
          FeatureColumn{"lj", "lj_phase2_" + std::to_string(i),
                        "dimensionless", false, "", "mol1"});
    }
  }
  if (include_qnative) {
    columns.push_back(
        FeatureColumn{"qnative", "Q", "fraction", false, "", "mol1"});
  }
  if (include_dihedral) {
    for (const auto &[name, residue] : dihedral.tica_feature_columns()) {
      columns.push_back(FeatureColumn{"dihedral", name, "dimensionless",
                                      false, residue, "mol1"});
    }
  }
  if (include_cmap) {
    columns.push_back(
        FeatureColumn{"cmap", "contact_count", "count", false, "", "mol1"});
    if (source == "cmap") {
      columns.push_back(FeatureColumn{"cmap", "contact_count_normalized",
                                      "fraction", false, "", "mol1"});
    }
  }
  return columns;
}

bool build_observable_feature_vector(
    const arguments &Arguments, DCCMNetwork &dccm_analyzer, int n_ca,
    Eigen::MatrixXd &mobile, Eigen::MatrixXd &ref_centered,
    Eigen::Matrix3d &rotation, bool &ref_initialized,
    const TitanQNativeAnalyzer &qnative, const TitanDihedralAnalyzer &dihedral,
    const TitanContactMapAnalyzer &cmap, const TitanLJAnalyzer &lj_analyzer,
    Eigen::VectorXd &out_values) {
  const std::string &source = Arguments.feature_export.source;
  std::vector<double> values;
  const bool include_dccm =
      source == "dccm" || source == "mixed" ||
      (source == "auto" && Arguments.msm.flag_tica && n_ca > 0);
  const bool include_lj =
      source == "lj" ||
      (source == "auto" && n_ca == 0 && lj_analyzer.has_feature_vector());
  const bool include_qnative =
      source == "qnative" || source == "mixed" ||
      (source == "auto" && qnative.enabled());
  const bool include_dihedral =
      source == "dihedral" || source == "mixed" ||
      (source == "auto" && dihedral.enabled());
  const bool include_cmap = source == "cmap" ||
                            (source == "mixed" && cmap.enabled()) ||
                            (source == "auto" && cmap.enabled());

  if (include_dccm &&
      !append_dccm_pairwise_features(values, dccm_analyzer, n_ca, mobile,
                                     ref_centered, rotation, ref_initialized))
    return false;
  if (include_lj) {
    if (!lj_analyzer.has_feature_vector())
      return false;
    const Eigen::VectorXd &lj = lj_analyzer.latest_feature_vector();
    for (Eigen::Index i = 0; i < lj.size(); ++i)
      values.push_back(lj(i));
  }
  if (include_qnative) {
    if (!qnative.latest_q_valid())
      return false;
    values.push_back(qnative.latest_q());
  }
  if (include_dihedral) {
    if (!dihedral.append_latest_tica_features(values))
      return false;
  }
  if (include_cmap) {
    if (!cmap.latest_contact_count_valid())
      return false;
    values.push_back(cmap.latest_contact_count());
    if (source == "cmap")
      values.push_back(cmap.latest_contact_count_normalized());
  }
  out_values.resize(static_cast<Eigen::Index>(values.size()));
  for (Eigen::Index i = 0; i < out_values.size(); ++i)
    out_values(i) = values[static_cast<std::size_t>(i)];
  return true;
}

std::string format_binary_bytes(long double bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  constexpr long double kib = 1024.0L;
  constexpr long double mib = kib * 1024.0L;
  constexpr long double gib = mib * 1024.0L;
  if (bytes >= gib) {
    oss << (bytes / gib) << " GiB";
  } else if (bytes >= mib) {
    oss << (bytes / mib) << " MiB";
  } else if (bytes >= kib) {
    oss << (bytes / kib) << " KiB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

bool should_enable_dense_pair_cache(const arguments &Arguments,
                                    size_t mol1_atoms, size_t mol2_atoms,
                                    size_t available_ram,
                                    std::string &reason) {
  if (Arguments.force_f == "MARTINI") {
    reason = "disabled for MARTINI";
    return false;
  }

  const bool sg_cache_requested =
      Arguments.flag_sub_groups &&
      (Arguments.sg_mode == "cache" || Arguments.sg_cache == "on");
  if (!(Arguments.flag_cache == 1 || Arguments.force_map == 1 ||
        Arguments.force_map == 3 ||
        (Arguments.edecomp.flag_edecomp &&
         (Arguments.edecomp.source == "mm" ||
          Arguments.edecomp.source == "all")) ||
        sg_cache_requested)) {
    reason = "not requested by active modules";
    return false;
  }

  const long double pair_count =
      static_cast<long double>(mol1_atoms) * static_cast<long double>(mol2_atoms);
  const long double bytes_per_pair =
      static_cast<long double>(sizeof(double)) *
      static_cast<long double>(Arguments.flag_cal_force ? 8 : 2);
  const long double estimated_bytes = pair_count * bytes_per_pair;

  if (Arguments.flag_gbsa && Arguments.flag_xdr) {
    reason = "disabled for XDR + GBSA to avoid multi-GB pair cache (" +
             format_binary_bytes(estimated_bytes) + ")";
    return false;
  }

  constexpr long double hard_cap_bytes = 1.5L * 1024.0L * 1024.0L * 1024.0L;
  if (estimated_bytes > hard_cap_bytes) {
    reason = "estimated cache " + format_binary_bytes(estimated_bytes) +
             " exceeds hard cap";
    return false;
  }

  if (available_ram > 0) {
    const long double ram_limit =
        static_cast<long double>(available_ram) * 0.20L;
    if (estimated_bytes > ram_limit) {
      reason = "estimated cache " + format_binary_bytes(estimated_bytes) +
               " exceeds 20% of available RAM";
      return false;
    }
  }

  reason = "enabled (" + format_binary_bytes(estimated_bytes) + ")";
  return true;
}

bool edecomp_mm_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp &&
         (Arguments.edecomp.source == "mm" ||
          Arguments.edecomp.source == "all");
}

bool edecomp_gbsa_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp && Arguments.flag_gbsa &&
         (Arguments.edecomp.source == "gbsa" ||
          Arguments.edecomp.source == "all");
}

bool edecomp_pbsa_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp && Arguments.pbe.pbsa &&
         (Arguments.edecomp.source == "pbsa" ||
          Arguments.edecomp.source == "all");
}

titan_energy_decomp::RankingMetric
edecomp_rank_metric(const std::string &rank_by) {
  if (rank_by == "mean-abs")
    return titan_energy_decomp::RankingMetric::MeanAbs;
  return titan_energy_decomp::RankingMetric::AbsMean;
}

titan_energy_decomp::EngineConfig make_edecomp_config(
    const arguments &Arguments) {
  titan_energy_decomp::EngineConfig config;
  if (Arguments.edecomp.edge_top == "all") {
    config.top_edge_count = std::numeric_limits<std::size_t>::max();
  } else {
    config.top_edge_count =
        static_cast<std::size_t>(std::stoull(Arguments.edecomp.edge_top));
  }
  config.edge_metric = edecomp_rank_metric(Arguments.edecomp.rank_by);
  config.ranking_metric = edecomp_rank_metric(Arguments.edecomp.rank_by);
  config.sort_direction = titan_energy_decomp::SortDirection::Descending;
  return config;
}

std::string edecomp_output_suffix(const arguments &Arguments,
                                  const std::vector<std::string> &input_line,
                                  int input_name) {
  if (Arguments.flag_separation_results)
    return {};
  const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                           Arguments.flag_pattern_input || Arguments.flag_xdr;
  if (!multi_input)
    return {};
  const int index =
      input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
  return "_" + std::to_string(index);
}

std::optional<std::size_t>
find_edecomp_entity_index(const titan_energy_decomp::EntityAxis &axis,
                          const std::string &chain, int resnr) {
  for (std::size_t i = 0; i < axis.entities.size(); ++i) {
    const auto &entity = axis.entities[i];
    if (entity.chain == chain &&
        static_cast<int>(entity.residue_number) == resnr) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t>
find_edecomp_atom_entity_index(const titan_energy_decomp::EntityAxis &axis,
                               std::size_t one_based_atom_index,
                               const std::string &chain, int resnr,
                               const std::string &atom_name) {
  if (one_based_atom_index > 0 &&
      one_based_atom_index <= axis.entities.size()) {
    return one_based_atom_index - 1;
  }
  for (std::size_t i = 0; i < axis.entities.size(); ++i) {
    const auto &entity = axis.entities[i];
    if (entity.chain == chain &&
        static_cast<int>(entity.residue_number) == resnr &&
        (atom_name.empty() || entity.atom_name == atom_name)) {
      return i;
    }
  }
  return std::nullopt;
}

std::vector<titan_energy_decomp::EndpointEntityRow>
make_gbsa_endpoint_rows(const GBSA_Aggregator &frame_decomp,
                        const titan_energy_decomp::EndpointEntityDecompositionEngine
                            &engine) {
  std::vector<titan_energy_decomp::EndpointEntityRow> rows;
  rows.reserve(frame_decomp.get_decomp().size());
  const auto &mol1_residues = engine.mol1_axes().residues;
  const auto &mol2_residues = engine.mol2_axes().residues;
  for (const auto &entry : frame_decomp.get_decomp()) {
    const std::string role = entry.is_ligand ? "mol2" : "mol1";
    const auto &axis = entry.is_ligand ? mol2_residues : mol1_residues;
    const auto entity_index =
        find_edecomp_entity_index(axis, entry.chain_str, entry.resnr);
    if (!entity_index)
      continue;
    rows.push_back({titan_energy_decomp::EntityKind::Residue, role,
                    *entity_index, 0.5 * entry.E_elec, 0.5 * entry.E_vdw,
                    entry.G_polar, entry.G_nonpolar});
  }
  return rows;
}

std::vector<titan_energy_decomp::EndpointEntityRow> make_pbsa_endpoint_rows(
    const std::vector<TitanPBE::EDecompRow> &frame_rows,
    const titan_energy_decomp::EndpointEntityDecompositionEngine &engine) {
  std::vector<titan_energy_decomp::EndpointEntityRow> rows;
  rows.reserve(frame_rows.size());
  for (const TitanPBE::EDecompRow &entry : frame_rows) {
    const std::string role =
        (entry.molecule == "ligand" || entry.molecule == "mol2") ? "mol2"
                                                                  : "mol1";
    const bool atom_level = entry.unit_type == "atom";
    const auto &axes = role == "mol2" ? engine.mol2_axes()
                                      : engine.mol1_axes();
    const auto entity_index =
        atom_level ? find_edecomp_atom_entity_index(
                         axes.atoms, entry.atom_index, entry.chain,
                         static_cast<int>(entry.resnr), entry.atom_name)
                   : find_edecomp_entity_index(
                         axes.residues, entry.chain,
                         static_cast<int>(entry.resnr));
    if (!entity_index)
      continue;
    rows.push_back({atom_level ? titan_energy_decomp::EntityKind::Atom
                               : titan_energy_decomp::EntityKind::Residue,
                    role, *entity_index, entry.elec_kj_mol, entry.vdw_kj_mol,
                    entry.polar_kj_mol, entry.nonpolar_kj_mol});
  }
  return rows;
}

void auto_adjust_polymer_type(arguments &Arguments, Multi_MOL &mol,
                              std::fstream &log, bool is_mol1) {
  const std::string detected_type =
      PolymerClassification::detect_nucleic_polymer_type(mol.info);
  if (detected_type.empty())
    return;

  std::string &mol_type =
      is_mol1 ? Arguments.mol_1_type : Arguments.mol_2_type;
  const bool explicit_type =
      is_mol1 ? Arguments.mol_1_type_explicit : Arguments.mol_2_type_explicit;
  const std::string label = is_mol1 ? "Molecule 1 Type" : "Molecule 2 Type";
  const std::string molecule_name = is_mol1 ? "molecule 1" : "molecule 2";

  if (!explicit_type && is_generic_autotype_candidate(mol_type)) {
    mol_type = detected_type;
    mol.set_mol_type(detected_type);
    for (auto &frame : mol.m_mol)
      frame.mol_type = detected_type;

    const std::string message =
        "Auto-detected " + molecule_name + " as " + detected_type +
        " from topology.";
    TitanUI::print_warning(message);
    TitanUI::log_kv(log, label, mol_type);
    TitanUI::log_message(log, "[Auto-Type] " + message);
    return;
  }

  const std::string resolved_upper =
      PolymerClassification::uppercase_copy(mol_type);
  const std::string detected_upper =
      PolymerClassification::uppercase_copy(detected_type);
  const bool resolved_matches_detected =
      resolved_upper.find(detected_upper) != std::string::npos;
  if (!resolved_matches_detected) {
    const std::string message =
        PolymerClassification::uppercase_copy(molecule_name) +
        " topology looks like " + detected_type +
        " but resolved type is " + mol_type +
        ". RNA/DNA-aware modules will fall back to residue/atom detection.";
    TitanUI::print_warning(message);
    TitanUI::log_message(log, "[Auto-Type] " + message);
  }
}

} // namespace

int engine(arguments &Arguments) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_TEXT);
  _setmode(_fileno(stderr), _O_TEXT);
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);

  // KHÔNG bật ENABLE_VIRTUAL_TERMINAL_PROCESSING ở đây nữa
  // Windows Terminal tự bật VT cho mọi app hiện đại
  // Việc bật thủ công kích hoạt DA response từ ConPTY
#endif
  TitanUI::print_logo();
  Arguments.post_process();
  /************************************/
  folders paths;
  vector<string> list_paths;
  de_arg DE_arguments;
  if (Arguments.de.flag_find_pulling_direct == true)
    DE_arguments.parse_input(Arguments);
  paths.add_paths(Arguments, list_paths);
  paths.create_folders(Arguments);

  int sc_flag{Arguments.flag_all_side_chain_ct};
  auto total_run_time{chrono::steady_clock::now()};
  string force_f{Arguments.force_f};

  if (Arguments.flag_cal_force && Arguments.force_map > 0) {
    fstream lg_force_inter("force_lg.txt", ios::out);
    if (Arguments.mol_2_type.find("protein") != string::npos)
      lg_force_inter << "#Res" << setw(11) << "Electro." << setw(12)
                     << "Van der W." << '\n';
    else
      lg_force_inter << "#Atom/Res" << setw(11) << "Electro." << setw(12)
                     << "Van der W." << '\n';
    fstream rec_force_inter("force_rec.txt", ios::out);
    rec_force_inter << "#Res" << setw(12) << "Electro." << setw(12)
                    << "Van der W." << '\n';
    lg_force_inter.close();
    rec_force_inter.close();
  }

  /*************************************/
  double d_standard{Arguments.d_standard};
  int num_amine{};
  size_t num_threads{Arguments.nt}, tot_snap_allfiles{};
  TitanExecutor executor(std::max<size_t>(1, num_threads));
  string name_pro{}, name_lig{};
  para parameter;

  vector<double> E_v, E_e;
  vector<size_t> size_snap;
  vector<vector<double>> E_v_vector, E_e_vector, Volumes;
  vector<force> Force_e, Force_v;
  vector<vector<force>> Force_e_vector, Force_v_vector;
  vector<energy_res> Force_res_stat_mol1, Force_res_stat_mol2;
  vector<vector<energy_groups>> EF_subgroups;
  SgStreamingAnalyzer sg_streaming_analyzer;
  bool sg_streaming_initialized{false};
  vector<mindistance_group> MIN_DIS;
  vector<vector<size_t>> Hbonds_vector;
  amino_acid amine;
  vector<double> time_of_snap;
  vector<string> input_line, cscd_out_name;
  all_cscd cscd_out_dis;
  std::vector<MOL> saved_pro_frame0, saved_lig_frame0, saved_membrane_frame0;
  bool topo_2_is_open{Arguments.flag_topo_2},
      topo_is_open(Arguments.flag_topo_1), status{true};

  groups_cal Groups_from_index_file;

  info info_output{Arguments};
  vector<traj_hb> res_hb_rec, res_hb_lg;

  /************************************/
  fstream cal_out;
  /*************************************/
  fstream log;
  log.open("log.txt", ios::out);
  log.close();
  log.open("log.txt", ios::out | ios::app);
  TitanUI::log_logo(log);
  /*************************************/
  /******************************************/

  if (Arguments.output_manifest.output_manifest ||
      Arguments.output_manifest.schema_export == "json") {
    auto manifest_rows =
        titan_output_manifest::build_output_manifest(Arguments, info_output);
    titan_output_manifest::export_output_manifest(
        manifest_rows, info_output.get_titan_outputs_manifest_filename(), &log);
    titan_output_manifest::export_run_config_summary(
        Arguments, manifest_rows,
        info_output.get_titan_run_config_summary_filename(), &log);
    if (Arguments.output_manifest.schema_export == "json") {
      titan_config_schema::export_schema_json(
          Arguments, info_output.get_titan_schema_filename(), &log);
    }
    if (Arguments.output_manifest.dry_run) {
      TitanUI::print_info(
          "Dry-run complete: wrote manifest/config/schema outputs and skipped trajectory processing.");
      TitanUI::log_message(
          log,
          "[DryRun] Exiting before coordinate loading and trajectory processing.");
      return 0;
    }
  }

  cal_out.open("cal.txt", ios::out);

  if (Arguments.com_dis_energy_2mol != 4) {
    for (size_t i{}; i < paths.get_size(); ++i) {
      fstream output_tot_dis(
          paths.get_path(i) + "/" + info_output.getDisCom2Mol(), ios::out);
      fstream output_elec_dis(paths.get_path(i) + "/electrostatic_" +
                                  info_output.getDisCom2Mol(),
                              ios::out);
      fstream output_vdw_dis(paths.get_path(i) + "/vanderWaals_" +
                                 info_output.getDisCom2Mol(),
                             ios::out);

      output_elec_dis.close();
      output_tot_dis.close();
      output_vdw_dis.close();
    }
  }

  if (Arguments.flag_h_bond == 1) {
    init_hbond_output_files(Arguments);
  }

  TitanUI::log_section(log, "TITAN ENGINE INITIALIZATION");
  TitanUI::log_kv(log, "Software Build", info_output.getBuild());
  TitanUI::log_kv(log, "Software Version", titan_build::product_version());
  TitanUI::log_kv(log, "Build Number",
                  titan_build::build_number_string());
  TitanUI::log_kv(log, "Git Commit", titan_build::git_commit());
  TitanUI::log_kv(log, "Git State", titan_build::git_state_name());
  TitanUI::log_kv(log, "Release Build",
                  titan_build::release_build() ? "yes" : "no");
  TitanUI::log_kv(log, "Source Date", titan_build::source_date_utc());
  if (std::string_view(titan_build::ci_build_number()).empty() == false) {
    TitanUI::log_kv(log, "CI Provider", titan_build::ci_provider());
    TitanUI::log_kv(log, "CI Build", titan_build::ci_build_number());
    TitanUI::log_kv(log, "CI Run ID", titan_build::ci_run_id());
  }
  TitanUI::log_kv(log, "Build Profile", titan_build::build_profile_name());
  TitanUI::log_kv(log, "Compiler", titan_build::compiler_name());
  TitanUI::log_kv(log, "Target OS", titan_build::target_os_name());
  TitanUI::log_kv(log, "Target Architecture",
                  titan_build::target_arch_name());
  TitanUI::log_kv(log, "C++23 Floor", titan_build::cxx23_floor_name());
  TitanUI::log_kv(log, "Compiled SIMD",
                  titan_build::simd_level_name(
                      titan_build::compiled_simd_level()));
  TitanUI::log_kv(log, "Runtime SIMD",
                  titan_build::simd_level_name(
                      titan_build::runtime_simd_level()));
  TitanUI::log_kv(log, "Runtime Supports Build SIMD",
                  titan_build::runtime_supports_compiled_simd() ? "yes"
                                                                 : "no");
  TitanUI::log_kv(log, "Runtime Dispatch",
                  titan_build::runtime_dispatch_mode_name());
  TitanUI::log_kv(log, "std::jthread",
                  TITAN_HAS_STD_JTHREAD ? "available" : "missing");

  time_t timeRun{time(nullptr)};
  char time_str[100];
  strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S",
           localtime(&timeRun));
  TitanUI::log_kv(log, "Execution Time", std::string(time_str));
  TitanUI::log_kv(log, "Force Field", force_f);
  TitanUI::log_kv(log, "Number of Threads", std::to_string(num_threads));

  TitanUI::log_section(log, "SYSTEM CONFIGURATION");
  TitanUI::log_kv(log, "Input File (-f)", Arguments.pdb_name);
  if (Arguments.flag_topo_1)
    TitanUI::log_kv(log, "Topology File (-topo)", Arguments.topo_name);
  if (Arguments.flag_topo_2)
    TitanUI::log_kv(log, "Topology File 2 (-topo2)", Arguments.topo_name_2);

  TitanUI::log_kv(log, "Molecule 1 Type", Arguments.mol_1_type);
  TitanUI::log_kv(log, "Molecule 2 Type", Arguments.mol_2_type);
  TitanUI::log_kv(log, "Trajectory Semantics",
                  Arguments.trajectory_semantics == "remd_state"
                      ? "remd-state (equilibrium/descriptive only)"
                      : "continuous");

  std::string group_str = std::to_string(Arguments.group_1) + " - " +
                          std::to_string(Arguments.group_2);
  TitanUI::log_kv(log, "Interaction Groups (-g)", group_str);

  size_t available_ram = get_available_ram();
  if (available_ram > 0) {
    TitanUI::log_kv(log, "Available RAM",
                    std::to_string(available_ram / (1024 * 1024)) + " MB");
  }

  // Strip trailing zeros for doubles using a lambda or just stringstream
  auto format_double = [](double v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
  };

  TitanUI::log_kv(log, "Distance Cutoff (-dsc)",
                  format_double(Arguments.d_standard) + " A");

  if (Arguments.flag_gbsa) {
    TitanUI::log_section(log, "GBSA CONFIGURATION");
    TitanUI::log_kv(log, "IGB Model", std::to_string(Arguments.igb));
    TitanUI::log_kv(log, "Internal Dielectric",
                    format_double(Arguments.int_diel));
    TitanUI::log_kv(log, "External Dielectric",
                    format_double(Arguments.ext_diel));
    TitanUI::log_kv(log, "Salt Concentration",
                    format_double(Arguments.saltcon) + " M");
    TitanUI::log_kv(log, "GBSA Temperature",
                    format_double(Arguments.gbsa_temperature) + " K");
    TitanUI::log_kv(log, "GBSA SASA Points",
                    std::to_string(Arguments.gbsa_sasa_points));
    TitanUI::log_kv(log, "Born Model",
                    "Amber HCT/OBC analytical pair descreening");
    TitanUI::log_kv(log, "Salt Screening",
                    "Amber Debye kappa with 0.73 ion-exclusion factor");
    TitanUI::log_kv(log, "IE Minimum Samples",
                    std::to_string(Arguments.gbsa_ie_min_samples));
    TitanUI::log_kv(log, "IE Filter Sigma",
                    format_double(Arguments.gbsa_ie_filter_sigma));
  }
  if (Arguments.pbe.flag_pbe) {
    TitanUI::log_section(log, "PBE CONFIGURATION");
    TitanUI::log_kv(log, "Model", "linear PB / two-dielectric");
    TitanUI::log_kv(log, "Surface", Arguments.pbe.surface);
    TitanUI::log_kv(log, "Probe Radius",
                    format_double(Arguments.pbe.probe_radius) + " A");
    if (!Arguments.pbe.test_pqr.empty())
      TitanUI::log_kv(log, "Test PQR Adapter", Arguments.pbe.test_pqr);
    TitanUI::log_kv(log, "Scale", format_double(Arguments.pbe.scale) + " pt/A");
    TitanUI::log_kv(log, "Margin", format_double(Arguments.pbe.margin) + " A");
    TitanUI::log_kv(log, "Gridbox", Arguments.pbe.gridbox);
    TitanUI::log_kv(log, "Internal Dielectric",
                    format_double(Arguments.pbe.int_diel));
    TitanUI::log_kv(log, "External Dielectric",
                    format_double(Arguments.pbe.ext_diel));
    TitanUI::log_kv(log, "Salt Concentration",
                    format_double(Arguments.pbe.salt) + " M");
    TitanUI::log_kv(log, "DeltaG Solv",
                    Arguments.pbe.dgsolv ? "enabled (complex/mol1/mol2)"
                                         : "off");
    TitanUI::log_kv(log, "MM/PBSA",
                    Arguments.pbe.pbsa ? "enabled" : "off");
    TitanUI::log_kv(log, "Kernel Effective",
                    "scalar-threaded v1 (AVX2 reserved for later hot paths)");
    TitanUI::log_kv(log, "Map Output",
                    Arguments.pbe.map + " / " + Arguments.pbe.map_frames);
  }

  const bool need_tica_projection_cache =
      Arguments.msm.flag_msm || Arguments.dham.flag_dham ||
      Arguments.umap.flag_umap ||
      (Arguments.zts.flag_zts && Arguments.zts.zts_domain == "tica");

  std::string active_modules;
  if (Arguments.pbe.flag_pbe)
    append_module_label(active_modules, "PBE");
  if (Arguments.pbe.pbsa)
    append_module_label(active_modules, "MM/PBSA");
  if (Arguments.flag_cas)
    append_module_label(active_modules, "CAS");
  if (Arguments.dssp.flag_dssp &&
      (TitanDSSPAnalyzer::supports_moltype(Arguments.mol_1_type) ||
       TitanDSSPAnalyzer::supports_moltype(Arguments.mol_2_type))) {
    append_module_label(active_modules, "DSSP");
  }
  if (Arguments.lj.flag_lj)
    append_module_label(active_modules, "LJ");
  if (Arguments.sq.flag_sq)
    append_module_label(active_modules, "SQ");
  if (Arguments.cmap.flag_cmap)
    append_module_label(active_modules, "CMap");
  if (Arguments.dihedral.flag_dihedral)
    append_module_label(active_modules, "Dihedral");
  if (Arguments.qnative.flag_qnative)
    append_module_label(active_modules, "QNative");
  if (Arguments.msd.flag_msd)
    append_module_label(active_modules, "MSD");
  if (Arguments.membrane.flag_membrane)
    append_module_label(active_modules, "Membrane");
  if (Arguments.s2.flag_s2)
    append_module_label(active_modules, "S2");
  if (Arguments.rg.flag_rg2)
    append_module_label(active_modules, "Rg2");
  if (Arguments.rms.flag_rms)
    append_module_label(active_modules, "RMSD/RMSF");
  if (Arguments.flag_network || Arguments.community.flag_community)
    append_module_label(active_modules, "DCCM/Network");
  if (Arguments.corr_knn.flag_corr_knn)
    append_module_label(active_modules, "Corr-kNN");
  if (Arguments.community.flag_community)
    append_module_label(active_modules, "Community");
  if (Arguments.flag_pca)
    append_module_label(active_modules, "PCA_FEL");
  if (Arguments.flag_tda)
    append_module_label(active_modules, "TDA");
  if (Arguments.msm.flag_tica)
    append_module_label(active_modules, "TICA");
  if (Arguments.msm.flag_msm)
    append_module_label(active_modules, "MSM");
  if (Arguments.zts.flag_zts)
    append_module_label(active_modules, "ZTS");
  if (Arguments.umap.flag_umap)
    append_module_label(active_modules, "UMAP");
  if (Arguments.dham.flag_dham)
    append_module_label(active_modules, "DHAM");
  if (Arguments.tpt.flag_tpt)
    append_module_label(active_modules, "TPT");
  if (Arguments.flag_h_bond)
    append_module_label(active_modules, "H-Bonds");
  if (Arguments.n_wat >= 0)
    append_module_label(active_modules, "Water");
  if (Arguments.edecomp.flag_edecomp)
    append_module_label(active_modules, "EDecomp");
  if (edecomp_gbsa_requested(Arguments))
    append_module_label(active_modules, "EDecomp-GBSA");
  if (Arguments.hotspot.flag_hotspot_report)
    append_module_label(active_modules, "Hotspot");
  if (Arguments.flag_allostery)
    append_module_label(active_modules, "Allostery");
  if (Arguments.lspmd.flag_lspmd)
    append_module_label(active_modules, "LSP-MD");

  if (!active_modules.empty()) {
    TitanUI::log_section(log, "REQUESTED MODULES");
    TitanUI::log_kv(log, "Requested Features", active_modules);
    // PBC algorithm in use
    if (Arguments.pbc.solv_pbctopo)
      TitanUI::log_kv(log, "PBC Algorithm",
                      Arguments.pbc.pbctopo_mode == "llps-local"
                          ? "pbctopo (LLPS local molecule unwrap)"
                          : "pbctopo (VIBE: Virtual Intact Block Extraction)");
    else if (Arguments.pbc.solv_pbcn)
      TitanUI::log_kv(log, "PBC Algorithm",
                      Arguments.pbc.auto_enabled_pbcn
                          ? "FixBox-style staged envelope (auto-enabled)"
                          : "FixBox-style staged envelope");
    else if (Arguments.pbc.solv_pbc_int == 1)
      TitanUI::log_kv(log, "PBC Algorithm",
                      std::string("Internal-reference envelope (chain ref=") +
                          std::to_string(Arguments.pbc.chain_ref) +
                          (Arguments.pbc.auto_enabled_pbci ? ", auto-enabled)"
                                                          : ")"));
    else if (Arguments.pbc.solv_pbc == 1)
      TitanUI::log_kv(log, "PBC Algorithm", "Prob (legacy)");
    else
      TitanUI::log_kv(log, "PBC Algorithm", "None");
    if (!Arguments.flag_nocal || Arguments.flag_gbsa ||
        Arguments.pbe.flag_pbe)
      TitanUI::log_kv(log, "Energy Distance Policy",
                      "preprocessed_raw (no implicit MIC in energy kernels)");
    if (Arguments.pbc.auto_enabled_pbcn || Arguments.pbc.auto_enabled_pbci) {
      std::vector<std::string> auto_pbc_modules;
      if (Arguments.flag_nocal)
        auto_pbc_modules.push_back("Nocal");
      if (Arguments.rms.flag_rms)
        auto_pbc_modules.push_back("RMSD/RMSF");
      if (Arguments.rg.flag_rg2)
        auto_pbc_modules.push_back("Rg2");
      if (Arguments.msd.flag_msd)
        auto_pbc_modules.push_back("MSD");
      if (Arguments.lj.flag_lj)
        auto_pbc_modules.push_back("LJ");
      if (Arguments.sq.flag_sq)
        auto_pbc_modules.push_back("SQ");
      std::string auto_pbc_message;
      for (std::size_t i = 0; i < auto_pbc_modules.size(); ++i) {
        if (i > 0)
          auto_pbc_message +=
              (i + 1 == auto_pbc_modules.size()) ? " and " : ", ";
        auto_pbc_message += auto_pbc_modules[i];
      }
      if (auto_pbc_modules.size() == 1 &&
          (auto_pbc_modules.front() == "LJ" ||
           auto_pbc_modules.front() == "SQ"))
        auto_pbc_message += " analysis";
      auto_pbc_message +=
          " requested without explicit PBC mode; enabled -pbci 0";
      TitanUI::log_kv(log, "PBC Auto-Enable", auto_pbc_message);
    }
  }

  take_para(parameter, force_f, Arguments);
  take_amino_acids(amine, num_amine, force_f, Arguments);
  take_ff_para(amine, parameter);

  int input_name{};
  gen_input_list(Arguments, input_line, paths, log, input_name);

  if (edecomp_mm_requested(Arguments) && Arguments.flag_nocal) {
    TitanUI::print_error(
        "Energy decomposition: -edecomp-source=mm requires energy "
        "calculation; remove -nocal or select a future non-MM adapter.");
    return 1;
  }

  // ── Auto-detect XDR trajectory from -i file list when -s was used ──
  // If -s set flag_xdr but no -traj was given, detect from input_line
  if (Arguments.flag_xdr && Arguments.traj_file.empty() &&
      !input_line.empty()) {
    bool detected = false;
    for (const auto &file : input_line) {
      if (file.length() <= 4)
        continue;
      std::string ext = file.substr(file.length() - 4);
      if (ext == ".xtc" || ext == ".trr") {
        const std::string detected_type = (ext == ".xtc") ? "xtc" : "trr";
        if (!detected) {
          Arguments.traj_file = file;
          Arguments.xdr_type = detected_type;
          detected = true;
        }
        TitanUI::log_kv(log, "Auto-detect XDR",
                        detected_type + " from " + file);
      }
    }
  }

  GBSA_Aggregator global_gbsa_agg(Arguments, info_output);
  CASAnalyzer cas_analyzer;
  TitanDSSPAnalyzer dssp_mol1_analyzer;
  TitanDSSPAnalyzer dssp_mol2_analyzer;
  TitanHBondKineticsAnalyzer hbk_analyzer;
  TitanLJAnalyzer lj_analyzer;
  TitanSqAnalyzer sq_analyzer;
  TitanContactMapAnalyzer cmap_analyzer;
  TitanInterfaceAnalyzer interface_analyzer;
  TitanDihedralAnalyzer dihedral_analyzer;
  TitanQNativeAnalyzer qnative_analyzer;
  TitanObservableFeatureRouter feature_router;
  TitanPerformanceObserver perf_observer;
  TitanMSDAnalyzer msd_analyzer;
  TitanMembraneAnalyzer membrane_analyzer;
  TitanS2Analyzer s2_analyzer;
  TitanRgAnalyzer rg_analyzer;
  TitanRMSAnalyzer rms_analyzer;
  std::vector<std::string> lj_feature_files;
  NucleicProfiler na_profiler;
  TDAAnalyzer tda_analyzer;
  bool global_has_nucleic_acid = false;

  // Network Analysis (DCCM) Variables
  DCCMNetwork dccm_analyzer;
  bool dccm_analyzer_initialized = false;
  std::chrono::milliseconds time_gbsa{0}, time_cas{0}, time_dssp{0},
      time_hbk{0}, time_lj{0}, time_sq{0}, time_cmap{0},
      time_interface{0},
      time_dihedral{0}, time_qnative{0}, time_msd{0}, time_rg{0}, time_rms{0},
      time_membrane{0}, time_s2{0}, time_edecomp{0},
      time_network{0}, time_tda{0}, time_llps{0}, time_tica{0};
  std::unique_ptr<TitanLLPSCluster> dbscan_llps;
  EnergyDecompositionProvider llps_energy_provider;
  std::chrono::milliseconds time_llps_dbscan{0};

  WaterAnalyzer water_analyzer;
  MLFeatureExtractor ml_extractor;
  std::unique_ptr<TitanKinetics> tica_analyzer;
  std::unique_ptr<TitanMSM> msm_analyzer;
  std::unique_ptr<TitanUMAP> umap_analyzer;
  std::unique_ptr<TitanLLPS> llps_analyzer;
  std::unique_ptr<TitanGIST> gist_analyzer;
  TitanPBE::Analyzer pbe_analyzer;
  EnergyDecompositionProvider pbe_energy_provider;
  EnergyDecompositionProvider edecomp_energy_provider;
  std::unique_ptr<titan_energy_decomp::EnergyDecompositionEngine>
      edecomp_engine;
  std::unique_ptr<titan_energy_decomp::EndpointEntityDecompositionEngine>
      edecomp_gbsa_engine;
  std::unique_ptr<titan_energy_decomp::EndpointEntityDecompositionEngine>
      edecomp_pbsa_engine;

  dssp_mol1_analyzer.configure(Arguments.dssp);
  dssp_mol2_analyzer.configure(Arguments.dssp);
  dssp_mol1_analyzer.configure_segments(Arguments.segments, 1);
  dssp_mol2_analyzer.configure_segments(Arguments.segments, 2);
  hbk_analyzer.configure(Arguments.hbk);
  lj_analyzer.configure(Arguments.lj);
  sq_analyzer.configure(Arguments.sq);
  cmap_analyzer.configure(Arguments.cmap);
  interface_analyzer.configure(Arguments.interface_analysis,
                               Arguments.segments);
  dihedral_analyzer.configure(Arguments.dihedral);
  qnative_analyzer.configure(Arguments.qnative);
  msd_analyzer.configure(Arguments.msd);
  membrane_analyzer.configure(Arguments.membrane);
  s2_analyzer.configure(Arguments.s2);
  rg_analyzer.configure(Arguments.rg);
  rms_analyzer.configure(Arguments.rms);
  pbe_analyzer.configure(Arguments);
  perf_observer.configure(Arguments.perf_report.mode);

  std::chrono::milliseconds time_cal{}, time_fmap{},
      time_subgroups_cal{}, time_solve_pbc{}, time_sc{}, time_hb{}, time_pdb{},
      time_other{}, time_print_force{}, time_maxforce{}, time_contact{},
      time_input_init{}, time_input_stream{},
      time_cscd{0}, time_water{0}, time_na{0}, time_pbe{0};
  auto elapsed_ms_since = [](const std::chrono::steady_clock::time_point &start)
      -> double {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  Multi_MOL pro_in(force_f), lig(force_f), water_traj(force_f),
      membrane_traj(force_f);
  bool topo_initialized = false;
  bool charge_initialized = false;

  // --- IFP (Interaction Fingerprint) Initialization ---
  std::unique_ptr<TitanIFP> ifp_analyzer;
  std::unique_ptr<TitanAwhIfp> awh_ifp_analyzer;
  if (Arguments.ifp.flag_ifp) {
    ifp_analyzer = std::make_unique<TitanIFP>();
    TitanUI::print_info("[IFP] Interaction Fingerprint module enabled.");
  }
  std::ofstream ifp_pbctopo_filter_report;
  size_t ifp_pbctopo_total_frames = 0;
  size_t ifp_pbctopo_included_frames = 0;
  size_t ifp_pbctopo_missing_status_frames = 0;
  std::map<std::string, size_t> ifp_pbctopo_status_counts;
  if (Arguments.ifp.pbctopo_certified_only) {
    ifp_pbctopo_filter_report.open(
        info_output.get_ifp_pbctopo_frame_filter_filename());
    if (!ifp_pbctopo_filter_report.is_open()) {
      TitanUI::print_error(
          "[IFP] Cannot open PBCTOPO frame-filter audit output.");
      return 1;
    }
    ifp_pbctopo_filter_report
        << "trajectory_id,trajectory_frame_index,global_frame_index,time_ps,"
           "pbctopo_status,ifp_included\n";
  }
  auto include_ifp_frame =
      [&](const Multi_MOL &source, size_t chunk_frame_index,
          size_t trajectory_id, size_t trajectory_frame_index,
          size_t global_frame_index) {
        if (!Arguments.ifp.pbctopo_certified_only)
          return true;

        ++ifp_pbctopo_total_frames;
        const bool have_status =
            source.has_pbctopo_frame_status(chunk_frame_index);
        const std::string status =
            have_status
                ? std::string(source.pbctopo_frame_status(chunk_frame_index))
                : "missing_status";
        const bool included =
            have_status && source.pbctopo_frame_certified(chunk_frame_index);
        if (!have_status)
          ++ifp_pbctopo_missing_status_frames;
        if (included)
          ++ifp_pbctopo_included_frames;
        ++ifp_pbctopo_status_counts[status];

        const double time_ps =
            chunk_frame_index < source.m_mol.size()
                ? source.m_mol[chunk_frame_index].time_of_snap
                : std::numeric_limits<double>::quiet_NaN();
        ifp_pbctopo_filter_report
            << trajectory_id << ',' << trajectory_frame_index << ','
            << global_frame_index << ',' << std::setprecision(12) << time_ps
            << ',' << status << ',' << (included ? 1 : 0) << '\n';
        return included;
      };
  if (Arguments.awh.flag_awh_ifp) {
    awh_ifp_analyzer = std::make_unique<TitanAwhIfp>();
    const size_t expected_awh_walkers = input_line.empty() ? 1u : input_line.size();
    if (!awh_ifp_analyzer->initialize(Arguments, info_output,
                                      expected_awh_walkers)) {
      TitanUI::print_error(
          "AWH: failed to initialize RC-binned IFP workflow.");
      return 1;
    }
    TitanUI::print_info(
        "[AWH] RC-binned IFP workflow enabled for biased AWH analysis.");
  }
  std::chrono::milliseconds time_ifp{0};

  // === TICA workspace — hoisted outside outer loop (Fix 1: single reference
  // frame shared across all replicas for H-REMD cross-replica consistency) ===
  int tica_n_ca_global = 0, tica_feat_dim_global = 0;
  Eigen::VectorXd tica_features_global;
  Eigen::MatrixXd tica_mobile_global, tica_ref_centered_global;
  Eigen::Matrix3d tica_R_global = Eigen::Matrix3d::Identity();
  bool tica_ref_initialized_global = false;
  bool tica_use_lj_features_global = false;
  bool tica_use_observable_features_global = false;
  while (true) {
    const auto trajectory_timing_start = std::chrono::steady_clock::now();
    const auto time_gbsa_base = time_gbsa;
    const auto time_cas_base = time_cas;
    const auto time_dssp_base = time_dssp;
    const auto time_hbk_base = time_hbk;
    const auto time_lj_base = time_lj;
    const auto time_sq_base = time_sq;
    const auto time_cmap_base = time_cmap;
    const auto time_interface_base = time_interface;
    const auto time_dihedral_base = time_dihedral;
    const auto time_qnative_base = time_qnative;
    const auto time_msd_base = time_msd;
    const auto time_membrane_base = time_membrane;
    const auto time_s2_base = time_s2;
    const auto time_rg_base = time_rg;
    const auto time_rms_base = time_rms;
    const auto time_network_base = time_network;
    const auto time_tica_base = time_tica;
    const auto time_cal_base = time_cal;
    const auto time_sc_base = time_sc;
    const auto time_cscd_base = time_cscd;
    const auto time_hb_base = time_hb;
    const auto time_pdb_base = time_pdb;
    const auto time_fmap_base = time_fmap;
    const auto time_other_base = time_other;
    const auto time_water_base = time_water;
    const auto time_subgroups_cal_base = time_subgroups_cal;
    const auto time_na_base = time_na;
    const auto time_ifp_base = time_ifp;
    const auto time_input_init_base = time_input_init;
    const auto time_input_stream_base = time_input_stream;
    const auto time_solve_pbc_base = time_solve_pbc;
    const auto time_llps_base = time_llps;
    const auto time_pbe_base = time_pbe;

    if (!input_line.empty()) {
      if (Arguments.flag_all_folders == false) {
        name_lig = "lg_" + input_line[input_name - Arguments.b_input] +
                   Arguments.get_input_file_type();
        name_pro = "rec_" + input_line[input_name - Arguments.b_input] +
                   Arguments.get_input_file_type();
      } else {

        name_lig = fs::path(input_line[input_name - Arguments.b_input])
                       .parent_path()
                       .string() +
                   "/lg_" +
                   fs::path(input_line[input_name - Arguments.b_input])
                       .stem()
                       .string() +
                   Arguments.get_input_file_type();
        name_pro = fs::path(input_line[input_name - Arguments.b_input])
                       .parent_path()
                       .string() +
                   "/rec_" +
                   fs::path(input_line[input_name - Arguments.b_input])
                       .stem()
                       .string() +
                   Arguments.get_input_file_type();
      }
    } else if (Arguments.flag_1_pdb == false &&
               Arguments.flag_all_folders == false && Arguments.flag_listfile) {
      name_lig = Arguments.input_pattern;
      name_pro = Arguments.input_pattern;
    } else {
      name_lig =
          "lg_" + to_string(input_name) + Arguments.get_input_file_type();
      name_pro =
          "rec_" + to_string(input_name) + Arguments.get_input_file_type();
    }

    h_groups groups_1, groups_2;
    vector<energy_res> Energy_res_mol1, Energy_res_mol2, Force_res_mol1,
        Force_res_mol2;
    unordered_map<pair<string, size_t>, size_t, pair_hash> map_energy_res1,
        map_energy_res2;
    // vector<vector<com> > COM_vec_rec, COM_vec_lg;
    vector<energy_groups> Energy_groups_out;
    vector<fstream> out_cal_dis_files;
    traj_hb hb_rec_1traj, hb_lg_1traj;

    // Water Bridge Tracking
    if (Arguments.n_wat >= 0 && input_name == Arguments.b_input) {
      std::ofstream wb_out(info_output.get_global_water_bridges_timeseries());
      wb_out << "Traj_ID,Frame,Time_ps,Water_ID,Protein_Atom,Ligand_Atom,"
                "Distance_A,Angle_deg\n";
    }
    water_analyzer.reset_trajectory_state();

    auto start_input_init{std::chrono::steady_clock::now()};

    // --- Total File Size ---
    size_t total_traj_bytes = 1;
    std::error_code ec;
    std::string ref_file = Arguments.flag_1_pdb ? Arguments.pdb_name : name_pro;
    if (std::filesystem::exists(ref_file, ec)) {
      total_traj_bytes = std::filesystem::file_size(ref_file, ec);
      if (total_traj_bytes == 0)
        total_traj_bytes = 1;
    }

    // Initialize the topology (Frame 0) and setup groups
    if (!topo_initialized) {
      auto init_result = initialize_topology_streaming(
          Arguments, log, name_pro, name_lig, force_f, input_name, input_line,
          topo_2_is_open, topo_is_open, pro_in, lig, water_traj, membrane_traj,
          Groups_from_index_file, parameter, amine, info_output);

      if (!init_result) {
        if (Arguments.gist.flag_gist) {
          TitanUI::print_error("GIST initialization failed: " + init_result.error());
          return 1;
        }
        if (Arguments.flag_nocal) {
          TitanUI::print_info("Nocal mode is turned on!");
          return 0; // nocal: init parsed what it could, done
        }
        if (input_name > Arguments.b_input)
          --input_name;
        status = false;
        TitanUI::print_error("Initialization failed: " + init_result.error());
        break;
      }
      topo_initialized = true;

    } else {
      // Subsequent files: ONLY reload Frame 0 coordinates, skip topo parsing
      if (Arguments.flag_1_pdb && Arguments.flag_xdr) {
        // XDR multi-trajectory mode reuses the same reference topology (-s)
        // for every replica/chunk listed in -i. Re-parsing the reference via
        // pre_proc_pdb(..., -1, ...) clears topology-derived state
        // (parameters/index metadata) and can leave later analyses with an
        // inconsistent object graph when the next XTC/TRR starts streaming.
        // Keep the first parsed topology and restore the cached reference frame
        // instead; only the active trajectory filename changes per file.
        if (!saved_pro_frame0.empty())
          pro_in.m_mol = saved_pro_frame0;
        if (!saved_lig_frame0.empty())
          lig.m_mol = saved_lig_frame0;
        water_traj.m_mol.clear();
        if (!saved_membrane_frame0.empty())
          membrane_traj.m_mol = saved_membrane_frame0;
      } else {
        pro_in.m_mol.clear();
        lig.m_mol.clear();
        water_traj.m_mol.clear();
        membrane_traj.m_mol.clear();

        // Reload coordinates-only from new file (flag_raw=1 = coords only, no
        // topo)
        if (Arguments.flag_1_pdb) {
          // Preserve topology-derived parameters across replicas while
          // refreshing the per-file metadata copied from the reference
          // structure.
          pro_in.info.clear();
          lig.info.clear();
          if (Arguments.n_wat >= 0)
            water_traj.info.clear();
          if (Arguments.needs_membrane_coordinates())
            membrane_traj.info.clear();
          const int r =
              pre_proc_pdb(force_f, pro_in, lig, water_traj, membrane_traj,
                           Groups_from_index_file, -1, Arguments, amine,
                           info_output);
          if (r != 0) {
            TitanUI::print_error("Failed to reload coordinates for: " +
                                 Arguments.pdb_name);
            break;
          }
        } else {
          pro_in.take_coordinate(name_pro, force_f, Arguments, amine,
                                 info_output, 1, 1);
          lig.take_coordinate(name_lig, force_f, Arguments, amine, info_output,
                              1, 1);
        }
      }

      if (pro_in.m_mol.empty() ||
          (Arguments.needs_mol2_coordinates() && lig.m_mol.empty())) {
        TitanUI::print_error("Coordinate reload failed: empty mol for " +
                             name_pro);
        break;
      }

      if (Arguments.flag_1_pdb && !input_line.empty()) {
        if (Arguments.flag_xdr) {
          // XDR multi-replica: keep the -s reference PDB, switch trajectory
          Arguments.traj_file = input_line[input_name - Arguments.b_input];
        } else {
          Arguments.pdb_name = input_line[input_name - Arguments.b_input];
          Arguments.add_input_file_type();
        }
      }

      Arguments.setPdbname(Arguments.flag_1_pdb
                               ? Arguments.pdb_name
                               : name_lig.substr(name_lig.find("_")));
      info_output.setPdbName(Arguments);

      TitanUI::log_kv(
          log, "Topo Cache",
          "Reusing topology, reloaded coords from: " +
              (Arguments.flag_1_pdb ? Arguments.pdb_name : name_pro));
    }

    info_output.get_info_group(Groups_from_index_file);
    process_time_of_infiles(pro_in, Arguments, info_output, time_of_snap);
    set_chainnum(pro_in, lig, Arguments);

    if (Arguments.opdb == true && Arguments.flag_sep_opdb == false) {
      fstream out_pdb(Arguments.get_pdbOutput(), ios::out);
      out_pdb.close();
    }

    if (Arguments.com_dis_2mol != 0) {
      fstream output_dis_2mol_file(
          paths.get_path(input_name - Arguments.b_input) + "/" +
              info_output.getDisCom2Mol(),
          ios::out);
      output_dis_2mol_file.close();
    }

    // --- CHARGE INITIALIZATION (Frame 0) ---
    const bool needs_charge_init =
        !Arguments.flag_nocal || Arguments.flag_gbsa || Arguments.flag_alpb ||
        Arguments.cmap.flag_cmap || Arguments.qnative.flag_qnative ||
        Arguments.flag_cal_force || Arguments.flag_maxforce ||
        Arguments.flag_use_CI ||
        Arguments.flag_cal_dipole_moment ||
        Arguments.flag_charged_atom_distance_dis || Arguments.flag_cas ||
        Arguments.ifp.flag_ifp || Arguments.flag_pqr == 1 ||
        TitanFfLib::parameter_audit_path() != nullptr;
    if (!charge_initialized && !needs_charge_init) {
      TitanUI::log_kv(log, "Charge Init",
                      "Skipped (no charge-dependent module active)");
      charge_initialized = true;
    }
    if (!charge_initialized) {
      const bool initialize_mol2_parameters =
          Arguments.needs_mol2_coordinates() && !lig.m_mol.empty();
      const auto charge_policy = TitanFfLib::resolve_charge_source_policy(
          Arguments.flag_use_common_topo, topo_is_open, topo_2_is_open);
      TitanUI::log_kv(log, "Charge Source Mol1",
                      charge_policy.molecule1_uses_topology ? "topology"
                                                            : "lib_ff");
      if (initialize_mol2_parameters)
        TitanUI::log_kv(log, "Charge Source Mol2",
                        charge_policy.molecule2_uses_topology ? "topology"
                                                              : "lib_ff");
      pro_in.take_charge(
          amine, parameter, pro_in.m_mol[0].size(), 0ull, force_f,
          charge_policy.molecule1_uses_topology ? 1 : 0, &Arguments, &executor,
          "mol1");
      if (initialize_mol2_parameters)
        lig.take_charge(
            amine, parameter, lig.m_mol[0].size(), 0ull, force_f,
            charge_policy.molecule2_uses_topology ? 1 : 0, &Arguments,
            &executor, "mol2");
      if (const char *audit_path = TitanFfLib::parameter_audit_path()) {
        const auto audit = TitanFfLib::write_parameter_audit(
            audit_path, pro_in, initialize_mol2_parameters ? &lig : nullptr,
            charge_policy);
        if (!audit) {
          TitanUI::print_error(audit.error());
          TitanUI::log_kv(log, "Parameter Audit Error", audit.error());
          return 1;
        }
        TitanUI::log_kv(log, "Parameter Audit", audit_path);
      }
      charge_initialized = true;
    }

    auto_adjust_polymer_type(Arguments, pro_in, log, true);
    auto_adjust_polymer_type(Arguments, lig, log, false);
    const std::string resolved_mol1_type =
        (Arguments.flag_nocal && Arguments.lj.flag_lj) ? "lj"
                                                       : Arguments.mol_1_type;
    const std::string resolved_mol2_type =
        (Arguments.flag_nocal && Arguments.lj.flag_lj) ? "none"
                                                       : Arguments.mol_2_type;
    TitanUI::log_kv(log, "Resolved Molecule 1 Type", resolved_mol1_type);
    TitanUI::log_kv(log, "Resolved Molecule 2 Type", resolved_mol2_type);
    global_has_nucleic_acid =
        PolymerClassification::is_nucleic_type(Arguments.mol_1_type) ||
        PolymerClassification::is_nucleic_type(Arguments.mol_2_type) ||
        !PolymerClassification::detect_nucleic_polymer_type(pro_in.info).empty() ||
        !PolymerClassification::detect_nucleic_polymer_type(lig.info).empty();
    if (global_has_nucleic_acid) {
      na_profiler.initialize_if_needed(pro_in.Index, lig.Index, pro_in.info,
                                       lig.info);
    }

    cache_sigma_epsilon CachedData{Arguments.distance_energy_cutoff};
    CachedData.force_field_name = force_f;
    if (Arguments.flag_use_common_topo && !Arguments.flag_gbsa && !Arguments.flag_nocal) {
      try {
        CachedData.nonbonded_topology = titan_nb::read_topology(Arguments.topo_name);
        titan_nb::validate_selection(*CachedData.nonbonded_topology, pro_in.info);
        titan_nb::validate_selection(*CachedData.nonbonded_topology, lig.info);
        TitanUI::log_kv(log, "Nonbonded topology",
                        "source-bound exclusions and listed pairs; " +
                        std::to_string(CachedData.nonbonded_topology->listed_pairs.size()) + " listed pair(s)");
      } catch (const std::exception &error) {
        TitanUI::print_error(error.what());
        TitanUI::log_message(log, error.what());
        return 1;
      }
    }
    // Martini CG systems are too large for O(N×M) pair cache (~GB).
    // Energy functions handle empty cache via elec_ener.empty() checks.
    // Downstream emap/force_map are guarded to skip when cache is empty.
    const size_t pro_atom_count =
        !pro_in.m_mol.empty() ? pro_in.m_mol[0].size() : pro_in.info.size();
    const size_t lig_atom_count =
        !lig.m_mol.empty() ? lig.m_mol[0].size() : lig.info.size();

    if (pro_in.m_mol.empty() || lig.m_mol.empty()) {
      TitanUI::log_kv(log, "Frame0 Atom Counts",
                      "Fallback to topology info (mol1=" +
                          std::to_string(pro_atom_count) + ", mol2=" +
                          std::to_string(lig_atom_count) + ")");
    }

    std::string pair_cache_reason;
    const bool enable_dense_pair_cache = should_enable_dense_pair_cache(
        Arguments, pro_atom_count, lig_atom_count, available_ram,
        pair_cache_reason);
    TitanUI::log_kv(log, "Dense Pair Cache", pair_cache_reason);
    if (enable_dense_pair_cache) {
      CachedData.set_size(pro_atom_count * lig_atom_count, Arguments,
                          pro_atom_count, lig_atom_count);
    }
    TitanUI::log_kv(log, "Pair Cache Entries", std::to_string(CachedData.elec_ener.size()));

    if (lig.m_mol.empty())
      lig.m_mol.resize(1);

    // --- MARTINI TOPOLOGY ASSIGNMENT (charges, LJ params, chain IDs) ---
    // Must run BEFORE total_charge() so that parameters[].charge is set.
    if (force_f == "MARTINI") {
      pro_in.parse_martini_topology(Arguments, log, &CachedData);
      lig.parse_martini_topology(Arguments, log, &CachedData);
    }

    if ((force_f.find("CHARMM") != string::npos ||
         force_f.find("GROMOS") != string::npos) &&
        !parameter.nonbond_overrides.empty()) {
      size_t ntypes = 0;
      for (const auto &[atom_type, type_id] : parameter.atom_type_to_id) {
        (void)atom_type;
        ntypes = std::max(ntypes, type_id + 1);
      }

      const auto safe_mul = [](size_t a, size_t b, size_t &out) {
        if (a == 0 || b == 0) {
          out = 0;
          return true;
        }
        if (a > std::numeric_limits<size_t>::max() / b)
          return false;
        out = a * b;
        return true;
      };

      size_t dense_entries = 0;
      bool dense_ok = safe_mul(ntypes, ntypes, dense_entries);
      const long double dense_bytes =
          dense_ok ? static_cast<long double>(dense_entries) *
                         static_cast<long double>(sizeof(double) * 2)
                   : std::numeric_limits<long double>::infinity();
      const long double dense_cap_bytes =
          available_ram > 0
              ? std::min(128.0L * 1024.0L * 1024.0L,
                         static_cast<long double>(available_ram) * 0.02L)
              : (128.0L * 1024.0L * 1024.0L);

      if (ntypes > 0 && dense_ok && dense_bytes <= dense_cap_bytes) {
        std::vector<double> sigma_table(dense_entries, -1.0);
        std::vector<double> epsilon_table(dense_entries, -1.0);
        for (const auto &rule : parameter.nonbond_overrides) {
          const auto it_i = parameter.atom_type_to_id.find(rule.atom_type_i);
          const auto it_j = parameter.atom_type_to_id.find(rule.atom_type_j);
          if (it_i == parameter.atom_type_to_id.end() ||
              it_j == parameter.atom_type_to_id.end())
            continue;

          const size_t idx_ij = it_i->second * ntypes + it_j->second;
          const size_t idx_ji = it_j->second * ntypes + it_i->second;
          if (idx_ij >= sigma_table.size() || idx_ji >= sigma_table.size())
            continue;
          sigma_table[idx_ij] = rule.sigma;
          sigma_table[idx_ji] = rule.sigma;
          epsilon_table[idx_ij] = rule.epsilon;
          epsilon_table[idx_ji] = rule.epsilon;
        }
        CachedData.set_charmm_pair_table(std::move(sigma_table),
                                         std::move(epsilon_table),
                                         static_cast<int>(ntypes));
        TitanUI::log_kv(
            log, "Pair LJ Overrides",
            std::to_string(parameter.nonbond_overrides.size()) +
                " pair(s) loaded from [ nonbond_params ] (dense table, " +
                std::to_string(ntypes) + " type-id span)");
      } else {
        std::vector<std::tuple<std::uint64_t, double, double>> sparse_rules;
        sparse_rules.reserve(parameter.nonbond_overrides.size());
        for (const auto &rule : parameter.nonbond_overrides) {
          const auto it_i = parameter.atom_type_to_id.find(rule.atom_type_i);
          const auto it_j = parameter.atom_type_to_id.find(rule.atom_type_j);
          if (it_i == parameter.atom_type_to_id.end() ||
              it_j == parameter.atom_type_to_id.end())
            continue;

          const std::uint32_t lo =
              static_cast<std::uint32_t>(std::min(it_i->second, it_j->second));
          const std::uint32_t hi =
              static_cast<std::uint32_t>(std::max(it_i->second, it_j->second));
          const std::uint64_t key =
              (static_cast<std::uint64_t>(lo) << 32) |
              static_cast<std::uint64_t>(hi);
          sparse_rules.emplace_back(key, rule.sigma, rule.epsilon);
        }

        std::sort(sparse_rules.begin(), sparse_rules.end(),
                  [](const auto &lhs, const auto &rhs) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                  });

        std::vector<std::uint64_t> sparse_keys;
        std::vector<double> sparse_sigma;
        std::vector<double> sparse_epsilon;
        sparse_keys.reserve(sparse_rules.size());
        sparse_sigma.reserve(sparse_rules.size());
        sparse_epsilon.reserve(sparse_rules.size());
        for (const auto &[key, sigma, epsilon] : sparse_rules) {
          if (!sparse_keys.empty() && sparse_keys.back() == key) {
            sparse_sigma.back() = sigma;
            sparse_epsilon.back() = epsilon;
            continue;
          }
          sparse_keys.push_back(key);
          sparse_sigma.push_back(sigma);
          sparse_epsilon.push_back(epsilon);
        }

        CachedData.set_charmm_pair_sparse(std::move(sparse_keys),
                                          std::move(sparse_sigma),
                                          std::move(sparse_epsilon));
        TitanUI::log_kv(
            log, "Pair LJ Overrides",
            std::to_string(parameter.nonbond_overrides.size()) +
                " pair(s) loaded from [ nonbond_params ] (sparse table, " +
                std::to_string(ntypes) + " type-id span)");
      }
    }

    double chg_pro = pro_in.total_charge();
    double chg_lig = lig.total_charge();

    TitanUI::log_section(log, "SYSTEM TOPOLOGY & TRAJECTORY");
    std::string current_input_file;
    if (Arguments.flag_xdr) {
      current_input_file = "PDB=" + Arguments.pdb_name;
      if (!Arguments.traj_file.empty())
        current_input_file += "; TRAJ=" + Arguments.traj_file;
    } else if (Arguments.flag_1_pdb) {
      current_input_file = Arguments.pdb_name;
    } else {
      current_input_file = "REC=" + name_pro + "; LIG=" + name_lig;
    }
    TitanUI::log_kv(log, "Current Input File", current_input_file);
    const std::string receptor_group_label =
        info_output.get_group_1_name() + " [" +
        std::to_string(Arguments.group_1) + "] (" +
        std::to_string(pro_in.info.size()) + " atoms)";
    const std::string ligand_group_label =
        info_output.get_group_2_name() + " [" +
        std::to_string(Arguments.group_2) + "] (" +
        std::to_string(lig.info.size()) + " atoms)";
    TitanUI::log_kv(log, "Receptor Group",
                    receptor_group_label);
    TitanUI::log_kv(log, "Ligand Group",
                    ligand_group_label);

    auto format_double = [](double v) {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    };
    TitanUI::log_kv(log, "Receptor Charge", format_double(chg_pro) + " e");
    TitanUI::log_kv(log, "Ligand Charge", format_double(chg_lig) + " e");
    if (force_f == "MARTINI") {
      TitanUI::log_kv(log, "Martini LJ",
                      "pair-specific matrix ([ nonbond_params ])");
      TitanUI::log_kv(log, "Martini Electrostatics",
                      martini::electrostatics_summary(Arguments));
    }

    if (Arguments.flag_sub_groups && !sg_streaming_initialized) {
      sg_streaming_analyzer.init(Groups_from_index_file, pro_in, lig,
                                 CachedData, Arguments, available_ram, &log);
      sg_streaming_initialized = true;
    }

    if (Arguments.de.flag_find_pulling_direct) {
      auto snapshot_status = prepare_pull_direct_snapshot(
          Arguments, DE_arguments, force_f, name_pro, name_lig, pro_in, lig,
          water_traj, Groups_from_index_file, amine, info_output, executor,
          num_threads, log);
      if (!snapshot_status) {
        TitanUI::print_error(snapshot_status.error());
        TitanUI::log_message(log, "[Pull-direct] " + snapshot_status.error());
        return 1;
      }
      return find_pull_direct(pro_in, lig, CachedData, DE_arguments, Arguments,
                              info_output, executor, Groups_from_index_file);
    }

    const AlignedDoubleVector &rec_vdw_radii = pro_in.parameters.radius;
    const AlignedDoubleVector &lg_vdw_radii = lig.parameters.radius;

    if (Arguments.flag_cas)
      cas_analyzer.init(pro_in, lig, Arguments.mol_2_type);
    const bool dssp_allow_mol2 =
        !Arguments.flag_nocal ||
        Arguments.needs_mol2_for_interface_analysis() ||
        Arguments.needs_mol2_for_cmap_interface() ||
        Arguments.needs_mol2_for_qnative_interface() ||
        Arguments.needs_mol2_for_msd();
    const bool dssp_mol1_requested =
        Arguments.dssp.flag_dssp &&
        TitanDSSPAnalyzer::supports_moltype(Arguments.mol_1_type);
    const bool dssp_mol2_requested =
        Arguments.dssp.flag_dssp && dssp_allow_mol2 &&
        TitanDSSPAnalyzer::supports_moltype(Arguments.mol_2_type);
    const std::size_t dssp_mol1_compatible_residues =
        TitanDSSPAnalyzer::compatible_backbone_residue_count(pro_in.info);
    const std::size_t dssp_mol2_compatible_residues =
        TitanDSSPAnalyzer::compatible_backbone_residue_count(lig.info);
    const bool dssp_mol1_has_backbone =
        dssp_mol1_compatible_residues >= 3;
    const bool dssp_mol2_has_backbone =
        dssp_mol2_compatible_residues >= 3;

    if (dssp_mol1_requested && !dssp_mol1_has_backbone) {
      const std::string message =
          Arguments.lj.flag_lj
              ? "[DSSP] Auto-skipping mol1: no DSSP-compatible protein "
                "backbone detected (typical for LJ/monoatomic jobs)."
              : "[DSSP] Auto-skipping mol1: no DSSP-compatible protein "
                "backbone detected in topology.";
      TitanUI::print_info(message);
      TitanUI::log_message(log, message);
      TitanUI::log_kv(log, "DSSP mol1",
                      "auto-skipped (compatible backbone residues=" +
                          std::to_string(dssp_mol1_compatible_residues) +
                          ", selected atoms=" +
                          std::to_string(pro_in.info.size()) + ")");
    }
    if (dssp_mol1_requested && dssp_mol1_has_backbone) {
      try {
        dssp_mol1_analyzer.init(pro_in, "mol1");
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
      TitanUI::log_kv(log, "DSSP mol1 residues",
                      std::to_string(dssp_mol1_analyzer.residue_count()));
    }
    if (dssp_mol2_requested && !dssp_mol2_has_backbone) {
      const std::string message =
          "[DSSP] Auto-skipping mol2: no DSSP-compatible protein backbone "
          "detected in topology.";
      TitanUI::print_info(message);
      TitanUI::log_message(log, message);
      TitanUI::log_kv(log, "DSSP mol2",
                      "auto-skipped (compatible backbone residues=" +
                          std::to_string(dssp_mol2_compatible_residues) +
                          ", selected atoms=" +
                          std::to_string(lig.info.size()) + ")");
    }
    if (dssp_mol2_requested && dssp_mol2_has_backbone) {
      try {
        dssp_mol2_analyzer.init(lig, "mol2");
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
      TitanUI::log_kv(log, "DSSP mol2 residues",
                      std::to_string(dssp_mol2_analyzer.residue_count()));
    }
    if (Arguments.lj.flag_lj) {
      lj_analyzer.init(pro_in, Arguments, log);
    }
    if (Arguments.hbk.flag_hbk) {
      const bool water_stream_available =
          Arguments.n_wat >= 0 && !water_traj.m_mol.empty();
      const Multi_MOL *lig_ptr = lig.m_mol.empty() ? nullptr : &lig;
      hbk_analyzer.init(pro_in, lig_ptr, water_stream_available, Arguments, log);
    }
    if (Arguments.sq.flag_sq) {
      sq_analyzer.init(pro_in, Arguments, log);
    }
    if (Arguments.cmap.flag_cmap) {
      const Multi_MOL *cmap_mol2 =
          Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
      cmap_analyzer.init(pro_in, cmap_mol2, Arguments, log);
    }
    if (Arguments.interface_analysis.flag_interface_analysis) {
      try {
        const Multi_MOL *interface_mol2 =
            Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
        interface_analyzer.init(pro_in, interface_mol2, Arguments, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
    }
    if (Arguments.dihedral.flag_dihedral) {
      try {
        dihedral_analyzer.init(pro_in, Arguments, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
    }
    if (Arguments.qnative.flag_qnative) {
      Multi_MOL qnative_ref_pro(force_f), qnative_ref_lig(force_f),
          qnative_ref_water(force_f), qnative_ref_membrane(force_f);
      const Multi_MOL *qnative_ref_pro_ptr = &pro_in;
      const Multi_MOL *qnative_mol2 =
          Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
      if (Arguments.qnative.ref_explicit) {
        arguments qnative_ref_args = Arguments;
        qnative_ref_args.pdb_name = Arguments.qnative.ref_file;
        qnative_ref_args.flag_1_pdb = true;
        qnative_ref_args.flag_xdr = false;
        qnative_ref_args.traj_file.clear();
        const int ref_status = pre_proc_pdb(
            force_f, qnative_ref_pro, qnative_ref_lig, qnative_ref_water,
            qnative_ref_membrane, Groups_from_index_file, 0, qnative_ref_args,
            amine, info_output);
        if (ref_status != 0 || qnative_ref_pro.m_mol.empty()) {
          TitanUI::print_error("QNative: failed to load -qnative-ref " +
                               Arguments.qnative.ref_file);
          return 1;
        }
        if (qnative_ref_pro.info.empty() ||
            qnative_ref_pro.parameters.size() < qnative_ref_pro.info.size()) {
          TitanUI::print_error(
              "QNative: -qnative-ref molecule 1 topology/parameters are incomplete.");
          return 1;
        }
        if (Arguments.needs_mol2_for_qnative_interface()) {
          if (qnative_ref_lig.m_mol.empty()) {
            TitanUI::print_error(
                "QNative: -qnative-scope=interface requires molecule 2 in -qnative-ref.");
            return 1;
          }
          if (qnative_ref_lig.info.empty() ||
              qnative_ref_lig.parameters.size() < qnative_ref_lig.info.size()) {
            TitanUI::print_error(
                "QNative: -qnative-ref molecule 2 topology/parameters are incomplete.");
            return 1;
          }
          qnative_mol2 = &qnative_ref_lig;
        }
        qnative_ref_pro_ptr = &qnative_ref_pro;
      }
      qnative_analyzer.init(
          *qnative_ref_pro_ptr, qnative_mol2, Arguments, log,
          Arguments.qnative.ref_explicit ? &pro_in : nullptr,
          (Arguments.qnative.ref_explicit &&
           Arguments.needs_mol2_for_qnative_interface())
              ? &lig
              : nullptr);
    }
    if (Arguments.msd.flag_msd) {
      try {
        msd_analyzer.init(pro_in,
                          Arguments.needs_mol2_for_msd() ? &lig : nullptr,
                          Arguments, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
    }
    if (Arguments.membrane.flag_membrane) {
      try {
        membrane_analyzer.init(membrane_traj, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
    }
    if (Arguments.s2.flag_s2) {
      try {
        s2_analyzer.init(pro_in, Arguments, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
    }
    if (Arguments.rg.flag_rg2) {
      rg_analyzer.init(pro_in, lig, Arguments, log);
    }
    if (Arguments.rms.flag_rms) {
      rms_analyzer.init(pro_in, lig, Arguments, log);
    }

    if (Arguments.flag_network || Arguments.flag_pca ||
        Arguments.corr_knn.flag_corr_knn) {
      if (!dccm_analyzer_initialized) {
        dccm_analyzer.init(pro_in, lig, Arguments);
        dccm_analyzer_initialized = true;
      }
      const bool needs_side_trace_cache =
          Arguments.flag_allostery ||
          (Arguments.corr_knn.flag_corr_knn &&
           Arguments.corr_knn.pair_use_contact_prior);
      TitanUI::log_kv(log, "Network/PCA Nodes",
                      std::to_string(dccm_analyzer.get_resnrs().size()));
      TitanUI::log_kv(
          log, "DCCM Trace Scope",
          Arguments.flag_nocal
              ? "mol1 only (-nocal; mol2 trace nodes disabled)"
              : "mol1 + mol2 (compatible trace nodes only)");
      TitanUI::log_kv(log, "DCCM Trace Cache",
                      dccm_analyzer.stores_full_trajectory()
                          ? (needs_side_trace_cache
                                 ? "full CA + Cbeta/N trajectory"
                                 : "full CA trajectory")
                          : "latest CA frame only (TICA streaming)");
    }
    // LSP-MD standalone: init DCCM data collector without DCCM/PCA
    if (Arguments.lspmd.flag_lspmd && !Arguments.flag_network &&
        !Arguments.flag_pca) {
      if (!dccm_analyzer_initialized) {
        dccm_analyzer.init(pro_in, lig, Arguments);
        dccm_analyzer_initialized = true;
      }
      const bool needs_side_trace_cache =
          Arguments.flag_allostery ||
          (Arguments.corr_knn.flag_corr_knn &&
           Arguments.corr_knn.pair_use_contact_prior);
      TitanUI::log_kv(log, "LSP-MD Nodes",
                      std::to_string(dccm_analyzer.get_resnrs().size()));
      TitanUI::log_kv(
          log, "DCCM Trace Scope",
          Arguments.flag_nocal
              ? "mol1 only (-nocal; mol2 trace nodes disabled)"
              : "mol1 + mol2 (compatible trace nodes only)");
      TitanUI::log_kv(log, "DCCM Trace Cache",
                      dccm_analyzer.stores_full_trajectory()
                          ? (needs_side_trace_cache
                                 ? "full CA + Cbeta/N trajectory"
                                 : "full CA trajectory")
                          : "latest CA frame only (TICA streaming)");
    }

    // --- STREAMING ARCHITECTURE INITIALIZATION ---
    ChunkStreamer streamer(Arguments, info_output, Groups_from_index_file,
                           force_f, executor, num_threads);

    auto stream_ok =
        streamer.open_streams(name_pro, name_lig, pro_in, lig, membrane_traj);
    if (!stream_ok) {
      TitanUI::print_error(stream_ok.error());
      return 1;
    }

    size_t available_ram_bytes = get_available_ram();
    size_t CHUNK_SIZE = 100; // fallback default

    if (total_traj_bytes > 0 && available_ram_bytes > 0 &&
        total_traj_bytes <= available_ram_bytes * 0.5 && !Arguments.flag_xdr) {
      // File fits in memory AND is PDB — load everything at once
      CHUNK_SIZE = std::numeric_limits<size_t>::max();
      TitanUI::log_kv(log, "Memory Strategy",
                      "In-Memory Execution (File < 50% RAM)");
    } else if (Arguments.flag_xdr) {
      // XDR: stream conservatively. XTC/TRR frames are expanded into
      // double-precision coordinate buffers plus per-frame topology copies,
      // which makes GBSA especially memory-hungry on large systems.
      const size_t selected_atoms_per_frame =
          Groups_from_index_file.groups_atom_number(
              Groups_from_index_file.group_1_index()) +
          (!Arguments.flag_nocal
               ? Groups_from_index_file.groups_atom_number(
                     Groups_from_index_file.group_2_index())
               : 0) +
          (Arguments.n_wat >= 0
               ? Groups_from_index_file.groups_atom_number(Arguments.n_wat)
               : 0) +
          (Arguments.needs_membrane_coordinates() &&
                   Arguments.membrane.resolved_group >= 0
               ? Groups_from_index_file.groups_atom_number(
                     static_cast<size_t>(Arguments.membrane.resolved_group))
               : 0);

      // 3 coordinates * 8 bytes, multiplied by a safety factor to account for
      // frame objects, box/time metadata, copied topology fields, and transient
      // GBSA work buffers.
      const size_t coord_bytes =
          selected_atoms_per_frame * 3ULL * sizeof(double);
      const size_t frame_safety_factor = Arguments.flag_gbsa ? 8ULL : 4ULL;
      const size_t bytes_per_frame =
          std::max(coord_bytes * frame_safety_factor, size_t(1));

      const size_t target_chunk_bytes =
          Arguments.flag_gbsa ? (16ULL * 1024ULL * 1024ULL)
                              : (64ULL * 1024ULL * 1024ULL);
      const size_t ram_limited_chunk_bytes =
          available_ram_bytes > 0
              ? std::max<size_t>(8ULL * 1024ULL * 1024ULL,
                                 available_ram_bytes /
                                     (Arguments.flag_gbsa ? 16ULL : 8ULL))
              : target_chunk_bytes;
      const size_t effective_chunk_budget =
          std::min(target_chunk_bytes, ram_limited_chunk_bytes);

      const size_t max_xdr_chunk = Arguments.flag_gbsa ? 4ULL : 64ULL;
      CHUNK_SIZE = std::max(
          size_t(1),
          std::min(max_xdr_chunk, effective_chunk_budget / bytes_per_frame));

      TitanUI::log_kv(log, "XDR Selected Atoms",
                      std::to_string(selected_atoms_per_frame));
      TitanUI::log_kv(log, "XDR Frame Memory Est",
                      std::to_string(bytes_per_frame / (1024ULL * 1024ULL)) +
                          " MB");
      TitanUI::log_kv(log, "Memory Strategy",
                      "XDR Streaming (Adaptive Chunk = " +
                          std::to_string(CHUNK_SIZE) + " frame(s), budget " +
                          std::to_string(effective_chunk_budget /
                                         (1024ULL * 1024ULL)) +
                          " MB)");
    } else {
      // Adaptive chunk: target ~256 MB per chunk, clamped [10, 5000] frames
      size_t atoms_per_frame = pro_in.info.size() + lig.info.size();
      size_t bytes_per_frame = std::max(atoms_per_frame * 80, size_t(1));
      constexpr size_t TARGET_CHUNK_BYTES = 256ULL * 1024 * 1024; // 256 MB
      CHUNK_SIZE =
          std::max(size_t(10), std::min(size_t(5000),
                                        TARGET_CHUNK_BYTES / bytes_per_frame));
      TitanUI::log_kv(log, "Memory Strategy",
                      "Out-of-Core Streaming (Adaptive Chunk = " +
                          std::to_string(CHUNK_SIZE) + " frames, " +
                          std::to_string(bytes_per_frame / 1024) +
                          " KB/frame)");
    }

    time_input_init += std::chrono::duration_cast<milliseconds>(
        std::chrono::steady_clock::now() - start_input_init);

    // --- TICA INITIALIZATION (after DCCM init, uses Cα count) ---
    if (Arguments.msm.flag_tica && !tica_analyzer) {
      int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
      const bool use_lj_tica_features =
          Arguments.feature_export.source == "lj" ||
          (!Arguments.feature_export.source_explicit && Arguments.lj.flag_lj &&
           lj_analyzer.phase2_ready() && n_ca == 0);
      if (Arguments.feature_export.source == "lj" &&
          !lj_analyzer.phase2_ready()) {
        TitanUI::print_error(
            "Feature export: -feature-source=lj requires active LJ phase-2 features.");
        return 1;
      }
      const bool use_observable_tica_features =
          feature_source_is_observable_tica(Arguments);
      std::string tica_feature_source =
          use_observable_tica_features
              ? ("observable feature source: " + Arguments.feature_export.source)
              : (use_lj_tica_features
                     ? "LJ condensation phase-2 features (density, cluster, CN, shell)"
                     : "pairwise C-alpha distances");
      // Science fix: pairwise inter-Cα distances (rotation+translation
      // invariant) D = N_ca*(N_ca-1)/2 — eliminates 6 spurious DOF from raw xyz
      std::vector<FeatureColumn> feature_columns;
      if (Arguments.feature_export.export_features ||
          use_observable_tica_features) {
        feature_columns = build_observable_feature_columns(
            Arguments, n_ca, qnative_analyzer, dihedral_analyzer,
            cmap_analyzer, lj_analyzer, use_lj_tica_features);
      }
      int tica_feature_dim = use_observable_tica_features
                                 ? static_cast<int>(feature_columns.size())
                                 : (use_lj_tica_features
                                        ? lj_analyzer.feature_dim()
                                        : n_ca * (n_ca - 1) / 2);
      if (use_observable_tica_features && tica_feature_dim <= 0) {
        TitanUI::print_error(
            "Feature export: selected -feature-source produced zero TICA features.");
        return 1;
      }
      const double selected_dt_ps =
          Arguments.dt > 0 ? static_cast<double>(Arguments.dt)
                           : info_output.get_current_dt();
      const double tica_dt_ns =
          selected_dt_ps > 0.0 ? selected_dt_ps / 1000.0 : 0.001;
      TicaBackend requested_backend = TicaBackend::Auto;
      if (!tica_backend_from_string(Arguments.msm.tica_backend,
                                    requested_backend)) {
        requested_backend = TicaBackend::Auto;
      }

      auto bytes_to_mb_string = [](long double bytes) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1)
            << (bytes / (1024.0L * 1024.0L)) << " MB";
        return oss.str();
      };

      const size_t atoms_per_frame =
          std::max<size_t>(1, pro_in.info.size() + lig.info.size());
      long double bytes_per_frame_estimate =
          static_cast<long double>(atoms_per_frame) *
          static_cast<long double>(Arguments.flag_xdr ? 12.0 : 80.0);
      if (bytes_per_frame_estimate < 1.0L) {
        bytes_per_frame_estimate = 1.0L;
      }
      long double estimated_total_traj_bytes =
          static_cast<long double>(std::max<size_t>(1, total_traj_bytes)) *
          static_cast<long double>(std::max<size_t>(1, Arguments.number_of_files));
      TitanTicaFrameEstimateConfig frame_estimate_config;
      frame_estimate_config.begin_time_ps = Arguments.begin_time;
      frame_estimate_config.end_time_ps = Arguments.end_time;
      frame_estimate_config.dt_ps = static_cast<double>(Arguments.dt);
      frame_estimate_config.begin_set = Arguments.flag_begin_set;
      frame_estimate_config.end_set = Arguments.flag_end_set;
      frame_estimate_config.trajectory_count =
          input_line.empty() ? 0 : input_line.size();
      frame_estimate_config.number_of_files = Arguments.number_of_files;
      frame_estimate_config.fallback_estimated_frames =
          estimated_total_traj_bytes / bytes_per_frame_estimate;
      if (Arguments.flag_xdr && Arguments.xdr_type == "xtc")
        frame_estimate_config.xtc_paths = input_line;
      const TitanTicaFrameEstimate frame_estimate =
          estimate_tica_selected_frames(frame_estimate_config);
      long double estimated_frames =
          static_cast<long double>(std::max<std::size_t>(1, frame_estimate.frames));
      const int sketch_rank =
          std::max(Arguments.msm.tica_components, Arguments.msm.tica_rank);
      const long double dense_memory_estimate =
          6.0L * static_cast<long double>(tica_feature_dim) *
          static_cast<long double>(tica_feature_dim) *
          static_cast<long double>(sizeof(double));
      const long double dual_memory_estimate =
          3.0L * estimated_frames * static_cast<long double>(tica_feature_dim) *
          static_cast<long double>(sizeof(double));
      const long double randomized_memory_estimate =
          (estimated_frames * static_cast<long double>(sketch_rank + 8) +
           static_cast<long double>(tica_feature_dim) *
               static_cast<long double>(2 * (sketch_rank + 8))) *
          static_cast<long double>(sizeof(double));
      TitanTicaBackendPolicyConfig backend_policy_config;
      backend_policy_config.requested_backend = requested_backend;
      backend_policy_config.available_ram_bytes =
          static_cast<long double>(available_ram_bytes);
      backend_policy_config.ram_budget_fraction = 0.30L;
      backend_policy_config.dense_memory_estimate = dense_memory_estimate;
      backend_policy_config.dual_memory_estimate = dual_memory_estimate;
      backend_policy_config.estimated_frames = estimated_frames;
      backend_policy_config.feature_dim = tica_feature_dim;
      backend_policy_config.need_projection_cache = need_tica_projection_cache;
      const TitanTicaBackendPolicyDecision backend_decision =
          select_tica_backend_for_policy(backend_policy_config);
      const TicaBackend selected_backend = backend_decision.selected_backend;

      long double hard_memory_cap_bytes = backend_decision.ram_budget;
      if (hard_memory_cap_bytes <= 0.0L) {
        constexpr long double kExactDualPeakFactor = 3.0L;
        hard_memory_cap_bytes = backend_decision.large_matrix_value_limit *
                                sizeof(double) * kExactDualPeakFactor;
      }
      const size_t hard_memory_cap = static_cast<size_t>(std::min<long double>(
          hard_memory_cap_bytes,
          static_cast<long double>(std::numeric_limits<size_t>::max())));

      tica_analyzer = std::make_unique<TitanKinetics>(
          tica_feature_dim, Arguments.msm.tica_lag,
          Arguments.msm.tica_components, tica_dt_ns, selected_backend,
          Arguments.msm.tica_rank, hard_memory_cap,
          selected_backend == TicaBackend::ExactDual);
      tica_use_lj_features_global = use_lj_tica_features;
      tica_use_observable_features_global = use_observable_tica_features;
      if (Arguments.feature_export.export_features ||
          use_observable_tica_features) {
        if (feature_columns.empty()) {
          feature_columns = build_observable_feature_columns(
              Arguments, n_ca, qnative_analyzer, dihedral_analyzer,
              cmap_analyzer, lj_analyzer, use_lj_tica_features);
        }
        feature_router.configure(Arguments.feature_export.export_features,
                                 use_observable_tica_features,
                                 Arguments.feature_export.source);
        feature_router.begin_trajectory(std::move(feature_columns),
                                        Arguments.feature_export.source);
      }
      TitanUI::log_kv(log, "TICA Feature Source", tica_feature_source);
      TitanUI::log_kv(
          log, "TICA Trace Scope",
          use_observable_tica_features
              ? "observable feature router"
              : use_lj_tica_features
              ? "mol1 only (LJ feature backend)"
              : (Arguments.flag_nocal
                     ? "mol1 only (-nocal; mol2 ignored for DCCM/TICA)"
                     : "mol1 + mol2 trace nodes (if compatible)"));
      if (use_lj_tica_features) {
        TitanUI::log_kv(log, "TICA LJ Atoms",
                        std::to_string(lj_analyzer.atom_count()));
        TitanUI::log_kv(log, "TICA Feature Dim (LJ phase-2)",
                        std::to_string(tica_feature_dim));
      }
      if (!use_lj_tica_features && !use_observable_tica_features) {
      TitanUI::log_kv(log, "TICA Cα Atoms", std::to_string(n_ca));
        TitanUI::log_kv(log, "TICA Feature Dim (pairwise dist)",
                        std::to_string(tica_feature_dim));
      }
      if (use_observable_tica_features) {
        TitanUI::log_kv(log, "TICA Feature Dim (observable)",
                        std::to_string(tica_feature_dim));
      }
      TitanUI::log_kv(log, "TICA Lag Time",
                      std::to_string(Arguments.msm.tica_lag) + " frames");
      TitanUI::log_kv(log, "TICA Components",
                      std::to_string(Arguments.msm.tica_components));
      TitanUI::log_kv(log, "TICA Backend Requested",
                      tica_backend_to_string(requested_backend));
      TitanUI::log_kv(log, "TICA Backend Selected",
                      tica_backend_to_string(selected_backend));
      if (backend_decision.large_matrix_limit_exceeded) {
        TitanUI::log_kv(log, "TICA Large-Matrix Policy",
                        "exact covariance/replay selected by memory budget; approximation requires explicit opt-in");
      }
      TitanUI::log_kv(log, "TICA Rank",
                      std::to_string(Arguments.msm.tica_rank));
      TitanUI::log_kv(log, "TICA Estimated Frames",
                      std::to_string(static_cast<int64_t>(estimated_frames)));
      TitanUI::log_kv(log, "TICA Frame Estimate Source",
                      frame_estimate.source);
      if (!frame_estimate.diagnostic.empty()) {
        TitanUI::log_kv(log, "TICA Frame Scan Fallback",
                        frame_estimate.diagnostic);
        TitanUI::print_warning("TICA: " + frame_estimate.diagnostic +
                               "; using " + frame_estimate.source +
                               " estimate.");
      }
      TitanUI::log_kv(
          log, "TICA Raw Feature Values",
          std::to_string(static_cast<int64_t>(
              std::ceil(backend_decision.raw_feature_values))));
      TitanUI::log_kv(
          log, "TICA Large-Matrix Limit",
          std::to_string(static_cast<int64_t>(
              std::ceil(backend_decision.large_matrix_value_limit))));
      TitanUI::log_kv(log, "TICA Dense Memory Est",
                      bytes_to_mb_string(dense_memory_estimate));
      TitanUI::log_kv(log, "TICA Dual Cache Est",
                      bytes_to_mb_string(dual_memory_estimate));
      TitanUI::log_kv(log, "TICA Randomized Est",
                      bytes_to_mb_string(randomized_memory_estimate));
      TitanUI::log_kv(log, "TICA Hard Memory Cap",
                      bytes_to_mb_string(hard_memory_cap_bytes));
    }

    if (Arguments.msm.flag_tica && tica_analyzer &&
        (Arguments.feature_export.export_features ||
         tica_use_observable_features_global) &&
        input_name > Arguments.b_input) {
      const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
      auto feature_columns = build_observable_feature_columns(
          Arguments, n_ca, qnative_analyzer, dihedral_analyzer, cmap_analyzer,
          lj_analyzer, tica_use_lj_features_global);
      feature_router.configure(Arguments.feature_export.export_features,
                               tica_use_observable_features_global,
                               Arguments.feature_export.source);
      feature_router.begin_trajectory(std::move(feature_columns),
                                      Arguments.feature_export.source);
    }

    if (!Arguments.msm.flag_tica && Arguments.feature_export.export_features) {
      const int n_ca = static_cast<int>(dccm_analyzer.get_resnrs().size());
      auto feature_columns = build_observable_feature_columns(
          Arguments, n_ca, qnative_analyzer, dihedral_analyzer, cmap_analyzer,
          lj_analyzer, false);
      if (feature_columns.empty()) {
        TitanUI::print_warning(
            "Feature export: no active frame feature producers were resolved.");
      } else {
        feature_router.configure(true, false, Arguments.feature_export.source);
        feature_router.begin_trajectory(std::move(feature_columns),
                                        Arguments.feature_export.source);
        TitanUI::log_kv(log, "Feature Export Dim",
                        std::to_string(feature_router.feature_dim()));
      }
    }

    // --- LLPS INITIALIZATION (after Martini topo parsed, chainnum set) ---
    if (Arguments.llps.flag_llps && !llps_analyzer) {
      // Build chain_start / chain_len from pro_in.info (chainnum assigned by
      // impl_martini)
      const auto &mol_info_vec = pro_in.info;
      const int n_info = static_cast<int>(mol_info_vec.size());
      int n_chains_llps = 0;
      for (int i = 0; i < n_info; ++i)
        n_chains_llps =
            std::max(n_chains_llps,
                     static_cast<int>(mol_info_vec[i].get_chainnum()) + 1);
      std::vector<int> llps_chain_start(n_chains_llps, -1);
      std::vector<int> llps_chain_len(n_chains_llps, 0);
      for (int i = 0; i < n_info; ++i) {
        int c = static_cast<int>(mol_info_vec[i].get_chainnum());
        if (c < 0 || c >= n_chains_llps)
          continue;
        if (llps_chain_start[c] < 0)
          llps_chain_start[c] = i;
        llps_chain_len[c]++;
      }
      llps_analyzer = std::make_unique<TitanLLPS>(
          n_chains_llps, llps_chain_start, llps_chain_len,
          Arguments.llps.llps_rcut, Arguments.llps.llps_perc_thr,
          Arguments.llps.llps_dimer_size, Arguments.llps.llps_contacts,
          Arguments.llps.llps_voxel_size, Arguments.llps.llps_dynamics,
          Arguments.llps.llps_dynamics_detailed,
          Arguments.llps.llps_prna_rcut, Arguments.llps.llps_kernel,
          Arguments.llps.llps_density);
      llps_analyzer->configure_phase({Arguments.llps.llps_phase_geometry,
          Arguments.llps.llps_phase_axis, Arguments.llps.llps_phase_dense,
          Arguments.llps.llps_phase_dilute_inner, Arguments.llps.llps_phase_dilute_outer});
      llps_analyzer->set_sticker_beads(mol_info_vec,
                                       Arguments.llps.llps_sticker_beads);
      if ((Arguments.mol_2_type.find("rna") != std::string::npos ||
           Arguments.mol_2_type.find("RNA") != std::string::npos) &&
          !lig.info.empty()) {
        llps_analyzer->set_rna_metadata(lig.info);
      }
      llps_analyzer->configure_trace_outputs(info_output);
      TitanUI::log_section(log, "LLPS Run Context");
      TitanUI::log_kv(log, "LLPS Enabled", "Yes");
      TitanUI::log_kv(log, "LLPS Chains", std::to_string(n_chains_llps));
      TitanUI::log_kv(log, "LLPS Macro-Nodes",
                      std::to_string(llps_analyzer->n_macronodes()));
      TitanUI::log_kv(log, "LLPS Dimer Size",
                      std::to_string(Arguments.llps.llps_dimer_size));
      TitanUI::log_kv(log, "LLPS Contact Rcut (nm)",
                      std::to_string(Arguments.llps.llps_rcut));
      TitanUI::log_kv(log, "LLPS Percolation Threshold",
                      std::to_string(Arguments.llps.llps_perc_thr));
      TitanUI::log_kv(log, "LLPS Density Mode",
                      Arguments.llps.llps_density);
      TitanUI::log_kv(log, "LLPS Kernel Policy",
                      Arguments.llps.llps_kernel);
      TitanUI::log_kv(log, "LLPS Dynamics",
                      Arguments.llps.llps_dynamics ? "enabled" : "disabled");
      TitanUI::log_kv(log, "LLPS Detailed Dynamics",
                      Arguments.llps.llps_dynamics_detailed ? "enabled"
                                                            : "disabled");
      TitanUI::log_kv(log, "LLPS Stickers",
                      Arguments.llps.llps_sticker_beads);
      // Start dedicated LLPS worker thread (pipeline mode)
      llps_analyzer->start_async();
    }

    // --- LLPS DBSCAN (bead-level morphology) ---
    if (Arguments.llps.flag_llps && Arguments.llps.llps_dbscan && !dbscan_llps) {
      dbscan_llps = std::make_unique<TitanLLPSCluster>();
      const std::string pdb_out =
          info_output.get_llps_dbscan_droplets_filename();
      const std::string csv_out =
          info_output.get_llps_dbscan_timeseries_filename();
      // [BUG 1 fix] Use user-supplied eps (Å) and min_pts; default 15 Å / 5
      // beads. Pass csv_out so data is flushed per-frame (crash-safe, ISSUE 4
      // fix).
      TitanUI::log_kv(log, "LLPS-DBSCAN eps",
                      std::to_string(Arguments.llps.llps_dbscan_eps) + " A");
      TitanUI::log_kv(log, "LLPS-DBSCAN min_pts",
                      std::to_string(Arguments.llps.llps_dbscan_minpts));
      dbscan_llps->init(Arguments.llps.llps_dbscan_eps,
                        Arguments.llps.llps_dbscan_minpts, pdb_out, csv_out);
      dbscan_llps->configure_dynamics(
          Arguments.llps.llps_dynamics, Arguments.llps.llps_dynamics_detailed,
          Arguments.llps.llps_maxlag, Arguments.llps.llps_lineage_overlap,
          std::max(1, Arguments.llps.llps_dimer_size),
          Arguments.llps.llps_msd_fit_start_ns,
          Arguments.llps.llps_msd_fit_end_ns);
      std::string llps_mol2_type = Arguments.mol_2_type;
      std::transform(llps_mol2_type.begin(), llps_mol2_type.end(),
                     llps_mol2_type.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                     });
      const bool llps_has_rna =
          llps_mol2_type.find("rna") != std::string::npos && !lig.info.empty();
      const float prna_rcut =
          (Arguments.llps.llps_prna_rcut > 0.0f)
              ? Arguments.llps.llps_prna_rcut
              : Arguments.llps.llps_rcut;
      const float intra_rcut =
          (Arguments.llps.llps_rna_intra_rcut > 0.0f)
              ? Arguments.llps.llps_rna_intra_rcut
              : Arguments.llps.llps_rcut;
      TitanUI::log_kv(log, "LLPS-DBSCAN Output PDB", pdb_out);
      TitanUI::log_kv(log, "LLPS-DBSCAN Output CSV", csv_out);
      TitanUI::log_kv(log, "LLPS-Fuzzy",
                      Arguments.llps.llps_fuzzy ? "enabled" : "disabled");
      TitanUI::log_kv(log, "LLPS RNA Detected",
                      llps_has_rna ? "yes" : "no");
      TitanUI::log_kv(log, "LLPS Protein-RNA Rcut (nm)",
                      std::to_string(prna_rcut));
      TitanUI::log_kv(log, "LLPS RNA Intra Rcut (nm)",
                      std::to_string(intra_rcut));
      TitanUI::log_kv(log, "LLPS RNA Compaction Mode",
                      Arguments.llps.llps_rna_compaction);
      TitanUI::log_kv(log, "LLPS RNA Information Mode",
                      Arguments.llps.llps_rna_information);
      TitanUI::log_kv(log, "LLPS RNA Topology Mode",
                      Arguments.llps.llps_rna_topology);
      TitanUI::log_kv(log, "LLPS RNA Topology Detailed",
                      Arguments.llps.llps_rna_topology_detailed ? "enabled"
                                                                : "disabled");
      TitanUI::log_kv(log, "LLPS Energy Correlation",
                      Arguments.llps.llps_energy_correlation);
      TitanUI::log_kv(log, "LLPS Paper Bundle",
                      Arguments.llps.llps_paper ? "enabled" : "disabled");
      const bool llps_fractal_enabled =
          Arguments.llps.llps_paper_fractal == "auto"
              ? Arguments.llps.llps_paper
              : (Arguments.llps.llps_paper_fractal == "true" ||
                 Arguments.llps.llps_paper_fractal == "yes" ||
                 Arguments.llps.llps_paper_fractal == "on" ||
                 Arguments.llps.llps_paper_fractal == "1");
      TitanUI::log_kv(log, "LLPS Paper Fractal",
                      llps_fractal_enabled ? "enabled" : "disabled");
      if (Arguments.llps.llps_paper) {
        TitanUI::log_kv(log, "LLPS Paper Profile Bin (nm)",
                        std::to_string(
                            Arguments.llps.llps_paper_profile_bin_nm));
        TitanUI::log_kv(log, "LLPS Paper Blocks",
                        Arguments.llps.llps_paper_blocks > 0
                            ? std::to_string(Arguments.llps.llps_paper_blocks)
                            : "auto");
        TitanUI::log_kv(log, "LLPS Paper Geometry",
                        Arguments.llps.llps_paper_geometry);
      }
      dbscan_llps->configure_rna_compaction(
          Arguments.llps.llps_rna_compaction, Arguments.llps.llps_dynamics,
          Arguments.llps.llps_dynamics_detailed, llps_has_rna,
          Arguments.llps.llps_sticker_beads, prna_rcut, intra_rcut,
          Arguments.llps.llps_rna_longrange_min);
      dbscan_llps->configure_rna_information(
          Arguments.llps.llps_rna_information, Arguments.llps.llps_dynamics,
          Arguments.llps.llps_dynamics_detailed, llps_has_rna,
          Arguments.llps.llps_rna_info_rg_bins,
          Arguments.llps.llps_rna_info_longrange_bins);
      dbscan_llps->configure_rna_topology(
          Arguments.llps.llps_rna_topology, Arguments.llps.llps_dynamics,
          Arguments.llps.llps_rna_topology_detailed, llps_has_rna,
          Arguments.llps.llps_rcut, prna_rcut,
          Arguments.llps.llps_rna_longrange_min,
          Arguments.llps.llps_rna_bridge_geom_min,
          Arguments.llps.llps_rna_patch_gap_max,
          Arguments.llps.llps_rna_clamp_require_loop, info_output);
      llps_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                     static_cast<double>(prna_rcut),
                                     prna_rcut > 0.0f);
      dbscan_llps->configure_energy_correlation(
          Arguments.llps.llps_energy_correlation, Arguments.llps.llps_dynamics,
          llps_has_rna, Arguments.flag_nocal, &llps_energy_provider,
          Arguments.llps.llps_energy_detailed, info_output);
      dbscan_llps->set_kernel_mode(Arguments.llps.llps_kernel);
      dbscan_llps->configure_paper(
          {Arguments.llps.llps_paper,
           Arguments.llps.llps_paper_profile_bin_nm,
           Arguments.llps.llps_paper_blocks,
           Arguments.llps.llps_paper_geometry,
           llps_fractal_enabled, Arguments.llps.llps_phase_axis});
      dbscan_llps->configure_trace_outputs(info_output);
      if (Arguments.llps.llps_fuzzy) {
        dbscan_llps->enable_fuzzy(true);
        TitanUI::print_info("[LLPS-Fuzzy] Fuzzy EM refinement enabled.");
      }
    }

    if (Arguments.pbe.pbsa) {
      const double pbsa_cutoff_nm = titan_resolve_sg_cutoff_nm(Arguments);
      pbe_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                    pbsa_cutoff_nm, pbsa_cutoff_nm > 0.0);
      TitanUI::log_kv(log, "MM/PBSA", "enabled");
      TitanUI::log_kv(log, "MM/PBSA Energy Source",
                      "PBE deltaG_solv + force-field elec/vdw + SASA IE");
      TitanUI::log_kv(
          log, "MM/PBSA Pair Cutoff",
          pbsa_cutoff_nm > 0.0 ? std::to_string(pbsa_cutoff_nm) + " nm"
                               : "none");
    }

    if (edecomp_mm_requested(Arguments)) {
      const double edecomp_cutoff_nm = titan_resolve_sg_cutoff_nm(Arguments);
      edecomp_energy_provider.configure(pro_in, lig, CachedData, Arguments,
                                        edecomp_cutoff_nm,
                                        edecomp_cutoff_nm > 0.0);
      if (!edecomp_engine) {
        edecomp_engine =
            std::make_unique<titan_energy_decomp::EnergyDecompositionEngine>(
                titan_energy_decomp::make_axes(pro_in, "mol1"),
                titan_energy_decomp::make_axes(lig, "mol2"),
                make_edecomp_config(Arguments));
      }
      TitanUI::log_kv(log, "Energy Decomposition",
                      "MM elec+vdW enabled");
      TitanUI::log_kv(
          log, "Energy Decomposition Pair Cutoff",
          edecomp_cutoff_nm > 0.0
              ? std::to_string(edecomp_cutoff_nm) + " nm fallback cutoff"
              : "none");
    }

    if (edecomp_gbsa_requested(Arguments) && !edecomp_gbsa_engine) {
      edecomp_gbsa_engine =
          std::make_unique<titan_energy_decomp::EndpointEntityDecompositionEngine>(
              titan_energy_decomp::make_axes(pro_in, "mol1"),
              titan_energy_decomp::make_axes(lig, "mol2"),
              make_edecomp_config(Arguments));
      TitanUI::log_kv(log, "Energy Decomposition",
                      "MM/GBSA endpoint maps enabled");
    }

    if (edecomp_pbsa_requested(Arguments) && !edecomp_pbsa_engine) {
      edecomp_pbsa_engine =
          std::make_unique<titan_energy_decomp::EndpointEntityDecompositionEngine>(
              titan_energy_decomp::make_axes(pro_in, "mol1"),
              titan_energy_decomp::make_axes(lig, "mol2"),
              make_edecomp_config(Arguments));
      pbe_analyzer.set_edecomp_sink(
          [&](const std::vector<TitanPBE::EDecompRow> &rows) {
            if (!edecomp_pbsa_engine)
              return;
            edecomp_pbsa_engine->process_frame(
                make_pbsa_endpoint_rows(rows, *edecomp_pbsa_engine));
          });
      TitanUI::log_kv(log, "Energy Decomposition",
                      "MM/PBSA endpoint maps enabled");
    }

    // --- GIST Water Thermodynamics Initialization ---
    if (Arguments.gist.flag_gist) {
      // The group index must be resolved before topology/coordinate loading.
      if (Arguments.n_wat < 0 || water_traj.info.empty() || water_traj.info.size()%3) {
        TitanUI::print_error("GIST has no complete water topology; select a full O,H,H water group with -n_wat.");
        return 1;
      }
      for (size_t i=0;i<water_traj.info.size();++i) {
        const auto name=water_traj.info[i].get_atom();
        const bool oxygen=(name=="O" || name=="OW" || name=="OH2");
        const bool hydrogen=!name.empty() && name[0]=='H';
        if ((i%3==0 && !oxygen) || (i%3!=0 && !hydrogen)) {
          TitanUI::print_error("GIST water group must be ordered oxygen, hydrogen, hydrogen for each molecule.");
          return 1;
        }
      }

      if (!gist_analyzer && !lig.m_mol.empty() &&
          lig.m_mol[0].get_pos_x().size() > 0) {
        gist_analyzer = std::make_unique<TitanGIST>();
        gist_analyzer->init(
            lig.m_mol[0], Arguments.gist.gist_padding,
            Arguments.gist.gist_resolution, Arguments.gist.gist_temperature,
            Arguments.gist.gist_cutoff, Arguments.nt > 0 ? Arguments.nt : 1,
            Arguments.gist.gist_water_model, pro_in.parameters, pro_in.info,
            lig.parameters, lig.info);
        // BUG2: pass DCCM's exact Cα indices for TICA centroid
        if (!dccm_analyzer.get_ca_indices().empty())
          gist_analyzer->set_tica_ca_indices(dccm_analyzer.get_ca_indices());
        TitanUI::log_kv(log, "GIST Water Model",
                        Arguments.gist.gist_water_model);
        TitanUI::log_kv(log, "GIST Resolution",
                        std::to_string(Arguments.gist.gist_resolution) + " A");
        TitanUI::log_kv(log, "GIST Voxels",
                        std::to_string(gist_analyzer->n_voxels()));
        TitanUI::log_kv(log, "GIST Cutoff",
                        std::to_string(Arguments.gist.gist_cutoff) + " A");
      }
    }

    // --- MSM INITIALIZATION (after TICA init, shares n_components) ---
    if (need_tica_projection_cache && tica_analyzer && !msm_analyzer) {
      const double selected_dt_ps =
          Arguments.dt > 0 ? static_cast<double>(Arguments.dt)
                           : info_output.get_current_dt();
      const double msm_dt_ns =
          selected_dt_ps > 0.0 ? selected_dt_ps / 1000.0 : 0.001;
      msm_analyzer = std::make_unique<TitanMSM>(
          Arguments.msm.msm_clusters, tica_analyzer->n_components(), msm_dt_ns);
      if (Arguments.msm.flag_msm) {
        const std::string msm_cluster_cfg =
            Arguments.msm.msm_clusters_user_set
                ? std::to_string(Arguments.msm.msm_clusters)
                : "auto (default " +
                      std::to_string(Arguments.msm.msm_clusters) + ")";
        TitanUI::log_kv(log, "MSM Clusters", msm_cluster_cfg);
        TitanUI::log_kv(log, "MSM Lag",
                        std::to_string(Arguments.msm.msm_lag) + " frames");
        TitanUI::log_kv(log, "MSM PCCA",
                        Arguments.msm.msm_run_pcca ? "enabled" : "disabled");
      } else {
        TitanUI::log_kv(log, "TICA Projection Cache",
                        "enabled for downstream modules");
      }
      TitanUI::log_kv(log, "TICA Projection Strategy",
                      "rebuild projected frames from TitanKinetics feature "
                      "cache after solve_tica()");
    }

    // Fix 5: hoist TICA workspace OUTSIDE streaming loops
    // (pairwise distance vector + Kabsch mobile/reference matrices — zero heap
    // per frame)
    // Fix 1: these are aliases to global variables hoisted outside while(true)
    // so all replicas share a single Kabsch reference frame.
    if (Arguments.msm.flag_tica && tica_analyzer) {
      tica_feat_dim_global = tica_analyzer->feature_dim();
      if (tica_feat_dim_global > 0 &&
          tica_features_global.size() != tica_feat_dim_global) {
        tica_features_global.resize(tica_feat_dim_global);
      }
      if (!tica_use_lj_features_global && tica_n_ca_global == 0) {
        tica_n_ca_global = static_cast<int>(dccm_analyzer.get_resnrs().size());
        if (tica_n_ca_global > 0) {
          tica_mobile_global.resize(tica_n_ca_global, 3);
          tica_ref_centered_global.resize(tica_n_ca_global, 3);
        }
      }
    }
    const int& tica_n_ca       = tica_n_ca_global;
    auto& tica_features        = tica_features_global;
    auto& tica_mobile          = tica_mobile_global;
    auto& tica_ref_centered    = tica_ref_centered_global;
    auto& tica_R               = tica_R_global;
    bool& tica_ref_initialized = tica_ref_initialized_global;
    const bool use_lj_tica_features = tica_use_lj_features_global;
    const bool use_observable_tica_features =
        tica_use_observable_features_global;
    const bool tica_or_export_needs_dccm =
        (Arguments.msm.flag_tica && !use_lj_tica_features &&
         feature_source_needs_dccm(Arguments,
                                   !use_observable_tica_features)) ||
        (feature_router.active() && !feature_router.consume_tica() &&
         feature_source_needs_dccm(Arguments, false));

    std::string resolved_modules;
    std::string skipped_modules;
    if (pbe_analyzer.enabled())
      append_module_label(resolved_modules, "PBE");
    if (Arguments.pbe.pbsa)
      append_module_label(resolved_modules, "MM/PBSA");
    if (Arguments.flag_cas)
      append_module_label(resolved_modules, "CAS");
    const bool dssp_runtime_enabled =
        dssp_mol1_analyzer.enabled() || dssp_mol2_analyzer.enabled();
  if (dssp_runtime_enabled) {
    append_module_label(resolved_modules, "DSSP");
  } else if (dssp_mol1_requested || dssp_mol2_requested) {
    append_module_label(skipped_modules,
                        "DSSP(auto-skipped: no compatible protein backbone)");
  }
  if (lj_analyzer.enabled())
    append_module_label(resolved_modules, "LJ");
  if (cmap_analyzer.enabled())
    append_module_label(resolved_modules, "CMap");
  if (interface_analyzer.enabled())
    append_module_label(resolved_modules, "Interface");
  if (dihedral_analyzer.enabled())
    append_module_label(resolved_modules, "Dihedral");
  if (qnative_analyzer.enabled())
    append_module_label(resolved_modules, "QNative");
  if (msd_analyzer.enabled())
    append_module_label(resolved_modules, "MSD");
  if (membrane_analyzer.enabled())
    append_module_label(resolved_modules, "Membrane");
  if (s2_analyzer.enabled())
    append_module_label(resolved_modules, "S2");
  if (rg_analyzer.enabled())
    append_module_label(resolved_modules, "Rg2");
  if (rms_analyzer.enabled())
    append_module_label(resolved_modules, "RMSD/RMSF");
  if (Arguments.flag_network || Arguments.community.flag_community)
    append_module_label(resolved_modules, "DCCM/Network");
    if (Arguments.corr_knn.flag_corr_knn)
      append_module_label(resolved_modules, "Corr-kNN");
  if (Arguments.community.flag_community)
    append_module_label(resolved_modules, "Community");
  if (Arguments.flag_pca)
    append_module_label(resolved_modules, "PCA_FEL");
    if (Arguments.flag_tda)
      append_module_label(resolved_modules, "TDA");
    if (Arguments.msm.flag_tica && tica_analyzer)
      append_module_label(resolved_modules, "TICA");
    if (Arguments.msm.flag_msm && msm_analyzer)
      append_module_label(resolved_modules, "MSM");
    if (Arguments.umap.flag_umap)
      append_module_label(resolved_modules, "UMAP");
    if (Arguments.dham.flag_dham && msm_analyzer)
      append_module_label(resolved_modules, "DHAM");
    if (Arguments.tpt.flag_tpt && msm_analyzer)
      append_module_label(resolved_modules, "TPT");
    if (Arguments.flag_h_bond)
      append_module_label(resolved_modules, "H-Bonds");
    if (Arguments.n_wat >= 0)
      append_module_label(resolved_modules, "Water");
    if (Arguments.flag_allostery)
      append_module_label(resolved_modules, "Allostery");
    if (Arguments.lspmd.flag_lspmd)
      append_module_label(resolved_modules, "LSP-MD");

    if (!resolved_modules.empty() || !skipped_modules.empty()) {
      TitanUI::log_section(log, "RESOLVED MODULES");
      TitanUI::log_kv(log, "Active Features",
                      resolved_modules.empty() ? "None" : resolved_modules);
    }

    std::string streaming_modules;
    if (pbe_analyzer.enabled())
      append_module_label(streaming_modules, "PBE");
    if (Arguments.pbe.pbsa)
      append_module_label(streaming_modules, "MM/PBSA");
    if (Arguments.flag_gbsa)
      append_module_label(streaming_modules, "GBSA");
    if (Arguments.flag_cas)
      append_module_label(streaming_modules, "CAS");
    if (global_has_nucleic_acid)
      append_module_label(streaming_modules, "RNA-Profile");
    if (dssp_runtime_enabled)
      append_module_label(streaming_modules, "DSSP");
    if (lj_analyzer.enabled())
      append_module_label(streaming_modules, "LJ");
    if (sq_analyzer.enabled())
      append_module_label(streaming_modules, "SQ");
    if (cmap_analyzer.enabled())
      append_module_label(streaming_modules, "CMap");
    if (dihedral_analyzer.enabled())
      append_module_label(streaming_modules, "Dihedral");
    if (qnative_analyzer.enabled())
      append_module_label(streaming_modules, "QNative");
    if (msd_analyzer.enabled())
      append_module_label(streaming_modules, "MSD");
    if (membrane_analyzer.enabled())
      append_module_label(streaming_modules, "Membrane");
    if (s2_analyzer.enabled())
      append_module_label(streaming_modules, "S2");
    if (rg_analyzer.enabled())
      append_module_label(streaming_modules, "Rg2");
    if (rms_analyzer.enabled())
      append_module_label(streaming_modules, "RMSD/RMSF");
    if (Arguments.flag_network || Arguments.community.flag_community ||
        Arguments.corr_knn.flag_corr_knn ||
        Arguments.flag_pca || Arguments.lspmd.flag_lspmd) {
      append_module_label(streaming_modules, "DCCM/PCA");
    }
    if (Arguments.msm.flag_tica && tica_analyzer)
      append_module_label(streaming_modules, "TICA");
    if (Arguments.ifp.flag_ifp && ifp_analyzer)
      append_module_label(streaming_modules, "IFP");
    if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer)
      append_module_label(streaming_modules, "AWH-IFP");
    if (Arguments.llps.flag_llps)
      append_module_label(streaming_modules, "LLPS");
    if (edecomp_mm_requested(Arguments))
      append_module_label(streaming_modules, "EDecomp");
    if (edecomp_gbsa_requested(Arguments))
      append_module_label(streaming_modules, "EDecomp-GBSA");
    if (edecomp_pbsa_requested(Arguments))
      append_module_label(streaming_modules, "EDecomp-PBSA");
    if (streaming_modules.empty())
      streaming_modules = "I/O";
    if (!resolved_modules.empty() || !skipped_modules.empty()) {
      TitanUI::log_kv(log, "Streaming Features", streaming_modules);
      if (!skipped_modules.empty())
        TitanUI::log_kv(log, "Skipped Features", skipped_modules);
    }
    TitanUI::print_runtime_module_summary(
        active_modules, resolved_modules.empty() ? "None" : resolved_modules,
        skipped_modules, streaming_modules);

    if (pbe_analyzer.enabled()) {
      pbe_analyzer.begin_trajectory(
          static_cast<std::size_t>(input_name - Arguments.b_input + 1),
          progress_source_label(Arguments, name_pro));
    }
    if (edecomp_mm_requested(Arguments) && edecomp_engine) {
      edecomp_engine->begin_trajectory(
          progress_source_label(Arguments, name_pro));
    }
    if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
      edecomp_gbsa_engine->begin_trajectory(
          progress_source_label(Arguments, name_pro));
    }
    if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine) {
      edecomp_pbsa_engine->begin_trajectory(
          progress_source_label(Arguments, name_pro));
    }

    size_t global_frame_counter = 0;
    bool is_streaming = true;
    size_t NUM = 0;

    // ========================================================================
    // Save Frame 0 topology+coords for post-loop functions that need m_mol
    // (e.g. print_energy_pdbtype which writes B-factor PDB).
    // The streaming loop clears pro_in/lig.m_mol each iteration.
    // ========================================================================
    saved_pro_frame0 = pro_in.m_mol;
    saved_lig_frame0 = lig.m_mol;
    saved_membrane_frame0 = membrane_traj.m_mol;
    hb_rec_1traj.clear_frame();
    hb_lg_1traj.clear_frame();

    const bool need_com_distance =
        (Arguments.com_dis_energy_2mol != 4) || (Arguments.com_dis_2mol != 0);
    const bool need_distance_scan =
        Arguments.ifp.flag_ifp || Arguments.flag_cas || Arguments.flag_ml ||
        Arguments.cal_dis_sc_contact ||
        (Arguments.flag_tda && Arguments.flag_sidechain) ||
        (Arguments.flag_cal_force &&
         (Arguments.force_map == 1 || Arguments.force_map == 3)) ||
        (Arguments.flag_sub_groups && Arguments.flag_cal_mindis) ||
        need_com_distance;

    // ========================================================================
    // CHUNK PREFETCH (XDR only)
    // ========================================================================
    // XTC/TRR decode (decompress + float->double scatter) used to run on the
    // main thread with the worker pool idle, then analysis ran with the reader
    // idle. The reader now fills back buffers for chunk N+1 while chunk N is
    // analysed, and only the frame vectors are swapped, so the shared topology
    // in pro_in/lig/water_traj/membrane_traj is never touched or duplicated.
    //
    // Safety: read_next_chunk() is the sole streamer entry point inside the
    // loop, start_chunk_prefetch() only runs after the previous read has been
    // consumed by take_next_chunk(), and take_next_chunk() joins before any
    // further read, so exactly one thread uses the streamer at a time. The
    // worker is declared after the buffers it writes, so its destructor joins
    // before those buffers die on any early return.
    //
    // The PDB path keeps the synchronous read because its loader also
    // populates topology, not just frames.
    Multi_MOL pro_next(force_f), lig_next(force_f), water_next(force_f),
        membrane_next(force_f);
    std::optional<std::expected<size_t, std::string>> prefetched_chunk;
    // TITAN_DISABLE_CHUNK_PREFETCH forces the previous serial reader. It exists
    // so the overlapped and serial paths can be diffed on the same binary, and
    // as an operator kill switch if a host ever misbehaves under the extra
    // reader thread. Both paths must produce identical output.
    const bool chunk_prefetch_enabled =
        Arguments.flag_xdr && CHUNK_SIZE != std::numeric_limits<size_t>::max() &&
        std::getenv("TITAN_DISABLE_CHUNK_PREFETCH") == nullptr;
    std::optional<titan::jthread> chunk_prefetch_worker;

    auto start_chunk_prefetch = [&]() {
      if (!chunk_prefetch_enabled || chunk_prefetch_worker)
        return;
      chunk_prefetch_worker.emplace([&]() {
        prefetched_chunk = streamer.read_next_chunk(
            CHUNK_SIZE, pro_next, lig_next, water_next, membrane_next);
      });
    };

    auto take_next_chunk = [&]() {
      if (!chunk_prefetch_worker)
        return streamer.read_next_chunk(CHUNK_SIZE, pro_in, lig, water_traj,
                                        membrane_traj);
      chunk_prefetch_worker->join();
      chunk_prefetch_worker.reset();
      auto result = std::move(*prefetched_chunk);
      prefetched_chunk.reset();
      // read_next_chunk() clears its destination frame vectors on entry, so the
      // back buffers hold exactly what a synchronous read would have produced.
      std::swap(pro_in.m_mol, pro_next.m_mol);
      std::swap(lig.m_mol, lig_next.m_mol);
      std::swap(water_traj.m_mol, water_next.m_mol);
      std::swap(membrane_traj.m_mol, membrane_next.m_mol);
      return result;
    };

    // ========================================================================
    // MAIN STREAMING LOOP
    // ========================================================================
    while (is_streaming) {
      const auto start_input_stream = std::chrono::steady_clock::now();
      // With prefetch active this measures the reader stall, not the decode.
      auto expected_frames = take_next_chunk();
      time_input_stream += std::chrono::duration_cast<milliseconds>(
          std::chrono::steady_clock::now() - start_input_stream);
      if (!expected_frames) {
        TitanUI::print_error(expected_frames.error());
        return 1;
      }

      size_t chunk_num_snaps = expected_frames.value();
      if (chunk_num_snaps == 0) {
        is_streaming = false;
        break;
      }

      NUM = chunk_num_snaps;

      // Decode chunk N+1 while chunk N is analysed below.
      start_chunk_prefetch();

      // --- PBC & WATER SHIFT LOGIC (Applied Per-Chunk) ---
      struct shift_vector {
        double x, y, z;
      };
      std::vector<shift_vector> old_cogs(pro_in.m_mol.size(), {0.0, 0.0, 0.0});
      if (Arguments.n_wat >= 0) {
        for (size_t snap = 0; snap < pro_in.m_mol.size(); ++snap) {
          double old_cx = 0.0, old_cy = 0.0, old_cz = 0.0;
          size_t p_size = pro_in.m_mol[snap].pos.x.size();
          if (p_size > 0) {
            for (size_t i = 0; i < p_size; ++i) {
              old_cx += pro_in.m_mol[snap].pos.x[i];
              old_cy += pro_in.m_mol[snap].pos.y[i];
              old_cz += pro_in.m_mol[snap].pos.z[i];
            }
            old_cx /= p_size;
            old_cy /= p_size;
            old_cz /= p_size;
          }
          old_cogs[snap] = {old_cx, old_cy, old_cz};
        }
      }

      auto start_pbc = std::chrono::steady_clock::now();
      const bool have_mol2_frames =
          !pro_in.m_mol.empty() && lig.m_mol.size() >= pro_in.m_mol.size() &&
          std::all_of(pro_in.m_mol.begin(), pro_in.m_mol.end(),
                      [&](const MOL &frame) {
                        const size_t index =
                            static_cast<size_t>(&frame - pro_in.m_mol.data());
                        return lig.m_mol[index].size() != 0;
                      });
      const auto align_loaded_complex = [&]() {
        if (have_mol2_frames)
          align_complex(pro_in, lig, 0, 0);
      };
      if (Arguments.pbc.solv_pbctopo) {
        pro_in.solve_pbc_pbctopo(
            lig, Arguments, &executor, tot_snap_allfiles + global_frame_counter);
      } else if (Arguments.pbc.solv_pbcn) {
        pro_in.solv_pbc_pbcn(lig, Arguments);
        align_loaded_complex();
      } else if (Arguments.pbc.solv_pbc_int == 1) {
        if (Arguments.pbc.chain_ref == -2) {
          executor.parallel_do(num_threads, [&](size_t worker_id) {
            pro_in.solve_pbc_int(worker_id, num_threads);
          });
        }
        if (Arguments.pbc.chain_ref > -2) {
          pro_in.solve_pbc_new(lig, Arguments, amine);
          align_loaded_complex();
        }
      }
      if (Arguments.pbc.solv_pbc == 1) {
        executor.parallel_do(num_threads, [&](size_t worker_id) {
          pro_in.pbc_prob(lig, worker_id, num_threads);
        });
        align_loaded_complex();
      }
      time_solve_pbc += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_pbc);

      // ── NOCAL MODE: PBC only → write PDB/PQR → skip energy ──────────
      if (Arguments.flag_nocal) {
        // Keep a loaded two-molecule system together. Single-molecule nocal
        // workflows still center mol1 independently.
        if (Arguments.pbc.solv_pbcn || Arguments.pbc.solv_pbctopo ||
            Arguments.pbc.solv_pbc_int)
          if (!have_mol2_frames)
            pro_in.align_to_box();

        // PBC clustering (non-int mode)
        if (Arguments.pbc.solv_pbc_clust && !Arguments.pbc.solv_pbc_int)
          pro_in.solve_pbc_cluster(Arguments, info_output);

        TitanUI::log_kv(
            log, "Nocal Chunk",
            "Writing frames " + std::to_string(global_frame_counter + 1) +
                " to " +
                std::to_string(global_frame_counter + chunk_num_snaps));

        for (size_t chunk_snap_idx = 0; chunk_snap_idx < chunk_num_snaps;
             ++chunk_snap_idx) {
          const size_t global_snap_idx = global_frame_counter + chunk_snap_idx;
          const int progress_traj_id = input_name - Arguments.b_input + 1;
          std::string status_text =
              "Traj " + std::to_string(progress_traj_id) + " | Frame " +
              std::to_string(global_snap_idx + 1) + " | Global " +
              std::to_string((tot_snap_allfiles + global_snap_idx + 1));

          const std::string module_name =
              progress_source_label(Arguments, name_pro);

          if (Arguments.flag_xdr) {
            TitanUI::show_continuous_progress(module_name, global_snap_idx + 1,
                                              status_text, streaming_modules);
          } else {
            TitanUI::show_progress_exact(module_name, chunk_snap_idx + 1,
                                         std::max<size_t>(1, chunk_num_snaps),
                                         status_text, streaming_modules);
          }

          if (pbe_analyzer.enabled()) {
            auto start_pbe = std::chrono::steady_clock::now();
            const MOL *pbe_lig_frame =
                (chunk_snap_idx < lig.m_mol.size()) ? &lig.m_mol[chunk_snap_idx]
                                                    : nullptr;
            const Multi_MOL_parameters *pbe_lig_params =
                pbe_lig_frame ? &lig.parameters : nullptr;
            pbe_analyzer.consume_frame(
                pro_in.m_mol[chunk_snap_idx], pro_in.parameters, pbe_lig_frame,
                pbe_lig_params,
                static_cast<std::size_t>(progress_traj_id), global_snap_idx,
                tot_snap_allfiles + global_snap_idx, executor, log,
                Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
            time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_pbe);
          }

          // COM distance between molecules
          if (Arguments.com_dis_2mol != 0) {
            fstream output_dis_2mol_file(
                paths.get_path(input_name - Arguments.b_input) + "/" +
                    info_output.getDisCom2Mol(),
                ios::out | ios::app);
            pro_in.impl_com_dis_2mol(lig, Arguments, info_output,
                                     output_dis_2mol_file, chunk_snap_idx);
          }

          // Dipole moment
          if (Arguments.flag_cal_dipole_moment)
            pro_in.cal_dipole_moment(
                info_output, Arguments,
                Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4));

          // Charged atom distance distribution
          if (Arguments.flag_charged_atom_distance_dis)
            pro_in.cal_distribution_distance_charged_atoms(Arguments,
                                                           info_output);

          // Volume calculation
          if (Arguments.flag_cal_vol) {
            vector<double> volumes_temp;
            pro_in.cal_volumes(volumes_temp);
            Volumes.emplace_back(std::move(volumes_temp));
          }

          if (dssp_mol1_analyzer.enabled() ||
              dssp_mol2_analyzer.enabled()) {
            auto start_dssp = std::chrono::steady_clock::now();
            if (dssp_mol1_analyzer.enabled())
              dssp_mol1_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            if (dssp_mol2_analyzer.enabled())
              dssp_mol2_analyzer.process_frame(lig, chunk_snap_idx, executor);
            time_dssp += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_dssp);
          }

          if (lj_analyzer.enabled()) {
            auto start_lj = std::chrono::steady_clock::now();
            lj_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            time_lj += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_lj);
          }

          if (sq_analyzer.enabled()) {
            auto start_sq = std::chrono::steady_clock::now();
            sq_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            time_sq += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_sq);
          }

          if (cmap_analyzer.enabled()) {
            auto start_cmap = std::chrono::steady_clock::now();
            const Multi_MOL *cmap_mol2 =
                Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
            cmap_analyzer.process_frame(pro_in, cmap_mol2, chunk_snap_idx,
                                        executor);
            const auto end_cmap = std::chrono::steady_clock::now();
            time_cmap += std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_cmap - start_cmap);
            perf_observer.add_frame_elapsed(
                "CMap", static_cast<std::uint64_t>(
                            tot_snap_allfiles + global_frame_counter +
                            chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_cmap -
                                                          start_cmap)
                    .count());
          }

          if (Arguments.ifp.flag_ifp && ifp_analyzer &&
              include_ifp_frame(
                  pro_in, chunk_snap_idx,
                  static_cast<size_t>(progress_traj_id), global_snap_idx + 1,
                  tot_snap_allfiles + global_snap_idx + 1)) {
            auto start_sc = std::chrono::steady_clock::now();
            std::vector<d_min> ifp_dis_min(num_threads);

            if (pro_in.Index.size() > 1) {
              executor.parallel_for(
                  pro_in.Index.size() - 1, num_threads, 1,
                  [&](size_t /*worker_id*/, size_t begin, size_t end) {
                    for (size_t residue_idx = begin; residue_idx < end;
                         ++residue_idx) {
                      pro_in.m_mol[chunk_snap_idx].take_com(
                          pro_in.parameters, residue_idx,
                          pro_in.Index[residue_idx + 1].get_res_pos_e(0),
                          pro_in.Index, pro_in.info);
                    }
                  });
            }

            if (lig.Index.size() > 1) {
              executor.parallel_for(
                  lig.Index.size() - 1, num_threads, 1,
                  [&](size_t /*worker_id*/, size_t begin, size_t end) {
                    for (size_t residue_idx = begin; residue_idx < end;
                         ++residue_idx) {
                      lig.m_mol[chunk_snap_idx].take_com(
                          lig.parameters, residue_idx,
                          lig.Index[residue_idx + 1].get_res_pos_e(0),
                          lig.Index, lig.info);
                    }
                  });
            }

            impl_add_coms_2_mols(lig, pro_in, Arguments, chunk_snap_idx);
            executor.parallel_do(num_threads, [&](size_t worker_id) {
              take_min(
                  pro_in.m_mol[chunk_snap_idx].Com,
                  lig.m_mol[chunk_snap_idx].Com, ifp_dis_min[worker_id],
                  worker_id * pro_in.m_mol[chunk_snap_idx].Com.size() /
                      num_threads,
                  (worker_id + 1) *
                      pro_in.m_mol[chunk_snap_idx].Com.size() / num_threads,
                  0, lig.m_mol[chunk_snap_idx].Com.size(), Arguments, 0,
                  d_standard);
            });
            time_sc += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_sc);

            auto start_ifp = std::chrono::steady_clock::now();
            const size_t lg_snap_ifp =
                (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
            ifp_analyzer->process_frame(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[lg_snap_ifp],
                pro_in.info, lig.info, pro_in.Index, lig.Index,
                pro_in.info.size(), lig.info.size(), ifp_dis_min,
                Arguments.ifp, executor, num_threads, false);
            time_ifp += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_ifp);
          }

          if (interface_analyzer.enabled()) {
            const auto start_interface = std::chrono::steady_clock::now();
            const Multi_MOL *interface_mol2 =
                Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
            try {
              interface_analyzer.process_frame(
                  pro_in, interface_mol2, chunk_snap_idx, global_snap_idx,
                  executor);
            } catch (const std::exception &e) {
              TitanUI::print_error(std::string("Interface analysis: ") +
                                   e.what());
              return 1;
            }
            const auto end_interface = std::chrono::steady_clock::now();
            time_interface +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_interface - start_interface);
            perf_observer.add_frame_elapsed(
                "Interface",
                static_cast<std::uint64_t>(tot_snap_allfiles +
                                           global_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_interface -
                                                          start_interface)
                    .count());
          }

          if (dihedral_analyzer.enabled()) {
            auto start_dihedral = std::chrono::steady_clock::now();
            dihedral_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            const auto end_dihedral = std::chrono::steady_clock::now();
            time_dihedral +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_dihedral - start_dihedral);
            perf_observer.add_frame_elapsed(
                "Dihedral",
                static_cast<std::uint64_t>(tot_snap_allfiles +
                                           global_frame_counter +
                                           chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_dihedral -
                                                          start_dihedral)
                    .count());
          }

          if (qnative_analyzer.enabled()) {
            auto start_qnative = std::chrono::steady_clock::now();
            const Multi_MOL *qnative_mol2 =
                Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
            qnative_analyzer.process_frame(pro_in, qnative_mol2, chunk_snap_idx,
                                           executor);
            const auto end_qnative = std::chrono::steady_clock::now();
            time_qnative +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_qnative - start_qnative);
            perf_observer.add_frame_elapsed(
                "QNative",
                static_cast<std::uint64_t>(tot_snap_allfiles +
                                           global_frame_counter +
                                           chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_qnative -
                                                          start_qnative)
                    .count());
          }

          if (msd_analyzer.enabled()) {
            auto start_msd = std::chrono::steady_clock::now();
            const Multi_MOL *msd_mol2 =
                Arguments.needs_mol2_for_msd() ? &lig : nullptr;
            msd_analyzer.process_frame(pro_in, msd_mol2, chunk_snap_idx,
                                       executor);
            const auto end_msd = std::chrono::steady_clock::now();
            time_msd += std::chrono::duration_cast<std::chrono::milliseconds>(
                end_msd - start_msd);
            perf_observer.add_frame_elapsed(
                "MSD", static_cast<std::uint64_t>(
                           tot_snap_allfiles + global_frame_counter +
                           chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_msd - start_msd)
                    .count());
          }

          if (membrane_analyzer.enabled() &&
              chunk_snap_idx < membrane_traj.m_mol.size()) {
            auto start_membrane = std::chrono::steady_clock::now();
            membrane_analyzer.process_frame(membrane_traj, &pro_in,
                                            chunk_snap_idx, executor);
            const auto end_membrane = std::chrono::steady_clock::now();
            time_membrane +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_membrane - start_membrane);
            perf_observer.add_frame_elapsed(
                "Membrane",
                static_cast<std::uint64_t>(tot_snap_allfiles +
                                           global_frame_counter +
                                           chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_membrane -
                                                          start_membrane)
                    .count());
          }

          if (s2_analyzer.enabled()) {
            auto start_s2 = std::chrono::steady_clock::now();
            s2_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            const auto end_s2 = std::chrono::steady_clock::now();
            time_s2 += std::chrono::duration_cast<std::chrono::milliseconds>(
                end_s2 - start_s2);
            perf_observer.add_frame_elapsed(
                "S2", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                 global_frame_counter +
                                                 chunk_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap,
                std::chrono::duration<double, std::milli>(end_s2 - start_s2)
                    .count());
          }

          if (rg_analyzer.enabled()) {
            auto start_rg = std::chrono::steady_clock::now();
            rg_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
            time_rg += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_rg);
          }

          if (Arguments.flag_nocal && use_lj_tica_features &&
              !use_observable_tica_features &&
              Arguments.msm.flag_tica && tica_analyzer &&
              lj_analyzer.has_feature_vector()) {
            auto start_tica = std::chrono::steady_clock::now();
            const int current_replica = input_name - Arguments.b_input;
            const size_t absolute_global_snap_idx =
                tot_snap_allfiles + global_frame_counter + chunk_snap_idx;
            const double current_box_volume_nm3 =
                frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);
            const auto start_tica_feature_build =
                std::chrono::steady_clock::now();
            tica_features = lj_analyzer.latest_feature_vector();
            tica_analyzer->add_feature_build_seconds(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    start_tica_feature_build)
                    .count());
            tica_analyzer->accumulate_frame(tica_features);

            if (need_tica_projection_cache && msm_analyzer) {
              if (tica_analyzer->is_solved()) {
                Eigen::VectorXd proj =
                    tica_analyzer->project_frame(tica_features);
                msm_analyzer->add_projected_frame(
                    proj, current_replica,
                    static_cast<int64_t>(absolute_global_snap_idx),
                    current_box_volume_nm3);
                msm_analyzer->record_frame_time(
                    pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica);
              } else {
                msm_analyzer->add_pending_frame_metadata(
                    pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
                    static_cast<int64_t>(absolute_global_snap_idx),
                    current_box_volume_nm3);
              }
            }

            time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_tica);
          }

          if (rms_analyzer.enabled()) {
            auto start_rms = std::chrono::steady_clock::now();
            rms_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
            time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_rms);
          }

          const int current_replica = input_name - Arguments.b_input;
          const size_t absolute_global_snap_idx =
              tot_snap_allfiles + global_frame_counter + chunk_snap_idx;
          const double current_box_volume_nm3 =
              frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);

          auto start_network = std::chrono::steady_clock::now();
          if (Arguments.flag_network || Arguments.flag_pca ||
              Arguments.corr_knn.flag_corr_knn ||
              Arguments.lspmd.flag_lspmd ||
              tica_or_export_needs_dccm) {
            dccm_analyzer.process_frame(
                pro_in, lig, chunk_snap_idx, absolute_global_snap_idx,
                pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
                current_box_volume_nm3);
          }
          time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_network);

          if (Arguments.msm.flag_tica && tica_analyzer &&
              use_observable_tica_features) {
            auto start_tica = std::chrono::steady_clock::now();
            const auto start_tica_feature_build =
                std::chrono::steady_clock::now();
            Eigen::VectorXd observable_features;
            const bool have_tica_frame = build_observable_feature_vector(
                Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
                tica_ref_centered, tica_R, tica_ref_initialized,
                qnative_analyzer, dihedral_analyzer, cmap_analyzer,
                lj_analyzer, observable_features);
            tica_analyzer->add_feature_build_seconds(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    start_tica_feature_build)
                    .count());
            if (have_tica_frame &&
                feature_router.accept_frame(
                    static_cast<std::uint64_t>(absolute_global_snap_idx),
                    pro_in.m_mol[chunk_snap_idx].time_of_snap,
                    observable_features)) {
              tica_analyzer->accumulate_frame(observable_features);
              if (need_tica_projection_cache && msm_analyzer) {
                if (tica_analyzer->is_solved()) {
                  Eigen::VectorXd proj =
                      tica_analyzer->project_frame(observable_features);
                  msm_analyzer->add_projected_frame(
                      proj, current_replica,
                      static_cast<int64_t>(absolute_global_snap_idx),
                      current_box_volume_nm3);
                  msm_analyzer->record_frame_time(
                      pro_in.m_mol[chunk_snap_idx].time_of_snap,
                      current_replica);
                } else {
                  msm_analyzer->add_pending_frame_metadata(
                      pro_in.m_mol[chunk_snap_idx].time_of_snap,
                      current_replica,
                      static_cast<int64_t>(absolute_global_snap_idx),
                      current_box_volume_nm3);
                }
              }
            }
            time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_tica);
          } else if (Arguments.msm.flag_tica && tica_analyzer &&
                     !use_lj_tica_features) {
            auto start_tica = std::chrono::steady_clock::now();
            const auto start_tica_feature_build =
                std::chrono::steady_clock::now();
            bool have_tica_frame = false;
            if (tica_n_ca > 0) {
              size_t nf = dccm_analyzer.get_total_frames();
              if (nf > 0) {
                const auto &tx = dccm_analyzer.get_latest_traj_x();
                const auto &ty = dccm_analyzer.get_latest_traj_y();
                const auto &tz = dccm_analyzer.get_latest_traj_z();

                for (int a = 0; a < tica_n_ca; ++a) {
                  tica_mobile(a, 0) = tx[static_cast<size_t>(a)];
                  tica_mobile(a, 1) = ty[static_cast<size_t>(a)];
                  tica_mobile(a, 2) = tz[static_cast<size_t>(a)];
                }

                if (!tica_ref_initialized) {
                  tica_ref_centered = tica_mobile;
                  {
                    Eigen::RowVectorXd col_mean =
                        tica_ref_centered.colwise().mean();
                    tica_ref_centered.rowwise() -= col_mean;
                  }
                  tica_mobile = tica_ref_centered;
                  tica_ref_initialized = true;
                } else {
                  TitanAlignment::align_frame_inplace(tica_ref_centered,
                                                      tica_mobile, tica_R);
                }

                int feat_idx = 0;
                for (int a = 0; a < tica_n_ca; ++a) {
                  for (int b = a + 1; b < tica_n_ca; ++b) {
                    double dx = tica_mobile(a, 0) - tica_mobile(b, 0);
                    double dy = tica_mobile(a, 1) - tica_mobile(b, 1);
                    double dz = tica_mobile(a, 2) - tica_mobile(b, 2);
                    tica_features[feat_idx++] =
                        std::sqrt(dx * dx + dy * dy + dz * dz);
                  }
                }

                have_tica_frame = true;
              }
            }
            tica_analyzer->add_feature_build_seconds(
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    start_tica_feature_build)
                    .count());

            if (have_tica_frame) {
              const double current_box_volume_nm3 =
                  frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);
              tica_analyzer->accumulate_frame(tica_features);

              if (need_tica_projection_cache && msm_analyzer) {
                if (tica_analyzer->is_solved()) {
                  Eigen::VectorXd proj =
                      tica_analyzer->project_frame(tica_features);
                  msm_analyzer->add_projected_frame(
                      proj, current_replica,
                      static_cast<int64_t>(absolute_global_snap_idx),
                      current_box_volume_nm3);
                  msm_analyzer->record_frame_time(
                      pro_in.m_mol[chunk_snap_idx].time_of_snap,
                      current_replica);
                } else {
                  msm_analyzer->add_pending_frame_metadata(
                      pro_in.m_mol[chunk_snap_idx].time_of_snap,
                      current_replica,
                      static_cast<int64_t>(absolute_global_snap_idx),
                      current_box_volume_nm3);
                }
              }
            }

            time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_tica);
          }

          if (feature_router.active() && !feature_router.consume_tica()) {
            Eigen::VectorXd observable_features;
            const bool have_feature_frame = build_observable_feature_vector(
                Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
                tica_ref_centered, tica_R, tica_ref_initialized,
                qnative_analyzer, dihedral_analyzer, cmap_analyzer,
                lj_analyzer, observable_features);
            if (have_feature_frame) {
              const bool accepted_for_export = feature_router.accept_frame(
                  static_cast<std::uint64_t>(absolute_global_snap_idx),
                  pro_in.m_mol[chunk_snap_idx].time_of_snap,
                  observable_features);
              (void)accepted_for_export;
            }
          }

          // H-bond (self-interaction mode for nocal)
          if (Arguments.flag_h_bond == 1) {
            h_groups groups_1_nc, groups_2_nc;
            MOL temp_mol(pro_in.m_mol[chunk_snap_idx]);
            const size_t hbond_snap_idx = global_frame_counter + chunk_snap_idx;
            find_h_bonds(pro_in.m_mol[chunk_snap_idx], pro_in.m_mol[0],
                         temp_mol, temp_mol, groups_1_nc, groups_2_nc,
                         Arguments, hbond_snap_idx, pro_in.info,
                         pro_in.info, Hbonds_vector);
          }

          // PQR output
          if (Arguments.flag_pqr == 1)
            write_pqr(pro_in, lig, input_line, input_name, chunk_snap_idx,
                      Arguments, paths);

          // PDB output (critical for nocal)
          auto start_pdb = std::chrono::steady_clock::now();
          write_pdb(pro_in, lig, Arguments, chunk_snap_idx, paths, input_line,
                    input_name);
          time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_pdb);
        }

        global_frame_counter += chunk_num_snaps;
        continue; // skip energy calculations, read next chunk
      }
      // ── END NOCAL ───────────────────────────────────────────────────────

      if (Arguments.entropy_quasi) {
        if (Arguments.pbc.solv_pbc == 0 && Arguments.pbc.chain_ref == -1)
          align_complex(pro_in, lig, 0, 0);
        cal_entropy(pro_in, lig, Groups_from_index_file, Arguments, paths,
                    static_cast<size_t>(input_name), info_output);
      }

      if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
        size_t limit = std::min(pro_in.m_mol.size(), water_traj.m_mol.size());
        for (size_t snap = 0; snap < limit; ++snap) {
          size_t p_size = pro_in.m_mol[snap].pos.x.size();
          if (p_size > 0) {
            double new_cx = 0.0, new_cy = 0.0, new_cz = 0.0;
            for (size_t i = 0; i < p_size; ++i) {
              new_cx += pro_in.m_mol[snap].pos.x[i];
              new_cy += pro_in.m_mol[snap].pos.y[i];
              new_cz += pro_in.m_mol[snap].pos.z[i];
            }
            new_cx /= p_size;
            new_cy /= p_size;
            new_cz /= p_size;
            double shift_x = new_cx - old_cogs[snap].x;
            double shift_y = new_cy - old_cogs[snap].y;
            double shift_z = new_cz - old_cogs[snap].z;
            size_t w_size = water_traj.m_mol[snap].pos.x.size();
            for (size_t i = 0; i < w_size; ++i) {
              water_traj.m_mol[snap].pos.x[i] += shift_x;
              water_traj.m_mol[snap].pos.y[i] += shift_y;
              water_traj.m_mol[snap].pos.z[i] += shift_z;
            }
          }
        }
      }

      TitanUI::log_kv(
          log, "Processing Chunk",
          "Frames " + std::to_string(global_frame_counter + 1) + " to " +
              std::to_string(global_frame_counter + chunk_num_snaps));

      // --- INNER FRAME LOOP ---
      for (size_t chunk_snap_idx = 0; chunk_snap_idx < chunk_num_snaps;
           ++chunk_snap_idx) {
        size_t global_snap_idx = global_frame_counter + chunk_snap_idx;

        const size_t energy_lg_snap =
            (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
        if (energy_lg_snap >= lig.m_mol.size()) {
          TitanUI::print_error(
              "Energy validation: ligand frame is unavailable.");
          return 1;
        }
        const MOL &energy_lig_frame = lig.m_mol[energy_lg_snap];
        const auto energy_validation_begin = std::chrono::steady_clock::now();
        if (Arguments.flag_gbsa && Arguments.self_int) {
          const std::string message =
              std::string(titan_energy_validation::
                              kIdenticalGbsaSelectionsError) +
              ": MM/GBSA binding endpoints require distinct receptor and "
              "ligand selections; the same group cannot be used twice.";
          TitanUI::print_error(message);
          TitanUI::log_message(log, "[Validation] " + message);
          return 1;
        }
        if (Arguments.self_int && !CachedData.nonbonded_topology) {
          const std::string message =
              "unsupported_topology_self_energy: intramolecular energy/force "
              "requires topology exclusions and explicit 1-4 parameters; the "
              "legacy distance-based exclusion heuristic is not a valid "
              "substitute. Self-interaction energy output is disabled until "
              "those parameters are available to every energy/decomposition kernel.";
          TitanUI::print_error(message);
          TitanUI::log_message(log, "[Validation] " + message);
          return 1;
        }
        const auto invalid_coordinate =
            titan_energy_validation::find_nonfinite_coordinate(
                pro_in.m_mol[chunk_snap_idx], energy_lig_frame);
        if (invalid_coordinate) {
          std::ostringstream message;
          message << titan_energy_validation::kNonfiniteCoordinateError
                  << ": frame=" << (global_snap_idx + 1)
                  << ", molecule=" << invalid_coordinate->molecule
                  << ", atom=" << (invalid_coordinate->atom + 1)
                  << ", axis=" << invalid_coordinate->axis
                  << ", value=" << invalid_coordinate->value
                  << "; energy modules require finite Cartesian coordinates.";
          TitanUI::print_error(message.str());
          TitanUI::log_message(log, "[Validation] " + message.str());
          return 1;
        }
        const double minimum_pair_distance_nm =
            Arguments.flag_gbsa
                ? titan_energy_validation::kMinimumGbsaPairDistanceNm
                : titan_energy_validation::kMinimumPairDistanceNm;
        const auto overlap = titan_energy_validation::find_overlapping_atoms(
            pro_in.m_mol[chunk_snap_idx], energy_lig_frame,
            Arguments.self_int, executor, minimum_pair_distance_nm);
        const auto energy_validation_end = std::chrono::steady_clock::now();
        perf_observer.add_frame_elapsed(
            "EnergyValidation",
            static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
            pro_in.m_mol[chunk_snap_idx].time_of_snap,
            std::chrono::duration<double, std::milli>(energy_validation_end -
                                                      energy_validation_begin)
                .count(),
            "coordinate_overlap_gate", !overlap.has_value());
        if (overlap) {
          std::ostringstream message;
          message << (Arguments.flag_gbsa
                          ? titan_energy_validation::kCloseContactError
                          : titan_energy_validation::kOverlappingAtomsError)
                  << ": frame=" << (global_snap_idx + 1)
                  << ", mol1_atom=" << (overlap->mol1_atom + 1)
                  << ", mol2_atom=" << (overlap->mol2_atom + 1)
                  << ", distance_nm=" << std::scientific
                  << overlap->distance_nm
                  << "; "
                  << (Arguments.flag_gbsa
                          ? "MM/GBSA rejects cross-group distances below 0.05 "
                            "nm instead of suppressing their energy."
                          : "electrostatic/vdW energy requires distinct atom "
                            "coordinates.");
          TitanUI::print_error(message.str());
          TitanUI::log_message(log, "[Validation] " + message.str());
          return 1;
        }

        // --- LLPS per-frame analysis (async pipeline — overlaps energy calc)
        // ---
        if (Arguments.llps.flag_llps && llps_analyzer) {
          auto start_llps = std::chrono::steady_clock::now();
          const auto &mol_frame = pro_in.m_mol[chunk_snap_idx];
          // Box dimensions from CRYST1 (Å → nm)
          const float bx_nm = static_cast<float>(mol_frame.Box.get_a() / 10.0);
          const float by_nm = static_cast<float>(mol_frame.Box.get_b() / 10.0);
          const float bz_nm = static_cast<float>(mol_frame.Box.get_c() / 10.0);
          const float t_ns =
              static_cast<float>(mol_frame.time_of_snap / 1000.0);
          // Pass RNA (lig) data if mol_2_type is rna
          const MOL *rna_ptr = nullptr;
          const std::vector<mol_info> *rna_info_ptr = nullptr;
          if (Arguments.mol_2_type.find("rna") != std::string::npos ||
              Arguments.mol_2_type.find("RNA") != std::string::npos) {
            if (chunk_snap_idx < lig.m_mol.size()) {
              rna_ptr = &lig.m_mol[chunk_snap_idx];
              rna_info_ptr = &lig.info;
            }
          }
          // Submit to async queue — returns immediately, runs on worker thread
          llps_analyzer->process_frame_async(
              mol_frame, pro_in.info, bx_nm, by_nm, bz_nm, t_ns,
              static_cast<int>(global_snap_idx), rna_ptr, rna_info_ptr);
          time_llps += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_llps);
        }

        // --- LLPS DBSCAN per-frame ---
        if (dbscan_llps) {
          auto t0_dbscan = std::chrono::steady_clock::now();
          const MOL *rna_mol = nullptr;
          const std::vector<mol_info> *rna_info = nullptr;
          if (chunk_snap_idx < lig.m_mol.size()) {
            rna_mol = &lig.m_mol[chunk_snap_idx];
            rna_info = &lig.info;
          }
          static const MOL empty_mol;
          static const std::vector<mol_info> empty_info;
          // [BUG 2 note] pro.Box is used as the shared periodic box for all
          // nodes (protein + RNA/lig). This is correct when both molecules come
          // from the same trajectory frame. If lig were ever loaded from a
          // separate file with a different box, MIC for RNA nodes would be
          // wrong — document that here.
          dbscan_llps->process_frame(executor, pro_in.m_mol[chunk_snap_idx],
                                     pro_in.info, rna_mol ? *rna_mol : empty_mol,
                                     rna_info ? *rna_info : empty_info,
                                     pro_in.m_mol[chunk_snap_idx].time_of_snap,
                                     static_cast<int>(global_snap_idx + 1));
          time_llps_dbscan +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0_dbscan);
        }

        const int progress_traj_id = input_name - Arguments.b_input + 1;
        std::string status_text =
            "Traj " + std::to_string(progress_traj_id) + " | Frame " +
            std::to_string(global_snap_idx + 1) + " | Global " +
            std::to_string((tot_snap_allfiles + global_snap_idx + 1));

        const std::string module_name = progress_source_label(Arguments, name_pro);

        if (Arguments.flag_xdr) {
          TitanUI::show_continuous_progress(module_name, global_snap_idx + 1,
                                            status_text, streaming_modules);
        } else {
          TitanUI::show_progress_exact(module_name, chunk_snap_idx + 1,
                                       std::max<size_t>(1, chunk_num_snaps),
                                       status_text, streaming_modules);
        }

        if (pbe_analyzer.enabled()) {
          auto start_pbe = std::chrono::steady_clock::now();
          const size_t pbe_lg_snap = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
          const MOL *pbe_lig_frame =
              (!lig.m_mol.empty() && pbe_lg_snap < lig.m_mol.size())
                  ? &lig.m_mol[pbe_lg_snap]
                  : nullptr;
          const Multi_MOL_parameters *pbe_lig_params =
              pbe_lig_frame ? &lig.parameters : nullptr;
          pbe_analyzer.consume_frame(
              pro_in.m_mol[chunk_snap_idx], pro_in.parameters, pbe_lig_frame,
              pbe_lig_params, static_cast<std::size_t>(progress_traj_id),
              global_snap_idx, tot_snap_allfiles + global_snap_idx, executor,
              log, Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
          const auto end_pbe = std::chrono::steady_clock::now();
          time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_pbe - start_pbe);
          perf_observer.add_frame_elapsed(
              Arguments.pbe.pbsa ? "PBSA" : "PBE",
              static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_pbe - start_pbe)
                  .count());
        }

        vector<d_min> dis_min(num_threads);
        vector<cache_energy> Energy(num_threads);
        vector<cache_force> Force(num_threads);

        double E_V_{}, E_E_{};
        force f_e_{}, f_v_{};

        // Print formatted timestamps in a neat grid (10 columns)
        if (log.is_open()) {
          log << std::fixed << std::setprecision(1) << std::setw(10)
              << std::right << pro_in.m_mol[chunk_snap_idx].time_of_snap;
          if ((chunk_snap_idx + 1) % 10 == 0 || chunk_snap_idx + 1 == NUM) {
            log << "\n";
          }
        }

        AlignedDoubleVector rec_free_radii;
        AlignedDoubleVector lg_free_radii;
        AlignedDoubleVector rec_complex_radii;
        AlignedDoubleVector lg_complex_radii;
        const MOL &active_lg_mol = energy_lig_frame;
        const vector<mol_info> &active_lg_info = lig.info;
        const size_t ligand_atom_count = lig.m_mol[0].size();

        auto start_energy_eval = std::chrono::steady_clock::now();
        GBSA_energy_snap frame_gbsa_total;
        if (Arguments.flag_gbsa) {
          const bool capture_gbsa_timing = Arguments.gbsa_perf != "off";
          const auto gbsa_precompute_begin =
              capture_gbsa_timing ? std::chrono::steady_clock::now()
                                  : start_energy_eval;
          GBSASasaCache frame_sasa_cache;
#ifdef __AVX__
          rec_free_radii.resize(pro_in.m_mol[chunk_snap_idx].size());
          lg_free_radii.resize(active_lg_mol.size());
          rec_complex_radii.resize(pro_in.m_mol[chunk_snap_idx].size());
          lg_complex_radii.resize(active_lg_mol.size());

          // All four endpoints must share one rgbmax, otherwise the binding
          // cycle mixes cut and uncut descreening integrals.
          if (Arguments.igb == 66) {
            calculate_gbnsr6_binding_radii(pro_in.m_mol[chunk_snap_idx],active_lg_mol,
                rec_vdw_radii,lg_vdw_radii,pro_in.info,active_lg_info,
                rec_free_radii,lg_free_radii,rec_complex_radii,lg_complex_radii,
                Arguments.gbnsr6_spacing,&executor);
          } else {
          cal_effective_born_radii_parallel_avx2(
              pro_in.m_mol[chunk_snap_idx], rec_vdw_radii, pro_in.info,
              rec_free_radii, Arguments.igb, executor, nullptr, nullptr,
              nullptr, Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
          cal_effective_born_radii_parallel_avx2(
              active_lg_mol, lg_vdw_radii, active_lg_info, lg_free_radii,
              Arguments.igb, executor, nullptr, nullptr, nullptr,
              Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
          cal_effective_born_radii_parallel_avx2(
              pro_in.m_mol[chunk_snap_idx], rec_vdw_radii, pro_in.info,
              rec_complex_radii, Arguments.igb, executor, &active_lg_mol,
              &lg_vdw_radii, &active_lg_info, Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
          cal_effective_born_radii_parallel_avx2(
              active_lg_mol, lg_vdw_radii, active_lg_info, lg_complex_radii,
              Arguments.igb, executor, &pro_in.m_mol[chunk_snap_idx],
              &rec_vdw_radii, &pro_in.info, Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
          }

          const auto validate_born_radii = [&](const AlignedDoubleVector &values,
                                                const char *endpoint) {
            for (size_t atom = 0; atom < values.size(); ++atom) {
              if (!std::isfinite(values[atom]) || values[atom] <= 0.0) {
                std::ostringstream message;
                message << "invalid_gb_born_radius: frame="
                        << (global_snap_idx + 1) << ", endpoint=" << endpoint
                        << ", atom=" << (atom + 1)
                        << ", radius_nm=" << values[atom];
                TitanUI::print_error(message.str());
                TitanUI::log_message(log, "[Validation] " + message.str());
                return false;
              }
            }
            return true;
          };
          if (!validate_born_radii(rec_free_radii, "receptor_free") ||
              !validate_born_radii(lg_free_radii, "ligand_free") ||
              !validate_born_radii(rec_complex_radii, "receptor_complex") ||
              !validate_born_radii(lg_complex_radii, "ligand_complex")) {
            return 1;
          }

          build_gbsa_sasa_cache(pro_in.m_mol[chunk_snap_idx], active_lg_mol,
                                pro_in.parameters, lig.parameters,
                                frame_sasa_cache,
                                Arguments.gbsa_sasa_points,
                                Arguments.flag_alpb, &executor);
          if (Arguments.igb == 66)
            set_gbnsr6_electrostatic_sizes(pro_in.m_mol[chunk_snap_idx],active_lg_mol,
                pro_in.parameters,lig.parameters,pro_in.info,active_lg_info,frame_sasa_cache);
          if (Arguments.flag_alpb &&
              (!(frame_sasa_cache.alpb_rec_electrostatic_size_nm > 0.0) ||
               !std::isfinite(
                   frame_sasa_cache.alpb_rec_electrostatic_size_nm) ||
               !(frame_sasa_cache.alpb_lig_electrostatic_size_nm > 0.0) ||
               !std::isfinite(
                   frame_sasa_cache.alpb_lig_electrostatic_size_nm) ||
               !(frame_sasa_cache.alpb_complex_electrostatic_size_nm > 0.0) ||
               !std::isfinite(
                   frame_sasa_cache.alpb_complex_electrostatic_size_nm))) {
            std::ostringstream message;
            message << "invalid_alpb_electrostatic_size: frame="
                    << (global_snap_idx + 1) << ", receptor_nm="
                    << frame_sasa_cache.alpb_rec_electrostatic_size_nm
                    << ", ligand_nm="
                    << frame_sasa_cache.alpb_lig_electrostatic_size_nm
                    << ", complex_nm="
                    << frame_sasa_cache.alpb_complex_electrostatic_size_nm;
            TitanUI::print_error(message.str());
            TitanUI::log_message(log, "[Validation] " + message.str());
            return 1;
          }
#else
          std::cerr << "Error: AVX2 is required for GBSA!" << std::endl;
          exit(1);
#endif

          size_t active_lg_size = active_lg_mol.size();
          size_t rec_size = pro_in.m_mol[chunk_snap_idx].size();
          constexpr size_t gbsa_tile_atoms = 64;
          const bool tile_receptor = rec_size >= active_lg_size;
          const size_t tiled_atoms = tile_receptor ? rec_size : active_lg_size;
          const size_t gbsa_task_count =
              std::max<size_t>(1, (tiled_atoms + gbsa_tile_atoms - 1) /
                                      gbsa_tile_atoms);
          std::vector<GBSA_Aggregator> local_aggs(
              gbsa_task_count, GBSA_Aggregator(Arguments, info_output));
          std::vector<GBSA_energy_snap> gbsa_energies(gbsa_task_count);
          const auto gbsa_kernel_begin =
              capture_gbsa_timing ? std::chrono::steady_clock::now()
                                  : gbsa_precompute_begin;
          executor.parallel_for(
              gbsa_task_count, num_threads, 1,
              [&](size_t, size_t task_begin, size_t task_end) {
#ifdef __AVX__
                for (size_t task = task_begin; task < task_end; ++task) {
                  const size_t tile_begin = task * gbsa_tile_atoms;
                  const size_t rec_begin = tile_receptor ? tile_begin : 0;
                  const size_t rec_end =
                      tile_receptor
                          ? std::min(rec_size, tile_begin + gbsa_tile_atoms)
                          : rec_size;
                  const size_t lig_begin = tile_receptor ? 0 : tile_begin;
                  const size_t lig_end =
                      tile_receptor
                          ? active_lg_size
                          : std::min(active_lg_size,
                                     tile_begin + gbsa_tile_atoms);
                  cal_energy_gbsa_avx2(
                      pro_in.m_mol[chunk_snap_idx], active_lg_mol,
                      pro_in.info, active_lg_info, CachedData, rec_begin,
                      rec_end, lig_begin, lig_end, gbsa_energies[task],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters, pro_in.lcpo_p1, pro_in.lcpo_p2,
                      lig.lcpo_p1, lig.lcpo_p2, rec_free_radii,
                      lg_free_radii, rec_complex_radii, lg_complex_radii,
                      &local_aggs[task], &frame_sasa_cache, true,
                      tile_receptor || task == 0,
                      !tile_receptor || task == 0);
                }
#endif
              });

          const auto gbsa_merge_begin =
              capture_gbsa_timing ? std::chrono::steady_clock::now()
                                  : gbsa_kernel_begin;

          // Merge thread-local GBSA results back to the main pipeline
          frame_gbsa_total.time = pro_in.m_mol[chunk_snap_idx].time_of_snap;

          // double current_E_vdw = 0.0;
          // double current_E_elec = 0.0;

          GBSA_Aggregator frame_decomp(
              Arguments,
              info_output); // GÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢p
                            // cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡c
                            // luÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œng
                            // cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§a
                            // frame
                            // nÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â y

          for (size_t i{}; i < gbsa_task_count; ++i) {
            frame_gbsa_total.E_vdw += gbsa_energies[i].E_vdw;
            frame_gbsa_total.E_elec += gbsa_energies[i].E_elec;
            frame_gbsa_total.G_polar += gbsa_energies[i].G_polar;
            frame_gbsa_total.G_nonpolar += gbsa_energies[i].G_nonpolar;
            frame_gbsa_total.total_energy += gbsa_energies[i].total_energy;

            frame_decomp.merge(
                local_aggs
                    [i]); // GÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢p
                          // Thread -> Frame
          }
          if (!Energy.empty()) {
            Energy[0].vdw_ener = frame_gbsa_total.E_vdw;
            Energy[0].elec_ener = frame_gbsa_total.E_elec;
          }

          const auto finite_gbsa_frame = [](const GBSA_energy_snap &value) {
            return std::isfinite(value.E_vdw) &&
                   std::isfinite(value.E_elec) &&
                   std::isfinite(value.G_polar) &&
                   std::isfinite(value.G_nonpolar) &&
                   std::isfinite(value.total_energy);
          };
          if (!finite_gbsa_frame(frame_gbsa_total)) {
            TitanUI::print_error(
                "invalid_gbsa_energy: a non-finite MM/GBSA component was "
                "produced; the frame was not added to statistics.");
            return 1;
          }

          GBSA_energy_snap decomposition_sum;
          for (const auto &entry : frame_decomp.get_decomp()) {
            decomposition_sum.E_vdw += entry.E_vdw;
            decomposition_sum.E_elec += entry.E_elec;
            decomposition_sum.G_polar += entry.G_polar;
            decomposition_sum.G_nonpolar += entry.G_nonpolar;
            decomposition_sum.total_energy += entry.total;
          }
          const auto closes = [](double observed, double reference) {
            return std::abs(observed - reference) <=
                   1.0e-8 * std::max(1.0, std::abs(reference));
          };
          if (!closes(decomposition_sum.E_vdw, frame_gbsa_total.E_vdw) ||
              !closes(decomposition_sum.E_elec, frame_gbsa_total.E_elec) ||
              !closes(decomposition_sum.G_polar,
                      frame_gbsa_total.G_polar) ||
              !closes(decomposition_sum.G_nonpolar,
                      frame_gbsa_total.G_nonpolar) ||
              !closes(decomposition_sum.total_energy,
                      frame_gbsa_total.total_energy)) {
            TitanUI::print_error(
                "invalid_gbsa_decomposition_closure: receptor plus ligand "
                "attribution does not reproduce the frame energy.");
            return 1;
          }

          if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
            try {
              edecomp_gbsa_engine->process_frame(
                  make_gbsa_endpoint_rows(frame_decomp,
                                          *edecomp_gbsa_engine));
            } catch (const std::exception &e) {
              TitanUI::print_error(std::string("Energy decomposition GBSA: ") +
                                   e.what());
              return 1;
            }
          }

          global_gbsa_agg.commit_frame_decompositions(
              frame_decomp, tot_snap_allfiles + global_snap_idx);
          if (capture_gbsa_timing) {
            const auto gbsa_frame_end = std::chrono::steady_clock::now();
            frame_gbsa_total.time_precompute_ms =
                std::chrono::duration<double, std::milli>(
                    gbsa_kernel_begin - gbsa_precompute_begin)
                    .count();
            frame_gbsa_total.time_energy_kernel_ms =
                std::chrono::duration<double, std::milli>(
                    gbsa_merge_begin - gbsa_kernel_begin)
                    .count();
            frame_gbsa_total.time_decomposition_merge_ms =
                std::chrono::duration<double, std::milli>(
                    gbsa_frame_end - gbsa_merge_begin)
                    .count();
            frame_gbsa_total.time_total_ms =
                std::chrono::duration<double, std::milli>(
                    gbsa_frame_end - gbsa_precompute_begin)
                    .count();
          }
          if (Arguments.flag_alpb) {
            frame_gbsa_total.alpb_receptor_size_nm =
                frame_sasa_cache.alpb_rec_electrostatic_size_nm;
            frame_gbsa_total.alpb_ligand_size_nm =
                frame_sasa_cache.alpb_lig_electrostatic_size_nm;
            frame_gbsa_total.alpb_complex_size_nm =
                frame_sasa_cache.alpb_complex_electrostatic_size_nm;
          }
          global_gbsa_agg.add_snapshot(input_name - Arguments.b_input,
                                       frame_gbsa_total);

          const auto end_gbsa = std::chrono::steady_clock::now();
          time_gbsa += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_gbsa - start_energy_eval);
          perf_observer.add_frame_elapsed(
              "GBSA",
              static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_gbsa -
                                                        start_energy_eval)
                  .count());
        } else {
          auto dispatch_energy_workers = [&](auto &&worker_fn) {
            executor.parallel_do(num_threads, [&](size_t worker_id) {
              worker_fn(worker_id);
            });
          };
          const bool use_unfiltered_energy_intrin =
              Arguments.self_int == false &&
              Arguments.distance_energy_cutoff < 0.;

          if (Arguments.force_f.find("CHARMM") != string::npos) {
            dispatch_energy_workers([&](size_t i) {
              if (Arguments.flag_elec_screening == false) {
                if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
                  cal_energy_amber_intrin(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#else
                  cal_energy_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#endif
                } else {
                  cal_energy_force_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                }
              } else {
                if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
                  cal_energy_amber_screening_intrin(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#else
                  cal_energy_amber_screening(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#endif
                } else {
                  cal_energy_force_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                }
              }
            });

          } else if (Arguments.force_f.find("AMBER") != string::npos) {
            dispatch_energy_workers([&](size_t i) {
              if (Arguments.flag_elec_screening == false) {
                if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
                  if (use_unfiltered_energy_intrin) {
                    cal_energy_amber_intrin_nofilter(
                        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                        pro_in.info, lig.info, CachedData,
                        ligand_atom_count * i / num_threads,
                        ligand_atom_count * (i + 1) / num_threads, Energy[i],
                        chunk_snap_idx, Arguments, pro_in.parameters,
                        lig.parameters);
                  } else {
                    cal_energy_amber_intrin(
                        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                        pro_in.info, lig.info, CachedData,
                        ligand_atom_count * i / num_threads,
                        ligand_atom_count * (i + 1) / num_threads, Energy[i],
                        chunk_snap_idx, Arguments, pro_in.parameters,
                        lig.parameters);
                  }
#else
                  cal_energy_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#endif
                } else
                  cal_energy_force_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
              } else {
                if (Arguments.flag_cal_force == false)

#ifdef __AVX__
                  if (use_unfiltered_energy_intrin) {
                    cal_energy_amber_screening_intrin_nofilter(
                        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                        pro_in.info, lig.info, CachedData,
                        ligand_atom_count * i / num_threads,
                        ligand_atom_count * (i + 1) / num_threads, Energy[i],
                        chunk_snap_idx, Arguments, pro_in.parameters,
                        lig.parameters);
                  } else {
                    cal_energy_amber_screening_intrin(
                        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                        pro_in.info, lig.info, CachedData,
                        ligand_atom_count * i / num_threads,
                        ligand_atom_count * (i + 1) / num_threads, Energy[i],
                        chunk_snap_idx, Arguments, pro_in.parameters,
                        lig.parameters);
                  }
#else
                  cal_energy_amber_screening(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
#endif
                else
                  cal_energy_force_amber(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
              }
            });

          } else if (Arguments.force_f.find("OPLSAA") != string::npos) {
            dispatch_energy_workers([&](size_t i) {
              if (Arguments.flag_cal_force == false)
#ifdef __AVX__
                if (use_unfiltered_energy_intrin) {
                  cal_energy_opls_intrin_nofilter(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                } else {
                  cal_energy_opls_intrin(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                }
#else
                cal_energy_opls(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
#endif
              else
                cal_energy_force_opls(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
            });
          } else if (Arguments.force_f.find("MARTINI") != string::npos) {
            dispatch_energy_workers([&](size_t i) {
              if (Arguments.flag_cal_force == false)
#ifdef __AVX__
                cal_energy_martini_intrin(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
#else
                cal_energy_martini(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
#endif
              else
                cal_energy_force_martini(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
            });
          } else {
            dispatch_energy_workers([&](size_t i) {
              if (Arguments.flag_cal_force == false)
#ifdef __AVX__
                if (use_unfiltered_energy_intrin) {
                  cal_energy_gromos_intrin_nofilter(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                } else {
                  cal_energy_gromos_intrin(
                      pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                      pro_in.info, lig.info, CachedData,
                      ligand_atom_count * i / num_threads,
                      ligand_atom_count * (i + 1) / num_threads, Energy[i],
                      chunk_snap_idx, Arguments, pro_in.parameters,
                      lig.parameters);
                }
#else
                cal_energy_gromos(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
#endif
              else
                cal_energy_force_gromos(
                    pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                    pro_in.info, lig.info, CachedData,
                    ligand_atom_count * i / num_threads,
                    ligand_atom_count * (i + 1) / num_threads, Energy[i],
                    Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                    lig.parameters);
            });
          }
          time_cal += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_energy_eval);
          // cout << "done cal snap " << chunk_snap_idx << " ";
        } // Close else block for standard energy eval

        if (Arguments.self_int == true) {
          remove_double_counting(CachedData, Energy);
        }

        for (size_t i{}; i < Energy.size(); ++i) {
          E_E_ += Energy[i].elec_ener;
          E_V_ += Energy[i].vdw_ener;
          if (Arguments.flag_cal_force == true) {
            f_e_ += Force[i].forces[0];
            f_v_ += Force[i].forces[1];
          }
        }

        process_adding_energies_forces(
            Arguments, input_name, E_v, E_e, E_E_, E_V_, f_e_, f_v_, E_v_vector,
            E_e_vector, Force_e, Force_v, Force_e_vector, Force_v_vector,
            size_snap, global_snap_idx);

        // Keep time_of_snap in sync with E_e for streaming/XTC mode
        while (time_of_snap.size() < E_e.size()) {
          double t = pro_in.m_mol[chunk_snap_idx].time_of_snap;
          if (Arguments.tu == "ns")
            t /= 1000.0;
          time_of_snap.push_back(t);
        }

        if (edecomp_mm_requested(Arguments) && edecomp_engine) {
          auto start_edecomp = std::chrono::steady_clock::now();
          try {
            edecomp_engine->process_frame(
                {&CachedData, &edecomp_energy_provider,
                 &pro_in.m_mol[chunk_snap_idx],
                 &lig.m_mol[chunk_snap_idx]},
                executor);
          } catch (const std::exception &e) {
            TitanUI::print_error(std::string("Energy decomposition: ") +
                                 e.what());
            return 1;
          }
          const auto end_edecomp = std::chrono::steady_clock::now();
          time_edecomp += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_edecomp - start_edecomp);
          perf_observer.add_frame_elapsed(
              "EDecomp",
              static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_edecomp -
                                                        start_edecomp)
                  .count());
        }

        if (need_distance_scan) {
          auto start_sc{chrono::steady_clock::now()};

          if (pro_in.Index.size() > 1) {
            executor.parallel_for(
                pro_in.Index.size() - 1, num_threads, 1,
                [&](size_t /*worker_id*/, size_t begin, size_t end) {
                  for (size_t residue_idx = begin; residue_idx < end;
                       ++residue_idx) {
                    pro_in.m_mol[chunk_snap_idx].take_com(
                        pro_in.parameters, residue_idx,
                        pro_in.Index[residue_idx + 1].get_res_pos_e(0),
                        pro_in.Index, pro_in.info);
                  }
                });
          }

          if (lig.Index.size() > 1) {
            executor.parallel_for(
                lig.Index.size() - 1, num_threads, 1,
                [&](size_t /*worker_id*/, size_t begin, size_t end) {
                  for (size_t residue_idx = begin; residue_idx < end;
                       ++residue_idx) {
                    lig.m_mol[chunk_snap_idx].take_com(
                        lig.parameters, residue_idx,
                        lig.Index[residue_idx + 1].get_res_pos_e(0), lig.Index,
                        lig.info);
                  }
                });
          }

          impl_add_coms_2_mols(lig, pro_in, Arguments, chunk_snap_idx);
          /***************************************************************************************************************/

          executor.parallel_do(num_threads, [&](size_t i) {
            take_min(pro_in.m_mol[chunk_snap_idx].Com,
                     lig.m_mol[chunk_snap_idx].Com, dis_min[i],
                     i * pro_in.m_mol[chunk_snap_idx].Com.size() / num_threads,
                     (i + 1) * pro_in.m_mol[chunk_snap_idx].Com.size() /
                         num_threads,
                     0, lig.m_mol[chunk_snap_idx].Com.size(), Arguments, 0,
                     d_standard);
          });

          time_sc += chrono::duration_cast<chrono::milliseconds>(
              chrono::steady_clock::now() - start_sc);
          // cout << "\nsnap " << chunk_snap_idx << " mol1 consists of " <<
          // pro_in.m_mol[chunk_snap_idx].Com.size() << " mol2 consists of "
          // << lig.m_mol[chunk_snap_idx].Com.size() << " residues\n"; cout
          // << "\nsnap:
          // "<<chunk_snap_idx<< ": numof dismin " << dis_min.size() << " ";
          // for (auto& d : dis_min) 	cout << d.base.size() << ' '; cout <<
          // "mem size: " << Arguments.nt * (sizeof(std::vector<dmin_base>) +
          // (sizeof(dmin_base) * dis_min[0].base.size())); cout << '\n';
        }

        // --- IFP per-frame processing ---
        if (Arguments.ifp.flag_ifp && ifp_analyzer &&
            include_ifp_frame(
                pro_in, chunk_snap_idx,
                static_cast<size_t>(input_name - Arguments.b_input + 1),
                global_snap_idx + 1,
                tot_snap_allfiles + global_snap_idx + 1)) {
          auto start_ifp = std::chrono::steady_clock::now();
          size_t lg_snap_ifp = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
          TitanAwhIfp::AwhFrameContext awh_frame_ctx;
          if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer) {
            const int current_replica = input_name - Arguments.b_input;
            awh_frame_ctx = awh_ifp_analyzer->prepare_frame(
                current_replica, pro_in.m_mol[chunk_snap_idx].time_of_snap,
                static_cast<int64_t>(tot_snap_allfiles + global_snap_idx));
          }
          ifp_analyzer->process_frame(
              pro_in.m_mol[chunk_snap_idx],
              lig.m_mol[lg_snap_ifp],
              pro_in.info, lig.info,
              pro_in.Index, lig.Index,
              pro_in.info.size(),
              lig.info.size(),
              dis_min,
              Arguments.ifp,
              executor,
              num_threads,
              awh_frame_ctx.matched
          );
          if (awh_frame_ctx.matched && awh_ifp_analyzer) {
            awh_ifp_analyzer->consume_frame(
                awh_frame_ctx,
                ifp_analyzer->last_frame_contact_accumulators(),
                ifp_analyzer->last_frame_active_workers(),
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[lg_snap_ifp]);
          }
          time_ifp += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_ifp);
        }

        if (Arguments.cal_dis_sc_contact == true) {
          auto start_cscd = std::chrono::steady_clock::now();
          impl_cscd(pro_in, lig, Arguments, log, executor, chunk_snap_idx,
                    sc_flag, input_name, cscd_out_name, cscd_out_dis, paths,
                    out_cal_dis_files);
          time_cscd += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_cscd);
        }

        if (global_has_nucleic_acid) {
          auto start_na = std::chrono::steady_clock::now();
          na_profiler.process_frame(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, pro_in.Index, lig.Index, chunk_snap_idx,
              pro_in.m_mol[chunk_snap_idx].time_of_snap, executor,
              num_threads);
          time_na += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_na);
        }

        if (dssp_mol1_analyzer.enabled() ||
            dssp_mol2_analyzer.enabled()) {
          auto start_dssp = std::chrono::steady_clock::now();
            if (dssp_mol1_analyzer.enabled())
              dssp_mol1_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
            if (dssp_mol2_analyzer.enabled())
              dssp_mol2_analyzer.process_frame(lig, chunk_snap_idx, executor);
          time_dssp += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_dssp);
        }

        if (lj_analyzer.enabled()) {
          auto start_lj = std::chrono::steady_clock::now();
            lj_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
          time_lj += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_lj);
        }

        if (sq_analyzer.enabled()) {
          auto start_sq = std::chrono::steady_clock::now();
          sq_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
          time_sq += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_sq);
        }

        if (cmap_analyzer.enabled()) {
          auto start_cmap = std::chrono::steady_clock::now();
          const Multi_MOL *cmap_mol2 =
              Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
          cmap_analyzer.process_frame(pro_in, cmap_mol2, chunk_snap_idx,
                                      executor);
          const auto end_cmap = std::chrono::steady_clock::now();
          time_cmap += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_cmap - start_cmap);
          perf_observer.add_frame_elapsed(
              "CMap", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                 global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_cmap - start_cmap)
                  .count());
        }

        if (interface_analyzer.enabled()) {
          const auto start_interface = std::chrono::steady_clock::now();
          const Multi_MOL *interface_mol2 =
              Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
          try {
            interface_analyzer.process_frame(
                pro_in, interface_mol2, chunk_snap_idx, global_snap_idx,
                executor);
          } catch (const std::exception &e) {
            TitanUI::print_error(std::string("Interface analysis: ") +
                                 e.what());
            return 1;
          }
          const auto end_interface = std::chrono::steady_clock::now();
          time_interface +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  end_interface - start_interface);
          perf_observer.add_frame_elapsed(
              "Interface",
              static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_interface -
                                                        start_interface)
                  .count());
        }

        if (dihedral_analyzer.enabled()) {
          auto start_dihedral = std::chrono::steady_clock::now();
          dihedral_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
          const auto end_dihedral = std::chrono::steady_clock::now();
          time_dihedral += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_dihedral - start_dihedral);
          perf_observer.add_frame_elapsed(
              "Dihedral",
              static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_dihedral -
                                                        start_dihedral)
                  .count());
        }

        if (qnative_analyzer.enabled()) {
          auto start_qnative = std::chrono::steady_clock::now();
          const Multi_MOL *qnative_mol2 =
              Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
          qnative_analyzer.process_frame(pro_in, qnative_mol2, chunk_snap_idx,
                                         executor);
          const auto end_qnative = std::chrono::steady_clock::now();
          time_qnative +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  end_qnative - start_qnative);
          perf_observer.add_frame_elapsed(
              "QNative", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                    global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_qnative -
                                                        start_qnative)
                  .count());
        }

        if (msd_analyzer.enabled()) {
          auto start_msd = std::chrono::steady_clock::now();
          const Multi_MOL *msd_mol2 =
              Arguments.needs_mol2_for_msd() ? &lig : nullptr;
          msd_analyzer.process_frame(pro_in, msd_mol2, chunk_snap_idx,
                                     executor);
          const auto end_msd = std::chrono::steady_clock::now();
          time_msd += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_msd - start_msd);
          perf_observer.add_frame_elapsed(
              "MSD", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_msd - start_msd)
                  .count());
        }

        if (membrane_analyzer.enabled() &&
            chunk_snap_idx < membrane_traj.m_mol.size()) {
          auto start_membrane = std::chrono::steady_clock::now();
          membrane_analyzer.process_frame(membrane_traj, &pro_in,
                                          chunk_snap_idx, executor);
          const auto end_membrane = std::chrono::steady_clock::now();
          time_membrane +=
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  end_membrane - start_membrane);
          perf_observer.add_frame_elapsed(
              "Membrane", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                     global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_membrane -
                                                        start_membrane)
                  .count());
        }

        if (s2_analyzer.enabled()) {
          auto start_s2 = std::chrono::steady_clock::now();
          s2_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
          const auto end_s2 = std::chrono::steady_clock::now();
          time_s2 += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_s2 - start_s2);
          perf_observer.add_frame_elapsed(
              "S2", static_cast<std::uint64_t>(tot_snap_allfiles +
                                               global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_s2 - start_s2)
                  .count());
        }

        if (hbk_analyzer.enabled()) {
          auto start_hbk = std::chrono::steady_clock::now();
          size_t lig_snap_hbk = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
          const MOL *lig_frame_hbk =
              lig.m_mol.empty() ? nullptr : &lig.m_mol[lig_snap_hbk];
          size_t water_snap_hbk =
              (water_traj.m_mol.size() > 1) ? chunk_snap_idx : 0;
          const MOL *water_frame_hbk =
              water_traj.m_mol.empty() ? nullptr : &water_traj.m_mol[water_snap_hbk];
          hbk_analyzer.process_frame(pro_in.m_mol[chunk_snap_idx], lig_frame_hbk,
                                     water_frame_hbk, executor);
          time_hbk += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_hbk);
        }

        if (rg_analyzer.enabled()) {
          auto start_rg = std::chrono::steady_clock::now();
          rg_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
          time_rg += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_rg);
        }

        if (rms_analyzer.enabled()) {
          auto start_rms = std::chrono::steady_clock::now();
            rms_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
          time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_rms);
        }

        if (Arguments.flag_pqr == 1) {
          write_pqr(pro_in, lig, input_line, input_name, chunk_snap_idx,
                    Arguments, paths);
        }
        auto start_pdb = std::chrono::steady_clock::now();
        write_pdb(pro_in, lig, Arguments, chunk_snap_idx, paths, input_line,
                  input_name);
        time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_pdb);

        if (Arguments.flag_cas) {
          auto start_cas = std::chrono::steady_clock::now();

          cas_analyzer.process_frame(
              pro_in, lig, chunk_snap_idx, executor, num_threads, dis_min,
              CachedData, Arguments, info_output, rec_free_radii, lg_free_radii,
              rec_complex_radii, lg_complex_radii, frame_gbsa_total);

          const auto end_cas = std::chrono::steady_clock::now();
          time_cas += std::chrono::duration_cast<std::chrono::milliseconds>(
              end_cas - start_cas);
          perf_observer.add_frame_elapsed(
              "CAS", static_cast<std::uint64_t>(tot_snap_allfiles +
                                                global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              std::chrono::duration<double, std::milli>(end_cas - start_cas)
                  .count());
        }

        const int current_replica = input_name - Arguments.b_input;
        const size_t absolute_global_snap_idx =
            tot_snap_allfiles + global_snap_idx;
        const double current_box_volume_nm3 =
            frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);

        // Network Analysis (DCCM) Data Collection
        auto start_network = std::chrono::steady_clock::now();
        if (Arguments.flag_network || Arguments.flag_pca ||
            Arguments.corr_knn.flag_corr_knn ||
            Arguments.lspmd.flag_lspmd || tica_or_export_needs_dccm) {
          dccm_analyzer.process_frame(
              pro_in, lig, chunk_snap_idx, absolute_global_snap_idx,
              pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
              current_box_volume_nm3);
        }
        time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_network);

        // --- TICA per-frame accumulation (pairwise Cα distances + Kabsch
        // align) ---
        auto start_tica = std::chrono::steady_clock::now();
        if (Arguments.msm.flag_tica && tica_analyzer) {
          const auto start_tica_feature_build =
              std::chrono::steady_clock::now();
          bool have_tica_frame = false;
          if (use_observable_tica_features) {
            Eigen::VectorXd observable_features;
            have_tica_frame = build_observable_feature_vector(
                Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
                tica_ref_centered, tica_R, tica_ref_initialized,
                qnative_analyzer, dihedral_analyzer, cmap_analyzer,
                lj_analyzer, observable_features);
            if (have_tica_frame &&
                feature_router.accept_frame(
                    static_cast<std::uint64_t>(absolute_global_snap_idx),
                    pro_in.m_mol[chunk_snap_idx].time_of_snap,
                    observable_features)) {
              tica_features = observable_features;
            } else {
              have_tica_frame = false;
            }
          } else if (use_lj_tica_features && lj_analyzer.has_feature_vector()) {
            tica_features = lj_analyzer.latest_feature_vector();
            have_tica_frame = (tica_features.size() > 0);
          } else if (tica_n_ca > 0) {
            size_t nf = dccm_analyzer.get_total_frames();
            if (nf > 0) {
              const auto &tx = dccm_analyzer.get_latest_traj_x();
              const auto &ty = dccm_analyzer.get_latest_traj_y();
              const auto &tz = dccm_analyzer.get_latest_traj_z();

            // Fill tica_mobile (pre-allocated N_ca×3)
            for (int a = 0; a < tica_n_ca; ++a) {
              tica_mobile(a, 0) = tx[static_cast<size_t>(a)];
              tica_mobile(a, 1) = ty[static_cast<size_t>(a)];
              tica_mobile(a, 2) = tz[static_cast<size_t>(a)];
            }

            if (!tica_ref_initialized) {
              // Frame 0: center and store as Kabsch reference
              tica_ref_centered = tica_mobile;
              {
                Eigen::RowVectorXd col_mean = tica_ref_centered.colwise().mean();
                tica_ref_centered.rowwise() -= col_mean;
              }
              tica_mobile = tica_ref_centered;
              tica_ref_initialized = true;
            } else {
              // Frame > 0: zero-alloc Kabsch alignment
              TitanAlignment::align_frame_inplace(tica_ref_centered,
                                                  tica_mobile, tica_R);
            }

            // Science fix: pairwise inter-Cα distances (rotation+translation
            // invariant) D = N_ca*(N_ca-1)/2 — no spurious global
            // rotation/translation modes
            int feat_idx = 0;
            for (int a = 0; a < tica_n_ca; ++a) {
              for (int b = a + 1; b < tica_n_ca; ++b) {
                double dx = tica_mobile(a, 0) - tica_mobile(b, 0);
                double dy = tica_mobile(a, 1) - tica_mobile(b, 1);
                double dz = tica_mobile(a, 2) - tica_mobile(b, 2);
                tica_features[feat_idx++] =
                    std::sqrt(dx * dx + dy * dy + dz * dz);
              }
            }

              have_tica_frame = true;
            }
          }
          tica_analyzer->add_feature_build_seconds(
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now() -
                  start_tica_feature_build)
                  .count());
          if (have_tica_frame) {
            const double current_box_volume_nm3 =
                frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);
            tica_analyzer->accumulate_frame(tica_features);

            if (need_tica_projection_cache && msm_analyzer) {
              if (tica_analyzer->is_solved()) {
                Eigen::VectorXd proj =
                    tica_analyzer->project_frame(tica_features);
                msm_analyzer->add_projected_frame(
                    proj, current_replica,
                    static_cast<int64_t>(absolute_global_snap_idx),
                    current_box_volume_nm3);
                msm_analyzer->record_frame_time(
                    pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica);
              } else {
                msm_analyzer->add_pending_frame_metadata(
                    pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
                    static_cast<int64_t>(absolute_global_snap_idx),
                    current_box_volume_nm3);
              }
            }
          }
        }
        time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_tica);

        if (feature_router.active() && !feature_router.consume_tica()) {
          Eigen::VectorXd observable_features;
          const bool have_feature_frame = build_observable_feature_vector(
              Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
              tica_ref_centered, tica_R, tica_ref_initialized, qnative_analyzer,
              dihedral_analyzer, cmap_analyzer, lj_analyzer,
              observable_features);
          if (have_feature_frame) {
            const bool accepted_for_export = feature_router.accept_frame(
                static_cast<std::uint64_t>(absolute_global_snap_idx),
                pro_in.m_mol[chunk_snap_idx].time_of_snap, observable_features);
            (void)accepted_for_export;
          }
        }

        auto start_tda = std::chrono::steady_clock::now();
        if (Arguments.flag_tda) {
          if (!Arguments.flag_sidechain) {
            TitanUI::print_info(
                "TDA is running without sidechain filtering (extracting ALL "
                "heavy atoms). Enable flag -sc for active-only.");
            std::vector<int>
                target_idx_1; // Empty means "Extract ALL Heavy Atoms"
            std::vector<int> target_idx_2;
            std::string tda_out = "tda_global_pointcloud.csv";
            tda_analyzer.export_TDA_pointcloud(
                pro_in, lig, input_name, chunk_snap_idx, target_idx_1,
                target_idx_2, Arguments, tda_out);
          } else {
            std::unordered_set<int> active_1;
            std::unordered_set<int> active_2;

            bool mol1_lig = (Arguments.mol_1_type == "ligand" ||
                             Arguments.mol_1_type == "lig");
            bool mol2_lig = (Arguments.mol_2_type == "ligand" ||
                             Arguments.mol_2_type == "lig");

            // Directly grab the active residue/atom numbers forming contacts
            // from d_min array
            for (size_t t{}; t < num_threads; ++t) {
              for (const auto &contact : dis_min[t].base) {
                if (mol1_lig) {
                  active_1.insert(contact.atom_num_rec -
                                  1); // 0-indexed atom num
                } else {
                  active_1.insert(contact.resnr_rec);
                }

                if (mol2_lig) {
                  active_2.insert(contact.atom_num - 1); // 0-indexed atom num
                } else {
                  active_2.insert(contact.resnr_ab);
                }
              }
            }

            std::vector<int> target_idx_1(active_1.begin(), active_1.end());
            std::vector<int> target_idx_2(active_2.begin(), active_2.end());

            std::string tda_out = "tda_global_pointcloud.csv";
            tda_analyzer.export_TDA_pointcloud(
                pro_in, lig, input_name, chunk_snap_idx, target_idx_1,
                target_idx_2, Arguments, tda_out);
          }
        }
        time_tda += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_tda);

        auto start_fmap{std::chrono::steady_clock::now()};
        if (Arguments.flag_cal_force == true &&
            (Arguments.force_map == 1 || Arguments.force_map == 3) &&
            !CachedData.elec_ener.empty()) {
          wrap_cal_force_map(executor, pro_in, lig, Arguments, chunk_snap_idx,
                             dis_min, CachedData, Force_res_mol1,
                             Force_res_mol2, info_output);
        }
        time_fmap += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_fmap);

        auto start_hb{std::chrono::steady_clock::now()};

        if (Arguments.flag_h_bond == 1) {
          find_h_bonds(pro_in.m_mol[chunk_snap_idx], pro_in.m_mol[0],
                       lig.m_mol[chunk_snap_idx], lig.m_mol[0], groups_1,
                       groups_2, Arguments, global_snap_idx, pro_in.info,
                       lig.info, Hbonds_vector);

          if (Arguments.nt >= 2) {
            executor.parallel_do(2, [&](size_t worker_id) {
              if (worker_id == 0) {
                add_hb_res(pro_in.m_mol[chunk_snap_idx],
                           lig.m_mol[chunk_snap_idx], pro_in.info, lig.info,
                           Arguments, hb_rec_1traj);
              } else {
                add_hb_res(lig.m_mol[chunk_snap_idx],
                           pro_in.m_mol[chunk_snap_idx], lig.info, pro_in.info,
                           Arguments, hb_lg_1traj);
              }
            });
          } else {
            add_hb_res(pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                       pro_in.info, lig.info, Arguments, hb_rec_1traj);
            add_hb_res(lig.m_mol[chunk_snap_idx], pro_in.m_mol[chunk_snap_idx],
                       lig.info, pro_in.info, Arguments, hb_lg_1traj);
          }
        } // End of flag_h_bond block

        time_hb += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_hb);

        if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
          if (chunk_snap_idx >= water_traj.m_mol.size()) {
            if (chunk_snap_idx == 1) {
              std::cout
                  << "\n[WARNING] Water group has only "
                  << water_traj.m_mol.size() << " frame(s), but trajectory has "
                  << NUM << " frames.\n"
                  << "Water analysis will only process available frames!\n";
            }
          } else {
            // Optimization: Distill heavy atom indices just once
            static std::vector<size_t> rec_heavy_atoms;
            static std::vector<size_t> lig_heavy_atoms;

            if (chunk_snap_idx == 0) {
              rec_heavy_atoms.clear();
              lig_heavy_atoms.clear();

              for (size_t i = 0; i < pro_in.info.size(); ++i) {
                if (pro_in.info[i].get_atom()[0] != 'H')
                  rec_heavy_atoms.emplace_back(i);
              }
              for (size_t i = 0; i < lig.info.size(); ++i) {
                if (lig.info[i].get_atom()[0] != 'H')
                  lig_heavy_atoms.emplace_back(i);
              }
            }

            const MOL &w_frame = water_traj.m_mol[chunk_snap_idx];
            const MOL &p_frame = pro_in.m_mol[chunk_snap_idx];
            size_t l_snap = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
            const MOL &l_frame = lig.m_mol[l_snap];

            /*if (chunk_snap_idx == 0) {
              std::cout << "[DEBUG WB] Snapshot 0 - rec heavy atoms: "
                        << rec_heavy_atoms.size()
                        << ", lig heavy atoms: " << lig_heavy_atoms.size()
                        << ", water atoms: " << w_frame.pos.x.size() << "\n";
            }*/

            // GÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âi
            // class WaterAnalyzer
            // ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¹Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â£
            // ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¹Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â£c
            // tÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ch
            // file
            // cÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±c
            // kÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³
            // gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â»ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Ân
            // gÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ng!
            auto start_water = std::chrono::steady_clock::now();
            water_analyzer.process_frame(
                p_frame, l_frame, w_frame, rec_heavy_atoms, lig_heavy_atoms,
                pro_in.info, lig.info, input_name, chunk_snap_idx,
                info_output.get_global_water_bridges_timeseries(),
                info_output.get_water_pdb_filename(Arguments, input_line,
                                                   input_name));
            time_water += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_water);

            // --- GIST per-frame ---
            if (Arguments.gist.flag_gist && gist_analyzer) {
              const bool gist_use_tica =
                  Arguments.msm.flag_tica && tica_ref_initialized;
              gist_analyzer->process_frame(
                  p_frame, l_frame, w_frame, pro_in.info,
                  gist_use_tica ? tica_ref_centered : Eigen::MatrixXd{},
                  gist_use_tica ? tica_R : Eigen::Matrix3d::Identity(),
                  (global_snap_idx == 0), gist_use_tica);
            }
          }
        }
        auto start_subgroups = std::chrono::steady_clock::now();
        if (Arguments.flag_sub_groups == true && Arguments.flag_gbsa == false) {
          if (sg_streaming_analyzer.active()) {
            const std::size_t sg_lig_snap =
                (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
            sg_streaming_analyzer.process_frame(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[sg_lig_snap], pro_in,
                lig, CachedData, Arguments, executor,
                static_cast<std::size_t>(input_name - Arguments.b_input));
            if (Arguments.flag_cal_mindis == true)
              mindis_subgroups(Groups_from_index_file, pro_in, lig,
                               chunk_snap_idx, NUM, Arguments, executor,
                               MIN_DIS);
          } else if (!CachedData.elec_ener.empty()) {
            Energy_force_subgroups(Groups_from_index_file, pro_in, lig,
                                   CachedData, EF_subgroups, Energy_groups_out,
                                   chunk_snap_idx, NUM, Arguments, executor);
            if (Arguments.flag_cal_mindis == true)
              mindis_subgroups(Groups_from_index_file, pro_in, lig,
                               chunk_snap_idx, NUM, Arguments, executor,
                               MIN_DIS);
          }
        }
        time_subgroups_cal +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_subgroups);
        // cout << "check sub groups..";
        auto start_dis2commol = std::chrono::steady_clock::now();
        if (Arguments.com_dis_energy_2mol != 4)
          dis_com_2mol(pro_in, lig, chunk_snap_idx, NUM, Arguments, input_name,
                       paths, E_E_, E_V_, info_output);
        if (Arguments.com_dis_2mol != 0) {
          fstream output_dis_2mol_file(
              paths.get_path(input_name - Arguments.b_input) + "/" +
                  info_output.getDisCom2Mol(),
              ios::out | ios::app);
          pro_in.impl_com_dis_2mol(lig, Arguments, info_output,
                                   output_dis_2mol_file, chunk_snap_idx);
        }
        time_other += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_dis2commol);

        // --- ML Feature Extraction ---
        if (Arguments.flag_ml) {
          ml_extractor.process_frame(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, lig.parameters, dis_min, E_E_ + E_V_, 6.0);
        }

        // In chunk streaming, we do not clear individual MOLs here, we clear
        // the entire vector before reading the next chunk.

      } // End of inner chunk loop

      // We update the sizes for the next iteration
      // INITIATE THREAD-LOCAL FORCES FOR OMP LOOP
      for (size_t f = 0; f < pro_in.m_mol.size(); ++f) {
        if (pro_in.compute_forces)
          pro_in.m_mol[f].initialize_forces(num_threads);
      }
      for (size_t f = 0; f < lig.m_mol.size(); ++f) {
        if (lig.compute_forces)
          lig.m_mol[f].initialize_forces(num_threads);
      }

      global_frame_counter += chunk_num_snaps;

    } // End of outer chunk while loop

    bool consumed_reference_structural_frame = false;
    if (global_frame_counter == 0 && !Arguments.flag_xdr &&
        !saved_pro_frame0.empty()) {
      pro_in.m_mol = saved_pro_frame0;
      lig.m_mol = saved_lig_frame0;
      membrane_traj.m_mol = saved_membrane_frame0;

      if (dssp_mol1_analyzer.enabled() ||
          dssp_mol2_analyzer.enabled()) {
        auto start_dssp = std::chrono::steady_clock::now();
        if (dssp_mol1_analyzer.enabled())
          dssp_mol1_analyzer.process_frame(pro_in, 0, executor);
        if (dssp_mol2_analyzer.enabled() &&
            !lig.m_mol.empty())
          dssp_mol2_analyzer.process_frame(lig, 0, executor);
        time_dssp += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_dssp);
        consumed_reference_structural_frame = true;
      }

      const bool rms_reference_ready =
          Arguments.flag_nocal || rms_analyzer.mol2_trace_count() == 0 ||
          !lig.m_mol.empty();
      if (rms_analyzer.enabled() && rms_reference_ready) {
        auto start_rms = std::chrono::steady_clock::now();
        rms_analyzer.process_frame(pro_in, lig, 0, executor);
        time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_rms);
        consumed_reference_structural_frame = true;
      }

      if (cmap_analyzer.enabled()) {
        auto start_cmap = std::chrono::steady_clock::now();
        const Multi_MOL *cmap_mol2 =
            Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
        cmap_analyzer.process_frame(pro_in, cmap_mol2, 0, executor);
        time_cmap += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_cmap);
        consumed_reference_structural_frame = true;
      }

      if (interface_analyzer.enabled()) {
        const auto start_interface = std::chrono::steady_clock::now();
        const Multi_MOL *interface_mol2 =
            Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
        try {
          interface_analyzer.process_frame(pro_in, interface_mol2, 0, 0,
                                           executor);
        } catch (const std::exception &e) {
          TitanUI::print_error(std::string("Interface analysis: ") +
                               e.what());
          return 1;
        }
        time_interface +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_interface);
        consumed_reference_structural_frame = true;
      }

      if (qnative_analyzer.enabled()) {
        auto start_qnative = std::chrono::steady_clock::now();
        const Multi_MOL *qnative_mol2 =
            Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
        qnative_analyzer.process_frame(pro_in, qnative_mol2, 0, executor);
        time_qnative += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_qnative);
        consumed_reference_structural_frame = true;
      }

      if (msd_analyzer.enabled()) {
        auto start_msd = std::chrono::steady_clock::now();
        const Multi_MOL *msd_mol2 =
            Arguments.needs_mol2_for_msd() ? &lig : nullptr;
        msd_analyzer.process_frame(pro_in, msd_mol2, 0, executor);
        time_msd += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_msd);
        consumed_reference_structural_frame = true;
      }

      if (dihedral_analyzer.enabled()) {
        auto start_dihedral = std::chrono::steady_clock::now();
        dihedral_analyzer.process_frame(pro_in, 0, executor);
        time_dihedral += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_dihedral);
        consumed_reference_structural_frame = true;
      }

      if (membrane_analyzer.enabled() && !membrane_traj.m_mol.empty()) {
        auto start_membrane = std::chrono::steady_clock::now();
        membrane_analyzer.process_frame(membrane_traj, &pro_in, 0, executor);
        time_membrane +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_membrane);
        consumed_reference_structural_frame = true;
      }

      if (s2_analyzer.enabled()) {
        auto start_s2 = std::chrono::steady_clock::now();
        s2_analyzer.process_frame(pro_in, 0, executor);
        time_s2 += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_s2);
        consumed_reference_structural_frame = true;
      }
    }

    if (pbe_analyzer.enabled() && global_frame_counter == 0 &&
        !Arguments.flag_xdr && !saved_pro_frame0.empty()) {
      auto start_pbe = std::chrono::steady_clock::now();
      const MOL *pbe_lig_frame =
          saved_lig_frame0.empty() ? nullptr : &saved_lig_frame0.front();
      const Multi_MOL_parameters *pbe_lig_params =
          pbe_lig_frame ? &lig.parameters : nullptr;
      pbe_analyzer.consume_frame(
          saved_pro_frame0.front(), pro_in.parameters, pbe_lig_frame,
          pbe_lig_params,
          static_cast<std::size_t>(input_name - Arguments.b_input + 1), 0,
          tot_snap_allfiles, executor, log,
          Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
      time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_pbe);
    }

    if (consumed_reference_structural_frame) {
      global_frame_counter = 1;
      TitanUI::log_kv(log, "Reference Frame Analysis",
                      "Processed frame 0 for single-frame structural modules");
    }

    tot_snap_allfiles += global_frame_counter;

    // --- IFP post-trajectory export ---
    // (aggregate across all files at the very end, handled after all-files loop)

    // Reset TICA lag buffer at trajectory boundary
    if (Arguments.msm.flag_tica && tica_analyzer) {
      tica_analyzer->reset_lag_buffer();
    }
    // Insert MSM trajectory break at each file boundary
    if (need_tica_projection_cache && msm_analyzer &&
        Arguments.msm.msm_auto_boundary) {
      msm_analyzer->add_trajectory_break();
      const int64_t msm_frame_count =
          (msm_analyzer->n_frames() > 0) ? msm_analyzer->n_frames()
                                         : msm_analyzer->pending_feature_frames();
      TitanUI::log_kv(log, "MSM trajectory break at frame",
                      std::to_string(msm_frame_count));
    }

    if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
      water_analyzer.finalize_trajectory();
    }
    if (pbe_analyzer.enabled()) {
      pbe_analyzer.end_trajectory(log);
    }
    if (edecomp_mm_requested(Arguments) && edecomp_engine) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      try {
        const auto result = edecomp_engine->finalize_trajectory();
        titan_energy_decomp::export_result(
            result, base_dir,
            edecomp_output_suffix(Arguments, input_line, input_name));
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Energy decomposition finalize: ") +
                             e.what());
        return 1;
      }
    }
    if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      try {
        const auto result =
            edecomp_gbsa_engine->finalize_trajectory("mmgbsa");
        titan_energy_decomp::export_endpoint_result(
            result, base_dir, "mmgbsa",
            edecomp_output_suffix(Arguments, input_line, input_name));
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Energy decomposition GBSA finalize: ") +
                             e.what());
        return 1;
      }
    }
    if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      try {
        const auto result =
            edecomp_pbsa_engine->finalize_trajectory("mmpbsa");
        titan_energy_decomp::export_endpoint_result(
            result, base_dir, "mmpbsa",
            edecomp_output_suffix(Arguments, input_line, input_name));
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Energy decomposition PBSA finalize: ") +
                             e.what());
        return 1;
      }
    }
    dssp_mol1_analyzer.finalize_trajectory();
    dssp_mol2_analyzer.finalize_trajectory();
    if (lj_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const std::string lj_timeseries_out =
          base_dir + "/" +
          info_output.get_lj_timeseries_filename(Arguments, input_line,
                                                 input_name);
      const std::string lj_rho_out =
          base_dir + "/" +
          info_output.get_lj_rho_r_filename(Arguments, input_line, input_name);
      const std::string lj_feature_out =
          base_dir + "/" +
          info_output.get_lj_features_filename(Arguments, input_line,
                                               input_name);
      const std::string lj_summary_out =
          base_dir + "/" +
          info_output.get_lj_summary_filename(Arguments, input_line,
                                              input_name);
      lj_analyzer.finalize_trajectory(lj_timeseries_out, lj_rho_out,
                                      lj_feature_out,
                                      lj_summary_out, log);
      lj_feature_files.push_back(lj_feature_out);
    }
    if (sq_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      sq_analyzer.finalize_trajectory(base_dir, info_output, Arguments, input_line,
                                      input_name, log);
    }
    if (cmap_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_cmap_export = std::chrono::steady_clock::now();
      cmap_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                        input_line, input_name, log);
      perf_observer.add_export_elapsed(
          "CMap", elapsed_ms_since(start_cmap_export));
    }
    if (interface_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_interface_export = std::chrono::steady_clock::now();
      try {
        interface_analyzer.finalize_trajectory(
            base_dir, info_output, Arguments, input_line, input_name, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Interface analysis export: ") +
                             e.what());
        return 1;
      }
      perf_observer.add_export_elapsed(
          "Interface", elapsed_ms_since(start_interface_export));
    }
    if (qnative_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_qnative_export = std::chrono::steady_clock::now();
      qnative_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                           input_line, input_name, log);
      perf_observer.add_export_elapsed(
          "QNative", elapsed_ms_since(start_qnative_export));
    }
    if (msd_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_msd_export = std::chrono::steady_clock::now();
      try {
        msd_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                         input_line, input_name, executor, log);
      } catch (const std::exception &e) {
        TitanUI::print_error(e.what());
        return 1;
      }
      perf_observer.add_export_elapsed("MSD",
                                       elapsed_ms_since(start_msd_export));
    }
    if (dihedral_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_dihedral_export = std::chrono::steady_clock::now();
      dihedral_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                            input_line, input_name, log);
      perf_observer.add_export_elapsed(
          "Dihedral", elapsed_ms_since(start_dihedral_export));
    }
    if (feature_router.active()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      feature_router.export_outputs(
          base_dir + "/" +
              info_output.get_features_by_frame_filename(Arguments, input_line,
                                                         input_name),
          base_dir + "/" +
              info_output.get_features_metadata_filename(Arguments, input_line,
                                                         input_name),
          base_dir + "/" +
              info_output.get_feature_export_summary_filename(
                  Arguments, input_line, input_name),
          log);
    }
    if (membrane_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_membrane_export = std::chrono::steady_clock::now();
      membrane_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                            input_line, input_name, log);
      perf_observer.add_export_elapsed(
          "Membrane", elapsed_ms_since(start_membrane_export));
    }
    if (s2_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const auto start_s2_export = std::chrono::steady_clock::now();
      s2_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                      input_line, input_name, executor, log);
      perf_observer.add_export_elapsed("S2",
                                       elapsed_ms_since(start_s2_export));
    }
    if (hbk_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      hbk_analyzer.finalize_trajectory(base_dir, info_output, Arguments,
                                       input_line, input_name, log);
    }
    if (rg_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const std::string rg2_out =
          base_dir + "/" +
          info_output.get_rg2_filename(Arguments, input_line, input_name);
      const std::string rg2_summary_out =
          base_dir + "/" +
          info_output.get_rg2_summary_filename(Arguments, input_line, input_name);
      const std::string rg2_hist_out =
          base_dir + "/" +
          info_output.get_rg2_hist_filename(Arguments, input_line, input_name);
      const std::string traj_label =
          (!input_line.empty() && traj_index < input_line.size())
              ? input_line[traj_index]
              : ("traj_" + std::to_string(traj_index));
      rg_analyzer.finalize_trajectory(rg2_out, rg2_summary_out, rg2_hist_out,
                                      traj_index, traj_label, log);
    }

    if (rms_analyzer.enabled()) {
      const std::size_t traj_index =
          input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
      const std::size_t path_index =
          (Arguments.flag_separation_results && traj_index < paths.get_size())
              ? traj_index
              : 0;
      const std::string base_dir = paths.get_path(path_index);
      const std::string rmsd_out =
          base_dir + "/" +
          info_output.get_rmsd_filename(Arguments, input_line, input_name);
      const std::string rmsf_out =
          base_dir + "/" +
          info_output.get_rmsf_filename(Arguments, input_line, input_name);
      rms_analyzer.finalize_trajectory(rmsd_out, rmsf_out, log);
    }

    if (!out_cal_dis_files.empty()) {
      for (auto &file : out_cal_dis_files)
        if (file.is_open())
          file.close();
    }

    auto start_print_force = std::chrono::steady_clock::now();
    if (Arguments.flag_cal_force == true && Arguments.print_force_res == true) {
      print_force_res(Force_res_mol1, info_output.get_group_1_name(), paths,
                      input_name - Arguments.b_input, Arguments);
      print_force_res(Force_res_mol2, info_output.get_group_2_name(), paths,
                      input_name - Arguments.b_input, Arguments);
    }
    time_print_force += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_print_force);

    auto start_maxforce = std::chrono::steady_clock::now();
    if (Arguments.flag_maxforce == true) {
      find_maxforce_res(Force_res_mol1, info_output.get_group_1_name(), paths,
                        input_name - Arguments.b_input, Arguments,
                        Force_res_stat_mol1);
      find_maxforce_res(Force_res_mol2, info_output.get_group_2_name(), paths,
                        input_name - Arguments.b_input, Arguments,
                        Force_res_stat_mol2);
    }
    time_maxforce += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_maxforce);

    if (Arguments.flag_h_bond == 1) {
      hb_lg_1traj.snap_num = global_frame_counter;
      hb_rec_1traj.snap_num = global_frame_counter;
      res_hb_lg.emplace_back(hb_lg_1traj);
      res_hb_rec.emplace_back(hb_rec_1traj);
    }

    if (Arguments.flag_h_bond == 1) {
      auto start_hb_flush = std::chrono::steady_clock::now();
      finalize_hbond_output_files();
      time_hb += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_hb_flush);
    }

    if (Arguments.opdb) {
      auto start_pdb_flush = std::chrono::steady_clock::now();
      finalize_pdb_output_stream();
      time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_pdb_flush);
    }

    /**************************************************************************/
    TitanUI::log_section(log, "PERFORMANCE & TIMING REPORT");
    double total_traj_time = std::chrono::duration_cast<milliseconds>(
                                 std::chrono::steady_clock::now() -
                                 trajectory_timing_start)
                                 .count() /
                             1000.0;
    TitanUI::log_kv(log, "Timing Scope", "current trajectory/input");
    const double input_init_time =
        (time_input_init - time_input_init_base).count() / 1000.0;
    const double input_stream_time =
        (time_input_stream - time_input_stream_base).count() / 1000.0;
    const std::string input_init_label =
        Arguments.flag_xdr ? "Reference Parse + XDR Init" : "Input Init (PDB)";
    const std::string input_stream_label =
        Arguments.flag_xdr ? "Trajectory Streaming (XDR)"
                           : "Trajectory Streaming (PDB)";
    TitanUI::log_time(log, input_init_label, input_init_time);
    TitanUI::log_time(log, input_stream_label, input_stream_time);
    TitanUI::log_time(log, "Solving PBC",
                      (time_solve_pbc - time_solve_pbc_base).count() / 1000.0);
    TitanUI::log_time(log, "GBSA Calculation",
                      (time_gbsa - time_gbsa_base).count() / 1000.0);
    TitanUI::log_time(log, "PBE Calculation",
                      (time_pbe - time_pbe_base).count() / 1000.0);
    TitanUI::log_time(log, "CAS Calculation",
                      (time_cas - time_cas_base).count() / 1000.0);
    TitanUI::log_time(log, "DSSP Analysis",
                      (time_dssp - time_dssp_base).count() / 1000.0);
    TitanUI::log_time(log, "HBond Kinetics Analysis",
                      (time_hbk - time_hbk_base).count() / 1000.0);
    TitanUI::log_time(log, "LJ Analysis",
                      (time_lj - time_lj_base).count() / 1000.0);
    TitanUI::log_time(log, "SQ Analysis",
                      (time_sq - time_sq_base).count() / 1000.0);
    TitanUI::log_time(log, "CMap Analysis",
                      (time_cmap - time_cmap_base).count() / 1000.0);
    TitanUI::log_time(
        log, "Interface Analysis",
        (time_interface - time_interface_base).count() / 1000.0);
    TitanUI::log_time(log, "Dihedral Analysis",
                      (time_dihedral - time_dihedral_base).count() / 1000.0);
    TitanUI::log_time(log, "QNative Analysis",
                      (time_qnative - time_qnative_base).count() / 1000.0);
    TitanUI::log_time(log, "MSD Analysis",
                      (time_msd - time_msd_base).count() / 1000.0);
    TitanUI::log_time(log, "Membrane Analysis",
                      (time_membrane - time_membrane_base).count() / 1000.0);
    TitanUI::log_time(log, "S2 Analysis",
                      (time_s2 - time_s2_base).count() / 1000.0);
    TitanUI::log_time(log, "RG2 Analysis",
                      (time_rg - time_rg_base).count() / 1000.0);
    TitanUI::log_time(log, "RMSD/RMSF Analysis",
                      (time_rms - time_rms_base).count() / 1000.0);
    TitanUI::log_time(log, "Network Data Collection",
                      (time_network - time_network_base).count() / 1000.0);
    TitanUI::log_time(log, "Standard Elec+vdW Calc",
                      (time_cal - time_cal_base).count() / 1000.0);
    TitanUI::log_time(log, "Sidechain Contact",
                      (time_sc - time_sc_base).count() / 1000.0);
    TitanUI::log_time(log, "Sidechain CSCD",
                      (time_cscd - time_cscd_base).count() / 1000.0);
    TitanUI::log_time(log, "Hydrogen Bonds",
                      (time_hb - time_hb_base).count() / 1000.0);
    TitanUI::log_time(log, "PDB Output",
                      (time_pdb - time_pdb_base).count() / 1000.0);
    TitanUI::log_time(log, "Force Map (FMAP)",
                      (time_fmap - time_fmap_base).count() / 1000.0);
    TitanUI::log_time(log, "Contact Map & Dis2Mol",
                      (time_other - time_other_base).count() / 1000.0);
    TitanUI::log_time(log, "Water Bridges Analysis",
                      (time_water - time_water_base).count() / 1000.0);
    TitanUI::log_time(log, "Subgroups Calc",
                      (time_subgroups_cal - time_subgroups_cal_base).count() /
                          1000.0);
    TitanUI::log_time(log, "Nucleic Acid Profiler",
                      (time_na - time_na_base).count() / 1000.0);
    TitanUI::log_time(log, "LLPS Analysis",
                      (time_llps - time_llps_base).count() / 1000.0);
    TitanUI::log_time(log, "TICA/MSM Accumulation",
                      (time_tica - time_tica_base).count() / 1000.0);
    TitanUI::log_time(log, "IFP Analysis",
                      (time_ifp - time_ifp_base).count() / 1000.0);
    const double accounted_traj_time =
        input_init_time + input_stream_time +
        (time_solve_pbc - time_solve_pbc_base).count() / 1000.0 +
        (time_gbsa - time_gbsa_base).count() / 1000.0 +
        (time_pbe - time_pbe_base).count() / 1000.0 +
        (time_cas - time_cas_base).count() / 1000.0 +
        (time_dssp - time_dssp_base).count() / 1000.0 +
        (time_hbk - time_hbk_base).count() / 1000.0 +
        (time_lj - time_lj_base).count() / 1000.0 +
        (time_sq - time_sq_base).count() / 1000.0 +
        (time_cmap - time_cmap_base).count() / 1000.0 +
        (time_interface - time_interface_base).count() / 1000.0 +
        (time_dihedral - time_dihedral_base).count() / 1000.0 +
        (time_qnative - time_qnative_base).count() / 1000.0 +
        (time_msd - time_msd_base).count() / 1000.0 +
        (time_membrane - time_membrane_base).count() / 1000.0 +
        (time_s2 - time_s2_base).count() / 1000.0 +
        (time_rg - time_rg_base).count() / 1000.0 +
        (time_rms - time_rms_base).count() / 1000.0 +
        (time_network - time_network_base).count() / 1000.0 +
        (time_cal - time_cal_base).count() / 1000.0 +
        (time_sc - time_sc_base).count() / 1000.0 +
        (time_cscd - time_cscd_base).count() / 1000.0 +
        (time_hb - time_hb_base).count() / 1000.0 +
        (time_pdb - time_pdb_base).count() / 1000.0 +
        (time_fmap - time_fmap_base).count() / 1000.0 +
        (time_other - time_other_base).count() / 1000.0 +
        (time_water - time_water_base).count() / 1000.0 +
        (time_subgroups_cal - time_subgroups_cal_base).count() / 1000.0 +
        (time_na - time_na_base).count() / 1000.0 +
        (time_llps - time_llps_base).count() / 1000.0 +
        (time_tica - time_tica_base).count() / 1000.0 +
        (time_ifp - time_ifp_base).count() / 1000.0;
    const double unaccounted_traj_time =
        std::max(0.0, total_traj_time - accounted_traj_time);
    TitanUI::log_time(log, "Unaccounted / Orchestration",
                      unaccounted_traj_time);

    log << "-----------------------------------------------------------------"
           "--"
           "-------------\n";
    TitanUI::log_time(log, "TOTAL TRAJECTORY TIME", total_traj_time);
    std::cout << std::endl;
    //<< std::endl;
    //<< std::endl;
    if (check_end_cal(Arguments, input_line, input_name) == true) {
      break;
    }

    std::cout << std::endl;
  }

  TitanUI::print_info(
      "\033[3mFinalizing module calculations and aggregating data...\033[0m");

  if (pbe_analyzer.enabled()) {
    pbe_analyzer.finalize(log);
  }

  // --- IFP: Aggregate occupancy and export CSV ---
  if (Arguments.ifp.flag_ifp && ifp_analyzer) {
    if (Arguments.ifp.pbctopo_certified_only) {
      ifp_pbctopo_filter_report.flush();
      const size_t processed_frames = ifp_analyzer->processed_frame_count();
      std::ofstream summary(
          info_output.get_ifp_pbctopo_filter_summary_filename());
      if (!summary.is_open()) {
        TitanUI::print_error(
            "[IFP] Cannot open PBCTOPO frame-filter summary output.");
        return 1;
      }
      const auto status_count = [&](const char *status) {
        const auto found = ifp_pbctopo_status_counts.find(status);
        return found == ifp_pbctopo_status_counts.end() ? size_t{0}
                                                        : found->second;
      };
      summary
          << "criterion,total_frames,included_frames,excluded_frames,"
             "ifp_processed_frames,certified_frames,rescued_frames,"
             "weak_observation_frames,local_unwrap_frames,fallback_frames,"
             "missing_status_frames\n"
          << "status_equals_certified," << ifp_pbctopo_total_frames << ','
          << ifp_pbctopo_included_frames << ','
          << (ifp_pbctopo_total_frames - ifp_pbctopo_included_frames) << ','
          << processed_frames << ',' << status_count("certified") << ','
          << status_count("rescued") << ','
          << status_count("weak_observation") << ','
          << status_count("local_unwrap") << ','
          << status_count("fallback") << ','
          << ifp_pbctopo_missing_status_frames << '\n';
      TitanUI::print_info(
          "[IFP] PBCTOPO certified-frame filter included " +
          std::to_string(ifp_pbctopo_included_frames) + " of " +
          std::to_string(ifp_pbctopo_total_frames) + " frames.");
      if (ifp_pbctopo_missing_status_frames != 0 ||
          processed_frames != ifp_pbctopo_included_frames) {
        TitanUI::print_error(
            "[IFP] PBCTOPO frame-filter accounting failed closed.");
        return 1;
      }
    }
    ifp_analyzer->aggregate_occupancy();
    std::string ifp_outfile = Arguments.ifp.output_filename.empty()
                                  ? info_output.get_ifp_filename()
                                  : Arguments.ifp.output_filename;
    ifp_analyzer->export_csv(ifp_outfile);
    if (ifp_analyzer->has_rna_annotations()) {
      ifp_analyzer->export_rna_annotations_csv(
          info_output.get_ifp_rna_annotations_filename());
    }
  }
  if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer) {
    awh_ifp_analyzer->export_results(Arguments, info_output, pro_in.info,
                                     lig.info);
  }
  if (rg_analyzer.enabled() && rg_analyzer.has_multiple_trajectories()) {
    rg_analyzer.finalize_all(info_output.get_rg2_all_filename(),
                             info_output.get_rg2_summary_all_filename(),
                             info_output.get_rg2_hist_all_filename(), log);
  }
  if (cmap_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    cmap_analyzer.finalize_all(".", info_output, log);
  }
  if (interface_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input)) {
    try {
      interface_analyzer.finalize_all(".", info_output, Arguments, log);
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Interface analysis aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (qnative_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    qnative_analyzer.finalize_all(".", info_output, log);
  }
  if (s2_analyzer.enabled() &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    s2_analyzer.finalize_all(".", info_output, log);
  }
  if (edecomp_mm_requested(Arguments) && edecomp_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_all_results(edecomp_engine->finalize_all(),
                                              ".");
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_endpoint_all_results(
          edecomp_gbsa_engine->finalize_all("mmgbsa"),
          edecomp_gbsa_engine->finalize_pooled_frames("mmgbsa"), ".",
          "mmgbsa");
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition GBSA aggregate: ") +
                           e.what());
      return 1;
    }
  }
  if (edecomp_pbsa_requested(Arguments) && edecomp_pbsa_engine &&
      (input_line.size() > 1 || Arguments.flag_listfile ||
       Arguments.flag_pattern_input || Arguments.flag_xdr)) {
    try {
      titan_energy_decomp::export_endpoint_all_results(
          edecomp_pbsa_engine->finalize_all("mmpbsa"),
          edecomp_pbsa_engine->finalize_pooled_frames("mmpbsa"), ".",
          "mmpbsa");
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition PBSA aggregate: ") +
                           e.what());
      return 1;
    }
  }

  if (status == false && input_name == Arguments.b_input) {
    TitanUI::print_error("Error occurred during processing!");
    return 1;
  }

  TitanUI::print_module_header("Final Results Analysis & Summary");

  try {
    out(E_e, E_v, time_of_snap, cal_out, Arguments, input_name, E_e_vector,
        E_v_vector, size_snap);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] out(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] out() crashed!\n";
  }

  try {
    paths.out_to_folders(E_e, E_v, time_of_snap, E_e_vector, E_v_vector,
                         Arguments);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] paths.out_to_folders(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] paths.out_to_folders() crashed!\n";
  }

  if (Arguments.flag_cal_force == true) {
    try {
      out_force(Force_e, Force_v, time_of_snap, Arguments, input_name,
                Force_e_vector, Force_v_vector, size_snap);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] out_force(): " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] out_force() crashed!\n";
    }

    try {
      paths.out_force_to_folders(Force_e, Force_v, time_of_snap, Force_e_vector,
                                 Force_v_vector, Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] paths.out_force_to_folders(): " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] paths.out_force_to_folders() crashed!\n";
    }
  }

  if (Arguments.flag_maxforce == true) {
    try {
      stat_maxforce_res(Force_res_stat_mol1, info_output.get_group_1_name(),
                        info_output, Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] stat_maxforce_res() mol1: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] stat_maxforce_res() mol1 crashed!\n";
    }

    try {
      stat_maxforce_res(Force_res_stat_mol2, info_output.get_group_2_name(),
                        info_output, Arguments);
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] stat_maxforce_res() mol2: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] stat_maxforce_res() mol2 crashed!\n";
    }
  }

  try {
    impl_entropy_duan_zhang(paths, E_e_vector, E_v_vector, time_of_snap,
                            Arguments, E_e, E_v);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] impl_entropy_duan_zhang(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] impl_entropy_duan_zhang() crashed!\n";
  }

  try {
    print_cscd(Arguments, cscd_out_dis, cscd_out_name);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] print_cscd(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] print_cscd() crashed!\n";
  }

  try {
    paths.out_volume(time_of_snap, Volumes, Arguments);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] paths.out_volume(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] paths.out_volume() crashed!\n";
  }

  if (Arguments.flag_sub_groups == true) {
    if (sg_streaming_analyzer.active()) {
      try {
        sg_streaming_analyzer.finalize(Arguments);
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] sg_streaming_analyzer.finalize(): " << e.what()
                  << "\n";
      } catch (...) {
        std::cerr << "[ERROR] sg_streaming_analyzer.finalize() crashed!\n";
      }
    } else {
      try {
        post_process_Energy_force_subgroups(time_of_snap, EF_subgroups, paths,
                                            Arguments, info_output);
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] post_process_Energy_force_subgroups(): "
                  << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] post_process_Energy_force_subgroups() crashed!\n";
      }
    }

    if (Arguments.flag_cal_mindis == true) {
      try {
        post_proc_mindis(Groups_from_index_file, Arguments, info_output,
                         MIN_DIS);
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] post_proc_mindis(): " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] post_proc_mindis() crashed!\n";
      }
    }
  }

  try {
    stat_hbonds(Hbonds_vector, info_output, Arguments, time_of_snap,
                input_line);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] stat_hbonds(): " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[ERROR] stat_hbonds() crashed!\n";
  }

  if (Arguments.flag_h_bond == 1) {
    if (num_threads > 1) {
      executor.parallel_do(2, [&](size_t worker_id) {
        if (worker_id == 0) {
          stat_hb_res(res_hb_rec, info_output.get_res_list_hb_rec_filename());
        } else {
          stat_hb_res(res_hb_lg, info_output.get_res_list_hb_lg_filename());
        }
      });
    } else {
      try {
        stat_hb_res(res_hb_rec, info_output.get_res_list_hb_rec_filename());
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] stat_hb_res() rec: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] stat_hb_res() rec crashed!\n";
      }

      try {
        stat_hb_res(res_hb_lg, info_output.get_res_list_hb_lg_filename());
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] stat_hb_res() lg: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] stat_hb_res() lg crashed!\n";
      }
    }
  }

  // --- Post-loop analysis (extracted to engine_postproc.cpp) ---------------
  auto start_postproc = std::chrono::steady_clock::now();

  auto start_pp_gbsa = std::chrono::steady_clock::now();
  run_gbsa_postproc(Arguments, tot_snap_allfiles, ml_extractor, global_gbsa_agg,
                    cas_analyzer, na_profiler, global_has_nucleic_acid,
                    info_output, log);
  auto time_pp_gbsa = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_gbsa);

  if (Arguments.hotspot.flag_hotspot_report) {
    try {
      TitanUI::print_module_header("CAS + EDecomp Hotspot Report");
      titan_hotspot::export_report(
          ".", titan_hotspot::make_config_from_arguments(Arguments,
                                                         info_output));
      TitanUI::log_kv(log, "Hotspot Report",
                      info_output.get_hotspot_residue_ranking_filename());
      TitanUI::print_info("Hotspot report exported to " +
                          info_output.get_hotspot_residue_ranking_filename());
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Hotspot report: ") + e.what());
      return 1;
    }
  }

  auto start_pp_dssp = std::chrono::steady_clock::now();
  run_dssp_postproc(Arguments, dssp_mol1_analyzer, dssp_mol2_analyzer,
                    info_output, log);
  auto time_pp_dssp = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dssp);

  auto start_pp_tica = std::chrono::steady_clock::now();
  TitanValidation::RuntimeCache validation_runtime;
  titan_sparse_graph_module::TitanUnifiedGraphRuntime graph_runtime;
  graph_runtime.configure(Arguments);
  run_tica_msm_postproc(Arguments, dccm_analyzer, tica_analyzer, msm_analyzer,
                        executor, info_output, log, &validation_runtime);
  auto time_pp_tica = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tica);

  std::vector<double> dham_frame_weights;
  Eigen::MatrixXd dham_tpt_tpm;
  Eigen::VectorXd dham_tpt_pi;
  bool dham_tpt_converged = false;
  auto start_pp_dham = std::chrono::steady_clock::now();
  run_dham_postproc(Arguments, msm_analyzer, executor, info_output, log,
                    dham_frame_weights, &dham_tpt_tpm, &dham_tpt_pi,
                    &dham_tpt_converged, &validation_runtime.dham_context);
  if (Arguments.dham.flag_dham &&
      Arguments.eq_weight_source == "dham" && !dham_tpt_converged) {
    TitanUI::print_error(
        "DHAM was selected as the required equilibrium weight source but did "
        "not converge. No successful analysis marker should be written.");
    return 1;
  }
  auto time_pp_dham = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dham);

  std::vector<double> mbar_frame_weights;
  std::vector<double> mbar_pca_frame_weights;
  auto start_pp_mbar = std::chrono::steady_clock::now();
  const TitanMBAR::CoreResult mbar_result =
      run_mbar_postproc(Arguments, msm_analyzer, &dccm_analyzer, executor,
                        info_output, log, &mbar_frame_weights,
                        &mbar_pca_frame_weights);
  if (Arguments.mbar.flag_mbar &&
      Arguments.eq_weight_source == "mbar" &&
      (!mbar_result.solved || !mbar_result.converged ||
       mbar_result.global_frame_weights.empty())) {
    TitanUI::print_error(
        "MBAR was selected as the required equilibrium weight source but did "
        "not produce converged frame weights. No successful analysis marker "
        "should be written.");
    return 1;
  }
  auto time_pp_mbar = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_mbar);

  const PostprocEquilibriumWeights equilibrium_weights =
      resolve_postproc_equilibrium_weights(Arguments, &dham_frame_weights,
                                           &mbar_frame_weights,
                                           &mbar_pca_frame_weights, log);
  run_msm_equilibrium_postproc(Arguments, msm_analyzer, info_output, log,
                               equilibrium_weights);

  auto start_pp_tpt = std::chrono::steady_clock::now();
  run_tpt_postproc(Arguments, msm_analyzer, executor, info_output, log,
                   lj_feature_files, &rms_analyzer, &dssp_mol1_analyzer,
                   qnative_analyzer.enabled() ? &qnative_analyzer : nullptr,
                   &dham_tpt_tpm, &dham_tpt_pi,
                   dham_tpt_converged, &equilibrium_weights,
                   &validation_runtime, &graph_runtime);
  auto time_pp_tpt = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tpt);

  run_validation_postproc(Arguments, tica_analyzer, msm_analyzer, executor,
                          info_output, log, validation_runtime);

  TitanPCA::FesPostprocMetrics tica_fes_metrics;
  auto start_pp_tica_fes = std::chrono::steady_clock::now();
  run_tica_fes_postproc(Arguments, tica_analyzer, msm_analyzer, executor,
                        info_output, log, equilibrium_weights,
                        &tica_fes_metrics);
  auto time_pp_tica_fes = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_tica_fes);

  run_umap_postproc(Arguments, msm_analyzer, umap_analyzer, executor,
                    info_output, log, equilibrium_weights);

  // Drain LLPS worker queue before finalize
  auto start_pp_llps = std::chrono::steady_clock::now();
  if (Arguments.llps.flag_llps && llps_analyzer)
    llps_analyzer->flush_async();
  run_llps_postproc(Arguments, llps_analyzer, executor, info_output, log);
  auto time_pp_llps = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_llps);

  std::vector<std::string> replica_traj_files;
  if (!input_line.empty()) {
    if (Arguments.flag_xdr) {
      replica_traj_files = input_line;
    } else {
      replica_traj_files.reserve(input_line.size());
      for (const auto &stem : input_line)
        replica_traj_files.push_back(stem + ".xtc");
    }
  }

  TitanPCA::FesPostprocMetrics pca_fes_metrics;
  auto start_pp_dccm = std::chrono::steady_clock::now();
  run_dccm_pca_postproc(Arguments, tot_snap_allfiles, dccm_analyzer,
                        executor, time_network, info_output, log,
                        equilibrium_weights, replica_traj_files,
                        &pca_fes_metrics, &graph_runtime);
  auto time_pp_dccm = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_dccm);

  run_mbar_projection_exports_postproc(Arguments, msm_analyzer, &dccm_analyzer,
                                       executor, info_output, log,
                                       mbar_result);
  run_awh_exploratory_postproc(Arguments, msm_analyzer, umap_analyzer,
                               &dccm_analyzer, info_output, log);
  {
    std::vector<std::string> fes_outputs;
    if (Arguments.msm.flag_tica && msm_analyzer && msm_analyzer->n_frames() > 0 &&
        msm_analyzer->n_components() >= 2) {
      fes_outputs.push_back(info_output.get_tica_fes_2d_filename());
      fes_outputs.push_back(
          info_output.get_tica_cluster_representatives_filename());
      if (tica_fes_metrics.zts_ran)
        fes_outputs.push_back(info_output.get_tica_zts_path_filename());
    }
    if (Arguments.flag_pca) {
      fes_outputs.push_back(info_output.get_pca_fes_2d_filename());
      if (pca_fes_metrics.zts_ran)
        fes_outputs.push_back(info_output.get_pca_zts_path_filename());
    }
    if (!fes_outputs.empty()) {
      std::string joined = fes_outputs.front();
      for (size_t idx = 1; idx < fes_outputs.size(); ++idx)
        joined += "; " + fes_outputs[idx];
      TitanUI::log_kv(log, "2D FES Outputs", joined);
    }
  }

  auto start_pp_lspmd = std::chrono::steady_clock::now();
  run_lspmd_postproc(Arguments, dccm_analyzer, executor, info_output, log,
                     equilibrium_weights, &graph_runtime);
  auto time_pp_lspmd = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_lspmd);

  auto start_pp_graph = std::chrono::steady_clock::now();
  graph_runtime.export_all(Arguments, dccm_analyzer, executor, info_output, log);
  auto time_pp_graph = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_graph);

  auto start_pp_water = std::chrono::steady_clock::now();
  if (Arguments.n_wat >= 0) {
    TitanUI::print_module_header(
        "Multi-Trajectory Global Water Bridges Analysis");
    water_analyzer.export_global_summary(
        info_output.get_global_water_bridges_summary(),
        info_output.get_global_water_survival());
  }
  auto time_pp_water = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_water);

  auto start_pp_gist = std::chrono::steady_clock::now();
  if (Arguments.gist.flag_gist && (!gist_analyzer || gist_analyzer->processed_frames()==0)) {
    TitanUI::print_error("GIST processed no water frames; refusing a successful empty result.");
    return 1;
  }
  if (Arguments.gist.flag_gist && gist_analyzer) {
    TitanUI::print_module_header("GIST Water Thermodynamics");
    gist_analyzer->export_dx_maps(info_output.get_gist_prefix(),
                                  static_cast<int>(tot_snap_allfiles),
                                  Arguments.gist.gist_min_count);
  }
  auto time_pp_gist = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pp_gist);

  // --- LLPS DBSCAN export ---
  if (dbscan_llps) {
    TitanUI::print_module_header("LLPS DBSCAN Cluster Analysis");
    dbscan_llps->export_timeseries(
        info_output.get_llps_dbscan_timeseries_filename());
    TitanUI::log_section(log, "LLPS DBSCAN Output Audit");
    TitanUI::log_kv(log, info_output.get_llps_dbscan_timeseries_filename(),
                    "emitted");
    TitanUI::log_kv(log, info_output.get_llps_dbscan_droplets_filename(),
                    "emitted");
    TitanUI::log_kv(log, info_output.get_llps_droplet_shape_timeseries_filename(),
                    "emitted");
    TitanUI::log_kv(log, "LLPS Dynamics Outputs",
                    Arguments.llps.llps_dynamics ? "enabled" : "skipped");
    if (Arguments.llps.llps_dynamics) {
      dbscan_llps->finalize_dynamics(executor);
      dbscan_llps->export_dynamics(
          info_output, log,
          llps_analyzer ? &llps_analyzer->contact_dynamics() : nullptr);
    }
    if (Arguments.llps.llps_paper)
      dbscan_llps->export_paper_outputs(info_output, log);
    dbscan_llps->export_performance_report(info_output, executor);
    TitanUI::log_time(log, "LLPS DBSCAN", time_llps_dbscan.count() / 1000.0);
  }

  auto total_postproc = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_postproc);
  const auto time_pp_zts = tica_fes_metrics.zts_time + pca_fes_metrics.zts_time;

  // --- Post-processing timing report ---
  TitanUI::log_section(log, "POST-PROCESSING TIMING");
  TitanUI::log_time(log, "GBSA/CAS/ML Postproc", time_pp_gbsa.count() / 1000.0);
  TitanUI::log_time(log, "DSSP Postproc", time_pp_dssp.count() / 1000.0);
  TitanUI::log_time(log, "TICA/MSM Solve+Export",
                    time_pp_tica.count() / 1000.0);
  TitanUI::log_time(log, "DHAM Reweighting", time_pp_dham.count() / 1000.0);
  TitanUI::log_time(log, "TPT Postproc", time_pp_tpt.count() / 1000.0);
  TitanUI::log_time(log, "TICA 2D FES Export",
                    time_pp_tica_fes.count() / 1000.0);
  if (tica_fes_metrics.zts_ran || pca_fes_metrics.zts_ran) {
    TitanUI::log_time(log, "ZTS MEP", time_pp_zts.count() / 1000.0);
  }
  TitanUI::log_time(log, "LLPS Flush+Postproc", time_pp_llps.count() / 1000.0);
  TitanUI::log_time(log, "DCCM/PCA Postproc", time_pp_dccm.count() / 1000.0);
  TitanUI::log_time(log, "MBAR Postproc", time_pp_mbar.count() / 1000.0);
  TitanUI::log_time(log, "LSP-MD Postproc", time_pp_lspmd.count() / 1000.0);
  TitanUI::log_time(log, "Graph Analytics", time_pp_graph.count() / 1000.0);
  TitanUI::log_time(log, "Water Bridges Export",
                    time_pp_water.count() / 1000.0);
  TitanUI::log_time(log, "GIST Thermodynamics", time_pp_gist.count() / 1000.0);
  log << "--------------------------------------------------------------------"
         "----------\n";
  TitanUI::log_time(log, "TOTAL POST-PROCESSING",
                    total_postproc.count() / 1000.0);

  titan_uncertainty::export_uncertainty_outputs(Arguments, info_output, log);

  perf_observer.export_outputs(
      info_output.get_titan_performance_summary_filename(),
      info_output.get_titan_performance_modules_filename(),
      info_output.get_titan_performance_frames_filename(), log);

  TitanUI::log_section(log, "TITAN ENGINE TERMINATED SUCCESSFULLY");
  TitanUI::print_info(
      "\n\033[1;32mTITAN ENGINE TERMINATED SUCCESSFULLY\033[0m");
  double totalruntime = chrono::duration_cast<milliseconds>(
                            chrono::steady_clock::now() - total_run_time)
                            .count() /
                        1000.0;
  TitanUI::log_time(log, "GLOBAL TOTAL RUN TIME", totalruntime);
  TitanUI::log_message(
      log,
      "[TITAN] Historical note: TITAN is based on Universal Calculator and "
      "has used the TITAN name since 2026.");
  log << "\nContact: nguyenhoanglinh9@duytan.edu.vn\n";
  log.close();
  return 0;
}

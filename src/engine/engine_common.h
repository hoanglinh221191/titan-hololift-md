#pragma once

// Internals of engine() (src/engine/): includes, the helpers that
// impl_engine.cpp kept in an anonymous namespace (now engine_detail), and
// the types shared by the engine files. Only src/engine/*.cpp include it.

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

#include "nocal/nocal_analyses.h"
#include "postproc/engine_postproc.h"
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
#include "topology/fflib_charge_policy.h"
#include "topology/parameter_audit.h"
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

using namespace std;
using namespace chrono;

namespace engine_detail {

struct PullDirectSnapshotSelection {
  std::string source_file{};
  std::string input_mode{};
  std::string selection_rule{};
  double time_ps{0.0};
  std::size_t frame_index{0};
};

// Per-chunk centre of geometry of each frame (water shift).
struct shift_vector {
  double x, y, z;
};

void keep_only_selected_frame(Multi_MOL &mol, std::size_t frame_index);

std::string format_time_ps(double time_ps);

std::string progress_source_label(const arguments &Arguments,
                                  const std::string &name_pro);

std::string pull_direct_selection_rule(const arguments &Arguments);

void set_pull_snapshot_metadata(const PullDirectSnapshotSelection &selection,
                                de_arg &DE_arguments);

std::expected<PullDirectSnapshotSelection, std::string>
prepare_pull_direct_snapshot(arguments &Arguments, de_arg &DE_arguments,
                             const std::string &force_f,
                             const std::string &name_pro,
                             const std::string &name_lig, Multi_MOL &pro_in,
                             Multi_MOL &lig, Multi_MOL &water_traj,
                             groups_cal &Groups_from_index_file,
                             const amino_acid &amine, info &output_info,
                             TitanExecutor &executor, std::size_t num_threads,
                             std::fstream &log);

double determinant3x3(const BoxMatrix &box_matrix);

double frame_box_volume_nm3(const box &frame_box);

bool is_generic_autotype_candidate(const std::string &mol_type);

void append_module_label(std::string &modules, std::string_view label);

bool feature_source_is_observable_tica(const arguments &Arguments);

bool feature_source_needs_dccm(const arguments &Arguments,
                               bool legacy_tica_mode);

void add_dccm_feature_columns(std::vector<FeatureColumn> &columns, int n_ca);

bool append_dccm_pairwise_features(std::vector<double> &values,
                                   DCCMNetwork &dccm_analyzer, int n_ca,
                                   Eigen::MatrixXd &mobile,
                                   Eigen::MatrixXd &ref_centered,
                                   Eigen::Matrix3d &rotation,
                                   bool &ref_initialized);

std::vector<FeatureColumn> build_observable_feature_columns(
    const arguments &Arguments, int n_ca, const TitanQNativeAnalyzer &qnative,
    const TitanDihedralAnalyzer &dihedral, const TitanContactMapAnalyzer &cmap,
    const TitanLJAnalyzer &lj_analyzer, bool tica_use_lj_features);

bool build_observable_feature_vector(
    const arguments &Arguments, DCCMNetwork &dccm_analyzer, int n_ca,
    Eigen::MatrixXd &mobile, Eigen::MatrixXd &ref_centered,
    Eigen::Matrix3d &rotation, bool &ref_initialized,
    const TitanQNativeAnalyzer &qnative, const TitanDihedralAnalyzer &dihedral,
    const TitanContactMapAnalyzer &cmap, const TitanLJAnalyzer &lj_analyzer,
    Eigen::VectorXd &out_values);

std::string format_binary_bytes(long double bytes);

bool should_enable_dense_pair_cache(const arguments &Arguments,
                                    size_t mol1_atoms, size_t mol2_atoms,
                                    size_t available_ram,
                                    std::string &reason);

bool edecomp_mm_requested(const arguments &Arguments);

bool edecomp_gbsa_requested(const arguments &Arguments);

bool edecomp_pbsa_requested(const arguments &Arguments);

titan_energy_decomp::RankingMetric
edecomp_rank_metric(const std::string &rank_by);

titan_energy_decomp::EngineConfig make_edecomp_config(
    const arguments &Arguments);

std::string edecomp_output_suffix(const arguments &Arguments,
                                  const std::vector<std::string> &input_line,
                                  int input_name);

titan_energy_decomp::OutputUnit edecomp_output_unit(const arguments &Arguments);

std::optional<std::size_t>
find_edecomp_entity_index(const titan_energy_decomp::EntityAxis &axis,
                          const std::string &chain, int resnr);

std::optional<std::size_t>
find_edecomp_atom_entity_index(const titan_energy_decomp::EntityAxis &axis,
                               std::size_t one_based_atom_index,
                               const std::string &chain, int resnr,
                               const std::string &atom_name);

std::vector<titan_energy_decomp::EndpointEntityRow>
make_gbsa_endpoint_rows(const GBSA_Aggregator &frame_decomp,
                        const titan_energy_decomp::EndpointEntityDecompositionEngine
                            &engine);

std::vector<titan_energy_decomp::EndpointEntityRow> make_pbsa_endpoint_rows(
    const std::vector<TitanPBE::EDecompRow> &frame_rows,
    const titan_energy_decomp::EndpointEntityDecompositionEngine &engine);

void auto_adjust_polymer_type(arguments &Arguments, Multi_MOL &mol,
                              std::fstream &log, bool is_mol1);

} // namespace engine_detail

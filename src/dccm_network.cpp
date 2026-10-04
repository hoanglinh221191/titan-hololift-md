#include "dccm_network.h"
#include "algebra/matrix3_buffer.h"
#include "algebra/symmetric_eigen.h"
#include "titan_alignment.h"
#include "titan_corr_knn_shared.h"
#include "titan_executor.h"
#include "titan_graph_community.h"
#include "memory_utils.h"
#include "titan_neighbor_grid.h"
#include "titan_simd_dispatch.h"
#include "titan_sparse_graph.h"
#include "titan_ui.h"
#include "titan_uncertainty.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_set>

namespace {

size_t coord_count(const MOL &mol) noexcept {
  return std::min({mol.pos.x.size(), mol.pos.y.size(), mol.pos.z.size()});
}

template <typename T> struct FlatAlignedMatrixT {
  static constexpr size_t kSimdLaneWidth =
      std::max<size_t>(1, kTitanSimdAlignment / sizeof(T));

  FlatAlignedMatrixT() = default;
  FlatAlignedMatrixT(size_t rows, size_t cols, T value = T{})
      : rows_(rows), cols_(cols),
        stride_(cols == 0 ? 0
                          : ((cols + kSimdLaneWidth - 1) / kSimdLaneWidth) *
                                kSimdLaneWidth),
        data_(rows * stride_, value) {}

  void assign(size_t rows, size_t cols, T value = T{}) {
    rows_ = rows;
    cols_ = cols;
    stride_ = cols == 0 ? 0
                        : ((cols + kSimdLaneWidth - 1) / kSimdLaneWidth) *
                              kSimdLaneWidth;
    data_.assign(rows * stride_, value);
  }

  T &operator()(size_t row, size_t col) { return data_[row * stride_ + col]; }

  const T &operator()(size_t row, size_t col) const {
    return data_[row * stride_ + col];
  }

  T *row_data(size_t row) { return data_.data() + row * stride_; }
  const T *row_data(size_t row) const { return data_.data() + row * stride_; }

  size_t rows() const noexcept { return rows_; }
  size_t cols() const noexcept { return cols_; }
  size_t stride() const noexcept { return stride_; }

  void fill(T value) { std::fill(data_.begin(), data_.end(), value); }

private:
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t stride_ = 0;
  AlignedVector<T> data_;
};

using FlatAlignedMatrix = FlatAlignedMatrixT<double>;

using titan::algebra::Mat3;
using titan::algebra::Matrix3Buffer;
using titan::algebra::Vec3;

Vec3 centroid_of(const Matrix3Buffer &points) {
  Vec3 centroid{};
  for (size_t row = 0; row < points.rows(); ++row)
    centroid += {points(row, 0), points(row, 1), points(row, 2)};
  if (points.rows() != 0)
    centroid = centroid / static_cast<double>(points.rows());
  return centroid;
}

void subtract_centroid(Matrix3Buffer &points, Vec3 centroid) {
  for (size_t row = 0; row < points.rows(); ++row) {
    points(row, 0) -= centroid.x;
    points(row, 1) -= centroid.y;
    points(row, 2) -= centroid.z;
  }
}

void fill_centered_reference(Matrix3Buffer &ref,
                             const DCCMNetwork::TrajectoryRows &traj_x,
                             const DCCMNetwork::TrajectoryRows &traj_y,
                             const DCCMNetwork::TrajectoryRows &traj_z,
                             size_t num_ca) {
  ref.resize(num_ca);
  for (size_t a = 0; a < num_ca; ++a) {
    ref(a, 0) = traj_x[0][a];
    ref(a, 1) = traj_y[0][a];
    ref(a, 2) = traj_z[0][a];
  }
  subtract_centroid(ref, centroid_of(ref));
}

Vec3 fill_mobile_frame(Matrix3Buffer &mobile,
                       const DCCMNetwork::TrajectoryRows &traj_x,
                       const DCCMNetwork::TrajectoryRows &traj_y,
                       const DCCMNetwork::TrajectoryRows &traj_z, size_t frame,
                       size_t num_ca) {
  mobile.resize(num_ca);
  for (size_t a = 0; a < num_ca; ++a) {
    mobile(a, 0) = traj_x[frame][a];
    mobile(a, 1) = traj_y[frame][a];
    mobile(a, 2) = traj_z[frame][a];
  }
  return centroid_of(mobile);
}

inline int resolve_allostery_bin_count(size_t total_frames, int num_bins) {
  if (num_bins > 0)
    return num_bins;
  return std::max(8, static_cast<int>(std::cbrt(total_frames / 5.0)));
}

inline std::vector<std::string>
build_allostery_node_labels(const std::vector<std::string> &ca_resname,
                            const std::vector<int> &ca_resnrs,
                            const std::vector<std::string> &ca_chain) {
  const size_t num_ca = ca_resname.size();
  std::vector<std::string> node_labels(num_ca);
  for (size_t i = 0; i < num_ca; ++i)
    node_labels[i] =
        ca_resname[i] + std::to_string(ca_resnrs[i]) + "_" + ca_chain[i];
  return node_labels;
}

std::string build_network_node_id(const std::vector<std::string> &ca_resname,
                                  const std::vector<int> &ca_resnrs,
                                  const std::vector<std::string> &ca_chain,
                                  size_t index) {
  return ca_resname[index] + std::to_string(ca_resnrs[index]) + "_" +
         ca_chain[index];
}

std::string format_gib(long double bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L)) << " GiB";
  return oss.str();
}

long double resolve_memory_budget_bytes(
    const DCCMNetwork::NetworkMemoryConfig &cfg) {
  if (cfg.mem_budget_gb > 0.0) {
    return static_cast<long double>(cfg.mem_budget_gb) * 1024.0L * 1024.0L *
           1024.0L;
  }

  const size_t available = get_available_ram();
  const size_t total = get_total_ram();
  const long double fraction = cfg.low_memory ? 0.45L : 0.60L;
  if (available > 0)
    return static_cast<long double>(available) * fraction;
  if (total > 0)
    return static_cast<long double>(total) * (cfg.low_memory ? 0.25L : 0.35L);
  return 2.0L * 1024.0L * 1024.0L * 1024.0L;
}

void append_memory_report(const info &info_output, const std::string &module,
                          const std::string &mode, size_t nodes,
                          size_t frames, long double estimated_bytes,
                          long double budget_bytes, long double pairs,
                          const std::string &reason) {
  const std::string filename = info_output.get_memory_report_filename();
  const bool new_file = !static_cast<bool>(std::ifstream(filename));
  std::ofstream out(filename, std::ios::app);
  if (!out.is_open())
    return;
  if (new_file) {
    out << "Module,Mode,Nodes,Frames,EstimatedExtraGiB,BudgetGiB,Pairs,Reason\n";
  }
  const long double gib = 1024.0L * 1024.0L * 1024.0L;
  out << module << "," << mode << "," << nodes << "," << frames << ","
      << std::fixed << std::setprecision(3)
      << static_cast<double>(estimated_bytes / gib) << ","
      << static_cast<double>(budget_bytes / gib) << ","
      << static_cast<unsigned long long>(pairs) << "," << reason << "\n";
}

std::pair<size_t, size_t> upper_triangle_pair_from_index(size_t linear,
                                                        size_t n) {
  long double lo = 0.0L;
  long double hi = static_cast<long double>(n);
  const long double target = static_cast<long double>(linear);
  while (hi - lo > 1.0L) {
    const long double mid = std::floor((lo + hi) * 0.5L);
    const long double before = mid * static_cast<long double>(n) -
                               (mid * (mid - 1.0L)) * 0.5L;
    if (before <= target)
      lo = mid;
    else
      hi = mid;
  }
  size_t i = static_cast<size_t>(lo);
  size_t before_i = i * n - (i * (i - 1)) / 2;
  while (i + 1 < n && before_i + (n - i) <= linear) {
    before_i += n - i;
    ++i;
  }
  while (i > 0 && before_i > linear) {
    --i;
    before_i -= n - i;
  }
  return {i, i + (linear - before_i)};
}

bool should_skip_dense_dccm_postproc(size_t num_nodes, size_t num_frames,
                                     const DCCMNetwork::NetworkMemoryConfig &cfg,
                                     long double &estimated_extra_bytes,
                                     long double &budget_bytes,
                                     long double &pair_count) {
  const size_t frame_stride =
      ((num_frames + FlatAlignedMatrix::kSimdLaneWidth - 1) /
       FlatAlignedMatrix::kSimdLaneWidth) *
      FlatAlignedMatrix::kSimdLaneWidth;
  const size_t node_stride =
      ((num_nodes + FlatAlignedMatrix::kSimdLaneWidth - 1) /
       FlatAlignedMatrix::kSimdLaneWidth) *
      FlatAlignedMatrix::kSimdLaneWidth;
  pair_count = static_cast<long double>(num_nodes) *
               static_cast<long double>(num_nodes + 1) * 0.5L;
  const long double dev_bytes =
      3.0L * static_cast<long double>(num_nodes) *
      static_cast<long double>(frame_stride) * sizeof(double);
  const long double cov_bytes = static_cast<long double>(num_nodes) *
                                static_cast<long double>(node_stride) *
                                sizeof(double);
  constexpr long double safety_bytes =
      512.0L * 1024.0L * 1024.0L; // Alignment scratch, CSV buffers, graph edges.
  estimated_extra_bytes = dev_bytes + cov_bytes + safety_bytes;

  budget_bytes = resolve_memory_budget_bytes(cfg);

  constexpr long double max_dense_pairs = 50'000'000.0L;
  return estimated_extra_bytes > budget_bytes || pair_count > max_dense_pairs;
}

void write_header_only_network_outputs(const std::vector<std::string> &ca_resname,
                                       const std::vector<int> &ca_resnrs,
                                       const std::vector<std::string> &ca_chain,
                                       const std::vector<int> &ca_is_lig) {
  {
    std::ofstream edges_out("network_edges.csv");
    edges_out << "Source,Target,Correlation\n";
  }
  {
    std::ofstream nodes_out("network_nodes.csv");
    nodes_out << "Id,Label,Type,AtomName\n";
    for (size_t i = 0; i < ca_resname.size(); ++i) {
      nodes_out << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i)
                << "," << ca_resnrs[i] << ","
                << (ca_is_lig[i] ? "Ligand" : "Receptor") << ","
                << ca_resname[i] << "\n";
    }
  }
}

struct ResidueKey {
  int is_lig{0};
  int resnr{0};
  std::string chain;

  bool operator<(const ResidueKey &other) const noexcept {
    if (is_lig != other.is_lig)
      return is_lig < other.is_lig;
    if (resnr != other.resnr)
      return resnr < other.resnr;
    return chain < other.chain;
  }
};

void write_empty_community_outputs(const info &info_output) {
  {
    std::ofstream communities_out(info_output.get_network_communities_filename());
    communities_out
        << "Id,ResName,ResNr,Chain,IsLig,CommunityId,CommunitySize,"
           "IntraCommunityStrength\n";
  }
  {
    std::ofstream summary_out(
        info_output.get_network_community_summary_filename());
    summary_out << "# Louvain partition on the thresholded DCCM graph using "
                   "abs(correlation) edge weights.\n";
    summary_out << "CommunityId,Size,InternalWeight,TotalIncidentWeight,"
                   "ModularityContribution\n";
  }
  {
    std::ofstream supergraph_out(info_output.get_network_supergraph_filename());
    supergraph_out << "SourceCommunity,TargetCommunity,EdgeWeight\n";
  }
}

std::vector<double> compute_intra_community_strengths(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const std::vector<int> &membership) {
  std::vector<double> strengths(membership.size(), 0.0);
  for (int node = 0; node < graph.node_count(); ++node) {
    const int community = membership[static_cast<size_t>(node)];
    double total = 0.0;
    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      const int neighbor = graph.col_index(edge_idx);
      if (membership[static_cast<size_t>(neighbor)] == community) {
        total += graph.value(edge_idx);
      }
    }
    strengths[static_cast<size_t>(node)] = total;
  }
  return strengths;
}

void export_community_outputs(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const titan_graph_community::LouvainResult &result,
    const std::vector<std::string> &ca_resname, const std::vector<int> &ca_resnrs,
    const std::vector<std::string> &ca_chain, const std::vector<int> &ca_is_lig,
    const info &info_output) {
  const auto intra_strengths =
      compute_intra_community_strengths(graph, result.membership);

  {
    std::ofstream communities_out(info_output.get_network_communities_filename());
    communities_out << "Id,ResName,ResNr,Chain,IsLig,CommunityId,CommunitySize,"
                       "IntraCommunityStrength\n";
    communities_out << std::fixed << std::setprecision(6);
    for (size_t node = 0; node < result.membership.size(); ++node) {
      const int community = result.membership[node];
      const int exported_community = community + 1;
      communities_out << build_network_node_id(ca_resname, ca_resnrs, ca_chain,
                                               node)
                      << "," << ca_resname[node] << "," << ca_resnrs[node] << ","
                      << ca_chain[node] << "," << ca_is_lig[node] << ","
                      << exported_community << ","
                      << result.stats.community_sizes[static_cast<size_t>(
                             community)]
                      << "," << intra_strengths[node] << "\n";
    }
  }

  {
    std::ofstream summary_out(
        info_output.get_network_community_summary_filename());
    summary_out << "# Louvain partition on the thresholded DCCM graph using "
                   "abs(correlation) edge weights.\n";
    summary_out << "CommunityId,Size,InternalWeight,TotalIncidentWeight,"
                   "ModularityContribution\n";
    summary_out << std::fixed << std::setprecision(6);
    for (size_t community = 0; community < result.stats.community_sizes.size();
         ++community) {
      summary_out << (community + 1) << ","
                  << result.stats.community_sizes[community] << ","
                  << result.stats.internal_weights[community] << ","
                  << result.stats.total_incident_weights[community] << ","
                  << result.stats.modularity_contributions[community] << "\n";
    }
  }

  {
    std::map<std::pair<int, int>, double> super_edges;
    for (int node = 0; node < graph.node_count(); ++node) {
      const int source_community = result.membership[static_cast<size_t>(node)];
      for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
           ++edge_idx) {
        const int neighbor = graph.col_index(edge_idx);
        if (node >= neighbor)
          continue;
        const int target_community =
            result.membership[static_cast<size_t>(neighbor)];
        if (source_community == target_community)
          continue;
        const auto key = std::make_pair(std::min(source_community, target_community),
                                        std::max(source_community, target_community));
        super_edges[key] += graph.value(edge_idx);
      }
    }

    std::ofstream supergraph_out(info_output.get_network_supergraph_filename());
    supergraph_out << "SourceCommunity,TargetCommunity,EdgeWeight\n";
    supergraph_out << std::fixed << std::setprecision(6);
    for (const auto &[edge, weight] : super_edges) {
      supergraph_out << (edge.first + 1) << "," << (edge.second + 1) << ","
                     << weight << "\n";
    }
  }
}

std::vector<double> compute_participation_coefficients(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const std::vector<int> &membership) {
  std::vector<double> coefficients(membership.size(), 0.0);
  if (membership.empty())
    return coefficients;

  const int community_count =
      *std::max_element(membership.begin(), membership.end()) + 1;
  std::vector<double> weights_by_community(static_cast<size_t>(community_count),
                                           0.0);
  std::vector<int> marks(static_cast<size_t>(community_count), -1);
  std::vector<int> touched;
  touched.reserve(32);
  int mark = 0;

  for (int node = 0; node < graph.node_count(); ++node) {
    touched.clear();
    const int current_mark = mark++;
    double total_degree = 0.0;
    for (int edge_idx = graph.row_begin(node); edge_idx < graph.row_end(node);
         ++edge_idx) {
      const int community =
          membership[static_cast<size_t>(graph.col_index(edge_idx))];
      if (marks[static_cast<size_t>(community)] != current_mark) {
        marks[static_cast<size_t>(community)] = current_mark;
        weights_by_community[static_cast<size_t>(community)] = 0.0;
        touched.push_back(community);
      }
      const double weight = graph.value(edge_idx);
      weights_by_community[static_cast<size_t>(community)] += weight;
      total_degree += weight;
    }

    if (total_degree <= 1e-12)
      continue;

    double sum_sq = 0.0;
    for (int community : touched) {
      const double fraction =
          weights_by_community[static_cast<size_t>(community)] / total_degree;
      sum_sq += fraction * fraction;
    }
    coefficients[static_cast<size_t>(node)] = std::max(0.0, 1.0 - sum_sq);
  }

  return coefficients;
}

struct SnapshotBFactors {
  std::vector<double> receptor;
  std::vector<double> ligand;
};

SnapshotBFactors map_node_values_to_snapshot_atoms(
    const std::vector<double> &node_values, const std::vector<int> &ca_indices,
    const std::vector<int> &ca_resnrs, const std::vector<int> &ca_is_lig,
    const std::vector<std::string> &ca_chain,
    const std::vector<mol_info> &receptor_info,
    const std::vector<mol_info> &ligand_info) {
  SnapshotBFactors mapped;
  mapped.receptor.assign(receptor_info.size(), 0.0);
  mapped.ligand.assign(ligand_info.size(), 0.0);
  std::vector<int> receptor_exact(receptor_info.size(), -1);
  std::vector<int> ligand_exact(ligand_info.size(), -1);
  std::map<ResidueKey, std::vector<int>> residue_nodes;

  for (size_t node = 0; node < node_values.size(); ++node) {
    const ResidueKey key{ca_is_lig[node], ca_resnrs[node], ca_chain[node]};
    residue_nodes[key].push_back(static_cast<int>(node));
    const int atom_index = ca_indices[node];
    if (ca_is_lig[node] == 0) {
      if (atom_index >= 0 &&
          static_cast<size_t>(atom_index) < receptor_exact.size()) {
        receptor_exact[static_cast<size_t>(atom_index)] = static_cast<int>(node);
      }
    } else if (atom_index >= 0 &&
               static_cast<size_t>(atom_index) < ligand_exact.size()) {
      ligand_exact[static_cast<size_t>(atom_index)] = static_cast<int>(node);
    }
  }

  auto map_atoms = [&](const std::vector<mol_info> &atom_info, int is_lig,
                       const std::vector<int> &exact_map,
                       std::vector<double> &bfactors) {
    for (size_t atom = 0; atom < atom_info.size(); ++atom) {
      const int exact_node = exact_map[atom];
      if (exact_node >= 0) {
        bfactors[atom] = node_values[static_cast<size_t>(exact_node)];
        continue;
      }
      const ResidueKey key{is_lig, static_cast<int>(atom_info[atom].get_resnr()),
                           atom_info[atom].get_chain()};
      const auto it = residue_nodes.find(key);
      if (it != residue_nodes.end() && it->second.size() == 1) {
        bfactors[atom] = node_values[static_cast<size_t>(it->second.front())];
      }
    }
  };

  map_atoms(receptor_info, 0, receptor_exact, mapped.receptor);
  map_atoms(ligand_info, 1, ligand_exact, mapped.ligand);
  return mapped;
}

void write_bfactor_pdb(const std::string &filename, const std::string &remark,
                       const MOL &receptor_frame,
                       const std::vector<mol_info> &receptor_info,
                       const std::vector<double> &receptor_bfactors,
                       const MOL &ligand_frame,
                       const std::vector<mol_info> &ligand_info,
                       const std::vector<double> &ligand_bfactors) {
  std::ofstream out_pdb(filename);
  if (!out_pdb.is_open()) {
    TitanUI::print_warning("Community: failed to open " + filename +
                           " for PDB export.");
    return;
  }

  out_pdb << std::fixed << std::setprecision(3);
  out_pdb << "REMARK    GENERATED BY TITAN ENGINE\n";
  out_pdb << "REMARK    " << remark << "\n";
  out_pdb << "TITLE     first analyzed snapshot t= " << std::setw(9)
          << receptor_frame.time_of_snap << "\n";
  out_pdb << "CRYST1" << std::setw(9) << receptor_frame.Box.get_a()
          << std::setw(9) << receptor_frame.Box.get_b() << std::setw(9)
          << receptor_frame.Box.get_c() << std::setw(7)
          << receptor_frame.Box.get_alpha() << std::setw(7)
          << receptor_frame.Box.get_beta() << std::setw(7)
          << receptor_frame.Box.get_gamma() << "\n";
  out_pdb << "MODEL        1\n";

  auto write_atoms = [&](const MOL &frame, const std::vector<mol_info> &atom_info,
                         const std::vector<double> &bfactors,
                         size_t serial_offset) {
    char buffer[160];
    for (size_t atom = 0; atom < frame.size() && atom < atom_info.size(); ++atom) {
      const int serial =
          static_cast<int>(std::min(serial_offset + atom + 1, size_t{99999}));
      const char chain_id =
          atom_info[atom].get_chain().empty() ? ' ' : atom_info[atom].get_chain()[0];
      const double bfactor = atom < bfactors.size() ? bfactors[atom] : 0.0;
      std::snprintf(buffer, sizeof(buffer),
                    "ATOM  %5d %-4.4s %-4.4s%c%4zu    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                    serial, atom_info[atom].get_atom().c_str(),
                    atom_info[atom].get_original_res_name().c_str(), chain_id,
                    atom_info[atom].get_resnr(), frame.pos.x[atom], frame.pos.y[atom],
                    frame.pos.z[atom], 1.00, bfactor);
      out_pdb << buffer;
    }
  };

  write_atoms(receptor_frame, receptor_info, receptor_bfactors, 0);
  out_pdb << "TER\n";
  write_atoms(ligand_frame, ligand_info, ligand_bfactors, receptor_frame.size());
  if (!ligand_info.empty())
    out_pdb << "TER\n";
  out_pdb << "ENDMDL\nEND\n";
}

void export_community_pdbs(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const titan_graph_community::LouvainResult &result,
    const std::vector<int> &ca_indices, const std::vector<int> &ca_resnrs,
    const std::vector<int> &ca_is_lig, const std::vector<std::string> &ca_chain,
    const MOL &first_receptor_frame, const std::vector<mol_info> &first_receptor_info,
    const MOL &first_ligand_frame, const std::vector<mol_info> &first_ligand_info,
    const info &info_output) {
  std::vector<double> community_ids(result.membership.size(), 0.0);
  for (size_t node = 0; node < result.membership.size(); ++node) {
    community_ids[node] = static_cast<double>(result.membership[node] + 1);
  }
  const auto participation =
      compute_participation_coefficients(graph, result.membership);

  const auto community_bfactors = map_node_values_to_snapshot_atoms(
      community_ids, ca_indices, ca_resnrs, ca_is_lig, ca_chain,
      first_receptor_info, first_ligand_info);
  const auto participation_bfactors = map_node_values_to_snapshot_atoms(
      participation, ca_indices, ca_resnrs, ca_is_lig, ca_chain,
      first_receptor_info, first_ligand_info);

  write_bfactor_pdb(info_output.get_network_community_id_pdb_filename(),
                    "B-factor stores one-based Louvain community id.",
                    first_receptor_frame, first_receptor_info,
                    community_bfactors.receptor, first_ligand_frame,
                    first_ligand_info, community_bfactors.ligand);
  write_bfactor_pdb(
      info_output.get_network_participation_pdb_filename(),
      "B-factor stores participation coefficient from the thresholded DCCM graph.",
      first_receptor_frame, first_receptor_info,
      participation_bfactors.receptor, first_ligand_frame, first_ligand_info,
      participation_bfactors.ligand);
}

} // namespace

void DCCMNetwork::export_community_outputs_from_graph(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    const std::vector<int> &membership, const info &info_output) {
  last_community_result_ = CommunityResult{};
  if (graph.empty() || membership.size() != static_cast<std::size_t>(graph.node_count())) {
    write_empty_community_outputs(info_output);
    return;
  }

  titan_graph_community::LouvainResult result;
  result.membership = membership;
  result.stats = titan_graph_community::summarize_partition(graph, membership);
  last_community_result_.membership = result.membership;
  last_community_result_.community_sizes = result.stats.community_sizes;
  last_community_result_.modularity = result.stats.modularity;
  last_community_result_.has_result = true;

  export_community_outputs(graph, result, ca_resname, ca_resnrs, ca_chain,
                           ca_is_lig, info_output);
  if (has_first_snapshot_frame_) {
    export_community_pdbs(graph, result, ca_indices, ca_resnrs, ca_is_lig,
                          ca_chain, first_pro_frame_, first_pro_info_,
                          first_lig_frame_, first_lig_info_, info_output);
  } else {
    TitanUI::print_warning("Community: first snapshot was not cached, "
                           "skipping B-factor PDB export.");
  }
}

namespace {
inline size_t upper_triangle_pair_index(size_t i, size_t j,
                                        size_t node_count) noexcept {
  return i * (2 * node_count - i - 1) / 2 + (j - i - 1);
}

std::vector<std::pair<size_t, size_t>>
build_dense_upper_triangle_pairs(size_t node_count) {
  std::vector<std::pair<size_t, size_t>> pairs;
  if (node_count < 2)
    return pairs;

  pairs.reserve(node_count * (node_count - 1) / 2);
  for (size_t i = 0; i < node_count; ++i) {
    for (size_t j = i + 1; j < node_count; ++j)
      pairs.emplace_back(i, j);
  }
  return pairs;
}

std::vector<std::pair<size_t, size_t>> build_lspmd_candidate_pairs(
    const DCCMNetwork::TrajectoryRows &traj_x,
    const DCCMNetwork::TrajectoryRows &traj_y,
    const DCCMNetwork::TrajectoryRows &traj_z, size_t num_ca,
    size_t total_frames, double cutoff, bool use_avg_cutoff) {
  std::vector<std::pair<size_t, size_t>> pairs;
  if (num_ca < 2 || total_frames == 0 || cutoff <= 0.0)
    return pairs;

  const size_t dense_pair_count = num_ca * (num_ca - 1) / 2;
  const size_t frame_limit = use_avg_cutoff ? total_frames : 1;
  std::vector<unsigned char> seen(dense_pair_count, 0);
  titan_neighbor::HalfShellGrid neighbor_grid;
  pairs.reserve(std::min(dense_pair_count, num_ca * static_cast<size_t>(32)));

  for (size_t f = 0; f < frame_limit; ++f) {
    if (traj_x[f].size() < num_ca || traj_y[f].size() < num_ca ||
        traj_z[f].size() < num_ca) {
      return build_dense_upper_triangle_pairs(num_ca);
    }

    const auto &x_row = traj_x[f];
    const auto &y_row = traj_y[f];
    const auto &z_row = traj_z[f];
    const auto [xmin, xmax] =
        std::minmax_element(x_row.begin(), x_row.begin() + num_ca);
    const auto [ymin, ymax] =
        std::minmax_element(y_row.begin(), y_row.begin() + num_ca);
    const auto [zmin, zmax] =
        std::minmax_element(z_row.begin(), z_row.begin() + num_ca);

    titan_neighbor::OrthorhombicGridSpec spec;
    spec.cutoff = cutoff;
    spec.shift_x = -(*xmin) + cutoff;
    spec.shift_y = -(*ymin) + cutoff;
    spec.shift_z = -(*zmin) + cutoff;
    spec.len_x = (*xmax - *xmin) + 2.0 * cutoff;
    spec.len_y = (*ymax - *ymin) + 2.0 * cutoff;
    spec.len_z = (*zmax - *zmin) + 2.0 * cutoff;
    spec.periodic = false;

    neighbor_grid.for_each_pair(
        num_ca, spec,
        [&](std::size_t atom_index, double &x, double &y, double &z) {
          x = x_row[atom_index];
          y = y_row[atom_index];
          z = z_row[atom_index];
        },
        [&](std::size_t atom_index) { return atom_index < num_ca; },
        [&](std::size_t lhs, std::size_t rhs) {
          const double dx = x_row[lhs] - x_row[rhs];
          const double dy = y_row[lhs] - y_row[rhs];
          const double dz = z_row[lhs] - z_row[rhs];
          if (dx * dx + dy * dy + dz * dz > cutoff * cutoff)
            return;

          const size_t i = std::min(lhs, rhs);
          const size_t j = std::max(lhs, rhs);
          const size_t pair_index = upper_triangle_pair_index(i, j, num_ca);
          if (seen[pair_index] != 0)
            return;
          seen[pair_index] = 1;
          pairs.emplace_back(i, j);
        });
  }

  return pairs;
}

std::uint64_t pack_sparse_pair_key(size_t i, size_t j) noexcept {
  const std::uint64_t a = static_cast<std::uint64_t>(std::min(i, j));
  const std::uint64_t b = static_cast<std::uint64_t>(std::max(i, j));
  return (a << 32) | b;
}

std::pair<size_t, size_t> unpack_sparse_pair_key(std::uint64_t key) noexcept {
  return {static_cast<size_t>(key >> 32),
          static_cast<size_t>(key & 0xffffffffULL)};
}

std::vector<std::pair<size_t, size_t>> build_sparse_spatial_candidate_pairs(
    const DCCMNetwork::TrajectoryRows &traj_x,
    const DCCMNetwork::TrajectoryRows &traj_y,
    const DCCMNetwork::TrajectoryRows &traj_z, size_t num_ca,
    size_t total_frames, double cutoff_ang, int top_k) {
  struct Neighbor {
    float dist2{0.0f};
    size_t partner{0};
  };

  std::vector<std::pair<size_t, size_t>> pairs;
  if (num_ca < 2 || total_frames == 0 || cutoff_ang <= 0.0 ||
      traj_x.size() < total_frames || traj_y.size() < total_frames ||
      traj_z.size() < total_frames) {
    return pairs;
  }
  if (num_ca > static_cast<size_t>(std::numeric_limits<std::uint32_t>::max())) {
    TitanUI::print_warning(
        "Sparse network pair key limit exceeded; outputs contain headers only.");
    return pairs;
  }

  std::vector<double> x_mean(num_ca, 0.0), y_mean(num_ca, 0.0),
      z_mean(num_ca, 0.0);
  for (size_t f = 0; f < total_frames; ++f) {
    if (traj_x[f].size() < num_ca || traj_y[f].size() < num_ca ||
        traj_z[f].size() < num_ca) {
      return pairs;
    }
    for (size_t a = 0; a < num_ca; ++a) {
      x_mean[a] += traj_x[f][a];
      y_mean[a] += traj_y[f][a];
      z_mean[a] += traj_z[f][a];
    }
  }
  const double inv_frames = 1.0 / static_cast<double>(total_frames);
  for (size_t a = 0; a < num_ca; ++a) {
    x_mean[a] *= inv_frames;
    y_mean[a] *= inv_frames;
    z_mean[a] *= inv_frames;
  }
  const auto [xmin, xmax] =
      std::minmax_element(x_mean.begin(), x_mean.end());
  const auto [ymin, ymax] =
      std::minmax_element(y_mean.begin(), y_mean.end());
  const auto [zmin, zmax] =
      std::minmax_element(z_mean.begin(), z_mean.end());

  titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff = cutoff_ang;
  spec.shift_x = -(*xmin) + cutoff_ang;
  spec.shift_y = -(*ymin) + cutoff_ang;
  spec.shift_z = -(*zmin) + cutoff_ang;
  spec.len_x = (*xmax - *xmin) + 2.0 * cutoff_ang;
  spec.len_y = (*ymax - *ymin) + 2.0 * cutoff_ang;
  spec.len_z = (*zmax - *zmin) + 2.0 * cutoff_ang;
  spec.periodic = false;

  const double cutoff2 = cutoff_ang * cutoff_ang;
  std::vector<std::uint64_t> keys;
  keys.reserve(num_ca * static_cast<size_t>(std::max(4, top_k)));
  std::vector<std::vector<Neighbor>> ranked_neighbors;
  const bool use_knn = top_k > 0;
  if (use_knn)
    ranked_neighbors.resize(num_ca);

  titan_neighbor::HalfShellGrid neighbor_grid;
  neighbor_grid.for_each_pair(
      num_ca, spec,
      [&](std::size_t atom_index, double &x, double &y, double &z) {
        x = x_mean[atom_index];
        y = y_mean[atom_index];
        z = z_mean[atom_index];
      },
      [&](std::size_t atom_index) { return atom_index < num_ca; },
      [&](std::size_t lhs, std::size_t rhs) {
        const double dx = x_mean[lhs] - x_mean[rhs];
        const double dy = y_mean[lhs] - y_mean[rhs];
        const double dz = z_mean[lhs] - z_mean[rhs];
        const double dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 > cutoff2)
          return;
        const std::uint64_t key = pack_sparse_pair_key(lhs, rhs);
        keys.push_back(key);
        if (use_knn) {
          const float d2 = static_cast<float>(dist2);
          ranked_neighbors[lhs].push_back(Neighbor{d2, rhs});
          ranked_neighbors[rhs].push_back(Neighbor{d2, lhs});
        }
      });

  if (use_knn) {
    auto better = [](const Neighbor &lhs, const Neighbor &rhs) {
      if (lhs.dist2 != rhs.dist2)
        return lhs.dist2 < rhs.dist2;
      return lhs.partner < rhs.partner;
    };
    const size_t keep_k =
        std::min<size_t>(static_cast<size_t>(top_k), num_ca - 1);
    for (size_t i = 0; i < num_ca; ++i) {
      auto &neighbors = ranked_neighbors[i];
      if (neighbors.empty())
        continue;
      if (neighbors.size() > keep_k) {
        std::nth_element(neighbors.begin(),
                         neighbors.begin() +
                             static_cast<std::ptrdiff_t>(keep_k),
                         neighbors.end(), better);
        neighbors.resize(keep_k);
      }
      for (const auto &neighbor : neighbors)
        keys.push_back(pack_sparse_pair_key(i, neighbor.partner));
    }
  }

  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  pairs.reserve(keys.size());
  for (const std::uint64_t key : keys)
    pairs.push_back(unpack_sparse_pair_key(key));
  return pairs;
}

std::vector<std::pair<size_t, size_t>> build_allostery_candidate_pairs(
    const DCCMNetwork::TrajectoryRows &traj_x,
    const DCCMNetwork::TrajectoryRows &traj_y,
    const DCCMNetwork::TrajectoryRows &traj_z,
    const DCCMNetwork::TrajectoryRows &anchor_x,
    const DCCMNetwork::TrajectoryRows &anchor_y,
    const DCCMNetwork::TrajectoryRows &anchor_z, size_t num_ca,
    size_t total_frames, const DCCMNetwork::AllosteryConfig &cfg) {
  struct RankedNeighbor {
    size_t partner = 0;
    std::uint32_t score = 0;
    float min_dist2 = std::numeric_limits<float>::infinity();
  };

  const size_t dense_pair_count =
      num_ca < 2 ? 0 : num_ca * (num_ca - 1) / 2;
  constexpr size_t kSafeDensePairVectorLimit = 50'000'000;
  if (cfg.force_sparse_candidates ||
      (dense_pair_count > kSafeDensePairVectorLimit &&
       (cfg.cutoff_ang > 0.0 || cfg.knn > 0))) {
    const double cutoff = cfg.cutoff_ang > 0.0 ? cfg.cutoff_ang : 30.0;
    const int k = cfg.knn > 0 ? cfg.knn : 24;
    if (!cfg.force_sparse_candidates) {
      TitanUI::print_warning(
          "Allostery candidate pruning switched to memory-safe frame-0 "
          "spatial sparse mode; dense pair-vote buffers would exceed budget.");
    }
    return build_sparse_spatial_candidate_pairs(traj_x, traj_y, traj_z, num_ca,
                                                total_frames, cutoff, k);
  }
  if (dense_pair_count > kSafeDensePairVectorLimit && cfg.cutoff_ang <= 0.0 &&
      cfg.knn <= 0) {
    TitanUI::print_warning(
        "Allostery dense all-pairs request exceeds the safe pair-vector "
        "limit; using sparse defaults (30 A + top-24) instead.");
    return build_sparse_spatial_candidate_pairs(traj_x, traj_y, traj_z, num_ca,
                                                total_frames, 30.0, 24);
  }
  if (cfg.cutoff_ang <= 0.0 && cfg.knn <= 0)
    return build_dense_upper_triangle_pairs(num_ca);

  std::vector<std::pair<size_t, size_t>> pairs;
  std::vector<unsigned char> seen(dense_pair_count, 0);
  auto mark_pair = [&](size_t i, size_t j) {
    const size_t a = std::min(i, j);
    const size_t b = std::max(i, j);
    const size_t pair_index = upper_triangle_pair_index(a, b, num_ca);
    if (seen[pair_index] != 0)
      return false;
    seen[pair_index] = 1;
    pairs.emplace_back(a, b);
    return true;
  };
  auto has_complete_rows = [&](const DCCMNetwork::TrajectoryRows &x_rows,
                               const DCCMNetwork::TrajectoryRows &y_rows,
                               const DCCMNetwork::TrajectoryRows &z_rows,
                               size_t frame_limit) {
    if (x_rows.size() < frame_limit || y_rows.size() < frame_limit ||
        z_rows.size() < frame_limit) {
      return false;
    }
    for (size_t f = 0; f < frame_limit; ++f) {
      if (x_rows[f].size() < num_ca || y_rows[f].size() < num_ca ||
          z_rows[f].size() < num_ca) {
        return false;
      }
    }
    return true;
  };
  auto append_top_ranked_pairs =
      [&](const std::vector<std::uint32_t> &scores,
          const std::vector<float> &min_dist2, size_t k) {
        std::vector<std::vector<RankedNeighbor>> ranked_neighbors(num_ca);
        for (auto &neighbors : ranked_neighbors)
          neighbors.reserve(k * 2);
        for (size_t i = 0; i < num_ca; ++i) {
          for (size_t j = i + 1; j < num_ca; ++j) {
            const size_t pair_index = upper_triangle_pair_index(i, j, num_ca);
            if (scores[pair_index] == 0)
              continue;
            ranked_neighbors[i].push_back(
                RankedNeighbor{j, scores[pair_index], min_dist2[pair_index]});
            ranked_neighbors[j].push_back(
                RankedNeighbor{i, scores[pair_index], min_dist2[pair_index]});
          }
        }

        auto better_neighbor = [](const RankedNeighbor &lhs,
                                  const RankedNeighbor &rhs) {
          if (lhs.score != rhs.score)
            return lhs.score > rhs.score;
          if (lhs.min_dist2 != rhs.min_dist2)
            return lhs.min_dist2 < rhs.min_dist2;
          return lhs.partner < rhs.partner;
        };

        bool added_any = false;
        for (size_t i = 0; i < num_ca; ++i) {
          auto &neighbors = ranked_neighbors[i];
          if (neighbors.empty())
            continue;
          if (neighbors.size() > k) {
            std::nth_element(
                neighbors.begin(),
                neighbors.begin() + static_cast<std::ptrdiff_t>(k),
                neighbors.end(), better_neighbor);
            std::sort(neighbors.begin(),
                      neighbors.begin() + static_cast<std::ptrdiff_t>(k),
                      better_neighbor);
          } else {
            std::sort(neighbors.begin(), neighbors.end(), better_neighbor);
          }
          const size_t keep = std::min(k, neighbors.size());
          for (size_t n = 0; n < keep; ++n) {
            added_any |=
                mark_pair(std::min(i, neighbors[n].partner),
                          std::max(i, neighbors[n].partner));
          }
        }
        return added_any;
      };
  auto append_frame_voted_knn = [&](size_t k, size_t frame_limit) {
    if (!has_complete_rows(traj_x, traj_y, traj_z, frame_limit))
      return false;

    std::vector<std::uint32_t> vote_counts(dense_pair_count, 0);
    std::vector<float> min_dist2(dense_pair_count,
                                 std::numeric_limits<float>::infinity());
    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(num_ca > 0 ? num_ca - 1 : 0);
    auto rank_cmp = [](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    };

    for (size_t f = 0; f < frame_limit; ++f) {
      const auto &x_row = traj_x[f];
      const auto &y_row = traj_y[f];
      const auto &z_row = traj_z[f];

      for (size_t i = 0; i < num_ca; ++i) {
        distances.clear();
        const double xi = x_row[i];
        const double yi = y_row[i];
        const double zi = z_row[i];
        for (size_t j = 0; j < num_ca; ++j) {
          if (i == j)
            continue;
          const double dx = xi - x_row[j];
          const double dy = yi - y_row[j];
          const double dz = zi - z_row[j];
          distances.emplace_back(static_cast<float>(dx * dx + dy * dy + dz * dz),
                                 j);
        }

        if (distances.size() > k) {
          std::nth_element(distances.begin(),
                           distances.begin() +
                               static_cast<std::ptrdiff_t>(k),
                           distances.end(), rank_cmp);
          std::sort(distances.begin(),
                    distances.begin() + static_cast<std::ptrdiff_t>(k),
                    rank_cmp);
        } else {
          std::sort(distances.begin(), distances.end(), rank_cmp);
        }

        for (size_t n = 0; n < k; ++n) {
          const size_t partner = distances[n].second;
          const size_t a = std::min(i, partner);
          const size_t b = std::max(i, partner);
          const size_t pair_index = upper_triangle_pair_index(a, b, num_ca);
          if (vote_counts[pair_index] !=
              std::numeric_limits<std::uint32_t>::max()) {
            ++vote_counts[pair_index];
          }
          min_dist2[pair_index] =
              std::min(min_dist2[pair_index], distances[n].first);
        }
      }
    }

    return append_top_ranked_pairs(vote_counts, min_dist2, k);
  };
  auto append_contact_frequency_knn = [&](size_t k, size_t frame_limit) {
    if (!has_complete_rows(anchor_x, anchor_y, anchor_z, frame_limit))
      return false;

    const double cutoff = cfg.knn_contact_cutoff;
    const double cutoff2 = cutoff * cutoff;
    titan_neighbor::HalfShellGrid neighbor_grid;
    std::vector<std::uint32_t> contact_counts(dense_pair_count, 0);
    std::vector<float> min_dist2(dense_pair_count,
                                 std::numeric_limits<float>::infinity());
    bool saw_contact = false;

    for (size_t f = 0; f < frame_limit; ++f) {
      const auto &x_row = anchor_x[f];
      const auto &y_row = anchor_y[f];
      const auto &z_row = anchor_z[f];
      const auto [xmin, xmax] =
          std::minmax_element(x_row.begin(), x_row.begin() + num_ca);
      const auto [ymin, ymax] =
          std::minmax_element(y_row.begin(), y_row.begin() + num_ca);
      const auto [zmin, zmax] =
          std::minmax_element(z_row.begin(), z_row.begin() + num_ca);

      titan_neighbor::OrthorhombicGridSpec spec;
      spec.cutoff = cutoff;
      spec.shift_x = -(*xmin) + cutoff;
      spec.shift_y = -(*ymin) + cutoff;
      spec.shift_z = -(*zmin) + cutoff;
      spec.len_x = (*xmax - *xmin) + 2.0 * cutoff;
      spec.len_y = (*ymax - *ymin) + 2.0 * cutoff;
      spec.len_z = (*zmax - *zmin) + 2.0 * cutoff;
      spec.periodic = false;

      neighbor_grid.for_each_pair(
          num_ca, spec,
          [&](std::size_t atom_index, double &x, double &y, double &z) {
            x = x_row[atom_index];
            y = y_row[atom_index];
            z = z_row[atom_index];
          },
          [&](std::size_t atom_index) { return atom_index < num_ca; },
          [&](std::size_t lhs, std::size_t rhs) {
            const double dx = x_row[lhs] - x_row[rhs];
            const double dy = y_row[lhs] - y_row[rhs];
            const double dz = z_row[lhs] - z_row[rhs];
            const float dist2 =
                static_cast<float>(dx * dx + dy * dy + dz * dz);
            if (dist2 > cutoff2)
              return;

            saw_contact = true;
            const size_t i = std::min(lhs, rhs);
            const size_t j = std::max(lhs, rhs);
            const size_t pair_index = upper_triangle_pair_index(i, j, num_ca);
            if (contact_counts[pair_index] !=
                std::numeric_limits<std::uint32_t>::max()) {
              ++contact_counts[pair_index];
            }
            min_dist2[pair_index] = std::min(min_dist2[pair_index], dist2);
          });
    }

    if (!saw_contact)
      return false;
    return append_top_ranked_pairs(contact_counts, min_dist2, k);
  };

  if (cfg.cutoff_ang > 0.0) {
    auto cutoff_pairs = build_lspmd_candidate_pairs(
        traj_x, traj_y, traj_z, num_ca, total_frames, cfg.cutoff_ang,
        cfg.use_avg_cutoff);
    for (const auto &[i, j] : cutoff_pairs)
      mark_pair(i, j);
  }

  if (cfg.knn > 0 && num_ca > 1 && total_frames > 0) {
    const size_t k = std::min<size_t>(static_cast<size_t>(cfg.knn), num_ca - 1);
    const size_t frame_limit = cfg.use_avg_cutoff ? total_frames : 1;
    const bool can_rank = has_complete_rows(traj_x, traj_y, traj_z, frame_limit);
    if (!can_rank) {
      if (pairs.empty())
        return build_dense_upper_triangle_pairs(num_ca);
      std::sort(pairs.begin(), pairs.end());
      return pairs;
    }
    bool added_knn_pairs = false;
    if (cfg.knn_use_contact_prior) {
      added_knn_pairs = append_contact_frequency_knn(k, frame_limit);
    }
    if (!added_knn_pairs) {
      added_knn_pairs = append_frame_voted_knn(k, frame_limit);
    }
    if (!added_knn_pairs && pairs.empty())
      return build_dense_upper_triangle_pairs(num_ca);
  }

  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

bool has_complete_anchor_rows(const DCCMNetwork::TrajectoryRows &x_rows,
                              const DCCMNetwork::TrajectoryRows &y_rows,
                              const DCCMNetwork::TrajectoryRows &z_rows,
                              size_t num_ca, size_t total_frames) {
  if (x_rows.size() != total_frames || y_rows.size() != total_frames ||
      z_rows.size() != total_frames) {
    return false;
  }
  for (size_t f = 0; f < total_frames; ++f) {
    if (x_rows[f].size() < num_ca || y_rows[f].size() < num_ca ||
        z_rows[f].size() < num_ca) {
      return false;
    }
  }
  return true;
}


double normalized_l1_distance(const std::vector<double> &lhs,
                              const std::vector<double> &rhs) {
  if (lhs.size() != rhs.size() || lhs.empty())
    return 0.0;
  double diff_sum = 0.0;
  double mass = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    diff_sum += std::abs(lhs[i] - rhs[i]);
    mass += lhs[i] + rhs[i];
  }
  if (mass <= 0.0)
    return 0.0;
  return 0.5 * diff_sum / mass;
}

void smooth_histogram_kde_2d(std::vector<double> &hist, size_t bins,
                             double sigma) {
  if (hist.empty() || bins == 0 || sigma <= 1e-9)
    return;
  const int radius = std::max(1, static_cast<int>(std::ceil(2.5 * sigma)));
  std::vector<double> kernel(static_cast<size_t>(2 * radius + 1), 0.0);
  double ksum = 0.0;
  for (int d = -radius; d <= radius; ++d) {
    const double x = static_cast<double>(d) / sigma;
    const double w = std::exp(-0.5 * x * x);
    kernel[static_cast<size_t>(d + radius)] = w;
    ksum += w;
  }
  if (ksum <= 0.0)
    return;
  for (double &w : kernel)
    w /= ksum;

  std::vector<double> tmp(hist.size(), 0.0);
  for (size_t i = 0; i < bins; ++i) {
    for (size_t j = 0; j < bins; ++j) {
      double acc = 0.0;
      for (int dj = -radius; dj <= radius; ++dj) {
        const int jj = static_cast<int>(j) + dj;
        if (jj < 0 || jj >= static_cast<int>(bins))
          continue;
        acc += hist[i * bins + static_cast<size_t>(jj)] *
               kernel[static_cast<size_t>(dj + radius)];
      }
      tmp[i * bins + j] = acc;
    }
  }

  std::fill(hist.begin(), hist.end(), 0.0);
  for (size_t i = 0; i < bins; ++i) {
    for (size_t j = 0; j < bins; ++j) {
      double acc = 0.0;
      for (int di = -radius; di <= radius; ++di) {
        const int ii = static_cast<int>(i) + di;
        if (ii < 0 || ii >= static_cast<int>(bins))
          continue;
        acc += tmp[static_cast<size_t>(ii) * bins + j] *
               kernel[static_cast<size_t>(di + radius)];
      }
      hist[i * bins + j] = acc;
    }
  }
}
template <typename BinType>
size_t calculate_allostery_information_flat_impl(
    const DCCMNetwork::TrajectoryRows &traj_x,
    const DCCMNetwork::TrajectoryRows &traj_y,
    const DCCMNetwork::TrajectoryRows &traj_z,
    const DCCMNetwork::TrajectoryRows &anchor_x,
    const DCCMNetwork::TrajectoryRows &anchor_y,
    const DCCMNetwork::TrajectoryRows &anchor_z,
    const std::vector<std::string> &node_labels, size_t total_frames,
    TitanExecutor &executor, const DCCMNetwork::AllosteryConfig &cfg, int B,
    const std::string &gc_filename, const std::string &mi_filename,
    const std::string &te_filename,
    size_t *dense_pair_count_out = nullptr,
    std::vector<DCCMNetwork::StoredEdge> *gc_edges_out = nullptr,
    std::vector<DCCMNetwork::StoredEdge> *te_edges_out = nullptr) {
  if (total_frames == 0 || node_labels.empty())
    return 0;
  const size_t num_ca = node_labels.size();
  const size_t tau = static_cast<size_t>(std::max(1, cfg.tau_lag));
  if (tau >= total_frames) {
    std::cerr << "[Allostery] tau_lag (" << tau << ") >= total_frames ("
              << total_frames << "). Skipping.\n";
    return 0;
  }

  const size_t B_sz = static_cast<size_t>(B);
  std::string estimator_mode = cfg.estimator;
  std::transform(estimator_mode.begin(), estimator_mode.end(),
                 estimator_mode.begin(), [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  if (estimator_mode == "histogram")
    estimator_mode = "hist";
  const bool estimator_knn = (estimator_mode == "knn");
  const bool estimator_kde = (estimator_mode == "kde");
  if (B_sz > static_cast<size_t>(std::numeric_limits<BinType>::max())) {
    TitanUI::print_warning(
        "Allostery bin count exceeds selected bin storage. Skipping.");
    return 0;
  }

  const size_t dense_pair_count = num_ca * (num_ca - 1) / 2;
  if (dense_pair_count_out != nullptr)
    *dense_pair_count_out = dense_pair_count;
  titan_corr_knn::PairPruningConfig pruning_cfg;
  pruning_cfg.pair_topk = cfg.knn;
  pruning_cfg.pair_cutoff_ang = cfg.cutoff_ang;
  pruning_cfg.use_avg_cutoff = cfg.use_avg_cutoff;
  pruning_cfg.pair_use_contact_prior = cfg.knn_use_contact_prior;
  pruning_cfg.pair_contact_cutoff_ang = cfg.knn_contact_cutoff;
  const auto pairs = titan_corr_knn::build_candidate_pairs(
      traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z, num_ca,
      total_frames, pruning_cfg, dense_pair_count_out);
  const size_t n_pairs = pairs.size();
  std::ofstream gc_out(gc_filename);
  gc_out << "Source,Target,GC\n";
  std::ofstream mi_out(mi_filename);
  mi_out << "Source,Target,MI\n";
  std::ofstream te_out(te_filename);
  te_out << "Source,Target,Net_TE\n";
  if (gc_edges_out != nullptr)
    gc_edges_out->clear();
  if (te_edges_out != nullptr)
    te_edges_out->clear();
  if (n_pairs == 0)
    return 0;

  std::vector<double> mean_x(num_ca, 0.0), mean_y(num_ca, 0.0),
      mean_z(num_ca, 0.0);
  for (size_t f = 0; f < total_frames; ++f) {
    const auto &x_row = traj_x[f];
    const auto &y_row = traj_y[f];
    const auto &z_row = traj_z[f];
    for (size_t a = 0; a < num_ca; ++a) {
      mean_x[a] += x_row[a];
      mean_y[a] += y_row[a];
      mean_z[a] += z_row[a];
    }
  }
  const double inv_f = 1.0 / static_cast<double>(total_frames);
  for (size_t a = 0; a < num_ca; ++a) {
    mean_x[a] *= inv_f;
    mean_y[a] *= inv_f;
    mean_z[a] *= inv_f;
  }

  FlatAlignedMatrixT<BinType> proj_bins(num_ca, total_frames,
                                        static_cast<BinType>(0));
  std::vector<double> H1(num_ca, 0.0);
  std::vector<double> H_past;
  std::vector<double> H_auto;
  if (cfg.compute_te) {
    H_past.assign(num_ca, 0.0);
    H_auto.assign(num_ca, 0.0);
  }
  const double norm1_full =
      0.5 * static_cast<double>(B) + static_cast<double>(total_frames);
  const double norm1_tau =
      0.5 * static_cast<double>(B) +
      static_cast<double>(total_frames - tau);
  const double norm2_full =
      0.5 * static_cast<double>(B_sz * B_sz) +
      static_cast<double>(total_frames);
  const double norm2_tau =
      0.5 * static_cast<double>(B_sz * B_sz) +
      static_cast<double>(total_frames - tau);
  const double norm3_tau =
      0.5 * static_cast<double>(B_sz * B_sz * B_sz) +
      static_cast<double>(total_frames - tau);

  auto entropy_from_hist = [](const std::vector<double> &hist,
                              double norm) -> double {
    double H = 0.0;
    for (double count : hist) {
      const double p = count / norm;
      if (p > 0.0)
        H -= p * std::log(p);
    }
    return H;
  };

  auto estimate_mi = [&](const std::vector<double> &joint_hist,
                         const BinType *bins_i,
                         const BinType *bins_j) -> double {
    std::vector<double> hist_i(B_sz, 0.5);
    std::vector<double> hist_j(B_sz, 0.5);
    for (size_t f = 0; f < total_frames; ++f) {
      hist_i[static_cast<size_t>(bins_i[f])] += 1.0;
      hist_j[static_cast<size_t>(bins_j[f])] += 1.0;
    }

    std::vector<double> joint_est = joint_hist;
    if (estimator_kde) {
      double sigma = cfg.kde_bandwidth;
      if (sigma <= 0.0) {
        sigma = std::max(0.75, static_cast<double>(B) / 24.0);
      }
      smooth_histogram_kde_2d(joint_est, B_sz, sigma);
    }

    const double H_i = entropy_from_hist(hist_i, norm1_full);
    const double H_j = entropy_from_hist(hist_j, norm1_full);
    const double H_ij = entropy_from_hist(joint_est, norm2_full);
    double mi = H_i + H_j - H_ij;
    if (estimator_knn) {
      const double l1 = normalized_l1_distance(joint_est, joint_hist);
      mi *= (1.0 + 0.15 * l1);
    }
    return std::max(0.0, mi);
  };
  {
    const size_t atom_worker_count = std::max<size_t>(1, executor.capacity());
    const size_t atoms_per_worker =
        (num_ca + atom_worker_count - 1) / atom_worker_count;
    executor.parallel_do(atom_worker_count, [&](size_t t) {
      const size_t a_begin = t * atoms_per_worker;
      const size_t a_end = std::min(a_begin + atoms_per_worker, num_ca);
      if (a_begin >= a_end)
        return;
      AlignedDoubleVector atom_proj(total_frames, 0.0);
      std::vector<double> hist1(B_sz, 0.0);
      std::vector<double> hist2(cfg.compute_te ? B_sz * B_sz : 0, 0.0);
      for (size_t a = a_begin; a < a_end; ++a) {
        double c_xx = 0.0, c_xy = 0.0, c_xz = 0.0;
        double c_yy = 0.0, c_yz = 0.0, c_zz = 0.0;
        const double avg_x = mean_x[a];
        const double avg_y = mean_y[a];
        const double avg_z = mean_z[a];
        for (size_t f = 0; f < total_frames; ++f) {
          const double dx = traj_x[f][a] - avg_x;
          const double dy = traj_y[f][a] - avg_y;
          const double dz = traj_z[f][a] - avg_z;
          c_xx += dx * dx;
          c_xy += dx * dy;
          c_xz += dx * dz;
          c_yy += dy * dy;
          c_yz += dy * dz;
          c_zz += dz * dz;
        }

        Mat3 cov3{};
        const double inv_frames = 1.0 / static_cast<double>(total_frames);
        cov3(0, 0) = c_xx * inv_frames;
        cov3(0, 1) = c_xy * inv_frames;
        cov3(0, 2) = c_xz * inv_frames;
        cov3(1, 0) = cov3(0, 1);
        cov3(1, 1) = c_yy * inv_frames;
        cov3(1, 2) = c_yz * inv_frames;
        cov3(2, 0) = cov3(0, 2);
        cov3(2, 1) = cov3(1, 2);
        cov3(2, 2) = c_zz * inv_frames;

        const auto es = titan::algebra::symmetric_eigen3(cov3);
        const double ex = es.vectors(0, 2);
        const double ey = es.vectors(1, 2);
        const double ez = es.vectors(2, 2);

        double p_min = std::numeric_limits<double>::max();
        double p_max = std::numeric_limits<double>::lowest();
        for (size_t f = 0; f < total_frames; ++f) {
          const double proj_val = (traj_x[f][a] - avg_x) * ex +
                                  (traj_y[f][a] - avg_y) * ey +
                                  (traj_z[f][a] - avg_z) * ez;
          atom_proj[f] = proj_val;
          p_min = std::min(p_min, proj_val);
          p_max = std::max(p_max, proj_val);
        }

        double p_range = p_max - p_min;
        if (p_range < 1e-15)
          p_range = 1e-15;

        BinType *bins = proj_bins.row_data(a);
        std::fill(hist1.begin(), hist1.end(), 0.5);
        for (size_t f = 0; f < total_frames; ++f) {
          const double scaled = (atom_proj[f] - p_min) / p_range;
          int bin = static_cast<int>(scaled * static_cast<double>(B));
          if (bin < 0)
            bin = 0;
          else if (bin >= B)
            bin = B - 1;
          bins[f] = static_cast<BinType>(bin);
          hist1[static_cast<size_t>(bins[f])] += 1.0;
        }
        H1[a] = entropy_from_hist(hist1, norm1_full);

        if (cfg.compute_te) {
          std::fill(hist1.begin(), hist1.end(), 0.5);
          for (size_t f = tau; f < total_frames; ++f)
            hist1[static_cast<size_t>(bins[f - tau])] += 1.0;
          H_past[a] = entropy_from_hist(hist1, norm1_tau);

          std::fill(hist2.begin(), hist2.end(), 0.5);
          for (size_t f = tau; f < total_frames; ++f) {
            const size_t b_now = static_cast<size_t>(bins[f]);
            const size_t b_past = static_cast<size_t>(bins[f - tau]);
            hist2[b_now * B_sz + b_past] += 1.0;
          }
          H_auto[a] = entropy_from_hist(hist2, norm2_tau);
        }
      }
    });
  }

  std::vector<double> gc_vals(n_pairs, 0.0);
  std::vector<double> mi_vals(n_pairs, 0.0);
  std::vector<double> net_te_vals(cfg.compute_te ? n_pairs : 0, 0.0);

  auto worker = [&](size_t p_start, size_t p_end) {
    std::vector<double> hist2d(B_sz * B_sz, 0.0);
    std::vector<double> hist2d_b(cfg.compute_te ? B_sz * B_sz : 0, 0.0);
    std::vector<double> hist3d(cfg.compute_te ? B_sz * B_sz * B_sz : 0, 0.0);

    for (size_t p = p_start; p < p_end; ++p) {
      const size_t i = pairs[p].first;
      const size_t j = pairs[p].second;
      const BinType *bins_i = proj_bins.row_data(i);
      const BinType *bins_j = proj_bins.row_data(j);

      std::fill(hist2d.begin(), hist2d.end(), 0.5);
      if (cfg.compute_te)
        std::fill(hist2d_b.begin(), hist2d_b.end(), 0.5);
      for (size_t f = 0; f < total_frames; ++f) {
        const size_t bi = static_cast<size_t>(bins_i[f]);
        const size_t bj = static_cast<size_t>(bins_j[f]);
        hist2d[bi * B_sz + bj] += 1.0;
        if (cfg.compute_te && f >= tau) {
          const size_t bi_past = static_cast<size_t>(bins_i[f - tau]);
          const size_t bj_past = static_cast<size_t>(bins_j[f - tau]);
          hist2d_b[bi_past * B_sz + bj_past] += 1.0;
        }
      }

      double MI = estimate_mi(hist2d, bins_i, bins_j);
      if (!std::isfinite(MI) || MI < 0.0)
        MI = 0.0;
      mi_vals[p] = MI;
      gc_vals[p] = std::sqrt(1.0 - std::exp(-2.0 * MI));

      if (!cfg.compute_te)
        continue;

      const double H_past_ij = entropy_from_hist(hist2d_b, norm2_tau);
      auto calc_h3 = [&](const BinType *src_bins,
                         const BinType *dst_bins) -> double {
        std::fill(hist3d.begin(), hist3d.end(), 0.5);
        for (size_t f = tau; f < total_frames; ++f) {
          const size_t b1 = static_cast<size_t>(dst_bins[f]);
          const size_t b2 = static_cast<size_t>(dst_bins[f - tau]);
          const size_t b3 = static_cast<size_t>(src_bins[f - tau]);
          hist3d[(b1 * B_sz + b2) * B_sz + b3] += 1.0;
        }
        return entropy_from_hist(hist3d, norm3_tau);
      };

      const double H3_j_from_i = calc_h3(bins_i, bins_j);
      const double H3_i_from_j = calc_h3(bins_j, bins_i);
      const double te_ij = H_auto[j] + H_past_ij - H_past[j] - H3_j_from_i;
      const double te_ji = H_auto[i] + H_past_ij - H_past[i] - H3_i_from_j;
      net_te_vals[p] = te_ij - te_ji;
    }
  };

  const size_t worker_count = std::max<size_t>(1, executor.capacity());
  const size_t chunk = (n_pairs + worker_count - 1) / worker_count;
  executor.parallel_do(worker_count, [&](size_t t) {
    const size_t start = t * chunk;
    const size_t end = std::min(start + chunk, n_pairs);
    if (start < end)
      worker(start, end);
  });

  for (size_t p = 0; p < n_pairs; ++p) {
    const size_t i = pairs[p].first;
    const size_t j = pairs[p].second;
    mi_out << node_labels[i] << "," << node_labels[j] << "," << mi_vals[p]
           << "\n";
  }
  mi_out.close();

  for (size_t p = 0; p < n_pairs; ++p) {
      if (gc_vals[p] > 0.35) {
        const size_t i = pairs[p].first;
        const size_t j = pairs[p].second;
        gc_out << node_labels[i] << "," << node_labels[j] << "," << gc_vals[p]
               << "\n";
        if (gc_edges_out != nullptr) {
          gc_edges_out->push_back({static_cast<int>(i), static_cast<int>(j),
                                   gc_vals[p], gc_vals[p]});
        }
      }
    }
  gc_out.close();

  if (cfg.compute_te) {
    for (size_t p = 0; p < n_pairs; ++p) {
      if (std::abs(net_te_vals[p]) > 0.05) {
        size_t drv = pairs[p].first;
        size_t tgt = pairs[p].second;
        if (net_te_vals[p] <= 0.0)
          std::swap(drv, tgt);
        te_out << node_labels[drv] << "," << node_labels[tgt] << ","
               << std::abs(net_te_vals[p]) << "\n";
        if (te_edges_out != nullptr) {
          te_edges_out->push_back({static_cast<int>(drv), static_cast<int>(tgt),
                                   std::abs(net_te_vals[p]),
                                   std::abs(net_te_vals[p])});
        }
      }
    }
  }
  te_out.close();
  return n_pairs;
}

} // namespace

// ── Helper: find Cβ (or N for Gly) atom index within a residue ────────────────
int DCCMNetwork::find_cb_for_residue(const Multi_MOL &mol, int resnr,
                                     const std::string &chain, bool is_gly) {
  const std::string target = is_gly ? "N" : "CB";
  for (size_t i = 0; i < mol.info.size(); ++i) {
    std::string at = mol.info[i].get_atom();
    // trim whitespace
    auto p1 = at.find_first_not_of(" \t");
    auto p2 = at.find_last_not_of(" \t");
    if (p1 != std::string::npos)
      at = at.substr(p1, p2 - p1 + 1);
    if (at == target &&
        mol.info[i].get_resnr() == resnr &&
        mol.info[i].get_chain() == chain)
      return static_cast<int>(i);
  }
  return -1; // not found
}

// Hàm dot_product tối ưu bằng tập lệnh SIMD AVX2
inline double fast_dot_product(const double *a, const double *b, size_t n) {
  return titan_simd_dispatch::dot_product(a, b, n, false);
}

inline double fast_dot_product_aligned(const double *a, const double *b,
                                       size_t n) {
  return titan_simd_dispatch::dot_product(a, b, n, true);
}

void DCCMNetwork::init(const Multi_MOL &pro_in, const Multi_MOL &lig,
                       const arguments &args) {
  // CRITICAL FIX: Clear all state from previous trajectories (Batch Mode
  // Safety)
  ca_indices.clear();
  ca_resnrs.clear();
  ca_is_lig.clear();
  ca_chain.clear();
  ca_resname.clear();
  cb_indices.clear();
  traj_x.clear();
  traj_y.clear();
  traj_z.clear();
  traj_cbx.clear();
  traj_cby.clear();
  traj_cbz.clear();
  frame_global_indices_.clear();
  traj_times_.clear();
  traj_replicas_.clear();
  traj_box_volumes_nm3_.clear();
  latest_traj_x_.clear();
  latest_traj_y_.clear();
  latest_traj_z_.clear();
  first_pro_frame_.release_memory();
  first_lig_frame_.release_memory();
  first_pro_info_.clear();
  first_lig_info_.clear();
  has_first_snapshot_frame_ = false;
  total_frames = 0;
  kabsch_aligned_ = false;
  dense_postproc_skipped_due_memory_ = false;
  network_sparse_mode_active_ = false;
  last_network_graph_edges_.clear();
  last_allostery_gc_edges_.clear();
  last_allostery_te_edges_.clear();
  network_memory_cfg_.mode = args.network_mode;
  network_memory_cfg_.mem_budget_gb = args.mem_budget_gb;
  network_memory_cfg_.low_memory = args.low_memory;
  network_memory_cfg_.sparse_cut_ang = args.network_sparse_cut;
  network_memory_cfg_.sparse_k = args.network_sparse_k;
  network_memory_cfg_.cache_precision = args.network_cache_precision;
  community_cfg_.enabled = args.community.flag_community && !args.graph.flag_graph;
  community_cfg_.method = args.community.method;
  uncertainty_cfg_ = args.uncertainty;
  last_community_result_ = CommunityResult{};
  store_full_traj_ =
      args.flag_network_explicit || args.flag_pca || args.flag_allostery ||
      args.corr_knn.flag_corr_knn || args.community.flag_community ||
      args.lspmd.flag_lspmd;
  store_cb_traj_ = args.flag_allostery || args.lspmd.flag_lspmd ||
                   (args.corr_knn.flag_corr_knn &&
                    args.corr_knn.pair_use_contact_prior);

  const size_t pro_coord_count =
      pro_in.m_mol.empty() ? 0 : coord_count(pro_in.m_mol.front());
  const size_t lig_coord_count =
      lig.m_mol.empty() ? 0 : coord_count(lig.m_mol.front());
  size_t skipped_rec_nodes = 0;
  size_t skipped_lig_nodes = 0;
  size_t cb_fallback_nodes = 0;
  const bool include_lig_trace_nodes = !args.flag_nocal;

  bool is_lig_rec = (args.mol_1_type == "ligand" || args.mol_1_type == "lig");
  bool is_lig_lg = (args.mol_2_type == "ligand" || args.mol_2_type == "lig");

  // Collect Receptor Nodes
  for (size_t i = 0; i < pro_in.info.size(); ++i) {
    if (i >= pro_coord_count) {
      ++skipped_rec_nodes;
      continue;
    }
    std::string atom_name = pro_in.info[i].get_atom();
    if (is_lig_rec || atom_name == "CA" || atom_name == "CA " ||
        atom_name == "BB" || atom_name == "BB2" || atom_name == "C4'" ||
        atom_name == "C4*") {
      ca_indices.emplace_back((int)i);
      ca_resnrs.emplace_back(is_lig_rec ? (i + 1) : pro_in.info[i].get_resnr());
      ca_is_lig.emplace_back(0); // 0 = Receptor
      ca_chain.emplace_back(pro_in.info[i].get_chain());
      ca_resname.emplace_back(is_lig_rec ? pro_in.info[i].get_atom()
                                         : pro_in.info[i].get_res_name());
    }
  }
  // Collect Ligand Nodes
  if (include_lig_trace_nodes) {
    for (size_t i = 0; i < lig.info.size(); ++i) {
      if (i >= lig_coord_count) {
        ++skipped_lig_nodes;
        continue;
      }
      std::string atom_name = lig.info[i].get_atom();
      if (is_lig_lg || atom_name == "CA" || atom_name == "CA " ||
          atom_name == "BB" || atom_name == "BB2" || atom_name == "C4'" ||
          atom_name == "C4*") {
        ca_indices.emplace_back((int)i);
        ca_resnrs.emplace_back(is_lig_lg ? (i + 1) : lig.info[i].get_resnr());
        ca_is_lig.emplace_back(1); // 1 = Ligand
        ca_chain.emplace_back(lig.info[i].get_chain());
        ca_resname.emplace_back(is_lig_lg ? lig.info[i].get_atom()
                                          : lig.info[i].get_res_name());
      }
    }
  }

  // Only allostery needs the heavier Cβ/N side trace cache.
  // Corr-kNN only needs this cache when contact-frequency pruning is active.
  if (store_cb_traj_) {
    cb_indices.resize(ca_indices.size());
    for (size_t i = 0; i < ca_indices.size(); ++i) {
      std::string rn = ca_resname[i];
      auto p1 = rn.find_first_not_of(" \t");
      auto p2 = rn.find_last_not_of(" \t");
      if (p1 != std::string::npos)
        rn = rn.substr(p1, p2 - p1 + 1);
      bool is_gly = (rn == "GLY");

      const Multi_MOL &mol_ref = (ca_is_lig[i] == 0) ? pro_in : lig;
      const size_t coord_limit =
          (ca_is_lig[i] == 0) ? pro_coord_count : lig_coord_count;
      const int cb_idx =
          find_cb_for_residue(mol_ref, ca_resnrs[i], ca_chain[i], is_gly);
      if (cb_idx >= 0 && static_cast<size_t>(cb_idx) < coord_limit) {
        cb_indices[i] = cb_idx;
      } else {
        cb_indices[i] = -1;
        if (cb_idx >= 0) {
          ++cb_fallback_nodes;
        }
      }
    }
  } else {
    cb_indices.clear();
  }

  total_frames = 0;
  frame_global_indices_.clear();
  traj_box_volumes_nm3_.clear();
  if (skipped_rec_nodes > 0 || skipped_lig_nodes > 0 || cb_fallback_nodes > 0) {
    std::ostringstream oss;
    oss << "DCCM/TICA bounds filter active:";
    if (skipped_rec_nodes > 0) {
      oss << " skipped " << skipped_rec_nodes
          << " receptor topology atoms without coordinates";
    }
    if (skipped_lig_nodes > 0) {
      oss << (skipped_rec_nodes > 0 ? "," : "") << " skipped "
          << skipped_lig_nodes
          << " ligand topology atoms without coordinates";
    }
    if (cb_fallback_nodes > 0) {
      oss << ((skipped_rec_nodes > 0 || skipped_lig_nodes > 0) ? "," : "")
          << " fallback-to-CA for " << cb_fallback_nodes
          << " missing Cbeta/N nodes";
    }
    TitanUI::print_warning(oss.str() + ".");
  }
  if (ca_indices.empty()) {
    TitanUI::print_warning(
        "DCCM/TICA: no valid trace nodes remain after coordinate-bounds "
        "filtering.");
  }
}

void DCCMNetwork::process_frame(const Multi_MOL &pro_in, const Multi_MOL &lig,
                                size_t chunk_snap_idx,
                                size_t global_frame_idx,
                                double time_ps,
                                int replica_id,
                                double box_volume_nm3) {
  static bool warned_bad_frame = false;
  if (ca_indices.empty())
    return;

  if (chunk_snap_idx >= pro_in.m_mol.size()) {
    if (!warned_bad_frame) {
      TitanUI::print_warning(
          "DCCM/TICA: skipping frame because receptor frame index exceeds "
          "loaded chunk.");
      warned_bad_frame = true;
    }
    return;
  }

  AlignedDoubleVector cx(ca_indices.size(), 0.0),
      cy(ca_indices.size(), 0.0), cz(ca_indices.size(), 0.0);
  AlignedDoubleVector cbx, cby, cbz;
  if (store_cb_traj_) {
    cbx.assign(ca_indices.size(), 0.0);
    cby.assign(ca_indices.size(), 0.0);
    cbz.assign(ca_indices.size(), 0.0);
  }
  size_t l_snap = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;

  if (l_snap >= lig.m_mol.size()) {
    if (!warned_bad_frame) {
      TitanUI::print_warning(
          "DCCM/TICA: skipping frame because ligand frame index exceeds "
          "loaded chunk.");
      warned_bad_frame = true;
    }
    return;
  }

  if (!has_first_snapshot_frame_) {
    first_pro_frame_ = pro_in.m_mol[chunk_snap_idx];
    first_pro_info_ = pro_in.info;
    if (!lig.m_mol.empty()) {
      first_lig_frame_ = lig.m_mol[l_snap];
      first_lig_info_ = lig.info;
    } else {
      first_lig_frame_.release_memory();
      first_lig_info_.clear();
    }
    has_first_snapshot_frame_ = true;
  }

  const size_t pro_coord_count = coord_count(pro_in.m_mol[chunk_snap_idx]);
  const size_t lig_coord_count = coord_count(lig.m_mol[l_snap]);

  for (size_t i = 0; i < ca_indices.size(); ++i) {
    const size_t coord_limit =
        (ca_is_lig[i] == 0) ? pro_coord_count : lig_coord_count;
    const int idx = ca_indices[i];
    if (idx < 0 || static_cast<size_t>(idx) >= coord_limit) {
      if (!warned_bad_frame) {
        TitanUI::print_warning(
            "DCCM/TICA: skipping at least one frame because a trace atom index "
            "is outside the available coordinate arrays. This usually means "
            "topology and trajectory atom counts differ.");
        warned_bad_frame = true;
      }
      return;
    }
  }

  for (size_t i = 0; i < ca_indices.size(); ++i) {
    int idx = ca_indices[i];
    if (ca_is_lig[i] == 0) {
      cx[i] = pro_in.m_mol[chunk_snap_idx].pos.x[idx];
      cy[i] = pro_in.m_mol[chunk_snap_idx].pos.y[idx];
      cz[i] = pro_in.m_mol[chunk_snap_idx].pos.z[idx];
    } else {
      cx[i] = lig.m_mol[l_snap].pos.x[idx];
      cy[i] = lig.m_mol[l_snap].pos.y[idx];
      cz[i] = lig.m_mol[l_snap].pos.z[idx];
    }

    if (store_cb_traj_) {
      // Cβ/N coord: use cb_indices if valid, else fallback to CA
      int cb_idx = cb_indices[i];
      const size_t coord_limit =
          (ca_is_lig[i] == 0) ? pro_coord_count : lig_coord_count;
      if (cb_idx >= 0 && static_cast<size_t>(cb_idx) < coord_limit) {
        if (ca_is_lig[i] == 0) {
          cbx[i] = pro_in.m_mol[chunk_snap_idx].pos.x[cb_idx];
          cby[i] = pro_in.m_mol[chunk_snap_idx].pos.y[cb_idx];
          cbz[i] = pro_in.m_mol[chunk_snap_idx].pos.z[cb_idx];
        } else {
          cbx[i] = lig.m_mol[l_snap].pos.x[cb_idx];
          cby[i] = lig.m_mol[l_snap].pos.y[cb_idx];
          cbz[i] = lig.m_mol[l_snap].pos.z[cb_idx];
        }
      } else {
        cbx[i] = cx[i];
        cby[i] = cy[i];
        cbz[i] = cz[i];
      }
    }
  }
  if (store_full_traj_) {
    traj_x.emplace_back(std::move(cx));
    traj_y.emplace_back(std::move(cy));
    traj_z.emplace_back(std::move(cz));
    if (store_cb_traj_) {
      traj_cbx.emplace_back(std::move(cbx));
      traj_cby.emplace_back(std::move(cby));
      traj_cbz.emplace_back(std::move(cbz));
    }
    frame_global_indices_.push_back(static_cast<int64_t>(global_frame_idx));
    traj_times_.push_back(time_ps);
    traj_replicas_.push_back(replica_id);
    traj_box_volumes_nm3_.push_back(box_volume_nm3);
  } else {
    latest_traj_x_ = std::move(cx);
    latest_traj_y_ = std::move(cy);
    latest_traj_z_ = std::move(cz);
  }
  total_frames++;
}

void DCCMNetwork::calculate_and_export(TitanExecutor &executor,
                                       const info &info_output) {
  dense_postproc_skipped_due_memory_ = false;
  network_sparse_mode_active_ = false;
  if (total_frames == 0 || ca_indices.empty()) {
    last_community_result_ = CommunityResult{};
    if (community_cfg_.enabled) {
      TitanUI::print_warning(
          "Community: no DCCM frames or nodes were collected; writing "
          "header-only community outputs.");
      write_empty_community_outputs(info_output);
    }
    return;
  }
  size_t num_ca = ca_indices.size();

  long double estimated_extra_bytes = 0.0L;
  long double budget_bytes = 0.0L;
  long double pair_count = 0.0L;
  const bool dense_over_budget = should_skip_dense_dccm_postproc(
      num_ca, total_frames, network_memory_cfg_, estimated_extra_bytes,
      budget_bytes, pair_count);
  const bool force_dense = network_memory_cfg_.mode == "dense";
  const bool force_sparse = network_memory_cfg_.mode == "sparse";
  const bool run_sparse = force_sparse || (!force_dense && dense_over_budget);
  if (force_dense && dense_over_budget) {
    dense_postproc_skipped_due_memory_ = true;
    last_community_result_ = CommunityResult{};
    write_header_only_network_outputs(ca_resname, ca_resnrs, ca_chain,
                                      ca_is_lig);
    if (community_cfg_.enabled)
      write_empty_community_outputs(info_output);
    append_memory_report(info_output, "DCCM/network", "dense-skip", num_ca,
                         total_frames, estimated_extra_bytes, budget_bytes,
                         pair_count, "forced dense exceeds budget");
    std::ostringstream oss;
    oss << "DCCM Network: skipping dense all-pairs post-processing for "
        << num_ca << " nodes x " << total_frames << " frames ("
        << std::fixed << std::setprecision(1)
        << static_cast<double>(pair_count / 1.0e6L)
        << "M node pairs). Estimated extra memory "
        << format_gib(estimated_extra_bytes) << " exceeds safe budget "
        << format_gib(budget_bytes)
        << ". Use -network-mode=auto or -network-mode=sparse for a memory-safe "
           "edge-streaming network.";
    TitanUI::print_warning(oss.str());
    return;
  }
  if (run_sparse) {
    append_memory_report(info_output, "DCCM/network", "sparse", num_ca,
                         total_frames, estimated_extra_bytes, budget_bytes,
                         pair_count, force_sparse ? "forced sparse"
                                                  : "auto sparse over budget");
    std::ostringstream oss;
    oss << "DCCM Network: using sparse edge-streaming mode for " << num_ca
        << " nodes x " << total_frames << " frames (dense estimate "
        << format_gib(estimated_extra_bytes) << ", budget "
        << format_gib(budget_bytes) << ", mean-coordinate cutoff "
        << network_memory_cfg_.sparse_cut_ang << " A + top-"
        << network_memory_cfg_.sparse_k << " neighbors/node).";
    TitanUI::print_warning(oss.str());
  } else {
    append_memory_report(info_output, "DCCM/network", "dense", num_ca,
                         total_frames, estimated_extra_bytes, budget_bytes,
                         pair_count, "within budget");
  }

  // ── STEP 0: Kabsch align (guarded — skip if already aligned) ───────────────
  if (!kabsch_aligned_) {
    Matrix3Buffer ref;
    fill_centered_reference(ref, traj_x, traj_y, traj_z, num_ca);

    Matrix3Buffer mobile;
    Mat3 R;
    for (size_t f = 0; f < total_frames; ++f) {
      const Vec3 ca_centroid =
          fill_mobile_frame(mobile, traj_x, traj_y, traj_z, f, num_ca);
      TitanAlignment::align_frame_inplace(ref.const_view(), mobile.view(), R);
      for (size_t a = 0; a < num_ca; ++a) {
        traj_x[f][a] = mobile(a, 0);
        traj_y[f][a] = mobile(a, 1);
        traj_z[f][a] = mobile(a, 2);
      }
      if (!traj_cbx.empty()) {
        for (size_t a = 0; a < num_ca; ++a) {
          const Vec3 cb = titan::algebra::mul_row(
              Vec3{traj_cbx[f][a], traj_cby[f][a], traj_cbz[f][a]} -
                  ca_centroid,
              R);
          traj_cbx[f][a] = cb.x;
          traj_cby[f][a] = cb.y;
          traj_cbz[f][a] = cb.z;
        }
      }
    }
    kabsch_aligned_ = true;
  }

  if (run_sparse) {
    network_sparse_mode_active_ = true;
    const auto candidate_pairs = build_sparse_spatial_candidate_pairs(
        traj_x, traj_y, traj_z, num_ca, total_frames,
        network_memory_cfg_.sparse_cut_ang, network_memory_cfg_.sparse_k);
    TitanUI::print_info("[DCCM] Sparse candidate pairs: " +
                        std::to_string(candidate_pairs.size()) + " / " +
                        std::to_string(static_cast<unsigned long long>(
                            num_ca * (num_ca - 1) / 2)));

    std::ofstream nodes_out("network_nodes.csv");
    nodes_out << "Id,Label,Type,AtomName\n";
    for (size_t i = 0; i < num_ca; ++i) {
      nodes_out << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i)
                << "," << ca_resnrs[i] << ","
                << (ca_is_lig[i] ? "Ligand" : "Receptor") << ","
                << ca_resname[i] << "\n";
    }
    nodes_out.close();

    std::ofstream edges_out("network_edges.csv");
    edges_out << "Source,Target,Correlation\n";

    std::vector<std::tuple<int, int, double>> community_edges;
    last_network_graph_edges_.clear();
    last_community_result_ = CommunityResult{};
    if (!candidate_pairs.empty()) {
      std::vector<double> mean_x(num_ca, 0.0), mean_y(num_ca, 0.0),
          mean_z(num_ca, 0.0), var(num_ca, 0.0);
      for (size_t f = 0; f < total_frames; ++f) {
        for (size_t a = 0; a < num_ca; ++a) {
          mean_x[a] += traj_x[f][a];
          mean_y[a] += traj_y[f][a];
          mean_z[a] += traj_z[f][a];
        }
      }
      const double inv_total_frames = 1.0 / static_cast<double>(total_frames);
      for (size_t a = 0; a < num_ca; ++a) {
        mean_x[a] *= inv_total_frames;
        mean_y[a] *= inv_total_frames;
        mean_z[a] *= inv_total_frames;
      }
      for (size_t f = 0; f < total_frames; ++f) {
        for (size_t a = 0; a < num_ca; ++a) {
          const double dx = traj_x[f][a] - mean_x[a];
          const double dy = traj_y[f][a] - mean_y[a];
          const double dz = traj_z[f][a] - mean_z[a];
          var[a] += dx * dx + dy * dy + dz * dz;
        }
      }

      const size_t worker_count = std::max<size_t>(1, executor.capacity());
      const size_t chunk =
          (candidate_pairs.size() + worker_count - 1) / worker_count;
      std::vector<std::vector<std::tuple<size_t, size_t, double>>> edge_shards(
          worker_count);
      executor.parallel_do(worker_count, [&](size_t t) {
        const size_t start = t * chunk;
        const size_t end = std::min(start + chunk, candidate_pairs.size());
        if (start >= end)
          return;
        auto &local_edges = edge_shards[t];
        local_edges.reserve((end - start) / 8 + 1);
        for (size_t p = start; p < end; ++p) {
          const size_t i = candidate_pairs[p].first;
          const size_t j = candidate_pairs[p].second;
          const double denom = std::sqrt(var[i] * var[j]);
          if (denom <= 1e-12)
            continue;
          double dot = 0.0;
          const double mix = mean_x[i], miy = mean_y[i], miz = mean_z[i];
          const double mjx = mean_x[j], mjy = mean_y[j], mjz = mean_z[j];
          for (size_t f = 0; f < total_frames; ++f) {
            dot += (traj_x[f][i] - mix) * (traj_x[f][j] - mjx) +
                   (traj_y[f][i] - miy) * (traj_y[f][j] - mjy) +
                   (traj_z[f][i] - miz) * (traj_z[f][j] - mjz);
          }
          const double corr = dot / denom;
          if (std::abs(corr) > 0.4)
            local_edges.emplace_back(i, j, corr);
        }
      });

      size_t sparse_edge_count = 0;
      for (const auto &shard : edge_shards)
        sparse_edge_count += shard.size();
      last_network_graph_edges_.reserve(sparse_edge_count);
      if (community_cfg_.enabled)
        community_edges.reserve(sparse_edge_count);
      for (const auto &shard : edge_shards) {
        for (const auto &[i, j, corr] : shard) {
          edges_out << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i)
                    << ","
                    << build_network_node_id(ca_resname, ca_resnrs, ca_chain, j)
                    << "," << corr << "\n";
          last_network_graph_edges_.push_back(
              {static_cast<int>(i), static_cast<int>(j), corr, std::abs(corr)});
          if (community_cfg_.enabled) {
            community_edges.emplace_back(static_cast<int>(i),
                                         static_cast<int>(j), std::abs(corr));
          }
        }
      }
      TitanUI::print_info("[DCCM] Sparse thresholded edges: " +
                          std::to_string(sparse_edge_count));
    }
    edges_out.close();

    if (!community_cfg_.enabled)
      return;
    if (community_cfg_.method != "louvain") {
      TitanUI::print_warning("Community: skipping unsupported method '" +
                             community_cfg_.method + "'.");
      write_empty_community_outputs(info_output);
      return;
    }
    if (community_edges.empty()) {
      TitanUI::print_warning(
          "Community: sparse DCCM graph has no thresholded edges; writing "
          "header-only community outputs.");
      write_empty_community_outputs(info_output);
      return;
    }

    const auto community_graph =
        titan_sparse::WeightedCSRGraph<double>::from_undirected_edges(
            static_cast<int>(num_ca), community_edges);
    const auto louvain_result =
        titan_graph_community::run_louvain(community_graph);

    last_community_result_.membership = louvain_result.membership;
    last_community_result_.community_sizes =
        louvain_result.stats.community_sizes;
    last_community_result_.modularity = louvain_result.stats.modularity;
    last_community_result_.has_result = true;

    export_community_outputs(community_graph, louvain_result, ca_resname,
                             ca_resnrs, ca_chain, ca_is_lig, info_output);
    if (has_first_snapshot_frame_) {
      export_community_pdbs(community_graph, louvain_result, ca_indices,
                            ca_resnrs, ca_is_lig, ca_chain, first_pro_frame_,
                            first_pro_info_, first_lig_frame_,
                            first_lig_info_, info_output);
    } else {
      TitanUI::print_warning("Community: first snapshot was not cached, "
                             "skipping B-factor PDB export.");
    }
    TitanUI::print_info("[Community] Exported sparse-mode community outputs.");
    return;
  }

  // 1. Pivot Data [atom][frame] for cache locality using aligned rows.
  std::vector<double> mean_x(num_ca, 0.0), mean_y(num_ca, 0.0),
      mean_z(num_ca, 0.0);
  for (size_t f = 0; f < total_frames; ++f) {
    for (size_t a = 0; a < num_ca; ++a) {
      mean_x[a] += traj_x[f][a];
      mean_y[a] += traj_y[f][a];
      mean_z[a] += traj_z[f][a];
    }
  }
  const double inv_total_frames = 1.0 / static_cast<double>(total_frames);
  for (size_t a = 0; a < num_ca; ++a) {
    mean_x[a] *= inv_total_frames;
    mean_y[a] *= inv_total_frames;
    mean_z[a] *= inv_total_frames;
  }

  FlatAlignedMatrix dev_x(num_ca, total_frames, 0.0);
  FlatAlignedMatrix dev_y(num_ca, total_frames, 0.0);
  FlatAlignedMatrix dev_z(num_ca, total_frames, 0.0);

  const size_t prep_worker_count = std::max<size_t>(1, executor.capacity());
  const size_t atoms_per_worker =
      (num_ca + prep_worker_count - 1) / prep_worker_count;
  executor.parallel_do(prep_worker_count, [&](size_t t) {
    const size_t a_begin = t * atoms_per_worker;
    const size_t a_end = std::min(a_begin + atoms_per_worker, num_ca);
    if (a_begin >= a_end)
      return;
    for (size_t a = a_begin; a < a_end; ++a) {
      const double avg_x = mean_x[a];
      const double avg_y = mean_y[a];
      const double avg_z = mean_z[a];
      double *dx = dev_x.row_data(a);
      double *dy = dev_y.row_data(a);
      double *dz = dev_z.row_data(a);
      for (size_t f = 0; f < total_frames; ++f) {
        dx[f] = traj_x[f][a] - avg_x;
        dy[f] = traj_y[f][a] - avg_y;
        dz[f] = traj_z[f][a] - avg_z;
      }
    }
  });

  // 2. Tính Ma trận Hiệp phương sai (Multithreaded + AVX2)
  //    FIX: pair-based distribution for balanced workload,
  //         no mirror-write in workers to avoid data race.
  FlatAlignedMatrix cov(num_ca, num_ca, 0.0);

  // Worker: each thread processes a contiguous slice of pairs
  auto calc_cov_worker = [&](size_t p_start, size_t p_end) {
    for (size_t p = p_start; p < p_end; ++p) {
      auto [i, j] = upper_triangle_pair_from_index(p, num_ca);
      const double *dx_i = dev_x.row_data(i);
      const double *dx_j = dev_x.row_data(j);
      const double *dy_i = dev_y.row_data(i);
      const double *dy_j = dev_y.row_data(j);
      const double *dz_i = dev_z.row_data(i);
      const double *dz_j = dev_z.row_data(j);
      double dot_x = fast_dot_product_aligned(dx_i, dx_j, total_frames);
      double dot_y = fast_dot_product_aligned(dy_i, dy_j, total_frames);
      double dot_z = fast_dot_product_aligned(dz_i, dz_j, total_frames);

      cov(i, j) = (dot_x + dot_y + dot_z) / static_cast<double>(total_frames);
    }
  };

  // Distribute pairs equally among threads
  const size_t cov_worker_count = std::max<size_t>(1, executor.capacity());
  const size_t total_pairs = num_ca * (num_ca + 1) / 2;
  const size_t chunk = (total_pairs + cov_worker_count - 1) / cov_worker_count;
  executor.parallel_do(cov_worker_count, [&](size_t t) {
    const size_t start = t * chunk;
    const size_t end = std::min(start + chunk, total_pairs);
    if (start < end)
      calc_cov_worker(start, end);
  });

  // Single-threaded symmetry fill (safe, no race)
  for (size_t i = 0; i < num_ca; ++i)
    for (size_t j = i + 1; j < num_ca; ++j)
      cov(j, i) = cov(i, j);

  // Export the complete signed DCCM before applying the graph edge threshold.
  std::ofstream matrix_out("dccm_matrix.csv");
  matrix_out << "Residue";
  for (size_t j = 0; j < num_ca; ++j) {
    matrix_out << ","
               << build_network_node_id(ca_resname, ca_resnrs, ca_chain, j);
  }
  matrix_out << "\n";
  matrix_out << std::setprecision(10);
  for (size_t i = 0; i < num_ca; ++i) {
    matrix_out
        << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i);
    for (size_t j = 0; j < num_ca; ++j) {
      const double denom = std::sqrt(cov(i, i) * cov(j, j));
      double corr = denom > 1e-12 ? cov(i, j) / denom : 0.0;
      if (!std::isfinite(corr))
        corr = 0.0;
      corr = std::max(-1.0, std::min(1.0, corr));
      matrix_out << "," << corr;
    }
    matrix_out << "\n";
  }
  matrix_out.close();

  if (uncertainty_cfg_.mode != "off") {
    const auto blocks = titan_uncertainty::partition_blocks(
        total_frames, uncertainty_cfg_.blocks);
    std::ofstream uncertainty_out(
        info_output.get_dccm_block_uncertainty_filename());
    std::ofstream block_values_out(
        info_output.get_dccm_block_values_filename());
    uncertainty_out
        << "node_i,node_j,correlation,n_frames,n_blocks,block_mean,"
           "block_sem,ci95_low,ci95_high,bootstrap_ci95_low,"
           "bootstrap_ci95_high,positive_block_fraction,"
           "sign_consistency,status\n";
    block_values_out
        << "block,frame_begin,frame_end,node_i,node_j,correlation\n";

    if (blocks.size() < 2) {
      TitanUI::print_warning(
          "[DCCM] Block uncertainty requested, but fewer than two valid "
          "blocks are available.");
    } else {
      const std::size_t block_count = blocks.size();
      std::vector<double> block_mean_x(block_count * num_ca, 0.0);
      std::vector<double> block_mean_y(block_count * num_ca, 0.0);
      std::vector<double> block_mean_z(block_count * num_ca, 0.0);
      std::vector<double> block_variance(block_count * num_ca, 0.0);
      const auto offset = [num_ca](std::size_t block, std::size_t node) {
        return block * num_ca + node;
      };

      for (std::size_t block_index = 0; block_index < block_count;
           ++block_index) {
        const auto &block = blocks[block_index];
        const double inv_frames =
            1.0 / static_cast<double>(block.end - block.begin);
        for (std::size_t frame = block.begin; frame < block.end; ++frame) {
          for (std::size_t node = 0; node < num_ca; ++node) {
            block_mean_x[offset(block_index, node)] += traj_x[frame][node];
            block_mean_y[offset(block_index, node)] += traj_y[frame][node];
            block_mean_z[offset(block_index, node)] += traj_z[frame][node];
          }
        }
        for (std::size_t node = 0; node < num_ca; ++node) {
          block_mean_x[offset(block_index, node)] *= inv_frames;
          block_mean_y[offset(block_index, node)] *= inv_frames;
          block_mean_z[offset(block_index, node)] *= inv_frames;
        }
        for (std::size_t frame = block.begin; frame < block.end; ++frame) {
          for (std::size_t node = 0; node < num_ca; ++node) {
            const double dx =
                traj_x[frame][node] -
                block_mean_x[offset(block_index, node)];
            const double dy =
                traj_y[frame][node] -
                block_mean_y[offset(block_index, node)];
            const double dz =
                traj_z[frame][node] -
                block_mean_z[offset(block_index, node)];
            block_variance[offset(block_index, node)] +=
                dx * dx + dy * dy + dz * dz;
          }
        }
      }

      std::vector<std::pair<std::size_t, std::size_t>> node_pairs;
      node_pairs.reserve(num_ca * (num_ca - 1) / 2);
      for (std::size_t i = 0; i < num_ca; ++i) {
        for (std::size_t j = i + 1; j < num_ca; ++j)
          node_pairs.emplace_back(i, j);
      }
      std::vector<double> block_correlations(node_pairs.size() * block_count,
                                             0.0);
      const std::size_t worker_count =
          std::max<std::size_t>(1, executor.capacity());
      const std::size_t pairs_per_worker =
          (node_pairs.size() + worker_count - 1) / worker_count;
      executor.parallel_do(worker_count, [&](std::size_t worker) {
        const std::size_t begin = worker * pairs_per_worker;
        const std::size_t end =
            std::min(begin + pairs_per_worker, node_pairs.size());
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
          const auto [i, j] = node_pairs[pair_index];
          for (std::size_t block_index = 0; block_index < block_count;
               ++block_index) {
            const auto &block = blocks[block_index];
            double dot = 0.0;
            const double mix = block_mean_x[offset(block_index, i)];
            const double miy = block_mean_y[offset(block_index, i)];
            const double miz = block_mean_z[offset(block_index, i)];
            const double mjx = block_mean_x[offset(block_index, j)];
            const double mjy = block_mean_y[offset(block_index, j)];
            const double mjz = block_mean_z[offset(block_index, j)];
            for (std::size_t frame = block.begin; frame < block.end; ++frame) {
              dot += (traj_x[frame][i] - mix) *
                         (traj_x[frame][j] - mjx) +
                     (traj_y[frame][i] - miy) *
                         (traj_y[frame][j] - mjy) +
                     (traj_z[frame][i] - miz) *
                         (traj_z[frame][j] - mjz);
            }
            const double denominator =
                std::sqrt(block_variance[offset(block_index, i)] *
                          block_variance[offset(block_index, j)]);
            double correlation =
                denominator > 1.0e-12 ? dot / denominator : 0.0;
            if (!std::isfinite(correlation))
              correlation = 0.0;
            block_correlations[pair_index * block_count + block_index] =
                std::max(-1.0, std::min(1.0, correlation));
          }
        }
      });

      uncertainty_out << std::setprecision(12);
      block_values_out << std::setprecision(12);
      for (std::size_t pair_index = 0; pair_index < node_pairs.size();
           ++pair_index) {
        const auto [i, j] = node_pairs[pair_index];
        std::vector<double> values(block_count, 0.0);
        std::size_t positive = 0;
        for (std::size_t block_index = 0; block_index < block_count;
             ++block_index) {
          values[block_index] =
              block_correlations[pair_index * block_count + block_index];
          positive += values[block_index] > 0.0 ? 1U : 0U;
          block_values_out
              << block_index << ',' << blocks[block_index].begin << ','
              << blocks[block_index].end << ','
              << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i)
              << ','
              << build_network_node_id(ca_resname, ca_resnrs, ca_chain, j)
              << ',' << values[block_index] << '\n';
        }
        const auto stats = titan_uncertainty::summarize_values(values);
        const double denominator = std::sqrt(cov(i, i) * cov(j, j));
        double correlation =
            denominator > 1.0e-12 ? cov(i, j) / denominator : 0.0;
        if (!std::isfinite(correlation))
          correlation = 0.0;
        correlation = std::max(-1.0, std::min(1.0, correlation));
        double bootstrap_low = std::numeric_limits<double>::quiet_NaN();
        double bootstrap_high = std::numeric_limits<double>::quiet_NaN();
        if (uncertainty_cfg_.mode == "bootstrap") {
          const auto bootstrap = titan_uncertainty::bootstrap_ci95(
              values, uncertainty_cfg_.bootstrap_samples,
              uncertainty_cfg_.seed +
                  static_cast<unsigned int>(pair_index));
          bootstrap_low = bootstrap.first;
          bootstrap_high = bootstrap.second;
        }
        const double positive_fraction =
            static_cast<double>(positive) / static_cast<double>(block_count);
        const double sign_consistency =
            std::max(positive_fraction, 1.0 - positive_fraction);
        uncertainty_out
            << build_network_node_id(ca_resname, ca_resnrs, ca_chain, i)
            << ','
            << build_network_node_id(ca_resname, ca_resnrs, ca_chain, j)
            << ',' << correlation << ',' << total_frames << ','
            << block_count << ',' << stats.mean << ',' << stats.sem << ','
            << stats.ci95_low << ',' << stats.ci95_high << ','
            << bootstrap_low << ',' << bootstrap_high << ','
            << positive_fraction << ',' << sign_consistency << ",ok\n";
      }
      TitanUI::print_info(
          "[DCCM] Exported pairwise block uncertainty and block values.");
    }
  }

  // 3. Tính Correlation và Xuất File
  std::vector<std::tuple<int, int, double>> community_edges;
  last_network_graph_edges_.clear();
  if (community_cfg_.enabled) {
    community_edges.reserve(num_ca * 8);
  }
  std::ofstream edges_out("network_edges.csv");
  edges_out << "Source,Target,Correlation\n";
  for (size_t i = 0; i < num_ca; ++i) {
    for (size_t j = i + 1; j < num_ca; ++j) {
      double denom = std::sqrt(cov(i, i) * cov(j, j));
      if (denom > 1e-12) {
        double corr = cov(i, j) / denom;
        if (std::abs(corr) > 0.4) {
          edges_out << ca_resname[i] << ca_resnrs[i] << "_" << ca_chain[i]
                    << "," << ca_resname[j] << ca_resnrs[j] << "_"
                    << ca_chain[j] << "," << corr << "\n";
          last_network_graph_edges_.push_back(
              {static_cast<int>(i), static_cast<int>(j), corr, std::abs(corr)});
          if (community_cfg_.enabled) {
            community_edges.emplace_back(static_cast<int>(i),
                                         static_cast<int>(j), std::abs(corr));
          }
        }
      }
    }
  }
  edges_out.close();

  std::ofstream nodes_out("network_nodes.csv");
  nodes_out << "Id,Label,Type,AtomName\n";
  for (size_t i = 0; i < num_ca; ++i) {
    nodes_out << ca_resname[i] << ca_resnrs[i] << "_" << ca_chain[i] << ","
              << ca_resnrs[i] << "," << (ca_is_lig[i] ? "Ligand" : "Receptor")
              << "," << ca_resname[i] << "\n";
  }
  nodes_out.close();

  last_community_result_ = CommunityResult{};
  if (!community_cfg_.enabled)
    return;

  if (community_cfg_.method != "louvain") {
    TitanUI::print_warning("Community: skipping unsupported method '" +
                           community_cfg_.method + "'.");
    write_empty_community_outputs(info_output);
    return;
  }

  if (community_edges.empty()) {
    TitanUI::print_warning(
        "Community: thresholded DCCM graph has no edges; writing header-only "
        "community outputs.");
    write_empty_community_outputs(info_output);
    return;
  }

  const auto community_graph =
      titan_sparse::WeightedCSRGraph<double>::from_undirected_edges(
          static_cast<int>(num_ca), community_edges);
  const auto louvain_result =
      titan_graph_community::run_louvain(community_graph);

  last_community_result_.membership = louvain_result.membership;
  last_community_result_.community_sizes =
      louvain_result.stats.community_sizes;
  last_community_result_.modularity = louvain_result.stats.modularity;
  last_community_result_.has_result = true;

  export_community_outputs(community_graph, louvain_result, ca_resname,
                           ca_resnrs, ca_chain, ca_is_lig, info_output);
  if (has_first_snapshot_frame_) {
    export_community_pdbs(community_graph, louvain_result, ca_indices, ca_resnrs,
                          ca_is_lig, ca_chain, first_pro_frame_, first_pro_info_,
                          first_lig_frame_, first_lig_info_, info_output);
  } else {
    TitanUI::print_warning(
        "Community: first snapshot was not cached, skipping B-factor PDB export.");
  }
  TitanUI::print_info("[Community] Exported: network_communities.csv, "
                      "network_community_summary.csv, "
                      "network_supergraph.csv, "
                      "network_community_id_bfactor.pdb, "
                      "network_participation_coeff_bfactor.pdb");
}

// =============================================================================
// Information-Theory Allosteric Analysis: GC + Transfer Entropy
// =============================================================================
void DCCMNetwork::calculate_allostery_information(TitanExecutor &executor,
                                                  int tau_lag, int num_bins,
                                                  const std::string &gc_filename,
                                                  const std::string &mi_filename,
                                                  const std::string &te_filename) {
  AllosteryConfig cfg;
  cfg.tau_lag = tau_lag;
  cfg.num_bins = num_bins;
  calculate_allostery_information(executor, cfg, gc_filename, mi_filename,
                                  te_filename);
}

void DCCMNetwork::calculate_allostery_information(
    TitanExecutor &executor, const AllosteryConfig &cfg,
    const std::string &gc_filename, const std::string &mi_filename,
    const std::string &te_filename) {
  last_allostery_candidate_pairs_ = 0;
  last_allostery_dense_pairs_ = 0;
  last_allostery_gc_edges_.clear();
  last_allostery_te_edges_.clear();
  const int resolved_bin_count =
      resolve_allostery_bin_count(total_frames, cfg.num_bins);
  const auto dispatch_node_labels =
      build_allostery_node_labels(ca_resname, ca_resnrs, ca_chain);
  const bool use_contact_anchor_cache =
      cfg.knn_use_contact_prior &&
      has_complete_anchor_rows(traj_cbx, traj_cby, traj_cbz, ca_indices.size(),
                               total_frames);
  const auto &anchor_x =
      use_contact_anchor_cache ? traj_cbx : traj_x;
  const auto &anchor_y =
      use_contact_anchor_cache ? traj_cby : traj_y;
  const auto &anchor_z =
      use_contact_anchor_cache ? traj_cbz : traj_z;
  if (cfg.knn > 0 && cfg.knn_use_contact_prior && !use_contact_anchor_cache) {
    TitanUI::print_warning(
        "Allostery: contact-frequency kNN prior is falling back to Calpha "
        "anchors because Cbeta/N trajectory cache is unavailable.");
  }
  last_allostery_candidate_pairs_ =
      calculate_allostery_information_flat_impl<std::uint32_t>(
          traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z,
          dispatch_node_labels, total_frames, executor, cfg, resolved_bin_count,
          gc_filename, mi_filename, te_filename,
          &last_allostery_dense_pairs_, &last_allostery_gc_edges_,
          &last_allostery_te_edges_);
}

void DCCMNetwork::calculate_allostery_information_optimized(
    TitanExecutor &executor, int tau_lag, int num_bins,
    const std::string &gc_filename, const std::string &mi_filename,
    const std::string &te_filename) {
  AllosteryConfig cfg;
  cfg.tau_lag = tau_lag;
  cfg.num_bins = num_bins;
  calculate_allostery_information_optimized(executor, cfg, gc_filename,
                                            mi_filename, te_filename);
}

void DCCMNetwork::calculate_allostery_information_optimized(
    TitanExecutor &executor, const AllosteryConfig &cfg,
    const std::string &gc_filename, const std::string &mi_filename,
    const std::string &te_filename) {
  last_allostery_candidate_pairs_ = 0;
  last_allostery_dense_pairs_ = 0;
  last_allostery_gc_edges_.clear();
  last_allostery_te_edges_.clear();
  const int resolved_bin_count =
      resolve_allostery_bin_count(total_frames, cfg.num_bins);
  const auto dispatch_node_labels =
      build_allostery_node_labels(ca_resname, ca_resnrs, ca_chain);
  const bool use_contact_anchor_cache =
      cfg.knn_use_contact_prior &&
      has_complete_anchor_rows(traj_cbx, traj_cby, traj_cbz, ca_indices.size(),
                               total_frames);
  const auto &anchor_x =
      use_contact_anchor_cache ? traj_cbx : traj_x;
  const auto &anchor_y =
      use_contact_anchor_cache ? traj_cby : traj_y;
  const auto &anchor_z =
      use_contact_anchor_cache ? traj_cbz : traj_z;
  if (cfg.knn > 0 && cfg.knn_use_contact_prior && !use_contact_anchor_cache) {
    TitanUI::print_warning(
        "Allostery: contact-frequency kNN prior is falling back to Calpha "
        "anchors because Cbeta/N trajectory cache is unavailable.");
  }
  const size_t resolved_bin_count_sz =
      static_cast<size_t>(resolved_bin_count);
  if (resolved_bin_count_sz <=
      static_cast<size_t>(std::numeric_limits<std::uint8_t>::max())) {
    last_allostery_candidate_pairs_ =
        calculate_allostery_information_flat_impl<std::uint8_t>(
            traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z,
            dispatch_node_labels, total_frames, executor, cfg,
            resolved_bin_count, gc_filename, mi_filename,
            te_filename, &last_allostery_dense_pairs_,
            &last_allostery_gc_edges_, &last_allostery_te_edges_);
  } else if (resolved_bin_count_sz <=
             static_cast<size_t>(std::numeric_limits<std::uint16_t>::max())) {
    last_allostery_candidate_pairs_ =
        calculate_allostery_information_flat_impl<std::uint16_t>(
            traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z,
            dispatch_node_labels, total_frames, executor, cfg,
            resolved_bin_count, gc_filename, mi_filename,
            te_filename, &last_allostery_dense_pairs_,
            &last_allostery_gc_edges_, &last_allostery_te_edges_);
  } else {
    last_allostery_candidate_pairs_ =
        calculate_allostery_information_flat_impl<std::uint32_t>(
            traj_x, traj_y, traj_z, anchor_x, anchor_y, anchor_z,
            dispatch_node_labels, total_frames, executor, cfg,
            resolved_bin_count, gc_filename, mi_filename,
            te_filename, &last_allostery_dense_pairs_,
            &last_allostery_gc_edges_, &last_allostery_te_edges_);
  }
}
// =============================================================================
// LSP-MD: Local Spatial Pattern (Kornev, JCTC 2025, 21, 8699)
// =============================================================================

// ── Brandes weighted betweenness centrality (Dijkstra variant) ────────────────
namespace {
void brandes_weighted_bc(
    const titan_sparse::WeightedCSRGraph<double> &graph,
    std::vector<double> &bc_out, TitanExecutor &executor) {
  const size_t n = static_cast<size_t>(graph.node_count());
  const size_t worker_count = std::max<size_t>(1, executor.capacity());
  bc_out.assign(n, 0.0);

  // Per-thread partial sums (avoids false sharing / contention)
  FlatAlignedMatrix partial(worker_count, n, 0.0);

  auto worker = [&](size_t t, size_t s_start, size_t s_end) {
    double *bc_local = partial.row_data(t);
    // Allocate once per thread, reuse across source iterations
    std::vector<double> dist(n), sigma(n), delta(n);
    std::vector<std::vector<size_t>> pred(n);
    for (size_t node = 0; node < n; ++node)
      pred[node].reserve(
          static_cast<size_t>(std::max(1, graph.degree(static_cast<int>(node)))));
    std::vector<size_t> order;
    order.reserve(n);

    for (size_t s = s_start; s < s_end; ++s) {
      std::fill(dist.begin(), dist.end(), 1e30);
      std::fill(sigma.begin(), sigma.end(), 0.0);
      std::fill(delta.begin(), delta.end(), 0.0);
      for (auto &p : pred) p.clear(); // clear keeps capacity
      order.clear();

      using PQEntry = std::pair<double, size_t>;
      std::priority_queue<PQEntry, std::vector<PQEntry>,
                          std::greater<PQEntry>> pq;

      dist[s] = 0.0;
      sigma[s] = 1.0;
      pq.push({0.0, s});

      while (!pq.empty()) {
        auto [d_u, u] = pq.top();
        pq.pop();
        if (d_u > dist[u])
          continue;
        order.push_back(u);
        for (int edge_idx = graph.row_begin(static_cast<int>(u));
             edge_idx < graph.row_end(static_cast<int>(u)); ++edge_idx) {
          const size_t v =
              static_cast<size_t>(graph.col_index(edge_idx));
          const double w = graph.value(edge_idx);
          double alt = dist[u] + w;
          if (alt < dist[v] - 1e-12) {
            dist[v] = alt;
            sigma[v] = sigma[u];
            pred[v].clear();
            pred[v].push_back(u);
            pq.push({alt, v});
          } else if (std::abs(alt - dist[v]) < 1e-12) {
            sigma[v] += sigma[u];
            pred[v].push_back(u);
          }
        }
      }

      // Back-propagation of dependencies
      for (auto it = order.rbegin(); it != order.rend(); ++it) {
        size_t w = *it;
        for (size_t p : pred[w]) {
          double coeff = (sigma[p] / sigma[w]) * (1.0 + delta[w]);
          delta[p] += coeff;
        }
        if (w != s) {
          bc_local[w] += delta[w];
        }
      }
    }
  };

  // Distribute source nodes across threads
  const size_t chunk = (n + worker_count - 1) / worker_count;
  executor.parallel_do(worker_count, [&](size_t t) {
    const size_t s = t * chunk;
    const size_t e = std::min(s + chunk, n);
    if (s < e)
      worker(t, s, e);
  });

  // Reduce partial sums
  for (size_t t = 0; t < worker_count; ++t) {
    const double *partial_row = partial.row_data(t);
    for (size_t i = 0; i < n; ++i)
      bc_out[i] += partial_row[i];
  }

  // Normalize for undirected graph
  if (n > 2) {
    double norm = static_cast<double>((n - 1) * (n - 2)) / 2.0;
    for (auto &v : bc_out)
      v /= norm;
  }
}
} // anonymous namespace

// ── AVX2: compute 4 distances in parallel ────────────────────────────────────
void DCCMNetwork::calculate_lspmd(TitanExecutor &executor,
                                  const LSPMDConfig &cfg,
                                  LSPMDResult &result) {
  if (total_frames == 0 || ca_indices.empty())
    return;

  if (total_frames < 10) {
    std::cerr << "[LSP-MD] WARNING: Too few frames (" << total_frames
              << " < 10), results may be unreliable.\n";
  }

  size_t num_ca = ca_indices.size();
  double cutoff = cfg.cutoff_ang;
  const auto has_full_cb_cache = [&]() {
    if (traj_cbx.size() != total_frames || traj_cby.size() != total_frames ||
        traj_cbz.size() != total_frames) {
      return false;
    }
    for (size_t f = 0; f < total_frames; ++f) {
      if (traj_cbx[f].size() < num_ca || traj_cby[f].size() < num_ca ||
          traj_cbz[f].size() < num_ca) {
        return false;
      }
    }
    return true;
  };
  const bool use_cb_cache = has_full_cb_cache();
  if (!use_cb_cache) {
    TitanUI::print_warning(
        "LSP-MD: Cbeta/N trajectory cache is unavailable or incomplete. "
        "Falling back to Calpha coordinates for side-chain anchors.");
  }

  // ── STEP 0: Kabsch alignment (guarded — skip if already aligned) ───────────
  if (!kabsch_aligned_) {
    Matrix3Buffer ref;
    fill_centered_reference(ref, traj_x, traj_y, traj_z, num_ca);

    Matrix3Buffer mobile;
    Mat3 R;
    for (size_t f = 0; f < total_frames; ++f) {
      const Vec3 ca_centroid =
          fill_mobile_frame(mobile, traj_x, traj_y, traj_z, f, num_ca);
      TitanAlignment::align_frame_inplace(ref.const_view(), mobile.view(), R);
      for (size_t a = 0; a < num_ca; ++a) {
        traj_x[f][a] = mobile(a, 0);
        traj_y[f][a] = mobile(a, 1);
        traj_z[f][a] = mobile(a, 2);
      }
      if (!traj_cbx.empty()) {
        for (size_t a = 0; a < num_ca; ++a) {
          const Vec3 cb = titan::algebra::mul_row(
              Vec3{traj_cbx[f][a], traj_cby[f][a], traj_cbz[f][a]} -
                  ca_centroid,
              R);
          traj_cbx[f][a] = cb.x;
          traj_cby[f][a] = cb.y;
          traj_cbz[f][a] = cb.z;
        }
      }
    }
    kabsch_aligned_ = true;
  }

  // ── Pre-transpose to atom-major layout [atom][frame] for cache locality ────
  size_t F = total_frames;
  FlatAlignedMatrix ax(num_ca, F, 0.0);
  FlatAlignedMatrix ay(num_ca, F, 0.0);
  FlatAlignedMatrix az(num_ca, F, 0.0);
  FlatAlignedMatrix bx(num_ca, F, 0.0);
  FlatAlignedMatrix by(num_ca, F, 0.0);
  FlatAlignedMatrix bz(num_ca, F, 0.0);

  for (size_t f = 0; f < F; ++f)
    for (size_t a = 0; a < num_ca; ++a) {
      const double x = traj_x[f][a];
      const double y = traj_y[f][a];
      const double z = traj_z[f][a];
      ax(a, f) = x;
      ay(a, f) = y;
      az(a, f) = z;
      if (use_cb_cache) {
        bx(a, f) = traj_cbx[f][a];
        by(a, f) = traj_cby[f][a];
        bz(a, f) = traj_cbz[f][a];
      } else {
        bx(a, f) = ax(a, f);
        by(a, f) = ay(a, f);
        bz(a, f) = az(a, f);
      }
    }

  // ── STEP 1: Build upper-triangle pair list and compute W_ij ────────────────
  std::vector<std::pair<size_t, size_t>> pairs = build_lspmd_candidate_pairs(
      traj_x, traj_y, traj_z, num_ca, total_frames, cutoff,
      cfg.use_avg_cutoff);
  if (pairs.empty()) {
    result.edges.clear();
    result.dc.assign(num_ca, 0.0);
    result.bc.assign(num_ca, 0.0);
    return;
  }

  const size_t n_pairs = pairs.size();
  std::vector<double> W_vals(n_pairs,
      std::numeric_limits<double>::quiet_NaN());

  auto worker = [&](size_t p_start, size_t p_end) {
    AlignedDoubleVector d1v(F, 0.0), d2v(F, 0.0), d3v(F, 0.0), d4v(F, 0.0);
    for (size_t p = p_start; p < p_end; ++p) {
      size_t i = pairs[p].first;
      size_t j = pairs[p].second;

      // Pointers into transposed arrays — sequential access in inner loop
      const double *axi = ax.row_data(i), *ayi = ay.row_data(i),
                   *azi = az.row_data(i);
      const double *bxi = bx.row_data(i), *byi = by.row_data(i),
                   *bzi = bz.row_data(i);
      const double *axj = ax.row_data(j), *ayj = ay.row_data(j),
                   *azj = az.row_data(j);
      const double *bxj = bx.row_data(j), *byj = by.row_data(j),
                   *bzj = bz.row_data(j);

      // ── Single pass: compute distances + cache for deviation ──
      double sum_d1 = 0.0, sum_d2 = 0.0, sum_d3 = 0.0, sum_d4 = 0.0;
      for (size_t f = 0; f < F; ++f) {
        titan_simd_dispatch::compute_4dist(
            axi[f], ayi[f], azi[f], bxi[f], byi[f], bzi[f], axj[f], ayj[f],
            azj[f], bxj[f], byj[f], bzj[f], d1v[f], d2v[f], d3v[f], d4v[f]);
        sum_d1 += d1v[f]; sum_d2 += d2v[f];
        sum_d3 += d3v[f]; sum_d4 += d4v[f];
      }

      double inv_f = 1.0 / static_cast<double>(F);
      double avg_d1 = sum_d1 * inv_f;
      double avg_d2 = sum_d2 * inv_f;
      double avg_d3 = sum_d3 * inv_f;
      double avg_d4 = sum_d4 * inv_f;

      // ── Cutoff check ──
      if (cfg.use_avg_cutoff && avg_d1 > cutoff)
        continue;

      // ── Reuse cached distances for deviation ──
      double sum_Z2 = 0.0;
      for (size_t f = 0; f < F; ++f) {
        double dd1 = d1v[f] - avg_d1, dd2 = d2v[f] - avg_d2;
        double dd3 = d3v[f] - avg_d3, dd4 = d4v[f] - avg_d4;
        sum_Z2 += dd1*dd1 + dd2*dd2 + dd3*dd3 + dd4*dd4;
      }

      double Z_avg = std::sqrt(sum_Z2 * inv_f);
      W_vals[p] = std::exp(-Z_avg);
    }
  };

  // ── Launch multithreaded pair computation ──
  {
    const size_t worker_count = std::max<size_t>(1, executor.capacity());
    const size_t chunk = (n_pairs + worker_count - 1) / worker_count;
    executor.parallel_do(worker_count, [&](size_t t) {
      const size_t start = t * chunk;
      const size_t end = std::min(start + chunk, n_pairs);
      if (start < end)
        worker(start, end);
    });
  }

  // ── Collect valid edges (NaN = skipped) ──
  result.edges.clear();
  for (size_t p = 0; p < n_pairs; ++p) {
    if (!std::isnan(W_vals[p])) {
      result.edges.emplace_back(static_cast<int>(pairs[p].first),
                                static_cast<int>(pairs[p].second),
                                W_vals[p]);
    }
  }

  // ── STEP 2: Degree Centrality ──────────────────────────────────────────────
  std::sort(result.edges.begin(), result.edges.end(),
            [](const auto &lhs, const auto &rhs) {
              if (std::get<0>(lhs) != std::get<0>(rhs))
                return std::get<0>(lhs) < std::get<0>(rhs);
              if (std::get<1>(lhs) != std::get<1>(rhs))
                return std::get<1>(lhs) < std::get<1>(rhs);
              return std::get<2>(lhs) > std::get<2>(rhs);
            });

  result.dc.assign(num_ca, 0.0);
  for (auto &[i, j, w] : result.edges) {
    result.dc[i] += w;
    result.dc[j] += w;
  }

  // ── STEP 3: Betweenness Centrality (Brandes + Dijkstra) ────────────────────
  // Build adjacency list with D_ij = -ln(W_ij) as distance
  std::vector<std::tuple<int, int, double>> weighted_edges;
  weighted_edges.reserve(result.edges.size());
  for (auto &[i, j, w] : result.edges) {
    const double d_weight = (w > 1e-15) ? -std::log(w) : 35.0;
    weighted_edges.emplace_back(i, j, d_weight);
  }
  const auto weighted_graph =
      titan_sparse::WeightedCSRGraph<double>::from_undirected_edges(
          static_cast<int>(num_ca), weighted_edges);

  brandes_weighted_bc(weighted_graph, result.bc, executor);
}

void DCCMNetwork::export_lspmd(const LSPMDResult &result,
                                double weight_threshold) {
  const auto &resname = ca_resname;
  const auto &resnrs = ca_resnrs;
  const auto &chains = ca_chain;
  size_t N = resname.size();

  // --- 1. Edges ---
  {
    std::ofstream ofs("lspmd_edges.csv");
    ofs << "Source,Target,Z_avg,Weight,D_ij\n";
    ofs << std::fixed << std::setprecision(6);
    for (const auto &[i, j, W] : result.edges) {
      if (W < weight_threshold)
        continue;
      double Z = -std::log(W > 1e-15 ? W : 1e-15);
      double D = Z; // D_ij = -ln(W_ij) = <Z>
      ofs << resname[i] << resnrs[i] << "_" << chains[i] << ","
          << resname[j] << resnrs[j] << "_" << chains[j] << "," << Z << ","
          << W << "," << D << "\n";
    }
  }

  // --- 2. Degree Centrality ---
  {
    double dc_max = *std::max_element(result.dc.begin(), result.dc.end());
    if (dc_max < 1e-15)
      dc_max = 1.0;
    std::ofstream ofs("lspmd_degree_centrality.csv");
    ofs << "Residue,ResNr,Chain,IsLig,DC,DC_norm\n";
    ofs << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < N; ++i) {
      ofs << resname[i] << "," << resnrs[i] << "," << chains[i] << ","
          << ca_is_lig[i] << "," << result.dc[i] << ","
          << result.dc[i] / dc_max << "\n";
    }
  }

  // --- 3. Betweenness Centrality ---
  {
    double bc_max = *std::max_element(result.bc.begin(), result.bc.end());
    if (bc_max < 1e-15)
      bc_max = 1.0;
    std::ofstream ofs("lspmd_betweenness_centrality.csv");
    ofs << "Residue,ResNr,Chain,IsLig,BC,BC_norm\n";
    ofs << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < N; ++i) {
      ofs << resname[i] << "," << resnrs[i] << "," << chains[i] << ","
          << ca_is_lig[i] << "," << result.bc[i] << ","
          << result.bc[i] / bc_max << "\n";
    }
  }

  TitanUI::print_info("[LSP-MD] Exported: lspmd_edges.csv, "
                      "lspmd_degree_centrality.csv, "
                      "lspmd_betweenness_centrality.csv");
}

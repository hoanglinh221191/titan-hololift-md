#include "titan_energy_decomp_engine.h"

#include "Multi_MOL.h"
#include "aligned_allocator.h"
#include "cache_sigma_epsilon.h"
#include "titan_energy_decomposition.h"
#include "titan_executor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace titan_energy_decomp {
namespace {

struct EnergySums {
  double elec{0.0};
  double elec_abs{0.0};
  double vdw{0.0};
  double vdw_abs{0.0};
  double total{0.0};
  double total_abs{0.0};
  std::size_t samples{0};

  void add(double frame_elec, double frame_vdw) noexcept {
    const double frame_total = frame_elec + frame_vdw;
    elec += frame_elec;
    elec_abs += std::fabs(frame_elec);
    vdw += frame_vdw;
    vdw_abs += std::fabs(frame_vdw);
    total += frame_total;
    total_abs += std::fabs(frame_total);
    ++samples;
  }

  void add(const PairStatistics &stats) noexcept {
    elec += stats.elec_signed_mean;
    elec_abs += stats.elec_mean_abs;
    vdw += stats.vdw_signed_mean;
    vdw_abs += stats.vdw_mean_abs;
    total += stats.signed_mean;
    total_abs += stats.mean_abs;
    ++samples;
  }
};

struct EndpointSums {
  double e_elec{0.0};
  double e_vdw{0.0};
  double g_polar_gb{0.0};
  double g_nonpolar_sasa{0.0};
  double delta_g{0.0};
  double delta_g_abs{0.0};
  double pooled_frame_mean{0.0};
  std::size_t samples{0};

  void add(double elec, double vdw, double polar, double nonpolar) noexcept {
    const double mm = elec + vdw;
    const double total = mm + polar + nonpolar;
    e_elec += elec;
    e_vdw += vdw;
    g_polar_gb += polar;
    g_nonpolar_sasa += nonpolar;
    delta_g += total;
    delta_g_abs += std::fabs(total);
    pooled_frame_mean += total;
    ++samples;
  }

  void add(const EndpointEntityStatistics &stats) noexcept {
    e_elec += stats.e_elec_mean;
    e_vdw += stats.e_vdw_mean;
    g_polar_gb += stats.g_polar_gb_mean;
    g_nonpolar_sasa += stats.g_nonpolar_sasa_mean;
    delta_g += stats.delta_g_no_entropy_mean;
    delta_g_abs += stats.mean_abs;
    pooled_frame_mean += stats.pooled_frame_mean;
    ++samples;
  }
};

struct LevelAccumulators {
  std::size_t rows{0};
  std::size_t cols{0};
  std::vector<EnergySums> pairs;
  std::vector<EnergySums> row_rankings;
  std::vector<EnergySums> col_rankings;

  LevelAccumulators() = default;
  LevelAccumulators(std::size_t row_count, std::size_t col_count)
      : rows{row_count}, cols{col_count}, pairs(row_count * col_count),
        row_rankings(row_count), col_rankings(col_count) {}
};

struct FrameScratch {
  AlignedDoubleVector atom_elec;
  AlignedDoubleVector atom_vdw;
  AlignedDoubleVector residue_elec;
  AlignedDoubleVector residue_vdw;
  AlignedDoubleVector mol1_atom_elec;
  AlignedDoubleVector mol1_atom_vdw;
  AlignedDoubleVector mol2_atom_elec;
  AlignedDoubleVector mol2_atom_vdw;
  AlignedDoubleVector mol1_residue_elec;
  AlignedDoubleVector mol1_residue_vdw;
  AlignedDoubleVector mol2_residue_elec;
  AlignedDoubleVector mol2_residue_vdw;
};

struct EndpointLevelAccumulators {
  std::vector<EndpointSums> mol1;
  std::vector<EndpointSums> mol2;

  EndpointLevelAccumulators() = default;
  EndpointLevelAccumulators(std::size_t mol1_count, std::size_t mol2_count)
      : mol1(mol1_count), mol2(mol2_count) {}
};

PairStatistics finish(const EnergySums &sum) {
  PairStatistics out;
  if (sum.samples == 0)
    return out;
  const double denominator = static_cast<double>(sum.samples);
  out.elec_signed_mean = sum.elec / denominator;
  out.elec_abs_mean = std::fabs(out.elec_signed_mean);
  out.elec_mean_abs = sum.elec_abs / denominator;
  out.vdw_signed_mean = sum.vdw / denominator;
  out.vdw_abs_mean = std::fabs(out.vdw_signed_mean);
  out.vdw_mean_abs = sum.vdw_abs / denominator;
  out.signed_mean = sum.total / denominator;
  out.abs_mean = std::fabs(out.signed_mean);
  out.mean_abs = sum.total_abs / denominator;
  return out;
}

EndpointEntityStatistics finish_endpoint(const EndpointSums &sum) {
  EndpointEntityStatistics out;
  if (sum.samples == 0)
    return out;
  const double denominator = static_cast<double>(sum.samples);
  out.e_elec_mean = sum.e_elec / denominator;
  out.e_vdw_mean = sum.e_vdw / denominator;
  out.e_mm_mean = out.e_elec_mean + out.e_vdw_mean;
  out.g_polar_gb_mean = sum.g_polar_gb / denominator;
  out.g_nonpolar_sasa_mean = sum.g_nonpolar_sasa / denominator;
  out.delta_g_no_entropy_mean = sum.delta_g / denominator;
  out.abs_mean = std::fabs(out.delta_g_no_entropy_mean);
  out.mean_abs = sum.delta_g_abs / denominator;
  out.pooled_frame_mean = sum.pooled_frame_mean / denominator;
  out.samples = sum.samples;
  return out;
}

double metric_value(const PairStatistics &stats, RankingMetric metric) {
  switch (metric) {
  case RankingMetric::SignedMean:
    return stats.signed_mean;
  case RankingMetric::AbsMean:
    return stats.abs_mean;
  case RankingMetric::MeanAbs:
    return stats.mean_abs;
  }
  return stats.mean_abs;
}

bool value_before(double lhs, double rhs, SortDirection direction) {
  return direction == SortDirection::Descending ? lhs > rhs : lhs < rhs;
}

DenseMatrix finish_matrix(const LevelAccumulators &level) {
  DenseMatrix out;
  out.rows = level.rows;
  out.cols = level.cols;
  out.values.reserve(level.pairs.size());
  for (const auto &sum : level.pairs)
    out.values.push_back(finish(sum));
  return out;
}

std::vector<SparseEdge> finish_edges(const DenseMatrix &matrix,
                                     const EngineConfig &config) {
  std::vector<SparseEdge> edges;
  edges.reserve(matrix.values.size());
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      const PairStatistics &stats = matrix.values[row * matrix.cols + col];
      if (stats.elec_mean_abs == 0.0 && stats.vdw_mean_abs == 0.0)
        continue;
      edges.push_back({row, col, stats});
    }
  }
  std::stable_sort(edges.begin(), edges.end(), [&](const auto &lhs,
                                                   const auto &rhs) {
    const double lhs_value = metric_value(lhs.stats, config.edge_metric);
    const double rhs_value = metric_value(rhs.stats, config.edge_metric);
    if (lhs_value != rhs_value)
      return value_before(lhs_value, rhs_value, config.sort_direction);
    if (lhs.row != rhs.row)
      return lhs.row < rhs.row;
    return lhs.col < rhs.col;
  });
  if (edges.size() > config.top_edge_count)
    edges.resize(config.top_edge_count);
  return edges;
}

std::vector<EntityRanking>
finish_rankings(const std::vector<EnergySums> &sums,
                const EngineConfig &config) {
  std::vector<EntityRanking> out;
  out.reserve(sums.size());
  for (std::size_t i = 0; i < sums.size(); ++i)
    out.push_back({i, finish(sums[i])});
  std::stable_sort(out.begin(), out.end(), [&](const auto &lhs,
                                              const auto &rhs) {
    const double lhs_value = metric_value(lhs.stats, config.ranking_metric);
    const double rhs_value = metric_value(rhs.stats, config.ranking_metric);
    if (lhs_value != rhs_value)
      return value_before(lhs_value, rhs_value, config.sort_direction);
    return lhs.entity_index < rhs.entity_index;
  });
  return out;
}

std::string residue_label(std::string_view role, const mol_info &info) {
  return std::string(role) + "|" + info.get_chain() + ":" +
         std::to_string(info.get_resnr()) + ":" + info.get_res_name();
}

std::string atom_label(std::string_view role, const mol_info &info,
                       std::size_t atom_index) {
  return residue_label(role, info) + ":" + info.get_atom() + "#" +
         std::to_string(atom_index);
}

std::string residue_key(const mol_info &info) {
  return info.get_chain() + "\x1f" + std::to_string(info.get_chainnum()) +
         "\x1f" + std::to_string(info.get_respos()) + "\x1f" +
         std::to_string(info.get_resnr()) + "\x1f" + info.get_res_name();
}

void validate_axes(const EntityAxes &axes) {
  if (axes.atoms.kind != EntityKind::Atom ||
      axes.residues.kind != EntityKind::Residue) {
    throw std::invalid_argument("energy decomposition axes have wrong kind");
  }
  if (axes.atoms.entities.size() != axes.atoms.atom_to_entity.size() ||
      axes.residues.atom_to_entity.size() != axes.atoms.entities.size()) {
    throw std::invalid_argument(
        "energy decomposition atom mappings do not match atom axis");
  }
  for (const std::size_t residue : axes.residues.atom_to_entity) {
    if (residue >= axes.residues.entities.size()) {
      throw std::out_of_range(
          "energy decomposition residue mapping is out of range");
    }
  }
}

void add_level_frame(LevelAccumulators &level,
                     const AlignedDoubleVector &pair_elec,
                     const AlignedDoubleVector &pair_vdw,
                     const AlignedDoubleVector &row_elec,
                     const AlignedDoubleVector &row_vdw,
                     const AlignedDoubleVector &col_elec,
                     const AlignedDoubleVector &col_vdw) {
  for (std::size_t i = 0; i < level.pairs.size(); ++i)
    level.pairs[i].add(pair_elec[i], pair_vdw[i]);
  for (std::size_t i = 0; i < level.row_rankings.size(); ++i)
    level.row_rankings[i].add(row_elec[i], row_vdw[i]);
  for (std::size_t i = 0; i < level.col_rankings.size(); ++i)
    level.col_rankings[i].add(col_elec[i], col_vdw[i]);
}

void add_level_summary(LevelAccumulators &destination,
                       const LevelAccumulators &source) {
  for (std::size_t i = 0; i < destination.pairs.size(); ++i)
    destination.pairs[i].add(finish(source.pairs[i]));
  for (std::size_t i = 0; i < destination.row_rankings.size(); ++i)
    destination.row_rankings[i].add(finish(source.row_rankings[i]));
  for (std::size_t i = 0; i < destination.col_rankings.size(); ++i)
    destination.col_rankings[i].add(finish(source.col_rankings[i]));
}

void add_endpoint_summary(EndpointLevelAccumulators &destination,
                          const EndpointLevelAccumulators &source) {
  for (std::size_t i = 0; i < destination.mol1.size(); ++i)
    destination.mol1[i].add(finish_endpoint(source.mol1[i]));
  for (std::size_t i = 0; i < destination.mol2.size(); ++i)
    destination.mol2[i].add(finish_endpoint(source.mol2[i]));
}

std::string csv_field(const std::string &text) {
  if (text.find_first_of(",\"\n\r") == std::string::npos)
    return text;
  std::string escaped{"\""};
  for (const char ch : text) {
    escaped.push_back(ch);
    if (ch == '"')
      escaped.push_back('"');
  }
  escaped.push_back('"');
  return escaped;
}

void write_stats(std::ostream &out, const PairStatistics &stats) {
  out << stats.elec_signed_mean << ',' << stats.elec_abs_mean << ','
      << stats.elec_mean_abs << ',' << stats.vdw_signed_mean << ','
      << stats.vdw_abs_mean << ',' << stats.vdw_mean_abs << ','
      << stats.signed_mean << ',' << stats.abs_mean << ',' << stats.mean_abs;
}

double endpoint_metric_value(const EndpointEntityStatistics &stats,
                             RankingMetric metric) {
  switch (metric) {
  case RankingMetric::SignedMean:
    return stats.delta_g_no_entropy_mean;
  case RankingMetric::AbsMean:
    return stats.abs_mean;
  case RankingMetric::MeanAbs:
    return stats.mean_abs;
  }
  return stats.abs_mean;
}

void write_endpoint_stats(std::ostream &out,
                          const EndpointEntityStatistics &stats) {
  out << stats.e_elec_mean << ',' << stats.e_vdw_mean << ','
      << stats.e_mm_mean << ',' << stats.g_polar_gb_mean << ','
      << stats.g_nonpolar_sasa_mean << ','
      << stats.delta_g_no_entropy_mean << ',' << stats.abs_mean << ','
      << stats.mean_abs << ',' << stats.pooled_frame_mean << ','
      << stats.samples;
}

struct EndpointOutputLabels {
  std::string title;
  std::string polar;
  std::string nonpolar;
  std::string semantics_note;
};

EndpointOutputLabels endpoint_output_labels(std::string_view source) {
  if (source == "mmpbsa") {
    return {"MM/PBSA endpoint decomposition summary",
            "DeltaDeltaG_solv_PB", "DeltaG_nonpolar_SASA",
            "MM/PBSA maps are endpoint entity decompositions and must not be "
            "interpreted as pair matrices."};
  }
  return {"MM/GBSA endpoint decomposition summary", "G_polar_GB",
          "G_nonpolar_SASA",
          "MM/GBSA maps are endpoint entity decompositions and must not be "
          "interpreted as pair matrices."};
}

void write_component_matrix(const std::filesystem::path &path,
                            const DenseMatrix &matrix,
                            const EntityAxis &rows, const EntityAxis &cols,
                            std::string_view component) {
  std::ofstream out(path);
  out << "row_index,row_label,col_index,col_label,component,signed_mean,"
         "abs_mean,mean_abs\n";
  out << std::fixed << std::setprecision(10);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      const PairStatistics &stats = matrix.values[row * matrix.cols + col];
      double signed_mean = stats.signed_mean;
      double abs_mean = stats.abs_mean;
      double mean_abs = stats.mean_abs;
      if (component == "elec") {
        signed_mean = stats.elec_signed_mean;
        abs_mean = stats.elec_abs_mean;
        mean_abs = stats.elec_mean_abs;
      } else if (component == "vdw") {
        signed_mean = stats.vdw_signed_mean;
        abs_mean = stats.vdw_abs_mean;
        mean_abs = stats.vdw_mean_abs;
      }
      out << row << ',' << csv_field(rows.entities[row].label) << ',' << col
          << ',' << csv_field(cols.entities[col].label) << ',' << component
          << ',' << signed_mean << ',' << abs_mean << ',' << mean_abs << '\n';
    }
  }
}

void write_edges(const std::filesystem::path &path,
                 const std::vector<SparseEdge> &edges, const EntityAxis &rows,
                 const EntityAxis &cols) {
  std::ofstream out(path);
  out << "rank,row_index,row_label,col_index,col_label,elec_signed_mean,"
         "elec_abs_mean,elec_mean_abs,vdw_signed_mean,vdw_abs_mean,"
         "vdw_mean_abs,signed_mean,abs_mean,mean_abs\n";
  out << std::fixed << std::setprecision(10);
  for (std::size_t rank = 0; rank < edges.size(); ++rank) {
    const auto &edge = edges[rank];
    out << rank + 1 << ',' << edge.row << ','
        << csv_field(rows.entities[edge.row].label) << ',' << edge.col << ','
        << csv_field(cols.entities[edge.col].label) << ',';
    write_stats(out, edge.stats);
    out << '\n';
  }
}

struct RankingRow {
  std::string role;
  const EntityAxis *axis{nullptr};
  EntityRanking ranking;
};

void write_combined_rankings(const std::filesystem::path &path,
                             const std::vector<EntityRanking> &mol1_rankings,
                             const EntityAxis &mol1_axis,
                             const std::vector<EntityRanking> &mol2_rankings,
                             const EntityAxis &mol2_axis,
                             const EngineConfig &config) {
  std::vector<RankingRow> rows;
  rows.reserve(mol1_rankings.size() + mol2_rankings.size());
  for (const auto &ranking : mol1_rankings)
    rows.push_back({"mol1", &mol1_axis, ranking});
  for (const auto &ranking : mol2_rankings)
    rows.push_back({"mol2", &mol2_axis, ranking});
  std::stable_sort(rows.begin(), rows.end(), [&](const auto &lhs,
                                                 const auto &rhs) {
    const double lhs_value = metric_value(lhs.ranking.stats,
                                          config.ranking_metric);
    const double rhs_value = metric_value(rhs.ranking.stats,
                                          config.ranking_metric);
    if (lhs_value != rhs_value)
      return value_before(lhs_value, rhs_value, config.sort_direction);
    if (lhs.role != rhs.role)
      return lhs.role < rhs.role;
    return lhs.ranking.entity_index < rhs.ranking.entity_index;
  });

  std::ofstream out(path);
  out << "rank,molecule_role,entity_index,entity_label,elec_signed_mean,"
         "elec_abs_mean,elec_mean_abs,vdw_signed_mean,vdw_abs_mean,"
         "vdw_mean_abs,signed_mean,abs_mean,mean_abs\n";
  out << std::fixed << std::setprecision(10);
  for (std::size_t rank = 0; rank < rows.size(); ++rank) {
    const auto &row = rows[rank];
    out << rank + 1 << ',' << row.role << ',' << row.ranking.entity_index
        << ','
        << csv_field(row.axis->entities[row.ranking.entity_index].label)
        << ',';
    write_stats(out, row.ranking.stats);
    out << '\n';
  }
}

void write_endpoint_map(const std::filesystem::path &path,
                        const std::vector<EndpointEntityResultRow> &rows,
                        const EndpointOutputLabels &labels) {
  std::ofstream out(path);
  out << "molecule_role,entity_index,entity_label,E_elec,E_vdw,E_MM,"
      << labels.polar << ',' << labels.nonpolar
      << ",DeltaG_no_entropy,abs_mean,mean_abs,pooled_frame_mean,samples\n";
  out << std::fixed << std::setprecision(10);
  for (const auto &row : rows) {
    out << row.molecule_role << ',' << row.entity_index << ','
        << csv_field(row.metadata.label) << ',';
    write_endpoint_stats(out, row.stats);
    out << '\n';
  }
}

void write_endpoint_ranking(
    const std::filesystem::path &path,
    const std::vector<EndpointEntityResultRow> &rows,
    const EndpointOutputLabels &labels) {
  std::ofstream out(path);
  out << "rank,molecule_role,entity_index,entity_label,E_elec,E_vdw,E_MM,"
      << labels.polar << ',' << labels.nonpolar
      << ",DeltaG_no_entropy,abs_mean,mean_abs,pooled_frame_mean,samples\n";
  out << std::fixed << std::setprecision(10);
  for (std::size_t rank = 0; rank < rows.size(); ++rank) {
    const auto &row = rows[rank];
    out << rank + 1 << ',' << row.molecule_role << ',' << row.entity_index
        << ',' << csv_field(row.metadata.label) << ',';
    write_endpoint_stats(out, row.stats);
    out << '\n';
  }
}

} // namespace

struct EnergyDecompositionEngine::Accumulators {
  std::size_t frames{0};
  LevelAccumulators residues;
  LevelAccumulators atoms;

  Accumulators(const EntityAxes &mol1, const EntityAxes &mol2)
      : residues(mol2.residues.entities.size(),
                 mol1.residues.entities.size()),
        atoms(mol2.atoms.entities.size(), mol1.atoms.entities.size()) {}
};

struct EndpointEntityDecompositionEngine::Accumulators {
  std::size_t frames{0};
  EndpointLevelAccumulators residues;
  EndpointLevelAccumulators atoms;

  Accumulators(const EntityAxes &mol1, const EntityAxes &mol2)
      : residues(mol1.residues.entities.size(),
                 mol2.residues.entities.size()),
        atoms(mol1.atoms.entities.size(), mol2.atoms.entities.size()) {}
};

EntityAxes make_axes(const Multi_MOL &molecule,
                     std::string_view molecule_role) {
  EntityAxes out;
  out.residues.kind = EntityKind::Residue;
  out.atoms.kind = EntityKind::Atom;
  out.residues.atom_to_entity.reserve(molecule.info.size());
  out.atoms.atom_to_entity.reserve(molecule.info.size());
  out.atoms.entities.reserve(molecule.info.size());

  std::unordered_map<std::string, std::size_t> residue_indices;
  for (std::size_t atom = 0; atom < molecule.info.size(); ++atom) {
    const mol_info &info = molecule.info[atom];
    const std::string key = residue_key(info);
    auto [it, inserted] =
        residue_indices.emplace(key, out.residues.entities.size());
    if (inserted) {
      out.residues.entities.push_back(
          {it->second, std::string(molecule_role),
           residue_label(molecule_role, info), info.get_chain(),
           info.get_res_name(), {}, info.get_resnr()});
    }
    out.residues.atom_to_entity.push_back(it->second);
    out.atoms.atom_to_entity.push_back(atom);
    out.atoms.entities.push_back(
        {atom, std::string(molecule_role), atom_label(molecule_role, info, atom),
         info.get_chain(), info.get_res_name(), info.get_atom(),
         info.get_resnr()});
  }
  return out;
}

EnergyDecompositionEngine::EnergyDecompositionEngine(EntityAxes mol1_axes,
                                                     EntityAxes mol2_axes,
                                                     EngineConfig config)
    : mol1_axes_{std::move(mol1_axes)}, mol2_axes_{std::move(mol2_axes)},
      config_{config} {
  validate_axes(mol1_axes_);
  validate_axes(mol2_axes_);
  pooled_ = std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
  trajectory_equal_ =
      std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
}

EnergyDecompositionEngine::~EnergyDecompositionEngine() = default;

EndpointEntityDecompositionEngine::EndpointEntityDecompositionEngine(
    EntityAxes mol1_axes, EntityAxes mol2_axes, EngineConfig config)
    : mol1_axes_{std::move(mol1_axes)}, mol2_axes_{std::move(mol2_axes)},
      config_{config} {
  validate_axes(mol1_axes_);
  validate_axes(mol2_axes_);
  pooled_ = std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
  trajectory_equal_ =
      std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
}

EndpointEntityDecompositionEngine::~EndpointEntityDecompositionEngine() =
    default;

void EndpointEntityDecompositionEngine::begin_trajectory(
    std::string trajectory_id) {
  if (trajectory_open_) {
    throw std::logic_error(
        "endpoint energy decomposition trajectory is already in progress");
  }
  trajectory_id_ = std::move(trajectory_id);
  current_ = std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
  trajectory_open_ = true;
}

void EndpointEntityDecompositionEngine::process_frame(
    const std::vector<EndpointEntityRow> &rows) {
  if (!trajectory_open_ || current_ == nullptr) {
    throw std::logic_error(
        "endpoint energy decomposition process_frame requires begin_trajectory");
  }
  const auto add_row = [&](Accumulators &destination,
                           const EndpointEntityRow &row) {
    const bool mol1 = row.molecule_role == "mol1";
    const bool mol2 = row.molecule_role == "mol2";
    if (!mol1 && !mol2)
      throw std::invalid_argument(
          "endpoint energy decomposition row has unknown molecule role");
    EndpointLevelAccumulators &level =
        row.kind == EntityKind::Atom ? destination.atoms : destination.residues;
    std::vector<EndpointSums> &sums = mol1 ? level.mol1 : level.mol2;
    if (row.entity_index >= sums.size()) {
      throw std::out_of_range(
          "endpoint energy decomposition entity index is out of range");
    }
    sums[row.entity_index].add(row.e_elec, row.e_vdw, row.g_polar_gb,
                               row.g_nonpolar_sasa);
  };
  for (const auto &row : rows) {
    add_row(*current_, row);
    add_row(*pooled_, row);
  }
  ++current_->frames;
  ++pooled_->frames;
}

void EnergyDecompositionEngine::begin_trajectory(std::string trajectory_id) {
  if (trajectory_open_) {
    throw std::logic_error(
        "energy decomposition trajectory is already in progress");
  }
  trajectory_id_ = std::move(trajectory_id);
  current_ = std::make_unique<Accumulators>(mol1_axes_, mol2_axes_);
  trajectory_open_ = true;
}

void EnergyDecompositionEngine::process_frame(const FrameEnergySource &source,
                                              TitanExecutor &executor) {
  if (!trajectory_open_ || current_ == nullptr) {
    throw std::logic_error(
        "energy decomposition process_frame requires begin_trajectory");
  }
  const std::size_t mol1_atom_count = mol1_axes_.atoms.entities.size();
  const std::size_t mol2_atom_count = mol2_axes_.atoms.entities.size();
  const std::size_t atom_pair_count = mol1_atom_count * mol2_atom_count;
  const bool cache_ready =
      source.cache != nullptr &&
      source.cache->elec_ener.size() >= atom_pair_count &&
      source.cache->vdw_ener.size() >= atom_pair_count;
  const bool fallback_ready = source.fallback != nullptr &&
                              source.mol1_frame != nullptr &&
                              source.mol2_frame != nullptr;
  if (!cache_ready && !fallback_ready) {
    throw std::invalid_argument(
        "energy decomposition frame has neither aligned MM cache nor fallback");
  }

  FrameScratch scratch;
  scratch.atom_elec.assign(atom_pair_count, 0.0);
  scratch.atom_vdw.assign(atom_pair_count, 0.0);
  scratch.residue_elec.assign(mol1_axes_.residues.entities.size() *
                                  mol2_axes_.residues.entities.size(),
                              0.0);
  scratch.residue_vdw.assign(scratch.residue_elec.size(), 0.0);
  scratch.mol1_atom_elec.assign(mol1_atom_count, 0.0);
  scratch.mol1_atom_vdw.assign(mol1_atom_count, 0.0);
  scratch.mol2_atom_elec.assign(mol2_atom_count, 0.0);
  scratch.mol2_atom_vdw.assign(mol2_atom_count, 0.0);
  scratch.mol1_residue_elec.assign(mol1_axes_.residues.entities.size(), 0.0);
  scratch.mol1_residue_vdw.assign(mol1_axes_.residues.entities.size(), 0.0);
  scratch.mol2_residue_elec.assign(mol2_axes_.residues.entities.size(), 0.0);
  scratch.mol2_residue_vdw.assign(mol2_axes_.residues.entities.size(), 0.0);

  executor.parallel_for(
      mol2_atom_count, executor.capacity(), 1,
      [&](std::size_t, std::size_t row_begin, std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
          const std::size_t offset = row * mol1_atom_count;
          if (cache_ready) {
            std::size_t col = 0;
#if defined(__AVX2__)
            for (; col + 4 <= mol1_atom_count; col += 4) {
              const __m256d elec =
                  _mm256_loadu_pd(source.cache->elec_ener.data() + offset + col);
              const __m256d vdw =
                  _mm256_loadu_pd(source.cache->vdw_ener.data() + offset + col);
              _mm256_storeu_pd(scratch.atom_elec.data() + offset + col, elec);
              _mm256_storeu_pd(scratch.atom_vdw.data() + offset + col, vdw);
            }
#endif
            for (; col < mol1_atom_count; ++col) {
              scratch.atom_elec[offset + col] =
                  source.cache->elec_ener[offset + col];
              scratch.atom_vdw[offset + col] =
                  source.cache->vdw_ener[offset + col];
            }
          } else {
            for (std::size_t col = 0; col < mol1_atom_count; ++col) {
              const EnergyPairDecomposition pair = source.fallback->pair_energy(
                  *source.mol1_frame, *source.mol2_frame, col, row);
              scratch.atom_elec[offset + col] = pair.elec;
              scratch.atom_vdw[offset + col] = pair.vdw;
            }
          }
        }
      });

  const std::size_t mol1_residue_count = mol1_axes_.residues.entities.size();
  for (std::size_t row = 0; row < mol2_atom_count; ++row) {
    const std::size_t row_residue =
        mol2_axes_.residues.atom_to_entity[row];
    for (std::size_t col = 0; col < mol1_atom_count; ++col) {
      const std::size_t atom_pair = row * mol1_atom_count + col;
      const std::size_t col_residue =
          mol1_axes_.residues.atom_to_entity[col];
      const std::size_t residue_pair =
          row_residue * mol1_residue_count + col_residue;
      const double elec = scratch.atom_elec[atom_pair];
      const double vdw = scratch.atom_vdw[atom_pair];
      scratch.residue_elec[residue_pair] += elec;
      scratch.residue_vdw[residue_pair] += vdw;
      scratch.mol1_atom_elec[col] += elec;
      scratch.mol1_atom_vdw[col] += vdw;
      scratch.mol2_atom_elec[row] += elec;
      scratch.mol2_atom_vdw[row] += vdw;
    }
  }
  for (std::size_t row = 0; row < mol2_axes_.residues.entities.size(); ++row) {
    for (std::size_t col = 0; col < mol1_residue_count; ++col) {
      const std::size_t pair = row * mol1_residue_count + col;
      scratch.mol1_residue_elec[col] += scratch.residue_elec[pair];
      scratch.mol1_residue_vdw[col] += scratch.residue_vdw[pair];
      scratch.mol2_residue_elec[row] += scratch.residue_elec[pair];
      scratch.mol2_residue_vdw[row] += scratch.residue_vdw[pair];
    }
  }

  const auto add_frame = [&](Accumulators &destination) {
    add_level_frame(destination.atoms, scratch.atom_elec, scratch.atom_vdw,
                    scratch.mol2_atom_elec, scratch.mol2_atom_vdw,
                    scratch.mol1_atom_elec, scratch.mol1_atom_vdw);
    add_level_frame(destination.residues, scratch.residue_elec,
                    scratch.residue_vdw, scratch.mol2_residue_elec,
                    scratch.mol2_residue_vdw, scratch.mol1_residue_elec,
                    scratch.mol1_residue_vdw);
    ++destination.frames;
  };
  add_frame(*current_);
  add_frame(*pooled_);
}

ResultCore finish_result(const EnergyDecompositionEngine::Accumulators &source,
                         const EngineConfig &config, const EntityAxes &mol1_axes,
                         const EntityAxes &mol2_axes) {
  ResultCore out;
  out.frames = source.frames;
  out.mol1_axes = mol1_axes;
  out.mol2_axes = mol2_axes;
  out.residues = finish_matrix(source.residues);
  out.atoms = finish_matrix(source.atoms);
  out.residue_edges = finish_edges(out.residues, config);
  out.atom_edges = finish_edges(out.atoms, config);
  out.mol1_residue_rankings =
      finish_rankings(source.residues.col_rankings, config);
  out.mol2_residue_rankings =
      finish_rankings(source.residues.row_rankings, config);
  out.mol1_atom_rankings = finish_rankings(source.atoms.col_rankings, config);
  out.mol2_atom_rankings = finish_rankings(source.atoms.row_rankings, config);
  return out;
}

TrajectoryResult EnergyDecompositionEngine::finalize_trajectory() {
  if (!trajectory_open_ || current_ == nullptr) {
    throw std::logic_error(
        "energy decomposition finalize_trajectory requires open trajectory");
  }
  ResultCore core = finish_result(*current_, config_, mol1_axes_, mol2_axes_);
  add_level_summary(trajectory_equal_->atoms, current_->atoms);
  add_level_summary(trajectory_equal_->residues, current_->residues);
  ++trajectory_equal_->frames;
  ++trajectories_;

  TrajectoryResult out;
  static_cast<ResultCore &>(out) = std::move(core);
  out.trajectory_id = trajectory_id_;
  current_.reset();
  trajectory_id_.clear();
  trajectory_open_ = false;
  return out;
}

AllTrajectoryResult EnergyDecompositionEngine::finalize_all() const {
  if (trajectory_open_) {
    throw std::logic_error(
        "energy decomposition finalize_all requires finalized trajectory");
  }
  AllTrajectoryResult out;
  out.trajectories = trajectories_;
  out.trajectory_equal =
      finish_result(*trajectory_equal_, config_, mol1_axes_, mol2_axes_);
  out.pooled_frames = finish_result(*pooled_, config_, mol1_axes_, mol2_axes_);
  return out;
}

namespace {

std::vector<EndpointEntityResultRow>
finish_endpoint_level(const EndpointLevelAccumulators &level,
                      const EntityAxis &mol1_axis, const EntityAxis &mol2_axis,
                      EntityKind kind) {
  std::vector<EndpointEntityResultRow> out;
  out.reserve(level.mol1.size() + level.mol2.size());
  for (std::size_t i = 0; i < level.mol1.size(); ++i) {
    out.push_back(
        {kind, "mol1", i, mol1_axis.entities[i], finish_endpoint(level.mol1[i])});
  }
  for (std::size_t i = 0; i < level.mol2.size(); ++i) {
    out.push_back(
        {kind, "mol2", i, mol2_axis.entities[i], finish_endpoint(level.mol2[i])});
  }
  return out;
}

void sort_endpoint_ranking(std::vector<EndpointEntityResultRow> &rows,
                           const EngineConfig &config) {
  std::stable_sort(rows.begin(), rows.end(), [&](const auto &lhs,
                                                 const auto &rhs) {
    const double lhs_value =
        endpoint_metric_value(lhs.stats, config.ranking_metric);
    const double rhs_value =
        endpoint_metric_value(rhs.stats, config.ranking_metric);
    if (lhs_value != rhs_value)
      return value_before(lhs_value, rhs_value, config.sort_direction);
    if (lhs.molecule_role != rhs.molecule_role)
      return lhs.molecule_role < rhs.molecule_role;
    return lhs.entity_index < rhs.entity_index;
  });
}

EndpointEntityResult finish_endpoint_result(
    const EndpointEntityDecompositionEngine::Accumulators &source,
    const EngineConfig &config, const EntityAxes &mol1_axes,
    const EntityAxes &mol2_axes, std::string_view source_name,
    std::string trajectory_id, std::size_t trajectories) {
  EndpointEntityResult out;
  out.source = std::string(source_name);
  out.trajectory_id = std::move(trajectory_id);
  out.frames = source.frames;
  out.trajectories = trajectories;
  out.residue_map =
      finish_endpoint_level(source.residues, mol1_axes.residues,
                            mol2_axes.residues, EntityKind::Residue);
  out.atom_map = finish_endpoint_level(source.atoms, mol1_axes.atoms,
                                       mol2_axes.atoms, EntityKind::Atom);
  out.residue_ranking = out.residue_map;
  out.atom_ranking = out.atom_map;
  sort_endpoint_ranking(out.residue_ranking, config);
  sort_endpoint_ranking(out.atom_ranking, config);
  return out;
}

} // namespace

EndpointEntityResult EndpointEntityDecompositionEngine::finalize_trajectory(
    std::string_view source) {
  if (!trajectory_open_ || current_ == nullptr) {
    throw std::logic_error(
        "endpoint energy decomposition finalize_trajectory requires open trajectory");
  }
  EndpointEntityResult out =
      finish_endpoint_result(*current_, config_, mol1_axes_, mol2_axes_, source,
                             trajectory_id_, 1);
  add_endpoint_summary(trajectory_equal_->atoms, current_->atoms);
  add_endpoint_summary(trajectory_equal_->residues, current_->residues);
  ++trajectory_equal_->frames;
  ++trajectories_;
  current_.reset();
  trajectory_id_.clear();
  trajectory_open_ = false;
  return out;
}

EndpointEntityResult EndpointEntityDecompositionEngine::finalize_all(
    std::string_view source) const {
  if (trajectory_open_) {
    throw std::logic_error(
        "endpoint energy decomposition finalize_all requires finalized trajectory");
  }
  return finish_endpoint_result(*trajectory_equal_, config_, mol1_axes_,
                                mol2_axes_, source, "trajectory_equal",
                                trajectories_);
}

EndpointEntityResult
EndpointEntityDecompositionEngine::finalize_pooled_frames(
    std::string_view source) const {
  if (trajectory_open_) {
    throw std::logic_error(
        "endpoint energy decomposition finalize_pooled_frames requires finalized trajectory");
  }
  return finish_endpoint_result(*pooled_, config_, mol1_axes_, mol2_axes_,
                                source, "pooled_frames", trajectories_);
}

namespace {

std::string with_suffix(std::string_view stem, std::string_view suffix,
                        std::string_view extension) {
  return std::string(stem) + std::string(suffix) + std::string(extension);
}

void export_named_result(const ResultCore &result,
                         const std::filesystem::path &base_dir,
                         std::string_view suffix,
                         const EngineConfig &config) {
  std::filesystem::create_directories(base_dir);
  std::ofstream summary(
      base_dir / with_suffix("edecomp_summary", suffix, ".dat"));
  summary << "MM energy decomposition summary\n"
          << "frames: " << result.frames << '\n'
          << "abs_mean = abs(signed_mean)\n"
          << "mean_abs = mean(abs(frame_energy))\n"
          << "dense residue matrices retain every residue pair\n"
          << "atom outputs are sparse edge and entity ranking files\n"
          << "sparse edge files retain configured top interactions\n"
          << "trajectory_equal averages trajectory summaries equally\n"
          << "pooled_frames weights every processed frame equally\n"
          << "map_semantics: MM matrices are direct pairwise electrostatic "
             "plus van der Waals interactions. MM/GBSA and MM/PBSA maps are "
             "endpoint entity decompositions and must not be interpreted as "
             "pair matrices.\n"
          << "average_policy: Rankings use trajectory-equal means; "
             "pooled-frame means are exported as diagnostics.\n";

  write_component_matrix(
      base_dir /
          with_suffix("edecomp_mm_residue_matrix_elec", suffix, ".csv"),
      result.residues, result.mol2_axes.residues, result.mol1_axes.residues,
      "elec");
  write_component_matrix(
      base_dir / with_suffix("edecomp_mm_residue_matrix_vdw", suffix, ".csv"),
      result.residues, result.mol2_axes.residues, result.mol1_axes.residues,
      "vdw");
  write_component_matrix(
      base_dir /
          with_suffix("edecomp_mm_residue_matrix_total", suffix, ".csv"),
      result.residues, result.mol2_axes.residues, result.mol1_axes.residues,
      "total");
  write_edges(base_dir / with_suffix("edecomp_mm_residue_edges", suffix,
                                     ".csv"),
              result.residue_edges, result.mol2_axes.residues,
              result.mol1_axes.residues);
  write_edges(base_dir / with_suffix("edecomp_mm_atom_edges", suffix, ".csv"),
              result.atom_edges, result.mol2_axes.atoms, result.mol1_axes.atoms);
  write_combined_rankings(
      base_dir /
          with_suffix("edecomp_mm_residue_ranking", suffix, ".csv"),
      result.mol1_residue_rankings, result.mol1_axes.residues,
      result.mol2_residue_rankings, result.mol2_axes.residues, config);
  write_combined_rankings(
      base_dir / with_suffix("edecomp_mm_atom_ranking", suffix, ".csv"),
              result.mol1_atom_rankings, result.mol1_axes.atoms,
              result.mol2_atom_rankings, result.mol2_axes.atoms, config);
}

} // namespace

void export_result(const ResultCore &result,
                   const std::filesystem::path &base_dir,
                   std::string_view suffix) {
  export_named_result(result, base_dir, suffix, {});
}

void export_all_results(const AllTrajectoryResult &result,
                        const std::filesystem::path &base_dir,
                        std::string_view suffix) {
  export_named_result(result.trajectory_equal, base_dir, suffix, {});
  export_named_result(result.pooled_frames, base_dir, "_pooled_frames", {});
}

void export_endpoint_result(const EndpointEntityResult &result,
                            const std::filesystem::path &base_dir,
                            std::string_view source,
                            std::string_view suffix) {
  std::filesystem::create_directories(base_dir);
  const std::string source_text(source);
  const EndpointOutputLabels labels = endpoint_output_labels(source);
  write_endpoint_map(
      base_dir / with_suffix("edecomp_" + source_text + "_residue_map", suffix,
                             ".csv"),
      result.residue_map, labels);
  write_endpoint_ranking(
      base_dir /
          with_suffix("edecomp_" + source_text + "_residue_ranking", suffix,
                      ".csv"),
      result.residue_ranking, labels);
  write_endpoint_map(
      base_dir / with_suffix("edecomp_" + source_text + "_atom_map", suffix,
                             ".csv"),
      result.atom_map, labels);
  write_endpoint_ranking(
      base_dir / with_suffix("edecomp_" + source_text + "_atom_ranking", suffix,
                             ".csv"),
      result.atom_ranking, labels);

  std::ofstream summary(base_dir / with_suffix("edecomp_summary", suffix,
                                               ".dat"),
                        std::ios::app);
  summary << '\n' << labels.title << '\n'
          << "source: " << source_text << '\n'
          << "frames: " << result.frames << '\n'
          << "trajectories: " << result.trajectories << '\n'
          << "map_semantics: " << labels.semantics_note << '\n';
}

void export_endpoint_all_results(const EndpointEntityResult &trajectory_equal,
                                 const EndpointEntityResult &pooled_frames,
                                 const std::filesystem::path &base_dir,
                                 std::string_view source,
                                 std::string_view suffix) {
  export_endpoint_result(trajectory_equal, base_dir, source, suffix);
  export_endpoint_result(pooled_frames, base_dir, source, "_pooled_frames");
}

} // namespace titan_energy_decomp

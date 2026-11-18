#ifndef TITAN_ENERGY_DECOMP_ENGINE_H
#define TITAN_ENERGY_DECOMP_ENGINE_H

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class EnergyDecompositionProvider;
class MOL;
class Multi_MOL;
class TitanExecutor;
struct cache_sigma_epsilon;

namespace titan_energy_decomp {

enum class EntityKind { Residue, Atom };
enum class RankingMetric { SignedMean, AbsMean, MeanAbs };
enum class SortDirection { Descending, Ascending };

struct EntityMetadata {
  std::size_t entity_index{0};
  std::string molecule_role;
  std::string label;
  std::string chain;
  std::string residue_name;
  std::string atom_name;
  std::size_t residue_number{0};
};

struct EntityAxis {
  EntityKind kind{EntityKind::Atom};
  std::vector<EntityMetadata> entities;
  std::vector<std::size_t> atom_to_entity;
};

struct EntityAxes {
  EntityAxis residues;
  EntityAxis atoms;
};

[[nodiscard]] EntityAxes make_axes(const Multi_MOL &molecule,
                                   std::string_view molecule_role);

struct PairStatistics {
  double elec_signed_mean{0.0};
  double elec_abs_mean{0.0};
  double elec_mean_abs{0.0};
  double vdw_signed_mean{0.0};
  double vdw_abs_mean{0.0};
  double vdw_mean_abs{0.0};
  double signed_mean{0.0};
  double abs_mean{0.0};
  double mean_abs{0.0};
};

struct DenseMatrix {
  std::size_t rows{0};
  std::size_t cols{0};
  std::vector<PairStatistics> values;
};

struct SparseEdge {
  std::size_t row{0};
  std::size_t col{0};
  PairStatistics stats;
};

struct EntityRanking {
  std::size_t entity_index{0};
  PairStatistics stats;
};

struct EndpointEntityRow {
  EntityKind kind{EntityKind::Residue};
  std::string molecule_role;
  std::size_t entity_index{0};
  double e_elec{0.0};
  double e_vdw{0.0};
  double g_polar_gb{0.0};
  double g_nonpolar_sasa{0.0};
};

struct EndpointEntityStatistics {
  double e_elec_mean{0.0};
  double e_vdw_mean{0.0};
  double e_mm_mean{0.0};
  double g_polar_gb_mean{0.0};
  double g_nonpolar_sasa_mean{0.0};
  double delta_g_no_entropy_mean{0.0};
  double abs_mean{0.0};
  double mean_abs{0.0};
  double pooled_frame_mean{0.0};
  std::size_t samples{0};
};

struct EndpointEntityResultRow {
  EntityKind kind{EntityKind::Residue};
  std::string molecule_role;
  std::size_t entity_index{0};
  EntityMetadata metadata;
  EndpointEntityStatistics stats;
};

struct EndpointEntityResult {
  std::string source;
  std::string trajectory_id;
  std::size_t frames{0};
  std::size_t trajectories{0};
  std::vector<EndpointEntityResultRow> residue_map;
  std::vector<EndpointEntityResultRow> atom_map;
  std::vector<EndpointEntityResultRow> residue_ranking;
  std::vector<EndpointEntityResultRow> atom_ranking;
};

struct EngineConfig {
  std::size_t top_edge_count{200};
  RankingMetric edge_metric{RankingMetric::AbsMean};
  RankingMetric ranking_metric{RankingMetric::AbsMean};
  SortDirection sort_direction{SortDirection::Descending};
};

struct FrameEnergySource {
  const cache_sigma_epsilon *cache{nullptr};
  const EnergyDecompositionProvider *fallback{nullptr};
  const MOL *mol1_frame{nullptr};
  const MOL *mol2_frame{nullptr};
};

struct ResultCore {
  std::size_t frames{0};
  EntityAxes mol1_axes;
  EntityAxes mol2_axes;
  DenseMatrix residues;
  DenseMatrix atoms;
  std::vector<SparseEdge> residue_edges;
  std::vector<SparseEdge> atom_edges;
  std::vector<EntityRanking> mol1_residue_rankings;
  std::vector<EntityRanking> mol2_residue_rankings;
  std::vector<EntityRanking> mol1_atom_rankings;
  std::vector<EntityRanking> mol2_atom_rankings;
};

struct TrajectoryResult : ResultCore {
  std::string trajectory_id;
};

struct AllTrajectoryResult {
  std::size_t trajectories{0};
  ResultCore trajectory_equal;
  ResultCore pooled_frames;
};

class EnergyDecompositionEngine {
public:
  struct Accumulators;

  EnergyDecompositionEngine(EntityAxes mol1_axes, EntityAxes mol2_axes,
                            EngineConfig config = {});
  ~EnergyDecompositionEngine();

  void begin_trajectory(std::string trajectory_id);
  void process_frame(const FrameEnergySource &source, TitanExecutor &executor);
  [[nodiscard]] TrajectoryResult finalize_trajectory();
  [[nodiscard]] AllTrajectoryResult finalize_all() const;

  [[nodiscard]] const EntityAxes &mol1_axes() const noexcept {
    return mol1_axes_;
  }
  [[nodiscard]] const EntityAxes &mol2_axes() const noexcept {
    return mol2_axes_;
  }

private:
  EntityAxes mol1_axes_;
  EntityAxes mol2_axes_;
  EngineConfig config_;
  bool trajectory_open_{false};
  std::string trajectory_id_;
  std::unique_ptr<Accumulators> current_;
  std::unique_ptr<Accumulators> pooled_;
  std::unique_ptr<Accumulators> trajectory_equal_;
  std::size_t trajectories_{0};
};

class EndpointEntityDecompositionEngine {
public:
  struct Accumulators;

  EndpointEntityDecompositionEngine(EntityAxes mol1_axes, EntityAxes mol2_axes,
                                    EngineConfig config = {});
  ~EndpointEntityDecompositionEngine();

  void begin_trajectory(std::string trajectory_id);
  void process_frame(const std::vector<EndpointEntityRow> &rows);
  [[nodiscard]] EndpointEntityResult finalize_trajectory(
      std::string_view source = "mmgbsa");
  [[nodiscard]] EndpointEntityResult finalize_all(
      std::string_view source = "mmgbsa") const;
  [[nodiscard]] EndpointEntityResult finalize_pooled_frames(
      std::string_view source = "mmgbsa") const;
  [[nodiscard]] const EntityAxes &mol1_axes() const noexcept {
    return mol1_axes_;
  }
  [[nodiscard]] const EntityAxes &mol2_axes() const noexcept {
    return mol2_axes_;
  }

private:
  EntityAxes mol1_axes_;
  EntityAxes mol2_axes_;
  EngineConfig config_;
  bool trajectory_open_{false};
  std::string trajectory_id_;
  std::unique_ptr<Accumulators> current_;
  std::unique_ptr<Accumulators> pooled_;
  std::unique_ptr<Accumulators> trajectory_equal_;
  std::size_t trajectories_{0};
};

// Energies are accumulated in kJ/mol. Exports divide every energy column by
// kj_per_unit (TITAN's eu_par: 1 for kJ/mol, 4.184 for kcal/mol) and name the
// unit in edecomp_summary.dat; counts, indices and ranks are not scaled.
struct OutputUnit {
  std::string name{"kJ/mol"};
  double kj_per_unit{1.0};
};

void export_result(const ResultCore &result,
                   const std::filesystem::path &base_dir,
                   std::string_view suffix = {},
                   const OutputUnit &unit = {});
void export_all_results(const AllTrajectoryResult &result,
                        const std::filesystem::path &base_dir,
                        std::string_view suffix = "_average",
                        const OutputUnit &unit = {});
void export_endpoint_result(const EndpointEntityResult &result,
                            const std::filesystem::path &base_dir,
                            std::string_view source,
                            std::string_view suffix = {},
                            const OutputUnit &unit = {});
void export_endpoint_all_results(const EndpointEntityResult &trajectory_equal,
                                 const EndpointEntityResult &pooled_frames,
                                 const std::filesystem::path &base_dir,
                                 std::string_view source,
                                 std::string_view suffix = "_average",
                                 const OutputUnit &unit = {});

} // namespace titan_energy_decomp

#endif // TITAN_ENERGY_DECOMP_ENGINE_H

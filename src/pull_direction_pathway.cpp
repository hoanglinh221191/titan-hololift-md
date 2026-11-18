#include "pull_direction_pathway.h"

#include "MOL.h"
#include "functions.h"
#include "pathway_de_variants.h"
#include "titan_path/clearance.h"
#include "titan_path/cartesian_audit.h"
#include "titan_path/control.h"
#include "titan_path/continuity.h"
#include "titan_path/geometry.h"
#include "titan_path/navigation.h"
#include "titan_path/registration.h"
#include "titan_path/route.h"
#include "titan_path/status_numeric_io.h"
#include "titan_path/observed_boundary.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double kPathTol = 1.0e-8;
constexpr std::size_t kNearestResidueEdges = 3;
constexpr double kMinProgressSeedGapNorm = 0.12;
constexpr double kMinPatternStep = 0.05;
constexpr std::size_t kPathPatternMaxSweeps = 32;
constexpr double kPathwayProgressDedupEps = 1.0e-6;
constexpr double kPathwayDirectMinCellNorm = 1.0e-4;

using Point3 = std::array<double, 3>;

struct Interval {
  double begin{0.0};
  double end{0.0};
};

struct PathNode {
  std::size_t chainnum{0};
  std::size_t respos{0};
  std::string chain{" "};
  std::string res_name{"UNK"};
  Point3 com{0.0, 0.0, 0.0};
  double radius{0.0};
  std::size_t first_local_atom{0};
};

using PathSegment = titan_path::ArcLengthSegment;

struct PathContext {
  std::size_t group_index{0};
  std::string group_name;
  std::vector<PathNode> all_nodes;
  std::vector<PathNode> principal_nodes;
  std::vector<PathSegment> segments;
  std::vector<titan_path::CorridorSample> corridor_samples;
  std::vector<Interval> valid_intervals;
  std::size_t pruned_component_nodes{0};
  std::size_t pruned_branch_nodes{0};
  bool used_geometric_edges{false};
  bool explicit_route{false};
  std::string route_id;
  std::string route_source;
  std::string reference_structure_sha256;
  std::string route_file_sha256;
  std::string registration_mode{"legacy-auto"};
  double registration_rmsd{0.0};
  double total_length{0.0};
};

struct PathResult {
  bool ok{false};
  std::string error;
  PathContext path;
};

struct CandidateResult {
  cache_energy energy{};
  double raw_score{0.0};
  double progress{0.0};
  double angle_deg{0.0};
  force vector{};
  bool valid{false};
};

struct PathProjection {
  bool valid{false};
  double progress{0.0};
  double distance{std::numeric_limits<double>::infinity()};
  Point3 point{0.0, 0.0, 0.0};
  std::size_t segment_index{0};
};

struct PathwayState {
  PathProjection current_projection{};
  std::optional<double> previous_progress{};
  double continuity_half_window{0.0};
  std::vector<Interval> forward_intervals;
  PathwayMode mode{PathwayMode::Advance};
  double minimum_receptor_ligand_distance{
      std::numeric_limits<double>::infinity()};
  bool physical_exit_gate_enabled{false};
  bool physical_exit_passed{true};
  bool recovery_direct_probe_used{false};
};

struct PathwayWorkerContext {
  MOL scratch{};
  cache_sigma_epsilon eval_cache{};
  std::size_t candidate_proposals{0};
  std::size_t evaluations_done{0};
  std::size_t scan_shell_exclusions{0};
  std::size_t energy_evaluations_done{0};
  std::size_t probe_evaluations{0};
  std::size_t clearance_pair_checks{0};
  std::size_t clearance_rejections{0};
  std::size_t probe_clearance_rejections{0};
  std::size_t probe_mode_rejections{0};
  std::size_t swept_clearance_probe_evaluations{0};
  std::size_t swept_clearance_pair_checks{0};
  std::size_t swept_clearance_rejections{0};
  std::size_t swept_interior_only_rejections{0};
};

struct ClearanceEvaluation {
  double minimum_distance{std::numeric_limits<double>::infinity()};
  double endpoint_minimum_distance{std::numeric_limits<double>::infinity()};
  double witness_start_distance{std::numeric_limits<double>::infinity()};
  double witness_endpoint_distance{std::numeric_limits<double>::infinity()};
  double minimum_fraction{1.0};
  std::size_t receptor_index{0};
  std::size_t ligand_index{0};
  std::size_t pair_checks{0};
  bool passes_threshold{true};
  bool global_minimum_complete{false};
  bool threshold_violation_hidden_at_endpoints{false};
};

struct PathwayEvalBudget {
  std::size_t max_evals{0};
  std::atomic_size_t reserved{0};
};

struct PathwayDirectStats {
  bool enabled{false};
  bool used_fallback_budget{false};
  bool stopped_by_eval_budget{false};
  bool stopped_by_local_threshold{false};
  bool stopped_by_stagnation{false};
  std::size_t effective_eval_budget{0};
  std::size_t expansions{0};
  std::size_t batches{0};
  std::size_t max_batch_cells{0};
  std::size_t cells_remaining{0};
  std::size_t evaluated_candidates{0};
  std::size_t best_expansion{0};
  std::size_t phase_evaluations_used{0};
  double best_cell_size{0.0};
  double best_half_width_norm{0.0};
  double wall_seconds{0.0};
};

struct PathwayDirectCell {
  double center_norm{0.5};
  double half_width_norm{0.5};
  PathwayCandidateState state{};
  std::size_t depth{0};
  std::size_t id{0};
};

struct PathwayDirectDomain {
  std::vector<Interval> intervals;
  std::vector<double> prefix_lengths;
  double total_valid_length{0.0};
};

struct PathwayReportRow {
  PathwayCandidateState representative{};
  std::size_t representative_index{0};
  std::size_t count{0};
};

struct PathwayCorridorGridStats {
  bool enabled{false};
  bool stopped_by_eval_budget{false};
  std::size_t longitudinal_points{0};
  std::size_t proposed_targets{0};
  std::size_t evaluated_targets{0};
  std::size_t valid_targets{0};
};

struct PathwayGuidedSeedStats {
  bool enabled{false};
  bool stopped_by_eval_budget{false};
  std::size_t proposed_directions{0};
  std::size_t budget_granted_directions{0};
  std::size_t evaluated_directions{0};
  std::size_t valid_directions{0};
  Point3 guide_direction{0.0, 0.0, 0.0};
};

struct PathwayCorridorProposal {
  double progress{0.0};
  double offset_u{0.0};
  double offset_v{0.0};
  double radius{0.0};
  Point3 target{0.0, 0.0, 0.0};
};

struct PathwayFinalSelection {
  PathwayCandidateState candidate{};
  double minimum_energy_key{std::numeric_limits<double>::infinity()};
  std::size_t valid_archive_size{0};
  std::size_t forward_archive_size{0};
  std::size_t forward_progress_rejections{0};
  std::size_t epsilon_eligible_size{0};
  std::size_t mode_eligible_size{0};
  std::size_t maximum_probe_tier_size{0};
  std::size_t navigation_eligible_size{0};
  double minimum_navigation_cost{std::numeric_limits<double>::infinity()};
};

struct ResidueKey {
  std::size_t chainnum{0};
  std::size_t respos{0};
  bool operator==(const ResidueKey &rhs) const noexcept {
    return chainnum == rhs.chainnum && respos == rhs.respos;
  }
};

struct ResidueKeyHash {
  std::size_t operator()(const ResidueKey &key) const noexcept {
    return (key.chainnum << 32U) ^ key.respos;
  }
};

struct ResidueAcc {
  PathNode node;
  Point3 weighted{0.0, 0.0, 0.0};
  double mass{0.0};
};

class DisjointSet {
public:
  explicit DisjointSet(std::size_t n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  std::size_t find(std::size_t x) {
    if (parent[x] != x)
      parent[x] = find(parent[x]);
    return parent[x];
  }
  bool unite(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (rank[a] < rank[b])
      std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b])
      ++rank[a];
    return true;
  }

private:
  std::vector<std::size_t> parent;
  std::vector<int> rank;
};

Point3 sub_pt(const Point3 &lhs, const Point3 &rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

Point3 add_pt(const Point3 &lhs, const Point3 &rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

Point3 scale_pt(const Point3 &p, double scale) {
  return {p[0] * scale, p[1] * scale, p[2] * scale};
}

double dot_pt(const Point3 &lhs, const Point3 &rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

double norm_sq(const Point3 &p) { return dot_pt(p, p); }

double dist_sq(const Point3 &lhs, const Point3 &rhs) {
  return norm_sq(sub_pt(lhs, rhs));
}

double dist_pt(const Point3 &lhs, const Point3 &rhs) {
  return std::sqrt(dist_sq(lhs, rhs));
}

Point3 mol_com(const MOL &mol, const Multi_MOL_parameters &parameters) {
  Point3 com{0.0, 0.0, 0.0};
  double mass_sum = 0.0;
  for (std::size_t i = 0; i < mol.size(); ++i) {
    const double mass = parameters[i].get_mass();
    com[0] += mol.pos.x[i] * mass;
    com[1] += mol.pos.y[i] * mass;
    com[2] += mol.pos.z[i] * mass;
    mass_sum += mass;
  }
  if (mass_sum <= 0.0)
    return {0.0, 0.0, 0.0};
  return scale_pt(com, 1.0 / mass_sum);
}

force force_from_points(const Point3 &from, const Point3 &to) {
  return {to[0] - from[0], to[1] - from[1], to[2] - from[2]};
}

double angle_to_previous_vector(const force &vector, const force &old_vector) {
  const double old_mod = old_vector.modul();
  const double new_mod = vector.modul();
  if (old_mod < 1.0e-12 || new_mod < 1.0e-12)
    return 0.0;
  force new_scaled{vector};
  force old_scaled{old_vector};
  new_scaled *= old_mod;
  old_scaled *= new_mod;
  return 360.0 / 3.14159265358979323846 *
         std::atan2(force(new_scaled - old_scaled).modul(),
                    force(new_scaled + old_scaled).modul());
}

std::optional<force> project_vector_to_turn_cone(
    const force &requested, const force &old_vector, double max_turn_deg,
    bool &turn_capped) {
  const auto projected = titan_path::project_vector_to_turn_cone(
      {requested.Fx, requested.Fy, requested.Fz},
      {old_vector.Fx, old_vector.Fy, old_vector.Fz}, max_turn_deg);
  turn_capped = projected.capped;
  if (!projected.valid)
    return std::nullopt;
  return force{projected.vector[0], projected.vector[1], projected.vector[2]};
}

std::size_t clamp_pattern_tail_budget(std::size_t total_budget) {
  if (total_budget == 0)
    return 0;
  return std::clamp<std::size_t>(total_budget / 8, 48, 160);
}

std::string residue_label(const PathNode &node) {
  std::ostringstream oss;
  oss << node.res_name << node.respos << " (chain " << node.chain << ", id "
      << node.chainnum << ")";
  return oss.str();
}

std::string swept_clearance_pbc_preprocessor(const arguments &Arguments) {
  if (Arguments.pbc.solv_pbctopo)
    return "pbctopo-" + Arguments.pbc.pbctopo_mode;
  if (Arguments.pbc.solv_pbcn)
    return "pbcn";
  return "none";
}

double base_score(const cache_energy &trial, const cache_energy &reference,
                  int flag_score_function) {
  if (flag_score_function == 1) {
    return trial.elec_ener + trial.vdw_ener - reference.elec_ener -
           reference.vdw_ener +
           2.0 * std::fabs(trial.elec_ener + trial.vdw_ener);
  }
  const double exponent =
      1.6863 * (-trial.elec_ener - trial.vdw_ener + reference.elec_ener +
                reference.vdw_ener);
  return -std::exp(std::clamp(exponent, -700.0, 700.0));
}

ClearanceEvaluation evaluate_interatomic_clearance(
    const MOL &reference, const MOL &ligand, double rejection_threshold = 0.0) {
  ClearanceEvaluation result;
  double minimum_squared = std::numeric_limits<double>::infinity();
  const double threshold_squared = rejection_threshold * rejection_threshold;
  for (std::size_t j = 0; j < ligand.size(); ++j) {
    const double ligand_x = ligand.pos.x[j];
    const double ligand_y = ligand.pos.y[j];
    const double ligand_z = ligand.pos.z[j];
    for (std::size_t i = 0; i < reference.size(); ++i) {
      const double dx = reference.pos.x[i] - ligand_x;
      const double dy = reference.pos.y[i] - ligand_y;
      const double dz = reference.pos.z[i] - ligand_z;
      const double squared = dx * dx + dy * dy + dz * dz;
      ++result.pair_checks;
      if (squared < minimum_squared) {
        minimum_squared = squared;
        result.receptor_index = i;
        result.ligand_index = j;
      }
      if (rejection_threshold > 0.0 && squared < threshold_squared) {
        result.minimum_distance = std::sqrt(minimum_squared);
        result.endpoint_minimum_distance = result.minimum_distance;
        result.minimum_fraction = 1.0;
        result.passes_threshold = false;
        result.global_minimum_complete = false;
        return result;
      }
    }
  }
  result.minimum_distance = std::sqrt(minimum_squared);
  result.endpoint_minimum_distance = result.minimum_distance;
  result.minimum_fraction = 1.0;
  result.global_minimum_complete = true;
  return result;
}

ClearanceEvaluation evaluate_interatomic_swept_clearance(
    const MOL &reference, const MOL &ligand_start,
    const force &displacement, double rejection_threshold) {
  const titan_path::CartesianCoordinatesView reference_view{
      std::span<const double>(reference.pos.x.data(), reference.pos.x.size()),
      std::span<const double>(reference.pos.y.data(), reference.pos.y.size()),
      std::span<const double>(reference.pos.z.data(), reference.pos.z.size())};
  const titan_path::CartesianCoordinatesView ligand_view{
      std::span<const double>(ligand_start.pos.x.data(),
                              ligand_start.pos.x.size()),
      std::span<const double>(ligand_start.pos.y.data(),
                              ligand_start.pos.y.size()),
      std::span<const double>(ligand_start.pos.z.data(),
                              ligand_start.pos.z.size())};
  const auto swept = titan_path::evaluate_translation_sweep(
      reference_view, ligand_view,
      {displacement.Fx, displacement.Fy, displacement.Fz},
      rejection_threshold);
  return {swept.witness_distance_angstrom,
          swept.endpoint_distance_angstrom,
          swept.witness_start_distance_angstrom,
          swept.witness_endpoint_distance_angstrom,
          swept.witness_fraction,
          swept.receptor_index,
          swept.ligand_index,
          swept.pair_checks,
          swept.passes_threshold,
          swept.global_minimum_complete,
          swept.threshold_violation_hidden_at_endpoints};
}

double stable_pathway_energy_key(const cache_energy &trial,
                                 const cache_energy &reference,
                                 int flag_score_function, double raw_score) {
  if (flag_score_function == 1)
    return pathway_energy_key(raw_score);
  const double exponent =
      1.6863 * (-trial.elec_ener - trial.vdw_ener + reference.elec_ener +
                reference.vdw_ener);
  if (!std::isfinite(exponent))
    return std::numeric_limits<double>::quiet_NaN();
  const double softplus =
      std::max(exponent, 0.0) + std::log1p(std::exp(-std::fabs(exponent)));
  return -softplus;
}

cache_sigma_epsilon make_eval_cache(const cache_sigma_epsilon &source) {
  cache_sigma_epsilon scratch;
  scratch.charge = source.charge;
  scratch.force_field_name = source.force_field_name;
  scratch.cutoff_dis = source.cutoff_dis;
  scratch.charmm_pair_sigma = source.charmm_pair_sigma;
  scratch.charmm_pair_epsilon = source.charmm_pair_epsilon;
  scratch.charmm_type_count = source.charmm_type_count;
  scratch.charmm_pair_sparse_keys = source.charmm_pair_sparse_keys;
  scratch.charmm_pair_sparse_sigma = source.charmm_pair_sparse_sigma;
  scratch.charmm_pair_sparse_epsilon = source.charmm_pair_sparse_epsilon;
  scratch.martini_pair_sigma = source.martini_pair_sigma;
  scratch.martini_pair_epsilon = source.martini_pair_epsilon;
  scratch.martini_type_count = source.martini_type_count;
  return scratch;
}

void eval_energy_model(const MOL &rec, const MOL &lg,
                       const std::vector<mol_info> &info_rec,
                       const std::vector<mol_info> &info_lg,
                       cache_sigma_epsilon &CachedData, cache_energy &Energy,
                       const arguments &Arguments,
                       const Multi_MOL_parameters &rec_parameters,
                       const Multi_MOL_parameters &lg_parameters) {
  if (Arguments.force_f.find("CHARMM") != std::string::npos ||
      Arguments.force_f.find("AMBER") != std::string::npos) {
#if defined(__AVX__)
    cal_energy_amber_intrin(rec, lg, info_rec, info_lg, CachedData, 0,
                            lg.size(), Energy, 0, Arguments, rec_parameters,
                            lg_parameters);
#else
    cal_energy_amber(rec, lg, info_rec, info_lg, CachedData, 0, lg.size(),
                     Energy, 0, Arguments, rec_parameters, lg_parameters);
#endif
  } else if (Arguments.force_f.find("OPLSAA") != std::string::npos) {
#if defined(__AVX__)
    cal_energy_opls_intrin(rec, lg, info_rec, info_lg, CachedData, 0, lg.size(),
                           Energy, 0, Arguments, rec_parameters, lg_parameters);
#else
    cal_energy_opls(rec, lg, info_rec, info_lg, CachedData, 0, lg.size(),
                    Energy, 0, Arguments, rec_parameters, lg_parameters);
#endif
  } else if (Arguments.force_f.find("MARTINI") != std::string::npos) {
    cal_energy_martini(rec, lg, info_rec, info_lg, CachedData, 0, lg.size(),
                       Energy, 0, Arguments, rec_parameters, lg_parameters);
  } else {
    cal_energy_gromos(rec, lg, info_rec, info_lg, CachedData, 0, lg.size(),
                      Energy, 0, Arguments, rec_parameters, lg_parameters);
  }
}

cache_energy reference_energy(const Multi_MOL &rec, const Multi_MOL &lg,
                              cache_sigma_epsilon &CachedData,
                              const arguments &Arguments,
                              TitanExecutor &executor) {
  const std::size_t worker_count = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t slots =
      std::max<std::size_t>(1, std::min(executor.capacity(), worker_count));
  std::vector<cache_energy> parts(slots);
  const std::size_t lg_size = lg.m_mol[0].size();
  executor.parallel_for(
      lg_size, worker_count, std::max<std::size_t>(1, lg_size / slots),
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        if (Arguments.force_f.find("CHARMM") != std::string::npos ||
            Arguments.force_f.find("AMBER") != std::string::npos) {
#if defined(__AVX__)
          cal_energy_amber_intrin(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                                  CachedData, begin, end, parts[worker_id], 0,
                                  Arguments, rec.parameters, lg.parameters);
#else
          cal_energy_amber(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                           CachedData, begin, end, parts[worker_id], 0,
                           Arguments, rec.parameters, lg.parameters);
#endif
        } else if (Arguments.force_f.find("OPLSAA") != std::string::npos) {
#if defined(__AVX__)
          cal_energy_opls_intrin(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                                 CachedData, begin, end, parts[worker_id], 0,
                                 Arguments, rec.parameters, lg.parameters);
#else
          cal_energy_opls(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                          CachedData, begin, end, parts[worker_id], 0,
                          Arguments, rec.parameters, lg.parameters);
#endif
        } else if (Arguments.force_f.find("MARTINI") != std::string::npos) {
          cal_energy_martini(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                             CachedData, begin, end, parts[worker_id], 0,
                             Arguments, rec.parameters, lg.parameters);
        } else {
          cal_energy_gromos(rec.m_mol[0], lg.m_mol[0], rec.info, lg.info,
                            CachedData, begin, end, parts[worker_id], 0,
                            Arguments, rec.parameters, lg.parameters);
        }
      });
  cache_energy total;
  for (const auto &part : parts)
    total += part;
  return total;
}

void add_edge(std::map<std::pair<std::size_t, std::size_t>, double> &edges,
              std::size_t lhs, std::size_t rhs, double weight) {
  if (lhs == rhs)
    return;
  if (lhs > rhs)
    std::swap(lhs, rhs);
  const auto key = std::make_pair(lhs, rhs);
  const auto it = edges.find(key);
  if (it == edges.end())
    edges.emplace(key, weight);
  else
    it->second = std::min(it->second, weight);
}

std::vector<std::vector<std::pair<std::size_t, double>>>
build_adj(std::size_t n,
          const std::map<std::pair<std::size_t, std::size_t>, double> &edges) {
  std::vector<std::vector<std::pair<std::size_t, double>>> adj(n);
  for (const auto &[edge, weight] : edges) {
    adj[edge.first].push_back({edge.second, weight});
    adj[edge.second].push_back({edge.first, weight});
  }
  return adj;
}

std::vector<std::size_t>
component_nodes(std::size_t start,
                const std::vector<std::vector<std::pair<std::size_t, double>>> &adj) {
  std::vector<std::size_t> nodes;
  if (start >= adj.size())
    return nodes;
  std::vector<unsigned char> seen(adj.size(), 0);
  std::vector<std::size_t> stack{start};
  seen[start] = 1;
  while (!stack.empty()) {
    const std::size_t node = stack.back();
    stack.pop_back();
    nodes.push_back(node);
    for (const auto &[next, weight] : adj[node]) {
      (void)weight;
      if (seen[next] != 0)
        continue;
      seen[next] = 1;
      stack.push_back(next);
    }
  }
  return nodes;
}

std::vector<Interval> merge_intervals(std::vector<Interval> intervals) {
  if (intervals.empty())
    return intervals;
  std::sort(intervals.begin(), intervals.end(),
            [](const Interval &lhs, const Interval &rhs) {
              if (lhs.begin != rhs.begin)
                return lhs.begin < rhs.begin;
              return lhs.end < rhs.end;
            });
  std::vector<Interval> merged;
  merged.push_back(intervals.front());
  for (std::size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].begin <= merged.back().end + 1.0e-9)
      merged.back().end = std::max(merged.back().end, intervals[i].end);
    else
      merged.push_back(intervals[i]);
  }
  return merged;
}

std::vector<Interval> within_radius(const PathSegment &segment,
                                    const Point3 &center, double radius_sq) {
  std::vector<Interval> out;
  const double a = norm_sq(segment.d);
  if (a <= kPathTol)
    return out;
  const Point3 m = sub_pt(segment.a, center);
  const double b = 2.0 * dot_pt(m, segment.d);
  const double c = norm_sq(m) - radius_sq;
  const double disc = b * b - 4.0 * a * c;
  if (disc < 0.0) {
    if (c <= 0.0)
      out.push_back({0.0, 1.0});
    return out;
  }
  const double sqrt_disc = std::sqrt(std::max(0.0, disc));
  const double t1 = (-b - sqrt_disc) / (2.0 * a);
  const double t2 = (-b + sqrt_disc) / (2.0 * a);
  const double begin = std::clamp(std::min(t1, t2), 0.0, 1.0);
  const double end = std::clamp(std::max(t1, t2), 0.0, 1.0);
  if (begin <= end + 1.0e-12)
    out.push_back({begin, end});
  return out;
}

std::vector<Interval> subtract_interval(const Interval &outer,
                                        const Interval &inner) {
  if (inner.end <= outer.begin || inner.begin >= outer.end)
    return {outer};
  std::vector<Interval> out;
  if (inner.begin > outer.begin + 1.0e-9)
    out.push_back({outer.begin, std::min(inner.begin, outer.end)});
  if (inner.end < outer.end - 1.0e-9)
    out.push_back({std::max(inner.end, outer.begin), outer.end});
  return out;
}

std::vector<Interval> shell_intervals(const PathSegment &segment,
                                      const Point3 &center, double min_radius,
                                      double max_radius) {
  auto outer = within_radius(segment, center, max_radius * max_radius);
  if (outer.empty())
    return outer;
  if (min_radius <= 0.0)
    return outer;
  const double min_eff = std::max(0.0, min_radius - 1.0e-9);
  const auto inner = within_radius(segment, center, min_eff * min_eff);
  if (inner.empty())
    return outer;
  std::vector<Interval> out;
  for (const auto &o : outer) {
    std::vector<Interval> pending{o};
    for (const auto &i : inner) {
      std::vector<Interval> next;
      for (const auto &candidate : pending) {
        const auto pieces = subtract_interval(candidate, i);
        next.insert(next.end(), pieces.begin(), pieces.end());
      }
      pending = std::move(next);
    }
    out.insert(out.end(), pending.begin(), pending.end());
  }
  return out;
}

bool finalize_explicit_path_geometry(PathResult &result, const Multi_MOL &lg,
                                     const arguments &Arguments) {
  if (result.path.principal_nodes.size() < 2) {
    result.error = "Explicit route contains fewer than two waypoints.";
    return false;
  }
  double progress = 0.0;
  for (std::size_t i = 1; i < result.path.principal_nodes.size(); ++i) {
    PathSegment segment;
    segment.a = result.path.principal_nodes[i - 1].com;
    segment.d = sub_pt(result.path.principal_nodes[i].com, segment.a);
    segment.length = std::sqrt(norm_sq(segment.d));
    if (segment.length <= kPathTol) {
      result.error = "Explicit route contains duplicate consecutive waypoints.";
      return false;
    }
    segment.s0 = progress;
    progress += segment.length;
    result.path.segments.push_back(segment);
  }
  result.path.total_length = progress;

  const Point3 ligand_start = mol_com(lg.m_mol[0], lg.parameters);
  std::vector<Interval> valid;
  for (const auto &segment : result.path.segments) {
    auto shell = shell_intervals(segment, ligand_start,
                                 Arguments.de.de_min_scan_radius,
                                 Arguments.de.de_scan_radius);
    for (const auto &interval : shell) {
      valid.push_back(
          {segment.s0 + interval.begin * segment.length,
           segment.s0 + interval.end * segment.length});
    }
  }
  result.path.valid_intervals = merge_intervals(std::move(valid));
  if (result.path.valid_intervals.empty()) {
    result.error =
        "Explicit route does not intersect the requested scan-radius window.";
    return false;
  }
  return true;
}

PathResult build_explicit_path(const Multi_MOL &rec, const Multi_MOL &lg,
                               const arguments &Arguments) {
  PathResult result;
  const auto loaded =
      titan_path::load_route_file(Arguments.de.pathway_file);
  if (!loaded.ok) {
    result.error = loaded.error;
    return result;
  }
  if (!Arguments.de.pathway_route_id.empty() &&
      loaded.route.route_id != Arguments.de.pathway_route_id) {
    result.error = "Route ID mismatch: requested '" +
                   Arguments.de.pathway_route_id + "' but file declares '" +
                   loaded.route.route_id + "'.";
    return result;
  }

  titan_path::RouteDefinition registered_route = loaded.route;
  result.path.explicit_route = true;
  result.path.route_id = loaded.route.route_id;
  result.path.route_source = loaded.route.source;
  result.path.reference_structure_sha256 =
      loaded.route.reference_structure_sha256;
  result.path.route_file_sha256 = loaded.source_file_sha256;
  result.path.registration_mode =
      titan_path::registration_mode_name(loaded.route.registration);
  result.path.group_name = loaded.route.route_id;
  result.path.group_index = std::numeric_limits<std::size_t>::max();

  if (loaded.route.registration == titan_path::RegistrationMode::Rigid) {
    std::unordered_map<std::size_t, std::size_t> current_atom_by_original_id;
    for (std::size_t atom = 0; atom < rec.info.size(); ++atom) {
      if (!rec.info[atom].has_original_atom_id())
        continue;
      const std::size_t original_id = rec.info[atom].get_original_atom_id();
      if (!current_atom_by_original_id.emplace(original_id, atom).second) {
        result.error = "Current receptor contains duplicate original atom ID " +
                       std::to_string(original_id) + ".";
        return result;
      }
    }

    std::vector<titan_path::Vec3> reference_anchors;
    std::vector<titan_path::Vec3> current_anchors;
    reference_anchors.reserve(loaded.route.anchors.size());
    current_anchors.reserve(loaded.route.anchors.size());
    for (const auto &anchor : loaded.route.anchors) {
      const auto found =
          current_atom_by_original_id.find(anchor.original_atom_id);
      if (found == current_atom_by_original_id.end()) {
        result.error = "Route anchor original atom ID " +
                       std::to_string(anchor.original_atom_id) +
                       " is absent from routed g1.";
        return result;
      }
      const std::size_t atom = found->second;
      if (!anchor.chain.empty() && rec.info[atom].get_chain() != anchor.chain) {
        result.error = "Route anchor chain mismatch for original atom ID " +
                       std::to_string(anchor.original_atom_id) + ".";
        return result;
      }
      if (rec.info[atom].get_resnr() != anchor.residue_position) {
        result.error = "Route anchor residue-number mismatch for original atom ID " +
                       std::to_string(anchor.original_atom_id) + ": route declares " +
                       std::to_string(anchor.residue_position) +
                       " but current receptor has " +
                       std::to_string(rec.info[atom].get_resnr()) + ".";
        return result;
      }
      if (rec.info[atom].get_atom() != anchor.atom_name) {
        result.error = "Route anchor atom-name mismatch for original atom ID " +
                       std::to_string(anchor.original_atom_id) + ".";
        return result;
      }
      reference_anchors.push_back(anchor.reference_position_angstrom);
      current_anchors.push_back(
          {rec.m_mol[0].pos.x[atom], rec.m_mol[0].pos.y[atom],
           rec.m_mol[0].pos.z[atom]});
    }
    const auto transform =
        titan_path::fit_rigid_transform(reference_anchors, current_anchors);
    if (!transform.valid) {
      result.error = "Route registration failed: " + transform.error;
      return result;
    }
    if (transform.rmsd_angstrom >
        Arguments.de.path_registration_max_rmsd) {
      std::ostringstream message;
      message << "Route registration RMSD " << transform.rmsd_angstrom
              << " A exceeds -pathway-registration-max-rmsd "
              << Arguments.de.path_registration_max_rmsd << " A.";
      result.error = message.str();
      return result;
    }
    result.path.registration_rmsd = transform.rmsd_angstrom;
    registered_route = titan_path::apply_transform(loaded.route, transform);
  }

  result.path.all_nodes.reserve(registered_route.waypoints.size());
  result.path.principal_nodes.reserve(registered_route.waypoints.size());
  for (const auto &waypoint : registered_route.waypoints) {
    PathNode node;
    node.chainnum = 0;
    node.respos = waypoint.index;
    node.chain = registered_route.branch_id.empty()
                     ? "route"
                     : registered_route.branch_id;
    node.res_name = "WP";
    node.com = waypoint.position_angstrom;
    node.radius = waypoint.radius_angstrom;
    result.path.all_nodes.push_back(node);
    result.path.principal_nodes.push_back(std::move(node));
  }
  const auto corridor = titan_path::build_parallel_transport_corridor(
      registered_route, Arguments.de.path_corridor_spacing);
  if (!corridor.ok) {
    result.error = "Explicit route corridor construction failed: " +
                   corridor.error;
    return result;
  }
  result.path.corridor_samples = corridor.samples;
  if (!finalize_explicit_path_geometry(result, lg, Arguments))
    return result;
  result.ok = true;
  return result;
}

PathResult build_path(const Multi_MOL &rec, const Multi_MOL &lg,
                      groups_cal &groups, const arguments &Arguments) {
  if (!Arguments.de.pathway_file.empty())
    return build_explicit_path(rec, lg, Arguments);
  PathResult result;
  result.path.group_index = groups.group_pathway_index();
  result.path.group_name = groups.group_name(result.path.group_index);

  std::unordered_map<std::size_t, std::size_t> rec_local_from_global;
  const std::size_t group1 = groups.group_1_index();
  rec_local_from_global.reserve(groups.groups_atom_number(group1));
  for (std::size_t i = 0; i < groups.groups_atom_number(group1); ++i)
    rec_local_from_global.emplace(groups.atom_index(group1, i), i);

  std::unordered_map<ResidueKey, std::size_t, ResidueKeyHash> seen;
  std::vector<ResidueAcc> residues;
  residues.reserve(groups.groups_atom_number(result.path.group_index));
  for (std::size_t i = 0; i < groups.groups_atom_number(result.path.group_index); ++i) {
    const std::size_t global_atom = groups.atom_index(result.path.group_index, i);
    const auto local_it = rec_local_from_global.find(global_atom);
    if (local_it == rec_local_from_global.end()) {
      result.error = "Pathway atom is not present in routed g1.";
      return result;
    }
    const std::size_t local_atom = local_it->second;
    const mol_info &atom_info = rec.info[local_atom];
    const ResidueKey key{atom_info.get_chainnum(), atom_info.get_respos()};
    auto [it, inserted] = seen.emplace(key, residues.size());
    if (inserted) {
      ResidueAcc acc;
      acc.node.chainnum = atom_info.get_chainnum();
      acc.node.respos = atom_info.get_respos();
      acc.node.chain = atom_info.get_chain();
      acc.node.res_name = atom_info.get_original_res_name();
      acc.node.first_local_atom = local_atom;
      residues.push_back(std::move(acc));
      it->second = residues.size() - 1;
    }
    ResidueAcc &acc = residues[it->second];
    const double mass = rec.parameters[local_atom].get_mass();
    acc.mass += mass;
    acc.weighted[0] += rec.m_mol[0].pos.x[local_atom] * mass;
    acc.weighted[1] += rec.m_mol[0].pos.y[local_atom] * mass;
    acc.weighted[2] += rec.m_mol[0].pos.z[local_atom] * mass;
  }
  for (auto &acc : residues) {
    if (acc.mass <= 0.0)
      continue;
    acc.node.com = scale_pt(acc.weighted, 1.0 / acc.mass);
    result.path.all_nodes.push_back(acc.node);
  }
  if (result.path.all_nodes.size() < 2) {
    result.error = "Pathway group collapses to fewer than 2 residue COM nodes.";
    return result;
  }

  const Point3 lg_start = mol_com(lg.m_mol[0], lg.parameters);
  std::size_t start = 0;
  double best_d2 = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < result.path.all_nodes.size(); ++i) {
    const double d2 = dist_sq(result.path.all_nodes[i].com, lg_start);
    if (d2 < best_d2) {
      best_d2 = d2;
      start = i;
    }
  }

  std::map<std::pair<std::size_t, std::size_t>, double> edges;
  std::map<std::size_t, std::vector<std::size_t>> by_chain;
  for (std::size_t i = 0; i < result.path.all_nodes.size(); ++i)
    by_chain[result.path.all_nodes[i].chainnum].push_back(i);
  for (auto &[chain, nodes] : by_chain) {
    (void)chain;
    std::sort(nodes.begin(), nodes.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (result.path.all_nodes[lhs].respos != result.path.all_nodes[rhs].respos)
        return result.path.all_nodes[lhs].respos <
               result.path.all_nodes[rhs].respos;
      return lhs < rhs;
    });
    for (std::size_t i = 1; i < nodes.size(); ++i)
      add_edge(edges, nodes[i - 1], nodes[i],
               dist_pt(result.path.all_nodes[nodes[i - 1]].com,
                       result.path.all_nodes[nodes[i]].com));
  }

  auto adj = build_adj(result.path.all_nodes.size(), edges);
  auto component = component_nodes(start, adj);
  if (component.size() < result.path.all_nodes.size()) {
    result.path.used_geometric_edges = true;
    for (std::size_t i = 0; i < result.path.all_nodes.size(); ++i) {
      std::vector<std::pair<double, std::size_t>> nearest;
      for (std::size_t j = 0; j < result.path.all_nodes.size(); ++j) {
        if (i == j)
          continue;
        nearest.push_back({dist_sq(result.path.all_nodes[i].com,
                                   result.path.all_nodes[j].com),
                           j});
      }
      std::sort(nearest.begin(), nearest.end());
      for (std::size_t k = 0; k < std::min(kNearestResidueEdges, nearest.size());
           ++k) {
        const double distance = std::sqrt(nearest[k].first);
        if (distance > Arguments.de.path_graph_edge_cutoff)
          continue;
        add_edge(edges, i, nearest[k].second, distance);
      }
    }
    adj = build_adj(result.path.all_nodes.size(), edges);
    component = component_nodes(start, adj);
  }
  if (component.size() < 2) {
    result.error = "Pathway component near g2 has fewer than 2 nodes.";
    return result;
  }
  result.path.pruned_component_nodes =
      result.path.all_nodes.size() - component.size();

  std::unordered_map<std::size_t, std::size_t> local;
  for (std::size_t i = 0; i < component.size(); ++i)
    local.emplace(component[i], i);

  struct EdgeRec { std::size_t a{}, b{}; double w{}; };
  std::vector<EdgeRec> mst_input;
  for (const auto &[edge, weight] : edges) {
    const auto a = local.find(edge.first);
    const auto b = local.find(edge.second);
    if (a == local.end() || b == local.end())
      continue;
    mst_input.push_back({a->second, b->second, weight});
  }
  std::sort(mst_input.begin(), mst_input.end(),
            [](const EdgeRec &lhs, const EdgeRec &rhs) {
              if (lhs.w != rhs.w)
                return lhs.w < rhs.w;
              if (lhs.a != rhs.a)
                return lhs.a < rhs.a;
              return lhs.b < rhs.b;
            });
  DisjointSet dsu(component.size());
  std::vector<std::vector<std::pair<std::size_t, double>>> tree(component.size());
  for (const auto &edge : mst_input) {
    if (!dsu.unite(edge.a, edge.b))
      continue;
    tree[edge.a].push_back({edge.b, edge.w});
    tree[edge.b].push_back({edge.a, edge.w});
  }

  const std::size_t local_start = local[start];
  std::vector<int> parent(component.size(), -1);
  std::vector<double> geodesic(component.size(), 0.0);
  std::vector<std::size_t> stack{local_start};
  parent[local_start] = static_cast<int>(local_start);
  while (!stack.empty()) {
    const std::size_t node = stack.back();
    stack.pop_back();
    for (const auto &[next, weight] : tree[node]) {
      if (parent[next] != -1)
        continue;
      parent[next] = static_cast<int>(node);
      geodesic[next] = geodesic[node] + weight;
      stack.push_back(next);
    }
  }

  std::size_t local_terminal = local_start;
  for (std::size_t node = 0; node < component.size(); ++node) {
    const auto &candidate = result.path.all_nodes[component[node]];
    const auto &best = result.path.all_nodes[component[local_terminal]];
    const double cand_d2 = dist_sq(candidate.com, lg_start);
    const double best_terminal_d2 = dist_sq(best.com, lg_start);
    if (geodesic[node] > geodesic[local_terminal] + 1.0e-9) {
      local_terminal = node;
      continue;
    }
    if (std::fabs(geodesic[node] - geodesic[local_terminal]) <= 1.0e-9) {
      if (cand_d2 > best_terminal_d2 + 1.0e-9) {
        local_terminal = node;
        continue;
      }
      if (std::fabs(cand_d2 - best_terminal_d2) <= 1.0e-9 &&
          (candidate.chainnum < best.chainnum ||
           (candidate.chainnum == best.chainnum &&
            candidate.respos < best.respos))) {
        local_terminal = node;
      }
    }
  }

  std::vector<std::size_t> principal;
  for (std::size_t cursor = local_terminal;; cursor = static_cast<std::size_t>(parent[cursor])) {
    principal.push_back(component[cursor]);
    if (cursor == local_start)
      break;
  }
  std::reverse(principal.begin(), principal.end());
  for (std::size_t global_id : principal) {
    if (!result.path.principal_nodes.empty() &&
        dist_pt(result.path.principal_nodes.back().com,
                result.path.all_nodes[global_id].com) <= kPathTol)
      continue;
    result.path.principal_nodes.push_back(result.path.all_nodes[global_id]);
  }
  if (result.path.principal_nodes.size() < 2) {
    result.error = "Principal pathway degenerates to fewer than 2 unique nodes.";
    return result;
  }
  result.path.pruned_branch_nodes =
      component.size() - result.path.principal_nodes.size();

  double s = 0.0;
  for (std::size_t i = 1; i < result.path.principal_nodes.size(); ++i) {
    PathSegment segment;
    segment.a = result.path.principal_nodes[i - 1].com;
    segment.d = sub_pt(result.path.principal_nodes[i].com, segment.a);
    segment.length = std::sqrt(norm_sq(segment.d));
    if (segment.length <= kPathTol)
      continue;
    segment.s0 = s;
    s += segment.length;
    result.path.segments.push_back(segment);
  }
  result.path.total_length = s;
  if (result.path.segments.empty() || result.path.total_length <= kPathTol) {
    result.error = "Principal pathway has zero usable arclength.";
    return result;
  }

  std::vector<Interval> valid;
  for (const auto &segment : result.path.segments) {
    auto shell = shell_intervals(segment, lg_start,
                                 Arguments.de.de_min_scan_radius,
                                 Arguments.de.de_scan_radius);
    for (const auto &interval : shell) {
      valid.push_back(
          {segment.s0 + interval.begin * segment.length,
           segment.s0 + interval.end * segment.length});
    }
  }
  result.path.valid_intervals = merge_intervals(std::move(valid));
  if (result.path.valid_intervals.empty()) {
    result.error =
        "Forward pathway does not intersect the requested scan-radius window.";
    return result;
  }

  result.ok = true;
  return result;
}

Point3 point_on_path(const PathContext &path, double s) {
  if (s <= 0.0)
    return path.principal_nodes.front().com;
  if (s >= path.total_length)
    return path.principal_nodes.back().com;
  for (const auto &segment : path.segments) {
    if (s <= segment.s0 + segment.length + 1.0e-9) {
      const double t = (s - segment.s0) / segment.length;
      return add_pt(segment.a, scale_pt(segment.d, t));
    }
  }
  return path.principal_nodes.back().com;
}

double clamp_to_intervals(double value, const std::vector<Interval> &intervals) {
  if (intervals.empty())
    return value;
  if (value <= intervals.front().begin)
    return intervals.front().begin;
  if (value >= intervals.back().end)
    return intervals.back().end;
  for (const auto &interval : intervals) {
    if (value >= interval.begin && value <= interval.end)
      return value;
  }
  double best = intervals.front().begin;
  double best_gap = std::numeric_limits<double>::max();
  for (const auto &interval : intervals) {
    const double left = std::fabs(value - interval.begin);
    if (left < best_gap) {
      best_gap = left;
      best = interval.begin;
    }
    const double right = std::fabs(value - interval.end);
    if (right < best_gap) {
      best_gap = right;
      best = interval.end;
    }
  }
  return best;
}

double sample_interval(std::mt19937 &mt, const std::vector<Interval> &intervals) {
  double total = 0.0;
  for (const auto &interval : intervals)
    total += interval.end - interval.begin;
  std::uniform_real_distribution<double> pick(0.0, total);
  double marker = pick(mt);
  for (const auto &interval : intervals) {
    const double len = interval.end - interval.begin;
    if (marker <= len)
      return interval.begin + marker;
    marker -= len;
  }
  return intervals.back().end;
}

PathProjection project_point_to_path(
    const PathContext &path, const Point3 &point,
    double reference_progress = 0.0, double min_progress = 0.0,
    double max_progress = std::numeric_limits<double>::infinity()) {
  const auto projection = titan_path::project_point_to_polyline(
      path.segments, point, reference_progress, min_progress, max_progress);
  return {projection.valid, projection.progress_angstrom,
          projection.distance_angstrom, projection.point_angstrom,
          projection.segment_index};
}

std::vector<Interval> build_forward_intervals(const PathContext &path,
                                              const Point3 &center,
                                              double current_progress,
                                              double min_radius,
                                              double max_radius) {
  std::vector<Interval> valid;
  for (const auto &segment : path.segments) {
    const double segment_end = segment.s0 + segment.length;
    if (segment_end + 1.0e-9 < current_progress)
      continue;

    auto shell = shell_intervals(segment, center, min_radius, max_radius);
    for (const auto &interval : shell) {
      double begin = segment.s0 + interval.begin * segment.length;
      double end = segment.s0 + interval.end * segment.length;
      begin = std::max(begin, current_progress);
      end = std::min(end, segment_end);
      if (begin <= end + 1.0e-9)
        valid.push_back({begin, end});
    }
  }
  return merge_intervals(std::move(valid));
}

std::string pathway_mode_label(PathwayMode mode) {
  return mode == PathwayMode::Recovery ? "Recovery" : "Advance";
}

std::size_t total_pathway_evaluations_used(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.evaluations_done;
  return total;
}

std::size_t total_pathway_candidate_proposals(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.candidate_proposals;
  return total;
}

std::size_t total_pathway_energy_evaluations(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.energy_evaluations_done;
  return total;
}

std::size_t total_pathway_scan_shell_exclusions(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.scan_shell_exclusions;
  return total;
}

std::size_t total_pathway_clearance_pair_checks(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.clearance_pair_checks;
  return total;
}

std::size_t total_pathway_clearance_rejections(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.clearance_rejections;
  return total;
}

std::size_t total_pathway_probe_evaluations(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.probe_evaluations;
  return total;
}

std::size_t total_pathway_probe_clearance_rejections(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.probe_clearance_rejections;
  return total;
}

std::size_t total_pathway_probe_mode_rejections(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.probe_mode_rejections;
  return total;
}

std::size_t total_pathway_swept_probe_evaluations(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.swept_clearance_probe_evaluations;
  return total;
}

std::size_t total_pathway_swept_pair_checks(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.swept_clearance_pair_checks;
  return total;
}

std::size_t total_pathway_swept_rejections(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.swept_clearance_rejections;
  return total;
}

std::size_t total_pathway_swept_interior_only_rejections(
    const std::vector<PathwayWorkerContext> &workers) {
  std::size_t total = 0;
  for (const auto &worker : workers)
    total += worker.swept_interior_only_rejections;
  return total;
}

std::size_t pathway_parallel_grain(std::size_t total_items,
                                   std::size_t requested_workers,
                                   std::size_t ligand_atom_count) {
  const std::size_t workers = std::max<std::size_t>(1, requested_workers);
  if (total_items <= workers)
    return 1;
  const std::size_t min_items_per_worker =
      ligand_atom_count >= 2048 ? 4
      : ligand_atom_count >= 1024 ? 6
      : ligand_atom_count >= 256  ? 8
                                : 12;
  return std::max<std::size_t>(
      min_items_per_worker,
      (total_items + workers * 2 - 1) / (workers * 2));
}

bool pathway_candidate_improves(const PathwayCandidateState &lhs,
                                const PathwayCandidateState &rhs,
                                PathwayMode mode) {
  return pathway_candidate_better(lhs, 0, rhs, 0, mode);
}

std::vector<PathwayReportRow> build_grouped_report_rows(
    const std::vector<PathwayCandidateState> &population, PathwayMode mode) {
  std::vector<std::pair<double, std::size_t>> by_progress;
  by_progress.reserve(population.size());
  for (std::size_t i = 0; i < population.size(); ++i) {
    if (population[i].valid) {
      if (!std::isfinite(population[i].progress))
        throw std::invalid_argument("nonfinite report progress");
      by_progress.push_back({population[i].progress, i});
    }
  }
  std::sort(by_progress.begin(), by_progress.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first)
                return lhs.first < rhs.first;
              return lhs.second < rhs.second;
            });

  std::vector<PathwayReportRow> rows;
  for (const auto &[progress, index] : by_progress) {
    (void)progress;
    if (rows.empty() ||
        std::fabs(population[index].progress -
                  rows.back().representative.progress) > kPathwayProgressDedupEps) {
      PathwayReportRow row;
      row.representative = population[index];
      row.representative_index = index;
      row.count = 1;
      rows.push_back(row);
      continue;
    }
    auto &row = rows.back();
    ++row.count;
    if (pathway_candidate_better(population[index], index, row.representative,
                                 row.representative_index, mode)) {
      row.representative = population[index];
      row.representative_index = index;
    }
  }

  std::stable_sort(rows.begin(), rows.end(),
                   [&](const PathwayReportRow &lhs,
                       const PathwayReportRow &rhs) {
                     return pathway_candidate_strict_less(
                         lhs.representative, lhs.representative_index,
                         rhs.representative, rhs.representative_index, mode);
                   });
  return rows;
}

std::string variant_pattern_section_title(PullDeVariant variant) {
  switch (variant) {
  case PullDeVariant::NlShadeLbc:
    return "NL-SHADE-LBC Pattern Refinement";
  case PullDeVariant::Jso:
    return "JSO Pattern Refinement";
  default:
    return "Pathway Pattern Refinement";
  }
}

std::optional<PathwayMode>
read_previous_pathway_mode(const std::string &status_filename) {
  std::ifstream input(status_filename);
  std::string line;
  while (std::getline(input, line)) {
    if (line == "MODE=Recovery" || line == "MODE=RECOVERY")
      return PathwayMode::Recovery;
    if (line == "MODE=Advance" || line == "MODE=ADVANCE")
      return PathwayMode::Advance;
  }
  return std::nullopt;
}

PathwayFinalSelection select_pathway_final_candidate(
    const std::vector<PathwayCandidateState> &archive, PathwayMode mode,
    double energy_key_epsilon, bool geometry_only,
    double current_path_distance, bool fixed_probe_enabled,
    bool navigation_enabled, double navigation_cost_epsilon) {
  PathwayFinalSelection selection;
  std::vector<titan_path::SelectionCandidate> candidates;
  candidates.reserve(archive.size());
  for (const auto &candidate : archive) {
    titan_path::SelectionCandidate mapped{
        candidate.energy_key, candidate.delta_progress,
        candidate.pull_distance, candidate.minimum_clearance,
        candidate.angle_deg, candidate.valid};
    mapped.probe_length = candidate.probe_length;
    mapped.fixed_probe = candidate.fixed_probe;
    mapped.path_distance = candidate.applied_path_distance;
    mapped.reference_path_distance = current_path_distance;
    mapped.navigation_enabled = candidate.navigation_enabled;
    mapped.navigation_cost = candidate.navigation_cost;
    candidates.push_back(mapped);
  }
  const auto selected = titan_path::select_epsilon_progress_candidate(
      candidates,
      mode == PathwayMode::Recovery ? titan_path::RouteMode::Recovery
                                    : titan_path::RouteMode::Advance,
      energy_key_epsilon, geometry_only, fixed_probe_enabled,
      navigation_enabled && mode == PathwayMode::Advance
          ? std::optional<double>(navigation_cost_epsilon)
          : std::nullopt);
  selection.minimum_energy_key = selected.minimum_energy_key;
  selection.valid_archive_size = selected.valid_count;
  selection.forward_archive_size = selected.selectable_count;
  selection.forward_progress_rejections = selected.nonforward_count;
  selection.epsilon_eligible_size = selected.eligible_count;
  selection.mode_eligible_size = selected.mode_eligible_count;
  selection.maximum_probe_tier_size = selected.maximum_probe_tier_count;
  selection.navigation_eligible_size = selected.navigation_eligible_count;
  selection.minimum_navigation_cost = selected.minimum_navigation_cost;
  if (selected.found)
    selection.candidate = archive[selected.index];
  return selection;
}

PathwayState prepare_pathway_state(
    const PathContext &path, const Point3 &start_com,
    const arguments &Arguments,
    const std::optional<PathwayMode> previous_mode = std::nullopt,
    const std::optional<double> previous_progress = std::nullopt,
    bool observed_boundary = false) {
  PathwayState state;
  if (previous_progress.has_value()) {
    const auto projection = titan_path::project_with_continuity(
        path.segments, start_com, *previous_progress,
        observed_boundary ? 1.0 : Arguments.de.path_continuity_window, path.total_length);
    state.current_projection = {projection.valid, projection.progress_angstrom,
        projection.distance_angstrom, projection.point_angstrom, projection.segment_index};
    state.previous_progress = previous_progress;
    state.continuity_half_window = observed_boundary ? 1.0 : Arguments.de.path_continuity_window;
  } else {
    state.current_projection = project_point_to_path(path, start_com);
  }
  state.forward_intervals = build_forward_intervals(
      path, start_com, state.current_projection.progress,
      Arguments.de.de_min_scan_radius, Arguments.de.de_scan_radius);
  std::optional<titan_path::RouteMode> previous;
  if (previous_mode.has_value())
    previous = previous_mode == PathwayMode::Recovery
                   ? titan_path::RouteMode::Recovery
                   : titan_path::RouteMode::Advance;
  const auto next = titan_path::next_route_mode(
      previous, state.current_projection.distance,
      Arguments.de.path_recovery_enter_radius,
      Arguments.de.path_recovery_exit_radius);
  state.mode = next == titan_path::RouteMode::Recovery
                   ? PathwayMode::Recovery
                   : PathwayMode::Advance;
  return state;
}

PathwayCandidateState evaluate_candidate(
    const PathContext &path, const PathwayState &state, double progress,
    const Point3 &start_com, const Multi_MOL &rec, const Multi_MOL &lg,
    PathwayWorkerContext &worker, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    bool budget_granted, const Point3 *target_override = nullptr,
    double corridor_offset_u = 0.0, double corridor_offset_v = 0.0,
    double corridor_radius = 0.0, bool bypass_scan_shell = false) {
  ++worker.candidate_proposals;
  PathwayCandidateState result;
  result.progress = progress;
  result.requested_delta_progress =
      progress - state.current_projection.progress;
  const Point3 target =
      target_override == nullptr ? point_on_path(path, progress)
                                 : *target_override;
  result.requested_vector = force_from_points(start_com, target);
  result.vector = result.requested_vector;
  result.requested_pull_distance = result.requested_vector.modul();
  result.pull_distance = result.requested_pull_distance;
  result.corridor_offset_u = corridor_offset_u;
  result.corridor_offset_v = corridor_offset_v;
  result.corridor_radius = corridor_radius;
  if (!bypass_scan_shell && !titan_path::candidate_within_scan_shell(
          result.requested_pull_distance, DE_arguments.min_scan_radius,
          DE_arguments.scan_radius)) {
    ++worker.scan_shell_exclusions;
    return result;
  }
  result.requested_angle_deg =
      angle_to_previous_vector(result.requested_vector,
                               DE_arguments.old_vector);
  if (DE_arguments.flag_old_vec) {
    const auto projected = project_vector_to_turn_cone(
        result.requested_vector, DE_arguments.old_vector,
        state.mode == PathwayMode::Recovery &&
                Arguments.de.path_probe_length > 0.0
            ? Arguments.de.path_recovery_max_turn_angle
            : Arguments.de.path_max_turn_angle,
        result.turn_capped);
    if (!projected.has_value())
      return result;
    result.vector = *projected;
  }
  const force direction_before_probe_scaling = result.vector;
  result.angle_deg =
      angle_to_previous_vector(result.vector, DE_arguments.old_vector);
  if (!budget_granted) {
    return result;
  }
  ++worker.evaluations_done;

  const bool fixed_probe = Arguments.de.path_probe_length > 0.0;
  const std::size_t probe_count = fixed_probe
                                      ? std::max<std::size_t>(
                                            1, Arguments.de.path_probe_lengths.size())
                                      : 1;
  bool clearance_passed = false;
  bool mode_eligible_probe_selected = false;
  std::optional<PathwayCandidateState> mode_ineligible_fallback;
  for (std::size_t probe_index = 0; probe_index < probe_count; ++probe_index) {
    result.vector = direction_before_probe_scaling;
    if (fixed_probe) {
      const double probe_length = Arguments.de.path_probe_lengths.empty()
                                      ? Arguments.de.path_probe_length
                                      : Arguments.de.path_probe_lengths[probe_index];
      const auto scaled = titan_path::scale_vector_to_length(
          {result.vector.Fx, result.vector.Fy, result.vector.Fz}, probe_length);
      if (!scaled.has_value())
        return result;
      result.vector = force{(*scaled)[0], (*scaled)[1], (*scaled)[2]};
    }
    result.fixed_probe = fixed_probe;
    result.pull_distance = result.vector.modul();
    result.probe_length = result.pull_distance;
    const Point3 applied_endpoint =
        add_pt(start_com,
               {result.vector.Fx, result.vector.Fy, result.vector.Fz});
    // Keep the endpoint projection local to the current route coordinate.
    // The requested target may be far down a self-near polyline and must not
    // widen the continuity window or manufacture a nonlocal progress jump.
    const double projection_half_window =
        std::max(result.pull_distance, Arguments.de.path_corridor_spacing);
    const double projection_min = std::max(
        0.0, state.current_projection.progress - projection_half_window);
    const double projection_max = std::min(
        path.total_length,
        state.current_projection.progress + projection_half_window);
    const PathProjection applied_projection = project_point_to_path(
        path, applied_endpoint, state.current_projection.progress,
        projection_min, projection_max);
    if (!applied_projection.valid)
      return result;
    result.applied_progress = applied_projection.progress;
    result.delta_progress =
        applied_projection.progress - state.current_projection.progress;
    result.applied_path_distance = applied_projection.distance;
    result.navigation_enabled = Arguments.de.path_navigation_enabled;
    if (result.navigation_enabled) {
      const auto applied_frame = titan_path::sample_corridor_at_progress(
          path.corridor_samples, result.applied_progress);
      const auto navigation = titan_path::evaluate_navigation_cost(
          result.applied_path_distance, applied_frame.radius_angstrom,
          result.delta_progress, result.pull_distance,
          Arguments.de.path_navigation_tube_weight,
          Arguments.de.path_navigation_backtrack_weight);
      result.applied_corridor_radius =
          navigation.applied_corridor_radius_angstrom;
      result.navigation_tube_excess = navigation.tube_excess_angstrom;
      result.navigation_backtrack_distance =
          navigation.backtrack_distance_angstrom;
      result.navigation_tube_cost = navigation.tube_cost;
      result.navigation_backtrack_cost = navigation.backtrack_cost;
      result.navigation_cost = navigation.total_cost;
    }

    ++worker.probe_evaluations;
    const bool swept_clearance =
        Arguments.de.path_clearance_mode == "swept-linear-center";
    ClearanceEvaluation clearance;
    if (swept_clearance) {
      ++worker.swept_clearance_probe_evaluations;
      clearance = evaluate_interatomic_swept_clearance(
          rec.m_mol[0], lg.m_mol[0], result.vector,
          Arguments.de.path_min_clearance);
      worker.swept_clearance_pair_checks += clearance.pair_checks;
    } else {
      worker.scratch = lg.m_mol[0];
      worker.scratch.translate(result.vector.Fx, result.vector.Fy,
                               result.vector.Fz);
      clearance = evaluate_interatomic_clearance(
          rec.m_mol[0], worker.scratch, Arguments.de.path_min_clearance);
    }
    worker.clearance_pair_checks += clearance.pair_checks;
    result.minimum_clearance = clearance.minimum_distance;
    result.endpoint_minimum_clearance =
        clearance.endpoint_minimum_distance;
    result.swept_minimum_clearance =
        swept_clearance ? clearance.minimum_distance
                        : clearance.endpoint_minimum_distance;
    result.swept_minimum_fraction = clearance.minimum_fraction;
    result.swept_minimum_receptor_index = clearance.receptor_index;
    result.swept_minimum_ligand_index = clearance.ligand_index;
    result.swept_minimum_complete = clearance.global_minimum_complete;
    if (!clearance.passes_threshold) {
      ++worker.probe_clearance_rejections;
      if (swept_clearance)
        ++worker.swept_clearance_rejections;
      if (swept_clearance &&
          clearance.threshold_violation_hidden_at_endpoints)
        ++worker.swept_interior_only_rejections;
      continue;
    }
    clearance_passed = true;
    const bool advances = result.delta_progress > 1.0e-6;
    const bool descends =
        result.applied_path_distance + 1.0e-6 <
        state.current_projection.distance;
    const bool mode_eligible =
        !fixed_probe ||
        (state.mode == PathwayMode::Advance ? advances : descends);
    if (!mode_eligible) {
      ++worker.probe_mode_rejections;
      if (!mode_ineligible_fallback.has_value())
        mode_ineligible_fallback = result;
      continue;
    }
    mode_eligible_probe_selected = true;
    break;
  }
  if (!clearance_passed) {
    ++worker.clearance_rejections;
    return result;
  }
  if (!mode_eligible_probe_selected) {
    if (!mode_ineligible_fallback.has_value())
      return result;
    result = *mode_ineligible_fallback;
  }
  // The last attempted probe may differ from the retained fallback tier.
  // Reconstruct the exact retained displacement before the single energy call.
  worker.scratch = lg.m_mol[0];
  worker.scratch.translate(result.vector.Fx, result.vector.Fy,
                           result.vector.Fz);
  if (Arguments.de.path_geometry_only) {
    result.raw_score = 0.0;
    result.energy_key = 0.0;
    result.valid = std::isfinite(result.pull_distance) &&
                   std::isfinite(result.applied_progress) &&
                   std::isfinite(result.delta_progress) &&
                   std::isfinite(result.applied_path_distance) &&
                   (!result.navigation_enabled ||
                    std::isfinite(result.navigation_cost));
    return result;
  }
  ++worker.energy_evaluations_done;
  eval_energy_model(rec.m_mol[0], worker.scratch, rec.info, lg.info,
                    worker.eval_cache, result.energy, Arguments, rec.parameters,
                    lg.parameters);
  result.raw_score =
      base_score(result.energy, energy_ref, Arguments.de.de_flag_score_function);
  result.energy_key = stable_pathway_energy_key(
      result.energy, energy_ref, Arguments.de.de_flag_score_function,
      result.raw_score);
  result.valid = std::isfinite(result.raw_score) &&
                  std::isfinite(result.energy_key) &&
                  std::isfinite(result.pull_distance) &&
                  std::isfinite(result.applied_progress) &&
                  std::isfinite(result.delta_progress) &&
                  std::isfinite(result.applied_path_distance) &&
                  (!result.navigation_enabled ||
                   std::isfinite(result.navigation_cost));
  return result;
}

PullDeBatchEvaluation evaluate_candidate_batch(
    std::span<const double> proposals, std::span<PathwayCandidateState> results,
    const PathContext &path, const PathwayState &state, const Point3 &start_com,
    const Multi_MOL &rec, const Multi_MOL &lg,
    std::vector<PathwayWorkerContext> &workers, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    PathwayEvalBudget &eval_budget, TitanExecutor &executor,
    std::size_t requested_workers) {
  PullDeBatchEvaluation batch;
  if (proposals.empty())
    return batch;

  std::vector<std::pair<double, std::size_t>> ordered;
  ordered.reserve(proposals.size());
  for (std::size_t i = 0; i < proposals.size(); ++i) {
    if (!std::isfinite(proposals[i]))
      throw std::invalid_argument("nonfinite pathway proposal progress");
    ordered.push_back({proposals[i], i});
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto &lhs, const auto &rhs) {
              if (lhs.first != rhs.first)
                return lhs.first < rhs.first;
              return lhs.second < rhs.second;
            });

  std::vector<double> unique_progress;
  unique_progress.reserve(ordered.size());
  std::vector<std::size_t> proposal_to_unique(proposals.size(), 0);
  for (const auto &[progress, index] : ordered) {
    if (unique_progress.empty() ||
        std::fabs(progress - unique_progress.back()) > kPathwayProgressDedupEps) {
      unique_progress.push_back(progress);
    }
    proposal_to_unique[index] = unique_progress.size() - 1;
  }
  std::vector<PathwayCandidateState> unique_results(unique_progress.size());
  std::vector<unsigned char> budget_granted(unique_progress.size(), 0);

  std::size_t available = std::numeric_limits<std::size_t>::max();
  if (eval_budget.max_evals != 0) {
    const std::size_t reserved =
        eval_budget.reserved.load(std::memory_order_relaxed);
    available = reserved >= eval_budget.max_evals
                    ? 0
                    : eval_budget.max_evals - reserved;
  }
  std::size_t granted = 0;
  for (std::size_t i = 0; i < unique_progress.size(); ++i) {
    if (granted < available) {
      budget_granted[i] = 1;
      ++granted;
    }
  }
  if (eval_budget.max_evals != 0)
    eval_budget.reserved.fetch_add(granted, std::memory_order_relaxed);

  const std::size_t before = total_pathway_evaluations_used(workers);
  executor.parallel_for(
      unique_progress.size(), requested_workers,
      pathway_parallel_grain(unique_progress.size(), requested_workers,
                             lg.m_mol[0].size()),
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &worker = workers[worker_id];
        for (std::size_t i = begin; i < end; ++i) {
          unique_results[i] =
              evaluate_candidate(path, state, unique_progress[i], start_com, rec,
                                  lg, worker, energy_ref, Arguments,
                                  DE_arguments, budget_granted[i] != 0);
        }
      });
  const std::size_t after = total_pathway_evaluations_used(workers);
  for (std::size_t i = 0; i < proposals.size(); ++i)
    results[i] = unique_results[proposal_to_unique[i]];
  batch.evaluations_used = after - before;
  batch.budget_exhausted =
      (eval_budget.max_evals != 0 &&
       eval_budget.reserved.load(std::memory_order_relaxed) >=
           eval_budget.max_evals);
  return batch;
}

std::vector<double>
build_corridor_progress_grid(const std::vector<Interval> &intervals,
                             double spacing) {
  if (!std::isfinite(spacing) || spacing <= 0.0)
    throw std::invalid_argument("corridor grid spacing must be finite and positive");
  std::vector<double> progress;
  for (const auto &interval : intervals) {
    const double count_bound = std::ceil((interval.end-interval.begin)/spacing)+2.0;
    if (!std::isfinite(interval.begin) || !std::isfinite(interval.end) ||
        interval.end < interval.begin || !std::isfinite(count_bound) ||
        count_bound > static_cast<double>(titan_path::kMaximumCorridorSamples-progress.size()))
      throw std::invalid_argument("corridor grid sample count exceeds resource limit");
    progress.push_back(interval.begin);
    for (double value = interval.begin + spacing;
         value < interval.end - kPathwayProgressDedupEps; value += spacing) {
      if (value+spacing <= value || progress.size() >= titan_path::kMaximumCorridorSamples)
        throw std::invalid_argument("corridor grid increment/count is not representable");
      progress.push_back(value);
    }
    progress.push_back(interval.end);
  }
  std::sort(progress.begin(), progress.end());
  std::vector<double> grouped;
  grouped.reserve(progress.size());
  for (double value : progress)
    if (grouped.empty() || value-grouped.back() > kPathwayProgressDedupEps)
      grouped.push_back(value);
  return grouped;
}

PathwayCorridorGridStats run_pathway_corridor_grid(
    const PathContext &path, const PathwayState &state,
    const Point3 &start_com, const Multi_MOL &rec, const Multi_MOL &lg,
    std::vector<PathwayWorkerContext> &workers, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    PathwayEvalBudget &eval_budget, TitanExecutor &executor,
    std::size_t requested_workers,
    std::vector<PathwayCandidateState> &candidates,
    PathwayCandidateState &best_candidate) {
  PathwayCorridorGridStats stats;
  stats.enabled = true;
  if (path.corridor_samples.empty())
    return stats;

  const auto progress_grid = build_corridor_progress_grid(
      state.forward_intervals, Arguments.de.path_corridor_spacing);
  const std::size_t radial_samples = Arguments.de.path_corridor_radial_samples;
  constexpr std::size_t kMaximumCorridorProposals = 1000000;
  if (radial_samples >= kMaximumCorridorProposals ||
      progress_grid.size() > kMaximumCorridorProposals/(radial_samples+1))
    throw std::invalid_argument("corridor proposal count exceeds resource limit (1000000)");
  stats.longitudinal_points = progress_grid.size();
  std::vector<titan_path::CorridorSample> frames;
  frames.reserve(progress_grid.size());
  for (const double progress : progress_grid)
    frames.push_back(titan_path::sample_corridor_at_progress(
        path.corridor_samples, progress));

  std::vector<PathwayCorridorProposal> proposals;
  proposals.reserve(progress_grid.size() * (radial_samples + 1));
  for (const auto &frame : frames) {
    proposals.push_back({frame.progress_angstrom, 0.0, 0.0,
                         frame.radius_angstrom, frame.center_angstrom});
  }
  constexpr double kGoldenAngle = 2.39996322972865332223;
  for (std::size_t radial = 0; radial < radial_samples; ++radial) {
    const double fraction =
        std::sqrt((static_cast<double>(radial) + 0.5) /
                  static_cast<double>(radial_samples));
    const double angle = kGoldenAngle * static_cast<double>(radial);
    for (const auto &frame : frames) {
      const double offset_u =
          frame.radius_angstrom * fraction * std::cos(angle);
      const double offset_v =
          frame.radius_angstrom * fraction * std::sin(angle);
      proposals.push_back(
          {frame.progress_angstrom, offset_u, offset_v,
           frame.radius_angstrom,
           titan_path::corridor_point(frame, offset_u, offset_v)});
    }
  }
  stats.proposed_targets = proposals.size();
  candidates.assign(proposals.size(), PathwayCandidateState{});

  std::size_t available = std::numeric_limits<std::size_t>::max();
  if (eval_budget.max_evals != 0) {
    const std::size_t reserved =
        eval_budget.reserved.load(std::memory_order_relaxed);
    available = reserved >= eval_budget.max_evals
                    ? 0
                    : eval_budget.max_evals - reserved;
  }
  std::vector<unsigned char> grants(proposals.size(), 0);
  std::size_t feasible = 0;
  std::size_t granted = 0;
  for (std::size_t i = 0; i < proposals.size(); ++i) {
    const force vector = force_from_points(start_com, proposals[i].target);
    const double pull_distance = vector.modul();
    const bool distance_feasible =
        pull_distance + 1.0e-9 >= DE_arguments.min_scan_radius &&
        pull_distance <= DE_arguments.scan_radius + 1.0e-9;
    if (!distance_feasible)
      continue;
    ++feasible;
    if (granted < available) {
      grants[i] = 1;
      ++granted;
    }
  }
  if (eval_budget.max_evals != 0)
    eval_budget.reserved.fetch_add(granted, std::memory_order_relaxed);

  executor.parallel_for(
      proposals.size(), requested_workers,
      pathway_parallel_grain(proposals.size(), requested_workers,
                             lg.m_mol[0].size()),
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &worker = workers[worker_id];
        for (std::size_t i = begin; i < end; ++i) {
          candidates[i] = evaluate_candidate(
              path, state, proposals[i].progress, start_com, rec, lg, worker,
              energy_ref, Arguments, DE_arguments, grants[i] != 0,
              &proposals[i].target, proposals[i].offset_u,
              proposals[i].offset_v, proposals[i].radius);
        }
      });

  stats.evaluated_targets = granted;
  stats.stopped_by_eval_budget = granted < feasible;
  std::size_t best_index = 0;
  bool found = false;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].valid)
      continue;
    ++stats.valid_targets;
    if (!found || pathway_candidate_better(candidates[i], i,
                                           candidates[best_index], best_index,
                                           state.mode)) {
      best_index = i;
      found = true;
    }
  }
  if (found)
    best_candidate = candidates[best_index];
  return stats;
}

PathwayGuidedSeedStats run_pathway_guided_seeds(
    const PathContext &path, const PathwayState &state,
    const Point3 &start_com, const Multi_MOL &rec, const Multi_MOL &lg,
    std::vector<PathwayWorkerContext> &workers, const cache_energy &energy_ref,
    const arguments &Arguments, const de_arg &DE_arguments,
    PathwayEvalBudget &eval_budget, TitanExecutor &executor,
    std::size_t requested_workers,
    std::vector<PathwayCandidateState> &candidates,
    PathwayCandidateState &best_candidate) {
  PathwayGuidedSeedStats stats;
  stats.enabled = true;
  if (path.corridor_samples.empty() ||
      Arguments.de.path_probe_length <= 0.0)
    return stats;

  const double nominal_probe = Arguments.de.path_probe_lengths.empty()
                                   ? Arguments.de.path_probe_length
                                   : Arguments.de.path_probe_lengths.front();
  const auto current_frame = titan_path::sample_corridor_at_progress(
      path.corridor_samples, state.current_projection.progress);
  const titan_path::PathProjection public_projection{
      state.current_projection.valid,
      state.current_projection.progress,
      state.current_projection.distance,
      state.current_projection.point,
      state.current_projection.segment_index};
  const auto guide = titan_path::build_navigation_guide_direction(
      start_com, public_projection, current_frame,
      state.mode == PathwayMode::Recovery ? titan_path::RouteMode::Recovery
                                          : titan_path::RouteMode::Advance,
      nominal_probe);
  stats.guide_direction = guide;
  std::optional<titan_path::Vec3> old_direction;
  if (DE_arguments.flag_old_vec) {
    old_direction = titan_path::Vec3{DE_arguments.old_vector.Fx,
                                     DE_arguments.old_vector.Fy,
                                     DE_arguments.old_vector.Fz};
  }
  const auto seeds = titan_path::build_navigation_seed_directions(
      guide, old_direction, current_frame,
      Arguments.de.path_navigation_seed_radial_samples,
      Arguments.de.path_navigation_seed_amplitude);
  stats.proposed_directions = seeds.size();
  candidates.assign(seeds.size(), PathwayCandidateState{});
  if (seeds.empty())
    return stats;

  std::vector<Point3> targets(seeds.size());
  std::vector<double> requested_progress(seeds.size(),
                                         state.current_projection.progress);
  const double projection_half_window =
      std::max(nominal_probe, Arguments.de.path_corridor_spacing);
  const double projection_min = std::max(
      0.0, state.current_projection.progress - projection_half_window);
  const double projection_max = std::min(
      path.total_length,
      state.current_projection.progress + projection_half_window);
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    targets[index] = add_pt(
        start_com, scale_pt(seeds[index].direction_unit, nominal_probe));
    const auto projected = project_point_to_path(
        path, targets[index], state.current_projection.progress,
        projection_min, projection_max);
    if (projected.valid)
      requested_progress[index] = projected.progress;
  }

  std::size_t available = std::numeric_limits<std::size_t>::max();
  if (eval_budget.max_evals != 0) {
    const std::size_t reserved =
        eval_budget.reserved.load(std::memory_order_relaxed);
    available = reserved >= eval_budget.max_evals
                    ? 0
                    : eval_budget.max_evals - reserved;
  }
  const std::size_t granted = std::min(available, seeds.size());
  stats.budget_granted_directions = granted;
  if (eval_budget.max_evals != 0)
    eval_budget.reserved.fetch_add(granted, std::memory_order_relaxed);

  const std::size_t evaluations_before =
      total_pathway_evaluations_used(workers);
  executor.parallel_for(
      seeds.size(), requested_workers,
      pathway_parallel_grain(seeds.size(), requested_workers,
                             lg.m_mol[0].size()),
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &worker = workers[worker_id];
        for (std::size_t index = begin; index < end; ++index) {
          candidates[index] = evaluate_candidate(
              path, state, requested_progress[index], start_com, rec, lg,
              worker, energy_ref, Arguments, DE_arguments, index < granted,
              &targets[index], 0.0, 0.0, current_frame.radius_angstrom, true);
          candidates[index].guided_seed = true;
          candidates[index].guided_seed_index = seeds[index].index;
        }
      });

  stats.evaluated_directions =
      total_pathway_evaluations_used(workers) - evaluations_before;
  stats.stopped_by_eval_budget = granted < seeds.size();
  std::size_t best_index = 0;
  bool found = false;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (!candidates[index].valid)
      continue;
    ++stats.valid_directions;
    if (!found || pathway_candidate_better(
                      candidates[index], index, candidates[best_index],
                      best_index, state.mode)) {
      best_index = index;
      found = true;
    }
  }
  if (found)
    best_candidate = candidates[best_index];
  return stats;
}

PathwayPatternStats refine_pathway_candidate_pattern(
    PathwayCandidateState &best_candidate, const PathwayProgressDomain &domain,
    const PathwayBatchEvaluator &evaluator, double initial_step,
    PathwayMode mode) {
  PathwayPatternStats stats;
  stats.enabled = true;
  stats.initial_score = best_candidate.raw_score;
  const auto started_at = std::chrono::steady_clock::now();
  double step = std::max(initial_step, kMinPatternStep);

  while (stats.sweeps < kPathPatternMaxSweeps && step >= kMinPatternStep) {
    std::array<double, 2> proposals{
        domain.sanitize(best_candidate.progress + step),
        domain.sanitize(best_candidate.progress - step)};
    std::array<PathwayCandidateState, 2> trials{};
    const PullDeBatchEvaluation eval_info = evaluator(
        std::span<const double>(proposals), std::span<PathwayCandidateState>(trials));
    stats.evaluations_used += eval_info.evaluations_used;
    if (eval_info.budget_exhausted)
      stats.stopped_by_eval_budget = true;
    ++stats.sweeps;

    std::size_t best_trial = trials.size();
    for (std::size_t i = 0; i < trials.size(); ++i) {
      if (!trials[i].valid)
        continue;
      if (pathway_candidate_improves(trials[i], best_candidate, mode)) {
        best_trial = i;
        break;
      }
    }

    if (best_trial < trials.size()) {
      best_candidate = trials[best_trial];
      stats.improved = true;
      ++stats.accepted_moves;
      continue;
    }

    if (stats.stopped_by_eval_budget)
      break;
    step *= 0.5;
  }

  stats.final_score = best_candidate.raw_score;
  stats.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return stats;
}

PathwayPatternStats refine_pathway_variant_multi_seed_pattern(
    PathwayCandidateState &best_candidate,
    const std::vector<PathwayCandidateState> &candidate_pool,
    const PathwayProgressDomain &domain, const PathwayBatchEvaluator &evaluator,
    const PullDeVariantStats &variant_stats, double total_length,
    PathwayMode mode) {
  PathwayPatternStats aggregate;
  aggregate.enabled = true;
  aggregate.initial_score = best_candidate.raw_score;
  const auto started_at = std::chrono::steady_clock::now();
  const double seed_gap =
      std::max(0.1, total_length * kMinProgressSeedGapNorm);
  const auto seed_indices =
      select_pathway_pattern_seed_indices(candidate_pool, 3, seed_gap, mode);
  aggregate.seeds_attempted = seed_indices.empty() ? 1 : seed_indices.size();
  aggregate.distinct_seed_basins = aggregate.seeds_attempted;
  const double initial_step = pathway_variant_pattern_initial_step(
      variant_stats, std::max(0.1, domain.max_progress - domain.min_progress));

  if (seed_indices.empty()) {
    PathwayCandidateState local_best = best_candidate;
    PathwayPatternStats local_stats = refine_pathway_candidate_pattern(
        local_best, domain, evaluator, initial_step, mode);
    aggregate.sweeps = local_stats.sweeps;
    aggregate.accepted_moves = local_stats.accepted_moves;
    aggregate.evaluations_used = local_stats.evaluations_used;
    aggregate.stopped_by_eval_budget = local_stats.stopped_by_eval_budget;
    if (pathway_candidate_improves(local_best, best_candidate, mode)) {
      best_candidate = local_best;
      aggregate.improved = true;
    }
  } else {
    for (const std::size_t seed_index : seed_indices) {
      PathwayCandidateState local_best = candidate_pool[seed_index];
      PathwayPatternStats local_stats = refine_pathway_candidate_pattern(
          local_best, domain, evaluator, initial_step, mode);
      aggregate.sweeps += local_stats.sweeps;
      aggregate.accepted_moves += local_stats.accepted_moves;
      aggregate.evaluations_used += local_stats.evaluations_used;
      if (local_stats.improved &&
          pathway_candidate_improves(local_best, best_candidate, mode)) {
        best_candidate = local_best;
        aggregate.improved = true;
      }
      if (local_stats.stopped_by_eval_budget) {
        aggregate.stopped_by_eval_budget = true;
        break;
      }
    }
  }

  aggregate.final_score = best_candidate.raw_score;
  aggregate.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return aggregate;
}

std::size_t effective_pathway_direct_eval_budget(const de_arg &DE_arguments) {
  if (DE_arguments.pull_max_evals != 0)
    return DE_arguments.pull_max_evals;
  return std::max<std::size_t>(DE_arguments.pop_num * DE_arguments.ite_num,
                               1024);
}

double pathway_direct_local_threshold_size(const PathwayDirectDomain &domain) {
  const double span = std::max(1.0e-6, domain.total_valid_length);
  return std::max(kPathwayDirectMinCellNorm, 0.08 / span);
}

PathwayDirectDomain build_pathway_direct_domain(
    const std::vector<Interval> &intervals) {
  PathwayDirectDomain domain;
  domain.intervals = intervals;
  domain.prefix_lengths.reserve(intervals.size());
  double prefix = 0.0;
  for (const auto &interval : domain.intervals) {
    domain.prefix_lengths.push_back(prefix);
    prefix += std::max(0.0, interval.end - interval.begin);
  }
  domain.total_valid_length = prefix;
  return domain;
}

double pathway_direct_normalized_to_progress(const PathwayDirectDomain &domain,
                                             double normalized) {
  if (domain.intervals.empty())
    return 0.0;
  normalized = std::clamp(normalized, 0.0, 1.0);
  if (domain.total_valid_length <= kPathTol)
    return domain.intervals.front().begin;
  double cursor = normalized * domain.total_valid_length;
  for (std::size_t i = 0; i < domain.intervals.size(); ++i) {
    const double length =
        std::max(0.0, domain.intervals[i].end - domain.intervals[i].begin);
    if (cursor <= length + kPathTol || i + 1 == domain.intervals.size())
      return domain.intervals[i].begin + std::min(cursor, length);
    cursor -= length;
  }
  return domain.intervals.back().end;
}

double pathway_direct_cell_size_metric(const PathwayDirectCell &cell) {
  return cell.half_width_norm;
}

bool pathway_direct_cell_preferred(const PathwayDirectCell &lhs,
                                   const PathwayDirectCell &rhs,
                                   PathwayMode mode) {
  const double lhs_size = pathway_direct_cell_size_metric(lhs);
  const double rhs_size = pathway_direct_cell_size_metric(rhs);
  if (lhs.state.valid != rhs.state.valid)
    return lhs.state.valid;
  if (lhs.state.valid && rhs.state.valid &&
      !pathway_candidate_equivalent(lhs.state, rhs.state, mode)) {
    return pathway_candidate_better(lhs.state, lhs.id, rhs.state, rhs.id, mode);
  }
  if (std::fabs(lhs_size - rhs_size) > 1.0e-12)
    return lhs_size > rhs_size;
  if (lhs.depth != rhs.depth)
    return lhs.depth < rhs.depth;
  return lhs.id < rhs.id;
}

std::size_t select_pathway_direct_cell_index(
    const std::vector<PathwayDirectCell> &frontier, PathwayMode mode) {
  double largest_size = 0.0;
  for (const auto &cell : frontier)
    largest_size = std::max(largest_size, pathway_direct_cell_size_metric(cell));
  const double size_cutoff = largest_size * 0.5;

  std::size_t chosen = frontier.size();
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    if (pathway_direct_cell_size_metric(frontier[i]) + 1.0e-12 < size_cutoff)
      continue;
    if (chosen == frontier.size() ||
        pathway_direct_cell_preferred(frontier[i], frontier[chosen], mode)) {
      chosen = i;
    }
  }
  if (chosen != frontier.size())
    return chosen;

  chosen = 0;
  for (std::size_t i = 1; i < frontier.size(); ++i) {
    if (pathway_direct_cell_preferred(frontier[i], frontier[chosen], mode))
      chosen = i;
  }
  return chosen;
}

bool pathway_direct_cell_dominates(const PathwayDirectCell &lhs,
                                   const PathwayDirectCell &rhs,
                                   PathwayMode mode) {
  const double lhs_size = pathway_direct_cell_size_metric(lhs);
  const double rhs_size = pathway_direct_cell_size_metric(rhs);
  const bool no_worse_size = lhs_size + 1.0e-12 >= rhs_size;
  const bool lhs_better =
      pathway_candidate_better(lhs.state, lhs.id, rhs.state, rhs.id, mode);
  const bool rhs_better =
      pathway_candidate_better(rhs.state, rhs.id, lhs.state, lhs.id, mode);
  const bool no_worse_score = lhs_better || !rhs_better;
  const bool strictly_better = lhs_size > rhs_size + 1.0e-12 || lhs_better;
  return no_worse_size && no_worse_score && strictly_better;
}

std::vector<std::size_t> select_pathway_direct_cell_batch_indices(
    const std::vector<PathwayDirectCell> &frontier, std::size_t max_cells,
    PathwayMode mode) {
  std::vector<std::size_t> pareto;
  pareto.reserve(frontier.size());
  for (std::size_t i = 0; i < frontier.size(); ++i) {
    bool dominated = false;
    for (std::size_t j = 0; j < frontier.size(); ++j) {
      if (i == j)
        continue;
      if (pathway_direct_cell_dominates(frontier[j], frontier[i], mode)) {
        dominated = true;
        break;
      }
    }
    if (!dominated)
      pareto.push_back(i);
  }
  if (pareto.empty())
    return {select_pathway_direct_cell_index(frontier, mode)};

  std::sort(pareto.begin(), pareto.end(), [&](std::size_t lhs, std::size_t rhs) {
    if (pathway_candidate_strict_less(frontier[lhs].state, 0, frontier[rhs].state, 0, mode))
      return true;
    if (pathway_candidate_strict_less(frontier[rhs].state, 0, frontier[lhs].state, 0, mode))
      return false;
    const double lhs_size = pathway_direct_cell_size_metric(frontier[lhs]);
    const double rhs_size = pathway_direct_cell_size_metric(frontier[rhs]);
    if (lhs_size != rhs_size) return lhs_size > rhs_size;
    return frontier[lhs].id < frontier[rhs].id;
  });

  if (pareto.size() > max_cells)
    pareto.resize(max_cells);
  std::sort(pareto.begin(), pareto.end());
  return pareto;
}

PathwayDirectStats run_pathway_direct_search(
    PathwayCandidateState &best_candidate,
    std::vector<PathwayCandidateState> &ranked_candidates, std::size_t &best_stage,
    bool &global_budget_stop, const PathwayDirectDomain &direct_domain,
    const PathwayBatchEvaluator &evaluator, std::size_t direct_phase_budget,
    PathwayMode mode, TitanExecutor &executor, std::size_t requested_workers,
    std::size_t ligand_atom_count) {
  PathwayDirectStats stats;
  stats.enabled = true;
  stats.effective_eval_budget = direct_phase_budget;
  const auto started_at = std::chrono::steady_clock::now();
  const double local_threshold = pathway_direct_local_threshold_size(direct_domain);

  PathwayDirectCell root;
  std::array<double, 1> root_progress{
      pathway_direct_normalized_to_progress(direct_domain, 0.5)};
  std::array<PathwayCandidateState, 1> root_eval{};
  PullDeBatchEvaluation eval_info =
      evaluator(std::span<const double>(root_progress),
                std::span<PathwayCandidateState>(root_eval));
  root.center_norm = 0.5;
  root.state = root_eval[0];
  std::size_t phase_evals = eval_info.evaluations_used;
  if (eval_info.budget_exhausted) {
    stats.stopped_by_eval_budget = true;
    global_budget_stop = true;
  }

  std::vector<PathwayDirectCell> frontier;
  frontier.push_back(root);
  if (root.state.valid) {
    best_candidate = root.state;
    ranked_candidates.push_back(root.state);
    stats.best_cell_size = pathway_direct_cell_size_metric(root);
  }

  std::size_t next_cell_id = 1;
  const std::size_t max_batch_capacity = std::max<std::size_t>(1, requested_workers * 2);
  std::size_t expansions_since_best = 0;

  struct SplitPlan {
    std::size_t frontier_index{0};
    PathwayDirectCell middle{};
    std::array<PathwayDirectCell, 2> children{};
  };
  std::vector<SplitPlan> plans(max_batch_capacity);
  std::vector<double> child_norm(max_batch_capacity * 2, 0.0);
  std::vector<PathwayCandidateState> child_states(max_batch_capacity * 2);

  while (!frontier.empty()) {
    if (direct_phase_budget != 0 && phase_evals >= direct_phase_budget) {
      stats.stopped_by_eval_budget = true;
      global_budget_stop = true;
      break;
    }

    const std::size_t chosen_preview =
        select_pathway_direct_cell_index(frontier, mode);
    const double preview_size =
        pathway_direct_cell_size_metric(frontier[chosen_preview]);
    if (preview_size <= local_threshold) {
      stats.stopped_by_local_threshold = true;
      break;
    }
    if (expansions_since_best >= 32) {
      stats.stopped_by_stagnation = true;
      break;
    }

    const std::size_t remaining_evals =
        direct_phase_budget > phase_evals ? direct_phase_budget - phase_evals : 0;
    std::size_t batch_cells =
        std::min(max_batch_capacity,
                 std::min(frontier.size(),
                          std::max<std::size_t>(1, remaining_evals / 2)));
    if (batch_cells == 0)
      break;

    const auto selected_indices =
        select_pathway_direct_cell_batch_indices(frontier, batch_cells, mode);
    if (selected_indices.empty())
      break;

    std::size_t active_plans = 0;
    for (const std::size_t frontier_index : selected_indices) {
      PathwayDirectCell chosen = frontier[frontier_index];
      const double split_half = chosen.half_width_norm / 3.0;
      if (split_half < kPathwayDirectMinCellNorm)
        continue;

      SplitPlan plan;
      plan.frontier_index = frontier_index;
      plan.middle = chosen;
      plan.middle.depth = chosen.depth + 1;
      plan.middle.half_width_norm = split_half;
      plan.middle.id = next_cell_id++;
      for (std::size_t child = 0; child < 2; ++child) {
        plan.children[child] = plan.middle;
        plan.children[child].center_norm =
            chosen.center_norm + (child == 0 ? -split_half : split_half);
        plan.children[child].center_norm =
            std::clamp(plan.children[child].center_norm, 0.0, 1.0);
        plan.children[child].id = next_cell_id++;
        child_norm[active_plans * 2 + child] = pathway_direct_normalized_to_progress(
            direct_domain, plan.children[child].center_norm);
      }
      plans[active_plans++] = plan;
    }
    if (active_plans == 0)
      break;

    stats.max_batch_cells = std::max(stats.max_batch_cells, active_plans);
    eval_info = evaluator(
        std::span<const double>(child_norm.data(), active_plans * 2),
        std::span<PathwayCandidateState>(child_states.data(), active_plans * 2));
    phase_evals += eval_info.evaluations_used;
    if (eval_info.budget_exhausted) {
      stats.stopped_by_eval_budget = true;
      global_budget_stop = true;
    }

    for (std::size_t plan_index = 0; plan_index < active_plans; ++plan_index) {
      frontier[plans[plan_index].frontier_index] = plans[plan_index].middle;
      for (std::size_t child_index = 0; child_index < 2; ++child_index) {
        PathwayDirectCell child = plans[plan_index].children[child_index];
        child.state = child_states[plan_index * 2 + child_index];
        frontier.push_back(child);
        if (!child.state.valid)
          continue;
        ranked_candidates.push_back(child.state);
        if (!best_candidate.valid ||
            pathway_candidate_better(child.state, child.id, best_candidate,
                                     best_stage, mode)) {
          best_candidate = child.state;
          best_stage = stats.expansions + plan_index + 1;
          stats.best_expansion = best_stage;
          stats.best_cell_size = pathway_direct_cell_size_metric(child);
          stats.best_half_width_norm = child.half_width_norm;
          expansions_since_best = 0;
        }
      }
    }

    expansions_since_best += active_plans;
    stats.expansions += active_plans;
    ++stats.batches;
    if (stats.stopped_by_eval_budget)
      break;
  }

  stats.cells_remaining = frontier.size();
  stats.evaluated_candidates = ranked_candidates.size();
  stats.phase_evaluations_used = phase_evals;
  if (best_candidate.valid && stats.best_expansion == 0)
    stats.best_expansion = best_stage;
  stats.wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  return stats;
}

void write_pathway_identity(std::ostream &output, const PathContext &path,
                            const std::string &prefix = "") {
  if (path.explicit_route) {
    output << prefix << "route id: " << path.route_id << "\n";
    output << prefix << "route source: " << path.route_source << "\n";
    output << prefix << "reference sha256: "
           << path.reference_structure_sha256 << "\n";
    output << prefix << "route file sha256: " << path.route_file_sha256
           << "\n";
    output << prefix << "registration: " << path.registration_mode << "\n";
    output << prefix << "registration RMSD: " << path.registration_rmsd
           << " A\n";
  } else {
    output << prefix << "pathway group: " << path.group_name << " ["
           << path.group_index << "]\n";
  }
}

void write_pathway_status_file(
    const info &output_info, const std::string &status,
    const std::string &reason, const PathContext &path,
    const PathwayState &state, bool used_old_vector,
    const de_arg &DE_arguments, const arguments &Run_arguments,
    const Point3 &reference_com, const Point3 &ligand_com,
    const PathwayCandidateState *candidate = nullptr,
    const PathwayFinalSelection *selection = nullptr,
    std::size_t candidate_evaluations = 0,
    std::size_t energy_evaluations = 0,
    std::size_t clearance_pair_checks = 0,
    std::size_t clearance_rejections = 0,
    std::size_t candidate_proposals = 0,
    std::size_t scan_shell_exclusions = 0,
    std::size_t probe_evaluations = 0,
    std::size_t probe_clearance_rejections = 0,
    std::size_t probe_mode_rejections = 0,
    std::size_t swept_probe_evaluations = 0,
    std::size_t swept_pair_checks = 0,
    std::size_t swept_rejections = 0,
    std::size_t swept_interior_only_rejections = 0,
    titan_path::CandidateExhaustion exhaustion =
        titan_path::CandidateExhaustion::None,
    std::string_view reason_detail_override = {},
    const PathwayGuidedSeedStats *guided_stats = nullptr) {
  std::fstream status_file(output_info.get_pull_direction_status_filename(),
                           std::ios::out);
  titan_path::configure_status_numeric_output(status_file);
  status_file << "SCHEMA=TITAN_PATH_STATUS_V1\n";
  status_file << "STATUS=" << status << "\n";
  status_file << "REASON=" << reason << "\n";
  status_file << "REASON_DETAIL="
              << (reason_detail_override.empty()
                      ? titan_path::candidate_exhaustion_name(exhaustion)
                      : reason_detail_override)
              << "\n";
  status_file << "MODE=" << pathway_mode_label(state.mode) << "\n";
  status_file << "RECOVERY_DIRECT_PROBE_USED="
              << (state.recovery_direct_probe_used ? 1 : 0) << "\n";
  status_file << "PATH_PROGRESS_A=" << state.current_projection.progress
              << "\n";
  status_file << "PATH_TOTAL_A=" << path.total_length << "\n";
  status_file << "PROJECTION_CONTINUITY="
              << (state.previous_progress.has_value() ? "BOUNDED_PREVIOUS_PROGRESS" : "GLOBAL_INITIAL") << "\n";
  if (state.previous_progress.has_value()) {
    status_file << "PREVIOUS_PATH_PROGRESS_A=" << *state.previous_progress << "\n";
    status_file << "PROJECTION_WINDOW_MIN_A="
                << std::max(0.0, *state.previous_progress-state.continuity_half_window) << "\n";
    status_file << "PROJECTION_WINDOW_MAX_A="
                << std::min(path.total_length, *state.previous_progress+state.continuity_half_window) << "\n";
  }
  status_file << "PATH_DISTANCE=" << state.current_projection.distance << "\n";
  status_file << "PATH_DISTANCE_A=" << state.current_projection.distance
              << "\n";
  status_file << "MIN_RECEPTOR_LIGAND_DISTANCE_A="
              << state.minimum_receptor_ligand_distance << "\n";
  status_file << "PHYSICAL_EXIT_GATE_ENABLED="
              << (state.physical_exit_gate_enabled ? 1 : 0) << "\n";
  status_file << "PHYSICAL_EXIT_PASSED="
              << (state.physical_exit_passed ? 1 : 0) << "\n";
  status_file << "USED_OLD_VECTOR=" << (used_old_vector ? 1 : 0) << "\n";
  status_file << "MAX_TURN_ANGLE_DEG="
              << (state.mode == PathwayMode::Recovery &&
                          Run_arguments.de.path_probe_length > 0.0
                      ? Run_arguments.de.path_recovery_max_turn_angle
                      : Run_arguments.de.path_max_turn_angle)
              << "\n";
  status_file << "ADVANCE_MAX_TURN_ANGLE_DEG="
              << Run_arguments.de.path_max_turn_angle << "\n";
  status_file << "RECOVERY_MAX_TURN_ANGLE_DEG="
              << Run_arguments.de.path_recovery_max_turn_angle << "\n";
  status_file << "ENERGY_KEY_EPSILON="
              << Run_arguments.de.path_energy_key_epsilon << "\n";
  status_file << "GEOMETRY_ONLY="
              << (Run_arguments.de.path_geometry_only ? 1 : 0) << "\n";
  status_file << "MIN_CLEARANCE_THRESHOLD_A="
              << Run_arguments.de.path_min_clearance << "\n";
  status_file << "FIXED_PROBE_ENABLED="
              << (Run_arguments.de.path_probe_length > 0.0 ? 1 : 0) << "\n";
  status_file << "CONFIGURED_PROBE_LENGTH_A="
              << Run_arguments.de.path_probe_length << "\n";
  status_file << "CONFIGURED_PROBE_LENGTHS_A=";
  for (std::size_t index = 0;
       index < Run_arguments.de.path_probe_lengths.size(); ++index) {
    if (index != 0)
      status_file << ',';
    status_file << Run_arguments.de.path_probe_lengths[index];
  }
  status_file << "\n";
  status_file << "CANDIDATE_PROPOSALS=" << candidate_proposals << "\n";
  status_file << "CANDIDATE_EVALUATIONS=" << candidate_evaluations << "\n";
  status_file << "PROBE_EVALUATIONS=" << probe_evaluations << "\n";
  status_file << "SCAN_SHELL_EXCLUSIONS=" << scan_shell_exclusions << "\n";
  status_file << "ENERGY_EVALUATIONS=" << energy_evaluations << "\n";
  status_file << "CLEARANCE_PAIR_CHECKS=" << clearance_pair_checks << "\n";
  status_file << "CLEARANCE_REJECTIONS=" << clearance_rejections << "\n";
  status_file << "PROBE_CLEARANCE_REJECTIONS="
              << probe_clearance_rejections << "\n";
  status_file << "PROBE_MODE_REJECTIONS=" << probe_mode_rejections << "\n";
  status_file << "CLEARANCE_MODE="
              << Run_arguments.de.path_clearance_mode << "\n";
  if (Run_arguments.de.path_clearance_mode == "swept-linear-center") {
    status_file << "CLEARANCE_SCOPE=CONTINUOUS_LINEAR_TRANSLATION\n";
    status_file << "CLEARANCE_COORDINATE_FRAME="
                   "UPSTREAM_PBC_RECONSTRUCTED_CARTESIAN\n";
    status_file << "CLEARANCE_PERIODIC_DISTANCE_POLICY="
                   "DIRECT_IN_RECONSTRUCTED_FRAME\n";
    status_file << "CLEARANCE_PBC_PREPROCESSOR="
                << swept_clearance_pbc_preprocessor(Run_arguments) << "\n";
    status_file << "SWEPT_CLEARANCE_PROBE_EVALUATIONS="
                << swept_probe_evaluations << "\n";
    status_file << "SWEPT_CLEARANCE_PAIR_CHECKS=" << swept_pair_checks
                << "\n";
    status_file << "SWEPT_CLEARANCE_REJECTIONS=" << swept_rejections
                << "\n";
    status_file << "SWEPT_INTERIOR_ONLY_REJECTIONS="
                << swept_interior_only_rejections << "\n";
  }
  if (Run_arguments.de.path_navigation_enabled) {
    status_file << "NAVIGATION_ENABLED=1\n";
    status_file << "NAVIGATION_COST_KIND=ALGORITHMIC_DIMENSIONLESS\n";
    status_file << "NAVIGATION_SELECTION_SCOPE=ADVANCE_ONLY\n";
    status_file << "NAVIGATION_DIRECTIONALITY=GUIDED_SEEDS_PLUS_PROGRESS_SELECTION\n";
    status_file << "NAVIGATION_TUBE_WEIGHT="
                << Run_arguments.de.path_navigation_tube_weight << "\n";
    status_file << "NAVIGATION_BACKTRACK_WEIGHT="
                << Run_arguments.de.path_navigation_backtrack_weight << "\n";
    status_file << "NAVIGATION_COST_EPSILON="
                << Run_arguments.de.path_navigation_cost_epsilon << "\n";
    status_file << "NAVIGATION_SEED_RADIAL_SAMPLES="
                << Run_arguments.de.path_navigation_seed_radial_samples
                << "\n";
    status_file << "NAVIGATION_SEED_AMPLITUDE="
                << Run_arguments.de.path_navigation_seed_amplitude << "\n";
    status_file << "NAVIGATION_RECOVERY_TANGENT_WEIGHT="
                << titan_path::kNavigationRecoveryTangentWeight << "\n";
    status_file << "NAVIGATION_OLD_DIRECTION_BLEND_FRACTIONS=";
    for (std::size_t index = 0;
         index < titan_path::kNavigationOldDirectionBlendFractions.size();
         ++index) {
      if (index != 0)
        status_file << ',';
      status_file << titan_path::kNavigationOldDirectionBlendFractions[index];
    }
    status_file << "\n";
    status_file << "GUIDED_SEED_PROPOSALS="
                << (guided_stats == nullptr
                        ? 0
                        : guided_stats->proposed_directions)
                << "\n";
    status_file << "GUIDED_SEED_EVALUATIONS="
                << (guided_stats == nullptr
                        ? 0
                        : guided_stats->evaluated_directions)
                << "\n";
    status_file << "GUIDED_SEED_BUDGET_GRANTED="
                << (guided_stats == nullptr
                        ? 0
                        : guided_stats->budget_granted_directions)
                << "\n";
    status_file << "GUIDED_SEED_VALID="
                << (guided_stats == nullptr ? 0
                                            : guided_stats->valid_directions)
                << "\n";
    if (guided_stats != nullptr && guided_stats->enabled) {
      status_file << "NAVIGATION_GUIDE_X="
                  << guided_stats->guide_direction[0] << "\n";
      status_file << "NAVIGATION_GUIDE_Y="
                  << guided_stats->guide_direction[1] << "\n";
      status_file << "NAVIGATION_GUIDE_Z="
                  << guided_stats->guide_direction[2] << "\n";
    }
  }
  status_file << "REFERENCE_COM_X_A=" << reference_com[0] << "\n";
  status_file << "REFERENCE_COM_Y_A=" << reference_com[1] << "\n";
  status_file << "REFERENCE_COM_Z_A=" << reference_com[2] << "\n";
  status_file << "LIGAND_COM_X_A=" << ligand_com[0] << "\n";
  status_file << "LIGAND_COM_Y_A=" << ligand_com[1] << "\n";
  status_file << "LIGAND_COM_Z_A=" << ligand_com[2] << "\n";
  status_file << "ROUTE_KIND="
              << (path.explicit_route ? "EXPLICIT" : "LEGACY_GROUP") << "\n";
  if (path.explicit_route) {
    status_file << "ROUTE_ID=" << path.route_id << "\n";
    status_file << "ROUTE_SOURCE=" << path.route_source << "\n";
    status_file << "REFERENCE_SHA256=" << path.reference_structure_sha256
                << "\n";
    status_file << "ROUTE_FILE_SHA256=" << path.route_file_sha256 << "\n";
    status_file << "REGISTRATION_MODE=" << path.registration_mode << "\n";
    status_file << "REGISTRATION_RMSD_A=" << path.registration_rmsd << "\n";
  } else {
    status_file << "PATHWAY_GROUP=" << path.group_name << "\n";
    status_file << "PATHWAY_GROUP_INDEX=" << path.group_index << "\n";
  }
  if (candidate != nullptr && candidate->valid) {
    status_file << "REQUESTED_TARGET_PROGRESS_A=" << candidate->progress
                << "\n";
    status_file << "REQUESTED_DELTA_PROGRESS_A="
                << candidate->requested_delta_progress << "\n";
    // Keep the V1 TARGET field's historical requested-target meaning.  The
    // applied endpoint is reported by the additive APPLIED_* fields below.
    status_file << "TARGET_PROGRESS_A=" << candidate->progress
                << "\n";
    status_file << "APPLIED_PROGRESS_A=" << candidate->applied_progress
                << "\n";
    status_file << "APPLIED_DELTA_PROGRESS_A=" << candidate->delta_progress
                << "\n";
    status_file << "APPLIED_PATH_DISTANCE_A="
                << candidate->applied_path_distance << "\n";
    status_file << "REQUESTED_PULL_DISTANCE_A="
                << candidate->requested_pull_distance << "\n";
    status_file << "ROUTE_TARGET_VECTOR_X_A="
                << candidate->requested_vector.Fx << "\n";
    status_file << "ROUTE_TARGET_VECTOR_Y_A="
                << candidate->requested_vector.Fy << "\n";
    status_file << "ROUTE_TARGET_VECTOR_Z_A="
                << candidate->requested_vector.Fz << "\n";
    status_file << "ROUTE_TARGET_ANGLE_DEG="
                << candidate->requested_angle_deg << "\n";
    status_file << "TURN_CAPPED=" << (candidate->turn_capped ? 1 : 0) << "\n";
    titan_path::write_next_vector_status(
        status_file,
        {candidate->vector.Fx, candidate->vector.Fy, candidate->vector.Fz},
        candidate->pull_distance, candidate->angle_deg);
    status_file << "CORRIDOR_OFFSET_U_A="
                << candidate->corridor_offset_u << "\n";
    status_file << "CORRIDOR_OFFSET_V_A="
                << candidate->corridor_offset_v << "\n";
    status_file << "CORRIDOR_RADIUS_A=" << candidate->corridor_radius << "\n";
    status_file << "TARGET_MIN_CLEARANCE_A="
                << candidate->minimum_clearance << "\n";
    if (Run_arguments.de.path_clearance_mode == "swept-linear-center") {
      status_file << "TARGET_ENDPOINT_MIN_CLEARANCE_A="
                  << candidate->endpoint_minimum_clearance << "\n";
      status_file << "TARGET_SWEPT_MIN_CLEARANCE_A="
                  << candidate->swept_minimum_clearance << "\n";
      status_file << "TARGET_SWEPT_MIN_FRACTION="
                  << candidate->swept_minimum_fraction << "\n";
      status_file << "TARGET_SWEPT_MIN_RECEPTOR_INDEX="
                  << candidate->swept_minimum_receptor_index << "\n";
      status_file << "TARGET_SWEPT_MIN_LIGAND_INDEX="
                  << candidate->swept_minimum_ligand_index << "\n";
      status_file << "TARGET_SWEPT_MIN_COMPLETE="
                  << (candidate->swept_minimum_complete ? 1 : 0) << "\n";
    }
    status_file << "TARGET_CLEARANCE_VECTOR=NEXT_VECTOR\n";
    status_file << "TARGET_RAW_SCORE=" << candidate->raw_score << "\n";
    status_file << "TARGET_ENERGY_KEY=" << candidate->energy_key << "\n";
    if (candidate->navigation_enabled) {
      status_file << "TARGET_GUIDED_SEED="
                  << (candidate->guided_seed ? 1 : 0) << "\n";
      status_file << "TARGET_GUIDED_SEED_INDEX="
                  << candidate->guided_seed_index << "\n";
      status_file << "TARGET_APPLIED_CORRIDOR_RADIUS_A="
                  << candidate->applied_corridor_radius << "\n";
      status_file << "TARGET_NAVIGATION_TUBE_EXCESS_A="
                  << candidate->navigation_tube_excess << "\n";
      status_file << "TARGET_NAVIGATION_BACKTRACK_A="
                  << candidate->navigation_backtrack_distance << "\n";
      status_file << "TARGET_NAVIGATION_TUBE_COST="
                  << candidate->navigation_tube_cost << "\n";
      status_file << "TARGET_NAVIGATION_BACKTRACK_COST="
                  << candidate->navigation_backtrack_cost << "\n";
      status_file << "TARGET_NAVIGATION_COST="
                  << candidate->navigation_cost << "\n";
    }
  }
  if (selection != nullptr) {
    status_file << "ARCHIVE_MIN_ENERGY_KEY="
                << selection->minimum_energy_key << "\n";
    status_file << "VALID_ARCHIVE_SIZE="
                << selection->valid_archive_size << "\n";
    status_file << "FORWARD_ARCHIVE_SIZE="
                << selection->forward_archive_size << "\n";
    status_file << "FORWARD_PROGRESS_REJECTIONS="
                << selection->forward_progress_rejections << "\n";
    status_file << "EPSILON_ELIGIBLE_SIZE="
                << selection->epsilon_eligible_size << "\n";
    status_file << "MODE_ELIGIBLE_SIZE="
                << selection->mode_eligible_size << "\n";
    status_file << "MAXIMUM_PROBE_TIER_SIZE="
                << selection->maximum_probe_tier_size << "\n";
    status_file << "ENERGY_ELIGIBLE_SIZE="
                << selection->epsilon_eligible_size << "\n";
    if (Run_arguments.de.path_navigation_enabled) {
      status_file << "NAVIGATION_ELIGIBLE_SIZE="
                  << selection->navigation_eligible_size << "\n";
      if (std::isfinite(selection->minimum_navigation_cost))
        status_file << "ENERGY_ELIGIBLE_MIN_NAVIGATION_COST="
                    << selection->minimum_navigation_cost << "\n";
    }
  }
  if (DE_arguments.snapshot_metadata_valid)
    status_file << "SNAPSHOT_TIME_PS=" << DE_arguments.snapshot_time_ps << "\n";
}

} // namespace

int run_pathway_pull_direction_legacy_unused(
    Multi_MOL &rec, Multi_MOL &lg, cache_sigma_epsilon &CachedData,
    de_arg &DE_arguments, arguments &Arguments, info &output_info,
    TitanExecutor &executor, groups_cal &Groups_from_index_file) {
#if 0
  if (rec.m_mol[0].mindist_mol(lg.m_mol[0], Arguments, DE_arguments) == true) {
    TitanUI::print_info("Automatic termination is on!");
    return 1;
  }

  const auto built = build_path(rec, lg, Groups_from_index_file, Arguments);
  if (!built.ok) {
    TitanUI::print_error("Pull-direction pathway: " + built.error);
    return 1;
  }
  if (built.path.pruned_component_nodes > 0) {
    TitanUI::print_warning(
        "Pathway auto-order ignored " +
        std::to_string(built.path.pruned_component_nodes) +
        " residue node(s) outside the component nearest g2.");
  }
  if (built.path.pruned_branch_nodes > 0) {
    TitanUI::print_warning(
        "Pathway auto-order pruned " +
        std::to_string(built.path.pruned_branch_nodes) +
        " branch residue node(s) outside the principal path.");
  }

  const Point3 start_com = mol_com(lg.m_mol[0], lg.parameters);
  const cache_energy energy_ref =
      Arguments.de.path_geometry_only
          ? cache_energy{}
          : reference_energy(rec, lg, CachedData, Arguments, executor);
  const std::size_t worker_count = std::max<std::size_t>(1, Arguments.nt);

  std::vector<double> progress(DE_arguments.pop_num, 0.0);
  std::vector<CandidateResult> candidates(DE_arguments.pop_num);
  {
    std::random_device rd;
    std::mt19937 mt(rd());
    for (std::size_t i = 0; i < progress.size(); ++i)
      progress[i] = sample_interval(mt, built.path.valid_intervals);
  }

  executor.parallel_for(
      progress.size(), worker_count, 8,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        (void)worker_id;
        cache_sigma_epsilon eval_cache = make_eval_cache(CachedData);
        MOL scratch{lg.m_mol[0]};
        for (std::size_t i = begin; i < end; ++i)
          candidates[i] = evaluate_candidate(built.path, progress[i], start_com,
                                             rec, lg, eval_cache, energy_ref,
                                             Arguments, scratch);
      });

  double prev_best = std::numeric_limits<double>::max();
  int stagnation = 0;
  for (std::size_t iter = 0; iter < DE_arguments.ite_num; ++iter) {
    TitanUI::show_progress("Pulling Direction", iter, DE_arguments.ite_num,
                           "(Pathway Evolution)");
    const std::vector<double> old_progress = progress;
    executor.parallel_for(
        progress.size(), worker_count, 8,
        [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
          cache_sigma_epsilon eval_cache = make_eval_cache(CachedData);
          MOL scratch{lg.m_mol[0]};
          std::random_device rd;
          std::mt19937 mt(static_cast<unsigned>(
              rd() ^ (0x9e3779b9U + worker_id * 1315423911U +
                      static_cast<unsigned>(iter * 2654435761U))));
          std::uniform_int_distribution<std::size_t> choose(0, progress.size() - 1);
          std::uniform_real_distribution<double> prob(0.0, 1.0);
          for (std::size_t i = begin; i < end; ++i) {
            std::size_t a = choose(mt), b = choose(mt), c = choose(mt);
            while (a == b || a == c || b == c || a == i || b == i || c == i) {
              a = choose(mt);
              b = choose(mt);
              c = choose(mt);
            }
            double trial = old_progress[i];
            if (DE_arguments.algorithm == 0) {
              if (prob(mt) < DE_arguments.cross_prob || i == choose(mt))
                trial = old_progress[a] +
                        DE_arguments.diff_weight *
                            (old_progress[b] - old_progress[c]);
            } else if (prob(mt) < DE_arguments.cross_prob) {
              trial = old_progress[a] +
                      DE_arguments.F * (old_progress[b] - old_progress[c]);
            } else {
              trial = old_progress[a] +
                      0.5 * (DE_arguments.F + 1.0) *
                          (old_progress[b] + old_progress[c] -
                           2.0 * old_progress[a]);
            }
            trial = std::clamp(trial, 0.0, built.path.total_length);
            trial = clamp_to_intervals(trial, built.path.valid_intervals);
            const auto candidate = evaluate_candidate(
                built.path, trial, start_com, rec, lg, eval_cache, energy_ref,
                Arguments, scratch);
            if (candidate.total_score < candidates[i].total_score) {
              progress[i] = trial;
              candidates[i] = candidate;
            }
          }
        });

    double iter_best = std::numeric_limits<double>::max();
    for (const auto &candidate : candidates)
      iter_best = std::min(iter_best, candidate.total_score);
    if (iter_best < prev_best - 1.0e-8) {
      prev_best = iter_best;
      stagnation = 0;
    } else {
      ++stagnation;
    }
    if (stagnation >= DE_arguments.stagnation_limit)
      break;
  }

  std::size_t best_index = 0;
  for (std::size_t i = 1; i < candidates.size(); ++i) {
    if (candidates[i].total_score < candidates[best_index].total_score)
      best_index = i;
  }
  const Point3 best_com = add_pt(
      start_com,
      {candidates[best_index].vector.Fx, candidates[best_index].vector.Fy,
       candidates[best_index].vector.Fz});
  const double best_angle =
      angle_to_previous_vector(candidates[best_index].vector,
                               DE_arguments.old_vector);

  std::vector<std::pair<double, std::size_t>> ranked(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i)
    ranked[i] = {candidates[i].total_score, i};
  std::sort(ranked.begin(), ranked.end());

  std::fstream de_output(output_info.get_DE_output_filename(), std::ios::out);
  std::fstream de_log(output_info.get_DE_log_filename(),
                      std::ios::out | std::ios::app);
  std::fstream de_report("DE_report.dat", std::ios::out);
  de_report << std::fixed << std::setprecision(4);
  de_report << "TITAN Pathway Pull Report\n\n";
  de_report << "g1 reference: " << Arguments.group_1 << "\n";
  de_report << "g2 pulled: " << Arguments.group_2 << "\n";
  de_report << "pathway group: " << built.path.group_name << " ["
            << built.path.group_index << "]\n";
  de_report << "pathway rescue radius: " << Arguments.de.dis_along_path
            << " A\n";
  de_report << "raw nodes: " << built.path.all_nodes.size() << "\n";
  de_report << "principal nodes: " << built.path.principal_nodes.size() << "\n";
  de_report << "pruned disconnected: " << built.path.pruned_component_nodes
            << "\n";
  de_report << "pruned branches: " << built.path.pruned_branch_nodes << "\n";
  de_report << "used geometric edges: "
            << (built.path.used_geometric_edges ? "YES" : "NO") << "\n";
  if (built.path.pruned_component_nodes > 0)
    de_report << "warning: disconnected pathway nodes were ignored because they "
                 "are outside the start component near g2.\n";
  if (built.path.pruned_branch_nodes > 0)
    de_report << "warning: branch pathway nodes were pruned to keep a single "
                 "principal path.\n";
  de_report << "start residue: " << residue_label(built.path.principal_nodes.front())
            << "\n";
  de_report << "terminal residue: "
            << residue_label(built.path.principal_nodes.back()) << "\n";
  de_report << "arc length: " << built.path.total_length << " A\n";
  de_report << "best score: " << candidates[best_index].total_score << "\n";
  de_report << "best raw score: " << candidates[best_index].raw_score << "\n";
  de_report << "best progress: " << candidates[best_index].progress << " A\n";
  de_report << "pull vector: (" << candidates[best_index].vector.Fx << ", "
            << candidates[best_index].vector.Fy << ", "
            << candidates[best_index].vector.Fz << ")\n";
  de_report << "pull distance: " << candidates[best_index].vector.modul()
            << " A\n";
  de_report << "angle to prev vector: " << best_angle << " deg\n\n";
  de_report << "Top candidates\n";
  de_report << std::setw(8) << "Rank" << std::setw(8) << "Index"
            << std::setw(14) << "Score" << std::setw(14) << "RawScore"
            << std::setw(14) << "Progress" << std::setw(12) << "Pull(A)"
            << std::setw(14) << "Elec" << std::setw(14) << "VdW" << "\n";
  for (std::size_t rank = 0; rank < std::min<std::size_t>(50, ranked.size());
       ++rank) {
    const auto idx = ranked[rank].second;
    de_report << std::setw(8) << rank + 1 << std::setw(8) << idx
              << std::setw(14) << candidates[idx].total_score << std::setw(14)
              << candidates[idx].raw_score << std::setw(14)
              << candidates[idx].progress << std::setw(12)
              << candidates[idx].vector.modul() << std::setw(14)
              << candidates[idx].energy.elec_ener << std::setw(14)
              << candidates[idx].energy.vdw_ener << "\n";
  }

  de_output << std::fixed << std::setprecision(4);
  de_output << candidates[best_index].vector.Fx << "  "
            << candidates[best_index].vector.Fy << "  "
            << candidates[best_index].vector.Fz << "  Angle: " << best_angle
            << " -- Modul: " << candidates[best_index].vector.modul()
            << " Angstrom\n";
  de_output << "# Pull g2 toward the best reachable point on the auto-ordered "
               "pathway of g1\n";
  de_output << "# Start residue: " << residue_label(built.path.principal_nodes.front())
            << "\n";
  de_output << "# Terminal residue: "
            << residue_label(built.path.principal_nodes.back()) << "\n";
  de_output << "# Lig COM: (" << start_com[0] << ", " << start_com[1] << ", "
            << start_com[2] << ") -> (" << best_com[0] << ", " << best_com[1]
            << ", " << best_com[2] << ")\n";
  de_log << std::fixed << std::setprecision(4);
  de_log << candidates[best_index].vector.Fx << "  "
         << candidates[best_index].vector.Fy << "  "
         << candidates[best_index].vector.Fz << "  Angle: " << best_angle
         << " -- Modul: " << candidates[best_index].vector.modul()
         << " Angstrom  Score: " << candidates[best_index].total_score
         << "  Progress: " << candidates[best_index].progress
         << "\n--------------------------\n";

  std::fstream best_conf("best_conf.pdb", std::ios::out);
  MOL best_candidate{lg.m_mol[0]};
  best_candidate.translate(candidates[best_index].vector.Fx,
                           candidates[best_index].vector.Fy,
                           candidates[best_index].vector.Fz);
  best_conf.precision(3);
  for (std::size_t i = 0; i < best_candidate.size(); ++i) {
    best_conf << "ATOM" << std::setw(7) << i + 1 << std::setw(5)
              << lg.info[i].get_atom() << std::setw(4)
              << lg.info[i].get_original_res_name() << " A" << std::setw(4) << 1
              << std::setw(12) << std::fixed << best_candidate.pos.x[i]
              << std::setw(8) << best_candidate.pos.y[i] << std::setw(8)
              << best_candidate.pos.z[i] << std::setw(7) << 1.00
              << std::setw(6) << 0.00 << '\n';
  }

  std::fstream vector_pdb(output_info.get_DE_pdb_vector_filename(),
                          std::ios::out | std::ios::app);
  vector_pdb.precision(3);
  vector_pdb << "ATOM" << std::setw(7) << 1 << std::setw(5) << "CA"
             << std::setw(4) << "ALA"
             << " A" << std::setw(4) << 1 << std::setw(12) << std::fixed
             << start_com[0] << std::setw(8) << start_com[1] << std::setw(8)
             << start_com[2] << " " << std::setw(7) << 1.00 << " "
             << std::setw(6) << 0.00 << '\n';
  vector_pdb << "ATOM" << std::setw(7) << 2 << std::setw(5) << "CB"
             << std::setw(4) << "ALA"
             << " A" << std::setw(4) << 1 << std::setw(12) << std::fixed
             << best_com[0] << std::setw(8) << best_com[1] << std::setw(8)
             << best_com[2] << " " << std::setw(7) << 1.00 << " "
             << std::setw(6) << 0.00 << '\n';
  vector_pdb << "TER\n";
  return 0;
#endif
  (void)rec;
  (void)lg;
  (void)CachedData;
  (void)DE_arguments;
  (void)Arguments;
  (void)output_info;
  (void)executor;
  (void)Groups_from_index_file;
  return 1;
}

int run_pathway_pull_direction(Multi_MOL &rec, Multi_MOL &lg,
                               cache_sigma_epsilon &CachedData,
                               de_arg &DE_arguments, arguments &Arguments,
                               info &output_info, TitanExecutor &executor,
                               groups_cal &Groups_from_index_file) {
  // Fixed/tangent controls request validated Cartesian inputs, not a native
  // route candidate. In particular, an empty scan shell must not prevent their
  // independent supervisor from evaluating the prescribed control direction.
  if (const char *only = std::getenv("TITAN_PATH_CARTESIAN_ONLY"); only != nullptr) {
    const char *audit = std::getenv("TITAN_PATH_GEOMETRY_AUDIT");
    if (std::string_view(only) != "1" || audit == nullptr || *audit == '\0') {
      TitanUI::print_error("Cartesian-only export requires explicit audit output");
      return 1;
    }
    try {
      titan_path::write_cartesian_audit(audit, rec, lg);
    } catch (const std::exception &error) {
      TitanUI::print_error(std::string("Pull-direction Cartesian audit: ") + error.what());
      return 1;
    }
    return 0; // Observation export only; never authorizes MD or a direction.
  }
  const auto built = build_path(rec, lg, Groups_from_index_file, Arguments);
  if (!built.ok) {
    TitanUI::print_error("Pull-direction pathway: " + built.error);
    return 1;
  }
  if (built.path.pruned_component_nodes > 0) {
    TitanUI::print_warning(
        "Pathway auto-order ignored " +
        std::to_string(built.path.pruned_component_nodes) +
        " residue node(s) outside the component nearest g2.");
  }
  if (built.path.pruned_branch_nodes > 0) {
    TitanUI::print_warning(
        "Pathway auto-order pruned " +
        std::to_string(built.path.pruned_branch_nodes) +
        " branch residue node(s) outside the principal path.");
  }

  const Point3 start_com = mol_com(lg.m_mol[0], lg.parameters);
  const Point3 reference_com = mol_com(rec.m_mol[0], rec.parameters);
  if (const char *audit = std::getenv("TITAN_PATH_GEOMETRY_AUDIT");
      audit != nullptr && *audit != '\0') {
    try {
      titan_path::write_cartesian_audit(audit, rec, lg);
    } catch (const std::exception &error) {
      TitanUI::print_error(std::string("Pull-direction Cartesian audit: ") + error.what());
      return 1;
    }
  }
  auto previous_mode = read_previous_pathway_mode(
      output_info.get_pull_direction_status_filename());
  std::optional<double> previous_progress;
  PathwayState pathway_state;
  try {
    const char *observed = std::getenv("TITAN_PATH_OBSERVED_BOUNDARY");
    if (observed != nullptr && *observed != '\0') {
      if (!Arguments.de.path_previous_status.empty() || !DE_arguments.snapshot_metadata_valid)
        throw std::invalid_argument("observed boundary conflicts with legacy state or lacks snapshot time");
      std::ifstream boundary_file(observed);
      if (!boundary_file) throw std::invalid_argument("observed boundary file unavailable");
      const auto boundary = titan_path::read_observed_boundary(
          boundary_file, built.path.route_id, built.path.route_file_sha256,
          built.path.reference_structure_sha256, built.path.total_length,
          DE_arguments.snapshot_time_ps);
      previous_progress = boundary.progress_angstrom;
      previous_mode = boundary.mode == titan_path::RouteMode::Recovery
                          ? PathwayMode::Recovery : PathwayMode::Advance;
    }
    if (!Arguments.de.path_previous_status.empty()) {
      if (!DE_arguments.snapshot_metadata_valid)
        throw std::invalid_argument("continuity requires authoritative snapshot time");
      std::ifstream previous_file(Arguments.de.path_previous_status);
      if (!previous_file) throw std::invalid_argument("previous route status file unavailable");
      const auto previous = titan_path::read_previous_route_state(
          previous_file, built.path.route_id, built.path.route_file_sha256,
          built.path.reference_structure_sha256, built.path.total_length,
          DE_arguments.snapshot_time_ps);
      previous_progress = previous.progress_angstrom;
      previous_mode = previous.mode == titan_path::RouteMode::Recovery
                          ? PathwayMode::Recovery : PathwayMode::Advance;
    }
    pathway_state = prepare_pathway_state(built.path, start_com, Arguments,
                                         previous_mode, previous_progress,
                                         observed != nullptr && *observed != '\0');
    if (observed != nullptr && *observed != '\0' &&
        std::abs(pathway_state.current_projection.progress-*previous_progress) > .03)
      throw std::invalid_argument("native and observed boundary progress disagree");
  } catch (const std::exception &error) {
    TitanUI::print_error(std::string("Pull-direction continuity: ") + error.what());
    return 1;
  }
  pathway_state.minimum_receptor_ligand_distance =
      evaluate_interatomic_clearance(rec.m_mol[0], lg.m_mol[0])
          .minimum_distance;
  pathway_state.physical_exit_gate_enabled =
      Arguments.de.path_completion_min_separation > 0.0;
  pathway_state.physical_exit_passed =
      !pathway_state.physical_exit_gate_enabled ||
      pathway_state.minimum_receptor_ligand_distance >=
          Arguments.de.path_completion_min_separation;
  const bool near_route_terminal =
      pathway_state.current_projection.progress >=
      std::max(0.0, built.path.total_length -
                        Arguments.de.path_completion_tolerance);
  const bool route_complete =
      near_route_terminal && pathway_state.physical_exit_passed;
  if (route_complete) {
    write_pathway_status_file(output_info, "COMPLETE", "ROUTE_TERMINAL_REACHED",
                              built.path, pathway_state, false, DE_arguments,
                              Arguments, reference_com, start_com);
    std::fstream de_report("DE_report.dat", std::ios::out);
    de_report << std::fixed << std::setprecision(4);
    de_report << "TITAN Pathway Pull Report\n\n";
    de_report << "status: COMPLETE\n";
    de_report << "reason: ROUTE_TERMINAL_REACHED\n";
    write_pathway_identity(de_report, built.path);
    de_report << "current progress: "
              << pathway_state.current_projection.progress << " A\n";
    de_report << "route arc length: " << built.path.total_length << " A\n";
    de_report << "terminal tolerance: "
              << Arguments.de.path_completion_tolerance << " A\n";
    de_report << "minimum receptor-ligand distance: "
              << pathway_state.minimum_receptor_ligand_distance << " A\n";
    de_report << "physical exit threshold: "
              << Arguments.de.path_completion_min_separation << " A\n";
    TitanUI::print_info("Pathway route terminal reached.");
    return 0;
  }

  de_arg effective_DE_arguments = DE_arguments;
  const bool recovery_dropped_old_vector = false;

  const bool use_recovery_direct_probe =
      pathway_state.mode == PathwayMode::Recovery &&
      pathway_state.forward_intervals.empty() &&
      Arguments.de.path_probe_length > 0.0;
  std::string stop_reason;
  if (pathway_state.forward_intervals.empty() &&
      !use_recovery_direct_probe) {
    stop_reason = near_route_terminal && !pathway_state.physical_exit_passed
                      ? "TERMINAL_REACHED_WITHOUT_PHYSICAL_EXIT"
                      : "NO_REACHABLE_FORWARD_TARGET";
  }

  const auto started_at = std::chrono::steady_clock::now();
  const cache_energy energy_ref =
      Arguments.de.path_geometry_only
          ? cache_energy{}
          : reference_energy(rec, lg, CachedData, Arguments, executor);
  const std::size_t requested_workers = std::max<std::size_t>(1, Arguments.nt);
  const std::size_t worker_slots =
      std::max<std::size_t>(1, std::min(executor.capacity(), requested_workers));
  std::vector<PathwayWorkerContext> workers(worker_slots);
  for (auto &worker : workers) {
    worker.scratch = lg.m_mol[0];
    worker.eval_cache = make_eval_cache(CachedData);
  }

  PathwayEvalBudget eval_budget;

  const PullOptimizerKind optimizer_kind =
      classify_pull_optimizer(DE_arguments.pull_opt);
  const bool use_direct_family =
      optimizer_is_direct_family(DE_arguments.pull_opt);
  const bool use_corridor_grid =
      optimizer_kind == PullOptimizerKind::CorridorGrid;
  const bool use_variant_de =
      (optimizer_kind == PullOptimizerKind::DeShade ||
       optimizer_kind == PullOptimizerKind::DeLShade ||
       optimizer_kind == PullOptimizerKind::DeJso ||
       optimizer_kind == PullOptimizerKind::DeJsoPattern ||
       optimizer_kind == PullOptimizerKind::DeNlShadeLbc ||
       optimizer_kind == PullOptimizerKind::DeNlShadeLbcPattern);
  const bool use_variant_pattern =
      (optimizer_kind == PullOptimizerKind::DeJsoPattern ||
       optimizer_kind == PullOptimizerKind::DeNlShadeLbcPattern);
  const bool use_de_pattern = (optimizer_kind == PullOptimizerKind::DePattern);
  const bool use_direct_pattern =
      (optimizer_kind == PullOptimizerKind::DirectPattern);
  const std::size_t variant_total_budget =
      use_variant_de
          ? (DE_arguments.pull_max_evals != 0
                 ? DE_arguments.pull_max_evals
                 : DE_arguments.pop_num * (DE_arguments.ite_num + 1))
          : DE_arguments.pull_max_evals;
  const std::size_t direct_total_budget =
      use_direct_family ? effective_pathway_direct_eval_budget(DE_arguments) : 0;
  std::size_t variant_phase_budget = variant_total_budget;
  std::size_t direct_phase_budget = direct_total_budget;
  std::size_t pattern_reserved_budget = 0;
  if (use_variant_pattern && variant_total_budget != 0) {
    pattern_reserved_budget = clamp_pattern_tail_budget(variant_total_budget);
    if (variant_total_budget > pattern_reserved_budget)
      variant_phase_budget = variant_total_budget - pattern_reserved_budget;
  }
  if (use_direct_pattern && direct_total_budget != 0) {
    pattern_reserved_budget = clamp_pattern_tail_budget(direct_total_budget);
    if (direct_total_budget > pattern_reserved_budget)
      direct_phase_budget = direct_total_budget - pattern_reserved_budget;
  }
  eval_budget.max_evals = use_corridor_grid
                              ? DE_arguments.pull_max_evals
                          : use_variant_de
                              ? variant_phase_budget
                              : use_direct_family ? direct_phase_budget
                                                  : DE_arguments.pull_max_evals;

  auto sanitize_progress = [&](double progress) {
    progress = std::clamp(progress, pathway_state.forward_intervals.front().begin,
                          pathway_state.forward_intervals.back().end);
    return clamp_to_intervals(progress, pathway_state.forward_intervals);
  };
  const PathwayProgressDomain domain{
      pathway_state.forward_intervals.empty()
          ? 0.0
          : pathway_state.forward_intervals.front().begin,
      pathway_state.forward_intervals.empty()
          ? 0.0
          : pathway_state.forward_intervals.back().end,
      [&](std::mt19937 &mt) {
        return sample_interval(mt, pathway_state.forward_intervals);
      },
      sanitize_progress};
  const PathwayDirectDomain direct_domain =
      build_pathway_direct_domain(pathway_state.forward_intervals);

  std::vector<PathwayCandidateState> evaluation_archive;
  const auto batch_evaluator =
      [&](std::span<const double> proposals,
          std::span<PathwayCandidateState> evaluated_candidates) {
        std::vector<double> sanitized(proposals.size(), 0.0);
        for (std::size_t i = 0; i < proposals.size(); ++i)
          sanitized[i] = domain.sanitize(proposals[i]);
        const PullDeBatchEvaluation batch = evaluate_candidate_batch(
            std::span<const double>(sanitized), evaluated_candidates, built.path,
            pathway_state, start_com, rec, lg, workers, energy_ref, Arguments,
            effective_DE_arguments, eval_budget, executor, requested_workers);
        for (const auto &candidate : evaluated_candidates)
          if (candidate.valid)
            evaluation_archive.push_back(candidate);
        return batch;
      };

  std::vector<PathwayCandidateState> report_candidates;
  PathwayCandidateState best_candidate;
  PathwayFinalSelection final_selection;
  titan_path::FinalSelectionFailure final_selection_failure =
      titan_path::FinalSelectionFailure::None;
  bool final_selection_attempted = false;
  bool recovery_descent_exhausted = false;
  PullDeVariantStats de_variant_stats;
  PathwayLegacyDeStats legacy_stats;
  PathwayDirectStats direct_stats;
  PathwayCorridorGridStats corridor_grid_stats;
  PathwayGuidedSeedStats guided_seed_stats;
  PathwayPatternStats pattern_stats;
  std::vector<PathwayCandidateState> guided_candidates;
  PathwayCandidateState guided_best_candidate;
  std::size_t best_stage = 0;
  bool stopped_by_eval_budget = false;

  if (stop_reason.empty() && Arguments.de.path_navigation_enabled &&
      !use_recovery_direct_probe) {
    try {
      guided_seed_stats = run_pathway_guided_seeds(
          built.path, pathway_state, start_com, rec, lg, workers, energy_ref,
          Arguments, effective_DE_arguments, eval_budget, executor,
          requested_workers, guided_candidates, guided_best_candidate);
    } catch (const std::exception &error) {
      TitanUI::print_error("TITAN-Path guided seed generation failed: " +
                           std::string(error.what()));
      stop_reason = "NAVIGATION_GUIDE_DEGENERATE";
    }
    for (const auto &candidate : guided_candidates)
      if (candidate.valid)
        evaluation_archive.push_back(candidate);
  }

  if (stop_reason.empty()) {
    bool ok = false;
    if (use_recovery_direct_probe) {
      pathway_state.recovery_direct_probe_used = true;
      const Point3 recovery_target = pathway_state.current_projection.point;
      best_candidate = evaluate_candidate(
          built.path, pathway_state, pathway_state.current_projection.progress,
          start_com, rec, lg, workers.front(), energy_ref, Arguments,
          effective_DE_arguments, true, &recovery_target, 0.0, 0.0, 0.0,
          true);
      if (best_candidate.valid &&
          best_candidate.applied_path_distance + 1.0e-6 <
              pathway_state.current_projection.distance) {
        report_candidates.push_back(best_candidate);
        evaluation_archive.push_back(best_candidate);
        ok = true;
      } else if (best_candidate.valid) {
        report_candidates.push_back(best_candidate);
        stop_reason = "NO_RECOVERY_DESCENT_DIRECTION";
        recovery_descent_exhausted = true;
      }
    } else if (use_corridor_grid) {
      try {
        corridor_grid_stats = run_pathway_corridor_grid(
            built.path, pathway_state, start_com, rec, lg, workers, energy_ref,
            Arguments, effective_DE_arguments, eval_budget, executor,
            requested_workers, report_candidates, best_candidate);
      } catch (const std::exception &error) {
        TitanUI::print_error("TITAN-Path corridor evaluation failed: " +
                             std::string(error.what()));
        stop_reason = "CORRIDOR_GEOMETRY_DEGENERATE";
      }
      for (const auto &candidate : report_candidates)
        if (candidate.valid)
          evaluation_archive.push_back(candidate);
      ok = stop_reason.empty() && best_candidate.valid;
      stopped_by_eval_budget = corridor_grid_stats.stopped_by_eval_budget;
    } else if (use_variant_de) {
      PullDeVariantOptions variant_options;
      variant_options.variant = de_variant_from_optimizer(optimizer_kind);
      variant_options.seed = DE_arguments.pull_seed;
      variant_options.initial_population = DE_arguments.pop_num;
      variant_options.minimum_population = DE_arguments.min_pop_num;
      variant_options.max_generations = DE_arguments.ite_num;
      variant_options.planned_evaluations = variant_total_budget;
      variant_options.memory_size = DE_arguments.memory_size;
      variant_options.p_best_rate = DE_arguments.p_best_rate;
      variant_options.archive_scale = DE_arguments.archive_scale;
      variant_options.stagnation_limit = DE_arguments.stagnation_limit;
      ok = run_pathway_de_variant(variant_options, domain, pathway_state.mode,
                                  batch_evaluator, report_candidates,
                                  best_candidate, de_variant_stats);
      if (use_variant_pattern && ok) {
        eval_budget.max_evals = variant_total_budget;
        pattern_stats = refine_pathway_variant_multi_seed_pattern(
            best_candidate, report_candidates, domain, batch_evaluator,
            de_variant_stats, built.path.total_length, pathway_state.mode);
        if (pattern_stats.improved)
          report_candidates.push_back(best_candidate);
      }
      stopped_by_eval_budget = de_variant_stats.stopped_by_eval_budget ||
                               pattern_stats.stopped_by_eval_budget;
      best_stage = de_variant_stats.best_generation;
    } else if (use_direct_family) {
      direct_stats = run_pathway_direct_search(
          best_candidate, report_candidates, best_stage, stopped_by_eval_budget,
          direct_domain, batch_evaluator, direct_phase_budget, pathway_state.mode,
          executor, requested_workers, lg.m_mol[0].size());
      direct_stats.used_fallback_budget = (DE_arguments.pull_max_evals == 0);
      ok = best_candidate.valid;
      if (use_direct_pattern && ok) {
        eval_budget.max_evals = direct_total_budget;
        const double initial_step = std::max(
            0.1, 0.25 * std::max(0.1, domain.max_progress - domain.min_progress));
        pattern_stats = refine_pathway_candidate_pattern(
            best_candidate, domain, batch_evaluator, initial_step,
            pathway_state.mode);
        pattern_stats.seeds_attempted = 1;
        pattern_stats.distinct_seed_basins = 1;
        if (pattern_stats.improved)
          report_candidates.push_back(best_candidate);
      }
      stopped_by_eval_budget = direct_stats.stopped_by_eval_budget ||
                               pattern_stats.stopped_by_eval_budget;
    } else {
      ok = run_pathway_legacy_de(DE_arguments, domain, pathway_state.mode,
                                 batch_evaluator, report_candidates,
                                 best_candidate, legacy_stats);
      if (use_de_pattern && ok) {
        const double initial_step = std::max(
            0.1, 0.25 * std::max(0.1, domain.max_progress - domain.min_progress));
        pattern_stats = refine_pathway_candidate_pattern(
            best_candidate, domain, batch_evaluator, initial_step,
            pathway_state.mode);
        pattern_stats.seeds_attempted = 1;
        pattern_stats.distinct_seed_basins = 1;
        if (pattern_stats.improved)
          report_candidates.push_back(best_candidate);
      }
      stopped_by_eval_budget = legacy_stats.stopped_by_eval_budget ||
                               pattern_stats.stopped_by_eval_budget;
      best_stage = legacy_stats.best_generation;
    }
    for (const auto &candidate : guided_candidates)
      if (candidate.valid)
        report_candidates.push_back(candidate);
    if (!best_candidate.valid && guided_best_candidate.valid) {
      best_candidate = guided_best_candidate;
      ok = true;
    }
    stopped_by_eval_budget =
        stopped_by_eval_budget || guided_seed_stats.stopped_by_eval_budget;
    if (stop_reason.empty() && (!ok || !best_candidate.valid))
      stop_reason = use_recovery_direct_probe
                        ? "NO_VALID_RECOVERY_DIRECTION"
                        : "NO_VALID_FORWARD_DIRECTION";
  }

  if (stop_reason.empty() && best_candidate.valid) {
    final_selection_attempted = true;
    final_selection = select_pathway_final_candidate(
        evaluation_archive, pathway_state.mode,
        Arguments.de.path_energy_key_epsilon,
        Arguments.de.path_geometry_only,
        pathway_state.current_projection.distance,
        Arguments.de.path_probe_length > 0.0,
        Arguments.de.path_navigation_enabled,
        Arguments.de.path_navigation_cost_epsilon);
    if (final_selection.candidate.valid) {
      best_candidate = final_selection.candidate;
      report_candidates.push_back(best_candidate);
    } else {
      titan_path::SelectionResult selection_counts;
      selection_counts.found = final_selection.candidate.valid;
      selection_counts.valid_count = final_selection.valid_archive_size;
      selection_counts.selectable_count = final_selection.forward_archive_size;
      selection_counts.nonforward_count =
          final_selection.forward_progress_rejections;
      selection_counts.eligible_count = final_selection.epsilon_eligible_size;
      final_selection_failure = titan_path::classify_final_selection_failure(
          selection_counts,
          pathway_state.mode == PathwayMode::Recovery
              ? titan_path::RouteMode::Recovery
              : titan_path::RouteMode::Advance);
      if (final_selection_failure ==
          titan_path::FinalSelectionFailure::ForwardProgressExhausted) {
        stop_reason = "NO_FORWARD_PROGRESS_CANDIDATE";
      } else if (final_selection_failure ==
                 titan_path::FinalSelectionFailure::RecoveryDescentExhausted) {
        stop_reason = "NO_RECOVERY_DESCENT_DIRECTION";
        recovery_descent_exhausted = true;
      } else {
        stop_reason = pathway_state.mode == PathwayMode::Recovery
                          ? "NO_ENERGETICALLY_SELECTABLE_RECOVERY_TARGET"
                          : "NO_ENERGETICALLY_SELECTABLE_FORWARD_TARGET";
      }
    }
  }

  const double wall_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    started_at)
          .count();
  const std::size_t candidate_proposals =
      total_pathway_candidate_proposals(workers);
  const std::size_t total_evaluations = total_pathway_evaluations_used(workers);
  const std::size_t scan_shell_exclusions =
      total_pathway_scan_shell_exclusions(workers);
  const std::size_t energy_evaluations =
      total_pathway_energy_evaluations(workers);
  const std::size_t clearance_pair_checks =
      total_pathway_clearance_pair_checks(workers);
  const std::size_t clearance_rejections =
      total_pathway_clearance_rejections(workers);
  const std::size_t probe_evaluations =
      total_pathway_probe_evaluations(workers);
  const std::size_t probe_clearance_rejections =
      total_pathway_probe_clearance_rejections(workers);
  const std::size_t probe_mode_rejections =
      total_pathway_probe_mode_rejections(workers);
  const std::size_t swept_probe_evaluations =
      total_pathway_swept_probe_evaluations(workers);
  const std::size_t swept_pair_checks =
      total_pathway_swept_pair_checks(workers);
  const std::size_t swept_rejections =
      total_pathway_swept_rejections(workers);
  const std::size_t swept_interior_only_rejections =
      total_pathway_swept_interior_only_rejections(workers);
  const double evals_per_second =
      wall_seconds > 0.0 ? static_cast<double>(total_evaluations) / wall_seconds
                         : 0.0;
  const auto exhaustion =
      (stop_reason == "NO_VALID_FORWARD_DIRECTION" ||
       stop_reason == "NO_VALID_RECOVERY_DIRECTION")
          ? titan_path::classify_candidate_exhaustion(
                candidate_proposals, total_evaluations, scan_shell_exclusions,
                clearance_rejections, energy_evaluations)
          : titan_path::CandidateExhaustion::None;
  const std::string stop_reason_detail =
      recovery_descent_exhausted
          ? "RECOVERY_DESCENT_EXHAUSTED"
      : final_selection_failure == titan_path::FinalSelectionFailure::None
          ? titan_path::candidate_exhaustion_name(exhaustion)
          : titan_path::final_selection_failure_name(final_selection_failure);

  if (!stop_reason.empty()) {
    write_pathway_status_file(output_info, "STOP", stop_reason, built.path,
                              pathway_state,
                               effective_DE_arguments.flag_old_vec,
                               DE_arguments, Arguments, reference_com, start_com,
                               recovery_descent_exhausted ? &best_candidate
                                                          : nullptr,
                               final_selection_attempted ? &final_selection
                                                         : nullptr,
                               total_evaluations,
                               energy_evaluations, clearance_pair_checks,
                               clearance_rejections, candidate_proposals,
                               scan_shell_exclusions,
                               probe_evaluations,
                               probe_clearance_rejections,
                               probe_mode_rejections,
                                swept_probe_evaluations,
                                swept_pair_checks,
                                swept_rejections,
                                swept_interior_only_rejections,
                                exhaustion, stop_reason_detail,
                               &guided_seed_stats);
    std::fstream de_log(output_info.get_DE_log_filename(),
                        std::ios::out | std::ios::app);
    std::fstream de_report("DE_report.dat", std::ios::out);
    de_report << std::fixed << std::setprecision(4);
    de_report << "TITAN Pathway Pull Report\n\n";
    de_report << "status: STOP\n";
    de_report << "reason: " << stop_reason << "\n";
    de_report << "reason detail: " << stop_reason_detail << "\n";
    de_report << "optimizer: " << DE_arguments.pull_opt << "\n";
    write_pathway_identity(de_report, built.path);
    de_report << "graph edge cutoff: "
              << Arguments.de.path_graph_edge_cutoff << " A\n";
    de_report << "recovery enter/exit: "
              << Arguments.de.path_recovery_enter_radius << " / "
              << Arguments.de.path_recovery_exit_radius << " A\n";
    de_report << "current mode: " << pathway_mode_label(pathway_state.mode)
              << "\n";
    de_report << "current progress: " << pathway_state.current_projection.progress
              << " A\n";
    de_report << "current path distance: "
              << pathway_state.current_projection.distance << " A\n";
    de_report << "terminal residue: "
              << residue_label(built.path.principal_nodes.back()) << "\n";
    de_report << "candidate proposals: " << candidate_proposals << "\n";
    de_report << "total evaluations: " << total_evaluations << "\n";
    de_report << "scan-shell exclusions: " << scan_shell_exclusions << "\n";
    de_report << "energy evaluations: " << energy_evaluations << "\n";
    de_report << "clearance pair checks: " << clearance_pair_checks << "\n";
    de_report << "clearance rejections: " << clearance_rejections << "\n";
    de_report << "probe evaluations: " << probe_evaluations << "\n";
    de_report << "probe clearance rejections: "
              << probe_clearance_rejections << "\n";
    de_report << "probe mode rejections: " << probe_mode_rejections << "\n";
    if (Arguments.de.path_clearance_mode == "swept-linear-center") {
      de_report << "swept probe evaluations: " << swept_probe_evaluations
                << "\n";
      de_report << "swept pair checks: " << swept_pair_checks << "\n";
      de_report << "swept rejections: " << swept_rejections << "\n";
      de_report << "swept interior-only rejections: "
                << swept_interior_only_rejections << "\n";
    }
    if (final_selection_attempted) {
      de_report << "valid archive size: "
                << final_selection.valid_archive_size << "\n";
      de_report << "forward archive size: "
                << final_selection.forward_archive_size << "\n";
      de_report << "forward progress rejections: "
                << final_selection.forward_progress_rejections << "\n";
      de_report << "epsilon eligible size: "
                << final_selection.epsilon_eligible_size << "\n";
    }
    de_report << "wall time: " << wall_seconds << " s\n";
    de_report << "eval/s: " << evals_per_second << "\n";
    de_log << "STOP  Reason: " << stop_reason << "  Detail: "
           << stop_reason_detail
           << "  PathDist: "
           << pathway_state.current_projection.distance
           << "  Progress: " << pathway_state.current_projection.progress
           << "\n--------------------------\n";
    TitanUI::print_info("Pathway pull-direction stopped: " + stop_reason);
    return 0;
  }

  const Point3 best_com =
      add_pt(start_com, {best_candidate.vector.Fx, best_candidate.vector.Fy,
                         best_candidate.vector.Fz});
  const double best_angle = best_candidate.angle_deg;
  const auto report_rows =
      build_grouped_report_rows(report_candidates, pathway_state.mode);

  write_pathway_status_file(output_info, "CONTINUE", "OK", built.path,
                            pathway_state,
                             effective_DE_arguments.flag_old_vec,
                             DE_arguments, Arguments, reference_com, start_com,
                             &best_candidate, &final_selection,
                             total_evaluations, energy_evaluations,
                             clearance_pair_checks, clearance_rejections,
                             candidate_proposals, scan_shell_exclusions,
                             probe_evaluations, probe_clearance_rejections,
                             probe_mode_rejections,
                              swept_probe_evaluations, swept_pair_checks,
                              swept_rejections,
                              swept_interior_only_rejections,
                              titan_path::CandidateExhaustion::None, {},
                             &guided_seed_stats);

  std::fstream de_output(output_info.get_DE_output_filename(), std::ios::out);
  std::fstream de_log(output_info.get_DE_log_filename(),
                      std::ios::out | std::ios::app);
  std::fstream de_report("DE_report.dat", std::ios::out);
  std::fstream best_conf("best_conf.pdb", std::ios::out);
  std::fstream vector_pdb(output_info.get_DE_pdb_vector_filename(),
                          std::ios::out | std::ios::app);

  de_report << std::fixed << std::setprecision(4);
  de_report << "==============================================================="
               "=================\n";
  de_report << "  TITAN Pathway Pull Report\n";
  de_report << "==============================================================="
               "=================\n\n";
  de_report << "  Parameters\n";
  de_report << "  ----------\n";
  de_report << "  Optimizer          : " << DE_arguments.pull_opt << "\n";
  de_report << "  Optimizer family   : "
            << (use_corridor_grid ? "Corridor grid"
                                  : use_direct_family ? "Direct" : "DE")
            << "\n";
  de_report << "  Variant            : "
            << (use_variant_de
                    ? std::string(pull_de_variant_name(de_variant_stats.variant))
                    : DE_arguments.pull_opt)
            << "\n";
  write_pathway_identity(de_report, built.path, "  ");
  de_report << "  Graph edge cutoff  : "
            << Arguments.de.path_graph_edge_cutoff << " A\n";
  de_report << "  Recovery enter/exit: "
            << Arguments.de.path_recovery_enter_radius << " / "
            << Arguments.de.path_recovery_exit_radius << " A\n";
  de_report << "  Corridor spacing   : "
            << Arguments.de.path_corridor_spacing << " A\n";
  de_report << "  Corridor radial N  : "
            << Arguments.de.path_corridor_radial_samples << "\n";
  de_report << "  Energy-key epsilon : "
            << Arguments.de.path_energy_key_epsilon << "\n";
  de_report << "  Min clearance      : "
            << Arguments.de.path_min_clearance << " A\n";
  de_report << "  Clearance mode     : "
            << Arguments.de.path_clearance_mode << "\n";
  de_report << "  Fixed probe length : "
            << Arguments.de.path_probe_length << " A (0=legacy)\n";
  de_report << "  Probe ladder       : ";
  for (std::size_t index = 0;
       index < Arguments.de.path_probe_lengths.size(); ++index) {
    if (index != 0)
      de_report << ',';
    de_report << Arguments.de.path_probe_lengths[index];
  }
  de_report << " A\n";
  de_report << "  Navigation enabled : "
            << (Arguments.de.path_navigation_enabled ? "YES" : "NO")
            << "\n";
  if (Arguments.de.path_navigation_enabled) {
    de_report << "  Navigation kind    : algorithmic dimensionless cost"
              << " (not physical energy)\n";
    de_report << "  Nav tube/backtrack : "
              << Arguments.de.path_navigation_tube_weight << " / "
              << Arguments.de.path_navigation_backtrack_weight << "\n";
    de_report << "  Nav cost epsilon   : "
              << Arguments.de.path_navigation_cost_epsilon << "\n";
    de_report << "  Nav seed ring N/a  : "
              << Arguments.de.path_navigation_seed_radial_samples << " / "
              << Arguments.de.path_navigation_seed_amplitude << "\n";
    de_report << "  Nav recovery tangent: "
              << titan_path::kNavigationRecoveryTangentWeight << "\n";
  }
  de_report << "  Geometry-only      : "
            << (Arguments.de.path_geometry_only ? "YES" : "NO") << "\n";
  de_report << "  Old-vector recovery: "
            << (recovery_dropped_old_vector ? "DROPPED" : "UNCHANGED")
            << "\n";
  de_report << "  Max handoff turn   : "
            << Arguments.de.path_max_turn_angle << " deg\n";
  de_report << "  Recovery max turn  : "
            << Arguments.de.path_recovery_max_turn_angle << " deg\n";
  de_report << "  Current mode       : "
            << pathway_mode_label(pathway_state.mode) << "\n";
  de_report << "  Current progress   : "
            << pathway_state.current_projection.progress << " A\n";
  de_report << "  Current path dist  : "
            << pathway_state.current_projection.distance << " A\n";
  de_report << "  Min rec-lig dist   : "
            << pathway_state.minimum_receptor_ligand_distance << " A\n";
  de_report << "  Physical exit gate : "
            << (pathway_state.physical_exit_gate_enabled ? "ENABLED" : "DISABLED")
            << " (threshold " << Arguments.de.path_completion_min_separation
            << " A)\n";
  de_report << "  Arc length         : " << built.path.total_length << " A\n";
  de_report << "  Start residue      : "
            << residue_label(built.path.principal_nodes.front()) << "\n";
  de_report << "  Terminal residue   : "
            << residue_label(built.path.principal_nodes.back()) << "\n";
  de_report << "  Scan radius        : " << DE_arguments.scan_radius << " A\n";
  de_report << "  Min scan radius    : " << DE_arguments.min_scan_radius
            << " A\n";
  de_report << "  Restrain angle     : " << DE_arguments.restrain_angle
            << " deg\n";
  de_report << "  Seed               : " << DE_arguments.pull_seed << "\n\n";

  de_report << "  Benchmark\n";
  de_report << "  ---------\n";
  de_report << "  Candidate proposals: " << candidate_proposals << "\n";
  de_report << "  Total evaluations  : " << total_evaluations << "\n";
  de_report << "  Scan exclusions    : " << scan_shell_exclusions << "\n";
  de_report << "  Energy evaluations : " << energy_evaluations << "\n";
  de_report << "  Clearance pairs    : " << clearance_pair_checks << "\n";
  de_report << "  Clearance rejects  : " << clearance_rejections << "\n";
  de_report << "  Probe evals        : " << probe_evaluations << "\n";
  de_report << "  Probe clear rejects: " << probe_clearance_rejections << "\n";
  de_report << "  Probe mode rejects : " << probe_mode_rejections << "\n";
  if (Arguments.de.path_clearance_mode == "swept-linear-center") {
    de_report << "  Swept probe evals  : " << swept_probe_evaluations << "\n";
    de_report << "  Swept pair checks  : " << swept_pair_checks << "\n";
    de_report << "  Swept rejects      : " << swept_rejections << "\n";
    de_report << "  Interior-only hits : "
              << swept_interior_only_rejections << "\n";
  }
  if (guided_seed_stats.enabled) {
    de_report << "  Guided proposals   : "
              << guided_seed_stats.proposed_directions << "\n";
    de_report << "  Guided evaluations : "
              << guided_seed_stats.evaluated_directions << "\n";
    de_report << "  Guided budget grant: "
              << guided_seed_stats.budget_granted_directions << "\n";
    de_report << "  Guided valid       : "
              << guided_seed_stats.valid_directions << "\n";
  }
  de_report << "  Wall time          : " << wall_seconds << " s\n";
  de_report << "  Eval/s             : " << evals_per_second << "\n";
  de_report << "  Best stage         : " << best_stage << "\n";
  de_report << "  Eval budget stop   : "
            << (stopped_by_eval_budget ? "YES" : "NO") << "\n\n";

  if (use_corridor_grid) {
    de_report << "  Corridor Grid Stats\n";
    de_report << "  -------------------\n";
    de_report << "  Longitudinal points: "
              << corridor_grid_stats.longitudinal_points << "\n";
    de_report << "  Proposed targets   : "
              << corridor_grid_stats.proposed_targets << "\n";
    de_report << "  Evaluated targets  : "
              << corridor_grid_stats.evaluated_targets << "\n";
    de_report << "  Valid targets      : "
              << corridor_grid_stats.valid_targets << "\n\n";
  } else if (use_direct_family) {
    de_report << "  DIRECT Stats\n";
    de_report << "  ------------\n";
    de_report << "  Effective budget   : " << direct_stats.effective_eval_budget
              << "\n";
    de_report << "  Used fallback budg.: "
              << (direct_stats.used_fallback_budget ? "YES" : "NO") << "\n";
    de_report << "  Expansions         : " << direct_stats.expansions << "\n";
    de_report << "  Batches            : " << direct_stats.batches << "\n";
    de_report << "  Max batch cells    : " << direct_stats.max_batch_cells
              << "\n";
    de_report << "  Cells remaining    : " << direct_stats.cells_remaining
              << "\n";
    de_report << "  Best expansion     : " << direct_stats.best_expansion
              << "\n";
    de_report << "  Best cell size     : " << direct_stats.best_cell_size
              << "\n";
    de_report << "  Local threshold    : "
              << pathway_direct_local_threshold_size(direct_domain) << "\n";
    de_report << "  Local stop         : "
              << (direct_stats.stopped_by_local_threshold ? "YES" : "NO")
              << "\n";
    de_report << "  Stagnation stop    : "
              << (direct_stats.stopped_by_stagnation ? "YES" : "NO")
              << "\n\n";
  } else if (use_variant_de) {
    de_report << "  DE Variant Stats\n";
    de_report << "  ----------------\n";
    de_report << "  Reduction mode   : "
              << (de_variant_stats.nonlinear_population_reduction ? "nonlinear"
                                                                  : "linear/fixed")
              << "\n";
    de_report << "  Initial pop       : " << de_variant_stats.initial_population
              << "\n";
    de_report << "  Final pop         : " << de_variant_stats.final_population
              << "\n";
    de_report << "  Reductions        : "
              << de_variant_stats.population_reductions << "\n";
    de_report << "  Generations done  : "
              << de_variant_stats.generations_completed << "\n";
    de_report << "  Best generation   : " << de_variant_stats.best_generation
              << "\n";
    de_report << "  Final p-best      : " << de_variant_stats.final_p_best_rate
              << "\n";
    if (de_variant_stats.selective_pressure > 0.0) {
      de_report << "  Selective press.  : "
                << de_variant_stats.selective_pressure << "\n";
    }
    if (de_variant_stats.nonlinear_population_reduction) {
      de_report << "  Bias power F      : " << de_variant_stats.bias_power_f
                << "\n";
      de_report << "  Bias power CR     : " << de_variant_stats.bias_power_cr
                << "\n";
    }
    de_report << "  Stagnation stop   : "
              << (de_variant_stats.stopped_by_stagnation ? "YES" : "NO")
              << "\n\n";
  }

  if (pattern_stats.enabled) {
    const std::string title = use_direct_pattern
                                  ? std::string("DIRECT Pattern Refinement")
                              : use_variant_de
                                  ? variant_pattern_section_title(
                                        de_variant_stats.variant)
                                  : std::string("Pattern Refinement");
    de_report << "  " << title << "\n";
    de_report << "  " << std::string(title.size(), '-') << "\n";
    if (use_variant_pattern) {
      de_report << "  Variant budget     : " << variant_phase_budget << "\n";
      de_report << "  Pattern reserve    : " << pattern_reserved_budget << "\n";
    }
    if (use_direct_pattern) {
      de_report << "  Variant budget     : " << direct_phase_budget << "\n";
      de_report << "  Pattern reserve    : " << pattern_reserved_budget << "\n";
    }
    de_report << "  Raw score before   : " << pattern_stats.initial_score << "\n";
    de_report << "  Raw score after    : " << pattern_stats.final_score << "\n";
    de_report << "  Improved           : "
              << (pattern_stats.improved ? "YES" : "NO") << "\n";
    de_report << "  Sweeps             : " << pattern_stats.sweeps << "\n";
    de_report << "  Accepted moves     : " << pattern_stats.accepted_moves
              << "\n";
    de_report << "  Extra evals        : " << pattern_stats.evaluations_used
              << "\n";
    de_report << "  Seeds tried        : " << pattern_stats.seeds_attempted
              << "\n";
    de_report << "  Distinct basins    : "
              << pattern_stats.distinct_seed_basins << "\n\n";
  }

  de_report << "  Best Candidate\n";
  de_report << "  --------------\n";
  de_report << "  Best raw score     : " << best_candidate.raw_score << "\n";
  de_report << "  Best energy key    : " << best_candidate.energy_key << "\n";
  de_report << "  Archive E-key min  : "
            << final_selection.minimum_energy_key << "\n";
  de_report << "  Valid archive size : "
            << final_selection.valid_archive_size << "\n";
  de_report << "  Forward candidates : "
            << final_selection.forward_archive_size << "\n";
  de_report << "  Epsilon eligible   : "
            << final_selection.epsilon_eligible_size << "\n";
  de_report << "  Mode eligible      : "
            << final_selection.mode_eligible_size << "\n";
  de_report << "  Maximum probe tier : "
            << final_selection.maximum_probe_tier_size << "\n";
  if (Arguments.de.path_navigation_enabled) {
    de_report << "  Navigation eligible: "
              << final_selection.navigation_eligible_size << "\n";
    if (std::isfinite(final_selection.minimum_navigation_cost))
      de_report << "  Energy-pool nav min: "
                << final_selection.minimum_navigation_cost << "\n";
  }
  de_report << "  Final selection    : "
            << (pathway_state.mode == PathwayMode::Recovery
                    ? "recovery lexicographic"
                : Arguments.de.path_navigation_enabled
                    ? "energy epsilon, navigation epsilon, then maximum progress"
                : Arguments.de.path_geometry_only
                    ? "maximum progress geometry-only"
                    : "maximum progress within energy-key epsilon")
            << "\n";
  de_report << "  Requested progress : " << best_candidate.progress << " A\n";
  de_report << "  Requested dProgress: "
            << best_candidate.requested_delta_progress << " A\n";
  de_report << "  Applied progress   : " << best_candidate.applied_progress
            << " A\n";
  de_report << "  Applied dProgress  : " << best_candidate.delta_progress
            << " A\n";
  de_report << "  Applied path dist  : "
            << best_candidate.applied_path_distance << " A\n";
  de_report << "  Route target vector: ("
            << best_candidate.requested_vector.Fx << ", "
            << best_candidate.requested_vector.Fy << ", "
            << best_candidate.requested_vector.Fz << ")\n";
  de_report << "  Route target angle : "
            << best_candidate.requested_angle_deg << " deg\n";
  de_report << "  Turn projection    : "
            << (best_candidate.turn_capped ? "CAPPED" : "UNCHANGED") << "\n";
  de_report << "  Pull vector        : (" << best_candidate.vector.Fx << ", "
            << best_candidate.vector.Fy << ", " << best_candidate.vector.Fz
            << ")\n";
  de_report << "  Requested pull dist: "
            << best_candidate.requested_pull_distance << " A\n";
  de_report << "  Applied pull dist  : " << best_candidate.pull_distance
            << " A\n";
  de_report << "  Corridor offset    : ("
            << best_candidate.corridor_offset_u << ", "
            << best_candidate.corridor_offset_v << ") A of R="
            << best_candidate.corridor_radius << " A\n";
  de_report << "  Applied clearance  : "
            << best_candidate.minimum_clearance << " A\n";
  if (Arguments.de.path_clearance_mode == "swept-linear-center") {
    de_report << "  Endpoint clearance : "
              << best_candidate.endpoint_minimum_clearance << " A\n";
    de_report << "  Swept clearance    : "
              << best_candidate.swept_minimum_clearance << " A at tau="
              << best_candidate.swept_minimum_fraction << "\n";
    de_report << "  Swept witness pair : receptor["
              << best_candidate.swept_minimum_receptor_index << "] ligand["
              << best_candidate.swept_minimum_ligand_index << "]\n";
    de_report << "  Swept min complete : "
              << (best_candidate.swept_minimum_complete ? "YES" : "NO")
              << "\n";
  }
  if (best_candidate.navigation_enabled) {
    de_report << "  Guided seed        : "
              << (best_candidate.guided_seed ? "YES" : "NO") << " ["
              << best_candidate.guided_seed_index << "]\n";
    de_report << "  Applied corridor R : "
              << best_candidate.applied_corridor_radius << " A\n";
    de_report << "  Navigation excess  : "
              << best_candidate.navigation_tube_excess << " A\n";
    de_report << "  Navigation backtrack: "
              << best_candidate.navigation_backtrack_distance << " A\n";
    de_report << "  Navigation tube cost: "
              << best_candidate.navigation_tube_cost << "\n";
    de_report << "  Navigation back cost: "
              << best_candidate.navigation_backtrack_cost << "\n";
    de_report << "  Navigation cost    : "
              << best_candidate.navigation_cost
              << " (dimensionless algorithmic)\n";
  }
  de_report << "  Angle to old       : " << best_angle << " deg\n\n";

  de_report << "  Ranked Candidate Table\n";
  de_report << "  ----------------------\n";
  de_report << std::setw(8) << "Rank" << std::setw(8) << "Index"
            << std::setw(8) << "Count" << std::setw(14) << "RawScore"
            << std::setw(14) << "EKey" << std::setw(14) << "Progress"
            << std::setw(14) << "dProg" << std::setw(14) << "NavCost"
            << std::setw(12) << "Pull(A)"
            << std::setw(12) << "Angle" << std::setw(18) << "Elec"
            << std::setw(18) << "VdW" << "\n";
  for (std::size_t rank = 0;
       rank < std::min<std::size_t>(50, report_rows.size());
       ++rank) {
    const auto &row = report_rows[rank];
    const auto &candidate = row.representative;
    de_report << std::setw(8) << rank + 1 << std::setw(8)
              << row.representative_index << std::setw(8) << row.count
              << std::scientific << std::setw(14) << candidate.raw_score
              << std::setw(14) << candidate.energy_key << std::fixed
              << std::setw(14) << candidate.progress << std::setw(14)
              << candidate.delta_progress << std::setw(14)
              << (candidate.navigation_enabled ? candidate.navigation_cost
                                               : 0.0)
              << std::setw(12)
              << candidate.pull_distance << std::setw(12)
              << candidate.angle_deg
              << std::scientific << std::setw(18)
              << candidate.energy.elec_ener << std::setw(18)
              << candidate.energy.vdw_ener << std::fixed << "\n";
  }

  de_output << std::fixed << std::setprecision(4);
  de_output << best_candidate.vector.Fx << "  " << best_candidate.vector.Fy
            << "  " << best_candidate.vector.Fz << "  Angle: " << best_angle
            << " -- Modul: " << best_candidate.vector.modul() << " Angstrom\n";
  de_output << "# Pull g2 toward the selected reachable point on the forward "
               "route\n";
  if (built.path.explicit_route)
    de_output << "# Route ID: " << built.path.route_id << "\n";
  de_output << "# Current progress: " << pathway_state.current_projection.progress
            << " A\n";
  de_output << "# Terminal residue: "
            << residue_label(built.path.principal_nodes.back()) << "\n";
  de_output << "# Lig COM: (" << start_com[0] << ", " << start_com[1] << ", "
            << start_com[2] << ") -> (" << best_com[0] << ", " << best_com[1]
            << ", " << best_com[2] << ")\n";

  de_log << std::fixed << std::setprecision(4);
  de_log << best_candidate.vector.Fx << "  " << best_candidate.vector.Fy
         << "  " << best_candidate.vector.Fz << "  Angle: " << best_angle
         << " -- Modul: " << best_candidate.vector.modul()
         << " Angstrom  RawScore: " << best_candidate.raw_score
         << "  EnergyKey: " << best_candidate.energy_key
         << "  Progress: " << best_candidate.progress
         << "\n--------------------------\n";

  MOL best_conf_mol{lg.m_mol[0]};
  best_conf_mol.translate(best_candidate.vector.Fx, best_candidate.vector.Fy,
                          best_candidate.vector.Fz);
  best_conf.precision(3);
  for (std::size_t i = 0; i < best_conf_mol.size(); ++i) {
    best_conf << "ATOM" << std::setw(7) << i + 1 << std::setw(5)
              << lg.info[i].get_atom() << std::setw(4)
              << lg.info[i].get_original_res_name() << " A" << std::setw(4) << 1
              << std::setw(12) << std::fixed << best_conf_mol.pos.x[i]
              << std::setw(8) << best_conf_mol.pos.y[i] << std::setw(8)
              << best_conf_mol.pos.z[i] << std::setw(7) << 1.00
              << std::setw(6) << 0.00 << '\n';
  }

  vector_pdb.precision(3);
  vector_pdb << "ATOM" << std::setw(7) << 1 << std::setw(5) << "CA"
             << std::setw(4) << "ALA"
             << " A" << std::setw(4) << 1 << std::setw(12) << std::fixed
             << start_com[0] << std::setw(8) << start_com[1] << std::setw(8)
             << start_com[2] << " " << std::setw(7) << 1.00 << " "
             << std::setw(6) << 0.00 << '\n';
  vector_pdb << "ATOM" << std::setw(7) << 2 << std::setw(5) << "CB"
             << std::setw(4) << "ALA"
             << " A" << std::setw(4) << 1 << std::setw(12) << std::fixed
             << best_com[0] << std::setw(8) << best_com[1] << std::setw(8)
             << best_com[2] << " " << std::setw(7) << 1.00 << " "
             << std::setw(6) << 0.00 << '\n';
  vector_pdb << "TER\n";
  return 0;
}

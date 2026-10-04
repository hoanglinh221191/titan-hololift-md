#include "tabi_fmm_kifmm_hierarchy_scalar.h"

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

void hash_u64(std::uint64_t &hash, std::uint64_t value) noexcept {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  hash_u64(hash, bits);
}

std::uint64_t hash_values(std::span<const double> values) noexcept {
  std::uint64_t hash = kFnvOffset;
  for (const double value : values) {
    hash_double(hash, value);
  }
  return hash == 0 ? 1 : hash;
}

bool finite_source(const KifmmScalarSource &source) noexcept {
  return std::isfinite(source.position.x) &&
         std::isfinite(source.position.y) &&
         std::isfinite(source.position.z) && std::isfinite(source.strength);
}

std::vector<Vec3> source_positions(std::span<const KifmmScalarSource> sources) {
  std::vector<Vec3> positions;
  positions.reserve(sources.size());
  for (const KifmmScalarSource &source : sources) {
    positions.push_back(source.position);
  }
  return positions;
}

KifmmBox box_from_node(const KifmmHierarchyNode &node) noexcept {
  return {node.center, node.half_width, node.level,
          static_cast<std::size_t>(node.id)};
}

std::vector<KifmmScalarSource>
gather_sources(std::span<const KifmmScalarSource> sources,
               const std::vector<int> &indices) {
  std::vector<KifmmScalarSource> gathered;
  gathered.reserve(indices.size());
  for (const int index : indices) {
    gathered.push_back(sources[static_cast<std::size_t>(index)]);
  }
  return gathered;
}

void add_direct_pair_contribution(std::span<const KifmmScalarSource> sources,
                                  std::span<const Vec3> targets,
                                  const KifmmHierarchyNode &source_node,
                                  const KifmmHierarchyNode &target_node,
                                  KifmmKernel kernel, double kappa,
                                  std::vector<double> &values) {
  for (const int target_index : target_node.target_indices) {
    double sum = 0.0;
    double c = 0.0;
    const Vec3 target = targets[static_cast<std::size_t>(target_index)];
    for (const int source_index : source_node.source_indices) {
      const KifmmScalarSource &source =
          sources[static_cast<std::size_t>(source_index)];
      const double y =
          source.strength *
              scalar_kernel_value(kernel, target, source.position, kappa) -
          c;
      const double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    values[static_cast<std::size_t>(target_index)] += sum;
  }
}

KifmmEquivalentSolveResult solve_source_equivalent(
    const KifmmSurfaceRule &rule, KifmmEquivalentSystemCache &system_cache,
    std::span<const KifmmScalarSource> sources,
    const KifmmHierarchyNode &source_node, KifmmKernel kernel, double kappa,
    const std::string &solver, const std::string &condition_policy,
    double &rhs_seconds, double &solve_seconds) {
  const KifmmBox box = box_from_node(source_node);
  const auto system = build_equivalent_system_cached(
      system_cache, rule, box, kernel, kappa, solver, condition_policy);
  if (!system.finite) {
    KifmmEquivalentSolveResult failed;
    failed.status = system.status;
    failed.solver = solver;
    failed.finite = false;
    return failed;
  }
  const std::vector<KifmmScalarSource> node_sources =
      gather_sources(sources, source_node.source_indices);
  const auto rhs_start = Clock::now();
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(system.check_points.size()));
  for (std::size_t i = 0; i < system.check_points.size(); ++i) {
    rhs[static_cast<Eigen::Index>(i)] = direct_scalar_potential(
        node_sources, kernel, kappa, system.check_points[i].x);
  }
  rhs_seconds += elapsed_seconds(rhs_start, Clock::now());
  const auto solve_start = Clock::now();
  KifmmEquivalentSolveResult result =
      solve_equivalent_rhs(system, rhs, solver);
  solve_seconds += elapsed_seconds(solve_start, Clock::now());
  return result;
}

KifmmEquivalentSolveResult solve_rhs_equivalent(
    const KifmmSurfaceRule &rule, KifmmEquivalentSystemCache &system_cache,
    const KifmmHierarchyNode &node, KifmmKernel kernel, double kappa,
    const std::string &solver, const std::string &condition_policy,
    const Eigen::VectorXd &rhs, double &solve_seconds) {
  const KifmmBox box = box_from_node(node);
  const auto system = build_equivalent_system_cached(
      system_cache, rule, box, kernel, kappa, solver, condition_policy);
  const auto solve_start = Clock::now();
  KifmmEquivalentSolveResult result = solve_equivalent_rhs(system, rhs, solver);
  solve_seconds += elapsed_seconds(solve_start, Clock::now());
  return result;
}

std::vector<std::vector<int>>
build_nodes_by_level(const KifmmHierarchyTree &tree) {
  int max_level = 0;
  for (const KifmmHierarchyNode &node : tree.nodes) {
    max_level = std::max(max_level, node.level);
  }
  std::vector<std::vector<int>> levels(static_cast<std::size_t>(max_level + 1));
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (node.level >= 0 && node.level <= max_level) {
      levels[static_cast<std::size_t>(node.level)].push_back(node.id);
    }
  }
  return levels;
}

bool hierarchy_links_are_valid(const KifmmHierarchyTree &tree) noexcept {
  for (const KifmmHierarchyNode &node : tree.nodes) {
    for (const int child_id : node.child) {
      if (child_id < 0) {
        continue;
      }
      if (static_cast<std::size_t>(child_id) >= tree.nodes.size()) {
        return false;
      }
      const KifmmHierarchyNode &child =
          tree.nodes[static_cast<std::size_t>(child_id)];
      if (child.parent != node.id || child.level != node.level + 1) {
        return false;
      }
    }
  }
  return true;
}

bool finite_vector(const Eigen::VectorXd &values) noexcept {
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      return false;
    }
  }
  return true;
}

bool ensure_rhs_for_node(std::vector<Eigen::VectorXd> &rhs_by_node,
                         const KifmmEquivalentSystem &system, int node_id) {
  Eigen::VectorXd &rhs = rhs_by_node[static_cast<std::size_t>(node_id)];
  if (rhs.size() == 0) {
    rhs = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(system.check_points.size()));
    return true;
  }
  return rhs.size() == static_cast<Eigen::Index>(system.check_points.size());
}

void add_equivalent_to_points(const KifmmSurfaceRule &source_rule,
                              const KifmmHierarchyNode &source_node,
                              const KifmmEquivalentSolveResult &source_equiv,
                              KifmmKernel kernel, double kappa,
                              std::span<const KifmmSurfacePoint> points,
                              Eigen::VectorXd &rhs) {
  const KifmmBox source_box = box_from_node(source_node);
  for (std::size_t i = 0; i < points.size(); ++i) {
    rhs[static_cast<Eigen::Index>(i)] += equivalent_scalar_potential(
        source_rule, source_box, source_equiv, kernel, kappa, points[i].x);
  }
}

double relative_rms(std::span<const double> approx,
                    std::span<const double> direct) noexcept {
  if (approx.size() != direct.size() || approx.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  long double err_sq = 0.0L;
  long double ref_sq = 0.0L;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const long double d =
        static_cast<long double>(approx[i]) - static_cast<long double>(direct[i]);
    err_sq += d * d;
    ref_sq += static_cast<long double>(direct[i]) *
              static_cast<long double>(direct[i]);
  }
  return std::sqrt(static_cast<double>(
      err_sq / std::max(ref_sq, static_cast<long double>(
                                    std::numeric_limits<double>::min()))));
}

double max_abs_error(std::span<const double> approx,
                     std::span<const double> direct) noexcept {
  double max_error = 0.0;
  for (std::size_t i = 0; i < approx.size() && i < direct.size(); ++i) {
    max_error = std::max(max_error, std::abs(approx[i] - direct[i]));
  }
  return max_error;
}

bool finite_values(std::span<const double> values) noexcept {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

} // namespace

KifmmHierarchyScalarResult
run_hierarchical_scalar_kifmm(std::span<const KifmmScalarSource> sources,
                              std::span<const Vec3> targets,
                              const KifmmHierarchyScalarConfig &config) {
  const auto total_start = Clock::now();
  KifmmHierarchyScalarResult result;
  if (sources.empty() || targets.empty()) {
    result.status = "empty_input";
    return result;
  }
  if (!std::isfinite(config.kappa) || config.kappa < 0.0) {
    result.status = "invalid_kappa";
    return result;
  }
  for (const KifmmScalarSource &source : sources) {
    if (!finite_source(source)) {
      result.status = "invalid_source";
      return result;
    }
  }

  const std::vector<Vec3> positions = source_positions(sources);
  const KifmmHierarchyTree tree =
      build_strict_octree(positions, targets, config.hierarchy);
  result.stats = tree.stats;
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  const KifmmHierarchyWorklist worklist =
      build_hierarchy_worklist(tree, config.hierarchy);
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  result.stats.far_pair_count = static_cast<int>(worklist.m2l_pairs.size());
  result.stats.direct_pair_count =
      static_cast<int>(worklist.direct_leaf_pairs.size());
  result.stats.mac_accepted_pairs = worklist.mac_accepted_pairs;
  result.stats.check_guard_rejected_pairs =
      worklist.check_guard_rejected_pairs;
  result.stats.leaf_direct_pairs = worklist.leaf_direct_pairs;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }

  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.far_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.direct_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  result.stats.exact_pair_coverage =
      result.stats.far_panel_pairs + result.stats.direct_panel_pairs;
  const std::uint64_t expected_pairs =
      static_cast<std::uint64_t>(sources.size()) *
      static_cast<std::uint64_t>(targets.size());
  if (result.stats.exact_pair_coverage != expected_pairs) {
    result.status = "coverage_gap";
    return result;
  }
  result.stats.far_fraction =
      expected_pairs == 0
          ? 0.0
          : static_cast<double>(result.stats.far_panel_pairs) /
                static_cast<double>(expected_pairs);

  std::vector<double> approx(targets.size(), 0.0);
  std::vector<double> direct(targets.size(), 0.0);

  const auto direct_fallback_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    add_direct_pair_contribution(
        sources, targets, tree.nodes[static_cast<std::size_t>(source_node_id)],
        tree.nodes[static_cast<std::size_t>(target_node_id)], config.kernel,
        config.kappa, approx);
  }
  result.direct_fallback_seconds =
      elapsed_seconds(direct_fallback_start, Clock::now());

  KifmmEquivalentSystemCache system_cache;
  std::unordered_map<int, KifmmEquivalentSolveResult> source_equivalents;
  const KifmmSurfaceRule rule =
      make_cube_surface_rule(config.hierarchy.order,
                             config.hierarchy.equiv_scale,
                             config.hierarchy.upward_check_scale);
  const std::string solver = "qr";
  const auto p2m_start = Clock::now();
  double rhs_total_seconds = 0.0;
  double solve_total_seconds = 0.0;
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    (void)target_node_id;
    if (source_equivalents.find(source_node_id) != source_equivalents.end()) {
      continue;
    }
    double rhs_seconds = 0.0;
    double solve_seconds = 0.0;
    KifmmEquivalentSolveResult equivalent = solve_source_equivalent(
        rule, system_cache, sources,
        tree.nodes[static_cast<std::size_t>(source_node_id)], config.kernel,
        config.kappa, solver, config.hierarchy.condition_policy, rhs_seconds,
        solve_seconds);
    rhs_total_seconds += rhs_seconds;
    solve_total_seconds += solve_seconds;
    if (!equivalent.finite || equivalent.status != "pass") {
      result.status = "p2m_" + equivalent.status;
      return result;
    }
    source_equivalents.emplace(source_node_id, std::move(equivalent));
  }
  result.p2m_seconds = elapsed_seconds(p2m_start, Clock::now());
  result.p2m_rhs_seconds = rhs_total_seconds;
  result.p2m_solve_seconds = solve_total_seconds;
  result.m2m_seconds = 0.0;

  const auto m2l_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmEquivalentSolveResult &equivalent =
        source_equivalents.at(source_node_id);
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmBox box = box_from_node(source_node);
    for (const int target_index : target_node.target_indices) {
      const double value = equivalent_scalar_potential(
          rule, box, equivalent, config.kernel, config.kappa,
          targets[static_cast<std::size_t>(target_index)]);
      approx[static_cast<std::size_t>(target_index)] += value;
    }
  }
  result.m2l_seconds = elapsed_seconds(m2l_start, Clock::now());

  const auto oracle_start = Clock::now();
  if (config.build_dense_oracle) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
      direct[i] = direct_scalar_potential(sources, config.kernel, config.kappa,
                                          targets[i]);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());

  if (!finite_values(approx) || !finite_values(direct)) {
    result.status = "nonfinite_output";
    return result;
  }
  result.aggregate_relative_rms = relative_rms(approx, direct);
  result.max_abs_error = max_abs_error(approx, direct);
  result.output_hash = hash_values(approx);
  result.equivalent_factor_cache_entries =
      static_cast<int>(system_cache.entries.size());
  result.equivalent_factor_cache_hits =
      static_cast<int>(system_cache.stats.hits);
  result.equivalent_factor_cache_misses =
      static_cast<int>(system_cache.stats.misses);
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.status =
      std::isfinite(result.aggregate_relative_rms) ? "ok" : "nonfinite_rms";
  return result;
}

KifmmHierarchyScalarResult
run_full_hierarchical_scalar_kifmm(
    std::span<const KifmmScalarSource> sources, std::span<const Vec3> targets,
    const KifmmHierarchyScalarConfig &config) {
  const auto total_start = Clock::now();
  KifmmHierarchyScalarResult result;
  if (sources.empty() || targets.empty()) {
    result.status = "empty_input";
    return result;
  }
  if (!std::isfinite(config.kappa) || config.kappa < 0.0) {
    result.status = "invalid_kappa";
    return result;
  }
  for (const KifmmScalarSource &source : sources) {
    if (!finite_source(source)) {
      result.status = "invalid_source";
      return result;
    }
  }

  const std::vector<Vec3> positions = source_positions(sources);
  const KifmmHierarchyTree tree =
      build_strict_octree(positions, targets, config.hierarchy);
  result.stats = tree.stats;
  result.tree_hash = tree.tree_hash;
  if (tree.status != "ok") {
    result.status = tree.status;
    return result;
  }
  if (!hierarchy_links_are_valid(tree)) {
    result.status = "invalid_tree_links";
    return result;
  }
  const KifmmHierarchyWorklist worklist =
      build_hierarchy_worklist(tree, config.hierarchy);
  result.worklist_hash = worklist.worklist_hash;
  result.policy_hash = worklist.policy_hash;
  result.stats.far_pair_count = static_cast<int>(worklist.m2l_pairs.size());
  result.stats.direct_pair_count =
      static_cast<int>(worklist.direct_leaf_pairs.size());
  result.stats.mac_accepted_pairs = worklist.mac_accepted_pairs;
  result.stats.check_guard_rejected_pairs =
      worklist.check_guard_rejected_pairs;
  result.stats.leaf_direct_pairs = worklist.leaf_direct_pairs;
  if (worklist.worklist_hash == 0) {
    result.status = "worklist_gap";
    return result;
  }

  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.far_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    const KifmmHierarchyNode &source_node =
        tree.nodes[static_cast<std::size_t>(source_node_id)];
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    result.stats.direct_panel_pairs +=
        static_cast<std::uint64_t>(source_node.source_indices.size()) *
        static_cast<std::uint64_t>(target_node.target_indices.size());
  }
  result.stats.exact_pair_coverage =
      result.stats.far_panel_pairs + result.stats.direct_panel_pairs;
  const std::uint64_t expected_pairs =
      static_cast<std::uint64_t>(sources.size()) *
      static_cast<std::uint64_t>(targets.size());
  if (result.stats.exact_pair_coverage != expected_pairs) {
    result.status = "coverage_gap";
    return result;
  }
  result.stats.far_fraction =
      expected_pairs == 0
          ? 0.0
          : static_cast<double>(result.stats.far_panel_pairs) /
                static_cast<double>(expected_pairs);

  std::vector<double> approx(targets.size(), 0.0);
  std::vector<double> direct(targets.size(), 0.0);

  const auto direct_fallback_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] :
       worklist.direct_leaf_pairs) {
    add_direct_pair_contribution(
        sources, targets, tree.nodes[static_cast<std::size_t>(source_node_id)],
        tree.nodes[static_cast<std::size_t>(target_node_id)], config.kernel,
        config.kappa, approx);
  }
  result.direct_fallback_seconds =
      elapsed_seconds(direct_fallback_start, Clock::now());

  const auto upward_rule =
      make_cube_surface_rule(config.hierarchy.order,
                             config.hierarchy.equiv_scale,
                             config.hierarchy.upward_check_scale);
  const auto local_rule = make_cube_local_surface_rule(
      config.hierarchy.order, config.hierarchy.local_equiv_scale,
      config.hierarchy.local_check_scale);
  const std::string solver = "qr";
  KifmmEquivalentSystemCache upward_cache;
  KifmmEquivalentSystemCache local_cache;
  const std::size_t node_count = tree.nodes.size();
  std::vector<KifmmEquivalentSolveResult> upward(node_count);
  std::vector<bool> upward_valid(node_count, false);
  std::vector<Eigen::VectorXd> local_rhs(node_count);
  std::vector<KifmmEquivalentSolveResult> local(node_count);
  std::vector<bool> local_valid(node_count, false);
  const std::vector<std::vector<int>> nodes_by_level =
      build_nodes_by_level(tree);

  const auto p2m_start = Clock::now();
  double p2m_rhs_seconds = 0.0;
  double p2m_solve_seconds = 0.0;
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (!node.is_leaf || node.source_indices.empty()) {
      continue;
    }
    double rhs_seconds = 0.0;
    double solve_seconds = 0.0;
    KifmmEquivalentSolveResult equivalent = solve_source_equivalent(
        upward_rule, upward_cache, sources, node, config.kernel, config.kappa,
        solver, config.hierarchy.condition_policy, rhs_seconds, solve_seconds);
    p2m_rhs_seconds += rhs_seconds;
    p2m_solve_seconds += solve_seconds;
    if (!equivalent.finite || equivalent.status != "pass") {
      result.status = "p2m_" + equivalent.status;
      return result;
    }
    upward[static_cast<std::size_t>(node.id)] = std::move(equivalent);
    upward_valid[static_cast<std::size_t>(node.id)] = true;
    ++result.p2m_leaf_solves;
  }
  result.p2m_seconds = elapsed_seconds(p2m_start, Clock::now());
  result.p2m_rhs_seconds = p2m_rhs_seconds;
  result.p2m_solve_seconds = p2m_solve_seconds;

  const auto m2m_start = Clock::now();
  for (int level = static_cast<int>(nodes_by_level.size()) - 2; level >= 0;
       --level) {
    for (const int node_id : nodes_by_level[static_cast<std::size_t>(level)]) {
      const KifmmHierarchyNode &node =
          tree.nodes[static_cast<std::size_t>(node_id)];
      if (node.source_indices.empty()) {
        continue;
      }
      const KifmmBox box = box_from_node(node);
      const auto system = build_equivalent_system_cached(
          upward_cache, upward_rule, box, config.kernel, config.kappa, solver,
          config.hierarchy.condition_policy);
      if (!system.finite) {
        result.status = "m2m_" + system.status;
        return result;
      }
      Eigen::VectorXd rhs =
          Eigen::VectorXd::Zero(static_cast<Eigen::Index>(
              system.check_points.size()));
      bool used_child = false;
      for (const int child_id : node.child) {
        if (child_id < 0 ||
            !upward_valid[static_cast<std::size_t>(child_id)]) {
          continue;
        }
        add_equivalent_to_points(
            upward_rule, tree.nodes[static_cast<std::size_t>(child_id)],
            upward[static_cast<std::size_t>(child_id)], config.kernel,
            config.kappa, system.check_points, rhs);
        result.upward_rhs_adds +=
            static_cast<std::uint64_t>(system.check_points.size());
        ++result.m2m_translations;
        used_child = true;
      }
      if (!used_child || !finite_vector(rhs)) {
        continue;
      }
      double solve_seconds = 0.0;
      KifmmEquivalentSolveResult equivalent = solve_rhs_equivalent(
          upward_rule, upward_cache, node, config.kernel, config.kappa, solver,
          config.hierarchy.condition_policy, rhs, solve_seconds);
      if (!equivalent.finite || equivalent.status != "pass") {
        result.status = "m2m_" + equivalent.status;
        return result;
      }
      upward[static_cast<std::size_t>(node.id)] = std::move(equivalent);
      upward_valid[static_cast<std::size_t>(node.id)] = true;
      ++result.m2m_parent_solves;
    }
  }
  result.m2m_seconds = elapsed_seconds(m2m_start, Clock::now());

  const auto m2l_start = Clock::now();
  for (const auto &[target_node_id, source_node_id] : worklist.m2l_pairs) {
    if (!upward_valid[static_cast<std::size_t>(source_node_id)]) {
      result.status = "m2l_missing_upward";
      return result;
    }
    const KifmmHierarchyNode &target_node =
        tree.nodes[static_cast<std::size_t>(target_node_id)];
    const KifmmBox target_box = box_from_node(target_node);
    const auto target_system = build_equivalent_system_cached(
        local_cache, local_rule, target_box, config.kernel, config.kappa,
        solver, config.hierarchy.condition_policy);
    if (!target_system.finite ||
        !ensure_rhs_for_node(local_rhs, target_system, target_node_id)) {
      result.status = "m2l_local_system_gap";
      return result;
    }
    add_equivalent_to_points(
        upward_rule, tree.nodes[static_cast<std::size_t>(source_node_id)],
        upward[static_cast<std::size_t>(source_node_id)], config.kernel,
        config.kappa, target_system.check_points,
        local_rhs[static_cast<std::size_t>(target_node_id)]);
    result.local_rhs_adds +=
        static_cast<std::uint64_t>(target_system.check_points.size());
    ++result.m2l_translations;
  }
  result.m2l_seconds = elapsed_seconds(m2l_start, Clock::now());

  double local_solve_seconds = 0.0;
  const auto l2l_start = Clock::now();
  for (std::size_t level = 0; level < nodes_by_level.size(); ++level) {
    for (const int node_id : nodes_by_level[level]) {
      const KifmmHierarchyNode &node =
          tree.nodes[static_cast<std::size_t>(node_id)];
      Eigen::VectorXd &rhs = local_rhs[static_cast<std::size_t>(node_id)];
      if (rhs.size() > 0) {
        if (!finite_vector(rhs)) {
          result.status = "local_nonfinite_rhs";
          return result;
        }
        double solve_seconds = 0.0;
        KifmmEquivalentSolveResult equivalent = solve_rhs_equivalent(
            local_rule, local_cache, node, config.kernel, config.kappa, solver,
            config.hierarchy.condition_policy, rhs, solve_seconds);
        local_solve_seconds += solve_seconds;
        if (!equivalent.finite || equivalent.status != "pass") {
          result.status = "local_" + equivalent.status;
          return result;
        }
        local[static_cast<std::size_t>(node.id)] = std::move(equivalent);
        local_valid[static_cast<std::size_t>(node.id)] = true;
        ++result.local_node_solves;
      }
      if (!local_valid[static_cast<std::size_t>(node.id)]) {
        continue;
      }
      for (const int child_id : node.child) {
        if (child_id < 0) {
          continue;
        }
        const KifmmHierarchyNode &child =
            tree.nodes[static_cast<std::size_t>(child_id)];
        if (child.target_indices.empty()) {
          continue;
        }
        const KifmmBox child_box = box_from_node(child);
        const auto child_system = build_equivalent_system_cached(
            local_cache, local_rule, child_box, config.kernel, config.kappa,
            solver, config.hierarchy.condition_policy);
        if (!child_system.finite ||
            !ensure_rhs_for_node(local_rhs, child_system, child_id)) {
          result.status = "l2l_local_system_gap";
          return result;
        }
        add_equivalent_to_points(
            local_rule, node, local[static_cast<std::size_t>(node.id)],
            config.kernel, config.kappa, child_system.check_points,
            local_rhs[static_cast<std::size_t>(child_id)]);
        result.local_rhs_adds +=
            static_cast<std::uint64_t>(child_system.check_points.size());
        ++result.l2l_translations;
      }
    }
  }
  result.l2l_seconds = elapsed_seconds(l2l_start, Clock::now());
  result.local_solve_seconds = local_solve_seconds;

  const auto l2p_start = Clock::now();
  for (const KifmmHierarchyNode &node : tree.nodes) {
    if (!node.is_leaf || node.target_indices.empty() ||
        !local_valid[static_cast<std::size_t>(node.id)]) {
      continue;
    }
    const KifmmBox box = box_from_node(node);
    for (const int target_index : node.target_indices) {
      approx[static_cast<std::size_t>(target_index)] +=
          equivalent_scalar_potential(
              local_rule, box, local[static_cast<std::size_t>(node.id)],
              config.kernel, config.kappa,
              targets[static_cast<std::size_t>(target_index)]);
      ++result.l2p_target_count;
    }
  }
  result.l2p_seconds = elapsed_seconds(l2p_start, Clock::now());

  const auto oracle_start = Clock::now();
  if (config.build_dense_oracle) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
      direct[i] = direct_scalar_potential(sources, config.kernel, config.kappa,
                                          targets[i]);
    }
  }
  result.dense_oracle_seconds = elapsed_seconds(oracle_start, Clock::now());

  if (!finite_values(approx) || !finite_values(direct)) {
    result.status = "nonfinite_output";
    return result;
  }
  result.aggregate_relative_rms = relative_rms(approx, direct);
  result.max_abs_error = max_abs_error(approx, direct);
  result.output_hash = hash_values(approx);
  result.equivalent_factor_cache_entries =
      static_cast<int>(upward_cache.entries.size() + local_cache.entries.size());
  result.equivalent_factor_cache_hits =
      static_cast<int>(upward_cache.stats.hits + local_cache.stats.hits);
  result.equivalent_factor_cache_misses =
      static_cast<int>(upward_cache.stats.misses + local_cache.stats.misses);
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  result.status =
      std::isfinite(result.aggregate_relative_rms) ? "ok" : "nonfinite_rms";
  return result;
}

} // namespace titan::tabi::fmm_kifmm

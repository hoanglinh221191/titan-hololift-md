#ifndef TITAN_TABI_FMM_KIFMM_HIERARCHY_SCALAR_H
#define TITAN_TABI_FMM_KIFMM_HIERARCHY_SCALAR_H

#include "tabi_fmm_kifmm_hierarchy.h"
#include "tabi_fmm_kifmm_solve.h"

#include <cstdint>
#include <limits>
#include <span>
#include <string>

namespace titan::tabi::fmm_kifmm {

struct KifmmHierarchyScalarConfig {
  KifmmHierarchyPolicy hierarchy{};
  KifmmKernel kernel{KifmmKernel::Laplace};
  double kappa{0.0};
  bool build_dense_oracle{true};
};

struct KifmmHierarchyScalarResult {
  std::string status{"not_started"};
  KifmmHierarchyStats stats{};
  double aggregate_relative_rms{std::numeric_limits<double>::quiet_NaN()};
  double max_abs_error{std::numeric_limits<double>::quiet_NaN()};
  double p2m_seconds{0.0};
  double p2m_rhs_seconds{0.0};
  double p2m_solve_seconds{0.0};
  double m2m_seconds{0.0};
  double m2l_seconds{0.0};
  double local_solve_seconds{0.0};
  double l2l_seconds{0.0};
  double l2p_seconds{0.0};
  double direct_fallback_seconds{0.0};
  double dense_oracle_seconds{0.0};
  double total_seconds{0.0};
  int equivalent_factor_cache_entries{0};
  int equivalent_factor_cache_hits{0};
  int equivalent_factor_cache_misses{0};
  std::uint64_t p2m_leaf_solves{0};
  std::uint64_t m2m_translations{0};
  std::uint64_t m2m_parent_solves{0};
  std::uint64_t m2l_translations{0};
  std::uint64_t local_node_solves{0};
  std::uint64_t l2l_translations{0};
  std::uint64_t l2p_target_count{0};
  std::uint64_t local_rhs_adds{0};
  std::uint64_t upward_rhs_adds{0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
  std::uint64_t output_hash{0};
};

KifmmHierarchyScalarResult
run_hierarchical_scalar_kifmm(std::span<const KifmmScalarSource> sources,
                              std::span<const Vec3> targets,
                              const KifmmHierarchyScalarConfig &config);

KifmmHierarchyScalarResult
run_full_hierarchical_scalar_kifmm(
    std::span<const KifmmScalarSource> sources, std::span<const Vec3> targets,
    const KifmmHierarchyScalarConfig &config);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_HIERARCHY_SCALAR_H

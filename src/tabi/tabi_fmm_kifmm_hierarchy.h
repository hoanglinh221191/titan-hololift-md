#ifndef TITAN_TABI_FMM_KIFMM_HIERARCHY_H
#define TITAN_TABI_FMM_KIFMM_HIERARCHY_H

#include "tabi_fmm_kifmm_surface.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmHierarchyPolicy {
  int order{6};
  int max_depth{6};
  int leaf_size{32};
  double theta{0.6};
  double equiv_scale{1.05};
  double upward_check_scale{2.40};
  double local_equiv_scale{2.40};
  double local_check_scale{1.05};
  std::string condition_policy{"qr-rank-only"};
  std::string source_box_policy{"strict-octree"};
  std::string m2l_eval_policy{"scalar-existing"};
  // "static-translation-table-cache" stores canonical M2L translation
  // matrices by relative box geometry for the full hierarchy apply.
  // "target-node-streaming-cache" reuses M2L matrices only within one
  // target-node accumulation and releases them before the next target node.
  // "bounded-lru-cache" keeps canonical-offset keys but evicts least-recently
  // used matrices to respect m2l_matrix_cache_limit_mb.
  std::string m2l_matrix_cache_policy{"off"};
  double m2l_matrix_cache_limit_mb{0.0};
  std::string m2l_gemm_batch_policy{"per-pair"};
  std::string m2l_parallel_policy{"serial"};
  std::string tree_stop_policy{"baseline"};
  std::string jcp_channel_policy{"all"};
  double tree_panel_radius_factor{0.0};
  int tree_occupancy_stop{0};
  int m2l_parallel_workers{1};
};

struct KifmmHierarchyNode {
  int id{-1};
  int parent{-1};
  std::array<int, 8> child{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int level{0};
  Vec3 center{};
  double half_width{0.0};
  std::vector<int> source_indices{};
  std::vector<int> target_indices{};
  bool is_leaf{true};
};

struct KifmmHierarchyWorklist {
  std::vector<std::pair<int, int>> m2l_pairs{};
  std::vector<std::pair<int, int>> direct_leaf_pairs{};
  int mac_accepted_pairs{0};
  int check_guard_rejected_pairs{0};
  int leaf_direct_pairs{0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  std::uint64_t policy_hash{0};
};

struct KifmmHierarchyStats {
  int source_count{0};
  int target_count{0};
  int node_count{0};
  int leaf_count{0};
  int max_observed_depth{0};
  int unique_half_widths{0};
  int observed_level_count{0};
  int far_pair_count{0};
  int direct_pair_count{0};
  int mac_accepted_pairs{0};
  int check_guard_rejected_pairs{0};
  int leaf_direct_pairs{0};
  int radius_stop_leaf_count{0};
  int occupancy_stop_leaf_count{0};
  int leaf_size_stop_count{0};
  int max_depth_stop_count{0};
  int unsplittable_stop_count{0};
  std::uint64_t exact_pair_coverage{0};
  std::uint64_t far_panel_pairs{0};
  std::uint64_t direct_panel_pairs{0};
  double far_fraction{0.0};
  double min_leaf_half_width_over_panel_radius{0.0};
  double max_leaf_panel_radius{0.0};
  double mean_leaf_source_count{0.0};
  double mean_leaf_target_count{0.0};
};

struct KifmmHierarchyTree {
  std::string status{"not_started"};
  std::vector<KifmmHierarchyNode> nodes{};
  KifmmHierarchyStats stats{};
  std::uint64_t tree_hash{0};
};

KifmmHierarchyTree build_strict_octree(std::span<const Vec3> sources,
                                       std::span<const Vec3> targets,
                                       const KifmmHierarchyPolicy &policy);

KifmmHierarchyTree build_strict_octree(
    std::span<const Vec3> sources, std::span<const Vec3> targets,
    std::span<const double> source_radii,
    std::span<const double> target_radii, const KifmmHierarchyPolicy &policy);

KifmmHierarchyWorklist
build_hierarchy_worklist(const KifmmHierarchyTree &tree,
                         const KifmmHierarchyPolicy &policy);

void finalize_hierarchy_worklist(KifmmHierarchyWorklist &worklist,
                                 const KifmmHierarchyPolicy &policy);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_HIERARCHY_H

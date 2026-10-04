#ifndef TITAN_BIE_CARTESIAN_FMM_TYPES_H
#define TITAN_BIE_CARTESIAN_FMM_TYPES_H

#include "../../../aligned_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace titan::bie::fmm::cartesian {

inline constexpr int kMaximumExpansionOrder = 4;
inline constexpr std::size_t kMaximumCoefficientCount = 56;
inline constexpr std::size_t kJcpChannelCount = 4;

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct MultiIndex {
  int x{0};
  int y{0};
  int z{0};
};

struct PanelGeometry {
  Vec3 target{};
  Vec3 source{};
  Vec3 normal{0.0, 0.0, 1.0};
  double area{0.0};
};

struct JcpMoments {
  int order{1};
  std::size_t term_count{4};
  std::array<double, kMaximumCoefficientCount> phin{};
  std::array<std::array<double, kMaximumCoefficientCount>, 3> phi_normal{};
  bool finite_outputs{true};
};

struct JcpLocals {
  int order{1};
  std::size_t term_count{4};
  std::array<std::array<double, kMaximumCoefficientCount>, kJcpChannelCount>
      fields{};
  bool finite_outputs{true};
};

struct CartesianFmmNode {
  int parent{-1};
  std::array<int, 8> child{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int level{0};
  Vec3 center{};
  double half_width{0.0};
  std::size_t source_begin{0};
  std::size_t source_count{0};
  std::size_t target_begin{0};
  std::size_t target_count{0};
  std::size_t source_population{0};
  std::size_t target_population{0};
};

struct CartesianFmmPair {
  std::uint32_t target_node{0};
  std::uint32_t source_node{0};
  std::uint32_t translation{0};
};

struct CartesianFmmPlanOptions {
  int expansion_order{4};
  int leaf_size{64};
  int max_depth{10};
  double theta{0.45};
  double eps_ratio{80.0};
  double kappa{0.0};
};

struct CartesianFmmPlanStats {
  std::size_t panels{0};
  std::size_t nodes{0};
  std::size_t source_leaves{0};
  std::size_t target_leaves{0};
  std::size_t far_pairs{0};
  std::size_t direct_pairs{0};
  std::uint64_t far_panel_pairs{0};
  std::uint64_t direct_panel_pairs{0};
  std::uint64_t expected_panel_pairs{0};
  std::size_t translation_cache_entries{0};
  std::size_t translation_cache_hits{0};
  std::size_t translation_cache_bytes{0};
  std::size_t retained_plan_bytes{0};
  double tree_seconds{0.0};
  double worklist_seconds{0.0};
  double cache_seconds{0.0};
  double audit_seconds{0.0};
  double setup_seconds{0.0};
  double m2m_parity_relative_error{0.0};
  double l2l_parity_relative_error{0.0};
  std::uint64_t tree_hash{0};
  std::uint64_t worklist_hash{0};
  bool exact_pair_coverage{false};
  bool finite_outputs{true};
  std::string status{"not_started"};
};

struct CartesianFmmApplyStats {
  double p2m_seconds{0.0};
  double m2m_seconds{0.0};
  double m2l_seconds{0.0};
  double l2l_seconds{0.0};
  double l2p_seconds{0.0};
  double direct_seconds{0.0};
  double assemble_seconds{0.0};
  double apply_seconds{0.0};
  std::size_t retained_workspace_bytes{0};
  // Per-apply semantic direct panel-pair accounting.  This intentionally
  // excludes node-patch delta corrections: those refine coefficients for an
  // already applied direct pair and are not another panel-pair evaluation.
  std::uint64_t expected_direct_panel_pairs{0};
  std::uint64_t direct_panel_pairs{0};
  bool finite_outputs{true};
};

using CoefficientBuffer = AlignedDoubleVector;

} // namespace titan::bie::fmm::cartesian

#endif // TITAN_BIE_CARTESIAN_FMM_TYPES_H

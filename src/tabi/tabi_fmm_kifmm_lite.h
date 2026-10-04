#ifndef TITAN_TABI_FMM_KIFMM_LITE_H
#define TITAN_TABI_FMM_KIFMM_LITE_H

#include "tabi_fmm_kifmm_fields.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmLiteConfig {
  int order{6};
  int leaf_size{8};
  double opening_theta{0.6};
  double equiv_scale{1.05};
  double check_scale{2.40};
  double kappa{0.12746};
  double eps_ratio{80.0};
  double max_condition_estimate{1.0e12};
  double max_equivalent_relative_residual{1.0e-10};
  double max_aggregate_relative_rms{1.0e-6};
  double max_channel_relative_rms{5.0e-6};
  double min_far_panel_fraction{0.10};
  bool require_far_work{true};
  bool require_unit_normals{true};
  std::string solver{"qr"};
  std::string condition_policy{"qr-rank-only"};
  std::string source_box_policy{"tight-bbox"};
};

struct KifmmLiteTargetPanel {
  Vec3 position{};
  Vec3 normal{};
};

struct KifmmLiteResult {
  std::vector<KifmmJcpRow> rows{};
  std::vector<KifmmJcpChannelRow> channel_rows{};
  std::vector<KifmmJcpRow> direct_rows{};
  std::vector<KifmmJcpChannelRow> direct_channel_rows{};

  std::size_t source_panels{0};
  std::size_t target_panels{0};
  std::size_t source_nodes{0};
  std::size_t target_nodes{0};
  std::size_t source_leaves{0};
  std::size_t target_leaves{0};
  std::size_t source_tree_levels{0};
  std::size_t unique_source_node_half_widths{0};
  double source_root_half_width{0.0};
  bool level_cube_geometry{false};
  std::size_t accepted_far_cell_pairs{0};
  std::size_t direct_fallback_cell_pairs{0};
  std::size_t accepted_far_panel_pairs{0};
  std::size_t direct_fallback_panel_pairs{0};
  std::size_t equivalent_system_builds{0};
  std::size_t equivalent_rhs_solves{0};
  std::uint64_t worklist_hash{0};

  double aggregate_relative_rms{0.0};
  double k1_relative_rms{0.0};
  double k2_relative_rms{0.0};
  double k3_relative_rms{0.0};
  double k4_relative_rms{0.0};
  double max_condition_estimate{0.0};
  double max_equivalent_relative_residual{0.0};
  double min_far_panel_fraction{0.0};

  bool finite{false};
  std::string status{"not_started"};
};

KifmmLiteResult
run_kifmm_lite_matvec(std::span<const KifmmPanelSource> sources,
                      std::span<const KifmmLiteTargetPanel> targets,
                      const KifmmLiteConfig &config);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_LITE_H

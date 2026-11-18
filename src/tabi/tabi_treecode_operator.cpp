#include "tabi_treecode_internal.h"
#include "tabi_node_patch_curved.h"
#include "tabi_near.h"

#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::detail {
void apply_direct_operator_impl(const std::vector<Panel> &panels,
                           const std::vector<double> &density,
                           double contrast, double kappa,
                           TitanExecutor &executor,
                           std::vector<double> &out,
                           TreecodeStats *stats) {
  const std::size_t n = panels.size();
  out.assign(n, 0.0);
  if (n == 0)
    return;
  const std::size_t workers = effective_workers(executor, n);
  std::vector<std::size_t> local_near(workers, 0);
  executor.parallel_for(
      n, workers, 16,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::size_t near_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const double coupled =
              direct_sum(panels[i], i, panels, density, kappa, 0, n);
          near_count += n > 0 ? n - 1 : 0;
          out[i] = panels[i].diagonal * density[i] + contrast * coupled;
        }
        local_near[worker] += near_count;
      });
  if (stats != nullptr) {
    stats->matvec_mode = "direct";
    stats->taylor_order = 0;
    stats->derivative_mode = "none";
    stats->near_interactions +=
        std::accumulate(local_near.begin(), local_near.end(), std::size_t{0});
  }
}

void apply_treecode_operator_impl(const std::vector<Panel> &panels,
                             const std::vector<double> &density,
                             double contrast, double kappa,
                             const TreeWorkspace &tree,
                             TitanExecutor &executor,
                             std::vector<double> &out,
                             TreecodeStats *stats) {
  const std::size_t n = panels.size();
  if (n == 0) {
    out.clear();
    return;
  }
  if (tree.nodes.empty() || tree.order.size() != n) {
    apply_direct_operator(panels, density, contrast, kappa, executor, out,
                          stats);
    return;
  }

  const auto eval_begin = std::chrono::steady_clock::now();
  const int taylor_order = std::clamp(tree.taylor_order, 0, kTaylorOrderMax);
  const auto moment_begin = std::chrono::steady_clock::now();
  const std::vector<NodeMoments> moments =
      compute_node_moments(panels, density, tree, taylor_order, executor);
  const double moment_ms =
      elapsed_ms(moment_begin, std::chrono::steady_clock::now());
  const auto interaction_begin = std::chrono::steady_clock::now();
  std::vector<std::size_t> rank(n, 0);
  for (std::size_t pos = 0; pos < tree.order.size(); ++pos)
    rank[tree.order[pos]] = pos;
  const double interaction_ms =
      elapsed_ms(interaction_begin, std::chrono::steady_clock::now());

  out.assign(n, 0.0);
  const auto traversal_begin = std::chrono::steady_clock::now();
  const std::size_t workers = effective_workers(executor, n);
  std::vector<std::size_t> local_near(workers, 0);
  std::vector<std::size_t> local_far(workers, 0);
  std::vector<std::size_t> local_mac_reject(workers, 0);
  std::vector<std::size_t> local_mac_convergence_rho_reject(workers, 0);
  std::vector<std::size_t> local_mac_convergence_aspect_reject(workers, 0);
  std::vector<std::size_t> local_mac_convergence_rho_panel(workers, 0);
  std::vector<std::size_t> local_mac_convergence_aspect_panel(workers, 0);
  std::vector<std::size_t> local_mac_surface_margin_reject(workers, 0);
  std::vector<std::size_t> local_mac_surface_margin_panel(workers, 0);
  std::vector<std::size_t> local_leaf_mac_candidates(workers, 0);
  std::vector<std::size_t> local_leaf_mac_accepted(workers, 0);
  std::vector<std::size_t> local_leaf_mac_rejected(workers, 0);
  std::vector<std::size_t> local_leaf_mac_far_panels(workers, 0);
  std::vector<std::size_t> local_leaf_mac_saved_panels(workers, 0);
  std::vector<MacAuditTally> local_mac_audit(workers);
  std::vector<FragmentationTally> local_fragmentation_audit(workers);
  executor.parallel_for(
      n, workers, 16,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::vector<std::size_t> stack;
        stack.reserve(64);
        std::size_t near_count = 0;
        std::size_t far_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const Panel &target = panels[i];
          const std::size_t target_rank = rank[i];
          double coupled = 0.0;
          stack.clear();
          stack.push_back(tree.root_index);
          while (!stack.empty()) {
            const std::size_t node_index = stack.back();
            stack.pop_back();
            const TreeNode &node = tree.nodes[node_index];
            const bool leaf = node.child_count == 0;
            const bool contains_target =
                target_rank >= node.first_panel &&
                target_rank < node.first_panel + node.panel_count;
            const bool leaf_mac_enabled = tree.leaf_mac != "off";
            const bool evaluate_node_mac =
                !contains_target && (!leaf || leaf_mac_enabled);
            if (evaluate_node_mac) {
              if (leaf)
                ++local_leaf_mac_candidates[worker];
              const double dx = target.x - node.cx;
              const double dy = target.y - node.cy;
              const double dz = target.z - node.cz;
              const double distance =
                  std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
              const MacDecision mac = evaluate_mac(target, node, tree, distance);
              if (leaf) {
                if (mac.accepted)
                  ++local_leaf_mac_accepted[worker];
                else
                  ++local_leaf_mac_rejected[worker];
              }
              if (mac.ratio_accept && tree.mac_safety != "off") {
                record_far_geometric_audit(local_mac_audit[worker], node,
                                           mac.gap, mac.gap_ratio,
                                           mac.unsafe);
                if (mac.physical_gap_limited)
                  ++local_mac_audit[worker].physical_gap_rejects;
              }
              if (mac.convergence_rho_limited) {
                ++local_mac_convergence_rho_reject[worker];
                local_mac_convergence_rho_panel[worker] += node.panel_count;
              }
              if (mac.convergence_aspect_limited) {
                ++local_mac_convergence_aspect_reject[worker];
                local_mac_convergence_aspect_panel[worker] += node.panel_count;
              }
              if (mac.surface_margin_limited) {
                ++local_mac_surface_margin_reject[worker];
                local_mac_surface_margin_panel[worker] += node.panel_count;
              }
              if (mac.accepted && (!leaf || tree.leaf_mac == "on")) {
                bool leaf_mac_fallback_to_direct = false;
                const bool fragmented_far =
                    tree.fragmentation_safety != "off" && node.fragmented;
                if (fragmented_far) {
                  const bool split =
                      tree.fragmentation_safety == "split";
                  record_fragmentation_audit(local_fragmentation_audit[worker],
                                             node, split);
                  if (split) {
                    if (leaf) {
                      leaf_mac_fallback_to_direct = true;
                    } else {
                      push_child_nodes(tree, node, stack);
                      continue;
                    }
                  }
                }
                if (!leaf_mac_fallback_to_direct) {
                coupled += node_kernel_taylor(target, node,
                                               moments[node_index], taylor_order,
                                               kappa, tree.derivative_mode);
                if (leaf) {
                  local_leaf_mac_far_panels[worker] += node.panel_count;
                  local_leaf_mac_saved_panels[worker] += node.panel_count;
                }
                ++far_count;
                continue;
                }
              }
              if (!leaf)
                ++local_mac_reject[worker];
            }
            if (leaf) {
              coupled += direct_ordered_sum(
                  target, i, panels, density, kappa, tree.order,
                  node.first_panel, node.panel_count, near_count);
            } else {
              for (std::size_t child = 0; child < node.child_count; ++child) {
                const std::size_t child_index =
                    child_node_index(tree, node, child);
                if (child_index < tree.nodes.size())
                  stack.push_back(child_index);
              }
            }
          }
          out[i] = panels[i].diagonal * density[i] + contrast * coupled;
        }
        local_near[worker] += near_count;
        local_far[worker] += far_count;
      });
  const double traversal_ms =
      elapsed_ms(traversal_begin, std::chrono::steady_clock::now());

  if (stats != nullptr) {
    const std::size_t near_total =
        std::accumulate(local_near.begin(), local_near.end(), std::size_t{0});
    const std::size_t far_total =
        std::accumulate(local_far.begin(), local_far.end(), std::size_t{0});
    const double interaction_total =
        static_cast<double>(near_total + far_total);
    MacAuditTally mac_audit_total;
    for (const MacAuditTally &worker_audit : local_mac_audit)
      merge_far_geometric_audit(mac_audit_total, worker_audit);
    FragmentationTally fragmentation_total;
    for (const FragmentationTally &worker_audit : local_fragmentation_audit)
      merge_fragmentation_audit(fragmentation_total, worker_audit);
    stats->matvec_mode = "treecode";
    stats->taylor_order = taylor_order;
    stats->derivative_mode = tree.derivative_mode;
    stats->tree_moment_ms += moment_ms;
    stats->interaction_build_ms += interaction_ms;
    stats->tree_traversal_ms += traversal_ms;
    stats->tree_eval_ms +=
        elapsed_ms(eval_begin, std::chrono::steady_clock::now());
    if (interaction_total > 0.0) {
      stats->near_eval_ms += traversal_ms *
                             static_cast<double>(near_total) /
                             interaction_total;
      stats->far_eval_ms += traversal_ms *
                            static_cast<double>(far_total) /
                            interaction_total;
    }
    stats->near_interactions += near_total;
    stats->far_interactions += far_total;
    stats->leaf_mac_candidates += std::accumulate(
        local_leaf_mac_candidates.begin(), local_leaf_mac_candidates.end(),
        std::size_t{0});
    stats->leaf_mac_accepted += std::accumulate(
        local_leaf_mac_accepted.begin(), local_leaf_mac_accepted.end(),
        std::size_t{0});
    stats->leaf_mac_rejected += std::accumulate(
        local_leaf_mac_rejected.begin(), local_leaf_mac_rejected.end(),
        std::size_t{0});
    stats->leaf_mac_far_panels += std::accumulate(
        local_leaf_mac_far_panels.begin(), local_leaf_mac_far_panels.end(),
        std::size_t{0});
    stats->leaf_mac_direct_panels_saved_estimate += std::accumulate(
        local_leaf_mac_saved_panels.begin(), local_leaf_mac_saved_panels.end(),
        std::size_t{0});
    stats->mac_reject_radius_limited += std::accumulate(
        local_mac_reject.begin(), local_mac_reject.end(), std::size_t{0});
    stats->mac_convergence_rho_rejects += std::accumulate(
        local_mac_convergence_rho_reject.begin(),
        local_mac_convergence_rho_reject.end(), std::size_t{0});
    stats->mac_convergence_aspect_rejects += std::accumulate(
        local_mac_convergence_aspect_reject.begin(),
        local_mac_convergence_aspect_reject.end(), std::size_t{0});
    stats->mac_convergence_rho_reject_panels += std::accumulate(
        local_mac_convergence_rho_panel.begin(),
        local_mac_convergence_rho_panel.end(), std::size_t{0});
    stats->mac_convergence_aspect_reject_panels += std::accumulate(
        local_mac_convergence_aspect_panel.begin(),
        local_mac_convergence_aspect_panel.end(), std::size_t{0});
    stats->mac_surface_margin_rejects += std::accumulate(
        local_mac_surface_margin_reject.begin(),
        local_mac_surface_margin_reject.end(), std::size_t{0});
    stats->mac_surface_margin_reject_panels += std::accumulate(
        local_mac_surface_margin_panel.begin(),
        local_mac_surface_margin_panel.end(), std::size_t{0});
    stats->tree_builder = tree.tree_builder;
    stats->expansion_center = tree.expansion_center;
    stats->fragmentation_safety = tree.fragmentation_safety;
    stats->mac_safety = tree.mac_safety;
    stats->mac_gap_factor = tree.mac_gap_factor;
    stats->moment_audit = tree.moment_audit;
    stats->moment_construction = tree.moment_construction;
    stats->moment_sum = tree.moment_sum;
    stats->convergence_audit = tree.convergence_audit;
    stats->convergence_safety = tree.convergence_safety;
    stats->convergence_rho_cap = tree.convergence_rho_cap;
    stats->mac_aspect_cap = tree.mac_aspect_cap;
    stats->convergence_estimator = tree.convergence_estimator;
    stats->convergence_tail_threshold = tree.convergence_tail_threshold;
    stats->convergence_estimator_error_cap =
        tree.convergence_estimator_error_cap;
    stats->convergence_unstable_rho_cap =
        tree.convergence_unstable_rho_cap;
    stats->convergence_calibration = tree.convergence_calibration;
    stats->convergence_dielectric_amp = tree.convergence_dielectric_amp;
    stats->convergence_estimator_model = tree.convergence_estimator_model;
    stats->convergence_channel_calibration =
        tree.convergence_channel_calibration;
    stats->convergence_split_policy = tree.convergence_split_policy;
    stats->channel_missed_semantics = tree.channel_missed_semantics;
    stats->accepted_miss_diagnostics = tree.accepted_miss_diagnostics;
    stats->accepted_miss_absolute_floor =
        tree.accepted_miss_absolute_floor;
    stats->convergence_k3_accepted_rho_cap =
        tree.convergence_k3_accepted_rho_cap;
    stats->mac_distance_policy = tree.mac_distance_policy;
    stats->mac_surface_margin_factor = tree.mac_surface_margin_factor;
    stats->expansion_audit = tree.expansion_audit;
    stats->expansion_audit_channel = tree.expansion_audit_channel;
    copy_far_geometric_audit_to_stats(*stats, mac_audit_total);
    copy_fragmentation_audit_to_stats(*stats, fragmentation_total);
    stats->tree_depth_max = tree.depth_max;
    stats->node_radius_max = tree.node_radius_max;
    stats->node_radius_mean = tree.node_radius_mean;
    stats->node_aspect_ratio_max = tree.node_aspect_ratio_max;
    stats->node_aspect_ratio_mean = tree.node_aspect_ratio_mean;
  }
}

void apply_jcp_treecode_operator_impl(const std::vector<Panel> &panels,
                                 const std::vector<double> &unknowns,
                                 double eps_ratio, double kappa,
                                 double lhs_phi, double lhs_phin,
                                 bool coupled, const TreeWorkspace &tree,
                                 const std::string &near_integration,
                                 const std::string &quad_buffer,
                                 TitanExecutor &executor,
                                 std::vector<double> &out,
                                 TreecodeStats *stats) {
  const std::size_t n = panels.size();
  if (n == 0) {
    out.clear();
    return;
  }
  if (tree.fmm_m2l == "single-matvec-analytic") {
    if (unknowns.size() < (coupled ? 2 * n : n)) {
      throw std::runtime_error(
          "PBE BIE: -pbe-bie-fmm-m2l=single-matvec-analytic received an incomplete JCP unknown vector.");
    }
    if (near_integration != "centroid" && near_integration != "off" &&
        near_integration != "analytic_singular" &&
        near_integration != "duffy_adaptive") {
      throw std::runtime_error(
          "PBE BIE: -pbe-bie-fmm-m2l=single-matvec-analytic supports centroid, analytic_singular, or duffy_adaptive near integration.");
    }
    FmmM2lSingleMatvecStats fmm_stats;
    if (tree.fmm_m2l_workspace == nullptr) {
      tree.fmm_m2l_workspace =
          std::make_shared<FmmM2lSingleMatvecWorkspace>();
    }
    TitanPBE::Tabi::apply_jcp_fmm_m2l_single_matvec(
        panels, unknowns, eps_ratio, kappa, lhs_phi, lhs_phin,
        tree.leaf_size, tree.opening_theta, tree.taylor_order, executor, out,
        &fmm_stats, coupled, &tree.fmm_m2l_context, near_integration,
        quad_buffer, tree.fmm_m2l_workspace.get());
    if (stats != nullptr) {
      stats->matvec_mode = "jcp2013_fmm_m2l_single_matvec_analytic";
      stats->taylor_order = tree.taylor_order;
      stats->derivative_mode = tree.derivative_mode;
      stats->near_integration = near_integration;
      stats->quad_buffer = quad_buffer;
      stats->fmm_m2l_mode = tree.fmm_m2l;
      stats->fmm_m2l_expansion_order = fmm_stats.expansion_order;
      stats->fmm_m2l_near_pairs = fmm_stats.near_pairs;
      stats->fmm_m2l_far_pairs = fmm_stats.far_pairs;
      stats->fmm_m2l_setup_ms = fmm_stats.setup_ms;
      stats->fmm_m2l_apply_ms = fmm_stats.apply_ms;
      stats->fmm_m2l_finite_outputs = fmm_stats.finite_outputs;
      stats->near_kernel_policy = fmm_stats.near_kernel_policy;
      stats->near_self_pairs += fmm_stats.near_self_pairs;
      stats->near_one_ring_pairs += fmm_stats.near_one_ring_pairs;
      stats->near_duffy_pairs += fmm_stats.near_duffy_pairs;
      stats->near_centroid_pairs += fmm_stats.near_centroid_pairs;
      stats->near_fallbacks += fmm_stats.near_fallbacks;
      stats->duffy_subdivisions_max =
          std::max(stats->duffy_subdivisions_max,
                   fmm_stats.duffy_subdivisions_max);
      if (fmm_stats.near_duffy_pairs > 0) {
        stats->duffy_subdivisions_mean =
            fmm_stats.duffy_subdivisions_mean;
      }
      stats->near_interactions += fmm_stats.near_pairs;
      stats->far_interactions += fmm_stats.far_pairs;
      stats->tree_traversal_ms += fmm_stats.apply_ms;
    }
    return;
  }
  if (tree.nodes.empty() || tree.order.size() != n ||
      unknowns.size() < (coupled ? 2 * n : n)) {
    out.assign(coupled ? 2 * n : n, 0.0);
    const std::vector<std::size_t> order = [&] {
      std::vector<std::size_t> identity(n);
      std::iota(identity.begin(), identity.end(), std::size_t{0});
      return identity;
    }();
    const std::size_t workers = effective_workers(executor, n);
    std::vector<std::size_t> local_near(workers, 0);
    std::vector<TreecodeStats> local_near_stats(workers);
    const auto near_begin = std::chrono::steady_clock::now();
    executor.parallel_for(
        n, workers, 8,
        [&](std::size_t worker, std::size_t begin, std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            TreecodeStats &worker_stats = local_near_stats[worker];
            const JcpNearAccum coupled_sum = accumulate_jcp_near_ordered(
                panels[i], i, panels, unknowns, eps_ratio, kappa, coupled,
                order, 0, n, near_integration, quad_buffer, &worker_stats);
            out[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) *
                         unknowns[i] -
                     coupled_sum.phi;
            if (coupled)
              out[n + i] =
                  jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
                      unknowns[n + i] -
                  coupled_sum.phin;
          }
          local_near[worker] += local_near_stats[worker].near_centroid_pairs +
                                local_near_stats[worker].near_duffy_pairs +
                                local_near_stats[worker]
                                    .near_geometric_buffer_pairs;
        });
    if (stats != nullptr) {
      stats->matvec_mode = "jcp2013_direct_fallback";
      stats->taylor_order = 0;
      stats->derivative_mode = "none";
      stats->leaf_near_policy = "exact";
      stats->leaf_panel_kernel = "scalar";
      stats->near_integration = near_integration;
      stats->quad_buffer = quad_buffer;
      const double near_ms =
          elapsed_ms(near_begin, std::chrono::steady_clock::now());
      stats->near_integration_ms += near_ms;
      stats->near_eval_ms += near_ms;
      std::size_t near_total = 0;
      for (const TreecodeStats &worker_stats : local_near_stats) {
        const std::size_t previous_duffy = stats->near_duffy_pairs;
        stats->near_kernel_policy = worker_stats.near_kernel_policy;
        stats->near_self_pairs += worker_stats.near_self_pairs;
        stats->near_one_ring_pairs += worker_stats.near_one_ring_pairs;
        stats->near_duffy_pairs += worker_stats.near_duffy_pairs;
        stats->near_centroid_pairs += worker_stats.near_centroid_pairs;
        stats->near_fallbacks += worker_stats.near_fallbacks;
        merge_near_kernel_block_stats(*stats, worker_stats);
        stats->duffy_subdivisions_max =
            std::max(stats->duffy_subdivisions_max,
                     worker_stats.duffy_subdivisions_max);
        if (worker_stats.near_duffy_pairs > 0) {
          const double old_count = static_cast<double>(previous_duffy);
          const double new_count = static_cast<double>(stats->near_duffy_pairs);
          stats->duffy_subdivisions_mean =
              new_count <= 0.0
                  ? 0.0
                  : (stats->duffy_subdivisions_mean * old_count +
                     worker_stats.duffy_subdivisions_mean *
                         static_cast<double>(worker_stats.near_duffy_pairs)) /
                        new_count;
        }
        near_total += worker_stats.near_centroid_pairs +
                      worker_stats.near_duffy_pairs +
                      worker_stats.near_geometric_buffer_pairs;
      }
      stats->near_interactions += near_total;
    }
    return;
  }

  const auto eval_begin = std::chrono::steady_clock::now();
  const int taylor_order = std::clamp(tree.taylor_order, 0, kTaylorOrderMax);
  TreePanelCache fallback_panel_cache;
  const TreePanelCache *panel_cache = &tree.panel_cache;
  if (!panel_cache->valid_for(n)) {
    fallback_panel_cache = build_tree_panel_cache(panels, tree.order);
    panel_cache = &fallback_panel_cache;
  }
  const TreeUnknownCache unknown_cache =
      build_tree_unknown_cache(panels, unknowns, coupled, tree.order);
  const std::string leaf_panel_kernel =
      resolve_leaf_panel_kernel(tree.leaf_panel_kernel);
  const std::string leaf_pair_batch_kernel =
      resolve_jcp_direct_batch_kernel(
          tree, resolve_leaf_pair_batch_kernel(tree.leaf_pair_kernel,
                                               tree.target_batch_size));
  const bool leaf_kernel_profile_enabled =
      tree.leaf_kernel_profile == "summary";
  const auto moment_begin = std::chrono::steady_clock::now();
  const PackedJcpMoments jcp_moments = compute_jcp_node_moments_packed(
      panels, unknowns, coupled, tree, taylor_order, executor);
  const MomentAuditSummary moment_audit_summary = audit_jcp_moment_translation(
      panels, unknowns, coupled, tree, taylor_order, jcp_moments);
  const double moment_ms =
      elapsed_ms(moment_begin, std::chrono::steady_clock::now());
  const JcpFarEvaluator far_evaluator =
      make_jcp_far_evaluator(taylor_order, coupled, tree.derivative_mode);

  if (tree.interaction_mode == "onthefly") {
    const auto interaction_begin = std::chrono::steady_clock::now();
    std::vector<std::size_t> rank(n, 0);
    for (std::size_t pos = 0; pos < tree.order.size(); ++pos)
      rank[tree.order[pos]] = pos;
    const double interaction_ms =
        elapsed_ms(interaction_begin, std::chrono::steady_clock::now());

    out.assign(coupled ? 2 * n : n, 0.0);
    const std::size_t workers = effective_workers(executor, n);
    std::vector<std::size_t> local_leaf_residual(workers, 0);
    std::vector<std::size_t> local_far(workers, 0);
    std::vector<std::size_t> local_mac_reject(workers, 0);
    std::vector<std::size_t> local_mac_convergence_rho_reject(workers, 0);
    std::vector<std::size_t> local_mac_convergence_aspect_reject(workers, 0);
    std::vector<std::size_t> local_mac_convergence_rho_panel(workers, 0);
    std::vector<std::size_t> local_mac_convergence_aspect_panel(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_reject(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_panel(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_applied(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_skipped(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_adaptive_reject(
        workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_adaptive_panel(
        workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_skipped_accepted(
        workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_hard_applied(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_soft_applied(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_hard_reject(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_hard_panel(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_soft_reject(workers, 0);
    std::vector<std::size_t> local_mac_surface_margin_soft_panel(workers, 0);
    std::vector<std::size_t> local_k3_rho_cap_reject(workers, 0);
    std::vector<std::size_t> local_k3_rho_cap_panel(workers, 0);
    std::vector<std::size_t> local_leaf_mac_candidates(workers, 0);
    std::vector<std::size_t> local_leaf_mac_accepted(workers, 0);
    std::vector<std::size_t> local_leaf_mac_rejected(workers, 0);
    std::vector<std::size_t> local_leaf_mac_far_panels(workers, 0);
    std::vector<std::size_t> local_leaf_mac_saved_panels(workers, 0);
    std::vector<MacAuditTally> local_mac_audit(workers);
    std::vector<FragmentationTally> local_fragmentation_audit(workers);
    std::vector<ExpansionAuditTally> local_expansion_audit(workers);
    std::vector<ConvergenceAuditTally> local_convergence_audit(workers);
    std::vector<ConvergenceEstimatorTally> local_convergence_estimator(workers);
    std::vector<LeafPairBatchTally> local_leaf_pair_batch(workers);
    std::vector<LeafWorkTally> local_leaf_work(workers);
    std::vector<TreecodeStats> local_near_stats(workers);
    std::vector<double> batched_sum_phi(n, 0.0);
    std::vector<double> batched_sum_phin(coupled ? n : 0, 0.0);
    std::atomic_size_t expansion_audit_candidates{0};
    const auto traversal_begin = std::chrono::steady_clock::now();
    executor.parallel_for(
        n, workers, 8,
        [&](std::size_t worker, std::size_t begin, std::size_t end) {
          TreecodeStats &worker_near_stats = local_near_stats[worker];
          std::vector<TraversalItem> stack;
          std::vector<std::size_t> batch_probe_stack;
          std::vector<std::size_t> singular_order;
          std::vector<std::size_t> leaf_order;
          std::vector<std::size_t> batch_targets;
          std::vector<JcpAccum> batch_accum;
          std::unordered_set<std::size_t> skipped_leaf_pairs;
          std::unordered_map<std::size_t, bool> leaf_pair_probe_cache;
          std::unordered_map<std::size_t, unsigned char>
              leaf_pair_directness_cache;
          std::unordered_map<std::size_t, DeferredLeafPairGroup>
              leaf_pair_worklist;
          std::vector<DeferredLeafPairGroup> leaf_block_groups;
          std::vector<std::size_t> worklist_batch_targets;
          std::vector<JcpAccum> worklist_batch_accum;
          const bool deferred_worklist_active =
              leaf_pair_batching_enabled(tree) &&
              tree.leaf_pair_scheduler == "deferred-worklist";
          const bool target_leaf_block_active =
              leaf_pair_batching_enabled(tree) &&
              tree.leaf_pair_scheduler == "target-leaf-block";
          const bool cost_tile_worklist_active =
              leaf_pair_batching_enabled(tree) &&
              tree.leaf_pair_scheduler == "cost-tile-worklist";
          const bool deferred_batch_active =
              deferred_worklist_active || target_leaf_block_active ||
              cost_tile_worklist_active;
          std::vector<double> worklist_sum_phi;
          std::vector<double> worklist_sum_phin;
          if (deferred_batch_active) {
            worklist_sum_phi.assign(end - begin, 0.0);
            if (coupled)
              worklist_sum_phin.assign(end - begin, 0.0);
            if (deferred_worklist_active)
              leaf_pair_worklist.reserve(256);
            if (target_leaf_block_active || cost_tile_worklist_active)
              leaf_block_groups.reserve(64);
          }
          stack.reserve(64);
          batch_probe_stack.reserve(64);
          singular_order.reserve(tree.leaf_size);
          leaf_order.reserve(tree.leaf_size);
          batch_targets.reserve(tree.target_batch_size);
          worklist_batch_targets.reserve(tree.target_batch_size);
          if (tree.leaf_pair_probe_cache == "per-matvec")
            leaf_pair_probe_cache.reserve(256);
          if (tree.leaf_pair_directness_cache == "target-leaf-source-leaf")
            leaf_pair_directness_cache.reserve(256);
          std::size_t leaf_count = 0;
          std::size_t far_count = 0;
          LeafWorkTally &leaf_work_tally = local_leaf_work[worker];
          auto full_directness_probe =
              [&](std::size_t candidate,
                  std::size_t source_node_index) noexcept {
            std::size_t probe_node_visits = 0;
            const auto probe_begin = std::chrono::steady_clock::now();
            const bool would_visit = would_visit_leaf_direct_for_target(
                panels, tree, rank, jcp_moments, candidate, source_node_index,
                taylor_order, coupled, eps_ratio, batch_probe_stack,
                &probe_node_visits);
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            batch_tally.selection_probe_ms +=
                elapsed_ms(probe_begin, std::chrono::steady_clock::now());
            batch_tally.selection_probe_node_visits += probe_node_visits;
            return would_visit;
          };
          auto ancestor_directness_probe =
              [&](std::size_t candidate,
                  std::size_t source_node_index) noexcept {
            std::size_t probe_node_visits = 0;
            const auto probe_begin = std::chrono::steady_clock::now();
            const bool would_visit =
                would_visit_leaf_direct_for_target_ancestor_path(
                    panels, tree, rank, jcp_moments, candidate,
                    source_node_index, taylor_order, coupled, eps_ratio,
                    &probe_node_visits);
            const double probe_ms =
                elapsed_ms(probe_begin, std::chrono::steady_clock::now());
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            batch_tally.selection_probe_ms += probe_ms;
            batch_tally.selection_probe_node_visits += probe_node_visits;
            batch_tally.directness_ancestor_probe_ms += probe_ms;
            ++batch_tally.directness_ancestor_probe_calls;
            batch_tally.directness_ancestor_probe_node_visits +=
                probe_node_visits;
            return would_visit;
          };
          auto directness_probe = [&](std::size_t candidate,
                                      std::size_t source_node_index) noexcept {
            ++local_leaf_pair_batch[worker].selection_probe_calls;
            if (tree.leaf_pair_directness_probe == "ancestor-path")
              return ancestor_directness_probe(candidate, source_node_index);
            return full_directness_probe(candidate, source_node_index);
          };
          auto verify_directness =
              [&](std::size_t candidate, std::size_t source_node_index,
                  bool candidate_verdict) noexcept {
            if (tree.leaf_pair_directness_verify == "off")
              return;
            if (tree.leaf_pair_directness_probe == "full-traversal" &&
                tree.leaf_pair_directness_cache !=
                    "target-leaf-source-leaf")
              return;
            const bool should_verify =
                tree.leaf_pair_directness_verify == "full-small" ||
                ((candidate + 17 * source_node_index) % 64 == 0);
            if (!should_verify)
              return;
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            ++batch_tally.directness_verify_checks;
            const bool reference =
                full_directness_probe(candidate, source_node_index);
            if (reference != candidate_verdict)
              ++batch_tally.directness_verify_mismatches;
          };
          auto add_worklist_accum =
              [&](std::size_t target_index, const JcpAccum &accum) noexcept {
            if (target_index < begin || target_index >= end)
              return;
            const std::size_t offset = target_index - begin;
            if (offset >= worklist_sum_phi.size())
              return;
            worklist_sum_phi[offset] += accum.phi;
            if (coupled && offset < worklist_sum_phin.size())
              worklist_sum_phin[offset] += accum.phin;
          };
          auto should_verify_batch =
              [&](std::size_t target_index,
                  std::size_t source_key,
                  const std::string &verify_mode) noexcept {
            if (verify_mode == "off")
              return false;
            if (verify_mode == "full-small")
              return n <= 4096;
            return ((target_index + 31 * source_key) % 64) == 0;
          };
          auto verify_group_accum =
              [&](std::size_t target_index, const TreeNode &source_node,
                  std::size_t source_key, const JcpAccum &candidate_accum,
                  bool leaf_block_mode) noexcept {
            const std::string &verify_mode =
                leaf_block_mode ? tree.leaf_block_verify
                                : tree.leaf_pair_worklist_verify;
            if (!should_verify_batch(target_index, source_key, verify_mode))
              return;
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            if (leaf_block_mode)
              ++batch_tally.leaf_block_verify_checks;
            else
              ++batch_tally.worklist_verify_checks;
            if (source_node.panel_count == 0 ||
                source_node.first_panel >= tree.order.size() ||
                source_node.panel_count >
                    tree.order.size() - source_node.first_panel) {
              if (leaf_block_mode)
                ++batch_tally.leaf_block_verify_mismatches;
              else
                ++batch_tally.worklist_verify_mismatches;
              return;
            }
            std::size_t direct_count = 0;
            const JcpAccum reference = direct_jcp_ordered_sum(
                panels[target_index], target_index, *panel_cache,
                unknown_cache, eps_ratio, kappa, coupled, tree.order,
                source_node.first_panel, source_node.panel_count,
                leaf_panel_kernel, direct_count);
            (void)direct_count;
            if (jcp_rel_error(candidate_accum, reference, coupled) >
                1.0e-10) {
              if (leaf_block_mode)
                ++batch_tally.leaf_block_verify_mismatches;
              else
                ++batch_tally.worklist_verify_mismatches;
            }
          };
          auto flush_worklist_group =
              [&](DeferredLeafPairGroup &group, bool force_scalar,
                  bool leaf_block_mode) noexcept {
            if (group.items.empty() ||
                group.source_node_index >= tree.nodes.size())
              return;
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            const auto worklist_begin = std::chrono::steady_clock::now();
            TreeNode source_node = tree.nodes[group.source_node_index];
            if (group.source_first_panel !=
                    std::numeric_limits<std::size_t>::max() &&
                group.source_panel_count > 0) {
              source_node.first_panel = group.source_first_panel;
              source_node.panel_count = group.source_panel_count;
            }
            if (!leaf_block_mode)
              ++batch_tally.worklist_groups;
            std::size_t item_offset = 0;
            const std::size_t max_batch =
                std::max<std::size_t>(1, tree.target_batch_size);
            while (!force_scalar &&
                   item_offset + max_batch <= group.items.size()) {
              worklist_batch_targets.clear();
              const std::size_t batch_item_offset = item_offset;
              for (std::size_t k = 0; k < max_batch; ++k) {
                worklist_batch_targets.push_back(
                    group.items[item_offset + k].target_index);
              }
              std::size_t batch_near_count = 0;
              std::size_t batch_vectorized_sources = 0;
              std::size_t batch_vector_lanes = 0;
              std::size_t batch_soa_sources = 0;
              const auto batch_begin = std::chrono::steady_clock::now();
              direct_jcp_target_batch_tile_sum(
                  panels, worklist_batch_targets, *panel_cache,
                  unknown_cache, eps_ratio, kappa, coupled, source_node,
                  leaf_pair_batch_kernel, worklist_batch_accum,
                  batch_near_count, batch_vectorized_sources,
                  batch_vector_lanes, true, batch_soa_sources);
              const double batch_ms =
                  elapsed_ms(batch_begin, std::chrono::steady_clock::now());
              for (std::size_t k = 0;
                   k < worklist_batch_targets.size() &&
                   k < worklist_batch_accum.size();
                   ++k) {
                add_worklist_accum(worklist_batch_targets[k],
                                   worklist_batch_accum[k]);
                verify_group_accum(worklist_batch_targets[k], source_node,
                                   group.source_first_panel,
                                   worklist_batch_accum[k],
                                   leaf_block_mode);
              }
              ++batch_tally.interactions;
              if (leaf_block_mode)
                ++batch_tally.leaf_block_batched_groups;
              else
                ++batch_tally.worklist_batched_groups;
              ++batch_tally.batch_tasks;
              batch_tally.targets += worklist_batch_targets.size();
              batch_tally.source_panels +=
                  source_node.panel_count * worklist_batch_targets.size();
              batch_tally.source_panel_evals +=
                  source_node.panel_count * worklist_batch_targets.size();
              batch_tally.vectorized_sources += batch_vectorized_sources;
              batch_tally.vector_lanes += batch_vector_lanes;
              batch_tally.soa_sources += batch_soa_sources;
              if (batch_soa_sources > 0)
                batch_tally.soa_ms += batch_ms;
              else if (leaf_pair_batch_kernel == "target-avx2" ||
                       leaf_pair_batch_kernel == "target-avx2-v2" ||
                       leaf_pair_batch_kernel == "target-avx2-4x4")
                batch_tally.target_avx2_ms += batch_ms;
              batch_tally.ms += batch_ms;
              leaf_count += batch_near_count;
              for (std::size_t k = 0; k < max_batch; ++k) {
                add_leaf_work(
                    leaf_work_tally,
                    group.items[batch_item_offset + k].leaf_reason,
                    source_node.panel_count);
              }
              item_offset += max_batch;
            }
            for (; item_offset < group.items.size(); ++item_offset) {
              const std::size_t target_index =
                  group.items[item_offset].target_index;
              if (target_index >= panels.size())
                continue;
              std::size_t direct_count = 0;
              const auto scalar_tail_begin =
                  leaf_kernel_profile_enabled
                      ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
              const JcpAccum near = direct_jcp_ordered_sum(
                  panels[target_index], target_index, *panel_cache,
                  unknown_cache, eps_ratio, kappa, coupled, tree.order,
                  source_node.first_panel, source_node.panel_count,
                  leaf_panel_kernel, direct_count);
              const double scalar_tail_ms =
                  leaf_kernel_profile_enabled
                      ? elapsed_ms(scalar_tail_begin,
                                   std::chrono::steady_clock::now())
                      : 0.0;
              add_worklist_accum(target_index, near);
              verify_group_accum(target_index, source_node,
                                 group.source_first_panel, near,
                                 leaf_block_mode);
              leaf_count += direct_count;
              if (leaf_block_mode)
                ++batch_tally.leaf_block_scalar_tail_tasks;
              else
                ++batch_tally.worklist_scalar_flush_tasks;
              if (leaf_kernel_profile_enabled) {
                ++batch_tally.scalar_tasks;
                batch_tally.scalar_ms += scalar_tail_ms;
                batch_tally.tail_ms += scalar_tail_ms;
                batch_tally.source_panel_evals += source_node.panel_count;
              }
              add_leaf_work(leaf_work_tally, LeafWorkReason::ScalarTail,
                            direct_count);
              ++batch_tally.fallbacks;
              batch_tally.residual_panels += source_node.panel_count;
            }
            const double elapsed =
                elapsed_ms(worklist_begin, std::chrono::steady_clock::now());
            if (leaf_block_mode)
              batch_tally.leaf_block_flush_ms += elapsed;
            else
              batch_tally.worklist_ms += elapsed;
            group.items.clear();
          };
          auto flush_leaf_block_groups = [&](bool force_scalar) noexcept {
            if (cost_tile_worklist_active) {
              std::sort(leaf_block_groups.begin(), leaf_block_groups.end(),
                        [](const DeferredLeafPairGroup &a,
                           const DeferredLeafPairGroup &b) {
                          const std::size_t a_cost =
                              a.items.size() *
                              std::max<std::size_t>(1, a.source_panel_count);
                          const std::size_t b_cost =
                              b.items.size() *
                              std::max<std::size_t>(1, b.source_panel_count);
                          if (a_cost != b_cost)
                            return a_cost > b_cost;
                          if (a.target_leaf_index != b.target_leaf_index)
                            return a.target_leaf_index < b.target_leaf_index;
                          if (a.source_node_index != b.source_node_index)
                            return a.source_node_index < b.source_node_index;
                          return a.source_first_panel < b.source_first_panel;
                        });
            }
            for (DeferredLeafPairGroup &group : leaf_block_groups)
              flush_worklist_group(group, force_scalar, true);
            leaf_block_groups.clear();
          };
          auto enqueue_worklist_task =
              [&](std::size_t target_index,
                  std::size_t source_node_index,
                  LeafWorkReason leaf_reason) noexcept {
            const auto worklist_begin = std::chrono::steady_clock::now();
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            const std::size_t target_leaf =
                target_index < tree.target_leaf_index_by_panel.size()
                    ? tree.target_leaf_index_by_panel[target_index]
                    : std::numeric_limits<std::size_t>::max();
            if (target_leaf >= tree.target_panels_by_leaf.size())
              return false;
            const std::size_t group_key =
                leaf_pair_skip_key(tree, target_leaf, source_node_index);
            DeferredLeafPairGroup &group = leaf_pair_worklist[group_key];
            if (group.items.empty()) {
              group.target_leaf_index = target_leaf;
              group.source_node_index = source_node_index;
              if (source_node_index < tree.nodes.size()) {
                const TreeNode &source_node = tree.nodes[source_node_index];
                group.source_first_panel = source_node.first_panel;
                group.source_panel_count = source_node.panel_count;
              }
            }
            group.items.push_back({target_index, leaf_reason});
            ++batch_tally.worklist_tasks;
            batch_tally.worklist_active_groups_max =
                std::max(batch_tally.worklist_active_groups_max,
                         leaf_pair_worklist.size());
            batch_tally.worklist_ms +=
                elapsed_ms(worklist_begin, std::chrono::steady_clock::now());
            const std::size_t max_batch =
                std::max<std::size_t>(1, tree.target_batch_size);
            if (group.items.size() >= max_batch) {
              flush_worklist_group(group, false, false);
              leaf_pair_worklist.erase(group_key);
            } else if (leaf_pair_worklist.size() >
                       tree.leaf_pair_worklist_max_active) {
              auto first = leaf_pair_worklist.begin();
              if (first != leaf_pair_worklist.end()) {
                ++batch_tally.worklist_forced_flushes;
                flush_worklist_group(first->second, true, false);
                leaf_pair_worklist.erase(first);
              }
            }
            return true;
          };
          auto enqueue_leaf_block_task =
              [&](std::size_t target_index,
                  std::size_t source_node_index,
                  LeafWorkReason leaf_reason) noexcept {
            const auto group_begin = std::chrono::steady_clock::now();
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            const std::size_t target_leaf =
                target_index < tree.target_leaf_index_by_panel.size()
                    ? tree.target_leaf_index_by_panel[target_index]
                    : std::numeric_limits<std::size_t>::max();
            if (target_leaf >= tree.target_panels_by_leaf.size())
              return false;
            DeferredLeafPairGroup *group = nullptr;
            for (DeferredLeafPairGroup &candidate_group :
                 leaf_block_groups) {
              if (candidate_group.source_node_index == source_node_index &&
                  candidate_group.target_leaf_index == target_leaf) {
                group = &candidate_group;
                break;
              }
            }
            if (group == nullptr) {
              leaf_block_groups.push_back({});
              group = &leaf_block_groups.back();
              group->target_leaf_index = target_leaf;
              group->source_node_index = source_node_index;
              if (source_node_index < tree.nodes.size()) {
                const TreeNode &source_node = tree.nodes[source_node_index];
                group->source_first_panel = source_node.first_panel;
                group->source_panel_count = source_node.panel_count;
              }
              ++batch_tally.leaf_block_source_groups;
            }
            group->items.push_back({target_index, leaf_reason});
            batch_tally.leaf_block_group_ms +=
                elapsed_ms(group_begin, std::chrono::steady_clock::now());
            const std::size_t max_batch =
                std::max<std::size_t>(1, tree.target_batch_size);
            if (group->items.size() >= max_batch) {
              flush_worklist_group(*group, false, true);
              leaf_block_groups.erase(std::remove_if(
                                          leaf_block_groups.begin(),
                                          leaf_block_groups.end(),
                                          [](const DeferredLeafPairGroup &g) {
                                            return g.items.empty();
                                          }),
                                      leaf_block_groups.end());
            }
            return true;
          };
          auto enqueue_cost_tile_task =
              [&](std::size_t target_index,
                  std::size_t source_node_index,
                  LeafWorkReason leaf_reason) noexcept {
            const auto group_begin = std::chrono::steady_clock::now();
            LeafPairBatchTally &batch_tally = local_leaf_pair_batch[worker];
            if (source_node_index >= tree.nodes.size())
              return false;
            const std::size_t target_leaf =
                target_index < tree.target_leaf_index_by_panel.size()
                    ? tree.target_leaf_index_by_panel[target_index]
                    : std::numeric_limits<std::size_t>::max();
            if (target_leaf >= tree.target_panels_by_leaf.size())
              return false;
            const TreeNode &source_node = tree.nodes[source_node_index];
            constexpr std::size_t kCostTileSourcePanels = 32;
            const std::size_t source_end =
                source_node.first_panel + source_node.panel_count;
            for (std::size_t first = source_node.first_panel;
                 first < source_end; first += kCostTileSourcePanels) {
              const std::size_t count =
                  std::min<std::size_t>(kCostTileSourcePanels,
                                        source_end - first);
              DeferredLeafPairGroup *group = nullptr;
              for (DeferredLeafPairGroup &candidate_group :
                   leaf_block_groups) {
                if (candidate_group.source_node_index == source_node_index &&
                    candidate_group.target_leaf_index == target_leaf &&
                    candidate_group.source_first_panel == first) {
                  group = &candidate_group;
                  break;
                }
              }
              if (group == nullptr) {
                leaf_block_groups.push_back({});
                group = &leaf_block_groups.back();
                group->target_leaf_index = target_leaf;
                group->source_node_index = source_node_index;
                group->source_first_panel = first;
                group->source_panel_count = count;
                ++batch_tally.leaf_block_source_groups;
              }
              group->items.push_back({target_index, leaf_reason});
            }
            batch_tally.leaf_block_group_ms +=
                elapsed_ms(group_begin, std::chrono::steady_clock::now());
            return true;
          };
          std::size_t active_leaf_block_target_leaf =
              std::numeric_limits<std::size_t>::max();
          for (std::size_t i = begin; i < end; ++i) {
            const Panel &target = panels[i];
            const std::size_t target_rank = rank[i];
            if (target_leaf_block_active || cost_tile_worklist_active) {
              const std::size_t target_leaf =
                  i < tree.target_leaf_index_by_panel.size()
                      ? tree.target_leaf_index_by_panel[i]
                      : std::numeric_limits<std::size_t>::max();
              if (target_leaf != active_leaf_block_target_leaf) {
                flush_leaf_block_groups(!cost_tile_worklist_active);
                active_leaf_block_target_leaf = target_leaf;
                ++local_leaf_pair_batch[worker].leaf_block_target_leaves;
              }
              ++local_leaf_pair_batch[worker].leaf_block_targets;
            }
            double sum_phi_i =
                deferred_batch_active ? 0.0 : batched_sum_phi[i];
            double sum_phin_i =
                deferred_batch_active
                    ? 0.0
                    : (coupled ? batched_sum_phin[i] : 0.0);
            stack.clear();
            stack.push_back(
                {tree.root_index, LeafWorkReason::MacTerminal});
            while (!stack.empty()) {
              const TraversalItem item = stack.back();
              stack.pop_back();
              const std::size_t node_index = item.node_index;
              const LeafWorkReason leaf_reason = item.leaf_reason;
              if (node_index >= tree.nodes.size())
                continue;
              const TreeNode &node = tree.nodes[node_index];
              const bool is_leaf = node.child_count == 0;
              const bool contains_target =
                  target_rank >= node.first_panel &&
                  target_rank < node.first_panel + node.panel_count;
              LeafWorkReason child_leaf_reason = leaf_reason;
              const bool leaf_mac_enabled = tree.leaf_mac != "off";
              const bool evaluate_node_mac =
                  !contains_target && (!is_leaf || leaf_mac_enabled);
              if (evaluate_node_mac) {
                if (is_leaf)
                  ++local_leaf_mac_candidates[worker];
                const double dx = target.x - node.cx;
                const double dy = target.y - node.cy;
                const double dz = target.z - node.cz;
                const double distance = std::sqrt(
                    std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
                const MacDecision mac =
                    evaluate_mac(target, node, tree, distance);
                if (is_leaf) {
                  if (mac.accepted)
                    ++local_leaf_mac_accepted[worker];
                  else
                    ++local_leaf_mac_rejected[worker];
                }
                if (mac.surface_margin_applied)
                  ++local_mac_surface_margin_applied[worker];
                if (mac.surface_margin_skipped)
                  ++local_mac_surface_margin_skipped[worker];
                if (mac.surface_margin_hard_applied)
                  ++local_mac_surface_margin_hard_applied[worker];
                if (mac.surface_margin_soft_applied)
                  ++local_mac_surface_margin_soft_applied[worker];
                if (mac.ratio_accept && tree.mac_safety != "off") {
                  record_far_geometric_audit(local_mac_audit[worker], node,
                                             mac.gap, mac.gap_ratio,
                                             mac.unsafe);
                  if (mac.physical_gap_limited)
                    ++local_mac_audit[worker].physical_gap_rejects;
                }
                if (mac.convergence_rho_limited) {
                  ++local_mac_convergence_rho_reject[worker];
                  local_mac_convergence_rho_panel[worker] += node.panel_count;
                }
                if (mac.convergence_aspect_limited) {
                  ++local_mac_convergence_aspect_reject[worker];
                  local_mac_convergence_aspect_panel[worker] +=
                      node.panel_count;
                }
                if (mac.surface_margin_limited) {
                  ++local_mac_surface_margin_reject[worker];
                  local_mac_surface_margin_panel[worker] += node.panel_count;
                }
                if (mac.surface_margin_hard_limited) {
                  ++local_mac_surface_margin_hard_reject[worker];
                  local_mac_surface_margin_hard_panel[worker] +=
                      node.panel_count;
                }
                if (mac.surface_margin_soft_limited) {
                  ++local_mac_surface_margin_soft_reject[worker];
                  local_mac_surface_margin_soft_panel[worker] +=
                      node.panel_count;
                }
                if (mac.surface_margin_adaptive_limited) {
                  ++local_mac_surface_margin_adaptive_reject[worker];
                  local_mac_surface_margin_adaptive_panel[worker] +=
                      node.panel_count;
                }
                if (mac.physical_gap_limited)
                  child_leaf_reason = LeafWorkReason::PhysicalGap;
                else if (mac.surface_margin_soft_limited)
                  child_leaf_reason = LeafWorkReason::SurfaceMarginSoft;
                else if (mac.surface_margin_limited)
                  child_leaf_reason = LeafWorkReason::SurfaceMarginHard;
                else if (mac.convergence_rho_limited)
                  child_leaf_reason = LeafWorkReason::ConvergenceRho;
                else if (mac.convergence_aspect_limited)
                  child_leaf_reason = LeafWorkReason::ConvergenceAspect;
                else
                  child_leaf_reason = LeafWorkReason::MacTerminal;
                if (mac.accepted && (!is_leaf || tree.leaf_mac == "on")) {
                  if (mac.surface_margin_skipped)
                    ++local_mac_surface_margin_skipped_accepted[worker];
                  bool leaf_mac_fallback_to_direct = false;
                  const bool fragmented_far =
                      tree.fragmentation_safety != "off" && node.fragmented;
                  if (fragmented_far) {
                    const bool split =
                        tree.fragmentation_safety == "split";
                    record_fragmentation_audit(
                        local_fragmentation_audit[worker], node, split);
                    if (split) {
                      if (is_leaf) {
                        child_leaf_reason = LeafWorkReason::Fragmentation;
                        leaf_mac_fallback_to_direct = true;
                      } else {
                        push_child_nodes(tree, node, stack,
                                         LeafWorkReason::Fragmentation);
                        continue;
                      }
                    }
                  }
                  const std::array<NodeMoments, 3> normal_moments = {
                      jcp_moments.phi_normal[0][node_index],
                      jcp_moments.phi_normal[1][node_index],
                      jcp_moments.phi_normal[2][node_index]};
                  ConvergenceEstimatorDecision estimator;
                  if (!leaf_mac_fallback_to_direct) {
                    if (tree.convergence_audit != "off") {
                      record_far_convergence_audit(
                          local_convergence_audit[worker], node, mac.rho);
                    }
                    estimator = evaluate_convergence_estimator(
                        tree, node, jcp_moments, node_index, taylor_order,
                        coupled, mac.rho, eps_ratio);
                    record_convergence_estimator(
                        local_convergence_estimator[worker], node, estimator);
                    if (estimator.split_reject) {
                      if (is_leaf) {
                        child_leaf_reason =
                            LeafWorkReason::ConvergenceEstimator;
                        leaf_mac_fallback_to_direct = true;
                      } else {
                        const bool split_channel_audit =
                            tree.channel_missed_semantics == "accepted-only" &&
                            tree.expansion_audit_channel != "off" &&
                            tree.expansion_audit != "off";
                        const bool split_overreject_audit =
                            tree.convergence_overreject_audit != "off";
                        const std::string &split_audit_mode =
                            tree.expansion_audit != "off"
                                ? tree.expansion_audit
                                : tree.convergence_overreject_audit;
                        if ((split_channel_audit || split_overreject_audit) &&
                            should_audit_far_expansion_mode(
                                tree, split_audit_mode, n, node,
                                expansion_audit_candidates)) {
                          std::size_t direct_panel_count = 0;
                          const JcpChannelAccum direct_channels =
                              direct_jcp_ordered_channel_sum_scalar(
                                  target, i, *panel_cache, unknown_cache,
                                  eps_ratio, kappa, coupled, tree.order,
                                  node.first_panel, node.panel_count,
                                  direct_panel_count);
                          (void)direct_panel_count;
                          const JcpChannelAccum far_channels =
                              taylor_jcp_node_channel_sum(
                                  target, node, jcp_moments.phin[node_index],
                                  normal_moments, eps_ratio, kappa, coupled,
                                  taylor_order, tree.derivative_mode);
                          JcpChannelAccum kernel_tail_p;
                          JcpChannelAccum kernel_tail_pm1;
                          if (tree.convergence_kernel_response_audit !=
                              "off") {
                            kernel_tail_p =
                                taylor_jcp_node_channel_degree_sum(
                                    target, node,
                                    jcp_moments.phin[node_index],
                                    normal_moments, eps_ratio, kappa, coupled,
                                    taylor_order, tree.derivative_mode,
                                    taylor_order);
                            kernel_tail_pm1 =
                                taylor_jcp_node_channel_degree_sum(
                                    target, node,
                                    jcp_moments.phin[node_index],
                                    normal_moments, eps_ratio, kappa, coupled,
                                    taylor_order, tree.derivative_mode,
                                    std::max(0, taylor_order - 1));
                          }
                          if (split_channel_audit) {
                            record_split_far_expansion_channel_audit(
                                local_expansion_audit[worker],
                                sum_jcp_channels(far_channels),
                                sum_jcp_channels(direct_channels),
                                far_channels, direct_channels, estimator,
                                coupled, tree.convergence_estimator_error_cap,
                                tree.accepted_miss_absolute_floor);
                          }
                          if (split_overreject_audit) {
                            record_convergence_overreject_audit(
                                local_expansion_audit[worker],
                                sum_jcp_channels(far_channels),
                                sum_jcp_channels(direct_channels),
                                far_channels, direct_channels, estimator, node,
                                coupled, tree.convergence_estimator_error_cap,
                                tree.accepted_miss_absolute_floor,
                                tree.convergence_overreject_detail,
                                tree.convergence_shape_audit,
                                tree.convergence_kernel_response_audit, i,
                                node_index, mac.rho, distance, taylor_order,
                                kernel_tail_p, kernel_tail_pm1);
                          }
                        }
                        push_child_nodes(
                            tree, node, stack,
                            LeafWorkReason::ConvergenceEstimator);
                        continue;
                      }
                    }
                  }
                  const bool k3_rho_cap_active =
                      tree.convergence_estimator_model == "channel-tail-eps" &&
                      tree.convergence_k3_accepted_rho_cap > 0.0;
                  if (!leaf_mac_fallback_to_direct && k3_rho_cap_active &&
                      mac.rho > tree.convergence_k3_accepted_rho_cap) {
                    ++local_k3_rho_cap_reject[worker];
                    local_k3_rho_cap_panel[worker] += node.panel_count;
                    if (is_leaf) {
                      child_leaf_reason = LeafWorkReason::K3RhoCap;
                      leaf_mac_fallback_to_direct = true;
                    } else {
                      push_child_nodes(tree, node, stack,
                                       LeafWorkReason::K3RhoCap);
                      continue;
                    }
                  }
                  if (!leaf_mac_fallback_to_direct) {
                    const JcpAccum far = far_evaluator(
                        target, node, jcp_moments.phin[node_index],
                        jcp_moments.phi_normal[0][node_index],
                        jcp_moments.phi_normal[1][node_index],
                        jcp_moments.phi_normal[2][node_index], eps_ratio,
                        kappa);
                    if (should_audit_far_expansion(
                            tree, n, node, expansion_audit_candidates)) {
                      std::size_t direct_panel_count = 0;
                      const auto direct_begin =
                          std::chrono::steady_clock::now();
                      JcpAccum direct;
                      JcpChannelAccum direct_channels;
                      JcpChannelAccum far_channels;
                      const bool channel_audit =
                          tree.expansion_audit_channel != "off";
                      if (channel_audit) {
                        direct_channels =
                            direct_jcp_ordered_channel_sum_scalar(
                                target, i, *panel_cache, unknown_cache,
                                eps_ratio, kappa, coupled, tree.order,
                                node.first_panel, node.panel_count,
                                direct_panel_count);
                        direct = sum_jcp_channels(direct_channels);
                        far_channels = taylor_jcp_node_channel_sum(
                            target, node, jcp_moments.phin[node_index],
                            normal_moments, eps_ratio, kappa, coupled,
                            taylor_order, tree.derivative_mode);
                      } else {
                        direct = direct_jcp_ordered_sum(
                            target, i, *panel_cache, unknown_cache, eps_ratio,
                            kappa, coupled, tree.order, node.first_panel,
                            node.panel_count, leaf_panel_kernel,
                            direct_panel_count);
                      }
                      const double direct_ms = elapsed_ms(
                          direct_begin, std::chrono::steady_clock::now());
                      record_far_expansion_audit(
                          local_expansion_audit[worker], far, direct, coupled,
                          node.panel_count, direct_ms, estimator.active,
                          estimator.unstable, estimator.split_reject,
                          tree.convergence_estimator_error_cap,
                          tree.accepted_miss_absolute_floor,
                          tree.channel_missed_semantics);
                      if (channel_audit) {
                        record_far_expansion_channel_audit(
                            local_expansion_audit[worker], far_channels,
                            direct_channels, estimator,
                            tree.convergence_estimator_error_cap,
                            tree.accepted_miss_absolute_floor,
                            tree.channel_missed_semantics,
                            tree.accepted_miss_diagnostics, i, node_index,
                            node.panel_count, mac.rho);
                      }
                    }
                    sum_phi_i += far.phi;
                    if (coupled)
                      sum_phin_i += far.phin;
                    if (is_leaf) {
                      local_leaf_mac_far_panels[worker] += node.panel_count;
                      local_leaf_mac_saved_panels[worker] += node.panel_count;
                    }
                    ++far_count;
                    continue;
                  }
                }
                if (!is_leaf)
                  ++local_mac_reject[worker];
              }
              if (is_leaf) {
                const std::size_t skip_key =
                    leaf_pair_skip_key(tree, i, node_index);
                if (skipped_leaf_pairs.find(skip_key) !=
                    skipped_leaf_pairs.end()) {
                  continue;
                }
                singular_order.clear();
                leaf_order.clear();
                for (std::size_t pos = node.first_panel;
                     pos < node.first_panel + node.panel_count; ++pos) {
                  const std::size_t source_index = tree.order[pos];
                  const Panel &source = panels[source_index];
                  if (curved_dual_cell(source) || is_singular_near_pair(target, i, source, source_index) ||
                      classify_geometric_near_buffer_pair(
                          target, source, tree.quad_buffer)
                          .active) {
                    singular_order.push_back(source_index);
                  } else {
                    leaf_order.push_back(source_index);
                  }
                }
                if (!singular_order.empty()) {
                  const JcpNearAccum near = accumulate_jcp_near_ordered(
                      target, i, panels, unknowns, eps_ratio, kappa, coupled,
                      singular_order, 0, singular_order.size(),
                      near_integration, quad_buffer, &worker_near_stats);
                  sum_phi_i += near.phi;
                  if (coupled)
                    sum_phin_i += near.phin;
                }
                if (!leaf_order.empty()) {
                  const bool batch_candidate =
                      leaf_pair_batching_enabled(tree) && !contains_target &&
                      singular_order.empty() &&
                      leaf_order.size() == node.panel_count &&
                      node.panel_count >= tree.leaf_pair_min_source_panels;
                  if (batch_candidate && deferred_worklist_active) {
                    if (enqueue_worklist_task(i, node_index, leaf_reason))
                      continue;
                  }
                  if (batch_candidate && target_leaf_block_active) {
                    if (enqueue_leaf_block_task(i, node_index, leaf_reason))
                      continue;
                  }
                  if (batch_candidate && cost_tile_worklist_active) {
                    if (enqueue_cost_tile_task(i, node_index, leaf_reason))
                      continue;
                  }
                  if (batch_candidate) {
                    LeafPairBatchTally &batch_tally =
                        local_leaf_pair_batch[worker];
                    const auto selection_begin =
                        std::chrono::steady_clock::now();
                    batch_targets.clear();
                    batch_targets.push_back(i);
                    const std::size_t max_batch =
                        std::max<std::size_t>(1, tree.target_batch_size);
                    const std::size_t lookahead =
                        std::max<std::size_t>(4, tree.leaf_pair_max_lookahead);
                    const std::size_t primary_target_leaf =
                        i < tree.target_leaf_index_by_panel.size()
                            ? tree.target_leaf_index_by_panel[i]
                            : std::numeric_limits<std::size_t>::max();
                    bool target_leaf_source_leaf_all_direct = false;
                    const bool can_use_directness_pair_cache =
                        tree.leaf_pair_directness_probe == "ancestor-path" &&
                        tree.leaf_pair_directness_cache ==
                            "target-leaf-source-leaf" &&
                        tree.leaf_pair_target_group == "same-target-leaf" &&
                        primary_target_leaf <
                            tree.target_panels_by_leaf.size();
                    if (can_use_directness_pair_cache) {
                      LeafPairBatchTally &batch_tally =
                          local_leaf_pair_batch[worker];
                      const std::size_t pair_key =
                          leaf_pair_skip_key(tree, primary_target_leaf,
                                             node_index);
                      const auto cached =
                          leaf_pair_directness_cache.find(pair_key);
                      if (cached != leaf_pair_directness_cache.end()) {
                        ++batch_tally.directness_pair_cache_hits;
                        target_leaf_source_leaf_all_direct =
                            cached->second == 1;
                        if (!target_leaf_source_leaf_all_direct)
                          ++batch_tally.directness_mixed_fallbacks;
                      } else {
                        ++batch_tally.directness_pair_cache_misses;
                        bool all_direct = true;
                        const std::vector<std::size_t> &same_leaf_targets =
                            tree.target_panels_by_leaf[primary_target_leaf];
                        for (std::size_t candidate : same_leaf_targets) {
                          if (!ancestor_directness_probe(candidate,
                                                         node_index)) {
                            all_direct = false;
                            break;
                          }
                        }
                        leaf_pair_directness_cache.emplace(
                            pair_key, all_direct ? 1 : 2);
                        target_leaf_source_leaf_all_direct = all_direct;
                        if (all_direct)
                          ++batch_tally.directness_all_direct_proofs;
                        else
                          ++batch_tally.directness_mixed_fallbacks;
                      }
                    }
                    auto consider_batch_candidate =
                        [&](std::size_t candidate) {
                      ++batch_tally.selection_candidate_scans;
                      if (tree.leaf_pair_target_group == "same-target-leaf") {
                        const std::size_t candidate_target_leaf =
                            candidate < tree.target_leaf_index_by_panel.size()
                                ? tree.target_leaf_index_by_panel[candidate]
                                : std::numeric_limits<std::size_t>::max();
                        if (candidate_target_leaf != primary_target_leaf ||
                            candidate_target_leaf ==
                                std::numeric_limits<std::size_t>::max()) {
                          ++batch_tally
                                .selection_same_target_leaf_rejects;
                          return;
                        }
                      }
                      if (!leaf_pair_batch_clean_for_target(
                              panels, tree, node, candidate)) {
                        ++batch_tally.selection_clean_rejects;
                        return;
                      }
                      bool would_visit = target_leaf_source_leaf_all_direct;
                      const std::size_t probe_key =
                          leaf_pair_skip_key(tree, candidate, node_index);
                      if (!would_visit &&
                          tree.leaf_pair_probe_cache == "per-matvec") {
                        const auto found =
                            leaf_pair_probe_cache.find(probe_key);
                        if (found != leaf_pair_probe_cache.end()) {
                          ++batch_tally.selection_probe_cache_hits;
                          would_visit = found->second;
                        } else {
                          ++batch_tally.selection_probe_cache_misses;
                          would_visit =
                              directness_probe(candidate, node_index);
                          leaf_pair_probe_cache.emplace(probe_key,
                                                        would_visit);
                        }
                      } else if (!would_visit) {
                        would_visit = directness_probe(candidate, node_index);
                      }
                      verify_directness(candidate, node_index, would_visit);
                      if (!would_visit) {
                        return;
                      }
                      batch_targets.push_back(candidate);
                    };
                    if (tree.leaf_pair_target_group == "same-target-leaf" &&
                        primary_target_leaf < tree.target_panels_by_leaf.size()) {
                      const std::vector<std::size_t> &same_leaf_targets =
                          tree.target_panels_by_leaf[primary_target_leaf];
                      auto it = std::upper_bound(same_leaf_targets.begin(),
                                                 same_leaf_targets.end(), i);
                      std::size_t scanned = 0;
                      while (it != same_leaf_targets.end() &&
                             batch_targets.size() < max_batch &&
                             scanned < lookahead) {
                        const std::size_t candidate = *it++;
                        ++scanned;
                        if (candidate >= begin && candidate < end)
                          consider_batch_candidate(candidate);
                      }
                    } else {
                      const std::size_t candidate_end =
                          std::min(end, i + 1 + lookahead);
                      for (std::size_t candidate = i + 1;
                           candidate < candidate_end &&
                           batch_targets.size() < max_batch;
                           ++candidate) {
                        consider_batch_candidate(candidate);
                      }
                    }
                    const std::size_t required_targets =
                        tree.cluster_policy == "leaf-pair-balanced"
                            ? std::min<std::size_t>(
                                  tree.leaf_pair_min_targets,
                                  std::max<std::size_t>(
                                      2, tree.target_batch_size))
                            : tree.leaf_pair_min_targets;
                    batch_tally.selection_ms += elapsed_ms(
                        selection_begin, std::chrono::steady_clock::now());
                    if (batch_targets.size() >= required_targets) {
                      std::size_t batch_near_count = 0;
                      std::size_t batch_vectorized_sources = 0;
                      std::size_t batch_vector_lanes = 0;
                      std::size_t batch_soa_sources = 0;
                      const auto batch_begin =
                          std::chrono::steady_clock::now();
                      direct_jcp_target_batch_tile_sum(
                          panels, batch_targets, *panel_cache, unknown_cache,
                          eps_ratio, kappa, coupled, node,
                          leaf_pair_batch_kernel, batch_accum,
                          batch_near_count, batch_vectorized_sources,
                          batch_vector_lanes, true, batch_soa_sources);
                      const double batch_ms = elapsed_ms(
                          batch_begin, std::chrono::steady_clock::now());
                      if (!batch_accum.empty()) {
                        sum_phi_i += batch_accum[0].phi;
                        if (coupled)
                          sum_phin_i += batch_accum[0].phin;
                      }
                      for (std::size_t t = 1;
                           t < batch_targets.size() &&
                           t < batch_accum.size();
                           ++t) {
                        const std::size_t candidate = batch_targets[t];
                        batched_sum_phi[candidate] += batch_accum[t].phi;
                        if (coupled)
                          batched_sum_phin[candidate] +=
                              batch_accum[t].phin;
                        skipped_leaf_pairs.insert(
                            leaf_pair_skip_key(tree, candidate, node_index));
                      }
                      ++batch_tally.interactions;
                      ++batch_tally.batch_tasks;
                      batch_tally.targets += batch_targets.size();
                      batch_tally.source_panels +=
                          node.panel_count * batch_targets.size();
                      batch_tally.source_panel_evals +=
                          node.panel_count * batch_targets.size();
                      batch_tally.vectorized_sources +=
                          batch_vectorized_sources;
                      batch_tally.vector_lanes += batch_vector_lanes;
                      batch_tally.soa_sources += batch_soa_sources;
                      if (batch_soa_sources > 0)
                        batch_tally.soa_ms += batch_ms;
                      else if (leaf_pair_batch_kernel == "target-avx2" ||
                               leaf_pair_batch_kernel == "target-avx2-v2" ||
                               leaf_pair_batch_kernel == "target-avx2-4x4")
                        batch_tally.target_avx2_ms += batch_ms;
                      batch_tally.ms += batch_ms;
                      leaf_count += batch_near_count;
                      for (std::size_t t = 0; t < batch_targets.size(); ++t)
                        add_leaf_work(leaf_work_tally, leaf_reason,
                                      node.panel_count);
                      continue;
                    }
                    ++batch_tally.selection_insufficient_targets;
                    ++local_leaf_pair_batch[worker].fallbacks;
                    if (batch_targets.size() > 1) {
                      local_leaf_pair_batch[worker].residual_panels +=
                          node.panel_count * batch_targets.size();
                    }
                  } else if (leaf_pair_batching_enabled(tree)) {
                    ++local_leaf_pair_batch[worker].fallbacks;
                    local_leaf_pair_batch[worker].residual_panels +=
                        leaf_order.size();
                  }
                  const auto scalar_leaf_begin =
                      leaf_kernel_profile_enabled
                          ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
                  const std::size_t leaf_count_before = leaf_count;
                  const JcpAccum near = direct_jcp_ordered_sum(
                      target, i, *panel_cache, unknown_cache, eps_ratio, kappa,
                      coupled, leaf_order, 0, leaf_order.size(),
                      leaf_panel_kernel, leaf_count);
                  const std::size_t leaf_delta =
                      leaf_count >= leaf_count_before
                          ? leaf_count - leaf_count_before
                          : 0;
                  add_leaf_work(
                      leaf_work_tally,
                      singular_order.empty()
                          ? leaf_reason
                          : LeafWorkReason::SingularFallback,
                      leaf_delta);
                  if (leaf_pair_batching_enabled(tree) &&
                      leaf_kernel_profile_enabled) {
                    LeafPairBatchTally &batch_tally =
                        local_leaf_pair_batch[worker];
                    const double scalar_ms = elapsed_ms(
                        scalar_leaf_begin, std::chrono::steady_clock::now());
                    ++batch_tally.scalar_tasks;
                    batch_tally.scalar_ms += scalar_ms;
                    batch_tally.source_panel_evals += leaf_order.size();
                  }
                  sum_phi_i += near.phi;
                  if (coupled)
                    sum_phin_i += near.phin;
                }
              } else {
                push_child_nodes(tree, node, stack, child_leaf_reason);
              }
            }
            if (deferred_batch_active) {
              const std::size_t offset = i - begin;
              if (offset < worklist_sum_phi.size())
                worklist_sum_phi[offset] += sum_phi_i;
              if (coupled && offset < worklist_sum_phin.size())
                worklist_sum_phin[offset] += sum_phin_i;
            } else {
              out[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) *
                           unknowns[i] -
                       sum_phi_i;
              if (coupled)
                out[n + i] =
                    jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
                        unknowns[n + i] -
                    sum_phin_i;
            }
          }
          if (deferred_batch_active) {
            for (auto &entry : leaf_pair_worklist) {
              flush_worklist_group(entry.second, true, false);
            }
            leaf_pair_worklist.clear();
            if (target_leaf_block_active || cost_tile_worklist_active)
              flush_leaf_block_groups(!cost_tile_worklist_active);
            const auto writeback_begin = std::chrono::steady_clock::now();
            for (std::size_t i = begin; i < end; ++i) {
              const std::size_t offset = i - begin;
              const double sum_phi_i =
                  offset < worklist_sum_phi.size() ? worklist_sum_phi[offset]
                                                   : 0.0;
              const double sum_phin_i =
                  coupled && offset < worklist_sum_phin.size()
                      ? worklist_sum_phin[offset]
                      : 0.0;
              out[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) *
                           unknowns[i] -
                       sum_phi_i;
              if (coupled)
                out[n + i] =
                    jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
                        unknowns[n + i] -
                    sum_phin_i;
            }
            if (target_leaf_block_active || cost_tile_worklist_active) {
              local_leaf_pair_batch[worker].leaf_block_writeback_ms +=
                  elapsed_ms(writeback_begin,
                             std::chrono::steady_clock::now());
            }
          }
          local_leaf_residual[worker] += leaf_count;
          local_far[worker] += far_count;
        });
    const double traversal_ms =
        elapsed_ms(traversal_begin, std::chrono::steady_clock::now());

    if (stats != nullptr) {
      std::size_t singular_evaluated = 0;
      for (const TreecodeStats &worker_stats : local_near_stats)
        singular_evaluated += worker_stats.near_centroid_pairs +
                               worker_stats.near_duffy_pairs +
                               worker_stats.near_geometric_buffer_pairs;
      const std::size_t leaf_total =
          std::accumulate(local_leaf_residual.begin(),
                          local_leaf_residual.end(), std::size_t{0});
      const std::size_t far_total =
          std::accumulate(local_far.begin(), local_far.end(), std::size_t{0});
      MacAuditTally mac_audit_total;
      for (const MacAuditTally &worker_audit : local_mac_audit)
        merge_far_geometric_audit(mac_audit_total, worker_audit);
      FragmentationTally fragmentation_total;
      for (const FragmentationTally &worker_audit : local_fragmentation_audit)
        merge_fragmentation_audit(fragmentation_total, worker_audit);
      ExpansionAuditTally expansion_total;
      for (const ExpansionAuditTally &worker_audit : local_expansion_audit)
        merge_far_expansion_audit(expansion_total, worker_audit);
      ConvergenceAuditTally convergence_total;
      for (const ConvergenceAuditTally &worker_audit :
           local_convergence_audit)
        merge_far_convergence_audit(convergence_total, worker_audit);
      ConvergenceEstimatorTally estimator_total;
      for (const ConvergenceEstimatorTally &worker_audit :
           local_convergence_estimator)
        merge_convergence_estimator(estimator_total, worker_audit);
      LeafWorkTally leaf_work_total;
      for (const LeafWorkTally &worker_leaf_work : local_leaf_work) {
        leaf_work_total.mac_terminal_panels +=
            worker_leaf_work.mac_terminal_panels;
        leaf_work_total.physical_gap_panels +=
            worker_leaf_work.physical_gap_panels;
        leaf_work_total.fragmentation_panels +=
            worker_leaf_work.fragmentation_panels;
        leaf_work_total.convergence_panels +=
            worker_leaf_work.convergence_panels;
        leaf_work_total.convergence_estimator_panels +=
            worker_leaf_work.convergence_estimator_panels;
        leaf_work_total.convergence_rho_panels +=
            worker_leaf_work.convergence_rho_panels;
        leaf_work_total.convergence_aspect_panels +=
            worker_leaf_work.convergence_aspect_panels;
        leaf_work_total.surface_margin_panels +=
            worker_leaf_work.surface_margin_panels;
        leaf_work_total.surface_margin_hard_panels +=
            worker_leaf_work.surface_margin_hard_panels;
        leaf_work_total.surface_margin_soft_panels +=
            worker_leaf_work.surface_margin_soft_panels;
        leaf_work_total.k3_rho_cap_panels +=
            worker_leaf_work.k3_rho_cap_panels;
        leaf_work_total.singular_fallback_panels +=
            worker_leaf_work.singular_fallback_panels;
        leaf_work_total.scalar_tail_panels +=
            worker_leaf_work.scalar_tail_panels;
      }
      LeafPairBatchTally leaf_pair_batch_total;
      for (const LeafPairBatchTally &worker_batch : local_leaf_pair_batch) {
        leaf_pair_batch_total.interactions += worker_batch.interactions;
        leaf_pair_batch_total.targets += worker_batch.targets;
        leaf_pair_batch_total.source_panels += worker_batch.source_panels;
        leaf_pair_batch_total.residual_panels += worker_batch.residual_panels;
        leaf_pair_batch_total.fallbacks += worker_batch.fallbacks;
        leaf_pair_batch_total.vectorized_sources +=
            worker_batch.vectorized_sources;
        leaf_pair_batch_total.vector_lanes += worker_batch.vector_lanes;
        leaf_pair_batch_total.soa_sources += worker_batch.soa_sources;
        leaf_pair_batch_total.scalar_tasks += worker_batch.scalar_tasks;
        leaf_pair_batch_total.batch_tasks += worker_batch.batch_tasks;
        leaf_pair_batch_total.source_panel_evals +=
            worker_batch.source_panel_evals;
        leaf_pair_batch_total.selection_candidate_scans +=
            worker_batch.selection_candidate_scans;
        leaf_pair_batch_total.selection_clean_rejects +=
            worker_batch.selection_clean_rejects;
        leaf_pair_batch_total.selection_probe_calls +=
            worker_batch.selection_probe_calls;
        leaf_pair_batch_total.selection_probe_node_visits +=
            worker_batch.selection_probe_node_visits;
        leaf_pair_batch_total.selection_probe_cache_hits +=
            worker_batch.selection_probe_cache_hits;
        leaf_pair_batch_total.selection_probe_cache_misses +=
            worker_batch.selection_probe_cache_misses;
        leaf_pair_batch_total.selection_insufficient_targets +=
            worker_batch.selection_insufficient_targets;
        leaf_pair_batch_total.selection_same_target_leaf_rejects +=
            worker_batch.selection_same_target_leaf_rejects;
        leaf_pair_batch_total.directness_ancestor_probe_calls +=
            worker_batch.directness_ancestor_probe_calls;
        leaf_pair_batch_total.directness_ancestor_probe_node_visits +=
            worker_batch.directness_ancestor_probe_node_visits;
        leaf_pair_batch_total.directness_pair_cache_hits +=
            worker_batch.directness_pair_cache_hits;
        leaf_pair_batch_total.directness_pair_cache_misses +=
            worker_batch.directness_pair_cache_misses;
        leaf_pair_batch_total.directness_all_direct_proofs +=
            worker_batch.directness_all_direct_proofs;
        leaf_pair_batch_total.directness_mixed_fallbacks +=
            worker_batch.directness_mixed_fallbacks;
        leaf_pair_batch_total.directness_verify_checks +=
            worker_batch.directness_verify_checks;
        leaf_pair_batch_total.directness_verify_mismatches +=
            worker_batch.directness_verify_mismatches;
        leaf_pair_batch_total.worklist_tasks += worker_batch.worklist_tasks;
        leaf_pair_batch_total.worklist_groups += worker_batch.worklist_groups;
        leaf_pair_batch_total.worklist_batched_groups +=
            worker_batch.worklist_batched_groups;
        leaf_pair_batch_total.worklist_scalar_flush_tasks +=
            worker_batch.worklist_scalar_flush_tasks;
        leaf_pair_batch_total.worklist_forced_flushes +=
            worker_batch.worklist_forced_flushes;
        leaf_pair_batch_total.worklist_active_groups_max =
            std::max(leaf_pair_batch_total.worklist_active_groups_max,
                     worker_batch.worklist_active_groups_max);
        leaf_pair_batch_total.worklist_verify_checks +=
            worker_batch.worklist_verify_checks;
        leaf_pair_batch_total.worklist_verify_mismatches +=
            worker_batch.worklist_verify_mismatches;
        leaf_pair_batch_total.leaf_block_target_leaves +=
            worker_batch.leaf_block_target_leaves;
        leaf_pair_batch_total.leaf_block_targets +=
            worker_batch.leaf_block_targets;
        leaf_pair_batch_total.leaf_block_source_groups +=
            worker_batch.leaf_block_source_groups;
        leaf_pair_batch_total.leaf_block_batched_groups +=
            worker_batch.leaf_block_batched_groups;
        leaf_pair_batch_total.leaf_block_scalar_tail_tasks +=
            worker_batch.leaf_block_scalar_tail_tasks;
        leaf_pair_batch_total.leaf_block_verify_checks +=
            worker_batch.leaf_block_verify_checks;
        leaf_pair_batch_total.leaf_block_verify_mismatches +=
            worker_batch.leaf_block_verify_mismatches;
        leaf_pair_batch_total.directness_ancestor_probe_ms +=
            worker_batch.directness_ancestor_probe_ms;
        leaf_pair_batch_total.selection_probe_ms +=
            worker_batch.selection_probe_ms;
        leaf_pair_batch_total.selection_ms += worker_batch.selection_ms;
        leaf_pair_batch_total.worklist_ms += worker_batch.worklist_ms;
        leaf_pair_batch_total.leaf_block_group_ms +=
            worker_batch.leaf_block_group_ms;
        leaf_pair_batch_total.leaf_block_flush_ms +=
            worker_batch.leaf_block_flush_ms;
        leaf_pair_batch_total.leaf_block_writeback_ms +=
            worker_batch.leaf_block_writeback_ms;
        leaf_pair_batch_total.target_avx2_ms += worker_batch.target_avx2_ms;
        leaf_pair_batch_total.soa_ms += worker_batch.soa_ms;
        leaf_pair_batch_total.scalar_ms += worker_batch.scalar_ms;
        leaf_pair_batch_total.tail_ms += worker_batch.tail_ms;
        leaf_pair_batch_total.ms += worker_batch.ms;
      }
      stats->matvec_mode = "jcp2013_treecode";
      stats->interaction_mode = "onthefly";
      stats->taylor_order = taylor_order;
      stats->derivative_mode = tree.derivative_mode;
      stats->leaf_near_policy = tree.leaf_near_policy;
      stats->leaf_panel_kernel = leaf_panel_kernel;
      stats->leaf_tile_status = "off";
      stats->cluster_policy = tree.cluster_policy;
      stats->target_batch_size = static_cast<int>(tree.target_batch_size);
      stats->leaf_pair_min_source_panels =
          static_cast<int>(tree.leaf_pair_min_source_panels);
      stats->leaf_pair_min_targets =
          static_cast<int>(tree.leaf_pair_min_targets);
      stats->leaf_pair_batch_kernel = leaf_pair_batch_kernel;
      stats->leaf_kernel_profile = tree.leaf_kernel_profile;
      stats->leaf_work_profile = tree.leaf_work_profile;
      stats->jcp_direct_kernel = tree.jcp_direct_kernel;
      stats->leaf_pair_selection_profile = tree.leaf_pair_selection_profile;
      stats->leaf_pair_probe_cache = tree.leaf_pair_probe_cache;
      stats->leaf_pair_target_group = tree.leaf_pair_target_group;
      stats->leaf_pair_directness_probe = tree.leaf_pair_directness_probe;
      stats->leaf_pair_directness_cache = tree.leaf_pair_directness_cache;
      stats->leaf_pair_directness_verify = tree.leaf_pair_directness_verify;
      stats->leaf_pair_max_lookahead =
          static_cast<int>(tree.leaf_pair_max_lookahead);
      stats->near_integration = near_integration;
      stats->quad_buffer = quad_buffer;
      stats->tree_moment_ms += moment_ms;
      if (stats->matvec_calls == 0)
        stats->interaction_build_ms += interaction_ms;
      stats->tree_traversal_ms += traversal_ms;
      stats->tree_eval_ms +=
          elapsed_ms(eval_begin, std::chrono::steady_clock::now());
      stats->leaf_panel_eval_ms += traversal_ms;
      stats->leaf_residual_eval_ms += traversal_ms;
      stats->leaf_near_eval_ms += traversal_ms;
      stats->near_eval_ms += traversal_ms;
      for (const TreecodeStats &worker_stats : local_near_stats) {
        const std::size_t previous_duffy = stats->near_duffy_pairs;
        stats->near_kernel_policy = worker_stats.near_kernel_policy;
        stats->near_self_pairs += worker_stats.near_self_pairs;
        stats->near_one_ring_pairs += worker_stats.near_one_ring_pairs;
        stats->near_duffy_pairs += worker_stats.near_duffy_pairs;
        stats->near_centroid_pairs += worker_stats.near_centroid_pairs;
        stats->near_fallbacks += worker_stats.near_fallbacks;
        merge_near_kernel_block_stats(*stats, worker_stats);
        stats->duffy_subdivisions_max =
            std::max(stats->duffy_subdivisions_max,
                     worker_stats.duffy_subdivisions_max);
        if (worker_stats.near_duffy_pairs > 0) {
          const double old_count = static_cast<double>(previous_duffy);
          const double new_count =
              static_cast<double>(stats->near_duffy_pairs);
          stats->duffy_subdivisions_mean =
              new_count <= 0.0
                  ? 0.0
                  : (stats->duffy_subdivisions_mean * old_count +
                     worker_stats.duffy_subdivisions_mean *
                         static_cast<double>(worker_stats.near_duffy_pairs)) /
                        new_count;
        }
      }
      stats->near_centroid_pairs += leaf_total;
      stats->singular_near_interactions += singular_evaluated;
      stats->leaf_near_interactions += leaf_total;
      stats->leaf_panel_interactions += leaf_total;
      stats->leaf_residual_panel_interactions += leaf_total;
      stats->leaf_mac_candidates += std::accumulate(
          local_leaf_mac_candidates.begin(), local_leaf_mac_candidates.end(),
          std::size_t{0});
      stats->leaf_mac_accepted += std::accumulate(
          local_leaf_mac_accepted.begin(), local_leaf_mac_accepted.end(),
          std::size_t{0});
      stats->leaf_mac_rejected += std::accumulate(
          local_leaf_mac_rejected.begin(), local_leaf_mac_rejected.end(),
          std::size_t{0});
      stats->leaf_mac_far_panels += std::accumulate(
          local_leaf_mac_far_panels.begin(), local_leaf_mac_far_panels.end(),
          std::size_t{0});
      stats->leaf_mac_direct_panels_saved_estimate += std::accumulate(
          local_leaf_mac_saved_panels.begin(), local_leaf_mac_saved_panels.end(),
          std::size_t{0});
      stats->leaf_work_mac_terminal_panels +=
          leaf_work_total.mac_terminal_panels;
      stats->leaf_work_physical_gap_panels +=
          leaf_work_total.physical_gap_panels;
      stats->leaf_work_fragmentation_panels +=
          leaf_work_total.fragmentation_panels;
      stats->leaf_work_convergence_panels +=
          leaf_work_total.convergence_panels;
      stats->leaf_work_convergence_estimator_panels +=
          leaf_work_total.convergence_estimator_panels;
      stats->leaf_work_convergence_rho_panels +=
          leaf_work_total.convergence_rho_panels;
      stats->leaf_work_convergence_aspect_panels +=
          leaf_work_total.convergence_aspect_panels;
      stats->leaf_work_surface_margin_panels +=
          leaf_work_total.surface_margin_panels;
      stats->leaf_work_surface_margin_hard_panels +=
          leaf_work_total.surface_margin_hard_panels;
      stats->leaf_work_surface_margin_soft_panels +=
          leaf_work_total.surface_margin_soft_panels;
      stats->leaf_work_k3_rho_cap_panels +=
          leaf_work_total.k3_rho_cap_panels;
      stats->leaf_work_singular_fallback_panels +=
          leaf_work_total.singular_fallback_panels;
      stats->leaf_work_scalar_tail_panels +=
          leaf_work_total.scalar_tail_panels;
      stats->leaf_work_profile_accounted_panels +=
          leaf_work_tally_total(leaf_work_total);
      if (stats->leaf_work_profile_accounted_panels > 0) {
        const double accounted =
            static_cast<double>(stats->leaf_work_profile_accounted_panels);
        stats->leaf_work_mac_terminal_frac =
            static_cast<double>(stats->leaf_work_mac_terminal_panels) /
            accounted;
        stats->leaf_work_convergence_frac =
            static_cast<double>(stats->leaf_work_convergence_panels) /
            accounted;
        stats->leaf_work_surface_margin_frac =
            static_cast<double>(stats->leaf_work_surface_margin_panels) /
            accounted;
        stats->leaf_work_singular_fallback_frac =
            static_cast<double>(stats->leaf_work_singular_fallback_panels) /
            accounted;
        stats->leaf_work_scalar_tail_frac =
            static_cast<double>(stats->leaf_work_scalar_tail_panels) /
            accounted;
      }
      stats->leaf_pair_batch_interactions += leaf_pair_batch_total.interactions;
      stats->leaf_pair_batch_targets += leaf_pair_batch_total.targets;
      stats->leaf_pair_batch_source_panels +=
          leaf_pair_batch_total.source_panels;
      stats->leaf_pair_batch_residual_panels +=
          leaf_pair_batch_total.residual_panels;
      stats->leaf_pair_batch_fallbacks += leaf_pair_batch_total.fallbacks;
      stats->leaf_pair_batch_ms += leaf_pair_batch_total.ms;
      stats->leaf_pair_batch_vectorized_sources +=
          leaf_pair_batch_total.vectorized_sources;
      stats->leaf_pair_batch_vector_lanes +=
          leaf_pair_batch_total.vector_lanes;
      stats->leaf_pair_batch_pairs_per_second =
          stats->leaf_pair_batch_ms <= 0.0
              ? 0.0
              : static_cast<double>(stats->leaf_pair_batch_source_panels) /
                    (stats->leaf_pair_batch_ms / 1000.0);
      stats->leaf_kernel_scalar_tasks += leaf_pair_batch_total.scalar_tasks;
      stats->leaf_kernel_batch_tasks += leaf_pair_batch_total.batch_tasks;
      stats->leaf_kernel_source_panel_evals +=
          leaf_pair_batch_total.source_panel_evals;
      stats->leaf_kernel_scalar_ms += leaf_pair_batch_total.scalar_ms;
      stats->leaf_kernel_target_avx2_ms +=
          leaf_pair_batch_total.target_avx2_ms;
      stats->leaf_kernel_tail_ms += leaf_pair_batch_total.tail_ms;
      stats->leaf_pair_batch_soa_sources += leaf_pair_batch_total.soa_sources;
      stats->leaf_pair_batch_soa_ms += leaf_pair_batch_total.soa_ms;
      stats->leaf_pair_batch_scalar_tail_ms += leaf_pair_batch_total.tail_ms;
      stats->leaf_pair_batch_clean_source_panels_per_second =
          stats->leaf_pair_batch_ms <= 0.0
              ? 0.0
              : static_cast<double>(stats->leaf_pair_batch_source_panels) /
                    (stats->leaf_pair_batch_ms / 1000.0);
      stats->leaf_pair_selection_candidate_scans +=
          leaf_pair_batch_total.selection_candidate_scans;
      stats->leaf_pair_selection_clean_rejects +=
          leaf_pair_batch_total.selection_clean_rejects;
      stats->leaf_pair_selection_probe_calls +=
          leaf_pair_batch_total.selection_probe_calls;
      stats->leaf_pair_selection_probe_node_visits +=
          leaf_pair_batch_total.selection_probe_node_visits;
      stats->leaf_pair_selection_probe_cache_hits +=
          leaf_pair_batch_total.selection_probe_cache_hits;
      stats->leaf_pair_selection_probe_cache_misses +=
          leaf_pair_batch_total.selection_probe_cache_misses;
      stats->leaf_pair_selection_insufficient_targets +=
          leaf_pair_batch_total.selection_insufficient_targets;
      stats->leaf_pair_selection_same_target_leaf_rejects +=
          leaf_pair_batch_total.selection_same_target_leaf_rejects;
      stats->leaf_pair_selection_probe_ms +=
          leaf_pair_batch_total.selection_probe_ms;
      stats->leaf_pair_selection_ms += leaf_pair_batch_total.selection_ms;
      stats->leaf_pair_directness_ancestor_probe_calls +=
          leaf_pair_batch_total.directness_ancestor_probe_calls;
      stats->leaf_pair_directness_ancestor_probe_node_visits +=
          leaf_pair_batch_total.directness_ancestor_probe_node_visits;
      stats->leaf_pair_directness_ancestor_probe_ms +=
          leaf_pair_batch_total.directness_ancestor_probe_ms;
      stats->leaf_pair_directness_pair_cache_hits +=
          leaf_pair_batch_total.directness_pair_cache_hits;
      stats->leaf_pair_directness_pair_cache_misses +=
          leaf_pair_batch_total.directness_pair_cache_misses;
      stats->leaf_pair_directness_all_direct_proofs +=
          leaf_pair_batch_total.directness_all_direct_proofs;
      stats->leaf_pair_directness_mixed_fallbacks +=
          leaf_pair_batch_total.directness_mixed_fallbacks;
      stats->leaf_pair_directness_verify_checks +=
          leaf_pair_batch_total.directness_verify_checks;
      stats->leaf_pair_directness_verify_mismatches +=
          leaf_pair_batch_total.directness_verify_mismatches;
      stats->leaf_pair_scheduler = tree.leaf_pair_scheduler;
      stats->leaf_pair_worklist_max_active =
          static_cast<int>(tree.leaf_pair_worklist_max_active);
      stats->leaf_pair_worklist_verify = tree.leaf_pair_worklist_verify;
      stats->leaf_pair_worklist_tasks += leaf_pair_batch_total.worklist_tasks;
      stats->leaf_pair_worklist_groups += leaf_pair_batch_total.worklist_groups;
      stats->leaf_pair_worklist_batched_groups +=
          leaf_pair_batch_total.worklist_batched_groups;
      stats->leaf_pair_worklist_scalar_flush_tasks +=
          leaf_pair_batch_total.worklist_scalar_flush_tasks;
      stats->leaf_pair_worklist_forced_flushes +=
          leaf_pair_batch_total.worklist_forced_flushes;
      stats->leaf_pair_worklist_active_groups_max =
          std::max(stats->leaf_pair_worklist_active_groups_max,
                   leaf_pair_batch_total.worklist_active_groups_max);
      stats->leaf_pair_worklist_ms += leaf_pair_batch_total.worklist_ms;
      stats->leaf_pair_worklist_verify_checks +=
          leaf_pair_batch_total.worklist_verify_checks;
      stats->leaf_pair_worklist_verify_mismatches +=
          leaf_pair_batch_total.worklist_verify_mismatches;
      stats->leaf_block_verify = tree.leaf_block_verify;
      stats->leaf_block_target_leaves +=
          leaf_pair_batch_total.leaf_block_target_leaves;
      stats->leaf_block_targets += leaf_pair_batch_total.leaf_block_targets;
      stats->leaf_block_source_groups +=
          leaf_pair_batch_total.leaf_block_source_groups;
      stats->leaf_block_batched_groups +=
          leaf_pair_batch_total.leaf_block_batched_groups;
      stats->leaf_block_scalar_tail_tasks +=
          leaf_pair_batch_total.leaf_block_scalar_tail_tasks;
      stats->leaf_block_group_ms += leaf_pair_batch_total.leaf_block_group_ms;
      stats->leaf_block_flush_ms += leaf_pair_batch_total.leaf_block_flush_ms;
      stats->leaf_block_writeback_ms +=
          leaf_pair_batch_total.leaf_block_writeback_ms;
      stats->leaf_block_verify_checks +=
          leaf_pair_batch_total.leaf_block_verify_checks;
      stats->leaf_block_verify_mismatches +=
          leaf_pair_batch_total.leaf_block_verify_mismatches;
      stats->target_cluster_interactions += leaf_pair_batch_total.interactions;
      stats->target_cluster_targets += leaf_pair_batch_total.targets;
      stats->target_cluster_panels_pruned +=
          leaf_pair_batch_total.source_panels;
      stats->target_cluster_rejected += leaf_pair_batch_total.fallbacks;
      stats->target_cluster_eval_ms += leaf_pair_batch_total.ms;
      stats->far_node_interactions += far_total;
      stats->mac_reject_radius_limited += std::accumulate(
          local_mac_reject.begin(), local_mac_reject.end(), std::size_t{0});
      stats->mac_convergence_rho_rejects += std::accumulate(
          local_mac_convergence_rho_reject.begin(),
          local_mac_convergence_rho_reject.end(), std::size_t{0});
      stats->mac_convergence_aspect_rejects += std::accumulate(
          local_mac_convergence_aspect_reject.begin(),
          local_mac_convergence_aspect_reject.end(), std::size_t{0});
      stats->mac_convergence_rho_reject_panels += std::accumulate(
          local_mac_convergence_rho_panel.begin(),
          local_mac_convergence_rho_panel.end(), std::size_t{0});
      stats->mac_convergence_aspect_reject_panels += std::accumulate(
          local_mac_convergence_aspect_panel.begin(),
          local_mac_convergence_aspect_panel.end(), std::size_t{0});
      stats->mac_surface_margin_rejects += std::accumulate(
          local_mac_surface_margin_reject.begin(),
          local_mac_surface_margin_reject.end(), std::size_t{0});
      stats->mac_surface_margin_reject_panels += std::accumulate(
          local_mac_surface_margin_panel.begin(),
          local_mac_surface_margin_panel.end(), std::size_t{0});
      stats->mac_surface_margin_applied += std::accumulate(
          local_mac_surface_margin_applied.begin(),
          local_mac_surface_margin_applied.end(), std::size_t{0});
      stats->mac_surface_margin_skipped += std::accumulate(
          local_mac_surface_margin_skipped.begin(),
          local_mac_surface_margin_skipped.end(), std::size_t{0});
      stats->mac_surface_margin_adaptive_rejects += std::accumulate(
          local_mac_surface_margin_adaptive_reject.begin(),
          local_mac_surface_margin_adaptive_reject.end(), std::size_t{0});
      stats->mac_surface_margin_adaptive_reject_panels += std::accumulate(
          local_mac_surface_margin_adaptive_panel.begin(),
          local_mac_surface_margin_adaptive_panel.end(), std::size_t{0});
      stats->mac_surface_margin_skipped_accepted_far += std::accumulate(
          local_mac_surface_margin_skipped_accepted.begin(),
          local_mac_surface_margin_skipped_accepted.end(), std::size_t{0});
      stats->mac_surface_margin_hard_applied += std::accumulate(
          local_mac_surface_margin_hard_applied.begin(),
          local_mac_surface_margin_hard_applied.end(), std::size_t{0});
      stats->mac_surface_margin_soft_applied += std::accumulate(
          local_mac_surface_margin_soft_applied.begin(),
          local_mac_surface_margin_soft_applied.end(), std::size_t{0});
      stats->mac_surface_margin_hard_rejects += std::accumulate(
          local_mac_surface_margin_hard_reject.begin(),
          local_mac_surface_margin_hard_reject.end(), std::size_t{0});
      stats->mac_surface_margin_hard_reject_panels += std::accumulate(
          local_mac_surface_margin_hard_panel.begin(),
          local_mac_surface_margin_hard_panel.end(), std::size_t{0});
      stats->mac_surface_margin_soft_rejects += std::accumulate(
          local_mac_surface_margin_soft_reject.begin(),
          local_mac_surface_margin_soft_reject.end(), std::size_t{0});
      stats->mac_surface_margin_soft_reject_panels += std::accumulate(
          local_mac_surface_margin_soft_panel.begin(),
          local_mac_surface_margin_soft_panel.end(), std::size_t{0});
      stats->convergence_k3_rho_cap_rejects += std::accumulate(
          local_k3_rho_cap_reject.begin(), local_k3_rho_cap_reject.end(),
          std::size_t{0});
      stats->convergence_k3_rho_cap_reject_panels += std::accumulate(
          local_k3_rho_cap_panel.begin(), local_k3_rho_cap_panel.end(),
          std::size_t{0});
      stats->near_interactions += singular_evaluated + leaf_total;
      stats->far_interactions += far_total;
      stats->leaf_panel_pairs_per_second =
          stats->leaf_panel_eval_ms > 0.0
              ? static_cast<double>(stats->leaf_panel_interactions) /
                    (stats->leaf_panel_eval_ms / 1000.0)
              : 0.0;
      stats->node_bounds_mode = tree.node_bounds_mode;
      stats->tree_builder = tree.tree_builder;
      stats->expansion_center = tree.expansion_center;
      stats->fragmentation_safety = tree.fragmentation_safety;
      stats->mac_safety = tree.mac_safety;
      stats->mac_gap_factor = tree.mac_gap_factor;
      stats->moment_audit = tree.moment_audit;
      stats->moment_construction = tree.moment_construction;
      stats->moment_sum = tree.moment_sum;
      stats->convergence_audit = tree.convergence_audit;
      stats->convergence_safety = tree.convergence_safety;
      stats->convergence_rho_cap = tree.convergence_rho_cap;
      stats->mac_aspect_cap = tree.mac_aspect_cap;
      stats->convergence_estimator = tree.convergence_estimator;
      stats->convergence_tail_threshold = tree.convergence_tail_threshold;
      stats->convergence_estimator_error_cap =
          tree.convergence_estimator_error_cap;
      stats->convergence_unstable_rho_cap =
          tree.convergence_unstable_rho_cap;
      stats->convergence_calibration = tree.convergence_calibration;
      stats->convergence_dielectric_amp =
          tree.convergence_dielectric_amp;
      stats->convergence_estimator_model =
          tree.convergence_estimator_model;
      stats->convergence_channel_calibration =
          tree.convergence_channel_calibration;
      stats->convergence_split_policy = tree.convergence_split_policy;
      stats->channel_missed_semantics = tree.channel_missed_semantics;
      stats->accepted_miss_diagnostics = tree.accepted_miss_diagnostics;
      stats->accepted_miss_absolute_floor =
          tree.accepted_miss_absolute_floor;
      stats->convergence_k3_accepted_rho_cap =
          tree.convergence_k3_accepted_rho_cap;
      stats->mac_distance_policy = tree.mac_distance_policy;
      stats->mac_surface_margin_factor = tree.mac_surface_margin_factor;
      stats->mac_surface_margin_policy = tree.mac_surface_margin_policy;
      stats->mac_surface_margin_aspect_threshold =
          tree.mac_surface_margin_aspect_threshold;
      stats->mac_surface_margin_rho_trigger =
          tree.mac_surface_margin_rho_trigger;
      stats->mac_surface_margin_soft_aspect_threshold =
          tree.mac_surface_margin_soft_aspect_threshold;
      stats->mac_surface_margin_soft_rho_trigger =
          tree.mac_surface_margin_soft_rho_trigger;
      stats->mac_surface_margin_soft_factor =
          tree.mac_surface_margin_soft_factor;
      stats->leaf_kernel_profile = tree.leaf_kernel_profile;
      stats->jcp_direct_kernel = tree.jcp_direct_kernel;
      stats->expansion_audit = tree.expansion_audit;
      stats->expansion_audit_channel = tree.expansion_audit_channel;
      stats->moment_audit_nodes += moment_audit_summary.nodes;
      stats->moment_translation_rel_error_max =
          std::max(stats->moment_translation_rel_error_max,
                   moment_audit_summary.max_rel_error);
      if (moment_audit_summary.nodes > 0) {
        const double old_count =
            static_cast<double>(stats->moment_audit_nodes -
                                moment_audit_summary.nodes);
        const double new_count = static_cast<double>(stats->moment_audit_nodes);
        stats->moment_translation_rel_error_mean =
            new_count > 0.0
                ? (stats->moment_translation_rel_error_mean * old_count +
                   moment_audit_summary.mean_rel_error *
                       static_cast<double>(moment_audit_summary.nodes)) /
                      new_count
                : 0.0;
      }
      stats->moment_translation_offset_max =
          std::max(stats->moment_translation_offset_max,
                   moment_audit_summary.translation_offset_max);
      if (moment_audit_summary.translation_offset_mean > 0.0) {
        stats->moment_translation_offset_mean =
            moment_audit_summary.translation_offset_mean;
      }
      stats->moment_cancellation_ratio_max =
          std::max(stats->moment_cancellation_ratio_max,
                   moment_audit_summary.cancellation_ratio_max);
      if (moment_audit_summary.cancellation_ratio_mean > 0.0)
        stats->moment_cancellation_ratio_mean =
            moment_audit_summary.cancellation_ratio_mean;
      stats->moment_dynamic_range_max =
          std::max(stats->moment_dynamic_range_max,
                   moment_audit_summary.dynamic_range_max);
      copy_far_geometric_audit_to_stats(*stats, mac_audit_total);
      copy_fragmentation_audit_to_stats(*stats, fragmentation_total);
      if (tree.accepted_miss_diagnostics == "sampled-detail")
        write_accepted_miss_detail_csv(expansion_total);
      if (tree.convergence_overreject_detail == "sampled-detail")
        write_overreject_calibration_detail_csv(expansion_total);
      if (tree.convergence_shape_audit == "sampled-detail")
        write_convergence_shape_calibration_csv(expansion_total);
      if (tree.convergence_kernel_response_audit == "sampled-detail")
        write_kernel_response_calibration_csv(expansion_total);
      copy_far_expansion_audit_to_stats(*stats, expansion_total);
      copy_far_convergence_audit_to_stats(*stats, convergence_total);
      copy_convergence_estimator_to_stats(*stats, estimator_total);
      stats->tree_depth_max = tree.depth_max;
      stats->node_radius_max = tree.node_radius_max;
      stats->node_radius_mean = tree.node_radius_mean;
      stats->node_aspect_ratio_max = tree.node_aspect_ratio_max;
      stats->node_aspect_ratio_mean = tree.node_aspect_ratio_mean;
      const double calls_after = static_cast<double>(stats->matvec_calls + 1);
      stats->leaf_panel_work_per_matvec =
          static_cast<double>(stats->leaf_panel_interactions) / calls_after;
      stats->far_node_work_per_matvec =
          static_cast<double>(stats->far_node_interactions) / calls_after;
      const double mac_denom =
          static_cast<double>(stats->far_node_interactions +
                              stats->leaf_panel_interactions +
                              stats->leaf_tile_panel_interactions +
                              stats->leaf_residual_panel_interactions);
      stats->mac_accept_rate =
          mac_denom > 0.0
              ? static_cast<double>(stats->far_node_interactions) / mac_denom
              : 0.0;
      ++stats->matvec_calls;
    }
    return;
  }

  const TreeInteractionLists &interactions = tree.interactions;
  const double interaction_ms = interactions.valid_for(n) ? interactions.build_ms
                                                          : 0.0;
  out.assign(coupled ? 2 * n : n, 0.0);
  const std::size_t workers = effective_workers(executor, n);
  std::vector<double> sum_phi(n, 0.0);
  std::vector<double> sum_phin(coupled ? n : 0, 0.0);
  std::vector<std::size_t> local_leaf_residual(workers, 0);
  std::vector<std::size_t> local_leaf_tile_nodes(workers, 0);
  std::vector<std::size_t> local_leaf_tile_panels(workers, 0);
  std::vector<std::size_t> local_smooth_leaf(workers, 0);
  std::vector<std::size_t> local_far(workers, 0);
  std::vector<TreecodeStats> local_near_stats(workers);

  const auto singular_begin = std::chrono::steady_clock::now();
  executor.parallel_for(
      n, workers, 8,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        TreecodeStats &worker_near_stats = local_near_stats[worker];
        for (std::size_t i = begin; i < end; ++i) {
          const Panel &target = panels[i];
          const std::size_t singular_begin_pos =
              interactions.singular_offsets[i];
          const std::size_t singular_count =
              interactions.singular_offsets[i + 1] - singular_begin_pos;
          if (singular_count == 0)
            continue;
          const JcpNearAccum near = accumulate_jcp_near_ordered(
              target, i, panels, unknowns, eps_ratio, kappa, coupled,
              interactions.singular_indices, singular_begin_pos,
              singular_count, near_integration, quad_buffer,
              &worker_near_stats);
          sum_phi[i] += near.phi;
          if (coupled)
            sum_phin[i] += near.phin;
        }
      });
  const double singular_ms =
      elapsed_ms(singular_begin, std::chrono::steady_clock::now());

  const auto leaf_begin = std::chrono::steady_clock::now();
  executor.parallel_for(
      n, workers, 8,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::size_t residual_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const std::size_t leaf_begin_pos = interactions.leaf_offsets[i];
          const std::size_t leaf_count_i =
              interactions.leaf_offsets[i + 1] - leaf_begin_pos;
          if (leaf_count_i == 0)
            continue;
          const JcpAccum near = direct_jcp_ordered_sum(
              panels[i], i, *panel_cache, unknown_cache, eps_ratio, kappa,
              coupled, interactions.leaf_indices, leaf_begin_pos,
              leaf_count_i, leaf_panel_kernel, residual_count);
          sum_phi[i] += near.phi;
          if (coupled)
            sum_phin[i] += near.phin;
        }
        local_leaf_residual[worker] += residual_count;
      });
  const double leaf_residual_ms =
      elapsed_ms(leaf_begin, std::chrono::steady_clock::now());

  const auto tile_begin = std::chrono::steady_clock::now();
  executor.parallel_for(
      n, workers, 8,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::size_t tile_node_count = 0;
        std::size_t tile_panel_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const std::size_t tile_begin_pos =
              interactions.leaf_tile_offsets[i];
          const std::size_t tile_count_i =
              interactions.leaf_tile_offsets[i + 1] - tile_begin_pos;
          for (std::size_t pos = tile_begin_pos;
               pos < tile_begin_pos + tile_count_i; ++pos) {
            const std::size_t node_index =
                interactions.leaf_tile_node_indices[pos];
            if (node_index >= tree.nodes.size())
              continue;
            const std::size_t span_begin =
                pos + 1 < interactions.leaf_tile_span_offsets.size()
                    ? interactions.leaf_tile_span_offsets[pos]
                    : 0;
            const std::size_t span_end =
                pos + 1 < interactions.leaf_tile_span_offsets.size()
                    ? interactions.leaf_tile_span_offsets[pos + 1]
                    : span_begin;
            for (std::size_t span = span_begin; span < span_end; ++span) {
              if (span >= interactions.leaf_tile_span_firsts.size() ||
                  span >= interactions.leaf_tile_span_counts.size())
                continue;
              TreeNode span_node = tree.nodes[node_index];
              span_node.first_panel = interactions.leaf_tile_span_firsts[span];
              span_node.panel_count = interactions.leaf_tile_span_counts[span];
              std::size_t counted_panels = 0;
              const JcpAccum near = direct_jcp_ordered_tile_sum(
                  panels[i], i, *panel_cache, unknown_cache, eps_ratio, kappa,
                  coupled, span_node, leaf_panel_kernel, counted_panels);
              sum_phi[i] += near.phi;
              if (coupled)
                sum_phin[i] += near.phin;
              tile_panel_count += counted_panels;
            }
            ++tile_node_count;
          }
        }
        local_leaf_tile_nodes[worker] += tile_node_count;
        local_leaf_tile_panels[worker] += tile_panel_count;
      });
  const double leaf_tile_ms =
      elapsed_ms(tile_begin, std::chrono::steady_clock::now());
  const double leaf_ms = leaf_residual_ms + leaf_tile_ms;

  const auto smooth_leaf_begin = std::chrono::steady_clock::now();
  executor.parallel_for(
      n, workers, 8,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::size_t smooth_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const Panel &target = panels[i];
          const std::size_t smooth_begin_pos =
              interactions.smooth_leaf_offsets[i];
          const std::size_t smooth_count_i =
              interactions.smooth_leaf_offsets[i + 1] - smooth_begin_pos;
          for (std::size_t pos = smooth_begin_pos;
               pos < smooth_begin_pos + smooth_count_i; ++pos) {
            const std::size_t node_index =
                interactions.smooth_leaf_node_indices[pos];
            const TreeNode &node = tree.nodes[node_index];
            const JcpAccum smooth = far_evaluator(
                target, node, jcp_moments.phin[node_index],
                jcp_moments.phi_normal[0][node_index],
                jcp_moments.phi_normal[1][node_index],
                jcp_moments.phi_normal[2][node_index], eps_ratio, kappa);
            sum_phi[i] += smooth.phi;
            if (coupled)
              sum_phin[i] += smooth.phin;
            ++smooth_count;
          }
        }
        local_smooth_leaf[worker] += smooth_count;
      });
  const double smooth_leaf_ms =
      elapsed_ms(smooth_leaf_begin, std::chrono::steady_clock::now());

  // The far-field audits used to live only in the on-the-fly traversal above,
  // which is not the one this branch runs and not the default.  The effect was
  // that every audit counter came back zero on the default path, and a zero
  // reads exactly like a clean far field.  The decision about which pairs are
  // far was made once, when the interaction lists were built, so the geometry
  // behind it has to be recomputed here -- it is a node radius over a centre
  // distance, and cheap.  Everything else is the same free function the other
  // traversal calls, so the two report on the same definitions.
  const bool far_audits_active =
      tree.convergence_audit != "off" || tree.expansion_audit != "off" ||
      tree.expansion_audit_channel != "off" ||
      tree.convergence_estimator == "moment-tail" ||
      tree.convergence_estimator == "hybrid";
  std::vector<ExpansionAuditTally> local_far_expansion_audit(
      far_audits_active ? workers : 0U);
  std::vector<ConvergenceAuditTally> local_far_convergence_audit(
      far_audits_active ? workers : 0U);
  std::vector<ConvergenceEstimatorTally> local_far_estimator(
      far_audits_active ? workers : 0U);
  std::atomic_size_t far_expansion_audit_candidates{0};

  const auto far_begin = std::chrono::steady_clock::now();
  const auto far_targets =
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        std::size_t far_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
          const Panel &target = panels[i];
          const std::size_t far_begin_pos = interactions.far_offsets[i];
          const std::size_t far_count_i =
              interactions.far_offsets[i + 1] - far_begin_pos;
          std::size_t pos = far_begin_pos;
          // Four far nodes at a time; each node's term is the one the loop
          // below would compute, added in the same list order.
          if (far_evaluator.fast4 != nullptr && !far_audits_active) {
            for (; pos + 4 <= far_begin_pos + far_count_i; pos += 4) {
              const TreeNode *nodes4[4];
              const double *density4[16];
              for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t node_index =
                    interactions.far_node_indices[pos + lane];
                nodes4[lane] = &tree.nodes[node_index];
                density4[4 * lane] = jcp_moments.phin[node_index].values.data();
                for (std::size_t c = 0; c < 3; ++c)
                  density4[4 * lane + 1 + c] =
                      jcp_moments.phi_normal[c][node_index].values.data();
              }
              JcpAccum far4[4];
              far_evaluator.fast4(target, nodes4, density4, eps_ratio, kappa,
                                  far4);
              for (const JcpAccum &far : far4) {
                sum_phi[i] += far.phi;
                if (coupled)
                  sum_phin[i] += far.phin;
              }
              far_count += 4;
            }
          }
          for (; pos < far_begin_pos + far_count_i; ++pos) {
            const std::size_t node_index = interactions.far_node_indices[pos];
            const TreeNode &node = tree.nodes[node_index];
            const JcpAccum far = far_evaluator(
                target, node, jcp_moments.phin[node_index],
                jcp_moments.phi_normal[0][node_index],
                jcp_moments.phi_normal[1][node_index],
                jcp_moments.phi_normal[2][node_index], eps_ratio, kappa);
            if (far_audits_active) {
              const std::array<NodeMoments, 3> normal_moments = {
                  jcp_moments.phi_normal[0][node_index],
                  jcp_moments.phi_normal[1][node_index],
                  jcp_moments.phi_normal[2][node_index]};
              const double dx = target.x - node.cx;
              const double dy = target.y - node.cy;
              const double dz = target.z - node.cz;
              const double distance =
                  std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
              const MacDecision mac =
                  evaluate_mac(target, node, tree, distance);
              if (tree.convergence_audit != "off")
                record_far_convergence_audit(
                    local_far_convergence_audit[worker], node, mac.rho);
              const ConvergenceEstimatorDecision estimator =
                  evaluate_convergence_estimator(tree, node, jcp_moments,
                                                 node_index, taylor_order,
                                                 coupled, mac.rho, eps_ratio);
              record_convergence_estimator(local_far_estimator[worker], node,
                                           estimator);
              // The ground-truth comparator: the same pair summed directly.
              // Sampled, because it is the expensive one.
              if (should_audit_far_expansion(tree, n, node,
                                             far_expansion_audit_candidates)) {
                std::size_t direct_panel_count = 0;
                const auto direct_begin = std::chrono::steady_clock::now();
                // Per-channel when asked for, because the aggregate hides
                // which kernel carries the error, and the K4 hypersingular
                // channel is the one the FMM research line kept flagging.
                const bool channel_audit =
                    tree.expansion_audit_channel != "off";
                JcpAccum direct;
                JcpChannelAccum direct_channels;
                JcpChannelAccum far_channels;
                if (channel_audit) {
                  direct_channels = direct_jcp_ordered_channel_sum_scalar(
                      target, i, *panel_cache, unknown_cache, eps_ratio,
                      kappa, coupled, tree.order, node.first_panel,
                      node.panel_count, direct_panel_count);
                  direct = sum_jcp_channels(direct_channels);
                  far_channels = taylor_jcp_node_channel_sum(
                      target, node, jcp_moments.phin[node_index],
                      normal_moments, eps_ratio, kappa, coupled, taylor_order,
                      tree.derivative_mode);
                } else {
                  direct = direct_jcp_ordered_sum(
                      target, i, *panel_cache, unknown_cache, eps_ratio,
                      kappa, coupled, tree.order, node.first_panel,
                      node.panel_count, leaf_panel_kernel,
                      direct_panel_count);
                }
                const double direct_ms = elapsed_ms(
                    direct_begin, std::chrono::steady_clock::now());
                record_far_expansion_audit(
                    local_far_expansion_audit[worker], far, direct, coupled,
                    node.panel_count, direct_ms, estimator.active,
                    estimator.unstable, estimator.split_reject,
                    tree.convergence_estimator_error_cap,
                    tree.accepted_miss_absolute_floor,
                    tree.channel_missed_semantics);
                if (channel_audit)
                  record_far_expansion_channel_audit(
                      local_far_expansion_audit[worker], far_channels,
                      direct_channels, estimator,
                      tree.convergence_estimator_error_cap,
                      tree.accepted_miss_absolute_floor,
                      tree.channel_missed_semantics,
                      tree.accepted_miss_diagnostics, i, node_index,
                      node.panel_count, mac.rho);
              }
            }
            sum_phi[i] += far.phi;
            if (coupled)
              sum_phin[i] += far.phin;
            ++far_count;
          }
          out[i] = jcp_lhs_phi_for_panel(panels[i], eps_ratio, lhs_phi) *
                       unknowns[i] -
                   sum_phi[i];
          if (coupled)
            out[n + i] =
                jcp_lhs_phin_for_panel(panels[i], eps_ratio, lhs_phin) *
                    unknowns[n + i] -
                sum_phin[i];
        }
        local_far[worker] += far_count;
      };
  if (far_audits_active) {
    executor.parallel_for(n, workers, 8, far_targets);
  } else {
    // Targets go out from a shared counter in shrinking chunks (a guided
    // schedule) instead of one equal block per worker: on a hybrid CPU the
    // equal split waits for the slowest core. The early chunks stay large and
    // contiguous, because neighbouring targets share most of their far nodes
    // and a worker's cache keeps them. Each target's sum is still formed by
    // one worker in list order, so the result does not depend on the
    // schedule. The audits keep the static split above.
    constexpr std::size_t kFarTargetChunkMin = 32;
    std::atomic_size_t next_target{0};
    executor.parallel_for(
        workers, workers, 1,
        [&](std::size_t worker, std::size_t, std::size_t) {
          std::size_t begin = next_target.load(std::memory_order_relaxed);
          while (begin < n) {
            const std::size_t chunk =
                std::max(kFarTargetChunkMin, (n - begin) / (2 * workers));
            if (next_target.compare_exchange_weak(begin, begin + chunk,
                                                  std::memory_order_relaxed)) {
              far_targets(worker, begin, std::min(n, begin + chunk));
              begin = next_target.load(std::memory_order_relaxed);
            }
          }
        });
  }
  const double far_ms = elapsed_ms(far_begin, std::chrono::steady_clock::now());
  const double traversal_ms = singular_ms + leaf_ms + smooth_leaf_ms + far_ms;

  if (stats != nullptr) {
    if (far_audits_active) {
      ExpansionAuditTally expansion_total;
      for (const ExpansionAuditTally &worker_audit : local_far_expansion_audit)
        merge_far_expansion_audit(expansion_total, worker_audit);
      ConvergenceAuditTally convergence_total;
      for (const ConvergenceAuditTally &worker_audit :
           local_far_convergence_audit)
        merge_far_convergence_audit(convergence_total, worker_audit);
      ConvergenceEstimatorTally estimator_total;
      for (const ConvergenceEstimatorTally &worker_audit : local_far_estimator)
        merge_convergence_estimator(estimator_total, worker_audit);
      if (tree.accepted_miss_diagnostics == "sampled-detail")
        write_accepted_miss_detail_csv(expansion_total);
      copy_far_expansion_audit_to_stats(*stats, expansion_total);
      copy_far_convergence_audit_to_stats(*stats, convergence_total);
      copy_convergence_estimator_to_stats(*stats, estimator_total);
    }
    std::size_t singular_evaluated = 0;
    for (const TreecodeStats &worker_stats : local_near_stats)
      singular_evaluated += worker_stats.near_centroid_pairs +
                             worker_stats.near_duffy_pairs +
                             worker_stats.near_geometric_buffer_pairs;
    const std::size_t leaf_residual_total =
        std::accumulate(local_leaf_residual.begin(), local_leaf_residual.end(),
                        std::size_t{0});
    const std::size_t leaf_tile_node_total =
        std::accumulate(local_leaf_tile_nodes.begin(),
                        local_leaf_tile_nodes.end(), std::size_t{0});
    const std::size_t leaf_tile_panel_total =
        std::accumulate(local_leaf_tile_panels.begin(),
                        local_leaf_tile_panels.end(), std::size_t{0});
    const std::size_t leaf_total =
        leaf_residual_total + leaf_tile_panel_total;
    const std::size_t smooth_leaf_total =
        std::accumulate(local_smooth_leaf.begin(), local_smooth_leaf.end(),
                        std::size_t{0});
    const std::size_t near_total =
        singular_evaluated + leaf_total + smooth_leaf_total;
    const std::size_t far_total =
        std::accumulate(local_far.begin(), local_far.end(), std::size_t{0});
    stats->matvec_mode = "jcp2013_treecode";
    stats->taylor_order = taylor_order;
    stats->derivative_mode = tree.derivative_mode;
    stats->leaf_near_policy = tree.leaf_near_policy;
    stats->leaf_panel_kernel = leaf_panel_kernel;
    stats->interaction_mode = tree.interaction_mode;
    stats->cluster_policy = tree.cluster_policy;
    stats->target_batch_size = static_cast<int>(tree.target_batch_size);
    stats->leaf_pair_min_source_panels =
        static_cast<int>(tree.leaf_pair_min_source_panels);
    stats->leaf_pair_min_targets =
        static_cast<int>(tree.leaf_pair_min_targets);
    stats->leaf_pair_selection_profile = tree.leaf_pair_selection_profile;
    stats->leaf_pair_probe_cache = tree.leaf_pair_probe_cache;
    stats->leaf_pair_target_group = tree.leaf_pair_target_group;
    stats->leaf_pair_directness_probe = tree.leaf_pair_directness_probe;
    stats->leaf_pair_directness_cache = tree.leaf_pair_directness_cache;
    stats->leaf_pair_directness_verify = tree.leaf_pair_directness_verify;
    stats->leaf_pair_scheduler = tree.leaf_pair_scheduler;
    stats->leaf_pair_worklist_max_active =
        static_cast<int>(tree.leaf_pair_worklist_max_active);
    stats->leaf_pair_worklist_verify = tree.leaf_pair_worklist_verify;
    stats->leaf_block_verify = tree.leaf_block_verify;
    stats->leaf_pair_max_lookahead =
        static_cast<int>(tree.leaf_pair_max_lookahead);
    stats->convergence_split_selectivity =
        tree.convergence_split_selectivity;
    stats->convergence_split_absolute_floor =
        tree.convergence_split_absolute_floor;
    stats->convergence_overreject_audit =
        tree.convergence_overreject_audit;
    stats->convergence_overreject_detail =
        tree.convergence_overreject_detail;
    stats->convergence_shape_audit = tree.convergence_shape_audit;
    stats->convergence_kernel_response_audit =
        tree.convergence_kernel_response_audit;
    stats->near_integration = near_integration;
    stats->quad_buffer = quad_buffer;
    stats->tree_moment_ms += moment_ms;
    if (stats->matvec_calls == 0)
      stats->interaction_build_ms += interaction_ms;
    stats->tree_traversal_ms += traversal_ms;
    stats->tree_eval_ms +=
        elapsed_ms(eval_begin, std::chrono::steady_clock::now());
    stats->singular_near_eval_ms += singular_ms;
    stats->leaf_panel_eval_ms += leaf_ms;
    stats->leaf_tile_eval_ms += leaf_tile_ms;
    stats->leaf_residual_eval_ms += leaf_residual_ms;
    stats->smooth_leaf_node_eval_ms += smooth_leaf_ms;
    stats->leaf_near_eval_ms += leaf_ms + smooth_leaf_ms;
    stats->near_eval_ms += singular_ms + leaf_ms + smooth_leaf_ms;
    stats->far_eval_ms += far_ms;
    stats->near_integration_ms += singular_ms;
    for (const TreecodeStats &worker_stats : local_near_stats) {
      const std::size_t previous_duffy = stats->near_duffy_pairs;
      stats->near_kernel_policy = worker_stats.near_kernel_policy;
      stats->near_self_pairs += worker_stats.near_self_pairs;
      stats->near_one_ring_pairs += worker_stats.near_one_ring_pairs;
      stats->near_duffy_pairs += worker_stats.near_duffy_pairs;
      stats->near_centroid_pairs += worker_stats.near_centroid_pairs;
      stats->near_fallbacks += worker_stats.near_fallbacks;
      merge_near_kernel_block_stats(*stats, worker_stats);
      stats->duffy_subdivisions_max =
          std::max(stats->duffy_subdivisions_max,
                   worker_stats.duffy_subdivisions_max);
      if (worker_stats.near_duffy_pairs > 0) {
        const double old_count = static_cast<double>(previous_duffy);
        const double new_count = static_cast<double>(stats->near_duffy_pairs);
        stats->duffy_subdivisions_mean =
            new_count <= 0.0
                ? 0.0
                : (stats->duffy_subdivisions_mean * old_count +
                   worker_stats.duffy_subdivisions_mean *
                       static_cast<double>(worker_stats.near_duffy_pairs)) /
                      new_count;
      }
    }
    stats->near_centroid_pairs += leaf_total;
    stats->singular_near_interactions +=
        interactions.valid_for(n) ? interactions.singular_pair_count : 0;
    stats->leaf_near_interactions += leaf_total;
    stats->leaf_panel_interactions += leaf_total;
    stats->leaf_tile_interactions += leaf_tile_node_total;
    stats->leaf_tile_panel_interactions += leaf_tile_panel_total;
    stats->leaf_residual_panel_interactions += leaf_residual_total;
    stats->leaf_work_profile = tree.leaf_work_profile;
    stats->leaf_work_mac_terminal_panels += leaf_total;
    stats->leaf_work_profile_accounted_panels += leaf_total;
    stats->leaf_tile_status =
        tile_fallback_status(leaf_panel_kernel, leaf_tile_node_total);
    stats->smooth_leaf_node_interactions += smooth_leaf_total;
    stats->leaf_pruned_interactions +=
        interactions.valid_for(n) ? interactions.leaf_pruned_count : 0;
    stats->leaf_mac_candidates +=
        interactions.valid_for(n) ? interactions.leaf_mac_candidates : 0;
    stats->leaf_mac_accepted +=
        interactions.valid_for(n) ? interactions.leaf_mac_accepted : 0;
    stats->leaf_mac_rejected +=
        interactions.valid_for(n) ? interactions.leaf_mac_rejected : 0;
    stats->leaf_mac_far_panels +=
        interactions.valid_for(n) ? interactions.leaf_mac_far_panels : 0;
    stats->leaf_mac_direct_panels_saved_estimate +=
        interactions.valid_for(n)
            ? interactions.leaf_mac_direct_panels_saved_estimate
            : 0;
    stats->adaptive_smooth_leaf_accepted +=
        interactions.valid_for(n) ? interactions.adaptive_smooth_leaf_accepted
                                  : 0;
    stats->adaptive_smooth_leaf_rejected +=
        interactions.valid_for(n) ? interactions.adaptive_smooth_leaf_rejected
                                  : 0;
    stats->adaptive_span_tile_accepted +=
        interactions.valid_for(n) ? interactions.adaptive_span_tile_accepted
                                  : 0;
    stats->adaptive_span_tile_rejected +=
        interactions.valid_for(n) ? interactions.adaptive_span_tile_rejected
                                  : 0;
    stats->adaptive_exact_leaf_fallbacks +=
        interactions.valid_for(n) ? interactions.adaptive_exact_leaf_fallbacks
                                  : 0;
    stats->adaptive_reject_one_ring_overlap +=
        interactions.valid_for(n)
            ? interactions.adaptive_reject_one_ring_overlap
            : 0;
    stats->adaptive_reject_error_bound_failed +=
        interactions.valid_for(n)
            ? interactions.adaptive_reject_error_bound_failed
            : 0;
    stats->adaptive_reject_p4_check_failed +=
        interactions.valid_for(n) ? interactions.adaptive_reject_p4_check_failed
                                  : 0;
    stats->adaptive_reject_too_close +=
        interactions.valid_for(n) ? interactions.adaptive_reject_too_close : 0;
    stats->target_cluster_interactions +=
        interactions.valid_for(n) ? interactions.target_cluster_interactions
                                  : 0;
    stats->target_cluster_targets +=
        interactions.valid_for(n) ? interactions.target_cluster_targets : 0;
    stats->target_cluster_panels_pruned +=
        interactions.valid_for(n) ? interactions.target_cluster_panels_pruned
                                  : 0;
    stats->target_cluster_rejected +=
        interactions.valid_for(n) ? interactions.target_cluster_rejected : 0;
    stats->target_cluster_eval_ms += smooth_leaf_ms;
    stats->adaptive_level = tree.adaptive_level;
    const double possible_leaf_work =
        static_cast<double>(stats->leaf_panel_interactions +
                            stats->leaf_pruned_interactions);
    stats->leaf_work_reduction_ratio =
        possible_leaf_work > 0.0
            ? static_cast<double>(stats->leaf_pruned_interactions) /
                  possible_leaf_work
            : 0.0;
    stats->leaf_near_policy = tree.leaf_near_policy;
    stats->far_node_interactions += far_total;
    stats->mac_reject_radius_limited +=
        interactions.valid_for(n) ? interactions.mac_reject_radius_limited : 0;
    stats->mac_convergence_rho_rejects +=
        interactions.valid_for(n) ? interactions.mac_convergence_rho_rejects
                                  : 0;
    stats->mac_convergence_aspect_rejects +=
        interactions.valid_for(n) ? interactions.mac_convergence_aspect_rejects
                                  : 0;
    stats->mac_convergence_rho_reject_panels +=
        interactions.valid_for(n)
            ? interactions.mac_convergence_rho_reject_panels
            : 0;
    stats->mac_convergence_aspect_reject_panels +=
        interactions.valid_for(n)
            ? interactions.mac_convergence_aspect_reject_panels
            : 0;
    stats->mac_surface_margin_rejects +=
        interactions.valid_for(n) ? interactions.mac_surface_margin_rejects
                                  : 0;
    stats->mac_surface_margin_reject_panels +=
        interactions.valid_for(n)
            ? interactions.mac_surface_margin_reject_panels
            : 0;
    stats->near_interactions += near_total;
    stats->far_interactions += far_total;
    stats->leaf_panel_pairs_per_second =
        stats->leaf_panel_eval_ms > 0.0
            ? static_cast<double>(stats->leaf_panel_interactions) /
                  (stats->leaf_panel_eval_ms / 1000.0)
            : 0.0;
    stats->node_bounds_mode = tree.node_bounds_mode;
    stats->tree_builder = tree.tree_builder;
    stats->expansion_center = tree.expansion_center;
    stats->fragmentation_safety = tree.fragmentation_safety;
    stats->mac_safety = tree.mac_safety;
    stats->mac_gap_factor = tree.mac_gap_factor;
    stats->moment_audit = tree.moment_audit;
    stats->moment_construction = tree.moment_construction;
    stats->moment_sum = tree.moment_sum;
    stats->convergence_audit = tree.convergence_audit;
    stats->convergence_safety = tree.convergence_safety;
    stats->convergence_rho_cap = tree.convergence_rho_cap;
    stats->mac_aspect_cap = tree.mac_aspect_cap;
    stats->convergence_estimator = tree.convergence_estimator;
    stats->convergence_tail_threshold = tree.convergence_tail_threshold;
    stats->convergence_estimator_error_cap =
        tree.convergence_estimator_error_cap;
    stats->convergence_unstable_rho_cap =
        tree.convergence_unstable_rho_cap;
    stats->convergence_calibration = tree.convergence_calibration;
    stats->convergence_dielectric_amp = tree.convergence_dielectric_amp;
    stats->convergence_estimator_model = tree.convergence_estimator_model;
    stats->convergence_channel_calibration =
        tree.convergence_channel_calibration;
    stats->convergence_split_policy = tree.convergence_split_policy;
    stats->channel_missed_semantics = tree.channel_missed_semantics;
    stats->accepted_miss_diagnostics = tree.accepted_miss_diagnostics;
    stats->accepted_miss_absolute_floor =
        tree.accepted_miss_absolute_floor;
    stats->convergence_k3_accepted_rho_cap =
        tree.convergence_k3_accepted_rho_cap;
    stats->mac_distance_policy = tree.mac_distance_policy;
    stats->mac_surface_margin_factor = tree.mac_surface_margin_factor;
    stats->mac_surface_margin_policy = tree.mac_surface_margin_policy;
    stats->mac_surface_margin_aspect_threshold =
        tree.mac_surface_margin_aspect_threshold;
    stats->mac_surface_margin_rho_trigger =
        tree.mac_surface_margin_rho_trigger;
    stats->expansion_audit = tree.expansion_audit;
    stats->expansion_audit_channel = tree.expansion_audit_channel;
    stats->moment_audit_nodes += moment_audit_summary.nodes;
    stats->moment_translation_rel_error_max =
        std::max(stats->moment_translation_rel_error_max,
                 moment_audit_summary.max_rel_error);
    if (moment_audit_summary.nodes > 0) {
      const double old_count =
          static_cast<double>(stats->moment_audit_nodes -
                              moment_audit_summary.nodes);
      const double new_count = static_cast<double>(stats->moment_audit_nodes);
      stats->moment_translation_rel_error_mean =
          new_count > 0.0
              ? (stats->moment_translation_rel_error_mean * old_count +
                 moment_audit_summary.mean_rel_error *
                     static_cast<double>(moment_audit_summary.nodes)) /
                    new_count
              : 0.0;
    }
    stats->moment_translation_offset_max =
        std::max(stats->moment_translation_offset_max,
                 moment_audit_summary.translation_offset_max);
    if (moment_audit_summary.translation_offset_mean > 0.0)
      stats->moment_translation_offset_mean =
          moment_audit_summary.translation_offset_mean;
    stats->moment_cancellation_ratio_max =
        std::max(stats->moment_cancellation_ratio_max,
                 moment_audit_summary.cancellation_ratio_max);
    if (moment_audit_summary.cancellation_ratio_mean > 0.0)
      stats->moment_cancellation_ratio_mean =
          moment_audit_summary.cancellation_ratio_mean;
    stats->moment_dynamic_range_max =
        std::max(stats->moment_dynamic_range_max,
                 moment_audit_summary.dynamic_range_max);
    if (interactions.valid_for(n)) {
      MacAuditTally cached_audit;
      cached_audit.unsafe_count = interactions.far_geometric_unsafe_count;
      cached_audit.unsafe_panels = interactions.far_geometric_unsafe_panels;
      cached_audit.min_gap = interactions.far_geometric_min_gap;
      cached_audit.min_gap_ratio = interactions.far_geometric_gap_ratio_min;
      cached_audit.sum_gap_ratio = interactions.far_geometric_gap_ratio_sum;
      cached_audit.samples = interactions.far_geometric_audit_samples;
      cached_audit.physical_gap_rejects =
          interactions.mac_physical_gap_rejects;
      copy_far_geometric_audit_to_stats(*stats, cached_audit);
      FragmentationTally cached_fragmentation;
      cached_fragmentation.far_count = interactions.fragmented_far_count;
      cached_fragmentation.far_panels = interactions.fragmented_far_panels;
      cached_fragmentation.split_rejects =
          interactions.fragmented_split_rejects;
      cached_fragmentation.gap_ratio_sum =
          interactions.fragmentation_center_gap_ratio_sum;
      cached_fragmentation.gap_ratio_max =
          interactions.fragmentation_center_gap_ratio_max;
      copy_fragmentation_audit_to_stats(*stats, cached_fragmentation);
    }
    stats->tree_depth_max = tree.depth_max;
    stats->node_radius_max = tree.node_radius_max;
    stats->node_radius_mean = tree.node_radius_mean;
    stats->node_aspect_ratio_max = tree.node_aspect_ratio_max;
    stats->node_aspect_ratio_mean = tree.node_aspect_ratio_mean;
    const double calls_after = static_cast<double>(stats->matvec_calls + 1);
    stats->leaf_panel_work_per_matvec =
        static_cast<double>(stats->leaf_panel_interactions) / calls_after;
    stats->far_node_work_per_matvec =
        static_cast<double>(stats->far_node_interactions) / calls_after;
    const double mac_denom =
        static_cast<double>(stats->far_node_interactions +
                            stats->leaf_panel_interactions +
                            stats->leaf_tile_panel_interactions +
                            stats->leaf_residual_panel_interactions);
    stats->mac_accept_rate =
        mac_denom > 0.0
            ? static_cast<double>(stats->far_node_interactions) / mac_denom
            : 0.0;
    ++stats->matvec_calls;
  }
}

} // namespace TitanPBE::Tabi::detail

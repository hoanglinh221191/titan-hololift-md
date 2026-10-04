#include "bie_cartesian_fmm_operator.h"

#include "../../../titan_executor.h"
#include "bie_cartesian_fmm_translations.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace titan::bie::fmm::cartesian {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

template <typename Fn>
void dynamic_tiled_for(std::size_t count, std::size_t tile,
                       TitanExecutor &executor, Fn &&fn) {
  if (count == 0)
    return;
  const std::size_t workers =
      std::max<std::size_t>(1, std::min(executor.capacity(), count));
  std::atomic<std::size_t> next{0};
  executor.parallel_do(workers, [&](std::size_t worker) {
    for (;;) {
      const std::size_t begin = next.fetch_add(tile, std::memory_order_relaxed);
      if (begin >= count)
        break;
      const std::size_t end = std::min(count, begin + tile);
      for (std::size_t index = begin; index < end; ++index)
        fn(worker, index);
    }
  });
}

bool finite_outputs(std::span<const double> values) noexcept {
  for (const double value : values) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

} // namespace

CartesianFmmOperator::CartesianFmmOperator(
    std::shared_ptr<const CartesianFmmPlan> plan)
    : plan_(std::move(plan)) {
  if (!plan_ || plan_->stats().status != "ok")
    throw std::invalid_argument(
        "BIE Cartesian full FMM operator requires a valid plan");
  workspace_.prepare(*plan_);
}

CartesianFmmApplyStats
CartesianFmmOperator::apply_far(std::span<const double> phi,
                                std::span<const double> phin,
                                TitanExecutor &executor) {
  const auto apply_begin = Clock::now();
  const std::size_t panel_count = plan_->panels().size();
  if (phi.size() != panel_count || phin.size() != panel_count)
    throw std::invalid_argument(
        "BIE Cartesian full FMM density and panel counts differ");
  workspace_.clear(*plan_);
  CartesianFmmApplyStats stats;
  auto &moments = workspace_.moments();
  auto &locals = workspace_.locals();
  const int order = plan_->options().expansion_order;
  const std::size_t terms = coefficient_count(order);

  const auto p2m_begin = Clock::now();
  dynamic_tiled_for(
      plan_->source_leaves().size(), 4, executor,
      [&](std::size_t, std::size_t leaf_index) {
        const std::uint32_t node_id = plan_->source_leaves()[leaf_index];
        const CartesianFmmNode &node = plan_->nodes()[node_id];
        const auto all_indices = plan_->source_panel_indices();
        const std::span<const int> indices(
            all_indices.data() + node.source_begin, node.source_count);
        moments[node_id] = build_leaf_moments(plan_->panels(), phi, phin,
                                              indices, node.center, order);
      });
  stats.p2m_seconds = seconds_since(p2m_begin);

  const auto m2m_begin = Clock::now();
  const auto level_offsets = plan_->level_offsets();
  const auto nodes_by_level = plan_->nodes_by_level();
  if (level_offsets.size() >= 2) {
    for (std::size_t reverse_level = level_offsets.size() - 1;
         reverse_level-- > 1;) {
      const std::size_t level = reverse_level - 1;
      const std::size_t begin = level_offsets[level];
      const std::size_t count = level_offsets[level + 1] - begin;
      dynamic_tiled_for(
          count, 4, executor, [&](std::size_t, std::size_t offset) {
            const std::uint32_t node_id = nodes_by_level[begin + offset];
            const CartesianFmmNode &node = plan_->nodes()[node_id];
            if (node.source_population == 0 || node.source_count > 0)
              return;
            JcpMoments &parent = moments[node_id];
            parent.order = order;
            parent.term_count = terms;
            for (const int child_id : node.child) {
              if (child_id < 0)
                continue;
              const CartesianFmmNode &child =
                  plan_->nodes()[static_cast<std::size_t>(child_id)];
              if (child.source_population == 0)
                continue;
              accumulate_m2m(parent,
                             moments[static_cast<std::size_t>(child_id)],
                             node.center, child.center);
            }
          });
    }
  }
  stats.m2m_seconds = seconds_since(m2m_begin);

  const auto m2l_begin = Clock::now();
  dynamic_tiled_for(
      plan_->m2l_target_nodes().size(), 1, executor,
      [&](std::size_t, std::size_t target_index) {
        const std::uint32_t target_id = plan_->m2l_target_nodes()[target_index];
        JcpLocals &target_local = locals[target_id];
        const std::size_t begin = plan_->m2l_offsets()[target_id];
        const std::size_t end = plan_->m2l_offsets()[target_id + 1];
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
          const CartesianFmmPair &pair = plan_->m2l_pairs()[pair_index];
          accumulate_m2l(target_local, moments[pair.source_node],
                         plan_->coulomb_translation(pair.translation),
                         plan_->yukawa_translation(pair.translation),
                         plan_->options().eps_ratio, order);
        }
      });
  stats.m2l_seconds = seconds_since(m2l_begin);

  const auto l2l_begin = Clock::now();
  if (level_offsets.size() >= 3) {
    for (std::size_t level = 1; level + 1 < level_offsets.size(); ++level) {
      const std::size_t begin = level_offsets[level];
      const std::size_t count = level_offsets[level + 1] - begin;
      dynamic_tiled_for(
          count, 4, executor, [&](std::size_t, std::size_t offset) {
            const std::uint32_t node_id = nodes_by_level[begin + offset];
            const CartesianFmmNode &node = plan_->nodes()[node_id];
            if (node.target_population == 0 || node.parent < 0)
              return;
            const CartesianFmmNode &parent =
                plan_->nodes()[static_cast<std::size_t>(node.parent)];
            accumulate_l2l(locals[node_id],
                           locals[static_cast<std::size_t>(node.parent)],
                           parent.center, node.center);
          });
    }
  }
  stats.l2l_seconds = seconds_since(l2l_begin);

  const auto l2p_begin = Clock::now();
  auto &interaction_phi = workspace_.interaction_phi();
  auto &interaction_phin = workspace_.interaction_phin();
  dynamic_tiled_for(
      plan_->target_leaves().size(), 4, executor,
      [&](std::size_t, std::size_t leaf_index) {
        const std::uint32_t node_id = plan_->target_leaves()[leaf_index];
        const CartesianFmmNode &node = plan_->nodes()[node_id];
        const auto all_indices = plan_->target_panel_indices();
        for (std::size_t position = 0; position < node.target_count;
             ++position) {
          const int raw_panel = all_indices[node.target_begin + position];
          if (raw_panel < 0)
            continue;
          const std::size_t panel = static_cast<std::size_t>(raw_panel);
          const auto channels = evaluate_l2p(plan_->panels()[panel],
                                             node.center, locals[node_id]);
          interaction_phi[panel] += channels[0] + channels[1];
          interaction_phin[panel] += channels[2] + channels[3];
        }
      });
  stats.l2p_seconds = seconds_since(l2p_begin);
  stats.retained_workspace_bytes = workspace_.retained_bytes();
  stats.finite_outputs =
      finite_outputs(interaction_phi) && finite_outputs(interaction_phin);
  for (const JcpMoments &node_moments : moments)
    stats.finite_outputs = stats.finite_outputs && node_moments.finite_outputs;
  for (const JcpLocals &node_local : locals)
    stats.finite_outputs = stats.finite_outputs && node_local.finite_outputs;
  stats.apply_seconds = seconds_since(apply_begin);
  return stats;
}

} // namespace titan::bie::fmm::cartesian

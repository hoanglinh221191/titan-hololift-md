#include "bie_cartesian_fmm_solver_adapter.h"

#include "../tabi/tabi_near.h"
#include "../titan_executor.h"
#include "fmm/cartesian/bie_cartesian_fmm_checked_size.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace titan::bie {
namespace detail = fmm::cartesian::detail;
namespace {

using Clock = std::chrono::steady_clock;
namespace packed_fallback = titan::tabi::fmm_m2l;

class ApplyGuardReset {
public:
  explicit ApplyGuardReset(std::atomic_flag &guard) noexcept : guard_(guard) {}
  ApplyGuardReset(const ApplyGuardReset &) = delete;
  ApplyGuardReset &operator=(const ApplyGuardReset &) = delete;
  ~ApplyGuardReset() { guard_.clear(std::memory_order_release); }

private:
  std::atomic_flag &guard_;
};

double seconds_since(Clock::time_point begin) noexcept {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

#if defined(TITAN_CARTESIAN_FMM_TESTING)
bool testing_timing_value_valid(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

void testing_add_timing_value(double &total, double value,
                              const char *phase) {
  if (!testing_timing_value_valid(total) || !testing_timing_value_valid(value)) {
    throw std::runtime_error(std::string("P3D0 Cartesian direct timing is "
                                         "nonfinite or negative for ") +
                             phase);
  }
  total += value;
  if (!testing_timing_value_valid(total)) {
    throw std::runtime_error(std::string("P3D0 Cartesian direct timing "
                                         "accumulator overflow for ") +
                             phase);
  }
}

void testing_add_signed_timing_residual(double &total, double value,
                                        const char *phase) {
  if (!std::isfinite(total) || !std::isfinite(value)) {
    throw std::runtime_error(std::string("P3D0 Cartesian timing residual is "
                                         "nonfinite for ") +
                             phase);
  }
  total += value;
  if (!std::isfinite(total)) {
    throw std::runtime_error(std::string("P3D0 Cartesian timing residual "
                                         "accumulator overflow for ") +
                             phase);
  }
}

double testing_timing_tolerance(double parent_seconds) noexcept {
  return std::max(1.0e-6, 1.0e-5 * std::max(0.0, parent_seconds));
}
#endif

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

void accumulate_apply_stats(
    fmm::cartesian::CartesianFmmApplyStats &total,
    const fmm::cartesian::CartesianFmmApplyStats &value) {
  total.p2m_seconds += value.p2m_seconds;
  total.m2m_seconds += value.m2m_seconds;
  total.m2l_seconds += value.m2l_seconds;
  total.l2l_seconds += value.l2l_seconds;
  total.l2p_seconds += value.l2p_seconds;
  total.direct_seconds += value.direct_seconds;
  total.assemble_seconds += value.assemble_seconds;
  total.apply_seconds += value.apply_seconds;
  total.retained_workspace_bytes =
      std::max(total.retained_workspace_bytes, value.retained_workspace_bytes);
  const auto expected_pairs = detail::checked_add_or_error(
      total.expected_direct_panel_pairs, value.expected_direct_panel_pairs,
      "accumulating expected Cartesian direct panel pairs");
  const auto applied_pairs = detail::checked_add_or_error(
      total.direct_panel_pairs, value.direct_panel_pairs,
      "accumulating applied Cartesian direct panel pairs");
  if (!expected_pairs || !applied_pairs) {
    throw std::overflow_error(!expected_pairs ? expected_pairs.error()
                                              : applied_pairs.error());
  }
  total.expected_direct_panel_pairs = *expected_pairs;
  total.direct_panel_pairs = *applied_pairs;
  total.finite_outputs = total.finite_outputs && value.finite_outputs;
}

bool all_node_patch_panels(
    const std::vector<TitanPBE::Tabi::Panel> &panels) noexcept {
  if (panels.empty())
    return false;
  const auto *geometry = panels.front().node_patch_geometry.get();
  return geometry != nullptr &&
         std::all_of(panels.begin(), panels.end(),
                     [geometry](const TitanPBE::Tabi::Panel &panel) {
                       return panel.node_patch_geometry.get() == geometry &&
                              panel.node_patch_index !=
                                  static_cast<std::size_t>(-1);
                     });
}

std::uint64_t direct_leaf_pair_key(std::uint32_t target,
                                   std::uint32_t source) noexcept {
  return (static_cast<std::uint64_t>(target) << 32) |
         static_cast<std::uint64_t>(source);
}

void build_node_patch_direct_corrections(
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    const fmm::cartesian::CartesianFmmPlan &plan, double eps_ratio,
    double kappa, std::vector<std::size_t> &offsets,
    std::vector<NodePatchDirectCorrection> &corrections) {
  const std::size_t panel_count = panels.size();
  const auto offset_count = detail::checked_add_or_error(
      panel_count, std::size_t{1},
      "allocating Cartesian node-patch correction offsets");
  if (!offset_count)
    throw std::overflow_error(offset_count.error());
  const auto offset_bytes = detail::checked_container_allocation(
      *offset_count, offsets,
      "allocating Cartesian node-patch correction offsets");
  if (!offset_bytes)
    throw std::length_error(offset_bytes.error());
  offsets.assign(*offset_count, 0);
  corrections.clear();
  if (!all_node_patch_panels(panels))
    return;

  constexpr std::uint32_t invalid_leaf =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> target_leaf;
  std::vector<std::uint32_t> source_leaf;
  const auto target_leaf_bytes = detail::checked_container_allocation(
      panel_count, target_leaf,
      "allocating Cartesian node-patch target leaf map");
  const auto source_leaf_bytes = detail::checked_container_allocation(
      panel_count, source_leaf,
      "allocating Cartesian node-patch source leaf map");
  if (!target_leaf_bytes || !source_leaf_bytes) {
    throw std::length_error(!target_leaf_bytes ? target_leaf_bytes.error()
                                               : source_leaf_bytes.error());
  }
  target_leaf.assign(panel_count, invalid_leaf);
  source_leaf.assign(panel_count, invalid_leaf);
  for (const std::uint32_t node_id : plan.target_leaves()) {
    const auto &node = plan.nodes()[node_id];
    for (std::size_t pos = 0; pos < node.target_count; ++pos) {
      const int raw = plan.target_panel_indices()[node.target_begin + pos];
      if (raw >= 0 && static_cast<std::size_t>(raw) < panel_count)
        target_leaf[static_cast<std::size_t>(raw)] = node_id;
    }
  }
  for (const std::uint32_t node_id : plan.source_leaves()) {
    const auto &node = plan.nodes()[node_id];
    for (std::size_t pos = 0; pos < node.source_count; ++pos) {
      const int raw = plan.source_panel_indices()[node.source_begin + pos];
      if (raw >= 0 && static_cast<std::size_t>(raw) < panel_count)
        source_leaf[static_cast<std::size_t>(raw)] = node_id;
    }
  }
  if (std::any_of(target_leaf.begin(), target_leaf.end(),
                  [](std::uint32_t leaf) {
                    return leaf == invalid_leaf;
                  }) ||
      std::any_of(source_leaf.begin(), source_leaf.end(),
                  [](std::uint32_t leaf) {
                    return leaf == invalid_leaf;
                  })) {
    throw std::runtime_error(
        "BIE Cartesian full FMM node-patch leaf map is incomplete");
  }

  std::unordered_set<std::uint64_t> direct_leaf_pairs;
  const auto direct_leaf_reserve = detail::checked_multiply_or_error(
      plan.direct_pairs().size(), std::size_t{2},
      "reserving Cartesian node-patch direct leaf pairs");
  if (!direct_leaf_reserve)
    throw std::overflow_error(direct_leaf_reserve.error());
  if (*direct_leaf_reserve > direct_leaf_pairs.max_size()) {
    throw std::length_error(
        "BIE Cartesian full FMM node-patch direct leaf map exceeds container limit");
  }
  direct_leaf_pairs.reserve(*direct_leaf_reserve);
  for (const auto &pair : plan.direct_pairs())
    direct_leaf_pairs.insert(
        direct_leaf_pair_key(pair.target_node, pair.source_node));

  const auto &geometry = *panels.front().node_patch_geometry;
  std::vector<std::size_t> panel_for_node;
  const auto node_map_bytes = detail::checked_container_allocation(
      geometry.vertex_xyz.size(), panel_for_node,
      "allocating Cartesian node-patch panel map");
  const auto correction_bytes = detail::checked_container_allocation(
      geometry.adjacent_nodes.size(), corrections,
      "reserving Cartesian node-patch corrections");
  if (!node_map_bytes || !correction_bytes) {
    throw std::length_error(!node_map_bytes ? node_map_bytes.error()
                                            : correction_bytes.error());
  }
  panel_for_node.assign(geometry.vertex_xyz.size(), panel_count);
  corrections.reserve(geometry.adjacent_nodes.size());
  for (std::size_t panel = 0; panel < panel_count; ++panel) {
    const std::size_t node = panels[panel].node_patch_index;
    if (node >= panel_for_node.size() || panel_for_node[node] != panel_count)
      throw std::runtime_error(
          "BIE Cartesian full FMM node-patch panel map is invalid");
    panel_for_node[node] = panel;
  }

  corrections.reserve(geometry.adjacent_nodes.size());
  for (std::size_t target = 0; target < panel_count; ++target) {
    const std::size_t target_node = panels[target].node_patch_index;
    if (target_node + 1 >= geometry.adjacent_offsets.size())
      throw std::runtime_error(
          "BIE Cartesian full FMM node-patch adjacency is invalid");
    const std::size_t begin = geometry.adjacent_offsets[target_node];
    const std::size_t end = geometry.adjacent_offsets[target_node + 1];
    for (std::size_t pos = begin; pos < end; ++pos) {
      const std::size_t source_node = geometry.adjacent_nodes[pos];
      if (source_node >= panel_for_node.size())
        throw std::runtime_error(
            "BIE Cartesian full FMM node-patch neighbor is invalid");
      const std::size_t source = panel_for_node[source_node];
      if (source >= panel_count)
        throw std::runtime_error(
            "BIE Cartesian full FMM node-patch neighbor has no panel");
      if (!direct_leaf_pairs.contains(
              direct_leaf_pair_key(target_leaf[target], source_leaf[source])))
        continue;

      const auto analytic = TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
          panels[target], target, panels[source], source, eps_ratio, kappa,
          true, "analytic_singular", "off", nullptr);
      const auto centroid =
          TitanPBE::Tabi::evaluate_jcp_centroid_monopole_kernel(
              panels[target], panels[source], eps_ratio, kappa);
      const auto analytic_uncoupled =
          TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
              panels[target], target, panels[source], source, eps_ratio, 0.0,
              false, "analytic_singular", "off", nullptr);
      const auto centroid_uncoupled =
          TitanPBE::Tabi::evaluate_jcp_centroid_monopole_kernel(
              panels[target], panels[source], eps_ratio, 0.0);
      if (!analytic.valid || !analytic.one_ring || !centroid.valid ||
          !analytic_uncoupled.valid || !analytic_uncoupled.one_ring ||
          !centroid_uncoupled.valid) {
        throw std::runtime_error(
            "BIE Cartesian full FMM node-patch correction is invalid");
      }
      corrections.push_back(
          {static_cast<std::uint32_t>(source), analytic.k1 - centroid.k1,
           analytic.k2 - centroid.k2, analytic.k3 - centroid.k3,
           analytic.k4 - centroid.k4,
           analytic_uncoupled.k2 - centroid_uncoupled.k2});
    }
    offsets[target + 1] = corrections.size();
  }
}

packed_fallback::PackedJcpFallbackPlan
build_packed_direct_plan(const std::vector<TitanPBE::Tabi::Panel> &panels,
                         const fmm::cartesian::CartesianFmmPlan &plan) {
  std::vector<std::size_t> source_order;
  const auto source_order_bytes = detail::checked_container_allocation(
      plan.source_panel_indices().size(), source_order,
      "allocating Cartesian packed-direct source order");
  if (!source_order_bytes)
    throw std::length_error(source_order_bytes.error());
  source_order.reserve(plan.source_panel_indices().size());
  for (const int raw_panel : plan.source_panel_indices()) {
    if (raw_panel < 0 || static_cast<std::size_t>(raw_panel) >= panels.size())
      throw std::runtime_error(
          "BIE Cartesian full FMM source order is incomplete");
    source_order.push_back(static_cast<std::size_t>(raw_panel));
  }

  std::vector<packed_fallback::PackedFallbackLeafInput> leaves;
  const auto leaf_bytes = detail::checked_container_allocation(
      plan.direct_target_nodes().size(), leaves,
      "allocating Cartesian packed-direct target leaves");
  if (!leaf_bytes)
    throw std::length_error(leaf_bytes.error());
  leaves.reserve(plan.direct_target_nodes().size());
  for (const std::uint32_t target_node_id : plan.direct_target_nodes()) {
    const auto &target_node = plan.nodes()[target_node_id];
    packed_fallback::PackedFallbackLeafInput leaf;
    const auto target_bytes = detail::checked_container_allocation(
        target_node.target_count, leaf.target_panels,
        "allocating Cartesian packed-direct leaf targets");
    if (!target_bytes)
      throw std::length_error(target_bytes.error());
    leaf.target_panels.reserve(target_node.target_count);
    for (std::size_t position = 0; position < target_node.target_count;
         ++position) {
      const int raw_target =
          plan.target_panel_indices()[target_node.target_begin + position];
      if (raw_target < 0 ||
          static_cast<std::size_t>(raw_target) >= panels.size())
        throw std::runtime_error(
            "BIE Cartesian full FMM target order is incomplete");
      leaf.target_panels.push_back(static_cast<std::size_t>(raw_target));
    }
    const std::size_t pair_begin = plan.direct_offsets()[target_node_id];
    const std::size_t pair_end = plan.direct_offsets()[target_node_id + 1];
    const auto span_bytes = detail::checked_container_allocation(
        pair_end - pair_begin, leaf.source_spans,
        "allocating Cartesian packed-direct source spans");
    if (!span_bytes)
      throw std::length_error(span_bytes.error());
    leaf.source_spans.reserve(pair_end - pair_begin);
    for (std::size_t pair_index = pair_begin; pair_index < pair_end;
         ++pair_index) {
      const auto &pair = plan.direct_pairs()[pair_index];
      const auto &source_node = plan.nodes()[pair.source_node];
      leaf.source_spans.push_back(
          {source_node.source_begin, source_node.source_count, true});
    }
    leaves.push_back(std::move(leaf));
  }
  return packed_fallback::build_packed_jcp_fallback_plan(panels, source_order,
                                                         leaves);
}

} // namespace

CartesianFullFmmSolverOperator::CartesianFullFmmSolverOperator(
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    CartesianFullFmmPrepareOptions options,
    std::shared_ptr<const fmm::cartesian::CartesianFmmPlan> plan)
    : panels_(&panels), options_(std::move(options)),
      operator_(std::move(plan)) {
  const auto phi_bytes = detail::checked_container_allocation(
      panels.size(), phi_, "allocating Cartesian full-FMM phi workspace");
  const auto phin_bytes = detail::checked_container_allocation(
      panels.size(), phin_, "allocating Cartesian full-FMM phin workspace");
  if (!phi_bytes || !phin_bytes) {
    throw std::length_error(!phi_bytes ? phi_bytes.error() : phin_bytes.error());
  }
  phi_.resize(panels.size());
  phin_.resize(panels.size());
  stats_.plan = operator_.plan().stats();
  stats_.expansion_order = options_.expansion_order;
  stats_.leaf_size = operator_.plan().options().leaf_size;
  stats_.opening_theta = operator_.plan().options().theta;
  const auto direct_plan_begin = Clock::now();
  node_patch_direct_enabled_ = all_node_patch_panels(panels);
  packed_direct_enabled_ =
      options_.near_integration == "analytic_singular" &&
      options_.quad_buffer == "off" &&
      // Spherical triangle self/one-ring coefficients include curvature,
      // which the centroid packed kernel cannot represent; such panels take
      // the geometry-aware scalar kernel.
      std::none_of(panels.begin(), panels.end(),
                   [](const auto &p) {
                     return static_cast<bool>(p.sphere_patch_quadrature);
                   }) &&
      packed_fallback::packed_jcp_fallback_avx2_available();
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  if (options_.force_scalar_direct_for_testing)
    packed_direct_enabled_ = false;
#endif
  if (packed_direct_enabled_) {
    packed_direct_plan_ = build_packed_direct_plan(panels, operator_.plan());
    const auto packed_metrics = packed_direct_plan_.metrics();
    if (packed_metrics.total_pairs != stats_.plan.direct_panel_pairs) {
      std::ostringstream message;
      message << "BIE Cartesian full FMM packed direct plan coverage mismatch: "
              << "expected_direct_panel_pairs="
              << stats_.plan.direct_panel_pairs
              << " packed_planned_pairs=" << packed_metrics.total_pairs
              << " direct_leaf_pairs=" << stats_.plan.direct_pairs
              << " worklist_hash=" << stats_.plan.worklist_hash;
      throw std::runtime_error(message.str());
    }
    if (node_patch_direct_enabled_) {
      build_node_patch_direct_corrections(
          panels, operator_.plan(), options_.eps_ratio, options_.kappa,
          node_patch_correction_offsets_, node_patch_corrections_);
    }
  }
  stats_.direct_kernel = packed_direct_enabled_
                             ? (node_patch_direct_enabled_
                                    ? "packed-avx2-vector-exp-node-patch-corrected"
                                    : "packed-avx2-vector-exp")
                             : "scalar";
  stats_.node_patch_correction_pairs = node_patch_corrections_.size();
  stats_.direct_plan_seconds = seconds_since(direct_plan_begin);
  const auto packed_metrics = packed_direct_plan_.metrics();
  const auto correction_offset_bytes = detail::checked_container_capacity_bytes(
      node_patch_correction_offsets_,
      "accounting Cartesian node-patch correction offsets");
  const auto correction_bytes = detail::checked_container_capacity_bytes(
      node_patch_corrections_, "accounting Cartesian node-patch corrections");
  if (!correction_offset_bytes || !correction_bytes) {
    throw std::overflow_error(!correction_offset_bytes
                                  ? correction_offset_bytes.error()
                                  : correction_bytes.error());
  }
  const auto metadata_and_offsets = detail::checked_add_or_error(
      packed_metrics.metadata_bytes, *correction_offset_bytes,
      "summing Cartesian packed-direct metadata");
  const auto direct_plan_bytes = metadata_and_offsets
                                     ? detail::checked_add_or_error(
                                           *metadata_and_offsets,
                                           *correction_bytes,
                                           "summing Cartesian packed-direct plan")
                                     : metadata_and_offsets;
  if (!direct_plan_bytes)
    throw std::overflow_error(direct_plan_bytes.error());
  const auto retained_plan_bytes = detail::checked_add_or_error(
      stats_.plan.retained_plan_bytes, *direct_plan_bytes,
      "adding Cartesian packed-direct plan bytes");
  if (!retained_plan_bytes)
    throw std::overflow_error(retained_plan_bytes.error());
  stats_.plan.setup_seconds += stats_.direct_plan_seconds;
  stats_.plan.retained_plan_bytes = *retained_plan_bytes;
}

std::expected<std::unique_ptr<CartesianFullFmmSolverOperator>, std::string>
CartesianFullFmmSolverOperator::prepare(
    const std::vector<TitanPBE::Tabi::Panel> &panels,
    const CartesianFullFmmPrepareOptions &options, TitanExecutor &executor) {
  try {
    std::vector<fmm::cartesian::PanelGeometry> geometry;
    const auto geometry_bytes = detail::checked_container_allocation(
        panels.size(), geometry,
        "allocating Cartesian full-FMM panel geometry");
    if (!geometry_bytes)
      return std::unexpected(geometry_bytes.error());
    geometry.reserve(panels.size());
    for (const TitanPBE::Tabi::Panel &panel : panels) {
      fmm::cartesian::PanelGeometry converted;
      converted.target = {panel.x, panel.y, panel.z};
      converted.source =
          panel.has_source_quadrature_center
              ? fmm::cartesian::Vec3{panel.source_x, panel.source_y,
                                     panel.source_z}
              : converted.target;
      converted.normal = {panel.nx, panel.ny, panel.nz};
      converted.area = panel.area;
      geometry.push_back(converted);
    }
    fmm::cartesian::CartesianFmmPlanOptions plan_options;
    plan_options.expansion_order = options.expansion_order;
    plan_options.leaf_size = options.leaf_size;
    plan_options.max_depth = options.max_depth;
    plan_options.theta = options.theta;
    plan_options.eps_ratio = options.eps_ratio;
    plan_options.kappa = options.kappa;
    auto prepared = fmm::cartesian::prepare_cartesian_fmm_plan(
        geometry, plan_options, executor);
    if (!prepared)
      return std::unexpected(prepared.error());
    return std::unique_ptr<CartesianFullFmmSolverOperator>(
        new CartesianFullFmmSolverOperator(panels, options, *prepared));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        "BIE Cartesian full FMM allocation failed while preparing the operator");
  } catch (const std::length_error &error) {
    return std::unexpected(error.what());
  } catch (const std::overflow_error &error) {
    return std::unexpected(error.what());
  } catch (const std::exception &error) {
    return std::unexpected(error.what());
  }
}

void CartesianFullFmmSolverOperator::apply(const std::vector<double> &unknowns,
                                           double lhs_phi, double lhs_phin,
                                           bool coupled,
                                           TitanExecutor &executor,
                                           std::vector<double> &out,
                                           bool count_apply) {
  if (applying_.test_and_set(std::memory_order_acquire)) {
    throw std::logic_error(
        "BIE Cartesian full FMM concurrent or reentrant apply is unsupported");
  }
  const ApplyGuardReset apply_guard(applying_);
  const auto apply_begin = Clock::now();
  if (panels_ == nullptr)
    throw std::logic_error("BIE Cartesian full FMM panels are unavailable");
  const std::size_t panel_count = panels_->size();
  const auto required_unknowns = detail::checked_multiply_or_error(
      panel_count, coupled ? std::size_t{2} : std::size_t{1},
      "checking Cartesian full-FMM unknown vector size");
  if (!required_unknowns)
    throw std::overflow_error(required_unknowns.error());
  if (unknowns.size() < *required_unknowns)
    throw std::invalid_argument(
        "BIE Cartesian full FMM unknown vector is incomplete");
  for (std::size_t panel = 0; panel < panel_count; ++panel) {
    phi_[panel] = unknowns[panel];
    phin_[panel] = coupled ? unknowns[panel_count + panel] : 0.0;
  }

  fmm::cartesian::CartesianFmmApplyStats apply_stats =
      operator_.apply_far(phi_, phin_, executor);
  auto &interaction_phi = operator_.workspace().interaction_phi();
  auto &interaction_phin = operator_.workspace().interaction_phin();
  const auto direct_begin = Clock::now();
  std::atomic<std::uint64_t> direct_panel_pairs{0};
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  packed_fallback::PackedFallbackPhaseTiming testing_packed_phase{};
  bool testing_packed_phase_available = false;
  double testing_node_patch_seconds = 0.0;
  double testing_adapter_direct_seconds = 0.0;
  double testing_adapter_direct_residual_seconds = 0.0;
  double testing_adapter_direct_phase_sum_seconds = 0.0;
  double testing_adapter_direct_closure_seconds = 0.0;
  double testing_coverage_bookkeeping_seconds = 0.0;
#endif
  const auto &plan = operator_.plan();
  if (packed_direct_enabled_) {
    const auto direct = packed_fallback::accumulate_packed_jcp_fallback(
        packed_direct_plan_, *panels_, phi_, phin_, options_.eps_ratio,
        options_.kappa, coupled, options_.near_integration,
        options_.quad_buffer, "packed-avx2-vector-exp", executor,
        interaction_phi, interaction_phin);
    if (!direct.finite_outputs)
      throw std::runtime_error(
          "BIE Cartesian full FMM packed direct kernel failed");
#if defined(TITAN_CARTESIAN_FMM_TESTING)
    testing_packed_phase = direct.testing_phase_timing;
    testing_packed_phase_available = testing_packed_phase.available;
    if (!testing_packed_phase_available) {
      throw std::runtime_error(
          "P3D0 packed direct timing is unavailable in a testing build");
    }
#endif
    if (node_patch_direct_enabled_) {
#if defined(TITAN_CARTESIAN_FMM_TESTING)
      const auto testing_node_patch_begin = Clock::now();
#endif
      dynamic_tiled_for(panel_count, 64, executor,
                        [&](std::size_t, std::size_t target) {
        double phi_correction = 0.0;
        double phin_correction = 0.0;
        const std::size_t begin = node_patch_correction_offsets_[target];
        const std::size_t end = node_patch_correction_offsets_[target + 1];
        for (std::size_t pos = begin; pos < end; ++pos) {
          const NodePatchDirectCorrection &correction =
              node_patch_corrections_[pos];
          const std::size_t source = correction.source;
          if (coupled) {
            phi_correction =
                std::fma(correction.k1, phin_[source], phi_correction);
            phi_correction =
                std::fma(correction.k2, phi_[source], phi_correction);
            phin_correction =
                std::fma(correction.k3, phin_[source], phin_correction);
            phin_correction =
                std::fma(correction.k4, phi_[source], phin_correction);
          } else {
            phi_correction = std::fma(correction.uncoupled_k2, phi_[source],
                                      phi_correction);
          }
        }
        interaction_phi[target] += phi_correction;
        if (coupled)
          interaction_phin[target] += phin_correction;
      });
#if defined(TITAN_CARTESIAN_FMM_TESTING)
      testing_node_patch_seconds = seconds_since(testing_node_patch_begin);
#endif
    }
    direct_panel_pairs.store(direct.applied_pairs, std::memory_order_relaxed);
  } else {
#if defined(TITAN_CARTESIAN_FMM_TESTING)
    std::atomic<bool> injected_invalid_scalar_pair{false};
#endif
    dynamic_tiled_for(
        plan.direct_target_nodes().size(), 1, executor,
        [&](std::size_t, std::size_t target_group) {
          const std::uint32_t target_node_id =
              plan.direct_target_nodes()[target_group];
          const auto &target_node = plan.nodes()[target_node_id];
          const std::size_t pair_begin = plan.direct_offsets()[target_node_id];
          const std::size_t pair_end =
              plan.direct_offsets()[target_node_id + 1];
          std::uint64_t local_pairs = 0;
          for (std::size_t target_position = 0;
               target_position < target_node.target_count; ++target_position) {
            const int raw_target =
                plan.target_panel_indices()[target_node.target_begin +
                                            target_position];
            if (raw_target < 0 ||
                static_cast<std::size_t>(raw_target) >= panel_count) {
              throw std::runtime_error(
                  "BIE Cartesian full FMM scalar direct target panel index is invalid");
            }
            const std::size_t target = static_cast<std::size_t>(raw_target);
            double phi_sum = 0.0;
            double phin_sum = 0.0;
            for (std::size_t pair_index = pair_begin; pair_index < pair_end;
                 ++pair_index) {
              const auto &pair = plan.direct_pairs()[pair_index];
              const auto &source_node = plan.nodes()[pair.source_node];
              for (std::size_t source_position = 0;
                   source_position < source_node.source_count;
                   ++source_position) {
                const int raw_source =
                    plan.source_panel_indices()[source_node.source_begin +
                                                source_position];
                if (raw_source < 0 ||
                    static_cast<std::size_t>(raw_source) >= panel_count) {
                  throw std::runtime_error(
                      "BIE Cartesian full FMM scalar direct source panel index is invalid");
                }
                const std::size_t source = static_cast<std::size_t>(raw_source);
                const auto kernel =
                    TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
                        (*panels_)[target], target, (*panels_)[source], source,
                        options_.eps_ratio, options_.kappa, coupled,
                        options_.near_integration, options_.quad_buffer,
                        nullptr);
                bool force_invalid = false;
#if defined(TITAN_CARTESIAN_FMM_TESTING)
                force_invalid =
                    options_.inject_invalid_scalar_direct_pair_for_testing &&
                    !injected_invalid_scalar_pair.exchange(
                        true, std::memory_order_relaxed);
#endif
                const bool intentional_zero_self =
                    TitanPBE::Tabi::is_intentional_zero_self_kernel(
                        kernel, coupled, options_.near_integration);
                if ((!kernel.valid && !intentional_zero_self) ||
                    force_invalid) {
                  std::ostringstream message;
                  message << "BIE Cartesian full FMM scalar direct kernel invalid: "
                          << "target_panel=" << target
                          << " source_panel=" << source
                          << " target_node=" << target_node_id
                          << " source_node=" << pair.source_node
                          << " coupled=" << (coupled ? "true" : "false")
                          << " near_integration=" << options_.near_integration
                          << " quad_buffer=" << options_.quad_buffer
                          << " kernel_self_pair="
                          << (kernel.self_pair ? "true" : "false")
                          << " intentional_zero_self="
                          << (intentional_zero_self ? "true" : "false")
                          << " kernel_policy=" << kernel.policy
                          << " forced_test=" << (force_invalid ? "true" : "false");
                  throw std::runtime_error(message.str());
                }
                if (intentional_zero_self)
                  continue;
                if (coupled) {
                  phi_sum +=
                      kernel.k1 * phin_[source] + kernel.k2 * phi_[source];
                  phin_sum +=
                      kernel.k3 * phin_[source] + kernel.k4 * phi_[source];
                } else {
                  phi_sum += kernel.k2 * phi_[source];
                }
              }
              // A kernel/index failure above aborts the apply.  Once the
              // complete source-node span returns, each of its semantic panel
              // pairs has actually been evaluated; count it once per span,
              // not once per kernel call, to keep the scalar fallback hot
              // loop free of coverage-accounting overhead.
              const auto incremented = detail::checked_add_or_error(
                  local_pairs,
                  static_cast<std::uint64_t>(source_node.source_count),
                  "counting scalar direct panel pairs");
              if (!incremented)
                throw std::overflow_error(incremented.error());
              local_pairs = *incremented;
            }
            interaction_phi[target] += phi_sum;
            if (coupled)
              interaction_phin[target] += phin_sum;
          }
          direct_panel_pairs.fetch_add(local_pairs, std::memory_order_relaxed);
        });
  }
  apply_stats.direct_seconds = seconds_since(direct_begin);
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  if (testing_packed_phase_available) {
    testing_adapter_direct_seconds = apply_stats.direct_seconds;
    const double nested_seconds =
        testing_packed_phase.total_seconds + testing_node_patch_seconds;
    const double tolerance =
        testing_timing_tolerance(testing_adapter_direct_seconds);
    if (!testing_timing_value_valid(testing_adapter_direct_seconds) ||
        !testing_timing_value_valid(testing_packed_phase.total_seconds) ||
        !testing_timing_value_valid(testing_node_patch_seconds) ||
        !testing_timing_value_valid(nested_seconds) ||
        nested_seconds > testing_adapter_direct_seconds + tolerance) {
      throw std::runtime_error(
          "P3D0 Cartesian direct phase timing is nonfinite, negative, or exceeds adapter direct total");
    }
    testing_adapter_direct_residual_seconds =
        std::max(0.0, testing_adapter_direct_seconds - nested_seconds);
    testing_adapter_direct_phase_sum_seconds =
        nested_seconds + testing_adapter_direct_residual_seconds;
    testing_adapter_direct_closure_seconds =
        testing_adapter_direct_seconds - testing_adapter_direct_phase_sum_seconds;
    if (!testing_timing_value_valid(testing_adapter_direct_residual_seconds) ||
        !testing_timing_value_valid(testing_adapter_direct_phase_sum_seconds) ||
        !std::isfinite(testing_adapter_direct_closure_seconds) ||
        std::abs(testing_adapter_direct_closure_seconds) > tolerance) {
      throw std::runtime_error(
          "P3D0 Cartesian direct phase timing closure is invalid");
    }
  }
#endif
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  // Start this only at the production coverage bookkeeping seam.  The test-only
  // closure checks above must not be attributed to the coverage phase.
  const auto testing_coverage_begin = Clock::now();
#endif
  apply_stats.expected_direct_panel_pairs = plan.stats().direct_panel_pairs;
  apply_stats.direct_panel_pairs = direct_panel_pairs.load();
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  if (options_.inject_direct_coverage_mismatch_for_testing &&
      apply_stats.direct_panel_pairs > 0) {
    --apply_stats.direct_panel_pairs;
  }
#endif
  if (apply_stats.direct_panel_pairs !=
      apply_stats.expected_direct_panel_pairs) {
    std::ostringstream message;
    message << "BIE Cartesian full FMM direct coverage mismatch: "
            << "expected_direct_panel_pairs="
            << apply_stats.expected_direct_panel_pairs
            << " applied_direct_panel_pairs=" << apply_stats.direct_panel_pairs
            << " direct_kernel=" << stats_.direct_kernel
            << " direct_leaf_pairs=" << plan.stats().direct_pairs
            << " worklist_hash=" << plan.stats().worklist_hash
            << " node_patch_correction_pairs="
            << stats_.node_patch_correction_pairs;
    throw std::runtime_error(message.str());
  }
#if defined(TITAN_CARTESIAN_FMM_TESTING)
  if (testing_packed_phase_available) {
    testing_coverage_bookkeeping_seconds = seconds_since(testing_coverage_begin);
    if (!testing_timing_value_valid(testing_coverage_bookkeeping_seconds)) {
      throw std::runtime_error(
          "P3D0 Cartesian direct coverage bookkeeping timing is invalid");
    }
  }
#endif

  const auto assemble_begin = Clock::now();
  const auto output_bytes = detail::checked_container_allocation(
      *required_unknowns, out, "allocating Cartesian full-FMM output vector");
  if (!output_bytes)
    throw std::length_error(output_bytes.error());
  out.assign(*required_unknowns, 0.0);
  bool finite = apply_stats.finite_outputs;
  for (std::size_t panel = 0; panel < panel_count; ++panel) {
    out[panel] = TitanPBE::Tabi::jcp_lhs_phi_for_panel(
                     (*panels_)[panel], options_.eps_ratio, lhs_phi) *
                     unknowns[panel] -
                 interaction_phi[panel];
    finite = finite && std::isfinite(out[panel]);
    if (coupled) {
      out[panel_count + panel] =
          TitanPBE::Tabi::jcp_lhs_phin_for_panel((*panels_)[panel],
                                                 options_.eps_ratio, lhs_phin) *
              unknowns[panel_count + panel] -
          interaction_phin[panel];
      finite = finite && std::isfinite(out[panel_count + panel]);
    }
  }
  apply_stats.assemble_seconds = seconds_since(assemble_begin);
  const auto phi_workspace_bytes = detail::checked_container_capacity_bytes(
      phi_, "accounting Cartesian full-FMM phi workspace");
  const auto phin_workspace_bytes = detail::checked_container_capacity_bytes(
      phin_, "accounting Cartesian full-FMM phin workspace");
  if (!phi_workspace_bytes || !phin_workspace_bytes) {
    throw std::overflow_error(!phi_workspace_bytes ? phi_workspace_bytes.error()
                                                    : phin_workspace_bytes.error());
  }
  const auto density_workspace_bytes = detail::checked_add_or_error(
      *phi_workspace_bytes, *phin_workspace_bytes,
      "summing Cartesian full-FMM density workspaces");
  if (!density_workspace_bytes)
    throw std::overflow_error(density_workspace_bytes.error());
  const auto retained_workspace_bytes = detail::checked_add_or_error(
      operator_.workspace().retained_bytes(), *density_workspace_bytes,
      "summing Cartesian full-FMM retained workspace");
  if (!retained_workspace_bytes)
    throw std::overflow_error(retained_workspace_bytes.error());
  apply_stats.retained_workspace_bytes = *retained_workspace_bytes;
  apply_stats.finite_outputs = finite;
  apply_stats.apply_seconds = seconds_since(apply_begin);
  if (!finite)
    throw std::runtime_error(
        "BIE Cartesian full FMM produced non-finite output");
  stats_.last_apply = apply_stats;
  stats_.finite_outputs = stats_.finite_outputs && finite;
  if (count_apply) {
    if (stats_.apply_calls == std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error(
          "BIE Cartesian full FMM apply-call counter overflow");
    }
    ++stats_.apply_calls;
    accumulate_apply_stats(stats_.total_apply, apply_stats);
#if defined(TITAN_CARTESIAN_FMM_TESTING)
    if (testing_packed_phase_available) {
      CartesianFullFmmDirectPhaseTiming &timing =
          stats_.testing_direct_phase_timing;
      if (!timing.available) {
        if (stats_.apply_calls != 1) {
          throw std::runtime_error(
              "P3D0 Cartesian direct timing began after counted applies");
        }
        timing.available = true;
        timing.finite = true;
      }
      if (timing.counted_apply_calls + 1 != stats_.apply_calls) {
        throw std::runtime_error(
            "P3D0 Cartesian direct timing apply-count drift");
      }
      testing_add_timing_value(
          timing.packed_unknown_cache_reorder_seconds,
          testing_packed_phase.unknown_cache_reorder_seconds,
          "packed_unknown_cache_reorder");
      testing_add_timing_value(timing.packed_worker_scratch_prep_seconds,
                                testing_packed_phase.worker_scratch_prep_seconds,
                                "packed_worker_scratch_prep");
      testing_add_timing_value(timing.packed_parallel_region_seconds,
                                testing_packed_phase.parallel_region_seconds,
                                "packed_parallel_region");
      testing_add_timing_value(
          timing.packed_worker_metric_reduction_seconds,
          testing_packed_phase.worker_metric_reduction_seconds,
          "packed_worker_metric_reduction");
      testing_add_timing_value(timing.packed_finite_scan_seconds,
                                testing_packed_phase.finite_scan_seconds,
                                "packed_finite_scan");
      testing_add_timing_value(timing.packed_wrapper_residual_seconds,
                                testing_packed_phase.wrapper_residual_seconds,
                                "packed_wrapper_residual");
      testing_add_timing_value(timing.packed_phase_sum_seconds,
                                testing_packed_phase.phase_sum_seconds,
                                "packed_phase_sum");
      testing_add_timing_value(timing.packed_total_seconds,
                                testing_packed_phase.total_seconds,
                                "packed_total");
      testing_add_signed_timing_residual(
          timing.packed_closure_seconds,
          testing_packed_phase.closure_seconds, "packed_closure");
      testing_add_timing_value(timing.node_patch_correction_seconds,
                                testing_node_patch_seconds,
                                "node_patch_correction");
      testing_add_timing_value(timing.adapter_direct_total_seconds,
                                testing_adapter_direct_seconds,
                                "adapter_direct_total");
      testing_add_timing_value(timing.adapter_direct_residual_seconds,
                                testing_adapter_direct_residual_seconds,
                                "adapter_direct_residual");
      testing_add_timing_value(timing.adapter_coverage_bookkeeping_seconds,
                                testing_coverage_bookkeeping_seconds,
                                "adapter_coverage_bookkeeping");
      testing_add_timing_value(
          timing.adapter_residual_coverage_seconds,
          testing_adapter_direct_residual_seconds +
              testing_coverage_bookkeeping_seconds,
          "adapter_residual_coverage");
      testing_add_timing_value(timing.adapter_direct_phase_sum_seconds,
                                testing_adapter_direct_phase_sum_seconds,
                                "adapter_direct_phase_sum");
      testing_add_signed_timing_residual(
          timing.adapter_direct_closure_seconds,
          testing_adapter_direct_closure_seconds, "adapter_direct_closure");
      timing.counted_apply_calls = stats_.apply_calls;
      timing.expected_direct_panel_pairs =
          stats_.total_apply.expected_direct_panel_pairs;
      timing.applied_direct_panel_pairs = stats_.total_apply.direct_panel_pairs;
      timing.finite = timing.finite && apply_stats.finite_outputs;
      if (!timing.finite ||
          timing.expected_direct_panel_pairs != timing.applied_direct_panel_pairs ||
          timing.expected_direct_panel_pairs == 0 ||
          timing.adapter_direct_total_seconds !=
              stats_.total_apply.direct_seconds) {
        throw std::runtime_error(
            "P3D0 Cartesian direct timing aggregate invariant failed");
      }
    }
#endif
  }
}

#if defined(TITAN_CARTESIAN_FMM_TESTING)
titan::tabi::fmm_m2l::PackedFallbackPlanInspection
CartesianFullFmmSolverOperator::packed_direct_plan_inspection_for_testing()
    const {
  return packed_direct_enabled_
             ? packed_direct_plan_.inspection_for_testing()
             : titan::tabi::fmm_m2l::PackedFallbackPlanInspection{};
}

void CartesianFullFmmSolverOperator::hold_apply_guard_for_testing() {
  if (applying_.test_and_set(std::memory_order_acquire)) {
    throw std::logic_error(
        "BIE Cartesian full FMM test apply guard was already held");
  }
}

void CartesianFullFmmSolverOperator::release_apply_guard_for_testing() noexcept {
  applying_.clear(std::memory_order_release);
}
#endif

} // namespace titan::bie

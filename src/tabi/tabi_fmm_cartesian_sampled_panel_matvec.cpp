#include "tabi_fmm_cartesian_sampled_panel_matvec.h"

#include "tabi_fmm_m2l_prototype.h"
#include "tabi_near.h"
#include "titan_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace titan::tabi::fmm_kifmm {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start,
                       const Clock::time_point end) noexcept {
  return std::chrono::duration<double>(end - start).count();
}

std::uint64_t mix_hash(std::uint64_t hash, std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
  return hash;
}

void hash_double(std::uint64_t &hash, double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  hash = mix_hash(hash, bits);
}

std::uint64_t sampled_target_hash(
    std::span<const std::size_t> indices) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  hash ^= static_cast<std::uint64_t>(indices.size());
  hash *= 1099511628211ull;
  for (const std::size_t index : indices) {
    hash ^= static_cast<std::uint64_t>(index);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::uint64_t policy_hash(
    const CartesianSampledPanelMatvecConfig &config,
    std::uint64_t target_hash) noexcept {
  std::uint64_t hash = mix_hash(0xcbf29ce484222325ull, target_hash);
  hash = mix_hash(hash, static_cast<std::uint64_t>(config.order));
  hash_double(hash, config.theta);
  hash_double(hash, config.tail_tol);
  hash_double(hash, config.strict_geometry_theta);
  hash_double(hash, config.eps_ratio);
  hash_double(hash, config.kappa);
  hash_double(hash, config.max_aggregate_relative_rms);
  hash_double(hash, config.max_channel_relative_rms);
  hash = mix_hash(hash, config.cache_budget_mb);
  hash = mix_hash(hash, config.build_direct_reference ? 1ull : 0ull);
  for (const char ch : config.near_integration)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  for (const char ch : config.quad_buffer)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  for (const char ch : config.fallback_kernel)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  for (const char ch : config.refine_mode)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  for (const char ch : config.cache_mode)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  for (const char ch : config.plan_builder)
    hash = mix_hash(hash, static_cast<unsigned char>(ch));
  return hash;
}

bool valid_unknowns(std::span<const double> unknowns) noexcept {
  return std::all_of(unknowns.begin(), unknowns.end(), [](double value) {
    return std::isfinite(value);
  });
}

bool valid_target_indices(std::span<const std::size_t> indices,
                          std::size_t panel_count) noexcept {
  if (indices.empty())
    return false;
  std::vector<std::size_t> sorted(indices.begin(), indices.end());
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] >= panel_count)
      return false;
    if (i > 0 && sorted[i] == sorted[i - 1])
      return false;
  }
  return true;
}

bool valid_panel(const TitanPBE::Tabi::Panel &panel) noexcept {
  return std::isfinite(panel.x) && std::isfinite(panel.y) &&
         std::isfinite(panel.z) && std::isfinite(panel.nx) &&
         std::isfinite(panel.ny) && std::isfinite(panel.nz) &&
         std::isfinite(panel.area) && panel.area > 0.0;
}

bool finite_kifmm_rows(std::span<const KifmmJcpChannelRow> rows) noexcept {
  return std::all_of(rows.begin(), rows.end(), [](const auto &row) {
    return std::isfinite(row.k1) && std::isfinite(row.k2) &&
           std::isfinite(row.k3) && std::isfinite(row.k4);
  });
}

std::vector<titan::tabi::fmm_m2l::PrototypePanel>
to_prototype_panels(std::span<const TitanPBE::Tabi::Panel> panels) {
  std::vector<titan::tabi::fmm_m2l::PrototypePanel> out;
  out.reserve(panels.size());
  for (const TitanPBE::Tabi::Panel &panel : panels) {
    titan::tabi::fmm_m2l::PrototypePanel converted;
    converted.x = panel.x;
    converted.y = panel.y;
    converted.z = panel.z;
    converted.nx = panel.nx;
    converted.ny = panel.ny;
    converted.nz = panel.nz;
    converted.area = panel.area;
    converted.scalar_density = 0.0;
    out.push_back(converted);
  }
  return out;
}

struct ChannelRow {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
};

KifmmJcpChannelRow to_kifmm_row(const ChannelRow &row) noexcept {
  return {row.k1, row.k2, row.k3, row.k4};
}

struct Row {
  double phi{0.0};
  double phin{0.0};
};

struct DirectStats {
  std::size_t valid_pairs{0};
  std::size_t invalid_pairs{0};
};

Row row_from_channel(const ChannelRow &row) noexcept {
  return Row{row.k1 + row.k2, row.k3 + row.k4};
}

ChannelRow direct_channel_row(
    const TitanPBE::Tabi::Panel &target, std::size_t target_index,
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns, double eps_ratio, double kappa,
    const std::string &near_integration, const std::string &quad_buffer,
    DirectStats &stats) {
  const std::size_t n = panels.size();
  ChannelRow out;
  for (std::size_t source = 0; source < n; ++source) {
    const TitanPBE::Tabi::JcpNearPairKernel kernel =
        TitanPBE::Tabi::evaluate_jcp_near_pair_kernel(
            target, target_index, panels[source], source, eps_ratio, kappa,
            true, near_integration, quad_buffer, nullptr);
    if (!kernel.valid) {
      ++stats.invalid_pairs;
      continue;
    }
    ++stats.valid_pairs;
    out.k1 += kernel.k1 * unknowns[n + source];
    out.k2 += kernel.k2 * unknowns[source];
    out.k3 += kernel.k3 * unknowns[n + source];
    out.k4 += kernel.k4 * unknowns[source];
  }
  return out;
}

double rel_rms_rows(std::span<const Row> approx,
                    std::span<const Row> reference) noexcept {
  double err2 = 0.0;
  double ref2 = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double dphi = approx[i].phi - reference[i].phi;
    const double dphin = approx[i].phin - reference[i].phin;
    err2 += dphi * dphi + dphin * dphin;
    ref2 += reference[i].phi * reference[i].phi +
            reference[i].phin * reference[i].phin;
  }
  return std::sqrt(err2 / std::max(ref2, 1.0e-30));
}

double channel_value(const ChannelRow &row, int channel) noexcept {
  if (channel == 1)
    return row.k1;
  if (channel == 2)
    return row.k2;
  if (channel == 3)
    return row.k3;
  return row.k4;
}

double rel_rms_channel(std::span<const ChannelRow> approx,
                       std::span<const ChannelRow> reference,
                       int channel) noexcept {
  double err2 = 0.0;
  double ref2 = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double d = channel_value(approx[i], channel) -
                     channel_value(reference[i], channel);
    const double r = channel_value(reference[i], channel);
    err2 += d * d;
    ref2 += r * r;
  }
  return std::sqrt(err2 / std::max(ref2, 1.0e-30));
}

std::pair<double, double>
abs_and_reference_rms_channel(std::span<const ChannelRow> approx,
                              std::span<const ChannelRow> reference,
                              int channel) noexcept {
  double err2 = 0.0;
  double ref2 = 0.0;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const double d = channel_value(approx[i], channel) -
                     channel_value(reference[i], channel);
    const double r = channel_value(reference[i], channel);
    err2 += d * d;
    ref2 += r * r;
  }
  const double denom =
      static_cast<double>(std::max<std::size_t>(approx.size(), 1));
  return {std::sqrt(err2 / denom), std::sqrt(ref2 / denom)};
}

CartesianSampledPanelMatvecResult failed(std::string status,
                                         int order = 0) {
  CartesianSampledPanelMatvecResult result;
  result.status = std::move(status);
  result.execution_status = result.status;
  result.accuracy_status = "not_evaluated";
  result.accuracy_failure = "not_evaluated";
  result.order = order;
  result.finite = false;
  return result;
}

} // namespace

CartesianSampledPanelMatvecResult run_guarded_cartesian_p4_sampled_panel_matvec(
    std::span<const TitanPBE::Tabi::Panel> panels,
    std::span<const double> unknowns,
    std::span<const std::size_t> sampled_target_indices,
    const CartesianSampledPanelMatvecConfig &config) {
  const auto total_start = Clock::now();
  if (panels.empty())
    return failed("empty_cloud", config.order);
  if (config.order < 4 || config.order > 5)
    return failed("unsupported_order", config.order);
  if (unknowns.size() != 2 * panels.size() || !valid_unknowns(unknowns))
    return failed("invalid_unknowns", config.order);
  if (!valid_target_indices(sampled_target_indices, panels.size()))
    return failed("invalid_target_indices", config.order);
  if (config.near_integration != "analytic_singular" ||
      config.quad_buffer != "off")
    return failed("unsupported_near_policy", config.order);
  if (!std::isfinite(config.theta) || config.theta <= 0.0 ||
      !std::isfinite(config.tail_tol) || config.tail_tol <= 0.0 ||
      !std::isfinite(config.strict_geometry_theta) ||
      config.strict_geometry_theta <= 0.0 ||
      !std::isfinite(config.eps_ratio) || config.eps_ratio <= 0.0 ||
      !std::isfinite(config.kappa) || config.kappa < 0.0 ||
      !std::isfinite(config.max_aggregate_relative_rms) ||
      config.max_aggregate_relative_rms <= 0.0 ||
      !std::isfinite(config.max_channel_relative_rms) ||
      config.max_channel_relative_rms <= 0.0)
    return failed("invalid_config", config.order);
  if (config.cache_mode != "green-jet-compressed" &&
      config.cache_mode != "green-jet-tiled")
    return failed("unsupported_cache_mode", config.order);
  for (const TitanPBE::Tabi::Panel &panel : panels) {
    if (!valid_panel(panel))
      return failed("invalid_panel", config.order);
  }
  const std::vector<std::size_t> sampled_targets(sampled_target_indices.begin(),
                                                 sampled_target_indices.end());

  CartesianSampledPanelMatvecResult result;
  result.order = config.order;
  result.source_panels = panels.size();
  result.sampled_rows = sampled_target_indices.size();
  result.timing_scope = "sampled_rows_only";
  result.pair_count_scope = "sampled_rows_only";
  result.sampled_target_hash = sampled_target_hash(sampled_target_indices);
  result.policy_hash = policy_hash(config, result.sampled_target_hash);

  auto finish_failure = [&](std::string status) {
    result.status = std::move(status);
    result.execution_status = result.status;
    result.accuracy_status = "not_evaluated";
    result.accuracy_failure = "not_evaluated";
    result.finite = false;
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  };

  std::vector<TitanPBE::Tabi::Panel> production_panels(panels.begin(),
                                                       panels.end());
  std::vector<double> phi(unknowns.begin(),
                          unknowns.begin() +
                              static_cast<std::ptrdiff_t>(panels.size()));
  std::vector<double> phin(unknowns.begin() +
                               static_cast<std::ptrdiff_t>(panels.size()),
                           unknowns.end());
  std::vector<double> zero(panels.size(), 0.0);

  std::vector<ChannelRow> direct_channels(sampled_target_indices.size());
  std::vector<Row> direct_rows(sampled_target_indices.size());
  DirectStats direct_stats;
  const auto direct_start = Clock::now();
  if (config.build_direct_reference) {
    for (std::size_t local = 0; local < sampled_target_indices.size();
         ++local) {
      const std::size_t target = sampled_target_indices[local];
      direct_channels[local] = direct_channel_row(
          production_panels[target], target, panels, unknowns, config.eps_ratio,
          config.kappa, config.near_integration, config.quad_buffer,
          direct_stats);
      direct_rows[local] = row_from_channel(direct_channels[local]);
    }
  }
  result.direct_reference_seconds =
      elapsed_seconds(direct_start, Clock::now());
  result.production_direct_reference_valid_pairs = direct_stats.valid_pairs;
  result.invalid_pairs = direct_stats.invalid_pairs;
  if (result.invalid_pairs != 0)
    return finish_failure("invalid_production_near_kernel");

  titan::tabi::fmm_m2l::PrototypeConfig prototype_config;
  prototype_config.expansion_order = config.order;
  prototype_config.max_leaf_size = 64;
  prototype_config.separation_theta = config.theta;
  prototype_config.threads = 1;
  const auto proto_panels = to_prototype_panels(panels);
  const auto setup_start = Clock::now();
  titan::tabi::fmm_m2l::FmmM2lAnalyticContext context(
      prototype_config, proto_panels, config.eps_ratio, config.kappa);
  TitanExecutor executor(1);
  result.setup_seconds = elapsed_seconds(setup_start, Clock::now());

  const auto apply_start = Clock::now();
  const titan::tabi::fmm_m2l::JcpLocalCartesianCandidateResult phin_only =
      context.apply_local_cartesian_candidate(
          zero, phin, executor, true, config.near_integration,
          config.quad_buffer, &production_panels, config.refine_mode,
          config.tail_tol, config.cache_mode, config.cache_budget_mb,
          config.fallback_kernel, config.plan_builder, 4, "rademacher",
          "strict-geometry-mac", config.strict_geometry_theta,
          &sampled_targets);
  const titan::tabi::fmm_m2l::JcpLocalCartesianCandidateResult phi_only =
      context.apply_local_cartesian_candidate(
          phi, zero, executor, true, config.near_integration,
          config.quad_buffer, &production_panels, config.refine_mode,
          config.tail_tol, config.cache_mode, config.cache_budget_mb,
          config.fallback_kernel, config.plan_builder, 4, "rademacher",
          "strict-geometry-mac", config.strict_geometry_theta,
          &sampled_targets);
  result.apply_seconds = elapsed_seconds(apply_start, Clock::now());
  if (!phin_only.finite_outputs || !phi_only.finite_outputs)
    return finish_failure("cartesian_p4_nonfinite");

  result.fallback_kernel_effective = phin_only.fallback_effective;
  if (phi_only.fallback_effective != phin_only.fallback_effective)
    result.fallback_kernel_effective = "mixed";
  result.far_panel_pairs =
      std::max(phin_only.far_panel_pairs, phi_only.far_panel_pairs);
  result.linf_max_abs_tail =
      std::max(phin_only.linf_max_abs_tail, phi_only.linf_max_abs_tail);
  result.linf_max_relative_tail = std::max(phin_only.linf_max_relative_tail,
                                           phi_only.linf_max_relative_tail);
  result.geometry_guard_rejected_cells =
      std::max(phin_only.geometry_guard_rejected_cells,
               phi_only.geometry_guard_rejected_cells);
  result.geometry_guard_direct_fallback_pairs =
      std::max(phin_only.geometry_guard_direct_fallback_pairs,
               phi_only.geometry_guard_direct_fallback_pairs);
  result.geometry_ratio_max =
      std::max(phin_only.geometry_ratio_max, phi_only.geometry_ratio_max);
  result.fallback_refined_pairs =
      std::max(phin_only.fallback_refined_pairs, phi_only.fallback_refined_pairs);
  result.fallback_base_near_pairs =
      std::max(phin_only.fallback_base_near_pairs,
               phi_only.fallback_base_near_pairs);
  result.refined_cells =
      std::max(phin_only.refined_cells, phi_only.refined_cells);
  result.split_children =
      std::max(phin_only.split_children, phi_only.split_children);
  const auto fallback_pairs = [](const titan::tabi::fmm_m2l::
                                     JcpLocalCartesianCandidateResult &candidate) {
    const std::size_t packed_pairs =
        candidate.fallback_base_near_pairs + candidate.fallback_refined_pairs;
    return packed_pairs > 0 ? packed_pairs : candidate.direct_fallback_pairs;
  };
  result.fallback_panel_pairs =
      std::max(fallback_pairs(phin_only), fallback_pairs(phi_only));

  std::vector<ChannelRow> approx_channels(sampled_target_indices.size());
  std::vector<Row> approx_rows(sampled_target_indices.size());
  for (std::size_t local = 0; local < sampled_target_indices.size(); ++local) {
    const std::size_t target = sampled_target_indices[local];
    ChannelRow row;
    row.k1 = phin_only.phi[target];
    row.k2 = phi_only.phi[target];
    row.k3 = phin_only.phin[target];
    row.k4 = phi_only.phin[target];
    approx_channels[local] = row;
    approx_rows[local] = row_from_channel(row);
  }

  result.approximate_rows.resize(approx_channels.size());
  result.direct_rows.resize(direct_channels.size());
  for (std::size_t i = 0; i < approx_channels.size(); ++i) {
    result.approximate_rows[i] = to_kifmm_row(approx_channels[i]);
    result.direct_rows[i] = to_kifmm_row(direct_channels[i]);
  }
  if (!config.build_direct_reference) {
    result.aggregate_relative_rms =
        std::numeric_limits<double>::quiet_NaN();
    result.k1_relative_rms = std::numeric_limits<double>::quiet_NaN();
    result.k2_relative_rms = std::numeric_limits<double>::quiet_NaN();
    result.k3_relative_rms = std::numeric_limits<double>::quiet_NaN();
    result.k4_relative_rms = std::numeric_limits<double>::quiet_NaN();
    result.max_channel_relative_rms =
        std::numeric_limits<double>::quiet_NaN();
    result.max_channel_name = "none";
    result.k4_abs_rms = std::numeric_limits<double>::quiet_NaN();
    result.k4_reference_rms = std::numeric_limits<double>::quiet_NaN();
    result.finite = finite_kifmm_rows(result.approximate_rows);
    result.status = result.finite ? "pass" : "non_finite";
    result.execution_status = result.finite ? "pass" : "non_finite";
    result.accuracy_status = "not_evaluated";
    result.accuracy_failure = "direct_reference_disabled";
    result.total_seconds = elapsed_seconds(total_start, Clock::now());
    return result;
  }

  result.aggregate_relative_rms = rel_rms_rows(approx_rows, direct_rows);
  result.k1_relative_rms = rel_rms_channel(approx_channels, direct_channels, 1);
  result.k2_relative_rms = rel_rms_channel(approx_channels, direct_channels, 2);
  result.k3_relative_rms = rel_rms_channel(approx_channels, direct_channels, 3);
  result.k4_relative_rms = rel_rms_channel(approx_channels, direct_channels, 4);
  result.max_channel_relative_rms = result.k1_relative_rms;
  result.max_channel_name = "K1";
  if (result.k2_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k2_relative_rms;
    result.max_channel_name = "K2";
  }
  if (result.k3_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k3_relative_rms;
    result.max_channel_name = "K3";
  }
  if (result.k4_relative_rms > result.max_channel_relative_rms) {
    result.max_channel_relative_rms = result.k4_relative_rms;
    result.max_channel_name = "K4";
  }
  const auto [k4_abs, k4_ref] =
      abs_and_reference_rms_channel(approx_channels, direct_channels, 4);
  result.k4_abs_rms = k4_abs;
  result.k4_reference_rms = k4_ref;
  result.finite = std::isfinite(result.aggregate_relative_rms) &&
                  std::isfinite(result.k1_relative_rms) &&
                  std::isfinite(result.k2_relative_rms) &&
                  std::isfinite(result.k3_relative_rms) &&
                  std::isfinite(result.k4_relative_rms) &&
                  std::isfinite(result.max_channel_relative_rms) &&
                  std::isfinite(result.k4_abs_rms) &&
                  std::isfinite(result.k4_reference_rms);
  if (!result.finite) {
    result.status = "non_finite";
    result.execution_status = "non_finite";
    result.accuracy_status = "not_evaluated";
    result.accuracy_failure = "not_evaluated";
  } else {
    result.execution_status = "pass";
    const bool aggregate_gap =
        result.aggregate_relative_rms > config.max_aggregate_relative_rms;
    const bool channel_gap =
        result.max_channel_relative_rms > config.max_channel_relative_rms;
    if (!aggregate_gap && !channel_gap) {
      result.status = "pass";
      result.accuracy_status = "pass";
      result.accuracy_failure = "none";
    } else {
      result.status = "accuracy_gap";
      result.accuracy_status = "fail";
      if (aggregate_gap && channel_gap) {
        result.accuracy_failure = "aggregate_and_channel_gap";
      } else if (aggregate_gap) {
        result.accuracy_failure = "aggregate_gap";
      } else {
        result.accuracy_failure = "channel_gap";
      }
    }
  }
  result.total_seconds = elapsed_seconds(total_start, Clock::now());
  return result;
}

} // namespace titan::tabi::fmm_kifmm

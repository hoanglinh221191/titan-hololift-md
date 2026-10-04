#include "tabi_fmm_kifmm_cartesian_bridge.h"

#include "tabi_treecode_internal.h"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace titan::tabi::fmm_kifmm {
namespace {

using titan::tabi::fmm_m2l::CartesianJcpLocalP4;
using titan::tabi::fmm_m2l::CartesianJcpMoments;
using titan::tabi::fmm_m2l::JcpLocalField;
using titan::tabi::fmm_m2l::PrototypePanel;
using TitanPBE::Tabi::detail::MomentIndex;
using TitanPBE::Tabi::detail::NodeMoments;
using TitanPBE::Tabi::detail::TaylorSeries;

const std::array<KifmmFieldFamily, 4> kFamilies{
    KifmmFieldFamily::PhinCharge, KifmmFieldFamily::PhiNormalX,
    KifmmFieldFamily::PhiNormalY, KifmmFieldFamily::PhiNormalZ};

struct Rms {
  double err2{0.0};
  double ref2{0.0};
  double max_abs{0.0};

  void add(double approx, double ref) noexcept {
    const double diff = approx - ref;
    err2 += diff * diff;
    ref2 += ref * ref;
    max_abs = std::max(max_abs, std::abs(diff));
  }

  double rel() const noexcept {
    return std::sqrt(err2 / std::max(ref2, std::numeric_limits<double>::min()));
  }

  double abs_rms(std::size_t count) const noexcept {
    return std::sqrt(err2 / std::max<std::size_t>(1, count));
  }

  double ref_rms(std::size_t count) const noexcept {
    return std::sqrt(ref2 / std::max<std::size_t>(1, count));
  }
};

std::uint64_t mix_hash(std::uint64_t seed, std::uint64_t value) noexcept {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  return seed;
}

std::uint64_t hash_double(std::uint64_t seed, double value) noexcept {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return mix_hash(seed, bits);
}

std::uint64_t policy_hash(const KifmmCartesianBridgeConfig &config) noexcept {
  std::uint64_t h = 1469598103934665603ULL;
  h = mix_hash(h, static_cast<std::uint64_t>(config.surface_order));
  h = mix_hash(h, static_cast<std::uint64_t>(config.cartesian_order));
  h = hash_double(h, config.equiv_scale);
  h = hash_double(h, config.upward_check_scale);
  h = hash_double(h, config.local_equiv_scale);
  h = hash_double(h, config.local_check_scale);
  h = hash_double(h, config.kappa);
  h = hash_double(h, config.eps_ratio);
  return h;
}

int field_component(KifmmFieldFamily family) noexcept {
  switch (family) {
  case KifmmFieldFamily::PhiNormalX:
    return 0;
  case KifmmFieldFamily::PhiNormalY:
    return 1;
  case KifmmFieldFamily::PhiNormalZ:
    return 2;
  case KifmmFieldFamily::PhinCharge:
    return -1;
  }
  return -1;
}

NodeMoments to_node_moments(const std::array<double, titan::tabi::fmm_m2l::kCartesianJcpMomentMax> &values) {
  NodeMoments result;
  std::copy(values.begin(), values.end(), result.values.begin());
  return result;
}

std::size_t series_index(int ax, int ay, int az) noexcept {
  constexpr int side = TitanPBE::Tabi::detail::kSeriesSide;
  return static_cast<std::size_t>((ax * side + ay) * side + az);
}

double moment_field_value(const CartesianJcpMoments &moments,
                          KifmmFieldFamily family, KifmmKernel kernel,
                          const KifmmBox &box, Vec3 target,
                          double kappa) {
  const int order = moments.order;
  const double kk = kernel == KifmmKernel::Yukawa ? kappa : 0.0;
  if (family == KifmmFieldFamily::PhinCharge) {
    const NodeMoments phin = to_node_moments(moments.phin);
    const TaylorSeries series = TitanPBE::Tabi::detail::green_series(
        target.x, target.y, target.z, box.center.x, box.center.y,
        box.center.z, kk, order);
    return TitanPBE::Tabi::detail::taylor_sum_series(phin, order, series);
  }
  const int component = field_component(family);
  const NodeMoments normal =
      to_node_moments(moments.phi_normal[static_cast<std::size_t>(component)]);
  const TaylorSeries series =
      TitanPBE::Tabi::detail::source_normal_component_green_series(
          target.x, target.y, target.z, box.center.x, box.center.y,
          box.center.z, component, kk, order);
  return TitanPBE::Tabi::detail::taylor_sum_series(normal, order, series);
}

Eigen::MatrixXd build_m2e_rhs(const KifmmEquivalentSystem &system,
                              const CartesianJcpMoments &moments,
                              KifmmKernel kernel, const KifmmBox &box,
                              double kappa) {
  Eigen::MatrixXd rhs(static_cast<Eigen::Index>(system.check_points.size()), 4);
  for (std::size_t i = 0; i < system.check_points.size(); ++i) {
    const Vec3 target = system.check_points[i].x;
    for (std::size_t c = 0; c < kFamilies.size(); ++c) {
      rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(c)) =
          moment_field_value(moments, kFamilies[c], kernel, box, target,
                             kappa);
    }
  }
  return rhs;
}

void assign_batch(KifmmJcpChannels &channels, KifmmKernel kernel,
                  const std::vector<KifmmEquivalentSolveResult> &solves) {
  if (solves.size() != 4) {
    channels.status = "m2e_batch_size_mismatch";
    channels.finite = false;
    return;
  }
  if (kernel == KifmmKernel::Laplace) {
    channels.phin_laplace = solves[0];
    for (std::size_t i = 0; i < 3; ++i)
      channels.phi_normal_laplace[i] = solves[i + 1];
  } else {
    channels.phin_yukawa = solves[0];
    for (std::size_t i = 0; i < 3; ++i)
      channels.phi_normal_yukawa[i] = solves[i + 1];
  }
}

void update_channel_stats(KifmmJcpChannels &channels,
                          const std::vector<KifmmEquivalentSolveResult> &solves) {
  channels.equivalent_rhs_solves += solves.size();
  for (const auto &solve : solves) {
    channels.max_relative_residual =
        std::max(channels.max_relative_residual, solve.relative_residual);
    channels.max_condition_estimate =
        std::max(channels.max_condition_estimate, solve.condition_estimate);
    channels.finite = channels.finite && solve.finite;
  }
}

void add_series_to_field(std::array<double, titan::tabi::fmm_m2l::kCartesianLocalTermMax> &field,
                         const TaylorSeries &series, double scale,
                         int order) {
  const auto &indices = TitanPBE::Tabi::detail::moment_indices(order);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const MomentIndex &idx = indices[i];
    field[i] += scale * series.coeff[series_index(idx.ax, idx.ay, idx.az)];
  }
}

void add_equivalent_to_scalar_field(
    std::array<double, titan::tabi::fmm_m2l::kCartesianLocalTermMax> &field,
    const KifmmSurfaceRule &rule, const KifmmBox &box,
    const KifmmEquivalentSolveResult &charges, KifmmKernel kernel,
    double kappa, double scale, int order) {
  const std::vector<KifmmSurfacePoint> points =
      instantiate_surface(rule.unit_equiv, box);
  const double kk = kernel == KifmmKernel::Yukawa ? kappa : 0.0;
  const std::size_t n = std::min(points.size(), charges.charges.size());
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3 source = points[i].x;
    const TaylorSeries series = TitanPBE::Tabi::detail::green_series(
        source.x, source.y, source.z, box.center.x, box.center.y,
        box.center.z, kk, order);
    add_series_to_field(field, series, scale * charges.charges[i], order);
  }
}

std::vector<PrototypePanel> prototype_panels_from_sources(
    std::span<const KifmmPanelSource> sources) {
  std::vector<PrototypePanel> panels;
  panels.reserve(sources.size());
  for (const auto &source : sources) {
    PrototypePanel panel;
    panel.x = source.position.x;
    panel.y = source.position.y;
    panel.z = source.position.z;
    panel.nx = source.normal.x;
    panel.ny = source.normal.y;
    panel.nz = source.normal.z;
    panel.area = source.area;
    panel.scalar_density = 1.0;
    panels.push_back(panel);
  }
  return panels;
}

std::vector<double> phi_values(std::span<const KifmmPanelSource> sources) {
  std::vector<double> values;
  values.reserve(sources.size());
  for (const auto &source : sources)
    values.push_back(source.phi);
  return values;
}

std::vector<double> phin_values(std::span<const KifmmPanelSource> sources) {
  std::vector<double> values;
  values.reserve(sources.size());
  for (const auto &source : sources)
    values.push_back(source.phin);
  return values;
}

CartesianJcpMoments build_moments(std::span<const KifmmPanelSource> sources,
                                  const KifmmBox &box, int order) {
  std::vector<std::size_t> indices(sources.size());
  std::iota(indices.begin(), indices.end(), 0);
  return titan::tabi::fmm_m2l::build_cartesian_jcp_moments(
      prototype_panels_from_sources(sources), phi_values(sources),
      phin_values(sources), indices, box.center.x, box.center.y,
      box.center.z, order);
}

std::vector<KifmmPanelSource> source_fixture() {
  return {
      {{-0.18, -0.07, 0.09}, {0.804, -0.321, 0.500}, 0.72, 0.61, -0.37},
      {{0.15, 0.11, -0.04}, {-0.426, 0.853, 0.301}, 0.68, -0.42, 0.55},
      {{0.04, -0.16, -0.12}, {0.287, 0.095, 0.953}, 0.64, 0.33, 0.27},
      {{-0.09, 0.18, 0.05}, {-0.700, -0.140, 0.700}, 0.59, -0.21, -0.46},
      {{0.22, -0.02, 0.14}, {0.500, 0.500, -0.707}, 0.53, 0.47, 0.18},
      {{-0.21, 0.03, -0.15}, {-0.190, 0.950, -0.247}, 0.61, -0.39, 0.34},
  };
}

std::vector<Vec3> target_points() {
  return {{5.94, 0.04, -0.03}, {6.05, -0.05, 0.06}, {5.97, 0.07, 0.03},
          {6.08, 0.01, -0.07}, {5.92, -0.08, -0.04}, {6.04, 0.06, 0.08}};
}

std::vector<Vec3> target_normals() {
  return {{0.0, 0.0, 1.0}, {0.577, 0.577, 0.577}, {-0.408, 0.816, 0.408},
          {0.707, -0.707, 0.0}, {-0.302, -0.302, 0.905}, {0.196, 0.784, -0.588}};
}

std::vector<PrototypePanel> prototype_targets(std::span<const Vec3> targets,
                                              std::span<const Vec3> normals) {
  std::vector<PrototypePanel> panels;
  panels.reserve(targets.size());
  for (std::size_t i = 0; i < targets.size(); ++i) {
    PrototypePanel panel;
    panel.x = targets[i].x;
    panel.y = targets[i].y;
    panel.z = targets[i].z;
    panel.nx = normals[i].x;
    panel.ny = normals[i].y;
    panel.nz = normals[i].z;
    panel.area = 1.0;
    panels.push_back(panel);
  }
  return panels;
}

std::vector<KifmmJcpChannelRow>
direct_rows(std::span<const KifmmPanelSource> sources,
            std::span<const Vec3> targets, std::span<const Vec3> normals,
            const KifmmCartesianBridgeConfig &config) {
  std::vector<KifmmJcpChannelRow> rows;
  rows.reserve(targets.size());
  const double inv_eps = 1.0 / std::max(1.0e-12, config.eps_ratio);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    rows.push_back(assemble_direct_jcp_channel_row(
        sources, config.kappa, config.eps_ratio, inv_eps, targets[i],
        normals[i]));
  }
  return rows;
}

KifmmCartesianBridgeRow compare_channel_rows(
    std::string row_id, const std::vector<KifmmJcpChannelRow> &approx,
    const std::vector<KifmmJcpChannelRow> &direct,
    const KifmmCartesianBridgeConfig &config) {
  KifmmCartesianBridgeRow row;
  row.row_id = std::move(row_id);
  row.execution_status = "pass";
  row.surface_order = config.surface_order;
  row.cartesian_order = config.cartesian_order;
  row.policy_hash = policy_hash(config);
  if (approx.size() != direct.size() || approx.empty()) {
    row.execution_status = "fail";
    row.accuracy_status = "not_evaluated";
    row.accuracy_failure = "row_size_mismatch";
    return row;
  }
  std::array<Rms, 4> channel{};
  Rms aggregate;
  for (std::size_t i = 0; i < approx.size(); ++i) {
    const std::array<double, 4> a{approx[i].k1, approx[i].k2, approx[i].k3,
                                  approx[i].k4};
    const std::array<double, 4> d{direct[i].k1, direct[i].k2, direct[i].k3,
                                  direct[i].k4};
    for (std::size_t c = 0; c < 4; ++c) {
      channel[c].add(a[c], d[c]);
      aggregate.add(a[c], d[c]);
    }
  }
  row.aggregate_relative_rms = aggregate.rel();
  row.k1_relative_rms = channel[0].rel();
  row.k2_relative_rms = channel[1].rel();
  row.k3_relative_rms = channel[2].rel();
  row.k4_relative_rms = channel[3].rel();
  row.k4_abs_rms = channel[3].abs_rms(approx.size());
  row.k4_reference_rms = channel[3].ref_rms(approx.size());
  const std::array<double, 4> rel{row.k1_relative_rms, row.k2_relative_rms,
                                  row.k3_relative_rms, row.k4_relative_rms};
  const std::array<const char *, 4> names{"K1", "K2", "K3", "K4"};
  const auto max_it = std::max_element(rel.begin(), rel.end());
  const std::size_t max_index =
      static_cast<std::size_t>(std::distance(rel.begin(), max_it));
  row.max_channel_relative_rms = *max_it;
  row.max_channel_name = names[max_index];
  row.l2p_target_gradient_relative_rms =
      std::max(row.k3_relative_rms, row.k4_relative_rms);
  row.l2p_gradient_method = "cartesian_l2p_order";
  row.accuracy_status =
      row.aggregate_relative_rms <= config.aggregate_relative_rms_gate &&
              row.max_channel_relative_rms <= config.max_channel_relative_rms_gate
          ? "pass"
          : "fail";
  if (row.accuracy_status == "fail") {
    row.accuracy_failure =
        row.aggregate_relative_rms > config.aggregate_relative_rms_gate
            ? "aggregate_and_channel"
            : "channel";
  }
  return row;
}

std::vector<KifmmJcpChannelRow>
evaluate_m2e_direct_to_targets(const KifmmSurfaceRule &rule,
                               const KifmmBox &box,
                               const KifmmJcpChannels &channels,
                               std::span<const Vec3> targets,
                               std::span<const Vec3> normals,
                               const KifmmCartesianBridgeConfig &config) {
  std::vector<KifmmJcpChannelRow> rows;
  rows.reserve(targets.size());
  const double inv_eps = 1.0 / std::max(1.0e-12, config.eps_ratio);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    rows.push_back(assemble_kifmm_jcp_channel_row(
        rule, box, channels, config.kappa, config.eps_ratio, inv_eps,
        targets[i], normals[i]));
  }
  return rows;
}

std::vector<KifmmJcpChannelRow>
evaluate_cartesian_local_rows(const KifmmBox &box, const CartesianJcpLocalP4 &local,
                              std::span<const Vec3> targets,
                              std::span<const Vec3> normals,
                              const KifmmCartesianBridgeConfig &config) {
  std::vector<KifmmJcpChannelRow> rows;
  const std::vector<PrototypePanel> panels = prototype_targets(targets, normals);
  rows.reserve(panels.size());
  for (const auto &panel : panels) {
    const auto c = titan::tabi::fmm_m2l::evaluate_cartesian_jcp_l2p_order(
        panel, box.center.x, box.center.y, box.center.z, local,
        config.cartesian_order);
    rows.push_back({c[0], c[1], c[2], c[3]});
  }
  return rows;
}

KifmmJcpChannels solve_local_channels_from_sources(
    const KifmmSurfaceRule &local_rule, const KifmmBox &target_box,
    std::span<const KifmmPanelSource> sources,
    const KifmmCartesianBridgeConfig &config,
    KifmmEquivalentSystemCache *cache) {
  KifmmJcpChannels channels;
  channels.finite = true;
  channels.status = "pass";
  auto solve_kernel = [&](KifmmKernel kernel, double kappa) {
    KifmmEquivalentSystem system = build_equivalent_system_cached(
        *cache, local_rule, target_box, kernel, kappa, config.solver,
        config.condition_policy);
    Eigen::MatrixXd rhs(static_cast<Eigen::Index>(system.check_points.size()), 4);
    for (std::size_t i = 0; i < system.check_points.size(); ++i) {
      const Vec3 target = system.check_points[i].x;
      for (std::size_t c = 0; c < kFamilies.size(); ++c) {
        rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(c)) =
            direct_field_potential(sources, kFamilies[c], kernel, kappa,
                                   target);
      }
    }
    std::vector<KifmmEquivalentSolveResult> solves =
        solve_equivalent_rhs_batch(system, rhs, config.solver);
    assign_batch(channels, kernel, solves);
    update_channel_stats(channels, solves);
  };
  solve_kernel(KifmmKernel::Laplace, 0.0);
  solve_kernel(KifmmKernel::Yukawa, config.kappa);
  channels.equivalent_factor_cache_hits = cache->stats.hits;
  channels.equivalent_factor_cache_misses = cache->stats.misses;
  return channels;
}

KifmmJcpChannels solve_local_channels_from_upward_channels(
    const KifmmSurfaceRule &upward_rule, const KifmmBox &source_box,
    const KifmmJcpChannels &upward_channels,
    const KifmmSurfaceRule &local_rule, const KifmmBox &target_box,
    const KifmmCartesianBridgeConfig &config,
    KifmmEquivalentSystemCache *cache) {
  KifmmJcpChannels channels;
  channels.finite = true;
  channels.status = "pass";
  auto solve_kernel = [&](KifmmKernel kernel, double kappa) {
    KifmmEquivalentSystem system = build_equivalent_system_cached(
        *cache, local_rule, target_box, kernel, kappa, config.solver,
        config.condition_policy);
    Eigen::MatrixXd rhs(static_cast<Eigen::Index>(system.check_points.size()), 4);
    for (std::size_t i = 0; i < system.check_points.size(); ++i) {
      const Vec3 target = system.check_points[i].x;
      if (kernel == KifmmKernel::Laplace) {
        rhs(static_cast<Eigen::Index>(i), 0) = equivalent_field_potential(
            upward_rule, source_box, upward_channels.phin_laplace,
            KifmmFieldFamily::PhinCharge, KifmmKernel::Laplace, 0.0, target);
        for (std::size_t c = 0; c < 3; ++c) {
          rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(c + 1)) =
              equivalent_field_potential(
                  upward_rule, source_box,
                  upward_channels.phi_normal_laplace[c],
                  kFamilies[c + 1], KifmmKernel::Laplace, 0.0, target);
        }
      } else {
        rhs(static_cast<Eigen::Index>(i), 0) = equivalent_field_potential(
            upward_rule, source_box, upward_channels.phin_yukawa,
            KifmmFieldFamily::PhinCharge, KifmmKernel::Yukawa, config.kappa,
            target);
        for (std::size_t c = 0; c < 3; ++c) {
          rhs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(c + 1)) =
              equivalent_field_potential(
                  upward_rule, source_box,
                  upward_channels.phi_normal_yukawa[c],
                  kFamilies[c + 1], KifmmKernel::Yukawa, config.kappa, target);
        }
      }
    }
    std::vector<KifmmEquivalentSolveResult> solves =
        solve_equivalent_rhs_batch(system, rhs, config.solver);
    assign_batch(channels, kernel, solves);
    update_channel_stats(channels, solves);
  };
  solve_kernel(KifmmKernel::Laplace, 0.0);
  solve_kernel(KifmmKernel::Yukawa, config.kappa);
  channels.equivalent_factor_cache_hits = cache->stats.hits;
  channels.equivalent_factor_cache_misses = cache->stats.misses;
  return channels;
}

[[maybe_unused]] std::vector<KifmmPanelSource> equivalent_sources_from_channels(
    const KifmmSurfaceRule &rule, const KifmmBox &box,
    const KifmmJcpChannels &channels) {
  const std::vector<KifmmSurfacePoint> points =
      instantiate_surface(rule.unit_equiv, box);
  std::vector<KifmmPanelSource> sources;
  sources.reserve(points.size() * 8);
  auto append = [&](const KifmmEquivalentSolveResult &solve,
                    KifmmFieldFamily family) {
    for (std::size_t i = 0; i < points.size() && i < solve.charges.size(); ++i) {
      KifmmPanelSource source;
      source.position = points[i].x;
      source.normal = points[i].outward_normal;
      source.area = 1.0;
      if (family == KifmmFieldFamily::PhinCharge) {
        source.phin = solve.charges[i];
      } else {
        const int component = field_component(family);
        source.phi = solve.charges[i];
        source.normal = {component == 0 ? 1.0 : 0.0,
                         component == 1 ? 1.0 : 0.0,
                         component == 2 ? 1.0 : 0.0};
      }
      sources.push_back(source);
    }
  };
  append(channels.phin_laplace, KifmmFieldFamily::PhinCharge);
  append(channels.phin_yukawa, KifmmFieldFamily::PhinCharge);
  for (std::size_t i = 0; i < 3; ++i) {
    append(channels.phi_normal_laplace[i],
           static_cast<KifmmFieldFamily>(static_cast<int>(KifmmFieldFamily::PhiNormalX) + static_cast<int>(i)));
    append(channels.phi_normal_yukawa[i],
           static_cast<KifmmFieldFamily>(static_cast<int>(KifmmFieldFamily::PhiNormalX) + static_cast<int>(i)));
  }
  return sources;
}

KifmmCartesianBridgeRow scalar_m2e_row(const KifmmCartesianBridgeConfig &config,
                                      const KifmmSurfaceRule &upward_rule,
                                      const KifmmBox &source_box,
                                      const CartesianJcpMoments &moments,
                                      std::span<const KifmmPanelSource> sources,
                                      std::span<const Vec3> targets) {
  KifmmEquivalentSystemCache cache;
  KifmmJcpChannels channels = bridge_cartesian_moments_to_equivalents(
      upward_rule, source_box, moments, config, &cache);
  std::vector<KifmmScalarSource> scalar_sources;
  scalar_sources.reserve(sources.size());
  for (const auto &source : sources)
    scalar_sources.push_back({source.position, source.area * source.phin});
  const std::vector<KifmmSurfacePoint> points =
      instantiate_surface(upward_rule.unit_equiv, source_box);
  std::vector<KifmmScalarSource> equiv_sources;
  for (std::size_t i = 0; i < points.size() &&
                          i < channels.phin_laplace.charges.size();
       ++i) {
    equiv_sources.push_back({points[i].x, channels.phin_laplace.charges[i]});
  }
  Rms rms;
  for (const Vec3 target : targets) {
    rms.add(direct_scalar_potential(equiv_sources, KifmmKernel::Laplace, 0.0,
                                    target),
            direct_scalar_potential(scalar_sources, KifmmKernel::Laplace, 0.0,
                                    target));
  }
  KifmmCartesianBridgeRow row;
  row.row_id = "scalar_m2e_laplace_q8";
  row.execution_status = channels.finite ? "pass" : "fail";
  row.aggregate_relative_rms = rms.rel();
  row.k1_relative_rms = row.aggregate_relative_rms;
  row.max_channel_relative_rms = row.aggregate_relative_rms;
  row.max_channel_name = "scalar";
  row.m2e_max_relative_residual = channels.phin_laplace.relative_residual;
  row.surface_order = config.surface_order;
  row.cartesian_order = config.cartesian_order;
  row.policy_hash = policy_hash(config);
  row.accuracy_status = row.aggregate_relative_rms <= 1.0e-8 ? "pass" : "fail";
  row.accuracy_failure = row.accuracy_status == "pass" ? "none" : "scalar_m2e";
  return row;
}

KifmmCartesianBridgeRow scalar_e2l_row(const KifmmCartesianBridgeConfig &config,
                                      const KifmmSurfaceRule &local_rule,
                                      const KifmmBox &target_box,
                                      std::span<const KifmmPanelSource> sources,
                                      std::span<const Vec3> targets) {
  KifmmEquivalentSystemCache cache;
  KifmmJcpChannels local_channels = solve_local_channels_from_sources(
      local_rule, target_box, sources, config, &cache);
  CartesianJcpLocalP4 local;
  add_equivalent_to_scalar_field(
      local.fields[static_cast<std::size_t>(JcpLocalField::K1)], local_rule,
      target_box, local_channels.phin_laplace, KifmmKernel::Laplace, 0.0, 1.0,
      config.cartesian_order);
  Rms potential;
  Rms gradient;
  for (const Vec3 target : targets) {
    const double approx = titan::tabi::fmm_m2l::evaluate_cartesian_jcp_local_scalar(
        local, JcpLocalField::K1, target.x - target_box.center.x,
        target.y - target_box.center.y, target.z - target_box.center.z);
    std::vector<KifmmScalarSource> scalar_sources;
    for (const auto &source : sources)
      scalar_sources.push_back({source.position, source.area * source.phin});
    const double direct =
        direct_scalar_potential(scalar_sources, KifmmKernel::Laplace, 0.0, target);
    potential.add(approx, direct);
    const Vec3 direct_grad =
        direct_scalar_grad_target(scalar_sources, KifmmKernel::Laplace, 0.0, target);
    const auto local_grad =
        titan::tabi::fmm_m2l::evaluate_cartesian_jcp_local_gradient(
            local, JcpLocalField::K1, target.x - target_box.center.x,
            target.y - target_box.center.y, target.z - target_box.center.z,
            config.cartesian_order);
    gradient.add(local_grad[0], direct_grad.x);
  }
  KifmmCartesianBridgeRow row;
  row.row_id = "scalar_e2l_laplace_q8";
  row.execution_status = local.finite_outputs ? "pass" : "fail";
  row.aggregate_relative_rms = potential.rel();
  row.k1_relative_rms = potential.rel();
  row.k3_relative_rms = gradient.rel();
  row.max_channel_relative_rms = std::max(potential.rel(), gradient.rel());
  row.max_channel_name = gradient.rel() > potential.rel() ? "grad_x" : "potential";
  row.m2e_max_relative_residual = local_channels.max_relative_residual;
  row.l2p_gradient_method = "analytic_cartesian_local";
  row.l2p_target_gradient_relative_rms = gradient.rel();
  row.surface_order = config.surface_order;
  row.cartesian_order = config.cartesian_order;
  row.policy_hash = policy_hash(config);
  row.accuracy_status =
      row.aggregate_relative_rms <= 1.0e-8 && row.l2p_target_gradient_relative_rms <= 1.0e-8
          ? "pass"
          : "fail";
  row.accuracy_failure = row.accuracy_status == "pass" ? "none" : "scalar_e2l";
  return row;
}

} // namespace

KifmmJcpChannels bridge_cartesian_moments_to_equivalents(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &source_box,
    const CartesianJcpMoments &moments,
    const KifmmCartesianBridgeConfig &config,
    KifmmEquivalentSystemCache *system_cache) {
  KifmmEquivalentSystemCache local_cache;
  KifmmEquivalentSystemCache &cache =
      system_cache != nullptr ? *system_cache : local_cache;
  KifmmJcpChannels channels;
  channels.finite = true;
  channels.status = "pass";
  auto solve_kernel = [&](KifmmKernel kernel, double kappa) {
    KifmmEquivalentSystem system = build_equivalent_system_cached(
        cache, surface_rule, source_box, kernel, kappa, config.solver,
        config.condition_policy);
    const Eigen::MatrixXd rhs =
        build_m2e_rhs(system, moments, kernel, source_box, config.kappa);
    std::vector<KifmmEquivalentSolveResult> solves =
        solve_equivalent_rhs_batch(system, rhs, config.solver);
    assign_batch(channels, kernel, solves);
    update_channel_stats(channels, solves);
  };
  solve_kernel(KifmmKernel::Laplace, 0.0);
  solve_kernel(KifmmKernel::Yukawa, config.kappa);
  channels.equivalent_factor_cache_hits = cache.stats.hits;
  channels.equivalent_factor_cache_misses = cache.stats.misses;
  return channels;
}

CartesianJcpLocalP4 bridge_equivalent_channels_to_cartesian_local(
    const KifmmSurfaceRule &local_surface_rule, const KifmmBox &target_box,
    const KifmmJcpChannels &channels,
    const KifmmCartesianBridgeConfig &config) {
  CartesianJcpLocalP4 local;
  const double inv_eps = 1.0 / std::max(1.0e-12, config.eps_ratio);
  const int order = config.cartesian_order;
  add_equivalent_to_scalar_field(
      local.fields[static_cast<std::size_t>(JcpLocalField::K1)],
      local_surface_rule, target_box, channels.phin_laplace,
      KifmmKernel::Laplace, 0.0, 1.0, order);
  add_equivalent_to_scalar_field(
      local.fields[static_cast<std::size_t>(JcpLocalField::K1)],
      local_surface_rule, target_box, channels.phin_yukawa, KifmmKernel::Yukawa,
      config.kappa, -1.0, order);
  add_equivalent_to_scalar_field(
      local.fields[static_cast<std::size_t>(JcpLocalField::K3Potential)],
      local_surface_rule, target_box, channels.phin_laplace,
      KifmmKernel::Laplace, 0.0, 1.0, order);
  add_equivalent_to_scalar_field(
      local.fields[static_cast<std::size_t>(JcpLocalField::K3Potential)],
      local_surface_rule, target_box, channels.phin_yukawa, KifmmKernel::Yukawa,
      config.kappa, -inv_eps, order);
  for (std::size_t component = 0; component < 3; ++component) {
    add_equivalent_to_scalar_field(
        local.fields[static_cast<std::size_t>(JcpLocalField::K2)],
        local_surface_rule, target_box, channels.phi_normal_yukawa[component],
        KifmmKernel::Yukawa, config.kappa, config.eps_ratio, order);
    add_equivalent_to_scalar_field(
        local.fields[static_cast<std::size_t>(JcpLocalField::K2)],
        local_surface_rule, target_box, channels.phi_normal_laplace[component],
        KifmmKernel::Laplace, 0.0, -1.0, order);
    add_equivalent_to_scalar_field(
        local.fields[static_cast<std::size_t>(JcpLocalField::K4Potential)],
        local_surface_rule, target_box, channels.phi_normal_laplace[component],
        KifmmKernel::Laplace, 0.0, 1.0, order);
    add_equivalent_to_scalar_field(
        local.fields[static_cast<std::size_t>(JcpLocalField::K4Potential)],
        local_surface_rule, target_box, channels.phi_normal_yukawa[component],
        KifmmKernel::Yukawa, config.kappa, -1.0, order);
  }
  return local;
}

KifmmCartesianSplitLocalBridgeResult
bridge_equivalent_channels_to_cartesian_local_split_k4(
    const KifmmSurfaceRule &local_surface_rule, const KifmmBox &target_box,
    const KifmmJcpChannels &channels,
    const KifmmCartesianBridgeConfig &config) {
  KifmmCartesianSplitLocalBridgeResult result;
  result.combined_local = bridge_equivalent_channels_to_cartesian_local(
      local_surface_rule, target_box, channels, config);
  result.local_equiv_max_relative_residual =
      channels.max_relative_residual;
  const int order = config.cartesian_order;
  for (std::size_t component = 0; component < 3; ++component) {
    add_equivalent_to_scalar_field(
        result.k4_laplace_local
            .fields[static_cast<std::size_t>(JcpLocalField::K4Potential)],
        local_surface_rule, target_box,
        channels.phi_normal_laplace[component], KifmmKernel::Laplace, 0.0,
        1.0, order);
    add_equivalent_to_scalar_field(
        result.k4_yukawa_local
            .fields[static_cast<std::size_t>(JcpLocalField::K4Potential)],
        local_surface_rule, target_box,
        channels.phi_normal_yukawa[component], KifmmKernel::Yukawa,
        config.kappa, 1.0, order);
  }
  result.finite_outputs = result.combined_local.finite_outputs &&
                          result.k4_laplace_local.finite_outputs &&
                          result.k4_yukawa_local.finite_outputs;
  result.status = result.finite_outputs ? "pass" : "nonfinite_split_local";
  return result;
}

KifmmCartesianBridgeSuiteResult run_cartesian_kifmm_bridge_smoke_suite() {
  KifmmCartesianBridgeConfig config;
  config.surface_order = 8;
  config.cartesian_order = 5;
  const KifmmBox source_box{{0.0, 0.0, 0.0}, 1.0, 0, 1};
  const KifmmBox target_box{{6.0, 0.0, 0.0}, 1.0, 0, 2};
  const KifmmSurfaceRule upward_rule = make_cube_surface_rule(
      config.surface_order, config.equiv_scale, config.upward_check_scale);
  const KifmmSurfaceRule local_rule = make_cube_local_surface_rule(
      config.surface_order, config.local_equiv_scale,
      config.local_check_scale);
  const std::vector<KifmmPanelSource> sources = source_fixture();
  const std::vector<Vec3> targets = target_points();
  const std::vector<Vec3> normals = target_normals();
  const CartesianJcpMoments moments =
      build_moments(sources, source_box, config.cartesian_order);
  const std::vector<KifmmJcpChannelRow> direct =
      direct_rows(sources, targets, normals, config);

  KifmmCartesianBridgeSuiteResult suite;
  suite.required_rows = 5;
  suite.rows.push_back(
      scalar_m2e_row(config, upward_rule, source_box, moments, sources, targets));
  suite.rows.push_back(
      scalar_e2l_row(config, local_rule, target_box, sources, targets));

  KifmmEquivalentSystemCache m2e_cache;
  KifmmJcpChannels upward_channels = bridge_cartesian_moments_to_equivalents(
      upward_rule, source_box, moments, config, &m2e_cache);
  auto m2e_rows = evaluate_m2e_direct_to_targets(
      upward_rule, source_box, upward_channels, targets, normals, config);
  auto row = compare_channel_rows("jcp_m2e_eight_field_q8", m2e_rows, direct,
                                  config);
  row.m2e_max_relative_residual = upward_channels.max_relative_residual;
  suite.rows.push_back(row);

  KifmmEquivalentSystemCache local_cache;
  KifmmJcpChannels local_channels = solve_local_channels_from_sources(
      local_rule, target_box, sources, config, &local_cache);
  CartesianJcpLocalP4 local = bridge_equivalent_channels_to_cartesian_local(
      local_rule, target_box, local_channels, config);
  auto e2l_rows =
      evaluate_cartesian_local_rows(target_box, local, targets, normals, config);
  row = compare_channel_rows("jcp_e2l_eight_field_q8", e2l_rows, direct, config);
  row.m2e_max_relative_residual = local_channels.max_relative_residual;
  suite.rows.push_back(row);

  KifmmEquivalentSystemCache hybrid_cache;
  KifmmJcpChannels hybrid_local_channels =
      solve_local_channels_from_upward_channels(upward_rule, source_box,
                                                upward_channels, local_rule,
                                                target_box, config,
                                                &hybrid_cache);
  CartesianJcpLocalP4 hybrid_local =
      bridge_equivalent_channels_to_cartesian_local(
          local_rule, target_box, hybrid_local_channels, config);
  auto hybrid_rows = evaluate_cartesian_local_rows(target_box, hybrid_local,
                                                   targets, normals, config);
  row = compare_channel_rows("jcp_hybrid_box_q8", hybrid_rows, direct, config);
  row.m2e_max_relative_residual =
      std::max(upward_channels.max_relative_residual,
               hybrid_local_channels.max_relative_residual);
  suite.rows.push_back(row);

  bool execution_ok = true;
  bool accuracy_ok = true;
  bool e2l_gap = false;
  bool m2e_gap = false;
  bool k4_gap = false;
  for (const auto &r : suite.rows) {
    execution_ok = execution_ok && r.execution_status == "pass";
    accuracy_ok = accuracy_ok && r.accuracy_status == "pass";
    if (r.row_id.find("m2e") != std::string::npos &&
        r.accuracy_status != "pass")
      m2e_gap = true;
    if (r.row_id.find("e2l") != std::string::npos &&
        r.accuracy_status != "pass")
      e2l_gap = true;
    if (r.k4_relative_rms > config.max_channel_relative_rms_gate)
      k4_gap = true;
  }
  if (!execution_ok) {
    suite.decision = "fmm_e26_execution_gap";
  } else if (accuracy_ok) {
    suite.decision = "fmm_e26_synthetic_single_box_hybrid_bridge_pass";
  } else if (m2e_gap) {
    suite.decision = "fmm_e26_m2e_gap";
  } else if (e2l_gap) {
    suite.decision = "fmm_e26_e2l_gap";
  } else if (k4_gap) {
    suite.decision = "fmm_e26_k4_bridge_gap";
  } else {
    suite.decision = "fmm_e26_execution_gap";
  }
  return suite;
}

} // namespace titan::tabi::fmm_kifmm

#include "tabi_fmm_kifmm_fields.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace titan::tabi::fmm_kifmm {
namespace {

constexpr double kNormalUnitTolerance = 1.0e-6;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

Vec3 scale(Vec3 v, double s) noexcept {
  return {v.x * s, v.y * s, v.z * s};
}

double dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(Vec3 v) noexcept {
  return std::sqrt(dot(v, v));
}

double distance(Vec3 a, Vec3 b) noexcept {
  return norm({a.x - b.x, a.y - b.y, a.z - b.z});
}

void kahan_add(double value, double &sum, double &c) noexcept {
  const double y = value - c;
  const double t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

void kahan_add_vec(Vec3 value, Vec3 &sum, Vec3 &c) noexcept {
  kahan_add(value.x, sum.x, c.x);
  kahan_add(value.y, sum.y, c.y);
  kahan_add(value.z, sum.z, c.z);
}

int family_component_index(KifmmFieldFamily family) noexcept {
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

double component(Vec3 v, KifmmFieldFamily family) noexcept {
  switch (family) {
  case KifmmFieldFamily::PhiNormalX:
    return v.x;
  case KifmmFieldFamily::PhiNormalY:
    return v.y;
  case KifmmFieldFamily::PhiNormalZ:
    return v.z;
  case KifmmFieldFamily::PhinCharge:
    return 0.0;
  }
  return 0.0;
}

Vec3 grad_source(KifmmKernel kernel, Vec3 target, Vec3 source,
                 double kappa) noexcept {
  switch (kernel) {
  case KifmmKernel::Laplace:
    return grad_source_laplace_G(target, source);
  case KifmmKernel::Yukawa:
    return grad_source_yukawa_G(target, source, kappa);
  }
  return {};
}

bool valid_panel_source(const KifmmPanelSource &source) noexcept {
  if (!finite_vec(source.position) || !finite_vec(source.normal) ||
      !std::isfinite(source.area) || !std::isfinite(source.phi) ||
      !std::isfinite(source.phin) || !(source.area > 0.0)) {
    return false;
  }
  const double n = norm(source.normal);
  return n > 0.0 && std::isfinite(n) &&
         std::abs(n - 1.0) <= kNormalUnitTolerance;
}

bool valid_panel_sources(std::span<const KifmmPanelSource> sources) noexcept {
  for (const auto &source : sources) {
    if (!valid_panel_source(source)) {
      return false;
    }
  }
  return true;
}

bool point_inside_scaled_box(Vec3 p, const KifmmBox &box,
                             double scale_value) noexcept {
  if (!finite_vec(p) || !finite_vec(box.center) || !(box.half_width > 0.0) ||
      !std::isfinite(scale_value) || !(scale_value > 0.0)) {
    return false;
  }
  const double h = box.half_width * scale_value;
  return std::abs(p.x - box.center.x) <= h &&
         std::abs(p.y - box.center.y) <= h &&
         std::abs(p.z - box.center.z) <= h;
}

KifmmEquivalentSolveResult failed_result(
    const std::string &solver, const KifmmEquivalentSystem &system,
    std::string status,
    KifmmEquivalentSolveDiagnostics diagnostics = {}) {
  KifmmEquivalentSolveResult result;
  result.solver = solver;
  result.equivalent_points = system.equiv_points.size();
  result.check_points = system.check_points.size();
  result.finite = false;
  result.status = std::move(status);
  result.diagnostics = std::move(diagnostics);
  result.condition_estimate = result.diagnostics.condition_estimate;
  result.rank = result.diagnostics.rank;
  return result;
}

void update_panel_source_distances(KifmmEquivalentSolveDiagnostics &diag,
                                   const KifmmEquivalentSystem &system,
                                   std::span<const KifmmPanelSource> sources) {
  diag.min_source_equiv_distance = std::numeric_limits<double>::infinity();
  diag.min_source_check_distance = std::numeric_limits<double>::infinity();
  for (const auto &source : sources) {
    for (const auto &equiv : system.equiv_points) {
      diag.min_source_equiv_distance = std::min(
          diag.min_source_equiv_distance, distance(source.position, equiv.x));
    }
    for (const auto &check : system.check_points) {
      diag.min_source_check_distance = std::min(
          diag.min_source_check_distance, distance(source.position, check.x));
    }
  }
  if (!std::isfinite(diag.min_source_equiv_distance)) {
    diag.min_source_equiv_distance = 0.0;
  }
  if (!std::isfinite(diag.min_source_check_distance)) {
    diag.min_source_check_distance = 0.0;
  }
}

Eigen::VectorXd build_field_rhs(const KifmmEquivalentSystem &system,
                                std::span<const KifmmPanelSource> sources,
                                KifmmFieldFamily family,
                                KifmmKernel kernel, double kappa) {
  Eigen::VectorXd rhs(static_cast<Eigen::Index>(system.check_points.size()));
  for (std::size_t i = 0; i < system.check_points.size(); ++i) {
    rhs[static_cast<Eigen::Index>(i)] = direct_field_potential(
        sources, family, kernel, kappa, system.check_points[i].x);
  }
  return rhs;
}

Eigen::MatrixXd build_jcp_rhs_matrix(const KifmmEquivalentSystem &system,
                                     std::span<const KifmmPanelSource> sources,
                                     KifmmKernel kernel, double kappa) {
  Eigen::MatrixXd rhs(static_cast<Eigen::Index>(system.check_points.size()), 4);
  const std::array<KifmmFieldFamily, 4> families{
      KifmmFieldFamily::PhinCharge, KifmmFieldFamily::PhiNormalX,
      KifmmFieldFamily::PhiNormalY, KifmmFieldFamily::PhiNormalZ};
  for (std::size_t col = 0; col < families.size(); ++col) {
    for (std::size_t row = 0; row < system.check_points.size(); ++row) {
      rhs(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          direct_field_potential(sources, families[col], kernel, kappa,
                                 system.check_points[row].x);
    }
  }
  return rhs;
}

KifmmEquivalentSolveResult solve_field_checked(
    const KifmmEquivalentSystem &system, const KifmmBox &box,
    const KifmmSurfaceRule &surface_rule,
    std::span<const KifmmPanelSource> sources, KifmmFieldFamily family,
    KifmmKernel kernel, double kappa, const std::string &solver) {
  KifmmEquivalentSolveDiagnostics diag = system.diagnostics;
  update_panel_source_distances(diag, system, sources);
  if (!system.finite) {
    return failed_result(solver, system, system.status, diag);
  }
  if (!valid_panel_sources(sources)) {
    return failed_result(solver, system, "invalid_panel_source", diag);
  }
  for (const auto &source : sources) {
    if (!point_inside_scaled_box(source.position, box, 1.0)) {
      return failed_result(solver, system, "source_outside_box", diag);
    }
    if (!point_inside_scaled_box(source.position, box,
                                 surface_rule.equiv_scale)) {
      return failed_result(solver, system, "source_outside_equiv", diag);
    }
  }
  const Eigen::VectorXd rhs =
      build_field_rhs(system, sources, family, kernel, kappa);
  KifmmEquivalentSolveResult result = solve_equivalent_rhs(system, rhs, solver);
  result.diagnostics.min_source_equiv_distance =
      diag.min_source_equiv_distance;
  result.diagnostics.min_source_check_distance = diag.min_source_check_distance;
  return result;
}

double source_normal_field_sum(const std::array<double, 3> &values) noexcept {
  return values[0] + values[1] + values[2];
}

std::array<double, 3>
direct_normal_potential_components(std::span<const KifmmPanelSource> sources,
                                   KifmmKernel kernel, double kappa,
                                   Vec3 target) {
  return {direct_field_potential(sources, KifmmFieldFamily::PhiNormalX, kernel,
                                 kappa, target),
          direct_field_potential(sources, KifmmFieldFamily::PhiNormalY, kernel,
                                 kappa, target),
          direct_field_potential(sources, KifmmFieldFamily::PhiNormalZ, kernel,
                                 kappa, target)};
}

std::array<Vec3, 3>
direct_normal_grad_components(std::span<const KifmmPanelSource> sources,
                              KifmmKernel kernel, double kappa, Vec3 target) {
  return {direct_field_grad_target(sources, KifmmFieldFamily::PhiNormalX,
                                   kernel, kappa, target),
          direct_field_grad_target(sources, KifmmFieldFamily::PhiNormalY,
                                   kernel, kappa, target),
          direct_field_grad_target(sources, KifmmFieldFamily::PhiNormalZ,
                                   kernel, kappa, target)};
}

double sum_dot(const std::array<Vec3, 3> &values, Vec3 n) noexcept {
  return dot(values[0], n) + dot(values[1], n) + dot(values[2], n);
}

double max_result_residual(const KifmmEquivalentSolveResult &a,
                           double current) noexcept {
  if (std::isfinite(a.relative_residual)) {
    return std::max(current, a.relative_residual);
  }
  return current;
}

double max_result_condition(const KifmmEquivalentSolveResult &a,
                            double current) noexcept {
  if (std::isfinite(a.condition_estimate)) {
    return std::max(current, a.condition_estimate);
  }
  return current;
}

bool result_passed(const KifmmEquivalentSolveResult &result) noexcept {
  return result.finite && result.status == "pass";
}

} // namespace

double direct_field_potential(std::span<const KifmmPanelSource> sources,
                              KifmmFieldFamily family, KifmmKernel kernel,
                              double kappa, Vec3 target) {
  double sum = 0.0;
  double c = 0.0;
  for (const auto &source : sources) {
    double value = 0.0;
    if (family == KifmmFieldFamily::PhinCharge) {
      value = source.area * source.phin *
              scalar_kernel_value(kernel, target, source.position, kappa);
    } else {
      value = source.area * source.phi * component(source.normal, family) *
              component(grad_source(kernel, target, source.position, kappa),
                        family);
    }
    kahan_add(value, sum, c);
  }
  return sum;
}

Vec3 direct_field_grad_target(std::span<const KifmmPanelSource> sources,
                              KifmmFieldFamily family, KifmmKernel kernel,
                              double kappa, Vec3 target) {
  Vec3 sum{};
  Vec3 c{};
  for (const auto &source : sources) {
    Vec3 value{};
    if (family == KifmmFieldFamily::PhinCharge) {
      value = scale(scalar_kernel_grad_target(kernel, target, source.position,
                                              kappa),
                    source.area * source.phin);
    } else {
      const int c_idx = family_component_index(family);
      value = scale(grad_target_of_source_gradient_component(
                        kernel, target, source.position, kappa, c_idx),
                    source.area * source.phi * component(source.normal, family));
    }
    kahan_add_vec(value, sum, c);
  }
  return sum;
}

KifmmEquivalentSolveResult solve_upward_equivalent_field(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    std::span<const KifmmPanelSource> sources, KifmmFieldFamily family,
    KifmmKernel kernel, double kappa, const std::string &solver) {
  const KifmmEquivalentSystem system =
      build_equivalent_system(surface_rule, box, kernel, kappa);
  return solve_field_checked(system, box, surface_rule, sources, family, kernel,
                             kappa, solver);
}

double equivalent_field_potential(const KifmmSurfaceRule &surface_rule,
                                  const KifmmBox &box,
                                  const KifmmEquivalentSolveResult &result,
                                  KifmmFieldFamily, KifmmKernel kernel,
                                  double kappa, Vec3 target) {
  return equivalent_scalar_potential(surface_rule, box, result, kernel, kappa,
                                     target);
}

Vec3 equivalent_field_grad_target(const KifmmSurfaceRule &surface_rule,
                                  const KifmmBox &box,
                                  const KifmmEquivalentSolveResult &result,
                                  KifmmFieldFamily, KifmmKernel kernel,
                                  double kappa, Vec3 target) {
  return equivalent_scalar_grad_target(surface_rule, box, result, kernel,
                                       kappa, target);
}

KifmmJcpChannels assemble_kifmm_jcp_channels(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    std::span<const KifmmPanelSource> sources, double kappa, double, double,
    const std::string &solver, KifmmEquivalentSystemCache *system_cache,
    const std::string &condition_policy) {
  KifmmJcpChannels channels;
  const KifmmEquivalentSystemCacheStats before =
      system_cache != nullptr ? system_cache->stats
                              : KifmmEquivalentSystemCacheStats{};
  const KifmmEquivalentSystem laplace_system =
      system_cache != nullptr
          ? build_equivalent_system_cached(*system_cache, surface_rule, box,
                                           KifmmKernel::Laplace, 0.0, solver,
                                           condition_policy)
          : build_equivalent_system(surface_rule, box, KifmmKernel::Laplace,
                                    0.0, condition_policy);
  const KifmmEquivalentSystem yukawa_system =
      system_cache != nullptr
          ? build_equivalent_system_cached(*system_cache, surface_rule, box,
                                           KifmmKernel::Yukawa, kappa, solver,
                                           condition_policy)
          : build_equivalent_system(surface_rule, box, KifmmKernel::Yukawa,
                                    kappa, condition_policy);
  const KifmmEquivalentSystemCacheStats after =
      system_cache != nullptr ? system_cache->stats
                              : KifmmEquivalentSystemCacheStats{};
  channels.equivalent_system_builds = 2;
  channels.equivalent_rhs_solves = 8;
  channels.equivalent_rhs_builds = 8;
  channels.equivalent_factor_cache_hits = after.hits - before.hits;
  channels.equivalent_factor_cache_misses = after.misses - before.misses;
  channels.equivalent_matrix_build_seconds =
      after.matrix_build_seconds - before.matrix_build_seconds;
  channels.equivalent_condition_seconds =
      after.condition_seconds - before.condition_seconds;
  channels.equivalent_factor_seconds =
      after.factor_seconds - before.factor_seconds;
  if (!valid_panel_sources(sources)) {
    channels.status = "invalid_panel_source";
    channels.finite = false;
    return channels;
  }
  for (const auto &source : sources) {
    if (!point_inside_scaled_box(source.position, box, 1.0)) {
      channels.status = "source_outside_box";
      channels.finite = false;
      return channels;
    }
    if (!point_inside_scaled_box(source.position, box,
                                 surface_rule.equiv_scale)) {
      channels.status = "source_outside_equiv";
      channels.finite = false;
      return channels;
    }
  }

  const std::array<KifmmFieldFamily, 3> families{
      KifmmFieldFamily::PhiNormalX, KifmmFieldFamily::PhiNormalY,
      KifmmFieldFamily::PhiNormalZ};
  KifmmEquivalentSolveDiagnostics laplace_source_diag =
      laplace_system.diagnostics;
  KifmmEquivalentSolveDiagnostics yukawa_source_diag =
      yukawa_system.diagnostics;
  update_panel_source_distances(laplace_source_diag, laplace_system, sources);
  update_panel_source_distances(yukawa_source_diag, yukawa_system, sources);
  const auto rhs_start = Clock::now();
  const Eigen::MatrixXd laplace_rhs = build_jcp_rhs_matrix(
      laplace_system, sources, KifmmKernel::Laplace, 0.0);
  const Eigen::MatrixXd yukawa_rhs =
      build_jcp_rhs_matrix(yukawa_system, sources, KifmmKernel::Yukawa, kappa);
  channels.equivalent_rhs_build_seconds =
      elapsed_seconds(rhs_start, Clock::now());

  const auto solve_start = Clock::now();
  const std::vector<KifmmEquivalentSolveResult> laplace_results =
      solve_equivalent_rhs_batch(laplace_system, laplace_rhs, solver);
  const std::vector<KifmmEquivalentSolveResult> yukawa_results =
      solve_equivalent_rhs_batch(yukawa_system, yukawa_rhs, solver);
  channels.equivalent_solve_seconds =
      elapsed_seconds(solve_start, Clock::now());

  if (laplace_results.size() != 4 || yukawa_results.size() != 4) {
    channels.status = "field_solve_batch_failed";
    channels.finite = false;
    return channels;
  }

  channels.phin_laplace = laplace_results[0];
  channels.phin_yukawa = yukawa_results[0];
  channels.phin_laplace.diagnostics.min_source_equiv_distance =
      laplace_source_diag.min_source_equiv_distance;
  channels.phin_laplace.diagnostics.min_source_check_distance =
      laplace_source_diag.min_source_check_distance;
  channels.phin_yukawa.diagnostics.min_source_equiv_distance =
      yukawa_source_diag.min_source_equiv_distance;
  channels.phin_yukawa.diagnostics.min_source_check_distance =
      yukawa_source_diag.min_source_check_distance;
  for (std::size_t i = 0; i < families.size(); ++i) {
    channels.phi_normal_laplace[i] = laplace_results[i + 1];
    channels.phi_normal_yukawa[i] = yukawa_results[i + 1];
    channels.phi_normal_laplace[i].diagnostics.min_source_equiv_distance =
        laplace_source_diag.min_source_equiv_distance;
    channels.phi_normal_laplace[i].diagnostics.min_source_check_distance =
        laplace_source_diag.min_source_check_distance;
    channels.phi_normal_yukawa[i].diagnostics.min_source_equiv_distance =
        yukawa_source_diag.min_source_equiv_distance;
    channels.phi_normal_yukawa[i].diagnostics.min_source_check_distance =
        yukawa_source_diag.min_source_check_distance;
  }

  channels.finite =
      result_passed(channels.phin_laplace) && result_passed(channels.phin_yukawa);
  channels.max_relative_residual =
      max_result_residual(channels.phin_laplace, 0.0);
  channels.max_relative_residual =
      max_result_residual(channels.phin_yukawa, channels.max_relative_residual);
  channels.max_condition_estimate =
      max_result_condition(channels.phin_laplace, 0.0);
  channels.max_condition_estimate =
      max_result_condition(channels.phin_yukawa, channels.max_condition_estimate);
  for (std::size_t i = 0; i < 3; ++i) {
    channels.finite =
        channels.finite && result_passed(channels.phi_normal_laplace[i]) &&
        result_passed(channels.phi_normal_yukawa[i]);
    channels.max_relative_residual = max_result_residual(
        channels.phi_normal_laplace[i], channels.max_relative_residual);
    channels.max_relative_residual = max_result_residual(
        channels.phi_normal_yukawa[i], channels.max_relative_residual);
    channels.max_condition_estimate = max_result_condition(
        channels.phi_normal_laplace[i], channels.max_condition_estimate);
    channels.max_condition_estimate = max_result_condition(
        channels.phi_normal_yukawa[i], channels.max_condition_estimate);
  }
  channels.status = channels.finite ? "pass" : "field_solve_failed";
  return channels;
}

KifmmJcpChannelRow
assemble_direct_jcp_channel_row(std::span<const KifmmPanelSource> sources,
                                double kappa, double eps_ratio,
                                double inv_eps_ratio, Vec3 target,
                                Vec3 target_normal) {
  const double phin_lap =
      direct_field_potential(sources, KifmmFieldFamily::PhinCharge,
                             KifmmKernel::Laplace, 0.0, target);
  const double phin_yuk =
      direct_field_potential(sources, KifmmFieldFamily::PhinCharge,
                             KifmmKernel::Yukawa, kappa, target);
  const Vec3 phin_lap_grad =
      direct_field_grad_target(sources, KifmmFieldFamily::PhinCharge,
                               KifmmKernel::Laplace, 0.0, target);
  const Vec3 phin_yuk_grad =
      direct_field_grad_target(sources, KifmmFieldFamily::PhinCharge,
                               KifmmKernel::Yukawa, kappa, target);
  const auto normal_lap =
      direct_normal_potential_components(sources, KifmmKernel::Laplace, 0.0,
                                         target);
  const auto normal_yuk =
      direct_normal_potential_components(sources, KifmmKernel::Yukawa, kappa,
                                         target);
  const auto normal_lap_grad =
      direct_normal_grad_components(sources, KifmmKernel::Laplace, 0.0, target);
  const auto normal_yuk_grad =
      direct_normal_grad_components(sources, KifmmKernel::Yukawa, kappa,
                                    target);

  const double k1 = phin_lap - phin_yuk;
  const double k2 =
      eps_ratio * source_normal_field_sum(normal_yuk) -
      source_normal_field_sum(normal_lap);
  const double k3 =
      dot(phin_lap_grad, target_normal) -
      inv_eps_ratio * dot(phin_yuk_grad, target_normal);
  const double k4 =
      sum_dot(normal_lap_grad, target_normal) -
      sum_dot(normal_yuk_grad, target_normal);
  return {k1, k2, k3, k4};
}

KifmmJcpRow assemble_direct_jcp_row(std::span<const KifmmPanelSource> sources,
                                    double kappa, double eps_ratio,
                                    double inv_eps_ratio, Vec3 target,
                                    Vec3 target_normal) {
  const KifmmJcpChannelRow c = assemble_direct_jcp_channel_row(
      sources, kappa, eps_ratio, inv_eps_ratio, target, target_normal);
  return {c.k1 + c.k2, c.k3 + c.k4};
}

KifmmJcpChannelRow assemble_kifmm_jcp_channel_row(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    const KifmmJcpChannels &channels, double kappa, double eps_ratio,
    double inv_eps_ratio, Vec3 target, Vec3 target_normal) {
  const double phin_lap =
      equivalent_field_potential(surface_rule, box, channels.phin_laplace,
                                 KifmmFieldFamily::PhinCharge,
                                 KifmmKernel::Laplace, 0.0, target);
  const double phin_yuk =
      equivalent_field_potential(surface_rule, box, channels.phin_yukawa,
                                 KifmmFieldFamily::PhinCharge,
                                 KifmmKernel::Yukawa, kappa, target);
  const Vec3 phin_lap_grad = equivalent_field_grad_target(
      surface_rule, box, channels.phin_laplace, KifmmFieldFamily::PhinCharge,
      KifmmKernel::Laplace, 0.0, target);
  const Vec3 phin_yuk_grad = equivalent_field_grad_target(
      surface_rule, box, channels.phin_yukawa, KifmmFieldFamily::PhinCharge,
      KifmmKernel::Yukawa, kappa, target);

  std::array<double, 3> normal_lap{};
  std::array<double, 3> normal_yuk{};
  std::array<Vec3, 3> normal_lap_grad{};
  std::array<Vec3, 3> normal_yuk_grad{};
  const std::array<KifmmFieldFamily, 3> families{
      KifmmFieldFamily::PhiNormalX, KifmmFieldFamily::PhiNormalY,
      KifmmFieldFamily::PhiNormalZ};
  for (std::size_t i = 0; i < 3; ++i) {
    normal_lap[i] = equivalent_field_potential(
        surface_rule, box, channels.phi_normal_laplace[i], families[i],
        KifmmKernel::Laplace, 0.0, target);
    normal_yuk[i] = equivalent_field_potential(
        surface_rule, box, channels.phi_normal_yukawa[i], families[i],
        KifmmKernel::Yukawa, kappa, target);
    normal_lap_grad[i] = equivalent_field_grad_target(
        surface_rule, box, channels.phi_normal_laplace[i], families[i],
        KifmmKernel::Laplace, 0.0, target);
    normal_yuk_grad[i] = equivalent_field_grad_target(
        surface_rule, box, channels.phi_normal_yukawa[i], families[i],
        KifmmKernel::Yukawa, kappa, target);
  }

  const double k1 = phin_lap - phin_yuk;
  const double k2 =
      eps_ratio * source_normal_field_sum(normal_yuk) -
      source_normal_field_sum(normal_lap);
  const double k3 =
      dot(phin_lap_grad, target_normal) -
      inv_eps_ratio * dot(phin_yuk_grad, target_normal);
  const double k4 =
      sum_dot(normal_lap_grad, target_normal) -
      sum_dot(normal_yuk_grad, target_normal);
  return {k1, k2, k3, k4};
}

KifmmJcpRow assemble_kifmm_jcp_row(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmJcpChannels &channels,
                                   double kappa, double eps_ratio,
                                   double inv_eps_ratio, Vec3 target,
                                   Vec3 target_normal) {
  const KifmmJcpChannelRow c = assemble_kifmm_jcp_channel_row(
      surface_rule, box, channels, kappa, eps_ratio, inv_eps_ratio, target,
      target_normal);
  return {c.k1 + c.k2, c.k3 + c.k4};
}

} // namespace titan::tabi::fmm_kifmm

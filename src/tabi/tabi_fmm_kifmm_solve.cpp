#include "tabi_fmm_kifmm_solve.h"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>

namespace titan::tabi::fmm_kifmm {
namespace {

constexpr double kMaxEquivalentCondition = 1.0e12;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

bool finite_vec(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

Vec3 difference(Vec3 a, Vec3 b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

double norm(Vec3 v) noexcept {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

double distance(Vec3 a, Vec3 b) noexcept {
  return norm(difference(a, b));
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

Vec3 scale(Vec3 v, double s) noexcept {
  return {v.x * s, v.y * s, v.z * s};
}

bool finite_sources(std::span<const KifmmScalarSource> sources) noexcept {
  for (const auto &source : sources) {
    if (!finite_vec(source.position) || !std::isfinite(source.strength)) {
      return false;
    }
  }
  return true;
}

bool point_inside_scaled_box(Vec3 p, const KifmmBox &box, double scale) noexcept {
  if (!finite_vec(p) || !finite_vec(box.center) || !(box.half_width > 0.0) ||
      !std::isfinite(scale) || !(scale > 0.0)) {
    return false;
  }
  const double h = box.half_width * scale;
  return std::abs(p.x - box.center.x) <= h &&
         std::abs(p.y - box.center.y) <= h &&
         std::abs(p.z - box.center.z) <= h;
}

bool finite_vector(const Eigen::VectorXd &v) {
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (!std::isfinite(v[i])) {
      return false;
    }
  }
  return true;
}

bool finite_matrix(const Eigen::MatrixXd &m) {
  for (Eigen::Index r = 0; r < m.rows(); ++r) {
    for (Eigen::Index c = 0; c < m.cols(); ++c) {
      if (!std::isfinite(m(r, c))) {
        return false;
      }
    }
  }
  return true;
}

KifmmEquivalentSolver parse_solver(std::string_view solver) noexcept {
  if (solver == "qr") {
    return KifmmEquivalentSolver::QR;
  }
  if (solver == "svd") {
    return KifmmEquivalentSolver::SVD;
  }
  if (solver == "normal-tikhonov") {
    return KifmmEquivalentSolver::NormalTikhonov;
  }
  return KifmmEquivalentSolver::Unknown;
}

KifmmConditionPolicy parse_condition_policy(std::string_view policy) noexcept {
  if (policy == "svd-full-debug") {
    return KifmmConditionPolicy::SvdFullDebug;
  }
  if (policy == "svd-values") {
    return KifmmConditionPolicy::SvdValues;
  }
  if (policy == "qr-rank-only") {
    return KifmmConditionPolicy::QrRankOnly;
  }
  if (policy == "off") {
    return KifmmConditionPolicy::Off;
  }
  if (policy == "rank-bypass-diagnostic") {
    return KifmmConditionPolicy::RankBypassDiagnostic;
  }
  return KifmmConditionPolicy::Unknown;
}

std::string condition_policy_name(KifmmConditionPolicy policy) {
  switch (policy) {
  case KifmmConditionPolicy::SvdFullDebug:
    return "svd-full-debug";
  case KifmmConditionPolicy::SvdValues:
    return "svd-values";
  case KifmmConditionPolicy::QrRankOnly:
    return "qr-rank-only";
  case KifmmConditionPolicy::Off:
    return "off";
  case KifmmConditionPolicy::RankBypassDiagnostic:
    return "rank-bypass-diagnostic";
  case KifmmConditionPolicy::Unknown:
    return "unknown";
  }
  return "unknown";
}

KifmmEquivalentSolveResult failed_result(const std::string &solver,
                                         std::size_t n_equiv,
                                         std::size_t n_check,
                                         std::string status,
                                         KifmmEquivalentSolveDiagnostics diagnostics = {}) {
  KifmmEquivalentSolveResult result;
  result.solver = solver;
  result.equivalent_points = n_equiv;
  result.check_points = n_check;
  result.finite = false;
  result.status = std::move(status);
  result.diagnostics = diagnostics;
  return result;
}

Eigen::VectorXd solve_system(const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
                             KifmmEquivalentSolver solver) {
  if (solver == KifmmEquivalentSolver::QR) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
    Eigen::VectorXd x = qr.solve(b);
    for (int iter = 0; iter < 3; ++iter) {
      const Eigen::VectorXd residual = b - a * x;
      x += qr.solve(residual);
    }
    return x;
  }
  if (solver == KifmmEquivalentSolver::SVD) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.solve(b);
  }
  if (solver == KifmmEquivalentSolver::NormalTikhonov) {
    const Eigen::MatrixXd ata = a.transpose() * a;
    const Eigen::VectorXd atb = a.transpose() * b;
    const double scale = std::max(1.0, ata.diagonal().cwiseAbs().maxCoeff());
    Eigen::MatrixXd damped = ata;
    damped.diagonal().array() += 1.0e-12 * scale;
    return damped.ldlt().solve(atb);
  }
  return Eigen::VectorXd::Constant(a.cols(),
                                   std::numeric_limits<double>::quiet_NaN());
}

Eigen::MatrixXd solve_system_batch(const Eigen::MatrixXd &a,
                                   const Eigen::MatrixXd &b,
                                   KifmmEquivalentSolver solver) {
  if (solver == KifmmEquivalentSolver::QR) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
    Eigen::MatrixXd x = qr.solve(b);
    for (int iter = 0; iter < 3; ++iter) {
      const Eigen::MatrixXd residual = b - a * x;
      x += qr.solve(residual);
    }
    return x;
  }
  if (solver == KifmmEquivalentSolver::SVD) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.solve(b);
  }
  if (solver == KifmmEquivalentSolver::NormalTikhonov) {
    const Eigen::MatrixXd ata = a.transpose() * a;
    const Eigen::MatrixXd atb = a.transpose() * b;
    const double scale = std::max(1.0, ata.diagonal().cwiseAbs().maxCoeff());
    Eigen::MatrixXd damped = ata;
    damped.diagonal().array() += 1.0e-12 * scale;
    return damped.ldlt().solve(atb);
  }
  return Eigen::MatrixXd::Constant(a.cols(), b.cols(),
                                   std::numeric_limits<double>::quiet_NaN());
}

KifmmEquivalentSolveResult make_solve_result(
    const KifmmEquivalentSystem &system, const Eigen::VectorXd &rhs,
    const Eigen::VectorXd &q, const std::string &solver,
    KifmmEquivalentSolveDiagnostics diag) {
  const std::size_t n_equiv = system.equiv_points.size();
  const std::size_t n_check = system.check_points.size();
  if (q.size() != static_cast<Eigen::Index>(n_equiv) || !finite_vector(q)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_charges", diag);
  }
  diag.charge_norm = q.norm();

  const Eigen::VectorXd residual = system.check_to_equiv * q - rhs;
  if (!finite_vector(residual)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_residual", diag);
  }

  KifmmEquivalentSolveResult result;
  result.charges.assign(q.data(), q.data() + q.size());
  result.relative_residual =
      residual.norm() / std::max(rhs.norm(), std::numeric_limits<double>::min());
  result.max_abs_residual = residual.cwiseAbs().maxCoeff();
  result.finite = std::isfinite(result.relative_residual) &&
                  std::isfinite(result.max_abs_residual);
  result.solver = solver;
  result.status = result.finite ? "pass" : "nonfinite_residual";
  result.equivalent_points = n_equiv;
  result.check_points = n_check;
  result.rhs_norm = diag.rhs_norm;
  result.charge_norm = diag.charge_norm;
  result.condition_estimate = diag.condition_estimate;
  result.rank = diag.rank;
  result.diagnostics = diag;
  return result;
}

double condition_estimate(const Eigen::MatrixXd &a, int &rank,
                          double &sigma_min_out, double &sigma_max_out,
                          bool compute_vectors) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      a, compute_vectors ? (Eigen::ComputeThinU | Eigen::ComputeThinV) : 0);
  rank = static_cast<int>(svd.rank());
  sigma_min_out = 0.0;
  sigma_max_out = 0.0;
  if (svd.singularValues().size() == 0) {
    return 0.0;
  }
  const double sigma_max = svd.singularValues()[0];
  sigma_max_out = sigma_max;
  double sigma_min = std::numeric_limits<double>::infinity();
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    const double s = svd.singularValues()[i];
    if (s > 0.0) {
      sigma_min = std::min(sigma_min, s);
    }
  }
  if (!(sigma_min > 0.0) || !std::isfinite(sigma_min)) {
    return std::numeric_limits<double>::infinity();
  }
  sigma_min_out = sigma_min;
  return sigma_max / sigma_min;
}

void fill_rank_condition_diagnostics(
    KifmmEquivalentSolveDiagnostics &diag,
    const Eigen::MatrixXd &matrix,
    const std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> &qr,
    const std::string &condition_policy,
    KifmmEquivalentSystemCacheStats *stats = nullptr) {
  const KifmmConditionPolicy policy = parse_condition_policy(condition_policy);
  diag.condition_policy = condition_policy_name(policy);
  diag.condition_computed = false;
  diag.condition_gate_enabled = false;
  diag.condition_estimate = std::numeric_limits<double>::quiet_NaN();
  diag.sigma_min = std::numeric_limits<double>::quiet_NaN();
  diag.sigma_max = std::numeric_limits<double>::quiet_NaN();
  diag.rank = qr ? static_cast<int>(qr->rank()) : -1;
  if (policy == KifmmConditionPolicy::Unknown) {
    diag.rank = -1;
    diag.condition_policy = "unknown";
    return;
  }
  if (policy == KifmmConditionPolicy::QrRankOnly ||
      policy == KifmmConditionPolicy::Off ||
      policy == KifmmConditionPolicy::RankBypassDiagnostic) {
    return;
  }

  const auto condition_start = Clock::now();
  diag.condition_estimate = condition_estimate(
      matrix, diag.rank, diag.sigma_min, diag.sigma_max,
      policy == KifmmConditionPolicy::SvdFullDebug);
  if (stats != nullptr) {
    stats->condition_seconds += elapsed_seconds(condition_start, Clock::now());
  }
  diag.condition_computed = true;
  diag.condition_gate_enabled = true;
}

KifmmEquivalentSolveDiagnostics geometry_diagnostics(
    std::span<const KifmmSurfacePoint> equiv_points,
    std::span<const KifmmSurfacePoint> check_points,
    std::span<const KifmmScalarSource> sources) {
  KifmmEquivalentSolveDiagnostics diag;
  diag.min_check_equiv_distance = std::numeric_limits<double>::infinity();
  diag.min_source_equiv_distance = std::numeric_limits<double>::infinity();
  diag.min_source_check_distance = std::numeric_limits<double>::infinity();
  for (const auto &check : check_points) {
    for (const auto &equiv : equiv_points) {
      diag.min_check_equiv_distance =
          std::min(diag.min_check_equiv_distance, distance(check.x, equiv.x));
    }
  }
  for (const auto &source : sources) {
    for (const auto &equiv : equiv_points) {
      diag.min_source_equiv_distance = std::min(
          diag.min_source_equiv_distance, distance(source.position, equiv.x));
    }
    for (const auto &check : check_points) {
      diag.min_source_check_distance = std::min(
          diag.min_source_check_distance, distance(source.position, check.x));
    }
  }
  if (!std::isfinite(diag.min_check_equiv_distance)) {
    diag.min_check_equiv_distance = 0.0;
  }
  if (!std::isfinite(diag.min_source_equiv_distance)) {
    diag.min_source_equiv_distance = 0.0;
  }
  if (!std::isfinite(diag.min_source_check_distance)) {
    diag.min_source_check_distance = 0.0;
  }
  return diag;
}

void update_source_geometry_diagnostics(
    KifmmEquivalentSolveDiagnostics &diag,
    std::span<const KifmmSurfacePoint> equiv_points,
    std::span<const KifmmSurfacePoint> check_points,
    std::span<const KifmmScalarSource> sources) {
  diag.min_source_equiv_distance = std::numeric_limits<double>::infinity();
  diag.min_source_check_distance = std::numeric_limits<double>::infinity();
  for (const auto &source : sources) {
    for (const auto &equiv : equiv_points) {
      diag.min_source_equiv_distance = std::min(
          diag.min_source_equiv_distance, distance(source.position, equiv.x));
    }
    for (const auto &check : check_points) {
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

bool cache_key_matches(const KifmmEquivalentSystemCacheEntry &entry,
                       const KifmmSurfaceRule &surface_rule,
                       const KifmmBox &box, KifmmKernel kernel, double kappa,
                       const std::string &solver,
                       const std::string &condition_policy) noexcept {
  return entry.order == surface_rule.order &&
         entry.equiv_scale == surface_rule.equiv_scale &&
         entry.check_scale == surface_rule.check_scale &&
         entry.half_width == box.half_width && entry.kernel == kernel &&
         entry.kappa == kappa && entry.solver == solver &&
         entry.condition_policy == condition_policy;
}

KifmmEquivalentSystem translated_system_from_canonical(
    const KifmmEquivalentSystemCacheEntry &entry,
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box) {
  KifmmEquivalentSystem system;
  system.kernel = entry.kernel;
  system.kappa = entry.kappa;
  system.equiv_points = instantiate_surface(surface_rule.unit_equiv, box);
  system.check_points = instantiate_surface(surface_rule.unit_check, box);
  system.check_to_equiv = entry.canonical_system.check_to_equiv;
  system.qr = entry.canonical_system.qr;
  system.diagnostics =
      geometry_diagnostics(system.equiv_points, system.check_points, {});
  system.diagnostics.matrix_frobenius_norm =
      entry.canonical_system.diagnostics.matrix_frobenius_norm;
  system.diagnostics.condition_estimate =
      entry.canonical_system.diagnostics.condition_estimate;
  system.diagnostics.sigma_min =
      entry.canonical_system.diagnostics.sigma_min;
  system.diagnostics.sigma_max =
      entry.canonical_system.diagnostics.sigma_max;
  system.diagnostics.rank = entry.canonical_system.diagnostics.rank;
  system.diagnostics.condition_computed =
      entry.canonical_system.diagnostics.condition_computed;
  system.diagnostics.condition_gate_enabled =
      entry.canonical_system.diagnostics.condition_gate_enabled;
  system.diagnostics.condition_policy =
      entry.canonical_system.diagnostics.condition_policy;
  system.status = entry.canonical_system.status;
  system.finite = entry.canonical_system.finite;
  return system;
}

} // namespace

double direct_scalar_potential(std::span<const KifmmScalarSource> sources,
                               KifmmKernel kernel, double kappa,
                               Vec3 target) {
  double sum = 0.0;
  double c = 0.0;
  for (const auto &source : sources) {
    kahan_add(source.strength *
                  scalar_kernel_value(kernel, target, source.position, kappa),
              sum, c);
  }
  return sum;
}

Vec3 direct_scalar_grad_target(std::span<const KifmmScalarSource> sources,
                               KifmmKernel kernel, double kappa,
                               Vec3 target) {
  Vec3 sum{};
  Vec3 c{};
  for (const auto &source : sources) {
    kahan_add_vec(scale(scalar_kernel_grad_target(kernel, target,
                                                  source.position, kappa),
                        source.strength),
                  sum, c);
  }
  return sum;
}

KifmmEquivalentSystem build_equivalent_system(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    KifmmKernel kernel, double kappa,
    const std::string &condition_policy) {
  KifmmEquivalentSystem system;
  system.kernel = kernel;
  system.kappa = kappa;
  system.equiv_points = instantiate_surface(surface_rule.unit_equiv, box);
  system.check_points = instantiate_surface(surface_rule.unit_check, box);
  const std::size_t n_equiv = system.equiv_points.size();
  const std::size_t n_check = system.check_points.size();
  system.diagnostics =
      geometry_diagnostics(system.equiv_points, system.check_points, {});

  if (n_equiv == 0 || n_check == 0) {
    system.status = "empty_surface";
    return system;
  }
  if (!std::isfinite(kappa) || kappa < 0.0) {
    system.status = "invalid_kappa";
    return system;
  }
  constexpr double collision_tol = 1.0e-12;
  if (system.diagnostics.min_check_equiv_distance <= collision_tol) {
    system.status = "surface_collision";
    return system;
  }

  system.check_to_equiv.resize(static_cast<Eigen::Index>(n_check),
                               static_cast<Eigen::Index>(n_equiv));
  for (std::size_t i = 0; i < n_check; ++i) {
    for (std::size_t j = 0; j < n_equiv; ++j) {
      system.check_to_equiv(static_cast<Eigen::Index>(i),
                            static_cast<Eigen::Index>(j)) =
          scalar_kernel_value(kernel, system.check_points[i].x,
                              system.equiv_points[j].x, kappa);
    }
  }
  if (!finite_matrix(system.check_to_equiv)) {
    system.status = "nonfinite_matrix";
    return system;
  }

  system.diagnostics.matrix_frobenius_norm = system.check_to_equiv.norm();
  system.qr =
      std::make_shared<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>(
          system.check_to_equiv);
  fill_rank_condition_diagnostics(system.diagnostics, system.check_to_equiv,
                                  system.qr, condition_policy);
  if (system.diagnostics.rank < 0) {
    system.status = "unknown_condition_policy";
    return system;
  }
  system.finite = true;
  system.status = "pass";
  return system;
}

KifmmEquivalentSystem build_equivalent_system_cached(
    KifmmEquivalentSystemCache &cache, const KifmmSurfaceRule &surface_rule,
    const KifmmBox &box, KifmmKernel kernel, double kappa,
    const std::string &solver, const std::string &condition_policy) {
  for (const KifmmEquivalentSystemCacheEntry &entry : cache.entries) {
    if (cache_key_matches(entry, surface_rule, box, kernel, kappa, solver,
                          condition_policy)) {
      ++cache.stats.hits;
      return translated_system_from_canonical(entry, surface_rule, box);
    }
  }

  ++cache.stats.misses;
  KifmmEquivalentSystemCacheEntry entry;
  entry.order = surface_rule.order;
  entry.equiv_scale = surface_rule.equiv_scale;
  entry.check_scale = surface_rule.check_scale;
  entry.half_width = box.half_width;
  entry.kernel = kernel;
  entry.kappa = kappa;
  entry.solver = solver;
  entry.condition_policy = condition_policy;

  const KifmmBox canonical_box{{0.0, 0.0, 0.0}, box.half_width, box.level,
                               box.node_id};
  KifmmEquivalentSystem canonical;
  canonical.kernel = kernel;
  canonical.kappa = kappa;
  canonical.equiv_points =
      instantiate_surface(surface_rule.unit_equiv, canonical_box);
  canonical.check_points =
      instantiate_surface(surface_rule.unit_check, canonical_box);
  const std::size_t n_equiv = canonical.equiv_points.size();
  const std::size_t n_check = canonical.check_points.size();
  canonical.diagnostics =
      geometry_diagnostics(canonical.equiv_points, canonical.check_points, {});

  if (n_equiv == 0 || n_check == 0) {
    canonical.status = "empty_surface";
  } else if (!std::isfinite(kappa) || kappa < 0.0) {
    canonical.status = "invalid_kappa";
  } else if (canonical.diagnostics.min_check_equiv_distance <= 1.0e-12) {
    canonical.status = "surface_collision";
  } else {
    const auto matrix_start = Clock::now();
    canonical.check_to_equiv.resize(static_cast<Eigen::Index>(n_check),
                                    static_cast<Eigen::Index>(n_equiv));
    for (std::size_t i = 0; i < n_check; ++i) {
      for (std::size_t j = 0; j < n_equiv; ++j) {
        canonical.check_to_equiv(static_cast<Eigen::Index>(i),
                                 static_cast<Eigen::Index>(j)) =
            scalar_kernel_value(kernel, canonical.check_points[i].x,
                                canonical.equiv_points[j].x, kappa);
      }
    }
    cache.stats.matrix_build_seconds +=
        elapsed_seconds(matrix_start, Clock::now());
    ++cache.stats.matrix_builds;
    if (!finite_matrix(canonical.check_to_equiv)) {
      canonical.status = "nonfinite_matrix";
    } else {
      canonical.diagnostics.matrix_frobenius_norm =
          canonical.check_to_equiv.norm();
      const auto factor_start = Clock::now();
      canonical.qr =
          std::make_shared<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>(
              canonical.check_to_equiv);
      cache.stats.factor_seconds += elapsed_seconds(factor_start, Clock::now());
      ++cache.stats.factorizations;
      if (kernel == KifmmKernel::Laplace) {
        ++cache.stats.laplace_systems;
      } else {
        ++cache.stats.yukawa_systems;
      }
      fill_rank_condition_diagnostics(canonical.diagnostics,
                                      canonical.check_to_equiv, canonical.qr,
                                      condition_policy, &cache.stats);
      if (canonical.diagnostics.rank < 0) {
        canonical.status = "unknown_condition_policy";
        entry.canonical_system = std::move(canonical);
        cache.entries.push_back(std::move(entry));
        return translated_system_from_canonical(cache.entries.back(),
                                                surface_rule, box);
      }
      canonical.finite = true;
      canonical.status = "pass";
    }
  }

  entry.canonical_system = std::move(canonical);
  cache.entries.push_back(std::move(entry));
  return translated_system_from_canonical(cache.entries.back(), surface_rule,
                                          box);
}

KifmmEquivalentSolveResult
solve_equivalent_rhs(const KifmmEquivalentSystem &system,
                     const Eigen::VectorXd &rhs,
                     const std::string &solver) {
  const std::size_t n_equiv = system.equiv_points.size();
  const std::size_t n_check = system.check_points.size();
  KifmmEquivalentSolveDiagnostics diag = system.diagnostics;
  diag.rhs_norm = rhs.norm();
  if (!system.finite) {
    return failed_result(solver, n_equiv, n_check, system.status, diag);
  }
  if (rhs.size() != static_cast<Eigen::Index>(n_check) ||
      !finite_vector(rhs)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_rhs", diag);
  }
  const KifmmEquivalentSolver solver_mode = parse_solver(solver);
  if (solver_mode == KifmmEquivalentSolver::Unknown) {
    return failed_result(solver, n_equiv, n_check, "unknown_solver", diag);
  }
  const int full_rank =
      static_cast<int>(std::min(system.check_to_equiv.rows(),
                                system.check_to_equiv.cols()));
  const bool bypass_rank_gate =
      parse_condition_policy(diag.condition_policy) ==
      KifmmConditionPolicy::RankBypassDiagnostic;
  if (!bypass_rank_gate && diag.rank < full_rank) {
    return failed_result(solver, n_equiv, n_check, "rank_deficient", diag);
  }
  if (diag.condition_gate_enabled &&
      !(diag.condition_estimate <= kMaxEquivalentCondition)) {
    return failed_result(solver, n_equiv, n_check, "ill_conditioned", diag);
  }

  Eigen::VectorXd q;
  if (solver_mode == KifmmEquivalentSolver::QR) {
    if (!system.qr) {
      return failed_result(solver, n_equiv, n_check, "missing_qr", diag);
    }
    q = system.qr->solve(rhs);
    for (int iter = 0; iter < 3; ++iter) {
      const Eigen::VectorXd residual = rhs - system.check_to_equiv * q;
      q += system.qr->solve(residual);
    }
  } else {
    q = solve_system(system.check_to_equiv, rhs, solver_mode);
  }
  if (q.size() != static_cast<Eigen::Index>(n_equiv) || !finite_vector(q)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_charges", diag);
  }
  diag.charge_norm = q.norm();

  const Eigen::VectorXd residual = system.check_to_equiv * q - rhs;
  if (!finite_vector(residual)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_residual", diag);
  }

  KifmmEquivalentSolveResult result;
  result.charges.assign(q.data(), q.data() + q.size());
  result.relative_residual =
      residual.norm() / std::max(rhs.norm(), std::numeric_limits<double>::min());
  result.max_abs_residual = residual.cwiseAbs().maxCoeff();
  result.finite = std::isfinite(result.relative_residual) &&
                  std::isfinite(result.max_abs_residual);
  result.solver = solver;
  result.status = result.finite ? "pass" : "nonfinite_residual";
  result.equivalent_points = n_equiv;
  result.check_points = n_check;
  result.rhs_norm = diag.rhs_norm;
  result.charge_norm = diag.charge_norm;
  result.condition_estimate = diag.condition_estimate;
  result.rank = diag.rank;
  result.diagnostics = diag;
  return result;
}

std::vector<KifmmEquivalentSolveResult>
solve_equivalent_rhs_batch(const KifmmEquivalentSystem &system,
                           const Eigen::MatrixXd &rhs,
                           const std::string &solver) {
  const std::size_t n_equiv = system.equiv_points.size();
  const std::size_t n_check = system.check_points.size();
  std::vector<KifmmEquivalentSolveResult> results;
  if (rhs.cols() <= 0) {
    return results;
  }
  results.reserve(static_cast<std::size_t>(rhs.cols()));
  KifmmEquivalentSolveDiagnostics base_diag = system.diagnostics;
  if (!system.finite) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(
          failed_result(solver, n_equiv, n_check, system.status, base_diag));
    }
    return results;
  }
  if (rhs.rows() != static_cast<Eigen::Index>(n_check) ||
      !finite_matrix(rhs)) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(
          failed_result(solver, n_equiv, n_check, "nonfinite_rhs", base_diag));
    }
    return results;
  }
  const KifmmEquivalentSolver solver_mode = parse_solver(solver);
  if (solver_mode == KifmmEquivalentSolver::Unknown) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(
          failed_result(solver, n_equiv, n_check, "unknown_solver", base_diag));
    }
    return results;
  }
  const int full_rank =
      static_cast<int>(std::min(system.check_to_equiv.rows(),
                                system.check_to_equiv.cols()));
  const bool bypass_rank_gate =
      parse_condition_policy(base_diag.condition_policy) ==
      KifmmConditionPolicy::RankBypassDiagnostic;
  if (!bypass_rank_gate && base_diag.rank < full_rank) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(failed_result(solver, n_equiv, n_check,
                                      "rank_deficient", base_diag));
    }
    return results;
  }
  if (base_diag.condition_gate_enabled &&
      !(base_diag.condition_estimate <= kMaxEquivalentCondition)) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(failed_result(solver, n_equiv, n_check,
                                      "ill_conditioned", base_diag));
    }
    return results;
  }

  Eigen::MatrixXd q;
  if (solver_mode == KifmmEquivalentSolver::QR) {
    if (!system.qr) {
      for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
        results.push_back(
            failed_result(solver, n_equiv, n_check, "missing_qr", base_diag));
      }
      return results;
    }
    q = system.qr->solve(rhs);
    for (int iter = 0; iter < 3; ++iter) {
      const Eigen::MatrixXd residual = rhs - system.check_to_equiv * q;
      q += system.qr->solve(residual);
    }
  } else {
    q = solve_system_batch(system.check_to_equiv, rhs, solver_mode);
  }
  if (q.rows() != static_cast<Eigen::Index>(n_equiv) ||
      q.cols() != rhs.cols() || !finite_matrix(q)) {
    for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
      results.push_back(failed_result(solver, n_equiv, n_check,
                                      "nonfinite_charges", base_diag));
    }
    return results;
  }
  for (Eigen::Index col = 0; col < rhs.cols(); ++col) {
    KifmmEquivalentSolveDiagnostics diag = base_diag;
    diag.rhs_norm = rhs.col(col).norm();
    results.push_back(make_solve_result(system, rhs.col(col), q.col(col),
                                        solver, diag));
  }
  return results;
}

KifmmEquivalentSolveResult
solve_upward_equivalent(const KifmmSurfaceRule &surface_rule,
                        const KifmmBox &box,
                        std::span<const KifmmScalarSource> sources,
                        KifmmKernel kernel, double kappa,
                        const std::string &solver) {
  auto system = build_equivalent_system(surface_rule, box, kernel, kappa);
  KifmmEquivalentSolveDiagnostics diag = system.diagnostics;
  update_source_geometry_diagnostics(diag, system.equiv_points,
                                     system.check_points, sources);
  const std::size_t n_equiv = system.equiv_points.size();
  const std::size_t n_check = system.check_points.size();
  if (!system.finite) {
    return failed_result(solver, n_equiv, n_check, system.status, diag);
  }
  if (!finite_sources(sources)) {
    return failed_result(solver, n_equiv, n_check, "nonfinite_source", diag);
  }
  for (const auto &source : sources) {
    if (!point_inside_scaled_box(source.position, box, 1.0)) {
      return failed_result(solver, n_equiv, n_check, "source_outside_box",
                           diag);
    }
    if (!point_inside_scaled_box(source.position, box,
                                 surface_rule.equiv_scale)) {
      return failed_result(solver, n_equiv, n_check, "source_outside_equiv",
                           diag);
    }
  }
  Eigen::VectorXd b(static_cast<Eigen::Index>(n_check));
  for (std::size_t i = 0; i < n_check; ++i) {
    b[static_cast<Eigen::Index>(i)] =
        direct_scalar_potential(sources, kernel, kappa, system.check_points[i].x);
  }
  KifmmEquivalentSolveResult result = solve_equivalent_rhs(system, b, solver);
  result.diagnostics.min_source_equiv_distance =
      diag.min_source_equiv_distance;
  result.diagnostics.min_source_check_distance = diag.min_source_check_distance;
  return result;
}

double equivalent_scalar_potential(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmEquivalentSolveResult &result,
                                   KifmmKernel kernel, double kappa,
                                   Vec3 target) {
  const auto equiv_points = instantiate_surface(surface_rule.unit_equiv, box);
  if (!result.finite || result.charges.size() != equiv_points.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  double c = 0.0;
  for (std::size_t i = 0; i < equiv_points.size(); ++i) {
    kahan_add(result.charges[i] *
                  scalar_kernel_value(kernel, target, equiv_points[i].x,
                                      kappa),
              sum, c);
  }
  return sum;
}

Vec3 equivalent_scalar_grad_target(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmEquivalentSolveResult &result,
                                   KifmmKernel kernel, double kappa,
                                   Vec3 target) {
  const auto equiv_points = instantiate_surface(surface_rule.unit_equiv, box);
  if (!result.finite || result.charges.size() != equiv_points.size()) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan, nan};
  }
  Vec3 sum{};
  Vec3 c{};
  for (std::size_t i = 0; i < equiv_points.size(); ++i) {
    kahan_add_vec(scale(scalar_kernel_grad_target(kernel, target,
                                                  equiv_points[i].x, kappa),
                        result.charges[i]),
                  sum, c);
  }
  return sum;
}

} // namespace titan::tabi::fmm_kifmm

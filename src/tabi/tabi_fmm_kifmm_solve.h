#ifndef TITAN_TABI_FMM_KIFMM_SOLVE_H
#define TITAN_TABI_FMM_KIFMM_SOLVE_H

#include "tabi_fmm_kifmm_kernels.h"

#include <Eigen/Dense>
#include <Eigen/QR>

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace titan::tabi::fmm_kifmm {

struct KifmmScalarSource {
  Vec3 position{};
  double strength{0.0};
};

enum class KifmmEquivalentSolver {
  QR,
  SVD,
  NormalTikhonov,
  Unknown,
};

enum class KifmmConditionPolicy {
  SvdFullDebug,
  SvdValues,
  QrRankOnly,
  Off,
  RankBypassDiagnostic,
  Unknown,
};

struct KifmmEquivalentSolveDiagnostics {
  double rhs_norm{0.0};
  double charge_norm{0.0};
  double matrix_frobenius_norm{0.0};
  double condition_estimate{0.0};
  double sigma_min{0.0};
  double sigma_max{0.0};
  double min_check_equiv_distance{0.0};
  double min_source_equiv_distance{0.0};
  double min_source_check_distance{0.0};
  int rank{-1};
  bool condition_computed{true};
  bool condition_gate_enabled{true};
  std::string condition_policy{"svd-full-debug"};
};

struct KifmmEquivalentSolveResult {
  std::vector<double> charges{};
  double relative_residual{0.0};
  double max_abs_residual{0.0};
  double rhs_norm{0.0};
  double charge_norm{0.0};
  double condition_estimate{0.0};
  int rank{-1};
  bool finite{false};
  std::string solver{};
  std::string status{"not_started"};
  std::size_t equivalent_points{0};
  std::size_t check_points{0};
  KifmmEquivalentSolveDiagnostics diagnostics{};
};

struct KifmmEquivalentSystem {
  std::vector<KifmmSurfacePoint> equiv_points{};
  std::vector<KifmmSurfacePoint> check_points{};
  Eigen::MatrixXd check_to_equiv{};
  std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> qr{};
  KifmmEquivalentSolveDiagnostics diagnostics{};
  KifmmKernel kernel{KifmmKernel::Laplace};
  double kappa{0.0};
  std::string status{"not_started"};
  bool finite{false};
};

struct KifmmEquivalentSystemCacheStats {
  std::size_t hits{0};
  std::size_t misses{0};
  std::size_t matrix_builds{0};
  std::size_t factorizations{0};
  std::size_t laplace_systems{0};
  std::size_t yukawa_systems{0};
  double matrix_build_seconds{0.0};
  double condition_seconds{0.0};
  double factor_seconds{0.0};
};

struct KifmmEquivalentSystemCacheEntry {
  int order{0};
  double equiv_scale{0.0};
  double check_scale{0.0};
  double half_width{0.0};
  KifmmKernel kernel{KifmmKernel::Laplace};
  double kappa{0.0};
  std::string solver{};
  std::string condition_policy{"svd-full-debug"};
  KifmmEquivalentSystem canonical_system{};
};

struct KifmmEquivalentSystemCache {
  std::vector<KifmmEquivalentSystemCacheEntry> entries{};
  KifmmEquivalentSystemCacheStats stats{};
};

double direct_scalar_potential(std::span<const KifmmScalarSource> sources,
                               KifmmKernel kernel, double kappa,
                               Vec3 target);

Vec3 direct_scalar_grad_target(std::span<const KifmmScalarSource> sources,
                               KifmmKernel kernel, double kappa,
                               Vec3 target);

KifmmEquivalentSystem build_equivalent_system(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    KifmmKernel kernel, double kappa,
    const std::string &condition_policy = "svd-full-debug");

KifmmEquivalentSystem build_equivalent_system_cached(
    KifmmEquivalentSystemCache &cache, const KifmmSurfaceRule &surface_rule,
    const KifmmBox &box, KifmmKernel kernel, double kappa,
    const std::string &solver,
    const std::string &condition_policy = "svd-full-debug");

KifmmEquivalentSolveResult
solve_equivalent_rhs(const KifmmEquivalentSystem &system,
                     const Eigen::VectorXd &rhs,
                     const std::string &solver);

std::vector<KifmmEquivalentSolveResult>
solve_equivalent_rhs_batch(const KifmmEquivalentSystem &system,
                           const Eigen::MatrixXd &rhs,
                           const std::string &solver);

KifmmEquivalentSolveResult
solve_upward_equivalent(const KifmmSurfaceRule &surface_rule,
                        const KifmmBox &box,
                        std::span<const KifmmScalarSource> sources,
                        KifmmKernel kernel, double kappa,
                        const std::string &solver);

double equivalent_scalar_potential(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmEquivalentSolveResult &result,
                                   KifmmKernel kernel, double kappa,
                                   Vec3 target);

Vec3 equivalent_scalar_grad_target(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmEquivalentSolveResult &result,
                                   KifmmKernel kernel, double kappa,
                                   Vec3 target);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_SOLVE_H

#ifndef TITAN_TABI_FMM_KIFMM_FIELDS_H
#define TITAN_TABI_FMM_KIFMM_FIELDS_H

#include "tabi_fmm_kifmm_solve.h"

#include <array>
#include <cstddef>
#include <span>
#include <string>

namespace titan::tabi::fmm_kifmm {

enum class KifmmFieldFamily {
  PhinCharge,
  PhiNormalX,
  PhiNormalY,
  PhiNormalZ,
};

struct KifmmPanelSource {
  Vec3 position{};
  Vec3 normal{};
  double area{1.0};
  double phi{0.0};
  double phin{0.0};
};

struct KifmmJcpChannels {
  KifmmEquivalentSolveResult phin_laplace{};
  KifmmEquivalentSolveResult phin_yukawa{};
  std::array<KifmmEquivalentSolveResult, 3> phi_normal_laplace{};
  std::array<KifmmEquivalentSolveResult, 3> phi_normal_yukawa{};
  double max_relative_residual{0.0};
  double max_condition_estimate{0.0};
  std::size_t equivalent_system_builds{0};
  std::size_t equivalent_rhs_solves{0};
  std::size_t equivalent_rhs_builds{0};
  std::size_t equivalent_factor_cache_hits{0};
  std::size_t equivalent_factor_cache_misses{0};
  double equivalent_matrix_build_seconds{0.0};
  double equivalent_condition_seconds{0.0};
  double equivalent_factor_seconds{0.0};
  double equivalent_rhs_build_seconds{0.0};
  double equivalent_solve_seconds{0.0};
  bool finite{false};
  std::string status{"not_started"};
};

struct KifmmJcpRow {
  double phi_row{0.0};
  double phin_row{0.0};
};

struct KifmmJcpChannelRow {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
};

double direct_field_potential(std::span<const KifmmPanelSource> sources,
                              KifmmFieldFamily family, KifmmKernel kernel,
                              double kappa, Vec3 target);

Vec3 direct_field_grad_target(std::span<const KifmmPanelSource> sources,
                              KifmmFieldFamily family, KifmmKernel kernel,
                              double kappa, Vec3 target);

KifmmEquivalentSolveResult solve_upward_equivalent_field(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    std::span<const KifmmPanelSource> sources, KifmmFieldFamily family,
    KifmmKernel kernel, double kappa, const std::string &solver);

double equivalent_field_potential(const KifmmSurfaceRule &surface_rule,
                                  const KifmmBox &box,
                                  const KifmmEquivalentSolveResult &result,
                                  KifmmFieldFamily family,
                                  KifmmKernel kernel, double kappa,
                                  Vec3 target);

Vec3 equivalent_field_grad_target(const KifmmSurfaceRule &surface_rule,
                                  const KifmmBox &box,
                                  const KifmmEquivalentSolveResult &result,
                                  KifmmFieldFamily family,
                                  KifmmKernel kernel, double kappa,
                                  Vec3 target);

KifmmJcpChannels assemble_kifmm_jcp_channels(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    std::span<const KifmmPanelSource> sources, double kappa,
    double eps_ratio, double inv_eps_ratio, const std::string &solver,
    KifmmEquivalentSystemCache *system_cache = nullptr,
    const std::string &condition_policy = "svd-full-debug");

KifmmJcpRow assemble_direct_jcp_row(std::span<const KifmmPanelSource> sources,
                                    double kappa, double eps_ratio,
                                    double inv_eps_ratio, Vec3 target,
                                    Vec3 target_normal);

KifmmJcpChannelRow
assemble_direct_jcp_channel_row(std::span<const KifmmPanelSource> sources,
                                double kappa, double eps_ratio,
                                double inv_eps_ratio, Vec3 target,
                                Vec3 target_normal);

KifmmJcpRow assemble_kifmm_jcp_row(const KifmmSurfaceRule &surface_rule,
                                   const KifmmBox &box,
                                   const KifmmJcpChannels &channels,
                                   double kappa, double eps_ratio,
                                   double inv_eps_ratio, Vec3 target,
                                   Vec3 target_normal);

KifmmJcpChannelRow assemble_kifmm_jcp_channel_row(
    const KifmmSurfaceRule &surface_rule, const KifmmBox &box,
    const KifmmJcpChannels &channels, double kappa, double eps_ratio,
    double inv_eps_ratio, Vec3 target, Vec3 target_normal);

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_FIELDS_H

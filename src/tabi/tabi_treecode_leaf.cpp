#include "tabi_treecode_internal.h"
#include "tabi_near.h"
#include "tabi_node_patch_curved.h"

#include "../titan_avx2_math.h"
#include "../titan_executor.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::detail {
TreePanelCache build_tree_panel_cache(const std::vector<Panel> &panels,
                                      const std::vector<std::size_t> &order) {
  TreePanelCache cache;
  const std::size_t n = panels.size();
  cache.x.resize(n);
  cache.y.resize(n);
  cache.z.resize(n);
  cache.nx.resize(n);
  cache.ny.resize(n);
  cache.nz.resize(n);
  cache.area.resize(n);
  cache.ordered_x.resize(n);
  cache.ordered_y.resize(n);
  cache.ordered_z.resize(n);
  cache.ordered_nx.resize(n);
  cache.ordered_ny.resize(n);
  cache.ordered_nz.resize(n);
  cache.ordered_area.resize(n);
  cache.ordered_original_index.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Panel &panel = panels[i];
    cache.x[i] =
        panel.has_source_quadrature_center ? panel.source_x : panel.x;
    cache.y[i] =
        panel.has_source_quadrature_center ? panel.source_y : panel.y;
    cache.z[i] =
        panel.has_source_quadrature_center ? panel.source_z : panel.z;
    cache.nx[i] = panel_source_nx(panel);
    cache.ny[i] = panel_source_ny(panel);
    cache.nz[i] = panel_source_nz(panel);
    cache.area[i] = panel.area;
  }
  for (std::size_t pos = 0; pos < n; ++pos) {
    const std::size_t original = pos < order.size() ? order[pos] : pos;
    const Panel &panel = panels[original];
    cache.ordered_x[pos] =
        panel.has_source_quadrature_center ? panel.source_x : panel.x;
    cache.ordered_y[pos] =
        panel.has_source_quadrature_center ? panel.source_y : panel.y;
    cache.ordered_z[pos] =
        panel.has_source_quadrature_center ? panel.source_z : panel.z;
    cache.ordered_nx[pos] = panel_source_nx(panel);
    cache.ordered_ny[pos] = panel_source_ny(panel);
    cache.ordered_nz[pos] = panel_source_nz(panel);
    cache.ordered_area[pos] = panel.area;
    cache.ordered_original_index[pos] = original;
  }
  return cache;
}

TreeUnknownCache build_tree_unknown_cache(const std::vector<Panel> &panels,
                                          const std::vector<double> &unknowns,
                                          bool coupled,
                                          const std::vector<std::size_t> &order) {
  TreeUnknownCache cache;
  const std::size_t n = panels.size();
  cache.phi.assign(n, 0.0);
  cache.phin.assign(n, 0.0);
  cache.ordered_phi.assign(n, 0.0);
  cache.ordered_phin.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    cache.phi[i] = i < unknowns.size() ? unknowns[i] : 0.0;
    if (coupled && n + i < unknowns.size())
      cache.phin[i] = unknowns[n + i];
  }
  for (std::size_t pos = 0; pos < n; ++pos) {
    const std::size_t original = pos < order.size() ? order[pos] : pos;
    cache.ordered_phi[pos] = original < cache.phi.size() ? cache.phi[original] : 0.0;
    cache.ordered_phin[pos] =
        original < cache.phin.size() ? cache.phin[original] : 0.0;
  }
  return cache;
}

inline bool accumulate_jcp_leaf_source(
    JcpAccum &acc, double tx, double ty, double tz, double tnx, double tny,
    double tnz, double sx, double sy, double sz, double snx, double sny,
    double snz, double area, double phi_j, double phin_j, double eps_ratio,
    double inv_eps_ratio, double poisson_scale, double inv_four_pi,
    double kk, bool coupled) noexcept {
  const double ux = tx - sx;
  const double uy = ty - sy;
  const double uz = tz - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return false;
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double source_projection = snx * ux + sny * uy + snz * uz;
  const double target_projection = tnx * ux + tny * uy + tnz * uz;
  const double source_d0 = inv_four_pi * source_projection * inv_r3;
  if (coupled) {
    const double expkr = std::exp(-kk * r);
    const double one_plus_kr = 1.0 + kk * r;
    const double source_dk = expkr * one_plus_kr * source_d0;
    const double g0 = inv_four_pi * inv_r;
    const double gk = expkr * g0;
    const double target_d0 = -inv_four_pi * target_projection * inv_r3;
    const double target_dk = expkr * one_plus_kr * target_d0;
    const double nxu = target_projection * inv_r;
    const double nyu = source_projection * inv_r;
    const double nxny = tnx * snx + tny * sny + tnz * snz;
    const double nxu_nyu = nxu * nyu;
    const double tangent_metric = nxny - nxu_nyu;
    const double mixed0 = inv_four_pi * inv_r3 * (nxny - 3.0 * nxu_nyu);
    const double fprime_k_over_r =
        -expkr * one_plus_kr * inv_four_pi * inv_r3;
    const double fsecond_k =
        expkr * inv_four_pi *
        (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r3);
    const double mixedk =
        -fsecond_k * nxu_nyu - fprime_k_over_r * tangent_metric;
    acc.phi += area * ((g0 - gk) * phin_j +
                       (eps_ratio * source_dk - source_d0) * phi_j);
    acc.phin += area * ((target_d0 - inv_eps_ratio * target_dk) * phin_j +
                        (mixedk - mixed0) * phi_j);
  } else {
    acc.phi += area * poisson_scale * source_d0 * phi_j;
  }
  return true;
}

inline bool accumulate_jcp_leaf_source_channels(
    JcpChannelAccum &acc, double tx, double ty, double tz, double tnx,
    double tny, double tnz, double sx, double sy, double sz, double snx,
    double sny, double snz, double area, double phi_j, double phin_j,
    double eps_ratio, double inv_eps_ratio, double poisson_scale,
    double inv_four_pi, double kk, bool coupled) noexcept {
  const double ux = tx - sx;
  const double uy = ty - sy;
  const double uz = tz - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return false;
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double inv_r2 = inv_r * inv_r;
  const double inv_r3 = inv_r2 * inv_r;
  const double source_projection = snx * ux + sny * uy + snz * uz;
  const double target_projection = tnx * ux + tny * uy + tnz * uz;
  const double source_d0 = inv_four_pi * source_projection * inv_r3;
  if (coupled) {
    const double expkr = std::exp(-kk * r);
    const double one_plus_kr = 1.0 + kk * r;
    const double source_dk = expkr * one_plus_kr * source_d0;
    const double g0 = inv_four_pi * inv_r;
    const double gk = expkr * g0;
    const double target_d0 = -inv_four_pi * target_projection * inv_r3;
    const double target_dk = expkr * one_plus_kr * target_d0;
    const double nxu = target_projection * inv_r;
    const double nyu = source_projection * inv_r;
    const double nxny = tnx * snx + tny * sny + tnz * snz;
    const double nxu_nyu = nxu * nyu;
    const double tangent_metric = nxny - nxu_nyu;
    const double mixed0 = inv_four_pi * inv_r3 * (nxny - 3.0 * nxu_nyu);
    const double fprime_k_over_r =
        -expkr * one_plus_kr * inv_four_pi * inv_r3;
    const double fsecond_k =
        expkr * inv_four_pi *
        (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r3);
    const double mixedk =
        -fsecond_k * nxu_nyu - fprime_k_over_r * tangent_metric;
    acc.channel[kJcpK1PhiChannel].phi += area * (g0 - gk) * phin_j;
    acc.channel[kJcpK2PhiNormalChannel].phi +=
        area * (eps_ratio * source_dk - source_d0) * phi_j;
    acc.channel[kJcpK3PhinChannel].phin +=
        area * (target_d0 - inv_eps_ratio * target_dk) * phin_j;
    acc.channel[kJcpK4MixedChannel].phin += area * (mixedk - mixed0) * phi_j;
  } else {
    acc.channel[kJcpK2PhiNormalChannel].phi +=
        area * poisson_scale * source_d0 * phi_j;
  }
  return true;
}

JcpAccum direct_jcp_ordered_sum_scalar(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, std::size_t &near_count) noexcept {
  JcpAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = begin + count;
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t j = order[pos];
    if (j == target_index)
      continue;
    if (accumulate_jcp_leaf_source(
        acc, target.x, target.y, target.z, target.nx, target.ny, target.nz,
        panel_cache.x[j], panel_cache.y[j], panel_cache.z[j],
        panel_cache.nx[j], panel_cache.ny[j], panel_cache.nz[j],
        panel_cache.area[j], unknown_cache.phi[j], unknown_cache.phin[j],
        eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled))
      ++near_count;
  }
  return acc;
}

JcpChannelAccum direct_jcp_ordered_channel_sum_scalar(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, std::size_t &near_count) noexcept {
  JcpChannelAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = begin + count;
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t j = order[pos];
    if (j == target_index)
      continue;
    if (accumulate_jcp_leaf_source_channels(
            acc, target.x, target.y, target.z, target.nx, target.ny,
            target.nz, panel_cache.x[j], panel_cache.y[j], panel_cache.z[j],
            panel_cache.nx[j], panel_cache.ny[j], panel_cache.nz[j],
            panel_cache.area[j], unknown_cache.phi[j],
            unknown_cache.phin[j], eps_ratio, inv_eps_ratio, poisson_scale,
            inv_four_pi, kk, coupled))
      ++near_count;
  }
  return acc;
}

inline bool accumulate_jcp_ordered_tile_position(
    JcpAccum &acc, const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double inv_eps_ratio, double poisson_scale,
    double inv_four_pi, double kk, bool coupled, std::size_t pos) noexcept {
  const std::size_t original =
      pos < panel_cache.ordered_original_index.size()
          ? panel_cache.ordered_original_index[pos]
          : pos;
  if (original == target_index)
    return false;
  return accumulate_jcp_leaf_source(
      acc, target.x, target.y, target.z, target.nx, target.ny, target.nz,
      panel_cache.ordered_x[pos], panel_cache.ordered_y[pos],
      panel_cache.ordered_z[pos], panel_cache.ordered_nx[pos],
      panel_cache.ordered_ny[pos], panel_cache.ordered_nz[pos],
      panel_cache.ordered_area[pos], unknown_cache.ordered_phi[pos],
      unknown_cache.ordered_phin[pos], eps_ratio, inv_eps_ratio,
      poisson_scale, inv_four_pi, kk, coupled);
}

bool tile_position_is_excluded(const TreePanelCache &panel_cache,
                               std::size_t target_index, std::size_t pos,
                               const std::vector<std::size_t> *exclusions,
                               std::size_t exclusion_begin,
                               std::size_t exclusion_count) noexcept {
  const std::size_t original =
      pos < panel_cache.ordered_original_index.size()
          ? panel_cache.ordered_original_index[pos]
          : pos;
  if (original == target_index)
    return true;
  if (exclusions == nullptr || exclusion_count == 0)
    return false;
  const std::size_t end = exclusion_begin + exclusion_count;
  for (std::size_t k = exclusion_begin; k < end; ++k) {
    if ((*exclusions)[k] == original)
      return true;
  }
  return false;
}

inline bool accumulate_jcp_ordered_tile_position_masked(
    JcpAccum &acc, const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double inv_eps_ratio, double poisson_scale,
    double inv_four_pi, double kk, bool coupled, std::size_t pos,
    const std::vector<std::size_t> *exclusions, std::size_t exclusion_begin,
    std::size_t exclusion_count) noexcept {
  if (tile_position_is_excluded(panel_cache, target_index, pos, exclusions,
                                exclusion_begin, exclusion_count))
    return false;
  return accumulate_jcp_ordered_tile_position(
      acc, target, target_index, panel_cache, unknown_cache, eps_ratio,
      inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled, pos);
}

JcpAccum direct_jcp_ordered_tile_sum_scalar(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::size_t &near_count,
    const std::vector<std::size_t> *exclusions = nullptr,
    std::size_t exclusion_begin = 0,
    std::size_t exclusion_count = 0) noexcept {
  JcpAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = node.first_panel + node.panel_count;
  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    if (accumulate_jcp_ordered_tile_position_masked(
            acc, target, target_index, panel_cache, unknown_cache, eps_ratio,
            inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled, pos,
            exclusions, exclusion_begin, exclusion_count))
      ++near_count;
  }
  return acc;
}

#if TITAN_SIMD_X86
struct LeafTargetAvx2 {
  __m256d x, y, z, nx, ny, nz;
};

TITAN_TARGET_AVX2
inline LeafTargetAvx2 leaf_target_avx2(const Panel &target) noexcept {
  return {_mm256_set1_pd(target.x),  _mm256_set1_pd(target.y),
          _mm256_set1_pd(target.z),  _mm256_set1_pd(target.nx),
          _mm256_set1_pd(target.ny), _mm256_set1_pd(target.nz)};
}

TITAN_TARGET_AVX2
inline __m256d avx2_negate_pd(__m256d v) noexcept {
  return _mm256_xor_pd(v, _mm256_set1_pd(-0.0));
}

// accumulate_jcp_leaf_source for four sources, with the scalar operation
// order; exp is titan_avx2::exp_pd. Lanes set in `excluded`, or with
// r2 <= 1e-28, add nothing (NaN geometry still propagates, as in the scalar
// test). Returns the lanes that count as near interactions.
TITAN_TARGET_AVX2
inline __m256d accumulate_jcp_leaf_sources_avx2(
    __m256d &acc_phi, __m256d &acc_phin, const LeafTargetAvx2 &t, __m256d sx,
    __m256d sy, __m256d sz, __m256d snx, __m256d sny, __m256d snz,
    __m256d area, __m256d phi_j, __m256d phin_j, __m256d excluded,
    double eps_ratio, double inv_eps_ratio, double poisson_scale,
    double inv_four_pi, double kk, bool coupled) noexcept {
  const __m256d ux = _mm256_sub_pd(t.x, sx);
  const __m256d uy = _mm256_sub_pd(t.y, sy);
  const __m256d uz = _mm256_sub_pd(t.z, sz);
  const __m256d r2 = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(ux, ux), _mm256_mul_pd(uy, uy)),
      _mm256_mul_pd(uz, uz));
  const __m256d valid = _mm256_andnot_pd(
      excluded, _mm256_cmp_pd(r2, _mm256_set1_pd(1.0e-28), _CMP_NLE_UQ));
  const __m256d r = _mm256_sqrt_pd(r2);
  const __m256d inv_r = _mm256_div_pd(_mm256_set1_pd(1.0), r);
  const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
  const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
  const __m256d source_projection = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(snx, ux), _mm256_mul_pd(sny, uy)),
      _mm256_mul_pd(snz, uz));
  const __m256d inv_four_pi_v = _mm256_set1_pd(inv_four_pi);
  const __m256d source_d0 = _mm256_mul_pd(
      _mm256_mul_pd(inv_four_pi_v, source_projection), inv_r3);
  if (!coupled) {
    const __m256d term = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(area, _mm256_set1_pd(poisson_scale)),
                      source_d0),
        phi_j);
    acc_phi = _mm256_add_pd(acc_phi, _mm256_and_pd(term, valid));
    return valid;
  }
  const __m256d target_projection = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(t.nx, ux), _mm256_mul_pd(t.ny, uy)),
      _mm256_mul_pd(t.nz, uz));
  const __m256d kk_v = _mm256_set1_pd(kk);
  const __m256d expkr =
      titan_avx2::exp_pd(_mm256_mul_pd(_mm256_set1_pd(-kk), r));
  const __m256d one_plus_kr =
      _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(kk_v, r));
  const __m256d exp_one_plus_kr = _mm256_mul_pd(expkr, one_plus_kr);
  const __m256d source_dk = _mm256_mul_pd(exp_one_plus_kr, source_d0);
  const __m256d g0 = _mm256_mul_pd(inv_four_pi_v, inv_r);
  const __m256d gk = _mm256_mul_pd(expkr, g0);
  const __m256d target_d0 = _mm256_mul_pd(
      _mm256_mul_pd(_mm256_set1_pd(-inv_four_pi), target_projection), inv_r3);
  const __m256d target_dk = _mm256_mul_pd(exp_one_plus_kr, target_d0);
  const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
  const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
  const __m256d nxny = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(t.nx, snx), _mm256_mul_pd(t.ny, sny)),
      _mm256_mul_pd(t.nz, snz));
  const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
  const __m256d tangent_metric = _mm256_sub_pd(nxny, nxu_nyu);
  const __m256d mixed0 = _mm256_mul_pd(
      _mm256_mul_pd(inv_four_pi_v, inv_r3),
      _mm256_sub_pd(nxny, _mm256_mul_pd(_mm256_set1_pd(3.0), nxu_nyu)));
  const __m256d fprime_k_over_r = _mm256_mul_pd(
      _mm256_mul_pd(
          _mm256_mul_pd(avx2_negate_pd(expkr), one_plus_kr), inv_four_pi_v),
      inv_r3);
  const __m256d fsecond_k = _mm256_mul_pd(
      _mm256_mul_pd(expkr, inv_four_pi_v),
      _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(kk * kk), inv_r),
                        _mm256_mul_pd(_mm256_set1_pd(2.0 * kk), inv_r2)),
          _mm256_mul_pd(_mm256_set1_pd(2.0), inv_r3)));
  const __m256d mixedk =
      _mm256_sub_pd(_mm256_mul_pd(avx2_negate_pd(fsecond_k), nxu_nyu),
                    _mm256_mul_pd(fprime_k_over_r, tangent_metric));
  const __m256d phi_term = _mm256_mul_pd(
      area,
      _mm256_add_pd(
          _mm256_mul_pd(_mm256_sub_pd(g0, gk), phin_j),
          _mm256_mul_pd(
              _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(eps_ratio),
                                          source_dk),
                            source_d0),
              phi_j)));
  const __m256d phin_term = _mm256_mul_pd(
      area,
      _mm256_add_pd(
          _mm256_mul_pd(
              _mm256_sub_pd(target_d0,
                            _mm256_mul_pd(_mm256_set1_pd(inv_eps_ratio),
                                          target_dk)),
              phin_j),
          _mm256_mul_pd(_mm256_sub_pd(mixedk, mixed0), phi_j)));
  acc_phi = _mm256_add_pd(acc_phi, _mm256_and_pd(phi_term, valid));
  acc_phin = _mm256_add_pd(acc_phin, _mm256_and_pd(phin_term, valid));
  return valid;
}

TITAN_TARGET_AVX2
inline std::size_t avx2_lane_count(__m256d mask) noexcept {
  return static_cast<std::size_t>(
      std::popcount(static_cast<unsigned>(_mm256_movemask_pd(mask))));
}

TITAN_TARGET_AVX2
JcpAccum direct_jcp_ordered_tile_sum_avx2(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::size_t &near_count,
    const std::vector<std::size_t> *exclusions = nullptr,
    std::size_t exclusion_begin = 0,
    std::size_t exclusion_count = 0) noexcept {
  JcpAccum acc;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = node.first_panel + node.panel_count;
  std::size_t pos = node.first_panel;
  while (pos < end && (pos & std::size_t{3}) != 0) {
    if (accumulate_jcp_ordered_tile_position_masked(
            acc, target, target_index, panel_cache, unknown_cache, eps_ratio,
            inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled, pos,
            exclusions, exclusion_begin, exclusion_count))
      ++near_count;
    ++pos;
  }

  const LeafTargetAvx2 t = leaf_target_avx2(target);
  const __m256d none = _mm256_setzero_pd();
  __m256d acc_phi = _mm256_setzero_pd();
  __m256d acc_phin = _mm256_setzero_pd();
  for (; pos + 4 <= end; pos += 4) {
    bool masked_lane = false;
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (tile_position_is_excluded(panel_cache, target_index, pos + lane,
                                    exclusions, exclusion_begin,
                                    exclusion_count)) {
        masked_lane = true;
        break;
      }
    }
    if (masked_lane) {
      for (std::size_t lane = 0; lane < 4; ++lane) {
        if (accumulate_jcp_ordered_tile_position_masked(
                acc, target, target_index, panel_cache, unknown_cache,
                eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk,
                coupled, pos + lane, exclusions, exclusion_begin,
                exclusion_count))
          ++near_count;
      }
      continue;
    }
    near_count += avx2_lane_count(accumulate_jcp_leaf_sources_avx2(
        acc_phi, acc_phin, t, _mm256_load_pd(panel_cache.ordered_x.data() + pos),
        _mm256_load_pd(panel_cache.ordered_y.data() + pos),
        _mm256_load_pd(panel_cache.ordered_z.data() + pos),
        _mm256_load_pd(panel_cache.ordered_nx.data() + pos),
        _mm256_load_pd(panel_cache.ordered_ny.data() + pos),
        _mm256_load_pd(panel_cache.ordered_nz.data() + pos),
        _mm256_load_pd(panel_cache.ordered_area.data() + pos),
        _mm256_load_pd(unknown_cache.ordered_phi.data() + pos),
        _mm256_load_pd(unknown_cache.ordered_phin.data() + pos), none,
        eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled));
  }
  acc.phi += titan_avx2::hsum_pd(acc_phi);
  if (coupled)
    acc.phin += titan_avx2::hsum_pd(acc_phin);

  for (; pos < end; ++pos) {
    if (accumulate_jcp_ordered_tile_position_masked(
            acc, target, target_index, panel_cache, unknown_cache, eps_ratio,
            inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled, pos,
            exclusions, exclusion_begin, exclusion_count))
      ++near_count;
  }
  return acc;
}
#endif

JcpAccum direct_jcp_ordered_tile_sum(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    const std::string &kernel, std::size_t &near_count,
    const std::vector<std::size_t> *exclusions,
    std::size_t exclusion_begin,
    std::size_t exclusion_count) noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (kernel == "tile-avx2") {
    return direct_jcp_ordered_tile_sum_avx2(
        target, target_index, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, node, near_count, exclusions, exclusion_begin,
        exclusion_count);
  }
#endif
  return direct_jcp_ordered_tile_sum_scalar(
      target, target_index, panel_cache, unknown_cache, eps_ratio, kappa,
      coupled, node, near_count, exclusions, exclusion_begin,
      exclusion_count);
}

void direct_jcp_target_batch_tile_sum_scalar(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::vector<JcpAccum> &accum, std::size_t &near_count) noexcept {
  accum.assign(target_indices.size(), JcpAccum{});
  if (target_indices.empty())
    return;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = node.first_panel + node.panel_count;
  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    const std::size_t original =
        pos < panel_cache.ordered_original_index.size()
            ? panel_cache.ordered_original_index[pos]
            : pos;
    const double sx = panel_cache.ordered_x[pos];
    const double sy = panel_cache.ordered_y[pos];
    const double sz = panel_cache.ordered_z[pos];
    const double snx = panel_cache.ordered_nx[pos];
    const double sny = panel_cache.ordered_ny[pos];
    const double snz = panel_cache.ordered_nz[pos];
    const double area = panel_cache.ordered_area[pos];
    const double phi = unknown_cache.ordered_phi[pos];
    const double phin = unknown_cache.ordered_phin[pos];
    for (std::size_t t = 0; t < target_indices.size(); ++t) {
      const std::size_t target_index = target_indices[t];
      if (original == target_index || target_index >= panels.size())
        continue;
      const Panel &target = panels[target_index];
      if (accumulate_jcp_leaf_source(
              accum[t], target.x, target.y, target.z, target.nx, target.ny,
              target.nz, sx, sy, sz, snx, sny, snz, area, phi, phin,
              eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk,
              coupled))
        ++near_count;
    }
  }
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
inline __m256d avx2_lane_mask_pd(unsigned bits) noexcept {
  return _mm256_castsi256_pd(_mm256_set_epi64x(
      (bits & 0x8u) ? -1ll : 0ll, (bits & 0x4u) ? -1ll : 0ll,
      (bits & 0x2u) ? -1ll : 0ll, (bits & 0x1u) ? -1ll : 0ll));
}

// Double-precision exp for the screened direct kernel (titan_avx2_math.h).
TITAN_TARGET_AVX2
inline __m256d avx2_exp_accurate_pd(__m256d x) noexcept {
  return titan_avx2::exp_pd(x);
}

inline std::size_t popcount4(unsigned bits) noexcept {
  bits &= 0x0fu;
  return static_cast<std::size_t>((bits & 0x1u) + ((bits >> 1) & 0x1u) +
                                  ((bits >> 2) & 0x1u) +
                                  ((bits >> 3) & 0x1u));
}

TITAN_TARGET_AVX2
void direct_jcp_target_batch_tile_sum_avx2(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::vector<JcpAccum> &accum, std::size_t &near_count,
    std::size_t &vectorized_sources, std::size_t &vector_lanes,
    bool branchless_masked = false) noexcept {
  accum.assign(target_indices.size(), JcpAccum{});
  if (target_indices.empty())
    return;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = node.first_panel + node.panel_count;
  const __m256d eps_v = _mm256_set1_pd(eps_ratio);
  const __m256d inv_eps_v = _mm256_set1_pd(inv_eps_ratio);
  const __m256d poisson_scale_v = _mm256_set1_pd(poisson_scale);
  const __m256d inv_four_pi_v = _mm256_set1_pd(inv_four_pi);
  const __m256d kk_v = _mm256_set1_pd(kk);
  const __m256d one_v = _mm256_set1_pd(1.0);
  const __m256d two_v = _mm256_set1_pd(2.0);
  const __m256d three_v = _mm256_set1_pd(3.0);
  const __m256d neg_one_v = _mm256_set1_pd(-1.0);
  const __m256d r2_floor_v = _mm256_set1_pd(1.0e-28);
  alignas(32) double tx_lane[4];
  alignas(32) double ty_lane[4];
  alignas(32) double tz_lane[4];
  alignas(32) double tnx_lane[4];
  alignas(32) double tny_lane[4];
  alignas(32) double tnz_lane[4];
  alignas(32) double r_lane[4];
  alignas(32) double exp_lane[4];
  alignas(32) double phi_lane[4];
  alignas(32) double phin_lane[4];

  if (target_indices.size() == 4) {
    bool targets_valid = true;
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const std::size_t target_index = target_indices[lane];
      if (target_index >= panels.size()) {
        targets_valid = false;
        break;
      }
      const Panel &target = panels[target_index];
      tx_lane[lane] = target.x;
      ty_lane[lane] = target.y;
      tz_lane[lane] = target.z;
      tnx_lane[lane] = target.nx;
      tny_lane[lane] = target.ny;
      tnz_lane[lane] = target.nz;
    }
    if (targets_valid) {
      const __m256d tx = _mm256_load_pd(tx_lane);
      const __m256d ty = _mm256_load_pd(ty_lane);
      const __m256d tz = _mm256_load_pd(tz_lane);
      const __m256d tnx = _mm256_load_pd(tnx_lane);
      const __m256d tny = _mm256_load_pd(tny_lane);
      const __m256d tnz = _mm256_load_pd(tnz_lane);
      __m256d acc_phi_v = _mm256_setzero_pd();
      __m256d acc_phin_v = _mm256_setzero_pd();
      for (std::size_t pos = node.first_panel; pos < end; ++pos) {
        const std::size_t original =
            pos < panel_cache.ordered_original_index.size()
                ? panel_cache.ordered_original_index[pos]
                : pos;
        unsigned lane_bits = 0x0fu;
        for (std::size_t lane = 0; lane < 4; ++lane) {
          if (original == target_indices[lane])
            lane_bits &= ~(1u << lane);
        }
        if (!branchless_masked && lane_bits != 0x0fu) {
          for (std::size_t lane = 0; lane < 4; ++lane) {
            if (original == target_indices[lane])
              continue;
            const Panel &target = panels[target_indices[lane]];
            if (accumulate_jcp_leaf_source(
                    accum[lane], target.x, target.y, target.z, target.nx,
                    target.ny, target.nz, panel_cache.ordered_x[pos],
                    panel_cache.ordered_y[pos], panel_cache.ordered_z[pos],
                    panel_cache.ordered_nx[pos], panel_cache.ordered_ny[pos],
                    panel_cache.ordered_nz[pos],
                    panel_cache.ordered_area[pos],
                    unknown_cache.ordered_phi[pos],
                    unknown_cache.ordered_phin[pos], eps_ratio,
                    inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled))
              ++near_count;
          }
          continue;
        }
        const double sx = panel_cache.ordered_x[pos];
        const double sy = panel_cache.ordered_y[pos];
        const double sz = panel_cache.ordered_z[pos];
        const __m256d ux = _mm256_sub_pd(tx, _mm256_set1_pd(sx));
        const __m256d uy = _mm256_sub_pd(ty, _mm256_set1_pd(sy));
        const __m256d uz = _mm256_sub_pd(tz, _mm256_set1_pd(sz));
        const __m256d r2 = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(ux, ux), _mm256_mul_pd(uy, uy)),
            _mm256_mul_pd(uz, uz));
        const int valid_mask =
            _mm256_movemask_pd(_mm256_cmp_pd(r2, r2_floor_v, _CMP_GT_OQ));
        const unsigned valid_bits =
            lane_bits & static_cast<unsigned>(valid_mask);
        if (!branchless_masked && valid_bits != 0x0fu) {
          for (std::size_t lane = 0; lane < 4; ++lane) {
            const Panel &target = panels[target_indices[lane]];
            if (accumulate_jcp_leaf_source(
                    accum[lane], target.x, target.y, target.z, target.nx,
                    target.ny, target.nz, sx, sy, sz,
                    panel_cache.ordered_nx[pos],
                    panel_cache.ordered_ny[pos],
                    panel_cache.ordered_nz[pos],
                    panel_cache.ordered_area[pos],
                    unknown_cache.ordered_phi[pos],
                    unknown_cache.ordered_phin[pos], eps_ratio,
                    inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled))
              ++near_count;
          }
          continue;
        }
        const __m256d lane_mask = avx2_lane_mask_pd(valid_bits);
        const __m256d r2_safe =
            branchless_masked ? _mm256_max_pd(r2, r2_floor_v) : r2;
        const __m256d r = _mm256_sqrt_pd(r2_safe);
        const __m256d inv_r = _mm256_div_pd(one_v, r);
        const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
        const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
        const __m256d snx_v = _mm256_set1_pd(panel_cache.ordered_nx[pos]);
        const __m256d sny_v = _mm256_set1_pd(panel_cache.ordered_ny[pos]);
        const __m256d snz_v = _mm256_set1_pd(panel_cache.ordered_nz[pos]);
        const __m256d area =
            _mm256_set1_pd(panel_cache.ordered_area[pos]);
        const __m256d phi_j =
            _mm256_set1_pd(unknown_cache.ordered_phi[pos]);
        const __m256d phin_j =
            _mm256_set1_pd(unknown_cache.ordered_phin[pos]);
        const __m256d source_projection = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(snx_v, ux),
                          _mm256_mul_pd(sny_v, uy)),
            _mm256_mul_pd(snz_v, uz));
        const __m256d target_projection = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(tnx, ux), _mm256_mul_pd(tny, uy)),
            _mm256_mul_pd(tnz, uz));
        const __m256d source_d0 = _mm256_mul_pd(
            inv_four_pi_v, _mm256_mul_pd(source_projection, inv_r3));
        if (coupled) {
          _mm256_store_pd(r_lane, r);
          for (int lane = 0; lane < 4; ++lane)
            exp_lane[lane] = std::exp(-kk * r_lane[lane]);
          const __m256d expkr = _mm256_load_pd(exp_lane);
          const __m256d one_plus_kr =
              _mm256_add_pd(one_v, _mm256_mul_pd(kk_v, r));
          const __m256d source_dk =
              _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), source_d0);
          const __m256d g0 = _mm256_mul_pd(inv_four_pi_v, inv_r);
          const __m256d gk = _mm256_mul_pd(expkr, g0);
          const __m256d target_d0 = _mm256_mul_pd(
              neg_one_v,
              _mm256_mul_pd(inv_four_pi_v,
                            _mm256_mul_pd(target_projection, inv_r3)));
          const __m256d target_dk =
              _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), target_d0);
          const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
          const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
          const __m256d nxny = _mm256_add_pd(
              _mm256_add_pd(_mm256_mul_pd(tnx, snx_v),
                            _mm256_mul_pd(tny, sny_v)),
              _mm256_mul_pd(tnz, snz_v));
          const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
          const __m256d tangent_metric = _mm256_sub_pd(nxny, nxu_nyu);
          const __m256d mixed0 = _mm256_mul_pd(
              inv_four_pi_v,
              _mm256_mul_pd(inv_r3,
                            _mm256_sub_pd(nxny, _mm256_mul_pd(three_v, nxu_nyu))));
          const __m256d fprime_k_over_r = _mm256_mul_pd(
              neg_one_v,
              _mm256_mul_pd(
                  _mm256_mul_pd(expkr, one_plus_kr),
                  _mm256_mul_pd(inv_four_pi_v, inv_r3)));
          const __m256d fsecond_k = _mm256_mul_pd(
              expkr,
              _mm256_mul_pd(
                  inv_four_pi_v,
                  _mm256_add_pd(
                      _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(kk_v, kk_v),
                                                  inv_r),
                                    _mm256_mul_pd(_mm256_mul_pd(two_v, kk_v),
                                                  inv_r2)),
                      _mm256_mul_pd(two_v, inv_r3))));
          const __m256d mixedk = _mm256_sub_pd(
              _mm256_setzero_pd(), _mm256_add_pd(
                  _mm256_mul_pd(fsecond_k, nxu_nyu),
                  _mm256_mul_pd(fprime_k_over_r, tangent_metric)));
          __m256d phi_term = _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(_mm256_sub_pd(g0, gk), phin_j),
                  _mm256_mul_pd(
                      _mm256_sub_pd(_mm256_mul_pd(eps_v, source_dk),
                                    source_d0),
                      phi_j)));
          __m256d phin_term = _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(
                      _mm256_sub_pd(target_d0,
                                    _mm256_mul_pd(inv_eps_v, target_dk)),
                      phin_j),
                  _mm256_mul_pd(_mm256_sub_pd(mixedk, mixed0), phi_j)));
          if (branchless_masked) {
            phi_term = _mm256_and_pd(phi_term, lane_mask);
            phin_term = _mm256_and_pd(phin_term, lane_mask);
          }
          acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
          acc_phin_v = _mm256_add_pd(acc_phin_v, phin_term);
        } else {
          __m256d phi_term = _mm256_mul_pd(
              area,
              _mm256_mul_pd(poisson_scale_v,
                            _mm256_mul_pd(source_d0, phi_j)));
          if (branchless_masked)
            phi_term = _mm256_and_pd(phi_term, lane_mask);
          acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
        }
        near_count += branchless_masked ? popcount4(valid_bits) : 4;
        ++vectorized_sources;
        vector_lanes += 4;
      }
      _mm256_store_pd(phi_lane, acc_phi_v);
      if (coupled)
        _mm256_store_pd(phin_lane, acc_phin_v);
      for (std::size_t lane = 0; lane < 4; ++lane) {
        accum[lane].phi += phi_lane[lane];
        if (coupled)
          accum[lane].phin += phin_lane[lane];
      }
      return;
    }
  }

  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    const std::size_t original =
        pos < panel_cache.ordered_original_index.size()
            ? panel_cache.ordered_original_index[pos]
            : pos;
    const double sx = panel_cache.ordered_x[pos];
    const double sy = panel_cache.ordered_y[pos];
    const double sz = panel_cache.ordered_z[pos];
    const double snx = panel_cache.ordered_nx[pos];
    const double sny = panel_cache.ordered_ny[pos];
    const double snz = panel_cache.ordered_nz[pos];
    const double area_scalar = panel_cache.ordered_area[pos];
    const double phi_scalar = unknown_cache.ordered_phi[pos];
    const double phin_scalar = unknown_cache.ordered_phin[pos];
    std::size_t t = 0;
    for (; t + 4 <= target_indices.size(); t += 4) {
      bool clean_chunk = true;
      unsigned lane_bits = 0x0fu;
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const std::size_t target_index = target_indices[t + lane];
        if (target_index >= panels.size()) {
          clean_chunk = false;
          break;
        }
        if (target_index == original)
          lane_bits &= ~(1u << lane);
        if (!branchless_masked && target_index == original) {
          clean_chunk = false;
          break;
        }
        const Panel &target = panels[target_index];
        tx_lane[lane] = target.x;
        ty_lane[lane] = target.y;
        tz_lane[lane] = target.z;
        tnx_lane[lane] = target.nx;
        tny_lane[lane] = target.ny;
        tnz_lane[lane] = target.nz;
      }
      if (!clean_chunk) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const std::size_t target_index = target_indices[t + lane];
          if (target_index >= panels.size() || target_index == original)
            continue;
          const Panel &target = panels[target_index];
          if (accumulate_jcp_leaf_source(
                  accum[t + lane], target.x, target.y, target.z, target.nx,
                  target.ny, target.nz, sx, sy, sz, snx, sny, snz,
                  area_scalar, phi_scalar, phin_scalar, eps_ratio,
                  inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled))
            ++near_count;
        }
        continue;
      }
      const __m256d tx = _mm256_load_pd(tx_lane);
      const __m256d ty = _mm256_load_pd(ty_lane);
      const __m256d tz = _mm256_load_pd(tz_lane);
      const __m256d tnx = _mm256_load_pd(tnx_lane);
      const __m256d tny = _mm256_load_pd(tny_lane);
      const __m256d tnz = _mm256_load_pd(tnz_lane);
      const __m256d ux = _mm256_sub_pd(tx, _mm256_set1_pd(sx));
      const __m256d uy = _mm256_sub_pd(ty, _mm256_set1_pd(sy));
      const __m256d uz = _mm256_sub_pd(tz, _mm256_set1_pd(sz));
      const __m256d r2 = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(ux, ux), _mm256_mul_pd(uy, uy)),
          _mm256_mul_pd(uz, uz));
      const int valid_mask =
          _mm256_movemask_pd(_mm256_cmp_pd(r2, r2_floor_v, _CMP_GT_OQ));
      const unsigned valid_bits =
          lane_bits & static_cast<unsigned>(valid_mask);
      if (!branchless_masked && valid_bits != 0x0fu) {
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const std::size_t target_index = target_indices[t + lane];
          const Panel &target = panels[target_index];
          if (accumulate_jcp_leaf_source(
                  accum[t + lane], target.x, target.y, target.z, target.nx,
                  target.ny, target.nz, sx, sy, sz, snx, sny, snz,
                  area_scalar, phi_scalar, phin_scalar, eps_ratio,
                  inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled))
            ++near_count;
        }
        continue;
      }
      const __m256d lane_mask = avx2_lane_mask_pd(valid_bits);
      const __m256d r2_safe =
          branchless_masked ? _mm256_max_pd(r2, r2_floor_v) : r2;
      const __m256d r = _mm256_sqrt_pd(r2_safe);
      const __m256d inv_r = _mm256_div_pd(one_v, r);
      const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
      const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
      const __m256d snx_v = _mm256_set1_pd(snx);
      const __m256d sny_v = _mm256_set1_pd(sny);
      const __m256d snz_v = _mm256_set1_pd(snz);
      const __m256d area = _mm256_set1_pd(area_scalar);
      const __m256d phi_j = _mm256_set1_pd(phi_scalar);
      const __m256d phin_j = _mm256_set1_pd(phin_scalar);
      const __m256d source_projection = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(snx_v, ux),
                        _mm256_mul_pd(sny_v, uy)),
          _mm256_mul_pd(snz_v, uz));
      const __m256d target_projection = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(tnx, ux), _mm256_mul_pd(tny, uy)),
          _mm256_mul_pd(tnz, uz));
      const __m256d source_d0 = _mm256_mul_pd(
          inv_four_pi_v, _mm256_mul_pd(source_projection, inv_r3));
      if (coupled) {
        _mm256_store_pd(r_lane, r);
        for (int lane = 0; lane < 4; ++lane)
          exp_lane[lane] = std::exp(-kk * r_lane[lane]);
        const __m256d expkr = _mm256_load_pd(exp_lane);
        const __m256d one_plus_kr =
            _mm256_add_pd(one_v, _mm256_mul_pd(kk_v, r));
        const __m256d source_dk =
            _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), source_d0);
        const __m256d g0 = _mm256_mul_pd(inv_four_pi_v, inv_r);
        const __m256d gk = _mm256_mul_pd(expkr, g0);
        const __m256d target_d0 = _mm256_mul_pd(
            neg_one_v,
            _mm256_mul_pd(inv_four_pi_v,
                          _mm256_mul_pd(target_projection, inv_r3)));
        const __m256d target_dk =
            _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), target_d0);
        const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
        const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
        const __m256d nxny = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(tnx, snx_v),
                          _mm256_mul_pd(tny, sny_v)),
            _mm256_mul_pd(tnz, snz_v));
        const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
        const __m256d tangent_metric = _mm256_sub_pd(nxny, nxu_nyu);
        const __m256d mixed0 = _mm256_mul_pd(
            inv_four_pi_v,
            _mm256_mul_pd(inv_r3,
                          _mm256_sub_pd(nxny, _mm256_mul_pd(three_v, nxu_nyu))));
        const __m256d fprime_k_over_r = _mm256_mul_pd(
            neg_one_v,
            _mm256_mul_pd(
                _mm256_mul_pd(expkr, one_plus_kr),
                _mm256_mul_pd(inv_four_pi_v, inv_r3)));
        const __m256d fsecond_k = _mm256_mul_pd(
            expkr,
            _mm256_mul_pd(
                inv_four_pi_v,
                _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(kk_v, kk_v),
                                                inv_r),
                                  _mm256_mul_pd(_mm256_mul_pd(two_v, kk_v),
                                                inv_r2)),
                    _mm256_mul_pd(two_v, inv_r3))));
        const __m256d mixedk = _mm256_sub_pd(
            _mm256_setzero_pd(), _mm256_add_pd(
                _mm256_mul_pd(fsecond_k, nxu_nyu),
                _mm256_mul_pd(fprime_k_over_r, tangent_metric)));
        const __m256d phi_term = _mm256_mul_pd(
            area,
            _mm256_add_pd(
                _mm256_mul_pd(_mm256_sub_pd(g0, gk), phin_j),
                _mm256_mul_pd(
                    _mm256_sub_pd(_mm256_mul_pd(eps_v, source_dk),
                                  source_d0),
                    phi_j)));
        const __m256d phin_term = _mm256_mul_pd(
            area,
            _mm256_add_pd(
                _mm256_mul_pd(
                    _mm256_sub_pd(target_d0,
                                  _mm256_mul_pd(inv_eps_v, target_dk)),
                    phin_j),
                _mm256_mul_pd(_mm256_sub_pd(mixedk, mixed0), phi_j)));
        _mm256_store_pd(
            phi_lane,
            branchless_masked ? _mm256_and_pd(phi_term, lane_mask)
                              : phi_term);
        _mm256_store_pd(
            phin_lane,
            branchless_masked ? _mm256_and_pd(phin_term, lane_mask)
                              : phin_term);
        for (std::size_t lane = 0; lane < 4; ++lane) {
          accum[t + lane].phi += phi_lane[lane];
          accum[t + lane].phin += phin_lane[lane];
        }
      } else {
        const __m256d phi_term = _mm256_mul_pd(
            area, _mm256_mul_pd(poisson_scale_v,
                                _mm256_mul_pd(source_d0, phi_j)));
        _mm256_store_pd(
            phi_lane,
            branchless_masked ? _mm256_and_pd(phi_term, lane_mask)
                              : phi_term);
        for (std::size_t lane = 0; lane < 4; ++lane)
          accum[t + lane].phi += phi_lane[lane];
      }
      near_count += branchless_masked ? popcount4(valid_bits) : 4;
      ++vectorized_sources;
      vector_lanes += 4;
    }
    for (; t < target_indices.size(); ++t) {
      const std::size_t target_index = target_indices[t];
      if (original == target_index || target_index >= panels.size())
        continue;
      const Panel &target = panels[target_index];
      if (accumulate_jcp_leaf_source(
              accum[t], target.x, target.y, target.z, target.nx, target.ny,
              target.nz, sx, sy, sz, snx, sny, snz, area_scalar, phi_scalar,
              phin_scalar, eps_ratio, inv_eps_ratio, poisson_scale,
              inv_four_pi, kk, coupled))
        ++near_count;
    }
  }
}

TITAN_TARGET_AVX2
void direct_jcp_target_source_tile_sum_avx2_4x4(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::vector<JcpAccum> &accum, std::size_t &near_count,
    std::size_t &vectorized_sources, std::size_t &vector_lanes,
    bool vector_exp = false) noexcept {
  accum.assign(target_indices.size(), JcpAccum{});
  if (target_indices.empty())
    return;
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t source_begin = node.first_panel;
  const std::size_t source_end = node.first_panel + node.panel_count;
  const __m256d eps_v = _mm256_set1_pd(eps_ratio);
  const __m256d inv_eps_v = _mm256_set1_pd(inv_eps_ratio);
  const __m256d poisson_scale_v = _mm256_set1_pd(poisson_scale);
  const __m256d inv_four_pi_v = _mm256_set1_pd(inv_four_pi);
  const __m256d kk_v = _mm256_set1_pd(kk);
  const __m256d one_v = _mm256_set1_pd(1.0);
  const __m256d two_v = _mm256_set1_pd(2.0);
  const __m256d three_v = _mm256_set1_pd(3.0);
  const __m256d neg_one_v = _mm256_set1_pd(-1.0);
  const __m256d r2_floor_v = _mm256_set1_pd(1.0e-28);
  alignas(32) double tx_lane[4];
  alignas(32) double ty_lane[4];
  alignas(32) double tz_lane[4];
  alignas(32) double tnx_lane[4];
  alignas(32) double tny_lane[4];
  alignas(32) double tnz_lane[4];
  alignas(32) double r_lane[4];
  alignas(32) double exp_lane[4];
  alignas(32) double phi_lane[4];
  alignas(32) double phin_lane[4];

  auto process_source = [&](std::size_t pos, const __m256d &tx,
                            const __m256d &ty, const __m256d &tz,
                            const __m256d &tnx, const __m256d &tny,
                            const __m256d &tnz,
                            const std::size_t *targets,
                            __m256d &acc_phi_v,
                            __m256d &acc_phin_v) noexcept {
    const std::size_t original =
        pos < panel_cache.ordered_original_index.size()
            ? panel_cache.ordered_original_index[pos]
            : pos;
    unsigned lane_bits = 0x0fu;
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (original == targets[lane])
        lane_bits &= ~(1u << lane);
    }
    const double sx = panel_cache.ordered_x[pos];
    const double sy = panel_cache.ordered_y[pos];
    const double sz = panel_cache.ordered_z[pos];
    const __m256d ux = _mm256_sub_pd(tx, _mm256_set1_pd(sx));
    const __m256d uy = _mm256_sub_pd(ty, _mm256_set1_pd(sy));
    const __m256d uz = _mm256_sub_pd(tz, _mm256_set1_pd(sz));
    const __m256d r2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ux, ux), _mm256_mul_pd(uy, uy)),
        _mm256_mul_pd(uz, uz));
    const unsigned valid_bits =
        lane_bits & static_cast<unsigned>(_mm256_movemask_pd(
                        _mm256_cmp_pd(r2, r2_floor_v, _CMP_GT_OQ)));
    const __m256d lane_mask = avx2_lane_mask_pd(valid_bits);
    const __m256d r2_safe = _mm256_max_pd(r2, r2_floor_v);
    const __m256d r = _mm256_sqrt_pd(r2_safe);
    const __m256d inv_r = _mm256_div_pd(one_v, r);
    const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
    const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
    const __m256d snx_v = _mm256_set1_pd(panel_cache.ordered_nx[pos]);
    const __m256d sny_v = _mm256_set1_pd(panel_cache.ordered_ny[pos]);
    const __m256d snz_v = _mm256_set1_pd(panel_cache.ordered_nz[pos]);
    const __m256d area = _mm256_set1_pd(panel_cache.ordered_area[pos]);
    const __m256d phi_j = _mm256_set1_pd(unknown_cache.ordered_phi[pos]);
    const __m256d phin_j = _mm256_set1_pd(unknown_cache.ordered_phin[pos]);
    const __m256d source_projection = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(snx_v, ux),
                      _mm256_mul_pd(sny_v, uy)),
        _mm256_mul_pd(snz_v, uz));
    const __m256d target_projection = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(tnx, ux), _mm256_mul_pd(tny, uy)),
        _mm256_mul_pd(tnz, uz));
    const __m256d source_d0 =
        _mm256_mul_pd(inv_four_pi_v,
                      _mm256_mul_pd(source_projection, inv_r3));
    if (coupled) {
      __m256d expkr;
      if (vector_exp) {
        expkr =
            avx2_exp_accurate_pd(_mm256_mul_pd(_mm256_set1_pd(-kk), r));
      } else {
        _mm256_store_pd(r_lane, r);
        for (std::size_t lane = 0; lane < 4; ++lane)
          exp_lane[lane] = std::exp(-kk * r_lane[lane]);
        expkr = _mm256_load_pd(exp_lane);
      }
      const __m256d one_plus_kr =
          _mm256_add_pd(one_v, _mm256_mul_pd(kk_v, r));
      const __m256d source_dk =
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), source_d0);
      const __m256d g0 = _mm256_mul_pd(inv_four_pi_v, inv_r);
      const __m256d gk = _mm256_mul_pd(expkr, g0);
      const __m256d target_d0 = _mm256_mul_pd(
          neg_one_v,
          _mm256_mul_pd(inv_four_pi_v,
                        _mm256_mul_pd(target_projection, inv_r3)));
      const __m256d target_dk =
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), target_d0);
      const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
      const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
      const __m256d nxny = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(tnx, snx_v),
                        _mm256_mul_pd(tny, sny_v)),
          _mm256_mul_pd(tnz, snz_v));
      const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
      const __m256d tangent_metric = _mm256_sub_pd(nxny, nxu_nyu);
      const __m256d mixed0 = _mm256_mul_pd(
          inv_four_pi_v,
          _mm256_mul_pd(inv_r3,
                        _mm256_sub_pd(nxny, _mm256_mul_pd(three_v, nxu_nyu))));
      const __m256d fprime_k_over_r = _mm256_mul_pd(
          neg_one_v,
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr),
                        _mm256_mul_pd(inv_four_pi_v, inv_r3)));
      const __m256d fsecond_k = _mm256_mul_pd(
          expkr,
          _mm256_mul_pd(
              inv_four_pi_v,
              _mm256_add_pd(
                  _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(kk_v, kk_v),
                                              inv_r),
                                _mm256_mul_pd(_mm256_mul_pd(two_v, kk_v),
                                              inv_r2)),
                  _mm256_mul_pd(two_v, inv_r3))));
      const __m256d mixedk = _mm256_sub_pd(
          _mm256_setzero_pd(), _mm256_add_pd(
              _mm256_mul_pd(fsecond_k, nxu_nyu),
              _mm256_mul_pd(fprime_k_over_r, tangent_metric)));
      const __m256d phi_term = _mm256_and_pd(
          _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(_mm256_sub_pd(g0, gk), phin_j),
                  _mm256_mul_pd(
                      _mm256_sub_pd(_mm256_mul_pd(eps_v, source_dk),
                                    source_d0),
                      phi_j))),
          lane_mask);
      const __m256d phin_term = _mm256_and_pd(
          _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(
                      _mm256_sub_pd(target_d0,
                                    _mm256_mul_pd(inv_eps_v, target_dk)),
                      phin_j),
                  _mm256_mul_pd(_mm256_sub_pd(mixedk, mixed0), phi_j))),
          lane_mask);
      acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
      acc_phin_v = _mm256_add_pd(acc_phin_v, phin_term);
    } else {
      const __m256d phi_term = _mm256_and_pd(
          _mm256_mul_pd(area,
                        _mm256_mul_pd(poisson_scale_v,
                                      _mm256_mul_pd(source_d0, phi_j))),
          lane_mask);
      acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
    }
    near_count += popcount4(valid_bits);
  };

  std::size_t t = 0;
  for (; t + 4 <= target_indices.size(); t += 4) {
    bool targets_valid = true;
    std::size_t targets[4] = {};
    for (std::size_t lane = 0; lane < 4; ++lane) {
      targets[lane] = target_indices[t + lane];
      if (targets[lane] >= panels.size()) {
        targets_valid = false;
        break;
      }
      const Panel &target = panels[targets[lane]];
      tx_lane[lane] = target.x;
      ty_lane[lane] = target.y;
      tz_lane[lane] = target.z;
      tnx_lane[lane] = target.nx;
      tny_lane[lane] = target.ny;
      tnz_lane[lane] = target.nz;
    }
    if (!targets_valid)
      continue;
    const __m256d tx = _mm256_load_pd(tx_lane);
    const __m256d ty = _mm256_load_pd(ty_lane);
    const __m256d tz = _mm256_load_pd(tz_lane);
    const __m256d tnx = _mm256_load_pd(tnx_lane);
    const __m256d tny = _mm256_load_pd(tny_lane);
    const __m256d tnz = _mm256_load_pd(tnz_lane);
    __m256d acc_phi_v = _mm256_setzero_pd();
    __m256d acc_phin_v = _mm256_setzero_pd();
    std::size_t pos = source_begin;
    for (; pos + 4 <= source_end; pos += 4) {
      process_source(pos, tx, ty, tz, tnx, tny, tnz, targets, acc_phi_v,
                     acc_phin_v);
      process_source(pos + 1, tx, ty, tz, tnx, tny, tnz, targets,
                     acc_phi_v, acc_phin_v);
      process_source(pos + 2, tx, ty, tz, tnx, tny, tnz, targets,
                     acc_phi_v, acc_phin_v);
      process_source(pos + 3, tx, ty, tz, tnx, tny, tnz, targets,
                     acc_phi_v, acc_phin_v);
      vectorized_sources += 4;
      vector_lanes += 16;
    }
    for (; pos < source_end; ++pos) {
      process_source(pos, tx, ty, tz, tnx, tny, tnz, targets, acc_phi_v,
                     acc_phin_v);
      ++vectorized_sources;
      vector_lanes += 4;
    }
    _mm256_store_pd(phi_lane, acc_phi_v);
    if (coupled)
      _mm256_store_pd(phin_lane, acc_phin_v);
    for (std::size_t lane = 0; lane < 4; ++lane) {
      accum[t + lane].phi += phi_lane[lane];
      if (coupled)
        accum[t + lane].phin += phin_lane[lane];
    }
  }
  for (; t < target_indices.size(); ++t) {
    const std::size_t target_index = target_indices[t];
    if (target_index >= panels.size())
      continue;
    const Panel &target = panels[target_index];
    for (std::size_t pos = source_begin; pos < source_end; ++pos) {
      const std::size_t original =
          pos < panel_cache.ordered_original_index.size()
              ? panel_cache.ordered_original_index[pos]
              : pos;
      if (original == target_index)
        continue;
      if (accumulate_jcp_leaf_source(
              accum[t], target.x, target.y, target.z, target.nx, target.ny,
              target.nz, panel_cache.ordered_x[pos],
              panel_cache.ordered_y[pos], panel_cache.ordered_z[pos],
              panel_cache.ordered_nx[pos], panel_cache.ordered_ny[pos],
              panel_cache.ordered_nz[pos], panel_cache.ordered_area[pos],
              unknown_cache.ordered_phi[pos], unknown_cache.ordered_phin[pos],
              eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk,
              coupled))
        ++near_count;
    }
  }
}

TITAN_TARGET_AVX2
void direct_jcp_target_batch_soa_avx2(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    std::vector<JcpAccum> &accum, std::size_t &near_count,
    std::size_t &vectorized_sources, std::size_t &vector_lanes,
    std::size_t &soa_sources) noexcept {
  accum.assign(target_indices.size(), JcpAccum{});
  if (target_indices.size() != 4)
    return;
  for (std::size_t lane = 0; lane < 4; ++lane) {
    if (target_indices[lane] >= panels.size())
      return;
  }
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = node.first_panel + node.panel_count;
  const __m256d eps_v = _mm256_set1_pd(eps_ratio);
  const __m256d inv_eps_v = _mm256_set1_pd(inv_eps_ratio);
  const __m256d poisson_scale_v = _mm256_set1_pd(poisson_scale);
  const __m256d inv_four_pi_v = _mm256_set1_pd(inv_four_pi);
  const __m256d kk_v = _mm256_set1_pd(kk);
  const __m256d one_v = _mm256_set1_pd(1.0);
  const __m256d two_v = _mm256_set1_pd(2.0);
  const __m256d three_v = _mm256_set1_pd(3.0);
  const __m256d neg_one_v = _mm256_set1_pd(-1.0);
  const __m256d r2_floor_v = _mm256_set1_pd(1.0e-28);
  alignas(32) double tx_lane[4];
  alignas(32) double ty_lane[4];
  alignas(32) double tz_lane[4];
  alignas(32) double tnx_lane[4];
  alignas(32) double tny_lane[4];
  alignas(32) double tnz_lane[4];
  alignas(32) double r_lane[4];
  alignas(32) double exp_lane[4];
  alignas(32) double phi_lane[4];
  alignas(32) double phin_lane[4];
  for (std::size_t lane = 0; lane < 4; ++lane) {
    const Panel &target = panels[target_indices[lane]];
    tx_lane[lane] = target.x;
    ty_lane[lane] = target.y;
    tz_lane[lane] = target.z;
    tnx_lane[lane] = target.nx;
    tny_lane[lane] = target.ny;
    tnz_lane[lane] = target.nz;
  }
  const __m256d tx = _mm256_load_pd(tx_lane);
  const __m256d ty = _mm256_load_pd(ty_lane);
  const __m256d tz = _mm256_load_pd(tz_lane);
  const __m256d tnx = _mm256_load_pd(tnx_lane);
  const __m256d tny = _mm256_load_pd(tny_lane);
  const __m256d tnz = _mm256_load_pd(tnz_lane);
  __m256d acc_phi_v = _mm256_setzero_pd();
  __m256d acc_phin_v = _mm256_setzero_pd();
  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    const std::size_t original =
        pos < panel_cache.ordered_original_index.size()
            ? panel_cache.ordered_original_index[pos]
            : pos;
    unsigned lane_bits = 0x0fu;
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (original == target_indices[lane])
        lane_bits &= ~(1u << lane);
    }
    const double sx = panel_cache.ordered_x[pos];
    const double sy = panel_cache.ordered_y[pos];
    const double sz = panel_cache.ordered_z[pos];
    const __m256d ux = _mm256_sub_pd(tx, _mm256_set1_pd(sx));
    const __m256d uy = _mm256_sub_pd(ty, _mm256_set1_pd(sy));
    const __m256d uz = _mm256_sub_pd(tz, _mm256_set1_pd(sz));
    const __m256d r2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(ux, ux), _mm256_mul_pd(uy, uy)),
        _mm256_mul_pd(uz, uz));
    const unsigned valid_bits =
        lane_bits & static_cast<unsigned>(_mm256_movemask_pd(
                        _mm256_cmp_pd(r2, r2_floor_v, _CMP_GT_OQ)));
    const __m256d lane_mask = avx2_lane_mask_pd(valid_bits);
    const __m256d r2_safe = _mm256_max_pd(r2, r2_floor_v);
    const __m256d r = _mm256_sqrt_pd(r2_safe);
    const __m256d inv_r = _mm256_div_pd(one_v, r);
    const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
    const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
    const __m256d snx_v = _mm256_set1_pd(panel_cache.ordered_nx[pos]);
    const __m256d sny_v = _mm256_set1_pd(panel_cache.ordered_ny[pos]);
    const __m256d snz_v = _mm256_set1_pd(panel_cache.ordered_nz[pos]);
    const __m256d area = _mm256_set1_pd(panel_cache.ordered_area[pos]);
    const __m256d phi_j = _mm256_set1_pd(unknown_cache.ordered_phi[pos]);
    const __m256d phin_j = _mm256_set1_pd(unknown_cache.ordered_phin[pos]);
    const __m256d source_projection = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(snx_v, ux),
                      _mm256_mul_pd(sny_v, uy)),
        _mm256_mul_pd(snz_v, uz));
    const __m256d target_projection = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(tnx, ux), _mm256_mul_pd(tny, uy)),
        _mm256_mul_pd(tnz, uz));
    const __m256d source_d0 = _mm256_mul_pd(
        inv_four_pi_v, _mm256_mul_pd(source_projection, inv_r3));
    if (coupled) {
      _mm256_store_pd(r_lane, r);
      for (int lane = 0; lane < 4; ++lane)
        exp_lane[lane] = std::exp(-kk * r_lane[lane]);
      const __m256d expkr = _mm256_load_pd(exp_lane);
      const __m256d one_plus_kr =
          _mm256_add_pd(one_v, _mm256_mul_pd(kk_v, r));
      const __m256d source_dk =
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), source_d0);
      const __m256d g0 = _mm256_mul_pd(inv_four_pi_v, inv_r);
      const __m256d gk = _mm256_mul_pd(expkr, g0);
      const __m256d target_d0 = _mm256_mul_pd(
          neg_one_v,
          _mm256_mul_pd(inv_four_pi_v,
                        _mm256_mul_pd(target_projection, inv_r3)));
      const __m256d target_dk =
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr), target_d0);
      const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
      const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
      const __m256d nxny = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(tnx, snx_v),
                        _mm256_mul_pd(tny, sny_v)),
          _mm256_mul_pd(tnz, snz_v));
      const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
      const __m256d tangent_metric = _mm256_sub_pd(nxny, nxu_nyu);
      const __m256d mixed0 = _mm256_mul_pd(
          inv_four_pi_v,
          _mm256_mul_pd(inv_r3,
                        _mm256_sub_pd(nxny, _mm256_mul_pd(three_v, nxu_nyu))));
      const __m256d fprime_k_over_r = _mm256_mul_pd(
          neg_one_v,
          _mm256_mul_pd(_mm256_mul_pd(expkr, one_plus_kr),
                        _mm256_mul_pd(inv_four_pi_v, inv_r3)));
      const __m256d fsecond_k = _mm256_mul_pd(
          expkr,
          _mm256_mul_pd(
              inv_four_pi_v,
              _mm256_add_pd(
                  _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(kk_v, kk_v),
                                              inv_r),
                                _mm256_mul_pd(_mm256_mul_pd(two_v, kk_v),
                                              inv_r2)),
                  _mm256_mul_pd(two_v, inv_r3))));
      const __m256d mixedk = _mm256_sub_pd(
          _mm256_setzero_pd(), _mm256_add_pd(
              _mm256_mul_pd(fsecond_k, nxu_nyu),
              _mm256_mul_pd(fprime_k_over_r, tangent_metric)));
      const __m256d phi_term = _mm256_and_pd(
          _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(_mm256_sub_pd(g0, gk), phin_j),
                  _mm256_mul_pd(
                      _mm256_sub_pd(_mm256_mul_pd(eps_v, source_dk),
                                    source_d0),
                      phi_j))),
          lane_mask);
      const __m256d phin_term = _mm256_and_pd(
          _mm256_mul_pd(
              area,
              _mm256_add_pd(
                  _mm256_mul_pd(
                      _mm256_sub_pd(target_d0,
                                    _mm256_mul_pd(inv_eps_v, target_dk)),
                      phin_j),
                  _mm256_mul_pd(_mm256_sub_pd(mixedk, mixed0), phi_j))),
          lane_mask);
      acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
      acc_phin_v = _mm256_add_pd(acc_phin_v, phin_term);
    } else {
      const __m256d phi_term = _mm256_and_pd(
          _mm256_mul_pd(area,
                        _mm256_mul_pd(poisson_scale_v,
                                      _mm256_mul_pd(source_d0, phi_j))),
          lane_mask);
      acc_phi_v = _mm256_add_pd(acc_phi_v, phi_term);
    }
    near_count += popcount4(valid_bits);
    ++vectorized_sources;
    vector_lanes += 4;
    ++soa_sources;
  }
  _mm256_store_pd(phi_lane, acc_phi_v);
  if (coupled)
    _mm256_store_pd(phin_lane, acc_phin_v);
  for (std::size_t lane = 0; lane < 4; ++lane) {
    accum[lane].phi += phi_lane[lane];
    if (coupled)
      accum[lane].phin += phin_lane[lane];
  }
}
#endif

bool packed_jcp_avx2_available() noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  return titan_simd_dispatch::runtime_has_avx2();
#else
  return false;
#endif
}

void direct_jcp_packed_target_source_span_avx2(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, std::size_t source_first,
    std::size_t source_count, std::vector<JcpAccum> &accum,
    std::size_t &near_count, std::size_t &vectorized_sources,
    std::size_t &vector_lanes, bool vector_exp) noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (titan_simd_dispatch::runtime_has_avx2()) {
    TreeNode span;
    span.first_panel = source_first;
    span.panel_count = source_count;
    direct_jcp_target_source_tile_sum_avx2_4x4(
        panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, span, accum, near_count, vectorized_sources, vector_lanes,
        vector_exp);
    return;
  }
#else
  (void)panels;
  (void)panel_cache;
  (void)unknown_cache;
  (void)eps_ratio;
  (void)kappa;
  (void)coupled;
  (void)source_first;
  (void)source_count;
  (void)near_count;
  (void)vectorized_sources;
  (void)vector_lanes;
  (void)vector_exp;
#endif
  accum.assign(target_indices.size(), JcpAccum{});
}

void direct_jcp_target_batch_tile_sum(
    const std::vector<Panel> &panels,
    const std::vector<std::size_t> &target_indices,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled, const TreeNode &node,
    const std::string &kernel, std::vector<JcpAccum> &accum,
    std::size_t &near_count, std::size_t &vectorized_sources,
    std::size_t &vector_lanes, bool clean_source_span,
    std::size_t &soa_sources) noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if ((kernel == "soa-avx2" || kernel == "fused-soa-avx2") &&
      clean_source_span && target_indices.size() == 4) {
    direct_jcp_target_batch_soa_avx2(
        panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, node, accum, near_count, vectorized_sources, vector_lanes,
        soa_sources);
    return;
  }
  if (kernel == "target-avx2") {
    direct_jcp_target_batch_tile_sum_avx2(
        panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, node, accum, near_count, vectorized_sources, vector_lanes);
    return;
  }
  if (kernel == "target-avx2-v2") {
    direct_jcp_target_batch_tile_sum_avx2(
        panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, node, accum, near_count, vectorized_sources, vector_lanes,
        true);
    return;
  }
  if (kernel == "target-avx2-4x4") {
    direct_jcp_target_source_tile_sum_avx2_4x4(
        panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, node, accum, near_count, vectorized_sources, vector_lanes);
    return;
  }
#else
  (void)kernel;
  (void)vectorized_sources;
  (void)vector_lanes;
  (void)clean_source_span;
  (void)soa_sources;
#endif
  direct_jcp_target_batch_tile_sum_scalar(
      panels, target_indices, panel_cache, unknown_cache, eps_ratio, kappa,
      coupled, node, accum, near_count);
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2
JcpAccum direct_jcp_ordered_sum_avx2(
    const Panel &target, std::size_t target_index,
    const TreePanelCache &panel_cache, const TreeUnknownCache &unknown_cache,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, std::size_t &near_count) noexcept {
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double poisson_scale = eps_ratio - 1.0;
  const double inv_four_pi = 1.0 / (4.0 * kPi);
  const double kk = std::max(0.0, kappa);
  const std::size_t end = begin + count;
  const LeafTargetAvx2 t = leaf_target_avx2(target);
  const __m256i self = _mm256_set1_epi64x(static_cast<long long>(target_index));
  const double *x = panel_cache.x.data();
  const double *y = panel_cache.y.data();
  const double *z = panel_cache.z.data();
  const double *nx = panel_cache.nx.data();
  const double *ny = panel_cache.ny.data();
  const double *nz = panel_cache.nz.data();
  const double *area = panel_cache.area.data();
  const double *phi = unknown_cache.phi.data();
  const double *phin = unknown_cache.phin.data();
  __m256d acc_phi = _mm256_setzero_pd();
  __m256d acc_phin = _mm256_setzero_pd();
  std::size_t pos = begin;
  // Lanes are filled by scalar loads, not vgatherqpd: the index lists are
  // short and gathers are microcoded on CPUs with the GDS mitigation.
  for (; pos + 4 <= end; pos += 4) {
    const std::size_t j0 = order[pos];
    const std::size_t j1 = order[pos + 1];
    const std::size_t j2 = order[pos + 2];
    const std::size_t j3 = order[pos + 3];
    const __m256d excluded = _mm256_castsi256_pd(_mm256_cmpeq_epi64(
        _mm256_set_epi64x(static_cast<long long>(j3),
                          static_cast<long long>(j2),
                          static_cast<long long>(j1),
                          static_cast<long long>(j0)),
        self));
    near_count += avx2_lane_count(accumulate_jcp_leaf_sources_avx2(
        acc_phi, acc_phin, t, _mm256_set_pd(x[j3], x[j2], x[j1], x[j0]),
        _mm256_set_pd(y[j3], y[j2], y[j1], y[j0]),
        _mm256_set_pd(z[j3], z[j2], z[j1], z[j0]),
        _mm256_set_pd(nx[j3], nx[j2], nx[j1], nx[j0]),
        _mm256_set_pd(ny[j3], ny[j2], ny[j1], ny[j0]),
        _mm256_set_pd(nz[j3], nz[j2], nz[j1], nz[j0]),
        _mm256_set_pd(area[j3], area[j2], area[j1], area[j0]),
        _mm256_set_pd(phi[j3], phi[j2], phi[j1], phi[j0]),
        _mm256_set_pd(phin[j3], phin[j2], phin[j1], phin[j0]), excluded,
        eps_ratio, inv_eps_ratio, poisson_scale, inv_four_pi, kk, coupled));
  }
  JcpAccum acc = direct_jcp_ordered_sum_scalar(
      target, target_index, panel_cache, unknown_cache, eps_ratio, kappa,
      coupled, order, pos, end - pos, near_count);
  acc.phi += titan_avx2::hsum_pd(acc_phi);
  if (coupled)
    acc.phin += titan_avx2::hsum_pd(acc_phin);
  return acc;
}
#endif

std::string resolve_leaf_panel_kernel(const std::string &requested) {
  if (requested == "scalar")
    return "scalar";
  if (requested == "tile-scalar")
    return "tile-scalar";
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  // "auto" takes the AVX2 index-list kernel (rounding-level equal to the
  // scalar one); TITAN_FORCE_SCALAR=1 or a CPU without AVX2 keeps scalar.
  if ((requested == "auto" || requested == "avx2-gather" ||
       requested == "avx2") &&
      titan_simd_dispatch::runtime_has_avx2())
    return "avx2-gather";
  if ((requested == "tile" || requested == "tile-avx2") &&
      titan_simd_dispatch::runtime_has_avx2())
    return "tile-avx2";
#endif
  if (requested == "tile" || requested == "tile-avx2")
    return "tile-scalar";
  return "scalar";
}

std::string normalize_leaf_pair_kernel(const std::string &requested) {
  if (requested == "scalar" || requested == "target-avx2" ||
      requested == "target-avx2-v2" || requested == "target-avx2-4x4")
    return requested;
  return "auto";
}

std::string resolve_leaf_pair_batch_kernel(const std::string &requested,
                                           std::size_t target_batch_size) {
  if (requested == "scalar")
    return "scalar";
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (requested == "target-avx2-v2" && target_batch_size == 4 &&
      titan_simd_dispatch::runtime_has_avx2())
    return "target-avx2-v2";
  if (requested == "target-avx2-4x4" && target_batch_size == 4 &&
      titan_simd_dispatch::runtime_has_avx2())
    return "target-avx2-4x4";
  if ((requested == "auto" || requested == "target-avx2") &&
      target_batch_size == 4 && titan_simd_dispatch::runtime_has_avx2())
    return "target-avx2";
#else
  (void)target_batch_size;
#endif
  return "scalar";
}

std::string resolve_jcp_direct_batch_kernel(const TreeWorkspace &tree,
                                            const std::string &batch_kernel) {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if ((tree.jcp_direct_kernel == "soa-avx2" ||
       tree.jcp_direct_kernel == "fused-soa-avx2") &&
      batch_kernel == "target-avx2" && tree.target_batch_size == 4 &&
      titan_simd_dispatch::runtime_has_avx2())
    return tree.jcp_direct_kernel;
#else
  (void)tree;
#endif
  return batch_kernel;
}

JcpAccum direct_jcp_ordered_sum(const Panel &target, std::size_t target_index,
                                const TreePanelCache &panel_cache,
                                const TreeUnknownCache &unknown_cache,
                                double eps_ratio, double kappa, bool coupled,
                                const std::vector<std::size_t> &order,
                                std::size_t begin, std::size_t count,
                                const std::string &kernel,
                                std::size_t &near_count) noexcept {
#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (kernel == "avx2-gather" || kernel == "avx2") {
    return direct_jcp_ordered_sum_avx2(
        target, target_index, panel_cache, unknown_cache, eps_ratio, kappa,
        coupled, order, begin, count, near_count);
  }
#endif
  return direct_jcp_ordered_sum_scalar(
      target, target_index, panel_cache, unknown_cache, eps_ratio, kappa,
      coupled, order, begin, count, near_count);
}

bool leaf_pair_batching_enabled(const TreeWorkspace &tree) noexcept {
  return (tree.cluster_policy == "leaf-pair-strict" ||
          tree.cluster_policy == "leaf-pair-balanced") &&
         tree.target_batch_size > 1 && tree.leaf_pair_min_targets > 1;
}

std::size_t leaf_pair_skip_key(const TreeWorkspace &tree,
                               std::size_t target_index,
                               std::size_t node_index) noexcept {
  const std::size_t stride = std::max<std::size_t>(tree.nodes.size(), 1);
  return target_index * stride + node_index;
}

bool leaf_pair_batch_clean_for_target(const std::vector<Panel> &panels,
                                      const TreeWorkspace &tree,
                                      const TreeNode &node,
                                      std::size_t target_index) noexcept {
  if (target_index >= panels.size())
    return false;
  const Panel &target = panels[target_index];
  const std::size_t end = node.first_panel + node.panel_count;
  for (std::size_t pos = node.first_panel; pos < end; ++pos) {
    if (pos >= tree.order.size())
      return false;
    const std::size_t source_index = tree.order[pos];
    if (source_index >= panels.size())
      return false;
    const Panel &source = panels[source_index];
    if (curved_dual_cell(source) || is_singular_near_pair(target, target_index, source, source_index) ||
        classify_geometric_near_buffer_pair(target, source, tree.quad_buffer)
            .active) {
      return false;
    }
  }
  return true;
}

bool would_visit_leaf_direct_for_target(
    const std::vector<Panel> &panels, const TreeWorkspace &tree,
    const std::vector<std::size_t> &rank, const PackedJcpMoments &jcp_moments,
    std::size_t target_index, std::size_t source_node_index, int taylor_order,
    bool coupled, double eps_ratio, std::vector<std::size_t> &stack,
    std::size_t *node_visits = nullptr) noexcept {
  if (target_index >= panels.size() || source_node_index >= tree.nodes.size() ||
      target_index >= rank.size() || tree.nodes.empty())
    return false;
  if (tree.opening_theta <= 0.0)
    return true;
  const Panel &target = panels[target_index];
  const TreeNode &source_node = tree.nodes[source_node_index];
  const std::size_t target_rank = rank[target_index];
  stack.clear();
  stack.push_back(tree.root_index);
  while (!stack.empty()) {
    const std::size_t node_index = stack.back();
    stack.pop_back();
    if (node_index >= tree.nodes.size())
      continue;
    if (node_visits != nullptr)
      ++(*node_visits);
    const TreeNode &node = tree.nodes[node_index];
    const bool is_leaf = node.child_count == 0;
    const bool contains_target =
        target_rank >= node.first_panel &&
        target_rank < node.first_panel + node.panel_count;
    const bool leaf_mac_enabled = tree.leaf_mac == "on";
    const bool evaluate_node_mac =
        !contains_target && (!is_leaf || leaf_mac_enabled);
    if (evaluate_node_mac) {
      const double dx = target.x - node.cx;
      const double dy = target.y - node.cy;
      const double dz = target.z - node.cz;
      const double distance =
          std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
      const MacDecision mac = evaluate_mac(target, node, tree, distance);
      if (mac.accepted) {
        const bool source_inside_node =
            source_node.first_panel >= node.first_panel &&
            source_node.first_panel + source_node.panel_count <=
                node.first_panel + node.panel_count;
        if (tree.fragmentation_safety == "split" && node.fragmented) {
          if (is_leaf && node_index == source_node_index)
            return true;
          push_child_nodes(tree, node, stack);
          continue;
        }
        const ConvergenceEstimatorDecision estimator =
            evaluate_convergence_estimator(tree, node, jcp_moments, node_index,
                                           taylor_order, coupled, mac.rho,
                                           eps_ratio);
        if (estimator.split_reject) {
          if (is_leaf && node_index == source_node_index)
            return true;
          push_child_nodes(tree, node, stack);
          continue;
        }
        const bool k3_rho_cap_active =
            tree.convergence_estimator_model == "channel-tail-eps" &&
            tree.convergence_k3_accepted_rho_cap > 0.0;
        if (k3_rho_cap_active &&
            mac.rho > tree.convergence_k3_accepted_rho_cap) {
          if (is_leaf && node_index == source_node_index)
            return true;
          push_child_nodes(tree, node, stack);
          continue;
        }
        if (source_inside_node)
          return false;
        continue;
      }
    }
    if (is_leaf) {
      if (node_index == source_node_index)
        return true;
      continue;
    }
    push_child_nodes(tree, node, stack);
  }
  return false;
}

bool would_visit_leaf_direct_for_target_ancestor_path(
    const std::vector<Panel> &panels, const TreeWorkspace &tree,
    const std::vector<std::size_t> &rank, const PackedJcpMoments &jcp_moments,
    std::size_t target_index, std::size_t source_node_index, int taylor_order,
    bool coupled, double eps_ratio, std::size_t *node_visits = nullptr)
    noexcept {
  if (target_index >= panels.size() || source_node_index >= tree.nodes.size() ||
      target_index >= rank.size() || tree.nodes.empty())
    return false;
  if (tree.opening_theta <= 0.0)
    return true;
  if (source_node_index >= tree.leaf_ancestor_path_by_node.size() ||
      tree.leaf_ancestor_path_by_node[source_node_index].empty())
    return false;
  const Panel &target = panels[target_index];
  const std::size_t target_rank = rank[target_index];
  const std::vector<std::size_t> &path =
      tree.leaf_ancestor_path_by_node[source_node_index];
  for (std::size_t node_index : path) {
    if (node_index >= tree.nodes.size())
      return false;
    if (node_visits != nullptr)
      ++(*node_visits);
    const TreeNode &node = tree.nodes[node_index];
    const bool is_leaf = node.child_count == 0;
    const bool contains_target =
        target_rank >= node.first_panel &&
        target_rank < node.first_panel + node.panel_count;
    if (contains_target) {
      if (is_leaf)
        return node_index == source_node_index;
      continue;
    }
    const bool leaf_mac_enabled = tree.leaf_mac == "on";
    if (is_leaf && !leaf_mac_enabled)
      return node_index == source_node_index;
    const double dx = target.x - node.cx;
    const double dy = target.y - node.cy;
    const double dz = target.z - node.cz;
    const double distance =
        std::sqrt(std::max(1.0e-16, dx * dx + dy * dy + dz * dz));
    const MacDecision mac = evaluate_mac(target, node, tree, distance);
    if (!mac.accepted) {
      if (is_leaf)
        return node_index == source_node_index;
      continue;
    }
    if (tree.fragmentation_safety == "split" && node.fragmented) {
      if (is_leaf)
        return node_index == source_node_index;
      continue;
    }
    const ConvergenceEstimatorDecision estimator =
        evaluate_convergence_estimator(tree, node, jcp_moments, node_index,
                                       taylor_order, coupled, mac.rho,
                                       eps_ratio);
    if (estimator.split_reject) {
      if (is_leaf)
        return node_index == source_node_index;
      continue;
    }
    const bool k3_rho_cap_active =
        tree.convergence_estimator_model == "channel-tail-eps" &&
        tree.convergence_k3_accepted_rho_cap > 0.0;
    if (k3_rho_cap_active &&
        mac.rho > tree.convergence_k3_accepted_rho_cap) {
      if (is_leaf)
        return node_index == source_node_index;
      continue;
    }
    return false;
  }
  return false;
}


} // namespace TitanPBE::Tabi::detail

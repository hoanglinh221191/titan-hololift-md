#include "tabi_near.h"
#include "tabi_node_patch_curved.h"

#include "../titan_avx2_math.h"
#include "../titan_simd_dispatch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

namespace TitanPBE::Tabi {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct KernelValues {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
};

enum class RejectReason {
  None,
  CandidateWorseThanCentroid,
  ReferenceNonconverged,
  OrientationSignMismatch,
  StrictReferenceReject,
};

struct KernelBlockDecision {
  bool accepted{false};
  const char *accepted_policy{"centroid"};
  const char *rejected_policy{"centroid_guarded"};
  RejectReason reject_reason{RejectReason::None};
};

struct AcceptedKernelValues {
  KernelValues kernel{};
  std::array<KernelBlockDecision, 4> block{};
  std::array<double, 4> candidate_ref_rel{};
  std::array<double, 4> centroid_ref_rel{};
  std::array<bool, 4> candidate_worse_than_centroid{};
  std::array<bool, 4> k2_laplace_sign_ok{};
  std::array<double, 4> k2_yukawa_smooth_ref_rel{};
  std::array<double, 4> k2_orientation_pair_ref_rel{};
};

struct QuadPoint {
  double l0{0.0};
  double l1{0.0};
  double l2{0.0};
  double weight{0.0};
};

struct Quad1D {
  double x{0.0};
  double weight{0.0};
};

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct RawNearIntegrals {
  double g0_minus_gk{0.0};
  double d0_src{0.0};
  double dk_minus_d0_src{0.0};
  double d0_tgt{0.0};
  double dk_tgt{0.0};
  double mixedk_minus_mixed0{0.0};
};

struct K2CandidateInfo {
  bool laplace_sign_ok{true};
  double laplace_integral{0.0};
  double yukawa_smooth_integral{0.0};
};

double green(double r, double kappa) noexcept {
  if (r <= 1.0e-14)
    return 0.0;
  const double kk = std::max(0.0, kappa);
  return std::exp(-kk * r) / (4.0 * kPi * r);
}

double source_normal_green(double tx, double ty, double tz, double sx,
                           double sy, double sz, double snx, double sny,
                           double snz, double kappa) noexcept {
  const double ux = tx - sx;
  const double uy = ty - sy;
  const double uz = tz - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double ndotu = snx * ux + sny * uy + snz * uz;
  const double kk = std::max(0.0, kappa);
  return std::exp(-kk * r) * (1.0 + kk * r) * ndotu /
         (4.0 * kPi * r2 * r);
}

double target_normal_green(const Panel &target, double sx, double sy,
                           double sz, double kappa) noexcept {
  const double ux = target.x - sx;
  const double uy = target.y - sy;
  const double uz = target.z - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double ndotu = target.nx * ux + target.ny * uy + target.nz * uz;
  const double kk = std::max(0.0, kappa);
  return -std::exp(-kk * r) * (1.0 + kk * r) * ndotu /
         (4.0 * kPi * r2 * r);
}

double mixed_normal_green(const Panel &target, double sx, double sy, double sz,
                          double snx, double sny, double snz,
                          double kappa) noexcept {
  const double ux = target.x - sx;
  const double uy = target.y - sy;
  const double uz = target.z - sz;
  const double r2 = ux * ux + uy * uy + uz * uz;
  if (r2 <= 1.0e-28)
    return 0.0;
  const double r = std::sqrt(r2);
  const double inv_r = 1.0 / r;
  const double nxu =
      (target.nx * ux + target.ny * uy + target.nz * uz) * inv_r;
  const double nyu = (snx * ux + sny * uy + snz * uz) * inv_r;
  const double nxny = target.nx * snx + target.ny * sny + target.nz * snz;
  const double kk = std::max(0.0, kappa);
  const double expkr = std::exp(-kk * r);
  const double inv_r2 = inv_r * inv_r;
  const double fprime =
      -expkr * (1.0 + kk * r) * inv_r2 / (4.0 * kPi);
  const double fsecond =
      expkr *
      (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r2 * inv_r) /
      (4.0 * kPi);
  // grad_y G(x-y) = -grad_x G(x-y): the mixed normal derivative
  // is minus the radial Hessian contracted with the two outward normals.
  return -fsecond * nxu * nyu - (fprime * inv_r) * (nxny - nxu * nyu);
}

KernelValues point_kernel(const Panel &target, const Panel &source, double sx,
                          double sy, double sz, double eps_ratio,
                          double kappa) noexcept {
  const double dx = target.x - sx;
  const double dy = target.y - sy;
  const double dz = target.z - sz;
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (r <= 1.0e-14)
    return {};
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double g0 = green(r, 0.0);
  const double gk = green(r, kappa);
  // (sx, sy, sz) is the source point of the one-point rule, so the normal is
  // the source normal that goes with it (panel_source_nx).
  const double snx = panel_source_nx(source), sny = panel_source_ny(source),
               snz = panel_source_nz(source);
  const double d0_src = source_normal_green(target.x, target.y, target.z, sx,
                                            sy, sz, snx, sny, snz, 0.0);
  const double dk_src = source_normal_green(target.x, target.y, target.z, sx,
                                            sy, sz, snx, sny, snz, kappa);
  const double d0_tgt = target_normal_green(target, sx, sy, sz, 0.0);
  const double dk_tgt = target_normal_green(target, sx, sy, sz, kappa);
  const double d2_0 = mixed_normal_green(target, sx, sy, sz, snx, sny, snz, 0.0);
  const double d2_k =
      mixed_normal_green(target, sx, sy, sz, snx, sny, snz, kappa);
  return {g0 - gk, eps_ratio * dk_src - d0_src,
          d0_tgt - inv_eps_ratio * dk_tgt, d2_k - d2_0};
}

// point_kernel() with one sqrt and one exp instead of seven and eight.  Each
// expression is copied from green(), source_normal_green(),
// target_normal_green() and mixed_normal_green(); only exp(-0*r), which is
// exactly 1 for finite r, and the repeated sqrt/exp calls are replaced by the
// identical values already computed.  The result is therefore point_kernel()
// bit for bit.  Returns false for a non-finite r, where exp(-0*r) is not 1;
// the caller then uses point_kernel().
bool point_kernel_shared(const Panel &target, const Panel &source, double sx,
                         double sy, double sz, double eps_ratio, double kappa,
                         KernelValues &out) noexcept {
  const double dx = target.x - sx;
  const double dy = target.y - sy;
  const double dz = target.z - sz;
  const double r2 = dx * dx + dy * dy + dz * dz;
  const double r = std::sqrt(r2);
  if (!std::isfinite(r))
    return false;
  if (r <= 1.0e-14) {
    out = {};
    return true;
  }
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double kk0 = std::max(0.0, 0.0);
  const double expkr0 = 1.0;
  const double kk = std::max(0.0, kappa);
  const double expkr = std::exp(-kk * r);
  const double g0 = expkr0 / (4.0 * kPi * r);
  const double gk = expkr / (4.0 * kPi * r);

  double d0_src = 0.0;
  double dk_src = 0.0;
  double d0_tgt = 0.0;
  double dk_tgt = 0.0;
  double d2_0 = 0.0;
  double d2_k = 0.0;
  if (!(r2 <= 1.0e-28)) {
    const double snx = panel_source_nx(source), sny = panel_source_ny(source),
                 snz = panel_source_nz(source);
    const double ndotu_src = snx * dx + sny * dy + snz * dz;
    const double ndotu_tgt =
        target.nx * dx + target.ny * dy + target.nz * dz;
    d0_src = expkr0 * (1.0 + kk0 * r) * ndotu_src / (4.0 * kPi * r2 * r);
    dk_src = expkr * (1.0 + kk * r) * ndotu_src / (4.0 * kPi * r2 * r);
    d0_tgt = -expkr0 * (1.0 + kk0 * r) * ndotu_tgt / (4.0 * kPi * r2 * r);
    dk_tgt = -expkr * (1.0 + kk * r) * ndotu_tgt / (4.0 * kPi * r2 * r);

    const double inv_r = 1.0 / r;
    const double nxu = ndotu_tgt * inv_r;
    const double nyu = ndotu_src * inv_r;
    const double nxny = target.nx * snx + target.ny * sny + target.nz * snz;
    const double inv_r2 = inv_r * inv_r;
    const double fprime0 = -expkr0 * (1.0 + kk0 * r) * inv_r2 / (4.0 * kPi);
    const double fsecond0 =
        expkr0 *
        (kk0 * kk0 * inv_r + 2.0 * kk0 * inv_r2 + 2.0 * inv_r2 * inv_r) /
        (4.0 * kPi);
    d2_0 = -fsecond0 * nxu * nyu - (fprime0 * inv_r) * (nxny - nxu * nyu);
    const double fprime = -expkr * (1.0 + kk * r) * inv_r2 / (4.0 * kPi);
    const double fsecond =
        expkr *
        (kk * kk * inv_r + 2.0 * kk * inv_r2 + 2.0 * inv_r2 * inv_r) /
        (4.0 * kPi);
    d2_k = -fsecond * nxu * nyu - (fprime * inv_r) * (nxny - nxu * nyu);
  }
  out = {g0 - gk, eps_ratio * dk_src - d0_src,
         d0_tgt - inv_eps_ratio * dk_tgt, d2_k - d2_0};
  return true;
}

Vec3 make_vec(double x, double y, double z) noexcept { return {x, y, z}; }

Vec3 operator+(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(double s, const Vec3 &a) noexcept {
  return {s * a.x, s * a.y, s * a.z};
}

double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

double norm(const Vec3 &a) noexcept { return std::sqrt(dot(a, a)); }

double distance_sq(const Vec3 &a, const Vec3 &b) noexcept {
  const Vec3 d = a - b;
  return dot(d, d);
}

Vec3 panel_vertex(const Panel &panel, int index) noexcept {
  switch (index) {
  case 0:
    return make_vec(panel.v0x, panel.v0y, panel.v0z);
  case 1:
    return make_vec(panel.v1x, panel.v1y, panel.v1z);
  default:
    return make_vec(panel.v2x, panel.v2y, panel.v2z);
  }
}

Vec3 closest_point_on_triangle(const Vec3 &p, const Vec3 &a, const Vec3 &b,
                               const Vec3 &c) noexcept {
  const Vec3 ab = b - a;
  const Vec3 ac = c - a;
  const Vec3 ap = p - a;
  const double d1 = dot(ab, ap);
  const double d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0)
    return a;

  const Vec3 bp = p - b;
  const double d3 = dot(ab, bp);
  const double d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3)
    return b;

  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    const double v = d1 / (d1 - d3);
    return a + v * ab;
  }

  const Vec3 cp = p - c;
  const double d5 = dot(ab, cp);
  const double d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6)
    return c;

  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    const double w = d2 / (d2 - d6);
    return a + w * ac;
  }

  const double va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + w * (c - b);
  }

  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom;
  const double w = vc * denom;
  return a + v * ab + w * ac;
}

double point_triangle_distance_sq(const Vec3 &p, const Vec3 &a, const Vec3 &b,
                                  const Vec3 &c) noexcept {
  return distance_sq(p, closest_point_on_triangle(p, a, b, c));
}

double segment_segment_distance_sq(const Vec3 &p1, const Vec3 &q1,
                                   const Vec3 &p2, const Vec3 &q2) noexcept {
  constexpr double eps = 1.0e-14;
  const Vec3 d1 = q1 - p1;
  const Vec3 d2 = q2 - p2;
  const Vec3 r = p1 - p2;
  const double a = dot(d1, d1);
  const double e = dot(d2, d2);
  const double f = dot(d2, r);
  double s = 0.0;
  double t = 0.0;
  if (a <= eps && e <= eps)
    return distance_sq(p1, p2);
  if (a <= eps) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    const double c = dot(d1, r);
    if (e <= eps) {
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      const double b = dot(d1, d2);
      const double denom = a * e - b * b;
      if (std::abs(denom) > eps)
        s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
      const double tnom = b * s + f;
      if (tnom < 0.0) {
        t = 0.0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (tnom > e) {
        t = 1.0;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      } else {
        t = tnom / e;
      }
    }
  }
  const Vec3 c1 = p1 + s * d1;
  const Vec3 c2 = p2 + t * d2;
  return distance_sq(c1, c2);
}

double panel_max_edge_length(const Panel &panel) noexcept {
  if (!panel.has_triangle_geometry)
    return 0.0;
  const Vec3 a = panel_vertex(panel, 0);
  const Vec3 b = panel_vertex(panel, 1);
  const Vec3 c = panel_vertex(panel, 2);
  return std::sqrt(std::max({distance_sq(a, b), distance_sq(b, c),
                             distance_sq(c, a)}));
}

double panel_centroid_radius(const Panel &panel) noexcept {
  if (!panel.has_triangle_geometry)
    return 0.0;
  const Vec3 center = make_vec(panel.x, panel.y, panel.z);
  return std::sqrt(std::max({distance_sq(center, panel_vertex(panel, 0)),
                             distance_sq(center, panel_vertex(panel, 1)),
                             distance_sq(center, panel_vertex(panel, 2))}));
}

double triangle_triangle_distance_sq(const Panel &a_panel,
                                     const Panel &b_panel) noexcept {
  const std::array<Vec3, 3> a = {panel_vertex(a_panel, 0),
                                panel_vertex(a_panel, 1),
                                panel_vertex(a_panel, 2)};
  const std::array<Vec3, 3> b = {panel_vertex(b_panel, 0),
                                panel_vertex(b_panel, 1),
                                panel_vertex(b_panel, 2)};
  double best = std::numeric_limits<double>::infinity();
  for (const Vec3 &v : a)
    best = std::min(best, point_triangle_distance_sq(v, b[0], b[1], b[2]));
  for (const Vec3 &v : b)
    best = std::min(best, point_triangle_distance_sq(v, a[0], a[1], a[2]));
  for (int i = 0; i < 3; ++i) {
    const Vec3 a0 = a[static_cast<std::size_t>(i)];
    const Vec3 a1 = a[static_cast<std::size_t>((i + 1) % 3)];
    for (int j = 0; j < 3; ++j) {
      const Vec3 b0 = b[static_cast<std::size_t>(j)];
      const Vec3 b1 = b[static_cast<std::size_t>((j + 1) % 3)];
      best = std::min(best, segment_segment_distance_sq(a0, a1, b0, b1));
    }
  }
  return std::isfinite(best) ? best : 0.0;
}

double oriented_solid_angle(const Panel &target,
                            const Panel &source) noexcept {
  const Vec3 x = make_vec(target.x, target.y, target.z);
  const Vec3 a = panel_vertex(source, 0) - x;
  const Vec3 b = panel_vertex(source, 1) - x;
  const Vec3 c = panel_vertex(source, 2) - x;
  const double la = norm(a);
  const double lb = norm(b);
  const double lc = norm(c);
  if (la <= 1.0e-14 || lb <= 1.0e-14 || lc <= 1.0e-14)
    return 0.0;
  const double numerator = dot(a, cross(b, c));
  const double denominator =
      la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
  double omega = 2.0 * std::atan2(numerator, denominator);
  const Vec3 v0 = panel_vertex(source, 0);
  const Vec3 v1 = panel_vertex(source, 1);
  const Vec3 v2 = panel_vertex(source, 2);
  const Vec3 tri_normal = cross(v1 - v0, v2 - v0);
  const Vec3 source_normal = make_vec(source.nx, source.ny, source.nz);
  if (dot(tri_normal, source_normal) < 0.0)
    omega = -omega;
  return omega;
}

bool finite_kernel(const KernelValues &kernel) noexcept {
  return std::isfinite(kernel.k1) && std::isfinite(kernel.k2) &&
         std::isfinite(kernel.k3) && std::isfinite(kernel.k4);
}

bool relative_guard_ok(double candidate, double reference,
                       double rel_limit) noexcept {
  if (!std::isfinite(candidate) || !std::isfinite(reference))
    return false;
  const double scale = std::max({1.0e-14, std::abs(reference),
                                 0.25 * std::abs(candidate)});
  return std::abs(candidate - reference) <= rel_limit * scale;
}

bool same_sign_for_significant_values(double a, double b) noexcept {
  const double scale = std::max({1.0e-14, std::abs(a), std::abs(b)});
  if (std::abs(a) <= 1.0e-12 * scale || std::abs(b) <= 1.0e-12 * scale)
    return true;
  return a * b >= 0.0;
}

double relative_error_value(double value, double reference) noexcept {
  if (!std::isfinite(value) || !std::isfinite(reference))
    return std::numeric_limits<double>::infinity();
  const double scale =
      std::max({1.0e-14, std::abs(reference), 0.25 * std::abs(value)});
  return std::abs(value - reference) / scale;
}

std::size_t near_block_bucket_index(std::size_t block, std::size_t pair_type,
                                    std::size_t orientation) noexcept {
  return (block * kNearPairTypeCount + pair_type) * kNearOrientationCount +
         orientation;
}

std::size_t classify_pair_type(bool self_pair, int shared_vertices) noexcept {
  if (self_pair)
    return 0;
  if (shared_vertices >= 2)
    return 1;
  if (shared_vertices == 1)
    return 2;
  return 3;
}

std::size_t classify_orientation(const Panel &target,
                                 const Panel &source) noexcept {
  const double dotn = target.nx * source.nx + target.ny * source.ny +
                      target.nz * source.nz;
  if (dotn >= 0.25)
    return 0;
  if (dotn <= -0.25)
    return 1;
  return 2;
}

bool finite_kernel_values(const KernelValues &kernel) noexcept {
  return std::isfinite(kernel.k1) && std::isfinite(kernel.k2) &&
         std::isfinite(kernel.k3) && std::isfinite(kernel.k4);
}

KernelValues centroid_integral(const Panel &target, const Panel &source,
                               double eps_ratio, double kappa) noexcept {
  const double sx = source.has_source_quadrature_center ? source.source_x
                                                        : source.x;
  const double sy = source.has_source_quadrature_center ? source.source_y
                                                        : source.y;
  const double sz = source.has_source_quadrature_center ? source.source_z
                                                        : source.z;
  KernelValues kernel = point_kernel(target, source, sx, sy, sz, eps_ratio,
                                     kappa);
  kernel.k1 *= source.area;
  kernel.k2 *= source.area;
  kernel.k3 *= source.area;
  kernel.k4 *= source.area;
  return kernel;
}

// centroid_integral() through point_kernel_shared(); false when that declines.
bool centroid_integral_shared(const Panel &target, const Panel &source,
                              double eps_ratio, double kappa,
                              KernelValues &kernel) noexcept {
  const double sx = source.has_source_quadrature_center ? source.source_x
                                                        : source.x;
  const double sy = source.has_source_quadrature_center ? source.source_y
                                                        : source.y;
  const double sz = source.has_source_quadrature_center ? source.source_z
                                                        : source.z;
  if (!point_kernel_shared(target, source, sx, sy, sz, eps_ratio, kappa,
                           kernel))
    return false;
  kernel.k1 *= source.area;
  kernel.k2 *= source.area;
  kernel.k3 *= source.area;
  kernel.k4 *= source.area;
  return true;
}

[[maybe_unused]] bool share_vertex(const Panel &a, const Panel &b) noexcept {
  if (!a.has_triangle_geometry || !b.has_triangle_geometry)
    return false;
  const std::array<std::size_t, 3> av = {a.v0_index, a.v1_index, a.v2_index};
  const std::array<std::size_t, 3> bv = {b.v0_index, b.v1_index, b.v2_index};
  for (std::size_t ai : av) {
    for (std::size_t bi : bv) {
      if (ai != static_cast<std::size_t>(-1) && ai == bi)
        return true;
    }
  }
  return false;
}

int shared_vertex_count(const Panel &a, const Panel &b) noexcept {
  if (!a.has_triangle_geometry || !b.has_triangle_geometry)
    return 0;
  const std::array<std::size_t, 3> av = {a.v0_index, a.v1_index, a.v2_index};
  const std::array<std::size_t, 3> bv = {b.v0_index, b.v1_index, b.v2_index};
  int count = 0;
  for (std::size_t ai : av) {
    if (ai == static_cast<std::size_t>(-1))
      continue;
    for (std::size_t bi : bv) {
      if (ai == bi) {
        ++count;
        break;
      }
    }
  }
  return count;
}

double self_green_disk_integral(double area, double kappa) noexcept {
  const double radius = std::sqrt(std::max(0.0, area) / kPi);
  if (radius <= 0.0)
    return 0.0;
  const double kk = std::max(0.0, kappa);
  if (kk <= 1.0e-14)
    return 0.5 * radius;
  return (1.0 - std::exp(-kk * radius)) / (2.0 * kk);
}

bool is_node_patch_panel(const Panel &panel) noexcept;

// Discrete mean curvature at a node patch, from its one-ring.  On a sphere of
// radius R, -2*n.(x_j-x_i)/|x_j-x_i|^2 equals 1/R exactly for every neighbour
// and any mesh, so the average is unbiased there.  Returns 0 when no one-ring
// is available, which restores the previous flat behaviour exactly.
double node_patch_mean_curvature(const Panel &panel) noexcept {
  if (!is_node_patch_panel(panel))
    return 0.0;
  const NodePatchGeometry &geometry = *panel.node_patch_geometry;
  const std::size_t node = panel.node_patch_index;
  if (node + 1 >= geometry.adjacent_offsets.size() ||
      node >= geometry.vertex_xyz.size())
    return 0.0;
  const auto &xyz = geometry.vertex_xyz[node];
  const Vec3 origin{xyz[0], xyz[1], xyz[2]};
  const Vec3 normal{panel.nx, panel.ny, panel.nz};
  double sum = 0.0;
  std::size_t count = 0;
  for (std::size_t pos = geometry.adjacent_offsets[node];
       pos < geometry.adjacent_offsets[node + 1]; ++pos) {
    const std::size_t other = geometry.adjacent_nodes[pos];
    if (other >= geometry.vertex_xyz.size())
      continue;
    const auto &other_xyz = geometry.vertex_xyz[other];
    const Vec3 edge =
        Vec3{other_xyz[0], other_xyz[1], other_xyz[2]} - origin;
    const double length2 = dot(edge, edge);
    if (!(length2 > 1.0e-24))
      continue;
    sum += -2.0 * dot(normal, edge) / length2;
    ++count;
  }
  if (count == 0)
    return 0.0;
  const double curvature = sum / static_cast<double>(count);
  return std::isfinite(curvature) ? curvature : 0.0;
}

[[maybe_unused]] KernelValues self_singular_integral(const Panel &source,
                                                     double eps_ratio,
                                                     double kappa) noexcept {
  // The jump terms remain on the analytic JCP LHS.  For the collocation
  // source triangle itself, integrate only the single-layer singular part
  // with an equal-area disk surrogate.
  const double g0 = self_green_disk_integral(source.area, 0.0);
  const double gk = self_green_disk_integral(source.area, kappa);
  // The derivative kernels' principal value is zero only for a FLAT patch,
  // where n_y.(x-y) vanishes identically on the patch itself.  Expanding the
  // surface about the collocation point gives, for principal curvatures k1,k2,
  //   integral dG0/dn_y dS = -(k1+k2)/2 * sqrt(A/pi)/4 + O(A),
  // and dG/dn_x agrees with dG/dn_y at this order.  On a sphere the mean
  // curvature is 1/R and this reproduces the closed form.  The screened minus
  // unscreened difference is O(kappa^2 * A^(3/2)) and is dropped with it.
  // k4 keeps its flat finite part; no curvature term is claimed for it.
  // Leading order only.  A scale sweep on the sphere puts the best-fit
  // coefficient near 2.2 and drifting with resolution, so the mechanism is
  // right but the magnitude for a barycentric dual cell is not settled; a
  // fitted constant would not be physics.  See the 2026-09-15 report.
  const double layer = -node_patch_mean_curvature(source) *
                       std::sqrt(std::max(0.0, source.area) / kPi) / 4.0;
  return {g0 - gk, (eps_ratio - 1.0) * layer,
          (1.0 - 1.0 / std::max(1.0e-12, eps_ratio)) * layer, 0.0};
}

AcceptedKernelValues self_singular_decision(const Panel &source,
                                            double eps_ratio, double kappa) noexcept {
  AcceptedKernelValues out;
  out.kernel = self_singular_integral(source, eps_ratio, kappa);
  out.block = {KernelBlockDecision{true, "self_asi", "centroid_guarded"},
               KernelBlockDecision{true, "self_curved_pv", "centroid_guarded"},
               KernelBlockDecision{true, "self_curved_pv", "centroid_guarded"},
               KernelBlockDecision{true, "self_flat_pv", "centroid_guarded"}};
  return out;
}

void accumulate_raw_near(RawNearIntegrals &raw, const Panel &target,
                         const Panel &source, const Vec3 &point,
                         double weight, double kappa) noexcept {
  const double dx = target.x - point.x;
  const double dy = target.y - point.y;
  const double dz = target.z - point.z;
  const double r2 = dx * dx + dy * dy + dz * dz;
  if (r2 <= 1.0e-28)
    return;
  const double r = std::sqrt(r2);
  const double g0 = green(r, 0.0);
  const double gk = green(r, kappa);
  const double d0_src =
      source_normal_green(target.x, target.y, target.z, point.x, point.y,
                          point.z, source.nx, source.ny, source.nz, 0.0);
  const double dk_src =
      source_normal_green(target.x, target.y, target.z, point.x, point.y,
                          point.z, source.nx, source.ny, source.nz, kappa);
  const double d0_tgt = target_normal_green(target, point.x, point.y, point.z,
                                            0.0);
  const double dk_tgt =
      target_normal_green(target, point.x, point.y, point.z, kappa);
  const double mixed0 =
      mixed_normal_green(target, point.x, point.y, point.z, source.nx,
                         source.ny, source.nz, 0.0);
  const double mixedk =
      mixed_normal_green(target, point.x, point.y, point.z, source.nx,
                         source.ny, source.nz, kappa);
  raw.g0_minus_gk += weight * (g0 - gk);
  raw.d0_src += weight * d0_src;
  raw.dk_minus_d0_src += weight * (dk_src - d0_src);
  raw.d0_tgt += weight * d0_tgt;
  raw.dk_tgt += weight * dk_tgt;
  raw.mixedk_minus_mixed0 += weight * (mixedk - mixed0);
}

KernelValues duffy_cpv_integral(const Panel &target, const Panel &source,
                                double eps_ratio, double kappa,
                                bool self_pair,
                                K2CandidateInfo *k2_info = nullptr) noexcept {
  static constexpr std::array<Quad1D, 8> q1 = {
      Quad1D{0.0198550717512319, 0.0506142681451881},
      Quad1D{0.1016667612931876, 0.1111905172266872},
      Quad1D{0.2372337950418355, 0.1568533229389436},
      Quad1D{0.4082826787521751, 0.1813418916891810},
      Quad1D{0.5917173212478249, 0.1813418916891810},
      Quad1D{0.7627662049581645, 0.1568533229389436},
      Quad1D{0.8983332387068124, 0.1111905172266872},
      Quad1D{0.9801449282487681, 0.0506142681451881}};

  const Vec3 v0 = panel_vertex(source, 0);
  const Vec3 v1 = panel_vertex(source, 1);
  const Vec3 v2 = panel_vertex(source, 2);
  const Vec3 target_point = make_vec(target.x, target.y, target.z);
  const Vec3 singular =
      self_pair ? target_point : closest_point_on_triangle(target_point, v0, v1, v2);
  const std::array<Vec3, 3> verts = {v0, v1, v2};
  RawNearIntegrals raw;
  for (int edge = 0; edge < 3; ++edge) {
    const Vec3 a = verts[edge];
    const Vec3 b = verts[(edge + 1) % 3];
    const Vec3 pa = a - singular;
    const Vec3 pb = b - singular;
    const double jac_base = norm(cross(pa, pb));
    if (jac_base <= 1.0e-18)
      continue;
    for (const Quad1D &u_quad : q1) {
      const double u = u_quad.x;
      for (const Quad1D &v_quad : q1) {
        const double v = v_quad.x;
        const Vec3 radial = (1.0 - v) * pa + v * pb;
        const Vec3 point = singular + u * radial;
        const double weight = jac_base * u * u_quad.weight * v_quad.weight;
        accumulate_raw_near(raw, target, source, point, weight, kappa);
      }
    }
  }

  double d0_src = raw.d0_src;
  bool laplace_sign_ok = true;
  if (self_pair) {
    d0_src = 0.0;
  } else {
    const double centroid_d0 =
        source_normal_green(target.x, target.y, target.z, source.x, source.y,
                            source.z, source.nx, source.ny, source.nz, 0.0) *
        source.area;
    const double analytic_d0 = -oriented_solid_angle(target, source) /
                               (4.0 * kPi);
    laplace_sign_ok =
        std::isfinite(analytic_d0) &&
        (same_sign_for_significant_values(analytic_d0, raw.d0_src) ||
         same_sign_for_significant_values(analytic_d0, centroid_d0));
    d0_src = std::isfinite(analytic_d0) ? analytic_d0 : centroid_d0;
    if (!laplace_sign_ok && relative_guard_ok(analytic_d0, centroid_d0, 0.20))
      d0_src = analytic_d0;
  }
  if (k2_info != nullptr) {
    k2_info->laplace_sign_ok = laplace_sign_ok;
    k2_info->laplace_integral = d0_src;
    k2_info->yukawa_smooth_integral = raw.dk_minus_d0_src;
  }

  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const double dk_src = d0_src + raw.dk_minus_d0_src;
  return {raw.g0_minus_gk, eps_ratio * dk_src - d0_src,
          raw.d0_tgt - inv_eps_ratio * raw.dk_tgt,
          raw.mixedk_minus_mixed0};
}

Panel make_subtriangle_panel(const Panel &source, const Vec3 &a, const Vec3 &b,
                             const Vec3 &c) noexcept;
KernelValues triangle_gauss_subdivided_integral(const Panel &target,
                                                const Panel &source,
                                                double eps_ratio,
                                                double kappa) noexcept;

bool is_node_patch_panel(const Panel &panel) noexcept {
  return panel.node_patch_geometry != nullptr &&
         panel.node_patch_index != static_cast<std::size_t>(-1);
}

bool node_patch_adjacent(const Panel &target, const Panel &source) noexcept {
  if (!is_node_patch_panel(target) || !is_node_patch_panel(source) ||
      target.node_patch_geometry.get() != source.node_patch_geometry.get())
    return false;
  const NodePatchGeometry &geometry = *source.node_patch_geometry;
  const std::size_t source_node = source.node_patch_index;
  if (source_node + 1 >= geometry.adjacent_offsets.size())
    return false;
  const auto begin = geometry.adjacent_nodes.begin() +
                     geometry.adjacent_offsets[source_node];
  const auto end = geometry.adjacent_nodes.begin() +
                   geometry.adjacent_offsets[source_node + 1];
  return std::binary_search(begin, end, target.node_patch_index);
}

Vec3 node_patch_triangle_vertex(const NodePatchGeometry &geometry,
                                const NodePatchTriangle &triangle,
                                int local) noexcept {
  const auto &xyz = geometry.vertex_xyz[
      triangle.vertices[static_cast<std::size_t>(local)]];
  return Vec3{xyz[0], xyz[1], xyz[2]};
}

KernelValues node_patch_near_integral(const Panel &target,
                                      const Panel &source, double eps_ratio,
                                      double kappa, bool self_pair) {
  const NodePatchGeometry &geometry = *source.node_patch_geometry;
  const std::size_t node = source.node_patch_index;
  if (node + 1 >= geometry.incident_offsets.size())
    throw std::runtime_error(
        "PBE BIE: invalid node-patch incident-triangle topology.");
  KernelValues out;
  const std::size_t begin = geometry.incident_offsets[node];
  const std::size_t end = geometry.incident_offsets[node + 1];
  for (std::size_t pos = begin; pos < end; ++pos) {
    const NodePatchTriangle &triangle =
        geometry.triangles[geometry.incident_triangles[pos]];
    int local = -1;
    for (int k = 0; k < 3; ++k) {
      if (triangle.vertices[static_cast<std::size_t>(k)] == node) {
        local = k;
        break;
      }
    }
    if (local < 0)
      throw std::runtime_error(
          "PBE BIE: node-patch incidence does not reference its node.");
    const Vec3 vertex =
        node_patch_triangle_vertex(geometry, triangle, local);
    const Vec3 next =
        node_patch_triangle_vertex(geometry, triangle, (local + 1) % 3);
    const Vec3 previous =
        node_patch_triangle_vertex(geometry, triangle, (local + 2) % 3);
    const Vec3 midpoint_next = 0.5 * (vertex + next);
    const Vec3 midpoint_previous = 0.5 * (previous + vertex);
    const Vec3 centroid = (1.0 / 3.0) * (vertex + next + previous);
    Panel triangle_source = source;
    triangle_source.nx = triangle.nx;
    triangle_source.ny = triangle.ny;
    triangle_source.nz = triangle.nz;
    triangle_source.node_patch_geometry.reset();
    triangle_source.node_patch_index = static_cast<std::size_t>(-1);
    const std::array<Panel, 2> pieces = {
        make_subtriangle_panel(triangle_source, vertex, midpoint_next,
                               centroid),
        make_subtriangle_panel(triangle_source, vertex, centroid,
                               midpoint_previous)};
    for (const Panel &piece : pieces) {
      const KernelValues value =
          self_pair ? duffy_cpv_integral(target, piece, eps_ratio, kappa, true)
                    : triangle_gauss_subdivided_integral(target, piece,
                                                         eps_ratio, kappa);
      out.k1 += value.k1;
      out.k2 += value.k2;
      out.k3 += value.k3;
      out.k4 += value.k4;
    }
  }
  return out;
}

[[maybe_unused]] KernelValues
triangle_gauss_integral(const Panel &target, const Panel &source,
                        double eps_ratio, double kappa) noexcept {
  static constexpr std::array<QuadPoint, 7> points = {
      QuadPoint{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.2250000000000000},
      QuadPoint{0.4701420641051151, 0.4701420641051151, 0.0597158717897698,
                0.1323941527885062},
      QuadPoint{0.4701420641051151, 0.0597158717897698, 0.4701420641051151,
                0.1323941527885062},
      QuadPoint{0.0597158717897698, 0.4701420641051151, 0.4701420641051151,
                0.1323941527885062},
      QuadPoint{0.1012865073234563, 0.1012865073234563, 0.7974269853530873,
                0.1259391805448272},
      QuadPoint{0.1012865073234563, 0.7974269853530873, 0.1012865073234563,
                0.1259391805448272},
      QuadPoint{0.7974269853530873, 0.1012865073234563, 0.1012865073234563,
                0.1259391805448272}};
  KernelValues out;
  if (!source.has_triangle_geometry || source.area <= 0.0)
    return centroid_integral(target, source, eps_ratio, kappa);
  double g0_integral = 0.0;
  double gk_integral = 0.0;
  double d0_src_integral = 0.0;
  double dk_src_integral = 0.0;
  double d0_tgt_integral = 0.0;
  double dk_tgt_integral = 0.0;
  double mixed0_integral = 0.0;
  double mixedk_integral = 0.0;
  for (const QuadPoint &q : points) {
    const double sx = q.l0 * source.v0x + q.l1 * source.v1x +
                      q.l2 * source.v2x;
    const double sy = q.l0 * source.v0y + q.l1 * source.v1y +
                      q.l2 * source.v2y;
    const double sz = q.l0 * source.v0z + q.l1 * source.v1z +
                      q.l2 * source.v2z;
    const double w = source.area * q.weight;
    const double dx = target.x - sx;
    const double dy = target.y - sy;
    const double dz = target.z - sz;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (r <= 1.0e-14)
      continue;
    g0_integral += w * green(r, 0.0);
    gk_integral += w * green(r, kappa);
    d0_src_integral +=
        w * source_normal_green(target.x, target.y, target.z, sx, sy, sz,
                                source.nx, source.ny, source.nz, 0.0);
    dk_src_integral +=
        w * source_normal_green(target.x, target.y, target.z, sx, sy, sz,
                                source.nx, source.ny, source.nz, kappa);
    d0_tgt_integral += w * target_normal_green(target, sx, sy, sz, 0.0);
    dk_tgt_integral += w * target_normal_green(target, sx, sy, sz, kappa);
    mixed0_integral +=
        w * mixed_normal_green(target, sx, sy, sz, source.nx, source.ny,
                               source.nz, 0.0);
    mixedk_integral +=
        w * mixed_normal_green(target, sx, sy, sz, source.nx, source.ny,
                               source.nz, kappa);
  }
  auto solid_angle = [](const Panel &target, const Panel &source) noexcept {
    const double ax = source.v0x - target.x;
    const double ay = source.v0y - target.y;
    const double az = source.v0z - target.z;
    const double bx = source.v1x - target.x;
    const double by = source.v1y - target.y;
    const double bz = source.v1z - target.z;
    const double cx = source.v2x - target.x;
    const double cy = source.v2y - target.y;
    const double cz = source.v2z - target.z;
    const double la = std::sqrt(ax * ax + ay * ay + az * az);
    const double lb = std::sqrt(bx * bx + by * by + bz * bz);
    const double lc = std::sqrt(cx * cx + cy * cy + cz * cz);
    if (la <= 1.0e-14 || lb <= 1.0e-14 || lc <= 1.0e-14)
      return 0.0;
    const double cross_x = by * cz - bz * cy;
    const double cross_y = bz * cx - bx * cz;
    const double cross_z = bx * cy - by * cx;
    const double numerator = ax * cross_x + ay * cross_y + az * cross_z;
    const double ab = ax * bx + ay * by + az * bz;
    const double bc = bx * cx + by * cy + bz * cz;
    const double ca = cx * ax + cy * ay + cz * az;
    const double denominator = la * lb * lc + ab * lc + bc * la + ca * lb;
    return 2.0 * std::atan2(numerator, denominator);
  };
  const double omega = solid_angle(target, source);
  if (std::isfinite(omega) && std::abs(omega) > 1.0e-14) {
    double analytic_d0_src = -omega / (4.0 * kPi);
    if (analytic_d0_src * d0_src_integral < 0.0)
      analytic_d0_src = -analytic_d0_src;
    d0_src_integral = analytic_d0_src;
  }
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  out.k1 = g0_integral - gk_integral;
  out.k2 = eps_ratio * dk_src_integral - d0_src_integral;
  out.k3 = d0_tgt_integral - inv_eps_ratio * dk_tgt_integral;
  out.k4 = mixedk_integral - mixed0_integral;
  return out;
}

Panel make_subtriangle_panel(const Panel &source, const Vec3 &a, const Vec3 &b,
                             const Vec3 &c) noexcept {
  Panel sub = source;
  sub.v0x = a.x;
  sub.v0y = a.y;
  sub.v0z = a.z;
  sub.v1x = b.x;
  sub.v1y = b.y;
  sub.v1z = b.z;
  sub.v2x = c.x;
  sub.v2y = c.y;
  sub.v2z = c.z;
  const Vec3 normal_area = cross(b - a, c - a);
  sub.area = 0.5 * norm(normal_area);
  sub.x = (a.x + b.x + c.x) / 3.0;
  sub.y = (a.y + b.y + c.y) / 3.0;
  sub.z = (a.z + b.z + c.z) / 3.0;
  sub.has_triangle_geometry = source.has_triangle_geometry && sub.area > 0.0;
  // A piece is a flat triangle in its own right: its centroid and normal are
  // its source point, not the centroid of the node patch it was cut from.
  sub.has_source_quadrature_center = false;
  sub.v0_index = static_cast<std::size_t>(-1);
  sub.v1_index = static_cast<std::size_t>(-1);
  sub.v2_index = static_cast<std::size_t>(-1);
  return sub;
}

KernelValues triangle_gauss_subdivided_integral(const Panel &target,
                                                const Panel &source,
                                                double eps_ratio,
                                                double kappa) noexcept {
  if (!source.has_triangle_geometry || source.area <= 0.0)
    return triangle_gauss_integral(target, source, eps_ratio, kappa);
  const Vec3 a = panel_vertex(source, 0);
  const Vec3 b = panel_vertex(source, 1);
  const Vec3 c = panel_vertex(source, 2);
  const Vec3 ab = 0.5 * (a + b);
  const Vec3 bc = 0.5 * (b + c);
  const Vec3 ca = 0.5 * (c + a);
  const std::array<Panel, 4> first_level = {
      make_subtriangle_panel(source, a, ab, ca),
      make_subtriangle_panel(source, ab, b, bc),
      make_subtriangle_panel(source, ca, bc, c),
      make_subtriangle_panel(source, ab, bc, ca)};
  KernelValues out;
  for (const Panel &sub : first_level) {
    if (!sub.has_triangle_geometry)
      continue;
    const Vec3 sa = panel_vertex(sub, 0);
    const Vec3 sb = panel_vertex(sub, 1);
    const Vec3 sc = panel_vertex(sub, 2);
    const Vec3 sab = 0.5 * (sa + sb);
    const Vec3 sbc = 0.5 * (sb + sc);
    const Vec3 sca = 0.5 * (sc + sa);
    const std::array<Panel, 4> second_level = {
        make_subtriangle_panel(sub, sa, sab, sca),
        make_subtriangle_panel(sub, sab, sb, sbc),
        make_subtriangle_panel(sub, sca, sbc, sc),
        make_subtriangle_panel(sub, sab, sbc, sca)};
    for (const Panel &leaf : second_level) {
      if (!leaf.has_triangle_geometry)
        continue;
      const KernelValues k =
          triangle_gauss_integral(target, leaf, eps_ratio, kappa);
      out.k1 += k.k1;
      out.k2 += k.k2;
      out.k3 += k.k3;
      out.k4 += k.k4;
    }
  }
  return out;
}

void add_stats(TreecodeStats *stats, bool self_pair, bool one_ring,
               bool duffy, bool centroid, bool fallback,
               std::size_t subdivisions) noexcept {
  if (stats == nullptr)
    return;
  if (self_pair)
    ++stats->near_self_pairs;
  if (one_ring)
    ++stats->near_one_ring_pairs;
  if (duffy)
    ++stats->near_duffy_pairs;
  if (centroid)
    ++stats->near_centroid_pairs;
  if (fallback)
    ++stats->near_fallbacks;
  if (subdivisions > 0) {
    const double count = static_cast<double>(stats->near_duffy_pairs);
    stats->duffy_subdivisions_mean =
        count <= 1.0
            ? static_cast<double>(subdivisions)
            : (stats->duffy_subdivisions_mean * (count - 1.0) +
               static_cast<double>(subdivisions)) /
                  count;
    stats->duffy_subdivisions_max =
        std::max(stats->duffy_subdivisions_max, subdivisions);
  }
}

std::size_t geometric_gap_bucket(double gap_ratio) noexcept {
  if (gap_ratio < 0.25)
    return 0;
  if (gap_ratio < 0.50)
    return 1;
  if (gap_ratio <= 0.75)
    return 2;
  return 3;
}

std::size_t geometric_center_bucket(double center_ratio) noexcept {
  if (center_ratio < 1.0)
    return 0;
  if (center_ratio < 2.0)
    return 1;
  return 2;
}

std::size_t geometric_kernel_bucket(const std::string &kernel) noexcept {
  if (kernel == "gauss7")
    return 1;
  if (kernel == "subdivided")
    return 2;
  return 0;
}

std::size_t geometric_audit_index(double gap_ratio, double center_ratio,
                                  const std::string &kernel) noexcept {
  const std::size_t gap = geometric_gap_bucket(gap_ratio);
  const std::size_t center = geometric_center_bucket(center_ratio);
  const std::size_t kernel_index = geometric_kernel_bucket(kernel);
  return (gap * kGeometricCenterBucketCount + center) *
             kGeometricKernelCount +
         kernel_index;
}

void add_geometric_buffer_stats(TreecodeStats *stats,
                                const GeometricNearDecision &decision,
                                const std::string &kernel) noexcept {
  if (stats == nullptr || !decision.active)
    return;
  const std::size_t old_count = stats->near_geometric_buffer_pairs;
  ++stats->near_geometric_buffer_pairs;
  if (kernel == "subdivided")
    ++stats->near_geometric_buffer_adaptive_pairs;
  else if (kernel == "gauss7")
    ++stats->near_geometric_buffer_gauss_pairs;
  else
    ++stats->near_geometric_buffer_centroid_pairs;
  if (old_count == 0 || stats->near_geometric_buffer_min_gap <= 0.0) {
    stats->near_geometric_buffer_min_gap = decision.gap;
  } else {
    stats->near_geometric_buffer_min_gap =
        std::min(stats->near_geometric_buffer_min_gap, decision.gap);
  }
  const double total = static_cast<double>(old_count + 1);
  stats->near_geometric_buffer_mean_gap_ratio =
      (stats->near_geometric_buffer_mean_gap_ratio *
           static_cast<double>(old_count) +
       decision.gap_ratio) /
      total;
  stats->near_geometric_buffer_max_edge_mean =
      (stats->near_geometric_buffer_max_edge_mean *
           static_cast<double>(old_count) +
       decision.max_edge) /
      total;
  const std::size_t index = geometric_audit_index(
      decision.gap_ratio, decision.center_ratio, kernel);
  if (index < stats->geometric_near_audit.size()) {
    auto &bucket = stats->geometric_near_audit[index];
    if (bucket.pairs == 0 || bucket.min_gap <= 0.0)
      bucket.min_gap = decision.gap;
    else
      bucket.min_gap = std::min(bucket.min_gap, decision.gap);
    bucket.max_gap_ratio =
        std::max(bucket.max_gap_ratio, decision.gap_ratio);
    bucket.sum_gap_ratio += decision.gap_ratio;
    bucket.sum_center_ratio += decision.center_ratio;
    ++bucket.pairs;
  }
}

void record_block_decisions(TreecodeStats *stats,
                            const AcceptedKernelValues &accepted,
                            bool one_ring, std::size_t pair_type,
                            std::size_t orientation) noexcept {
  if (stats == nullptr)
    return;
  auto record = [](std::size_t &accepted_count, std::size_t &rejected_count,
                   std::string &policy,
                   const KernelBlockDecision &decision) noexcept {
    if (decision.accepted) {
      ++accepted_count;
    } else {
      ++rejected_count;
    }
    if (accepted_count > 0 && rejected_count > 0)
      policy = "mixed_guarded";
    else
      policy = decision.accepted ? decision.accepted_policy
                                 : decision.rejected_policy;
  };
  record(stats->near_k1_accepted, stats->near_k1_rejected,
         stats->near_k1_policy, accepted.block[0]);
  record(stats->near_k2_accepted, stats->near_k2_rejected,
         stats->near_k2_policy, accepted.block[1]);
  record(stats->near_k3_accepted, stats->near_k3_rejected,
         stats->near_k3_policy, accepted.block[2]);
  record(stats->near_k4_accepted, stats->near_k4_rejected,
         stats->near_k4_policy, accepted.block[3]);
  for (std::size_t block = 0; block < accepted.block.size(); ++block) {
    const std::size_t bucket_index =
        near_block_bucket_index(block, pair_type, orientation);
    if (bucket_index >= stats->near_block_audit.size())
      continue;
    auto &bucket = stats->near_block_audit[bucket_index];
    if (accepted.block[block].accepted)
      ++bucket.accepted;
    else
      ++bucket.rejected;
    if (accepted.candidate_worse_than_centroid[block])
      ++bucket.candidate_worse_than_centroid;
    if (!accepted.block[block].accepted) {
      switch (accepted.block[block].reject_reason) {
      case RejectReason::ReferenceNonconverged:
        ++bucket.reference_nonconverged;
        break;
      case RejectReason::OrientationSignMismatch:
        ++bucket.orientation_sign_mismatch;
        break;
      case RejectReason::StrictReferenceReject:
        ++bucket.strict_ref_reject;
        break;
      case RejectReason::CandidateWorseThanCentroid:
        break;
      case RejectReason::None:
        ++bucket.strict_ref_reject;
        break;
      }
    }
    if (accepted.candidate_ref_rel[block] > bucket.max_candidate_ref_rel)
      bucket.max_candidate_ref_rel = accepted.candidate_ref_rel[block];
    bucket.sum_candidate_ref_rel += accepted.candidate_ref_rel[block];
    if (accepted.centroid_ref_rel[block] > bucket.max_centroid_ref_rel)
      bucket.max_centroid_ref_rel = accepted.centroid_ref_rel[block];
    if (block == 1) {
      if (accepted.k2_laplace_sign_ok[block])
        ++bucket.k2_laplace_sign_ok;
      if (accepted.k2_yukawa_smooth_ref_rel[block] >
          bucket.max_k2_yukawa_smooth_ref_rel)
        bucket.max_k2_yukawa_smooth_ref_rel =
            accepted.k2_yukawa_smooth_ref_rel[block];
      if (accepted.k2_orientation_pair_ref_rel[block] >
          bucket.max_k2_orientation_pair_ref_rel)
        bucket.max_k2_orientation_pair_ref_rel =
            accepted.k2_orientation_pair_ref_rel[block];
    }
    ++bucket.sample_count;
  }
  if (!one_ring)
    return;
  auto record_one_ring = [](std::size_t &accepted_count,
                            std::size_t &rejected_count,
                            const KernelBlockDecision &decision) noexcept {
    if (decision.accepted)
      ++accepted_count;
    else
      ++rejected_count;
  };
  record_one_ring(stats->near_one_ring_k1_accepted,
                  stats->near_one_ring_k1_rejected, accepted.block[0]);
  record_one_ring(stats->near_one_ring_k2_accepted,
                  stats->near_one_ring_k2_rejected, accepted.block[1]);
  record_one_ring(stats->near_one_ring_k3_accepted,
                  stats->near_one_ring_k3_rejected, accepted.block[2]);
  record_one_ring(stats->near_one_ring_k4_accepted,
                  stats->near_one_ring_k4_rejected, accepted.block[3]);
}

void apply_kernel(JcpNearAccum &acc, const KernelValues &kernel,
                  const std::vector<double> &unknowns, std::size_t n,
                  std::size_t j, bool coupled) noexcept {
  const double phi_j = unknowns[j];
  if (!coupled) {
    acc.phi += kernel.k2 * phi_j;
    return;
  }
  const double phin_j = unknowns[n + j];
  acc.phi += kernel.k1 * phin_j + kernel.k2 * phi_j;
  acc.phin += kernel.k3 * phin_j + kernel.k4 * phi_j;
}

bool has_all_triangle_geometry(const std::vector<Panel> &panels) noexcept {
  if (panels.empty())
    return false;
  for (const Panel &panel : panels) {
    if (!panel.has_triangle_geometry)
      return false;
  }
  return true;
}

const char *kernel_policy_for_mode(const std::string &near_integration) noexcept {
  if (near_integration == "analytic_singular")
    return "self_asi";
  if (near_integration == "duffy_adaptive")
    return "one_ring_cpv";
  return "centroid";
}

AcceptedKernelValues guarded_one_ring_cpv_integral_decision(
    const Panel &target, const Panel &source, double eps_ratio,
    double kappa) noexcept;

AcceptedKernelValues guarded_one_ring_cpv_integral_decision(
    const Panel &target, const Panel &source, double eps_ratio,
    double kappa) noexcept {
  const KernelValues centroid =
      centroid_integral(target, source, eps_ratio, kappa);
  K2CandidateInfo k2_info;
  const KernelValues duffy =
      duffy_cpv_integral(target, source, eps_ratio, kappa, false, &k2_info);
  const KernelValues reference_low =
      triangle_gauss_integral(target, source, eps_ratio, kappa);
  const KernelValues reference =
      triangle_gauss_subdivided_integral(target, source, eps_ratio, kappa);
  Panel flipped_source = source;
  flipped_source.nx = -flipped_source.nx;
  flipped_source.ny = -flipped_source.ny;
  flipped_source.nz = -flipped_source.nz;
  const KernelValues flipped_reference =
      triangle_gauss_subdivided_integral(target, flipped_source, eps_ratio,
                                         kappa);
  const double k2_orientation_pair_ref_rel =
      relative_error_value(reference.k2, -flipped_reference.k2);
  const double k2_laplace = k2_info.laplace_integral;
  const double k2_reference_smooth =
      (reference.k2 - (eps_ratio - 1.0) * k2_laplace) /
      std::max(1.0e-12, eps_ratio);
  const double k2_yukawa_smooth_ref_rel =
      relative_error_value(k2_info.yukawa_smooth_integral,
                           k2_reference_smooth);
  const std::array<double, 4> candidate_values = {duffy.k1, duffy.k2,
                                                  duffy.k3, duffy.k4};
  const std::array<double, 4> centroid_values = {centroid.k1, centroid.k2,
                                                 centroid.k3, centroid.k4};
  const std::array<double, 4> reference_values = {
      reference.k1, reference.k2, reference.k3, reference.k4};
  const std::array<double, 4> reference_low_values = {
      reference_low.k1, reference_low.k2, reference_low.k3, reference_low.k4};
  const std::array<double, 4> reference_convergence_limits = {0.10, 0.50,
                                                             0.50, 0.50};
  std::array<bool, 4> reference_converged{};
  std::array<bool, 4> orientation_mismatch{};
  std::array<bool, 4> accept{};
  for (std::size_t block = 0; block < accept.size(); ++block) {
    const double ref_ref_rel =
        relative_error_value(reference_low_values[block],
                             reference_values[block]);
    reference_converged[block] =
        finite_kernel_values(reference_low) && finite_kernel_values(reference) &&
        std::isfinite(ref_ref_rel) &&
        ref_ref_rel <= reference_convergence_limits[block];
    orientation_mismatch[block] =
        block == 1 &&
        (!k2_info.laplace_sign_ok ||
         !same_sign_for_significant_values(candidate_values[block],
                                           reference_values[block]) ||
         !relative_guard_ok(reference_values[block],
                            -flipped_reference.k2, 0.02));
    const bool candidate_worse =
        relative_error_value(candidate_values[block], reference_values[block]) >
        relative_error_value(centroid_values[block], reference_values[block]) +
            1.0e-12;
    const double candidate_ref_rel =
        relative_error_value(candidate_values[block], reference_values[block]);
    const double centroid_ref_rel =
        relative_error_value(centroid_values[block], reference_values[block]);
    const bool candidate_improves_centroid =
        candidate_ref_rel <= 0.95 * centroid_ref_rel;
    accept[block] =
        block == 1 && reference_converged[block] &&
        !orientation_mismatch[block] && !candidate_worse &&
        candidate_improves_centroid &&
        candidate_ref_rel <= 1.0e-8 &&
        relative_guard_ok(candidate_values[block], reference_values[block],
                          0.02);
  }
  AcceptedKernelValues out;
  out.kernel = {accept[0] ? duffy.k1 : centroid.k1,
                accept[1] ? duffy.k2 : centroid.k2,
                accept[2] ? duffy.k3 : centroid.k3,
                accept[3] ? duffy.k4 : centroid.k4};
  const auto reject_reason_for_block =
      [&](std::size_t block) noexcept -> RejectReason {
    if (accept[block])
      return RejectReason::None;
    if (!reference_converged[block])
      return RejectReason::ReferenceNonconverged;
    if (orientation_mismatch[block])
      return RejectReason::OrientationSignMismatch;
    if (relative_error_value(candidate_values[block], reference_values[block]) >
        relative_error_value(centroid_values[block], reference_values[block]) +
            1.0e-12) {
      return RejectReason::CandidateWorseThanCentroid;
    }
    return RejectReason::StrictReferenceReject;
  };
  const auto reject_policy_for_reason = [](RejectReason reason) noexcept {
    switch (reason) {
    case RejectReason::ReferenceNonconverged:
      return "reject_reference_nonconverged";
    case RejectReason::OrientationSignMismatch:
      return "reject_orientation_sign_mismatch";
    case RejectReason::CandidateWorseThanCentroid:
      return "reject_candidate_worse_than_centroid";
    case RejectReason::StrictReferenceReject:
      return "reject_strict_reference";
    case RejectReason::None:
      return "centroid_guarded";
    }
    return "centroid_guarded";
  };
  const std::array<RejectReason, 4> reasons = {
      reject_reason_for_block(0), reject_reason_for_block(1),
      reject_reason_for_block(2), reject_reason_for_block(3)};
  out.block = {
      KernelBlockDecision{accept[0], "duffy_g0_gk",
                          reject_policy_for_reason(reasons[0]), reasons[0]},
      KernelBlockDecision{accept[1], "duffy_source_normal",
                          reject_policy_for_reason(reasons[1]), reasons[1]},
      KernelBlockDecision{accept[2], "duffy_target_normal",
                          reject_policy_for_reason(reasons[2]), reasons[2]},
      KernelBlockDecision{accept[3], "duffy_mixed_normal",
                          reject_policy_for_reason(reasons[3]), reasons[3]}};
  for (std::size_t block = 0; block < out.block.size(); ++block) {
    out.candidate_ref_rel[block] =
        relative_error_value(candidate_values[block], reference_values[block]);
    out.centroid_ref_rel[block] =
        relative_error_value(centroid_values[block], reference_values[block]);
    out.candidate_worse_than_centroid[block] =
        out.candidate_ref_rel[block] > out.centroid_ref_rel[block] + 1.0e-12;
  }
  out.k2_laplace_sign_ok[1] = k2_info.laplace_sign_ok;
  out.k2_yukawa_smooth_ref_rel[1] = k2_yukawa_smooth_ref_rel;
  out.k2_orientation_pair_ref_rel[1] = k2_orientation_pair_ref_rel;
  return out;
}

// Gauss-Legendre on [0,1]. Immutable rules are shared between surface builds.
std::vector<Quad1D> sphere_gauss_rule(int order) {
  std::vector<Quad1D> rule;
  for (int i = 0; i < order; ++i) {
    double z = std::cos(kPi * (i + 0.75) / (order + 0.5));
    double derivative = 0.0;
    for (int iteration = 0; iteration < 64; ++iteration) {
      double p = 1.0, previous = 0.0;
      for (int k = 1; k <= order; ++k) {
        const double old = p;
        p = ((2 * k - 1) * z * p - (k - 1) * previous) / k;
        previous = old;
      }
      derivative = order * (z * p - previous) / (z * z - 1.0);
      const double step = p / derivative;
      z -= step;
      if (std::abs(step) < 2.0e-15)
        break;
    }
    rule.push_back({0.5 * (1.0 + z),
                    1.0 / ((1.0 - z * z) * derivative * derivative)});
  }
  return rule;
}

const std::vector<Quad1D> &sphere_rule(std::size_t level) {
  static const std::array<std::vector<Quad1D>, 7> rules = {
      sphere_gauss_rule(8), sphere_gauss_rule(16), sphere_gauss_rule(32),
      sphere_gauss_rule(64), sphere_gauss_rule(128), sphere_gauss_rule(256),
      sphere_gauss_rule(512)};
  return rules.at(level);
}

double sphere_self_rule(const std::array<Vec3, 3> &vertices,
                        const std::vector<Quad1D> &rule,
                        SpherePatchQuadrature *prepared = nullptr,
                        const Vec3 *target_point = nullptr) {
  const Vec3 center = (1.0 / 3.0) * (vertices[0] + vertices[1] + vertices[2]);
  const Vec3 target = target_point ? *target_point : (1.0 / norm(center)) * center;
  double integral = 0.0;
  for (int edge = 0; edge < 3; ++edge) {
    const Vec3 a = vertices[edge], b = vertices[(edge + 1) % 3];
    for (const auto &u : rule) {
      for (const auto &v : rule) {
        const Vec3 radial = (1.0 - v.x) * a + v.x * b - center;
        const Vec3 p = center + u.x * radial;
        const double length = norm(p);
        const Vec3 y = (1.0 / length) * p;
        const double weight = u.weight * v.weight * u.x *
            std::abs(dot(p, cross(radial, b - a))) /
            (length * length * length);
        // On a unit sphere n_y.(x-y)=-|x-y|^2/2. Use this cancellation-free
        // form for the weakly singular curved self term, separate from jump.
        const Vec3 delta = target - y;
        const double distance = norm(delta);
        if (!(distance > 1.0e-14))
          return std::numeric_limits<double>::quiet_NaN();
        integral += target_point
            ? weight * dot(y, delta) / (4.0 * kPi * distance * distance * distance)
            : -weight / (8.0 * kPi * distance);
        if (prepared)
          prepared->nodes.push_back({y.x, y.y, y.z, weight});
      }
    }
  }
  return integral;
}

// For a self target t inside a spherical triangle, polar integration gives
// integral_S 1/|y-t| dS = integral_boundary |y-t| d(azimuth).
// This removes the radial singularity analytically. Adaptive 1D integration
// resolves the narrow angular peaks of clipped slivers without an N^2 rule.
double sphere_self_boundary(const std::array<Vec3, 3> &vertices) {
  using L = std::array<long double, 3>;
  auto dotl = [](L a, L b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto crossl = [](L a, L b) {
    return L{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]};
  };
  L target{0, 0, 0};
  for (const auto &v : vertices) {
    target[0] += v.x;
    target[1] += v.y;
    target[2] += v.z;
  }
  const long double target_length = std::sqrt(dotl(target, target));
  for (auto &x : target)
    x /= target_length;
  double integral = 0;
  for (int edge = 0; edge < 3; ++edge) {
    const auto &av = vertices[edge], &bv = vertices[(edge + 1) % 3];
    const L a{av.x, av.y, av.z}, e{static_cast<long double>(bv.x) - av.x,
                                   static_cast<long double>(bv.y) - av.y,
                                   static_cast<long double>(bv.z) - av.z};
    const long double numerator = std::abs(dotl(target, crossl(a, e)));
    auto rule = [&](double lo, double hi, std::size_t level) {
      long double sum = 0;
      for (const auto &v : sphere_rule(level)) {
        const long double f = lo + (static_cast<long double>(hi) - lo) * v.x;
        const L p{a[0] + f * e[0], a[1] + f * e[1], a[2] + f * e[2]};
        const long double length = std::sqrt(dotl(p, p));
        const L delta{p[0] / length - target[0], p[1] / length - target[1],
                      p[2] / length - target[2]};
        const L transverse = crossl(target, p);
        const long double denominator = dotl(transverse, transverse);
        if (!(denominator > 0))
          return std::numeric_limits<double>::quiet_NaN();
        sum +=
            v.weight * std::sqrt(dotl(delta, delta)) * numerator / denominator;
      }
      return static_cast<double>((hi - lo) * sum);
    };
    auto adaptive = [&](auto &&self, double lo, double hi,
                        int depth) -> double {
      const double coarse = rule(lo, hi, 0), fine = rule(lo, hi, 1);
      if (std::isfinite(fine) &&
          std::abs(fine - coarse) <= 1e-8 * std::abs(fine) + 2e-13 * (hi - lo))
        return fine;
      if (depth == 18) {
        // A clipped sliver: the edge interval is 2^-18 of the edge and its
        // integral already negligible against the patch.  The finer value
        // is kept and counted rather than ending the solve, as for an
        // unconverged curved near pair.  A non-finite value still stops it.
        if (!std::isfinite(fine))
          throw std::runtime_error("sphere_patch_self_boundary_nonconverged");
        static std::atomic<std::size_t> unconverged{0};
        if (unconverged.fetch_add(1, std::memory_order_relaxed) == 0)
          std::cerr << "TABI sphere-patch: a self-boundary interval of a "
                       "clipped sliver did not converge by depth 18; its "
                       "finer estimate is used (last change "
                    << std::abs(fine - coarse) << ", further ones not "
                    << "reported)\n";
        return fine;
      }
      const double mid = (lo + hi) * .5;
      return self(self, lo, mid, depth + 1) + self(self, mid, hi, depth + 1);
    };
    integral += adaptive(adaptive, 0, 1, 0);
  }
  return -integral / (8 * kPi);
}

bool sphere_poisson_near(const Panel &target, const Panel &source,
                         double eps_ratio, bool self, double &value) {
  const auto &prepared = *source.sphere_patch_quadrature;
  if (self) {
    value = (eps_ratio - 1.0) * prepared.self_double_layer;
    return std::isfinite(value);
  }
  const auto &sphere = sphere_patch_sphere(source);
  const Vec3 t = {(target.x - sphere[0]) / sphere[3],
                  (target.y - sphere[1]) / sphere[3],
                  (target.z - sphere[2]) / sphere[3]};
  const bool same_sphere = target.sphere_patch_quadrature &&
      sphere_patch_sphere(target) == sphere;
  if (!same_sphere) {
    // Rare but reachable at a globally merged vertex of touching spheres.
    // Refine this near integral instead of applying the same-sphere identity.
    const Vec3 origin = {sphere[0],sphere[1],sphere[2]};
    const std::array<Vec3,3> vertices = {
        (1.0/sphere[3])*(panel_vertex(source,0)-origin),
        (1.0/sphere[3])*(panel_vertex(source,1)-origin),
        (1.0/sphere[3])*(panel_vertex(source,2)-origin)};
    double previous = sphere_self_rule(vertices,sphere_rule(0),nullptr,&t);
    if (!std::isfinite(previous)) return false;
    for (std::size_t level=1;level<=4;++level) {
      const double current = sphere_self_rule(vertices,sphere_rule(level),nullptr,&t);
      if (!std::isfinite(current)) return false;
      const double tolerance = 1e-7*std::max(std::abs(current),std::abs(previous)) +
          64*std::numeric_limits<double>::epsilon();
      if (std::abs(current-previous)<=tolerance) {
        value=(eps_ratio-1)*current;
        return std::isfinite(value);
      }
      previous=current;
    }
    return false;
  }
  double integral = 0.0;
  for (const auto &node : prepared.nodes) {
    const Vec3 y = {node[0], node[1], node[2]};
    const Vec3 delta = t - y;
    const double distance = norm(delta);
    if (!(distance > 1.0e-14))
      return false;
    integral += same_sphere ? -node[3] / (8.0 * kPi * distance)
        : node[3] * dot(y, delta) /
              (4.0 * kPi * distance * distance * distance);
  }
  value = (eps_ratio - 1.0) * integral;
  return std::isfinite(value);
}

// Screened-minus-unscreened radial factors, written so that no digits are
// lost near the singular point.  With s = kappa*r and
// g(r) = (exp(-s)-1)/(4*pi*r) the difference of the two Green functions,
//   E1(s) = 1-(1+s)exp(-s) = integral_0^s t exp(-t) dt,  g'(r) = E1(s)/(4 pi r^2)
//   Q(s)  = exp(-s)/s + 2exp(-s)/s^2 + 2(exp(-s)-1)/s^3,  g''(r) = kappa^3 Q(s)/(4 pi)
// Evaluating g' and g'' by subtracting the two kernels instead would cancel
// twelve digits at small r, because each carries a 1/r^3 term that the
// difference does not.  Both series below are the exact Taylor expansions.
double screened_e1(double s) noexcept {
  if (s < 0.5) {
    const double s2 = s * s;
    return s2 * (1.0 / 2.0 +
                 s * (-1.0 / 3.0 +
                      s * (1.0 / 8.0 +
                           s * (-1.0 / 30.0 + s * (1.0 / 144.0)))));
  }
  return 1.0 - (1.0 + s) * std::exp(-s);
}

double screened_q(double s) noexcept {
  if (s < 0.5)
    return -1.0 / 3.0 +
           s * (1.0 / 4.0 + s * (-1.0 / 10.0 + s * (1.0 / 36.0)));
  const double e = std::exp(-s);
  return e / s + 2.0 * e / (s * s) + 2.0 * std::expm1(-s) / (s * s * s);
}

// Curved self and one-ring integration for the coupled (salt > 0) kernels on a
// sphere-patch surface.  The unscreened double layer is taken from the cached,
// adaptively converged self value where it applies; only the screened
// correction is re-integrated on the stored nodes, and that correction is
// strictly less singular than either kernel alone.
bool sphere_coupled_near(const Panel &target, const Panel &source,
                         double eps_ratio, double kappa, bool self,
                         KernelValues &out) {
  const auto &prepared = *source.sphere_patch_quadrature;
  const auto &sphere = sphere_patch_sphere(source);
  const double radius = sphere[3];
  if (!(radius > 0.0) || !(eps_ratio > 0.0))
    return false;
  if (!self && !(target.sphere_patch_quadrature &&
                 sphere_patch_sphere(target) == sphere))
    return false; // cross-sphere coupled pairs keep the centroid rule
  const Vec3 t = {(target.x - sphere[0]) / radius,
                  (target.y - sphere[1]) / radius,
                  (target.z - sphere[2]) / radius};
  const double kappa_hat = kappa * radius;
  double single = 0.0;   // integral of (G0 - Gk), in unit-sphere measure
  double layer0 = 0.0;   // integral of dG0/dn_y
  double layer_k = 0.0;  // integral of (dGk/dn_y - dG0/dn_y)
  double mixed = 0.0;    // integral of (d2Gk - d2G0), in unit-sphere measure
  for (const auto &node : prepared.nodes) {
    const Vec3 y = {node[0], node[1], node[2]};
    const double distance = norm(t - y);
    if (!(distance > 1.0e-14))
      return false;
    const double s = kappa_hat * distance;
    const double e1 = screened_e1(s);
    const double weight = node[3];
    single += weight * (-std::expm1(-s)) / distance;
    layer0 += weight * (-1.0) / distance;
    layer_k += weight * e1 / distance;
    // n_x.n_y = 1 - d^2/2 and (n_x.u)(n_y.u)/r^2 = -d^2/4 on one sphere.
    mixed += weight * (kappa_hat * kappa_hat * kappa_hat * screened_q(s) *
                           distance * distance / 4.0 -
                       e1 * (1.0 - distance * distance / 4.0) /
                           (distance * distance * distance));
  }
  const double layer_unscreened =
      self ? prepared.self_double_layer : layer0 / (8.0 * kPi);
  const double layer_screened = layer_k / (8.0 * kPi);
  out.k1 = radius * single / (4.0 * kPi);
  out.k2 = (eps_ratio - 1.0) * layer_unscreened + eps_ratio * layer_screened;
  out.k3 = (1.0 - 1.0 / eps_ratio) * layer_unscreened -
           layer_screened / eps_ratio;
  out.k4 = mixed / (4.0 * kPi * radius);
  return std::isfinite(out.k1) && std::isfinite(out.k2) &&
         std::isfinite(out.k3) && std::isfinite(out.k4);
}

const char *poisson_policy(const Panel &panel, const std::string &requested) noexcept {
  if (panel.sphere_patch_quadrature && requested != "centroid")
    return "sphere_poisson_curved";
  if (is_node_patch_panel(panel))
    return "poisson_node_patch";
  return "poisson_centroid";
}

} // namespace

std::shared_ptr<const SpherePatchQuadrature> prepare_sphere_patch_quadrature(
    const std::array<double, 3> &a, const std::array<double, 3> &b,
    const std::array<double, 3> &c,
    std::shared_ptr<const std::vector<std::array<double, 4>>> spheres,
    bool clipped) {
  const std::array<Vec3, 3> vertices = {
      {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}, {c[0], c[1], c[2]}}};
  const std::array<double, 3> edges = {norm(vertices[1] - vertices[0]),
                                       norm(vertices[2] - vertices[1]),
                                       norm(vertices[0] - vertices[2])};
  if (*std::min_element(edges.begin(), edges.end()) <= 1.0e-14)
    throw std::runtime_error("sphere_patch_degenerate_triangle");
  auto prepared = std::make_shared<SpherePatchQuadrature>();
  prepared->spheres = std::move(spheres);
  const double largest = *std::max_element(edges.begin(), edges.end());
  const bool sliver =
      largest > 4.0 * *std::min_element(edges.begin(), edges.end()) ||
      norm(cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) <
          .1 * largest * largest;
  const std::size_t near_level = sliver ? 1 : 0;
  const auto &rule = sphere_rule(near_level);
  prepared->nodes.reserve(3 * rule.size() * rule.size());
  double previous = sphere_self_rule(vertices, rule, prepared.get());
  if (!std::isfinite(previous) ||
      prepared->nodes.size() != 3 * rule.size() * rule.size())
    throw std::runtime_error("sphere_patch_quadrature_nodes_invalid");
  if (clipped || near_level > 0) {
    prepared->self_double_layer = sphere_self_boundary(vertices);
    return prepared;
  }
  for (std::size_t level = near_level + 1; level < 7; ++level) {
    const double current = sphere_self_rule(vertices, sphere_rule(level));
    const double tolerance = 1.0e-8 * std::max(std::abs(current), std::abs(previous)) +
        64.0 * std::numeric_limits<double>::epsilon();
    if (std::isfinite(current) && std::abs(current - previous) <= tolerance) {
      prepared->self_double_layer = current;
      return prepared;
    }
    previous = current;
  }
  throw std::runtime_error("sphere_patch_self_quadrature_nonconverged");
}

std::size_t sphere_patch_quadrature_storage_bytes(const std::vector<Panel> &panels) {
  std::unordered_set<const SpherePatchQuadrature *> seen;
  std::unordered_set<const void *> seen_spheres;
  std::size_t bytes = 0;
  for (const auto &panel : panels) {
    const auto *quadrature = panel.sphere_patch_quadrature.get();
    if (quadrature && seen.insert(quadrature).second) {
      bytes += sizeof(*quadrature) + quadrature->nodes.capacity() *
          sizeof(std::array<double, 4>);
      if (quadrature->spheres && seen_spheres.insert(quadrature->spheres.get()).second)
        bytes += sizeof(*quadrature->spheres) + quadrature->spheres->capacity() *
            sizeof(std::array<double, 4>);
    }
  }
  return bytes;
}

std::string resolve_near_integration_mode(const std::string &requested,
                                          const std::vector<Panel> &panels) {
  const std::string mode = requested.empty() ? std::string("auto") : requested;
  const bool node_patch_ready =
      !panels.empty() &&
      std::all_of(panels.begin(), panels.end(), is_node_patch_panel);
  const bool triangle_ready =
      has_all_triangle_geometry(panels) || node_patch_ready;
  if (mode == "auto")
    return std::any_of(panels.begin(), panels.end(), [](const Panel &panel) {
      return static_cast<bool>(panel.sphere_patch_quadrature);
    }) ? "analytic_singular" : "centroid";
  if (mode == "centroid")
    return mode;
  if ((mode == "analytic_singular" || mode == "duffy_adaptive") &&
      !triangle_ready) {
    throw std::runtime_error(
        "PBE BIE: -pbe-bie-near-integration requires triangle surface "
        "geometry; use -pbe-bie-geometry=triangle or near-integration=auto.");
  }
  if (mode != "analytic_singular" && mode != "duffy_adaptive")
    throw std::runtime_error(
        "PBE BIE: invalid near integration mode; expected auto, centroid, "
        "analytic_singular, or duffy_adaptive.");
  return mode;
}

std::string resolve_quad_buffer_mode(const std::string &requested,
                                     const std::string &near_integration,
                                     const std::vector<Panel> &panels) {
  const std::string mode = requested.empty() ? std::string("auto") : requested;
  if (mode == "off")
    return "off";
  (void)near_integration;
  if (mode == "auto")
    return "off";
  if (mode != "gauss7" && mode != "adaptive")
    throw std::runtime_error(
        "PBE BIE: invalid quad-buffer mode; expected auto, off, gauss7, "
        "or adaptive.");
  const bool node_patch_ready =
      !panels.empty() &&
      std::all_of(panels.begin(), panels.end(), is_node_patch_panel);
  if (!has_all_triangle_geometry(panels) && !node_patch_ready) {
    throw std::runtime_error(
        "PBE BIE: -pbe-bie-quad-buffer requires triangle surface geometry; "
        "use -pbe-bie-geometry=triangle or quad-buffer=auto/off.");
  }
  return mode;
}

GeometricNearDecision classify_geometric_near_buffer_pair(
    const Panel &target, const Panel &source,
    const std::string &quad_buffer) noexcept {
  GeometricNearDecision decision;
  if (quad_buffer == "off" || quad_buffer.empty() ||
      !target.has_triangle_geometry || !source.has_triangle_geometry) {
    return decision;
  }
  const double target_edge = panel_max_edge_length(target);
  const double source_edge = panel_max_edge_length(source);
  const double max_edge = std::max(target_edge, source_edge);
  if (!std::isfinite(max_edge) || max_edge <= 1.0e-12)
    return decision;
  const double threshold = 0.75 * max_edge;
  const Vec3 target_center = make_vec(target.x, target.y, target.z);
  const Vec3 source_center = make_vec(source.x, source.y, source.z);
  const double center_distance =
      std::sqrt(std::max(0.0, distance_sq(target_center, source_center)));
  const double radius_sum =
      panel_centroid_radius(target) + panel_centroid_radius(source);
  if (center_distance - radius_sum > threshold)
    return decision;
  const double gap = std::sqrt(std::max(
      0.0, triangle_triangle_distance_sq(target, source)));
  const double gap_ratio = gap / max_edge;
  if (!std::isfinite(gap_ratio) || gap_ratio > 0.75)
    return decision;
  decision.active = true;
  decision.gap = gap;
  decision.gap_ratio = gap_ratio;
  decision.center_ratio = center_distance / max_edge;
  decision.max_edge = max_edge;
  decision.adaptive = quad_buffer == "adaptive" && gap_ratio <= 0.35;
  return decision;
}

JcpNearPairKernel evaluate_jcp_near_pair_kernel(
    const Panel &target, std::size_t target_index, const Panel &source,
    std::size_t source_index, double eps_ratio, double kappa, bool coupled,
    const std::string &near_integration, const std::string &quad_buffer,
    TreecodeStats *stats) {
  JcpNearPairKernel selected;
  selected.self_pair = source_index == target_index;
  const bool node_patch_pair =
      is_node_patch_panel(target) && is_node_patch_panel(source);
  const int shared_vertices =
      selected.self_pair ? 3 : shared_vertex_count(target, source);
  selected.one_ring =
      !selected.self_pair &&
      (node_patch_pair ? node_patch_adjacent(target, source)
                       : shared_vertices > 0);
  KernelValues kernel;
  bool used_duffy = false;
  bool used_centroid = false;
  bool fallback = false;
  std::size_t subdivisions = 0;
  GeometricNearDecision geometric_decision;
  std::string geometric_kernel;
  AcceptedKernelValues block_decision;
  bool has_block_decision = false;

  if (node_patch_pair && curved_dual_cell(source) && !selected.self_pair &&
      !selected.one_ring && curved_pair_takes_one_point(target, source)) {
    // FMM route, outside the curved near zone: the one-point kernel the
    // packed near loop and the multipole moments use for this cell.
    kernel = centroid_integral(target, source, eps_ratio, coupled ? kappa : 0.0);
    if (!coupled)
      kernel = {0.0, kernel.k2, 0.0, 0.0};
    used_centroid = true;
  } else if (node_patch_pair && curved_dual_cell(source)) {
    const auto block = curved_node_pair(target, target_index, source,
                                        source_index, eps_ratio, kappa);
    kernel = {coupled ? block[0] : 0.0, block[1],
              coupled ? block[2] : 0.0, coupled ? block[3] : 0.0};
    used_duffy = true;
    subdivisions = selected.self_pair || selected.one_ring ? 16 : 4;
    // On the FMM route a curved pair beyond the one-ring replaces the
    // one-point kernel, so the packed near loop must correct it as it
    // corrects the one-ring.
    if (!selected.self_pair && !selected.one_ring &&
        source.node_patch_geometry->curved->one_point_beyond_radii > 0.0) {
      selected.geometric_buffer = true;
      geometric_kernel = "curved_near";
    }
  } else if (node_patch_pair && !coupled) {
    if (selected.self_pair)
      return selected;
    kernel = selected.one_ring
                 ? node_patch_near_integral(target, source, eps_ratio, 0.0,
                                            false)
                 : centroid_integral(target, source, eps_ratio, 0.0);
    used_duffy = selected.one_ring;
    used_centroid = !selected.one_ring;
    subdivisions = selected.one_ring ? 32 : 0;
  } else if (node_patch_pair && selected.self_pair) {
    kernel = self_singular_integral(source, eps_ratio, kappa);
    used_duffy = true;
    subdivisions = 1;
  } else if (!coupled && source.sphere_patch_quadrature &&
             near_integration != "centroid" &&
             (selected.self_pair || selected.one_ring)) {
    if (sphere_poisson_near(target, source, eps_ratio, selected.self_pair, kernel.k2)) {
      used_duffy = true;
      subdivisions = 3;
    } else {
      kernel = centroid_integral(target, source, eps_ratio, 0.0);
      used_centroid = true;
      fallback = true;
    }
  } else if (node_patch_pair && selected.one_ring) {
    // Coupled one-ring pairs used to fall through to a centroid monopole,
    // so the patch integration above applied only at zero salt.  Which near
    // rule is used must not depend on ionic strength.
    kernel = node_patch_near_integral(target, source, eps_ratio, kappa, false);
    used_duffy = true;
    subdivisions = 32;
  } else if (coupled && source.sphere_patch_quadrature &&
             near_integration != "centroid" &&
             (selected.self_pair || selected.one_ring)) {
    // Same correction as the scalar branch above, for salted runs.  Without
    // it every ionic-strength calculation keeps the flat-panel self term.
    if (sphere_coupled_near(target, source, eps_ratio, kappa,
                            selected.self_pair, kernel)) {
      used_duffy = true;
      subdivisions = 3;
    } else if (selected.self_pair) {
      kernel = self_singular_integral(source, eps_ratio, kappa);
      used_duffy = true;
      subdivisions = 1;
      fallback = true;
    } else {
      kernel = centroid_integral(target, source, eps_ratio, kappa);
      used_centroid = true;
      fallback = true;
    }
  } else if (!coupled) {
    if (selected.self_pair)
      return selected;
    kernel = centroid_integral(target, source, eps_ratio, 0.0);
    used_centroid = true;
  } else if (near_integration == "centroid") {
    if (selected.self_pair)
      return selected;
    kernel = centroid_integral(target, source, eps_ratio, kappa);
    used_centroid = true;
  } else if (selected.self_pair) {
    if (!target.has_triangle_geometry || !source.has_triangle_geometry)
      throw std::runtime_error(
          "PBE BIE: analytic near integration requires triangle geometry "
          "for self panels.");
    block_decision = self_singular_decision(source, eps_ratio, kappa);
    kernel = block_decision.kernel;
    has_block_decision = true;
    used_duffy = true;
    subdivisions = 3;
  } else if (selected.one_ring && near_integration == "duffy_adaptive") {
    if (!target.has_triangle_geometry || !source.has_triangle_geometry)
      throw std::runtime_error(
          "PBE BIE: Duffy one-ring near integration requires triangle "
          "geometry.");
    block_decision =
        guarded_one_ring_cpv_integral_decision(target, source, eps_ratio,
                                               kappa);
    kernel = block_decision.kernel;
    has_block_decision = true;
    used_duffy = true;
    subdivisions = 8;
  } else if (selected.one_ring) {
    kernel = centroid_integral(target, source, eps_ratio, kappa);
    used_centroid = true;
  } else if ((geometric_decision = classify_geometric_near_buffer_pair(
                  target, source, quad_buffer))
                 .active) {
    selected.geometric_buffer = true;
    if (geometric_decision.adaptive) {
      kernel =
          triangle_gauss_subdivided_integral(target, source, eps_ratio, kappa);
      geometric_kernel = "subdivided";
      subdivisions = 16;
    } else {
      kernel = triangle_gauss_integral(target, source, eps_ratio, kappa);
      geometric_kernel = "gauss7";
      subdivisions = 1;
    }
  } else {
    kernel = centroid_integral(target, source, eps_ratio, kappa);
    used_centroid = true;
  }

  if (!finite_kernel(kernel))
    throw std::runtime_error(
        "PBE BIE: non-finite analytic near-integration kernel.");
  selected.k1 = kernel.k1;
  selected.k2 = kernel.k2;
  selected.k3 = kernel.k3;
  selected.k4 = kernel.k4;
  selected.used_duffy = used_duffy;
  selected.used_centroid = used_centroid;
  selected.valid = true;
  selected.policy = selected.geometric_buffer
                        ? geometric_kernel
                        : (used_duffy ? std::string("duffy")
                                      : (used_centroid ? std::string("centroid")
                                                       : std::string("near")));
  add_stats(stats, selected.self_pair, selected.one_ring, used_duffy,
            used_centroid, fallback, subdivisions);
  if (geometric_decision.active)
    add_geometric_buffer_stats(stats, geometric_decision, geometric_kernel);
  if (has_block_decision) {
    const std::size_t pair_type =
        classify_pair_type(selected.self_pair, shared_vertices);
    const std::size_t orientation =
        selected.self_pair ? 0 : classify_orientation(target, source);
    record_block_decisions(stats, block_decision, selected.one_ring, pair_type,
                           orientation);
  }
  if (stats != nullptr) {
    stats->near_integration = near_integration;
    stats->quad_buffer = quad_buffer;
    const char *policy = curved_dual_cell(target) ? "curved_dual" : coupled ? kernel_policy_for_mode(near_integration)
                                 : poisson_policy(target, near_integration);
    if (stats->near_kernel_policy != policy)
      stats->near_kernel_policy = policy;
  }
  return selected;
}

JcpNearPairKernel evaluate_jcp_centroid_monopole_kernel(
    const Panel &target, const Panel &source, double eps_ratio,
    double kappa) noexcept {
  JcpNearPairKernel selected;
  const KernelValues kernel =
      centroid_integral(target, source, eps_ratio, kappa);
  if (!finite_kernel(kernel))
    return selected;
  selected.k1 = kernel.k1;
  selected.k2 = kernel.k2;
  selected.k3 = kernel.k3;
  selected.k4 = kernel.k4;
  selected.used_centroid = true;
  selected.valid = true;
  selected.policy = "centroid_monopole";
  return selected;
}

#if TITAN_SIMD_X86
namespace {

TITAN_TARGET_AVX2
inline bool all_lanes_finite(__m256d v) noexcept {
  const __m256d magnitude = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
  return _mm256_movemask_pd(_mm256_cmp_pd(
             magnitude,
             _mm256_set1_pd(std::numeric_limits<double>::infinity()),
             _CMP_LT_OQ)) == 0xf;
}

// Plain-centroid kernels of four sources for one target, added into four
// per-lane row sums (acc_phi/acc_phin, 4 doubles each). Each pointer holds the
// four lanes of one field; phin is unused for uncoupled rows. The closed forms
// are those of point_kernel_shared() with one division per lane (1/r) and the
// accurate AVX2 exp, so each term agrees with the scalar path to rounding.
// Lanes with r <= 1e-14 contribute nothing, as in the scalar kernel. If any
// lane has a non-finite r or kernel, nothing is added and false is returned,
// so the caller reruns those pairs on the scalar path and keeps its failure
// behaviour.
TITAN_TARGET_AVX2
bool plain_centroid_quad_compute_avx2(
    const Panel &target, const double *sx, const double *sy, const double *sz,
    const double *snx, const double *sny, const double *snz,
    const double *area, const double *phi, const double *phin,
    double eps_ratio, double kk, bool coupled, double *acc_phi,
    double *acc_phin) noexcept {
  const __m256d dx = _mm256_sub_pd(_mm256_set1_pd(target.x), _mm256_loadu_pd(sx));
  const __m256d dy = _mm256_sub_pd(_mm256_set1_pd(target.y), _mm256_loadu_pd(sy));
  const __m256d dz = _mm256_sub_pd(_mm256_set1_pd(target.z), _mm256_loadu_pd(sz));
  const __m256d r2 = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
      _mm256_mul_pd(dz, dz));
  const __m256d r = _mm256_sqrt_pd(r2);
  const __m256d infinity =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  if (_mm256_movemask_pd(_mm256_cmp_pd(r, infinity, _CMP_LT_OQ)) != 0xf)
    return false;
  const __m256d live = _mm256_cmp_pd(r, _mm256_set1_pd(1.0e-14), _CMP_GT_OQ);
  const __m256d one = _mm256_set1_pd(1.0);
  const __m256d r_safe = _mm256_blendv_pd(one, r, live);
  const __m256d inv_r = _mm256_div_pd(one, r_safe);
  const __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
  const __m256d inv_r3 = _mm256_mul_pd(inv_r2, inv_r);
  const __m256d q = _mm256_set1_pd(1.0 / (4.0 * kPi));
  const __m256d snx_v = _mm256_loadu_pd(snx);
  const __m256d sny_v = _mm256_loadu_pd(sny);
  const __m256d snz_v = _mm256_loadu_pd(snz);
  const __m256d source_projection = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(snx_v, dx), _mm256_mul_pd(sny_v, dy)),
      _mm256_mul_pd(snz_v, dz));
  const __m256d source_d0 =
      _mm256_mul_pd(_mm256_mul_pd(q, source_projection), inv_r3);
  const __m256d area_v = _mm256_loadu_pd(area);
  const __m256d phi_v = _mm256_loadu_pd(phi);
  if (!coupled) {
    const __m256d k2 = _mm256_mul_pd(
        area_v, _mm256_mul_pd(_mm256_set1_pd(eps_ratio - 1.0), source_d0));
    if (!all_lanes_finite(k2))
      return false;
    _mm256_store_pd(acc_phi,
                    _mm256_add_pd(_mm256_load_pd(acc_phi),
                                  _mm256_and_pd(live, _mm256_mul_pd(k2, phi_v))));
    return true;
  }

  const __m256d kappa = _mm256_set1_pd(kk);
  const __m256d kr = _mm256_mul_pd(kappa, r_safe);
  const __m256d e = titan_avx2::exp_pd(_mm256_sub_pd(_mm256_setzero_pd(), kr));
  const __m256d e1 = _mm256_mul_pd(e, _mm256_add_pd(one, kr));
  const __m256d g0 = _mm256_mul_pd(q, inv_r);
  const __m256d k1 = _mm256_mul_pd(area_v, _mm256_sub_pd(g0, _mm256_mul_pd(e, g0)));
  const __m256d k2 = _mm256_mul_pd(
      area_v, _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(eps_ratio),
                                          _mm256_mul_pd(e1, source_d0)),
                            source_d0));
  const __m256d tnx = _mm256_set1_pd(target.nx);
  const __m256d tny = _mm256_set1_pd(target.ny);
  const __m256d tnz = _mm256_set1_pd(target.nz);
  const __m256d target_projection = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(tnx, dx), _mm256_mul_pd(tny, dy)),
      _mm256_mul_pd(tnz, dz));
  const __m256d target_d0 = _mm256_sub_pd(
      _mm256_setzero_pd(),
      _mm256_mul_pd(_mm256_mul_pd(q, target_projection), inv_r3));
  const double inv_eps_ratio = 1.0 / std::max(1.0e-12, eps_ratio);
  const __m256d k3 = _mm256_mul_pd(
      area_v, _mm256_sub_pd(target_d0,
                            _mm256_mul_pd(_mm256_set1_pd(inv_eps_ratio),
                                          _mm256_mul_pd(e1, target_d0))));
  const __m256d nxu = _mm256_mul_pd(target_projection, inv_r);
  const __m256d nyu = _mm256_mul_pd(source_projection, inv_r);
  const __m256d nxny = _mm256_add_pd(
      _mm256_add_pd(_mm256_mul_pd(tnx, snx_v), _mm256_mul_pd(tny, sny_v)),
      _mm256_mul_pd(tnz, snz_v));
  const __m256d nxu_nyu = _mm256_mul_pd(nxu, nyu);
  const __m256d mixed0 = _mm256_mul_pd(
      _mm256_mul_pd(q, inv_r3),
      _mm256_sub_pd(nxny, _mm256_mul_pd(_mm256_set1_pd(3.0), nxu_nyu)));
  const __m256d fprime_over_r = _mm256_sub_pd(
      _mm256_setzero_pd(), _mm256_mul_pd(_mm256_mul_pd(e1, q), inv_r3));
  const __m256d fsecond = _mm256_mul_pd(
      _mm256_mul_pd(e, q),
      _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(kk * kk), inv_r),
                        _mm256_mul_pd(_mm256_set1_pd(2.0 * kk), inv_r2)),
          _mm256_mul_pd(_mm256_set1_pd(2.0), inv_r3)));
  const __m256d mixedk = _mm256_sub_pd(
      _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(fsecond, nxu_nyu)),
      _mm256_mul_pd(fprime_over_r, _mm256_sub_pd(nxny, nxu_nyu)));
  const __m256d k4 = _mm256_mul_pd(area_v, _mm256_sub_pd(mixedk, mixed0));
  if (!all_lanes_finite(k1) || !all_lanes_finite(k2) ||
      !all_lanes_finite(k3) || !all_lanes_finite(k4))
    return false;
  const __m256d phin_v = _mm256_loadu_pd(phin);
  const __m256d add_phi = _mm256_add_pd(_mm256_mul_pd(k1, phin_v),
                                        _mm256_mul_pd(k2, phi_v));
  const __m256d add_phin = _mm256_add_pd(_mm256_mul_pd(k3, phin_v),
                                         _mm256_mul_pd(k4, phi_v));
  _mm256_store_pd(acc_phi, _mm256_add_pd(_mm256_load_pd(acc_phi),
                                         _mm256_and_pd(live, add_phi)));
  _mm256_store_pd(acc_phin, _mm256_add_pd(_mm256_load_pd(acc_phin),
                                          _mm256_and_pd(live, add_phin)));
  return true;
}

// Four sources read from their Panels.
TITAN_TARGET_AVX2
bool plain_centroid_quad_avx2(const Panel &target,
                              const std::vector<Panel> &panels,
                              const std::size_t (&source)[4],
                              const std::vector<double> &unknowns,
                              std::size_t n, double eps_ratio, double kk,
                              bool coupled, double *acc_phi,
                              double *acc_phin) noexcept {
  alignas(32) double sx[4], sy[4], sz[4], snx[4], sny[4], snz[4], area[4],
      phi[4], phin[4];
  for (int lane = 0; lane < 4; ++lane) {
    const Panel &s = panels[source[lane]];
    const bool centre = s.has_source_quadrature_center;
    sx[lane] = centre ? s.source_x : s.x;
    sy[lane] = centre ? s.source_y : s.y;
    sz[lane] = centre ? s.source_z : s.z;
    snx[lane] = panel_source_nx(s);
    sny[lane] = panel_source_ny(s);
    snz[lane] = panel_source_nz(s);
    area[lane] = s.area;
    phi[lane] = unknowns[source[lane]];
    phin[lane] = coupled ? unknowns[n + source[lane]] : 0.0;
  }
  return plain_centroid_quad_compute_avx2(target, sx, sy, sz, snx, sny, snz,
                                          area, phi, phin, eps_ratio, kk,
                                          coupled, acc_phi, acc_phin);
}

// Four sources read from a NearSourceCache; consecutive indices are used in
// place, others gathered.
TITAN_TARGET_AVX2
bool plain_centroid_quad_cached_avx2(const Panel &target,
                                     const NearSourceCache &cache,
                                     const std::size_t (&source)[4],
                                     const std::vector<double> &unknowns,
                                     std::size_t n, double eps_ratio,
                                     double kk, bool coupled, double *acc_phi,
                                     double *acc_phin) noexcept {
  const std::size_t j = source[0];
  const double *phin_base = coupled ? unknowns.data() + n : nullptr;
  if (source[1] == j + 1 && source[2] == j + 2 && source[3] == j + 3) {
    return plain_centroid_quad_compute_avx2(
        target, &cache.x[j], &cache.y[j], &cache.z[j], &cache.nx[j],
        &cache.ny[j], &cache.nz[j], &cache.area[j], &unknowns[j],
        coupled ? phin_base + j : nullptr, eps_ratio, kk, coupled, acc_phi,
        acc_phin);
  }
  alignas(32) double sx[4], sy[4], sz[4], snx[4], sny[4], snz[4], area[4],
      phi[4], phin[4];
  for (int lane = 0; lane < 4; ++lane) {
    const std::size_t k = source[lane];
    sx[lane] = cache.x[k];
    sy[lane] = cache.y[k];
    sz[lane] = cache.z[k];
    snx[lane] = cache.nx[k];
    sny[lane] = cache.ny[k];
    snz[lane] = cache.nz[k];
    area[lane] = cache.area[k];
    phi[lane] = unknowns[k];
    phin[lane] = coupled ? phin_base[k] : 0.0;
  }
  return plain_centroid_quad_compute_avx2(target, sx, sy, sz, snx, sny, snz,
                                          area, phi, phin, eps_ratio, kk,
                                          coupled, acc_phi, acc_phin);
}

// Lanes of sources j..j+3 that are the target itself or share a vertex with
// it (bit l set for lane l). Target indices equal to -1 are skipped, so a
// panel without triangle geometry never matches another one.
TITAN_TARGET_AVX2
int special_quad_mask(const NearSourceCache &cache, std::size_t j,
                      std::size_t self, const std::size_t (&target)[3]) noexcept {
  const __m256i index = _mm256_add_epi64(
      _mm256_set1_epi64x(static_cast<long long>(j)), _mm256_set_epi64x(3, 2, 1, 0));
  __m256i special = _mm256_cmpeq_epi64(
      index, _mm256_set1_epi64x(static_cast<long long>(self)));
  const __m256i a = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cache.v0.data() + j));
  const __m256i b = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cache.v1.data() + j));
  const __m256i c = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>(cache.v2.data() + j));
  for (const std::size_t t : target) {
    if (t == static_cast<std::size_t>(-1))
      continue;
    const __m256i tv = _mm256_set1_epi64x(static_cast<long long>(t));
    special = _mm256_or_si256(
        special, _mm256_or_si256(_mm256_cmpeq_epi64(a, tv),
                                 _mm256_or_si256(_mm256_cmpeq_epi64(b, tv),
                                                 _mm256_cmpeq_epi64(c, tv))));
  }
  return _mm256_movemask_pd(_mm256_castsi256_pd(special));
}

} // namespace
#endif

NearSourceCache build_near_source_cache(const std::vector<Panel> &panels) {
  NearSourceCache cache;
  const std::size_t n = panels.size();
  for (auto *field : {&cache.x, &cache.y, &cache.z, &cache.nx, &cache.ny,
                      &cache.nz, &cache.area})
    field->resize(n);
  cache.v0.resize(n);
  cache.v1.resize(n);
  cache.v2.resize(n);
  constexpr std::size_t none = static_cast<std::size_t>(-1);
  for (std::size_t i = 0; i < n; ++i) {
    const Panel &p = panels[i];
    const bool centre = p.has_source_quadrature_center;
    cache.x[i] = centre ? p.source_x : p.x;
    cache.y[i] = centre ? p.source_y : p.y;
    cache.z[i] = centre ? p.source_z : p.z;
    cache.nx[i] = panel_source_nx(p);
    cache.ny[i] = panel_source_ny(p);
    cache.nz[i] = panel_source_nz(p);
    cache.area[i] = p.area;
    cache.v0[i] = p.has_triangle_geometry ? p.v0_index : none;
    cache.v1[i] = p.has_triangle_geometry ? p.v1_index : none;
    cache.v2[i] = p.has_triangle_geometry ? p.v2_index : none;
  }
  return cache;
}

JcpNearAccum accumulate_jcp_near_ordered(
    const Panel &target, std::size_t target_index,
    const std::vector<Panel> &panels, const std::vector<double> &unknowns,
    double eps_ratio, double kappa, bool coupled,
    const std::vector<std::size_t> &order, std::size_t begin,
    std::size_t count, const std::string &near_integration,
    const std::string &quad_buffer,
    TreecodeStats *stats, const NearSourceCache *sources) {
  JcpNearAccum acc;
  const std::size_t n = panels.size();
  const std::size_t end = begin + count;
  if(curved_dual_cell(target)) {
    const auto&cache=*target.node_patch_geometry->curved;
    if(!cache.pair_blocks.empty() && n==cache.cells.size() &&
       target.node_patch_index==target_index) {
      if(cache.eps_ratio!=eps_ratio||cache.kappa!=kappa)
        throw std::runtime_error("curved_dual_kernel_cache_parameter_mismatch");
      const auto*row=cache.pair_blocks.data()+target_index*n;
      for(std::size_t pos=begin;pos<end;++pos){const auto j=order[pos];
        if(panels[j].node_patch_geometry.get()!=target.node_patch_geometry.get() || panels[j].node_patch_index>=n)
          throw std::runtime_error("curved_dual_mixed_cache_geometry");
        const auto&k=row[panels[j].node_patch_index];
        acc.phi+=k[1]*unknowns[j];
        if(coupled){acc.phi+=k[0]*unknowns[n+j];acc.phin+=k[2]*unknowns[n+j]+k[3]*unknowns[j];}
        if(stats)add_stats(stats,j==target_index,j!=target_index&&node_patch_adjacent(target,panels[j]),true,false,false,1);
      }
      if(stats){stats->near_integration=near_integration;stats->quad_buffer=quad_buffer;stats->near_kernel_policy="curved_dual";}
      return acc;
    }
  }
  // Most pairs of a row are well separated and end in the plain centroid rule
  // of evaluate_jcp_near_pair_kernel(): the final else when coupled (or its
  // "centroid" mode branch), the kappa = 0 centroid when not.  Those pairs are
  // recognised with the same tests, which depend on strings and on the target
  // only through values hoisted here, and counted in bulk.  With AVX2 they
  // are evaluated four at a time into per-lane sums; otherwise, and for the
  // tail of a row, by the shared sqrt/exp scalar kernel.  Every other pair,
  // and any pair whose kernel is non-finite, takes the full dispatcher.
  const bool centroid_mode = near_integration == "centroid";
  const bool buffer_off = quad_buffer == "off" || quad_buffer.empty() ||
                          !target.has_triangle_geometry;
  const bool target_node_patch = is_node_patch_panel(target);
  const double plain_kappa = coupled ? kappa : 0.0;
  std::size_t plain_pairs = 0;
  const auto add_dispatched = [&](std::size_t j) {
    const JcpNearPairKernel selected = evaluate_jcp_near_pair_kernel(
        target, target_index, panels[j], j, eps_ratio, kappa, coupled,
        near_integration, quad_buffer, stats);
    if (!selected.valid)
      return;
    KernelValues kernel;
    kernel.k1 = selected.k1;
    kernel.k2 = selected.k2;
    kernel.k3 = selected.k3;
    kernel.k4 = selected.k4;
    apply_kernel(acc, kernel, unknowns, n, j, coupled);
  };
  const auto add_plain_scalar = [&](std::size_t j) {
    KernelValues kernel;
    if (centroid_integral_shared(target, panels[j], eps_ratio, plain_kappa,
                                 kernel) &&
        finite_kernel(kernel)) {
      apply_kernel(acc, kernel, unknowns, n, j, coupled);
      ++plain_pairs;
      return;
    }
    add_dispatched(j);
  };
#if TITAN_SIMD_X86
  // A target with a non-finite position or normal stays on the scalar path,
  // where the uncoupled rows still evaluate (and reject) its normal terms.
  const bool vector_plain =
      titan_simd_dispatch::runtime_has_avx2() && std::isfinite(target.x) &&
      std::isfinite(target.y) && std::isfinite(target.z) &&
      std::isfinite(target.nx) && std::isfinite(target.ny) &&
      std::isfinite(target.nz);
  alignas(32) double lane_phi[4] = {0.0, 0.0, 0.0, 0.0};
  alignas(32) double lane_phin[4] = {0.0, 0.0, 0.0, 0.0};
  std::size_t pending[4];
  int pending_count = 0;
  // With a source cache, a target that is not a node patch and a row without
  // an active geometric buffer, the only special pairs are the self pair and
  // one-ring pairs sharing a vertex; both are recognised from the cached
  // vertex indices, and plain pairs read the cached fields.
  if (vector_plain && sources != nullptr && sources->x.size() == n &&
      !target_node_patch && (buffer_off || centroid_mode || !coupled)) {
    constexpr std::size_t none = static_cast<std::size_t>(-1);
    const std::size_t t0 = target.has_triangle_geometry ? target.v0_index : none;
    const std::size_t t1 = target.has_triangle_geometry ? target.v1_index : none;
    const std::size_t t2 = target.has_triangle_geometry ? target.v2_index : none;
    const std::size_t *v0 = sources->v0.data();
    const std::size_t *v1 = sources->v1.data();
    const std::size_t *v2 = sources->v2.data();
    const auto shares = [&](std::size_t t, std::size_t j) {
      return t != none && (t == v0[j] || t == v1[j] || t == v2[j]);
    };
    const double kk = std::max(0.0, plain_kappa);
    const std::size_t target_vertices[3] = {t0, t1, t2};
    for (std::size_t pos = begin; pos < end; ++pos) {
      const std::size_t j = order[pos];
      // Four consecutive plain sources go straight to the kernel.
      if (pending_count == 0 && pos + 4 <= end && order[pos + 1] == j + 1 &&
          order[pos + 2] == j + 2 && order[pos + 3] == j + 3 &&
          special_quad_mask(*sources, j, target_index, target_vertices) == 0) {
        const std::size_t quad[4] = {j, j + 1, j + 2, j + 3};
        if (plain_centroid_quad_cached_avx2(target, *sources, quad, unknowns,
                                            n, eps_ratio, kk, coupled,
                                            lane_phi, lane_phin))
          plain_pairs += 4;
        else
          for (const std::size_t p : quad)
            add_plain_scalar(p);
        pos += 3;
        continue;
      }
      if (j == target_index || shares(t0, j) || shares(t1, j) ||
          shares(t2, j)) {
        add_dispatched(j);
        continue;
      }
      pending[pending_count++] = j;
      if (pending_count == 4) {
        pending_count = 0;
        if (plain_centroid_quad_cached_avx2(target, *sources, pending,
                                            unknowns, n, eps_ratio, kk,
                                            coupled, lane_phi, lane_phin))
          plain_pairs += 4;
        else
          for (const std::size_t p : pending)
            add_plain_scalar(p);
      }
    }
  } else
#endif
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t j = order[pos];
    const Panel &source = panels[j];
    if (j != target_index) {
      const bool node_patch_pair =
          target_node_patch && is_node_patch_panel(source);
      const bool one_ring = node_patch_pair
                                ? node_patch_adjacent(target, source)
                                : shared_vertex_count(target, source) > 0;
      // A curved pair outside the FMM near zone takes the same one-point
      // kernel as any plain pair (evaluate_jcp_near_pair_kernel).
      bool plain = !one_ring && !(node_patch_pair && curved_dual_cell(source) &&
                                  !curved_pair_takes_one_point(target, source));
      if (plain && coupled && !centroid_mode && !buffer_off &&
          source.has_triangle_geometry)
        plain = !classify_geometric_near_buffer_pair(target, source,
                                                     quad_buffer)
                     .active;
      if (plain) {
#if TITAN_SIMD_X86
        if (vector_plain) {
          pending[pending_count++] = j;
          if (pending_count == 4) {
            pending_count = 0;
            if (plain_centroid_quad_avx2(target, panels, pending, unknowns, n,
                                         eps_ratio, std::max(0.0, plain_kappa),
                                         coupled, lane_phi, lane_phin))
              plain_pairs += 4;
            else
              for (const std::size_t p : pending)
                add_plain_scalar(p);
          }
          continue;
        }
#endif
        add_plain_scalar(j);
        continue;
      }
    }
    add_dispatched(j);
  }
#if TITAN_SIMD_X86
  for (int p = 0; p < pending_count; ++p)
    add_plain_scalar(pending[p]);
  acc.phi += (lane_phi[0] + lane_phi[1]) + (lane_phi[2] + lane_phi[3]);
  acc.phin += (lane_phin[0] + lane_phin[1]) + (lane_phin[2] + lane_phin[3]);
#endif
  if (stats != nullptr)
    stats->near_centroid_pairs += plain_pairs;
  if (stats != nullptr) {
    stats->near_integration = near_integration;
    stats->quad_buffer = quad_buffer;
    const char *policy = curved_dual_cell(target) ? "curved_dual" : coupled ? kernel_policy_for_mode(near_integration)
                                 : poisson_policy(target, near_integration);
    if (stats->near_kernel_policy != policy)
      stats->near_kernel_policy = policy;
  }
  return acc;
}

} // namespace TitanPBE::Tabi

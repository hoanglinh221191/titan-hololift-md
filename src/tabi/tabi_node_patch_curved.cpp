#include "tabi_node_patch_curved.h"
#include "tabi_pn_shared_separation.h"
#include "tabi_shared_vertex_separation.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "../titan_executor.h"
#include "../titan_avx2_math.h"
#include "../titan_simd_dispatch.h"
#include <algorithm>
#include <cstring>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
namespace TitanPBE::Tabi {
namespace {
using V = std::array<double, 3>;
using B = std::array<double, 4>;
constexpr double pi = std::numbers::pi;
V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V mul(V a, double b) { return {a[0] * b, a[1] * b, a[2] * b}; }
double dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double norm(V a) { return std::sqrt(dot(a, a)); }
V cross(V a, V b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
std::vector<std::array<double, 2>> gauss(int n) {
  std::vector<std::array<double, 2>> rule;
  for (int i = 0; i < n; ++i) {
    double z = std::cos(pi * (i + .75) / (n + .5)), d = 0;
    for (int it = 0; it < 64; ++it) {
      double p = 1, prev = 0;
      for (int k = 1; k <= n; ++k) {
        double old = p;
        p = ((2 * k - 1) * z * p - (k - 1) * prev) / k;
        prev = old;
      }
      d = n * (z * p - prev) / (z * z - 1);
      double step = p / d;
      z -= step;
      if (std::abs(step) < 2e-15)
        break;
    }
    rule.push_back({(z + 1) / 2, 1 / ((1 - z * z) * d * d)});
  }
  return rule;
}
// The rule of each order is built once: the adaptive near quadrature asks for
// orders up to 256 on every near pair.
const std::vector<std::array<double, 2>> &gauss_rule(int n) {
  switch (n) {
  case 2: { static const auto rule = gauss(2); return rule; }
  case 3: { static const auto rule = gauss(3); return rule; }
  case 4: { static const auto rule = gauss(4); return rule; }
  case 8: { static const auto rule = gauss(8); return rule; }
  case 16: { static const auto rule = gauss(16); return rule; }
  case 32: { static const auto rule = gauss(32); return rule; }
  case 64: { static const auto rule = gauss(64); return rule; }
  case 128: { static const auto rule = gauss(128); return rule; }
  case 256: { static const auto rule = gauss(256); return rule; }
  default:
    throw std::invalid_argument("curved_dual_unsupported_gauss_order");
  }
}
// A dual cell takes two quads from every incident triangle (half 0: vertex,
// edge midpoint ab, centroid; half 1: vertex, centroid, edge midpoint ac).
// A quad is integrated over a rectangle [u0,u1] x [v0,v1] of its unit square
// as an order x order Gauss product; the unit square gives the stored samples
// bit for bit, and smaller rectangles serve the adaptive subdivision of a
// near pair.
struct QuadRect {
  double u0{0}, u1{1}, v0{0}, v1{1};
};
// One quad of the dual cell of node in incident triangle pos, mapped radially
// onto the triangle's parent sphere; f receives each sample.
template <typename F>
void visit_sphere_quad(const std::vector<std::array<double, 4>> &spheres,
                       const NodePatchGeometry &g, std::size_t node,
                       std::size_t pos, int half, const QuadRect &rect,
                       int order, F &&f) {
  const auto &rule = gauss_rule(order);
  const auto ti = g.incident_triangles[pos];
  const auto &t = g.triangles[ti];
  const auto &s = spheres[ti];
  V origin{s[0], s[1], s[2]};
  int local = 0;
  while (t.vertices[local] != node)
    ++local;
  V a = sub(g.vertex_xyz[node], origin),
    b = sub(g.vertex_xyz[t.vertices[(local + 1) % 3]], origin),
    c = sub(g.vertex_xyz[t.vertices[(local + 2) % 3]], origin);
  V middle = mul(add(add(a, b), c), 1. / 3);
  const std::array<V, 2> edge =
      half == 0 ? std::array<V, 2>{mul(add(a, b), .5), middle}
                : std::array<V, 2>{middle, mul(add(a, c), .5)};
  const double su = rect.u1 - rect.u0, sv = rect.v1 - rect.v0;
  for (auto ru : rule)
    for (auto rv : rule) {
      const double u = rect.u0 + su * ru[0], v = rect.v0 + sv * rv[0];
      V du = sub(add(mul(edge[0], 1 - v), mul(edge[1], v)), a),
        dv = mul(sub(edge[1], edge[0]), u), p = add(a, mul(du, u));
      double len = norm(p);
      if (!(len > 0))
        throw std::runtime_error("curved_dual_radial_map_singular");
      V normal = mul(p, 1 / len);
      double w = (ru[1] * su) * (rv[1] * sv) * s[3] * s[3] *
                 std::abs(dot(p, cross(du, dv))) / (len * len * len);
      if (!(w > 0) || !std::isfinite(w))
        throw std::runtime_error("curved_dual_invalid_weight");
      f(CurvedDualSample{add(origin, mul(normal, s[3])), normal, w});
    }
}
template <typename F>
void visit_sphere_samples(const std::vector<std::array<double, 4>> &spheres,
                          const NodePatchGeometry &g, std::size_t node,
                          int order, F &&f) {
  for (auto pos = g.incident_offsets[node]; pos < g.incident_offsets[node + 1];
       ++pos)
    for (int half = 0; half < 2; ++half)
      visit_sphere_quad(spheres, g, node, pos, half, QuadRect{}, order, f);
}
std::vector<CurvedDualSample>
samples(const std::vector<std::array<double, 4>> &spheres,
        const NodePatchGeometry &g, std::size_t node, int order) {
  std::vector<CurvedDualSample> out;
  visit_sphere_samples(spheres, g, node, order,
                       [&](const CurvedDualSample &q) { out.push_back(q); });
  return out;
}
// Curved PN triangle (Vlachos, Peters, Boyd, Mitchell 2001): the cubic Bezier
// patch that interpolates three vertex positions and is tangent to their
// normals.  The edge control points are the flat edge thirds projected onto
// the tangent planes at their nearer vertex; the centre point is pulled off
// the flat centroid by half the edge-point offset.
struct PnPatch {
  std::array<V, 10> b; // 300 030 003 210 120 021 012 102 201 111
};
// Bit 1, 2, 4 of straight make the edge p1p2, p2p3, p3p1 the straight
// segment.  Both control points of an edge depend on that edge alone, so two
// patches that agree on an edge's ends, normals and bit share the edge curve;
// with all three bits set the patch is the flat triangle.
PnPatch pn_patch(const V &p1, const V &p2, const V &p3, const V &n1,
                 const V &n2, const V &n3, unsigned straight = 0) {
  auto edge = [](const V &pa, const V &pb, const V &na, bool flat) {
    // (2 pa + pb - w na) / 3 with w = (pb - pa).na projects the flat third
    // point onto the tangent plane at pa; a straight edge keeps it (w = 0).
    const double w = flat ? 0.0 : dot(sub(pb, pa), na);
    return mul(sub(add(mul(pa, 2.0), pb), mul(na, w)), 1.0 / 3.0);
  };
  const bool s12 = straight & 1U, s23 = straight & 2U, s31 = straight & 4U;
  PnPatch t;
  t.b[0] = p1; t.b[1] = p2; t.b[2] = p3;
  t.b[3] = edge(p1, p2, n1, s12); t.b[4] = edge(p2, p1, n2, s12);
  t.b[5] = edge(p2, p3, n2, s23); t.b[6] = edge(p3, p2, n3, s23);
  t.b[7] = edge(p3, p1, n3, s31); t.b[8] = edge(p1, p3, n1, s31);
  V e{};
  for (int k = 3; k < 9; ++k) e = add(e, t.b[k]);
  e = mul(e, 1.0 / 6.0);
  const V v = mul(add(add(p1, p2), p3), 1.0 / 3.0);
  t.b[9] = add(e, mul(sub(e, v), 0.5));
  return t;
}
// Position and the three barycentric partials of the cubic patch.
void pn_evaluate(const PnPatch &t, double l1, double l2, double l3, V &pos,
                 V &d1, V &d2, V &d3) {
  const auto &b = t.b;
  // Bernstein cubic: sum b_ijk * 3!/(i!j!k!) l1^i l2^j l3^k.
  pos = add(add(add(mul(b[0], l1 * l1 * l1), mul(b[1], l2 * l2 * l2)),
                add(mul(b[2], l3 * l3 * l3), mul(b[3], 3 * l1 * l1 * l2))),
            add(add(add(mul(b[4], 3 * l1 * l2 * l2), mul(b[5], 3 * l2 * l2 * l3)),
                    add(mul(b[6], 3 * l2 * l3 * l3), mul(b[7], 3 * l1 * l3 * l3))),
                add(mul(b[8], 3 * l1 * l1 * l3), mul(b[9], 6 * l1 * l2 * l3))));
  d1 = add(add(add(mul(b[0], 3 * l1 * l1), mul(b[3], 6 * l1 * l2)),
               add(mul(b[4], 3 * l2 * l2), mul(b[7], 3 * l3 * l3))),
           add(mul(b[8], 6 * l1 * l3), mul(b[9], 6 * l2 * l3)));
  d2 = add(add(add(mul(b[1], 3 * l2 * l2), mul(b[3], 3 * l1 * l1)),
               add(mul(b[4], 6 * l1 * l2), mul(b[5], 6 * l2 * l3))),
           add(mul(b[6], 3 * l3 * l3), mul(b[9], 6 * l1 * l3)));
  d3 = add(add(add(mul(b[2], 3 * l3 * l3), mul(b[5], 3 * l2 * l2)),
               add(mul(b[6], 6 * l2 * l3), mul(b[7], 6 * l1 * l3))),
           add(mul(b[8], 3 * l1 * l1), mul(b[9], 6 * l1 * l2)));
}
// Same dual-cell decomposition and Gauss layout as samples(), but the piece is
// mapped onto the PN patch of its parent triangle instead of onto an exact
// sphere.  The area element is |dP/du x dP/dv| through the chain rule, and the
// normal is the patch's own geometric normal, oriented by the flat triangle.
// A PN patch whose geometric normal turns more than acos(kPnFlatCosine) away
// from its flat triangle anywhere on a 6x6 barycentric lattice, or whose node
// normals do, is folded or nearly so.  Its samples would count the fold twice
// with reversed normals, which gives the operator local junk that GMRES
// cannot get past.  Such a triangle is flat.  Its edges are then straight in
// the neighbouring patches too: a neighbour that kept its curved edge would
// leave a sliver of open surface along it (0.07 A at the midpoint of a 1 A
// edge in a steep case).  A straightened edge can in turn fold a neighbour,
// so the rule is applied until no further triangle turns flat.  The decision
// is per edge, so both triangles of an edge and every dual cell built from
// them agree.
constexpr double kPnFlatCosine = 0.3;
// The PN patch of triangle ti in its own vertex order.
PnPatch pn_triangle_patch(const std::vector<std::array<double, 3>> &normals,
                          const NodePatchGeometry &g, std::size_t ti,
                          unsigned straight) {
  const auto &t = g.triangles[ti];
  std::array<V, 3> p, n;
  for (int k = 0; k < 3; ++k) {
    const auto v = t.vertices[k];
    p[k] = {g.vertex_xyz[v][0], g.vertex_xyz[v][1], g.vertex_xyz[v][2]};
    n[k] = normals[v];
  }
  return pn_patch(p[0], p[1], p[2], n[0], n[1], n[2], straight);
}
// Bernstein-Bezier polynomials on a triangle.  The coefficient (i, j, n-i-j)
// of degree n sits at bernstein_index(n, i, j): rows by the first index i,
// from n down to 0.
constexpr int bernstein_index(int n, int i, int j) {
  int before = 0;
  for (int a = n; a > i; --a)
    before += n - a + 1;
  return before + j;
}
double multinomial(int n, int i, int j) {
  const double factorial[] = {1, 1, 2, 6, 24};
  return factorial[n] / (factorial[i] * factorial[j] * factorial[n - i - j]);
}
// Control point b_{i, j, 3-i-j} of a PN patch.
const V &cubic_point(const PnPatch &p, int i, int j) {
  static constexpr int slot[4][4] = {
      {2, 6, 5, 1}, {7, 9, 4, -1}, {8, 3, -1, -1}, {0, -1, -1, -1}};
  return p.b[slot[i][j]];
}
// The normal component of the patch Jacobian, winding * (dP/ds x dP/dt).face
// with s = l1, t = l2, as a quartic in Bernstein form.  dP/ds and dP/dt are
// quadratics with coefficients 3 (b_{beta+e1} - b_{beta+e3}) and
// 3 (b_{beta+e2} - b_{beta+e3}); the product of Bernstein polynomials of
// degrees 2 and 2 is the quartic with weights C(beta) C(gamma) / C(delta).
using Quartic = std::array<double, 15>;
Quartic pn_jacobian_normal(const PnPatch &p, const V &face, double winding) {
  std::array<V, 6> ds, dt;
  for (int i = 0; i <= 2; ++i)
    for (int j = 0; i + j <= 2; ++j) {
      const int q = bernstein_index(2, i, j);
      const V &e3 = cubic_point(p, i, j);
      ds[q] = mul(sub(cubic_point(p, i + 1, j), e3), 3.0);
      dt[q] = mul(sub(cubic_point(p, i, j + 1), e3), 3.0);
    }
  Quartic f{};
  for (int i1 = 0; i1 <= 2; ++i1)
    for (int j1 = 0; i1 + j1 <= 2; ++j1)
      for (int i2 = 0; i2 <= 2; ++i2)
        for (int j2 = 0; i2 + j2 <= 2; ++j2) {
          const double weight = multinomial(2, i1, j1) * multinomial(2, i2, j2) /
                                multinomial(4, i1 + i2, j1 + j2);
          f[bernstein_index(4, i1 + i2, j1 + j2)] +=
              weight * winding *
              dot(cross(ds[bernstein_index(2, i1, j1)],
                        dt[bernstein_index(2, i2, j2)]),
                  face);
        }
  return f;
}
using Bary = std::array<double, 3>;
// The blossom (symmetric multi-affine form) of a quartic at four points, by
// de Casteljau steps with a different point each.
double blossom(const Quartic &c, const std::array<Bary, 4> &u) {
  Quartic cur = c, next{};
  for (int n = 4, r = 0; r < 4; ++r, --n) {
    for (int i = 0; i <= n - 1; ++i)
      for (int j = 0; i + j <= n - 1; ++j)
        next[bernstein_index(n - 1, i, j)] =
            u[r][0] * cur[bernstein_index(n, i + 1, j)] +
            u[r][1] * cur[bernstein_index(n, i, j + 1)] +
            u[r][2] * cur[bernstein_index(n, i, j)];
    cur = next;
  }
  return cur[0];
}
// True when the quartic f is positive on the whole sub-triangle a, b, c (in
// the barycentric coordinates of the patch).  Positive Bernstein coefficients
// on the sub-triangle prove it (the polynomial is a convex combination of
// them); a non-positive corner value disproves it; otherwise the sub-triangle
// is split into four, down to depth levels.  Undecided at the last level
// counts as not proven.
bool quartic_positive(const Quartic &f, const Bary &a, const Bary &b,
                      const Bary &c, int depth) {
  double lowest = std::numeric_limits<double>::infinity();
  Quartic local{};
  for (int i = 0; i <= 4; ++i)
    for (int j = 0; i + j <= 4; ++j) {
      std::array<Bary, 4> u;
      for (int r = 0; r < 4; ++r)
        u[r] = r < i ? a : r < i + j ? b : c;
      local[bernstein_index(4, i, j)] = blossom(f, u);
      lowest = std::min(lowest, local[bernstein_index(4, i, j)]);
    }
  if (lowest > 0)
    return true;
  if (!(local[bernstein_index(4, 4, 0)] > 0) ||
      !(local[bernstein_index(4, 0, 4)] > 0) ||
      !(local[bernstein_index(4, 0, 0)] > 0) || depth == 0)
    return false;
  const auto mid = [](const Bary &x, const Bary &y) {
    return Bary{0.5 * (x[0] + y[0]), 0.5 * (x[1] + y[1]), 0.5 * (x[2] + y[2])};
  };
  const Bary ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
  return quartic_positive(f, a, ab, ca, depth - 1) &&
         quartic_positive(f, ab, b, bc, depth - 1) &&
         quartic_positive(f, ca, bc, c, depth - 1) &&
         quartic_positive(f, ab, bc, ca, depth - 1);
}
// Depth of the fold proof: 4^4 sub-triangles at most.
constexpr int kPnFoldProofDepth = 4;
bool pn_triangle_steep(const std::vector<std::array<double, 3>> &normals,
                       const NodePatchGeometry &g, std::size_t ti,
                       unsigned straight) {
  const auto &t = g.triangles[ti];
  const V face{t.nx, t.ny, t.nz};
  for (int k = 0; k < 3; ++k)
    if (dot(normals[t.vertices[k]], face) < kPnFlatCosine)
      return true;
  const auto &p0 = g.vertex_xyz[t.vertices[0]], &p1 = g.vertex_xyz[t.vertices[1]],
             &p2 = g.vertex_xyz[t.vertices[2]];
  const double winding =
      dot(cross(sub(p1, p0), sub(p2, p0)), face) < 0 ? -1.0 : 1.0;
  const PnPatch patch = pn_triangle_patch(normals, g, ti, straight);
  constexpr int kLattice = 6;
  for (int i = 0; i <= kLattice; ++i)
    for (int j = 0; i + j <= kLattice; ++j) {
      const double l1 = double(i) / kLattice, l2 = double(j) / kLattice;
      V P, d1, d2, d3;
      pn_evaluate(patch, l1, l2, 1 - l1 - l2, P, d1, d2, d3);
      const V jacobian = cross(sub(d1, d3), sub(d2, d3));
      const double length = norm(jacobian);
      if (!(length > 0) ||
          winding * dot(jacobian, face) < kPnFlatCosine * length)
        return true;
    }
  // The lattice sees the steepness at its nodes only; a fold can lie
  // between them.  The orientation is proven on the whole triangle instead.
  return !quartic_positive(pn_jacobian_normal(patch, face, winding),
                           {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, kPnFoldProofDepth);
}
// Distances between flat triangles (Ericson, Real-Time Collision Detection,
// 5.1.5 and 5.1.9), for the proof that two non-adjacent patches are apart.
V closest_on_triangle(const V &p, const V &a, const V &b, const V &c) {
  const V ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
  const double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return a;
  const V bp = sub(p, b);
  const double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return b;
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return add(a, mul(ab, d1 / (d1 - d3)));
  const V cp = sub(p, c);
  const double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return c;
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return add(a, mul(ac, d2 / (d2 - d6)));
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
  const double denom = 1 / (va + vb + vc);
  return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}
double segment_distance(const V &p1, const V &q1, const V &p2, const V &q2) {
  const V d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
  const double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  double s = 0, t = 0;
  if (a <= 1e-300 && e <= 1e-300)
    return norm(r);
  if (a <= 1e-300) {
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    const double c = dot(d1, r);
    if (e <= 1e-300) {
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      const double b = dot(d1, d2), denom = a * e - b * b;
      s = denom > 0 ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1) {
        t = 1;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  return norm(sub(add(p1, mul(d1, s)), add(p2, mul(d2, t))));
}
// Whether segment p q crosses triangle a b c (Moller-Trumbore).
bool segment_crosses_triangle(const V &p, const V &q, const V &a, const V &b,
                              const V &c) {
  const V d = sub(q, p), e1 = sub(b, a), e2 = sub(c, a), h = cross(d, e2);
  const double det = dot(e1, h);
  if (std::abs(det) < 1e-300)
    return false;
  const double inv = 1 / det;
  const V s = sub(p, a);
  const double u = inv * dot(s, h);
  if (u < 0 || u > 1)
    return false;
  const V qv = cross(s, e1);
  const double v = inv * dot(d, qv);
  if (v < 0 || u + v > 1)
    return false;
  const double t = inv * dot(e2, qv);
  return t >= 0 && t <= 1;
}
double triangle_distance(const std::array<V, 3> &x, const std::array<V, 3> &y) {
  for (int k = 0; k < 3; ++k)
    if (segment_crosses_triangle(x[k], x[(k + 1) % 3], y[0], y[1], y[2]) ||
        segment_crosses_triangle(y[k], y[(k + 1) % 3], x[0], x[1], x[2]))
      return 0.0;
  double best = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 3; ++k) {
    best = std::min(best, norm(sub(x[k], closest_on_triangle(x[k], y[0], y[1], y[2]))));
    best = std::min(best, norm(sub(y[k], closest_on_triangle(y[k], x[0], x[1], x[2]))));
    for (int m = 0; m < 3; ++m)
      best = std::min(best, segment_distance(x[k], x[(k + 1) % 3], y[m],
                                             y[(m + 1) % 3]));
  }
  return best;
}
std::array<V, 3> flat_triangle(const NodePatchGeometry &g, std::size_t ti) {
  const auto &t = g.triangles[ti];
  return {g.vertex_xyz[t.vertices[0]], g.vertex_xyz[t.vertices[1]],
          g.vertex_xyz[t.vertices[2]]};
}
// A patch lies in the convex hull of its control points, and the distance to
// the flat triangle is convex, so every point of the patch is within the
// largest control-point distance of the triangle.
double pn_deviation(const PnPatch &patch, const std::array<V, 3> &flat) {
  double deviation = 0.0;
  for (const V &b : patch.b)
    deviation = std::max(
        deviation, norm(sub(b, closest_on_triangle(b, flat[0], flat[1], flat[2]))));
  return deviation;
}
// A piece of a PN patch as a cubic Bezier triangle of its own: the control
// net (bernstein_index(3, i, j)), the flat triangle of its corners and the
// largest distance of the net from it, which bounds the piece as above.
using CubicNet = std::array<V, 10>;
struct PatchPiece {
  CubicNet net;
  std::array<V, 3> flat;
  double deviation{0.0}, size{0.0};
};
PatchPiece make_piece(const CubicNet &net) {
  PatchPiece p{net, {net[bernstein_index(3, 3, 0)], net[bernstein_index(3, 0, 3)],
                     net[bernstein_index(3, 0, 0)]}};
  for (const V &b : net)
    p.deviation = std::max(
        p.deviation, norm(sub(b, closest_on_triangle(b, p.flat[0], p.flat[1], p.flat[2]))));
  for (int k = 0; k < 3; ++k)
    p.size = std::max(p.size, norm(sub(p.flat[k], p.flat[(k + 1) % 3])));
  return p;
}
PatchPiece patch_piece(const PnPatch &patch) {
  CubicNet net;
  for (int i = 0; i <= 3; ++i)
    for (int j = 0; i + j <= 3; ++j)
      net[bernstein_index(3, i, j)] = cubic_point(patch, i, j);
  return make_piece(net);
}
// The control net of the sub-triangle a, b, c of a cubic Bezier triangle: its
// coefficient (i, j, k) is the blossom at i copies of a, j of b, k of c.
CubicNet sub_net(const CubicNet &c, const Bary &a, const Bary &b, const Bary &d) {
  CubicNet out;
  for (int i = 0; i <= 3; ++i)
    for (int j = 0; i + j <= 3; ++j) {
      CubicNet cur = c, next{};
      for (int n = 3, r = 0; r < 3; ++r, --n) {
        const Bary &u = r < i ? a : r < i + j ? b : d;
        for (int p = 0; p <= n - 1; ++p)
          for (int q = 0; p + q <= n - 1; ++q)
            next[bernstein_index(n - 1, p, q)] =
                add(add(mul(cur[bernstein_index(n, p + 1, q)], u[0]),
                        mul(cur[bernstein_index(n, p, q + 1)], u[1])),
                    mul(cur[bernstein_index(n, p, q)], u[2]));
        cur = next;
      }
      out[bernstein_index(3, i, j)] = cur[0];
    }
  return out;
}
// Whether two pieces are proven apart: their flat triangles farther apart
// than the sum of their deviations, or else, splitting the larger piece into
// four (the deviation falls with the square of the size, the distance of two
// pieces of one smooth sheet does not), every quarter proven apart from the
// other piece, down to depth splits.
bool pieces_apart(const PatchPiece &x, const PatchPiece &y, int depth) {
  double scale = 1.0;
  for (const auto &tri : {x.flat, y.flat})
    for (const auto &p : tri)
      scale = std::max(scale, norm(sub(p, x.flat[0])));
  const double pad = 256 * std::numeric_limits<double>::epsilon() * scale;
  if (triangle_distance(x.flat, y.flat) > x.deviation + y.deviation + pad)
    return true;
  if (depth == 0)
    return false;
  const bool split_x = x.size >= y.size;
  const PatchPiece &big = split_x ? x : y, &other = split_x ? y : x;
  const Bary e1{1, 0, 0}, e2{0, 1, 0}, e3{0, 0, 1}, m12{.5, .5, 0},
      m23{0, .5, .5}, m31{.5, 0, .5};
  for (const auto &corners : std::array<std::array<Bary, 3>, 4>{
           {{e1, m12, m31}, {m12, e2, m23}, {m31, m23, e3}, {m12, m23, m31}}})
    if (!pieces_apart(make_piece(sub_net(big.net, corners[0], corners[1], corners[2])),
                      other, depth - 1))
      return false;
  return true;
}
// Splits of the pair proof: two sheets 1e-3 A apart need pieces about
// 0.05 A across, four splits of each patch of a 1 A mesh.
constexpr int kPnApartProofDepth = 8;
// Adjacent patches must meet only on their named common vertex/edge. The
// projected-union test proves injectivity and boundary separation together;
// subdivision retains exact logical boundary identities, with no excluded
// band around the common feature. Inconclusive pairs are repaired or refused.
bool pn_adjacent_patches_apart(const PnPatch &a,
                               const std::array<std::size_t, 3> &a_ids,
                               const PnPatch &b,
                               const std::array<std::size_t, 3> &b_ids,
                               bool both_flat) {
  pn_check::Piece pa{a.b}, pb{b.b};
  std::vector<std::size_t> ids, shared;
  for (const auto &vertices : {a_ids, b_ids}) {
    for (int i = 0; i < 3; ++i)
      for (int j = i + 1; j < 3; ++j)
        if (vertices[i] == vertices[j])
          return false;
    for (auto id : vertices)
      if (std::find(ids.begin(), ids.end(), id) == ids.end())
        ids.push_back(id);
  }
  for (auto id : a_ids)
    if (std::find(b_ids.begin(), b_ids.end(), id) != b_ids.end())
      shared.push_back(id);
  if (shared.empty() || shared.size() > 2)
    return false;
  for (int k = 0; k < 3; ++k) {
    pa.key[k][std::find(ids.begin(), ids.end(), a_ids[k]) - ids.begin()] = 1;
    pb.key[k][std::find(ids.begin(), ids.end(), b_ids[k]) - ids.begin()] = 1;
  }
  const auto index = [](const auto &vertices, std::size_t id) {
    return int(std::find(vertices.begin(), vertices.end(), id) - vertices.begin());
  };
  for (auto id : shared)
    if (a.b[index(a_ids, id)] != b.b[index(b_ids, id)])
      return false;
  if (shared.size() == 2) {
    const int ai = index(a_ids, shared[0]), aj = index(a_ids, shared[1]);
    const int bi = index(b_ids, shared[0]), bj = index(b_ids, shared[1]);
    if (((ai + 1) % 3 == aj) == ((bi + 1) % 3 == bj) ||
        pn_check::edge(pa, ai, aj).p != pn_check::edge(pb, bi, bj).p)
      return false;
  }
  // Flat non-coplanar neighbours have an inexpensive interval certificate.
  // Coplanar/unresolved cases still go through the projected-union proof.
  if (both_flat) {
    struct Point { double x, y, z; };
    const auto point = [](const V &v) { return Point{v[0], v[1], v[2]}; };
    const V origin = a.b[index(a_ids, shared[0])];
    if (shared.size() == 1) {
      std::vector<V> av, bv;
      for (int k = 0; k < 3; ++k) {
        if (a_ids[k] != shared[0]) av.push_back(a.b[k]);
        if (b_ids[k] != shared[0]) bv.push_back(b.b[k]);
      }
      if (detail::certified_shared_vertex_separation(
              point(origin), point(av[0]), point(av[1]), point(bv[0]), point(bv[1])))
        return true;
    } else {
      const V end = a.b[index(a_ids, shared[1])];
      V av{}, bv{};
      for (int k = 0; k < 3; ++k) {
        if (a_ids[k] != shared[0] && a_ids[k] != shared[1]) av = a.b[k];
        if (b_ids[k] != shared[0] && b_ids[k] != shared[1]) bv = b.b[k];
      }
      using namespace detail::shared_vertex_interval;
      if (determinant_sign(difference(point(end), point(origin)),
                           difference(point(av), point(origin)),
                           difference(point(bv), point(origin))) != 0)
        return true;
    }
  }
  const V origin = pa.p[0];
  double scale = 0;
  for (const auto &net : {a.b, b.b})
    for (const auto &p : net) {
      for (double v : p)
        if (!std::isfinite(v)) return false;
      scale = std::max(scale, norm(sub(p, origin)));
    }
  if (!(scale > 0) || !std::isfinite(scale)) return false;
  for (auto &p : pa.p) p = mul(sub(p, origin), 1 / scale);
  for (auto &p : pb.p) p = mul(sub(p, origin), 1 / scale);
  const auto separate = [](const auto &x, const auto &y, bool recursive) {
    for (int axis = 0; axis < 3; ++axis) {
      double xmax = -INFINITY, xmin = INFINITY, ymax = -INFINITY, ymin = INFINITY;
      for (const auto &p : x) { xmin = std::min(xmin, p[axis]); xmax = std::max(xmax, p[axis]); }
      for (const auto &p : y) { ymin = std::min(ymin, p[axis]); ymax = std::max(ymax, p[axis]); }
      constexpr double pad = 256 * std::numeric_limits<double>::epsilon();
      if (xmax + pad < ymin || ymax + pad < xmin) return true;
    }
    return recursive && pieces_apart(patch_piece(PnPatch{x}), patch_piece(PnPatch{y}),
                                      kPnApartProofDepth);
  };
  const auto certify = [](const auto &net, const V &normal) {
    Quartic f = pn_jacobian_normal(PnPatch{net}, normal, 1.0);
    double scale = 0;
    for (double v : f) scale = std::max(scale, std::abs(v));
    for (double &v : f)
      v -= 8192 * std::numeric_limits<double>::epsilon() * std::max(1.0, scale);
    return quartic_positive(f, {1,0,0}, {0,1,0}, {0,0,1}, kPnFoldProofDepth);
  };
  const auto normal = [](const auto &net, const auto &lambda) {
    V p, d1, d2, d3;
    pn_evaluate(PnPatch{net}, lambda[0], lambda[1], lambda[2], p, d1, d2, d3);
    return cross(sub(d1, d3), sub(d2, d3));
  };
  const auto subdivide = [](const auto &net) {
    const auto piece = patch_piece(PnPatch{net});
    std::array<pn_check::Net, 4> children;
    const std::array<std::array<Bary,3>,4> corners{{
        {{{1,0,0},{.5,.5,0},{.5,0,.5}}}, {{{.5,.5,0},{0,1,0},{0,.5,.5}}},
        {{{.5,0,.5},{0,.5,.5},{0,0,1}}}, {{{.5,.5,0},{0,.5,.5},{.5,0,.5}}}}};
    for (int k = 0; k < 4; ++k) {
      const auto sub = sub_net(piece.net, corners[k][0], corners[k][1], corners[k][2]);
      for (int i = 0; i <= 3; ++i)
        for (int j = 0; i + j <= 3; ++j) {
          static constexpr int slot[4][4] = {{2,6,5,1},{7,9,4,-1},{8,3,-1,-1},{0,-1,-1,-1}};
          children[k][slot[i][j]] = sub[bernstein_index(3,i,j)];
        }
    }
    return children;
  };
  std::size_t visits = 0;
  return pn_check::pair(pa, pb, separate, certify, normal, subdivide, 0, visits);
}
// Triangles whose patch cannot be proven apart except at its logical common
// feature: disjoint patches are separated when their flat triangles are
// farther apart than the sum of the patch deviations, or when their pieces
// are (pieces_apart).  The flat triangulation is free of self-intersections,
// so flattening both may settle a pair; unresolved flat contacts are refused.
//
// One lift (the repair rounds and the flattening passes) proves the same
// pairs again and again while only the triangles around the vertices it
// turned change.  A pair's classification is a function of its two
// triangles' vertex normals, edge masks and flatness, so PnPairCache keeps
// it and a pair is proved again only when one of its triangles changed
// (compared bit for bit).
struct PnPairCache {
  std::vector<std::array<std::array<double, 3>, 3>> normals;
  std::vector<std::uint8_t> mask, flat, seen;
  std::unordered_map<std::uint64_t, std::uint8_t> classes; // 0 apart, 1 flat contact, 2 unproven
  double seconds{0.0};
  std::size_t calls{0}, proved{0}, reused{0};
};
std::vector<std::size_t> pn_unproven_pairs_impl(
    const std::vector<std::array<double, 3>> &normals,
    const std::vector<std::uint8_t> &mask, const std::vector<std::uint8_t> &flat,
    const NodePatchGeometry &g, std::size_t &flat_contacts,
    TitanExecutor *executor, PnPairCache *cache);
std::vector<std::size_t> pn_unproven_pairs(
    const std::vector<std::array<double, 3>> &normals,
    const std::vector<std::uint8_t> &mask, const std::vector<std::uint8_t> &flat,
    const NodePatchGeometry &g, std::size_t &flat_contacts,
    TitanExecutor *executor = nullptr, PnPairCache *cache = nullptr) {
  const auto begin = std::chrono::steady_clock::now();
  auto result = pn_unproven_pairs_impl(normals, mask, flat, g, flat_contacts,
                                       executor, cache);
  if (cache != nullptr) {
    cache->seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    ++cache->calls;
  }
  return result;
}
std::vector<std::size_t> pn_unproven_pairs_impl(
    const std::vector<std::array<double, 3>> &normals,
    const std::vector<std::uint8_t> &mask, const std::vector<std::uint8_t> &flat,
    const NodePatchGeometry &g, std::size_t &flat_contacts,
    TitanExecutor *executor, PnPairCache *cache) {
  const std::size_t nt = g.triangles.size();
  // Triangles changed since the cache last saw them (all, without a cache).
  std::vector<std::uint8_t> changed(nt, 1U);
  if (cache != nullptr) {
    if (cache->seen.size() != nt) {
      cache->normals.assign(nt, {});
      cache->mask.assign(nt, 0U);
      cache->flat.assign(nt, 0U);
      cache->seen.assign(nt, 0U);
      cache->classes.clear();
    }
    for (std::size_t ti = 0; ti < nt; ++ti) {
      const auto &v = g.triangles[ti].vertices;
      const std::array<std::array<double, 3>, 3> now{normals[v[0]], normals[v[1]],
                                                     normals[v[2]]};
      const bool same = cache->seen[ti] &&
                        std::memcmp(now.data(), cache->normals[ti].data(),
                                    sizeof(now)) == 0 &&
                        cache->mask[ti] == mask[ti] && cache->flat[ti] == flat[ti];
      changed[ti] = same ? 0U : 1U;
      cache->normals[ti] = now;
      cache->mask[ti] = mask[ti];
      cache->flat[ti] = flat[ti];
      cache->seen[ti] = 1U;
    }
  }
  std::vector<std::array<V, 3>> tri(nt);
  std::vector<V> centre(nt);
  std::vector<double> deviation(nt, 0.0), reach(nt, 0.0);
  double cell = 0.0;
  // Each triangle's bound is its own (in parallel); the cell size is then
  // taken in triangle order, as before.
  const auto bound = [&](std::size_t, std::size_t begin, std::size_t end) {
    for (std::size_t ti = begin; ti < end; ++ti) {
      tri[ti] = flat_triangle(g, ti);
      centre[ti] = mul(add(add(tri[ti][0], tri[ti][1]), tri[ti][2]), 1.0 / 3.0);
      if (!flat[ti])
        deviation[ti] =
            pn_deviation(pn_triangle_patch(normals, g, ti, mask[ti]), tri[ti]);
      for (const V &p : tri[ti])
        reach[ti] = std::max(reach[ti], norm(sub(p, centre[ti])));
      reach[ti] += deviation[ti];
    }
  };
  if (executor != nullptr && executor->capacity() > 1U && nt > 256U)
    executor->parallel_for(nt, executor->capacity(), 64U, bound);
  else
    bound(0U, 0U, nt);
  for (std::size_t ti = 0; ti < nt; ++ti)
    cell = std::max(cell, 2 * reach[ti]);
  std::vector<std::size_t> unproven;
  flat_contacts = 0;
  if (!(cell > 0))
    return unproven;
  const auto key = [cell](const V &p, int dx, int dy, int dz) {
    constexpr std::int64_t bias = std::int64_t{1} << 20;
    constexpr std::uint64_t field = (std::uint64_t{1} << 21) - 1U;
    const auto axis = [&](double x, int d) {
      return (static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(std::floor(x / cell)) + d + bias) &
              field);
    };
    return (axis(p[0], dx) << 42) | (axis(p[1], dy) << 21) | axis(p[2], dz);
  };
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid;
  for (std::size_t ti = 0; ti < nt; ++ti)
    grid[key(centre[ti], 0, 0, 0)].push_back(ti);
  std::vector<std::uint8_t> marked(nt, 0U);
  // Every pair proof is independent of the others, so the triangles are
  // scanned in parallel: each worker lists the two triangles of each pair it
  // cannot prove apart and counts its flat contacts, and the lists are merged
  // in worker order.  The set of marked triangles is the serial scan's; only
  // the order of `unproven` can differ, and its consumers use it as a set.
  const std::size_t workers =
      executor != nullptr ? std::max<std::size_t>(1U, std::min(executor->capacity(), nt))
                          : 1U;
  std::vector<std::vector<std::size_t>> hits(workers);
  std::vector<std::size_t> contacts(workers, 0U), proved(workers, 0U),
      reused(workers, 0U);
  std::vector<std::vector<std::pair<std::uint64_t, std::uint8_t>>> fresh(workers);
  const auto scan = [&](std::size_t worker, std::size_t begin, std::size_t end) {
    auto &out = hits[worker];
    std::size_t &touching = contacts[worker];
    for (std::size_t ti = begin; ti < end; ++ti)
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            const auto found = grid.find(key(centre[ti], dx, dy, dz));
            if (found == grid.end())
              continue;
            for (std::size_t tj : found->second) {
              if (tj <= ti)
                continue;
              const auto &a = g.triangles[ti].vertices, &b = g.triangles[tj].vertices;
              bool shared = false;
              for (auto u : a)
                for (auto v : b)
                  shared = shared || u == v;
              if (norm(sub(centre[ti], centre[tj])) > reach[ti] + reach[tj])
                continue;
              const std::uint64_t pair_key =
                  static_cast<std::uint64_t>(ti) * nt + tj;
              std::uint8_t cls = 255U;
              if (cache != nullptr && !changed[ti] && !changed[tj]) {
                const auto hit = cache->classes.find(pair_key);
                if (hit != cache->classes.end())
                  cls = hit->second;
              }
              if (cls == 255U) {
                bool apart = false;
                if (shared)
                  apart = pn_adjacent_patches_apart(
                      pn_triangle_patch(normals, g, ti, mask[ti]), a,
                      pn_triangle_patch(normals, g, tj, mask[tj]), b,
                      flat[ti] && flat[tj]);
                else
                  apart = pieces_apart(
                      patch_piece(pn_triangle_patch(normals, g, ti, mask[ti])),
                      patch_piece(pn_triangle_patch(normals, g, tj, mask[tj])),
                      kPnApartProofDepth);
                cls = apart ? 0U : (flat[ti] && flat[tj] ? 1U : 2U);
                ++proved[worker];
                if (cache != nullptr)
                  fresh[worker].push_back({pair_key, cls});
              } else {
                ++reused[worker];
              }
              if (cls == 0U)
                continue;
              if (cls == 1U) {
                ++touching;
                continue;
              }
              out.push_back(ti);
              out.push_back(tj);
            }
          }
  };
  if (workers > 1U)
    executor->parallel_for(nt, workers, 64U, scan);
  else
    scan(0, 0, nt);
  if (cache != nullptr)
    for (std::size_t w = 0; w < workers; ++w) {
      cache->proved += proved[w];
      cache->reused += reused[w];
      for (const auto &[pair_key, cls] : fresh[w])
        cache->classes.insert_or_assign(pair_key, cls);
    }
  for (std::size_t w = 0; w < workers; ++w) {
    flat_contacts += contacts[w];
    for (const std::size_t t : hits[w])
      if (!flat[t] && !marked[t]) {
        marked[t] = 1U;
        unproven.push_back(t);
      }
  }
  return unproven;
}
// The PN lift keeps one normal per vertex, for the patches and for the
// collocation alike, so every patch around a vertex is tangent there to the
// plane normal to it: the lifted surface is smooth at the collocation point,
// where the jump coefficient and the target-normal kernel assume it.
// Flattening a patch, or straightening an edge in a neighbour, tilts that
// patch's tangent plane at its corners (18-21 deg in a steep case), and the
// self integral of the target-normal kernel then grows like the logarithm of
// the quadrature order instead of converging.  So a triangle whose patch is
// steep, folds, or cannot be proven apart from another is not flattened;
// its vertices change normal instead: first to the area-weighted star normal
// of the flat mesh, then, round by round, halfway toward the mean normal of
// their offending faces, which makes a patch flat in the limit while it
// stays tangent at each vertex to the plane all its neighbours share.  Edges
// follow from their end points and normals, so the patches stay joined.  The
// rounds stop on a full re-test of every triangle and every pair with the
// normals as they then are, so no patch a changed normal touches goes
// unchecked.  What no normal mends (a triangle turned against its own
// vertices, a fold of the flat mesh, sheets that nearly touch) is returned
// for pn_flatten_residual, with the normals of that last re-test.
struct PnRepairReport {
  std::size_t star{0};          // vertices given the star normal
  std::size_t turned{0};        // of them, turned further toward their faces
  std::size_t rounds{0};        // repair rounds that changed a normal
  std::size_t flat_contacts{0}; // non-adjacent flat pairs touching (defect)
};
constexpr int kPnRepairRounds = 16;
std::vector<std::uint8_t> pn_repair_normals(std::vector<std::array<double, 3>> &normals,
                       const std::vector<std::array<double, 3>> &star,
                       const NodePatchGeometry &g, PnRepairReport &report,
                       TitanExecutor *executor = nullptr,
                       PnPairCache *cache = nullptr) {
  const std::size_t nt = g.triangles.size(), nv = normals.size();
  const std::vector<std::uint8_t> curved_edges(nt, 0U), none_flat(nt, 0U);
  std::vector<std::uint8_t> level(nv, 0U);
  report = PnRepairReport{};
  std::string history;
  std::size_t example = nt, fewest = nt + 1;
  int stalled = 0;
  std::vector<std::uint8_t> bad(nt, 0U);
  for (int round = 0; round <= kPnRepairRounds; ++round) {
    bad.assign(nt, 0U);
    std::size_t steep = 0, apart = 0;
    {
      // The steepness proofs are independent; each worker marks its own
      // triangles (distinct bytes of bad) and remembers the last one it saw.
      const std::size_t workers =
          executor != nullptr ? std::max<std::size_t>(1U, std::min(executor->capacity(), nt))
                              : 1U;
      std::vector<std::size_t> count(workers, 0U), last(workers, nt);
      const auto scan = [&](std::size_t worker, std::size_t begin, std::size_t end) {
        for (std::size_t ti = begin; ti < end; ++ti)
          if (pn_triangle_steep(normals, g, ti, 0U)) {
            bad[ti] = 1U;
            ++count[worker];
            last[worker] = ti;
          }
      };
      if (workers > 1U)
        executor->parallel_for(nt, workers, 64U, scan);
      else
        scan(0, 0, nt);
      for (std::size_t w = 0; w < workers; ++w) {
        steep += count[w];
        if (last[w] < nt)
          example = last[w];
      }
    }
    for (std::size_t ti : pn_unproven_pairs(normals, curved_edges, none_flat, g,
                                           report.flat_contacts, executor,
                                           cache)) {
      apart += bad[ti] ? 0 : 1;
      bad[ti] = 1U;
      if (example == nt)
        example = ti;
    }
    history += " " + std::to_string(steep) + "/" + std::to_string(apart);
    if (std::find(bad.begin(), bad.end(), 1U) == bad.end()) {
      if (report.flat_contacts != 0)
        throw std::runtime_error("curved_pn_intersection_unresolved");
      return {};
    }
    // Two rounds without fewer bad triangles: what is left is no normal's.
    const auto count = std::size_t(std::count(bad.begin(), bad.end(), 1U));
    stalled = count < fewest ? 0 : stalled + 1;
    fewest = std::min(fewest, count);
    if (round == kPnRepairRounds || stalled >= 2)
      break;
    std::vector<V> faces(nv, V{});
    for (std::size_t ti = 0; ti < nt; ++ti)
      if (bad[ti])
        for (const auto v : g.triangles[ti].vertices)
          faces[v] = add(faces[v],
                         V{g.triangles[ti].nx, g.triangles[ti].ny, g.triangles[ti].nz});
    bool changed = false;
    for (std::size_t v = 0; v < nv; ++v) {
      if (!(norm(faces[v]) > 0))
        continue;
      V next;
      if (level[v] == 0U) {
        next = star[v];
        ++report.star;
      } else {
        next = add(normals[v], mul(faces[v], 1 / norm(faces[v])));
        if (level[v] == 1U)
          ++report.turned;
      }
      const double length = norm(next);
      if (!(length > 1e-12))
        continue;
      next = mul(next, 1 / length);
      changed = changed || next != normals[v];
      normals[v] = next;
      level[v] = std::uint8_t(std::min(255, level[v] + 1));
    }
    if (!changed)
      break;
    ++report.rounds;
  }
  std::size_t steep = 0;
  for (std::size_t ti = 0; ti < nt; ++ti)
    steep += bad[ti] && pn_triangle_steep(normals, g, ti, 0U) ? 1U : 0U;
  std::cerr << "TABI curved PN: " << std::count(bad.begin(), bad.end(), 1U)
            << " triangle(s) no vertex normal mends, " << steep
            << " of them steep or folded, the others not provably apart from "
               "another patch (by round, steep or folded / not apart only:"
            << history << ")";
  if (example < nt) {
    const auto &t = g.triangles[example];
    const V face{t.nx, t.ny, t.nz};
    std::cerr << "; e.g. triangle " << example << ", face-normal cosines";
    for (const auto v : t.vertices)
      std::cerr << " " << dot(normals[v], face);
  }
  std::cerr << "; flattened, their corner vertices collocate off the vertex\n";
  return bad;
}
// The triangles pn_repair_normals returns are flat, so the surface stays
// valid, and every edge of a flat triangle is straight in both triangles that
// share it, so the neighbours stay joined; a straightened edge can fold a
// neighbour, and a pair still not proven apart is flattened too, until
// neither pass flattens anything.  The vertices of these patches are corners
// of the lifted surface: their cells collocate off the vertex.
struct PnLiftReport {
  std::size_t flattened{0};     // flat triangles, for any reason
  std::size_t straightened{0};  // edges straight because a neighbour is flat
  std::size_t flat_contacts{0}; // flat non-adjacent pairs touching (mesh defect)
};
std::vector<std::uint8_t>
pn_flatten_residual(const std::vector<std::array<double, 3>> &normals,
                    const NodePatchGeometry &g, std::vector<std::uint8_t> flat,
                    PnLiftReport &report, TitanExecutor *executor = nullptr,
                    PnPairCache *cache = nullptr) {
  const std::size_t nt = g.triangles.size();
  const std::uint64_t nv = g.vertex_xyz.size();
  const auto key = [nv](std::size_t a, std::size_t b) {
    return a < b ? a * nv + b : b * nv + a;
  };
  std::vector<std::uint8_t> mask(nt, 0U);
  std::vector<std::size_t> retest;
  report = PnLiftReport{};
  while (true) {
    for (bool changed = true; changed;) {
      std::unordered_set<std::uint64_t> straight;
      for (std::size_t ti = 0; ti < nt; ++ti)
        if (flat[ti])
          for (int k = 0; k < 3; ++k)
            straight.insert(key(g.triangles[ti].vertices[k],
                                g.triangles[ti].vertices[(k + 1) % 3]));
      retest.clear();
      for (std::size_t ti = 0; ti < nt; ++ti) {
        std::uint8_t m = 0U;
        for (int k = 0; k < 3; ++k)
          if (straight.count(key(g.triangles[ti].vertices[k],
                                 g.triangles[ti].vertices[(k + 1) % 3])))
            m |= std::uint8_t(1U << k);
        if (!flat[ti] && m != mask[ti])
          retest.push_back(ti);
        mask[ti] = m;
      }
      changed = false;
      for (std::size_t ti : retest)
        if (pn_triangle_steep(normals, g, ti, mask[ti])) {
          flat[ti] = 1U;
          changed = true;
        }
    }
    const auto unproven =
        pn_unproven_pairs(normals, mask, flat, g, report.flat_contacts, executor,
                          cache);
    if (unproven.empty()) {
      if (report.flat_contacts != 0)
        throw std::runtime_error("curved_pn_intersection_unresolved");
      break;
    }
    for (std::size_t ti : unproven)
      flat[ti] = 1U;
  }
  for (std::size_t ti = 0; ti < nt; ++ti) {
    if (flat[ti])
      ++report.flattened;
    else
      report.straightened += std::size_t(std::popcount(unsigned(mask[ti])));
  }
  return mask;
}
// The largest distance between the two patches along their shared edges,
// sampled at seven interior points of every edge.  The shared-edge rule makes
// it rounding; it certifies the lifted surface the quadrature integrates, not
// only the flat triangulation it came from.
double pn_shared_edge_gap(const std::vector<std::array<double, 3>> &normals,
                          const std::vector<std::uint8_t> &mask,
                          const NodePatchGeometry &g) {
  double gap = 0.0;
  for (std::size_t ti = 0; ti < g.triangles.size(); ++ti) {
    const auto &t = g.triangles[ti];
    const PnPatch pi = pn_triangle_patch(normals, g, ti, mask.empty() ? 0U : mask[ti]);
    for (int k = 0; k < 3; ++k) {
      const std::size_t a = t.vertices[k], b = t.vertices[(k + 1) % 3];
      for (auto pos = g.incident_offsets[a]; pos < g.incident_offsets[a + 1];
           ++pos) {
        const auto tj = g.incident_triangles[pos];
        if (tj <= ti)
          continue;
        const auto &u = g.triangles[tj];
        int la = -1, lb = -1;
        for (int m = 0; m < 3; ++m) {
          if (u.vertices[m] == a) la = m;
          if (u.vertices[m] == b) lb = m;
        }
        if (la < 0 || lb < 0)
          continue;
        const PnPatch pj = pn_triangle_patch(normals, g, tj, mask.empty() ? 0U : mask[tj]);
        for (int q = 1; q <= 7; ++q) {
          const double s = q / 8.0;
          std::array<double, 3> li{}, lj{};
          li[k] = 1 - s;
          li[(k + 1) % 3] = s;
          lj[la] = 1 - s;
          lj[lb] = s;
          V xi, xj, d1, d2, d3;
          pn_evaluate(pi, li[0], li[1], li[2], xi, d1, d2, d3);
          pn_evaluate(pj, lj[0], lj[1], lj[2], xj, d1, d2, d3);
          gap = std::max(gap, norm(sub(xi, xj)));
        }
      }
    }
  }
  return gap;
}

// One quad of the dual cell (see QuadRect) on the PN patch of its triangle:
// the patch in the order node, next, previous, and the quad's far edge.  A
// point (u, v) of the unit square maps linearly to barycentric
// lambda(u,v) = a + u ((1-v) e0 + v e1 - a), a = (1, 0, 0).
struct PnQuad {
  PnPatch patch;
  V face;
  std::array<std::array<double, 3>, 2> edge;
};
PnQuad pn_quad(const std::vector<std::array<double, 3>> &normals,
               const std::vector<std::uint8_t> &straight_edges,
               const NodePatchGeometry &g, std::size_t node, std::size_t pos,
               int half) {
  const auto ti = g.incident_triangles[pos];
  const auto &t = g.triangles[ti];
  int local = 0;
  while (t.vertices[local] != node)
    ++local;
  const std::size_t ia = t.vertices[local],
                    ib = t.vertices[(local + 1) % 3],
                    ic = t.vertices[(local + 2) % 3];
  const V pa{g.vertex_xyz[ia][0], g.vertex_xyz[ia][1], g.vertex_xyz[ia][2]},
      pb{g.vertex_xyz[ib][0], g.vertex_xyz[ib][1], g.vertex_xyz[ib][2]},
      pc{g.vertex_xyz[ic][0], g.vertex_xyz[ic][1], g.vertex_xyz[ic][2]};
  // The triangle's mask rotated to the patch order a, b, c.
  const unsigned m = straight_edges.empty() ? 0U : straight_edges[ti];
  const unsigned straight = ((m >> local) & 1U) |
                            (((m >> ((local + 1) % 3)) & 1U) << 1) |
                            (((m >> ((local + 2) % 3)) & 1U) << 2);
  // Barycentric coordinates of the fan corners: vertex a, then the two
  // edge midpoints and the centroid, matching samples() exactly.
  const std::array<double, 3> Mab{.5, .5, 0}, Mac{.5, 0, .5},
      C{1. / 3, 1. / 3, 1. / 3};
  return {pn_patch(pa, pb, pc, normals[ia], normals[ib], normals[ic], straight),
          V{t.nx, t.ny, t.nz},
          half == 0 ? std::array<std::array<double, 3>, 2>{Mab, C}
                    : std::array<std::array<double, 3>, 2>{C, Mac}};
}
// Position, unit normal (oriented by the flat triangle) and area element
// |dP/du x dP/dv| of the quad at (u, v).
// Position and the two parameter derivatives of the quad at (u, v); at u = 0
// (the vertex) Pv vanishes and Pu is the direction of the ray at v.
void pn_quad_frame(const PnQuad &q, double u, double v, V &P, V &Pu, V &Pv) {
  const std::array<double, 3> La{1, 0, 0};
  std::array<double, 3> lam{}, dlu{}, dlv{};
  for (int k = 0; k < 3; ++k) {
    const double ev = (1 - v) * q.edge[0][k] + v * q.edge[1][k];
    lam[k] = La[k] + u * (ev - La[k]);
    dlu[k] = ev - La[k];
    dlv[k] = u * (q.edge[1][k] - q.edge[0][k]);
  }
  V d1, d2, d3;
  pn_evaluate(q.patch, lam[0], lam[1], lam[2], P, d1, d2, d3);
  Pu = add(add(mul(d1, dlu[0]), mul(d2, dlu[1])), mul(d3, dlu[2]));
  Pv = add(add(mul(d1, dlv[0]), mul(d2, dlv[1])), mul(d3, dlv[2]));
}
void pn_quad_point(const PnQuad &q, double u, double v, V &P, V &nrm,
                   double &area) {
  V Pu, Pv;
  pn_quad_frame(q, u, v, P, Pu, Pv);
  nrm = cross(Pu, Pv);
  area = norm(nrm);
  if (!(area > 0) || !std::isfinite(area))
    throw std::runtime_error("curved_dual_pn_degenerate_area");
  nrm = mul(nrm, 1 / area);
  if (dot(nrm, q.face) < 0)
    nrm = mul(nrm, -1.0);
}
template <typename F>
void visit_pn_quad(const std::vector<std::array<double, 3>> &normals,
                   const std::vector<std::uint8_t> &straight_edges,
                   const NodePatchGeometry &g, std::size_t node,
                   std::size_t pos, int half, const QuadRect &rect, int order,
                   F &&f) {
  const auto &rule = gauss_rule(order);
  const PnQuad q = pn_quad(normals, straight_edges, g, node, pos, half);
  const double du = rect.u1 - rect.u0, dv = rect.v1 - rect.v0;
  for (auto ru : rule)
    for (auto rv : rule) {
      const double u = rect.u0 + du * ru[0], v = rect.v0 + dv * rv[0];
      V P, nrm;
      double area;
      pn_quad_point(q, u, v, P, nrm, area);
      const double w = (ru[1] * du) * (rv[1] * dv) * area;
      f(CurvedDualSample{P, nrm, w});
    }
}
// The physical polar angle about a singular point as the angular variable of
// a Duffy-type rule.  The rule's rays leave the point in the directions
// D(t) = (1 - t) D0 + t D1, linear in its angular parameter t, which turn
// through phi1 = angle(D0, D1); with phi the variable,
//   t(phi) = |D0| sin phi / (|D1| sin(phi1 - phi) + |D0| sin phi),
//   dt/dphi = |D0| |D1| sin phi1 / (|D1| sin(phi1 - phi) + |D0| sin phi)^2.
// After the radial Jacobian the kernels' 1/r leaves 1/|D(t)| in t, which on a
// sliver (|D(t)| nearly zero somewhere along t) is a narrow peak that Gauss
// in t resolves only as exp(-n h/L) - 1UV0 and 4MTH had self blocks still
// moving at order 128 in cells whose triangles have 6-11 and 169-174 degree
// angles at the vertex; in phi it is the smooth ray length of the polar form.
struct PolarMap {
  double phi1{0.0}, n0{0.0}, n1{0.0}, sin1{0.0};
  bool ok{false};
};
PolarMap polar_map(const V &d0, const V &d1) {
  PolarMap m;
  m.n0 = norm(d0);
  m.n1 = norm(d1);
  m.phi1 = std::atan2(norm(cross(d0, d1)), dot(d0, d1));
  m.sin1 = std::sin(m.phi1);
  m.ok = m.n0 > 0 && m.n1 > 0 && m.phi1 > 1e-12;
  return m;
}
// t and dt/dphi at phi in [0, phi1].
void polar_t(const PolarMap &m, double phi, double &t, double &dt) {
  const double a = m.n0 * std::sin(phi), b = m.n1 * std::sin(m.phi1 - phi);
  const double den = a + b;
  t = a / den;
  dt = m.n0 * m.n1 * m.sin1 / (den * den);
}
// A quad whose integrand is singular at the point s = (su, sv) of its unit
// square (an offset collocation point, on the square's boundary or inside):
// the square is split into the triangles s, c_k, c_k+1 over its four sides
// (those through s vanish and are skipped), and each is integrated by the
// Duffy map (rho, t) -> s + rho ((1-t) c_k + t c_k+1 - s), whose Jacobian
// rho cancels the 1/r of the kernels at s, with t taken by the polar angle
// of the rays at s (polar_map, directions from the quad's derivatives there).
template <typename F>
void visit_pn_quad_duffy(const std::vector<std::array<double, 3>> &normals,
                         const std::vector<std::uint8_t> &straight_edges,
                         const NodePatchGeometry &g, std::size_t node,
                         std::size_t pos, int half, double su, double sv,
                         int order, F &&f) {
  const auto &rule = gauss_rule(order);
  const PnQuad q = pn_quad(normals, straight_edges, g, node, pos, half);
  V Ps, Pu, Pv;
  pn_quad_frame(q, su, sv, Ps, Pu, Pv);
  const std::array<std::array<double, 2>, 4> corner{
      {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
  for (int k = 0; k < 4; ++k) {
    const auto &c0 = corner[k], &c1 = corner[(k + 1) % 4];
    const double det = std::abs((c0[0] - su) * (c1[1] - sv) -
                                (c0[1] - sv) * (c1[0] - su));
    if (!(det > 1e-14))
      continue;
    const PolarMap m =
        polar_map(add(mul(Pu, c0[0] - su), mul(Pv, c0[1] - sv)),
                  add(mul(Pu, c1[0] - su), mul(Pv, c1[1] - sv)));
    if (!m.ok)
      throw std::runtime_error("curved_dual_pn_degenerate_area");
    // The angular substitution depends on the angular node alone: its end
    // point on the side (eu, ev) and its weight factor phi1 w_a dt/dphi are
    // taken once per angular node, not once per sample, with the same
    // products in the same order as before.
    struct Angular {
      double eu, ev, weight;
    };
    std::array<Angular, 256> angular;
    const std::size_t na = rule.size();
    for (std::size_t a = 0; a < na; ++a) {
      double t, dt;
      polar_t(m, m.phi1 * rule[a][0], t, dt);
      angular[a] = {(1 - t) * c0[0] + t * c1[0], (1 - t) * c0[1] + t * c1[1],
                    m.phi1 * rule[a][1] * dt};
    }
    for (auto rr : rule)
      for (std::size_t a = 0; a < na; ++a) {
        const Angular &an = angular[a];
        const double u = su + rr[0] * (an.eu - su), v = sv + rr[0] * (an.ev - sv);
        V P, nrm;
        double area;
        pn_quad_point(q, u, v, P, nrm, area);
        f(CurvedDualSample{P, nrm, rr[1] * an.weight * rr[0] * det * area});
      }
  }
}
// The self samples of a cell that collocates at its vertex: each quad's ray
// parameter u by Gauss and its angle by the polar angle at the vertex (the
// rays there point along Pu(0, v), linear in v).  The quads of
// visit_pn_samples with v replaced by the polar angle.
template <typename F>
void visit_pn_samples_polar(const std::vector<std::array<double, 3>> &normals,
                            const std::vector<std::uint8_t> &straight_edges,
                            const NodePatchGeometry &g, std::size_t node,
                            int order, F &&f) {
  const auto &rule = gauss_rule(order);
  for (auto pos = g.incident_offsets[node]; pos < g.incident_offsets[node + 1];
       ++pos)
    for (int half = 0; half < 2; ++half) {
      const PnQuad q = pn_quad(normals, straight_edges, g, node, pos, half);
      V P0, D0, D1, unused;
      pn_quad_frame(q, 0.0, 0.0, P0, D0, unused);
      pn_quad_frame(q, 0.0, 1.0, P0, D1, unused);
      const PolarMap m = polar_map(D0, D1);
      if (!m.ok)
        throw std::runtime_error("curved_dual_pn_degenerate_area");
      // v and the angular weight factor depend on the angular node alone
      // (see visit_pn_quad_duffy); same products, same order.
      struct Angular {
        double v, weight;
      };
      std::array<Angular, 256> angular;
      const std::size_t na = rule.size();
      for (std::size_t a = 0; a < na; ++a) {
        double v, dv;
        polar_t(m, m.phi1 * rule[a][0], v, dv);
        angular[a] = {v, m.phi1 * rule[a][1] * dv};
      }
      for (auto ru : rule)
        for (std::size_t a = 0; a < na; ++a) {
          V P, nrm;
          double area;
          pn_quad_point(q, ru[0], angular[a].v, P, nrm, area);
          f(CurvedDualSample{P, nrm, ru[1] * angular[a].weight * area});
        }
    }
}
// The rules of the graded self block (graded_self_block), on a sub-rectangle
// of a quad.  The polar fan of visit_pn_samples_polar on [0, u1] x [v0, v1]:
// the rays from the vertex point along Pu(0, v), linear in v there too.
template <typename F>
void visit_pn_rect_polar(const PnQuad &q, double u1, double v0, double v1,
                         int order, F &&f) {
  const auto &rule = gauss_rule(order);
  V P0, D0, D1, unused;
  pn_quad_frame(q, 0.0, v0, P0, D0, unused);
  pn_quad_frame(q, 0.0, v1, P0, D1, unused);
  const PolarMap m = polar_map(D0, D1);
  if (!m.ok)
    throw std::runtime_error("curved_dual_pn_degenerate_area");
  for (auto ru : rule)
    for (auto ra : rule) {
      double t, dt;
      polar_t(m, m.phi1 * ra[0], t, dt);
      V P, nrm;
      double area;
      pn_quad_point(q, u1 * ru[0], v0 + (v1 - v0) * t, P, nrm, area);
      f(CurvedDualSample{P, nrm, (u1 * ru[1]) * (m.phi1 * ra[1] * dt) *
                                     (v1 - v0) * area});
    }
}
// The polar Duffy rule of visit_pn_quad_duffy on the rectangle r, singular
// at s = (su, sv) on its boundary or inside: the triangles s, c_k, c_k+1
// over the sides of r.
template <typename F>
void visit_pn_rect_duffy(const PnQuad &q, const QuadRect &r, double su,
                         double sv, int order, F &&f) {
  const auto &rule = gauss_rule(order);
  V Ps, Pu, Pv;
  pn_quad_frame(q, su, sv, Ps, Pu, Pv);
  const std::array<std::array<double, 2>, 4> corner{
      {{r.u0, r.v0}, {r.u1, r.v0}, {r.u1, r.v1}, {r.u0, r.v1}}};
  for (int k = 0; k < 4; ++k) {
    const auto &c0 = corner[k], &c1 = corner[(k + 1) % 4];
    const double det = std::abs((c0[0] - su) * (c1[1] - sv) -
                                (c0[1] - sv) * (c1[0] - su));
    if (!(det > 1e-14 * (r.u1 - r.u0) * (r.v1 - r.v0)))
      continue;
    const PolarMap m =
        polar_map(add(mul(Pu, c0[0] - su), mul(Pv, c0[1] - sv)),
                  add(mul(Pu, c1[0] - su), mul(Pv, c1[1] - sv)));
    if (!m.ok)
      throw std::runtime_error("curved_dual_pn_degenerate_area");
    for (auto rr : rule)
      for (auto ra : rule) {
        double t, dt;
        polar_t(m, m.phi1 * ra[0], t, dt);
        const double eu = (1 - t) * c0[0] + t * c1[0],
                     ev = (1 - t) * c0[1] + t * c1[1];
        V P, nrm;
        double area;
        pn_quad_point(q, su + rr[0] * (eu - su), sv + rr[0] * (ev - sv), P,
                      nrm, area);
        f(CurvedDualSample{P, nrm,
                           rr[1] * (m.phi1 * ra[1] * dt) * rr[0] * det * area});
      }
  }
}
// The polar Duffy rule of an off-vertex collocation point s = (2/3, 1/6, 1/6)
// over the cell's part of its host patch, the polygon a, M_ab, C, M_ac, in
// the triangle's own barycentric coordinates (lambda_b, lambda_c).  The quad
// parameters (u, v) of visit_pn_quad_duffy collapse the side u = 0 onto the
// vertex, so their rays from s are not straight lines of the patch, and on a
// sliver host - s then lies close to the vertex - the polar substitution,
// which assumes rays linear in the parameter, no longer smooths the
// integrand (6LZG: K4 still moving 6e-5 at order 128).  In barycentric
// coordinates the patch is a cubic polynomial, the rays are straight, and
// the four triangles s, c_k, c_k+1 over the polygon's sides take the same
// polar Duffy rule.
template <typename F>
void visit_pn_host_duffy_bary(const PnQuad &q, int order, F &&f) {
  const auto &rule = gauss_rule(order);
  const auto frame = [&](double lb, double lc, V &P, V &Pb, V &Pc) {
    V d1, d2, d3;
    pn_evaluate(q.patch, 1 - lb - lc, lb, lc, P, d1, d2, d3);
    Pb = sub(d2, d1);
    Pc = sub(d3, d1);
  };
  const double sb = 1.0 / 6, sc = 1.0 / 6;
  V Ps, Pb, Pc;
  frame(sb, sc, Ps, Pb, Pc);
  // (lambda_b, lambda_c) of a, M_ab, C, M_ac.
  const std::array<std::array<double, 2>, 4> corner{
      {{0.0, 0.0}, {0.5, 0.0}, {1.0 / 3, 1.0 / 3}, {0.0, 0.5}}};
  for (int k = 0; k < 4; ++k) {
    const auto &c0 = corner[k], &c1 = corner[(k + 1) % 4];
    const double det = std::abs((c0[0] - sb) * (c1[1] - sc) -
                                (c0[1] - sc) * (c1[0] - sb));
    if (!(det > 1e-14))
      continue;
    const PolarMap m =
        polar_map(add(mul(Pb, c0[0] - sb), mul(Pc, c0[1] - sc)),
                  add(mul(Pb, c1[0] - sb), mul(Pc, c1[1] - sc)));
    if (!m.ok)
      throw std::runtime_error("curved_dual_pn_degenerate_area");
    for (auto rr : rule)
      for (auto ra : rule) {
        double t, dt;
        polar_t(m, m.phi1 * ra[0], t, dt);
        const double eb = (1 - t) * c0[0] + t * c1[0],
                     ec = (1 - t) * c0[1] + t * c1[1];
        V P, db, dc;
        frame(sb + rr[0] * (eb - sb), sc + rr[0] * (ec - sc), P, db, dc);
        V nrm = cross(db, dc);
        const double area = norm(nrm);
        if (!(area > 0) || !std::isfinite(area))
          throw std::runtime_error("curved_dual_pn_degenerate_area");
        nrm = mul(nrm, 1 / area);
        if (dot(nrm, q.face) < 0)
          nrm = mul(nrm, -1.0);
        f(CurvedDualSample{P, nrm,
                           rr[1] * (m.phi1 * ra[1] * dt) * rr[0] * det * area});
      }
  }
}
template <typename F>
void visit_pn_samples(const std::vector<std::array<double, 3>> &normals,
                      const std::vector<std::uint8_t> &straight_edges,
                      const NodePatchGeometry &g, std::size_t node, int order,
                      F &&f) {
  for (auto pos = g.incident_offsets[node]; pos < g.incident_offsets[node + 1];
       ++pos)
    for (int half = 0; half < 2; ++half)
      visit_pn_quad(normals, straight_edges, g, node, pos, half, QuadRect{},
                    order, f);
}
std::vector<CurvedDualSample>
pn_samples(const std::vector<std::array<double, 3>> &normals,
           const std::vector<std::uint8_t> &straight_edges,
           const NodePatchGeometry &g, std::size_t node, int order) {
  std::vector<CurvedDualSample> out;
  visit_pn_samples(normals, straight_edges, g, node, order,
                   [&](const CurvedDualSample &q) { out.push_back(q); });
  return out;
}
B integrate_scalar(const Panel &target, std::span<const CurvedDualSample> q,
                   double eps, double kappa, B sum = {}) {
  const V x{target.x, target.y, target.z}, nx{target.nx, target.ny, target.nz};
  for (const auto &s : q) {
    const V delta = sub(x, s.xyz);
    const double r = norm(delta);
    if (!(r > 1e-14))
      throw std::runtime_error("curved_dual_quadrature_collision");
    const double z = kappa * r, e = std::exp(-z);
    // Analytic series only where subtraction would lose relative precision.
    // At the switch its omitted absolute term is below 3e-23.
    const double e1 =
        z < .01
            ? z * z *
                  (.5 +
                   z * (-1. / 3 +
                        z * (1. / 8 +
                             z * (-1. / 30 + z * (1. / 144 + z * (-1. / 840 +
                                                                  z / 5760))))))
            : -std::expm1(-z) - z * e;
    const double a = dot(nx, delta) / r, b = dot(s.normal, delta) / r,
                 c = dot(nx, s.normal);
    const double scale = 1 / (4 * pi * r * r), src = b * scale,
                 tgt = -a * scale;
    const double gp = e1 * scale, gpp = (z * z * e - 2 * e1) * scale / r;
    const B value{static_cast<double>(-std::expm1(-z) / (4 * pi * r)),
                  static_cast<double>((eps - 1 - eps * e1) * src),
                  static_cast<double>((1 - 1 / eps + e1 / eps) * tgt),
                  static_cast<double>(-gpp * a * b - gp / r * (c - a * b))};
    for (int k = 0; k < 4; ++k)
      sum[k] += s.weight * value[k];
  }
  return sum;
}
#if TITAN_SIMD_X86
// integrate_scalar four samples at a time.  exp is titan_avx2::exp_pd (about
// one ulp); 1 - e^-z comes from its series below z = 0.1, where the
// subtraction would lose relative accuracy, and e1 = 1 - (1 + z) e^-z from
// the same series as the scalar path below z = 0.01.  Results agree with the
// scalar path to rounding, not bit for bit.
TITAN_TARGET_AVX2
B integrate_avx2(const Panel &target, std::span<const CurvedDualSample> q,
                 double eps, double kappa) {
  const std::size_t full = q.size() & ~std::size_t{3};
  const __m256d tx = _mm256_set1_pd(target.x), ty = _mm256_set1_pd(target.y),
                tz = _mm256_set1_pd(target.z);
  const __m256d tnx = _mm256_set1_pd(target.nx),
                tny = _mm256_set1_pd(target.ny),
                tnz = _mm256_set1_pd(target.nz);
  const __m256d vk = _mm256_set1_pd(kappa), one = _mm256_set1_pd(1.0),
                two = _mm256_set1_pd(2.0);
  const __m256d inv4pi = _mm256_set1_pd(1.0 / (4 * pi));
  const __m256d c_src = _mm256_set1_pd(eps - 1), c_srce = _mm256_set1_pd(eps);
  const __m256d c_tgt = _mm256_set1_pd(1 - 1 / eps),
                c_tgte = _mm256_set1_pd(1 / eps);
  __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
  bool collision = false;
  for (std::size_t i = 0; i < full; i += 4) {
    const auto &a0 = q[i], &a1 = q[i + 1], &a2 = q[i + 2], &a3 = q[i + 3];
    const __m256d dx = _mm256_sub_pd(
        tx, _mm256_set_pd(a3.xyz[0], a2.xyz[0], a1.xyz[0], a0.xyz[0]));
    const __m256d dy = _mm256_sub_pd(
        ty, _mm256_set_pd(a3.xyz[1], a2.xyz[1], a1.xyz[1], a0.xyz[1]));
    const __m256d dz = _mm256_sub_pd(
        tz, _mm256_set_pd(a3.xyz[2], a2.xyz[2], a1.xyz[2], a0.xyz[2]));
    const __m256d snx =
        _mm256_set_pd(a3.normal[0], a2.normal[0], a1.normal[0], a0.normal[0]);
    const __m256d sny =
        _mm256_set_pd(a3.normal[1], a2.normal[1], a1.normal[1], a0.normal[1]);
    const __m256d snz =
        _mm256_set_pd(a3.normal[2], a2.normal[2], a1.normal[2], a0.normal[2]);
    const __m256d w = _mm256_set_pd(a3.weight, a2.weight, a1.weight, a0.weight);
    const __m256d r2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    const __m256d r = _mm256_sqrt_pd(r2);
    collision |= _mm256_movemask_pd(
                     _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_NGT_UQ)) != 0;
    const __m256d inv_r = _mm256_div_pd(one, r);
    const __m256d z = _mm256_mul_pd(vk, r);
    const __m256d e = titan_avx2::exp_pd(_mm256_sub_pd(_mm256_setzero_pd(), z));
    // 1 - e^-z = z (1 - z/2 (1 - z/3 (1 - z/4 (...)))), to z^12 below 0.1.
    __m256d series = one;
    for (int k = 12; k >= 2; --k)
      series = _mm256_sub_pd(
          one, _mm256_mul_pd(_mm256_mul_pd(z, _mm256_set1_pd(1.0 / k)), series));
    const __m256d em1 = _mm256_blendv_pd(
        _mm256_sub_pd(one, e), _mm256_mul_pd(z, series),
        _mm256_cmp_pd(z, _mm256_set1_pd(0.1), _CMP_LT_OQ));
    // e1 = 1 - (1 + z) e^-z: the scalar path's series below z = 0.01.
    __m256d e1s = _mm256_set1_pd(1.0 / 5760);
    e1s = _mm256_add_pd(_mm256_set1_pd(-1.0 / 840), _mm256_mul_pd(z, e1s));
    e1s = _mm256_add_pd(_mm256_set1_pd(1.0 / 144), _mm256_mul_pd(z, e1s));
    e1s = _mm256_add_pd(_mm256_set1_pd(-1.0 / 30), _mm256_mul_pd(z, e1s));
    e1s = _mm256_add_pd(_mm256_set1_pd(1.0 / 8), _mm256_mul_pd(z, e1s));
    e1s = _mm256_add_pd(_mm256_set1_pd(-1.0 / 3), _mm256_mul_pd(z, e1s));
    e1s = _mm256_add_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(z, e1s));
    e1s = _mm256_mul_pd(_mm256_mul_pd(z, z), e1s);
    const __m256d e1 = _mm256_blendv_pd(
        _mm256_sub_pd(em1, _mm256_mul_pd(z, e)), e1s,
        _mm256_cmp_pd(z, _mm256_set1_pd(0.01), _CMP_LT_OQ));
    const __m256d a = _mm256_mul_pd(
        _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(tnx, dx), _mm256_mul_pd(tny, dy)),
                      _mm256_mul_pd(tnz, dz)),
        inv_r);
    const __m256d b = _mm256_mul_pd(
        _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(snx, dx), _mm256_mul_pd(sny, dy)),
                      _mm256_mul_pd(snz, dz)),
        inv_r);
    const __m256d c = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(tnx, snx), _mm256_mul_pd(tny, sny)),
        _mm256_mul_pd(tnz, snz));
    const __m256d scale = _mm256_mul_pd(inv4pi, _mm256_mul_pd(inv_r, inv_r));
    const __m256d src = _mm256_mul_pd(b, scale);
    const __m256d tgt = _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(a, scale));
    const __m256d gp = _mm256_mul_pd(e1, scale);
    const __m256d gpp = _mm256_mul_pd(
        _mm256_mul_pd(
            _mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(z, z), e),
                          _mm256_mul_pd(two, e1)),
            scale),
        inv_r);
    const __m256d ab = _mm256_mul_pd(a, b);
    const __m256d v0 = _mm256_mul_pd(_mm256_mul_pd(em1, inv4pi), inv_r);
    const __m256d v1 =
        _mm256_mul_pd(_mm256_sub_pd(c_src, _mm256_mul_pd(c_srce, e1)), src);
    const __m256d v2 =
        _mm256_mul_pd(_mm256_add_pd(c_tgt, _mm256_mul_pd(e1, c_tgte)), tgt);
    const __m256d v3 = _mm256_sub_pd(
        _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(gpp, ab)),
        _mm256_mul_pd(_mm256_mul_pd(gp, inv_r), _mm256_sub_pd(c, ab)));
    s0 = _mm256_add_pd(s0, _mm256_mul_pd(w, v0));
    s1 = _mm256_add_pd(s1, _mm256_mul_pd(w, v1));
    s2 = _mm256_add_pd(s2, _mm256_mul_pd(w, v2));
    s3 = _mm256_add_pd(s3, _mm256_mul_pd(w, v3));
  }
  if (collision)
    throw std::runtime_error("curved_dual_quadrature_collision");
  const B partial{titan_avx2::hsum_pd(s0), titan_avx2::hsum_pd(s1),
                  titan_avx2::hsum_pd(s2), titan_avx2::hsum_pd(s3)};
  return integrate_scalar(target, q.subspan(full), eps, kappa, partial);
}
#endif
B integrate(const Panel &target, std::span<const CurvedDualSample> q,
            double eps, double kappa) {
#if TITAN_SIMD_X86
  const B sum = titan_simd_dispatch::runtime_has_avx2() && q.size() >= 8
                    ? integrate_avx2(target, q, eps, kappa)
                    : integrate_scalar(target, q, eps, kappa);
#else
  const B sum = integrate_scalar(target, q, eps, kappa);
#endif
  for (double v : sum)
    if (!std::isfinite(v))
      throw std::runtime_error("curved_dual_nonfinite_kernel");
  return sum;
}
// A pair outside the one-ring whose target lies within two cell radii of the
// source vertex is integrated to convergence instead of with the stored
// samples: Gauss orders 8..256 per direction on the parent surface, accepted
// after two successive agreements of all four coefficients within
// 1e-10 + 2e-6 relative. The stored samples (order 2-4) lose K2 as two sheets
// approach: against an order-256 reference, 0.017% at a 0.2 A gap between two
// R = 1.7 A spheres, 1.5% at 0.05 A, 18% at 0.01 A and 71% at 0.002 A. The
// test is geometric, so it does not depend on the Krylov vector.
// The kernel summed over the samples a visitor produces.  Samples stream
// through a fixed tile, so no order^2 array is allocated per pair.
template <typename Visit>
B integrate_streamed(const Panel &target, double eps, double kappa,
                     Visit &&visit) {
  B current{};
  std::array<CurvedDualSample, 64> tile;
  std::size_t used = 0;
  const auto flush = [&] {
    if (used == 0)
      return;
    const B value = integrate(target, std::span(tile.data(), used), eps, kappa);
    for (int k = 0; k < 4; ++k)
      current[k] += value[k];
    used = 0;
  };
  const auto accumulate = [&](const CurvedDualSample &q) {
    tile[used++] = q;
    if (used == tile.size())
      flush();
  };
  visit(accumulate);
  flush();
  return current;
}
// The size of each kernel's terms in a sum over samples, with the geometric
// projections (n.(x - y)/r, n_x.n_y) replaced by their bound 1: what the
// rounding of those projections, of order one ulp of 1 per sample, can move
// the sum by.  K1 has no projection and no cancellation (1 - e^-kr comes from
// expm1), so its size is the sum itself.
B rounding_sizes(const Panel &target, std::span<const CurvedDualSample> q,
                 double eps, double kappa) {
  const V x{target.x, target.y, target.z};
  B size{};
  for (const auto &s : q) {
    const double r = norm(sub(x, s.xyz));
    const double z = kappa * r, e = std::exp(-z);
    const double e1 = z < .01 ? z * z * (.5 - z / 3) : -std::expm1(-z) - z * e;
    const double scale = 1 / (4 * pi * r * r);
    const double gp = e1 * scale, gpp = (z * z * e - 2 * e1) * scale / r;
    size[0] += s.weight * (-std::expm1(-z) / (4 * pi * r));
    size[1] += s.weight * std::abs(eps - 1 - eps * e1) * scale;
    size[2] += s.weight * std::abs(1 - 1 / eps + e1 / eps) * scale;
    size[3] += s.weight * (std::abs(gpp) + 2 * std::abs(gp / r));
  }
  return size;
}
// integrate_streamed that also sums rounding_sizes into size when given one.
template <typename Visit>
B integrate_streamed_sized(const Panel &target, double eps, double kappa,
                           B *size, Visit &&visit) {
  B current{};
  std::array<CurvedDualSample, 64> tile;
  std::size_t used = 0;
  const auto flush = [&] {
    if (used == 0)
      return;
    const auto span = std::span(tile.data(), used);
    const B value = integrate(target, span, eps, kappa);
    for (int k = 0; k < 4; ++k)
      current[k] += value[k];
    if (size) {
      const B bound = rounding_sizes(target, span, eps, kappa);
      for (int k = 0; k < 4; ++k)
        (*size)[k] += bound[k];
    }
    used = 0;
  };
  const auto accumulate = [&](const CurvedDualSample &q) {
    tile[used++] = q;
    if (used == tile.size())
      flush();
  };
  visit(accumulate);
  flush();
  return current;
}
// One source cell at a given Gauss order on its parent surface.
B integrate_at_order(const Panel &target, const NodePatchGeometry &g,
                     const CurvedNodePatchGeometry &curved, std::size_t source,
                     int order, double eps, double kappa) {
  return integrate_streamed(target, eps, kappa, [&](const auto &accumulate) {
    if (curved.parent_spheres.empty())
      visit_pn_samples(curved.pn_node_normals, curved.pn_straight_edges, g,
                       source, order, accumulate);
    else
      visit_sphere_samples(curved.parent_spheres, g, source, order, accumulate);
  });
}
// Near pairs within two cell radii of the source vertex.  The cell is split
// into rectangles of its quads' parameter squares until every rectangle is
// admissible: the bounding sphere of its samples (radius r), enlarged by
// kBoundSafety for the part of the curved piece outside the samples, lies at
// least kAdmissibleRatio of that radius from the target.  An admissible
// rectangle takes the kOrder x kOrder Gauss product.  The integrand is
// analytic in the parameters up to the target's foot on the extended patch,
// so along either parameter the nearest singularity is at least d/r
// half-lengths from the centre, and the Gauss error of n points is of order
// rho^-2n with rho = d/r + sqrt((d/r)^2 - 1) (Bernstein ellipse; Trefethen,
// Approximation Theory and Approximation Practice, theorem 19.3).  At the
// threshold d = 2 * 1.3 r, rho = 5.0 and n = 8 that is 7e-12 per rectangle;
// the earlier ratio 3 (d/r = 3.9, rho = 7.7) gave 7e-15, five orders below the
// 1e-6 the self blocks are held to and the 1e-4..1e-3 the p1 far field leaves
// in the operator.  Measured against the ratio-3 rule (18 Oct 2026): worst
// 8.5e-13 on the two-sphere oracle (gaps 0.2..0.002 A, order-256 reference,
// test/bie_curved_near_quadrature.cpp); on 1A63 and 1SH1 at h = 1.25 A every
// stored one-ring and near-zone block above 1e-3 of the largest of its kind
// moved by at most 3.1e-7 of itself (4.6e-8 of the largest block), the
// energies agree to ten digits, and the one-ring thread time fell 2.4x
// (docs/TITAN_ONE_RING_ADMISSIBILITY_2026-10-18.md; order 6 would give 4.4x
// but lets about a hundred large blocks move 1e-6..3e-6, past the self-block
// standard, so it was not taken).  The error is set by the geometry whatever
// the distance of the target to the cell: a
// target almost on the cell, as where two sheets nearly touch, only deepens
// the subdivision around its foot point.  The rule rests on the geometry,
// not on two estimates agreeing, which both miss a peak narrower than their
// samples.  A rectangle below kSmallestRectangle that is still too close
// cannot be resolved in double precision (the target is then essentially on
// the cell, a defect of the surface); it is left out and the pair rejected
// as unresolved.  gap is the smallest distance of the target from a sample
// of the cell seen on the way, which the subdivision drives down to the
// distance of the target from the cell.
// The Gauss tiles of one source cell, shared by the targets it is integrated
// against: a tile, its centre and its bounding radius depend on the source
// rectangle only, so each is formed once per cell and reused bit for bit.
struct TileMemoKey {
  std::size_t pos{0};
  int half{0};
  std::uint64_t u0{0}, u1{0}, v0{0}, v1{0};
  bool operator==(const TileMemoKey &) const = default;
};
struct TileMemoKeyHash {
  std::size_t operator()(const TileMemoKey &k) const noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint64_t x : {static_cast<std::uint64_t>(k.pos),
                                  static_cast<std::uint64_t>(k.half), k.u0, k.u1,
                                  k.v0, k.v1}) {
      h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<std::size_t>(h);
  }
};
struct TileMemoEntry {
  std::array<CurvedDualSample, 64> tile;
  std::size_t used{0};
  V centre{};
  double radius{0.0}; // before the safety factor
};
using TileMemo = std::unordered_map<TileMemoKey, TileMemoEntry, TileMemoKeyHash>;

struct AdmissibleResult {
  B value{};
  bool resolved{true};
  double gap{std::numeric_limits<double>::infinity()};
  // Work counters (diagnostic): rectangles whose Gauss tile was formed, and
  // those of them that were integrated rather than split.
  std::size_t tiles_visited{0}, tiles_integrated{0};
};
AdmissibleResult integrate_admissible(const Panel &target,
                                      const NodePatchGeometry &g,
                                      const CurvedNodePatchGeometry &curved,
                                      std::size_t source, double eps,
                                      double kappa,
                                      std::size_t skip_pos = static_cast<std::size_t>(-1),
                                      TileMemo *memo = nullptr) {
  constexpr double kAdmissibleRatio = 2.0, kBoundSafety = 1.3,
                   kSmallestRectangle = 1e-12;
  constexpr std::size_t kMaxSplits = std::size_t{1} << 20;
  constexpr int kOrder = 8;
  const V x{target.x, target.y, target.z};
  struct Pending {
    std::size_t pos;
    int half;
    QuadRect rect;
  };
  std::vector<Pending> stack;
  for (auto pos = g.incident_offsets[source]; pos < g.incident_offsets[source + 1];
       ++pos)
    for (int half = 0; pos != skip_pos && half < 2; ++half)
      stack.push_back({pos, half, QuadRect{}});
  AdmissibleResult result;
  std::array<CurvedDualSample, kOrder * kOrder> tile;
  std::size_t splits = 0;
  while (!stack.empty()) {
    const Pending piece = stack.back();
    stack.pop_back();
    ++result.tiles_visited;
    // Form the tile with its centre and radius, or take them from the memo.
    const auto form = [&](std::array<CurvedDualSample, kOrder * kOrder> &into,
                          std::size_t &count, V &centre_out, double &radius_out) {
      count = 0;
      const auto collect = [&](const CurvedDualSample &q) { into[count++] = q; };
      if (curved.parent_spheres.empty())
        visit_pn_quad(curved.pn_node_normals, curved.pn_straight_edges, g, source,
                      piece.pos, piece.half, piece.rect, kOrder, collect);
      else
        visit_sphere_quad(curved.parent_spheres, g, source, piece.pos, piece.half,
                          piece.rect, kOrder, collect);
      V c{};
      for (std::size_t k = 0; k < count; ++k)
        c = add(c, into[k].xyz);
      c = mul(c, 1.0 / static_cast<double>(count));
      double r = 0.0;
      for (std::size_t k = 0; k < count; ++k)
        r = std::max(r, norm(sub(into[k].xyz, c)));
      centre_out = c;
      radius_out = r;
    };
    const CurvedDualSample *samples = tile.data();
    std::size_t used = 0;
    V centre{};
    double radius = 0.0;
    if (memo != nullptr) {
      const TileMemoKey memo_key{piece.pos, piece.half,
                                 std::bit_cast<std::uint64_t>(piece.rect.u0),
                                 std::bit_cast<std::uint64_t>(piece.rect.u1),
                                 std::bit_cast<std::uint64_t>(piece.rect.v0),
                                 std::bit_cast<std::uint64_t>(piece.rect.v1)};
      auto [entry, inserted] = memo->try_emplace(memo_key);
      if (inserted)
        form(entry->second.tile, entry->second.used, entry->second.centre,
             entry->second.radius);
      samples = entry->second.tile.data();
      used = entry->second.used;
      centre = entry->second.centre;
      radius = entry->second.radius;
    } else {
      form(tile, used, centre, radius);
    }
    for (std::size_t k = 0; k < used; ++k)
      result.gap = std::min(result.gap, norm(sub(x, samples[k].xyz)));
    radius *= kBoundSafety;
    const double distance = norm(sub(x, centre));
    if (distance >= kAdmissibleRatio * radius) {
      ++result.tiles_integrated;
      const B value = integrate(target, std::span(samples, used), eps, kappa);
      for (int c = 0; c < 4; ++c)
        result.value[c] += value[c];
      continue;
    }
    if (radius < kSmallestRectangle || ++splits > kMaxSplits) {
      result.resolved = false;
      continue;
    }
    const double um = 0.5 * (piece.rect.u0 + piece.rect.u1),
                 vm = 0.5 * (piece.rect.v0 + piece.rect.v1);
    for (int k = 0; k < 4; ++k)
      stack.push_back({piece.pos, piece.half,
                       QuadRect{k & 1 ? um : piece.rect.u0,
                                k & 1 ? piece.rect.u1 : um,
                                k & 2 ? vm : piece.rect.v0,
                                k & 2 ? piece.rect.v1 : vm}});
  }
  return result;
}
// Gauss order of the near-zone pairs beyond two cell radii on the FMM route.
// The stored samples (order 2 by default) are sized for the energy sum; two to
// six radii from a cell they leave about 1e-3 in the Kirkwood energy, order 4
// about 1e-5 (test/tabi_node_patch_kirkwood.py).
constexpr int kFmmNearZoneOrder = 4;
// A pair by admissible subdivision. An unresolved rectangle invalidates the
// whole pair: a partial integral must never enter the operator or its cache.
B admissible_pair(const Panel &target, const NodePatchGeometry &g,
                  const CurvedNodePatchGeometry &curved, std::size_t source,
                  double eps, double kappa) {
  const AdmissibleResult r =
      integrate_admissible(target, g, curved, source, eps, kappa);
  curved.near_adaptive_evaluations.fetch_add(1, std::memory_order_relaxed);
  double smallest = curved.near_min_gap.load(std::memory_order_relaxed);
  while (r.gap < smallest &&
         !curved.near_min_gap.compare_exchange_weak(smallest, r.gap,
                                                    std::memory_order_relaxed)) {
  }
  if (!r.resolved) {
    const std::size_t earlier =
        curved.near_unresolved_evaluations.fetch_add(1, std::memory_order_relaxed);
    if (earlier == 0)
      std::cerr << "TABI curved near: rejecting unresolved near pair (target "
                   "within " << r.gap << " A of source cell " << source << ")\n";
    throw std::runtime_error("curved_near_quadrature_unresolved");
  }
  return r.value;
}
// The self block of a cell that collocates off its vertex: the two quads of
// its host patch, which meet on the diagonal through the collocation point at
// (u, v) = (1/2, 1) of half 0 and (1/2, 0) of half 1, by the Duffy rule
// there; the rest of the cell, at a distance from the point, by admissible
// subdivision.
// The part of an off-vertex cell away from its host patch, by admissible
// subdivision; it does not depend on the Gauss order of the singular rule.
B offset_rest(const Panel &target, const NodePatchGeometry &g,
              const CurvedNodePatchGeometry &curved, std::size_t i, double eps,
              double kappa) {
  const AdmissibleResult rest = integrate_admissible(
      target, g, curved, i, eps, kappa, curved.cells.at(i).host_pos);
  if (!rest.resolved)
    throw std::runtime_error("curved_near_quadrature_unresolved");
  return rest.value;
}
// size takes the rounding sizes of the singular part only: the rest is the
// same sum at every order, so it does not enter the change between orders.
// rest, when given, is offset_rest computed once for the order sequence.
B offset_self_block(const Panel &target, const NodePatchGeometry &g,
                    const CurvedNodePatchGeometry &curved, std::size_t i,
                    int order, double eps, double kappa, B *size = nullptr,
                    const B *rest = nullptr) {
  const std::size_t host = curved.cells.at(i).host_pos;
  B sum = integrate_streamed_sized(target, eps, kappa, size,
                                   [&](const auto &accumulate) {
    visit_pn_quad_duffy(curved.pn_node_normals, curved.pn_straight_edges, g, i,
                        host, 0, 0.5, 1.0, order, accumulate);
    visit_pn_quad_duffy(curved.pn_node_normals, curved.pn_straight_edges, g, i,
                        host, 1, 0.5, 0.0, order, accumulate);
  });
  const B away = rest ? *rest : offset_rest(target, g, curved, i, eps, kappa);
  for (int c = 0; c < 4; ++c)
    sum[c] += away[c];
  return sum;
}
// The self block of cell i by the Gauss sequence base, 2 base, ... 128 (the
// polar Duffy rule at the collocation point for a cell that collocates off
// its vertex, the polar vertex fan otherwise), accepted when two orders agree
// (self_orders_agree: each kernel held to its own size).  A block not
// converged by 128 stops the lift (the caller refuses it).
struct SelfBlock {
  B value{}, previous{};
  int order{0};
  double change{0.0};
  bool converged{false};
  int fallback_step{0}; // >0: accepted by the second rule at this step
};
// A self block at one order, with the rounding_sizes of its order-dependent
// part (the singular rule; an offset cell's rest is the same at every order).
struct SelfValue {
  B value{}, size{};
};
// Two orders of a self block agree when every coefficient moved by at most
// 1e-6 of itself, or by no more than the rounding of its own sum allows,
// 64 ulp of its rounding size (the larger of the two orders'): a kernel that
// cancels to nearly zero, as K2 and K3 over a nearly flat cell (1NEQ: 6e-14
// and 7e-16, moving by 2e-15 between orders), is judged against its own
// terms, never against another kernel.  change is the largest move of a
// coefficient over its tolerance, scaled so that 1e-6 is the bound.
bool self_orders_agree(const SelfValue &current, const SelfValue &previous,
                       double &change) {
  bool agree = true;
  change = 0.0;
  for (int c = 0; c < 4; ++c) {
    const double d = std::abs(current.value[c] - previous.value[c]);
    const double floor =
        64 * std::numeric_limits<double>::epsilon() *
        std::max(current.size[c], previous.size[c]);
    const double tolerance = 1e-6 * std::abs(current.value[c]) + floor;
    agree &= d <= tolerance;
    change = std::max(change, 1e-6 * d / std::max(tolerance, 1e-300));
  }
  return agree;
}
// The self block of cell i at one Gauss order, with its rounding sizes when
// sized (otherwise they stay zero).
SelfValue self_block_in(const Panel &target, const NodePatchGeometry &g,
                        const CurvedNodePatchGeometry &curved, std::size_t i,
                        bool exact, int order, double eps, double kappa,
                        bool sized, const B *rest = nullptr) {
  SelfValue out;
  B *size = sized ? &out.size : nullptr;
  if (curved.cells.at(i).offset) {
    out.value =
        offset_self_block(target, g, curved, i, order, eps, kappa, size, rest);
    return out;
  }
  out.value = integrate_streamed_sized(
      target, eps, kappa, size, [&](const auto &accumulate) {
        if (exact)
          visit_sphere_samples(curved.parent_spheres, g, i, order, accumulate);
        else
          visit_pn_samples_polar(curved.pn_node_normals,
                                 curved.pn_straight_edges, g, i, order,
                                 accumulate);
      });
  return out;
}
// The star of cell i - its incident triangles and their vertices - moved so
// that the cell's vertex is the origin (local node 0).  Near the collocation
// point a sample is then known to the rounding of its own small coordinates.
// In place, the difference x - y of two coordinates of some 20 A loses the
// digits the singular kernel needs: at order 128 the nearest Gauss nodes lie
// ~1e-5 A from the point, n.(x - y) ~ r^2 is ~1e-11 A^2, and the ~4e-15 A
// rounding of absolute coordinates made a self block drift with the order
// instead of converging (1A63 cell 20946: K2 moved 8e-11, 1.6e-10, 2.4e-9
// from 16 to 128).  The geometry is the same up to that translation; an off-
// vertex collocation point is taken again on the moved host patch.
struct LocalStar {
  NodePatchGeometry g;
  CurvedNodePatchGeometry curved;
  Panel target;
};
void make_local_star(const NodePatchGeometry &g,
                     const CurvedNodePatchGeometry &curved, std::size_t i,
                     const Panel &target, LocalStar &star) {
  const V o{g.vertex_xyz[i][0], g.vertex_xyz[i][1], g.vertex_xyz[i][2]};
  std::vector<std::size_t> global{i};
  const auto local_of = [&](std::size_t v) {
    const auto at = std::find(global.begin(), global.end(), v);
    if (at != global.end())
      return static_cast<std::size_t>(at - global.begin());
    global.push_back(v);
    return global.size() - 1;
  };
  const std::size_t begin = g.incident_offsets[i], end = g.incident_offsets[i + 1];
  for (std::size_t pos = begin; pos < end; ++pos) {
    const std::size_t ti = g.incident_triangles[pos];
    NodePatchTriangle t = g.triangles[ti];
    for (auto &v : t.vertices)
      v = local_of(v);
    star.g.triangles.push_back(t);
    star.g.incident_triangles.push_back(pos - begin);
    if (!curved.pn_straight_edges.empty())
      star.curved.pn_straight_edges.push_back(curved.pn_straight_edges[ti]);
    if (!curved.parent_spheres.empty()) {
      auto s = curved.parent_spheres[ti];
      for (int k = 0; k < 3; ++k)
        s[k] -= o[k];
      star.curved.parent_spheres.push_back(s);
    }
  }
  for (const std::size_t v : global) {
    star.g.vertex_xyz.push_back({g.vertex_xyz[v][0] - o[0],
                                 g.vertex_xyz[v][1] - o[1],
                                 g.vertex_xyz[v][2] - o[2]});
    if (!curved.pn_node_normals.empty())
      star.curved.pn_node_normals.push_back(curved.pn_node_normals[v]);
  }
  star.g.incident_offsets.assign(global.size() + 1, end - begin);
  star.g.incident_offsets[0] = 0;
  star.curved.representation = curved.representation;
  star.curved.cells.resize(1);
  const auto &cell = curved.cells.at(i);
  star.curved.cells[0].offset = cell.offset;
  star.curved.cells[0].host_pos = cell.host_pos - begin;
  star.target = target;
  star.target.x -= o[0];
  star.target.y -= o[1];
  star.target.z -= o[2];
  if (cell.offset) {
    const PnQuad q = pn_quad(star.curved.pn_node_normals,
                             star.curved.pn_straight_edges, star.g, 0,
                             star.curved.cells[0].host_pos, 0);
    V P, normal;
    double area;
    pn_quad_point(q, 0.5, 1.0, P, normal, area);
    star.target.x = P[0];
    star.target.y = P[1];
    star.target.z = P[2];
    star.target.nx = normal[0];
    star.target.ny = normal[1];
    star.target.nz = normal[2];
  }
}
SelfValue self_block_at(const LocalStar &star, bool exact, int order,
                        double eps, double kappa, bool sized = true,
                        const B *rest = nullptr) {
  return self_block_in(star.target, star.g, star.curved, 0, exact, order, eps,
                       kappa, sized, rest);
}
// The self block of a PN cell on its quads split into 2^level x 2^level
// rectangles, for a block the Gauss sequence leaves moving at 128: on a cell
// of slivers (6LZG has triangles of 0.3-1 degrees) the patches vary so fast
// that a single rule converges only algebraically there (K4 moving 6e-5 at
// 128, about 37x per doubling, so order 512 would be needed).  The
// rectangle that holds the singular point takes the polar rule there - the
// vertex fan on [0, h] x [v0, v1], or the polar Duffy rule at the offset
// point s, which is a corner of two rectangles of each host half - and every
// other rectangle the tensor Gauss rule, at least one of its own sizes from
// the point.  The levels go on until two agree as the orders must
// (self_orders_agree), so a result is still accepted only on agreement.
SelfValue graded_self_block(const LocalStar &star, int level, double eps,
                            double kappa, const B *rest) {
  constexpr int kOrder = 16;
  const auto &curved = star.curved;
  const auto &g = star.g;
  const auto &cell = curved.cells[0];
  const int n = 1 << level;
  const double h = 1.0 / n;
  SelfValue out;
  out.value = integrate_streamed_sized(
      star.target, eps, kappa, &out.size, [&](const auto &accumulate) {
        for (auto pos = g.incident_offsets[0]; pos < g.incident_offsets[1];
             ++pos) {
          if (cell.offset && pos != cell.host_pos)
            continue; // the rest, by admissible subdivision
          for (int half = 0; half < 2; ++half) {
            const PnQuad q = pn_quad(curved.pn_node_normals,
                                     curved.pn_straight_edges, g, 0, pos, half);
            const double su = 0.5, sv = half == 0 ? 1.0 : 0.0;
            for (int i = 0; i < n; ++i)
              for (int j = 0; j < n; ++j) {
                const QuadRect r{i * h, (i + 1) * h, j * h, (j + 1) * h};
                if (!cell.offset && i == 0)
                  visit_pn_rect_polar(q, r.u1, r.v0, r.v1, kOrder, accumulate);
                else if (cell.offset && r.u0 <= su && su <= r.u1 &&
                         r.v0 <= sv && sv <= r.v1)
                  visit_pn_rect_duffy(q, r, su, sv, kOrder, accumulate);
                else
                  visit_pn_quad(curved.pn_node_normals,
                                curved.pn_straight_edges, g, 0, pos, half, r,
                                kOrder, accumulate);
              }
          }
        }
      });
  if (rest)
    for (int c = 0; c < 4; ++c)
      out.value[c] += (*rest)[c];
  return out;
}
constexpr int kGradedMaxLevel = 5;
// The self block of an off-vertex cell with its host part by the barycentric
// polar Duffy rule (visit_pn_host_duffy_bary) and the rest added.
SelfValue bary_self_block(const LocalStar &star, int order, double eps,
                          double kappa, const B &rest) {
  const PnQuad q = pn_quad(star.curved.pn_node_normals,
                           star.curved.pn_straight_edges, star.g, 0,
                           star.curved.cells[0].host_pos, 0);
  SelfValue out;
  out.value = integrate_streamed_sized(
      star.target, eps, kappa, &out.size, [&](const auto &accumulate) {
        visit_pn_host_duffy_bary(q, order, accumulate);
      });
  for (int c = 0; c < 4; ++c)
    out.value[c] += rest[c];
  return out;
}
SelfBlock converged_self_block(const Panel &target, const NodePatchGeometry &g,
                               const CurvedNodePatchGeometry &curved,
                               std::size_t i, bool exact, int base, double eps,
                               double kappa) {
  LocalStar star;
  make_local_star(g, curved, i, target, star);
  // The rounding sizes cost as much as the block, so they are summed only
  // once a block fails the plain 1e-6 test (sizes zero): from then on, and
  // for the order it is compared with.
  bool sized = false;
  // An off-vertex cell's part away from the host patch, once for all orders.
  const bool offset = star.curved.cells[0].offset;
  const B rest = offset ? offset_rest(star.target, star.g, star.curved, 0, eps,
                                      kappa)
                        : B{};
  const auto block = [&](int order) {
    return self_block_at(star, exact, order, eps, kappa, sized,
                         offset ? &rest : nullptr);
  };
  SelfBlock result;
  SelfValue previous = block(base);
  for (int order = 2 * base; order <= 128; order *= 2) {
    SelfValue current = block(order);
    double change = 0.0;
    bool converged = self_orders_agree(current, previous, change);
    if (!converged && !sized) {
      sized = true;
      previous = block(order / 2);
      current = block(order);
      converged = self_orders_agree(current, previous, change);
    }
    result = {current.value, previous.value, order, change, converged};
    if (converged)
      return result;
    previous = current;
  }
  // Still moving at 128: a second rule of its own kind - for an off-vertex
  // cell the polar Duffy rule in the host's barycentric coordinates (orders
  // 16 .. 128), for a vertex cell the graded rule (levels 1 .. 5) - accepted
  // when two of its own steps agree, or when one of them agrees with the
  // order-128 Gauss value: two rules with different nodes giving one value to
  // the same per-kernel tolerance.
  if (!exact) {
    const SelfValue gauss = previous; // order 128, sized
    const auto second = [&](int step) {
      return offset ? bary_self_block(star, 8 << step, eps, kappa, rest)
                    : graded_self_block(star, step, eps, kappa, nullptr);
    };
    const int last = offset ? 4 : kGradedMaxLevel; // orders 16..128 or levels
    SelfValue coarse = second(1);
    for (int step = 2; step <= last; ++step) {
      const SelfValue fine = second(step);
      double change = 0.0, cross = 0.0;
      const bool own = self_orders_agree(fine, coarse, change);
      const bool with_gauss = !own && self_orders_agree(fine, gauss, cross);
      if (own || with_gauss) {
        result = {fine.value, own ? coarse.value : gauss.value, 128,
                  own ? change : cross, true, step};
        return result;
      }
      coarse = fine;
    }
  }
  return result;
}
B geometric_pair(const Panel &target, const NodePatchGeometry &g,
                 const CurvedNodePatchGeometry &curved, std::size_t source,
                 double eps, double kappa) {
  const auto &cell = curved.cells.at(source);
  const V x{target.x, target.y, target.z};
  const double distance = norm(sub(x, cell.reference));
  if (distance > 2 * cell.radius)
    return curved.one_point_beyond_radii > 0.0
               ? integrate_at_order(target, g, curved, source,
                                    kFmmNearZoneOrder, eps, kappa)
               : integrate(target, cell.samples, eps, kappa);
  // Use the same geometric resolution rule on FMM, classic and direct paths.
  // Agreement of two global rules can miss an arbitrarily narrow peak.
  return admissible_pair(target, g, curved, source, eps, kappa);
}
} // namespace
bool prepare_curved_node_patches(const std::vector<Panel> &triangles,
                                 std::vector<Panel> &nodes, double eps,
                                 double kappa, std::size_t budget,
                                 TitanExecutor &executor, bool eager,
                                 int sample_order,
                                 double one_point_beyond_radii) {
  // Gauss points per direction in each of the two quads a dual cell takes
  // from every incident triangle: about 12*order^2 stored samples per cell.
  // They feed the far moments, the cached near pairs and the energy; order
  // 4 is the historical default.
  if (sample_order < 2 || sample_order > 4)
    throw std::invalid_argument("curved sample order must be 2, 3 or 4");
  if (nodes.empty())
    return false;
  // Exact spherical parents when every triangle carries sphere-patch
  // quadrature; otherwise a PN-triangle lift from the mesh itself, which
  // needs only the vertex positions and the node normals that
  // build_node_patch_panels has already set: the analytic SES normal where
  // the surface engine supplied one, the area-weighted star normal otherwise.
  // TITAN_CURVED_NODE_PATCH = auto (default) | exact | pn | off selects the
  // representation for A/B work: "off" returns false so the flat legacy cell
  // runs, "pn" forces the mesh lift even where exact spheres exist, which is
  // how the lift is checked against a known answer.
  static const std::string mode = [] {
    const char *raw = std::getenv("TITAN_CURVED_NODE_PATCH");
    return std::string(raw ? raw : "auto");
  }();
  if (mode == "off")
    return false;
  // First Gauss order of the self-block sequence (converged_self_block: base,
  // 2 base, ... 128, accepted when two orders agree).  The acceptance test is
  // the same from any base, so the base only decides where the sequence
  // starts looking: from 8 the cells that used to stop at 32 stop at 16 with
  // the same two-order agreement, and the energy does not move (1A63 h=1.25:
  // identical to ten digits, self-block thread time 3.5x lower).
  // TITAN_CURVED_RING_ORDER = 8 | 16 | 32 | 64 changes it for A/B work.
  static const int ring_order = [] {
    const char *raw = std::getenv("TITAN_CURVED_RING_ORDER");
    const int order = raw ? std::atoi(raw) : 8;
    return order == 16 || order == 32 || order == 64 ? order : 8;
  }();
  const bool have_spheres =
      std::all_of(triangles.begin(), triangles.end(), [](const Panel &p) {
        return bool(p.sphere_patch_quadrature);
      });
  if (mode == "exact" && !have_spheres)
    return false;
  const bool exact = have_spheres && mode != "pn";
  if (!exact)
    for (const Panel &p : nodes)
      if (!std::isfinite(p.nx + p.ny + p.nz) ||
          !(p.nx * p.nx + p.ny * p.ny + p.nz * p.nz > 0.5))
        return false; // no usable node normal: keep the flat legacy cell
  auto g =
      std::make_shared<NodePatchGeometry>(*nodes.front().node_patch_geometry);
  auto curved = std::make_shared<CurvedNodePatchGeometry>();
  // Wall time of each stage, logged at the end.
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  Clock::time_point t_repair = t_start, t_flat = t_start;
  PnPairCache pair_cache;
  // The incident position of the patch a corner cell collocates in, or npos.
  constexpr std::size_t npos = static_cast<std::size_t>(-1);
  std::vector<std::size_t> offset_host(nodes.size(), npos);
  if (exact) {
    curved->parent_spheres.reserve(triangles.size());
    for (const Panel &triangle : triangles)
      curved->parent_spheres.push_back(sphere_patch_sphere(triangle));
  } else {
    curved->pn_node_normals.reserve(nodes.size());
    for (const Panel &node : nodes)
      curved->pn_node_normals.push_back({node.nx, node.ny, node.nz});
    // Vertex normals for the patches and the collocation alike; a vertex
    // whose analytic normal leaves a bad patch takes the star normal (the
    // flat cell's source normal, set by build_node_patch_panels) and beyond.
    std::vector<std::array<double, 3>> star;
    star.reserve(nodes.size());
    for (const Panel &node : nodes)
      star.push_back({node.source_nx, node.source_ny, node.source_nz});
    PnRepairReport repair;
    const auto residual = pn_repair_normals(curved->pn_node_normals, star, *g,
                                            repair, &executor, &pair_cache);
    t_repair = Clock::now();
    if (!residual.empty()) {
      PnLiftReport lift;
      curved->pn_straight_edges =
          pn_flatten_residual(curved->pn_node_normals, *g, residual, lift,
                              &executor, &pair_cache);
      curved->pn_residual_flat_triangles = lift.flattened;
      curved->pn_straightened_edges = lift.straightened;
      repair.flat_contacts = lift.flat_contacts;
    }
    t_flat = Clock::now();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].nx = curved->pn_node_normals[i][0];
      nodes[i].ny = curved->pn_node_normals[i][1];
      nodes[i].nz = curved->pn_node_normals[i][2];
    }
    // A vertex is a corner when an edge through it is straight (in a flat
    // triangle or beside one): that patch's tangent plane there is not the
    // plane normal to the vertex normal.  Its cell collocates at a smooth
    // point of its largest incident patch, with that patch's own normal, so
    // the equation for the normal derivative is set where the derivative
    // has a meaning; the vertex keeps its normal for the patches.
    const auto &edges = curved->pn_straight_edges;
    for (std::size_t i = 0; !edges.empty() && i < nodes.size(); ++i) {
      bool corner = false;
      double best_area = -1.0;
      std::size_t host = 0;
      for (auto pos = g->incident_offsets[i]; pos < g->incident_offsets[i + 1];
           ++pos) {
        const auto ti = g->incident_triangles[pos];
        const auto &t = g->triangles[ti];
        int local = 0;
        while (t.vertices[local] != i)
          ++local;
        // The two edges through the vertex: local -> local+1, local-1 -> local.
        corner = corner || (edges[ti] >> local & 1U) ||
                 (edges[ti] >> ((local + 2) % 3) & 1U);
        const double area =
            norm(cross(sub(g->vertex_xyz[t.vertices[1]], g->vertex_xyz[t.vertices[0]]),
                       sub(g->vertex_xyz[t.vertices[2]], g->vertex_xyz[t.vertices[0]])));
        if (area > best_area) {
          best_area = area;
          host = pos;
        }
      }
      if (!corner)
        continue;
      // (2/3, 1/6, 1/6) is (u, v) = (1/2, 1) on half 0 and (1/2, 0) on half 1.
      const PnQuad q = pn_quad(curved->pn_node_normals, edges, *g, i, host, 0);
      V P, normal;
      double area;
      pn_quad_point(q, 0.5, 1.0, P, normal, area);
      if (!std::isfinite(P[0] + P[1] + P[2] + normal[0] + normal[1] + normal[2]))
        throw std::runtime_error("curved_pn_offset_collocation_invalid");
      curved->pn_max_collocation_offset =
          std::max(curved->pn_max_collocation_offset, norm(sub(P, g->vertex_xyz[i])));
      nodes[i].x = P[0];
      nodes[i].y = P[1];
      nodes[i].z = P[2];
      nodes[i].nx = normal[0];
      nodes[i].ny = normal[1];
      nodes[i].nz = normal[2];
      offset_host[i] = host;
      ++curved->pn_corner_vertices;
    }
    curved->pn_star_normal_vertices = repair.star;
    curved->pn_turned_normal_vertices = repair.turned;
    curved->pn_repair_rounds = repair.rounds;
    curved->pn_flat_contacts = repair.flat_contacts;
    curved->pn_max_shared_edge_gap =
        pn_shared_edge_gap(curved->pn_node_normals, curved->pn_straight_edges, *g);
    std::cerr << "TABI curved PN: " << curved->pn_star_normal_vertices << " of "
              << nodes.size() << " vertices took the star normal ("
              << curved->pn_turned_normal_vertices
              << " turned further toward their faces) in "
              << curved->pn_repair_rounds << " round(s); "
              << curved->pn_residual_flat_triangles << " triangles flat, "
              << curved->pn_straightened_edges
              << " neighbouring edges straight, "
              << curved->pn_corner_vertices
              << " corner vertices collocating off the vertex (largest offset "
              << curved->pn_max_collocation_offset
              << " A); largest shared-edge gap "
              << curved->pn_max_shared_edge_gap << " A\n";
    if (!(curved->pn_max_shared_edge_gap <= 1e-9))
      throw std::runtime_error("curved_pn_shared_edge_gap");
  }
  const auto t_lift = Clock::now();
  curved->representation = exact ? "curved_dual" : "curved_pn";
  curved->self_base_order = ring_order;
  const std::size_t n = nodes.size();
  curved->cells.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto &cell = curved->cells[i];
    cell.reference = {nodes[i].x, nodes[i].y, nodes[i].z};
    cell.offset = offset_host[i] != npos;
    cell.host_pos = cell.offset ? offset_host[i] : 0;
  }
  curved->eps_ratio = eps;
  curved->kappa = kappa;
  if (one_point_beyond_radii > 0.0) {
    curved->one_point_beyond_radii = one_point_beyond_radii;
    eager = false;
  }
  // Reconstruct collocation normals from the exact parent surfaces. A crease
  // does not have a unique smooth-surface normal and must not be averaged away.
  // PN mode skips this entirely and keeps the node normals it was given.
  for (std::size_t i = 0; exact && i < n; ++i) {
    V normal{};
    bool first = true;
    for (auto pos = g->incident_offsets[i]; pos < g->incident_offsets[i + 1];
         ++pos) {
      const auto ti = g->incident_triangles[pos];
      const auto &s = sphere_patch_sphere(triangles[ti]);
      const V v = mul(sub(g->vertex_xyz[i], {s[0], s[1], s[2]}), 1 / s[3]);
      if (!first && norm(sub(normal, v)) > 1e-7)
        throw std::runtime_error("curved_dual_nonunique_vertex_normal");
      normal = v;
      first = false;
      const auto &t = g->triangles[ti];
      int local = 0;
      while (t.vertices[local] != i)
        ++local;
      V origin{s[0], s[1], s[2]},
          a = sub(g->vertex_xyz[i], origin),
          b = sub(g->vertex_xyz[t.vertices[(local + 1) % 3]], origin),
          c = sub(g->vertex_xyz[t.vertices[(local + 2) % 3]], origin);
      for (const auto &corner :
           std::array<V, 3>{add(a, b), add(a, c), add(add(a, b), c)}) {
        V mapped = mul(corner, s[3] / norm(corner));
        // Normalized positive combinations stay in this cap: its cosine is
        // positive, so the maximum angular distance is attained at a corner.
        if (dot(mapped, a) <= 0)
          throw std::runtime_error("curved_dual_cap_exceeds_hemisphere");
        curved->cells[i].radius = std::max(curved->cells[i].radius,
                                           norm(sub(mapped, a)) * (1 + 1e-12));
      }
    }
    nodes[i].nx = normal[0];
    nodes[i].ny = normal[1];
    nodes[i].nz = normal[2];
  }
  std::vector<int> self_order(n, 0);
  std::vector<double> self_change(n, 0.0);
  std::vector<std::uint8_t> self_open(n, 0U), self_fallback(n, 0U);
  // Thread time of the self and one-ring blocks, summed over the workers.
  std::atomic<long long> self_ns{0}, ring_ns{0};
  // Work counters of the one-ring blocks (diagnostic).
  std::atomic<std::size_t> ring_pairs{0}, ring_visited{0}, ring_integrated{0},
      ring_max_visited{0};
  const auto t_cells_begin = Clock::now();
  executor.parallel_for(
      n, std::min<std::size_t>(8, executor.capacity()), 1,
      [&](std::size_t, std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) {
          auto &cell = curved->cells[i];
          cell.samples =
              exact ? samples(curved->parent_spheres, *g, i, sample_order)
                    : pn_samples(curved->pn_node_normals,
                                 curved->pn_straight_edges, *g, i,
                                 sample_order);
          if (!exact) {
            // Bounding radius for the treecode AABB: the exact path took it
            // from the parent sphere; here it is the farthest sample.
            const V x{nodes[i].x, nodes[i].y, nodes[i].z};
            for (const auto &q : cell.samples)
              cell.radius = std::max(cell.radius, norm(sub(q.xyz, x)) * (1 + 1e-12));
          }
          for (const auto &q : cell.samples) {
            cell.area += q.weight;
            cell.center = add(cell.center, mul(q.xyz, q.weight));
            cell.normal_integral =
                add(cell.normal_integral, mul(q.normal, q.weight));
          }
          cell.center = mul(cell.center, 1 / cell.area);
          // The classic treecode's moments; the FMM route does not read them.
          if (!(one_point_beyond_radii > 0.0))
            cell.moments.assign(4, {});
          for (const auto &q : cell.samples) {
            if (cell.moments.empty())
              break;
            V d = sub(q.xyz, cell.center);
            double px[6]{1}, py[6]{1}, pz[6]{1};
            for (int k = 1; k < 6; ++k) {
              px[k] = px[k - 1] * d[0];
              py[k] = py[k - 1] * d[1];
              pz[k] = pz[k - 1] * d[2];
            }
            for (int a = 0; a < 6; ++a)
              for (int b = 0; b + a < 6; ++b)
                for (int c = 0; c + a + b < 6; ++c) {
                  const auto index = kMomentSlot[a * 36 + b * 6 + c];
                  double w = q.weight * px[a] * py[b] * pz[c];
                  cell.moments[0][index] += w;
                  for (int j = 0; j < 3; ++j)
                    cell.moments[j + 1][index] += w * q.normal[j];
                }
          }
          cell.near_targets.assign(
              g->adjacent_nodes.begin() + g->adjacent_offsets[i],
              g->adjacent_nodes.begin() + g->adjacent_offsets[i + 1]);
          cell.near_targets.push_back(i);
          std::sort(cell.near_targets.begin(), cell.near_targets.end());
          // The self block by the Gauss sequence; a one-ring target, which a
          // sliver can bring close to the cell (and which, collocating off
          // its vertex, can lie inside a patch next to it), by admissible
          // subdivision.
          TileMemo memo;
          for (auto target : cell.near_targets) {
            const auto t0 = Clock::now();
            if (target == i) {
              const SelfBlock self = converged_self_block(
                  nodes[i], *g, *curved, i, exact, ring_order, eps, kappa);
              self_order[i] = self.order;
              self_change[i] = self.change;
              self_open[i] = self.converged ? 0U : 1U;
              self_fallback[i] = static_cast<std::uint8_t>(self.fallback_step);
              cell.near_blocks.push_back(self.value);
              self_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    Clock::now() - t0).count(),
                                std::memory_order_relaxed);
              continue;
            }
            const AdmissibleResult ring = integrate_admissible(
                nodes[target], *g, *curved, i, eps, kappa,
                static_cast<std::size_t>(-1), &memo);
            if (!ring.resolved)
              throw std::runtime_error("curved_near_quadrature_unresolved");
            ring_pairs.fetch_add(1, std::memory_order_relaxed);
            ring_visited.fetch_add(ring.tiles_visited, std::memory_order_relaxed);
            ring_integrated.fetch_add(ring.tiles_integrated,
                                      std::memory_order_relaxed);
            for (std::size_t seen = ring_max_visited.load(std::memory_order_relaxed);
                 ring.tiles_visited > seen &&
                 !ring_max_visited.compare_exchange_weak(
                     seen, ring.tiles_visited, std::memory_order_relaxed);) {
            }
            cell.near_blocks.push_back(ring.value);
            ring_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  Clock::now() - t0).count(),
                              std::memory_order_relaxed);
          }
        }
      });
  const auto t_cells = Clock::now();
  {
    std::array<std::size_t, 8> by_order{};  // final order 8, 16, ..., 128, other
    std::size_t fallback = 0;
    for (std::size_t i = 0; i < n; ++i) {
      int slot = 0;
      for (int o = 8; o < self_order[i] && slot < 6; o *= 2)
        ++slot;
      ++by_order[self_order[i] >= 8 && self_order[i] <= 128 &&
                         (self_order[i] & (self_order[i] - 1)) == 0
                     ? static_cast<std::size_t>(slot)
                     : 7U];
      fallback += self_fallback[i] != 0U;
    }
    std::cerr << "TABI curved work: one-ring pairs " << ring_pairs.load()
              << ", tiles formed " << ring_visited.load() << " (integrated "
              << ring_integrated.load() << ", most for one pair "
              << ring_max_visited.load() << "); self blocks by final order 8/16/32/64/128: "
              << by_order[0] << '/' << by_order[1] << '/' << by_order[2] << '/'
              << by_order[3] << '/' << by_order[4] << ", other " << by_order[7]
              << ", second rule " << fallback << "\n";
  }
  std::size_t worst_cell = n;
  for (std::size_t i = 0; i < n; ++i) {
    curved->self_max_order = std::max(curved->self_max_order, self_order[i]);
    curved->self_unconverged_blocks += self_open[i];
    curved->self_fallback_blocks += self_fallback[i] != 0U;
    curved->self_max_fallback_step =
        std::max<int>(curved->self_max_fallback_step, self_fallback[i]);
    if (self_change[i] > curved->self_worst_change) {
      curved->self_worst_change = self_change[i];
      worst_cell = i;
    }
  }
  std::cerr << "TABI curved cells: self blocks (Gauss sequence from "
            << ring_order << ") agreed by order " << curved->self_max_order
            << " at most";
  if (curved->self_fallback_blocks != 0)
    std::cerr << ", " << curved->self_fallback_blocks
              << " more by a second rule (barycentric Duffy for an off-vertex "
                 "cell, graded rectangles for a vertex cell; step "
              << curved->self_max_fallback_step << " at most)";
  std::cerr << "; " << curved->self_unconverged_blocks
            << " still moving (largest last change, against a tolerance "
               "of 1e-6: " << curved->self_worst_change << ", cell "
            << worst_cell << ")\n";
  if (curved->self_unconverged_blocks != 0) {
    // What a refused block looked like, for the first few: where the cell
    // collocates and its four coefficients along the order sequence.
    std::size_t shown = 0;
    for (std::size_t i = 0; i < n && shown < 4; ++i) {
      if (!self_open[i])
        continue;
      ++shown;
      const auto &cell = curved->cells[i];
      std::cerr << "TABI curved cells: cell " << i
                << (cell.offset ? " (collocates off its vertex)" : " (at its vertex)")
                << ", " << (g->incident_offsets[i + 1] - g->incident_offsets[i])
                << " incident triangles, area " << cell.area << " A^2; angles (deg)"
                << " at the vertex / smallest of each triangle (* host):";
      for (std::size_t pos = g->incident_offsets[i]; pos < g->incident_offsets[i + 1]; ++pos) {
        const auto &t = g->triangles[g->incident_triangles[pos]];
        std::array<double, 3> angle{};
        for (int c = 0; c < 3; ++c) {
          const auto &p0 = g->vertex_xyz[t.vertices[c]], &p1 = g->vertex_xyz[t.vertices[(c + 1) % 3]],
                     &p2 = g->vertex_xyz[t.vertices[(c + 2) % 3]];
          const V e1 = sub(V{p1[0], p1[1], p1[2]}, V{p0[0], p0[1], p0[2]}),
                  e2 = sub(V{p2[0], p2[1], p2[2]}, V{p0[0], p0[1], p0[2]});
          angle[c] = std::acos(std::clamp(dot(e1, e2) / (norm(e1) * norm(e2)), -1.0, 1.0)) * 180 / pi;
        }
        int local = 0;
        while (t.vertices[local] != i)
          ++local;
        std::cerr << " " << (cell.offset && pos == cell.host_pos ? "*" : "")
                  << std::setprecision(3) << angle[local] << "/"
                  << *std::min_element(angle.begin(), angle.end());
      }
      std::cerr << "; K1..K4 by order:";
      LocalStar star;
      make_local_star(*g, *curved, i, nodes[i], star);
      for (int order = ring_order; order <= 128; order *= 2) {
        const SelfValue b = self_block_at(star, exact, order, eps, kappa);
        std::cerr << "\n  " << order << ":";
        for (int c = 0; c < 4; ++c)
          std::cerr << " " << std::setprecision(12) << b.value[c];
        std::cerr << " (rounding sizes";
        for (int c = 0; c < 4; ++c)
          std::cerr << " " << std::setprecision(3) << b.size[c];
        std::cerr << ")";
      }
      if (!exact) {
        const B rest = cell.offset ? offset_rest(star.target, star.g,
                                                 star.curved, 0, eps, kappa)
                                   : B{};
        // The second rule, as converged_self_block tried it.
        for (int step = 1; step <= (cell.offset ? 4 : kGradedMaxLevel); ++step) {
          const SelfValue b =
              cell.offset ? bary_self_block(star, 8 << step, eps, kappa, rest)
                          : graded_self_block(star, step, eps, kappa, nullptr);
          std::cerr << "\n  " << (cell.offset ? "barycentric order " : "graded level ")
                    << (cell.offset ? 8 << step : step) << ":";
          for (int c = 0; c < 4; ++c)
            std::cerr << " " << std::setprecision(12) << b.value[c];
        }
      }
      std::cerr << std::setprecision(6) << "\n";
    }
    throw std::runtime_error("curved_self_quadrature_unconverged");
  }
  // FMM route: every pair inside a source cell's near zone (the test of
  // curved_pair_takes_one_point, on the same coordinates) is integrated once
  // here and kept with the cell beside its self and one-ring blocks, so the
  // packed near corrections, the ILU blocks and the audit rows all read one
  // value instead of each integrating it again.  Vertices are hashed on a
  // grid of the largest reach, so a query visits 27 grid cells.
  const auto t_zone_begin = Clock::now();
  if (curved->one_point_beyond_radii > 0.0 && n > 1) {
    double max_radius = 0.0;
    for (const auto &c : curved->cells)
      max_radius = std::max(max_radius, c.radius);
    const double grid = curved->one_point_beyond_radii * max_radius;
    if (grid > 0.0 && std::isfinite(grid)) {
      const auto grid_cell = [&](const Panel &p) {
        return std::array<std::int64_t, 3>{
            static_cast<std::int64_t>(std::floor(p.x / grid)),
            static_cast<std::int64_t>(std::floor(p.y / grid)),
            static_cast<std::int64_t>(std::floor(p.z / grid))};
      };
      const auto grid_key = [](const std::array<std::int64_t, 3> &c) {
        constexpr std::int64_t bias = std::int64_t{1} << 20;
        constexpr std::uint64_t mask = (std::uint64_t{1} << 21) - 1U;
        return ((static_cast<std::uint64_t>(c[0] + bias) & mask) << 42) |
               ((static_cast<std::uint64_t>(c[1] + bias) & mask) << 21) |
               (static_cast<std::uint64_t>(c[2] + bias) & mask);
      };
      std::unordered_map<std::uint64_t, std::vector<std::size_t>> buckets;
      for (std::size_t i = 0; i < n; ++i)
        buckets[grid_key(grid_cell(nodes[i]))].push_back(i);
      executor.parallel_for(
          n, executor.capacity(), 1,
          [&](std::size_t, std::size_t begin, std::size_t end) {
            std::vector<std::size_t> zone;
            for (auto j = begin; j < end; ++j) {
              auto &cell = curved->cells[j];
              const Panel &source = nodes[j];
              const double reach = curved->one_point_beyond_radii * cell.radius;
              const auto home = grid_cell(source);
              zone.clear();
              for (std::int64_t dx = -1; dx <= 1; ++dx)
                for (std::int64_t dy = -1; dy <= 1; ++dy)
                  for (std::int64_t dz = -1; dz <= 1; ++dz) {
                    const auto found = buckets.find(
                        grid_key({home[0] + dx, home[1] + dy, home[2] + dz}));
                    if (found == buckets.end())
                      continue;
                    for (const std::size_t i : found->second) {
                      const double ex = nodes[i].x - source.x,
                                   ey = nodes[i].y - source.y,
                                   ez = nodes[i].z - source.z;
                      if (!(ex * ex + ey * ey + ez * ez > reach * reach) &&
                          !std::binary_search(cell.near_targets.begin(),
                                              cell.near_targets.end(), i))
                        zone.push_back(i);
                    }
                  }
              if (zone.empty())
                continue;
              std::sort(zone.begin(), zone.end());
              // Beyond two radii geometric_pair integrates the cell at the
              // fixed near-zone order; those samples are drawn once per
              // source here rather than once per target.
              std::vector<CurvedDualSample> fixed_order;
              const auto zone_pair = [&](std::size_t i) {
                const V x{nodes[i].x, nodes[i].y, nodes[i].z};
                if (!(norm(sub(x, cell.reference)) > 2 * cell.radius))
                  return geometric_pair(nodes[i], *g, *curved, j, eps, kappa);
                if (fixed_order.empty())
                  fixed_order =
                      exact ? samples(curved->parent_spheres, *g, j,
                                      kFmmNearZoneOrder)
                            : pn_samples(curved->pn_node_normals,
                                         curved->pn_straight_edges, *g, j,
                                         kFmmNearZoneOrder);
                return integrate(nodes[i], fixed_order, eps, kappa);
              };
              std::vector<std::size_t> targets;
              std::vector<B> blocks;
              targets.reserve(cell.near_targets.size() + zone.size());
              blocks.reserve(targets.capacity());
              std::size_t a = 0, b = 0;
              while (a < cell.near_targets.size() || b < zone.size()) {
                if (b == zone.size() ||
                    (a < cell.near_targets.size() &&
                     cell.near_targets[a] < zone[b])) {
                  targets.push_back(cell.near_targets[a]);
                  blocks.push_back(cell.near_blocks[a++]);
                } else {
                  targets.push_back(zone[b]);
                  blocks.push_back(zone_pair(zone[b]));
                  ++b;
                }
              }
              cell.near_targets = std::move(targets);
              cell.near_blocks = std::move(blocks);
            }
          });
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    auto &p = nodes[i];
    auto &c = curved->cells[i];
    p.area = c.area;
    p.source_x = c.center[0];
    p.source_y = c.center[1];
    p.source_z = c.center[2];
    // The curved cell's own mean normal: with the centroid and the area it
    // gives the cell's zeroth moment up to second order in its size, which is
    // what the one-point pairs and the multipole moments sum.  The
    // collocation normal p.n stays the surface normal at the vertex.
    const double length = norm(c.normal_integral);
    p.source_nx = c.normal_integral[0] / length;
    p.source_ny = c.normal_integral[1] / length;
    p.source_nz = c.normal_integral[2] / length;
    p.has_source_quadrature_center = true;
    p.node_patch_geometry = g;
  }
  g->curved = curved;
  const auto t_cache_begin = Clock::now();
  if (eager && n && n <= budget / sizeof(B) / n) {
    curved->pair_blocks.resize(n * n);
    executor.parallel_for(
        n, std::min<std::size_t>(8, executor.capacity()), 1,
        [&](std::size_t, std::size_t begin, std::size_t end) {
          for (auto i = begin; i < end; ++i)
            for (std::size_t j = 0; j < n; ++j) {
              const auto &c = curved->cells[j];
              auto found = std::lower_bound(c.near_targets.begin(),
                                            c.near_targets.end(), i);
              curved->pair_blocks[i * n + j] =
                  found != c.near_targets.end() && *found == i
                      ? c.near_blocks[found - c.near_targets.begin()]
                      : geometric_pair(nodes[i], *g, *curved, j, eps, kappa);
            }
        });
  } else if (budget && n) {
    // Cache only accessed pairs. Bounded probing may decline admission when
    // crowded; that changes cost, never the evaluated kernel or accuracy.
    // The FMM route keeps all of its curved pairs with the cells already.
    const std::size_t per_cell =
        curved->one_point_beyond_radii > 0.0 ? 0 : 256;
    curved->sparse_capacity = std::min(budget / sizeof(CurvedKernelCacheEntry),
                                       n * std::min<std::size_t>(n, per_cell));
    if (curved->sparse_capacity)
      curved->sparse_blocks =
          std::make_unique<CurvedKernelCacheEntry[]>(curved->sparse_capacity);
  }
  {
    // Wall time of each stage; the self and one-ring figures are thread time
    // summed over the workers, as labelled.  Printed after the pair cache so
    // the line covers the whole preparation on every route.
    const auto s = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto t_now = Clock::now();
    std::cerr << std::setprecision(3) << "TABI curved timing: lift "
              << s(t_start, t_lift) << " s (normal repair " << s(t_start, t_repair)
              << ", flatten " << s(t_repair, t_flat) << ", corners and edge gap "
              << s(t_flat, t_lift) << "), cells " << s(t_cells_begin, t_cells)
              << " s (thread time: self blocks " << 1e-9 * self_ns.load()
              << " s, one-ring blocks " << 1e-9 * ring_ns.load()
              << " s), near zone " << s(t_zone_begin, t_cache_begin)
              << " s, pair cache " << s(t_cache_begin, t_now) << " s, other "
              << s(t_lift, t_cells_begin) + s(t_cells, t_zone_begin)
              << " s; PN pair proofs " << pair_cache.seconds << " s in "
              << pair_cache.calls << " call(s), " << pair_cache.proved
              << " pair(s) proved, " << pair_cache.reused << " reused\n"
              << std::setprecision(6);
  }
  curved->storage_bytes =
      sizeof(*curved) + curved->cells.capacity() * sizeof(CurvedDualCell) +
      curved->pair_blocks.capacity() * sizeof(B) +
      curved->parent_spheres.capacity() * sizeof(std::array<double, 4>) +
      curved->pn_node_normals.capacity() * sizeof(std::array<double, 3>) +
      curved->pn_straight_edges.capacity() +
      curved->sparse_capacity * sizeof(CurvedKernelCacheEntry);
  for (const auto &c : curved->cells)
    curved->storage_bytes +=
        c.samples.capacity() * sizeof(CurvedDualSample) +
        c.near_targets.capacity() * sizeof(std::size_t) +
        c.near_blocks.capacity() * sizeof(B) +
        c.moments.capacity() * sizeof(std::array<double, kCurvedMomentCount>);
  return true;
}
std::array<double, 4> curved_node_pair(const Panel &target, std::size_t i,
                                       const Panel &source, std::size_t j,
                                       double eps, double kappa) {
  (void)i;
  (void)j;
  i = target.node_patch_index;
  j = source.node_patch_index;
  const auto &g = *source.node_patch_geometry->curved;
  if (g.eps_ratio != eps || g.kappa != kappa)
    throw std::runtime_error("curved_dual_kernel_cache_parameter_mismatch");
  if (target.node_patch_geometry.get() == source.node_patch_geometry.get()) {
    if (!g.pair_blocks.empty())
      return g.pair_blocks.at(i * g.cells.size() + j);
    const auto &c = g.cells.at(j);
    auto found =
        std::lower_bound(c.near_targets.begin(), c.near_targets.end(), i);
    if (found != c.near_targets.end() && *found == i)
      return c.near_blocks[found - c.near_targets.begin()];
    // The FMM route stores its near-zone corrections once, so only the
    // adaptive pairs (within two cell radii) are worth a cache slot there;
    // a fixed-order pair is cheaper to recompute than to crowd them out.
    if (g.one_point_beyond_radii > 0.0) {
      const V x{target.x, target.y, target.z};
      const double distance = norm(sub(x, c.reference));
      if (distance > 2 * c.radius)
        return geometric_pair(target, *source.node_patch_geometry, g, j, eps,
                              kappa);
    }
    if (g.sparse_capacity) {
      const std::size_t key = i * g.cells.size() + j + 1;
      std::size_t hash = key;
      hash ^= hash >> 30;
      hash *= 0xbf58476d1ce4e5b9ULL;
      hash ^= hash >> 27;
      hash *= 0x94d049bb133111ebULL;
      hash ^= hash >> 31;
      for (std::size_t probe = 0; probe < 16; ++probe) {
        auto &slot = g.sparse_blocks[(hash + probe) % g.sparse_capacity];
        auto observed = slot.key.load(std::memory_order_acquire);
        if (observed == 0 && slot.key.compare_exchange_strong(
                                 observed, key, std::memory_order_acq_rel)) {
          try {
            slot.value = geometric_pair(target, *source.node_patch_geometry, g,
                                        j, eps, kappa);
            slot.ready.store(1, std::memory_order_release);
            return slot.value;
          } catch (...) {
            slot.ready.store(2, std::memory_order_release);
            throw;
          }
        }
        if (observed == key) {
          unsigned char ready;
          while ((ready = slot.ready.load(std::memory_order_acquire)) == 0)
            std::this_thread::yield();
          if (ready != 1)
            throw std::runtime_error("curved_dual_cached_quadrature_failed");
          return slot.value;
        }
      }
    }
  }
  return geometric_pair(target, *source.node_patch_geometry, g, j, eps, kappa);
}
CurvedNodePatchGeometry::~CurvedNodePatchGeometry() {
  const std::size_t adaptive = near_adaptive_evaluations.load();
  if (adaptive != 0)
    std::cerr << "TABI curved near: " << adaptive
              << " near-pair evaluation(s) by admissible subdivision, "
              << near_unresolved_evaluations.load()
              << " unresolved; smallest target-to-cell gap "
              << near_min_gap.load() << " A\n";
}
std::size_t curved_node_storage_bytes(const std::vector<Panel> &p) noexcept {
  return p.empty() || !p.front().node_patch_geometry ||
                 !p.front().node_patch_geometry->curved
             ? 0
             : p.front().node_patch_geometry->curved->storage_bytes;
}
} // namespace TitanPBE::Tabi

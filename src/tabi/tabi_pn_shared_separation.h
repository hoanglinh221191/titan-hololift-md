#pragma once
// Sufficient continuous separation tests for adjacent cubic Bezier triangles.
// A positive projected Jacobian plus a simple Jordan boundary gives degree
// one, hence an injective projection. Vertex-only neighbours additionally need
// disjoint tangent sectors to exclude containment. Inconclusive tests
// subdivide; no finite-width neighbourhood of a shared boundary is silently
// excluded.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
namespace TitanPBE::Tabi::pn_check {
using V = std::array<double, 3>;
using Key = std::array<double, 6>;
using Net = std::array<V, 10>;
struct Piece {
  Net p;
  std::array<Key, 3> key;
};
struct Curve {
  std::array<V, 4> p;
  std::array<Key, 2> key;
};
inline V sub(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline V mul(V a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double norm(V a) { return std::sqrt(dot(a, a)); }
inline V cross(V a, V b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
inline Key midpoint(Key a, Key b) {
  for (int k = 0; k < 6; ++k)
    a[k] = (a[k] + b[k]) * .5;
  return a;
}
inline int control(int a, int b) {
  constexpr int indices[3][3] = {{0, 3, 8}, {4, 1, 5}, {7, 6, 2}};
  return indices[a][b];
}
inline Curve edge(const Piece &p, int a, int b) {
  return {{p.p[a], p.p[control(a, b)], p.p[control(b, a)], p.p[b]},
          {p.key[a], p.key[b]}};
}
inline bool monotone(const Curve &c) {
  auto e = sub(c.p[3], c.p[0]);
  double scale = norm(e);
  if (!(scale > 0))
    return false;
  for (int k = 0; k < 3; ++k) {
    auto d = sub(c.p[k + 1], c.p[k]);
    if (!(dot(d, e) > 8192 * std::numeric_limits<double>::epsilon() *
                          std::max(1., norm(d)) * scale))
      return false;
  }
  return true;
}
inline std::array<Curve, 2> split(const Curve &c) {
  auto a = mul(add(c.p[0], c.p[1]), .5), b = mul(add(c.p[1], c.p[2]), .5),
       d = mul(add(c.p[2], c.p[3]), .5), e = mul(add(a, b), .5),
       f = mul(add(b, d), .5), m = mul(add(e, f), .5);
  auto key = midpoint(c.key[0], c.key[1]);
  return {{{{c.p[0], a, e, m}, {c.key[0], key}},
           {{m, f, d, c.p[3]}, {key, c.key[1]}}}};
}
inline double size(const Curve &c) {
  double result = 0;
  for (auto p : c.p)
    for (auto q : c.p)
      result = std::max(result, dot(sub(p, q), sub(p, q)));
  return result;
}
inline bool curve_axis(const Curve &a, const Curve &b, V axis) {
  double length = norm(axis);
  if (!(length > 1e-30))
    return false;
  auto origin = a.p[0];
  double alo = 1e300, ahi = -1e300, blo = 1e300, bhi = -1e300, scale = 1;
  for (auto p : a.p) {
    auto d = sub(p, origin);
    double x = dot(d, axis);
    alo = std::min(alo, x);
    ahi = std::max(ahi, x);
    scale = std::max(scale, norm(d));
  }
  for (auto p : b.p) {
    auto d = sub(p, origin);
    double x = dot(d, axis);
    blo = std::min(blo, x);
    bhi = std::max(bhi, x);
    scale = std::max(scale, norm(d));
  }
  double pad = 8192 * std::numeric_limits<double>::epsilon() * length * scale;
  return ahi + pad < blo || bhi + pad < alo;
}
inline bool endpoint_cones(const Curve &a, const Curve &b, int ia, int ib) {
  const auto anchor = a.p[ia ? 3 : 0];
  auto da = sub(a.p[ia ? 2 : 1], anchor),
       db = sub(b.p[ib ? 2 : 1], b.p[ib ? 3 : 0]);
  double la = norm(da), lb = norm(db);
  if (!(la > 0 && lb > 0))
    return false;
  da = mul(da, 1 / la);
  db = mul(db, 1 / lb);
  std::array<V, 3> axes{sub(da, db), V{-da[1], da[0], 0}, V{-db[1], db[0], 0}};
  for (auto axis : axes) {
    double length = norm(axis);
    if (!(length > 1e-15))
      continue;
    bool plus = true, minus = true;
    for (int k = 0; k < 4; ++k)
      if (k != (ia ? 3 : 0)) {
        auto d = sub(a.p[k], anchor);
        double value = dot(d, axis),
               pad = 8192 * std::numeric_limits<double>::epsilon() * length *
                     std::max(1., norm(d));
        plus &= value > pad;
        minus &= value < -pad;
      }
    for (int k = 0; k < 4; ++k)
      if (k != (ib ? 3 : 0)) {
        auto d = sub(b.p[k], b.p[ib ? 3 : 0]);
        double value = dot(d, axis),
               pad = 8192 * std::numeric_limits<double>::epsilon() * length *
                     std::max(1., norm(d));
        plus &= value < -pad;
        minus &= value > pad;
      }
    if (plus || minus)
      return true;
  }
  return false;
}
inline bool curves_disjoint(const Curve &a, const Curve &b, int depth,
                            std::size_t &visits) {
  if (++visits > 5000 || depth > 48)
    return false;
  if (curve_axis(a, b, {1, 0, 0}) || curve_axis(a, b, {0, 1, 0}))
    return true;
  for (const auto *c : {&a, &b})
    for (int k = 0; k < 3; ++k) {
      auto d = sub(c->p[k + 1], c->p[k]);
      if (curve_axis(a, b, {-d[1], d[0], 0}))
        return true;
    }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      if (a.key[i] == b.key[j] && endpoint_cones(a, b, i, j))
        return true;
  if (size(a) >= size(b)) {
    auto children = split(a);
    return curves_disjoint(children[0], b, depth + 1, visits) &&
           curves_disjoint(children[1], b, depth + 1, visits);
  }
  auto children = split(b);
  return curves_disjoint(a, children[0], depth + 1, visits) &&
         curves_disjoint(a, children[1], depth + 1, visits);
}
inline bool curve_simple(const Curve &c, int depth = 0) {
  if (monotone(c))
    return true;
  if (depth >= 16)
    return false;
  auto parts = split(c);
  std::size_t visits = 0;
  return curves_disjoint(parts[0], parts[1], 0, visits) &&
         curve_simple(parts[0], depth + 1) && curve_simple(parts[1], depth + 1);
}
inline bool simple_boundary(const Net &net, V normal) {
  double ln = norm(normal);
  if (!(ln > 0))
    return false;
  normal = mul(normal, 1 / ln);
  V origin = net[0], u = sub(net[1], origin);
  u = sub(u, mul(normal, dot(u, normal)));
  double lu = norm(u), scale = 0;
  if (!(lu > 0))
    return false;
  u = mul(u, 1 / lu);
  auto v = cross(normal, u);
  for (auto p : net)
    scale = std::max(scale, norm(sub(p, origin)));
  if (!(scale > 0))
    return false;
  Piece projected{net};
  for (int i = 0; i < 3; ++i)
    projected.key[i][i] = 1;
  for (auto &p : projected.p) {
    auto d = mul(sub(p, origin), 1 / scale);
    p = {dot(d, u), dot(d, v), 0};
  }
  std::array<Curve, 3> boundary{edge(projected, 0, 1), edge(projected, 1, 2),
                                edge(projected, 2, 0)};
  for (const auto &c : boundary)
    if (!curve_simple(c))
      return false;
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j) {
      std::size_t visits = 0;
      if (!curves_disjoint(boundary[i], boundary[j], 0, visits))
        return false;
    }
  return true;
}
inline bool sectors_disjoint(const Piece &a, const Piece &b, int ia, int ib) {
  auto ca = edge(a, ia, (ia + 1) % 3), da = edge(a, ia, (ia + 2) % 3),
       cb = edge(b, ib, (ib + 1) % 3), db = edge(b, ib, (ib + 2) % 3);
  // Strict homogeneous half-plane feasibility for both complete tangent cones.
  // Include pairs from the same cone: cross-cone bisectors alone are
  // incomplete.
  std::array<V, 4> raw{sub(ca.p[1], ca.p[0]), sub(da.p[1], da.p[0]),
                       sub(cb.p[0], cb.p[1]), sub(db.p[0], db.p[1])};
  auto directions = raw;
  for (auto &v : directions) {
    double n = norm(v);
    if (!(n > 0))
      return false;
    v = mul(v, 1 / n);
  }
  for (int i = 0; i < 4; ++i)
    for (int j = i; j < 4; ++j) {
      auto axis = add(directions[i], directions[j]);
      double n = norm(axis);
      if (!(n > 1e-15))
        continue;
      axis = mul(axis, 1 / n);
      bool pass = true;
      for (auto v : raw)
        pass &= dot(v, axis) > 8192 * std::numeric_limits<double>::epsilon();
      if (pass)
        return true;
    }
  return false;
}
template <class Certify, class Normal>
bool projected_union(const Piece &a, const Piece &b, Certify certify,
                     Normal normal) {
  std::vector<std::pair<int, int>> shared;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (a.key[i] == b.key[j])
        shared.push_back({i, j});
  if (shared.empty() || shared.size() > 2)
    return false;
  if (shared.size() == 2) {
    auto [a0, b0] = shared[0];
    auto [a1, b1] = shared[1];
    if (((a0 + 1) % 3 == a1) == ((b0 + 1) % 3 == b1))
      return false;
  }
  V wa{}, wb{};
  for (auto [i, j] : shared) {
    wa[i] = 1. / shared.size();
    wb[j] = 1. / shared.size();
  }
  auto na = normal(a.p, wa), nb = normal(b.p, wb);
  double la = norm(na), lb = norm(nb);
  if (!(la > 0 && lb > 0))
    return false;
  auto n = add(mul(na, 1 / la), mul(nb, 1 / lb));
  double ln = norm(n);
  if (!(ln > 1e-12))
    return false;
  n = mul(n, 1 / ln);
  auto curve =
      edge(a, shared[0].first,
           shared.size() == 2 ? shared[1].first : (shared[0].first + 1) % 3);
  auto tangent = shared.size() == 2
                     ? add(add(sub(curve.p[1], curve.p[0]),
                               mul(sub(curve.p[2], curve.p[1]), 2)),
                           sub(curve.p[3], curve.p[2]))
                     : sub(curve.p[1], curve.p[0]);
  tangent = sub(tangent, mul(n, dot(tangent, n)));
  double lt = norm(tangent);
  if (!(lt > 1e-15))
    return false;
  auto u = mul(tangent, 1 / lt), v = cross(n, u), origin = a.p[shared[0].first];
  auto plane_normal = cross(u, v);
  if (!certify(a.p, plane_normal) || !certify(b.p, plane_normal))
    return false;
  auto project = [&](const Piece &piece) {
    auto result = piece;
    for (auto &p : result.p) {
      auto d = sub(p, origin);
      p = {dot(d, u), dot(d, v), 0};
    }
    return result;
  };
  auto pa = project(a), pb = project(b);
  if (shared.size() == 1 &&
      !sectors_disjoint(pa, pb, shared[0].first, shared[0].second))
    return false;
  std::vector<Curve> boundary;
  for (const auto *piece : {&pa, &pb})
    for (int i = 0; i < 3; ++i) {
      auto c = edge(*piece, i, (i + 1) % 3);
      bool common = false;
      if (shared.size() == 2) {
        auto k0 = a.key[shared[0].first], k1 = a.key[shared[1].first];
        common = (c.key[0] == k0 && c.key[1] == k1) ||
                 (c.key[0] == k1 && c.key[1] == k0);
      }
      if (!common) {
        if (!curve_simple(c))
          return false;
        boundary.push_back(c);
      }
    }
  for (std::size_t i = 0; i < boundary.size(); ++i)
    for (std::size_t j = i + 1; j < boundary.size(); ++j) {
      std::size_t visits = 0;
      if (!curves_disjoint(boundary[i], boundary[j], 0, visits))
        return false;
    }
  return true;
}
template <class Subdivide>
std::array<Piece, 4> children(const Piece &p, Subdivide subdivide) {
  auto nets = subdivide(p.p);
  auto a = p.key[0], b = p.key[1], c = p.key[2], ab = midpoint(a, b),
       bc = midpoint(b, c), ac = midpoint(a, c);
  return {{{nets[0], {a, ab, ac}},
           {nets[1], {ab, b, bc}},
           {nets[2], {ac, bc, c}},
           {nets[3], {ab, bc, ac}}}};
}
template <class Separate, class Certify, class Normal, class Subdivide>
bool pair(const Piece &a, const Piece &b, Separate separate, Certify certify,
          Normal normal, Subdivide subdivide, int depth, std::size_t &visits) {
  if (++visits > 20000 || depth > 16)
    return false;
  if (separate(a.p, b.p, false))
    return true;
  bool shared = false;
  for (const auto &x : a.key)
    for (const auto &y : b.key)
      shared |= x == y;
  if (shared && projected_union(a, b, certify, normal))
    return true;
  if (!shared)
    return separate(a.p, b.p, true);
  auto ac = children(a, subdivide), bc = children(b, subdivide);
  for (const auto &x : ac)
    for (const auto &y : bc)
      if (!pair(x, y, separate, certify, normal, subdivide, depth + 1, visits))
        return false;
  return true;
}
} // namespace TitanPBE::Tabi::pn_check

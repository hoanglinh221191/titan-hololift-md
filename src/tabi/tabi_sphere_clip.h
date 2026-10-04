#ifndef TITAN_TABI_SPHERE_CLIP_H
#define TITAN_TABI_SPHERE_CLIP_H

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// Clean-room clipping of minor-arc triangles on a unit sphere. A covering
// ball removes n.u > c. Small-circle boundaries are approximated from the
// exposed side; the resulting great-circle triangles never enter a covering
// ball (apart from roundoff). This is a convergent surface approximation, not
// a stitched, exact small-circle BRep. No external mesher code is used.
namespace TitanPBE::Tabi::SphereClip {
using V = std::array<double, 3>;
using Triangle = std::array<V, 3>;
struct Cap {
  V n;
  double c;
};
inline V add(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline V mul(V a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline V cross(V a, V b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
inline double norm(V a) { return std::sqrt(dot(a, a)); }
inline V unit(V a) { return mul(a, 1.0 / norm(a)); }
inline double area(const Triangle &t) {
  return std::abs(
      2 * std::atan2(dot(t[0], cross(t[1], t[2])),
                     1 + dot(t[0], t[1]) + dot(t[1], t[2]) + dot(t[2], t[0])));
}
inline double maximum(const Triangle &t, V n) {
  bool inside = true;
  double best = std::max({dot(n, t[0]), dot(n, t[1]), dot(n, t[2])});
  for (int i = 0; i < 3; ++i) {
    const V a = t[i], b = t[(i + 1) % 3], g = cross(a, b);
    const double g2 = dot(g, g);
    if (dot(g, t[(i + 2) % 3]) * dot(g, n) < -1e-28 * g2)
      inside = false;
    if (g2 < 1e-28)
      continue;
    V p = add(n, mul(g, -dot(n, g) / g2));
    if (norm(p) < 1e-14)
      continue;
    p = unit(p);
    if (dot(cross(a, p), g) >= -1e-14 * g2 &&
        dot(cross(p, b), g) >= -1e-14 * g2)
      best = std::max(best, dot(n, p));
  }
  return inside ? 1.0 : best;
}
inline std::array<Triangle, 4> split(const Triangle &t) {
  const V ab = unit(add(t[0], t[1])), bc = unit(add(t[1], t[2])),
          ca = unit(add(t[2], t[0]));
  return {Triangle{t[0], ab, ca}, Triangle{t[1], bc, ab},
          Triangle{t[2], ca, bc}, Triangle{ab, bc, ca}};
}

inline std::vector<V> cut(const std::vector<V> &p, V n, double c) {
  std::vector<V> out;
  if (p.empty())
    return out;
  auto append = [&](V v) {
    if (out.empty() || norm(add(v, mul(out.back(), -1))) > 1e-13)
      out.push_back(v);
  };
  for (std::size_t i = 0; i < p.size(); ++i) {
    const V a = p[i], b = p[(i + 1) % p.size()];
    const bool ina = dot(n, a) <= c, inb = dot(n, b) <= c;
    if (ina)
      append(a);
    if (ina == inb)
      continue;
    // Normalized linear interpolation follows the minor great-circle arc.
    // Retain the exposed endpoint of the final bracket to avoid penetration.
    double lo = 0, hi = 1;
    for (int k = 0; k < 52; ++k) {
      const double m = (lo + hi) * 0.5;
      const V u = unit(add(mul(a, 1 - m), mul(b, m)));
      if ((dot(n, u) <= c) == ina)
        lo = m;
      else
        hi = m;
    }
    const double f = ina ? lo : hi;
    append(unit(add(mul(a, 1 - f), mul(b, f))));
  }
  if (out.size() > 1 && norm(add(out.front(), mul(out.back(), -1))) < 1e-13)
    out.pop_back();
  return out;
}

inline void clip(const Triangle &t, const std::vector<Cap> &caps,
                 std::vector<Triangle> &out, int boundary_levels = 1,
                 int depth = 0) {
  bool boundary = false, hidden_component = false;
  for (const auto &cap : caps) {
    const double hi = maximum(t, cap.n), lo = -maximum(t, mul(cap.n, -1));
    if (lo >= cap.c + 2e-14)
      return;
    if (hi <= cap.c + 2e-14)
      continue;
    boundary = true;
    const double vmax =
        std::max({dot(cap.n, t[0]), dot(cap.n, t[1]), dot(cap.n, t[2])});
    const double vmin =
        std::min({dot(cap.n, t[0]), dot(cap.n, t[1]), dot(cap.n, t[2])});
    // A cap or exposed island entirely inside the triangle must not be missed
    // by endpoint-only clipping. Resolve it locally before polygon clipping.
    hidden_component = hidden_component || vmax <= cap.c || vmin >= cap.c;
  }
  if (!boundary) {
    out.push_back(t);
    return;
  }
  if (depth < boundary_levels || (hidden_component && depth < 12)) {
    for (const auto &child : split(t))
      clip(child, caps, out, boundary_levels, depth + 1);
    return;
  }
  std::vector<V> polygon{t[0], t[1], t[2]};
  for (const auto &cap : caps) {
    if (polygon.size() < 3)
      return;
    double hi = -1;
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i)
      hi = std::max(hi,
                    maximum({polygon[0], polygon[i], polygon[i + 1]}, cap.n));
    if (hi <= cap.c + 2e-14)
      continue;
    if (cap.c <= 0) {
      // The exposed small cap is geodesically convex: arc chords joining
      // exact circle intersections remain on the exposed side.
      polygon = cut(polygon, cap.n, cap.c);
    } else {
      V center{0, 0, 0};
      for (V v : polygon)
        center = add(center, v);
      V lateral = add(center, mul(cap.n, -dot(center, cap.n)));
      if (norm(lateral) < 1e-13) {
        lateral = add(polygon[0], mul(cap.n, -dot(polygon[0], cap.n)));
        if (norm(lateral) < 1e-13)
          return;
      }
      const V u0 =
          add(mul(cap.n, cap.c),
              mul(unit(lateral), std::sqrt(std::max(0., 1 - cap.c * cap.c))));
      const V g = add(cap.n, mul(u0, -cap.c));
      // g.u <= 0 implies n.u <= c*(u0.u) <= c for c>0.
      polygon = cut(polygon, g, 0);
    }
  }
  if (polygon.size() < 3)
    return;
  // Choose the fan with the best worst triangle quality; avoid unnecessary
  // centroid fans and their extra unknowns at every molecular seam.
  std::size_t anchor = 0;
  double best = -1;
  for (std::size_t a = 0; a < polygon.size(); ++a) {
    double quality = 1;
    for (std::size_t k = 1; k + 1 < polygon.size(); ++k) {
      const Triangle tri{polygon[a], polygon[(a + k) % polygon.size()],
                         polygon[(a + k + 1) % polygon.size()]};
      const double sum = 6 - 2 * (dot(tri[0], tri[1]) + dot(tri[1], tri[2]) +
                                  dot(tri[2], tri[0]));
      quality = std::min(quality, area(tri) / std::max(sum, 1e-30));
    }
    if (quality > best) {
      best = quality;
      anchor = a;
    }
  }
  for (std::size_t k = 1; k + 1 < polygon.size(); ++k) {
    Triangle tri{polygon[anchor], polygon[(anchor + k) % polygon.size()],
                 polygon[(anchor + k + 1) % polygon.size()]};
    if (area(tri) > 1e-15)
      out.push_back(tri);
  }
}
} // namespace TitanPBE::Tabi::SphereClip
#endif

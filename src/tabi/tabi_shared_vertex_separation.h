#ifndef TITAN_TABI_SHARED_VERTEX_SEPARATION_H
#define TITAN_TABI_SHARED_VERTEX_SEPARATION_H
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace TitanPBE::Tabi::detail {
namespace shared_vertex_interval {
struct Interval {
  long double lo, hi;
};
inline Interval whole() noexcept {
  return {-std::numeric_limits<long double>::infinity(),
          std::numeric_limits<long double>::infinity()};
}
inline Interval rounded(long double lo, long double hi) noexcept {
  if (!std::isfinite(lo) || !std::isfinite(hi))
    return whole();
  return {std::nextafter(lo, -std::numeric_limits<long double>::infinity()),
          std::nextafter(hi, std::numeric_limits<long double>::infinity())};
}
inline Interval sub(Interval a, Interval b) noexcept {
  return rounded(a.lo - b.hi, a.hi - b.lo);
}
inline Interval add(Interval a, Interval b) noexcept {
  return rounded(a.lo + b.lo, a.hi + b.hi);
}
inline Interval mul(Interval a, Interval b) noexcept {
  const std::array<long double, 4> p{a.lo * b.lo, a.lo * b.hi, a.hi * b.lo,
                                     a.hi * b.hi};
  for (auto v : p)
    if (!std::isfinite(v))
      return whole();
  return rounded(*std::min_element(p.begin(), p.end()),
                 *std::max_element(p.begin(), p.end()));
}
template <class Point>
inline std::array<Interval, 3> difference(const Point &a,
                                          const Point &o) noexcept {
  return {sub({a.x, a.x}, {o.x, o.x}), sub({a.y, a.y}, {o.y, o.y}),
          sub({a.z, a.z}, {o.z, o.z})};
}
inline int determinant_sign(const std::array<Interval, 3> &a,
                            const std::array<Interval, 3> &b,
                            const std::array<Interval, 3> &c) noexcept {
  const auto x = mul(a[0], sub(mul(b[1], c[2]), mul(b[2], c[1])));
  const auto y = mul(a[1], sub(mul(b[2], c[0]), mul(b[0], c[2])));
  const auto z = mul(a[2], sub(mul(b[0], c[1]), mul(b[1], c[0])));
  const auto d = add(add(x, y), z);
  return d.lo > 0   ? 1
         : d.hi < 0 ? -1
                    : 0; // 0 means unresolved, not exact coplanarity.
}
} // namespace shared_vertex_interval

// One-sided certificate: true proves that two nondegenerate triangles meet
// only at their explicitly shared vertex. False leaves the existing complete
// intersection test in control. No geometric tolerance is enlarged.
template <class Point>
inline bool certified_shared_vertex_separation(const Point &o, const Point &a,
                                               const Point &b, const Point &c,
                                               const Point &d) noexcept {
  static_assert(std::is_same_v<decltype(o.x), double> &&
                    std::is_same_v<decltype(o.y), double> &&
                    std::is_same_v<decltype(o.z), double>,
                "The certificate requires binary64 coordinate members");
#if defined(__FAST_MATH__)
  return false; // Directed interval bounds require ordinary FP semantics.
#else
  const std::array<double, 3> origin{o.x, o.y, o.z}, av{a.x, a.y, a.z},
      bv{b.x, b.y, b.z}, cv{c.x, c.y, c.z}, dv{d.x, d.y, d.z};
  for (int k = 0; k < 3; ++k) {
    if (!std::isfinite(origin[k]) || !std::isfinite(av[k]) ||
        !std::isfinite(bv[k]) || !std::isfinite(cv[k]) || !std::isfinite(dv[k]))
      return false;
  }
  for (int k = 0; k < 3; ++k) {
    if (av[k] <= origin[k] && bv[k] <= origin[k] && cv[k] >= origin[k] &&
        dv[k] >= origin[k] &&
        ((av[k] < origin[k] && bv[k] < origin[k]) ||
         (cv[k] > origin[k] && dv[k] > origin[k])))
      return true;
    if (av[k] >= origin[k] && bv[k] >= origin[k] && cv[k] <= origin[k] &&
        dv[k] <= origin[k] &&
        ((av[k] > origin[k] && bv[k] > origin[k]) ||
         (cv[k] < origin[k] && dv[k] < origin[k])))
      return true;
  }
  using namespace shared_vertex_interval;
  const auto u = difference(a, o), v = difference(b, o), w = difference(c, o),
             z = difference(d, o);
  const int abc = determinant_sign(u, v, w), abd = determinant_sign(u, v, z);
  if (abc != 0 && abc == abd)
    return true;
  const int cda = determinant_sign(w, z, u), cdb = determinant_sign(w, z, v);
  if (cda != 0 && cda == cdb)
    return true;
  // For noncoplanar planes, d=(u x v)x(w x z) has cone coefficients
  // (-det(w,z,v),det(w,z,u)) and (det(u,v,z),-det(u,v,w)).
  // Opposite ray signs mean that the two triangle intervals share only0.
  if (abc && abd && cda && cdb && abc == -abd && cda == -cdb)
    return abd == cdb;
  return false;
#endif
}
} // namespace TitanPBE::Tabi::detail
#endif

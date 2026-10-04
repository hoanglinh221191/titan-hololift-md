#pragma once

#include "algebra/mat3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace titan::algebra {

struct SymmetricEigen3 {
  std::array<double, 3> values{};
  Mat3 vectors{Mat3::identity()};
};

namespace detail {

template <std::size_t N>
using SymmetricArray = std::array<double, N * N>;

template <std::size_t N>
[[nodiscard]] constexpr double &sym_at(SymmetricArray<N> &a, std::size_t r,
                                       std::size_t c) noexcept {
  return a[r * N + c];
}

template <std::size_t N>
[[nodiscard]] constexpr double sym_at(const SymmetricArray<N> &a,
                                      std::size_t r,
                                      std::size_t c) noexcept {
  return a[r * N + c];
}

template <std::size_t N>
[[nodiscard]] SymmetricArray<N> identity_array() noexcept {
  SymmetricArray<N> out{};
  for (std::size_t i = 0; i < N; ++i)
    sym_at<N>(out, i, i) = 1.0;
  return out;
}

template <std::size_t N>
void jacobi_rotate(SymmetricArray<N> &a, SymmetricArray<N> &vectors,
                   std::size_t p, std::size_t q) noexcept {
  const double apq = sym_at<N>(a, p, q);
  if (apq == 0.0)
    return;

  const double app = sym_at<N>(a, p, p);
  const double aqq = sym_at<N>(a, q, q);
  const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  for (std::size_t k = 0; k < N; ++k) {
    if (k == p || k == q)
      continue;
    const double akp = sym_at<N>(a, k, p);
    const double akq = sym_at<N>(a, k, q);
    const double new_kp = c * akp - s * akq;
    const double new_kq = s * akp + c * akq;
    sym_at<N>(a, k, p) = new_kp;
    sym_at<N>(a, p, k) = new_kp;
    sym_at<N>(a, k, q) = new_kq;
    sym_at<N>(a, q, k) = new_kq;
  }

  const double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
  const double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
  sym_at<N>(a, p, p) = new_pp;
  sym_at<N>(a, q, q) = new_qq;
  sym_at<N>(a, p, q) = 0.0;
  sym_at<N>(a, q, p) = 0.0;

  for (std::size_t k = 0; k < N; ++k) {
    const double vkp = sym_at<N>(vectors, k, p);
    const double vkq = sym_at<N>(vectors, k, q);
    sym_at<N>(vectors, k, p) = c * vkp - s * vkq;
    sym_at<N>(vectors, k, q) = s * vkp + c * vkq;
  }
}

template <std::size_t N>
void normalize_column(SymmetricArray<N> &vectors, std::size_t col) noexcept {
  double n2 = 0.0;
  for (std::size_t r = 0; r < N; ++r)
    n2 += sym_at<N>(vectors, r, col) * sym_at<N>(vectors, r, col);
  if (n2 <= 0.0)
    return;
  const double inv_n = 1.0 / std::sqrt(n2);
  std::size_t pivot = 0;
  double pivot_abs = 0.0;
  for (std::size_t r = 0; r < N; ++r) {
    sym_at<N>(vectors, r, col) *= inv_n;
    const double a = std::abs(sym_at<N>(vectors, r, col));
    if (a > pivot_abs) {
      pivot_abs = a;
      pivot = r;
    }
  }
  if (sym_at<N>(vectors, pivot, col) < 0.0) {
    for (std::size_t r = 0; r < N; ++r)
      sym_at<N>(vectors, r, col) = -sym_at<N>(vectors, r, col);
  }
}

template <std::size_t N>
void sort_eigensystem(std::array<double, N> &values,
                      SymmetricArray<N> &vectors) noexcept {
  for (std::size_t i = 0; i < N; ++i) {
    std::size_t best = i;
    for (std::size_t j = i + 1; j < N; ++j) {
      if (values[j] < values[best])
        best = j;
    }
    if (best == i)
      continue;
    std::swap(values[i], values[best]);
    for (std::size_t r = 0; r < N; ++r)
      std::swap(sym_at<N>(vectors, r, i), sym_at<N>(vectors, r, best));
  }
}

template <std::size_t N>
void jacobi_symmetric(SymmetricArray<N> &a, SymmetricArray<N> &vectors,
                      double eps = 1.0e-14,
                      std::size_t max_sweeps = 48) noexcept {
  vectors = identity_array<N>();
  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 1;
    double max_offdiag = 0.0;
    for (std::size_t r = 0; r < N; ++r) {
      for (std::size_t c = r + 1; c < N; ++c) {
        const double value = std::abs(sym_at<N>(a, r, c));
        if (value > max_offdiag) {
          max_offdiag = value;
          p = r;
          q = c;
        }
      }
    }
    if (max_offdiag <= eps)
      break;
    jacobi_rotate<N>(a, vectors, p, q);
  }
}

} // namespace detail

[[nodiscard]] inline SymmetricEigen3 symmetric_eigen3(const Mat3 &m) noexcept {
  detail::SymmetricArray<3> a{};
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      detail::sym_at<3>(a, r, c) = 0.5 * (m(r, c) + m(c, r));

  detail::SymmetricArray<3> vectors{};
  detail::jacobi_symmetric<3>(a, vectors);

  SymmetricEigen3 out{};
  for (std::size_t i = 0; i < 3; ++i)
    out.values[i] = detail::sym_at<3>(a, i, i);
  detail::sort_eigensystem(out.values, vectors);
  for (std::size_t c = 0; c < 3; ++c) {
    detail::normalize_column<3>(vectors, c);
    for (std::size_t r = 0; r < 3; ++r)
      out.vectors(r, c) = detail::sym_at<3>(vectors, r, c);
  }
  return out;
}

[[nodiscard]] inline std::array<double, 4>
dominant_eigenvector_symmetric4(const std::array<double, 16> &m) noexcept {
  detail::SymmetricArray<4> a{};
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t c = 0; c < 4; ++c)
      detail::sym_at<4>(a, r, c) = 0.5 * (m[r * 4 + c] + m[c * 4 + r]);

  detail::SymmetricArray<4> vectors{};
  detail::jacobi_symmetric<4>(a, vectors, 1.0e-14, 64);

  std::array<double, 4> values{};
  for (std::size_t i = 0; i < 4; ++i)
    values[i] = detail::sym_at<4>(a, i, i);
  detail::sort_eigensystem(values, vectors);
  detail::normalize_column<4>(vectors, 3);

  std::array<double, 4> out{};
  for (std::size_t r = 0; r < 4; ++r)
    out[r] = detail::sym_at<4>(vectors, r, 3);
  return out;
}

} // namespace titan::algebra

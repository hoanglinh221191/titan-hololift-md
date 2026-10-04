#pragma once

#include "algebra/vec3.h"

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace titan::algebra {

namespace detail {

template <typename Matrix>
concept WritableDense3x3 =
    std::default_initializable<remove_cvref_t<Matrix>> &&
    requires(remove_cvref_t<Matrix> m) {
      typename remove_cvref_t<Matrix>::Index;
      { m(static_cast<typename remove_cvref_t<Matrix>::Index>(0),
          static_cast<typename remove_cvref_t<Matrix>::Index>(0)) };
    };

} // namespace detail

struct Mat3 {
  std::array<double, 9> v{};

  [[nodiscard]] constexpr double operator()(std::size_t r,
                                            std::size_t c) const noexcept {
    return v[r * 3 + c];
  }

  [[nodiscard]] constexpr double &operator()(std::size_t r,
                                             std::size_t c) noexcept {
    return v[r * 3 + c];
  }

  [[nodiscard]] static constexpr Mat3 identity() noexcept {
    Mat3 m{};
    m(0, 0) = 1.0;
    m(1, 1) = 1.0;
    m(2, 2) = 1.0;
    return m;
  }

  template <typename Matrix>
    requires detail::WritableDense3x3<Matrix>
  [[nodiscard]] operator Matrix() const {
    Matrix out{};
    using M = detail::remove_cvref_t<Matrix>;
    using Index = typename M::Index;
    if constexpr (requires(Matrix &m, Index rows, Index cols) {
                    m.resize(rows, cols);
                  }) {
      out.resize(static_cast<Index>(3), static_cast<Index>(3));
    }
    for (std::size_t r = 0; r < 3; ++r)
      for (std::size_t c = 0; c < 3; ++c)
        out(static_cast<Index>(r), static_cast<Index>(c)) = (*this)(r, c);
    return out;
  }
};

[[nodiscard]] constexpr Mat3 transpose(const Mat3 &m) noexcept {
  Mat3 out{};
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      out(r, c) = m(c, r);
  return out;
}

[[nodiscard]] constexpr double determinant(const Mat3 &m) noexcept {
  return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
         m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
         m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

[[nodiscard]] constexpr Vec3 mul(const Mat3 &m, Vec3 x) noexcept {
  return {m(0, 0) * x.x + m(0, 1) * x.y + m(0, 2) * x.z,
          m(1, 0) * x.x + m(1, 1) * x.y + m(1, 2) * x.z,
          m(2, 0) * x.x + m(2, 1) * x.y + m(2, 2) * x.z};
}

[[nodiscard]] constexpr Vec3 mul_row(Vec3 x, const Mat3 &m) noexcept {
  return {x.x * m(0, 0) + x.y * m(1, 0) + x.z * m(2, 0),
          x.x * m(0, 1) + x.y * m(1, 1) + x.z * m(2, 1),
          x.x * m(0, 2) + x.y * m(1, 2) + x.z * m(2, 2)};
}

[[nodiscard]] constexpr Mat3 mul(const Mat3 &a, const Mat3 &b) noexcept {
  Mat3 out{};
  for (std::size_t r = 0; r < 3; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t k = 0; k < 3; ++k)
        out(r, c) += a(r, k) * b(k, c);
  return out;
}

[[nodiscard]] inline bool is_orthonormal(const Mat3 &m,
                                         double eps = 1.0e-10) noexcept {
  const Mat3 mtm = mul(transpose(m), m);
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t c = 0; c < 3; ++c) {
      const double expected = r == c ? 1.0 : 0.0;
      if (std::abs(mtm(r, c) - expected) > eps)
        return false;
    }
  }
  return true;
}

} // namespace titan::algebra

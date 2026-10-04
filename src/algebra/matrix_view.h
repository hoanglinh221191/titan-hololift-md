#pragma once

#include <cstddef>

namespace titan::algebra {

struct Matrix3ConstView {
  const double *data{nullptr};
  std::size_t rows{0};
  std::ptrdiff_t row_stride{3};
  std::ptrdiff_t col_stride{1};

  [[nodiscard]] constexpr double operator()(std::size_t r,
                                            std::size_t c) const noexcept {
    return data[static_cast<std::ptrdiff_t>(r) * row_stride +
                static_cast<std::ptrdiff_t>(c) * col_stride];
  }
};

struct Matrix3View {
  double *data{nullptr};
  std::size_t rows{0};
  std::ptrdiff_t row_stride{3};
  std::ptrdiff_t col_stride{1};

  [[nodiscard]] constexpr double operator()(std::size_t r,
                                            std::size_t c) const noexcept {
    return data[static_cast<std::ptrdiff_t>(r) * row_stride +
                static_cast<std::ptrdiff_t>(c) * col_stride];
  }

  [[nodiscard]] constexpr double &operator()(std::size_t r,
                                             std::size_t c) noexcept {
    return data[static_cast<std::ptrdiff_t>(r) * row_stride +
                static_cast<std::ptrdiff_t>(c) * col_stride];
  }
};

} // namespace titan::algebra

#pragma once

#include "algebra/matrix_view.h"

#include <cstddef>
#include <vector>

namespace titan::algebra {

class Matrix3Buffer {
public:
  Matrix3Buffer() = default;

  explicit Matrix3Buffer(std::size_t rows) { resize(rows); }

  void resize(std::size_t rows) { data_.assign(rows * 3, 0.0); }

  [[nodiscard]] std::size_t rows() const noexcept { return data_.size() / 3; }

  [[nodiscard]] double *data() noexcept { return data_.data(); }
  [[nodiscard]] const double *data() const noexcept { return data_.data(); }

  [[nodiscard]] double operator()(std::size_t r,
                                  std::size_t c) const noexcept {
    return data_[r * 3 + c];
  }

  [[nodiscard]] double &operator()(std::size_t r,
                                   std::size_t c) noexcept {
    return data_[r * 3 + c];
  }

  [[nodiscard]] Matrix3View view() noexcept {
    return {data_.data(), rows(), 3, 1};
  }

  [[nodiscard]] Matrix3ConstView const_view() const noexcept {
    return {data_.data(), rows(), 3, 1};
  }

private:
  std::vector<double> data_;
};

} // namespace titan::algebra

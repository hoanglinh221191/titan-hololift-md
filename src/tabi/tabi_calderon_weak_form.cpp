#include "tabi_calderon_weak_form.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::calderon {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kKernelScale = 1.0 / (4.0 * kPi);

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

[[nodiscard]] Vec3 operator+(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 operator-(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 operator*(double scale, const Vec3 &value) noexcept {
  return {scale * value.x, scale * value.y, scale * value.z};
}

[[nodiscard]] double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

[[nodiscard]] double norm(const Vec3 &value) noexcept {
  return std::sqrt(dot(value, value));
}

struct TriangleGeometry {
  std::array<Vec3, 3> point{};
  std::array<std::size_t, 3> vertex{};
  Vec3 normal{};
  double area{0.0};
  std::array<Vec3, 3> surface_gradient{};
  std::array<Vec3, 3> surface_curl{};
};

[[nodiscard]] Vec3 vertex_point(const SurfaceMesh &mesh,
                                 std::size_t index) noexcept {
  const auto &vertex = mesh.vertices[index];
  return {vertex.x, vertex.y, vertex.z};
}

[[nodiscard]] bool build_triangle_geometry(const SurfaceMesh &mesh,
                                            std::size_t index,
                                            TriangleGeometry &out) {
  const Triangle &triangle = mesh.triangles[index];
  if (triangle.v0 >= mesh.vertices.size() || triangle.v1 >= mesh.vertices.size() ||
      triangle.v2 >= mesh.vertices.size()) {
    return false;
  }
  out.vertex = {triangle.v0, triangle.v1, triangle.v2};
  out.point = {vertex_point(mesh, triangle.v0), vertex_point(mesh, triangle.v1),
               vertex_point(mesh, triangle.v2)};
  const Vec3 normal_vector =
      cross(out.point[1] - out.point[0], out.point[2] - out.point[0]);
  const double twice_area = norm(normal_vector);
  if (!(twice_area > 1.0e-15) || !std::isfinite(twice_area)) {
    return false;
  }
  out.area = 0.5 * twice_area;
  out.normal = (1.0 / twice_area) * normal_vector;
  out.surface_gradient[0] =
      (1.0 / twice_area) * cross(out.normal, out.point[2] - out.point[1]);
  out.surface_gradient[1] =
      (1.0 / twice_area) * cross(out.normal, out.point[0] - out.point[2]);
  out.surface_gradient[2] =
      (1.0 / twice_area) * cross(out.normal, out.point[1] - out.point[0]);
  for (std::size_t local = 0; local < 3; ++local) {
    out.surface_curl[local] = cross(out.normal, out.surface_gradient[local]);
  }
  return true;
}

struct TriangleQuadraturePoint {
  std::array<double, 3> lambda{};
  double weight{0.0};
};

constexpr std::array<TriangleQuadraturePoint, 7> kTriangleGauss7 = {
    TriangleQuadraturePoint{{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
                           0.2250000000000000},
    TriangleQuadraturePoint{{0.4701420641051151, 0.4701420641051151,
                            0.0597158717897698},
                           0.1323941527885062},
    TriangleQuadraturePoint{{0.4701420641051151, 0.0597158717897698,
                            0.4701420641051151},
                           0.1323941527885062},
    TriangleQuadraturePoint{{0.0597158717897698, 0.4701420641051151,
                            0.4701420641051151},
                           0.1323941527885062},
    TriangleQuadraturePoint{{0.1012865073234563, 0.1012865073234563,
                            0.7974269853530873},
                           0.1259391805448272},
    TriangleQuadraturePoint{{0.1012865073234563, 0.7974269853530873,
                            0.1012865073234563},
                           0.1259391805448272},
    TriangleQuadraturePoint{{0.7974269853530873, 0.1012865073234563,
                            0.1012865073234563},
                           0.1259391805448272}};

constexpr std::array<double, 6> kGauss6Points = {
    0.03376524289842399, 0.16939530676686774, 0.38069040695840155,
    0.61930959304159845, 0.83060469323313226, 0.96623475710157603};
constexpr std::array<double, 6> kGauss6Weights = {
    0.08566224618958517, 0.18038078652406930, 0.23395696728634552,
    0.23395696728634552, 0.18038078652406930, 0.08566224618958517};

[[nodiscard]] Vec3 interpolate(const TriangleGeometry &triangle,
                                const std::array<double, 3> &lambda) noexcept {
  return lambda[0] * triangle.point[0] + lambda[1] * triangle.point[1] +
         lambda[2] * triangle.point[2];
}

[[nodiscard]] std::array<double, 3>
barycentric_coordinates(const TriangleGeometry &triangle,
                        const Vec3 &point) noexcept {
  const Vec3 v0 = triangle.point[1] - triangle.point[0];
  const Vec3 v1 = triangle.point[2] - triangle.point[0];
  const Vec3 v2 = point - triangle.point[0];
  const double d00 = dot(v0, v0);
  const double d01 = dot(v0, v1);
  const double d11 = dot(v1, v1);
  const double d20 = dot(v2, v0);
  const double d21 = dot(v2, v1);
  const double denominator = d00 * d11 - d01 * d01;
  if (std::abs(denominator) < 1.0e-300) {
    return {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  }
  const double lambda1 = (d11 * d20 - d01 * d21) / denominator;
  const double lambda2 = (d00 * d21 - d01 * d20) / denominator;
  return {1.0 - lambda1 - lambda2, lambda1, lambda2};
}

struct KernelIntegral {
  double single_constant{0.0};
  std::array<double, 3> single_basis{};
  std::array<double, 3> double_basis{};
};

void add_regular_triangle_integral(const Vec3 &target,
                                   const TriangleGeometry &source,
                                   const std::array<Vec3, 3> &subtriangle,
                                   KernelIntegral &out) {
  const Vec3 cross_vector =
      cross(subtriangle[1] - subtriangle[0], subtriangle[2] - subtriangle[0]);
  const double area = 0.5 * norm(cross_vector);
  if (!(area > 1.0e-18)) {
    return;
  }
  for (const TriangleQuadraturePoint &quadrature : kTriangleGauss7) {
    const Vec3 source_point = quadrature.lambda[0] * subtriangle[0] +
                              quadrature.lambda[1] * subtriangle[1] +
                              quadrature.lambda[2] * subtriangle[2];
    const Vec3 delta = target - source_point;
    const double distance_squared = dot(delta, delta);
    if (!(distance_squared > 1.0e-24)) {
      continue;
    }
    const double distance = std::sqrt(distance_squared);
    const double weight = area * quadrature.weight;
    const double single = weight * kKernelScale / distance;
    const double normal_derivative =
        weight * kKernelScale * dot(source.normal, delta) /
        (distance_squared * distance);
    const auto lambda = barycentric_coordinates(source, source_point);
    out.single_constant += single;
    for (std::size_t local = 0; local < 3; ++local) {
      out.single_basis[local] += single * lambda[local];
      out.double_basis[local] += normal_derivative * lambda[local];
    }
  }
}

void add_regular_triangle_integral_recursive(
    const Vec3 &target, const TriangleGeometry &source,
    const std::array<Vec3, 3> &subtriangle, int depth, KernelIntegral &out) {
  if (depth <= 0) {
    add_regular_triangle_integral(target, source, subtriangle, out);
    return;
  }
  const Vec3 ab = 0.5 * (subtriangle[0] + subtriangle[1]);
  const Vec3 bc = 0.5 * (subtriangle[1] + subtriangle[2]);
  const Vec3 ca = 0.5 * (subtriangle[2] + subtriangle[0]);
  add_regular_triangle_integral_recursive(target, source,
                                          {subtriangle[0], ab, ca}, depth - 1,
                                          out);
  add_regular_triangle_integral_recursive(target, source,
                                          {ab, subtriangle[1], bc}, depth - 1,
                                          out);
  add_regular_triangle_integral_recursive(target, source,
                                          {ca, bc, subtriangle[2]}, depth - 1,
                                          out);
  add_regular_triangle_integral_recursive(target, source, {ab, bc, ca},
                                          depth - 1, out);
}

void add_duffy_self_integral(const Vec3 &target,
                             const TriangleGeometry &source,
                             KernelIntegral &out) {
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const Vec3 a = source.point[edge];
    const Vec3 b = source.point[(edge + 1) % 3];
    const Vec3 u = a - target;
    const Vec3 v = b - target;
    const double cross_norm = norm(cross(u, v));
    if (!(cross_norm > 1.0e-18)) {
      continue;
    }
    for (std::size_t ir = 0; ir < kGauss6Points.size(); ++ir) {
      const double radius = kGauss6Points[ir];
      const double radius_weight = kGauss6Weights[ir];
      for (std::size_t is = 0; is < kGauss6Points.size(); ++is) {
        const double mix = kGauss6Points[is];
        const Vec3 direction = (1.0 - mix) * u + mix * v;
        const double direction_norm = norm(direction);
        if (!(direction_norm > 1.0e-18)) {
          continue;
        }
        const Vec3 source_point = target + radius * direction;
        const double kernel_weight =
            radius_weight * kGauss6Weights[is] * radius * cross_norm *
            kKernelScale / (radius * direction_norm);
        const auto lambda = barycentric_coordinates(source, source_point);
        out.single_constant += kernel_weight;
        for (std::size_t local = 0; local < 3; ++local) {
          out.single_basis[local] += kernel_weight * lambda[local];
        }
      }
    }
  }
  // The source-normal derivative is exactly zero for x and y on the same
  // planar triangle; its jump is represented by the Calderon half identity.
}

[[nodiscard]] int shared_vertex_count(const TriangleGeometry &left,
                                      const TriangleGeometry &right) noexcept {
  int count = 0;
  for (const std::size_t a : left.vertex) {
    for (const std::size_t b : right.vertex) {
      if (a == b) {
        ++count;
      }
    }
  }
  return count;
}

class DenseMatrix {
public:
  DenseMatrix() = default;
  DenseMatrix(std::size_t rows, std::size_t columns)
      : rows_(rows), columns_(columns), values_(rows * columns, 0.0) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
  double &operator()(std::size_t row, std::size_t column) {
    return values_[row * columns_ + column];
  }
  [[nodiscard]] double operator()(std::size_t row, std::size_t column) const {
    return values_[row * columns_ + column];
  }

private:
  std::size_t rows_{0};
  std::size_t columns_{0};
  std::vector<double> values_{};
};

[[nodiscard]] DenseMatrix identity(std::size_t dimension) {
  DenseMatrix out(dimension, dimension);
  for (std::size_t i = 0; i < dimension; ++i) {
    out(i, i) = 1.0;
  }
  return out;
}

[[nodiscard]] DenseMatrix transpose(const DenseMatrix &matrix) {
  DenseMatrix out(matrix.columns(), matrix.rows());
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
      out(column, row) = matrix(row, column);
    }
  }
  return out;
}

void symmetrize(DenseMatrix &matrix) {
  if (matrix.rows() != matrix.columns()) {
    throw std::runtime_error("dense Calderon symmetry requires a square matrix");
  }
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t column = row + 1; column < matrix.columns(); ++column) {
      const double value = 0.5 * (matrix(row, column) + matrix(column, row));
      matrix(row, column) = value;
      matrix(column, row) = value;
    }
  }
}

[[nodiscard]] DenseMatrix multiply(const DenseMatrix &left,
                                   const DenseMatrix &right) {
  if (left.columns() != right.rows()) {
    throw std::runtime_error("dense Calderon matrix shape mismatch");
  }
  DenseMatrix out(left.rows(), right.columns());
  for (std::size_t row = 0; row < left.rows(); ++row) {
    for (std::size_t middle = 0; middle < left.columns(); ++middle) {
      const double value = left(row, middle);
      if (value == 0.0) {
        continue;
      }
      for (std::size_t column = 0; column < right.columns(); ++column) {
        out(row, column) += value * right(middle, column);
      }
    }
  }
  return out;
}

[[nodiscard]] std::vector<double>
multiply(const DenseMatrix &matrix, std::span<const double> values) {
  if (matrix.columns() != values.size()) {
    throw std::runtime_error("dense Calderon action shape mismatch");
  }
  std::vector<double> out(matrix.rows(), 0.0);
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
      out[row] += matrix(row, column) * values[column];
    }
  }
  return out;
}

[[nodiscard]] double frobenius_norm(const DenseMatrix &matrix) noexcept {
  double sum = 0.0;
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
      sum += matrix(row, column) * matrix(row, column);
    }
  }
  return std::sqrt(sum);
}

[[nodiscard]] double relative_difference(const DenseMatrix &left,
                                         const DenseMatrix &right) {
  if (left.rows() != right.rows() || left.columns() != right.columns()) {
    throw std::runtime_error("dense Calderon difference shape mismatch");
  }
  DenseMatrix difference(left.rows(), left.columns());
  for (std::size_t row = 0; row < left.rows(); ++row) {
    for (std::size_t column = 0; column < left.columns(); ++column) {
      difference(row, column) = left(row, column) - right(row, column);
    }
  }
  return frobenius_norm(difference) / std::max(1.0e-300, frobenius_norm(right));
}

[[nodiscard]] DenseMatrix solve_left(DenseMatrix system, DenseMatrix right,
                                     double &pivot_ratio) {
  if (system.rows() != system.columns() || system.rows() != right.rows()) {
    throw std::runtime_error("dense Calderon solve shape mismatch");
  }
  const std::size_t dimension = system.rows();
  double pivot_min = std::numeric_limits<double>::infinity();
  double pivot_max = 0.0;
  for (std::size_t column = 0; column < dimension; ++column) {
    std::size_t pivot = column;
    double pivot_abs = std::abs(system(column, column));
    for (std::size_t row = column + 1; row < dimension; ++row) {
      const double candidate = std::abs(system(row, column));
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (!(pivot_abs > 1.0e-15) || !std::isfinite(pivot_abs)) {
      throw std::runtime_error("dense Calderon mass matrix is singular");
    }
    pivot_min = std::min(pivot_min, pivot_abs);
    pivot_max = std::max(pivot_max, pivot_abs);
    if (pivot != column) {
      for (std::size_t j = column; j < dimension; ++j) {
        std::swap(system(column, j), system(pivot, j));
      }
      for (std::size_t j = 0; j < right.columns(); ++j) {
        std::swap(right(column, j), right(pivot, j));
      }
    }
    const double diagonal = system(column, column);
    for (std::size_t row = column + 1; row < dimension; ++row) {
      const double factor = system(row, column) / diagonal;
      system(row, column) = 0.0;
      for (std::size_t j = column + 1; j < dimension; ++j) {
        system(row, j) -= factor * system(column, j);
      }
      for (std::size_t j = 0; j < right.columns(); ++j) {
        right(row, j) -= factor * right(column, j);
      }
    }
  }
  for (std::size_t row = dimension; row-- > 0;) {
    const double diagonal = system(row, row);
    for (std::size_t column = 0; column < right.columns(); ++column) {
      double value = right(row, column);
      for (std::size_t j = row + 1; j < dimension; ++j) {
        value -= system(row, j) * right(j, column);
      }
      right(row, column) = value / diagonal;
    }
  }
  pivot_ratio = pivot_min / std::max(1.0e-300, pivot_max);
  return right;
}

void add_p1_mass(const TriangleGeometry &triangle, DenseMatrix &mass) {
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double coefficient = i == j ? triangle.area / 6.0 : triangle.area / 12.0;
      mass(triangle.vertex[i], triangle.vertex[j]) += coefficient;
    }
  }
}

[[nodiscard]] DenseMatrix scaled_identity(std::size_t dimension, double scale) {
  DenseMatrix out = identity(dimension);
  for (std::size_t i = 0; i < dimension; ++i) {
    out(i, i) = scale;
  }
  return out;
}

[[nodiscard]] DenseMatrix scaled_matrix(const DenseMatrix &matrix,
                                        double scale) {
  DenseMatrix out(matrix.rows(), matrix.columns());
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
      out(row, column) = scale * matrix(row, column);
    }
  }
  return out;
}

[[nodiscard]] DenseMatrix subtract(const DenseMatrix &left,
                                   const DenseMatrix &right) {
  if (left.rows() != right.rows() || left.columns() != right.columns()) {
    throw std::runtime_error("dense Calderon subtraction shape mismatch");
  }
  DenseMatrix out(left.rows(), left.columns());
  for (std::size_t row = 0; row < left.rows(); ++row) {
    for (std::size_t column = 0; column < left.columns(); ++column) {
      out(row, column) = left(row, column) - right(row, column);
    }
  }
  return out;
}

[[nodiscard]] DenseMatrix block_2x2(const DenseMatrix &top_left,
                                    const DenseMatrix &top_right,
                                    const DenseMatrix &bottom_left,
                                    const DenseMatrix &bottom_right) {
  if (top_left.rows() != top_right.rows() ||
      bottom_left.rows() != bottom_right.rows() ||
      top_left.columns() != bottom_left.columns() ||
      top_right.columns() != bottom_right.columns()) {
    throw std::runtime_error("dense Calderon block shape mismatch");
  }
  DenseMatrix out(top_left.rows() + bottom_left.rows(),
                  top_left.columns() + top_right.columns());
  for (std::size_t row = 0; row < top_left.rows(); ++row) {
    for (std::size_t column = 0; column < top_left.columns(); ++column) {
      out(row, column) = top_left(row, column);
    }
    for (std::size_t column = 0; column < top_right.columns(); ++column) {
      out(row, top_left.columns() + column) = top_right(row, column);
    }
  }
  for (std::size_t row = 0; row < bottom_left.rows(); ++row) {
    for (std::size_t column = 0; column < bottom_left.columns(); ++column) {
      out(top_left.rows() + row, column) = bottom_left(row, column);
    }
    for (std::size_t column = 0; column < bottom_right.columns(); ++column) {
      out(top_left.rows() + row, top_left.columns() + column) =
          bottom_right(row, column);
    }
  }
  return out;
}

[[nodiscard]] DenseMatrix deterministic_mixed_probes(std::size_t dimension) {
  DenseMatrix probes(dimension, 3);
  for (std::size_t row = 0; row < dimension; ++row) {
    const std::size_t index = row + 1;
    probes(row, 0) = (index % 2 == 0) ? -1.0 : 1.0;
    probes(row, 1) = (index % 3 == 0) ? -0.75 : 0.5;
    probes(row, 2) = (index % 5 < 2) ? 0.25 : -1.25;
  }
  return probes;
}

[[nodiscard]] double constant_action_relative(const DenseMatrix &matrix) {
  double squared = 0.0;
  double scale = 0.0;
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    double value = 0.0;
    for (std::size_t column = 0; column < matrix.columns(); ++column) {
      value += matrix(row, column);
      scale += matrix(row, column) * matrix(row, column);
    }
    squared += value * value;
  }
  return std::sqrt(squared) / std::max(1.0e-300, std::sqrt(scale));
}

struct DualP0Patch {
  std::array<Vec3, 3> point{};
  std::size_t owner{0};
  std::size_t parent{0};
  double area{0.0};
};

[[nodiscard]] double triangle_area(const std::array<Vec3, 3> &point) noexcept {
  return 0.5 * norm(cross(point[1] - point[0], point[2] - point[0]));
}

[[nodiscard]] std::vector<DualP0Patch>
build_barycentric_dual_p0_patches(const std::vector<TriangleGeometry> &triangles) {
  std::vector<DualP0Patch> patches;
  patches.reserve(6 * triangles.size());
  const auto append = [&](std::size_t owner, std::size_t parent,
                          const std::array<Vec3, 3> &point) {
    const double area = triangle_area(point);
    if (!(area > 1.0e-18) || !std::isfinite(area)) {
      throw std::runtime_error("degenerate barycentric dual patch");
    }
    patches.push_back({point, owner, parent, area});
  };
  for (std::size_t parent = 0; parent < triangles.size(); ++parent) {
    const TriangleGeometry &triangle = triangles[parent];
    const Vec3 mid01 = 0.5 * (triangle.point[0] + triangle.point[1]);
    const Vec3 mid12 = 0.5 * (triangle.point[1] + triangle.point[2]);
    const Vec3 mid20 = 0.5 * (triangle.point[2] + triangle.point[0]);
    const Vec3 centre = (1.0 / 3.0) *
                        (triangle.point[0] + triangle.point[1] + triangle.point[2]);
    append(triangle.vertex[0], parent, {triangle.point[0], mid01, centre});
    append(triangle.vertex[0], parent, {triangle.point[0], centre, mid20});
    append(triangle.vertex[1], parent, {triangle.point[1], mid12, centre});
    append(triangle.vertex[1], parent, {triangle.point[1], centre, mid01});
    append(triangle.vertex[2], parent, {triangle.point[2], mid20, centre});
    append(triangle.vertex[2], parent, {triangle.point[2], centre, mid12});
  }
  return patches;
}

[[nodiscard]] Vec3 interpolate_patch(const DualP0Patch &patch,
                                      const std::array<double, 3> &lambda) noexcept {
  return lambda[0] * patch.point[0] + lambda[1] * patch.point[1] +
         lambda[2] * patch.point[2];
}

struct JcpPointKernel {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
};

[[nodiscard]] JcpPointKernel evaluate_jcp_point_kernel(
    const Vec3 &target, const Vec3 &target_normal, const Vec3 &source,
    const Vec3 &source_normal, double eps_ratio, double kappa) noexcept {
  const Vec3 delta = target - source;
  const double distance_squared = dot(delta, delta);
  const double epsilon = std::max(1.0e-12, eps_ratio);
  const double screening = std::max(0.0, kappa);
  if (distance_squared <= 1.0e-24) {
    return {screening * kKernelScale, 0.0, 0.0, 0.0};
  }
  const double distance = std::sqrt(distance_squared);
  const double expkr = std::exp(-screening * distance);
  const double inverse_distance = 1.0 / distance;
  const double inverse_distance_squared = inverse_distance * inverse_distance;
  const double source_projection = dot(source_normal, delta);
  const double target_projection = dot(target_normal, delta);
  const double d0_source = kKernelScale * source_projection /
                           (distance_squared * distance);
  const double dk_source = expkr * (1.0 + screening * distance) * d0_source;
  const double d0_target = -kKernelScale * target_projection /
                           (distance_squared * distance);
  const double dk_target = expkr * (1.0 + screening * distance) * d0_target;
  const double target_unit_projection = target_projection * inverse_distance;
  const double source_unit_projection = source_projection * inverse_distance;
  const double normal_dot = dot(target_normal, source_normal);
  const auto mixed = [&](double local_kappa) noexcept {
    const double local_exp = std::exp(-local_kappa * distance);
    const double first = -local_exp * (1.0 + local_kappa * distance) *
                         inverse_distance_squared * kKernelScale;
    const double second = local_exp *
                          (local_kappa * local_kappa * inverse_distance +
                           2.0 * local_kappa * inverse_distance_squared +
                           2.0 * inverse_distance_squared * inverse_distance) *
                          kKernelScale;
    return second * target_unit_projection * source_unit_projection +
           (first * inverse_distance) *
               (normal_dot - target_unit_projection * source_unit_projection);
  };
  return {kKernelScale * (1.0 - expkr) * inverse_distance,
          epsilon * dk_source - d0_source,
          d0_target - dk_target / epsilon,
          mixed(screening) - mixed(0.0)};
}

struct JcpSourceIntegral {
  double k1{0.0};
  double k2{0.0};
  double k3{0.0};
  double k4{0.0};
  std::array<double, 3> k1_basis{};
  std::array<double, 3> k2_basis{};
  std::array<double, 3> k3_basis{};
  std::array<double, 3> k4_basis{};
};

void add_jcp_source_integral(const Vec3 &target, const Vec3 &target_normal,
                             const TriangleGeometry &source_parent,
                             const std::array<Vec3, 3> &source_patch,
                             double eps_ratio, double kappa,
                             JcpSourceIntegral &out) {
  const double area = triangle_area(source_patch);
  if (!(area > 1.0e-18)) {
    return;
  }
  for (const TriangleQuadraturePoint &quadrature : kTriangleGauss7) {
    const Vec3 source = quadrature.lambda[0] * source_patch[0] +
                        quadrature.lambda[1] * source_patch[1] +
                        quadrature.lambda[2] * source_patch[2];
    const JcpPointKernel kernel = evaluate_jcp_point_kernel(
        target, target_normal, source, source_parent.normal, eps_ratio, kappa);
    const double weight = area * quadrature.weight;
    const auto lambda = barycentric_coordinates(source_parent, source);
    out.k1 += weight * kernel.k1;
    out.k2 += weight * kernel.k2;
    out.k3 += weight * kernel.k3;
    out.k4 += weight * kernel.k4;
    for (std::size_t local = 0; local < 3; ++local) {
      out.k1_basis[local] += weight * kernel.k1 * lambda[local];
      out.k2_basis[local] += weight * kernel.k2 * lambda[local];
      out.k3_basis[local] += weight * kernel.k3 * lambda[local];
      out.k4_basis[local] += weight * kernel.k4 * lambda[local];
    }
  }
}

void add_jcp_source_integral_recursive(
    const Vec3 &target, const Vec3 &target_normal,
    const TriangleGeometry &source_parent,
    const std::array<Vec3, 3> &source_patch, int depth, double eps_ratio,
    double kappa, JcpSourceIntegral &out) {
  if (depth <= 0) {
    add_jcp_source_integral(target, target_normal, source_parent, source_patch,
                            eps_ratio, kappa, out);
    return;
  }
  const Vec3 mid01 = 0.5 * (source_patch[0] + source_patch[1]);
  const Vec3 mid12 = 0.5 * (source_patch[1] + source_patch[2]);
  const Vec3 mid20 = 0.5 * (source_patch[2] + source_patch[0]);
  add_jcp_source_integral_recursive(target, target_normal, source_parent,
                                    {source_patch[0], mid01, mid20}, depth - 1,
                                    eps_ratio, kappa, out);
  add_jcp_source_integral_recursive(target, target_normal, source_parent,
                                    {mid01, source_patch[1], mid12}, depth - 1,
                                    eps_ratio, kappa, out);
  add_jcp_source_integral_recursive(target, target_normal, source_parent,
                                    {mid20, mid12, source_patch[2]}, depth - 1,
                                    eps_ratio, kappa, out);
  add_jcp_source_integral_recursive(target, target_normal, source_parent,
                                    {mid01, mid12, mid20}, depth - 1,
                                    eps_ratio, kappa, out);
}

struct ScreenedSourceIntegral {
  double green_kappa_constant{0.0};
  double green_difference_constant{0.0};
  std::array<double, 3> green_kappa_basis{};
};

void add_regular_screened_source_integral(
    const Vec3 &target, const TriangleGeometry &source_parent,
    const std::array<Vec3, 3> &source_patch, double kappa,
    bool include_green_kappa, bool include_difference,
    ScreenedSourceIntegral &out) {
  const double area = triangle_area(source_patch);
  if (!(area > 1.0e-18)) {
    return;
  }
  const double screening = std::max(0.0, kappa);
  for (const TriangleQuadraturePoint &quadrature : kTriangleGauss7) {
    const Vec3 source = quadrature.lambda[0] * source_patch[0] +
                        quadrature.lambda[1] * source_patch[1] +
                        quadrature.lambda[2] * source_patch[2];
    const Vec3 delta = target - source;
    const double distance_squared = dot(delta, delta);
    const double weight = area * quadrature.weight;
    if (distance_squared <= 1.0e-24) {
      if (include_difference) {
        out.green_difference_constant -= weight * screening * kKernelScale;
      }
      continue;
    }
    const double distance = std::sqrt(distance_squared);
    const double green_zero = kKernelScale / distance;
    const double green_kappa = std::exp(-screening * distance) * green_zero;
    if (include_difference) {
      out.green_difference_constant += weight * (green_kappa - green_zero);
    }
    if (include_green_kappa) {
      const auto lambda = barycentric_coordinates(source_parent, source);
      out.green_kappa_constant += weight * green_kappa;
      for (std::size_t local = 0; local < 3; ++local) {
        out.green_kappa_basis[local] += weight * green_kappa * lambda[local];
      }
    }
  }
}

void add_regular_screened_source_integral_recursive(
    const Vec3 &target, const TriangleGeometry &source_parent,
    const std::array<Vec3, 3> &source_patch, int depth, double kappa,
    bool include_green_kappa, bool include_difference,
    ScreenedSourceIntegral &out) {
  if (depth <= 0) {
    add_regular_screened_source_integral(
        target, source_parent, source_patch, kappa, include_green_kappa,
        include_difference, out);
    return;
  }
  const Vec3 mid01 = 0.5 * (source_patch[0] + source_patch[1]);
  const Vec3 mid12 = 0.5 * (source_patch[1] + source_patch[2]);
  const Vec3 mid20 = 0.5 * (source_patch[2] + source_patch[0]);
  add_regular_screened_source_integral_recursive(
      target, source_parent, {source_patch[0], mid01, mid20}, depth - 1,
      kappa, include_green_kappa, include_difference, out);
  add_regular_screened_source_integral_recursive(
      target, source_parent, {mid01, source_patch[1], mid12}, depth - 1,
      kappa, include_green_kappa, include_difference, out);
  add_regular_screened_source_integral_recursive(
      target, source_parent, {mid20, mid12, source_patch[2]}, depth - 1,
      kappa, include_green_kappa, include_difference, out);
  add_regular_screened_source_integral_recursive(
      target, source_parent, {mid01, mid12, mid20}, depth - 1, kappa,
      include_green_kappa, include_difference, out);
}

void add_duffy_screened_green_integral(const Vec3 &target,
                                       const TriangleGeometry &source,
                                       double kappa,
                                       ScreenedSourceIntegral &out) {
  const double screening = std::max(0.0, kappa);
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const Vec3 a = source.point[edge];
    const Vec3 b = source.point[(edge + 1) % 3];
    const Vec3 u = a - target;
    const Vec3 v = b - target;
    const double cross_norm = norm(cross(u, v));
    if (!(cross_norm > 1.0e-18)) {
      continue;
    }
    for (std::size_t ir = 0; ir < kGauss6Points.size(); ++ir) {
      const double radius = kGauss6Points[ir];
      for (std::size_t is = 0; is < kGauss6Points.size(); ++is) {
        const double mix = kGauss6Points[is];
        const Vec3 direction = (1.0 - mix) * u + mix * v;
        const double direction_norm = norm(direction);
        if (!(direction_norm > 1.0e-18)) {
          continue;
        }
        const Vec3 source_point = target + radius * direction;
        const double weight =
            kGauss6Weights[ir] * kGauss6Weights[is] * cross_norm *
            kKernelScale * std::exp(-screening * radius * direction_norm) /
            direction_norm;
        const auto lambda = barycentric_coordinates(source, source_point);
        out.green_kappa_constant += weight;
        for (std::size_t local = 0; local < 3; ++local) {
          out.green_kappa_basis[local] += weight * lambda[local];
        }
      }
    }
  }
}

[[nodiscard]] double screened_flat_mixed_kernel(double distance,
                                                 double kappa) noexcept {
  if (!(distance > 1.0e-18)) {
    return 0.0;
  }
  const double x = std::max(0.0, kappa) * distance;
  double numerator = 0.0;
  if (x < 1.0e-4) {
    const double x2 = x * x;
    numerator = x2 * (0.5 + x * (-1.0 / 3.0 + x * (0.125 - x / 30.0)));
  } else {
    numerator = 1.0 - std::exp(-x) * (1.0 + x);
  }
  return kKernelScale * numerator /
         (distance * distance * distance);
}

void add_duffy_self_jcp_k4_integral(const Vec3 &target,
                                    const TriangleGeometry &source,
                                    double kappa, JcpSourceIntegral &out) {
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const Vec3 a = source.point[edge];
    const Vec3 b = source.point[(edge + 1) % 3];
    const Vec3 u = a - target;
    const Vec3 v = b - target;
    const double cross_norm = norm(cross(u, v));
    if (!(cross_norm > 1.0e-18)) {
      continue;
    }
    for (std::size_t ir = 0; ir < kGauss6Points.size(); ++ir) {
      const double radius = kGauss6Points[ir];
      for (std::size_t is = 0; is < kGauss6Points.size(); ++is) {
        const double mix = kGauss6Points[is];
        const Vec3 direction = (1.0 - mix) * u + mix * v;
        const double direction_norm = norm(direction);
        if (!(direction_norm > 1.0e-18)) {
          continue;
        }
        const double distance = radius * direction_norm;
        const double weight = kGauss6Weights[ir] * kGauss6Weights[is] *
                              radius * cross_norm;
        const double kernel = screened_flat_mixed_kernel(distance, kappa);
        const Vec3 source_point = target + radius * direction;
        const auto lambda = barycentric_coordinates(source, source_point);
        out.k4 += weight * kernel;
        for (std::size_t local = 0; local < 3; ++local) {
          out.k4_basis[local] += weight * kernel * lambda[local];
        }
      }
    }
  }
}

} // namespace

WeakCalderonResult assemble_laplace_weak_calderon_identity(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const WeakCalderonConfig &config) {
  const auto begin = std::chrono::steady_clock::now();
  WeakCalderonResult result;
  result.p1_dofs = mesh.vertices.size();
  result.triangles = mesh.triangles.size();
  if (!spaces.quality.valid) {
    result.status = "invalid_trace_space";
    return result;
  }
  if (mesh.vertices.empty() || mesh.triangles.empty() ||
      mesh.vertices.size() != spaces.p1_vertex_lumped_mass.size()) {
    result.status = "invalid_mesh";
    return result;
  }
  if (config.adjacent_pair_subdivision_depth < 0 || config.duffy_order != 6 ||
      !(config.identity_relative_tolerance > 0.0) ||
      !std::isfinite(config.identity_relative_tolerance)) {
    result.status = "invalid_config";
    return result;
  }

  try {
    std::vector<TriangleGeometry> triangles(mesh.triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
      if (!build_triangle_geometry(mesh, index, triangles[index])) {
        result.status = "invalid_triangle_geometry";
        return result;
      }
    }
    const std::size_t n = mesh.vertices.size();
    DenseMatrix mass(n, n);
    DenseMatrix single_layer(n, n);
    DenseMatrix double_layer(n, n);
    DenseMatrix hypersingular(n, n);
    for (const TriangleGeometry &target_triangle : triangles) {
      add_p1_mass(target_triangle, mass);
    }

    for (std::size_t target_index = 0; target_index < triangles.size(); ++target_index) {
      const TriangleGeometry &target_triangle = triangles[target_index];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target_point = interpolate(target_triangle, target_quadrature.lambda);
        const double target_weight = target_triangle.area * target_quadrature.weight;
        for (std::size_t source_index = 0; source_index < triangles.size(); ++source_index) {
          const TriangleGeometry &source_triangle = triangles[source_index];
          const bool self_pair = target_index == source_index;
          KernelIntegral integral;
          if (self_pair) {
            add_duffy_self_integral(target_point, source_triangle, integral);
          } else if (shared_vertex_count(target_triangle, source_triangle) > 0) {
            add_regular_triangle_integral_recursive(
                target_point, source_triangle, source_triangle.point,
                config.adjacent_pair_subdivision_depth, integral);
          } else {
            add_regular_triangle_integral(target_point, source_triangle,
                                          source_triangle.point, integral);
          }
          for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t row = target_triangle.vertex[i];
            const double test_value = target_quadrature.lambda[i];
            for (std::size_t j = 0; j < 3; ++j) {
              const std::size_t column = source_triangle.vertex[j];
              single_layer(row, column) +=
                  target_weight * test_value * integral.single_basis[j];
              double_layer(row, column) +=
                  target_weight * test_value * integral.double_basis[j];
              hypersingular(row, column) +=
                  target_weight * dot(target_triangle.surface_curl[i],
                                      source_triangle.surface_curl[j]) *
                  integral.single_constant;
            }
          }
        }
      }
    }

    // V and W are symmetric bilinear forms. The source-only adaptive split
    // for adjacent triangle pairs is intentionally conservative but slightly
    // asymmetric; restore the Galerkin symmetry before the operator identity.
    symmetrize(single_layer);
    symmetrize(hypersingular);

    const DenseMatrix adjoint_double_layer = transpose(double_layer);
    const DenseMatrix mass_block = [&] {
      DenseMatrix out(2 * n, 2 * n);
      for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
          out(row, column) = mass(row, column);
          out(n + row, n + column) = mass(row, column);
        }
      }
      return out;
    }();
    DenseMatrix calderon_weak(2 * n, 2 * n);
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        calderon_weak(row, column) = double_layer(row, column);
        calderon_weak(row, n + column) = -single_layer(row, column);
        calderon_weak(n + row, column) = -hypersingular(row, column);
        calderon_weak(n + row, n + column) = -adjoint_double_layer(row, column);
      }
    }

    double mass_pivot_ratio = 0.0;
    const DenseMatrix strong_calderon =
        solve_left(mass_block, calderon_weak, mass_pivot_ratio);
    const DenseMatrix strong_square = multiply(strong_calderon, strong_calderon);
    const DenseMatrix weak_square = multiply(calderon_weak, strong_calderon);
    result.mass_pivot_ratio = mass_pivot_ratio;
    result.single_layer_symmetry_relative =
        relative_difference(single_layer, transpose(single_layer));
    result.hypersingular_symmetry_relative =
        relative_difference(hypersingular, transpose(hypersingular));
    result.hypersingular_constant_relative = constant_action_relative(hypersingular);
    result.strong_identity_relative = relative_difference(
        strong_square, scaled_identity(2 * n, 0.25));
    // C M^-1 C is represented in a non-orthonormal coefficient basis. The
    // mass-scaled identity is therefore the strong-form metric above; retain
    // the raw coefficient Frobenius value only as an assembly diagnostic.
    result.weak_identity_mass_scaled_relative = result.strong_identity_relative;
    result.weak_identity_raw_frobenius_relative = relative_difference(
        weak_square, scaled_matrix(mass_block, 0.25));
    const DenseMatrix strong_k = solve_left(mass, double_layer, mass_pivot_ratio);
    const DenseMatrix strong_v = solve_left(mass, single_layer, mass_pivot_ratio);
    const DenseMatrix strong_w = solve_left(mass, hypersingular, mass_pivot_ratio);
    const DenseMatrix strong_kp = solve_left(mass, adjoint_double_layer, mass_pivot_ratio);
    result.k_v_commutator_relative = relative_difference(
        multiply(strong_k, strong_v), multiply(strong_v, strong_kp));
    result.w_k_commutator_relative = relative_difference(
        multiply(strong_w, strong_k), multiply(strong_kp, strong_w));
    const bool finite = std::isfinite(result.mass_pivot_ratio) &&
                        std::isfinite(result.strong_identity_relative) &&
                        std::isfinite(result.weak_identity_mass_scaled_relative) &&
                        std::isfinite(result.weak_identity_raw_frobenius_relative) &&
                        std::isfinite(result.single_layer_symmetry_relative) &&
                        std::isfinite(result.hypersingular_symmetry_relative) &&
                        std::isfinite(result.hypersingular_constant_relative) &&
                        std::isfinite(result.k_v_commutator_relative) &&
                        std::isfinite(result.w_k_commutator_relative);
    if (!finite) {
      result.status = "nonfinite";
    } else if (result.mass_pivot_ratio < 1.0e-12) {
      result.status = "mass_pivot_gap";
    } else if (result.strong_identity_relative <= config.identity_relative_tolerance) {
      result.status = "pass";
    } else {
      result.status = "identity_gap";
    }
  } catch (const std::exception &) {
    result.status = "assembly_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.assembly_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

MixedTraceCalderonResult assemble_jcp_mixed_trace_pairing_calibration(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const MixedTraceCalderonConfig &config) {
  const auto begin = std::chrono::steady_clock::now();
  MixedTraceCalderonResult result;
  result.p1_dofs = mesh.vertices.size();
  result.dual_p0_dofs = mesh.vertices.size();
  result.screened_hypersingular_status =
      "separated_pair_parity_only_c5c2_required";
  result.derivative_block_relation =
      "no_scalar_epsilon_adjoint_identity_for_jcp_k2_k3";
  if (!spaces.quality.valid) {
    result.status = "invalid_trace_space";
    return result;
  }
  if (mesh.vertices.empty() || mesh.triangles.empty() ||
      mesh.vertices.size() != spaces.p1_vertex_lumped_mass.size()) {
    result.status = "invalid_mesh";
    return result;
  }
  if (config.adjacent_pair_subdivision_depth < 0 ||
      !(config.eps_ratio > 0.0) || !(config.kappa >= 0.0) ||
      !std::isfinite(config.eps_ratio) || !std::isfinite(config.kappa) ||
      config.adjacent_pair_subdivision_depth > 6) {
    result.status = "invalid_config";
    return result;
  }

  try {
    std::vector<TriangleGeometry> triangles(mesh.triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
      if (!build_triangle_geometry(mesh, index, triangles[index])) {
        result.status = "invalid_triangle_geometry";
        return result;
      }
    }
    const std::vector<DualP0Patch> dual_patches =
        build_barycentric_dual_p0_patches(triangles);
    const std::size_t n = mesh.vertices.size();
    result.dual_p0_subtriangles = dual_patches.size();
    DenseMatrix pairing(n, n);
    double dual_area = 0.0;
    for (const DualP0Patch &patch : dual_patches) {
      const TriangleGeometry &parent = triangles[patch.parent];
      dual_area += patch.area;
      for (const TriangleQuadraturePoint &quadrature : kTriangleGauss7) {
        const Vec3 point = interpolate_patch(patch, quadrature.lambda);
        const auto lambda = barycentric_coordinates(parent, point);
        const double weight = patch.area * quadrature.weight;
        for (std::size_t local = 0; local < 3; ++local) {
          pairing(patch.owner, parent.vertex[local]) += weight * lambda[local];
        }
      }
    }
    result.dual_p0_partition_relative =
        std::abs(dual_area - spaces.quality.total_area) /
        std::max(1.0e-300, spaces.quality.total_area);
    double pairing_pivot_ratio = 0.0;
    const DenseMatrix pairing_inverse =
        solve_left(pairing, identity(n), pairing_pivot_ratio);
    result.pairing_pivot_ratio = pairing_pivot_ratio;
    result.pairing_inverse_relative = relative_difference(
        multiply(pairing, pairing_inverse), identity(n));

    DenseMatrix k1(n, n);
    DenseMatrix k2(n, n);
    DenseMatrix k3(n, n);
    for (const DualP0Patch &target_patch : dual_patches) {
      const TriangleGeometry &target_parent = triangles[target_patch.parent];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target = interpolate_patch(target_patch, target_quadrature.lambda);
        const double target_weight = target_patch.area * target_quadrature.weight;
        for (const DualP0Patch &source_patch : dual_patches) {
          JcpSourceIntegral integral;
          add_jcp_source_integral(target, target_parent.normal,
                                  triangles[source_patch.parent], source_patch.point,
                                  config.eps_ratio, config.kappa, integral);
          k1(target_patch.owner, source_patch.owner) += target_weight * integral.k1;
        }
        for (std::size_t source_index = 0; source_index < triangles.size();
             ++source_index) {
          if (source_index == target_patch.parent) {
            continue;
          }
          const TriangleGeometry &source_parent = triangles[source_index];
          const int depth = shared_vertex_count(target_parent, source_parent) > 0
                                ? config.adjacent_pair_subdivision_depth
                                : 0;
          JcpSourceIntegral integral;
          add_jcp_source_integral_recursive(
              target, target_parent.normal, source_parent, source_parent.point,
              depth, config.eps_ratio, config.kappa, integral);
          for (std::size_t local = 0; local < 3; ++local) {
            k2(target_patch.owner, source_parent.vertex[local]) +=
                target_weight * integral.k2_basis[local];
          }
        }
      }
    }
    for (std::size_t target_index = 0; target_index < triangles.size();
         ++target_index) {
      const TriangleGeometry &target_parent = triangles[target_index];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target = interpolate(target_parent, target_quadrature.lambda);
        const double target_weight = target_parent.area * target_quadrature.weight;
        for (const DualP0Patch &source_patch : dual_patches) {
          if (source_patch.parent == target_index) {
            continue;
          }
          const TriangleGeometry &source_parent = triangles[source_patch.parent];
          const int depth = shared_vertex_count(target_parent, source_parent) > 0
                                ? config.adjacent_pair_subdivision_depth
                                : 0;
          JcpSourceIntegral integral;
          add_jcp_source_integral_recursive(
              target, target_parent.normal, source_parent, source_patch.point,
              depth, config.eps_ratio, config.kappa, integral);
          for (std::size_t local = 0; local < 3; ++local) {
            k3(target_parent.vertex[local], source_patch.owner) +=
                target_weight * target_quadrature.lambda[local] * integral.k3;
          }
        }
      }
    }

    result.k1_symmetry_relative = relative_difference(k1, transpose(k1));
    result.k2_k3_negative_epsilon_closure_relative = relative_difference(
        transpose(k2), scaled_matrix(k3, -config.eps_ratio));
    result.k2_k3_positive_epsilon_closure_relative = relative_difference(
        transpose(k2), scaled_matrix(k3, config.eps_ratio));

    double k4_difference_squared = 0.0;
    double k4_reference_squared = 0.0;
    double k23_difference_squared = 0.0;
    double k23_reference_squared = 0.0;
    std::size_t k4_samples = 0;
    for (std::size_t target_index = 0;
         target_index < triangles.size() && k4_samples < 24; ++target_index) {
      const TriangleGeometry &target_triangle = triangles[target_index];
      const Vec3 target = interpolate(target_triangle, {1.0 / 3.0, 1.0 / 3.0,
                                                         1.0 / 3.0});
      for (std::size_t source_index = target_index + 1;
           source_index < triangles.size() && k4_samples < 24; ++source_index) {
        const TriangleGeometry &source_triangle = triangles[source_index];
        if (shared_vertex_count(target_triangle, source_triangle) != 0) {
          continue;
        }
        const Vec3 source = interpolate(source_triangle,
                                        {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
        const JcpPointKernel forward_kernel = evaluate_jcp_point_kernel(
            target, target_triangle.normal, source, source_triangle.normal,
            config.eps_ratio, config.kappa);
        const JcpPointKernel reverse_kernel = evaluate_jcp_point_kernel(
            source, source_triangle.normal, target, target_triangle.normal,
            config.eps_ratio, config.kappa);
        const double forward = forward_kernel.k4;
        const double reverse = reverse_kernel.k4;
        const double difference = forward - reverse;
        k4_difference_squared += difference * difference;
        k4_reference_squared += forward * forward + reverse * reverse;
        const double derivative_difference =
            forward_kernel.k2 + config.eps_ratio * reverse_kernel.k3;
        k23_difference_squared += derivative_difference * derivative_difference;
        k23_reference_squared += forward_kernel.k2 * forward_kernel.k2 +
                                 (config.eps_ratio * reverse_kernel.k3) *
                                     (config.eps_ratio * reverse_kernel.k3);
        ++k4_samples;
      }
    }
    if (k4_samples == 0) {
      result.status = "missing_separated_k4_samples";
      return result;
    }
    result.k4_separated_pair_symmetry_relative =
        std::sqrt(k4_difference_squared) /
        std::max(1.0e-300, std::sqrt(k4_reference_squared));
    result.k2_k3_separated_nonclosure_relative =
        std::sqrt(k23_difference_squared) /
        std::max(1.0e-300, std::sqrt(k23_reference_squared));
    const Vec3 zero_target = interpolate(triangles.front(),
                                         {0.2, 0.3, 0.5});
    const Vec3 zero_source = interpolate(triangles.back(),
                                         {0.3, 0.2, 0.5});
    const JcpPointKernel zero_kernel = evaluate_jcp_point_kernel(
        zero_target, triangles.front().normal, zero_source, triangles.back().normal,
        1.0, 0.0);
    result.zero_screening_kernel_absolute = std::max(
        std::max(std::abs(zero_kernel.k1), std::abs(zero_kernel.k2)),
        std::max(std::abs(zero_kernel.k3), std::abs(zero_kernel.k4)));

    const bool finite = std::isfinite(result.dual_p0_partition_relative) &&
                        std::isfinite(result.pairing_pivot_ratio) &&
                        std::isfinite(result.pairing_inverse_relative) &&
                        std::isfinite(result.k1_symmetry_relative) &&
                        std::isfinite(result.k2_k3_negative_epsilon_closure_relative) &&
                        std::isfinite(result.k2_k3_positive_epsilon_closure_relative) &&
                        std::isfinite(result.k2_k3_separated_nonclosure_relative) &&
                        std::isfinite(result.k4_separated_pair_symmetry_relative) &&
                        std::isfinite(result.zero_screening_kernel_absolute);
    if (!finite) {
      result.status = "nonfinite";
    } else if (result.pairing_pivot_ratio < 1.0e-12 ||
               result.pairing_inverse_relative > 1.0e-11 ||
               result.dual_p0_partition_relative > 1.0e-12) {
      result.status = "mixed_riesz_pairing_gap";
    } else if (result.k1_symmetry_relative > 1.0e-11 ||
               result.k4_separated_pair_symmetry_relative > 1.0e-11 ||
               result.zero_screening_kernel_absolute > 1.0e-13) {
      result.status = "jcp_kernel_parity_gap";
    } else {
      result.status = "pass";
    }
  } catch (const std::exception &) {
    result.status = "assembly_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.assembly_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

ScreenedHypersingularResult assemble_jcp_screened_hypersingular_weak_form(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const ScreenedHypersingularConfig &config) {
  const auto begin = std::chrono::steady_clock::now();
  ScreenedHypersingularResult result;
  result.p1_dofs = mesh.vertices.size();
  result.triangles = mesh.triangles.size();
  result.weak_form_sign = "positive_maue_for_jcp_k4";
  result.separated_pair_comparison_status =
      "not_pairwise_comparable_after_maue_integration_by_parts";
  if (!spaces.quality.valid) {
    result.status = "invalid_trace_space";
    return result;
  }
  if (mesh.vertices.empty() || mesh.triangles.empty()) {
    result.status = "invalid_mesh";
    return result;
  }
  if (config.adjacent_pair_subdivision_depth < 0 ||
      config.adjacent_pair_subdivision_depth > 6 || !(config.kappa > 0.0) ||
      !std::isfinite(config.kappa)) {
    result.status = "invalid_config";
    return result;
  }

  try {
    std::vector<TriangleGeometry> triangles(mesh.triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
      if (!build_triangle_geometry(mesh, index, triangles[index])) {
        result.status = "invalid_triangle_geometry";
        return result;
      }
    }
    const std::size_t n = mesh.vertices.size();
    DenseMatrix weak_negative(n, n);
    DenseMatrix weak_negative_separated(n, n);
    DenseMatrix direct_separated(n, n);
    DenseMatrix direct_global(n, n);
    for (std::size_t target_index = 0; target_index < triangles.size();
         ++target_index) {
      const TriangleGeometry &target_triangle = triangles[target_index];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target = interpolate(target_triangle, target_quadrature.lambda);
        const double target_weight = target_triangle.area * target_quadrature.weight;
        for (std::size_t source_index = 0; source_index < triangles.size();
             ++source_index) {
          const TriangleGeometry &source_triangle = triangles[source_index];
          const bool self_pair = target_index == source_index;
          const bool adjacent_pair =
              !self_pair && shared_vertex_count(target_triangle, source_triangle) > 0;
          const bool separated_pair = !self_pair && !adjacent_pair;
          ScreenedSourceIntegral screened;
          if (self_pair) {
            // G_k-G_0 is bounded on the panel while G_k retains a weak
            // singularity; use regular and Duffy parts separately.
            add_regular_screened_source_integral(
                target, source_triangle, source_triangle.point, config.kappa,
                false, true, screened);
            add_duffy_screened_green_integral(target, source_triangle,
                                               config.kappa, screened);
          } else {
            const int depth = adjacent_pair
                                  ? config.adjacent_pair_subdivision_depth
                                  : 0;
            add_regular_screened_source_integral_recursive(
                target, source_triangle, source_triangle.point, depth,
                config.kappa, true, true, screened);
          }
          JcpSourceIntegral direct;
          if (self_pair) {
            add_duffy_self_jcp_k4_integral(target, source_triangle,
                                            config.kappa, direct);
          } else {
            const int depth = adjacent_pair
                                  ? config.adjacent_pair_subdivision_depth
                                  : 0;
            add_jcp_source_integral_recursive(
                target, target_triangle.normal, source_triangle,
                source_triangle.point, depth, 1.0, config.kappa, direct);
          }
          if (separated_pair) {
            ++result.separated_source_evaluations;
          }
          const double normal_dot =
              dot(target_triangle.normal, source_triangle.normal);
          for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t row = target_triangle.vertex[i];
            const double target_basis = target_quadrature.lambda[i];
            for (std::size_t j = 0; j < 3; ++j) {
              const std::size_t column = source_triangle.vertex[j];
              const double maue =
                  dot(target_triangle.surface_curl[i],
                      source_triangle.surface_curl[j]) *
                      screened.green_difference_constant +
                  config.kappa * config.kappa * normal_dot * target_basis *
                      screened.green_kappa_basis[j];
              const double weak_value = target_weight * maue;
              weak_negative(row, column) += weak_value;
              direct_global(row, column) +=
                  target_weight * target_basis * direct.k4_basis[j];
              if (separated_pair) {
                weak_negative_separated(row, column) += weak_value;
                direct_separated(row, column) +=
                    target_weight * target_basis * direct.k4_basis[j];
              }
            }
          }
        }
      }
    }
    if (result.separated_source_evaluations == 0) {
      result.status = "missing_separated_pairs";
      return result;
    }
    result.weak_pre_symmetry_relative =
        relative_difference(weak_negative, transpose(weak_negative));
    symmetrize(weak_negative);
    result.weak_symmetry_relative =
        relative_difference(weak_negative, transpose(weak_negative));
    result.direct_pre_symmetry_relative =
        relative_difference(direct_global, transpose(direct_global));
    symmetrize(direct_global);
    result.direct_symmetry_relative =
        relative_difference(direct_global, transpose(direct_global));
    result.global_direct_positive_sign_relative =
        relative_difference(direct_global, weak_negative);
    result.global_direct_negative_sign_relative = relative_difference(
        direct_global, scaled_matrix(weak_negative, -1.0));
    result.global_direct_weak_relative =
        result.global_direct_positive_sign_relative;
    result.positive_sign_separated_relative =
        relative_difference(direct_separated, weak_negative_separated);
    result.negative_sign_separated_relative = relative_difference(
        direct_separated, scaled_matrix(weak_negative_separated, -1.0));
    result.selected_sign_separated_relative =
        result.positive_sign_separated_relative;
    const bool finite = std::isfinite(result.weak_pre_symmetry_relative) &&
                        std::isfinite(result.weak_symmetry_relative) &&
                        std::isfinite(result.direct_pre_symmetry_relative) &&
                        std::isfinite(result.direct_symmetry_relative) &&
                        std::isfinite(result.global_direct_negative_sign_relative) &&
                        std::isfinite(result.global_direct_positive_sign_relative) &&
                        std::isfinite(result.global_direct_weak_relative) &&
                        std::isfinite(result.negative_sign_separated_relative) &&
                        std::isfinite(result.positive_sign_separated_relative) &&
                        std::isfinite(result.selected_sign_separated_relative);
    if (!finite) {
      result.status = "nonfinite";
    } else if (result.weak_pre_symmetry_relative > 0.01 ||
               result.weak_symmetry_relative > 1.0e-12) {
      result.status = "weak_symmetry_gap";
    } else {
      result.status = "pass";
    }
  } catch (const std::exception &) {
    result.status = "assembly_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.assembly_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

static CompleteMixedJcpResult assemble_complete_mixed_jcp_calibration_impl(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const CompleteMixedJcpConfig &config,
    std::span<const double> action_input = {},
    std::vector<double> *action_output = nullptr) {
  const auto begin = std::chrono::steady_clock::now();
  CompleteMixedJcpResult result;
  result.p1_dofs = mesh.vertices.size();
  result.dual_p0_dofs = mesh.vertices.size();
  result.weak_form_sign = "positive_maue_for_jcp_k4_and_a_equals_jump_minus_k";
  if (!spaces.quality.valid) {
    result.status = "invalid_trace_space";
    return result;
  }
  if (mesh.vertices.empty() || mesh.triangles.empty() ||
      mesh.vertices.size() != spaces.p1_vertex_lumped_mass.size()) {
    result.status = "invalid_mesh";
    return result;
  }
  if (config.adjacent_pair_subdivision_depth < 0 ||
      config.adjacent_pair_subdivision_depth > 6 ||
      !(config.eps_ratio > 0.0) || !(config.kappa > 0.0) ||
      !(config.direct_action_relative_tolerance > 0.0) ||
      !std::isfinite(config.eps_ratio) || !std::isfinite(config.kappa) ||
      !std::isfinite(config.direct_action_relative_tolerance)) {
    result.status = "invalid_config";
    return result;
  }

  try {
    std::vector<TriangleGeometry> triangles(mesh.triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
      if (!build_triangle_geometry(mesh, index, triangles[index])) {
        result.status = "invalid_triangle_geometry";
        return result;
      }
    }
    const std::size_t n = mesh.vertices.size();
    const std::vector<DualP0Patch> dual_patches =
        build_barycentric_dual_p0_patches(triangles);
    result.dual_p0_subtriangles = dual_patches.size();

    // B maps P1 coefficients to dual-P0 tests. Its transpose maps the normal
    // trace coefficients into P1 tests, so the full mixed Riesz map is
    // diag(B, B^T), not an implicit diagonal scaling.
    DenseMatrix pairing(n, n);
    for (const DualP0Patch &patch : dual_patches) {
      const TriangleGeometry &parent = triangles[patch.parent];
      for (const TriangleQuadraturePoint &quadrature : kTriangleGauss7) {
        const Vec3 point = interpolate_patch(patch, quadrature.lambda);
        const auto lambda = barycentric_coordinates(parent, point);
        const double weight = patch.area * quadrature.weight;
        for (std::size_t local = 0; local < 3; ++local) {
          pairing(patch.owner, parent.vertex[local]) += weight * lambda[local];
        }
      }
    }
    double pairing_pivot_ratio = 0.0;
    const DenseMatrix pairing_inverse =
        solve_left(pairing, identity(n), pairing_pivot_ratio);
    result.pairing_pivot_ratio = pairing_pivot_ratio;
    result.pairing_inverse_relative = relative_difference(
        multiply(pairing, pairing_inverse), identity(n));

    // K1/K2/K3 use the existing C5C.1 mixed trace quadrature and the same
    // JCP kernel coefficients as the production near operator.
    DenseMatrix k1(n, n);
    DenseMatrix k2(n, n);
    DenseMatrix k3(n, n);
    for (const DualP0Patch &target_patch : dual_patches) {
      const TriangleGeometry &target_parent = triangles[target_patch.parent];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target =
            interpolate_patch(target_patch, target_quadrature.lambda);
        const double target_weight = target_patch.area * target_quadrature.weight;
        for (const DualP0Patch &source_patch : dual_patches) {
          JcpSourceIntegral integral;
          add_jcp_source_integral(target, target_parent.normal,
                                  triangles[source_patch.parent], source_patch.point,
                                  config.eps_ratio, config.kappa, integral);
          k1(target_patch.owner, source_patch.owner) +=
              target_weight * integral.k1;
        }
        for (std::size_t source_index = 0; source_index < triangles.size();
             ++source_index) {
          if (source_index == target_patch.parent) {
            continue;
          }
          const TriangleGeometry &source_parent = triangles[source_index];
          const int depth = shared_vertex_count(target_parent, source_parent) > 0
                                ? config.adjacent_pair_subdivision_depth
                                : 0;
          JcpSourceIntegral integral;
          add_jcp_source_integral_recursive(
              target, target_parent.normal, source_parent, source_parent.point,
              depth, config.eps_ratio, config.kappa, integral);
          for (std::size_t local = 0; local < 3; ++local) {
            k2(target_patch.owner, source_parent.vertex[local]) +=
                target_weight * integral.k2_basis[local];
          }
        }
      }
    }
    for (std::size_t target_index = 0; target_index < triangles.size();
         ++target_index) {
      const TriangleGeometry &target_parent = triangles[target_index];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target = interpolate(target_parent, target_quadrature.lambda);
        const double target_weight =
            target_parent.area * target_quadrature.weight;
        for (const DualP0Patch &source_patch : dual_patches) {
          if (source_patch.parent == target_index) {
            continue;
          }
          const TriangleGeometry &source_parent = triangles[source_patch.parent];
          const int depth = shared_vertex_count(target_parent, source_parent) > 0
                                ? config.adjacent_pair_subdivision_depth
                                : 0;
          JcpSourceIntegral integral;
          add_jcp_source_integral_recursive(
              target, target_parent.normal, source_parent, source_patch.point,
              depth, config.eps_ratio, config.kappa, integral);
          for (std::size_t local = 0; local < 3; ++local) {
            k3(target_parent.vertex[local], source_patch.owner) +=
                target_weight * target_quadrature.lambda[local] * integral.k3;
          }
        }
      }
    }

    // K4 is the C5C.2 Maue weak form and the C5C.3 global direct oracle.
    DenseMatrix k4_weak(n, n);
    DenseMatrix k4_direct(n, n);
    for (std::size_t target_index = 0; target_index < triangles.size();
         ++target_index) {
      const TriangleGeometry &target_triangle = triangles[target_index];
      for (const TriangleQuadraturePoint &target_quadrature : kTriangleGauss7) {
        const Vec3 target =
            interpolate(target_triangle, target_quadrature.lambda);
        const double target_weight =
            target_triangle.area * target_quadrature.weight;
        for (std::size_t source_index = 0; source_index < triangles.size();
             ++source_index) {
          const TriangleGeometry &source_triangle = triangles[source_index];
          const bool self_pair = target_index == source_index;
          const bool adjacent_pair =
              !self_pair &&
              shared_vertex_count(target_triangle, source_triangle) > 0;
          const bool separated_pair = !self_pair && !adjacent_pair;
          ScreenedSourceIntegral screened;
          if (self_pair) {
            add_regular_screened_source_integral(
                target, source_triangle, source_triangle.point, config.kappa,
                false, true, screened);
            add_duffy_screened_green_integral(target, source_triangle,
                                               config.kappa, screened);
          } else {
            const int depth = adjacent_pair
                                  ? config.adjacent_pair_subdivision_depth
                                  : 0;
            add_regular_screened_source_integral_recursive(
                target, source_triangle, source_triangle.point, depth,
                config.kappa, true, true, screened);
          }
          JcpSourceIntegral direct;
          if (self_pair) {
            add_duffy_self_jcp_k4_integral(target, source_triangle,
                                            config.kappa, direct);
          } else {
            const int depth = adjacent_pair
                                  ? config.adjacent_pair_subdivision_depth
                                  : 0;
            add_jcp_source_integral_recursive(
                target, target_triangle.normal, source_triangle,
                source_triangle.point, depth, 1.0, config.kappa, direct);
          }
          if (separated_pair) {
            ++result.separated_source_evaluations;
          }
          const double normal_dot =
              dot(target_triangle.normal, source_triangle.normal);
          for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t row = target_triangle.vertex[i];
            const double target_basis = target_quadrature.lambda[i];
            for (std::size_t j = 0; j < 3; ++j) {
              const std::size_t column = source_triangle.vertex[j];
              const double maue =
                  dot(target_triangle.surface_curl[i],
                      source_triangle.surface_curl[j]) *
                      screened.green_difference_constant +
                  config.kappa * config.kappa * normal_dot * target_basis *
                      screened.green_kappa_basis[j];
              k4_weak(row, column) += target_weight * maue;
              k4_direct(row, column) +=
                  target_weight * target_basis * direct.k4_basis[j];
            }
          }
        }
      }
    }
    if (result.separated_source_evaluations == 0) {
      result.status = "missing_separated_pairs";
      return result;
    }
    symmetrize(k4_weak);
    symmetrize(k4_direct);
    result.direct_k4_weak_relative = relative_difference(k4_direct, k4_weak);

    const DenseMatrix zero(n, n);
    const DenseMatrix riesz = block_2x2(pairing, zero, zero, transpose(pairing));
    const DenseMatrix interaction_weak = block_2x2(k2, k1, k4_weak, k3);
    const DenseMatrix interaction_direct = block_2x2(k2, k1, k4_direct, k3);
    result.lhs_phi = 0.5 * (1.0 + config.eps_ratio);
    result.lhs_phin = 0.5 * (1.0 + 1.0 / config.eps_ratio);
    const DenseMatrix lhs = block_2x2(
        scaled_matrix(pairing, result.lhs_phi), zero, zero,
        scaled_matrix(transpose(pairing), result.lhs_phin));
    const DenseMatrix system_weak = subtract(lhs, interaction_weak);
    const DenseMatrix system_direct = subtract(lhs, interaction_direct);

    double riesz_pivot_ratio = 0.0;
    const DenseMatrix strong_interaction =
        solve_left(riesz, interaction_weak, riesz_pivot_ratio);
    const DenseMatrix strong_direct_interaction =
        solve_left(riesz, interaction_direct, riesz_pivot_ratio);
    const DenseMatrix strong_system =
        solve_left(riesz, system_weak, riesz_pivot_ratio);
    const DenseMatrix strong_direct_system =
        solve_left(riesz, system_direct, riesz_pivot_ratio);
    if (action_output != nullptr) {
      if (action_input.size() != 2 * n) {
        result.status = "invalid_action_size";
        return result;
      }
      *action_output = multiply(strong_system, action_input);
    }
    result.strong_interaction_relative = relative_difference(
        strong_interaction, strong_direct_interaction);
    result.strong_system_relative =
        relative_difference(strong_system, strong_direct_system);
    const DenseMatrix probes = deterministic_mixed_probes(2 * n);
    result.deterministic_probe_relative = relative_difference(
        multiply(strong_system, probes), multiply(strong_direct_system, probes));

    const bool finite = std::isfinite(result.lhs_phi) &&
                        std::isfinite(result.lhs_phin) &&
                        std::isfinite(result.pairing_pivot_ratio) &&
                        std::isfinite(result.pairing_inverse_relative) &&
                        std::isfinite(result.direct_k4_weak_relative) &&
                        std::isfinite(result.strong_interaction_relative) &&
                        std::isfinite(result.strong_system_relative) &&
                        std::isfinite(result.deterministic_probe_relative);
    if (!finite) {
      result.status = "nonfinite";
    } else if (result.pairing_pivot_ratio < 1.0e-12 ||
               result.pairing_inverse_relative > 1.0e-11) {
      result.status = "mixed_riesz_pairing_gap";
    } else if (result.direct_k4_weak_relative >
                   config.direct_action_relative_tolerance ||
               result.strong_interaction_relative >
                   config.direct_action_relative_tolerance ||
               result.strong_system_relative >
                   config.direct_action_relative_tolerance ||
               result.deterministic_probe_relative >
                   config.direct_action_relative_tolerance) {
      result.status = "global_direct_action_gap";
    } else {
      result.status = "pass";
    }
  } catch (const std::exception &) {
    result.status = "assembly_failure";
  }
  const auto end = std::chrono::steady_clock::now();
  result.assembly_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

CompleteMixedJcpResult assemble_complete_mixed_jcp_calibration(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    const CompleteMixedJcpConfig &config) {
  return assemble_complete_mixed_jcp_calibration_impl(mesh, spaces, config);
}

CompleteMixedJcpActionResult apply_complete_mixed_jcp_dense_reference(
    const SurfaceMesh &mesh, const TraceSpaces &spaces,
    std::span<const double> unknowns, const CompleteMixedJcpConfig &config) {
  CompleteMixedJcpActionResult action;
  if (unknowns.size() != 2 * mesh.vertices.size()) {
    action.status = "invalid_action_size";
    return action;
  }
  action.calibration = assemble_complete_mixed_jcp_calibration_impl(
      mesh, spaces, config, unknowns, &action.action);
  action.status = action.calibration.status;
  if (action.status != "pass") {
    action.action.clear();
  }
  return action;
}

} // namespace TitanPBE::Tabi::calderon

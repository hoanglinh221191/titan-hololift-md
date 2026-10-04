#include "titan_rotational_diffusion.h"

#include "BoxMatrix.h"
#include "titan_executor.h"
#include "titan_ui.h"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace titan_rotational_diffusion {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kPi = 3.14159265358979323846;

Eigen::Matrix3d covariance_matrix(const Covariance3 &covariance) {
  Eigen::Matrix3d matrix;
  matrix << covariance.values[0], covariance.values[3], covariance.values[4],
      covariance.values[3], covariance.values[1], covariance.values[5],
      covariance.values[4], covariance.values[5], covariance.values[2];
  return matrix;
}

bool valid_time_axis(std::span<const double> time_ps) {
  for (std::size_t i = 0; i < time_ps.size(); ++i) {
    if (!std::isfinite(time_ps[i]) ||
        (i > 0 && !(time_ps[i] > time_ps[i - 1])))
      return false;
  }
  return true;
}

bool physically_valid_covariance(const Covariance3 &covariance) {
  const Eigen::Matrix3d matrix = covariance_matrix(covariance);
  if (!matrix.allFinite() || matrix.trace() > 1.0 + 1.0e-10)
    return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(matrix);
  return eigen.info() == Eigen::Success &&
         eigen.eigenvalues().minCoeff() >= -1.0e-10;
}

Covariance3 covariance_values(const Eigen::Matrix3d &matrix) {
  return Covariance3{{matrix(0, 0), matrix(1, 1), matrix(2, 2), matrix(0, 1),
                      matrix(0, 2), matrix(1, 2)}};
}

Eigen::Matrix3d axes_matrix(const std::array<double, 9> &axes) {
  Eigen::Matrix3d matrix;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      matrix(row, col) = axes[static_cast<std::size_t>(row * 3 + col)];
  return matrix;
}

std::array<double, 9> axes_values(const Eigen::Matrix3d &axes) {
  std::array<double, 9> values{};
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      values[static_cast<std::size_t>(row * 3 + col)] = axes(row, col);
  return values;
}

Eigen::Matrix3d canonicalize_axes(Eigen::Matrix3d axes) {
  for (int row = 0; row < 2; ++row) {
    int largest = 0;
    axes.row(row).cwiseAbs().maxCoeff(&largest);
    if (axes(row, largest) < 0.0)
      axes.row(row) *= -1.0;
  }
  if (axes.determinant() < 0.0)
    axes.row(2) *= -1.0;
  return axes;
}

std::array<double, 3>
instantaneous_diffusion(double lag_time_ps, const Eigen::Vector3d &eigenvalues,
                        bool &valid);

double horizontal_sum(__m256d values) {
  alignas(32) std::array<double, 4> lanes{};
  _mm256_store_pd(lanes.data(), values);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

std::vector<LagResult>
compute_lag_results_soa(std::span<const double> qw, std::span<const double> qx,
                        std::span<const double> qy, std::span<const double> qz,
                        std::span<const double> time_ps, std::size_t maxlag,
                        TitanExecutor &executor) {
  const std::size_t frames = qw.size();
  if (frames < 2 || qx.size() != frames || qy.size() != frames ||
      qz.size() != frames || time_ps.size() != frames ||
      !valid_time_axis(time_ps))
    return {};
  maxlag = std::min(maxlag, frames - 1);
  std::vector<LagResult> results(maxlag);
  executor.parallel_for(maxlag, std::max<std::size_t>(1, executor.capacity()),
                        8, [&](std::size_t, std::size_t begin,
                               std::size_t end) {
    for (std::size_t lag0 = begin; lag0 < end; ++lag0) {
      const std::size_t lag = lag0 + 1;
      const std::size_t origins = frames - lag;
      LagResult row;
      row.lag_index = lag;
      std::size_t origin = 0;
#ifdef __AVX2__
      __m256d sum_xx = _mm256_setzero_pd();
      __m256d sum_yy = _mm256_setzero_pd();
      __m256d sum_zz = _mm256_setzero_pd();
      __m256d sum_xy = _mm256_setzero_pd();
      __m256d sum_xz = _mm256_setzero_pd();
      __m256d sum_yz = _mm256_setzero_pd();
      alignas(32) std::array<double, 4> lane_w{};
      alignas(32) std::array<double, 4> lane_x{};
      alignas(32) std::array<double, 4> lane_y{};
      alignas(32) std::array<double, 4> lane_z{};
      for (; origin + 4 <= origins; origin += 4) {
        const __m256d w1 = _mm256_loadu_pd(qw.data() + origin);
        const __m256d x1 = _mm256_loadu_pd(qx.data() + origin);
        const __m256d y1 = _mm256_loadu_pd(qy.data() + origin);
        const __m256d z1 = _mm256_loadu_pd(qz.data() + origin);
        const __m256d w2 = _mm256_loadu_pd(qw.data() + origin + lag);
        const __m256d x2 = _mm256_loadu_pd(qx.data() + origin + lag);
        const __m256d y2 = _mm256_loadu_pd(qy.data() + origin + lag);
        const __m256d z2 = _mm256_loadu_pd(qz.data() + origin + lag);
        __m256d rw = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(w1, w2), _mm256_mul_pd(x1, x2)),
            _mm256_add_pd(_mm256_mul_pd(y1, y2), _mm256_mul_pd(z1, z2)));
        __m256d rx = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(x1, w2), _mm256_mul_pd(w1, x2)),
            _mm256_sub_pd(_mm256_mul_pd(z1, y2), _mm256_mul_pd(y1, z2)));
        __m256d ry = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(y1, w2), _mm256_mul_pd(w1, y2)),
            _mm256_sub_pd(_mm256_mul_pd(x1, z2), _mm256_mul_pd(z1, x2)));
        __m256d rz = _mm256_add_pd(
            _mm256_sub_pd(_mm256_mul_pd(z1, w2), _mm256_mul_pd(w1, z2)),
            _mm256_sub_pd(_mm256_mul_pd(y1, x2), _mm256_mul_pd(x1, y2)));
        _mm256_store_pd(lane_w.data(), rw);
        _mm256_store_pd(lane_x.data(), rx);
        _mm256_store_pd(lane_y.data(), ry);
        _mm256_store_pd(lane_z.data(), rz);
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const double norm = std::sqrt(
              lane_w[lane] * lane_w[lane] + lane_x[lane] * lane_x[lane] +
              lane_y[lane] * lane_y[lane] + lane_z[lane] * lane_z[lane]);
          if (norm > kEpsilon) {
            lane_w[lane] /= norm;
            lane_x[lane] /= norm;
            lane_y[lane] /= norm;
            lane_z[lane] /= norm;
          }
          const double theta =
              2.0 * std::acos(std::clamp(std::abs(lane_w[lane]), 0.0, 1.0));
          const double cosine = std::cos(theta);
          row.rotational_msd_rad2 += theta * theta;
          row.c1 += cosine;
          row.c2 += 0.5 * (3.0 * cosine * cosine - 1.0);
          row.lag_time_ps += time_ps[origin + lane + lag] -
                             time_ps[origin + lane];
        }
        rx = _mm256_load_pd(lane_x.data());
        ry = _mm256_load_pd(lane_y.data());
        rz = _mm256_load_pd(lane_z.data());
        sum_xx = _mm256_add_pd(sum_xx, _mm256_mul_pd(rx, rx));
        sum_yy = _mm256_add_pd(sum_yy, _mm256_mul_pd(ry, ry));
        sum_zz = _mm256_add_pd(sum_zz, _mm256_mul_pd(rz, rz));
        sum_xy = _mm256_add_pd(sum_xy, _mm256_mul_pd(rx, ry));
        sum_xz = _mm256_add_pd(sum_xz, _mm256_mul_pd(rx, rz));
        sum_yz = _mm256_add_pd(sum_yz, _mm256_mul_pd(ry, rz));
        row.samples += 4;
      }
      row.covariance.values[0] += horizontal_sum(sum_xx);
      row.covariance.values[1] += horizontal_sum(sum_yy);
      row.covariance.values[2] += horizontal_sum(sum_zz);
      row.covariance.values[3] += horizontal_sum(sum_xy);
      row.covariance.values[4] += horizontal_sum(sum_xz);
      row.covariance.values[5] += horizontal_sum(sum_yz);
#endif
      for (; origin < origins; ++origin) {
        const Quaternion relative = normalized(
            multiply(Quaternion{qw[origin], qx[origin], qy[origin], qz[origin]},
                     conjugate(Quaternion{qw[origin + lag], qx[origin + lag],
                                          qy[origin + lag],
                                          qz[origin + lag]})));
        const double theta =
            2.0 * std::acos(std::clamp(std::abs(relative.w), 0.0, 1.0));
        const double cosine = std::cos(theta);
        row.rotational_msd_rad2 += theta * theta;
        row.c1 += cosine;
        row.c2 += 0.5 * (3.0 * cosine * cosine - 1.0);
        row.covariance.values[0] += relative.x * relative.x;
        row.covariance.values[1] += relative.y * relative.y;
        row.covariance.values[2] += relative.z * relative.z;
        row.covariance.values[3] += relative.x * relative.y;
        row.covariance.values[4] += relative.x * relative.z;
        row.covariance.values[5] += relative.y * relative.z;
        row.lag_time_ps += time_ps[origin + lag] - time_ps[origin];
        ++row.samples;
      }
      if (row.samples > 0) {
        const double denom = static_cast<double>(row.samples);
        row.rotational_msd_rad2 /= denom;
        row.c1 /= denom;
        row.c2 /= denom;
        row.orientation_correlation = row.c1;
        row.lag_time_ps /= denom;
        for (double &value : row.covariance.values)
          value /= denom;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
            covariance_matrix(row.covariance));
        if (eigen.info() == Eigen::Success) {
          row.instantaneous_d_per_ps = instantaneous_diffusion(
              row.lag_time_ps, eigen.eigenvalues(), row.instantaneous_valid);
          row.instantaneous_axes =
              axes_values(canonicalize_axes(eigen.eigenvectors().transpose()));
        }
      }
      results[lag0] = row;
    }
  });
  return results;
}

bool finite_positive(const std::array<double, 3> &values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value > 0.0;
  });
}

Covariance3 favro_model(double lag_time_ps, const std::array<double, 3> &d,
                       const Eigen::Matrix3d &axes) {
  Eigen::Matrix3d diagonal = Eigen::Matrix3d::Zero();
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;
    diagonal(i, i) =
        0.25 *
        (1.0 + std::exp(-(d[j] + d[k]) * lag_time_ps) -
         std::exp(-(d[i] + d[j]) * lag_time_ps) -
         std::exp(-(d[i] + d[k]) * lag_time_ps));
  }
  return covariance_values(axes.transpose() * diagonal * axes);
}

std::array<double, 3>
instantaneous_diffusion(double lag_time_ps, const Eigen::Vector3d &eigenvalues,
                        bool &valid) {
  std::array<double, 3> d{};
  valid = lag_time_ps > 0.0;
  std::array<double, 3> exponents{
      1.0 - 2.0 * eigenvalues[1] - 2.0 * eigenvalues[2],
      1.0 - 2.0 * eigenvalues[2] - 2.0 * eigenvalues[0],
      1.0 - 2.0 * eigenvalues[0] - 2.0 * eigenvalues[1]};
  for (double exponent : exponents)
    valid = valid && std::isfinite(exponent) && exponent > kEpsilon;
  if (!valid)
    return d;
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const int k = (i + 2) % 3;
    d[static_cast<std::size_t>(i)] =
        -std::log(exponents[j] * exponents[k] / exponents[i]) /
        (2.0 * lag_time_ps);
  }
  valid = finite_positive(d);
  return d;
}

double covariance_residual(const std::vector<LagResult> &rows,
                           std::size_t begin, std::size_t end,
                           const Eigen::Matrix<double, 6, 1> &params) {
  const std::array<double, 3> d{std::exp(params[0]), std::exp(params[1]),
                                std::exp(params[2])};
  const Eigen::Vector3d rotation = params.tail<3>();
  const double angle = rotation.norm();
  Eigen::Matrix3d axes = Eigen::Matrix3d::Identity();
  if (angle > kEpsilon)
    axes = Eigen::AngleAxisd(angle, rotation / angle).toRotationMatrix();
  double residual = 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    const Covariance3 model = favro_model(rows[i].lag_time_ps, d, axes);
    for (std::size_t lane = 0; lane < 6; ++lane) {
      const double delta = model.values[lane] - rows[i].covariance.values[lane];
      residual += delta * delta;
    }
  }
  return residual;
}

Eigen::VectorXd covariance_residual_vector(
    const std::vector<LagResult> &rows, std::size_t begin, std::size_t end,
    const Eigen::Matrix<double, 6, 1> &params) {
  const std::array<double, 3> d{std::exp(params[0]), std::exp(params[1]),
                                std::exp(params[2])};
  const Eigen::Vector3d rotation = params.tail<3>();
  const double angle = rotation.norm();
  Eigen::Matrix3d axes = Eigen::Matrix3d::Identity();
  if (angle > kEpsilon)
    axes = Eigen::AngleAxisd(angle, rotation / angle).toRotationMatrix();
  Eigen::VectorXd residuals(static_cast<Eigen::Index>((end - begin) * 6));
  Eigen::Index output = 0;
  for (std::size_t i = begin; i < end; ++i) {
    const Covariance3 model = favro_model(rows[i].lag_time_ps, d, axes);
    for (std::size_t lane = 0; lane < 6; ++lane)
      residuals[output++] =
          model.values[lane] - rows[i].covariance.values[lane];
  }
  return residuals;
}

std::pair<std::size_t, std::size_t>
resolve_fit_window(const std::vector<LagResult> &results,
                   const std::string &fit_window, std::string &policy) {
  if (results.empty())
    return {0, 0};
  if (fit_window != "auto") {
    const auto comma = fit_window.find(',');
    if (comma == std::string::npos)
      throw std::invalid_argument(
          "MSD: -msd-rot-fit-window must be auto or tmin,tmax.");
    const double tmin = std::stod(fit_window.substr(0, comma));
    const double tmax = std::stod(fit_window.substr(comma + 1));
    if (!(tmax > tmin))
      throw std::invalid_argument(
          "MSD: -msd-rot-fit-window requires tmax > tmin.");
    std::size_t begin = 0;
    while (begin < results.size() && results[begin].lag_time_ps < tmin)
      ++begin;
    std::size_t end = begin;
    while (end < results.size() && results[end].lag_time_ps <= tmax)
      ++end;
    policy = "explicit";
    return {begin, end};
  }

  constexpr std::size_t stable_rows = 5;
  if (results.size() >= stable_rows) {
    for (std::size_t begin = 0; begin + stable_rows <= results.size();
         ++begin) {
      bool valid = true;
      std::array<double, 3> mean{};
      for (std::size_t i = begin; i < begin + stable_rows; ++i) {
        valid = valid && results[i].instantaneous_valid;
        for (std::size_t axis = 0; axis < 3; ++axis)
          mean[axis] += results[i].instantaneous_d_per_ps[axis];
      }
      if (!valid)
        continue;
      for (double &value : mean)
        value /= static_cast<double>(stable_rows);
      for (std::size_t axis = 0; axis < 3 && valid; ++axis) {
        double variance = 0.0;
        for (std::size_t i = begin; i < begin + stable_rows; ++i) {
          const double delta = results[i].instantaneous_d_per_ps[axis] -
                               mean[axis];
          variance += delta * delta;
        }
        const double cv =
            std::sqrt(variance / static_cast<double>(stable_rows)) / mean[axis];
        valid = std::isfinite(cv) && cv <= 0.10;
      }
      const Eigen::Matrix3d first =
          axes_matrix(results[begin].instantaneous_axes);
      for (std::size_t i = begin + 1; i < begin + stable_rows && valid; ++i) {
        const Eigen::Matrix3d current =
            axes_matrix(results[i].instantaneous_axes);
        for (int axis = 0; axis < 3; ++axis) {
          const double dot =
              std::clamp(std::abs(first.row(axis).dot(current.row(axis))), 0.0,
                         1.0);
          valid = valid && std::acos(dot) * 180.0 / kPi <= 10.0;
        }
      }
      if (valid) {
        policy = "auto_stable";
        return {begin, begin + stable_rows};
      }
    }
  }

  const std::size_t fallback =
      std::min<std::size_t>(10, std::max<std::size_t>(
                                    3, static_cast<std::size_t>(
                                           std::ceil(0.2 * results.size()))));
  policy = "auto_fallback_early";
  return {0, std::min(results.size(), fallback)};
}

Quaternion quaternion_from_matrix(const Eigen::Matrix3d &matrix) {
  Eigen::Quaterniond eigen_quaternion(matrix);
  eigen_quaternion.normalize();
  Quaternion quaternion{eigen_quaternion.w(), eigen_quaternion.x(),
                        eigen_quaternion.y(), eigen_quaternion.z()};
  if (quaternion.w < 0.0) {
    quaternion.w *= -1.0;
    quaternion.x *= -1.0;
    quaternion.y *= -1.0;
    quaternion.z *= -1.0;
  }
  return quaternion;
}

std::vector<Point3> reconstruct_points(std::span<const Point3> points,
                                       const box *frame_box) {
  std::vector<Point3> reconstructed(points.begin(), points.end());
  if (frame_box == nullptr || reconstructed.empty())
    return reconstructed;
  const BoxMatrix box_matrix(*frame_box);
  for (std::size_t i = 1; i < reconstructed.size(); ++i) {
    double dx = reconstructed[i].x - reconstructed[0].x;
    double dy = reconstructed[i].y - reconstructed[0].y;
    double dz = reconstructed[i].z - reconstructed[0].z;
    box_matrix.apply_mic(dx, dy, dz);
    reconstructed[i] = Point3{reconstructed[0].x + dx,
                              reconstructed[0].y + dy,
                              reconstructed[0].z + dz};
  }
  return reconstructed;
}

std::vector<std::vector<Point3>>
unwrap_bodies(const std::vector<std::vector<Point3>> &bodies,
              const std::vector<box> &boxes) {
  std::vector<std::vector<Point3>> unwrapped = bodies;
  if (bodies.empty())
    return unwrapped;

  unwrapped.front() = reconstruct_points(bodies.front(), &boxes.front());
  for (std::size_t frame = 1; frame < bodies.size(); ++frame) {
    const BoxMatrix previous_box(boxes[frame - 1]);
    for (std::size_t atom = 0; atom < bodies[frame].size(); ++atom) {
      double dx = bodies[frame][atom].x - bodies[frame - 1][atom].x;
      double dy = bodies[frame][atom].y - bodies[frame - 1][atom].y;
      double dz = bodies[frame][atom].z - bodies[frame - 1][atom].z;
      previous_box.apply_mic(dx, dy, dz);
      unwrapped[frame][atom] =
          Point3{unwrapped[frame - 1][atom].x + dx,
                 unwrapped[frame - 1][atom].y + dy,
                 unwrapped[frame - 1][atom].z + dz};
    }
  }
  return unwrapped;
}

} // namespace

Quaternion normalized(Quaternion q) {
  const double norm =
      std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (!(norm > kEpsilon))
    return Quaternion::identity();
  q.w /= norm;
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  return q;
}

Quaternion canonicalize_hemisphere(Quaternion q, const Quaternion &previous) {
  if (q.w * previous.w + q.x * previous.x + q.y * previous.y +
          q.z * previous.z <
      0.0) {
    q.w *= -1.0;
    q.x *= -1.0;
    q.y *= -1.0;
    q.z *= -1.0;
  }
  return q;
}

Quaternion conjugate(const Quaternion &q) noexcept {
  return Quaternion{q.w, -q.x, -q.y, -q.z};
}

Quaternion multiply(const Quaternion &lhs, const Quaternion &rhs) noexcept {
  return Quaternion{
      lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
      lhs.w * rhs.x + rhs.w * lhs.x + lhs.y * rhs.z - lhs.z * rhs.y,
      lhs.w * rhs.y + rhs.w * lhs.y + lhs.z * rhs.x - lhs.x * rhs.z,
      lhs.w * rhs.z + rhs.w * lhs.z + lhs.x * rhs.y - lhs.y * rhs.x};
}

double angular_distance_rad(const Quaternion &lhs, const Quaternion &rhs) {
  const Quaternion relative = normalized(multiply(lhs, conjugate(rhs)));
  return 2.0 * std::acos(std::clamp(std::abs(relative.w), 0.0, 1.0));
}

OrientationFit fit_orientation(std::span<const Point3> reference,
                               std::span<const Point3> mobile,
                               std::span<const double> masses,
                               const box *frame_box) {
  OrientationFit fit;
  if (reference.size() < 3 || mobile.size() != reference.size())
    return fit;
  if (frame_box != nullptr && !BoxMatrix::valid_box(*frame_box))
    return fit;

  const std::vector<Point3> reconstructed =
      reconstruct_points(mobile, frame_box);
  const std::vector<Point3> reconstructed_reference =
      reconstruct_points(reference, frame_box);

  double mass_sum = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i)
    mass_sum += masses.size() == reference.size() && masses[i] > 0.0
                    ? masses[i]
                    : 0.0;
  const bool unit_weights = !(mass_sum > 0.0);
  if (unit_weights)
    mass_sum = static_cast<double>(reference.size());

  Eigen::Vector3d ref_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d mob_center = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double weight =
        unit_weights ? 1.0 : (masses[i] > 0.0 ? masses[i] : 0.0);
    ref_center += weight *
                  Eigen::Vector3d(reconstructed_reference[i].x,
                                  reconstructed_reference[i].y,
                                  reconstructed_reference[i].z);
    mob_center += weight * Eigen::Vector3d(reconstructed[i].x,
                                           reconstructed[i].y,
                                           reconstructed[i].z);
  }
  ref_center /= mass_sum;
  mob_center /= mass_sum;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  Eigen::MatrixXd weighted_reference(reference.size(), 3);
  Eigen::MatrixXd weighted_mobile(reference.size(), 3);
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double weight =
        unit_weights ? 1.0 : (masses[i] > 0.0 ? masses[i] : 0.0);
    const Eigen::Vector3d ref =
        Eigen::Vector3d(reconstructed_reference[i].x,
                        reconstructed_reference[i].y,
                        reconstructed_reference[i].z) -
        ref_center;
    const Eigen::Vector3d mob = Eigen::Vector3d(reconstructed[i].x,
                                                reconstructed[i].y,
                                                reconstructed[i].z) -
                                mob_center;
    weighted_reference.row(static_cast<Eigen::Index>(i)) =
        std::sqrt(weight) * ref;
    weighted_mobile.row(static_cast<Eigen::Index>(i)) =
        std::sqrt(weight) * mob;
    covariance += weight * mob * ref.transpose();
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> reference_rank_svd(weighted_reference);
  Eigen::JacobiSVD<Eigen::MatrixXd> mobile_rank_svd(weighted_mobile);
  if (reference_rank_svd.rank() < 2 || mobile_rank_svd.rank() < 2)
    return fit;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,
                                        Eigen::ComputeFullU |
                                            Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  Eigen::Matrix3d v = svd.matrixV();
  if ((u * v.transpose()).determinant() < 0.0)
    u.col(2) *= -1.0;
  const Eigen::Matrix3d align = u * v.transpose();
  double weighted_d2 = 0.0;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double weight =
        unit_weights ? 1.0 : (masses[i] > 0.0 ? masses[i] : 0.0);
    const Eigen::Vector3d ref =
        Eigen::Vector3d(reconstructed_reference[i].x,
                        reconstructed_reference[i].y,
                        reconstructed_reference[i].z) -
        ref_center;
    const Eigen::Vector3d mob = Eigen::Vector3d(reconstructed[i].x,
                                                reconstructed[i].y,
                                                reconstructed[i].z) -
                                mob_center;
    weighted_d2 += weight * (align * mob - ref).squaredNorm();
  }
  fit.valid = true;
  fit.orientation = quaternion_from_matrix(align.transpose());
  fit.rmsd_a = std::sqrt(weighted_d2 / mass_sum);
  return fit;
}

Covariance3 covariance_scalar(const Point3 *reference, const Point3 *frame,
                              std::size_t count) {
  Covariance3 covariance{};
  if (count == 0)
    return covariance;
  for (std::size_t i = 0; i < count; ++i) {
    const double x = frame[i].x - reference[i].x;
    const double y = frame[i].y - reference[i].y;
    const double z = frame[i].z - reference[i].z;
    covariance.values[0] += x * x;
    covariance.values[1] += y * y;
    covariance.values[2] += z * z;
    covariance.values[3] += x * y;
    covariance.values[4] += x * z;
    covariance.values[5] += y * z;
  }
  for (double &value : covariance.values)
    value /= static_cast<double>(count);
  return covariance;
}

Covariance3 covariance_optimized(const Point3 *reference, const Point3 *frame,
                                 std::size_t count) {
  // Point3 is AoS and may be unaligned. The production quaternion path below
  // is the aligned AVX2 hot path; keep this diagnostic helper portable.
  return covariance_scalar(reference, frame, count);
}

Covariance3 favro_covariance_model(double lag_time_ps,
                                   const std::array<double, 3> &d,
                                   const std::array<double, 9> &axes) {
  return favro_model(lag_time_ps, d, axes_matrix(axes));
}

void update_instantaneous_tensor(LagResult &row) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(
      covariance_matrix(row.covariance));
  if (eigen.info() != Eigen::Success) {
    row.instantaneous_valid = false;
    return;
  }
  row.instantaneous_d_per_ps =
      instantaneous_diffusion(row.lag_time_ps, eigen.eigenvalues(),
                              row.instantaneous_valid);
  row.instantaneous_axes =
      axes_values(canonicalize_axes(eigen.eigenvectors().transpose()));
}

std::vector<LagResult>
compute_lag_results(std::span<const Quaternion> orientations,
                    std::span<const double> time_ps, std::size_t maxlag,
                    TitanExecutor &executor) {
  AlignedDoubleVector qw, qx, qy, qz;
  qw.reserve(orientations.size());
  qx.reserve(orientations.size());
  qy.reserve(orientations.size());
  qz.reserve(orientations.size());
  for (const Quaternion &orientation : orientations) {
    qw.push_back(orientation.w);
    qx.push_back(orientation.x);
    qy.push_back(orientation.y);
    qz.push_back(orientation.z);
  }
  return compute_lag_results_soa(qw, qx, qy, qz, time_ps, maxlag, executor);
}

std::vector<LagResult>
compute_lag_results_scalar(std::span<const Quaternion> orientations,
                           std::span<const double> time_ps,
                           std::size_t maxlag) {
  const std::size_t frames = orientations.size();
  if (frames < 2 || time_ps.size() != frames || !valid_time_axis(time_ps))
    return {};
  maxlag = std::min(maxlag, frames - 1);
  std::vector<LagResult> results(maxlag);
  for (std::size_t lag0 = 0; lag0 < maxlag; ++lag0) {
    LagResult &row = results[lag0];
    row.lag_index = lag0 + 1;
    for (std::size_t origin = 0; origin + row.lag_index < frames; ++origin) {
      const Quaternion relative = normalized(
          multiply(orientations[origin],
                   conjugate(orientations[origin + row.lag_index])));
      const double theta =
          2.0 * std::acos(std::clamp(std::abs(relative.w), 0.0, 1.0));
      const double cosine = std::cos(theta);
      row.rotational_msd_rad2 += theta * theta;
      row.c1 += cosine;
      row.c2 += 0.5 * (3.0 * cosine * cosine - 1.0);
      row.covariance.values[0] += relative.x * relative.x;
      row.covariance.values[1] += relative.y * relative.y;
      row.covariance.values[2] += relative.z * relative.z;
      row.covariance.values[3] += relative.x * relative.y;
      row.covariance.values[4] += relative.x * relative.z;
      row.covariance.values[5] += relative.y * relative.z;
      row.lag_time_ps +=
          time_ps[origin + row.lag_index] - time_ps[origin];
      ++row.samples;
    }
    if (row.samples > 0) {
      const double denom = static_cast<double>(row.samples);
      row.rotational_msd_rad2 /= denom;
      row.c1 /= denom;
      row.c2 /= denom;
      row.orientation_correlation = row.c1;
      row.lag_time_ps /= denom;
      for (double &value : row.covariance.values)
        value /= denom;
      update_instantaneous_tensor(row);
    }
  }
  return results;
}

TensorFit fit_tensor(std::vector<LagResult> &results,
                     const std::string &fit_window) {
  TensorFit fit;
  for (LagResult &row : results)
    if (!row.instantaneous_valid)
      update_instantaneous_tensor(row);
  if (fit_window != "auto") {
    const auto comma = fit_window.find(',');
    if (comma == std::string::npos)
      throw std::invalid_argument(
          "MSD: -msd-rot-fit-window must be auto or tmin,tmax.");
    fit.tmin_ps = std::stod(fit_window.substr(0, comma));
    fit.tmax_ps = std::stod(fit_window.substr(comma + 1));
  }
  const auto [begin, end] =
      resolve_fit_window(results, fit_window, fit.window_policy);
  fit.points = end > begin ? end - begin : 0;
  if (fit.points > 0) {
    fit.tmin_ps = results[begin].lag_time_ps;
    fit.tmax_ps = results[end - 1].lag_time_ps;
  }
  if (fit.points < 3) {
    fit.status = "insufficient_points";
    return fit;
  }
  for (std::size_t i = begin; i < end; ++i) {
    if (!std::isfinite(results[i].lag_time_ps) ||
        !(results[i].lag_time_ps > 0.0) ||
        !physically_valid_covariance(results[i].covariance)) {
      fit.status = "invalid_fit_data";
      return fit;
    }
  }

  std::array<double, 3> initial{1.0e-6, 1.0e-6, 1.0e-6};
  for (std::size_t axis = 0; axis < 3; ++axis) {
    std::vector<double> values;
    for (std::size_t i = begin; i < end; ++i)
      if (results[i].instantaneous_valid)
        values.push_back(results[i].instantaneous_d_per_ps[axis]);
    if (!values.empty()) {
      std::sort(values.begin(), values.end());
      initial[axis] = std::max(values[values.size() / 2], 1.0e-12);
    }
  }

  Eigen::Matrix<double, 6, 1> params;
  params << std::log(initial[0]), std::log(initial[1]), std::log(initial[2]),
      0.0, 0.0, 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    if (!results[i].instantaneous_valid)
      continue;
    const Eigen::Matrix3d initial_axes =
        axes_matrix(results[i].instantaneous_axes);
    const Eigen::AngleAxisd angle_axis(initial_axes);
    params.tail<3>() = angle_axis.axis() * angle_axis.angle();
    break;
  }
  double lambda = 1.0e-4;
  double best = covariance_residual(results, begin, end, params);
  bool converged = false;
  bool solver_failed = !std::isfinite(best);
  for (int iteration = 0; iteration < 100; ++iteration) {
    if (solver_failed)
      break;
    const Eigen::VectorXd residuals =
        covariance_residual_vector(results, begin, end, params);
    if (!residuals.allFinite()) {
      solver_failed = true;
      break;
    }
    Eigen::MatrixXd jacobian(residuals.size(), 6);
    constexpr double step = 1.0e-5;
    for (int p = 0; p < 6; ++p) {
      auto trial = params;
      trial[p] += step;
      const Eigen::VectorXd plus =
          covariance_residual_vector(results, begin, end, trial);
      trial[p] -= 2.0 * step;
      const Eigen::VectorXd minus =
          covariance_residual_vector(results, begin, end, trial);
      jacobian.col(p) = (plus - minus) / (2.0 * step);
    }
    if (!jacobian.allFinite()) {
      solver_failed = true;
      break;
    }
    const Eigen::Matrix<double, 6, 1> gradient =
        jacobian.transpose() * residuals;
    if (gradient.norm() <= 1.0e-12) {
      converged = true;
      break;
    }
    Eigen::Matrix<double, 6, 6> hessian =
        jacobian.transpose() * jacobian;
    for (int p = 0; p < 6; ++p)
      hessian(p, p) += lambda;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(hessian);
    if (ldlt.info() != Eigen::Success) {
      solver_failed = true;
      break;
    }
    const Eigen::Matrix<double, 6, 1> delta = -ldlt.solve(gradient);
    if (ldlt.info() != Eigen::Success || !delta.allFinite()) {
      solver_failed = true;
      break;
    }
    if (delta.norm() <= 1.0e-12 * std::max(1.0, params.norm())) {
      converged = true;
      break;
    }
    auto trial = params + delta;
    const double residual = covariance_residual(results, begin, end, trial);
    if (residual < best) {
      params = trial;
      if (std::abs(best - residual) <= 1.0e-14 * std::max(1.0, best)) {
        best = residual;
        converged = true;
        break;
      }
      best = residual;
      lambda *= 0.5;
    } else {
      lambda *= 10.0;
      if (!std::isfinite(lambda) || lambda > 1.0e16)
        solver_failed = true;
    }
  }

  fit.d_per_ps = {std::exp(params[0]), std::exp(params[1]),
                  std::exp(params[2])};
  const Eigen::Vector3d rotation = params.tail<3>();
  Eigen::Matrix3d axes = Eigen::Matrix3d::Identity();
  if (rotation.norm() > kEpsilon)
    axes = Eigen::AngleAxisd(rotation.norm(), rotation.normalized())
               .toRotationMatrix();
  std::array<std::size_t, 3> order{0, 1, 2};
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return fit.d_per_ps[lhs] < fit.d_per_ps[rhs];
  });
  std::array<double, 3> sorted_d{};
  Eigen::Matrix3d sorted_axes;
  for (std::size_t row = 0; row < 3; ++row) {
    sorted_d[row] = fit.d_per_ps[order[row]];
    sorted_axes.row(static_cast<Eigen::Index>(row)) =
        axes.row(static_cast<Eigen::Index>(order[row]));
  }
  fit.d_per_ps = sorted_d;
  sorted_axes = canonicalize_axes(sorted_axes);
  fit.axes = axes_values(sorted_axes);
  fit.residual = best;
  fit.ok = converged && finite_positive(fit.d_per_ps) && std::isfinite(best);
  fit.status = fit.ok ? "ok"
                      : (solver_failed ? "fit_failed" : "fit_not_converged");
  for (LagResult &row : results)
    row.covariance_fit = favro_model(row.lag_time_ps, fit.d_per_ps, sorted_axes);
  return fit;
}

void Analyzer::configure(bool enabled, std::string fit_window,
                         std::string maxlag, std::string pbc,
                         std::string selection) {
  requested_ = enabled;
  fit_window_ = std::move(fit_window);
  maxlag_ = std::move(maxlag);
  pbc_ = std::move(pbc);
  selection_ = std::move(selection);
}

void Analyzer::init(
    const std::vector<titan_selection::SelectedAtom> &selected_atoms,
    std::span<const double> selected_masses, std::fstream &log) {
  if (!requested_)
    return;
  if (selected_atoms.size() < 3)
    throw std::runtime_error(
        "MSD rotational: selection requires at least 3 atoms.");
  selected_atoms_ = selected_atoms;
  selected_masses_.assign(selected_masses.begin(), selected_masses.end());
  if (selected_masses_.size() != selected_atoms_.size())
    selected_masses_.assign(selected_atoms_.size(), 0.0);
  initialized_ = true;
  reset_trajectory_state();
  if (log.is_open()) {
    TitanUI::log_kv(log, "MSD Rotational", "enabled");
    TitanUI::log_kv(log, "MSD Rotational Reference",
                    "first_analyzed_frame");
  }
}

bool Analyzer::valid_box(const box &frame_box) const noexcept {
  return BoxMatrix::valid_box(frame_box);
}

std::vector<Point3> Analyzer::reconstruct_body(const MOL &frame) const {
  if (!frame.coordinates_consistent())
    throw std::runtime_error("MSD rotational: inconsistent coordinate arrays.");
  std::vector<Point3> points;
  points.reserve(selected_atoms_.size());
  for (const auto &atom : selected_atoms_) {
    const auto index = static_cast<std::size_t>(atom.atom_index);
    if (index >= frame.pos.x.size())
      throw std::runtime_error(
          "MSD rotational: selected atom index exceeds frame size.");
    points.push_back(Point3{frame.pos.x[index], frame.pos.y[index],
                            frame.pos.z[index]});
  }
  if ((pbc_ == "unwrap" || pbc_ == "mic") && !valid_box(frame.Box))
    throw std::runtime_error("MSD rotational: -msd-pbc=" + pbc_ +
                             " requires valid periodic boxes for all frames.");
  return points;
}

void Analyzer::append_frame(const MOL &frame) {
  if (!enabled())
    return;
  std::vector<Point3> points = reconstruct_body(frame);
  const bool has_box = valid_box(frame.Box);
  const double time_ps = frame.get_time();
  if (!std::isfinite(time_ps) ||
      (!time_ps_.empty() && !(time_ps > time_ps_.back())))
    throw std::runtime_error(
        "MSD rotational: frame timestamps must be finite and strictly increasing.");
  box_valid_.push_back(has_box);
  time_ps_.push_back(time_ps);
  boxes_.push_back(frame.Box);
  bodies_.push_back(std::move(points));
}

void Analyzer::export_msd(const std::string &filename,
                          const std::vector<LagResult> &results) const {
  std::ofstream out(filename);
  out << "lag_index lag_time_ps rotational_msd_rad2 C1 C2 samples\n";
  out << std::fixed << std::setprecision(8);
  for (const LagResult &row : results)
    out << row.lag_index << ' ' << row.lag_time_ps << ' '
        << row.rotational_msd_rad2 << ' ' << row.c1 << ' ' << row.c2 << ' '
        << row.samples << '\n';
}

void Analyzer::export_quaternions(const std::string &filename) const {
  std::ofstream out(filename);
  out << "frame,time_ps,qw,qx,qy,qz,kabsch_rmsd_A\n";
  out << std::fixed << std::setprecision(8);
  for (std::size_t i = 0; i < time_ps_.size(); ++i)
    out << i << ',' << time_ps_[i] << ',' << qw_[i] << ',' << qx_[i] << ','
        << qy_[i] << ',' << qz_[i] << ',' << rmsd_a_[i] << '\n';
}

void Analyzer::export_covariance(const std::string &filename,
                                 const std::vector<LagResult> &results) const {
  std::ofstream out(filename);
  out << "lag_index,lag_time_ps,Qxx,Qyy,Qzz,Qxy,Qxz,Qyz,Qxx_fit,Qyy_fit,"
         "Qzz_fit,Qxy_fit,Qxz_fit,Qyz_fit,D1_inst_per_ps,D2_inst_per_ps,"
         "D3_inst_per_ps\n";
  out << std::fixed << std::setprecision(10);
  for (const LagResult &row : results) {
    out << row.lag_index << ',' << row.lag_time_ps;
    for (double value : row.covariance.values)
      out << ',' << value;
    for (double value : row.covariance_fit.values)
      out << ',' << value;
    for (double value : row.instantaneous_d_per_ps)
      out << ',' << value;
    out << '\n';
  }
}

void Analyzer::export_summary(const std::string &filename, const TensorFit &fit,
                              const std::string &effective_pbc,
                              const std::string &pbc_warning) const {
  std::ofstream out(filename);
  const double rmsd_mean =
      rmsd_a_.empty()
          ? 0.0
          : std::accumulate(rmsd_a_.begin(), rmsd_a_.end(), 0.0) /
                static_cast<double>(rmsd_a_.size());
  const double rmsd_max =
      rmsd_a_.empty() ? 0.0 : *std::max_element(rmsd_a_.begin(), rmsd_a_.end());
  out << std::fixed << std::setprecision(8);
  out << "frames: " << time_ps_.size() << '\n';
  out << "selected_atoms: " << selected_atoms_.size() << '\n';
  out << "selection: " << selection_ << '\n';
  out << "reference_source: first_analyzed_frame\n";
  out << "reference_note: Orientations are Kabsch fits to the first analyzed "
         "frame. Interpret the tensor as rigid-body rotational diffusion and "
         "inspect kabsch_rmsd_A diagnostics for flexible selections.\n";
  out << "finite_size_note: Reported rotational diffusion is the uncorrected "
         "trajectory estimate; periodic-box finite-size correction may be "
         "needed.\n";
  out << "pbc_policy_requested: " << pbc_ << '\n';
  out << "pbc_policy_effective: " << effective_pbc << '\n';
  if (!pbc_warning.empty())
    out << "pbc_warning: " << pbc_warning << '\n';
  out << "kabsch_rmsd_mean_A: " << rmsd_mean << '\n';
  out << "kabsch_rmsd_max_A: " << rmsd_max << '\n';
  out << "fit_status: " << fit.status << '\n';
  out << "fit_window_policy: " << fit.window_policy << '\n';
  out << "fit_window_ps: " << fit.tmin_ps << ',' << fit.tmax_ps << '\n';
  out << "fit_points: " << fit.points << '\n';
  out << "D1_rad2_per_ps: " << fit.d_per_ps[0] << '\n';
  out << "D2_rad2_per_ps: " << fit.d_per_ps[1] << '\n';
  out << "D3_rad2_per_ps: " << fit.d_per_ps[2] << '\n';
  out << "D_iso_rad2_per_ps: "
      << (fit.d_per_ps[0] + fit.d_per_ps[1] + fit.d_per_ps[2]) / 3.0 << '\n';
  out << "fit_residual: " << fit.residual << '\n';
  for (int axis = 0; axis < 3; ++axis)
    out << "principal_axis_" << axis + 1 << ": "
        << fit.axes[static_cast<std::size_t>(axis * 3)] << ','
        << fit.axes[static_cast<std::size_t>(axis * 3 + 1)] << ','
        << fit.axes[static_cast<std::size_t>(axis * 3 + 2)] << '\n';
}

void Analyzer::finalize(const std::string &msd_filename,
                        const std::string &quaternion_filename,
                        const std::string &covariance_filename,
                        const std::string &summary_filename,
                        TitanExecutor &executor, std::fstream &log) {
  if (!enabled())
    return;
  const bool has_frames = !bodies_.empty();
  const bool all_boxes_valid =
      has_frames &&
      std::all_of(box_valid_.begin(), box_valid_.end(), [](bool valid) {
        return valid;
      });
  const bool unwrap_body =
      pbc_ == "unwrap" || (pbc_ == "auto" && all_boxes_valid);
  const bool reconstruct_with_mic = pbc_ == "mic";
  std::vector<std::vector<Point3>> orientation_bodies = bodies_;
  if (unwrap_body) {
    orientation_bodies = unwrap_bodies(bodies_, boxes_);
  } else if (reconstruct_with_mic) {
    for (std::size_t frame = 0; frame < orientation_bodies.size(); ++frame)
      orientation_bodies[frame] =
          reconstruct_points(orientation_bodies[frame], &boxes_[frame]);
  }
  if (!bodies_.empty()) {
    reference_ = orientation_bodies.front();
    for (std::size_t frame = 0; frame < bodies_.size(); ++frame) {
      const OrientationFit frame_fit = fit_orientation(
          reference_, orientation_bodies[frame], selected_masses_);
      if (!frame_fit.valid)
        throw std::runtime_error(
            "MSD rotational: selection cannot define a stable body frame.");
      Quaternion quaternion = frame_fit.orientation;
      if (!qw_.empty())
        quaternion = canonicalize_hemisphere(
            quaternion,
            Quaternion{qw_.back(), qx_.back(), qy_.back(), qz_.back()});
      qw_.push_back(quaternion.w);
      qx_.push_back(quaternion.x);
      qy_.push_back(quaternion.y);
      qz_.push_back(quaternion.z);
      rmsd_a_.push_back(frame_fit.rmsd_a);
    }
  }
  const std::size_t maxlag =
      qw_.size() > 1
          ? std::min<std::size_t>(
                qw_.size() - 1,
                maxlag_ == "auto" ? 1000
                                  : static_cast<std::size_t>(std::stoul(maxlag_)))
          : 0;
  std::vector<LagResult> results =
      compute_lag_results_soa(qw_, qx_, qy_, qz_, time_ps_, maxlag, executor);
  const TensorFit fit = fit_tensor(results, fit_window_);
  std::string effective_pbc = has_frames ? pbc_ : "not_evaluated";
  std::string warning;
  if (has_frames && pbc_ == "auto") {
    effective_pbc = all_boxes_valid ? "unwrap" : "raw";
    if (!all_boxes_valid)
      warning =
          "auto selected raw body coordinates because at least one frame has no valid box";
  } else if (has_frames && pbc_ == "mic") {
    effective_pbc = "mic_body_reconstruction";
    warning = "mic is a short-lag body reconstruction policy";
  }
  export_msd(msd_filename, results);
  export_quaternions(quaternion_filename);
  export_covariance(covariance_filename, results);
  export_summary(summary_filename, fit, effective_pbc, warning);
  if (log.is_open()) {
    TitanUI::log_kv(log, "Rotational MSD", msd_filename);
    TitanUI::log_kv(log, "Rotational Diffusion Summary", summary_filename);
  }
  reset_trajectory_state();
}

void Analyzer::reset_trajectory_state() {
  reference_.clear();
  bodies_.clear();
  boxes_.clear();
  time_ps_.clear();
  box_valid_.clear();
  qw_.clear();
  qx_.clear();
  qy_.clear();
  qz_.clear();
  rmsd_a_.clear();
}

} // namespace titan_rotational_diffusion

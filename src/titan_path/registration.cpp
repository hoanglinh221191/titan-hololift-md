#include "titan_path/registration.h"

#include <Eigen/Dense>

#include <cmath>
#include <limits>

namespace titan_path {
namespace {

bool finite(const Vec3 &point) {
  return std::isfinite(point[0]) && std::isfinite(point[1]) &&
         std::isfinite(point[2]);
}

Eigen::Vector3d eigen(const Vec3 &point) {
  return {point[0], point[1], point[2]};
}

Vec3 array(const Eigen::Vector3d &point) {
  return {point.x(), point.y(), point.z()};
}

} // namespace

RigidTransform
fit_rigid_transform(const std::vector<Vec3> &reference_angstrom,
                    const std::vector<Vec3> &current_angstrom) {
  RigidTransform result;
  if (reference_angstrom.size() != current_angstrom.size()) {
    result.error = "reference/current anchor counts differ";
    return result;
  }
  if (reference_angstrom.size() < 3) {
    result.error = "rigid registration requires at least three anchors";
    return result;
  }
  for (std::size_t i = 0; i < reference_angstrom.size(); ++i) {
    if (!finite(reference_angstrom[i]) || !finite(current_angstrom[i])) {
      result.error = "anchor coordinates must be finite";
      return result;
    }
  }

  Eigen::Vector3d reference_centroid = Eigen::Vector3d::Zero();
  Eigen::Vector3d current_centroid = Eigen::Vector3d::Zero();
  for (std::size_t i = 0; i < reference_angstrom.size(); ++i) {
    reference_centroid += eigen(reference_angstrom[i]);
    current_centroid += eigen(current_angstrom[i]);
  }
  const double count = static_cast<double>(reference_angstrom.size());
  reference_centroid /= count;
  current_centroid /= count;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  double reference_spread = 0.0;
  for (std::size_t i = 0; i < reference_angstrom.size(); ++i) {
    const Eigen::Vector3d reference =
        eigen(reference_angstrom[i]) - reference_centroid;
    const Eigen::Vector3d current =
        eigen(current_angstrom[i]) - current_centroid;
    covariance += reference * current.transpose();
    reference_spread += reference.squaredNorm();
  }
  if (reference_spread <= std::numeric_limits<double>::epsilon()) {
    result.error = "reference anchors have zero spatial extent";
    return result;
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d provisional = svd.matrixV() * svd.matrixU().transpose();
  if (provisional.determinant() < 0.0)
    correction(2, 2) = -1.0;
  const Eigen::Matrix3d rotation =
      svd.matrixV() * correction * svd.matrixU().transpose();
  const Eigen::Vector3d translation =
      current_centroid - rotation * reference_centroid;

  double squared_error = 0.0;
  for (std::size_t i = 0; i < reference_angstrom.size(); ++i) {
    const Eigen::Vector3d residual =
        rotation * eigen(reference_angstrom[i]) + translation -
        eigen(current_angstrom[i]);
    squared_error += residual.squaredNorm();
  }

  result.valid = true;
  result.determinant = rotation.determinant();
  result.translation_angstrom = array(translation);
  result.rmsd_angstrom = std::sqrt(squared_error / count);
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t column = 0; column < 3; ++column)
      result.rotation[row * 3 + column] = rotation(row, column);
  return result;
}

Vec3 apply_transform(const RigidTransform &transform,
                     const Vec3 &point_angstrom) {
  Vec3 output{};
  for (std::size_t row = 0; row < 3; ++row) {
    output[row] = transform.translation_angstrom[row];
    for (std::size_t column = 0; column < 3; ++column)
      output[row] +=
          transform.rotation[row * 3 + column] * point_angstrom[column];
  }
  return output;
}

RouteDefinition apply_transform(const RouteDefinition &route,
                                const RigidTransform &transform) {
  RouteDefinition output = route;
  for (auto &anchor : output.anchors)
    anchor.reference_position_angstrom =
        apply_transform(transform, anchor.reference_position_angstrom);
  for (auto &waypoint : output.waypoints)
    waypoint.position_angstrom =
        apply_transform(transform, waypoint.position_angstrom);
  return output;
}

} // namespace titan_path

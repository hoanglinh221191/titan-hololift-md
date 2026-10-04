#ifndef TITAN_PATH_REGISTRATION_H
#define TITAN_PATH_REGISTRATION_H

#include "titan_path/route.h"

#include <array>
#include <string>
#include <vector>

namespace titan_path {

struct RigidTransform {
  bool valid{false};
  std::array<double, 9> rotation{1.0, 0.0, 0.0, 0.0, 1.0,
                                 0.0, 0.0, 0.0, 1.0};
  Vec3 translation_angstrom{};
  double rmsd_angstrom{0.0};
  double determinant{1.0};
  std::string error;
};

[[nodiscard]] RigidTransform
fit_rigid_transform(const std::vector<Vec3> &reference_angstrom,
                    const std::vector<Vec3> &current_angstrom);

[[nodiscard]] Vec3 apply_transform(const RigidTransform &transform,
                                   const Vec3 &point_angstrom);

[[nodiscard]] RouteDefinition
apply_transform(const RouteDefinition &route,
                const RigidTransform &transform);

} // namespace titan_path

#endif

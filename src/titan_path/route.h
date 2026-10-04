#ifndef TITAN_PATH_ROUTE_H
#define TITAN_PATH_ROUTE_H

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace titan_path {

using Vec3 = std::array<double, 3>;

enum class RegistrationMode { Static, Rigid };

struct RouteAnchor {
  std::size_t original_atom_id{0};
  std::string chain;
  std::size_t residue_position{0};
  std::string atom_name;
  Vec3 reference_position_angstrom{};
};

struct RouteWaypoint {
  std::size_t index{0};
  Vec3 position_angstrom{};
  double radius_angstrom{0.0};
};

struct RouteDefinition {
  int schema_version{1};
  std::string route_id;
  std::string branch_id;
  std::string source;
  std::string reference_structure_sha256;
  RegistrationMode registration{RegistrationMode::Rigid};
  std::size_t start_index{0};
  std::size_t terminal_index{0};
  std::vector<RouteAnchor> anchors;
  std::vector<RouteWaypoint> waypoints;
  double total_length_angstrom{0.0};
};

struct RouteLoadResult {
  bool ok{false};
  RouteDefinition route;
  std::string source_file_sha256;
  std::string error;
};

[[nodiscard]] RouteLoadResult
load_route_file(const std::filesystem::path &path);

[[nodiscard]] std::string
serialize_route_canonical(const RouteDefinition &route);

[[nodiscard]] const char *registration_mode_name(RegistrationMode mode);

} // namespace titan_path

#endif

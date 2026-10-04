#ifndef TITAN_SURFACE_H
#define TITAN_SURFACE_H

#include <cstddef>
#include <span>
#include <string>
#include <vector>

class TitanExecutor;

namespace TitanSurface {

inline constexpr int kDefaultPointCount = 96;
inline constexpr double kDefaultProbeRadiusA = 1.4;

struct Atom {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double radius{0.0};
};

struct Panel {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double nx{0.0};
  double ny{0.0};
  double nz{0.0};
  double area{0.0};
  std::size_t atom_index{0};
};

struct Options {
  double probe_radius{kDefaultProbeRadiusA};
  double min_cell_size{1.0};
  int point_count{kDefaultPointCount};
  std::string surface_mode{"sas"};
  std::string acceleration{"auto"};
  bool emit_panels{false};
  bool emit_neighbor_workspace{false};
};

struct Diagnostics {
  int point_count{kDefaultPointCount};
  std::size_t target_count{0};
  std::size_t exposed_points{0};
  std::size_t buried_points{0};
  std::string acceleration_effective{"legacy"};
  std::size_t cell_count{0};
  std::size_t reordered_atoms{0};
  double neighbor_build_ms{0.0};
  double exposure_ms{0.0};
  std::size_t max_neighbor_count{0};
  double mean_neighbor_count{0.0};
  std::size_t padded_neighbor_slots{0};
  double build_ms{0.0};
};

struct Workspace {
  std::vector<double> area_by_atom;
  std::vector<Panel> panels;
  std::vector<std::size_t> neighbor_offsets;
  std::vector<std::size_t> neighbors;
  Diagnostics diagnostics;
  double total_area{0.0};
};

Workspace build_workspace(std::span<const Atom> atoms, const Options &options,
                          std::span<const std::size_t> targets = {},
                          TitanExecutor *executor = nullptr);

} // namespace TitanSurface

#endif // TITAN_SURFACE_H

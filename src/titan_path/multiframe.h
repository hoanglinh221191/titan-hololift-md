#ifndef TITAN_PATH_MULTIFRAME_H
#define TITAN_PATH_MULTIFRAME_H

#include "titan_path/control.h"
#include "titan_path/geometry.h"
#include <optional>
#include <string>
#include <vector>

namespace titan_path {
struct MultiFrameInput {
  std::vector<Vec3> reference_core, current_core, waypoints;
  std::vector<Vec3> receptor, ligand;
  std::vector<double> ligand_masses;
  Vec3 old_direction{}, displacement{};
  std::optional<RouteMode> previous_mode;
  std::optional<std::array<double, 3>> observed_progress;
  RouteMode native_mode{RouteMode::Advance};
};
struct MultiFrameMetrics {
  std::string name, reason;
  bool passed{false};
  bool start_available{false}, end_available{false};
  RouteMode mode{RouteMode::Advance};
  double rmsd{0}, shape_ratio{0}, start_progress{0}, start_distance{0};
  double end_progress{0}, end_distance{0};
};
struct MultiFrameResult {
  bool accepted{false};
  std::string reason;
  double clearance{0}, turn_deg{0};
  std::vector<MultiFrameMetrics> frames;
};
// Pure veto on one supplied vector; not a search, status writer or MD executor.
[[nodiscard]] MultiFrameResult
validate_multiframe_direction(const MultiFrameInput &input);
} // namespace titan_path
#endif

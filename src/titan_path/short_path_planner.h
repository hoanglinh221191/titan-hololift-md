#ifndef TITAN_PATH_SHORT_PATH_PLANNER_H
#define TITAN_PATH_SHORT_PATH_PLANNER_H
#include "titan_path/control.h"
#include <span>
#include <vector>

namespace titan_path {
// Ranking backend for a bounded layered path search. Geometry/energy evaluation
// is supplied separately; no runtime, MD or scheduler authority lives here.
struct ShortPathRanking {
  std::vector<std::size_t> indices;
  double minimum_energy_key;
  double minimum_navigation_cost;
  bool md_authorized{false};
};

// Bands are anchored ONCE to the complete frontier, not moved after each pick.
[[nodiscard]] ShortPathRanking rank_short_path_frontier(
    std::span<const SelectionCandidate> candidates, std::size_t width,
    bool terminal, double energy_epsilon=0.05, double navigation_epsilon=0.02);
}
#endif

#ifndef TITAN_PBCTOPO_SPATIAL_LIFT_H
#define TITAN_PBCTOPO_SPATIAL_LIFT_H

#include "types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace titan_pbctopo {

inline constexpr std::uint32_t PBCTOPO_SPATIAL_LIFT_POLICY_VERSION = 1;

struct PbctopoSpatialLiftAtom {
  int owner = 0;
  std::uint64_t source_atom_id = 0;
  std::array<double, 3> wrapped_cartesian{};
};

// Endpoints index the canonical atom order sorted by (owner, source_atom_id).
struct PbctopoSpatialLiftEdge {
  std::uint32_t atom_a = 0;
  std::uint32_t atom_b = 0;

  bool operator==(const PbctopoSpatialLiftEdge &) const = default;
};

struct PbctopoSpatialLiftResult {
  // Images are returned in the caller's atom order.
  std::vector<Int3> atom_images;
  PbctopoHash128 identity;
  std::size_t ambiguous_hard_edges = 0;
  std::size_t cycle_residuals = 0;
  bool valid = false;
  std::string error;
};

PbctopoSpatialLiftResult build_pbctopo_spatial_lift(
    std::span<const double, 9> box_matrix,
    std::span<const PbctopoSpatialLiftAtom> atoms,
    std::span<const PbctopoSpatialLiftEdge> hard_edges);

} // namespace titan_pbctopo

#endif

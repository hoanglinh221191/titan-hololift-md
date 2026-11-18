#ifndef TITAN_PBCTOPO_PERIODIC_CONTACT_AUDIT_H
#define TITAN_PBCTOPO_PERIODIC_CONTACT_AUDIT_H

// Periodic contact network of one frame, read from the wrapped input.
//
// Units are the parts whose relative lattice images VIBE assigns (components
// in the global solver, molecular units in LLPS-local mode).  Each unit is
// whole in the lifted gauge y_i = x_i + B n_i (n_i: lift image of atom i).
// A heavy-atom pair (i in unit u, j in unit v) closer than the contact cutoff
// under the minimum-image convention, x_j - x_i + B m, is the relation
//
//     o_v - o_u = s,   s = m - (n_j - n_i),
//
// on the unit offsets o.  For u == v and s != 0 the unit touches its own
// image.  A finite placement that keeps every relation exists exactly when
// the sum of s around every cycle of this graph is zero (self-relations are
// cycles of length one).  The cycle sums span a sublattice of Z^3; its rank
// is 1 for a filament, 2 for a sheet and 3 for a network that percolates in
// every direction.
//
// The same scan reports atom pairs between different units that are closer
// than the steric threshold.  Under periodic boundary conditions such an
// overlap is in the simulated system for every placement, so it is a
// property of the input, not of a candidate placement.

#include "lattice_math.h"
#include "lattice_span.h"
#include "types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class box;

namespace titan_pbctopo {

inline constexpr std::uint32_t PBCTOPO_PERIODIC_CONTACT_POLICY_VERSION = 1;

struct PbctopoUnitRelation {
  std::size_t unit_a = 0; // unit_a <= unit_b
  std::size_t unit_b = 0;
  Int3 shift{};           // o_b - o_a; canonical direction when unit_a == unit_b
  std::size_t atom_pairs = 0;
  double min_d2 = std::numeric_limits<double>::infinity();
};

struct PbctopoInputOverlapAudit {
  bool evaluated = false;
  bool complete = false;
  double vdw_scale = 0.0;
  double fallback_cutoff_A = 0.0;
  std::size_t overlap_pairs = 0;
  std::size_t overlap_unit_pairs = 0;
  // Over inter-unit atom pairs found within the contact cutoff.
  double min_distance_A = std::numeric_limits<double>::quiet_NaN();
  double min_ratio = std::numeric_limits<double>::quiet_NaN();
};

struct PbctopoContactCycleLattice {
  bool evaluated = false;
  PbctopoLatticeSpan span;
  std::size_t units = 0;
  std::size_t graph_components = 0;
  std::size_t periodic_graph_components = 0;
  std::size_t max_component_rank = 0;
  std::size_t nonzero_cycle_relations = 0;
  std::size_t self_image_units = 0;
  std::size_t self_image_atom_pairs = 0;
  double self_image_min_distance_A = std::numeric_limits<double>::quiet_NaN();
};

struct PbctopoPeriodicContactInput {
  std::span<const PbctopoAtom> atoms;
  std::span<const std::size_t> atom_unit;   // unit index per atom
  std::span<const Int3> lift_images;        // lift image per atom
  std::size_t unit_count = 0;
  double contact_cutoff_A = 4.5;
  double overlap_vdw_scale = 0.55;
  double overlap_fallback_cutoff_A = 1.05;
  // Pairs exempt from the overlap report (bonded across units), keyed by
  // (min(i, j) << 32) | max(i, j) over input atom indices.
  const std::unordered_set<std::uint64_t> *overlap_excluded_pairs = nullptr;
  // Extra relations on atom pairs regardless of distance (required contacts).
  std::span<const std::pair<std::size_t, std::size_t>> required_atom_pairs;
};

struct PbctopoPeriodicContactAudit {
  bool evaluated = false;
  bool complete = false;
  std::string error;
  std::vector<PbctopoUnitRelation> relations; // sorted (unit_a, unit_b, shift)
  PbctopoInputOverlapAudit overlap;
  PbctopoContactCycleLattice cycle;
};

PbctopoPeriodicContactAudit
audit_pbctopo_periodic_contacts(const box &Box,
                                const PbctopoPeriodicContactInput &input);

PbctopoContactCycleLattice compute_pbctopo_contact_cycle_lattice(
    std::size_t unit_count, std::span<const PbctopoUnitRelation> relations);

// Images n_i that make every unit whole, from unwrapped points p_i and the
// wrapped atoms: n_i = round(S (p_i - x_i)), with S the inverse cell matrix.
bool pbctopo_lift_images_from_points(const box &Box,
                                     std::span<const PbctopoAtom> atoms,
                                     std::span<const Vec3> points,
                                     std::vector<Int3> &images);

const char *pbctopo_steric_policy_name(bool steric_filter);

} // namespace titan_pbctopo

#endif

#include "certification.h"
#include "required_contacts.h"
#include "steric_clash.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

namespace titan_pbctopo {

namespace {

struct PbctopoStericCellKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const PbctopoStericCellKey &) const = default;
};

struct PbctopoStericCellKeyHash {
  std::size_t operator()(const PbctopoStericCellKey &key) const noexcept {
    std::uint64_t value = static_cast<std::uint64_t>(key.x);
    value ^= static_cast<std::uint64_t>(key.y) +
             0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    value ^= static_cast<std::uint64_t>(key.z) +
             0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    return static_cast<std::size_t>(value);
  }
};

std::uint64_t steric_pair_key(std::size_t a, std::size_t b) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return (static_cast<std::uint64_t>(lo) << 32) |
         static_cast<std::uint64_t>(hi);
}

bool point_is_finite(const Vec3 &point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

Int3 canonical_winding_vector(Int3 value) {
  const auto magnitude = [](std::int64_t component) noexcept {
    return component < 0
               ? static_cast<std::uint64_t>(-(component + 1)) + 1u
               : static_cast<std::uint64_t>(component);
  };
  const std::uint64_t divisor =
      std::gcd(std::gcd(magnitude(value.x), magnitude(value.y)),
               magnitude(value.z));
  if (divisor > 1) {
    const auto divide = [divisor](std::int64_t component) noexcept {
      if (divisor >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
        return component == std::numeric_limits<std::int64_t>::min()
                   ? std::int64_t{-1}
                   : std::int64_t{0};
      }
      return component / static_cast<std::int64_t>(divisor);
    };
    value.x = divide(value.x);
    value.y = divide(value.y);
    value.z = divide(value.z);
  }
  const std::int64_t first_nonzero =
      value.x != 0 ? value.x : (value.y != 0 ? value.y : value.z);
  if (first_nonzero < 0) {
    Int3 negated{};
    if (checked_int3_negate(value, negated))
      value = negated;
  }
  return value;
}

long double winding_determinant(const Int3 &a, const Int3 &b,
                                const Int3 &c) {
  return static_cast<long double>(a.x) *
             (static_cast<long double>(b.y) * c.z -
              static_cast<long double>(b.z) * c.y) -
         static_cast<long double>(a.y) *
             (static_cast<long double>(b.x) * c.z -
              static_cast<long double>(b.z) * c.x) +
         static_cast<long double>(a.z) *
             (static_cast<long double>(b.x) * c.y -
              static_cast<long double>(b.y) * c.x);
}

bool winding_vectors_collinear(const Int3 &a, const Int3 &b) {
  return static_cast<long double>(a.y) * b.z ==
             static_cast<long double>(a.z) * b.y &&
         static_cast<long double>(a.z) * b.x ==
             static_cast<long double>(a.x) * b.z &&
         static_cast<long double>(a.x) * b.y ==
             static_cast<long double>(a.y) * b.x;
}

void append_winding_basis(PbctopoImageEquationResiduals &result,
                          Int3 residual) {
  residual = canonical_winding_vector(residual);
  if (is_zero_delta(residual) || result.topological_winding_rank >= 3)
    return;
  if (result.topological_winding_rank == 1 &&
      winding_vectors_collinear(result.topological_winding_basis[0],
                                residual)) {
    return;
  }
  if (result.topological_winding_rank == 2 &&
      winding_determinant(result.topological_winding_basis[0],
                          result.topological_winding_basis[1], residual) ==
          0.0L) {
    return;
  }
  result.topological_winding_basis[result.topological_winding_rank++] =
      residual;
}

} // namespace

PbctopoImageEquationResiduals evaluate_pbctopo_image_equations(
    std::span<const PbctopoAtom> wrapped_atoms,
    std::span<const Vec3> placed_points,
    std::span<const PbctopoLocalEdge> local_edges,
    const PbctopoLatticeMetric &lattice, double image_tolerance) {
  PbctopoImageEquationResiduals result;
  if (wrapped_atoms.empty() || placed_points.size() != wrapped_atoms.size() ||
      !std::isfinite(image_tolerance) || image_tolerance <= 0.0) {
    result.atom_image_residuals = 1;
    return result;
  }

  std::vector<Int3> atom_images(wrapped_atoms.size());
  std::vector<unsigned char> atom_image_valid(wrapped_atoms.size(), 0);
  for (std::size_t atom = 0; atom < wrapped_atoms.size(); ++atom) {
    const auto &wrapped = wrapped_atoms[atom];
    const auto &placed = placed_points[atom];
    if (!std::isfinite(placed.x) || !std::isfinite(placed.y) ||
        !std::isfinite(placed.z)) {
      ++result.atom_image_residuals;
      continue;
    }
    const auto scaled = lattice.cartesian_to_scaled(
        placed.x - wrapped.x, placed.y - wrapped.y, placed.z - wrapped.z);
    if (!std::isfinite(scaled[0]) || !std::isfinite(scaled[1]) ||
        !std::isfinite(scaled[2])) {
      ++result.atom_image_residuals;
      continue;
    }
    Int3 image{};
    if (!pbctopo_checked_round_image(scaled[0], image.x) ||
        !pbctopo_checked_round_image(scaled[1], image.y) ||
        !pbctopo_checked_round_image(scaled[2], image.z)) {
      ++result.atom_image_residuals;
      continue;
    }
    const double error = std::max(
        {std::fabs(scaled[0] - static_cast<double>(image.x)),
         std::fabs(scaled[1] - static_cast<double>(image.y)),
         std::fabs(scaled[2] - static_cast<double>(image.z))});
    result.max_atom_image_error = std::max(result.max_atom_image_error, error);
    if (error > image_tolerance) {
      ++result.atom_image_residuals;
      continue;
    }
    atom_images[atom] = image;
    atom_image_valid[atom] = 1;
  }

  struct ExpectedHardEdge {
    std::size_t a = 0;
    std::size_t b = 0;
    Int3 delta{};
  };
  std::vector<ExpectedHardEdge> expected_hard_edges;
  expected_hard_edges.reserve(local_edges.size());
  for (const auto &edge : local_edges) {
    const bool topology = edge.kind == PbctopoLocalEdgeKind::TopologyBond;
    const bool metadata = edge.kind == PbctopoLocalEdgeKind::MetadataSkeleton;
    if (!topology && !metadata)
      continue;
    if (topology)
      ++result.topology_edges_checked;
    else
      ++result.metadata_edges_checked;
    auto add_image_residual = [&]() {
      if (topology)
        ++result.topology_edge_residuals;
      else
        ++result.metadata_edge_image_residuals;
    };
    if (edge.a >= wrapped_atoms.size() || edge.b >= wrapped_atoms.size() ||
        edge.a == edge.b || atom_image_valid[edge.a] == 0 ||
        atom_image_valid[edge.b] == 0) {
      add_image_residual();
      continue;
    }
    const auto &a = wrapped_atoms[edge.a];
    const auto &b = wrapped_atoms[edge.b];
    const auto nearest = lattice.nearest_image_scaled(
        b.sx - a.sx, b.sy - a.sy, b.sz - a.sz);
    if (!nearest.valid) {
      add_image_residual();
      continue;
    }
    const Int3 expected{nearest.image_x, nearest.image_y, nearest.image_z};
    if (nearest.ambiguous) {
      if (topology)
        ++result.ambiguous_topology_edges;
      else
        ++result.ambiguous_metadata_edges;
    }
    Int3 actual{};
    if (!checked_int3_subtract(atom_images[edge.b], atom_images[edge.a],
                               actual)) {
      add_image_residual();
      continue;
    }
    const bool physically_equivalent =
        lattice.image_is_nearest_equivalent_scaled(
            b.sx - a.sx, b.sy - a.sy, b.sz - a.sz, actual.x, actual.y,
            actual.z, nearest);
    expected_hard_edges.push_back(
        {edge.a, edge.b, physically_equivalent ? actual : expected});
    if (!physically_equivalent)
      add_image_residual();
    if (metadata) {
      const Vec3 delta = placed_points[edge.b] - placed_points[edge.a];
      const double placed_d2 =
          delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
      const double distance_error =
          std::fabs(std::sqrt(std::max(0.0, placed_d2)) -
                    std::sqrt(std::max(0.0, nearest.d2)));
      result.max_metadata_edge_distance_error_A =
          std::max(result.max_metadata_edge_distance_error_A, distance_error);
      const double distance_tolerance =
          std::max(1.0e-5, 1.0e-6 * std::sqrt(std::max(0.0, nearest.d2)));
      if (!std::isfinite(placed_d2) || !std::isfinite(distance_error) ||
          distance_error > distance_tolerance) {
        ++result.metadata_edge_distance_residuals;
      }
    }
  }

  struct TopologyNeighbor {
    std::size_t atom = 0;
    Int3 delta{};
  };
  std::vector<std::vector<TopologyNeighbor>> adjacency(wrapped_atoms.size());
  for (const auto &edge : expected_hard_edges) {
    Int3 reverse_delta{};
    if (!checked_int3_negate(edge.delta, reverse_delta)) {
      ++result.topology_cycle_residuals;
      continue;
    }
    adjacency[edge.a].push_back({edge.b, edge.delta});
    adjacency[edge.b].push_back({edge.a, reverse_delta});
  }
  std::vector<Int3> topology_potential(wrapped_atoms.size());
  std::vector<unsigned char> topology_visited(wrapped_atoms.size(), 0);
  std::vector<unsigned char> topology_potential_valid(wrapped_atoms.size(), 0);
  for (std::size_t root = 0; root < wrapped_atoms.size(); ++root) {
    if (topology_visited[root] != 0 || adjacency[root].empty())
      continue;
    std::queue<std::size_t> pending;
    pending.push(root);
    topology_visited[root] = 1;
    topology_potential_valid[root] = 1;
    while (!pending.empty()) {
      const std::size_t atom = pending.front();
      pending.pop();
      for (const auto &neighbor : adjacency[atom]) {
        if (topology_visited[neighbor.atom] != 0)
          continue;
        topology_visited[neighbor.atom] = 1;
        if (topology_potential_valid[atom] != 0 &&
            checked_int3_add(topology_potential[atom], neighbor.delta,
                             topology_potential[neighbor.atom])) {
          topology_potential_valid[neighbor.atom] = 1;
        } else {
          ++result.topology_cycle_residuals;
        }
        pending.push(neighbor.atom);
      }
    }
  }
  for (const auto &edge : expected_hard_edges) {
    if (topology_potential_valid[edge.a] == 0 ||
        topology_potential_valid[edge.b] == 0) {
      ++result.topology_cycle_residuals;
      continue;
    }
    Int3 actual{};
    if (!checked_int3_subtract(topology_potential[edge.b],
                               topology_potential[edge.a], actual)) {
      ++result.topology_cycle_residuals;
      continue;
    }
    if (!same_delta(actual, edge.delta)) {
      ++result.topology_cycle_residuals;
      Int3 residual{};
      if (checked_int3_subtract(actual, edge.delta, residual))
        append_winding_basis(result, residual);
    }
  }
  result.input_valid = true;
  return result;
}

PbctopoRequiredContactResiduals evaluate_pbctopo_required_contacts(
    std::span<const PbctopoAtom> atoms,
    std::span<const Vec3> placed_points,
    std::span<const PbctopoLocalEdge> local_edges,
    std::span<const PbctopoRequiredContact> explicit_contacts) {
  PbctopoRequiredContactResiduals result;
  result.explicit_constraints_requested = !explicit_contacts.empty();
  if (atoms.empty() || placed_points.size() != atoms.size())
    return result;

  for (const auto &edge : local_edges) {
    if (edge.kind != PbctopoLocalEdgeKind::TopologyBond)
      continue;
    if (edge.a >= atoms.size() || edge.b >= atoms.size() || edge.a == edge.b)
      continue;
    if (atoms[edge.a].owner == atoms[edge.b].owner)
      continue;

    ++result.contacts_checked;
    ++result.topology_contacts_checked;
    const Vec3 delta = placed_points[edge.a] - placed_points[edge.b];
    const double d2 = delta.x * delta.x + delta.y * delta.y +
                      delta.z * delta.z;
    if (!pbctopo_required_contact_satisfied(d2, edge.break_cutoff_d2)) {
      ++result.contacts_lost;
      ++result.topology_contacts_lost;
    }
  }

  for (const auto &contact : explicit_contacts) {
    if (contact.a >= atoms.size() || contact.b >= atoms.size() ||
        contact.a == contact.b || !std::isfinite(contact.max_distance_A) ||
        contact.max_distance_A <= 0.0 || !std::isfinite(contact.weight) ||
        contact.weight <= 0.0) {
      return result;
    }
    ++result.contacts_checked;
    ++result.explicit_contacts_checked;
    result.explicit_weight_checked += contact.weight;
    const Vec3 delta = placed_points[contact.a] - placed_points[contact.b];
    const double d2 = delta.x * delta.x + delta.y * delta.y +
                      delta.z * delta.z;
    const double max_d2 = contact.max_distance_A * contact.max_distance_A;
    if (!pbctopo_required_contact_satisfied(d2, max_d2)) {
      ++result.contacts_lost;
      ++result.explicit_contacts_lost;
      result.explicit_weight_lost += contact.weight;
    }
  }
  result.input_valid = true;
  return result;
}

PbctopoAssignmentPointResiduals
evaluate_pbctopo_assignment_point_consistency(
    std::span<const PbctopoAtom> component_reference_atoms,
    std::span<const Vec3> placed_points,
    std::span<const std::size_t> atom_component_index,
    const PbctopoImageAssignment &assignment,
    const PbctopoLatticeMetric &lattice, double image_tolerance) {
  PbctopoAssignmentPointResiduals result;
  if (component_reference_atoms.empty() ||
      placed_points.size() != component_reference_atoms.size() ||
      atom_component_index.size() != component_reference_atoms.size() ||
      !assignment.valid || assignment.component_offsets.empty() ||
      !std::isfinite(image_tolerance) || image_tolerance <= 0.0) {
    result.point_residuals = 1;
    return result;
  }

  for (std::size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
    const std::size_t component_idx = atom_component_index[atom_idx];
    if (component_idx >= assignment.component_offsets.size() ||
        !point_is_finite(placed_points[atom_idx])) {
      ++result.point_residuals;
      continue;
    }
    const auto &reference = component_reference_atoms[atom_idx];
    if (!std::isfinite(reference.x) || !std::isfinite(reference.y) ||
        !std::isfinite(reference.z)) {
      ++result.point_residuals;
      continue;
    }
    const auto scaled = lattice.cartesian_to_scaled(
        placed_points[atom_idx].x - reference.x,
        placed_points[atom_idx].y - reference.y,
        placed_points[atom_idx].z - reference.z);
    const auto &canonical = assignment.component_offsets[component_idx];
    Int3 absolute_offset{};
    if (!checked_int3_add(canonical, assignment.global_gauge,
                          absolute_offset)) {
      ++result.point_residuals;
      continue;
    }
    const double expected_x = static_cast<double>(absolute_offset.x);
    const double expected_y = static_cast<double>(absolute_offset.y);
    const double expected_z = static_cast<double>(absolute_offset.z);
    const double error = std::max(
        {std::fabs(scaled[0] - expected_x),
         std::fabs(scaled[1] - expected_y),
         std::fabs(scaled[2] - expected_z)});
    if (!std::isfinite(error)) {
      ++result.point_residuals;
      continue;
    }
    result.max_point_error = std::max(result.max_point_error, error);
    if (error > image_tolerance)
      ++result.point_residuals;
  }
  result.input_valid = true;
  return result;
}

namespace {

PbctopoStericAudit evaluate_pbctopo_sterics_impl(
    std::span<const PbctopoAtom> atoms,
    std::span<const Vec3> placed_points,
    std::span<const std::size_t> atom_component_index,
    const std::unordered_set<std::uint64_t> &excluded_pairs,
    double fallback_cutoff, PbctopoTimingStats *timing,
    const PbctopoLatticeMetric *wrapped_lattice,
    bool metadata_internal_only) {
  PbctopoStericAudit result;
  if (atoms.empty() || placed_points.size() != atoms.size() ||
      atom_component_index.size() != atoms.size() ||
      !std::isfinite(fallback_cutoff) || fallback_cutoff <= 0.0) {
    return result;
  }
  for (std::size_t component_idx : atom_component_index) {
    if (component_idx >= atoms.size())
      return result;
  }

  double max_radius = 0.0;
  for (const auto &atom : atoms) {
    switch (atom.steric_radius_source) {
    case PbctopoStericRadiusSource::ForceFieldTopology:
    case PbctopoStericRadiusSource::ForceFieldParameter:
      ++result.force_field_radius_atoms;
      break;
    case PbctopoStericRadiusSource::ExplicitZeroLennardJones:
      ++result.explicit_zero_lj_radius_atoms;
      break;
    case PbctopoStericRadiusSource::InferredElement:
      ++result.inferred_radius_atoms;
      break;
    case PbctopoStericRadiusSource::GenericFallback:
      ++result.generic_fallback_radius_atoms;
      break;
    case PbctopoStericRadiusSource::Unavailable:
    default:
      ++result.unavailable_radius_atoms;
      break;
    }
    if (atom.has_vdw_radius && std::isfinite(atom.steric_radius)) {
      max_radius = std::max(max_radius, atom.steric_radius);
    }
  }
  const double grid_cutoff = std::max(fallback_cutoff, 2.0 * max_radius);
  if (!std::isfinite(grid_cutoff) || grid_cutoff <= 0.0)
    return result;
  result.input_valid = true;
  result.termination_reason =
      PbctopoStericTerminationReason::NumericalFailure;
  const double inv_grid = 1.0 / grid_cutoff;
  const double cell_min =
      static_cast<double>(std::numeric_limits<std::int64_t>::min() + 1);
  const double cell_max =
      static_cast<double>(std::numeric_limits<std::int64_t>::max() - 1);
  auto atom_cell = [&](std::size_t atom_idx, PbctopoStericCellKey &cell) {
    const Vec3 &point = placed_points[atom_idx];
    if (!point_is_finite(point))
      return false;
    const double cell_x = std::floor(point.x * inv_grid);
    const double cell_y = std::floor(point.y * inv_grid);
    const double cell_z = std::floor(point.z * inv_grid);
    if (cell_x < cell_min || cell_x > cell_max || cell_y < cell_min ||
        cell_y > cell_max || cell_z < cell_min || cell_z > cell_max) {
      return false;
    }
    cell = {static_cast<std::int64_t>(cell_x),
            static_cast<std::int64_t>(cell_y),
            static_cast<std::int64_t>(cell_z)};
    return true;
  };

  // The scan visits, for each atom in index order, the earlier atoms of the
  // 27 cells around it: cells in (dx, dy, dz) order, atoms of a cell in
  // index order.  That order decides ties of the running minima and which
  // clash ends the scan, so it is kept exactly.  Instead of a hash grid
  // filled as the scan goes (27 lookups and a push per atom), the atoms are
  // sorted once by (cell, index): the three cells along z around a given
  // (x, y) are then one contiguous run, found by one binary search, already
  // in dz order and index order.  An atom whose point or cell is invalid
  // stops the scan when it is reached, so it and every later atom are left
  // out of the index; nothing before it could meet them.
  struct CellEntry {
    std::int64_t x;
    std::int64_t y;
    std::int64_t z;
    std::size_t atom;
  };
  const auto entry_less = [](const CellEntry &lhs, const CellEntry &rhs) {
    if (lhs.x != rhs.x)
      return lhs.x < rhs.x;
    if (lhs.y != rhs.y)
      return lhs.y < rhs.y;
    if (lhs.z != rhs.z)
      return lhs.z < rhs.z;
    return lhs.atom < rhs.atom;
  };
  thread_local std::vector<CellEntry> sorted_cells;
  sorted_cells.clear();
  sorted_cells.reserve(atoms.size());
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    PbctopoStericCellKey cell;
    if (!atom_cell(atom_idx, cell))
      break;
    sorted_cells.push_back({cell.x, cell.y, cell.z, atom_idx});
  }
  std::sort(sorted_cells.begin(), sorted_cells.end(), entry_less);

  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const Vec3 &point = placed_points[atom_idx];
    PbctopoStericCellKey cell;
    if (!atom_cell(atom_idx, cell))
      return result;
    const std::int64_t cx = cell.x;
    const std::int64_t cy = cell.y;
    const std::int64_t cz = cell.z;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        const CellEntry first{cx + dx, cy + dy, cz - 1, 0};
        for (auto it = std::lower_bound(sorted_cells.begin(),
                                        sorted_cells.end(), first, entry_less);
             it != sorted_cells.end() && it->x == cx + dx &&
             it->y == cy + dy && it->z <= cz + 1;
             ++it) {
          const std::size_t other_idx = it->atom;
          if (other_idx >= atom_idx)
            continue;
          {
            const bool same_component =
                atom_component_index[atom_idx] ==
                atom_component_index[other_idx];
            const bool metadata_internal_pair =
                same_component && wrapped_lattice != nullptr &&
                wrapped_lattice->valid() &&
                !atoms[atom_idx].has_topology_graph &&
                !atoms[other_idx].has_topology_graph &&
                atoms[atom_idx].has_topology_key &&
                atoms[other_idx].has_topology_key;
            if (metadata_internal_only && !metadata_internal_pair)
              continue;
            if (same_component && !metadata_internal_pair)
              continue;
            // The exclusion lookup comes after the cheap tests above: a pair
            // they skip is skipped whether or not it is excluded, and most
            // neighbouring pairs lie within one component.
            if (excluded_pairs.contains(
                    steric_pair_key(atom_idx, other_idx))) {
              continue;
            }
            const Vec3 &other = placed_points[other_idx];
            const double ddx = point.x - other.x;
            const double ddy = point.y - other.y;
            const double ddz = point.z - other.z;
            const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (!std::isfinite(d2))
              return result;
            ++result.pair_checks;
            if (metadata_internal_pair)
              ++result.metadata_internal_pairs_checked;
            const auto threshold = pbctopo_pair_clash_threshold(
                atoms[atom_idx], atoms[other_idx], fallback_cutoff);
            const double margin = std::sqrt(std::max(0.0, d2)) -
                                  threshold.cutoff;
            // Two different pairs: the one closest to clashing (smallest
            // margin, which decides the certificate) and the one closest in
            // distance (min_pair_d2, which candidate ranking reads).  With
            // unequal radii they need not coincide; reporting the margin
            // pair's distance as the minimum distance was a defect.
            if (margin < result.min_margin_A) {
              result.min_margin_A = margin;
              result.min_margin_pair_d2 = d2;
              result.clash_atom_a = atom_idx;
              result.clash_atom_b = other_idx;
            }
            if (!same_component && d2 < result.min_pair_d2) {
              result.min_pair_d2 = d2;
              result.min_pair_atom_a = atom_idx;
              result.min_pair_atom_b = other_idx;
            }
            const bool placed_clash = pbctopo_atoms_clash(
                d2, atoms[atom_idx], atoms[other_idx], fallback_cutoff,
                timing);
            if (metadata_internal_pair && placed_clash) {
              const auto wrapped_nearest =
                  wrapped_lattice->nearest_image_cartesian(
                      atoms[atom_idx].x - atoms[other_idx].x,
                      atoms[atom_idx].y - atoms[other_idx].y,
                      atoms[atom_idx].z - atoms[other_idx].z);
              if (!wrapped_nearest.valid) {
                result.termination_reason =
                    PbctopoStericTerminationReason::NumericalFailure;
                return result;
              }
              const bool clash_already_present =
                  wrapped_nearest.d2 + 1.0e-12 < threshold.cutoff2;
              if (!clash_already_present) {
                result.metadata_internal_clashes = 1;
                result.hard_clashes = 1;
                result.input_valid = true;
                result.decision_complete = true;
                result.termination_reason =
                    PbctopoStericTerminationReason::FirstHardClash;
                return result;
              }
              continue;
            }
            if (!metadata_internal_pair && placed_clash) {
              result.hard_clashes = 1;
              result.input_valid = true;
              result.decision_complete = true;
              result.termination_reason =
                  PbctopoStericTerminationReason::FirstHardClash;
              return result;
            }
          }
        }
      }
    }
  }
  result.decision_complete = true;
  result.pair_enumeration_complete = true;
  result.scan_complete = true;
  result.termination_reason = PbctopoStericTerminationReason::Completed;
  result.input_valid = true;
  return result;
}

bool steric_radius_model_valid(std::span<const PbctopoAtom> atoms,
                               double fallback_cutoff) {
  if (!std::isfinite(fallback_cutoff) || fallback_cutoff <= 0.0)
    return false;
  constexpr double radius_epsilon = 1.0e-12;
  for (const auto &atom : atoms) {
    if (!std::isfinite(atom.vdw_radius) ||
        !std::isfinite(atom.steric_radius) || atom.vdw_radius < 0.0 ||
        atom.steric_radius < 0.0) {
      return false;
    }
    switch (atom.steric_radius_source) {
    case PbctopoStericRadiusSource::ForceFieldTopology:
    case PbctopoStericRadiusSource::ForceFieldParameter:
    case PbctopoStericRadiusSource::InferredElement:
    case PbctopoStericRadiusSource::GenericFallback:
      if (!atom.has_vdw_radius || atom.vdw_radius <= radius_epsilon ||
          atom.steric_radius <= radius_epsilon ||
          atom.steric_radius > atom.vdw_radius + radius_epsilon) {
        return false;
      }
      break;
    case PbctopoStericRadiusSource::ExplicitZeroLennardJones:
      if (!atom.has_vdw_radius || atom.vdw_radius > radius_epsilon ||
          atom.steric_radius > radius_epsilon) {
        return false;
      }
      break;
    case PbctopoStericRadiusSource::Unavailable:
      if (atom.has_vdw_radius || atom.vdw_radius > radius_epsilon ||
          atom.steric_radius > radius_epsilon) {
        return false;
      }
      break;
    default:
      return false;
    }
  }
  return true;
}

} // namespace

PbctopoStericAudit evaluate_pbctopo_final_sterics(
    std::span<const PbctopoAtom> atoms,
    std::span<const Vec3> placed_points,
    std::span<const std::size_t> atom_component_index,
    const std::unordered_set<std::uint64_t> &excluded_pairs,
    double fallback_cutoff, PbctopoTimingStats *timing,
    const PbctopoLatticeMetric *wrapped_lattice) {
  return evaluate_pbctopo_sterics_impl(
      atoms, placed_points, atom_component_index, excluded_pairs,
      fallback_cutoff, timing, wrapped_lattice, false);
}

PbctopoFixedCertificateAudit evaluate_pbctopo_fixed_certificate_invariants(
    std::span<const PbctopoAtom> wrapped_atoms,
    std::span<const Vec3> component_reference_points,
    std::span<const std::size_t> atom_component_index,
    std::span<const PbctopoLocalEdge> local_edges,
    std::span<const PbctopoRequiredContact> explicit_contacts,
    const std::unordered_set<std::uint64_t> &excluded_pairs,
    double fallback_cutoff, const PbctopoLatticeMetric &wrapped_lattice,
    PbctopoTimingStats *timing) {
  PbctopoFixedCertificateAudit result;
  result.evaluated = true;
  if (wrapped_atoms.empty() ||
      component_reference_points.size() != wrapped_atoms.size() ||
      atom_component_index.size() != wrapped_atoms.size() ||
      !wrapped_lattice.valid() || !std::isfinite(fallback_cutoff) ||
      fallback_cutoff <= 0.0) {
    return result;
  }
  for (std::size_t component_idx : atom_component_index) {
    if (component_idx >= wrapped_atoms.size())
      return result;
  }

  result.radius_model_valid =
      steric_radius_model_valid(wrapped_atoms, fallback_cutoff);
  const PbctopoStericAudit fixed_sterics = evaluate_pbctopo_sterics_impl(
      wrapped_atoms, component_reference_points, atom_component_index,
      excluded_pairs, fallback_cutoff, timing, &wrapped_lattice, true);
  result.metadata_internal_pairs_checked =
      fixed_sterics.metadata_internal_pairs_checked;
  result.metadata_internal_clashes =
      fixed_sterics.metadata_internal_clashes;
  result.internal_steric_scan_complete =
      fixed_sterics.input_valid && fixed_sterics.scan_complete;
  if (!fixed_sterics.input_valid)
    return result;

  auto check_fixed_contact = [&](std::size_t a, std::size_t b,
                                 double maximum_squared) {
    ++result.required_contacts_checked;
    const Vec3 delta =
        component_reference_points[a] - component_reference_points[b];
    const double d2 =
        delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    if (!pbctopo_required_contact_satisfied(d2, maximum_squared))
      ++result.required_contacts_lost;
  };

  for (const auto &edge : local_edges) {
    if (edge.kind != PbctopoLocalEdgeKind::TopologyBond)
      continue;
    if (edge.a >= wrapped_atoms.size() || edge.b >= wrapped_atoms.size() ||
        edge.a == edge.b || !std::isfinite(edge.break_cutoff_d2) ||
        edge.break_cutoff_d2 < 0.0) {
      return result;
    }
    if (wrapped_atoms[edge.a].owner == wrapped_atoms[edge.b].owner ||
        atom_component_index[edge.a] != atom_component_index[edge.b]) {
      continue;
    }
    check_fixed_contact(edge.a, edge.b, edge.break_cutoff_d2);
  }

  for (const auto &contact : explicit_contacts) {
    if (contact.a >= wrapped_atoms.size() ||
        contact.b >= wrapped_atoms.size() || contact.a == contact.b ||
        !std::isfinite(contact.max_distance_A) ||
        contact.max_distance_A <= 0.0 || !std::isfinite(contact.weight) ||
        contact.weight <= 0.0) {
      return result;
    }
    if (atom_component_index[contact.a] !=
        atom_component_index[contact.b]) {
      continue;
    }
    check_fixed_contact(contact.a, contact.b,
                        contact.max_distance_A * contact.max_distance_A);
  }

  result.required_contact_scan_complete = true;
  result.input_valid = true;
  return result;
}

PbctopoGeometryClassification classify_pbctopo_final_geometry(
    std::span<const Vec3> placed_points,
    const PbctopoLatticeMetric &lattice, double unit_box_tolerance,
    const PbctopoImageEquationResiduals *image_equations) {
  PbctopoGeometryClassification result;
  if (placed_points.empty() || !std::isfinite(unit_box_tolerance) ||
      unit_box_tolerance < 0.0) {
    return result;
  }

  std::array<double, 3> minimum{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  std::array<double, 3> maximum{
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  std::array<double, 3> cartesian_minimum{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  std::array<double, 3> cartesian_maximum{
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  for (const Vec3 &point : placed_points) {
    if (!point_is_finite(point))
      return result;
    const auto scaled = lattice.cartesian_to_scaled(point.x, point.y, point.z);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(scaled[axis]))
        return result;
      minimum[axis] = std::min(minimum[axis], scaled[axis]);
      maximum[axis] = std::max(maximum[axis], scaled[axis]);
    }
    const std::array<double, 3> cartesian{point.x, point.y, point.z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      cartesian_minimum[axis] =
          std::min(cartesian_minimum[axis], cartesian[axis]);
      cartesian_maximum[axis] =
          std::max(cartesian_maximum[axis], cartesian[axis]);
    }
  }
  bool compact = true;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    result.scaled_span[axis] = maximum[axis] - minimum[axis];
    result.cartesian_extent_A[axis] =
        cartesian_maximum[axis] - cartesian_minimum[axis];
    compact = compact &&
              result.scaled_span[axis] <= 1.0 + unit_box_tolerance;
  }
  result.fits_current_basis_fundamental_parallelepiped = compact;
  result.fits_one_fundamental_cell = compact;
  if (image_equations != nullptr && image_equations->input_valid) {
    result.topological_winding_basis =
        image_equations->topological_winding_basis;
    result.topological_winding_rank =
        image_equations->topological_winding_rank;
  }
  const bool winding = result.topological_winding_rank > 0;
  if (compact && !winding)
    result.geometry_class = PbctopoGeometryClass::CompactUnitBox;
  else if (!compact && !winding)
    result.geometry_class = PbctopoGeometryClass::Extended;
  else if (compact)
    result.geometry_class = PbctopoGeometryClass::Winding;
  else
    result.geometry_class = PbctopoGeometryClass::ExtendedAndWinding;
  result.evaluated = true;
  return result;
}

PbctopoHardFeasibility evaluate_pbctopo_hard_feasibility(
    const PbctopoCutCandidate &candidate,
    const PbctopoImageEquationResiduals &image_equations,
    const PbctopoAssignmentPointResiduals &assignment_points,
    const PbctopoStericAudit &steric_audit,
    PbctopoRequiredContactResiduals required_contacts) {
  PbctopoHardFeasibility hard;
  hard.evaluated = candidate.valid;
  hard.assignment_available = candidate.assignment.valid &&
                              image_equations.input_valid &&
                              assignment_points.input_valid;
  hard.certificate_atoms_total =
      candidate.hard_feasibility.certificate_atoms_total;
  hard.hard_graph_atoms_covered =
      candidate.hard_feasibility.hard_graph_atoms_covered;
  hard.topology_bond_atoms_covered =
      candidate.hard_feasibility.topology_bond_atoms_covered;
  hard.metadata_only_atoms =
      candidate.hard_feasibility.metadata_only_atoms;
  hard.topology_edges_available =
      std::max(candidate.hard_feasibility.topology_edges_available,
               image_equations.topology_edges_checked);
  hard.metadata_edges_available =
      std::max(candidate.hard_feasibility.metadata_edges_available,
               image_equations.metadata_edges_checked);
  const std::size_t topology_and_metadata_edges =
      hard.metadata_edges_available >
              std::numeric_limits<std::size_t>::max() -
                  hard.topology_edges_available
          ? std::numeric_limits<std::size_t>::max()
          : hard.topology_edges_available +
                hard.metadata_edges_available;
  hard.hard_connectivity_edges_available =
      std::max(candidate.hard_feasibility.hard_connectivity_edges_available,
               topology_and_metadata_edges);
  hard.topology_edges_checked = image_equations.topology_edges_checked;
  hard.ambiguous_topology_edges = image_equations.ambiguous_topology_edges;
  hard.topology_edge_residuals = image_equations.topology_edge_residuals;
  hard.metadata_edges_checked = image_equations.metadata_edges_checked;
  hard.ambiguous_metadata_edges = image_equations.ambiguous_metadata_edges;
  hard.metadata_edge_image_residuals =
      image_equations.metadata_edge_image_residuals;
  hard.metadata_edge_distance_residuals =
      image_equations.metadata_edge_distance_residuals;
  hard.max_metadata_edge_distance_error_A =
      image_equations.max_metadata_edge_distance_error_A;
  hard.hard_edge_residuals =
      image_equations.topology_edge_residuals +
      image_equations.metadata_edge_image_residuals +
      image_equations.metadata_edge_distance_residuals;
  hard.atom_image_residuals = image_equations.atom_image_residuals;
  hard.max_atom_image_error = image_equations.max_atom_image_error;
  hard.assignment_point_residuals = assignment_points.point_residuals;
  hard.max_assignment_point_error = assignment_points.max_point_error;
  hard.topology_cycle_residuals = image_equations.topology_cycle_residuals;
  hard.unsupported_winding_rank =
      image_equations.topological_winding_rank;
  hard.steric_audit_available = steric_audit.input_valid;
  hard.steric_decision_complete = steric_audit.decision_complete;
  hard.steric_pair_enumeration_complete =
      steric_audit.pair_enumeration_complete;
  hard.steric_termination_reason =
      static_cast<std::uint8_t>(steric_audit.termination_reason);
  hard.steric_scan_complete = steric_audit.scan_complete;
  hard.steric_pairs_checked = steric_audit.pair_checks;
  hard.hard_clashes = steric_audit.hard_clashes;
  hard.metadata_internal_pairs_checked =
      steric_audit.metadata_internal_pairs_checked;
  hard.metadata_internal_clashes = steric_audit.metadata_internal_clashes;
  hard.steric_force_field_atoms = steric_audit.force_field_radius_atoms;
  hard.steric_explicit_zero_lj_atoms =
      steric_audit.explicit_zero_lj_radius_atoms;
  hard.steric_inferred_atoms = steric_audit.inferred_radius_atoms;
  hard.steric_generic_fallback_atoms =
      steric_audit.generic_fallback_radius_atoms;
  hard.steric_unavailable_atoms = steric_audit.unavailable_radius_atoms;
  hard.min_steric_margin_A = steric_audit.min_margin_A;
  hard.required_contacts_checked = required_contacts.contacts_checked;
  hard.required_contacts_lost = required_contacts.contacts_lost;
  hard.explicit_interface_contacts_checked =
      required_contacts.explicit_contacts_checked;
  hard.explicit_interface_contacts_lost =
      required_contacts.explicit_contacts_lost;
  hard.explicit_interface_weight_checked =
      required_contacts.explicit_weight_checked;
  hard.explicit_interface_weight_lost =
      required_contacts.explicit_weight_lost;
  if (required_contacts.explicit_constraints_requested &&
      !required_contacts.input_valid) {
    ++hard.required_contacts_lost;
    ++hard.explicit_interface_contacts_lost;
  }
  return hard;
}

bool certify_pbctopo_candidate(PbctopoCutCandidate &candidate,
                               const PbctopoImageEquationResiduals &image_equations,
                               const PbctopoAssignmentPointResiduals &assignment_points,
                               const PbctopoStericAudit &steric_audit,
                               PbctopoRequiredContactResiduals required_contacts) {
  candidate.hard_feasibility = evaluate_pbctopo_hard_feasibility(
      candidate, image_equations, assignment_points, steric_audit,
      required_contacts);
  return candidate.hard_feasibility.certified();
}

} // namespace titan_pbctopo

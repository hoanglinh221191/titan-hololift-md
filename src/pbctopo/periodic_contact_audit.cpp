#include "periodic_contact_audit.h"

#include "../BoxMatrix.h"
#include "../box_type.h"
#include "steric_clash.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace titan_pbctopo {
namespace {

struct CellKey {
  int x = 0;
  int y = 0;
  int z = 0;
  bool operator==(const CellKey &) const = default;
};

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const noexcept {
    std::size_t value = static_cast<std::size_t>(key.x) * 73856093u;
    value ^= static_cast<std::size_t>(key.y) * 19349663u;
    value ^= static_cast<std::size_t>(key.z) * 83492791u;
    return value;
  }
};

struct UnitPairKey {
  std::size_t a = 0;
  std::size_t b = 0;
  bool operator==(const UnitPairKey &) const = default;
};

struct UnitPairKeyHash {
  std::size_t operator()(const UnitPairKey &key) const noexcept {
    return std::hash<std::size_t>{}(key.a) * 1000003u ^
           std::hash<std::size_t>{}(key.b);
  }
};

struct ShiftRecord {
  Int3 shift{};
  std::size_t atom_pairs = 0;
  double min_d2 = std::numeric_limits<double>::infinity();
};

std::uint64_t pair_key(std::size_t a, std::size_t b) {
  const std::size_t lo = std::min(a, b);
  const std::size_t hi = std::max(a, b);
  return (static_cast<std::uint64_t>(lo) << 32) |
         static_cast<std::uint64_t>(hi);
}

double overlap_cutoff(const PbctopoAtom &lhs, const PbctopoAtom &rhs,
                      double scale, double fallback_cutoff, bool &exempt) {
  exempt = lhs.steric_radius_source ==
               PbctopoStericRadiusSource::ExplicitZeroLennardJones ||
           rhs.steric_radius_source ==
               PbctopoStericRadiusSource::ExplicitZeroLennardJones;
  if (exempt)
    return 0.0;
  const bool radii_valid = lhs.has_vdw_radius && rhs.has_vdw_radius &&
                           std::isfinite(lhs.vdw_radius) &&
                           std::isfinite(rhs.vdw_radius) &&
                           lhs.vdw_radius >= 0.0 && rhs.vdw_radius >= 0.0;
  return radii_valid
             ? std::max(1.0e-6, scale * (lhs.vdw_radius + rhs.vdw_radius))
             : fallback_cutoff;
}

} // namespace

PbctopoContactCycleLattice compute_pbctopo_contact_cycle_lattice(
    std::size_t unit_count, std::span<const PbctopoUnitRelation> relations) {
  PbctopoContactCycleLattice result;
  result.units = unit_count;
  struct Adjacent {
    std::size_t unit = 0;
    Int3 shift{};
  };
  std::vector<std::vector<Adjacent>> adjacency(unit_count);
  for (const auto &relation : relations) {
    if (relation.unit_a >= unit_count || relation.unit_b >= unit_count)
      return result;
    if (relation.unit_a == relation.unit_b)
      continue;
    adjacency[relation.unit_a].push_back({relation.unit_b, relation.shift});
    adjacency[relation.unit_b].push_back(
        {relation.unit_a,
         {-relation.shift.x, -relation.shift.y, -relation.shift.z}});
  }
  // Potentials on a spanning forest; o_b = o_a + s along tree edges.
  std::vector<Int3> potential(unit_count);
  std::vector<std::size_t> label(unit_count,
                                 std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> queue;
  queue.reserve(unit_count);
  for (std::size_t root = 0; root < unit_count; ++root) {
    if (label[root] != std::numeric_limits<std::size_t>::max())
      continue;
    const std::size_t component = result.graph_components++;
    label[root] = component;
    potential[root] = {};
    queue.clear();
    queue.push_back(root);
    for (std::size_t head = 0; head < queue.size(); ++head) {
      const std::size_t current = queue[head];
      for (const auto &next : adjacency[current]) {
        if (label[next.unit] != std::numeric_limits<std::size_t>::max())
          continue;
        label[next.unit] = component;
        potential[next.unit] = {potential[current].x + next.shift.x,
                                potential[current].y + next.shift.y,
                                potential[current].z + next.shift.z};
        queue.push_back(next.unit);
      }
    }
  }
  std::vector<PbctopoLatticeSpan> component_span(result.graph_components);
  std::vector<unsigned char> self_unit(unit_count, 0);
  for (const auto &relation : relations) {
    Int3 cycle{};
    if (relation.unit_a == relation.unit_b) {
      cycle = relation.shift;
      if (!is_zero_delta(cycle)) {
        self_unit[relation.unit_a] = 1;
        result.self_image_atom_pairs += relation.atom_pairs;
        const double d = std::sqrt(std::max(0.0, relation.min_d2));
        if (!(d >= result.self_image_min_distance_A))
          result.self_image_min_distance_A = d;
      }
    } else {
      const Int3 &pa = potential[relation.unit_a];
      const Int3 &pb = potential[relation.unit_b];
      cycle = {pa.x + relation.shift.x - pb.x, pa.y + relation.shift.y - pb.y,
               pa.z + relation.shift.z - pb.z};
    }
    if (is_zero_delta(cycle))
      continue;
    ++result.nonzero_cycle_relations;
    pbctopo_lattice_span_add(component_span[label[relation.unit_a]], cycle);
    pbctopo_lattice_span_add(result.span, cycle);
  }
  for (const auto &span : component_span) {
    if (span.rank > 0)
      ++result.periodic_graph_components;
    result.max_component_rank = std::max(result.max_component_rank, span.rank);
  }
  result.self_image_units = static_cast<std::size_t>(
      std::count(self_unit.begin(), self_unit.end(), 1));
  result.evaluated = true;
  return result;
}

PbctopoPeriodicContactAudit
audit_pbctopo_periodic_contacts(const box &Box,
                                const PbctopoPeriodicContactInput &input) {
  PbctopoPeriodicContactAudit result;
  result.evaluated = true;
  result.overlap.evaluated = true;
  result.overlap.vdw_scale = input.overlap_vdw_scale;
  result.overlap.fallback_cutoff_A = input.overlap_fallback_cutoff_A;
  const auto &atoms = input.atoms;
  if (atoms.size() != input.atom_unit.size() ||
      atoms.size() != input.lift_images.size()) {
    result.error = "periodic contact audit input size mismatch";
    return result;
  }
  for (const std::size_t unit : input.atom_unit) {
    if (unit >= input.unit_count) {
      result.error = "periodic contact audit unit index out of range";
      return result;
    }
  }
  if (atoms.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    result.error = "periodic contact audit atom count exceeds the pair key";
    return result;
  }
  const double cutoff = input.contact_cutoff_A;
  if (!std::isfinite(cutoff) || !(cutoff > 0.0)) {
    result.error = "periodic contact audit cutoff is not positive";
    return result;
  }
  const BoxMatrix box_matrix(Box);
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid()) {
    result.error = "periodic contact audit lattice is invalid";
    return result;
  }
  auto bound = [&](int column) {
    return std::min(0.5, cutoff * std::sqrt(box_matrix.inv[0][column] *
                                                box_matrix.inv[0][column] +
                                            box_matrix.inv[1][column] *
                                                box_matrix.inv[1][column] +
                                            box_matrix.inv[2][column] *
                                                box_matrix.inv[2][column]) +
                             1.0e-9);
  };
  int counts[3]{};
  for (int axis = 0; axis < 3; ++axis) {
    const double b = bound(axis);
    const double raw = std::floor(1.0 / std::max(b, 1.0e-6));
    if (!std::isfinite(raw) || raw < 1.0 || raw > 1.0e6) {
      result.error = "periodic contact audit cell setup failed";
      return result;
    }
    counts[axis] = static_cast<int>(raw);
  }
  auto cell_of = [&](const PbctopoAtom &atom, CellKey &key) {
    const double s[3]{atom.sx, atom.sy, atom.sz};
    int idx[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(s[axis]))
        return false;
      const double wrapped = s[axis] - std::floor(s[axis]);
      idx[axis] = std::clamp(
          static_cast<int>(std::floor(wrapped * counts[axis])), 0,
          counts[axis] - 1);
    }
    key = {idx[0], idx[1], idx[2]};
    return true;
  };
  auto wrap = [](int value, int mod) {
    int out = value % mod;
    return out < 0 ? out + mod : out;
  };

  std::unordered_map<UnitPairKey, std::vector<ShiftRecord>, UnitPairKeyHash>
      shifts_by_pair;
  std::unordered_map<std::uint64_t, char> overlap_unit_pairs;
  auto record_relation = [&](std::size_t i, std::size_t j,
                             const PbctopoNearestImage &mic) {
    // mic: x_j - x_i + B m is the nearest image; s = m - (n_j - n_i).
    std::size_t ua = input.atom_unit[i];
    std::size_t ub = input.atom_unit[j];
    Int3 s{mic.image_x - (input.lift_images[j].x - input.lift_images[i].x),
           mic.image_y - (input.lift_images[j].y - input.lift_images[i].y),
           mic.image_z - (input.lift_images[j].z - input.lift_images[i].z)};
    if (ua == ub) {
      if (is_zero_delta(s))
        return;
      s = pbctopo_canonical_lattice_direction(s);
    } else if (ub < ua) {
      std::swap(ua, ub);
      s = {-s.x, -s.y, -s.z};
    }
    auto &records = shifts_by_pair[{ua, ub}];
    auto it = std::find_if(records.begin(), records.end(),
                           [&](const ShiftRecord &r) { return same_delta(r.shift, s); });
    if (it == records.end()) {
      records.push_back({s, 0, std::numeric_limits<double>::infinity()});
      it = std::prev(records.end());
    }
    ++it->atom_pairs;
    it->min_d2 = std::min(it->min_d2, mic.d2);
  };

  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> grid;
  grid.reserve(std::max<std::size_t>(64, atoms.size() / 2));
  const double cutoff2 = cutoff * cutoff;
  // A reduced basis vector is at most twice the shortest lattice vector in
  // three dimensions, so half its length bounds the shortest vector below.
  const double shortest_lower = 0.5 * lattice.shortest_basis_length();
  const double same_image_skip2 =
      std::isfinite(shortest_lower) && shortest_lower > 2.0 * cutoff ? cutoff2
                                                                     : 0.0;
  bool complete = true;
  double min_ratio = std::numeric_limits<double>::infinity();
  double min_distance2 = std::numeric_limits<double>::infinity();
  std::vector<CellKey> neighbor_cells;
  neighbor_cells.reserve(27);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const auto &atom = atoms[i];
    CellKey key;
    if (!cell_of(atom, key)) {
      complete = false;
      continue;
    }
    neighbor_cells.clear();
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          const CellKey n{wrap(key.x + dx, counts[0]), wrap(key.y + dy, counts[1]),
                          wrap(key.z + dz, counts[2])};
          if (std::find(neighbor_cells.begin(), neighbor_cells.end(), n) ==
              neighbor_cells.end())
            neighbor_cells.push_back(n);
        }
    for (const auto &cell : neighbor_cells) {
      const auto found = grid.find(cell);
      if (found == grid.end())
        continue;
      for (const std::size_t j : found->second) {
        const auto &other = atoms[j];
        const bool same_unit = input.atom_unit[i] == input.atom_unit[j];
        const bool both_heavy = atom.is_heavy && other.is_heavy;
        if (same_unit && !both_heavy)
          continue;
        // Within one unit, atoms in the same lift image whose plain distance
        // v is under the cutoff have s = 0 when every nonzero lattice vector
        // is longer than twice the cutoff (|B m| <= |v| + |v + B m|).
        if (same_unit && input.lift_images[i] == input.lift_images[j] &&
            same_image_skip2 > 0.0) {
          const double ddx = other.x - atom.x;
          const double ddy = other.y - atom.y;
          const double ddz = other.z - atom.z;
          if (ddx * ddx + ddy * ddy + ddz * ddz < same_image_skip2)
            continue;
        }
        const PbctopoNearestImage mic = lattice.nearest_image_cartesian(
            other.x - atom.x, other.y - atom.y, other.z - atom.z);
        if (!mic.valid) {
          complete = false;
          continue;
        }
        if (mic.d2 > cutoff2)
          continue;
        if (both_heavy)
          record_relation(i, j, mic);
        if (same_unit)
          continue;
        if (input.overlap_excluded_pairs != nullptr &&
            input.overlap_excluded_pairs->contains(pair_key(i, j)))
          continue;
        bool exempt = false;
        const double threshold =
            overlap_cutoff(atom, other, input.overlap_vdw_scale,
                           input.overlap_fallback_cutoff_A, exempt);
        if (exempt)
          continue;
        const double d = std::sqrt(std::max(0.0, mic.d2));
        min_distance2 = std::min(min_distance2, mic.d2);
        if (threshold > 0.0)
          min_ratio = std::min(min_ratio, d / threshold);
        if (mic.d2 + 1.0e-12 < threshold * threshold) {
          ++result.overlap.overlap_pairs;
          overlap_unit_pairs.emplace(
              pair_key(input.atom_unit[i], input.atom_unit[j]), 1);
        }
      }
    }
    grid[key].push_back(i);
  }
  for (const auto &[a, b] : input.required_atom_pairs) {
    if (a >= atoms.size() || b >= atoms.size() || a == b) {
      complete = false;
      continue;
    }
    const PbctopoNearestImage mic = lattice.nearest_image_cartesian(
        atoms[b].x - atoms[a].x, atoms[b].y - atoms[a].y, atoms[b].z - atoms[a].z);
    if (!mic.valid) {
      complete = false;
      continue;
    }
    record_relation(a, b, mic);
  }

  result.relations.reserve(shifts_by_pair.size());
  for (const auto &[key, records] : shifts_by_pair) {
    for (const auto &record : records)
      result.relations.push_back(
          {key.a, key.b, record.shift, record.atom_pairs, record.min_d2});
  }
  std::sort(result.relations.begin(), result.relations.end(),
            [](const PbctopoUnitRelation &lhs, const PbctopoUnitRelation &rhs) {
              return std::tie(lhs.unit_a, lhs.unit_b, lhs.shift.x, lhs.shift.y,
                              lhs.shift.z) <
                     std::tie(rhs.unit_a, rhs.unit_b, rhs.shift.x, rhs.shift.y,
                              rhs.shift.z);
            });
  result.overlap.overlap_unit_pairs = overlap_unit_pairs.size();
  if (std::isfinite(min_distance2))
    result.overlap.min_distance_A = std::sqrt(min_distance2);
  if (std::isfinite(min_ratio))
    result.overlap.min_ratio = min_ratio;
  result.overlap.complete = complete;
  result.cycle = compute_pbctopo_contact_cycle_lattice(
      input.unit_count, std::span<const PbctopoUnitRelation>(result.relations));
  result.complete = complete && result.cycle.evaluated;
  if (!complete)
    result.error = "periodic contact audit skipped atoms or image queries";
  return result;
}

bool pbctopo_lift_images_from_points(const box &Box,
                                     std::span<const PbctopoAtom> atoms,
                                     std::span<const Vec3> points,
                                     std::vector<Int3> &images) {
  if (atoms.size() != points.size())
    return false;
  const BoxMatrix box_matrix(Box);
  images.assign(atoms.size(), Int3{});
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const double dx = points[i].x - atoms[i].x;
    const double dy = points[i].y - atoms[i].y;
    const double dz = points[i].z - atoms[i].z;
    const double f[3]{
        dx * box_matrix.inv[0][0] + dy * box_matrix.inv[1][0] + dz * box_matrix.inv[2][0],
        dx * box_matrix.inv[0][1] + dy * box_matrix.inv[1][1] + dz * box_matrix.inv[2][1],
        dx * box_matrix.inv[0][2] + dy * box_matrix.inv[1][2] + dz * box_matrix.inv[2][2]};
    std::int64_t r[3]{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!std::isfinite(f[axis]) || std::abs(f[axis]) > 1.0e9)
        return false;
      const double rounded = std::nearbyint(f[axis]);
      if (std::abs(f[axis] - rounded) > 1.0e-3)
        return false;
      r[axis] = static_cast<std::int64_t>(rounded);
    }
    images[i] = {r[0], r[1], r[2]};
  }
  return true;
}

const char *pbctopo_steric_policy_name(bool steric_filter) {
  return steric_filter ? "filter" : "report";
}

} // namespace titan_pbctopo

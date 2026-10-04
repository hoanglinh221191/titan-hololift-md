#include "tabi_ses_patch_provenance.h"

#include "tabi_ses_patch_intersections.h"
#include "tabi_ses_patch_ray.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

[[nodiscard]] Vec3 subtract(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 add(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 scale(double factor, const Vec3 &value) noexcept {
  return {factor * value.x, factor * value.y, factor * value.z};
}

[[nodiscard]] double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] double norm(const Vec3 &value) noexcept {
  return std::sqrt(dot(value, value));
}

[[nodiscard]] bool finite_vec(const Vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool checked_add(std::size_t &value,
                               std::size_t increment) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - increment)
    return false;
  value += increment;
  return true;
}

[[nodiscard]] bool checked_payload(std::size_t capacity, std::size_t width,
                                   std::size_t &bytes) noexcept {
  if (capacity > std::numeric_limits<std::size_t>::max() / width)
    return false;
  return checked_add(bytes, capacity * width);
}

[[nodiscard]] bool checked_multiply(std::size_t left, std::size_t right,
                                    std::size_t &result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

[[nodiscard]] bool valid_patch_kind(PatchKind kind) noexcept {
  switch (kind) {
  case PatchKind::contact:
  case PatchKind::toroidal:
  case PatchKind::reentrant:
    return true;
  }
  return false;
}

[[nodiscard]] bool bounds_finite_ordered(const PatchBounds &bounds) noexcept {
  return finite_vec(bounds.lower) && finite_vec(bounds.upper) &&
         bounds.lower.x <= bounds.upper.x && bounds.lower.y <= bounds.upper.y &&
         bounds.lower.z <= bounds.upper.z;
}

[[nodiscard]] bool bounds_equal(const PatchBounds &left,
                                const PatchBounds &right) noexcept {
  return left.lower.x == right.lower.x && left.lower.y == right.lower.y &&
         left.lower.z == right.lower.z && left.upper.x == right.upper.x &&
         left.upper.y == right.upper.y && left.upper.z == right.upper.z;
}

[[nodiscard]] bool bounds_contain(const PatchBounds &outer,
                                  const PatchBounds &inner) noexcept {
  return bounds_finite_ordered(outer) && bounds_finite_ordered(inner) &&
         outer.lower.x <= inner.lower.x && outer.lower.y <= inner.lower.y &&
         outer.lower.z <= inner.lower.z && outer.upper.x >= inner.upper.x &&
         outer.upper.y >= inner.upper.y && outer.upper.z >= inner.upper.z;
}

[[nodiscard]] bool patch_ref_in_range(const PatchComplex &complex,
                                      LocalPatchRef ref) noexcept {
  if (!valid_patch_kind(ref.kind))
    return false;
  switch (ref.kind) {
  case PatchKind::contact:
    return ref.patch_index < complex.contacts.size();
  case PatchKind::toroidal:
    return ref.patch_index < complex.tori.size();
  case PatchKind::reentrant:
    return ref.patch_index < complex.reentrant.size();
  }
  return false;
}

[[nodiscard]] bool
canonical_patch_id_checked(const PatchComplex &complex, LocalPatchRef ref,
                           std::uint32_t &canonical_id) noexcept {
  if (!patch_ref_in_range(complex, ref))
    return false;
  std::size_t value = ref.patch_index;
  if (ref.kind == PatchKind::toroidal) {
    if (!checked_add(value, complex.contacts.size()))
      return false;
  } else if (ref.kind == PatchKind::reentrant) {
    if (!checked_add(value, complex.contacts.size()) ||
        !checked_add(value, complex.tori.size()))
      return false;
  }
  if (value > std::numeric_limits<std::uint32_t>::max())
    return false;
  canonical_id = static_cast<std::uint32_t>(value);
  return true;
}

[[nodiscard]] bool validate_offsets(const std::vector<std::size_t> &offsets,
                                    std::size_t block_count,
                                    std::size_t payload_size) noexcept {
  if (block_count == std::numeric_limits<std::size_t>::max() ||
      offsets.size() != block_count + 1U || offsets.empty() ||
      offsets.front() != 0U || offsets.back() != payload_size)
    return false;
  for (std::size_t block = 0U; block < block_count; ++block) {
    if (offsets[block] > offsets[block + 1U] ||
        offsets[block + 1U] > payload_size)
      return false;
  }
  return true;
}

[[nodiscard]] bool
validate_spatial_index(const PatchComplex &complex,
                       const LocalPatchProjectionIndex &index,
                       std::size_t total_patches) {
  if (index.spatial_entries.size() != total_patches ||
      total_patches > std::numeric_limits<std::uint32_t>::max())
    return false;
  if (total_patches == 0U)
    return index.spatial_nodes.empty();
  if (index.spatial_nodes.empty() ||
      index.spatial_nodes.size() > std::numeric_limits<std::uint32_t>::max())
    return false;
  if (index.spatial_nodes.front().begin != 0U ||
      index.spatial_nodes.front().end != index.spatial_entries.size())
    return false;

  std::vector<std::uint8_t> canonical_seen(total_patches, 0U);
  for (const SpatialPatchEntry &entry : index.spatial_entries) {
    if (!patch_ref_in_range(complex, entry.patch) ||
        !bounds_finite_ordered(entry.bounds))
      return false;
    PatchRefId packed;
    if (!PatchRefId::try_make(entry.patch.kind, entry.patch.patch_index,
                              packed) ||
        !packed.valid())
      return false;
    std::uint32_t expected_id = 0U;
    if (!canonical_patch_id_checked(complex, entry.patch, expected_id) ||
        entry.canonical_id != expected_id ||
        expected_id >= canonical_seen.size() ||
        canonical_seen[expected_id] != 0U)
      return false;
    const PatchBounds expected_bounds = outward_padded_patch_bounds(
        complex, entry.patch.kind, entry.patch.patch_index);
    if (!bounds_finite_ordered(expected_bounds) ||
        !bounds_equal(entry.bounds, expected_bounds))
      return false;
    canonical_seen[expected_id] = 1U;
  }
  if (std::any_of(canonical_seen.begin(), canonical_seen.end(),
                  [](std::uint8_t seen) { return seen != 1U; }))
    return false;

  constexpr std::uint32_t sentinel = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint8_t> node_seen(index.spatial_nodes.size(), 0U);
  std::vector<std::uint32_t> stack;
  stack.reserve(index.spatial_nodes.size());
  stack.push_back(0U);
  while (!stack.empty()) {
    const std::uint32_t node_index = stack.back();
    stack.pop_back();
    if (node_index >= index.spatial_nodes.size() || node_seen[node_index] != 0U)
      return false;
    node_seen[node_index] = 1U;
    const SpatialPatchNode &node = index.spatial_nodes[node_index];
    if (!bounds_finite_ordered(node.bounds) || node.begin >= node.end ||
        node.end > index.spatial_entries.size())
      return false;
    if (node.is_leaf()) {
      if (node.right != sentinel)
        return false;
      for (std::size_t position = node.begin; position < node.end; ++position) {
        if (!bounds_contain(node.bounds,
                            index.spatial_entries[position].bounds))
          return false;
      }
      continue;
    }
    if (node.right == sentinel || node.left == node.right ||
        node.left >= index.spatial_nodes.size() ||
        node.right >= index.spatial_nodes.size())
      return false;
    const SpatialPatchNode &left = index.spatial_nodes[node.left];
    const SpatialPatchNode &right = index.spatial_nodes[node.right];
    if (left.begin != node.begin || left.end != right.begin ||
        right.end != node.end || !bounds_contain(node.bounds, left.bounds) ||
        !bounds_contain(node.bounds, right.bounds))
      return false;
    stack.push_back(node.right);
    stack.push_back(node.left);
  }
  return std::all_of(node_seen.begin(), node_seen.end(),
                     [](std::uint8_t seen) { return seen == 1U; });
}

[[nodiscard]] bool
validate_constraint_index(const PatchComplex &complex,
                          const LocalPatchProjectionIndex &index) noexcept {
  if (!validate_offsets(index.contact_constraint_offsets,
                        complex.contacts.size(),
                        index.contact_constraints.size()) ||
      !validate_offsets(index.contact_two_active_offsets,
                        complex.contacts.size(),
                        index.contact_two_active_directions.size()) ||
      index.contact_domain_empty.size() != complex.contacts.size() ||
      !validate_offsets(index.reentrant_constraint_offsets,
                        complex.reentrant.size(),
                        index.reentrant_constraints.size()) ||
      !validate_offsets(index.reentrant_two_active_offsets,
                        complex.reentrant.size(),
                        index.reentrant_two_active_directions.size()) ||
      index.reentrant_domain_empty.size() != complex.reentrant.size())
    return false;
  const auto valid_constraint = [](const SpherePatchConstraint &constraint) {
    const double normal_squared = dot(constraint.normal, constraint.normal);
    return finite_vec(constraint.normal) &&
           std::isfinite(constraint.minimum_dot) &&
           std::isfinite(normal_squared) &&
           std::abs(normal_squared - 1.0) <= 2.0e-10;
  };
  if (!std::all_of(index.contact_constraints.begin(),
                   index.contact_constraints.end(), valid_constraint) ||
      !std::all_of(index.reentrant_constraints.begin(),
                   index.reentrant_constraints.end(), valid_constraint))
    return false;
  const auto valid_direction = [](const Vec3 &direction) {
    const double length_squared = dot(direction, direction);
    return finite_vec(direction) && std::isfinite(length_squared) &&
           std::abs(length_squared - 1.0) <= 2.0e-12;
  };
  if (!std::all_of(index.contact_two_active_directions.begin(),
                   index.contact_two_active_directions.end(),
                   valid_direction) ||
      !std::all_of(index.reentrant_two_active_directions.begin(),
                   index.reentrant_two_active_directions.end(),
                   valid_direction) ||
      !std::all_of(index.contact_domain_empty.begin(),
                   index.contact_domain_empty.end(),
                   [](std::uint8_t empty) { return empty <= 1U; }) ||
      !std::all_of(index.reentrant_domain_empty.begin(),
                   index.reentrant_domain_empty.end(),
                   [](std::uint8_t empty) { return empty <= 1U; }))
    return false;
  return true;
}

[[nodiscard]] std::size_t relation_index(PatchRelation relation) noexcept {
  return static_cast<std::size_t>(relation);
}

[[nodiscard]] PatchRelation worse_relation(PatchRelation first,
                                           PatchRelation second) noexcept {
  const auto rank = [](PatchRelation relation) {
    switch (relation) {
    case PatchRelation::same_surface:
      return 0;
    case PatchRelation::curve_adjacent:
      return 1;
    case PatchRelation::point_only:
      return 2;
    case PatchRelation::separated:
      return 3;
    case PatchRelation::uncertified:
      return 4;
    }
    return 4;
  };
  return rank(first) >= rank(second) ? first : second;
}

} // namespace

PatchComplexFingerprint
fingerprint_patch_complex(const PatchComplex &complex) noexcept {
  PatchComplexFingerprint result;
  std::uint64_t first = 1469598103934665603ULL;
  std::uint64_t second = 0x9e3779b97f4a7c15ULL;
  const auto mix = [&](std::uint64_t value) {
    first ^= value;
    first *= 1099511628211ULL;
    second ^= value + 0x9e3779b97f4a7c15ULL + (second << 6U) + (second >> 2U);
    second = std::rotl(second, 27) * 0x94d049bb133111ebULL;
  };
  const auto mix_size = [&](std::size_t value) {
    mix(static_cast<std::uint64_t>(value));
  };
  const auto mix_double = [&](double value) {
    mix(std::bit_cast<std::uint64_t>(value));
    return std::isfinite(value);
  };
  bool valid = complex.status == "pass" &&
               std::isfinite(complex.probe_radius) &&
               complex.probe_radius > 0.0 &&
               complex.atom_neighbors.size() == complex.atoms.size();
  mix(0x5041544348434f4dULL);
  valid = mix_double(complex.probe_radius) && valid;
  mix_size(complex.atoms.size());
  for (const Atom &atom : complex.atoms) {
    valid = mix_double(atom.x) && mix_double(atom.y) && mix_double(atom.z) &&
            mix_double(atom.radius) && atom.radius > 0.0 && valid;
  }
  mix_size(complex.atom_neighbors.size());
  for (std::size_t atom = 0U; atom < complex.atom_neighbors.size(); ++atom) {
    const std::vector<std::size_t> &neighbors = complex.atom_neighbors[atom];
    mix_size(neighbors.size());
    for (const std::size_t neighbor : neighbors) {
      mix_size(neighbor);
      valid = neighbor < complex.atoms.size() && neighbor != atom && valid;
    }
  }
  mix_size(complex.contacts.size());
  for (const ContactPatch &patch : complex.contacts) {
    mix_size(patch.atom);
    valid = patch.atom < complex.atoms.size() && mix_double(patch.center.x) &&
            mix_double(patch.center.y) && mix_double(patch.center.z) &&
            mix_double(patch.atom_radius) &&
            mix_double(patch.expanded_radius) && patch.atom_radius > 0.0 &&
            patch.expanded_radius > patch.atom_radius && valid;
  }
  mix_size(complex.tori.size());
  for (const ToroidalPatch &patch : complex.tori) {
    mix_size(patch.atom_i);
    mix_size(patch.atom_j);
    valid = patch.atom_i < complex.atoms.size() &&
            patch.atom_j < complex.atoms.size() &&
            patch.atom_i != patch.atom_j && mix_double(patch.circle_center.x) &&
            mix_double(patch.circle_center.y) &&
            mix_double(patch.circle_center.z) && mix_double(patch.axis.x) &&
            mix_double(patch.axis.y) && mix_double(patch.axis.z) &&
            mix_double(patch.basis_u.x) && mix_double(patch.basis_u.y) &&
            mix_double(patch.basis_u.z) && mix_double(patch.basis_v.x) &&
            mix_double(patch.basis_v.y) && mix_double(patch.basis_v.z) &&
            mix_double(patch.circle_radius) &&
            mix_double(patch.axis_offset_i) &&
            mix_double(patch.expanded_radius_i) &&
            mix_double(patch.expanded_radius_j) && valid;
    mix_size(patch.exposed_intervals.size());
    for (const AngularInterval &interval : patch.exposed_intervals) {
      valid = mix_double(interval.begin) && mix_double(interval.end) && valid;
    }
  }
  mix_size(complex.reentrant.size());
  for (const ReentrantPatch &patch : complex.reentrant) {
    valid = mix_double(patch.probe_center.x) &&
            mix_double(patch.probe_center.y) &&
            mix_double(patch.probe_center.z) && valid;
    for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
      mix_size(patch.atoms[vertex]);
      valid = patch.atoms[vertex] < complex.atoms.size() &&
              mix_double(patch.contact_directions[vertex].x) &&
              mix_double(patch.contact_directions[vertex].y) &&
              mix_double(patch.contact_directions[vertex].z) && valid;
    }
  }
  result.first = first;
  result.second = second;
  result.valid = valid;
  return result;
}

std::size_t
AnalyticPatchAdjacencyOracle::retained_payload_bytes() const noexcept {
  std::size_t bytes = 0U;
  if (!checked_payload(contacts_.capacity(), sizeof(ContactData), bytes) ||
      !checked_payload(torus_atoms_.capacity(), sizeof(TorusAtoms), bytes) ||
      !checked_payload(reentrant_.capacity(), sizeof(ReentrantData), bytes) ||
      !checked_payload(junctions_.capacity(), sizeof(TripleJunction), bytes) ||
      !checked_payload(torus_junction_order_.capacity(), sizeof(std::uint32_t),
                       bytes) ||
      !checked_add(bytes, seams_.retained_payload_bytes()))
    return std::numeric_limits<std::size_t>::max();
  return bytes;
}

PatchRelation
AnalyticPatchAdjacencyOracle::relation(PatchRefId first,
                                       PatchRefId second) const noexcept {
  const auto valid_ref = [&](PatchRefId ref) {
    if (!ref.valid())
      return false;
    switch (ref.kind()) {
    case PatchKind::contact:
      return ref.index() < contacts_.size();
    case PatchKind::toroidal:
      return ref.index() < torus_atoms_.size();
    case PatchKind::reentrant:
      return ref.index() < reentrant_.size();
    }
    return false;
  };
  if (!valid_ref(first) || !valid_ref(second))
    return PatchRelation::uncertified;
  if (first == second)
    return PatchRelation::same_surface;
  if (!certified_)
    return PatchRelation::uncertified;

  const auto bounds_for = [&](PatchRefId ref) -> const Bounds * {
    switch (ref.kind()) {
    case PatchKind::contact:
      return &contacts_[ref.index()].bounds;
    case PatchKind::toroidal:
      return &torus_atoms_[ref.index()].bounds;
    case PatchKind::reentrant:
      return &reentrant_[ref.index()].bounds;
    }
    return nullptr;
  };
  const auto bounds_separated = [&](PatchRefId left, PatchRefId right) {
    const Bounds *a = bounds_for(left);
    const Bounds *b = bounds_for(right);
    if (a == nullptr || b == nullptr)
      return false;
    const double coordinate_scale = std::max(
        {1.0, std::abs(a->lower.x), std::abs(a->lower.y), std::abs(a->lower.z),
         std::abs(a->upper.x), std::abs(a->upper.y), std::abs(a->upper.z),
         std::abs(b->lower.x), std::abs(b->lower.y), std::abs(b->lower.z),
         std::abs(b->upper.x), std::abs(b->upper.y), std::abs(b->upper.z)});
    const double tolerance =
        256.0 * std::numeric_limits<double>::epsilon() * coordinate_scale;
    return a->upper.x + tolerance < b->lower.x ||
           b->upper.x + tolerance < a->lower.x ||
           a->upper.y + tolerance < b->lower.y ||
           b->upper.y + tolerance < a->lower.y ||
           a->upper.z + tolerance < b->lower.z ||
           b->upper.z + tolerance < a->lower.z;
  };
  const auto torus_side_certified = [&](std::size_t torus_index,
                                        std::uint32_t atom) {
    if (torus_index >= torus_atoms_.size() ||
        torus_index >= seams_.torus_contact_curve_certified.size())
      return false;
    const TorusAtoms &torus = torus_atoms_[torus_index];
    if (atom == torus.atom_i)
      return seams_.torus_contact_curve_certified[torus_index][0] != 0U;
    if (atom == torus.atom_j)
      return seams_.torus_contact_curve_certified[torus_index][1] != 0U;
    return false;
  };

  PatchRefId left = first;
  PatchRefId right = second;
  if (static_cast<unsigned>(left.kind()) > static_cast<unsigned>(right.kind()))
    std::swap(left, right);

  if (left.kind() == PatchKind::contact && right.kind() == PatchKind::contact) {
    return bounds_separated(left, right) ? PatchRelation::separated
                                         : PatchRelation::uncertified;
  }

  if (left.kind() == PatchKind::contact &&
      right.kind() == PatchKind::toroidal) {
    const std::uint32_t atom = contacts_[left.index()].atom;
    if (torus_side_certified(right.index(), atom))
      return PatchRelation::curve_adjacent;
    const TorusAtoms &torus = torus_atoms_[right.index()];
    if (atom == torus.atom_i || atom == torus.atom_j)
      return PatchRelation::uncertified;
    return bounds_separated(left, right) ? PatchRelation::separated
                                         : PatchRelation::uncertified;
  }

  if (left.kind() == PatchKind::contact &&
      right.kind() == PatchKind::reentrant) {
    const ReentrantData &reentrant = reentrant_[right.index()];
    const std::uint32_t atom = contacts_[left.index()].atom;
    bool incident = false;
    for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
      incident = incident || reentrant.atoms[vertex] == atom;
      const std::size_t junction_index = right.index() * 3U + vertex;
      if (junction_index < junctions_.size()) {
        const TripleJunction &junction = junctions_[junction_index];
        if (junction.certified != 0U && junction.contact == left &&
            junction.reentrant == right)
          return PatchRelation::point_only;
      }
    }
    if (incident)
      return PatchRelation::uncertified;
    return bounds_separated(left, right) ? PatchRelation::separated
                                         : PatchRelation::uncertified;
  }

  if (left.kind() == PatchKind::toroidal &&
      right.kind() == PatchKind::toroidal) {
    const std::uint32_t packed_a = std::min(left.packed(), right.packed());
    const std::uint32_t packed_b = std::max(left.packed(), right.packed());
    const auto key = [&](std::uint32_t junction_index) {
      const TripleJunction &junction = junctions_[junction_index];
      return std::pair{std::min(junction.torus_first.packed(),
                                junction.torus_second.packed()),
                       std::max(junction.torus_first.packed(),
                                junction.torus_second.packed())};
    };
    const auto first_match = std::lower_bound(
        torus_junction_order_.begin(), torus_junction_order_.end(),
        std::pair{packed_a, packed_b},
        [&](std::uint32_t junction_index,
            const std::pair<std::uint32_t, std::uint32_t> &value) {
          return key(junction_index) < value;
        });
    std::size_t matches = 0U;
    for (auto current = first_match;
         current != torus_junction_order_.end() &&
         key(*current) == std::pair{packed_a, packed_b};
         ++current)
      ++matches;
    if (matches == 1U)
      return PatchRelation::point_only;
    if (matches > 1U)
      return PatchRelation::uncertified;
    return bounds_separated(left, right) ? PatchRelation::separated
                                         : PatchRelation::uncertified;
  }

  if (left.kind() == PatchKind::toroidal &&
      right.kind() == PatchKind::reentrant) {
    const std::size_t edge_begin = right.index() * 3U;
    for (std::size_t edge = 0U; edge < 3U; ++edge) {
      const std::size_t record = edge_begin + edge;
      if (record < seams_.reentrant_edges.size() &&
          seams_.reentrant_edges[record].relation ==
              PatchRelation::curve_adjacent &&
          seams_.reentrant_edges[record].reentrant == right &&
          seams_.reentrant_edges[record].torus == left)
        return PatchRelation::curve_adjacent;
    }
    return bounds_separated(left, right) ? PatchRelation::separated
                                         : PatchRelation::uncertified;
  }

  if (left.kind() == PatchKind::reentrant &&
      right.kind() == PatchKind::reentrant) {
    const ReentrantData &a = reentrant_[left.index()];
    const ReentrantData &b = reentrant_[right.index()];
    const double center_distance =
        norm(subtract(a.probe_center, b.probe_center));
    const double scale_ = std::max(
        {1.0, probe_radius_, norm(a.probe_center), norm(b.probe_center)});
    const double tolerance =
        512.0 * std::numeric_limits<double>::epsilon() * scale_;
    if (!std::isfinite(center_distance))
      return PatchRelation::uncertified;
    if (center_distance <= tolerance) {
      std::array<std::uint32_t, 3> atoms_a = a.atoms;
      std::array<std::uint32_t, 3> atoms_b = b.atoms;
      std::sort(atoms_a.begin(), atoms_a.end());
      std::sort(atoms_b.begin(), atoms_b.end());
      if (atoms_a != atoms_b)
        return PatchRelation::uncertified;
      bool directions_compatible = true;
      for (std::size_t ai = 0U; ai < 3U; ++ai) {
        bool matched = false;
        for (std::size_t bi = 0U; bi < 3U; ++bi) {
          if (a.atoms[ai] != b.atoms[bi])
            continue;
          matched = true;
          const double direction_residual = norm(
              subtract(a.contact_directions[ai], b.contact_directions[bi]));
          directions_compatible =
              directions_compatible && std::isfinite(direction_residual) &&
              direction_residual <=
                  512.0 * std::numeric_limits<double>::epsilon();
          break;
        }
        directions_compatible = directions_compatible && matched;
      }
      // For coincident centres, equal atom triples and contact directions
      // generate the same deterministic spherical-domain constraints from
      // this already validated immutable complex/index pair.
      if (directions_compatible)
        return PatchRelation::same_surface;
      return PatchRelation::uncertified;
    }
    if (center_distance > 2.0 * probe_radius_ + tolerance)
      return PatchRelation::separated;
    // Tangency requires a bounded-domain membership proof at the tangent
    // point, and strict overlap may hide an unrepresented trimming curve.
    return PatchRelation::uncertified;
  }

  return PatchRelation::uncertified;
}

AnalyticPatchAdjacencyOracle
build_analytic_patch_adjacency_oracle(const PatchComplex &complex,
                                      const LocalPatchProjectionIndex &index) {
  const auto build_begin = std::chrono::steady_clock::now();
  AnalyticPatchAdjacencyOracle result;
  try {
    result.probe_radius_ = complex.probe_radius;
    result.seams_ = build_analytic_patch_seam_certificate(
        complex, SeamCertificateStorage::retain_relations);
    const std::size_t atom_count = complex.atoms.size();
    std::size_t total_patches = complex.contacts.size();
    const bool total_patches_valid =
        checked_add(total_patches, complex.tori.size()) &&
        checked_add(total_patches, complex.reentrant.size()) &&
        total_patches <= std::numeric_limits<std::uint32_t>::max();
    bool structure_valid =
        complex.status == "pass" && atom_count > 0U &&
        atom_count <= std::numeric_limits<std::uint32_t>::max() &&
        std::isfinite(complex.probe_radius) && complex.probe_radius > 0.0 &&
        complex.contacts.size() == atom_count &&
        complex.contacts.size() <= PatchRefId::index_mask + 1ULL &&
        complex.tori.size() <= PatchRefId::index_mask + 1ULL &&
        complex.reentrant.size() <= PatchRefId::index_mask + 1ULL &&
        total_patches_valid;
    for (const Atom &atom : complex.atoms) {
      structure_valid = structure_valid && std::isfinite(atom.x) &&
                        std::isfinite(atom.y) && std::isfinite(atom.z) &&
                        std::isfinite(atom.radius) && atom.radius > 0.0;
    }

    bool atom_csr_valid =
        atom_count != std::numeric_limits<std::size_t>::max() &&
        index.atom_offsets.size() == atom_count + 1U &&
        !index.atom_offsets.empty() && index.atom_offsets.front() == 0U &&
        index.atom_offsets.back() == index.patches.size();
    for (std::size_t atom = 0U; atom_csr_valid && atom < atom_count; ++atom) {
      atom_csr_valid =
          index.atom_offsets[atom] <= index.atom_offsets[atom + 1U] &&
          index.atom_offsets[atom + 1U] <= index.patches.size();
    }

    std::vector<std::uint8_t> contact_incidence(complex.contacts.size(), 0U);
    std::vector<std::uint8_t> torus_incidence(complex.tori.size(), 0U);
    std::vector<std::uint8_t> reentrant_incidence(complex.reentrant.size(), 0U);
    for (std::size_t atom = 0U; atom_csr_valid && atom < atom_count; ++atom) {
      for (std::size_t position = index.atom_offsets[atom];
           position < index.atom_offsets[atom + 1U]; ++position) {
        const LocalPatchRef ref = index.patches[position];
        PatchRefId packed;
        if (!patch_ref_in_range(complex, ref) ||
            !PatchRefId::try_make(ref, packed) || !packed.valid()) {
          atom_csr_valid = false;
          break;
        }
        switch (ref.kind) {
        case PatchKind::contact:
          if (complex.contacts[ref.patch_index].atom != atom ||
              contact_incidence[ref.patch_index] ==
                  std::numeric_limits<std::uint8_t>::max()) {
            atom_csr_valid = false;
          } else {
            ++contact_incidence[ref.patch_index];
          }
          break;
        case PatchKind::toroidal:
          if ((complex.tori[ref.patch_index].atom_i != atom &&
               complex.tori[ref.patch_index].atom_j != atom) ||
              torus_incidence[ref.patch_index] ==
                  std::numeric_limits<std::uint8_t>::max()) {
            atom_csr_valid = false;
          } else {
            ++torus_incidence[ref.patch_index];
          }
          break;
        case PatchKind::reentrant:
          if (std::find(complex.reentrant[ref.patch_index].atoms.begin(),
                        complex.reentrant[ref.patch_index].atoms.end(), atom) ==
                  complex.reentrant[ref.patch_index].atoms.end() ||
              reentrant_incidence[ref.patch_index] ==
                  std::numeric_limits<std::uint8_t>::max()) {
            atom_csr_valid = false;
          } else {
            ++reentrant_incidence[ref.patch_index];
          }
          break;
        }
        if (!atom_csr_valid)
          break;
      }
    }
    atom_csr_valid =
        atom_csr_valid &&
        std::all_of(contact_incidence.begin(), contact_incidence.end(),
                    [](std::uint8_t count) { return count == 1U; }) &&
        std::all_of(torus_incidence.begin(), torus_incidence.end(),
                    [](std::uint8_t count) { return count == 2U; }) &&
        std::all_of(reentrant_incidence.begin(), reentrant_incidence.end(),
                    [](std::uint8_t count) { return count == 3U; });
    const PatchComplexFingerprint index_complex_fingerprint =
        fingerprint_patch_complex(complex);
    const bool index_structure_valid =
        index.source_complex == &complex && index_complex_fingerprint.valid &&
        index.source_complex_fingerprint.valid &&
        index.source_complex_fingerprint == index_complex_fingerprint &&
        index.bvh_invariants_certified &&
        index.bounded_domains_certified && atom_csr_valid &&
        validate_constraint_index(complex, index) &&
        validate_spatial_index(complex, index, total_patches);
    structure_valid = structure_valid && index_structure_valid;

    result.contacts_.reserve(complex.contacts.size());
    for (std::size_t patch_index = 0U; patch_index < complex.contacts.size();
         ++patch_index) {
      const ContactPatch &patch = complex.contacts[patch_index];
      structure_valid = structure_valid && patch.atom == patch_index &&
                        patch.atom < atom_count && finite_vec(patch.center) &&
                        std::isfinite(patch.atom_radius) &&
                        patch.atom_radius > 0.0 &&
                        std::isfinite(patch.expanded_radius) &&
                        patch.expanded_radius > patch.atom_radius;
      const PatchBounds bounds =
          outward_padded_patch_bounds(complex, PatchKind::contact, patch_index);
      structure_valid = structure_valid && bounds_finite_ordered(bounds);
      result.contacts_.push_back(
          {patch.atom <= std::numeric_limits<std::uint32_t>::max()
               ? static_cast<std::uint32_t>(patch.atom)
               : 0U,
           patch.center,
           patch.atom_radius,
           {bounds.lower, bounds.upper}});
    }

    result.torus_atoms_.reserve(complex.tori.size());
    for (std::size_t patch_index = 0U; patch_index < complex.tori.size();
         ++patch_index) {
      const ToroidalPatch &patch = complex.tori[patch_index];
      const PatchBounds bounds = outward_padded_patch_bounds(
          complex, PatchKind::toroidal, patch_index);
      structure_valid = structure_valid && patch.atom_i < atom_count &&
                        patch.atom_j < atom_count &&
                        patch.atom_i != patch.atom_j &&
                        bounds_finite_ordered(bounds);
      result.torus_atoms_.push_back(
          {patch.atom_i <= std::numeric_limits<std::uint32_t>::max()
               ? static_cast<std::uint32_t>(patch.atom_i)
               : 0U,
           patch.atom_j <= std::numeric_limits<std::uint32_t>::max()
               ? static_cast<std::uint32_t>(patch.atom_j)
               : 0U,
           {bounds.lower, bounds.upper}});
    }

    result.reentrant_.reserve(complex.reentrant.size());
    for (std::size_t patch_index = 0U; patch_index < complex.reentrant.size();
         ++patch_index) {
      const ReentrantPatch &patch = complex.reentrant[patch_index];
      bool patch_valid = finite_vec(patch.probe_center);
      std::array<std::uint32_t, 3> atoms{};
      for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
        const double direction_squared = dot(patch.contact_directions[vertex],
                                             patch.contact_directions[vertex]);
        patch_valid = patch_valid && patch.atoms[vertex] < atom_count &&
                      finite_vec(patch.contact_directions[vertex]) &&
                      std::isfinite(direction_squared) &&
                      std::abs(direction_squared - 1.0) <= 2.0e-10;
        atoms[vertex] = patch.atoms[vertex] < atom_count &&
                                patch.atoms[vertex] <=
                                    std::numeric_limits<std::uint32_t>::max()
                            ? static_cast<std::uint32_t>(patch.atoms[vertex])
                            : 0U;
      }
      patch_valid = patch_valid && atoms[0] != atoms[1] &&
                    atoms[0] != atoms[2] && atoms[1] != atoms[2];
      structure_valid = structure_valid && patch_valid;
      const PatchBounds bounds = outward_padded_patch_bounds(
          complex, PatchKind::reentrant, patch_index);
      structure_valid = structure_valid && bounds_finite_ordered(bounds);
      result.reentrant_.push_back({atoms,
                                   patch.probe_center,
                                   patch.contact_directions,
                                   {bounds.lower, bounds.upper}});
    }

    std::size_t reentrant_edge_count = 0U;
    const bool reentrant_edge_count_valid =
        checked_multiply(complex.reentrant.size(), 3U, reentrant_edge_count);
    const bool seam_structure_valid =
        reentrant_edge_count_valid &&
        result.seams_.source_complex == &complex &&
        result.seams_.structure_certified &&
        result.seams_.torus_contact_curve_certified.size() ==
            complex.tori.size() &&
        result.seams_.reentrant_edges.size() == reentrant_edge_count;
    structure_valid = structure_valid && seam_structure_valid;

    if (!structure_valid || !result.seams_.all_seams_certified) {
      result.certified_ = false;
      result.status_ =
          !structure_valid
              ? (!index_structure_valid ? "invalid_index" : "invalid_structure")
              : result.seams_.status;
      result.build_ms_ = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - build_begin)
                             .count();
      return result;
    }

    result.junctions_.reserve(reentrant_edge_count);
    for (std::size_t reentrant_index = 0U;
         reentrant_index < complex.reentrant.size(); ++reentrant_index) {
      PatchRefId reentrant_ref;
      if (!PatchRefId::try_make(PatchKind::reentrant, reentrant_index,
                                reentrant_ref) ||
          !reentrant_ref.valid()) {
        structure_valid = false;
        break;
      }
      const AnalyticPatchAdjacencyOracle::ReentrantData &reentrant =
          result.reentrant_[reentrant_index];
      for (std::size_t vertex = 0U; vertex < 3U; ++vertex) {
        AnalyticPatchAdjacencyOracle::TripleJunction junction;
        junction.reentrant = reentrant_ref;
        const std::uint32_t atom = reentrant.atoms[vertex];
        bool certified =
            atom < result.contacts_.size() &&
            PatchRefId::try_make(PatchKind::contact, atom, junction.contact) &&
            junction.contact.valid();
        const std::size_t edge_a = vertex;
        const std::size_t edge_b = (vertex + 2U) % 3U;
        const std::size_t record_a = reentrant_index * 3U + edge_a;
        const std::size_t record_b = reentrant_index * 3U + edge_b;
        certified = certified &&
                    record_a < result.seams_.reentrant_edges.size() &&
                    record_b < result.seams_.reentrant_edges.size();
        if (certified) {
          const ReentrantEdgeSeamCertificate &first_edge =
              result.seams_.reentrant_edges[record_a];
          const ReentrantEdgeSeamCertificate &second_edge =
              result.seams_.reentrant_edges[record_b];
          junction.torus_first = first_edge.torus;
          junction.torus_second = second_edge.torus;
          certified =
              first_edge.reentrant == reentrant_ref &&
              second_edge.reentrant == reentrant_ref &&
              first_edge.edge == edge_a && second_edge.edge == edge_b &&
              first_edge.relation == PatchRelation::curve_adjacent &&
              second_edge.relation == PatchRelation::curve_adjacent &&
              junction.torus_first.valid() && junction.torus_second.valid() &&
              junction.torus_first.kind() == PatchKind::toroidal &&
              junction.torus_second.kind() == PatchKind::toroidal &&
              junction.torus_first.index() < result.torus_atoms_.size() &&
              junction.torus_second.index() < result.torus_atoms_.size() &&
              junction.torus_first != junction.torus_second;
        }
        if (certified) {
          const auto side_certified = [&](PatchRefId torus_ref) {
            const std::size_t torus_index = torus_ref.index();
            if (torus_index >= result.torus_atoms_.size() ||
                torus_index >=
                    result.seams_.torus_contact_curve_certified.size())
              return false;
            const AnalyticPatchAdjacencyOracle::TorusAtoms &torus =
                result.torus_atoms_[torus_index];
            if (atom == torus.atom_i)
              return result.seams_
                         .torus_contact_curve_certified[torus_index][0] != 0U;
            if (atom == torus.atom_j)
              return result.seams_
                         .torus_contact_curve_certified[torus_index][1] != 0U;
            return false;
          };
          certified = side_certified(junction.torus_first) &&
                      side_certified(junction.torus_second);
        }
        junction.point = add(
            reentrant.probe_center,
            scale(complex.probe_radius, reentrant.contact_directions[vertex]));
        if (certified) {
          const AnalyticPatchAdjacencyOracle::ContactData &contact =
              result.contacts_[atom];
          const double residual = std::abs(
              norm(subtract(junction.point, contact.center)) - contact.radius);
          const double residual_scale =
              std::max({1.0, contact.radius, complex.probe_radius,
                        norm(junction.point), norm(contact.center)});
          certified = finite_vec(junction.point) && std::isfinite(residual) &&
                      residual <= 512.0 *
                                      std::numeric_limits<double>::epsilon() *
                                      residual_scale;
          if (certified) {
            const double tolerance = 1024.0 *
                                     std::numeric_limits<double>::epsilon() *
                                     residual_scale;
            const LocalProjectionHit bounded_contact =
                project_patch_bounded_closest(complex, index,
                                              PatchKind::contact, atom,
                                              junction.point, tolerance);
            certified = bounded_contact.found &&
                        bounded_contact.bounded_closest_certified &&
                        bounded_contact.normal_certified &&
                        bounded_contact.kind == PatchKind::contact &&
                        bounded_contact.patch_index == atom &&
                        std::isfinite(bounded_contact.distance) &&
                        bounded_contact.distance <= tolerance &&
                        finite_vec(bounded_contact.point) &&
                        norm(subtract(bounded_contact.point, junction.point)) <=
                            tolerance;
          }
        }
        junction.certified = certified ? 1U : 0U;
        result.junctions_.push_back(junction);
      }
    }
    for (std::size_t index_ = 0U; index_ < result.junctions_.size(); ++index_) {
      if (result.junctions_[index_].certified == 0U ||
          index_ > std::numeric_limits<std::uint32_t>::max())
        continue;
      result.torus_junction_order_.push_back(
          static_cast<std::uint32_t>(index_));
    }
    std::sort(
        result.torus_junction_order_.begin(),
        result.torus_junction_order_.end(),
        [&](std::uint32_t left, std::uint32_t right) {
          const AnalyticPatchAdjacencyOracle::TripleJunction &a =
              result.junctions_[left];
          const AnalyticPatchAdjacencyOracle::TripleJunction &b =
              result.junctions_[right];
          return std::tuple{
                     std::min(a.torus_first.packed(), a.torus_second.packed()),
                     std::max(a.torus_first.packed(), a.torus_second.packed()),
                     left} <
                 std::tuple{
                     std::min(b.torus_first.packed(), b.torus_second.packed()),
                     std::max(b.torus_first.packed(), b.torus_second.packed()),
                     right};
        });

    result.certified_ =
        structure_valid && result.junctions_.size() == reentrant_edge_count;
    result.status_ = result.certified_ ? "pass" : "invalid_structure";
    result.build_ms_ = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - build_begin)
                           .count();
    return result;
  } catch (const std::bad_alloc &) {
    AnalyticPatchAdjacencyOracle failed;
    failed.status_ = "alloc_fail";
    failed.build_ms_ = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - build_begin)
                           .count();
    return failed;
  } catch (const std::length_error &) {
    AnalyticPatchAdjacencyOracle failed;
    failed.status_ = "length_err";
    failed.build_ms_ = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - build_begin)
                           .count();
    return failed;
  }
}

PatchRayProvenanceAudit
audit_patch_ray_provenance(const PatchRayGrid &grid,
                           const AnalyticPatchAdjacencyOracle &oracle) {
  const auto begin = std::chrono::steady_clock::now();
  PatchRayProvenanceAudit result;
  result.oracle_build_ms = oracle.build_ms();
  result.oracle_retained_bytes = oracle.retained_payload_bytes();
  const auto stamp_time = [&]() noexcept {
    result.audit_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  };
  try {
    bool valid = grid.provenance.enabled &&
                 std::isfinite(grid.provenance.coordinate_merge_tolerance) &&
                 grid.provenance.coordinate_merge_tolerance >= 0.0 &&
                 std::isfinite(grid.provenance.near_pair_tolerance) &&
                 grid.provenance.near_pair_tolerance >= 0.0;
    std::size_t recomputed_crossings = 0U;
    std::size_t recomputed_seam_groups = 0U;
    std::size_t recomputed_merged_groups = 0U;
    for (std::size_t axis = 0U; axis < grid.axes.size(); ++axis) {
      const AxisRayCrossings &crossings = grid.axes[axis];
      const AxisPatchProvenance &axis_provenance = grid.provenance.axes[axis];
      const std::size_t row_count = crossings.row_count();
      valid = valid && row_count != std::numeric_limits<std::size_t>::max() &&
              crossings.offsets.size() == row_count + 1U &&
              (!crossings.offsets.empty() && crossings.offsets.front() == 0U) &&
              (!crossings.offsets.empty() &&
               crossings.offsets.back() == crossings.coordinates.size()) &&
              axis_provenance.validate(crossings.coordinates.size()) &&
              checked_add(recomputed_crossings, crossings.coordinates.size()) &&
              checked_add(recomputed_seam_groups,
                          axis_provenance.seam_groups.size()) &&
              checked_add(recomputed_merged_groups,
                          axis_provenance.single_ref_multiplicities.size());
      if (valid) {
        for (const SparsePatchSeamGroup &group : axis_provenance.seam_groups) {
          bool repeated = false;
          for (std::size_t position = group.refs_begin;
               position < group.refs_end; ++position) {
            repeated = repeated ||
                       axis_provenance.seam_ref_multiplicities[position] > 1U;
          }
          if (repeated && !checked_add(recomputed_merged_groups, 1U)) {
            valid = false;
            break;
          }
        }
      }
      for (std::size_t row = 0U; valid && row + 1U < crossings.offsets.size();
           ++row) {
        valid = crossings.offsets[row] <= crossings.offsets[row + 1U] &&
                crossings.offsets[row + 1U] <= crossings.coordinates.size();
        if (!valid)
          break;
        const std::size_t row_begin = crossings.offsets[row];
        const std::size_t row_end = crossings.offsets[row + 1U];
        for (std::size_t crossing = row_begin; crossing < row_end; ++crossing) {
          const double coordinate = crossings.coordinates[crossing];
          if (!std::isfinite(coordinate) ||
              (crossing > row_begin &&
               coordinate <= crossings.coordinates[crossing - 1U])) {
            valid = false;
            break;
          }
        }
      }
    }
    valid = valid && recomputed_crossings == grid.provenance.unique_crossings &&
            recomputed_seam_groups == grid.provenance.seam_groups &&
            recomputed_merged_groups == grid.provenance.repeated_root_groups &&
            recomputed_merged_groups ==
                grid.provenance.same_primitive_near_pair_unclassified;
    if (!valid) {
      result.status = "invalid_provenance";
      stamp_time();
      return result;
    }

    bool counters_valid = true;
    result.same_primitive_merged_groups = recomputed_merged_groups;
    result.same_primitive_near_pair_unclassified =
        result.same_primitive_merged_groups;
    for (std::size_t axis = 0U; axis < grid.axes.size(); ++axis) {
      const AxisPatchProvenance &provenance = grid.provenance.axes[axis];
      for (const SparsePatchSeamGroup &group : provenance.seam_groups) {
        counters_valid =
            counters_valid && checked_add(result.seam_groups_audited, 1U);
        const std::span<const PatchRefId> refs =
            provenance.refs(group.crossing_index);
        bool curve = false;
        bool point = false;
        bool incompatible = false;
        bool uncertified = false;
        for (std::size_t first = 0U; first < refs.size(); ++first) {
          for (std::size_t second = first + 1U; second < refs.size();
               ++second) {
            const PatchRelation relation =
                oracle.relation(refs[first], refs[second]);
            counters_valid =
                counters_valid &&
                checked_add(result.relation_counts[relation_index(relation)],
                            1U);
            curve = curve || relation == PatchRelation::curve_adjacent;
            point = point || relation == PatchRelation::point_only;
            incompatible = incompatible || relation == PatchRelation::separated;
            uncertified = uncertified || relation == PatchRelation::uncertified;
          }
        }
        if (incompatible)
          counters_valid = counters_valid &&
                           checked_add(result.incompatible_crossing_groups, 1U);
        if (uncertified)
          counters_valid = counters_valid &&
                           checked_add(result.uncertified_crossing_groups, 1U);
        if (!incompatible && !uncertified && point)
          counters_valid =
              counters_valid && checked_add(result.point_only_groups, 1U);
        else if (!incompatible && !uncertified && curve)
          counters_valid =
              counters_valid && checked_add(result.curve_adjacent_groups, 1U);
      }

      const AxisRayCrossings &crossings = grid.axes[axis];
      for (std::size_t row = 0U; row < crossings.row_count(); ++row) {
        const std::size_t row_begin = crossings.offsets[row];
        const std::size_t row_end = crossings.offsets[row + 1U];
        for (std::size_t crossing = row_begin; crossing + 1U < row_end;
             ++crossing) {
          const double separation = crossings.coordinates[crossing + 1U] -
                                    crossings.coordinates[crossing];
          if (!std::isfinite(separation) || separation < 0.0) {
            result.status = "invalid_provenance";
            stamp_time();
            return result;
          }
          if (separation > grid.provenance.near_pair_tolerance)
            continue;
          counters_valid =
              counters_valid && checked_add(result.adjacent_near_pairs, 1U);
          const auto first_refs = provenance.refs(crossing);
          const auto second_refs = provenance.refs(crossing + 1U);
          PatchRelation aggregate = PatchRelation::same_surface;
          bool same_primitive = false;
          bool separated = false;
          bool uncertified = false;
          for (const PatchRefId first_ref : first_refs) {
            for (const PatchRefId second_ref : second_refs) {
              same_primitive = same_primitive || first_ref == second_ref;
              const PatchRelation pair_relation =
                  oracle.relation(first_ref, second_ref);
              separated =
                  separated || pair_relation == PatchRelation::separated;
              uncertified =
                  uncertified || pair_relation == PatchRelation::uncertified;
              aggregate = worse_relation(aggregate, pair_relation);
            }
          }
          counters_valid =
              counters_valid &&
              checked_add(
                  result.near_pair_relation_counts[relation_index(aggregate)],
                  1U);
          if (separated)
            counters_valid = counters_valid &&
                             checked_add(result.incompatible_near_pairs, 1U);
          if (uncertified)
            counters_valid = counters_valid &&
                             checked_add(result.uncertified_near_pairs, 1U);
          if (same_primitive)
            counters_valid =
                counters_valid &&
                checked_add(result.same_primitive_adjacent_near_pairs, 1U) &&
                checked_add(result.same_primitive_near_pair_unclassified, 1U);
        }
      }
    }
    if (!counters_valid) {
      result.status = "counter_overflow";
      stamp_time();
      return result;
    }
    if (result.seam_groups_audited != grid.provenance.seam_groups) {
      result.status = "invalid_provenance";
      stamp_time();
      return result;
    }

    result.relation_audit_complete = grid.provenance.certified &&
                                     oracle.certified() &&
                                     result.uncertified_crossing_groups == 0U &&
                                     result.uncertified_near_pairs == 0U;
    // Coordinate merging retains every patch identity but not each identity's
    // raw root displacement or derivative. Therefore even curve-adjacent and
    // point-only relations remain location-uncertified in Phase 2A.
    result.crossing_location_certified = false;
    if (!grid.provenance.certified) {
      result.status = grid.provenance.status;
    } else if (!oracle.certified()) {
      result.status = std::string(oracle.status());
    } else if (result.incompatible_crossing_groups != 0U) {
      result.status = "incompatible_crossing";
    } else if (result.incompatible_near_pairs != 0U) {
      result.status = "incompatible_near_pair";
    } else if (result.uncertified_crossing_groups != 0U ||
               result.uncertified_near_pairs != 0U) {
      result.status = "uncertified_relation";
    } else if (result.same_primitive_near_pair_unclassified != 0U) {
      result.status = "same_primitive_unclassified";
    } else if (result.seam_groups_audited != 0U ||
               result.adjacent_near_pairs != 0U) {
      result.status = "crossing_location_uncertified";
    } else {
      result.status = "relations_complete_no_location_claim";
    }
    stamp_time();
    return result;
  } catch (const std::bad_alloc &) {
    result.status = "alloc_fail";
    stamp_time();
    return result;
  } catch (const std::length_error &) {
    result.status = "length_err";
    stamp_time();
    return result;
  }
}

} // namespace TitanPBE::Tabi::analytic_ses

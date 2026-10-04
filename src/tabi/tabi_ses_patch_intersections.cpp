#include "tabi_ses_patch_intersections.h"

#include "tabi_ses_patch_provenance.h"
#include "tabi_ses_quartic.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <tuple>
#include <utility>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

constexpr double kGeometryRelTol = 2.0e-12;
constexpr double kActivePlaneCrossSquaredThreshold = 1.0e-14;
constexpr double kSphereConstraintTolerance = 4.0e-10;

[[nodiscard]] Vec3 add(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 subtract(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 scale(double value, const Vec3 &a) noexcept {
  return {value * a.x, value * a.y, value * a.z};
}

[[nodiscard]] double dot(const Vec3 &a, const Vec3 &b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] double norm_squared(const Vec3 &a) noexcept { return dot(a, a); }

[[nodiscard]] double norm(const Vec3 &a) noexcept {
  return std::sqrt(norm_squared(a));
}

[[nodiscard]] Vec3 normalized(const Vec3 &a) noexcept {
  const double length = norm(a);
  return length > 1.0e-300 ? scale(1.0 / length, a) : Vec3{};
}

[[nodiscard]] Vec3 atom_center(const Atom &atom) noexcept {
  return {atom.x, atom.y, atom.z};
}

[[nodiscard]] Vec3 direction(RayAxis axis) noexcept {
  switch (axis) {
  case RayAxis::x:
    return {1.0, 0.0, 0.0};
  case RayAxis::y:
    return {0.0, 1.0, 0.0};
  case RayAxis::z:
    return {0.0, 0.0, 1.0};
  }
  return {1.0, 0.0, 0.0};
}

[[nodiscard]] double component(const Vec3 &point, RayAxis axis) noexcept {
  switch (axis) {
  case RayAxis::x:
    return point.x;
  case RayAxis::y:
    return point.y;
  case RayAxis::z:
    return point.z;
  }
  return point.x;
}

[[nodiscard]] Vec3 line_point(RayAxis axis, double coordinate, double u,
                              double v) noexcept {
  switch (axis) {
  case RayAxis::x:
    return {coordinate, u, v};
  case RayAxis::y:
    return {u, coordinate, v};
  case RayAxis::z:
    return {u, v, coordinate};
  }
  return {coordinate, u, v};
}

[[nodiscard]] double normalize_angle(double angle) noexcept {
  const double period = 2.0 * std::numbers::pi;
  angle = std::fmod(angle, period);
  return angle < 0.0 ? angle + period : angle;
}

[[nodiscard]] bool angle_in_exposed_intervals(
    double angle, const std::vector<AngularInterval> &intervals) noexcept {
  angle = normalize_angle(angle);
  for (const AngularInterval &interval : intervals) {
    if (angle >= interval.begin - kAnalyticSesAngleTolerance &&
        angle <= interval.end + kAnalyticSesAngleTolerance)
      return true;
  }
  return false;
}

[[nodiscard]] bool make_directed_minor_arc(double endpoint_a, double endpoint_b,
                                           TorusBetaMinorArc &arc) noexcept {
  arc = {};
  if (!std::isfinite(endpoint_a) || !std::isfinite(endpoint_b))
    return false;
  const double period = 2.0 * std::numbers::pi;
  endpoint_a = normalize_angle(endpoint_a);
  endpoint_b = normalize_angle(endpoint_b);
  const auto ccw_distance = [&](double from, double to) {
    double delta = to - from;
    if (delta < 0.0)
      delta += period;
    return delta;
  };
  const double forward_span = ccw_distance(endpoint_a, endpoint_b);
  arc.start = forward_span <= std::numbers::pi ? endpoint_a : endpoint_b;
  arc.span =
      forward_span <= std::numbers::pi ? forward_span : period - forward_span;
  return std::isfinite(arc.start) && std::isfinite(arc.span) &&
         arc.start >= 0.0 && arc.start < period && arc.span >= 0.0 &&
         arc.span <= std::numbers::pi;
}

[[nodiscard]] bool
angle_on_directed_minor_arc(double angle,
                            const TorusBetaMinorArc &arc) noexcept {
  const double period = 2.0 * std::numbers::pi;
  if (!std::isfinite(angle) || !std::isfinite(arc.start) ||
      !std::isfinite(arc.span) || arc.start < 0.0 || arc.start >= period ||
      arc.span < 0.0 || arc.span > std::numbers::pi)
    return false;
  angle = normalize_angle(angle);
  double distance = angle - arc.start;
  if (distance < 0.0)
    distance += period;
  return distance <= arc.span + kAnalyticSesAngleTolerance;
}

[[nodiscard]] bool angle_on_minor_arc(double angle, double endpoint_a,
                                      double endpoint_b) noexcept {
  const double period = 2.0 * std::numbers::pi;
  angle = normalize_angle(angle);
  endpoint_a = normalize_angle(endpoint_a);
  endpoint_b = normalize_angle(endpoint_b);
  const auto ccw_distance = [&](double from, double to) {
    double delta = to - from;
    if (delta < 0.0)
      delta += period;
    return delta;
  };
  const double forward_span = ccw_distance(endpoint_a, endpoint_b);
  const bool on_forward = ccw_distance(endpoint_a, angle) <=
                          forward_span + kAnalyticSesAngleTolerance;
  return forward_span <= std::numbers::pi ? on_forward : !on_forward;
}

[[nodiscard]] bool
spherical_triangle_contains(const ReentrantPatch &patch,
                            const Vec3 &direction_) noexcept {
  const Vec3 anchor = normalized(
      add(add(patch.contact_directions[0], patch.contact_directions[1]),
          patch.contact_directions[2]));
  if (norm_squared(anchor) <= 1.0e-20)
    return false;
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const Vec3 &a = patch.contact_directions[edge];
    const Vec3 &b = patch.contact_directions[(edge + 1U) % 3U];
    const Vec3 plane = cross(a, b);
    const double reference = dot(plane, anchor);
    const double value = dot(plane, direction_);
    if (reference >= 0.0 ? value < -2.0e-10 : value > 2.0e-10)
      return false;
  }
  return true;
}

[[nodiscard]] bool contact_point_is_exposed(const PatchComplex &complex,
                                            const ContactPatch &patch,
                                            const Vec3 &point) noexcept {
  const Vec3 radial = normalized(subtract(point, patch.center));
  const Vec3 probe_center =
      add(patch.center, scale(patch.expanded_radius, radial));
  for (const std::size_t atom_index : complex.atom_neighbors[patch.atom]) {
    const double radius =
        complex.atoms[atom_index].radius + complex.probe_radius;
    const double tolerance = kGeometryRelTol * std::max(1.0, radius * radius);
    if (norm_squared(
            subtract(probe_center, atom_center(complex.atoms[atom_index]))) <
        radius * radius - tolerance)
      return false;
  }
  return true;
}

struct CellKeyIndex {
  std::int32_t x{0};
  std::int32_t y{0};
  std::int32_t z{0};
  std::uint32_t index{0};
};

[[nodiscard]] auto cell_tuple(std::int32_t x, std::int32_t y,
                              std::int32_t z) noexcept {
  return std::tuple{x, y, z};
}

[[nodiscard]] std::array<std::int32_t, 3>
probe_cell(const ReentrantProbeIndex &index, const Vec3 &point) noexcept {
  return {static_cast<std::int32_t>(
              std::floor((point.x - index.origin.x) / index.cell_size)),
          static_cast<std::int32_t>(
              std::floor((point.y - index.origin.y) / index.cell_size)),
          static_cast<std::int32_t>(
              std::floor((point.z - index.origin.z) / index.cell_size))};
}

[[nodiscard]] const ProbeCell *find_cell(const ReentrantProbeIndex &index,
                                         std::int32_t x, std::int32_t y,
                                         std::int32_t z) noexcept {
  const auto key = cell_tuple(x, y, z);
  const auto found =
      std::lower_bound(index.cells.begin(), index.cells.end(), key,
                       [](const ProbeCell &cell, const auto &value) {
                         return cell_tuple(cell.x, cell.y, cell.z) < value;
                       });
  if (found == index.cells.end() ||
      cell_tuple(found->x, found->y, found->z) != key)
    return nullptr;
  return &*found;
}

[[nodiscard]] bool covered_by_reentrant_probe(const PatchComplex &complex,
                                              const ReentrantProbeIndex &index,
                                              const Vec3 &point,
                                              std::size_t ignore) noexcept {
  if (index.cells.empty())
    return false;
  const auto cell = probe_cell(index, point);
  const double radius2 = complex.probe_radius * complex.probe_radius;
  const double tolerance = 2.0e-11 * std::max(1.0, radius2);
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const ProbeCell *range =
            find_cell(index, cell[0] + dx, cell[1] + dy, cell[2] + dz);
        if (range == nullptr)
          continue;
        for (std::uint32_t position = range->begin; position < range->end;
             ++position) {
          const std::size_t probe_index = index.probe_indices[position];
          if (probe_index == ignore)
            continue;
          if (norm_squared(subtract(
                  point, complex.reentrant[probe_index].probe_center)) <
              radius2 - tolerance)
            return true;
        }
      }
    }
  }
  return false;
}

void append_coordinate(PatchAxisRoots &roots, double coordinate,
                       bool primitive_simplicity_certified) noexcept {
  if (!std::isfinite(coordinate) || roots.count >= roots.coordinates.size())
    return;
  roots.coordinates[roots.count] = coordinate;
  roots.primitive_simplicity_certified[roots.count] =
      primitive_simplicity_certified ? 1U : 0U;
  ++roots.count;
}

void sort_coordinates(PatchAxisRoots &roots) noexcept {
  for (std::size_t index = 1U; index < roots.count; ++index) {
    const double coordinate = roots.coordinates[index];
    const std::uint8_t primitive_simplicity_certified =
        roots.primitive_simplicity_certified[index];
    std::size_t position = index;
    while (position > 0U && coordinate < roots.coordinates[position - 1U]) {
      roots.coordinates[position] = roots.coordinates[position - 1U];
      roots.primitive_simplicity_certified[position] =
          roots.primitive_simplicity_certified[position - 1U];
      --position;
    }
    roots.coordinates[position] = coordinate;
    roots.primitive_simplicity_certified[position] =
        primitive_simplicity_certified;
  }
}

[[nodiscard]] Vec3 patch_outward_normal(const PatchComplex &complex,
                                        PatchKind kind, std::size_t patch_index,
                                        const Vec3 &point) noexcept {
  if (kind == PatchKind::contact) {
    return normalized(subtract(point, complex.contacts[patch_index].center));
  }
  if (kind == PatchKind::reentrant) {
    return scale(-1.0,
                 normalized(subtract(
                     point, complex.reentrant[patch_index].probe_center)));
  }
  const ToroidalPatch &patch = complex.tori[patch_index];
  const Vec3 q = subtract(point, patch.circle_center);
  const double axial = dot(q, patch.axis);
  const Vec3 radial_vector = subtract(q, scale(axial, patch.axis));
  const double radial_length = norm(radial_vector);
  if (!(radial_length > 1.0e-12))
    return {};
  const Vec3 probe_center = add(
      patch.circle_center,
      scale(patch.circle_radius, scale(1.0 / radial_length, radial_vector)));
  return scale(-1.0, normalized(subtract(point, probe_center)));
}

[[nodiscard]] std::uint32_t
canonical_patch_id(const PatchComplex &complex, PatchKind kind,
                   std::size_t patch_index) noexcept {
  if (kind == PatchKind::contact)
    return static_cast<std::uint32_t>(patch_index);
  if (kind == PatchKind::toroidal)
    return static_cast<std::uint32_t>(complex.contacts.size() + patch_index);
  return static_cast<std::uint32_t>(complex.contacts.size() +
                                    complex.tori.size() + patch_index);
}

[[nodiscard]] std::size_t patch_atom_hint(const PatchComplex &complex,
                                          PatchKind kind,
                                          std::size_t patch_index) noexcept {
  if (kind == PatchKind::contact)
    return complex.contacts[patch_index].atom;
  if (kind == PatchKind::toroidal) {
    const ToroidalPatch &patch = complex.tori[patch_index];
    return std::min(patch.atom_i, patch.atom_j);
  }
  const auto &atoms = complex.reentrant[patch_index].atoms;
  return std::min({atoms[0], atoms[1], atoms[2]});
}

[[nodiscard]] PatchBounds merge_bounds(const PatchBounds &a,
                                       const PatchBounds &b) noexcept {
  return {{std::min(a.lower.x, b.lower.x), std::min(a.lower.y, b.lower.y),
           std::min(a.lower.z, b.lower.z)},
          {std::max(a.upper.x, b.upper.x), std::max(a.upper.y, b.upper.y),
           std::max(a.upper.z, b.upper.z)}};
}

[[nodiscard]] bool
bounds_are_finite_and_ordered(const PatchBounds &bounds) noexcept {
  return std::isfinite(bounds.lower.x) && std::isfinite(bounds.lower.y) &&
         std::isfinite(bounds.lower.z) && std::isfinite(bounds.upper.x) &&
         std::isfinite(bounds.upper.y) && std::isfinite(bounds.upper.z) &&
         bounds.lower.x <= bounds.upper.x && bounds.lower.y <= bounds.upper.y &&
         bounds.lower.z <= bounds.upper.z;
}

[[nodiscard]] bool bounds_contain(const PatchBounds &outer,
                                  const PatchBounds &inner) noexcept {
  return bounds_are_finite_and_ordered(outer) &&
         bounds_are_finite_and_ordered(inner) &&
         outer.lower.x <= inner.lower.x && outer.lower.y <= inner.lower.y &&
         outer.lower.z <= inner.lower.z && outer.upper.x >= inner.upper.x &&
         outer.upper.y >= inner.upper.y && outer.upper.z >= inner.upper.z;
}

[[nodiscard]] PatchBounds outward_padded_bounds(PatchBounds bounds) noexcept {
  const auto pad_axis = [](double &lower, double &upper) {
    const double extent = std::abs(upper - lower);
    const double coordinate_scale =
        std::max({1.0, std::abs(lower), std::abs(upper), extent});
    const double padding =
        32.0 * std::numeric_limits<double>::epsilon() * coordinate_scale;
    lower = std::nextafter(lower - padding,
                           -std::numeric_limits<double>::infinity());
    upper = std::nextafter(upper + padding,
                           std::numeric_limits<double>::infinity());
  };
  pad_axis(bounds.lower.x, bounds.upper.x);
  pad_axis(bounds.lower.y, bounds.upper.y);
  pad_axis(bounds.lower.z, bounds.upper.z);
  return bounds;
}

[[nodiscard]] double bounds_centroid_component(const PatchBounds &bounds,
                                               int axis) noexcept {
  if (axis == 0)
    return 0.5 * (bounds.lower.x + bounds.upper.x);
  if (axis == 1)
    return 0.5 * (bounds.lower.y + bounds.upper.y);
  return 0.5 * (bounds.lower.z + bounds.upper.z);
}

[[nodiscard]] double
point_bounds_distance_squared(const Vec3 &point,
                              const PatchBounds &bounds) noexcept {
  const auto outside = [](double value, double lower, double upper) {
    if (value < lower)
      return lower - value;
    if (value > upper)
      return value - upper;
    return 0.0;
  };
  const double dx = outside(point.x, bounds.lower.x, bounds.upper.x);
  const double dy = outside(point.y, bounds.lower.y, bounds.upper.y);
  const double dz = outside(point.z, bounds.lower.z, bounds.upper.z);
  return dx * dx + dy * dy + dz * dz;
}

[[nodiscard]] Vec3 deterministic_perpendicular(const Vec3 &axis) noexcept {
  const Vec3 seed =
      std::abs(axis.x) < 0.75 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
  return normalized(cross(axis, seed));
}

[[nodiscard]] bool point_lexicographic_less(const Vec3 &a,
                                            const Vec3 &b) noexcept {
  return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
}

struct ExactPatchProjection {
  Vec3 point;
  Vec3 normal;
  double distance{std::numeric_limits<double>::infinity()};
  bool found{false};
  bool domain_certified{true};
  bool domain_empty{false};
  bool normal_certified{false};
};

void retain_exact_candidate(ExactPatchProjection &best, const Vec3 &query,
                            const Vec3 &point, const Vec3 &normal,
                            bool normal_certified = true) noexcept {
  const double normal2 = norm_squared(normal);
  const double distance = norm(subtract(point, query));
  if (!std::isfinite(distance) || !(normal2 > 1.0e-20))
    return;
  const bool replace = !best.found || distance < best.distance ||
                       (distance == best.distance &&
                        point_lexicographic_less(point, best.point));
  if (!replace)
    return;
  best.point = point;
  best.normal = normalized(normal);
  best.distance = distance;
  best.found = true;
  best.normal_certified = normal_certified;
}

[[nodiscard]] bool finite_direction(const Vec3 &value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

[[nodiscard]] bool sphere_constraints_are_well_formed(
    std::span<const SpherePatchConstraint> constraints) noexcept {
  for (const SpherePatchConstraint &constraint : constraints) {
    if (!finite_direction(constraint.normal) ||
        !std::isfinite(constraint.minimum_dot) ||
        std::abs(norm_squared(constraint.normal) - 1.0) > 2.0e-10)
      return false;
  }
  return true;
}

[[nodiscard]] bool
sphere_direction_is_feasible(std::span<const SpherePatchConstraint> constraints,
                             const Vec3 &direction_) noexcept {
  if (!finite_direction(direction_) ||
      std::abs(norm_squared(direction_) - 1.0) > 2.0e-12)
    return false;
  for (const SpherePatchConstraint &constraint : constraints) {
    if (dot(constraint.normal, direction_) <
        constraint.minimum_dot - kSphereConstraintTolerance)
      return false;
  }
  return true;
}

// Deterministic slow oracle and index-builder primitive. Pair order is the
// canonical (first, second) index order and both algebraic intersections are
// retained; no geometric epsilon deduplication changes the candidate set.
template <typename Emit>
[[nodiscard]] bool enumerate_feasible_two_active_directions(
    std::span<const SpherePatchConstraint> constraints, Emit &&emit,
    PatchKind kind, std::size_t patch, ActivePairCacheFailure *failure) {
  const auto record_failure = [&](ActivePairCacheFailureReason reason,
                                  std::size_t first, std::size_t second,
                                  double cross_squared, double base_squared,
                                  double first_residual, double second_residual,
                                  const Vec3 *candidate) {
    if (failure == nullptr ||
        failure->reason != ActivePairCacheFailureReason::none)
      return;
    failure->reason = reason;
    failure->kind = kind;
    failure->patch = patch;
    failure->first = first;
    failure->second = second;
    if (first < constraints.size()) {
      failure->first_normal2 = norm_squared(constraints[first].normal);
      failure->first_minimum_dot = constraints[first].minimum_dot;
    }
    if (second < constraints.size()) {
      failure->second_normal2 = norm_squared(constraints[second].normal);
      failure->second_minimum_dot = constraints[second].minimum_dot;
    }
    failure->cross_squared = cross_squared;
    failure->base_squared = base_squared;
    failure->first_active_residual = first_residual;
    failure->second_active_residual = second_residual;
    if (candidate != nullptr && finite_direction(*candidate)) {
      double minimum_margin = std::numeric_limits<double>::infinity();
      for (const SpherePatchConstraint &constraint : constraints) {
        minimum_margin =
            std::min(minimum_margin, dot(constraint.normal, *candidate) -
                                         constraint.minimum_dot);
      }
      failure->minimum_feasibility_margin = minimum_margin;
      failure->candidate_feasible =
          minimum_margin >= -kSphereConstraintTolerance;
    }
  };
  for (std::size_t constraint = 0; constraint < constraints.size();
       ++constraint) {
    const SpherePatchConstraint &value = constraints[constraint];
    if (!finite_direction(value.normal) || !std::isfinite(value.minimum_dot) ||
        std::abs(norm_squared(value.normal) - 1.0) > 2.0e-10) {
      record_failure(ActivePairCacheFailureReason::malformed_constraint,
                     constraint, std::numeric_limits<std::size_t>::max(), 0.0,
                     0.0, 0.0, 0.0, nullptr);
      return false;
    }
  }
  if (!sphere_constraints_are_well_formed(constraints))
    return false;
  for (std::size_t first = 0; first < constraints.size(); ++first) {
    const Vec3 n = constraints[first].normal;
    const double b = constraints[first].minimum_dot;
    if (b < -1.0 - kSphereConstraintTolerance ||
        b > 1.0 + kSphereConstraintTolerance)
      continue;
    for (std::size_t second = first + 1U; second < constraints.size();
         ++second) {
      const Vec3 m = constraints[second].normal;
      const double c = constraints[second].minimum_dot;
      if (c < -1.0 - kSphereConstraintTolerance ||
          c > 1.0 + kSphereConstraintTolerance)
        continue;
      // The patch builder permits a small unit-normal residual.  Solve the
      // two-plane Gram system using the stored normals (rather than assuming
      // n.n == m.m == 1), and retain extra precision until the final Vec3.
      // This keeps the active-plane proof independent of that permitted
      // residual without relaxing the validation tolerance below.
      const long double nx = static_cast<long double>(n.x);
      const long double ny = static_cast<long double>(n.y);
      const long double nz = static_cast<long double>(n.z);
      const long double mx = static_cast<long double>(m.x);
      const long double my = static_cast<long double>(m.y);
      const long double mz = static_cast<long double>(m.z);
      const long double cross_x = ny * mz - nz * my;
      const long double cross_y = nz * mx - nx * mz;
      const long double cross_z = nx * my - ny * mx;
      const long double cross_squared_ld =
          cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
      const double cross_squared = static_cast<double>(cross_squared_ld);
      if (!std::isfinite(cross_squared_ld)) {
        record_failure(ActivePairCacheFailureReason::nonfinite_algebra, first,
                       second, cross_squared, 0.0, 0.0, 0.0, nullptr);
        return false;
      }
      if (cross_squared_ld == 0.0L)
        continue;
      if (cross_squared_ld <=
          static_cast<long double>(kActivePlaneCrossSquaredThreshold)) {
        record_failure(ActivePairCacheFailureReason::near_parallel, first,
                       second, cross_squared, 0.0, 0.0, 0.0, nullptr);
        return false;
      }
      const long double normal_n_squared = nx * nx + ny * ny + nz * nz;
      const long double normal_m_squared = mx * mx + my * my + mz * mz;
      const long double cosine = nx * mx + ny * my + nz * mz;
      const long double b_ld = static_cast<long double>(b);
      const long double c_ld = static_cast<long double>(c);
      const long double coefficient_n =
          (b_ld * normal_m_squared - cosine * c_ld) / cross_squared_ld;
      const long double coefficient_m =
          (c_ld * normal_n_squared - cosine * b_ld) / cross_squared_ld;
      if (!std::isfinite(coefficient_n) || !std::isfinite(coefficient_m)) {
        record_failure(ActivePairCacheFailureReason::nonfinite_algebra, first,
                       second, cross_squared, 0.0, 0.0, 0.0, nullptr);
        return false;
      }
      const long double base_x = coefficient_n * nx + coefficient_m * mx;
      const long double base_y = coefficient_n * ny + coefficient_m * my;
      const long double base_z = coefficient_n * nz + coefficient_m * mz;
      const long double base2_ld =
          base_x * base_x + base_y * base_y + base_z * base_z;
      const double base2 = static_cast<double>(base2_ld);
      if (!std::isfinite(base2_ld)) {
        record_failure(ActivePairCacheFailureReason::nonfinite_algebra, first,
                       second, cross_squared, base2, 0.0, 0.0, nullptr);
        return false;
      }
      // A robustly computed base outside the unit sphere means the two active
      // planes do not intersect the sphere.  It is not a near-boundary
      // candidate and must not be clamped into one.
      if (base2_ld > 1.0L)
        continue;
      const long double inverse_cross_norm = 1.0L / std::sqrt(cross_squared_ld);
      const long double line_x = inverse_cross_norm * cross_x;
      const long double line_y = inverse_cross_norm * cross_y;
      const long double line_z = inverse_cross_norm * cross_z;
      const long double height = std::sqrt(std::max(0.0L, 1.0L - base2_ld));
      const auto process = [&](long double sign) {
        const Vec3 raw_direction{
            static_cast<double>(base_x + sign * height * line_x),
            static_cast<double>(base_y + sign * height * line_y),
            static_cast<double>(base_z + sign * height * line_z)};
        const Vec3 direction_ = normalized(raw_direction);
        const double first_residual = std::abs(dot(n, direction_) - b);
        const double second_residual = std::abs(dot(m, direction_) - c);
        if (!finite_direction(direction_) ||
            std::abs(norm_squared(direction_) - 1.0) > 2.0e-12) {
          record_failure(ActivePairCacheFailureReason::nonfinite_algebra, first,
                         second, cross_squared, base2, first_residual,
                         second_residual, &direction_);
          return false;
        }
        if (first_residual > kSphereConstraintTolerance ||
            second_residual > kSphereConstraintTolerance) {
          record_failure(ActivePairCacheFailureReason::active_residual, first,
                         second, cross_squared, base2, first_residual,
                         second_residual, &direction_);
          return false;
        }
        if (!sphere_direction_is_feasible(constraints, direction_))
          return true;
        if (emit(direction_))
          return true;
        record_failure(ActivePairCacheFailureReason::storage, first, second,
                       cross_squared, base2, first_residual, second_residual,
                       &direction_);
        return false;
      };
      if (!process(1.0L))
        return false;
      if (height > 0.0L && !process(-1.0L))
        return false;
    }
  }
  return true;
}

[[nodiscard]] bool has_feasible_zero_or_one_active_direction(
    std::span<const SpherePatchConstraint> constraints) noexcept {
  const Vec3 objective = normalized(
      {0.42426406871192851, 0.56568542494923801, 0.70710678118654757});
  const auto feasible = [&](const Vec3 &raw_direction) {
    const Vec3 direction_ = normalized(raw_direction);
    return norm_squared(direction_) > 1.0e-20 &&
           sphere_direction_is_feasible(constraints, direction_);
  };
  if (feasible(objective))
    return true;
  for (const SpherePatchConstraint &constraint : constraints) {
    const Vec3 n = constraint.normal;
    const double b = constraint.minimum_dot;
    if (b < -1.0 - kSphereConstraintTolerance ||
        b > 1.0 + kSphereConstraintTolerance)
      continue;
    const double clamped = std::clamp(b, -1.0, 1.0);
    const double circle_radius =
        std::sqrt(std::max(0.0, 1.0 - clamped * clamped));
    const Vec3 perpendicular = subtract(objective, scale(dot(objective, n), n));
    if (norm_squared(perpendicular) > 1.0e-24) {
      if (feasible(add(scale(clamped, n),
                       scale(circle_radius, normalized(perpendicular)))))
        return true;
    } else {
      const Vec3 basis = deterministic_perpendicular(n);
      if (feasible(add(scale(clamped, n), scale(circle_radius, basis))) ||
          feasible(subtract(scale(clamped, n), scale(circle_radius, basis))))
        return true;
    }
  }
  return false;
}

[[nodiscard]] bool build_two_active_direction_cache(
    const std::vector<std::size_t> &constraint_offsets,
    const std::vector<SpherePatchConstraint> &constraints,
    std::size_t patch_count, PatchKind kind,
    std::vector<std::size_t> &vertex_offsets, std::vector<Vec3> &vertices,
    std::vector<std::uint8_t> &domain_empty,
    ActivePairCacheFailure &failure) noexcept {
  vertex_offsets.clear();
  vertices.clear();
  domain_empty.clear();
  if (constraint_offsets.size() != patch_count + 1U ||
      constraint_offsets.empty() ||
      constraint_offsets.back() > constraints.size())
    return false;
  try {
    if (patch_count == std::numeric_limits<std::size_t>::max() ||
        patch_count + 1U > vertex_offsets.max_size() ||
        patch_count > domain_empty.max_size())
      return false;
    vertex_offsets.reserve(patch_count + 1U);
    domain_empty.reserve(patch_count);
    vertex_offsets.push_back(0U);
    for (std::size_t patch = 0; patch < patch_count; ++patch) {
      const std::size_t begin = constraint_offsets[patch];
      const std::size_t end = constraint_offsets[patch + 1U];
      if (begin > end || end > constraints.size())
        return false;
      const std::span<const SpherePatchConstraint> patch_constraints(
          constraints.data() + begin, end - begin);
      if (!sphere_constraints_are_well_formed(patch_constraints)) {
        (void)enumerate_feasible_two_active_directions(
            patch_constraints, [](const Vec3 &) { return true; }, kind, patch,
            &failure);
        return false;
      }
      const bool impossible = std::any_of(
          patch_constraints.begin(), patch_constraints.end(),
          [](const SpherePatchConstraint &constraint) {
            return constraint.minimum_dot > 1.0 + kSphereConstraintTolerance;
          });
      if (impossible) {
        domain_empty.push_back(1U);
        vertex_offsets.push_back(vertices.size());
        continue;
      }
      const std::size_t vertex_begin = vertices.size();
      const bool enumerated = enumerate_feasible_two_active_directions(
          patch_constraints,
          [&](const Vec3 &direction_) {
            if (vertices.size() == vertices.max_size())
              return false;
            vertices.push_back(direction_);
            return true;
          },
          kind, patch, &failure);
      if (!enumerated || vertex_offsets.size() == vertex_offsets.max_size())
        return false;
      const bool nonempty =
          vertices.size() > vertex_begin ||
          has_feasible_zero_or_one_active_direction(patch_constraints);
      domain_empty.push_back(nonempty ? 0U : 1U);
      vertex_offsets.push_back(vertices.size());
    }
  } catch (...) {
    if (failure.reason == ActivePairCacheFailureReason::none) {
      failure.reason = ActivePairCacheFailureReason::storage;
      failure.kind = kind;
    }
    return false;
  }
  return vertex_offsets.size() == patch_count + 1U &&
         vertex_offsets.back() == vertices.size() &&
         domain_empty.size() == patch_count;
}

[[nodiscard]] ExactPatchProjection
project_constrained_sphere(const Vec3 &center, double radius,
                           std::span<const SpherePatchConstraint> constraints,
                           std::span<const Vec3> cached_two_active_directions,
                           const Vec3 &query, double normal_sign,
                           bool exhaustive_two_active) noexcept {
  ExactPatchProjection best;
  if (!(radius > 0.0) || !std::isfinite(radius) ||
      !sphere_constraints_are_well_formed(constraints)) {
    best.domain_certified = false;
    return best;
  }
  const auto consider_direction = [&](const Vec3 &raw_direction,
                                      bool check_feasibility) {
    const Vec3 direction_ = normalized(raw_direction);
    if (!(norm_squared(direction_) > 1.0e-20) || !finite_direction(direction_))
      return;
    if (check_feasibility &&
        !sphere_direction_is_feasible(constraints, direction_))
      return;
    retain_exact_candidate(best, query, add(center, scale(radius, direction_)),
                           scale(normal_sign, direction_));
  };

  const Vec3 objective = subtract(query, center);
  if (norm_squared(objective) > 1.0e-24) {
    consider_direction(objective, true);
  } else {
    consider_direction({1.0, 0.0, 0.0}, true);
    consider_direction({-1.0, 0.0, 0.0}, true);
    consider_direction({0.0, 1.0, 0.0}, true);
    consider_direction({0.0, -1.0, 0.0}, true);
    consider_direction({0.0, 0.0, 1.0}, true);
    consider_direction({0.0, 0.0, -1.0}, true);
  }

  // One-active extrema retain the exact O(m^2) feasibility check.
  for (std::size_t first = 0; first < constraints.size(); ++first) {
    const Vec3 n = constraints[first].normal;
    const double b = constraints[first].minimum_dot;
    if (b < -1.0 - kSphereConstraintTolerance ||
        b > 1.0 + kSphereConstraintTolerance)
      continue;
    const double clamped = std::clamp(b, -1.0, 1.0);
    const double circle_radius =
        std::sqrt(std::max(0.0, 1.0 - clamped * clamped));
    const Vec3 perpendicular = subtract(objective, scale(dot(objective, n), n));
    if (norm_squared(perpendicular) > 1.0e-24) {
      consider_direction(add(scale(clamped, n),
                             scale(circle_radius, normalized(perpendicular))),
                         true);
    } else {
      const Vec3 basis = deterministic_perpendicular(n);
      consider_direction(add(scale(clamped, n), scale(circle_radius, basis)),
                         true);
      consider_direction(
          subtract(scale(clamped, n), scale(circle_radius, basis)), true);
    }
  }

  if (exhaustive_two_active) {
    const bool enumerated = enumerate_feasible_two_active_directions(
        constraints,
        [&](const Vec3 &direction_) {
          consider_direction(direction_, false);
          return true;
        },
        PatchKind::contact, std::numeric_limits<std::size_t>::max(), nullptr);
    if (!enumerated)
      best.domain_certified = false;
  } else {
    for (const Vec3 &direction_ : cached_two_active_directions) {
      if (!finite_direction(direction_) ||
          std::abs(norm_squared(direction_) - 1.0) > 2.0e-12) {
        best.domain_certified = false;
        continue;
      }
      consider_direction(direction_, false);
    }
  }
  return best;
}

[[nodiscard]] std::pair<double, double>
torus_beta_endpoints(const PatchComplex &complex,
                     const ToroidalPatch &patch) noexcept {
  const double center_distance =
      norm(subtract(atom_center(complex.atoms[patch.atom_j]),
                    atom_center(complex.atoms[patch.atom_i])));
  const double beta_i = normalize_angle(
      std::atan2(-patch.axis_offset_i / patch.expanded_radius_i,
                 -patch.circle_radius / patch.expanded_radius_i));
  const double beta_j = normalize_angle(std::atan2(
      (center_distance - patch.axis_offset_i) / patch.expanded_radius_j,
      -patch.circle_radius / patch.expanded_radius_j));
  return {beta_i, beta_j};
}

[[nodiscard]] bool finite_atom_geometry(const Atom &atom) noexcept {
  return std::isfinite(atom.x) && std::isfinite(atom.y) &&
         std::isfinite(atom.z) && std::isfinite(atom.radius) &&
         atom.radius >= 0.0;
}

[[nodiscard]] bool close_geometry_value(double actual, double expected,
                                        double scale) noexcept {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             2.0e-10 * std::max({1.0, std::abs(actual), std::abs(expected),
                                 std::abs(scale)});
}

[[nodiscard]] Vec3 torus_radial(const ToroidalPatch &patch,
                                double theta) noexcept {
  return add(scale(std::cos(theta), patch.basis_u),
             scale(std::sin(theta), patch.basis_v));
}

[[nodiscard]] Vec3 torus_surface_point(const PatchComplex &complex,
                                       const ToroidalPatch &patch, double theta,
                                       double beta) noexcept {
  const Vec3 radial = torus_radial(patch, theta);
  return add(
      patch.circle_center,
      add(scale(patch.circle_radius + complex.probe_radius * std::cos(beta),
                radial),
          scale(complex.probe_radius * std::sin(beta), patch.axis)));
}

[[nodiscard]] ExactPatchProjection
project_constrained_torus(const PatchComplex &complex,
                          const LocalPatchProjectionIndex &index,
                          const ToroidalPatch &patch, const Vec3 &query,
                          bool apply_reentrant_clipping = true) noexcept {
  ExactPatchProjection best;
  const auto [beta_i, beta_j] = torus_beta_endpoints(complex, patch);
  const auto consider_angles = [&](double theta, double beta) {
    theta = normalize_angle(theta);
    beta = normalize_angle(beta);
    if (!angle_in_exposed_intervals(theta, patch.exposed_intervals) ||
        !angle_on_minor_arc(beta, beta_i, beta_j))
      return;
    const double radial_distance =
        patch.circle_radius + complex.probe_radius * std::cos(beta);
    const double envelope_tolerance =
        2.0e-12 * std::max({1.0, patch.circle_radius, complex.probe_radius});
    if (radial_distance < -envelope_tolerance)
      return;
    const Vec3 radial = torus_radial(patch, theta);
    const Vec3 tube_direction =
        add(scale(std::cos(beta), radial), scale(std::sin(beta), patch.axis));
    const Vec3 point = torus_surface_point(complex, patch, theta, beta);
    if (apply_reentrant_clipping &&
        covered_by_reentrant_probe(complex, index.reentrant_probes, point,
                                   std::numeric_limits<std::size_t>::max()))
      return;
    retain_exact_candidate(best, query, point, scale(-1.0, tube_direction),
                           std::abs(radial_distance) > envelope_tolerance);
  };

  // Interior foot on the complete torus.
  const Vec3 q = subtract(query, patch.circle_center);
  const double axial = dot(q, patch.axis);
  const Vec3 radial_vector = subtract(q, scale(axial, patch.axis));
  if (norm_squared(radial_vector) > 1.0e-24) {
    const Vec3 radial = normalized(radial_vector);
    const double theta = normalize_angle(
        std::atan2(dot(radial, patch.basis_v), dot(radial, patch.basis_u)));
    const Vec3 probe_center =
        add(patch.circle_center, scale(patch.circle_radius, radial));
    const Vec3 tube = subtract(query, probe_center);
    if (norm_squared(tube) > 1.0e-24) {
      const Vec3 tube_direction = normalized(tube);
      const double beta = normalize_angle(std::atan2(
          dot(tube_direction, patch.axis), dot(tube_direction, radial)));
      consider_angles(theta, beta);
    }
  }

  // Each interval is a bounded parameter rectangle on the physical envelope.
  // theta edges are torus-reentrant seams; beta edges are contact-torus seams.
  for (const AngularInterval &interval : patch.exposed_intervals) {
    const std::array<double, 2> theta_edges = {interval.begin, interval.end};
    const std::array<double, 2> beta_edges = {beta_i, beta_j};
    for (const double theta : theta_edges) {
      const Vec3 radial = torus_radial(patch, theta);
      const Vec3 probe_center =
          add(patch.circle_center, scale(patch.circle_radius, radial));
      const Vec3 tube = subtract(query, probe_center);
      if (norm_squared(tube) > 1.0e-24) {
        const double beta = normalize_angle(
            std::atan2(dot(tube, patch.axis), dot(tube, radial)));
        consider_angles(theta, beta);
      }
      for (const double beta : beta_edges)
        consider_angles(theta, beta);
    }
    for (const double beta : beta_edges) {
      const double circle_radius =
          patch.circle_radius + complex.probe_radius * std::cos(beta);
      const Vec3 circle_center =
          add(patch.circle_center,
              scale(complex.probe_radius * std::sin(beta), patch.axis));
      const Vec3 circle_q = subtract(query, circle_center);
      const double u = dot(circle_q, patch.basis_u);
      const double v = dot(circle_q, patch.basis_v);
      if (std::hypot(u, v) > 1.0e-14 && std::abs(circle_radius) > 1.0e-14) {
        const double theta =
            normalize_angle(std::atan2(circle_radius * v, circle_radius * u));
        if (theta >= interval.begin - 1.0e-12 &&
            theta <= interval.end + 1.0e-12)
          consider_angles(theta, beta);
      }
      consider_angles(interval.begin, beta);
      consider_angles(interval.end, beta);
    }
  }

  // A spindle torus has an additional physical-envelope boundary where the
  // radial coefficient vanishes. The full implicit torus continues onto an
  // inner sheet there, but that sheet is not generated by the closest probe
  // centre and must not participate in SES projection.
  if (patch.circle_radius < complex.probe_radius) {
    const double beta = std::acos(
        std::clamp(-patch.circle_radius / complex.probe_radius, -1.0, 1.0));
    const std::array<double, 2> envelope_edges = {
        beta, normalize_angle(2.0 * std::numbers::pi - beta)};
    for (const double envelope_beta : envelope_edges) {
      if (!angle_on_minor_arc(envelope_beta, beta_i, beta_j))
        continue;
      for (const AngularInterval &interval : patch.exposed_intervals)
        consider_angles(interval.begin, envelope_beta);
    }
  }
  return best;
}

[[nodiscard]] ExactPatchProjection
project_bounded_patch(const PatchComplex &complex,
                      const LocalPatchProjectionIndex &index, PatchKind kind,
                      std::size_t patch_index, const Vec3 &query) noexcept {
  if (kind == PatchKind::contact) {
    ExactPatchProjection result;
    const ContactPatch &patch = complex.contacts[patch_index];
    if (index.contact_constraint_offsets.size() !=
            complex.contacts.size() + 1U ||
        index.contact_two_active_offsets.size() !=
            complex.contacts.size() + 1U ||
        index.contact_domain_empty.size() != complex.contacts.size() ||
        patch_index + 1U >= index.contact_constraint_offsets.size() ||
        patch_index + 1U >= index.contact_two_active_offsets.size()) {
      result.domain_certified = false;
      return result;
    }
    const std::size_t begin = index.contact_constraint_offsets[patch_index];
    const std::size_t end = index.contact_constraint_offsets[patch_index + 1U];
    const std::size_t vertex_begin =
        index.contact_two_active_offsets[patch_index];
    const std::size_t vertex_end =
        index.contact_two_active_offsets[patch_index + 1U];
    if (begin > end || end > index.contact_constraints.size() ||
        vertex_begin > vertex_end ||
        vertex_end > index.contact_two_active_directions.size()) {
      result.domain_certified = false;
      return result;
    }
    if (index.contact_domain_empty[patch_index] != 0U) {
      result.domain_empty = true;
      return result;
    }
    return project_constrained_sphere(
        patch.center, patch.atom_radius,
        std::span<const SpherePatchConstraint>(index.contact_constraints)
            .subspan(begin, end - begin),
        std::span<const Vec3>(index.contact_two_active_directions)
            .subspan(vertex_begin, vertex_end - vertex_begin),
        query, 1.0, false);
  }
  if (kind == PatchKind::toroidal)
    return project_constrained_torus(complex, index, complex.tori[patch_index],
                                     query);

  ExactPatchProjection result;
  if (index.reentrant_constraint_offsets.size() !=
          complex.reentrant.size() + 1U ||
      index.reentrant_two_active_offsets.size() !=
          complex.reentrant.size() + 1U ||
      index.reentrant_domain_empty.size() != complex.reentrant.size() ||
      patch_index + 1U >= index.reentrant_constraint_offsets.size() ||
      patch_index + 1U >= index.reentrant_two_active_offsets.size()) {
    result.domain_certified = false;
    return result;
  }
  const std::size_t begin = index.reentrant_constraint_offsets[patch_index];
  const std::size_t end = index.reentrant_constraint_offsets[patch_index + 1U];
  const std::size_t vertex_begin =
      index.reentrant_two_active_offsets[patch_index];
  const std::size_t vertex_end =
      index.reentrant_two_active_offsets[patch_index + 1U];
  if (begin > end || end > index.reentrant_constraints.size() ||
      vertex_begin > vertex_end ||
      vertex_end > index.reentrant_two_active_directions.size()) {
    result.domain_certified = false;
    return result;
  }
  if (index.reentrant_domain_empty[patch_index] != 0U) {
    result.domain_empty = true;
    return result;
  }
  const ReentrantPatch &patch = complex.reentrant[patch_index];
  return project_constrained_sphere(
      patch.probe_center, complex.probe_radius,
      std::span<const SpherePatchConstraint>(index.reentrant_constraints)
          .subspan(begin, end - begin),
      std::span<const Vec3>(index.reentrant_two_active_directions)
          .subspan(vertex_begin, vertex_end - vertex_begin),
      query, -1.0, false);
}

[[nodiscard]] ExactPatchProjection project_bounded_patch_exhaustive(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    PatchKind kind, std::size_t patch_index, const Vec3 &query) noexcept {
  if (kind == PatchKind::toroidal)
    return project_bounded_patch(complex, index, kind, patch_index, query);

  ExactPatchProjection result;
  if (kind == PatchKind::contact) {
    if (index.contact_constraint_offsets.size() !=
            complex.contacts.size() + 1U ||
        index.contact_domain_empty.size() != complex.contacts.size() ||
        patch_index + 1U >= index.contact_constraint_offsets.size()) {
      result.domain_certified = false;
      return result;
    }
    const std::size_t begin = index.contact_constraint_offsets[patch_index];
    const std::size_t end = index.contact_constraint_offsets[patch_index + 1U];
    if (begin > end || end > index.contact_constraints.size()) {
      result.domain_certified = false;
      return result;
    }
    if (index.contact_domain_empty[patch_index] != 0U) {
      result.domain_empty = true;
      return result;
    }
    const ContactPatch &patch = complex.contacts[patch_index];
    return project_constrained_sphere(
        patch.center, patch.atom_radius,
        std::span<const SpherePatchConstraint>(index.contact_constraints)
            .subspan(begin, end - begin),
        {}, query, 1.0, true);
  }

  if (index.reentrant_constraint_offsets.size() !=
          complex.reentrant.size() + 1U ||
      index.reentrant_domain_empty.size() != complex.reentrant.size() ||
      patch_index + 1U >= index.reentrant_constraint_offsets.size()) {
    result.domain_certified = false;
    return result;
  }
  const std::size_t begin = index.reentrant_constraint_offsets[patch_index];
  const std::size_t end = index.reentrant_constraint_offsets[patch_index + 1U];
  if (begin > end || end > index.reentrant_constraints.size()) {
    result.domain_certified = false;
    return result;
  }
  if (index.reentrant_domain_empty[patch_index] != 0U) {
    result.domain_empty = true;
    return result;
  }
  const ReentrantPatch &patch = complex.reentrant[patch_index];
  return project_constrained_sphere(
      patch.probe_center, complex.probe_radius,
      std::span<const SpherePatchConstraint>(index.reentrant_constraints)
          .subspan(begin, end - begin),
      {}, query, -1.0, true);
}

} // namespace

bool torus_beta_minor_arc_for_audit(const PatchComplex &complex,
                                    std::size_t torus_index,
                                    TorusBetaMinorArc &arc) noexcept {
  arc = {};
  if (torus_index >= complex.tori.size() ||
      !std::isfinite(complex.probe_radius) || complex.probe_radius <= 0.0)
    return false;
  const ToroidalPatch &patch = complex.tori[torus_index];
  if (patch.atom_i >= complex.atoms.size() ||
      patch.atom_j >= complex.atoms.size() || patch.atom_i == patch.atom_j ||
      !finite_atom_geometry(complex.atoms[patch.atom_i]) ||
      !finite_atom_geometry(complex.atoms[patch.atom_j]) ||
      !std::isfinite(patch.axis_offset_i) ||
      !std::isfinite(patch.circle_radius) || patch.circle_radius <= 0.0 ||
      !std::isfinite(patch.expanded_radius_i) ||
      !std::isfinite(patch.expanded_radius_j) ||
      patch.expanded_radius_i <= 0.0 || patch.expanded_radius_j <= 0.0)
    return false;

  const Atom &atom_i = complex.atoms[patch.atom_i];
  const Atom &atom_j = complex.atoms[patch.atom_j];
  const Vec3 delta = subtract(atom_center(atom_j), atom_center(atom_i));
  const double center_distance = norm(delta);
  const double expanded_i = atom_i.radius + complex.probe_radius;
  const double expanded_j = atom_j.radius + complex.probe_radius;
  if (!std::isfinite(center_distance) || center_distance <= 0.0 ||
      !std::isfinite(expanded_i) || !std::isfinite(expanded_j) ||
      expanded_i <= 0.0 || expanded_j <= 0.0 ||
      !close_geometry_value(patch.expanded_radius_i, expanded_i,
                            center_distance) ||
      !close_geometry_value(patch.expanded_radius_j, expanded_j,
                            center_distance))
    return false;

  const double expected_offset =
      (center_distance * center_distance + expanded_i * expanded_i -
       expanded_j * expanded_j) /
      (2.0 * center_distance);
  const double circle_squared =
      expanded_i * expanded_i - expected_offset * expected_offset;
  const double geometry_scale =
      std::max({1.0, center_distance, expanded_i, expanded_j,
                std::abs(expected_offset), patch.circle_radius});
  if (!std::isfinite(expected_offset) || !std::isfinite(circle_squared) ||
      circle_squared < -2.0e-10 * geometry_scale * geometry_scale ||
      !close_geometry_value(patch.axis_offset_i, expected_offset,
                            geometry_scale) ||
      !close_geometry_value(patch.circle_radius,
                            std::sqrt(std::max(0.0, circle_squared)),
                            geometry_scale))
    return false;

  const Vec3 expected_axis = scale(1.0 / center_distance, delta);
  const Vec3 expected_center =
      add(atom_center(atom_i), scale(expected_offset, expected_axis));
  if (!close_geometry_value(patch.axis.x, expected_axis.x, 1.0) ||
      !close_geometry_value(patch.axis.y, expected_axis.y, 1.0) ||
      !close_geometry_value(patch.axis.z, expected_axis.z, 1.0) ||
      !close_geometry_value(patch.circle_center.x, expected_center.x,
                            geometry_scale) ||
      !close_geometry_value(patch.circle_center.y, expected_center.y,
                            geometry_scale) ||
      !close_geometry_value(patch.circle_center.z, expected_center.z,
                            geometry_scale))
    return false;

  const auto [beta_i, beta_j] = torus_beta_endpoints(complex, patch);
  return make_directed_minor_arc(beta_i, beta_j, arc);
}

bool directed_minor_arc_from_endpoints_for_audit(
    double endpoint_a, double endpoint_b, TorusBetaMinorArc &arc) noexcept {
  return make_directed_minor_arc(endpoint_a, endpoint_b, arc);
}

bool angle_on_torus_beta_production_minor_arc_for_audit(
    double angle, double endpoint_a, double endpoint_b) noexcept {
  return angle_on_minor_arc(angle, endpoint_a, endpoint_b);
}

bool angle_on_torus_beta_minor_arc_for_audit(
    double angle, const TorusBetaMinorArc &arc) noexcept {
  return angle_on_directed_minor_arc(angle, arc);
}

ReentrantProbeIndex build_reentrant_probe_index(const PatchComplex &complex) {
  ReentrantProbeIndex result;
  if (complex.reentrant.empty())
    return result;
  result.cell_size = std::max(1.0e-6, complex.probe_radius);
  result.origin = {std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()};
  for (const ReentrantPatch &patch : complex.reentrant) {
    result.origin.x = std::min(result.origin.x, patch.probe_center.x);
    result.origin.y = std::min(result.origin.y, patch.probe_center.y);
    result.origin.z = std::min(result.origin.z, patch.probe_center.z);
  }
  result.origin = subtract(
      result.origin, {result.cell_size, result.cell_size, result.cell_size});

  std::vector<CellKeyIndex> entries;
  entries.reserve(complex.reentrant.size());
  for (std::size_t i = 0; i < complex.reentrant.size(); ++i) {
    const auto cell = probe_cell(result, complex.reentrant[i].probe_center);
    entries.push_back(
        {cell[0], cell[1], cell[2], static_cast<std::uint32_t>(i)});
  }
  std::sort(entries.begin(), entries.end(),
            [](const CellKeyIndex &a, const CellKeyIndex &b) {
              return std::tie(a.x, a.y, a.z, a.index) <
                     std::tie(b.x, b.y, b.z, b.index);
            });
  result.probe_indices.reserve(entries.size());
  for (std::size_t begin = 0; begin < entries.size();) {
    std::size_t end = begin + 1U;
    while (end < entries.size() && entries[end].x == entries[begin].x &&
           entries[end].y == entries[begin].y &&
           entries[end].z == entries[begin].z)
      ++end;
    ProbeCell cell;
    cell.x = entries[begin].x;
    cell.y = entries[begin].y;
    cell.z = entries[begin].z;
    cell.begin = static_cast<std::uint32_t>(result.probe_indices.size());
    for (std::size_t i = begin; i < end; ++i)
      result.probe_indices.push_back(entries[i].index);
    cell.end = static_cast<std::uint32_t>(result.probe_indices.size());
    result.cells.push_back(cell);
    begin = end;
  }
  return result;
}

AnalyticPatchSeamCertificate build_analytic_patch_seam_certificate(
    const PatchComplex &complex, SeamCertificateStorage storage,
    std::span<const std::size_t> atom_offsets,
    std::span<const LocalPatchRef> atom_patches) {
  AnalyticPatchSeamCertificate result;
  result.source_complex = &complex;
  const bool retain_relations =
      storage == SeamCertificateStorage::retain_relations;
  if (complex.status != "pass" || complex.atoms.empty() ||
      !std::isfinite(complex.probe_radius) || !(complex.probe_radius > 0.0)) {
    result.status = "invalid_patch_complex";
    return result;
  }
  const auto finite_atom = [](const Atom &atom) {
    return std::isfinite(atom.x) && std::isfinite(atom.y) &&
           std::isfinite(atom.z) && std::isfinite(atom.radius) &&
           atom.radius > 0.0;
  };
  if (!std::all_of(complex.atoms.begin(), complex.atoms.end(), finite_atom)) {
    result.status = "invalid_patch_complex";
    return result;
  }
  if (complex.atoms.size() > std::numeric_limits<std::uint32_t>::max() ||
      complex.contacts.size() >
          static_cast<std::size_t>(PatchRefId::index_mask) + 1U ||
      complex.tori.size() >
          static_cast<std::size_t>(PatchRefId::index_mask) + 1U ||
      complex.reentrant.size() >
          static_cast<std::size_t>(PatchRefId::index_mask) + 1U ||
      complex.reentrant.size() > std::numeric_limits<std::size_t>::max() / 3U) {
    result.status = "patch_ref_overflow";
    return result;
  }

  const auto finite_vec = [](const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
  };
  for (const ContactPatch &contact : complex.contacts) {
    const Atom *atom = contact.atom < complex.atoms.size()
                           ? &complex.atoms[contact.atom]
                           : nullptr;
    const double structure_scale =
        atom == nullptr
            ? 1.0
            : std::max({1.0, std::abs(atom->x), std::abs(atom->y),
                        std::abs(atom->z), atom->radius, complex.probe_radius});
    const double structure_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() * structure_scale;
    const double center_residual =
        atom == nullptr ? std::numeric_limits<double>::infinity()
                        : norm(subtract(contact.center, atom_center(*atom)));
    const double radius_residual =
        atom == nullptr ? std::numeric_limits<double>::infinity()
                        : std::abs(contact.atom_radius - atom->radius);
    const double expanded_residual =
        atom == nullptr ? std::numeric_limits<double>::infinity()
                        : std::abs(contact.expanded_radius -
                                   (atom->radius + complex.probe_radius));
    if (contact.atom >= complex.atoms.size() || !finite_vec(contact.center) ||
        !std::isfinite(contact.atom_radius) ||
        !std::isfinite(contact.expanded_radius) ||
        !std::isfinite(center_residual) || !std::isfinite(radius_residual) ||
        !std::isfinite(expanded_residual) || !(contact.atom_radius > 0.0) ||
        !(contact.expanded_radius > 0.0) ||
        center_residual > structure_tolerance ||
        radius_residual > structure_tolerance ||
        expanded_residual > structure_tolerance) {
      result.status = "invalid_patch_complex";
      return result;
    }
  }
  const double seam_tolerance = 2.0e-8 * std::max(1.0, complex.probe_radius);
  const auto periodic_angle_distance = [](double a, double b) {
    const double period = 2.0 * std::numbers::pi;
    const double delta = std::abs(normalize_angle(a) - normalize_angle(b));
    return std::min(delta, period - delta);
  };

  struct PairTorus {
    std::uint32_t atom_a{0};
    std::uint32_t atom_b{0};
    std::uint32_t torus{0};
  };
  std::vector<PairTorus> pair_tori;
  if (retain_relations) {
    pair_tori.reserve(complex.tori.size());
    result.torus_contact_curve_certified.assign(
        complex.tori.size(), std::array<std::uint8_t, 2>{0U, 0U});
  }
  bool structure_certified = true;
  if (!retain_relations) {
    bool valid_atom_csr = atom_offsets.size() == complex.atoms.size() + 1U &&
                          !atom_offsets.empty() &&
                          atom_offsets.back() <= atom_patches.size();
    for (std::size_t atom = 0;
         valid_atom_csr && atom + 1U < atom_offsets.size(); ++atom)
      valid_atom_csr = atom_offsets[atom] <= atom_offsets[atom + 1U];
    if (!valid_atom_csr) {
      result.status = "invalid_atom_patch_csr";
      return result;
    }
  }
  for (std::size_t patch_index = 0; patch_index < complex.tori.size();
       ++patch_index) {
    const ToroidalPatch &patch = complex.tori[patch_index];
    PatchRefId packed;
    if (!PatchRefId::try_make(PatchKind::toroidal, patch_index, packed) ||
        patch.atom_i >= complex.atoms.size() ||
        patch.atom_j >= complex.atoms.size() || patch.atom_i == patch.atom_j) {
      structure_certified = false;
      ++result.uncertified_seams;
      continue;
    }
    (void)packed;
    if (retain_relations) {
      pair_tori.push_back(
          {static_cast<std::uint32_t>(std::min(patch.atom_i, patch.atom_j)),
           static_cast<std::uint32_t>(std::max(patch.atom_i, patch.atom_j)),
           static_cast<std::uint32_t>(patch_index)});
    }

    bool domain_certified =
        finite_vec(patch.circle_center) && finite_vec(patch.axis) &&
        finite_vec(patch.basis_u) && finite_vec(patch.basis_v) &&
        std::abs(norm_squared(patch.axis) - 1.0) <= 2.0e-10 &&
        std::abs(norm_squared(patch.basis_u) - 1.0) <= 2.0e-10 &&
        std::abs(norm_squared(patch.basis_v) - 1.0) <= 2.0e-10 &&
        std::abs(dot(patch.axis, patch.basis_u)) <= 2.0e-10 &&
        std::abs(dot(patch.axis, patch.basis_v)) <= 2.0e-10 &&
        std::abs(dot(patch.basis_u, patch.basis_v)) <= 2.0e-10 &&
        patch.circle_radius > 0.0 && std::isfinite(patch.circle_radius) &&
        std::isfinite(patch.axis_offset_i) &&
        std::isfinite(patch.expanded_radius_i) &&
        std::isfinite(patch.expanded_radius_j) &&
        patch.expanded_radius_i > 0.0 && patch.expanded_radius_j > 0.0 &&
        !patch.exposed_intervals.empty();
    const auto [beta_i, beta_j] = torus_beta_endpoints(complex, patch);
    std::array<bool, 2> side_certified = {
        domain_certified && std::isfinite(beta_i),
        domain_certified && std::isfinite(beta_j)};
    for (const AngularInterval &interval : patch.exposed_intervals) {
      const bool valid_interval =
          std::isfinite(interval.begin) && std::isfinite(interval.end) &&
          interval.begin >= -1.0e-12 &&
          interval.end <= 2.0 * std::numbers::pi + 1.0e-12 &&
          interval.end + 1.0e-12 >= interval.begin;
      if (!valid_interval) {
        side_certified = {false, false};
        continue;
      }
      const double theta = 0.5 * (interval.begin + interval.end);
      const Vec3 point_i = torus_surface_point(complex, patch, theta, beta_i);
      const Vec3 point_j = torus_surface_point(complex, patch, theta, beta_j);
      const double residual_i = std::abs(
          norm(subtract(point_i, atom_center(complex.atoms[patch.atom_i]))) -
          complex.atoms[patch.atom_i].radius);
      const double residual_j = std::abs(
          norm(subtract(point_j, atom_center(complex.atoms[patch.atom_j]))) -
          complex.atoms[patch.atom_j].radius);
      if (!finite_vec(point_i) || !std::isfinite(residual_i) ||
          residual_i > seam_tolerance)
        side_certified[0] = false;
      if (!finite_vec(point_j) || !std::isfinite(residual_j) ||
          residual_j > seam_tolerance)
        side_certified[1] = false;
    }
    for (std::size_t side = 0; side < side_certified.size(); ++side) {
      if (side_certified[side]) {
        if (retain_relations)
          result.torus_contact_curve_certified[patch_index][side] = 1U;
        ++result.certified_torus_contact_seams;
      } else {
        ++result.uncertified_seams;
      }
    }
  }
  if (retain_relations) {
    std::sort(pair_tori.begin(), pair_tori.end(),
              [](const PairTorus &left, const PairTorus &right) {
                return std::tie(left.atom_a, left.atom_b, left.torus) <
                       std::tie(right.atom_a, right.atom_b, right.torus);
              });
  }

  if (retain_relations)
    result.reentrant_edges.reserve(complex.reentrant.size() * 3U);
  for (std::size_t patch_index = 0; patch_index < complex.reentrant.size();
       ++patch_index) {
    const ReentrantPatch &reentrant = complex.reentrant[patch_index];
    PatchRefId reentrant_ref;
    const bool valid_ref =
        PatchRefId::try_make(PatchKind::reentrant, patch_index, reentrant_ref);
    bool valid_patch = valid_ref && finite_vec(reentrant.probe_center);
    for (std::size_t atom : reentrant.atoms)
      valid_patch = valid_patch && atom < complex.atoms.size();
    for (const Vec3 &direction_ : reentrant.contact_directions)
      valid_patch = valid_patch && finite_vec(direction_) &&
                    std::abs(norm_squared(direction_) - 1.0) <= 2.0e-10;
    if (!valid_patch)
      structure_certified = false;

    for (std::size_t edge = 0; edge < 3U; ++edge) {
      ReentrantEdgeSeamCertificate certificate;
      certificate.reentrant = reentrant_ref;
      certificate.edge = static_cast<std::uint8_t>(edge);
      if (!valid_patch) {
        if (retain_relations)
          result.reentrant_edges.push_back(certificate);
        ++result.uncertified_seams;
        continue;
      }
      const std::uint32_t atom_a = static_cast<std::uint32_t>(
          std::min(reentrant.atoms[edge], reentrant.atoms[(edge + 1U) % 3U]));
      const std::uint32_t atom_b = static_cast<std::uint32_t>(
          std::max(reentrant.atoms[edge], reentrant.atoms[(edge + 1U) % 3U]));

      std::size_t welded_count = 0U;
      PatchRefId welded_torus;
      const auto consider_torus = [&](std::size_t torus_index) {
        if (torus_index >= complex.tori.size() ||
            (retain_relations &&
             (result.torus_contact_curve_certified[torus_index][0] == 0U ||
              result.torus_contact_curve_certified[torus_index][1] == 0U)))
          return;
        const ToroidalPatch &torus = complex.tori[torus_index];
        const Vec3 circle_delta =
            subtract(reentrant.probe_center, torus.circle_center);
        const double axial = dot(circle_delta, torus.axis);
        const Vec3 radial_vector =
            subtract(circle_delta, scale(axial, torus.axis));
        const double radial_norm = norm(radial_vector);
        if (!finite_vec(circle_delta) || !std::isfinite(axial) ||
            !finite_vec(radial_vector) || !std::isfinite(radial_norm) ||
            !(radial_norm > 1.0e-300) || std::abs(axial) > seam_tolerance ||
            std::abs(radial_norm - torus.circle_radius) > seam_tolerance)
          return;
        const Vec3 radial = normalized(radial_vector);
        const double theta = normalize_angle(
            std::atan2(dot(radial, torus.basis_v), dot(radial, torus.basis_u)));
        if (!finite_vec(radial) ||
            std::abs(norm_squared(radial) - 1.0) > 2.0e-10 ||
            !std::isfinite(theta))
          return;
        bool endpoint = false;
        for (const AngularInterval &interval : torus.exposed_intervals) {
          endpoint = endpoint ||
                     periodic_angle_distance(theta, interval.begin) <= 2.0e-8 ||
                     periodic_angle_distance(theta, interval.end) <= 2.0e-8;
        }
        if (!endpoint)
          return;

        const std::size_t oriented_atom_a = reentrant.atoms[edge];
        const std::size_t oriented_atom_b = reentrant.atoms[(edge + 1U) % 3U];
        const auto [beta_i, beta_j] = torus_beta_endpoints(complex, torus);
        const double beta_a = torus.atom_i == oriented_atom_a ? beta_i : beta_j;
        const double beta_b = torus.atom_i == oriented_atom_b ? beta_i : beta_j;
        if (!std::isfinite(beta_a) || !std::isfinite(beta_b))
          return;
        const Vec3 torus_a = torus_surface_point(complex, torus, theta, beta_a);
        const Vec3 torus_b = torus_surface_point(complex, torus, theta, beta_b);
        const Vec3 sphere_a = add(
            reentrant.probe_center,
            scale(complex.probe_radius, reentrant.contact_directions[edge]));
        const Vec3 sphere_b =
            add(reentrant.probe_center,
                scale(complex.probe_radius,
                      reentrant.contact_directions[(edge + 1U) % 3U]));
        const double seam_a = norm(subtract(torus_a, sphere_a));
        const double seam_b = norm(subtract(torus_b, sphere_b));
        if (!finite_vec(torus_a) || !finite_vec(torus_b) ||
            !finite_vec(sphere_a) || !finite_vec(sphere_b) ||
            !std::isfinite(seam_a) || !std::isfinite(seam_b) ||
            seam_a > seam_tolerance || seam_b > seam_tolerance)
          return;
        PatchRefId candidate;
        if (!PatchRefId::try_make(PatchKind::toroidal, torus_index,
                                  candidate)) {
          structure_certified = false;
          return;
        }
        welded_torus = candidate;
        ++welded_count;
      };
      if (retain_relations) {
        const PairTorus lower_key{atom_a, atom_b, 0U};
        const PairTorus upper_key{atom_a, atom_b,
                                  std::numeric_limits<std::uint32_t>::max()};
        const auto first = std::lower_bound(
            pair_tori.begin(), pair_tori.end(), lower_key,
            [](const PairTorus &left, const PairTorus &right) {
              return std::tie(left.atom_a, left.atom_b, left.torus) <
                     std::tie(right.atom_a, right.atom_b, right.torus);
            });
        const auto last = std::upper_bound(
            pair_tori.begin(), pair_tori.end(), upper_key,
            [](const PairTorus &left, const PairTorus &right) {
              return std::tie(left.atom_a, left.atom_b, left.torus) <
                     std::tie(right.atom_a, right.atom_b, right.torus);
            });
        for (auto current = first; current != last; ++current)
          consider_torus(current->torus);
      } else {
        const std::size_t lookup_atom = reentrant.atoms[edge];
        for (std::size_t position = atom_offsets[lookup_atom];
             position < atom_offsets[lookup_atom + 1U]; ++position) {
          const LocalPatchRef ref = atom_patches[position];
          if (ref.kind != PatchKind::toroidal)
            continue;
          const std::size_t torus_index = ref.patch_index;
          if (torus_index >= complex.tori.size())
            continue;
          const ToroidalPatch &torus = complex.tori[torus_index];
          if (std::min(torus.atom_i, torus.atom_j) != atom_a ||
              std::max(torus.atom_i, torus.atom_j) != atom_b)
            continue;
          consider_torus(torus_index);
        }
      }
      if (welded_count == 1U) {
        certificate.torus = welded_torus;
        certificate.relation = PatchRelation::curve_adjacent;
        ++result.certified_torus_reentrant_seams;
      } else {
        ++result.uncertified_seams;
      }
      if (retain_relations)
        result.reentrant_edges.push_back(certificate);
    }
  }

  result.structure_certified =
      structure_certified &&
      (!retain_relations ||
       (result.torus_contact_curve_certified.size() == complex.tori.size() &&
        result.reentrant_edges.size() == complex.reentrant.size() * 3U));
  result.all_seams_certified =
      result.structure_certified && result.uncertified_seams == 0U;
  result.status =
      result.all_seams_certified ? "pass" : "uncertified_analytic_seam";
  return result;
}

LocalPatchProjectionIndex build_local_patch_projection_index(
    const PatchComplex &complex,
    LocalPatchProjectionIndexBuildOptions options) {
  LocalPatchProjectionIndex result;
  result.source_complex = &complex;
  result.bounded_domains_certified = false;
  const std::uint64_t patch_count =
      static_cast<std::uint64_t>(complex.contacts.size()) +
      static_cast<std::uint64_t>(complex.tori.size()) +
      static_cast<std::uint64_t>(complex.reentrant.size());
  bool input_valid =
      complex.status == "pass" && !complex.atoms.empty() &&
      complex.atoms.size() != std::numeric_limits<std::size_t>::max() &&
      complex.atom_neighbors.size() == complex.atoms.size() &&
      std::isfinite(complex.probe_radius) && complex.probe_radius > 0.0 &&
      complex.contacts.size() <= std::numeric_limits<std::uint32_t>::max() &&
      complex.tori.size() <= std::numeric_limits<std::uint32_t>::max() &&
      complex.reentrant.size() <= std::numeric_limits<std::uint32_t>::max() &&
      patch_count <= std::numeric_limits<std::uint32_t>::max();
  for (const Atom &atom : complex.atoms) {
    input_valid = input_valid && std::isfinite(atom.x) &&
                  std::isfinite(atom.y) && std::isfinite(atom.z) &&
                  std::isfinite(atom.radius) && atom.radius > 0.0;
  }
  for (std::size_t atom = 0U;
       input_valid && atom < complex.atom_neighbors.size(); ++atom) {
    input_valid = std::all_of(
        complex.atom_neighbors[atom].begin(),
        complex.atom_neighbors[atom].end(),
        [&](std::size_t neighbor) { return neighbor < complex.atoms.size(); });
  }
  for (const ContactPatch &patch : complex.contacts) {
    input_valid = input_valid && patch.atom < complex.atoms.size() &&
                  finite_direction(patch.center) &&
                  std::isfinite(patch.atom_radius) && patch.atom_radius > 0.0 &&
                  std::isfinite(patch.expanded_radius) &&
                  patch.expanded_radius > patch.atom_radius;
  }
  for (const ToroidalPatch &patch : complex.tori) {
    input_valid =
        input_valid && patch.atom_i < complex.atoms.size() &&
        patch.atom_j < complex.atoms.size() && patch.atom_i != patch.atom_j &&
        finite_direction(patch.circle_center) && finite_direction(patch.axis) &&
        finite_direction(patch.basis_u) && finite_direction(patch.basis_v) &&
        std::isfinite(patch.circle_radius) &&
        std::isfinite(patch.axis_offset_i) &&
        std::isfinite(patch.expanded_radius_i) &&
        std::isfinite(patch.expanded_radius_j);
    for (const AngularInterval &interval : patch.exposed_intervals)
      input_valid = input_valid && std::isfinite(interval.begin) &&
                    std::isfinite(interval.end);
  }
  for (const ReentrantPatch &patch : complex.reentrant) {
    input_valid = input_valid && finite_direction(patch.probe_center);
    for (std::size_t vertex = 0U; vertex < 3U; ++vertex)
      input_valid = input_valid && patch.atoms[vertex] < complex.atoms.size() &&
                    finite_direction(patch.contact_directions[vertex]);
  }
  if (!input_valid)
    return result;
  if (options.retain_complex_fingerprint) {
    result.source_complex_fingerprint = fingerprint_patch_complex(complex);
    if (!result.source_complex_fingerprint.valid)
      return result;
  }
  result.bounded_domains_certified = true;
  result.reentrant_probes = build_reentrant_probe_index(complex);

  std::vector<std::size_t> counts(complex.atoms.size(), 0U);
  for (const ContactPatch &patch : complex.contacts)
    ++counts[patch.atom];
  for (const ToroidalPatch &patch : complex.tori) {
    ++counts[patch.atom_i];
    ++counts[patch.atom_j];
  }
  for (const ReentrantPatch &patch : complex.reentrant) {
    for (const std::size_t atom : patch.atoms)
      ++counts[atom];
  }
  result.atom_offsets.resize(complex.atoms.size() + 1U, 0U);
  for (std::size_t atom = 0; atom < counts.size(); ++atom)
    result.atom_offsets[atom + 1U] = result.atom_offsets[atom] + counts[atom];
  result.patches.resize(result.atom_offsets.back());
  std::vector<std::size_t> cursor = result.atom_offsets;
  const auto append = [&](std::size_t atom, PatchKind kind,
                          std::size_t patch_index) {
    result.patches[cursor[atom]++] = {static_cast<std::uint32_t>(patch_index),
                                      kind};
  };
  for (std::size_t index = 0; index < complex.contacts.size(); ++index)
    append(complex.contacts[index].atom, PatchKind::contact, index);
  for (std::size_t index = 0; index < complex.tori.size(); ++index) {
    const ToroidalPatch &patch = complex.tori[index];
    append(patch.atom_i, PatchKind::toroidal, index);
    append(patch.atom_j, PatchKind::toroidal, index);
  }
  for (std::size_t index = 0; index < complex.reentrant.size(); ++index) {
    for (const std::size_t atom : complex.reentrant[index].atoms)
      append(atom, PatchKind::reentrant, index);
  }

  result.contact_constraint_offsets.reserve(complex.contacts.size() + 1U);
  result.contact_constraint_offsets.push_back(0U);
  for (const ContactPatch &patch : complex.contacts) {
    if (patch.atom >= complex.atoms.size()) {
      result.bounded_domains_certified = false;
      result.contact_constraint_offsets.push_back(
          result.contact_constraints.size());
      continue;
    }
    for (const std::size_t neighbor : complex.atom_neighbors[patch.atom]) {
      const Vec3 delta =
          subtract(atom_center(complex.atoms[neighbor]), patch.center);
      const double distance = norm(delta);
      const double neighbor_radius =
          complex.atoms[neighbor].radius + complex.probe_radius;
      if (!(distance > 1.0e-14)) {
        if (patch.expanded_radius + 1.0e-14 < neighbor_radius)
          result.contact_constraints.push_back({{1.0, 0.0, 0.0}, 2.0});
        continue;
      }
      const Vec3 axis = scale(1.0 / distance, delta);
      const double maximum_dot =
          (patch.expanded_radius * patch.expanded_radius + distance * distance -
           neighbor_radius * neighbor_radius) /
          (2.0 * patch.expanded_radius * distance);
      result.contact_constraints.push_back({scale(-1.0, axis), -maximum_dot});
    }
    result.contact_constraint_offsets.push_back(
        result.contact_constraints.size());
  }

  result.reentrant_constraint_offsets.reserve(complex.reentrant.size() + 1U);
  result.reentrant_constraint_offsets.push_back(0U);
  for (std::size_t patch_index = 0; patch_index < complex.reentrant.size();
       ++patch_index) {
    const ReentrantPatch &patch = complex.reentrant[patch_index];
    const Vec3 anchor = normalized(
        add(add(patch.contact_directions[0], patch.contact_directions[1]),
            patch.contact_directions[2]));
    if (!(norm_squared(anchor) > 1.0e-20))
      result.bounded_domains_certified = false;
    for (std::size_t edge = 0; edge < 3U; ++edge) {
      Vec3 plane = cross(patch.contact_directions[edge],
                         patch.contact_directions[(edge + 1U) % 3U]);
      const double plane_length = norm(plane);
      if (!(plane_length > 1.0e-14)) {
        result.bounded_domains_certified = false;
        result.reentrant_constraints.push_back({{0.0, 0.0, 0.0}, 2.0});
        continue;
      }
      plane = scale(1.0 / plane_length, plane);
      if (dot(plane, anchor) < 0.0)
        plane = scale(-1.0, plane);
      result.reentrant_constraints.push_back({plane, 0.0});
    }

    // covered_by_reentrant_probe clips a spherical triangle by every
    // overlapping probe sphere. On the unit sphere each such boundary is the
    // analytic small circle dot(-axis, direction) = -distance/(2*probe).
    std::vector<std::uint32_t> blockers;
    const auto center_cell =
        probe_cell(result.reentrant_probes, patch.probe_center);
    for (int dz = -2; dz <= 2; ++dz) {
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
          const ProbeCell *cell =
              find_cell(result.reentrant_probes, center_cell[0] + dx,
                        center_cell[1] + dy, center_cell[2] + dz);
          if (cell == nullptr)
            continue;
          for (std::uint32_t position = cell->begin; position < cell->end;
               ++position) {
            const std::uint32_t blocker =
                result.reentrant_probes.probe_indices[position];
            if (blocker != patch_index)
              blockers.push_back(blocker);
          }
        }
      }
    }
    std::sort(blockers.begin(), blockers.end());
    blockers.erase(std::unique(blockers.begin(), blockers.end()),
                   blockers.end());
    for (const std::uint32_t blocker : blockers) {
      const Vec3 delta =
          subtract(complex.reentrant[blocker].probe_center, patch.probe_center);
      const double distance = norm(delta);
      const double scale_length = std::max(1.0, complex.probe_radius);
      if (!(distance > 1.0e-12 * scale_length) ||
          distance > 2.0 * complex.probe_radius + 2.0e-12 * scale_length)
        continue;
      const Vec3 axis = scale(1.0 / distance, delta);
      result.reentrant_constraints.push_back(
          {scale(-1.0, axis), -distance / (2.0 * complex.probe_radius)});
    }
    result.reentrant_constraint_offsets.push_back(
        result.reentrant_constraints.size());
  }

  const auto finite_vec = [](const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
  };
  const auto validate_constraint_blocks =
      [&](const std::vector<std::size_t> &offsets,
          const std::vector<SpherePatchConstraint> &constraints,
          std::size_t patch_count_) {
        if (offsets.size() != patch_count_ + 1U || offsets.empty() ||
            offsets.back() > constraints.size()) {
          result.bounded_domains_certified = false;
          return;
        }
        for (std::size_t patch_index = 0; patch_index < patch_count_;
             ++patch_index) {
          const std::size_t begin = offsets[patch_index];
          const std::size_t end = offsets[patch_index + 1U];
          if (begin > end || end > constraints.size()) {
            result.bounded_domains_certified = false;
            continue;
          }
          for (std::size_t first = begin; first < end; ++first) {
            const SpherePatchConstraint &a = constraints[first];
            if (!finite_vec(a.normal) || !std::isfinite(a.minimum_dot) ||
                std::abs(norm_squared(a.normal) - 1.0) > 2.0e-10) {
              result.bounded_domains_certified = false;
            }
          }
        }
      };
  validate_constraint_blocks(result.contact_constraint_offsets,
                             result.contact_constraints,
                             complex.contacts.size());
  validate_constraint_blocks(result.reentrant_constraint_offsets,
                             result.reentrant_constraints,
                             complex.reentrant.size());

  const auto active_pair_cache_begin = std::chrono::steady_clock::now();
  const bool contact_cache_certified = build_two_active_direction_cache(
      result.contact_constraint_offsets, result.contact_constraints,
      complex.contacts.size(), PatchKind::contact,
      result.contact_two_active_offsets, result.contact_two_active_directions,
      result.contact_domain_empty, result.active_pair_cache_failure);
  const bool reentrant_cache_certified = build_two_active_direction_cache(
      result.reentrant_constraint_offsets, result.reentrant_constraints,
      complex.reentrant.size(), PatchKind::reentrant,
      result.reentrant_two_active_offsets,
      result.reentrant_two_active_directions, result.reentrant_domain_empty,
      result.active_pair_cache_failure);
  result.active_pair_cache_build_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - active_pair_cache_begin)
          .count();
  if (!contact_cache_certified || !reentrant_cache_certified)
    result.bounded_domains_certified = false;

  // Reuse the exact seam predicates through a summary-only sink. Production
  // projection setup retains no provenance vectors and keeps its existing
  // atom-local pair lookup; the audit oracle requests detailed records later.
  const double seam_tolerance = 2.0e-8 * std::max(1.0, complex.probe_radius);
  const AnalyticPatchSeamCertificate seam_certificate =
      build_analytic_patch_seam_certificate(
          complex, SeamCertificateStorage::summary_only, result.atom_offsets,
          result.patches);
  if (!seam_certificate.all_seams_certified)
    result.bounded_domains_certified = false;

  // Reentrant balls associated with theta endpoints touch, but do not clip,
  // the physical torus envelope. Any strict interior overlap from another
  // probe would create an unrepresented trimming curve, so detect it at setup
  // and fail the closest-point certificate rather than overclaim.
  const auto torus_reentrant_audit_begin = std::chrono::steady_clock::now();
  for (const ToroidalPatch &torus : complex.tori) {
    if (result.reentrant_probes.cells.empty())
      break;
    PatchBounds bounds =
        patch_bounds(complex, PatchKind::toroidal,
                     static_cast<std::size_t>(&torus - complex.tori.data()));
    const Vec3 margin{complex.probe_radius, complex.probe_radius,
                      complex.probe_radius};
    const auto lower_cell =
        probe_cell(result.reentrant_probes, subtract(bounds.lower, margin));
    const auto upper_cell =
        probe_cell(result.reentrant_probes, add(bounds.upper, margin));
    std::vector<std::uint32_t> blockers;
    for (std::int32_t z = lower_cell[2]; z <= upper_cell[2]; ++z) {
      for (std::int32_t y = lower_cell[1]; y <= upper_cell[1]; ++y) {
        for (std::int32_t x = lower_cell[0]; x <= upper_cell[0]; ++x) {
          const ProbeCell *cell = find_cell(result.reentrant_probes, x, y, z);
          if (cell == nullptr)
            continue;
          for (std::uint32_t position = cell->begin; position < cell->end;
               ++position)
            blockers.push_back(result.reentrant_probes.probe_indices[position]);
        }
      }
    }
    std::sort(blockers.begin(), blockers.end());
    blockers.erase(std::unique(blockers.begin(), blockers.end()),
                   blockers.end());
    for (const std::uint32_t blocker : blockers) {
      ++result.torus_reentrant_pairs_audited;
      const ExactPatchProjection nearest = project_constrained_torus(
          complex, result, torus, complex.reentrant[blocker].probe_center,
          false);
      if (nearest.found &&
          nearest.distance < complex.probe_radius - seam_tolerance) {
        result.bounded_domains_certified = false;
        break;
      }
    }
  }
  result.torus_reentrant_audit_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - torus_reentrant_audit_begin)
          .count();

  result.spatial_entries.reserve(static_cast<std::size_t>(patch_count));
  const auto append_spatial = [&](PatchKind kind, std::size_t patch_index) {
    result.spatial_entries.push_back(
        {outward_padded_patch_bounds(complex, kind, patch_index),
         {static_cast<std::uint32_t>(patch_index), kind},
         canonical_patch_id(complex, kind, patch_index)});
  };
  for (std::size_t index = 0; index < complex.contacts.size(); ++index)
    append_spatial(PatchKind::contact, index);
  for (std::size_t index = 0; index < complex.tori.size(); ++index)
    append_spatial(PatchKind::toroidal, index);
  for (std::size_t index = 0; index < complex.reentrant.size(); ++index)
    append_spatial(PatchKind::reentrant, index);

  const bool spatial_entries_valid =
      std::all_of(result.spatial_entries.begin(), result.spatial_entries.end(),
                  [](const SpatialPatchEntry &entry) {
                    return bounds_are_finite_and_ordered(entry.bounds);
                  });
  constexpr std::uint32_t kLeafSize = 8U;
  result.spatial_nodes.reserve(result.spatial_entries.empty()
                                   ? 0U
                                   : result.spatial_entries.size() / 2U + 1U);
  const auto build_node = [&](auto &&self, std::uint32_t begin,
                              std::uint32_t end) -> std::uint32_t {
    PatchBounds bounds = result.spatial_entries[begin].bounds;
    Vec3 centroid_lower{bounds_centroid_component(bounds, 0),
                        bounds_centroid_component(bounds, 1),
                        bounds_centroid_component(bounds, 2)};
    Vec3 centroid_upper = centroid_lower;
    for (std::uint32_t position = begin + 1U; position < end; ++position) {
      const PatchBounds &entry_bounds = result.spatial_entries[position].bounds;
      bounds = merge_bounds(bounds, entry_bounds);
      const Vec3 centroid{bounds_centroid_component(entry_bounds, 0),
                          bounds_centroid_component(entry_bounds, 1),
                          bounds_centroid_component(entry_bounds, 2)};
      centroid_lower.x = std::min(centroid_lower.x, centroid.x);
      centroid_lower.y = std::min(centroid_lower.y, centroid.y);
      centroid_lower.z = std::min(centroid_lower.z, centroid.z);
      centroid_upper.x = std::max(centroid_upper.x, centroid.x);
      centroid_upper.y = std::max(centroid_upper.y, centroid.y);
      centroid_upper.z = std::max(centroid_upper.z, centroid.z);
    }

    const std::uint32_t node_index =
        static_cast<std::uint32_t>(result.spatial_nodes.size());
    result.spatial_nodes.push_back({outward_padded_bounds(bounds), begin, end});
    if (end - begin <= kLeafSize)
      return node_index;

    const Vec3 span = subtract(centroid_upper, centroid_lower);
    int axis = 0;
    double longest = span.x;
    if (span.y > longest) {
      axis = 1;
      longest = span.y;
    }
    if (span.z > longest)
      axis = 2;
    const std::uint32_t middle = begin + (end - begin) / 2U;
    std::nth_element(
        result.spatial_entries.begin() + begin,
        result.spatial_entries.begin() + middle,
        result.spatial_entries.begin() + end,
        [axis](const SpatialPatchEntry &a, const SpatialPatchEntry &b) {
          const double ca = bounds_centroid_component(a.bounds, axis);
          const double cb = bounds_centroid_component(b.bounds, axis);
          if (ca < cb)
            return true;
          if (cb < ca)
            return false;
          return a.canonical_id < b.canonical_id;
        });
    const std::uint32_t left = self(self, begin, middle);
    const std::uint32_t right = self(self, middle, end);
    result.spatial_nodes[node_index].left = left;
    result.spatial_nodes[node_index].right = right;
    return node_index;
  };
  if (spatial_entries_valid && !result.spatial_entries.empty())
    (void)build_node(build_node, 0U,
                     static_cast<std::uint32_t>(result.spatial_entries.size()));

  result.bvh_invariants_certified = spatial_entries_valid;
  const std::uint32_t no_child = std::numeric_limits<std::uint32_t>::max();
  if (result.spatial_entries.empty()) {
    result.bvh_invariants_certified =
        result.bvh_invariants_certified && result.spatial_nodes.empty();
  } else if (!result.bvh_invariants_certified || result.spatial_nodes.empty() ||
             result.spatial_nodes[0].begin != 0U ||
             result.spatial_nodes[0].end != result.spatial_entries.size()) {
    result.bvh_invariants_certified = false;
  } else {
    std::vector<std::uint8_t> visited(result.spatial_nodes.size(), 0U);
    std::vector<std::uint32_t> stack = {0U};
    while (!stack.empty() && result.bvh_invariants_certified) {
      const std::uint32_t node_index = stack.back();
      stack.pop_back();
      if (node_index >= result.spatial_nodes.size() ||
          visited[node_index] != 0U) {
        result.bvh_invariants_certified = false;
        break;
      }
      visited[node_index] = 1U;
      const SpatialPatchNode &node = result.spatial_nodes[node_index];
      if (!bounds_are_finite_and_ordered(node.bounds) ||
          node.begin >= node.end || node.end > result.spatial_entries.size()) {
        result.bvh_invariants_certified = false;
        break;
      }
      const bool left_missing = node.left == no_child;
      const bool right_missing = node.right == no_child;
      if (left_missing || right_missing) {
        if (!(left_missing && right_missing) ||
            node.end - node.begin > kLeafSize)
          result.bvh_invariants_certified = false;
        for (std::uint32_t position = node.begin;
             position < node.end && result.bvh_invariants_certified;
             ++position) {
          if (!bounds_contain(node.bounds,
                              result.spatial_entries[position].bounds))
            result.bvh_invariants_certified = false;
        }
        continue;
      }
      if (node.left >= result.spatial_nodes.size() ||
          node.right >= result.spatial_nodes.size()) {
        result.bvh_invariants_certified = false;
        break;
      }
      const SpatialPatchNode &left = result.spatial_nodes[node.left];
      const SpatialPatchNode &right = result.spatial_nodes[node.right];
      if (left.begin != node.begin || left.end != right.begin ||
          right.end != node.end || !bounds_contain(node.bounds, left.bounds) ||
          !bounds_contain(node.bounds, right.bounds)) {
        result.bvh_invariants_certified = false;
        break;
      }
      stack.push_back(node.right);
      stack.push_back(node.left);
    }
    if (result.bvh_invariants_certified &&
        std::any_of(visited.begin(), visited.end(),
                    [](std::uint8_t value) { return value == 0U; }))
      result.bvh_invariants_certified = false;
  }
  result.bounded_domains_certified =
      result.bounded_domains_certified && result.bvh_invariants_certified;
  return result;
}

PatchBounds patch_bounds(const PatchComplex &complex, PatchKind kind,
                         std::size_t patch_index) noexcept {
  switch (kind) {
  case PatchKind::contact: {
    const ContactPatch &patch = complex.contacts[patch_index];
    const Vec3 extent{patch.atom_radius, patch.atom_radius, patch.atom_radius};
    return {subtract(patch.center, extent), add(patch.center, extent)};
  }
  case PatchKind::toroidal: {
    const ToroidalPatch &patch = complex.tori[patch_index];
    // Bound the actual stored parameterization rather than assuming its frame
    // is exactly orthonormal.  For one coordinate, radial(theta) has amplitude
    // h=hypot(basis_u,basis_v); the minor-circle term is then bounded by
    // rp*hypot(h,axis).  This also avoids the near-axis cancellation in
    // sqrt(1-axis_component^2).
    const auto extent = [&](double basis_u_component, double basis_v_component,
                            double axis_component) {
      const double radial_amplitude =
          std::hypot(basis_u_component, basis_v_component);
      return patch.circle_radius * radial_amplitude +
             complex.probe_radius *
                 std::hypot(radial_amplitude, axis_component);
    };
    const Vec3 radius{extent(patch.basis_u.x, patch.basis_v.x, patch.axis.x),
                      extent(patch.basis_u.y, patch.basis_v.y, patch.axis.y),
                      extent(patch.basis_u.z, patch.basis_v.z, patch.axis.z)};
    return {subtract(patch.circle_center, radius),
            add(patch.circle_center, radius)};
  }
  case PatchKind::reentrant: {
    const ReentrantPatch &patch = complex.reentrant[patch_index];
    const Vec3 extent{complex.probe_radius, complex.probe_radius,
                      complex.probe_radius};
    return {subtract(patch.probe_center, extent),
            add(patch.probe_center, extent)};
  }
  }
  return {};
}

PatchBounds outward_padded_patch_bounds(const PatchComplex &complex,
                                        PatchKind kind,
                                        std::size_t patch_index) noexcept {
  return outward_padded_bounds(patch_bounds(complex, kind, patch_index));
}

LocalProjectionHit
project_patch_bounded_closest(const PatchComplex &complex,
                              const LocalPatchProjectionIndex &index,
                              PatchKind kind, std::size_t patch_index,
                              const Vec3 &point, double max_distance) noexcept {
  LocalProjectionHit result;
  const bool valid_index =
      (kind == PatchKind::contact && patch_index < complex.contacts.size()) ||
      (kind == PatchKind::toroidal && patch_index < complex.tori.size()) ||
      (kind == PatchKind::reentrant && patch_index < complex.reentrant.size());
  if (complex.status != "pass" || !valid_index || !std::isfinite(point.x) ||
      !std::isfinite(point.y) || !std::isfinite(point.z) ||
      !(max_distance >= 0.0) || !std::isfinite(max_distance))
    return result;
  const ExactPatchProjection candidate =
      project_bounded_patch(complex, index, kind, patch_index, point);
  result.candidate_patches = 1U;
  result.normal_certified = candidate.normal_certified;
  if (!candidate.found ||
      candidate.distance > max_distance + 2.0e-10 * std::max(1.0, max_distance))
    return result;
  result.point = candidate.point;
  result.normal = candidate.normal;
  result.kind = kind;
  result.patch_index = patch_index;
  result.atom_index = patch_atom_hint(complex, kind, patch_index);
  result.distance = candidate.distance;
  result.found = true;
  result.bounded_closest_certified = candidate.domain_certified &&
                                     index.bounded_domains_certified &&
                                     candidate.normal_certified;
  return result;
}

LocalProjectionHit project_patch_bounded_closest_exhaustive_for_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    PatchKind kind, std::size_t patch_index, const Vec3 &point,
    double max_distance) noexcept {
  LocalProjectionHit result;
  const bool valid_index =
      (kind == PatchKind::contact && patch_index < complex.contacts.size()) ||
      (kind == PatchKind::toroidal && patch_index < complex.tori.size()) ||
      (kind == PatchKind::reentrant && patch_index < complex.reentrant.size());
  if (complex.status != "pass" || !valid_index || !std::isfinite(point.x) ||
      !std::isfinite(point.y) || !std::isfinite(point.z) ||
      !(max_distance >= 0.0) || !std::isfinite(max_distance))
    return result;
  const ExactPatchProjection candidate = project_bounded_patch_exhaustive(
      complex, index, kind, patch_index, point);
  result.candidate_patches = 1U;
  result.normal_certified = candidate.normal_certified;
  if (!candidate.found ||
      candidate.distance > max_distance + 2.0e-10 * std::max(1.0, max_distance))
    return result;
  result.point = candidate.point;
  result.normal = candidate.normal;
  result.kind = kind;
  result.patch_index = patch_index;
  result.atom_index = patch_atom_hint(complex, kind, patch_index);
  result.distance = candidate.distance;
  result.found = true;
  result.bounded_closest_certified = candidate.domain_certified &&
                                     index.bounded_domains_certified &&
                                     candidate.normal_certified;
  return result;
}

template <bool CollectDiagnostics>
static PatchAxisRoots intersect_patch_axis_impl(
    const PatchComplex &complex, const ReentrantProbeIndex &probe_index,
    PatchKind kind, std::size_t patch_index, RayAxis axis, double u, double v,
    bool retain_tangential, QuarticDiagnostics *quartic_diagnostics,
    TorusPatchAxisDiagnostics *filter_diagnostics) noexcept {
  if constexpr (CollectDiagnostics)
    *quartic_diagnostics = {};
  if constexpr (CollectDiagnostics) {
    if (filter_diagnostics != nullptr)
      *filter_diagnostics = {};
  }
  PatchAxisRoots result;
  if (complex.status != "pass" || !std::isfinite(u) || !std::isfinite(v)) {
    result.certified = false;
    return result;
  }
  const Vec3 ray_direction = direction(axis);
  const double tangent_tolerance = 1.0e-10;

  if (kind == PatchKind::contact || kind == PatchKind::reentrant) {
    const Vec3 center = kind == PatchKind::contact
                            ? complex.contacts[patch_index].center
                            : complex.reentrant[patch_index].probe_center;
    const double radius = kind == PatchKind::contact
                              ? complex.contacts[patch_index].atom_radius
                              : complex.probe_radius;
    const double center_coordinate = component(center, axis);
    const Vec3 closest = line_point(axis, center_coordinate, u, v);
    const double perpendicular2 = norm_squared(subtract(closest, center));
    const double radius2 = radius * radius;
    const double tolerance = 2.0e-12 * std::max(1.0, radius2);
    if (perpendicular2 > radius2 + tolerance)
      return result;
    const double delta = std::sqrt(std::max(0.0, radius2 - perpendicular2));
    const std::array<double, 2> coordinates = {center_coordinate - delta,
                                               center_coordinate + delta};
    const std::size_t count = delta <= std::sqrt(tolerance) ? 1U : 2U;
    for (std::size_t side = 0; side < count; ++side) {
      const Vec3 point = line_point(axis, coordinates[side], u, v);
      const Vec3 radial = normalized(subtract(point, center));
      const double directional_normal =
          kind == PatchKind::contact
              ? std::abs(dot(radial, ray_direction))
              : std::abs(dot(scale(-1.0, radial), ray_direction));
      if (kind == PatchKind::contact) {
        const ContactPatch &patch = complex.contacts[patch_index];
        if (!contact_point_is_exposed(complex, patch, point) ||
            (!retain_tangential && directional_normal <= tangent_tolerance))
          continue;
      } else {
        const ReentrantPatch &patch = complex.reentrant[patch_index];
        if (!spherical_triangle_contains(patch, radial) ||
            covered_by_reentrant_probe(complex, probe_index, point,
                                       patch_index) ||
            (!retain_tangential && directional_normal <= tangent_tolerance))
          continue;
      }
      // The sphere branch intentionally preserves its historical coordinate
      // rule, which represents roots closer than sqrt(tolerance) by one
      // coordinate. Such a coordinate is not a certified parity-changing
      // simple root even when its sampled normal exceeds tangent_tolerance.
      append_coordinate(result, coordinates[side],
                        count == 2U && directional_normal > tangent_tolerance);
    }
  } else {
    const ToroidalPatch &patch = complex.tori[patch_index];
    const double center_coordinate = component(patch.circle_center, axis);
    const Vec3 local_origin = line_point(axis, center_coordinate, u, v);
    const Vec3 q0 = subtract(local_origin, patch.circle_center);
    const double axial0 = dot(q0, patch.axis);
    const double axis_component = dot(ray_direction, patch.axis);
    const double q0_squared = norm_squared(q0);
    const double major2 = patch.circle_radius * patch.circle_radius;
    const double minor2 = complex.probe_radius * complex.probe_radius;
    const double p0 = q0_squared + major2 - minor2;
    const double p1 = 2.0 * dot(q0, ray_direction);
    const double u0 = q0_squared - axial0 * axial0;
    const double u1 = 2.0 * (dot(q0, ray_direction) - axis_component * axial0);
    const double u2 = 1.0 - axis_component * axis_component;
    const std::array<double, 5> coefficients = {
        p0 * p0 - 4.0 * major2 * u0, 2.0 * p0 * p1 - 4.0 * major2 * u1,
        p1 * p1 + 2.0 * p0 - 4.0 * major2 * u2, 2.0 * p1, 1.0};
    const QuarticRoots roots = [&]() {
      if constexpr (CollectDiagnostics)
        return solve_real_quartic(coefficients, quartic_diagnostics);
      return solve_real_quartic(coefficients);
    }();
    result.certified = roots.certified;
    if constexpr (CollectDiagnostics) {
      if (filter_diagnostics != nullptr)
        filter_diagnostics->root_count = roots.count;
    }
    for (std::size_t root = 0; root < roots.count; ++root) {
      const double coordinate = center_coordinate + roots.values[root];
      const Vec3 point = line_point(axis, coordinate, u, v);
      TorusRootFilterDiagnostic *filter = nullptr;
      if constexpr (CollectDiagnostics) {
        if (filter_diagnostics != nullptr)
          filter = &filter_diagnostics->roots[root];
      }
      if (filter != nullptr) {
        filter->quartic_root = roots.values[root];
        filter->coordinate = coordinate;
        filter->point = point;
      }
      const Vec3 q = subtract(point, patch.circle_center);
      const double axial = dot(q, patch.axis);
      const Vec3 radial_vector = subtract(q, scale(axial, patch.axis));
      const double radial_length = norm(radial_vector);
      if (filter != nullptr)
        filter->radial_length = radial_length;
      if (!(radial_length > 1.0e-12)) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::radial_axis_singular;
        continue;
      }
      const Vec3 radial = scale(1.0 / radial_length, radial_vector);
      const double theta = normalize_angle(
          std::atan2(dot(radial, patch.basis_v), dot(radial, patch.basis_u)));
      if (filter != nullptr)
        filter->theta = theta;
      if (!angle_in_exposed_intervals(theta, patch.exposed_intervals)) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::theta_interval;
        continue;
      }
      const Vec3 probe_center =
          add(patch.circle_center, scale(patch.circle_radius, radial));
      const Vec3 tube_vector = subtract(point, probe_center);
      const double tube_length = norm(tube_vector);
      const double tube_tolerance =
          2.0e-9 * std::max({1.0, patch.circle_radius, complex.probe_radius});
      if (filter != nullptr) {
        filter->tube_length = tube_length;
        filter->tube_residual = tube_length - complex.probe_radius;
      }
      if (std::abs(tube_length - complex.probe_radius) > tube_tolerance) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::tube_consistency;
        continue;
      }
      const Vec3 tube_direction = scale(1.0 / tube_length, tube_vector);
      const Atom &atom_i = complex.atoms[patch.atom_i];
      const Atom &atom_j = complex.atoms[patch.atom_j];
      const double center_distance =
          norm(subtract(atom_center(atom_j), atom_center(atom_i)));
      const double beta = normalize_angle(std::atan2(
          dot(tube_direction, patch.axis), dot(tube_direction, radial)));
      if (filter != nullptr)
        filter->beta = beta;
      const double beta_i = normalize_angle(
          std::atan2(-patch.axis_offset_i / patch.expanded_radius_i,
                     -patch.circle_radius / patch.expanded_radius_i));
      const double beta_j = normalize_angle(std::atan2(
          (center_distance - patch.axis_offset_i) / patch.expanded_radius_j,
          -patch.circle_radius / patch.expanded_radius_j));
      if (!angle_on_minor_arc(beta, beta_i, beta_j)) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::beta_minor_arc;
        continue;
      }
      if (covered_by_reentrant_probe(complex, probe_index, point,
                                     std::numeric_limits<std::size_t>::max())) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::reentrant_coverage;
        continue;
      }
      const Vec3 outward = scale(-1.0, tube_direction);
      const double directional_normal = std::abs(dot(outward, ray_direction));
      if (filter != nullptr)
        filter->directional_normal = directional_normal;
      if (!retain_tangential && directional_normal <= tangent_tolerance) {
        if (filter != nullptr)
          filter->reason = TorusRootFilterReason::tangential;
        continue;
      }
      // The certified quartic solver supplies the complete distinct real-root
      // set. On a regular torus branch a nonzero directional surface normal
      // proves the implicit ray root is simple; repeated/tangent roots are
      // therefore retained only as audit-uncertain when explicitly requested.
      append_coordinate(result, coordinate,
                        directional_normal > tangent_tolerance);
      if (filter != nullptr)
        filter->reason = TorusRootFilterReason::accepted;
    }
  }

  sort_coordinates(result);
  return result;
}

PatchAxisRoots intersect_patch_axis(const PatchComplex &complex,
                                    const ReentrantProbeIndex &probe_index,
                                    PatchKind kind, std::size_t patch_index,
                                    RayAxis axis, double u, double v,
                                    bool retain_tangential) noexcept {
  return intersect_patch_axis_impl<false>(complex, probe_index, kind,
                                          patch_index, axis, u, v,
                                          retain_tangential, nullptr, nullptr);
}

PatchAxisRoots intersect_patch_axis_with_quartic_diagnostics(
    const PatchComplex &complex, const ReentrantProbeIndex &probe_index,
    PatchKind kind, std::size_t patch_index, RayAxis axis, double u, double v,
    QuarticDiagnostics *diagnostics, bool retain_tangential) noexcept {
  if (diagnostics == nullptr)
    return intersect_patch_axis_impl<false>(
        complex, probe_index, kind, patch_index, axis, u, v, retain_tangential,
        nullptr, nullptr);
  return intersect_patch_axis_impl<true>(
      complex, probe_index, kind, patch_index, axis, u, v, retain_tangential,
      diagnostics, nullptr);
}

PatchAxisRoots intersect_patch_axis_with_quartic_diagnostics(
    const PatchComplex &complex, const ReentrantProbeIndex &probe_index,
    PatchKind kind, std::size_t patch_index, RayAxis axis, double u, double v,
    QuarticDiagnostics *quartic_diagnostics,
    TorusPatchAxisDiagnostics *filter_diagnostics,
    bool retain_tangential) noexcept {
  if (quartic_diagnostics == nullptr && filter_diagnostics == nullptr)
    return intersect_patch_axis_impl<false>(
        complex, probe_index, kind, patch_index, axis, u, v, retain_tangential,
        nullptr, nullptr);
  QuarticDiagnostics local_quartic_diagnostics;
  QuarticDiagnostics *const quartic_output = quartic_diagnostics == nullptr
                                                 ? &local_quartic_diagnostics
                                                 : quartic_diagnostics;
  return intersect_patch_axis_impl<true>(
      complex, probe_index, kind, patch_index, axis, u, v, retain_tangential,
      quartic_output, filter_diagnostics);
}

LocalProjectionHit
project_local_patch_axis(const PatchComplex &complex,
                         const LocalPatchProjectionIndex &index,
                         std::size_t atom_index, const Vec3 &point,
                         RayAxis axis, double max_axis_distance) noexcept {
  LocalProjectionHit best;
  if (complex.status != "pass" || atom_index >= complex.atoms.size() ||
      index.atom_offsets.size() != complex.atoms.size() + 1U ||
      !(max_axis_distance >= 0.0) || !std::isfinite(max_axis_distance))
    return best;

  const double max_distance_tolerance =
      2.0e-12 * std::max(1.0, max_axis_distance);
  const auto orthogonal_coordinates = [&]() {
    switch (axis) {
    case RayAxis::x:
      return std::pair{point.y, point.z};
    case RayAxis::y:
      return std::pair{point.x, point.z};
    case RayAxis::z:
      return std::pair{point.x, point.y};
    }
    return std::pair{point.y, point.z};
  }();
  const auto consider_ref = [&](LocalPatchRef ref) {
    const std::size_t patch_index = ref.patch_index;
    const PatchAxisRoots roots = intersect_patch_axis(
        complex, index.reentrant_probes, ref.kind, patch_index, axis,
        orthogonal_coordinates.first, orthogonal_coordinates.second, true);
    if (!roots.certified)
      return;
    for (std::size_t root = 0; root < roots.count; ++root) {
      const double coordinate = roots.coordinates[root];
      const double distance = std::abs(coordinate - component(point, axis));
      if (distance > max_axis_distance + max_distance_tolerance)
        continue;
      bool replace = !best.found || distance < best.axis_distance;
      if (!replace && best.found && distance == best.axis_distance) {
        replace =
            std::tuple{static_cast<int>(ref.kind), patch_index, coordinate} <
            std::tuple{static_cast<int>(best.kind), best.patch_index,
                       component(best.point, axis)};
      }
      if (!replace)
        continue;
      const Vec3 projected =
          line_point(axis, coordinate, orthogonal_coordinates.first,
                     orthogonal_coordinates.second);
      const Vec3 normal =
          patch_outward_normal(complex, ref.kind, patch_index, projected);
      if (norm_squared(normal) <= 1.0e-20)
        continue;
      best.point = projected;
      best.normal = normal;
      best.kind = ref.kind;
      best.patch_index = patch_index;
      best.atom_index = patch_atom_hint(complex, ref.kind, patch_index);
      best.distance = distance;
      best.axis_distance = distance;
      best.normal_certified = true;
      best.found = true;
    }
  };
  const auto consider_atom = [&](std::size_t atom) {
    if (atom >= complex.atoms.size())
      return;
    for (std::size_t position = index.atom_offsets[atom];
         position < index.atom_offsets[atom + 1U]; ++position) {
      consider_ref(index.patches[position]);
    }
  };

  consider_atom(atom_index);
  for (const std::size_t neighbor : complex.atom_neighbors[atom_index])
    consider_atom(neighbor);
  return best;
}

LocalProjectionHit project_local_patch_closest_with_hint_policy_for_audit(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    std::size_t atom_index, const Vec3 &point, double max_distance,
    ProjectionHintPolicy hint_policy, bool collect_kind_timing) noexcept {
  LocalProjectionHit best;
  if (complex.status != "pass" || atom_index >= complex.atoms.size() ||
      !std::isfinite(point.x) || !std::isfinite(point.y) ||
      !std::isfinite(point.z) || !(max_distance >= 0.0) ||
      !std::isfinite(max_distance))
    return best;
  if (index.atom_offsets.size() != complex.atoms.size() + 1U) {
    best.traversal_invariant_failed = true;
    return best;
  }
  const double pruning_tolerance = 2.0e-10 * std::max(1.0, max_distance);
  std::size_t candidate_patches = 0U;
  std::size_t bound_pruned_patches = 0U;
  std::size_t spatial_nodes_visited = 0U;
  std::size_t owner_hint_candidates = 0U;
  std::array<std::size_t, 3> candidate_patches_by_kind{};
  std::array<std::size_t, 3> empty_short_circuits_by_kind{};
  std::array<std::uint64_t, 3> projector_nanoseconds_by_kind{};
  bool best_from_owner_hint = false;
  bool projector_domain_certified = index.bounded_domains_certified;

  const auto consider = [&](PatchKind kind, std::size_t patch_index,
                            bool from_owner_hint) {
    const std::size_t kind_slot = kind == PatchKind::contact
                                      ? 0U
                                      : (kind == PatchKind::toroidal ? 1U : 2U);
    ++candidate_patches;
    ++candidate_patches_by_kind[kind_slot];
    if (from_owner_hint)
      ++owner_hint_candidates;
    std::chrono::steady_clock::time_point projector_begin;
    if (collect_kind_timing)
      projector_begin = std::chrono::steady_clock::now();
    const ExactPatchProjection candidate =
        project_bounded_patch(complex, index, kind, patch_index, point);
    if (collect_kind_timing) {
      const auto elapsed =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - projector_begin)
              .count();
      if (elapsed > 0)
        projector_nanoseconds_by_kind[kind_slot] +=
            static_cast<std::uint64_t>(elapsed);
    }
    if (candidate.domain_empty)
      ++empty_short_circuits_by_kind[kind_slot];
    projector_domain_certified =
        projector_domain_certified && candidate.domain_certified;
    if (!candidate.found)
      return;
    if (!std::isfinite(candidate.distance) ||
        candidate.distance > max_distance + pruning_tolerance)
      return;
    const bool replace =
        !best.found || candidate.distance < best.distance ||
        (candidate.distance == best.distance &&
         std::tuple{static_cast<int>(kind), patch_index, candidate.point.x,
                    candidate.point.y, candidate.point.z} <
             std::tuple{static_cast<int>(best.kind), best.patch_index,
                        best.point.x, best.point.y, best.point.z});
    if (!replace)
      return;
    best.point = candidate.point;
    best.normal = candidate.normal;
    best.kind = kind;
    best.patch_index = patch_index;
    best.atom_index = patch_atom_hint(complex, kind, patch_index);
    best.distance = candidate.distance;
    best.normal_certified = candidate.normal_certified;
    best.found = true;
    best_from_owner_hint = from_owner_hint;
  };

  const std::size_t total_patch_count =
      complex.contacts.size() + complex.tori.size() + complex.reentrant.size();
  struct QueryStampStorage {
    std::vector<std::uint32_t> stamps;
    std::uint32_t generation{0U};
  };
  thread_local QueryStampStorage query_stamps;
  bool stamps_ready = false;
  bool stamp_allocation_failed = false;
  try {
    if (query_stamps.stamps.size() < total_patch_count)
      query_stamps.stamps.resize(total_patch_count, 0U);
    ++query_stamps.generation;
    if (query_stamps.generation == 0U) {
      std::fill(query_stamps.stamps.begin(), query_stamps.stamps.end(), 0U);
      query_stamps.generation = 1U;
    }
    stamps_ready = true;
  } catch (...) {
    // Hints are optional. The immutable BVH remains the correctness path if a
    // thread-local stamp buffer cannot be grown.
    stamp_allocation_failed = true;
  }

  const auto consider_hint_atom = [&](std::size_t atom) {
    if (!stamps_ready || atom >= complex.atoms.size())
      return;
    for (std::size_t position = index.atom_offsets[atom];
         position < index.atom_offsets[atom + 1U]; ++position) {
      const LocalPatchRef ref = index.patches[position];
      const std::uint32_t canonical_id =
          canonical_patch_id(complex, ref.kind, ref.patch_index);
      if (query_stamps.stamps[canonical_id] == query_stamps.generation)
        continue;
      query_stamps.stamps[canonical_id] = query_stamps.generation;
      consider(ref.kind, ref.patch_index, true);
    }
  };
  if (hint_policy != ProjectionHintPolicy::bvh_only)
    consider_hint_atom(atom_index);
  if (hint_policy == ProjectionHintPolicy::owner_and_neighbors) {
    for (const std::size_t neighbor : complex.atom_neighbors[atom_index])
      consider_hint_atom(neighbor);
  }

  // Owner/neighbour patches only seed an upper bound. Every unvisited patch
  // whose AABB can tie or improve it is still evaluated by the flat BVH.
  bool traversal_invariant_failed =
      !index.bvh_invariants_certified ||
      index.spatial_entries.size() != total_patch_count ||
      (total_patch_count > 0U && index.spatial_nodes.empty());
  if (!traversal_invariant_failed && !index.spatial_nodes.empty() &&
      !index.spatial_entries.empty()) {
    std::array<std::uint32_t, 64> stack{};
    std::size_t stack_size = 0U;
    stack[stack_size++] = 0U;
    while (stack_size > 0U) {
      const std::uint32_t node_index = stack[--stack_size];
      if (node_index >= index.spatial_nodes.size()) {
        traversal_invariant_failed = true;
        break;
      }
      const SpatialPatchNode &node = index.spatial_nodes[node_index];
      if (node.begin > node.end || node.end > index.spatial_entries.size()) {
        traversal_invariant_failed = true;
        break;
      }
      ++spatial_nodes_visited;
      const double limit = best.found ? best.distance : max_distance;
      const double certified_limit = limit + pruning_tolerance;
      if (point_bounds_distance_squared(point, node.bounds) >
          certified_limit * certified_limit) {
        bound_pruned_patches += static_cast<std::size_t>(node.end - node.begin);
        continue;
      }
      if (node.is_leaf()) {
        for (std::uint32_t position = node.begin; position < node.end;
             ++position) {
          const SpatialPatchEntry &entry = index.spatial_entries[position];
          const double entry_limit =
              (best.found ? best.distance : max_distance) + pruning_tolerance;
          if (point_bounds_distance_squared(point, entry.bounds) >
              entry_limit * entry_limit) {
            ++bound_pruned_patches;
            continue;
          }
          if (stamps_ready && query_stamps.stamps[entry.canonical_id] ==
                                  query_stamps.generation)
            continue;
          consider(entry.patch.kind, entry.patch.patch_index, false);
        }
        continue;
      }

      if (node.left >= index.spatial_nodes.size() ||
          node.right >= index.spatial_nodes.size()) {
        traversal_invariant_failed = true;
        break;
      }
      const SpatialPatchNode &left = index.spatial_nodes[node.left];
      const SpatialPatchNode &right = index.spatial_nodes[node.right];
      const double left_distance =
          point_bounds_distance_squared(point, left.bounds);
      const double right_distance =
          point_bounds_distance_squared(point, right.bounds);
      const bool left_first =
          left_distance < right_distance ||
          (left_distance == right_distance && node.left < node.right);
      const std::uint32_t near_child = left_first ? node.left : node.right;
      const std::uint32_t far_child = left_first ? node.right : node.left;
      // Median splitting keeps the depth below 32 for the uint32 patch limit.
      if (stack_size + 2U > stack.size()) {
        traversal_invariant_failed = true;
        break;
      }
      stack[stack_size++] = far_child;
      stack[stack_size++] = near_child;
    }
  }

  const auto attach_telemetry = [&]() {
    best.candidate_patches = candidate_patches;
    best.candidate_patches_by_kind = candidate_patches_by_kind;
    best.empty_domain_short_circuits_by_kind = empty_short_circuits_by_kind;
    best.projector_nanoseconds_by_kind = projector_nanoseconds_by_kind;
    best.bound_pruned_patches = bound_pruned_patches;
    best.spatial_nodes_visited = spatial_nodes_visited;
    best.owner_hint_candidates = owner_hint_candidates;
    best.stamp_bytes_per_worker =
        query_stamps.stamps.capacity() * sizeof(std::uint32_t);
    best.owner_hint_hit = best.found && best_from_owner_hint;
    best.spatial_fallback = best.found && !best_from_owner_hint;
    best.stamp_allocation_failed = stamp_allocation_failed;
    best.traversal_invariant_failed = traversal_invariant_failed;
    best.bounded_closest_certified =
        best.found && !traversal_invariant_failed &&
        projector_domain_certified && best.normal_certified;
  };
  if (traversal_invariant_failed) {
    best = {};
    attach_telemetry();
    return best;
  }
  if (best.found) {
    attach_telemetry();
    return best;
  }

  // A Euclidean projection onto the unbounded primitive can fall just outside
  // a bounded patch at a contact/toroidal/reentrant seam.  Only for these rare
  // misses, retain the existing owner-local coordinate-line fallback.
  for (const RayAxis axis : {RayAxis::x, RayAxis::y, RayAxis::z}) {
    const LocalProjectionHit candidate = project_local_patch_axis(
        complex, index, atom_index, point, axis, max_distance);
    if (!candidate.found)
      continue;
    const bool replace =
        !best.found || candidate.distance < best.distance ||
        (candidate.distance == best.distance &&
         std::tuple{static_cast<int>(candidate.kind), candidate.patch_index,
                    candidate.point.x, candidate.point.y, candidate.point.z} <
             std::tuple{static_cast<int>(best.kind), best.patch_index,
                        best.point.x, best.point.y, best.point.z});
    if (replace) {
      best = candidate;
      best_from_owner_hint = true;
    }
  }
  best.candidate_patches = candidate_patches;
  best.candidate_patches_by_kind = candidate_patches_by_kind;
  best.empty_domain_short_circuits_by_kind = empty_short_circuits_by_kind;
  best.projector_nanoseconds_by_kind = projector_nanoseconds_by_kind;
  best.bound_pruned_patches = bound_pruned_patches;
  best.spatial_nodes_visited = spatial_nodes_visited;
  best.owner_hint_candidates = owner_hint_candidates;
  best.stamp_bytes_per_worker =
      query_stamps.stamps.capacity() * sizeof(std::uint32_t);
  best.owner_hint_hit = best.found;
  best.spatial_fallback = false;
  best.axis_fallback = best.found;
  best.stamp_allocation_failed = stamp_allocation_failed;
  best.traversal_invariant_failed = false;
  best.bounded_closest_certified = false;
  return best;
}

LocalProjectionHit project_local_patch_closest(
    const PatchComplex &complex, const LocalPatchProjectionIndex &index,
    std::size_t atom_index, const Vec3 &point, double max_distance) noexcept {
  // The owner patch is a cheap upper-bound seed. Neighbour patches are still
  // proved by the same complete BVH traversal, while eagerly projecting every
  // neighbour duplicates almost all exact work on dense molecular surfaces.
  return project_local_patch_closest_with_hint_policy_for_audit(
      complex, index, atom_index, point, max_distance,
      ProjectionHintPolicy::owner_only, false);
}

LocalProjectionHit project_local_patch_x(const PatchComplex &complex,
                                         const LocalPatchProjectionIndex &index,
                                         std::size_t atom_index,
                                         const Vec3 &point,
                                         double max_axis_distance) noexcept {
  return project_local_patch_axis(complex, index, atom_index, point, RayAxis::x,
                                  max_axis_distance);
}

} // namespace TitanPBE::Tabi::analytic_ses

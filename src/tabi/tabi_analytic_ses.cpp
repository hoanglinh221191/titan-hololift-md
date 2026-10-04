#include "tabi_analytic_ses.h"
#include "tabi_ses_quartic.h"

#include "../titan_executor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <unordered_map>
#include <utility>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

constexpr double kGeometryRelTol = 2.0e-12;

struct CellKey {
  int x{0};
  int y{0};
  int z{0};
  bool operator==(const CellKey &) const = default;
};

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const noexcept {
    std::size_t hash = static_cast<std::size_t>(key.x) * 73856093U;
    hash ^= static_cast<std::size_t>(key.y) * 19349663U;
    hash ^= static_cast<std::size_t>(key.z) * 83492791U;
    return hash;
  }
};

struct SpatialBins {
  Vec3 origin;
  double cell_size{1.0};
  std::unordered_map<CellKey, std::vector<std::size_t>, CellKeyHash> cells;
};

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

[[nodiscard]] CellKey cell_for(const SpatialBins &bins,
                               const Vec3 &point) noexcept {
  return {
      static_cast<int>(std::floor((point.x - bins.origin.x) / bins.cell_size)),
      static_cast<int>(std::floor((point.y - bins.origin.y) / bins.cell_size)),
      static_cast<int>(std::floor((point.z - bins.origin.z) / bins.cell_size))};
}

[[nodiscard]] SpatialBins build_bins(std::span<const Atom> atoms,
                                     double probe_radius) {
  SpatialBins bins;
  bins.origin = {std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()};
  double max_expanded_radius = 0.0;
  for (const Atom &atom : atoms) {
    const Vec3 center = atom_center(atom);
    bins.origin.x = std::min(bins.origin.x, center.x);
    bins.origin.y = std::min(bins.origin.y, center.y);
    bins.origin.z = std::min(bins.origin.z, center.z);
    max_expanded_radius =
        std::max(max_expanded_radius, atom.radius + probe_radius);
  }
  bins.cell_size = std::max(1.0e-6, max_expanded_radius);
  bins.cells.reserve(atoms.size() * 2U);
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    bins.cells[cell_for(bins, atom_center(atoms[index]))].push_back(index);
  }
  return bins;
}

[[nodiscard]] std::vector<std::vector<std::size_t>>
build_neighbors(std::span<const Atom> atoms, double probe_radius,
                const SpatialBins &bins) {
  std::vector<std::vector<std::size_t>> neighbors(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const Vec3 ci = atom_center(atoms[i]);
    const CellKey cell = cell_for(bins, ci);
    const double ri = atoms[i].radius + probe_radius;
    for (int dz = -2; dz <= 2; ++dz) {
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
          const auto found =
              bins.cells.find({cell.x + dx, cell.y + dy, cell.z + dz});
          if (found == bins.cells.end())
            continue;
          for (const std::size_t j : found->second) {
            if (j <= i)
              continue;
            const double rj = atoms[j].radius + probe_radius;
            const double distance2 =
                norm_squared(subtract(atom_center(atoms[j]), ci));
            const double sum = ri + rj;
            if (distance2 <= sum * sum * (1.0 + kGeometryRelTol)) {
              neighbors[i].push_back(j);
              neighbors[j].push_back(i);
            }
          }
        }
      }
    }
  }
  for (auto &list : neighbors) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  return neighbors;
}

[[nodiscard]] bool contains_index(std::span<const std::size_t> values,
                                  std::size_t value) noexcept {
  return std::find(values.begin(), values.end(), value) != values.end();
}

[[nodiscard]] bool empty_probe_center(std::span<const Atom> atoms,
                                      double probe_radius, const Vec3 &center,
                                      std::span<const std::size_t> ignore) {
  for (std::size_t index = 0; index < atoms.size(); ++index) {
    if (contains_index(ignore, index))
      continue;
    const double radius = atoms[index].radius + probe_radius;
    const double tolerance = kGeometryRelTol * std::max(1.0, radius * radius);
    if (norm_squared(subtract(center, atom_center(atoms[index]))) <
        radius * radius - tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool
empty_probe_center_local(std::span<const Atom> atoms, double probe_radius,
                         const Vec3 &center,
                         std::span<const std::size_t> candidates,
                         std::span<const std::size_t> ignore) {
  for (const std::size_t index : candidates) {
    if (contains_index(ignore, index))
      continue;
    const double radius = atoms[index].radius + probe_radius;
    const double tolerance = kGeometryRelTol * std::max(1.0, radius * radius);
    if (norm_squared(subtract(center, atom_center(atoms[index]))) <
        radius * radius - tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Vec3 perpendicular_unit(const Vec3 &axis) noexcept {
  const Vec3 seed =
      std::abs(axis.x) < 0.75 ? Vec3{1.0, 0.0, 0.0} : Vec3{0.0, 1.0, 0.0};
  return normalized(cross(axis, seed));
}

[[nodiscard]] double normalize_angle(double angle) noexcept {
  const double period = 2.0 * std::numbers::pi;
  angle = std::fmod(angle, period);
  return angle < 0.0 ? angle + period : angle;
}

[[nodiscard]] Vec3 circle_point(const ToroidalPatch &patch,
                                double angle) noexcept {
  return add(
      patch.circle_center,
      scale(patch.circle_radius, add(scale(std::cos(angle), patch.basis_u),
                                     scale(std::sin(angle), patch.basis_v))));
}

[[nodiscard]] bool angle_in_interval(double angle,
                                     const AngularInterval &interval) noexcept {
  angle = normalize_angle(angle);
  return angle >= interval.begin - kAnalyticSesAngleTolerance &&
         angle <= interval.end + kAnalyticSesAngleTolerance;
}

[[nodiscard]] bool angle_in_exposed_intervals(
    double angle, std::span<const AngularInterval> intervals) noexcept {
  return std::any_of(intervals.begin(), intervals.end(),
                     [angle](const AngularInterval &interval) {
                       return angle_in_interval(angle, interval);
                     });
}

// The probe touches the two atoms along the minor geodesic between their
// contact directions.  Selecting the arc through beta=pi is only valid for
// equal (or nearly equal) expanded radii; for unequal radii both contacts can
// lie on the same side of pi and that rule selects the solvent-facing major
// arc instead.
[[nodiscard]] bool angle_on_minor_arc(double angle, double begin,
                                      double end) noexcept {
  angle = normalize_angle(angle);
  begin = normalize_angle(begin);
  end = normalize_angle(end);
  const auto ccw_distance = [](double from, double to) {
    const double period = 2.0 * std::numbers::pi;
    double value = to - from;
    if (value < 0.0)
      value += period;
    return value;
  };
  const double span = ccw_distance(begin, end);
  const bool angle_on_ccw =
      ccw_distance(begin, angle) <= span + kAnalyticSesAngleTolerance;
  return span <= std::numbers::pi ? angle_on_ccw : !angle_on_ccw;
}

[[nodiscard]] bool contact_point_is_exposed(const PatchComplex &complex,
                                            const ContactPatch &patch,
                                            const Vec3 &point) {
  const Vec3 radial = normalized(subtract(point, patch.center));
  const Vec3 probe_center =
      add(patch.center, scale(patch.expanded_radius, radial));
  const std::array<std::size_t, 1> ignore = {patch.atom};
  return empty_probe_center_local(complex.atoms, complex.probe_radius,
                                  probe_center,
                                  complex.atom_neighbors[patch.atom], ignore);
}

[[nodiscard]] bool spherical_triangle_contains(const ReentrantPatch &patch,
                                               const Vec3 &direction) noexcept {
  Vec3 anchor = normalized(
      add(add(patch.contact_directions[0], patch.contact_directions[1]),
          patch.contact_directions[2]));
  if (norm_squared(anchor) <= 1.0e-20)
    return false;
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const Vec3 &a = patch.contact_directions[edge];
    const Vec3 &b = patch.contact_directions[(edge + 1U) % 3U];
    const Vec3 plane = cross(a, b);
    const double reference = dot(plane, anchor);
    const double value = dot(plane, direction);
    if (reference >= 0.0 ? value < -2.0e-10 : value > 2.0e-10)
      return false;
  }
  return true;
}

void append_sphere_ray_roots(std::vector<RayHit> &hits, const Vec3 &center,
                             double radius, double y, double z, PatchKind kind,
                             std::size_t patch_index,
                             const std::function<bool(const Vec3 &)> &accept,
                             double normal_sign) {
  const double dy = y - center.y;
  const double dz = z - center.z;
  const double radial2 = dy * dy + dz * dz;
  const double radius2 = radius * radius;
  const double tolerance = 2.0e-12 * std::max(1.0, radius2);
  if (radial2 > radius2 + tolerance)
    return;
  const double dx = std::sqrt(std::max(0.0, radius2 - radial2));
  const std::array<double, 2> roots = {center.x - dx, center.x + dx};
  const std::size_t count = dx <= std::sqrt(tolerance) ? 1U : 2U;
  for (std::size_t side = 0; side < count; ++side) {
    const Vec3 point{roots[side], y, z};
    if (!accept(point))
      continue;
    const Vec3 radial = normalized(subtract(point, center));
    hits.push_back({point.x, scale(normal_sign, radial), kind, patch_index});
  }
}

[[nodiscard]] std::vector<AngularInterval>
exposed_circle_intervals(const ToroidalPatch &patch,
                         std::span<const Atom> atoms, double probe_radius,
                         std::span<const std::size_t> blockers) {
  const double period = 2.0 * std::numbers::pi;
  std::vector<double> events = {0.0, period};
  for (const std::size_t k : blockers) {
    if (k == patch.atom_i || k == patch.atom_j)
      continue;
    const Vec3 delta = subtract(patch.circle_center, atom_center(atoms[k]));
    const double radius = atoms[k].radius + probe_radius;
    const double a = 2.0 * patch.circle_radius * dot(delta, patch.basis_u);
    const double b = 2.0 * patch.circle_radius * dot(delta, patch.basis_v);
    const double c = norm_squared(delta) +
                     patch.circle_radius * patch.circle_radius -
                     radius * radius;
    const double amplitude = std::hypot(a, b);
    if (amplitude <= 1.0e-15)
      continue;
    const double ratio = std::clamp(-c / amplitude, -1.0, 1.0);
    if (std::abs(c) <= amplitude * (1.0 + kGeometryRelTol)) {
      const double phase = std::atan2(b, a);
      const double offset = std::acos(ratio);
      events.push_back(normalize_angle(phase - offset));
      events.push_back(normalize_angle(phase + offset));
    }
  }
  std::sort(events.begin(), events.end());
  events.erase(std::unique(events.begin(), events.end(),
                           [](double a, double b) {
                             return std::abs(a - b) <=
                                    kAnalyticSesAngleTolerance;
                           }),
               events.end());
  if (events.empty() || events.front() > kAnalyticSesAngleTolerance)
    events.insert(events.begin(), 0.0);
  if (events.back() < period - kAnalyticSesAngleTolerance)
    events.push_back(period);

  std::vector<AngularInterval> result;
  const std::array<std::size_t, 2> ignore = {patch.atom_i, patch.atom_j};
  for (std::size_t index = 0; index + 1 < events.size(); ++index) {
    const double begin = events[index];
    const double end = events[index + 1];
    if (end - begin <= kAnalyticSesAngleTolerance)
      continue;
    const double midpoint = 0.5 * (begin + end);
    if (empty_probe_center_local(atoms, probe_radius,
                                 circle_point(patch, midpoint), blockers,
                                 ignore)) {
      if (!result.empty() &&
          std::abs(result.back().end - begin) <= kAnalyticSesAngleTolerance)
        result.back().end = end;
      else
        result.push_back({begin, end});
    }
  }
  return result;
}

[[nodiscard]] std::array<Vec3, 2>
three_sphere_intersections(const Atom &atom_i, const Atom &atom_j,
                           const Atom &atom_k, double probe_radius, bool &valid,
                           bool &tangent) noexcept {
  valid = false;
  tangent = false;
  const Vec3 p1 = atom_center(atom_i);
  const Vec3 p2 = atom_center(atom_j);
  const Vec3 p3 = atom_center(atom_k);
  const double r1 = atom_i.radius + probe_radius;
  const double r2 = atom_j.radius + probe_radius;
  const double r3 = atom_k.radius + probe_radius;
  const Vec3 d12 = subtract(p2, p1);
  const double d = norm(d12);
  if (!(d > 1.0e-12))
    return {};
  const Vec3 ex = scale(1.0 / d, d12);
  const double i = dot(ex, subtract(p3, p1));
  const Vec3 p3_plane = subtract(subtract(p3, p1), scale(i, ex));
  const double p3_plane_norm = norm(p3_plane);
  if (!(p3_plane_norm > 1.0e-12))
    return {};
  const Vec3 ey = scale(1.0 / p3_plane_norm, p3_plane);
  const Vec3 ez = cross(ex, ey);
  const double j = dot(ey, subtract(p3, p1));
  if (!(std::abs(j) > 1.0e-12))
    return {};
  const double x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
  const double y =
      (r1 * r1 - r3 * r3 + i * i + j * j - 2.0 * i * x) / (2.0 * j);
  double z2 = r1 * r1 - x * x - y * y;
  const double tolerance =
      kGeometryRelTol * std::max({1.0, r1 * r1, r2 * r2, r3 * r3});
  if (z2 < -tolerance)
    return {};
  z2 = std::max(0.0, z2);
  const double z = std::sqrt(z2);
  const Vec3 base = add(p1, add(scale(x, ex), scale(y, ey)));
  valid = true;
  tangent = z <= std::sqrt(tolerance);
  return {add(base, scale(z, ez)), subtract(base, scale(z, ez))};
}

void build_ray_buckets(PatchComplex &complex) {
  double lower_y = std::numeric_limits<double>::infinity();
  double lower_z = std::numeric_limits<double>::infinity();
  double upper_y = -std::numeric_limits<double>::infinity();
  double upper_z = -std::numeric_limits<double>::infinity();
  double max_expanded_radius = 0.0;
  for (const ContactPatch &patch : complex.contacts) {
    lower_y = std::min(lower_y, patch.center.y - patch.atom_radius);
    lower_z = std::min(lower_z, patch.center.z - patch.atom_radius);
    upper_y = std::max(upper_y, patch.center.y + patch.atom_radius);
    upper_z = std::max(upper_z, patch.center.z + patch.atom_radius);
    max_expanded_radius = std::max(max_expanded_radius, patch.expanded_radius);
  }
  for (const ToroidalPatch &patch : complex.tori) {
    const double extent_y =
        patch.circle_radius *
            std::sqrt(std::max(0.0, 1.0 - patch.axis.y * patch.axis.y)) +
        complex.probe_radius;
    const double extent_z =
        patch.circle_radius *
            std::sqrt(std::max(0.0, 1.0 - patch.axis.z * patch.axis.z)) +
        complex.probe_radius;
    lower_y = std::min(lower_y, patch.circle_center.y - extent_y);
    lower_z = std::min(lower_z, patch.circle_center.z - extent_z);
    upper_y = std::max(upper_y, patch.circle_center.y + extent_y);
    upper_z = std::max(upper_z, patch.circle_center.z + extent_z);
  }
  for (const ReentrantPatch &patch : complex.reentrant) {
    lower_y = std::min(lower_y, patch.probe_center.y - complex.probe_radius);
    lower_z = std::min(lower_z, patch.probe_center.z - complex.probe_radius);
    upper_y = std::max(upper_y, patch.probe_center.y + complex.probe_radius);
    upper_z = std::max(upper_z, patch.probe_center.z + complex.probe_radius);
  }
  complex.ray_bucket_size = std::max(1.0e-6, max_expanded_radius);
  const double margin = 1.0e-9 * complex.ray_bucket_size;
  complex.ray_bucket_origin = {0.0, lower_y - margin, lower_z - margin};
  complex.ray_bucket_ny =
      static_cast<std::size_t>(std::floor((upper_y - lower_y + 2.0 * margin) /
                                          complex.ray_bucket_size)) +
      1U;
  complex.ray_bucket_nz =
      static_cast<std::size_t>(std::floor((upper_z - lower_z + 2.0 * margin) /
                                          complex.ray_bucket_size)) +
      1U;
  if (complex.ray_bucket_ny == 0 || complex.ray_bucket_nz == 0 ||
      complex.ray_bucket_ny >
          std::numeric_limits<std::size_t>::max() / complex.ray_bucket_nz) {
    return;
  }
  complex.ray_buckets.resize(complex.ray_bucket_ny * complex.ray_bucket_nz);
  const auto insert_bounds = [&](double y0, double y1, double z0, double z1,
                                 std::size_t index, auto member) {
    const auto coordinate = [](double value, double origin, double cell,
                               std::size_t count) {
      const double raw = std::floor((value - origin) / cell);
      if (raw <= 0.0)
        return std::size_t{0};
      if (raw >= static_cast<double>(count - 1U))
        return count - 1U;
      return static_cast<std::size_t>(raw);
    };
    const std::size_t iy0 =
        coordinate(y0, complex.ray_bucket_origin.y, complex.ray_bucket_size,
                   complex.ray_bucket_ny);
    const std::size_t iy1 =
        coordinate(y1, complex.ray_bucket_origin.y, complex.ray_bucket_size,
                   complex.ray_bucket_ny);
    const std::size_t iz0 =
        coordinate(z0, complex.ray_bucket_origin.z, complex.ray_bucket_size,
                   complex.ray_bucket_nz);
    const std::size_t iz1 =
        coordinate(z1, complex.ray_bucket_origin.z, complex.ray_bucket_size,
                   complex.ray_bucket_nz);
    for (std::size_t iz = iz0; iz <= iz1; ++iz) {
      for (std::size_t iy = iy0; iy <= iy1; ++iy) {
        (complex.ray_buckets[iy + complex.ray_bucket_ny * iz].*member)
            .push_back(index);
      }
    }
  };
  for (std::size_t index = 0; index < complex.contacts.size(); ++index) {
    const ContactPatch &patch = complex.contacts[index];
    insert_bounds(
        patch.center.y - patch.atom_radius, patch.center.y + patch.atom_radius,
        patch.center.z - patch.atom_radius, patch.center.z + patch.atom_radius,
        index, &RayBucket::contacts);
  }
  for (std::size_t index = 0; index < complex.tori.size(); ++index) {
    const ToroidalPatch &patch = complex.tori[index];
    const double extent_y =
        patch.circle_radius *
            std::sqrt(std::max(0.0, 1.0 - patch.axis.y * patch.axis.y)) +
        complex.probe_radius;
    const double extent_z =
        patch.circle_radius *
            std::sqrt(std::max(0.0, 1.0 - patch.axis.z * patch.axis.z)) +
        complex.probe_radius;
    insert_bounds(patch.circle_center.y - extent_y,
                  patch.circle_center.y + extent_y,
                  patch.circle_center.z - extent_z,
                  patch.circle_center.z + extent_z, index, &RayBucket::tori);
  }
  for (std::size_t index = 0; index < complex.reentrant.size(); ++index) {
    const ReentrantPatch &patch = complex.reentrant[index];
    insert_bounds(patch.probe_center.y - complex.probe_radius,
                  patch.probe_center.y + complex.probe_radius,
                  patch.probe_center.z - complex.probe_radius,
                  patch.probe_center.z + complex.probe_radius, index,
                  &RayBucket::reentrant);
  }
}

} // namespace

bool probe_center_is_empty(const PatchComplex &complex, const Vec3 &center,
                           std::span<const std::size_t> ignore) {
  return empty_probe_center(complex.atoms, complex.probe_radius, center,
                            ignore);
}

PatchComplex build_patch_complex(std::span<const Atom> input_atoms,
                                 double probe_radius, TitanExecutor *executor,
                                 bool build_ray_index) {
  PatchComplex complex;
  complex.probe_radius = probe_radius;
  if (!(probe_radius > 0.0) || !std::isfinite(probe_radius)) {
    complex.status = "invalid_probe_radius";
    return complex;
  }
  complex.atoms.reserve(input_atoms.size());
  for (const Atom &atom : input_atoms) {
    if (std::isfinite(atom.x) && std::isfinite(atom.y) &&
        std::isfinite(atom.z) && std::isfinite(atom.radius) &&
        atom.radius > 0.0) {
      complex.atoms.push_back(atom);
    }
  }
  if (complex.atoms.empty()) {
    complex.status = "no_valid_atoms";
    return complex;
  }

  const SpatialBins bins = build_bins(complex.atoms, probe_radius);
  complex.atom_neighbors = build_neighbors(complex.atoms, probe_radius, bins);
  const auto &neighbors = complex.atom_neighbors;
  complex.contacts.reserve(complex.atoms.size());
  for (std::size_t i = 0; i < complex.atoms.size(); ++i) {
    complex.contacts.push_back({i, atom_center(complex.atoms[i]),
                                complex.atoms[i].radius,
                                complex.atoms[i].radius + probe_radius});
  }

  struct PatchSlab {
    std::vector<ToroidalPatch> tori;
    std::vector<ReentrantPatch> reentrant;
    std::size_t candidate_pairs{0};
    std::size_t candidate_triples{0};
    std::size_t buried_pair_circles{0};
    std::size_t rejected_probe_centres{0};
  };
  const std::size_t workers =
      executor == nullptr
          ? 1U
          : std::min(executor->capacity(), complex.atoms.size());
  std::vector<PatchSlab> slabs(workers);
  const auto build_slab = [&](std::size_t worker, std::size_t begin_atom,
                              std::size_t end_atom) {
    PatchSlab &slab = slabs[worker];
    slab.tori.reserve((end_atom - begin_atom) * 6U);
    slab.reentrant.reserve((end_atom - begin_atom) * 4U);
    for (std::size_t i = begin_atom; i < end_atom; ++i) {
      const Vec3 ci = atom_center(complex.atoms[i]);
      const double ri = complex.atoms[i].radius + probe_radius;
      for (const std::size_t j : neighbors[i]) {
        if (j <= i)
          continue;
        ++slab.candidate_pairs;
        const Vec3 cj = atom_center(complex.atoms[j]);
        const double rj = complex.atoms[j].radius + probe_radius;
        const Vec3 delta = subtract(cj, ci);
        const double distance = norm(delta);
        const double tolerance =
            kGeometryRelTol * std::max({1.0, ri, rj, distance});
        if (!(distance > std::abs(ri - rj) + tolerance) ||
            !(distance < ri + rj - tolerance)) {
          continue;
        }
        const Vec3 axis = scale(1.0 / distance, delta);
        const double offset =
            (ri * ri - rj * rj + distance * distance) / (2.0 * distance);
        const double circle_radius2 = ri * ri - offset * offset;
        if (!(circle_radius2 > tolerance * tolerance))
          continue;
        ToroidalPatch patch;
        patch.atom_i = i;
        patch.atom_j = j;
        patch.axis = axis;
        patch.basis_u = perpendicular_unit(axis);
        patch.basis_v = cross(axis, patch.basis_u);
        patch.circle_center = add(ci, scale(offset, axis));
        patch.circle_radius = std::sqrt(circle_radius2);
        patch.axis_offset_i = offset;
        patch.expanded_radius_i = ri;
        patch.expanded_radius_j = rj;
        std::vector<std::size_t> blockers = neighbors[i];
        blockers.insert(blockers.end(), neighbors[j].begin(),
                        neighbors[j].end());
        std::sort(blockers.begin(), blockers.end());
        blockers.erase(std::unique(blockers.begin(), blockers.end()),
                       blockers.end());
        patch.exposed_intervals = exposed_circle_intervals(
            patch, complex.atoms, probe_radius, blockers);
        if (patch.exposed_intervals.empty()) {
          ++slab.buried_pair_circles;
        } else {
          slab.tori.push_back(std::move(patch));
        }

        std::vector<std::size_t> common;
        std::set_intersection(neighbors[i].begin(), neighbors[i].end(),
                              neighbors[j].begin(), neighbors[j].end(),
                              std::back_inserter(common));
        for (const std::size_t k : common) {
          if (k <= j)
            continue;
          ++slab.candidate_triples;
          bool valid = false;
          bool tangent = false;
          const auto centers = three_sphere_intersections(
              complex.atoms[i], complex.atoms[j], complex.atoms[k],
              probe_radius, valid, tangent);
          if (!valid)
            continue;
          const std::array<std::size_t, 3> triple = {i, j, k};
          const std::size_t count = tangent ? 1U : 2U;
          for (std::size_t side = 0; side < count; ++side) {
            const Vec3 center = centers[side];
            if (!empty_probe_center_local(complex.atoms, probe_radius, center,
                                          neighbors[i], triple)) {
              ++slab.rejected_probe_centres;
              continue;
            }
            ReentrantPatch patch;
            patch.atoms = triple;
            patch.probe_center = center;
            for (std::size_t q = 0; q < 3; ++q) {
              const std::size_t atom_index = triple[q];
              patch.contact_directions[q] = normalized(
                  subtract(atom_center(complex.atoms[atom_index]), center));
            }
            slab.reentrant.push_back(patch);
          }
        }
      }
    }
  };
  if (executor != nullptr && workers > 1U)
    executor->parallel_for(complex.atoms.size(), workers, 8U, build_slab);
  else
    build_slab(0U, 0U, complex.atoms.size());

  std::size_t torus_count = 0U;
  std::size_t reentrant_count = 0U;
  for (const PatchSlab &slab : slabs) {
    complex.candidate_pairs += slab.candidate_pairs;
    complex.candidate_triples += slab.candidate_triples;
    complex.buried_pair_circles += slab.buried_pair_circles;
    complex.rejected_probe_centres += slab.rejected_probe_centres;
    torus_count += slab.tori.size();
    reentrant_count += slab.reentrant.size();
  }
  complex.tori.reserve(torus_count);
  complex.reentrant.reserve(reentrant_count);
  for (PatchSlab &slab : slabs) {
    for (ToroidalPatch &patch : slab.tori)
      complex.tori.push_back(std::move(patch));
    // Different atom triples that share one probe centre describe distinct
    // faces of a higher-order spherical polygon. Keeping only the first
    // quantized centre silently removed valid reentrant area for four-atom
    // and near-degenerate contacts.
    for (ReentrantPatch &patch : slab.reentrant)
      complex.reentrant.push_back(std::move(patch));
  }
  if (build_ray_index)
    build_ray_buckets(complex);
  complex.status = "pass";
  return complex;
}

namespace {

std::vector<RayHit> intersect_ray_x_bucket(const PatchComplex &complex,
                                           const RayBucket &bucket, double y,
                                           double z) {
  std::vector<RayHit> hits;
  if (complex.status != "pass")
    return hits;
  const auto covered_by_reentrant_probe = [&](const Vec3 &point,
                                              std::size_t ignore) {
    const double radius2 = complex.probe_radius * complex.probe_radius;
    const double tolerance = 2.0e-11 * std::max(1.0, radius2);
    for (const std::size_t probe_index : bucket.reentrant) {
      if (probe_index == ignore)
        continue;
      if (norm_squared(
              subtract(point, complex.reentrant[probe_index].probe_center)) <
          radius2 - tolerance)
        return true;
    }
    return false;
  };

  for (const std::size_t index : bucket.contacts) {
    const ContactPatch &patch = complex.contacts[index];
    append_sphere_ray_roots(
        hits, patch.center, patch.atom_radius, y, z, PatchKind::contact, index,
        [&](const Vec3 &point) {
          return contact_point_is_exposed(complex, patch, point);
        },
        1.0);
  }

  for (const std::size_t index : bucket.tori) {
    const ToroidalPatch &patch = complex.tori[index];
    // Form the quartic in a coordinate t local to the torus centre.  Using the
    // world x coordinate here used to inflate coefficient dynamic range under
    // a rigid translation and changed the number of reported intersections.
    const double center_coordinate = patch.circle_center.x;
    const Vec3 q0{0.0, y - patch.circle_center.y, z - patch.circle_center.z};
    const double axial0 = dot(q0, patch.axis);
    const double q0_squared = norm_squared(q0);
    const double major2 = patch.circle_radius * patch.circle_radius;
    const double minor2 = complex.probe_radius * complex.probe_radius;
    const double p0 = q0_squared + major2 - minor2;
    const double p1 = 0.0;
    const double u0 = q0_squared - axial0 * axial0;
    const double u1 = 2.0 * (q0.x - patch.axis.x * axial0);
    const double u2 = 1.0 - patch.axis.x * patch.axis.x;
    const std::array<double, 5> coefficients = {
        p0 * p0 - 4.0 * major2 * u0, 2.0 * p0 * p1 - 4.0 * major2 * u1,
        p1 * p1 + 2.0 * p0 - 4.0 * major2 * u2, 2.0 * p1, 1.0};
    const QuarticRoots roots = solve_real_quartic(coefficients);
    if (!roots.certified)
      continue;
    for (std::size_t root = 0; root < roots.count; ++root) {
      const double x = center_coordinate + roots.values[root];
      const Vec3 point{x, y, z};
      const Vec3 q = subtract(point, patch.circle_center);
      const double axial = dot(q, patch.axis);
      const Vec3 radial_vector = subtract(q, scale(axial, patch.axis));
      const double radial_length = norm(radial_vector);
      if (!(radial_length > 1.0e-12))
        continue;
      const Vec3 radial = scale(1.0 / radial_length, radial_vector);
      const double theta = normalize_angle(
          std::atan2(dot(radial, patch.basis_v), dot(radial, patch.basis_u)));
      if (!angle_in_exposed_intervals(theta, patch.exposed_intervals))
        continue;
      const Vec3 probe_center =
          add(patch.circle_center, scale(patch.circle_radius, radial));
      const Vec3 tube_vector = subtract(point, probe_center);
      const double tube_length = norm(tube_vector);
      const double tube_tolerance =
          2.0e-9 * std::max({1.0, patch.circle_radius, complex.probe_radius});
      // The polynomial implicitization contains the self-intersecting inner
      // sheet of a spindle torus (major radius < probe radius).  It is not the
      // envelope generated by the closest probe-centre circle point.  Do not
      // turn that algebraic branch into a seemingly valid surface normal by
      // normalizing it blindly.
      if (std::abs(tube_length - complex.probe_radius) > tube_tolerance)
        continue;
      const Vec3 tube_direction = scale(1.0 / tube_length, tube_vector);
      const Atom &atom_i = complex.atoms[patch.atom_i];
      const Atom &atom_j = complex.atoms[patch.atom_j];
      const double center_distance =
          norm(subtract(atom_center(atom_j), atom_center(atom_i)));
      const double beta = normalize_angle(std::atan2(
          dot(tube_direction, patch.axis), dot(tube_direction, radial)));
      const double beta_i = normalize_angle(
          std::atan2(-patch.axis_offset_i / patch.expanded_radius_i,
                     -patch.circle_radius / patch.expanded_radius_i));
      const double beta_j = normalize_angle(std::atan2(
          (center_distance - patch.axis_offset_i) / patch.expanded_radius_j,
          -patch.circle_radius / patch.expanded_radius_j));
      if (!angle_on_minor_arc(beta, beta_i, beta_j))
        continue;
      if (covered_by_reentrant_probe(point,
                                     std::numeric_limits<std::size_t>::max()))
        continue;
      // The solvent lies on the probe-centre side of a reentrant patch.
      const Vec3 outward = scale(-1.0, tube_direction);
      hits.push_back({x, outward, PatchKind::toroidal, index});
    }
  }

  for (const std::size_t index : bucket.reentrant) {
    const ReentrantPatch &patch = complex.reentrant[index];
    append_sphere_ray_roots(
        hits, patch.probe_center, complex.probe_radius, y, z,
        PatchKind::reentrant, index,
        [&](const Vec3 &point) {
          return spherical_triangle_contains(
                     patch, normalized(subtract(point, patch.probe_center))) &&
                 !covered_by_reentrant_probe(point, index);
        },
        -1.0);
  }

  std::sort(hits.begin(), hits.end(), [](const RayHit &a, const RayHit &b) {
    if (a.x != b.x)
      return a.x < b.x;
    return static_cast<int>(a.kind) < static_cast<int>(b.kind);
  });
  std::vector<RayHit> deduplicated;
  for (const RayHit &hit : hits) {
    if (!deduplicated.empty() &&
        std::abs(hit.x - deduplicated.back().x) <=
            2.0e-8 * std::max({1.0, std::abs(hit.x),
                               std::abs(deduplicated.back().x)})) {
      RayHit &kept = deduplicated.back();
      const Vec3 normal_sum = add(kept.normal, hit.normal);
      if (norm_squared(normal_sum) > 1.0e-20)
        kept.normal = normalized(normal_sum);
      continue;
    }
    deduplicated.push_back(hit);
  }
  return deduplicated;
}

} // namespace

std::vector<RayHit> intersect_ray_x(const PatchComplex &complex, double y,
                                    double z) {
  if (complex.status != "pass" || complex.ray_buckets.empty() ||
      y < complex.ray_bucket_origin.y || z < complex.ray_bucket_origin.z)
    return {};
  const std::size_t iy = static_cast<std::size_t>(
      std::floor((y - complex.ray_bucket_origin.y) / complex.ray_bucket_size));
  const std::size_t iz = static_cast<std::size_t>(
      std::floor((z - complex.ray_bucket_origin.z) / complex.ray_bucket_size));
  if (iy >= complex.ray_bucket_ny || iz >= complex.ray_bucket_nz)
    return {};
  return intersect_ray_x_bucket(
      complex, complex.ray_buckets[iy + complex.ray_bucket_ny * iz], y, z);
}

} // namespace TitanPBE::Tabi::analytic_ses

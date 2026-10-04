#include "tabi_ses_torus_interval_bounds.h"

#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

struct IndexRange {
  std::size_t begin{0U};
  std::size_t end{0U};
};

struct BoundRectangle {
  PatchRefId ref;
  PatchBounds bounds;
};

struct RowCandidateEvent {
  std::uint8_t axis{0U};
  std::size_t row{0U};
  PatchRefId ref;
};

struct BuildFailure {
  std::string_view status;
};

[[nodiscard]] bool checked_add(std::size_t &value,
                               std::size_t increment) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - increment)
    return false;
  value += increment;
  return true;
}

[[nodiscard]] bool checked_multiply(std::size_t left, std::size_t right,
                                    std::size_t &result) noexcept {
  result = 0U;
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_payload(std::size_t capacity, std::size_t width,
                                   std::size_t &bytes) noexcept {
  std::size_t payload = 0U;
  return checked_multiply(capacity, width, payload) &&
         checked_add(bytes, payload);
}

[[nodiscard]] bool finite_bounds(const PatchBounds &bounds) noexcept {
  return std::isfinite(bounds.lower.x) && std::isfinite(bounds.lower.y) &&
         std::isfinite(bounds.lower.z) && std::isfinite(bounds.upper.x) &&
         std::isfinite(bounds.upper.y) && std::isfinite(bounds.upper.z) &&
         bounds.lower.x <= bounds.upper.x && bounds.lower.y <= bounds.upper.y &&
         bounds.lower.z <= bounds.upper.z;
}

[[nodiscard]] bool pad_bound(double &lower, double &upper, double spacing,
                             double factor) noexcept {
  const double extent = std::abs(upper - lower);
  const double scale =
      std::max({1.0, std::abs(lower), std::abs(upper), extent, spacing});
  const double padding =
      factor * std::numeric_limits<double>::epsilon() * scale;
  if (!std::isfinite(padding))
    return false;
  lower =
      std::nextafter(lower - padding, -std::numeric_limits<double>::infinity());
  upper =
      std::nextafter(upper + padding, std::numeric_limits<double>::infinity());
  return std::isfinite(lower) && std::isfinite(upper) && lower <= upper;
}

[[nodiscard]] bool full_worklist_bound(const PatchComplex &complex,
                                       PatchKind kind, std::size_t patch_index,
                                       double spacing,
                                       PatchBounds &bounds) noexcept {
  bounds = outward_padded_patch_bounds(complex, kind, patch_index);
  return finite_bounds(bounds) &&
         pad_bound(bounds.lower.x, bounds.upper.x, spacing, 128.0) &&
         pad_bound(bounds.lower.y, bounds.upper.y, spacing, 128.0) &&
         pad_bound(bounds.lower.z, bounds.upper.z, spacing, 128.0);
}

[[nodiscard]] bool finite_torus_carrier(const PatchComplex &complex,
                                        const ToroidalPatch &patch) noexcept {
  const auto finite = [](const Vec3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
  };
  return std::isfinite(complex.probe_radius) && complex.probe_radius > 0.0 &&
         patch.atom_i < complex.atoms.size() &&
         patch.atom_j < complex.atoms.size() && patch.atom_i != patch.atom_j &&
         finite(patch.circle_center) && finite(patch.axis) &&
         finite(patch.basis_u) && finite(patch.basis_v) &&
         std::isfinite(patch.circle_radius) && patch.circle_radius > 0.0 &&
         std::isfinite(patch.axis_offset_i) &&
         std::isfinite(patch.expanded_radius_i) &&
         std::isfinite(patch.expanded_radius_j) &&
         patch.expanded_radius_i > 0.0 && patch.expanded_radius_j > 0.0;
}

[[nodiscard]] double dot(const Vec3 &left, const Vec3 &right) noexcept {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

[[nodiscard]] bool
torus_frame_matches_domain_contract(const ToroidalPatch &patch) noexcept {
  // This mirrors the bounded-domain frame checks used by the projection
  // index. If that contract drifts, the parity fixture must fail rather than
  // silently enabling theta-dependent pruning for a different frame.
  return std::abs(dot(patch.axis, patch.axis) - 1.0) <= 2.0e-10 &&
         std::abs(dot(patch.basis_u, patch.basis_u) - 1.0) <= 2.0e-10 &&
         std::abs(dot(patch.basis_v, patch.basis_v) - 1.0) <= 2.0e-10 &&
         std::abs(dot(patch.axis, patch.basis_u)) <= 2.0e-10 &&
         std::abs(dot(patch.axis, patch.basis_v)) <= 2.0e-10 &&
         std::abs(dot(patch.basis_u, patch.basis_v)) <= 2.0e-10;
}

[[nodiscard]] bool valid_interval(const AngularInterval &interval) noexcept {
  const double period = 2.0 * std::numbers::pi;
  return std::isfinite(interval.begin) && std::isfinite(interval.end) &&
         interval.begin >= 0.0 && interval.end <= period &&
         interval.begin <= interval.end;
}

[[nodiscard]] bool
equivalent_angle_in_interval(double angle,
                             const AngularInterval &interval) noexcept {
  const double period = 2.0 * std::numbers::pi;
  const double scale = std::max(
      {1.0, std::abs(angle), std::abs(interval.begin), std::abs(interval.end)});
  const double tolerance =
      128.0 * std::numeric_limits<double>::epsilon() * scale;
  for (const double shifted : {angle - period, angle, angle + period}) {
    if (shifted >= interval.begin - tolerance &&
        shifted <= interval.end + tolerance)
      return true;
  }
  return false;
}

[[nodiscard]] bool
component_closed_interval_range(double basis_u, double basis_v,
                                const AngularInterval &interval,
                                double &minimum, double &maximum) noexcept {
  if (!std::isfinite(basis_u) || !std::isfinite(basis_v) ||
      !valid_interval(interval))
    return false;
  const auto evaluate = [&](double theta) {
    return basis_u * std::cos(theta) + basis_v * std::sin(theta);
  };
  const double begin = evaluate(interval.begin);
  const double end = evaluate(interval.end);
  if (!std::isfinite(begin) || !std::isfinite(end))
    return false;
  minimum = std::min(begin, end);
  maximum = std::max(begin, end);
  const double amplitude = std::hypot(basis_u, basis_v);
  if (!std::isfinite(amplitude))
    return false;
  if (amplitude == 0.0)
    return true;
  double maximum_angle = std::atan2(basis_v, basis_u);
  if (maximum_angle < 0.0)
    maximum_angle += 2.0 * std::numbers::pi;
  double minimum_angle = maximum_angle + std::numbers::pi;
  if (minimum_angle >= 2.0 * std::numbers::pi)
    minimum_angle -= 2.0 * std::numbers::pi;
  if (equivalent_angle_in_interval(maximum_angle, interval))
    maximum = amplitude;
  if (equivalent_angle_in_interval(minimum_angle, interval))
    minimum = -amplitude;
  return std::isfinite(minimum) && std::isfinite(maximum) && minimum <= maximum;
}

[[nodiscard]] bool component_interval_range(double basis_u, double basis_v,
                                            const AngularInterval &interval,
                                            double &minimum,
                                            double &maximum) noexcept {
  if (!valid_interval(interval))
    return false;
  const double period = 2.0 * std::numbers::pi;
  const double expanded_begin = interval.begin - kAnalyticSesAngleTolerance;
  const double expanded_end = interval.end + kAnalyticSesAngleTolerance;
  std::array<AngularInterval, 3> segments{};
  std::size_t segment_count = 0U;
  segments[segment_count++] = {std::max(0.0, expanded_begin),
                               std::min(period, expanded_end)};
  if (expanded_begin < 0.0)
    segments[segment_count++] = {period + expanded_begin, period};
  if (expanded_end > period)
    segments[segment_count++] = {0.0, expanded_end - period};
  minimum = std::numeric_limits<double>::infinity();
  maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t segment = 0U; segment < segment_count; ++segment) {
    double segment_minimum = 0.0;
    double segment_maximum = 0.0;
    if (!component_closed_interval_range(basis_u, basis_v, segments[segment],
                                         segment_minimum, segment_maximum))
      return false;
    minimum = std::min(minimum, segment_minimum);
    maximum = std::max(maximum, segment_maximum);
  }
  return std::isfinite(minimum) && std::isfinite(maximum) && minimum <= maximum;
}

[[nodiscard]] bool torus_interval_bound_impl(const PatchComplex &complex,
                                             std::size_t torus_index,
                                             const AngularInterval &interval,
                                             double spacing,
                                             PatchBounds &bounds) noexcept {
  if (torus_index >= complex.tori.size() || !std::isfinite(spacing) ||
      !(spacing > 0.0))
    return false;
  const ToroidalPatch &patch = complex.tori[torus_index];
  if (!finite_torus_carrier(complex, patch) ||
      !torus_frame_matches_domain_contract(patch) || !valid_interval(interval))
    return false;
  const std::array<double, 3> center = {
      patch.circle_center.x, patch.circle_center.y, patch.circle_center.z};
  const std::array<double, 3> basis_u = {patch.basis_u.x, patch.basis_u.y,
                                         patch.basis_u.z};
  const std::array<double, 3> basis_v = {patch.basis_v.x, patch.basis_v.y,
                                         patch.basis_v.z};
  const std::array<double, 3> axis = {patch.axis.x, patch.axis.y, patch.axis.z};
  std::array<double, 3> lower{};
  std::array<double, 3> upper{};
  for (std::size_t coordinate = 0U; coordinate < 3U; ++coordinate) {
    double radial_minimum = 0.0;
    double radial_maximum = 0.0;
    if (!component_interval_range(basis_u[coordinate], basis_v[coordinate],
                                  interval, radial_minimum, radial_maximum))
      return false;
    const double radial_amplitude =
        std::hypot(basis_u[coordinate], basis_v[coordinate]);
    const double minor_extent =
        complex.probe_radius * std::hypot(radial_amplitude, axis[coordinate]);
    lower[coordinate] = center[coordinate] +
                        patch.circle_radius * radial_minimum - minor_extent;
    upper[coordinate] = center[coordinate] +
                        patch.circle_radius * radial_maximum + minor_extent;
    if (!std::isfinite(lower[coordinate]) ||
        !std::isfinite(upper[coordinate]) ||
        !pad_bound(lower[coordinate], upper[coordinate], spacing, 256.0))
      return false;
  }
  bounds = {{lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}};
  PatchBounds full;
  if (!full_worklist_bound(complex, PatchKind::toroidal, torus_index, spacing,
                           full))
    return false;
  bounds.lower.x = std::max(bounds.lower.x, full.lower.x);
  bounds.lower.y = std::max(bounds.lower.y, full.lower.y);
  bounds.lower.z = std::max(bounds.lower.z, full.lower.z);
  bounds.upper.x = std::min(bounds.upper.x, full.upper.x);
  bounds.upper.y = std::min(bounds.upper.y, full.upper.y);
  bounds.upper.z = std::min(bounds.upper.z, full.upper.z);
  return finite_bounds(bounds);
}

[[nodiscard]] bool axis_grid(const PatchMappingGridSpec &spec, RayAxis axis,
                             double &lower_u, std::size_t &u_count,
                             double &lower_v, std::size_t &v_count) noexcept {
  switch (axis) {
  case RayAxis::x:
    lower_u = spec.lower.y;
    u_count = spec.ny;
    lower_v = spec.lower.z;
    v_count = spec.nz;
    return true;
  case RayAxis::y:
    lower_u = spec.lower.x;
    u_count = spec.nx;
    lower_v = spec.lower.z;
    v_count = spec.nz;
    return true;
  case RayAxis::z:
    lower_u = spec.lower.x;
    u_count = spec.nx;
    lower_v = spec.lower.y;
    v_count = spec.ny;
    return true;
  }
  return false;
}

[[nodiscard]] bool projected_bounds(const PatchBounds &bounds, RayAxis axis,
                                    double &lower_u, double &upper_u,
                                    double &lower_v, double &upper_v) noexcept {
  switch (axis) {
  case RayAxis::x:
    lower_u = bounds.lower.y;
    upper_u = bounds.upper.y;
    lower_v = bounds.lower.z;
    upper_v = bounds.upper.z;
    return true;
  case RayAxis::y:
    lower_u = bounds.lower.x;
    upper_u = bounds.upper.x;
    lower_v = bounds.lower.z;
    upper_v = bounds.upper.z;
    return true;
  case RayAxis::z:
    lower_u = bounds.lower.x;
    upper_u = bounds.upper.x;
    lower_v = bounds.lower.y;
    upper_v = bounds.upper.y;
    return true;
  }
  return false;
}

[[nodiscard]] bool grid_coordinate(double lower, double spacing,
                                   std::size_t index,
                                   double &coordinate) noexcept {
  coordinate = lower + spacing * static_cast<double>(index);
  return std::isfinite(coordinate);
}

[[nodiscard]] bool grid_index_range(double lower, double spacing,
                                    std::size_t count, double bound_lower,
                                    double bound_upper,
                                    IndexRange &range) noexcept {
  range = {};
  if (count == 0U || !std::isfinite(lower) || !std::isfinite(spacing) ||
      !(spacing > 0.0) || !std::isfinite(bound_lower) ||
      !std::isfinite(bound_upper) || bound_lower > bound_upper)
    return false;
  const auto first_not_less = [&](double bound, bool strict) {
    std::size_t begin = 0U;
    std::size_t end = count;
    while (begin < end) {
      const std::size_t middle = begin + (end - begin) / 2U;
      double coordinate = 0.0;
      if (!grid_coordinate(lower, spacing, middle, coordinate))
        return count;
      const bool before = strict ? coordinate <= bound : coordinate < bound;
      if (before)
        begin = middle + 1U;
      else
        end = middle;
    }
    return begin;
  };
  range.begin = first_not_less(bound_lower, false);
  range.end = first_not_less(bound_upper, true);
  return range.begin <= range.end && range.end <= count;
}

[[nodiscard]] bool event_less(const RowCandidateEvent &left,
                              const RowCandidateEvent &right) noexcept {
  if (left.axis != right.axis)
    return left.axis < right.axis;
  if (left.row != right.row)
    return left.row < right.row;
  return left.ref < right.ref;
}

[[nodiscard]] bool event_equal(const RowCandidateEvent &left,
                               const RowCandidateEvent &right) noexcept {
  return left.axis == right.axis && left.row == right.row &&
         left.ref == right.ref;
}

[[nodiscard]] bool patch_ref_in_range(const PatchComplex &complex,
                                      PatchRefId ref) noexcept {
  if (!ref.valid())
    return false;
  switch (ref.kind()) {
  case PatchKind::contact:
    return ref.index() < complex.contacts.size();
  case PatchKind::toroidal:
    return ref.index() < complex.tori.size();
  case PatchKind::reentrant:
    return ref.index() < complex.reentrant.size();
  }
  return false;
}

[[nodiscard]] bool
retained_payload_bytes(const PreparedTorusIntervalRowWorklist &prepared,
                       std::size_t &bytes) noexcept {
  bytes = 0U;
  for (const PatchRayAxisRowWorklist &axis : prepared.axes) {
    if (!checked_payload(axis.offsets.capacity(), sizeof(std::size_t), bytes) ||
        !checked_payload(axis.candidates.capacity(), sizeof(PatchRefId), bytes))
      return false;
  }
  return checked_payload(prepared.torus_interval_offsets.capacity(),
                         sizeof(std::size_t), bytes) &&
         checked_payload(prepared.torus_interval_bounds.capacity(),
                         sizeof(TorusExposedIntervalBound), bytes) &&
         checked_payload(prepared.torus_fallback_reasons.capacity(),
                         sizeof(TorusIntervalFallbackReason), bytes);
}

template <typename T>
[[nodiscard]] bool add_vector_payload(const std::vector<T> &values,
                                      std::size_t &bytes) noexcept {
  return checked_payload(values.capacity(), sizeof(T), bytes);
}

} // namespace

bool torus_exposed_interval_bound_for_audit(const PatchComplex &complex,
                                            std::size_t torus_index,
                                            std::size_t interval_index,
                                            double grid_spacing,
                                            PatchBounds &bounds) noexcept {
  bounds = {};
  return torus_index < complex.tori.size() &&
         interval_index < complex.tori[torus_index].exposed_intervals.size() &&
         torus_interval_bound_impl(
             complex, torus_index,
             complex.tori[torus_index].exposed_intervals[interval_index],
             grid_spacing, bounds);
}

bool torus_exposed_theta_component_range_for_audit(
    double basis_u_component, double basis_v_component,
    const AngularInterval &interval, double &minimum,
    double &maximum) noexcept {
  minimum = 0.0;
  maximum = 0.0;
  return component_interval_range(basis_u_component, basis_v_component,
                                  interval, minimum, maximum);
}

PreparedTorusIntervalRowWorklist build_torus_interval_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor, TorusIntervalRowWorklistBuildOptions options) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  PreparedTorusIntervalRowWorklist result;
  const auto elapsed = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count();
  };
  const auto fail = [&](std::string_view status) {
    PreparedTorusIntervalRowWorklist failed;
    failed.status.assign(status);
    failed.build_ms = elapsed();
    return failed;
  };

  try {
    std::array<std::size_t, 3> row_counts{};
    if (complex.status != "pass" ||
        !validate_patch_mapping_grid_spec(spec, &row_counts, nullptr))
      return fail("invalid_input");
    result.complex_fingerprint = fingerprint_patch_complex(complex);
    result.lattice_identity = make_patch_ray_lattice_identity(
        spec.lower, spec.grid_spacing, spec.nx, spec.ny, spec.nz);
    if (!result.lattice_identity.valid)
      return fail("invalid_input");
    result.requested_workers = executor == nullptr ? 1U : executor->capacity();
    if (result.requested_workers == 0U)
      return fail("invalid_executor");

    std::size_t patch_count = 0U;
    if (!checked_add(patch_count, complex.contacts.size()) ||
        !checked_add(patch_count, complex.tori.size()) ||
        !checked_add(patch_count, complex.reentrant.size()))
      return fail("count_overflow");
    if (patch_count == 0U)
      return fail("invalid_patches");

    std::vector<TorusIntervalFallbackReason> plans(
        complex.tori.size(), TorusIntervalFallbackReason::none);
    std::vector<std::size_t> record_counts(complex.tori.size(), 0U);
    for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
      const ToroidalPatch &patch = complex.tori[torus];
      PatchBounds full;
      if (!finite_torus_carrier(complex, patch) ||
          !full_worklist_bound(complex, PatchKind::toroidal, torus,
                               spec.grid_spacing, full))
        return fail("invalid_patch_geometry");
      TorusIntervalFallbackReason reason = TorusIntervalFallbackReason::none;
      if (patch.exposed_intervals.empty()) {
        reason = TorusIntervalFallbackReason::empty_intervals;
      } else if (!torus_frame_matches_domain_contract(patch)) {
        reason = TorusIntervalFallbackReason::invalid_frame;
      } else if (patch.exposed_intervals.size() >
                 std::numeric_limits<std::uint32_t>::max()) {
        return fail("count_overflow");
      } else {
        for (const AngularInterval &interval : patch.exposed_intervals) {
          if (!valid_interval(interval)) {
            reason = TorusIntervalFallbackReason::invalid_interval;
            break;
          }
          PatchBounds sub_bound;
          if (!torus_interval_bound_impl(complex, torus, interval,
                                         spec.grid_spacing, sub_bound)) {
            reason = TorusIntervalFallbackReason::sub_bound_failure;
            break;
          }
        }
      }
      plans[torus] = reason;
      record_counts[torus] = reason == TorusIntervalFallbackReason::none
                                 ? patch.exposed_intervals.size()
                                 : 1U;
    }

    std::size_t offset_count = complex.tori.size();
    if (!checked_add(offset_count, 1U))
      return fail("count_overflow");
    result.torus_interval_offsets.resize(offset_count, 0U);
    for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
      std::size_t next = result.torus_interval_offsets[torus];
      if (!checked_add(next, record_counts[torus]))
        return fail("count_overflow");
      result.torus_interval_offsets[torus + 1U] = next;
    }
    result.torus_interval_bounds.resize(result.torus_interval_offsets.back());
    result.torus_fallback_reasons = plans;
    std::vector<std::uint8_t> bounds_workers(result.requested_workers, 0U);
    const auto bounds_started = Clock::now();
    const auto populate_torus = [&](std::size_t torus) {
      PatchRefId ref;
      if (!PatchRefId::try_make(PatchKind::toroidal, torus, ref))
        throw BuildFailure{"invalid_ref"};
      const std::size_t begin = result.torus_interval_offsets[torus];
      const TorusIntervalFallbackReason reason = plans[torus];
      if (reason != TorusIntervalFallbackReason::none) {
        PatchBounds full;
        if (!full_worklist_bound(complex, PatchKind::toroidal, torus,
                                 spec.grid_spacing, full))
          throw BuildFailure{"invalid_patch_geometry"};
        result.torus_interval_bounds[begin] = {
            ref, std::numeric_limits<std::uint32_t>::max(), full, true};
        return;
      }
      const ToroidalPatch &patch = complex.tori[torus];
      for (std::size_t interval = 0U; interval < patch.exposed_intervals.size();
           ++interval) {
        PatchBounds sub_bound;
        if (!torus_interval_bound_impl(complex, torus,
                                       patch.exposed_intervals[interval],
                                       spec.grid_spacing, sub_bound))
          throw BuildFailure{"sub_bound_failure"};
        result.torus_interval_bounds[begin + interval] = {
            ref, static_cast<std::uint32_t>(interval), sub_bound, false};
      }
    };
    if (executor != nullptr && result.requested_workers > 1U &&
        !complex.tori.empty()) {
      executor->parallel_for(
          complex.tori.size(), result.requested_workers, 1U,
          [&](std::size_t worker, std::size_t begin, std::size_t end) {
            if (worker >= bounds_workers.size())
              throw BuildFailure{"worker_range"};
            bounds_workers[worker] = 1U;
            for (std::size_t torus = begin; torus < end; ++torus)
              populate_torus(torus);
          });
    } else {
      bounds_workers[0] = 1U;
      for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus)
        populate_torus(torus);
    }
    result.bounds_workers_used = static_cast<std::size_t>(std::count(
        bounds_workers.begin(), bounds_workers.end(), std::uint8_t{1U}));
    if (result.bounds_workers_used == 0U)
      throw BuildFailure{"worker_range"};
    result.bounds_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - bounds_started)
            .count();

    for (std::size_t torus = 0U; torus < plans.size(); ++torus) {
      const TorusIntervalFallbackReason reason = plans[torus];
      const std::size_t value = static_cast<std::size_t>(reason);
      if (value >= result.fallback_reason_counts.size() ||
          !checked_add(result.fallback_reason_counts[value], 1U))
        throw BuildFailure{"count_overflow"};
      if (reason == TorusIntervalFallbackReason::none) {
        if (!checked_add(result.exposed_interval_rectangles,
                         record_counts[torus]))
          throw BuildFailure{"count_overflow"};
      } else if (!checked_add(result.fallback_tori, 1U)) {
        throw BuildFailure{"count_overflow"};
      }
    }

    std::vector<BoundRectangle> rectangles;
    std::size_t rectangle_count = complex.contacts.size();
    if (!checked_add(rectangle_count, result.torus_interval_bounds.size()) ||
        !checked_add(rectangle_count, complex.reentrant.size()))
      throw BuildFailure{"count_overflow"};
    rectangles.reserve(rectangle_count);
    const auto append_full = [&](PatchKind kind, std::size_t count) {
      for (std::size_t patch = 0U; patch < count; ++patch) {
        PatchRefId ref;
        PatchBounds bounds;
        if (!PatchRefId::try_make(kind, patch, ref))
          throw BuildFailure{"invalid_ref"};
        if (!full_worklist_bound(complex, kind, patch, spec.grid_spacing,
                                 bounds))
          throw BuildFailure{"invalid_patch_geometry"};
        rectangles.push_back({ref, bounds});
      }
    };
    append_full(PatchKind::contact, complex.contacts.size());
    for (const TorusExposedIntervalBound &bound : result.torus_interval_bounds)
      rectangles.push_back({bound.ref, bound.bounds});
    append_full(PatchKind::reentrant, complex.reentrant.size());
    if (rectangles.size() != rectangle_count)
      throw BuildFailure{"count_overflow"};

    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      result.axes[axis].axis = static_cast<RayAxis>(axis);
      double lower_u = 0.0;
      double lower_v = 0.0;
      if (!axis_grid(spec, result.axes[axis].axis, lower_u,
                     result.axes[axis].u_count, lower_v,
                     result.axes[axis].v_count) ||
          !checked_add(result.total_rows, row_counts[axis]))
        throw BuildFailure{"count_overflow"};
    }

    const auto rectangle_pair_count = [&](const BoundRectangle &rectangle,
                                          std::size_t &pairs) {
      pairs = 0U;
      for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
        const RayAxis axis = static_cast<RayAxis>(axis_index);
        double lower_u = 0.0;
        double lower_v = 0.0;
        std::size_t u_count = 0U;
        std::size_t v_count = 0U;
        double bound_lower_u = 0.0;
        double bound_upper_u = 0.0;
        double bound_lower_v = 0.0;
        double bound_upper_v = 0.0;
        IndexRange u;
        IndexRange v;
        if (!axis_grid(spec, axis, lower_u, u_count, lower_v, v_count) ||
            !projected_bounds(rectangle.bounds, axis, bound_lower_u,
                              bound_upper_u, bound_lower_v, bound_upper_v) ||
            !grid_index_range(lower_u, spec.grid_spacing, u_count,
                              bound_lower_u, bound_upper_u, u) ||
            !grid_index_range(lower_v, spec.grid_spacing, v_count,
                              bound_lower_v, bound_upper_v, v))
          throw BuildFailure{"invalid_patch_geometry"};
        std::size_t axis_pairs = 0U;
        if (!checked_multiply(u.end - u.begin, v.end - v.begin, axis_pairs) ||
            !checked_add(pairs, axis_pairs))
          throw BuildFailure{"count_overflow"};
      }
    };
    for (const BoundRectangle &rectangle : rectangles) {
      std::size_t pairs = 0U;
      rectangle_pair_count(rectangle, pairs);
      if (!checked_add(result.raw_rectangle_candidate_pairs, pairs))
        throw BuildFailure{"count_overflow"};
      if (result.raw_rectangle_candidate_pairs >
          options.raw_candidate_pair_limit)
        throw BuildFailure{"candidate_limit"};
    }

    const auto event_started = Clock::now();
    std::vector<RowCandidateEvent> events;
    events.reserve(result.raw_rectangle_candidate_pairs);
    const auto append_rectangle_events = [&](const BoundRectangle &rectangle) {
      for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
        const RayAxis axis = static_cast<RayAxis>(axis_index);
        double lower_u = 0.0;
        double lower_v = 0.0;
        std::size_t u_count = 0U;
        std::size_t v_count = 0U;
        double bound_lower_u = 0.0;
        double bound_upper_u = 0.0;
        double bound_lower_v = 0.0;
        double bound_upper_v = 0.0;
        IndexRange u;
        IndexRange v;
        if (!axis_grid(spec, axis, lower_u, u_count, lower_v, v_count) ||
            !projected_bounds(rectangle.bounds, axis, bound_lower_u,
                              bound_upper_u, bound_lower_v, bound_upper_v) ||
            !grid_index_range(lower_u, spec.grid_spacing, u_count,
                              bound_lower_u, bound_upper_u, u) ||
            !grid_index_range(lower_v, spec.grid_spacing, v_count,
                              bound_lower_v, bound_upper_v, v))
          throw BuildFailure{"invalid_patch_geometry"};
        for (std::size_t iv = v.begin; iv < v.end; ++iv) {
          for (std::size_t iu = u.begin; iu < u.end; ++iu) {
            std::size_t row = 0U;
            if (!checked_multiply(iv, u_count, row) || !checked_add(row, iu))
              throw BuildFailure{"count_overflow"};
            events.push_back(
                {static_cast<std::uint8_t>(axis_index), row, rectangle.ref});
          }
        }
      }
    };
    if (options.reverse_rectangle_schedule_for_audit) {
      for (auto rectangle = rectangles.rbegin(); rectangle != rectangles.rend();
           ++rectangle)
        append_rectangle_events(*rectangle);
    } else {
      for (const BoundRectangle &rectangle : rectangles)
        append_rectangle_events(rectangle);
    }
    if (events.size() != result.raw_rectangle_candidate_pairs)
      throw BuildFailure{"count_overflow"};
    result.event_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - event_started)
            .count();

    const auto sort_started = Clock::now();
    std::sort(events.begin(), events.end(), event_less);
    events.erase(std::unique(events.begin(), events.end(), event_equal),
                 events.end());
    result.candidate_pairs = events.size();
    std::array<std::vector<std::size_t>, 3> counts;
    for (std::size_t axis = 0U; axis < 3U; ++axis)
      counts[axis].assign(row_counts[axis], 0U);
    for (const RowCandidateEvent &event : events) {
      if (event.axis >= 3U || event.row >= counts[event.axis].size() ||
          !checked_add(counts[event.axis][event.row], 1U))
        throw BuildFailure{"count_overflow"};
    }
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      std::size_t sentinel = row_counts[axis];
      if (!checked_add(sentinel, 1U))
        throw BuildFailure{"count_overflow"};
      result.axes[axis].offsets.assign(sentinel, 0U);
      std::size_t prefix = 0U;
      for (std::size_t row = 0U; row < row_counts[axis]; ++row) {
        const std::size_t occupancy = counts[axis][row];
        if (occupancy != 0U && !checked_add(result.nonempty_rows, 1U))
          throw BuildFailure{"count_overflow"};
        result.maximum_row_occupancy =
            std::max(result.maximum_row_occupancy, occupancy);
        if (!checked_add(prefix, occupancy))
          throw BuildFailure{"count_overflow"};
        result.axes[axis].offsets[row + 1U] = prefix;
      }
      result.axes[axis].candidates.resize(prefix);
      for (std::size_t row = 0U; row < row_counts[axis]; ++row)
        counts[axis][row] = result.axes[axis].offsets[row];
    }
    for (const RowCandidateEvent &event : events) {
      const std::size_t position = counts[event.axis][event.row]++;
      if (position >= result.axes[event.axis].candidates.size())
        throw BuildFailure{"count_overflow"};
      result.axes[event.axis].candidates[position] = event.ref;
    }
    result.sort_prefix_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - sort_started)
            .count();

    std::size_t retained = 0U;
    if (!retained_payload_bytes(result, retained))
      throw BuildFailure{"count_overflow"};
    result.retained_bytes = retained;
    std::size_t peak = retained;
    if (!add_vector_payload(plans, peak) ||
        !add_vector_payload(record_counts, peak) ||
        !add_vector_payload(bounds_workers, peak) ||
        !add_vector_payload(rectangles, peak) ||
        !add_vector_payload(events, peak))
      throw BuildFailure{"count_overflow"};
    for (const auto &axis_counts : counts) {
      if (!add_vector_payload(axis_counts, peak))
        throw BuildFailure{"count_overflow"};
    }
    result.accounted_peak_temporary_bytes = peak;
    result.interval_worklist_complete = true;
    result.status = result.fallback_tori == 0U ? "pass" : "pass_with_fallback";
    result.build_ms = elapsed();
    return result;
  } catch (const std::bad_alloc &) {
    return fail("alloc_fail");
  } catch (const BuildFailure &failure) {
    return fail(failure.status);
  } catch (const std::length_error &) {
    return fail("count_overflow");
  }
}

bool validate_torus_interval_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTorusIntervalRowWorklist &prepared) noexcept {
  std::array<std::size_t, 3> row_counts{};
  const PatchComplexFingerprint fingerprint =
      fingerprint_patch_complex(complex);
  const PatchRayProvenance::LatticeIdentity lattice =
      make_patch_ray_lattice_identity(spec.lower, spec.grid_spacing, spec.nx,
                                      spec.ny, spec.nz);
  if (complex.status != "pass" ||
      !validate_patch_mapping_grid_spec(spec, &row_counts, nullptr) ||
      !lattice.valid || prepared.complex_fingerprint != fingerprint ||
      prepared.lattice_identity != lattice ||
      (prepared.status != "pass" && prepared.status != "pass_with_fallback") ||
      !prepared.interval_worklist_complete ||
      prepared.requested_workers == 0U || prepared.bounds_workers_used == 0U ||
      prepared.bounds_workers_used > prepared.requested_workers ||
      !std::isfinite(prepared.bounds_ms) || prepared.bounds_ms < 0.0 ||
      !std::isfinite(prepared.event_ms) || prepared.event_ms < 0.0 ||
      !std::isfinite(prepared.sort_prefix_ms) ||
      prepared.sort_prefix_ms < 0.0 || !std::isfinite(prepared.build_ms) ||
      prepared.build_ms < 0.0 ||
      prepared.torus_interval_offsets.size() != complex.tori.size() + 1U ||
      prepared.torus_fallback_reasons.size() != complex.tori.size() ||
      prepared.torus_interval_offsets.empty() ||
      prepared.torus_interval_offsets.front() != 0U ||
      prepared.torus_interval_offsets.back() !=
          prepared.torus_interval_bounds.size())
    return false;

  std::size_t fallback_tori = 0U;
  std::size_t exposed_rectangles = 0U;
  std::array<std::size_t, 5> reasons{};
  for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
    const std::size_t begin = prepared.torus_interval_offsets[torus];
    const std::size_t end = prepared.torus_interval_offsets[torus + 1U];
    if (begin > end || end > prepared.torus_interval_bounds.size() ||
        begin == end)
      return false;
    const TorusIntervalFallbackReason reason =
        prepared.torus_fallback_reasons[torus];
    const std::size_t reason_index = static_cast<std::size_t>(reason);
    if (reason_index >= reasons.size() ||
        !checked_add(reasons[reason_index], 1U))
      return false;
    PatchRefId expected;
    if (!PatchRefId::try_make(PatchKind::toroidal, torus, expected))
      return false;
    if (reason == TorusIntervalFallbackReason::none) {
      if (end - begin != complex.tori[torus].exposed_intervals.size() ||
          !checked_add(exposed_rectangles, end - begin))
        return false;
    } else {
      if (end - begin != 1U || !checked_add(fallback_tori, 1U))
        return false;
    }
    for (std::size_t position = begin; position < end; ++position) {
      const TorusExposedIntervalBound &bound =
          prepared.torus_interval_bounds[position];
      if (bound.ref != expected || !finite_bounds(bound.bounds) ||
          bound.fallback_full_torus !=
              (reason != TorusIntervalFallbackReason::none))
        return false;
      if (reason == TorusIntervalFallbackReason::none) {
        if (bound.interval_index != position - begin)
          return false;
      } else if (bound.interval_index !=
                 std::numeric_limits<std::uint32_t>::max()) {
        return false;
      }
    }
  }
  if (fallback_tori != prepared.fallback_tori ||
      exposed_rectangles != prepared.exposed_interval_rectangles ||
      reasons != prepared.fallback_reason_counts ||
      (fallback_tori == 0U) != (prepared.status == "pass"))
    return false;

  std::size_t total_rows = 0U;
  std::size_t candidate_pairs = 0U;
  std::size_t nonempty_rows = 0U;
  std::size_t maximum_occupancy = 0U;
  for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
    const PatchRayAxisRowWorklist &axis = prepared.axes[axis_index];
    double lower_u = 0.0;
    double lower_v = 0.0;
    std::size_t expected_u = 0U;
    std::size_t expected_v = 0U;
    if (axis.axis != static_cast<RayAxis>(axis_index) ||
        !axis_grid(spec, axis.axis, lower_u, expected_u, lower_v, expected_v) ||
        axis.u_count != expected_u || axis.v_count != expected_v ||
        axis.offsets.size() != row_counts[axis_index] + 1U ||
        axis.offsets.empty() || axis.offsets.front() != 0U ||
        axis.offsets.back() != axis.candidates.size() ||
        !checked_add(total_rows, row_counts[axis_index]) ||
        !checked_add(candidate_pairs, axis.candidates.size()))
      return false;
    for (std::size_t row = 0U; row < row_counts[axis_index]; ++row) {
      const std::size_t begin = axis.offsets[row];
      const std::size_t end = axis.offsets[row + 1U];
      if (begin > end || end > axis.candidates.size())
        return false;
      if (begin != end) {
        if (!checked_add(nonempty_rows, 1U))
          return false;
        maximum_occupancy = std::max(maximum_occupancy, end - begin);
      }
      for (std::size_t position = begin; position < end; ++position) {
        if (!patch_ref_in_range(complex, axis.candidates[position]) ||
            (position > begin &&
             !(axis.candidates[position - 1U] < axis.candidates[position])))
          return false;
      }
    }
  }
  std::size_t retained = 0U;
  return total_rows == prepared.total_rows &&
         candidate_pairs == prepared.candidate_pairs &&
         nonempty_rows == prepared.nonempty_rows &&
         maximum_occupancy == prepared.maximum_row_occupancy &&
         prepared.raw_rectangle_candidate_pairs >= prepared.candidate_pairs &&
         retained_payload_bytes(prepared, retained) &&
         retained == prepared.retained_bytes &&
         prepared.accounted_peak_temporary_bytes >= prepared.retained_bytes;
}

} // namespace TitanPBE::Tabi::analytic_ses

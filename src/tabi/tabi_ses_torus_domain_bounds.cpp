#include "tabi_ses_torus_domain_bounds.h"

#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <numbers>
#include <stdexcept>
#include <string_view>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

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

template <typename T>
[[nodiscard]] bool add_vector_payload(const std::vector<T> &values,
                                      std::size_t &bytes) noexcept {
  return checked_payload(values.capacity(), sizeof(T), bytes);
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

[[nodiscard]] bool
valid_closed_interval(const AngularInterval &interval) noexcept {
  const double period = 2.0 * std::numbers::pi;
  return std::isfinite(interval.begin) && std::isfinite(interval.end) &&
         interval.begin >= 0.0 && interval.begin <= interval.end &&
         interval.end <= period;
}

[[nodiscard]] bool
angle_in_closed_interval(double angle,
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

[[nodiscard]] bool component_closed_range(double coefficient_cos,
                                          double coefficient_sin,
                                          const AngularInterval &interval,
                                          double &minimum,
                                          double &maximum) noexcept {
  if (!std::isfinite(coefficient_cos) || !std::isfinite(coefficient_sin) ||
      !valid_closed_interval(interval))
    return false;
  const auto evaluate = [&](double angle) {
    return coefficient_cos * std::cos(angle) +
           coefficient_sin * std::sin(angle);
  };
  const double begin = evaluate(interval.begin);
  const double end = evaluate(interval.end);
  if (!std::isfinite(begin) || !std::isfinite(end))
    return false;
  minimum = std::min(begin, end);
  maximum = std::max(begin, end);
  const double amplitude = std::hypot(coefficient_cos, coefficient_sin);
  if (!std::isfinite(amplitude))
    return false;
  if (amplitude == 0.0)
    return true;
  double maximum_angle = std::atan2(coefficient_sin, coefficient_cos);
  if (maximum_angle < 0.0)
    maximum_angle += 2.0 * std::numbers::pi;
  double minimum_angle = maximum_angle + std::numbers::pi;
  if (minimum_angle >= 2.0 * std::numbers::pi)
    minimum_angle -= 2.0 * std::numbers::pi;
  if (angle_in_closed_interval(maximum_angle, interval))
    maximum = amplitude;
  if (angle_in_closed_interval(minimum_angle, interval))
    minimum = -amplitude;
  return std::isfinite(minimum) && std::isfinite(maximum) && minimum <= maximum;
}

[[nodiscard]] bool
directed_beta_segments(const TorusBetaMinorArc &arc,
                       std::array<AngularInterval, 2> &segments,
                       std::size_t &segment_count) noexcept {
  segment_count = 0U;
  const double period = 2.0 * std::numbers::pi;
  if (!std::isfinite(arc.start) || !std::isfinite(arc.span) ||
      arc.start < 0.0 || arc.start >= period || arc.span < 0.0 ||
      arc.span > std::numbers::pi)
    return false;
  // Production accepts ccw(start,beta) <= span + tolerance.  Unlike theta
  // intervals, this does not expand before start.
  const double accepted_span = arc.span + kAnalyticSesAngleTolerance;
  const double end = arc.start + accepted_span;
  if (!std::isfinite(end) || accepted_span < 0.0 || accepted_span > period)
    return false;
  if (end <= period) {
    segments[segment_count++] = {arc.start, end};
  } else {
    segments[segment_count++] = {arc.start, period};
    segments[segment_count++] = {0.0, end - period};
  }
  return segment_count != 0U;
}

[[nodiscard]] bool beta_trigonometric_ranges(const TorusBetaMinorArc &arc,
                                             double &cos_minimum,
                                             double &cos_maximum,
                                             double &sin_minimum,
                                             double &sin_maximum) noexcept {
  std::array<AngularInterval, 2> segments{};
  std::size_t count = 0U;
  if (!directed_beta_segments(arc, segments, count))
    return false;
  cos_minimum = std::numeric_limits<double>::infinity();
  cos_maximum = -std::numeric_limits<double>::infinity();
  sin_minimum = std::numeric_limits<double>::infinity();
  sin_maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t segment = 0U; segment < count; ++segment) {
    double local_cos_minimum = 0.0;
    double local_cos_maximum = 0.0;
    double local_sin_minimum = 0.0;
    double local_sin_maximum = 0.0;
    if (!component_closed_range(1.0, 0.0, segments[segment], local_cos_minimum,
                                local_cos_maximum) ||
        !component_closed_range(0.0, 1.0, segments[segment], local_sin_minimum,
                                local_sin_maximum))
      return false;
    cos_minimum = std::min(cos_minimum, local_cos_minimum);
    cos_maximum = std::max(cos_maximum, local_cos_maximum);
    sin_minimum = std::min(sin_minimum, local_sin_minimum);
    sin_maximum = std::max(sin_maximum, local_sin_maximum);
  }
  if (!std::isfinite(cos_minimum) || !std::isfinite(cos_maximum) ||
      !std::isfinite(sin_minimum) || !std::isfinite(sin_maximum) ||
      cos_minimum > cos_maximum || sin_minimum > sin_maximum)
    return false;
  const double padding = 64.0 * std::numeric_limits<double>::epsilon();
  cos_minimum = std::nextafter(cos_minimum - padding,
                               -std::numeric_limits<double>::infinity());
  cos_maximum = std::nextafter(cos_maximum + padding,
                               std::numeric_limits<double>::infinity());
  sin_minimum = std::nextafter(sin_minimum - padding,
                               -std::numeric_limits<double>::infinity());
  sin_maximum = std::nextafter(sin_maximum + padding,
                               std::numeric_limits<double>::infinity());
  return std::isfinite(cos_minimum) && std::isfinite(cos_maximum) &&
         std::isfinite(sin_minimum) && std::isfinite(sin_maximum);
}

[[nodiscard]] bool interval_product(double left_minimum, double left_maximum,
                                    double right_minimum, double right_maximum,
                                    double &minimum, double &maximum) noexcept {
  const std::array<double, 4> products = {
      left_minimum * right_minimum, left_minimum * right_maximum,
      left_maximum * right_minimum, left_maximum * right_maximum};
  if (!std::all_of(products.begin(), products.end(),
                   [](double value) { return std::isfinite(value); }))
    return false;
  minimum = *std::min_element(products.begin(), products.end());
  maximum = *std::max_element(products.begin(), products.end());
  return minimum <= maximum;
}

[[nodiscard]] bool scaled_interval(double scale, double minimum, double maximum,
                                   double &scaled_minimum,
                                   double &scaled_maximum) noexcept {
  if (!std::isfinite(scale) || !std::isfinite(minimum) ||
      !std::isfinite(maximum) || minimum > maximum)
    return false;
  const double first = scale * minimum;
  const double second = scale * maximum;
  if (!std::isfinite(first) || !std::isfinite(second))
    return false;
  scaled_minimum = std::min(first, second);
  scaled_maximum = std::max(first, second);
  return true;
}

[[nodiscard]] bool intersect_bounds(PatchBounds &bounds,
                                    const PatchBounds &constraint) noexcept {
  if (!finite_bounds(bounds) || !finite_bounds(constraint))
    return false;
  bounds.lower.x = std::max(bounds.lower.x, constraint.lower.x);
  bounds.lower.y = std::max(bounds.lower.y, constraint.lower.y);
  bounds.lower.z = std::max(bounds.lower.z, constraint.lower.z);
  bounds.upper.x = std::min(bounds.upper.x, constraint.upper.x);
  bounds.upper.y = std::min(bounds.upper.y, constraint.upper.y);
  bounds.upper.z = std::min(bounds.upper.z, constraint.upper.z);
  return finite_bounds(bounds);
}

[[nodiscard]] bool torus_domain_bound_impl(const PatchComplex &complex,
                                           std::size_t torus_index,
                                           std::size_t interval_index,
                                           double spacing,
                                           PatchBounds &bounds) noexcept {
  bounds = {};
  if (torus_index >= complex.tori.size() || !std::isfinite(spacing) ||
      spacing <= 0.0)
    return false;
  const ToroidalPatch &patch = complex.tori[torus_index];
  if (interval_index >= patch.exposed_intervals.size() ||
      !std::isfinite(complex.probe_radius) || complex.probe_radius <= 0.0 ||
      !std::isfinite(patch.circle_radius) || patch.circle_radius <= 0.0)
    return false;
  TorusBetaMinorArc beta_arc;
  if (!torus_beta_minor_arc_for_audit(complex, torus_index, beta_arc))
    return false;
  double cos_minimum = 0.0;
  double cos_maximum = 0.0;
  double sin_minimum = 0.0;
  double sin_maximum = 0.0;
  if (!beta_trigonometric_ranges(beta_arc, cos_minimum, cos_maximum,
                                 sin_minimum, sin_maximum))
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
    double s_minimum = 0.0;
    double s_maximum = 0.0;
    if (!torus_exposed_theta_component_range_for_audit(
            basis_u[coordinate], basis_v[coordinate],
            patch.exposed_intervals[interval_index], s_minimum, s_maximum))
      return false;
    const double k_minimum =
        patch.circle_radius + complex.probe_radius * cos_minimum;
    const double k_maximum =
        patch.circle_radius + complex.probe_radius * cos_maximum;
    double radial_minimum = 0.0;
    double radial_maximum = 0.0;
    double axial_minimum = 0.0;
    double axial_maximum = 0.0;
    if (!interval_product(k_minimum, k_maximum, s_minimum, s_maximum,
                          radial_minimum, radial_maximum) ||
        !scaled_interval(complex.probe_radius * axis[coordinate], sin_minimum,
                         sin_maximum, axial_minimum, axial_maximum))
      return false;
    lower[coordinate] = center[coordinate] + radial_minimum + axial_minimum;
    upper[coordinate] = center[coordinate] + radial_maximum + axial_maximum;
    if (!std::isfinite(lower[coordinate]) ||
        !std::isfinite(upper[coordinate]) ||
        !pad_bound(lower[coordinate], upper[coordinate], spacing, 512.0))
      return false;
  }
  bounds = {{lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}};
  PatchBounds theta_bound;
  if (!torus_exposed_interval_bound_for_audit(
          complex, torus_index, interval_index, spacing, theta_bound) ||
      !intersect_bounds(bounds, theta_bound))
    return false;
  // D2a already intersects this bound with the D1 worklist bound, including
  // its extra row-range padding. The smaller raw BVH bound must not be applied
  // again here because it could erase that conservative grid margin.
  return true;
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

[[nodiscard]] bool row_coordinates(const PatchMappingGridSpec &spec,
                                   RayAxis axis, std::size_t row, double &u,
                                   double &v) noexcept {
  double lower_u = 0.0;
  double lower_v = 0.0;
  std::size_t u_count = 0U;
  std::size_t v_count = 0U;
  std::size_t row_count = 0U;
  if (!axis_grid(spec, axis, lower_u, u_count, lower_v, v_count) ||
      u_count == 0U || !checked_multiply(u_count, v_count, row_count) ||
      row >= row_count)
    return false;
  const std::size_t iu = row % u_count;
  const std::size_t iv = row / u_count;
  u = lower_u + spec.grid_spacing * static_cast<double>(iu);
  v = lower_v + spec.grid_spacing * static_cast<double>(iv);
  return std::isfinite(u) && std::isfinite(v);
}

[[nodiscard]] bool projected_contains(const PatchBounds &bounds, RayAxis axis,
                                      double u, double v) noexcept {
  if (!finite_bounds(bounds) || !std::isfinite(u) || !std::isfinite(v))
    return false;
  switch (axis) {
  case RayAxis::x:
    return u >= bounds.lower.y && u <= bounds.upper.y && v >= bounds.lower.z &&
           v <= bounds.upper.z;
  case RayAxis::y:
    return u >= bounds.lower.x && u <= bounds.upper.x && v >= bounds.lower.z &&
           v <= bounds.upper.z;
  case RayAxis::z:
    return u >= bounds.lower.x && u <= bounds.upper.x && v >= bounds.lower.y &&
           v <= bounds.upper.y;
  }
  return false;
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
retained_payload_bytes(const PreparedTorusDomainRowWorklist &prepared,
                       std::size_t &bytes) noexcept {
  bytes = 0U;
  for (const PatchRayAxisRowWorklist &axis : prepared.axes) {
    if (!checked_payload(axis.offsets.capacity(), sizeof(std::size_t), bytes) ||
        !checked_payload(axis.candidates.capacity(), sizeof(PatchRefId), bytes))
      return false;
  }
  return checked_payload(prepared.torus_domain_offsets.capacity(),
                         sizeof(std::size_t), bytes) &&
         checked_payload(prepared.torus_domain_bounds.capacity(),
                         sizeof(TorusDomainIntervalBound), bytes) &&
         checked_payload(prepared.torus_fallback_reasons.capacity(),
                         sizeof(TorusDomainFallbackReason), bytes);
}

} // namespace

bool torus_domain_interval_bound_for_audit(const PatchComplex &complex,
                                           std::size_t torus_index,
                                           std::size_t interval_index,
                                           double grid_spacing,
                                           PatchBounds &bounds) noexcept {
  return torus_domain_bound_impl(complex, torus_index, interval_index,
                                 grid_spacing, bounds);
}

bool torus_beta_trigonometric_ranges_for_audit(const TorusBetaMinorArc &arc,
                                               double &cos_minimum,
                                               double &cos_maximum,
                                               double &sin_minimum,
                                               double &sin_maximum) noexcept {
  cos_minimum = 0.0;
  cos_maximum = 0.0;
  sin_minimum = 0.0;
  sin_maximum = 0.0;
  return beta_trigonometric_ranges(arc, cos_minimum, cos_maximum, sin_minimum,
                                   sin_maximum);
}

PreparedTorusDomainRowWorklist build_torus_domain_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor, TorusDomainRowWorklistBuildOptions options) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  PreparedTorusDomainRowWorklist result;
  const auto elapsed = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count();
  };
  const auto fail = [&](std::string_view status) {
    PreparedTorusDomainRowWorklist failed;
    failed.status.assign(status);
    failed.build_ms = elapsed();
    return failed;
  };

  try {
    std::array<std::size_t, 3> row_counts{};
    if (complex.status != "pass" ||
        !validate_patch_mapping_grid_spec(spec, &row_counts, nullptr))
      return fail("invalid_input");
    result.requested_workers = executor == nullptr ? 1U : executor->capacity();
    if (result.requested_workers == 0U)
      return fail("invalid_executor");
    result.complex_fingerprint = fingerprint_patch_complex(complex);
    result.lattice_identity = make_patch_ray_lattice_identity(
        spec.lower, spec.grid_spacing, spec.nx, spec.ny, spec.nz);
    if (!result.lattice_identity.valid)
      return fail("invalid_input");

    const auto theta_started = Clock::now();
    TorusIntervalRowWorklistBuildOptions theta_options;
    theta_options.reverse_rectangle_schedule_for_audit =
        options.reverse_torus_schedule_for_audit;
    theta_options.raw_candidate_pair_limit =
        options.theta_raw_candidate_pair_limit;
    PreparedTorusIntervalRowWorklist theta = build_torus_interval_row_worklist(
        complex, spec, executor, theta_options);
    result.theta_worklist_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - theta_started)
            .count();
    if (!validate_torus_interval_row_worklist(complex, spec, theta))
      return fail("invalid_theta_worklist");
    result.theta_candidate_pairs = theta.candidate_pairs;

    std::vector<std::size_t> record_counts(complex.tori.size(), 0U);
    std::vector<TorusDomainFallbackReason> plans(
        complex.tori.size(), TorusDomainFallbackReason::none);
    for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
      const TorusIntervalFallbackReason theta_reason =
          theta.torus_fallback_reasons[torus];
      if (theta_reason != TorusIntervalFallbackReason::none) {
        plans[torus] = TorusDomainFallbackReason::theta_worklist_fallback;
        continue;
      }
      TorusBetaMinorArc arc;
      if (!torus_beta_minor_arc_for_audit(complex, torus, arc)) {
        plans[torus] = TorusDomainFallbackReason::invalid_beta_domain;
        continue;
      }
      const ToroidalPatch &patch = complex.tori[torus];
      bool valid = true;
      for (std::size_t interval = 0U; interval < patch.exposed_intervals.size();
           ++interval) {
        PatchBounds trial;
        if (!torus_domain_bound_impl(complex, torus, interval,
                                     spec.grid_spacing, trial)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        plans[torus] = TorusDomainFallbackReason::beta_bound_failure;
        continue;
      }
      record_counts[torus] = patch.exposed_intervals.size();
    }

    std::size_t offset_count = complex.tori.size();
    if (!checked_add(offset_count, 1U))
      return fail("count_overflow");
    result.torus_domain_offsets.assign(offset_count, 0U);
    for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
      std::size_t next = result.torus_domain_offsets[torus];
      if (!checked_add(next, record_counts[torus]))
        return fail("count_overflow");
      result.torus_domain_offsets[torus + 1U] = next;
    }
    result.torus_domain_bounds.resize(result.torus_domain_offsets.back());
    result.torus_fallback_reasons = plans;

    std::vector<std::uint8_t> bounds_workers(result.requested_workers, 0U);
    const auto bounds_started = Clock::now();
    const auto populate_torus = [&](std::size_t torus) {
      if (plans[torus] != TorusDomainFallbackReason::none)
        return;
      PatchRefId ref;
      if (!PatchRefId::try_make(PatchKind::toroidal, torus, ref))
        throw BuildFailure{"invalid_ref"};
      const std::size_t begin = result.torus_domain_offsets[torus];
      const ToroidalPatch &patch = complex.tori[torus];
      for (std::size_t interval = 0U; interval < patch.exposed_intervals.size();
           ++interval) {
        PatchBounds bounds;
        if (!torus_domain_bound_impl(complex, torus, interval,
                                     spec.grid_spacing, bounds))
          throw BuildFailure{"beta_bound_failure"};
        result.torus_domain_bounds[begin + interval] = {
            ref, static_cast<std::uint32_t>(interval), bounds};
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
    result.beta_bounds_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - bounds_started)
            .count();

    for (std::size_t torus = 0U; torus < plans.size(); ++torus) {
      const std::size_t reason = static_cast<std::size_t>(plans[torus]);
      if (reason >= result.fallback_reason_counts.size() ||
          !checked_add(result.fallback_reason_counts[reason], 1U))
        throw BuildFailure{"count_overflow"};
      if (plans[torus] == TorusDomainFallbackReason::none) {
        if (!checked_add(result.domain_rectangles, record_counts[torus]))
          throw BuildFailure{"count_overflow"};
      } else if (plans[torus] ==
                 TorusDomainFallbackReason::theta_worklist_fallback) {
        if (!checked_add(result.full_fallback_tori, 1U))
          throw BuildFailure{"count_overflow"};
      } else if (!checked_add(result.beta_fallback_to_theta_tori, 1U)) {
        throw BuildFailure{"count_overflow"};
      }
    }

    const auto filter_started = Clock::now();
    std::array<std::vector<std::size_t>, 3> counts;
    std::size_t kept_pairs = 0U;
    const auto keep_candidate = [&](RayAxis axis, std::size_t row,
                                    PatchRefId ref) {
      if (ref.kind() != PatchKind::toroidal)
        return true;
      const std::size_t torus = ref.index();
      if (torus >= plans.size())
        throw BuildFailure{"invalid_theta_worklist"};
      if (plans[torus] != TorusDomainFallbackReason::none)
        return true;
      double u = 0.0;
      double v = 0.0;
      if (!row_coordinates(spec, axis, row, u, v))
        throw BuildFailure{"invalid_grid"};
      const std::size_t begin = result.torus_domain_offsets[torus];
      const std::size_t end = result.torus_domain_offsets[torus + 1U];
      for (std::size_t position = begin; position < end; ++position) {
        if (projected_contains(result.torus_domain_bounds[position].bounds,
                               axis, u, v))
          return true;
      }
      return false;
    };

    for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
      const PatchRayAxisRowWorklist &source = theta.axes[axis_index];
      PatchRayAxisRowWorklist &target = result.axes[axis_index];
      target.axis = source.axis;
      target.u_count = source.u_count;
      target.v_count = source.v_count;
      counts[axis_index].assign(row_counts[axis_index], 0U);
      for (std::size_t row = 0U; row < row_counts[axis_index]; ++row) {
        for (const PatchRefId ref : source.row(row)) {
          if (keep_candidate(source.axis, row, ref) &&
              (!checked_add(counts[axis_index][row], 1U) ||
               !checked_add(kept_pairs, 1U)))
            throw BuildFailure{"count_overflow"};
        }
      }
      if (kept_pairs > options.candidate_pair_limit)
        throw BuildFailure{"candidate_limit"};
      std::size_t sentinel = row_counts[axis_index];
      if (!checked_add(sentinel, 1U))
        throw BuildFailure{"count_overflow"};
      target.offsets.assign(sentinel, 0U);
      std::size_t prefix = 0U;
      for (std::size_t row = 0U; row < row_counts[axis_index]; ++row) {
        const std::size_t occupancy = counts[axis_index][row];
        if (occupancy != 0U && !checked_add(result.nonempty_rows, 1U))
          throw BuildFailure{"count_overflow"};
        result.maximum_row_occupancy =
            std::max(result.maximum_row_occupancy, occupancy);
        if (!checked_add(prefix, occupancy))
          throw BuildFailure{"count_overflow"};
        target.offsets[row + 1U] = prefix;
      }
      target.candidates.resize(prefix);
      for (std::size_t row = 0U; row < row_counts[axis_index]; ++row)
        counts[axis_index][row] = target.offsets[row];

      for (std::size_t row = 0U; row < row_counts[axis_index]; ++row) {
        const std::span<const PatchRefId> candidates = source.row(row);
        const auto append = [&](PatchRefId ref) {
          if (!keep_candidate(source.axis, row, ref))
            return;
          const std::size_t position = counts[axis_index][row]++;
          if (position >= target.candidates.size())
            throw BuildFailure{"count_overflow"};
          target.candidates[position] = ref;
        };
        if (options.reverse_torus_schedule_for_audit) {
          for (auto position = candidates.rbegin();
               position != candidates.rend(); ++position)
            append(*position);
        } else {
          for (const PatchRefId ref : candidates)
            append(ref);
        }
        const std::size_t begin = target.offsets[row];
        const std::size_t end = target.offsets[row + 1U];
        std::sort(target.candidates.begin() +
                      static_cast<std::ptrdiff_t>(begin),
                  target.candidates.begin() + static_cast<std::ptrdiff_t>(end));
      }
      if (!checked_add(result.candidate_pairs, target.candidates.size()) ||
          !checked_add(result.total_rows, row_counts[axis_index]))
        throw BuildFailure{"count_overflow"};
    }
    if (result.candidate_pairs != kept_pairs ||
        result.candidate_pairs > options.candidate_pair_limit)
      throw BuildFailure{"candidate_limit"};
    result.filter_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - filter_started)
            .count();

    std::size_t retained = 0U;
    if (!retained_payload_bytes(result, retained))
      throw BuildFailure{"count_overflow"};
    result.retained_bytes = retained;
    std::size_t later_phase_peak = retained;
    if (!add_vector_payload(record_counts, later_phase_peak) ||
        !add_vector_payload(plans, later_phase_peak) ||
        !add_vector_payload(bounds_workers, later_phase_peak) ||
        !checked_add(later_phase_peak, theta.retained_bytes))
      throw BuildFailure{"count_overflow"};
    for (const auto &axis_counts : counts) {
      if (!add_vector_payload(axis_counts, later_phase_peak))
        throw BuildFailure{"count_overflow"};
    }
    result.accounted_peak_temporary_bytes =
        std::max(theta.accounted_peak_temporary_bytes, later_phase_peak);
    result.domain_worklist_complete = true;
    if (result.full_fallback_tori != 0U)
      result.status = "pass_with_full_fallback";
    else if (result.beta_fallback_to_theta_tori != 0U)
      result.status = "pass_with_theta_fallback";
    else
      result.status = "pass";
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

bool validate_torus_domain_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTorusDomainRowWorklist &prepared) noexcept {
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
      (prepared.status != "pass" &&
       prepared.status != "pass_with_theta_fallback" &&
       prepared.status != "pass_with_full_fallback") ||
      !prepared.domain_worklist_complete || prepared.requested_workers == 0U ||
      prepared.bounds_workers_used == 0U ||
      prepared.bounds_workers_used > prepared.requested_workers ||
      prepared.torus_domain_offsets.size() != complex.tori.size() + 1U ||
      prepared.torus_fallback_reasons.size() != complex.tori.size() ||
      prepared.torus_domain_offsets.empty() ||
      prepared.torus_domain_offsets.front() != 0U ||
      prepared.torus_domain_offsets.back() !=
          prepared.torus_domain_bounds.size() ||
      !std::isfinite(prepared.theta_worklist_ms) ||
      prepared.theta_worklist_ms < 0.0 ||
      !std::isfinite(prepared.beta_bounds_ms) ||
      prepared.beta_bounds_ms < 0.0 || !std::isfinite(prepared.filter_ms) ||
      prepared.filter_ms < 0.0 || !std::isfinite(prepared.build_ms) ||
      prepared.build_ms < 0.0)
    return false;

  std::size_t domain_rectangles = 0U;
  std::size_t beta_fallbacks = 0U;
  std::size_t full_fallbacks = 0U;
  std::array<std::size_t, 4> reasons{};
  for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
    const std::size_t begin = prepared.torus_domain_offsets[torus];
    const std::size_t end = prepared.torus_domain_offsets[torus + 1U];
    if (begin > end || end > prepared.torus_domain_bounds.size())
      return false;
    const TorusDomainFallbackReason reason =
        prepared.torus_fallback_reasons[torus];
    const std::size_t reason_index = static_cast<std::size_t>(reason);
    if (reason_index >= reasons.size() ||
        !checked_add(reasons[reason_index], 1U))
      return false;
    if (reason == TorusDomainFallbackReason::none) {
      if (end - begin != complex.tori[torus].exposed_intervals.size() ||
          !checked_add(domain_rectangles, end - begin))
        return false;
    } else {
      if (begin != end)
        return false;
      if (reason == TorusDomainFallbackReason::theta_worklist_fallback) {
        if (!checked_add(full_fallbacks, 1U))
          return false;
      } else if (!checked_add(beta_fallbacks, 1U)) {
        return false;
      }
    }
    PatchRefId expected;
    if (!PatchRefId::try_make(PatchKind::toroidal, torus, expected))
      return false;
    for (std::size_t position = begin; position < end; ++position) {
      const TorusDomainIntervalBound &bound =
          prepared.torus_domain_bounds[position];
      if (bound.ref != expected || bound.interval_index != position - begin ||
          !finite_bounds(bound.bounds))
        return false;
    }
  }
  const std::string expected_status =
      full_fallbacks != 0U   ? "pass_with_full_fallback"
      : beta_fallbacks != 0U ? "pass_with_theta_fallback"
                             : "pass";
  if (domain_rectangles != prepared.domain_rectangles ||
      beta_fallbacks != prepared.beta_fallback_to_theta_tori ||
      full_fallbacks != prepared.full_fallback_tori ||
      reasons != prepared.fallback_reason_counts ||
      prepared.status != expected_status)
    return false;

  std::size_t total_rows = 0U;
  std::size_t candidate_pairs = 0U;
  std::size_t nonempty_rows = 0U;
  std::size_t maximum_occupancy = 0U;
  for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
    const PatchRayAxisRowWorklist &axis = prepared.axes[axis_index];
    double lower_u = 0.0;
    double lower_v = 0.0;
    std::size_t u_count = 0U;
    std::size_t v_count = 0U;
    if (axis.axis != static_cast<RayAxis>(axis_index) ||
        !axis_grid(spec, axis.axis, lower_u, u_count, lower_v, v_count) ||
        axis.u_count != u_count || axis.v_count != v_count ||
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
         candidate_pairs <= prepared.theta_candidate_pairs &&
         nonempty_rows == prepared.nonempty_rows &&
         maximum_occupancy == prepared.maximum_row_occupancy &&
         retained_payload_bytes(prepared, retained) &&
         retained == prepared.retained_bytes &&
         prepared.accounted_peak_temporary_bytes >= prepared.retained_bytes;
}

} // namespace TitanPBE::Tabi::analytic_ses

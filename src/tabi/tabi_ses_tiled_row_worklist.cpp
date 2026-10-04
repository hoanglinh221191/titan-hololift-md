#include "tabi_ses_tiled_row_worklist.h"

#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

struct IndexRange {
  std::size_t begin{0U};
  std::size_t end{0U};
};

struct BoundRectangle {
  PatchRefId ref;
  RayAxis axis{RayAxis::x};
  IndexRange u;
  IndexRange v;
};

struct AxisTileLayout {
  std::size_t global_begin{0U};
  std::size_t u_tiles{0U};
  std::size_t v_tiles{0U};
  std::size_t u_count{0U};
  std::size_t v_count{0U};
};

struct RowRefPair {
  std::size_t row{0U};
  PatchRefId ref;
};

struct WorkerScratch {
  std::vector<RowRefPair> pairs;
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

[[nodiscard]] bool checked_payload(std::size_t count, std::size_t width,
                                   std::size_t &bytes) noexcept {
  std::size_t payload = 0U;
  return checked_multiply(count, width, payload) && checked_add(bytes, payload);
}

template <typename T>
[[nodiscard]] bool add_vector_capacity(const std::vector<T> &values,
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

[[nodiscard]] bool grid_coordinate(double lower, double spacing,
                                   std::size_t index,
                                   double &coordinate) noexcept {
  coordinate = lower + spacing * static_cast<double>(index);
  return std::isfinite(coordinate);
}

// Exact inclusive binary-search contract used by D1/D2a/D2b. No rounded
// division is used at a bound boundary.
[[nodiscard]] bool grid_index_range(double lower, double spacing,
                                    std::size_t count, double bound_lower,
                                    double bound_upper,
                                    IndexRange &range) noexcept {
  range = {};
  if (count == 0U || !std::isfinite(lower) || !std::isfinite(spacing) ||
      spacing <= 0.0 || !std::isfinite(bound_lower) ||
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

[[nodiscard]] bool append_rectangles(
    const PatchMappingGridSpec &spec, PatchRefId ref, const PatchBounds &bounds,
    std::vector<BoundRectangle> &rectangles, std::size_t &raw_row_pairs,
    const TiledPatchRayRowWorklistBuildOptions &options) {
  if (!ref.valid() || !finite_bounds(bounds))
    return false;
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
    BoundRectangle rectangle;
    rectangle.ref = ref;
    rectangle.axis = axis;
    if (!axis_grid(spec, axis, lower_u, u_count, lower_v, v_count) ||
        !projected_bounds(bounds, axis, bound_lower_u, bound_upper_u,
                          bound_lower_v, bound_upper_v) ||
        !grid_index_range(lower_u, spec.grid_spacing, u_count, bound_lower_u,
                          bound_upper_u, rectangle.u) ||
        !grid_index_range(lower_v, spec.grid_spacing, v_count, bound_lower_v,
                          bound_upper_v, rectangle.v))
      return false;
    std::size_t pairs = 0U;
    if (!checked_multiply(rectangle.u.end - rectangle.u.begin,
                          rectangle.v.end - rectangle.v.begin, pairs) ||
        !checked_add(raw_row_pairs, pairs) ||
        raw_row_pairs > options.raw_row_pair_limit)
      throw BuildFailure{"raw_row_pair_limit"};
    rectangles.push_back(rectangle);
  }
  return true;
}

[[nodiscard]] bool row_ref_less(const RowRefPair &left,
                                const RowRefPair &right) noexcept {
  if (left.row != right.row)
    return left.row < right.row;
  return left.ref < right.ref;
}

[[nodiscard]] bool row_ref_equal(const RowRefPair &left,
                                 const RowRefPair &right) noexcept {
  return left.row == right.row && left.ref == right.ref;
}

[[nodiscard]] bool
retained_payload_bytes(const PreparedTiledPatchRayRowWorklist &prepared,
                       std::size_t &bytes) noexcept {
  bytes = 0U;
  for (const PatchRayAxisRowWorklist &axis : prepared.axes) {
    if (!checked_payload(axis.offsets.capacity(), sizeof(std::size_t), bytes) ||
        !checked_payload(axis.candidates.capacity(), sizeof(PatchRefId), bytes))
      return false;
  }
  return true;
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

} // namespace

PreparedTiledPatchRayRowWorklist build_tiled_patch_ray_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    TitanExecutor *executor, TiledPatchRayRowWorklistBuildOptions options) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  PreparedTiledPatchRayRowWorklist result;
  const auto elapsed = [&]() {
    return std::chrono::duration<double, std::milli>(Clock::now() - started)
        .count();
  };
  const auto fail = [&](std::string_view status) {
    PreparedTiledPatchRayRowWorklist failed;
    failed.status.assign(status);
    failed.build_ms = elapsed();
    return failed;
  };

  try {
    std::array<std::size_t, 3> row_counts{};
    if (complex.status != "pass" ||
        !validate_patch_mapping_grid_spec(spec, &row_counts, nullptr) ||
        options.tile_u_rows == 0U || options.tile_v_rows == 0U)
      return fail("invalid_input");
    result.requested_workers = executor == nullptr ? 1U : executor->capacity();
    if (result.requested_workers == 0U)
      return fail("invalid_executor");
    result.complex_fingerprint = fingerprint_patch_complex(complex);
    result.lattice_identity = make_patch_ray_lattice_identity(
        spec.lower, spec.grid_spacing, spec.nx, spec.ny, spec.nz);
    if (!result.lattice_identity.valid)
      return fail("invalid_input");
    result.tile_u_rows = options.tile_u_rows;
    result.tile_v_rows = options.tile_v_rows;

    const auto bounds_started = Clock::now();
    std::vector<BoundRectangle> rectangles;
    std::size_t base_patch_count = complex.contacts.size();
    if (!checked_add(base_patch_count, complex.reentrant.size()))
      return fail("count_overflow");
    std::size_t rectangle_reserve = 0U;
    if (!checked_multiply(base_patch_count, 3U, rectangle_reserve))
      return fail("count_overflow");
    for (const ToroidalPatch &torus : complex.tori) {
      std::size_t records =
          std::max<std::size_t>(1U, torus.exposed_intervals.size());
      std::size_t torus_rectangles = 0U;
      if (!checked_multiply(records, 3U, torus_rectangles) ||
          !checked_add(rectangle_reserve, torus_rectangles))
        return fail("count_overflow");
    }
    rectangles.reserve(rectangle_reserve);

    const auto append_full_patch = [&](PatchKind kind, std::size_t index) {
      PatchRefId ref;
      PatchBounds bounds;
      if (!PatchRefId::try_make(kind, index, ref) ||
          !patch_ray_full_padded_bound_for_audit(complex, kind, index,
                                                 spec.grid_spacing, bounds) ||
          !append_rectangles(spec, ref, bounds, rectangles,
                             result.raw_row_pairs, options))
        throw BuildFailure{"invalid_patch_geometry"};
    };
    for (std::size_t contact = 0U; contact < complex.contacts.size(); ++contact)
      append_full_patch(PatchKind::contact, contact);

    std::size_t maximum_bound_plan_scratch = 0U;
    {
      std::vector<PatchBounds> theta_bounds;
      std::vector<PatchBounds> domain_bounds;
      const auto account_bound_scratch = [&]() {
        std::size_t bound_capacity = theta_bounds.capacity();
        std::size_t bound_bytes = 0U;
        if (!checked_add(bound_capacity, domain_bounds.capacity()) ||
            !checked_multiply(bound_capacity, sizeof(PatchBounds), bound_bytes))
          throw BuildFailure{"count_overflow"};
        maximum_bound_plan_scratch =
            std::max(maximum_bound_plan_scratch, bound_bytes);
      };
      for (std::size_t torus = 0U; torus < complex.tori.size(); ++torus) {
        PatchRefId ref;
        if (!PatchRefId::try_make(PatchKind::toroidal, torus, ref))
          throw BuildFailure{"invalid_ref"};
        const std::size_t interval_count =
            complex.tori[torus].exposed_intervals.size();
        theta_bounds.clear();
        domain_bounds.clear();
        bool theta_valid = interval_count != 0U;
        if (theta_valid) {
          theta_bounds.resize(interval_count);
          account_bound_scratch();
          for (std::size_t interval = 0U; interval < interval_count;
               ++interval) {
            if (!torus_exposed_interval_bound_for_audit(
                    complex, torus, interval, spec.grid_spacing,
                    theta_bounds[interval])) {
              theta_valid = false;
              break;
            }
          }
        }
        if (!theta_valid) {
          account_bound_scratch();
          append_full_patch(PatchKind::toroidal, torus);
          if (!checked_add(result.torus_full_fallback_plans, 1U))
            throw BuildFailure{"count_overflow"};
          continue;
        }
        domain_bounds.resize(interval_count);
        account_bound_scratch();
        bool domain_valid = true;
        for (std::size_t interval = 0U; interval < interval_count; ++interval) {
          if (!torus_domain_interval_bound_for_audit(complex, torus, interval,
                                                     spec.grid_spacing,
                                                     domain_bounds[interval])) {
            domain_valid = false;
            break;
          }
        }
        const std::vector<PatchBounds> &selected =
            domain_valid ? domain_bounds : theta_bounds;
        for (const PatchBounds &bounds : selected) {
          if (!append_rectangles(spec, ref, bounds, rectangles,
                                 result.raw_row_pairs, options))
            throw BuildFailure{"invalid_patch_geometry"};
        }
        if (domain_valid) {
          if (!checked_add(result.torus_domain_plans, 1U))
            throw BuildFailure{"count_overflow"};
        } else if (!checked_add(result.torus_theta_fallback_plans, 1U)) {
          throw BuildFailure{"count_overflow"};
        }
        account_bound_scratch();
      }
    }
    for (std::size_t reentrant = 0U; reentrant < complex.reentrant.size();
         ++reentrant)
      append_full_patch(PatchKind::reentrant, reentrant);
    result.bound_rectangles = rectangles.size();
    result.bounds_plan_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - bounds_started)
            .count();

    const auto tile_started = Clock::now();
    std::array<AxisTileLayout, 3> layouts{};
    std::size_t global_tile_count = 0U;
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      double lower_u = 0.0;
      double lower_v = 0.0;
      if (!axis_grid(spec, static_cast<RayAxis>(axis), lower_u,
                     layouts[axis].u_count, lower_v, layouts[axis].v_count))
        throw BuildFailure{"invalid_grid"};
      layouts[axis].global_begin = global_tile_count;
      layouts[axis].u_tiles =
          layouts[axis].u_count / options.tile_u_rows +
          static_cast<std::size_t>(
              layouts[axis].u_count % options.tile_u_rows != 0U);
      layouts[axis].v_tiles =
          layouts[axis].v_count / options.tile_v_rows +
          static_cast<std::size_t>(
              layouts[axis].v_count % options.tile_v_rows != 0U);
      std::size_t axis_tiles = 0U;
      if (!checked_multiply(layouts[axis].u_tiles, layouts[axis].v_tiles,
                            axis_tiles) ||
          !checked_add(global_tile_count, axis_tiles) ||
          !checked_add(result.total_rows, row_counts[axis]))
        throw BuildFailure{"count_overflow"};
      result.axes[axis].axis = static_cast<RayAxis>(axis);
      result.axes[axis].u_count = layouts[axis].u_count;
      result.axes[axis].v_count = layouts[axis].v_count;
    }
    result.total_tiles = global_tile_count;
    std::vector<std::size_t> tile_rectangle_counts(global_tile_count, 0U);
    const auto rectangle_tile_range = [&](const BoundRectangle &rectangle,
                                          auto &&consumer) {
      const std::size_t axis = static_cast<std::size_t>(rectangle.axis);
      if (axis >= layouts.size())
        throw BuildFailure{"invalid_rectangle"};
      if (rectangle.u.begin == rectangle.u.end ||
          rectangle.v.begin == rectangle.v.end)
        return;
      const AxisTileLayout &layout = layouts[axis];
      const std::size_t tile_u_begin = rectangle.u.begin / options.tile_u_rows;
      const std::size_t tile_u_end =
          (rectangle.u.end - 1U) / options.tile_u_rows + 1U;
      const std::size_t tile_v_begin = rectangle.v.begin / options.tile_v_rows;
      const std::size_t tile_v_end =
          (rectangle.v.end - 1U) / options.tile_v_rows + 1U;
      if (tile_u_end > layout.u_tiles || tile_v_end > layout.v_tiles)
        throw BuildFailure{"invalid_rectangle"};
      for (std::size_t tv = tile_v_begin; tv < tile_v_end; ++tv) {
        for (std::size_t tu = tile_u_begin; tu < tile_u_end; ++tu) {
          std::size_t local = 0U;
          if (!checked_multiply(tv, layout.u_tiles, local) ||
              !checked_add(local, tu) ||
              !checked_add(local, layout.global_begin) ||
              local >= global_tile_count)
            throw BuildFailure{"count_overflow"};
          consumer(local);
        }
      }
    };
    const auto count_rectangle_tiles = [&](const BoundRectangle &rectangle) {
      rectangle_tile_range(rectangle, [&](std::size_t tile) {
        if (!checked_add(tile_rectangle_counts[tile], 1U) ||
            !checked_add(result.rectangle_tile_pairs, 1U) ||
            result.rectangle_tile_pairs > options.rectangle_tile_pair_limit)
          throw BuildFailure{"rectangle_tile_pair_limit"};
      });
    };
    if (options.reverse_rectangle_schedule_for_audit) {
      for (auto rectangle = rectangles.rbegin(); rectangle != rectangles.rend();
           ++rectangle)
        count_rectangle_tiles(*rectangle);
    } else {
      for (const BoundRectangle &rectangle : rectangles)
        count_rectangle_tiles(rectangle);
    }
    std::size_t tile_offset_count = global_tile_count;
    if (!checked_add(tile_offset_count, 1U))
      throw BuildFailure{"count_overflow"};
    std::vector<std::size_t> tile_rectangle_offsets(tile_offset_count, 0U);
    std::size_t prefix = 0U;
    for (std::size_t tile = 0U; tile < global_tile_count; ++tile) {
      if (!checked_add(prefix, tile_rectangle_counts[tile]))
        throw BuildFailure{"count_overflow"};
      tile_rectangle_offsets[tile + 1U] = prefix;
    }
    if (prefix != result.rectangle_tile_pairs)
      throw BuildFailure{"count_overflow"};
    std::vector<std::size_t> tile_rectangles(prefix, 0U);
    for (std::size_t tile = 0U; tile < global_tile_count; ++tile)
      tile_rectangle_counts[tile] = tile_rectangle_offsets[tile];
    const auto fill_rectangle_tiles = [&](std::size_t rectangle_index) {
      rectangle_tile_range(rectangles[rectangle_index], [&](std::size_t tile) {
        const std::size_t position = tile_rectangle_counts[tile]++;
        if (position >= tile_rectangles.size())
          throw BuildFailure{"count_overflow"};
        tile_rectangles[position] = rectangle_index;
      });
    };
    if (options.reverse_rectangle_schedule_for_audit) {
      for (std::size_t index = rectangles.size(); index > 0U; --index)
        fill_rectangle_tiles(index - 1U);
    } else {
      for (std::size_t index = 0U; index < rectangles.size(); ++index)
        fill_rectangle_tiles(index);
    }
    for (std::size_t tile = 0U; tile < global_tile_count; ++tile) {
      if (tile_rectangle_counts[tile] != tile_rectangle_offsets[tile + 1U])
        throw BuildFailure{"tile_plan_mismatch"};
    }
    std::size_t tile_plan_phase_upper = 0U;
    if (!add_vector_capacity(rectangles, tile_plan_phase_upper) ||
        !add_vector_capacity(tile_rectangle_counts, tile_plan_phase_upper) ||
        !add_vector_capacity(tile_rectangle_offsets, tile_plan_phase_upper) ||
        !add_vector_capacity(tile_rectangles, tile_plan_phase_upper))
      throw BuildFailure{"count_overflow"};
    std::vector<std::size_t>().swap(tile_rectangle_counts);
    std::vector<std::size_t> active_tiles;
    std::vector<std::size_t> raw_pairs_per_tile(global_tile_count, 0U);
    for (std::size_t tile = 0U; tile < global_tile_count; ++tile) {
      if (tile_rectangle_offsets[tile] == tile_rectangle_offsets[tile + 1U])
        continue;
      active_tiles.push_back(tile);
      std::size_t axis = 0U;
      while (axis + 1U < layouts.size() &&
             tile >= layouts[axis + 1U].global_begin)
        ++axis;
      const AxisTileLayout &layout = layouts[axis];
      const std::size_t local = tile - layout.global_begin;
      const std::size_t tu = local % layout.u_tiles;
      const std::size_t tv = local / layout.u_tiles;
      std::size_t tile_u_begin = 0U;
      std::size_t tile_v_begin = 0U;
      if (!checked_multiply(tu, options.tile_u_rows, tile_u_begin) ||
          !checked_multiply(tv, options.tile_v_rows, tile_v_begin))
        throw BuildFailure{"count_overflow"};
      const std::size_t tile_u_end =
          tile_u_begin +
          std::min(options.tile_u_rows, layout.u_count - tile_u_begin);
      const std::size_t tile_v_end =
          tile_v_begin +
          std::min(options.tile_v_rows, layout.v_count - tile_v_begin);
      std::size_t raw = 0U;
      for (std::size_t position = tile_rectangle_offsets[tile];
           position < tile_rectangle_offsets[tile + 1U]; ++position) {
        const BoundRectangle &rectangle = rectangles[tile_rectangles[position]];
        const std::size_t u_begin = std::max(tile_u_begin, rectangle.u.begin);
        const std::size_t u_end = std::min(tile_u_end, rectangle.u.end);
        const std::size_t v_begin = std::max(tile_v_begin, rectangle.v.begin);
        const std::size_t v_end = std::min(tile_v_end, rectangle.v.end);
        std::size_t pairs = 0U;
        if (u_begin > u_end || v_begin > v_end ||
            !checked_multiply(u_end - u_begin, v_end - v_begin, pairs) ||
            !checked_add(raw, pairs))
          throw BuildFailure{"count_overflow"};
      }
      if (raw > options.per_tile_scratch_pair_limit)
        throw BuildFailure{"per_tile_scratch_limit"};
      raw_pairs_per_tile[tile] = raw;
      result.maximum_raw_pairs_per_tile =
          std::max(result.maximum_raw_pairs_per_tile, raw);
    }
    result.active_tiles = active_tiles.size();
    result.scheduled_workers =
        active_tiles.empty()
            ? 1U
            : std::min(result.requested_workers, active_tiles.size());
    std::size_t tiled_raw_sum = 0U;
    for (const std::size_t raw : raw_pairs_per_tile) {
      if (!checked_add(tiled_raw_sum, raw))
        throw BuildFailure{"count_overflow"};
    }
    if (tiled_raw_sum != result.raw_row_pairs)
      throw BuildFailure{"tile_plan_mismatch"};
    result.tile_plan_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - tile_started)
            .count();

    std::array<std::vector<std::size_t>, 3> row_candidate_counts;
    for (std::size_t axis = 0U; axis < 3U; ++axis)
      row_candidate_counts[axis].assign(row_counts[axis], 0U);
    std::vector<WorkerScratch> worker_scratch(result.scheduled_workers);
    std::vector<std::uint8_t> count_workers(result.scheduled_workers, 0U);
    std::vector<std::uint8_t> fill_workers(result.scheduled_workers, 0U);
    const auto populate_tile_scratch = [&](std::size_t tile,
                                           std::vector<RowRefPair> &scratch) {
      scratch.clear();
      if (tile >= global_tile_count)
        throw BuildFailure{"tile_range"};
      const std::size_t expected_raw = raw_pairs_per_tile[tile];
      if (expected_raw > options.per_tile_scratch_pair_limit)
        throw BuildFailure{"per_tile_scratch_limit"};
      scratch.reserve(expected_raw);
      std::size_t axis = 0U;
      while (axis + 1U < layouts.size() &&
             tile >= layouts[axis + 1U].global_begin)
        ++axis;
      const AxisTileLayout &layout = layouts[axis];
      const std::size_t local = tile - layout.global_begin;
      const std::size_t tu = local % layout.u_tiles;
      const std::size_t tv = local / layout.u_tiles;
      std::size_t tile_u_begin = 0U;
      std::size_t tile_v_begin = 0U;
      if (!checked_multiply(tu, options.tile_u_rows, tile_u_begin) ||
          !checked_multiply(tv, options.tile_v_rows, tile_v_begin))
        throw BuildFailure{"count_overflow"};
      const std::size_t tile_u_end =
          tile_u_begin +
          std::min(options.tile_u_rows, layout.u_count - tile_u_begin);
      const std::size_t tile_v_end =
          tile_v_begin +
          std::min(options.tile_v_rows, layout.v_count - tile_v_begin);
      for (std::size_t position = tile_rectangle_offsets[tile];
           position < tile_rectangle_offsets[tile + 1U]; ++position) {
        const BoundRectangle &rectangle = rectangles[tile_rectangles[position]];
        const std::size_t u_begin = std::max(tile_u_begin, rectangle.u.begin);
        const std::size_t u_end = std::min(tile_u_end, rectangle.u.end);
        const std::size_t v_begin = std::max(tile_v_begin, rectangle.v.begin);
        const std::size_t v_end = std::min(tile_v_end, rectangle.v.end);
        for (std::size_t v = v_begin; v < v_end; ++v) {
          for (std::size_t u = u_begin; u < u_end; ++u) {
            std::size_t row = 0U;
            if (!checked_multiply(v, layout.u_count, row) ||
                !checked_add(row, u) || row >= row_counts[axis])
              throw BuildFailure{"count_overflow"};
            scratch.push_back({row, rectangle.ref});
          }
        }
      }
      if (scratch.size() != expected_raw)
        throw BuildFailure{"tile_plan_mismatch"};
      std::sort(scratch.begin(), scratch.end(), row_ref_less);
      scratch.erase(std::unique(scratch.begin(), scratch.end(), row_ref_equal),
                    scratch.end());
      return axis;
    };

    const auto run_dynamic_tiles = [&](bool fill_phase) {
      std::atomic<std::size_t> cursor{0U};
      const auto worker = [&](std::size_t worker_index) {
        if (worker_index >= worker_scratch.size())
          throw BuildFailure{"worker_range"};
        std::vector<RowRefPair> &scratch = worker_scratch[worker_index].pairs;
        while (true) {
          const std::size_t ordinal = cursor.fetch_add(1U);
          if (ordinal >= active_tiles.size())
            break;
          const std::size_t scheduled =
              options.reverse_tile_schedule_for_audit
                  ? active_tiles[active_tiles.size() - 1U - ordinal]
                  : active_tiles[ordinal];
          const std::size_t axis = populate_tile_scratch(scheduled, scratch);
          if (fill_phase) {
            fill_workers[worker_index] = 1U;
            for (const RowRefPair &pair : scratch) {
              const std::size_t position =
                  row_candidate_counts[axis][pair.row]++;
              if (position >= result.axes[axis].candidates.size())
                throw BuildFailure{"count_overflow"};
              result.axes[axis].candidates[position] = pair.ref;
            }
          } else {
            count_workers[worker_index] = 1U;
            for (const RowRefPair &pair : scratch) {
              if (!checked_add(row_candidate_counts[axis][pair.row], 1U))
                throw BuildFailure{"count_overflow"};
            }
          }
        }
      };
      if (executor != nullptr && result.scheduled_workers > 1U &&
          !active_tiles.empty())
        executor->parallel_do(result.scheduled_workers, worker);
      else
        worker(0U);
    };

    const auto count_started = Clock::now();
    run_dynamic_tiles(false);
    result.count_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - count_started)
            .count();
    result.count_workers_used = static_cast<std::size_t>(std::count(
        count_workers.begin(), count_workers.end(), std::uint8_t{1U}));

    const auto prefix_started = Clock::now();
    std::size_t final_pairs = 0U;
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      std::size_t sentinel = row_counts[axis];
      if (!checked_add(sentinel, 1U))
        throw BuildFailure{"count_overflow"};
      result.axes[axis].offsets.assign(sentinel, 0U);
      std::size_t prefix_pairs = 0U;
      for (std::size_t row = 0U; row < row_counts[axis]; ++row) {
        if (!checked_add(prefix_pairs, row_candidate_counts[axis][row]))
          throw BuildFailure{"count_overflow"};
        result.axes[axis].offsets[row + 1U] = prefix_pairs;
      }
      if (!checked_add(final_pairs, prefix_pairs) ||
          final_pairs > options.final_candidate_pair_limit)
        throw BuildFailure{"final_candidate_limit"};
    }
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      const std::size_t prefix_pairs = result.axes[axis].offsets.back();
      result.axes[axis].candidates.resize(prefix_pairs);
      for (std::size_t row = 0U; row < row_counts[axis]; ++row)
        row_candidate_counts[axis][row] = result.axes[axis].offsets[row];
    }
    result.deduplicated_candidate_pairs = final_pairs;
    result.prefix_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - prefix_started)
            .count();

    const auto fill_started = Clock::now();
    run_dynamic_tiles(true);
    result.fill_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - fill_started)
            .count();
    result.fill_workers_used = static_cast<std::size_t>(
        std::count(fill_workers.begin(), fill_workers.end(), std::uint8_t{1U}));
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      for (std::size_t row = 0U; row < row_counts[axis]; ++row) {
        if (row_candidate_counts[axis][row] !=
            result.axes[axis].offsets[row + 1U])
          throw BuildFailure{"fill_mismatch"};
      }
    }

    std::size_t retained = 0U;
    if (!retained_payload_bytes(result, retained))
      throw BuildFailure{"count_overflow"};
    result.retained_bytes = retained;
    std::size_t observed_scratch = 0U;
    for (const WorkerScratch &scratch : worker_scratch) {
      if (!checked_payload(scratch.pairs.capacity(), sizeof(RowRefPair),
                           observed_scratch))
        throw BuildFailure{"count_overflow"};
    }
    result.observed_worker_scratch_bytes = observed_scratch;
    std::size_t tiled_phase_upper = retained;
    if (!add_vector_capacity(rectangles, tiled_phase_upper) ||
        !add_vector_capacity(tile_rectangle_offsets, tiled_phase_upper) ||
        !add_vector_capacity(tile_rectangles, tiled_phase_upper) ||
        !add_vector_capacity(active_tiles, tiled_phase_upper) ||
        !add_vector_capacity(raw_pairs_per_tile, tiled_phase_upper) ||
        !add_vector_capacity(count_workers, tiled_phase_upper) ||
        !add_vector_capacity(fill_workers, tiled_phase_upper))
      throw BuildFailure{"count_overflow"};
    if (!checked_payload(worker_scratch.capacity(), sizeof(WorkerScratch),
                         tiled_phase_upper))
      throw BuildFailure{"count_overflow"};
    for (const auto &counts : row_candidate_counts) {
      if (!add_vector_capacity(counts, tiled_phase_upper))
        throw BuildFailure{"count_overflow"};
    }
    std::size_t deterministic_scratch_upper = 0U;
    if (!checked_multiply(result.scheduled_workers,
                          result.maximum_raw_pairs_per_tile,
                          deterministic_scratch_upper) ||
        !checked_multiply(deterministic_scratch_upper, sizeof(RowRefPair),
                          deterministic_scratch_upper))
      throw BuildFailure{"count_overflow"};
    const std::size_t scratch_upper =
        std::max(deterministic_scratch_upper, observed_scratch);
    if (!checked_add(tiled_phase_upper, scratch_upper))
      throw BuildFailure{"count_overflow"};
    std::size_t bounds_phase_upper = 0U;
    if (!checked_multiply(rectangles.capacity(), sizeof(BoundRectangle),
                          bounds_phase_upper))
      throw BuildFailure{"count_overflow"};
    if (!checked_add(bounds_phase_upper, maximum_bound_plan_scratch))
      throw BuildFailure{"count_overflow"};
    result.accounted_peak_temporary_upper_bound_bytes = std::max(
        {bounds_phase_upper, tile_plan_phase_upper, tiled_phase_upper});
    result.tiled_worklist_complete = true;
    if (result.torus_full_fallback_plans != 0U)
      result.status = "pass_with_full_fallback";
    else if (result.torus_theta_fallback_plans != 0U)
      result.status = "pass_with_theta_fallback";
    else
      result.status = "pass";
    result.build_ms = elapsed();
    return result;
  } catch (const std::bad_alloc &) {
    return fail("alloc_fail");
  } catch (const std::length_error &) {
    return fail("count_overflow");
  } catch (const BuildFailure &failure) {
    return fail(failure.status);
  }
}

bool validate_tiled_patch_ray_row_worklist(
    const PatchComplex &complex, const PatchMappingGridSpec &spec,
    const PreparedTiledPatchRayRowWorklist &prepared) noexcept {
  std::array<std::size_t, 3> row_counts{};
  const PatchComplexFingerprint fingerprint =
      fingerprint_patch_complex(complex);
  const PatchRayProvenance::LatticeIdentity lattice =
      make_patch_ray_lattice_identity(spec.lower, spec.grid_spacing, spec.nx,
                                      spec.ny, spec.nz);
  if (complex.status != "pass" ||
      !validate_patch_mapping_grid_spec(spec, &row_counts, nullptr) ||
      prepared.complex_fingerprint != fingerprint || !lattice.valid ||
      prepared.lattice_identity != lattice ||
      (prepared.status != "pass" &&
       prepared.status != "pass_with_theta_fallback" &&
       prepared.status != "pass_with_full_fallback") ||
      !prepared.tiled_worklist_complete || prepared.requested_workers == 0U ||
      prepared.tile_u_rows == 0U || prepared.tile_v_rows == 0U ||
      prepared.active_tiles > prepared.total_tiles ||
      prepared.scheduled_workers == 0U ||
      prepared.scheduled_workers > prepared.requested_workers ||
      prepared.count_workers_used > prepared.scheduled_workers ||
      prepared.fill_workers_used > prepared.scheduled_workers ||
      (prepared.active_tiles != 0U && (prepared.count_workers_used == 0U ||
                                       prepared.fill_workers_used == 0U)) ||
      !std::isfinite(prepared.bounds_plan_ms) ||
      prepared.bounds_plan_ms < 0.0 || !std::isfinite(prepared.tile_plan_ms) ||
      prepared.tile_plan_ms < 0.0 || !std::isfinite(prepared.count_ms) ||
      prepared.count_ms < 0.0 || !std::isfinite(prepared.prefix_ms) ||
      prepared.prefix_ms < 0.0 || !std::isfinite(prepared.fill_ms) ||
      prepared.fill_ms < 0.0 || !std::isfinite(prepared.build_ms) ||
      prepared.build_ms < 0.0)
    return false;
  const std::size_t expected_scheduled_workers =
      prepared.active_tiles == 0U
          ? 1U
          : std::min(prepared.requested_workers, prepared.active_tiles);
  std::size_t torus_plan_count = prepared.torus_domain_plans;
  if (prepared.scheduled_workers != expected_scheduled_workers ||
      !checked_add(torus_plan_count, prepared.torus_theta_fallback_plans) ||
      !checked_add(torus_plan_count, prepared.torus_full_fallback_plans) ||
      torus_plan_count != complex.tori.size())
    return false;
  const std::string_view expected_status =
      prepared.torus_full_fallback_plans != 0U    ? "pass_with_full_fallback"
      : prepared.torus_theta_fallback_plans != 0U ? "pass_with_theta_fallback"
                                                  : "pass";
  if (prepared.status != expected_status ||
      prepared.raw_row_pairs < prepared.deduplicated_candidate_pairs ||
      prepared.rectangle_tile_pairs < prepared.active_tiles ||
      prepared.maximum_raw_pairs_per_tile > prepared.raw_row_pairs)
    return false;
  std::size_t total_rows = 0U;
  std::size_t candidate_pairs = 0U;
  std::size_t expected_total_tiles = 0U;
  for (std::size_t axis_index = 0U; axis_index < 3U; ++axis_index) {
    const PatchRayAxisRowWorklist &axis = prepared.axes[axis_index];
    double lower_u = 0.0;
    double lower_v = 0.0;
    std::size_t u_count = 0U;
    std::size_t v_count = 0U;
    std::size_t offset_sentinel = row_counts[axis_index];
    if (!checked_add(offset_sentinel, 1U))
      return false;
    if (axis.axis != static_cast<RayAxis>(axis_index) ||
        !axis_grid(spec, axis.axis, lower_u, u_count, lower_v, v_count) ||
        axis.u_count != u_count || axis.v_count != v_count ||
        axis.offsets.size() != offset_sentinel || axis.offsets.empty() ||
        axis.offsets.front() != 0U ||
        axis.offsets.back() != axis.candidates.size() ||
        !checked_add(total_rows, row_counts[axis_index]) ||
        !checked_add(candidate_pairs, axis.candidates.size()))
      return false;
    const std::size_t u_tiles =
        u_count / prepared.tile_u_rows +
        static_cast<std::size_t>(u_count % prepared.tile_u_rows != 0U);
    const std::size_t v_tiles =
        v_count / prepared.tile_v_rows +
        static_cast<std::size_t>(v_count % prepared.tile_v_rows != 0U);
    std::size_t axis_tiles = 0U;
    if (!checked_multiply(u_tiles, v_tiles, axis_tiles) ||
        !checked_add(expected_total_tiles, axis_tiles))
      return false;
    for (std::size_t row = 0U; row < row_counts[axis_index]; ++row) {
      const std::size_t begin = axis.offsets[row];
      const std::size_t end = axis.offsets[row + 1U];
      if (begin > end || end > axis.candidates.size())
        return false;
      for (std::size_t position = begin; position < end; ++position) {
        if (!patch_ref_in_range(complex, axis.candidates[position]) ||
            (position > begin &&
             !(axis.candidates[position - 1U] < axis.candidates[position])))
          return false;
      }
    }
  }
  std::size_t retained = 0U;
  std::size_t retained_plus_scratch = prepared.retained_bytes;
  return total_rows == prepared.total_rows &&
         expected_total_tiles == prepared.total_tiles &&
         candidate_pairs == prepared.deduplicated_candidate_pairs &&
         retained_payload_bytes(prepared, retained) &&
         retained == prepared.retained_bytes &&
         checked_add(retained_plus_scratch,
                     prepared.observed_worker_scratch_bytes) &&
         prepared.accounted_peak_temporary_upper_bound_bytes >=
             retained_plus_scratch;
}

} // namespace TitanPBE::Tabi::analytic_ses

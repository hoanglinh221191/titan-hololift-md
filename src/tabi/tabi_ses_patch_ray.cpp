#include "tabi_ses_patch_ray.h"

#include "titan_executor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace TitanPBE::Tabi::analytic_ses {
namespace {

constexpr std::uint64_t kAxisShift = 62U;
constexpr std::uint64_t kRowMask = (std::uint64_t{1} << kAxisShift) - 1U;
constexpr std::uint64_t kProvenanceRowMask = (std::uint64_t{1} << 30U) - 1U;
constexpr std::uint64_t kProvenanceRefMask =
    std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] bool checked_add(std::size_t left, std::size_t right,
                               std::size_t &result) noexcept {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    return false;
  result = left + right;
  return true;
}

[[nodiscard]] bool checked_add_assign(std::size_t &result,
                                      std::size_t value) noexcept {
  return checked_add(result, value, result);
}

[[nodiscard]] bool checked_multiply(std::size_t left, std::size_t right,
                                    std::size_t &result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_vector_bytes(std::size_t capacity, std::size_t width,
                                        std::size_t &bytes) noexcept {
  std::size_t contribution = 0U;
  return checked_multiply(capacity, width, contribution) &&
         checked_add_assign(bytes, contribution);
}

[[nodiscard]] double canonical_root_coordinate(double coordinate) noexcept {
  return coordinate == 0.0 ? 0.0 : coordinate;
}

void increment_root_multiplicity_or_invalidate(
    std::uint32_t &multiplicity) noexcept {
  if (multiplicity == 0U ||
      multiplicity == std::numeric_limits<std::uint32_t>::max()) {
    multiplicity = 0U;
    return;
  }
  ++multiplicity;
}

struct Event {
  std::uint64_t key{0};
  double coordinate{0.0};
};

static_assert(sizeof(Event) == 16);

// Audit-on keeps the event slab at the production 16-byte width. The upper
// 32 bits retain (axis,row) and the lower 32 bits retain PatchRefId.
struct ProvenanceEvent {
  std::uint64_t key{0};
  double coordinate{0.0};
};

static_assert(sizeof(ProvenanceEvent) == sizeof(Event));

[[nodiscard]] std::uint64_t event_key(RayAxis axis, std::size_t row) noexcept {
  return (static_cast<std::uint64_t>(axis) << kAxisShift) |
         static_cast<std::uint64_t>(row);
}

[[nodiscard]] RayAxis event_axis(std::uint64_t key) noexcept {
  return static_cast<RayAxis>(key >> kAxisShift);
}

[[nodiscard]] std::size_t event_row(std::uint64_t key) noexcept {
  return static_cast<std::size_t>(key & kRowMask);
}

[[nodiscard]] bool event_less(const Event &a, const Event &b) noexcept {
  if (a.key != b.key)
    return a.key < b.key;
  return a.coordinate < b.coordinate;
}

[[nodiscard]] std::uint64_t provenance_event_key(RayAxis axis, std::size_t row,
                                                 PatchRefId patch) noexcept {
  return (static_cast<std::uint64_t>(axis) << kAxisShift) |
         (static_cast<std::uint64_t>(row) << 32U) |
         static_cast<std::uint64_t>(patch.packed());
}

[[nodiscard]] std::uint64_t provenance_ray_key(std::uint64_t key) noexcept {
  return key >> 32U;
}

[[nodiscard]] PatchRefId provenance_event_patch(std::uint64_t key) noexcept {
  PatchRefId result;
  const auto packed = static_cast<std::uint32_t>(key & kProvenanceRefMask);
  const auto kind = static_cast<PatchKind>(packed >> PatchRefId::index_bits);
  const auto index = static_cast<std::size_t>(packed & PatchRefId::index_mask);
  if (!PatchRefId::try_make(kind, index, result))
    return {};
  return result;
}

[[nodiscard]] bool provenance_event_less(const ProvenanceEvent &a,
                                         const ProvenanceEvent &b) noexcept {
  const std::uint64_t a_ray = provenance_ray_key(a.key);
  const std::uint64_t b_ray = provenance_ray_key(b.key);
  if (a_ray != b_ray)
    return a_ray < b_ray;
  if (a.coordinate != b.coordinate)
    return a.coordinate < b.coordinate;
  return (a.key & kProvenanceRefMask) < (b.key & kProvenanceRefMask);
}

struct AxisSpec {
  RayAxis axis{RayAxis::x};
  double lower_u{0.0};
  double lower_v{0.0};
  std::size_t nu{0};
  std::size_t nv{0};
};

[[nodiscard]] AxisSpec axis_spec(RayAxis axis, const Vec3 &lower,
                                 std::size_t nx, std::size_t ny,
                                 std::size_t nz) noexcept {
  switch (axis) {
  case RayAxis::x:
    return {axis, lower.y, lower.z, ny, nz};
  case RayAxis::y:
    return {axis, lower.x, lower.z, nx, nz};
  case RayAxis::z:
    return {axis, lower.x, lower.y, nx, ny};
  }
  return {};
}

[[nodiscard]] std::pair<double, double>
projected_bounds(const PatchBounds &bounds, RayAxis axis,
                 bool first_coordinate) noexcept {
  switch (axis) {
  case RayAxis::x:
    return first_coordinate ? std::pair{bounds.lower.y, bounds.upper.y}
                            : std::pair{bounds.lower.z, bounds.upper.z};
  case RayAxis::y:
    return first_coordinate ? std::pair{bounds.lower.x, bounds.upper.x}
                            : std::pair{bounds.lower.z, bounds.upper.z};
  case RayAxis::z:
    return first_coordinate ? std::pair{bounds.lower.x, bounds.upper.x}
                            : std::pair{bounds.lower.y, bounds.upper.y};
  }
  return {};
}

[[nodiscard]] bool finite_ordered_bounds(const PatchBounds &bounds) noexcept {
  return std::isfinite(bounds.lower.x) && std::isfinite(bounds.lower.y) &&
         std::isfinite(bounds.lower.z) && std::isfinite(bounds.upper.x) &&
         std::isfinite(bounds.upper.y) && std::isfinite(bounds.upper.z) &&
         bounds.lower.x <= bounds.upper.x && bounds.lower.y <= bounds.upper.y &&
         bounds.lower.z <= bounds.upper.z;
}

[[nodiscard]] std::pair<std::size_t, std::size_t>
grid_interval(double lo, double hi, double grid_lower, double spacing,
              std::size_t count) noexcept {
  if (count == 0U || !std::isfinite(lo) || !std::isfinite(hi) ||
      !std::isfinite(grid_lower) || !std::isfinite(spacing) ||
      !(spacing > 0.0) || lo > hi)
    return {1U, 0U};
  const double grid_upper =
      grid_lower + spacing * static_cast<double>(count - 1U);
  if (!std::isfinite(grid_upper) || hi < grid_lower || lo > grid_upper)
    return {1U, 0U};
  const double scale = std::max({1.0, std::abs(lo), std::abs(hi), spacing});
  const double padding = 64.0 * std::numeric_limits<double>::epsilon() * scale;
  const double first = std::ceil((lo - padding - grid_lower) / spacing);
  const double last = std::floor((hi + padding - grid_lower) / spacing);
  if (!std::isfinite(first) || !std::isfinite(last) || last < 0.0 ||
      first > static_cast<double>(count - 1U))
    return {1U, 0U};
  const double clamped_first = std::max(0.0, first);
  const double clamped_last = std::min(static_cast<double>(count - 1U), last);
  if (clamped_first > clamped_last)
    return {1U, 0U};
  return {static_cast<std::size_t>(clamped_first),
          static_cast<std::size_t>(clamped_last)};
}

struct PatchAddress {
  PatchKind kind{PatchKind::contact};
  std::size_t index{0};
};

[[nodiscard]] PatchAddress patch_address(const PatchComplex &complex,
                                         std::size_t linear) noexcept {
  if (linear < complex.contacts.size())
    return {PatchKind::contact, linear};
  linear -= complex.contacts.size();
  if (linear < complex.tori.size())
    return {PatchKind::toroidal, linear};
  linear -= complex.tori.size();
  return {PatchKind::reentrant, linear};
}

template <class EventType, class Less, class Visitor>
bool for_each_merged_event(
    const std::vector<std::vector<EventType>> &per_worker, Less less,
    Visitor &&visitor, std::size_t *scratch_payload_bytes = nullptr) {
  struct Cursor {
    std::size_t worker{0};
    std::size_t position{0};
  };
  const auto greater = [&](const Cursor &a, const Cursor &b) {
    return less(per_worker[b.worker][b.position],
                per_worker[a.worker][a.position]);
  };
  std::vector<Cursor> queue;
  queue.reserve(per_worker.size());
  for (std::size_t worker = 0; worker < per_worker.size(); ++worker) {
    if (!per_worker[worker].empty()) {
      queue.push_back({worker, 0U});
      std::push_heap(queue.begin(), queue.end(), greater);
    }
  }
  while (!queue.empty()) {
    std::pop_heap(queue.begin(), queue.end(), greater);
    Cursor cursor = queue.back();
    queue.pop_back();
    visitor(per_worker[cursor.worker][cursor.position]);
    ++cursor.position;
    if (cursor.position < per_worker[cursor.worker].size()) {
      queue.push_back(cursor);
      std::push_heap(queue.begin(), queue.end(), greater);
    }
  }
  if (scratch_payload_bytes != nullptr) {
    std::size_t bytes = 0U;
    if (!checked_vector_bytes(queue.capacity(), sizeof(Cursor), bytes))
      return false;
    *scratch_payload_bytes = std::max(*scratch_payload_bytes, bytes);
  }
  return true;
}

template <class Feed, class Visitor>
bool for_each_patch_coordinate_group(
    Feed &&feed, double tolerance, Visitor &&visitor,
    std::size_t *scratch_payload_bytes = nullptr) {
  std::uint64_t current_ray = std::numeric_limits<std::uint64_t>::max();
  double current_coordinate = 0.0;
  std::vector<PatchRefId> refs;
  std::vector<std::uint32_t> multiplicities;
  bool scratch_valid = true;
  const auto update_scratch = [&]() {
    if (scratch_payload_bytes == nullptr)
      return;
    std::size_t bytes = 0U;
    if (!checked_vector_bytes(refs.capacity(), sizeof(PatchRefId), bytes) ||
        !checked_vector_bytes(multiplicities.capacity(), sizeof(std::uint32_t),
                              bytes)) {
      scratch_valid = false;
      return;
    }
    *scratch_payload_bytes = std::max(*scratch_payload_bytes, bytes);
  };
  const auto flush = [&]() {
    if (refs.empty())
      return;
    std::sort(refs.begin(), refs.end());
    multiplicities.clear();
    std::size_t unique_count = 0U;
    for (const PatchRefId ref : refs) {
      if (unique_count == 0U || refs[unique_count - 1U] != ref) {
        refs[unique_count++] = ref;
        multiplicities.push_back(1U);
      } else {
        // A patch-axis primitive has at most four algebraic roots, so reaching
        // this limit means the event stream is malformed. The zero sentinel is
        // sticky and makes transactional provenance append fail closed.
        increment_root_multiplicity_or_invalidate(multiplicities.back());
      }
    }
    refs.resize(unique_count);
    update_scratch();
    visitor(current_ray, current_coordinate, std::span<const PatchRefId>(refs),
            std::span<const std::uint32_t>(multiplicities));
    refs.clear();
  };
  const bool feed_valid =
      feed([&](std::uint64_t ray, double coordinate, PatchRefId patch) {
        if (refs.empty() || ray != current_ray ||
            std::abs(coordinate - current_coordinate) > tolerance) {
          flush();
          current_ray = ray;
          current_coordinate = coordinate;
        }
        refs.push_back(patch);
        update_scratch();
      });
  flush();
  return scratch_valid && feed_valid;
}

void finalize_provenance_certificate(PatchRayGrid &grid,
                                     std::size_t total_odd_rows,
                                     bool csr_valid) {
  if (!grid.provenance.enabled)
    return;
  grid.provenance.certified =
      csr_valid && grid.status == "pass" && total_odd_rows == 0U;
  if (!csr_valid) {
    grid.provenance.status = "csr_invalid";
  } else if (total_odd_rows != 0U) {
    grid.provenance.status = "unresolved_odd_rows";
  } else {
    grid.provenance.status = grid.provenance.certified ? "pass" : grid.status;
  }
}

} // namespace

PatchRayProvenance::LatticeIdentity make_patch_ray_lattice_identity(
    const Vec3 &lower, double grid_spacing, std::size_t nx, std::size_t ny,
    std::size_t nz) noexcept {
  PatchRayProvenance::LatticeIdentity result;
  if (!std::isfinite(lower.x) || !std::isfinite(lower.y) ||
      !std::isfinite(lower.z) || !std::isfinite(grid_spacing) ||
      !(grid_spacing > 0.0) || nx == 0U || ny == 0U || nz == 0U)
    return result;
  result.scalar_bits = {std::bit_cast<std::uint64_t>(lower.x),
                        std::bit_cast<std::uint64_t>(lower.y),
                        std::bit_cast<std::uint64_t>(lower.z),
                        std::bit_cast<std::uint64_t>(grid_spacing)};
  result.counts = {nx, ny, nz};
  result.valid = true;
  return result;
}

static PatchRayGrid
build_patch_ray_grid_impl(const PatchComplex &complex, const Vec3 &lower,
                          double grid_spacing, std::size_t nx, std::size_t ny,
                          std::size_t nz, TitanExecutor *executor,
                          PatchRayBuildOptions options) {
  const auto begin = std::chrono::steady_clock::now();
  PatchRayGrid result;
  result.provenance.enabled = options.retain_provenance;
  result.provenance.status =
      options.retain_provenance ? "building" : "disabled";
  result.axes[0].axis = RayAxis::x;
  result.axes[0].u_count = ny;
  result.axes[0].v_count = nz;
  result.axes[1].axis = RayAxis::y;
  result.axes[1].u_count = nx;
  result.axes[1].v_count = nz;
  result.axes[2].axis = RayAxis::z;
  result.axes[2].u_count = nx;
  result.axes[2].v_count = ny;
  std::size_t total_patches = complex.contacts.size();
  if (!checked_add_assign(total_patches, complex.tori.size()) ||
      !checked_add_assign(total_patches, complex.reentrant.size())) {
    result.status = "patch_count_overflow";
    result.provenance.status = result.status;
    return result;
  }
  if (complex.status != "pass" || !std::isfinite(grid_spacing) ||
      !(grid_spacing > 0.0) || !std::isfinite(lower.x) ||
      !std::isfinite(lower.y) || !std::isfinite(lower.z) || nx < 2 || ny < 2 ||
      nz < 2 || total_patches == 0) {
    result.status = "invalid_input";
    result.provenance.status = result.status;
    return result;
  }
  if (executor != nullptr && executor->capacity() == 0U) {
    result.status = "invalid_executor_capacity";
    result.provenance.status = result.status;
    return result;
  }
  for (const AxisRayCrossings &axis : result.axes) {
    const std::size_t row_count = axis.row_count();
    if (row_count == std::numeric_limits<std::size_t>::max() ||
        row_count >= (std::uint64_t{1} << kAxisShift)) {
      result.status = "ray_grid_too_large";
      result.provenance.status = result.status;
      return result;
    }
    if (options.retain_provenance && row_count > kProvenanceRowMask) {
      result.status = "provenance_row_overflow";
      result.provenance.status = result.status;
      return result;
    }
  }
  if (options.retain_provenance &&
      (complex.contacts.size() >
           static_cast<std::size_t>(PatchRefId::index_mask) + 1U ||
       complex.tori.size() >
           static_cast<std::size_t>(PatchRefId::index_mask) + 1U ||
       complex.reentrant.size() >
           static_cast<std::size_t>(PatchRefId::index_mask) + 1U)) {
    result.status = "provenance_patch_ref_overflow";
    result.provenance.status = result.status;
    return result;
  }
  if (options.retain_provenance) {
    result.provenance.lattice_identity = make_patch_ray_lattice_identity(
        lower, grid_spacing, nx, ny, nz);
    if (!result.provenance.lattice_identity.valid) {
      result.status = "invalid_lattice_identity";
      result.provenance.status = result.status;
      return result;
    }
    result.provenance.complex_fingerprint = fingerprint_patch_complex(complex);
    if (!result.provenance.complex_fingerprint.valid) {
      result.status = "invalid_complex_fingerprint";
      result.provenance.status = result.status;
      return result;
    }
  }

  const ReentrantProbeIndex probe_index = build_reentrant_probe_index(complex);
  std::size_t probe_index_bytes = 0U;
  if (!checked_vector_bytes(probe_index.cells.capacity(), sizeof(ProbeCell),
                            probe_index_bytes) ||
      !checked_vector_bytes(probe_index.probe_indices.capacity(),
                            sizeof(std::uint32_t), probe_index_bytes)) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  result.peak_temporary_bytes = probe_index_bytes;
  const std::size_t workers =
      executor == nullptr ? 1U : std::min(executor->capacity(), total_patches);
  std::vector<std::vector<Event>> events(workers);
  std::vector<std::vector<ProvenanceEvent>> provenance_events(
      options.retain_provenance ? workers : 0U);
  std::vector<std::vector<std::uint64_t>> uncertified_rows(workers);
  std::vector<std::size_t> uncertified(workers, 0U);
  std::atomic<bool> invalid_patch_ref{false};
  std::atomic<bool> invalid_patch_geometry{false};
  std::atomic<bool> worker_counter_overflow{false};
  const auto process = [&](std::size_t worker, std::size_t patch_begin,
                           std::size_t patch_end) {
    std::vector<Event> &local = events[worker];
    std::vector<ProvenanceEvent> *local_provenance =
        options.retain_provenance ? &provenance_events[worker] : nullptr;
    std::vector<std::uint64_t> &local_uncertified_rows =
        uncertified_rows[worker];
    for (std::size_t linear = patch_begin; linear < patch_end; ++linear) {
      const PatchAddress address = patch_address(complex, linear);
      PatchRefId patch_ref;
      if (options.retain_provenance &&
          !PatchRefId::try_make(address.kind, address.index, patch_ref)) {
        invalid_patch_ref.store(true, std::memory_order_relaxed);
        continue;
      }
      const PatchBounds bounds =
          patch_bounds(complex, address.kind, address.index);
      if (!finite_ordered_bounds(bounds)) {
        invalid_patch_geometry.store(true, std::memory_order_relaxed);
        continue;
      }
      for (RayAxis axis : {RayAxis::x, RayAxis::y, RayAxis::z}) {
        const AxisSpec spec = axis_spec(axis, lower, nx, ny, nz);
        const auto u_bounds = projected_bounds(bounds, axis, true);
        const auto v_bounds = projected_bounds(bounds, axis, false);
        const auto u_range = grid_interval(u_bounds.first, u_bounds.second,
                                           spec.lower_u, grid_spacing, spec.nu);
        const auto v_range = grid_interval(v_bounds.first, v_bounds.second,
                                           spec.lower_v, grid_spacing, spec.nv);
        if (u_range.first > u_range.second || v_range.first > v_range.second)
          continue;
        for (std::size_t iv = v_range.first; iv <= v_range.second; ++iv) {
          const double v =
              spec.lower_v + grid_spacing * static_cast<double>(iv);
          for (std::size_t iu = u_range.first; iu <= u_range.second; ++iu) {
            const double u =
                spec.lower_u + grid_spacing * static_cast<double>(iu);
            const PatchAxisRoots roots = intersect_patch_axis(
                complex, probe_index, address.kind, address.index, axis, u, v);
            const std::size_t row = iu + spec.nu * iv;
            const std::uint64_t key = event_key(axis, row);
            if (!roots.certified) {
              if (!checked_add_assign(uncertified[worker], 1U))
                worker_counter_overflow.store(true, std::memory_order_relaxed);
              local_uncertified_rows.push_back(key);
              continue;
            }
            for (std::size_t root = 0; root < roots.count; ++root) {
              const double coordinate =
                  canonical_root_coordinate(roots.coordinates[root]);
              if (!std::isfinite(coordinate)) {
                invalid_patch_geometry.store(true, std::memory_order_relaxed);
                continue;
              }
              if (local_provenance != nullptr) {
                local_provenance->push_back(
                    {provenance_event_key(axis, row, patch_ref), coordinate});
              } else {
                local.push_back({key, coordinate});
              }
            }
          }
        }
      }
    }
    if (local_provenance != nullptr) {
      std::sort(local_provenance->begin(), local_provenance->end(),
                provenance_event_less);
    } else {
      std::sort(local.begin(), local.end(), event_less);
    }
    std::sort(local_uncertified_rows.begin(), local_uncertified_rows.end());
    local_uncertified_rows.erase(std::unique(local_uncertified_rows.begin(),
                                             local_uncertified_rows.end()),
                                 local_uncertified_rows.end());
  };

  if (executor != nullptr && workers > 1U) {
    executor->parallel_for(total_patches, workers, 32U, process);
  } else {
    process(0U, 0U, total_patches);
  }
  if (invalid_patch_ref.load(std::memory_order_relaxed)) {
    result.status = "provenance_patch_ref_overflow";
    result.provenance.status = result.status;
    return result;
  }
  if (invalid_patch_geometry.load(std::memory_order_relaxed)) {
    result.status = "invalid_patch_geometry";
    result.provenance.status = result.status;
    return result;
  }
  if (worker_counter_overflow.load(std::memory_order_relaxed)) {
    result.status = "counter_overflow";
    result.provenance.status = result.status;
    return result;
  }

  std::size_t event_slab_bytes = 0U;
  bool telemetry_valid =
      checked_vector_bytes(events.capacity(), sizeof(std::vector<Event>),
                           event_slab_bytes) &&
      checked_vector_bytes(provenance_events.capacity(),
                           sizeof(std::vector<ProvenanceEvent>),
                           event_slab_bytes) &&
      checked_vector_bytes(uncertified_rows.capacity(),
                           sizeof(std::vector<std::uint64_t>),
                           event_slab_bytes) &&
      checked_vector_bytes(uncertified.capacity(), sizeof(std::size_t),
                           event_slab_bytes);
  if (options.retain_provenance) {
    for (const std::vector<ProvenanceEvent> &worker_events : provenance_events)
      telemetry_valid =
          telemetry_valid &&
          checked_vector_bytes(worker_events.capacity(),
                               sizeof(ProvenanceEvent), event_slab_bytes);
  } else {
    for (const std::vector<Event> &worker_events : events)
      telemetry_valid = telemetry_valid &&
                        checked_vector_bytes(worker_events.capacity(),
                                             sizeof(Event), event_slab_bytes);
  }
  for (const std::vector<std::uint64_t> &worker_rows : uncertified_rows)
    telemetry_valid =
        telemetry_valid &&
        checked_vector_bytes(worker_rows.capacity(), sizeof(std::uint64_t),
                             event_slab_bytes);
  if (!telemetry_valid) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  std::size_t initial_live_bytes = 0U;
  if (!checked_add(probe_index_bytes, event_slab_bytes, initial_live_bytes)) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  result.peak_temporary_bytes =
      std::max(result.peak_temporary_bytes, initial_live_bytes);
  if (options.retain_provenance) {
    for (const std::vector<ProvenanceEvent> &worker_events : provenance_events)
      telemetry_valid =
          telemetry_valid &&
          checked_add_assign(result.raw_events, worker_events.size());
  } else {
    for (const std::vector<Event> &worker_events : events)
      telemetry_valid =
          telemetry_valid &&
          checked_add_assign(result.raw_events, worker_events.size());
  }
  for (const std::size_t count : uncertified)
    telemetry_valid =
        telemetry_valid &&
        checked_add_assign(result.uncertified_intersection_queries, count);
  if (!telemetry_valid) {
    result.status = "counter_overflow";
    result.provenance.status = result.status;
    return result;
  }
  std::array<std::vector<std::size_t>, 3> row_counts;
  std::array<std::vector<std::uint8_t>, 3> row_uncertified;
  for (std::size_t axis = 0; axis < 3; ++axis)
    row_counts[axis].assign(result.axes[axis].row_count(), 0U);
  for (std::size_t axis = 0; axis < 3; ++axis)
    row_uncertified[axis].assign(result.axes[axis].row_count(), 0U);
  for (const std::vector<std::uint64_t> &worker_rows : uncertified_rows) {
    for (const std::uint64_t key : worker_rows) {
      const std::size_t axis = static_cast<std::size_t>(event_axis(key));
      const std::size_t row = event_row(key);
      if (axis < row_uncertified.size() && row < row_uncertified[axis].size())
        row_uncertified[axis][row] = 1U;
    }
  }
  for (std::vector<std::uint64_t> &worker_rows : uncertified_rows)
    std::vector<std::uint64_t>().swap(worker_rows);
  std::vector<std::vector<std::uint64_t>>().swap(uncertified_rows);
  std::size_t row_count_bytes = 0U;
  std::size_t row_uncertified_bytes = 0U;
  for (const std::vector<std::size_t> &counts : row_counts)
    telemetry_valid =
        telemetry_valid &&
        checked_vector_bytes(counts.capacity(), sizeof(std::size_t),
                             row_count_bytes);
  for (const std::vector<std::uint8_t> &flags : row_uncertified) {
    telemetry_valid =
        telemetry_valid &&
        checked_vector_bytes(flags.capacity(), sizeof(std::uint8_t),
                             row_uncertified_bytes);
  }
  telemetry_valid = telemetry_valid &&
                    checked_add_assign(row_count_bytes, row_uncertified_bytes);
  if (!telemetry_valid) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  double local_scale = std::max(1.0, grid_spacing);
  for (const Atom &atom : complex.atoms)
    local_scale = std::max(local_scale, atom.radius + complex.probe_radius);
  const double duplicate_tolerance =
      std::max(512.0 * std::numeric_limits<double>::epsilon() * local_scale,
               1.0e-11 * grid_spacing);
  if (options.retain_provenance) {
    result.provenance.coordinate_merge_tolerance = duplicate_tolerance;
    // Independent, scale-aware numerical uncertainty band for audit only.
    // It is derived from sqrt(machine epsilon), not a fixed Angstrom cutoff,
    // and never removes/collapses either crossing.
    result.provenance.near_pair_tolerance =
        8.0 * std::sqrt(std::numeric_limits<double>::epsilon()) *
        std::max(local_scale, grid_spacing);
  }
  std::size_t merge_queue_bytes = 0U;
  std::size_t group_scratch_bytes = 0U;
  std::size_t repair_group_scratch_bytes = 0U;
  bool merge_traversal_valid = true;
  const auto recompute_perturbed_row = [&](RayAxis axis, std::size_t row,
                                           double delta_u, double delta_v,
                                           std::vector<double> &coordinates) {
    coordinates.clear();
    const AxisSpec spec = axis_spec(axis, lower, nx, ny, nz);
    const std::size_t iu = row % spec.nu;
    const std::size_t iv = row / spec.nu;
    const double u =
        spec.lower_u + grid_spacing * static_cast<double>(iu) + delta_u;
    const double v =
        spec.lower_v + grid_spacing * static_cast<double>(iv) + delta_v;
    for (std::size_t linear = 0; linear < total_patches; ++linear) {
      const PatchAddress address = patch_address(complex, linear);
      const PatchAxisRoots roots = intersect_patch_axis(
          complex, probe_index, address.kind, address.index, axis, u, v);
      if (!roots.certified)
        return false;
      for (std::size_t root = 0; root < roots.count; ++root) {
        const double coordinate =
            canonical_root_coordinate(roots.coordinates[root]);
        if (!std::isfinite(coordinate))
          return false;
        coordinates.push_back(coordinate);
      }
    }
    std::sort(coordinates.begin(), coordinates.end());
    coordinates.erase(std::unique(coordinates.begin(), coordinates.end(),
                                  [&](double first, double second) {
                                    return std::abs(first - second) <=
                                           duplicate_tolerance;
                                  }),
                      coordinates.end());
    return true;
  };
  const auto recompute_perturbed_row_with_provenance =
      [&](RayAxis axis, std::size_t row, double delta_u, double delta_v,
          std::vector<double> &coordinates, AxisPatchProvenance &provenance,
          std::vector<ProvenanceEvent> &root_events) {
        coordinates.clear();
        provenance.canonical_refs.clear();
        provenance.seam_groups.clear();
        provenance.seam_refs.clear();
        provenance.seam_ref_multiplicities.clear();
        provenance.single_ref_multiplicities.clear();
        root_events.clear();
        const AxisSpec spec = axis_spec(axis, lower, nx, ny, nz);
        const std::size_t iu = row % spec.nu;
        const std::size_t iv = row / spec.nu;
        const double u =
            spec.lower_u + grid_spacing * static_cast<double>(iu) + delta_u;
        const double v =
            spec.lower_v + grid_spacing * static_cast<double>(iv) + delta_v;
        for (std::size_t linear = 0; linear < total_patches; ++linear) {
          const PatchAddress address = patch_address(complex, linear);
          PatchRefId patch_ref;
          if (!PatchRefId::try_make(address.kind, address.index, patch_ref))
            return false;
          const PatchAxisRoots roots = intersect_patch_axis(
              complex, probe_index, address.kind, address.index, axis, u, v);
          if (!roots.certified)
            return false;
          for (std::size_t root = 0; root < roots.count; ++root) {
            const double coordinate =
                canonical_root_coordinate(roots.coordinates[root]);
            if (!std::isfinite(coordinate))
              return false;
            root_events.push_back(
                {provenance_event_key(axis, row, patch_ref), coordinate});
          }
        }
        std::sort(root_events.begin(), root_events.end(),
                  provenance_event_less);
        bool valid = true;
        const auto feed = [&](auto &&accept) {
          for (const ProvenanceEvent &event : root_events) {
            accept(provenance_ray_key(event.key), event.coordinate,
                   provenance_event_patch(event.key));
          }
          return true;
        };
        const bool grouped = for_each_patch_coordinate_group(
            feed, duplicate_tolerance,
            [&](std::uint64_t ray, double coordinate,
                std::span<const PatchRefId> refs,
                std::span<const std::uint32_t> multiplicities) {
              const std::uint64_t expected =
                  (static_cast<std::uint64_t>(axis) << 30U) |
                  static_cast<std::uint64_t>(row);
              if (ray != expected || !provenance.append(refs, multiplicities)) {
                valid = false;
                return;
              }
              coordinates.push_back(coordinate);
            },
            &repair_group_scratch_bytes);
        return grouped && valid && provenance.validate(coordinates.size());
      };

  std::uint64_t previous_key = std::numeric_limits<std::uint64_t>::max();
  double previous_coordinate = 0.0;
  bool invalid_event_key = false;
  if (options.retain_provenance) {
    std::size_t unique_crossings = 0U;
    const auto feed = [&](auto &&accept) {
      const bool merged = for_each_merged_event(
          provenance_events, provenance_event_less,
          [&](const ProvenanceEvent &event) {
            accept(provenance_ray_key(event.key), event.coordinate,
                   provenance_event_patch(event.key));
          },
          &merge_queue_bytes);
      merge_traversal_valid = merge_traversal_valid && merged;
      return merged;
    };
    const bool grouped = for_each_patch_coordinate_group(
        feed, duplicate_tolerance,
        [&](std::uint64_t ray, double, std::span<const PatchRefId> refs,
            std::span<const std::uint32_t>) {
          const std::size_t axis_index = static_cast<std::size_t>(ray >> 30U);
          const std::size_t row =
              static_cast<std::size_t>(ray & kProvenanceRowMask);
          if (axis_index >= result.axes.size() ||
              row >= result.axes[axis_index].row_count() || refs.empty()) {
            invalid_event_key = true;
            return;
          }
          if (row_counts[axis_index][row] ==
              std::numeric_limits<std::size_t>::max()) {
            invalid_event_key = true;
            return;
          }
          ++row_counts[axis_index][row];
          if (!checked_add_assign(unique_crossings, 1U))
            invalid_event_key = true;
        },
        &group_scratch_bytes);
    merge_traversal_valid = merge_traversal_valid && grouped;
    if (unique_crossings > result.raw_events) {
      invalid_event_key = true;
    } else {
      result.duplicate_events = result.raw_events - unique_crossings;
    }
  } else {
    merge_traversal_valid =
        for_each_merged_event(
            events, event_less,
            [&](const Event &event) {
              const std::size_t axis_index =
                  static_cast<std::size_t>(event_axis(event.key));
              const std::size_t row = event_row(event.key);
              if (axis_index >= result.axes.size() ||
                  row >= result.axes[axis_index].row_count()) {
                invalid_event_key = true;
                return;
              }
              if (event.key == previous_key &&
                  std::abs(event.coordinate - previous_coordinate) <=
                      duplicate_tolerance) {
                if (!checked_add_assign(result.duplicate_events, 1U))
                  invalid_event_key = true;
                return;
              }
              if (row_counts[axis_index][row] ==
                  std::numeric_limits<std::size_t>::max()) {
                invalid_event_key = true;
                return;
              }
              ++row_counts[axis_index][row];
              previous_key = event.key;
              previous_coordinate = event.coordinate;
            },
            &merge_queue_bytes) &&
        merge_traversal_valid;
  }
  std::size_t merge_scratch_bytes = 0U;
  telemetry_valid =
      checked_add(merge_queue_bytes, group_scratch_bytes, merge_scratch_bytes);
  std::size_t first_merge_live_bytes = 0U;
  telemetry_valid =
      telemetry_valid &&
      checked_add(probe_index_bytes, event_slab_bytes,
                  first_merge_live_bytes) &&
      checked_add_assign(first_merge_live_bytes, row_count_bytes) &&
      checked_add_assign(first_merge_live_bytes, merge_scratch_bytes);
  if (telemetry_valid)
    result.peak_temporary_bytes =
        std::max(result.peak_temporary_bytes, first_merge_live_bytes);
  if (invalid_event_key || !merge_traversal_valid || !telemetry_valid) {
    result.status =
        invalid_event_key ? "invalid_event_key" : "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }

  for (std::size_t axis = 0; axis < 3; ++axis) {
    AxisRayCrossings &crossings = result.axes[axis];
    crossings.offsets.resize(crossings.row_count() + 1U, 0U);
    for (std::size_t row = 0; row < crossings.row_count(); ++row) {
      if (!checked_add(crossings.offsets[row], row_counts[axis][row],
                       crossings.offsets[row + 1U])) {
        result.status = "csr_offset_overflow";
        result.provenance.status = result.status;
        return result;
      }
      if ((row_counts[axis][row] & 1U) != 0U)
        ++crossings.odd_rows;
    }
    crossings.coordinates.reserve(crossings.offsets.back());
    if (options.retain_provenance) {
      result.provenance.axes[axis].canonical_refs.reserve(
          crossings.offsets.back());
    }
  }
  const auto csr_capacity_bytes = [&](std::size_t &bytes) {
    bytes = 0U;
    for (const AxisRayCrossings &crossings : result.axes) {
      if (!checked_vector_bytes(crossings.coordinates.capacity(),
                                sizeof(double), bytes) ||
          !checked_vector_bytes(crossings.offsets.capacity(),
                                sizeof(std::size_t), bytes))
        return false;
    }
    return true;
  };
  const auto provenance_capacity_bytes = [&](std::size_t &bytes) {
    bytes = 0U;
    for (const AxisPatchProvenance &axis : result.provenance.axes) {
      std::size_t axis_bytes = 0U;
      if (!axis.retained_payload_bytes(axis_bytes) ||
          !checked_add_assign(bytes, axis_bytes))
        return false;
    }
    return true;
  };
  std::size_t csr_bytes = 0U;
  std::size_t provenance_bytes = 0U;
  std::size_t second_merge_live_bytes = 0U;
  telemetry_valid =
      csr_capacity_bytes(csr_bytes) &&
      provenance_capacity_bytes(provenance_bytes) &&
      checked_add(probe_index_bytes, event_slab_bytes,
                  second_merge_live_bytes) &&
      checked_add_assign(second_merge_live_bytes, row_count_bytes) &&
      checked_add_assign(second_merge_live_bytes, csr_bytes) &&
      checked_add_assign(second_merge_live_bytes, provenance_bytes) &&
      checked_add_assign(second_merge_live_bytes, merge_scratch_bytes);
  if (!telemetry_valid) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  result.peak_temporary_bytes =
      std::max(result.peak_temporary_bytes, second_merge_live_bytes);

  // A second k-way merge writes the exact CSR payload.  Keeping the sorted
  // worker slabs and making two linear passes avoids the former all-events
  // `merged` vector, which doubled peak event memory on large proteins.
  previous_key = std::numeric_limits<std::uint64_t>::max();
  previous_coordinate = 0.0;
  bool provenance_write_failed = false;
  if (options.retain_provenance) {
    const auto feed = [&](auto &&accept) {
      const bool merged = for_each_merged_event(
          provenance_events, provenance_event_less,
          [&](const ProvenanceEvent &event) {
            accept(provenance_ray_key(event.key), event.coordinate,
                   provenance_event_patch(event.key));
          },
          &merge_queue_bytes);
      merge_traversal_valid = merge_traversal_valid && merged;
      return merged;
    };
    const bool grouped = for_each_patch_coordinate_group(
        feed, duplicate_tolerance,
        [&](std::uint64_t ray, double coordinate,
            std::span<const PatchRefId> refs,
            std::span<const std::uint32_t> multiplicities) {
          const std::size_t axis_index = static_cast<std::size_t>(ray >> 30U);
          if (axis_index >= result.axes.size() ||
              !result.provenance.axes[axis_index].append(refs,
                                                         multiplicities)) {
            provenance_write_failed = true;
            return;
          }
          result.axes[axis_index].coordinates.push_back(coordinate);
        },
        &group_scratch_bytes);
    merge_traversal_valid = merge_traversal_valid && grouped;
  } else {
    merge_traversal_valid =
        for_each_merged_event(
            events, event_less,
            [&](const Event &event) {
              if (event.key == previous_key &&
                  std::abs(event.coordinate - previous_coordinate) <=
                      duplicate_tolerance)
                return;
              const std::size_t axis_index =
                  static_cast<std::size_t>(event_axis(event.key));
              result.axes[axis_index].coordinates.push_back(event.coordinate);
              previous_key = event.key;
              previous_coordinate = event.coordinate;
            },
            &merge_queue_bytes) &&
        merge_traversal_valid;
  }
  merge_scratch_bytes = 0U;
  telemetry_valid =
      checked_add(merge_queue_bytes, group_scratch_bytes,
                  merge_scratch_bytes) &&
      csr_capacity_bytes(csr_bytes) &&
      provenance_capacity_bytes(provenance_bytes) &&
      checked_add(probe_index_bytes, event_slab_bytes,
                  second_merge_live_bytes) &&
      checked_add_assign(second_merge_live_bytes, row_count_bytes) &&
      checked_add_assign(second_merge_live_bytes, csr_bytes) &&
      checked_add_assign(second_merge_live_bytes, provenance_bytes) &&
      checked_add_assign(second_merge_live_bytes, merge_scratch_bytes);
  if (telemetry_valid)
    result.peak_temporary_bytes =
        std::max(result.peak_temporary_bytes, second_merge_live_bytes);
  if (provenance_write_failed || !merge_traversal_valid || !telemetry_valid) {
    result.status = provenance_write_failed ? "provenance_csr_overflow"
                                            : "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  for (std::vector<Event> &worker_events : events) {
    std::vector<Event>().swap(worker_events);
  }
  std::vector<std::vector<Event>>().swap(events);
  for (std::vector<ProvenanceEvent> &worker_events : provenance_events) {
    std::vector<ProvenanceEvent>().swap(worker_events);
  }
  std::vector<std::vector<ProvenanceEvent>>().swap(provenance_events);
  for (std::vector<std::size_t> &counts : row_counts)
    std::vector<std::size_t>().swap(counts);

  // Exact seams and tangencies can put a grid line on a degenerate patch
  // boundary. Resolve only those rare odd rows by a deterministic
  // infinitesimal orthogonal perturbation. The regular patch-major path and
  // its compact CSR remain untouched for every even row.
  const double perturbation =
      std::max(1.0e-7 * grid_spacing,
               1024.0 * std::numeric_limits<double>::epsilon() * local_scale);
  const std::array<std::pair<double, double>, 8> perturbations = {
      std::pair{perturbation, 0.0},
      std::pair{-perturbation, 0.0},
      std::pair{0.0, perturbation},
      std::pair{0.0, -perturbation},
      std::pair{perturbation, perturbation},
      std::pair{-perturbation, perturbation},
      std::pair{perturbation, -perturbation},
      std::pair{-perturbation, -perturbation}};
  for (std::size_t axis_index = 0U; axis_index < result.axes.size();
       ++axis_index) {
    AxisRayCrossings &crossings = result.axes[axis_index];
    const bool has_uncertified_row =
        std::find(row_uncertified[axis_index].begin(),
                  row_uncertified[axis_index].end(),
                  std::uint8_t{1}) != row_uncertified[axis_index].end();
    if (crossings.odd_rows == 0U && !has_uncertified_row)
      continue;
    std::vector<std::size_t> repaired_offsets(crossings.row_count() + 1U, 0U);
    std::vector<double> repaired_coordinates;
    repaired_coordinates.reserve(crossings.coordinates.size());
    std::vector<double> candidate;
    AxisPatchProvenance repaired_provenance;
    AxisPatchProvenance candidate_provenance;
    std::vector<ProvenanceEvent> candidate_events;
    std::vector<std::uint32_t> selected_multiplicities;
    if (options.retain_provenance) {
      repaired_provenance.canonical_refs.reserve(crossings.coordinates.size());
    }
    bool provenance_repair_failed = false;
    bool repair_counter_overflow = false;
    std::size_t unresolved_odd_rows = 0U;
    for (std::size_t row = 0; row < crossings.row_count(); ++row) {
      std::span<const double> selected = crossings.row(row);
      const std::size_t original_crossing_begin = crossings.offsets[row];
      bool repaired = false;
      bool selected_candidate_provenance = false;
      const bool was_odd = (selected.size() & 1U) != 0U;
      const bool was_uncertified = row_uncertified[axis_index][row] != 0U;
      if (was_odd || was_uncertified) {
        for (const auto &[delta_u, delta_v] : perturbations) {
          const bool recomputed =
              options.retain_provenance
                  ? recompute_perturbed_row_with_provenance(
                        crossings.axis, row, delta_u, delta_v, candidate,
                        candidate_provenance, candidate_events)
                  : recompute_perturbed_row(crossings.axis, row, delta_u,
                                            delta_v, candidate);
          if (recomputed && (candidate.size() & 1U) == 0U) {
            selected = candidate;
            repaired = true;
            selected_candidate_provenance = options.retain_provenance;
            if (!checked_add_assign(result.degenerate_rows_repaired, 1U))
              repair_counter_overflow = true;
            if (was_uncertified)
              if (!checked_add_assign(result.uncertified_rows_repaired, 1U))
                repair_counter_overflow = true;
            break;
          }
        }
        if (!repaired && was_odd)
          if (!checked_add_assign(unresolved_odd_rows, 1U))
            repair_counter_overflow = true;
        if (!repaired && was_uncertified)
          if (!checked_add_assign(result.unresolved_uncertified_rows, 1U))
            repair_counter_overflow = true;
        std::size_t repair_live_bytes = probe_index_bytes;
        std::size_t repaired_provenance_bytes = 0U;
        std::size_t candidate_provenance_bytes = 0U;
        telemetry_valid =
            csr_capacity_bytes(csr_bytes) &&
            provenance_capacity_bytes(provenance_bytes) &&
            repaired_provenance.retained_payload_bytes(
                repaired_provenance_bytes) &&
            candidate_provenance.retained_payload_bytes(
                candidate_provenance_bytes) &&
            checked_add_assign(repair_live_bytes, row_uncertified_bytes) &&
            checked_add_assign(repair_live_bytes, csr_bytes) &&
            checked_add_assign(repair_live_bytes, provenance_bytes) &&
            checked_vector_bytes(repaired_offsets.capacity(),
                                 sizeof(std::size_t), repair_live_bytes) &&
            checked_vector_bytes(repaired_coordinates.capacity(),
                                 sizeof(double), repair_live_bytes) &&
            checked_vector_bytes(candidate.capacity(), sizeof(double),
                                 repair_live_bytes) &&
            checked_add_assign(repair_live_bytes, repaired_provenance_bytes) &&
            checked_add_assign(repair_live_bytes, candidate_provenance_bytes) &&
            checked_vector_bytes(candidate_events.capacity(),
                                 sizeof(ProvenanceEvent), repair_live_bytes) &&
            checked_vector_bytes(selected_multiplicities.capacity(),
                                 sizeof(std::uint32_t), repair_live_bytes) &&
            checked_add_assign(repair_live_bytes, repair_group_scratch_bytes);
        if (telemetry_valid)
          result.peak_temporary_bytes =
              std::max(result.peak_temporary_bytes, repair_live_bytes);
      }
      repaired_coordinates.insert(repaired_coordinates.end(), selected.begin(),
                                  selected.end());
      if (options.retain_provenance) {
        for (std::size_t crossing = 0; crossing < selected.size(); ++crossing) {
          const AxisPatchProvenance &source_provenance =
              selected_candidate_provenance
                  ? candidate_provenance
                  : result.provenance.axes[axis_index];
          const std::size_t source_crossing =
              selected_candidate_provenance
                  ? crossing
                  : original_crossing_begin + crossing;
          const std::span<const PatchRefId> refs =
              source_provenance.refs(source_crossing);
          selected_multiplicities.clear();
          selected_multiplicities.reserve(refs.size());
          for (std::size_t ordinal = 0; ordinal < refs.size(); ++ordinal) {
            selected_multiplicities.push_back(
                source_provenance.ref_multiplicity(source_crossing, ordinal));
          }
          if (refs.empty() ||
              !repaired_provenance.append(refs, selected_multiplicities)) {
            provenance_repair_failed = true;
            break;
          }
        }
      }
      repaired_offsets[row + 1U] = repaired_coordinates.size();
    }
    if (repair_counter_overflow || !telemetry_valid ||
        provenance_repair_failed ||
        (options.retain_provenance &&
         !repaired_provenance.validate(repaired_coordinates.size()))) {
      result.status =
          repair_counter_overflow
              ? "counter_overflow"
              : (!telemetry_valid ? "byte_telemetry_overflow"
                                  : "provenance_repair_csr_invalid");
      result.provenance.status = result.status;
      return result;
    }
    crossings.coordinates = std::move(repaired_coordinates);
    crossings.offsets = std::move(repaired_offsets);
    if (options.retain_provenance)
      result.provenance.axes[axis_index] = std::move(repaired_provenance);
    crossings.odd_rows = unresolved_odd_rows;
  }
  for (const AxisRayCrossings &crossings : result.axes) {
    telemetry_valid =
        telemetry_valid &&
        checked_vector_bytes(crossings.coordinates.capacity(), sizeof(double),
                             result.retained_bytes) &&
        checked_vector_bytes(crossings.offsets.capacity(), sizeof(std::size_t),
                             result.retained_bytes);
  }
  if (!telemetry_valid) {
    result.status = "byte_telemetry_overflow";
    result.provenance.status = result.status;
    return result;
  }
  std::size_t total_odd_rows = 0U;
  for (const AxisRayCrossings &crossings : result.axes) {
    if (!checked_add_assign(total_odd_rows, crossings.odd_rows)) {
      result.status = "counter_overflow";
      result.provenance.status = result.status;
      return result;
    }
  }
  // Preserve the production patch-ray contract: unresolved odd rows are
  // consumed by the existing multi-axis cell logic and have never changed
  // PatchRayGrid::status.  Provenance is stricter and may not call such a grid
  // certified, but audit-only mode must not alter default geometry/status.
  result.status =
      result.unresolved_uncertified_rows == 0U ? "pass" : "uncertified_quartic";
  if (options.retain_provenance) {
    bool valid_provenance = true;
    for (std::size_t axis_index = 0; axis_index < result.axes.size();
         ++axis_index) {
      const AxisPatchProvenance &axis = result.provenance.axes[axis_index];
      const bool axis_valid =
          axis.validate(result.axes[axis_index].coordinates.size());
      if (!axis_valid) {
        valid_provenance = false;
        continue;
      }
      valid_provenance = valid_provenance &&
                         checked_add_assign(result.provenance.unique_crossings,
                                            axis.canonical_refs.size()) &&
                         checked_add_assign(result.provenance.seam_groups,
                                            axis.seam_groups.size());
      if (axis.seam_groups.size() > axis.canonical_refs.size())
        valid_provenance = false;
      std::size_t axis_ref_count =
          axis.seam_groups.size() <= axis.canonical_refs.size()
              ? axis.canonical_refs.size() - axis.seam_groups.size()
              : 0U;
      valid_provenance =
          valid_provenance &&
          checked_add_assign(axis_ref_count, axis.seam_refs.size()) &&
          checked_add_assign(result.provenance.refs_total, axis_ref_count);
      result.provenance.max_refs_per_crossing =
          std::max<std::size_t>(result.provenance.max_refs_per_crossing,
                                axis.canonical_refs.empty() ? 0U : 1U);
      for (const SparsePatchSeamGroup &group : axis.seam_groups) {
        result.provenance.max_refs_per_crossing = std::max<std::size_t>(
            result.provenance.max_refs_per_crossing,
            static_cast<std::size_t>(group.refs_end - group.refs_begin));
      }
      std::size_t axis_roots = axis.canonical_refs.size();
      valid_provenance =
          valid_provenance &&
          checked_add_assign(
              result.provenance.same_primitive_near_pair_unclassified,
              axis.single_ref_multiplicities.size()) &&
          checked_add_assign(result.provenance.repeated_root_groups,
                             axis.single_ref_multiplicities.size());
      for (const SparseRootMultiplicity &group :
           axis.single_ref_multiplicities) {
        valid_provenance =
            valid_provenance &&
            checked_add_assign(
                axis_roots, static_cast<std::size_t>(group.multiplicity - 1U));
      }
      for (const SparsePatchSeamGroup &group : axis.seam_groups) {
        std::size_t group_roots = 0U;
        bool group_repeated = false;
        for (std::size_t position = group.refs_begin; position < group.refs_end;
             ++position) {
          const std::uint32_t multiplicity =
              axis.seam_ref_multiplicities[position];
          valid_provenance =
              valid_provenance && checked_add_assign(group_roots, multiplicity);
          group_repeated = group_repeated || multiplicity > 1U;
        }
        valid_provenance = valid_provenance && group_roots != 0U &&
                           checked_add_assign(axis_roots, group_roots - 1U);
        if (group_repeated)
          valid_provenance =
              valid_provenance &&
              checked_add_assign(result.provenance.repeated_root_groups, 1U) &&
              checked_add_assign(
                  result.provenance.same_primitive_near_pair_unclassified, 1U);
      }
      valid_provenance =
          valid_provenance &&
          checked_add_assign(result.provenance.roots_total, axis_roots);
      std::size_t axis_bytes = 0U;
      valid_provenance =
          valid_provenance && axis.retained_payload_bytes(axis_bytes) &&
          checked_add_assign(result.provenance.retained_bytes, axis_bytes);
    }
    std::size_t expected_crossings = 0U;
    valid_provenance =
        valid_provenance &&
        checked_add(result.axes[0].coordinates.size(),
                    result.axes[1].coordinates.size(), expected_crossings) &&
        checked_add_assign(expected_crossings,
                           result.axes[2].coordinates.size());
    if (!valid_provenance ||
        result.provenance.unique_crossings != expected_crossings) {
      result.status = "provenance_csr_invalid";
      result.provenance.status = result.status;
      return result;
    }
    if (!checked_add_assign(result.retained_bytes,
                            result.provenance.retained_bytes) ||
        !csr_capacity_bytes(csr_bytes) ||
        !provenance_capacity_bytes(provenance_bytes) ||
        !checked_add(probe_index_bytes, csr_bytes, second_merge_live_bytes) ||
        !checked_add_assign(second_merge_live_bytes, provenance_bytes)) {
      result.status = "byte_telemetry_overflow";
      result.provenance.status = result.status;
      return result;
    }
    finalize_provenance_certificate(result, total_odd_rows, true);
    result.peak_temporary_bytes =
        std::max(result.peak_temporary_bytes, second_merge_live_bytes);
    result.provenance.peak_temporary_bytes = result.peak_temporary_bytes;
  }
  result.build_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
  if (options.retain_provenance)
    result.provenance.build_ms = result.build_ms;
  return result;
}

#ifdef TITAN_TABI_SES_TEST_HOOKS
void patch_ray_increment_root_multiplicity_for_test(
    std::uint32_t &multiplicity) noexcept {
  increment_root_multiplicity_or_invalidate(multiplicity);
}

void patch_ray_finalize_provenance_for_test(PatchRayGrid &grid) {
  std::size_t total_odd_rows = 0U;
  bool csr_valid = grid.provenance.enabled;
  for (std::size_t axis = 0U; axis < grid.axes.size(); ++axis) {
    csr_valid =
        csr_valid &&
        checked_add_assign(total_odd_rows, grid.axes[axis].odd_rows) &&
        grid.provenance.axes[axis].validate(grid.axes[axis].coordinates.size());
  }
  finalize_provenance_certificate(grid, total_odd_rows, csr_valid);
}
#endif

PatchRayGrid build_patch_ray_grid(const PatchComplex &complex,
                                  const Vec3 &lower, double grid_spacing,
                                  std::size_t nx, std::size_t ny,
                                  std::size_t nz, TitanExecutor *executor,
                                  PatchRayBuildOptions options) {
  const auto failure = [&](const char *status) {
    PatchRayGrid result;
    result.axes[0].axis = RayAxis::x;
    result.axes[0].u_count = ny;
    result.axes[0].v_count = nz;
    result.axes[1].axis = RayAxis::y;
    result.axes[1].u_count = nx;
    result.axes[1].v_count = nz;
    result.axes[2].axis = RayAxis::z;
    result.axes[2].u_count = nx;
    result.axes[2].v_count = ny;
    result.provenance.enabled = options.retain_provenance;
    result.provenance.certified = false;
    result.provenance.status = options.retain_provenance ? status : "disabled";
    result.status = status;
    return result;
  };
  try {
#ifdef TITAN_TABI_SES_TEST_HOOKS
    if (options.simulate_allocation_failure)
      throw std::bad_alloc();
#endif
    return build_patch_ray_grid_impl(complex, lower, grid_spacing, nx, ny, nz,
                                     executor, options);
  } catch (const std::bad_alloc &) {
    return failure("alloc_fail");
  } catch (const std::length_error &) {
    return failure("length_err");
  }
}

} // namespace TitanPBE::Tabi::analytic_ses

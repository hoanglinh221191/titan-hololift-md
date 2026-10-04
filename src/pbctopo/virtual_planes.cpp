#include "virtual_planes.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan_pbctopo {
namespace {

constexpr double kPlaneEps = 1.0e-9;
constexpr double kCenterQuantization = 1.0e9;

double axis_coord(const PbctopoAtom &atom, char axis) {
  if (axis == 'x')
    return atom.sx;
  if (axis == 'y')
    return atom.sy;
  return atom.sz;
}

double clamp_center(double center) {
  if (!std::isfinite(center))
    return 1.5;
  if (center < 0.5)
    return 0.5;
  if (center > 2.5)
    return 2.5;
  return center;
}

void push_unique_plane(std::vector<double> &planes, double value) {
  if (value + kPlaneEps < 0.5 || value - kPlaneEps > 2.5)
    return;
  value = clamp_center(value);
  for (double existing : planes) {
    if (std::fabs(existing - value) <= kPlaneEps)
      return;
  }
  planes.push_back(value);
}

std::size_t clamp_bin(long long value, std::size_t bins) {
  if (bins == 0 || value < 0)
    return 0;
  const long long max_bin = static_cast<long long>(bins) - 1;
  if (value > max_bin)
    return static_cast<std::size_t>(max_bin);
  return static_cast<std::size_t>(value);
}

std::size_t bin_for_center(double center, std::size_t bins) {
  if (bins <= 1)
    return 0;
  const double scaled =
      (clamp_center(center) - 0.5) / 2.0 * static_cast<double>(bins);
  return clamp_bin(static_cast<long long>(std::floor(scaled)), bins);
}

double bin_midpoint(std::size_t begin, std::size_t end, std::size_t bins) {
  if (bins == 0)
    return 1.5;
  const double step = 2.0 / static_cast<double>(bins);
  return clamp_center(0.5 + (static_cast<double>(begin + end) + 1.0) *
                                0.5 * step);
}

std::size_t interval_index(const std::vector<double> &planes, double center) {
  if (planes.size() < 2)
    return 0;
  const double clamped = clamp_center(center);
  auto it = std::upper_bound(planes.begin(), planes.end(), clamped + kPlaneEps);
  if (it == planes.begin())
    return 0;
  const std::size_t idx =
      static_cast<std::size_t>(std::distance(planes.begin(), it) - 1);
  return std::min(idx, planes.size() - 2);
}

std::uint64_t hash_interval_key(std::size_t ix, std::size_t iy,
                                std::size_t iz) {
  const auto mix = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  std::uint64_t hash = mix(static_cast<std::uint64_t>(ix));
  hash ^= mix(static_cast<std::uint64_t>(iy) + hash);
  hash ^= mix(static_cast<std::uint64_t>(iz) + hash);
  return hash;
}

void sort_unique_axis_coords(std::vector<double> &coords) {
  std::sort(coords.begin(), coords.end());
  std::vector<double> unique_coords;
  unique_coords.reserve(coords.size());
  for (double value : coords) {
    if (!std::isfinite(value))
      continue;
    if (!unique_coords.empty() &&
        std::fabs(unique_coords.back() - value) <= kPlaneEps) {
      continue;
    }
    unique_coords.push_back(value);
  }
  coords.swap(unique_coords);
}

} // namespace

std::vector<double>
build_dense_virtual_box_event_planes(const std::vector<PbctopoAtom> &atoms,
                                     char axis) {
  std::vector<double> planes;
  planes.reserve(atoms.size() * 2 + 2);
  planes.push_back(0.5);
  planes.push_back(2.5);
  for (const auto &atom : atoms) {
    const double coord = axis_coord(atom, axis);
    planes.push_back(coord + 0.5);
    planes.push_back(coord + 1.5);
  }

  std::sort(planes.begin(), planes.end());
  std::vector<double> unique_planes;
  unique_planes.reserve(planes.size());
  for (double value : planes) {
    if (value + kPlaneEps < 0.5 || value - kPlaneEps > 2.5)
      continue;
    if (!unique_planes.empty() &&
        std::fabs(unique_planes.back() - value) <= kPlaneEps) {
      continue;
    }
    unique_planes.push_back(value);
  }

  if (unique_planes.size() < 2) {
    unique_planes = {0.5, 2.5};
  } else {
    if (std::fabs(unique_planes.front() - 0.5) > kPlaneEps)
      unique_planes.insert(unique_planes.begin(), 0.5);
    if (std::fabs(unique_planes.back() - 2.5) > kPlaneEps)
      unique_planes.push_back(2.5);
  }
  return unique_planes;
}

std::vector<double>
build_density_virtual_box_event_planes(const std::vector<PbctopoAtom> &atoms,
                                       char axis, std::size_t histogram_bins,
                                       std::size_t max_planes) {
  if (max_planes < 3 || histogram_bins < 4)
    return {};

  std::vector<double> planes;
  planes.reserve(max_planes);
  push_unique_plane(planes, 0.5);
  push_unique_plane(planes, 1.5);
  push_unique_plane(planes, 2.5);

  std::vector<std::size_t> density(histogram_bins, 0);
  for (const auto &atom : atoms) {
    const double coord = axis_coord(atom, axis);
    const double lower_event = coord + 0.5;
    const double upper_event = coord + 1.5;
    if (lower_event + kPlaneEps >= 0.5 && lower_event - kPlaneEps <= 2.5)
      ++density[bin_for_center(lower_event, histogram_bins)];
    if (upper_event + kPlaneEps >= 0.5 && upper_event - kPlaneEps <= 2.5)
      ++density[bin_for_center(upper_event, histogram_bins)];
  }

  struct VoidRun {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t length = 0;
    double center = 1.5;
  };
  std::vector<VoidRun> void_runs;
  for (std::size_t idx = 0; idx < histogram_bins;) {
    if (density[idx] != 0) {
      ++idx;
      continue;
    }
    const std::size_t begin = idx;
    while (idx < histogram_bins && density[idx] == 0)
      ++idx;
    const std::size_t end = idx - 1;
    void_runs.push_back(
        {begin, end, end - begin + 1,
         bin_midpoint(begin, end, histogram_bins)});
  }
  std::sort(void_runs.begin(), void_runs.end(),
            [](const VoidRun &lhs, const VoidRun &rhs) {
              if (lhs.length != rhs.length)
                return lhs.length > rhs.length;
              return lhs.center < rhs.center;
            });
  for (const VoidRun &run : void_runs) {
    if (planes.size() >= max_planes)
      break;
    push_unique_plane(planes, run.center);
  }

  struct MinBin {
    std::size_t idx = 0;
    std::size_t count = 0;
    double center = 1.5;
  };
  std::vector<MinBin> local_minima;
  for (std::size_t idx = 0; idx < histogram_bins; ++idx) {
    const std::size_t prev =
        (idx == 0) ? std::numeric_limits<std::size_t>::max() : density[idx - 1];
    const std::size_t next = (idx + 1 == histogram_bins)
                                 ? std::numeric_limits<std::size_t>::max()
                                 : density[idx + 1];
    if (density[idx] <= prev && density[idx] <= next) {
      local_minima.push_back(
          {idx, density[idx], bin_midpoint(idx, idx, histogram_bins)});
    }
  }
  std::sort(local_minima.begin(), local_minima.end(),
            [](const MinBin &lhs, const MinBin &rhs) {
              if (lhs.count != rhs.count)
                return lhs.count < rhs.count;
              return lhs.center < rhs.center;
            });
  for (const MinBin &minimum : local_minima) {
    if (planes.size() >= max_planes)
      break;
    push_unique_plane(planes, minimum.center);
  }

  std::sort(planes.begin(), planes.end());
  return planes;
}

PbctopoVirtualPlaneSet build_density_cut_plane_shortlist_or_dense_fallback(
    const std::vector<PbctopoAtom> &atoms,
    PbctopoVirtualPlaneBuildOptions options) {
  PbctopoVirtualPlaneSet result;
  result.dense_equivalent_x = atoms.size() * 2 + 2;
  result.dense_equivalent_y = atoms.size() * 2 + 2;
  result.dense_equivalent_z = atoms.size() * 2 + 2;

  if (options.max_sparse_planes_per_axis < 3 || options.histogram_bins < 4) {
    result.x = build_dense_virtual_box_event_planes(atoms, 'x');
    result.y = build_dense_virtual_box_event_planes(atoms, 'y');
    result.z = build_dense_virtual_box_event_planes(atoms, 'z');
    result.dense_equivalent_x = result.x.size();
    result.dense_equivalent_y = result.y.size();
    result.dense_equivalent_z = result.z.size();
    result.used_dense_fallback = true;
    return result;
  }

  result.x = build_density_virtual_box_event_planes(
      atoms, 'x', options.histogram_bins, options.max_sparse_planes_per_axis);
  result.y = build_density_virtual_box_event_planes(
      atoms, 'y', options.histogram_bins, options.max_sparse_planes_per_axis);
  result.z = build_density_virtual_box_event_planes(
      atoms, 'z', options.histogram_bins, options.max_sparse_planes_per_axis);

  if (result.x.size() < 2 || result.y.size() < 2 || result.z.size() < 2) {
    result.x = build_dense_virtual_box_event_planes(atoms, 'x');
    result.y = build_dense_virtual_box_event_planes(atoms, 'y');
    result.z = build_dense_virtual_box_event_planes(atoms, 'z');
    result.dense_equivalent_x = result.x.size();
    result.dense_equivalent_y = result.y.size();
    result.dense_equivalent_z = result.z.size();
    result.used_dense_fallback = true;
  }
  return result;
}

void PbctopoVirtualStateTracker::reserve(std::size_t expected_states) {
  buckets_.reserve(expected_states);
}

void PbctopoVirtualStateTracker::reset(const std::vector<double> &x_planes,
                                       const std::vector<double> &y_planes,
                                       const std::vector<double> &z_planes) {
  x_planes_ = x_planes;
  y_planes_ = y_planes;
  z_planes_ = z_planes;
  clear();
}

void PbctopoVirtualStateTracker::clear() {
  buckets_.clear();
  collisions_ = 0;
  inserted_ = 0;
}

PbctopoVirtualStateTracker::CenterKey
PbctopoVirtualStateTracker::quantize_center(double cut_x, double cut_y,
                                            double cut_z) {
  return {static_cast<std::int64_t>(
              std::llround(clamp_center(cut_x) * kCenterQuantization)),
          static_cast<std::int64_t>(
              std::llround(clamp_center(cut_y) * kCenterQuantization)),
          static_cast<std::int64_t>(
              std::llround(clamp_center(cut_z) * kCenterQuantization))};
}

std::uint64_t PbctopoVirtualStateTracker::sparse_state_key(double cut_x,
                                                           double cut_y,
                                                           double cut_z) const {
  return hash_interval_key(interval_index(x_planes_, cut_x),
                           interval_index(y_planes_, cut_y),
                           interval_index(z_planes_, cut_z));
}

bool PbctopoVirtualStateTracker::try_insert(double cut_x, double cut_y,
                                            double cut_z) {
  const std::uint64_t state_key = sparse_state_key(cut_x, cut_y, cut_z);
  CenterKey center_key = quantize_center(cut_x, cut_y, cut_z);
  std::vector<CenterKey> &bucket = buckets_[state_key];
  for (const CenterKey &existing : bucket) {
    if (existing == center_key)
      return false;
  }
  if (!bucket.empty())
    ++collisions_;
  bucket.push_back(center_key);
  ++inserted_;
  return true;
}

bool PbctopoVirtualStateTracker::try_insert(double cut_x, double cut_y,
                                            double cut_z,
                                            std::uint64_t shift_signature) {
  const std::uint64_t state_key = sparse_state_key(cut_x, cut_y, cut_z);
  CenterKey center_key = quantize_center(cut_x, cut_y, cut_z);
  center_key.shift_signature = shift_signature;
  center_key.has_shift_signature = true;
  std::vector<CenterKey> &bucket = buckets_[state_key];
  for (const CenterKey &existing : bucket) {
    if (existing == center_key)
      return false;
    if (existing.has_shift_signature &&
        existing.shift_signature == shift_signature) {
      return false;
    }
  }
  if (!bucket.empty())
    ++collisions_;
  bucket.push_back(center_key);
  ++inserted_;
  return true;
}

void PbctopoVirtualAssignmentSignatureTable::clear() {
  x_coords_.clear();
  y_coords_.clear();
  z_coords_.clear();
}

void PbctopoVirtualAssignmentSignatureTable::append_axis_coords(
    const std::vector<PbctopoAtom> &atoms, char axis,
    std::vector<double> &coords) {
  coords.clear();
  coords.reserve(atoms.size());
  for (const auto &atom : atoms) {
    const double coord = axis_coord(atom, axis);
    if (coord + kPlaneEps >= 0.0 && coord - kPlaneEps <= 1.0)
      coords.push_back(coord);
  }
  sort_unique_axis_coords(coords);
}

void PbctopoVirtualAssignmentSignatureTable::build(
    const std::vector<PbctopoAtom> &atoms) {
  append_axis_coords(atoms, 'x', x_coords_);
  append_axis_coords(atoms, 'y', y_coords_);
  append_axis_coords(atoms, 'z', z_coords_);
}

std::uint64_t PbctopoVirtualAssignmentSignatureTable::axis_interval_signature(
    const std::vector<double> &unique_coords, double center) {
  const double clamped = clamp_center(center);
  const auto upper_count = [&](double threshold) {
    return static_cast<std::uint64_t>(std::distance(
        unique_coords.begin(),
        std::upper_bound(unique_coords.begin(), unique_coords.end(),
                         threshold + kPlaneEps)));
  };
  const std::uint64_t lower_events = upper_count(clamped - 0.5);
  const std::uint64_t upper_events = upper_count(clamped - 1.5);
  // Dense legacy state key used the interval index in the unique plane list
  // {0.5, coord+0.5, coord+1.5, 2.5}. The leading 0.5 plane cancels with
  // the "-1" interval conversion, so the two coordinate counts are enough.
  return lower_events + upper_events;
}

std::uint64_t PbctopoVirtualAssignmentSignatureTable::state_signature(
    double cut_x, double cut_y, double cut_z) const {
  return hash_interval_key(axis_interval_signature(x_coords_, cut_x),
                           axis_interval_signature(y_coords_, cut_y),
                           axis_interval_signature(z_coords_, cut_z));
}

} // namespace titan_pbctopo

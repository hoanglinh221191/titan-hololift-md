#include "component_graph.h"

#include "candidate_order.h"
#include "component_graph_math.h"
#include "component_metric.h"
#include "component_shift_domain.h"
#include "identifiability.h"
#include "lattice_math.h"
#include "required_contacts.h"
#include "steric_clash.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>

using namespace std;

namespace titan_pbctopo {

namespace {

struct PbctopoContactEpochScratch {
  vector<uint32_t> epochs;
  uint32_t epoch = 0;
  size_t marked = 0;

  void begin(size_t count, PbctopoTimingStats *timing) {
    if (epochs.size() < count) {
      epochs.resize(count, 0u);
      if (timing != nullptr)
        ++timing->component_graph_contact_epoch_scratch_resizes;
    }
    ++epoch;
    if (epoch == 0u) {
      fill(epochs.begin(), epochs.end(), 0u);
      epoch = 1u;
      if (timing != nullptr)
        ++timing->component_graph_contact_epoch_wrap_clears;
    }
    marked = 0;
  }

  void mark(size_t index) {
    if (index >= epochs.size() || epochs[index] == epoch)
      return;
    epochs[index] = epoch;
    ++marked;
  }
};

size_t find_shift_index(const vector<PbctopoShift> &shifts, const Int3 &delta) {
  for (size_t idx = 0; idx < shifts.size(); ++idx) {
    if (shifts[idx].ix == delta.x && shifts[idx].iy == delta.y &&
        shifts[idx].iz == delta.z) {
      return idx;
    }
  }
  return numeric_limits<size_t>::max();
}

bool append_shift_index_if_present(const vector<PbctopoShift> &shifts,
                                   const Int3 &delta, bool allow_identity,
                                   vector<size_t> &ordered_indices,
                                   vector<unsigned char> &seen_indices) {
  if (!allow_identity && is_zero_delta(delta))
    return false;
  const size_t shift_idx = find_shift_index(shifts, delta);
  if (shift_idx == numeric_limits<size_t>::max() ||
      shift_idx >= seen_indices.size() || seen_indices[shift_idx] != 0) {
    return false;
  }
  seen_indices[shift_idx] = 1;
  ordered_indices.push_back(shift_idx);
  return true;
}

} // namespace

vector<size_t> build_pair_shift_order(const vector<PbctopoShift> &shifts,
                                      const Int3 *expected_delta,
                                      PbctopoTimingStats *timing) {
  vector<size_t> ordered_indices;
  ordered_indices.reserve(shifts.size());
  vector<unsigned char> seen_indices(shifts.size(), 0);

  if (expected_delta != nullptr) {
    if (append_shift_index_if_present(shifts, *expected_delta, true,
                                      ordered_indices, seen_indices) &&
        timing != nullptr) {
      ++timing->component_pair_warm_shift_hits;
    }
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0)
            continue;
          Int3 neighbor{};
          if (!checked_int3_add(*expected_delta, Int3{dx, dy, dz},
                                neighbor)) {
            continue;
          }
          if (append_shift_index_if_present(shifts, neighbor, true,
                                            ordered_indices, seen_indices) &&
              timing != nullptr) {
            ++timing->component_pair_warm_shift_hits;
          }
        }
      }
    }
  }

  for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
    if (seen_indices[shift_idx] != 0)
      continue;
    seen_indices[shift_idx] = 1;
    ordered_indices.push_back(shift_idx);
  }
  return ordered_indices;
}
bool component_bbox_may_clash(const PbctopoShiftedComponent &lhs,
                              const PbctopoShiftedComponent &rhs,
                              double cutoff) {
  const auto gap = [](double min_a, double max_a, double min_b, double max_b) {
    if (max_a < min_b)
      return min_b - max_a;
    if (max_b < min_a)
      return min_a - max_b;
    return 0.0;
  };
  const double dx = gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  if (dx > cutoff)
    return false;
  const double dy = gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  if (dy > cutoff)
    return false;
  const double dz = gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dz <= cutoff;
}

PbctopoComponentBaseGeometry
build_component_base_geometry(const vector<PbctopoAtom> &atoms,
                               const vector<size_t> &component) {
  PbctopoComponentBaseGeometry base;
  base.x.reserve(component.size());
  base.y.reserve(component.size());
  base.z.reserve(component.size());
  base.atom_indices.reserve(component.size());
  auto append_atom = [&](size_t atom_idx) {
    const auto &atom = atoms[atom_idx];
    base.x.push_back(atom.x);
    base.y.push_back(atom.y);
    base.z.push_back(atom.z);
    base.atom_indices.push_back(atom_idx);
    base.center = base.center + Vec3{atom.x, atom.y, atom.z};
    base.min_x = min(base.min_x, atom.x);
    base.min_y = min(base.min_y, atom.y);
    base.min_z = min(base.min_z, atom.z);
    base.max_x = max(base.max_x, atom.x);
    base.max_y = max(base.max_y, atom.y);
    base.max_z = max(base.max_z, atom.z);
  };
  for (size_t atom_idx : component) {
    if (atom_idx < atoms.size() && atoms[atom_idx].is_heavy)
      append_atom(atom_idx);
  }
  if (base.x.empty()) {
    for (size_t atom_idx : component) {
      if (atom_idx < atoms.size())
        append_atom(atom_idx);
    }
  }
  if (!base.x.empty()) {
    base.center = base.center * (1.0 / static_cast<double>(base.x.size()));
  }
  return base;
}

PbctopoShiftedComponentDescriptor
describe_shifted_component(const PbctopoComponentBaseGeometry &base,
                           const PbctopoShift &shift) {
  PbctopoShiftedComponentDescriptor descriptor;
  descriptor.translation_norm2 = pbctopo_translation_norm2(shift);
  descriptor.center = {base.center.x + shift.tx, base.center.y + shift.ty,
                       base.center.z + shift.tz};
  descriptor.min_x = base.min_x + shift.tx;
  descriptor.min_y = base.min_y + shift.ty;
  descriptor.min_z = base.min_z + shift.tz;
  descriptor.max_x = base.max_x + shift.tx;
  descriptor.max_y = base.max_y + shift.ty;
  descriptor.max_z = base.max_z + shift.tz;
  return descriptor;
}

void fill_shifted_component_from_base(
    const PbctopoComponentBaseGeometry &base, const PbctopoShift &shift,
    double inv_grid, double inv_contact_grid, bool build_clash_grid,
    PbctopoShiftedComponent &shifted) {
  const PbctopoShiftedComponentDescriptor descriptor =
      describe_shifted_component(base, shift);
  const size_t atom_count = base.x.size();
  shifted.points.resize(atom_count);
  shifted.atom_indices = base.atom_indices;
  shifted.clash_keys.resize(atom_count);
  shifted.contact_keys.resize(atom_count);
  shifted.clash_grid.clear();
  if (build_clash_grid)
    shifted.clash_grid.reserve(max<size_t>(32, atom_count));
  shifted.contact_grid.clear();
  shifted.contact_grid.reserve(max<size_t>(32, atom_count));
  shifted.translation_norm2 = descriptor.translation_norm2;
  shifted.spatial_keys_valid = true;
  shifted.center = descriptor.center;
  shifted.min_x = descriptor.min_x;
  shifted.min_y = descriptor.min_y;
  shifted.min_z = descriptor.min_z;
  shifted.max_x = descriptor.max_x;
  shifted.max_y = descriptor.max_y;
  shifted.max_z = descriptor.max_z;
  for (size_t local_idx = 0; local_idx < atom_count; ++local_idx) {
    const Vec3 point{base.x[local_idx] + shift.tx,
                     base.y[local_idx] + shift.ty,
                     base.z[local_idx] + shift.tz};
    shifted.points[local_idx] = point;
    if (!pbctopo_make_cell_key(point.x, point.y, point.z, inv_grid,
                                shifted.clash_keys[local_idx])) {
      shifted.spatial_keys_valid = false;
      shifted.clash_grid.clear();
      shifted.contact_grid.clear();
      return;
    }
    if (build_clash_grid)
      shifted.clash_grid[shifted.clash_keys[local_idx]].push_back(local_idx);
    ClashCellKey contact_key;
    if (!pbctopo_make_cell_key(point.x, point.y, point.z, inv_contact_grid,
                                contact_key)) {
      shifted.spatial_keys_valid = false;
      shifted.clash_grid.clear();
      shifted.contact_grid.clear();
      return;
    }
    shifted.contact_keys[local_idx] = contact_key;
    shifted.contact_grid[contact_key].push_back(local_idx);
  }
}

class PbctopoLazyShiftedComponents {
public:
  PbctopoLazyShiftedComponents(const vector<PbctopoAtom> &atoms,
                               const vector<vector<size_t>> &components,
                               const vector<PbctopoShift> &shifts,
                               double clash_cutoff, double contact_cutoff,
                               PbctopoTimingStats *timing,
                               bool build_clash_grid = false)
      : shifts_(shifts), inv_grid_(1.0 / clash_cutoff),
        inv_contact_grid_(1.0 / contact_cutoff),
        build_clash_grid_(build_clash_grid), timing_(timing) {
    bases_.reserve(components.size());
    for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
      bases_.push_back(build_component_base_geometry(atoms, components[comp_idx]));
    }
    materialized_.reserve(
        components.size() * min<size_t>(shifts_.size(), 8));
    if (timing_ != nullptr) {
      timing_->component_graph_shift_descriptor_domain +=
          components.size() * shifts_.size();
      timing_->component_graph_shift_cache_dense_entries_equivalent +=
          components.size() * shifts_.size();
    }
  }

  PbctopoShiftedComponentDescriptor descriptor(size_t comp_idx,
                                               size_t shift_idx) const {
    if (timing_ != nullptr)
      ++timing_->component_graph_shift_descriptors;
    return describe_shifted_component(bases_[comp_idx], shifts_[shift_idx]);
  }

  size_t component_count() const noexcept { return bases_.size(); }

  const PbctopoComponentBaseGeometry &base_geometry(size_t comp_idx) const {
    return bases_[comp_idx];
  }

  const PbctopoShiftedComponent &materialize(size_t comp_idx, size_t shift_idx) {
    const size_t key = comp_idx * shifts_.size() + shift_idx;
    auto [it, inserted] = materialized_.try_emplace(key);
    if (inserted) {
      const auto started = chrono::steady_clock::now();
      fill_shifted_component_from_base(
          bases_[comp_idx], shifts_[shift_idx], inv_grid_, inv_contact_grid_,
          build_clash_grid_, it->second);
      if (timing_ != nullptr) {
        ++timing_->component_graph_shift_materialized;
        timing_->component_graph_shift_cache_entries_peak =
            max(timing_->component_graph_shift_cache_entries_peak,
                materialized_.size());
        timing_->component_graph_lazy_materialize_s +=
            chrono::duration<double>(chrono::steady_clock::now() - started)
                .count();
      }
    }
    return it->second;
  }

private:
  const vector<PbctopoShift> &shifts_;
  vector<PbctopoComponentBaseGeometry> bases_;
  unordered_map<size_t, PbctopoShiftedComponent> materialized_;
  double inv_grid_ = 1.0;
  double inv_contact_grid_ = 1.0;
  bool build_clash_grid_ = false;
  PbctopoTimingStats *timing_ = nullptr;
};

vector<vector<PbctopoShiftedComponent>>
precompute_shifted_components(const vector<PbctopoAtom> &atoms,
                              const vector<vector<size_t>> &components,
                              const vector<PbctopoShift> &shifts,
                              double clash_cutoff, double contact_cutoff,
                              bool build_clash_grid) {
  vector<vector<PbctopoShiftedComponent>> precomputed(components.size());
  const double inv_grid = 1.0 / clash_cutoff;
  const double inv_contact_grid = 1.0 / contact_cutoff;
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    const PbctopoComponentBaseGeometry base =
        build_component_base_geometry(atoms, components[comp_idx]);
    auto &shifted_list = precomputed[comp_idx];
    shifted_list.resize(shifts.size());
    for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
      fill_shifted_component_from_base(base, shifts[shift_idx], inv_grid,
                                       inv_contact_grid, build_clash_grid,
                                       shifted_list[shift_idx]);
    }
  }
  return precomputed;
}

double shifted_component_bbox_distance2(const PbctopoShiftedComponent &lhs,
                                        const PbctopoShiftedComponent &rhs) {
  const auto gap = [](double min_a, double max_a, double min_b, double max_b) {
    if (max_a < min_b)
      return min_b - max_a;
    if (max_b < min_a)
      return min_a - max_b;
    return 0.0;
  };
  const double dx = gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  const double dy = gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  const double dz = gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dx * dx + dy * dy + dz * dz;
}

double shifted_component_bbox_distance2(
    const PbctopoShiftedComponentDescriptor &lhs,
    const PbctopoShiftedComponentDescriptor &rhs) {
  const auto gap = [](double min_a, double max_a, double min_b, double max_b) {
    if (max_a < min_b)
      return min_b - max_a;
    if (max_b < min_a)
      return min_a - max_b;
    return 0.0;
  };
  const double dx = gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  const double dy = gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  const double dz = gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dx * dx + dy * dy + dz * dz;
}

namespace {

struct PbctopoComponentPairSpatialShortlist {
  size_t component_count = 0;
  vector<vector<size_t>> contact_shift_indices;
  bool valid = false;

  const vector<size_t> &for_pair(size_t lhs, size_t rhs) const {
    return contact_shift_indices[lhs * component_count + rhs];
  }
};

PbctopoComponentPairSpatialShortlist build_component_pair_spatial_shortlist(
    const PbctopoLazyShiftedComponents &shifted_components,
    const vector<PbctopoShift> &shifts, size_t identity_shift_pos,
    double contact_cutoff, PbctopoTimingStats *timing) {
  PbctopoComponentPairSpatialShortlist result;
  const auto started = chrono::steady_clock::now();
  const size_t component_count = shifted_components.component_count();
  result.component_count = component_count;
  const auto finish = [&](bool valid) {
    result.valid = valid;
    if (timing != nullptr) {
      ++timing->component_graph_spatial_shortlist_builds;
      if (!valid)
        ++timing->component_graph_spatial_shortlist_fallbacks;
      timing->component_graph_spatial_shortlist_s +=
          chrono::duration<double>(chrono::steady_clock::now() - started)
              .count();
    }
    return result;
  };
  if (component_count == 0 || shifts.empty() ||
      identity_shift_pos >= shifts.size() || !isfinite(contact_cutoff) ||
      contact_cutoff < 0.0 ||
      component_count > numeric_limits<size_t>::max() / component_count) {
    return finish(false);
  }
  result.contact_shift_indices.resize(component_count * component_count);

  double max_component_width = 0.0;
  vector<Vec3> bbox_centers(component_count);
  vector<PbctopoShiftedComponentDescriptor> identity_descriptors;
  identity_descriptors.reserve(component_count);
  for (size_t comp_idx = 0; comp_idx < component_count; ++comp_idx) {
    const auto &base = shifted_components.base_geometry(comp_idx);
    if (base.x.empty() || !isfinite(base.min_x) || !isfinite(base.min_y) ||
        !isfinite(base.min_z) || !isfinite(base.max_x) ||
        !isfinite(base.max_y) || !isfinite(base.max_z)) {
      return finish(false);
    }
    const double width_x = base.max_x - base.min_x;
    const double width_y = base.max_y - base.min_y;
    const double width_z = base.max_z - base.min_z;
    if (width_x < 0.0 || width_y < 0.0 || width_z < 0.0)
      return finish(false);
    max_component_width =
        max({max_component_width, width_x, width_y, width_z});
    bbox_centers[comp_idx] = {0.5 * (base.min_x + base.max_x),
                              0.5 * (base.min_y + base.max_y),
                              0.5 * (base.min_z + base.max_z)};
    identity_descriptors.push_back(
        shifted_components.descriptor(comp_idx, identity_shift_pos));
  }
  const double raw_cell_size = max_component_width + contact_cutoff;
  if (!isfinite(raw_cell_size) || raw_cell_size <= 0.0)
    return finish(false);
  // Bbox midpoints of any contact-capable pair differ by at most this value
  // on each axis, so the 27 neighboring cells form a complete shortlist.
  const double cell_size =
      nextafter(raw_cell_size, numeric_limits<double>::infinity());
  const double inv_cell_size = 1.0 / cell_size;
  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> base_grid;
  base_grid.reserve(max<size_t>(32, component_count * 2));
  for (size_t comp_idx = 0; comp_idx < component_count; ++comp_idx) {
    ClashCellKey key;
    const auto &center = bbox_centers[comp_idx];
    if (!pbctopo_make_cell_key(center.x, center.y, center.z, inv_cell_size,
                                key)) {
      return finish(false);
    }
    base_grid[key].push_back(comp_idx);
  }

  const double contact_cutoff2 = contact_cutoff * contact_cutoff;
  for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
    const auto &shift = shifts[shift_idx];
    if (!isfinite(shift.tx) || !isfinite(shift.ty) || !isfinite(shift.tz))
      return finish(false);
    for (size_t lhs = 0; lhs < component_count; ++lhs) {
      const Vec3 query_center{bbox_centers[lhs].x - shift.tx,
                              bbox_centers[lhs].y - shift.ty,
                              bbox_centers[lhs].z - shift.tz};
      ClashCellKey query_key;
      if (!pbctopo_make_cell_key(query_center.x, query_center.y,
                                  query_center.z, inv_cell_size, query_key)) {
        return finish(false);
      }
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            ClashCellKey neighbor_key;
            if (!pbctopo_offset_cell_key(query_key, dx, dy, dz,
                                          neighbor_key)) {
              return finish(false);
            }
            const auto cell_it = base_grid.find(neighbor_key);
            if (cell_it == base_grid.end())
              continue;
            for (size_t rhs : cell_it->second) {
              if (lhs >= rhs)
                continue;
              const double bbox_d2 = shifted_component_bbox_distance2(
                  identity_descriptors[lhs],
                  shifted_components.descriptor(rhs, shift_idx));
              if (bbox_d2 <= contact_cutoff2) {
                result.contact_shift_indices[lhs * component_count + rhs]
                    .push_back(shift_idx);
              }
            }
          }
        }
      }
    }
  }
  return finish(true);
}

} // namespace

PbctopoPairShiftScore make_bbox_only_pair_shift_score(const PbctopoShift &shift,
                                                       double bbox_d2) {
  PbctopoPairShiftScore score;
  score.shift_index = shift.shift_index;
  score.ix = shift.ix;
  score.iy = shift.iy;
  score.iz = shift.iz;
  score.shift_norm2 = pbctopo_translation_norm2(shift);
  score.bbox_d2 = bbox_d2;
  score.min_pair_d2 = bbox_d2;
  score.valid = true;
  score.complete = true;
  return score;
}

bool better_pair_shift_score(const PbctopoPairShiftScore &lhs,
                             const PbctopoPairShiftScore &rhs) {
  if (!lhs.valid || !lhs.complete)
    return false;
  if (!rhs.valid || !rhs.complete)
    return true;
  if (lhs.ultra_clash != rhs.ultra_clash)
    return rhs.ultra_clash;
  int metric_cmp = compare_pbctopo_metric(lhs.contact_fraction,
                                           rhs.contact_fraction,
                                           {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp > 0;
  metric_cmp = compare_pbctopo_metric(lhs.soft_penalty_mean,
                                       rhs.soft_penalty_mean,
                                           {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_metric(lhs.shift_norm2, rhs.shift_norm2,
                                       {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.ix != rhs.ix)
    return lhs.ix < rhs.ix;
  if (lhs.iy != rhs.iy)
    return lhs.iy < rhs.iy;
  return lhs.iz < rhs.iz;
}

void insert_pair_shift_candidate(vector<PbctopoPairShiftCandidate> &top,
                                 const PbctopoPairShiftCandidate &candidate,
                                 size_t keep) {
  if (!candidate.score.valid || !candidate.score.complete)
    return;
  auto pos = top.begin();
  while (pos != top.end() &&
         better_pair_shift_score(pos->score, candidate.score)) {
    ++pos;
  }
  top.insert(pos, candidate);
  if (top.size() > keep)
    top.resize(keep);
}

bool better_component_graph_edge(const PbctopoGraphEdge &lhs,
                                 const PbctopoGraphEdge &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.ultra_clash != rhs.ultra_clash)
    return rhs.ultra_clash;
  int metric_cmp = compare_pbctopo_metric(lhs.contact_fraction,
                                           rhs.contact_fraction,
                                           {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp > 0;
  metric_cmp = compare_pbctopo_metric(lhs.confidence, rhs.confidence,
                                           {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp > 0;
  metric_cmp = compare_pbctopo_metric(lhs.soft_penalty_mean,
                                       rhs.soft_penalty_mean,
                                       {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_metric(lhs.shift_norm2, rhs.shift_norm2,
                                       {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.a != rhs.a)
    return lhs.a < rhs.a;
  return lhs.b < rhs.b;
}

static PbctopoPairShiftScore score_component_pair_shift(
    const PbctopoShiftedComponent &lhs, const PbctopoShiftedComponent &rhs,
    const PbctopoShift &shift, double hard_cutoff, double contact_cutoff,
    const vector<PbctopoAtom> &atoms,
    PbctopoTimingStats *timing) {
  PbctopoPairShiftScore score;
  score.shift_index = shift.shift_index;
  score.ix = shift.ix;
  score.iy = shift.iy;
  score.iz = shift.iz;
  score.shift_norm2 = pbctopo_translation_norm2(shift);
  if (!lhs.spatial_keys_valid || !rhs.spatial_keys_valid) {
    score.complete = true;
    return score;
  }
  score.bbox_d2 = shifted_component_bbox_distance2(lhs, rhs);
  score.min_pair_d2 = score.bbox_d2;
  score.valid = true;

  const double contact_cutoff2 = contact_cutoff * contact_cutoff;
  if (score.bbox_d2 > contact_cutoff2)
    return score;

  thread_local PbctopoContactEpochScratch rhs_contact_scratch;
  rhs_contact_scratch.begin(rhs.points.size(), timing);

  for (size_t lhs_idx = 0; lhs_idx < lhs.points.size(); ++lhs_idx) {
    const auto &lhs_point = lhs.points[lhs_idx];
    const ClashCellKey &contact_key = lhs.contact_keys[lhs_idx];
    bool atom_has_contact = false;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            ClashCellKey neighbor;
            if (!pbctopo_offset_cell_key(contact_key, dx, dy, dz, neighbor)) {
              score.valid = false;
              score.complete = true;
              return score;
            }
            const auto it = rhs.contact_grid.find(neighbor);
            if (it == rhs.contact_grid.end())
              continue;
            for (size_t rhs_idx : it->second) {
              const auto &rhs_point = rhs.points[rhs_idx];
              const double ddx = lhs_point.x - rhs_point.x;
              const double ddy = lhs_point.y - rhs_point.y;
              const double ddz = lhs_point.z - rhs_point.z;
              const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 < score.min_pair_d2)
                score.min_pair_d2 = d2;
              const size_t lhs_atom_idx = lhs.atom_indices[lhs_idx];
              const size_t rhs_atom_idx = rhs.atom_indices[rhs_idx];
              if (lhs_atom_idx < atoms.size() && rhs_atom_idx < atoms.size() &&
                  pbctopo_atoms_clash(d2, atoms[lhs_atom_idx],
                                       atoms[rhs_atom_idx], hard_cutoff,
                                       timing)) {
                score.ultra_clash = true;
                score.quality_score = -numeric_limits<double>::infinity();
                score.complete = true;
                return score;
              }
              if (d2 <= contact_cutoff2)
                ++score.contact_pairs;
              if (lhs_atom_idx < atoms.size() &&
                  rhs_atom_idx < atoms.size()) {
                score.soft_penalty += pbctopo_pair_soft_overlap_penalty(
                    d2, atoms[lhs_atom_idx], atoms[rhs_atom_idx], hard_cutoff);
              }
              if (d2 <= contact_cutoff2) {
                atom_has_contact = true;
                rhs_contact_scratch.mark(rhs_idx);
              }
            }
          }
        }
      }
    if (atom_has_contact)
      ++score.lhs_contact_atoms;
  }

  score.rhs_contact_atoms = rhs_contact_scratch.marked;

  finalize_normalized_pair_shift_score(score, lhs.points.size(),
                                       rhs.points.size(), contact_cutoff);
  return score;
}

struct ComponentGraphDsu {
  vector<size_t> parent;
  vector<size_t> rank;

  explicit ComponentGraphDsu(size_t n) : parent(n), rank(n, 0) {
    iota(parent.begin(), parent.end(), 0);
  }

  size_t find(size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool unite(size_t a, size_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (rank[a] < rank[b])
      swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b])
      ++rank[a];
    return true;
  }
};
Int3 edge_delta(const PbctopoGraphEdge &edge) {
  return {edge.dx, edge.dy, edge.dz};
}

bool same_pair(const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
  return lhs.a == rhs.a && lhs.b == rhs.b;
}

bool same_edge_option(const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
  return same_pair(lhs, rhs) && lhs.dx == rhs.dx && lhs.dy == rhs.dy &&
         lhs.dz == rhs.dz;
}

bool build_component_tree_from_ranked_edges(
    size_t component_count, const vector<PbctopoGraphEdge> &ranked_edges,
    const vector<PbctopoGraphEdge> &forced_edges, vector<PbctopoGraphEdge> &tree_edges) {
  tree_edges.clear();
  ComponentGraphDsu dsu(component_count);
  auto add_edge = [&](const PbctopoGraphEdge &edge) {
    if (!edge.valid || edge.ultra_clash)
      return false;
    if (!dsu.unite(edge.a, edge.b))
      return false;
    tree_edges.push_back(edge);
    return true;
  };

  for (const auto &edge : forced_edges) {
    if (!add_edge(edge))
      return false;
  }

  for (const auto &edge : ranked_edges) {
    bool pair_forced = false;
    bool exact_forced = false;
    for (const auto &forced : forced_edges) {
      if (same_pair(edge, forced))
        pair_forced = true;
      if (same_edge_option(edge, forced)) {
        exact_forced = true;
        break;
      }
    }
    if (pair_forced && !exact_forced)
      continue;
    add_edge(edge);
    if (tree_edges.size() >= component_count - 1)
      break;
  }

  return tree_edges.size() + 1 == component_count;
}

bool solve_component_offsets_from_tree(
    size_t component_count, const vector<PbctopoGraphEdge> &tree_edges,
    vector<Int3> &offsets) {
  offsets.assign(component_count, {numeric_limits<int>::max(),
                                   numeric_limits<int>::max(),
                                   numeric_limits<int>::max()});
  vector<vector<pair<size_t, Int3>>> adjacency(component_count);
  for (const auto &edge : tree_edges) {
    const Int3 delta = edge_delta(edge);
    Int3 reverse_delta{};
    if (!checked_int3_negate(delta, reverse_delta))
      return false;
    adjacency[edge.a].push_back({edge.b, delta});
    adjacency[edge.b].push_back({edge.a, reverse_delta});
  }

  vector<char> seen(component_count, 0);
  queue<size_t> q;
  offsets[0] = {0, 0, 0};
  seen[0] = 1;
  q.push(0);
  while (!q.empty()) {
    const size_t current = q.front();
    q.pop();
    for (const auto &edge : adjacency[current]) {
      const size_t next = edge.first;
      if (seen[next] != 0)
        continue;
      if (!checked_int3_add(offsets[current], edge.second, offsets[next]))
        return false;
      seen[next] = 1;
      q.push(next);
    }
  }
  return find(seen.begin(), seen.end(), 0) == seen.end();
}

bool compare_int3_lexicographic(const Int3 &lhs, const Int3 &rhs) {
  if (lhs.x != rhs.x)
    return lhs.x < rhs.x;
  if (lhs.y != rhs.y)
    return lhs.y < rhs.y;
  return lhs.z < rhs.z;
}

bool same_int3(const Int3 &lhs, const Int3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool same_int3_offsets(const vector<Int3> &lhs, const vector<Int3> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    if (!same_int3(lhs[idx], rhs[idx]))
      return false;
  }
  return true;
}

bool oriented_pair_delta(const PbctopoPairShiftCandidateSet &set,
                         const PbctopoPairShiftCandidate &candidate,
                         size_t from, size_t to, Int3 &delta) {
  if (from == set.a && to == set.b) {
    delta = candidate.delta;
    return true;
  }
  if (from == set.b && to == set.a)
    return checked_int3_negate(candidate.delta, delta);
  return false;
}

struct PbctopoCyclePenaltyContext {
  box Box{};
  double inverse_basis2 = 1.0;
};

PbctopoCyclePenaltyContext make_cycle_penalty_context(box Box) {
  PbctopoCyclePenaltyContext context;
  context.Box = Box;
  const double shortest_basis =
      PbctopoLatticeMetric(Box).shortest_basis_length();
  if (isfinite(shortest_basis) && shortest_basis > 1.0e-9)
    context.inverse_basis2 = 1.0 / (shortest_basis * shortest_basis);
  return context;
}

double robust_cycle_penalty(Int3 residual,
                            const PbctopoCyclePenaltyContext &context) {
  const double residual2 = pbctopo_translation_norm2(context.Box, residual);
  if (!isfinite(residual2))
    return numeric_limits<double>::infinity();
  return min(residual2 * context.inverse_basis2, 4.0);
}

PbctopoCycleSyncMetrics evaluate_pair_graph_metrics_for_offsets(
    const vector<Int3> &offsets,
    const vector<PbctopoPairShiftCandidateSet> &pair_candidates, box Box) {
  PbctopoCycleSyncMetrics metrics;
  const PbctopoCyclePenaltyContext penalty_context =
      make_cycle_penalty_context(Box);
  double total_cost = 0.0;
  double total_support = 0.0;
  for (const auto &set : pair_candidates) {
    if (set.a >= offsets.size() || set.b >= offsets.size() ||
        set.candidates.empty()) {
      continue;
    }
    Int3 observed{};
    if (!checked_int3_subtract(offsets[set.b], offsets[set.a], observed)) {
      ++metrics.inconsistent_edge_count;
      continue;
    }
    double best_penalty = numeric_limits<double>::infinity();
    double best_weight = 0.0;
    for (const auto &candidate : set.candidates) {
      const double weight = max(1.0, candidate.weight);
      Int3 residual{};
      if (!checked_int3_subtract(observed, candidate.delta, residual))
        continue;
      const double penalty =
          robust_cycle_penalty(residual, penalty_context) * weight;
      if (penalty < best_penalty) {
        best_penalty = penalty;
        best_weight = weight;
      }
    }
    if (!isfinite(best_penalty))
      continue;
    total_cost += best_penalty;
    total_support += best_weight;
    if (best_penalty > 1.0e-9)
      ++metrics.inconsistent_edge_count;
  }
  metrics.cycle_support =
      static_cast<size_t>(llround(max(0.0, total_support)));
  metrics.cycle_mismatch =
      total_support > 0.0 ? total_cost / total_support : total_cost;
  const PbctopoCycleSyncMetrics fundamental =
      compute_fundamental_cycle_residuals(offsets.size(), pair_candidates, Box);
  metrics.fundamental_cycle_count = fundamental.fundamental_cycle_count;
  metrics.fundamental_cycle_residual_rms =
      fundamental.fundamental_cycle_residual_rms;
  metrics.fundamental_cycle_residual_max =
      fundamental.fundamental_cycle_residual_max;
  return metrics;
}

struct PbctopoCycleSyncBeamState {
  vector<Int3> offsets;
  vector<char> assigned;
  double cost = 0.0;
  double support = 0.0;
  size_t inconsistent_edges = 0;
};

PbctopoCycleSyncResult solve_component_offsets_cycle_consistent(
    size_t component_count, box Box,
    const vector<PbctopoPairShiftCandidateSet> &pair_candidates,
    size_t beam_width) {
  PbctopoCycleSyncResult result;
  if (component_count == 0)
    return result;
  if (component_count == 1) {
    result.valid = true;
    result.offsets.assign(1, {0, 0, 0});
    return result;
  }
  const PbctopoCyclePenaltyContext penalty_context =
      make_cycle_penalty_context(Box);

  vector<vector<size_t>> incident(component_count);
  for (size_t edge_idx = 0; edge_idx < pair_candidates.size(); ++edge_idx) {
    const auto &set = pair_candidates[edge_idx];
    if (set.a < component_count && set.b < component_count &&
        !set.candidates.empty()) {
      incident[set.a].push_back(edge_idx);
      incident[set.b].push_back(edge_idx);
    }
  }

  vector<size_t> order;
  order.reserve(component_count);
  vector<char> in_order(component_count, 0);
  order.push_back(0);
  in_order[0] = 1;
  while (order.size() < component_count) {
    size_t best = numeric_limits<size_t>::max();
    size_t best_links = 0;
    for (size_t comp = 0; comp < component_count; ++comp) {
      if (in_order[comp])
        continue;
      size_t links = 0;
      for (size_t edge_idx : incident[comp]) {
        const auto &set = pair_candidates[edge_idx];
        const size_t other = set.a == comp ? set.b : set.a;
        if (other < in_order.size() && in_order[other])
          ++links;
      }
      if (links > best_links ||
          (links == best_links && best == numeric_limits<size_t>::max())) {
        best = comp;
        best_links = links;
      }
    }
    if (best == numeric_limits<size_t>::max() || best_links == 0)
      return result;
    in_order[best] = 1;
    order.push_back(best);
  }

  vector<PbctopoCycleSyncBeamState> beam(1);
  beam[0].offsets.assign(component_count, {0, 0, 0});
  beam[0].assigned.assign(component_count, 0);
  beam[0].assigned[0] = 1;

  const size_t keep = max<size_t>(1, min<size_t>(beam_width, 256));
  for (size_t order_pos = 1; order_pos < order.size(); ++order_pos) {
    const size_t comp = order[order_pos];
    vector<PbctopoCycleSyncBeamState> next;
    for (const auto &state : beam) {
      vector<Int3> proposed_offsets;
      for (size_t edge_idx : incident[comp]) {
        const auto &set = pair_candidates[edge_idx];
        const size_t other = set.a == comp ? set.b : set.a;
        if (other >= state.assigned.size() || !state.assigned[other])
          continue;
        for (const auto &candidate : set.candidates) {
          Int3 delta_other_to_comp{};
          Int3 proposed{};
          if (!oriented_pair_delta(set, candidate, other, comp,
                                   delta_other_to_comp) ||
              !checked_int3_add(state.offsets[other], delta_other_to_comp,
                                proposed)) {
            continue;
          }
          proposed_offsets.push_back(proposed);
        }
      }
      sort(proposed_offsets.begin(), proposed_offsets.end(),
           compare_int3_lexicographic);
      proposed_offsets.erase(unique(proposed_offsets.begin(),
                                    proposed_offsets.end(), same_int3),
                             proposed_offsets.end());
      for (const Int3 &offset : proposed_offsets) {
        PbctopoCycleSyncBeamState trial = state;
        trial.offsets[comp] = offset;
        trial.assigned[comp] = 1;
        for (size_t edge_idx : incident[comp]) {
          const auto &set = pair_candidates[edge_idx];
          const size_t other = set.a == comp ? set.b : set.a;
          if (other >= trial.assigned.size() || !trial.assigned[other])
            continue;
          double best_edge_penalty = numeric_limits<double>::infinity();
          double best_edge_weight = 0.0;
          for (const auto &candidate : set.candidates) {
            Int3 expected{};
            Int3 observed{};
            Int3 residual{};
            if (!oriented_pair_delta(set, candidate, other, comp, expected) ||
                !checked_int3_subtract(trial.offsets[comp],
                                       trial.offsets[other], observed) ||
                !checked_int3_subtract(observed, expected, residual)) {
              continue;
            }
            const double penalty =
                robust_cycle_penalty(residual, penalty_context) *
                max(1.0, candidate.weight);
            if (penalty < best_edge_penalty) {
              best_edge_penalty = penalty;
              best_edge_weight = max(1.0, candidate.weight);
            }
          }
          if (isfinite(best_edge_penalty)) {
            trial.cost += best_edge_penalty;
            trial.support += best_edge_weight;
            if (best_edge_penalty > 1.0e-9)
              ++trial.inconsistent_edges;
          }
        }
        next.push_back(std::move(trial));
      }
    }
    sort(next.begin(), next.end(), [](const auto &lhs, const auto &rhs) {
      int metric_cmp = compare_pbctopo_metric(lhs.cost, rhs.cost,
                                               {1.0e-9, 0.0});
      if (metric_cmp != 0)
        return metric_cmp < 0;
      metric_cmp = compare_pbctopo_metric(lhs.support, rhs.support,
                                           {1.0e-9, 0.0});
      if (metric_cmp != 0)
        return metric_cmp > 0;
      return lhs.inconsistent_edges < rhs.inconsistent_edges;
    });
    if (next.size() > keep)
      next.resize(keep);
    beam = std::move(next);
    if (beam.empty())
      return result;
  }

  const auto &best = beam.front();
  result.valid = true;
  result.offsets = best.offsets;
  if (!rebase_component_offsets(result.offsets, Box))
    return {};
  result.metrics.cycle_support =
      static_cast<size_t>(llround(max(0.0, best.support)));
  result.metrics.cycle_mismatch =
      best.support > 0.0 ? best.cost / best.support : best.cost;
  result.metrics.inconsistent_edge_count = best.inconsistent_edges;
  const PbctopoCycleSyncMetrics fundamental =
      compute_fundamental_cycle_residuals(component_count, pair_candidates,
                                          Box);
  result.metrics.fundamental_cycle_count = fundamental.fundamental_cycle_count;
  result.metrics.fundamental_cycle_residual_rms =
      fundamental.fundamental_cycle_residual_rms;
  result.metrics.fundamental_cycle_residual_max =
      fundamental.fundamental_cycle_residual_max;
  return result;
}

bool graph_edge_has_zero_shift(const PbctopoGraphEdge &edge) {
  return edge.dx == 0 && edge.dy == 0 && edge.dz == 0;
}

bool graph_edge_matches_delta(const PbctopoGraphEdge &edge, const Int3 &delta) {
  return edge.dx == delta.x && edge.dy == delta.y && edge.dz == delta.z;
}

vector<PbctopoGraphEdge> build_component_graph_edge_candidates(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &components, const vector<PbctopoShift> &shifts,
    const RuntimeCutoffs &cutoffs, PbctopoTimingStats *timing,
    const vector<Int3> *warm_component_offsets,
    vector<PbctopoPairShiftCandidateSet> *pair_candidates_out,
    const vector<PbctopoRequiredContact> *required_contacts) {
  bool relative_domain_analytic = false;
  const auto relative_domain = build_component_relative_shift_domain(
      shifts, &relative_domain_analytic);
  if (timing != nullptr) {
    if (relative_domain_analytic)
      ++timing->component_relative_domain_analytic_builds;
    else
      ++timing->component_relative_domain_generic_builds;
  }
  vector<PbctopoShift> pair_shifts;
  pair_shifts.reserve(relative_domain.size());
  for (const auto &relative : relative_domain)
    pair_shifts.push_back(relative.shift);
  PbctopoRequiredContactConstraintContext required_contact_context;
  const bool has_required_contacts =
      required_contacts != nullptr && !required_contacts->empty();
  if (has_required_contacts) {
    required_contact_context =
        build_pbctopo_required_contact_constraint_context(
            atoms, components, Box, *required_contacts);
    if (timing != nullptr) {
      ++timing->component_required_contact_constraint_builds;
      timing->component_required_contact_constraint_pairs +=
          required_contact_context.groups.size();
    }
    if (!required_contact_context.valid ||
        required_contact_context.infeasible) {
      if (timing != nullptr)
        ++timing->component_required_contact_infeasible;
      return {};
    }
  }
  size_t identity_shift_pos = 0;
  for (size_t shift_idx = 0; shift_idx < pair_shifts.size(); ++shift_idx) {
    if (pair_shifts[shift_idx].ix == 0 && pair_shifts[shift_idx].iy == 0 &&
        pair_shifts[shift_idx].iz == 0) {
      identity_shift_pos = shift_idx;
      break;
    }
  }

  const auto precompute_started = chrono::steady_clock::now();
  PbctopoLazyShiftedComponents shifted_components(
      atoms, components, pair_shifts, cutoffs.clash_cutoff,
      cutoffs.pair_contact_cutoff, timing);
  if (timing != nullptr) {
    timing->component_graph_input_shift_domain_max =
        max(timing->component_graph_input_shift_domain_max, shifts.size());
    timing->component_graph_relative_shift_domain_max =
        max(timing->component_graph_relative_shift_domain_max,
            pair_shifts.size());
    timing->component_graph_precompute_s +=
        chrono::duration<double>(chrono::steady_clock::now() - precompute_started)
            .count();
  }

  vector<PbctopoGraphEdge> ranked_edges;
  ranked_edges.reserve((components.size() * (components.size() - 1)) / 2);
  const auto pair_score_started = chrono::steady_clock::now();
  const double lazy_materialize_before =
      timing != nullptr ? timing->component_graph_lazy_materialize_s : 0.0;
  const double contact_cutoff2 =
      cutoffs.pair_contact_cutoff * cutoffs.pair_contact_cutoff;
  vector<size_t> bbox_only_ranked_shift_indices(pair_shifts.size());
  iota(bbox_only_ranked_shift_indices.begin(),
       bbox_only_ranked_shift_indices.end(), 0);
  stable_sort(bbox_only_ranked_shift_indices.begin(),
              bbox_only_ranked_shift_indices.end(),
              [&](size_t lhs_idx, size_t rhs_idx) {
                return better_pair_shift_score(
                    make_bbox_only_pair_shift_score(
                        pair_shifts[lhs_idx],
                        numeric_limits<double>::infinity()),
                    make_bbox_only_pair_shift_score(
                        pair_shifts[rhs_idx],
                        numeric_limits<double>::infinity()));
              });
  const PbctopoComponentPairSpatialShortlist spatial_shortlist =
      build_component_pair_spatial_shortlist(
          shifted_components, pair_shifts, identity_shift_pos,
          cutoffs.pair_contact_cutoff, timing);
  vector<uint32_t> included_shift_epochs(pair_shifts.size(), 0u);
  uint32_t included_shift_epoch = 0u;
  for (size_t lhs = 0; lhs < components.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < components.size(); ++rhs) {
      PbctopoPairShiftScore best_score;
      PbctopoPairShiftScore second_score;
      vector<PbctopoPairShiftCandidate> top_pair_candidates;
      if (pair_candidates_out != nullptr)
        top_pair_candidates.reserve(3);
      Int3 expected_delta;
      const Int3 *expected_delta_ptr = nullptr;
      if (warm_component_offsets != nullptr &&
          warm_component_offsets->size() == components.size()) {
        if (checked_int3_subtract((*warm_component_offsets)[rhs],
                                  (*warm_component_offsets)[lhs],
                                  expected_delta)) {
          expected_delta_ptr = &expected_delta;
        }
      }
      const bool has_pair_contact_constraints =
          has_required_contacts &&
          find_pbctopo_required_contact_constraint_group(
              required_contact_context, lhs, rhs) != nullptr;
      const PbctopoShiftedComponentDescriptor lhs_descriptor =
          shifted_components.descriptor(lhs, identity_shift_pos);
      const auto accept_score = [&](size_t shift_idx,
                                    const PbctopoPairShiftScore &score) {
        const auto &shift = pair_shifts[shift_idx];
        if (pair_candidates_out != nullptr) {
          insert_pair_shift_candidate(
              top_pair_candidates,
              PbctopoPairShiftCandidate{{shift.ix, shift.iy, shift.iz}, score,
                                        0.0},
              3);
        }
        if (better_pair_shift_score(score, best_score)) {
          second_score = best_score;
          best_score = score;
        } else if (better_pair_shift_score(score, second_score)) {
          second_score = score;
        }
      };
      const bool use_spatial_shortlist =
          spatial_shortlist.valid && !has_pair_contact_constraints;
      if (use_spatial_shortlist) {
        ++included_shift_epoch;
        if (included_shift_epoch == 0u) {
          fill(included_shift_epochs.begin(), included_shift_epochs.end(), 0u);
          included_shift_epoch = 1u;
        }
        vector<size_t> score_shift_indices =
            spatial_shortlist.for_pair(lhs, rhs);
        for (size_t shift_idx : score_shift_indices)
          included_shift_epochs[shift_idx] = included_shift_epoch;
        const size_t bbox_only_keep =
            pair_candidates_out != nullptr ? size_t{3} : size_t{2};
        size_t bbox_only_added = 0;
        for (size_t shift_idx : bbox_only_ranked_shift_indices) {
          if (included_shift_epochs[shift_idx] == included_shift_epoch)
            continue;
          score_shift_indices.push_back(shift_idx);
          included_shift_epochs[shift_idx] = included_shift_epoch;
          if (++bbox_only_added >= bbox_only_keep)
            break;
        }
        for (size_t shift_idx : score_shift_indices) {
          const auto &shift = pair_shifts[shift_idx];
          const double bbox_d2 = shifted_component_bbox_distance2(
              lhs_descriptor, shifted_components.descriptor(rhs, shift_idx));
          PbctopoPairShiftScore score;
          if (bbox_d2 > contact_cutoff2) {
            score = make_bbox_only_pair_shift_score(shift, bbox_d2);
            if (timing != nullptr)
              ++timing->component_graph_bbox_only_scores;
          } else {
            score = score_component_pair_shift(
                shifted_components.materialize(lhs, identity_shift_pos),
                shifted_components.materialize(rhs, shift_idx), shift,
                cutoffs.clash_cutoff, cutoffs.pair_contact_cutoff, atoms,
                timing);
          }
          accept_score(shift_idx, score);
        }
        if (timing != nullptr) {
          ++timing->component_graph_spatial_shortlist_pairs;
          timing->component_graph_spatial_shortlist_shift_scores_avoided +=
              pair_shifts.size() - score_shift_indices.size();
        }
      } else {
        const vector<size_t> ordered_shift_indices =
            build_pair_shift_order(pair_shifts, expected_delta_ptr, timing);
        for (size_t shift_idx : ordered_shift_indices) {
          const auto &shift = pair_shifts[shift_idx];
          const Int3 relative_delta{shift.ix, shift.iy, shift.iz};
          if (has_pair_contact_constraints) {
            if (timing != nullptr)
              ++timing->component_required_contact_relative_deltas_checked;
            if (!pbctopo_required_contact_relative_shift_allowed(
                    required_contact_context, atoms, lhs, rhs,
                    relative_delta)) {
              if (timing != nullptr)
                ++timing->component_required_contact_relative_deltas_rejected;
              continue;
            }
          }
          const double bbox_d2 = shifted_component_bbox_distance2(
              lhs_descriptor,
              shifted_components.descriptor(rhs, shift_idx));
          PbctopoPairShiftScore score;
          if (bbox_d2 > contact_cutoff2) {
            score = make_bbox_only_pair_shift_score(shift, bbox_d2);
            if (timing != nullptr)
              ++timing->component_graph_bbox_only_scores;
          } else {
            score = score_component_pair_shift(
                shifted_components.materialize(lhs, identity_shift_pos),
                shifted_components.materialize(rhs, shift_idx), shift,
                cutoffs.clash_cutoff, cutoffs.pair_contact_cutoff, atoms,
                timing);
          }
          accept_score(shift_idx, score);
        }
      }
      PbctopoGraphEdge edge;
      edge.a = lhs;
      edge.b = rhs;
      edge.dx = best_score.ix;
      edge.dy = best_score.iy;
      edge.dz = best_score.iz;
      edge.contact_atoms = best_score.contact_atoms;
      edge.contact_pairs = best_score.contact_pairs;
      edge.contact_fraction = best_score.contact_fraction;
      edge.soft_penalty = best_score.soft_penalty;
      edge.soft_penalty_mean = best_score.soft_penalty_mean;
      edge.min_pair_d2 = best_score.min_pair_d2;
      edge.bbox_d2 = best_score.bbox_d2;
      edge.shift_norm2 = best_score.shift_norm2;
      edge.confidence =
          pbctopo_pair_shift_confidence(best_score, second_score);
      edge.ultra_clash = best_score.ultra_clash;
      edge.valid = best_score.valid && best_score.complete;
      ranked_edges.push_back(edge);
      if (pair_candidates_out != nullptr && !top_pair_candidates.empty()) {
        for (size_t candidate_idx = 0;
             candidate_idx < top_pair_candidates.size(); ++candidate_idx) {
          const double confidence =
              candidate_idx == 0 ? edge.confidence : 0.0;
          top_pair_candidates[candidate_idx].weight =
              pbctopo_pair_shift_candidate_weight(
                  top_pair_candidates[candidate_idx].score, confidence);
        }
        pair_candidates_out->push_back({lhs, rhs, top_pair_candidates});
        if (timing != nullptr)
          timing->component_cycle_sync_pair_candidates +=
              top_pair_candidates.size();
      }
    }
  }
  if (timing != nullptr) {
    const double lazy_materialize_delta =
        timing->component_graph_lazy_materialize_s - lazy_materialize_before;
    timing->component_graph_pair_score_s +=
        max(0.0, chrono::duration<double>(chrono::steady_clock::now() -
                                          pair_score_started)
                     .count() -
                     lazy_materialize_delta);
    timing->component_graph_edges += ranked_edges.size();
  }
  return ranked_edges;
}

bool solve_component_offsets_via_pair_graph(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &components, const vector<PbctopoShift> &shifts,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set, vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, PbctopoCutCandidate &best_cut,
    double &shift_norm2, size_t component_beam_width,
    PbctopoTimingStats *timing,
    const PbctopoRefineScoringCallbacks &refine_callbacks,
    AnchorBackend anchor_backend,
    const vector<Int3> *warm_component_offsets,
    PbctopoWarmState *warm_state,
    const PbctopoCutCandidate *warm_reference_candidate,
    vector<Int3> *tree_offsets_out,
    vector<Int3> *refined_offsets_out,
    vector<PbctopoGraphEdge> *tree_edges_out,
    PbctopoCycleSyncMetrics *cycle_metrics_out,
    const PbctopoAnchorGraphContext *anchor_context,
    const vector<PbctopoRequiredContact> *required_contacts,
    PbctopoIdentifiability *identifiability_out) {
  if (identifiability_out != nullptr)
    *identifiability_out = {};
  if (components.empty())
    return false;

  if (cycle_metrics_out != nullptr)
    *cycle_metrics_out = {};
  vector<PbctopoPairShiftCandidateSet> pair_candidates;
  vector<PbctopoGraphEdge> ranked_edges = build_component_graph_edge_candidates(
      atoms, Box, components, shifts, cutoffs, timing, warm_component_offsets,
      &pair_candidates, required_contacts);
  if (identifiability_out != nullptr) {
    const vector<size_t> atom_component_index =
        build_atom_component_index(atoms.size(), components);
    const span<const PbctopoRequiredContact> contacts =
        required_contacts != nullptr
            ? span<const PbctopoRequiredContact>(*required_contacts)
            : span<const PbctopoRequiredContact>{};
    *identifiability_out = evaluate_pbctopo_identifiability(
        components.size(), pair_candidates, contacts, atom_component_index);
  }
  const auto tree_started = chrono::steady_clock::now();
  sort(ranked_edges.begin(), ranked_edges.end(), better_component_graph_edge);
  vector<PbctopoGraphEdge> tree_edges;
  if (!build_component_tree_from_ranked_edges(components.size(), ranked_edges, {},
                                              tree_edges))
    return false;
  vector<Int3> offsets;
  if (!solve_component_offsets_from_tree(components.size(), tree_edges, offsets))
    return false;
  if (timing != nullptr) {
    timing->component_graph_tree_s +=
        chrono::duration<double>(chrono::steady_clock::now() - tree_started)
            .count();
  }

  if (!rebase_component_offsets(offsets, Box))
    return false;
  if (tree_offsets_out != nullptr)
    *tree_offsets_out = offsets;
  if (tree_edges_out != nullptr)
    *tree_edges_out = tree_edges;
  const auto refine_started = chrono::steady_clock::now();
  best_cut = improve_component_offsets_locally(atoms, components, Box, shifts,
                                               local_edges, edge_set,
                                               continuity_pairs, anchor_indices,
                                               offsets,
                                               cutoffs.clash_cutoff,
                                               cutoffs.continuity_cutoff,
                                               placed_points, refine_callbacks,
                                               anchor_backend,
                                               timing, warm_state,
                                               warm_reference_candidate,
                                               anchor_context);
  if (timing != nullptr) {
    timing->component_graph_refine_s +=
        chrono::duration<double>(chrono::steady_clock::now() - refine_started)
            .count();
  }
  PbctopoCutCandidate tree_cut = best_cut;
  vector<Vec3> tree_points = placed_points;
  vector<Int3> tree_refined_offsets = offsets;
  PbctopoWarmState tree_warm_state;
  if (warm_state != nullptr)
    tree_warm_state = *warm_state;

  const auto cycle_started = chrono::steady_clock::now();
  if (timing != nullptr)
    ++timing->component_cycle_sync_attempts;
  PbctopoCycleSyncResult cycle_result = solve_component_offsets_cycle_consistent(
      components.size(), Box, pair_candidates,
      max<size_t>(64, component_beam_width));
  if (timing != nullptr) {
    timing->component_cycle_sync_s +=
        chrono::duration<double>(chrono::steady_clock::now() - cycle_started)
            .count();
    timing->component_cycle_sync_cycle_support +=
        cycle_result.metrics.cycle_support;
    timing->component_cycle_sync_cycle_mismatch +=
        cycle_result.metrics.cycle_mismatch;
    timing->component_cycle_sync_inconsistent_edges +=
        cycle_result.metrics.inconsistent_edge_count;
  }
  if (cycle_result.valid &&
      !same_int3_offsets(cycle_result.offsets, tree_refined_offsets)) {
    vector<Vec3> cycle_points;
    vector<Int3> cycle_offsets = cycle_result.offsets;
    const auto cycle_refine_started = chrono::steady_clock::now();
    PbctopoCutCandidate cycle_cut = improve_component_offsets_locally(
        atoms, components, Box, shifts, local_edges, edge_set, continuity_pairs,
        anchor_indices, cycle_offsets, cutoffs.clash_cutoff,
        cutoffs.continuity_cutoff, cycle_points, refine_callbacks,
        anchor_backend, timing, warm_state, warm_reference_candidate,
        anchor_context);
    if (timing != nullptr) {
      timing->component_graph_refine_s +=
          chrono::duration<double>(chrono::steady_clock::now() -
                                   cycle_refine_started)
              .count();
    }
    if (cycle_cut.valid && cycle_points.size() == atoms.size() &&
        refine_callbacks.better_cut_candidate(cycle_cut, tree_cut)) {
      best_cut = cycle_cut;
      placed_points = std::move(cycle_points);
      offsets = std::move(cycle_offsets);
      if (timing != nullptr)
        ++timing->component_cycle_sync_used;
    } else {
      best_cut = tree_cut;
      placed_points = std::move(tree_points);
      offsets = std::move(tree_refined_offsets);
      if (warm_state != nullptr)
        *warm_state = tree_warm_state;
    }
  }

  if (cycle_metrics_out != nullptr) {
    *cycle_metrics_out =
        evaluate_pair_graph_metrics_for_offsets(offsets, pair_candidates, Box);
  }

  if (refined_offsets_out != nullptr)
    *refined_offsets_out = offsets;
  shift_norm2 = best_cut.shift_norm2;
  return best_cut.valid && placed_points.size() == atoms.size();
}
bool solve_component_offsets_from_ranked_edges(
    size_t component_count, box Box, const vector<vector<size_t>> &components,
    const vector<PbctopoGraphEdge> &ranked_edges, vector<Int3> &offsets) {
  vector<PbctopoGraphEdge> tree_edges;
  if (!build_component_tree_from_ranked_edges(component_count, ranked_edges, {},
                                              tree_edges))
    return false;
  if (!solve_component_offsets_from_tree(component_count, tree_edges, offsets))
    return false;
  if (!rebase_component_offsets(offsets, Box))
    return false;
  return true;
}

} // namespace titan_pbctopo

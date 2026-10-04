#include "component_search.h"

#include "candidate_order.h"
#include "component_metric.h"
#include "component_search_pair_table.h"
#include "component_shift_domain.h"
#include "component_shift_geometry.h"
#include "reference_strain.h"
#include "required_contacts.h"
#include "steric_clash.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

namespace titan_pbctopo {
namespace {

std::size_t saturating_size_add(std::size_t lhs, std::size_t rhs) {
  return rhs > std::numeric_limits<std::size_t>::max() - lhs
             ? std::numeric_limits<std::size_t>::max()
             : lhs + rhs;
}

std::size_t saturating_size_multiply(std::size_t lhs, std::size_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs
             ? std::numeric_limits<std::size_t>::max()
             : lhs * rhs;
}

double component_search_bbox_distance2(const PbctopoShiftedComponent &lhs,
                                       const PbctopoShiftedComponent &rhs) {
  const auto axis_gap = [](double min_a, double max_a, double min_b,
                           double max_b) {
    if (max_a < min_b)
      return min_b - max_a;
    if (max_b < min_a)
      return min_a - max_b;
    return 0.0;
  };
  const double dx = axis_gap(lhs.min_x, lhs.max_x, rhs.min_x, rhs.max_x);
  const double dy = axis_gap(lhs.min_y, lhs.max_y, rhs.min_y, rhs.max_y);
  const double dz = axis_gap(lhs.min_z, lhs.max_z, rhs.min_z, rhs.max_z);
  return dx * dx + dy * dy + dz * dz;
}

std::vector<std::size_t> build_search_atom_component_index(
    std::size_t atom_count,
    const std::vector<std::vector<std::size_t>> &components) {
  std::vector<std::size_t> atom_component_index(
      atom_count, std::numeric_limits<std::size_t>::max());
  for (std::size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    for (std::size_t atom_idx : components[comp_idx]) {
      if (atom_idx < atom_component_index.size())
        atom_component_index[atom_idx] = comp_idx;
    }
  }
  return atom_component_index;
}

struct PbctopoSearchContinuityPrefix {
  std::vector<std::vector<std::size_t>> pairs_closing_at_depth;
};

PbctopoSearchContinuityPrefix build_search_continuity_prefix(
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &atom_component_index,
    const std::vector<std::size_t> &search_order, std::size_t component_count) {
  PbctopoSearchContinuityPrefix prefix;
  prefix.pairs_closing_at_depth.assign(search_order.size(), {});
  if (continuity_pairs.empty() || atom_component_index.empty() ||
      search_order.empty())
    return prefix;

  std::vector<std::size_t> component_depth(
      component_count, std::numeric_limits<std::size_t>::max());
  for (std::size_t depth = 0; depth < search_order.size(); ++depth) {
    if (search_order[depth] < component_depth.size())
      component_depth[search_order[depth]] = depth;
  }

  for (std::size_t pair_idx = 0; pair_idx < continuity_pairs.size();
       ++pair_idx) {
    const auto &pair = continuity_pairs[pair_idx];
    if (pair.a >= atom_component_index.size() ||
        pair.b >= atom_component_index.size())
      continue;
    const std::size_t comp_a = atom_component_index[pair.a];
    const std::size_t comp_b = atom_component_index[pair.b];
    if (comp_a >= component_depth.size() || comp_b >= component_depth.size())
      continue;
    const std::size_t depth_a = component_depth[comp_a];
    const std::size_t depth_b = component_depth[comp_b];
    if (depth_a == std::numeric_limits<std::size_t>::max() ||
        depth_b == std::numeric_limits<std::size_t>::max())
      continue;
    const std::size_t close_depth = std::max(depth_a, depth_b);
    if (close_depth < prefix.pairs_closing_at_depth.size())
      prefix.pairs_closing_at_depth[close_depth].push_back(pair_idx);
  }
  return prefix;
}

bool consume_search_continuity_prefix(
    const std::vector<Vec3> &placed_points,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &pair_indices, double hard_cutoff2,
    const PbctopoComponentCandidate *incumbent, double &path_sum,
    double &max_d2, PbctopoTimingStats *timing,
    bool *incumbent_bound_pruned = nullptr) {
  if (pair_indices.empty())
    return true;

  using clock = std::chrono::steady_clock;
  const auto started = clock::now();
  auto finish = [&](bool ok) {
    if (timing != nullptr) {
      timing->component_search_prefix_continuity_s +=
          std::chrono::duration<double>(clock::now() - started).count();
    }
    return ok;
  };

  const bool has_incumbent = incumbent != nullptr && incumbent->valid &&
                             std::isfinite(incumbent->continuity_path);

  for (std::size_t pair_idx : pair_indices) {
    if (pair_idx >= continuity_pairs.size())
      continue;
    const auto &pair = continuity_pairs[pair_idx];
    if (pair.a >= placed_points.size() || pair.b >= placed_points.size())
      continue;
    const Vec3 &point_a = placed_points[pair.a];
    const Vec3 &point_b = placed_points[pair.b];
    const double dx = point_a.x - point_b.x;
    const double dy = point_a.y - point_b.y;
    const double dz = point_a.z - point_b.z;
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (timing != nullptr)
      ++timing->component_search_prefix_continuity_checks;
    const bool hard_pair = pbctopo_continuity_is_hard(pair.kind);
    if (hard_pair && d2 > max_d2)
      max_d2 = d2;
    if (pbctopo_continuity_hard_violation(pair, d2, hard_cutoff2)) {
      if (timing != nullptr) {
        ++timing->component_search_prefix_continuity_prunes;
        ++timing->component_search_prefix_continuity_hard_rejects;
      }
      return finish(false);
    }
    path_sum += pair.weight *
                pbctopo_symmetric_reference_strain(d2, pair.ref_d2);
    if (has_incumbent &&
        path_sum > incumbent->continuity_path + 1.0e-6) {
      if (incumbent_bound_pruned != nullptr)
        *incumbent_bound_pruned = true;
      if (timing != nullptr)
        ++timing->component_search_prefix_continuity_prunes;
      return finish(false);
    }
  }
  return finish(true);
}

bool consume_search_continuity_prefix_from_assignment(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &atom_component_index,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<std::size_t> &selected_shift,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &pair_indices, double hard_cutoff2,
    const PbctopoComponentCandidate *incumbent, double &path_sum,
    double &max_d2, PbctopoTimingStats *timing,
    std::size_t override_comp_idx = std::numeric_limits<std::size_t>::max(),
    std::size_t override_shift_idx = std::numeric_limits<std::size_t>::max(),
    bool *incumbent_bound_pruned = nullptr) {
  if (pair_indices.empty())
    return true;

  using clock = std::chrono::steady_clock;
  const auto started = clock::now();
  auto finish = [&](bool ok) {
    if (timing != nullptr) {
      timing->component_search_prefix_continuity_s +=
          std::chrono::duration<double>(clock::now() - started).count();
    }
    return ok;
  };
  const bool has_incumbent = incumbent != nullptr && incumbent->valid &&
                             std::isfinite(incumbent->continuity_path);

  for (std::size_t pair_idx : pair_indices) {
    if (pair_idx >= continuity_pairs.size())
      continue;
    const auto &pair = continuity_pairs[pair_idx];
    if (pair.a >= atoms.size() || pair.b >= atoms.size() ||
        pair.a >= atom_component_index.size() ||
        pair.b >= atom_component_index.size()) {
      continue;
    }
    const std::size_t comp_a = atom_component_index[pair.a];
    const std::size_t comp_b = atom_component_index[pair.b];
    if (comp_a >= selected_shift.size() || comp_b >= selected_shift.size())
      return finish(false);
    const std::size_t shift_a_idx =
        comp_a == override_comp_idx ? override_shift_idx : selected_shift[comp_a];
    const std::size_t shift_b_idx =
        comp_b == override_comp_idx ? override_shift_idx : selected_shift[comp_b];
    if (shift_a_idx >= shifts.size() || shift_b_idx >= shifts.size())
      return finish(false);
    const auto &shift_a = shifts[shift_a_idx];
    const auto &shift_b = shifts[shift_b_idx];
    const double dx = atoms[pair.a].x + shift_a.tx -
                      atoms[pair.b].x - shift_b.tx;
    const double dy = atoms[pair.a].y + shift_a.ty -
                      atoms[pair.b].y - shift_b.ty;
    const double dz = atoms[pair.a].z + shift_a.tz -
                      atoms[pair.b].z - shift_b.tz;
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (timing != nullptr)
      ++timing->component_search_prefix_continuity_checks;
    const bool hard_pair = pbctopo_continuity_is_hard(pair.kind);
    if (hard_pair)
      max_d2 = std::max(max_d2, d2);
    if (pbctopo_continuity_hard_violation(pair, d2, hard_cutoff2)) {
      if (timing != nullptr) {
        ++timing->component_search_prefix_continuity_prunes;
        ++timing->component_search_prefix_continuity_hard_rejects;
      }
      return finish(false);
    }
    path_sum += pair.weight *
                pbctopo_symmetric_reference_strain(d2, pair.ref_d2);
    if (has_incumbent && path_sum > incumbent->continuity_path + 1.0e-6) {
      if (incumbent_bound_pruned != nullptr)
        *incumbent_bound_pruned = true;
      if (timing != nullptr)
        ++timing->component_search_prefix_continuity_prunes;
      return finish(false);
    }
  }
  return finish(true);
}

PbctopoComponentPairShiftEntry compute_component_pair_shift_entry(
    const PbctopoShiftedComponent &lhs,
    const PbctopoShiftedComponent &rhs, double clash_cutoff,
    double bbox_search_cutoff,
    const std::vector<PbctopoAtom> &atoms, PbctopoTimingStats *timing) {
  PbctopoComponentPairShiftEntry entry;
  if (!lhs.spatial_keys_valid || !rhs.spatial_keys_valid) {
    entry.hard_clash = true;
    return entry;
  }
  entry.bbox_d2 = component_search_bbox_distance2(lhs, rhs);
  if (entry.bbox_d2 > bbox_search_cutoff * bbox_search_cutoff)
    return entry;

  const std::vector<std::size_t> *scan_atom_indices = &lhs.atom_indices;
  const std::vector<std::size_t> *grid_atom_indices = &rhs.atom_indices;
  const PbctopoShiftedComponent *scan_shifted = &lhs;
  const PbctopoShiftedComponent *grid_shifted = &rhs;
  if (lhs.points.size() > rhs.points.size()) {
    scan_atom_indices = &rhs.atom_indices;
    grid_atom_indices = &lhs.atom_indices;
    scan_shifted = &rhs;
    grid_shifted = &lhs;
  }

  for (std::size_t local_idx = 0; local_idx < scan_shifted->points.size();
       ++local_idx) {
    const std::size_t atom_idx = (*scan_atom_indices)[local_idx];
    const auto &point = scan_shifted->points[local_idx];
    const auto &key = scan_shifted->clash_keys[local_idx];
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          ClashCellKey neighbor;
          if (!pbctopo_offset_cell_key(key, dx, dy, dz, neighbor)) {
            entry.hard_clash = true;
            return entry;
          }
          const auto it = grid_shifted->clash_grid.find(neighbor);
          if (it == grid_shifted->clash_grid.end())
            continue;
          for (std::size_t other_local_idx : it->second) {
            const std::size_t other_idx = (*grid_atom_indices)[other_local_idx];
            const auto &other = grid_shifted->points[other_local_idx];
            const double ddx = point.x - other.x;
            const double ddy = point.y - other.y;
            const double ddz = point.z - other.z;
            const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 < entry.min_pair_d2) {
              entry.min_pair_d2 = d2;
              entry.clash_atom_a = atom_idx;
              entry.clash_atom_b = other_idx;
            }
            if (atom_idx < atoms.size() && other_idx < atoms.size() &&
                pbctopo_atoms_clash(d2, atoms[atom_idx], atoms[other_idx],
                                     clash_cutoff, timing)) {
              entry.hard_clash = true;
              return entry;
            }
          }
        }
      }
    }
  }
  return entry;
}

PbctopoComponentPairShiftTable build_component_pair_shift_table(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts, box Box, double clash_cutoff,
    double search_cutoff, double contact_cutoff,
    PbctopoTimingStats *timing,
    const std::vector<PbctopoRequiredContact> *required_contacts,
    bool *hard_constraints_feasible) {
  using clock = std::chrono::steady_clock;
  const auto started = clock::now();
  PbctopoComponentPairShiftTable table;
  const std::size_t component_count = components.size();
  const std::size_t shift_count = shifts.size();
  table.reset(component_count, shifts);
  if (hard_constraints_feasible != nullptr)
    *hard_constraints_feasible = true;

  PbctopoRequiredContactConstraintContext required_contact_context;
  const bool has_required_contacts =
      required_contacts != nullptr && !required_contacts->empty();
  if (has_required_contacts) {
    required_contact_context =
        build_pbctopo_required_contact_constraint_context(
            atoms, components, Box, *required_contacts);
    const bool feasible = required_contact_context.valid &&
                          !required_contact_context.infeasible;
    if (hard_constraints_feasible != nullptr)
      *hard_constraints_feasible = feasible;
    if (timing != nullptr) {
      ++timing->component_required_contact_constraint_builds;
      timing->component_required_contact_constraint_pairs +=
          required_contact_context.groups.size();
      if (!feasible)
        ++timing->component_required_contact_infeasible;
    }
    if (!feasible)
      return table;
  }

  std::vector<PbctopoComponentBaseGeometry> base_geometry;
  std::vector<PbctopoComponentBaseGeometry> all_atom_base_geometry;
  base_geometry.reserve(component_count);
  all_atom_base_geometry.reserve(component_count);
  for (const auto &component : components) {
    base_geometry.push_back(
        pbctopo_build_component_base_geometry(atoms, component));
    all_atom_base_geometry.push_back(
        pbctopo_build_component_all_atom_base_geometry(atoms, component));
  }
  bool relative_domain_analytic = false;
  const auto relative_shifts = build_component_relative_shift_domain(
      shifts, &relative_domain_analytic);
  if (timing != nullptr) {
    if (relative_domain_analytic)
      ++timing->component_relative_domain_analytic_builds;
    else
      ++timing->component_relative_domain_generic_builds;
  }
  const PbctopoShift identity_shift{};
  std::vector<PbctopoShiftedComponent> identity_components(component_count);
  std::vector<PbctopoShiftedComponent> all_atom_identity_components(
      component_count);
  std::vector<unsigned char> identity_materialized(component_count, 0);
  std::vector<unsigned char> all_atom_identity_materialized(component_count,
                                                             0);
  std::vector<PbctopoShiftedComponentDescriptor> identity_descriptors;
  identity_descriptors.reserve(component_count);
  for (const auto &base : all_atom_base_geometry)
    identity_descriptors.push_back(
        pbctopo_describe_shifted_component(base, identity_shift));
  PbctopoShiftedComponent rhs_shifted;
  PbctopoShiftedComponent all_atom_rhs_shifted;
  double materialize_s = 0.0;
  std::size_t materialized_count = 0;

  for (std::size_t rhs_comp = 1; rhs_comp < component_count; ++rhs_comp) {
    for (const auto &relative : relative_shifts) {
      const auto rhs_descriptor = pbctopo_describe_shifted_component(
          all_atom_base_geometry[rhs_comp], relative.shift);
      bool rhs_materialized = false;
      bool all_atom_rhs_materialized = false;
      for (std::size_t lhs_comp = 0; lhs_comp < rhs_comp; ++lhs_comp) {
        const Int3 relative_delta{relative.shift.ix, relative.shift.iy,
                                  relative.shift.iz};
        const bool has_pair_contact_constraints =
            has_required_contacts &&
            find_pbctopo_required_contact_constraint_group(
                required_contact_context, lhs_comp, rhs_comp) != nullptr;
        if (has_pair_contact_constraints) {
          if (timing != nullptr)
            ++timing->component_required_contact_relative_deltas_checked;
          if (!pbctopo_required_contact_relative_shift_allowed(
                  required_contact_context, atoms, lhs_comp, rhs_comp,
                  relative_delta)) {
            table.mark_relative_incompatible(lhs_comp, rhs_comp,
                                             relative_delta);
            if (timing != nullptr)
              ++timing->component_required_contact_relative_deltas_rejected;
            continue;
          }
        }
        const double bbox_d2 = pbctopo_shifted_bbox_distance2(
            identity_descriptors[lhs_comp], rhs_descriptor);
        if (bbox_d2 > search_cutoff * search_cutoff) {
          table.add_bbox_prefiltered(2 * relative.multiplicity);
          continue;
        }
        if (identity_materialized[lhs_comp] == 0) {
          const auto materialize_started = clock::now();
          pbctopo_materialize_shifted_component(
              base_geometry[lhs_comp], identity_shift, search_cutoff,
              contact_cutoff, true, false, identity_components[lhs_comp]);
          materialize_s +=
              std::chrono::duration<double>(clock::now() - materialize_started)
                  .count();
          identity_materialized[lhs_comp] = 1;
          ++materialized_count;
        }
        if (!rhs_materialized) {
          const auto materialize_started = clock::now();
          pbctopo_materialize_shifted_component(
              base_geometry[rhs_comp], relative.shift, search_cutoff,
              contact_cutoff, true, false, rhs_shifted);
          materialize_s +=
              std::chrono::duration<double>(clock::now() - materialize_started)
                  .count();
          rhs_materialized = true;
          ++materialized_count;
        }
        const auto heavy_entry = compute_component_pair_shift_entry(
            identity_components[lhs_comp], rhs_shifted, clash_cutoff,
            search_cutoff, atoms, timing);
        if (heavy_entry.hard_clash) {
          table.store_relative_pair(lhs_comp, rhs_comp, relative_delta,
                                    heavy_entry);
          continue;
        }

        PbctopoComponentPairShiftEntry exact_entry = heavy_entry;
        const bool requires_all_atom_scan =
            all_atom_base_geometry[lhs_comp].x.size() !=
                base_geometry[lhs_comp].x.size() ||
            all_atom_base_geometry[rhs_comp].x.size() !=
                base_geometry[rhs_comp].x.size();
        if (requires_all_atom_scan) {
          if (all_atom_identity_materialized[lhs_comp] == 0) {
            const auto materialize_started = clock::now();
            pbctopo_materialize_shifted_component(
                all_atom_base_geometry[lhs_comp], identity_shift,
                search_cutoff, contact_cutoff, true, false,
                all_atom_identity_components[lhs_comp]);
            materialize_s += std::chrono::duration<double>(
                                 clock::now() - materialize_started)
                                 .count();
            all_atom_identity_materialized[lhs_comp] = 1;
            ++materialized_count;
            if (timing != nullptr)
              ++timing->component_pair_table_all_atom_materialized;
          }
          if (!all_atom_rhs_materialized) {
            const auto materialize_started = clock::now();
            pbctopo_materialize_shifted_component(
                all_atom_base_geometry[rhs_comp], relative.shift,
                search_cutoff, contact_cutoff, true, false,
                all_atom_rhs_shifted);
            materialize_s += std::chrono::duration<double>(
                                 clock::now() - materialize_started)
                                 .count();
            all_atom_rhs_materialized = true;
            ++materialized_count;
            if (timing != nullptr)
              ++timing->component_pair_table_all_atom_materialized;
          }
          exact_entry = compute_component_pair_shift_entry(
              all_atom_identity_components[lhs_comp], all_atom_rhs_shifted,
              clash_cutoff, search_cutoff, atoms, timing);
          if (timing != nullptr) {
            ++timing->component_pair_table_all_atom_relative_checks;
            if (exact_entry.hard_clash)
              ++timing->component_pair_table_all_atom_relative_rejects;
          }
        }
        if (exact_entry.hard_clash ||
            std::isfinite(exact_entry.min_pair_d2)) {
          table.store_relative_pair(lhs_comp, rhs_comp, relative_delta,
                                    exact_entry);
        }
      }
    }
  }

  if (timing != nullptr) {
    ++timing->component_pair_table_builds;
    timing->component_pair_table_s +=
        std::chrono::duration<double>(clock::now() - started).count();
    timing->component_pair_table_component_count =
        std::max(timing->component_pair_table_component_count, component_count);
    timing->component_pair_table_shift_count =
        std::max(timing->component_pair_table_shift_count, shift_count);
    timing->component_pair_table_relative_deltas = std::max(
        timing->component_pair_table_relative_deltas,
        relative_shifts.size());
    const std::size_t component_pair_count =
        component_count > 1
            ? (component_count * (component_count - 1)) / 2
            : 0;
    timing->component_pair_table_relative_reuses +=
        2 * component_pair_count *
        (shift_count * shift_count - relative_shifts.size());
    timing->component_search_shift_descriptors +=
        component_pair_count * (relative_shifts.size() + 1);
    timing->component_search_shift_materialized += materialized_count;
    timing->component_search_lazy_materialize_s += materialize_s;
    timing->component_pair_table_dense_entries_equivalent =
        std::max(timing->component_pair_table_dense_entries_equivalent,
                 table.dense_entries_equivalent());
    timing->component_pair_table_materialized_entries =
        std::max(timing->component_pair_table_materialized_entries,
                 table.materialized_entries());
    timing->component_pair_table_bbox_prefiltered += table.bbox_prefiltered();
    timing->component_pair_table_sparse_rows =
        std::max(timing->component_pair_table_sparse_rows, table.sparse_rows());
    timing->component_pair_table_peak_bytes_estimate =
        std::max(timing->component_pair_table_peak_bytes_estimate,
                 table.peak_bytes_estimate());
  }
  return table;
}

std::vector<std::size_t> build_component_search_order(
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<std::size_t> &anchor_indices,
    const std::vector<std::size_t> &atom_component_index,
    const PbctopoComponentPairShiftTable &pair_table,
    std::size_t identity_shift_pos) {
  std::vector<std::size_t> order;
  if (components.empty())
    return order;
  order.reserve(components.size());
  order.push_back(0);
  if (components.size() == 1)
    return order;

  std::vector<std::size_t> anchor_counts(components.size(), 0);
  for (std::size_t anchor_idx : anchor_indices) {
    if (anchor_idx < atom_component_index.size()) {
      const std::size_t comp_idx = atom_component_index[anchor_idx];
      if (comp_idx < anchor_counts.size())
        ++anchor_counts[comp_idx];
    }
  }

  std::vector<std::size_t> remaining;
  remaining.reserve(components.size() - 1);
  for (std::size_t comp_idx = 1; comp_idx < components.size(); ++comp_idx)
    remaining.push_back(comp_idx);

  std::stable_sort(remaining.begin(), remaining.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     if (components[lhs].size() != components[rhs].size())
                       return components[lhs].size() > components[rhs].size();
                     if (anchor_counts[lhs] != anchor_counts[rhs])
                       return anchor_counts[lhs] > anchor_counts[rhs];

                     const std::size_t lhs_compatible_shifts =
                         pair_table.compat_mask(0, identity_shift_pos, lhs)
                             .count();
                     const std::size_t rhs_compatible_shifts =
                         pair_table.compat_mask(0, identity_shift_pos, rhs)
                             .count();
                     if (lhs_compatible_shifts != rhs_compatible_shifts)
                       return lhs_compatible_shifts < rhs_compatible_shifts;
                     return lhs < rhs;
                   });

  order.insert(order.end(), remaining.begin(), remaining.end());
  return order;
}

struct PbctopoComponentSearchShiftOption {
  std::size_t shift_idx = 0;
  double next_min_pair_d2 = std::numeric_limits<double>::infinity();
  double shift_norm2 = std::numeric_limits<double>::infinity();
  std::size_t future_remaining_shifts =
      std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  bool warm_priority = false;
};

struct PbctopoBeamState {
  std::vector<std::size_t> selected_shift;
  std::vector<PbctopoShiftMask> allowed_masks;
  double shift_norm2 = 0.0;
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double continuity_path_prefix = 0.0;
  double continuity_max_d2_prefix = 0.0;
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  std::size_t future_remaining_shifts =
      std::numeric_limits<std::size_t>::max();
  std::size_t warm_hits = 0;
};

struct PbctopoProvisionalBeamChild {
  std::size_t parent_state_idx = 0;
  std::size_t shift_idx = 0;
  double shift_norm2 = 0.0;
  double min_pair_d2 = std::numeric_limits<double>::infinity();
  double continuity_path_prefix = 0.0;
  double continuity_max_d2_prefix = 0.0;
  std::size_t clash_atom_a = std::numeric_limits<std::size_t>::max();
  std::size_t clash_atom_b = std::numeric_limits<std::size_t>::max();
  std::size_t future_remaining_shifts =
      std::numeric_limits<std::size_t>::max();
  std::size_t warm_hits = 0;
};

std::vector<Int3> normalize_search_warm_offsets(
    const std::vector<Int3> &warm_offsets,
    box Box) {
  if (warm_offsets.empty())
    return {};
  std::vector<Int3> rebased = warm_offsets;
  if (!rebase_component_offsets(rebased, Box))
    return {};
  std::vector<Int3> normalized;
  normalized.reserve(rebased.size());
  const Int3 root = rebased.front();
  for (const auto &offset : rebased) {
    Int3 relative{};
    if (!checked_int3_subtract(offset, root, relative))
      return {};
    normalized.push_back(relative);
  }
  return normalized;
}

std::vector<PbctopoComponentSearchShiftOption> build_component_search_shift_order(
    const std::vector<PbctopoShift> &shifts, std::size_t actual_comp_idx,
    std::size_t depth, const std::vector<std::size_t> &search_order,
    const std::vector<std::size_t> &selected_shift,
    const std::vector<PbctopoShiftMask> &allowed_masks,
    const PbctopoComponentPairShiftTable &pair_table,
    double current_min_pair_d2, std::size_t clash_atom_a,
    std::size_t clash_atom_b,
    const std::vector<Int3> *normalized_warm_offsets = nullptr,
    PbctopoTimingStats *timing = nullptr) {
  std::vector<PbctopoComponentSearchShiftOption> ordered;
  if (actual_comp_idx >= allowed_masks.size())
    return ordered;
  const PbctopoShiftMask current_mask = allowed_masks[actual_comp_idx];
  if (current_mask.empty()) {
    if (timing != nullptr) {
      ++timing->component_pair_table_prunes;
      ++timing->component_search_zero_current_mask_prunes;
    }
    return ordered;
  }
  ordered.reserve(current_mask.count());

  Int3 warm_delta;
  const bool has_warm_delta =
      (depth > 0 && normalized_warm_offsets != nullptr &&
       actual_comp_idx < normalized_warm_offsets->size())
          ? (warm_delta = (*normalized_warm_offsets)[actual_comp_idx], true)
          : false;

  current_mask.for_each_set_bit([&](std::size_t shift_idx) {
    if (shift_idx >= shifts.size())
      return;
    PbctopoComponentSearchShiftOption option;
    option.shift_idx = shift_idx;
    option.next_min_pair_d2 = current_min_pair_d2;
    option.future_remaining_shifts = 0;
    option.shift_norm2 = pbctopo_translation_norm2(shifts[shift_idx]);
    option.clash_atom_a = clash_atom_a;
    option.clash_atom_b = clash_atom_b;
    option.warm_priority =
        has_warm_delta && shifts[shift_idx].ix == warm_delta.x &&
        shifts[shift_idx].iy == warm_delta.y &&
        shifts[shift_idx].iz == warm_delta.z;

    for (std::size_t prev_depth = 0; prev_depth < depth; ++prev_depth) {
      const std::size_t prev_comp_idx = search_order[prev_depth];
      const std::size_t prev_shift_idx = selected_shift[prev_comp_idx];
      if (prev_shift_idx >= pair_table.shift_count)
        continue;
      const auto &entry =
          pair_table.at(actual_comp_idx, shift_idx, prev_comp_idx,
                        prev_shift_idx);
      if (entry.min_pair_d2 < option.next_min_pair_d2) {
        option.next_min_pair_d2 = entry.min_pair_d2;
        option.clash_atom_a = entry.clash_atom_a;
        option.clash_atom_b = entry.clash_atom_b;
      }
    }

    bool future_zero_mask = false;
    for (std::size_t future_depth = depth + 1;
         future_depth < search_order.size(); ++future_depth) {
      const std::size_t future_comp_idx = search_order[future_depth];
      if (future_comp_idx >= allowed_masks.size())
        continue;
      PbctopoShiftMask next_mask = allowed_masks[future_comp_idx];
      next_mask.intersect_with(
          pair_table.compat_mask(actual_comp_idx, shift_idx, future_comp_idx));
      if (next_mask.empty()) {
        future_zero_mask = true;
        break;
      }
      option.future_remaining_shifts += next_mask.count();
    }
    if (future_zero_mask) {
      if (timing != nullptr)
        ++timing->component_pair_table_prunes;
      if (timing != nullptr)
        ++timing->component_search_zero_future_mask_prunes;
      return;
    }
    ordered.push_back(option);
  });

  std::sort(ordered.begin(), ordered.end(),
            [](const PbctopoComponentSearchShiftOption &lhs,
               const PbctopoComponentSearchShiftOption &rhs) {
              if (lhs.warm_priority != rhs.warm_priority)
                return lhs.warm_priority;
              if (lhs.future_remaining_shifts != rhs.future_remaining_shifts)
                return lhs.future_remaining_shifts < rhs.future_remaining_shifts;
               const int metric_cmp = compare_pbctopo_metric(
                   lhs.shift_norm2, rhs.shift_norm2, {1.0e-9, 0.0});
               if (metric_cmp != 0)
                 return metric_cmp < 0;
              return lhs.shift_idx < rhs.shift_idx;
            });
  if (!ordered.empty() && ordered.front().warm_priority && timing != nullptr)
    ++timing->component_search_shift_warm_hits;
  return ordered;
}

bool better_component_beam_state(const PbctopoBeamState &lhs,
                                 const PbctopoBeamState &rhs) {
  int metric_cmp = compare_pbctopo_metric(
      lhs.continuity_path_prefix, rhs.continuity_path_prefix,
      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_metric(
      lhs.continuity_max_d2_prefix, rhs.continuity_max_d2_prefix,
      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.warm_hits != rhs.warm_hits)
    return lhs.warm_hits > rhs.warm_hits;
  if (lhs.future_remaining_shifts != rhs.future_remaining_shifts)
    return lhs.future_remaining_shifts < rhs.future_remaining_shifts;
  metric_cmp = compare_pbctopo_metric(lhs.shift_norm2, rhs.shift_norm2,
                                      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return lhs.selected_shift < rhs.selected_shift;
}

bool better_provisional_beam_child(
    const PbctopoProvisionalBeamChild &lhs,
    const PbctopoProvisionalBeamChild &rhs,
    const std::vector<PbctopoBeamState> &parents,
    std::size_t assigned_comp_idx) {
  int metric_cmp = compare_pbctopo_metric(
      lhs.continuity_path_prefix, rhs.continuity_path_prefix,
      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_metric(
      lhs.continuity_max_d2_prefix, rhs.continuity_max_d2_prefix,
      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.warm_hits != rhs.warm_hits)
    return lhs.warm_hits > rhs.warm_hits;
  if (lhs.future_remaining_shifts != rhs.future_remaining_shifts)
    return lhs.future_remaining_shifts < rhs.future_remaining_shifts;
  metric_cmp = compare_pbctopo_metric(lhs.shift_norm2, rhs.shift_norm2,
                                      {1.0e-9, 0.0});
  if (metric_cmp != 0)
    return metric_cmp < 0;

  const auto &lhs_parent = parents[lhs.parent_state_idx].selected_shift;
  const auto &rhs_parent = parents[rhs.parent_state_idx].selected_shift;
  const std::size_t component_count =
      std::min(lhs_parent.size(), rhs_parent.size());
  for (std::size_t comp_idx = 0; comp_idx < component_count; ++comp_idx) {
    const std::size_t lhs_shift =
        comp_idx == assigned_comp_idx ? lhs.shift_idx : lhs_parent[comp_idx];
    const std::size_t rhs_shift =
        comp_idx == assigned_comp_idx ? rhs.shift_idx : rhs_parent[comp_idx];
    if (lhs_shift != rhs_shift)
      return lhs_shift < rhs_shift;
  }
  return lhs_parent.size() < rhs_parent.size();
}

bool build_component_points_from_selected_shifts(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<std::size_t> &selected_shift,
    std::vector<Vec3> &placed_points, double &shift_norm2) {
  if (components.size() != selected_shift.size())
    return false;
  placed_points.resize(atoms.size());
  shift_norm2 = 0.0;
  for (std::size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    const std::size_t shift_idx = selected_shift[comp_idx];
    if (shift_idx >= shifts.size())
      return false;
    const auto &component = components[comp_idx];
    const auto &shift = shifts[shift_idx];
    shift_norm2 += pbctopo_translation_norm2(shift);
    for (std::size_t atom_idx : component) {
      if (atom_idx >= atoms.size())
        return false;
      placed_points[atom_idx] = {atoms[atom_idx].x + shift.tx,
                                 atoms[atom_idx].y + shift.ty,
                                 atoms[atom_idx].z + shift.tz};
    }
  }
  shift_norm2 /= static_cast<double>(components.size());
  return true;
}

std::vector<Int3> component_offsets_from_selected_shifts(
    const std::vector<std::size_t> &selected_shift,
    const std::vector<PbctopoShift> &shifts) {
  std::vector<Int3> offsets;
  offsets.reserve(selected_shift.size());
  for (std::size_t shift_idx : selected_shift) {
    if (shift_idx >= shifts.size())
      return {};
    const auto &shift = shifts[shift_idx];
    offsets.push_back({shift.ix, shift.iy, shift.iz});
  }
  return offsets;
}

std::size_t component_shift_shell_radius(
    const std::vector<PbctopoShift> &shifts) {
  std::size_t radius = 0;
  for (const auto &shift : shifts) {
    const auto abs_index = [](int value) {
      return static_cast<std::size_t>(
          value < 0 ? -static_cast<std::int64_t>(value)
                    : static_cast<std::int64_t>(value));
    };
    radius = std::max(
        radius, std::max({abs_index(shift.ix), abs_index(shift.iy),
                          abs_index(shift.iz)}));
  }
  return radius;
}

void annotate_search_domain_witness(
    PbctopoComponentCandidate &candidate,
    const std::vector<std::size_t> &selected_shift,
    const std::vector<PbctopoShift> &shifts, std::size_t shell_radius) {
  candidate.component_search_shell_radius = shell_radius;
  candidate.bounded_domain_only = true;
  candidate.domain_completeness =
      PbctopoDomainCompleteness::BoundedExhaustive;
  candidate.search_domain_audit_known =
      !selected_shift.empty() && selected_shift.size() > 0;
  if (!candidate.search_domain_audit_known)
    return;
  for (std::size_t shift_idx : selected_shift) {
    if (shift_idx >= shifts.size()) {
      candidate.search_domain_audit_known = false;
      candidate.selected_assignment_touches_shell_boundary = false;
      return;
    }
    const auto &shift = shifts[shift_idx];
    const auto on_boundary = [shell_radius](int value) {
      const std::size_t magnitude = static_cast<std::size_t>(
          value < 0 ? -static_cast<std::int64_t>(value)
                    : static_cast<std::int64_t>(value));
      return shell_radius > 0 && magnitude == shell_radius;
    };
    if (on_boundary(shift.ix) || on_boundary(shift.iy) ||
        on_boundary(shift.iz)) {
      candidate.selected_assignment_touches_shell_boundary = true;
      return;
    }
  }
}

void record_component_search_survivor(
    const PbctopoComponentCandidate &candidate,
    const std::vector<std::size_t> &selected_shift,
    const std::vector<PbctopoShift> &shifts,
    const PbctopoComponentSearchCallbacks &callbacks,
    std::vector<PbctopoComponentSearchSurvivor> *survivors_out,
    std::size_t survivor_limit) {
  if (survivors_out == nullptr || survivor_limit == 0 || !candidate.valid ||
      candidate.ultra_clash) {
    return;
  }
  if (survivors_out->size() >= survivor_limit &&
      !callbacks.better_candidate(candidate, survivors_out->back().candidate)) {
    return;
  }
  std::vector<Int3> offsets =
      component_offsets_from_selected_shifts(selected_shift, shifts);
  if (offsets.size() != selected_shift.size())
    return;
  auto existing = std::find_if(
      survivors_out->begin(), survivors_out->end(),
      [&](const PbctopoComponentSearchSurvivor &survivor) {
        return survivor.component_offsets == offsets;
      });
  if (existing != survivors_out->end()) {
    if (!callbacks.better_candidate(candidate, existing->candidate))
      return;
    existing->candidate = candidate;
    while (existing != survivors_out->begin()) {
      auto previous = existing;
      --previous;
      if (!callbacks.better_candidate(existing->candidate,
                                      previous->candidate)) {
        break;
      }
      std::iter_swap(previous, existing);
      existing = previous;
    }
  } else {
    const auto insertion = std::upper_bound(
        survivors_out->begin(), survivors_out->end(), candidate,
        [&](const PbctopoComponentCandidate &value,
            const PbctopoComponentSearchSurvivor &survivor) {
          return callbacks.better_candidate(value, survivor.candidate);
        });
    survivors_out->insert(insertion, {candidate, std::move(offsets)});
  }
  if (survivors_out->size() > survivor_limit)
    survivors_out->pop_back();
}

std::size_t identity_shift_index(const std::vector<PbctopoShift> &shifts) {
  for (std::size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
    if (shifts[shift_idx].ix == 0 && shifts[shift_idx].iy == 0 &&
        shifts[shift_idx].iz == 0) {
      return shift_idx;
    }
  }
  return 0;
}

std::size_t component_domain_product_upper_bound(
    const std::vector<PbctopoShiftMask> &domains,
    std::size_t saturation_limit) {
  if (domains.empty())
    return 1;
  if (saturation_limit == 0)
    return 0;
  std::size_t product = 1;
  for (const auto &domain : domains) {
    const std::size_t count = domain.count();
    if (count == 0)
      return 0;
    if (product > saturation_limit / count)
      return saturation_limit;
    product *= count;
  }
  return product;
}

bool enforce_search_arc_consistency(
    const PbctopoComponentPairShiftTable &pair_table,
    std::vector<PbctopoShiftMask> &domains, PbctopoTimingStats *timing) {
  using clock = std::chrono::steady_clock;
  const auto started = clock::now();
  std::size_t passes = 0;
  std::size_t revisions = 0;
  std::size_t removals = 0;
  std::size_t queue_pushes = 0;
  const bool consistent = pbctopo_enforce_pair_arc_consistency(
      pair_table, domains, &passes, &revisions, &removals, &queue_pushes);
  if (timing != nullptr) {
    timing->component_search_arc_consistency_passes += passes;
    timing->component_search_arc_consistency_revisions += revisions;
    timing->component_search_arc_consistency_removals += removals;
    timing->component_search_arc_consistency_queue_pushes += queue_pushes;
    timing->component_search_arc_consistency_failures += consistent ? 0 : 1;
    timing->component_search_arc_consistency_s +=
        std::chrono::duration<double>(clock::now() - started).count();
    const std::size_t filtered_upper_bound =
        component_domain_product_upper_bound(
            domains, kPbctopoExactAssignmentBudget + 1);
    timing->component_search_filtered_state_upper_bound = std::max(
        timing->component_search_filtered_state_upper_bound,
        filtered_upper_bound);
  }
  return consistent;
}

} // namespace

const char *pbctopo_search_stop_reason_name(PbctopoSearchStopReason reason) {
  switch (reason) {
  case PbctopoSearchStopReason::StateLimit:
    return "state_limit";
  case PbctopoSearchStopReason::WallTimeLimit:
    return "wall_time_limit";
  case PbctopoSearchStopReason::MemoryLimit:
    return "memory_limit";
  case PbctopoSearchStopReason::None:
  default:
    return "none";
  }
}

std::size_t pbctopo_component_assignment_upper_bound(
    std::size_t component_count, std::size_t shift_count,
    std::size_t saturation_limit) {
  if (component_count <= 1)
    return 1;
  if (shift_count == 0 || saturation_limit == 0)
    return 0;

  std::size_t state_count = 1;
  for (std::size_t depth = 1; depth < component_count; ++depth) {
    if (state_count > saturation_limit / shift_count)
      return saturation_limit;
    state_count *= shift_count;
  }
  return state_count;
}

PbctopoComponentCandidate search_component_shift_candidate_beam(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, std::vector<Vec3> &best_points_out,
    std::size_t beam_width, const PbctopoComponentSearchCallbacks &callbacks,
    const std::vector<Int3> *warm_component_offsets,
    const PbctopoCutCandidate *incumbent_cut,
    const std::vector<Vec3> *incumbent_points, PbctopoTimingStats *timing,
    AnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context,
    std::size_t exact_recovery_budget,
    PbctopoComponentSearchDiagnostics *diagnostics,
    std::vector<PbctopoComponentSearchSurvivor> *survivors_out,
    std::size_t survivor_limit,
    const std::vector<PbctopoRequiredContact> *required_contacts,
    bool allow_adaptive_retry) {
  if (diagnostics != nullptr) {
    *diagnostics = {};
    diagnostics->attempted = true;
  }
  if (survivors_out != nullptr)
    survivors_out->clear();
  const std::size_t shift_shell_radius =
      component_shift_shell_radius(shifts);
  if (diagnostics != nullptr) {
    diagnostics->component_search_shell_radius = shift_shell_radius;
    diagnostics->bounded_domain_only = true;
  }
  beam_width = std::max<std::size_t>(
      1, std::min(beam_width, static_cast<std::size_t>(kMaxBeamWidth)));
  PbctopoComponentCandidate best;
  if (components.empty() || !callbacks.ready())
    return best;
  if (timing != nullptr) {
    timing->component_search_shift_domain_max =
        std::max(timing->component_search_shift_domain_max, shifts.size());
  }

  std::vector<Vec3> placed_points(atoms.size());
  best_points_out.clear();
  best_points_out.reserve(atoms.size());

  const double clash_search_cutoff =
      pbctopo_max_clash_cutoff(atoms, cutoffs.clash_cutoff);

  const std::size_t identity_shift_pos = identity_shift_index(shifts);

  if (incumbent_cut != nullptr && incumbent_cut->valid &&
      !incumbent_cut->ultra_clash && incumbent_points != nullptr &&
      incumbent_points->size() == atoms.size()) {
    best = callbacks.build_seed(*incumbent_cut, components, *incumbent_points);
    if (timing != nullptr)
      ++timing->component_search_seed_graph;
  } else if (timing != nullptr) {
    ++timing->component_search_seed_none;
  }

  bool hard_constraints_feasible = true;
  const PbctopoComponentPairShiftTable pair_table =
      build_component_pair_shift_table(
          atoms, components, shifts, Box, cutoffs.clash_cutoff,
          clash_search_cutoff, cutoffs.pair_contact_cutoff, timing,
          required_contacts, &hard_constraints_feasible);
  if (!hard_constraints_feasible)
    return {};
  const std::vector<std::size_t> atom_component_index =
      build_search_atom_component_index(atoms.size(), components);
  const std::vector<std::size_t> search_order = build_component_search_order(
      components, anchor_indices, atom_component_index, pair_table,
      identity_shift_pos);
  const PbctopoSearchContinuityPrefix continuity_prefix =
      build_search_continuity_prefix(continuity_pairs, atom_component_index,
                                     search_order, components.size());
  const std::vector<Int3> normalized_warm_offsets =
      (warm_component_offsets != nullptr &&
       warm_component_offsets->size() == components.size())
          ? normalize_search_warm_offsets(*warm_component_offsets, Box)
          : std::vector<Int3>{};
  const std::vector<Int3> *normalized_warm_offsets_ptr =
      normalized_warm_offsets.empty() ? nullptr : &normalized_warm_offsets;

  const PbctopoShiftMask full_shift_mask = PbctopoShiftMask::full(shifts.size());
  std::vector<PbctopoShiftMask> initial_allowed_masks(components.size(),
                                                      full_shift_mask);
  std::vector<std::size_t> initial_selected_shift(components.size(),
                                                  shifts.size());
  if (!search_order.empty() && identity_shift_pos < shifts.size()) {
    const std::size_t root_comp_idx = search_order.front();
    initial_allowed_masks[root_comp_idx] =
        PbctopoShiftMask::only(shifts.size(), identity_shift_pos);
    initial_selected_shift[root_comp_idx] = identity_shift_pos;
    for (std::size_t depth = 1; depth < search_order.size(); ++depth) {
      const std::size_t future_comp_idx = search_order[depth];
      initial_allowed_masks[future_comp_idx].intersect_with(
          pair_table.compat_mask(root_comp_idx, identity_shift_pos,
                                 future_comp_idx));
      if (initial_allowed_masks[future_comp_idx].empty() && timing != nullptr) {
        ++timing->component_pair_table_prunes;
        ++timing->component_search_zero_future_mask_prunes;
      }
    }
  }
  if (!enforce_search_arc_consistency(pair_table, initial_allowed_masks,
                                      timing)) {
    return best;
  }
  const std::size_t filtered_state_upper_bound =
      component_domain_product_upper_bound(
          initial_allowed_masks,
          exact_recovery_budget > 0 ? exact_recovery_budget + 1
                                    : kPbctopoExactAssignmentBudget + 1);
  if (exact_recovery_budget > 0 &&
      filtered_state_upper_bound <= exact_recovery_budget) {
    if (timing != nullptr)
      ++timing->component_search_exact_recoveries;
    return search_component_shift_candidate(
        atoms, Box, components, shifts, continuity_pairs, anchor_indices,
        cutoffs, best_points_out, callbacks, warm_component_offsets,
        incumbent_cut, incumbent_points, timing, anchor_backend,
        anchor_context, diagnostics, survivors_out, survivor_limit,
        required_contacts);
  }
  if (diagnostics != nullptr)
    diagnostics->beam_used = true;

  std::vector<PbctopoBeamState> beam_states;
  PbctopoBeamState root_state;
  root_state.selected_shift = initial_selected_shift;
  root_state.allowed_masks = initial_allowed_masks;
  root_state.future_remaining_shifts = 0;
  const double continuity_cutoff2 =
      cutoffs.continuity_cutoff * cutoffs.continuity_cutoff;
  if (!search_order.empty() &&
      !consume_search_continuity_prefix_from_assignment(
          atoms, atom_component_index, shifts, root_state.selected_shift,
          continuity_pairs, continuity_prefix.pairs_closing_at_depth[0],
          continuity_cutoff2, best.valid ? &best : nullptr,
          root_state.continuity_path_prefix,
          root_state.continuity_max_d2_prefix, timing)) {
    return best;
  }
  beam_states.push_back(std::move(root_state));
  if (timing != nullptr)
    ++timing->component_search_depth_visits[0];

  for (std::size_t depth = search_order.empty() ? 0 : 1;
       depth < search_order.size(); ++depth) {
    std::vector<PbctopoProvisionalBeamChild> provisional_children;
    provisional_children.reserve(
        beam_states.size() * std::min<std::size_t>(beam_width, shifts.size()));
    const std::size_t actual_comp_idx = search_order[depth];
    for (std::size_t state_idx = 0; state_idx < beam_states.size(); ++state_idx) {
      const auto &state = beam_states[state_idx];
      if (timing != nullptr &&
          depth < timing->component_search_depth_visits.size()) {
        ++timing->component_search_depth_visits[depth];
      }
      const std::vector<PbctopoComponentSearchShiftOption> ordered_shifts =
          build_component_search_shift_order(
              shifts, actual_comp_idx, depth, search_order, state.selected_shift,
              state.allowed_masks, pair_table, state.min_pair_d2,
              state.clash_atom_a, state.clash_atom_b,
              normalized_warm_offsets_ptr, timing);
      for (const auto &option : ordered_shifts) {
        if (timing != nullptr) {
          ++timing->component_shift_trials;
          ++timing->component_large_mode_beam_expanded;
        }
        PbctopoProvisionalBeamChild child;
        child.parent_state_idx = state_idx;
        child.shift_idx = option.shift_idx;
        child.shift_norm2 = state.shift_norm2 +
                            pbctopo_translation_norm2(shifts[option.shift_idx]);
        child.min_pair_d2 = option.next_min_pair_d2;
        child.clash_atom_a = option.clash_atom_a;
        child.clash_atom_b = option.clash_atom_b;
        child.future_remaining_shifts = option.future_remaining_shifts;
        child.warm_hits = state.warm_hits + (option.warm_priority ? 1u : 0u);
        child.continuity_path_prefix = state.continuity_path_prefix;
        child.continuity_max_d2_prefix = state.continuity_max_d2_prefix;
        if (depth < continuity_prefix.pairs_closing_at_depth.size() &&
            !consume_search_continuity_prefix_from_assignment(
                atoms, atom_component_index, shifts, state.selected_shift,
                continuity_pairs,
                continuity_prefix.pairs_closing_at_depth[depth],
                continuity_cutoff2, best.valid ? &best : nullptr,
                child.continuity_path_prefix,
                child.continuity_max_d2_prefix, timing, actual_comp_idx,
                option.shift_idx)) {
          continue;
        }
        provisional_children.push_back(std::move(child));
      }
    }
    if (provisional_children.empty()) {
      beam_states.clear();
      break;
    }
    if (timing != nullptr)
      timing->component_search_beam_provisional_children +=
          provisional_children.size();
    const std::size_t provisional_child_count = provisional_children.size();
    const auto better_child = [&](const PbctopoProvisionalBeamChild &lhs,
                                  const PbctopoProvisionalBeamChild &rhs) {
      return better_provisional_beam_child(lhs, rhs, beam_states,
                                           actual_comp_idx);
    };
    if (provisional_children.size() > beam_width) {
      std::partial_sort(
          provisional_children.begin(),
          provisional_children.begin() +
              static_cast<std::ptrdiff_t>(beam_width),
          provisional_children.end(), better_child);
      provisional_children.resize(beam_width);
    } else {
      std::sort(provisional_children.begin(), provisional_children.end(),
                better_child);
    }

    std::vector<PbctopoBeamState> next_states;
    next_states.reserve(provisional_children.size());
    std::vector<std::size_t> remaining_parent_uses(beam_states.size(), 0);
    for (const auto &child : provisional_children)
      ++remaining_parent_uses[child.parent_state_idx];
    std::size_t parent_state_moves = 0;
    for (const auto &child : provisional_children) {
      const std::size_t parent_idx = child.parent_state_idx;
      const bool last_parent_use = --remaining_parent_uses[parent_idx] == 0;
      PbctopoBeamState next_state;
      if (last_parent_use) {
        next_state = std::move(beam_states[parent_idx]);
        ++parent_state_moves;
      } else {
        next_state = beam_states[parent_idx];
      }
      next_state.selected_shift[actual_comp_idx] = child.shift_idx;
      if (actual_comp_idx < next_state.allowed_masks.size() &&
          child.shift_idx < shifts.size()) {
        next_state.allowed_masks[actual_comp_idx] =
            PbctopoShiftMask::only(shifts.size(), child.shift_idx);
      }
      for (std::size_t future_depth = depth + 1;
           future_depth < search_order.size(); ++future_depth) {
        const std::size_t future_comp_idx = search_order[future_depth];
        next_state.allowed_masks[future_comp_idx].intersect_with(
            pair_table.compat_mask(actual_comp_idx, child.shift_idx,
                                   future_comp_idx));
      }
      next_state.shift_norm2 = child.shift_norm2;
      next_state.min_pair_d2 = child.min_pair_d2;
      next_state.continuity_path_prefix = child.continuity_path_prefix;
      next_state.continuity_max_d2_prefix = child.continuity_max_d2_prefix;
      next_state.clash_atom_a = child.clash_atom_a;
      next_state.clash_atom_b = child.clash_atom_b;
      next_state.future_remaining_shifts = child.future_remaining_shifts;
      next_state.warm_hits = child.warm_hits;
      next_states.push_back(std::move(next_state));
    }
    if (timing != nullptr) {
      timing->component_large_mode_beam_kept += next_states.size();
      timing->component_search_beam_state_materializations += next_states.size();
      timing->component_search_beam_state_copies_avoided +=
          provisional_child_count > next_states.size()
              ? provisional_child_count - next_states.size()
              : 0;
      timing->component_search_beam_state_copies_avoided += parent_state_moves;
      timing->component_search_beam_parent_state_moves += parent_state_moves;
    }
    beam_states = std::move(next_states);
  }

  if (timing != nullptr)
    timing->component_large_mode_beam_complete_assignments += beam_states.size();
  const std::size_t exact_rescore_count = beam_states.size();
  if (diagnostics != nullptr) {
    diagnostics->beam_complete_assignments = beam_states.size();
    diagnostics->beam_survivors_exactly_rescored = exact_rescore_count;
  }
  std::sort(beam_states.begin(), beam_states.end(),
            better_component_beam_state);
  if (timing != nullptr) {
    timing->component_large_mode_beam_final_rescored += exact_rescore_count;
    ++timing->component_large_mode_beam_final_rescore_all_frames;
  }
  for (std::size_t state_idx = 0; state_idx < exact_rescore_count;
       ++state_idx) {
    double candidate_shift_norm2 = 0.0;
    if (!build_component_points_from_selected_shifts(
            atoms, components, shifts,
            beam_states[state_idx].selected_shift, placed_points,
            candidate_shift_norm2)) {
      continue;
    }
    PbctopoComponentCandidate candidate = callbacks.score_assignment(
        atoms, placed_points, components, beam_states[state_idx].min_pair_d2,
        beam_states[state_idx].clash_atom_a,
        beam_states[state_idx].clash_atom_b, continuity_pairs, anchor_indices,
        cutoffs.continuity_cutoff, candidate_shift_norm2,
        best.valid ? &best : nullptr, timing, anchor_backend, anchor_context);
    annotate_search_domain_witness(candidate,
                                   beam_states[state_idx].selected_shift,
                                   shifts, shift_shell_radius);
    record_component_search_survivor(
        candidate, beam_states[state_idx].selected_shift, shifts, callbacks,
        survivors_out, survivor_limit);
    if (callbacks.better_candidate(candidate, best)) {
      best = candidate;
      best_points_out = placed_points;
    }
  }

  if (!best.valid && allow_adaptive_retry && beam_width < kMaxBeamWidth) {
    const std::size_t expanded_width =
        std::min<std::size_t>(kMaxBeamWidth, beam_width * 2);
    if (timing != nullptr) {
      ++timing->component_search_beam_adaptive_retries;
      timing->component_search_beam_adaptive_max_width =
          std::max(timing->component_search_beam_adaptive_max_width,
                   expanded_width);
    }
    PbctopoComponentCandidate retry = search_component_shift_candidate_beam(
        atoms, Box, components, shifts, continuity_pairs, anchor_indices,
        cutoffs, best_points_out, expanded_width, callbacks,
        warm_component_offsets, incumbent_cut, incumbent_points, timing,
        anchor_backend, anchor_context, exact_recovery_budget, diagnostics,
        survivors_out, survivor_limit, required_contacts, false);
    if (retry.valid && timing != nullptr)
      ++timing->component_search_beam_adaptive_retry_successes;
    return retry;
  }
  if (diagnostics != nullptr && best.search_domain_audit_known) {
    diagnostics->selected_assignment_boundary_known = true;
    diagnostics->selected_assignment_touches_shell_boundary =
        best.selected_assignment_touches_shell_boundary;
  }
  return best;
}

PbctopoComponentCandidate search_component_shift_candidate(
    const std::vector<PbctopoAtom> &atoms, box Box,
    const std::vector<std::vector<std::size_t>> &components,
    const std::vector<PbctopoShift> &shifts,
    const std::vector<PbctopoContinuityPair> &continuity_pairs,
    const std::vector<std::size_t> &anchor_indices,
    const RuntimeCutoffs &cutoffs, std::vector<Vec3> &best_points_out,
    const PbctopoComponentSearchCallbacks &callbacks,
    const std::vector<Int3> *warm_component_offsets,
    const PbctopoCutCandidate *incumbent_cut,
    const std::vector<Vec3> *incumbent_points, PbctopoTimingStats *timing,
    AnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context,
    PbctopoComponentSearchDiagnostics *diagnostics,
    std::vector<PbctopoComponentSearchSurvivor> *survivors_out,
    std::size_t survivor_limit,
    const std::vector<PbctopoRequiredContact> *required_contacts,
    const PbctopoComponentSearchLimits *limits) {
  using search_clock = std::chrono::steady_clock;
  const auto search_started = search_clock::now();
  if (diagnostics != nullptr) {
    *diagnostics = {};
    diagnostics->attempted = true;
  }
  if (survivors_out != nullptr)
    survivors_out->clear();
  PbctopoComponentCandidate best;
  if (components.empty() || !callbacks.ready())
    return best;
  bool incumbent_bound_pruning_used = false;
  bool search_aborted = false;
  bool bounded_domain_infeasible_proven = false;
  const bool enumerate_feasible_assignments =
      limits != nullptr && limits->enumerate_feasible_assignments;
  const bool hard_certificate_domain_available =
      callbacks.input_atoms_cover_certificate_scope &&
      callbacks.hard_certificate_domain_complete;
  std::size_t feasible_assignments_within_domain = 0;
  std::size_t soft_score_valid_assignments_within_domain = 0;
  const std::size_t shift_shell_radius =
      component_shift_shell_radius(shifts);
  PbctopoSearchStopReason stop_reason = PbctopoSearchStopReason::None;
  std::size_t states_visited = 0;
  std::size_t memory_bytes_estimate = 0;
  auto elapsed_seconds = [&]() {
    return std::chrono::duration<double>(search_clock::now() - search_started)
        .count();
  };
  auto stop_search = [&](PbctopoSearchStopReason reason) {
    if (!search_aborted) {
      search_aborted = true;
      stop_reason = reason;
    }
  };
  auto complete_exact_diagnostics = [&]() {
    const bool certificate_search_exhaustive =
        !search_aborted && hard_certificate_domain_available;
    best.search_domain_evidence_complete = certificate_search_exhaustive;
    if (survivors_out != nullptr) {
      for (auto &survivor : *survivors_out) {
        survivor.candidate.search_domain_evidence_complete =
            certificate_search_exhaustive;
      }
    }
    if (diagnostics == nullptr)
      return;
    diagnostics->search_completed_under_admissible_bounds = !search_aborted;
    diagnostics->incumbent_bound_pruning_used =
        incumbent_bound_pruning_used;
    diagnostics->best_objective_proven = !search_aborted && best.valid;
    // Objective bounds are strict under the comparator, so tied optima remain
    // reachable even though non-optimal feasible assignments may be pruned.
    diagnostics->equivalent_optima_complete =
        !search_aborted && best.valid;
    diagnostics->feasible_assignment_set_enumerated =
        !search_aborted && enumerate_feasible_assignments &&
        hard_certificate_domain_available;
    diagnostics->bounded_domain_feasibility_known =
        !search_aborted && hard_certificate_domain_available;
    diagnostics->bounded_domain_feasible =
        !search_aborted && !bounded_domain_infeasible_proven &&
        feasible_assignments_within_domain > 0;
    diagnostics->feasible_assignments_within_domain =
        diagnostics->feasible_assignment_set_enumerated
            ? feasible_assignments_within_domain
            : 0;
    diagnostics->soft_score_valid_assignments_within_domain =
        diagnostics->feasible_assignment_set_enumerated
            ? soft_score_valid_assignments_within_domain
            : 0;
    diagnostics->resource_limit_hit = search_aborted;
    diagnostics->stop_reason = stop_reason;
    diagnostics->states_visited = states_visited;
    diagnostics->memory_bytes_estimate = memory_bytes_estimate;
    diagnostics->wall_seconds = elapsed_seconds();
    diagnostics->offset_enumeration_exhaustive =
        diagnostics->feasible_assignment_set_enumerated;
    diagnostics->exhaustive = diagnostics->offset_enumeration_exhaustive;
    diagnostics->all_atom_feasibility_exhaustive =
        !search_aborted && callbacks.input_atoms_cover_certificate_scope;
    diagnostics->certificate_search_exhaustive =
        certificate_search_exhaustive;
    diagnostics->component_search_shell_radius = shift_shell_radius;
    diagnostics->bounded_domain_only = true;
    diagnostics->selected_assignment_boundary_known =
        best.search_domain_audit_known;
    diagnostics->selected_assignment_touches_shell_boundary =
        best.search_domain_audit_known &&
        best.selected_assignment_touches_shell_boundary;
    diagnostics->optimality_gap_known =
        diagnostics->best_objective_proven;
    diagnostics->optimality_gap =
        diagnostics->optimality_gap_known
            ? 0.0
            : std::numeric_limits<double>::quiet_NaN();
  };
  if (timing != nullptr) {
    timing->component_search_shift_domain_max =
        std::max(timing->component_search_shift_domain_max, shifts.size());
  }

  std::size_t non_pair_memory_estimate = 0;
  const auto add_non_pair_memory = [&](std::size_t bytes) {
    non_pair_memory_estimate =
        saturating_size_add(non_pair_memory_estimate, bytes);
  };
  add_non_pair_memory(
      saturating_size_multiply(atoms.size(), 2 * sizeof(Vec3)));
  add_non_pair_memory(saturating_size_multiply(
      components.size(), 3 * sizeof(std::size_t) + sizeof(Int3)));
  memory_bytes_estimate = non_pair_memory_estimate;
  if (limits != nullptr && limits->max_memory_bytes > 0 &&
      memory_bytes_estimate > limits->max_memory_bytes) {
    stop_search(PbctopoSearchStopReason::MemoryLimit);
    complete_exact_diagnostics();
    return best;
  }

  std::vector<Vec3> placed_points(atoms.size());
  best_points_out.clear();
  best_points_out.reserve(atoms.size());
  const double clash_search_cutoff =
      pbctopo_max_clash_cutoff(atoms, cutoffs.clash_cutoff);
  std::vector<std::size_t> selected_shift(components.size(), shifts.size());

  const std::size_t identity_shift_pos = identity_shift_index(shifts);

  for (const auto &component : components) {
    for (std::size_t atom_idx : component) {
      placed_points[atom_idx] = {atoms[atom_idx].x, atoms[atom_idx].y,
                                 atoms[atom_idx].z};
    }
  }

  // Exact-search proof bounds must come from leaves enumerated in this bounded
  // shift domain. The graph incumbent has no domain-membership witness here.
  (void)incumbent_cut;
  (void)incumbent_points;
  if (timing != nullptr) {
    ++timing->component_search_seed_none;
  }

  bool hard_constraints_feasible = true;
  const PbctopoComponentPairShiftTable pair_table =
      build_component_pair_shift_table(
          atoms, components, shifts, Box, cutoffs.clash_cutoff,
          clash_search_cutoff, cutoffs.pair_contact_cutoff, timing,
          required_contacts, &hard_constraints_feasible);
  memory_bytes_estimate = saturating_size_add(
      non_pair_memory_estimate, pair_table.peak_bytes_estimate());
  if (limits != nullptr && limits->max_memory_bytes > 0 &&
      memory_bytes_estimate > limits->max_memory_bytes) {
    stop_search(PbctopoSearchStopReason::MemoryLimit);
  }
  if (!search_aborted && limits != nullptr &&
      limits->max_wall_seconds > 0.0 &&
      elapsed_seconds() > limits->max_wall_seconds) {
    stop_search(PbctopoSearchStopReason::WallTimeLimit);
  }
  if (search_aborted) {
    complete_exact_diagnostics();
    return best;
  }
  if (!hard_constraints_feasible) {
    bounded_domain_infeasible_proven = true;
    complete_exact_diagnostics();
    return {};
  }
  const std::vector<std::size_t> atom_component_index =
      build_search_atom_component_index(atoms.size(), components);
  const std::vector<std::size_t> search_order = build_component_search_order(
      components, anchor_indices, atom_component_index, pair_table,
      identity_shift_pos);
  const PbctopoSearchContinuityPrefix continuity_prefix =
      build_search_continuity_prefix(continuity_pairs, atom_component_index,
                                     search_order, components.size());
  const std::vector<Int3> normalized_warm_offsets =
      (warm_component_offsets != nullptr &&
       warm_component_offsets->size() == components.size())
          ? normalize_search_warm_offsets(*warm_component_offsets, Box)
          : std::vector<Int3>{};
  const std::vector<Int3> *normalized_warm_offsets_ptr =
      normalized_warm_offsets.empty() ? nullptr : &normalized_warm_offsets;

  const PbctopoShiftMask full_shift_mask = PbctopoShiftMask::full(shifts.size());
  std::vector<PbctopoShiftMask> initial_allowed_masks(components.size(),
                                                      full_shift_mask);
  if (!search_order.empty() && identity_shift_pos < shifts.size()) {
    const std::size_t root_comp_idx = search_order.front();
    initial_allowed_masks[root_comp_idx] =
        PbctopoShiftMask::only(shifts.size(), identity_shift_pos);
    selected_shift[root_comp_idx] = identity_shift_pos;
    for (std::size_t depth = 1; depth < search_order.size(); ++depth) {
      const std::size_t future_comp_idx = search_order[depth];
      initial_allowed_masks[future_comp_idx].intersect_with(
          pair_table.compat_mask(root_comp_idx, identity_shift_pos,
                                 future_comp_idx));
      if (initial_allowed_masks[future_comp_idx].empty() && timing != nullptr) {
        ++timing->component_pair_table_prunes;
        ++timing->component_search_zero_future_mask_prunes;
      }
    }
  }
  if (!enforce_search_arc_consistency(pair_table, initial_allowed_masks,
                                      timing)) {
    bounded_domain_infeasible_proven = true;
    complete_exact_diagnostics();
    return best;
  }

  add_non_pair_memory(saturating_size_multiply(
      continuity_prefix.pairs_closing_at_depth.capacity(),
      sizeof(std::vector<std::size_t>)));
  for (const auto &pair_indices : continuity_prefix.pairs_closing_at_depth) {
    add_non_pair_memory(saturating_size_multiply(
        pair_indices.capacity(), sizeof(std::size_t)));
  }
  add_non_pair_memory(saturating_size_multiply(
      initial_allowed_masks.capacity(), sizeof(PbctopoShiftMask)));
  const std::size_t full_mask_extra_bytes =
      full_shift_mask.storage_bytes_estimate() > sizeof(PbctopoShiftMask)
          ? full_shift_mask.storage_bytes_estimate() -
                sizeof(PbctopoShiftMask)
          : 0;
  add_non_pair_memory(full_mask_extra_bytes);
  for (const auto &mask : initial_allowed_masks) {
    const std::size_t mask_bytes = mask.storage_bytes_estimate();
    if (mask_bytes > sizeof(PbctopoShiftMask))
      add_non_pair_memory(mask_bytes - sizeof(PbctopoShiftMask));
  }
  const std::size_t component_count = components.size();
  const std::size_t trail_capacity_twice =
      component_count == std::numeric_limits<std::size_t>::max()
          ? std::numeric_limits<std::size_t>::max()
          : saturating_size_multiply(component_count, component_count + 1);
  if (trail_capacity_twice == std::numeric_limits<std::size_t>::max()) {
    stop_search(PbctopoSearchStopReason::MemoryLimit);
    complete_exact_diagnostics();
    return best;
  }
  const std::size_t trail_capacity = trail_capacity_twice / 2;
  using AllowedMaskRestore =
      std::pair<std::size_t, PbctopoShiftMask>;
  add_non_pair_memory(saturating_size_multiply(
      trail_capacity,
      saturating_size_add(sizeof(AllowedMaskRestore),
                          full_mask_extra_bytes)));
  add_non_pair_memory(saturating_size_multiply(
      saturating_size_multiply(component_count, shifts.size()),
      sizeof(PbctopoComponentSearchShiftOption)));
  if (survivors_out != nullptr && survivor_limit > 0) {
    add_non_pair_memory(saturating_size_multiply(
        survivor_limit, sizeof(PbctopoComponentSearchSurvivor)));
    add_non_pair_memory(saturating_size_multiply(
        saturating_size_multiply(survivor_limit, component_count),
        sizeof(Int3)));
  }
  memory_bytes_estimate = saturating_size_add(
      non_pair_memory_estimate, pair_table.peak_bytes_estimate());
  if (limits != nullptr && limits->max_memory_bytes > 0 &&
      memory_bytes_estimate > limits->max_memory_bytes) {
    stop_search(PbctopoSearchStopReason::MemoryLimit);
    complete_exact_diagnostics();
    return best;
  }
  std::vector<AllowedMaskRestore> allowed_mask_trail;
  allowed_mask_trail.reserve(trail_capacity);

  double initial_continuity_path = 0.0;
  double initial_continuity_max_d2 = 0.0;
  const double continuity_cutoff2 =
      cutoffs.continuity_cutoff * cutoffs.continuity_cutoff;
  if (!search_order.empty() &&
      !consume_search_continuity_prefix(
          placed_points, continuity_pairs,
          continuity_prefix.pairs_closing_at_depth[0], continuity_cutoff2,
          !enumerate_feasible_assignments && best.valid ? &best : nullptr,
          initial_continuity_path,
          initial_continuity_max_d2, timing,
          &incumbent_bound_pruning_used)) {
    if (!incumbent_bound_pruning_used)
      bounded_domain_infeasible_proven = true;
    complete_exact_diagnostics();
    return best;
  }

  std::function<void(std::size_t, double, double, double, double, std::size_t,
                     std::size_t, std::vector<PbctopoShiftMask> &)>
      dfs = [&](std::size_t depth, double shift_norm2,
                double continuity_path_prefix,
                double continuity_max_d2_prefix, double min_pair_d2,
                std::size_t clash_atom_a, std::size_t clash_atom_b,
                std::vector<PbctopoShiftMask> &allowed_masks) {
        if (search_aborted)
          return;
        if (limits != nullptr && limits->max_states > 0 &&
            states_visited >= limits->max_states) {
          stop_search(PbctopoSearchStopReason::StateLimit);
          return;
        }
        ++states_visited;
        if (limits != nullptr && limits->max_wall_seconds > 0.0 &&
            elapsed_seconds() > limits->max_wall_seconds) {
          stop_search(PbctopoSearchStopReason::WallTimeLimit);
          return;
        }
        if (limits != nullptr && limits->max_memory_bytes > 0 &&
            (states_visited == 1 || (states_visited & 63u) == 0u)) {
          memory_bytes_estimate = saturating_size_add(
              non_pair_memory_estimate, pair_table.peak_bytes_estimate());
          if (memory_bytes_estimate > limits->max_memory_bytes) {
            stop_search(PbctopoSearchStopReason::MemoryLimit);
            return;
          }
        }
        if (timing != nullptr &&
            depth < timing->component_search_depth_visits.size()) {
          ++timing->component_search_depth_visits[depth];
        }
        if (depth >= search_order.size()) {
          PbctopoLeafEvaluation evaluation;
          evaluation.hard_domain_feasible =
              hard_certificate_domain_available;
          if (evaluation.hard_domain_feasible &&
              feasible_assignments_within_domain <
              std::numeric_limits<std::size_t>::max()) {
            ++feasible_assignments_within_domain;
          }
          evaluation.score = callbacks.score_assignment(
              atoms, placed_points, components, min_pair_d2, clash_atom_a,
              clash_atom_b, continuity_pairs, anchor_indices,
              cutoffs.continuity_cutoff,
              shift_norm2 / static_cast<double>(components.size()),
              !enumerate_feasible_assignments && best.valid ? &best : nullptr,
              timing, anchor_backend,
              anchor_context);
          PbctopoComponentCandidate &candidate = evaluation.score;
          evaluation.soft_score_valid =
              evaluation.hard_domain_feasible && candidate.valid &&
              !candidate.ultra_clash;
          if (evaluation.soft_score_valid &&
              soft_score_valid_assignments_within_domain <
                  std::numeric_limits<std::size_t>::max()) {
            ++soft_score_valid_assignments_within_domain;
          }
          annotate_search_domain_witness(candidate, selected_shift, shifts,
                                         shift_shell_radius);
          if (limits != nullptr && limits->max_wall_seconds > 0.0 &&
              elapsed_seconds() > limits->max_wall_seconds) {
            stop_search(PbctopoSearchStopReason::WallTimeLimit);
            return;
          }
          record_component_search_survivor(
              candidate, selected_shift, shifts, callbacks, survivors_out,
              survivor_limit);
          const bool replaces_best =
              callbacks.better_candidate(candidate, best);
          const bool equivalent_to_best =
              candidate.valid && best.valid && !replaces_best &&
              !callbacks.better_candidate(best, candidate);
          if (replaces_best) {
            best = candidate;
            best_points_out = placed_points;
            if (diagnostics != nullptr)
              diagnostics->equivalent_best_assignments_within_domain = 1;
          } else if (equivalent_to_best && diagnostics != nullptr) {
            ++diagnostics->equivalent_best_assignments_within_domain;
          }
          return;
        }

        const std::size_t actual_comp_idx = search_order[depth];
        const std::vector<PbctopoComponentSearchShiftOption> ordered_shifts =
            build_component_search_shift_order(
                shifts, actual_comp_idx, depth, search_order, selected_shift,
                allowed_masks, pair_table, min_pair_d2, clash_atom_a,
                clash_atom_b, normalized_warm_offsets_ptr, timing);
        for (const auto &option : ordered_shifts) {
          if (search_aborted)
            break;
          const std::size_t shift_idx = option.shift_idx;
          if (timing != nullptr)
            ++timing->component_shift_trials;
          const auto &component = components[actual_comp_idx];
          const auto &shift = shifts[shift_idx];
          for (std::size_t atom_idx : component) {
            if (atom_idx >= atoms.size())
              continue;
            placed_points[atom_idx] = {atoms[atom_idx].x + shift.tx,
                                       atoms[atom_idx].y + shift.ty,
                                       atoms[atom_idx].z + shift.tz};
          }
          selected_shift[actual_comp_idx] = shift_idx;
          const std::size_t trail_mark = allowed_mask_trail.size();
          if (actual_comp_idx < allowed_masks.size() &&
              shift_idx < shifts.size()) {
            allowed_mask_trail.push_back(
                {actual_comp_idx, allowed_masks[actual_comp_idx]});
            allowed_masks[actual_comp_idx] =
                PbctopoShiftMask::only(shifts.size(), shift_idx);
          }
          for (std::size_t future_depth = depth + 1;
               future_depth < search_order.size(); ++future_depth) {
            const std::size_t future_comp_idx = search_order[future_depth];
            PbctopoShiftMask updated_mask = allowed_masks[future_comp_idx];
            if (updated_mask.intersect_with(pair_table.compat_mask(
                    actual_comp_idx, shift_idx, future_comp_idx))) {
              allowed_mask_trail.push_back(
                  {future_comp_idx, allowed_masks[future_comp_idx]});
              allowed_masks[future_comp_idx] = std::move(updated_mask);
            }
          }
          double next_continuity_path = continuity_path_prefix;
          double next_continuity_max_d2 = continuity_max_d2_prefix;
          bool prefix_ok = true;
          if (depth < continuity_prefix.pairs_closing_at_depth.size()) {
            prefix_ok = consume_search_continuity_prefix(
                placed_points, continuity_pairs,
                continuity_prefix.pairs_closing_at_depth[depth],
                continuity_cutoff2,
                !enumerate_feasible_assignments && best.valid ? &best
                                                              : nullptr,
                next_continuity_path, next_continuity_max_d2, timing,
                &incumbent_bound_pruning_used);
          }
          if (prefix_ok) {
            dfs(depth + 1,
                shift_norm2 + pbctopo_translation_norm2(shift),
                next_continuity_path, next_continuity_max_d2,
                option.next_min_pair_d2, option.clash_atom_a,
                option.clash_atom_b, allowed_masks);
          }
          while (allowed_mask_trail.size() > trail_mark) {
            auto &restore = allowed_mask_trail.back();
            allowed_masks[restore.first] = std::move(restore.second);
            allowed_mask_trail.pop_back();
          }
          selected_shift[actual_comp_idx] = shifts.size();
        }
      };

  dfs(search_order.empty() ? 0 : 1, 0.0, initial_continuity_path,
      initial_continuity_max_d2, std::numeric_limits<double>::infinity(),
      std::numeric_limits<std::size_t>::max(),
      std::numeric_limits<std::size_t>::max(), initial_allowed_masks);
  memory_bytes_estimate = saturating_size_add(
      non_pair_memory_estimate, pair_table.peak_bytes_estimate());
  if (limits != nullptr && limits->max_memory_bytes > 0 &&
      memory_bytes_estimate > limits->max_memory_bytes)
    stop_search(PbctopoSearchStopReason::MemoryLimit);
  complete_exact_diagnostics();
  return best;
}

} // namespace titan_pbctopo

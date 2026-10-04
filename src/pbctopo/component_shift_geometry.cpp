#include "component_shift_geometry.h"

#include "component_metric.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace titan_pbctopo {

namespace {

PbctopoComponentBaseGeometry build_component_base_geometry_impl(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &component, bool heavy_only) {
  PbctopoComponentBaseGeometry base;
  base.x.reserve(component.size());
  base.y.reserve(component.size());
  base.z.reserve(component.size());
  base.atom_indices.reserve(component.size());
  auto append_atom = [&](std::size_t atom_idx) {
    const auto &atom = atoms[atom_idx];
    base.x.push_back(atom.x);
    base.y.push_back(atom.y);
    base.z.push_back(atom.z);
    base.atom_indices.push_back(atom_idx);
    base.center = base.center + Vec3{atom.x, atom.y, atom.z};
    base.min_x = std::min(base.min_x, atom.x);
    base.min_y = std::min(base.min_y, atom.y);
    base.min_z = std::min(base.min_z, atom.z);
    base.max_x = std::max(base.max_x, atom.x);
    base.max_y = std::max(base.max_y, atom.y);
    base.max_z = std::max(base.max_z, atom.z);
  };
  for (std::size_t atom_idx : component) {
    if (atom_idx < atoms.size() && (!heavy_only || atoms[atom_idx].is_heavy))
      append_atom(atom_idx);
  }
  if (heavy_only && base.x.empty()) {
    for (std::size_t atom_idx : component) {
      if (atom_idx < atoms.size())
        append_atom(atom_idx);
    }
  }
  if (!base.x.empty())
    base.center = base.center * (1.0 / static_cast<double>(base.x.size()));
  return base;
}

} // namespace

PbctopoComponentBaseGeometry pbctopo_build_component_base_geometry(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &component) {
  return build_component_base_geometry_impl(atoms, component, true);
}

PbctopoComponentBaseGeometry pbctopo_build_component_all_atom_base_geometry(
    const std::vector<PbctopoAtom> &atoms,
    const std::vector<std::size_t> &component) {
  return build_component_base_geometry_impl(atoms, component, false);
}

PbctopoShiftedComponentDescriptor pbctopo_describe_shifted_component(
    const PbctopoComponentBaseGeometry &base, const PbctopoShift &shift) {
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

void pbctopo_materialize_shifted_component(
    const PbctopoComponentBaseGeometry &base, const PbctopoShift &shift,
    double clash_cutoff, double contact_cutoff, bool build_clash_grid,
    bool build_contact_grid,
    PbctopoShiftedComponent &shifted) {
  const double inv_grid = 1.0 / clash_cutoff;
  const double inv_contact_grid =
      build_contact_grid ? 1.0 / contact_cutoff : 0.0;
  const auto descriptor = pbctopo_describe_shifted_component(base, shift);
  const std::size_t atom_count = base.x.size();
  shifted.points.resize(atom_count);
  shifted.atom_indices = base.atom_indices;
  shifted.clash_keys.resize(atom_count);
  if (build_contact_grid)
    shifted.contact_keys.resize(atom_count);
  else
    shifted.contact_keys.clear();
  shifted.clash_grid.clear();
  if (build_clash_grid)
    shifted.clash_grid.reserve(std::max<std::size_t>(32, atom_count));
  shifted.contact_grid.clear();
  if (build_contact_grid)
    shifted.contact_grid.reserve(std::max<std::size_t>(32, atom_count));
  shifted.translation_norm2 = descriptor.translation_norm2;
  shifted.spatial_keys_valid = true;
  shifted.center = descriptor.center;
  shifted.min_x = descriptor.min_x;
  shifted.min_y = descriptor.min_y;
  shifted.min_z = descriptor.min_z;
  shifted.max_x = descriptor.max_x;
  shifted.max_y = descriptor.max_y;
  shifted.max_z = descriptor.max_z;
  for (std::size_t local_idx = 0; local_idx < atom_count; ++local_idx) {
    const Vec3 point{base.x[local_idx] + shift.tx,
                     base.y[local_idx] + shift.ty,
                     base.z[local_idx] + shift.tz};
    shifted.points[local_idx] = point;
    ClashCellKey clash_key;
    if (!pbctopo_make_cell_key(point.x, point.y, point.z, inv_grid,
                                clash_key)) {
      shifted.spatial_keys_valid = false;
      shifted.clash_grid.clear();
      shifted.contact_grid.clear();
      return;
    }
    shifted.clash_keys[local_idx] = clash_key;
    if (build_clash_grid)
      shifted.clash_grid[clash_key].push_back(local_idx);
    if (build_contact_grid) {
      ClashCellKey contact_key;
      if (!pbctopo_make_cell_key(point.x, point.y, point.z,
                                  inv_contact_grid, contact_key)) {
        shifted.spatial_keys_valid = false;
        shifted.clash_grid.clear();
        shifted.contact_grid.clear();
        return;
      }
      shifted.contact_keys[local_idx] = contact_key;
      shifted.contact_grid[contact_key].push_back(local_idx);
    }
  }
}

double pbctopo_shifted_bbox_distance2(
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

} // namespace titan_pbctopo

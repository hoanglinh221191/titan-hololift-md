#include "temporal_replay.h"

#include "lattice_math.h"

#include <unordered_map>
#include <unordered_set>

namespace titan_pbctopo {

bool replay_pbctopo_assignment_delta(
    std::span<const Vec3> applied_points,
    std::span<const std::uint32_t> atom_component_index,
    const PbctopoImageAssignment &applied_assignment,
    const PbctopoImageAssignment &selected_assignment, box Box,
    std::vector<Vec3> &replayed_points) {
  replayed_points.clear();
  if (!applied_assignment.valid || !selected_assignment.valid ||
      applied_points.empty() ||
      applied_points.size() != atom_component_index.size() ||
      applied_assignment.component_offsets.size() !=
          selected_assignment.component_offsets.size() ||
      applied_assignment.component_ids.size() !=
          applied_assignment.component_offsets.size() ||
      selected_assignment.component_ids.size() !=
          selected_assignment.component_offsets.size() ||
      applied_assignment.layout_signature == 0 ||
      applied_assignment.layout_signature !=
          selected_assignment.layout_signature) {
    return false;
  }

  const std::size_t component_count =
      applied_assignment.component_offsets.size();
  std::unordered_map<std::uint64_t, std::size_t> selected_component_index;
  selected_component_index.reserve(component_count);
  for (std::size_t comp_idx = 0; comp_idx < component_count; ++comp_idx) {
    if (!selected_component_index
             .emplace(selected_assignment.component_ids[comp_idx], comp_idx)
             .second) {
      return false;
    }
  }
  std::unordered_set<std::uint64_t> applied_component_ids;
  applied_component_ids.reserve(component_count);
  std::vector<Vec3> translations(component_count);
  const PbctopoLatticeMetric lattice(Box);
  for (std::size_t comp_idx = 0; comp_idx < component_count; ++comp_idx) {
    const std::uint64_t component_id =
        applied_assignment.component_ids[comp_idx];
    if (!applied_component_ids.insert(component_id).second)
      return false;
    const auto selected_it = selected_component_index.find(component_id);
    if (selected_it == selected_component_index.end())
      return false;
    Int3 delta{};
    if (!checked_int3_subtract(
            selected_assignment.component_offsets[selected_it->second],
            applied_assignment.component_offsets[comp_idx], delta)) {
      return false;
    }
    const auto cart = lattice.scaled_to_cartesian(
        static_cast<double>(delta.x), static_cast<double>(delta.y),
        static_cast<double>(delta.z));
    translations[comp_idx] = {cart[0], cart[1], cart[2]};
  }

  replayed_points.resize(applied_points.size());
  for (std::size_t atom_idx = 0; atom_idx < applied_points.size(); ++atom_idx) {
    const std::size_t comp_idx = atom_component_index[atom_idx];
    if (comp_idx >= translations.size()) {
      replayed_points.clear();
      return false;
    }
    replayed_points[atom_idx] = {
        applied_points[atom_idx].x + translations[comp_idx].x,
        applied_points[atom_idx].y + translations[comp_idx].y,
        applied_points[atom_idx].z + translations[comp_idx].z};
  }
  return true;
}

} // namespace titan_pbctopo

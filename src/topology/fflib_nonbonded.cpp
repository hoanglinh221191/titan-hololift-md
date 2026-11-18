// Atom-type lookup in the nonbonded table and the ENE-08 table check.
#include "topology/fflib_assigner.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace TitanFfLib {

bool fill_vdw_from_atom_type(Multi_MOL_parameters &parameters,
                             std::size_t atom_index, const para &parameter,
                             std::string_view atom_type) {
  if (atom_index >= parameters.size())
    return false;

  // ENE-08: the registered definition (last one read, so topology atom types
  // override lib_ff) replaces the former backward scan over the whole table,
  // which is the same choice at O(1) instead of O(table) string compares.
  const auto it = parameter.atom_type_to_id.find(std::string(atom_type));
  if (it != parameter.atom_type_to_id.end() && it->second < parameter.base.size()) {
    const auto &base = parameter.base[it->second];
    parameters[atom_index].set_epsilon_sigma_radius(base.c12_epsilon, base.c6_sigma,
                                                    base.radius);
    parameters[atom_index].set_type_id(static_cast<int>(it->second));
    return true;
  }
  parameters[atom_index].set_type_id(-1);
  return false;
}

NonbondedTableCheck validate_nonbonded_table(const para &parameter) {
  NonbondedTableCheck out;
  out.entries = parameter.base.size();
  std::unordered_map<std::string, std::vector<std::size_t>> by_name;
  by_name.reserve(parameter.base.size());
  for (std::size_t i = 0; i < parameter.base.size(); ++i) {
    if (!parameter.base[i].atom_type.empty())
      by_name[parameter.base[i].atom_type].push_back(i);
  }
  std::vector<std::string> names;
  for (const auto &[name, indices] : by_name) {
    if (indices.size() > 1)
      names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  for (const auto &name : names) {
    const auto &indices = by_name[name];
    ++out.duplicate_types;
    const auto registered = parameter.atom_type_to_id.find(name);
    const std::size_t active =
        (registered != parameter.atom_type_to_id.end() &&
         registered->second < parameter.base.size())
            ? registered->second
            : indices.back();
    const auto &chosen = parameter.base[active];
    const bool active_zero = chosen.c6_sigma == 0.0 && chosen.c12_epsilon == 0.0;
    bool other_nonzero = false;
    for (const std::size_t j : indices) {
      if (j != active && (parameter.base[j].c6_sigma != 0.0 ||
                          parameter.base[j].c12_epsilon != 0.0))
        other_nonzero = true;
    }
    if (active_zero && other_nonzero) {
      ++out.shadowed_types;
      if (out.shadowed_examples.size() < 12)
        out.shadowed_examples.push_back(name);
    }
  }
  return out;
}

} // namespace TitanFfLib

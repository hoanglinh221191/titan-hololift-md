#include "titan_contact_geometry.h"

#include <algorithm>
#include <unordered_map>

namespace titan_contact_geometry {

bool ContactGeometryCache::build(const Multi_MOL &mol, std::uint8_t molecule,
                                 titan_contact::ContactMode mode) {
  mode_ = mode;
  selections_.clear();
  residues_.clear();

  const std::size_t atom_count = mol.info.size();
  if (mode_ == titan_contact::ContactMode::SidechainCOM &&
      mol.parameters.size() < atom_count)
    return false;

  std::unordered_map<std::size_t, std::size_t> chain_ordinals;
  for (std::size_t atom_idx = 0; atom_idx < atom_count; ++atom_idx) {
    titan_contact::ResidueKey key =
        titan_contact::residue_key_from_atom(mol.info[atom_idx], 0, molecule);
    auto found = std::find_if(selections_.begin(), selections_.end(),
                              [&](const ResidueSelection &item) {
                                return item.residue == key;
                              });
    if (found == selections_.end()) {
      key.ordinal = ++chain_ordinals[key.chainnum];
      selections_.push_back(ResidueSelection{key, {}, {}, {}, 0.0});
      found = std::prev(selections_.end());
    }
    found->atoms.push_back(atom_idx);
    if (mode_ == titan_contact::ContactMode::SidechainCOM) {
      if (mol.info[atom_idx].is_side_chain())
        found->selected_atoms.push_back(atom_idx);
    } else if (titan_contact::is_selected_contact_atom(mol.info[atom_idx],
                                                       mode_)) {
      found->selected_atoms.push_back(atom_idx);
    }
  }

  if (mode_ != titan_contact::ContactMode::SidechainCOM) {
    selections_.erase(
        std::remove_if(selections_.begin(), selections_.end(),
                       [](const ResidueSelection &item) {
                         return item.selected_atoms.empty();
                       }),
        selections_.end());
  } else {
    for (ResidueSelection &selection : selections_) {
      const auto &indices = selection.selected_atoms.empty()
                                ? selection.atoms
                                : selection.selected_atoms;
      if (indices.empty())
        return false;

      selection.com_weights.assign(indices.size(), 0.0);
      selection.com_weight_sum = 0.0;
      for (std::size_t i = 0; i < indices.size(); ++i) {
        const double mass = mol.parameters.mass[indices[i]];
        if (mass > 0.0) {
          selection.com_weights[i] = mass;
          selection.com_weight_sum += mass;
        }
      }
      if (!(selection.com_weight_sum > 0.0)) {
        std::fill(selection.com_weights.begin(), selection.com_weights.end(),
                  1.0);
        selection.com_weight_sum =
            static_cast<double>(selection.com_weights.size());
      }
    }
  }

  residues_.reserve(selections_.size());
  for (const ResidueSelection &selection : selections_)
    residues_.push_back(selection.residue);

  return true;
}

bool ContactGeometryCache::build_points(
    const Multi_MOL &mol, std::size_t snap_idx,
    std::vector<ContactPoint> &points) const {
  points.clear();
  if (snap_idx >= mol.m_mol.size())
    return false;

  const MOL &frame = mol.m_mol[snap_idx];
  if (!frame.coordinates_consistent())
    return false;
  const std::size_t atom_limit = frame.size();

  if (mode_ == titan_contact::ContactMode::SidechainCOM) {
    points.reserve(selections_.size());
    for (const ResidueSelection &selection : selections_) {
      const auto &indices = selection.selected_atoms.empty()
                                ? selection.atoms
                                : selection.selected_atoms;
      if (indices.empty() || indices.size() != selection.com_weights.size() ||
          !(selection.com_weight_sum > 0.0))
        return false;

      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      for (std::size_t i = 0; i < indices.size(); ++i) {
        const std::size_t atom_idx = indices[i];
        if (atom_idx >= atom_limit)
          return false;
        const double weight = selection.com_weights[i];
        x += frame.pos.x[atom_idx] * weight;
        y += frame.pos.y[atom_idx] * weight;
        z += frame.pos.z[atom_idx] * weight;
      }
      points.push_back(ContactPoint{selection.residue,
                                    x / selection.com_weight_sum,
                                    y / selection.com_weight_sum,
                                    z / selection.com_weight_sum});
    }
    return true;
  }

  for (const ResidueSelection &selection : selections_) {
    for (const std::size_t atom_idx : selection.selected_atoms) {
      if (atom_idx >= atom_limit)
        return false;
      points.push_back(ContactPoint{selection.residue, frame.pos.x[atom_idx],
                                    frame.pos.y[atom_idx],
                                    frame.pos.z[atom_idx]});
    }
  }
  return true;
}

} // namespace titan_contact_geometry

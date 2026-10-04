#include "MOL.h"

#include <algorithm>
#include <stdexcept>

namespace {

struct CenterResult {
  double x{};
  double y{};
  double z{};
  double mass{};
  bool valid{false};
};

template <typename Predicate>
CenterResult compute_center(const MOL &mol,
                            const Multi_MOL_parameters &parameters,
                            std::size_t begin, std::size_t end,
                            Predicate include_atom) {
  if (begin > end)
    throw std::runtime_error("COM: invalid atom range.");
  if (!mol.coordinates_consistent())
    throw std::runtime_error("COM: inconsistent coordinate arrays.");
  if (end > mol.size() || end > parameters.size())
    throw std::runtime_error("COM: atom range exceeds coordinate/parameter table.");

  CenterResult weighted;
  std::size_t selected_atoms = 0;
  double unit_x = 0.0;
  double unit_y = 0.0;
  double unit_z = 0.0;

  for (std::size_t atom = begin; atom < end; ++atom) {
    if (!include_atom(atom))
      continue;
    ++selected_atoms;
    unit_x += mol.pos.x[atom];
    unit_y += mol.pos.y[atom];
    unit_z += mol.pos.z[atom];

    const double mass = parameters.mass[atom];
    if (mass > 0.0) {
      weighted.x += mol.pos.x[atom] * mass;
      weighted.y += mol.pos.y[atom] * mass;
      weighted.z += mol.pos.z[atom] * mass;
      weighted.mass += mass;
    }
  }

  if (selected_atoms == 0)
    return CenterResult{};

  if (weighted.mass > 0.0) {
    weighted.x /= weighted.mass;
    weighted.y /= weighted.mass;
    weighted.z /= weighted.mass;
    weighted.valid = true;
    return weighted;
  }

  const double denom = static_cast<double>(selected_atoms);
  return CenterResult{unit_x / denom, unit_y / denom, unit_z / denom, denom,
                      true};
}

CenterResult compute_residue_center(const MOL &mol,
                                    const Multi_MOL_parameters &parameters,
                                    const std::vector<mol_info> &info,
                                    std::size_t begin, std::size_t end,
                                    bool sidechain_only) {
  if (end > info.size())
    throw std::runtime_error("COM: atom range exceeds topology table.");
  return compute_center(mol, parameters, begin, end, [&](std::size_t atom) {
    return !sidechain_only || info[atom].is_side_chain();
  });
}

std::pair<std::size_t, std::size_t>
residue_range_0based(const std::vector<class index> &Index, std::size_t chain,
                     std::size_t residue, std::size_t end_1based) {
  if (chain >= Index.size() || Index[chain].get_size_v() == 0)
    throw std::runtime_error("COM: invalid residue index.");
  if (residue >= Index[chain].get_size_v())
    throw std::runtime_error("COM: invalid residue position.");

  const std::size_t begin_1based = Index[chain].get_res_pos_e(residue);
  const std::size_t next_1based =
      (residue + 1 < Index[chain].get_size_v())
          ? Index[chain].get_res_pos_e(residue + 1)
          : end_1based;
  if (begin_1based == 0 || next_1based == 0 || next_1based < begin_1based)
    throw std::runtime_error("COM: invalid residue atom range.");
  return {begin_1based - 1, next_1based - 1};
}

} // namespace

// take center of mass of side chain of residues
void MOL::take_com(Multi_MOL_parameters &parameters, size_t I, size_t end,
                   vector<class index> &Index, vector<mol_info> &info) {
  if (I >= Index.size() || Index[I].get_size_v() == 0)
    throw std::runtime_error("COM: invalid chain index.");

  for (size_t i{}; i < Index[I].get_size_v(); ++i) {
    const auto [begin_atom, end_atom] = residue_range_0based(Index, I, i, end);
    CenterResult center =
        compute_residue_center(*this, parameters, info, begin_atom, end_atom,
                               true);
    if (!center.valid) {
      center = compute_residue_center(*this, parameters, info, begin_atom,
                                      end_atom, false);
    }
    if (!center.valid)
      throw std::runtime_error("COM: residue has no atoms.");
    Index[I].set_res_coor_e(i, center.x, center.y, center.z);
  }
}

// take center of geometry of residues
void MOL::take_com_all_atoms(Multi_MOL_parameters &parameters, size_t I,
                             size_t end, vector<class index> &Index,
                             vector<mol_info> &info, multi_points &com_res) {
  if (I >= Index.size() || Index[I].get_size_v() == 0)
    throw std::runtime_error("COM: invalid chain index.");

  for (size_t i{}; i < Index[I].get_size_v(); ++i) {
    const auto [begin_atom, end_atom] = residue_range_0based(Index, I, i, end);
    const CenterResult center =
        compute_residue_center(*this, parameters, info, begin_atom, end_atom,
                               false);
    if (!center.valid)
      throw std::runtime_error("COM: residue has no atoms.");
    com_res.emplace(point(center.x, center.y, center.z), center.mass,
                    to_string(Index[I].get_res_num_e(i)));
  }
}

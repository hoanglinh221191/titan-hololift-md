#ifndef TITAN_INTERACTION_PRIMITIVES_H
#define TITAN_INTERACTION_PRIMITIVES_H

#include "MOL.h"
#include "MOL_info.h"
#include "index_v2.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace titan_interaction {

bool is_hydrogen_atom(const mol_info &info);
bool same_residue(const std::vector<mol_info> &info, std::size_t lhs,
                  std::size_t rhs);

std::pair<std::size_t, std::size_t>
find_residue_full_range(const std::vector<class index> &Index,
                        std::size_t chainid, std::size_t target_resnr,
                        std::size_t mol_total_atoms);

std::pair<std::size_t, std::size_t>
find_residue_span_by_atom(const std::vector<mol_info> &info,
                          std::size_t atom_idx);

bool is_hbond_donor(const std::string &atom_name,
                    const std::string &res_name);
bool is_hbond_acceptor(const std::string &atom_name,
                       const std::string &res_name);
bool is_acidic(const std::string &atom_name, const std::string &res_name);
bool is_basic(const std::string &atom_name, const std::string &res_name);

std::size_t find_bonded_hydrogen(const MOL &mol,
                                 const std::vector<mol_info> &info,
                                 std::size_t donor_idx,
                                 std::size_t res_start,
                                 std::size_t res_end);

bool try_calculate_angle_deg(const MOL &donor_mol, std::size_t donor_idx,
                             std::size_t hydrogen_idx,
                             const MOL &acceptor_mol,
                             std::size_t acceptor_idx, double cos_cutoff,
                             double &angle_deg) noexcept;

} // namespace titan_interaction

#endif // TITAN_INTERACTION_PRIMITIVES_H

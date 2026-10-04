#include "titan_interaction_primitives.h"

#include "polymer_classification.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace PC = PolymerClassification;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool is_his_variant(const std::string &res_name_upper) {
  return res_name_upper == "HIS" || res_name_upper == "HSD" ||
         res_name_upper == "HSE" || res_name_upper == "HSP" ||
         res_name_upper == "HID" || res_name_upper == "HIE" ||
         res_name_upper == "HIP" || res_name_upper == "HISD" ||
         res_name_upper == "HISE" || res_name_upper == "HISH" ||
         res_name_upper == "HISA" || res_name_upper == "HISB";
}

bool is_nucleic_base_donor(const std::string &atom_name,
                           const std::string &res_name_upper) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  if (PC::is_purine_residue_name(res_name_upper)) {
    if (res_name_upper == "A" || res_name_upper == "RA" ||
        res_name_upper == "ADE" || res_name_upper == "RIA" ||
        res_name_upper == "DA") {
      return atom == "N6";
    }
    return atom == "N1" || atom == "N2";
  }
  if (res_name_upper == "C" || res_name_upper == "RC" ||
      res_name_upper == "CYT" || res_name_upper == "RCY" ||
      res_name_upper == "DC") {
    return atom == "N4";
  }
  if (res_name_upper == "U" || res_name_upper == "RU" ||
      res_name_upper == "URA" || res_name_upper == "RUR" ||
      res_name_upper == "DT" || res_name_upper == "THY") {
    return atom == "N3";
  }
  return atom == "O2'";
}

bool is_nucleic_base_acceptor(const std::string &atom_name,
                              const std::string &res_name_upper) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  if (res_name_upper == "A" || res_name_upper == "RA" ||
      res_name_upper == "ADE" || res_name_upper == "RIA" ||
      res_name_upper == "DA") {
    return atom == "N1" || atom == "N3" || atom == "N7";
  }
  if (res_name_upper == "G" || res_name_upper == "RG" ||
      res_name_upper == "GUA" || res_name_upper == "RGU" ||
      res_name_upper == "DG") {
    return atom == "O6" || atom == "N3" || atom == "N7";
  }
  if (res_name_upper == "C" || res_name_upper == "RC" ||
      res_name_upper == "CYT" || res_name_upper == "RCY" ||
      res_name_upper == "DC") {
    return atom == "O2" || atom == "N3";
  }
  if (res_name_upper == "U" || res_name_upper == "RU" ||
      res_name_upper == "URA" || res_name_upper == "RUR" ||
      res_name_upper == "DT" || res_name_upper == "THY") {
    return atom == "O2" || atom == "O4";
  }
  return false;
}

bool is_nucleic_sugar_acceptor(const std::string &atom_name) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  return atom == "O2'" || atom == "O3'" || atom == "O4'" || atom == "O5'";
}

double clamp_cosine(double value) noexcept {
  return std::clamp(value, -1.0, 1.0);
}

} // namespace

namespace titan_interaction {

bool is_hydrogen_atom(const mol_info &info) {
  const auto &atom_type = info.get_atom_type();
  const std::string atom_name = info.get_atom();
  return (!atom_type.empty() && atom_type[0] == 'H') ||
         (!atom_name.empty() && atom_name[0] == 'H') ||
         (atom_name.size() >= 2 && atom_name[1] == 'H');
}

bool same_residue(const std::vector<mol_info> &info, std::size_t lhs,
                  std::size_t rhs) {
  return lhs < info.size() && rhs < info.size() &&
         info[lhs].get_chainnum() == info[rhs].get_chainnum() &&
         info[lhs].get_resnr() == info[rhs].get_resnr();
}

std::pair<std::size_t, std::size_t>
find_residue_full_range(const std::vector<class index> &Index,
                        std::size_t chainid, std::size_t target_resnr,
                        std::size_t mol_total_atoms) {
  if (chainid >= Index.size())
    return {SIZE_MAX, 0};
  const auto &idx = Index[chainid];
  std::size_t first_pos = SIZE_MAX;
  std::size_t last_pos = SIZE_MAX;
  for (std::size_t i = 0; i < idx.get_size_v(); ++i) {
    if (idx.get_res_num_e(i) == target_resnr) {
      if (first_pos == SIZE_MAX)
        first_pos = i;
      last_pos = i;
    }
  }
  if (first_pos == SIZE_MAX)
    return {SIZE_MAX, 0};
  const std::size_t start = idx.get_res_pos_e(first_pos) - 1;
  std::size_t end = mol_total_atoms;
  if (last_pos + 1 < idx.get_size_v()) {
    end = idx.get_res_pos_e(last_pos + 1) - 1;
  } else if (chainid + 1 < Index.size()) {
    end = Index[chainid + 1].get_res_pos_e(0) - 1;
  }
  return {start, end};
}

std::pair<std::size_t, std::size_t>
find_residue_span_by_atom(const std::vector<mol_info> &info,
                          std::size_t atom_idx) {
  if (atom_idx >= info.size())
    return {0, 0};
  std::size_t begin = atom_idx;
  while (begin > 0 && same_residue(info, begin - 1, atom_idx))
    --begin;
  std::size_t end = atom_idx + 1;
  while (end < info.size() && same_residue(info, atom_idx, end))
    ++end;
  return {begin, end};
}

bool is_hbond_donor(const std::string &atom_name, const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (PC::is_nucleic_residue_name(upper))
    return is_nucleic_base_donor(atom_name, upper);

  if (atom_name == "N")
    return true;
  if ((upper == "SER" || upper == "THR") && atom_name == "OG")
    return true;
  if (upper == "THR" && atom_name == "OG1")
    return true;
  if (upper == "TYR" && atom_name == "OH")
    return true;
  if (upper == "ARG" &&
      (atom_name == "NH1" || atom_name == "NH2" || atom_name == "NE"))
    return true;
  if (upper == "LYS" && atom_name == "NZ")
    return true;
  if (is_his_variant(upper) &&
      (atom_name == "ND1" || atom_name == "NE2"))
    return true;
  if (upper == "ASN" && atom_name == "ND2")
    return true;
  if (upper == "GLN" && atom_name == "NE2")
    return true;
  if (upper == "TRP" && atom_name == "NE1")
    return true;
  if (upper == "CYS" && atom_name == "SG")
    return true;
  return false;
}

bool is_hbond_acceptor(const std::string &atom_name,
                       const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (PC::is_nucleic_residue_name(upper)) {
    return is_nucleic_base_acceptor(atom_name, upper) ||
           is_nucleic_sugar_acceptor(atom_name) ||
           PC::is_phosphate_oxygen(atom_name);
  }

  if (atom_name == "O")
    return true;
  if ((upper == "ASP" || upper == "ASH") &&
      (atom_name == "OD1" || atom_name == "OD2"))
    return true;
  if ((upper == "GLU" || upper == "GLH") &&
      (atom_name == "OE1" || atom_name == "OE2"))
    return true;
  if ((upper == "SER" || upper == "THR") && atom_name == "OG")
    return true;
  if (upper == "THR" && atom_name == "OG1")
    return true;
  if (upper == "TYR" && atom_name == "OH")
    return true;
  if (upper == "ASN" && atom_name == "OD1")
    return true;
  if (upper == "GLN" && atom_name == "OE1")
    return true;
  if (is_his_variant(upper) &&
      (atom_name == "ND1" || atom_name == "NE2"))
    return true;
  if (upper == "MET" && atom_name == "SD")
    return true;
  return false;
}

bool is_acidic(const std::string &atom_name, const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (upper == "ASP" && (atom_name == "OD1" || atom_name == "OD2"))
    return true;
  if (upper == "GLU" && (atom_name == "OE1" || atom_name == "OE2"))
    return true;
  return false;
}

bool is_basic(const std::string &atom_name, const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (upper == "ARG" &&
      (atom_name == "NH1" || atom_name == "NH2" || atom_name == "NE"))
    return true;
  if (upper == "LYS" && atom_name == "NZ")
    return true;
  if (is_his_variant(upper) &&
      (atom_name == "ND1" || atom_name == "NE2"))
    return true;
  return false;
}

std::size_t find_bonded_hydrogen(const MOL &mol,
                                 const std::vector<mol_info> &info,
                                 std::size_t donor_idx,
                                 std::size_t res_start,
                                 std::size_t res_end) {
  if (donor_idx >= mol.size() || donor_idx >= info.size())
    return SIZE_MAX;
  for (std::size_t h = res_start; h < res_end && h < info.size() &&
                                  h < mol.size();
       ++h) {
    if (!is_hydrogen_atom(info[h]))
      continue;
    const double dx = mol.pos.x[h] - mol.pos.x[donor_idx];
    const double dy = mol.pos.y[h] - mol.pos.y[donor_idx];
    const double dz = mol.pos.z[h] - mol.pos.z[donor_idx];
    if ((dx * dx + dy * dy + dz * dz) < 1.44)
      return h;
  }
  return SIZE_MAX;
}

bool try_calculate_angle_deg(const MOL &donor_mol, std::size_t donor_idx,
                             std::size_t hydrogen_idx,
                             const MOL &acceptor_mol,
                             std::size_t acceptor_idx, double cos_cutoff,
                             double &angle_deg) noexcept {
  const double dh_x =
      donor_mol.pos.x[donor_idx] - donor_mol.pos.x[hydrogen_idx];
  const double dh_y =
      donor_mol.pos.y[donor_idx] - donor_mol.pos.y[hydrogen_idx];
  const double dh_z =
      donor_mol.pos.z[donor_idx] - donor_mol.pos.z[hydrogen_idx];

  const double ah_x =
      acceptor_mol.pos.x[acceptor_idx] - donor_mol.pos.x[hydrogen_idx];
  const double ah_y =
      acceptor_mol.pos.y[acceptor_idx] - donor_mol.pos.y[hydrogen_idx];
  const double ah_z =
      acceptor_mol.pos.z[acceptor_idx] - donor_mol.pos.z[hydrogen_idx];

  const double dh_norm_sq = dh_x * dh_x + dh_y * dh_y + dh_z * dh_z;
  const double ah_norm_sq = ah_x * ah_x + ah_y * ah_y + ah_z * ah_z;
  if (dh_norm_sq <= std::numeric_limits<double>::epsilon() ||
      ah_norm_sq <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double dot = dh_x * ah_x + dh_y * ah_y + dh_z * ah_z;
  const double cosine =
      clamp_cosine(dot / std::sqrt(dh_norm_sq * ah_norm_sq));
  if (cosine > cos_cutoff)
    return false;

  angle_deg = std::acos(cosine) * 180.0 / kPi;
  return true;
}

} // namespace titan_interaction

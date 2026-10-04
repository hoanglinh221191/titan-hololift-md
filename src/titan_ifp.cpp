#include "titan_ifp.h"
#include "polymer_classification.h"
#include "titan_executor.h"
#include "titan_interaction_primitives.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
namespace PC = PolymerClassification;

namespace {

constexpr size_t kBaseRegion = 0;
constexpr size_t kSugarRegion = 1;
constexpr size_t kPhosphateRegion = 2;

bool is_his_variant(const std::string &res_name_upper) {
  return res_name_upper == "HIS" || res_name_upper == "HSD" ||
         res_name_upper == "HSE" || res_name_upper == "HSP" ||
         res_name_upper == "HID" || res_name_upper == "HIE" ||
         res_name_upper == "HIP" || res_name_upper == "HISD" ||
         res_name_upper == "HISE" || res_name_upper == "HISH" ||
         res_name_upper == "HISA" || res_name_upper == "HISB";
}

bool is_protein_aromatic_residue(const std::string &res_name_upper) {
  return res_name_upper == "PHE" || res_name_upper == "TYR" ||
         res_name_upper == "TRP" || is_his_variant(res_name_upper);
}

[[maybe_unused]] bool is_nucleic_base_donor(const std::string &atom_name,
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

[[maybe_unused]] bool
is_nucleic_base_acceptor(const std::string &atom_name,
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

[[maybe_unused]] bool is_nucleic_sugar_acceptor(const std::string &atom_name) {
  const std::string atom = PC::normalize_nucleic_atom_name(atom_name);
  return atom == "O2'" || atom == "O3'" || atom == "O4'" || atom == "O5'";
}

bool is_protein_hydrophobic_atom(const std::string &atom_name,
                                 const std::string &res_name_upper) {
  if (res_name_upper == "ASP" || res_name_upper == "ASH" ||
      res_name_upper == "GLU" || res_name_upper == "GLH" ||
      res_name_upper == "ARG" || res_name_upper == "LYS" ||
      res_name_upper == "ASN" || res_name_upper == "GLN" ||
      is_his_variant(res_name_upper)) {
    return false;
  }

  if (atom_name.empty())
    return false;
  if (atom_name[0] == 'C' && atom_name != "C")
    return true;
  return atom_name == "SD" && res_name_upper == "MET";
}

size_t region_index(PC::NucleicAtomRegion region) {
  switch (region) {
  case PC::NucleicAtomRegion::Base:
    return kBaseRegion;
  case PC::NucleicAtomRegion::Sugar:
    return kSugarRegion;
  case PC::NucleicAtomRegion::Phosphate:
    return kPhosphateRegion;
  default:
    return static_cast<size_t>(-1);
  }
}

void mark_nucleic_region(std::array<bool, 3> &flags, const mol_info &info) {
  if (!PC::is_nucleic_residue_name(PC::uppercase_copy(info.get_res_name())))
    return;
  const size_t idx =
      region_index(PC::classify_nucleic_atom_region(info.get_atom()));
  if (idx < flags.size())
    flags[idx] = true;
}

std::pair<size_t, size_t>
find_residue_full_range(const std::vector<class index> &Index, size_t chainid,
                        size_t target_resnr, size_t mol_total_atoms) {
  if (chainid >= Index.size())
    return {SIZE_MAX, 0};
  const auto &idx = Index[chainid];
  size_t first_pos = SIZE_MAX;
  size_t last_pos = SIZE_MAX;
  for (size_t i = 0; i < idx.get_size_v(); ++i) {
    if (idx.get_res_num_e(i) == target_resnr) {
      if (first_pos == SIZE_MAX)
        first_pos = i;
      last_pos = i;
    }
  }
  if (first_pos == SIZE_MAX)
    return {SIZE_MAX, 0};
  const size_t start = idx.get_res_pos_e(first_pos) - 1;
  size_t end = mol_total_atoms;
  if (last_pos + 1 < idx.get_size_v()) {
    end = idx.get_res_pos_e(last_pos + 1) - 1;
  } else if (chainid + 1 < Index.size()) {
    end = Index[chainid + 1].get_res_pos_e(0) - 1;
  }
  return {start, end};
}

} // namespace

bool TitanIFP::is_aromatic_residue(const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  return is_protein_aromatic_residue(upper) ||
         PC::is_nucleic_residue_name(upper);
}

bool TitanIFP::is_hydrophobic_atom(const std::string &atom_name,
                                   const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (PC::is_nucleic_residue_name(upper))
    return PC::is_nucleobase_carbon(atom_name);
  return is_protein_hydrophobic_atom(atom_name, upper);
}

bool TitanIFP::is_hbond_donor(const std::string &atom,
                              const std::string &res_name) {
  return titan_interaction::is_hbond_donor(atom, res_name);
}

bool TitanIFP::is_hbond_acceptor(const std::string &atom,
                                 const std::string &res_name) {
  return titan_interaction::is_hbond_acceptor(atom, res_name);
}

bool TitanIFP::is_acidic(const std::string &atom, const std::string &res_name) {
  return titan_interaction::is_acidic(atom, res_name);
}

bool TitanIFP::is_basic(const std::string &atom, const std::string &res_name) {
  return titan_interaction::is_basic(atom, res_name);
}

bool TitanIFP::is_halogen(const std::string &atom) {
  return atom == "CL" || atom == "BR" || atom == "I" ||
         atom == "Cl" || atom == "Br";
}

bool TitanIFP::is_halogen_acceptor(const std::string &atom) {
  if (atom.empty())
    return false;
  return atom[0] == 'O' || atom[0] == 'N' || atom[0] == 'S';
}

bool TitanIFP::is_cation_atom(const std::string &atom,
                              const std::string &res_name) {
  const std::string upper = PC::uppercase_copy(res_name);
  if (upper == "ARG" && (atom == "NH1" || atom == "NH2" || atom == "NE"))
    return true;
  if (upper == "LYS" && atom == "NZ")
    return true;
  if ((upper == "HSP" || upper == "HIP" || upper == "HISH") &&
      (atom == "ND1" || atom == "NE2"))
    return true;
  return false;
}

TitanIFP::RingAtoms
TitanIFP::get_ring_atoms(const std::string &res_name,
                         const std::vector<mol_info> &info, size_t start,
                         size_t end) {
  const std::string upper = PC::uppercase_copy(res_name);

  static const std::vector<std::string> phe_ring = {
      "CG", "CD1", "CD2", "CE1", "CE2", "CZ"};
  static const std::vector<std::string> tyr_ring = {
      "CG", "CD1", "CD2", "CE1", "CE2", "CZ"};
  static const std::vector<std::string> trp_ring = {
      "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"};
  static const std::vector<std::string> his_ring = {
      "CG", "ND1", "CD2", "CE1", "NE2"};
  static const std::vector<std::string> purine_ring = {
      "N9", "C8", "N7", "C5", "C4", "N3", "C2", "N1", "C6"};
  static const std::vector<std::string> pyrimidine_ring = {
      "N1", "C2", "N3", "C4", "C5", "C6"};

  const std::vector<std::string> *ring_names = nullptr;
  if (upper == "PHE")
    ring_names = &phe_ring;
  else if (upper == "TYR")
    ring_names = &tyr_ring;
  else if (upper == "TRP")
    ring_names = &trp_ring;
  else if (is_his_variant(upper))
    ring_names = &his_ring;
  else if (PC::is_purine_residue_name(upper))
    ring_names = &purine_ring;
  else if (PC::is_pyrimidine_residue_name(upper))
    ring_names = &pyrimidine_ring;

  RingAtoms indices;
  if (ring_names == nullptr)
    return indices;

  for (size_t i = start; i < end; ++i) {
    const std::string atom_name =
        PC::normalize_nucleic_atom_name(info[i].get_atom());
    for (const auto &expected : *ring_names) {
      if (atom_name == expected) {
        indices.push_back(i);
        break;
      }
    }
  }
  return indices;
}

bool TitanIFP::compute_ring_geometry(const MOL &mol,
                                     const RingAtoms &ring_indices,
                                     titan::algebra::Vec3 &com,
                                     titan::algebra::Vec3 &normal) {
  using titan::algebra::Vec3;

  if (ring_indices.size() < 3)
    return false;

  com = {};
  for (auto idx : ring_indices) {
    com += Vec3{mol.pos.x[idx], mol.pos.y[idx], mol.pos.z[idx]};
  }
  com = com / static_cast<double>(ring_indices.size());

  const Vec3 p0{mol.pos.x[ring_indices.idx[0]],
                mol.pos.y[ring_indices.idx[0]],
                mol.pos.z[ring_indices.idx[0]]};
  const Vec3 p1{mol.pos.x[ring_indices.idx[1]],
                mol.pos.y[ring_indices.idx[1]],
                mol.pos.z[ring_indices.idx[1]]};
  const Vec3 p2{mol.pos.x[ring_indices.idx[2]],
                mol.pos.y[ring_indices.idx[2]],
                mol.pos.z[ring_indices.idx[2]]};
  const Vec3 n = titan::algebra::cross(p1 - p0, p2 - p0);
  const double n_norm = titan::algebra::norm(n);
  if (n_norm < 1e-10)
    return false;
  normal = n / n_norm;
  return true;
}

size_t TitanIFP::find_bonded_hydrogen(const MOL &mol,
                                      const std::vector<mol_info> &info,
                                      size_t donor_idx, size_t res_start,
                                      size_t res_end) const {
  return titan_interaction::find_bonded_hydrogen(mol, info, donor_idx,
                                                 res_start, res_end);
}

bool TitanIFP::check_hbond(const MOL &rec_mol, const MOL &lig_mol,
                           const std::vector<mol_info> &rec_info,
                           const std::vector<mol_info> &lig_info, size_t ar,
                           size_t al, size_t rs, size_t re, size_t ls,
                           size_t le, double d, const IfpArgs &args) const {
  if (d >= args.cutoff_hbond)
    return false;

  const auto &atom_r = rec_info[ar].get_atom();
  const auto &res_r = rec_info[ar].get_res_name();
  const auto &atom_l = lig_info[al].get_atom();
  const auto &res_l = lig_info[al].get_res_name();

  if (is_hbond_donor(atom_r, res_r) && is_hbond_acceptor(atom_l, res_l)) {
    const size_t h_idx = find_bonded_hydrogen(rec_mol, rec_info, ar, rs, re);
    if (h_idx != SIZE_MAX) {
      const double hx = rec_mol.pos.x[h_idx];
      const double hy = rec_mol.pos.y[h_idx];
      const double hz = rec_mol.pos.z[h_idx];
      const double v_hd_x = hx - rec_mol.pos.x[ar];
      const double v_hd_y = hy - rec_mol.pos.y[ar];
      const double v_hd_z = hz - rec_mol.pos.z[ar];
      const double v_ha_x = lig_mol.pos.x[al] - hx;
      const double v_ha_y = lig_mol.pos.y[al] - hy;
      const double v_ha_z = lig_mol.pos.z[al] - hz;
      const double dot = v_hd_x * v_ha_x + v_hd_y * v_ha_y + v_hd_z * v_ha_z;
      const double mag_hd =
          std::sqrt(v_hd_x * v_hd_x + v_hd_y * v_hd_y + v_hd_z * v_hd_z);
      const double mag_ha =
          std::sqrt(v_ha_x * v_ha_x + v_ha_y * v_ha_y + v_ha_z * v_ha_z);
      if (mag_hd > 1e-10 && mag_ha > 1e-10) {
        if ((dot / (mag_hd * mag_ha)) <= -0.7071)
          return true;
      }
    }
  }

  if (is_hbond_donor(atom_l, res_l) && is_hbond_acceptor(atom_r, res_r)) {
    const size_t h_idx = find_bonded_hydrogen(lig_mol, lig_info, al, ls, le);
    if (h_idx != SIZE_MAX) {
      const double hx = lig_mol.pos.x[h_idx];
      const double hy = lig_mol.pos.y[h_idx];
      const double hz = lig_mol.pos.z[h_idx];
      const double v_hd_x = hx - lig_mol.pos.x[al];
      const double v_hd_y = hy - lig_mol.pos.y[al];
      const double v_hd_z = hz - lig_mol.pos.z[al];
      const double v_ha_x = rec_mol.pos.x[ar] - hx;
      const double v_ha_y = rec_mol.pos.y[ar] - hy;
      const double v_ha_z = rec_mol.pos.z[ar] - hz;
      const double dot = v_hd_x * v_ha_x + v_hd_y * v_ha_y + v_hd_z * v_ha_z;
      const double mag_hd =
          std::sqrt(v_hd_x * v_hd_x + v_hd_y * v_hd_y + v_hd_z * v_hd_z);
      const double mag_ha =
          std::sqrt(v_ha_x * v_ha_x + v_ha_y * v_ha_y + v_ha_z * v_ha_z);
      if (mag_hd > 1e-10 && mag_ha > 1e-10) {
        if ((dot / (mag_hd * mag_ha)) <= -0.7071)
          return true;
      }
    }
  }

  return false;
}

bool TitanIFP::check_salt_bridge(const mol_info &info_a,
                                 const mol_info &info_b) {
  const auto &atom_a = info_a.get_atom();
  const auto &res_a = info_a.get_res_name();
  const auto &atom_b = info_b.get_atom();
  const auto &res_b = info_b.get_res_name();

  const bool protein_ionic =
      (is_acidic(atom_a, res_a) && is_basic(atom_b, res_b)) ||
      (is_basic(atom_a, res_a) && is_acidic(atom_b, res_b));
  if (protein_ionic)
    return true;

  const std::string upper_a = PC::uppercase_copy(res_a);
  const std::string upper_b = PC::uppercase_copy(res_b);
  const bool a_is_phosphate =
      PC::is_nucleic_residue_name(upper_a) && PC::is_phosphate_oxygen(atom_a);
  const bool b_is_phosphate =
      PC::is_nucleic_residue_name(upper_b) && PC::is_phosphate_oxygen(atom_b);
  return (a_is_phosphate && is_basic(atom_b, res_b)) ||
         (b_is_phosphate && is_basic(atom_a, res_a));
}

bool TitanIFP::check_hydrophobic(const mol_info &info_a,
                                 const mol_info &info_b) {
  return is_hydrophobic_atom(info_a.get_atom(), info_a.get_res_name()) &&
         is_hydrophobic_atom(info_b.get_atom(), info_b.get_res_name());
}

bool TitanIFP::check_pi_pi(const MOL &rec_mol, const MOL &lig_mol,
                           size_t rec_start, size_t rec_end, size_t lig_start,
                           size_t lig_end,
                           const std::vector<mol_info> &rec_info,
                           const std::vector<mol_info> &lig_info,
                           const std::string &rec_res_name,
                           const std::string &lig_res_name,
                           double cutoff, double cutoff_cation_pi,
                           bool &is_cation_pi) const {
  is_cation_pi = false;

  const auto rec_ring = get_ring_atoms(rec_res_name, rec_info, rec_start, rec_end);
  const auto lig_ring = get_ring_atoms(lig_res_name, lig_info, lig_start, lig_end);
  const bool rec_aromatic = !rec_ring.empty();
  const bool lig_aromatic = !lig_ring.empty();

  using titan::algebra::Vec3;

  Vec3 rec_com, rec_normal, lig_com, lig_normal;
  const bool rec_geom_ok =
      rec_aromatic && compute_ring_geometry(rec_mol, rec_ring, rec_com, rec_normal);
  const bool lig_geom_ok =
      lig_aromatic && compute_ring_geometry(lig_mol, lig_ring, lig_com, lig_normal);

  bool pi_pi_detected = false;
  if (rec_geom_ok && lig_geom_ok) {
    const double com_dist = titan::algebra::norm(rec_com - lig_com);
    if (com_dist < cutoff) {
      const double cos_angle =
          std::abs(titan::algebra::dot(rec_normal, lig_normal));
      if (cos_angle > 0.866 || (cos_angle >= 0.0 && cos_angle < 0.643))
        pi_pi_detected = true;
    }
  }

  if (rec_geom_ok) {
    for (size_t al = lig_start; al < lig_end; ++al) {
      if (is_cation_atom(lig_info[al].get_atom(), lig_info[al].get_res_name())) {
        const Vec3 cat_pos{lig_mol.pos.x[al], lig_mol.pos.y[al],
                           lig_mol.pos.z[al]};
        if (titan::algebra::norm(rec_com - cat_pos) < cutoff_cation_pi) {
          is_cation_pi = true;
          break;
        }
      }
    }
  }

  if (lig_geom_ok && !is_cation_pi) {
    for (size_t ar = rec_start; ar < rec_end; ++ar) {
      if (is_cation_atom(rec_info[ar].get_atom(), rec_info[ar].get_res_name())) {
        const Vec3 cat_pos{rec_mol.pos.x[ar], rec_mol.pos.y[ar],
                           rec_mol.pos.z[ar]};
        if (titan::algebra::norm(lig_com - cat_pos) < cutoff_cation_pi) {
          is_cation_pi = true;
          break;
        }
      }
    }
  }

  return pi_pi_detected;
}

bool TitanIFP::check_halogen(const MOL &rec_mol, const MOL &lig_mol,
                             const std::vector<mol_info> &rec_info,
                             const std::vector<mol_info> &lig_info, size_t ar,
                             size_t al, size_t rs, size_t re, size_t ls,
                             size_t le, double d) const {
  (void)d;
  const auto &atom_r = rec_info[ar].get_atom();
  const auto &atom_l = lig_info[al].get_atom();

  if (is_halogen(atom_l) && is_halogen_acceptor(atom_r)) {
    for (size_t c = ls; c < le; ++c) {
      if (lig_info[c].get_atom().empty() || lig_info[c].get_atom()[0] != 'C')
        continue;
      const double cx_dx = lig_mol.pos.x[c] - lig_mol.pos.x[al];
      const double cx_dy = lig_mol.pos.y[c] - lig_mol.pos.y[al];
      const double cx_dz = lig_mol.pos.z[c] - lig_mol.pos.z[al];
      const double cx_d = std::sqrt(cx_dx * cx_dx + cx_dy * cx_dy + cx_dz * cx_dz);
      if (cx_d < 2.1 && cx_d > 0.1) {
        const double v_cx_x = lig_mol.pos.x[c] - lig_mol.pos.x[al];
        const double v_cx_y = lig_mol.pos.y[c] - lig_mol.pos.y[al];
        const double v_cx_z = lig_mol.pos.z[c] - lig_mol.pos.z[al];
        const double v_xa_x = rec_mol.pos.x[ar] - lig_mol.pos.x[al];
        const double v_xa_y = rec_mol.pos.y[ar] - lig_mol.pos.y[al];
        const double v_xa_z = rec_mol.pos.z[ar] - lig_mol.pos.z[al];
        const double dot = v_cx_x * v_xa_x + v_cx_y * v_xa_y + v_cx_z * v_xa_z;
        const double m1 = std::sqrt(v_cx_x * v_cx_x + v_cx_y * v_cx_y +
                                    v_cx_z * v_cx_z);
        const double m2 = std::sqrt(v_xa_x * v_xa_x + v_xa_y * v_xa_y +
                                    v_xa_z * v_xa_z);
        if (m1 > 1e-10 && m2 > 1e-10 && (dot / (m1 * m2)) < -0.9397)
          return true;
      }
    }
  }

  if (is_halogen(atom_r) && is_halogen_acceptor(atom_l)) {
    for (size_t c = rs; c < re; ++c) {
      if (rec_info[c].get_atom().empty() || rec_info[c].get_atom()[0] != 'C')
        continue;
      const double cx_dx = rec_mol.pos.x[c] - rec_mol.pos.x[ar];
      const double cx_dy = rec_mol.pos.y[c] - rec_mol.pos.y[ar];
      const double cx_dz = rec_mol.pos.z[c] - rec_mol.pos.z[ar];
      const double cx_d = std::sqrt(cx_dx * cx_dx + cx_dy * cx_dy + cx_dz * cx_dz);
      if (cx_d < 2.1 && cx_d > 0.1) {
        const double v_cx_x = rec_mol.pos.x[c] - rec_mol.pos.x[ar];
        const double v_cx_y = rec_mol.pos.y[c] - rec_mol.pos.y[ar];
        const double v_cx_z = rec_mol.pos.z[c] - rec_mol.pos.z[ar];
        const double v_xa_x = lig_mol.pos.x[al] - rec_mol.pos.x[ar];
        const double v_xa_y = lig_mol.pos.y[al] - rec_mol.pos.y[ar];
        const double v_xa_z = lig_mol.pos.z[al] - rec_mol.pos.z[ar];
        const double dot = v_cx_x * v_xa_x + v_cx_y * v_xa_y + v_cx_z * v_xa_z;
        const double m1 = std::sqrt(v_cx_x * v_cx_x + v_cx_y * v_cx_y +
                                    v_cx_z * v_cx_z);
        const double m2 = std::sqrt(v_xa_x * v_xa_x + v_xa_y * v_xa_y +
                                    v_xa_z * v_xa_z);
        if (m1 > 1e-10 && m2 > 1e-10 && (dot / (m1 * m2)) < -0.9397)
          return true;
      }
    }
  }

  return false;
}

void TitanIFP::ensure_worker_storage(size_t worker_count) {
  if (worker_frame_results_.size() < worker_count)
    worker_frame_results_.resize(worker_count);
  if (worker_contact_accumulators_.size() < worker_count)
    worker_contact_accumulators_.resize(worker_count);

  for (size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
    worker_frame_results_[worker_id].contacts.clear();
    worker_frame_results_[worker_id].rna_regions.clear();
    worker_contact_accumulators_[worker_id].clear();
  }
}

std::vector<TitanIFP::PreparedResiduePair> TitanIFP::build_prepared_pairs(
    const std::vector<d_min> &dis_min, const std::vector<class index> &rec_index,
    const std::vector<class index> &lig_index, size_t rec_total_atoms,
    size_t lig_total_atoms) const {
  size_t total_contacts = 0;
  for (const auto &dm : dis_min)
    total_contacts += dm.base.size();

  std::vector<dmin_base> unique_pairs;
  unique_pairs.reserve(total_contacts);
  for (const auto &dm : dis_min)
    unique_pairs.insert(unique_pairs.end(), dm.base.begin(), dm.base.end());

  std::sort(unique_pairs.begin(), unique_pairs.end(),
            [](const dmin_base &lhs, const dmin_base &rhs) {
              if (lhs.chainid_rec != rhs.chainid_rec)
                return lhs.chainid_rec < rhs.chainid_rec;
              if (lhs.resnr_rec != rhs.resnr_rec)
                return lhs.resnr_rec < rhs.resnr_rec;
              if (lhs.chainid_ab != rhs.chainid_ab)
                return lhs.chainid_ab < rhs.chainid_ab;
              return lhs.resnr_ab < rhs.resnr_ab;
            });
  unique_pairs.erase(
      std::unique(unique_pairs.begin(), unique_pairs.end(),
                  [](const dmin_base &lhs, const dmin_base &rhs) {
                    return lhs.chainid_rec == rhs.chainid_rec &&
                           lhs.resnr_rec == rhs.resnr_rec &&
                           lhs.chainid_ab == rhs.chainid_ab &&
                           lhs.resnr_ab == rhs.resnr_ab;
                  }),
      unique_pairs.end());

  std::vector<PreparedResiduePair> prepared_pairs;
  prepared_pairs.reserve(unique_pairs.size());
  for (const auto &contact : unique_pairs) {
    const auto [rs, re] = find_residue_full_range(
        rec_index, contact.chainid_rec, contact.resnr_rec, rec_total_atoms);
    if (rs == SIZE_MAX || rs >= re)
      continue;

    const auto [ls, le] = find_residue_full_range(
        lig_index, contact.chainid_ab, contact.resnr_ab, lig_total_atoms);
    if (ls == SIZE_MAX || ls >= le)
      continue;

    PreparedResiduePair prepared;
    prepared.contact = contact;
    prepared.rs = rs;
    prepared.re = re;
    prepared.ls = ls;
    prepared.le = le;
    prepared.pair_has_nucleic =
        PC::is_nucleic_residue_name(PC::uppercase_copy(contact.res_name_rec)) ||
        PC::is_nucleic_residue_name(PC::uppercase_copy(contact.res_name_ab));
    prepared_pairs.push_back(std::move(prepared));
  }
  return prepared_pairs;
}

bool TitanIFP::analyze_prepared_pair(const PreparedResiduePair &pair,
                                     const MOL &rec_mol, const MOL &lig_mol,
                                     const std::vector<mol_info> &rec_info,
                                     const std::vector<mol_info> &lig_info,
                                     const IfpArgs &args, IFP_Contact &result,
                                     std::array<bool, 3> &region_flags) const {
  if (pair.rs >= pair.re || pair.ls >= pair.le)
    return false;
  if (pair.rs >= rec_mol.size() || pair.re > rec_mol.size() ||
      pair.ls >= lig_mol.size() || pair.le > lig_mol.size()) {
    return false;
  }

  result.resnr_rec = pair.contact.resnr_rec;
  result.resnr_lig = pair.contact.resnr_ab;
  result.chainid_rec = pair.contact.chainid_rec;
  result.chainid_lig = pair.contact.chainid_ab;
  result.res_name_rec = pair.contact.res_name_rec;
  result.res_name_lig = pair.contact.res_name_ab;
  region_flags = {};

  const double max_cutoff = std::max(
      {args.cutoff_hbond, args.cutoff_salt_bridge, args.cutoff_hydrophobic,
       args.cutoff_halogen, args.cutoff_pi_pi, args.cutoff_cation_pi});
  const double max_cutoff_sq = max_cutoff * max_cutoff;

  for (size_t ar = pair.rs; ar < pair.re; ++ar) {
    const double rx = rec_mol.pos.x[ar];
    const double ry = rec_mol.pos.y[ar];
    const double rz = rec_mol.pos.z[ar];

    constexpr uint8_t pairwise_mask =
        (1u << BIT_HBOND) | (1u << BIT_SALT) | (1u << BIT_HYDROPHOBIC) |
        (1u << BIT_HALOGEN);
    const bool all_pairwise_detected =
        (result.interactions.to_ulong() & pairwise_mask) == pairwise_mask;
    if (!pair.pair_has_nucleic && all_pairwise_detected)
      break;
    if (pair.pair_has_nucleic && all_pairwise_detected &&
        region_flags[kBaseRegion] && region_flags[kSugarRegion] &&
        region_flags[kPhosphateRegion]) {
      break;
    }

    size_t al = pair.ls;
    const size_t vec_end = pair.ls + ((pair.le - pair.ls) / 4) * 4;
    for (; al < vec_end; al += 4) {
      const unsigned bitmask = titan_simd_dispatch::point_vs_four_cutoff_mask(
          rx, ry, rz, &lig_mol.pos.x[al], &lig_mol.pos.y[al],
          &lig_mol.pos.z[al], max_cutoff_sq, 0.0, false);
      if (bitmask == 0)
        continue;

      for (int bit = 0; bit < 4; ++bit) {
        if (!(bitmask & (1 << bit)))
          continue;
        const size_t al_idx = al + static_cast<size_t>(bit);
        const double ddx = rx - lig_mol.pos.x[al_idx];
        const double ddy = ry - lig_mol.pos.y[al_idx];
        const double ddz = rz - lig_mol.pos.z[al_idx];
        const double d = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

        if (pair.pair_has_nucleic) {
          mark_nucleic_region(region_flags, rec_info[ar]);
          mark_nucleic_region(region_flags, lig_info[al_idx]);
        }

        if (!result.interactions[BIT_HBOND] &&
            check_hbond(rec_mol, lig_mol, rec_info, lig_info, ar, al_idx,
                        pair.rs, pair.re, pair.ls, pair.le, d, args)) {
          result.interactions.set(BIT_HBOND);
        }
        if (!result.interactions[BIT_SALT] && d < args.cutoff_salt_bridge &&
            check_salt_bridge(rec_info[ar], lig_info[al_idx])) {
          result.interactions.set(BIT_SALT);
        }
        if (!result.interactions[BIT_HYDROPHOBIC] &&
            d < args.cutoff_hydrophobic &&
            check_hydrophobic(rec_info[ar], lig_info[al_idx])) {
          result.interactions.set(BIT_HYDROPHOBIC);
        }
        if (!result.interactions[BIT_HALOGEN] && d < args.cutoff_halogen &&
            check_halogen(rec_mol, lig_mol, rec_info, lig_info, ar, al_idx,
                          pair.rs, pair.re, pair.ls, pair.le, d)) {
          result.interactions.set(BIT_HALOGEN);
        }
      }
    }

    for (; al < pair.le; ++al) {
      const double ddx = rx - lig_mol.pos.x[al];
      const double ddy = ry - lig_mol.pos.y[al];
      const double ddz = rz - lig_mol.pos.z[al];
      const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
      if (d2 >= max_cutoff_sq)
        continue;
      const double d = std::sqrt(d2);

      if (pair.pair_has_nucleic) {
        mark_nucleic_region(region_flags, rec_info[ar]);
        mark_nucleic_region(region_flags, lig_info[al]);
      }
      if (!result.interactions[BIT_HBOND] &&
          check_hbond(rec_mol, lig_mol, rec_info, lig_info, ar, al, pair.rs,
                      pair.re, pair.ls, pair.le, d, args)) {
        result.interactions.set(BIT_HBOND);
      }
      if (!result.interactions[BIT_SALT] && d < args.cutoff_salt_bridge &&
          check_salt_bridge(rec_info[ar], lig_info[al])) {
        result.interactions.set(BIT_SALT);
      }
      if (!result.interactions[BIT_HYDROPHOBIC] &&
          d < args.cutoff_hydrophobic &&
          check_hydrophobic(rec_info[ar], lig_info[al])) {
        result.interactions.set(BIT_HYDROPHOBIC);
      }
      if (!result.interactions[BIT_HALOGEN] && d < args.cutoff_halogen &&
          check_halogen(rec_mol, lig_mol, rec_info, lig_info, ar, al, pair.rs,
                        pair.re, pair.ls, pair.le, d)) {
        result.interactions.set(BIT_HALOGEN);
      }
    }
  }

  if (is_aromatic_residue(pair.contact.res_name_rec) ||
      is_aromatic_residue(pair.contact.res_name_ab)) {
    bool is_cat_pi = false;
    if (check_pi_pi(rec_mol, lig_mol, pair.rs, pair.re, pair.ls, pair.le,
                    rec_info, lig_info, pair.contact.res_name_rec,
                    pair.contact.res_name_ab, args.cutoff_pi_pi,
                    args.cutoff_cation_pi, is_cat_pi)) {
      result.interactions.set(BIT_PI_PI);
    }
    if (is_cat_pi)
      result.interactions.set(BIT_CATION_PI);
  }

  return result.interactions.any();
}

TitanIFP::FrameAnalysisResult TitanIFP::analyze_frame(
    const MOL &rec_mol, const MOL &lig_mol,
    const std::vector<mol_info> &rec_info,
    const std::vector<mol_info> &lig_info,
    const std::vector<class index> &rec_index,
    const std::vector<class index> &lig_index, size_t rec_total_atoms,
    size_t lig_total_atoms, const std::vector<d_min> &dis_min,
    const IfpArgs &args) const {
  FrameAnalysisResult analysis;
  const auto prepared_pairs = build_prepared_pairs(
      dis_min, rec_index, lig_index, rec_total_atoms, lig_total_atoms);

  for (const auto &pair : prepared_pairs) {
    IFP_Contact result;
    std::array<bool, 3> region_flags{};
    const bool has_contact =
        analyze_prepared_pair(pair, rec_mol, lig_mol, rec_info, lig_info, args,
                              result, region_flags);
    if (has_contact)
      analysis.contacts.push_back(result);

    if (pair.pair_has_nucleic &&
        (region_flags[kBaseRegion] || region_flags[kSugarRegion] ||
         region_flags[kPhosphateRegion])) {
      FrameRnaRegionContact region_contact;
      region_contact.resnr_rec = result.resnr_rec;
      region_contact.resnr_lig = result.resnr_lig;
      region_contact.chainid_rec = result.chainid_rec;
      region_contact.chainid_lig = result.chainid_lig;
      region_contact.res_name_rec = result.res_name_rec;
      region_contact.res_name_lig = result.res_name_lig;
      region_contact.flags = region_flags;
      analysis.rna_regions.push_back(std::move(region_contact));
    }
  }

  return analysis;
}

IFP_Frame TitanIFP::collect_frame_contacts(
    const MOL &rec_mol, const MOL &lig_mol,
    const std::vector<mol_info> &rec_info,
    const std::vector<mol_info> &lig_info,
    const std::vector<class index> &rec_index,
    const std::vector<class index> &lig_index, size_t rec_total_atoms,
    size_t lig_total_atoms, const std::vector<d_min> &dis_min,
    const IfpArgs &args) const {
  return analyze_frame(rec_mol, lig_mol, rec_info, lig_info, rec_index,
                       lig_index, rec_total_atoms, lig_total_atoms, dis_min,
                       args)
      .contacts;
}

void TitanIFP::accumulate_frame(const FrameAnalysisResult &frame) {
  for (const auto &result : frame.contacts) {
    const ResKey key{result.chainid_rec, result.resnr_rec, result.chainid_lig,
                     result.resnr_lig};
    auto &record = occupancy_map_[key];
    if (record.res_name_rec.empty()) {
      record.resnr_rec = result.resnr_rec;
      record.resnr_lig = result.resnr_lig;
      record.chainid_rec = result.chainid_rec;
      record.chainid_lig = result.chainid_lig;
      record.res_name_rec = result.res_name_rec;
      record.res_name_lig = result.res_name_lig;
    }
    for (int bit = 0; bit < 6; ++bit) {
      if (result.interactions[bit])
        record.counts[static_cast<size_t>(bit)]++;
    }
  }

  for (const auto &region : frame.rna_regions) {
    const ResKey key{region.chainid_rec, region.resnr_rec, region.chainid_lig,
                     region.resnr_lig};
    auto &record = rna_region_map_[key];
    if (record.res_name_rec.empty()) {
      record.resnr_rec = region.resnr_rec;
      record.resnr_lig = region.resnr_lig;
      record.chainid_rec = region.chainid_rec;
      record.chainid_lig = region.chainid_lig;
      record.res_name_rec = region.res_name_rec;
      record.res_name_lig = region.res_name_lig;
    }
    for (size_t idx = 0; idx < region.flags.size(); ++idx) {
      if (region.flags[idx])
        record.counts[idx]++;
    }
  }
}

void TitanIFP::accumulate_contact_counts(ContactCountAccumulator &accumulator,
                                         const IFP_Contact &contact) {
  const ResKey key{contact.chainid_rec, contact.resnr_rec, contact.chainid_lig,
                   contact.resnr_lig};
  auto &record = accumulator[key];
  if (record.res_name_rec.empty()) {
    record.resnr_rec = contact.resnr_rec;
    record.resnr_lig = contact.resnr_lig;
    record.chainid_rec = contact.chainid_rec;
    record.chainid_lig = contact.chainid_lig;
    record.res_name_rec = contact.res_name_rec;
    record.res_name_lig = contact.res_name_lig;
  }
  for (int bit = 0; bit < 6; ++bit) {
    if (contact.interactions[bit])
      record.counts[static_cast<size_t>(bit)]++;
  }
}

void TitanIFP::process_frame(
    const MOL &rec_mol, const MOL &lig_mol,
    const std::vector<mol_info> &rec_info,
    const std::vector<mol_info> &lig_info,
    const std::vector<class index> &rec_index,
    const std::vector<class index> &lig_index, size_t rec_total_atoms,
    size_t lig_total_atoms, const std::vector<d_min> &dis_min,
    const IfpArgs &args, TitanExecutor &executor, size_t requested_workers,
    bool build_awh_accumulators) {
  ++total_frames_;
  const auto prepared_pairs = build_prepared_pairs(
      dis_min, rec_index, lig_index, rec_total_atoms, lig_total_atoms);

  if (prepared_pairs.empty()) {
    last_frame_active_workers_ = 0;
    for (auto &frame_result : worker_frame_results_) {
      frame_result.contacts.clear();
      frame_result.rna_regions.clear();
    }
    for (auto &accumulator : worker_contact_accumulators_)
      accumulator.clear();
    return;
  }

  const size_t requested = std::max<size_t>(1, requested_workers);
  const size_t grain_limited = (prepared_pairs.size() + 63) / 64;
  const size_t active_workers =
      std::max<size_t>(1, std::min(executor.capacity(),
                                   std::min(requested, grain_limited)));
  ensure_worker_storage(active_workers);

  executor.parallel_for(
      prepared_pairs.size(), requested_workers, 64,
      [&](size_t worker_id, size_t begin, size_t end) {
        auto &frame_result = worker_frame_results_[worker_id];
        auto &contact_counts = worker_contact_accumulators_[worker_id];
        for (size_t pair_idx = begin; pair_idx < end; ++pair_idx) {
          const auto &pair = prepared_pairs[pair_idx];
          IFP_Contact result;
          std::array<bool, 3> region_flags{};
          const bool has_contact = analyze_prepared_pair(
              pair, rec_mol, lig_mol, rec_info, lig_info, args, result,
              region_flags);
          if (has_contact) {
            frame_result.contacts.push_back(result);
            if (build_awh_accumulators)
              accumulate_contact_counts(contact_counts, result);
          }

          if (pair.pair_has_nucleic &&
              (region_flags[kBaseRegion] || region_flags[kSugarRegion] ||
               region_flags[kPhosphateRegion])) {
            FrameRnaRegionContact region_contact;
            region_contact.resnr_rec = result.resnr_rec;
            region_contact.resnr_lig = result.resnr_lig;
            region_contact.chainid_rec = result.chainid_rec;
            region_contact.chainid_lig = result.chainid_lig;
            region_contact.res_name_rec = result.res_name_rec;
            region_contact.res_name_lig = result.res_name_lig;
            region_contact.flags = region_flags;
            frame_result.rna_regions.push_back(std::move(region_contact));
          }
        }
      });

  last_frame_active_workers_ = active_workers;
  for (size_t worker_id = 0; worker_id < active_workers; ++worker_id)
    accumulate_frame(worker_frame_results_[worker_id]);
}
void TitanIFP::aggregate_occupancy() {
  if (total_frames_ == 0)
    return;

  const double inv_frames = 100.0 / static_cast<double>(total_frames_);
  for (auto &[key, record] : occupancy_map_) {
    (void)key;
    for (size_t idx = 0; idx < record.occupancy.size(); ++idx)
      record.occupancy[idx] = static_cast<double>(record.counts[idx]) * inv_frames;
  }
  for (auto &[key, record] : rna_region_map_) {
    (void)key;
    for (size_t idx = 0; idx < record.occupancy.size(); ++idx)
      record.occupancy[idx] = static_cast<double>(record.counts[idx]) * inv_frames;
  }
}

void TitanIFP::export_csv(const std::string &filename) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_error("[IFP] Cannot open output file: " + filename);
    return;
  }

  out << "ChainID_Rec,ResID_Rec,ResName_Rec,ChainID_Lig,ResID_Lig,ResName_Lig,"
         "HBond(%),SaltBridge(%),PiPi(%),CationPi(%),Hydrophobic(%),Halogen(%)"
      << '\n';
  out << std::fixed << std::setprecision(1);
  for (const auto &[key, record] : occupancy_map_) {
    (void)key;
    out << record.chainid_rec << ',' << record.resnr_rec << ','
        << record.res_name_rec << ',' << record.chainid_lig << ','
        << record.resnr_lig << ',' << record.res_name_lig << ','
        << record.occupancy[BIT_HBOND] << ','
        << record.occupancy[BIT_SALT] << ','
        << record.occupancy[BIT_PI_PI] << ','
        << record.occupancy[BIT_CATION_PI] << ','
        << record.occupancy[BIT_HYDROPHOBIC] << ','
        << record.occupancy[BIT_HALOGEN] << '\n';
  }
  TitanUI::print_info("[IFP] Results written to " + filename + " (" +
                      std::to_string(occupancy_map_.size()) +
                      " residue pairs)");
}

void TitanIFP::export_rna_annotations_csv(const std::string &filename) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_error("[IFP] Cannot open RNA annotation file: " + filename);
    return;
  }

  out << "ChainID_Rec,ResID_Rec,ResName_Rec,ChainID_Lig,ResID_Lig,ResName_Lig,"
         "BaseContact(%),SugarContact(%),PhosphateContact(%),DominantRNARegion\n";
  out << std::fixed << std::setprecision(1);
  const auto dominant_region_label = [](const auto &record) {
    std::array<std::pair<double, const char *>, 3> ranked{{
        {record.occupancy[kBaseRegion], "base"},
        {record.occupancy[kSugarRegion], "sugar"},
        {record.occupancy[kPhosphateRegion], "phosphate"},
    }};
    std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.first > rhs.first;
    });
    if (ranked[0].first <= 0.0)
      return std::string("mixed");
    if (ranked[1].first > 0.0 && ranked[0].first < ranked[1].first * 1.5)
      return std::string("mixed");
    return std::string(ranked[0].second);
  };
  for (const auto &[key, record] : rna_region_map_) {
    (void)key;
    out << record.chainid_rec << ',' << record.resnr_rec << ','
        << record.res_name_rec << ',' << record.chainid_lig << ','
        << record.resnr_lig << ',' << record.res_name_lig << ','
        << record.occupancy[kBaseRegion] << ','
        << record.occupancy[kSugarRegion] << ','
        << record.occupancy[kPhosphateRegion] << ','
        << dominant_region_label(record) << '\n';
  }
  TitanUI::print_info("[IFP] RNA annotations written to " + filename + " (" +
                      std::to_string(rna_region_map_.size()) +
                      " residue pairs)");
}

#include "stat_hb_and_sc_contact.h"

#include "MOL.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <unordered_map>

namespace {

inline unsigned long long make_hb_residue_key(std::size_t chain_id,
                                              std::size_t res_num) noexcept {
  return (static_cast<unsigned long long>(chain_id) << 32) |
         static_cast<unsigned long long>(res_num);
}

res_hb &get_or_create_hb_residue(traj_hb &traj, const mol_info &atom_info) {
  const unsigned long long key =
      make_hb_residue_key(atom_info.get_chainnum(), atom_info.get_resnr());
  const auto found = traj.residue_index.find(key);
  if (found != traj.residue_index.end()) {
    return traj.residues_hb[found->second];
  }

  const std::size_t new_index = traj.residues_hb.size();
  traj.residues_hb.emplace_back(atom_info.get_original_res_name(),
                                atom_info.get_chain(), atom_info.get_resnr(), 0,
                                0, 0, atom_info.get_chainnum());
  traj.residue_index.emplace(key, new_index);
  return traj.residues_hb.back();
}

} // namespace

int traj_hb::add_hb(MOL &mol, vector<mol_info> &info_mol) {
  for (size_t j = 0; j < mol.list_h_bonds.dat.size(); ++j) {
    const size_t acceptor_index = mol.list_h_bonds.dat[j].get_index_acceptor();
    if (acceptor_index >= info_mol.size()) {
      continue;
    }
    res_hb &entry = get_or_create_hb_residue(*this, info_mol[acceptor_index]);
    ++entry.h_bond_num;
    ++entry.acceptor_hbond_num;
  }
  return 0;
}

int add_hb_res(MOL &pro_in, MOL &lig, vector<mol_info> &info_rec,
               vector<mol_info> &info_lg, const arguments &Arguments,
               traj_hb &res_hb_rec_temp) {
  (void)Arguments;
  (void)info_lg;

  res_hb_rec_temp.add_hb(pro_in, info_rec);

  for (size_t j = 0; j < lig.list_h_bonds.dat.size(); ++j) {
    const size_t donor_index = lig.list_h_bonds.dat[j].get_index_donor();
    if (donor_index >= info_rec.size()) {
      continue;
    }
    res_hb &entry = get_or_create_hb_residue(res_hb_rec_temp, info_rec[donor_index]);
    ++entry.h_bond_num;
    ++entry.donor_hbond_num;
  }
  return 0;
}

bool traj_hb::check_in(const res_hb &res_hbond, size_t &pos) {
  for (size_t i = 0; i < residues_hb.size(); ++i) {
    if (residues_hb[i].equal_to(res_hbond)) {
      pos = i;
      return true;
    }
  }
  return false;
}

int traj_hb::filt_hb(vector<traj_hb> &hb_reslist) {
  residues_hb.clear();
  residue_index.clear();
  snap_num = 0;

  for (size_t j = 0; j < hb_reslist.size(); ++j) {
    for (size_t k = 0; k < hb_reslist[j].residues_hb.size(); ++k) {
      const res_hb &source = hb_reslist[j].residues_hb[k];
      const unsigned long long key =
          make_hb_residue_key(source.chainid, source.resnum);
      const auto found = residue_index.find(key);
      if (found != residue_index.end()) {
        res_hb &target = residues_hb[found->second];
        target.h_bond_num += source.h_bond_num;
        target.acceptor_hbond_num += source.acceptor_hbond_num;
        target.donor_hbond_num += source.donor_hbond_num;
      } else {
        const std::size_t new_index = residues_hb.size();
        residues_hb.emplace_back(source);
        residue_index.emplace(key, new_index);
      }
    }
    snap_num += hb_reslist[j].snap_num;
  }
  return 0;
}

int stat_hb_res(vector<traj_hb> &hb_reslist, string output_filename) {
  traj_hb stat_out_hb;
  stat_out_hb.filt_hb(hb_reslist);

  std::sort(stat_out_hb.residues_hb.begin(), stat_out_hb.residues_hb.end(),
            [](const res_hb &lhs, const res_hb &rhs) {
              if (lhs.chainid != rhs.chainid) {
                return lhs.chainid < rhs.chainid;
              }
              if (lhs.resnum != rhs.resnum) {
                return lhs.resnum < rhs.resnum;
              }
              return lhs.resname < rhs.resname;
            });

  fstream output(output_filename, ios::out);
  output << "#Total frames for all trajectories: " << stat_out_hb.snap_num
         << " ; traj num: " << hb_reslist.size() << '\n';
  output << "#Chain\tResNum\tResName\tAvgHB\tAvgHB_Acceptor\tAvgHB_Donor\n";

  if (stat_out_hb.snap_num == 0) {
    return 0;
  }

  for (size_t i = 0; i < stat_out_hb.residues_hb.size(); ++i) {
    output << stat_out_hb.residues_hb[i].chain << '\t'
           << stat_out_hb.residues_hb[i].resnum << '\t'
           << stat_out_hb.residues_hb[i].resname << '\t' << std::fixed
           << std::setprecision(6)
           << stat_out_hb.residues_hb[i].h_bond_num * 1.0 /
                  stat_out_hb.snap_num
           << '\t'
           << stat_out_hb.residues_hb[i].acceptor_hbond_num * 1.0 /
                  stat_out_hb.snap_num
           << '\t'
           << stat_out_hb.residues_hb[i].donor_hbond_num * 1.0 /
                  stat_out_hb.snap_num
           << '\n';
  }
  return 0;
}

#ifndef STAT_HB_SC_H
#define STAT_HB_SC_H
#include "MOL.h"
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct res_sc {
  string resname{};
  char chain{};
  size_t resnum{}, sc_num{}, chainid{};
  res_sc() : resname{}, chain{'A'}, resnum{}, sc_num{}, chainid{} {}
  res_sc(string _resname, size_t _resnum, size_t _sc_num, size_t _chainid,
         char _chain)
      : resname{_resname}, chain{_chain}, resnum{_resnum}, sc_num{_sc_num},
        chainid{_chainid} {}
  inline bool operator<(const res_sc &rhs) const {
    return (resnum < rhs.resnum || resname < rhs.resname ||
            chainid < rhs.chainid);
  }
  inline bool operator>(const res_sc &rhs) const {
    return (resnum > rhs.resnum || resname > rhs.resname ||
            chainid > rhs.chainid);
  }
  inline bool operator==(const res_sc &rhs) const {
    return (resnum == rhs.resnum && resname == rhs.resname &&
            chainid == rhs.chainid);
  }
  inline bool equal_to(const res_sc &rhs) const {
    return chainid == rhs.chainid && resnum == rhs.resnum &&
           resname == rhs.resname;
  }
};

struct res_sc_hash {
  std::size_t operator()(res_sc const &inres_sc) const {
    std::size_t h1 = std::hash<string>()(inres_sc.resname);
    std::size_t h2 = std::hash<size_t>()(inres_sc.resnum);
    std::size_t h3 = std::hash<size_t>()(inres_sc.chainid);
    std::size_t h4 = std::hash<char>()(inres_sc.chain);

    return h1 ^ h2 ^ h3 ^ h4;
  }
};

struct traj_sc {
  vector<res_sc> residues_sc;
  // set<res_sc> set_residues_sc;
  unordered_map<res_sc, size_t, res_sc_hash> map_residues_sc;
  size_t snap_num{};
  traj_sc() = default;
  ~traj_sc() = default;
  traj_sc(string _resname, size_t _resnum, size_t _sc_num, size_t _chainid,
          char _chain) {
    residues_sc.emplace_back(_resname, _resnum, _sc_num, _chainid, _chain);
  }
  traj_sc(const traj_sc &rhs) : snap_num{rhs.snap_num} {
    residues_sc = rhs.residues_sc;
  }
  traj_sc(traj_sc &&rhs) : snap_num{rhs.snap_num} {
    residues_sc = std::move(rhs.residues_sc);
  }
  traj_sc &operator=(const traj_sc &rhs) {
    residues_sc = rhs.residues_sc;
    map_residues_sc = rhs.map_residues_sc;
    snap_num = rhs.snap_num;
    return *this;
  }
  traj_sc &operator=(traj_sc &&rhs) {
    residues_sc = std::move(rhs.residues_sc);
    map_residues_sc = std::move(rhs.map_residues_sc);
    snap_num = rhs.snap_num;
    return *this;
  }

  int add_sc(string _resname, size_t _resnum, size_t _sc_num, size_t _chainid,
             char _chain);
  int filt_sc(vector<traj_sc> &sc_reslist);
  bool check_in(const res_sc &res_contact, size_t &pos);
  void clear_frame() {
    residues_sc.clear();
    map_residues_sc.clear();
    snap_num = 0;
  }
  bool clear_map() {
    map_residues_sc.clear();
    return true;
  }
};

struct atom_hb {
  string atom_name{};
  bool equal_to(const atom_hb &rhs) { return (atom_name == rhs.atom_name); }
  bool operator!=(const atom_hb &rhs) const {
    return atom_name != rhs.atom_name;
  }
  bool operator==(const atom_hb &rhs) const {
    return atom_name == rhs.atom_name;
  }
};

struct res_hb {
  vector<atom_hb> atoms;
  string resname;
  string chain;
  size_t resnum, h_bond_num, acceptor_hbond_num, donor_hbond_num, chainid;
  res_hb()
      : resname{}, chain{}, resnum{}, h_bond_num{}, acceptor_hbond_num{},
        donor_hbond_num{}, chainid{} {}
  res_hb(string _resname, string _chain, size_t _resnum, size_t _hb_num,
         size_t _acc_hb_num, size_t _donor_hb_num,
         size_t _chainid)
      : resname{_resname}, chain{_chain}, resnum{_resnum}, h_bond_num{_hb_num},
        acceptor_hbond_num{_acc_hb_num}, donor_hbond_num{_donor_hb_num},
        chainid{_chainid} {}
  res_hb(string _resname, string _chain, size_t _resnum, size_t _hb_num,
         size_t _acc_hb_num, size_t _donor_hb_num, size_t _chainid,
         vector<atom_hb> &_atoms)
      : resname{_resname}, chain{_chain}, resnum{_resnum}, h_bond_num{_hb_num},
        acceptor_hbond_num{_acc_hb_num}, donor_hbond_num{_donor_hb_num},
        chainid{_chainid} {
    atoms = _atoms;
  }
  inline bool equal_to(const res_hb &rhs) const {
    return chainid == rhs.chainid && resnum == rhs.resnum;
  }

  bool operator==(const res_hb &rhs) const {
    if (chainid != rhs.chainid || resnum != rhs.resnum)
      return false;

    if (!rhs.atoms.empty() && !atoms.empty() &&
        rhs.atoms.size() == atoms.size()) {
      for (auto &a : atoms)
        for (auto &a_rhs : rhs.atoms)
          if (a != a_rhs)
            return false;
      return true;
    } else
      return false;
  }
};

struct traj_hb {
  vector<res_hb> residues_hb;
  unordered_map<unsigned long long, size_t> residue_index;
  size_t snap_num{};
  traj_hb() = default;
  traj_hb(string _resname, string _chain, size_t _resnum, size_t _hb_num,
          size_t _chainid) {
    residues_hb.emplace_back(_resname, _chain, _resnum, _hb_num, _hb_num, 0,
                             _chainid);
    residue_index.emplace((static_cast<unsigned long long>(_chainid) << 32) |
                              static_cast<unsigned long long>(_resnum),
                          0);
  }
  traj_hb(const traj_hb &rhs) : snap_num{rhs.snap_num} {
    residues_hb = rhs.residues_hb;
    residue_index = rhs.residue_index;
  }
  traj_hb(traj_hb &&rhs) : snap_num{rhs.snap_num} {
    residues_hb = std::move(rhs.residues_hb);
    residue_index = std::move(rhs.residue_index);
  }
  traj_hb &operator=(const traj_hb &rhs) {
    residues_hb = rhs.residues_hb;
    residue_index = rhs.residue_index;
    snap_num = rhs.snap_num;
    return *this;
  }
  traj_hb &operator=(traj_hb &&rhs) {
    residues_hb = std::move(rhs.residues_hb);
    residue_index = std::move(rhs.residue_index);
    snap_num = rhs.snap_num;
    return *this;
  }
  ~traj_hb() = default;
  int filt_hb(vector<traj_hb> &hb_reslist);
  int add_hb(MOL &mol, vector<mol_info> &info_mol);
  bool check_in(const res_hb &res_hbond, size_t &pos);
  void clear_frame() {
    residues_hb.clear();
    residue_index.clear();
    snap_num = 0;
  }
};

struct map_pos {
  string chain_1{}, chain_2{};
  size_t chain_1id{}, chain_2id{};
  size_t resnum_1{}, resnum_2{}, cont_num{}, snap_num{};
  map_pos() = default;
  ~map_pos() = default;
  map_pos(const map_pos &rhs)
      : chain_1{rhs.chain_1}, chain_2{rhs.chain_2}, chain_1id{rhs.chain_1id},
        chain_2id{rhs.chain_2id}, resnum_1{rhs.resnum_1},
        resnum_2{rhs.resnum_2}, cont_num{rhs.cont_num}, snap_num{rhs.snap_num} {
  }
  map_pos(const map_pos &&rhs)
      : chain_1{rhs.chain_1}, chain_2{rhs.chain_2}, chain_1id{rhs.chain_1id},
        chain_2id{rhs.chain_2id}, resnum_1{rhs.resnum_1},
        resnum_2{rhs.resnum_2}, cont_num{rhs.cont_num}, snap_num{rhs.snap_num} {
  }
  map_pos &operator=(const map_pos &rhs) {
    chain_1 = rhs.chain_1;
    chain_2 = rhs.chain_2;
    chain_1id = rhs.chain_1id;
    chain_2id = rhs.chain_2id;
    resnum_1 = rhs.resnum_1;
    resnum_2 = rhs.resnum_2;
    cont_num = rhs.cont_num;
    snap_num = rhs.snap_num;
    return *this;
  }
  map_pos &operator=(const map_pos &&rhs) {
    chain_1 = rhs.chain_1;
    chain_2 = rhs.chain_2;
    chain_1id = rhs.chain_1id;
    chain_2id = rhs.chain_2id;
    resnum_1 = rhs.resnum_1;
    resnum_2 = rhs.resnum_2;
    cont_num = rhs.cont_num;
    snap_num = rhs.snap_num;
    return *this;
  }
};

#endif // !STAT_HB_SC_H

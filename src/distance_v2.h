#ifndef DISTANCE_H
#define DISTANCE_H
#include <string>
#include <utility>
#include <vector>

struct dmin_base {
  // ── Data members (declaration order = init-list order) ──
  size_t resnr_rec{}, resnr_ab{};
  size_t atom_num{}, atom_num_rec{};
  size_t chainid_rec{}, chainid_ab{};
  size_t respos_rec{};
  double d{};
  std::string chain_rec, chain_ab;
  std::string res_name_rec, res_name_ab;

  // Default constructor
  dmin_base() noexcept = default;

  // Parameterized constructor (init-list follows declaration order)
  template <typename _d, typename _chain, typename _chain2, typename _res1,
            typename _res2, typename _num, typename _num2, typename _num3,
            typename _num4, typename _numchainid, typename _numchainid2,
            typename _numchainid3>
  dmin_base(_d &&_dis, _chain &&_chain_ab, _chain2 &&_chain_rec,
            _num &&_resnr_ab, _num2 &&_resnr_rec, _num3 &&_atom_num,
            _num4 &&_atom_num_rec, _res1 &&_resname_rec, _res2 &&_resname_ab,
            _numchainid &&_chainid_rec, _numchainid2 &&_chainid_ab,
            _numchainid3 &&_respos_rec)
      : resnr_rec{static_cast<size_t>(_resnr_rec)},
        resnr_ab{static_cast<size_t>(_resnr_ab)},
        atom_num{static_cast<size_t>(_atom_num)},
        atom_num_rec{static_cast<size_t>(_atom_num_rec)},
        chainid_rec{static_cast<size_t>(_chainid_rec)},
        chainid_ab{static_cast<size_t>(_chainid_ab)},
        respos_rec{static_cast<size_t>(_respos_rec)},
        d{static_cast<double>(_dis)},
        chain_rec{std::forward<_chain2>(_chain_rec)},
        chain_ab{std::forward<_chain>(_chain_ab)},
        res_name_rec{std::forward<_res1>(_resname_rec)},
        res_name_ab{std::forward<_res2>(_resname_ab)} {}

  ~dmin_base() = default;

  // ── Comparison operators (lexicographic) ──
  // Sort by: chainid_rec → chainid_ab → resnr_rec → resnr_ab → respos_rec
  //          → atom_num_rec → atom_num
  inline bool operator<(const dmin_base &rhs) const {
    if (chainid_rec != rhs.chainid_rec)
      return chainid_rec < rhs.chainid_rec;
    if (chainid_ab != rhs.chainid_ab)
      return chainid_ab < rhs.chainid_ab;
    if (resnr_rec != rhs.resnr_rec)
      return resnr_rec < rhs.resnr_rec;
    if (resnr_ab != rhs.resnr_ab)
      return resnr_ab < rhs.resnr_ab;
    if (respos_rec != rhs.respos_rec)
      return respos_rec < rhs.respos_rec;
    if (atom_num_rec != rhs.atom_num_rec)
      return atom_num_rec < rhs.atom_num_rec;
    return atom_num < rhs.atom_num;
  }

  inline bool operator>(const dmin_base &rhs) const { return rhs < *this; }

  inline bool operator==(const dmin_base &rhs) const {
    return chainid_rec == rhs.chainid_rec && chainid_ab == rhs.chainid_ab &&
           resnr_ab == rhs.resnr_ab && resnr_rec == rhs.resnr_rec &&
           atom_num == rhs.atom_num && atom_num_rec == rhs.atom_num_rec &&
           res_name_rec == rhs.res_name_rec && res_name_ab == rhs.res_name_ab &&
           respos_rec == rhs.respos_rec;
  }
};

class d_min {
  static constexpr int DEFAULT_RESERVE = 10000;

public:
  d_min() { base.reserve(DEFAULT_RESERVE); }
  std::vector<dmin_base> base;
  ~d_min() = default;
};

#endif

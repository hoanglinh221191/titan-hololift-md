#ifndef MISCELL_H
#define MISCELL_H
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// #include"MOL.h"
#include "force.h"
#include "uti_func.h"

using namespace std;

struct hbonds_pos {
  size_t index_mol;
  vector<size_t> index_rep;
  template <typename T, typename T2>
  hbonds_pos(T &&index_mol, T2 &&index_rep_)
      : index_mol{static_cast<size_t>(index_mol)} {
    index_rep.emplace_back(index_rep_);
  }
};

struct data_out {
  size_t resnr{};
  double E_E{}, E_V{};
  string chain{}, resname{};
  force F_elec{}, F_vdw{};
  data_out()
      : resnr{}, E_E{}, E_V{}, chain{" "}, resname{}, F_elec{}, F_vdw{} {}
  // data_out(size_t _resnr, string ch, double ee, double ev) :resnr{ _resnr },
  // E_E{ ee }, E_V{ ev }, chain{ ch } {} data_out(size_t _resnr, string ch,
  // double ee, double ev, force& felec, force& fvdw) 	:resnr{ _resnr }, E_E{
  // ee
  //}, E_V{ ev }, chain{ ch }, F_elec {felec}, F_vdw{fvdw} {}
  data_out(size_t _resnr, string ch, string rname, double ee, double ev)
      : resnr{_resnr}, E_E{ee}, E_V{ev}, chain{ch}, resname{rname} {}
  data_out(size_t _resnr, string ch, string rname, double ee, double ev,
           force &felec, force &fvdw)
      : resnr{_resnr}, E_E{ee}, E_V{ev}, chain{ch}, resname{rname},
        F_elec{felec}, F_vdw{fvdw} {}
  data_out &operator=(const data_out &rhs) {
    resnr = rhs.resnr;
    E_E = rhs.E_E;
    E_V = rhs.E_V;
    chain = rhs.chain;
    resname = rhs.resname;
    F_elec = rhs.F_elec;
    F_vdw = rhs.F_vdw;
    return *this;
  }
  ~data_out() = default;
};
struct energy_res_base {
  double e_e, e_v, time;
  force F_elec, F_vdw;

  energy_res_base() : e_e{}, e_v{}, time{}, F_elec(), F_vdw() {}
  energy_res_base(const energy_res_base &rhs)
      : e_e{rhs.e_e}, e_v{rhs.e_v}, time{rhs.time}, F_elec{rhs.F_elec},
        F_vdw{rhs.F_vdw} {}
  energy_res_base(energy_res_base &&rhs)
      : e_e{rhs.e_e}, e_v{rhs.e_v}, time{rhs.time}, F_elec{rhs.F_elec},
        F_vdw{rhs.F_vdw} {}
  energy_res_base &operator=(const energy_res_base &rhs) {
    e_e = rhs.e_e;
    e_v = rhs.e_v;
    time = rhs.time;
    F_elec = rhs.F_elec;
    F_vdw = rhs.F_vdw;
    return *this;
  }
  energy_res_base &operator=(energy_res_base &&rhs) {
    e_e = rhs.e_e;
    e_v = rhs.e_v;
    time = rhs.time;
    F_elec = rhs.F_elec;
    F_vdw = rhs.F_vdw;
    return *this;
  }
  template <typename T>
  energy_res_base(T &&_e, T &&_v)
      : e_e{std::forward<T>(_e)}, e_v{std::forward<T>(_v)} {}
  template <typename T>
  energy_res_base(T &&_e, T &&_v, force &_f_e, force &_f_v)
      : e_e{std::forward<T>(_e)}, e_v{std::forward<T>(_v)}, F_elec(_f_e),
        F_vdw(_f_v) {}
  template <typename T, typename T2>
  energy_res_base(T &&_e, T &&_v, T2 &&_t)
      : e_e{std::forward<T>(_e)}, e_v{std::forward<T>(_v)},
        time{std::forward<T2>(_t)} {}
  template <typename T, typename T2>
  energy_res_base(T &&_e, T &&_v, force &_f_e, force &_f_v, T2 &&_t)
      : e_e{std::forward<T>(_e)}, e_v{std::forward<T>(_v)},
        time{std::forward<T2>(_t)}, F_elec(_f_e), F_vdw(_f_v) {}
  energy_res_base &operator+=(const energy_res_base &rhs) {
    e_e += rhs.e_e;
    e_v += rhs.e_v;
    F_elec += rhs.F_elec;
    F_vdw += rhs.F_vdw;
    // time += rhs.time;
    return *this;
  }
  ~energy_res_base() = default;
};
struct energy_res {
  vector<energy_res_base> base;
  size_t resnr{};
  string chain{}, resname{};

  energy_res() : resnr{}, chain{}, resname{} { base.reserve(50000); }
  energy_res(size_t _res_nr, string _chain, energy_res_base &_base)
      : resnr{_res_nr}, chain{_chain} {
    base.emplace_back(_base);
  }
  energy_res(size_t _res_nr, string _chain, string _resname,
             energy_res_base &_base)
      : resnr{_res_nr}, chain{_chain}, resname{_resname} {
    base.emplace_back(_base);
  }
  energy_res(const energy_res &rhs)
      : resnr{rhs.resnr}, chain{rhs.chain}, resname{rhs.resname} {
    base = rhs.base;
  }
  energy_res(energy_res &&rhs)
      : resnr{rhs.resnr}, chain{rhs.chain}, resname{rhs.resname} {
    base = std::move(rhs.base);
  }
  bool emplace(energy_res_base &_base) {
    base.emplace_back(_base);
    return true;
  }
  template <typename T> void emplace_back(T &&_e, T &&_v) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v));
  }
  template <typename T>
  void emplace_back(T &&_e, T &&_v, force &_f_e, force &_f_v) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v);
  }
  template <typename T, typename T2> void emplace_back(T &&_e, T &&_v, T2 &&_t) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v),
                      std::forward<T2>(_t));
  }
  template <typename T, typename T2>
  void emplace_back(T &&_e, T &&_v, force &_f_e, force &_f_v, T2 &&_t) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v,
                      std::forward<T2>(_t));
  }

  template <typename T, typename T2>
  energy_res(size_t _resnr, string _chain, string _resname, T &&_e, T &&_v,
             T2 &&_t)
      : resnr(_resnr), chain(_chain), resname(_resname) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v),
                      std::forward<T2>(_t));
  }

  template <typename T, typename T2>
  energy_res(size_t _resnr, string _chain, string _resname, T &&_e, T &&_v,
             force &_f_e, force &_f_v, T2 &&_t)
      : resnr(_resnr), chain(_chain), resname(_resname) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v,
                      std::forward<T2>(_t));
  }
  bool operator==(const energy_res &rhs) const {
    return resnr == rhs.resnr && chain == rhs.chain;
  }
  bool operator<(const energy_res &rhs) const {
    return resnr < rhs.resnr && chain < rhs.chain;
  }
  ~energy_res() = default;
};

struct energy_groups {
  vector<energy_res_base> base;
  string group_name;
  energy_groups() : group_name{"unknown"} { base.reserve(50000); }
  energy_groups(energy_res_base &rhs, string name_group1, string name_group2)
      : group_name{name_group1 + "-" + name_group2} {
    base.emplace_back(rhs);
  }
  template <typename T> void emplace_back(T &&_e, T &&_v) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v));
  }
  template <typename T>
  void emplace_back(T &&_e, T &&_v, force &_f_e, force &_f_v) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v);
  }
  template <typename T, typename T2> void emplace_back(T &&_e, T &&_v, T2 &&_t) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v),
                      std::forward<T2>(_t));
  }
  template <typename T, typename T2>
  void emplace_back(T &&_e, T &&_v, force &_f_e, force &_f_v, T2 &&_t) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v,
                      std::forward<T2>(_t));
  }
  template <typename T, typename T2>
  energy_groups(string name, T &&_e, T &&_v, T2 &&_t) : group_name(name) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v),
                      std::forward<T2>(_t));
  }
  template <typename T, typename T2>
  energy_groups(string name, T &&_e, T &&_v, force &_f_e, force &_f_v, T2 &&_t)
      : group_name(name) {
    base.emplace_back(std::forward<T>(_e), std::forward<T>(_v), _f_e, _f_v,
                      std::forward<T2>(_t));
  }
  int push(energy_res_base &rhs, string name_group1, string name_group2) {
    base.emplace_back(rhs);
    group_name = name_group1 + "-" + name_group2;
    return 0;
  }
  ~energy_groups() = default;
};

struct topo_info_temp {
  string atom_name{}, atom_type{}, res_name{};
  double atom_charge{}, atom_mass{1.};
  int res_nr{-1};
  topo_info_temp() = default;
  topo_info_temp(string a_name, string a_type, string r_name, double a_charge,
                 double a_mass, int r_nr)
      : atom_name{a_name}, atom_type{a_type}, res_name{r_name},
        atom_charge{a_charge}, atom_mass{a_mass}, res_nr{r_nr} {}
};

struct point {
  double x{}, y{}, z{};
  point() = default;
  point &operator+=(const point &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
  }

  point &operator/=(double m) {
    x /= m;
    y /= m;
    z /= m;
    return *this;
  }

  point operator*(double m) {
    point temp{};
    temp.x = x * m;
    temp.y = y * m;
    temp.z = z * m;
    return temp;
  }

  point(double _x, double _y, double _z) : x{_x}, y{_y}, z{_z} {}
};

struct multi_points {
  vector<point> points;
  vector<double> mass;
  vector<string> name;
  int emplace(point _p, double _mass, string _name) {
    points.emplace_back(_p);
    mass.emplace_back(_mass);
    name.emplace_back(_name);
    return 0;
  }
};

struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(std::pair<T1, T2> const &pair) const {
    // std::size_t h1 = std::hash<T1>()(pair.first);
    // std::size_t h2 = std::hash<T2>()(pair.second);

    // return h1 ^ h2;
    size_t hash1 = std::hash<T1>{}(pair.first);
    size_t hash2 = std::hash<T2>{}(pair.second);
    return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
  }
};

struct tuple3_hash {
  template <class T1, class T2, class T3>
  std::size_t operator()(std::tuple<T1, T2, T3> const &tupl) const {
    std::size_t h1 = std::hash<T1>()(get<0>(tupl));
    std::size_t h2 = std::hash<T2>()(get<1>(tupl));
    std::size_t h3 = std::hash<T3>()(get<2>(tupl));

    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

struct tuple_hash {
  template <class T1, class T2, class T3, class T4>
  std::size_t operator()(std::tuple<T1, T2, T3, T4> const &tupl) const {
    std::size_t h1 = std::hash<T1>()(get<0>(tupl));
    std::size_t h2 = std::hash<T2>()(get<1>(tupl));
    std::size_t h3 = std::hash<T3>()(get<2>(tupl));
    std::size_t h4 = std::hash<T4>()(get<3>(tupl));

    h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    h1 ^= h4 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
    return h1;
  }
};

struct mindistance_group {
  string group1name, group2name;
  vector<vector<pair<double, double>>> data;
  mindistance_group() = default;
  ~mindistance_group() = default;
  mindistance_group(string _gname1, string _gname2,
                    vector<pair<double, double>> &dat)
      : group1name{_gname1}, group2name{_gname2} {
    data.emplace_back(dat);
  }
  mindistance_group(mindistance_group &rhs)
      : group1name{rhs.group1name}, group2name{rhs.group2name}, data{rhs.data} {
  }
  mindistance_group(mindistance_group &&rhs) noexcept
      : group1name{rhs.group1name}, group2name{rhs.group2name}, data{rhs.data} {
  }
  mindistance_group &operator=(mindistance_group &rhs) {
    group1name = rhs.group1name;
    group2name = rhs.group2name;
    data = rhs.data;
    return *this;
  }
  mindistance_group &operator=(mindistance_group &&rhs) noexcept {
    group1name = rhs.group1name;
    group2name = rhs.group2name;
    data = rhs.data;
    return *this;
  }
};
#endif

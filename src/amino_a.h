#ifndef AMINO_H
#define AMINO_H
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

struct amino_hash {
  template <class T> std::size_t operator()(T &tupl) const {
    std::size_t h1 = std::hash<T>()(tupl);

    return h1;
  }
};

struct info_aa {
  int charge_group;
  double charge, atom_mass, epsilon, sigma, radius;
  string atom, atom_type;
  info_aa()
      : charge_group{1}, charge{}, atom_mass{1.}, epsilon{}, sigma{}, radius{},
        atom{"NULL"}, atom_type{"NULL"} {}
  info_aa(string _atom, string _atom_type, double _charge)
      : charge_group{1}, charge{_charge}, atom_mass{1.}, epsilon{}, sigma{},
        radius{}, atom{_atom}, atom_type{_atom_type} {}
  info_aa(int _charge_group, double _charge, double _atom_mass, double _epsilon,
          double _sigma, double _radius, string _atom, string _atom_type)
      : charge_group{_charge_group}, charge{_charge}, atom_mass{_atom_mass},
        epsilon{_epsilon}, sigma{_sigma}, radius{_radius}, atom{_atom},
        atom_type{_atom_type} {}
};

class amino_acid {
public:
  unordered_map<string, vector<info_aa>> amino_acids;
  bool check_res_name(string res_name) const {
    if (amino_acids.find(res_name) != amino_acids.end())
      return true;
    else
      return false;
  }
};
#endif
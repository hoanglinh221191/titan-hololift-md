#ifndef C6C12_H
#define C6C12_H
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
struct para_base {
  double c6_sigma, c12_epsilon, mass, radius, charge;
  string atom_type, code_name;
  para_base()
      : c6_sigma{}, c12_epsilon{}, mass{1.0}, radius{}, charge{}, code_name{} {}
  template <typename T1, typename T11, typename T2, typename T3>
  para_base(T1 &&_c6_sigma, T1 &&_c12_epsilon, T11 &&_mass, T3 &&_radius,
            T2 &&_atom_type)
      : c6_sigma{std::forward<T1>(_c6_sigma)},
        c12_epsilon{std::forward<T1>(_c12_epsilon)},
        mass{std::forward<T11>(_mass)}, radius{std::forward<T1>(_radius)},
        atom_type{std::forward<T2>(_atom_type)} {}

  template <typename T1, typename T11, typename T2, typename T3>
  para_base(T1 &&_c6_sigma, T1 &&_c12_epsilon, T11 &&_mass, T3 &&_radius,
            T1 &&_charge, T2 &&_atom_type, T2 &&_code_name)
      : c6_sigma{std::forward<T1>(_c6_sigma)},
        c12_epsilon{std::forward<T1>(_c12_epsilon)},
        mass{std::forward<T11>(_mass)}, radius{std::forward<T1>(_radius)},
        charge{_charge}, atom_type{std::forward<T2>(_atom_type)},
        code_name{_code_name} {}
};
class para {
public:
  struct nonbond_pair_override {
    string atom_type_i;
    string atom_type_j;
    double sigma{};
    double epsilon{};
  };

  vector<para_base> base;
  std::unordered_map<string, size_t> atom_type_to_id;
  vector<nonbond_pair_override> nonbond_overrides;
  para() { base.reserve(200); }

  void register_atom_type(const string &atom_type) {
    if (!atom_type.empty())
      atom_type_to_id.insert_or_assign(atom_type, base.size() - 1);
  }

  void add_nonbond_override(string atom_type_i, string atom_type_j,
                            double sigma, double epsilon) {
    if (atom_type_i.empty() || atom_type_j.empty())
      return;

    if (atom_type_j < atom_type_i)
      std::swap(atom_type_i, atom_type_j);

    for (auto &item : nonbond_overrides) {
      if (item.atom_type_i == atom_type_i && item.atom_type_j == atom_type_j) {
        item.sigma = sigma;
        item.epsilon = epsilon;
        return;
      }
    }

    nonbond_overrides.push_back({std::move(atom_type_i), std::move(atom_type_j),
                                 sigma, epsilon});
  }
};
#endif

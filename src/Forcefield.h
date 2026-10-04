#ifndef FORCEFIELD_H
#define FORCEFIELD_H
#include <string>
using std::string;
class forcefield {
public:
  string atom_type;
  double sigma, epsilon;
  double atom_mass;
  forcefield() : atom_type{"NULL"}, sigma{0.0}, epsilon{0.0}, atom_mass{1.0} {}
};
#endif
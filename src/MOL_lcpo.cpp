#include "MOL.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;
#include "Multi_MOL.h"

using namespace std;

void Multi_MOL::assign_lcpo_parameters(size_t num_atoms,
                                       const vector<string> &atom_names,
                                       const vector<string> &res_names) {
  // Dùng resize và std::fill để đảm bảo an toàn kích thước mảng
  lcpo_p1.resize(num_atoms, 1.0);
  lcpo_p2.resize(num_atoms, -0.33);
  lcpo_p3.resize(num_atoms, 0.0);
  lcpo_p4.resize(num_atoms, 0.0);

  for (size_t i = 0; i < num_atoms; ++i) {
    if (atom_names[i].empty())
      continue;

    string name = atom_names[i];
    // Future expansion for p1..p4 heuristic could go here
  }
}
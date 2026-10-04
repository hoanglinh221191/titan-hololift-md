#include "MOL.h"
#include "functions.h"
#include <cmath>
#include <vector>

using namespace std;
double d_standard = 3.0;
double min_dis(const MOL &mol_1, const MOL &mol_2, int coor) {
  if (coor == 0) {
    double min(1000.0);
    for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
      for (size_t j = 0; j < mol_2.pos.x.size(); ++j) {
        if (fabs(min) >= (fabs(mol_1.pos.x[i] - mol_2.pos.x[j])))
          min = mol_1.pos.x[i] - mol_2.pos.x[j];
      }
    }
    return min;
  } else if (coor == 1) {
    double min(1000.0);
    for (size_t i = 0; i < mol_1.pos.y.size(); ++i) {
      for (size_t j = 0; j < mol_2.pos.y.size(); ++j) {
        if (fabs(min) >= (fabs(mol_1.pos.y[i] - mol_2.pos.y[j])))
          min = mol_1.pos.y[i] - mol_2.pos.y[j];
      }
    }
    return min;
  } else if (coor == 2) {
    double min(1000.0);
    for (size_t i = 0; i < mol_1.pos.z.size(); ++i) {
      for (size_t j = 0; j < mol_2.pos.z.size(); ++j) {
        if (fabs(min) >= fabs(mol_1.pos.z[i] - mol_2.pos.z[j])) {
          min = mol_1.pos.z[i] - mol_2.pos.z[j];
        }
      }
    }
    return min;
  } else
    return 1;
}
void Fix_pbc(MOL &mol_1, int coor, const double &min_dis, double size) {
  // cout << coor;
  if (coor == 0) {
    if (min_dis > 0)
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] -= size;
      }
    else
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] += size;
      }
  } else if (coor == 1) {
    if (min_dis > 0)
      for (size_t i = 0; i < mol_1.pos.y.size(); ++i) {
        mol_1.pos.y[i] -= size;
      }
    else
      for (size_t i = 0; i < mol_1.pos.y.size(); ++i) {
        mol_1.pos.y[i] += size;
      }
  } else if (coor == 2) {
    if (min_dis > 0)
      for (size_t i = 0; i < mol_1.pos.z.size(); ++i) {
        mol_1.pos.z[i] -= size;
      }
    else
      for (size_t i = 0; i < mol_1.pos.z.size(); ++i) {
        mol_1.pos.z[i] += size;
      }
  }
}
void solve_pbc(MOL &mol_1, MOL &mol_2, box &box_type) {
  if (box_type.get_box_type() == "cubic") {
    double size = box_type.get_a();
    for (int i = 0; i < 3; ++i) {
      if ((fabs(min_dis(mol_1, mol_2, i))) >= d_standard) {
        Fix_pbc(mol_1, i, min_dis(mol_1, mol_2, i), size);
        bool boo(fabs(min_dis(mol_1, mol_2, i)) >= 0.25 * size);
      } else
        continue;
    }
  } else if (box_type.get_box_type() == "rectangular") {
    double size_temp[3] = {box_type.get_a(), box_type.get_b(),
                           box_type.get_c()};
    for (int i = 0; i < 3; ++i) {
      if ((fabs(min_dis(mol_1, mol_2, i))) >= d_standard) {
        Fix_pbc(mol_1, i, min_dis(mol_1, mol_2, i), size_temp[i]);
        bool boo(fabs(min_dis(mol_1, mol_2, i)) >= 0.25 * size_temp[i]);
      } else
        continue;
    }
  } else if (box_type.get_box_type() == "dodecahedron") {

    double min{};
    {
      double temp = sqrt(2.0 * pow(box_type.get_c() * 0.5, 2.0));
      double dis_z = sqrt(pow(box_type.get_c(), 2.0) - pow(temp, 2.0));
      min = min_dis(mol_1, mol_2, 2);
      if (fabs(min) >= d_standard * dis_z / box_type.get_a()) {
        fix_dodecahedron(mol_1, mol_2, box_type);
        Fix_pbc(mol_1, 2, min, dis_z);
      }
    }
    {
      min = min_dis(mol_1, mol_2, 0);
      if (fabs(min) >= d_standard) {
        Fix_pbc(mol_1, 0, min, box_type.get_a());
      }
    }
    {
      min = min_dis(mol_1, mol_2, 1);
      if (fabs(min) >= d_standard) {
        Fix_pbc(mol_1, 1, min, box_type.get_b());
      }
    }
  }
}
void fix_dodecahedron(MOL &mol_1, MOL &mol_2, box &box_type) {
  int flag[2][4] = {};
  double test_flag[2][4] = {};
  for (size_t i = 0; i < mol_2.pos.x.size(); ++i) {
    // x
    if (mol_2.pos.x[i] > box_type.get_a())
      test_flag[0][3] += 1.0;
    else if (mol_2.pos.x[i] > 0.5 * box_type.get_a())
      test_flag[0][2] += 1.0;
    else if (mol_2.pos.x[i] > 0.0)
      test_flag[0][1] += 1.0;
    else
      test_flag[0][0] += 1.0;
    // y
    if (mol_2.pos.y[i] > box_type.get_b())
      test_flag[1][3] += 1.0;
    else if (mol_2.pos.y[i] > 0.5 * box_type.get_b())
      test_flag[1][2] += 1.0;
    else if (mol_2.pos.y[i] > 0.0)
      test_flag[1][1] += 1.0;
    else
      test_flag[1][0] += 1.0;
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      if (test_flag[i][j] > 0.5 * mol_2.pos.x.size())
        flag[i][j] = 1;

  if (flag[0][3] == 1 || flag[0][2] == 1)
    if (flag[1][3] == 1 || flag[1][1] == 1)
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] += box_type.get_a() / 2;
        mol_1.pos.y[i] -= box_type.get_b() / 2;
      }
    else
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] += box_type.get_a() / 2;
        mol_1.pos.y[i] += box_type.get_b() / 2;
      }
  else if (flag[0][1] == 1 || flag[0][0] == 1)
    if (flag[1][3] == 1 || flag[1][1] == 1)
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] -= box_type.get_a() / 2;
        mol_1.pos.y[i] -= box_type.get_b() / 2;
      }
    else
      for (size_t i = 0; i < mol_1.pos.x.size(); ++i) {
        mol_1.pos.x[i] -= box_type.get_a() / 2;
        mol_1.pos.y[i] += box_type.get_b() / 2;
      }
}

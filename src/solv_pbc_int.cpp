#include "functions.h"
#include <algorithm>
#include <cmath>
#include <iostream>

const double par = 0.14;
const double par_2 = 0.7;
void pbc_multi_snap(Multi_MOL &multi_mol, size_t begin, size_t end);
void pbc_multi_chain(MOL &mol, vector<pair<size_t, size_t>> &neighbors,
                     vector<class index> &Index);
void pbc_chain(MOL &mol, size_t ind, vector<class index> &Index);
int multi_chain(MOL &mol, size_t ind, vector<pair<size_t, size_t>> &neighbors,
                vector<class index> &Index);
void fix_pair_z(MOL &mol, size_t atom1, size_t atom2);
void fix_pair_y(MOL &mol, size_t atom1, size_t atom2);
void fix_pair_x(MOL &mol, size_t atom1, size_t atom2);
int fix_pair_coor(MOL &mol, size_t atom1, size_t atom2) {
  fix_pair_z(mol, atom1, atom2);
  fix_pair_y(mol, atom1, atom2);
  fix_pair_x(mol, atom1, atom2);
  return 0;
}
int Multi_MOL::solve_pbc_int(size_t init, size_t nt) {
  pbc_multi_snap(*this, init * m_mol.size() / nt,
                 (init + 1) * m_mol.size() / nt);
  return 0;
}
void pbc_multi_snap(Multi_MOL &multi_mol, size_t begin, size_t end) {
  vector<pair<size_t, size_t>> neighbors;
  for (size_t i{begin}; i < end; ++i)
    pbc_multi_chain(multi_mol.m_mol[i], neighbors, multi_mol.Index);
}
void pbc_multi_chain(MOL &mol, vector<pair<size_t, size_t>> &neighbors,
                     vector<class index> &Index) {
  for (size_t i{}; i < Index.size() - 1; ++i) {
    pbc_chain(mol, i, Index);
    multi_chain(mol, i, neighbors, Index);
  }
}
void pbc_chain(MOL &mol, size_t ind, vector<class index> &Index) {
  for (size_t j{Index[ind].get_res_pos_e(0)};
       j < Index[ind + 1].get_res_pos_e(0); ++j) {
    fix_pair_coor(mol, j, j + 1);
  }
}
int multi_chain(MOL &mol, size_t ind, vector<pair<size_t, size_t>> &neighbors,
                vector<class index> &Index) {
  if (ind == Index.size() - 2)
    return 0;
  double v1[3]{}, v2[3]{}, v3[3]{};
  mol.Box.get_v1(v1);
  mol.Box.get_v2(v2);
  mol.Box.get_v3(v3);
  size_t index_chain_1{}, index_chain_2{};
  double crit_dis_x{par * v1[0]};
  double crit_dis_y{par * v2[1]};
  double crit_dis_z{par * v3[2]};
  size_t ii[4]{};
  for (int ite{}; ite < 2; ++ite) {
    double X_min{100000.}, Y_min{100000.}, Z_min{100000.}, d_min{100000.};
    // double x, y, z;
    for (size_t j{Index[ind].get_res_pos_e(0) - 1};
         j < Index[ind + 1].get_res_pos_e(0) - 1; ++j) {
      for (size_t k{Index[ind + 1].get_res_pos_e(0) - 1};
           k < Index[ind + 2].get_res_pos_e(0) - 1; ++k) {
        double d{sqrt(square(mol.pos.x[j] - mol.pos.x[k]) +
                      square(mol.pos.y[j] - mol.pos.y[k]) +
                      square(mol.pos.z[j] - mol.pos.z[k]))};
        if (d < d_min) {
          d_min = d;
          ii[0] = j;
        }
        if (fabs(mol.pos.x[j] - mol.pos.x[k]) < X_min) {
          X_min = fabs(mol.pos.x[j] - mol.pos.x[k]);
          ii[1] = j;
        }
        if (fabs(mol.pos.y[j] - mol.pos.y[k]) < Y_min) {
          Y_min = fabs(mol.pos.y[j] - mol.pos.y[k]);
          ii[2] = j;
        }
        if (fabs(mol.pos.z[j] - mol.pos.z[k]) < Z_min) {
          Z_min = fabs(mol.pos.z[j] - mol.pos.z[k]);
          ii[3] = j;
        }
      }
    }

    if (Z_min > crit_dis_z) {
      if (d_min > par_2 * sqrt(square(crit_dis_x) + square(crit_dis_y) +
                               square(crit_dis_z)))
        for (size_t i{Index[ind + 1].get_res_pos_e(0) - 1};
             i < Index[ind + 2].get_res_pos_e(0) - 1; ++i) {
          fix_pair_z(mol, ii[1], i);
        }
    } else if (Y_min > crit_dis_y) {
      if (d_min > par_2 * sqrt(square(crit_dis_x) + square(crit_dis_y) +
                               square(crit_dis_z)))
        for (size_t i{Index[ind + 1].get_res_pos_e(0) - 1};
             i < Index[ind + 2].get_res_pos_e(0) - 1; ++i) {
          fix_pair_y(mol, ii[2], i);
        }
    } else if (X_min > crit_dis_x) {
      if (d_min > par_2 * sqrt(square(crit_dis_x) + square(crit_dis_y) +
                               square(crit_dis_z)))
        for (size_t i{Index[ind + 1].get_res_pos_e(0) - 1};
             i < Index[ind + 2].get_res_pos_e(0) - 1; ++i) {
          fix_pair_x(mol, ii[3], i);
        }
    } else {
      continue;
    }
  }
  return 0;
}
void fix_pair_z(MOL &mol, size_t atom1, size_t atom2) {
  double crit_dis_z{par * mol.Box.get_c()};
  double Z{mol.pos.z[atom1] - mol.pos.z[atom2]};
  if (fabs(Z) >= crit_dis_z) {
    if (mol.Box.get_box_type() == "cubic" ||
        mol.Box.get_box_type() == "rectangular") {
      if (Z > 0.)
        mol.pos.z[atom2] += mol.Box.get_c();
      else
        mol.pos.z[atom2] -= mol.Box.get_c();
    } else if (mol.Box.get_box_type() == "dodecahedron" ||
               mol.Box.get_box_type() == "triclinic") {
      double vz[3];
      mol.Box.get_v3(vz);
      if (Z > 0.)
        mol.pos.z[atom2] += vz[2];
      else
        mol.pos.z[atom2] -= vz[2];

      if (mol.pos.y[atom1] - mol.pos.y[atom2] > 0.)
        mol.pos.y[atom2] += vz[1];
      else
        mol.pos.y[atom2] -= vz[1];

      if (mol.pos.x[atom1] - mol.pos.x[atom2] > 0.)
        mol.pos.x[atom2] += vz[0];
      else
        mol.pos.x[atom2] -= vz[0];
    } else {
    }
  }
}
void fix_pair_y(MOL &mol, size_t atom1, size_t atom2) {
  double crit_dis_y{par * mol.Box.get_b()};
  double Y{mol.pos.y[atom1] - mol.pos.y[atom2]};
  if (fabs(Y) >= crit_dis_y) {
    if (mol.Box.get_box_type() == "cubic" ||
        mol.Box.get_box_type() == "rectangular") {
      if (Y > 0.)
        mol.pos.y[atom2] += mol.Box.get_b();
      else
        mol.pos.y[atom2] -= mol.Box.get_b();
    } else if (mol.Box.get_box_type() == "dodecahedron" ||
               mol.Box.get_box_type() == "triclinic") {
      double vy[3];
      mol.Box.get_v2(vy);
      if (Y > 0.)
        mol.pos.y[atom2] += vy[1];
      else
        mol.pos.y[atom2] -= vy[1];

      if (mol.pos.x[atom1] - mol.pos.x[atom2] > 0.)
        mol.pos.x[atom2] += vy[0];
      else
        mol.pos.x[atom2] -= vy[0];
    } else {
    }
  }
}
void fix_pair_x(MOL &mol, size_t atom1, size_t atom2) {
  double crit_dis_x{par * mol.Box.get_a()};
  double X{mol.pos.x[atom1] - mol.pos.x[atom2]};
  if (fabs(X) >= crit_dis_x) {
    if (mol.Box.get_box_type() == "cubic" ||
        mol.Box.get_box_type() == "rectangular") {
      if (X > 0.)
        mol.pos.x[atom2] += mol.Box.get_a();
      else
        mol.pos.x[atom2] -= mol.Box.get_a();
    } else if (mol.Box.get_box_type() == "dodecahedron" ||
               mol.Box.get_box_type() == "triclinic") {
      double vx[3];
      mol.Box.get_v1(vx);
      if (mol.pos.x[atom1] - mol.pos.x[atom2] > 0.)
        mol.pos.x[atom2] += vx[0];
      else
        mol.pos.x[atom2] -= vx[0];
    } else {
    }
  }
}
/*int find_neighbor_chains(vector<pair<size_t, size_t> >& neighbors, MOL&
mol_in)
{
        vector<double> com_x, com_y, com_z;
        for (size_t i{};i < mol_in.Index.size() - 1;++i)
        {
                double x{}, y{}, z{}, mass{};
                for (size_t j = mol_in.Index[i].get_res_pos_e(0) - 1;j <
mol_in.Index[i + 1].get_res_pos_e(0) - 1;++j)
                {
                        x += mol_in.pos.x[j]* mol_in.Molecule[j].get_atom_mass();
                        y += mol_in.pos.y[j]* mol_in.Molecule[j].get_atom_mass();
                        z += mol_in.pos.z[j]* mol_in.Molecule[j].get_atom_mass();
                        mass += mol_in.Molecule[j].get_atom_mass();
                }
                com_x.emplace_back(x / mass);com_y.emplace_back(y /
mass);com_z.emplace_back(z / mass);
        }
        for (size_t i{};i < com_x.size();++i)
        {
                double d_min{ 100000000. };
                pair<size_t, size_t> index;
                for (size_t j{};j < com_x.size();++j)
                {
                        if (i != j)
                        {
                                double d{ sqrt(pow(com_x[i] - com_x[j],2.) +
pow(com_y[i] - com_y[j],2.) + pow(com_z[i] - com_z[j],2.)) }; if (d_min > d)
                                {
                                        d_min = d;
                                        index.first = i;index.second = j;
                                }
                        }
                }
                neighbors.emplace_back(index);
        }
        return 0;
}*/

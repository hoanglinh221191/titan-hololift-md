#include "Multi_MOL.h"
#include "info.h"
#include <iostream>
#include <map>

using namespace std;

int init_rejected_images(vector<size_t> &rejected_images, arguments &Arguments);
bool check_in_rejected_images(vector<size_t> &rejected_images, size_t im);

int Multi_MOL::init_chains_connect(arguments &Arguments,
                                   const amino_acid &amine,
                                   map<size_t, size_t> &chains_connect) {
  Multi_MOL pdbref;
  ::info ref_info(Arguments);
  pdbref.take_coordinate(Arguments.pbc.ref_pdb_name, Arguments.force_f, Arguments,
                         amine, ref_info, 0);

  if (pdbref.m_mol.empty())
    return 1;

  for (size_t i{1}; i < pdbref.Index.size() - 1; ++i) {
    double mindis{10e14};
    size_t pos{};
    for (size_t k{}; k < i; ++k) {
      if (i == k)
        continue;
      for (size_t j{pdbref.Index[i].get_res_pos_e(0) - 1};
           j < pdbref.Index[i + 1].get_res_pos_e(0) - 1; ++j)
        for (size_t l{pdbref.Index[k].get_res_pos_e(0) - 1};
             l < pdbref.Index[k + 1].get_res_pos_e(0) - 1; ++l) {
          double dis{sqrt(square(pdbref.m_mol[0].pos.x[j] -
                                 pdbref.m_mol[0].pos.x[l]) +
                          square(pdbref.m_mol[0].pos.y[j] -
                                 pdbref.m_mol[0].pos.y[l]) +
                          square(pdbref.m_mol[0].pos.z[j] -
                                 pdbref.m_mol[0].pos.z[l]))};
          if (mindis > dis) {
            mindis = dis;
            pos = k;
          }
        }
    }

    chains_connect.emplace(i, pos);
  }

  cout << "Connect map:\n";
  for (auto &m : chains_connect)
    cout << m.first << " - " << m.second << '\n';
  return 0;
}

int Multi_MOL::solv_pbc_ref(arguments &Arguments, const amino_acid &amine,
                            map<size_t, size_t> &chains_connect, size_t ind,
                            size_t ref_ind) {

  vector<size_t> rejected_images;

  if (Arguments.pbc.solv_pbc_clust == true)
    init_rejected_images(rejected_images, Arguments);

  if (ind - ref_ind == 0)
    this->init_chains_connect(Arguments, amine, chains_connect);

  this->impl_solve_pbc_chain(ind, Arguments, rejected_images);

  Multi_MOL pbc_images;
  pbc_images.m_mol.reserve(27);
  this->m_mol[ind].create_pbc_images(Arguments, pbc_images, Index);

  double centerofrefchain[3]{};
  if (Arguments.pbc.pbc_envi == true)
    this->m_mol[ind].find_cog_chain(Arguments.pbc.chain_ref, Arguments, this->Index,
                                    centerofrefchain);

  for (size_t k{1}; k < Index.size() - 1; ++k) {
    if (Arguments.pbc.pbc_envi == true) {
      bool in_evi{false};
      for (size_t c{}; c < Arguments.pbc.pbc_envi_chain.size(); ++c)
        if (Index[k].get_chain() == Arguments.pbc.pbc_envi_chain[c]) {
          in_evi = true;
          break;
        }
      if (in_evi == true) {
        this->solv_pbc_envi(ind, Arguments, rejected_images, k, pbc_images,
                            centerofrefchain);
        continue;
      }
    }

    double distance_min{10000000.};
    size_t pos{};
    bool flag_min{false};

    for (size_t i{}; i < pbc_images.m_mol.size(); ++i) {
      if (Arguments.pbc.solv_pbc_clust == true &&
          check_in_rejected_images(rejected_images, i) == true)
        continue;

      double d{this->find_min_dis(ind, i, chains_connect[k], k, pbc_images)};
      if (d < distance_min) {
        distance_min = d;
        pos = i;
        flag_min = true;
      }
    }

    if (flag_min == true)
      for (size_t i{Index[k].get_res_pos_e(0) - 1};
           i < Index[k + 1].get_res_pos_e(0) - 1; ++i) {
        m_mol[ind].pos.x[i] = pbc_images.m_mol[pos].pos.x[i]; m_mol[ind].pos.y[i] = pbc_images.m_mol[pos].pos.y[i]; m_mol[ind].pos.z[i] = pbc_images.m_mol[pos].pos.z[i];
      }
  }

  return 0;
}

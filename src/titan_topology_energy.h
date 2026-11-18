#pragma once
#include "topology/nonbonded_topology.h"
#include "cache_sigma_epsilon.h"
#include "MOL.h"
#include "MOL_info.h"

namespace titan_nb {
struct PairEnergy {
  double elec{},vdw{};
  // Force on atom A = prefactor * (r_A-r_B), with coordinates in nm.
  double force_elec_prefactor{},force_vdw_prefactor{};
};
void validate_selection(const Topology &topology,const std::vector<mol_info> &info);
PairEnergy evaluate_pair(const cache_sigma_epsilon &cache,const arguments &args,
                         const mol_info &a,const mol_info &b,double distance_nm,
                         double cutoff_nm);
bool topology_energy_required(const cache_sigma_epsilon &cache,const arguments &args);
void calculate(const MOL &rec,const MOL &lig,const std::vector<mol_info> &rec_info,
               const std::vector<mol_info> &lig_info,cache_sigma_epsilon &cache,
               std::size_t begin,std::size_t end,cache_energy &energy,
               cache_force *force_out,const arguments &args);
} // namespace titan_nb

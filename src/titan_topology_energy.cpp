#include "titan_topology_energy.h"
#include "martini_nonbonded.h"
#include "titan_electrostatics.h"
#include "titan_energy_validation.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace titan_nb {
void validate_selection(const Topology &topology,const std::vector<mol_info> &info) {
  std::unordered_set<std::size_t> seen;
  for(const auto &atom:info) {
    if(!atom.has_original_atom_id() || atom.get_original_atom_id()>=topology.atoms.size() ||
       !seen.insert(atom.get_original_atom_id()).second)
      throw std::runtime_error("unsupported_topology_self_energy: selection lacks unique source atom identities");
    const auto &bound=topology.atoms[atom.get_original_atom_id()];
    if(bound.name!=atom.get_atom())
      throw std::runtime_error("unsupported_topology_self_energy: source atom name does not match topology at "+
                               std::to_string(atom.get_original_atom_id()+1));
  }
}
bool topology_energy_required(const cache_sigma_epsilon &cache,const arguments &args) {
  if(args.self_int && !cache.nonbonded_topology)
    throw std::runtime_error("unsupported_topology_self_energy: full identity-bound common topology is required");
  return bool(cache.nonbonded_topology);
}
PairEnergy evaluate_pair(const cache_sigma_epsilon &cache,const arguments &args,
                         const mol_info &a,const mol_info &b,double r,double cutoff) {
  if(!cache.nonbonded_topology || !a.has_original_atom_id() || !b.has_original_atom_id())
    throw std::runtime_error("unsupported_topology_self_energy: missing topology/atom identity");
  PairEnergy out;
  if(a.get_original_atom_id()==b.get_original_atom_id() ||
     (args.self_int && args.flag_interchain_energy && a.get_chain()==b.get_chain()))return out;
  const auto terms=cache.nonbonded_topology->pair(a.get_original_atom_id(),b.get_original_atom_id());
  if(terms.normal.c6==0 && terms.normal.c12==0 && terms.normal_charge==0 &&
     terms.listed.c6==0 && terms.listed.c12==0 && terms.listed_charge==0)return out;
  if(!std::isfinite(r) || r<=titan_energy_validation::kMinimumPairDistanceNm)
    throw std::runtime_error("invalid_overlapping_atoms: topology pair requires a positive finite distance");
  const double inv=1/r, inv2=inv*inv, inv6=inv2*inv2*inv2;
  const double k=138.935457839/std::max(1e-12,args.epsilon_in);
  const bool is_martini=args.force_f.find("MARTINI")!=std::string::npos;
  auto cfg=martini::electrostatic_config(args);
  if(is_martini && cutoff<=0)cutoff=cfg.cutoff_nm;
  if(cutoff<=0 || r<=cutoff) {
    out.vdw=terms.normal.c12*inv6*inv6-terms.normal.c6*inv6;
    out.force_vdw_prefactor=(12*terms.normal.c12*inv6*inv6-6*terms.normal.c6*inv6)*inv2;
    if(is_martini) {
      const double krf=0.5/(cutoff*cutoff*cutoff),crf=1.5/cutoff;
      out.elec=k*terms.normal_charge*(inv+krf*r*r-crf);
      out.force_elec_prefactor=k*terms.normal_charge*(inv*inv2-2*krf);
    } else {
      const double kappa=args.flag_elec_screening && args.salt_concentration>0
        ? 1/titan_electrostatics::debye_length_nm(args.salt_concentration) : 0;
      const double screening=std::exp(-kappa*r);
      out.elec=k*terms.normal_charge*screening*inv;
      out.force_elec_prefactor=k*terms.normal_charge*screening*(inv*inv2+kappa*inv2);
    }
  }
  // Listed interactions are additional, not a replacement for non-excluded
  // normal interactions. GROMACS always evaluates them with plain Coulomb.
  out.vdw+=terms.listed.c12*inv6*inv6-terms.listed.c6*inv6;
  out.force_vdw_prefactor+=(12*terms.listed.c12*inv6*inv6-6*terms.listed.c6*inv6)*inv2;
  out.elec+=k*terms.listed_charge*inv;
  out.force_elec_prefactor+=k*terms.listed_charge*inv*inv2;
  return out;
}
void calculate(const MOL &rec,const MOL &lig,const std::vector<mol_info> &rec_info,
               const std::vector<mol_info> &lig_info,cache_sigma_epsilon &cache,
               std::size_t begin,std::size_t end,cache_energy &energy,
               cache_force *force_out,const arguments &args) {
  if(!cache.nonbonded_topology || end>lig.size() || rec_info.size()!=rec.size() || lig_info.size()!=lig.size())
    throw std::runtime_error("invalid topology energy dimensions");
  energy.set_energy(0,0);if(force_out)force_out->set_force(0,0,0,0,0,0);
  for(std::size_t j=begin;j<end;++j)for(std::size_t i=0;i<rec.size();++i) {
    const double dx=0.1*(rec.pos.x[i]-lig.pos.x[j]),dy=0.1*(rec.pos.y[i]-lig.pos.y[j]),dz=0.1*(rec.pos.z[i]-lig.pos.z[j]);
    const auto pair=evaluate_pair(cache,args,rec_info[i],lig_info[j],std::sqrt(dx*dx+dy*dy+dz*dz),args.distance_energy_cutoff);
    energy+=cache_energy(pair.elec,pair.vdw);
    const std::size_t pos=rec.size()*j+i;
    if(!cache.elec_ener.empty())cache.set_elec_vdw(pos,pair.elec,pair.vdw);
    if(force_out) {
      const cache_force force(pair.force_elec_prefactor*dx,pair.force_elec_prefactor*dy,pair.force_elec_prefactor*dz,
                              pair.force_vdw_prefactor*dx,pair.force_vdw_prefactor*dy,pair.force_vdw_prefactor*dz);
      *force_out+=force;
      if(cache.has_forces())cache.set_force(pos,force.forces[0].Fx,force.forces[0].Fy,force.forces[0].Fz,
                                           force.forces[1].Fx,force.forces[1].Fy,force.forces[1].Fz);
    }
  }
}
} // namespace titan_nb

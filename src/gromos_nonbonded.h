#pragma once
#include "charmm_nonbonded.h"

namespace gromos {
// For GROMOS, the legacy sigma/epsilon parameter slots contain C6/C12
// (kJ mol^-1 nm^6 / nm^12). Overrides retain that same convention.
inline void pair_params(const cache_sigma_epsilon &cache,
                        const Multi_MOL_parameters &rec,
                        const Multi_MOL_parameters &lig, std::size_t i,
                        std::size_t j, double &c6, double &c12) {
  if (charmm::lookup_pair_override(cache, rec, lig, i, j, c6, c12)) return;
  c6=std::sqrt(rec[i].get_sigma()*lig[j].get_sigma());
  c12=std::sqrt(rec[i].get_epsilon()*lig[j].get_epsilon());
}
} // namespace gromos

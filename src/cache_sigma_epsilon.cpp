#include "cache_sigma_epsilon.h"
#include "memory_utils.h"

int cache_sigma_epsilon::set_size(std::size_t MOL_size,
                                  const arguments &Arguments,
                                  std::size_t mol_1_size,
                                  std::size_t mol_2_size) {
  cutoff_dis = Arguments.distance_energy_cutoff;
  force_field_name = Arguments.force_f;

  // --- Dynamic RAM guard ---
  // Estimate total cache footprint: 2 energy arrays always,
  // + 6 force arrays if force calculation is requested.
  constexpr std::size_t bytes_per_double = sizeof(double);   // 8
  const std::size_t n_arrays =
      Arguments.flag_cal_force == true ? 8 : 2;              // energy + force
  const std::size_t estimated_bytes = MOL_size * bytes_per_double * n_arrays;

  const std::size_t avail_ram = get_available_ram();
  constexpr double ram_fraction = 0.3; // never exceed 30% of available RAM

  if (Arguments.flag_cache == 2 ||
      (avail_ram > 0 && estimated_bytes > static_cast<std::size_t>(
                            ram_fraction * static_cast<double>(avail_ram)))) {
    return 1;
  }

  elec_ener.assign(MOL_size, 0.0);
  vdw_ener.assign(MOL_size, 0.0);

  if (Arguments.flag_cal_force == true) {
    opt_forces.emplace();
    opt_forces->assign(MOL_size);
  }

  return 0;
}

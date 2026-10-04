#ifndef TITAN_ENERGY_DECOMPOSITION_H
#define TITAN_ENERGY_DECOMPOSITION_H

#include "Multi_MOL.h"
#include "cache_sigma_epsilon.h"

#include <cstddef>
#include <span>
#include <string>

struct arguments;

struct EnergyPairDecomposition {
  double elec{0.0};
  double vdw{0.0};

  [[nodiscard]] double total() const noexcept { return elec + vdw; }

  EnergyPairDecomposition &operator+=(const EnergyPairDecomposition &rhs) {
    elec += rhs.elec;
    vdw += rhs.vdw;
    return *this;
  }
};

[[nodiscard]] bool titan_energy_force_field_contains(const std::string &ff,
                                                     const char *needle);
[[nodiscard]] double titan_energy_mic_delta(double d, double box_len);
[[nodiscard]] double titan_resolve_sg_cutoff_nm(const arguments &args);
[[nodiscard]] constexpr std::size_t
titan_energy_cache_index(std::size_t mol2_atom, std::size_t mol1_size,
                         std::size_t mol1_atom) noexcept {
  return mol2_atom * mol1_size + mol1_atom;
}

class EnergyDecompositionProvider {
public:
  EnergyDecompositionProvider() = default;

  void configure(const Multi_MOL &rec, const Multi_MOL &lig,
                 const cache_sigma_epsilon &cached_data,
                 const arguments &args, double cutoff_nm,
                 bool cutoff_active);

  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool has_force_parameters() const noexcept;
  [[nodiscard]] double cutoff_nm() const noexcept { return cutoff_nm_; }
  [[nodiscard]] bool cutoff_active() const noexcept { return cutoff_active_; }
  [[nodiscard]] const std::string &force_field() const noexcept {
    return force_field_;
  }

  [[nodiscard]] EnergyPairDecomposition
  pair_energy(const MOL &rec_frame, const MOL &lig_frame,
              std::size_t rec_pos, std::size_t lig_pos) const;
  [[nodiscard]] bool inter_pair_energies_batch(
      const MOL &rec_frame, const MOL &lig_frame, std::size_t rec_begin,
      std::size_t rec_end, std::size_t lig_begin, std::size_t lig_end,
      std::span<double> elec_out, std::span<double> vdw_out) const;
  [[nodiscard]] EnergyPairDecomposition
  rec_pair_energy(const MOL &rec_frame, std::size_t rec_pos_a,
                  std::size_t rec_pos_b) const;
  [[nodiscard]] EnergyPairDecomposition
  lig_pair_energy(const MOL &lig_frame, std::size_t lig_pos_a,
                  std::size_t lig_pos_b) const;
  [[nodiscard]] const mol_info *rec_info(std::size_t pos) const noexcept;
  [[nodiscard]] const mol_info *lig_info(std::size_t pos) const noexcept;

private:
  [[nodiscard]] EnergyPairDecomposition
  pair_energy_generic(const MOL &frame_a, const Multi_MOL &mol_a,
                      std::size_t pos_a, const MOL &frame_b,
                      const Multi_MOL &mol_b, std::size_t pos_b) const;
  [[nodiscard]] EnergyPairDecomposition pair_energy_generic_at_distance(
      const MOL &frame_a, const Multi_MOL &mol_a, std::size_t pos_a,
      const MOL &frame_b, const Multi_MOL &mol_b, std::size_t pos_b,
      double d_nm) const;
  const Multi_MOL *rec_{nullptr};
  const Multi_MOL *lig_{nullptr};
  const cache_sigma_epsilon *cached_data_{nullptr};
  const arguments *args_{nullptr};
  double cutoff_nm_{-1.0};
  bool cutoff_active_{false};
  std::string force_field_;
};

#endif // TITAN_ENERGY_DECOMPOSITION_H

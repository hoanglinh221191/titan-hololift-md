#ifndef TITAN_ENERGY_STATS_H
#define TITAN_ENERGY_STATS_H

#include "Multi_MOL.h"
#include "force.h"
#include "miscellaneous.h"

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace titan_energy_stats {

struct EnergyAxisEntry {
  std::size_t chain_id{};
  std::size_t key_id{};
  std::size_t resnr{};
  std::size_t internal_resnr{};
  std::size_t atom_index1{};
  std::string chain;
  std::string resname;
  std::string atom_name;
  bool atom_level{};
};

struct EnergyAxis {
  std::vector<EnergyAxisEntry> entries;
  std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t,
                     pair_hash>
      lookup;
  bool atom_level{};
};

struct RunningStat {
  std::size_t n{};
  double mean{};
  double m2{};

  void add(double value) noexcept;
  double variance() const noexcept;
  double sd() const noexcept;
};

struct RunningForceStat {
  RunningStat fx;
  RunningStat fy;
  RunningStat fz;
  RunningStat magnitude;

  void add(const force &value) noexcept;
  force mean_force() const noexcept;
  force sd_force() const noexcept;
};

struct ResidueEnergyKey {
  std::string chain;
  std::size_t resnr{};
  std::string resname;

  bool operator<(const ResidueEnergyKey &rhs) const noexcept;
};

struct ResidueEnergySummary {
  ResidueEnergyKey key;
  std::size_t samples{};
  double mean_elec{};
  double sd_elec{};
  double mean_vdw{};
  double sd_vdw{};
  double mean_total{};
  double sd_total{};
  force mean_force_elec;
  force sd_force_elec;
  force mean_force_vdw;
  force sd_force_vdw;
  force mean_force_total;
  force sd_force_total;
  double mean_force_elec_mag{};
  double sd_force_elec_mag{};
  double mean_force_vdw_mag{};
  double sd_force_vdw_mag{};
  double mean_force_total_mag{};
  double sd_force_total_mag{};
};

bool is_ligand_type(std::string_view mol_type) noexcept;
bool is_ligand_mol(const Multi_MOL &mol) noexcept;

EnergyAxis build_energy_axis(const Multi_MOL &mol);
std::pair<std::size_t, std::size_t>
axis_lookup_key_for_atom(const Multi_MOL &mol, std::size_t atom_index0);

std::string axis_label(const EnergyAxisEntry &entry);
std::string tsv_escape(std::string value);

std::vector<ResidueEnergySummary>
summarize_residue_energy(const std::vector<std::vector<energy_res>> &stats);

} // namespace titan_energy_stats

#endif // TITAN_ENERGY_STATS_H

#include "titan_energy_stats.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <tuple>

namespace titan_energy_stats {
namespace {

std::string normalized_chain_label(const std::string &chain,
                                   std::size_t chain_id) {
  if (chain.empty() || chain == " ")
    return "chain" + std::to_string(chain_id);
  return chain;
}

std::string sanitize_label_part(std::string value) {
  for (char &ch : value) {
    if (ch == '\t' || ch == '\n' || ch == '\r' ||
        std::isspace(static_cast<unsigned char>(ch)))
      ch = '_';
  }
  return value;
}

struct ResidueRunningStats {
  RunningStat elec;
  RunningStat vdw;
  RunningStat total;
  RunningForceStat force_elec;
  RunningForceStat force_vdw;
  RunningForceStat force_total;

  void add(const energy_res_base &sample) noexcept {
    elec.add(sample.e_e);
    vdw.add(sample.e_v);
    total.add(sample.e_e + sample.e_v);
    force_elec.add(sample.F_elec);
    force_vdw.add(sample.F_vdw);
    force_total.add(sample.F_elec + sample.F_vdw);
  }
};

} // namespace

void RunningStat::add(double value) noexcept {
  ++n;
  const double delta = value - mean;
  mean += delta / static_cast<double>(n);
  const double delta2 = value - mean;
  m2 += delta * delta2;
}

double RunningStat::variance() const noexcept {
  return n == 0 ? 0.0 : m2 / static_cast<double>(n);
}

double RunningStat::sd() const noexcept { return std::sqrt(variance()); }

void RunningForceStat::add(const force &value) noexcept {
  fx.add(value.Fx);
  fy.add(value.Fy);
  fz.add(value.Fz);
  magnitude.add(value.modul());
}

force RunningForceStat::mean_force() const noexcept {
  return force(fx.mean, fy.mean, fz.mean);
}

force RunningForceStat::sd_force() const noexcept {
  return force(fx.sd(), fy.sd(), fz.sd());
}

bool ResidueEnergyKey::operator<(const ResidueEnergyKey &rhs) const noexcept {
  return std::tie(chain, resnr, resname) <
         std::tie(rhs.chain, rhs.resnr, rhs.resname);
}

bool is_ligand_type(std::string_view mol_type) noexcept {
  return mol_type == "lig" || mol_type == "ligand";
}

bool is_ligand_mol(const Multi_MOL &mol) noexcept {
  return is_ligand_type(mol.get_mol_type());
}

EnergyAxis build_energy_axis(const Multi_MOL &mol) {
  EnergyAxis axis;
  axis.atom_level = is_ligand_mol(mol);
  axis.entries.reserve(axis.atom_level ? mol.info.size() : mol.Index.size());
  axis.lookup.reserve(axis.atom_level ? mol.info.size() : mol.Index.size());

  if (axis.atom_level) {
    for (std::size_t atom = 0; atom < mol.info.size(); ++atom) {
      const auto &inf = mol.info[atom];
      EnergyAxisEntry entry;
      entry.chain_id = inf.get_chainnum();
      entry.key_id = atom + 1;
      entry.resnr = inf.get_resnr();
      entry.internal_resnr = inf.get_internal_resnr();
      entry.atom_index1 = atom + 1;
      entry.chain = inf.get_chain();
      entry.resname = inf.get_original_res_name();
      entry.atom_name = inf.get_atom();
      entry.atom_level = true;
      axis.lookup[{entry.chain_id, entry.key_id}] = axis.entries.size();
      axis.entries.emplace_back(std::move(entry));
    }
    return axis;
  }

  if (mol.Index.size() > 1) {
    for (std::size_t chain_idx = 0; chain_idx + 1 < mol.Index.size();
         ++chain_idx) {
      const auto &chain = mol.Index[chain_idx];
      for (std::size_t res_idx = 0; res_idx < chain.get_size_v(); ++res_idx) {
        const std::size_t atom_pos1 = chain.get_res_pos_e(res_idx);
        if (atom_pos1 == 0 || atom_pos1 > mol.info.size())
          continue;
        const auto &inf = mol.info[atom_pos1 - 1];
        EnergyAxisEntry entry;
        entry.chain_id = inf.get_chainnum();
        entry.key_id = inf.get_internal_resnr();
        entry.resnr = chain.get_res_num_e(res_idx);
        entry.internal_resnr = inf.get_internal_resnr();
        entry.atom_index1 = 0;
        entry.chain = chain.get_chain();
        entry.resname = chain.get_res_name_e(res_idx);
        entry.atom_name.clear();
        entry.atom_level = false;
        auto [it, inserted] =
            axis.lookup.try_emplace({entry.chain_id, entry.key_id},
                                    axis.entries.size());
        if (inserted)
          axis.entries.emplace_back(std::move(entry));
      }
    }
  }

  if (!axis.entries.empty())
    return axis;

  for (std::size_t atom = 0; atom < mol.info.size(); ++atom) {
    const auto &inf = mol.info[atom];
    EnergyAxisEntry entry;
    entry.chain_id = inf.get_chainnum();
    entry.key_id = inf.get_internal_resnr();
    entry.resnr = inf.get_resnr();
    entry.internal_resnr = inf.get_internal_resnr();
    entry.chain = inf.get_chain();
    entry.resname = inf.get_original_res_name();
    entry.atom_level = false;
    auto [it, inserted] =
        axis.lookup.try_emplace({entry.chain_id, entry.key_id},
                                axis.entries.size());
    if (inserted)
      axis.entries.emplace_back(std::move(entry));
  }

  return axis;
}

std::pair<std::size_t, std::size_t>
axis_lookup_key_for_atom(const Multi_MOL &mol, std::size_t atom_index0) {
  if (atom_index0 >= mol.info.size())
    return {0u, 0u};
  const auto &inf = mol.info[atom_index0];
  const std::size_t entity =
      is_ligand_mol(mol) ? atom_index0 + 1 : inf.get_internal_resnr();
  return {inf.get_chainnum(), entity};
}

std::string axis_label(const EnergyAxisEntry &entry) {
  const std::string chain =
      sanitize_label_part(normalized_chain_label(entry.chain, entry.chain_id));
  const std::string resname = sanitize_label_part(entry.resname);
  if (entry.atom_level) {
    return chain + ":atom" + std::to_string(entry.atom_index1) + ":" +
           resname + std::to_string(entry.resnr) + ":" +
           sanitize_label_part(entry.atom_name);
  }
  return chain + ":" + std::to_string(entry.resnr) + ":" + resname;
}

std::string tsv_escape(std::string value) {
  for (char &ch : value) {
    if (ch == '\t' || ch == '\n' || ch == '\r')
      ch = ' ';
  }
  return value;
}

std::vector<ResidueEnergySummary>
summarize_residue_energy(const std::vector<std::vector<energy_res>> &stats) {
  std::map<ResidueEnergyKey, ResidueRunningStats> accumulators;

  for (const auto &trajectory : stats) {
    for (const auto &residue : trajectory) {
      ResidueEnergyKey key{residue.chain, residue.resnr, residue.resname};
      auto &running = accumulators[key];
      for (const auto &sample : residue.base)
        running.add(sample);
    }
  }

  std::vector<ResidueEnergySummary> out;
  out.reserve(accumulators.size());
  for (const auto &[key, running] : accumulators) {
    ResidueEnergySummary summary;
    summary.key = key;
    summary.samples = running.total.n;
    summary.mean_elec = running.elec.mean;
    summary.sd_elec = running.elec.sd();
    summary.mean_vdw = running.vdw.mean;
    summary.sd_vdw = running.vdw.sd();
    summary.mean_total = running.total.mean;
    summary.sd_total = running.total.sd();
    summary.mean_force_elec = running.force_elec.mean_force();
    summary.sd_force_elec = running.force_elec.sd_force();
    summary.mean_force_vdw = running.force_vdw.mean_force();
    summary.sd_force_vdw = running.force_vdw.sd_force();
    summary.mean_force_total = running.force_total.mean_force();
    summary.sd_force_total = running.force_total.sd_force();
    summary.mean_force_elec_mag = running.force_elec.magnitude.mean;
    summary.sd_force_elec_mag = running.force_elec.magnitude.sd();
    summary.mean_force_vdw_mag = running.force_vdw.magnitude.mean;
    summary.sd_force_vdw_mag = running.force_vdw.magnitude.sd();
    summary.mean_force_total_mag = running.force_total.magnitude.mean;
    summary.sd_force_total_mag = running.force_total.magnitude.sd();
    out.emplace_back(std::move(summary));
  }
  return out;
}

} // namespace titan_energy_stats

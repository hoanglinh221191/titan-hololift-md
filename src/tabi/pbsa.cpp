#include "analyzer_impl.h"

#include "../titan_energy_decomposition.h"
#include "../titan_executor.h"
#include "../titan_interaction_entropy.h"
#include "../titan_sasa.h"
#include "../titan_ui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TitanPBE {
namespace {

constexpr double kKcalToKJ = 4.184;
constexpr double kKJToKcal = 1.0 / kKcalToKJ;
constexpr double kKBKJPerMolK = 0.00831446261815324;

std::size_t effective_workers(const TitanExecutor &executor,
                              std::size_t work_items) noexcept {
  if (work_items == 0)
    return 1;
  return std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), work_items));
}

std::string format_fixed(double value, int digits = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(digits) << value;
  return oss.str();
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '\"')
      out += "\"\"";
    else
      out += ch;
  }
  out += "\"";
  return out;
}

bool is_ligand_type(const std::string &type) {
  return type == "ligand" || type == "lig";
}

bool pbsa_frame_usable(const FrameResult &row) {
  if (!row.pbsa_enabled || row.pbsa_status != "ok")
    return false;
  const std::array<double, 7> required = {
      row.pbsa_elec_kj_mol,       row.pbsa_vdw_kj_mol,
      row.pbsa_mm_kj_mol,         row.pbsa_delta_sasa_a2,
      row.pbsa_nonpolar_kj_mol,   row.pbsa_delta_solv_kj_mol,
      row.pbsa_no_entropy_kj_mol};
  return std::all_of(required.begin(), required.end(),
                     [](double value) { return std::isfinite(value); });
}

} // namespace

AnalyzerImpl::PbsaComponents AnalyzerImpl::compute_pbsa_components(
    const MOL &mol1, const Multi_MOL_parameters &params1, const MOL &mol2,
    const Multi_MOL_parameters &params2, const std::vector<Atom> &mol1_atoms,
    const std::vector<Atom> &mol2_atoms,
    const std::vector<Atom> &complex_atoms, double deltaG_solv_kj_mol,
    TitanExecutor &executor,
    const EnergyDecompositionProvider *energy_provider) const {
  (void)params1;
  (void)params2;
  if (energy_provider == nullptr || !energy_provider->ready()) {
    throw std::runtime_error(
        "PBSA: -pbsa requires initialized two-molecule force-field "
        "parameters for elec/vdw interaction energy.");
  }

  PbsaComponents comp;
  const std::size_t n1 = mol1.size();
  const std::size_t n2 = mol2.size();
  const std::size_t workers = effective_workers(executor, n1);
  std::vector<double> local_elec(workers, 0.0);
  std::vector<double> local_vdw(workers, 0.0);
  executor.parallel_for(
      n1, workers, 1,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        double elec = 0.0;
        double vdw = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
          for (std::size_t j = 0; j < n2; ++j) {
            const EnergyPairDecomposition pair =
                energy_provider->pair_energy(mol1, mol2, i, j);
            elec += pair.elec;
            vdw += pair.vdw;
          }
        }
        local_elec[worker] = elec;
        local_vdw[worker] = vdw;
      });
  for (std::size_t worker = 0; worker < workers; ++worker) {
    comp.elec_kj_mol += local_elec[worker];
    comp.vdw_kj_mol += local_vdw[worker];
  }
  comp.mm_kj_mol = comp.elec_kj_mol + comp.vdw_kj_mol;

  comp.sasa_mol1_a2 = compute_sasa_total(mol1_atoms, executor);
  comp.sasa_mol2_a2 = compute_sasa_total(mol2_atoms, executor);
  comp.sasa_complex_a2 = compute_sasa_total(complex_atoms, executor);
  comp.delta_sasa_a2 =
      comp.sasa_complex_a2 - comp.sasa_mol1_a2 - comp.sasa_mol2_a2;
  comp.nonpolar_kj_mol =
      TitanSASA::nonpolar_energy_from_delta_sasa_kj(comp.delta_sasa_a2);
  comp.delta_solv_kj_mol = deltaG_solv_kj_mol;
  comp.no_entropy_kj_mol =
      comp.mm_kj_mol + comp.delta_solv_kj_mol + comp.nonpolar_kj_mol;
  const std::array<double, 10> required = {
      comp.elec_kj_mol,       comp.vdw_kj_mol,
      comp.mm_kj_mol,         comp.sasa_complex_a2,
      comp.sasa_mol1_a2,      comp.sasa_mol2_a2,
      comp.delta_sasa_a2,     comp.nonpolar_kj_mol,
      comp.delta_solv_kj_mol, comp.no_entropy_kj_mol};
  if (!std::all_of(required.begin(), required.end(),
                   [](double value) { return std::isfinite(value); }))
    comp.status = "nonfinite_energy";
  return comp;
}

std::string AnalyzerImpl::effective_pbsa_decomp() const {
  if (args_.pbsa_decomp != "auto")
    return args_.pbsa_decomp;
  const bool mol1_ligand = is_ligand_type(args_.pbsa_mol1_type);
  const bool mol2_ligand = is_ligand_type(args_.pbsa_mol2_type);
  return mol1_ligand != mol2_ligand ? "hybrid" : "off";
}

std::vector<AnalyzerImpl::ResidueFrameRow> AnalyzerImpl::compute_pbsa_residue_rows(
    const MOL &mol1, const MOL &mol2, const std::vector<Atom> &mol1_atoms,
    const std::vector<Atom> &mol2_atoms,
    const std::vector<Atom> &complex_atoms,
    const ReactionSolve &complex_solve, const ReactionSolve &mol1_solve,
    const ReactionSolve &mol2_solve, const FrameResult &result,
    TitanExecutor &executor,
    const EnergyDecompositionProvider *energy_provider) const {
  std::vector<ResidueFrameRow> rows;
  std::string decomp = effective_pbsa_decomp();
  if (edecomp_sink_ && decomp != "hybrid")
    decomp = "residue";
  if ((decomp != "residue" && decomp != "hybrid") ||
      energy_provider == nullptr ||
      !energy_provider->ready())
    return rows;

  const std::size_t n1 = mol1_atoms.size();
  const std::size_t n2 = mol2_atoms.size();
  if (complex_solve.atom_reaction_kj_mol.size() < n1 + n2 ||
      mol1_solve.atom_reaction_kj_mol.size() < n1 ||
      mol2_solve.atom_reaction_kj_mol.size() < n2)
    return rows;

  const bool hybrid = decomp == "hybrid";
  const bool mol1_atom = hybrid && is_ligand_type(args_.pbsa_mol1_type);
  const bool mol2_atom = hybrid && is_ligand_type(args_.pbsa_mol2_type);
  const char *mol1_role =
      hybrid ? (mol1_atom ? "ligand" : "receptor") : "receptor";
  const char *mol2_role =
      hybrid ? (mol2_atom ? "ligand" : "receptor") : "ligand";

  auto make_key = [](const char *molecule, bool atom_unit,
                     const mol_info *info, std::size_t atom_index) {
    ResidueKey key;
    key.molecule = molecule;
    key.unit_type = atom_unit ? "atom" : "residue";
    if (info != nullptr) {
      key.chainnum = info->get_chainnum();
      key.chain = info->get_chain();
      key.resnr = info->get_resnr();
      key.respos = info->get_respos();
      key.resname = info->get_res_name();
      if (atom_unit)
        key.atom_name = info->get_atom();
    } else {
      key.resnr = atom_index + 1;
      key.respos = atom_index + 1;
      key.resname = "UNK";
    }
    if (atom_unit)
      key.atom_index = atom_index + 1;
    return key;
  };
  auto key_id = [](const ResidueKey &key) {
    return key.molecule + "|" + key.unit_type + "|" +
           std::to_string(key.chainnum) + "|" + key.chain + "|" +
           std::to_string(key.resnr) + "|" + std::to_string(key.respos) +
           "|" + key.resname + "|" + std::to_string(key.atom_index);
  };
  auto add = [&](std::map<std::string, ResidueContribution> &accum,
                 const ResidueKey &key, double elec, double vdw, double polar,
                 double nonpolar) {
    ResidueContribution &entry = accum[key_id(key)];
    entry.key = key;
    entry.elec_kj_mol += elec;
    entry.vdw_kj_mol += vdw;
    entry.mm_kj_mol += elec + vdw;
    entry.polar_kj_mol += polar;
    entry.nonpolar_kj_mol += nonpolar;
  };

  std::map<std::string, ResidueContribution> accum;
  for (std::size_t i = 0; i < n1; ++i) {
    const ResidueKey rec_key =
        make_key(mol1_role, mol1_atom, energy_provider->rec_info(i), i);
    for (std::size_t j = 0; j < n2; ++j) {
      const ResidueKey lig_key =
          make_key(mol2_role, mol2_atom, energy_provider->lig_info(j), j);
      const EnergyPairDecomposition pair =
          energy_provider->pair_energy(mol1, mol2, i, j);
      add(accum, rec_key, 0.5 * pair.elec, 0.5 * pair.vdw, 0.0, 0.0);
      add(accum, lig_key, 0.5 * pair.elec, 0.5 * pair.vdw, 0.0, 0.0);
    }
  }

  const std::vector<double> sasa_complex =
      compute_sasa_per_atom(complex_atoms, executor);
  const std::vector<double> sasa_mol1 =
      compute_sasa_per_atom(mol1_atoms, executor);
  const std::vector<double> sasa_mol2 =
      compute_sasa_per_atom(mol2_atoms, executor);

  for (std::size_t i = 0; i < n1; ++i) {
    const ResidueKey rec_key =
        make_key(mol1_role, mol1_atom, energy_provider->rec_info(i), i);
    const double polar =
        complex_solve.atom_reaction_kj_mol[i] -
        mol1_solve.atom_reaction_kj_mol[i];
    const double delta_sasa =
        (i < sasa_complex.size() ? sasa_complex[i] : 0.0) -
        (i < sasa_mol1.size() ? sasa_mol1[i] : 0.0);
    add(accum, rec_key, 0.0, 0.0, polar,
        TitanSASA::nonpolar_energy_from_delta_sasa_kj(delta_sasa));
  }
  for (std::size_t j = 0; j < n2; ++j) {
    const ResidueKey lig_key =
        make_key(mol2_role, mol2_atom, energy_provider->lig_info(j), j);
    const std::size_t complex_index = n1 + j;
    const double polar =
        complex_solve.atom_reaction_kj_mol[complex_index] -
        mol2_solve.atom_reaction_kj_mol[j];
    const double delta_sasa =
        (complex_index < sasa_complex.size() ? sasa_complex[complex_index]
                                             : 0.0) -
        (j < sasa_mol2.size() ? sasa_mol2[j] : 0.0);
    add(accum, lig_key, 0.0, 0.0, polar,
        TitanSASA::nonpolar_energy_from_delta_sasa_kj(delta_sasa));
  }

  rows.reserve(accum.size());
  for (auto &[id, contribution] : accum) {
    (void)id;
    contribution.total_kj_mol = contribution.mm_kj_mol +
                                contribution.polar_kj_mol +
                                contribution.nonpolar_kj_mol;
    ResidueFrameRow row;
    static_cast<ResidueContribution &>(row) = contribution;
    row.trajectory_id = result.trajectory_id;
    row.frame_index = result.frame_index;
    row.global_index = result.global_index;
    row.time_ps = result.time_ps;
    rows.push_back(row);
  }
  return rows;
}

void AnalyzerImpl::accumulate_pbsa_residue_rows(
    const FrameResult &result, const std::vector<ResidueFrameRow> &rows) {
  if (!pbsa_frame_usable(result) || rows.empty())
    return;

  auto key_id = [](const ResidueKey &key) {
    return key.molecule + "|" + key.unit_type + "|" +
           std::to_string(key.chainnum) + "|" + key.chain + "|" +
           std::to_string(key.resnr) + "|" + std::to_string(key.respos) +
           "|" + key.resname + "|" + std::to_string(key.atom_index);
  };
  auto add_stats = [&](ResidueStats &stats, const ResidueFrameRow &row) {
    stats.key = row.key;
    ++stats.frames;
    add_scalar(stats.elec, row.elec_kj_mol);
    add_scalar(stats.vdw, row.vdw_kj_mol);
    add_scalar(stats.mm, row.mm_kj_mol);
    add_scalar(stats.polar, row.polar_kj_mol);
    add_scalar(stats.nonpolar, row.nonpolar_kj_mol);
    add_scalar(stats.total, row.total_kj_mol);
      if (args_.pbsa_ie && result.pbsa_status == "ok")
        stats.ie_values_kj_mol.push_back(row.mm_kj_mol);
  };

  double elec = 0.0;
  double vdw = 0.0;
  double polar = 0.0;
  double nonpolar = 0.0;
  double total = 0.0;
  for (const ResidueFrameRow &row : rows) {
    const std::string id = key_id(row.key);
    add_stats(pbsa_residue_stats_[id], row);
    add_stats(pbsa_residue_traj_stats_[std::to_string(row.trajectory_id) +
                                       "|" + id],
              row);
    elec += row.elec_kj_mol;
    vdw += row.vdw_kj_mol;
    polar += row.polar_kj_mol;
    nonpolar += row.nonpolar_kj_mol;
    total += row.total_kj_mol;
  }

  pbsa_residue_closure_elec_max_abs_kj_ =
      std::max(pbsa_residue_closure_elec_max_abs_kj_,
               std::abs(elec - result.pbsa_elec_kj_mol));
  pbsa_residue_closure_vdw_max_abs_kj_ =
      std::max(pbsa_residue_closure_vdw_max_abs_kj_,
               std::abs(vdw - result.pbsa_vdw_kj_mol));
  pbsa_residue_closure_polar_max_abs_kj_ =
      std::max(pbsa_residue_closure_polar_max_abs_kj_,
               std::abs(polar - result.pbsa_delta_solv_kj_mol));
  pbsa_residue_closure_nonpolar_max_abs_kj_ =
      std::max(pbsa_residue_closure_nonpolar_max_abs_kj_,
               std::abs(nonpolar - result.pbsa_nonpolar_kj_mol));
  pbsa_residue_closure_max_abs_kj_ =
      std::max(pbsa_residue_closure_max_abs_kj_,
               std::abs(total - result.pbsa_no_entropy_kj_mol));

  if (args_.pbsa_residue_frames == "all") {
    pbsa_residue_frame_rows_.insert(pbsa_residue_frame_rows_.end(),
                                    rows.begin(), rows.end());
  }
}

void AnalyzerImpl::add_scalar(ScalarStats &stats, double value) {
  if (!std::isfinite(value))
    return;
  ++stats.count;
  const double n = static_cast<double>(stats.count);
  const double delta = value - stats.mean;
  stats.mean += delta / n;
  stats.m2 += delta * (value - stats.mean);
  if (stats.count == 1) {
    stats.min = value;
    stats.max = value;
  } else {
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
  }
}

double AnalyzerImpl::scalar_stddev(const ScalarStats &stats) {
  if (stats.count <= 1)
    return 0.0;
  return std::sqrt(stats.m2 / static_cast<double>(stats.count - 1));
}

double AnalyzerImpl::scalar_mean_or_nan(const ScalarStats &stats) {
  return stats.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                          : stats.mean;
}

double AnalyzerImpl::scalar_stddev_or_nan(const ScalarStats &stats) {
  return stats.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                          : scalar_stddev(stats);
}

void AnalyzerImpl::accumulate_pbsa(const FrameResult &result) {
  if (!result.pbsa_enabled)
    return;
  ++pbsa_frames_;
  const bool usable = pbsa_frame_usable(result);
  if (!usable) {
    ++pbsa_failed_;
  } else {
    ++pbsa_valid_;
    add_scalar(pbsa_elec_stats_, result.pbsa_elec_kj_mol);
    add_scalar(pbsa_vdw_stats_, result.pbsa_vdw_kj_mol);
    add_scalar(pbsa_mm_stats_, result.pbsa_mm_kj_mol);
    add_scalar(pbsa_delta_sasa_stats_, result.pbsa_delta_sasa_a2);
    add_scalar(pbsa_nonpolar_stats_, result.pbsa_nonpolar_kj_mol);
    add_scalar(pbsa_polar_stats_, result.pbsa_delta_solv_kj_mol);
    add_scalar(pbsa_no_entropy_stats_, result.pbsa_no_entropy_kj_mol);
  }
  pbsa_rows_.push_back(result);
  if (args_.pbsa_ie && usable)
    pbsa_ie_values_kj_.push_back(result.pbsa_mm_kj_mol);
}

AnalyzerImpl::PbsaEntropyResult
AnalyzerImpl::compute_pbsa_entropy_for_values(
    const std::vector<double> &values) const {
  PbsaEntropyResult out;
  out.samples = values.size();
  if (!args_.pbsa_ie) {
    out.status = "disabled";
    return out;
  }

  std::size_t block_size = 0;
  if (args_.pbsa_ie_block == "auto") {
    block_size = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::floor(std::sqrt(values.size()))));
  } else {
    block_size =
        std::max<std::size_t>(1, static_cast<std::size_t>(
                                     std::stoi(args_.pbsa_ie_block)));
  }

  TitanEntropy::InteractionEntropyOptions options;
  options.rt_energy = kKBKJPerMolK * std::max(1.0, args_.temp);
  options.filter_sigma = args_.pbsa_ie_filter_sigma;
  options.min_samples =
      static_cast<std::size_t>(std::max(1, args_.pbsa_ie_min_samples));
  options.block_size = block_size;
  const TitanEntropy::InteractionEntropyResult entropy =
      TitanEntropy::compute_interaction_entropy(
          std::span<const double>(values.data(), values.size()), options);
  out.samples = entropy.samples;
  out.filtered_samples = entropy.filtered_samples;
  out.ess = entropy.ess;
  out.minus_tds_kj_mol = entropy.minus_tds_energy;
  out.block_size = entropy.block_size;
  out.block_count = entropy.block_count;
  out.block_mean_kj_mol = entropy.block_mean_energy;
  out.block_std_kj_mol = entropy.block_std_energy;
  out.block_drift_kj_mol = entropy.block_drift_energy;
  out.status = entropy.status;
  return out;
}

AnalyzerImpl::PbsaEntropyResult AnalyzerImpl::compute_pbsa_entropy() const {
  return compute_pbsa_entropy_for_values(pbsa_ie_values_kj_);
}

void AnalyzerImpl::write_pbsa_energy_csv(const PbsaEntropyResult &entropy) const {
  if (!args_.pbsa)
    return;

  std::ofstream out("mm_pbsa_energy.csv", std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_energy.csv.");
    return;
  }

  out << "trajectory_id,frame_index,global_index,time_ps,backend,"
         "atom_count_complex,atom_count_receptor,atom_count_ligand,"
         "grid_nx,grid_ny,grid_nz,solver_mode,mg_levels,"
         "mg_cycles_complex,mg_cycles_receptor,mg_cycles_ligand,"
         "status,complex_status,receptor_status,ligand_status,"
         "included_in_statistics,"
         "complex_solvent_iter,complex_vacuum_iter,"
         "receptor_solvent_iter,receptor_vacuum_iter,"
         "ligand_solvent_iter,ligand_vacuum_iter,"
         "complex_solvent_rms,complex_vacuum_rms,"
         "receptor_solvent_rms,receptor_vacuum_rms,"
         "ligand_solvent_rms,ligand_vacuum_rms,"
         "complex_solvent_max_dphi,complex_vacuum_max_dphi,"
         "receptor_solvent_max_dphi,receptor_vacuum_max_dphi,"
         "ligand_solvent_max_dphi,ligand_vacuum_max_dphi,"
         "E_elec_kJmol,E_elec_kcalmol,E_vdw_kJmol,E_vdw_kcalmol,"
         "E_MM_kJmol,E_MM_kcalmol,"
         "G_solv_complex_kJmol,G_solv_complex_kcalmol,"
         "G_solv_receptor_kJmol,G_solv_receptor_kcalmol,"
         "G_solv_ligand_kJmol,G_solv_ligand_kcalmol,"
         "DeltaDeltaG_solv_kJmol,DeltaDeltaG_solv_kcalmol,"
         "DeltaG_nonpolar_kJmol,DeltaG_nonpolar_kcalmol,"
         "deltaSASA_A2,minus_TDeltaS_IE_kJmol,"
         "minus_TDeltaS_IE_kcalmol,DeltaG_bind_kJmol,"
         "DeltaG_bind_kcalmol,ie_status\n";

  out << std::setprecision(10);
  const bool entropy_available = !args_.pbsa_ie || entropy.status == "ok";
  const double minus_tds =
      !args_.pbsa_ie
          ? 0.0
          : (entropy.status == "ok"
                 ? entropy.minus_tds_kj_mol
                 : std::numeric_limits<double>::quiet_NaN());
  for (const FrameResult &row : pbsa_rows_) {
    const bool usable = pbsa_frame_usable(row);
    const double row_minus_tds =
        usable && entropy_available
            ? minus_tds
            : std::numeric_limits<double>::quiet_NaN();
    const double dg_bind = usable && entropy_available
                               ? row.pbsa_no_entropy_kj_mol + minus_tds
                               : std::numeric_limits<double>::quiet_NaN();
    out << row.trajectory_id << ',' << row.frame_index << ','
        << row.global_index << ',' << row.time_ps << ',' << row.pbe_backend
        << ','
        << row.atom_count << ',' << row.pbsa_receptor_atom_count << ','
        << row.pbsa_ligand_atom_count << ',' << row.nx << ',' << row.ny
        << ',' << row.nz << ',' << row.solver_mode << ',' << row.mg_levels
        << ',' << row.mg_cycles << ',' << row.pbsa_receptor_mg_cycles << ','
        << row.pbsa_ligand_mg_cycles << ',' << row.pbsa_status << ','
        << row.pbsa_complex_status << ',' << row.pbsa_receptor_status << ','
        << row.pbsa_ligand_status << ',' << (usable ? "true" : "false") << ','
        << row.solvent_iterations << ','
        << row.vacuum_iterations << ','
        << row.pbsa_receptor_solvent_iterations << ','
        << row.pbsa_receptor_vacuum_iterations << ','
        << row.pbsa_ligand_solvent_iterations << ','
        << row.pbsa_ligand_vacuum_iterations << ',' << row.solvent_rms << ','
        << row.vacuum_rms << ',' << row.pbsa_receptor_solvent_rms << ','
        << row.pbsa_receptor_vacuum_rms << ','
        << row.pbsa_ligand_solvent_rms << ','
        << row.pbsa_ligand_vacuum_rms << ','
        << row.solvent_max_dphi << ',' << row.vacuum_max_dphi << ','
        << row.pbsa_receptor_solvent_max_dphi << ','
        << row.pbsa_receptor_vacuum_max_dphi << ','
        << row.pbsa_ligand_solvent_max_dphi << ','
        << row.pbsa_ligand_vacuum_max_dphi << ',' << row.pbsa_elec_kj_mol
        << ',' << row.pbsa_elec_kj_mol * kKJToKcal << ','
        << row.pbsa_vdw_kj_mol << ',' << row.pbsa_vdw_kj_mol * kKJToKcal
        << ',' << row.pbsa_mm_kj_mol << ','
        << row.pbsa_mm_kj_mol * kKJToKcal << ','
        << row.complex_energy_kj_mol << ','
        << row.complex_energy_kcal_mol << ',' << row.mol1_energy_kj_mol
        << ',' << row.mol1_energy_kcal_mol << ','
        << row.mol2_energy_kj_mol << ',' << row.mol2_energy_kcal_mol << ','
        << row.pbsa_delta_solv_kj_mol << ','
        << row.pbsa_delta_solv_kcal_mol << ','
        << row.pbsa_nonpolar_kj_mol << ','
        << row.pbsa_nonpolar_kj_mol * kKJToKcal << ','
        << row.pbsa_delta_sasa_a2 << ',' << row_minus_tds << ','
        << row_minus_tds * kKJToKcal << ',' << dg_bind << ','
        << dg_bind * kKJToKcal << ','
        << (usable ? entropy.status : "frame_excluded") << '\n';
  }
}

void AnalyzerImpl::write_pbsa_decomposition(
    const PbsaEntropyResult &entropy) const {
  const std::string effective_decomp = effective_pbsa_decomp();
  if (!args_.pbsa || effective_decomp == "off")
    return;

  std::ofstream out("mm_pbsa_decomposition.dat",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning(
        "PBSA: failed to write mm_pbsa_decomposition.dat.");
    return;
  }

  out << "# TITAN MM/PBSA endpoint decomposition\n";
  out << "requested_decomp " << args_.pbsa_decomp << "\n";
  out << "effective_decomp " << effective_decomp << "\n";
  out << "residue_summary "
      << (effective_decomp == "residue" ? "mm_pbsa_residue_summary.csv"
                                          : "off")
      << "\n";
  out << "residue_by_trajectory "
      << (effective_decomp == "residue"
              ? "mm_pbsa_residue_by_trajectory.csv"
              : "off")
      << "\n";
  out << "residue_frame "
      << (effective_decomp == "residue" &&
                  args_.pbsa_residue_frames == "all"
              ? "mm_pbsa_residue_frame.csv"
              : "off")
      << "\n";
  out << "unit_summary "
      << (effective_decomp == "hybrid" ? "mm_pbsa_unit_summary.csv" : "off")
      << "\n";
  out << "unit_by_trajectory "
      << (effective_decomp == "hybrid" ? "mm_pbsa_unit_by_trajectory.csv"
                                        : "off")
      << "\n";
  out << "unit_frame "
      << (effective_decomp == "hybrid" &&
                  args_.pbsa_residue_frames == "all"
              ? "mm_pbsa_unit_frame.csv"
              : "off")
      << "\n";
  out << "residue_closure_total_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_max_abs_kj_, 10) << "\n";
  out << "residue_closure_elec_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_elec_max_abs_kj_, 10) << "\n";
  out << "residue_closure_vdw_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_vdw_max_abs_kj_, 10) << "\n";
  out << "residue_closure_polar_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_polar_max_abs_kj_, 10) << "\n";
  out << "residue_closure_nonpolar_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_nonpolar_max_abs_kj_, 10) << "\n";
  out << "unit_closure_total_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_max_abs_kj_, 10) << "\n";
  out << "unit_closure_elec_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_elec_max_abs_kj_, 10) << "\n";
  out << "unit_closure_vdw_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_vdw_max_abs_kj_, 10) << "\n";
  out << "unit_closure_polar_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_polar_max_abs_kj_, 10) << "\n";
  out << "unit_closure_nonpolar_max_abs_kJmol "
      << format_fixed(pbsa_residue_closure_nonpolar_max_abs_kj_, 10) << "\n";
  out << "unit_entropy_closure not_applicable_nonlinear\n";
  out << "interaction_entropy_status " << entropy.status << "\n";
  out << "component,mean_kJmol,mean_kcalmol,std_kJmol,std_kcalmol\n";
  auto write_stat_row = [&](const char *name, const ScalarStats &stats) {
    const double mean = scalar_mean_or_nan(stats);
    const double sd = scalar_stddev_or_nan(stats);
    out << name << ',' << format_fixed(mean) << ','
        << format_fixed(mean * kKJToKcal) << ',' << format_fixed(sd)
        << ',' << format_fixed(sd * kKJToKcal) << "\n";
  };
  write_stat_row("DeltaE_elec", pbsa_elec_stats_);
  write_stat_row("DeltaE_vdw", pbsa_vdw_stats_);
  write_stat_row("DeltaE_MM", pbsa_mm_stats_);
  write_stat_row("DeltaDeltaG_solv_PB", pbsa_polar_stats_);
  write_stat_row("DeltaG_nonpolar_SASA", pbsa_nonpolar_stats_);
  write_stat_row("DeltaG_no_entropy", pbsa_no_entropy_stats_);
}

void AnalyzerImpl::write_pbsa_residue_outputs() const {
  if (!args_.pbsa || effective_pbsa_decomp() != "residue")
    return;

  auto write_key = [](std::ofstream &out, const ResidueKey &key) {
    out << csv_escape(key.molecule) << ',' << key.chainnum << ','
        << csv_escape(key.chain) << ',' << key.resnr << ',' << key.respos
        << ',' << csv_escape(key.resname);
  };
  auto write_stats_values = [&](std::ofstream &out, const ScalarStats &stats) {
    out << ',' << format_fixed(stats.mean, 10) << ','
        << format_fixed(scalar_stddev(stats), 10) << ','
        << format_fixed(stats.min, 10) << ',' << format_fixed(stats.max, 10);
  };
  auto write_stats_header = [](std::ofstream &out) {
    const std::array<const char *, 6> names = {
        "E_elec", "E_vdw", "E_MM", "DeltaDeltaG_solv_PB",
        "DeltaG_nonpolar_SASA", "DeltaG_residue"};
    for (const char *name : names) {
      out << ',' << name << "_kJmol_mean," << name << "_kJmol_std," << name
          << "_kJmol_min," << name << "_kJmol_max";
    }
    out << ",DeltaG_residue_kcalmol_mean\n";
  };
  auto write_stats_row = [&](std::ofstream &out, const ResidueStats &stats) {
    write_key(out, stats.key);
    out << ',' << stats.frames;
    write_stats_values(out, stats.elec);
    write_stats_values(out, stats.vdw);
    write_stats_values(out, stats.mm);
    write_stats_values(out, stats.polar);
    write_stats_values(out, stats.nonpolar);
    write_stats_values(out, stats.total);
    out << ',' << format_fixed(stats.total.mean * kKJToKcal, 10) << "\n";
  };

  std::ofstream summary("mm_pbsa_residue_summary.csv",
                        std::ios::out | std::ios::trunc);
  if (summary.is_open()) {
    summary << "molecule,chainnum,chain,resnr,respos,resname,frames";
    write_stats_header(summary);
    for (const auto &[id, stats] : pbsa_residue_stats_) {
      (void)id;
      write_stats_row(summary, stats);
    }
  } else {
    TitanUI::print_warning(
        "PBSA: failed to write mm_pbsa_residue_summary.csv.");
  }

  std::ofstream by_traj("mm_pbsa_residue_by_trajectory.csv",
                        std::ios::out | std::ios::trunc);
  if (by_traj.is_open()) {
    by_traj << "trajectory_id,molecule,chainnum,chain,resnr,respos,resname,"
               "frames";
    write_stats_header(by_traj);
    for (const auto &[id, stats] : pbsa_residue_traj_stats_) {
      const std::size_t cut = id.find('|');
      by_traj << (cut == std::string::npos ? id : id.substr(0, cut)) << ',';
      write_stats_row(by_traj, stats);
    }
  } else {
    TitanUI::print_warning(
        "PBSA: failed to write mm_pbsa_residue_by_trajectory.csv.");
  }

  if (args_.pbsa_residue_frames == "all") {
    std::ofstream frame("mm_pbsa_residue_frame.csv",
                        std::ios::out | std::ios::trunc);
    if (frame.is_open()) {
      frame << "trajectory_id,frame_index,global_index,time_ps,molecule,"
               "chainnum,chain,resnr,respos,resname,E_elec_kJmol,"
               "E_vdw_kJmol,E_MM_kJmol,DeltaDeltaG_solv_PB_kJmol,"
               "DeltaG_nonpolar_SASA_kJmol,DeltaG_residue_kJmol,"
               "DeltaG_residue_kcalmol\n";
      frame << std::setprecision(10);
      for (const ResidueFrameRow &row : pbsa_residue_frame_rows_) {
        frame << row.trajectory_id << ',' << row.frame_index << ','
              << row.global_index << ',' << row.time_ps << ',';
        write_key(frame, row.key);
        frame << ',' << row.elec_kj_mol << ',' << row.vdw_kj_mol << ','
              << row.mm_kj_mol << ',' << row.polar_kj_mol << ','
              << row.nonpolar_kj_mol << ',' << row.total_kj_mol << ','
              << row.total_kj_mol * kKJToKcal << "\n";
      }
    } else {
      TitanUI::print_warning(
          "PBSA: failed to write mm_pbsa_residue_frame.csv.");
    }
  }
}

void AnalyzerImpl::write_pbsa_unit_outputs() const {
  if (!args_.pbsa || effective_pbsa_decomp() != "hybrid")
    return;

  auto write_key = [](std::ofstream &out, const ResidueKey &key) {
    out << csv_escape(key.molecule) << ',' << key.unit_type << ','
        << key.chainnum << ',' << csv_escape(key.chain) << ',' << key.resnr
        << ',' << key.respos << ',' << csv_escape(key.resname) << ',';
    if (key.unit_type == "atom")
      out << key.atom_index;
    out << ',' << csv_escape(key.atom_name);
  };
  auto write_mean_std = [&](std::ofstream &out, const ScalarStats &stats) {
    out << ',' << format_fixed(stats.mean, 10) << ','
        << format_fixed(scalar_stddev(stats), 10);
  };
  auto write_header = [](std::ofstream &out) {
    out << ",E_elec_kJmol_mean,E_elec_kJmol_std,"
           "E_vdw_kJmol_mean,E_vdw_kJmol_std,"
           "E_MM_kJmol_mean,E_MM_kJmol_std,"
           "DeltaDeltaG_solv_PB_kJmol_mean,"
           "DeltaDeltaG_solv_PB_kJmol_std,"
           "DeltaG_nonpolar_SASA_kJmol_mean,"
           "DeltaG_nonpolar_SASA_kJmol_std,"
           "DeltaG_no_entropy_kJmol_mean,DeltaG_no_entropy_kJmol_std,"
           "DeltaG_no_entropy_kcalmol_mean,IE_status,IE_samples,IE_ESS,"
           "minus_TDeltaS_IE_kJmol,minus_TDeltaS_IE_kcalmol,"
           "DeltaG_bind_IE_kJmol,DeltaG_bind_IE_kcalmol\n";
  };
  auto write_stats_row = [&](std::ofstream &out, const ResidueStats &stats) {
    const PbsaEntropyResult entropy =
        compute_pbsa_entropy_for_values(stats.ie_values_kj_mol);
    const double minus_tds =
        !args_.pbsa_ie
            ? 0.0
            : (entropy.status == "ok"
                   ? entropy.minus_tds_kj_mol
                   : std::numeric_limits<double>::quiet_NaN());
    const double dg_ie =
        std::isfinite(minus_tds)
            ? stats.total.mean + minus_tds
            : std::numeric_limits<double>::quiet_NaN();
    write_key(out, stats.key);
    out << ',' << stats.frames;
    write_mean_std(out, stats.elec);
    write_mean_std(out, stats.vdw);
    write_mean_std(out, stats.mm);
    write_mean_std(out, stats.polar);
    write_mean_std(out, stats.nonpolar);
    write_mean_std(out, stats.total);
    out << ',' << format_fixed(stats.total.mean * kKJToKcal, 10) << ','
        << entropy.status << ',' << entropy.samples << ','
        << format_fixed(entropy.ess, 4) << ',' << format_fixed(minus_tds, 10)
        << ',' << format_fixed(minus_tds * kKJToKcal, 10) << ','
        << format_fixed(dg_ie, 10) << ','
        << format_fixed(dg_ie * kKJToKcal, 10) << "\n";
  };

  std::ofstream summary("mm_pbsa_unit_summary.csv",
                        std::ios::out | std::ios::trunc);
  if (summary.is_open()) {
    summary << "molecule_role,unit_type,chainnum,chain,resnr,respos,resname,"
               "atom_index,atom_name,frames";
    write_header(summary);
    for (const auto &[id, stats] : pbsa_residue_stats_) {
      (void)id;
      write_stats_row(summary, stats);
    }
  } else {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_unit_summary.csv.");
  }

  std::ofstream by_traj("mm_pbsa_unit_by_trajectory.csv",
                        std::ios::out | std::ios::trunc);
  if (by_traj.is_open()) {
    by_traj << "trajectory_id,molecule_role,unit_type,chainnum,chain,resnr,"
               "respos,resname,atom_index,atom_name,frames";
    write_header(by_traj);
    for (const auto &[id, stats] : pbsa_residue_traj_stats_) {
      const std::size_t cut = id.find('|');
      by_traj << (cut == std::string::npos ? id : id.substr(0, cut)) << ',';
      write_stats_row(by_traj, stats);
    }
  } else {
    TitanUI::print_warning(
        "PBSA: failed to write mm_pbsa_unit_by_trajectory.csv.");
  }

  if (args_.pbsa_residue_frames == "all") {
    std::ofstream frame("mm_pbsa_unit_frame.csv",
                        std::ios::out | std::ios::trunc);
    if (frame.is_open()) {
      frame << "trajectory_id,frame_index,global_index,time_ps,molecule_role,"
               "unit_type,chainnum,chain,resnr,respos,resname,atom_index,"
               "atom_name,E_elec_kJmol,E_vdw_kJmol,E_MM_kJmol,"
               "DeltaDeltaG_solv_PB_kJmol,DeltaG_nonpolar_SASA_kJmol,"
               "DeltaG_no_entropy_kJmol,DeltaG_no_entropy_kcalmol\n";
      frame << std::setprecision(10);
      for (const ResidueFrameRow &row : pbsa_residue_frame_rows_) {
        frame << row.trajectory_id << ',' << row.frame_index << ','
              << row.global_index << ',' << row.time_ps << ',';
        write_key(frame, row.key);
        frame << ',' << row.elec_kj_mol << ',' << row.vdw_kj_mol << ','
              << row.mm_kj_mol << ',' << row.polar_kj_mol << ','
              << row.nonpolar_kj_mol << ',' << row.total_kj_mol << ','
              << row.total_kj_mol * kKJToKcal << "\n";
      }
    } else {
      TitanUI::print_warning("PBSA: failed to write mm_pbsa_unit_frame.csv.");
    }
  }
}

void AnalyzerImpl::write_pbsa_by_trajectory() const {
  if (!args_.pbsa)
    return;
  struct TrajStats {
    std::size_t frames{0};
    std::size_t valid{0};
    std::size_t failed{0};
    ScalarStats elec;
    ScalarStats vdw;
    ScalarStats mm;
    ScalarStats polar;
    ScalarStats nonpolar;
    ScalarStats no_entropy;
    std::vector<double> ie_values;
  };
  std::map<std::size_t, TrajStats> stats_by_traj;
  for (const FrameResult &row : pbsa_rows_) {
    TrajStats &stats = stats_by_traj[row.trajectory_id];
    ++stats.frames;
    const bool usable = pbsa_frame_usable(row);
    if (!usable) {
      ++stats.failed;
      continue;
    }
    ++stats.valid;
    add_scalar(stats.elec, row.pbsa_elec_kj_mol);
    add_scalar(stats.vdw, row.pbsa_vdw_kj_mol);
    add_scalar(stats.mm, row.pbsa_mm_kj_mol);
    add_scalar(stats.polar, row.pbsa_delta_solv_kj_mol);
    add_scalar(stats.nonpolar, row.pbsa_nonpolar_kj_mol);
    add_scalar(stats.no_entropy, row.pbsa_no_entropy_kj_mol);
    if (args_.pbsa_ie)
      stats.ie_values.push_back(row.pbsa_mm_kj_mol);
  }

  std::ofstream out("mm_pbsa_by_trajectory.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_by_trajectory.csv.");
    return;
  }
  out << "trajectory_id,frames,valid,failed,statistics_status,"
         "E_elec_kJmol_mean,E_elec_kJmol_std,"
         "E_vdw_kJmol_mean,E_vdw_kJmol_std,E_MM_kJmol_mean,"
         "E_MM_kJmol_std,DeltaDeltaG_solv_PB_kJmol_mean,"
         "DeltaDeltaG_solv_PB_kJmol_std,DeltaG_nonpolar_SASA_kJmol_mean,"
         "DeltaG_nonpolar_SASA_kJmol_std,DeltaG_bind_noIE_kJmol,"
         "DeltaG_bind_noIE_kcalmol,IE_status,IE_samples,IE_ESS,"
         "minus_TDeltaS_IE_kJmol,minus_TDeltaS_IE_kcalmol,"
         "DeltaG_bind_IE_kJmol,DeltaG_bind_IE_kcalmol\n";
  for (const auto &[trajectory_id, stats] : stats_by_traj) {
    const PbsaEntropyResult entropy =
        compute_pbsa_entropy_for_values(stats.ie_values);
    const double minus_tds =
        !args_.pbsa_ie
            ? 0.0
            : (entropy.status == "ok"
                   ? entropy.minus_tds_kj_mol
                   : std::numeric_limits<double>::quiet_NaN());
    const double no_entropy = scalar_mean_or_nan(stats.no_entropy);
    const double dg_ie = stats.valid > 0 && std::isfinite(minus_tds)
                             ? no_entropy + minus_tds
                             : std::numeric_limits<double>::quiet_NaN();
    const char *statistics_status =
        stats.valid == 0 ? "no_valid_frames"
                         : (stats.failed > 0 ? "partial" : "ok");
    out << trajectory_id << ',' << stats.frames << ',' << stats.valid << ','
        << stats.failed << ',' << statistics_status << ','
        << format_fixed(scalar_mean_or_nan(stats.elec), 10) << ','
        << format_fixed(scalar_stddev_or_nan(stats.elec), 10) << ','
        << format_fixed(scalar_mean_or_nan(stats.vdw), 10) << ','
        << format_fixed(scalar_stddev_or_nan(stats.vdw), 10) << ','
        << format_fixed(scalar_mean_or_nan(stats.mm), 10) << ','
        << format_fixed(scalar_stddev_or_nan(stats.mm), 10) << ','
        << format_fixed(scalar_mean_or_nan(stats.polar), 10) << ','
        << format_fixed(scalar_stddev_or_nan(stats.polar), 10) << ','
        << format_fixed(scalar_mean_or_nan(stats.nonpolar), 10) << ','
        << format_fixed(scalar_stddev_or_nan(stats.nonpolar), 10) << ','
        << format_fixed(no_entropy, 10) << ','
        << format_fixed(no_entropy * kKJToKcal, 10) << ','
        << entropy.status << ',' << entropy.samples << ','
        << format_fixed(entropy.ess, 4) << ',' << format_fixed(minus_tds, 10)
        << ',' << format_fixed(minus_tds * kKJToKcal, 10) << ','
        << format_fixed(dg_ie, 10) << ','
        << format_fixed(dg_ie * kKJToKcal, 10) << "\n";
  }
}

void AnalyzerImpl::write_pbsa_ie_diagnostics(
    const PbsaEntropyResult &global_entropy) const {
  if (!args_.pbsa)
    return;
  std::ofstream out("mm_pbsa_ie_diagnostics.csv",
                    std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_ie_diagnostics.csv.");
    return;
  }
  out << "scope,trajectory_id,block_index,samples,filtered_samples,ess,"
         "status,block_size,block_count,minus_TDeltaS_IE_kJmol,"
         "minus_TDeltaS_IE_kcalmol,block_mean_kJmol,block_std_kJmol,"
         "block_drift_kJmol\n";
  auto write_entropy = [&](const char *scope, std::size_t traj,
                           int block_index,
                           const PbsaEntropyResult &entropy) {
    const double minus_tds =
        !args_.pbsa_ie
            ? 0.0
            : (entropy.status == "ok"
                   ? entropy.minus_tds_kj_mol
                   : std::numeric_limits<double>::quiet_NaN());
    out << scope << ',' << traj << ',' << block_index << ','
        << entropy.samples << ',' << entropy.filtered_samples << ','
        << format_fixed(entropy.ess, 4) << ',' << entropy.status << ','
        << entropy.block_size << ',' << entropy.block_count << ','
        << format_fixed(minus_tds, 10) << ','
        << format_fixed(minus_tds * kKJToKcal, 10) << ','
        << format_fixed(entropy.block_mean_kj_mol, 10) << ','
        << format_fixed(entropy.block_std_kj_mol, 10) << ','
        << format_fixed(entropy.block_drift_kj_mol, 10) << "\n";
  };
  write_entropy("global", 0, -1, global_entropy);

  std::map<std::size_t, std::vector<double>> values_by_traj;
  for (const FrameResult &row : pbsa_rows_) {
    if (args_.pbsa_ie && pbsa_frame_usable(row))
      values_by_traj[row.trajectory_id].push_back(row.pbsa_mm_kj_mol);
  }
  for (const auto &[trajectory_id, values] : values_by_traj)
    write_entropy("trajectory", trajectory_id, -1,
                  compute_pbsa_entropy_for_values(values));

  const std::size_t block_size =
      global_entropy.block_size > 0 ? global_entropy.block_size : 1;
  for (std::size_t begin = 0, block_index = 0;
       begin < pbsa_ie_values_kj_.size();
       begin += block_size, ++block_index) {
    const std::size_t end =
        std::min(pbsa_ie_values_kj_.size(), begin + block_size);
    std::vector<double> block(
        pbsa_ie_values_kj_.begin() + static_cast<std::ptrdiff_t>(begin),
        pbsa_ie_values_kj_.begin() + static_cast<std::ptrdiff_t>(end));
    write_entropy("block", 0, static_cast<int>(block_index),
                  compute_pbsa_entropy_for_values(block));
  }
}

void AnalyzerImpl::write_pbsa_performance() const {
  if (!args_.pbsa || args_.pbsa_perf == "off")
    return;
  std::ofstream out("mm_pbsa_performance.dat", std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_performance.dat.");
    return;
  }
  auto add_time = [](ScalarStats &stats, double value) {
    AnalyzerImpl::add_scalar(stats, value);
  };
  ScalarStats complex_t;
  ScalarStats receptor_t;
  ScalarStats ligand_t;
  ScalarStats pb_solve_t;
  ScalarStats components_t;
  ScalarStats residue_t;
  ScalarStats frame_t;
  double slowest_ms = 0.0;
  std::size_t slowest_global = 0;
  for (const FrameResult &row : pbsa_rows_) {
    add_time(complex_t, row.pbsa_time_complex_ms);
    add_time(receptor_t, row.pbsa_time_receptor_ms);
    add_time(ligand_t, row.pbsa_time_ligand_ms);
    add_time(pb_solve_t, row.pbsa_time_complex_ms + row.pbsa_time_receptor_ms +
                             row.pbsa_time_ligand_ms);
    add_time(components_t, row.pbsa_time_components_ms);
    add_time(residue_t, row.pbsa_time_residue_ms);
    add_time(frame_t, row.pbsa_time_frame_ms);
    if (row.pbsa_time_frame_ms > slowest_ms) {
      slowest_ms = row.pbsa_time_frame_ms;
      slowest_global = row.global_index;
    }
  }
  auto write_time = [&](const char *name, const ScalarStats &stats) {
    out << name << "_ms_mean " << format_fixed(stats.mean, 6) << "\n";
    out << name << "_ms_max " << format_fixed(stats.max, 6) << "\n";
  };
  out << "# TITAN MM/PBSA performance summary\n";
  out << "timing_granularity natural_boundaries\n";
  out << "interaction_entropy_source E_MM_elec_plus_vdw\n";
  out << "frames " << pbsa_rows_.size() << "\n";
  const std::size_t endpoint_solves_per_frame =
      args_.backend == "tabi" ? 3U : 6U;
  out << "endpoint_solves "
      << (pbsa_rows_.size() * endpoint_solves_per_frame) << "\n";
  out << "max_grid_points " << max_grid_points_seen_ << "\n";
  out << "max_memory_bytes " << max_memory_bytes_seen_ << "\n";
  out << "residue_decomp " << args_.pbsa_decomp << "\n";
  out << "grid_reuse_requested " << args_.pbsa_grid_reuse << "\n";
  out << "grid_reuse_effective "
      << (pbsa_rows_.empty() ? "off"
                             : pbsa_rows_.front().pbsa_grid_reuse_effective)
      << "\n";
  out << "grid_workspace_reuse_hits "
      << grid_reuse_stats_.workspace_reuse_hits << "\n";
  out << "grid_workspace_reallocations "
      << grid_reuse_stats_.workspace_reallocations << "\n";
  out << "grid_medium_cache_hits "
      << grid_reuse_stats_.medium_cache_hits << "\n";
  out << "grid_medium_cache_misses "
      << grid_reuse_stats_.medium_cache_misses << "\n";
  out << "grid_medium_cache_stores "
      << grid_reuse_stats_.medium_cache_stores << "\n";
  write_time("complex_solve", complex_t);
  write_time("receptor_solve", receptor_t);
  write_time("ligand_solve", ligand_t);
  write_time("pb_solve", pb_solve_t);
  write_time("components", components_t);
  write_time("mm_sasa_components", components_t);
  write_time("residue_decomp", residue_t);
  write_time("decomposition", residue_t);
  write_time("frame_total", frame_t);
  write_time("total", frame_t);
  out << "entropy_output_ms " << format_fixed(pbsa_entropy_output_ms_, 6)
      << "\n";
  out << "slowest_global_frame " << slowest_global << "\n";
  out << "slowest_frame_ms " << format_fixed(slowest_ms, 6) << "\n";

  if (args_.pbsa_perf == "detail") {
    std::ofstream detail("mm_pbsa_performance_frames.csv",
                         std::ios::out | std::ios::trunc);
    if (detail.is_open()) {
      detail << "trajectory_id,frame_index,global_index,time_ps,"
                "complex_solve_ms,receptor_solve_ms,ligand_solve_ms,"
                "pb_solve_ms,"
                "components_ms,mm_sasa_components_ms,residue_decomp_ms,"
                "frame_total_ms,"
                "grid_points,memory_bytes,status\n";
      detail << std::setprecision(10);
      for (const FrameResult &row : pbsa_rows_) {
        detail << row.trajectory_id << ',' << row.frame_index << ','
               << row.global_index << ',' << row.time_ps << ','
               << row.pbsa_time_complex_ms << ','
               << row.pbsa_time_receptor_ms << ','
               << row.pbsa_time_ligand_ms << ','
               << (row.pbsa_time_complex_ms + row.pbsa_time_receptor_ms +
                   row.pbsa_time_ligand_ms)
               << ','
               << row.pbsa_time_components_ms << ','
               << row.pbsa_time_components_ms << ','
               << row.pbsa_time_residue_ms << ','
               << row.pbsa_time_frame_ms << ',' << row.grid_points << ','
               << row.memory_bytes << ',' << row.pbsa_status << "\n";
      }
    } else {
      TitanUI::print_warning(
          "PBSA: failed to write mm_pbsa_performance_frames.csv.");
    }

    std::ofstream endpoints("mm_pbsa_endpoint_timing.csv",
                            std::ios::out | std::ios::trunc);
    if (endpoints.is_open()) {
      endpoints << "trajectory_id,frame_index,global_index,time_ps,endpoint,"
                   "phase,timing_variant,"
                   "setup_ms,surface_ms,epsilon_operator_ms,boundary_ms,"
                   "solver_ms,energy_ms,reuse_surface_hits,"
                   "reuse_operator_hits,status\n";
      endpoints << std::setprecision(10);
      for (const PbsaEndpointTimingRow &row : pbsa_endpoint_timing_rows_) {
        endpoints << row.trajectory_id << ',' << row.frame_index << ','
                  << row.global_index << ',' << row.time_ps << ','
                  << row.endpoint << ',' << row.phase << ',' << row.variant
                  << ',' << row.timing.setup_ms << ','
                  << row.timing.surface_ms << ',' << row.timing.epsilon_ms
                  << ',' << row.timing.boundary_ms << ','
                  << row.timing.solver_ms << ',' << row.timing.energy_ms << ','
                  << row.reuse_surface_hits << ',' << row.reuse_operator_hits
                  << ',' << row.status << "\n";
      }
    } else {
      TitanUI::print_warning(
          "PBSA: failed to write mm_pbsa_endpoint_timing.csv.");
    }
  }

  if (args_.pbsa_grid_reuse == "validate") {
    std::ofstream validate("mm_pbsa_grid_reuse_validation.csv",
                           std::ios::out | std::ios::trunc);
    if (validate.is_open()) {
      validate << "trajectory_id,frame_index,global_index,endpoint,"
                  "reuse_status,reuse_surface_hits,reuse_operator_hits,"
                  "reaction_energy_delta_kJmol,"
                  "DeltaG_no_entropy_delta_kJmol\n";
      validate << std::setprecision(10);
      for (const FrameResult &row : pbsa_rows_) {
        const double bind_delta =
            row.pbsa_complex_reuse_delta_kj_mol -
            row.pbsa_receptor_reuse_delta_kj_mol -
            row.pbsa_ligand_reuse_delta_kj_mol;
        auto write_delta = [&](const char *endpoint, double delta,
                               std::size_t surface_hits,
                               std::size_t operator_hits) {
          validate << row.trajectory_id << ',' << row.frame_index << ','
                   << row.global_index << ',' << endpoint << ','
                   << row.pbsa_grid_reuse_effective << ',' << surface_hits
                   << ',' << operator_hits << ',' << delta << ','
                   << bind_delta << "\n";
        };
        write_delta("complex", row.pbsa_complex_reuse_delta_kj_mol,
                    row.pbsa_complex_reuse_surface_hits,
                    row.pbsa_complex_reuse_operator_hits);
        write_delta("receptor", row.pbsa_receptor_reuse_delta_kj_mol,
                    row.pbsa_receptor_reuse_surface_hits,
                    row.pbsa_receptor_reuse_operator_hits);
        write_delta("ligand", row.pbsa_ligand_reuse_delta_kj_mol,
                    row.pbsa_ligand_reuse_surface_hits,
                    row.pbsa_ligand_reuse_operator_hits);
      }
    } else {
      TitanUI::print_warning(
          "PBSA: failed to write mm_pbsa_grid_reuse_validation.csv.");
    }
  }
}

void AnalyzerImpl::write_pbsa_summary(std::fstream &log) {
  if (!args_.pbsa)
    return;

  const bool capture_performance = args_.pbsa_perf != "off";
  const auto entropy_output_begin =
      capture_performance ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  const PbsaEntropyResult entropy = compute_pbsa_entropy();
  write_pbsa_energy_csv(entropy);
  write_pbsa_decomposition(entropy);
  write_pbsa_residue_outputs();
  write_pbsa_unit_outputs();
  write_pbsa_by_trajectory();
  write_pbsa_ie_diagnostics(entropy);
  if (capture_performance) {
    pbsa_entropy_output_ms_ =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - entropy_output_begin)
            .count();
  }
  write_pbsa_performance();

  const double dg_no_entropy_kj = scalar_mean_or_nan(pbsa_no_entropy_stats_);
  const double minus_tds_ie_kj =
      !args_.pbsa_ie
          ? 0.0
          : (entropy.status == "ok"
                 ? entropy.minus_tds_kj_mol
                 : std::numeric_limits<double>::quiet_NaN());
  const double dg_binding_ie_kj =
      std::isfinite(dg_no_entropy_kj) && std::isfinite(minus_tds_ie_kj)
          ? dg_no_entropy_kj + minus_tds_ie_kj
          : std::numeric_limits<double>::quiet_NaN();

  std::ofstream out("mm_pbsa_summary.dat", std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBSA: failed to write mm_pbsa_summary.dat.");
    return;
  }

  auto write_energy_stats = [&](const char *name, const ScalarStats &stats) {
    const double mean = scalar_mean_or_nan(stats);
    const double stddev = scalar_stddev_or_nan(stats);
    const double minimum = stats.count == 0
                               ? std::numeric_limits<double>::quiet_NaN()
                               : stats.min;
    const double maximum = stats.count == 0
                               ? std::numeric_limits<double>::quiet_NaN()
                               : stats.max;
    out << name << "_kJmol_mean " << format_fixed(mean) << "\n";
    out << name << "_kJmol_std " << format_fixed(stddev) << "\n";
    out << name << "_kJmol_min " << format_fixed(minimum) << "\n";
    out << name << "_kJmol_max " << format_fixed(maximum) << "\n";
    out << name << "_kcalmol_mean " << format_fixed(mean * kKJToKcal)
        << "\n";
    out << name << "_kcalmol_std "
        << format_fixed(stddev * kKJToKcal) << "\n";
    out << name << "_kcalmol_min " << format_fixed(minimum * kKJToKcal)
        << "\n";
    out << name << "_kcalmol_max " << format_fixed(maximum * kKJToKcal)
        << "\n";
  };

  out << "# TITAN MM/PBSA summary\n";
  out << "model MM/PBSA_PBE_linear\n";
  out << "pbsa_backend " << args_.backend << "\n";
  out << "pbsa_enabled true\n";
  out << "pbsa_components_requested " << args_.pbsa_components << "\n";
  out << "pbsa_decomp_requested " << args_.pbsa_decomp << "\n";
  out << "pbsa_decomp_effective " << effective_pbsa_decomp() << "\n";
  out << "pbsa_grid_reuse_requested " << args_.pbsa_grid_reuse << "\n";
  out << "pbsa_grid_reuse_effective "
      << (pbsa_rows_.empty() ? "off"
                             : pbsa_rows_.front().pbsa_grid_reuse_effective)
      << "\n";
  out << "frames_total " << pbsa_frames_ << "\n";
  out << "mm_energy_distance_policy preprocessed_raw\n";
  out << "frames_valid " << pbsa_valid_ << "\n";
  out << "frames_failed " << pbsa_failed_ << "\n";
  out << "statistics_status "
      << (pbsa_valid_ == 0 ? "no_valid_frames"
                           : (pbsa_failed_ > 0 ? "partial" : "ok"))
      << "\n";
  out << "failure_policy strict\n";
  out << "temperature_K " << format_fixed(args_.temp) << "\n";
  out << "pbe_salt_M " << format_fixed(args_.salt) << "\n";
  out << "pbe_surface " << args_.surface << "\n";
  out << "sasa_probe_A " << format_fixed(args_.probe_radius) << "\n";
  out << "sasa_points_per_atom " << args_.pbsa_sasa_points << "\n";
  out << "nonpolar_gamma_kJmol_A2 "
      << format_fixed(TitanSASA::kNonpolarGammaKJPerMolA2, 8) << "\n";
  out << "nonpolar_gamma_kcalmol_A2 "
      << format_fixed(TitanSASA::kNonpolarGammaKcalPerMolA2, 8) << "\n";
  out << "endpoint_solves_per_frame "
      << (args_.backend == "tabi" ? 3 : 6) << "\n";
  out << "energy_timeseries mm_pbsa_energy.csv\n";
  if (effective_pbsa_decomp() != "off")
    out << "decomposition mm_pbsa_decomposition.dat\n";
  out << "interaction_entropy_enabled " << (args_.pbsa_ie ? "true" : "false")
      << "\n";
  out << "interaction_entropy_status " << entropy.status << "\n";
  out << "interaction_entropy_min_samples " << args_.pbsa_ie_min_samples
      << "\n";
  out << "interaction_entropy_filter_sigma "
      << format_fixed(args_.pbsa_ie_filter_sigma) << "\n";
  out << "interaction_entropy_block_config " << args_.pbsa_ie_block << "\n";
  out << "interaction_entropy_source E_MM_elec_plus_vdw\n";
  out << "interaction_entropy_samples " << entropy.samples
      << "\n";
  out << "interaction_entropy_filtered_samples " << entropy.filtered_samples
      << "\n";
  out << "interaction_entropy_ESS " << format_fixed(entropy.ess, 4) << "\n";
  out << "interaction_entropy_block_size " << entropy.block_size << "\n";
  out << "interaction_entropy_block_count " << entropy.block_count << "\n";
  out << "interaction_entropy_block_mean_kJmol "
      << format_fixed(entropy.block_mean_kj_mol) << "\n";
  out << "interaction_entropy_block_std_kJmol "
      << format_fixed(entropy.block_std_kj_mol) << "\n";
  out << "interaction_entropy_block_drift_kJmol "
      << format_fixed(entropy.block_drift_kj_mol) << "\n";
  out << "interaction_entropy_diagnostics mm_pbsa_ie_diagnostics.csv\n";
  out << "per_trajectory_summary mm_pbsa_by_trajectory.csv\n";
  if (args_.pbsa_perf != "off")
    out << "performance_summary mm_pbsa_performance.dat\n";
  if (args_.pbsa_perf == "detail")
    out << "endpoint_timing mm_pbsa_endpoint_timing.csv\n";
  if (args_.pbsa_grid_reuse == "validate")
    out << "grid_reuse_validation mm_pbsa_grid_reuse_validation.csv\n";
  if (effective_pbsa_decomp() == "residue") {
    out << "residue_summary mm_pbsa_residue_summary.csv\n";
    out << "residue_by_trajectory mm_pbsa_residue_by_trajectory.csv\n";
    out << "residue_closure_total_max_abs_kJmol "
        << format_fixed(pbsa_residue_closure_max_abs_kj_, 10) << "\n";
  }
  if (effective_pbsa_decomp() == "hybrid") {
    out << "unit_summary mm_pbsa_unit_summary.csv\n";
    out << "unit_by_trajectory mm_pbsa_unit_by_trajectory.csv\n";
    out << "unit_entropy_closure not_applicable_nonlinear\n";
    out << "unit_closure_total_max_abs_kJmol "
        << format_fixed(pbsa_residue_closure_max_abs_kj_, 10) << "\n";
  }
  write_energy_stats("E_elec", pbsa_elec_stats_);
  write_energy_stats("E_vdw", pbsa_vdw_stats_);
  write_energy_stats("E_MM", pbsa_mm_stats_);
  write_energy_stats("G_polar_PB", pbsa_polar_stats_);
  write_energy_stats("DeltaDeltaG_solv_PB", pbsa_polar_stats_);
  write_energy_stats("G_nonpolar_SASA", pbsa_nonpolar_stats_);
  out << "deltaSASA_A2_mean "
      << format_fixed(scalar_mean_or_nan(pbsa_delta_sasa_stats_))
      << "\n";
  out << "deltaSASA_A2_std "
      << format_fixed(scalar_stddev_or_nan(pbsa_delta_sasa_stats_)) << "\n";
  out << "deltaSASA_A2_min "
      << format_fixed(pbsa_delta_sasa_stats_.count == 0
                          ? std::numeric_limits<double>::quiet_NaN()
                          : pbsa_delta_sasa_stats_.min)
      << "\n";
  out << "deltaSASA_A2_max "
      << format_fixed(pbsa_delta_sasa_stats_.count == 0
                          ? std::numeric_limits<double>::quiet_NaN()
                          : pbsa_delta_sasa_stats_.max)
      << "\n";
  write_energy_stats("DeltaG_no_entropy", pbsa_no_entropy_stats_);
  out << "minus_TdS_IE_kJmol " << format_fixed(minus_tds_ie_kj) << "\n";
  out << "minus_TdS_IE_kcalmol "
      << format_fixed(minus_tds_ie_kj * kKJToKcal) << "\n";
  out << "DeltaG_binding_IE_kJmol " << format_fixed(dg_binding_ie_kj)
      << "\n";
  out << "DeltaG_binding_IE_kcalmol "
      << format_fixed(dg_binding_ie_kj * kKJToKcal) << "\n";
  std::map<std::size_t, std::vector<const FrameResult *>> rows_by_traj;
  for (const FrameResult &row : pbsa_rows_)
    rows_by_traj[row.trajectory_id].push_back(&row);
  ScalarStats traj_equal_noie;
  ScalarStats traj_equal_ie;
  for (const auto &[trajectory_id, rows] : rows_by_traj) {
    (void)trajectory_id;
    ScalarStats noie;
    std::vector<double> ie_values;
    for (const FrameResult *row : rows) {
      if (!pbsa_frame_usable(*row))
        continue;
      add_scalar(noie, row->pbsa_no_entropy_kj_mol);
      if (args_.pbsa_ie)
        ie_values.push_back(row->pbsa_mm_kj_mol);
    }
    if (noie.count == 0)
      continue;
    const PbsaEntropyResult traj_entropy =
        compute_pbsa_entropy_for_values(ie_values);
    const double traj_minus_tds =
        !args_.pbsa_ie
            ? 0.0
            : (traj_entropy.status == "ok"
                   ? traj_entropy.minus_tds_kj_mol
                   : std::numeric_limits<double>::quiet_NaN());
    add_scalar(traj_equal_noie, noie.mean);
    if (std::isfinite(traj_minus_tds))
      add_scalar(traj_equal_ie, noie.mean + traj_minus_tds);
  }
  out << "trajectory_equal_DeltaG_no_entropy_kJmol_mean "
      << format_fixed(scalar_mean_or_nan(traj_equal_noie)) << "\n";
  out << "trajectory_equal_DeltaG_no_entropy_kJmol_std "
      << format_fixed(scalar_stddev_or_nan(traj_equal_noie)) << "\n";
  out << "trajectory_equal_DeltaG_binding_IE_kJmol_mean "
      << format_fixed(scalar_mean_or_nan(traj_equal_ie)) << "\n";
  out << "trajectory_equal_DeltaG_binding_IE_kJmol_std "
      << format_fixed(scalar_stddev_or_nan(traj_equal_ie)) << "\n";

  if (log.is_open()) {
    TitanUI::log_section(log, "MM/PBSA OUTPUT AUDIT");
    TitanUI::log_kv(log, "Frames", std::to_string(pbsa_frames_));
    TitanUI::log_kv(log, "Valid", std::to_string(pbsa_valid_));
    TitanUI::log_kv(log, "Failed", std::to_string(pbsa_failed_));
    TitanUI::log_kv(log, "Endpoint solves/frame",
                    args_.backend == "tabi" ? "3" : "6");
    TitanUI::log_kv(log, "IE Source", "E_MM_elec_plus_vdw");
    TitanUI::log_kv(log, "IE Status", entropy.status);
    TitanUI::log_kv(log, "Interaction Entropy",
                    format_fixed(minus_tds_ie_kj) + " kJ/mol / " +
                        format_fixed(minus_tds_ie_kj * kKJToKcal) +
                        " kcal/mol");
    TitanUI::log_kv(log, "DeltaG Binding IE",
                    format_fixed(dg_binding_ie_kj) + " kJ/mol / " +
                        format_fixed(dg_binding_ie_kj * kKJToKcal) +
                        " kcal/mol");
    TitanUI::log_kv(log, "Summary", "mm_pbsa_summary.dat");
    TitanUI::log_kv(log, "Energy CSV", "mm_pbsa_energy.csv");
    TitanUI::log_kv(log, "By Trajectory", "mm_pbsa_by_trajectory.csv");
    TitanUI::log_kv(log, "IE Diagnostics", "mm_pbsa_ie_diagnostics.csv");
    if (args_.pbsa_perf != "off")
      TitanUI::log_kv(log, "Performance", "mm_pbsa_performance.dat");
    if (args_.pbsa_perf != "off") {
      double pb_solve_ms = 0.0;
      double components_ms = 0.0;
      double decomp_ms = 0.0;
      double frame_ms = 0.0;
      for (const FrameResult &row : pbsa_rows_) {
        pb_solve_ms += row.pbsa_time_complex_ms + row.pbsa_time_receptor_ms +
                       row.pbsa_time_ligand_ms;
        components_ms += row.pbsa_time_components_ms;
        decomp_ms += row.pbsa_time_residue_ms;
        frame_ms += row.pbsa_time_frame_ms;
      }
      TitanUI::log_section(log, "MM/PBSA COMPONENT TIMING");
      TitanUI::log_kv(log, "Timing Granularity", "natural_boundaries");
      TitanUI::log_time(log, "PB Solve", pb_solve_ms / 1000.0);
      TitanUI::log_time(log, "MM + SASA Components", components_ms / 1000.0);
      TitanUI::log_time(log, "Decomposition", decomp_ms / 1000.0);
      TitanUI::log_time(log, "Entropy + Output",
                        pbsa_entropy_output_ms_ / 1000.0);
      TitanUI::log_time(log, "Frame Total", frame_ms / 1000.0);
    }
    if (effective_pbsa_decomp() != "off")
      TitanUI::log_kv(log, "Decomposition", "mm_pbsa_decomposition.dat");
    if (effective_pbsa_decomp() == "residue")
      TitanUI::log_kv(log, "Residue Summary",
                      "mm_pbsa_residue_summary.csv");
    if (effective_pbsa_decomp() == "hybrid")
      TitanUI::log_kv(log, "Unit Summary", "mm_pbsa_unit_summary.csv");
  }

  if (pbsa_failed_ > 0) {
    throw std::runtime_error(
        "MM/PBSA strict failure policy: " + std::to_string(pbsa_failed_) +
        " of " + std::to_string(pbsa_frames_) +
        " frame(s) were excluded because an endpoint did not converge or an "
        "energy was non-finite. Partial diagnostic outputs were written, but "
        "the run is not scientifically valid.");
  }
}

} // namespace TitanPBE

#include "cas_analysis.h"
#include "gbsa_intrin.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

std::string trim_copy(std::string value) {
  const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(),
                           [&](unsigned char c) { return !is_space(c); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](unsigned char c) { return !is_space(c); })
                  .base(),
              value.end());
  return value;
}

std::string uppercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

bool is_retained_alanine_atom(const std::string &atom_name) {
  const std::string atom = uppercase_copy(trim_copy(atom_name));
  return atom == "N" || atom == "H" || atom == "HN" || atom == "CA" ||
         atom == "HA" || atom == "HA1" || atom == "HA2" || atom == "C" ||
         atom == "O" || atom == "OXT" || atom == "CB" || atom == "HB" ||
         atom == "HB1" || atom == "HB2" || atom == "HB3";
}

CASResidueKey make_key(const std::string &role, const mol_info &info) {
  return CASResidueKey{role,
                       info.get_chainnum(),
                       info.get_chain(),
                       info.get_resnr(),
                       info.get_res_name()};
}

CASResidueKey make_contact_key(const std::string &role, std::size_t chainnum,
                               const std::string &chain, std::size_t resnr,
                               const std::string &resname) {
  return CASResidueKey{role, chainnum, chain, resnr, resname};
}

std::string status_for_residue(const std::string &resname,
                               const std::vector<CASMutationAtom> &plan) {
  const std::string res = uppercase_copy(trim_copy(resname));
  if (res == "ALA")
    return "noop_alanine";
  if (res == "GLY" || plan.empty())
    return "no_mutatable_sidechain";
  return "valid_mutation";
}

bool status_is_mutatable(const std::string &status) {
  return status == "valid_mutation";
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char c : value) {
    if (c == '"')
      out += "\"\"";
    else
      out += c;
  }
  out += '"';
  return out;
}

std::vector<CAS_Result>
sorted_rows(const std::unordered_map<CASResidueKey, CAS_Result,
                                    CASResidueKeyHash> &data) {
  std::vector<CAS_Result> rows;
  rows.reserve(data.size());
  for (const auto &[_, result] : data)
    rows.push_back(result);
  std::sort(rows.begin(), rows.end(), [](const CAS_Result &lhs,
                                         const CAS_Result &rhs) {
    if (lhs.key.role != rhs.key.role)
      return lhs.key.role < rhs.key.role;
    if (lhs.key.chainnum != rhs.key.chainnum)
      return lhs.key.chainnum < rhs.key.chainnum;
    if (lhs.key.chain != rhs.key.chain)
      return lhs.key.chain < rhs.key.chain;
    if (lhs.key.resnr != rhs.key.resnr)
      return lhs.key.resnr < rhs.key.resnr;
    return lhs.key.resname < rhs.key.resname;
  });
  return rows;
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

std::size_t CASResidueKeyHash::operator()(const CASResidueKey &key) const
    noexcept {
  std::size_t seed = std::hash<std::string>{}(key.role);
  auto mix = [&](std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  };
  mix(std::hash<std::size_t>{}(key.chainnum));
  mix(std::hash<std::string>{}(key.chain));
  mix(std::hash<std::size_t>{}(key.resnr));
  mix(std::hash<std::string>{}(key.resname));
  return seed;
}

CASAnalyzer::CASAnalyzer() : global_total_frames(0), has_reference(false) {}

void CASAnalyzer::init(Multi_MOL &pro_in, Multi_MOL &lig,
                       const std::string &mol_2_type) {
  auto init_cas_data =
      [&](Multi_MOL &mol, const std::string &role,
          std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
              &cas_data,
          std::unordered_map<CASResidueKey, std::vector<CASMutationAtom>,
                             CASResidueKeyHash> &mutation_plans,
          std::vector<CASResidueKey> &unique) {
        unique.clear();
        mutation_plans.clear();

        for (const auto &atom_info : mol.info) {
          const CASResidueKey key = make_key(role, atom_info);
          if (cas_data.find(key) == cas_data.end()) {
            unique.push_back(key);
            CAS_Result result;
            result.key = key;
            cas_data.emplace(key, std::move(result));
          }
        }

        mutation_plans.reserve(unique.size());
        for (size_t atom_idx = 0; atom_idx < mol.info.size(); ++atom_idx) {
          const auto &atom_info = mol.info[atom_idx];
          if (is_retained_alanine_atom(atom_info.get_atom()))
            continue;

          const CASResidueKey key = make_key(role, atom_info);
          mutation_plans[key].push_back(CASMutationAtom{
              atom_idx,
              mol.parameters[atom_idx].get_charge(),
              mol.parameters[atom_idx].get_epsilon(),
              mol.parameters[atom_idx].get_sigma(),
              mol.parameters[atom_idx].get_radius(),
              mol.parameters[atom_idx].get_type_id()});
        }

        for (auto &key : unique) {
          auto &result = cas_data[key];
          const auto plan_it = mutation_plans.find(key);
          static const std::vector<CASMutationAtom> empty_plan;
          const auto &plan =
              plan_it == mutation_plans.end() ? empty_plan : plan_it->second;
          result.mutation_status = status_for_residue(key.resname, plan);
        }
      };

  if (cas_receptor_data.empty()) {
    init_cas_data(pro_in, "mol1", cas_receptor_data, receptor_mutation_plans,
                  unique_res_rec);
  }

  ligand_scan_enabled_ = mol_2_type.find("protein") != std::string::npos;
  if (cas_ligand_data.empty() && ligand_scan_enabled_) {
    init_cas_data(lig, "mol2", cas_ligand_data, ligand_mutation_plans,
                  unique_res_lig);
  }
}

void CASAnalyzer::process_frame(Multi_MOL &pro_in, Multi_MOL &lig,
                                size_t num_snap, TitanExecutor &executor,
                                size_t num_threads,
                                const std::vector<d_min> &dis_min,
                                cache_sigma_epsilon &CachedData,
                                const arguments &Arguments, info &info_output,
                                const AlignedDoubleVector &rec_free_radii,
                                const AlignedDoubleVector &lg_free_radii,
                                const AlignedDoubleVector &rec_complex_radii,
                                const AlignedDoubleVector &lg_complex_radii,
                                const GBSA_energy_snap &frame_gbsa_total) {
  (void)info_output;
  const auto cas_begin = std::chrono::steady_clock::now();

  global_total_frames++;
  perf_.frames++;
  if (!has_reference) {
    ref_pro_frame = pro_in.m_mol[num_snap];
    if (!lig.m_mol.empty()) {
      ref_lig_frame = lig.m_mol[(lig.m_mol.size() > 1) ? num_snap : 0];
    }
    ref_pro_info = pro_in.info;
    ref_lig_info = lig.info;
    has_reference = true;
  }

  size_t total_contacts = 0;
  for (const auto &thread_contacts : dis_min)
    total_contacts += thread_contacts.base.size();

  std::unordered_map<CASResidueKey, double, CASResidueKeyHash> rec_min_dist;
  std::unordered_map<CASResidueKey, double, CASResidueKeyHash> lig_min_dist;
  rec_min_dist.reserve(total_contacts);
  lig_min_dist.reserve(total_contacts);

  auto update_min_distance =
      [](std::unordered_map<CASResidueKey, double, CASResidueKeyHash> &min_map,
         const CASResidueKey &key, double distance) {
        const auto [it, inserted] = min_map.emplace(key, distance);
        if (!inserted && distance < it->second)
          it->second = distance;
      };

  for (const auto &thread_contacts : dis_min) {
    for (const auto &contact : thread_contacts.base) {
      update_min_distance(
          rec_min_dist,
          make_contact_key("mol1", contact.chainid_rec, contact.chain_rec,
                           contact.resnr_rec, contact.res_name_rec),
          contact.d);
      update_min_distance(
          lig_min_dist,
          make_contact_key("mol2", contact.chainid_ab, contact.chain_ab,
                           contact.resnr_ab, contact.res_name_ab),
          contact.d);
    }
  }

  auto apply_mutation_plan = [](const std::vector<CASMutationAtom> &plan,
                                Multi_MOL_parameters &parameters) {
    for (const auto &atom : plan) {
      parameters[atom.atom_index].set_charge(0.0);
      parameters[atom.atom_index].set_epsilon_sigma_radius(0.0, 0.0, 0.001);
      parameters[atom.atom_index].set_type_id(-1);
    }
  };

  auto restore_mutation_plan = [](const std::vector<CASMutationAtom> &plan,
                                  Multi_MOL_parameters &parameters) {
    for (const auto &atom : plan) {
      parameters[atom.atom_index].set_charge(atom.charge);
      parameters[atom.atom_index].set_epsilon_sigma_radius(
          atom.epsilon, atom.sigma, atom.radius);
      parameters[atom.atom_index].set_type_id(atom.type_id);
    }
  };

  auto perform_cas_scan =
      [&](Multi_MOL &mol,
          std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
              &cas_data,
          const std::unordered_map<CASResidueKey, double, CASResidueKeyHash>
              &active_contact_distances,
          const std::unordered_map<CASResidueKey, std::vector<CASMutationAtom>,
                                   CASResidueKeyHash> &mutation_plans,
          bool mutate_receptor) {
        if (active_contact_distances.empty())
          return;

        std::vector<CASResidueKey> active_residues;
        std::vector<double> active_min_distances;
        active_residues.reserve(active_contact_distances.size());
        active_min_distances.reserve(active_contact_distances.size());
        for (const auto &[key, min_distance] : active_contact_distances) {
          if (cas_data.find(key) == cas_data.end())
            continue;
          active_residues.push_back(key);
          active_min_distances.push_back(min_distance);
        }
        if (active_residues.empty())
          return;

        const size_t n_res = active_residues.size();
        perf_.active_residue_evaluations += n_res;

        const size_t active_lg_snap = (lig.m_mol.size() > 1) ? num_snap : 0;
        const MOL &active_lg_mol = lig.m_mol[active_lg_snap];
        const size_t rec_size_ = pro_in.m_mol[num_snap].size();
        const size_t active_lg_size_ = active_lg_mol.size();
        const MOL &worker_frame = mol.m_mol[num_snap];
        const size_t m_size = worker_frame.size();
        const size_t partner_size = mutate_receptor ? active_lg_size_ : rec_size_;

        std::vector<CAS_Result> frame_results(n_res);
        for (size_t idx = 0; idx < n_res; ++idx) {
          auto &result = frame_results[idx];
          result.key = active_residues[idx];
          result.mutation_status = cas_data[result.key].mutation_status;
        }

        const size_t requested_workers =
            std::max<size_t>(1, std::min(num_threads, n_res));
        worker_scratch_.resize(requested_workers);
        std::vector<double> local_gbsa_ms(requested_workers, 0.0);
        std::vector<std::size_t> local_mutations(requested_workers, 0);
        std::vector<std::size_t> local_skipped_ala(requested_workers, 0);
        std::vector<std::size_t> local_skipped_no_sidechain(requested_workers,
                                                           0);

        for (size_t worker_id = 0; worker_id < requested_workers; ++worker_id) {
          auto &scratch = worker_scratch_[worker_id];
          scratch.mut_free_radii.resize(m_size);
          scratch.mut_complex_radii.resize(m_size);
          scratch.partner_complex_radii.resize(partner_size);
          scratch.parameters = mol.parameters;
        }

        executor.parallel_for(
            n_res, requested_workers, 1,
            [&](size_t worker_id, size_t begin, size_t end) {
              auto &scratch = worker_scratch_[worker_id];
              for (size_t ri = begin; ri < end; ++ri) {
                const CASResidueKey &key = active_residues[ri];
                const double min_d = active_min_distances[ri];
                auto &result = frame_results[ri];
                result.distance_sum += min_d;
                result.contact_frames++;

                if (result.mutation_status == "noop_alanine") {
                  local_skipped_ala[worker_id]++;
                  continue;
                }
                if (result.mutation_status == "no_mutatable_sidechain") {
                  local_skipped_no_sidechain[worker_id]++;
                  continue;
                }

                const auto plan_it = mutation_plans.find(key);
                if (plan_it == mutation_plans.end() || plan_it->second.empty()) {
                  local_skipped_no_sidechain[worker_id]++;
                  continue;
                }

                local_mutations[worker_id]++;
                const auto gbsa_begin = std::chrono::steady_clock::now();
                apply_mutation_plan(plan_it->second, scratch.parameters);

                std::fill(scratch.mut_free_radii.begin(),
                          scratch.mut_free_radii.end(), 0.0);
                std::fill(scratch.mut_complex_radii.begin(),
                          scratch.mut_complex_radii.end(), 0.0);

                // Mutant endpoints use the same rgbmax as the wild-type run,
                // otherwise the alanine-scan difference mixes conventions.
                // The partner's free radii are untouched by the mutation, but
                // its complex radii are not: the deleted side-chain atoms no
                // longer descreen it, so they are recomputed with the mutant
                // as environment. Reusing the wild-type complex radii would
                // leave the partner buried by atoms that are gone.
                if (mutate_receptor && active_lg_size_ > 0) {
                  cal_free_and_complex_born_radii_avx2(
                      worker_frame, scratch.parameters.radius, mol.info,
                      scratch.mut_free_radii, scratch.mut_complex_radii,
                      Arguments.igb, 0, m_size, active_lg_mol,
                      lig.parameters.radius, lig.info, Arguments.gbsa_rgbmax,
                      Arguments.gbnsr6_spacing);
                  cal_effective_born_radii_avx2(
                      active_lg_mol, lig.parameters.radius, lig.info,
                      scratch.partner_complex_radii, Arguments.igb, 0,
                      active_lg_size_, &worker_frame,
                      &scratch.parameters.radius, &mol.info,
                      Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
                } else if (!mutate_receptor && rec_size_ > 0) {
                  cal_free_and_complex_born_radii_avx2(
                      worker_frame, scratch.parameters.radius, mol.info,
                      scratch.mut_free_radii, scratch.mut_complex_radii,
                      Arguments.igb, 0, m_size, pro_in.m_mol[num_snap],
                      pro_in.parameters.radius, pro_in.info,
                      Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
                  cal_effective_born_radii_avx2(
                      pro_in.m_mol[num_snap], pro_in.parameters.radius,
                      pro_in.info, scratch.partner_complex_radii,
                      Arguments.igb, 0, rec_size_, &worker_frame,
                      &scratch.parameters.radius, &mol.info,
                      Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
                } else {
                  cal_effective_born_radii_avx2(
                      worker_frame, scratch.parameters.radius,
                      mol.info, scratch.mut_free_radii, Arguments.igb, 0, m_size,
                      nullptr, nullptr, nullptr, Arguments.gbsa_rgbmax, Arguments.gbnsr6_spacing);
                }

                GBSA_energy_snap mut_gbsa_total;
                if (mutate_receptor) {
                  cal_energy_gbsa_avx2(
                      worker_frame, active_lg_mol, mol.info, lig.info,
                      CachedData, 0, m_size, 0, active_lg_size_,
                      mut_gbsa_total, num_snap, Arguments, scratch.parameters,
                      lig.parameters, mol.lcpo_p1, mol.lcpo_p2, lig.lcpo_p1,
                      lig.lcpo_p2, scratch.mut_free_radii, lg_free_radii,
                      scratch.mut_complex_radii, scratch.partner_complex_radii,
                      nullptr, nullptr, false);
                } else {
                  cal_energy_gbsa_avx2(
                      pro_in.m_mol[num_snap], worker_frame, pro_in.info,
                      mol.info, CachedData, 0, rec_size_, 0, m_size,
                      mut_gbsa_total, num_snap, Arguments, pro_in.parameters,
                      scratch.parameters, pro_in.lcpo_p1, pro_in.lcpo_p2,
                      mol.lcpo_p1, mol.lcpo_p2, rec_free_radii,
                      scratch.mut_free_radii, scratch.partner_complex_radii,
                      scratch.mut_complex_radii, nullptr, nullptr, false);
                }

                restore_mutation_plan(plan_it->second, scratch.parameters);
                local_gbsa_ms[worker_id] += elapsed_ms(
                    gbsa_begin, std::chrono::steady_clock::now());

                result.ddG_vdw +=
                    (mut_gbsa_total.E_vdw - frame_gbsa_total.E_vdw);
                result.ddG_elec +=
                    (mut_gbsa_total.E_elec - frame_gbsa_total.E_elec);
                result.ddG_polar +=
                    (mut_gbsa_total.G_polar - frame_gbsa_total.G_polar);
                result.ddG_nonpolar +=
                    (mut_gbsa_total.G_nonpolar -
                     frame_gbsa_total.G_nonpolar);
                result.ddG_total +=
                    (mut_gbsa_total.total_energy -
                     frame_gbsa_total.total_energy);
              }
            });

        for (size_t worker_id = 0; worker_id < requested_workers; ++worker_id) {
          perf_.gbsa_mutation_ms += local_gbsa_ms[worker_id];
          perf_.mutation_evaluations += local_mutations[worker_id];
          perf_.skipped_noop_alanine += local_skipped_ala[worker_id];
          perf_.skipped_no_mutatable_sidechain +=
              local_skipped_no_sidechain[worker_id];
        }

        for (const auto &result : frame_results) {
          auto &target = cas_data[result.key];
          target.ddG_vdw += result.ddG_vdw;
          target.ddG_elec += result.ddG_elec;
          target.ddG_polar += result.ddG_polar;
          target.ddG_nonpolar += result.ddG_nonpolar;
          target.ddG_total += result.ddG_total;
          target.distance_sum += result.distance_sum;
          target.contact_frames += result.contact_frames;
        }
      };

  perform_cas_scan(pro_in, cas_receptor_data, rec_min_dist,
                   receptor_mutation_plans, true);
  if (ligand_scan_enabled_) {
    perform_cas_scan(lig, cas_ligand_data, lig_min_dist, ligand_mutation_plans,
                     false);
  } else if (!lig_min_dist.empty()) {
    perf_.ligand_scan_skipped_nonprotein_frames++;
  }

  perf_.cas_total_ms += elapsed_ms(cas_begin, std::chrono::steady_clock::now());
}

void CASAnalyzer::export_results(const std::string &rec_file,
                                 const std::string &lig_file, double eu_par) {
  if (global_total_frames == 0)
    return;

  auto write_data =
      [&](const std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
              &cas_data,
          const std::string &file) {
        if (cas_data.empty())
          return;

        std::ofstream out(file);
        out << "role,chainnum,chain,resnr,resname,mutation_status,"
               "occupancy_fraction,ddG_total_mean,ddG_total_bound_mean,"
               "ddG_elec_mean,ddG_vdw_mean,ddG_polar_mean,ddG_nonpolar_mean,"
               "avg_distance_A,contact_frames,total_frames\n";

        for (const auto &res : sorted_rows(cas_data)) {
          if (res.contact_frames == 0)
            continue;

          const double occupancy =
              static_cast<double>(res.contact_frames) / global_total_frames;
          if (occupancy < 0.10)
            continue;

          const double frame_norm =
              global_total_frames == 0 ? 1.0
                                       : static_cast<double>(global_total_frames);
          const double contact_norm =
              res.contact_frames == 0 ? 1.0
                                      : static_cast<double>(res.contact_frames);
          // ddG is accumulated in kJ/mol; eu_par is kJ per output unit
          // (4.184 for kcal/mol), so the output value is kJ / eu_par, as in
          // every other TITAN energy file.
          const double total_mean = (res.ddG_total / frame_norm) / eu_par;
          const double total_bound = (res.ddG_total / contact_norm) / eu_par;
          const double elec_mean = (res.ddG_elec / frame_norm) / eu_par;
          const double vdw_mean = (res.ddG_vdw / frame_norm) / eu_par;
          const double polar_mean = (res.ddG_polar / frame_norm) / eu_par;
          const double nonpolar_mean =
              (res.ddG_nonpolar / frame_norm) / eu_par;
          const double avg_distance = res.distance_sum / contact_norm;

          out << csv_escape(res.key.role) << ',' << res.key.chainnum << ','
              << csv_escape(res.key.chain) << ',' << res.key.resnr << ','
              << csv_escape(res.key.resname) << ','
              << csv_escape(res.mutation_status) << ',' << std::fixed
              << std::setprecision(6) << occupancy << ',' << total_mean << ','
              << total_bound << ',' << elec_mean << ',' << vdw_mean << ','
              << polar_mean << ',' << nonpolar_mean << ',' << avg_distance
              << ',' << res.contact_frames << ',' << global_total_frames
              << '\n';
        }
      };

  write_data(cas_receptor_data, rec_file);
  write_data(cas_ligand_data, lig_file);
}

void CASAnalyzer::export_hotspot_pdb(const std::string &out_pdb_filename,
                                     double eu_par) {
  if (!has_reference || global_total_frames == 0)
    return;

  FILE *out_file = fopen(out_pdb_filename.c_str(), "w");
  if (!out_file)
    return;

  fprintf(out_file,
          "REMARK TITAN CAS B-factor stores ddG_total_mean for valid GBSA "
          "alanine-like side-chain neutralization rows only.\n");
  fprintf(out_file,
          "REMARK CAS does not relax or repack mutated structures; inspect "
          "cas_summary.dat and mutation_status before interpretation.\n");

  int atom_serial = 1;

  auto write_atoms =
      [&](const MOL &mol, const std::vector<mol_info> &info,
          const std::unordered_map<CASResidueKey, CAS_Result, CASResidueKeyHash>
              &cas_data,
          const std::string &role, size_t chain_offset) {
        for (size_t i = 0; i < mol.size(); ++i) {
          const CASResidueKey key = make_key(role, info[i]);
          double b_factor = 0.00;
          const auto it = cas_data.find(key);
          if (it != cas_data.end()) {
            const CAS_Result &res = it->second;
            const double occupancy =
                static_cast<double>(res.contact_frames) / global_total_frames;
            if (occupancy >= 0.10 && status_is_mutatable(res.mutation_status))
              b_factor = (res.ddG_total / global_total_frames) / eu_par;
          }

          const size_t cid = info[i].get_chainnum() + chain_offset;
          const char chain_char =
              (cid < 26) ? ('A' + static_cast<char>(cid))
                         : ('a' + static_cast<char>(cid - 26));

          fprintf(out_file,
                  "ATOM  %5d %-4.4s %-4.4s%c%4zu    "
                  "%8.3f%8.3f%8.3f%6.2f%6.2f\n",
                  atom_serial++, info[i].get_atom().c_str(),
                  info[i].get_res_name().c_str(), chain_char,
                  static_cast<size_t>(info[i].get_resnr()), mol.pos.x[i],
                  mol.pos.y[i], mol.pos.z[i], 1.00, b_factor);
        }
      };

  size_t max_pro_chainid = 0;
  for (const auto &mi : ref_pro_info)
    max_pro_chainid = std::max(max_pro_chainid, mi.get_chainnum());

  write_atoms(ref_pro_frame, ref_pro_info, cas_receptor_data, "mol1", 0);
  if (ref_lig_frame.size() > 0) {
    fprintf(out_file, "TER\n");
    write_atoms(ref_lig_frame, ref_lig_info, cas_ligand_data, "mol2",
                max_pro_chainid + 1);
  }
  fprintf(out_file, "END\n");
  fclose(out_file);
}

void CASAnalyzer::export_summary(const std::string &summary_file,
                                 const arguments &Arguments) const {
  std::ofstream out(summary_file);
  out << "method: GBSA alanine-like side-chain neutralization scan\n";
  out << "gbsa_model_igb: " << Arguments.igb << '\n';
  out << "cutoff_source: -dsc contact list, " << Arguments.d_standard
      << " A\n";
  out << "occupancy_output_cutoff: 0.10\n";
  out << "mutation_policy: retain backbone and alanine atom set; zero charge, "
         "epsilon, sigma and set radius 0.001 for other side-chain atoms\n";
  out << "born_radii_policy: per mutation, mutant free and complex radii and "
         "partner complex radii (mutant as environment) are recomputed; "
         "partner free radii are the wild-type ones\n";
  out << "energy_unit: " << Arguments.eu << '\n';
  out << "limitation_note: CAS does not perform structural relaxation, side-chain "
         "repacking, or explicit mutant topology regeneration.\n";
  out << "reference_note: hotspot PDB uses first analyzed frame coordinates and "
         "stores ddG_total_mean in B-factor for valid_mutation rows only.\n";
  out << "frames: " << perf_.frames << '\n';
  out << "receptor_residues: " << cas_receptor_data.size() << '\n';
  out << "ligand_scan_enabled: "
      << (ligand_scan_enabled_ ? "true" : "false") << '\n';
  out << "ligand_residues: " << cas_ligand_data.size() << '\n';
  if (!ligand_scan_enabled_)
    out << "ligand_note: mol2 is not protein-like; ligand CAS scan skipped.\n";
}

void CASAnalyzer::export_performance(const std::string &performance_file) const {
  std::ofstream out(performance_file);
  out << "frames " << perf_.frames << '\n';
  out << "active_residue_evaluations " << perf_.active_residue_evaluations
      << '\n';
  out << "mutation_evaluations " << perf_.mutation_evaluations << '\n';
  out << "skipped_noop_alanine " << perf_.skipped_noop_alanine << '\n';
  out << "skipped_no_mutatable_sidechain "
      << perf_.skipped_no_mutatable_sidechain << '\n';
  out << "ligand_scan_skipped_nonprotein_frames "
      << perf_.ligand_scan_skipped_nonprotein_frames << '\n';
  out << std::fixed << std::setprecision(3);
  out << "gbsa_mutation_ms " << perf_.gbsa_mutation_ms << '\n';
  out << "cas_total_ms " << perf_.cas_total_ms << '\n';
}

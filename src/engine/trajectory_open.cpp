#include "engine/engine_run.h"
#include "topology/fflib_assigner.h"
#include "gbsa/gb_topology_pairs.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::open_trajectory() {
  trajectory_timing_start = std::chrono::steady_clock::now();
  time_gbsa_base = time_gbsa;
  time_cas_base = time_cas;
  time_dssp_base = time_dssp;
  time_hbk_base = time_hbk;
  time_lj_base = time_lj;
  time_sq_base = time_sq;
  time_cmap_base = time_cmap;
  time_interface_base = time_interface;
  time_dihedral_base = time_dihedral;
  time_qnative_base = time_qnative;
  time_msd_base = time_msd;
  time_membrane_base = time_membrane;
  time_s2_base = time_s2;
  time_rg_base = time_rg;
  time_rms_base = time_rms;
  time_network_base = time_network;
  time_tica_base = time_tica;
  time_cal_base = time_cal;
  time_sc_base = time_sc;
  time_cscd_base = time_cscd;
  time_hb_base = time_hb;
  time_pdb_base = time_pdb;
  time_fmap_base = time_fmap;
  time_other_base = time_other;
  time_water_base = time_water;
  time_subgroups_cal_base = time_subgroups_cal;
  time_na_base = time_na;
  time_ifp_base = time_ifp;
  time_input_init_base = time_input_init;
  time_input_stream_base = time_input_stream;
  time_solve_pbc_base = time_solve_pbc;
  time_llps_base = time_llps;
  time_pbe_base = time_pbe;

  if (!input_line.empty()) {
    if (Arguments.flag_all_folders == false) {
      name_lig = "lg_" + input_line[input_name - Arguments.b_input] +
                 Arguments.get_input_file_type();
      name_pro = "rec_" + input_line[input_name - Arguments.b_input] +
                 Arguments.get_input_file_type();
    } else {

      name_lig = fs::path(input_line[input_name - Arguments.b_input])
                     .parent_path()
                     .string() +
                 "/lg_" +
                 fs::path(input_line[input_name - Arguments.b_input])
                     .stem()
                     .string() +
                 Arguments.get_input_file_type();
      name_pro = fs::path(input_line[input_name - Arguments.b_input])
                     .parent_path()
                     .string() +
                 "/rec_" +
                 fs::path(input_line[input_name - Arguments.b_input])
                     .stem()
                     .string() +
                 Arguments.get_input_file_type();
    }
  } else if (Arguments.flag_1_pdb == false &&
             Arguments.flag_all_folders == false && Arguments.flag_listfile) {
    name_lig = Arguments.input_pattern;
    name_pro = Arguments.input_pattern;
  } else {
    name_lig =
        "lg_" + to_string(input_name) + Arguments.get_input_file_type();
    name_pro =
        "rec_" + to_string(input_name) + Arguments.get_input_file_type();
  }

  // vector<vector<com> > COM_vec_rec, COM_vec_lg;

  // Water Bridge Tracking
  if (Arguments.n_wat >= 0 && input_name == Arguments.b_input) {
    std::ofstream wb_out(info_output.get_global_water_bridges_timeseries());
    wb_out << "Traj_ID,Frame,Time_ps,Water_ID,Protein_Atom,Ligand_Atom,"
              "Distance_A,Angle_deg\n";
  }
  water_analyzer.reset_trajectory_state();

  start_input_init = std::chrono::steady_clock::time_point{std::chrono::steady_clock::now()};

  // --- Total File Size ---
  total_traj_bytes = 1;
  ref_file = Arguments.flag_1_pdb ? Arguments.pdb_name : name_pro;
  if (std::filesystem::exists(ref_file, ec)) {
    total_traj_bytes = std::filesystem::file_size(ref_file, ec);
    if (total_traj_bytes == 0)
      total_traj_bytes = 1;
  }

  // Initialize the topology (Frame 0) and setup groups
  if (!topo_initialized) {
    auto init_result = initialize_topology_streaming(
        Arguments, log, name_pro, name_lig, force_f, input_name, input_line,
        topo_2_is_open, topo_is_open, pro_in, lig, water_traj, membrane_traj,
        Groups_from_index_file, parameter, amine, info_output);

    if (!init_result) {
      if (Arguments.gist.flag_gist) {
        TitanUI::print_error("GIST initialization failed: " + init_result.error());
        return 1;
      }
      // A VIBE run (-pbctopo) writes nothing without its topology, so a
      // failed initialization must not end as a successful nocal run.
      if (Arguments.flag_nocal && Arguments.pbc.solv_pbctopo) {
        TitanUI::print_error("Initialization failed: " + init_result.error());
        return 1;
      }
      if (Arguments.flag_nocal) {
        TitanUI::print_info("Nocal mode is turned on!");
        return 0; // nocal: init parsed what it could, done
      }
      if (input_name > Arguments.b_input)
        --input_name;
      status = false;
      TitanUI::print_error("Initialization failed: " + init_result.error());
      return stop_trajectories();
    }
    topo_initialized = true;

  } else {
    // Subsequent files: ONLY reload Frame 0 coordinates, skip topo parsing
    if (Arguments.flag_1_pdb && Arguments.flag_xdr) {
      // XDR multi-trajectory mode reuses the same reference topology (-s)
      // for every replica/chunk listed in -i. Re-parsing the reference via
      // pre_proc_pdb(..., -1, ...) clears topology-derived state
      // (parameters/index metadata) and can leave later analyses with an
      // inconsistent object graph when the next XTC/TRR starts streaming.
      // Keep the first parsed topology and restore the cached reference frame
      // instead; only the active trajectory filename changes per file.
      if (!saved_pro_frame0.empty())
        pro_in.m_mol = saved_pro_frame0;
      if (!saved_lig_frame0.empty())
        lig.m_mol = saved_lig_frame0;
      water_traj.m_mol.clear();
      if (!saved_membrane_frame0.empty())
        membrane_traj.m_mol = saved_membrane_frame0;
    } else {
      pro_in.m_mol.clear();
      lig.m_mol.clear();
      water_traj.m_mol.clear();
      membrane_traj.m_mol.clear();

      // Reload coordinates-only from new file (flag_raw=1 = coords only, no
      // topo)
      if (Arguments.flag_1_pdb) {
        // Preserve topology-derived parameters across replicas while
        // refreshing the per-file metadata copied from the reference
        // structure.
        pro_in.info.clear();
        lig.info.clear();
        if (Arguments.n_wat >= 0)
          water_traj.info.clear();
        if (Arguments.needs_membrane_coordinates())
          membrane_traj.info.clear();
        const int r =
            pre_proc_pdb(force_f, pro_in, lig, water_traj, membrane_traj,
                         Groups_from_index_file, -1, Arguments, amine,
                         info_output);
        if (r != 0) {
          TitanUI::print_error("Failed to reload coordinates for: " +
                               Arguments.pdb_name);
          return stop_trajectories();
        }
      } else {
        pro_in.take_coordinate(name_pro, force_f, Arguments, amine,
                               info_output, 1, 1);
        lig.take_coordinate(name_lig, force_f, Arguments, amine, info_output,
                            1, 1);
      }
    }

    if (pro_in.m_mol.empty() ||
        (Arguments.needs_mol2_coordinates() && lig.m_mol.empty())) {
      TitanUI::print_error("Coordinate reload failed: empty mol for " +
                           name_pro);
      return stop_trajectories();
    }

    if (Arguments.flag_1_pdb && !input_line.empty()) {
      if (Arguments.flag_xdr) {
        // XDR multi-replica: keep the -s reference PDB, switch trajectory
        Arguments.traj_file = input_line[input_name - Arguments.b_input];
      } else {
        Arguments.pdb_name = input_line[input_name - Arguments.b_input];
        Arguments.add_input_file_type();
      }
    }

    Arguments.setPdbname(Arguments.flag_1_pdb
                             ? Arguments.pdb_name
                             : name_lig.substr(name_lig.find("_")));
    info_output.setPdbName(Arguments);

    TitanUI::log_kv(
        log, "Topo Cache",
        "Reusing topology, reloaded coords from: " +
            (Arguments.flag_1_pdb ? Arguments.pdb_name : name_pro));
  }

  info_output.get_info_group(Groups_from_index_file);
  process_time_of_infiles(pro_in, Arguments, info_output);
  set_chainnum(pro_in, lig, Arguments);

  if (Arguments.opdb == true && Arguments.flag_sep_opdb == false) {
    fstream out_pdb(Arguments.get_pdbOutput(), ios::out);
    out_pdb.close();
  }

  return std::nullopt;
}

std::optional<int> EngineTrajectory::prepare_topology() {

  // --- CHARGE INITIALIZATION (Frame 0) ---
  needs_charge_init =
        !Arguments.flag_nocal || Arguments.flag_gbsa || Arguments.flag_alpb ||
        Arguments.cmap.flag_cmap || Arguments.qnative.flag_qnative ||
        Arguments.flag_cal_force || Arguments.flag_maxforce ||
        Arguments.flag_use_CI ||
        Arguments.flag_cal_dipole_moment ||
        Arguments.flag_charged_atom_distance_dis || Arguments.flag_cas ||
        Arguments.ifp.flag_ifp || Arguments.flag_pqr == 1 ||
        TitanFfLib::parameter_audit_path() != nullptr;
  if (!charge_initialized && !needs_charge_init) {
    TitanUI::log_kv(log, "Charge Init",
                    "Skipped (no charge-dependent module active)");
    charge_initialized = true;
  }
  if (!charge_initialized) {
    const bool initialize_mol2_parameters =
        Arguments.needs_mol2_coordinates() && !lig.m_mol.empty();
    const auto charge_policy = TitanFfLib::resolve_charge_source_policy(
        Arguments.flag_use_common_topo, topo_is_open, topo_2_is_open);
    // ENE-08 guard: refuse a nonbonded table in which a pair/NBFIX or
    // malformed row has shadowed a real atom type with sigma = epsilon = 0.
    const auto table_check = TitanFfLib::validate_nonbonded_table(parameter);
    TitanUI::log_kv(log, "Nonbonded Type Table",
                    std::to_string(table_check.entries) + " entries, " +
                        std::to_string(table_check.duplicate_types) +
                        " type name(s) defined more than once (last definition wins)");
    if (table_check.shadowed_types > 0) {
      std::string examples;
      for (const auto &name : table_check.shadowed_examples)
        examples += (examples.empty() ? "" : " ") + name;
      const std::string message =
          "invalid_nonbonded_type_table: " +
          std::to_string(table_check.shadowed_types) +
          " atom type(s) whose active definition has sigma = epsilon = 0 while an "
          "earlier definition is non-zero (" + examples +
          "); a pair/NBFIX or malformed row was read as an atom type. Refusing to "
          "assign LJ parameters.";
      TitanUI::print_error(message);
      TitanUI::log_message(log, "[Validation] " + message);
      return 1;
    }
    TitanUI::log_kv(log, "Charge Source Mol1",
                    charge_policy.molecule1_uses_topology ? "topology"
                                                          : "lib_ff");
    if (initialize_mol2_parameters)
      TitanUI::log_kv(log, "Charge Source Mol2",
                      charge_policy.molecule2_uses_topology ? "topology"
                                                            : "lib_ff");
    pro_in.take_charge(
        amine, parameter, pro_in.m_mol[0].size(), 0ull, force_f,
        charge_policy.molecule1_uses_topology ? 1 : 0, &Arguments, &executor,
        "mol1");
    if (initialize_mol2_parameters)
      lig.take_charge(
          amine, parameter, lig.m_mol[0].size(), 0ull, force_f,
          charge_policy.molecule2_uses_topology ? 1 : 0, &Arguments,
          &executor, "mol2");
    if (const char *audit_path = TitanFfLib::parameter_audit_path()) {
      const auto audit = TitanFfLib::write_parameter_audit(
          audit_path, pro_in, initialize_mol2_parameters ? &lig : nullptr,
          charge_policy);
      if (!audit) {
        TitanUI::print_error(audit.error());
        TitanUI::log_kv(log, "Parameter Audit Error", audit.error());
        return 1;
      }
      TitanUI::log_kv(log, "Parameter Audit", audit_path);
    }
    charge_initialized = true;
  }

  auto_adjust_polymer_type(Arguments, pro_in, log, true);
  auto_adjust_polymer_type(Arguments, lig, log, false);
  resolved_mol1_type =
        (Arguments.flag_nocal && Arguments.lj.flag_lj) ? "lj"
                                                       : Arguments.mol_1_type;
  resolved_mol2_type =
        (Arguments.flag_nocal && Arguments.lj.flag_lj) ? "none"
                                                       : Arguments.mol_2_type;
  TitanUI::log_kv(log, "Resolved Molecule 1 Type", resolved_mol1_type);
  TitanUI::log_kv(log, "Resolved Molecule 2 Type", resolved_mol2_type);
  global_has_nucleic_acid =
      PolymerClassification::is_nucleic_type(Arguments.mol_1_type) ||
      PolymerClassification::is_nucleic_type(Arguments.mol_2_type) ||
      !PolymerClassification::detect_nucleic_polymer_type(pro_in.info).empty() ||
      !PolymerClassification::detect_nucleic_polymer_type(lig.info).empty();
  if (global_has_nucleic_acid) {
    na_profiler.initialize_if_needed(pro_in.Index, lig.Index, pro_in.info,
                                     lig.info);
  }

  CachedData = cache_sigma_epsilon{Arguments.distance_energy_cutoff};
  CachedData.force_field_name = force_f;
  // ENE-22 (22/10/2026): MM/GBSA reads the topology too, for the exclusions
  // and listed pairs between its two groups (gbsa/gb_topology_pairs.h). On
  // that route a topology that does not bind every selected atom by name is
  // reported and left out, as before (ENE-12 tolerates a few mismatched
  // names there); the bonded-groups check then guards the run.
  if (Arguments.flag_use_common_topo && !Arguments.flag_nocal) {
    try {
      CachedData.nonbonded_topology = titan_nb::read_topology(Arguments.topo_name);
      titan_nb::validate_selection(*CachedData.nonbonded_topology, pro_in.info);
      titan_nb::validate_selection(*CachedData.nonbonded_topology, lig.info);
      TitanUI::log_kv(log, "Nonbonded topology",
                      "source-bound exclusions and listed pairs; " +
                      std::to_string(CachedData.nonbonded_topology->listed_pairs.size()) + " listed pair(s)");
      if (Arguments.flag_gbsa) {
        auto pairs = std::make_shared<titan_gbsa::TopologyPairs>(
            titan_gbsa::build_topology_pairs(*CachedData.nonbonded_topology,
                                             pro_in.info, lig.info));
        TitanUI::log_kv(log, "MM/GBSA Topology Pairs",
                        std::to_string(pairs->excluded) + " excluded, " +
                            std::to_string(pairs->listed) +
                            " listed between the two groups");
        if (!pairs->pairs.empty()) {
          TitanUI::print_info(
              "MM/GBSA: " + std::to_string(pairs->excluded) +
              " excluded and " + std::to_string(pairs->listed) +
              " listed pair(s) between the groups taken from the topology.");
          CachedData.gbsa_topology_pairs = std::move(pairs);
        }
      }
    } catch (const std::exception &error) {
      if (!Arguments.flag_gbsa) {
        TitanUI::print_error(error.what());
        TitanUI::log_message(log, error.what());
        return 1;
      }
      CachedData.nonbonded_topology.reset();
      CachedData.gbsa_topology_pairs.reset();
      const std::string note =
          std::string("MM/GBSA: topology exclusions not applied (") +
          error.what() + ").";
      TitanUI::print_warning(note);
      TitanUI::log_message(log, note);
    }
  }
  // Martini CG systems are too large for O(N×M) pair cache (~GB).
  // Energy functions handle empty cache via elec_ener.empty() checks.
  // Downstream emap/force_map are guarded to skip when cache is empty.
  pro_atom_count =
        !pro_in.m_mol.empty() ? pro_in.m_mol[0].size() : pro_in.info.size();
  lig_atom_count =
        !lig.m_mol.empty() ? lig.m_mol[0].size() : lig.info.size();

  if (pro_in.m_mol.empty() || lig.m_mol.empty()) {
    TitanUI::log_kv(log, "Frame0 Atom Counts",
                    "Fallback to topology info (mol1=" +
                        std::to_string(pro_atom_count) + ", mol2=" +
                        std::to_string(lig_atom_count) + ")");
  }

  enable_dense_pair_cache = should_enable_dense_pair_cache(
        Arguments, pro_atom_count, lig_atom_count, available_ram,
        pair_cache_reason);
  TitanUI::log_kv(log, "Dense Pair Cache", pair_cache_reason);
  if (enable_dense_pair_cache) {
    CachedData.set_size(pro_atom_count * lig_atom_count, Arguments,
                        pro_atom_count, lig_atom_count);
  }
  TitanUI::log_kv(log, "Pair Cache Entries", std::to_string(CachedData.elec_ener.size()));

  if (lig.m_mol.empty())
    lig.m_mol.resize(1);

  // --- MARTINI TOPOLOGY ASSIGNMENT (charges, LJ params, chain IDs) ---
  // Must run BEFORE total_charge() so that parameters[].charge is set.
  if (force_f == "MARTINI") {
    pro_in.parse_martini_topology(Arguments, log, &CachedData);
    lig.parse_martini_topology(Arguments, log, &CachedData);
  }

  // ENE-11 (21/10/2026): the pair-override table serves every force field
  // with a combination rule (AMBER and OPLS kernels read it through
  // titan_lj_pair_override.h); MARTINI keeps its own pair matrix.
  if (force_f.find("MARTINI") == string::npos &&
      !parameter.nonbond_overrides.empty()) {
    size_t ntypes = 0;
    for (const auto &[atom_type, type_id] : parameter.atom_type_to_id) {
      (void)atom_type;
      ntypes = std::max(ntypes, type_id + 1);
    }

    const auto safe_mul = [](size_t a, size_t b, size_t &out) {
      if (a == 0 || b == 0) {
        out = 0;
        return true;
      }
      if (a > std::numeric_limits<size_t>::max() / b)
        return false;
      out = a * b;
      return true;
    };

    size_t dense_entries = 0;
    bool dense_ok = safe_mul(ntypes, ntypes, dense_entries);
    const long double dense_bytes =
        dense_ok ? static_cast<long double>(dense_entries) *
                       static_cast<long double>(sizeof(double) * 2)
                 : std::numeric_limits<long double>::infinity();
    const long double dense_cap_bytes =
        available_ram > 0
            ? std::min(128.0L * 1024.0L * 1024.0L,
                       static_cast<long double>(available_ram) * 0.02L)
            : (128.0L * 1024.0L * 1024.0L);

    if (ntypes > 0 && dense_ok && dense_bytes <= dense_cap_bytes) {
      std::vector<double> sigma_table(dense_entries, -1.0);
      std::vector<double> epsilon_table(dense_entries, -1.0);
      size_t resolved = 0;
      for (const auto &rule : parameter.nonbond_overrides) {
        const auto it_i = parameter.atom_type_to_id.find(rule.atom_type_i);
        const auto it_j = parameter.atom_type_to_id.find(rule.atom_type_j);
        if (it_i == parameter.atom_type_to_id.end() ||
            it_j == parameter.atom_type_to_id.end())
          continue;

        const size_t idx_ij = it_i->second * ntypes + it_j->second;
        const size_t idx_ji = it_j->second * ntypes + it_i->second;
        if (idx_ij >= sigma_table.size() || idx_ji >= sigma_table.size())
          continue;
        sigma_table[idx_ij] = rule.sigma;
        sigma_table[idx_ji] = rule.sigma;
        epsilon_table[idx_ij] = rule.epsilon;
        epsilon_table[idx_ji] = rule.epsilon;
        ++resolved;
      }
      CachedData.set_charmm_pair_table(std::move(sigma_table),
                                       std::move(epsilon_table),
                                       static_cast<int>(ntypes));
      TitanUI::log_kv(
          log, "Pair LJ Overrides",
          std::to_string(parameter.nonbond_overrides.size()) +
              " pair(s) loaded from [ nonbond_params ], " +
              std::to_string(resolved) +
              " with both atom types defined (dense table, " +
              std::to_string(ntypes) + " type-id span)");
    } else {
      std::vector<std::tuple<std::uint64_t, double, double>> sparse_rules;
      sparse_rules.reserve(parameter.nonbond_overrides.size());
      for (const auto &rule : parameter.nonbond_overrides) {
        const auto it_i = parameter.atom_type_to_id.find(rule.atom_type_i);
        const auto it_j = parameter.atom_type_to_id.find(rule.atom_type_j);
        if (it_i == parameter.atom_type_to_id.end() ||
            it_j == parameter.atom_type_to_id.end())
          continue;

        const std::uint32_t lo =
            static_cast<std::uint32_t>(std::min(it_i->second, it_j->second));
        const std::uint32_t hi =
            static_cast<std::uint32_t>(std::max(it_i->second, it_j->second));
        const std::uint64_t key =
            (static_cast<std::uint64_t>(lo) << 32) |
            static_cast<std::uint64_t>(hi);
        sparse_rules.emplace_back(key, rule.sigma, rule.epsilon);
      }

      std::sort(sparse_rules.begin(), sparse_rules.end(),
                [](const auto &lhs, const auto &rhs) {
                  return std::get<0>(lhs) < std::get<0>(rhs);
                });

      std::vector<std::uint64_t> sparse_keys;
      std::vector<double> sparse_sigma;
      std::vector<double> sparse_epsilon;
      sparse_keys.reserve(sparse_rules.size());
      sparse_sigma.reserve(sparse_rules.size());
      sparse_epsilon.reserve(sparse_rules.size());
      for (const auto &[key, sigma, epsilon] : sparse_rules) {
        if (!sparse_keys.empty() && sparse_keys.back() == key) {
          sparse_sigma.back() = sigma;
          sparse_epsilon.back() = epsilon;
          continue;
        }
        sparse_keys.push_back(key);
        sparse_sigma.push_back(sigma);
        sparse_epsilon.push_back(epsilon);
      }

      const size_t sparse_keys_count = sparse_keys.size();
      CachedData.set_charmm_pair_sparse(std::move(sparse_keys),
                                        std::move(sparse_sigma),
                                        std::move(sparse_epsilon));
      TitanUI::log_kv(
          log, "Pair LJ Overrides",
          std::to_string(parameter.nonbond_overrides.size()) +
              " pair(s) loaded from [ nonbond_params ], " +
              std::to_string(sparse_keys_count) +
              " with both atom types defined (sparse table, " +
              std::to_string(ntypes) + " type-id span)");
    }
  }

  chg_pro = pro_in.total_charge();
  chg_lig = lig.total_charge();

  TitanUI::log_section(log, "SYSTEM TOPOLOGY & TRAJECTORY");
  return std::nullopt;
}

} // namespace engine_detail

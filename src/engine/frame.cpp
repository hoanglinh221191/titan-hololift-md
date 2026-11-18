#include "engine/engine_run.h"
#include "energy/group_bonds.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::process_frame(size_t chunk_snap_idx) {
  size_t global_snap_idx = global_frame_counter + chunk_snap_idx;

  const size_t energy_lg_snap =
      (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
  if (energy_lg_snap >= lig.m_mol.size()) {
    TitanUI::print_error(
        "Energy validation: ligand frame is unavailable.");
    return 1;
  }
  const MOL &energy_lig_frame = lig.m_mol[energy_lg_snap];
  const auto energy_validation_begin = std::chrono::steady_clock::now();
  if (Arguments.flag_gbsa && Arguments.self_int) {
    const std::string message =
        std::string(titan_energy_validation::
                        kIdenticalGbsaSelectionsError) +
        ": MM/GBSA binding endpoints require distinct receptor and "
        "ligand selections; the same group cannot be used twice.";
    TitanUI::print_error(message);
    TitanUI::log_message(log, "[Validation] " + message);
    return 1;
  }
  if (Arguments.self_int && !CachedData.nonbonded_topology) {
    const std::string message =
        "unsupported_topology_self_energy: intramolecular energy/force "
        "requires topology exclusions and explicit 1-4 parameters; the "
        "legacy distance-based exclusion heuristic is not a valid "
        "substitute. Self-interaction energy output is disabled until "
        "those parameters are available to every energy/decomposition kernel.";
    TitanUI::print_error(message);
    TitanUI::log_message(log, "[Validation] " + message);
    return 1;
  }
  // ENE-22 (22/10/2026): without topology exclusions (lib_ff and MM/GBSA
  // routes) two groups joined by a covalent bond give E_vdW ~1e5 kJ/mol.
  if (global_snap_idx == 0 && !Arguments.self_int &&
      !CachedData.nonbonded_topology &&
      Arguments.force_f.find("MARTINI") == string::npos) {
    const std::string bonded = titan_group_bonds::bonded_groups_error(
        pro_in.m_mol[chunk_snap_idx], pro_in.info, energy_lig_frame, lig.info,
        executor);
    if (!bonded.empty()) {
      TitanUI::print_error(bonded);
      TitanUI::log_message(log, "[Validation] " + bonded);
      return 1;
    }
  }
  const auto invalid_coordinate =
      titan_energy_validation::find_nonfinite_coordinate(
          pro_in.m_mol[chunk_snap_idx], energy_lig_frame);
  if (invalid_coordinate) {
    std::ostringstream message;
    message << titan_energy_validation::kNonfiniteCoordinateError
            << ": frame=" << (global_snap_idx + 1)
            << ", molecule=" << invalid_coordinate->molecule
            << ", atom=" << (invalid_coordinate->atom + 1)
            << ", axis=" << invalid_coordinate->axis
            << ", value=" << invalid_coordinate->value
            << "; energy modules require finite Cartesian coordinates.";
    TitanUI::print_error(message.str());
    TitanUI::log_message(log, "[Validation] " + message.str());
    return 1;
  }
  const double minimum_pair_distance_nm =
      Arguments.flag_gbsa
          ? titan_energy_validation::kMinimumGbsaPairDistanceNm
          : titan_energy_validation::kMinimumPairDistanceNm;
  const auto overlap = titan_energy_validation::find_overlapping_atoms(
      pro_in.m_mol[chunk_snap_idx], energy_lig_frame,
      Arguments.self_int, executor, minimum_pair_distance_nm);
  const auto energy_validation_end = std::chrono::steady_clock::now();
  perf_observer.add_frame_elapsed(
      "EnergyValidation",
      static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
      pro_in.m_mol[chunk_snap_idx].time_of_snap,
      std::chrono::duration<double, std::milli>(energy_validation_end -
                                                energy_validation_begin)
          .count(),
      "coordinate_overlap_gate", !overlap.has_value());
  if (overlap) {
    std::ostringstream message;
    message << (Arguments.flag_gbsa
                    ? titan_energy_validation::kCloseContactError
                    : titan_energy_validation::kOverlappingAtomsError)
            << ": frame=" << (global_snap_idx + 1)
            << ", mol1_atom=" << (overlap->mol1_atom + 1)
            << ", mol2_atom=" << (overlap->mol2_atom + 1)
            << ", distance_nm=" << std::scientific
            << overlap->distance_nm
            << "; "
            << (Arguments.flag_gbsa
                    ? "MM/GBSA rejects cross-group distances below 0.05 "
                      "nm instead of suppressing their energy."
                    : "electrostatic/vdW energy requires distinct atom "
                      "coordinates.");
    TitanUI::print_error(message.str());
    TitanUI::log_message(log, "[Validation] " + message.str());
    return 1;
  }

  // --- LLPS per-frame analysis (async pipeline — overlaps energy calc)
  // ---
  if (Arguments.llps.flag_llps && llps_analyzer) {
    auto start_llps = std::chrono::steady_clock::now();
    const auto &mol_frame = pro_in.m_mol[chunk_snap_idx];
    // Box dimensions from CRYST1 (Å → nm)
    const float bx_nm = static_cast<float>(mol_frame.Box.get_a() / 10.0);
    const float by_nm = static_cast<float>(mol_frame.Box.get_b() / 10.0);
    const float bz_nm = static_cast<float>(mol_frame.Box.get_c() / 10.0);
    const float t_ns =
        static_cast<float>(mol_frame.time_of_snap / 1000.0);
    // Pass RNA (lig) data if mol_2_type is rna
    const MOL *rna_ptr = nullptr;
    const std::vector<mol_info> *rna_info_ptr = nullptr;
    if (Arguments.mol_2_type.find("rna") != std::string::npos ||
        Arguments.mol_2_type.find("RNA") != std::string::npos) {
      if (chunk_snap_idx < lig.m_mol.size()) {
        rna_ptr = &lig.m_mol[chunk_snap_idx];
        rna_info_ptr = &lig.info;
      }
    }
    // Submit to async queue — returns immediately, runs on worker thread
    llps_analyzer->process_frame_async(
        mol_frame, pro_in.info, bx_nm, by_nm, bz_nm, t_ns,
        static_cast<int>(global_snap_idx), rna_ptr, rna_info_ptr);
    time_llps += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_llps);
  }

  // --- LLPS DBSCAN per-frame ---
  if (dbscan_llps) {
    auto t0_dbscan = std::chrono::steady_clock::now();
    const MOL *rna_mol = nullptr;
    const std::vector<mol_info> *rna_info = nullptr;
    if (chunk_snap_idx < lig.m_mol.size()) {
      rna_mol = &lig.m_mol[chunk_snap_idx];
      rna_info = &lig.info;
    }
    static const MOL empty_mol;
    static const std::vector<mol_info> empty_info;
    // [BUG 2 note] pro.Box is used as the shared periodic box for all
    // nodes (protein + RNA/lig). This is correct when both molecules come
    // from the same trajectory frame. If lig were ever loaded from a
    // separate file with a different box, MIC for RNA nodes would be
    // wrong — document that here.
    dbscan_llps->process_frame(executor, pro_in.m_mol[chunk_snap_idx],
                               pro_in.info, rna_mol ? *rna_mol : empty_mol,
                               rna_info ? *rna_info : empty_info,
                               pro_in.m_mol[chunk_snap_idx].time_of_snap,
                               static_cast<int>(global_snap_idx + 1));
    time_llps_dbscan +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0_dbscan);
  }

  const int progress_traj_id = input_name - Arguments.b_input + 1;
  std::string status_text =
      "Traj " + std::to_string(progress_traj_id) + " | Frame " +
      std::to_string(global_snap_idx + 1) + " | Global " +
      std::to_string((tot_snap_allfiles + global_snap_idx + 1));

  const std::string module_name = progress_source_label(Arguments, name_pro);

  if (Arguments.flag_xdr) {
    TitanUI::show_continuous_progress(module_name, global_snap_idx + 1,
                                      status_text, streaming_modules);
  } else {
    TitanUI::show_progress_exact(module_name, chunk_snap_idx + 1,
                                 std::max<size_t>(1, chunk_num_snaps),
                                 status_text, streaming_modules);
  }

  if (pbe_analyzer.enabled()) {
    auto start_pbe = std::chrono::steady_clock::now();
    const size_t pbe_lg_snap = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
    const MOL *pbe_lig_frame =
        (!lig.m_mol.empty() && pbe_lg_snap < lig.m_mol.size())
            ? &lig.m_mol[pbe_lg_snap]
            : nullptr;
    const Multi_MOL_parameters *pbe_lig_params =
        pbe_lig_frame ? &lig.parameters : nullptr;
    pbe_analyzer.consume_frame(
        pro_in.m_mol[chunk_snap_idx], pro_in.parameters, pbe_lig_frame,
        pbe_lig_params, static_cast<std::size_t>(progress_traj_id),
        global_snap_idx, tot_snap_allfiles + global_snap_idx, executor,
        log, Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
    const auto end_pbe = std::chrono::steady_clock::now();
    time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_pbe - start_pbe);
    perf_observer.add_frame_elapsed(
        Arguments.pbe.pbsa ? "PBSA" : "PBE",
        static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_pbe - start_pbe)
            .count());
  }

  vector<d_min> dis_min(num_threads);
  vector<cache_energy> Energy(num_threads);
  vector<cache_force> Force(num_threads);

  double E_V_{}, E_E_{};
  force f_e_{}, f_v_{};

  // Print formatted timestamps in a neat grid (10 columns)
  if (log.is_open()) {
    log << std::fixed << std::setprecision(1) << std::setw(10)
        << std::right << pro_in.m_mol[chunk_snap_idx].time_of_snap;
    if ((chunk_snap_idx + 1) % 10 == 0 || chunk_snap_idx + 1 == NUM) {
      log << "\n";
    }
  }

  AlignedDoubleVector rec_free_radii;
  AlignedDoubleVector lg_free_radii;
  AlignedDoubleVector rec_complex_radii;
  AlignedDoubleVector lg_complex_radii;
  const MOL &active_lg_mol = energy_lig_frame;
  const vector<mol_info> &active_lg_info = lig.info;
  const size_t ligand_atom_count = lig.m_mol[0].size();

  auto start_energy_eval = std::chrono::steady_clock::now();
  GBSA_energy_snap frame_gbsa_total;
  if (Arguments.flag_gbsa) {
    const bool capture_gbsa_timing = Arguments.gbsa_perf != "off";
    const auto gbsa_precompute_begin =
        capture_gbsa_timing ? std::chrono::steady_clock::now()
                            : start_energy_eval;
    GBSASasaCache frame_sasa_cache;
#ifdef __AVX__
    rec_free_radii.resize(pro_in.m_mol[chunk_snap_idx].size());
    lg_free_radii.resize(active_lg_mol.size());
    rec_complex_radii.resize(pro_in.m_mol[chunk_snap_idx].size());
    lg_complex_radii.resize(active_lg_mol.size());

    // All four endpoints must share one rgbmax, otherwise the binding
    // cycle mixes cut and uncut descreening integrals.
    if (Arguments.igb == 66) {
      calculate_gbnsr6_binding_radii(pro_in.m_mol[chunk_snap_idx],active_lg_mol,
          rec_vdw_radii,lg_vdw_radii,pro_in.info,active_lg_info,
          rec_free_radii,lg_free_radii,rec_complex_radii,lg_complex_radii,
          Arguments.gbnsr6_spacing,&executor);
    } else {
    cal_binding_born_radii_parallel_avx2(
        pro_in.m_mol[chunk_snap_idx], rec_vdw_radii, pro_in.info,
        active_lg_mol, lg_vdw_radii, active_lg_info, Arguments.igb,
        executor, Arguments.gbsa_rgbmax, rec_free_radii, lg_free_radii,
        rec_complex_radii, lg_complex_radii);
    }

    const auto validate_born_radii = [&](const AlignedDoubleVector &values,
                                          const char *endpoint) {
      for (size_t atom = 0; atom < values.size(); ++atom) {
        if (!std::isfinite(values[atom]) || values[atom] <= 0.0) {
          std::ostringstream message;
          message << "invalid_gb_born_radius: frame="
                  << (global_snap_idx + 1) << ", endpoint=" << endpoint
                  << ", atom=" << (atom + 1)
                  << ", radius_nm=" << values[atom];
          TitanUI::print_error(message.str());
          TitanUI::log_message(log, "[Validation] " + message.str());
          return false;
        }
      }
      return true;
    };
    if (!validate_born_radii(rec_free_radii, "receptor_free") ||
        !validate_born_radii(lg_free_radii, "ligand_free") ||
        !validate_born_radii(rec_complex_radii, "receptor_complex") ||
        !validate_born_radii(lg_complex_radii, "ligand_complex")) {
      return 1;
    }

    build_gbsa_sasa_cache(pro_in.m_mol[chunk_snap_idx], active_lg_mol,
                          pro_in.parameters, lig.parameters,
                          frame_sasa_cache,
                          Arguments.gbsa_sasa_points,
                          Arguments.flag_alpb, &executor);
    if (Arguments.igb == 66)
      set_gbnsr6_electrostatic_sizes(pro_in.m_mol[chunk_snap_idx],active_lg_mol,
          pro_in.parameters,lig.parameters,pro_in.info,active_lg_info,frame_sasa_cache);
    if (Arguments.flag_alpb &&
        (!(frame_sasa_cache.alpb_rec_electrostatic_size_nm > 0.0) ||
         !std::isfinite(
             frame_sasa_cache.alpb_rec_electrostatic_size_nm) ||
         !(frame_sasa_cache.alpb_lig_electrostatic_size_nm > 0.0) ||
         !std::isfinite(
             frame_sasa_cache.alpb_lig_electrostatic_size_nm) ||
         !(frame_sasa_cache.alpb_complex_electrostatic_size_nm > 0.0) ||
         !std::isfinite(
             frame_sasa_cache.alpb_complex_electrostatic_size_nm))) {
      std::ostringstream message;
      message << "invalid_alpb_electrostatic_size: frame="
              << (global_snap_idx + 1) << ", receptor_nm="
              << frame_sasa_cache.alpb_rec_electrostatic_size_nm
              << ", ligand_nm="
              << frame_sasa_cache.alpb_lig_electrostatic_size_nm
              << ", complex_nm="
              << frame_sasa_cache.alpb_complex_electrostatic_size_nm;
      TitanUI::print_error(message.str());
      TitanUI::log_message(log, "[Validation] " + message.str());
      return 1;
    }
#else
    std::cerr << "Error: AVX2 is required for GBSA!" << std::endl;
    exit(1);
#endif

    size_t active_lg_size = active_lg_mol.size();
    size_t rec_size = pro_in.m_mol[chunk_snap_idx].size();
    // ENE-15: the cross pairs and the larger molecule's one-body terms
    // (self, nonpolar, intramolecular Delta-GB rows) run in tiles of the
    // larger molecule; the smaller molecule's one-body terms get tiles of
    // their own instead of all landing on task 0.
    constexpr size_t gbsa_tile_atoms = 64;
    const bool tile_receptor = rec_size >= active_lg_size;
    const size_t tiled_atoms = tile_receptor ? rec_size : active_lg_size;
    const size_t other_atoms = tile_receptor ? active_lg_size : rec_size;
    const size_t cross_task_count =
        (tiled_atoms + gbsa_tile_atoms - 1) / gbsa_tile_atoms;
    const size_t onebody_task_count =
        (other_atoms + gbsa_tile_atoms - 1) / gbsa_tile_atoms;
    const size_t gbsa_task_count =
        std::max<size_t>(1, cross_task_count + onebody_task_count);
    std::vector<GBSA_Aggregator> local_aggs(
        gbsa_task_count, GBSA_Aggregator(Arguments, info_output));
    std::vector<GBSA_energy_snap> gbsa_energies(gbsa_task_count);
    const auto gbsa_kernel_begin =
        capture_gbsa_timing ? std::chrono::steady_clock::now()
                            : gbsa_precompute_begin;
    executor.parallel_for(
        gbsa_task_count, num_threads, 1,
        [&](size_t, size_t task_begin, size_t task_end) {
#ifdef __AVX__
          for (size_t task = task_begin; task < task_end; ++task) {
            size_t rec_begin = 0, rec_end = 0, lig_begin = 0, lig_end = 0;
            bool rec_onebody = false, lig_onebody = false;
            if (task < cross_task_count) {
              // cross pairs of a tile of the larger molecule with all of the
              // smaller one, plus the tile's own one-body terms
              const size_t tile_begin = task * gbsa_tile_atoms;
              const size_t tile_end =
                  std::min(tiled_atoms, tile_begin + gbsa_tile_atoms);
              rec_begin = tile_receptor ? tile_begin : 0;
              rec_end = tile_receptor ? tile_end : rec_size;
              lig_begin = tile_receptor ? 0 : tile_begin;
              lig_end = tile_receptor ? active_lg_size : tile_end;
              rec_onebody = tile_receptor;
              lig_onebody = !tile_receptor;
            } else {
              // one-body terms of a tile of the smaller molecule; the empty
              // range of the other molecule leaves no cross pair here
              const size_t tile_begin =
                  (task - cross_task_count) * gbsa_tile_atoms;
              const size_t tile_end =
                  std::min(other_atoms, tile_begin + gbsa_tile_atoms);
              rec_begin = tile_receptor ? 0 : tile_begin;
              rec_end = tile_receptor ? 0 : tile_end;
              lig_begin = tile_receptor ? tile_begin : 0;
              lig_end = tile_receptor ? tile_end : 0;
              rec_onebody = !tile_receptor;
              lig_onebody = tile_receptor;
            }
            cal_energy_gbsa_avx2(
                pro_in.m_mol[chunk_snap_idx], active_lg_mol,
                pro_in.info, active_lg_info, CachedData, rec_begin,
                rec_end, lig_begin, lig_end, gbsa_energies[task],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters, pro_in.lcpo_p1, pro_in.lcpo_p2,
                lig.lcpo_p1, lig.lcpo_p2, rec_free_radii,
                lg_free_radii, rec_complex_radii, lg_complex_radii,
                &local_aggs[task], &frame_sasa_cache, true, rec_onebody,
                lig_onebody, CachedData.gbsa_topology_pairs.get());
          }
#endif
        });

    const auto gbsa_merge_begin =
        capture_gbsa_timing ? std::chrono::steady_clock::now()
                            : gbsa_kernel_begin;

    // Merge thread-local GBSA results back to the main pipeline
    frame_gbsa_total.time = pro_in.m_mol[chunk_snap_idx].time_of_snap;

    // double current_E_vdw = 0.0;
    // double current_E_elec = 0.0;

    // Per-residue decomposition of this frame, merged from the task-local
    // aggregators (ENE-17(e): the former comment was mis-encoded text).
    GBSA_Aggregator frame_decomp(Arguments, info_output);

    for (size_t i{}; i < gbsa_task_count; ++i) {
      frame_gbsa_total.E_vdw += gbsa_energies[i].E_vdw;
      frame_gbsa_total.E_elec += gbsa_energies[i].E_elec;
      frame_gbsa_total.G_polar += gbsa_energies[i].G_polar;
      frame_gbsa_total.G_nonpolar += gbsa_energies[i].G_nonpolar;
      frame_gbsa_total.total_energy += gbsa_energies[i].total_energy;

      frame_decomp.merge(
          local_aggs
              [i]);
                    // Thread -> Frame
    }
    if (!Energy.empty()) {
      Energy[0].vdw_ener = frame_gbsa_total.E_vdw;
      Energy[0].elec_ener = frame_gbsa_total.E_elec;
    }

    const auto finite_gbsa_frame = [](const GBSA_energy_snap &value) {
      return std::isfinite(value.E_vdw) &&
             std::isfinite(value.E_elec) &&
             std::isfinite(value.G_polar) &&
             std::isfinite(value.G_nonpolar) &&
             std::isfinite(value.total_energy);
    };
    if (!finite_gbsa_frame(frame_gbsa_total)) {
      TitanUI::print_error(
          "invalid_gbsa_energy: a non-finite MM/GBSA component was "
          "produced; the frame was not added to statistics.");
      return 1;
    }

    GBSA_energy_snap decomposition_sum;
    for (const auto &entry : frame_decomp.get_decomp()) {
      decomposition_sum.E_vdw += entry.E_vdw;
      decomposition_sum.E_elec += entry.E_elec;
      decomposition_sum.G_polar += entry.G_polar;
      decomposition_sum.G_nonpolar += entry.G_nonpolar;
      decomposition_sum.total_energy += entry.total;
    }
    const auto closes = [](double observed, double reference) {
      return std::abs(observed - reference) <=
             1.0e-8 * std::max(1.0, std::abs(reference));
    };
    if (!closes(decomposition_sum.E_vdw, frame_gbsa_total.E_vdw) ||
        !closes(decomposition_sum.E_elec, frame_gbsa_total.E_elec) ||
        !closes(decomposition_sum.G_polar,
                frame_gbsa_total.G_polar) ||
        !closes(decomposition_sum.G_nonpolar,
                frame_gbsa_total.G_nonpolar) ||
        !closes(decomposition_sum.total_energy,
                frame_gbsa_total.total_energy)) {
      TitanUI::print_error(
          "invalid_gbsa_decomposition_closure: receptor plus ligand "
          "attribution does not reproduce the frame energy.");
      return 1;
    }

    if (edecomp_gbsa_requested(Arguments) && edecomp_gbsa_engine) {
      try {
        edecomp_gbsa_engine->process_frame(
            make_gbsa_endpoint_rows(frame_decomp,
                                    *edecomp_gbsa_engine));
      } catch (const std::exception &e) {
        TitanUI::print_error(std::string("Energy decomposition GBSA: ") +
                             e.what());
        return 1;
      }
    }

    global_gbsa_agg.commit_frame_decompositions(
        frame_decomp, tot_snap_allfiles + global_snap_idx);
    if (capture_gbsa_timing) {
      const auto gbsa_frame_end = std::chrono::steady_clock::now();
      frame_gbsa_total.time_precompute_ms =
          std::chrono::duration<double, std::milli>(
              gbsa_kernel_begin - gbsa_precompute_begin)
              .count();
      frame_gbsa_total.time_energy_kernel_ms =
          std::chrono::duration<double, std::milli>(
              gbsa_merge_begin - gbsa_kernel_begin)
              .count();
      frame_gbsa_total.time_decomposition_merge_ms =
          std::chrono::duration<double, std::milli>(
              gbsa_frame_end - gbsa_merge_begin)
              .count();
      frame_gbsa_total.time_total_ms =
          std::chrono::duration<double, std::milli>(
              gbsa_frame_end - gbsa_precompute_begin)
              .count();
    }
    if (Arguments.flag_alpb) {
      frame_gbsa_total.alpb_receptor_size_nm =
          frame_sasa_cache.alpb_rec_electrostatic_size_nm;
      frame_gbsa_total.alpb_ligand_size_nm =
          frame_sasa_cache.alpb_lig_electrostatic_size_nm;
      frame_gbsa_total.alpb_complex_size_nm =
          frame_sasa_cache.alpb_complex_electrostatic_size_nm;
    }
    global_gbsa_agg.add_snapshot(input_name - Arguments.b_input,
                                 frame_gbsa_total);

    const auto end_gbsa = std::chrono::steady_clock::now();
    time_gbsa += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_gbsa - start_energy_eval);
    perf_observer.add_frame_elapsed(
        "GBSA",
        static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_gbsa -
                                                  start_energy_eval)
            .count());
  } else {
    // Liquidity v3 (performance only): TITAN_LLPS_SKIP_PAIR_ENERGY=1
    // skips the group-1/group-2 pair energy (6-7 s per frame for the
    // 23k-bead N+RNA slab). The LLPS liquidity block never reads it;
    // electrostatic/vdW/total_energy outputs are then zero.
    static const bool skip_pair_energy = [] {
      const char *v = std::getenv("TITAN_LLPS_SKIP_PAIR_ENERGY");
      const bool on = (v != nullptr && v[0] == '1');
      if (on)
        std::cerr << "[TITAN] TITAN_LLPS_SKIP_PAIR_ENERGY=1: pair "
                     "energy evaluation skipped; energy outputs are "
                     "zero.\n";
      return on;
    }();
    auto dispatch_energy_workers = [&](auto &&worker_fn) {
      if (skip_pair_energy)
        return;
      executor.parallel_do(num_threads, [&](size_t worker_id) {
        worker_fn(worker_id);
      });
    };
    const bool use_unfiltered_energy_intrin =
        Arguments.self_int == false &&
        Arguments.distance_energy_cutoff < 0.;

    if (Arguments.force_f.find("CHARMM") != string::npos) {
      dispatch_energy_workers([&](size_t i) {
        if (Arguments.flag_elec_screening == false) {
          if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
            cal_energy_amber_intrin(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#else
            cal_energy_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#endif
          } else {
            cal_energy_force_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          }
        } else {
          if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
            cal_energy_amber_screening_intrin(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#else
            cal_energy_amber_screening(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#endif
          } else {
            cal_energy_force_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          }
        }
      });

    } else if (Arguments.force_f.find("AMBER") != string::npos) {
      dispatch_energy_workers([&](size_t i) {
        if (Arguments.flag_elec_screening == false) {
          if (Arguments.flag_cal_force == false) {
#ifdef __AVX__
            if (use_unfiltered_energy_intrin) {
              cal_energy_amber_intrin_nofilter(
                  pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                  pro_in.info, lig.info, CachedData,
                  ligand_atom_count * i / num_threads,
                  ligand_atom_count * (i + 1) / num_threads, Energy[i],
                  chunk_snap_idx, Arguments, pro_in.parameters,
                  lig.parameters);
            } else {
              cal_energy_amber_intrin(
                  pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                  pro_in.info, lig.info, CachedData,
                  ligand_atom_count * i / num_threads,
                  ligand_atom_count * (i + 1) / num_threads, Energy[i],
                  chunk_snap_idx, Arguments, pro_in.parameters,
                  lig.parameters);
            }
#else
            cal_energy_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#endif
          } else
            cal_energy_force_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
        } else {
          if (Arguments.flag_cal_force == false)

#ifdef __AVX__
            if (use_unfiltered_energy_intrin) {
              cal_energy_amber_screening_intrin_nofilter(
                  pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                  pro_in.info, lig.info, CachedData,
                  ligand_atom_count * i / num_threads,
                  ligand_atom_count * (i + 1) / num_threads, Energy[i],
                  chunk_snap_idx, Arguments, pro_in.parameters,
                  lig.parameters);
            } else {
              cal_energy_amber_screening_intrin(
                  pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                  pro_in.info, lig.info, CachedData,
                  ligand_atom_count * i / num_threads,
                  ligand_atom_count * (i + 1) / num_threads, Energy[i],
                  chunk_snap_idx, Arguments, pro_in.parameters,
                  lig.parameters);
            }
#else
            cal_energy_amber_screening(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
#endif
          else
            cal_energy_force_amber(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
        }
      });

    } else if (Arguments.force_f.find("OPLSAA") != string::npos) {
      dispatch_energy_workers([&](size_t i) {
        if (Arguments.flag_cal_force == false)
#ifdef __AVX__
          if (use_unfiltered_energy_intrin) {
            cal_energy_opls_intrin_nofilter(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          } else {
            cal_energy_opls_intrin(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          }
#else
          cal_energy_opls(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
#endif
        else
          cal_energy_force_opls(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
      });
    } else if (Arguments.force_f.find("MARTINI") != string::npos) {
      dispatch_energy_workers([&](size_t i) {
        if (Arguments.flag_cal_force == false)
#ifdef __AVX__
          cal_energy_martini_intrin(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
#else
          cal_energy_martini(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
#endif
        else
          cal_energy_force_martini(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
      });
    } else {
      dispatch_energy_workers([&](size_t i) {
        if (Arguments.flag_cal_force == false)
#ifdef __AVX__
          if (use_unfiltered_energy_intrin) {
            cal_energy_gromos_intrin_nofilter(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          } else {
            cal_energy_gromos_intrin(
                pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                pro_in.info, lig.info, CachedData,
                ligand_atom_count * i / num_threads,
                ligand_atom_count * (i + 1) / num_threads, Energy[i],
                chunk_snap_idx, Arguments, pro_in.parameters,
                lig.parameters);
          }
#else
          cal_energy_gromos(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
#endif
        else
          cal_energy_force_gromos(
              pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
              pro_in.info, lig.info, CachedData,
              ligand_atom_count * i / num_threads,
              ligand_atom_count * (i + 1) / num_threads, Energy[i],
              Force[i], chunk_snap_idx, Arguments, pro_in.parameters,
              lig.parameters);
      });
    }
    time_cal += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_energy_eval);
    // cout << "done cal snap " << chunk_snap_idx << " ";
  } // Close else block for standard energy eval

  if (Arguments.self_int == true) {
    remove_double_counting(CachedData, Energy);
  }

  for (size_t i{}; i < Energy.size(); ++i) {
    E_E_ += Energy[i].elec_ener;
    E_V_ += Energy[i].vdw_ener;
    if (Arguments.flag_cal_force == true) {
      f_e_ += Force[i].forces[0];
      f_v_ += Force[i].forces[1];
    }
  }

  process_adding_energies_forces(
      Arguments, input_name, E_v, E_e, E_E_, E_V_, f_e_, f_v_, E_v_vector,
      E_e_vector, Force_e, Force_v, Force_e_vector, Force_v_vector,
      size_snap, global_snap_idx);

  // One time per energy row, from the frame that created the row: the
  // frames of the first input (later inputs add into the same rows), or
  // every frame with -cat-result.
  while (time_of_snap.size() < E_e.size())
    time_of_snap.push_back(
        Arguments.output_time(pro_in.m_mol[chunk_snap_idx].time_of_snap));

  if (edecomp_mm_requested(Arguments) && edecomp_engine) {
    auto start_edecomp = std::chrono::steady_clock::now();
    try {
      edecomp_engine->process_frame(
          {&CachedData, &edecomp_energy_provider,
           &pro_in.m_mol[chunk_snap_idx],
           &lig.m_mol[chunk_snap_idx]},
          executor);
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Energy decomposition: ") +
                           e.what());
      return 1;
    }
    const auto end_edecomp = std::chrono::steady_clock::now();
    time_edecomp += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_edecomp - start_edecomp);
    perf_observer.add_frame_elapsed(
        "EDecomp",
        static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_edecomp -
                                                  start_edecomp)
            .count());
  }

  if (need_distance_scan) {
    auto start_sc{chrono::steady_clock::now()};

    if (pro_in.Index.size() > 1) {
      executor.parallel_for(
          pro_in.Index.size() - 1, num_threads, 1,
          [&](size_t /*worker_id*/, size_t begin, size_t end) {
            for (size_t residue_idx = begin; residue_idx < end;
                 ++residue_idx) {
              pro_in.m_mol[chunk_snap_idx].take_com(
                  pro_in.parameters, residue_idx,
                  pro_in.Index[residue_idx + 1].get_res_pos_e(0),
                  pro_in.Index, pro_in.info);
            }
          });
    }

    if (lig.Index.size() > 1) {
      executor.parallel_for(
          lig.Index.size() - 1, num_threads, 1,
          [&](size_t /*worker_id*/, size_t begin, size_t end) {
            for (size_t residue_idx = begin; residue_idx < end;
                 ++residue_idx) {
              lig.m_mol[chunk_snap_idx].take_com(
                  lig.parameters, residue_idx,
                  lig.Index[residue_idx + 1].get_res_pos_e(0), lig.Index,
                  lig.info);
            }
          });
    }

    impl_add_coms_2_mols(lig, pro_in, Arguments, chunk_snap_idx);
    /***************************************************************************************************************/

    executor.parallel_do(num_threads, [&](size_t i) {
      take_min(pro_in.m_mol[chunk_snap_idx].Com,
               lig.m_mol[chunk_snap_idx].Com, dis_min[i],
               i * pro_in.m_mol[chunk_snap_idx].Com.size() / num_threads,
               (i + 1) * pro_in.m_mol[chunk_snap_idx].Com.size() /
                   num_threads,
               0, lig.m_mol[chunk_snap_idx].Com.size(), Arguments, 0,
               d_standard);
    });

    time_sc += chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - start_sc);
    // cout << "\nsnap " << chunk_snap_idx << " mol1 consists of " <<
    // pro_in.m_mol[chunk_snap_idx].Com.size() << " mol2 consists of "
    // << lig.m_mol[chunk_snap_idx].Com.size() << " residues\n"; cout
    // << "\nsnap:
    // "<<chunk_snap_idx<< ": numof dismin " << dis_min.size() << " ";
    // for (auto& d : dis_min) 	cout << d.base.size() << ' '; cout <<
    // "mem size: " << Arguments.nt * (sizeof(std::vector<dmin_base>) +
    // (sizeof(dmin_base) * dis_min[0].base.size())); cout << '\n';
  }

  // --- IFP per-frame processing ---
  if (Arguments.ifp.flag_ifp && ifp_analyzer &&
      include_ifp_frame(
          pro_in, chunk_snap_idx,
          static_cast<size_t>(input_name - Arguments.b_input + 1),
          global_snap_idx + 1,
          tot_snap_allfiles + global_snap_idx + 1)) {
    auto start_ifp = std::chrono::steady_clock::now();
    size_t lg_snap_ifp = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
    TitanAwhIfp::AwhFrameContext awh_frame_ctx;
    if (Arguments.awh.flag_awh_ifp && awh_ifp_analyzer) {
      const int current_replica = input_name - Arguments.b_input;
      awh_frame_ctx = awh_ifp_analyzer->prepare_frame(
          current_replica, pro_in.m_mol[chunk_snap_idx].time_of_snap,
          static_cast<int64_t>(tot_snap_allfiles + global_snap_idx));
    }
    ifp_analyzer->process_frame(
        pro_in.m_mol[chunk_snap_idx],
        lig.m_mol[lg_snap_ifp],
        pro_in.info, lig.info,
        pro_in.Index, lig.Index,
        pro_in.info.size(),
        lig.info.size(),
        dis_min,
        Arguments.ifp,
        executor,
        num_threads,
        awh_frame_ctx.matched
    );
    if (awh_frame_ctx.matched && awh_ifp_analyzer) {
      awh_ifp_analyzer->consume_frame(
          awh_frame_ctx,
          ifp_analyzer->last_frame_contact_accumulators(),
          ifp_analyzer->last_frame_active_workers(),
          pro_in.m_mol[chunk_snap_idx], lig.m_mol[lg_snap_ifp]);
    }
    time_ifp += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_ifp);
  }

  if (Arguments.cal_dis_sc_contact == true) {
    auto start_cscd = std::chrono::steady_clock::now();
    if (impl_cscd(pro_in, lig, Arguments, executor, num_threads,
                  chunk_snap_idx, global_snap_idx, sc_flag,
                  cscd_out_dis) != 0)
      return 1;
    time_cscd += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_cscd);
  }

  if (global_has_nucleic_acid) {
    auto start_na = std::chrono::steady_clock::now();
    na_profiler.process_frame(
        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
        pro_in.info, lig.info, pro_in.Index, lig.Index, chunk_snap_idx,
        pro_in.m_mol[chunk_snap_idx].time_of_snap, executor,
        num_threads);
    time_na += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_na);
  }

  if (dssp_mol1_analyzer.enabled() ||
      dssp_mol2_analyzer.enabled()) {
    auto start_dssp = std::chrono::steady_clock::now();
      if (dssp_mol1_analyzer.enabled())
        dssp_mol1_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
      if (dssp_mol2_analyzer.enabled())
        dssp_mol2_analyzer.process_frame(lig, chunk_snap_idx, executor);
    time_dssp += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_dssp);
  }

  if (lj_analyzer.enabled()) {
    auto start_lj = std::chrono::steady_clock::now();
      lj_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
    time_lj += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_lj);
  }

  if (sq_analyzer.enabled()) {
    auto start_sq = std::chrono::steady_clock::now();
    sq_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
    time_sq += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_sq);
  }

  if (cmap_analyzer.enabled()) {
    auto start_cmap = std::chrono::steady_clock::now();
    const Multi_MOL *cmap_mol2 =
        Arguments.needs_mol2_for_cmap_interface() ? &lig : nullptr;
    cmap_analyzer.process_frame(pro_in, cmap_mol2, chunk_snap_idx,
                                executor);
    const auto end_cmap = std::chrono::steady_clock::now();
    time_cmap += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_cmap - start_cmap);
    perf_observer.add_frame_elapsed(
        "CMap", static_cast<std::uint64_t>(tot_snap_allfiles +
                                           global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_cmap - start_cmap)
            .count());
  }

  if (interface_analyzer.enabled()) {
    const auto start_interface = std::chrono::steady_clock::now();
    const Multi_MOL *interface_mol2 =
        Arguments.needs_mol2_for_interface_analysis() ? &lig : nullptr;
    try {
      interface_analyzer.process_frame(
          pro_in, interface_mol2, chunk_snap_idx, global_snap_idx,
          executor);
    } catch (const std::exception &e) {
      TitanUI::print_error(std::string("Interface analysis: ") +
                           e.what());
      return 1;
    }
    const auto end_interface = std::chrono::steady_clock::now();
    time_interface +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_interface - start_interface);
    perf_observer.add_frame_elapsed(
        "Interface",
        static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_interface -
                                                  start_interface)
            .count());
  }

  if (dihedral_analyzer.enabled()) {
    auto start_dihedral = std::chrono::steady_clock::now();
    dihedral_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
    const auto end_dihedral = std::chrono::steady_clock::now();
    time_dihedral += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_dihedral - start_dihedral);
    perf_observer.add_frame_elapsed(
        "Dihedral",
        static_cast<std::uint64_t>(tot_snap_allfiles + global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_dihedral -
                                                  start_dihedral)
            .count());
  }

  if (qnative_analyzer.enabled()) {
    auto start_qnative = std::chrono::steady_clock::now();
    const Multi_MOL *qnative_mol2 =
        Arguments.needs_mol2_for_qnative_interface() ? &lig : nullptr;
    qnative_analyzer.process_frame(pro_in, qnative_mol2, chunk_snap_idx,
                                   executor);
    const auto end_qnative = std::chrono::steady_clock::now();
    time_qnative +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_qnative - start_qnative);
    perf_observer.add_frame_elapsed(
        "QNative", static_cast<std::uint64_t>(tot_snap_allfiles +
                                              global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_qnative -
                                                  start_qnative)
            .count());
  }

  if (msd_analyzer.enabled()) {
    auto start_msd = std::chrono::steady_clock::now();
    const Multi_MOL *msd_mol2 =
        Arguments.needs_mol2_for_msd() ? &lig : nullptr;
    msd_analyzer.process_frame(pro_in, msd_mol2, chunk_snap_idx,
                               executor);
    const auto end_msd = std::chrono::steady_clock::now();
    time_msd += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_msd - start_msd);
    perf_observer.add_frame_elapsed(
        "MSD", static_cast<std::uint64_t>(tot_snap_allfiles +
                                          global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_msd - start_msd)
            .count());
  }

  if (membrane_analyzer.enabled() &&
      chunk_snap_idx < membrane_traj.m_mol.size()) {
    auto start_membrane = std::chrono::steady_clock::now();
    membrane_analyzer.process_frame(membrane_traj, &pro_in,
                                    chunk_snap_idx, executor);
    const auto end_membrane = std::chrono::steady_clock::now();
    time_membrane +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_membrane - start_membrane);
    perf_observer.add_frame_elapsed(
        "Membrane", static_cast<std::uint64_t>(tot_snap_allfiles +
                                               global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_membrane -
                                                  start_membrane)
            .count());
  }

  if (s2_analyzer.enabled()) {
    auto start_s2 = std::chrono::steady_clock::now();
    s2_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
    const auto end_s2 = std::chrono::steady_clock::now();
    time_s2 += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_s2 - start_s2);
    perf_observer.add_frame_elapsed(
        "S2", static_cast<std::uint64_t>(tot_snap_allfiles +
                                         global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_s2 - start_s2)
            .count());
  }

  if (hbk_analyzer.enabled()) {
    auto start_hbk = std::chrono::steady_clock::now();
    size_t lig_snap_hbk = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
    const MOL *lig_frame_hbk =
        lig.m_mol.empty() ? nullptr : &lig.m_mol[lig_snap_hbk];
    size_t water_snap_hbk =
        (water_traj.m_mol.size() > 1) ? chunk_snap_idx : 0;
    const MOL *water_frame_hbk =
        water_traj.m_mol.empty() ? nullptr : &water_traj.m_mol[water_snap_hbk];
    hbk_analyzer.process_frame(pro_in.m_mol[chunk_snap_idx], lig_frame_hbk,
                               water_frame_hbk, executor);
    time_hbk += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_hbk);
  }

  if (rg_analyzer.enabled()) {
    auto start_rg = std::chrono::steady_clock::now();
    rg_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
    time_rg += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_rg);
  }

  if (rms_analyzer.enabled()) {
    auto start_rms = std::chrono::steady_clock::now();
      rms_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
    time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_rms);
  }

  if (Arguments.flag_pqr == 1) {
    write_pqr(pro_in, lig, input_line, input_name, chunk_snap_idx,
              Arguments, paths);
  }
  auto start_pdb = std::chrono::steady_clock::now();
  write_pdb(pro_in, lig, Arguments, chunk_snap_idx, paths, input_line,
            input_name);
  time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pdb);

  if (Arguments.flag_cas) {
    auto start_cas = std::chrono::steady_clock::now();

    cas_analyzer.process_frame(
        pro_in, lig, chunk_snap_idx, executor, num_threads, dis_min,
        CachedData, Arguments, info_output, rec_free_radii, lg_free_radii,
        rec_complex_radii, lg_complex_radii, frame_gbsa_total);

    const auto end_cas = std::chrono::steady_clock::now();
    time_cas += std::chrono::duration_cast<std::chrono::milliseconds>(
        end_cas - start_cas);
    perf_observer.add_frame_elapsed(
        "CAS", static_cast<std::uint64_t>(tot_snap_allfiles +
                                          global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_cas - start_cas)
            .count());
  }

  const int current_replica = input_name - Arguments.b_input;
  const size_t absolute_global_snap_idx =
      tot_snap_allfiles + global_snap_idx;
  const double current_box_volume_nm3 =
      frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);

  // Network Analysis (DCCM) Data Collection
  auto start_network = std::chrono::steady_clock::now();
  if (Arguments.flag_network || Arguments.flag_pca ||
      Arguments.corr_knn.flag_corr_knn ||
      Arguments.lspmd.flag_lspmd || tica_or_export_needs_dccm) {
    dccm_analyzer.process_frame(
        pro_in, lig, chunk_snap_idx, absolute_global_snap_idx,
        pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
        current_box_volume_nm3);
  }
  time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_network);

  // --- TICA per-frame accumulation (pairwise Cα distances + Kabsch
  // align) ---
  auto start_tica = std::chrono::steady_clock::now();
  if (Arguments.msm.flag_tica && tica_analyzer) {
    const auto start_tica_feature_build =
        std::chrono::steady_clock::now();
    bool have_tica_frame = false;
    if (use_observable_tica_features) {
      Eigen::VectorXd observable_features;
      have_tica_frame = build_observable_feature_vector(
          Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
          tica_ref_centered, tica_R, tica_ref_initialized,
          qnative_analyzer, dihedral_analyzer, cmap_analyzer,
          lj_analyzer, observable_features);
      if (have_tica_frame &&
          feature_router.accept_frame(
              static_cast<std::uint64_t>(absolute_global_snap_idx),
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              observable_features)) {
        tica_features = observable_features;
      } else {
        have_tica_frame = false;
      }
    } else if (use_lj_tica_features && lj_analyzer.has_feature_vector()) {
      tica_features = lj_analyzer.latest_feature_vector();
      have_tica_frame = (tica_features.size() > 0);
    } else if (tica_n_ca > 0) {
      size_t nf = dccm_analyzer.get_total_frames();
      if (nf > 0) {
        const auto &tx = dccm_analyzer.get_latest_traj_x();
        const auto &ty = dccm_analyzer.get_latest_traj_y();
        const auto &tz = dccm_analyzer.get_latest_traj_z();

      // Fill tica_mobile (pre-allocated N_ca×3)
      for (int a = 0; a < tica_n_ca; ++a) {
        tica_mobile(a, 0) = tx[static_cast<size_t>(a)];
        tica_mobile(a, 1) = ty[static_cast<size_t>(a)];
        tica_mobile(a, 2) = tz[static_cast<size_t>(a)];
      }

      if (!tica_ref_initialized) {
        // Frame 0: center and store as Kabsch reference
        tica_ref_centered = tica_mobile;
        {
          Eigen::RowVectorXd col_mean = tica_ref_centered.colwise().mean();
          tica_ref_centered.rowwise() -= col_mean;
        }
        tica_mobile = tica_ref_centered;
        tica_ref_initialized = true;
      } else {
        // Frame > 0: zero-alloc Kabsch alignment
        TitanAlignment::align_frame_inplace(tica_ref_centered,
                                            tica_mobile, tica_R);
      }

      // Science fix: pairwise inter-Cα distances (rotation+translation
      // invariant) D = N_ca*(N_ca-1)/2 — no spurious global
      // rotation/translation modes
      int feat_idx = 0;
      for (int a = 0; a < tica_n_ca; ++a) {
        for (int b = a + 1; b < tica_n_ca; ++b) {
          double dx = tica_mobile(a, 0) - tica_mobile(b, 0);
          double dy = tica_mobile(a, 1) - tica_mobile(b, 1);
          double dz = tica_mobile(a, 2) - tica_mobile(b, 2);
          tica_features[feat_idx++] =
              std::sqrt(dx * dx + dy * dy + dz * dz);
        }
      }

        have_tica_frame = true;
      }
    }
    tica_analyzer->add_feature_build_seconds(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            start_tica_feature_build)
            .count());
    if (have_tica_frame) {
      const double current_box_volume_nm3 =
          frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);
      tica_analyzer->accumulate_frame(tica_features);

      if (need_tica_projection_cache && msm_analyzer) {
        if (tica_analyzer->is_solved()) {
          Eigen::VectorXd proj =
              tica_analyzer->project_frame(tica_features);
          msm_analyzer->add_projected_frame(
              proj, current_replica,
              static_cast<int64_t>(absolute_global_snap_idx),
              current_box_volume_nm3);
          msm_analyzer->record_frame_time(
              pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica);
        } else {
          msm_analyzer->add_pending_frame_metadata(
              pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
              static_cast<int64_t>(absolute_global_snap_idx),
              current_box_volume_nm3);
        }
      }
    }
  }
  time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_tica);

  if (feature_router.active() && !feature_router.consume_tica()) {
    Eigen::VectorXd observable_features;
    const bool have_feature_frame = build_observable_feature_vector(
        Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
        tica_ref_centered, tica_R, tica_ref_initialized, qnative_analyzer,
        dihedral_analyzer, cmap_analyzer, lj_analyzer,
        observable_features);
    if (have_feature_frame) {
      const bool accepted_for_export = feature_router.accept_frame(
          static_cast<std::uint64_t>(absolute_global_snap_idx),
          pro_in.m_mol[chunk_snap_idx].time_of_snap, observable_features);
      (void)accepted_for_export;
    }
  }

  auto start_tda = std::chrono::steady_clock::now();
  if (Arguments.flag_tda) {
    if (!Arguments.flag_sidechain) {
      TitanUI::print_info(
          "TDA is running without sidechain filtering (extracting ALL "
          "heavy atoms). Enable flag -sc for active-only.");
      std::vector<int>
          target_idx_1; // Empty means "Extract ALL Heavy Atoms"
      std::vector<int> target_idx_2;
      std::string tda_out = "tda_global_pointcloud.csv";
      tda_analyzer.export_TDA_pointcloud(
          pro_in, lig, input_name, chunk_snap_idx, target_idx_1,
          target_idx_2, Arguments, tda_out);
    } else {
      std::unordered_set<int> active_1;
      std::unordered_set<int> active_2;

      bool mol1_lig = (Arguments.mol_1_type == "ligand" ||
                       Arguments.mol_1_type == "lig");
      bool mol2_lig = (Arguments.mol_2_type == "ligand" ||
                       Arguments.mol_2_type == "lig");

      // Directly grab the active residue/atom numbers forming contacts
      // from d_min array
      for (size_t t{}; t < num_threads; ++t) {
        for (const auto &contact : dis_min[t].base) {
          if (mol1_lig) {
            active_1.insert(contact.atom_num_rec -
                            1); // 0-indexed atom num
          } else {
            active_1.insert(contact.resnr_rec);
          }

          if (mol2_lig) {
            active_2.insert(contact.atom_num - 1); // 0-indexed atom num
          } else {
            active_2.insert(contact.resnr_ab);
          }
        }
      }

      std::vector<int> target_idx_1(active_1.begin(), active_1.end());
      std::vector<int> target_idx_2(active_2.begin(), active_2.end());

      std::string tda_out = "tda_global_pointcloud.csv";
      tda_analyzer.export_TDA_pointcloud(
          pro_in, lig, input_name, chunk_snap_idx, target_idx_1,
          target_idx_2, Arguments, tda_out);
    }
  }
  time_tda += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_tda);

  auto start_fmap{std::chrono::steady_clock::now()};
  if (Arguments.flag_cal_force == true &&
      (Arguments.force_map == 1 || Arguments.force_map == 3) &&
      !CachedData.elec_ener.empty()) {
    wrap_cal_force_map(executor, pro_in, lig, Arguments, chunk_snap_idx,
                       dis_min, CachedData, Force_res_mol1,
                       Force_res_mol2, info_output);
  }
  time_fmap += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_fmap);

  auto start_hb{std::chrono::steady_clock::now()};

  if (Arguments.flag_h_bond == 1) {
    find_h_bonds(pro_in.m_mol[chunk_snap_idx], pro_in.m_mol[0],
                 lig.m_mol[chunk_snap_idx], lig.m_mol[0], groups_1,
                 groups_2, Arguments, global_snap_idx, pro_in.info,
                 lig.info, Hbonds_vector, &executor);

    if (Arguments.nt >= 2) {
      executor.parallel_do(2, [&](size_t worker_id) {
        if (worker_id == 0) {
          add_hb_res(pro_in.m_mol[chunk_snap_idx],
                     lig.m_mol[chunk_snap_idx], pro_in.info, lig.info,
                     Arguments, hb_rec_1traj);
        } else {
          add_hb_res(lig.m_mol[chunk_snap_idx],
                     pro_in.m_mol[chunk_snap_idx], lig.info, pro_in.info,
                     Arguments, hb_lg_1traj);
        }
      });
    } else {
      add_hb_res(pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
                 pro_in.info, lig.info, Arguments, hb_rec_1traj);
      add_hb_res(lig.m_mol[chunk_snap_idx], pro_in.m_mol[chunk_snap_idx],
                 lig.info, pro_in.info, Arguments, hb_lg_1traj);
    }
  } // End of flag_h_bond block

  time_hb += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_hb);

  if (Arguments.n_wat >= 0 && !water_traj.m_mol.empty()) {
    if (chunk_snap_idx >= water_traj.m_mol.size()) {
      if (chunk_snap_idx == 1) {
        std::cout
            << "\n[WARNING] Water group has only "
            << water_traj.m_mol.size() << " frame(s), but trajectory has "
            << NUM << " frames.\n"
            << "Water analysis will only process available frames!\n";
      }
    } else {
      // Optimization: Distill heavy atom indices just once
      static std::vector<size_t> rec_heavy_atoms;
      static std::vector<size_t> lig_heavy_atoms;

      if (chunk_snap_idx == 0) {
        rec_heavy_atoms.clear();
        lig_heavy_atoms.clear();

        for (size_t i = 0; i < pro_in.info.size(); ++i) {
          if (pro_in.info[i].get_atom()[0] != 'H')
            rec_heavy_atoms.emplace_back(i);
        }
        for (size_t i = 0; i < lig.info.size(); ++i) {
          if (lig.info[i].get_atom()[0] != 'H')
            lig_heavy_atoms.emplace_back(i);
        }
      }

      const MOL &w_frame = water_traj.m_mol[chunk_snap_idx];
      const MOL &p_frame = pro_in.m_mol[chunk_snap_idx];
      size_t l_snap = (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
      const MOL &l_frame = lig.m_mol[l_snap];

      /*if (chunk_snap_idx == 0) {
        std::cout << "[DEBUG WB] Snapshot 0 - rec heavy atoms: "
                  << rec_heavy_atoms.size()
                  << ", lig heavy atoms: " << lig_heavy_atoms.size()
                  << ", water atoms: " << w_frame.pos.x.size() << "\n";
      }*/

      // Water bridges and water-mediated contacts (WaterAnalyzer).
      auto start_water = std::chrono::steady_clock::now();
      water_analyzer.process_frame(
          p_frame, l_frame, w_frame, rec_heavy_atoms, lig_heavy_atoms,
          pro_in.info, lig.info, input_name, chunk_snap_idx,
          info_output.get_global_water_bridges_timeseries(),
          info_output.get_water_pdb_filename(Arguments, input_line,
                                             input_name));
      time_water += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_water);

      // --- GIST per-frame ---
      if (Arguments.gist.flag_gist && gist_analyzer) {
        const bool gist_use_tica =
            Arguments.msm.flag_tica && tica_ref_initialized;
        gist_analyzer->process_frame(
            p_frame, l_frame, w_frame, pro_in.info,
            gist_use_tica ? tica_ref_centered : Eigen::MatrixXd{},
            gist_use_tica ? tica_R : Eigen::Matrix3d::Identity(),
            (global_snap_idx == 0), gist_use_tica, &executor);
      }
    }
  }
  auto start_subgroups = std::chrono::steady_clock::now();
  if (Arguments.flag_sub_groups == true && Arguments.flag_gbsa == false) {
    if (sg_streaming_analyzer.active()) {
      const std::size_t sg_lig_snap =
          (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
      sg_streaming_analyzer.process_frame(
          pro_in.m_mol[chunk_snap_idx], lig.m_mol[sg_lig_snap], pro_in,
          lig, CachedData, Arguments, executor,
          static_cast<std::size_t>(input_name - Arguments.b_input));
      if (Arguments.flag_cal_mindis == true)
        mindis_subgroups(Groups_from_index_file, pro_in, lig,
                         chunk_snap_idx, NUM, Arguments, executor,
                         MIN_DIS);
    } else if (!CachedData.elec_ener.empty()) {
      Energy_force_subgroups(Groups_from_index_file, pro_in, lig,
                             CachedData, EF_subgroups, Energy_groups_out,
                             chunk_snap_idx, NUM, Arguments, executor);
      if (Arguments.flag_cal_mindis == true)
        mindis_subgroups(Groups_from_index_file, pro_in, lig,
                         chunk_snap_idx, NUM, Arguments, executor,
                         MIN_DIS);
    }
  }
  time_subgroups_cal +=
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start_subgroups);
  // cout << "check sub groups..";
  auto start_dis2commol = std::chrono::steady_clock::now();
  if (Arguments.com_dis_energy_2mol != 4)
    dis_com_2mol(pro_in, lig, chunk_snap_idx, NUM, Arguments, input_name,
                 paths, E_E_, E_V_, info_output);
  // -com-dis (the only src/nocal/ analysis the energy mode runs).
  try {
    nocal_analyses.process_frame(Arguments, pro_in, lig, chunk_snap_idx);
  } catch (const std::exception &e) {
    TitanUI::print_error(e.what());
    return 1;
  }
  time_other += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_dis2commol);

  // --- ML Feature Extraction ---
  if (Arguments.flag_ml) {
    ml_extractor.process_frame(
        pro_in.m_mol[chunk_snap_idx], lig.m_mol[chunk_snap_idx],
        pro_in.info, lig.info, lig.parameters, dis_min, E_E_ + E_V_, 6.0);
  }

  // In chunk streaming, we do not clear individual MOLs here, we clear
  // the entire vector before reading the next chunk.
  return std::nullopt;
}

} // namespace engine_detail

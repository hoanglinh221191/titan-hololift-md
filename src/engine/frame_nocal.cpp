#include "engine/engine_run.h"
#include "energy/group_bonds.h"

namespace engine_detail {

std::optional<int> EngineTrajectory::process_nocal_frame(size_t chunk_snap_idx) {
  const size_t global_snap_idx = global_frame_counter + chunk_snap_idx;
  // One time per frame number (the rows of the H-bond and volume tables),
  // from the first input that reaches it.
  while (time_of_snap.size() < global_snap_idx)
    time_of_snap.push_back(std::numeric_limits<double>::quiet_NaN());
  if (time_of_snap.size() == global_snap_idx)
    time_of_snap.push_back(
        Arguments.output_time(pro_in.m_mol[chunk_snap_idx].time_of_snap));
  const int progress_traj_id = input_name - Arguments.b_input + 1;
  std::string status_text =
      "Traj " + std::to_string(progress_traj_id) + " | Frame " +
      std::to_string(global_snap_idx + 1) + " | Global " +
      std::to_string((tot_snap_allfiles + global_snap_idx + 1));

  const std::string module_name =
      progress_source_label(Arguments, name_pro);

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
    const MOL *pbe_lig_frame =
        (chunk_snap_idx < lig.m_mol.size()) ? &lig.m_mol[chunk_snap_idx]
                                            : nullptr;
    // ENE-22: the MM part of MM/PBSA has the same inter-group sums.
    if (Arguments.pbe.pbsa && global_snap_idx == 0 && pbe_lig_frame &&
        !Arguments.self_int && !CachedData.nonbonded_topology) {
      const std::string bonded = titan_group_bonds::bonded_groups_error(
          pro_in.m_mol[chunk_snap_idx], pro_in.info, *pbe_lig_frame, lig.info,
          executor);
      if (!bonded.empty()) {
        TitanUI::print_error(bonded);
        TitanUI::log_message(log, "[Validation] " + bonded);
        return 1;
      }
    }
    const Multi_MOL_parameters *pbe_lig_params =
        pbe_lig_frame ? &lig.parameters : nullptr;
    pbe_analyzer.consume_frame(
        pro_in.m_mol[chunk_snap_idx], pro_in.parameters, pbe_lig_frame,
        pbe_lig_params,
        static_cast<std::size_t>(progress_traj_id), global_snap_idx,
        tot_snap_allfiles + global_snap_idx, executor, log,
        Arguments.pbe.pbsa ? &pbe_energy_provider : nullptr);
    time_pbe += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_pbe);
  }

  // COM distance, dipole, charged-atom distances, water dipole/field,
  // residue-to-COM distances and box volume of this frame (src/nocal/).
  try {
    nocal_analyses.process_frame(Arguments, pro_in, lig, chunk_snap_idx);
  } catch (const std::exception &e) {
    TitanUI::print_error(e.what());
    return 1;
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
        "CMap", static_cast<std::uint64_t>(
                    tot_snap_allfiles + global_frame_counter +
                    chunk_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_cmap -
                                                  start_cmap)
            .count());
  }

  if (Arguments.ifp.flag_ifp && ifp_analyzer &&
      include_ifp_frame(
          pro_in, chunk_snap_idx,
          static_cast<size_t>(progress_traj_id), global_snap_idx + 1,
          tot_snap_allfiles + global_snap_idx + 1)) {
    auto start_sc = std::chrono::steady_clock::now();
    std::vector<d_min> ifp_dis_min(num_threads);

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
                  lig.Index[residue_idx + 1].get_res_pos_e(0),
                  lig.Index, lig.info);
            }
          });
    }

    impl_add_coms_2_mols(lig, pro_in, Arguments, chunk_snap_idx);
    executor.parallel_do(num_threads, [&](size_t worker_id) {
      take_min(
          pro_in.m_mol[chunk_snap_idx].Com,
          lig.m_mol[chunk_snap_idx].Com, ifp_dis_min[worker_id],
          worker_id * pro_in.m_mol[chunk_snap_idx].Com.size() /
              num_threads,
          (worker_id + 1) *
              pro_in.m_mol[chunk_snap_idx].Com.size() / num_threads,
          0, lig.m_mol[chunk_snap_idx].Com.size(), Arguments, 0,
          d_standard);
    });
    time_sc += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_sc);

    auto start_ifp = std::chrono::steady_clock::now();
    const size_t lg_snap_ifp =
        (lig.m_mol.size() > 1) ? chunk_snap_idx : 0;
    ifp_analyzer->process_frame(
        pro_in.m_mol[chunk_snap_idx], lig.m_mol[lg_snap_ifp],
        pro_in.info, lig.info, pro_in.Index, lig.Index,
        pro_in.info.size(), lig.info.size(), ifp_dis_min,
        Arguments.ifp, executor, num_threads, false);
    time_ifp += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_ifp);
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
        static_cast<std::uint64_t>(tot_snap_allfiles +
                                   global_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_interface -
                                                  start_interface)
            .count());
  }

  if (dihedral_analyzer.enabled()) {
    auto start_dihedral = std::chrono::steady_clock::now();
    dihedral_analyzer.process_frame(pro_in, chunk_snap_idx, executor);
    const auto end_dihedral = std::chrono::steady_clock::now();
    time_dihedral +=
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_dihedral - start_dihedral);
    perf_observer.add_frame_elapsed(
        "Dihedral",
        static_cast<std::uint64_t>(tot_snap_allfiles +
                                   global_frame_counter +
                                   chunk_snap_idx),
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
        "QNative",
        static_cast<std::uint64_t>(tot_snap_allfiles +
                                   global_frame_counter +
                                   chunk_snap_idx),
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
        "MSD", static_cast<std::uint64_t>(
                   tot_snap_allfiles + global_frame_counter +
                   chunk_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_msd - start_msd)
            .count());
  }

  // -hbk in a -nocal run: the energy frame loop calls it, this loop did not,
  // so the summary said "active" over zero frames (STR-21 follow-up).
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
        "Membrane",
        static_cast<std::uint64_t>(tot_snap_allfiles +
                                   global_frame_counter +
                                   chunk_snap_idx),
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
                                         global_frame_counter +
                                         chunk_snap_idx),
        pro_in.m_mol[chunk_snap_idx].time_of_snap,
        std::chrono::duration<double, std::milli>(end_s2 - start_s2)
            .count());
  }

  if (rg_analyzer.enabled()) {
    auto start_rg = std::chrono::steady_clock::now();
    rg_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
    time_rg += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_rg);
  }

  if (Arguments.flag_nocal && use_lj_tica_features &&
      !use_observable_tica_features &&
      Arguments.msm.flag_tica && tica_analyzer &&
      lj_analyzer.has_feature_vector()) {
    auto start_tica = std::chrono::steady_clock::now();
    const int current_replica = input_name - Arguments.b_input;
    const size_t absolute_global_snap_idx =
        tot_snap_allfiles + global_frame_counter + chunk_snap_idx;
    const double current_box_volume_nm3 =
        frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);
    const auto start_tica_feature_build =
        std::chrono::steady_clock::now();
    tica_features = lj_analyzer.latest_feature_vector();
    tica_analyzer->add_feature_build_seconds(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            start_tica_feature_build)
            .count());
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

    time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_tica);
  }

  if (rms_analyzer.enabled()) {
    auto start_rms = std::chrono::steady_clock::now();
    rms_analyzer.process_frame(pro_in, lig, chunk_snap_idx, executor);
    time_rms += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_rms);
  }

  const int current_replica = input_name - Arguments.b_input;
  const size_t absolute_global_snap_idx =
      tot_snap_allfiles + global_frame_counter + chunk_snap_idx;
  const double current_box_volume_nm3 =
      frame_box_volume_nm3(pro_in.m_mol[chunk_snap_idx].Box);

  auto start_network = std::chrono::steady_clock::now();
  if (Arguments.flag_network || Arguments.flag_pca ||
      Arguments.corr_knn.flag_corr_knn ||
      Arguments.lspmd.flag_lspmd ||
      tica_or_export_needs_dccm) {
    dccm_analyzer.process_frame(
        pro_in, lig, chunk_snap_idx, absolute_global_snap_idx,
        pro_in.m_mol[chunk_snap_idx].time_of_snap, current_replica,
        current_box_volume_nm3);
  }
  time_network += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_network);

  if (Arguments.msm.flag_tica && tica_analyzer &&
      use_observable_tica_features) {
    auto start_tica = std::chrono::steady_clock::now();
    const auto start_tica_feature_build =
        std::chrono::steady_clock::now();
    Eigen::VectorXd observable_features;
    const bool have_tica_frame = build_observable_feature_vector(
        Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
        tica_ref_centered, tica_R, tica_ref_initialized,
        qnative_analyzer, dihedral_analyzer, cmap_analyzer,
        lj_analyzer, observable_features);
    tica_analyzer->add_feature_build_seconds(
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            start_tica_feature_build)
            .count());
    if (have_tica_frame &&
        feature_router.accept_frame(
            static_cast<std::uint64_t>(absolute_global_snap_idx),
            pro_in.m_mol[chunk_snap_idx].time_of_snap,
            observable_features)) {
      tica_analyzer->accumulate_frame(observable_features);
      if (need_tica_projection_cache && msm_analyzer) {
        if (tica_analyzer->is_solved()) {
          Eigen::VectorXd proj =
              tica_analyzer->project_frame(observable_features);
          msm_analyzer->add_projected_frame(
              proj, current_replica,
              static_cast<int64_t>(absolute_global_snap_idx),
              current_box_volume_nm3);
          msm_analyzer->record_frame_time(
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              current_replica);
        } else {
          msm_analyzer->add_pending_frame_metadata(
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              current_replica,
              static_cast<int64_t>(absolute_global_snap_idx),
              current_box_volume_nm3);
        }
      }
    }
    time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_tica);
  } else if (Arguments.msm.flag_tica && tica_analyzer &&
             !use_lj_tica_features) {
    auto start_tica = std::chrono::steady_clock::now();
    const auto start_tica_feature_build =
        std::chrono::steady_clock::now();
    bool have_tica_frame = false;
    if (tica_n_ca > 0) {
      size_t nf = dccm_analyzer.get_total_frames();
      if (nf > 0) {
        const auto &tx = dccm_analyzer.get_latest_traj_x();
        const auto &ty = dccm_analyzer.get_latest_traj_y();
        const auto &tz = dccm_analyzer.get_latest_traj_z();

        for (int a = 0; a < tica_n_ca; ++a) {
          tica_mobile(a, 0) = tx[static_cast<size_t>(a)];
          tica_mobile(a, 1) = ty[static_cast<size_t>(a)];
          tica_mobile(a, 2) = tz[static_cast<size_t>(a)];
        }

        if (!tica_ref_initialized) {
          tica_ref_centered = tica_mobile;
          {
            Eigen::RowVectorXd col_mean =
                tica_ref_centered.colwise().mean();
            tica_ref_centered.rowwise() -= col_mean;
          }
          tica_mobile = tica_ref_centered;
          tica_ref_initialized = true;
        } else {
          TitanAlignment::align_frame_inplace(tica_ref_centered,
                                              tica_mobile, tica_R);
        }

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
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              current_replica);
        } else {
          msm_analyzer->add_pending_frame_metadata(
              pro_in.m_mol[chunk_snap_idx].time_of_snap,
              current_replica,
              static_cast<int64_t>(absolute_global_snap_idx),
              current_box_volume_nm3);
        }
      }
    }

    time_tica += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_tica);
  }

  if (feature_router.active() && !feature_router.consume_tica()) {
    Eigen::VectorXd observable_features;
    const bool have_feature_frame = build_observable_feature_vector(
        Arguments, dccm_analyzer, tica_n_ca, tica_mobile,
        tica_ref_centered, tica_R, tica_ref_initialized,
        qnative_analyzer, dihedral_analyzer, cmap_analyzer,
        lj_analyzer, observable_features);
    if (have_feature_frame) {
      const bool accepted_for_export = feature_router.accept_frame(
          static_cast<std::uint64_t>(absolute_global_snap_idx),
          pro_in.m_mol[chunk_snap_idx].time_of_snap,
          observable_features);
      (void)accepted_for_export;
    }
  }

  // H-bond (self-interaction mode for nocal). find_h_bonds() finds the donor
  // and acceptor groups at the first frame of an input only, so they must
  // outlive the frame: the trajectory's groups_1/groups_2.
  if (Arguments.flag_h_bond == 1) {
    MOL temp_mol(pro_in.m_mol[chunk_snap_idx]);
    find_h_bonds(pro_in.m_mol[chunk_snap_idx], pro_in.m_mol[0],
                 temp_mol, temp_mol, groups_1, groups_2,
                 Arguments, global_snap_idx, pro_in.info,
                 pro_in.info, Hbonds_vector, &executor);
  }

  // PQR output
  if (Arguments.flag_pqr == 1)
    write_pqr(pro_in, lig, input_line, input_name, chunk_snap_idx,
              Arguments, paths);

  // PDB output (critical for nocal)
  auto start_pdb = std::chrono::steady_clock::now();
  write_pdb(pro_in, lig, Arguments, chunk_snap_idx, paths, input_line,
            input_name);
  time_pdb += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_pdb);
  return std::nullopt;
}

} // namespace engine_detail

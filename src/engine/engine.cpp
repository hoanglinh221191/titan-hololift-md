#include "engine/engine_run.h"

// engine(): the post-processing driver of TITAN. The former function body is
// split into EngineRun (setup, trajectory loop, finish) and EngineTrajectory
// (one trajectory: stages, chunk loop, frame bodies); see engine_run.h.

namespace engine_detail {

EngineRun::EngineRun(arguments &Arguments) : Arguments(Arguments) {}

int EngineRun::run() {
  if (const std::optional<int> exit_code = setup())
    return *exit_code;
  while (true) {
    EngineTrajectory trajectory(*this);
    if (const std::optional<int> exit_code = trajectory.run_trajectory())
      return *exit_code;
    if (trajectory.stopped())
      break;
  }
  if (const std::optional<int> exit_code = finish_outputs())
    return *exit_code;
  return finish_postproc().value_or(0);
}

EngineTrajectory::EngineTrajectory(EngineRun &run)
    : run(run),
      Arguments(run.Arguments),
      paths(run.paths),
      DE_arguments(run.DE_arguments),
      sc_flag(run.sc_flag),
      force_f(run.force_f),
      d_standard(run.d_standard),
      num_threads(run.num_threads),
      tot_snap_allfiles(run.tot_snap_allfiles),
      executor(*run.executor),
      name_pro(run.name_pro),
      name_lig(run.name_lig),
      parameter(run.parameter),
      E_v(run.E_v),
      E_e(run.E_e),
      size_snap(run.size_snap),
      E_v_vector(run.E_v_vector),
      E_e_vector(run.E_e_vector),
      nocal_analyses(run.nocal_analyses),
      Force_e(run.Force_e),
      Force_v(run.Force_v),
      Force_e_vector(run.Force_e_vector),
      Force_v_vector(run.Force_v_vector),
      Force_res_stat_mol1(run.Force_res_stat_mol1),
      Force_res_stat_mol2(run.Force_res_stat_mol2),
      EF_subgroups(run.EF_subgroups),
      sg_streaming_analyzer(run.sg_streaming_analyzer),
      sg_streaming_initialized(run.sg_streaming_initialized),
      MIN_DIS(run.MIN_DIS),
      Hbonds_vector(run.Hbonds_vector),
      amine(run.amine),
      time_of_snap(run.time_of_snap),
      input_line(run.input_line),
      cscd_out_dis(run.cscd_out_dis),
      saved_pro_frame0(run.saved_pro_frame0),
      saved_lig_frame0(run.saved_lig_frame0),
      saved_membrane_frame0(run.saved_membrane_frame0),
      topo_2_is_open(run.topo_2_is_open),
      topo_is_open(run.topo_is_open),
      status(run.status),
      Groups_from_index_file(run.Groups_from_index_file),
      info_output(*run.info_output),
      res_hb_rec(run.res_hb_rec),
      res_hb_lg(run.res_hb_lg),
      log(run.log),
      available_ram(run.available_ram),
      need_tica_projection_cache(run.need_tica_projection_cache),
      active_modules(run.active_modules),
      input_name(run.input_name),
      global_gbsa_agg(*run.global_gbsa_agg),
      cas_analyzer(run.cas_analyzer),
      dssp_mol1_analyzer(run.dssp_mol1_analyzer),
      dssp_mol2_analyzer(run.dssp_mol2_analyzer),
      hbk_analyzer(run.hbk_analyzer),
      lj_analyzer(run.lj_analyzer),
      sq_analyzer(run.sq_analyzer),
      cmap_analyzer(run.cmap_analyzer),
      interface_analyzer(run.interface_analyzer),
      dihedral_analyzer(run.dihedral_analyzer),
      qnative_analyzer(run.qnative_analyzer),
      feature_router(run.feature_router),
      perf_observer(run.perf_observer),
      msd_analyzer(run.msd_analyzer),
      membrane_analyzer(run.membrane_analyzer),
      s2_analyzer(run.s2_analyzer),
      rg_analyzer(run.rg_analyzer),
      rms_analyzer(run.rms_analyzer),
      lj_feature_files(run.lj_feature_files),
      na_profiler(run.na_profiler),
      tda_analyzer(run.tda_analyzer),
      global_has_nucleic_acid(run.global_has_nucleic_acid),
      dccm_analyzer(run.dccm_analyzer),
      dccm_analyzer_initialized(run.dccm_analyzer_initialized),
      time_gbsa(run.time_gbsa),
      time_cas(run.time_cas),
      time_dssp(run.time_dssp),
      time_hbk(run.time_hbk),
      time_lj(run.time_lj),
      time_sq(run.time_sq),
      time_cmap(run.time_cmap),
      time_interface(run.time_interface),
      time_dihedral(run.time_dihedral),
      time_qnative(run.time_qnative),
      time_msd(run.time_msd),
      time_rg(run.time_rg),
      time_rms(run.time_rms),
      time_membrane(run.time_membrane),
      time_s2(run.time_s2),
      time_edecomp(run.time_edecomp),
      time_network(run.time_network),
      time_tda(run.time_tda),
      time_llps(run.time_llps),
      time_tica(run.time_tica),
      dbscan_llps(run.dbscan_llps),
      llps_energy_provider(run.llps_energy_provider),
      time_llps_dbscan(run.time_llps_dbscan),
      water_analyzer(run.water_analyzer),
      ml_extractor(run.ml_extractor),
      tica_analyzer(run.tica_analyzer),
      msm_analyzer(run.msm_analyzer),
      llps_analyzer(run.llps_analyzer),
      gist_analyzer(run.gist_analyzer),
      pbe_analyzer(run.pbe_analyzer),
      pbe_energy_provider(run.pbe_energy_provider),
      edecomp_energy_provider(run.edecomp_energy_provider),
      edecomp_engine(run.edecomp_engine),
      edecomp_gbsa_engine(run.edecomp_gbsa_engine),
      edecomp_pbsa_engine(run.edecomp_pbsa_engine),
      time_cal(run.time_cal),
      time_fmap(run.time_fmap),
      time_subgroups_cal(run.time_subgroups_cal),
      time_solve_pbc(run.time_solve_pbc),
      time_sc(run.time_sc),
      time_hb(run.time_hb),
      time_pdb(run.time_pdb),
      time_other(run.time_other),
      time_print_force(run.time_print_force),
      time_maxforce(run.time_maxforce),
      time_input_init(run.time_input_init),
      time_input_stream(run.time_input_stream),
      time_cscd(run.time_cscd),
      time_water(run.time_water),
      time_na(run.time_na),
      time_pbe(run.time_pbe),
      pro_in(run.pro_in),
      lig(run.lig),
      water_traj(run.water_traj),
      membrane_traj(run.membrane_traj),
      topo_initialized(run.topo_initialized),
      charge_initialized(run.charge_initialized),
      ifp_analyzer(run.ifp_analyzer),
      awh_ifp_analyzer(run.awh_ifp_analyzer),
      time_ifp(run.time_ifp),
      tica_n_ca_global(run.tica_n_ca_global),
      tica_feat_dim_global(run.tica_feat_dim_global),
      tica_features_global(run.tica_features_global),
      tica_mobile_global(run.tica_mobile_global),
      tica_ref_centered_global(run.tica_ref_centered_global),
      tica_R_global(run.tica_R_global),
      tica_ref_initialized_global(run.tica_ref_initialized_global),
      tica_use_lj_features_global(run.tica_use_lj_features_global),
      tica_use_observable_features_global(run.tica_use_observable_features_global),
      rec_vdw_radii(pro_in.parameters.radius),
      lg_vdw_radii(lig.parameters.radius),
      tica_n_ca(tica_n_ca_global),
      tica_features(tica_features_global),
      tica_mobile(tica_mobile_global),
      tica_ref_centered(tica_ref_centered_global),
      tica_R(tica_R_global),
      tica_ref_initialized(tica_ref_initialized_global) {}

std::optional<int> EngineTrajectory::run_trajectory() {
  if (const std::optional<int> exit_code = open_trajectory())
    return exit_code;
  if (stop_)
    return std::nullopt;
  if (const std::optional<int> exit_code = prepare_topology())
    return exit_code;
  if (const std::optional<int> exit_code = prepare_analyzers())
    return exit_code;
  if (const std::optional<int> exit_code = init_streaming())
    return exit_code;
  if (const std::optional<int> exit_code = init_modules())
    return exit_code;
  if (const std::optional<int> exit_code = stream_chunks())
    return exit_code;
  if (const std::optional<int> exit_code = finish_trajectory())
    return exit_code;
  if (const std::optional<int> exit_code = report_trajectory())
    return exit_code;
  if (stop_)
    return std::nullopt;
  return std::nullopt;
}

} // namespace engine_detail

int engine(arguments &Arguments) {
  engine_detail::EngineRun state(Arguments);
  return state.run();
}

#include "postproc/postproc_common.h"
#include "postproc/dham_frame_weights.h"
#include "titan_tram.h"

#include <fstream>
#include <utility>

using namespace postproc_detail;

namespace {

// KIN-13: citation lines in every -dham output.
void write_tram_header(std::ostream &out, const std::string &title,
                       const arguments &args) {
  out << "# " << title << "\n";
  out << "# estimator: TRAM (transition-based reweighting analysis method) "
         "over the replica temperatures\n";
  out << "# target_temperature_K " << args.dham.target_temp << "\n";
  for (const auto &reference : TitanTram::references())
    out << "# cite: " << reference << "\n";
}

void write_tram_summary(const arguments &args, const TitanTram &tram,
                        const TitanTram::Report &report,
                        const std::vector<double> &temps,
                        const std::string &input_mode) {
  std::ofstream out("dham_tram_summary.dat");
  if (!out)
    return;
  out << std::setprecision(12);
  write_tram_header(out, "-dham summary", args);
  out << "input_mode " << input_mode << "\n";
  out << "ensemble_temperatures_K";
  for (double t : temps)
    out << ' ' << t;
  out << "\n";
  out << "target_ensemble "
      << (tram.target_ensemble() >= 0 ? std::to_string(tram.target_ensemble())
                                      : std::string("none"))
      << "\n";
  out << "states " << tram.n_states() << "\n";
  out << "states_with_samples " << report.states_with_samples << "\n";
  out << "samples " << report.samples << "\n";
  out << "transitions " << report.transitions << "\n";
  out << "iterations " << report.iterations << "\n";
  out << "converged " << (report.converged ? "yes" : "no") << "\n";
  out << "max_change_kT " << report.max_change << "\n";
  out << "max_row_sum_error " << report.max_row_sum_error << "\n";
  if (!report.message.empty())
    out << "message " << report.message << "\n";
  out << "note_samples every frame with an energy is a sample; its own "
         "energy (U, or U + P V in NPT) sets its weight at every temperature\n";
  out << "note_transitions transitions are counted at the MSM lag inside one "
         "replica, between two samples and not across a trajectory break; "
         "replica-exchange inputs need -dham-segment-map so that no "
         "transition spans an exchange\n";
  out << "note_censoring keeping only exchange-free windows is a selection "
         "that TRAM does not correct; rates need lag times well below the "
         "time between accepted exchanges\n";
  out << "note_kinetics the transition matrix is that of the ensemble at the "
         "target temperature; an unsimulated target temperature has "
         "stationary quantities and weights only\n";
}

// Solves the loaded model, writes pi, FES, the target transition matrix and
// the summary, and maps the sample weights back to the projected frames.
bool finish_tram(const arguments &args, TitanTram &tram,
                 const std::vector<double> &temps,
                 const std::vector<std::pair<int, std::size_t>> &sample_frames,
                 const std::unordered_map<int, std::vector<int>> &per_replica_dtrajs,
                 const std::unordered_map<int, std::vector<std::size_t>>
                     &per_replica_kept_positions,
                 const TitanMSM &msm, TitanExecutor &executor,
                 info &info_output, std::fstream &log,
                 const std::string &input_mode,
                 std::vector<double> &out_frame_weights,
                 Eigen::MatrixXd *out_reweighted_tpm,
                 Eigen::VectorXd *out_reweighted_pi) {
  const TitanTram::Report report =
      tram.solve(args.dham.max_iter, args.dham.tolerance, &executor);
  TitanUI::log_kv(log, "DHAM Estimator", "TRAM (Wu et al., PNAS 2016)");
  TitanUI::log_kv(log, "DHAM Input Mode", input_mode);
  TitanUI::log_kv(log, "DHAM TRAM Samples", std::to_string(report.samples));
  TitanUI::log_kv(log, "DHAM TRAM Transitions",
                  std::to_string(static_cast<long long>(report.transitions)));
  TitanUI::log_kv(log, "DHAM TRAM Iterations", std::to_string(report.iterations));
  TitanUI::log_kv(log, "DHAM TRAM Max Change (kT)",
                  std::to_string(report.max_change));
  write_tram_summary(args, tram, report, temps, input_mode);
  TitanUI::print_info("DHAM: TRAM summary and references written to "
                      "dham_tram_summary.dat");
  if (!report.converged) {
    TitanUI::print_warning("DHAM (TRAM): " + report.message);
    return false;
  }

  const Eigen::VectorXd &pi = tram.target_stationary();
  if (out_reweighted_pi)
    *out_reweighted_pi = pi;
  {
    std::ofstream f(info_output.get_dham_pi_filename());
    if (f) {
      write_tram_header(f, "-dham stationary distribution at the target "
                           "temperature", args);
      f << "# state_id  p_i\n";
      for (int i = 0; i < pi.size(); ++i)
        f << i << "  " << std::fixed << std::setprecision(8) << pi[i] << "\n";
      TitanUI::log_kv(log, "DHAM Pi File", info_output.get_dham_pi_filename());
    }
  }
  {
    const Eigen::VectorXd g = tram.target_free_energy_kJ();
    const std::string unit = args.eu_par > 1.1 ? "kcal/mol" : "kJ/mol";
    std::ofstream f(info_output.get_dham_fes_filename());
    if (f) {
      write_tram_header(f, "-dham free energy profile at the target "
                           "temperature (" + unit + ", min=0)", args);
      f << "# state_id  G_" << unit << "\n";
      for (int i = 0; i < g.size(); ++i) {
        f << i << "  " << std::fixed << std::setprecision(4);
        if (std::isinf(g[i]))
          f << "inf";
        else
          f << g[i] / args.eu_par;
        f << "\n";
      }
      TitanUI::log_kv(log, "DHAM FES File", info_output.get_dham_fes_filename());
    }
  }
  {
    const int target = tram.target_ensemble();
    std::ofstream f(info_output.get_dham_tpm_filename());
    if (target < 0) {
      if (f) {
        write_tram_header(f, "-dham transition matrix: none", args);
        f << "# no replica runs at the target temperature; TRAM estimates "
             "transition matrices of simulated ensembles only\n";
      }
      TitanUI::print_warning(
          "DHAM (TRAM): no replica at the target temperature, so no target "
          "transition matrix; stationary distribution, FES and frame weights "
          "are written.");
    } else {
      const Eigen::MatrixXd tpm = tram.transition_matrix(target);
      if (f) {
        write_tram_header(f, "-dham reversible transition matrix of replica " +
                                 std::to_string(target) + " (the target "
                                 "temperature)", args);
        f << "# " << tpm.rows() << "x" << tpm.cols()
          << "; rows of states without transitions in that replica are nan\n";
        f << std::fixed << std::setprecision(8);
        for (int i = 0; i < tpm.rows(); ++i) {
          for (int j = 0; j < tpm.cols(); ++j) {
            if (j > 0)
              f << ' ';
            if (std::isfinite(tpm(i, j)))
              f << tpm(i, j);
            else
              f << "nan";
          }
          f << "\n";
        }
        TitanUI::log_kv(log, "DHAM TPM File", info_output.get_dham_tpm_filename());
      }
      if (tpm.allFinite()) {
        if (out_reweighted_tpm)
          *out_reweighted_tpm = tpm;
      } else {
        TitanUI::print_warning(
            "DHAM (TRAM): some states have no transitions in the replica at "
            "the target temperature; its transition matrix is incomplete and "
            "is not passed to TPT or validation.");
      }
    }
  }

  // sample weights back to the projected frames
  const std::vector<double> weights = tram.sample_weights();
  std::unordered_map<int, std::vector<double>> per_replica_weights;
  for (const auto &[replica, dtraj] : per_replica_dtrajs)
    per_replica_weights[replica].assign(dtraj.size(), 0.0);
  for (std::size_t x = 0; x < weights.size() && x < sample_frames.size(); ++x)
    per_replica_weights[sample_frames[x].first][sample_frames[x].second] =
        weights[x];
  const auto &frame_replica_ids = msm.get_frame_replica_ids();
  const auto &frame_global_indices = msm.get_frame_global_indices();
  if (frame_replica_ids.size() != frame_global_indices.size()) {
    TitanUI::print_warning(
        "DHAM: projected frame metadata mismatch: replica_ids=" +
        std::to_string(frame_replica_ids.size()) + ", global_indices=" +
        std::to_string(frame_global_indices.size()) +
        ". Some frame weights may be dropped.");
  }
  // KIN-16: weights are indexed by position in the window-matched subset;
  // map them back through the kept positions, never by a running offset.
  titan_dham_weights::MapStats map_stats;
  out_frame_weights = titan_dham_weights::map_to_global_frames(
      per_replica_weights, per_replica_kept_positions, frame_replica_ids,
      frame_global_indices, map_stats);
  double weight_sum = 0.0;
  for (double w : out_frame_weights)
    weight_sum += w;
  TitanUI::print_info("DHAM: Computed " + std::to_string(map_stats.mapped_frames) +
                      " global frame weights for reweighted FES.");
  TitanUI::print_info("DHAM: Weight sum = " + std::to_string(weight_sum) +
                      " (expected ~1.0)");
  TitanUI::log_kv(log, "DHAM Weighted Frames",
                  std::to_string(map_stats.mapped_frames));
  TitanUI::log_kv(log, "DHAM Frames Outside Energy Window",
                  std::to_string(map_stats.dropped_by_window));
  if (map_stats.dropped_by_window > 0) {
    TitanUI::print_info(
        "DHAM: " + std::to_string(map_stats.dropped_by_window) +
        " projected frames have no matched energy sample and carry weight 0.");
  }
  const std::size_t dropped =
      map_stats.dropped_missing_replica + map_stats.inconsistent_positions;
  if (dropped > 0) {
    TitanUI::print_warning("DHAM: Dropped " + std::to_string(dropped) +
                           " projected frames while rebuilding global weights.");
  }
  if (map_stats.dropped_invalid_global_index > 0) {
    TitanUI::print_warning(
        "DHAM: Dropped " + std::to_string(map_stats.dropped_invalid_global_index) +
        " projected frames with invalid global indices.");
  }
  if (map_stats.duplicate_global_indices > 0) {
    TitanUI::print_warning(
        "DHAM: Encountered " +
        std::to_string(map_stats.duplicate_global_indices) +
        " duplicate global frame indices while assigning weights.");
  }
  return true;
}

} // namespace


// ============================================================================
// 6. DHAM post-processing
// ============================================================================
void run_dham_postproc(const arguments &args,
                       std::unique_ptr<TitanMSM> &msm_analyzer,
                       TitanExecutor &executor, info &info_output,
                       std::fstream &log,
                       std::vector<double> &out_frame_weights,
                       Eigen::MatrixXd *out_reweighted_tpm,
                       Eigen::VectorXd *out_reweighted_pi,
                       bool *out_converged,
                       TitanValidation::DhamRebuildContext
                           *out_validation_dham_context) {
  out_frame_weights.clear();
  if (out_reweighted_tpm)
    out_reweighted_tpm->resize(0, 0);
  if (out_reweighted_pi)
    out_reweighted_pi->resize(0);
  if (out_converged)
    *out_converged = false;
  if (out_validation_dham_context)
    *out_validation_dham_context = {};
  if (!args.dham.flag_dham || !msm_analyzer)
    return;

  auto start_dham = std::chrono::steady_clock::now();
  TitanUI::print_module_header("DHAM Post-Processing");
  const bool use_npt_observable = (args.dham.ensemble == "npt");
  if (args.lj.flag_lj) {
    TitanUI::log_kv(log, "DHAM Feature Domain", "LJ phase-2 feature space");
  } else {
    TitanUI::log_kv(log, "DHAM Feature Domain",
                    "protein network -> TICA -> MSM feature space");
  }
  TitanUI::log_kv(log, "DHAM Ensemble", args.dham.ensemble);
  if (use_npt_observable) {
    TitanUI::log_kv(log, "DHAM Thermodynamic Observable",
                    "enthalpy-like reweighting H = column1 + P*V(box)");
    TitanUI::log_kv(log, "DHAM Target Pressure (bar)",
                    std::to_string(args.dham.target_pressure_bar));
    TitanUI::print_info(
        "DHAM: NPT mode enabled. Using H = E_col1 + P*V with P = " +
        std::to_string(args.dham.target_pressure_bar) +
        " bar and V from trajectory box metadata.");
    TitanUI::print_warning(
        "DHAM NPT: for best thermodynamic consistency, the first numeric "
        "column in each .xvg should be configurational/potential energy, not "
        "a mixed observable.");
  } else {
    TitanUI::log_kv(log, "DHAM Thermodynamic Observable",
                    "energy-based NVT/REMD reweighting");
  }

  // 1. Read temperatures
  if (args.dham.temps_file.empty()) {
    TitanUI::print_error("DHAM: -dham-temps not set. Skipping DHAM.");
    return;
  }
  auto temps = TitanDHAM::read_temps(args.dham.temps_file);
  if (temps.empty()) {
    TitanUI::print_error("DHAM: No temperatures read. Skipping.");
    return;
  }

  // 2. Read energy file list
  const bool segment_audit = !args.dham.segment_map_file.empty();
  if (args.dham.ener_list_file.empty() && !segment_audit) {
    TitanUI::print_error("DHAM: -dham-ener not set. Skipping DHAM.");
    return;
  }
  auto ener_paths = segment_audit ? std::vector<std::string>{} : TitanDHAM::read_file_list(args.dham.ener_list_file);
  if (ener_paths.empty() && !segment_audit) {
    TitanUI::print_error("DHAM: No energy files read. Skipping.");
    return;
  }

  const int n_replicas = static_cast<int>(temps.size());
  if (!segment_audit && static_cast<int>(ener_paths.size()) != n_replicas) {
    TitanUI::print_error(
        "DHAM: #temps (" + std::to_string(n_replicas) +
        ") != #energy_files (" + std::to_string(ener_paths.size()) + ").");
    return;
  }

  // Validate n_replicas if explicitly set
  if (args.dham.n_replicas > 0 && args.dham.n_replicas != n_replicas) {
    TitanUI::print_warning(
        "DHAM: --dham-n-replicas=" + std::to_string(args.dham.n_replicas) +
        " != temps file count (" + std::to_string(n_replicas) +
        "). Using temps file count.");
  }

  TitanUI::print_info("DHAM: " + std::to_string(n_replicas) + " replicas, " +
                       "target T = " + std::to_string(args.dham.target_temp) + " K");
  // Fix C: Document the 1-file-per-replica assumption
  TitanUI::print_warning(
      "DHAM: Each input trajectory file is treated as one replica. "
      "Concatenation does not establish physical continuity; REMD inputs "
      "need audited segment/exchange metadata.");

  // 3. Extract per-replica data from MSM
  if (!msm_analyzer->is_clustered()) {
    TitanUI::print_error("DHAM: MSM K-means has not run. "
        "Ensure -msm flag is set and enough frames are collected.");
    return;
  }
  if (!segment_audit && !msm_analyzer->is_built()) {
    TitanUI::print_error(
        "DHAM: MSM transition model has not been built successfully.");
    return;
  }

  const int dham_lag = msm_analyzer->msm_lag_time();
  if (dham_lag <= 0) {
    TitanUI::print_error("DHAM: invalid MSM lag time. Aborting DHAM.");
    return;
  }

  auto per_replica_dtrajs     = msm_analyzer->get_per_replica_dtrajs();
  auto per_replica_timestamps = msm_analyzer->get_per_replica_timestamps();
  auto per_replica_box_volumes = use_npt_observable
      ? msm_analyzer->get_per_replica_box_volumes_nm3()
      : std::unordered_map<int, std::vector<double>>{};

  if (per_replica_dtrajs.empty()) {
    TitanUI::print_error("DHAM: No per-replica discrete trajectories. "
                          "Was replica_id passed to add_projected_frame?");
    return;
  }

  if (segment_audit) {
    // KIN-13: audited replica-exchange metadata. Samples are the frames with
    // an exact energy; transitions are the windows inside one segment with
    // no accepted exchange and exact energies at both ends.
    try {
      const auto metadata = titan_dham_segments::Metadata::read(
          args.dham.segment_map_file, n_replicas);
      const int n_states = msm_analyzer->n_clusters();
      TitanTram tram(n_states, temps, args.dham.target_temp);
      std::vector<std::pair<int, std::size_t>> sample_frames;
      std::vector<titan_dham_segments::CountAudit> audits;
      const double pv_factor =
          use_npt_observable ? args.dham.target_pressure_bar * kBarNm3ToKJPerMol
                             : 0.0;
      for (int k = 0; k < n_replicas; ++k) {
        const auto &rep = metadata.replicas[k];
        if (std::abs(rep.temperature - temps[k]) > 1e-6 ||
            (use_npt_observable && std::abs(rep.pressure - args.dham.target_pressure_bar) > 1e-6))
          throw std::invalid_argument("DHAM segment metadata ensemble mismatch");
        const auto &states = per_replica_dtrajs.at(k);
        const auto &times = per_replica_timestamps.at(k);
        if (use_npt_observable) {
          const auto &volumes = per_replica_box_volumes.at(k);
          if (volumes.size() != times.size())
            throw std::invalid_argument("DHAM segment volume dimensions differ");
          for (size_t i = 0; i < times.size(); ++i) {
            const double expected = metadata.match(k, times[i]).volume;
            if (!std::isfinite(volumes[i]) ||
                std::abs(volumes[i] - expected) > std::max(1e-4, 2e-6 * expected))
              throw std::invalid_argument("DHAM segment trajectory volume mismatch");
          }
        }
        audits.push_back(titan_dham_segments::audit_counts(
            metadata, k, states, times, dham_lag, n_states));
        for (size_t i = 0; i < times.size(); ++i) {
          const auto &frame = metadata.match(k, times[i]);
          if (!std::isfinite(frame.energy))
            continue;
          tram.add_sample(k, states[i], frame.energy + pv_factor * frame.volume);
          sample_frames.push_back({k, i});
        }
        for (const auto &[pair, n] : audits.back().counts)
          tram.add_transition(k, pair.first, pair.second, static_cast<double>(n));
      }
      std::ofstream counts("dham_segment_counts.csv"), decisions("dham_segment_decisions.csv");
      if (!counts || !decisions) throw std::runtime_error("Cannot write DHAM segment audit");
      counts << "replica,state_from,state_to,eligible_count\n";
      decisions << "replica,decision,windows\n";
      size_t eligible = 0, windows = 0;
      for (int k = 0; k < n_replicas; ++k) {
        for (const auto &[pair, n] : audits[k].counts)
          counts << k << ',' << pair.first << ',' << pair.second << ',' << n << '\n';
        for (const auto &[decision, n] : audits[k].decisions) {
          decisions << k << ',' << titan_dham_segments::name(decision) << ',' << n << '\n';
          windows += n;
          if (decision == titan_dham_segments::Decision::eligible) eligible += n;
        }
      }
      counts.flush();decisions.flush();
      if (!counts || !decisions) throw std::runtime_error("Incomplete DHAM segment audit output");
      TitanUI::log_kv(log, "DHAM Segment Eligible Windows",
                      std::to_string(eligible) + "/" + std::to_string(windows));
      TitanUI::print_info("DHAM: " + std::to_string(eligible) + " of " +
                          std::to_string(windows) +
                          " lag windows lie inside one segment without an "
                          "accepted exchange and with exact energies at both "
                          "ends; they are the TRAM transition counts.");
      TitanUI::print_warning(
          "DHAM: dropping windows that contain an exchange is a selection TRAM "
          "does not correct; rates need lag times well below the time between "
          "accepted exchanges. Bootstrap/CK validation is not run in segment mode.");
      const bool converged = finish_tram(
          args, tram, temps, sample_frames, per_replica_dtrajs, {},
          *msm_analyzer, executor, info_output, log, "segment_map",
          out_frame_weights, out_reweighted_tpm, out_reweighted_pi);
      TitanUI::log_kv(log, "DHAM Converged", converged ? "yes" : "no");
      if (out_converged)
        *out_converged = converged;
    } catch (const std::exception &error) {
      TitanUI::print_error(std::string("DHAM segment input rejected: ") + error.what());
      out_frame_weights.clear();
      if (out_reweighted_tpm)
        out_reweighted_tpm->resize(0, 0);
      if (out_reweighted_pi)
        out_reweighted_pi->resize(0);
    }
    return;
  }

  // 4. Parse + align energy .xvg files with optional per-replica
  // window matching against the common trajectory/energy overlap.
  std::unordered_map<int, std::vector<double>> per_replica_energies;
  std::unordered_map<int, std::vector<std::pair<double, double>>> parsed_xvg;
  std::mutex parsed_xvg_mutex;
  const int n_io_threads =
      std::min(static_cast<int>(executor.capacity()), n_replicas);
  const int chunk = (n_replicas + n_io_threads - 1) / n_io_threads;
  executor.parallel_do(static_cast<size_t>(n_io_threads), [&](size_t t_sz) {
    const int t = static_cast<int>(t_sz);
    const int ks = t * chunk;
    const int ke = std::min(ks + chunk, n_replicas);
    if (ks >= ke)
      return;
    for (int k = ks; k < ke; ++k) {
      auto xvg_data = TitanDHAM::parse_xvg(ener_paths[static_cast<size_t>(k)]);
      if (!xvg_data.empty())
        xvg_data = filter_postproc_energy_pairs(xvg_data, args);
      std::lock_guard<std::mutex> lock(parsed_xvg_mutex);
      parsed_xvg[k] = std::move(xvg_data);
    }
  });
  TitanUI::log_kv(log, "DHAM Window Matching", args.dham.window_match);
  if (args.dham.dt_energy_ps > 0.0) {
    TitanUI::print_info("DHAM: Using user-specified dt_energy = " +
                        std::to_string(args.dham.dt_energy_ps) + " ps.");
  } else {
    TitanUI::print_info(
        "DHAM: Auto-detecting dt_energy independently for each energy file.");
  }
  if (args.dham.window_match == "off") {
    TitanUI::print_info(
        "DHAM: Window matching disabled. TITAN will align against the full "
        "trajectory time axis.");
  } else if (args.dham.window_match == "strict") {
    TitanUI::print_info(
        "DHAM: Strict window matching enabled. TITAN will abort if any "
        "replica has a trajectory/energy mismatch.");
  } else {
    TitanUI::print_info(
        "DHAM: Auto window matching enabled. TITAN will trim each replica to "
        "the common trajectory/energy overlap before DHAM.");
  }

  auto subset_by_indices = [](const auto &src,
                              const std::vector<std::size_t> &indices) {
    using value_type = typename std::decay_t<decltype(src)>::value_type;
    std::vector<value_type> out;
    out.reserve(indices.size());
    for (std::size_t idx : indices) {
      if (idx < src.size())
        out.push_back(src[idx]);
    }
    return out;
  };
  auto format_ps = [](double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << value;
    return oss.str();
  };

  std::size_t window_total_original_frames = 0;
  std::size_t window_total_kept_frames = 0;
  std::size_t window_total_dropped_frames = 0;
  std::size_t window_total_unmatched_frames = 0;
  std::size_t window_trimmed_replicas = 0;
  std::size_t window_replicas_without_overlap = 0;
  std::size_t window_missing_energy_files = 0;
  std::size_t window_truncated_start = 0;
  std::size_t window_truncated_end = 0;
  std::size_t window_irregular_dt = 0;
  std::size_t window_sparse_energy = 0;
  std::size_t window_energy_coarser = 0;         // KIN-21
  std::size_t window_frames_without_own_row = 0; // KIN-21
  bool strict_window_violation = false;
  std::vector<DhamWindowAuditRow> window_audit_rows;
  window_audit_rows.reserve(static_cast<std::size_t>(n_replicas));
  // KIN-16: for each window-matched replica, the position of every kept frame
  // among that replica's projected frames; the frame weights are mapped back
  // through it (a replica without an entry kept all of its frames).
  std::unordered_map<int, std::vector<std::size_t>> per_replica_kept_positions;

  for (int k = 0; k < n_replicas; ++k) {
    auto it_xvg = parsed_xvg.find(k);
    auto it_dtraj = per_replica_dtrajs.find(k);
    auto it_ts = per_replica_timestamps.find(k);
    DhamWindowAuditRow audit;
    audit.replica_id = k;
    audit.traj_present =
        (it_dtraj != per_replica_dtrajs.end() && it_ts != per_replica_timestamps.end());
    audit.energy_present =
        (it_xvg != parsed_xvg.end() && !it_xvg->second.empty());
    if (audit.traj_present) {
      audit.traj_frames = it_dtraj->second.size();
      if (!it_ts->second.empty()) {
        audit.traj_min_ps = it_ts->second.front();
        audit.traj_max_ps = it_ts->second.back();
      }
    }
    if (it_xvg != parsed_xvg.end()) {
      audit.energy_points = it_xvg->second.size();
      if (!it_xvg->second.empty()) {
        audit.energy_min_ps = it_xvg->second.front().first;
        audit.energy_max_ps = it_xvg->second.back().first;
      }
    }
    if (it_xvg == parsed_xvg.end() || it_xvg->second.empty() ||
        it_dtraj == per_replica_dtrajs.end() ||
        it_ts == per_replica_timestamps.end()) {
      if (!audit.energy_present)
        ++window_missing_energy_files;
      audit.status = !audit.traj_present ? "missing_trajectory" :
                     (!audit.energy_present ? "missing_or_empty_energy" :
                                              "incomplete_inputs");
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    const double dt_energy_ps =
        (args.dham.dt_energy_ps > 0.0) ? args.dham.dt_energy_ps : 0.0;
    if (args.dham.window_match == "off") {
      const auto matched = TitanDHAM::window_match_energy_to_dtraj(
          it_ts->second, it_xvg->second, dt_energy_ps);
      audit.inferred_dt_energy_ps = matched.inferred_dt_energy_ps;
      audit.overlap_found = matched.overlap_found;
      audit.overlap_start_ps = matched.overlap_start_ps;
      audit.overlap_end_ps = matched.overlap_end_ps;
      audit.kept_frames = matched.kept_frames;
      audit.dropped_outside_overlap = matched.dropped_outside_overlap;
      audit.unmatched_in_overlap = matched.unmatched_in_overlap;
      audit.energy_coarser_than_traj = matched.energy_coarser_than_traj;
      audit.duplicate_energy_matches = matched.duplicate_energy_matches;
      audit.inferred_dt_traj_ps = matched.inferred_dt_traj_ps;
      if (audit.energy_coarser_than_traj) {
        ++window_energy_coarser;
        window_frames_without_own_row += matched.duplicate_energy_matches;
      }
      audit.sparse_energy = (audit.energy_points < 2);
      audit.irregular_energy_dt =
          detect_irregular_energy_dt(it_xvg->second, matched.inferred_dt_energy_ps);
      per_replica_energies[k] = TitanDHAM::align_energy_to_dtraj(
          it_ts->second, it_xvg->second, dt_energy_ps);
      audit.status = "legacy_off";
      if (audit.irregular_energy_dt)
        ++window_irregular_dt;
      if (audit.sparse_energy)
        ++window_sparse_energy;
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    const auto matched = TitanDHAM::window_match_energy_to_dtraj(
        it_ts->second, it_xvg->second, dt_energy_ps);
    audit.inferred_dt_energy_ps = matched.inferred_dt_energy_ps;
    audit.overlap_found = matched.overlap_found;
    audit.overlap_start_ps = matched.overlap_start_ps;
    audit.overlap_end_ps = matched.overlap_end_ps;
    audit.kept_frames = matched.kept_frames;
    audit.dropped_outside_overlap = matched.dropped_outside_overlap;
    audit.unmatched_in_overlap = matched.unmatched_in_overlap;
    audit.energy_coarser_than_traj = matched.energy_coarser_than_traj;
    audit.duplicate_energy_matches = matched.duplicate_energy_matches;
    audit.inferred_dt_traj_ps = matched.inferred_dt_traj_ps;
    if (audit.energy_coarser_than_traj) {
      ++window_energy_coarser;
      window_frames_without_own_row += matched.duplicate_energy_matches;
    }
    const double coverage_tol =
        std::max(1e-6, 2.0 * std::max(1e-6, matched.inferred_dt_energy_ps));
    audit.truncated_start =
        std::isfinite(matched.traj_min_ps) && std::isfinite(matched.energy_min_ps) &&
        matched.energy_min_ps > matched.traj_min_ps + coverage_tol;
    audit.truncated_end =
        std::isfinite(matched.traj_max_ps) && std::isfinite(matched.energy_max_ps) &&
        matched.energy_max_ps < matched.traj_max_ps - coverage_tol;
    audit.sparse_energy = (audit.energy_points < 2);
    audit.irregular_energy_dt =
        detect_irregular_energy_dt(it_xvg->second, matched.inferred_dt_energy_ps);
    window_total_original_frames += matched.original_frames;
    window_total_kept_frames += matched.kept_frames;
    window_total_dropped_frames += matched.dropped_outside_overlap;
    window_total_unmatched_frames += matched.unmatched_in_overlap;
    if (audit.truncated_start)
      ++window_truncated_start;
    if (audit.truncated_end)
      ++window_truncated_end;
    if (audit.irregular_energy_dt)
      ++window_irregular_dt;
    if (audit.sparse_energy)
      ++window_sparse_energy;

    const bool no_usable_overlap =
        !matched.overlap_found || matched.kept_frames == 0;
    const bool has_window_loss =
        matched.dropped_outside_overlap > 0 || matched.unmatched_in_overlap > 0;

    if (no_usable_overlap) {
      ++window_replicas_without_overlap;
      const std::string msg =
          "DHAM: Replica " + std::to_string(k) +
          " has no usable overlap between trajectory [" +
          format_ps(matched.traj_min_ps) + ", " + format_ps(matched.traj_max_ps) +
          "] ps and energy [" + format_ps(matched.energy_min_ps) + ", " +
          format_ps(matched.energy_max_ps) + "] ps.";
      if (args.dham.window_match == "strict") {
        TitanUI::print_warning(msg);
        strict_window_violation = true;
      } else {
        TitanUI::print_warning(msg + " Replica will be ignored by DHAM.");
      }
      audit.status = "no_overlap";
      window_audit_rows.push_back(std::move(audit));
      per_replica_dtrajs.erase(k);
      per_replica_timestamps.erase(k);
      per_replica_energies.erase(k);
      if (use_npt_observable)
        per_replica_box_volumes.erase(k);
      continue;
    }

    if (matched.dropped_outside_overlap > 0)
      ++window_trimmed_replicas;
    if (args.dham.window_match == "strict" && has_window_loss) {
      TitanUI::print_warning(
          "DHAM: Replica " + std::to_string(k) +
          " violated strict window matching. kept=" +
          std::to_string(matched.kept_frames) + "/" +
          std::to_string(matched.original_frames) + ", dropped_outside=" +
          std::to_string(matched.dropped_outside_overlap) +
          ", unmatched_inside=" +
          std::to_string(matched.unmatched_in_overlap) + ".");
      strict_window_violation = true;
      audit.status = "strict_rejected";
      window_audit_rows.push_back(std::move(audit));
      continue;
    }

    if (has_window_loss) {
      TitanUI::print_info(
          "DHAM: Replica " + std::to_string(k) + " window-matched to [" +
          format_ps(matched.overlap_start_ps) + ", " +
          format_ps(matched.overlap_end_ps) + "] ps; kept " +
          std::to_string(matched.kept_frames) + "/" +
          std::to_string(matched.original_frames) + " frames, dropped " +
          std::to_string(matched.dropped_outside_overlap) + ", unmatched " +
          std::to_string(matched.unmatched_in_overlap) + ".");
    }

    if (audit.truncated_start && audit.truncated_end) {
      audit.status = "truncated_both";
    } else if (audit.truncated_start) {
      audit.status = "truncated_start";
    } else if (audit.truncated_end) {
      audit.status = "truncated_end";
    } else if (audit.irregular_energy_dt) {
      audit.status = "irregular_dt";
    } else if (audit.sparse_energy) {
      audit.status = "sparse_energy";
    } else if (has_window_loss) {
      audit.status = "trimmed_to_overlap";
    } else if (audit.energy_coarser_than_traj) {
      audit.status = "energy_coarser_than_traj";
    } else {
      audit.status = "ok";
    }

    {
      std::vector<std::size_t> kept_positions;
      kept_positions.reserve(matched.kept_indices.size());
      for (std::size_t idx : matched.kept_indices) {
        if (idx < it_dtraj->second.size())
          kept_positions.push_back(idx);
      }
      per_replica_kept_positions[k] = std::move(kept_positions);
    }
    per_replica_dtrajs[k] =
        subset_by_indices(it_dtraj->second, matched.kept_indices);
    per_replica_timestamps[k] = matched.kept_dtraj_times_ps;
    per_replica_energies[k] = matched.aligned_values;
    if (use_npt_observable) {
      auto it_volume = per_replica_box_volumes.find(k);
      if (it_volume != per_replica_box_volumes.end()) {
        per_replica_box_volumes[k] =
            subset_by_indices(it_volume->second, matched.kept_indices);
      }
    }
    window_audit_rows.push_back(std::move(audit));
  }

  if (args.dham.window_match != "off") {
    TitanUI::log_kv(log, "DHAM Window Frames Kept",
                    std::to_string(window_total_kept_frames) + "/" +
                        std::to_string(window_total_original_frames));
    TitanUI::log_kv(log, "DHAM Window Frames Dropped",
                    std::to_string(window_total_dropped_frames));
    TitanUI::log_kv(log, "DHAM Window Frames Unmatched",
                    std::to_string(window_total_unmatched_frames));
    TitanUI::log_kv(log, "DHAM Window Replicas Trimmed",
                    std::to_string(window_trimmed_replicas));
    TitanUI::log_kv(log, "DHAM Window Replicas No Overlap",
                    std::to_string(window_replicas_without_overlap));
    if (window_total_dropped_frames > 0 || window_total_unmatched_frames > 0) {
      TitanUI::print_info(
        "DHAM: Window matching kept " +
          std::to_string(window_total_kept_frames) + "/" +
          std::to_string(window_total_original_frames) +
          " trajectory frames across replicas.");
    }
  }
  TitanUI::log_kv(log, "DHAM Window Energy Missing",
                  std::to_string(window_missing_energy_files));
  TitanUI::log_kv(log, "DHAM Window Truncated Start",
                  std::to_string(window_truncated_start));
  TitanUI::log_kv(log, "DHAM Window Truncated End",
                  std::to_string(window_truncated_end));
  TitanUI::log_kv(log, "DHAM Window Irregular dt",
                  std::to_string(window_irregular_dt));
  TitanUI::log_kv(log, "DHAM Window Sparse Energy",
                  std::to_string(window_sparse_energy));
  TitanUI::log_kv(log, "DHAM Window Energy Coarser Than Traj",
                  std::to_string(window_energy_coarser));
  if (window_energy_coarser > 0) {
    TitanUI::print_warning(
        "DHAM: the energy grid is coarser than the trajectory grid for " +
        std::to_string(window_energy_coarser) +
        " replica(s). One frame per energy row carries the energy; " +
        std::to_string(window_frames_without_own_row) +
        " frames between rows are kept for the transition counts but not "
        "used in the state energies.");
  }
  write_dham_window_audit_file(info_output.get_dham_window_audit_filename(),
                               window_audit_rows);
  TitanUI::print_info("DHAM: Exported window audit to " +
                      info_output.get_dham_window_audit_filename());
  if (strict_window_violation) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "strict window matching rejected one or more replicas");
    TitanUI::print_warning(
        "DHAM: strict window matching failed. Skipping DHAM reweighting.");
    return;
  }
  std::size_t post_match_candidate_transitions = 0;
  for (const auto &[replica_id, dtraj] : per_replica_dtrajs) {
    (void)replica_id;
    if (dtraj.size() > static_cast<std::size_t>(dham_lag)) {
      post_match_candidate_transitions +=
          dtraj.size() - static_cast<std::size_t>(dham_lag);
    }
  }
  TitanUI::log_kv(log, "DHAM Window Candidate Transitions",
                  std::to_string(post_match_candidate_transitions));
  if (post_match_candidate_transitions == 0) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "zero lag-valid transitions after window matching");
    TitanUI::print_warning(
        "DHAM: Window-matched data contain no lag-valid transitions for the "
        "current MSM lag. Merge matching trajectory/energy windows, reduce "
        "-msm-lag, or disable automatic trimming with -dham-window-match off.");
    return;
  }

  std::unordered_map<int, std::vector<double>> per_replica_observables;
  if (use_npt_observable) {
    const double pv_factor =
        args.dham.target_pressure_bar * kBarNm3ToKJPerMol;
    size_t replicas_missing_volume = 0;
    size_t volume_size_mismatches = 0;
    size_t frames_missing_valid_volume = 0;
    size_t frames_combined = 0;

    for (int k = 0; k < n_replicas; ++k) {
      auto it_energy = per_replica_energies.find(k);
      if (it_energy == per_replica_energies.end())
        continue;

      auto it_volume = per_replica_box_volumes.find(k);
      if (it_volume == per_replica_box_volumes.end()) {
        ++replicas_missing_volume;
        continue;
      }

      const auto &energies = it_energy->second;
      const auto &volumes_nm3 = it_volume->second;
      if (volumes_nm3.size() != energies.size()) {
        ++volume_size_mismatches;
      }

      std::vector<double> observable(
          energies.size(), std::numeric_limits<double>::quiet_NaN());
      const size_t n_frames = std::min(energies.size(), volumes_nm3.size());
      for (size_t t = 0; t < n_frames; ++t) {
        const double e = energies[t];
        const double v = volumes_nm3[t];
        if (!std::isfinite(e) || !std::isfinite(v) || v < 0.0) {
          ++frames_missing_valid_volume;
          continue;
        }
        observable[t] = e + pv_factor * v;
        ++frames_combined;
      }
      per_replica_observables[k] = std::move(observable);
    }

    if (replicas_missing_volume > 0) {
      TitanUI::print_error(
          "DHAM NPT: Missing per-frame box volumes for " +
          std::to_string(replicas_missing_volume) + "/" +
          std::to_string(n_replicas) +
          " replicas. Re-run with the current TITAN build so TICA/MSM caches "
          "trajectory volume metadata.");
      return;
    }
    if (frames_combined == 0) {
      TitanUI::print_error(
          "DHAM NPT: No valid E(t) + P*V(t) samples were constructed from the "
          "aligned energy traces and trajectory box metadata.");
      return;
    }
    if (volume_size_mismatches > 0) {
      TitanUI::print_warning(
          "DHAM NPT: " + std::to_string(volume_size_mismatches) +
          " replica volume series had a different length than the aligned "
          "energy series; truncation to the common prefix was applied.");
    }
    if (frames_missing_valid_volume > 0) {
      TitanUI::print_warning(
          "DHAM NPT: " + std::to_string(frames_missing_valid_volume) +
          " aligned frames had invalid energy/volume data and were excluded "
          "from the enthalpy-like observable.");
    }
    TitanUI::print_info(
        "DHAM NPT: Constructed " + std::to_string(frames_combined) +
        " frame-level H = E_col1 + P*V samples.");
  } else {
    per_replica_observables = per_replica_energies;
  }
  if (per_replica_observables.empty()) {
    TitanUI::log_kv(log, "DHAM Converged", "no");
    TitanUI::log_kv(log, "DHAM Abort Reason",
                    "no usable aligned observables remained after window matching");
    TitanUI::print_warning(
        "DHAM: No usable aligned thermodynamic observables remain after "
        "window matching.");
    return;
  }

  if (out_validation_dham_context) {
    out_validation_dham_context->available = true;
    out_validation_dham_context->n_states = msm_analyzer->n_clusters();
    out_validation_dham_context->lag_frames = dham_lag;
    out_validation_dham_context->n_replicas = n_replicas;
    out_validation_dham_context->temps_k = temps;
    out_validation_dham_context->target_temp_k = args.dham.target_temp;
    out_validation_dham_context->max_iter = args.dham.max_iter;
    out_validation_dham_context->tolerance = args.dham.tolerance;
    out_validation_dham_context->ensemble = args.dham.ensemble;
    out_validation_dham_context->per_replica_dtrajs = per_replica_dtrajs;
    out_validation_dham_context->per_replica_timestamps_ps =
        per_replica_timestamps;
    out_validation_dham_context->per_replica_observables =
        per_replica_observables;
  }

  // 5. TRAM over the replica temperatures (KIN-13). Every frame with an
  // aligned observable is a sample; transitions at the MSM lag inside one
  // replica, between two samples, not across a trajectory break.
  const int n_states = msm_analyzer->n_clusters();
  TitanTram tram(n_states, temps, args.dham.target_temp);
  std::vector<std::pair<int, std::size_t>> sample_frames;
  size_t frames_without_observable = 0;
  for (int k = 0; k < n_replicas; ++k) {
    auto it = per_replica_dtrajs.find(k);
    auto it_obs = per_replica_observables.find(k);
    if (it == per_replica_dtrajs.end() || it_obs == per_replica_observables.end())
      continue;
    const auto &dtraj = it->second;
    const auto &obs = it_obs->second;
    for (size_t t = 0; t < dtraj.size(); ++t) {
      const int state = dtraj[t];
      if (state < 0 || state >= n_states)
        continue;
      if (t >= obs.size() || !std::isfinite(obs[t])) {
        ++frames_without_observable;
        continue;
      }
      tram.add_sample(k, state, obs[t]);
      sample_frames.push_back({k, t});
    }
  }
  if (frames_without_observable > 0) {
    TitanUI::print_warning(
        "DHAM: " + std::to_string(frames_without_observable) +
        " frames have no aligned thermodynamic observable; they are not TRAM "
        "samples and transitions touching them are not counted. Match the "
        "trajectory output to the energy output (-dt) to use every frame.");
  }

  TitanUI::print_info("DHAM: Using MSM lag = " + std::to_string(dham_lag) +
                        " frames for count matrix construction.");
  size_t total_counts = 0;
  size_t skipped_breaks = 0;
  size_t skipped_missing_energy = 0;
  size_t skipped_invalid_states = 0;
  for (int k = 0; k < n_replicas; ++k) {
    auto it = per_replica_dtrajs.find(k);
    if (it == per_replica_dtrajs.end()) continue;
    const auto &dtraj = it->second;
    auto it_obs = per_replica_observables.find(k);
    auto it_ts = per_replica_timestamps.find(k);
    const bool have_times = (it_ts != per_replica_timestamps.end() &&
                              it_ts->second.size() == dtraj.size());
    // Expected gap = lag * dt_frame. KIN-19: dt_frame is the median positive
    // spacing of the replica (as for the energy grid in window matching),
    // not ts[1] - ts[0]: a repeated first frame (two concatenated parts that
    // both hold t = 0) gave dt = 0 and turned every transition into a break.
    double dt_frame = 1.0, expected_gap = dham_lag;
    if (have_times && it_ts->second.size() > 1) {
      dt_frame = TitanDHAM::median_positive_spacing(it_ts->second);
      if (!(dt_frame > 0.0)) {
        TitanUI::print_warning(
            "DHAM: replica " + std::to_string(k) +
            " has no increasing timestamps; all its transitions are treated "
            "as breaks.");
        dt_frame = std::numeric_limits<double>::quiet_NaN();
      }
      expected_gap = dham_lag * dt_frame;
    }
    for (size_t t = 0; t + static_cast<size_t>(dham_lag) < dtraj.size(); ++t) {
      if (have_times && !std::isfinite(expected_gap)) {
        ++skipped_breaks;
        continue;
      }
      if (have_times) {
        double actual_gap = it_ts->second[t + static_cast<size_t>(dham_lag)]
                          - it_ts->second[t];
        if (std::abs(actual_gap - expected_gap) > 0.1 * std::abs(expected_gap)) {
          ++skipped_breaks;
          continue; // trajectory break — skip phantom transition
        }
      }
      const int state_i = dtraj[t];
      const int state_j = dtraj[t + static_cast<size_t>(dham_lag)];
      if (state_i < 0 || state_i >= n_states || state_j < 0 || state_j >= n_states) {
        ++skipped_invalid_states;
        continue;
      }
      const size_t t_end = t + static_cast<size_t>(dham_lag);
      if (it_obs == per_replica_observables.end() || t_end >= it_obs->second.size() ||
          !std::isfinite(it_obs->second[t]) || !std::isfinite(it_obs->second[t_end])) {
        ++skipped_missing_energy;
        continue;
      }
      tram.add_transition(k, state_i, state_j);
      ++total_counts;
    }
  }
  if (skipped_breaks > 0) {
    TitanUI::print_warning("DHAM: Skipped " + std::to_string(skipped_breaks) +
        " transitions across trajectory breaks (kept " +
        std::to_string(total_counts) + ").");
  }
  if (skipped_missing_energy > 0) {
    TitanUI::print_warning("DHAM: Skipped " +
        std::to_string(skipped_missing_energy) +
        " transitions with an end frame without aligned thermodynamic observable.");
  }
  if (skipped_invalid_states > 0) {
    TitanUI::print_warning("DHAM: Skipped " +
        std::to_string(skipped_invalid_states) +
        " transitions with invalid microstate indices.");
  }

  const bool converged = finish_tram(
      args, tram, temps, sample_frames, per_replica_dtrajs,
      per_replica_kept_positions, *msm_analyzer, executor, info_output, log,
      "replica_trajectories", out_frame_weights, out_reweighted_tpm,
      out_reweighted_pi);
  TitanUI::log_kv(log, "DHAM Converged", converged ? "yes" : "no");
  if (out_converged)
    *out_converged = converged;
  if (!converged) {
    out_frame_weights.clear();
    if (out_reweighted_tpm)
      out_reweighted_tpm->resize(0, 0);
    if (out_reweighted_pi)
      out_reweighted_pi->resize(0);
    TitanUI::print_warning(
        "DHAM: TRAM did not converge - PCA/TICA FES will use uniform weights. "
        "Increase -dham-max-iter or check the aligned energy data.");
  }

  auto time_dham = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_dham);
  TitanUI::log_time(log, "DHAM Analysis", time_dham.count() / 1000.0);
}

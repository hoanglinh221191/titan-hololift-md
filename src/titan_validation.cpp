#include "titan_validation.h"

#include "titan_ui.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace TitanValidation {

namespace {

std::vector<int> resolve_candidate_axis(const std::vector<int> &requested,
                                        int fallback_value, int max_value = 0) {
  std::vector<int> values = requested;
  if (values.empty())
    values.push_back(fallback_value);
  std::vector<int> normalized;
  normalized.reserve(values.size());
  for (int value : values) {
    if (value <= 0)
      continue;
    if (max_value > 0)
      value = std::min(value, max_value);
    if (value <= 0)
      continue;
    if (std::find(normalized.begin(), normalized.end(), value) ==
        normalized.end()) {
      normalized.push_back(value);
    }
  }
  if (normalized.empty() && fallback_value > 0)
    normalized.push_back(max_value > 0 ? std::min(fallback_value, max_value)
                                       : fallback_value);
  return normalized;
}

std::vector<ModelConfig>
build_candidate_grid(const arguments &args, const TitanKinetics &tica_analyzer,
                     const TitanMSM &msm_analyzer, int current_effective_k) {
  const int current_components =
      std::max(1, std::min(tica_analyzer.retained_components(),
                           msm_analyzer.n_components()));
  const int current_lag = std::max(1, args.msm.msm_lag);
  const int current_clusters = std::max(2, current_effective_k);

  const auto components = resolve_candidate_axis(
      args.validation.vamp_components, current_components,
      std::max(1, tica_analyzer.retained_components()));
  const auto lags =
      resolve_candidate_axis(args.validation.vamp_lags, current_lag);
  const auto clusters = resolve_candidate_axis(
      args.validation.vamp_clusters, current_clusters,
      std::max(2, static_cast<int>(msm_analyzer.n_frames())));

  std::vector<ModelConfig> grid;
  for (int component : components) {
    for (int lag : lags) {
      for (int cluster : clusters) {
        if (cluster <= 1)
          continue;
        grid.push_back({component, lag, cluster});
      }
    }
  }
  return grid;
}

bool same_config(const ModelConfig &lhs, const ModelConfig &rhs) {
  return lhs.components == rhs.components && lhs.lag_frames == rhs.lag_frames &&
         lhs.n_clusters == rhs.n_clusters;
}

std::pair<double, double> mean_std(const std::vector<double> &values) {
  std::pair<double, double> out{std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN()};
  if (values.empty())
    return out;
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double var = 0.0;
  for (double value : values)
    var += (value - mean) * (value - mean);
  var /= std::max<std::size_t>(1, values.size() - 1);
  out.first = mean;
  out.second = std::sqrt(std::max(0.0, var));
  return out;
}

double percentile_sorted(std::vector<double> values, double q) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](double value) {
                                return !std::isfinite(value);
                              }),
               values.end());
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double pos =
      std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  if (lo == hi)
    return values[lo];
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

void write_vamp_ranking_csv(const std::string &filename,
                            const VampSweepResult &sweep) {
  std::ofstream out(filename);
  out << "Components,LagFrames,Clusters,FoldsRequested,FoldsEffective,"
         "FoldsScored,ScoreMean,ScoreStd,Status\n";
  for (const auto &candidate : sweep.candidates) {
    out << candidate.config.components << ',' << candidate.config.lag_frames
        << ',' << candidate.config.n_clusters << ','
        << candidate.folds_requested << ',' << candidate.folds_effective << ','
        << candidate.folds_scored << ',' << candidate.score_mean << ','
        << candidate.score_std << ',' << candidate.status << '\n';
  }
}

void write_selected_model_file(const std::string &filename,
                               const RuntimeCache &cache) {
  std::ofstream out(filename);
  out << "# TITAN validation selected model\n";
  out << "current_components " << cache.current_effective_model.components << "\n";
  out << "current_lag_frames " << cache.current_effective_model.lag_frames << "\n";
  out << "current_clusters " << cache.current_effective_model.n_clusters << "\n";
  out << "selected_components " << cache.vamp_sweep.selected.components << "\n";
  out << "selected_lag_frames " << cache.vamp_sweep.selected.lag_frames << "\n";
  out << "selected_clusters " << cache.vamp_sweep.selected.n_clusters << "\n";
  out << "auto_pick_used " << (cache.vamp_sweep.used_auto_pick ? "yes" : "no")
      << "\n";
  out << "auto_pick_applied " << (cache.auto_pick_applied ? "yes" : "no")
      << "\n";
  out << "selected_is_current "
      << (cache.vamp_sweep.selected_is_current ? "yes" : "no") << "\n";
  if (!cache.vamp_sweep.warning_message.empty())
    out << "warning " << cache.vamp_sweep.warning_message << "\n";
}

void write_ck_files(const std::string &summary_filename,
                    const std::string &states_filename,
                    const std::vector<ChapmanKolmogorovSummary> &summaries) {
  std::ofstream summary_out(summary_filename);
  summary_out << "# model_source lag_frames multiple direct_transitions "
                 "rows_compared weighted_frobenius weighted_MAE max_abs\n";
  for (const auto &summary : summaries) {
    for (const auto &metric : summary.metrics) {
      summary_out << summary.model_source << ' ' << summary.lag_frames << ' '
                  << metric.multiple << ' ' << metric.direct_transition_count
                  << ' ' << metric.rows_compared << ' '
                  << metric.weighted_frobenius << ' ' << metric.weighted_mae
                  << ' ' << metric.max_abs << '\n';
    }
  }

  std::ofstream states_out(states_filename);
  states_out << "# model_source multiple state row_weight row_mae "
                "row_frobenius row_max_abs\n";
  for (const auto &summary : summaries) {
    for (const auto &metric : summary.metrics) {
      for (const auto &row : metric.row_errors) {
        states_out << summary.model_source << ' ' << row.multiple << ' '
                   << row.state << ' ' << row.row_weight << ' ' << row.row_mae
                   << ' ' << row.row_frobenius << ' ' << row.row_max_abs
                   << '\n';
      }
    }
  }
}

void write_msm_bootstrap_files(const std::string &summary_filename,
                               const std::string &states_filename,
                               const MsmBootstrapAggregate &bootstrap,
                               const Eigen::VectorXd &reference_pi,
                               const std::vector<double> &reference_its,
                               const std::vector<int> &ck_multiples) {
  std::ofstream summary_out(summary_filename);
  summary_out << "# TITAN validation MSM bootstrap summary\n";
  summary_out << "requested_samples " << bootstrap.requested_samples << "\n";
  summary_out << "successful_samples " << bootstrap.successful_samples << "\n";
  summary_out << "# metric index reference mean std p2.5 p97.5\n";
  for (std::size_t idx = 0; idx < bootstrap.implied_timescale_samples.size();
       ++idx) {
    const auto stats = mean_std(bootstrap.implied_timescale_samples[idx]);
    const double ref =
        idx < reference_its.size() ? reference_its[idx]
                                   : std::numeric_limits<double>::quiet_NaN();
    summary_out << "implied_timescale_ns " << idx << ' ' << ref << ' '
                << stats.first << ' ' << stats.second << ' '
                << percentile_sorted(bootstrap.implied_timescale_samples[idx], 0.025)
                << ' '
                << percentile_sorted(bootstrap.implied_timescale_samples[idx], 0.975)
                << "\n";
  }
  for (std::size_t idx = 0; idx < ck_multiples.size(); ++idx) {
    const auto frob_stats = mean_std(bootstrap.ck_weighted_frobenius_samples[idx]);
    const auto mae_stats = mean_std(bootstrap.ck_weighted_mae_samples[idx]);
    summary_out << "ck_weighted_frobenius " << ck_multiples[idx] << ' '
                << std::numeric_limits<double>::quiet_NaN() << ' '
                << frob_stats.first << ' ' << frob_stats.second << ' '
                << percentile_sorted(bootstrap.ck_weighted_frobenius_samples[idx],
                                     0.025)
                << ' '
                << percentile_sorted(bootstrap.ck_weighted_frobenius_samples[idx],
                                     0.975)
                << "\n";
    summary_out << "ck_weighted_mae " << ck_multiples[idx] << ' '
                << std::numeric_limits<double>::quiet_NaN() << ' '
                << mae_stats.first << ' ' << mae_stats.second << ' '
                << percentile_sorted(bootstrap.ck_weighted_mae_samples[idx], 0.025)
                << ' '
                << percentile_sorted(bootstrap.ck_weighted_mae_samples[idx], 0.975)
                << "\n";
  }

  std::ofstream states_out(states_filename);
  states_out << "# state pi_ref pi_mean pi_std\n";
  const double n =
      static_cast<double>(std::max(1, bootstrap.successful_samples));
  for (int state = 0; state < reference_pi.size(); ++state) {
    const double mean = bootstrap.stationary_sum[state] / n;
    const double var = std::max(
        0.0, bootstrap.stationary_sq_sum[state] / n - mean * mean);
    states_out << state << ' ' << reference_pi[state] << ' ' << mean << ' '
               << std::sqrt(var) << "\n";
  }
}

void write_tpt_bootstrap_files(const std::string &summary_filename,
                               const std::string &states_filename,
                               const TptBootstrapAggregate &bootstrap,
                               const TitanTPT &reference_tpt,
                               const std::string &model_source) {
  std::ofstream out(summary_filename);
  out << "# TITAN TPT bootstrap summary\n";
  out << "model_source " << model_source << "\n";
  out << "requested_samples " << bootstrap.requested_samples << "\n";
  out << "successful_samples " << bootstrap.successful_samples << "\n";
  out << "# metric mean std p2.5 p97.5\n";

  const auto flux_stats = mean_std(bootstrap.total_reactive_flux);
  out << "total_reactive_flux " << flux_stats.first << ' ' << flux_stats.second
      << ' ' << percentile_sorted(bootstrap.total_reactive_flux, 0.025) << ' '
      << percentile_sorted(bootstrap.total_reactive_flux, 0.975) << "\n";
  const auto rate_stats = mean_std(bootstrap.rate_source_to_sink_per_lag);
  out << "rate_source_to_sink_per_lag " << rate_stats.first << ' '
      << rate_stats.second << ' '
      << percentile_sorted(bootstrap.rate_source_to_sink_per_lag, 0.025) << ' '
      << percentile_sorted(bootstrap.rate_source_to_sink_per_lag, 0.975)
      << "\n";
  const auto mfpt_stats = mean_std(bootstrap.mfpt_source_to_sink_lags);
  out << "mfpt_source_to_sink_lags " << mfpt_stats.first << ' '
      << mfpt_stats.second << ' '
      << percentile_sorted(bootstrap.mfpt_source_to_sink_lags, 0.025) << ' '
      << percentile_sorted(bootstrap.mfpt_source_to_sink_lags, 0.975)
      << "\n";
  const auto src_stats = mean_std(bootstrap.source_population);
  out << "source_population " << src_stats.first << ' ' << src_stats.second
      << ' ' << percentile_sorted(bootstrap.source_population, 0.025) << ' '
      << percentile_sorted(bootstrap.source_population, 0.975) << "\n";
  const auto sink_stats = mean_std(bootstrap.sink_population);
  out << "sink_population " << sink_stats.first << ' ' << sink_stats.second
      << ' ' << percentile_sorted(bootstrap.sink_population, 0.025) << ' '
      << percentile_sorted(bootstrap.sink_population, 0.975) << "\n";

  std::ofstream states_out(states_filename);
  states_out << "# state q_plus_ref q_plus_mean q_plus_std reactive_density_ref "
                "reactive_density_mean reactive_density_std\n";
  const double n =
      static_cast<double>(std::max(1, bootstrap.successful_samples));
  for (int state = 0; state < reference_tpt.n_states(); ++state) {
    const double q_mean = bootstrap.q_forward_sum[state] / n;
    const double q_var =
        std::max(0.0, bootstrap.q_forward_sq_sum[state] / n - q_mean * q_mean);
    const double rho_mean = bootstrap.reactive_density_sum[state] / n;
    const double rho_var = std::max(
        0.0, bootstrap.reactive_density_sq_sum[state] / n - rho_mean * rho_mean);
    states_out << state << ' ' << reference_tpt.forward_committor()[state] << ' '
               << q_mean << ' ' << std::sqrt(q_var) << ' '
               << reference_tpt.reactive_density()[state] << ' ' << rho_mean
               << ' ' << std::sqrt(rho_var) << "\n";
  }
}

void write_validation_summary_file(
    const std::string &filename, const RuntimeCache &cache,
    const std::vector<ChapmanKolmogorovSummary> &ck_summaries,
    const MsmBootstrapAggregate *msm_bootstrap,
    const TptBootstrapAggregate *tpt_bootstrap) {
  std::ofstream out(filename);
  out << "# TITAN validation summary\n";
  out << "vamp_candidates " << cache.vamp_sweep.candidates.size() << "\n";
  out << "selected_components " << cache.vamp_sweep.selected.components << "\n";
  out << "selected_lag_frames " << cache.vamp_sweep.selected.lag_frames << "\n";
  out << "selected_clusters " << cache.vamp_sweep.selected.n_clusters << "\n";
  out << "auto_pick_applied " << (cache.auto_pick_applied ? "yes" : "no")
      << "\n";
  out << "ck_models " << ck_summaries.size() << "\n";
  for (const auto &summary : ck_summaries) {
    out << "ck_model " << summary.model_source << ' '
        << (summary.valid ? "valid" : "invalid") << ' ' << summary.metrics.size()
        << "\n";
  }
  if (msm_bootstrap) {
    out << "msm_bootstrap_requested " << msm_bootstrap->requested_samples
        << "\n";
    out << "msm_bootstrap_successful " << msm_bootstrap->successful_samples
        << "\n";
  } else {
    out << "msm_bootstrap_requested 0\n";
    out << "msm_bootstrap_successful 0\n";
  }
  if (tpt_bootstrap) {
    out << "tpt_bootstrap_requested " << tpt_bootstrap->requested_samples
        << "\n";
    out << "tpt_bootstrap_successful " << tpt_bootstrap->successful_samples
        << "\n";
  } else {
    out << "tpt_bootstrap_requested 0\n";
    out << "tpt_bootstrap_successful 0\n";
  }
}

} // namespace

bool validation_bootstrap_matches_tpt_legacy(const arguments &args) {
  return args.validation.flag_validation && args.validation.bootstrap_samples > 0 &&
         args.tpt.flag_tpt && args.tpt.bootstrap_samples > 0 &&
         args.validation.bootstrap_samples == args.tpt.bootstrap_samples &&
         args.validation.bootstrap_seed == args.tpt.bootstrap_seed;
}

void maybe_run_preselect(const arguments &args, const TitanKinetics &tica_analyzer,
                         int current_effective_k,
                         std::unique_ptr<TitanMSM> &msm_analyzer,
                         TitanExecutor &executor, std::fstream &log,
                         RuntimeCache &cache) {
  if (!args.validation.flag_validation || !args.validation.auto_pick ||
      cache.have_vamp_sweep || !msm_analyzer || msm_analyzer->n_frames() <= 0) {
    return;
  }

  cache.current_effective_model = {
      std::max(1, std::min(tica_analyzer.retained_components(),
                           msm_analyzer->n_components())),
      std::max(1, args.msm.msm_lag), std::max(2, current_effective_k)};

  auto sweep = evaluate_vamp_candidates(
      *msm_analyzer, cache.current_effective_model.components,
      build_candidate_grid(args, tica_analyzer, *msm_analyzer,
                           current_effective_k),
      args.validation.vamp_folds, args.msm.msm_kmeans_iter,
      static_cast<unsigned int>(args.msm.msm_seed), executor);
  sweep.used_auto_pick = true;
  sweep.selected_is_current =
      !sweep.valid || same_config(sweep.selected, cache.current_effective_model);
  if (!sweep.valid) {
    sweep.selected = cache.current_effective_model;
    sweep.selected_is_current = true;
  }

  cache.have_vamp_sweep = true;
  cache.auto_pick_applied = false;
  if (sweep.valid && !sweep.selected_is_current && sweep.winning_model) {
    msm_analyzer = std::move(sweep.winning_model);
    cache.auto_pick_applied = true;
    TitanUI::print_info(
        "Validation: auto-pick selected MSM candidate components=" +
        std::to_string(sweep.selected.components) + ", lag=" +
        std::to_string(sweep.selected.lag_frames) + ", clusters=" +
        std::to_string(sweep.selected.n_clusters) + ".");
    TitanUI::log_kv(log, "Validation Auto-Pick",
                    "components=" + std::to_string(sweep.selected.components) +
                        ", lag=" + std::to_string(sweep.selected.lag_frames) +
                        ", clusters=" +
                        std::to_string(sweep.selected.n_clusters));
  } else {
    TitanUI::log_kv(log, "Validation Auto-Pick",
                    "kept current MSM configuration");
  }
  cache.vamp_sweep = std::move(sweep);
}

void run_postproc(const arguments &args, const TitanKinetics &tica_analyzer,
                  std::unique_ptr<TitanMSM> &msm_analyzer,
                  TitanExecutor &executor, info &info_output,
                  std::fstream &log, RuntimeCache &cache) {
  if (!args.validation.flag_validation || !msm_analyzer || !msm_analyzer->is_built())
    return;

  auto start_validation = std::chrono::steady_clock::now();
  TitanUI::print_module_header("Validation Analysis");

  cache.current_effective_model = {msm_analyzer->n_components(),
                                   msm_analyzer->msm_lag_time(),
                                   msm_analyzer->n_clusters()};
  if (!cache.have_vamp_sweep) {
    cache.vamp_sweep = evaluate_vamp_candidates(
        *msm_analyzer, cache.current_effective_model.components,
        build_candidate_grid(args, tica_analyzer, *msm_analyzer,
                             msm_analyzer->n_clusters()),
        args.validation.vamp_folds, args.msm.msm_kmeans_iter,
        static_cast<unsigned int>(args.msm.msm_seed), executor);
    cache.vamp_sweep.used_auto_pick = false;
    cache.vamp_sweep.selected_is_current =
        !cache.vamp_sweep.valid ||
        same_config(cache.vamp_sweep.selected, cache.current_effective_model);
    if (!cache.vamp_sweep.valid) {
      cache.vamp_sweep.selected = cache.current_effective_model;
      cache.vamp_sweep.selected_is_current = true;
    }
    cache.have_vamp_sweep = true;
  }
  cache.vamp_sweep.selected_is_current =
      same_config(cache.vamp_sweep.selected, cache.current_effective_model);

  write_vamp_ranking_csv(info_output.get_validation_vamp2_ranking_filename(),
                         cache.vamp_sweep);
  write_selected_model_file(info_output.get_validation_selected_model_filename(),
                            cache);

  std::vector<ChapmanKolmogorovSummary> ck_summaries;
  const auto msm_segments = build_msm_state_segments(*msm_analyzer);
  const auto msm_ck = compute_ck_for_transition_model(
      msm_analyzer->tpm(), msm_analyzer->msm_lag_time(), msm_analyzer->n_clusters(),
      msm_segments, args.validation.ck_multiples, "msm_current");
  if (msm_ck.valid)
    ck_summaries.push_back(msm_ck);

  if (cache.tpt_context.available) {
    if (cache.tpt_context.model_source.rfind("dham_reweighted_", 0) == 0 &&
        cache.dham_context.available) {
      const auto dham_ck = compute_ck_for_dham(
          cache.dham_context, cache.tpt_context.model_tpm,
          cache.tpt_context.lag_frames, args.validation.ck_multiples,
          cache.tpt_context.model_source);
      if (dham_ck.valid)
        ck_summaries.push_back(dham_ck);
    } else if (!cache.tpt_context.bootstrap_segments.empty() &&
               cache.tpt_context.model_source != "msm_built_model") {
      const auto tpt_ck = compute_ck_for_transition_model(
          cache.tpt_context.model_tpm, cache.tpt_context.lag_frames,
          cache.tpt_context.n_states, cache.tpt_context.bootstrap_segments,
          args.validation.ck_multiples, cache.tpt_context.model_source);
      if (tpt_ck.valid)
        ck_summaries.push_back(tpt_ck);
    }
  }
  write_ck_files(info_output.get_validation_ck_summary_filename(),
                 info_output.get_validation_ck_states_filename(), ck_summaries);

  MsmBootstrapAggregate msm_bootstrap;
  TptBootstrapAggregate tpt_bootstrap;
  bool have_msm_bootstrap = false;
  bool have_tpt_bootstrap = false;
  if (args.validation.bootstrap_samples > 0) {
    std::string msm_bootstrap_error;
    if (run_msm_bootstrap(msm_segments, msm_analyzer->n_clusters(),
                          msm_analyzer->msm_lag_time(), msm_analyzer->lag_time_ns(),
                          args.validation.ck_multiples, args.msm.msm_nts,
                          args.validation.bootstrap_samples,
                          args.validation.bootstrap_seed,
                          static_cast<int>(executor.capacity()), executor,
                          msm_bootstrap, msm_bootstrap_error)) {
      have_msm_bootstrap = true;
      write_msm_bootstrap_files(
          info_output.get_validation_msm_bootstrap_summary_filename(),
          info_output.get_validation_msm_bootstrap_states_filename(),
          msm_bootstrap, msm_analyzer->stationary_dist(),
          msm_analyzer->compute_implied_timescales(args.msm.msm_nts),
          args.validation.ck_multiples);
    } else {
      TitanUI::print_warning("Validation MSM bootstrap: " + msm_bootstrap_error);
    }

    if (cache.tpt_context.available) {
      try {
        TitanTPT reference_tpt(cache.tpt_context.model_tpm, cache.tpt_context.model_pi);
        reference_tpt.compute(cache.tpt_context.source_states,
                              cache.tpt_context.sink_states,
                              cache.tpt_context.top_paths,
                              cache.tpt_context.flux_cut, nullptr, 1);
        std::string tpt_bootstrap_error;
        const bool is_dham_model =
            cache.tpt_context.model_source.rfind("dham_reweighted_", 0) == 0;
        const bool ok =
            is_dham_model
                ? run_dham_tpt_bootstrap(
                      cache.dham_context, cache.tpt_context.top_paths,
                      cache.tpt_context.flux_cut,
                      cache.tpt_context.source_states,
                      cache.tpt_context.sink_states,
                      args.validation.bootstrap_samples,
                      args.validation.bootstrap_seed,
                      static_cast<int>(executor.capacity()), executor,
                      tpt_bootstrap, tpt_bootstrap_error)
                : run_tpt_bootstrap(
                      cache.tpt_context.bootstrap_segments,
                      cache.tpt_context.n_states,
                      cache.tpt_context.lag_frames,
                      cache.tpt_context.lag_time_ns,
                      cache.tpt_context.symmetrize_bootstrap,
                      cache.tpt_context.source_states,
                      cache.tpt_context.sink_states,
                      cache.tpt_context.top_paths,
                      cache.tpt_context.flux_cut,
                      static_cast<int>(executor.capacity()),
                      args.validation.bootstrap_samples,
                      args.validation.bootstrap_seed, executor, tpt_bootstrap,
                      tpt_bootstrap_error);
        if (ok) {
          have_tpt_bootstrap = true;
          write_tpt_bootstrap_files(
              info_output.get_validation_tpt_bootstrap_summary_filename(),
              info_output.get_validation_tpt_bootstrap_states_filename(),
              tpt_bootstrap, reference_tpt, cache.tpt_context.model_source);
          if (validation_bootstrap_matches_tpt_legacy(args)) {
            write_tpt_bootstrap_files(
                info_output.get_tpt_bootstrap_summary_filename(),
                info_output.get_tpt_bootstrap_states_filename(),
                tpt_bootstrap, reference_tpt, cache.tpt_context.model_source);
          }
        } else {
          TitanUI::print_warning("Validation TPT bootstrap: " +
                                 tpt_bootstrap_error);
        }
      } catch (const std::exception &e) {
        TitanUI::print_warning(std::string("Validation TPT bootstrap: ") +
                               e.what());
      }
    }
  }

  write_validation_summary_file(
      info_output.get_validation_summary_filename(), cache, ck_summaries,
      have_msm_bootstrap ? &msm_bootstrap : nullptr,
      have_tpt_bootstrap ? &tpt_bootstrap : nullptr);

  TitanUI::log_kv(log, "Validation VAMP Candidates",
                  std::to_string(cache.vamp_sweep.candidates.size()));
  TitanUI::log_kv(log, "Validation Selected Model",
                  "components=" + std::to_string(cache.vamp_sweep.selected.components) +
                      ", lag=" +
                      std::to_string(cache.vamp_sweep.selected.lag_frames) +
                      ", clusters=" +
                      std::to_string(cache.vamp_sweep.selected.n_clusters));
  if (have_msm_bootstrap) {
    TitanUI::log_kv(log, "Validation MSM Bootstrap Successful",
                    std::to_string(msm_bootstrap.successful_samples));
  }
  if (have_tpt_bootstrap) {
    TitanUI::log_kv(log, "Validation TPT Bootstrap Successful",
                    std::to_string(tpt_bootstrap.successful_samples));
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_validation);
  TitanUI::log_time(log, "Validation Analysis", elapsed.count() / 1000.0);
}

} // namespace TitanValidation

#include "pathway_de_variants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>

namespace {

constexpr double kMinPatternStep = 0.05;
constexpr double kNlShadeSelectivePressure = 4.0;
constexpr double kProgressCompareEps = 1.0e-6;
constexpr double kPullDistanceCompareEps = 1.0e-6;
constexpr double kEnergyKeyCompareEps = 1.0e-9;
constexpr double kAngleCompareEps = 1.0e-6;

int compare_scalar(double lhs, double rhs, double eps) {
  if (lhs < rhs - eps)
    return -1;
  if (lhs > rhs + eps)
    return 1;
  return 0;
}

// Sorting requires a strict weak ordering.  Pairwise epsilon equivalence is
// not transitive, so it is reserved for decisions, never std::sort predicates.
// Give NaNs an explicit order as a defensive measure for diagnostic populations;
// the validity gate remains responsible for admitting physical candidates.
int compare_scalar_exact(double lhs, double rhs, double) {
  const bool lhs_nan = std::isnan(lhs);
  const bool rhs_nan = std::isnan(rhs);
  if (lhs_nan != rhs_nan)
    return lhs_nan ? 1 : -1;
  if (lhs < rhs)
    return -1;
  if (lhs > rhs)
    return 1;
  return 0;
}

int compare_candidates(const PathwayCandidateState &lhs, std::size_t lhs_index,
                       const PathwayCandidateState &rhs, std::size_t rhs_index,
                       PathwayMode mode, bool include_index,
                       bool exact_order = false) {
  if (lhs.valid != rhs.valid)
    return lhs.valid ? -1 : 1;
  if (!lhs.valid)
    return include_index && lhs_index != rhs_index
               ? (lhs_index < rhs_index ? -1 : 1)
               : 0;

  // Production populations share one probe mode.  Give mixed diagnostic
  // populations an explicit mode order too: a conditional probe comparison
  // only when both are fixed could otherwise create another ordering cycle.
  if (exact_order && lhs.fixed_probe != rhs.fixed_probe)
    return lhs.fixed_probe ? -1 : 1;
  const auto compare_value = exact_order ? compare_scalar_exact : compare_scalar;
  if (lhs.fixed_probe && rhs.fixed_probe) {
    if (const int cmp = compare_value(rhs.probe_length, lhs.probe_length,
                                       kPullDistanceCompareEps);
        cmp != 0) {
      return cmp;
    }
  }

  if (mode == PathwayMode::Recovery) {
    if (const int cmp = compare_value(lhs.applied_path_distance,
                                       rhs.applied_path_distance,
                                       kPullDistanceCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(rhs.delta_progress, lhs.delta_progress,
                           kProgressCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(lhs.pull_distance, rhs.pull_distance,
                           kPullDistanceCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(lhs.energy_key, rhs.energy_key,
                           kEnergyKeyCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(lhs.progress, rhs.progress, kProgressCompareEps);
        cmp != 0) {
      return cmp;
    }
  } else {
    if (const int cmp =
            compare_value(lhs.energy_key, rhs.energy_key,
                           kEnergyKeyCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(rhs.delta_progress, lhs.delta_progress,
                           kProgressCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(lhs.pull_distance, rhs.pull_distance,
                           kPullDistanceCompareEps);
        cmp != 0) {
      return cmp;
    }
    if (const int cmp =
            compare_value(lhs.angle_deg, rhs.angle_deg, kAngleCompareEps);
        cmp != 0) {
      return cmp;
    }
  }

  if (include_index && lhs_index != rhs_index)
    return lhs_index < rhs_index ? -1 : 1;
  return 0;
}

std::mt19937 make_candidate_rng(unsigned int seed, std::size_t generation,
                                std::size_t index, PullDeVariant variant) {
  const unsigned int variant_tag =
      static_cast<unsigned int>(variant) * 0x9e3779b9U;
  std::seed_seq seq{seed,
                    static_cast<unsigned int>(generation),
                    static_cast<unsigned int>(generation >> 32),
                    static_cast<unsigned int>(index),
                    static_cast<unsigned int>(index >> 32),
                    seed ^ variant_tag};
  return std::mt19937(seq);
}

std::vector<std::size_t>
sorted_candidate_indices(const std::vector<PathwayCandidateState> &population,
                         PathwayMode mode) {
  std::vector<std::size_t> indices(population.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::stable_sort(indices.begin(), indices.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return pathway_candidate_strict_less(
                         population[lhs], lhs, population[rhs], rhs, mode);
                   });
  return indices;
}

std::size_t best_candidate_index(
    const std::vector<PathwayCandidateState> &population, PathwayMode mode) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < population.size(); ++i) {
    if (compare_candidates(population[i], i, population[best], best, mode,
                           true) < 0)
      best = i;
  }
  return best;
}

double weighted_arithmetic_mean(const std::vector<double> &values,
                                const std::vector<double> &weights) {
  double numer = 0.0;
  double denom = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    numer += weights[i] * values[i];
    denom += weights[i];
  }
  return denom > 0.0 ? numer / denom : 0.0;
}

double weighted_lehmer_mean(const std::vector<double> &values,
                            const std::vector<double> &weights) {
  double numer = 0.0;
  double denom = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    numer += weights[i] * values[i] * values[i];
    denom += weights[i] * values[i];
  }
  return denom > 0.0 ? numer / denom : 0.0;
}

double weighted_generalized_lehmer_mean(const std::vector<double> &values,
                                        const std::vector<double> &weights,
                                        double power) {
  constexpr double kEps = 1.0e-12;
  double numer = 0.0;
  double denom = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = std::max(values[i], kEps);
    numer += weights[i] * std::pow(value, power);
    denom += weights[i] * std::pow(value, power - 1.0);
  }
  return denom > 0.0 ? numer / denom : 0.0;
}

std::size_t pbest_count(std::size_t population_size, double p_rate) {
  return std::max<std::size_t>(
      2, std::min<std::size_t>(population_size, static_cast<std::size_t>(
                                                    std::ceil(p_rate *
                                                              population_size))));
}

std::size_t sample_excluding(std::mt19937 &mt, std::size_t upper_exclusive,
                             const std::vector<std::size_t> &forbidden) {
  std::uniform_int_distribution<std::size_t> choose(0, upper_exclusive - 1);
  std::size_t value = choose(mt);
  while (std::find(forbidden.begin(), forbidden.end(), value) !=
         forbidden.end()) {
    value = choose(mt);
  }
  return value;
}

double sampled_scale_factor(double mu_f, std::mt19937 &mt) {
  std::cauchy_distribution<double> cauchy(mu_f, 0.1);
  double f = 0.0;
  do {
    f = cauchy(mt);
    if (f > 1.0)
      f = 1.0;
  } while (f <= 0.0);
  return f;
}

double sampled_crossover(double mu_cr, std::mt19937 &mt) {
  if (mu_cr < 0.0)
    return 0.0;
  std::normal_distribution<double> normal(mu_cr, 0.1);
  return std::clamp(normal(mt), 0.0, 1.0);
}

std::size_t planned_eval_budget(const PullDeVariantOptions &options) {
  if (options.planned_evaluations != 0)
    return options.planned_evaluations;
  return options.initial_population * (options.max_generations + 1);
}

double progress_ratio(std::size_t evals, std::size_t planned_evals) {
  if (planned_evals == 0)
    return 0.0;
  return std::clamp(static_cast<double>(evals) /
                        static_cast<double>(planned_evals),
                    0.0, 1.0);
}

double variant_pbest_rate(const PullDeVariantOptions &options,
                          std::size_t evals_used) {
  const double progress = progress_ratio(evals_used, planned_eval_budget(options));
  if (options.variant == PullDeVariant::Jso ||
      options.variant == PullDeVariant::NlShadeLbc) {
    return std::max(0.5 * options.p_best_rate,
                    options.p_best_rate * (1.0 - 0.5 * progress));
  }
  return options.p_best_rate;
}

double variant_bias_power_f(const PullDeVariantOptions &options,
                            std::size_t evals_used) {
  if (options.variant != PullDeVariant::NlShadeLbc)
    return 2.0;
  const double progress = progress_ratio(evals_used, planned_eval_budget(options));
  return 3.5 - 2.0 * progress;
}

double variant_bias_power_cr(const PullDeVariantOptions &options,
                             std::size_t evals_used) {
  if (options.variant != PullDeVariant::NlShadeLbc)
    return 1.0;
  const double progress = progress_ratio(evals_used, planned_eval_budget(options));
  return 1.0 + 0.5 * progress;
}

void adjust_jso_parameters(double &f, double &cr, double &fw,
                           const PullDeVariantOptions &options,
                           std::size_t evals_used) {
  fw = 1.2 * f;
  const std::size_t planned = planned_eval_budget(options);
  if (planned == 0)
    return;
  if (4 * evals_used < planned) {
    cr = std::max(cr, 0.7);
  } else if (2 * evals_used < planned) {
    cr = std::max(cr, 0.6);
  }
  if (10 * evals_used < 6 * planned && f > 0.7)
    f = 0.7;
  if (10 * evals_used < 4 * planned)
    fw = 10 * evals_used < 2 * planned ? 0.7 * f : 0.8 * f;
}

void trim_archive(std::vector<double> &archive, std::size_t capacity,
                  std::mt19937 &mt) {
  if (capacity == 0) {
    archive.clear();
    return;
  }
  while (archive.size() > capacity) {
    std::uniform_int_distribution<std::size_t> choose(0, archive.size() - 1);
    archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(choose(mt)));
  }
}

std::size_t desired_population_size(const PullDeVariantOptions &options,
                                    std::size_t evals_used) {
  if (options.variant == PullDeVariant::Shade)
    return options.initial_population;
  const std::size_t planned = planned_eval_budget(options);
  if (planned == 0)
    return options.initial_population;
  if (options.variant == PullDeVariant::NlShadeLbc) {
    const double progress = progress_ratio(evals_used, planned);
    const double progress_curve =
        progress <= 0.0 ? 0.0 : std::pow(progress, 1.0 - progress);
    const double desired =
        (static_cast<double>(options.minimum_population) -
         static_cast<double>(options.initial_population)) *
            progress_curve +
        static_cast<double>(options.initial_population);
    return std::max<std::size_t>(options.minimum_population,
                                 static_cast<std::size_t>(std::round(desired)));
  }
  const double desired =
      -static_cast<double>(options.initial_population - options.minimum_population) *
          static_cast<double>(evals_used) / static_cast<double>(planned) +
      static_cast<double>(options.initial_population);
  return std::max<std::size_t>(options.minimum_population,
                               static_cast<std::size_t>(std::round(desired)));
}

void reduce_population_if_needed(std::vector<PathwayCandidateState> &population,
                                 const PullDeVariantOptions &options,
                                 PullDeVariantStats &stats,
                                 std::size_t evals_used, PathwayMode mode) {
  const std::size_t desired = desired_population_size(options, evals_used);
  if (desired >= population.size())
    return;
  const auto ranked = sorted_candidate_indices(population, mode);
  std::vector<std::size_t> to_erase;
  to_erase.reserve(population.size() - desired);
  for (std::size_t i = desired; i < ranked.size(); ++i)
    to_erase.push_back(ranked[i]);
  std::sort(to_erase.rbegin(), to_erase.rend());
  for (const std::size_t index : to_erase) {
    population.erase(population.begin() + static_cast<std::ptrdiff_t>(index));
  }
  ++stats.population_reductions;
}

std::size_t sample_rank_biased_population_index(
    std::mt19937 &mt, const std::vector<std::size_t> &ranked,
    const std::vector<std::size_t> &forbidden, double selective_pressure) {
  std::vector<std::size_t> candidates;
  std::vector<double> weights;
  candidates.reserve(ranked.size());
  weights.reserve(ranked.size());
  const double population_size = std::max<double>(1.0, ranked.size());
  for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
    const std::size_t index = ranked[rank];
    if (std::find(forbidden.begin(), forbidden.end(), index) != forbidden.end())
      continue;
    candidates.push_back(index);
    weights.push_back(std::exp(-selective_pressure *
                               static_cast<double>(rank + 1) / population_size));
  }
  if (candidates.empty())
    return sample_excluding(mt, ranked.size(), forbidden);
  std::discrete_distribution<std::size_t> choose(weights.begin(), weights.end());
  return candidates[choose(mt)];
}

void update_final_spreads(const std::vector<PathwayCandidateState> &population,
                          const PathwayCandidateState &best_candidate,
                          PullDeVariantStats &stats) {
  if (!best_candidate.valid)
    return;
  for (const auto &candidate : population) {
    if (!candidate.valid)
      continue;
    stats.final_rho_spread =
        std::max(stats.final_rho_spread,
                 std::fabs(candidate.progress - best_candidate.progress));
  }
}

} // namespace

bool run_pathway_legacy_de(const de_arg &arguments,
                           const PathwayProgressDomain &domain,
                           PathwayMode mode,
                           const PathwayBatchEvaluator &evaluator,
                           std::vector<PathwayCandidateState> &population,
                           PathwayCandidateState &best_candidate,
                           PathwayLegacyDeStats &stats) {
  stats = {};
  stats.enabled = true;

  population.assign(arguments.pop_num, PathwayCandidateState{});
  std::vector<double> initial(arguments.pop_num, 0.0);
  for (std::size_t i = 0; i < population.size(); ++i) {
    std::mt19937 mt =
        make_candidate_rng(arguments.pull_seed, 0, i, PullDeVariant::Legacy);
    initial[i] = domain.sample(mt);
  }

  PullDeBatchEvaluation eval_info =
      evaluator(std::span<const double>(initial), std::span<PathwayCandidateState>(population));
  if (eval_info.budget_exhausted)
    stats.stopped_by_eval_budget = true;
  if (population.empty())
    return false;

  const std::size_t initial_best = best_candidate_index(population, mode);
  if (!population[initial_best].valid)
    return false;
  best_candidate = population[initial_best];
  stats.best_generation = 0;
  int stagnation_count = 0;

  for (std::size_t generation = 0; generation < arguments.ite_num; ++generation) {
    std::vector<double> trial_progress(population.size(), 0.0);
    for (std::size_t i = 0; i < population.size(); ++i) {
      std::mt19937 mt =
          make_candidate_rng(arguments.pull_seed, generation + 1, i,
                             PullDeVariant::Legacy);
      std::uniform_real_distribution<double> prob(0.0, 1.0);
      std::uniform_int_distribution<std::size_t> choose(0, population.size() - 1);

      std::size_t a = choose(mt);
      std::size_t b = choose(mt);
      std::size_t c = choose(mt);
      while (a == b || a == c || b == c || a == i || b == i || c == i) {
        a = choose(mt);
        b = choose(mt);
        c = choose(mt);
      }

      double trial = population[i].progress;
      if (arguments.algorithm == 0) {
        if (prob(mt) < arguments.cross_prob || i == choose(mt)) {
          trial = population[a].progress +
                  arguments.diff_weight *
                      (population[b].progress - population[c].progress);
        }
      } else if (prob(mt) < arguments.cross_prob) {
        trial = population[a].progress +
                arguments.F *
                    (population[b].progress - population[c].progress);
      } else {
        trial = population[a].progress +
                0.5 * (arguments.F + 1.0) *
                    (population[b].progress + population[c].progress -
                     2.0 * population[a].progress);
      }
      trial_progress[i] = domain.sanitize(trial);
    }

    std::vector<PathwayCandidateState> trial_states(population.size());
    eval_info = evaluator(std::span<const double>(trial_progress),
                          std::span<PathwayCandidateState>(trial_states));
    if (eval_info.budget_exhausted)
      stats.stopped_by_eval_budget = true;

    bool generation_improved = false;
    for (std::size_t i = 0; i < population.size(); ++i) {
      if (!trial_states[i].valid)
        continue;
      if (compare_candidates(trial_states[i], i, population[i], i, mode,
                             false) >= 0)
        continue;
      population[i] = trial_states[i];
      if (compare_candidates(population[i], i, best_candidate, initial_best,
                             mode, false) < 0) {
        best_candidate = population[i];
        generation_improved = true;
        stats.best_generation = generation + 1;
      }
    }

    stats.generations_completed = generation + 1;
    if (generation_improved)
      stagnation_count = 0;
    else
      ++stagnation_count;
    if (stats.stopped_by_eval_budget)
      break;
    if (stagnation_count >= arguments.stagnation_limit) {
      stats.stopped_by_stagnation = true;
      break;
    }
  }

  return best_candidate.valid;
}

bool run_pathway_de_variant(const PullDeVariantOptions &options,
                            const PathwayProgressDomain &domain,
                            PathwayMode mode,
                            const PathwayBatchEvaluator &evaluator,
                            std::vector<PathwayCandidateState> &population,
                            PathwayCandidateState &best_candidate,
                            PullDeVariantStats &stats) {
  stats = {};
  stats.enabled = true;
  stats.nonlinear_population_reduction =
      (options.variant == PullDeVariant::NlShadeLbc);
  stats.variant = options.variant;
  stats.variant_name = std::string(pull_de_variant_name(options.variant));
  stats.memory_size = options.memory_size;
  stats.p_best_rate = options.p_best_rate;
  stats.archive_scale = options.archive_scale;
  stats.initial_population = options.initial_population;
  stats.selective_pressure = stats.nonlinear_population_reduction
                                 ? kNlShadeSelectivePressure
                                 : 0.0;

  population.assign(options.initial_population, PathwayCandidateState{});
  std::vector<double> initial(options.initial_population, 0.0);
  for (std::size_t i = 0; i < initial.size(); ++i) {
    std::mt19937 mt =
        make_candidate_rng(options.seed, 0, i, options.variant);
    initial[i] = domain.sample(mt);
  }

  PullDeBatchEvaluation eval_info =
      evaluator(std::span<const double>(initial), std::span<PathwayCandidateState>(population));
  if (eval_info.budget_exhausted)
    stats.stopped_by_eval_budget = true;

  const std::size_t initial_best_index = best_candidate_index(population, mode);
  if (!population[initial_best_index].valid)
    return false;

  best_candidate = population[initial_best_index];
  std::size_t evals_used = eval_info.evaluations_used;
  std::size_t memory_slot = 0;
  int stagnation_count = 0;
  double last_p_best_rate = options.p_best_rate;
  double last_bias_power_f = variant_bias_power_f(options, evals_used);
  double last_bias_power_cr = variant_bias_power_cr(options, evals_used);

  const std::size_t memory_size = std::max<std::size_t>(1, options.memory_size);
  std::vector<double> memory_f(memory_size,
                               (options.variant == PullDeVariant::Jso ||
                                options.variant == PullDeVariant::NlShadeLbc)
                                   ? 0.3
                                   : 0.5);
  std::vector<double> memory_cr(memory_size,
                                (options.variant == PullDeVariant::Jso ||
                                 options.variant == PullDeVariant::NlShadeLbc)
                                    ? 0.8
                                    : 0.5);
  std::vector<double> archive;
  archive.reserve(static_cast<std::size_t>(
      std::round(options.archive_scale * options.initial_population)));
  stats.archive_capacity = archive.capacity();

  for (std::size_t generation = 0; generation < options.max_generations;
       ++generation) {
    const auto ranked = sorted_candidate_indices(population, mode);
    const std::size_t population_size = population.size();
    std::vector<double> trial_progress(population_size, 0.0);
    std::vector<PathwayCandidateState> trial_states(population_size);
    std::vector<double> used_f(population_size, 0.0);
    std::vector<double> used_cr(population_size, 0.0);

    for (std::size_t i = 0; i < population_size; ++i) {
      std::mt19937 mt =
          make_candidate_rng(options.seed, generation + 1, i, options.variant);
      std::uniform_int_distribution<std::size_t> choose_mem(0, memory_size - 1);
      const std::size_t memory_index = choose_mem(mt);

      double mu_f = memory_f[memory_index];
      double mu_cr = memory_cr[memory_index];
      if (options.variant == PullDeVariant::Jso &&
          memory_index == memory_size - 1) {
        mu_f = 0.9;
        mu_cr = 0.9;
      }

      double cr = sampled_crossover(mu_cr, mt);
      double f = sampled_scale_factor(mu_f, mt);
      double fw = f;
      if (options.variant == PullDeVariant::Jso ||
          options.variant == PullDeVariant::NlShadeLbc) {
        adjust_jso_parameters(f, cr, fw, options, evals_used);
      }

      const double current_p = variant_pbest_rate(options, evals_used);
      last_p_best_rate = current_p;
      last_bias_power_f = variant_bias_power_f(options, evals_used);
      last_bias_power_cr = variant_bias_power_cr(options, evals_used);
      const std::size_t pbest_size = pbest_count(population_size, current_p);
      std::uniform_int_distribution<std::size_t> choose_pbest(0, pbest_size - 1);
      const std::size_t pbest_index = ranked[choose_pbest(mt)];

      const double xi = population[i].progress;
      const double xpbest = population[pbest_index].progress;
      std::size_t r1 = 0;
      std::size_t r2_population = 0;
      std::size_t r2_archive = 0;
      bool use_archive_r2 = false;
      if (options.variant == PullDeVariant::NlShadeLbc) {
        r1 = sample_rank_biased_population_index(mt, ranked, {i, pbest_index},
                                                 kNlShadeSelectivePressure);
        use_archive_r2 =
            !archive.empty() && std::bernoulli_distribution(0.5)(mt);
        if (use_archive_r2) {
          std::uniform_int_distribution<std::size_t> choose_archive(
              0, archive.size() - 1);
          r2_archive = choose_archive(mt);
        } else {
          r2_population = sample_rank_biased_population_index(
              mt, ranked, {i, pbest_index, r1}, kNlShadeSelectivePressure);
        }
      } else {
        r1 = sample_excluding(mt, population_size, {i, pbest_index});
        const std::size_t combined_size = population_size + archive.size();
        const std::size_t r2_combined =
            combined_size > 0 ? sample_excluding(mt, combined_size, {i, r1}) : 0;
        use_archive_r2 = (r2_combined >= population_size);
        if (use_archive_r2)
          r2_archive = r2_combined - population_size;
        else
          r2_population = r2_combined;
      }

      const double xr1 = population[r1].progress;
      const double xr2 =
          use_archive_r2 ? archive[r2_archive] : population[r2_population].progress;

      const double mutant =
          xi + fw * (xpbest - xi) + f * (xr1 - xr2);
      double trial = mutant;
      if (cr < 0.0)
        trial = xi;
      trial_progress[i] = domain.sanitize(trial);
      used_f[i] = f;
      used_cr[i] = cr;
    }

    eval_info = evaluator(std::span<const double>(trial_progress),
                          std::span<PathwayCandidateState>(trial_states));
    evals_used += eval_info.evaluations_used;
    if (eval_info.budget_exhausted)
      stats.stopped_by_eval_budget = true;

    std::vector<double> success_f;
    std::vector<double> success_cr;
    std::vector<double> delta_f;
    success_f.reserve(population_size);
    success_cr.reserve(population_size);
    delta_f.reserve(population_size);

    bool generation_improved = false;
    for (std::size_t i = 0; i < population_size; ++i) {
      if (!trial_states[i].valid)
        continue;
      if (compare_candidates(trial_states[i], i, population[i], i, mode,
                             false) >= 0)
        continue;

      archive.push_back(population[i].progress);
      const double improvement =
          std::fabs(population[i].delta_progress - trial_states[i].delta_progress) +
          std::fabs(population[i].pull_distance - trial_states[i].pull_distance) +
          std::fabs(population[i].energy_key - trial_states[i].energy_key);
      if (improvement > 0.0) {
        success_f.push_back(used_f[i]);
        success_cr.push_back(used_cr[i]);
        delta_f.push_back(improvement);
      }
      population[i] = trial_states[i];
      if (compare_candidates(population[i], i, best_candidate,
                             initial_best_index, mode, false) < 0) {
        best_candidate = population[i];
        generation_improved = true;
      }
    }

    std::mt19937 trim_rng = make_candidate_rng(
        options.seed, generation + 1, population_size + 17, options.variant);
    const std::size_t desired_archive_capacity =
        options.variant == PullDeVariant::Shade
            ? static_cast<std::size_t>(
                  std::round(options.archive_scale * options.initial_population))
            : static_cast<std::size_t>(
                  std::round(options.archive_scale * population.size()));
    trim_archive(archive, desired_archive_capacity, trim_rng);
    stats.archive_capacity = desired_archive_capacity;

    if (!success_f.empty()) {
      std::vector<double> weights(success_f.size(), 0.0);
      const double delta_sum =
          std::accumulate(delta_f.begin(), delta_f.end(), 0.0);
      if (delta_sum > 0.0) {
        for (std::size_t i = 0; i < weights.size(); ++i)
          weights[i] = delta_f[i] / delta_sum;
        if (options.variant == PullDeVariant::NlShadeLbc) {
          memory_f[memory_slot] = weighted_generalized_lehmer_mean(
              success_f, weights, last_bias_power_f);
          memory_cr[memory_slot] = weighted_generalized_lehmer_mean(
              success_cr, weights, last_bias_power_cr);
        } else {
          memory_f[memory_slot] = weighted_lehmer_mean(success_f, weights);
          memory_cr[memory_slot] =
              weighted_arithmetic_mean(success_cr, weights);
        }
        memory_slot = (memory_slot + 1) % memory_size;
      }
      stagnation_count = 0;
      if (generation_improved)
        stats.best_generation = generation + 1;
    } else {
      ++stagnation_count;
    }

    reduce_population_if_needed(population, options, stats, evals_used, mode);
    if (!population.empty()) {
      const std::size_t best_index = best_candidate_index(population, mode);
      if (population[best_index].valid &&
          compare_candidates(population[best_index], best_index, best_candidate,
                             initial_best_index, mode, false) < 0) {
        best_candidate = population[best_index];
        stats.best_generation = generation + 1;
      }
    }

    stats.generations_completed = generation + 1;
    if (stats.stopped_by_eval_budget)
      break;
    if (stagnation_count >= options.stagnation_limit) {
      stats.stopped_by_stagnation = true;
      break;
    }
  }

  stats.final_population = population.size();
  stats.final_p_best_rate = last_p_best_rate;
  stats.bias_power_f = last_bias_power_f;
  stats.bias_power_cr = last_bias_power_cr;
  update_final_spreads(population, best_candidate, stats);
  return best_candidate.valid;
}

double pathway_variant_pattern_initial_step(const PullDeVariantStats &stats,
                                            double progress_window) {
  const double default_step = std::max(kMinPatternStep, 0.25 * progress_window);
  const double population_scale = std::max(
      1.0,
      std::sqrt(static_cast<double>(std::max<std::size_t>(1, stats.final_population))));
  const double seeded = std::max(stats.final_rho_spread, default_step);
  return std::max(kMinPatternStep, seeded / population_scale);
}

std::vector<std::size_t> select_pathway_pattern_seed_indices(
    const std::vector<PathwayCandidateState> &population, std::size_t max_seeds,
    double min_progress_gap, PathwayMode mode) {
  const auto ranked = sorted_candidate_indices(population, mode);
  std::vector<std::size_t> selected;
  selected.reserve(std::min(max_seeds, ranked.size()));
  for (const std::size_t candidate_index : ranked) {
    if (!population[candidate_index].valid)
      continue;
    bool far_enough = true;
    for (const std::size_t chosen_index : selected) {
      if (std::fabs(population[candidate_index].progress -
                    population[chosen_index].progress) < min_progress_gap) {
        far_enough = false;
        break;
      }
    }
    if (!far_enough)
      continue;
    selected.push_back(candidate_index);
    if (selected.size() >= max_seeds)
      break;
  }
  return selected;
}

double pathway_energy_key(double raw_score) {
  if (!std::isfinite(raw_score))
    return std::numeric_limits<double>::infinity();
  const double magnitude = std::log1p(std::fabs(raw_score));
  return std::signbit(raw_score) ? -magnitude : magnitude;
}

bool pathway_candidate_better(const PathwayCandidateState &lhs,
                              std::size_t lhs_index,
                              const PathwayCandidateState &rhs,
                              std::size_t rhs_index, PathwayMode mode) {
  return compare_candidates(lhs, lhs_index, rhs, rhs_index, mode, true) < 0;
}

bool pathway_candidate_strict_less(const PathwayCandidateState &lhs,
                                   std::size_t lhs_index,
                                   const PathwayCandidateState &rhs,
                                   std::size_t rhs_index, PathwayMode mode) {
  return compare_candidates(lhs, lhs_index, rhs, rhs_index, mode, true, true) < 0;
}

bool pathway_candidate_equivalent(const PathwayCandidateState &lhs,
                                  const PathwayCandidateState &rhs,
                                  PathwayMode mode) {
  return compare_candidates(lhs, 0, rhs, 0, mode, false) == 0;
}

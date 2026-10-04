#include "pull_de_variants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kScoreImproveEps = 1.0e-8;
constexpr double kNlShadeSelectivePressure = 4.0;

double normalize_angle_2pi(double angle) {
  angle = std::fmod(angle, 2.0 * kPi);
  if (angle < 0.0)
    angle += 2.0 * kPi;
  return angle;
}

double clamp_theta(double theta) {
  if (theta < 0.0)
    return -theta;
  if (theta > kPi)
    return 2.0 * kPi - theta;
  return theta;
}

force sanitize_spherical(force spherical) {
  spherical.Fx = clamp_theta(spherical.Fx);
  spherical.Fy = normalize_angle_2pi(spherical.Fy);
  spherical.Fz = std::fabs(spherical.Fz);
  return spherical;
}

double wrap_angle_pi(double angle) {
  angle = std::fmod(angle + kPi, 2.0 * kPi);
  if (angle < 0.0)
    angle += 2.0 * kPi;
  return angle - kPi;
}

void cartesian_to_spherical(double x, double y, double z, double &theta,
                            double &phi) {
  const double radius = std::sqrt(x * x + y * y + z * z);
  if (radius < 1.0e-12) {
    theta = 0.0;
    phi = 0.0;
    return;
  }
  theta = std::acos(z / radius);
  phi = std::atan2(y, x);
  if (phi < 0.0)
    phi += 2.0 * kPi;
}

double angle_to_old_vector_deg(const PullDeVariantOptions &options,
                               const force &spherical) {
  if (!options.has_old_vector)
    return 0.0;
  const double old_mod = options.old_vector.modul();
  if (old_mod < 1.0e-12)
    return 0.0;

  const double sin_theta = std::sin(spherical.Fx);
  const double nx = sin_theta * std::cos(spherical.Fy);
  const double ny = sin_theta * std::sin(spherical.Fy);
  const double nz = std::cos(spherical.Fx);
  const double ox = options.old_vector.Fx / old_mod;
  const double oy = options.old_vector.Fy / old_mod;
  const double oz = options.old_vector.Fz / old_mod;
  const double cos_val = std::clamp(nx * ox + ny * oy + nz * oz, -1.0, 1.0);
  return std::acos(cos_val) * 57.2957795;
}

bool is_angle_allowed(const PullDeVariantOptions &options,
                      const force &spherical) {
  if (!options.has_old_vector)
    return true;
  return angle_to_old_vector_deg(options, spherical) <
         options.restrain_angle_deg;
}

force sample_initial_spherical_candidate(std::mt19937 &mt,
                                         const PullDeVariantOptions &options) {
  std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * kPi);
  std::uniform_real_distribution<double> dist_cos_theta(-1.0, 1.0);
  std::uniform_real_distribution<double> dist_rho(options.min_scan_radius,
                                                  options.scan_radius);
  std::normal_distribution<double> dist_noise(0.0, 0.1);

  double theta_ref = 0.0;
  double phi_ref = 0.0;
  if (options.has_old_vector) {
    cartesian_to_spherical(options.old_vector.Fx, options.old_vector.Fy,
                           options.old_vector.Fz, theta_ref, phi_ref);
  }

  for (int attempt = 0; attempt < 500; ++attempt) {
    force spherical{std::acos(dist_cos_theta(mt)), dist_phi(mt),
                    dist_rho(mt)};
    if (is_angle_allowed(options, spherical))
      return spherical;
  }

  force fallback{theta_ref + dist_noise(mt), phi_ref + dist_noise(mt),
                 dist_rho(mt)};
  return sanitize_spherical(fallback);
}

std::mt19937 make_candidate_rng(unsigned int seed, std::size_t generation,
                                std::size_t index, PullDeVariant variant) {
  const unsigned int variant_tag = static_cast<unsigned int>(variant) * 0x9e3779b9U;
  std::seed_seq seq{seed,
                    static_cast<unsigned int>(generation),
                    static_cast<unsigned int>(generation >> 32),
                    static_cast<unsigned int>(index),
                    static_cast<unsigned int>(index >> 32),
                    seed ^ variant_tag};
  return std::mt19937(seq);
}

bool score_improves(double lhs, double rhs) {
  return lhs + kScoreImproveEps < rhs;
}

std::size_t best_candidate_index(const std::vector<PullCandidateState> &population) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < population.size(); ++i) {
    if (population[i].score < population[best].score)
      best = i;
  }
  return best;
}

std::vector<std::size_t>
sorted_candidate_indices(const std::vector<PullCandidateState> &population) {
  std::vector<std::size_t> indices(population.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::sort(indices.begin(), indices.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              if (std::fabs(population[lhs].score - population[rhs].score) >
                  kScoreImproveEps)
                return population[lhs].score < population[rhs].score;
              return lhs < rhs;
            });
  return indices;
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
      2, std::min<std::size_t>(population_size,
                               static_cast<std::size_t>(
                                   std::ceil(p_rate * population_size))));
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

void trim_archive(std::vector<force> &archive, std::size_t capacity,
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

std::size_t planned_eval_budget(const PullDeVariantOptions &options) {
  if (options.planned_evaluations != 0)
    return options.planned_evaluations;
  return options.initial_population * (options.max_generations + 1);
}

double progress_ratio(std::size_t evals, std::size_t planned_evals) {
  if (planned_evals == 0)
    return 0.0;
  return std::clamp(static_cast<double>(evals) / planned_evals, 0.0, 1.0);
}

double variant_pbest_rate(const PullDeVariantOptions &options,
                          std::size_t evals_used) {
  const double progress = progress_ratio(evals_used, planned_eval_budget(options));
  if (options.variant == PullDeVariant::Jso) {
    return std::max(0.5 * options.p_best_rate,
                    options.p_best_rate * (1.0 - 0.5 * progress));
  }
  if (options.variant == PullDeVariant::NlShadeLbc)
    return std::max(0.5 * options.p_best_rate,
                    options.p_best_rate * (1.0 - 0.5 * progress));
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
  if (10 * evals_used < 4 * planned) {
    fw = 10 * evals_used < 2 * planned ? 0.7 * f : 0.8 * f;
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
    return std::max<std::size_t>(
        options.minimum_population,
        static_cast<std::size_t>(std::round(desired)));
  }
  const double desired =
      -static_cast<double>(options.initial_population - options.minimum_population) *
          static_cast<double>(evals_used) / static_cast<double>(planned) +
      static_cast<double>(options.initial_population);
  return std::max<std::size_t>(
      options.minimum_population,
      static_cast<std::size_t>(std::round(desired)));
}

void reduce_population_if_needed(std::vector<PullCandidateState> &population,
                                 const PullDeVariantOptions &options,
                                 PullDeVariantStats &stats,
                                 std::size_t evals_used) {
  const std::size_t desired = desired_population_size(options, evals_used);
  if (desired >= population.size())
    return;
  const auto ranked = sorted_candidate_indices(population);
  std::vector<std::size_t> to_erase;
  to_erase.reserve(population.size() - desired);
  for (std::size_t i = desired; i < ranked.size(); ++i)
    to_erase.push_back(ranked[i]);
  std::sort(to_erase.rbegin(), to_erase.rend());
  for (const std::size_t index : to_erase)
    population.erase(population.begin() + static_cast<std::ptrdiff_t>(index));
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

void update_final_spreads(const std::vector<PullCandidateState> &population,
                          const PullCandidateState &best_candidate,
                          PullDeVariantStats &stats) {
  if (!best_candidate.valid)
    return;
  for (const auto &candidate : population) {
    if (!candidate.valid)
      continue;
    stats.final_theta_spread_rad =
        std::max(stats.final_theta_spread_rad,
                 std::fabs(candidate.spherical.Fx - best_candidate.spherical.Fx));
    stats.final_phi_spread_rad =
        std::max(stats.final_phi_spread_rad,
                 std::fabs(wrap_angle_pi(candidate.spherical.Fy -
                                         best_candidate.spherical.Fy)));
    stats.final_rho_spread =
        std::max(stats.final_rho_spread,
                 std::fabs(candidate.spherical.Fz - best_candidate.spherical.Fz));
  }
}

} // namespace

PullOptimizerKind classify_pull_optimizer(std::string_view pull_opt) {
  if (pull_opt == "de")
    return PullOptimizerKind::De;
  if (pull_opt == "de+pattern")
    return PullOptimizerKind::DePattern;
  if (pull_opt == "de-shade")
    return PullOptimizerKind::DeShade;
  if (pull_opt == "de-l-shade")
    return PullOptimizerKind::DeLShade;
  if (pull_opt == "de-jso")
    return PullOptimizerKind::DeJso;
  if (pull_opt == "de-jso+pattern")
    return PullOptimizerKind::DeJsoPattern;
  if (pull_opt == "de-nl-shade-lbc")
    return PullOptimizerKind::DeNlShadeLbc;
  if (pull_opt == "de-nl-shade-lbc+pattern")
    return PullOptimizerKind::DeNlShadeLbcPattern;
  if (pull_opt == "direct")
    return PullOptimizerKind::Direct;
  if (pull_opt == "direct+pattern")
    return PullOptimizerKind::DirectPattern;
  if (pull_opt == "corridor-grid")
    return PullOptimizerKind::CorridorGrid;
  return PullOptimizerKind::Unknown;
}

bool optimizer_is_de_family(std::string_view pull_opt) {
  const PullOptimizerKind kind = classify_pull_optimizer(pull_opt);
  return kind == PullOptimizerKind::De || kind == PullOptimizerKind::DePattern ||
         kind == PullOptimizerKind::DeShade ||
         kind == PullOptimizerKind::DeLShade ||
         kind == PullOptimizerKind::DeJso ||
         kind == PullOptimizerKind::DeJsoPattern ||
         kind == PullOptimizerKind::DeNlShadeLbc ||
         kind == PullOptimizerKind::DeNlShadeLbcPattern;
}

bool optimizer_is_direct_family(std::string_view pull_opt) {
  const PullOptimizerKind kind = classify_pull_optimizer(pull_opt);
  return kind == PullOptimizerKind::Direct ||
         kind == PullOptimizerKind::DirectPattern;
}

PullDeVariant de_variant_from_optimizer(PullOptimizerKind kind) {
  switch (kind) {
  case PullOptimizerKind::DeShade:
    return PullDeVariant::Shade;
  case PullOptimizerKind::DeLShade:
    return PullDeVariant::LShade;
  case PullOptimizerKind::DeJso:
  case PullOptimizerKind::DeJsoPattern:
    return PullDeVariant::Jso;
  case PullOptimizerKind::DeNlShadeLbc:
  case PullOptimizerKind::DeNlShadeLbcPattern:
    return PullDeVariant::NlShadeLbc;
  default:
    return PullDeVariant::Legacy;
  }
}

std::string_view pull_de_variant_name(PullDeVariant variant) {
  switch (variant) {
  case PullDeVariant::Shade:
    return "de-shade";
  case PullDeVariant::LShade:
    return "de-l-shade";
  case PullDeVariant::Jso:
    return "de-jso";
  case PullDeVariant::NlShadeLbc:
    return "de-nl-shade-lbc";
  default:
    return "de";
  }
}

bool run_pull_de_variant(const PullDeVariantOptions &options,
                         const PullDeBatchEvaluator &evaluator,
                         std::vector<PullCandidateState> &population,
                         PullCandidateState &best_candidate,
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

  population.assign(options.initial_population, PullCandidateState{});
  std::mt19937 init_rng(options.seed);
  for (auto &candidate : population)
    candidate.spherical = sample_initial_spherical_candidate(init_rng, options);

  std::vector<force> proposals(population.size());
  for (std::size_t i = 0; i < population.size(); ++i)
    proposals[i] = population[i].spherical;
  PullDeBatchEvaluation eval_info =
      evaluator(std::span<const force>(proposals),
                std::span<PullCandidateState>(population));
  if (eval_info.budget_exhausted)
    stats.stopped_by_eval_budget = true;

  const std::size_t initial_best_index = best_candidate_index(population);
  if (!population[initial_best_index].valid)
    return false;

  best_candidate = population[initial_best_index];
  std::size_t best_index = initial_best_index;
  double current_best_score = best_candidate.score;
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
  std::vector<force> archive;
  archive.reserve(static_cast<std::size_t>(
      std::round(options.archive_scale * options.initial_population)));
  stats.archive_capacity = archive.capacity();

  for (std::size_t generation = 0; generation < options.max_generations;
       ++generation) {
    const auto ranked = sorted_candidate_indices(population);
    const std::size_t population_size = population.size();
    std::vector<force> trial_spherical(population_size);
    std::vector<PullCandidateState> trial_states(population_size);
    std::vector<double> used_f(population_size, 0.0);
    std::vector<double> used_cr(population_size, 0.0);

    for (std::size_t i = 0; i < population_size; ++i) {
      std::mt19937 mt =
          make_candidate_rng(options.seed, generation + 1, i, options.variant);
      std::uniform_int_distribution<std::size_t> choose_mem(0, memory_size - 1);
      const std::size_t memory_index = choose_mem(mt);

      double mu_f = memory_f[memory_index];
      double mu_cr = memory_cr[memory_index];
      if (options.variant == PullDeVariant::Jso && memory_index == memory_size - 1) {
        mu_f = 0.9;
        mu_cr = 0.9;
      }

      double cr = sampled_crossover(mu_cr, mt);
      double f = sampled_scale_factor(mu_f, mt);
      double fw = f;
      if (options.variant == PullDeVariant::Jso ||
          options.variant == PullDeVariant::NlShadeLbc)
        adjust_jso_parameters(f, cr, fw, options, evals_used);

      const double current_p = variant_pbest_rate(options, evals_used);
      last_p_best_rate = current_p;
      last_bias_power_f = variant_bias_power_f(options, evals_used);
      last_bias_power_cr = variant_bias_power_cr(options, evals_used);
      const std::size_t pbest_size = pbest_count(population_size, current_p);
      std::uniform_int_distribution<std::size_t> choose_pbest(0, pbest_size - 1);
      const std::size_t pbest_index = ranked[choose_pbest(mt)];

      const force &xi = population[i].spherical;
      const force &xpbest = population[pbest_index].spherical;
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
        const std::size_t r2_combined = combined_size > 0
                                            ? sample_excluding(mt, combined_size,
                                                               {i, r1})
                                            : 0;
        use_archive_r2 = (r2_combined >= population_size);
        if (use_archive_r2) {
          r2_archive = r2_combined - population_size;
        } else {
          r2_population = r2_combined;
        }
      }
      const force &xr1 = population[r1].spherical;
      const force &xr2 = use_archive_r2 ? archive[r2_archive]
                                        : population[r2_population].spherical;

      std::array<double, 3> mutant{
          xi.Fx + fw * (xpbest.Fx - xi.Fx) + f * (xr1.Fx - xr2.Fx),
          xi.Fy + fw * (xpbest.Fy - xi.Fy) + f * (xr1.Fy - xr2.Fy),
          xi.Fz + fw * (xpbest.Fz - xi.Fz) + f * (xr1.Fz - xr2.Fz)};

      std::array<double, 3> target{xi.Fx, xi.Fy, xi.Fz};
      std::array<double, 3> trial = target;
      std::uniform_real_distribution<double> prob(0.0, 1.0);
      std::uniform_int_distribution<int> choose_dim(0, 2);
      const int guaranteed_dim = choose_dim(mt);
      for (int dim = 0; dim < 3; ++dim) {
        if (prob(mt) < cr || dim == guaranteed_dim)
          trial[dim] = mutant[dim];
      }

      trial_spherical[i] =
          sanitize_spherical(force{trial[0], trial[1], trial[2]});
      used_f[i] = f;
      used_cr[i] = cr;
    }

    eval_info = evaluator(std::span<const force>(trial_spherical),
                          std::span<PullCandidateState>(trial_states));
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
      if (!score_improves(trial_states[i].score, population[i].score))
        continue;

      archive.push_back(population[i].spherical);
      const double improvement = std::fabs(population[i].score - trial_states[i].score);
      if (improvement > 0.0) {
        success_f.push_back(used_f[i]);
        success_cr.push_back(used_cr[i]);
        delta_f.push_back(improvement);
      }
      population[i] = trial_states[i];
      if (score_improves(population[i].score, current_best_score)) {
        current_best_score = population[i].score;
        best_index = i;
        best_candidate = population[i];
        generation_improved = true;
      }
    }

    std::mt19937 trim_rng =
        make_candidate_rng(options.seed, generation + 1, population_size + 17,
                           options.variant);
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

    reduce_population_if_needed(population, options, stats, evals_used);
    if (!population.empty()) {
      best_index = best_candidate_index(population);
      if (population[best_index].valid &&
          score_improves(population[best_index].score, best_candidate.score)) {
        best_candidate = population[best_index];
        current_best_score = best_candidate.score;
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

#ifndef PULL_DE_VARIANTS_H
#define PULL_DE_VARIANTS_H

#include "cache_sigma_epsilon.h"
#include "differential_evolution.h"
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

enum class PullOptimizerKind {
  De,
  DePattern,
  DeShade,
  DeLShade,
  DeJso,
  DeJsoPattern,
  DeNlShadeLbc,
  DeNlShadeLbcPattern,
  Direct,
  DirectPattern,
  CorridorGrid,
  Unknown
};

enum class PullDeVariant { Legacy, Shade, LShade, Jso, NlShadeLbc };

struct PullCandidateState {
  force spherical{};
  force translation{};
  cache_energy energy{};
  double score{std::numeric_limits<double>::infinity()};
  double angle_deg{180.0};
  bool valid{false};
};

struct PullDeBatchEvaluation {
  bool budget_exhausted{false};
  std::size_t evaluations_used{0};
};

struct PullDeVariantOptions {
  PullDeVariant variant{PullDeVariant::Legacy};
  unsigned int seed{42};
  std::size_t initial_population{0};
  std::size_t minimum_population{4};
  std::size_t max_generations{0};
  std::size_t planned_evaluations{0};
  std::size_t memory_size{10};
  double p_best_rate{0.2};
  double archive_scale{1.0};
  int stagnation_limit{20};
  double scan_radius{2.0};
  double min_scan_radius{0.5};
  bool has_old_vector{false};
  force old_vector{};
  double restrain_angle_deg{90.0};
};

struct PullDeVariantStats {
  bool enabled{false};
  bool nonlinear_population_reduction{false};
  PullDeVariant variant{PullDeVariant::Legacy};
  std::string variant_name{};
  std::size_t memory_size{0};
  double p_best_rate{0.0};
  double final_p_best_rate{0.0};
  double archive_scale{0.0};
  std::size_t archive_capacity{0};
  std::size_t initial_population{0};
  std::size_t final_population{0};
  std::size_t population_reductions{0};
  std::size_t generations_completed{0};
  std::size_t best_generation{0};
  double selective_pressure{0.0};
  double bias_power_f{0.0};
  double bias_power_cr{0.0};
  double final_theta_spread_rad{0.0};
  double final_phi_spread_rad{0.0};
  double final_rho_spread{0.0};
  bool stopped_by_eval_budget{false};
  bool stopped_by_stagnation{false};
};

using PullDeBatchEvaluator = std::function<PullDeBatchEvaluation(
    std::span<const force> proposals,
    std::span<PullCandidateState> evaluated_candidates)>;

PullOptimizerKind classify_pull_optimizer(std::string_view pull_opt);
bool optimizer_is_de_family(std::string_view pull_opt);
bool optimizer_is_direct_family(std::string_view pull_opt);
PullDeVariant de_variant_from_optimizer(PullOptimizerKind kind);
std::string_view pull_de_variant_name(PullDeVariant variant);

bool run_pull_de_variant(const PullDeVariantOptions &options,
                         const PullDeBatchEvaluator &evaluator,
                         std::vector<PullCandidateState> &population,
                         PullCandidateState &best_candidate,
                         PullDeVariantStats &stats);

#endif

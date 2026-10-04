#ifndef TITAN_TICA_BACKEND_POLICY_H
#define TITAN_TICA_BACKEND_POLICY_H

#include "titan_kinetics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct TitanTicaBackendPolicyConfig {
  TicaBackend requested_backend{TicaBackend::Auto};
  long double available_ram_bytes{0.0L};
  long double ram_budget_fraction{0.30L};
  long double dense_memory_estimate{0.0L};
  long double dual_memory_estimate{0.0L};
  long double estimated_frames{0.0L};
  int feature_dim{0};
  bool need_projection_cache{false};
  long double large_matrix_value_limit{250000000.0L};
};

struct TitanTicaBackendPolicyDecision {
  TicaBackend selected_backend{TicaBackend::Auto};
  long double ram_budget{0.0L};
  long double raw_feature_values{0.0L};
  long double large_matrix_value_limit{250000000.0L};
  bool large_matrix_limit_exceeded{false};
  bool selected_by_large_matrix_limit{false};
  bool hard_memory_cap_exceeded{false};
  bool explicit_exact_dual_overridden{false};
};

inline long double titan_tica_positive_or_zero(long double value) {
  return std::isfinite(value) && value > 0.0L ? value : 0.0L;
}

inline TitanTicaBackendPolicyDecision
select_tica_backend_for_policy(const TitanTicaBackendPolicyConfig &cfg) {
  TitanTicaBackendPolicyDecision decision;
  decision.selected_backend = cfg.requested_backend;
  decision.ram_budget =
      cfg.available_ram_bytes > 0.0L
          ? cfg.available_ram_bytes * std::max(0.0L, cfg.ram_budget_fraction)
          : 0.0L;
  decision.raw_feature_values =
      titan_tica_positive_or_zero(cfg.estimated_frames) *
      static_cast<long double>(std::max(0, cfg.feature_dim));
  decision.large_matrix_value_limit =
      titan_tica_positive_or_zero(cfg.large_matrix_value_limit);
  decision.large_matrix_limit_exceeded =
      decision.large_matrix_value_limit > 0.0L &&
      decision.raw_feature_values >= decision.large_matrix_value_limit;
  decision.hard_memory_cap_exceeded =
      decision.ram_budget > 0.0L &&
      titan_tica_positive_or_zero(cfg.dual_memory_estimate) >
          decision.ram_budget;

  // Never silently discard slow directions to satisfy a memory limit.
  // Dense covariance with a disk spool supports exact projection replay in
  // O(D^2) RAM, independently of trajectory length.
  const long double budget = decision.ram_budget > 0 ? decision.ram_budget
                                                   : 512.0L * 1024 * 1024;
  const bool dense_fits = titan_tica_positive_or_zero(cfg.dense_memory_estimate) <= budget;
  const bool dual_fits = titan_tica_positive_or_zero(cfg.dual_memory_estimate) <= budget;
  if (cfg.requested_backend == TicaBackend::Auto) {
    if (dense_fits)
      decision.selected_backend = cfg.need_projection_cache
          ? TicaBackend::ExactStreaming : TicaBackend::ExactDense;
    else if (dual_fits)
      decision.selected_backend = TicaBackend::ExactDual;
    else
      throw std::runtime_error("TICA exact estimates exceed the memory budget; reduce features or explicitly request randomized approximation.");
  } else if (cfg.requested_backend == TicaBackend::ExactDual && !dual_fits) {
    throw std::runtime_error("Requested exact_dual exceeds the memory budget; use exact_streaming or explicitly request randomized approximation.");
  } else if ((cfg.requested_backend == TicaBackend::ExactDense ||
              cfg.requested_backend == TicaBackend::ExactStreaming) && !dense_fits) {
    throw std::runtime_error("Requested exact covariance exceeds the memory budget.");
  }
  if (cfg.need_projection_cache && decision.selected_backend == TicaBackend::ExactDense)
    decision.selected_backend = TicaBackend::ExactStreaming;

  return decision;
}

#endif // TITAN_TICA_BACKEND_POLICY_H

#ifndef TITAN_PBCTOPO_EVIDENCE_SUPPORT_POLICY_H
#define TITAN_PBCTOPO_EVIDENCE_SUPPORT_POLICY_H

#include "component_graph_math.h"
#include "types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace titan_pbctopo {

inline constexpr std::uint32_t PBCTOPO_EVIDENCE_SUPPORT_POLICY_VERSION = 1;
inline constexpr std::string_view PBCTOPO_SCORED_PAIR_SUPPORT_POLICY =
    "pbctopo-scored-pair-decision-support-v1";
inline constexpr std::string_view PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_POLICY =
    "pbctopo-selected-compatibility-decision-support-v1";
inline constexpr double PBCTOPO_SCORED_PAIR_SUPPORT_THRESHOLD = 0.05;
inline constexpr double PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_THRESHOLD = 0.5;

enum class PbctopoEvidenceSupportReason : std::uint8_t {
  None,
  ExplicitRelation,
  NoCandidates,
  BestInvalid,
  BestIncomplete,
  BestUltraClash,
  NoContactPairs,
  NonfiniteContactFraction,
  NonpositiveContactFraction,
  NonfiniteRankConfidence,
  BelowThreshold,
  NoObservedHypothesis,
  NoCompatibleHypothesis,
};

inline constexpr std::string_view pbctopo_evidence_support_reason_name(
    PbctopoEvidenceSupportReason reason) noexcept {
  switch (reason) {
  case PbctopoEvidenceSupportReason::None:
    return "none";
  case PbctopoEvidenceSupportReason::ExplicitRelation:
    return "explicit_relation";
  case PbctopoEvidenceSupportReason::NoCandidates:
    return "no_candidates";
  case PbctopoEvidenceSupportReason::BestInvalid:
    return "best_invalid";
  case PbctopoEvidenceSupportReason::BestIncomplete:
    return "best_incomplete";
  case PbctopoEvidenceSupportReason::BestUltraClash:
    return "best_ultra_clash";
  case PbctopoEvidenceSupportReason::NoContactPairs:
    return "no_contact_pairs";
  case PbctopoEvidenceSupportReason::NonfiniteContactFraction:
    return "nonfinite_contact_fraction";
  case PbctopoEvidenceSupportReason::NonpositiveContactFraction:
    return "nonpositive_contact_fraction";
  case PbctopoEvidenceSupportReason::NonfiniteRankConfidence:
    return "nonfinite_rank_confidence";
  case PbctopoEvidenceSupportReason::BelowThreshold:
    return "below_threshold";
  case PbctopoEvidenceSupportReason::NoObservedHypothesis:
    return "no_observed_hypothesis";
  case PbctopoEvidenceSupportReason::NoCompatibleHypothesis:
    return "no_compatible_hypothesis";
  }
  return "unknown";
}

struct PbctopoScoredPairEvidenceSupport {
  std::uint32_t policy_version = PBCTOPO_EVIDENCE_SUPPORT_POLICY_VERSION;
  double support_primary = 0.0;
  double decision_threshold = PBCTOPO_SCORED_PAIR_SUPPORT_THRESHOLD;
  double contact_fraction = 0.0;
  double rank_confidence = 0.0;
  std::size_t candidate_count = 0;
  std::size_t contact_pairs = 0;
  bool evaluated = false;
  bool raw_valid = true;
  bool contact_fraction_defined = false;
  bool hard_gate_passed = false;
  bool active_at_threshold = false;
  PbctopoEvidenceSupportReason reason =
      PbctopoEvidenceSupportReason::NoCandidates;
};

inline PbctopoScoredPairEvidenceSupport
evaluate_pbctopo_scored_pair_evidence_support(
    const PbctopoPairShiftCandidateSet &pair_set) noexcept {
  PbctopoScoredPairEvidenceSupport result;
  result.evaluated = true;
  result.candidate_count = pair_set.candidates.size();
  if (pair_set.candidates.empty())
    return result;

  const auto &best = pair_set.candidates.front().score;
  result.contact_pairs = best.contact_pairs;
  if (std::isfinite(best.contact_fraction)) {
    result.contact_fraction = best.contact_fraction == 0.0
                                  ? 0.0
                                  : best.contact_fraction;
    result.contact_fraction_defined = true;
  }
  result.rank_confidence =
      pair_set.candidates.size() > 1
          ? pbctopo_pair_shift_confidence(
                best, pair_set.candidates[1].score)
          : 0.0;
  if (!std::isfinite(result.rank_confidence)) {
    result.rank_confidence = 0.0;
    result.raw_valid = false;
    result.reason = PbctopoEvidenceSupportReason::NonfiniteRankConfidence;
    return result;
  }
  if (!best.valid) {
    result.reason = PbctopoEvidenceSupportReason::BestInvalid;
    return result;
  }
  if (!best.complete) {
    result.reason = PbctopoEvidenceSupportReason::BestIncomplete;
    return result;
  }
  if (best.ultra_clash) {
    result.reason = PbctopoEvidenceSupportReason::BestUltraClash;
    return result;
  }
  if (best.contact_pairs == 0) {
    result.reason = PbctopoEvidenceSupportReason::NoContactPairs;
    return result;
  }
  if (!result.contact_fraction_defined) {
    result.raw_valid = false;
    result.reason = PbctopoEvidenceSupportReason::NonfiniteContactFraction;
    return result;
  }
  if (!(result.contact_fraction > 0.0)) {
    result.reason = PbctopoEvidenceSupportReason::NonpositiveContactFraction;
    return result;
  }

  result.hard_gate_passed = true;
  result.support_primary =
      std::max(result.rank_confidence, result.contact_fraction);
  result.active_at_threshold =
      result.support_primary >= result.decision_threshold;
  result.reason = result.active_at_threshold
                      ? PbctopoEvidenceSupportReason::None
                      : PbctopoEvidenceSupportReason::BelowThreshold;
  return result;
}

struct PbctopoSelectedCompatibilitySupport {
  std::uint32_t policy_version = PBCTOPO_EVIDENCE_SUPPORT_POLICY_VERSION;
  double support_primary = 0.0;
  double decision_threshold =
      PBCTOPO_SELECTED_COMPATIBILITY_SUPPORT_THRESHOLD;
  std::size_t observed_hypothesis_count = 0;
  std::size_t compatible_hypothesis_count = 0;
  std::uint64_t closest_shift_linf_distance = 0;
  bool evaluated = false;
  bool explicit_relation = false;
  bool compatibility_match = false;
  bool closest_shift_linf_distance_defined = false;
  bool active_at_threshold = false;
  PbctopoEvidenceSupportReason reason =
      PbctopoEvidenceSupportReason::NoObservedHypothesis;
};

inline constexpr std::uint64_t
pbctopo_ordered_int64_key(std::int64_t value) noexcept {
  return static_cast<std::uint64_t>(value) ^ (std::uint64_t{1} << 63);
}

inline constexpr std::uint64_t
pbctopo_abs_int64_difference(std::int64_t lhs, std::int64_t rhs) noexcept {
  const std::uint64_t a = pbctopo_ordered_int64_key(lhs);
  const std::uint64_t b = pbctopo_ordered_int64_key(rhs);
  return a >= b ? a - b : b - a;
}

inline constexpr std::uint64_t
pbctopo_lattice_linf_distance(const Int3 &lhs, const Int3 &rhs) noexcept {
  const std::uint64_t dx = pbctopo_abs_int64_difference(lhs.x, rhs.x);
  const std::uint64_t dy = pbctopo_abs_int64_difference(lhs.y, rhs.y);
  const std::uint64_t dz = pbctopo_abs_int64_difference(lhs.z, rhs.z);
  return std::max(dx, std::max(dy, dz));
}

inline PbctopoSelectedCompatibilitySupport
evaluate_pbctopo_selected_compatibility_support(
    bool explicit_relation, std::span<const Int3> observed_relative_shifts,
    const Int3 &selected_relative_shift) noexcept {
  PbctopoSelectedCompatibilitySupport result;
  result.evaluated = true;
  result.explicit_relation = explicit_relation;
  result.observed_hypothesis_count = observed_relative_shifts.size();

  for (const auto &observed_shift : observed_relative_shifts) {
    const std::uint64_t distance = pbctopo_lattice_linf_distance(
        observed_shift, selected_relative_shift);
    if (!result.closest_shift_linf_distance_defined ||
        distance < result.closest_shift_linf_distance) {
      result.closest_shift_linf_distance = distance;
      result.closest_shift_linf_distance_defined = true;
    }
    if (distance == 0)
      ++result.compatible_hypothesis_count;
  }

  result.compatibility_match = result.compatible_hypothesis_count != 0;
  result.active_at_threshold =
      result.explicit_relation || result.compatibility_match;
  result.support_primary = result.active_at_threshold ? 1.0 : 0.0;
  if (result.explicit_relation) {
    result.reason = PbctopoEvidenceSupportReason::ExplicitRelation;
  } else if (result.compatibility_match) {
    result.reason = PbctopoEvidenceSupportReason::None;
  } else if (observed_relative_shifts.empty()) {
    result.reason = PbctopoEvidenceSupportReason::NoObservedHypothesis;
  } else {
    result.reason = PbctopoEvidenceSupportReason::NoCompatibleHypothesis;
  }
  return result;
}

struct PbctopoSelectedEvidenceRelationAudit {
  std::size_t component_a = 0;
  std::size_t component_b = 0;
  std::uint64_t component_id_a = 0;
  std::uint64_t component_id_b = 0;
  bool evaluated = false;
  bool observed_hypothesis_relation = false;
  std::size_t compatible_contact_count = 0;
  std::size_t alternative_hypothesis_contact_count = 0;
  std::size_t compatible_contact_loss_count = 0;
  PbctopoSelectedCompatibilitySupport support;
};

} // namespace titan_pbctopo

#endif

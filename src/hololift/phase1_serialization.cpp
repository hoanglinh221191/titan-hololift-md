#include "phase1_serialization.h"

#include "durable_artifact.h"
#include "frame_binding.h"
#include "serialization.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <system_error>
#include <utility>

namespace titan_hololift {
namespace {

void write_json_string(std::ostream &out, std::string_view value) {
  out << '"';
  for (const char raw_byte : value) {
    const auto byte = static_cast<unsigned char>(raw_byte);
    switch (byte) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (byte < 0x20) {
        out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned int>(byte) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(byte);
      }
    }
  }
  out << '"';
}

std::string format_sha256_digest(std::span<const std::byte, 32> digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    const auto value = std::to_integer<unsigned int>(digest[index]);
    result[2 * index] = digits[value >> 4];
    result[2 * index + 1] = digits[value & 0x0fU];
  }
  return result;
}

void write_bool(std::ostream &out, bool value) {
  out << (value ? "true" : "false");
}

void write_number_or_null(std::ostream &out, double value) {
  if (std::isfinite(value))
    out << value;
  else
    out << "null";
}

void write_image(std::ostream &out, const HoloLiftLatticeImage &image) {
  out << '[' << image.x << ',' << image.y << ',' << image.z << ']';
}

void write_integer_matrix(std::ostream &out,
                          const std::array<std::int64_t, 9> &matrix) {
  out << '[';
  for (std::size_t index = 0; index < matrix.size(); ++index) {
    if (index != 0)
      out << ',';
    out << matrix[index];
  }
  out << ']';
}

std::string_view time_scaling_name(HoloLiftPhase1TimeScaling value) {
  switch (value) {
  case HoloLiftPhase1TimeScaling::PerFrameDisplacement:
    return "per_frame_displacement";
  case HoloLiftPhase1TimeScaling::Diffusive:
    return "diffusive";
  case HoloLiftPhase1TimeScaling::Ballistic:
    return "ballistic";
  }
  return "unknown";
}

std::string_view loss_name(HoloLiftPhase1Loss value) {
  switch (value) {
  case HoloLiftPhase1Loss::Quadratic:
    return "quadratic";
  }
  return "unknown";
}

std::string_view component_weighting_name(
    HoloLiftPhase1ComponentWeighting value) {
  switch (value) {
  case HoloLiftPhase1ComponentWeighting::AtomCount:
    return "atom_count";
  case HoloLiftPhase1ComponentWeighting::AtomicMass:
    return "atomic_mass";
  }
  return "unknown";
}

std::string_view basis_policy_name(HoloLiftLatticeBasisPolicy value) {
  switch (value) {
  case HoloLiftLatticeBasisPolicy::RequireBasisContinuous:
    return "require_basis_continuous";
  case HoloLiftLatticeBasisPolicy::TransportUnimodularBasis:
    return "transport_unimodular_basis_experimental";
  }
  return "unknown";
}

bool same_number(double lhs, double rhs) noexcept {
  return lhs == rhs || (std::isnan(lhs) && std::isnan(rhs));
}

bool same_phase1_config(const HoloLiftPhase1Config &lhs,
                        const HoloLiftPhase1Config &rhs) noexcept {
  return same_number(lhs.ordinal_rank_weight, rhs.ordinal_rank_weight) &&
         lhs.ordinal_rank_saturated == rhs.ordinal_rank_saturated &&
         lhs.evidence_absent_zero_rank == rhs.evidence_absent_zero_rank &&
         lhs.allow_policy_admitted_candidates == rhs.allow_policy_admitted_candidates &&
         same_number(lhs.unique_evidence_mismatch_weight,
                     rhs.unique_evidence_mismatch_weight) &&
         same_number(lhs.supported_evidence_mismatch_weight,
                     rhs.supported_evidence_mismatch_weight) &&
         same_number(lhs.physical_transition_weight,
                     rhs.physical_transition_weight) &&
         same_number(lhs.temporal_scale, rhs.temporal_scale) &&
         same_number(lhs.variance_floor_A2, rhs.variance_floor_A2) &&
         same_number(lhs.bounded_anchor_competitor_penalty,
                     rhs.bounded_anchor_competitor_penalty) &&
         same_number(lhs.relative_cost_tolerance,
                     rhs.relative_cost_tolerance) &&
         same_number(lhs.basis_transport_max_relative_mismatch,
                     rhs.basis_transport_max_relative_mismatch) &&
         same_number(lhs.basis_transport_ambiguity_tolerance,
                     rhs.basis_transport_ambiguity_tolerance) &&
         same_number(lhs.basis_transport_nonidentity_max_relative_mismatch,
                     rhs.basis_transport_nonidentity_max_relative_mismatch) &&
         same_number(lhs.basis_transport_minimum_improvement_ratio,
                     rhs.basis_transport_minimum_improvement_ratio) &&
         lhs.time_scaling == rhs.time_scaling && lhs.loss == rhs.loss &&
         lhs.lattice_basis_policy == rhs.lattice_basis_policy &&
         lhs.dp_worker_count == rhs.dp_worker_count &&
         lhs.pin_strong_bounded_anchors == rhs.pin_strong_bounded_anchors;
}

bool same_algorithmic_frame(const HoloLiftPhase1FrameSelection &lhs,
                            const HoloLiftPhase1FrameSelection &rhs) noexcept {
  return lhs.use == rhs.use &&
         lhs.preceding_transition_gap == rhs.preceding_transition_gap &&
         lhs.segment_index == rhs.segment_index &&
         lhs.candidate_index == rhs.candidate_index &&
         same_number(lhs.emission_cost, rhs.emission_cost) &&
         same_number(lhs.transition_cost, rhs.transition_cost) &&
         same_number(lhs.cumulative_segment_cost,
                     rhs.cumulative_segment_cost) &&
         lhs.selected_global_gauge_increment ==
             rhs.selected_global_gauge_increment &&
         lhs.cumulative_global_gauge == rhs.cumulative_global_gauge &&
         lhs.cumulative_global_gauge_input_basis ==
             rhs.cumulative_global_gauge_input_basis &&
         lhs.lattice_basis_transport_to_segment_reference ==
             rhs.lattice_basis_transport_to_segment_reference &&
         same_number(lhs.component_displacement_rms_A,
                     rhs.component_displacement_rms_A) &&
         same_number(lhs.component_displacement_max_A,
                     rhs.component_displacement_max_A) &&
         same_number(lhs.component_injectivity_radius_lower_bound_A,
                     rhs.component_injectivity_radius_lower_bound_A) &&
         same_number(lhs.component_injectivity_margin_A,
                     rhs.component_injectivity_margin_A) &&
         lhs.component_nearest_image_violations ==
             rhs.component_nearest_image_violations &&
         lhs.component_injectivity_uncertified ==
             rhs.component_injectivity_uncertified &&
         same_number(lhs.lattice_basis_transport_relative_mismatch,
                     rhs.lattice_basis_transport_relative_mismatch) &&
         same_number(lhs.lattice_basis_transport_identity_relative_mismatch,
                     rhs.lattice_basis_transport_identity_relative_mismatch) &&
         same_number(lhs.lattice_basis_transport_improvement_ratio,
                     rhs.lattice_basis_transport_improvement_ratio) &&
         lhs.global_gauge_increment_ambiguous ==
             rhs.global_gauge_increment_ambiguous &&
         lhs.lattice_basis_transport_applied ==
             rhs.lattice_basis_transport_applied &&
         lhs.lattice_basis_transport_ambiguous ==
             rhs.lattice_basis_transport_ambiguous &&
         lhs.lattice_basis_transport_identity_fast_path ==
             rhs.lattice_basis_transport_identity_fast_path &&
         lhs.lattice_basis_transport_local_search_completed ==
             rhs.lattice_basis_transport_local_search_completed &&
         lhs.lattice_basis_transport_accepted_under_policy ==
             rhs.lattice_basis_transport_accepted_under_policy &&
         lhs.lattice_basis_transport_explicit_remap_provenance_available ==
             rhs.lattice_basis_transport_explicit_remap_provenance_available &&
         lhs.lattice_basis_transport_provenance_sufficient ==
             rhs.lattice_basis_transport_provenance_sufficient &&
         lhs.quadratic_direct_residual_fallback ==
             rhs.quadratic_direct_residual_fallback &&
         lhs.pinned_strong_bounded_anchor ==
             rhs.pinned_strong_bounded_anchor &&
         lhs.changed_from_framewise == rhs.changed_from_framewise &&
         lhs.selected_candidate_provider_unsupported ==
             rhs.selected_candidate_provider_unsupported &&
         lhs.preceding_transition_provider_unsupported ==
             rhs.preceding_transition_provider_unsupported;
}

bool same_algorithmic_segment(const HoloLiftPhase1Segment &lhs,
                              const HoloLiftPhase1Segment &rhs) noexcept {
  return lhs.frame_begin == rhs.frame_begin &&
         lhs.frame_count == rhs.frame_count &&
         lhs.topology_epoch_index == rhs.topology_epoch_index &&
         same_number(lhs.objective, rhs.objective) &&
         same_number(lhs.second_best_objective,
                     rhs.second_best_objective) &&
         same_number(lhs.absolute_path_gap, rhs.absolute_path_gap) &&
         same_number(lhs.relative_path_gap, rhs.relative_path_gap) &&
         lhs.optimal_path_count_capped == rhs.optimal_path_count_capped &&
         lhs.temporal_transition_count == rhs.temporal_transition_count &&
         lhs.provider_unsupported_selected_candidate_count ==
             rhs.provider_unsupported_selected_candidate_count &&
         lhs.provider_unsupported_transition_count ==
             rhs.provider_unsupported_transition_count &&
         lhs.ambiguous_gauge_transition_count ==
             rhs.ambiguous_gauge_transition_count &&
         lhs.component_nearest_image_violations ==
             rhs.component_nearest_image_violations &&
         lhs.component_injectivity_uncertified ==
             rhs.component_injectivity_uncertified &&
         lhs.gauge_replay_arithmetic_residuals ==
             rhs.gauge_replay_arithmetic_residuals &&
         lhs.lattice_basis_transports_applied ==
             rhs.lattice_basis_transports_applied &&
         lhs.quadratic_direct_residual_fallbacks ==
             rhs.quadratic_direct_residual_fallbacks &&
         lhs.pretransformed_candidate_component_images ==
             rhs.pretransformed_candidate_component_images &&
         lhs.peak_layer_state_count == rhs.peak_layer_state_count &&
         lhs.compact_backpointer_bytes_estimate ==
             rhs.compact_backpointer_bytes_estimate &&
          same_number(lhs.maximum_lattice_basis_transport_relative_mismatch,
                      rhs.maximum_lattice_basis_transport_relative_mismatch) &&
          lhs.component_weighting == rhs.component_weighting &&
          lhs.exact_on_retained_graph == rhs.exact_on_retained_graph &&
          lhs.optimal_path_proven_on_complete_bounded_domain ==
              rhs.optimal_path_proven_on_complete_bounded_domain &&
          lhs.unique_on_configured_retained_state_graph ==
             rhs.unique_on_configured_retained_state_graph &&
         lhs.retained_candidate_bands_complete ==
             rhs.retained_candidate_bands_complete &&
         lhs.bounded_domain_only == rhs.bounded_domain_only &&
         lhs.anchors_hard_pinned == rhs.anchors_hard_pinned &&
         lhs.all_global_gauge_steps_solved ==
             rhs.all_global_gauge_steps_solved &&
         lhs.lattice_basis_transport_accepted_under_policy ==
             rhs.lattice_basis_transport_accepted_under_policy &&
         lhs.lattice_basis_transport_provenance_sufficient ==
             rhs.lattice_basis_transport_provenance_sufficient &&
         lhs.local_transport_search_completed ==
             rhs.local_transport_search_completed &&
         lhs.explicit_remap_provenance_available ==
             rhs.explicit_remap_provenance_available &&
         lhs.component_representative_injectivity_audit_passed ==
             rhs.component_representative_injectivity_audit_passed &&
         lhs.gauge_replay_arithmetic_closed ==
             rhs.gauge_replay_arithmetic_closed &&
         lhs.selected_transition_time_reversal_audited ==
             rhs.selected_transition_time_reversal_audited &&
         lhs.selected_transition_time_reversal_consistent ==
             rhs.selected_transition_time_reversal_consistent;
}

bool same_algorithmic_result(const HoloLiftPhase1Result &lhs,
                             const HoloLiftPhase1Result &rhs) noexcept {
  if (lhs.api_version != rhs.api_version ||
      !same_phase1_config(lhs.config, rhs.config) ||
      lhs.frames.size() != rhs.frames.size() ||
      lhs.segments.size() != rhs.segments.size() ||
      !same_number(lhs.objective, rhs.objective) ||
      lhs.temporal_observation_frames != rhs.temporal_observation_frames ||
      lhs.observation_gap_frames != rhs.observation_gap_frames ||
      lhs.changed_from_framewise_frames != rhs.changed_from_framewise_frames ||
      lhs.basis_transport_transition_gaps !=
          rhs.basis_transport_transition_gaps ||
      lhs.ambiguous_basis_transport_transition_gaps !=
          rhs.ambiguous_basis_transport_transition_gaps ||
      lhs.basis_discontinuity_transition_gaps !=
          rhs.basis_discontinuity_transition_gaps ||
      lhs.basis_mismatch_transition_gaps !=
          rhs.basis_mismatch_transition_gaps ||
      lhs.quadratic_direct_residual_fallbacks !=
          rhs.quadratic_direct_residual_fallbacks ||
      lhs.pretransformed_candidate_component_images !=
          rhs.pretransformed_candidate_component_images ||
      lhs.source_frame_count != rhs.source_frame_count ||
      lhs.imported_observation_frame_count !=
          rhs.imported_observation_frame_count ||
      lhs.imported_observation_coverage_complete !=
          rhs.imported_observation_coverage_complete ||
      lhs.source_trajectory_coverage_known !=
          rhs.source_trajectory_coverage_known ||
      lhs.source_trajectory_coverage_complete !=
          rhs.source_trajectory_coverage_complete ||
      lhs.temporal_transition_coverage_complete !=
          rhs.temporal_transition_coverage_complete ||
      lhs.temporal_coverage_complete != rhs.temporal_coverage_complete ||
      lhs.all_segment_objectives_unique !=
          rhs.all_segment_objectives_unique ||
      lhs.all_observation_bands_complete !=
          rhs.all_observation_bands_complete ||
      lhs.exact_on_retained_graph != rhs.exact_on_retained_graph) {
    return false;
  }
  for (std::size_t index = 0; index < lhs.frames.size(); ++index) {
    if (!same_algorithmic_frame(lhs.frames[index], rhs.frames[index]))
      return false;
  }
  for (std::size_t index = 0; index < lhs.segments.size(); ++index) {
    if (!same_algorithmic_segment(lhs.segments[index], rhs.segments[index]))
      return false;
  }
  const auto &lc = lhs.certificate;
  const auto &rc = rhs.certificate;
  return lc.all_frames_trajectory_bound == rc.all_frames_trajectory_bound &&
         lc.all_spatial_lifts_replayed == rc.all_spatial_lifts_replayed &&
         lc.all_selected_candidates_hard_feasible ==
             rc.all_selected_candidates_hard_feasible &&
         lc.all_global_gauge_steps_solved ==
             rc.all_global_gauge_steps_solved &&
         lc.lattice_basis_transport_accepted_under_policy ==
             rc.lattice_basis_transport_accepted_under_policy &&
         lc.lattice_basis_transport_provenance_sufficient ==
             rc.lattice_basis_transport_provenance_sufficient &&
         lc.local_transport_search_completed ==
             rc.local_transport_search_completed &&
         lc.explicit_remap_provenance_available ==
             rc.explicit_remap_provenance_available &&
         lc.component_representative_injectivity_audit_passed ==
             rc.component_representative_injectivity_audit_passed &&
         lc.gauge_replay_arithmetic_closed ==
             rc.gauge_replay_arithmetic_closed &&
         lc.source_trajectory_coverage_complete ==
             rc.source_trajectory_coverage_complete &&
         lc.retained_candidate_bands_complete ==
             rc.retained_candidate_bands_complete &&
         lc.optimal_path_exact_on_retained_graph ==
             rc.optimal_path_exact_on_retained_graph &&
         lc.optimal_path_proven_on_complete_bounded_domain ==
             rc.optimal_path_proven_on_complete_bounded_domain &&
         lc.unique_on_configured_retained_state_graph ==
             rc.unique_on_configured_retained_state_graph &&
         lc.selected_transition_time_reversal_audited ==
             rc.selected_transition_time_reversal_audited &&
         lc.selected_transition_time_reversal_consistent ==
             rc.selected_transition_time_reversal_consistent &&
         lc.robust_loss_exact == rc.robust_loss_exact;
}

std::expected<void, std::string> validate_result_binding(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1Result &result) {
  const auto replay = solve_hololift_phase1_temporal_path(
      store, prepared_frames, result.config);
  if (!replay)
    return std::unexpected("Phase 1 result replay failed: " + replay.error());
  if (!same_algorithmic_result(result, *replay)) {
    return std::unexpected(
        "Phase 1 result transport/path/gauge/objective replay mismatch");
  }
  if (result.api_version != HOLOLIFT_PHASE1_API_VERSION ||
      result.frames.size() != store.frames().size() ||
      !std::isfinite(result.objective)) {
    return std::unexpected("Phase 1 result/store binding is invalid");
  }
  std::size_t temporal_observation_frames = 0;
  std::size_t observation_gap_frames = 0;
  std::size_t changed_from_framewise_frames = 0;
  std::size_t basis_transport_transition_gaps = 0;
  std::size_t ambiguous_basis_transport_transition_gaps = 0;
  std::size_t basis_discontinuity_transition_gaps = 0;
  std::size_t basis_mismatch_transition_gaps = 0;
  bool all_selected_hard_feasible = true;
  for (std::size_t frame_index = 0; frame_index < result.frames.size();
       ++frame_index) {
    const auto &selection = result.frames[frame_index];
    if (selection.use != hololift_phase1_frame_use(store.frames()[frame_index]))
      return std::unexpected("Phase 1 frame-use classification is inconsistent");
    if (selection.changed_from_framewise)
      ++changed_from_framewise_frames;
    if (selection.preceding_transition_gap !=
        HoloLiftPhase1TransitionGap::None) {
      ++basis_transport_transition_gaps;
      if (selection.preceding_transition_gap ==
          HoloLiftPhase1TransitionGap::AmbiguousBasisTransport) {
        ++ambiguous_basis_transport_transition_gaps;
      } else if (selection.preceding_transition_gap ==
                 HoloLiftPhase1TransitionGap::BasisDiscontinuity) {
        ++basis_discontinuity_transition_gaps;
      } else {
        ++basis_mismatch_transition_gaps;
      }
    }
    if (selection.use != HoloLiftPhase1FrameUse::TemporalObservation) {
      ++observation_gap_frames;
      continue;
    }
    ++temporal_observation_frames;
    const auto &frame = store.frames()[frame_index];
    if (selection.candidate_index < frame.candidates.begin ||
        selection.candidate_index >=
            frame.candidates.begin + frame.candidates.count ||
        !std::isfinite(selection.lattice_basis_transport_relative_mismatch) ||
        !std::isfinite(selection.maximum_atom_peculiar_displacement_A)) {
      return std::unexpected(
          "Phase 1 selected candidate is outside its frame band");
    }
    const auto transport = make_hololift_lattice_basis_transport(
        selection.lattice_basis_transport_to_segment_reference,
        frame.box_matrix);
    const auto input_gauge =
        transport ? hololift_lattice_image_from_reference(
                        selection.cumulative_global_gauge, *transport)
                  : std::expected<HoloLiftLatticeImage, std::string>(
                        std::unexpected(transport.error()));
    if (!input_gauge ||
        *input_gauge != selection.cumulative_global_gauge_input_basis) {
      return std::unexpected(
          "Phase 1 selected lattice-basis gauge is not replayable");
    }
    all_selected_hard_feasible =
        all_selected_hard_feasible &&
        store.candidates()[selection.candidate_index].evidence.hard_feasible;
  }

  const auto &source_coverage = store.source_coverage();
  if (temporal_observation_frames == 0)
    all_selected_hard_feasible = false;
  const bool expected_imported_coverage =
      observation_gap_frames == 0 &&
      temporal_observation_frames == store.frames().size();
  const bool expected_source_coverage =
      source_coverage.complete() && expected_imported_coverage;
  const bool expected_transition_coverage =
      expected_source_coverage && result.segments.size() == 1 &&
      !result.segments.empty();
  if (result.temporal_observation_frames != temporal_observation_frames ||
      result.observation_gap_frames != observation_gap_frames ||
      result.changed_from_framewise_frames !=
          changed_from_framewise_frames ||
      result.basis_transport_transition_gaps !=
          basis_transport_transition_gaps ||
      result.ambiguous_basis_transport_transition_gaps !=
          ambiguous_basis_transport_transition_gaps ||
      result.basis_discontinuity_transition_gaps !=
          basis_discontinuity_transition_gaps ||
      result.basis_mismatch_transition_gaps !=
          basis_mismatch_transition_gaps ||
      result.source_frame_count != source_coverage.source_frame_count ||
      result.imported_observation_frame_count != store.frames().size() ||
      result.imported_observation_coverage_complete !=
          expected_imported_coverage ||
      result.source_trajectory_coverage_known != source_coverage.known() ||
      result.source_trajectory_coverage_complete != expected_source_coverage ||
      result.temporal_transition_coverage_complete !=
          expected_transition_coverage ||
      result.temporal_coverage_complete != expected_transition_coverage) {
    return std::unexpected("Phase 1 result coverage fields are inconsistent");
  }

  bool all_gauge_steps_solved = !result.segments.empty();
  bool all_basis_transports_accepted = !result.segments.empty();
  bool all_basis_transport_provenance_sufficient =
      !result.segments.empty();
  bool all_local_transport_searches_completed = !result.segments.empty();
  bool any_explicit_remap_provenance = false;
  bool all_component_injectivity_passed = !result.segments.empty();
  bool all_gauge_replay_closed = !result.segments.empty();
  bool all_atom_injectivity_audited = !result.segments.empty();
  bool all_atom_injectivity_passed = !result.segments.empty();
  bool all_independent_cochains_audited = !result.segments.empty();
  bool all_independent_cochains_closed = !result.segments.empty();
  bool all_reversal_audited = !result.segments.empty();
  bool all_reversal_consistent = !result.segments.empty();
  bool all_segments_materialized = !result.segments.empty();
  bool all_segments_bounded_domain_only = !result.segments.empty();
  bool all_segments_exact_on_retained_graph = !result.segments.empty();
  bool all_segment_objectives_unique = !result.segments.empty();
  bool all_observation_bands_complete = !result.segments.empty();
  for (const auto &segment : result.segments) {
    if (segment.frame_begin > result.frames.size() ||
        segment.frame_count > result.frames.size() - segment.frame_begin)
      return std::unexpected("Phase 1 result segment range is invalid");
    std::size_t unsupported_candidates = 0;
    std::size_t unsupported_transitions = 0;
    bool previous_unsupported = false;
    for (std::size_t row = 0; row < segment.frame_count; ++row) {
      const auto &selection = result.frames[segment.frame_begin + row];
      if (selection.candidate_index >= store.candidates().size())
        return std::unexpected("Phase 1 result candidate is invalid");
      const bool unsupported =
          !store.candidates()[selection.candidate_index].evidence.hard_feasible;
      const bool transition_unsupported = row > 0 &&
          (unsupported || previous_unsupported);
      if (selection.selected_candidate_provider_unsupported != unsupported ||
          selection.preceding_transition_provider_unsupported != transition_unsupported)
        return std::unexpected("Phase 1 provider-support flags are inconsistent");
      unsupported_candidates += unsupported ? 1 : 0;
      unsupported_transitions += transition_unsupported ? 1 : 0;
      previous_unsupported = unsupported;
    }
    const bool expected_atom_pass =
        segment.atom_temporal_injectivity_audited &&
        segment.atom_temporal_nearest_image_violations == 0 &&
        segment.atom_temporal_injectivity_uncertified == 0 &&
        segment.independent_temporal_image_ambiguities == 0;
    const bool expected_independent_closed =
        segment.independent_space_time_cochain_audited &&
        segment.independent_temporal_image_ambiguities == 0 &&
        segment.independent_space_time_curvature_residuals == 0;
    const bool expected_local_certificate =
        unsupported_candidates == 0 && unsupported_transitions == 0 &&
        segment.coordinates_materialized_and_replay_consistent &&
        segment.all_global_gauge_steps_solved &&
        segment.lattice_basis_transport_accepted_under_policy &&
        segment.lattice_basis_transport_provenance_sufficient &&
        segment.temporal_transition_count > 0 &&
        segment.ambiguous_gauge_transition_count == 0 &&
        segment.component_nearest_image_violations == 0 &&
        segment.component_injectivity_uncertified == 0 &&
        segment.component_representative_injectivity_audit_passed &&
        segment.gauge_replay_arithmetic_closed && expected_atom_pass &&
        segment.independent_space_time_cochain_audited &&
        expected_independent_closed &&
        segment.selected_transition_time_reversal_audited &&
        segment.selected_transition_time_reversal_consistent;
    const bool expected_complete_domain_optimality =
        segment.exact_on_retained_graph &&
        segment.retained_candidate_bands_complete &&
        segment.bounded_domain_only;
    if (segment.frame_begin > result.frames.size() ||
        segment.frame_count > result.frames.size() - segment.frame_begin ||
        !std::isfinite(segment.objective) ||
        segment.provider_unsupported_selected_candidate_count != unsupported_candidates ||
        segment.provider_unsupported_transition_count != unsupported_transitions ||
        !std::isfinite(
            segment.maximum_lattice_basis_transport_relative_mismatch) ||
        !std::isfinite(segment.maximum_atom_peculiar_displacement_A) ||
        segment.coordinates_materialized_and_replay_consistent !=
            segment.all_atom_reconstruction_audited ||
        segment.atom_temporal_injectivity_audit_passed !=
            expected_atom_pass ||
        segment.independent_space_time_cochain_closed !=
            expected_independent_closed ||
        segment.optimal_path_proven_on_complete_bounded_domain !=
            expected_complete_domain_optimality ||
        segment.temporal_lift_locally_certified !=
            expected_local_certificate) {
      return std::unexpected("Phase 1 result segment is invalid");
    }
    all_gauge_steps_solved &= segment.all_global_gauge_steps_solved;
    all_basis_transports_accepted &=
        segment.lattice_basis_transport_accepted_under_policy;
    all_basis_transport_provenance_sufficient &=
        segment.lattice_basis_transport_provenance_sufficient;
    all_local_transport_searches_completed &=
        segment.local_transport_search_completed;
    any_explicit_remap_provenance |=
        segment.explicit_remap_provenance_available;
    all_component_injectivity_passed &=
        segment.component_representative_injectivity_audit_passed;
    all_gauge_replay_closed &= segment.gauge_replay_arithmetic_closed;
    all_atom_injectivity_audited &=
        segment.atom_temporal_injectivity_audited;
    all_atom_injectivity_passed &=
        segment.atom_temporal_injectivity_audit_passed;
    all_independent_cochains_audited &=
        segment.independent_space_time_cochain_audited;
    all_independent_cochains_closed &=
        segment.independent_space_time_cochain_closed;
    all_reversal_audited &=
        segment.selected_transition_time_reversal_audited;
    all_reversal_consistent &=
        segment.selected_transition_time_reversal_consistent;
    all_segments_materialized &=
        segment.coordinates_materialized_and_replay_consistent;
    all_segments_bounded_domain_only &= segment.bounded_domain_only;
    all_segments_exact_on_retained_graph &= segment.exact_on_retained_graph;
    all_segment_objectives_unique &=
        segment.unique_on_configured_retained_state_graph;
    all_observation_bands_complete &=
        segment.retained_candidate_bands_complete;
  }
  const auto &certificate = result.certificate;
  const bool expected_materialized =
      result.reconstruction_audit.selected_frames_complete &&
      all_segments_materialized;
  const bool expected_local_certificate =
      result.temporal_transition_coverage_complete &&
      std::all_of(result.segments.begin(), result.segments.end(),
                  [](const auto &segment) {
                    return segment.temporal_lift_locally_certified;
                  });
  const bool expected_complete_domain_optimality =
      result.exact_on_retained_graph &&
      result.all_observation_bands_complete &&
      all_segments_bounded_domain_only && result.temporal_coverage_complete;
  const bool expected_complete_domain_trajectory_claim_eligible =
      expected_local_certificate &&
      certificate.all_frames_trajectory_bound &&
      certificate.all_spatial_lifts_replayed &&
      certificate.all_selected_candidates_hard_feasible &&
      result.source_trajectory_coverage_complete &&
      result.all_observation_bands_complete && result.exact_on_retained_graph &&
      expected_complete_domain_optimality;
  if (result.exact_on_retained_graph !=
          all_segments_exact_on_retained_graph ||
      result.all_segment_objectives_unique != all_segment_objectives_unique ||
      result.all_observation_bands_complete !=
          all_observation_bands_complete ||
      certificate.all_selected_candidates_hard_feasible !=
          all_selected_hard_feasible ||
      certificate.all_atom_reconstruction_audited !=
          result.reconstruction_audit.selected_frames_complete ||
      certificate.coordinates_materialized_and_replay_consistent !=
          expected_materialized ||
      certificate.all_global_gauge_steps_solved != all_gauge_steps_solved ||
      certificate.lattice_basis_transport_accepted_under_policy !=
          all_basis_transports_accepted ||
      certificate.lattice_basis_transport_provenance_sufficient !=
          all_basis_transport_provenance_sufficient ||
      certificate.local_transport_search_completed !=
          all_local_transport_searches_completed ||
      certificate.explicit_remap_provenance_available !=
          any_explicit_remap_provenance ||
      certificate.component_representative_injectivity_audit_passed !=
          all_component_injectivity_passed ||
      certificate.gauge_replay_arithmetic_closed !=
          all_gauge_replay_closed ||
      certificate.atom_temporal_injectivity_audited !=
          all_atom_injectivity_audited ||
      certificate.atom_temporal_injectivity_audit_passed !=
          all_atom_injectivity_passed ||
      certificate.independent_space_time_cochain_audited !=
          all_independent_cochains_audited ||
      certificate.independent_space_time_cochain_closed !=
          all_independent_cochains_closed ||
      certificate.selected_transition_time_reversal_audited !=
          all_reversal_audited ||
      certificate.selected_transition_time_reversal_consistent !=
          (all_reversal_audited && all_reversal_consistent) ||
       certificate.temporal_lift_locally_certified !=
           expected_local_certificate ||
      certificate.complete_domain_trajectory_claim_eligible !=
          expected_complete_domain_trajectory_claim_eligible ||
      certificate.source_trajectory_coverage_complete !=
          result.source_trajectory_coverage_complete ||
      certificate.retained_candidate_bands_complete !=
          result.all_observation_bands_complete ||
      certificate.optimal_path_exact_on_retained_graph !=
          result.exact_on_retained_graph ||
      certificate.optimal_path_proven_on_complete_bounded_domain !=
          expected_complete_domain_optimality ||
      certificate.unique_on_configured_retained_state_graph !=
          result.all_segment_objectives_unique ||
      certificate.robust_loss_exact !=
          (result.config.loss == HoloLiftPhase1Loss::Quadratic)) {
    return std::unexpected(
        "Phase 1 reconstruction certificate is internally inconsistent");
  }
  return {};
}

} // namespace

HoloLiftArtifactWriteResult write_hololift_phase1_result_json(
    const HoloLiftObservationStore &store,
    std::span<const HoloLiftPhase1PreparedFrame> prepared_frames,
    const HoloLiftPhase1AuditedResult &audited_result,
    const std::filesystem::path &path) {
  const auto not_committed = [](std::string message)
      -> HoloLiftArtifactWriteResult {
    return detail::artifact_write_failure(
        HoloLiftArtifactCommitState::NotCommitted, std::move(message));
  };
  const auto sealed = validate_hololift_phase1_audited_result(
      store, prepared_frames, audited_result);
  if (!sealed)
    return not_committed(sealed.error());
  const auto *result_ptr = audited_result.result_if_valid();
  if (result_ptr == nullptr)
    return not_committed("Phase 1 audited result is moved-from");
  const auto &result = *result_ptr;
  const auto valid = validate_result_binding(store, prepared_frames, result);
  if (!valid)
    return not_committed(valid.error());
  const auto phase0_payload_sha256 =
      hololift_observation_store_payload_sha256(store);
  if (!phase0_payload_sha256)
    return not_committed(phase0_payload_sha256.error());

  static std::atomic<std::uint64_t> counter{0};
  std::filesystem::path temporary = path;
  temporary += ".tmp." +
               std::to_string(static_cast<std::uint64_t>(
                   std::chrono::steady_clock::now().time_since_epoch().count())) +
               "." + std::to_string(counter.fetch_add(1));
  const auto cleanup_temporary = [&] {
    std::error_code ignored;
    if (std::filesystem::exists(temporary, ignored) && !ignored)
      std::filesystem::remove(temporary, ignored);
  };
  const auto fail_not_committed = [&](std::string message)
      -> HoloLiftArtifactWriteResult {
    cleanup_temporary();
    return not_committed(std::move(message));
  };

  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out)
    return fail_not_committed(
        "cannot open temporary HoloLift Phase 1 result");
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\n  \"format\":\"titan-hololift-phase1-result\",\n";
  out << "  \"format_version\":"
      << HOLOLIFT_PHASE1_RESULT_FORMAT_VERSION << ",\n";
  out << "  \"phase1_api_version\":" << result.api_version << ",\n";
  out << "  \"audit_seal_version\":"
      << audited_result.audit_seal_version() << ",\n";
  out << "  \"trajectory_audit_digest_algorithm\":\"sha256-256\",\n";
  out << "  \"trajectory_audit_digest_sha256\":";
  write_json_string(
      out, format_sha256_digest(
               audited_result.trajectory_audit_digest_sha256()));
  out << ",\n";
  out << "  \"source\":{\"producer_version\":";
  write_json_string(out, store.source_contract().producer_version);
  out << ",\"package_id\":";
  write_json_string(out, store.source_contract().package_id);
  out << ",\"trajectory_id\":";
  write_json_string(out, store.source_contract().trajectory_id);
  out << ",\"phase0_payload_sha256\":";
  write_json_string(out, *phase0_payload_sha256);
  out << "},\n";

  const auto &config = result.config;
  out << "  \"config\":{\"ordinal_rank_weight\":";
  write_number_or_null(out, config.ordinal_rank_weight);
  out << ",\"ordinal_rank_saturated\":";
  write_bool(out, config.ordinal_rank_saturated);
  out << ",\"evidence_absent_zero_rank\":";
  write_bool(out, config.evidence_absent_zero_rank);
  out << ",\"allow_policy_admitted_candidates\":";
  write_bool(out, config.allow_policy_admitted_candidates);
  out << ",\"unique_evidence_mismatch_weight\":";
  write_number_or_null(out, config.unique_evidence_mismatch_weight);
  // Written only when set, like its digest entry, so records of runs without
  // it are unchanged.
  if (config.supported_evidence_mismatch_weight != 0.0) {
    out << ",\"supported_evidence_mismatch_weight\":";
    write_number_or_null(out, config.supported_evidence_mismatch_weight);
  }
  out << ",\"physical_transition_weight\":";
  write_number_or_null(out, config.physical_transition_weight);
  out << ",\"temporal_scale\":";
  write_number_or_null(out, config.temporal_scale);
  out << ",\"variance_floor_A2\":";
  write_number_or_null(out, config.variance_floor_A2);
  out << ",\"bounded_anchor_competitor_penalty\":";
  write_number_or_null(out, config.bounded_anchor_competitor_penalty);
  out << ",\"relative_cost_tolerance\":";
  write_number_or_null(out, config.relative_cost_tolerance);
  out << ",\"basis_transport_max_relative_mismatch\":";
  write_number_or_null(out, config.basis_transport_max_relative_mismatch);
  out << ",\"basis_transport_ambiguity_tolerance\":";
  write_number_or_null(out, config.basis_transport_ambiguity_tolerance);
  out << ",\"basis_transport_nonidentity_max_relative_mismatch\":";
  write_number_or_null(
      out, config.basis_transport_nonidentity_max_relative_mismatch);
  out << ",\"basis_transport_minimum_improvement_ratio\":";
  write_number_or_null(out,
                       config.basis_transport_minimum_improvement_ratio);
  out << ",\"lattice_basis_policy\":";
  write_json_string(out, basis_policy_name(config.lattice_basis_policy));
  out << ",\"time_scaling\":";
  write_json_string(out, time_scaling_name(config.time_scaling));
  out << ",\"loss\":";
  write_json_string(out, loss_name(config.loss));
  out << ",\"dp_worker_count\":" << config.dp_worker_count;
  out << ",\"pin_strong_bounded_anchors\":";
  write_bool(out, config.pin_strong_bounded_anchors);
  out << "},\n";

  out << "  \"summary\":{\"objective\":";
  write_number_or_null(out, result.objective);
  out << ",\"source_frame_count\":" << result.source_frame_count
      << ",\"imported_observation_frame_count\":"
      << result.imported_observation_frame_count
      << ",\"temporal_observation_frames\":"
      << result.temporal_observation_frames
      << ",\"observation_gap_frames\":" << result.observation_gap_frames
      << ",\"changed_from_framewise_frames\":"
      << result.changed_from_framewise_frames
      << ",\"basis_transport_transition_gaps\":"
      << result.basis_transport_transition_gaps
      << ",\"ambiguous_basis_transport_transition_gaps\":"
      << result.ambiguous_basis_transport_transition_gaps
      << ",\"basis_discontinuity_transition_gaps\":"
      << result.basis_discontinuity_transition_gaps
      << ",\"basis_mismatch_transition_gaps\":"
      << result.basis_mismatch_transition_gaps
      << ",\"quadratic_direct_residual_fallbacks\":"
      << result.quadratic_direct_residual_fallbacks
      << ",\"pretransformed_candidate_component_images\":"
      << result.pretransformed_candidate_component_images
      << ",\"imported_observation_coverage_complete\":";
  write_bool(out, result.imported_observation_coverage_complete);
  out << ",\"temporal_coverage_complete\":";
  write_bool(out, result.temporal_coverage_complete);
  out << ",\"all_segment_objectives_unique\":";
  write_bool(out, result.all_segment_objectives_unique);
  out << ",\"all_observation_bands_complete\":";
  write_bool(out, result.all_observation_bands_complete);
  out << ",\"source_trajectory_coverage_known\":";
  write_bool(out, result.source_trajectory_coverage_known);
  out << ",\"source_trajectory_coverage_complete\":";
  write_bool(out, result.source_trajectory_coverage_complete);
  out << ",\"temporal_transition_coverage_complete\":";
  write_bool(out, result.temporal_transition_coverage_complete);
  out << ",\"exact_on_retained_graph\":";
  write_bool(out, result.exact_on_retained_graph);
  out << ",\"reconstruction_temporal_frames_audited\":"
      << result.reconstruction_audit.temporal_frames_audited
      << ",\"reconstruction_atoms_audited\":"
      << result.reconstruction_audit.atoms_audited
      << ",\"atom_temporal_transitions_audited\":"
      << result.reconstruction_audit.atom_temporal_transitions_audited
      << ",\"atom_temporal_nearest_image_violations\":"
      << result.reconstruction_audit.atom_temporal_nearest_image_violations
      << ",\"atom_temporal_injectivity_uncertified\":"
      << result.reconstruction_audit.atom_temporal_injectivity_uncertified
      << ",\"independent_temporal_image_ambiguities\":"
      << result.reconstruction_audit.independent_temporal_image_ambiguities
      << ",\"independent_space_time_edges_audited\":"
      << result.reconstruction_audit.independent_space_time_edges_audited
      << ",\"independent_space_time_curvature_residuals\":"
      << result.reconstruction_audit
             .independent_space_time_curvature_residuals
      << ",\"reconstruction_maximum_component_representative_error_A\":";
  write_number_or_null(
      out,
      result.reconstruction_audit.maximum_component_representative_error_A);
  out << ",\"reconstruction_selected_frames_complete\":";
  write_bool(out, result.reconstruction_audit.selected_frames_complete);
  out << ",\"maximum_atom_peculiar_displacement_A\":";
  write_number_or_null(
      out, result.reconstruction_audit.maximum_atom_peculiar_displacement_A);
  out << ",\"atom_temporal_injectivity_audited\":";
  write_bool(out,
             result.reconstruction_audit.atom_temporal_injectivity_audited);
  out << ",\"independent_space_time_cochain_audited\":";
  write_bool(out, result.reconstruction_audit
                      .independent_space_time_cochain_audited);
  out << "},\n";

  const auto &certificate = result.certificate;
  out << "  \"certificate\":{\"all_frames_trajectory_bound\":";
  write_bool(out, certificate.all_frames_trajectory_bound);
  out << ",\"all_spatial_lifts_replayed\":";
  write_bool(out, certificate.all_spatial_lifts_replayed);
  out << ",\"all_selected_candidates_hard_feasible\":";
  write_bool(out, certificate.all_selected_candidates_hard_feasible);
  out << ",\"all_global_gauge_steps_solved\":";
  write_bool(out, certificate.all_global_gauge_steps_solved);
  out << ",\"lattice_basis_transport_accepted_under_policy\":";
  write_bool(out,
             certificate.lattice_basis_transport_accepted_under_policy);
  out << ",\"lattice_basis_transport_provenance_sufficient\":";
  write_bool(out,
             certificate.lattice_basis_transport_provenance_sufficient);
  out << ",\"local_transport_search_completed\":";
  write_bool(out, certificate.local_transport_search_completed);
  out << ",\"explicit_remap_provenance_available\":";
  write_bool(out, certificate.explicit_remap_provenance_available);
  out << ",\"component_representative_injectivity_audit_passed\":";
  write_bool(
      out, certificate.component_representative_injectivity_audit_passed);
  out << ",\"gauge_replay_arithmetic_closed\":";
  write_bool(out, certificate.gauge_replay_arithmetic_closed);
  out << ",\"atom_temporal_injectivity_audited\":";
  write_bool(out, certificate.atom_temporal_injectivity_audited);
  out << ",\"atom_temporal_injectivity_audit_passed\":";
  write_bool(out, certificate.atom_temporal_injectivity_audit_passed);
  out << ",\"independent_space_time_cochain_audited\":";
  write_bool(out, certificate.independent_space_time_cochain_audited);
  out << ",\"independent_space_time_cochain_closed\":";
  write_bool(out, certificate.independent_space_time_cochain_closed);
  out << ",\"all_atom_reconstruction_audited\":";
  write_bool(out, certificate.all_atom_reconstruction_audited);
  out << ",\"coordinates_materialized_and_replay_consistent\":";
  write_bool(out,
             certificate.coordinates_materialized_and_replay_consistent);
  out << ",\"temporal_lift_locally_certified\":";
  write_bool(out, certificate.temporal_lift_locally_certified);
  out << ",\"complete_domain_trajectory_claim_eligible\":";
  write_bool(out,
             certificate.complete_domain_trajectory_claim_eligible);
  out << ",\"source_trajectory_coverage_complete\":";
  write_bool(out, certificate.source_trajectory_coverage_complete);
  out << ",\"retained_candidate_bands_complete\":";
  write_bool(out, certificate.retained_candidate_bands_complete);
  out << ",\"optimal_path_exact_on_retained_graph\":";
  write_bool(out, certificate.optimal_path_exact_on_retained_graph);
  out << ",\"optimal_path_proven_on_complete_bounded_domain\":";
  write_bool(out,
             certificate.optimal_path_proven_on_complete_bounded_domain);
  out << ",\"unique_on_configured_retained_state_graph\":";
  write_bool(out, certificate.unique_on_configured_retained_state_graph);
  out << ",\"selected_transition_time_reversal_audited\":";
  write_bool(out, certificate.selected_transition_time_reversal_audited);
  out << ",\"selected_transition_time_reversal_consistent\":";
  write_bool(out, certificate.selected_transition_time_reversal_consistent);
  out << ",\"robust_loss_exact\":";
  write_bool(out, certificate.robust_loss_exact);
  out << "},\n";

  out << "  \"segments\":[";
  for (std::size_t index = 0; index < result.segments.size(); ++index) {
    const auto &segment = result.segments[index];
    if (index != 0)
      out << ',';
    out << "{\"frame_begin\":" << segment.frame_begin
        << ",\"frame_count\":" << segment.frame_count
        << ",\"topology_epoch_index\":" << segment.topology_epoch_index
        << ",\"objective\":";
    write_number_or_null(out, segment.objective);
    out << ",\"second_best_objective\":";
    write_number_or_null(out, segment.second_best_objective);
    out << ",\"absolute_path_gap\":";
    write_number_or_null(out, segment.absolute_path_gap);
    out << ",\"relative_path_gap\":";
    write_number_or_null(out, segment.relative_path_gap);
    out << ",\"optimal_path_count_capped\":"
        << segment.optimal_path_count_capped
        << ",\"temporal_transition_count\":"
        << segment.temporal_transition_count
        << ",\"ambiguous_gauge_transition_count\":"
        << segment.ambiguous_gauge_transition_count
        << ",\"component_nearest_image_violations\":"
        << segment.component_nearest_image_violations
        << ",\"provider_unsupported_selected_candidate_count\":"
        << segment.provider_unsupported_selected_candidate_count
        << ",\"provider_unsupported_transition_count\":"
        << segment.provider_unsupported_transition_count
        << ",\"component_injectivity_uncertified\":"
        << segment.component_injectivity_uncertified
        << ",\"gauge_replay_arithmetic_residuals\":"
        << segment.gauge_replay_arithmetic_residuals
        << ",\"temporal_frames_audited\":"
        << segment.temporal_frames_audited
        << ",\"atoms_audited\":" << segment.atoms_audited
        << ",\"atom_temporal_transitions_audited\":"
        << segment.atom_temporal_transitions_audited
        << ",\"atom_temporal_nearest_image_violations\":"
        << segment.atom_temporal_nearest_image_violations
        << ",\"atom_temporal_injectivity_uncertified\":"
        << segment.atom_temporal_injectivity_uncertified
        << ",\"independent_temporal_image_ambiguities\":"
        << segment.independent_temporal_image_ambiguities
        << ",\"independent_space_time_edges_audited\":"
        << segment.independent_space_time_edges_audited
        << ",\"independent_space_time_curvature_residuals\":"
        << segment.independent_space_time_curvature_residuals
        << ",\"lattice_basis_transports_applied\":"
        << segment.lattice_basis_transports_applied
        << ",\"quadratic_direct_residual_fallbacks\":"
        << segment.quadratic_direct_residual_fallbacks
        << ",\"pretransformed_candidate_component_images\":"
        << segment.pretransformed_candidate_component_images
        << ",\"peak_layer_state_count\":"
        << segment.peak_layer_state_count
        << ",\"compact_backpointer_bytes_estimate\":"
        << segment.compact_backpointer_bytes_estimate
        << ",\"bounded_domain_only\":";
    write_bool(out, segment.bounded_domain_only);
    out << ",\"anchors_hard_pinned\":";
    write_bool(out, segment.anchors_hard_pinned);
    out << ",\"exact_on_retained_graph\":";
    write_bool(out, segment.exact_on_retained_graph);
    out << ",\"optimal_path_proven_on_complete_bounded_domain\":";
    write_bool(out,
               segment.optimal_path_proven_on_complete_bounded_domain);
    out << ",\"unique_on_configured_retained_state_graph\":";
    write_bool(out, segment.unique_on_configured_retained_state_graph);
    out << ",\"retained_candidate_bands_complete\":";
    write_bool(out, segment.retained_candidate_bands_complete);
    out << ",\"all_global_gauge_steps_solved\":";
    write_bool(out, segment.all_global_gauge_steps_solved);
    out << ",\"lattice_basis_transport_accepted_under_policy\":";
    write_bool(out,
               segment.lattice_basis_transport_accepted_under_policy);
    out << ",\"lattice_basis_transport_provenance_sufficient\":";
    write_bool(out,
               segment.lattice_basis_transport_provenance_sufficient);
    out << ",\"local_transport_search_completed\":";
    write_bool(out, segment.local_transport_search_completed);
    out << ",\"explicit_remap_provenance_available\":";
    write_bool(out, segment.explicit_remap_provenance_available);
    out << ",\"maximum_lattice_basis_transport_relative_mismatch\":";
    write_number_or_null(
        out, segment.maximum_lattice_basis_transport_relative_mismatch);
    out << ",\"maximum_component_representative_error_A\":";
    write_number_or_null(
        out, segment.maximum_component_representative_error_A);
    out << ",\"maximum_atom_peculiar_displacement_A\":";
    write_number_or_null(out,
                         segment.maximum_atom_peculiar_displacement_A);
    out << ",\"component_weighting\":";
    write_json_string(out,
                      component_weighting_name(segment.component_weighting));
    out << ",\"all_atom_reconstruction_audited\":";
    write_bool(out, segment.all_atom_reconstruction_audited);
    out << ",\"coordinates_materialized_and_replay_consistent\":";
    write_bool(out, segment.coordinates_materialized_and_replay_consistent);
    out << ",\"temporal_lift_locally_certified\":";
    write_bool(out, segment.temporal_lift_locally_certified);
    out << ",\"gauge_replay_arithmetic_closed\":";
    write_bool(out, segment.gauge_replay_arithmetic_closed);
    out << ",\"atom_temporal_injectivity_audited\":";
    write_bool(out, segment.atom_temporal_injectivity_audited);
    out << ",\"atom_temporal_injectivity_audit_passed\":";
    write_bool(out, segment.atom_temporal_injectivity_audit_passed);
    out << ",\"independent_space_time_cochain_audited\":";
    write_bool(out, segment.independent_space_time_cochain_audited);
    out << ",\"independent_space_time_cochain_closed\":";
    write_bool(out, segment.independent_space_time_cochain_closed);
    out << ",\"component_representative_injectivity_audit_passed\":";
    write_bool(out,
               segment.component_representative_injectivity_audit_passed);
    out << ",\"selected_transition_time_reversal_audited\":";
    write_bool(out, segment.selected_transition_time_reversal_audited);
    out << ",\"selected_transition_time_reversal_consistent\":";
    write_bool(out, segment.selected_transition_time_reversal_consistent);
    out << '}';
  }
  out << "],\n";

  out << "  \"frames\":[";
  for (std::size_t index = 0; index < result.frames.size(); ++index) {
    const auto &source = store.frames()[index];
    const auto &selection = result.frames[index];
    if (index != 0)
      out << ',';
    out << "{\"store_frame_index\":" << index << ",\"frame\":"
        << source.frame << ",\"trajectory_frame_index\":"
        << source.trajectory_frame_index << ",\"time_ps\":";
    write_number_or_null(out, source.time_ps);
    out << ",\"wrapped_coordinate_hash\":";
    write_json_string(out,
                      format_hololift_hash128(source.wrapped_coordinate_hash));
    out << ",\"spatial_lift_identity\":";
    write_json_string(out,
                      format_hololift_hash128(source.spatial_lift_identity));
    out << ",\"use\":";
    write_json_string(out, hololift_phase1_frame_use_name(selection.use));
    out << ",\"preceding_transition_gap\":";
    write_json_string(
        out,
        hololift_phase1_transition_gap_name(
            selection.preceding_transition_gap));
    out << ",\"segment_index\":";
    if (selection.segment_index == HOLOLIFT_NO_INDEX)
      out << "null";
    else
      out << selection.segment_index;
    out << ",\"candidate_index\":";
    if (selection.candidate_index == HOLOLIFT_NO_INDEX) {
      out << "null,\"assignment_identity\":null";
    } else {
      out << selection.candidate_index << ",\"assignment_identity\":";
      write_json_string(
          out, format_hololift_hash128(
                   store.candidates()[selection.candidate_index]
                       .assignment_identity));
    }
    out << ",\"emission_cost\":";
    write_number_or_null(out, selection.emission_cost);
    out << ",\"transition_cost\":";
    write_number_or_null(out, selection.transition_cost);
    out << ",\"cumulative_segment_cost\":";
    write_number_or_null(out, selection.cumulative_segment_cost);
    out << ",\"selected_global_gauge_increment\":";
    write_image(out, selection.selected_global_gauge_increment);
    out << ",\"cumulative_global_gauge\":";
    write_image(out, selection.cumulative_global_gauge);
    out << ",\"cumulative_global_gauge_input_basis\":";
    write_image(out, selection.cumulative_global_gauge_input_basis);
    out << ",\"lattice_basis_transport_to_segment_reference\":";
    write_integer_matrix(
        out, selection.lattice_basis_transport_to_segment_reference);
    out << ",\"lattice_basis_transport_relative_mismatch\":";
    write_number_or_null(
        out, selection.lattice_basis_transport_relative_mismatch);
    out << ",\"lattice_basis_transport_identity_relative_mismatch\":";
    write_number_or_null(
        out, selection.lattice_basis_transport_identity_relative_mismatch);
    out << ",\"lattice_basis_transport_improvement_ratio\":";
    write_number_or_null(
        out, selection.lattice_basis_transport_improvement_ratio);
    out << ",\"lattice_basis_transport_applied\":";
    write_bool(out, selection.lattice_basis_transport_applied);
    out << ",\"lattice_basis_transport_ambiguous\":";
    write_bool(out, selection.lattice_basis_transport_ambiguous);
    out << ",\"lattice_basis_transport_identity_fast_path\":";
    write_bool(out, selection.lattice_basis_transport_identity_fast_path);
    out << ",\"lattice_basis_transport_local_search_completed\":";
    write_bool(
        out, selection.lattice_basis_transport_local_search_completed);
    out << ",\"lattice_basis_transport_accepted_under_policy\":";
    write_bool(
        out, selection.lattice_basis_transport_accepted_under_policy);
    out << ",\"lattice_basis_transport_explicit_remap_provenance_available\":";
    write_bool(
        out,
        selection
            .lattice_basis_transport_explicit_remap_provenance_available);
    out << ",\"lattice_basis_transport_provenance_sufficient\":";
    write_bool(
        out, selection.lattice_basis_transport_provenance_sufficient);
    out << ",\"quadratic_direct_residual_fallback\":";
    write_bool(out, selection.quadratic_direct_residual_fallback);
    out << ",\"component_displacement_rms_A\":";
    write_number_or_null(out, selection.component_displacement_rms_A);
    out << ",\"component_displacement_max_A\":";
    write_number_or_null(out, selection.component_displacement_max_A);
    out << ",\"component_injectivity_margin_A\":";
    write_number_or_null(out, selection.component_injectivity_margin_A);
    out << ",\"component_injectivity_radius_lower_bound_A\":";
    write_number_or_null(
        out, selection.component_injectivity_radius_lower_bound_A);
    out << ",\"component_nearest_image_violations\":"
        << selection.component_nearest_image_violations
        << ",\"component_injectivity_uncertified\":"
        << selection.component_injectivity_uncertified
        << ",\"atoms_audited\":" << selection.atoms_audited
        << ",\"atom_temporal_transitions_audited\":"
        << selection.atom_temporal_transitions_audited
        << ",\"atom_temporal_nearest_image_violations\":"
        << selection.atom_temporal_nearest_image_violations
        << ",\"atom_temporal_injectivity_uncertified\":"
        << selection.atom_temporal_injectivity_uncertified
        << ",\"independent_temporal_image_ambiguities\":"
        << selection.independent_temporal_image_ambiguities
        << ",\"independent_space_time_edges_audited\":"
        << selection.independent_space_time_edges_audited
        << ",\"independent_space_time_curvature_residuals\":"
        << selection.independent_space_time_curvature_residuals
        << ",\"maximum_component_representative_error_A\":";
    write_number_or_null(
        out, selection.maximum_component_representative_error_A);
    out
        << ",\"maximum_atom_peculiar_displacement_A\":";
    write_number_or_null(out,
                         selection.maximum_atom_peculiar_displacement_A);
    out << ",\"global_gauge_increment_ambiguous\":";
    write_bool(out, selection.global_gauge_increment_ambiguous);
    out << ",\"pinned_strong_bounded_anchor\":";
    write_bool(out, selection.pinned_strong_bounded_anchor);
    out << ",\"changed_from_framewise\":";
    write_bool(out, selection.changed_from_framewise);
    out << ",\"selected_candidate_provider_unsupported\":";
    write_bool(out, selection.selected_candidate_provider_unsupported);
    out << ",\"preceding_transition_provider_unsupported\":";
    write_bool(out, selection.preceding_transition_provider_unsupported);
    out << '}';
  }
  out << "]\n}\n";
  out.flush();
  out.close();
  if (!out)
    return fail_not_committed("failed to write HoloLift Phase 1 result");
  auto replaced = detail::durably_replace_file(
      temporary, path, "HoloLift Phase 1 result");
  if (!replaced) {
    auto issue = std::move(replaced.error());
    if (!issue.committed())
      cleanup_temporary();
    return std::unexpected(std::move(issue));
  }
  return *replaced;
}

} // namespace titan_hololift

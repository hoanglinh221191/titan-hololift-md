#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

bool PbctopoFrameSolver::certify_and_commit() {
  output_candidate_committed = false;
  output_certificate_rejects = 0;
  output_all_atom_certificate_rejects = 0;
  timing.pbctopo_output_certification_band_candidates +=
      output_candidate_band.size();
  timing.pbctopo_output_certification_band_peak =
      max(timing.pbctopo_output_certification_band_peak,
          output_candidate_band.size());
  if (output_candidate_band.size() > kInitialOutputCertificationBand)
    ++timing.pbctopo_output_certification_band_expanded_frames;
  output_certification_points.reserve(base_atoms.size());
  certify_complete_uncertainty_band =
      titan_pbctopo::pbctopo_should_certify_complete_uncertainty_band(
          complete_feasible_band_retained,
          frame_report.component_feasible_assignment_set_enumerated);
  if (certify_complete_uncertainty_band) {
    optional<PbctopoCutCandidate> complete_band_winner;
    optional<PbctopoSelectedEvidenceAudit> complete_band_winner_audit;
    size_t complete_band_winner_index = 0;
    size_t complete_band_winner_all_atom_rejects = 0;
    PbctopoFrameMode complete_band_winner_mode = PbctopoFrameMode::None;
    vector<Int3> complete_band_winner_offsets;
    bool evidence_audit_failed = false;
    // Every band candidate is certified and audited, with no early exit,
    // so the geometry of all of them is computed first -- on the
    // TitanExecutor's workers when it has any -- and the loop below then
    // replays, in band order, exactly what the serial loop did with it.
    // Workers need the component context built and every offset vector
    // complete (the only paths on which certification writes frame state
    // before it finishes); otherwise the loop certifies serially as before.
    const size_t band_size = output_candidate_band.size();
    const size_t band_workers =
        executor != nullptr ? min(executor->capacity(), band_size) : 1;
    const bool band_parallel =
        band_workers > 1 && component_context_ready &&
        all_of(output_candidate_band.begin(), output_candidate_band.end(),
               [&](const auto &snapshot) {
                 return snapshot.offsets.size() == components.size();
               });
    vector<PbctopoCutCandidate> band_candidates;
    vector<vector<Vec3>> band_points;
    vector<CandidateCertificationWork> band_work;
    if (band_parallel) {
      band_candidates.resize(band_size);
      band_points.resize(band_size);
      band_work.resize(band_size);
      // Built here, on the frame's thread, so that the workers find it.
      ensure_edge_image_cache();
      executor->parallel_for(
          band_size, band_workers, 1,
          [&](size_t, size_t begin, size_t end) {
            for (size_t index = begin; index < end; ++index) {
              const auto &snapshot = output_candidate_band[index];
              // The serial loop builds the points on the frame's thread
              // too, so their lattice queries belong in the carried delta.
              const auto worker_before =
                  titan_pbctopo::pbctopo_lattice_query_audit();
              band_candidates[index] = snapshot.candidate;
              double shift_norm2 = 0.0;
              build_component_points_from_offsets(
                  component_atoms, components, self.m_mol[frame_idx].Box,
                  snapshot.offsets, band_points[index], shift_norm2);
              compute_candidate_certification(
                  band_work[index], band_candidates[index],
                  band_points[index], &snapshot.offsets, true);
              band_work[index].worker_lattice_delta =
                  titan_pbctopo::pbctopo_lattice_query_audit_delta(
                      worker_before,
                      titan_pbctopo::pbctopo_lattice_query_audit());
            }
          });
    }
    for (size_t band_index = 0; band_index < output_candidate_band.size();
         ++band_index) {
      auto &snapshot = output_candidate_band[band_index];
      PbctopoCutCandidate candidate =
          band_parallel ? std::move(band_candidates[band_index])
                        : snapshot.candidate;
      if (band_parallel) {
        output_certification_points = band_points[band_index];
      } else {
        double output_shift_norm2 = 0.0;
        build_component_points_from_offsets(
            component_atoms, components, self.m_mol[frame_idx].Box,
            snapshot.offsets, output_certification_points,
            output_shift_norm2);
      }
      ++timing.pbctopo_output_certification_band_rebuilds;
      const bool certified =
          band_parallel
              ? apply_candidate_certification(band_work[band_index],
                                              candidate)
              : certify_candidate_for_output(candidate,
                                             output_certification_points,
                                             &snapshot.offsets);
      if (!certified) {
        ++output_certificate_rejects;
        ++output_all_atom_certificate_rejects;
        ++timing.pbctopo_output_certification_band_rejects;
        if (heavy_search_omits_atoms) {
          ++timing.pbctopo_heavy_search_all_atom_candidate_rejects;
          if (band_index == 0)
            ++timing.pbctopo_heavy_search_top_candidate_rejects;
        }
        ++timing.pbctopo_diagnostic_only_candidates;
        continue;
      }
      // The audit the serial loop computed here has the same inputs as the
      // one certification made for the evidence annotation, so a worker's
      // copy of that one is reused.
      const auto audit_started = clock::now();
      auto audit =
          band_parallel && band_work[band_index].evidence.has_value()
              ? *band_work[band_index].evidence
              : titan_pbctopo::audit_pbctopo_selected_hypotheses(
                    span<const PbctopoAtom>(base_atoms),
                    span<const Vec3>(output_certification_points),
                    base_anchor_context.contact_hypotheses,
                    candidate.assignment, true,
                    explicit_evidence_component_edges);
      timing.pbctopo_selected_evidence_audit_s +=
          chrono::duration<double>(clock::now() - audit_started).count();
      if (!audit || !audit->evaluated) {
        evidence_audit_failed = true;
        ++timing.pbctopo_output_certification_band_rejects;
        ++timing.pbctopo_diagnostic_only_candidates;
        continue;
      }
      ++timing.pbctopo_selected_evidence_audits;
      timing.pbctopo_selected_evidence_contacts_checked +=
          audit->contacts_checked;
      timing.pbctopo_selected_evidence_contacts_lost +=
          audit->contacts_lost;
      if (audit->state != PbctopoEvidenceState::Contradicted) {
        record_temporal_certified_candidate(candidate,
                                            output_certification_points,
                                            &snapshot.offsets);
      }
      if (!complete_band_winner.has_value()) {
        complete_band_winner = candidate;
        complete_band_winner_audit = *audit;
        cache_candidate_evidence_audit(candidate.assignment, *audit);
        complete_band_winner_index = band_index;
        complete_band_winner_all_atom_rejects =
            output_all_atom_certificate_rejects;
        complete_band_winner_mode = snapshot.mode;
        complete_band_winner_offsets = snapshot.offsets;
      }
    }
    const bool complete_band_exported =
        titan_pbctopo::pbctopo_complete_uncertainty_band_exported(
            !evidence_audit_failed, output_certificate_rejects,
            report_certified_candidates.size(),
            frame_report.component_feasible_assignments_within_domain);
    if (!complete_band_exported) {
      frame_identifiability.feasible_assignment_set_enumerated = false;
      frame_identifiability.feasible_assignments_within_domain = 0;
      // apply_pbctopo_search_identifiability derives evidence_uniqueness_known
      // from feasible_assignment_set_enumerated, so retracting the enumeration
      // here has to retract the claim that uniqueness is known along with it.
      // Leaving it set writes a frame that says the evidence uniqueness is
      // known while reporting no feasible-set audit, a pair that downstream
      // frame-binding validators reject as invalid_candidate_set. It only
      // shows on frames with no evidence-compatible candidate, where the
      // band cannot be exported.
      frame_identifiability.evidence_uniqueness_known = false;
      frame_identifiability.evidence_unique_within_search_domain = false;
    }
    if (complete_band_winner.has_value() &&
        complete_band_winner_audit.has_value()) {
      PbctopoCutCandidate candidate = *complete_band_winner;
      selected_evidence_audit = *complete_band_winner_audit;
      apply_selected_evidence_to_frame_identifiability();
      double output_shift_norm2 = 0.0;
      build_component_points_from_offsets(
          component_atoms, components, self.m_mol[frame_idx].Box,
          complete_band_winner_offsets, output_certification_points,
          output_shift_norm2);
      export_h9_partition_and_orbit(candidate,
                                    complete_band_winner_offsets);
      const bool evidence_contradicted =
          selected_evidence_audit.state ==
          PbctopoEvidenceState::Contradicted;
      const bool evidence_weak =
          selected_evidence_audit.state ==
              PbctopoEvidenceState::WeakDisconnected ||
          pbctopo_evidence_state_underdetermined(selected_evidence_audit.state);
      const bool output_identified = selected_relative_images_identified();
      bool winner_commit_ok = true;
      if (evidence_contradicted) {
        warm_state = frame_start_warm_state;
      } else if (output_identified && !evidence_weak) {
        if (!commit_certified_candidate_to_warm_state(candidate)) {
          ++output_certificate_rejects;
          ++timing.pbctopo_output_certification_band_rejects;
          winner_commit_ok = false;
        }
      } else {
        warm_state = frame_start_warm_state;
        ++timing.pbctopo_weak_observation_warm_preserved;
      }
      if (winner_commit_ok) {
        best_safe_candidate = candidate;
        best_safe_points = output_certification_points;
        projection_clash_skip_active =
            candidate_has_projection_clash_margin(
                best_safe_candidate, runtime_cutoffs.clash_cutoff);
        apply_cut_candidate_timed(best_safe_points);
        finalize_frame_report(
            evidence_contradicted
                ? PbctopoFrameStatus::Contradicted
                : output_identified && !evidence_weak
                      ? PbctopoFrameStatus::Certified
                      : PbctopoFrameStatus::WeakObservation,
            complete_band_winner_mode == PbctopoFrameMode::None
                ? PbctopoFrameMode::Virtual
                : complete_band_winner_mode,
            evidence_contradicted
                ? PbctopoFailureReason::EvidenceContradicted
                : output_identified && !evidence_weak
                      ? PbctopoFailureReason::None
                      : pbctopo_weak_evidence_reason(selected_evidence_audit.state),
            &best_safe_candidate);
        output_candidate_committed = true;
        // Every member ranked above the winner was rejected, so these are
        // the recoveries the first-certifiable loop below counts.
        if (complete_band_winner_index > 0)
          ++timing.pbctopo_output_certification_band_recoveries;
        if (complete_band_winner_index >=
            kInitialOutputCertificationBand) {
          ++timing.pbctopo_output_certification_band_expanded_recoveries;
        }
        if (heavy_search_omits_atoms &&
            complete_band_winner_all_atom_rejects > 0) {
          ++timing.pbctopo_heavy_search_all_atom_ranking_recoveries;
        }
      }
    } else if (evidence_audit_failed) {
      warm_state = frame_start_warm_state;
      reset_kinematic_seed_history(warm_state, timing);
      temporal_cut_candidates[frame_idx].clear();
      report_certified_candidates.clear();
          report_rejected_candidates.clear();
      record_frame_failure(
          PbctopoFailureReason::EvidenceHypothesisUnavailable);
    }
  }
  for (size_t band_index = 0;
       !certify_complete_uncertainty_band &&
       band_index < output_candidate_band.size(); ++band_index) {
    auto &snapshot = output_candidate_band[band_index];
    PbctopoCutCandidate candidate = snapshot.candidate;
    double output_shift_norm2 = 0.0;
    build_component_points_from_offsets(
        component_atoms, components, self.m_mol[frame_idx].Box,
        snapshot.offsets, output_certification_points, output_shift_norm2);
    ++timing.pbctopo_output_certification_band_rebuilds;
    if (!certify_candidate_for_output(candidate, output_certification_points,
                                      &snapshot.offsets)) {
      ++output_certificate_rejects;
      ++output_all_atom_certificate_rejects;
      ++timing.pbctopo_output_certification_band_rejects;
      if (heavy_search_omits_atoms) {
        ++timing.pbctopo_heavy_search_all_atom_candidate_rejects;
        if (band_index == 0)
          ++timing.pbctopo_heavy_search_top_candidate_rejects;
      }
      ++timing.pbctopo_diagnostic_only_candidates;
      continue;
    }
    audit_selected_evidence(output_certification_points,
                            candidate.assignment);
    if (!selected_evidence_audit.evaluated) {
      warm_state = frame_start_warm_state;
      reset_kinematic_seed_history(warm_state, timing);
      temporal_cut_candidates[frame_idx].clear();
      report_certified_candidates.clear();
          report_rejected_candidates.clear();
      ++output_certificate_rejects;
      ++timing.pbctopo_output_certification_band_rejects;
      ++timing.pbctopo_diagnostic_only_candidates;
      record_frame_failure(
          PbctopoFailureReason::EvidenceHypothesisUnavailable);
      continue;
    }
    export_h9_partition_and_orbit(candidate, snapshot.offsets);
    const bool evidence_contradicted =
        selected_evidence_audit.state == PbctopoEvidenceState::Contradicted;
    const bool evidence_weak =
        selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
        pbctopo_evidence_state_underdetermined(selected_evidence_audit.state);
    if (!evidence_contradicted) {
      record_temporal_certified_candidate(candidate,
                                          output_certification_points,
                                          &snapshot.offsets);
    }
    const bool output_identified = selected_relative_images_identified();
    if (evidence_contradicted) {
      warm_state = frame_start_warm_state;
    } else if (output_identified && !evidence_weak) {
      if (!commit_certified_candidate_to_warm_state(candidate)) {
        ++output_certificate_rejects;
        ++timing.pbctopo_output_certification_band_rejects;
        ++timing.pbctopo_diagnostic_only_candidates;
        continue;
      }
    } else {
      warm_state = frame_start_warm_state;
      ++timing.pbctopo_weak_observation_warm_preserved;
    }
    best_safe_candidate = candidate;
    best_safe_points = output_certification_points;
    projection_clash_skip_active = candidate_has_projection_clash_margin(
        best_safe_candidate, runtime_cutoffs.clash_cutoff);
    apply_cut_candidate_timed(best_safe_points);
    finalize_frame_report(
        evidence_contradicted
            ? PbctopoFrameStatus::Contradicted
            : output_identified && !evidence_weak
                  ? PbctopoFrameStatus::Certified
                                : PbctopoFrameStatus::WeakObservation,
        snapshot.mode == PbctopoFrameMode::None
            ? PbctopoFrameMode::Virtual
            : snapshot.mode,
        evidence_contradicted
            ? PbctopoFailureReason::EvidenceContradicted
            : output_identified && !evidence_weak
                  ? PbctopoFailureReason::None
                  : pbctopo_weak_evidence_reason(selected_evidence_audit.state),
        &best_safe_candidate);
    output_candidate_committed = true;
    if (output_certificate_rejects > 0)
      ++timing.pbctopo_output_certification_band_recoveries;
    if (band_index >= kInitialOutputCertificationBand)
      ++timing.pbctopo_output_certification_band_expanded_recoveries;
    if (heavy_search_omits_atoms &&
        output_all_atom_certificate_rejects > 0) {
      ++timing.pbctopo_heavy_search_all_atom_ranking_recoveries;
    }
    break;
  }
  if (output_candidate_committed)
    return false;
  if (heavy_search_omits_atoms && !output_candidate_band.empty() &&
      output_all_atom_certificate_rejects ==
          output_candidate_band.size()) {
    ++timing.pbctopo_heavy_search_all_atom_band_exhausted;
  }
  if (output_certificate_rejects > 0) {
    best_safe_certification_rejected = true;
    record_frame_failure(PbctopoFailureReason::CertificationReject);
    append_pbctopo_log_line(
        "[pbctopo] rejected output certification band at frame " +
        to_string(static_cast<long long>(
            self.m_mol[frame_idx].time_of_snap)) +
        ": assignments_tested=" +
        to_string(output_certificate_rejects) +
        ", band_size=" + to_string(output_candidate_band.size()));
  }

  if ((!best_safe_candidate.valid || best_safe_candidate.ultra_clash ||
       best_safe_certification_rejected) &&
      split_rescue_available) {
    ensure_component_context();
    ++timing.hull_rescue_attempts;
    PbctopoCutCandidate best_hull_candidate;
    vector<Vec3> best_hull_points;
    best_hull_points.reserve(base_atoms.size());
    Int3 best_hull_shift{};
    int best_hull_owner = -1;
    double best_hull_volume = numeric_limits<double>::infinity();
    size_t hull_valid_candidates = 0;
    vector<int> rescue_owner_candidates;
    rescue_owner_candidates.reserve(2);
    rescue_owner_candidates.push_back(split_rescue_detail.owner_b);
    if (split_rescue_detail.owner_a != split_rescue_detail.owner_b)
      rescue_owner_candidates.push_back(split_rescue_detail.owner_a);
    for (int rescue_owner : rescue_owner_candidates) {
      build_owner_spans(base_atoms, rescue_owner, hull_owner_spans);
      if (hull_owner_spans.empty())
        continue;
      for (const auto &shift : shifts) {
        if (shift.ix == 0 && shift.iy == 0 && shift.iz == 0)
          continue;
        hull_rescue_points.clear();
        pbctopo_hull::shift_points_by_spans(
            split_rescue_points, hull_owner_spans,
            Vec3{shift.tx, shift.ty, shift.tz}, hull_rescue_points,
            &hull_scratch);
        PbctopoCutCandidate hull_candidate;
        const PbctopoCutCandidate *hull_incumbent =
            best_hull_candidate.valid ? &best_hull_candidate : nullptr;
        if (!validate_rescue_candidate_points(hull_rescue_points,
                                              hull_incumbent,
                                              hull_candidate)) {
          continue;
        }
        double hull_volume = numeric_limits<double>::infinity();
        if (!pbctopo_hull::quickhull_volume(hull_rescue_points, hull_volume,
                                            &hull_scratch)) {
          continue;
        }
        ++hull_valid_candidates;
        const bool better_hull =
            !best_hull_candidate.valid ||
            hull_volume + 1.0e-6 < best_hull_volume ||
            (fabs(hull_volume - best_hull_volume) <= 1.0e-6 &&
             better_cut_candidate(hull_candidate, best_hull_candidate));
        if (!better_hull)
          continue;
        best_hull_candidate = hull_candidate;
        best_hull_volume = hull_volume;
        best_hull_shift = {shift.ix, shift.iy, shift.iz};
        best_hull_owner = rescue_owner;
        best_hull_points = hull_rescue_points;
      }
    }
    if (best_hull_candidate.valid) {
      append_pbctopo_log_line(
          "[pbctopo] hull rescue candidate at frame " +
          to_string(static_cast<long long>(
              self.m_mol[frame_idx].time_of_snap)) +
          " from " + split_rescue_source + ": moved_owner=" +
          to_string(best_hull_owner) + ", applied_shift=" +
          format_int3(best_hull_shift) + ", hull_volume=" +
          to_string(best_hull_volume) + ", valid_candidates=" +
          to_string(hull_valid_candidates) + ", avx2=" +
          string(pbctopo_hull::avx2_available() ? "on" : "off"));
      if (accept_rescue_candidate(best_hull_points, best_hull_candidate)) {
        ++timing.hull_rescue_successes;
        const bool rescue_identified = selected_relative_images_identified();
        const bool rescue_evidence_weak =
            selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
            pbctopo_evidence_state_underdetermined(selected_evidence_audit.state);
        finalize_frame_report(rescue_identified && !rescue_evidence_weak
                                  ? (selected_evidence_audit.state ==
                                             PbctopoEvidenceState::Contradicted
                                         ? PbctopoFrameStatus::Contradicted
                                         : PbctopoFrameStatus::Rescued)
                                  : (selected_evidence_audit.state ==
                                             PbctopoEvidenceState::Contradicted
                                         ? PbctopoFrameStatus::Contradicted
                                         : PbctopoFrameStatus::WeakObservation),
                              PbctopoFrameMode::HullRescue,
                              selected_evidence_audit.state ==
                                      PbctopoEvidenceState::Contradicted
                                  ? PbctopoFailureReason::EvidenceContradicted
                                  : rescue_identified
                                        ? PbctopoFailureReason::None
                                        : pbctopo_weak_evidence_reason(selected_evidence_audit.state),
                              &best_safe_candidate);
        return false;
      }
    }

    ++timing.distance_rescue_attempts;
    vector<Vec3> rescued_points;
    rescued_points.reserve(base_atoms.size());
    Int3 rescue_shift{};
    double rescue_raw_delta_norm = numeric_limits<double>::infinity();
    if (rescue_split_contacting_owner_candidate(
            split_rescue_points, base_atoms, self.m_mol[frame_idx].Box,
            local_hard_required_owner_pairs, shifts, split_rescue_detail,
            rescued_points, rescue_shift, rescue_raw_delta_norm)) {
      PbctopoCutCandidate rescued_candidate;
      if (validate_rescue_candidate_points(rescued_points, nullptr,
                                           rescued_candidate)) {
        append_pbctopo_log_line(
            "[pbctopo] distance rescue candidate at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            " from " + split_rescue_source + ": applied_shift=" +
            format_int3(rescue_shift) + ", rescued_raw_delta_norm=" +
            to_string(rescue_raw_delta_norm));
        if (accept_rescue_candidate(rescued_points, rescued_candidate)) {
          ++timing.distance_rescue_successes;
          const bool rescue_identified =
              selected_relative_images_identified();
          const bool rescue_evidence_weak =
              selected_evidence_audit.state == PbctopoEvidenceState::WeakDisconnected ||
              pbctopo_evidence_state_underdetermined(selected_evidence_audit.state);
          finalize_frame_report(rescue_identified && !rescue_evidence_weak
                                    ? (selected_evidence_audit.state ==
                                               PbctopoEvidenceState::Contradicted
                                           ? PbctopoFrameStatus::Contradicted
                                           : PbctopoFrameStatus::Rescued)
                                    : (selected_evidence_audit.state ==
                                               PbctopoEvidenceState::Contradicted
                                           ? PbctopoFrameStatus::Contradicted
                                           : PbctopoFrameStatus::WeakObservation),
                                PbctopoFrameMode::DistanceRescue,
                                selected_evidence_audit.state ==
                                        PbctopoEvidenceState::Contradicted
                                    ? PbctopoFailureReason::EvidenceContradicted
                                    : rescue_identified
                                          ? PbctopoFailureReason::None
                                          : pbctopo_weak_evidence_reason(selected_evidence_audit.state),
                                &best_safe_candidate);
          return false;
        }
      }
    }
  }

  if (best_candidate.valid)
    ++timing.pbctopo_diagnostic_only_candidates;
  lattice_query_failed =
      frame_fatal_lattice_query_failures > 0;
  fallback_reason =
      pbctopo_resolve_fallback_reason(lattice_query_failed,
                                      frame_report.fallback_reason);
  fallback_message =
      "[pbctopo] frame " +
      to_string(static_cast<long long>(self.m_mol[frame_idx].time_of_snap)) +
      (lattice_query_failed
           ? " left unchanged after a failed lattice-image query: "
           : " fell back to pbci=0 after no certified image assignment: ") +
      "safe_candidates=" +
      to_string(safe_candidates) + ", best.min_pair_d2=" +
      to_string(best_candidate.min_pair_d2) + ", best.kappa2=" +
      to_string(best_candidate.shape_kappa2) + ", best.asph=" +
      to_string(best_candidate.shape_asphericity) + ", best.bbox=" +
      to_string(best_candidate.bbox_volume);
  append_pbctopo_log_line(fallback_message);
  TitanUI::print_warning(fallback_message);
  ++fallback_frames;
  projection_clash_skip_active = false;
  reset_uncommitted_warm_state();
  if (!lattice_query_failed)
    fallback_pbci_frame(frame_idx);
  finalize_frame_report(PbctopoFrameStatus::Fallback,
                        lattice_query_failed ? PbctopoFrameMode::None
                                             : PbctopoFrameMode::Pbci,
                        fallback_reason, nullptr);
  return true;
}

} // namespace pbctopo_solve

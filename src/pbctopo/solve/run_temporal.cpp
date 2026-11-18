#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

void PbctopoSolveRun::run_temporal() {

  if (temporal_mode == PbctopoTemporalMode::Viterbi) {
    vector<vector<PbctopoTemporalCandidate>> temporal_frames(frame_count);
    for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
      const auto &cuts = temporal_cut_candidates[frame_idx];
      const auto &greedy = greedy_selected_candidates[frame_idx];
      if (cuts.empty() || !greedy.valid || !greedy.assignment.valid ||
          !greedy.hard_feasibility.certified())
        continue;
      ++timing.pbctopo_temporal_candidate_frames;
      timing.pbctopo_temporal_candidates += cuts.size();
      temporal_frames[frame_idx].reserve(cuts.size());
      double v1[3]{}, v2[3]{}, v3[3]{};
      self.m_mol[frame_idx].Box.get_v1(v1);
      self.m_mol[frame_idx].Box.get_v2(v2);
      self.m_mol[frame_idx].Box.get_v3(v3);
      const auto temporal_gram =
          titan_pbctopo::pbctopo_temporal_normalized_gram(
              {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
               v3[0], v3[1], v3[2]});
      const size_t layout_idx = temporal_frame_layout[frame_idx];
      for (size_t rank = 0; rank < cuts.size(); ++rank) {
        PbctopoTemporalCandidate candidate;
        candidate.center = {cuts[rank].cut_x, cuts[rank].cut_y,
                            cuts[rank].cut_z};
        candidate.normalized_gram = temporal_gram;
        candidate.metric_available = true;
        candidate.emission_cost =
            candidate_temporal_emission_cost(cuts.front(), cuts[rank]);
        candidate.component_offsets =
            cuts[rank].assignment.component_offsets;
        if (layout_idx < temporal_component_layouts.size()) {
          candidate.component_ids =
              temporal_component_layouts[layout_idx].component_ids;
          candidate.component_weights =
              temporal_component_layouts[layout_idx].component_weights;
        }
        candidate.assignment_signature = cuts[rank].assignment.signature;
        candidate.certified = cuts[rank].hard_feasibility.certified() &&
                              cuts[rank].assignment.valid;
        temporal_frames[frame_idx].push_back(std::move(candidate));
      }
    }

    const auto temporal_result = titan_pbctopo::solve_pbctopo_temporal_viterbi(
        temporal_frames, Arguments.pbc.pbctopo_temporal_penalty);
    ofstream temporal_report = open_pbctopo_temporal_path_report();
    if (!temporal_report.is_open()) {
      append_pbctopo_log_line(
          "[pbctopo] observation export unavailable: "
          "artifact=pbctopo_temporal_path.csv");
    }
    const size_t invalid = numeric_limits<size_t>::max();
    if (temporal_result.valid) {
      for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        const size_t selected = temporal_result.selected[frame_idx];
        if (selected == invalid ||
            selected >= temporal_cut_candidates[frame_idx].size())
          continue;
        const auto &chosen = temporal_cut_candidates[frame_idx][selected];
        const auto &greedy = greedy_selected_candidates[frame_idx];
        const bool changed = !same_pbctopo_image_assignment(
            chosen.assignment, greedy.assignment);
        if (changed)
          ++timing.pbctopo_temporal_proposed_changes;
        ++timing.pbctopo_temporal_viterbi_frames;
        ++timing.pbctopo_temporal_replay_frames;

        bool replay_applied = !changed;
        vector<PbctopoAtom> replay_atoms;
        optional<titan_pbctopo::PbctopoSelectedEvidenceAudit>
            temporal_selected_audit;
        if (changed) {
          const size_t layout_idx = temporal_frame_layout[frame_idx];
          replay_atoms.reserve(trajectory_context.total_atoms_hint);
          append_atoms_for_system(self.m_mol[frame_idx], &self.info,
                                  &trajectory_context.rec_static, 0,
                                  replay_atoms);
          if (frame_idx < mol2.m_mol.size() &&
              !mol2.m_mol[frame_idx].pos.x.empty()) {
            append_atoms_for_system(mol2.m_mol[frame_idx], &mol2.info,
                                    &trajectory_context.lig_static, 1,
                                    replay_atoms);
          }
          vector<Vec3> applied_points;
          applied_points.reserve(replay_atoms.size());
          for (const auto &atom : replay_atoms)
            applied_points.push_back({atom.x, atom.y, atom.z});
          vector<Vec3> replayed_points;
          if (layout_idx < temporal_component_layouts.size() &&
              replay_pbctopo_assignment_delta(
                  applied_points,
                  temporal_component_layouts[layout_idx].atom_component_index,
                  greedy.assignment, chosen.assignment,
                  self.m_mol[frame_idx].Box, replayed_points)) {
            const auto temporal_explicit_evidence_component_edges =
                build_explicit_evidence_component_edges(
                    temporal_component_layouts[layout_idx].atom_component_index,
                    temporal_component_layouts[layout_idx].component_count);
            const auto temporal_audit_started = clock::now();
            auto audit = titan_pbctopo::audit_pbctopo_selected_hypotheses(
                span<const PbctopoAtom>(replay_atoms),
                span<const Vec3>(replayed_points),
                temporal_contact_hypotheses[frame_idx], chosen.assignment,
                true, temporal_explicit_evidence_component_edges);
            timing.pbctopo_selected_evidence_audit_s +=
                chrono::duration<double>(clock::now() - temporal_audit_started)
                    .count();
            if (audit && audit->evaluated &&
                audit->state != PbctopoEvidenceState::Contradicted) {
              temporal_selected_audit = *audit;
              apply_pbctopo_points_direct(replay_atoms, replayed_points, self,
                                          mol2, frame_idx);
              replay_applied = true;
              ++timing.pbctopo_temporal_replay_applied;
              ++timing.pbctopo_temporal_path_changes;
            } else {
              ++timing.pbctopo_temporal_replay_skipped;
              if (!audit) {
                append_pbctopo_log_line(
                    "[pbctopo] temporal evidence replay failed closed at frame " +
                    to_string(frame_idx) + ": " + audit.error());
              }
            }
          } else {
            ++timing.pbctopo_temporal_replay_skipped;
          }
        }
        if (deferred_frame_report_valid[frame_idx] != 0) {
          auto &report = deferred_frame_reports[frame_idx];
          const auto &actual_output_candidate =
              changed && replay_applied ? chosen : greedy;
          const bool temporal_output_selected = !changed || replay_applied;
          report.temporal_selected = temporal_output_selected;
          report.temporal_changed_from_greedy = changed && replay_applied;
          report.temporal_assignment_signature =
              temporal_output_selected
                  ? actual_output_candidate.assignment.signature
                  : 0;
          if (changed && replay_applied) {
            const auto &temporal_audit = *temporal_selected_audit;
            ++timing.pbctopo_selected_evidence_audits;
            timing.pbctopo_selected_evidence_contacts_checked +=
                temporal_audit.contacts_checked;
            timing.pbctopo_selected_evidence_contacts_lost +=
                temporal_audit.contacts_lost;
            report.evidence_state = temporal_audit.state;
            report.evidence_consistency_evaluated = temporal_audit.evaluated;
            report.evidence_consistent = temporal_audit.consistent;
            report.evidence_graph_connected =
                temporal_audit.evidence_graph_connected;
            titan_pbctopo::apply_pbctopo_final_selected_evidence(
                report.identifiability, temporal_audit.evaluated,
                temporal_audit.evidence_graph_connected,
                temporal_audit.evidence_component_count,
                temporal_audit.evidence_edge_count,
                temporal_audit.explicit_contact_evidence_edges,
                temporal_audit.scored_contact_evidence_edges,
                temporal_audit.evidence_connected_component_count);
            report.soft_observed_contact_pair_hash =
                temporal_audit.observed_pair_hash;
            report.soft_observed_lost_pair_hash =
                temporal_audit.lost_pair_hash;
            report.soft_observed_all_hypotheses_hash =
                temporal_audit.all_hypotheses_hash;
            report.soft_observed_selected_hypothesis_hash =
                temporal_audit.selected_hypothesis_hash;
            report.soft_observed_selected_compatible_pair_hash =
                temporal_audit.selected_compatible_pair_hash;
            report.soft_observed_hypothesis_count =
                temporal_audit.hypothesis_count;
            report.soft_observed_component_relation_count =
                temporal_audit.component_relation_count;
            report.soft_observed_ambiguous_relation_count =
                temporal_audit.ambiguous_component_relation_count;
            report.soft_observed_compatible_hypothesis_count =
                temporal_audit.compatible_hypothesis_count;
            report.soft_observed_selected_compatible_contact_count =
                temporal_audit.selected_compatible_contact_count;
            report.soft_observed_alternative_hypothesis_contact_count =
                temporal_audit.alternative_hypothesis_contact_count;
            report.soft_observed_selected_compatible_contact_loss_count =
                temporal_audit.selected_compatible_contact_loss_count;
            report.soft_observed_no_support_relation_count =
                temporal_audit.no_supported_contact_relation_count;
            report.soft_observed_selected_relation_has_support =
                temporal_audit.selected_relation_has_support;
            report.soft_observed_contacts_checked =
                temporal_audit.contacts_checked;
            report.soft_observed_contacts_lost = temporal_audit.contacts_lost;
            const bool was_contradicted =
                report.status == PbctopoFrameStatus::Contradicted;
            const bool was_weak =
                report.status == PbctopoFrameStatus::WeakObservation;
            const bool rescue_provenance =
                report.status == PbctopoFrameStatus::Rescued ||
                report.mode == PbctopoFrameMode::HullRescue ||
                report.mode == PbctopoFrameMode::DistanceRescue;
            if (temporal_audit.state ==
                PbctopoEvidenceState::Contradicted) {
              report.status = PbctopoFrameStatus::Contradicted;
              report.fallback_reason =
                  PbctopoFailureReason::EvidenceContradicted;
              report.committed_candidate_count = 0;
            } else if (temporal_audit.state ==
                       PbctopoEvidenceState::WeakDisconnected ||
                       pbctopo_evidence_state_underdetermined(temporal_audit.state)) {
              report.status = PbctopoFrameStatus::WeakObservation;
              report.fallback_reason =
                  pbctopo_weak_evidence_reason(temporal_audit.state);
              report.committed_candidate_count = 1;
            } else {
              report.status = rescue_provenance
                                  ? PbctopoFrameStatus::Rescued
                                  : PbctopoFrameStatus::Certified;
              report.fallback_reason = PbctopoFailureReason::None;
              report.committed_candidate_count = 1;
            }
            const bool is_contradicted =
                report.status == PbctopoFrameStatus::Contradicted;
            const bool is_weak =
                report.status == PbctopoFrameStatus::WeakObservation;
            if (!was_contradicted && is_contradicted) {
              ++timing.pbctopo_contradicted_output_frames;
            } else if (was_contradicted && !is_contradicted &&
                       timing.pbctopo_contradicted_output_frames > 0) {
              --timing.pbctopo_contradicted_output_frames;
            }
            if (!was_weak && is_weak) {
              ++timing.pbctopo_weak_observation_output_frames;
            } else if (was_weak && !is_weak &&
                       timing.pbctopo_weak_observation_output_frames > 0) {
              --timing.pbctopo_weak_observation_output_frames;
            }
            const bool old_anchor_ambiguous =
                report.anchor_graph.ambiguity_count > 0;
            populate_pbctopo_selected_candidate_report(
                report, chosen, replay_atoms, runtime_cutoffs.clash_cutoff,
                runtime_cutoffs.continuity_cutoff,
                temporal_cut_candidates[frame_idx]);
            const bool new_anchor_ambiguous =
                report.anchor_graph.ambiguity_count > 0;
            if (old_anchor_ambiguous != new_anchor_ambiguous) {
              if (new_anchor_ambiguous) {
                ++timing.pbctopo_anchor_ambiguous_frames;
              } else if (timing.pbctopo_anchor_ambiguous_frames > 0) {
                --timing.pbctopo_anchor_ambiguous_frames;
              }
            }
          }
        }
        if (temporal_report.is_open()) {
          temporal_report
                          << titan_pbctopo::VIBE_OBSERVATION_SCHEMA_VERSION
                          << ','
                          << titan_pbctopo::vibe_observation_producer_version()
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_COORDINATE_UNIT
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_TIME_UNIT << ','
                          << titan_pbctopo::VIBE_OBSERVATION_ENDIANNESS
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_FLOATING_FORMAT
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_IDENTITY_HASH_ALGORITHM
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_BINDING_HASH_ALGORITHM
                          << ','
                          << titan_pbctopo::
                                 VIBE_OBSERVATION_PAYLOAD_CHECKSUM_ALGORITHM
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_BINDING_MODE << ','
                          << titan_pbctopo::VIBE_OBSERVATION_NEGATIVE_ZERO_POLICY
                          << ','
                          << titan_pbctopo::VIBE_OBSERVATION_NONFINITE_POLICY
                          << ",experimental_temporal_path,"
                             "experimental_viterbi_diagnostic,"
                          << frame_report_offset + frame_idx << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 self.m_mol[frame_idx].time_of_snap)
                          << ',' << selected + 1 << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_x)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_y)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 chosen.cut_z)
                          << ','
                          << titan_pbctopo::format_pbctopo_report_double(
                                 temporal_frames[frame_idx][selected]
                                     .emission_cost)
                          << ',' << chosen.assignment.signature << ','
                          << (changed ? 1 : 0) << ','
                          << (replay_applied ? 1 : 0) << '\n';
        }
      }
      append_pbctopo_log_line(
          "[pbctopo temporal] mode=viterbi, frames=" +
          to_string(timing.pbctopo_temporal_viterbi_frames) +
          ", candidates=" + to_string(timing.pbctopo_temporal_candidates) +
          ", proposed_changes=" +
          to_string(timing.pbctopo_temporal_proposed_changes) +
          ", applied_changes=" +
          to_string(timing.pbctopo_temporal_path_changes) +
          ", replay=immutable_assignment_delta" +
          ", total_cost=" + to_string(temporal_result.total_cost));
    }
    if (pbctopo_frame_report.is_open()) {
      for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
        if (deferred_frame_report_valid[frame_idx] != 0) {
          auto &report = deferred_frame_reports[frame_idx];
          if (frame_idx < self.pbctopo_frame_status_.size()) {
            self.pbctopo_frame_status_[frame_idx] =
                titan_pbctopo::pbctopo_frame_status_name(report.status);
            self.pbctopo_frame_certified_[frame_idx] =
                report.status == PbctopoFrameStatus::Certified ? 1 : 0;
          }
          if (pbctopo_assignment_set_report.is_open() &&
              report.status != PbctopoFrameStatus::Contradicted &&
              !temporal_cut_candidates[frame_idx].empty()) {
            // selected_framewise marks the greedy choice. After a replay the
            // report's assignment_signature names the temporal output, and
            // consumers derive changed-from-greedy by comparing the two.
            const PbctopoCutCandidate *selected_for_report =
                &greedy_selected_candidates[frame_idx];
            double v1[3]{}, v2[3]{}, v3[3]{};
            self.m_mol[frame_idx].Box.get_v1(v1);
            self.m_mol[frame_idx].Box.get_v2(v2);
            self.m_mol[frame_idx].Box.get_v3(v3);
            const auto normalized_gram =
                titan_pbctopo::pbctopo_temporal_normalized_gram(
                    {v1[0], v1[1], v1[2], v2[0], v2[1], v2[2],
                     v3[0], v3[1], v3[2]});
            write_pbctopo_assignment_set_rows(
                pbctopo_assignment_set_report, report.frame, report.time_ps,
                temporal_cut_candidates[frame_idx], selected_for_report,
                report.identifiability, normalized_gram, better_cut_candidate,
                report.topology_epoch_id,
                report.component_soft_score_valid_assignments_within_domain,
                report.spatial_lift_ambiguous_hard_edges,
                report.spatial_lift_cycle_residuals, report.status);
          }
          write_pbctopo_frame_report_row(pbctopo_frame_report,
                                         report);
        }
      }
    }
  }
}

} // namespace pbctopo_solve

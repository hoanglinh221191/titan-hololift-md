#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

void PbctopoFrameSolver::search_components() {

  // Keep component scoring aligned with the original local graph.
  component_search_used = primary_component_graph_used;
  stage_started = clock::now();
  if (can_use_component_logic) {
    ++timing.component_frames;
    ensure_component_context();
    timing.component_prep_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
    size_t effective_component_count = components.size();
    string component_graph_fallthrough_reason;
    const string component_mode_label =
        pbctopo_component_mode_name(component_mode);
    const PbctopoCutCandidate *component_warm_reference =
        (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
            ? &best_safe_candidate
            : ((warm_state.has_projection_candidate &&
                warm_state.previous_projection_candidate.valid &&
                !warm_state.previous_projection_candidate.ultra_clash)
                   ? &warm_state.previous_projection_candidate
                   : nullptr);
    auto accept_component_cut = [&](PbctopoCutCandidate &candidate,
                                     const vector<Vec3> &candidate_points,
                                     const vector<Int3> *accepted_offsets,
                                     const string &source_label) {
      if (!candidate.valid || candidate.ultra_clash ||
          candidate_points.size() != base_atoms.size()) {
        return false;
      }
      if (!candidate.hard_feasibility.certified() ||
          !candidate.assignment.valid) {
        if (!certify_candidate_for_output(candidate, candidate_points,
                                          accepted_offsets)) {
          record_frame_failure(PbctopoFailureReason::CertificationReject);
          append_pbctopo_log_line(
              "[pbctopo] component candidate failed independent output "
              "certificate at frame " +
              to_string(static_cast<long long>(
                  self.m_mol[frame_idx].time_of_snap)) +
              " via " + source_label + ": hard_clashes=" +
              to_string(candidate.hard_feasibility.hard_clashes) +
              ", assignment_point_residuals=" +
              to_string(
                  candidate.hard_feasibility.assignment_point_residuals) +
              ", topology_edge_residuals=" +
              to_string(candidate.hard_feasibility.hard_edge_residuals));
          return false;
        }
      }
      const PbctopoCutCandidate *virtual_cloud_reference =
          best_virtual_geometry_reference.valid &&
                  !best_virtual_geometry_reference.ultra_clash
              ? &best_virtual_geometry_reference
              : nullptr;
      if (component_candidate_is_outlier(candidate,
                                         virtual_cloud_reference)) {
        if (source_label.find("search") != string::npos ||
            source_label.find("beam") != string::npos) {
          ++timing.component_search_outlier_observations;
        } else {
          ++timing.component_graph_outlier_observations;
        }
        append_pbctopo_log_line(
            "[pbctopo] observe component geometry deviation at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            " via " + source_label + ": principal_rg2_max=" +
            to_string(candidate.principal_rg2_max) +
            ", virtual_principal_rg2_max=" +
            to_string(virtual_cloud_reference->principal_rg2_max) +
            ", kappa2=" + to_string(candidate.shape_kappa2) +
            ", virtual_kappa2=" +
            to_string(virtual_cloud_reference->shape_kappa2) +
            ", asphericity=" +
            to_string(candidate.shape_asphericity) +
            ", virtual_asphericity=" +
            to_string(virtual_cloud_reference->shape_asphericity) +
            ", policy=soft_observation");
      }
      PbctopoOwnerPairSplitDetail split_detail;
      size_t split_owner_pairs = 0;
      double max_owner_raw_minus_minimg_delta = 0.0;
      const bool owner_com_split = candidate_has_split_contacting_owners(
              candidate_points, base_atoms, self.m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, split_owner_pairs,
              max_owner_raw_minus_minimg_delta, &split_detail);
      candidate.split_owner_pairs = split_owner_pairs;
      candidate.max_owner_raw_minus_minimg_delta =
          max_owner_raw_minus_minimg_delta;
      if (owner_com_split) {
        if (split_detail.valid) {
          append_pbctopo_log_line(
              "[pbctopo] observe owner_com_image_gap at frame " +
              to_string(static_cast<long long>(
                  self.m_mol[frame_idx].time_of_snap)) +
              " via " + source_label + ": owners=" +
              to_string(split_detail.owner_a) + "-" +
              to_string(split_detail.owner_b) +
              ", raw_delta_scaled=" +
              format_vec3(split_detail.raw_delta_scaled) +
              ", min_image_delta_scaled=" +
              format_vec3(split_detail.minimg_delta_scaled) +
              ", policy=diagnostic_only");
        }
      }
      bool accepted = false;
      bool better_than_best = false;
      if (!best_safe_candidate.valid ||
          better_cut_candidate(candidate, best_safe_candidate)) {
        best_safe_candidate = candidate;
        if (accepted_offsets != nullptr &&
            accepted_offsets->size() == components.size()) {
          best_safe_candidate.assignment = make_pbctopo_image_assignment(
              *accepted_offsets, components, component_atoms);
        }
        best_safe_points = candidate_points;
        accepted = true;
        better_than_best = true;
      } else if (candidate_close_enough_to_skip_search(candidate,
                                                       best_safe_candidate)) {
        accepted = true;
      }
      if (!accepted)
        return false;
      PbctopoFrameMode candidate_mode = PbctopoFrameMode::ComponentGraph;
      if (source_label.find("beam") != string::npos) {
        candidate_mode = PbctopoFrameMode::ComponentSearchBeam;
      } else if (source_label.find("search") != string::npos) {
        candidate_mode = PbctopoFrameMode::ComponentSearchExact;
      } else if (source_label.find("warm") != string::npos) {
        candidate_mode = PbctopoFrameMode::Warm;
      }
      record_temporal_certified_candidate(candidate, candidate_points,
                                          accepted_offsets);
      record_report_soft_candidate(candidate);
      record_output_candidate(candidate, candidate_points, candidate_mode,
                              accepted_offsets);
      frame_report.mode = candidate_mode;
      component_search_used = true;
      if (accepted_offsets != nullptr &&
          accepted_offsets->size() == components.size()) {
        warm_state.previous_component_offsets = *accepted_offsets;
      } else {
        warm_state.previous_component_offsets =
            extract_offsets_and_time(component_atoms, candidate_points);
      }
      if (better_than_best) {
        append_pbctopo_log_line(
            "[pbctopo] " + source_label + ": mode=" + component_mode_label +
            ", raw_components=" + to_string(raw_component_count) +
            ", effective_components=" +
            to_string(effective_component_count) + ", min_pair_d2=" +
            to_string(candidate.min_pair_d2) + ", kappa2=" +
            to_string(candidate.shape_kappa2) + ", asph=" +
            to_string(candidate.shape_asphericity) + ", bbox=" +
            to_string(candidate.bbox_volume));
      }
      return true;
    };
    auto component_candidate_as_cut =
        [&](const PbctopoComponentCandidate &component_candidate,
            const vector<Vec3> &candidate_points,
            const PbctopoCutCandidate *continuity_incumbent,
            PbctopoCutCandidate &component_as_cut) {
          if (!component_candidate.valid || component_candidate.ultra_clash ||
              candidate_points.size() != base_atoms.size()) {
            return false;
          }
          PointStats component_score_stats;
          for (const auto &point : candidate_points)
            component_score_stats.add(point.x, point.y, point.z);
          double continuity_path = numeric_limits<double>::infinity();
          double continuity_max_d2 = numeric_limits<double>::infinity();
          double continuity_min_margin_A =
              numeric_limits<double>::quiet_NaN();
          size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
          size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
          if (!continuity_metrics_ok(candidate_points,
                                     trajectory_context.continuity_pairs,
                                     runtime_cutoffs.continuity_cutoff,
                                     continuity_path, continuity_max_d2,
                                     continuity_min_margin_A,
                                     continuity_margin_atom_a,
                                     continuity_margin_atom_b,
                                     continuity_incumbent)) {
            return false;
          }
          component_as_cut =
              score_cut_candidate(component_scoring_edges, candidate_points,
                                  1.5, 1.5, 1.5, 0.0, component_score_stats,
                                  component_candidate.shift_norm2);
          component_as_cut.min_pair_d2 = component_candidate.min_pair_d2;
          component_as_cut.continuity_path = continuity_path;
          component_as_cut.continuity_max_d2 = continuity_max_d2;
          component_as_cut.continuity_min_margin_A =
              continuity_min_margin_A;
          component_as_cut.continuity_margin_atom_a =
              continuity_margin_atom_a;
          component_as_cut.continuity_margin_atom_b =
              continuity_margin_atom_b;
          component_as_cut.anchor_mst2 = component_candidate.anchor_mst2;
          component_as_cut.anchor_max_d2 = component_candidate.anchor_max_d2;
          component_as_cut.anchor_graph = component_candidate.anchor_graph;
          component_as_cut.clash_atom_a = component_candidate.clash_atom_a;
          component_as_cut.clash_atom_b = component_candidate.clash_atom_b;
          component_as_cut.component_search_shell_radius =
              component_candidate.component_search_shell_radius;
          component_as_cut.search_domain_audit_known =
              component_candidate.search_domain_audit_known;
          component_as_cut.search_domain_evidence_complete =
              component_candidate.search_domain_evidence_complete;
          component_as_cut.selected_assignment_touches_shell_boundary =
              component_candidate
                  .selected_assignment_touches_shell_boundary;
          component_as_cut.domain_expansion_attempted =
              component_candidate.domain_expansion_attempted;
          component_as_cut.domain_expansion_exhausted =
              component_candidate.domain_expansion_exhausted;
          component_as_cut.bounded_domain_only =
              component_candidate.bounded_domain_only;
          component_as_cut.ultra_clash = component_candidate.ultra_clash;
          component_as_cut.valid = component_candidate.valid;
          return component_as_cut.valid && !component_as_cut.ultra_clash;
        };
    auto accept_component_candidate =
        [&](const PbctopoComponentCandidate &component_candidate,
            const vector<Vec3> &candidate_points,
            const vector<Int3> *accepted_offsets,
            const string &source_label) {
          if (!component_candidate.valid || component_candidate.ultra_clash ||
              candidate_points.size() != base_atoms.size()) {
            return false;
          }
          PbctopoCutCandidate component_as_cut;
          if (!component_candidate_as_cut(
                  component_candidate, candidate_points,
                  &best_safe_candidate, component_as_cut)) {
            record_frame_failure(PbctopoFailureReason::ContinuityReject);
            return false;
          }
          return accept_component_cut(component_as_cut, candidate_points,
                                      accepted_offsets, source_label);
        };
    auto record_component_search_survivors =
        [&](const vector<PbctopoComponentSearchSurvivor> &survivors,
            PbctopoFrameMode mode,
            const PbctopoComponentCandidate *search_best,
            bool search_equivalence_known) {
          timing.component_search_certification_survivors +=
              survivors.size();
          vector<Vec3> survivor_points;
          survivor_points.reserve(base_atoms.size());
          for (const auto &survivor : survivors) {
            if (survivor.component_offsets.size() != components.size())
              continue;
            double shift_norm2 = 0.0;
            build_component_points_from_offsets(
                component_atoms, components, self.m_mol[frame_idx].Box,
                survivor.component_offsets, survivor_points, shift_norm2);
            PbctopoComponentCandidate scored_survivor = survivor.candidate;
            scored_survivor.shift_norm2 = shift_norm2;
            PbctopoCutCandidate survivor_as_cut;
            if (!component_candidate_as_cut(scored_survivor, survivor_points,
                                            nullptr, survivor_as_cut)) {
              continue;
            }
            survivor_as_cut.search_equivalence_known =
                search_equivalence_known && search_best != nullptr &&
                search_best->valid;
            survivor_as_cut.equivalent_to_search_best =
                survivor_as_cut.search_equivalence_known &&
                !better_component_candidate(scored_survivor, *search_best) &&
                !better_component_candidate(*search_best, scored_survivor);
            if (record_output_candidate(survivor_as_cut, survivor_points,
                                        mode,
                                        &survivor.component_offsets)) {
              ++timing.component_search_certification_band_retained;
            }
          }
        };
    if (same_component_warm_layout) {
      vector<Int3> warm_offsets = warm_state.previous_component_offsets;
      if (!rebase_component_offsets(warm_offsets,
                                    self.m_mol[frame_idx].Box)) {
        warm_offsets.clear();
      }
      PbctopoCutCandidate warm_cut = improve_component_offsets_locally(
          component_atoms, components, self.m_mol[frame_idx].Box, shifts,
          component_scoring_edges, steric_exclusion_set,
          trajectory_context.continuity_pairs,
          trajectory_context.continuity_anchor_indices, warm_offsets,
          runtime_cutoffs.clash_cutoff,
          runtime_cutoffs.continuity_cutoff, component_points,
          make_component_refine_callbacks(), anchor_backend, &timing,
          &warm_state, component_warm_reference, &base_anchor_context);
      PbctopoOwnerPairSplitDetail warm_cut_split_detail;
      size_t warm_cut_split_owner_pairs = 0;
      double warm_cut_split_gap = 0.0;
      const bool warm_cut_split =
          warm_cut.valid && !warm_cut.ultra_clash &&
          candidate_has_split_contacting_owners(
              component_points, base_atoms, self.m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, warm_cut_split_owner_pairs,
              warm_cut_split_gap, &warm_cut_split_detail);
      warm_cut.split_owner_pairs = warm_cut_split_owner_pairs;
      warm_cut.max_owner_raw_minus_minimg_delta = warm_cut_split_gap;
      if (warm_cut_split && warm_cut_split_detail.valid) {
        append_pbctopo_log_line(
            "[pbctopo] observe owner_com_image_gap at frame " +
            to_string(
                static_cast<long long>(self.m_mol[frame_idx].time_of_snap)) +
            " via warm_component_local_improve: owners=" +
            to_string(warm_cut_split_detail.owner_a) + "-" +
            to_string(warm_cut_split_detail.owner_b) +
            ", raw_delta_scaled=" +
            format_vec3(warm_cut_split_detail.raw_delta_scaled) +
            ", min_image_delta_scaled=" +
            format_vec3(warm_cut_split_detail.minimg_delta_scaled) +
            ", policy=diagnostic_only");
      }
      if (warm_cut.valid && !warm_cut.ultra_clash) {
        bool warm_accepted = false;
        if (!best_safe_candidate.valid ||
            better_cut_candidate(warm_cut, best_safe_candidate)) {
          best_safe_candidate = warm_cut;
          best_safe_points = component_points;
          warm_accepted = true;
          frame_report.mode = PbctopoFrameMode::Warm;
          record_temporal_certified_candidate(warm_cut, component_points,
                                              &warm_offsets);
          record_report_soft_candidate(warm_cut);
          record_output_candidate(warm_cut, component_points,
                                  PbctopoFrameMode::Warm,
                                  &warm_offsets);
        } else if (candidate_close_enough_to_skip_search(
                       warm_cut, best_safe_candidate)) {
          warm_accepted = true;
        }
        if (warm_accepted) {
          ++timing.component_warm_used;
          warm_state.previous_component_offsets = warm_offsets;
        }
      }
    }
    if (component_mode == PbctopoComponentMode::Exact) {
      double component_shift_norm2 = 0.0;
      PbctopoCutCandidate graph_as_cut;
      vector<Int3> graph_tree_offsets;
      vector<Int3> graph_refined_offsets;
      vector<PbctopoGraphEdge> graph_tree_edges;
      PbctopoCycleSyncMetrics graph_cycle_metrics;
      if (!component_search_used) {
        PbctopoWarmState graph_trial_warm_state = warm_state;
        const auto component_graph_started = clock::now();
        const bool graph_ok = solve_component_offsets_via_pair_graph(
            component_atoms, self.m_mol[frame_idx].Box, components, shifts,
            component_scoring_edges, steric_exclusion_set, component_points,
            trajectory_context.continuity_pairs,
            trajectory_context.continuity_anchor_indices, runtime_cutoffs,
            graph_as_cut, component_shift_norm2, beam_width, &timing,
            make_component_refine_callbacks(), anchor_backend,
            same_component_warm_layout ? &warm_state.previous_component_offsets
                                       : nullptr,
            &graph_trial_warm_state, component_warm_reference,
            &graph_tree_offsets,
            &graph_refined_offsets, &graph_tree_edges, &graph_cycle_metrics,
            &base_anchor_context, &explicit_required_contacts,
            &frame_identifiability);
        timing.component_graph_s +=
            chrono::duration<double>(clock::now() - component_graph_started)
                .count();
        frame_report.soft_pair_cycle_support =
            graph_cycle_metrics.cycle_support;
        frame_report.soft_pair_cycle_mismatch =
            graph_cycle_metrics.cycle_mismatch;
        frame_report.soft_pair_inconsistent_edges =
            graph_cycle_metrics.inconsistent_edge_count;
        frame_report.soft_pair_fundamental_cycle_count =
            graph_cycle_metrics.fundamental_cycle_count;
        frame_report.soft_pair_fundamental_cycle_residual_rms =
            graph_cycle_metrics.fundamental_cycle_residual_rms;
        frame_report.soft_pair_fundamental_cycle_residual_max =
            graph_cycle_metrics.fundamental_cycle_residual_max;
        if (!graph_ok) {
          ++timing.component_graph_failed;
          component_graph_fallthrough_reason = "graph_fail";
          record_frame_failure(PbctopoFailureReason::ComponentGraphFailed);
          clear_component_trial_hints(warm_state);
        }
        if (graph_ok && component_points.size() != base_atoms.size()) {
          ++timing.component_graph_points_mismatch;
          component_graph_fallthrough_reason = "points_mismatch";
        }
        if (graph_ok && component_points.size() == base_atoms.size()) {
          if (!pbctopo_relative_images_identified(frame_identifiability)) {
            ++timing.component_graph_unidentified_output_rejects;
            component_graph_fallthrough_reason = "evidence_disconnected";
            record_frame_failure(PbctopoFailureReason::EvidenceDisconnected);
            append_pbctopo_log_line(
                "[pbctopo] component graph excluded from output band at "
                "frame " +
                to_string(static_cast<long long>(
                    self.m_mol[frame_idx].time_of_snap)) +
                ": identified=0, evidence_components=" +
                to_string(frame_identifiability
                              .evidence_connected_component_count) +
                ", policy=continue_virtual_and_exact_search");
          } else if (accept_component_cut(graph_as_cut, component_points,
                                          &graph_refined_offsets,
                                          "component graph solve")) {
            warm_state = std::move(graph_trial_warm_state);
            ++timing.component_graph_used;
          } else if (!graph_as_cut.valid) {
            ++timing.component_graph_invalid;
            component_graph_fallthrough_reason = "invalid";
            record_frame_failure(PbctopoFailureReason::ProjectionInvalid);
          } else if (graph_as_cut.ultra_clash) {
            ++timing.component_graph_ultraclash;
            component_graph_fallthrough_reason = "ultra_clash";
            record_frame_failure(PbctopoFailureReason::FullVerifyClash);
          } else {
            ++timing.component_graph_not_better_not_close;
            record_frame_failure(PbctopoFailureReason::NotBetterThanIncumbent);
            const char *close_reject_reason =
                candidate_skip_search_reject_reason(graph_as_cut,
                                                    best_safe_candidate);
            component_graph_fallthrough_reason =
                string("not_better_not_close/") +
                (close_reject_reason != nullptr ? close_reject_reason
                                               : "unknown");
          }
        }
      }

      if (!component_search_used || audit_exact_search ||
          components.size() == 2) {
        ++timing.component_search_frames;
        const PbctopoImageAssignment audit_reference_assignment =
            best_safe_candidate.assignment;
        auto component_clash_atom_summary =
            [&](size_t atom_idx) -> string {
          if (atom_idx >= component_atoms.size())
            return "na";
          size_t component_idx = numeric_limits<size_t>::max();
          for (size_t idx = 0; idx < components.size(); ++idx) {
            if (binary_search(components[idx].begin(), components[idx].end(),
                              atom_idx)) {
              component_idx = idx;
              break;
            }
          }
          const PbctopoAtom &atom = component_atoms[atom_idx];
          return to_string(atom_idx) + "/global=" +
                 to_string(atom.global_atom_id) + "/owner=" +
                 to_string(atom.owner) + "/component=" +
                 (component_idx == numeric_limits<size_t>::max()
                      ? string("na")
                      : to_string(component_idx)) +
                 "/heavy=" + (atom.is_heavy ? "yes" : "no");
        };
        append_pbctopo_log_line(
            "[pbctopo] component graph fell through to search at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": mode=" + component_mode_label +
            ", raw_components=" + to_string(raw_component_count) +
            ", effective_components=" +
            to_string(effective_component_count) + ", reason=" +
            component_graph_fallthrough_reason + ", anchor_count=" +
            to_string(trajectory_context.continuity_anchor_indices.size()) +
            ", anchor_policy=full" +
            ", graph_valid=" + string(graph_as_cut.valid ? "yes" : "no") +
            ", graph_ultra_clash=" +
            string(graph_as_cut.ultra_clash ? "yes" : "no") +
            ", graph_anchor_max_d2=" + to_string(graph_as_cut.anchor_max_d2) +
            ", ref_anchor_max_d2=" + to_string(best_safe_candidate.anchor_max_d2) +
            ", graph_anchor_mst2=" + to_string(graph_as_cut.anchor_mst2) +
            ", ref_anchor_mst2=" + to_string(best_safe_candidate.anchor_mst2) +
            ", graph_min_pair_d2=" + to_string(graph_as_cut.min_pair_d2) +
            ", graph_clash_atom_a=" +
            component_clash_atom_summary(graph_as_cut.clash_atom_a) +
            ", graph_clash_atom_b=" +
            component_clash_atom_summary(graph_as_cut.clash_atom_b) +
            ", graph_kappa2=" + to_string(graph_as_cut.shape_kappa2) +
            ", graph_asph=" + to_string(graph_as_cut.shape_asphericity) +
            ", graph_bbox=" + to_string(graph_as_cut.bbox_volume));
        append_pbctopo_log_line(
            "[pbctopo] component graph offsets at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": tree_offsets=" + format_component_offsets(graph_tree_offsets) +
            ", refined_offsets=" + format_component_offsets(graph_refined_offsets) +
            ", tree_edges=" + format_graph_edges(graph_tree_edges));
        const auto component_search_started = clock::now();
        const size_t exact_state_upper_bound =
            pbctopo_component_assignment_upper_bound(
                components.size(), shifts.size(),
                kPbctopoExactAssignmentBudget + 1);
        timing.component_search_exact_state_upper_bound =
            max(timing.component_search_exact_state_upper_bound,
                exact_state_upper_bound);
        const bool use_budgeted_beam =
            !audit_exact_search &&
            exact_state_upper_bound > kPbctopoExactAssignmentBudget;
        const bool retain_complete_feasible_band =
            !use_budgeted_beam &&
            exact_state_upper_bound <= kMaxCompleteOutputCertificationBand;
        const size_t component_search_survivor_limit =
            retain_complete_feasible_band
                ? max<size_t>(1, exact_state_upper_bound)
                : kMaxOutputCertificationBand;
        output_certification_band_limit =
            max(output_certification_band_limit,
                component_search_survivor_limit);
        const PbctopoComponentSearchLimits complete_band_limits{
            0, 0.0, 0, true};
        const PbctopoComponentSearchLimits *component_search_limits =
            audit_exact_search
                ? &audit_exact_limits
                : (retain_complete_feasible_band ? &complete_band_limits
                                                 : nullptr);
        if (use_budgeted_beam) {
          ++timing.component_search_exact_budget_skips;
          append_pbctopo_log_line(
              "[pbctopo] replace exact search with beam at frame " +
              to_string(static_cast<long long>(
                  self.m_mol[frame_idx].time_of_snap)) +
              ": assignment_upper_bound=" +
              to_string(exact_state_upper_bound) +
              ", budget=" + to_string(kPbctopoExactAssignmentBudget));
        }
        const auto search_incumbent =
            (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
                ? &best_safe_candidate
                : nullptr;
        const auto search_incumbent_points =
            (search_incumbent != nullptr &&
             best_safe_points.size() == base_atoms.size())
                ? &best_safe_points
                : nullptr;
        PbctopoComponentSearchDiagnostics search_diagnostics;
        vector<PbctopoComponentSearchSurvivor> search_survivors;
        PbctopoComponentCandidate component_candidate =
            use_budgeted_beam
                ? search_component_shift_candidate_beam(
                      component_atoms, self.m_mol[frame_idx].Box,
                      components, shifts,
                      trajectory_context.continuity_pairs,
                      trajectory_context.continuity_anchor_indices,
                      runtime_cutoffs, component_points, beam_width,
                      make_frame_component_search_callbacks(),
                      same_component_warm_layout
                          ? &warm_state.previous_component_offsets
                          : nullptr,
                      search_incumbent, search_incumbent_points, &timing,
                      anchor_backend, &base_anchor_context,
                      kPbctopoExactAssignmentBudget, &search_diagnostics,
                      &search_survivors, component_search_survivor_limit,
                      &explicit_required_contacts)
                : search_component_shift_candidate(
                      component_atoms, self.m_mol[frame_idx].Box,
                      components, shifts,
                      trajectory_context.continuity_pairs,
                      trajectory_context.continuity_anchor_indices,
                      runtime_cutoffs, component_points,
                      make_frame_component_search_callbacks(),
                      same_component_warm_layout
                          ? &warm_state.previous_component_offsets
                          : nullptr,
                      search_incumbent, search_incumbent_points, &timing,
                      anchor_backend, &base_anchor_context,
                      &search_diagnostics, &search_survivors,
                      component_search_survivor_limit,
                      &explicit_required_contacts,
                      component_search_limits);
        frame_report.component_search_attempted =
            search_diagnostics.attempted;
        frame_report.component_search_completed_under_admissible_bounds =
            search_diagnostics.search_completed_under_admissible_bounds;
        frame_report.component_search_incumbent_bound_pruning_used =
            search_diagnostics.incumbent_bound_pruning_used;
        frame_report.component_search_best_objective_proven =
            search_diagnostics.best_objective_proven;
        frame_report.component_search_equivalent_optima_complete =
            search_diagnostics.equivalent_optima_complete;
        frame_report.component_feasible_assignment_set_enumerated =
            search_diagnostics.feasible_assignment_set_enumerated;
        frame_report.component_feasible_assignments_within_domain =
            search_diagnostics.feasible_assignments_within_domain;
        complete_feasible_band_retained =
            retain_complete_feasible_band &&
            search_diagnostics.feasible_assignment_set_enumerated &&
            search_survivors.size() ==
                search_diagnostics.feasible_assignments_within_domain;
        frame_report
            .component_soft_score_valid_assignments_within_domain =
            search_diagnostics.soft_score_valid_assignments_within_domain;
        frame_report.component_bounded_domain_feasibility_known =
            search_diagnostics.bounded_domain_feasibility_known;
        frame_report.component_bounded_domain_feasible =
            search_diagnostics.bounded_domain_feasible;
        frame_report.component_search_shell_radius =
            search_diagnostics.component_search_shell_radius;
        frame_report.domain_expansion_attempted =
            search_diagnostics.domain_expansion_attempted;
        frame_report.domain_expansion_exhausted =
            search_diagnostics.domain_expansion_exhausted;
        frame_report.bounded_domain_only =
            search_diagnostics.bounded_domain_only;
        frame_report.component_search_exhaustive =
            search_diagnostics.exhaustive;
        frame_report.component_offset_enumeration_exhaustive =
            search_diagnostics.offset_enumeration_exhaustive;
        frame_report.component_all_atom_feasibility_exhaustive =
            search_diagnostics.all_atom_feasibility_exhaustive;
        frame_report.component_certificate_search_exhaustive =
            search_diagnostics.certificate_search_exhaustive;
        frame_report.output_certificate_search_exhaustive =
            frame_report.output_certificate_search_exhaustive ||
            search_diagnostics.certificate_search_exhaustive;
        titan_pbctopo::apply_pbctopo_search_identifiability(
            frame_identifiability,
            search_diagnostics.certificate_search_exhaustive,
            search_diagnostics.equivalent_optima_complete,
            search_diagnostics.equivalent_best_assignments_within_domain,
            search_diagnostics.feasible_assignment_set_enumerated,
            search_diagnostics.feasible_assignments_within_domain);
        frame_report.beam_survivors_exactly_rescored =
            search_diagnostics.beam_survivors_exactly_rescored;
        frame_report.component_search_optimality_gap_known =
            search_diagnostics.optimality_gap_known;
        frame_report.component_search_optimality_gap =
            search_diagnostics.optimality_gap;
        append_pbctopo_log_line(
            "[pbctopo] component search domain audit at frame " +
            to_string(frame_report.frame) + ": shell_radius=" +
            to_string(search_diagnostics.component_search_shell_radius) +
            ", selected_boundary_known=" +
            string(search_diagnostics.selected_assignment_boundary_known
                       ? "yes"
                       : "no") +
            ", selected_touches_boundary=" +
            string(search_diagnostics.selected_assignment_boundary_known
                       ? (search_diagnostics
                                  .selected_assignment_touches_shell_boundary
                              ? "yes"
                              : "no")
                       : "na") +
            ", expansion_attempted=" +
            string(search_diagnostics.domain_expansion_attempted ? "yes"
                                                                  : "no") +
            ", expansion_exhausted=" +
            string(search_diagnostics.domain_expansion_exhausted ? "yes"
                                                                  : "no") +
            ", bounded_domain_only=" +
            string(search_diagnostics.bounded_domain_only ? "yes" : "no") +
            ", hard_feasible_assignments=" +
            to_string(
                search_diagnostics.feasible_assignments_within_domain) +
            ", soft_score_valid_assignments=" +
            to_string(search_diagnostics
                          .soft_score_valid_assignments_within_domain));
        timing.component_search_s +=
            chrono::duration<double>(clock::now() - component_search_started)
                .count();
        frame_report.component_audit_exact_completed =
            audit_exact_search &&
            !search_diagnostics.resource_limit_hit &&
            search_diagnostics.certificate_search_exhaustive &&
            search_diagnostics.feasible_assignment_set_enumerated;
        frame_report.component_audit_exact_incomplete =
            audit_exact_search &&
            !frame_report.component_audit_exact_completed;
        frame_report.component_audit_exact_incomplete_reason =
            !frame_report.component_audit_exact_incomplete
                ? "none"
                : search_diagnostics.resource_limit_hit
                      ? pbctopo_search_stop_reason_name(
                            search_diagnostics.stop_reason)
                      : !search_diagnostics.certificate_search_exhaustive
                            ? "certificate_domain_incomplete"
                            : "feasible_set_not_enumerated";
        frame_report.component_audit_exact_states_visited =
            audit_exact_search ? search_diagnostics.states_visited : 0;
        frame_report.component_audit_exact_wall_seconds =
            audit_exact_search ? search_diagnostics.wall_seconds : 0.0;
        frame_report.component_audit_exact_memory_bytes_estimate =
            audit_exact_search
                ? search_diagnostics.memory_bytes_estimate
                : 0;
        if (frame_report.component_audit_exact_completed)
          ++timing.component_audit_exact_completed;
        if (frame_report.component_audit_exact_incomplete) {
          ++timing.component_audit_exact_incomplete;
          switch (search_diagnostics.stop_reason) {
          case PbctopoSearchStopReason::StateLimit:
            ++timing.component_audit_exact_state_limit;
            break;
          case PbctopoSearchStopReason::WallTimeLimit:
            ++timing.component_audit_exact_wall_time_limit;
            break;
          case PbctopoSearchStopReason::MemoryLimit:
            ++timing.component_audit_exact_memory_limit;
            break;
          case PbctopoSearchStopReason::None:
            break;
          }
          append_pbctopo_log_line(
              "[pbctopo] audit-exact incomplete at frame " +
              to_string(static_cast<long long>(
                  self.m_mol[frame_idx].time_of_snap)) +
              ": reason=" +
              frame_report.component_audit_exact_incomplete_reason +
              ", states=" +
              to_string(search_diagnostics.states_visited) +
              ", wall_s=" +
              to_string(search_diagnostics.wall_seconds) +
              ", memory_bytes_estimate=" +
              to_string(search_diagnostics.memory_bytes_estimate) +
              ", action=retain_certified_initializer");
        } else {
          record_component_search_survivors(
              search_survivors,
              search_diagnostics.beam_used
                  ? PbctopoFrameMode::ComponentSearchBeam
                  : PbctopoFrameMode::ComponentSearchExact,
              &component_candidate,
              !search_diagnostics.beam_used &&
                  search_diagnostics.certificate_search_exhaustive &&
                  search_diagnostics.equivalent_optima_complete);
        }
        vector<Int3> search_offsets;
        PbctopoImageAssignment audited_assignment;
        if (!frame_report.component_audit_exact_incomplete &&
            component_candidate.valid && !component_candidate.ultra_clash &&
            component_points.size() == component_atoms.size()) {
          search_offsets =
              extract_offsets_and_time(component_atoms, component_points);
          if (search_offsets.size() == components.size()) {
            audited_assignment = make_pbctopo_image_assignment(
                search_offsets, components, component_atoms);
          }
        }
        if (frame_report.component_audit_exact_completed &&
            audit_reference_assignment.valid && audited_assignment.valid) {
          frame_report.component_audit_exact_reference_comparison_available =
              true;
          frame_report.component_audit_exact_reference_assignment_match =
              same_pbctopo_image_assignment(audit_reference_assignment,
                                             audited_assignment);
          if (frame_report
                  .component_audit_exact_reference_assignment_match) {
            ++timing.component_audit_exact_reference_matches;
          } else {
            ++timing.component_audit_exact_reference_mismatches;
            append_pbctopo_log_line(
                "[pbctopo] audit-exact differs from certified initializer "
                "at frame " +
                to_string(static_cast<long long>(
                    self.m_mol[frame_idx].time_of_snap)) +
                ": initializer_assignment=" +
                to_string(audit_reference_assignment.signature) +
                ", exact_assignment=" +
                to_string(audited_assignment.signature));
          }
        }
        const vector<Int3> *search_offsets_ptr =
            search_offsets.size() == components.size() ? &search_offsets
                                                       : nullptr;
        if (!frame_report.component_audit_exact_incomplete &&
            accept_component_candidate(
                component_candidate, component_points, search_offsets_ptr,
                search_diagnostics.beam_used
                    ? "component beam budget winner"
                    : "component search winner")) {
          warm_state.previous_component_offsets = search_offsets;
          append_pbctopo_log_line(
              "[pbctopo] component search winner offsets at frame " +
              to_string(static_cast<long long>(
                  self.m_mol[frame_idx].time_of_snap)) +
              ": offsets=" + format_component_offsets(search_offsets));
        }
        if (component_search_used || !best_safe_candidate.valid) {
          append_pbctopo_log_line(
              "[pbctopo] component-shift search: mode=" + component_mode_label +
              ", raw_components=" + to_string(raw_component_count) +
              ", effective_components=" +
              to_string(effective_component_count) + ", components=" +
              to_string(components.size()) + ", ultra_clash=" +
              string(component_candidate.ultra_clash ? "yes" : "no") +
              ", min_pair_d2=" + to_string(component_candidate.min_pair_d2) +
              ", kappa2=" + to_string(component_candidate.shape_kappa2) +
              ", asph=" + to_string(component_candidate.shape_asphericity) +
              ", bbox=" + to_string(component_candidate.bbox_volume));
        }
      }
    } else if (component_mode == PbctopoComponentMode::MergeThenGraph) {
      ++timing.component_large_mode_merge_frames;
      const PbctopoMergedComponentPlan merge_plan =
          build_chain_merged_component_plan(
              component_atoms, self.m_mol[frame_idx].Box, components);
      effective_component_count = merge_plan.merged_components.size();
      if (raw_component_count > effective_component_count)
        timing.component_large_mode_merge_count +=
            raw_component_count - effective_component_count;
      const size_t merged_exact_state_upper_bound =
          pbctopo_component_assignment_upper_bound(
              effective_component_count, shifts.size(),
              kPbctopoExactAssignmentBudget + 1);
      timing.component_search_exact_state_upper_bound =
          max(timing.component_search_exact_state_upper_bound,
              merged_exact_state_upper_bound);
      const bool merged_exact_within_budget =
          merged_exact_state_upper_bound <= kPbctopoExactAssignmentBudget;
      const bool allow_exact_search_on_merged =
          effective_component_count <= exact_component_limit &&
          merged_exact_within_budget;
      if (effective_component_count <= exact_component_limit &&
          !merged_exact_within_budget) {
        ++timing.component_search_exact_budget_skips;
        append_pbctopo_log_line(
            "[pbctopo] skip merged exact search at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": assignment_upper_bound=" +
            to_string(merged_exact_state_upper_bound) +
            ", budget=" + to_string(kPbctopoExactAssignmentBudget) +
            ", policy=component_graph_refine");
      }
      if (!allow_exact_search_on_merged)
        ++timing.component_large_mode_pair_only_frames;

      PbctopoCutCandidate merged_mode_cut;
      vector<Int3> merged_mode_raw_offsets;
      double component_shift_norm2 = 0.0;
      const auto merge_mode_started = clock::now();
      const bool merge_ok = solve_large_component_merge_mode(
          component_atoms, self.m_mol[frame_idx].Box, components, merge_plan,
          shifts, component_scoring_edges, steric_exclusion_set,
          component_points,
          trajectory_context.continuity_pairs,
          trajectory_context.continuity_anchor_indices, runtime_cutoffs,
          merged_mode_cut, component_shift_norm2, &timing, anchor_backend,
          same_component_warm_layout ? &warm_state.previous_component_offsets
                                     : nullptr,
          &warm_state, component_warm_reference, allow_exact_search_on_merged,
          &merged_mode_raw_offsets, &base_anchor_context,
          &explicit_required_contacts);
      timing.component_graph_s +=
          chrono::duration<double>(clock::now() - merge_mode_started).count();
      if (merge_ok &&
          accept_component_cut(merged_mode_cut, component_points,
                               &merged_mode_raw_offsets,
                               "component merge-graph solve")) {
        ++timing.component_graph_used;
      } else {
        ++timing.component_large_mode_fallback_frames;
        record_frame_failure(!merge_ok
                                 ? PbctopoFailureReason::ComponentGraphFailed
                                 : PbctopoFailureReason::NotBetterThanIncumbent);
        const char *merge_reject_reason =
            merge_ok ? candidate_skip_search_reject_reason(
                           merged_mode_cut, best_safe_candidate)
                     : nullptr;
        component_graph_fallthrough_reason = !merge_ok
                                                 ? "graph_invalid"
                                                 : string("not_better_not_close/") +
                                                        (merge_reject_reason != nullptr
                                                             ? merge_reject_reason
                                                             : "unknown");
        append_pbctopo_log_line(
            "[pbctopo] large-component merge mode fell back at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": mode=" + component_mode_label +
            ", raw_components=" + to_string(raw_component_count) +
            ", effective_components=" +
            to_string(effective_component_count) + ", reason=" +
            component_graph_fallthrough_reason);
      }
    } else if (component_mode == PbctopoComponentMode::Beam) {
      ++timing.component_large_mode_beam_frames;
      effective_component_count = components.size();
      const auto beam_started = clock::now();
      PbctopoComponentSearchDiagnostics search_diagnostics;
      vector<PbctopoComponentSearchSurvivor> search_survivors;
      PbctopoComponentCandidate beam_candidate =
          search_component_shift_candidate_beam(
              component_atoms, self.m_mol[frame_idx].Box, components, shifts,
              trajectory_context.continuity_pairs,
              trajectory_context.continuity_anchor_indices, runtime_cutoffs,
              component_points, beam_width,
              make_frame_component_search_callbacks(),
              same_component_warm_layout
                  ? &warm_state.previous_component_offsets
                  : nullptr,
              (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
                  ? &best_safe_candidate
                  : nullptr,
              (best_safe_candidate.valid && !best_safe_candidate.ultra_clash &&
               best_safe_points.size() == base_atoms.size())
                  ? &best_safe_points
                  : nullptr,
              &timing, anchor_backend, &base_anchor_context, 0,
              &search_diagnostics, &search_survivors,
              kMaxOutputCertificationBand, &explicit_required_contacts);
      frame_report.component_search_attempted = search_diagnostics.attempted;
      frame_report.component_search_completed_under_admissible_bounds =
          search_diagnostics.search_completed_under_admissible_bounds;
      frame_report.component_search_incumbent_bound_pruning_used =
          search_diagnostics.incumbent_bound_pruning_used;
      frame_report.component_search_best_objective_proven =
          search_diagnostics.best_objective_proven;
      frame_report.component_search_equivalent_optima_complete =
          search_diagnostics.equivalent_optima_complete;
      frame_report.component_feasible_assignment_set_enumerated =
          search_diagnostics.feasible_assignment_set_enumerated;
      frame_report.component_feasible_assignments_within_domain =
          search_diagnostics.feasible_assignments_within_domain;
      frame_report.component_soft_score_valid_assignments_within_domain =
          search_diagnostics.soft_score_valid_assignments_within_domain;
      frame_report.component_bounded_domain_feasibility_known =
          search_diagnostics.bounded_domain_feasibility_known;
      frame_report.component_bounded_domain_feasible =
          search_diagnostics.bounded_domain_feasible;
      frame_report.component_search_shell_radius =
          search_diagnostics.component_search_shell_radius;
      frame_report.domain_expansion_attempted =
          search_diagnostics.domain_expansion_attempted;
      frame_report.domain_expansion_exhausted =
          search_diagnostics.domain_expansion_exhausted;
      frame_report.bounded_domain_only =
          search_diagnostics.bounded_domain_only;
      frame_report.component_search_exhaustive =
          search_diagnostics.exhaustive;
      frame_report.component_offset_enumeration_exhaustive =
          search_diagnostics.offset_enumeration_exhaustive;
      frame_report.component_all_atom_feasibility_exhaustive =
          search_diagnostics.all_atom_feasibility_exhaustive;
      frame_report.component_certificate_search_exhaustive =
          search_diagnostics.certificate_search_exhaustive;
      frame_report.output_certificate_search_exhaustive =
          frame_report.output_certificate_search_exhaustive ||
          search_diagnostics.certificate_search_exhaustive;
      titan_pbctopo::apply_pbctopo_search_identifiability(
          frame_identifiability,
          search_diagnostics.certificate_search_exhaustive,
          search_diagnostics.equivalent_optima_complete,
          search_diagnostics.equivalent_best_assignments_within_domain,
          false, 0);
      frame_report.beam_survivors_exactly_rescored =
          search_diagnostics.beam_survivors_exactly_rescored;
      frame_report.component_search_optimality_gap_known =
          search_diagnostics.optimality_gap_known;
      frame_report.component_search_optimality_gap =
          search_diagnostics.optimality_gap;
      append_pbctopo_log_line(
          "[pbctopo] component beam domain audit at frame " +
          to_string(frame_report.frame) + ": shell_radius=" +
          to_string(search_diagnostics.component_search_shell_radius) +
          ", selected_boundary_known=" +
          string(search_diagnostics.selected_assignment_boundary_known
                     ? "yes"
                     : "no") +
          ", selected_touches_boundary=" +
          string(search_diagnostics.selected_assignment_boundary_known
                     ? (search_diagnostics
                                .selected_assignment_touches_shell_boundary
                            ? "yes"
                            : "no")
                     : "na") +
          ", bounded_domain_only=" +
          string(search_diagnostics.bounded_domain_only ? "yes" : "no"));
      timing.component_search_s +=
          chrono::duration<double>(clock::now() - beam_started).count();
      record_component_search_survivors(
          search_survivors, PbctopoFrameMode::ComponentSearchBeam,
          &beam_candidate, false);
      if (!accept_component_candidate(beam_candidate, component_points, nullptr,
                                      "component beam solve")) {
        ++timing.component_large_mode_fallback_frames;
        record_frame_failure(beam_candidate.valid && !beam_candidate.ultra_clash
                                 ? PbctopoFailureReason::NotBetterThanIncumbent
                                 : PbctopoFailureReason::BeamEmpty);
        PbctopoCutCandidate beam_as_cut =
            component_candidate_to_cut_candidate(beam_candidate);
        const char *beam_reject_reason =
            (beam_candidate.valid && !beam_candidate.ultra_clash)
                ? candidate_skip_search_reject_reason(beam_as_cut,
                                                      best_safe_candidate)
                : nullptr;
        component_graph_fallthrough_reason =
            beam_candidate.valid && !beam_candidate.ultra_clash
                ? string("not_better_not_close/") +
                      (beam_reject_reason != nullptr
                           ? beam_reject_reason
                           : "unknown")
                : "beam_no_complete_assignment";
        append_pbctopo_log_line(
            "[pbctopo] large-component beam mode fell back at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": mode=" + component_mode_label +
            ", raw_components=" + to_string(raw_component_count) +
            ", effective_components=" +
            to_string(effective_component_count) + ", reason=" +
            component_graph_fallthrough_reason + ", min_pair_d2=" +
            to_string(beam_candidate.min_pair_d2) + ", kappa2=" +
            to_string(beam_candidate.shape_kappa2) + ", asph=" +
            to_string(beam_candidate.shape_asphericity) + ", bbox=" +
            to_string(beam_candidate.bbox_volume));
      }
    }
  } else {
    timing.component_prep_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  warm_state.previous_component_layout = component_layout;
  warm_state.has_component_offsets =
      warm_state.previous_component_offsets.size() == components.size() &&
      !components.empty();
  if (!warm_state.has_component_offsets)
    clear_component_trial_hints(warm_state);

  if (best_safe_candidate.valid && !best_safe_candidate.ultra_clash &&
      best_safe_points.size() == base_atoms.size()) {
    record_output_candidate(
        best_safe_candidate, best_safe_points,
        frame_report.mode == PbctopoFrameMode::None
            ? PbctopoFrameMode::Virtual
            : frame_report.mode);
  }
}

} // namespace pbctopo_solve

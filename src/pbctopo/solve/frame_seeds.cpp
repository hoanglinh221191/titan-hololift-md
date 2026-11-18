#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

bool PbctopoFrameSolver::validate_rescue_candidate_points(const vector<Vec3> &candidate_points,
                                                          const PbctopoCutCandidate *incumbent_cut,
                                                          PbctopoCutCandidate &candidate_out) {
  if (candidate_points.size() != base_atoms.size())
    return false;
  size_t split_owner_pairs = 0;
  double split_gap = 0.0;
  PbctopoOwnerPairSplitDetail detail;
  candidate_has_split_contacting_owners(
      candidate_points, base_atoms, self.m_mol[frame_idx].Box,
      local_hard_required_owner_pairs, split_owner_pairs, split_gap,
      &detail);

  const vector<Int3> rescue_offsets = extract_component_offsets_from_points(
      base_atoms, components, self.m_mol[frame_idx].Box,
      candidate_points);
  const double rescue_shift_norm2 = pbctopo_component_shift_metric(
      self.m_mol[frame_idx].Box, rescue_offsets);

  const PbctopoLatticeMetric rescue_lattice(
      self.m_mol[frame_idx].Box);
  double min_sx = numeric_limits<double>::infinity();
  double min_sy = numeric_limits<double>::infinity();
  double min_sz = numeric_limits<double>::infinity();
  double max_sx = -numeric_limits<double>::infinity();
  double max_sy = -numeric_limits<double>::infinity();
  double max_sz = -numeric_limits<double>::infinity();
  for (const auto &point : candidate_points) {
    const auto scaled = rescue_lattice.cartesian_to_scaled(
        point.x, point.y, point.z);
    const double sx = scaled[0];
    const double sy = scaled[1];
    const double sz = scaled[2];
    min_sx = min(min_sx, sx);
    min_sy = min(min_sy, sy);
    min_sz = min(min_sz, sz);
    max_sx = max(max_sx, sx);
    max_sy = max(max_sy, sy);
    max_sz = max(max_sz, sz);
  }
  auto normalize_rescue_center = [](double center) {
    if (!isfinite(center))
      return 1.5;
    center = fmod(center - 0.5, 1.0);
    if (center < 0.0)
      center += 1.0;
    return clamp_virtual_box_center(center + 0.5);
  };
  const double rescue_cut_x =
      normalize_rescue_center(0.5 * (min_sx + max_sx));
  const double rescue_cut_y =
      normalize_rescue_center(0.5 * (min_sy + max_sy));
  const double rescue_cut_z =
      normalize_rescue_center(0.5 * (min_sz + max_sz));

  const PointStats rescue_stats = build_point_stats(candidate_points);
  candidate_out = score_cut_candidate(
      component_scoring_edges, candidate_points, rescue_cut_x,
      rescue_cut_y, rescue_cut_z, 0.0, rescue_stats,
      rescue_shift_norm2);
  double continuity_path = numeric_limits<double>::infinity();
  double continuity_max_d2 = numeric_limits<double>::infinity();
  double continuity_min_margin_A =
      numeric_limits<double>::quiet_NaN();
  size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
  size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
  double anchor_mst2 = numeric_limits<double>::infinity();
  double anchor_max_d2 = numeric_limits<double>::infinity();
  PbctopoAnchorGraphScore anchor_graph_score;
  const bool rescue_ok =
      candidate_out.valid &&
      continuity_metrics_ok(candidate_points,
                            trajectory_context.continuity_pairs,
                            runtime_cutoffs.continuity_cutoff,
                            continuity_path, continuity_max_d2,
                            continuity_min_margin_A,
                            continuity_margin_atom_a,
                            continuity_margin_atom_b,
                            incumbent_cut) &&
      anchor_graph_metrics_enhanced(
          candidate_points, trajectory_context.continuity_anchor_indices,
          trajectory_context.continuity_pairs, anchor_mst2,
          anchor_max_d2, anchor_graph_score, incumbent_cut,
          PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
          &base_anchor_context);
  if (!rescue_ok)
    return false;

  candidate_out.continuity_path = continuity_path;
  candidate_out.continuity_max_d2 = continuity_max_d2;
  candidate_out.continuity_min_margin_A = continuity_min_margin_A;
  candidate_out.continuity_margin_atom_a = continuity_margin_atom_a;
  candidate_out.continuity_margin_atom_b = continuity_margin_atom_b;
  candidate_out.split_owner_pairs = split_owner_pairs;
  candidate_out.max_owner_raw_minus_minimg_delta = split_gap;
  candidate_out.anchor_mst2 = anchor_mst2;
  candidate_out.anchor_max_d2 = anchor_max_d2;
  candidate_out.anchor_graph = anchor_graph_score;
  analyze_cut_candidate_ultraclash(
      candidate_points, steric_exclusion_set, candidate_out,
      runtime_cutoffs.clash_cutoff, &base_atoms,
      &atom_component_index);
  return !candidate_out.ultra_clash;
}

void PbctopoFrameSolver::record_report_soft_candidate(const PbctopoCutCandidate &candidate) {
  if (!candidate.valid)
    return;
  ++report_soft_valid_candidates;
  if (!candidate.ultra_clash)
    ++report_nonclashing_candidates;
}

void PbctopoFrameSolver::collect_seeds() {

  warm_points.reserve(base_atoms.size());
  coarse_warm_points.reserve(projection_atoms.size());
  prefetched_safe_points.reserve(base_atoms.size());
  prefetched_safe_mode = PbctopoFrameMode::None;
  split_rescue_points.reserve(base_atoms.size());
  // Owner COM separation is diagnostic-only; it must not drive rescue.
  split_rescue_available = false;
  if (warm_state.has_projection_candidate) {
    const auto early_projection_started = clock::now();
    const PbctopoCutCandidate warm_projection_reference =
        warm_state.previous_projection_candidate;
    bool warm_projection_accepted = false;
    bool warm_projection_full_evaluated = false;
    PbctopoCutCandidate early_projection;
    PbctopoCutCandidate early_projection_coarse = evaluate_virtual_box_candidate(
        projection_atoms, projection_edges, projection_edge_set,
        translation_cache, self.m_mol[frame_idx].Box,
        warm_state.previous_projection_candidate.cut_x,
        warm_state.previous_projection_candidate.cut_y,
        warm_state.previous_projection_candidate.cut_z,
        warm_state.previous_projection_candidate.gap_score, coarse_warm_points,
        runtime_cutoffs.clash_cutoff, &projection_continuity_pairs,
        &projection_anchor_indices, &projection_hard_required_owner_pairs,
        runtime_cutoffs.continuity_cutoff, nullptr,
        skip_projection_candidate_clash, numeric_limits<double>::quiet_NaN(),
        nullptr, &projection_anchor_context,
        PbctopoVirtualSeedSource::Previous);
    if (skip_projection_candidate_clash)
      ++timing.projection_clash_skip_evals;
    if (early_projection_coarse.valid && !early_projection_coarse.ultra_clash) {
      warm_projection_full_evaluated = true;
      early_projection = evaluate_virtual_box_candidate(
          base_atoms, local_edges, steric_exclusion_set, translation_cache,
          self.m_mol[frame_idx].Box,
          warm_state.previous_projection_candidate.cut_x,
          warm_state.previous_projection_candidate.cut_y,
          warm_state.previous_projection_candidate.cut_z,
          warm_state.previous_projection_candidate.gap_score, warm_points,
          runtime_cutoffs.clash_cutoff, &trajectory_context.continuity_pairs,
          &trajectory_context.continuity_anchor_indices,
          &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
          nullptr, false, self.m_mol[frame_idx].time_of_snap,
          "warm_projection", &base_anchor_context,
          PbctopoVirtualSeedSource::Previous, &atom_component_index);
      if (early_projection.valid && !early_projection.ultra_clash &&
        candidate_close_enough_to_skip_search(
            early_projection, warm_projection_reference)) {
        warm_projection_accepted = true;
        ++timing.early_projection_used;
        warm_state.previous_projection_candidate = early_projection;
        warm_state.has_projection_candidate = true;
        if (better_cut_candidate(early_projection,
                                 best_verified_virtual_candidate)) {
          best_verified_virtual_candidate = early_projection;
        }
        if (better_cut_candidate(early_projection, prefetched_safe_candidate)) {
          prefetched_safe_candidate = early_projection;
          prefetched_safe_points = warm_points;
          prefetched_safe_mode = PbctopoFrameMode::EarlyProjection;
          frame_report.mode = PbctopoFrameMode::EarlyProjection;
        }
      }
    }
    if (!warm_projection_accepted) {
      ++timing.pbctopo_warm_projection_rejects;
      const char *reason =
          !early_projection_coarse.valid
              ? "coarse_invalid"
              : (early_projection_coarse.ultra_clash
                     ? "coarse_ultraclash"
                     : (!warm_projection_full_evaluated
                            ? "full_not_evaluated"
                            : candidate_skip_search_reject_reason(
                                  early_projection,
                                  warm_projection_reference)));
      append_pbctopo_log_line(
          "[pbctopo] warm projection rejected at frame " +
          to_string(static_cast<long long>(
              self.m_mol[frame_idx].time_of_snap)) +
          ": reason=" + (reason != nullptr ? reason : "unknown"));
    }
    timing.early_projection_s +=
        chrono::duration<double>(clock::now() - early_projection_started)
            .count();
  }
  if (can_use_component_logic && warm_state.has_component_offsets &&
      same_component_layout(components,
                            warm_state.previous_component_layout)) {
    const auto warm_component_started = clock::now();
    ensure_component_context();
    vector<Int3> warm_offsets = warm_state.previous_component_offsets;
    if (!rebase_component_offsets(warm_offsets,
                                  self.m_mol[frame_idx].Box)) {
      warm_offsets.clear();
    }
    PbctopoCutCandidate early_component = score_component_offset_solution(
        component_atoms, components, self.m_mol[frame_idx].Box, warm_offsets,
        local_edges, steric_exclusion_set, runtime_cutoffs.clash_cutoff,
        trajectory_context.continuity_pairs,
        trajectory_context.continuity_anchor_indices,
        runtime_cutoffs.continuity_cutoff, warm_points,
        make_component_refine_callbacks(),
        &warm_state.previous_projection_candidate, nullptr, true,
        PbctopoRefineScorePhase::Exact, nullptr, nullptr, nullptr,
        anchor_backend, &base_anchor_context);
    PbctopoOwnerPairSplitDetail early_component_split_detail;
    size_t early_component_split_owner_pairs = 0;
    double early_component_split_gap = 0.0;
    const bool early_component_split =
        early_component.valid && !early_component.ultra_clash &&
        candidate_has_split_contacting_owners(
            warm_points, base_atoms, self.m_mol[frame_idx].Box,
            local_hard_required_owner_pairs, early_component_split_owner_pairs,
            early_component_split_gap, &early_component_split_detail);
    early_component.split_owner_pairs = early_component_split_owner_pairs;
    early_component.max_owner_raw_minus_minimg_delta =
        early_component_split_gap;
    if (early_component_split && early_component_split_detail.valid) {
      append_pbctopo_log_line(
          "[pbctopo] observe owner_com_image_gap at frame " +
          to_string(static_cast<long long>(self.m_mol[frame_idx].time_of_snap)) +
          " via warm_component: owners=" +
          to_string(early_component_split_detail.owner_a) + "-" +
          to_string(early_component_split_detail.owner_b) +
          ", raw_delta_scaled=" +
          format_vec3(early_component_split_detail.raw_delta_scaled) +
          ", min_image_delta_scaled=" +
          format_vec3(early_component_split_detail.minimg_delta_scaled) +
          ", policy=diagnostic_only");
    }
    const bool warm_component_accepted =
        early_component.valid && !early_component.ultra_clash &&
        candidate_close_enough_to_skip_search(
            early_component, warm_state.previous_projection_candidate);
    if (warm_component_accepted) {
      ++timing.component_warm_used;
      warm_state.previous_projection_candidate = early_component;
      warm_state.previous_component_offsets = warm_offsets;
      warm_state.previous_component_layout = component_layout;
      warm_state.has_projection_candidate = true;
      warm_state.has_component_offsets = true;
      if (better_cut_candidate(early_component, prefetched_safe_candidate)) {
        prefetched_safe_candidate = early_component;
        prefetched_safe_points = warm_points;
        prefetched_safe_mode = PbctopoFrameMode::Warm;
        frame_report.mode = PbctopoFrameMode::Warm;
      }
    } else {
      ++timing.pbctopo_warm_component_rejects;
      const char *reason = candidate_skip_search_reject_reason(
          early_component, warm_state.previous_projection_candidate);
      append_pbctopo_log_line(
          "[pbctopo] warm component rejected at frame " +
          to_string(static_cast<long long>(
              self.m_mol[frame_idx].time_of_snap)) +
          ": reason=" + (reason != nullptr ? reason : "unknown"));
    }
    timing.warm_component_s +=
        chrono::duration<double>(clock::now() - warm_component_started)
            .count();
  }

  component_scoring_edges = local_edges;
  primary_component_graph_used = false;
  if (can_use_component_logic &&
      component_mode == PbctopoComponentMode::Exact) {
    ++timing.component_graph_primary_attempts;
    ensure_component_context();
    PbctopoCutCandidate primary_cut;
    double primary_shift_norm2 = 0.0;
    vector<Int3> primary_tree_offsets;
    vector<Int3> primary_refined_offsets;
    vector<PbctopoGraphEdge> primary_tree_edges;
    PbctopoCycleSyncMetrics primary_cycle_metrics;
    const PbctopoWarmState primary_graph_warm_baseline = warm_state;
    PbctopoWarmState primary_graph_accepted_warm_state;
    bool primary_graph_unidentified = false;
    const PbctopoCutCandidate *primary_warm_reference =
        warm_state.has_projection_candidate &&
                warm_state.previous_projection_candidate.valid &&
                !warm_state.previous_projection_candidate.ultra_clash
            ? &warm_state.previous_projection_candidate
            : nullptr;
    auto try_primary_component_shell = [&](int shell_radius) {
      vector<PbctopoShift> expanded_shifts;
      const vector<PbctopoShift> *shell_shifts = &shifts;
      if (component_context_shell_radius != shell_radius) {
        expanded_shifts = build_topology_shifts(
            self.m_mol[frame_idx].Box, shell_radius);
        shell_shifts = &expanded_shifts;
      }
      timing.component_image_shell_max_radius =
          max(timing.component_image_shell_max_radius,
              static_cast<size_t>(shell_radius));
      timing.component_image_shell_max_shift_count =
          max(timing.component_image_shell_max_shift_count,
              shell_shifts->size());
      primary_cut = {};
      primary_shift_norm2 = 0.0;
      primary_tree_offsets.clear();
      primary_refined_offsets.clear();
      primary_tree_edges.clear();
      primary_cycle_metrics = {};
      component_points.clear();
      PbctopoWarmState shell_trial_warm_state =
          primary_graph_warm_baseline;

      const auto primary_graph_started = clock::now();
      const bool graph_ok = solve_component_offsets_via_pair_graph(
          component_atoms, self.m_mol[frame_idx].Box, components,
          *shell_shifts,
          component_scoring_edges, steric_exclusion_set, component_points,
          trajectory_context.continuity_pairs,
          trajectory_context.continuity_anchor_indices, runtime_cutoffs,
          primary_cut, primary_shift_norm2, beam_width, &timing,
          make_component_refine_callbacks(), anchor_backend,
          same_component_warm_layout
              ? &warm_state.previous_component_offsets
              : nullptr,
          &shell_trial_warm_state, primary_warm_reference,
          &primary_tree_offsets,
          &primary_refined_offsets, &primary_tree_edges,
          &primary_cycle_metrics, &base_anchor_context,
          &explicit_required_contacts, &frame_identifiability);
      timing.component_graph_s +=
          chrono::duration<double>(clock::now() - primary_graph_started)
              .count();
      frame_report.soft_pair_cycle_support =
          primary_cycle_metrics.cycle_support;
      frame_report.soft_pair_cycle_mismatch =
          primary_cycle_metrics.cycle_mismatch;
      frame_report.soft_pair_inconsistent_edges =
          primary_cycle_metrics.inconsistent_edge_count;
      frame_report.soft_pair_fundamental_cycle_count =
          primary_cycle_metrics.fundamental_cycle_count;
      frame_report.soft_pair_fundamental_cycle_residual_rms =
          primary_cycle_metrics.fundamental_cycle_residual_rms;
      frame_report.soft_pair_fundamental_cycle_residual_max =
          primary_cycle_metrics.fundamental_cycle_residual_max;

      size_t split_pairs = 0;
      double split_gap = 0.0;
      const bool split =
          graph_ok && primary_cut.valid && !primary_cut.ultra_clash &&
          component_points.size() == base_atoms.size() &&
          candidate_has_split_contacting_owners(
              component_points, base_atoms, self.m_mol[frame_idx].Box,
              local_hard_required_owner_pairs, split_pairs, split_gap);
      if (split) {
        primary_cut.split_owner_pairs = split_pairs;
        primary_cut.max_owner_raw_minus_minimg_delta = split_gap;
      }
      const bool candidate_complete =
          graph_ok && primary_cut.valid && !primary_cut.ultra_clash &&
          component_points.size() == base_atoms.size() &&
          primary_refined_offsets.size() == components.size();
      if (!candidate_complete)
        return false;
      if (!pbctopo_relative_images_identified(frame_identifiability)) {
        primary_graph_unidentified = true;
        return false;
      }
      if (!certify_candidate_for_output(primary_cut, component_points,
                                        &primary_refined_offsets)) {
        return false;
      }
      primary_graph_accepted_warm_state =
          std::move(shell_trial_warm_state);
      return true;
    };

    const int primary_initial_shell_radius =
        component_shell_radius_explicit
            ? configured_component_shell_radius
            : (components.size() == 2 ? 2 : 1);
    bool primary_certified =
        try_primary_component_shell(primary_initial_shell_radius);
    if (!primary_certified && !component_shell_radius_explicit &&
        primary_initial_shell_radius < 2) {
      ++timing.component_image_shell_expansions;
      primary_certified = try_primary_component_shell(2);
      if (primary_certified)
        ++timing.component_image_shell_retry_successes;
    }
    if (primary_certified) {
      warm_state = std::move(primary_graph_accepted_warm_state);
      record_temporal_certified_candidate(primary_cut, component_points,
                                          &primary_refined_offsets);
      prefetched_safe_candidate = primary_cut;
      prefetched_safe_points = component_points;
      prefetched_safe_mode = PbctopoFrameMode::ComponentGraph;
      warm_state.previous_component_offsets = primary_refined_offsets;
      warm_state.previous_component_layout = component_layout;
      warm_state.has_component_offsets = true;
      frame_report.mode = PbctopoFrameMode::ComponentGraph;
      primary_component_graph_used = true;
      ++timing.component_graph_primary_certified;
      ++timing.component_graph_primary_virtual_skips;
      ++timing.component_graph_used;
    } else {
      ++timing.component_graph_primary_fallbacks;
      if (primary_graph_unidentified) {
        ++timing.component_graph_primary_unidentified;
        append_pbctopo_log_line(
            "[pbctopo] primary component graph cannot bypass virtual search "
            "at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": identified=0, evidence_components=" +
            to_string(
                frame_identifiability.evidence_connected_component_count) +
            ", unconstrained_relative_components=" +
            to_string(
                frame_identifiability.unconstrained_relative_components));
      }
    }
  }

  if (!primary_component_graph_used) {
    stage_started = clock::now();
    collect_projection_volume_seeds(projection_atoms, 16,
                                    kProjectionTopSeeds, 0.22,
                                    volume_seeds);
  if (warm_state.has_projection_candidate) {
    PbctopoSeed3D previous_seed;
    previous_seed.cx =
        clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_x);
    previous_seed.cy =
        clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_y);
    previous_seed.cz =
        clamp_virtual_box_center(warm_state.previous_projection_candidate.cut_z);
    previous_seed.score =
        max(1.0, warm_state.previous_projection_candidate.gap_score + 1.0);
    previous_seed.source = PbctopoVirtualSeedSource::Previous;
    volume_seeds.push_back(previous_seed);
    ++timing.previous_seed_used;
  }
  if (kinematic_seed_history_ready) {
    const PbctopoVirtualSeedCenter predicted_center =
        extrapolate_virtual_seed_center(
            virtual_seed_center_from_candidate(kinematic_seed_previous),
            virtual_seed_center_from_candidate(kinematic_seed_latest),
            PbctopoLatticeMetric(self.m_mol[frame_idx].Box));
    PbctopoSeed3D kinematic_seed;
    kinematic_seed.cx = predicted_center.cx;
    kinematic_seed.cy = predicted_center.cy;
    kinematic_seed.cz = predicted_center.cz;
    kinematic_seed.score =
        max(1.0, kinematic_seed_latest.gap_score + 0.5);
    kinematic_seed.source = PbctopoVirtualSeedSource::Kinematic;
    volume_seeds.push_back(kinematic_seed);
    ++timing.pbctopo_kinematic_seed_candidates;
  }
  timing.seed_s += chrono::duration<double>(clock::now() - stage_started).count();

  virtual_seed_hints.reserve(volume_seeds.size());
  for (const auto &seed : volume_seeds) {
    virtual_seed_hints.push_back(
        {seed.cx, seed.cy, seed.cz, seed.score, seed.source});
  }
  stage_started = clock::now();
  PbctopoVirtualDiscreteSearchOptions discrete_options;
  discrete_options.top_k_per_axis = 4;
  discrete_options.max_exhaustive_states = 4096;
  build_pbctopo_virtual_discrete_search_plan(
      projection_atoms, projection_edges, self.m_mol[frame_idx].Box,
      virtual_seed_hints, discrete_options, virtual_search_workspace,
      virtual_search_plan);
  const double seam_sweep_s =
      chrono::duration<double>(clock::now() - stage_started).count();
  timing.event_planes_s += seam_sweep_s;
  timing.pbctopo_virtual_seam_sweep_s += seam_sweep_s;
  ++timing.pbctopo_virtual_discrete_frames;
  timing.pbctopo_virtual_axis_states_x +=
      virtual_search_plan.x_states.size();
  timing.pbctopo_virtual_axis_states_y +=
      virtual_search_plan.y_states.size();
  timing.pbctopo_virtual_axis_states_z +=
      virtual_search_plan.z_states.size();
  timing.pbctopo_virtual_sweep_events_x +=
      virtual_search_plan.event_count_x;
  timing.pbctopo_virtual_sweep_events_y +=
      virtual_search_plan.event_count_y;
  timing.pbctopo_virtual_sweep_events_z +=
      virtual_search_plan.event_count_z;
  if (virtual_search_plan.workspace_reused)
    ++timing.pbctopo_virtual_sweep_workspace_reuse_frames;
  timing.pbctopo_virtual_sweep_workspace_peak_bytes =
      max(timing.pbctopo_virtual_sweep_workspace_peak_bytes,
          virtual_search_plan.workspace_bytes_estimate);
  if (numeric_limits<uint64_t>::max() - timing.pbctopo_virtual_state_space <
      virtual_search_plan.total_state_space) {
    timing.pbctopo_virtual_state_space = numeric_limits<uint64_t>::max();
  } else {
    timing.pbctopo_virtual_state_space +=
        virtual_search_plan.total_state_space;
  }
  timing.pbctopo_virtual_states_evaluated +=
      virtual_search_plan.states.size();
  if (virtual_search_plan.search_exhaustive)
    ++timing.pbctopo_virtual_seam_enumeration_exhaustive_frames;
  else
    ++timing.pbctopo_virtual_search_shortlist_frames;
  if (virtual_search_plan.branch_and_bound_used)
    ++timing.pbctopo_virtual_branch_and_bound_frames;
  if (isfinite(virtual_search_plan.omitted_seam_lower_bound)) {
    timing.pbctopo_virtual_omitted_seam_lower_bound_min =
        min(timing.pbctopo_virtual_omitted_seam_lower_bound_min,
            virtual_search_plan.omitted_seam_lower_bound);
  }
  if (isfinite(virtual_search_plan.lower_bound_gap)) {
    timing.pbctopo_virtual_lower_bound_gap_min =
        min(timing.pbctopo_virtual_lower_bound_gap_min,
            virtual_search_plan.lower_bound_gap);
  }
  frame_report.virtual_seam_enumeration_exhaustive =
      virtual_search_plan.search_exhaustive;
  frame_report.virtual_projection_scoring_exhaustive =
      virtual_search_plan.search_exhaustive &&
      virtual_search_plan.states.size() ==
          virtual_search_plan.total_state_space;
  frame_report.virtual_full_atom_verification_exhaustive = false;
  if (frame_report.virtual_projection_scoring_exhaustive)
    ++timing.pbctopo_virtual_projection_scoring_exhaustive_frames;
  if (frame_report.virtual_full_atom_verification_exhaustive)
    ++timing.pbctopo_virtual_full_atom_verification_exhaustive_frames;
  frame_report.virtual_search_exhaustive =
      frame_report.virtual_seam_enumeration_exhaustive &&
      frame_report.virtual_projection_scoring_exhaustive &&
      frame_report.virtual_full_atom_verification_exhaustive;
  if (frame_report.virtual_search_exhaustive)
    ++timing.pbctopo_virtual_search_exhaustive_frames;
  frame_report.virtual_branch_and_bound_used =
      virtual_search_plan.branch_and_bound_used;
  frame_report.virtual_state_space = virtual_search_plan.total_state_space;
  frame_report.virtual_states_evaluated = virtual_search_plan.states.size();
  frame_report.virtual_omitted_seam_lower_bound =
      virtual_search_plan.omitted_seam_lower_bound;
  frame_report.virtual_lower_bound_gap = virtual_search_plan.lower_bound_gap;
  }

  best_safe_certification_rejected = false;
  safe_candidates = 0;
  frame_anchor_ambiguity_counted = false;
  output_certification_band_limit = kMaxOutputCertificationBand;
  complete_feasible_band_retained = false;
  output_candidate_band.reserve(kMaxOutputCertificationBand);
}

} // namespace pbctopo_solve

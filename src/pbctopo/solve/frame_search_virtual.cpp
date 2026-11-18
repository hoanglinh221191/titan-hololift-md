#include "pbctopo/solve/solver.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

void PbctopoFrameSolver::accumulate_worker_lattice_audit(const titan_pbctopo::PbctopoLatticeQueryAudit &before) {
  const auto after =
      titan_pbctopo::pbctopo_lattice_query_audit();
  const auto delta =
      titan_pbctopo::pbctopo_lattice_query_audit_delta(before, after);
  coarse_worker_lattice_queries.fetch_add(delta.queries,
                                           memory_order_relaxed);
  coarse_worker_lattice_failures.fetch_add(delta.failures,
                                            memory_order_relaxed);
  coarse_worker_lattice_ambiguities.fetch_add(delta.ambiguous,
                                               memory_order_relaxed);
  coarse_worker_lattice_budget_failures.fetch_add(
      delta.enumeration_budget_failures, memory_order_relaxed);
}

void PbctopoFrameSolver::search_virtual() {
  placed_points.reserve(projection_atoms.size());
  best_safe_points.reserve(base_atoms.size());
  coarse_candidates.reserve(12);
  coarse_seed_options.reserve(virtual_search_plan.states.size());
  if (prefetched_safe_candidate.valid && !prefetched_safe_candidate.ultra_clash &&
      prefetched_safe_points.size() == base_atoms.size()) {
    best_candidate = prefetched_safe_candidate;
    best_safe_candidate = prefetched_safe_candidate;
    best_safe_points = prefetched_safe_points;
    safe_candidates = 1;
    record_temporal_certified_candidate(prefetched_safe_candidate,
                                        prefetched_safe_points);
    record_report_soft_candidate(prefetched_safe_candidate);
    record_output_candidate(prefetched_safe_candidate,
                            prefetched_safe_points,
                            prefetched_safe_mode);
  }

  stage_started = clock::now();
  has_kinematic_hint = any_of(
      virtual_seed_hints.begin(), virtual_seed_hints.end(),
      [](const PbctopoVirtualDiscreteSeedHint &hint) {
        return hint.source == PbctopoVirtualSeedSource::Kinematic;
      });
  kinematic_states_evaluated = 0;
  for (size_t state_idx = 0;
       state_idx < virtual_search_plan.states.size(); ++state_idx) {
    const auto &state = virtual_search_plan.states[state_idx];
    PbctopoSeed3D seed;
    seed.cx = state.cx;
    seed.cy = state.cy;
    seed.cz = state.cz;
    // Density and temporal hints order exact states only; they are not part
    // of the physical candidate score.
    seed.score = 0.0;
    seed.ix = state.ix;
    seed.iy = state.iy;
    seed.iz = state.iz;
    seed.source = state.source;
    if (seed.source == PbctopoVirtualSeedSource::Kinematic) {
      ++timing.pbctopo_kinematic_seed_evaluated;
      ++kinematic_states_evaluated;
    }
    coarse_seed_options.push_back(
        {static_cast<uint64_t>(state_idx), seed});
  }
  if (has_kinematic_hint && kinematic_states_evaluated == 0)
    ++timing.pbctopo_kinematic_seed_deduped;
  timing.coarse_candidates += coarse_seed_options.size();
  requested_threads =
      (Arguments.nt > 1) ? Arguments.nt : static_cast<size_t>(1);
  executor_capacity =
      (executor != nullptr) ? max<size_t>(1, executor->capacity()) : requested_threads;
  coarse_threads =
      min(executor_capacity, coarse_seed_options.size());
  coarse_results.assign(coarse_seed_options.size(), PbctopoCutCandidate{});
  if (executor != nullptr && coarse_threads > 1 && coarse_seed_options.size() >= 4 &&
      projection_atoms.size() >= 2000) {
    executor->parallel_for(
        coarse_seed_options.size(), coarse_threads, 1,
        [&](size_t, size_t begin, size_t end) {
          const auto worker_lattice_audit_before =
              titan_pbctopo::pbctopo_lattice_query_audit();
          vector<Vec3> local_points;
          local_points.reserve(projection_atoms.size());
          for (size_t option_idx = begin; option_idx < end; ++option_idx) {
            const auto &seed = coarse_seed_options[option_idx].seed;
            coarse_results[option_idx] = evaluate_virtual_box_candidate(
                projection_atoms, projection_edges, projection_edge_set,
                translation_cache, self.m_mol[frame_idx].Box, seed.cx,
                seed.cy, seed.cz, seed.score,
                local_points, runtime_cutoffs.clash_cutoff,
                &projection_continuity_pairs, &projection_anchor_indices,
                &projection_hard_required_owner_pairs,
                runtime_cutoffs.continuity_cutoff, nullptr,
                skip_projection_candidate_clash,
                numeric_limits<double>::quiet_NaN(), nullptr,
                &projection_anchor_context, seed.source);
            if (skip_projection_candidate_clash)
                coarse_projection_skip_evals.fetch_add(
                  1, memory_order_relaxed);
          }
          accumulate_worker_lattice_audit(worker_lattice_audit_before);
        });
  } else if (coarse_threads > 1 && coarse_seed_options.size() >= 4 &&
             projection_atoms.size() >= 2000) {
    vector<titan::jthread> workers;
    workers.reserve(coarse_threads);
    const size_t chunk =
        (coarse_seed_options.size() + coarse_threads - 1) / coarse_threads;
    for (size_t thread_idx = 0; thread_idx < coarse_threads; ++thread_idx) {
      const size_t begin = thread_idx * chunk;
      const size_t end = min(begin + chunk, coarse_seed_options.size());
      if (begin >= end)
        continue;
      workers.emplace_back([&, begin, end]() {
        const auto worker_lattice_audit_before =
            titan_pbctopo::pbctopo_lattice_query_audit();
        vector<Vec3> local_points;
        local_points.reserve(projection_atoms.size());
        for (size_t option_idx = begin; option_idx < end; ++option_idx) {
          const auto &seed = coarse_seed_options[option_idx].seed;
          coarse_results[option_idx] = evaluate_virtual_box_candidate(
              projection_atoms, projection_edges, projection_edge_set,
              translation_cache, self.m_mol[frame_idx].Box, seed.cx, seed.cy,
              seed.cz, seed.score,
              local_points, runtime_cutoffs.clash_cutoff,
              &projection_continuity_pairs, &projection_anchor_indices,
              &projection_hard_required_owner_pairs,
              runtime_cutoffs.continuity_cutoff, nullptr,
              skip_projection_candidate_clash,
              numeric_limits<double>::quiet_NaN(), nullptr,
              &projection_anchor_context, seed.source);
          if (skip_projection_candidate_clash)
            coarse_projection_skip_evals.fetch_add(
                1, memory_order_relaxed);
        }
        accumulate_worker_lattice_audit(worker_lattice_audit_before);
      });
    }
  } else {
    for (size_t option_idx = 0; option_idx < coarse_seed_options.size();
         ++option_idx) {
      const auto &seed = coarse_seed_options[option_idx].seed;
      coarse_results[option_idx] = evaluate_virtual_box_candidate(
          projection_atoms, projection_edges, projection_edge_set,
          translation_cache, self.m_mol[frame_idx].Box, seed.cx, seed.cy,
          seed.cz, seed.score,
          placed_points, runtime_cutoffs.clash_cutoff,
          &projection_continuity_pairs, &projection_anchor_indices,
          &projection_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
          nullptr, skip_projection_candidate_clash,
          numeric_limits<double>::quiet_NaN(), nullptr,
          &projection_anchor_context, seed.source);
      if (skip_projection_candidate_clash)
        coarse_projection_skip_evals.fetch_add(1, memory_order_relaxed);
    }
  }
  timing.projection_clash_skip_evals +=
      coarse_projection_skip_evals.load(memory_order_relaxed);
  worker_lattice_queries =
      coarse_worker_lattice_queries.load(memory_order_relaxed);
  worker_lattice_failures =
      coarse_worker_lattice_failures.load(memory_order_relaxed);
  worker_lattice_ambiguities =
      coarse_worker_lattice_ambiguities.load(memory_order_relaxed);
  worker_lattice_budget_failures =
      coarse_worker_lattice_budget_failures.load(memory_order_relaxed);
  timing.pbctopo_lattice_worker_queries += worker_lattice_queries;
  timing.pbctopo_lattice_worker_query_failures += worker_lattice_failures;
  timing.pbctopo_lattice_worker_query_ambiguities +=
      worker_lattice_ambiguities;
  timing.pbctopo_lattice_worker_query_budget_failures +=
      worker_lattice_budget_failures;
  timing.pbctopo_lattice_query_failures += worker_lattice_failures;
  timing.pbctopo_lattice_optional_query_failures += worker_lattice_failures;
  timing.pbctopo_lattice_query_ambiguities += worker_lattice_ambiguities;
  timing.pbctopo_lattice_query_budget_failures +=
      worker_lattice_budget_failures;
  for (const auto &candidate : coarse_results) {
    if (better_preclash_cut_candidate(
            candidate, best_coarse_virtual_geometry_seed)) {
      best_coarse_virtual_geometry_seed = candidate;
    }
    push_top_virtual_box_candidate(coarse_candidates, candidate,
                                   kProjectionKeepCandidates);
  }
  timing.coarse_s += chrono::duration<double>(clock::now() - stage_started).count();

  stage_started = clock::now();
  verify_points.reserve(base_atoms.size());
  if (best_coarse_virtual_geometry_seed.valid) {
    ++timing.pbctopo_verify_geometry_buffer_reuses;
    ++timing.pbctopo_virtual_geometry_reference_evals;
    PbctopoCutCandidate geometry_reference = evaluate_virtual_box_candidate(
        base_atoms, local_edges, steric_exclusion_set, translation_cache,
        self.m_mol[frame_idx].Box,
        best_coarse_virtual_geometry_seed.cut_x,
        best_coarse_virtual_geometry_seed.cut_y,
        best_coarse_virtual_geometry_seed.cut_z,
        best_coarse_virtual_geometry_seed.gap_score,
        verify_points, runtime_cutoffs.clash_cutoff,
        &trajectory_context.continuity_pairs,
        &trajectory_context.continuity_anchor_indices,
        &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
        nullptr, false, self.m_mol[frame_idx].time_of_snap,
        "virtual_geometry_reference", &base_anchor_context,
        best_coarse_virtual_geometry_seed.virtual_seed_source,
        &atom_component_index);
    if (geometry_reference.valid) {
      if (geometry_reference.ultra_clash) {
        ++timing.pbctopo_virtual_geometry_reference_clashing;
      } else {
        best_virtual_geometry_reference = geometry_reference;
        ++timing.pbctopo_virtual_geometry_reference_safe;
        ++safe_candidates;
        record_temporal_certified_candidate(geometry_reference,
                                            verify_points);
        record_report_soft_candidate(geometry_reference);
        record_output_candidate(geometry_reference, verify_points,
                                PbctopoFrameMode::Virtual);
        best_verified_virtual_candidate = geometry_reference;
        if (better_cut_candidate(geometry_reference, best_safe_candidate)) {
          best_safe_candidate = geometry_reference;
          best_safe_points = verify_points;
          frame_report.mode = PbctopoFrameMode::Virtual;
        }
        if (better_cut_candidate(geometry_reference, best_candidate)) {
          best_candidate = geometry_reference;
        }
      }
    }
  }
  refine_limit = coarse_candidates.size();
  if (coarse_candidates.size() >= 2) {
    const auto &top1 = coarse_candidates[0];
    const auto &top2 = coarse_candidates[1];
    if (top1.broken_edges + 1 < top2.broken_edges ||
        top1.edge_stretch + 1.0e-6 < top2.edge_stretch * 0.85) {
      refine_limit = 1;
    } else if (coarse_candidates.size() >= 3) {
      const auto &top3 = coarse_candidates[2];
      if (top2.broken_edges + 1 < top3.broken_edges ||
          top2.edge_stretch + 1.0e-6 < top3.edge_stretch * 0.85) {
        refine_limit = 2;
      }
    }
  }
  for (size_t coarse_idx = 0; coarse_idx < refine_limit; ++coarse_idx) {
    const auto &coarse = coarse_candidates[coarse_idx];
    const PbctopoCutCandidate *verify_incumbent =
        (best_safe_candidate.valid && !best_safe_candidate.ultra_clash)
            ? &best_safe_candidate
            : nullptr;
    PbctopoCutCandidate verified_candidate = evaluate_virtual_box_candidate(
        base_atoms, local_edges, steric_exclusion_set, translation_cache,
        self.m_mol[frame_idx].Box,
        coarse.cut_x, coarse.cut_y, coarse.cut_z, coarse.gap_score,
        verify_points,
        runtime_cutoffs.clash_cutoff, &trajectory_context.continuity_pairs,
        &trajectory_context.continuity_anchor_indices,
        &local_hard_required_owner_pairs, runtime_cutoffs.continuity_cutoff,
        verify_incumbent, false, self.m_mol[frame_idx].time_of_snap,
        "verified_candidate", &base_anchor_context,
        coarse.virtual_seed_source, &atom_component_index);
    ++timing.refine_candidates;
    if (verified_candidate.valid)
      ++valid_candidates;
    if (!verified_candidate.valid)
      continue;
    const bool virtual_geometry_outlier = component_candidate_is_outlier(
        verified_candidate,
        best_virtual_geometry_reference.valid
            ? &best_virtual_geometry_reference
            : nullptr);
    if (!verified_candidate.ultra_clash) {
      if (virtual_geometry_outlier) {
        ++timing.pbctopo_virtual_geometry_outlier_observations;
        append_pbctopo_log_line(
            "[pbctopo] observe virtual geometry deviation at frame " +
            to_string(static_cast<long long>(
                self.m_mol[frame_idx].time_of_snap)) +
            ": principal_rg2_max=" +
            to_string(verified_candidate.principal_rg2_max) +
            ", reference_principal_rg2_max=" +
            to_string(best_virtual_geometry_reference.principal_rg2_max) +
            ", kappa2=" + to_string(verified_candidate.shape_kappa2) +
            ", reference_kappa2=" +
            to_string(best_virtual_geometry_reference.shape_kappa2) +
            ", policy=soft_observation");
      }
      if (better_preclash_cut_candidate(
              verified_candidate, best_virtual_geometry_reference)) {
        best_virtual_geometry_reference = verified_candidate;
      }
      ++safe_candidates;
      record_temporal_certified_candidate(verified_candidate,
                                          verify_points);
      record_report_soft_candidate(verified_candidate);
      record_output_candidate(verified_candidate, verify_points,
                              PbctopoFrameMode::Virtual);
      if (better_cut_candidate(verified_candidate,
                               best_verified_virtual_candidate)) {
        best_verified_virtual_candidate = verified_candidate;
      }
      if (better_cut_candidate(verified_candidate, best_safe_candidate)) {
        best_safe_candidate = verified_candidate;
        best_safe_points = verify_points;
        frame_report.mode = PbctopoFrameMode::Virtual;
      }
    }
    if (better_cut_candidate(verified_candidate, best_candidate)) {
      best_candidate = verified_candidate;
    }
  }
  timing.refine_s += chrono::duration<double>(clock::now() - stage_started).count();
  if (best_verified_virtual_candidate.valid &&
      !best_verified_virtual_candidate.ultra_clash &&
      best_verified_virtual_candidate.virtual_seed_source ==
          PbctopoVirtualSeedSource::Kinematic) {
    ++timing.pbctopo_kinematic_seed_best;
  }
}

} // namespace pbctopo_solve

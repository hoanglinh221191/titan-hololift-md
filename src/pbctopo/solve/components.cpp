#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

PbctopoRefineScoringCallbacks make_component_refine_callbacks() {
  PbctopoRefineScoringCallbacks callbacks;
  callbacks.score_cut_candidate = score_cut_candidate;
  callbacks.score_cut_candidate_principal_only =
      score_cut_candidate_principal_only;
  callbacks.continuity_metrics_ok = continuity_metrics_ok;
  callbacks.analyze_cut_candidate_ultraclash =
      analyze_cut_candidate_ultraclash;
  callbacks.better_cut_candidate = better_cut_candidate;
  callbacks.better_preclash_cut_candidate = better_preclash_cut_candidate;
  callbacks.better_stage1_cut_candidate = better_stage1_cut_candidate;
  callbacks.cut_candidate_cannot_beat_incumbent =
      cut_candidate_cannot_beat_incumbent;
  callbacks.edge_scan_cannot_beat_threshold = edge_scan_cannot_beat_threshold;
  callbacks.candidate_close_enough_to_skip_search =
      candidate_close_enough_to_skip_search;
  return callbacks;
}

const char *pbctopo_component_mode_name(PbctopoComponentMode mode) {
  switch (mode) {
  case PbctopoComponentMode::Disabled:
    return "disabled";
  case PbctopoComponentMode::Exact:
    return "small_graph";
  case PbctopoComponentMode::MergeThenGraph:
    return "merge_then_graph";
  case PbctopoComponentMode::Beam:
    return "beam";
  }
  return "unknown";
}

PbctopoComponentMode select_pbctopo_component_mode(size_t raw_component_count,
                                                   size_t exact_component_limit) {
  if (raw_component_count == 0)
    return PbctopoComponentMode::Disabled;
  const size_t bounded_exact_limit =
      min(exact_component_limit, kPbctopoExactComponentLimitMax);
  if (raw_component_count <= bounded_exact_limit)
    return PbctopoComponentMode::Exact;
  if (raw_component_count <= kPbctopoMergeModeMaxRawComponents)
    return PbctopoComponentMode::MergeThenGraph;
  return PbctopoComponentMode::Beam;
}

PbctopoMergedComponentPlan build_chain_merged_component_plan(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &raw_components) {
  PbctopoMergedComponentPlan plan;
  plan.raw_to_merged.assign(raw_components.size(), numeric_limits<size_t>::max());
  plan.raw_internal_offsets.assign(raw_components.size(), {});
  if (raw_components.empty())
    return plan;

  vector<PbctopoRawComponentMetadata> metadata;
  metadata.reserve(raw_components.size());
  for (size_t raw_idx = 0; raw_idx < raw_components.size(); ++raw_idx) {
    const auto &component = raw_components[raw_idx];
    if (component.empty())
      continue;
    PbctopoRawComponentMetadata meta;
    meta.raw_index = raw_idx;
    meta.atom_count = component.size();
    const auto &first_atom = atoms[component.front()];
    meta.owner = first_atom.owner;
    meta.chain_key = first_atom.chain_key;
    for (size_t atom_idx : component) {
      if (atom_idx >= atoms.size())
        continue;
      meta.min_local_atom_index =
          min(meta.min_local_atom_index, atoms[atom_idx].local_atom_index);
      meta.max_local_atom_index =
          max(meta.max_local_atom_index, atoms[atom_idx].local_atom_index);
      meta.min_atom_index = min(meta.min_atom_index, atom_idx);
      if (meta.first_atom_index == numeric_limits<size_t>::max() ||
          atoms[atom_idx].local_atom_index <
              atoms[meta.first_atom_index].local_atom_index) {
        meta.first_atom_index = atom_idx;
      }
      if (meta.last_atom_index == numeric_limits<size_t>::max() ||
          atoms[atom_idx].local_atom_index >
              atoms[meta.last_atom_index].local_atom_index) {
        meta.last_atom_index = atom_idx;
      }
    }
    metadata.push_back(meta);
  }

  stable_sort(metadata.begin(), metadata.end(),
              [](const PbctopoRawComponentMetadata &lhs,
                 const PbctopoRawComponentMetadata &rhs) {
                if (lhs.owner != rhs.owner)
                  return lhs.owner < rhs.owner;
                if (lhs.chain_key != rhs.chain_key)
                  return lhs.chain_key < rhs.chain_key;
                if (lhs.min_local_atom_index != rhs.min_local_atom_index)
                  return lhs.min_local_atom_index < rhs.min_local_atom_index;
                return lhs.raw_index < rhs.raw_index;
              });

  vector<size_t> current_raw_group;
  vector<size_t> current_atoms;
  vector<size_t> first_atom_by_raw(raw_components.size(),
                                   numeric_limits<size_t>::max());
  vector<size_t> last_atom_by_raw(raw_components.size(),
                                  numeric_limits<size_t>::max());
  for (const auto &meta : metadata) {
    first_atom_by_raw[meta.raw_index] = meta.first_atom_index;
    last_atom_by_raw[meta.raw_index] = meta.last_atom_index;
  }
  const PbctopoLatticeMetric lattice(Box);
  current_raw_group.reserve(kPbctopoMergedNodeMaxRawComponents);
  current_atoms.reserve(kPbctopoMergedNodeMaxAtoms);
  int current_owner = 0;
  uint64_t current_chain_key = 0;
  size_t current_last_local_atom = 0;
  bool has_current_group = false;

  auto flush_group = [&]() {
    if (current_raw_group.empty())
      return;
    Int3 previous_offset{};
    for (size_t pos = 1; pos < current_raw_group.size(); ++pos) {
      const size_t previous_raw = current_raw_group[pos - 1];
      const size_t current_raw = current_raw_group[pos];
      const size_t previous_atom = last_atom_by_raw[previous_raw];
      const size_t current_atom = first_atom_by_raw[current_raw];
      if (previous_atom >= atoms.size() || current_atom >= atoms.size())
        continue;
      const auto mic = lattice.nearest_image_scaled(
          atoms[current_atom].sx -
              (atoms[previous_atom].sx + previous_offset.x),
          atoms[current_atom].sy -
              (atoms[previous_atom].sy + previous_offset.y),
          atoms[current_atom].sz -
              (atoms[previous_atom].sz + previous_offset.z));
      if (!mic.valid)
        continue;
      previous_offset = {mic.image_x, mic.image_y, mic.image_z};
      plan.raw_internal_offsets[current_raw] = previous_offset;
    }
    const size_t merged_idx = plan.merged_components.size();
    sort(current_atoms.begin(), current_atoms.end());
    plan.merged_components.push_back(current_atoms);
    plan.merged_to_raw.push_back(current_raw_group);
    for (size_t raw_idx : current_raw_group) {
      if (raw_idx < plan.raw_to_merged.size())
        plan.raw_to_merged[raw_idx] = merged_idx;
    }
    current_raw_group.clear();
    current_atoms.clear();
    has_current_group = false;
  };

  for (const auto &meta : metadata) {
    const bool contiguous_with_current =
        has_current_group && meta.owner == current_owner &&
        meta.chain_key == current_chain_key &&
        meta.min_local_atom_index == current_last_local_atom + 1;
    const bool fits_current =
        contiguous_with_current &&
        current_raw_group.size() < kPbctopoMergedNodeMaxRawComponents &&
        current_atoms.size() + meta.atom_count <= kPbctopoMergedNodeMaxAtoms;
    if (!fits_current)
      flush_group();
    if (!has_current_group) {
      current_owner = meta.owner;
      current_chain_key = meta.chain_key;
      current_last_local_atom = meta.max_local_atom_index;
      has_current_group = true;
    } else {
      current_last_local_atom = meta.max_local_atom_index;
    }
    current_raw_group.push_back(meta.raw_index);
    current_atoms.insert(current_atoms.end(), raw_components[meta.raw_index].begin(),
                         raw_components[meta.raw_index].end());
  }
  flush_group();

  return plan;
}

vector<Int3> expand_merged_offsets_to_raw_components(
    const vector<Int3> &merged_offsets, box Box,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan) {
  vector<Int3> raw_offsets(raw_components.size(), {0, 0, 0});
  for (size_t merged_idx = 0; merged_idx < plan.merged_to_raw.size(); ++merged_idx) {
    const Int3 merged_offset =
        merged_idx < merged_offsets.size() ? merged_offsets[merged_idx] : Int3{};
    for (size_t raw_idx : plan.merged_to_raw[merged_idx]) {
      if (raw_idx < raw_offsets.size()) {
        const Int3 internal = raw_idx < plan.raw_internal_offsets.size()
                                  ? plan.raw_internal_offsets[raw_idx]
                                  : Int3{};
        if (!checked_int3_add(merged_offset, internal,
                              raw_offsets[raw_idx])) {
          return {};
        }
      }
    }
  }
  if (!rebase_component_offsets(raw_offsets, Box))
    return {};
  return raw_offsets;
}

bool derive_merged_warm_offsets(const vector<Int3> &raw_offsets,
                                const PbctopoMergedComponentPlan &plan,
                                vector<Int3> &merged_offsets) {
  if (raw_offsets.empty() || plan.merged_to_raw.empty())
    return false;
  merged_offsets.clear();
  merged_offsets.reserve(plan.merged_to_raw.size());
  for (const auto &group : plan.merged_to_raw) {
    if (group.empty())
      return false;
    const size_t first_raw_idx = group.front();
    if (first_raw_idx >= raw_offsets.size())
      return false;
    const Int3 first_internal =
        first_raw_idx < plan.raw_internal_offsets.size()
            ? plan.raw_internal_offsets[first_raw_idx]
            : Int3{};
    Int3 candidate{};
    if (!checked_int3_subtract(raw_offsets[first_raw_idx], first_internal,
                               candidate)) {
      return false;
    }
    for (size_t raw_idx : group) {
      if (raw_idx >= raw_offsets.size())
        return false;
      const Int3 internal = raw_idx < plan.raw_internal_offsets.size()
                                ? plan.raw_internal_offsets[raw_idx]
                                : Int3{};
      Int3 residual{};
      if (!checked_int3_subtract(raw_offsets[raw_idx], internal, residual))
        return false;
      if (!same_delta(residual, candidate)) {
        return false;
      }
    }
    merged_offsets.push_back(candidate);
  }
  return true;
}

vector<PbctopoAtom> apply_merged_component_internal_offsets(
    const vector<PbctopoAtom> &atoms,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &plan, box Box) {
  vector<PbctopoAtom> aligned = atoms;
  const PbctopoLatticeMetric lattice(Box);
  const size_t raw_count =
      min(raw_components.size(), plan.raw_internal_offsets.size());
  for (size_t raw_idx = 0; raw_idx < raw_count; ++raw_idx) {
    const Int3 offset = plan.raw_internal_offsets[raw_idx];
    for (size_t atom_idx : raw_components[raw_idx]) {
      if (atom_idx >= aligned.size())
        continue;
      auto &atom = aligned[atom_idx];
      atom.sx += offset.x;
      atom.sy += offset.y;
      atom.sz += offset.z;
      const auto cart = lattice.scaled_to_cartesian(atom.sx, atom.sy, atom.sz);
      atom.x = cart[0];
      atom.y = cart[1];
      atom.z = cart[2];
    }
  }
  return aligned;
}

bool graph_seed_quick_score_better(const PbctopoGraphSeedQuickScore &lhs,
                                   const PbctopoGraphSeedQuickScore &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  int metric_cmp = compare_pbctopo_double_metric(
      lhs.continuity_path, rhs.continuity_path, 1.0e-6);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  int anchor_cmp = compare_anchor_graph_score(lhs.anchor_graph,
                                              rhs.anchor_graph);
  if (anchor_cmp == 0) {
    anchor_cmp = compare_pbctopo_double_metric(lhs.anchor_max_d2,
                                               rhs.anchor_max_d2, 1.0e-6);
  }
  if (anchor_cmp == 0) {
    anchor_cmp = compare_pbctopo_double_metric(lhs.anchor_mst2,
                                               rhs.anchor_mst2, 1.0e-6);
  }
  if (anchor_cmp != 0)
    return anchor_cmp < 0;
  const int principal_max_cmp =
      compare_principal_rg2(lhs.principal_rg2_max, rhs.principal_rg2_max);
  if (principal_max_cmp != 0)
    return principal_max_cmp < 0;
  const int principal_mid_cmp =
      compare_principal_rg2(lhs.principal_rg2_mid, rhs.principal_rg2_mid);
  if (principal_mid_cmp != 0)
    return principal_mid_cmp < 0;
  const int principal_min_cmp =
      compare_principal_rg2(lhs.principal_rg2_min, rhs.principal_rg2_min);
  if (principal_min_cmp != 0)
    return principal_min_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2,
                                              1.0e-6);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return false;
}

PbctopoGraphSeedQuickScore score_component_graph_seed_quick(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points, const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2,
    PbctopoAnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context) {
  PbctopoGraphSeedQuickScore score;
  if (placed_points.size() != atoms.size())
    return score;
  double continuity_min_margin_A = numeric_limits<double>::quiet_NaN();
  size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
  size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
  if (!continuity_metrics_ok(placed_points, continuity_pairs, continuity_cutoff,
                             score.continuity_path, score.continuity_max_d2,
                             continuity_min_margin_A,
                             continuity_margin_atom_a,
                             continuity_margin_atom_b, nullptr)) {
    return score;
  }
  const vector<size_t> half_anchor_indices =
      build_coarse_anchor_subset(atoms, anchor_indices);
  const vector<size_t> third_anchor_indices =
      build_third_anchor_subset(atoms, anchor_indices);
  const vector<size_t> *quick_anchor_indices = &anchor_indices;
  if (anchor_indices.size() > 256 && third_anchor_indices.size() >= 2) {
    quick_anchor_indices = &third_anchor_indices;
  } else if (half_anchor_indices.size() >= 2) {
    quick_anchor_indices = &half_anchor_indices;
  }
  if (!anchor_graph_metrics_enhanced(
          placed_points, *quick_anchor_indices, continuity_pairs,
          score.anchor_mst2, score.anchor_max_d2, score.anchor_graph, nullptr,
          PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
          anchor_context)) {
    return score;
  }
  const PointStats stats = build_point_stats(placed_points);
  if (stats.count == 0)
    return score;
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  score.principal_rg2_max = principal_rg2[0];
  score.principal_rg2_mid = principal_rg2[1];
  score.principal_rg2_min = principal_rg2[2];
  score.shift_norm2 = shift_norm2;
  score.valid = true;
  return score;
}

bool kinematic_virtual_candidate_usable(
    const PbctopoCutCandidate &candidate) {
  return candidate.valid && !candidate.ultra_clash &&
         isfinite(candidate.cut_x) && isfinite(candidate.cut_y) &&
         isfinite(candidate.cut_z);
}

bool has_consecutive_kinematic_seed_history(const PbctopoWarmState &warm_state,
                                            size_t frame_idx) {
  return warm_state.has_kinematic_previous_virtual_candidate &&
         warm_state.has_kinematic_latest_virtual_candidate &&
         warm_state.kinematic_previous_virtual_frame + 1 ==
             warm_state.kinematic_latest_virtual_frame &&
         warm_state.kinematic_latest_virtual_frame + 1 == frame_idx &&
         kinematic_virtual_candidate_usable(
             warm_state.kinematic_previous_virtual_candidate) &&
         kinematic_virtual_candidate_usable(
             warm_state.kinematic_latest_virtual_candidate);
}

void reset_kinematic_seed_history(PbctopoWarmState &warm_state,
                                  PbctopoTimingStats &timing) {
  if (warm_state.has_kinematic_previous_virtual_candidate ||
      warm_state.has_kinematic_latest_virtual_candidate) {
    ++timing.pbctopo_kinematic_seed_history_resets;
  }
  warm_state.kinematic_previous_virtual_candidate = PbctopoCutCandidate{};
  warm_state.kinematic_latest_virtual_candidate = PbctopoCutCandidate{};
  warm_state.kinematic_previous_virtual_frame =
      numeric_limits<size_t>::max();
  warm_state.kinematic_latest_virtual_frame =
      numeric_limits<size_t>::max();
  warm_state.has_kinematic_previous_virtual_candidate = false;
  warm_state.has_kinematic_latest_virtual_candidate = false;
}

void update_kinematic_seed_history(PbctopoWarmState &warm_state,
                                   const PbctopoCutCandidate &candidate,
                                   size_t frame_idx,
                                   PbctopoTimingStats &timing) {
  if (!kinematic_virtual_candidate_usable(candidate)) {
    reset_kinematic_seed_history(warm_state, timing);
    return;
  }

  if (warm_state.has_kinematic_latest_virtual_candidate &&
      warm_state.kinematic_latest_virtual_frame + 1 == frame_idx) {
    warm_state.kinematic_previous_virtual_candidate =
        warm_state.kinematic_latest_virtual_candidate;
    warm_state.kinematic_previous_virtual_frame =
        warm_state.kinematic_latest_virtual_frame;
    warm_state.has_kinematic_previous_virtual_candidate = true;
  } else if (warm_state.has_kinematic_latest_virtual_candidate ||
             warm_state.has_kinematic_previous_virtual_candidate) {
    reset_kinematic_seed_history(warm_state, timing);
  }

  warm_state.kinematic_latest_virtual_candidate = candidate;
  warm_state.kinematic_latest_virtual_frame = frame_idx;
  warm_state.has_kinematic_latest_virtual_candidate = true;
}

vector<PbctopoGraphSeedSolveResult> build_large_component_graph_seed_candidates(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &components, const vector<PbctopoShift> &shifts,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, const PbctopoRuntimeCutoffs &cutoffs,
    const vector<Int3> *warm_component_offsets, PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context,
    const vector<titan_pbctopo::PbctopoRequiredContact> *required_contacts) {
  const vector<PbctopoGraphEdge> base_edges = build_component_graph_edge_candidates(
      atoms, Box, components, shifts, cutoffs, timing, warm_component_offsets,
      nullptr, required_contacts);
  vector<PbctopoGraphSeedSolveResult> results;
  results.reserve(3);
  unordered_set<string> seen_offset_signatures;

  auto try_seed = [&](const char *seed_name,
                      const function<void(vector<PbctopoGraphEdge> &)> &sort_fn) {
    vector<PbctopoGraphEdge> ranked_edges = base_edges;
    sort_fn(ranked_edges);
    vector<Int3> offsets;
    if (!solve_component_offsets_from_ranked_edges(components.size(), Box, components,
                                                   ranked_edges, offsets)) {
      return;
    }
    const string signature = format_component_offsets(offsets);
    if (!seen_offset_signatures.insert(signature).second)
      return;
    vector<Vec3> seed_points;
    double seed_shift_norm2 = 0.0;
    build_component_points_from_offsets(atoms, components, Box, offsets, seed_points,
                                        seed_shift_norm2);
    PbctopoGraphSeedSolveResult result;
    result.seed_name = seed_name;
    result.offsets = offsets;
    result.quick_score = score_component_graph_seed_quick(
        atoms, components, seed_points, continuity_pairs, anchor_indices,
        cutoffs.continuity_cutoff, seed_shift_norm2, anchor_backend,
        anchor_context);
    result.valid = result.quick_score.valid;
    results.push_back(std::move(result));
  };

  try_seed("baseline", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    sort(ranked_edges.begin(), ranked_edges.end(), better_component_graph_edge);
  });

  try_seed("nonzero", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    stable_sort(ranked_edges.begin(), ranked_edges.end(),
                [](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                  if (!lhs.valid)
                    return false;
                  if (!rhs.valid)
                    return true;
                  if (lhs.ultra_clash != rhs.ultra_clash)
                    return rhs.ultra_clash;
                  const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                  const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                  if (lhs_zero != rhs_zero)
                    return !lhs_zero;
                  int metric_cmp = compare_pbctopo_double_metric(
                      lhs.contact_fraction, rhs.contact_fraction, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.confidence, rhs.confidence, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.soft_penalty_mean, rhs.soft_penalty_mean, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.min_pair_d2, rhs.min_pair_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.bbox_d2, rhs.bbox_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.shift_norm2, rhs.shift_norm2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  if (lhs.a != rhs.a)
                    return lhs.a < rhs.a;
                  return lhs.b < rhs.b;
                });
  });

  try_seed("warm", [&](vector<PbctopoGraphEdge> &ranked_edges) {
    if (warm_component_offsets == nullptr ||
        warm_component_offsets->size() != components.size()) {
      stable_sort(ranked_edges.begin(), ranked_edges.end(),
                  [](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                    if (!lhs.valid)
                      return false;
                    if (!rhs.valid)
                      return true;
                    if (lhs.ultra_clash != rhs.ultra_clash)
                      return rhs.ultra_clash;
                    const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                    const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                    if (lhs_zero != rhs_zero)
                      return !lhs_zero;
                    return better_component_graph_edge(lhs, rhs);
                  });
      return;
    }
    stable_sort(ranked_edges.begin(), ranked_edges.end(),
                [&](const PbctopoGraphEdge &lhs, const PbctopoGraphEdge &rhs) {
                  if (!lhs.valid)
                    return false;
                  if (!rhs.valid)
                    return true;
                  if (lhs.ultra_clash != rhs.ultra_clash)
                    return rhs.ultra_clash;
                  Int3 lhs_expected{};
                  Int3 rhs_expected{};
                  const bool lhs_expected_valid = checked_int3_subtract(
                      (*warm_component_offsets)[lhs.b],
                      (*warm_component_offsets)[lhs.a], lhs_expected);
                  const bool rhs_expected_valid = checked_int3_subtract(
                      (*warm_component_offsets)[rhs.b],
                      (*warm_component_offsets)[rhs.a], rhs_expected);
                  const bool lhs_warm =
                      lhs_expected_valid &&
                      graph_edge_matches_delta(lhs, lhs_expected);
                  const bool rhs_warm =
                      rhs_expected_valid &&
                      graph_edge_matches_delta(rhs, rhs_expected);
                  if (lhs_warm != rhs_warm)
                    return lhs_warm;
                  const bool lhs_zero = graph_edge_has_zero_shift(lhs);
                  const bool rhs_zero = graph_edge_has_zero_shift(rhs);
                  if (lhs_zero != rhs_zero)
                    return !lhs_zero;
                  int metric_cmp = compare_pbctopo_double_metric(
                      lhs.contact_fraction, rhs.contact_fraction, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.confidence, rhs.confidence, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.soft_penalty_mean, rhs.soft_penalty_mean, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.min_pair_d2, rhs.min_pair_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp > 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.bbox_d2, rhs.bbox_d2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  metric_cmp = compare_pbctopo_double_metric(
                      lhs.shift_norm2, rhs.shift_norm2, 1.0e-9);
                  if (metric_cmp != 0)
                    return metric_cmp < 0;
                  if (lhs.a != rhs.a)
                    return lhs.a < rhs.a;
                  return lhs.b < rhs.b;
                });
  });

  return results;
}

vector<PbctopoAtom> apply_spatial_lift(
    const vector<PbctopoAtom> &wrapped_atoms,
    const titan_pbctopo::PbctopoSpatialLiftResult &lift, box Box) {
  vector<PbctopoAtom> unwrapped = wrapped_atoms;
  if (!lift.valid || lift.atom_images.size() != wrapped_atoms.size())
    return unwrapped;
  for (size_t atom_idx = 0; atom_idx < unwrapped.size(); ++atom_idx) {
    auto &atom = unwrapped[atom_idx];
    const auto &image = lift.atom_images[atom_idx];
    atom.sx = wrapped_atoms[atom_idx].sx + static_cast<double>(image.x);
    atom.sy = wrapped_atoms[atom_idx].sy + static_cast<double>(image.y);
    atom.sz = wrapped_atoms[atom_idx].sz + static_cast<double>(image.z);
    scaled_to_cartesian(Box, atom.sx, atom.sy, atom.sz, atom.x, atom.y, atom.z);
  }
  return unwrapped;
}

PbctopoComponentCandidate build_component_candidate_seed(
    const PbctopoCutCandidate &source, const vector<vector<size_t>> &components,
    const vector<Vec3> &placed_points) {
  PbctopoComponentCandidate seed;
  if (!source.valid || placed_points.empty())
    return seed;
  seed.continuity_path = source.continuity_path;
  seed.continuity_max_d2 = source.continuity_max_d2;
  seed.continuity_min_margin_A = source.continuity_min_margin_A;
  seed.continuity_margin_atom_a = source.continuity_margin_atom_a;
  seed.continuity_margin_atom_b = source.continuity_margin_atom_b;
  seed.anchor_mst2 = source.anchor_mst2;
  seed.anchor_max_d2 = source.anchor_max_d2;
  seed.anchor_graph = source.anchor_graph;
  seed.principal_rg2_max = source.principal_rg2_max;
  seed.principal_rg2_mid = source.principal_rg2_mid;
  seed.principal_rg2_min = source.principal_rg2_min;
  seed.bbox_volume = source.bbox_volume;
  seed.rg2 = source.rg2;
  seed.shape_kappa2 = source.shape_kappa2;
  seed.shape_asphericity = source.shape_asphericity;
  seed.shape_acylindricity = source.shape_acylindricity;
  seed.shift_norm2 = source.shift_norm2;
  seed.min_pair_d2 = source.min_pair_d2;
  seed.clash_atom_a = source.clash_atom_a;
  seed.clash_atom_b = source.clash_atom_b;
  seed.ultra_clash = source.ultra_clash;
  vector<Vec3> component_centers;
  component_centers.reserve(components.size());
  for (const auto &component : components) {
    if (component.empty())
      continue;
    Vec3 center{};
    for (size_t atom_idx : component)
      center = center + placed_points[atom_idx];
    component_centers.push_back(
        center * (1.0 / static_cast<double>(component.size())));
  }
  seed.component_rg2 = compute_rg2(component_centers);
  const PointStats component_stats = build_point_stats(component_centers);
  const auto component_principal_rg2 = compute_principal_rg2(component_stats);
  const auto component_shape_metrics =
      compute_gyration_shape_metrics(component_principal_rg2);
  seed.component_kappa2 = component_shape_metrics.kappa2;
  seed.component_asphericity = component_shape_metrics.asphericity;
  seed.component_acylindricity = component_shape_metrics.acylindricity;
  seed.component_mst2 = compute_mst2(component_centers);
  seed.valid = true;
  return seed;
}

PbctopoComponentCandidate score_component_assignment(
    const vector<PbctopoAtom> &atoms, const vector<Vec3> &placed_points,
    const vector<vector<size_t>> &components, double min_pair_d2,
    size_t clash_atom_a, size_t clash_atom_b,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    double shift_norm2, const PbctopoComponentCandidate *incumbent,
    PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context) {
  using clock = chrono::steady_clock;
  const auto started = clock::now();
  auto finish = [&](PbctopoComponentCandidate candidate) {
    if (timing != nullptr) {
      ++timing->component_score_calls;
      timing->component_score_s +=
          chrono::duration<double>(clock::now() - started).count();
    }
    return candidate;
  };
  PbctopoComponentCandidate candidate;
  const bool debug_small_search = (components.size() <= 2);
  if (atoms.empty() || placed_points.size() != atoms.size())
    return finish(candidate);
  candidate.shift_norm2 = shift_norm2;
  candidate.min_pair_d2 = min_pair_d2;
  candidate.clash_atom_a = clash_atom_a;
  candidate.clash_atom_b = clash_atom_b;
  PbctopoCutCandidate incumbent_cut_storage;
  const PbctopoCutCandidate *incumbent_cut = nullptr;
  if (incumbent != nullptr && incumbent->valid && !incumbent->ultra_clash) {
    incumbent_cut_storage = component_candidate_to_cut_candidate(*incumbent);
    incumbent_cut = &incumbent_cut_storage;
  }

  auto stage_started = clock::now();
  if (!continuity_metrics_ok(placed_points, continuity_pairs, continuity_cutoff,
                             candidate.continuity_path,
                             candidate.continuity_max_d2,
                             candidate.continuity_min_margin_A,
                             candidate.continuity_margin_atom_a,
                             candidate.continuity_margin_atom_b,
                             incumbent_cut)) {
    if (timing != nullptr) {
      ++timing->component_score_continuity_rejects;
      timing->component_score_continuity_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
    }
    if (debug_small_search) {
      append_pbctopo_log_line(
          "[pbctopo-debug] component leaf rejected by continuity: max_d2=" +
          to_string(candidate.continuity_max_d2) +
          ", path=" + to_string(candidate.continuity_path));
    }
    return finish(candidate);
  }
  if (timing != nullptr) {
    timing->component_score_continuity_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Continuity)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_continuity;
    return finish(candidate);
  }

  stage_started = clock::now();
  if (!anchor_graph_metrics_enhanced(
          placed_points, anchor_indices, continuity_pairs, candidate.anchor_mst2,
          candidate.anchor_max_d2, candidate.anchor_graph, incumbent_cut,
          PbctopoRefineScorePhase::Exact, anchor_backend, nullptr,
          anchor_context)) {
    if (timing != nullptr) {
      ++timing->component_score_anchor_rejects;
      timing->component_score_anchor_s +=
          chrono::duration<double>(clock::now() - stage_started).count();
    }
    if (debug_small_search) {
      append_pbctopo_log_line(
          "[pbctopo-debug] component leaf rejected by anchor-graph: max_d2=" +
          to_string(candidate.anchor_max_d2) +
          ", mst2=" + to_string(candidate.anchor_mst2));
    }
    return finish(candidate);
  }
  if (timing != nullptr) {
    timing->component_score_anchor_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Anchor)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_anchor;
    return finish(candidate);
  }

  stage_started = clock::now();
  const PointStats stats = build_point_stats(placed_points);
  if (stats.count == 0)
    return finish(candidate);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.rg2 = compute_rg2_from_stats(stats);
  candidate.bbox_volume = compute_bbox_volume(stats);
  if (timing != nullptr) {
    timing->component_score_stats_principal_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Principal)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_principal;
    return finish(candidate);
  }

  stage_started = clock::now();
  vector<Vec3> component_centers;
  component_centers.reserve(components.size());
  for (const auto &component : components) {
    if (component.empty())
      continue;
    Vec3 center{};
    for (size_t atom_idx : component)
      center = center + placed_points[atom_idx];
    component_centers.push_back(
        center * (1.0 / static_cast<double>(component.size())));
  }
  candidate.component_rg2 = compute_rg2(component_centers);
  const PointStats component_stats = build_point_stats(component_centers);
  const auto component_principal_rg2 = compute_principal_rg2(component_stats);
  const auto component_shape_metrics =
      compute_gyration_shape_metrics(component_principal_rg2);
  candidate.component_kappa2 = component_shape_metrics.kappa2;
  candidate.component_asphericity = component_shape_metrics.asphericity;
  candidate.component_acylindricity = component_shape_metrics.acylindricity;
  candidate.component_mst2 = compute_mst2(component_centers);
  if (timing != nullptr) {
    timing->component_score_component_metrics_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent,
          PbctopoComponentCandidatePrefixStage::ComponentMetrics)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_component_metrics;
    return finish(candidate);
  }
  if (component_candidate_cannot_beat_incumbent_prefix(
          candidate, incumbent, PbctopoComponentCandidatePrefixStage::Final)) {
    if (timing != nullptr)
      ++timing->component_score_prefix_prunes_after_final;
    return finish(candidate);
  }
  candidate.valid = true;
  if (debug_small_search) {
    append_pbctopo_log_line(
        "[pbctopo-debug] component leaf valid: cont_max_d2=" +
        to_string(candidate.continuity_max_d2) +
        ", cont_path=" + to_string(candidate.continuity_path) +
        ", anchor_max_d2=" + to_string(candidate.anchor_max_d2) +
        ", anchor_mst2=" + to_string(candidate.anchor_mst2) +
        ", princ_max=" + to_string(candidate.principal_rg2_max) +
        ", comp_mst2=" + to_string(candidate.component_mst2) +
        ", kappa2=" + to_string(candidate.shape_kappa2) +
        ", asph=" + to_string(candidate.shape_asphericity));
  }
  return finish(candidate);
}

PbctopoComponentSearchCallbacks make_component_search_callbacks() {
  PbctopoComponentSearchCallbacks callbacks;
  callbacks.build_seed = build_component_candidate_seed;
  callbacks.score_assignment = score_component_assignment;
  callbacks.better_candidate = better_component_candidate;
  return callbacks;
}

bool solve_large_component_merge_mode(
    const vector<PbctopoAtom> &atoms, box Box,
    const vector<vector<size_t>> &raw_components,
    const PbctopoMergedComponentPlan &merge_plan,
    const vector<PbctopoShift> &shifts,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set, vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices,
    const PbctopoRuntimeCutoffs &cutoffs, PbctopoCutCandidate &best_cut,
    double &shift_norm2, PbctopoTimingStats *timing,
    PbctopoAnchorBackend anchor_backend,
    const vector<Int3> *warm_component_offsets,
    PbctopoWarmState *warm_state,
    const PbctopoCutCandidate *warm_reference_candidate,
    bool allow_exact_search_on_merged,
    vector<Int3> *best_raw_offsets_out,
    const PbctopoAnchorGraphContext *anchor_context,
    const vector<titan_pbctopo::PbctopoRequiredContact> *required_contacts) {
  best_cut = {};
  shift_norm2 = 0.0;
  placed_points.clear();
  if (merge_plan.merged_components.empty())
    return false;

  const vector<PbctopoAtom> merged_atoms =
      apply_merged_component_internal_offsets(atoms, raw_components,
                                              merge_plan, Box);
  if (timing != nullptr) {
    ++timing->component_large_mode_internal_offsets_preserved;
    timing->component_large_mode_internal_offset_nonzero +=
        count_if(merge_plan.raw_internal_offsets.begin(),
                 merge_plan.raw_internal_offsets.end(),
                 [](const Int3 &offset) { return !is_zero_delta(offset); });
  }

  vector<Int3> merged_warm_offsets;
  const vector<Int3> *merged_warm_offsets_ptr =
      (warm_component_offsets != nullptr &&
       derive_merged_warm_offsets(*warm_component_offsets, merge_plan,
                                  merged_warm_offsets))
          ? &merged_warm_offsets
          : nullptr;

  vector<Vec3> best_points_local;
  vector<Int3> best_raw_offsets;
  auto consider_raw_offsets = [&](const vector<Int3> &raw_offsets) {
    vector<Int3> refine_offsets = raw_offsets;
    vector<Vec3> refine_points;
    PbctopoCutCandidate candidate = improve_component_offsets_locally(
        atoms, raw_components, Box, shifts, local_edges, edge_set,
        continuity_pairs, anchor_indices, refine_offsets, cutoffs.clash_cutoff,
        cutoffs.continuity_cutoff, refine_points,
        make_component_refine_callbacks(), anchor_backend, timing, warm_state,
        warm_reference_candidate, anchor_context);
    if (candidate.valid && !candidate.ultra_clash &&
        refine_points.size() == atoms.size() &&
        (!best_cut.valid || better_cut_candidate(candidate, best_cut))) {
      best_cut = candidate;
      best_points_local = refine_points;
      best_raw_offsets = refine_offsets;
    }
  };

  const vector<PbctopoGraphSeedSolveResult> graph_seed_results =
      build_large_component_graph_seed_candidates(
          merged_atoms, Box, merge_plan.merged_components, shifts,
          continuity_pairs, anchor_indices, cutoffs, merged_warm_offsets_ptr, timing,
          anchor_backend, anchor_context, required_contacts);
  if (timing != nullptr)
    timing->component_large_mode_graph_seed_count += graph_seed_results.size();

  const PbctopoGraphSeedSolveResult *selected_seed = nullptr;
  for (const auto &seed_result : graph_seed_results) {
    if (!seed_result.valid)
      continue;
    if (selected_seed == nullptr ||
        graph_seed_quick_score_better(seed_result.quick_score,
                                      selected_seed->quick_score)) {
      selected_seed = &seed_result;
    }
  }
  if (selected_seed == nullptr && !graph_seed_results.empty())
    selected_seed = &graph_seed_results.front();

  if (selected_seed != nullptr &&
      selected_seed->offsets.size() == merge_plan.merged_components.size()) {
    if (timing != nullptr) {
      ++timing->component_large_mode_graph_seed_refined;
      if (string(selected_seed->seed_name) != "baseline")
        ++timing->component_large_mode_graph_seed_nonbaseline_win;
    }
    consider_raw_offsets(expand_merged_offsets_to_raw_components(
        selected_seed->offsets, Box, raw_components, merge_plan));
  }

  if (allow_exact_search_on_merged) {
    vector<Vec3> merged_search_points;
    const PbctopoCutCandidate *search_incumbent =
        (best_cut.valid && !best_cut.ultra_clash) ? &best_cut : warm_reference_candidate;
    const vector<Vec3> *search_incumbent_points =
        (best_cut.valid && !best_cut.ultra_clash &&
         best_points_local.size() == atoms.size())
            ? &best_points_local
            : nullptr;
    if (timing != nullptr)
      ++timing->component_search_frames;
    PbctopoComponentCandidate merged_candidate = search_component_shift_candidate(
        merged_atoms, Box, merge_plan.merged_components, shifts, continuity_pairs,
        anchor_indices, cutoffs, merged_search_points,
        make_component_search_callbacks(), merged_warm_offsets_ptr,
        search_incumbent, search_incumbent_points, timing, anchor_backend,
        anchor_context, nullptr, nullptr, 0, required_contacts);
    if (merged_candidate.valid && !merged_candidate.ultra_clash &&
        merged_search_points.size() == atoms.size()) {
      const vector<Int3> merged_search_offsets = extract_component_offsets_from_points(
          merged_atoms, merge_plan.merged_components, Box,
          merged_search_points);
      if (merged_search_offsets.size() == merge_plan.merged_components.size()) {
        consider_raw_offsets(expand_merged_offsets_to_raw_components(
            merged_search_offsets, Box, raw_components, merge_plan));
      }
    }
  }

  if (!best_cut.valid || best_points_local.size() != atoms.size())
    return false;
  placed_points = best_points_local;
  shift_norm2 = best_cut.shift_norm2;
  if (best_raw_offsets_out != nullptr)
    *best_raw_offsets_out = best_raw_offsets;
  return true;
}

} // namespace pbctopo_solve

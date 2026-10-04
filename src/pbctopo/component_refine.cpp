#include "component_refine.h"

#include "component_metric.h"
#include "reference_strain.h"
#include "warm_state.h"

#include "../BoxMatrix.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

using namespace std;

namespace titan_pbctopo {

using pbctopo_hull::norm2;
using titan_gyration::build_point_stats;
using titan_gyration::PointStats;

namespace {

size_t find_shift_index(const vector<PbctopoShift> &shifts, const Int3 &delta) {
  for (size_t idx = 0; idx < shifts.size(); ++idx) {
    if (shifts[idx].ix == delta.x && shifts[idx].iy == delta.y &&
        shifts[idx].iz == delta.z) {
      return idx;
    }
  }
  return numeric_limits<size_t>::max();
}

bool append_shift_index_if_present(const vector<PbctopoShift> &shifts,
                                   const Int3 &delta, bool allow_identity,
                                   vector<size_t> &ordered_indices,
                                   vector<unsigned char> &seen_indices) {
  if (!allow_identity && is_zero_delta(delta))
    return false;
  const size_t shift_idx = find_shift_index(shifts, delta);
  if (shift_idx == numeric_limits<size_t>::max() ||
      shift_idx >= seen_indices.size() || seen_indices[shift_idx] != 0) {
    return false;
  }
  seen_indices[shift_idx] = 1;
  ordered_indices.push_back(shift_idx);
  return true;
}

} // namespace

void clear_component_trial_hints(PbctopoWarmState &warm_state) {
  warm_state.previous_stage1_delta_hints_by_component.clear();
  warm_state.previous_stage2_delta_hints_by_component.clear();
  warm_state.has_component_trial_hints = false;
}

bool component_trial_hints_compatible(const PbctopoWarmState &warm_state,
                                      size_t component_count) {
  return warm_state.has_component_trial_hints &&
         warm_state.previous_stage1_delta_hints_by_component.size() ==
             component_count &&
         warm_state.previous_stage2_delta_hints_by_component.size() ==
             component_count;
}

vector<size_t> build_refine_shift_order(
    const vector<PbctopoShift> &shifts, const Int3 *warm_target_delta,
    const vector<Int3> *stage2_hints, const vector<Int3> *stage1_hints,
    PbctopoTimingStats *timing) {
  vector<size_t> ordered_indices;
  ordered_indices.reserve(shifts.size());
  vector<unsigned char> seen_indices(shifts.size(), 0);

  if (warm_target_delta != nullptr &&
      append_shift_index_if_present(shifts, *warm_target_delta, false,
                                    ordered_indices, seen_indices) &&
      timing != nullptr) {
    ++timing->component_refine_warm_shift_hits;
  }
  auto append_hint_deltas = [&](const vector<Int3> *hint_deltas) {
    if (hint_deltas == nullptr)
      return;
    for (const auto &delta : *hint_deltas) {
      if (append_shift_index_if_present(shifts, delta, false, ordered_indices,
                                        seen_indices) &&
          timing != nullptr) {
        ++timing->component_refine_shortlist_hint_hits;
      }
    }
  };
  append_hint_deltas(stage2_hints);
  append_hint_deltas(stage1_hints);

  for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
    const auto &shift = shifts[shift_idx];
    if ((shift.ix == 0 && shift.iy == 0 && shift.iz == 0) ||
        seen_indices[shift_idx] != 0) {
      continue;
    }
    seen_indices[shift_idx] = 1;
    ordered_indices.push_back(shift_idx);
  }
  return ordered_indices;
}
vector<size_t> build_sampled_anchor_subset(const vector<PbctopoAtom> &atoms,
                                           const vector<size_t> &anchor_indices,
                                           size_t interior_step,
                                           size_t keep_all_cutoff) {
  if (anchor_indices.empty() || atoms.empty())
    return anchor_indices;
  if (interior_step == 0)
    return anchor_indices;

  vector<size_t> coarse_anchor_indices;
  coarse_anchor_indices.reserve(anchor_indices.size());

  auto append_chain_subset = [&](size_t begin, size_t end) {
    if (begin >= end)
      return;
    const size_t count = end - begin;
    if (count <= keep_all_cutoff) {
      coarse_anchor_indices.insert(coarse_anchor_indices.end(),
                                   anchor_indices.begin() + begin,
                                   anchor_indices.begin() + end);
      return;
    }
    coarse_anchor_indices.push_back(anchor_indices[begin]);
    for (size_t idx = begin + 1; idx + 1 < end; idx += interior_step)
      coarse_anchor_indices.push_back(anchor_indices[idx]);
    coarse_anchor_indices.push_back(anchor_indices[end - 1]);
  };

  size_t chain_begin = 0;
  while (chain_begin < anchor_indices.size()) {
    size_t chain_end = chain_begin + 1;
    const size_t anchor_idx = anchor_indices[chain_begin];
    const uint64_t chain_key =
        (anchor_idx < atoms.size()) ? atoms[anchor_idx].chain_key : 0;
    while (chain_end < anchor_indices.size()) {
      const size_t next_anchor_idx = anchor_indices[chain_end];
      const uint64_t next_chain_key =
          (next_anchor_idx < atoms.size()) ? atoms[next_anchor_idx].chain_key : 0;
      if (next_chain_key != chain_key)
        break;
      ++chain_end;
    }
    append_chain_subset(chain_begin, chain_end);
    chain_begin = chain_end;
  }

  return coarse_anchor_indices;
}

vector<size_t> build_coarse_anchor_subset(const vector<PbctopoAtom> &atoms,
                                          const vector<size_t> &anchor_indices) {
  return build_sampled_anchor_subset(atoms, anchor_indices, 2, 6);
}

vector<size_t> build_third_anchor_subset(const vector<PbctopoAtom> &atoms,
                                         const vector<size_t> &anchor_indices) {
  return build_sampled_anchor_subset(atoms, anchor_indices, 3, 9);
}

vector<size_t> build_tapered_stage1_anchor_subset(
    const vector<PbctopoAtom> &atoms, const vector<size_t> &anchor_indices) {
  if (anchor_indices.empty() || atoms.empty())
    return anchor_indices;

  vector<size_t> stage1_anchor_indices;
  const bool ultra_large_stage1 = anchor_indices.size() >= 1024;
  stage1_anchor_indices.reserve(max<size_t>(
      8, anchor_indices.size() / (ultra_large_stage1 ? 8 : 5)));

  auto append_unique = [&](size_t anchor_idx) {
    if (!stage1_anchor_indices.empty() &&
        stage1_anchor_indices.back() == anchor_idx) {
      return;
    }
    stage1_anchor_indices.push_back(anchor_idx);
  };

  auto append_chain_subset = [&](size_t begin, size_t end) {
    if (begin >= end)
      return;
    const size_t count = end - begin;
    if (count <= 18) {
      stage1_anchor_indices.insert(stage1_anchor_indices.end(),
                                   anchor_indices.begin() + begin,
                                   anchor_indices.begin() + end);
      return;
    }

    const size_t outer_span =
        max<size_t>(4, count / (ultra_large_stage1 ? 5 : 4));
    const size_t middle_begin = min(end, begin + outer_span);
    const size_t middle_end = (end > outer_span) ? max(begin, end - outer_span)
                                                 : begin;
    const size_t outer_step = ultra_large_stage1 ? 6 : 5;
    const size_t middle_step = ultra_large_stage1 ? 12 : 8;

    append_unique(anchor_indices[begin]);
    for (size_t idx = begin + 1; idx < middle_begin; idx += outer_step)
      append_unique(anchor_indices[idx]);
    for (size_t idx = middle_begin; idx < middle_end; idx += middle_step)
      append_unique(anchor_indices[idx]);
    for (size_t idx = middle_end; idx + 1 < end; idx += outer_step)
      append_unique(anchor_indices[idx]);
    append_unique(anchor_indices[end - 1]);

    const size_t quarter = begin + count / 4;
    const size_t halfway = begin + count / 2;
    const size_t three_quarter = begin + (3 * count) / 4;
    if (quarter < end)
      append_unique(anchor_indices[quarter]);
    if (halfway < end)
      append_unique(anchor_indices[halfway]);
    if (three_quarter < end)
      append_unique(anchor_indices[three_quarter]);
  };

  size_t chain_begin = 0;
  while (chain_begin < anchor_indices.size()) {
    size_t chain_end = chain_begin + 1;
    const size_t anchor_idx = anchor_indices[chain_begin];
    const uint64_t chain_key =
        (anchor_idx < atoms.size()) ? atoms[anchor_idx].chain_key : 0;
    while (chain_end < anchor_indices.size()) {
      const size_t next_anchor_idx = anchor_indices[chain_end];
      const uint64_t next_chain_key =
          (next_anchor_idx < atoms.size()) ? atoms[next_anchor_idx].chain_key
                                           : 0;
      if (next_chain_key != chain_key)
        break;
      ++chain_end;
    }
    append_chain_subset(chain_begin, chain_end);
    chain_begin = chain_end;
  }

  return stage1_anchor_indices;
}
void build_component_points_from_offsets(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<Int3> &offsets, vector<Vec3> &placed_points,
    double &shift_norm2) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  placed_points.resize(atoms.size());
  shift_norm2 = pbctopo_component_shift_metric(Box, offsets);
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    const auto &offset = offsets[comp_idx];
    const Vec3 translation{
        offset.x * v1[0] + offset.y * v2[0] + offset.z * v3[0],
        offset.x * v1[1] + offset.y * v2[1] + offset.z * v3[1],
        offset.x * v1[2] + offset.y * v2[2] + offset.z * v3[2]};
    for (size_t atom_idx : components[comp_idx]) {
      placed_points[atom_idx] = {atoms[atom_idx].x + translation.x,
                                 atoms[atom_idx].y + translation.y,
                                 atoms[atom_idx].z + translation.z};
    }
  }
}

struct PbctopoBoxBasis {
  Vec3 v1{};
  Vec3 v2{};
  Vec3 v3{};
};

PbctopoBoxBasis build_box_basis(box Box) {
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  return {{v1[0], v1[1], v1[2]},
          {v2[0], v2[1], v2[2]},
          {v3[0], v3[1], v3[2]}};
}

Vec3 component_translation(const PbctopoBoxBasis &basis, const Int3 &offset) {
  return {offset.x * basis.v1.x + offset.y * basis.v2.x +
              offset.z * basis.v3.x,
          offset.x * basis.v1.y + offset.y * basis.v2.y +
              offset.z * basis.v3.y,
          offset.x * basis.v1.z + offset.y * basis.v2.z +
              offset.z * basis.v3.z};
}

PointStats translate_point_stats(const PointStats &base, const Vec3 &delta) {
  if (base.count == 0)
    return {};
  PointStats translated = base;
  const double n = static_cast<double>(base.count);
  translated.sum_x = base.sum_x + n * delta.x;
  translated.sum_y = base.sum_y + n * delta.y;
  translated.sum_z = base.sum_z + n * delta.z;
  translated.sum_x2 =
      base.sum_x2 + 2.0 * delta.x * base.sum_x + n * delta.x * delta.x;
  translated.sum_y2 =
      base.sum_y2 + 2.0 * delta.y * base.sum_y + n * delta.y * delta.y;
  translated.sum_z2 =
      base.sum_z2 + 2.0 * delta.z * base.sum_z + n * delta.z * delta.z;
  translated.sum_xy = base.sum_xy + delta.x * base.sum_y +
                      delta.y * base.sum_x + n * delta.x * delta.y;
  translated.sum_xz = base.sum_xz + delta.x * base.sum_z +
                      delta.z * base.sum_x + n * delta.x * delta.z;
  translated.sum_yz = base.sum_yz + delta.y * base.sum_z +
                      delta.z * base.sum_y + n * delta.y * delta.z;
  translated.sum_sq =
      translated.sum_x2 + translated.sum_y2 + translated.sum_z2;
  translated.min_x = base.min_x + delta.x;
  translated.min_y = base.min_y + delta.y;
  translated.min_z = base.min_z + delta.z;
  translated.max_x = base.max_x + delta.x;
  translated.max_y = base.max_y + delta.y;
  translated.max_z = base.max_z + delta.z;
  return translated;
}

void merge_point_stats(PointStats &dst, const PointStats &src) {
  if (src.count == 0)
    return;
  if (dst.count == 0) {
    dst = src;
    return;
  }
  dst.count += src.count;
  dst.sum_x += src.sum_x;
  dst.sum_y += src.sum_y;
  dst.sum_z += src.sum_z;
  dst.sum_sq += src.sum_sq;
  dst.sum_x2 += src.sum_x2;
  dst.sum_y2 += src.sum_y2;
  dst.sum_z2 += src.sum_z2;
  dst.sum_xy += src.sum_xy;
  dst.sum_xz += src.sum_xz;
  dst.sum_yz += src.sum_yz;
  dst.min_x = min(dst.min_x, src.min_x);
  dst.min_y = min(dst.min_y, src.min_y);
  dst.min_z = min(dst.min_z, src.min_z);
  dst.max_x = max(dst.max_x, src.max_x);
  dst.max_y = max(dst.max_y, src.max_y);
  dst.max_z = max(dst.max_z, src.max_z);
}

void accumulate_component_graph_refine_build_points(
    PbctopoTimingStats *timing, PbctopoRefineScorePhase phase, double delta) {
  if (timing == nullptr)
    return;
  switch (phase) {
  case PbctopoRefineScorePhase::Exact:
    timing->component_graph_refine_build_points_s += delta;
    break;
  case PbctopoRefineScorePhase::CoarseStage1:
    timing->component_graph_refine_coarse_s += delta;
    timing->component_graph_refine_coarse_stage1_s += delta;
    timing->component_graph_refine_coarse_stage1_build_points_s += delta;
    break;
  case PbctopoRefineScorePhase::CoarseStage2:
    timing->component_graph_refine_coarse_s += delta;
    timing->component_graph_refine_coarse_stage2_s += delta;
    timing->component_graph_refine_coarse_stage2_build_points_s += delta;
    break;
  }
}

struct PbctopoRefineScratchState {
  vector<Vec3> *points = nullptr;
  size_t generation = 0;
  size_t component = numeric_limits<size_t>::max();
  bool valid = false;
};

struct PbctopoRefinePlacementWorkspace {
  const vector<PbctopoAtom> *atoms = nullptr;
  const vector<vector<size_t>> *components = nullptr;
  box source_box{};
  PbctopoBoxBasis basis{};
  vector<PointStats> base_component_stats;
  vector<Int3> baseline_offsets;
  vector<Vec3> baseline_points;
  vector<PbctopoRefineScratchState> scratch_states;
  size_t generation = 0;
  bool initialized = false;

  void initialize(const vector<PbctopoAtom> &atoms_ref,
                  const vector<vector<size_t>> &components_ref, box Box) {
    atoms = &atoms_ref;
    components = &components_ref;
    source_box = Box;
    basis = build_box_basis(Box);
    base_component_stats.assign(components_ref.size(), {});
    for (size_t comp_idx = 0; comp_idx < components_ref.size(); ++comp_idx) {
      PointStats stats;
      for (size_t atom_idx : components_ref[comp_idx]) {
        if (atom_idx >= atoms_ref.size())
          continue;
        const auto &atom = atoms_ref[atom_idx];
        stats.add(atom.x, atom.y, atom.z);
      }
      base_component_stats[comp_idx] = stats;
    }
    initialized = true;
  }

  void reset_baseline(const vector<Int3> &offsets,
                      const vector<Vec3> &placed_points) {
    vector<size_t> changed_components;
    const bool can_update_scratch =
        (initialized && atoms != nullptr && components != nullptr &&
         placed_points.size() == atoms->size() &&
         baseline_points.size() == atoms->size() &&
         baseline_offsets.size() == offsets.size() &&
         offsets.size() == components->size());
    if (can_update_scratch) {
      for (size_t comp_idx = 0; comp_idx < offsets.size(); ++comp_idx) {
        if (!same_delta(offsets[comp_idx], baseline_offsets[comp_idx]))
          changed_components.push_back(comp_idx);
      }
      for (auto &state : scratch_states) {
        if (!state.valid || state.points == nullptr ||
            state.generation != generation ||
            state.points->size() != atoms->size()) {
          state.valid = false;
          continue;
        }
        if (state.component != numeric_limits<size_t>::max()) {
          if (state.component >= components->size()) {
            state.valid = false;
            continue;
          }
          for (size_t atom_idx : (*components)[state.component]) {
            if (atom_idx < baseline_points.size() &&
                atom_idx < state.points->size()) {
              (*state.points)[atom_idx] = baseline_points[atom_idx];
            }
          }
        }
        for (size_t comp_idx : changed_components) {
          for (size_t atom_idx : (*components)[comp_idx]) {
            if (atom_idx < placed_points.size() &&
                atom_idx < state.points->size()) {
              (*state.points)[atom_idx] = placed_points[atom_idx];
            }
          }
        }
        state.component = numeric_limits<size_t>::max();
      }
    } else {
      for (auto &state : scratch_states)
        state.valid = false;
    }
    baseline_offsets = offsets;
    baseline_points = placed_points;
    ++generation;
    if (can_update_scratch) {
      for (auto &state : scratch_states) {
        if (state.valid)
          state.generation = generation;
      }
    }
  }

  PbctopoRefineScratchState &scratch_state(vector<Vec3> &points) {
    for (auto &state : scratch_states) {
      if (state.points == &points)
        return state;
    }
    scratch_states.push_back({&points, 0, numeric_limits<size_t>::max(), false});
    return scratch_states.back();
  }

  bool single_component_delta(size_t comp_idx, const vector<Int3> &offsets,
                              bool &global_rebase) const {
    global_rebase = false;
    if (!initialized || components == nullptr || atoms == nullptr ||
        offsets.size() != baseline_offsets.size() ||
        baseline_offsets.size() != components->size() ||
        baseline_points.size() != atoms->size() || comp_idx >= components->size()) {
      global_rebase = true;
      return false;
    }
    size_t changed_count = 0;
    size_t changed_idx = numeric_limits<size_t>::max();
    for (size_t idx = 0; idx < offsets.size(); ++idx) {
      if (same_delta(offsets[idx], baseline_offsets[idx]))
        continue;
      ++changed_count;
      changed_idx = idx;
      if (changed_count > 1) {
        global_rebase = true;
        return false;
      }
    }
    if (changed_count == 0)
      return true;
    if (changed_idx == comp_idx)
      return true;
    global_rebase = true;
    return false;
  }

  PointStats stats_from_offsets(const vector<Int3> &offsets) const {
    PointStats merged;
    if (components == nullptr)
      return merged;
    const size_t count =
        min({components->size(), offsets.size(), base_component_stats.size()});
    for (size_t comp_idx = 0; comp_idx < count; ++comp_idx) {
      const Vec3 translation = component_translation(basis, offsets[comp_idx]);
      merge_point_stats(
          merged, translate_point_stats(base_component_stats[comp_idx],
                                        translation));
    }
    return merged;
  }

  bool prepare_trial(size_t comp_idx, const vector<Int3> &offsets,
                     vector<Vec3> &out_points, PointStats &out_stats,
                     double &shift_norm2, PbctopoTimingStats *timing,
                     PbctopoRefineScorePhase phase) {
    bool global_rebase = false;
    const bool incremental = single_component_delta(comp_idx, offsets,
                                                    global_rebase);
    auto build_started = chrono::steady_clock::now();
    if (incremental) {
      auto &state = scratch_state(out_points);
      if (!state.valid || state.generation != generation ||
          (state.component != comp_idx &&
           state.component != numeric_limits<size_t>::max())) {
        out_points = baseline_points;
        state.valid = true;
        state.generation = generation;
      }
      state.component = comp_idx;
      const Vec3 translation = component_translation(basis, offsets[comp_idx]);
      for (size_t atom_idx : (*components)[comp_idx]) {
        if (atom_idx >= atoms->size() || atom_idx >= out_points.size())
          continue;
        const auto &atom = (*atoms)[atom_idx];
        out_points[atom_idx] = {atom.x + translation.x, atom.y + translation.y,
                                atom.z + translation.z};
      }
      if (timing != nullptr) {
        ++timing->component_graph_refine_trial_build_incremental;
        accumulate_component_graph_refine_build_points(
            timing, phase,
            chrono::duration<double>(chrono::steady_clock::now() -
                                     build_started)
                .count());
      }
      auto stats_started = chrono::steady_clock::now();
      out_stats = stats_from_offsets(offsets);
      shift_norm2 = pbctopo_component_shift_metric(source_box, offsets);
      if (timing != nullptr) {
        ++timing->component_graph_refine_stats_cached;
        timing->component_graph_refine_stats_s +=
            chrono::duration<double>(chrono::steady_clock::now() -
                                     stats_started)
                .count();
      }
      return true;
    }

    if (timing != nullptr && global_rebase)
      ++timing->component_graph_refine_trial_build_global_rebase;
    build_component_points_from_offsets(*atoms, *components, source_box, offsets,
                                        out_points, shift_norm2);
    scratch_state(out_points).valid = false;
    if (timing != nullptr) {
      ++timing->component_graph_refine_trial_build_full;
      accumulate_component_graph_refine_build_points(
          timing, phase,
          chrono::duration<double>(chrono::steady_clock::now() - build_started)
              .count());
    }
    auto stats_started = chrono::steady_clock::now();
    out_stats = build_point_stats(out_points);
    if (timing != nullptr) {
      ++timing->component_graph_refine_stats_full;
      timing->component_graph_refine_stats_s +=
          chrono::duration<double>(chrono::steady_clock::now() -
                                   stats_started)
              .count();
    }
    return false;
  }
};
PbctopoCutCandidate score_component_placed_points_solution(
    const vector<Vec3> &placed_points, double shift_norm2,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set, double clash_cutoff,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, double continuity_cutoff,
    const PbctopoRefineScoringCallbacks &callbacks,
    const PbctopoCutCandidate *incumbent,
    PbctopoTimingStats *timing, bool run_ultraclash,
    PbctopoRefineScorePhase phase,
    const PbctopoStage1EdgeContext *stage1_edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    AnchorBackend anchor_backend,
    const PointStats *precomputed_stats,
    const PbctopoAnchorGraphContext *anchor_context,
    const vector<PbctopoAtom> *clash_atoms,
    const vector<size_t> *atom_component_index) {
  if (timing != nullptr) {
    switch (phase) {
    case PbctopoRefineScorePhase::Exact:
      ++timing->component_graph_refine_score_calls;
      break;
    case PbctopoRefineScorePhase::CoarseStage1:
      ++timing->component_graph_refine_coarse_score_calls;
      ++timing->component_graph_refine_coarse_stage1_score_calls;
      break;
    case PbctopoRefineScorePhase::CoarseStage2:
      ++timing->component_graph_refine_coarse_score_calls;
      ++timing->component_graph_refine_coarse_stage2_score_calls;
      break;
    }
  }
  auto accumulate_continuity = [&](double delta) {
    if (timing == nullptr)
      return;
    switch (phase) {
    case PbctopoRefineScorePhase::Exact:
      timing->component_graph_refine_continuity_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage1:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage1_s += delta;
      timing->component_graph_refine_coarse_stage1_continuity_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage2:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage2_s += delta;
      timing->component_graph_refine_coarse_stage2_continuity_s += delta;
      break;
    }
  };
  auto accumulate_anchor = [&](double delta) {
    if (timing == nullptr)
      return;
    switch (phase) {
    case PbctopoRefineScorePhase::Exact:
      timing->component_graph_refine_anchor_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage1:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage1_s += delta;
      timing->component_graph_refine_coarse_stage1_anchor_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage2:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage2_s += delta;
      timing->component_graph_refine_coarse_stage2_anchor_s += delta;
      break;
    }
  };
  auto accumulate_cut_score = [&](double delta) {
    if (timing == nullptr)
      return;
    switch (phase) {
    case PbctopoRefineScorePhase::Exact:
      timing->component_graph_refine_cut_score_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage1:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage1_s += delta;
      timing->component_graph_refine_coarse_stage1_cut_score_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage2:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage2_s += delta;
        timing->component_graph_refine_coarse_stage2_cut_score_s += delta;
      break;
    }
  };
  chrono::steady_clock::time_point stage_started;
  if (timing != nullptr)
    stage_started = chrono::steady_clock::now();
  double continuity_path = numeric_limits<double>::infinity();
  double continuity_max_d2 = numeric_limits<double>::infinity();
  double continuity_min_margin_A = numeric_limits<double>::quiet_NaN();
  size_t continuity_margin_atom_a = numeric_limits<size_t>::max();
  size_t continuity_margin_atom_b = numeric_limits<size_t>::max();
  if (!callbacks.continuity_metrics_ok(
          placed_points, continuity_pairs, continuity_cutoff, continuity_path,
          continuity_max_d2, continuity_min_margin_A,
          continuity_margin_atom_a, continuity_margin_atom_b, incumbent)) {
    if (timing != nullptr) {
      accumulate_continuity(
          chrono::duration<double>(chrono::steady_clock::now() - stage_started)
              .count());
    }
    return {};
  }
  if (timing != nullptr) {
    accumulate_continuity(
        chrono::duration<double>(chrono::steady_clock::now() - stage_started)
            .count());
    stage_started = chrono::steady_clock::now();
  }
  double anchor_mst2 = numeric_limits<double>::infinity();
  double anchor_max_d2 = numeric_limits<double>::infinity();
  PbctopoAnchorGraphScore anchor_graph_score;
  if (!anchor_graph_metrics_enhanced(
          placed_points, anchor_indices, continuity_pairs, anchor_mst2,
          anchor_max_d2, anchor_graph_score, incumbent, phase, anchor_backend,
          timing, anchor_context)) {
    if (timing != nullptr) {
      accumulate_anchor(
          chrono::duration<double>(chrono::steady_clock::now() - stage_started)
              .count());
    }
    return {};
  }
  if (timing != nullptr) {
    accumulate_anchor(
        chrono::duration<double>(chrono::steady_clock::now() - stage_started)
            .count());
    stage_started = chrono::steady_clock::now();
  }
  PointStats stats_storage;
  const PointStats *stats_ptr = precomputed_stats;
  if (stats_ptr == nullptr) {
    const auto stats_started = chrono::steady_clock::now();
    stats_storage = build_point_stats(placed_points);
    stats_ptr = &stats_storage;
    if (timing != nullptr) {
      ++timing->component_graph_refine_stats_full;
      timing->component_graph_refine_stats_s +=
          chrono::duration<double>(chrono::steady_clock::now() -
                                   stats_started)
              .count();
    }
  }
  const PointStats &stats = *stats_ptr;
  PbctopoCutCandidate candidate;
  PbctopoCutScoreBreakdown cut_breakdown;
  PbctopoCutCandidate pre_edge_seed;
  pre_edge_seed.continuity_path = continuity_path;
  pre_edge_seed.continuity_max_d2 = continuity_max_d2;
  pre_edge_seed.continuity_min_margin_A = continuity_min_margin_A;
  pre_edge_seed.continuity_margin_atom_a = continuity_margin_atom_a;
  pre_edge_seed.continuity_margin_atom_b = continuity_margin_atom_b;
  pre_edge_seed.anchor_mst2 = anchor_mst2;
  pre_edge_seed.anchor_max_d2 = anchor_max_d2;
  pre_edge_seed.anchor_graph = anchor_graph_score;
  if (phase == PbctopoRefineScorePhase::CoarseStage1) {
    candidate = callbacks.score_cut_candidate_principal_only(
        local_edges, placed_points, 1.5, 1.5, 1.5, 0.0, stats, shift_norm2,
        &cut_breakdown, stage1_edge_context, edge_scan_soa,
        edge_prune_threshold, &pre_edge_seed);
  } else {
    candidate =
        callbacks.score_cut_candidate(local_edges, placed_points, 1.5, 1.5, 1.5, 0.0,
                            stats, shift_norm2,
                            (phase == PbctopoRefineScorePhase::CoarseStage2)
                                ? stage1_edge_context
                                : nullptr,
                            edge_scan_soa, edge_prune_threshold, &pre_edge_seed,
                            phase);
  }
  if (!candidate.valid) {
    if (timing != nullptr) {
      accumulate_cut_score(
          chrono::duration<double>(chrono::steady_clock::now() - stage_started)
              .count());
    }
    return {};
  }
  if (timing != nullptr && phase == PbctopoRefineScorePhase::CoarseStage1) {
    timing->component_graph_refine_coarse_stage1_cut_score_principal_s +=
        cut_breakdown.principal_s;
    timing->component_graph_refine_coarse_stage1_cut_score_edge_scan_s +=
        cut_breakdown.edge_scan_s;
  }
  candidate.continuity_path = continuity_path;
  candidate.continuity_max_d2 = continuity_max_d2;
  candidate.continuity_min_margin_A = continuity_min_margin_A;
  candidate.continuity_margin_atom_a = continuity_margin_atom_a;
  candidate.continuity_margin_atom_b = continuity_margin_atom_b;
  candidate.anchor_mst2 = anchor_mst2;
  candidate.anchor_max_d2 = anchor_max_d2;
  candidate.anchor_graph = anchor_graph_score;
  if (callbacks.cut_candidate_cannot_beat_incumbent(candidate, incumbent)) {
    if (timing != nullptr) {
      accumulate_cut_score(
          chrono::duration<double>(chrono::steady_clock::now() - stage_started)
              .count());
      if (phase == PbctopoRefineScorePhase::Exact)
        ++timing->component_graph_refine_preclash_prunes;
    }
    return {};
  }
  if (timing != nullptr) {
    accumulate_cut_score(
        chrono::duration<double>(chrono::steady_clock::now() - stage_started)
            .count());
  }
  if (run_ultraclash) {
    if (timing != nullptr)
      stage_started = chrono::steady_clock::now();
    callbacks.analyze_cut_candidate_ultraclash(
        placed_points, edge_set, candidate, clash_cutoff, clash_atoms,
        atom_component_index);
    if (timing != nullptr) {
      timing->component_graph_refine_ultraclash_s +=
          chrono::duration<double>(chrono::steady_clock::now() - stage_started)
              .count();
    }
  }
  return candidate;
}

PbctopoCutCandidate score_component_offset_solution(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<Int3> &offsets,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set, double clash_cutoff,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices,
    double continuity_cutoff, vector<Vec3> &placed_points,
    const PbctopoRefineScoringCallbacks &callbacks,
    const PbctopoCutCandidate *incumbent,
    PbctopoTimingStats *timing, bool run_ultraclash,
    PbctopoRefineScorePhase phase,
    const PbctopoStage1EdgeContext *stage1_edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    AnchorBackend anchor_backend,
    const PbctopoAnchorGraphContext *anchor_context,
    const vector<size_t> *atom_component_index) {
  auto accumulate_build_points = [&](double delta) {
    if (timing == nullptr)
      return;
    switch (phase) {
    case PbctopoRefineScorePhase::Exact:
      timing->component_graph_refine_build_points_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage1:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage1_s += delta;
      timing->component_graph_refine_coarse_stage1_build_points_s += delta;
      break;
    case PbctopoRefineScorePhase::CoarseStage2:
      timing->component_graph_refine_coarse_s += delta;
      timing->component_graph_refine_coarse_stage2_s += delta;
      timing->component_graph_refine_coarse_stage2_build_points_s += delta;
      break;
    }
  };
  double shift_norm2 = 0.0;
  chrono::steady_clock::time_point stage_started;
  if (timing != nullptr)
    stage_started = chrono::steady_clock::now();
  build_component_points_from_offsets(atoms, components, Box, offsets, placed_points,
                                      shift_norm2);
  if (timing != nullptr) {
    accumulate_build_points(
        chrono::duration<double>(chrono::steady_clock::now() - stage_started)
            .count());
  }
  vector<size_t> local_atom_component_index;
  if (run_ultraclash && atom_component_index == nullptr) {
    local_atom_component_index =
        build_atom_component_index(atoms.size(), components);
    atom_component_index = &local_atom_component_index;
  }
  return score_component_placed_points_solution(
      placed_points, shift_norm2, local_edges, edge_set, clash_cutoff,
      continuity_pairs, anchor_indices, continuity_cutoff, callbacks, incumbent,
      timing, run_ultraclash, phase, stage1_edge_context, edge_scan_soa,
      edge_prune_threshold, anchor_backend, nullptr, anchor_context, &atoms,
      atom_component_index);
}

PbctopoCutCandidate improve_component_offsets_locally(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<PbctopoShift> &shifts,
    const vector<PbctopoLocalEdge> &local_edges,
    const unordered_set<uint64_t> &edge_set,
    const vector<PbctopoContinuityPair> &continuity_pairs,
    const vector<size_t> &anchor_indices, vector<Int3> &offsets,
    double clash_cutoff, double continuity_cutoff, vector<Vec3> &placed_points,
    const PbctopoRefineScoringCallbacks &callbacks,
    AnchorBackend anchor_backend,
    PbctopoTimingStats *timing, PbctopoWarmState *warm_state,
    const PbctopoCutCandidate *warm_reference_candidate,
    const PbctopoAnchorGraphContext *anchor_context) {
  const vector<size_t> &exact_anchor_indices = anchor_indices;
  const vector<size_t> atom_component_index =
      build_atom_component_index(atoms.size(), components);
  PbctopoCutCandidate best = score_component_offset_solution(
      atoms, components, Box, offsets, local_edges, edge_set, clash_cutoff,
      continuity_pairs, exact_anchor_indices, continuity_cutoff, placed_points,
      callbacks, nullptr, timing, true, PbctopoRefineScorePhase::Exact,
      nullptr, nullptr, nullptr, anchor_backend, anchor_context,
      &atom_component_index);
  if (!best.valid || offsets.size() < 2) {
    if (warm_state != nullptr && (!best.valid || best.ultra_clash))
      clear_component_trial_hints(*warm_state);
    return best;
  }
  PbctopoRefinePlacementWorkspace refine_workspace;
  refine_workspace.initialize(atoms, components, Box);

  const vector<size_t> half_anchor_indices =
      build_coarse_anchor_subset(atoms, anchor_indices);
  const vector<size_t> third_anchor_indices =
      build_third_anchor_subset(atoms, anchor_indices);
  const vector<size_t> tapered_stage1_anchor_indices =
      build_tapered_stage1_anchor_subset(atoms, anchor_indices);
  const bool use_three_stage_anchor =
      anchor_indices.size() > 256 &&
      tapered_stage1_anchor_indices.size() >= 2 &&
      third_anchor_indices.size() >= 2 &&
      tapered_stage1_anchor_indices.size() + 4 < third_anchor_indices.size() &&
      third_anchor_indices.size() + 2 < anchor_indices.size();
  const bool use_two_stage_anchor =
      !use_three_stage_anchor &&
      anchor_indices.size() >= 8 && half_anchor_indices.size() >= 2 &&
      half_anchor_indices.size() + 2 < anchor_indices.size();
  const PbctopoEdgeScanSoA edge_scan_soa =
      (use_two_stage_anchor || use_three_stage_anchor)
          ? build_edge_scan_soa(local_edges)
          : PbctopoEdgeScanSoA{};
  const bool warm_offsets_compatible =
      (warm_state != nullptr && warm_state->has_component_offsets &&
       warm_state->previous_component_offsets.size() == offsets.size() &&
       same_component_layout(components,
                             warm_state->previous_component_layout));
  const bool warm_trial_hints_compatible =
      (warm_state != nullptr &&
       component_trial_hints_compatible(*warm_state, offsets.size()) &&
       same_component_layout(components,
                             warm_state->previous_component_layout));
  const bool strong_warm_candidate =
      (warm_reference_candidate != nullptr && warm_reference_candidate->valid &&
       !warm_reference_candidate->ultra_clash &&
       callbacks.candidate_close_enough_to_skip_search(best, *warm_reference_candidate));
  size_t stage1_keep = 6;
  size_t stage2_keep = 3;
  size_t exact_keep = 3;
  if (strong_warm_candidate &&
      (use_two_stage_anchor || use_three_stage_anchor)) {
    if (use_three_stage_anchor)
      stage1_keep = 4;
    stage2_keep = 2;
    exact_keep = 2;
    if (timing != nullptr)
      ++timing->component_refine_fanout_reduced;
  }

  struct PbctopoRefineTrialCandidate {
    vector<Int3> offsets;
    PbctopoCutCandidate candidate;
  };
  auto push_top_refine_trial =
      [](vector<PbctopoRefineTrialCandidate> &top_trials,
         const vector<Int3> &trial_offsets,
         const PbctopoCutCandidate &trial_candidate, size_t max_keep,
         bool (*better_candidate)(const PbctopoCutCandidate &,
                                  const PbctopoCutCandidate &)) {
        if (!trial_candidate.valid)
          return;
        PbctopoRefineTrialCandidate entry;
        entry.offsets = trial_offsets;
        entry.candidate = trial_candidate;
        auto insert_pos = top_trials.begin();
        while (insert_pos != top_trials.end() &&
               !better_candidate(entry.candidate, insert_pos->candidate)) {
          ++insert_pos;
        }
         top_trials.insert(insert_pos, std::move(entry));
         if (top_trials.size() > max_keep)
           top_trials.resize(max_keep);
      };
  vector<vector<Int3>> stage1_absolute_hint_offsets(offsets.size());
  vector<vector<Int3>> stage2_absolute_hint_offsets(offsets.size());
  auto score_refine_trial =
      [&](size_t trial_comp_idx, const vector<Int3> &trial_offsets_ref,
          const vector<size_t> &trial_anchor_indices, vector<Vec3> &trial_points,
          const PbctopoCutCandidate *incumbent, bool run_ultraclash,
          PbctopoRefineScorePhase phase,
          const PbctopoStage1EdgeContext *stage1_edge_context,
          const PbctopoEdgeScanSoA *edge_scan_soa,
          const PbctopoCutCandidate *edge_prune_threshold) {
        PointStats trial_stats;
        double trial_shift_norm2 = 0.0;
        refine_workspace.prepare_trial(trial_comp_idx, trial_offsets_ref,
                                       trial_points, trial_stats,
                                       trial_shift_norm2, timing, phase);
        return score_component_placed_points_solution(
            trial_points, trial_shift_norm2, local_edges, edge_set,
            clash_cutoff, continuity_pairs, trial_anchor_indices,
            continuity_cutoff, callbacks, incumbent, timing, run_ultraclash,
            phase, stage1_edge_context, edge_scan_soa, edge_prune_threshold,
            anchor_backend, &trial_stats,
            (&trial_anchor_indices == &exact_anchor_indices) ? anchor_context
                                                             : nullptr,
            &atoms, &atom_component_index);
      };
  auto capture_component_hint_offsets =
      [](const vector<PbctopoRefineTrialCandidate> &source_trials,
         size_t comp_idx, vector<vector<Int3>> &target_offsets) {
        if (comp_idx >= target_offsets.size())
          return;
        auto &dst = target_offsets[comp_idx];
        dst.clear();
        dst.reserve(source_trials.size());
        for (const auto &trial : source_trials) {
          if (comp_idx < trial.offsets.size())
            dst.push_back(trial.offsets[comp_idx]);
        }
      };
  for (int pass = 0; pass < 2; ++pass) {
    bool improved = false;
    vector<Int3> trial_offsets = offsets;
    vector<Vec3> local_points;
    local_points.reserve(atoms.size());
    vector<Vec3> coarse_points;
    coarse_points.reserve(atoms.size());
    for (size_t comp_idx = 1; comp_idx < offsets.size(); ++comp_idx) {
      refine_workspace.reset_baseline(offsets, placed_points);
      vector<Int3> best_offsets = offsets;
      PbctopoCutCandidate best_local = best;
      PbctopoStage1EdgeContext stage1_edge_context;
      if (use_two_stage_anchor || use_three_stage_anchor) {
        stage1_edge_context = build_stage1_edge_context(
            local_edges, atom_component_index, comp_idx, placed_points);
      }
      Int3 warm_target_delta;
      const Int3 *warm_target_delta_ptr = nullptr;
      if (warm_offsets_compatible &&
          comp_idx < warm_state->previous_component_offsets.size()) {
        if (checked_int3_subtract(
                warm_state->previous_component_offsets[comp_idx],
                offsets[comp_idx], warm_target_delta)) {
          warm_target_delta_ptr = &warm_target_delta;
        }
      }
      const vector<Int3> *stage1_hint_deltas =
          (warm_trial_hints_compatible &&
           comp_idx < warm_state->previous_stage1_delta_hints_by_component.size())
              ? &warm_state->previous_stage1_delta_hints_by_component[comp_idx]
              : nullptr;
      const vector<Int3> *stage2_hint_deltas =
          (warm_trial_hints_compatible &&
           comp_idx < warm_state->previous_stage2_delta_hints_by_component.size())
              ? &warm_state->previous_stage2_delta_hints_by_component[comp_idx]
              : nullptr;
      const vector<size_t> ordered_shift_indices = build_refine_shift_order(
          shifts, warm_target_delta_ptr, stage2_hint_deltas, stage1_hint_deltas,
          timing);
      if (!use_two_stage_anchor && !use_three_stage_anchor) {
        for (size_t shift_pos : ordered_shift_indices) {
          const auto &shift = shifts[shift_pos];
          trial_offsets = offsets;
          if (!checked_int3_add(offsets[comp_idx],
                                Int3{shift.ix, shift.iy, shift.iz},
                                trial_offsets[comp_idx])) {
            continue;
          }
          bool rebase_ok = false;
          if (timing != nullptr) {
            const auto rebase_started = chrono::steady_clock::now();
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
            timing->component_graph_refine_rebase_s +=
                chrono::duration<double>(chrono::steady_clock::now() -
                                         rebase_started)
                    .count();
          } else {
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
          }
          if (!rebase_ok)
            continue;
          PbctopoCutCandidate trial = score_refine_trial(
              comp_idx, trial_offsets, exact_anchor_indices, local_points,
              &best_local, true, PbctopoRefineScorePhase::Exact, nullptr,
              nullptr, nullptr);
          if (callbacks.better_cut_candidate(trial, best_local)) {
            best_local = trial;
            best_offsets = trial_offsets;
            placed_points = local_points;
          }
        }
      } else if (use_two_stage_anchor) {
        vector<PbctopoRefineTrialCandidate> top_trials;
        top_trials.reserve(stage2_keep);
        for (size_t shift_pos : ordered_shift_indices) {
          const auto &shift = shifts[shift_pos];
          trial_offsets = offsets;
          if (!checked_int3_add(offsets[comp_idx],
                                Int3{shift.ix, shift.iy, shift.iz},
                                trial_offsets[comp_idx])) {
            continue;
          }
          bool rebase_ok = false;
          if (timing != nullptr) {
            const auto rebase_started = chrono::steady_clock::now();
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
            timing->component_graph_refine_rebase_s +=
                chrono::duration<double>(chrono::steady_clock::now() -
                                         rebase_started)
                    .count();
          } else {
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
          }
          if (!rebase_ok)
            continue;
          const PbctopoCutCandidate *stage2_prune_threshold =
              (top_trials.size() >= stage2_keep) ? &top_trials.back().candidate
                                                 : nullptr;
          PbctopoCutCandidate coarse_trial = score_refine_trial(
              comp_idx, trial_offsets, half_anchor_indices, coarse_points,
              nullptr, false, PbctopoRefineScorePhase::CoarseStage2,
              &stage1_edge_context, &edge_scan_soa, stage2_prune_threshold);
          push_top_refine_trial(top_trials, trial_offsets, coarse_trial,
                                stage2_keep,
                                callbacks.better_preclash_cut_candidate);
        }
        capture_component_hint_offsets(top_trials, comp_idx,
                                       stage2_absolute_hint_offsets);
        for (size_t shortlist_idx = 0;
             shortlist_idx < top_trials.size() && shortlist_idx < exact_keep;
             ++shortlist_idx) {
          const auto &shortlisted = top_trials[shortlist_idx];
          if (timing != nullptr)
            ++timing->component_graph_refine_exact_rescore_calls;
          PbctopoCutCandidate trial = score_refine_trial(
              comp_idx, shortlisted.offsets, exact_anchor_indices, local_points,
              &best_local, true, PbctopoRefineScorePhase::Exact, nullptr,
              nullptr, nullptr);
          if (callbacks.better_cut_candidate(trial, best_local)) {
            best_local = trial;
            best_offsets = shortlisted.offsets;
            placed_points = local_points;
          }
        }
      } else {
        vector<PbctopoRefineTrialCandidate> sparse_top_trials;
        sparse_top_trials.reserve(stage1_keep);
        for (size_t shift_pos : ordered_shift_indices) {
          const auto &shift = shifts[shift_pos];
          trial_offsets = offsets;
          if (!checked_int3_add(offsets[comp_idx],
                                Int3{shift.ix, shift.iy, shift.iz},
                                trial_offsets[comp_idx])) {
            continue;
          }
          bool rebase_ok = false;
          if (timing != nullptr) {
            const auto rebase_started = chrono::steady_clock::now();
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
            timing->component_graph_refine_rebase_s +=
                chrono::duration<double>(chrono::steady_clock::now() -
                                         rebase_started)
                    .count();
          } else {
            rebase_ok = rebase_component_offsets(trial_offsets, Box);
          }
          if (!rebase_ok)
            continue;
          const PbctopoCutCandidate *stage1_prune_threshold =
              (sparse_top_trials.size() >= stage1_keep)
                  ? &sparse_top_trials.back().candidate
                  : nullptr;
          PbctopoCutCandidate sparse_trial = score_refine_trial(
              comp_idx, trial_offsets, tapered_stage1_anchor_indices,
              coarse_points, nullptr, false,
              PbctopoRefineScorePhase::CoarseStage1, &stage1_edge_context,
              &edge_scan_soa, stage1_prune_threshold);
          push_top_refine_trial(sparse_top_trials, trial_offsets, sparse_trial,
                                stage1_keep,
                                callbacks.better_stage1_cut_candidate);
        }
        capture_component_hint_offsets(sparse_top_trials, comp_idx,
                                       stage1_absolute_hint_offsets);

        vector<PbctopoRefineTrialCandidate> top_trials;
        top_trials.reserve(stage2_keep);
        for (const auto &shortlisted : sparse_top_trials) {
          const PbctopoCutCandidate *stage2_prune_threshold =
              (top_trials.size() >= stage2_keep)
                  ? &top_trials.back().candidate
                  : nullptr;
          PbctopoCutCandidate coarse_trial = score_refine_trial(
              comp_idx, shortlisted.offsets, third_anchor_indices,
              coarse_points, nullptr, false,
              PbctopoRefineScorePhase::CoarseStage2, &stage1_edge_context,
              &edge_scan_soa, stage2_prune_threshold);
          push_top_refine_trial(top_trials, shortlisted.offsets, coarse_trial,
                                stage2_keep,
                                callbacks.better_preclash_cut_candidate);
        }
        capture_component_hint_offsets(top_trials, comp_idx,
                                       stage2_absolute_hint_offsets);

        for (size_t shortlist_idx = 0;
             shortlist_idx < top_trials.size() && shortlist_idx < exact_keep;
             ++shortlist_idx) {
          const auto &shortlisted = top_trials[shortlist_idx];
          if (timing != nullptr)
            ++timing->component_graph_refine_exact_rescore_calls;
          PbctopoCutCandidate trial = score_refine_trial(
              comp_idx, shortlisted.offsets, exact_anchor_indices, local_points,
              &best_local, true, PbctopoRefineScorePhase::Exact, nullptr,
              nullptr, nullptr);
          if (callbacks.better_cut_candidate(trial, best_local)) {
            best_local = trial;
            best_offsets = shortlisted.offsets;
            placed_points = local_points;
          }
        }
      }
      if (callbacks.better_cut_candidate(best_local, best)) {
        offsets = best_offsets;
        best = best_local;
        improved = true;
      }
    }
    if (!improved)
      break;
  }

  if (warm_state != nullptr) {
    if (!best.valid || best.ultra_clash) {
      clear_component_trial_hints(*warm_state);
    } else {
      warm_state->previous_stage1_delta_hints_by_component.assign(offsets.size(),
                                                                  {});
      warm_state->previous_stage2_delta_hints_by_component.assign(offsets.size(),
                                                                  {});
      auto convert_absolute_offsets_to_deltas =
          [&](const vector<vector<Int3>> &absolute_offsets,
              vector<vector<Int3>> &delta_hints) {
            for (size_t comp_idx = 0; comp_idx < absolute_offsets.size() &&
                                      comp_idx < delta_hints.size();
                 ++comp_idx) {
              for (const auto &absolute_offset : absolute_offsets[comp_idx]) {
                Int3 delta{};
                if (!checked_int3_subtract(absolute_offset,
                                           offsets[comp_idx], delta)) {
                  continue;
                }
                if (is_zero_delta(delta))
                  continue;
                auto &dst = delta_hints[comp_idx];
                if (find_if(dst.begin(), dst.end(),
                            [&](const Int3 &existing) {
                              return same_delta(existing, delta);
                            }) == dst.end()) {
                  dst.push_back(delta);
                }
              }
            }
          };
      convert_absolute_offsets_to_deltas(
          stage1_absolute_hint_offsets,
          warm_state->previous_stage1_delta_hints_by_component);
      convert_absolute_offsets_to_deltas(
          stage2_absolute_hint_offsets,
          warm_state->previous_stage2_delta_hints_by_component);
      warm_state->has_component_trial_hints = false;
      for (size_t comp_idx = 0;
           comp_idx < warm_state->previous_stage1_delta_hints_by_component.size();
           ++comp_idx) {
        if (!warm_state->previous_stage1_delta_hints_by_component[comp_idx]
                 .empty() ||
            !warm_state->previous_stage2_delta_hints_by_component[comp_idx]
                 .empty()) {
          warm_state->has_component_trial_hints = true;
          break;
        }
      }
    }
  }
  return best;
}
vector<size_t> build_atom_component_index(
    size_t atom_count, const vector<vector<size_t>> &components) {
  vector<size_t> atom_component_index(atom_count, numeric_limits<size_t>::max());
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    for (size_t atom_idx : components[comp_idx]) {
      if (atom_idx < atom_component_index.size())
        atom_component_index[atom_idx] = comp_idx;
    }
  }
  return atom_component_index;
}

PbctopoStage1EdgeContext build_stage1_edge_context(
    const vector<PbctopoLocalEdge> &local_edges,
    const vector<size_t> &atom_component_index, size_t moving_component_idx,
    const vector<Vec3> &placed_points) {
  PbctopoStage1EdgeContext context;
  if (atom_component_index.empty() || placed_points.empty())
    return context;

  context.variable_edge_indices.reserve(local_edges.size() / 4 + 1);
  for (size_t edge_idx = 0; edge_idx < local_edges.size(); ++edge_idx) {
    const auto &edge = local_edges[edge_idx];
    const size_t comp_a = (edge.a < atom_component_index.size())
                              ? atom_component_index[edge.a]
                              : numeric_limits<size_t>::max();
    const size_t comp_b = (edge.b < atom_component_index.size())
                              ? atom_component_index[edge.b]
                              : numeric_limits<size_t>::max();
    const bool variable =
        comp_a != comp_b &&
        (comp_a == moving_component_idx || comp_b == moving_component_idx);
    if (variable) {
      context.variable_edge_indices.push_back(edge_idx);
      continue;
    }

    const Vec3 delta = placed_points[edge.a] - placed_points[edge.b];
    const double d2 = norm2(delta);
    if (d2 > edge.break_cutoff_d2)
      ++context.fixed_broken_edges;
    context.fixed_edge_stretch +=
        pbctopo_symmetric_reference_strain(d2, edge.ref_d2);
  }
  return context;
}

PbctopoEdgeScanSoA
build_edge_scan_soa(const vector<PbctopoLocalEdge> &local_edges) {
  PbctopoEdgeScanSoA soa;
  soa.a.reserve(local_edges.size());
  soa.b.reserve(local_edges.size());
  soa.ref_d2.reserve(local_edges.size());
  soa.inv_ref_d2.reserve(local_edges.size());
  soa.break_cutoff_d2.reserve(local_edges.size());
  for (const auto &edge : local_edges) {
    soa.a.push_back(edge.a);
    soa.b.push_back(edge.b);
    soa.ref_d2.push_back(edge.ref_d2);
    soa.inv_ref_d2.push_back(edge.inv_ref_d2);
    soa.break_cutoff_d2.push_back(edge.break_cutoff_d2);
  }
  return soa;
}

} // namespace titan_pbctopo

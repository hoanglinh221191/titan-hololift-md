#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

bool update_cut_candidate_with_edge_distance(
    PbctopoCutCandidate &candidate, double d2, double ref_d2, double inv_ref_d2,
    double break_cutoff_d2, const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  if (d2 > break_cutoff_d2)
    ++candidate.broken_edges;
  (void)inv_ref_d2;
  candidate.edge_stretch +=
      pbctopo_symmetric_reference_strain(d2, ref_d2);
  return edge_prune_threshold == nullptr ||
         !edge_scan_cannot_beat_threshold(candidate, edge_prune_threshold,
                                          phase);
}

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
TITAN_TARGET_AVX2
bool accumulate_edge_scan_avx2(
    PbctopoCutCandidate &candidate, const PbctopoEdgeScanSoA &edge_scan_soa,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  if (placed_points.empty())
    return true;

  const auto process_block = [&](const size_t *edge_indices,
                                 size_t block_count) -> bool {
    alignas(32) long long atom_a_lane[4]{};
    alignas(32) long long atom_b_lane[4]{};
    alignas(32) double ref_lane[4]{};
    alignas(32) double inv_ref_lane[4]{};
    alignas(32) double cutoff_lane[4]{};
    for (size_t lane = 0; lane < block_count; ++lane) {
      const size_t edge_idx = edge_indices[lane];
      atom_a_lane[lane] = static_cast<long long>(edge_scan_soa.a[edge_idx]);
      atom_b_lane[lane] = static_cast<long long>(edge_scan_soa.b[edge_idx]);
      ref_lane[lane] = edge_scan_soa.ref_d2[edge_idx];
      inv_ref_lane[lane] = edge_scan_soa.inv_ref_d2[edge_idx];
      cutoff_lane[lane] = edge_scan_soa.break_cutoff_d2[edge_idx];
    }
    const auto &a0 = placed_points[static_cast<size_t>(atom_a_lane[0])];
    const auto &a1 = placed_points[static_cast<size_t>(atom_a_lane[1])];
    const auto &a2 = placed_points[static_cast<size_t>(atom_a_lane[2])];
    const auto &a3 = placed_points[static_cast<size_t>(atom_a_lane[3])];
    const auto &b0 = placed_points[static_cast<size_t>(atom_b_lane[0])];
    const auto &b1 = placed_points[static_cast<size_t>(atom_b_lane[1])];
    const auto &b2 = placed_points[static_cast<size_t>(atom_b_lane[2])];
    const auto &b3 = placed_points[static_cast<size_t>(atom_b_lane[3])];
    const __m256d ax = _mm256_set_pd(a3.x, a2.x, a1.x, a0.x);
    const __m256d ay = _mm256_set_pd(a3.y, a2.y, a1.y, a0.y);
    const __m256d az = _mm256_set_pd(a3.z, a2.z, a1.z, a0.z);
    const __m256d bx = _mm256_set_pd(b3.x, b2.x, b1.x, b0.x);
    const __m256d by = _mm256_set_pd(b3.y, b2.y, b1.y, b0.y);
    const __m256d bz = _mm256_set_pd(b3.z, b2.z, b1.z, b0.z);
    const __m256d dx = _mm256_sub_pd(ax, bx);
    const __m256d dy = _mm256_sub_pd(ay, by);
    const __m256d dz = _mm256_sub_pd(az, bz);
    const __m256d d2 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
        _mm256_mul_pd(dz, dz));
    alignas(32) double d2_lane[4];
    _mm256_store_pd(d2_lane, d2);

    for (size_t lane = 0; lane < block_count; ++lane) {
      if (!update_cut_candidate_with_edge_distance(
              candidate, d2_lane[lane], ref_lane[lane], inv_ref_lane[lane],
              cutoff_lane[lane], edge_prune_threshold, phase)) {
        return false;
      }
    }
    return true;
  };

  if (edge_context != nullptr) {
    size_t block[4]{};
    size_t block_count = 0;
    for (size_t edge_idx : edge_context->variable_edge_indices) {
      if (edge_idx >= edge_scan_soa.a.size())
        continue;
      block[block_count++] = edge_idx;
      if (block_count == 4) {
        if (!process_block(block, block_count))
          return false;
        block_count = 0;
      }
    }
    if (block_count != 0 && !process_block(block, block_count))
      return false;
    return true;
  }

  size_t edge_idx = 0;
  size_t block[4]{};
  while (edge_idx + 4 <= edge_scan_soa.a.size()) {
    block[0] = edge_idx + 0;
    block[1] = edge_idx + 1;
    block[2] = edge_idx + 2;
    block[3] = edge_idx + 3;
    if (!process_block(block, 4))
      return false;
    edge_idx += 4;
  }
  for (; edge_idx < edge_scan_soa.a.size(); ++edge_idx) {
    const size_t atom_a = edge_scan_soa.a[edge_idx];
    const size_t atom_b = edge_scan_soa.b[edge_idx];
    const Vec3 delta = placed_points[atom_a] - placed_points[atom_b];
    const double d2 = norm2(delta);
    if (!update_cut_candidate_with_edge_distance(
            candidate, d2, edge_scan_soa.ref_d2[edge_idx],
            edge_scan_soa.inv_ref_d2[edge_idx],
            edge_scan_soa.break_cutoff_d2[edge_idx], edge_prune_threshold,
            phase)) {
      return false;
    }
  }
  return true;
}
#endif

bool accumulate_edge_scan(
    PbctopoCutCandidate &candidate, const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    PbctopoRefineScorePhase phase) {
  const bool use_soa =
      edge_scan_soa != nullptr && edge_scan_soa->a.size() == local_edges.size() &&
      edge_scan_soa->b.size() == local_edges.size() &&
      edge_scan_soa->ref_d2.size() == local_edges.size() &&
      edge_scan_soa->inv_ref_d2.size() == local_edges.size() &&
      edge_scan_soa->break_cutoff_d2.size() == local_edges.size();

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  const size_t selected_edge_count =
      edge_context != nullptr ? edge_context->variable_edge_indices.size()
                              : local_edges.size();
  const bool simd_enabled =
      use_soa && titan_simd_dispatch::runtime_has_avx2() &&
      selected_edge_count >= 16;
  if (simd_enabled) {
    return accumulate_edge_scan_avx2(candidate, *edge_scan_soa, placed_points,
                                     edge_context, edge_prune_threshold,
                                     phase);
  }
#endif

  const auto accumulate_edge = [&](size_t edge_idx) {
    size_t atom_a = 0;
    size_t atom_b = 0;
    double ref_d2 = 0.0;
    double inv_ref_d2 = 0.0;
    double break_cutoff_d2 = 0.0;
    if (use_soa) {
      atom_a = edge_scan_soa->a[edge_idx];
      atom_b = edge_scan_soa->b[edge_idx];
      ref_d2 = edge_scan_soa->ref_d2[edge_idx];
      inv_ref_d2 = edge_scan_soa->inv_ref_d2[edge_idx];
      break_cutoff_d2 = edge_scan_soa->break_cutoff_d2[edge_idx];
    } else {
      const auto &edge = local_edges[edge_idx];
      atom_a = edge.a;
      atom_b = edge.b;
      ref_d2 = edge.ref_d2;
      inv_ref_d2 = edge.inv_ref_d2;
      break_cutoff_d2 = edge.break_cutoff_d2;
    }
    const Vec3 delta = placed_points[atom_a] - placed_points[atom_b];
    const double d2 = norm2(delta);
    return update_cut_candidate_with_edge_distance(
        candidate, d2, ref_d2, inv_ref_d2, break_cutoff_d2,
        edge_prune_threshold, phase);
  };

  if (edge_context != nullptr) {
    for (size_t edge_idx : edge_context->variable_edge_indices) {
      if (edge_idx >= local_edges.size())
        continue;
      if (!accumulate_edge(edge_idx))
        return false;
    }
    return true;
  }

  for (size_t edge_idx = 0; edge_idx < local_edges.size(); ++edge_idx) {
    if (!accumulate_edge(edge_idx))
      return false;
  }
  return true;
}

bool edge_scan_cannot_beat_threshold(const PbctopoCutCandidate &partial,
                                     const PbctopoCutCandidate *threshold,
                                     PbctopoRefineScorePhase phase) {
  if (threshold == nullptr || !threshold->valid)
    return false;
  const int prefix_cmp =
      compare_candidate_prefix_before_edge_stretch(partial, *threshold, phase);
  if (prefix_cmp > 0)
    return true;
  if (prefix_cmp < 0)
    return false;
  return partial.edge_stretch > threshold->edge_stretch + 1.0e-6;
}

} // namespace pbctopo_solve

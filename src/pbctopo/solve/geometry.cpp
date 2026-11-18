#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

PbctopoRuntimeCutoffs resolve_pbctopo_cutoffs(const arguments &Arguments) {
  return titan_pbctopo::resolve_runtime_cutoffs(Arguments);
}

PbctopoAnchorBackend resolve_pbctopo_anchor_backend(const arguments &Arguments) {
  return titan_pbctopo::resolve_anchor_backend(Arguments);
}

const char *pbctopo_anchor_backend_name(PbctopoAnchorBackend backend) {
  return titan_pbctopo::anchor_backend_name(backend);
}

PbctopoSolveMode resolve_pbctopo_solve_mode(const arguments &Arguments) {
  return titan_pbctopo::resolve_solve_mode(Arguments);
}

const char *pbctopo_solve_mode_name(PbctopoSolveMode mode) {
  return titan_pbctopo::solve_mode_name(mode);
}

PbctopoTranslationCache build_virtual_box_translation_cache(box Box) {
  PbctopoTranslationCache cache;
  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  for (int ix = -2; ix <= 2; ++ix) {
    for (int iy = -2; iy <= 2; ++iy) {
      for (int iz = -2; iz <= 2; ++iz) {
        cache.by_shift[ix + 2][iy + 2][iz + 2] = {
            ix * v1[0] + iy * v2[0] + iz * v3[0],
            ix * v1[1] + iy * v2[1] + iz * v3[1],
            ix * v1[2] + iy * v2[2] + iz * v3[2]};
      }
    }
  }
  return cache;
}

vector<pair<int, int>>
build_contacting_owner_pairs(const vector<PbctopoLocalEdge> &local_edges,
                             const vector<PbctopoAtom> &atoms, box Box,
                             double contact_cutoff,
                             bool hard_required_only) {
  vector<pair<int, int>> owner_pairs;
  unordered_set<uint64_t> seen;
  seen.reserve(local_edges.size());
  auto register_owner_pair = [&](int owner_a, int owner_b) {
    if (owner_a == owner_b)
      return false;
    const int lo = min(owner_a, owner_b);
    const int hi = max(owner_a, owner_b);
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(lo)) << 32) |
        static_cast<uint32_t>(hi);
    if (!seen.insert(key).second)
      return false;
    owner_pairs.emplace_back(lo, hi);
    return true;
  };
  for (const auto &edge : local_edges) {
    if (edge.a >= atoms.size() || edge.b >= atoms.size())
      continue;
    if (hard_required_only &&
        edge.kind != PbctopoLocalEdgeKind::TopologyBond) {
      continue;
    }
    register_owner_pair(atoms[edge.a].owner, atoms[edge.b].owner);
  }

  if (hard_required_only)
    return owner_pairs;

  if (atoms.size() < 2 || contact_cutoff <= kDistanceEps)
    return owner_pairs;

  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid())
    return owner_pairs;
  const BoxMatrix box_matrix(Box);
  const double cutoff2 = contact_cutoff * contact_cutoff;
  const double sx_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][0] * box_matrix.inv[0][0] +
                                     box_matrix.inv[1][0] * box_matrix.inv[1][0] +
                                     box_matrix.inv[2][0] * box_matrix.inv[2][0]) +
                    1.0e-9);
  const double sy_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][1] * box_matrix.inv[0][1] +
                                     box_matrix.inv[1][1] * box_matrix.inv[1][1] +
                                     box_matrix.inv[2][1] * box_matrix.inv[2][1]) +
                    1.0e-9);
  const double sz_bound =
      min(0.5, contact_cutoff * sqrt(box_matrix.inv[0][2] * box_matrix.inv[0][2] +
                                     box_matrix.inv[1][2] * box_matrix.inv[1][2] +
                                     box_matrix.inv[2][2] * box_matrix.inv[2][2]) +
                    1.0e-9);
  const int nx = max(1, static_cast<int>(floor(1.0 / max(sx_bound, 1.0e-6))));
  const int ny = max(1, static_cast<int>(floor(1.0 / max(sy_bound, 1.0e-6))));
  const int nz = max(1, static_cast<int>(floor(1.0 / max(sz_bound, 1.0e-6))));
  const double cell_sx = 1.0 / static_cast<double>(nx);
  const double cell_sy = 1.0 / static_cast<double>(ny);
  const double cell_sz = 1.0 / static_cast<double>(nz);
  auto wrap_index = [](int value, int mod) {
    int wrapped = value % mod;
    if (wrapped < 0)
      wrapped += mod;
    return wrapped;
  };

  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> contact_grid;
  contact_grid.reserve(max<size_t>(64, atoms.size() / 2));
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto &atom = atoms[atom_idx];
    if (!atom.is_heavy)
      continue;
    const int cx = min(nx - 1, static_cast<int>(floor(atom.sx / cell_sx)));
    const int cy = min(ny - 1, static_cast<int>(floor(atom.sy / cell_sy)));
    const int cz = min(nz - 1, static_cast<int>(floor(atom.sz / cell_sz)));
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const ClashCellKey key{wrap_index(cx + dx, nx), wrap_index(cy + dy, ny),
                                 wrap_index(cz + dz, nz)};
          const auto it = contact_grid.find(key);
          if (it == contact_grid.end())
            continue;
          for (size_t other_idx : it->second) {
            const auto &other = atoms[other_idx];
            if (atom.owner == other.owner)
              continue;
            const auto mic = lattice.nearest_image_cartesian(
                other.x - atom.x, other.y - atom.y, other.z - atom.z);
            const double d2 = mic.d2;
            if (d2 > cutoff2)
              continue;
            register_owner_pair(atom.owner, other.owner);
          }
        }
      }
    }
    contact_grid[{cx, cy, cz}].push_back(atom_idx);
  }
  return owner_pairs;
}

unordered_map<int, Vec3>
compute_owner_com_cartesian(const vector<Vec3> &placed_points,
                            const vector<PbctopoAtom> &atoms) {
  unordered_map<int, Vec3> owner_sums;
  unordered_map<int, size_t> owner_counts;
  owner_sums.reserve(8);
  owner_counts.reserve(8);
  const size_t atom_count = min(placed_points.size(), atoms.size());
  for (size_t atom_idx = 0; atom_idx < atom_count; ++atom_idx) {
    const int owner = atoms[atom_idx].owner;
    Vec3 &sum = owner_sums[owner];
    sum.x += placed_points[atom_idx].x;
    sum.y += placed_points[atom_idx].y;
    sum.z += placed_points[atom_idx].z;
    ++owner_counts[owner];
  }
  for (auto &[owner, sum] : owner_sums) {
    const size_t count = owner_counts[owner];
    if (count == 0)
      continue;
    const double inv_count = 1.0 / static_cast<double>(count);
    sum.x *= inv_count;
    sum.y *= inv_count;
    sum.z *= inv_count;
  }
  return owner_sums;
}

unordered_map<int, Vec3>
compute_owner_com_scaled(box Box,
                         const unordered_map<int, Vec3> &owner_com_cartesian) {
  unordered_map<int, Vec3> owner_com_scaled;
  owner_com_scaled.reserve(owner_com_cartesian.size());
  const PbctopoLatticeMetric lattice(Box);
  for (const auto &[owner, point] : owner_com_cartesian) {
    const auto scaled = lattice.cartesian_to_scaled(point.x, point.y, point.z);
    owner_com_scaled[owner] = {scaled[0], scaled[1], scaled[2]};
  }
  return owner_com_scaled;
}

bool candidate_has_split_contacting_owners(
    const vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs, size_t &split_owner_pairs,
    double &max_owner_raw_minus_minimg_delta,
    PbctopoOwnerPairSplitDetail *detail) {
  split_owner_pairs = 0;
  max_owner_raw_minus_minimg_delta = 0.0;
  if (detail != nullptr)
    *detail = {};
  if (contacting_owner_pairs.empty() || placed_points.empty() || atoms.empty())
    return false;

  const auto owner_com_cartesian = compute_owner_com_cartesian(placed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid())
    return false;
  const double box_scale = max(lattice.shortest_basis_length(), 1.0e-12);

  for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
    const auto it_a = owner_com_scaled.find(owner_a);
    const auto it_b = owner_com_scaled.find(owner_b);
    if (it_a == owner_com_scaled.end() || it_b == owner_com_scaled.end())
      continue;

    const Vec3 raw_delta_scaled{it_b->second.x - it_a->second.x,
                                it_b->second.y - it_a->second.y,
                                it_b->second.z - it_a->second.z};
    const auto mic = lattice.nearest_image_scaled(
        raw_delta_scaled.x, raw_delta_scaled.y, raw_delta_scaled.z);
    if (!mic.valid)
      continue;
    const Vec3 minimg_delta_scaled{mic.sx, mic.sy, mic.sz};
    const Vec3 split_delta{raw_delta_scaled.x - minimg_delta_scaled.x,
                           raw_delta_scaled.y - minimg_delta_scaled.y,
                           raw_delta_scaled.z - minimg_delta_scaled.z};
    const double axis_shift_max =
        max({fabs(split_delta.x), fabs(split_delta.y), fabs(split_delta.z)});
    const auto raw_cart = lattice.scaled_to_cartesian(
        raw_delta_scaled.x, raw_delta_scaled.y, raw_delta_scaled.z);
    const double raw_norm_A = sqrt(raw_cart[0] * raw_cart[0] +
                                   raw_cart[1] * raw_cart[1] +
                                   raw_cart[2] * raw_cart[2]);
    const double minimg_norm_A = sqrt(max(0.0, mic.d2));
    const double norm_gap = (raw_norm_A - minimg_norm_A) / box_scale;
    if (axis_shift_max <= 0.5 || norm_gap <= 0.20)
      continue;

    ++split_owner_pairs;
    if (norm_gap > max_owner_raw_minus_minimg_delta) {
      max_owner_raw_minus_minimg_delta = norm_gap;
      if (detail != nullptr) {
        detail->owner_a = owner_a;
        detail->owner_b = owner_b;
        detail->raw_delta_scaled = raw_delta_scaled;
        detail->minimg_delta_scaled = minimg_delta_scaled;
        detail->raw_norm = raw_norm_A;
        detail->minimg_norm = minimg_norm_A;
        detail->valid = true;
      }
    }
  }

  return split_owner_pairs > 0;
}

bool reimage_contacting_owner_points_minimum_image(
    vector<Vec3> &placed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs) {
  if (contacting_owner_pairs.empty() || placed_points.size() != atoms.size())
    return false;

  const auto owner_com_cartesian = compute_owner_com_cartesian(placed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  const PbctopoLatticeMetric lattice(Box);
  if (!lattice.valid() || owner_com_scaled.size() < 2)
    return false;

  unordered_map<int, vector<int>> adjacency;
  adjacency.reserve(owner_com_scaled.size());
  for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
    if (owner_com_scaled.find(owner_a) == owner_com_scaled.end() ||
        owner_com_scaled.find(owner_b) == owner_com_scaled.end()) {
      continue;
    }
    adjacency[owner_a].push_back(owner_b);
    adjacency[owner_b].push_back(owner_a);
  }
  if (adjacency.empty())
    return false;

  vector<int> owners;
  owners.reserve(owner_com_scaled.size());
  for (const auto &[owner, _] : owner_com_scaled)
    owners.push_back(owner);
  sort(owners.begin(), owners.end());

  unordered_map<int, Int3> owner_shifts;
  owner_shifts.reserve(owner_com_scaled.size());
  queue<int> pending;
  for (int root_owner : owners) {
    if (owner_shifts.find(root_owner) != owner_shifts.end())
      continue;
    owner_shifts[root_owner] = {0, 0, 0};
    pending.push(root_owner);
    while (!pending.empty()) {
      const int owner_u = pending.front();
      pending.pop();
      const auto adj_it = adjacency.find(owner_u);
      if (adj_it == adjacency.end())
        continue;
      const Vec3 &com_u = owner_com_scaled.at(owner_u);
      const Int3 shift_u = owner_shifts.at(owner_u);
      for (int owner_v : adj_it->second) {
        const Vec3 &com_v = owner_com_scaled.at(owner_v);
        const auto mic = lattice.nearest_image_scaled(
            com_v.x - com_u.x, com_v.y - com_u.y, com_v.z - com_u.z);
        if (!mic.valid)
          return false;
        Int3 proposed_shift{};
        if (!checked_int3_add(
                shift_u, Int3{mic.image_x, mic.image_y, mic.image_z},
                proposed_shift)) {
          return false;
        }
        const auto [shift_it, inserted] =
            owner_shifts.emplace(owner_v, proposed_shift);
        if (inserted)
          pending.push(owner_v);
      }
    }
  }

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  bool changed = false;
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const auto shift_it = owner_shifts.find(atoms[atom_idx].owner);
    if (shift_it == owner_shifts.end())
      continue;
    const Int3 &shift = shift_it->second;
    if (shift.x == 0 && shift.y == 0 && shift.z == 0)
      continue;
    placed_points[atom_idx].x += shift.x * v1[0] + shift.y * v2[0] + shift.z * v3[0];
    placed_points[atom_idx].y += shift.x * v1[1] + shift.y * v2[1] + shift.z * v3[1];
    placed_points[atom_idx].z += shift.x * v1[2] + shift.y * v2[2] + shift.z * v3[2];
    changed = true;
  }
  return changed;
}

void build_owner_spans(const vector<PbctopoAtom> &atoms, int owner,
                       vector<PbctopoHullSpan> &spans) {
  spans.clear();
  size_t span_begin = numeric_limits<size_t>::max();
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (atoms[atom_idx].owner == owner) {
      if (span_begin == numeric_limits<size_t>::max())
        span_begin = atom_idx;
      continue;
    }
    if (span_begin != numeric_limits<size_t>::max()) {
      spans.push_back({span_begin, atom_idx});
      span_begin = numeric_limits<size_t>::max();
    }
  }
  if (span_begin != numeric_limits<size_t>::max())
    spans.push_back({span_begin, atoms.size()});
}

bool rescue_split_contacting_owner_candidate(
    const vector<Vec3> &seed_points, const vector<PbctopoAtom> &atoms, box Box,
    const vector<pair<int, int>> &contacting_owner_pairs,
    const vector<PbctopoShift> &shifts,
    const PbctopoOwnerPairSplitDetail &split_detail, vector<Vec3> &rescued_points,
    Int3 &applied_shift, double &best_raw_delta_norm) {
  rescued_points.clear();
  applied_shift = {};
  best_raw_delta_norm = numeric_limits<double>::infinity();
  if (!split_detail.valid || seed_points.size() != atoms.size() ||
      contacting_owner_pairs.empty() || shifts.empty()) {
    return false;
  }

  const auto owner_com_cartesian = compute_owner_com_cartesian(seed_points, atoms);
  const auto owner_com_scaled = compute_owner_com_scaled(Box, owner_com_cartesian);
  size_t best_shift_idx = numeric_limits<size_t>::max();
  double best_score = numeric_limits<double>::infinity();
  int owner_to_move = -1;
  vector<int> move_candidates;
  move_candidates.reserve(2);
  move_candidates.push_back(split_detail.owner_b);
  if (split_detail.owner_a != split_detail.owner_b)
    move_candidates.push_back(split_detail.owner_a);

  for (int move_owner : move_candidates) {
    const int ref_owner =
        move_owner == split_detail.owner_b ? split_detail.owner_a
                                           : split_detail.owner_b;
    const auto ref_it = owner_com_scaled.find(ref_owner);
    const auto move_it = owner_com_scaled.find(move_owner);
    if (ref_it == owner_com_scaled.end() || move_it == owner_com_scaled.end())
      continue;

    const Vec3 &ref_com = ref_it->second;
    const Vec3 &move_com = move_it->second;
    for (size_t shift_idx = 0; shift_idx < shifts.size(); ++shift_idx) {
      const auto &shift = shifts[shift_idx];
      const Vec3 shifted_move_scaled{move_com.x + static_cast<double>(shift.ix),
                                     move_com.y + static_cast<double>(shift.iy),
                                     move_com.z + static_cast<double>(shift.iz)};
      const Vec3 raw_delta_scaled{shifted_move_scaled.x - ref_com.x,
                                  shifted_move_scaled.y - ref_com.y,
                                  shifted_move_scaled.z - ref_com.z};
      const double raw_norm = sqrt(norm2(raw_delta_scaled));
      double score = raw_norm;
      for (const auto &[owner_a, owner_b] : contacting_owner_pairs) {
        if (owner_a != move_owner && owner_b != move_owner)
          continue;
        const int other_owner = (owner_a == move_owner) ? owner_b : owner_a;
        const auto other_it = owner_com_scaled.find(other_owner);
        if (other_it == owner_com_scaled.end())
          continue;
        Vec3 pair_raw_delta_scaled{};
        if (owner_a == move_owner) {
          pair_raw_delta_scaled = {
              other_it->second.x - shifted_move_scaled.x,
              other_it->second.y - shifted_move_scaled.y,
              other_it->second.z - shifted_move_scaled.z};
        } else {
          pair_raw_delta_scaled = {
              shifted_move_scaled.x - other_it->second.x,
              shifted_move_scaled.y - other_it->second.y,
              shifted_move_scaled.z - other_it->second.z};
        }
        score += 0.25 * sqrt(norm2(pair_raw_delta_scaled));
      }
      if (score + 1.0e-9 < best_score) {
        best_score = score;
        best_raw_delta_norm = raw_norm;
        best_shift_idx = shift_idx;
        applied_shift = {shift.ix, shift.iy, shift.iz};
        owner_to_move = move_owner;
      }
    }
  }

  if (best_shift_idx == numeric_limits<size_t>::max() || owner_to_move < 0)
    return false;
  if (applied_shift.x == 0 && applied_shift.y == 0 && applied_shift.z == 0)
    return false;

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  const Vec3 translation{
      applied_shift.x * v1[0] + applied_shift.y * v2[0] + applied_shift.z * v3[0],
      applied_shift.x * v1[1] + applied_shift.y * v2[1] + applied_shift.z * v3[1],
      applied_shift.x * v1[2] + applied_shift.y * v2[2] + applied_shift.z * v3[2]};

  rescued_points = seed_points;
  for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    if (atoms[atom_idx].owner != owner_to_move)
      continue;
    rescued_points[atom_idx].x += translation.x;
    rescued_points[atom_idx].y += translation.y;
    rescued_points[atom_idx].z += translation.z;
  }
  return true;
}

bool continuity_metrics_ok(
    const vector<Vec3> &placed_points,
    const vector<PbctopoContinuityPair> &continuity_pairs, double hard_cutoff,
    double &path_sum, double &max_d2, double &min_hard_margin_A,
    size_t &margin_atom_a, size_t &margin_atom_b,
    const PbctopoCutCandidate *incumbent) {
  path_sum = 0.0;
  max_d2 = 0.0;
  min_hard_margin_A = numeric_limits<double>::quiet_NaN();
  margin_atom_a = numeric_limits<size_t>::max();
  margin_atom_b = numeric_limits<size_t>::max();
  if (continuity_pairs.empty())
    return true;

  const double hard_cutoff2 = hard_cutoff * hard_cutoff;
  bool has_hard_margin = false;
  const bool has_incumbent =
      (incumbent != nullptr && incumbent->valid &&
       isfinite(incumbent->continuity_path) &&
       isfinite(incumbent->continuity_max_d2));

  auto consume_pair_metric = [&](double d2,
                                 const PbctopoContinuityPair &pair) -> bool {
    if (pbctopo_continuity_is_hard(pair.kind)) {
      if (d2 > max_d2)
        max_d2 = d2;
      const double margin_A = pbctopo_continuity_hard_margin_A(
          pair, d2, hard_cutoff2);
      if (!has_hard_margin || margin_A < min_hard_margin_A) {
        has_hard_margin = true;
        min_hard_margin_A = margin_A;
        margin_atom_a = pair.a;
        margin_atom_b = pair.b;
      }
      if (pbctopo_continuity_hard_violation(pair, d2, hard_cutoff2))
        return false;
    }
    path_sum += pair.weight *
                pbctopo_symmetric_reference_strain(d2, pair.ref_d2);
    if (has_incumbent &&
        path_sum > incumbent->continuity_path + 1.0e-6) {
      return false;
    }
    return true;
  };

#if TITAN_SIMD_X86 && TITAN_RUNTIME_TARGET_ATTRS
  if (titan_simd_dispatch::runtime_has_avx2() &&
      continuity_pairs.size() >= 8) {
    alignas(32) double ax[4];
    alignas(32) double ay[4];
    alignas(32) double az[4];
    alignas(32) double bx[4];
    alignas(32) double by[4];
    alignas(32) double bz[4];
    alignas(32) double d2_lane[4];
    size_t pair_idx = 0;
    for (; pair_idx + 4 <= continuity_pairs.size(); pair_idx += 4) {
      for (size_t lane = 0; lane < 4; ++lane) {
        const auto &pair = continuity_pairs[pair_idx + lane];
        const Vec3 &point_a = placed_points[pair.a];
        const Vec3 &point_b = placed_points[pair.b];
        ax[lane] = point_a.x;
        ay[lane] = point_a.y;
        az[lane] = point_a.z;
        bx[lane] = point_b.x;
        by[lane] = point_b.y;
        bz[lane] = point_b.z;
      }

      const __m256d dx =
          _mm256_sub_pd(_mm256_load_pd(ax), _mm256_load_pd(bx));
      const __m256d dy =
          _mm256_sub_pd(_mm256_load_pd(ay), _mm256_load_pd(by));
      const __m256d dz =
          _mm256_sub_pd(_mm256_load_pd(az), _mm256_load_pd(bz));
      const __m256d d2 = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      _mm256_store_pd(d2_lane, d2);

      for (size_t lane = 0; lane < 4; ++lane) {
        if (!consume_pair_metric(d2_lane[lane],
                                 continuity_pairs[pair_idx + lane]))
          return false;
      }
    }
    for (; pair_idx < continuity_pairs.size(); ++pair_idx) {
      const auto &pair = continuity_pairs[pair_idx];
      const Vec3 delta = placed_points[pair.a] - placed_points[pair.b];
      const double d2 = norm2(delta);
      if (!consume_pair_metric(d2, pair))
        return false;
    }
    return true;
  }
#endif

  for (const auto &pair : continuity_pairs) {
    const Vec3 delta = placed_points[pair.a] - placed_points[pair.b];
    const double d2 = norm2(delta);
    if (!consume_pair_metric(d2, pair))
      return false;
  }
  return true;
}

vector<Int3> extract_component_offsets_from_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    box Box, const vector<Vec3> &placed_points) {
  vector<Int3> offsets;
  if (placed_points.size() != atoms.size())
    return offsets;
  offsets.resize(components.size());
  const PbctopoLatticeMetric lattice(Box);
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    if (components[comp_idx].empty())
      continue;
    const size_t atom_idx = components[comp_idx].front();
    const double dx = placed_points[atom_idx].x - atoms[atom_idx].x;
    const double dy = placed_points[atom_idx].y - atoms[atom_idx].y;
    const double dz = placed_points[atom_idx].z - atoms[atom_idx].z;
    const auto scaled = lattice.cartesian_to_scaled(dx, dy, dz);
    Int3 offset{};
    if (!titan_pbctopo::pbctopo_checked_round_image(scaled[0], offset.x) ||
        !titan_pbctopo::pbctopo_checked_round_image(scaled[1], offset.y) ||
        !titan_pbctopo::pbctopo_checked_round_image(scaled[2], offset.z)) {
      return {};
    }
    offsets[comp_idx] = offset;
  }
  if (!rebase_component_offsets(offsets, Box))
    return {};
  return offsets;
}

// Offsets from extract_component_offsets_from_points are rebased to the
// closest-lattice-mean gauge, but placed points keep the gauge that placed
// them. Shift every offset by the one lattice vector that puts the first atom
// of the first non-empty component on its placed point. A common shift only
// changes the gauge: the relative assignment, and every other atom's check
// against it, stay as they were.
bool align_component_offsets_to_points(
    const vector<PbctopoAtom> &atoms, const vector<vector<size_t>> &components,
    const PbctopoLatticeMetric &lattice, const vector<Vec3> &placed_points,
    vector<Int3> &offsets) {
  if (offsets.size() != components.size())
    return false;
  size_t reference_component = components.size();
  for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
    if (!components[comp_idx].empty()) {
      reference_component = comp_idx;
      break;
    }
  }
  if (reference_component == components.size())
    return false;
  const size_t reference_atom = components[reference_component].front();
  if (reference_atom >= atoms.size() || reference_atom >= placed_points.size())
    return false;
  const auto absolute_scaled = lattice.cartesian_to_scaled(
      placed_points[reference_atom].x - atoms[reference_atom].x,
      placed_points[reference_atom].y - atoms[reference_atom].y,
      placed_points[reference_atom].z - atoms[reference_atom].z);
  Int3 absolute_reference_offset{};
  if (!titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[0], absolute_reference_offset.x) ||
      !titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[1], absolute_reference_offset.y) ||
      !titan_pbctopo::pbctopo_checked_round_image(
          absolute_scaled[2], absolute_reference_offset.z)) {
    return false;
  }
  Int3 global_offset_correction{};
  if (!titan_pbctopo::checked_int3_subtract(absolute_reference_offset,
                                            offsets[reference_component],
                                            global_offset_correction)) {
    return false;
  }
  for (Int3 &offset : offsets) {
    Int3 corrected{};
    if (!titan_pbctopo::checked_int3_add(offset, global_offset_correction,
                                         corrected)) {
      return false;
    }
    offset = corrected;
  }
  return true;
}

} // namespace pbctopo_solve

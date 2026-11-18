#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

[[maybe_unused]] vector<PbctopoCutOption>
collect_cut_options(const vector<PbctopoAtom> &atoms, char axis,
                    size_t max_options) {
  vector<double> coords;
  coords.reserve(atoms.size() * 2 + 2);
  coords.push_back(0.5);
  coords.push_back(2.5);
  for (const auto &atom : atoms) {
    const double coord =
        (axis == 'x') ? atom.sx : ((axis == 'y') ? atom.sy : atom.sz);
    coords.push_back(coord + 0.5);
    coords.push_back(coord + 1.5);
  }
  if (coords.empty())
    return {{1.5, 2.0}};

  sort(coords.begin(), coords.end());
  vector<PbctopoCutOption> options;
  options.reserve(coords.size());
  for (size_t i = 0; i + 1 < coords.size(); ++i) {
    const double current = coords[i];
    const double next = coords[i + 1];
    const double gap = next - current;
    if (gap <= 1.0e-9)
      continue;
    const double center = 0.5 * (current + next);
    if (center + 1.0e-9 < 0.5 || center - 1.0e-9 > 2.5)
      continue;
    options.push_back({center, gap});
  }

  sort(options.begin(), options.end(),
       [](const PbctopoCutOption &lhs, const PbctopoCutOption &rhs) {
          const int metric_cmp =
              compare_pbctopo_double_metric(lhs.gap, rhs.gap, 1.0e-9);
          if (metric_cmp != 0)
            return metric_cmp > 0;
         return lhs.cut < rhs.cut;
       });

  vector<PbctopoCutOption> best;
  best.reserve(min(max_options, options.size() + 1));
  auto push_unique = [&](double cut, double gap) {
    for (const auto &item : best) {
      if (fabs(item.cut - cut) <= 1.0e-6)
        return;
    }
    best.push_back({cut, gap});
  };

  for (const auto &option : options) {
    push_unique(option.cut, option.gap);
    if (best.size() >= max_options)
      break;
  }
  push_unique(1.5, 0.0);
  return best;
}

[[maybe_unused]] void
build_cut_positions(const vector<PbctopoAtom> &atoms, box Box, double cut_x,
                    double cut_y, double cut_z, vector<Vec3> &placed_points,
                    PointStats &stats, double &shift_norm2) {
  auto select_supercell_coord = [](double wrapped_coord, double center,
                                   double &selected_coord, int &selected_shift) {
    const double lo = center - 0.5;
    const double hi = center + 0.5;
    constexpr double kSelectEps = 1.0e-9;
    for (int copy_idx = 0; copy_idx < 3; ++copy_idx) {
      const double candidate = wrapped_coord + static_cast<double>(copy_idx);
      if (candidate + kSelectEps < lo)
        continue;
      if (candidate >= hi - kSelectEps)
        continue;
      selected_coord = candidate;
      selected_shift = copy_idx - 1;
      return;
    }

    const int copy_idx =
        max(0, min(2, static_cast<int>(ceil(lo - wrapped_coord - kSelectEps))));
    selected_coord = wrapped_coord + static_cast<double>(copy_idx);
    selected_shift = copy_idx - 1;
  };

  placed_points.resize(atoms.size());
  stats = PointStats{};
  shift_norm2 = 0.0;
  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    int shift_x = 0;
    int shift_y = 0;
    int shift_z = 0;
    select_supercell_coord(atom.sx, cut_x, sx, shift_x);
    select_supercell_coord(atom.sy, cut_y, sy, shift_y);
    select_supercell_coord(atom.sz, cut_z, sz, shift_z);
    Vec3 point{};
    scaled_to_cartesian(Box, sx, sy, sz, point.x, point.y, point.z);
    placed_points[idx] = point;
    stats.add(point.x, point.y, point.z);

    shift_norm2 += static_cast<double>(shift_x * shift_x + shift_y * shift_y +
                                       shift_z * shift_z);
  }
}

PbctopoCutCandidate score_cut_candidate(const vector<PbctopoLocalEdge> &local_edges,
                                        const vector<Vec3> &placed_points,
                                        double cut_x, double cut_y, double cut_z,
                                        double gap_score, const PointStats &stats,
                                        double shift_norm2,
                                        const PbctopoStage1EdgeContext *edge_context,
                                        const PbctopoEdgeScanSoA *edge_scan_soa,
                                        const PbctopoCutCandidate *edge_prune_threshold,
                                        const PbctopoCutCandidate *pre_edge_seed,
                                        PbctopoRefineScorePhase phase) {
  PbctopoCutCandidate candidate =
      (pre_edge_seed != nullptr) ? *pre_edge_seed : PbctopoCutCandidate{};
  candidate.cut_x = cut_x;
  candidate.cut_y = cut_y;
  candidate.cut_z = cut_z;
  candidate.gap_score = gap_score;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.valid = false;

  if (stats.count == 0 || placed_points.empty())
    return candidate;

  candidate.bbox_volume = compute_bbox_volume(stats);
  candidate.rg2 = compute_rg2_from_stats(stats);
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.shift_norm2 = shift_norm2;

  candidate.broken_edges = (edge_context != nullptr)
                               ? edge_context->fixed_broken_edges
                               : 0;
  candidate.edge_stretch = (edge_context != nullptr)
                               ? edge_context->fixed_edge_stretch
                               : 0.0;

  if (!accumulate_edge_scan(candidate, local_edges, placed_points, edge_context,
                            edge_scan_soa, edge_prune_threshold, phase)) {
    return {};
  }

  candidate.valid = true;
  return candidate;
}

PbctopoCutCandidate score_cut_candidate_principal_only(
    const vector<PbctopoLocalEdge> &local_edges,
    const vector<Vec3> &placed_points, double cut_x, double cut_y, double cut_z,
    double gap_score, const PointStats &stats, double shift_norm2,
    PbctopoCutScoreBreakdown *breakdown,
    const PbctopoStage1EdgeContext *edge_context,
    const PbctopoEdgeScanSoA *edge_scan_soa,
    const PbctopoCutCandidate *edge_prune_threshold,
    const PbctopoCutCandidate *pre_edge_seed) {
  PbctopoCutCandidate candidate =
      (pre_edge_seed != nullptr) ? *pre_edge_seed : PbctopoCutCandidate{};
  candidate.cut_x = cut_x;
  candidate.cut_y = cut_y;
  candidate.cut_z = cut_z;
  candidate.gap_score = gap_score;
  candidate.broken_edges = 0;
  candidate.edge_stretch = 0.0;
  candidate.valid = false;

  if (stats.count == 0 || placed_points.empty())
    return candidate;

  chrono::steady_clock::time_point section_started;
  if (breakdown != nullptr)
    section_started = chrono::steady_clock::now();
  const array<double, 3> principal_rg2 = compute_principal_rg2(stats);
  if (breakdown != nullptr) {
    breakdown->principal_s +=
        chrono::duration<double>(chrono::steady_clock::now() - section_started)
            .count();
    section_started = chrono::steady_clock::now();
  }
  candidate.principal_rg2_max = principal_rg2[0];
  candidate.principal_rg2_mid = principal_rg2[1];
  candidate.principal_rg2_min = principal_rg2[2];
  const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
  candidate.shape_kappa2 = shape_metrics.kappa2;
  candidate.shape_asphericity = shape_metrics.asphericity;
  candidate.shape_acylindricity = shape_metrics.acylindricity;
  candidate.shift_norm2 = shift_norm2;

  candidate.broken_edges = (edge_context != nullptr)
                               ? edge_context->fixed_broken_edges
                               : 0;
  candidate.edge_stretch = (edge_context != nullptr)
                               ? edge_context->fixed_edge_stretch
                               : 0.0;

  if (!accumulate_edge_scan(candidate, local_edges, placed_points, edge_context,
                            edge_scan_soa, edge_prune_threshold,
                            PbctopoRefineScorePhase::CoarseStage1)) {
    if (breakdown != nullptr) {
      breakdown->edge_scan_s +=
          chrono::duration<double>(chrono::steady_clock::now() -
                                   section_started)
              .count();
    }
    return {};
  }
  if (breakdown != nullptr) {
    breakdown->edge_scan_s +=
        chrono::duration<double>(chrono::steady_clock::now() - section_started)
            .count();
  }

  candidate.valid = true;
  return candidate;
}

void analyze_cut_candidate_ultraclash(const vector<Vec3> &placed_points,
                                       const unordered_set<uint64_t> &edge_set,
                                       PbctopoCutCandidate &candidate,
                                       double cutoff,
                                       const vector<PbctopoAtom> *atoms,
                                       const vector<size_t> *atom_component_index) {
  thread_local PbctopoUltraclashScratch scratch;
  candidate.min_pair_d2 = numeric_limits<double>::infinity();
  candidate.clash_atom_a = numeric_limits<size_t>::max();
  candidate.clash_atom_b = numeric_limits<size_t>::max();
  candidate.ultra_clash = false;
  if (!candidate.valid || placed_points.size() < 2)
    return;

  const bool filter_heavy =
      atoms != nullptr && atoms->size() == placed_points.size();
  const double grid_cutoff =
      filter_heavy ? pbctopo_max_clash_cutoff(*atoms, cutoff) : cutoff;
  const double inv_grid = 1.0 / grid_cutoff;
  const double cutoff2 = cutoff * cutoff;
  const bool filter_same_component =
      atom_component_index != nullptr &&
      atom_component_index->size() == placed_points.size() &&
      candidate.broken_edges == 0;
  auto accumulate_pair = [&](size_t atom_idx, size_t other_idx) {
    if (filter_same_component &&
        (*atom_component_index)[atom_idx] ==
            (*atom_component_index)[other_idx])
      return false;
    if (edge_set.find(atom_pair_key(atom_idx, other_idx)) != edge_set.end())
      return false;
    const auto &point = placed_points[atom_idx];
    const auto &other = placed_points[other_idx];
    const double dx_atom = point.x - other.x;
    const double dy_atom = point.y - other.y;
    const double dz_atom = point.z - other.z;
    const double d2 = dx_atom * dx_atom + dy_atom * dy_atom + dz_atom * dz_atom;
    if (d2 < candidate.min_pair_d2) {
      candidate.min_pair_d2 = d2;
      candidate.clash_atom_a = atom_idx;
      candidate.clash_atom_b = other_idx;
    }
    const bool clashes =
        filter_heavy
            ? pbctopo_atoms_clash(d2, (*atoms)[atom_idx],
                                   (*atoms)[other_idx], cutoff)
            : d2 + kDistanceEps < cutoff2;
    if (clashes) {
      candidate.ultra_clash = true;
      return true;
    }
    return false;
  };

  auto analyze_with_hash_grid = [&]() {
    unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> clash_grid;
    clash_grid.reserve(max<size_t>(64, placed_points.size()));
    for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
      if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
        continue;
      const auto &point = placed_points[atom_idx];
      ClashCellKey own_cell;
      if (!titan_pbctopo::pbctopo_make_cell_key(
              point.x, point.y, point.z, inv_grid, own_cell)) {
        candidate.ultra_clash = true;
        return;
      }

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            ClashCellKey neighbor;
            if (!titan_pbctopo::pbctopo_offset_cell_key(
                    own_cell, dx, dy, dz, neighbor)) {
              candidate.ultra_clash = true;
              return;
            }
            const auto it = clash_grid.find(neighbor);
            if (it == clash_grid.end())
              continue;
            for (size_t other_idx : it->second) {
              if (accumulate_pair(atom_idx, other_idx))
                return;
            }
          }
        }
      }

      clash_grid[own_cell].push_back(atom_idx);
    }
  };

  scratch.cell_x.resize(placed_points.size());
  scratch.cell_y.resize(placed_points.size());
  scratch.cell_z.resize(placed_points.size());
  int64_t min_cx = numeric_limits<int64_t>::max();
  int64_t min_cy = numeric_limits<int64_t>::max();
  int64_t min_cz = numeric_limits<int64_t>::max();
  int64_t max_cx = numeric_limits<int64_t>::min();
  int64_t max_cy = numeric_limits<int64_t>::min();
  int64_t max_cz = numeric_limits<int64_t>::min();

  for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
    if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
      continue;
    const auto &point = placed_points[atom_idx];
    ClashCellKey cell;
    if (!titan_pbctopo::pbctopo_make_cell_key(
            point.x, point.y, point.z, inv_grid, cell)) {
      candidate.ultra_clash = true;
      return;
    }
    scratch.cell_x[atom_idx] = cell.x;
    scratch.cell_y[atom_idx] = cell.y;
    scratch.cell_z[atom_idx] = cell.z;
    min_cx = min(min_cx, cell.x);
    min_cy = min(min_cy, cell.y);
    min_cz = min(min_cz, cell.z);
    max_cx = max(max_cx, cell.x);
    max_cy = max(max_cy, cell.y);
    max_cz = max(max_cz, cell.z);
  }
  if (min_cx == numeric_limits<int64_t>::max())
    return;

  const size_t max_cells = kPbctopoDenseClashGridMaxCells;
  const auto bounded_span = [](int64_t lower, int64_t upper, size_t &span) {
    const long double width = static_cast<long double>(upper) -
                              static_cast<long double>(lower) + 1.0L;
    if (!(width >= 1.0L) || width > static_cast<long double>(max_cells))
      return false;
    span = static_cast<size_t>(width);
    return true;
  };
  size_t nx = 0;
  size_t ny = 0;
  size_t nz = 0;
  if (!bounded_span(min_cx, max_cx, nx) ||
      !bounded_span(min_cy, max_cy, ny) ||
      !bounded_span(min_cz, max_cz, nz) || ny > (max_cells / nx) ||
      nz > (max_cells / (nx * ny))) {
    analyze_with_hash_grid();
    return;
  }

  const size_t total_cells = nx * ny * nz;
  scratch.head.resize(total_cells);
  scratch.head_epoch.resize(total_cells, 0u);
  ++scratch.current_epoch;
  if (scratch.current_epoch == 0u) {
    scratch.current_epoch = 1u;
    fill(scratch.head_epoch.begin(), scratch.head_epoch.end(), 0u);
  }
  scratch.next.resize(placed_points.size());
  const size_t plane_stride = nx * ny;
  auto cell_index = [&](int64_t cx, int64_t cy, int64_t cz) {
    return (static_cast<size_t>(static_cast<long double>(cz) - min_cz) *
            plane_stride) +
           (static_cast<size_t>(static_cast<long double>(cy) - min_cy) * nx) +
           static_cast<size_t>(static_cast<long double>(cx) - min_cx);
  };

  for (size_t atom_idx = 0; atom_idx < placed_points.size(); ++atom_idx) {
    if (filter_heavy && !(*atoms)[atom_idx].is_heavy)
      continue;
    const int64_t cx = scratch.cell_x[atom_idx];
    const int64_t cy = scratch.cell_y[atom_idx];
    const int64_t cz = scratch.cell_z[atom_idx];
    for (int64_t nz_cell = max(min_cz, cz - 1);
         nz_cell <= min(max_cz, cz + 1);
         ++nz_cell) {
      for (int64_t ny_cell = max(min_cy, cy - 1);
           ny_cell <= min(max_cy, cy + 1);
           ++ny_cell) {
        for (int64_t nx_cell = max(min_cx, cx - 1);
             nx_cell <= min(max_cx, cx + 1);
             ++nx_cell) {
          const size_t neighbor_cell_idx = cell_index(nx_cell, ny_cell, nz_cell);
          int other_atom =
              (scratch.head_epoch[neighbor_cell_idx] == scratch.current_epoch)
                  ? scratch.head[neighbor_cell_idx]
                  : -1;
          while (other_atom >= 0) {
            if (accumulate_pair(atom_idx, static_cast<size_t>(other_atom)))
              return;
            other_atom = scratch.next[static_cast<size_t>(other_atom)];
          }
        }
      }
    }

    const size_t own_cell_idx = cell_index(cx, cy, cz);
    scratch.next[atom_idx] =
        (scratch.head_epoch[own_cell_idx] == scratch.current_epoch)
            ? scratch.head[own_cell_idx]
            : -1;
    scratch.head[own_cell_idx] = static_cast<int>(atom_idx);
    scratch.head_epoch[own_cell_idx] = scratch.current_epoch;
  }
}

void apply_cut_candidate(const vector<PbctopoAtom> &atoms,
                         const vector<Vec3> &placed_points,
                         Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                         size_t frame_index) {
  if (atoms.empty() || placed_points.size() != atoms.size())
    return;

  box Box = mol1_system.m_mol[frame_index].Box;
  const PbctopoLatticeMetric lattice(Box);
  double min_sx = numeric_limits<double>::infinity();
  double min_sy = numeric_limits<double>::infinity();
  double min_sz = numeric_limits<double>::infinity();
  double max_sx = -numeric_limits<double>::infinity();
  double max_sy = -numeric_limits<double>::infinity();
  double max_sz = -numeric_limits<double>::infinity();

  for (const auto &point : placed_points) {
    const auto scaled = lattice.cartesian_to_scaled(point.x, point.y, point.z);
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

  double v1[3]{}, v2[3]{}, v3[3]{};
  Box.get_v1(v1);
  Box.get_v2(v2);
  Box.get_v3(v3);
  const Vec3 recenter_shift{
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[0] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[0] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[0],
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[1] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[1] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[1],
      (0.5 - 0.5 * (min_sx + max_sx)) * v1[2] +
          (0.5 - 0.5 * (min_sy + max_sy)) * v2[2] +
          (0.5 - 0.5 * (min_sz + max_sz)) * v3[2]};

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    const Vec3 point{placed_points[idx].x + recenter_shift.x,
                     placed_points[idx].y + recenter_shift.y,
                     placed_points[idx].z + recenter_shift.z};
    if (atom.owner == 0) {
      mol1_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol1_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol1_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    } else if (frame_index < mol2_system.m_mol.size()) {
      mol2_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol2_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol2_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    }
  }
}

void apply_pbctopo_points_direct(const vector<PbctopoAtom> &atoms,
                                 const vector<Vec3> &placed_points,
                                 Multi_MOL &mol1_system,
                                 Multi_MOL &mol2_system,
                                 size_t frame_index) {
  if (atoms.empty() || placed_points.size() != atoms.size())
    return;

  for (size_t idx = 0; idx < atoms.size(); ++idx) {
    const auto &atom = atoms[idx];
    const Vec3 &point = placed_points[idx];
    if (atom.owner == 0) {
      mol1_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol1_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol1_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    } else if (frame_index < mol2_system.m_mol.size()) {
      mol2_system.m_mol[frame_index].pos.x[atom.local_atom_index] = point.x;
      mol2_system.m_mol[frame_index].pos.y[atom.local_atom_index] = point.y;
      mol2_system.m_mol[frame_index].pos.z[atom.local_atom_index] = point.z;
    }
  }
}

[[maybe_unused]] void
append_fragments_for_system(const Multi_MOL &system, const MOL &mol, int owner,
                            vector<PbctopoFragment> &fragments,
                            vector<PbctopoAtom> &atoms) {
  const auto ranges = build_ranges_for_system(system, mol);
  for (const auto &range : ranges) {
    PbctopoFragment fragment;
    fragment.id = fragments.size();
    fragment.owner = owner;
    fragment.source_index = range.source_index;
    fragment.begin_local = range.begin;
    fragment.end_local = range.end;
    fragment.base_atom_begin = atoms.size();
    fragment.atom_count = range.end - range.begin;
    fragment.topo_backed = range.topo_backed;

    for (size_t local = range.begin; local < range.end; ++local) {
      double r[3]{mol.pos.x[local], mol.pos.y[local], mol.pos.z[local]};
      double s[3]{};
      mol.Box.to_scaled(r, s);
      for (double &coord : s)
        coord -= floor(coord);
      scaled_to_cartesian(mol.Box, s[0], s[1], s[2], r[0], r[1], r[2]);
      PbctopoAtom atom;
      atom.global_atom_id = atoms.size();
      atom.fragment_id = fragment.id;
      atom.owner = owner;
      atom.local_atom_index = local;
      atom.source_atom_id =
          local < system.info.size() &&
                  system.info[local].has_original_atom_id()
              ? system.info[local].get_original_atom_id()
              : local;
      if (const auto mass = system.topology.atomic_mass(static_cast<std::uint32_t>(local));
          mass.has_value()) {
        atom.atomic_mass = *mass;
        atom.has_atomic_mass = true;
      }
      atom.x = r[0];
      atom.y = r[1];
      atom.z = r[2];
      atom.sx = s[0];
      atom.sy = s[1];
      atom.sz = s[2];
      atoms.push_back(atom);
    }
    fragment.base_atom_end = atoms.size();
    fragments.push_back(fragment);
  }
}

} // namespace pbctopo_solve

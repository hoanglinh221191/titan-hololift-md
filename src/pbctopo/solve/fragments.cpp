#include "pbctopo/solve/solve_common.h"

namespace pbctopo_solve {

vector<PbctopoShift> build_topology_shifts(box Box, int shell_radius) {
  return titan_pbctopo::build_component_image_shifts(Box, shell_radius);
}

[[maybe_unused]] void
build_supercell_for_frame(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoAtom> &base_atoms,
                          const vector<PbctopoShift> &shifts,
                          vector<PbctopoFragmentCopyNode> &nodes,
                          vector<PbctopoAtomCopy> &atom_copies) {
  nodes.clear();
  atom_copies.clear();
  nodes.reserve(fragments.size() * shifts.size());
  atom_copies.reserve(base_atoms.size() * shifts.size());

  for (const auto &fragment : fragments) {
    for (const auto &shift : shifts) {
      PbctopoFragmentCopyNode node;
      node.node_id = nodes.size();
      node.fragment_id = fragment.id;
      node.shift_index = shift.shift_index;
      node.shift_ix = shift.ix;
      node.shift_iy = shift.iy;
      node.shift_iz = shift.iz;
      node.tx = shift.tx;
      node.ty = shift.ty;
      node.tz = shift.tz;
      node.atom_begin = atom_copies.size();
      node.min_x = numeric_limits<double>::infinity();
      node.min_y = numeric_limits<double>::infinity();
      node.min_z = numeric_limits<double>::infinity();
      node.max_x = -numeric_limits<double>::infinity();
      node.max_y = -numeric_limits<double>::infinity();
      node.max_z = -numeric_limits<double>::infinity();

      for (size_t atom_idx = fragment.base_atom_begin;
           atom_idx < fragment.base_atom_end; ++atom_idx) {
        const PbctopoAtom &base_atom = base_atoms[atom_idx];
        PbctopoAtomCopy atom_copy;
        atom_copy.global_atom_id = base_atom.global_atom_id;
        atom_copy.fragment_id = fragment.id;
        atom_copy.node_id = node.node_id;
        atom_copy.owner = base_atom.owner;
        atom_copy.local_atom_index = base_atom.local_atom_index;
        atom_copy.x = base_atom.x + shift.tx;
        atom_copy.y = base_atom.y + shift.ty;
        atom_copy.z = base_atom.z + shift.tz;
        node.min_x = min(node.min_x, atom_copy.x);
        node.min_y = min(node.min_y, atom_copy.y);
        node.min_z = min(node.min_z, atom_copy.z);
        node.max_x = max(node.max_x, atom_copy.x);
        node.max_y = max(node.max_y, atom_copy.y);
        node.max_z = max(node.max_z, atom_copy.z);
        atom_copies.push_back(atom_copy);
      }

      node.atom_end = atom_copies.size();
      nodes.push_back(node);
    }
  }
}

bool are_topology_neighbors(const PbctopoFragment &lhs,
                            const PbctopoFragment &rhs) {
  if (!lhs.topo_backed || !rhs.topo_backed || lhs.owner != rhs.owner)
    return false;
  if (lhs.source_index + 1 == rhs.source_index ||
      rhs.source_index + 1 == lhs.source_index)
    return true;
  return false;
}

[[maybe_unused]] vector<vector<size_t>>
build_fragment_copy_graph(const vector<PbctopoFragment> &fragments,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          double geometry_cutoff, double topology_cutoff) {
  vector<vector<size_t>> adjacency(nodes.size());
  const double geometry_cutoff2 = geometry_cutoff * geometry_cutoff;
  const double topology_cutoff2 = topology_cutoff * topology_cutoff;
  const double max_cutoff = max(geometry_cutoff, topology_cutoff);
  const double inv_cell = 1.0 / max_cutoff;
  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> cell_nodes;
  cell_nodes.reserve(max<size_t>(64, nodes.size() * 2));

  for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
    const auto &node = nodes[node_id];
    const int min_cx = static_cast<int>(floor((node.min_x - max_cutoff) * inv_cell));
    const int min_cy = static_cast<int>(floor((node.min_y - max_cutoff) * inv_cell));
    const int min_cz = static_cast<int>(floor((node.min_z - max_cutoff) * inv_cell));
    const int max_cx = static_cast<int>(floor((node.max_x + max_cutoff) * inv_cell));
    const int max_cy = static_cast<int>(floor((node.max_y + max_cutoff) * inv_cell));
    const int max_cz = static_cast<int>(floor((node.max_z + max_cutoff) * inv_cell));

    for (int cx = min_cx; cx <= max_cx; ++cx) {
      for (int cy = min_cy; cy <= max_cy; ++cy) {
        for (int cz = min_cz; cz <= max_cz; ++cz)
          cell_nodes[ClashCellKey{cx, cy, cz}].push_back(node_id);
      }
    }
  }

  unordered_set<uint64_t> seen_pairs;
  seen_pairs.reserve(max<size_t>(128, nodes.size() * 8));
  for (const auto &cell_entry : cell_nodes) {
    const auto &cell_list = cell_entry.second;
    for (size_t a = 0; a < cell_list.size(); ++a) {
      for (size_t b = a + 1; b < cell_list.size(); ++b) {
        size_t lhs_id = cell_list[a];
        size_t rhs_id = cell_list[b];
        if (lhs_id == rhs_id)
          continue;
        if (lhs_id > rhs_id)
          swap(lhs_id, rhs_id);
        const uint64_t pair_key =
            (static_cast<uint64_t>(lhs_id) << 32) | static_cast<uint64_t>(rhs_id);
        if (!seen_pairs.insert(pair_key).second)
          continue;

        const auto &lhs = nodes[lhs_id];
        const auto &rhs = nodes[rhs_id];
        if (lhs.fragment_id == rhs.fragment_id)
          continue;

        const bool topo_related =
            are_topology_neighbors(fragments[lhs.fragment_id],
                                   fragments[rhs.fragment_id]);
        const double cutoff2 = topo_related ? topology_cutoff2 : geometry_cutoff2;
        if (bbox_distance2(lhs, rhs) > cutoff2)
          continue;

        adjacency[lhs_id].push_back(rhs_id);
        adjacency[rhs_id].push_back(lhs_id);
      }
    }
  }
  return adjacency;
}

[[maybe_unused]] void
sort_fragment_copy_graph(vector<vector<size_t>> &adjacency,
                         const vector<PbctopoFragmentCopyNode> &nodes) {
  for (size_t node_id = 0; node_id < adjacency.size(); ++node_id) {
    auto &neighbors = adjacency[node_id];
    sort(neighbors.begin(), neighbors.end(),
         [&](size_t lhs, size_t rhs) {
           const double lhs_d2 = bbox_distance2(nodes[node_id], nodes[lhs]);
           const double rhs_d2 = bbox_distance2(nodes[node_id], nodes[rhs]);
           if (fabs(lhs_d2 - rhs_d2) > 1.0e-9)
             return lhs_d2 < rhs_d2;
           const double lhs_shift = node_shift_norm2(nodes[lhs]);
           const double rhs_shift = node_shift_norm2(nodes[rhs]);
           if (fabs(lhs_shift - rhs_shift) > 1.0e-9)
             return lhs_shift < rhs_shift;
           return lhs < rhs;
         });
  }
}

double min_bbox_distance2_to_selected(
    size_t node_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes) {
  double best_d2 = numeric_limits<double>::infinity();
  for (size_t selected_id : selected_nodes)
    best_d2 = min(best_d2, bbox_distance2(nodes[node_id], nodes[selected_id]));
  return best_d2;
}

[[maybe_unused]] bool
build_candidate_block_from_root(size_t root_node,
                                const vector<PbctopoFragment> &fragments,
                                const vector<PbctopoFragmentCopyNode> &nodes,
                                const vector<vector<size_t>> &adjacency,
                                size_t total_atom_count,
                                vector<size_t> &component_nodes) {
  component_nodes.clear();
  if (root_node >= nodes.size())
    return false;

  vector<int> fragment_assignment(fragments.size(), -1);
  const size_t root_fragment = nodes[root_node].fragment_id;
  fragment_assignment[root_fragment] = static_cast<int>(root_node);
  component_nodes.push_back(root_node);

  size_t component_atom_count = fragments[root_fragment].atom_count;
  if (component_atom_count > total_atom_count)
    return false;

  while (component_nodes.size() < fragments.size()) {
    size_t next_fragment = numeric_limits<size_t>::max();
    size_t next_node = numeric_limits<size_t>::max();
    double next_d2 = numeric_limits<double>::infinity();
    double next_shift = numeric_limits<double>::infinity();

    for (size_t fragment_id = 0; fragment_id < fragments.size(); ++fragment_id) {
      if (fragment_assignment[fragment_id] != -1)
        continue;
      vector<size_t> options =
          collect_fragment_options(fragment_id, component_nodes, nodes, adjacency, 1);
      if (options.empty())
        continue;
      const size_t candidate_node = options.front();
      const double candidate_d2 =
          min_bbox_distance2_to_selected(candidate_node, component_nodes, nodes);
      const double candidate_shift = node_shift_norm2(nodes[candidate_node]);
      if (next_fragment == numeric_limits<size_t>::max() ||
          candidate_d2 < next_d2 - 1.0e-9 ||
          (fabs(candidate_d2 - next_d2) <= 1.0e-9 &&
           candidate_shift < next_shift - 1.0e-9) ||
          (fabs(candidate_d2 - next_d2) <= 1.0e-9 &&
           fabs(candidate_shift - next_shift) <= 1.0e-9 &&
           candidate_node < next_node)) {
        next_fragment = fragment_id;
        next_node = candidate_node;
        next_d2 = candidate_d2;
        next_shift = candidate_shift;
      }
    }

    if (next_fragment == numeric_limits<size_t>::max())
      return false;

    fragment_assignment[next_fragment] = static_cast<int>(next_node);
    component_nodes.push_back(next_node);
    component_atom_count += fragments[next_fragment].atom_count;
    if (component_atom_count > total_atom_count)
      return false;
  }

  if (component_atom_count != total_atom_count ||
      component_nodes.size() != fragments.size()) {
    return false;
  }

  sort(component_nodes.begin(), component_nodes.end());
  return true;
}

vector<size_t> collect_fragment_options(
    size_t fragment_id, const vector<size_t> &selected_nodes,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options) {
  unordered_map<size_t, double> candidate_dist2;
  candidate_dist2.reserve(16);
  for (size_t selected_node : selected_nodes) {
    for (size_t neighbor : adjacency[selected_node]) {
      if (nodes[neighbor].fragment_id != fragment_id)
        continue;
      const double d2 = bbox_distance2(nodes[selected_node], nodes[neighbor]);
      const auto it = candidate_dist2.find(neighbor);
      if (it == candidate_dist2.end() || d2 < it->second)
        candidate_dist2[neighbor] = d2;
    }
  }

  vector<pair<size_t, double>> ranked(candidate_dist2.begin(),
                                      candidate_dist2.end());
  sort(ranked.begin(), ranked.end(),
       [&](const pair<size_t, double> &lhs, const pair<size_t, double> &rhs) {
         int metric_cmp = compare_pbctopo_double_metric(
             lhs.second, rhs.second, 1.0e-9);
         if (metric_cmp != 0)
           return metric_cmp < 0;
         const double lhs_shift = node_shift_norm2(nodes[lhs.first]);
         const double rhs_shift = node_shift_norm2(nodes[rhs.first]);
         metric_cmp = compare_pbctopo_double_metric(lhs_shift, rhs_shift,
                                                     1.0e-9);
         if (metric_cmp != 0)
           return metric_cmp < 0;
         return lhs.first < rhs.first;
       });

  vector<size_t> options;
  options.reserve(min(max_options, ranked.size()));
  for (size_t i = 0; i < ranked.size() && i < max_options; ++i)
    options.push_back(ranked[i].first);
  return options;
}

[[maybe_unused]] void enumerate_connected_block_candidates_recursive(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment,
    vector<int> &assignment, vector<size_t> &selected_nodes,
    unordered_set<vector<size_t>, NodeVectorHash> &seen_components,
    vector<vector<size_t>> &out_components) {
  if (selected_nodes.size() == fragments.size()) {
    vector<size_t> candidate_nodes = selected_nodes;
    sort(candidate_nodes.begin(), candidate_nodes.end());
    if (seen_components.insert(candidate_nodes).second)
      out_components.push_back(std::move(candidate_nodes));
    return;
  }

  size_t next_fragment = numeric_limits<size_t>::max();
  vector<size_t> next_options;
  for (size_t fragment_id = 0; fragment_id < fragments.size(); ++fragment_id) {
    if (assignment[fragment_id] != -1)
      continue;
    vector<size_t> options =
        collect_fragment_options(fragment_id, selected_nodes, nodes, adjacency,
                                 max_options_per_fragment);
    if (options.empty())
      continue;
    if (next_fragment == numeric_limits<size_t>::max() ||
        options.size() < next_options.size()) {
      next_fragment = fragment_id;
      next_options = std::move(options);
    }
  }

  if (next_fragment == numeric_limits<size_t>::max())
    return;

  for (size_t node_id : next_options) {
    assignment[next_fragment] = static_cast<int>(node_id);
    selected_nodes.push_back(node_id);
    enumerate_connected_block_candidates_recursive(
        fragments, nodes, adjacency, max_options_per_fragment, assignment,
        selected_nodes, seen_components, out_components);
    selected_nodes.pop_back();
    assignment[next_fragment] = -1;
  }
}

[[maybe_unused]] vector<vector<size_t>> enumerate_connected_block_candidates(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t max_options_per_fragment) {
  vector<vector<size_t>> out_components;
  if (fragments.empty())
    return out_components;

  unordered_set<vector<size_t>, NodeVectorHash> seen_components;
  const size_t root_fragment = 0;
  for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
    if (nodes[node_id].fragment_id != root_fragment)
      continue;
    vector<int> assignment(fragments.size(), -1);
    vector<size_t> selected_nodes{node_id};
    assignment[root_fragment] = static_cast<int>(node_id);
    enumerate_connected_block_candidates_recursive(
        fragments, nodes, adjacency, max_options_per_fragment, assignment,
        selected_nodes, seen_components, out_components);
  }

  return out_components;
}

double compute_mst2(const vector<Vec3> &points) {
  if (points.size() < 2)
    return 0.0;

  vector<double> best(points.size(), numeric_limits<double>::infinity());
  vector<char> used(points.size(), 0);
  best[0] = 0.0;
  double mst2 = 0.0;

  for (size_t step = 0; step < points.size(); ++step) {
    size_t next = numeric_limits<size_t>::max();
    double next_cost = numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i] != 0)
        continue;
      if (best[i] < next_cost) {
        next = i;
        next_cost = best[i];
      }
    }
    if (next == numeric_limits<size_t>::max())
      break;
    used[next] = 1;
    mst2 += next_cost;
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i] != 0)
        continue;
      const double d2 = norm2(points[next] - points[i]);
      if (d2 < best[i])
        best[i] = d2;
    }
  }

  return mst2;
}

size_t count_component_connectivity_edges(
    const vector<size_t> &component_nodes,
    const vector<vector<size_t>> &adjacency) {
  if (component_nodes.empty() || adjacency.empty())
    return 0;
  vector<char> node_mask(adjacency.size(), 0);
  for (size_t node_id : component_nodes) {
    if (node_id < node_mask.size())
      node_mask[node_id] = 1;
  }
  size_t edge_count = 0;
  for (size_t node_id : component_nodes) {
    if (node_id >= adjacency.size())
      continue;
    for (size_t neighbor : adjacency[node_id]) {
      if (neighbor >= node_mask.size() || node_mask[neighbor] == 0 ||
          node_id >= neighbor) {
        continue;
      }
      ++edge_count;
    }
  }
  return edge_count;
}

PbctopoCandidate
score_coarse_node_combination(const vector<size_t> &component_nodes,
                              const vector<PbctopoFragmentCopyNode> &nodes,
                              const vector<vector<size_t>> &adjacency) {
  PbctopoCandidate candidate;
  candidate.node_ids = component_nodes;
  if (component_nodes.empty())
    return candidate;

  vector<Vec3> centers;
  centers.reserve(component_nodes.size());
  double min_x = numeric_limits<double>::infinity();
  double min_y = numeric_limits<double>::infinity();
  double min_z = numeric_limits<double>::infinity();
  double max_x = -numeric_limits<double>::infinity();
  double max_y = -numeric_limits<double>::infinity();
  double max_z = -numeric_limits<double>::infinity();
  candidate.shift_norm2 = 0.0;

  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    min_x = min(min_x, node.min_x);
    min_y = min(min_y, node.min_y);
    min_z = min(min_z, node.min_z);
    max_x = max(max_x, node.max_x);
    max_y = max(max_y, node.max_y);
    max_z = max(max_z, node.max_z);
    centers.push_back(node_bbox_center(node));
    candidate.shift_norm2 += node_shift_norm2(node);
  }

  candidate.bbox_volume =
      max(0.0, max_x - min_x) * max(0.0, max_y - min_y) * max(0.0, max_z - min_z);
  candidate.rg2 = compute_rg2(centers);
  candidate.fragment_rg2 = candidate.rg2;
  const PointStats center_stats = build_point_stats(centers);
  const auto center_principal_rg2 = compute_principal_rg2(center_stats);
  const auto center_shape_metrics =
      compute_gyration_shape_metrics(center_principal_rg2);
  candidate.shape_kappa2 = center_shape_metrics.kappa2;
  candidate.shape_asphericity = center_shape_metrics.asphericity;
  candidate.shape_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_kappa2 = center_shape_metrics.kappa2;
  candidate.fragment_asphericity = center_shape_metrics.asphericity;
  candidate.fragment_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_mst2 = compute_mst2(centers);
  candidate.connectivity_edges =
      count_component_connectivity_edges(component_nodes, adjacency);
  candidate.valid = true;
  return candidate;
}

bool better_pbctopo_coarse_candidate(const PbctopoCandidate &lhs,
                                     const PbctopoCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.connectivity_edges != rhs.connectivity_edges)
    return lhs.connectivity_edges > rhs.connectivity_edges;
  int metric_cmp =
      compare_pbctopo_double_metric(lhs.fragment_mst2, rhs.fragment_mst2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int fragment_kappa_cmp =
      compare_shape_kappa(lhs.fragment_kappa2, rhs.fragment_kappa2);
  if (fragment_kappa_cmp != 0)
    return fragment_kappa_cmp < 0;
  const int fragment_asph_cmp = compare_shape_moment(
      lhs.fragment_asphericity, rhs.fragment_asphericity);
  if (fragment_asph_cmp != 0)
    return fragment_asph_cmp < 0;
  const int fragment_acyl_cmp = compare_shape_moment(
      lhs.fragment_acylindricity, rhs.fragment_acylindricity);
  if (fragment_acyl_cmp != 0)
    return fragment_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.bbox_volume, rhs.bbox_volume);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int shape_kappa_cmp =
      compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2);
  if (shape_kappa_cmp != 0)
    return shape_kappa_cmp < 0;
  const int shape_asph_cmp =
      compare_shape_moment(lhs.shape_asphericity, rhs.shape_asphericity);
  if (shape_asph_cmp != 0)
    return shape_asph_cmp < 0;
  const int shape_acyl_cmp =
      compare_shape_moment(lhs.shape_acylindricity, rhs.shape_acylindricity);
  if (shape_acyl_cmp != 0)
    return shape_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return lhs.node_ids < rhs.node_ids;
}

[[maybe_unused]] void enumerate_small_fragment_shift_combinations_recursive(
    size_t fragment_id, const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, vector<size_t> &selected_nodes,
    vector<PbctopoCandidate> &shortlist, size_t shortlist_limit) {
  if (fragment_id == fragments.size()) {
    PbctopoCandidate candidate =
        score_coarse_node_combination(selected_nodes, nodes, adjacency);
    shortlist.push_back(std::move(candidate));
    if (shortlist.size() > shortlist_limit * 2) {
      sort(shortlist.begin(), shortlist.end(), better_pbctopo_coarse_candidate);
      shortlist.resize(shortlist_limit);
    }
    return;
  }

  const size_t base_node = fragment_id * 27;
  for (size_t shift_idx = 0; shift_idx < 27; ++shift_idx) {
    selected_nodes.push_back(base_node + shift_idx);
    enumerate_small_fragment_shift_combinations_recursive(
        fragment_id + 1, fragments, nodes, adjacency, selected_nodes,
        shortlist, shortlist_limit);
    selected_nodes.pop_back();
  }
}

[[maybe_unused]] vector<vector<size_t>>
enumerate_small_fragment_shift_combinations(
    const vector<PbctopoFragment> &fragments,
    const vector<PbctopoFragmentCopyNode> &nodes,
    const vector<vector<size_t>> &adjacency, size_t shortlist_limit) {
  vector<PbctopoCandidate> shortlist;
  vector<size_t> selected_nodes;
  selected_nodes.reserve(fragments.size());
  enumerate_small_fragment_shift_combinations_recursive(
      0, fragments, nodes, adjacency, selected_nodes, shortlist,
      shortlist_limit);
  sort(shortlist.begin(), shortlist.end(), better_pbctopo_coarse_candidate);
  if (shortlist.size() > shortlist_limit)
    shortlist.resize(shortlist_limit);

  vector<vector<size_t>> components;
  components.reserve(shortlist.size());
  for (auto &candidate : shortlist)
    components.push_back(candidate.node_ids);
  return components;
}

[[maybe_unused]] PbctopoCandidate
score_component_candidate(const vector<size_t> &component_nodes,
                          const vector<PbctopoFragmentCopyNode> &nodes,
                          const vector<PbctopoAtomCopy> &atom_copies,
                          const vector<vector<size_t>> &adjacency) {
  PbctopoCandidate candidate;
  candidate.node_ids = component_nodes;
  if (component_nodes.empty())
    return candidate;

  vector<Vec3> centers;
  PointStats stats;
  candidate.shift_norm2 = 0.0;
  centers.reserve(component_nodes.size());
  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    candidate.shift_norm2 +=
        static_cast<double>(node.shift_ix * node.shift_ix + node.shift_iy * node.shift_iy +
                            node.shift_iz * node.shift_iz) *
        static_cast<double>(node.atom_end - node.atom_begin);
    centers.push_back(node_bbox_center(node));
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom = atom_copies[atom_idx];
      stats.add(atom.x, atom.y, atom.z);
    }
  }

  candidate.hull_valid = false;
  candidate.hull_volume = numeric_limits<double>::infinity();
  if (stats.count > 0) {
    candidate.bbox_volume = compute_bbox_volume(stats);
    candidate.rg2 = compute_rg2_from_stats(stats);
    const auto principal_rg2 = compute_principal_rg2(stats);
    const auto shape_metrics = compute_gyration_shape_metrics(principal_rg2);
    candidate.shape_kappa2 = shape_metrics.kappa2;
    candidate.shape_asphericity = shape_metrics.asphericity;
    candidate.shape_acylindricity = shape_metrics.acylindricity;
  }
  candidate.fragment_rg2 = compute_rg2(centers);
  const PointStats center_stats = build_point_stats(centers);
  const auto center_principal_rg2 = compute_principal_rg2(center_stats);
  const auto center_shape_metrics =
      compute_gyration_shape_metrics(center_principal_rg2);
  candidate.fragment_kappa2 = center_shape_metrics.kappa2;
  candidate.fragment_asphericity = center_shape_metrics.asphericity;
  candidate.fragment_acylindricity = center_shape_metrics.acylindricity;
  candidate.fragment_mst2 = compute_mst2(centers);
  candidate.connectivity_edges =
      count_component_connectivity_edges(component_nodes, adjacency);
  candidate.valid = stats.count > 0;
  return candidate;
}

[[maybe_unused]] bool
analyze_candidate_hard_clash(const vector<size_t> &component_nodes,
                             const vector<PbctopoFragmentCopyNode> &nodes,
                             const vector<PbctopoAtomCopy> &atom_copies,
                             double clash_cutoff,
                             double &min_interfragment_d2) {
  min_interfragment_d2 = numeric_limits<double>::infinity();
  if (component_nodes.empty())
    return false;

  size_t total_atoms = 0;
  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    total_atoms += node.atom_end - node.atom_begin;
  }

  unordered_map<ClashCellKey, vector<size_t>, ClashCellKeyHash> clash_grid;
  clash_grid.reserve(max<size_t>(64, total_atoms / 2));
  const double inv_grid = 1.0 / clash_cutoff;
  const double clash_cutoff2 = clash_cutoff * clash_cutoff;

  for (size_t node_id : component_nodes) {
    const auto &node = nodes[node_id];
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom = atom_copies[atom_idx];
      const int cx = static_cast<int>(floor(atom.x * inv_grid));
      const int cy = static_cast<int>(floor(atom.y * inv_grid));
      const int cz = static_cast<int>(floor(atom.z * inv_grid));

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const ClashCellKey neighbor{cx + dx, cy + dy, cz + dz};
            const auto it = clash_grid.find(neighbor);
            if (it == clash_grid.end())
              continue;
            for (size_t other_idx : it->second) {
              const auto &other = atom_copies[other_idx];
              if (other.fragment_id == atom.fragment_id)
                continue;
              const double dx_atom = atom.x - other.x;
              const double dy_atom = atom.y - other.y;
              const double dz_atom = atom.z - other.z;
              const double d2 =
                  dx_atom * dx_atom + dy_atom * dy_atom + dz_atom * dz_atom;
              if (d2 < min_interfragment_d2)
                min_interfragment_d2 = d2;
              if (d2 + kDistanceEps < clash_cutoff2)
                return true;
            }
          }
        }
      }

      clash_grid[ClashCellKey{cx, cy, cz}].push_back(atom_idx);
    }
  }

  return false;
}

[[maybe_unused]] bool
better_pbctopo_candidate(const PbctopoCandidate &lhs,
                         const PbctopoCandidate &rhs) {
  if (!lhs.valid)
    return false;
  if (!rhs.valid)
    return true;
  if (lhs.hard_clash != rhs.hard_clash)
    return rhs.hard_clash;
  if (lhs.connectivity_edges != rhs.connectivity_edges)
    return lhs.connectivity_edges > rhs.connectivity_edges;
  int metric_cmp = compare_pbctopo_double_metric(lhs.fragment_mst2,
                                                  rhs.fragment_mst2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  const int fragment_kappa_cmp =
      compare_shape_kappa(lhs.fragment_kappa2, rhs.fragment_kappa2);
  if (fragment_kappa_cmp != 0)
    return fragment_kappa_cmp < 0;
  const int fragment_asph_cmp = compare_shape_moment(
      lhs.fragment_asphericity, rhs.fragment_asphericity);
  if (fragment_asph_cmp != 0)
    return fragment_asph_cmp < 0;
  const int fragment_acyl_cmp = compare_shape_moment(
      lhs.fragment_acylindricity, rhs.fragment_acylindricity);
  if (fragment_acyl_cmp != 0)
    return fragment_acyl_cmp < 0;
  const int shape_kappa_cmp =
      compare_shape_kappa(lhs.shape_kappa2, rhs.shape_kappa2);
  if (shape_kappa_cmp != 0)
    return shape_kappa_cmp < 0;
  const int shape_asph_cmp =
      compare_shape_moment(lhs.shape_asphericity, rhs.shape_asphericity);
  if (shape_asph_cmp != 0)
    return shape_asph_cmp < 0;
  const int shape_acyl_cmp =
      compare_shape_moment(lhs.shape_acylindricity, rhs.shape_acylindricity);
  if (shape_acyl_cmp != 0)
    return shape_acyl_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.bbox_volume,
                                              rhs.bbox_volume);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  if (lhs.hull_valid && rhs.hull_valid &&
      (metric_cmp = compare_pbctopo_double_metric(lhs.hull_volume,
                                                   rhs.hull_volume)) != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.min_interfragment_d2,
                                              rhs.min_interfragment_d2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  metric_cmp = compare_pbctopo_double_metric(lhs.shift_norm2, rhs.shift_norm2);
  if (metric_cmp != 0)
    return metric_cmp < 0;
  return lhs.node_ids < rhs.node_ids;
}

[[maybe_unused]] void
apply_pbctopo_candidate(Multi_MOL &mol1_system, Multi_MOL &mol2_system,
                        size_t frame_index, const PbctopoCandidate &candidate,
                        const vector<PbctopoFragmentCopyNode> &nodes,
                        const vector<PbctopoAtomCopy> &atom_copies) {
  for (size_t node_id : candidate.node_ids) {
    const auto &node = nodes[node_id];
    for (size_t atom_idx = node.atom_begin; atom_idx < node.atom_end; ++atom_idx) {
      const auto &atom_copy = atom_copies[atom_idx];
      if (atom_copy.owner == 0) {
        mol1_system.m_mol[frame_index].pos.x[atom_copy.local_atom_index] =
            atom_copy.x;
        mol1_system.m_mol[frame_index].pos.y[atom_copy.local_atom_index] =
            atom_copy.y;
        mol1_system.m_mol[frame_index].pos.z[atom_copy.local_atom_index] =
            atom_copy.z;
      } else if (frame_index < mol2_system.m_mol.size()) {
        mol2_system.m_mol[frame_index].pos.x[atom_copy.local_atom_index] =
            atom_copy.x;
        mol2_system.m_mol[frame_index].pos.y[atom_copy.local_atom_index] =
            atom_copy.y;
        mol2_system.m_mol[frame_index].pos.z[atom_copy.local_atom_index] =
            atom_copy.z;
      }
    }
  }
}

} // namespace pbctopo_solve

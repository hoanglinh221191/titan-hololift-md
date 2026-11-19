#include "pbctopo/solve/solver.h"

#include "pbctopo/anchor_mst_rigid.h"

#include "Multi_MOL.h"
#include "titan_executor.h"

using namespace std;

namespace pbctopo_solve {

void PbctopoFrameSolver::require_complete_eobs(const PbctopoAnchorGraphContext &context,
                                               const char *scope) {
  if (context.soft_observed_contact_construction_attempted &&
      !context.soft_observed_contact_construction_complete) {
    throw runtime_error(
        string("PBCTOPO E_obs construction failed closed for ") + scope +
        ": cell_setup_failures=" +
        to_string(context.soft_observed_contact_cell_setup_failures) +
        ", atom_index_failures=" +
        to_string(context.soft_observed_contact_atom_index_failures) +
        ", mic_query_failures=" +
        to_string(context.soft_observed_contact_mic_query_failures));
  }
}

void PbctopoFrameSolver::ensure_component_context() {
  if (component_context_ready)
    return;
  const auto component_context_started = clock::now();
  component_atoms = apply_spatial_lift(
      base_atoms, frame_spatial_lift, self.m_mol[frame_idx].Box);
  if (can_use_component_logic) {
    shifts = build_topology_shifts(self.m_mol[frame_idx].Box,
                                   component_context_shell_radius);
    timing.component_image_shell_max_radius =
        max(timing.component_image_shell_max_radius,
            static_cast<size_t>(component_context_shell_radius));
    timing.component_image_shell_max_shift_count =
        max(timing.component_image_shell_max_shift_count, shifts.size());
    // Component placements translate component_atoms component by component.
    base_anchor_context.rigid_anchor_mst =
        titan_pbctopo::make_pbctopo_rigid_anchor_mst(component_atoms,
                                                     components);
  } else {
    base_anchor_context.rigid_anchor_mst.reset();
  }
  component_context_ready = true;
  ++timing.component_context_builds;
  timing.component_context_s +=
      chrono::duration<double>(clock::now() - component_context_started).count();
}

const titan_pbctopo::PbctopoEdgeImageCache *
PbctopoFrameSolver::ensure_edge_image_cache() {
  if (!certification_lattice.has_value())
    return nullptr;
  if (!edge_image_cache.has_value() ||
      !edge_image_cache->matches(base_atoms, local_edges,
                                 *certification_lattice)) {
    edge_image_cache = titan_pbctopo::build_pbctopo_edge_image_cache(
        base_atoms, local_edges, *certification_lattice);
  }
  return &*edge_image_cache;
}

vector<Int3> PbctopoFrameSolver::extract_offsets_and_time(const vector<PbctopoAtom> &source_atoms,
                                                          const vector<Vec3> &points) {
  const auto extract_started = clock::now();
  auto offsets = extract_component_offsets_from_points(
      source_atoms, components, self.m_mol[frame_idx].Box, points);
  ++timing.component_offset_extracts;
  timing.component_offset_extract_s +=
      chrono::duration<double>(clock::now() - extract_started).count();
  return offsets;
}

bool PbctopoFrameSolver::component_certificate_domain_complete() {
  if (component_certificate_domain_complete_cache >= 0)
    return component_certificate_domain_complete_cache != 0;
  ensure_component_context();
  bool complete = (*certification_lattice).valid() &&
                  component_atoms.size() == base_atoms.size() &&
                  atom_component_index.size() == base_atoms.size() &&
                  !components.empty();
  if (complete) {
    vector<unsigned char> membership_count(base_atoms.size(), 0);
    for (size_t comp_idx = 0; comp_idx < components.size() && complete;
         ++comp_idx) {
      for (size_t atom_idx : components[comp_idx]) {
        if (atom_idx >= base_atoms.size() ||
            atom_component_index[atom_idx] != comp_idx ||
            membership_count[atom_idx] != 0) {
          complete = false;
          break;
        }
        membership_count[atom_idx] = 1;
      }
    }
    complete = complete &&
               all_of(membership_count.begin(), membership_count.end(),
                      [](unsigned char count) { return count == 1; });
  }
  if (complete) {
    for (const auto &edge : local_edges) {
      if (!titan_pbctopo::pbctopo_edge_is_hard_connectivity(edge.kind))
        continue;
      if (edge.a >= atom_component_index.size() ||
          edge.b >= atom_component_index.size() ||
          atom_component_index[edge.a] != atom_component_index[edge.b]) {
        complete = false;
        break;
      }
    }
  }
  if (complete) {
    vector<Vec3> component_identity_points;
    component_identity_points.reserve(component_atoms.size());
    for (const auto &atom : component_atoms)
      component_identity_points.push_back({atom.x, atom.y, atom.z});
    const auto image_equations = evaluate_pbctopo_image_equations(
        base_atoms, component_identity_points, local_edges,
        (*certification_lattice));
    vector<Int3> identity_offsets(components.size());
    const auto identity_assignment =
        make_pbctopo_image_assignment(identity_offsets, components,
                                       component_atoms);
    const auto assignment_points =
        titan_pbctopo::evaluate_pbctopo_assignment_point_consistency(
            component_atoms, component_identity_points,
            atom_component_index, identity_assignment,
            (*certification_lattice));
    complete = identity_assignment.valid &&
               image_equations.topology_edge_residuals == 0 &&
               image_equations.metadata_edge_image_residuals == 0 &&
               image_equations.metadata_edge_distance_residuals == 0 &&
               image_equations.metadata_edges_checked >=
                   metadata_edge_count &&
               image_equations.atom_image_residuals == 0 &&
               image_equations.topology_cycle_residuals == 0 &&
               image_equations.topological_winding_rank == 0 &&
               assignment_points.input_valid &&
               assignment_points.point_residuals == 0;
    frame_report.fixed_certificate_audit =
        evaluate_pbctopo_fixed_certificate_invariants(
            base_atoms, component_identity_points, atom_component_index,
            local_edges, explicit_required_contacts,
            steric_exclusion_set, runtime_cutoffs.clash_cutoff,
            (*certification_lattice), &timing);
    complete =
        complete && frame_report.fixed_certificate_audit.passed();
  }
  component_certificate_domain_complete_cache = complete ? 1 : 0;
  return complete;
}

PbctopoComponentSearchCallbacks PbctopoFrameSolver::make_frame_component_search_callbacks() {
  ensure_component_context();
  auto callbacks = make_component_search_callbacks();
  callbacks.input_atoms_cover_certificate_scope =
      component_atoms.size() == base_atoms.size();
  callbacks.hard_certificate_domain_complete =
      component_certificate_domain_complete();
  callbacks.executor = executor;
  return callbacks;
}

bool PbctopoFrameSolver::prepare_frame() {
  ++timing.frames;
  frame_fatal_lattice_query_failures = 0;
  lattice_query_audit_scope.emplace(
      timing, frame_fatal_lattice_query_failures);
  frame_start_warm_state = warm_state;
  kinematic_seed_history_ready =
      has_consecutive_kinematic_seed_history(warm_state, frame_idx);
  kinematic_seed_previous =
      warm_state.kinematic_previous_virtual_candidate;
  kinematic_seed_latest =
      warm_state.kinematic_latest_virtual_candidate;
  frame_report.frame = frame_report_offset + frame_idx;
  frame_report.time_ps = self.m_mol[frame_idx].time_of_snap;
  frame_report.temporal_inference_experimental =
      temporal_mode == PbctopoTemporalMode::Viterbi;
  frame_report.component_audit_exact_requested = audit_exact_search;
  if (audit_exact_search)
    ++timing.component_audit_exact_frames;
  frame_report_started = clock::now();
  frame_graph_s_before = timing.graph_s + timing.component_graph_s;
  frame_virtual_s_before =
      timing.event_planes_s + timing.coarse_s + timing.refine_s +
      timing.early_projection_s;
  frame_component_s_before =
      timing.component_prep_s + timing.component_graph_s +
      timing.component_search_s + timing.warm_component_s;
  skip_projection_candidate_clash = projection_clash_skip_active;
  if (skip_projection_candidate_clash)
    ++timing.projection_clash_skip_frames;
  scratch.reset_frame();
  stage_started = clock::now();
  append_atoms_for_system(self.m_mol[frame_idx], &self.info,
                          &trajectory_context.rec_static, 0, base_atoms);
  if (frame_idx < mol2.m_mol.size() && !mol2.m_mol[frame_idx].pos.x.empty()) {
    append_atoms_for_system(mol2.m_mol[frame_idx], &mol2.info,
                            &trajectory_context.lig_static, 1, base_atoms);
  }
  frame_binding =
      titan_pbctopo::build_pbctopo_frame_binding_observation(
          frame_report.frame, self.m_mol[frame_idx].Box, base_atoms);
  frame_report.trajectory_frame_index =
      frame_binding.trajectory_frame_index;
  frame_report.box_matrix = frame_binding.box_matrix;
  frame_report.box_hash = frame_binding.box_hash;
  frame_report.atom_selection_hash = frame_binding.atom_selection_hash;
  frame_report.canonical_atom_universe_hash =
      frame_binding.canonical_atom_universe_hash;
  frame_report.wrapped_coordinate_hash =
      frame_binding.wrapped_coordinate_hash;
  frame_report.frame_binding_valid = frame_binding.valid;
  prepare_pbctopo_steric_radii(base_atoms,
                               runtime_cutoffs.vdw_clash_scale);
  timing.append_s += chrono::duration<double>(clock::now() - stage_started).count();

  if (base_atoms.empty()) {
    frame_report.status = PbctopoFrameStatus::Fallback;
    frame_report.mode = PbctopoFrameMode::Pbci;
    frame_report.fallback_reason = PbctopoFailureReason::NoSeed;
    frame_report.time_total_ms =
        chrono::duration<double, milli>(clock::now() - frame_report_started)
            .count();
    write_or_defer_frame_report(frame_idx, frame_report);
    reset_uncommitted_warm_state();
    return false;
  }

  frame_lattice.emplace(self.m_mol[frame_idx].Box);
  ++timing.pbctopo_lattice_audits;
  if (std::isfinite((*frame_lattice).condition_estimate())) {
    timing.pbctopo_lattice_condition_max =
        std::max(timing.pbctopo_lattice_condition_max,
                 (*frame_lattice).condition_estimate());
  }
  if ((*frame_lattice).reduction_fallback())
    ++timing.pbctopo_lattice_reduction_fallback_frames;
  if (!(*frame_lattice).valid()) {
    ++timing.pbctopo_lattice_invalid_frames;
    const string invalid_lattice_message =
        "[pbctopo] frame " +
        to_string(static_cast<long long>(
            self.m_mol[frame_idx].time_of_snap)) +
        " left unchanged: invalid lattice status=" +
        titan_pbctopo::pbctopo_lattice_status_name((*frame_lattice).status()) +
        ", condition_estimate=" +
        to_string((*frame_lattice).condition_estimate());
    append_pbctopo_log_line(invalid_lattice_message);
    TitanUI::print_warning(invalid_lattice_message);
    frame_report.status = PbctopoFrameStatus::Fallback;
    frame_report.mode = PbctopoFrameMode::None;
    frame_report.fallback_reason = PbctopoFailureReason::InvalidLattice;
    frame_report.time_total_ms =
        chrono::duration<double, milli>(clock::now() - frame_report_started)
            .count();
    write_or_defer_frame_report(frame_idx, frame_report);
    reset_uncommitted_warm_state();
    return false;
  }

  stage_started = clock::now();
  build_heavy_atom_subset(base_atoms, projection_atoms);
  heavy_search_omits_atoms =
      projection_atoms.size() < base_atoms.size();
  if (heavy_search_omits_atoms)
    ++timing.pbctopo_heavy_search_frames;
  timing.heavy_subset_s +=
      chrono::duration<double>(clock::now() - stage_started).count();
  stage_started = clock::now();
  translation_cache =
      build_virtual_box_translation_cache(self.m_mol[frame_idx].Box);
  timing.translation_cache_s +=
      chrono::duration<double>(clock::now() - stage_started).count();

  stage_started = clock::now();
  has_ligand_frame =
      frame_idx < mol2.m_mol.size() && !mol2.m_mol[frame_idx].pos.x.empty();
  rec_use_topology =
      trajectory_context.rec_static.topology_graph_available;
  lig_use_topology =
      (has_ligand_frame &&
       trajectory_context.lig_static.topology_graph_available);
  rec_use_metadata =
      !rec_use_topology &&
      trajectory_context.rec_static.metadata_graph_available;
  lig_use_metadata =
      has_ligand_frame &&
      (!trajectory_context.lig_static.topology_graph_available &&
       trajectory_context.lig_static.metadata_graph_available);
  rec_use_geometry = !rec_use_topology && !rec_use_metadata;
  lig_use_geometry =
      has_ligand_frame && !lig_use_topology && !lig_use_metadata;
  any_topology_graph = rec_use_topology || lig_use_topology;
  any_metadata_graph = rec_use_metadata || lig_use_metadata;
  any_geometry_graph = rec_use_geometry || lig_use_geometry;
  if (any_topology_graph && any_metadata_graph) {
    frame_report.certificate_graph_source =
        PbctopoCertificateGraphSource::Mixed;
  } else if (any_topology_graph) {
    frame_report.certificate_graph_source =
        PbctopoCertificateGraphSource::GromacsTopology;
  } else if (any_metadata_graph) {
    frame_report.certificate_graph_source =
        PbctopoCertificateGraphSource::MetadataChainResidue;
  }

  local_edges.clear();
  local_edge_set.clear();
  projected_static_edges =
      (rec_use_topology
           ? trajectory_context.rec_static.projected_topology_bonds.size()
           : rec_use_metadata
                 ? trajectory_context.rec_static.projected_metadata_edges.size()
                 : 0) +
      (lig_use_topology
           ? trajectory_context.lig_static.projected_topology_bonds.size()
           : lig_use_metadata
                 ? trajectory_context.lig_static.projected_metadata_edges.size()
                 : 0);
  local_edges.reserve(projected_static_edges +
                      (any_geometry_graph ? base_atoms.size() * 2 : 0));
  local_edge_set.reserve(local_edges.capacity() * 2 + 1);

  if (any_topology_graph)
    ++timing.pbctopo_topology_static_graph_frames;
  if (any_metadata_graph)
    ++timing.pbctopo_metadata_graph_frames;

  if (rec_use_topology) {
    append_topology_bond_edges_for_system(
        trajectory_context.rec_static, 0, base_atoms,
        self.m_mol[frame_idx].Box, local_edges, local_edge_set, timing);
  } else if (rec_use_metadata) {
    append_metadata_graph_edges_for_system(trajectory_context.rec_static, 0,
                                           base_atoms, local_edges,
                                           local_edge_set, timing);
  }
  if (lig_use_topology) {
    append_topology_bond_edges_for_system(
        trajectory_context.lig_static,
        trajectory_context.rec_static.atom_count, base_atoms,
        self.m_mol[frame_idx].Box, local_edges, local_edge_set, timing);
  } else if (lig_use_metadata) {
    append_metadata_graph_edges_for_system(
        trajectory_context.lig_static,
        trajectory_context.rec_static.atom_count, base_atoms, local_edges,
        local_edge_set, timing);
  }
  static_graph_done = clock::now();
  timing.pbctopo_topology_edge_append_s +=
      chrono::duration<double>(static_graph_done - stage_started).count();

  if (any_geometry_graph) {
    const size_t edge_count_before_geometry = local_edges.size();
    vector<char> geometry_eligible(base_atoms.size(), 0);
    if (rec_use_geometry) {
      const size_t rec_end = min(trajectory_context.rec_static.atom_count,
                                 geometry_eligible.size());
      fill(geometry_eligible.begin(), geometry_eligible.begin() + rec_end, 1);
    }
    if (lig_use_geometry) {
      const size_t lig_begin = trajectory_context.rec_static.atom_count;
      const size_t lig_end = min(
          geometry_eligible.size(),
          lig_begin + trajectory_context.lig_static.atom_count);
      if (lig_begin < lig_end)
        fill(geometry_eligible.begin() + lig_begin,
             geometry_eligible.begin() + lig_end, 1);
    }
    build_local_neighborhood_graph(base_atoms, self.m_mol[frame_idx].Box,
                                   local_edges, local_edge_set,
                                   runtime_cutoffs.bond_cutoff, true,
                                   geometry_eligible);
    timing.pbctopo_geometry_edges_added +=
        local_edges.size() - edge_count_before_geometry;
    ++timing.pbctopo_geometry_fallback_frames;
  }
  timing.pbctopo_local_graph_geometry_s +=
      chrono::duration<double>(clock::now() - static_graph_done).count();
  timing.graph_s += chrono::duration<double>(clock::now() - stage_started).count();

  hard_connectivity_edge_count = 0;
  topology_edge_count = 0;
  metadata_edge_count = 0;
  soft_geometry_edge_count = 0;
  local_graph_atom_coverage.assign(base_atoms.size(), 0);
  steric_exclusion_set.clear();
  steric_exclusion_set.reserve(local_edges.size() + 1);
  for (const auto &edge : local_edges) {
    const bool edge_indices_valid =
        edge.a < base_atoms.size() && edge.b < base_atoms.size();
    if (titan_pbctopo::pbctopo_edge_is_hard_connectivity(edge.kind)) {
      ++hard_connectivity_edge_count;
      if (edge_indices_valid) {
        local_graph_atom_coverage[edge.a] |= 0x1u;
        local_graph_atom_coverage[edge.b] |= 0x1u;
      }
    }
    if (edge.kind == PbctopoLocalEdgeKind::TopologyBond) {
      ++topology_edge_count;
      if (edge_indices_valid) {
        local_graph_atom_coverage[edge.a] |= 0x2u;
        local_graph_atom_coverage[edge.b] |= 0x2u;
      }
    }
    if (edge.kind == PbctopoLocalEdgeKind::MetadataSkeleton) {
      ++metadata_edge_count;
      if (edge_indices_valid) {
        local_graph_atom_coverage[edge.a] |= 0x4u;
        local_graph_atom_coverage[edge.b] |= 0x4u;
      }
    }
    if (edge.kind == PbctopoLocalEdgeKind::GeometryHeuristic)
      ++soft_geometry_edge_count;
    if (titan_pbctopo::pbctopo_edge_is_steric_exclusion(edge.kind))
      steric_exclusion_set.insert(atom_pair_key(edge.a, edge.b));
  }
  hard_graph_atoms_covered = 0;
  topology_bond_atoms_covered = 0;
  metadata_only_atoms = 0;
  for (const std::uint8_t coverage : local_graph_atom_coverage) {
    hard_graph_atoms_covered += (coverage & 0x1u) != 0;
    topology_bond_atoms_covered += (coverage & 0x2u) != 0;
    metadata_only_atoms +=
        (coverage & 0x4u) != 0 && (coverage & 0x2u) == 0;
  }
  timing.pbctopo_hard_connectivity_edges += hard_connectivity_edge_count;
  timing.pbctopo_soft_geometry_edges += soft_geometry_edge_count;
  timing.pbctopo_steric_exclusion_pairs += steric_exclusion_set.size();

  if (solve_mode == PbctopoSolveMode::LlpsLocal) {
    const PbctopoLlpsLocalUnits llps_units =
        build_pbctopo_llps_local_units(self, mol2, base_atoms, local_edges);
    llps_local_unit_source =
        merge_pbctopo_llps_unit_source(llps_local_unit_source,
                                       llps_units.source);
    const auto llps_lattice_audit_before =
        titan_pbctopo::pbctopo_lattice_query_audit();
    vector<Vec3> llps_points = unwrap_pbctopo_llps_local_points(
        base_atoms, local_edges, self.m_mol[frame_idx].Box, llps_units,
        timing);
    const auto llps_lattice_audit_delta =
        titan_pbctopo::pbctopo_lattice_query_audit_delta(
            llps_lattice_audit_before,
            titan_pbctopo::pbctopo_lattice_query_audit());
    frame_fatal_lattice_query_failures +=
        llps_lattice_audit_delta.failures;
    if (llps_lattice_audit_delta.failures > 0) {
      const string failure_message =
          "[pbctopo] LLPS-local frame " +
          to_string(static_cast<long long>(
              self.m_mol[frame_idx].time_of_snap)) +
          " left unchanged after a failed lattice-image query";
      append_pbctopo_log_line(failure_message);
      TitanUI::print_warning(failure_message);
      ++fallback_frames;
      frame_report.status = PbctopoFrameStatus::Fallback;
      frame_report.certificate_scope = PbctopoCertificateScope::None;
      frame_report.mode = PbctopoFrameMode::None;
      frame_report.component_count = llps_units.units.size();
      frame_report.fallback_reason =
          PbctopoFailureReason::LatticeQueryFailed;
      frame_report.time_graph_ms =
          max(0.0, (timing.graph_s + timing.component_graph_s -
                    frame_graph_s_before) *
                       1000.0);
      frame_report.time_total_ms =
          chrono::duration<double, milli>(clock::now() -
                                          frame_report_started)
              .count();
      write_or_defer_frame_report(frame_idx, frame_report);
      reset_uncommitted_warm_state();
      return false;
    }
    stage_started = clock::now();
    apply_pbctopo_points_direct(base_atoms, llps_points, self, mol2,
                                frame_idx);
    timing.apply_s +=
        chrono::duration<double>(clock::now() - stage_started).count();
    ++timing.apply_calls;
    ++timing.pbctopo_llps_local_frames;
    timing.pbctopo_llps_local_units += llps_units.units.size();
    ++valid_candidates;

    {
      // Local mode places no unit relative to another; the contact network
      // of the whole units is still reported (rank 2 for a bilayer).
      vector<size_t> unit_of_atom(base_atoms.size());
      size_t unit_count = llps_units.units.size();
      bool units_valid = llps_units.atom_unit.size() == base_atoms.size();
      for (size_t atom_idx = 0; units_valid && atom_idx < base_atoms.size();
           ++atom_idx) {
        const int unit = llps_units.atom_unit[atom_idx];
        unit_of_atom[atom_idx] =
            unit >= 0 ? static_cast<size_t>(unit) : unit_count++;
      }
      vector<Int3> unit_images;
      const bool lift_usable =
          units_valid &&
          titan_pbctopo::pbctopo_lift_images_from_points(
              self.m_mol[frame_idx].Box, span<const PbctopoAtom>(base_atoms),
              span<const Vec3>(llps_points), unit_images);
      if (!lift_usable) {
        unit_images.assign(base_atoms.size(), Int3{});
        if (!units_valid)
          for (size_t atom_idx = 0; atom_idx < base_atoms.size(); ++atom_idx)
            unit_of_atom[atom_idx] = atom_idx;
        unit_count = units_valid ? unit_count : base_atoms.size();
      }
      titan_pbctopo::PbctopoPeriodicContactInput periodic_input;
      periodic_input.atoms = span<const PbctopoAtom>(base_atoms);
      periodic_input.atom_unit = span<const size_t>(unit_of_atom);
      periodic_input.lift_images = span<const Int3>(unit_images);
      periodic_input.unit_count = unit_count;
      periodic_input.contact_cutoff_A = runtime_cutoffs.pair_contact_cutoff;
      periodic_input.overlap_vdw_scale = runtime_cutoffs.overlap_vdw_scale;
      periodic_input.overlap_fallback_cutoff_A =
          runtime_cutoffs.overlap_fallback_cutoff;
      periodic_input.overlap_excluded_pairs = &steric_exclusion_set;
      apply_periodic_contact_audit(
          titan_pbctopo::audit_pbctopo_periodic_contacts(
              self.m_mol[frame_idx].Box, periodic_input),
          lift_usable);
    }
    frame_report.candidate_steric_counts_evaluated = false;
    frame_report.status = PbctopoFrameStatus::LocalUnwrap;
    frame_report.evidence_state = PbctopoEvidenceState::NotApplicable;
    frame_report.certificate_scope =
        PbctopoCertificateScope::LocalUnwrap;
    frame_report.mode = PbctopoFrameMode::LlpsLocal;
    frame_report.component_count = llps_units.units.size();
    frame_report.soft_valid_candidate_count = 1;
    frame_report.nonclashing_candidate_count = 0;
    frame_report.unique_certified_assignment_count = 0;
    frame_report.committed_candidate_count = 1;
    frame_report.fallback_reason = PbctopoFailureReason::None;
    frame_report.time_graph_ms =
        max(0.0, (timing.graph_s + timing.component_graph_s -
                  frame_graph_s_before) *
                     1000.0);
    frame_report.time_virtual_ms =
        max(0.0, (timing.coarse_s + timing.refine_s +
                  timing.early_projection_s - frame_virtual_s_before) *
                     1000.0);
    frame_report.time_component_ms =
        max(0.0, (timing.component_prep_s + timing.component_graph_s +
                  timing.component_search_s + timing.warm_component_s -
                  frame_component_s_before) *
                     1000.0);
    frame_report.time_total_ms =
        chrono::duration<double, milli>(clock::now() - frame_report_started)
            .count();
    write_or_defer_frame_report(frame_idx, frame_report);
    reset_uncommitted_warm_state();
    return false;
  }

  stage_started = clock::now();
  build_projection_edge_subset(base_atoms, projection_atoms, local_edges,
                               projection_edges, projection_edge_set,
                               projection_index);
  build_projection_continuity_subset(base_atoms, projection_atoms,
                                     trajectory_context.continuity_pairs,
                                     projection_continuity_pairs,
                                     projection_index);
  build_projection_anchor_subset(base_atoms,
                                 trajectory_context.continuity_anchor_indices,
                                 projection_anchor_indices, projection_index);
  local_hard_required_owner_pairs =
      build_contacting_owner_pairs(local_edges, base_atoms,
                                   self.m_mol[frame_idx].Box,
                                   runtime_cutoffs.pair_contact_cutoff,
                                   true);
  projection_hard_required_owner_pairs =
      build_contacting_owner_pairs(projection_edges, projection_atoms,
                                   self.m_mol[frame_idx].Box,
                                   runtime_cutoffs.pair_contact_cutoff,
                                   true);
  timing.pbctopo_hard_required_owner_pairs +=
      local_hard_required_owner_pairs.size();
  timing.projection_edge_s +=
      chrono::duration<double>(clock::now() - stage_started).count();

  base_anchor_context =
      build_anchor_graph_context(base_atoms, self.m_mol[frame_idx].Box,
                                 trajectory_context.continuity_pairs,
                                 trajectory_context.continuity_anchor_indices,
                                 runtime_cutoffs.pair_contact_cutoff,
                                 &timing);
  projection_anchor_context =
      build_anchor_graph_context(projection_atoms, self.m_mol[frame_idx].Box,
                                 projection_continuity_pairs,
                                 projection_anchor_indices,
                                 runtime_cutoffs.pair_contact_cutoff,
                                 &timing);
  return true;
}

void PbctopoFrameSolver::classify_frame() {
  require_complete_eobs(base_anchor_context, "all-atom selected system");
  require_complete_eobs(projection_anchor_context, "projection system");
  use_anchor_order =
      anchor_metric_mode == PbctopoAnchorMetricMode::Full ||
      anchor_metric_mode == PbctopoAnchorMetricMode::NoContact;
  use_anchor_contact =
      anchor_metric_mode == PbctopoAnchorMetricMode::Full ||
      anchor_metric_mode == PbctopoAnchorMetricMode::NoOrder;
  base_anchor_context.use_order_metrics = use_anchor_order;
  base_anchor_context.use_contact_metrics = use_anchor_contact;
  projection_anchor_context.use_order_metrics = use_anchor_order;
  projection_anchor_context.use_contact_metrics = use_anchor_contact;

  stage_started = clock::now();
  components = build_hard_connected_components_from_local_edges(
      base_atoms.size(), local_edges);
  atom_component_index =
      build_atom_component_index(base_atoms.size(), components);
  explicit_evidence_component_edges =
      build_explicit_evidence_component_edges(atom_component_index,
                                              components.size());
  frame_identifiability =
      titan_pbctopo::evaluate_pbctopo_identifiability(
          components.size(), {}, explicit_required_contacts,
          atom_component_index);
  frame_hard_graph_source =
      topology_edge_count > 0 && metadata_edge_count > 0
          ? PbctopoHardGraphSource::MixedHard
          : topology_edge_count > 0
                ? PbctopoHardGraphSource::ExplicitTopology
                : metadata_edge_count > 0
                      ? PbctopoHardGraphSource::ValidatedMetadata
                      : PbctopoHardGraphSource::None;
  frame_report.hard_graph_source = frame_hard_graph_source;
  layout_observation =
      titan_pbctopo::build_pbctopo_component_layout_observation(
          components, base_atoms, local_edges, frame_hard_graph_source);
  frame_report.spatial_lift_policy_version =
      titan_pbctopo::PBCTOPO_SPATIAL_LIFT_POLICY_VERSION;
  if (layout_observation.valid && frame_binding.valid) {
    vector<titan_pbctopo::PbctopoSpatialLiftAtom> lift_atoms;
    lift_atoms.reserve(base_atoms.size());
    for (const auto &atom : base_atoms) {
      lift_atoms.push_back(
          {atom.owner, static_cast<uint64_t>(atom.source_atom_id),
           {atom.x, atom.y, atom.z}});
    }
    vector<titan_pbctopo::PbctopoSpatialLiftEdge> lift_edges;
    lift_edges.reserve(layout_observation.hard_edges.size());
    for (const auto &edge : layout_observation.hard_edges)
      lift_edges.push_back({edge.atom_a, edge.atom_b});
    frame_spatial_lift = titan_pbctopo::build_pbctopo_spatial_lift(
        frame_binding.box_matrix, lift_atoms, lift_edges);
    frame_report.spatial_lift_identity = frame_spatial_lift.identity;
    frame_report.spatial_lift_ambiguous_hard_edges =
        frame_spatial_lift.ambiguous_hard_edges;
    frame_report.spatial_lift_cycle_residuals =
        frame_spatial_lift.cycle_residuals;
    frame_report.spatial_lift_valid = frame_spatial_lift.valid;
    frame_report.bonded_winding_rank = frame_spatial_lift.winding_rank;
    frame_report.bonded_winding_basis = frame_spatial_lift.winding_basis;
    if (!frame_spatial_lift.valid) {
      append_pbctopo_log_line(
          "[pbctopo] spatial lift unavailable at frame " +
          to_string(frame_report.frame) + ": reason=" +
          frame_spatial_lift.error);
    }
  }
  if (layout_observation.valid) {
    const auto existing_layout = find_if(
        emitted_component_layouts.begin(), emitted_component_layouts.end(),
        [&](const PbctopoComponentLayoutObservation &existing) {
          return existing.topology_epoch_id ==
                 layout_observation.topology_epoch_id;
        });
    if (existing_layout == emitted_component_layouts.end()) {
      frame_report.topology_epoch_id =
          layout_observation.topology_epoch_id;
      if (pbctopo_component_layout_report.is_open()) {
        write_pbctopo_component_layout_rows(
            pbctopo_component_layout_report, layout_observation);
      }
      if (pbctopo_hard_edge_report.is_open())
        write_pbctopo_hard_edge_rows(pbctopo_hard_edge_report,
                                     layout_observation);
      emitted_component_layouts.push_back(layout_observation);
    } else if (titan_pbctopo::same_pbctopo_component_layout_observation(
                   *existing_layout, layout_observation)) {
      frame_report.topology_epoch_id =
          layout_observation.topology_epoch_id;
    } else {
      append_pbctopo_log_line(
          "[pbctopo] topology epoch hash collision at frame " +
          to_string(frame_report.frame) +
          ": action=topology_epoch_unavailable");
    }
  } else {
    append_pbctopo_log_line(
        "[pbctopo] component membership artifact unavailable at frame " +
        to_string(frame_report.frame) +
        ": action=observation_provenance_incomplete");
  }
  component_layout =
      build_component_layout_signature(components);
  base_anchor_context.contact_hypotheses.attempted = true;
  if (frame_spatial_lift.valid &&
      frame_spatial_lift.ambiguous_hard_edges == 0 &&
      frame_spatial_lift.cycle_residuals == 0 &&
      frame_spatial_lift.atom_images.size() == base_atoms.size()) {
    vector<Int3> identity_offsets(components.size());
    const auto identity_assignment = make_pbctopo_image_assignment(
        identity_offsets, components, base_atoms);
    base_anchor_context.contact_hypotheses =
        titan_pbctopo::build_pbctopo_contact_hypotheses(
            span<const PbctopoAtom>(base_atoms),
            span<const PbctopoAnchorContactEdge>(
                base_anchor_context.soft_observed_contact_edges),
            span<const size_t>(atom_component_index),
            span<const Int3>(frame_spatial_lift.atom_images),
            identity_assignment);
  } else {
    base_anchor_context.contact_hypotheses.error =
        "spatial lift is invalid, ambiguous, or inconsistent";
  }
  {
    // Periodic contact network and input overlaps (schema 8).  Relations
    // need the lift that makes each component whole; without it only the
    // overlap scan is reported.
    const bool lift_usable =
        frame_spatial_lift.valid &&
        frame_spatial_lift.ambiguous_hard_edges == 0 &&
        frame_spatial_lift.cycle_residuals == 0 &&
        frame_spatial_lift.atom_images.size() == base_atoms.size();
    vector<Int3> zero_images;
    if (!lift_usable)
      zero_images.assign(base_atoms.size(), Int3{});
    vector<pair<size_t, size_t>> required_pairs;
    required_pairs.reserve(explicit_required_contacts.size());
    for (const auto &contact : explicit_required_contacts)
      required_pairs.emplace_back(contact.a, contact.b);
    titan_pbctopo::PbctopoPeriodicContactInput periodic_input;
    periodic_input.atoms = span<const PbctopoAtom>(base_atoms);
    periodic_input.atom_unit = span<const size_t>(atom_component_index);
    periodic_input.lift_images =
        lift_usable ? span<const Int3>(frame_spatial_lift.atom_images)
                    : span<const Int3>(zero_images);
    periodic_input.unit_count = components.size();
    periodic_input.contact_cutoff_A = runtime_cutoffs.pair_contact_cutoff;
    periodic_input.overlap_vdw_scale = runtime_cutoffs.overlap_vdw_scale;
    periodic_input.overlap_fallback_cutoff_A =
        runtime_cutoffs.overlap_fallback_cutoff;
    periodic_input.overlap_excluded_pairs = &steric_exclusion_set;
    periodic_input.required_atom_pairs =
        span<const pair<size_t, size_t>>(required_pairs);
    apply_periodic_contact_audit(
        titan_pbctopo::audit_pbctopo_periodic_contacts(
            self.m_mol[frame_idx].Box, periodic_input),
        lift_usable);
    base_anchor_context.contact_hypotheses.contact_cycle =
        frame_report.contact_cycle;
  }
  if (!base_anchor_context.contact_hypotheses.complete) {
    append_pbctopo_log_line(
        "[pbctopo] shift-hypothesis construction unavailable at frame " +
        to_string(frame_report.frame) + ": " +
        base_anchor_context.contact_hypotheses.error);
  }
  if (temporal_mode == PbctopoTemporalMode::Viterbi)
    temporal_contact_hypotheses[frame_idx] =
        base_anchor_context.contact_hypotheses;
  if (temporal_mode == PbctopoTemporalMode::Viterbi &&
      !atom_component_index.empty() &&
      components.size() <= numeric_limits<uint32_t>::max()) {
    vector<uint32_t> compact_layout;
    compact_layout.reserve(atom_component_index.size());
    bool layout_valid = true;
    for (size_t comp_idx : atom_component_index) {
      if (comp_idx >= components.size() ||
          comp_idx > numeric_limits<uint32_t>::max()) {
        layout_valid = false;
        break;
      }
      compact_layout.push_back(static_cast<uint32_t>(comp_idx));
    }
    if (layout_valid) {
      size_t layout_idx = 0;
      for (; layout_idx < temporal_component_layouts.size(); ++layout_idx) {
        if (temporal_component_layouts[layout_idx].component_count ==
                components.size() &&
            temporal_component_layouts[layout_idx].atom_component_index ==
                compact_layout) {
          break;
        }
      }
      if (layout_idx == temporal_component_layouts.size()) {
        vector<double> component_weights;
        component_weights.reserve(components.size());
        for (const auto &component : components)
          component_weights.push_back(static_cast<double>(component.size()));
        temporal_component_layouts.push_back(
            {std::move(compact_layout), component_layout.component_hashes,
             std::move(component_weights), components.size()});
      }
      temporal_frame_layout[frame_idx] = layout_idx;
    }
  }
  timing.component_split_s +=
      chrono::duration<double>(clock::now() - stage_started).count();
  raw_component_count = components.size();
  component_mode =
      select_pbctopo_component_mode(raw_component_count,
                                    exact_component_limit);
  can_use_component_logic =
      hard_connectivity_edge_count > 0 &&
      component_mode != PbctopoComponentMode::Disabled;
  if (hard_connectivity_edge_count == 0 && raw_component_count > 1)
    ++timing.pbctopo_component_logic_disabled_no_hard_graph;
  same_component_warm_layout =
      warm_state.has_component_offsets &&
      same_component_layout(components,
                            warm_state.previous_component_layout);
  if (warm_state.has_component_offsets && !same_component_warm_layout)
    ++timing.pbctopo_warm_layout_mismatches;
  if (!same_component_warm_layout)
    clear_component_trial_hints(warm_state);
  component_context_shell_radius =
      component_shell_radius_explicit
          ? configured_component_shell_radius
          : (components.size() == 2 ? 2 : 1);
  component_context_ready = false;
}

void PbctopoFrameSolver::prepare_component_context() {
  certification_lattice.emplace(
      self.m_mol[frame_idx].Box);
  if (h9_orbit_enabled &&
      h9_component_representative_report.is_open()) {
    ensure_component_context();
    const vector<Int3> identity_offsets(components.size());
    const PbctopoImageAssignment representative_layout =
        make_pbctopo_image_assignment(identity_offsets, components,
                                       component_atoms);
    double v1[3]{}, v2[3]{}, v3[3]{};
    self.m_mol[frame_idx].Box.get_v1(v1);
    self.m_mol[frame_idx].Box.get_v2(v2);
    self.m_mol[frame_idx].Box.get_v3(v3);
    const auto wrap_unit_interval = [](double value) {
      double wrapped = value - floor(value);
      if (wrapped >= 1.0)
        wrapped = 0.0;
      if (wrapped < 0.0)
        wrapped += 1.0;
      return wrapped == 0.0 ? 0.0 : wrapped;
    };
    for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
      const auto &component = components[comp_idx];
      bool use_mass = !component.empty();
      double total_mass = 0.0;
      for (const size_t atom_idx : component) {
        if (atom_idx >= component_atoms.size() ||
            !component_atoms[atom_idx].has_atomic_mass ||
            !isfinite(component_atoms[atom_idx].atomic_mass) ||
            component_atoms[atom_idx].atomic_mass <= 0.0) {
          use_mass = false;
          break;
        }
        total_mass += component_atoms[atom_idx].atomic_mass;
      }
      if (!isfinite(total_mass) || total_mass <= 0.0)
        use_mass = false;
      Vec3 representative{};
      double denominator = use_mass ? total_mass
                                    : static_cast<double>(component.size());
      bool representative_valid = representative_layout.valid &&
                                  !component.empty() &&
                                  isfinite(denominator) &&
                                  denominator > 0.0 &&
                                  (*certification_lattice).valid();
      if (representative_valid) {
        for (const size_t atom_idx : component) {
          if (atom_idx >= component_atoms.size()) {
            representative_valid = false;
            break;
          }
          const double weight =
              use_mass ? component_atoms[atom_idx].atomic_mass : 1.0;
          representative.x += weight * component_atoms[atom_idx].x;
          representative.y += weight * component_atoms[atom_idx].y;
          representative.z += weight * component_atoms[atom_idx].z;
        }
      }
      if (representative_valid) {
        representative.x /= denominator;
        representative.y /= denominator;
        representative.z /= denominator;
        representative_valid = isfinite(representative.x) &&
                               isfinite(representative.y) &&
                               isfinite(representative.z);
      }
      array<double, 3> scaled{};
      array<double, 3> wrapped_cartesian{};
      if (representative_valid) {
        scaled = (*certification_lattice).cartesian_to_scaled(
            representative.x, representative.y, representative.z);
        representative_valid = isfinite(scaled[0]) && isfinite(scaled[1]) &&
                               isfinite(scaled[2]);
      }
      if (representative_valid) {
        scaled[0] = wrap_unit_interval(scaled[0]);
        scaled[1] = wrap_unit_interval(scaled[1]);
        scaled[2] = wrap_unit_interval(scaled[2]);
        wrapped_cartesian = (*certification_lattice).scaled_to_cartesian(
            scaled[0], scaled[1], scaled[2]);
        representative_valid = isfinite(wrapped_cartesian[0]) &&
                               isfinite(wrapped_cartesian[1]) &&
                               isfinite(wrapped_cartesian[2]);
      }
      const uint64_t component_id =
          representative_layout.valid &&
                  comp_idx < representative_layout.component_ids.size()
              ? representative_layout.component_ids[comp_idx]
              : 0;
      h9_component_representative_report
          << "h9_component_representative_v1," << frame_report.frame << ','
          << self.m_mol[frame_idx].time_of_snap << ',' << comp_idx << ','
          << component_id << ',' << component.size() << ','
          << (use_mass ? "mass_weighted" : "atom_count_mean") << ','
          << (use_mass ? total_mass : 0.0) << ','
          << static_cast<int>(representative_valid) << ',' << v1[0] << ','
          << v1[1] << ',' << v1[2] << ',' << v2[0] << ',' << v2[1] << ','
          << v2[2] << ',' << v3[0] << ',' << v3[1] << ',' << v3[2] << ',';
      if (representative_valid) {
        h9_component_representative_report
            << scaled[0] << ',' << scaled[1] << ',' << scaled[2] << ','
            << wrapped_cartesian[0] << ',' << wrapped_cartesian[1] << ','
            << wrapped_cartesian[2];
      }
      h9_component_representative_report << '\n';
    }
  }
  component_certificate_domain_complete_cache = -1;
  report_soft_valid_candidates = 0;
  report_nonclashing_candidates = 0;
  // Failed hard-certificate evaluations are audit records, not members of
  // the certified band.  Neither selection nor certified counts use them.
  evidence_relation_frame_written = false;
  evidence_relation_export_enabled =
      pbctopo_candidate_evidence_summary_report.is_open() &&
      pbctopo_candidate_evidence_relation_report.is_open();
}

} // namespace pbctopo_solve

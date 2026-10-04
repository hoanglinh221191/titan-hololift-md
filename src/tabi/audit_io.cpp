#include "analyzer_impl.h"

#include "../titan_ui.h"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TitanPBE {
namespace {

std::size_t audit_idx3(std::size_t i, std::size_t j, std::size_t k,
                       std::size_t nx, std::size_t ny) noexcept {
  return (k * ny + j) * nx + i;
}

std::string audit_format_suffix(std::size_t value, int width) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << value;
  return oss.str();
}

std::string audit_csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos)
    return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"')
      out += "\"\"";
    else
      out += ch;
  }
  out += '"';
  return out;
}

std::int64_t audit_count_delta(std::size_t current,
                               std::size_t previous) noexcept {
  if (current >= previous)
    return static_cast<std::int64_t>(current - previous);
  return -static_cast<std::int64_t>(previous - current);
}

double audit_norm3(double x, double y, double z) noexcept {
  return std::sqrt(x * x + y * y + z * z);
}

std::array<double, 2> audit_plane_distances(double grid_coord) noexcept {
  const double integer_dist = std::abs(grid_coord - std::round(grid_coord));
  const double half_dist =
      std::abs((grid_coord - 0.5) - std::round(grid_coord - 0.5));
  return {integer_dist, half_dist};
}

const char *audit_stage_hint(double min_integer,
                             double min_half) noexcept {
  return min_integer <= min_half ? "charge_grid_or_surface_center"
                                 : "midpoint_epsilon_or_boundary";
}

} // namespace

bool AnalyzerImpl::stability_audit_enabled() const noexcept {
  return args_.stability_audit != "off" || args_.stability_mode != "off";
}

bool AnalyzerImpl::stability_detail_enabled() const noexcept {
  return args_.stability_audit == "detail";
}

bool AnalyzerImpl::stability_grid_anchor_enabled() const noexcept {
  return args_.stability_mode == "anchor" ||
         args_.stability_mode == "hysteresis" ||
         args_.stability_mode == "ensemble";
}

bool AnalyzerImpl::stability_hysteresis_enabled() const noexcept {
  return args_.stability_mode == "hysteresis" &&
         !stability_probe_solve_active_;
}

double AnalyzerImpl::compute_anchor_margin_min_A(
    const GridShape &shape, const std::vector<Atom> &atoms) const {
  if (shape.n == 0 || shape.h <= 0.0 || atoms.empty())
    return 0.0;
  const double max_x = shape.ox + static_cast<double>(shape.nx - 1) * shape.h;
  const double max_y = shape.oy + static_cast<double>(shape.ny - 1) * shape.h;
  const double max_z = shape.oz + static_cast<double>(shape.nz - 1) * shape.h;
  double margin = std::numeric_limits<double>::infinity();
  for (const Atom &atom : atoms) {
    const double radius = std::max(0.0, atom.radius);
    margin = std::min(margin, atom.x - radius - shape.ox);
    margin = std::min(margin, max_x - (atom.x + radius));
    margin = std::min(margin, atom.y - radius - shape.oy);
    margin = std::min(margin, max_y - (atom.y + radius));
    margin = std::min(margin, atom.z - radius - shape.oz);
    margin = std::min(margin, max_z - (atom.z + radius));
  }
  return std::isfinite(margin) ? margin : 0.0;
}

AnalyzerImpl::StabilityAudit AnalyzerImpl::compute_stability_audit(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const ReactionSolve &solve) const {
  StabilityAudit audit;
  audit.stability_mode = args_.stability_mode;
  audit.grid_anchor_effective = effective_grid_anchor();
  audit.native_energy_kj_mol = result.energy_kj_mol;
  audit.stable_energy_kj_mol = result.energy_kj_mol;
  if (last_stability_ensemble_.valid) {
    audit.ensemble_samples = last_stability_ensemble_.samples;
    audit.ensemble_mean_kj_mol = last_stability_ensemble_.mean_kj_mol;
    audit.ensemble_stddev_kj_mol = last_stability_ensemble_.stddev_kj_mol;
    audit.ensemble_min_kj_mol = last_stability_ensemble_.min_kj_mol;
    audit.ensemble_max_kj_mol = last_stability_ensemble_.max_kj_mol;
    audit.stable_energy_kj_mol = last_stability_ensemble_.mean_kj_mol;
  }
  audit.energy_delta_vs_native_kj_mol =
      audit.stable_energy_kj_mol - audit.native_energy_kj_mol;
  audit.hysteresis_cells_kept = stability_hysteresis_cells_kept_frame_;
  audit.hysteresis_cells_released = stability_hysteresis_cells_released_frame_;
  const GridShape &shape = solve.shape.n > 0 ? solve.shape : GridShape{};
  if (shape.n == 0 || shape.h <= 0.0 || atoms.empty()) {
    audit.suspect_stage = last_stability_state_.valid ? "missing_grid"
                                                       : "first_frame";
    audit.anchor_status = stability_grid_anchor_enabled() ? "missing_grid"
                                                          : "off";
    return audit;
  }
  audit.anchor_margin_min_A = compute_anchor_margin_min_A(shape, atoms);
  if (stability_grid_anchor_enabled() || args_.grid_anchor == "fixed-center" ||
      args_.grid_anchor == "first-frame") {
    audit.anchor_status =
        audit.anchor_margin_min_A < 0.0 ? "out_of_margin" : "ok";
  } else {
    audit.anchor_status = "per_frame";
  }

  audit.min_integer_grid_units = std::numeric_limits<double>::infinity();
  audit.min_half_grid_units = std::numeric_limits<double>::infinity();
  audit.min_grid_plane_A = std::numeric_limits<double>::infinity();
  std::vector<StabilityAtomSample> samples;
  samples.reserve(std::min<std::size_t>(atoms.size(), 32));

  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
    const Atom &atom = atoms[atom_idx];
    const double gx = (atom.x - shape.ox) / shape.h;
    const double gy = (atom.y - shape.oy) / shape.h;
    const double gz = (atom.z - shape.oz) / shape.h;
    const auto dx = audit_plane_distances(gx);
    const auto dy = audit_plane_distances(gy);
    const auto dz = audit_plane_distances(gz);
    const double min_integer = std::min({dx[0], dy[0], dz[0]});
    const double min_half = std::min({dx[1], dy[1], dz[1]});
    const double min_any = std::min(min_integer, min_half);
    audit.min_integer_grid_units =
        std::min(audit.min_integer_grid_units, min_integer);
    audit.min_half_grid_units =
        std::min(audit.min_half_grid_units, min_half);
    audit.min_grid_plane_A =
        std::min(audit.min_grid_plane_A, min_any * shape.h);

    if (min_integer <= 1.0e-6)
      ++audit.near_integer_atoms_1e6;
    if (min_integer <= 1.0e-4)
      ++audit.near_integer_atoms_1e4;
    if (min_integer <= 1.0e-3)
      ++audit.near_integer_atoms_1e3;
    if (min_half <= 1.0e-6)
      ++audit.near_half_atoms_1e6;
    if (min_half <= 1.0e-4)
      ++audit.near_half_atoms_1e4;
    if (min_half <= 1.0e-3)
      ++audit.near_half_atoms_1e3;

    StabilityAtomSample sample;
    sample.atom_index = atom_idx + 1;
    sample.x = atom.x;
    sample.y = atom.y;
    sample.z = atom.z;
    sample.q = atom.q;
    sample.radius = atom.radius;
    sample.grid_x = gx;
    sample.grid_y = gy;
    sample.grid_z = gz;
    sample.min_integer_grid_units = min_integer;
    sample.min_half_grid_units = min_half;
    sample.min_grid_plane_A = min_any * shape.h;
    sample.atom_name = atom.atom_name;
    sample.resname = atom.resname;
    sample.resnr = atom.resnr;
    sample.chain = atom.chain;
    sample.stage_hint = audit_stage_hint(min_integer, min_half);
    samples.push_back(std::move(sample));
  }

  std::sort(samples.begin(), samples.end(),
            [](const StabilityAtomSample &a,
               const StabilityAtomSample &b) {
              return a.min_grid_plane_A < b.min_grid_plane_A;
            });
  if (samples.size() > 16)
    samples.resize(16);
  audit.samples = std::move(samples);

  if (!std::isfinite(audit.min_integer_grid_units))
    audit.min_integer_grid_units = 0.0;
  if (!std::isfinite(audit.min_half_grid_units))
    audit.min_half_grid_units = 0.0;
  if (!std::isfinite(audit.min_grid_plane_A))
    audit.min_grid_plane_A = 0.0;

  audit.charge_cell_flip_candidates = audit.near_integer_atoms_1e3;
  audit.midpoint_flip_candidates = audit.near_half_atoms_1e3;
  audit.surface_flip_candidates =
      std::max(audit.near_integer_atoms_1e3, audit.near_half_atoms_1e3);
  audit.ion_access_flip_candidates =
      args_.ion_access == "stern"
          ? audit.surface_flip_candidates
          : std::max(audit.near_integer_atoms_1e4,
                     audit.near_half_atoms_1e4);

  const double center_x = shape.ox +
                          0.5 * static_cast<double>(shape.nx - 1) * shape.h;
  const double center_y = shape.oy +
                          0.5 * static_cast<double>(shape.ny - 1) * shape.h;
  const double center_z = shape.oz +
                          0.5 * static_cast<double>(shape.nz - 1) * shape.h;
  if (!last_stability_state_.valid) {
    audit.suspect_stage = "first_frame";
    return audit;
  }

  const StabilityFrameState &prev = last_stability_state_;
  audit.grid_center_delta_A =
      audit_norm3(center_x - prev.center_x, center_y - prev.center_y,
                  center_z - prev.center_z);
  audit.grid_origin_delta_A =
      audit_norm3(shape.ox - prev.origin_x, shape.oy - prev.origin_y,
                  shape.oz - prev.origin_z);
  audit.energy_delta_kj_mol = result.energy_kj_mol - prev.energy_kj_mol;
  audit.solute_delta = audit_count_delta(result.solute_points,
                                         prev.solute_points);
  audit.solvent_delta = audit_count_delta(result.solvent_points,
                                          prev.solvent_points);
  audit.ion_accessible_delta =
      audit_count_delta(result.ion_accessible_points,
                        prev.ion_accessible_points);
  audit.stern_excluded_delta =
      audit_count_delta(result.stern_excluded_points,
                        prev.stern_excluded_points);
  audit.iscm_boundary_delta =
      audit_count_delta(result.iscm_boundary_points,
                        prev.iscm_boundary_points);
  audit.iscm_charged_boundary_delta =
      audit_count_delta(result.iscm_charged_boundary_points,
                        prev.iscm_charged_boundary_points);
  audit.vdwms_added_delta =
      audit_count_delta(result.vdwms_added, prev.vdwms_added);
  audit.vdwms_removed_delta =
      audit_count_delta(result.vdwms_removed, prev.vdwms_removed);
  audit.charge_assigned_delta =
      result.charge_assigned - prev.charge_assigned;
  audit.charge_moment_delta =
      result.charge_moment_error - prev.charge_moment_error;
  audit.boundary_rms_delta = result.boundary_rms - prev.boundary_rms;
  audit.boundary_max_delta = result.boundary_max - prev.boundary_max;

  const bool shape_changed =
      shape.nx != prev.nx || shape.ny != prev.ny || shape.nz != prev.nz ||
      std::abs(shape.h - prev.h) > 1.0e-12 ||
      audit.grid_center_delta_A > std::max(1.0e-9, shape.h * 1.0e-9) ||
      audit.grid_origin_delta_A > std::max(1.0e-9, shape.h * 1.0e-9);
  const bool charge_changed =
      std::abs(audit.charge_assigned_delta) > 1.0e-8 ||
      std::abs(audit.charge_moment_delta) > 1.0e-8;
  const bool surface_changed =
      audit.solute_delta != 0 || audit.solvent_delta != 0 ||
      audit.vdwms_added_delta != 0 || audit.vdwms_removed_delta != 0;
  const bool ion_changed =
      audit.ion_accessible_delta != 0 || audit.stern_excluded_delta != 0;
  const bool iscm_changed =
      audit.iscm_boundary_delta != 0 ||
      audit.iscm_charged_boundary_delta != 0;
  const bool boundary_changed =
      std::abs(audit.boundary_rms_delta) > 1.0e-8 ||
      std::abs(audit.boundary_max_delta) > 1.0e-8;
  const bool near_grid =
      audit.near_integer_atoms_1e3 > 0 || audit.near_half_atoms_1e3 > 0;
  const double abs_energy_delta = std::abs(audit.energy_delta_kj_mol);

  if (shape_changed)
    audit.suspect_stage = "grid_center";
  else if (charge_changed && audit.charge_cell_flip_candidates > 0)
    audit.suspect_stage = "charge_grid";
  else if (surface_changed)
    audit.suspect_stage = "surface_midpoint";
  else if (ion_changed)
    audit.suspect_stage = "ion_access";
  else if (iscm_changed)
    audit.suspect_stage = "iscm_boundary";
  else if (boundary_changed)
    audit.suspect_stage = "boundary";
  else if (near_grid && abs_energy_delta > 1.0e-3)
    audit.suspect_stage = "grid_boundary_candidate";
  else if (abs_energy_delta > 10.0)
    audit.suspect_stage = "energy_delta_unclassified";
  else
    audit.suspect_stage = "stable";
  return audit;
}

void AnalyzerImpl::maybe_write_stability_audit(
    const FrameResult &result, const std::vector<Atom> &atoms,
    const ReactionSolve &solve) {
  if (!stability_audit_enabled())
    return;

  const StabilityAudit audit =
      compute_stability_audit(result, atoms, solve);
  if (!stability_audit_out_.is_open()) {
    stability_audit_out_.open("pbe_stability_audit.csv",
                              std::ios::out | std::ios::trunc);
    if (!stability_audit_out_.is_open())
      throw std::runtime_error("PBE: failed to open pbe_stability_audit.csv.");
  }
  if (!stability_audit_header_written_) {
    stability_audit_out_
        << "trajectory_id,frame_index,global_index,time_ps,fixture,atom_count,"
           "stability_mode,grid_anchor_effective,anchor_status,"
           "anchor_margin_min_A,hysteresis_cells_kept,"
           "hysteresis_cells_released,native_energy_kJmol,"
           "stable_energy_kJmol,energy_delta_vs_native_kJmol,"
           "ensemble_samples,ensemble_mean_kJmol,ensemble_stddev_kJmol,"
           "ensemble_min_kJmol,ensemble_max_kJmol,"
           "grid_nx,grid_ny,grid_nz,grid_spacing_A,grid_origin_x_A,"
           "grid_origin_y_A,grid_origin_z_A,grid_center_x_A,"
           "grid_center_y_A,grid_center_z_A,grid_center_delta_A,"
           "grid_origin_delta_A,energy_kJmol,energy_delta_kJmol,"
           "min_atom_to_integer_grid_units,min_atom_to_half_grid_units,"
           "min_atom_to_grid_plane_A,near_integer_atoms_1e-6,"
           "near_integer_atoms_1e-4,near_integer_atoms_1e-3,"
           "near_half_atoms_1e-6,near_half_atoms_1e-4,"
           "near_half_atoms_1e-3,charge_cell_flip_candidates,"
           "surface_flip_candidates,midpoint_flip_candidates,"
           "ion_access_flip_candidates,solute_points,solute_delta,"
           "solvent_points,solvent_delta,ion_accessible_points,"
           "ion_accessible_delta,stern_excluded_points,stern_excluded_delta,"
           "iscm_boundary_points,iscm_boundary_delta,"
           "iscm_charged_boundary_points,iscm_charged_boundary_delta,"
           "vdwms_added,vdwms_removed,vdwms_added_delta,"
           "vdwms_removed_delta,charge_assigned_delta,"
           "charge_moment_delta,boundary_rms_delta,boundary_max_delta,"
           "status,suspect_stage\n";
    stability_audit_header_written_ = true;
  }
  stability_audit_out_ << result.trajectory_id << ',' << result.frame_index
                       << ',' << result.global_index << ','
                       << std::setprecision(12) << result.time_ps << ','
                       << audit_csv_escape(current_source_label_) << ','
                       << result.atom_count << ','
                       << audit_csv_escape(audit.stability_mode) << ','
                       << audit_csv_escape(audit.grid_anchor_effective) << ','
                       << audit_csv_escape(audit.anchor_status) << ','
                       << audit.anchor_margin_min_A << ','
                       << audit.hysteresis_cells_kept << ','
                       << audit.hysteresis_cells_released << ','
                       << audit.native_energy_kj_mol << ','
                       << audit.stable_energy_kj_mol << ','
                       << audit.energy_delta_vs_native_kj_mol << ','
                       << audit.ensemble_samples << ','
                       << audit.ensemble_mean_kj_mol << ','
                       << audit.ensemble_stddev_kj_mol << ','
                       << audit.ensemble_min_kj_mol << ','
                       << audit.ensemble_max_kj_mol << ','
                       << result.nx << ','
                       << result.ny << ',' << result.nz << ','
                       << result.grid_spacing_A << ','
                       << result.grid_origin_x_A << ','
                       << result.grid_origin_y_A << ','
                       << result.grid_origin_z_A << ','
                       << result.grid_center_x_A << ','
                       << result.grid_center_y_A << ','
                       << result.grid_center_z_A << ','
                       << audit.grid_center_delta_A << ','
                       << audit.grid_origin_delta_A << ','
                       << result.energy_kj_mol << ','
                       << audit.energy_delta_kj_mol << ','
                       << audit.min_integer_grid_units << ','
                       << audit.min_half_grid_units << ','
                       << audit.min_grid_plane_A << ','
                       << audit.near_integer_atoms_1e6 << ','
                       << audit.near_integer_atoms_1e4 << ','
                       << audit.near_integer_atoms_1e3 << ','
                       << audit.near_half_atoms_1e6 << ','
                       << audit.near_half_atoms_1e4 << ','
                       << audit.near_half_atoms_1e3 << ','
                       << audit.charge_cell_flip_candidates << ','
                       << audit.surface_flip_candidates << ','
                       << audit.midpoint_flip_candidates << ','
                       << audit.ion_access_flip_candidates << ','
                       << result.solute_points << ',' << audit.solute_delta
                       << ',' << result.solvent_points << ','
                       << audit.solvent_delta << ','
                       << result.ion_accessible_points << ','
                       << audit.ion_accessible_delta << ','
                       << result.stern_excluded_points << ','
                       << audit.stern_excluded_delta << ','
                       << result.iscm_boundary_points << ','
                       << audit.iscm_boundary_delta << ','
                       << result.iscm_charged_boundary_points << ','
                       << audit.iscm_charged_boundary_delta << ','
                       << result.vdwms_added << ',' << result.vdwms_removed
                       << ',' << audit.vdwms_added_delta << ','
                       << audit.vdwms_removed_delta << ','
                       << audit.charge_assigned_delta << ','
                       << audit.charge_moment_delta << ','
                       << audit.boundary_rms_delta << ','
                       << audit.boundary_max_delta << ','
                       << audit_csv_escape(result.status) << ','
                       << audit_csv_escape(audit.suspect_stage) << '\n';

  if (stability_detail_enabled()) {
    if (!stability_samples_out_.is_open()) {
      stability_samples_out_.open("pbe_stability_atom_samples.csv",
                                  std::ios::out | std::ios::trunc);
      if (!stability_samples_out_.is_open()) {
        throw std::runtime_error(
            "PBE: failed to open pbe_stability_atom_samples.csv.");
      }
    }
    if (!stability_samples_header_written_) {
      stability_samples_out_
          << "trajectory_id,frame_index,global_index,rank,atom_index,"
             "atom_name,resname,resnr,chain,x_A,y_A,z_A,charge,radius_A,"
             "grid_x,grid_y,grid_z,min_integer_grid_units,"
             "min_half_grid_units,min_grid_plane_A,stage_hint\n";
      stability_samples_header_written_ = true;
    }
    for (std::size_t rank = 0; rank < audit.samples.size(); ++rank) {
      const StabilityAtomSample &sample = audit.samples[rank];
      stability_samples_out_
          << result.trajectory_id << ',' << result.frame_index << ','
          << result.global_index << ',' << (rank + 1) << ','
          << sample.atom_index << ','
          << audit_csv_escape(sample.atom_name) << ','
          << audit_csv_escape(sample.resname) << ',' << sample.resnr << ','
          << audit_csv_escape(sample.chain) << ',' << std::setprecision(12)
          << sample.x << ',' << sample.y << ',' << sample.z << ','
          << sample.q << ',' << sample.radius << ',' << sample.grid_x << ','
          << sample.grid_y << ',' << sample.grid_z << ','
          << sample.min_integer_grid_units << ','
          << sample.min_half_grid_units << ','
          << sample.min_grid_plane_A << ','
          << audit_csv_escape(sample.stage_hint) << '\n';
    }
  }

  StabilityFrameState state;
  state.valid = true;
  state.nx = result.nx;
  state.ny = result.ny;
  state.nz = result.nz;
  state.h = result.grid_spacing_A;
  state.origin_x = result.grid_origin_x_A;
  state.origin_y = result.grid_origin_y_A;
  state.origin_z = result.grid_origin_z_A;
  state.center_x = result.grid_center_x_A;
  state.center_y = result.grid_center_y_A;
  state.center_z = result.grid_center_z_A;
  state.energy_kj_mol = result.energy_kj_mol;
  state.solute_points = result.solute_points;
  state.solvent_points = result.solvent_points;
  state.ion_accessible_points = result.ion_accessible_points;
  state.stern_excluded_points = result.stern_excluded_points;
  state.iscm_boundary_points = result.iscm_boundary_points;
  state.iscm_charged_boundary_points = result.iscm_charged_boundary_points;
  state.vdwms_added = result.vdwms_added;
  state.vdwms_removed = result.vdwms_removed;
  state.charge_assigned = result.charge_assigned;
  state.charge_moment_error = result.charge_moment_error;
  state.boundary_rms = result.boundary_rms;
  state.boundary_max = result.boundary_max;
  last_stability_state_ = state;
}

bool AnalyzerImpl::should_write_map() const {
  return args_.map != "off" && args_.map_frames != "none";
}

std::string AnalyzerImpl::map_suffix(const FrameResult &result) const {
  if (args_.map_frames == "last")
    return "_last";
  return "_traj" + audit_format_suffix(result.trajectory_id, 4) + "_frame" +
         audit_format_suffix(result.frame_index, 6);
}

void AnalyzerImpl::maybe_write_maps(const GridShape &shape,
                                    const FrameResult &result,
                                    bool write_phi_now) {
  if (!should_write_map())
    return;
  if (args_.map_frames == "first" && wrote_first_map_)
    return;

  const auto map_time_begin = std::chrono::steady_clock::now();
  const std::string suffix = map_suffix(result);
  if (write_phi_now && (args_.map == "phi" || args_.map == "all")) {
    write_cube("pbe_phi" + suffix + ".cube", shape, "phi", false);
  }
  if (!write_phi_now && (args_.map == "epsilon" || args_.map == "all")) {
    write_cube("pbe_epsilon" + suffix + ".cube", shape, "epsilon", true);
  }
  if (!write_phi_now && (args_.map == "ion" || args_.map == "all")) {
    write_cube("pbe_ion" + suffix + ".cube", shape, "ion_accessible", false,
               true);
  }
  const bool completed_first =
      (write_phi_now && (args_.map == "phi" || args_.map == "all")) ||
      (!write_phi_now && (args_.map == "epsilon" || args_.map == "ion"));
  if (args_.map_frames == "first" && completed_first)
    wrote_first_map_ = true;
  add_timing_ms(
      TimingBucket::MapIo,
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - map_time_begin)
          .count());
}

void AnalyzerImpl::write_cube(const std::string &filename,
                              const GridShape &shape,
                              const std::string &field_name,
                              bool epsilon_field, bool ion_field) const {
  std::ofstream out(filename, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    TitanUI::print_warning("PBE: failed to write " + filename);
    return;
  }
  out << "TITAN PBE " << field_name << "\n";
  out << "units Angstrom; values are TITAN internal PBE units\n";
  out << std::setw(5) << 0 << " " << std::setw(12) << std::fixed
      << std::setprecision(6) << shape.ox << " " << std::setw(12) << shape.oy
      << " " << std::setw(12) << shape.oz << "\n";
  out << std::setw(5) << shape.nx << " " << std::setw(12) << shape.h << " "
      << std::setw(12) << 0.0 << " " << std::setw(12) << 0.0 << "\n";
  out << std::setw(5) << shape.ny << " " << std::setw(12) << 0.0 << " "
      << std::setw(12) << shape.h << " " << std::setw(12) << 0.0 << "\n";
  out << std::setw(5) << shape.nz << " " << std::setw(12) << 0.0 << " "
      << std::setw(12) << 0.0 << " " << std::setw(12) << shape.h << "\n";

  out << std::scientific << std::setprecision(6);
  int per_line = 0;
  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const std::size_t id = audit_idx3(i, j, k, shape.nx, shape.ny);
        const double value =
            ion_field
                ? (grid_.ion_accessible[id] ? 1.0 : 0.0)
                : (epsilon_field ? (grid_.inside[id] ? args_.int_diel
                                                : args_.ext_diel)
                                 : grid_.phi[id]);
        out << std::setw(13) << value;
        if (++per_line == 6) {
          out << "\n";
          per_line = 0;
        }
      }
    }
  }
  if (per_line != 0)
    out << "\n";
}

} // namespace TitanPBE

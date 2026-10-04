#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <cstdio>
#endif

#include "analyzer_impl.h"

#include "file_sha256.h"

#include "tabi_types.h"
#include "titan_energy_decomposition.h"
#include "titan_executor.h"
#include "titan_interaction_entropy.h"
#include "titan_sasa.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "analyzer_impl_detail.h"

namespace TitanPBE {

using analyzer_impl_detail::csv_double;
using analyzer_impl_detail::csv_escape;
using analyzer_impl_detail::csv_header_map;
using analyzer_impl_detail::csv_index;
using analyzer_impl_detail::idx3;
using analyzer_impl_detail::kKBKcalPerMolK;
using analyzer_impl_detail::kKBKJPerMolK;
using analyzer_impl_detail::kKcalToKJ;
using analyzer_impl_detail::kPi;
using analyzer_impl_detail::split_csv_simple;
using analyzer_impl_detail::trim_copy;

namespace {

constexpr double kDelphiEpk = 167100.9162872952;
constexpr double kDelphiReactionFieldEnergyFactor = 0.9549296586;

} // namespace

AnalyzerImpl::ParityAudit
AnalyzerImpl::compute_parity_audit(const GridShape &shape,
                                   const AlignedDoubleVector *source,
                                   const AlignedDoubleVector *residual) const {
  ParityAudit audit;
  if (shape.n == 0 || grid_.denom.size() != shape.n ||
      grid_.screen.size() != shape.n)
    return audit;

  double eps_min = std::numeric_limits<double>::max();
  double eps_max = 0.0;
  double eps_sum_total = 0.0;
  double screen_sumsq = 0.0;
  double source_sumsq = 0.0;
  std::size_t count = 0;
  for (std::size_t id = 0; id < shape.n; ++id) {
    const double eps_sum =
        std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
    eps_min = std::min(eps_min, eps_sum);
    eps_max = std::max(eps_max, eps_sum);
    eps_sum_total += eps_sum;
    if (grid_.screen[id] > 0.0) {
      ++audit.screen_points;
      audit.screen_sum += grid_.screen[id];
      screen_sumsq += grid_.screen[id] * grid_.screen[id];
      audit.screen_max = std::max(audit.screen_max, std::abs(grid_.screen[id]));
    }
    if (source != nullptr && source->size() == shape.n) {
      const double value = (*source)[id];
      audit.source_sum += value;
      source_sumsq += value * value;
      audit.source_max = std::max(audit.source_max, std::abs(value));
    }
    if (residual != nullptr && residual->size() == shape.n)
      audit.residual_max_abs =
          std::max(audit.residual_max_abs, std::abs((*residual)[id]));
    ++count;
  }
  audit.eps_sum_min = count > 0 ? eps_min : 0.0;
  audit.eps_sum_max = count > 0 ? eps_max : 0.0;
  audit.eps_sum_mean =
      count > 0 ? eps_sum_total / static_cast<double>(count) : 0.0;
  audit.screen_rms =
      audit.screen_points > 0
          ? std::sqrt(screen_sumsq / static_cast<double>(audit.screen_points))
          : 0.0;
  audit.source_rms =
      count > 0 ? std::sqrt(source_sumsq / static_cast<double>(count)) : 0.0;
  if (grid_.eps_x.size() == shape.n && grid_.eps_y.size() == shape.n &&
      grid_.eps_z.size() == shape.n) {
    const std::size_t xy = shape.nx * shape.ny;
    double midpoint_sum_total = 0.0;
    double midpoint_interior_total = 0.0;
    std::size_t midpoint_count = 0;
    std::size_t midpoint_interior_count = 0;
    double midpoint_min = std::numeric_limits<double>::max();
    double midpoint_max = std::numeric_limits<double>::lowest();
    for (std::size_t k = 0; k < shape.nz; ++k) {
      for (std::size_t j = 0; j < shape.ny; ++j) {
        for (std::size_t i = 0; i < shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          double midpoint_sum =
              grid_.eps_x[id] + grid_.eps_y[id] + grid_.eps_z[id];
          if (i > 0)
            midpoint_sum += grid_.eps_x[id - 1];
          if (j > 0)
            midpoint_sum += grid_.eps_y[id - shape.nx];
          if (k > 0)
            midpoint_sum += grid_.eps_z[id - xy];
          if (!std::isfinite(midpoint_sum))
            continue;
          midpoint_min = std::min(midpoint_min, midpoint_sum);
          midpoint_max = std::max(midpoint_max, midpoint_sum);
          midpoint_sum_total += midpoint_sum;
          ++midpoint_count;
          if (i > 0 && i + 1 < shape.nx && j > 0 && j + 1 < shape.ny && k > 0 &&
              k + 1 < shape.nz) {
            midpoint_interior_total += midpoint_sum;
            ++midpoint_interior_count;
          }
        }
      }
    }
    audit.midpoint_eps_sum_min = midpoint_count > 0 ? midpoint_min : 0.0;
    audit.midpoint_eps_sum_max = midpoint_count > 0 ? midpoint_max : 0.0;
    audit.midpoint_eps_sum_mean =
        midpoint_count > 0
            ? midpoint_sum_total / static_cast<double>(midpoint_count)
            : 0.0;
    audit.midpoint_eps_sum_interior_mean =
        midpoint_interior_count > 0
            ? midpoint_interior_total /
                  static_cast<double>(midpoint_interior_count)
            : 0.0;
  }
  return audit;
}

AnalyzerImpl::ForensicAudit
AnalyzerImpl::compute_forensic_audit(const GridShape &shape) const {
  ForensicAudit audit;
  if (shape.n == 0 || grid_.inside.size() != shape.n ||
      grid_.rho.size() != shape.n)
    return audit;

  const double voxel_volume = shape.h * shape.h * shape.h;
  std::size_t inside_count = 0;
  std::size_t face_count = 0;
  double eps_face_sum = 0.0;
  double eps_face_min = std::numeric_limits<double>::max();
  double eps_face_max = 0.0;

  auto add_face_eps = [&](double value) {
    if (!std::isfinite(value))
      return;
    eps_face_min = std::min(eps_face_min, value);
    eps_face_max = std::max(eps_face_max, value);
    eps_face_sum += value;
    ++face_count;
  };
  double eps_axis_sum[3] = {0.0, 0.0, 0.0};
  std::size_t eps_axis_count[3] = {0, 0, 0};
  std::size_t internal_axis_count[3] = {0, 0, 0};
  const double internal_threshold = 0.5 * (args_.int_diel + args_.ext_diel);
  auto add_axis_face_eps = [&](int axis, double value) {
    if (!std::isfinite(value) || axis < 0 || axis > 2)
      return;
    eps_axis_sum[axis] += value;
    ++eps_axis_count[axis];
    if (value <= internal_threshold)
      ++internal_axis_count[axis];
  };

  const std::size_t xy = shape.nx * shape.ny;
  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        if (grid_.inside[id])
          ++inside_count;
        if (i + 1 < shape.nx) {
          if (grid_.inside[id] != grid_.inside[id + 1])
            ++audit.face_cross_x;
          if (grid_.eps_x.size() == shape.n) {
            add_face_eps(grid_.eps_x[id]);
            add_axis_face_eps(0, grid_.eps_x[id]);
          }
        }
        if (j + 1 < shape.ny) {
          if (grid_.inside[id] != grid_.inside[id + shape.nx])
            ++audit.face_cross_y;
          if (grid_.eps_y.size() == shape.n) {
            add_face_eps(grid_.eps_y[id]);
            add_axis_face_eps(1, grid_.eps_y[id]);
          }
        }
        if (k + 1 < shape.nz) {
          if (grid_.inside[id] != grid_.inside[id + xy])
            ++audit.face_cross_z;
          if (grid_.eps_z.size() == shape.n) {
            add_face_eps(grid_.eps_z[id]);
            add_axis_face_eps(2, grid_.eps_z[id]);
          }
        }

        const double rho = grid_.rho[id];
        const double x = shape.ox + static_cast<double>(i) * shape.h;
        const double y = shape.oy + static_cast<double>(j) * shape.h;
        const double z = shape.oz + static_cast<double>(k) * shape.h;
        audit.rho_sum += rho;
        audit.rho_l1 += std::abs(rho);
        audit.rho_l2 += rho * rho;
        audit.rho_mx += rho * x;
        audit.rho_my += rho * y;
        audit.rho_mz += rho * z;
      }
    }
  }
  audit.vdw_volume_estimate_A3 =
      static_cast<double>(inside_count) * voxel_volume;
  audit.eps_face_min = face_count > 0 ? eps_face_min : 0.0;
  audit.eps_face_max = face_count > 0 ? eps_face_max : 0.0;
  audit.eps_face_mean =
      face_count > 0 ? eps_face_sum / static_cast<double>(face_count) : 0.0;
  audit.eps_face_count_x = eps_axis_count[0];
  audit.eps_face_count_y = eps_axis_count[1];
  audit.eps_face_count_z = eps_axis_count[2];
  audit.internal_face_count_x = internal_axis_count[0];
  audit.internal_face_count_y = internal_axis_count[1];
  audit.internal_face_count_z = internal_axis_count[2];
  audit.eps_face_mean_x =
      eps_axis_count[0] > 0
          ? eps_axis_sum[0] / static_cast<double>(eps_axis_count[0])
          : 0.0;
  audit.eps_face_mean_y =
      eps_axis_count[1] > 0
          ? eps_axis_sum[1] / static_cast<double>(eps_axis_count[1])
          : 0.0;
  audit.eps_face_mean_z =
      eps_axis_count[2] > 0
          ? eps_axis_sum[2] / static_cast<double>(eps_axis_count[2])
          : 0.0;
  audit.rho_l2 = std::sqrt(audit.rho_l2);

  double denom_sum = 0.0;
  std::size_t denom_count = 0;
  double denom_min = std::numeric_limits<double>::max();
  double denom_max = 0.0;
  if (grid_.denom.size() == shape.n) {
    for (double value : grid_.denom) {
      if (!std::isfinite(value))
        continue;
      denom_min = std::min(denom_min, value);
      denom_max = std::max(denom_max, value);
      denom_sum += value;
      ++denom_count;
    }
  }
  audit.denom_min = denom_count > 0 ? denom_min : 0.0;
  audit.denom_max = denom_count > 0 ? denom_max : 0.0;
  audit.denom_mean =
      denom_count > 0 ? denom_sum / static_cast<double>(denom_count) : 0.0;

  if (grid_.screen.size() == shape.n) {
    for (double value : grid_.screen) {
      if (!std::isfinite(value))
        continue;
      audit.screen_sum += value;
      audit.screen_max = std::max(audit.screen_max, std::abs(value));
    }
  }
  return audit;
}

AnalyzerImpl::IscmAudit AnalyzerImpl::compute_iscm_audit(
    const GridShape &shape, const std::vector<Atom> &atoms,
    const AlignedDoubleVector &solvent_phi,
    const AlignedDoubleVector &solvent_screen,
    const std::vector<unsigned char> &solvent_inside,
    const AlignedDoubleVector &solvent_eps_x,
    const AlignedDoubleVector &solvent_eps_y,
    const AlignedDoubleVector &solvent_eps_z,
    double grid_vacuum_phase_kcal_mol) const {
  IscmAudit audit;
  if (args_.test_pqr.empty()) {
    audit.status = "requires_test_pqr";
    return audit;
  }
  if (args_.dielectric_model != "two-dielectric" || uses_gaussian_model()) {
    audit.status = "unsupported_dielectric_model";
    return audit;
  }
  if (args_.model != "linear" && !delphi_nwt_parity_active()) {
    audit.status = "linear_only";
    return audit;
  }
  if (shape.n == 0 || shape.h <= 0.0 || atoms.empty() ||
      solvent_phi.size() != shape.n || solvent_inside.size() != shape.n ||
      grid_.rho.size() != shape.n) {
    audit.status = "missing_field";
    return audit;
  }

  const double kt_kj = kKBKJPerMolK * std::max(1.0, args_.temp);
  const double kt_kcal = kKBKcalPerMolK * std::max(1.0, args_.temp);
  const double epkt = kDelphiEpk / std::max(1.0, args_.temp);
  const double potential_scale = nonlinear_alpha();
  const double field_native_scale = field_to_native_factor();
  const double grid_scale = 1.0 / std::max(1.0e-12, shape.h);
  const double induced_scale =
      kDelphiReactionFieldEnergyFactor /
      (2.0 * std::max(1.0e-12, grid_scale) * std::max(1.0e-12, epkt));
  const std::size_t plane = shape.nx * shape.ny;
  const bool delphi_vdw_boundary =
      args_.surface == "delphi-vdw" && solvent_eps_x.size() == shape.n &&
      solvent_eps_y.size() == shape.n && solvent_eps_z.size() == shape.n;
  const bool native_iscm_positions =
      args_.surface == "delphi-vdw" &&
      last_iscm_surface_valid_.size() == shape.n &&
      last_iscm_surface_positions_.size() == shape.n;
  std::unordered_map<std::size_t, std::array<double, 3>> oracle_iscm_positions;
  if (std::filesystem::exists("pbe_oracle_iscm_full.csv")) {
    std::ifstream oracle("pbe_oracle_iscm_full.csv");
    std::string line;
    if (oracle.is_open() && std::getline(oracle, line)) {
      const auto header = csv_header_map(split_csv_simple(line));
      while (std::getline(oracle, line)) {
        if (trim_copy(line).empty())
          continue;
        const auto fields = split_csv_simple(line);
        std::size_t oi = 0, oj = 0, ok = 0;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        if (!csv_index(fields, header, "i", oi) ||
            !csv_index(fields, header, "j", oj) ||
            !csv_index(fields, header, "k", ok) ||
            !csv_double(fields, header, {"surface_x", "surface_x_A"}, sx) ||
            !csv_double(fields, header, {"surface_y", "surface_y_A"}, sy) ||
            !csv_double(fields, header, {"surface_z", "surface_z_A"}, sz))
          continue;
        if (oi >= shape.nx || oj >= shape.ny || ok >= shape.nz)
          continue;
        oracle_iscm_positions[idx3(oi, oj, ok, shape.nx, shape.ny)] = {sx, sy,
                                                                       sz};
      }
    }
  }
  audit.position_source =
      native_iscm_positions ? "native_delphi_vdw_scaled" : "grid_point";
  audit.native_independent = true;
  double position_delta_sq = 0.0;
  std::size_t position_delta_count = 0;
  struct IscmPositionDiff {
    std::size_t id{0};
    std::size_t i{0};
    std::size_t j{0};
    std::size_t k{0};
    double native_x{0.0};
    double native_y{0.0};
    double native_z{0.0};
    double oracle_x{0.0};
    double oracle_y{0.0};
    double oracle_z{0.0};
    double delta{0.0};
  };
  std::vector<IscmPositionDiff> position_diffs;
  if (args_.parity_dump == "all" && !oracle_iscm_positions.empty())
    position_diffs.reserve(oracle_iscm_positions.size());
  auto face_is_solute = [&](double value) {
    return std::abs(value - args_.int_diel) <= std::abs(value - args_.ext_diel);
  };

  std::vector<IscmSample> deterministic_samples;
  std::vector<IscmSample> top_samples;
  auto add_sample = [&](IscmSample sample) {
    if (deterministic_samples.size() < 16) {
      sample.kind = "boundary_sample";
      sample.rank = deterministic_samples.size() + 1;
      deterministic_samples.push_back(sample);
    }
    sample.kind = "top_energy";
    top_samples.push_back(sample);
    std::sort(top_samples.begin(), top_samples.end(),
              [](const IscmSample &a, const IscmSample &b) {
                return std::abs(a.energy_contribution_kt) >
                       std::abs(b.energy_contribution_kt);
              });
    if (top_samples.size() > 32)
      top_samples.resize(32);
  };

  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const unsigned char center_inside = solvent_inside[id];
        bool dielectric_boundary = false;
        if (delphi_vdw_boundary) {
          const double faces[6] = {
              solvent_eps_x[id], solvent_eps_x[id - 1],
              solvent_eps_y[id], solvent_eps_y[id - shape.nx],
              solvent_eps_z[id], solvent_eps_z[id - plane]};
          int solute_faces = 0;
          int solvent_faces = 0;
          for (double face_eps : faces) {
            if (!std::isfinite(face_eps))
              continue;
            if (face_is_solute(face_eps))
              ++solute_faces;
            else
              ++solvent_faces;
          }
          dielectric_boundary = solute_faces > 0 && solvent_faces > 0;
          if (dielectric_boundary) {
            ++audit.delphi_face_crossings;
            audit.delphi_eps_face_matches +=
                static_cast<std::size_t>(solute_faces);
          }
        } else {
          dielectric_boundary =
              solvent_inside[id - 1] != center_inside ||
              solvent_inside[id + 1] != center_inside ||
              solvent_inside[id - shape.nx] != center_inside ||
              solvent_inside[id + shape.nx] != center_inside ||
              solvent_inside[id - plane] != center_inside ||
              solvent_inside[id + plane] != center_inside;
        }
        if (!dielectric_boundary)
          continue;

        const double phi_center =
            potential_scale * solvent_phi[id] * field_native_scale;
        const double phi_neigh =
            potential_scale *
            (solvent_phi[id - 1] + solvent_phi[id + 1] +
             solvent_phi[id - shape.nx] + solvent_phi[id + shape.nx] +
             solvent_phi[id - plane] + solvent_phi[id + plane]) /
            6.0 * field_native_scale;
        double charge_adjustment = 0.0;
        if (std::abs(grid_.rho[id]) > 1.0e-14) {
          ++audit.charged_boundary_points;
          const double solvent_flag = center_inside ? 0.0 : 1.0;
          const double salt_diag = solvent_screen.size() == shape.n
                                       ? solvent_screen[id] * solvent_flag
                                       : 0.0;
          const double denom =
              6.0 * args_.int_diel / std::max(1.0e-12, epkt) + salt_diag;
          if (std::abs(denom) > 1.0e-30)
            charge_adjustment = grid_.rho[id] * 4.0 * kPi * grid_scale / denom;
        }

        const double phi_effective = phi_center - phi_neigh - charge_adjustment;
        const double induced_charge = phi_effective * induced_scale;
        double x = shape.ox + static_cast<double>(i) * shape.h;
        double y = shape.oy + static_cast<double>(j) * shape.h;
        double z = shape.oz + static_cast<double>(k) * shape.h;
        if (native_iscm_positions && last_iscm_surface_valid_[id] != 0U) {
          const auto &native_pos = last_iscm_surface_positions_[id];
          x = native_pos[0];
          y = native_pos[1];
          z = native_pos[2];
        }
        const auto oracle_pos_it = oracle_iscm_positions.find(id);
        if (oracle_pos_it != oracle_iscm_positions.end()) {
          const double dx = x - oracle_pos_it->second[0];
          const double dy = y - oracle_pos_it->second[1];
          const double dz = z - oracle_pos_it->second[2];
          const double delta_sq = dx * dx + dy * dy + dz * dz;
          position_delta_sq += delta_sq;
          ++position_delta_count;
          if (args_.parity_dump == "all") {
            position_diffs.push_back(
                {id, i, j, k, x, y, z, oracle_pos_it->second[0],
                 oracle_pos_it->second[1], oracle_pos_it->second[2],
                 std::sqrt(delta_sq)});
          }
        }
        double atom_potential_sum = 0.0;
        for (const Atom &atom : atoms) {
          const double dx = x - atom.x;
          const double dy = y - atom.y;
          const double dz = z - atom.z;
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 > 1.0e-24)
            atom_potential_sum += atom.q / std::sqrt(r2);
        }
        const double contribution_kt =
            0.5 * epkt * induced_charge * atom_potential_sum;
        audit.water_phase_kt += contribution_kt;
        audit.induced_charge_sum += induced_charge;
        audit.induced_charge_l1 += std::abs(induced_charge);
        audit.induced_charge_l2 += induced_charge * induced_charge;
        audit.induced_charge_max =
            std::max(audit.induced_charge_max, std::abs(induced_charge));
        ++audit.boundary_points;

        IscmSample sample;
        sample.i = i;
        sample.j = j;
        sample.k = k;
        sample.x = x;
        sample.y = y;
        sample.z = z;
        sample.phi_center = phi_center;
        sample.phi_neighbor_average = phi_neigh;
        sample.charge_adjustment = charge_adjustment;
        sample.induced_charge = induced_charge;
        sample.atom_potential_sum = atom_potential_sum;
        sample.energy_contribution_kt = contribution_kt;
        add_sample(sample);
      }
    }
  }

  audit.induced_charge_l2 = std::sqrt(audit.induced_charge_l2);
  audit.position_oracle_matches = position_delta_count;
  audit.position_rms_vs_oracle =
      position_delta_count > 0
          ? std::sqrt(position_delta_sq /
                      static_cast<double>(position_delta_count))
          : 0.0;
  if (!position_diffs.empty()) {
    std::sort(position_diffs.begin(), position_diffs.end(),
              [](const IscmPositionDiff &a, const IscmPositionDiff &b) {
                return a.delta > b.delta;
              });
    std::ofstream diff_out("pbe_iscm_position_diff.csv",
                           std::ios::out | std::ios::trunc);
    if (diff_out.is_open()) {
      diff_out << "rank,id,i,j,k,native_x,native_y,native_z,oracle_x,"
                  "oracle_y,oracle_z,delta_A\n";
      diff_out << std::setprecision(12);
      std::size_t rank = 0;
      for (const auto &diff : position_diffs) {
        if (++rank > 512 && diff.delta <= 1.0e-10)
          break;
        diff_out << rank << ',' << diff.id << ',' << diff.i << ',' << diff.j
                 << ',' << diff.k << ',' << diff.native_x << ','
                 << diff.native_y << ',' << diff.native_z << ','
                 << diff.oracle_x << ',' << diff.oracle_y << ','
                 << diff.oracle_z << ',' << diff.delta << '\n';
      }
    }
  }
  audit.water_phase_kj_mol = audit.water_phase_kt * kt_kj;
  audit.water_phase_kcal_mol = audit.water_phase_kt * kt_kcal;
  audit.vacuum_phase_kcal_mol = grid_vacuum_phase_kcal_mol;
  audit.vacuum_phase_kj_mol = grid_vacuum_phase_kcal_mol * kKcalToKJ;
  audit.vacuum_phase_kt =
      kt_kcal > 0.0 ? grid_vacuum_phase_kcal_mol / kt_kcal : 0.0;
  audit.water_minus_vacuum_kt = audit.water_phase_kt - audit.vacuum_phase_kt;
  audit.water_minus_vacuum_kj_mol =
      audit.water_phase_kj_mol - audit.vacuum_phase_kj_mol;
  audit.water_minus_vacuum_kcal_mol =
      audit.water_phase_kcal_mol - audit.vacuum_phase_kcal_mol;
  // DelPhi's two-dielectric VDW ISCM path finalizes corrected RF from the
  // water induced-surface-charge phase alone. Keep the vacuum grid phase in
  // the audit so convention differences remain inspectable.
  audit.corrected_kt = audit.water_phase_kt;
  audit.corrected_kj_mol = audit.corrected_kt * kt_kj;
  audit.corrected_kcal_mol = audit.corrected_kt * kt_kcal;
  audit.finalization_mode = "water_only";
  audit.status = audit.boundary_points > 0 ? "ok" : "no_boundary_points";

  audit.samples = deterministic_samples;
  for (std::size_t idx = 0; idx < top_samples.size(); ++idx) {
    top_samples[idx].rank = idx + 1;
    audit.samples.push_back(top_samples[idx]);
  }
  return audit;
}

void AnalyzerImpl::copy_iscm_audit_to_solve(const IscmAudit &audit,
                                            ReactionSolve &solve) const {
  solve.iscm_boundary_points = audit.boundary_points;
  solve.iscm_charged_boundary_points = audit.charged_boundary_points;
  solve.iscm_induced_charge_sum = audit.induced_charge_sum;
  solve.iscm_induced_charge_l1 = audit.induced_charge_l1;
  solve.iscm_induced_charge_l2 = audit.induced_charge_l2;
  solve.iscm_induced_charge_max = audit.induced_charge_max;
  solve.iscm_water_phase_kcal_mol = audit.water_phase_kcal_mol;
  solve.iscm_water_phase_kj_mol = audit.water_phase_kj_mol;
  solve.iscm_water_phase_kt = audit.water_phase_kt;
  solve.iscm_vacuum_phase_kcal_mol = audit.vacuum_phase_kcal_mol;
  solve.iscm_vacuum_phase_kj_mol = audit.vacuum_phase_kj_mol;
  solve.iscm_vacuum_phase_kt = audit.vacuum_phase_kt;
  solve.iscm_corrected_kcal_mol = audit.corrected_kcal_mol;
  solve.iscm_corrected_kj_mol = audit.corrected_kj_mol;
  solve.iscm_corrected_kt = audit.corrected_kt;
  solve.iscm_water_minus_vacuum_kcal_mol = audit.water_minus_vacuum_kcal_mol;
  solve.iscm_water_minus_vacuum_kj_mol = audit.water_minus_vacuum_kj_mol;
  solve.iscm_water_minus_vacuum_kt = audit.water_minus_vacuum_kt;
  solve.iscm_finalization_mode = audit.finalization_mode;
  solve.iscm_position_source = audit.position_source;
  solve.iscm_position_rms_vs_oracle = audit.position_rms_vs_oracle;
  solve.iscm_position_oracle_matches = audit.position_oracle_matches;
  solve.iscm_native_independent = audit.native_independent;
  solve.iscm_status = audit.status;
  solve.delphi_face_crossings = audit.delphi_face_crossings;
  solve.delphi_eps_face_matches = audit.delphi_eps_face_matches;
  solve.delphi_boundary_grid_points = audit.boundary_points;
  solve.delphi_charged_boundary_points = audit.charged_boundary_points;
  solve.delphi_iscm_status = audit.status;
}

AnalyzerImpl::NonlinearEnergyAudit AnalyzerImpl::compute_nonlinear_energy_audit(
    const GridShape &shape, const AlignedDoubleVector &solvent_phi,
    const AlignedDoubleVector &solvent_screen) const {
  NonlinearEnergyAudit audit;
  if (args_.model != "nonlinear") {
    audit.status = "linear_model";
    return audit;
  }
  if (args_.salt <= 0.0) {
    audit.status = "linearized_no_salt";
    return audit;
  }
  if (shape.n == 0 || solvent_phi.size() != shape.n) {
    audit.status = "missing_field";
    return audit;
  }

  const double ion_concentration = std::max(0.0, args_.salt);
  const double coeffs[5] = {
      -2.0 * ion_concentration,  0.0, -ion_concentration / 3.0, 0.0,
      -ion_concentration / 60.0,
  };
  const double d_coeffs[5] = {
      -coeffs[0] / 2.0, -coeffs[1] / 3.0, -coeffs[2] / 4.0,
      -coeffs[3] / 5.0, -coeffs[4] / 6.0,
  };
  const double alpha = nonlinear_alpha();
  const double cell_volume = shape.h * shape.h * shape.h;
  constexpr double kDelphiEnergyUnitConversion = 0.0006023;
  double rho_phi_accum = 0.0;
  double osmotic_accum = 0.0;

  for (std::size_t k = 0; k < shape.nz; ++k) {
    const double cut_x_dummy = 1.0;
    (void)cut_x_dummy;
    const double cut_z = (k == 0 || k + 1 == shape.nz) ? 0.5 : 1.0;
    for (std::size_t j = 0; j < shape.ny; ++j) {
      const double cut_y = cut_z * ((j == 0 || j + 1 == shape.ny) ? 0.5 : 1.0);
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const bool ion_accessible = solvent_screen.size() == shape.n
                                        ? solvent_screen[id] > 0.0
                                        : (id < grid_.ion_accessible.size() &&
                                           grid_.ion_accessible[id] != 0U);
        if (!ion_accessible)
          continue;
        const double cut_edge =
            cut_y * ((i == 0 || i + 1 == shape.nx) ? 0.5 : 1.0);
        const double phi = delphi_field_units_active()
                               ? solvent_phi[id]
                               : alpha * solvent_phi[id];
        const double charge_density =
            ((((coeffs[4] * phi + coeffs[3]) * phi + coeffs[2]) * phi +
              coeffs[1]) *
                 phi +
             coeffs[0]);
        const double osmotic_density =
            ((((d_coeffs[4] * phi + d_coeffs[3]) * phi + d_coeffs[2]) * phi +
              d_coeffs[1]) *
                 phi +
             d_coeffs[0]);
        const double ionic_charge =
            cut_edge * charge_density * phi * cell_volume;
        rho_phi_accum -= ionic_charge * phi;
        osmotic_accum += cut_edge * osmotic_density * phi * phi;
        ++audit.sampled_points;
      }
    }
  }

  double electrostatic_stress =
      rho_phi_accum * 0.5 * kDelphiEnergyUnitConversion;
  double osmotic_pressure =
      osmotic_accum * (-kDelphiEnergyUnitConversion * cell_volume);
  // pyDelPhi negates both nonlinear terms before adding them to the final
  // corrected RF energy.
  electrostatic_stress *= -1.0;
  osmotic_pressure *= -1.0;

  audit.electrostatic_stress_kt = electrostatic_stress;
  audit.osmotic_pressure_kt = osmotic_pressure;
  audit.total_correction_kt = electrostatic_stress + osmotic_pressure;
  const double kt_kj = kKBKJPerMolK * std::max(1.0, args_.temp);
  const double kt_kcal = kKBKcalPerMolK * std::max(1.0, args_.temp);
  audit.electrostatic_stress_kj_mol = audit.electrostatic_stress_kt * kt_kj;
  audit.electrostatic_stress_kcal_mol = audit.electrostatic_stress_kt * kt_kcal;
  audit.osmotic_pressure_kj_mol = audit.osmotic_pressure_kt * kt_kj;
  audit.osmotic_pressure_kcal_mol = audit.osmotic_pressure_kt * kt_kcal;
  audit.total_correction_kj_mol = audit.total_correction_kt * kt_kj;
  audit.total_correction_kcal_mol = audit.total_correction_kt * kt_kcal;
  audit.status = audit.sampled_points > 0 ? "ok" : "no_ion_accessible_points";
  return audit;
}

void AnalyzerImpl::copy_nonlinear_energy_audit_to_solve(
    const NonlinearEnergyAudit &audit, ReactionSolve &solve) const {
  solve.nonlinear_stress_kt = audit.electrostatic_stress_kt;
  solve.nonlinear_stress_kj_mol = audit.electrostatic_stress_kj_mol;
  solve.nonlinear_stress_kcal_mol = audit.electrostatic_stress_kcal_mol;
  solve.nonlinear_osmotic_kt = audit.osmotic_pressure_kt;
  solve.nonlinear_osmotic_kj_mol = audit.osmotic_pressure_kj_mol;
  solve.nonlinear_osmotic_kcal_mol = audit.osmotic_pressure_kcal_mol;
  solve.nonlinear_energy_correction_kt = audit.total_correction_kt;
  solve.nonlinear_energy_correction_kj_mol = audit.total_correction_kj_mol;
  solve.nonlinear_energy_correction_kcal_mol = audit.total_correction_kcal_mol;
  solve.nonlinear_energy_status = audit.status;
}

void AnalyzerImpl::update_newton_derivative_audit(
    const GridShape &shape, const AlignedDoubleVector &source,
    const AlignedDoubleVector &delta, double alpha, double clip,
    double screen_scale, ParityAudit &audit) const {
  if (!args_.test_newton_derivative || shape.n == 0 ||
      grid_.phi.size() != shape.n || source.size() != shape.n ||
      delta.size() != shape.n || grid_.rho.size() != shape.n)
    return;

  double max_delta = 0.0;
  for (double value : delta)
    max_delta = std::max(max_delta, std::abs(value));
  if (max_delta <= 1.0e-14)
    return;

  const std::size_t xy = shape.nx * shape.ny;
  const double rhs = nonlinear_rhs_scale(shape, alpha);
  const double eps_step = 1.0e-6 / std::max(1.0, max_delta);
  const std::size_t interior =
      shape.nx > 2 && shape.ny > 2 && shape.nz > 2
          ? (shape.nx - 2) * (shape.ny - 2) * (shape.nz - 2)
          : 0;
  const std::size_t stride = std::max<std::size_t>(1, interior / 4096);
  std::size_t ordinal = 0;
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i, ++ordinal) {
        if (ordinal % stride != 0)
          continue;
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double eps_sum =
            std::max(1.0e-12, grid_.denom[id] - grid_.screen[id]);
        const double diag =
            std::max(1.0e-12, eps_sum + (grid_.screen[id] * screen_scale) *
                                            nonlinear_source_derivative(
                                                grid_.phi[id], alpha, clip));
        const double neighbor_delta_sum =
            grid_.eps_x[id] * delta[id + 1] +
            grid_.eps_x[id - 1] * delta[id - 1] +
            grid_.eps_y[id] * delta[id + shape.nx] +
            grid_.eps_y[id - shape.nx] * delta[id - shape.nx] +
            grid_.eps_z[id] * delta[id + xy] +
            grid_.eps_z[id - xy] * delta[id - xy];
        const double analytic = diag * delta[id] - neighbor_delta_sum;

        const double neighbor_sum =
            grid_.eps_x[id] * grid_.phi[id + 1] +
            grid_.eps_x[id - 1] * grid_.phi[id - 1] +
            grid_.eps_y[id] * grid_.phi[id + shape.nx] +
            grid_.eps_y[id - shape.nx] * grid_.phi[id - shape.nx] +
            grid_.eps_z[id] * grid_.phi[id + xy] +
            grid_.eps_z[id - xy] * grid_.phi[id - xy];
        const double f0 = eps_sum * grid_.phi[id] - neighbor_sum +
                          (grid_.screen[id] * screen_scale) * source[id] -
                          rhs * grid_.rho[id];

        const auto trial_phi = [&](std::size_t probe) {
          return grid_.phi[probe] + eps_step * delta[probe];
        };
        const double trial_neighbor_sum =
            grid_.eps_x[id] * trial_phi(id + 1) +
            grid_.eps_x[id - 1] * trial_phi(id - 1) +
            grid_.eps_y[id] * trial_phi(id + shape.nx) +
            grid_.eps_y[id - shape.nx] * trial_phi(id - shape.nx) +
            grid_.eps_z[id] * trial_phi(id + xy) +
            grid_.eps_z[id - xy] * trial_phi(id - xy);
        const double trial_center = trial_phi(id);
        const double trial_source =
            nonlinear_source_value(trial_center, alpha, clip);
        const double f1 = eps_sum * trial_center - trial_neighbor_sum +
                          (grid_.screen[id] * screen_scale) * trial_source -
                          rhs * grid_.rho[id];
        const double finite_difference = (f1 - f0) / eps_step;
        const double abs_error = std::abs(analytic - finite_difference);
        const double rel_error =
            abs_error / std::max(1.0, std::abs(finite_difference));
        audit.derivative_max_abs =
            std::max(audit.derivative_max_abs, abs_error);
        audit.derivative_max_rel =
            std::max(audit.derivative_max_rel, rel_error);
        ++audit.derivative_samples;
      }
    }
  }
}


} // namespace TitanPBE

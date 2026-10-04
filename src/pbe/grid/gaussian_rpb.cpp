#include "tabi/analyzer_impl.h"

#include "pbe/grid/grid_common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstddef>
#include <utility>
#include <vector>

namespace TitanPBE {
namespace {

bool surface_uses_gaussian(const std::string &surface) noexcept {
  return surface == "gaussian" || surface == "gaussian-cutoff";
}

double pow_nonnegative(double value, double exponent) {
  const double x = std::max(0.0, value);
  if (std::abs(exponent - 1.0) <= 1.0e-12)
    return x;
  if (std::abs(exponent - 2.0) <= 1.0e-12)
    return x * x;
  if (std::abs(exponent - 0.5) <= 1.0e-12)
    return std::sqrt(x);
  return std::pow(x, exponent);
}

std::string rpb_csv_escape(const std::string &value) {
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

template <typename Shape>
bool rpb_is_boundary(std::size_t i, std::size_t j, std::size_t k,
                     const Shape &shape) noexcept {
  return i == 0 || j == 0 || k == 0 || i + 1 == shape.nx ||
         j + 1 == shape.ny || k + 1 == shape.nz;
}

} // namespace

bool AnalyzerImpl::uses_gaussian_model() const {
  return args_.dielectric_model == "gaussian" ||
         surface_uses_gaussian(args_.surface) ||
         args_.charge_assignment == "gaussian" || args_.biomodel == "rpbe";
}

double AnalyzerImpl::gaussian_influence_radius(const std::vector<Atom> &atoms) const {
  const double sigma = std::max(1.0e-12, args_.gaussian_sigma);
  const double exclusion_radius =
      std::max(std::max(0.0, args_.probe_radius), std::max(0.0, args_.ion_radius));
  double max_atom_radius = 0.0;
  for (const Atom &atom : atoms)
    max_atom_radius = std::max(max_atom_radius, std::max(1.0e-12, atom.radius));
  double influence = max_atom_radius + exclusion_radius;
  for (const Atom &atom : atoms) {
    const double radius = std::max(1.0e-12, atom.radius);
    const double gaussian_extent = 3.0 * radius * sigma;
    const double physical_extent = radius + exclusion_radius;
    influence = std::max(influence, std::max(gaussian_extent, physical_extent));
  }
  return std::max(max_atom_radius, influence);
}

double AnalyzerImpl::gaussian_density_at(const std::vector<Atom> &atoms, double x,
                                     double y, double z) const {
  const double exponent = std::max(1.0e-12, args_.gaussian_exp);
  const double sigma = std::max(1.0e-12, args_.gaussian_sigma);
  const double influence = gaussian_influence_radius(atoms);
  const double influence2 = influence * influence;
  double solvent_complement = 1.0;
  for (const Atom &atom : atoms) {
    const double r0 = std::max(1.0e-12, effective_surface_radius(atom));
    const double dx = x - atom.x;
    const double dy = y - atom.y;
    const double dz = z - atom.z;
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > influence2)
      continue;
    const double scaled2 = r2 / (sigma * sigma * r0 * r0);
    const double atom_density =
        std::clamp(std::exp(-std::pow(scaled2, exponent)), 0.0, 1.0);
    solvent_complement *= (1.0 - atom_density);
    if (solvent_complement <= 1.0e-15)
      return 1.0;
  }
  return std::clamp(1.0 - solvent_complement, 0.0, 1.0);
}

double AnalyzerImpl::gaussian_dielectric(double density, double eps_in,
                                     double eps_out) const {
  const double d = std::clamp(density, 0.0, 1.0);
  if (args_.surface == "gaussian-cutoff") {
    const double cutoff =
        args_.surface_cutoff == "auto" ? args_.density_cutoff
                                       : args_.surface_cutoff_value;
    if (d < cutoff)
      return eps_out;
    const double regional_eps =
        (args_.gap_diel > eps_in && args_.gap_diel < eps_out) ? args_.gap_diel
                                                              : eps_out;
    return d * eps_in + (1.0 - d) * regional_eps;
  }
  if (args_.surface == "gaussian") {
    const double srf_exp =
        4.0 / std::max(1.0e-12, args_.gaussian_exp);
    double surface = 0.0;
    if (d > 1.0e-12) {
      const double term = std::pow(1.0 / d - 1.0, srf_exp);
      surface = 1.0 / (1.0 + term);
    }
    const double gaussian_eps =
        eps_in + (args_.gap_diel - eps_in) * (1.0 - d);
    return gaussian_eps * surface + (1.0 - surface) * eps_out;
  }
  return eps_out - (eps_out - eps_in) * d;
}

std::vector<AnalyzerImpl::Atom>
AnalyzerImpl::adjusted_surface_atoms(const std::vector<Atom> &atoms) const {
  std::vector<Atom> adjusted = atoms;
  if (effective_radius_convention() != "delphi-bondi")
    return adjusted;
  for (Atom &atom : adjusted)
    atom.radius = effective_surface_radius(atom);
  return adjusted;
}

AnalyzerImpl::RpbAudit AnalyzerImpl::compute_rpb_source_audit(
    const GridShape &shape, const std::vector<Atom> &atoms,
    TitanExecutor &executor,
    const AlignedDoubleVector &eps_x, const AlignedDoubleVector &eps_y,
    const AlignedDoubleVector &eps_z, AlignedDoubleVector *source_out,
    AlignedDoubleVector *coulomb_out, double phase_eps_out) const {
  RpbAudit audit;
  if (args_.biomodel != "rpbe") {
    audit.status = "off";
    return audit;
  }
  if (shape.n == 0 || atoms.empty() || eps_x.size() != shape.n ||
      eps_y.size() != shape.n || eps_z.size() != shape.n) {
    audit.status = "missing_maps";
    return audit;
  }

  const std::size_t xy = shape.nx * shape.ny;
  const double inv_2h = 0.5 / std::max(1.0e-12, shape.h);
  const double indi_scaled =
      std::max(1.0e-30, args_.int_diel / std::max(1.0e-30, delphi_epkt()));
  struct RpbSourceAtom {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double q{0.0};
    double radius{1.0};
    double inv_sigma_radius2{0.0};
    double gaussian_factor{0.0};
    double gaussian_cutoff2{0.0};
    bool charged{false};
    bool gaussian_ready{false};
  };
  const std::vector<Atom> adjusted_atoms = adjusted_surface_atoms(atoms);
  std::vector<RpbSourceAtom> source_atoms;
  source_atoms.reserve(adjusted_atoms.size());
  const bool analytic_gaussian =
      args_.surface == "gaussian" &&
      last_gaussian_density_center_.size() == shape.n &&
      last_gaussian_surface_.size() == shape.n &&
      last_gaussian_eps_center_.size() == shape.n;
  const double gaussian_exponent = std::max(1.0e-12, args_.gaussian_exp);
  const double gaussian_sigma = std::max(1.0e-12, args_.gaussian_sigma);
  const double approx = 1.0e-12;
  const double density_cutoff_arg = -std::log(approx);
  const double density_cutoff_norm =
      pow_nonnegative(density_cutoff_arg, 1.0 / gaussian_exponent);
  for (const Atom &atom : adjusted_atoms) {
    RpbSourceAtom prepared;
    prepared.x = atom.x;
    prepared.y = atom.y;
    prepared.z = atom.z;
    prepared.q = atom.q;
    prepared.radius = std::max(1.0e-12, atom.radius);
    prepared.charged = std::abs(atom.q) > 0.0;
    const double sigma_radius = gaussian_sigma * prepared.radius;
    const double sigma_radius2 = sigma_radius * sigma_radius;
    if (sigma_radius2 > 1.0e-12) {
      prepared.inv_sigma_radius2 = 1.0 / sigma_radius2;
      prepared.gaussian_factor =
          (2.0 * gaussian_exponent) /
          pow_nonnegative(sigma_radius2, gaussian_exponent);
      prepared.gaussian_cutoff2 = sigma_radius2 * density_cutoff_norm;
      prepared.gaussian_ready = true;
    }
    source_atoms.push_back(prepared);
  }
  const double surface_exponent = 4.0 / gaussian_exponent;
  const double diff_gap_indi = args_.gap_diel - args_.int_diel;
  const double epsout = std::isfinite(phase_eps_out) ? phase_eps_out
                                                     : args_.ext_diel;
  const std::size_t workers = effective_workers(executor, shape.nz);
  std::vector<double> local_coulomb_sq(workers, 0.0);
  std::vector<double> local_grad_coulomb_sq(workers, 0.0);
  std::vector<double> local_grad_epsilon_sq(workers, 0.0);
  std::vector<double> local_source_sq(workers, 0.0);
  std::vector<double> local_source_max(workers, 0.0);
  std::vector<std::size_t> local_count(workers, 0);
  if (source_out != nullptr)
    source_out->assign(shape.n, 0.0);
  if (coulomb_out != nullptr)
    coulomb_out->assign(shape.n, 0.0);

  auto center_eps = [&](std::size_t id) {
    return (eps_x[id] + eps_y[id] + eps_z[id]) / 3.0;
  };

  executor.parallel_for(shape.nz, workers, 1,
                        [&](std::size_t worker, std::size_t z_begin,
                            std::size_t z_end) {
    double coulomb_sq = 0.0;
    double grad_coulomb_sq = 0.0;
    double grad_epsilon_sq = 0.0;
    double source_sq = 0.0;
    double source_max = 0.0;
    std::size_t count = 0;
    for (std::size_t k = z_begin; k < z_end; ++k) {
      const double z = shape.oz + static_cast<double>(k) * shape.h;
      for (std::size_t j = 0; j < shape.ny; ++j) {
        const double y = shape.oy + static_cast<double>(j) * shape.h;
        for (std::size_t i = 0; i < shape.nx; ++i) {
          const double x = shape.ox + static_cast<double>(i) * shape.h;
          double coulomb = 0.0;
          double gx = 0.0;
          double gy = 0.0;
          double gz = 0.0;
          double grad_epsin_x = 0.0;
          double grad_epsin_y = 0.0;
          double grad_epsin_z = 0.0;
          double grad_surface_x = 0.0;
          double grad_surface_y = 0.0;
          double grad_surface_z = 0.0;
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          const double total_density =
              analytic_gaussian
                  ? std::clamp(last_gaussian_density_center_[id], 0.0, 1.0)
                  : 0.0;
          const double surface =
              analytic_gaussian ? std::clamp(last_gaussian_surface_[id], 0.0,
                                             1.0)
                                : 0.0;
          double surface_derivative_factor = 0.0;
          if (analytic_gaussian && total_density > approx && surface > approx) {
            const double term = 1.0 / total_density - 1.0;
            if (std::abs(term) > approx ||
                std::abs(surface_exponent - std::round(surface_exponent)) <
                    1.0e-12) {
              surface_derivative_factor =
                  surface_exponent *
                  pow_nonnegative(term, surface_exponent - 1.0) *
                  surface * surface / (total_density * total_density);
            }
          }
          for (const RpbSourceAtom &atom : source_atoms) {
            const double dx = x - atom.x;
            const double dy = y - atom.y;
            const double dz = z - atom.z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (atom.charged && r2 > 1.0e-24) {
              const double r = std::sqrt(r2);
              const double inv = 1.0 / (r * indi_scaled);
              coulomb += atom.q * inv;
              const double grad_scale = atom.q / (r2 * r * indi_scaled);
              gx -= dx * grad_scale;
              gy -= dy * grad_scale;
              gz -= dz * grad_scale;
            }
            if (!analytic_gaussian)
              continue;
            if (r2 <= approx)
              continue;
            if (!atom.gaussian_ready)
              continue;
            if (r2 >= atom.gaussian_cutoff2)
              continue;
            const double normalized = r2 * atom.inv_sigma_radius2;
            const double g_arg = pow_nonnegative(normalized, gaussian_exponent);
            if (g_arg >= 700.0)
              continue;
            const double density = std::exp(-g_arg);
            if (density <= approx || std::abs(1.0 - density) <= approx)
              continue;
            const double density_factor =
                density * (1.0 - total_density) / (1.0 - density);
            const double dist_factor =
                pow_nonnegative(r2, gaussian_exponent - 1.0);
            const double base =
                atom.gaussian_factor * density_factor * dist_factor;
            const double eps_factor = diff_gap_indi * base;
            grad_epsin_x += dx * eps_factor;
            grad_epsin_y += dy * eps_factor;
            grad_epsin_z += dz * eps_factor;
            if (surface_derivative_factor != 0.0) {
              const double surface_factor = base * surface_derivative_factor;
              grad_surface_x -= surface_factor * dx;
              grad_surface_y -= surface_factor * dy;
              grad_surface_z -= surface_factor * dz;
            }
          }
          coulomb_sq += coulomb * coulomb;
          if (coulomb_out != nullptr)
            (*coulomb_out)[id] = coulomb;
          grad_coulomb_sq += gx * gx + gy * gy + gz * gz;
          if (i == 0 || j == 0 || k == 0 || i + 1 == shape.nx ||
              j + 1 == shape.ny || k + 1 == shape.nz)
            continue;
          double ex = 0.0;
          double ey = 0.0;
          double ez = 0.0;
          if (analytic_gaussian) {
            const double eps_center =
                args_.biomodel == "rpbe"
                    ? args_.int_diel +
                          (args_.gap_diel - args_.int_diel) *
                              (1.0 - total_density)
                    : last_gaussian_eps_center_[id];
            ex = grad_epsin_x * surface +
                 (eps_center - epsout) * grad_surface_x;
            ey = grad_epsin_y * surface +
                 (eps_center - epsout) * grad_surface_y;
            ez = grad_epsin_z * surface +
                 (eps_center - epsout) * grad_surface_z;
          } else {
            ex = (center_eps(id + 1) - center_eps(id - 1)) * inv_2h;
            ey = (center_eps(id + shape.nx) - center_eps(id - shape.nx)) *
                 inv_2h;
            ez = (center_eps(id + xy) - center_eps(id - xy)) * inv_2h;
          }
          const double source = ex * gx + ey * gy + ez * gz;
          if (source_out != nullptr)
            (*source_out)[id] = source;
          grad_epsilon_sq += ex * ex + ey * ey + ez * ez;
          source_sq += source * source;
          source_max = std::max(source_max, std::abs(source));
          ++count;
        }
      }
    }
    local_coulomb_sq[worker] += coulomb_sq;
    local_grad_coulomb_sq[worker] += grad_coulomb_sq;
    local_grad_epsilon_sq[worker] += grad_epsilon_sq;
    local_source_sq[worker] += source_sq;
    local_source_max[worker] = std::max(local_source_max[worker], source_max);
    local_count[worker] += count;
  });

  double coulomb_sq = 0.0;
  double grad_coulomb_sq = 0.0;
  double grad_epsilon_sq = 0.0;
  double source_sq = 0.0;
  double source_max = 0.0;
  std::size_t count = 0;
  for (std::size_t worker = 0; worker < workers; ++worker) {
    coulomb_sq += local_coulomb_sq[worker];
    grad_coulomb_sq += local_grad_coulomb_sq[worker];
    grad_epsilon_sq += local_grad_epsilon_sq[worker];
    source_sq += local_source_sq[worker];
    source_max = std::max(source_max, local_source_max[worker]);
    count += local_count[worker];
  }

  if (count == 0) {
    audit.status = "empty_interior";
    return audit;
  }
  const double denom = static_cast<double>(shape.n);
  audit.coulomb_norm = std::sqrt(coulomb_sq);
  audit.grad_coulomb_norm = std::sqrt(grad_coulomb_sq);
  audit.grad_epsilon_norm = std::sqrt(grad_epsilon_sq);
  audit.source_rms = std::sqrt(source_sq / denom);
  audit.source_max = source_max;
  audit.sampled_points = count;
  audit.status = std::isfinite(audit.source_rms) ? "ok" : "non_finite";
  return audit;
}

void AnalyzerImpl::maybe_write_rpb_source_audit(
    const FrameResult &result, const RpbAudit &audit) const {
  if (args_.biomodel != "rpbe")
    return;
  const std::string filename = "pbe_rpbe_source.csv";
  const bool need_header =
      !std::filesystem::exists(filename) ||
      std::filesystem::file_size(filename) == 0;
  std::ofstream out(filename, std::ios::out | std::ios::app);
  if (!out.is_open())
    return;
  if (need_header) {
    out << "trajectory_id,frame_index,global_index,time_ps,fixture,"
           "coulomb_norm,grad_coulomb_norm,grad_epsilon_norm,source_rms,"
           "source_max,sampled_points,status,final_energy_kJmol\n";
  }
  out << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << std::setprecision(10) << result.time_ps
      << ',' << rpb_csv_escape(args_.test_pqr) << ',' << audit.coulomb_norm
      << ',' << audit.grad_coulomb_norm << ',' << audit.grad_epsilon_norm
      << ',' << audit.source_rms << ',' << audit.source_max << ','
      << audit.sampled_points << ',' << rpb_csv_escape(audit.status) << ','
      << result.energy_kj_mol << '\n';
}

void AnalyzerImpl::maybe_write_rpb_source_top(
    const FrameResult &result, const GridShape &shape,
    const AlignedDoubleVector &source,
    const AlignedDoubleVector &coulomb) const {
  if (args_.biomodel != "rpbe" || args_.parity_dump == "off" ||
      source.size() != shape.n || shape.n == 0)
    return;

  std::vector<std::pair<double, std::size_t>> top;
  top.reserve(32);
  for (std::size_t id = 0; id < shape.n; ++id) {
    const double value = std::abs(source[id]);
    if (!std::isfinite(value) || value <= 0.0)
      continue;
    if (top.size() < 32) {
      top.push_back({value, id});
      continue;
    }
    auto min_it = std::min_element(top.begin(), top.end());
    if (min_it != top.end() && value > min_it->first)
      *min_it = {value, id};
  }
  if (top.empty())
    return;
  std::sort(top.begin(), top.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  const std::string filename = "pbe_rpbe_source_top.csv";
  const bool need_header =
      !std::filesystem::exists(filename) ||
      std::filesystem::file_size(filename) == 0;
  std::ofstream out(filename, std::ios::out | std::ios::app);
  if (!out.is_open())
    return;
  if (need_header) {
    out << "trajectory_id,frame_index,global_index,time_ps,rank,i,j,k,"
           "x_A,y_A,z_A,source,coulomb,density_center,surface,eps_center,"
           "is_boundary\n";
  }
  out << std::setprecision(12);
  const std::size_t xy = shape.nx * shape.ny;
  for (std::size_t rank = 0; rank < top.size(); ++rank) {
    const std::size_t id = top[rank].second;
    const std::size_t k = id / xy;
    const std::size_t rem = id - k * xy;
    const std::size_t j = rem / shape.nx;
    const std::size_t i = rem - j * shape.nx;
    const double density =
        last_gaussian_density_center_.size() == shape.n
            ? last_gaussian_density_center_[id]
            : 0.0;
    const double surface =
        last_gaussian_surface_.size() == shape.n ? last_gaussian_surface_[id]
                                                 : 0.0;
    const double eps_center =
        args_.surface == "gaussian"
            ? args_.int_diel +
                  (args_.gap_diel - args_.int_diel) *
                      (1.0 - std::clamp(density, 0.0, 1.0))
            : (last_gaussian_eps_center_.size() == shape.n
                   ? last_gaussian_eps_center_[id]
                   : 0.0);
    out << result.trajectory_id << ',' << result.frame_index << ','
        << result.global_index << ',' << result.time_ps << ',' << (rank + 1)
        << ',' << i << ',' << j << ',' << k << ','
        << shape.ox + static_cast<double>(i) * shape.h << ','
        << shape.oy + static_cast<double>(j) * shape.h << ','
        << shape.oz + static_cast<double>(k) * shape.h << ',' << source[id]
        << ',' << (coulomb.size() == shape.n ? coulomb[id] : 0.0) << ','
        << density << ',' << surface << ',' << eps_center << ','
        << (rpb_is_boundary(i, j, k, shape) ? 1 : 0) << '\n';
  }
}

} // namespace TitanPBE

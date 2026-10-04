#include "tabi/analyzer_impl.h"

#include "pbe/grid/grid_common.h"
#include "pbe/grid/ses_mask.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace TitanPBE {
namespace {

bool surface_uses_probe_exclusion(const std::string &surface) noexcept {
  return surface == "sas" || surface == "ses" || surface == "delphi-ses";
}

bool surface_uses_ses_flood_fill(const std::string &surface) noexcept {
  return surface == "ses" || surface == "delphi-ses";
}

std::string grid_anchor_key(double scale, double margin, int test_grid_size,
                            const std::string &gridbox,
                            const std::string &focus_mode) {
  std::ostringstream oss;
  oss << std::setprecision(17) << scale << '|' << margin << '|'
      << test_grid_size << '|' << gridbox << '|' << focus_mode;
  return oss.str();
}

std::string uppercase_letters(std::string value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (std::isalpha(static_cast<unsigned char>(ch)))
      out.push_back(
          static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
  }
  return out;
}

struct AtomSpatialBins {
  double min_x{0.0};
  double min_y{0.0};
  double min_z{0.0};
  double cell{1.0};
  std::size_t bx{1};
  std::size_t by{1};
  std::size_t bz{1};
  double max_radius{0.0};
  std::vector<std::vector<std::size_t>> bins;
  bool empty{true};
};

template <typename AtomRange>
AtomSpatialBins build_atom_bins(const AtomRange &atoms, double inflate,
                                double min_cell) {
  AtomSpatialBins out;
  if (atoms.empty())
    return out;

  out.empty = false;
  out.min_x = std::numeric_limits<double>::max();
  out.min_y = std::numeric_limits<double>::max();
  out.min_z = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = std::numeric_limits<double>::lowest();
  double max_z = std::numeric_limits<double>::lowest();
  for (const auto &atom : atoms) {
    out.max_radius = std::max(out.max_radius, atom.radius + inflate);
    out.min_x = std::min(out.min_x, atom.x);
    out.min_y = std::min(out.min_y, atom.y);
    out.min_z = std::min(out.min_z, atom.z);
    max_x = std::max(max_x, atom.x);
    max_y = std::max(max_y, atom.y);
    max_z = std::max(max_z, atom.z);
  }
  out.cell = std::max({out.max_radius, min_cell, 1.0e-6});
  out.bx = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_x - out.min_x) / out.cell)) + 1);
  out.by = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_y - out.min_y) / out.cell)) + 1);
  out.bz = std::max<std::size_t>(
      1,
      static_cast<std::size_t>(std::floor((max_z - out.min_z) / out.cell)) + 1);
  out.bins.assign(out.bx * out.by * out.bz, {});

  auto bin_index = [&](double x, double y, double z) {
    const std::size_t ix = std::min<std::size_t>(
        out.bx - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((x - out.min_x) / out.cell))));
    const std::size_t iy = std::min<std::size_t>(
        out.by - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((y - out.min_y) / out.cell))));
    const std::size_t iz = std::min<std::size_t>(
        out.bz - 1, static_cast<std::size_t>(
                        std::max(0.0, std::floor((z - out.min_z) / out.cell))));
    return (iz * out.by + iy) * out.bx + ix;
  };
  for (std::size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx)
    out.bins[bin_index(atoms[atom_idx].x, atoms[atom_idx].y, atoms[atom_idx].z)]
        .push_back(atom_idx);
  return out;
}

template <typename AtomRange>
bool point_inside_inflated_atom(const AtomSpatialBins &bins,
                                const AtomRange &atoms, double x, double y,
                                double z, double inflate) {
  if (bins.empty)
    return false;
  const std::size_t bix = std::min<std::size_t>(
      bins.bx - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((x - bins.min_x) / bins.cell))));
  const std::size_t biy = std::min<std::size_t>(
      bins.by - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((y - bins.min_y) / bins.cell))));
  const std::size_t biz = std::min<std::size_t>(
      bins.bz - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((z - bins.min_z) / bins.cell))));
  const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
  const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
  const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
  const std::size_t x1 = std::min(bins.bx - 1, bix + 1);
  const std::size_t y1 = std::min(bins.by - 1, biy + 1);
  const std::size_t z1 = std::min(bins.bz - 1, biz + 1);
  for (std::size_t kk = z0; kk <= z1; ++kk) {
    for (std::size_t jj = y0; jj <= y1; ++jj) {
      for (std::size_t ii = x0; ii <= x1; ++ii) {
        const auto &cell_atoms = bins.bins[(kk * bins.by + jj) * bins.bx + ii];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = atom.radius + inflate;
          const double dx = x - atom.x;
          const double dy = y - atom.y;
          const double dz = z - atom.z;
          if (dx * dx + dy * dy + dz * dz <= radius * radius)
            return true;
        }
      }
    }
  }
  return false;
}

template <typename AtomRange>
bool point_near_inflated_atom_surface(const AtomSpatialBins &bins,
                                      const AtomRange &atoms, double x,
                                      double y, double z, double inflate,
                                      double release_band) {
  if (bins.empty)
    return false;
  const std::size_t bix = std::min<std::size_t>(
      bins.bx - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((x - bins.min_x) / bins.cell))));
  const std::size_t biy = std::min<std::size_t>(
      bins.by - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((y - bins.min_y) / bins.cell))));
  const std::size_t biz = std::min<std::size_t>(
      bins.bz - 1, static_cast<std::size_t>(std::max(
                       0.0, std::floor((z - bins.min_z) / bins.cell))));
  const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
  const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
  const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
  const std::size_t x1 = std::min(bins.bx - 1, bix + 1);
  const std::size_t y1 = std::min(bins.by - 1, biy + 1);
  const std::size_t z1 = std::min(bins.bz - 1, biz + 1);
  for (std::size_t kk = z0; kk <= z1; ++kk) {
    for (std::size_t jj = y0; jj <= y1; ++jj) {
      for (std::size_t ii = x0; ii <= x1; ++ii) {
        const auto &cell_atoms = bins.bins[(kk * bins.by + jj) * bins.bx + ii];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = std::max(0.0, atom.radius + inflate);
          const double lower = std::max(0.0, radius - release_band);
          const double upper = radius + release_band;
          const double dx = x - atom.x;
          const double dy = y - atom.y;
          const double dz = z - atom.z;
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 >= lower * lower && d2 <= upper * upper)
            return true;
        }
      }
    }
  }
  return false;
}

} // namespace

std::string AnalyzerImpl::inferred_element_symbol(const Atom &atom) const {
  const std::string explicit_element = uppercase_letters(atom.element);
  if (!explicit_element.empty()) {
    if (explicit_element.size() == 1 || explicit_element == "CL" ||
        explicit_element == "BR" || explicit_element == "NA" ||
        explicit_element == "MG" || explicit_element == "ZN" ||
        explicit_element == "FE" || explicit_element == "CA")
      return explicit_element;
    return std::string(1, explicit_element.front());
  }

  const std::string name = uppercase_letters(atom.atom_name);
  if (name.empty())
    return "UNK";
  const std::string res = uppercase_letters(atom.resname);
  if (name == "CL" || name == "BR")
    return name;
  if (name == "NA" && (res == "NA" || res == "SOD"))
    return "NA";
  if (name == "MG" && res == "MG")
    return "MG";
  if (name == "ZN" && res == "ZN")
    return "ZN";
  if (name == "FE" && res == "FE")
    return "FE";
  if (name == "CA" && (res == "CA" || res == "CAL" || res == "CA2"))
    return "CA";
  return std::string(1, name.front());
}

double AnalyzerImpl::effective_surface_radius(const Atom &atom) const {
  if (effective_radius_convention() != "delphi-bondi")
    return atom.radius;
  if (!args_.test_pqr.empty() && atom.radius > 0.0) {
    return atom.radius;
  }
  const std::string element = inferred_element_symbol(atom);
  if (element == "H")
    return std::max(atom.radius, 1.30);
  if (element == "C")
    return std::max(atom.radius, 1.80);
  if (element == "N")
    return std::max(atom.radius, 1.60);
  if (element == "O")
    return std::max(atom.radius, 1.50);
  if (element == "S" || element == "P")
    return std::max(atom.radius, 1.80);
  if (element == "F")
    return std::max(atom.radius, 1.47);
  if (element == "CL")
    return std::max(atom.radius, 1.75);
  if (element == "BR")
    return std::max(atom.radius, 1.85);
  if (element == "I")
    return std::max(atom.radius, 1.98);
  return atom.radius;
}

std::string AnalyzerImpl::radius_media_source(const Atom &atom) const {
  if (effective_radius_convention() != "delphi-bondi")
    return "native";
  if (!args_.test_pqr.empty() && atom.radius > 0.0)
    return "pqr_siz_radius";
  if (effective_surface_radius(atom) > atom.radius + 1.0e-12)
    return "bondi_floor";
  return "input_radius";
}

std::string AnalyzerImpl::effective_radius_convention() const {
  if (args_.radius_convention != "auto")
    return args_.radius_convention;
  if (args_.surface.rfind("delphi-", 0) == 0 || uses_gaussian_model() ||
      args_.biomodel == "rpbe" || args_.charge_assignment == "delphi" ||
      args_.charge_assignment == "gaussian")
    return "delphi-bondi";
  return "native";
}

AnalyzerImpl::GridShape
AnalyzerImpl::build_shape(const std::vector<Atom> &atoms) const {
  return build_shape(atoms, args_.scale, args_.margin);
}

AnalyzerImpl::GridShape
AnalyzerImpl::build_shape(const std::vector<Atom> &atoms, double scale,
                          double margin) const {
  if (atoms.empty())
    throw std::runtime_error("PBE: no atoms were available for the frame.");

  double min_x = std::numeric_limits<double>::max();
  double min_y = std::numeric_limits<double>::max();
  double min_z = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = std::numeric_limits<double>::lowest();
  double max_z = std::numeric_limits<double>::lowest();
  double coord_min_x = std::numeric_limits<double>::max();
  double coord_min_y = std::numeric_limits<double>::max();
  double coord_min_z = std::numeric_limits<double>::max();
  double coord_max_x = std::numeric_limits<double>::lowest();
  double coord_max_y = std::numeric_limits<double>::lowest();
  double coord_max_z = std::numeric_limits<double>::lowest();
  double delphi_min_x = std::numeric_limits<double>::max();
  double delphi_min_y = std::numeric_limits<double>::max();
  double delphi_min_z = std::numeric_limits<double>::max();
  double delphi_max_x = std::numeric_limits<double>::lowest();
  double delphi_max_y = std::numeric_limits<double>::lowest();
  double delphi_max_z = std::numeric_limits<double>::lowest();
  const double surface_inflate =
      surface_uses_probe_exclusion(args_.surface) ? args_.probe_radius : 0.0;
  for (const Atom &atom : atoms) {
    coord_min_x = std::min(coord_min_x, atom.x);
    coord_min_y = std::min(coord_min_y, atom.y);
    coord_min_z = std::min(coord_min_z, atom.z);
    coord_max_x = std::max(coord_max_x, atom.x);
    coord_max_y = std::max(coord_max_y, atom.y);
    coord_max_z = std::max(coord_max_z, atom.z);
    const double delphi_r = effective_surface_radius(atom);
    delphi_min_x = std::min(delphi_min_x, atom.x - delphi_r);
    delphi_min_y = std::min(delphi_min_y, atom.y - delphi_r);
    delphi_min_z = std::min(delphi_min_z, atom.z - delphi_r);
    delphi_max_x = std::max(delphi_max_x, atom.x + delphi_r);
    delphi_max_y = std::max(delphi_max_y, atom.y + delphi_r);
    delphi_max_z = std::max(delphi_max_z, atom.z + delphi_r);
    const double r = delphi_r + surface_inflate;
    min_x = std::min(min_x, atom.x - r);
    min_y = std::min(min_y, atom.y - r);
    min_z = std::min(min_z, atom.z - r);
    max_x = std::max(max_x, atom.x + r);
    max_y = std::max(max_y, atom.y + r);
    max_z = std::max(max_z, atom.z + r);
  }

  min_x -= margin;
  min_y -= margin;
  min_z -= margin;
  max_x += margin;
  max_y += margin;
  max_z += margin;

  if (args_.gridbox == "cubic") {
    const double cx = 0.5 * (min_x + max_x);
    const double cy = 0.5 * (min_y + max_y);
    const double cz = 0.5 * (min_z + max_z);
    const double span = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
    min_x = cx - 0.5 * span;
    max_x = cx + 0.5 * span;
    min_y = cy - 0.5 * span;
    max_y = cy + 0.5 * span;
    min_z = cz - 0.5 * span;
    max_z = cz + 0.5 * span;
  }

  GridShape shape;
  shape.h = 1.0 / scale;
  if (!args_.test_pqr.empty() && args_.test_grid_size > 0) {
    const std::size_t n = static_cast<std::size_t>(args_.test_grid_size);
    double cx = 0.5 * (delphi_min_x + delphi_max_x);
    double cy = 0.5 * (delphi_min_y + delphi_max_y);
    double cz = 0.5 * (delphi_min_z + delphi_max_z);
    if (args_.test_center_user) {
      cx = args_.test_center_x;
      cy = args_.test_center_y;
      cz = args_.test_center_z;
    } else if (args_.focus_center_user) {
      cx = args_.focus_center_x;
      cy = args_.focus_center_y;
      cz = args_.focus_center_z;
    }
    shape.nx = n;
    shape.ny = n;
    shape.nz = n;
    shape.ox = cx - 0.5 * static_cast<double>(n - 1) * shape.h;
    shape.oy = cy - 0.5 * static_cast<double>(n - 1) * shape.h;
    shape.oz = cz - 0.5 * static_cast<double>(n - 1) * shape.h;
    if (shape.nx > std::numeric_limits<std::size_t>::max() / shape.ny ||
        shape.nx * shape.ny >
            std::numeric_limits<std::size_t>::max() / shape.nz) {
      throw std::runtime_error("PBE: grid dimensions overflow size_t.");
    }
    shape.n = shape.nx * shape.ny * shape.nz;
    shape.memory_bytes =
        shape.n * (7 * sizeof(double) + 2 * sizeof(unsigned char));
    if (shape.n > args_.max_grid_points) {
      std::ostringstream msg;
      msg << "PBE: requested grid " << shape.nx << "x" << shape.ny << "x"
          << shape.nz << " = " << shape.n << " points exceeds -pbe-max-grid "
          << args_.max_grid_points << ".";
      throw std::runtime_error(msg.str());
    }
    return apply_grid_anchor(shape, scale, margin);
  }
  shape.ox = min_x;
  shape.oy = min_y;
  shape.oz = min_z;
  auto count_axis = [&](double span) {
    std::size_t n =
        static_cast<std::size_t>(std::ceil(std::max(0.0, span) / shape.h)) + 1;
    n = std::max<std::size_t>(3, n);
    if ((n & 1ULL) == 0)
      ++n;
    return n;
  };
  shape.nx = count_axis(max_x - min_x);
  shape.ny = count_axis(max_y - min_y);
  shape.nz = count_axis(max_z - min_z);
  if (args_.focus_center_user && args_.focus == "delphi") {
    shape.ox = args_.focus_center_x -
               0.5 * static_cast<double>(shape.nx - 1) * shape.h;
    shape.oy = args_.focus_center_y -
               0.5 * static_cast<double>(shape.ny - 1) * shape.h;
    shape.oz = args_.focus_center_z -
               0.5 * static_cast<double>(shape.nz - 1) * shape.h;
  }

  if (shape.nx > std::numeric_limits<std::size_t>::max() / shape.ny ||
      shape.nx * shape.ny >
          std::numeric_limits<std::size_t>::max() / shape.nz) {
    throw std::runtime_error("PBE: grid dimensions overflow size_t.");
  }
  shape.n = shape.nx * shape.ny * shape.nz;
  shape.memory_bytes =
      shape.n * (7 * sizeof(double) + 2 * sizeof(unsigned char));
  if (shape.n > args_.max_grid_points) {
    std::ostringstream msg;
    msg << "PBE: requested grid " << shape.nx << "x" << shape.ny << "x"
        << shape.nz << " = " << shape.n << " points exceeds -pbe-max-grid "
        << args_.max_grid_points << ".";
    throw std::runtime_error(msg.str());
  }
  return apply_grid_anchor(shape, scale, margin);
}

AnalyzerImpl::GridShape AnalyzerImpl::apply_grid_anchor(const GridShape &input,
                                                        double scale,
                                                        double margin) const {
  GridShape shape = input;
  const std::string anchor = effective_grid_anchor();
  if (anchor == "auto" || anchor == "per-frame")
    return shape;

  if (anchor == "fixed-center") {
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    if (args_.test_center_user) {
      cx = args_.test_center_x;
      cy = args_.test_center_y;
      cz = args_.test_center_z;
    } else {
      cx = args_.focus_center_x;
      cy = args_.focus_center_y;
      cz = args_.focus_center_z;
    }
    shape.ox = cx - 0.5 * static_cast<double>(shape.nx - 1) * shape.h;
    shape.oy = cy - 0.5 * static_cast<double>(shape.ny - 1) * shape.h;
    shape.oz = cz - 0.5 * static_cast<double>(shape.nz - 1) * shape.h;
    return shape;
  }

  if (anchor == "first-frame") {
    const std::string key = grid_anchor_key(scale, margin, args_.test_grid_size,
                                            args_.gridbox, args_.focus);
    auto it = grid_anchor_shapes_.find(key);
    if (it == grid_anchor_shapes_.end()) {
      grid_anchor_shapes_.emplace(key, shape);
      return shape;
    }
    return it->second;
  }

  return shape;
}

std::string AnalyzerImpl::effective_grid_anchor() const {
  if (args_.grid_anchor == "fixed-center" || args_.grid_anchor == "first-frame")
    return args_.grid_anchor;
  if (stability_grid_anchor_enabled())
    return "first-frame";
  return args_.grid_anchor;
}

void AnalyzerImpl::reset_grid_anchor() const noexcept {
  grid_anchor_shapes_.clear();
}

void AnalyzerImpl::ensure_workspace(const GridShape &shape, bool reset_medium) {
  invalidate_nwt_half_workspace();
  const bool sized = grid_.sized(shape.n);
  if (sized) {
    ++grid_reuse_stats_.workspace_reuse_hits;
    std::fill(grid_.phi.begin(), grid_.phi.end(), 0.0);
    std::fill(grid_.rho.begin(), grid_.rho.end(), 0.0);
    if (reset_medium) {
      std::fill(grid_.eps_x.begin(), grid_.eps_x.end(), args_.ext_diel);
      std::fill(grid_.eps_y.begin(), grid_.eps_y.end(), args_.ext_diel);
      std::fill(grid_.eps_z.begin(), grid_.eps_z.end(), args_.ext_diel);
      std::fill(grid_.denom.begin(), grid_.denom.end(), 1.0);
      std::fill(grid_.screen.begin(), grid_.screen.end(), 0.0);
      std::fill(grid_.inside.begin(), grid_.inside.end(), 0U);
      std::fill(grid_.ion_accessible.begin(), grid_.ion_accessible.end(), 1U);
    }
  } else {
    ++grid_reuse_stats_.workspace_reallocations;
    grid_.reset(shape.n, GridOptions{args_.ext_diel});
  }
  last_gaussian_density_center_.clear();
  last_gaussian_density_midpoints_.clear();
  last_gaussian_eps_center_.clear();
  last_gaussian_surface_.clear();
  last_gaussian_grad_surface_.clear();
  last_gaussian_density_sum_ = 0.0;
  last_gaussian_cutoff_points_ = 0;
  last_gaussian_eps_min_ = 0.0;
  last_gaussian_eps_max_ = 0.0;
  last_gaussian_eps_mean_ = 0.0;
  last_vdwms_audit_ = VdwmsAudit{};
  last_iscm_surface_valid_.clear();
  last_iscm_surface_positions_.clear();
}

void AnalyzerImpl::classify_surface(const GridShape &shape,
                                    const std::vector<Atom> &atoms,
                                    TitanExecutor &executor) {
  const std::vector<Atom> surface_atoms = adjusted_surface_atoms(atoms);
  if (uses_gaussian_model()) {
    const double sigma = std::max(1.0e-12, args_.gaussian_sigma);
    const double grid_corner_extension = 0.5 * shape.h * std::sqrt(3.0);
    const double influence =
        std::max(shape.h, gaussian_influence_radius(surface_atoms) +
                              grid_corner_extension);
    const double influence2 = influence * influence;
    const AtomSpatialBins gaussian_bins =
        build_atom_bins(surface_atoms, std::max(0.0, influence), shape.h);
    const double gaussian_exponent = std::max(1.0e-12, args_.gaussian_exp);
    const int gaussian_radius_bins =
        static_cast<int>(
            std::ceil(std::max(shape.h, influence) / gaussian_bins.cell)) +
        1;
    auto density_at = [&](double x, double y, double z) {
      const int bx0 = static_cast<int>(
          std::floor((x - gaussian_bins.min_x) / gaussian_bins.cell));
      const int by0 = static_cast<int>(
          std::floor((y - gaussian_bins.min_y) / gaussian_bins.cell));
      const int bz0 = static_cast<int>(
          std::floor((z - gaussian_bins.min_z) / gaussian_bins.cell));
      double solvent_complement = 1.0;
      for (int dz = -gaussian_radius_bins; dz <= gaussian_radius_bins; ++dz) {
        const int bz = bz0 + dz;
        if (bz < 0 || bz >= static_cast<int>(gaussian_bins.bz))
          continue;
        for (int dy = -gaussian_radius_bins; dy <= gaussian_radius_bins; ++dy) {
          const int by = by0 + dy;
          if (by < 0 || by >= static_cast<int>(gaussian_bins.by))
            continue;
          for (int dx = -gaussian_radius_bins; dx <= gaussian_radius_bins;
               ++dx) {
            const int bx = bx0 + dx;
            if (bx < 0 || bx >= static_cast<int>(gaussian_bins.bx))
              continue;
            const auto &bucket =
                gaussian_bins
                    .bins[(static_cast<std::size_t>(bz) * gaussian_bins.by +
                           static_cast<std::size_t>(by)) *
                              gaussian_bins.bx +
                          static_cast<std::size_t>(bx)];
            for (std::size_t atom_idx : bucket) {
              const Atom &atom = surface_atoms[atom_idx];
              const double r0 = std::max(1.0e-12, atom.radius);
              const double ddx = x - atom.x;
              const double ddy = y - atom.y;
              const double ddz = z - atom.z;
              const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (r2 > influence2)
                continue;
              const double scaled2 = r2 / (sigma * sigma * r0 * r0);
              const double atom_density = std::clamp(
                  std::exp(-std::pow(scaled2, gaussian_exponent)), 0.0, 1.0);
              solvent_complement *= (1.0 - atom_density);
              if (solvent_complement <= 1.0e-15)
                return 1.0;
            }
          }
        }
      }
      return std::clamp(1.0 - solvent_complement, 0.0, 1.0);
    };
    const double cutoff = args_.surface_cutoff == "auto"
                              ? args_.density_cutoff
                              : args_.surface_cutoff_value;
    const std::size_t workers = effective_workers(executor, shape.nz);
    last_gaussian_density_center_.assign(shape.n, 0.0);
    std::vector<double> local_density_sum(workers, 0.0);
    std::vector<std::size_t> local_cutoff_points(workers, 0);
    executor.parallel_for(
        shape.nz, workers, 1,
        [&](std::size_t worker, std::size_t z_begin, std::size_t z_end) {
          double density_sum = 0.0;
          std::size_t cutoff_points = 0;
          for (std::size_t k = z_begin; k < z_end; ++k) {
            const double z = shape.oz + static_cast<double>(k) * shape.h;
            for (std::size_t j = 0; j < shape.ny; ++j) {
              const double y = shape.oy + static_cast<double>(j) * shape.h;
              for (std::size_t i = 0; i < shape.nx; ++i) {
                const double x = shape.ox + static_cast<double>(i) * shape.h;
                const double density = density_at(x, y, z);
                last_gaussian_density_center_[idx3(i, j, k, shape.nx,
                                                   shape.ny)] = density;
                density_sum += density;
                const bool inside = density >= cutoff;
                if (inside)
                  ++cutoff_points;
                grid_.inside[idx3(i, j, k, shape.nx, shape.ny)] =
                    inside ? 1U : 0U;
              }
            }
          }
          local_density_sum[worker] += density_sum;
          local_cutoff_points[worker] += cutoff_points;
        });
    last_gaussian_density_sum_ = std::accumulate(local_density_sum.begin(),
                                                 local_density_sum.end(), 0.0);
    last_gaussian_cutoff_points_ =
        std::accumulate(local_cutoff_points.begin(), local_cutoff_points.end(),
                        static_cast<std::size_t>(0));
    return;
  }
  double max_radius = 0.0;
  double min_x = std::numeric_limits<double>::max();
  double min_y = std::numeric_limits<double>::max();
  double min_z = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = std::numeric_limits<double>::lowest();
  double max_z = std::numeric_limits<double>::lowest();
  for (const Atom &atom : surface_atoms) {
    const double surface_radius =
        atom.radius + (surface_uses_probe_exclusion(args_.surface)
                           ? args_.probe_radius
                           : 0.0);
    max_radius = std::max(max_radius, surface_radius);
    min_x = std::min(min_x, atom.x);
    min_y = std::min(min_y, atom.y);
    min_z = std::min(min_z, atom.z);
    max_x = std::max(max_x, atom.x);
    max_y = std::max(max_y, atom.y);
    max_z = std::max(max_z, atom.z);
  }
  const double cell = std::max(max_radius, shape.h);
  const std::size_t bx = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor((max_x - min_x) / cell)) + 1);
  const std::size_t by = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor((max_y - min_y) / cell)) + 1);
  const std::size_t bz = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor((max_z - min_z) / cell)) + 1);

  std::vector<std::vector<std::size_t>> bins(bx * by * bz);
  auto bin_index = [&](double x, double y, double z) {
    const std::size_t ix = std::min<std::size_t>(
        bx - 1, static_cast<std::size_t>(
                    std::max(0.0, std::floor((x - min_x) / cell))));
    const std::size_t iy = std::min<std::size_t>(
        by - 1, static_cast<std::size_t>(
                    std::max(0.0, std::floor((y - min_y) / cell))));
    const std::size_t iz = std::min<std::size_t>(
        bz - 1, static_cast<std::size_t>(
                    std::max(0.0, std::floor((z - min_z) / cell))));
    return (iz * by + iy) * bx + ix;
  };
  for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size(); ++atom_idx) {
    const Atom &atom = surface_atoms[atom_idx];
    bins[bin_index(atom.x, atom.y, atom.z)].push_back(atom_idx);
  }

  const std::size_t workers = effective_workers(executor, shape.nz);
  executor.parallel_for(
      shape.nz, workers, 1,
      [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
        for (std::size_t k = z_begin; k < z_end; ++k) {
          const double z = shape.oz + static_cast<double>(k) * shape.h;
          for (std::size_t j = 0; j < shape.ny; ++j) {
            const double y = shape.oy + static_cast<double>(j) * shape.h;
            for (std::size_t i = 0; i < shape.nx; ++i) {
              const double x = shape.ox + static_cast<double>(i) * shape.h;
              const std::size_t bix = std::min<std::size_t>(
                  bx - 1, static_cast<std::size_t>(
                              std::max(0.0, std::floor((x - min_x) / cell))));
              const std::size_t biy = std::min<std::size_t>(
                  by - 1, static_cast<std::size_t>(
                              std::max(0.0, std::floor((y - min_y) / cell))));
              const std::size_t biz = std::min<std::size_t>(
                  bz - 1, static_cast<std::size_t>(
                              std::max(0.0, std::floor((z - min_z) / cell))));

              bool hit = false;
              const std::size_t x0 = (bix == 0) ? 0 : bix - 1;
              const std::size_t y0 = (biy == 0) ? 0 : biy - 1;
              const std::size_t z0 = (biz == 0) ? 0 : biz - 1;
              const std::size_t x1 = std::min(bx - 1, bix + 1);
              const std::size_t y1 = std::min(by - 1, biy + 1);
              const std::size_t z1 = std::min(bz - 1, biz + 1);
              for (std::size_t kk = z0; kk <= z1 && !hit; ++kk) {
                for (std::size_t jj = y0; jj <= y1 && !hit; ++jj) {
                  for (std::size_t ii = x0; ii <= x1 && !hit; ++ii) {
                    const auto &cell_atoms = bins[(kk * by + jj) * bx + ii];
                    for (std::size_t atom_idx : cell_atoms) {
                      const Atom &atom = surface_atoms[atom_idx];
                      const double dx = x - atom.x;
                      const double dy = y - atom.y;
                      const double dz = z - atom.z;
                      const double surface_radius =
                          atom.radius +
                          (surface_uses_probe_exclusion(args_.surface)
                               ? args_.probe_radius
                               : 0.0);
                      if (dx * dx + dy * dy + dz * dz <=
                          surface_radius * surface_radius) {
                        hit = true;
                        break;
                      }
                    }
                  }
                }
              }
              grid_.inside[idx3(i, j, k, shape.nx, shape.ny)] = hit ? 1U : 0U;
            }
          }
        }
      });

  if (!surface_uses_ses_flood_fill(args_.surface))
    return;

  erode_expanded_mask_to_ses(
      SesMaskShape{shape.nx, shape.ny, shape.nz, shape.h},
      std::span<const std::uint8_t>(grid_.inside.data(), grid_.inside.size()),
      args_.probe_radius,
      std::span<std::uint8_t>(grid_.inside.data(), grid_.inside.size()),
      &executor);

  if (shape.n >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error(
        "PBE: SES surface flood fill currently requires <= 2^32 grid points.");
  }

  std::vector<unsigned char> solvent_accessible(shape.n, 0);
  std::vector<std::uint32_t> queue;
  queue.reserve(std::min<std::size_t>(shape.n, 1048576));
  auto enqueue = [&](std::size_t id) {
    if (grid_.inside[id] || solvent_accessible[id])
      return;
    solvent_accessible[id] = 1U;
    queue.push_back(static_cast<std::uint32_t>(id));
  };

  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      enqueue(idx3(0, j, k, shape.nx, shape.ny));
      enqueue(idx3(shape.nx - 1, j, k, shape.nx, shape.ny));
    }
    for (std::size_t i = 0; i < shape.nx; ++i) {
      enqueue(idx3(i, 0, k, shape.nx, shape.ny));
      enqueue(idx3(i, shape.ny - 1, k, shape.nx, shape.ny));
    }
  }
  for (std::size_t j = 0; j < shape.ny; ++j) {
    for (std::size_t i = 0; i < shape.nx; ++i) {
      enqueue(idx3(i, j, 0, shape.nx, shape.ny));
      enqueue(idx3(i, j, shape.nz - 1, shape.nx, shape.ny));
    }
  }

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t id = static_cast<std::size_t>(queue[head]);
    const std::size_t i = id % shape.nx;
    const std::size_t j = (id / shape.nx) % shape.ny;
    const std::size_t k = id / (shape.nx * shape.ny);
    if (i > 0)
      enqueue(id - 1);
    if (i + 1 < shape.nx)
      enqueue(id + 1);
    if (j > 0)
      enqueue(id - shape.nx);
    if (j + 1 < shape.ny)
      enqueue(id + shape.nx);
    if (k > 0)
      enqueue(id - shape.nx * shape.ny);
    if (k + 1 < shape.nz)
      enqueue(id + shape.nx * shape.ny);
  }

  for (std::size_t id = 0; id < shape.n; ++id)
    grid_.inside[id] = solvent_accessible[id] ? 0U : 1U;
}

void AnalyzerImpl::classify_ion_accessibility(const GridShape &shape,
                                              const std::vector<Atom> &atoms,
                                              TitanExecutor &executor) {
  grid_.ion_accessible.assign(shape.n, 0);
  const bool delphi_ion_exclusion =
      (args_.test_operator_convention == "delphi" && !args_.test_pqr.empty()) ||
      args_.focus == "delphi";
  if (args_.ion_access == "dielectric" && !delphi_ion_exclusion) {
    for (std::size_t id = 0; id < shape.n; ++id)
      grid_.ion_accessible[id] = grid_.inside[id] ? 0U : 1U;
    return;
  }

  if (shape.n >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::runtime_error(
        "PBE: Stern ion flood fill currently requires <= 2^32 grid points.");
  }

  const AtomSpatialBins bins =
      build_atom_bins(atoms, args_.ion_radius, shape.h);
  std::vector<unsigned char> blocked(shape.n, 0);
  const std::size_t workers = effective_workers(executor, shape.nz);
  executor.parallel_for(
      shape.nz, workers, 1,
      [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
        for (std::size_t k = z_begin; k < z_end; ++k) {
          const double z = shape.oz + static_cast<double>(k) * shape.h;
          for (std::size_t j = 0; j < shape.ny; ++j) {
            const double y = shape.oy + static_cast<double>(j) * shape.h;
            for (std::size_t i = 0; i < shape.nx; ++i) {
              const double x = shape.ox + static_cast<double>(i) * shape.h;
              const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
              blocked[id] = point_inside_inflated_atom(bins, atoms, x, y, z,
                                                       args_.ion_radius)
                                ? 1U
                                : 0U;
            }
          }
        }
      });

  std::vector<std::uint32_t> queue;
  queue.reserve(std::min<std::size_t>(shape.n, 1048576));
  auto enqueue = [&](std::size_t id) {
    if (blocked[id] || grid_.ion_accessible[id])
      return;
    grid_.ion_accessible[id] = 1U;
    queue.push_back(static_cast<std::uint32_t>(id));
  };

  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      enqueue(idx3(0, j, k, shape.nx, shape.ny));
      enqueue(idx3(shape.nx - 1, j, k, shape.nx, shape.ny));
    }
    for (std::size_t i = 0; i < shape.nx; ++i) {
      enqueue(idx3(i, 0, k, shape.nx, shape.ny));
      enqueue(idx3(i, shape.ny - 1, k, shape.nx, shape.ny));
    }
  }
  for (std::size_t j = 0; j < shape.ny; ++j) {
    for (std::size_t i = 0; i < shape.nx; ++i) {
      enqueue(idx3(i, j, 0, shape.nx, shape.ny));
      enqueue(idx3(i, j, shape.nz - 1, shape.nx, shape.ny));
    }
  }

  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t id = static_cast<std::size_t>(queue[head]);
    const std::size_t i = id % shape.nx;
    const std::size_t j = (id / shape.nx) % shape.ny;
    const std::size_t k = id / (shape.nx * shape.ny);
    if (i > 0)
      enqueue(id - 1);
    if (i + 1 < shape.nx)
      enqueue(id + 1);
    if (j > 0)
      enqueue(id - shape.nx);
    if (j + 1 < shape.ny)
      enqueue(id + shape.nx);
    if (k > 0)
      enqueue(id - shape.nx * shape.ny);
    if (k + 1 < shape.nz)
      enqueue(id + shape.nx * shape.ny);
  }
}

void AnalyzerImpl::apply_stability_hysteresis(const GridShape &shape,
                                              const std::vector<Atom> &atoms) {
  stability_hysteresis_cells_kept_frame_ = 0;
  stability_hysteresis_cells_released_frame_ = 0;
  if (!stability_hysteresis_enabled() || !last_stability_medium_state_.valid)
    return;
  const StabilityMediumState &prev = last_stability_medium_state_;
  const bool same_grid =
      prev.nx == shape.nx && prev.ny == shape.ny && prev.nz == shape.nz &&
      std::abs(prev.h - shape.h) <= 1.0e-12 &&
      std::abs(prev.origin_x - shape.ox) <=
          std::max(1.0e-9, shape.h * 1.0e-9) &&
      std::abs(prev.origin_y - shape.oy) <=
          std::max(1.0e-9, shape.h * 1.0e-9) &&
      std::abs(prev.origin_z - shape.oz) <= std::max(1.0e-9, shape.h * 1.0e-9);
  if (!same_grid || prev.inside.size() != shape.n ||
      prev.ion_accessible.size() != shape.n || prev.eps_x.size() != shape.n ||
      prev.eps_y.size() != shape.n || prev.eps_z.size() != shape.n)
    return;

  const std::vector<Atom> surface_atoms = adjusted_surface_atoms(atoms);
  const double surface_inflate =
      surface_uses_probe_exclusion(args_.surface) ? args_.probe_radius : 0.0;
  const double release_band_A = std::max(1.0e-6, shape.h * 1.0e-3);
  const double max_hysteresis_inflate =
      std::max(surface_inflate, args_.ion_radius);
  const AtomSpatialBins surface_bins = build_atom_bins(
      surface_atoms, max_hysteresis_inflate + release_band_A, shape.h);
  auto near_surface = [&](double x, double y, double z, double inflate) {
    return point_near_inflated_atom_surface(surface_bins, surface_atoms, x, y,
                                            z, inflate, release_band_A);
  };

  const std::size_t xy = shape.nx * shape.ny;
  for (std::size_t k = 0; k < shape.nz; ++k) {
    const double z = shape.oz + static_cast<double>(k) * shape.h;
    for (std::size_t j = 0; j < shape.ny; ++j) {
      const double y = shape.oy + static_cast<double>(j) * shape.h;
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const double x = shape.ox + static_cast<double>(i) * shape.h;
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const bool inside_changed = grid_.inside[id] != prev.inside[id];
        const bool ion_changed =
            grid_.ion_accessible[id] != prev.ion_accessible[id];
        const bool eps_changed = grid_.eps_x[id] != prev.eps_x[id] ||
                                 grid_.eps_y[id] != prev.eps_y[id] ||
                                 grid_.eps_z[id] != prev.eps_z[id];
        if (!inside_changed && !ion_changed && !eps_changed)
          continue;

        bool keep_previous = false;
        if (inside_changed)
          keep_previous = near_surface(x, y, z, surface_inflate);
        if (!keep_previous && ion_changed)
          keep_previous = near_surface(
              x, y, z, std::max(surface_inflate, args_.ion_radius));
        if (!keep_previous && eps_changed) {
          keep_previous =
              (i + 1 < shape.nx &&
               near_surface(x + 0.5 * shape.h, y, z, surface_inflate)) ||
              (j + 1 < shape.ny &&
               near_surface(x, y + 0.5 * shape.h, z, surface_inflate)) ||
              (k + 1 < shape.nz &&
               near_surface(x, y, z + 0.5 * shape.h, surface_inflate));
        }

        if (keep_previous) {
          grid_.inside[id] = prev.inside[id];
          grid_.ion_accessible[id] = prev.ion_accessible[id];
          grid_.eps_x[id] = prev.eps_x[id];
          grid_.eps_y[id] = prev.eps_y[id];
          grid_.eps_z[id] = prev.eps_z[id];
          ++stability_hysteresis_cells_kept_frame_;
        } else {
          ++stability_hysteresis_cells_released_frame_;
        }
      }
    }
  }

  if (grid_.denom.size() == shape.n && grid_.screen.size() == shape.n) {
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          grid_.denom[id] = grid_.eps_x[id] + grid_.eps_x[id - 1] +
                            grid_.eps_y[id] + grid_.eps_y[id - shape.nx] +
                            grid_.eps_z[id] + grid_.eps_z[id - xy] +
                            grid_.screen[id];
        }
      }
    }
  }
}

void AnalyzerImpl::store_stability_medium_state(const GridShape &shape) {
  if (stability_probe_solve_active_)
    return;
  if (shape.n == 0 || grid_.inside.size() != shape.n ||
      grid_.ion_accessible.size() != shape.n || grid_.eps_x.size() != shape.n ||
      grid_.eps_y.size() != shape.n || grid_.eps_z.size() != shape.n)
    return;
  StabilityMediumState state;
  state.valid = true;
  state.nx = shape.nx;
  state.ny = shape.ny;
  state.nz = shape.nz;
  state.h = shape.h;
  state.origin_x = shape.ox;
  state.origin_y = shape.oy;
  state.origin_z = shape.oz;
  state.inside = grid_.inside;
  state.ion_accessible = grid_.ion_accessible;
  state.eps_x = grid_.eps_x;
  state.eps_y = grid_.eps_y;
  state.eps_z = grid_.eps_z;
  last_stability_medium_state_ = std::move(state);
}

} // namespace TitanPBE

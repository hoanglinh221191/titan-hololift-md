#include "tabi/analyzer_impl.h"

#include "pbe/grid/grid_common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace TitanPBE {
namespace {

template <typename Shape>
bool is_boundary(std::size_t i, std::size_t j, std::size_t k,
                 const Shape &shape) noexcept {
  return i == 0 || j == 0 || k == 0 || i + 1 == shape.nx ||
         j + 1 == shape.ny || k + 1 == shape.nz;
}

double clamp_fraction(double value) noexcept {
  if (value < 0.0)
    return 0.0;
  if (value > 1.0)
    return 1.0;
  return value;
}

double harmonic_eps(double a, double b) noexcept {
  if (a <= 0.0 || b <= 0.0)
    return 0.5 * (a + b);
  if (std::abs(a - b) <= 1.0e-14)
    return a;
  return 2.0 * a * b / (a + b);
}

double delphi_debye_length_a(double salt_m, double temp_k,
                             double eps_out) noexcept {
  if (salt_m <= 0.0)
    return 1.0e6;
  return 0.01990076478 *
         std::sqrt(std::max(1.0, temp_k) * std::max(1.0e-12, eps_out)) /
         std::sqrt(salt_m);
}

bool parse_double_token(const std::string &token, double &value) {
  try {
    std::size_t used = 0;
    value = std::stod(token, &used);
    return used == token.size() && std::isfinite(value);
  } catch (...) {
    return false;
  }
}

std::string trim_copy(const std::string &value) {
  std::size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first])))
    ++first;
  std::size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1])))
    --last;
  std::string out = value.substr(first, last - first);
  if (out.size() >= 2 && out.front() == '"' && out.back() == '"')
    out = out.substr(1, out.size() - 2);
  return out;
}

std::string lowercase_copy(std::string value) {
  for (char &ch : value)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

std::vector<std::string> split_csv_simple(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(trim_copy(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(trim_copy(field));
  return fields;
}

std::unordered_map<std::string, std::size_t>
csv_header_map(const std::vector<std::string> &header) {
  std::unordered_map<std::string, std::size_t> map;
  for (std::size_t i = 0; i < header.size(); ++i)
    map[lowercase_copy(header[i])] = i;
  return map;
}

bool csv_double(const std::vector<std::string> &fields,
                const std::unordered_map<std::string, std::size_t> &header,
                const std::vector<std::string> &names, double &value) {
  for (const std::string &name : names) {
    const auto it = header.find(lowercase_copy(name));
    if (it != header.end() && it->second < fields.size() &&
        parse_double_token(fields[it->second], value))
      return true;
  }
  return false;
}

bool csv_index(const std::vector<std::string> &fields,
               const std::unordered_map<std::string, std::size_t> &header,
               const std::string &name, std::size_t &value) {
  double parsed = 0.0;
  if (!csv_double(fields, header, {name}, parsed) || parsed < 0.0)
    return false;
  value = static_cast<std::size_t>(std::llround(parsed));
  return true;
}

std::string read_oracle_fixture_filter() {
  std::ifstream in("pbe_oracle_fixture.txt");
  if (!in.is_open())
    return {};
  std::string line;
  if (!std::getline(in, line))
    return {};
  return trim_copy(line);
}

bool oracle_fixture_matches(
    const std::vector<std::string> &fields,
    const std::unordered_map<std::string, std::size_t> &header,
    const std::string &fixture_filter) {
  if (fixture_filter.empty())
    return true;
  const auto it = header.find("fixture");
  if (it == header.end() || it->second >= fields.size())
    return true;
  return trim_copy(fields[it->second]) == fixture_filter;
}

bool parse_boolish(const std::string &value, bool &out) {
  const std::string v = lowercase_copy(trim_copy(value));
  if (v == "1" || v == "true" || v == "yes" || v == "solute" ||
      v == "inside") {
    out = true;
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "solvent" ||
      v == "outside") {
    out = false;
    return true;
  }
  double numeric = 0.0;
  if (parse_double_token(v, numeric)) {
    out = std::abs(numeric) > 0.5;
    return true;
  }
  return false;
}

std::string csv_escape(const std::string &value) {
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

bool surface_uses_probe_exclusion(const std::string &surface) noexcept {
  return surface == "sas" || surface == "ses" || surface == "delphi-ses";
}

bool surface_uses_face_refinement(const std::string &surface) noexcept {
  return surface == "delphi-ses" || surface == "delphi-vdw" ||
         surface == "gaussian" || surface == "gaussian-cutoff";
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
      1, static_cast<std::size_t>(std::floor((max_x - out.min_x) / out.cell)) +
             1);
  out.by = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor((max_y - out.min_y) / out.cell)) +
             1);
  out.bz = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::floor((max_z - out.min_z) / out.cell)) +
             1);
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
    out.bins[bin_index(atoms[atom_idx].x, atoms[atom_idx].y,
                       atoms[atom_idx].z)]
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
      bins.bx - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((x - bins.min_x) / bins.cell))));
  const std::size_t biy = std::min<std::size_t>(
      bins.by - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((y - bins.min_y) / bins.cell))));
  const std::size_t biz = std::min<std::size_t>(
      bins.bz - 1, static_cast<std::size_t>(
                       std::max(0.0, std::floor((z - bins.min_z) / bins.cell))));
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

double dielectric_from_inside_fraction(double inside_fraction, double eps_in,
                                       double eps_out) noexcept {
  const double f = std::clamp(inside_fraction, 0.0, 1.0);
  const double denom = f / std::max(1.0e-12, eps_in) +
                       (1.0 - f) / std::max(1.0e-12, eps_out);
  return denom > 0.0 ? 1.0 / denom : 0.5 * (eps_in + eps_out);
}

template <typename AtomRange>
double segment_inside_fraction(const AtomSpatialBins &bins,
                               const AtomRange &atoms, double x0, double y0,
                               double z0, double x1, double y1, double z1,
                               double inflate) {
  if (bins.empty)
    return 0.0;

  const double sx = x1 - x0;
  const double sy = y1 - y0;
  const double sz = z1 - z0;
  const double a = sx * sx + sy * sy + sz * sz;
  if (a <= 0.0)
    return 0.0;

  const double pad = bins.max_radius + 1.0e-12;
  auto clamp_cell = [](double value, std::size_t n) {
    return static_cast<std::size_t>(std::clamp<double>(
        std::floor(value), 0.0, static_cast<double>(n - 1)));
  };
  const std::size_t ix0 = clamp_cell(
      (std::min(x0, x1) - pad - bins.min_x) / bins.cell, bins.bx);
  const std::size_t ix1 = clamp_cell(
      (std::max(x0, x1) + pad - bins.min_x) / bins.cell, bins.bx);
  const std::size_t iy0 = clamp_cell(
      (std::min(y0, y1) - pad - bins.min_y) / bins.cell, bins.by);
  const std::size_t iy1 = clamp_cell(
      (std::max(y0, y1) + pad - bins.min_y) / bins.cell, bins.by);
  const std::size_t iz0 = clamp_cell(
      (std::min(z0, z1) - pad - bins.min_z) / bins.cell, bins.bz);
  const std::size_t iz1 = clamp_cell(
      (std::max(z0, z1) + pad - bins.min_z) / bins.cell, bins.bz);

  std::vector<std::pair<double, double>> intervals;
  for (std::size_t iz = iz0; iz <= iz1; ++iz) {
    for (std::size_t iy = iy0; iy <= iy1; ++iy) {
      for (std::size_t ix = ix0; ix <= ix1; ++ix) {
        const auto &cell_atoms = bins.bins[(iz * bins.by + iy) * bins.bx + ix];
        for (std::size_t atom_idx : cell_atoms) {
          const auto &atom = atoms[atom_idx];
          const double radius = atom.radius + inflate;
          const double px = x0 - atom.x;
          const double py = y0 - atom.y;
          const double pz = z0 - atom.z;
          const double b = 2.0 * (px * sx + py * sy + pz * sz);
          const double c = px * px + py * py + pz * pz - radius * radius;
          const double disc = b * b - 4.0 * a * c;
          if (disc < 0.0)
            continue;
          const double root = std::sqrt(std::max(0.0, disc));
          double t0 = (-b - root) / (2.0 * a);
          double t1 = (-b + root) / (2.0 * a);
          if (t1 < 0.0 || t0 > 1.0)
            continue;
          t0 = std::clamp(t0, 0.0, 1.0);
          t1 = std::clamp(t1, 0.0, 1.0);
          if (t1 > t0)
            intervals.emplace_back(t0, t1);
        }
      }
    }
  }
  if (intervals.empty())
    return 0.0;

  std::sort(intervals.begin(), intervals.end());
  double covered = 0.0;
  double begin = intervals.front().first;
  double end = intervals.front().second;
  for (std::size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].first <= end) {
      end = std::max(end, intervals[i].second);
    } else {
      covered += end - begin;
      begin = intervals[i].first;
      end = intervals[i].second;
    }
  }
  covered += end - begin;
  return std::clamp(covered, 0.0, 1.0);
}

} // namespace

AnalyzerImpl::ChargeDiagnostics
AnalyzerImpl::assign_charges(const GridShape &shape,
                         const std::vector<Atom> &atoms) {
  std::fill(grid_.rho.begin(), grid_.rho.end(), 0.0);
  ChargeDiagnostics diag;
  double original_mx = 0.0;
  double original_my = 0.0;
  double original_mz = 0.0;
  double assigned_mx = 0.0;
  double assigned_my = 0.0;
  double assigned_mz = 0.0;
  auto add_charge = [&](std::size_t i, std::size_t j, std::size_t k,
                        double q) {
    if (q == 0.0)
      return;
    grid_.rho[idx3(i, j, k, shape.nx, shape.ny)] += q;
    diag.assigned_charge += q;
    assigned_mx += q * (shape.ox + static_cast<double>(i) * shape.h);
    assigned_my += q * (shape.oy + static_cast<double>(j) * shape.h);
    assigned_mz += q * (shape.oz + static_cast<double>(k) * shape.h);
  };
  for (const Atom &atom : atoms) {
    original_mx += atom.q * atom.x;
    original_my += atom.q * atom.y;
    original_mz += atom.q * atom.z;
    if (args_.charge_assignment == "gaussian") {
      const double r0 = std::max(1.0e-12, effective_surface_radius(atom));
      const double sigma = std::max(1.0e-12, args_.gaussian_sigma);
      const double influence =
          std::max(shape.h, std::max(3.0 * sigma * r0,
                                     r0 + std::max(args_.probe_radius,
                                                   args_.ion_radius)));
      const int imin = std::max(
          0, static_cast<int>(std::floor((atom.x - influence - shape.ox) /
                                         shape.h)));
      const int imax = std::min(
          static_cast<int>(shape.nx) - 1,
          static_cast<int>(std::ceil((atom.x + influence - shape.ox) /
                                     shape.h)));
      const int jmin = std::max(
          0, static_cast<int>(std::floor((atom.y - influence - shape.oy) /
                                         shape.h)));
      const int jmax = std::min(
          static_cast<int>(shape.ny) - 1,
          static_cast<int>(std::ceil((atom.y + influence - shape.oy) /
                                     shape.h)));
      const int kmin = std::max(
          0, static_cast<int>(std::floor((atom.z - influence - shape.oz) /
                                         shape.h)));
      const int kmax = std::min(
          static_cast<int>(shape.nz) - 1,
          static_cast<int>(std::ceil((atom.z + influence - shape.oz) /
                                     shape.h)));
      double norm = 0.0;
      for (int kk = kmin; kk <= kmax; ++kk) {
        const double z = shape.oz + static_cast<double>(kk) * shape.h;
        for (int jj = jmin; jj <= jmax; ++jj) {
          const double y = shape.oy + static_cast<double>(jj) * shape.h;
          for (int ii = imin; ii <= imax; ++ii) {
            const double x = shape.ox + static_cast<double>(ii) * shape.h;
            const double dx = x - atom.x;
            const double dy = y - atom.y;
            const double dz = z - atom.z;
            const double scaled2 =
                (dx * dx + dy * dy + dz * dz) / (sigma * sigma * r0 * r0);
            norm += std::exp(-std::pow(scaled2, args_.gaussian_exp));
          }
        }
      }
      if (norm > 0.0) {
        for (int kk = kmin; kk <= kmax; ++kk) {
          const double z = shape.oz + static_cast<double>(kk) * shape.h;
          for (int jj = jmin; jj <= jmax; ++jj) {
            const double y = shape.oy + static_cast<double>(jj) * shape.h;
            for (int ii = imin; ii <= imax; ++ii) {
              const double x = shape.ox + static_cast<double>(ii) * shape.h;
              const double dx = x - atom.x;
              const double dy = y - atom.y;
              const double dz = z - atom.z;
              const double scaled2 =
                  (dx * dx + dy * dy + dz * dz) / (sigma * sigma * r0 * r0);
              const double weight =
                  std::exp(-std::pow(scaled2, args_.gaussian_exp)) / norm;
              add_charge(static_cast<std::size_t>(ii),
                         static_cast<std::size_t>(jj),
                         static_cast<std::size_t>(kk), atom.q * weight);
            }
          }
        }
        continue;
      }
    }
    double gx = (atom.x - shape.ox) / shape.h;
    double gy = (atom.y - shape.oy) / shape.h;
    double gz = (atom.z - shape.oz) / shape.h;

    if (args_.charge_assignment == "trilinear") {
      gx = std::clamp(gx, 0.0, static_cast<double>(shape.nx - 1));
      gy = std::clamp(gy, 0.0, static_cast<double>(shape.ny - 1));
      gz = std::clamp(gz, 0.0, static_cast<double>(shape.nz - 1));
    }
    const double fgx = std::floor(gx);
    const double fgy = std::floor(gy);
    const double fgz = std::floor(gz);
    if (args_.charge_assignment == "delphi" &&
        (fgx < 0.0 || fgy < 0.0 || fgz < 0.0 ||
         fgx > static_cast<double>(shape.nx - 2) ||
         fgy > static_cast<double>(shape.ny - 2) ||
         fgz > static_cast<double>(shape.nz - 2))) {
      continue;
    }
    std::size_t i0 = std::min<std::size_t>(
        shape.nx - 2, static_cast<std::size_t>(std::max(0.0, fgx)));
    std::size_t j0 = std::min<std::size_t>(
        shape.ny - 2, static_cast<std::size_t>(std::max(0.0, fgy)));
    std::size_t k0 = std::min<std::size_t>(
        shape.nz - 2, static_cast<std::size_t>(std::max(0.0, fgz)));
    const double fx = clamp_fraction(gx - static_cast<double>(i0));
    const double fy = clamp_fraction(gy - static_cast<double>(j0));
    const double fz = clamp_fraction(gz - static_cast<double>(k0));
    for (std::size_t dz = 0; dz <= 1; ++dz) {
      const double wz = dz ? fz : (1.0 - fz);
      for (std::size_t dy = 0; dy <= 1; ++dy) {
        const double wy = dy ? fy : (1.0 - fy);
        for (std::size_t dx = 0; dx <= 1; ++dx) {
          const double wx = dx ? fx : (1.0 - fx);
          add_charge(i0 + dx, j0 + dy, k0 + dz, atom.q * wx * wy * wz);
        }
      }
    }
  }
  const double dmx = assigned_mx - original_mx;
  const double dmy = assigned_my - original_my;
  const double dmz = assigned_mz - original_mz;
  diag.moment_error = std::sqrt(dmx * dmx + dmy * dmy + dmz * dmz);
  maybe_write_charge_grid_diff(shape, atoms);
  return diag;
}

bool AnalyzerImpl::oracle_replay_applies(const GridShape &shape) const {
  if (args_.test_pqr.empty())
    return false;
  if (args_.test_grid_size <= 0)
    return true;
  const std::size_t n = static_cast<std::size_t>(args_.test_grid_size);
  return shape.nx == n && shape.ny == n && shape.nz == n;
}

void AnalyzerImpl::maybe_write_charge_grid_diff(
    const GridShape &shape, const std::vector<Atom> &atoms) const {
  if (args_.test_charge_grid_diff == "off" || !oracle_replay_applies(shape) ||
      charge_grid_diff_header_written_)
    return;

  const std::array<std::string, 2> candidates = {
      "pbe_oracle_charge_grid_full.csv", "pbe_oracle_charge_grid.csv"};
  std::string oracle_path;
  for (const std::string &candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      oracle_path = candidate;
      break;
    }
  }
  if (oracle_path.empty()) {
    throw std::runtime_error(
        "PBE: -pbe-test-charge-grid-diff requires "
        "pbe_oracle_charge_grid_full.csv or pbe_oracle_charge_grid.csv in the "
        "run directory.");
  }

  struct OracleCharge {
    double rho{0.0};
    double rhs{0.0};
  };
  std::unordered_map<std::size_t, OracleCharge> oracle;
  std::ifstream in(oracle_path);
  if (!in.is_open())
    throw std::runtime_error("PBE: failed to open " + oracle_path + ".");
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty " + oracle_path + ".");
  const auto header = csv_header_map(split_csv_simple(line));
  const std::string fixture_filter = read_oracle_fixture_filter();
  const double rhs = effective_rhs_scale(shape);
  std::size_t oracle_rows = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    if (!oracle_fixture_matches(fields, header, fixture_filter))
      continue;
    double vacuum_flag = 0.0;
    if (csv_double(fields, header, {"vacuum"}, vacuum_flag) &&
        std::llround(vacuum_flag) != 0)
      continue;
    std::size_t i = 0, j = 0, k = 0;
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error("PBE: " + oracle_path +
                               " row is missing i,j,k.");
    }
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error("PBE: " + oracle_path +
                               " index is outside the grid.");
    double rho = 0.0;
    double scaled_rhs = 0.0;
    const bool has_rho = csv_double(fields, header, {"rho", "q", "charge"},
                                    rho);
    const bool has_scaled =
        csv_double(fields, header,
                   {"charge_map_scaled", "rhs", "rhs_value",
                    "charge_density", "oracle_rhs"},
                   scaled_rhs);
    if (!has_rho && !has_scaled)
      throw std::runtime_error("PBE: " + oracle_path +
                               " row is missing rho/q/charge or scaled RHS.");
    if (!has_rho)
      rho = scaled_rhs / std::max(1.0e-30, rhs);
    if (!has_scaled)
      scaled_rhs = rhs * rho;
    if (rho == 0.0 && scaled_rhs == 0.0)
      continue;
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    auto &entry = oracle[id];
    entry.rho += rho;
    entry.rhs += scaled_rhs;
    ++oracle_rows;
  }
  if (oracle_rows == 0)
    throw std::runtime_error("PBE: " + oracle_path +
                             " has no nonzero charge-grid rows.");

  const bool full = args_.test_charge_grid_diff == "full";
  const std::size_t sample_limit = full ? std::numeric_limits<std::size_t>::max()
                                        : static_cast<std::size_t>(512);
  std::ofstream out("pbe_charge_grid_diff.csv", std::ios::out | std::ios::trunc);
  if (!out.is_open())
    throw std::runtime_error("PBE: failed to open pbe_charge_grid_diff.csv.");
  out << "i,j,k,native_rho,oracle_rho,native_rhs,oracle_rhs,delta_rho,"
         "delta_rhs,nearest_atom,nearest_atom_name,nearest_residue,"
         "nearest_atom_dist_A,assignment_action,charge_grid_source\n";
  out << std::setprecision(17);

  auto nearest_atom = [&](std::size_t i, std::size_t j,
                          std::size_t k) -> std::tuple<std::size_t, double> {
    if (atoms.empty())
      return {0, 0.0};
    const double x = shape.ox + static_cast<double>(i) * shape.h;
    const double y = shape.oy + static_cast<double>(j) * shape.h;
    const double z = shape.oz + static_cast<double>(k) * shape.h;
    std::size_t best = 0;
    double best_d2 = std::numeric_limits<double>::max();
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      const double dx = x - atoms[a].x;
      const double dy = y - atoms[a].y;
      const double dz = z - atoms[a].z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = a;
      }
    }
    return {best + 1, std::sqrt(best_d2)};
  };

  std::size_t diff_rows = 0;
  double delta_rho_l1 = 0.0;
  double delta_rhs_l1 = 0.0;
  double delta_rho_l2 = 0.0;
  double delta_rhs_l2 = 0.0;
  double native_charge = 0.0;
  double oracle_charge = 0.0;
  std::size_t native_nonzero = 0;
  for (std::size_t id = 0; id < shape.n; ++id) {
    const double native_rho = grid_.rho[id];
    if (native_rho != 0.0)
      ++native_nonzero;
    native_charge += native_rho;
    auto it = oracle.find(id);
    const double oracle_rho = it == oracle.end() ? 0.0 : it->second.rho;
    const double oracle_rhs = it == oracle.end() ? 0.0 : it->second.rhs;
    oracle_charge += oracle_rho;
    const double native_rhs = rhs * native_rho;
    const double delta_rho = native_rho - oracle_rho;
    const double delta_rhs = native_rhs - oracle_rhs;
    delta_rho_l1 += std::abs(delta_rho);
    delta_rhs_l1 += std::abs(delta_rhs);
    delta_rho_l2 += delta_rho * delta_rho;
    delta_rhs_l2 += delta_rhs * delta_rhs;
    if (std::abs(delta_rho) <= 1.0e-12 &&
        std::abs(delta_rhs) <= 1.0e-8)
      continue;
    if (diff_rows >= sample_limit)
      continue;
    std::size_t i = 0, j = 0, k = 0;
    i = id % shape.nx;
    j = (id / shape.nx) % shape.ny;
    k = id / (shape.nx * shape.ny);
    const auto [atom_index, atom_dist] = nearest_atom(i, j, k);
    const Atom *atom = atom_index == 0 ? nullptr : &atoms[atom_index - 1];
    std::string action = "split_weight_delta";
    if (native_rho != 0.0 && oracle_rho == 0.0)
      action = "native_only";
    else if (native_rho == 0.0 && oracle_rho != 0.0)
      action = "oracle_only";
    if (is_boundary(i, j, k, shape))
      action = "boundary_" + action;
    out << i << ',' << j << ',' << k << ',' << native_rho << ','
        << oracle_rho << ',' << native_rhs << ',' << oracle_rhs << ','
        << delta_rho << ',' << delta_rhs << ',' << atom_index << ','
        << csv_escape(atom ? atom->atom_name : "NA") << ','
        << csv_escape(atom ? atom->resname + std::to_string(atom->resnr)
                           : "NA")
        << ',' << atom_dist << ',' << csv_escape(action) << ','
        << csv_escape(oracle_path) << '\n';
    ++diff_rows;
  }

  std::ofstream summary("pbe_charge_grid_diff_summary.csv",
                        std::ios::out | std::ios::trunc);
  if (!summary.is_open())
    throw std::runtime_error(
        "PBE: failed to open pbe_charge_grid_diff_summary.csv.");
  summary << "charge_grid_source,mode,native_nonzero,oracle_nonzero,"
             "diff_rows_written,diff_rows_total_estimate,native_charge,"
             "oracle_charge,delta_charge,delta_rho_l1,delta_rhs_l1,"
             "delta_rho_rms,delta_rhs_rms\n";
  const double n = std::max<std::size_t>(1, shape.n);
  summary << csv_escape(oracle_path) << ',' << csv_escape(args_.test_charge_grid_diff)
          << ',' << native_nonzero << ',' << oracle.size() << ','
          << diff_rows << ',' << (diff_rows >= sample_limit ? "sampled" : "complete")
          << ',' << native_charge << ',' << oracle_charge << ','
          << (native_charge - oracle_charge) << ',' << delta_rho_l1 << ','
          << delta_rhs_l1 << ',' << std::sqrt(delta_rho_l2 / n) << ','
          << std::sqrt(delta_rhs_l2 / n) << '\n';

  charge_grid_diff_header_written_ = true;
}

void AnalyzerImpl::apply_oracle_surface_map(const GridShape &shape) {
  if (args_.test_surface_map != "oracle" || !oracle_replay_applies(shape))
    return;
  std::ifstream in("pbe_oracle_surface_map.csv");
  if (!in.is_open()) {
    throw std::runtime_error(
        "PBE: -pbe-test-surface-map=oracle requires "
        "pbe_oracle_surface_map.csv in the run directory.");
  }
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty pbe_oracle_surface_map.csv.");
  const auto header = csv_header_map(split_csv_simple(line));
  const bool has_inside = header.find("inside") != header.end();
  if (!has_inside) {
    throw std::runtime_error(
        "PBE: pbe_oracle_surface_map.csv must contain i,j,k,inside columns.");
  }
  std::fill(grid_.inside.begin(), grid_.inside.end(), 0U);
  std::size_t rows = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    std::size_t i = 0, j = 0, k = 0;
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_surface_map.csv row is missing i,j,k.");
    }
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error(
          "PBE: pbe_oracle_surface_map.csv index is outside the grid.");
    bool inside = false;
    const auto inside_it = header.find("inside");
    if (inside_it == header.end() || inside_it->second >= fields.size() ||
        !parse_boolish(fields[inside_it->second], inside)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_surface_map.csv inside value is invalid.");
    }
    grid_.inside[idx3(i, j, k, shape.nx, shape.ny)] = inside ? 1U : 0U;
    ++rows;
  }
  if (rows == 0)
    throw std::runtime_error("PBE: pbe_oracle_surface_map.csv has no rows.");
}

AnalyzerImpl::ChargeDiagnostics
AnalyzerImpl::assign_oracle_charge_grid(const GridShape &shape,
                                    const std::vector<Atom> &atoms) {
  if (args_.test_charge_grid != "oracle" || !oracle_replay_applies(shape))
    return assign_charges(shape, atoms);
  if (args_.test_charge_grid_diff != "off" &&
      !charge_grid_diff_header_written_) {
    (void)assign_charges(shape, atoms);
  }
  std::string oracle_path = "pbe_oracle_charge_grid.csv";
  if (std::filesystem::exists("pbe_oracle_charge_grid_full.csv"))
    oracle_path = "pbe_oracle_charge_grid_full.csv";
  std::ifstream in(oracle_path);
  if (!in.is_open()) {
    throw std::runtime_error(
        "PBE: -pbe-test-charge-grid=oracle requires "
        "pbe_oracle_charge_grid.csv or pbe_oracle_charge_grid_full.csv in "
        "the run directory.");
  }
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty " + oracle_path + ".");
  const auto header = csv_header_map(split_csv_simple(line));
  const std::string fixture_filter = read_oracle_fixture_filter();
  std::fill(grid_.rho.begin(), grid_.rho.end(), 0.0);

  ChargeDiagnostics diag;
  double original_mx = 0.0;
  double original_my = 0.0;
  double original_mz = 0.0;
  for (const Atom &atom : atoms) {
    original_mx += atom.q * atom.x;
    original_my += atom.q * atom.y;
    original_mz += atom.q * atom.z;
  }

  double assigned_mx = 0.0;
  double assigned_my = 0.0;
  double assigned_mz = 0.0;
  std::size_t rows = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    if (!oracle_fixture_matches(fields, header, fixture_filter))
      continue;
    double vacuum_flag = 0.0;
    if (csv_double(fields, header, {"vacuum"}, vacuum_flag) &&
        std::llround(vacuum_flag) != 0)
      continue;
    std::size_t i = 0, j = 0, k = 0;
    double q = 0.0;
    double scaled_rhs = 0.0;
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error(
          "PBE: oracle charge-grid row is missing i,j,k.");
    }
    const bool has_rho = csv_double(fields, header, {"rho", "q", "charge"}, q);
    const bool has_scaled =
        csv_double(fields, header,
                   {"charge_map_scaled", "rhs", "rhs_value",
                    "charge_density", "oracle_rhs"},
                   scaled_rhs);
    if (!has_rho && !has_scaled) {
      throw std::runtime_error(
          "PBE: oracle charge-grid row is missing rho/q/charge or scaled RHS.");
    }
    if (!has_rho)
      q = scaled_rhs / std::max(1.0e-30, effective_rhs_scale(shape));
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error(
          "PBE: oracle charge-grid index is outside the grid.");
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    grid_.rho[id] += q;
    diag.assigned_charge += q;
    assigned_mx += q * (shape.ox + static_cast<double>(i) * shape.h);
    assigned_my += q * (shape.oy + static_cast<double>(j) * shape.h);
    assigned_mz += q * (shape.oz + static_cast<double>(k) * shape.h);
    ++rows;
  }
  if (rows == 0)
    throw std::runtime_error("PBE: pbe_oracle_charge_grid.csv has no rows.");
  const double dmx = assigned_mx - original_mx;
  const double dmy = assigned_my - original_my;
  const double dmz = assigned_mz - original_mz;
  diag.moment_error = std::sqrt(dmx * dmx + dmy * dmy + dmz * dmz);
  return diag;
}

void AnalyzerImpl::apply_oracle_midpoint_epsilon_map(const GridShape &shape) {
  if (args_.test_midpoint_epsilon != "oracle" ||
      !oracle_replay_applies(shape))
    return;
  std::ifstream in("pbe_oracle_midpoint_epsilon.csv");
  if (!in.is_open()) {
    throw std::runtime_error(
        "PBE: -pbe-test-midpoint-epsilon=oracle requires "
        "pbe_oracle_midpoint_epsilon.csv in the run directory.");
  }
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty pbe_oracle_midpoint_epsilon.csv.");
  const auto header = csv_header_map(split_csv_simple(line));
  std::vector<unsigned char> seen(shape.n, 0U);
  std::size_t rows = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    std::size_t i = 0, j = 0, k = 0;
    double ex = 0.0, ey = 0.0, ez = 0.0;
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv row is missing i,j,k.");
    }
    if (!csv_double(fields, header, {"eps_x", "epsilon_x", "ex"}, ex) ||
        !csv_double(fields, header, {"eps_y", "epsilon_y", "ey"}, ey) ||
        !csv_double(fields, header, {"eps_z", "epsilon_z", "ez"}, ez)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv row is missing eps_x/eps_y/eps_z.");
    }
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv index is outside the grid.");
    if (!std::isfinite(ex) || !std::isfinite(ey) || !std::isfinite(ez))
      throw std::runtime_error(
          "PBE: pbe_oracle_midpoint_epsilon.csv contains non-finite epsilon.");
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    grid_.eps_x[id] = ex;
    grid_.eps_y[id] = ey;
    grid_.eps_z[id] = ez;
    seen[id] = 1U;
    ++rows;
  }
  if (rows == 0)
    throw std::runtime_error(
        "PBE: pbe_oracle_midpoint_epsilon.csv has no rows.");
  if (std::any_of(seen.begin(), seen.end(),
                  [](unsigned char value) { return value == 0U; })) {
    throw std::runtime_error(
        "PBE: pbe_oracle_midpoint_epsilon.csv must provide one row for every grid point.");
  }
}

void AnalyzerImpl::apply_oracle_operator_maps(const GridShape &shape) {
  if ((args_.test_rhs_map != "oracle" && args_.test_screen_map != "oracle" &&
       args_.test_denom_map != "oracle") ||
      !oracle_replay_applies(shape)) {
    return;
  }
  std::ifstream in("pbe_oracle_operator_stage.csv");
  if (!in.is_open()) {
    throw std::runtime_error(
        "PBE: operator oracle replay requires pbe_oracle_operator_stage.csv in the run directory.");
  }
  std::string line;
  if (!std::getline(in, line))
    throw std::runtime_error("PBE: empty pbe_oracle_operator_stage.csv.");
  const auto header = csv_header_map(split_csv_simple(line));
  const std::string fixture_filter = read_oracle_fixture_filter();
  std::vector<unsigned char> seen(shape.n, 0U);
  const double oracle_rhs_scale =
      rhs_scale(shape) * std::max(1.0e-30, delphi_epkt());
  std::size_t rows = 0;
  std::size_t applied = 0;
  while (std::getline(in, line)) {
    if (trim_copy(line).empty())
      continue;
    const auto fields = split_csv_simple(line);
    if (!oracle_fixture_matches(fields, header, fixture_filter))
      continue;
    double vacuum_flag = 0.0;
    if (csv_double(fields, header, {"vacuum"}, vacuum_flag) &&
        std::llround(vacuum_flag) != 0) {
      continue;
    }
    std::size_t i = 0, j = 0, k = 0;
    if (!csv_index(fields, header, "i", i) ||
        !csv_index(fields, header, "j", j) ||
        !csv_index(fields, header, "k", k)) {
      throw std::runtime_error(
          "PBE: pbe_oracle_operator_stage.csv row is missing i,j,k.");
    }
    if (i >= shape.nx || j >= shape.ny || k >= shape.nz)
      throw std::runtime_error(
          "PBE: pbe_oracle_operator_stage.csv index is outside the grid.");
    if (is_boundary(i, j, k, shape))
      continue;
    const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
    if (args_.test_rhs_map == "oracle") {
      double scaled_rhs = 0.0;
      if (!csv_double(fields, header,
                      {"charge_map_scaled", "rhs", "rhs_value",
                       "charge_density"},
                      scaled_rhs)) {
        throw std::runtime_error(
            "PBE: pbe_oracle_operator_stage.csv row is missing charge_map_scaled/rhs.");
      }
      grid_.rho[id] = scaled_rhs / std::max(1.0e-30, oracle_rhs_scale);
    }
    if (args_.test_screen_map == "oracle") {
      double screen = 0.0;
      if (!csv_double(fields, header, {"screen", "screen_term"}, screen)) {
        throw std::runtime_error(
            "PBE: pbe_oracle_operator_stage.csv row is missing screen.");
      }
      grid_.screen[id] = screen;
      const std::size_t xy = shape.nx * shape.ny;
      if (i > 0 && j > 0 && k > 0 && i + 1 < shape.nx &&
          j + 1 < shape.ny && k + 1 < shape.nz) {
        grid_.denom[id] = grid_.eps_x[id] + grid_.eps_x[id - 1] + grid_.eps_y[id] +
                     grid_.eps_y[id - shape.nx] + grid_.eps_z[id] +
                     grid_.eps_z[id - xy] + grid_.screen[id];
      }
    }
    if (args_.test_denom_map == "oracle") {
      double denom = 0.0;
      if (!csv_double(fields, header,
                      {"denom", "eps_sum_plus_screen",
                       "epsilon_sum_plus_screen"},
                      denom)) {
        throw std::runtime_error(
            "PBE: pbe_oracle_operator_stage.csv row is missing denom.");
      }
      grid_.denom[id] = denom;
    }
    seen[id] = 1U;
    ++rows;
    ++applied;
  }
  if (rows == 0 || applied == 0)
    throw std::runtime_error(
        "PBE: pbe_oracle_operator_stage.csv has no solvent operator rows.");
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        if (!seen[id]) {
          throw std::runtime_error(
              "PBE: pbe_oracle_operator_stage.csv must provide one solvent "
              "row for every interior grid point when operator replay is "
              "enabled.");
        }
      }
    }
  }
}

void AnalyzerImpl::build_coefficients(const GridShape &shape,
                                  const std::vector<Atom> &atoms,
                                  double eps_in, double eps_out, double kappa,
                                  TitanExecutor &executor) {
  const std::vector<Atom> surface_atoms = adjusted_surface_atoms(atoms);
  auto center_eps = [&](std::size_t id) {
    return grid_.inside[id] ? eps_in : eps_out;
  };
  const bool refine_faces = surface_uses_face_refinement(args_.surface);
  const bool delphi_vdw_faces = args_.surface == "delphi-vdw";
  const bool ses_faces = args_.surface == "delphi-ses";
  const double surface_inflate =
      surface_uses_probe_exclusion(args_.surface) ? args_.probe_radius : 0.0;
  const AtomSpatialBins face_bins =
      refine_faces && !ses_faces
          ? build_atom_bins(surface_atoms, surface_inflate, shape.h)
          : AtomSpatialBins{};
  auto segment_face_eps = [&](double x0, double y0, double z0, double x1,
                              double y1, double z1) {
    const double inside_fraction = segment_inside_fraction(
        face_bins, surface_atoms, x0, y0, z0, x1, y1, z1, surface_inflate);
    return dielectric_from_inside_fraction(inside_fraction, eps_in, eps_out);
  };
  auto sampled_face_eps = [&](double x, double y, double z, int axis) {
    auto ses_inside_at = [&](double sx, double sy, double sz) {
      const double gx = std::clamp((sx - shape.ox) / shape.h, 0.0,
                                   static_cast<double>(shape.nx - 1));
      const double gy = std::clamp((sy - shape.oy) / shape.h, 0.0,
                                   static_cast<double>(shape.ny - 1));
      const double gz = std::clamp((sz - shape.oz) / shape.h, 0.0,
                                   static_cast<double>(shape.nz - 1));
      const std::size_t i0 = static_cast<std::size_t>(std::floor(gx));
      const std::size_t j0 = static_cast<std::size_t>(std::floor(gy));
      const std::size_t k0 = static_cast<std::size_t>(std::floor(gz));
      const std::size_t i1 = std::min(i0 + 1, shape.nx - 1);
      const std::size_t j1 = std::min(j0 + 1, shape.ny - 1);
      const std::size_t k1 = std::min(k0 + 1, shape.nz - 1);
      const double tx = gx - static_cast<double>(i0);
      const double ty = gy - static_cast<double>(j0);
      const double tz = gz - static_cast<double>(k0);
      auto value = [&](std::size_t i, std::size_t j, std::size_t k) {
        return static_cast<double>(
            grid_.inside[idx3(i, j, k, shape.nx, shape.ny)] != 0U);
      };
      const double c00 = value(i0, j0, k0) * (1.0 - tx) +
                         value(i1, j0, k0) * tx;
      const double c10 = value(i0, j1, k0) * (1.0 - tx) +
                         value(i1, j1, k0) * tx;
      const double c01 = value(i0, j0, k1) * (1.0 - tx) +
                         value(i1, j0, k1) * tx;
      const double c11 = value(i0, j1, k1) * (1.0 - tx) +
                         value(i1, j1, k1) * tx;
      const double c0 = c00 * (1.0 - ty) + c10 * ty;
      const double c1 = c01 * (1.0 - ty) + c11 * ty;
      return c0 * (1.0 - tz) + c1 * tz >= 0.5;
    };
    int inside_samples = 0;
    const double q = 0.25 * shape.h;
    for (int a = -1; a <= 1; a += 2) {
      for (int b = -1; b <= 1; b += 2) {
        double sx = x;
        double sy = y;
        double sz = z;
        if (axis == 0) {
          sy += static_cast<double>(a) * q;
          sz += static_cast<double>(b) * q;
        } else if (axis == 1) {
          sx += static_cast<double>(a) * q;
          sz += static_cast<double>(b) * q;
        } else {
          sx += static_cast<double>(a) * q;
          sy += static_cast<double>(b) * q;
        }
        if (ses_faces ? ses_inside_at(sx, sy, sz)
                      : point_inside_inflated_atom(face_bins, surface_atoms, sx,
                                                   sy, sz, surface_inflate))
          ++inside_samples;
      }
    }
    const double frac_inside = static_cast<double>(inside_samples) / 4.0;
    return frac_inside * eps_in + (1.0 - frac_inside) * eps_out;
  };

  if (uses_gaussian_model()) {
    const double sigma = std::max(1.0e-12, args_.gaussian_sigma);
    const bool gaussian_cutoff = args_.surface == "gaussian-cutoff";
    const bool gaussian_cutoff_vacuum =
        gaussian_cutoff && eps_out <= eps_in + 1.0e-9;
    const double gaussian_cutoff_epsout =
        gaussian_cutoff_vacuum ? eps_in : args_.ext_diel;
    const double gaussian_regional_outer_eps =
        gaussian_cutoff ? args_.ext_diel : eps_out;
    const double grid_corner_extension = 0.5 * shape.h * std::sqrt(3.0);
    const double influence =
        std::max(shape.h, gaussian_influence_radius(surface_atoms) +
                              grid_corner_extension);
    const double influence2 = influence * influence;
    const AtomSpatialBins gaussian_bins =
        build_atom_bins(surface_atoms, std::max(0.0, influence), shape.h);
    const double gaussian_exponent = std::max(1.0e-12, args_.gaussian_exp);
    const bool gaussian_exp_two =
        std::abs(gaussian_exponent - 2.0) <= 1.0e-12;
    const int gaussian_radius_bins = static_cast<int>(
        std::ceil(std::max(shape.h, influence) / gaussian_bins.cell)) + 1;
    struct GaussianAtomCoeff {
      double x{0.0};
      double y{0.0};
      double z{0.0};
      double inv_sigma_radius2{1.0};
      double ion_radius2{0.0};
    };
    std::vector<GaussianAtomCoeff> gaussian_coeffs(surface_atoms.size());
    const double sigma2 = sigma * sigma;
    for (std::size_t atom_idx = 0; atom_idx < surface_atoms.size();
         ++atom_idx) {
      const Atom &atom = surface_atoms[atom_idx];
      const double r0 = std::max(1.0e-12, atom.radius);
      const double ion_radius =
          r0 + std::max(0.0, args_.ion_radius);
      gaussian_coeffs[atom_idx] = {atom.x, atom.y, atom.z,
                                   1.0 / (sigma2 * r0 * r0),
                                   ion_radius * ion_radius};
    }
    auto density_at = [&](double x, double y, double z) {
      const int bx0 = static_cast<int>(std::floor((x - gaussian_bins.min_x) /
                                                  gaussian_bins.cell));
      const int by0 = static_cast<int>(std::floor((y - gaussian_bins.min_y) /
                                                  gaussian_bins.cell));
      const int bz0 = static_cast<int>(std::floor((z - gaussian_bins.min_z) /
                                                  gaussian_bins.cell));
      double solvent_complement = 1.0;
      for (int dz = -gaussian_radius_bins; dz <= gaussian_radius_bins; ++dz) {
        const int bz = bz0 + dz;
        if (bz < 0 || bz >= static_cast<int>(gaussian_bins.bz))
          continue;
        for (int dy = -gaussian_radius_bins; dy <= gaussian_radius_bins; ++dy) {
          const int by = by0 + dy;
          if (by < 0 || by >= static_cast<int>(gaussian_bins.by))
            continue;
          for (int dx = -gaussian_radius_bins; dx <= gaussian_radius_bins; ++dx) {
            const int bx = bx0 + dx;
            if (bx < 0 || bx >= static_cast<int>(gaussian_bins.bx))
              continue;
            const auto &bucket = gaussian_bins.bins
                [(static_cast<std::size_t>(bz) * gaussian_bins.by +
                  static_cast<std::size_t>(by)) *
                     gaussian_bins.bx +
                 static_cast<std::size_t>(bx)];
            for (std::size_t atom_idx : bucket) {
              const GaussianAtomCoeff &atom = gaussian_coeffs[atom_idx];
              const double ddx = x - atom.x;
              const double ddy = y - atom.y;
              const double ddz = z - atom.z;
              const double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (r2 > influence2)
                continue;
              const double scaled2 = r2 * atom.inv_sigma_radius2;
              const double exponent_value =
                  gaussian_exp_two ? scaled2 * scaled2
                                   : std::pow(scaled2, gaussian_exponent);
              const double atom_density =
                  std::clamp(std::exp(-exponent_value), 0.0, 1.0);
              solvent_complement *= (1.0 - atom_density);
              if (solvent_complement <= 1.0e-15)
                return 1.0;
            }
          }
        }
      }
      return std::clamp(1.0 - solvent_complement, 0.0, 1.0);
    };
    auto ion_excluded_at = [&](double x, double y, double z) {
      if (args_.surface != "gaussian-cutoff")
        return false;
      const int bx0 = static_cast<int>(std::floor((x - gaussian_bins.min_x) /
                                                  gaussian_bins.cell));
      const int by0 = static_cast<int>(std::floor((y - gaussian_bins.min_y) /
                                                  gaussian_bins.cell));
      const int bz0 = static_cast<int>(std::floor((z - gaussian_bins.min_z) /
                                                  gaussian_bins.cell));
      for (int dz = -gaussian_radius_bins; dz <= gaussian_radius_bins; ++dz) {
        const int bz = bz0 + dz;
        if (bz < 0 || bz >= static_cast<int>(gaussian_bins.bz))
          continue;
        for (int dy = -gaussian_radius_bins; dy <= gaussian_radius_bins; ++dy) {
          const int by = by0 + dy;
          if (by < 0 || by >= static_cast<int>(gaussian_bins.by))
            continue;
          for (int dx = -gaussian_radius_bins; dx <= gaussian_radius_bins; ++dx) {
            const int bx = bx0 + dx;
            if (bx < 0 || bx >= static_cast<int>(gaussian_bins.bx))
              continue;
            const auto &bucket = gaussian_bins.bins
                [(static_cast<std::size_t>(bz) * gaussian_bins.by +
                  static_cast<std::size_t>(by)) *
                     gaussian_bins.bx +
                 static_cast<std::size_t>(bx)];
            for (std::size_t atom_idx : bucket) {
              const GaussianAtomCoeff &atom = gaussian_coeffs[atom_idx];
              const double dxp = x - atom.x;
              const double dyp = y - atom.y;
              const double dzp = z - atom.z;
              if (dxp * dxp + dyp * dyp + dzp * dzp < atom.ion_radius2)
                return true;
            }
          }
        }
      }
      return false;
    };
    const bool have_gaussian_center_density =
        last_gaussian_density_center_.size() == shape.n;
    if (!have_gaussian_center_density)
      last_gaussian_density_center_.assign(shape.n, 0.0);
    last_gaussian_density_midpoints_.assign(shape.n * 3, 0.0);
    last_gaussian_eps_center_.assign(
        shape.n, gaussian_cutoff ? gaussian_cutoff_epsout : eps_out);
    std::vector<unsigned char> gaussian_ion_excluded_initial(shape.n, 0);
    const std::size_t workers = effective_workers(executor, shape.nz);
    if (!have_gaussian_center_density) {
      const auto center_density_begin = std::chrono::steady_clock::now();
      executor.parallel_for(
          shape.nz, workers, 1,
          [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
            for (std::size_t k = z_begin; k < z_end; ++k) {
              const double z = shape.oz + static_cast<double>(k) * shape.h;
              for (std::size_t j = 0; j < shape.ny; ++j) {
                const double y = shape.oy + static_cast<double>(j) * shape.h;
                for (std::size_t i = 0; i < shape.nx; ++i) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const double x =
                      shape.ox + static_cast<double>(i) * shape.h;
                  last_gaussian_density_center_[id] = density_at(x, y, z);
                }
              }
            }
          });
      add_timing_ms(TimingBucket::EpsilonCenterDensity,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        center_density_begin)
                        .count());
    }
    if (gaussian_cutoff) {
      const auto ion_exclusion_begin = std::chrono::steady_clock::now();
      executor.parallel_for(
          shape.nz, workers, 1,
          [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
            for (std::size_t k = z_begin; k < z_end; ++k) {
              const double z = shape.oz + static_cast<double>(k) * shape.h;
              for (std::size_t j = 0; j < shape.ny; ++j) {
                const double y = shape.oy + static_cast<double>(j) * shape.h;
                for (std::size_t i = 0; i < shape.nx; ++i) {
                  const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                  const double x =
                      shape.ox + static_cast<double>(i) * shape.h;
                  gaussian_ion_excluded_initial[id] =
                      ion_excluded_at(x, y, z) ? 1U : 0U;
                }
              }
            }
          });
      add_timing_ms(TimingBucket::EpsilonIonExclusion,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        ion_exclusion_begin)
                        .count());
    }
    const auto midpoint_density_begin = std::chrono::steady_clock::now();
    executor.parallel_for(
        shape.nz, workers, 1,
        [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
          for (std::size_t k = z_begin; k < z_end; ++k) {
            const double z = shape.oz + static_cast<double>(k) * shape.h;
            for (std::size_t j = 0; j < shape.ny; ++j) {
              const double y = shape.oy + static_cast<double>(j) * shape.h;
              for (std::size_t i = 0; i < shape.nx; ++i) {
                const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                const double x = shape.ox + static_cast<double>(i) * shape.h;
                const std::size_t base = id * 3;
                if (i + 1 < shape.nx)
                  last_gaussian_density_midpoints_[base] =
                      density_at(x + 0.5 * shape.h, y, z);
                if (j + 1 < shape.ny)
                  last_gaussian_density_midpoints_[base + 1] =
                      density_at(x, y + 0.5 * shape.h, z);
                if (k + 1 < shape.nz)
                  last_gaussian_density_midpoints_[base + 2] =
                      density_at(x, y, z + 0.5 * shape.h);
              }
            }
          }
        });
    add_timing_ms(TimingBucket::EpsilonMidpointDensity,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() -
                      midpoint_density_begin)
                      .count());
    auto gaussian_eps_for_cell = [&](double density, std::size_t id) {
      if (args_.surface == "gaussian-cutoff") {
        constexpr double kDelphiGaussianSoluteDensityThreshold = 0.02;
        const double d = std::clamp(density, 0.0, 1.0);
        if (d < kDelphiGaussianSoluteDensityThreshold)
          return gaussian_cutoff_epsout;
        const bool ion_excluded =
            id < gaussian_ion_excluded_initial.size() &&
            gaussian_ion_excluded_initial[id] != 0U;
        const double regional_eps =
            ion_excluded ? std::clamp(args_.gap_diel, eps_in,
                                      gaussian_regional_outer_eps)
                         : gaussian_regional_outer_eps;
        double eps_value = d * eps_in + (1.0 - d) * regional_eps;
        const double cutoff =
            args_.surface_cutoff == "auto" ? args_.density_cutoff
                                           : args_.surface_cutoff_value;
        if (gaussian_cutoff_vacuum && cutoff > 0.0 && d < cutoff)
          eps_value = gaussian_cutoff_epsout;
        return eps_value;
      }
      return gaussian_dielectric(density, eps_in, eps_out);
    };
    const auto epsilon_finalize_begin = std::chrono::steady_clock::now();
    executor.parallel_for(
        shape.nz, workers, 1,
        [&](std::size_t, std::size_t z_begin, std::size_t z_end) {
          for (std::size_t k = z_begin; k < z_end; ++k) {
            for (std::size_t j = 0; j < shape.ny; ++j) {
              for (std::size_t i = 0; i < shape.nx; ++i) {
                const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
                const double e0 =
                    args_.surface == "gaussian-cutoff"
                        ? eps_in +
                              (gaussian_regional_outer_eps - eps_in) *
                                  (1.0 - std::clamp(
                                             last_gaussian_density_center_[id],
                                             0.0, 1.0))
                        : gaussian_eps_for_cell(
                              last_gaussian_density_center_[id], id);
                last_gaussian_eps_center_[id] = e0;
                grid_.eps_x[id] = e0;
                grid_.eps_y[id] = e0;
                grid_.eps_z[id] = e0;
                const std::size_t base = id * 3;
                if (i + 1 < shape.nx)
                  grid_.eps_x[id] = gaussian_eps_for_cell(
                      last_gaussian_density_midpoints_[base], id);
                if (j + 1 < shape.ny)
                  grid_.eps_y[id] = gaussian_eps_for_cell(
                      last_gaussian_density_midpoints_[base + 1], id);
                if (k + 1 < shape.nz)
                  grid_.eps_z[id] = gaussian_eps_for_cell(
                      last_gaussian_density_midpoints_[base + 2], id);
              }
            }
          }
        });
    if (args_.surface == "gaussian-cutoff") {
      constexpr double kDelphiGaussianSoluteDensityThreshold = 0.02;
      constexpr double kDelphiGaussianEpsMaxMinRatio = 1.02;
      const std::size_t plane = shape.nx * shape.ny;
      for (int parity = 0; parity < 2; ++parity) {
        for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
          for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
            for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
              const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
              if (static_cast<int>(id & 1U) != parity)
                continue;
              const std::size_t base = id * 3;
              const double densities[6] = {
                  last_gaussian_density_midpoints_[base],
                  last_gaussian_density_midpoints_[base + 1],
                  last_gaussian_density_midpoints_[base + 2],
                  last_gaussian_density_midpoints_[(id - 1) * 3],
                  last_gaussian_density_midpoints_[(id - shape.nx) * 3 + 1],
                  last_gaussian_density_midpoints_[(id - plane) * 3 + 2]};
              bool near_solute = false;
              for (double density : densities) {
                if (density > kDelphiGaussianSoluteDensityThreshold) {
                  near_solute = true;
                  break;
                }
              }
              if (!near_solute)
                continue;
              grid_.ion_accessible[id] = 0U;
              const double neigh_eps[6] = {
                  grid_.eps_x[id], grid_.eps_y[id], grid_.eps_z[id], grid_.eps_x[id - 1],
                  grid_.eps_y[id - shape.nx], grid_.eps_z[id - plane]};
              double eps_min = neigh_eps[0];
              double eps_max = neigh_eps[0];
              double eps_sum = 0.0;
              for (double eps_value : neigh_eps) {
                eps_min = std::min(eps_min, eps_value);
                eps_max = std::max(eps_max, eps_value);
                eps_sum += eps_value;
              }
              if (eps_min > 1.0e-6 &&
                  eps_max / eps_min < kDelphiGaussianEpsMaxMinRatio) {
                const double eps_mean = eps_sum / 6.0;
                grid_.eps_x[id] = eps_mean;
                grid_.eps_y[id] = eps_mean;
                grid_.eps_z[id] = eps_mean;
                grid_.eps_x[id - 1] = eps_mean;
                grid_.eps_y[id - shape.nx] = eps_mean;
                grid_.eps_z[id - plane] = eps_mean;
                last_gaussian_eps_center_[id] = eps_mean;
              }
            }
          }
        }
      }
    }
    add_timing_ms(TimingBucket::EpsilonFinalize,
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() -
                      epsilon_finalize_begin)
                      .count());
    last_gaussian_surface_.assign(shape.n, 0.0);
    last_gaussian_grad_surface_.assign(shape.n * 3, 0.0);
    if (args_.surface == "gaussian") {
      const double surface_exp =
          4.0 / std::max(1.0e-12, args_.gaussian_exp);
      const double approx = 1.0e-12;
      for (std::size_t id = 0; id < shape.n; ++id) {
        const double density =
            std::clamp(last_gaussian_density_center_[id], 0.0, 1.0);
        if (density <= approx)
          continue;
        const double term = 1.0 / density - 1.0;
        if (term <= approx && std::abs(surface_exp - std::round(surface_exp)) >
                                   1.0e-12) {
          last_gaussian_surface_[id] = 1.0;
          continue;
        }
        last_gaussian_surface_[id] =
            1.0 / (1.0 + std::pow(std::max(0.0, term), surface_exp));
      }
    }
    double eps_min = std::numeric_limits<double>::max();
    double eps_max = std::numeric_limits<double>::lowest();
    double eps_sum = 0.0;
    std::size_t eps_count = 0;
    for (std::size_t id = 0; id < shape.n; ++id) {
      const double values[3] = {grid_.eps_x[id], grid_.eps_y[id], grid_.eps_z[id]};
      for (double eps_value : values) {
        if (!std::isfinite(eps_value))
          continue;
        eps_min = std::min(eps_min, eps_value);
        eps_max = std::max(eps_max, eps_value);
        eps_sum += eps_value;
        ++eps_count;
      }
    }
    last_gaussian_eps_min_ = eps_count > 0 ? eps_min : 0.0;
    last_gaussian_eps_max_ = eps_count > 0 ? eps_max : 0.0;
    last_gaussian_eps_mean_ =
        eps_count > 0 ? eps_sum / static_cast<double>(eps_count) : 0.0;
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
              double screen = effective_screen_term(shape, eps_out, kappa, id);
              if (args_.biomodel == "rpbe" && kappa > 0.0 &&
                  last_gaussian_surface_.size() == shape.n) {
                const double debye =
                    delphi_debye_length_a(args_.salt, args_.temp, eps_out);
                const double full_screen =
                    debye > 0.0 && std::isfinite(debye)
                        ? eps_out * shape.h * shape.h / (debye * debye)
                        : 0.0;
                screen =
                    full_screen *
                    (1.0 - std::clamp(last_gaussian_surface_[id], 0.0, 1.0));
              }
              grid_.screen[id] = screen;
          grid_.denom[id] = grid_.eps_x[id] + grid_.eps_x[id - 1] + grid_.eps_y[id] +
                       grid_.eps_y[id - shape.nx] + grid_.eps_z[id] +
                       grid_.eps_z[id - shape.nx * shape.ny] + screen;
        }
      }
    }
    if (eps_out > eps_in + 1.0e-9)
      apply_oracle_operator_maps(shape);
    return;
  }

  if (delphi_vdw_faces) {
    const DelphiVdwMediaMap media_map =
        build_delphi_vdw_media_map(shape, surface_atoms, executor);
    apply_delphi_vdw_media_map(shape, media_map, eps_in, eps_out);

    apply_oracle_midpoint_epsilon_map(shape);
    for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
      for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
        for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
          const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
          const double screen = effective_screen_term(shape, eps_out, kappa, id);
          grid_.screen[id] = screen;
          grid_.denom[id] = grid_.eps_x[id] + grid_.eps_x[id - 1] + grid_.eps_y[id] +
                       grid_.eps_y[id - shape.nx] + grid_.eps_z[id] +
                       grid_.eps_z[id - shape.nx * shape.ny] + screen;
        }
      }
    }
    if (eps_out > eps_in + 1.0e-9)
      apply_oracle_operator_maps(shape);
    return;
  }

  for (std::size_t k = 0; k < shape.nz; ++k) {
    for (std::size_t j = 0; j < shape.ny; ++j) {
      for (std::size_t i = 0; i < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double e0 = center_eps(id);
        grid_.eps_x[id] = e0;
        grid_.eps_y[id] = e0;
        grid_.eps_z[id] = e0;
        if (i + 1 < shape.nx) {
          const double e1 = center_eps(id + 1);
          const double x_face =
              shape.ox + (static_cast<double>(i) + 0.5) * shape.h;
          const double y_face = shape.oy + static_cast<double>(j) * shape.h;
          const double z_face = shape.oz + static_cast<double>(k) * shape.h;
          grid_.eps_x[id] =
              (delphi_vdw_faces && grid_.inside[id] != grid_.inside[id + 1])
                  ? segment_face_eps(shape.ox + static_cast<double>(i) *
                                                    shape.h,
                                     y_face, z_face,
                                     shape.ox + static_cast<double>(i + 1) *
                                                    shape.h,
                                     y_face, z_face)
              : (refine_faces && grid_.inside[id] != grid_.inside[id + 1])
                  ? sampled_face_eps(x_face, y_face, z_face, 0)
                  : harmonic_eps(e0, e1);
        }
        if (j + 1 < shape.ny) {
          const double e1 = center_eps(id + shape.nx);
          const double x_face = shape.ox + static_cast<double>(i) * shape.h;
          const double y_face =
              shape.oy + (static_cast<double>(j) + 0.5) * shape.h;
          const double z_face = shape.oz + static_cast<double>(k) * shape.h;
          grid_.eps_y[id] =
              (delphi_vdw_faces && grid_.inside[id] != grid_.inside[id + shape.nx])
                  ? segment_face_eps(x_face,
                                     shape.oy + static_cast<double>(j) *
                                                    shape.h,
                                     z_face, x_face,
                                     shape.oy + static_cast<double>(j + 1) *
                                                    shape.h,
                                     z_face)
              : (refine_faces && grid_.inside[id] != grid_.inside[id + shape.nx])
                  ? sampled_face_eps(x_face, y_face, z_face, 1)
                  : harmonic_eps(e0, e1);
        }
        if (k + 1 < shape.nz) {
          const std::size_t dz = shape.nx * shape.ny;
          const double e1 = center_eps(id + dz);
          const double x_face = shape.ox + static_cast<double>(i) * shape.h;
          const double y_face = shape.oy + static_cast<double>(j) * shape.h;
          const double z_face =
              shape.oz + (static_cast<double>(k) + 0.5) * shape.h;
          grid_.eps_z[id] =
              (delphi_vdw_faces && grid_.inside[id] != grid_.inside[id + dz])
                  ? segment_face_eps(x_face, y_face,
                                     shape.oz + static_cast<double>(k) *
                                                    shape.h,
                                     x_face, y_face,
                                     shape.oz + static_cast<double>(k + 1) *
                                                    shape.h)
              : (refine_faces && grid_.inside[id] != grid_.inside[id + dz])
                  ? sampled_face_eps(x_face, y_face, z_face, 2)
                  : harmonic_eps(e0, e1);
        }
      }
    }
  }

  apply_oracle_midpoint_epsilon_map(shape);
  for (std::size_t k = 1; k + 1 < shape.nz; ++k) {
    for (std::size_t j = 1; j + 1 < shape.ny; ++j) {
      for (std::size_t i = 1; i + 1 < shape.nx; ++i) {
        const std::size_t id = idx3(i, j, k, shape.nx, shape.ny);
        const double screen = effective_screen_term(shape, eps_out, kappa, id);
        grid_.screen[id] = screen;
        grid_.denom[id] = grid_.eps_x[id] + grid_.eps_x[id - 1] + grid_.eps_y[id] +
                     grid_.eps_y[id - shape.nx] + grid_.eps_z[id] +
                     grid_.eps_z[id - shape.nx * shape.ny] + screen;
      }
    }
  }
  if (eps_out > eps_in + 1.0e-9)
    apply_oracle_operator_maps(shape);
}

} // namespace TitanPBE

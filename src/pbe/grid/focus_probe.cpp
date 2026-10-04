#include "tabi/analyzer_impl.h"

#include "pbe/grid/grid_common.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TitanPBE {
namespace {

constexpr double kKcalToKJ = 4.184;
constexpr double kKJToKcal = 1.0 / kKcalToKJ;
constexpr double kKBKJPerMolK = 0.00831446261815324;

double clamp_fraction(double value) noexcept {
  if (value < 0.0)
    return 0.0;
  if (value > 1.0)
    return 1.0;
  return value;
}

template <typename Shape>
bool is_boundary(std::size_t i, std::size_t j, std::size_t k,
                 const Shape &shape) noexcept {
  return i == 0 || j == 0 || k == 0 || i + 1 == shape.nx ||
         j + 1 == shape.ny || k + 1 == shape.nz;
}

bool parse_double_token(const std::string &token, double &value) {
  try {
    std::size_t used = 0;
    const double parsed = std::stod(token, &used);
    if (used != token.size() || !std::isfinite(parsed))
      return false;
    value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

std::string uppercase_letters(std::string value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    if (std::isalpha(static_cast<unsigned char>(ch))) {
      out.push_back(static_cast<char>(
          std::toupper(static_cast<unsigned char>(ch))));
    }
  }
  return out;
}

std::string csv_escape(const std::string &value) {
  bool needs_quotes = false;
  for (char ch : value) {
    if (ch == ',' || ch == '"' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes)
    return value;
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char ch : value) {
    if (ch == '"')
      escaped.push_back('"');
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

} // namespace

std::vector<AnalyzerImpl::Atom>
AnalyzerImpl::load_probe_atoms(const std::string &path) const {
  std::vector<Atom> atoms;
  if (path.empty())
    return atoms;
  std::ifstream in(path);
  if (!in.is_open())
    throw std::runtime_error("PBE: failed to open probe file: " + path);
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      continue;
    const bool atom_record = line.compare(first, 4, "ATOM") == 0 ||
                             line.compare(first, 6, "HETATM") == 0;
    if (!atom_record)
      continue;
    std::istringstream iss(line);
    std::vector<double> numeric_tokens;
    std::string token;
    while (iss >> token) {
      double value = 0.0;
      if (parse_double_token(token, value))
        numeric_tokens.push_back(value);
    }
    if (numeric_tokens.size() < 5) {
      std::ostringstream msg;
      msg << "PBE: malformed probe file line " << line_no << " in " << path
          << ". Expected x y z charge radius numeric fields.";
      throw std::runtime_error(msg.str());
    }
    const std::size_t n = numeric_tokens.size();
    const double x = numeric_tokens[n - 5];
    const double y = numeric_tokens[n - 4];
    const double z = numeric_tokens[n - 3];
    const double q = numeric_tokens[n - 2];
    const double r = numeric_tokens[n - 1];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(q) || !std::isfinite(r) || r <= 0.0) {
      std::ostringstream msg;
      msg << "PBE: invalid coordinate, charge, or radius on probe file line "
          << line_no << " in " << path << ".";
      throw std::runtime_error(msg.str());
    }
    std::string element;
    std::string atom_name;
    std::string resname;
    std::string chain;
    int resnr = 0;
    {
      std::istringstream token_stream(line);
      std::vector<std::string> tokens;
      while (token_stream >> token)
        tokens.push_back(token);
      if (tokens.size() >= 3)
        atom_name = tokens[2];
      if (tokens.size() >= 4)
        resname = tokens[3];
      if (tokens.size() >= 5) {
        double maybe_resnr = 0.0;
        if (parse_double_token(tokens[4], maybe_resnr)) {
          resnr = static_cast<int>(std::lround(maybe_resnr));
        } else {
          chain = tokens[4];
          if (tokens.size() >= 6 && parse_double_token(tokens[5], maybe_resnr))
            resnr = static_cast<int>(std::lround(maybe_resnr));
        }
      }
      if (!tokens.empty()) {
        double maybe_element_number = 0.0;
        if (!parse_double_token(tokens.back(), maybe_element_number))
          element = tokens.back();
      }
    }
    atoms.push_back({x, y, z, q, r, element, atom_name, resname, resnr, chain});
  }
  return atoms;
}

bool AnalyzerImpl::interpolate_field_value(const GridShape &shape,
                                       const AlignedDoubleVector &field,
                                       double x, double y, double z,
                                       double &value) const {
  if (shape.nx < 2 || shape.ny < 2 || shape.nz < 2 ||
      field.size() != shape.n)
    return false;
  const double gx = (x - shape.ox) / shape.h;
  const double gy = (y - shape.oy) / shape.h;
  const double gz = (z - shape.oz) / shape.h;
  if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
      gx > static_cast<double>(shape.nx - 1) ||
      gy > static_cast<double>(shape.ny - 1) ||
      gz > static_cast<double>(shape.nz - 1))
    return false;

  const std::size_t i0 = std::min<std::size_t>(
      shape.nx - 2, static_cast<std::size_t>(std::floor(gx)));
  const std::size_t j0 = std::min<std::size_t>(
      shape.ny - 2, static_cast<std::size_t>(std::floor(gy)));
  const std::size_t k0 = std::min<std::size_t>(
      shape.nz - 2, static_cast<std::size_t>(std::floor(gz)));
  const double fx = clamp_fraction(gx - static_cast<double>(i0));
  const double fy = clamp_fraction(gy - static_cast<double>(j0));
  const double fz = clamp_fraction(gz - static_cast<double>(k0));

  value = 0.0;
  for (std::size_t dz = 0; dz <= 1; ++dz) {
    const double wz = dz ? fz : (1.0 - fz);
    for (std::size_t dy = 0; dy <= 1; ++dy) {
      const double wy = dy ? fy : (1.0 - fy);
      for (std::size_t dx = 0; dx <= 1; ++dx) {
        const double wx = dx ? fx : (1.0 - fx);
        value += wx * wy * wz *
                 field[idx3(i0 + dx, j0 + dy, k0 + dz, shape.nx, shape.ny)];
      }
    }
  }
  return true;
}

void AnalyzerImpl::seed_from_focus_source(const GridShape &shape,
                                      const GridShape &source_shape,
                                      const AlignedDoubleVector &source_phi,
                                      TitanExecutor &executor) {
  seed_from_focus_source(shape, source_shape, source_phi, executor, false);
}

void AnalyzerImpl::seed_from_focus_source(const GridShape &shape,
                                      const GridShape &source_shape,
                                      const AlignedDoubleVector &source_phi,
                                      TitanExecutor &executor,
                                      bool boundary_only) {
  if (source_phi.empty())
    return;
  const std::size_t workers = effective_workers(executor, shape.nz);
  executor.parallel_for(shape.nz, workers, 1,
                        [&](std::size_t, std::size_t z_begin,
                            std::size_t z_end) {
    for (std::size_t k = z_begin; k < z_end; ++k) {
      const double z = shape.oz + static_cast<double>(k) * shape.h;
      for (std::size_t j = 0; j < shape.ny; ++j) {
        const double y = shape.oy + static_cast<double>(j) * shape.h;
        for (std::size_t i = 0; i < shape.nx; ++i) {
          if (boundary_only && !is_boundary(i, j, k, shape))
            continue;
          const double x = shape.ox + static_cast<double>(i) * shape.h;
          double seeded = 0.0;
          if (interpolate_field_value(source_shape, source_phi, x, y, z,
                                      seeded)) {
            grid_.phi[idx3(i, j, k, shape.nx, shape.ny)] = seeded;
          }
        }
      }
    }
  });
}

std::vector<double>
AnalyzerImpl::interpolate_atoms(const GridShape &shape,
                            const std::vector<Atom> &atoms) const {
  std::vector<double> potentials;
  potentials.reserve(atoms.size());
  for (const Atom &atom : atoms) {
    double gx = (atom.x - shape.ox) / shape.h;
    double gy = (atom.y - shape.oy) / shape.h;
    double gz = (atom.z - shape.oz) / shape.h;
    gx = std::clamp(gx, 0.0, static_cast<double>(shape.nx - 1));
    gy = std::clamp(gy, 0.0, static_cast<double>(shape.ny - 1));
    gz = std::clamp(gz, 0.0, static_cast<double>(shape.nz - 1));
    std::size_t i0 = std::min<std::size_t>(
        shape.nx - 2, static_cast<std::size_t>(std::floor(gx)));
    std::size_t j0 = std::min<std::size_t>(
        shape.ny - 2, static_cast<std::size_t>(std::floor(gy)));
    std::size_t k0 = std::min<std::size_t>(
        shape.nz - 2, static_cast<std::size_t>(std::floor(gz)));
    const double fx = clamp_fraction(gx - static_cast<double>(i0));
    const double fy = clamp_fraction(gy - static_cast<double>(j0));
    const double fz = clamp_fraction(gz - static_cast<double>(k0));
    double value = 0.0;
    for (std::size_t dz = 0; dz <= 1; ++dz) {
      const double wz = dz ? fz : (1.0 - fz);
      for (std::size_t dy = 0; dy <= 1; ++dy) {
        const double wy = dy ? fy : (1.0 - fy);
        for (std::size_t dx = 0; dx <= 1; ++dx) {
          const double wx = dx ? fx : (1.0 - fx);
          value += wx * wy * wz *
                   grid_.phi[idx3(i0 + dx, j0 + dy, k0 + dz, shape.nx, shape.ny)];
        }
      }
    }
    potentials.push_back(value);
  }
  return potentials;
}

std::string AnalyzerImpl::maybe_write_probe_sites(const FrameResult &result,
                                              const std::vector<Atom> &atoms,
                                              const ReactionSolve &solve) {
  if (args_.probe_residue.empty() && args_.probe_site.empty() &&
      args_.focus != "delphi")
    return "off";
  auto uppercase_alnum = [](const std::string &value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
      if (std::isalnum(static_cast<unsigned char>(ch)))
        out.push_back(static_cast<char>(
            std::toupper(static_cast<unsigned char>(ch))));
    }
    return out;
  };
  auto digits_only = [](const std::string &value) {
    std::string out;
    for (char ch : value) {
      if (std::isdigit(static_cast<unsigned char>(ch)))
        out.push_back(ch);
    }
    return out;
  };
  auto site_tokens = [&]() {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream input(args_.probe_site);
    while (std::getline(input, token, ',')) {
      const std::string cleaned = uppercase_alnum(token);
      if (!cleaned.empty())
        tokens.push_back(cleaned);
    }
    return tokens;
  }();
  const std::string requested_residue = uppercase_alnum(args_.probe_residue);
  const std::string requested_residue_name = uppercase_letters(args_.probe_residue);
  const std::string requested_residue_number = digits_only(args_.probe_residue);
  auto residue_matches = [&](const Atom &atom) {
    if (requested_residue.empty())
      return true;
    const std::string atom_resname = uppercase_letters(atom.resname);
    const std::string atom_resid = std::to_string(atom.resnr);
    const std::string atom_key = atom_resname + atom_resid;
    if (requested_residue == atom_key || requested_residue == atom_resname)
      return true;
    if (!requested_residue_name.empty() &&
        requested_residue_name == atom_resname) {
      if (requested_residue_number.empty() ||
          requested_residue_number == atom_resid)
        return true;
    }
    return false;
  };
  auto site_matches = [&](const Atom &atom) {
    if (site_tokens.empty())
      return true;
    const std::string atom_name = uppercase_alnum(atom.atom_name);
    const std::string element = uppercase_letters(atom.element);
    for (const std::string &token : site_tokens) {
      if (token == "A" || token == "F")
        return true;
      if (token == "P") {
        if (element == "N" || element == "O" || element == "S" ||
            element == "P")
          return true;
      } else if (token == "Q") {
        if (std::abs(atom.q) >= 0.25)
          return true;
      } else if (atom_name == token || atom_name.find(token) != std::string::npos) {
        return true;
      }
    }
    return false;
  };
  double energy_kj_mol = 0.0;
  std::size_t matched = 0;
  std::string matched_ids;
  std::string matched_groups;
  std::string matching_semantics = "delphi_residue_site_groups";
  std::string status;
  if (solve.probe_energy_status != "off") {
    energy_kj_mol = solve.probe_energy_kj_mol;
    matched = solve.probe_matched_atoms;
    matched_ids = solve.probe_matched_ids;
    matched_groups = solve.probe_matched_groups;
    matching_semantics = solve.probe_matching_semantics;
    status = solve.probe_energy_status;
  } else {
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      const Atom &atom = atoms[i];
      if (!residue_matches(atom) || !site_matches(atom))
        continue;
      if (i < solve.atom_reaction_kj_mol.size())
        energy_kj_mol += solve.atom_reaction_kj_mol[i];
      if (!matched_ids.empty())
        matched_ids += ';';
      matched_ids += std::to_string(i + 1);
      if (!matched_groups.empty())
        matched_groups += ';';
      matched_groups += uppercase_letters(atom.resname) +
                        std::to_string(atom.resnr) + ':' +
                        uppercase_alnum(atom.atom_name);
      ++matched;
    }
    status = matched > 0 ? "ok" : "no_matching_atoms";
  }
  const std::string filename = "pbe_probe_sites.csv";
  const bool need_header =
      !std::filesystem::exists(filename) ||
      std::filesystem::file_size(filename) == 0;
  std::ofstream out(filename, std::ios::out | std::ios::app);
  if (!out.is_open())
    return "write_failed";
  if (need_header) {
    out << "trajectory_id,frame_index,global_index,time_ps,probe_residue,"
           "probe_site,matched_atoms,E_probe_kJmol,E_probe_kcalmol,"
           "E_probe_kT,status,solver_mode,focus_parent_grid,focus_child_grid,"
           "matched_atom_ids,matched_groups,matching_semantics\n";
  }
  const double kt = kKBKJPerMolK * std::max(1.0, args_.temp);
  out << result.trajectory_id << ',' << result.frame_index << ','
      << result.global_index << ',' << std::setprecision(10) << result.time_ps
      << ',' << csv_escape(args_.probe_residue) << ','
      << csv_escape(args_.probe_site) << ',' << matched << ','
      << energy_kj_mol << ',' << energy_kj_mol * kKJToKcal << ','
      << (kt > 0.0 ? energy_kj_mol / kt : 0.0) << ',' << status << ','
      << result.solver_mode << ',' << csv_escape(result.focus_parent_grid)
      << ',' << csv_escape(result.focus_child_grid) << ','
      << csv_escape(matched_ids) << ',' << csv_escape(matched_groups) << ','
      << csv_escape(matching_semantics) << '\n';
  return status;
}

} // namespace TitanPBE

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

using analyzer_impl_detail::kCoulombKcalPerMol;
using analyzer_impl_detail::parse_double_token;

namespace {

double fallback_radius_from_mass(double mass) noexcept {
  if (!std::isfinite(mass) || mass <= 0.0)
    return 0.0;
  if (mass < 2.5)
    return 1.20; // H
  if (mass < 13.5)
    return 1.70; // C
  if (mass < 15.5)
    return 1.55; // N
  if (mass < 18.5)
    return 1.52; // O/F
  if (mass < 32.5)
    return 1.80; // P/S
  return 1.80;
}

} // namespace

void AnalyzerImpl::configure(const arguments &args) {
  args_ = args.pbe;
  args_.pbsa_mol1_type = args.mol_1_type;
  args_.pbsa_mol2_type = args.mol_2_type;
  configured_threads_ = std::max<std::size_t>(1, args.nt);
  reset_grid_anchor();
  last_stability_state_ = StabilityFrameState{};
  last_stability_medium_state_ = StabilityMediumState{};
  last_stability_ensemble_ = StabilityEnsembleState{};
  stability_hysteresis_cells_kept_frame_ = 0;
  stability_hysteresis_cells_released_frame_ = 0;
  if (args_.pbsa) {
    args_.flag_pbe = true;
    args_.dgsolv = true;
  }
}

void AnalyzerImpl::begin_trajectory(std::size_t trajectory_id,
                                    const std::string &source_label) {
  if (!enabled())
    return;
  current_trajectory_id_ = trajectory_id;
  current_source_label_ = source_label;
  reset_grid_anchor();
  last_stability_state_ = StabilityFrameState{};
  last_stability_medium_state_ = StabilityMediumState{};
  last_stability_ensemble_ = StabilityEnsembleState{};
  stability_hysteresis_cells_kept_frame_ = 0;
  stability_hysteresis_cells_released_frame_ = 0;
}

void AnalyzerImpl::end_trajectory(std::fstream &log) {
  if (!enabled())
    return;
  if (log.is_open()) {
    TitanUI::log_kv(log, "PBE Trajectory Complete",
                    std::to_string(current_trajectory_id_) + " " +
                        current_source_label_);
  }
}

void AnalyzerImpl::collect_atoms(const MOL &mol,
                                 const Multi_MOL_parameters &params,
                                 std::vector<Atom> &atoms,
                                 const char *label) const {
  const std::size_t n = mol.size();
  if (params.size() < n) {
    throw std::runtime_error(std::string("PBE: ") + label +
                             " parameter table is shorter than coordinates.");
  }
  const auto xs = mol.get_pos_x();
  const auto ys = mol.get_pos_y();
  const auto zs = mol.get_pos_z();
  for (std::size_t i = 0; i < n; ++i) {
    const double q = params.charge[i];
    double r = params.radius[i];
    if ((!std::isfinite(r) || r <= 0.0) && i < params.mass.size())
      r = fallback_radius_from_mass(params.mass[i]);
    if (!std::isfinite(q) || !std::isfinite(r) || r <= 0.0) {
      std::ostringstream msg;
      msg << "PBE: missing/non-finite charge or VDW radius in " << label
          << " atom " << (i + 1)
          << ". Check lib_ff assignment before running -pbe.";
      throw std::runtime_error(msg.str());
    }
    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) ||
        !std::isfinite(zs[i])) {
      std::ostringstream msg;
      msg << "PBE: non-finite coordinate in " << label << " atom " << (i + 1)
          << ".";
      throw std::runtime_error(msg.str());
    }
    atoms.push_back({xs[i],
                     ys[i],
                     zs[i],
                     q,
                     r,
                     {},
                     {},
                     {},
                     0,
                     {},
                     static_cast<int>(i + 1)});
  }
}

void AnalyzerImpl::collect_test_pqr_atoms(std::vector<Atom> &atoms) const {
  collect_pqr_atoms(args_.test_pqr, atoms);
}

void AnalyzerImpl::collect_pqr_atoms(const std::string &pqr_path,
                                     std::vector<Atom> &atoms) const {
  std::ifstream in(pqr_path);
  if (!in.is_open()) {
    throw std::runtime_error("PBE: failed to open test PQR fixture: " +
                             pqr_path);
  }

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
      msg << "PBE: malformed PQR fixture line " << line_no << " in " << pqr_path
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
      msg << "PBE: invalid coordinate, charge, or radius on PQR fixture line "
          << line_no << " in " << pqr_path << ".";
      throw std::runtime_error(msg.str());
    }
    std::string element;
    std::string atom_name;
    std::string resname;
    std::string chain;
    int resnr = 0;
    int serial = static_cast<int>(atoms.size() + 1U);
    {
      std::istringstream token_stream(line);
      std::vector<std::string> tokens;
      while (token_stream >> token)
        tokens.push_back(token);
      if (tokens.size() >= 2) {
        double maybe_serial = 0.0;
        if (!parse_double_token(tokens[1], maybe_serial)) {
          std::ostringstream msg;
          msg << "PBE: invalid PQR atom serial on line " << line_no << " in "
              << pqr_path << ".";
          throw std::runtime_error(msg.str());
        }
        serial = static_cast<int>(std::lround(maybe_serial));
      }
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
    atoms.push_back(
        {x, y, z, q, r, element, atom_name, resname, resnr, chain, serial});
  }

  if (atoms.empty()) {
    throw std::runtime_error("PBE: no ATOM/HETATM records found in test PQR "
                             "fixture: " +
                             pqr_path);
  }
}

std::string
AnalyzerImpl::pqr_path_for_component(const std::string &component) const {
  if (component == "complex")
    return args_.test_pqr;
  if (component == "receptor")
    return args_.binding_receptor_pqr;
  if (component == "ligand")
    return args_.binding_ligand_pqr;
  return {};
}

std::string
AnalyzerImpl::pqr_sha256_for_component(const std::string &component) const {
  const std::string path = pqr_path_for_component(component);
  if (path.empty())
    return {};
  const auto cached = pqr_sha256_cache_.find(path);
  if (cached != pqr_sha256_cache_.end())
    return cached->second;
  const std::string digest = file_sha256_hex(path);
  pqr_sha256_cache_.emplace(path, digest);
  return digest;
}

void AnalyzerImpl::validate_binding_component_union(
    const std::vector<Atom> &complex_atoms,
    const std::vector<Atom> &receptor_atoms,
    const std::vector<Atom> &ligand_atoms) const {
  if (complex_atoms.size() != receptor_atoms.size() + ligand_atoms.size()) {
    throw std::runtime_error(
        "PBE: binding component atom counts do not reproduce the complex.");
  }
  auto atom_key = [](const Atom &atom) {
    std::ostringstream key;
    key << std::setprecision(17) << atom.serial << '|' << atom.chain << '|'
        << atom.resnr << '|' << atom.resname << '|' << atom.atom_name << '|'
        << atom.element << '|' << atom.x << '|' << atom.y << '|' << atom.z
        << '|' << atom.q << '|' << atom.radius;
    return key.str();
  };
  std::vector<std::string> complex_keys;
  std::vector<std::string> component_keys;
  complex_keys.reserve(complex_atoms.size());
  component_keys.reserve(complex_atoms.size());
  for (const Atom &atom : complex_atoms)
    complex_keys.push_back(atom_key(atom));
  for (const Atom &atom : receptor_atoms)
    component_keys.push_back(atom_key(atom));
  for (const Atom &atom : ligand_atoms)
    component_keys.push_back(atom_key(atom));
  std::sort(complex_keys.begin(), complex_keys.end());
  std::sort(component_keys.begin(), component_keys.end());
  if (complex_keys != component_keys) {
    throw std::runtime_error(
        "PBE: receptor and ligand PQRs are not an atomwise partition of the "
        "complex PQR.");
  }
}

double AnalyzerImpl::intercomponent_coulomb_energy_kcal(
    const std::vector<Atom> &receptor_atoms,
    const std::vector<Atom> &ligand_atoms, double eps) const {
  const double dielectric = std::max(1.0e-12, eps);
  double energy = 0.0;
  for (const Atom &receptor_atom : receptor_atoms) {
    for (const Atom &ligand_atom : ligand_atoms) {
      const double dx = receptor_atom.x - ligand_atom.x;
      const double dy = receptor_atom.y - ligand_atom.y;
      const double dz = receptor_atom.z - ligand_atom.z;
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r > 1.0e-12) {
        energy += kCoulombKcalPerMol * receptor_atom.q * ligand_atom.q /
                  (dielectric * r);
      }
    }
  }
  return energy;
}


} // namespace TitanPBE

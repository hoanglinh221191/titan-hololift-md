#include "energy/group_bonds.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace titan_group_bonds {
namespace {

// Largest bonded distance the scan can report: 1.2 x (P + P).
constexpr double kBondScale = 1.2;
constexpr double kCellWidthAngstrom = 2.6;

struct CellKey {
  std::int64_t x{}, y{}, z{};
  bool operator==(const CellKey &) const noexcept = default;
};

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const noexcept {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

CellKey cell_of(const MOL &mol, std::size_t atom) noexcept {
  return {static_cast<std::int64_t>(std::floor(mol.pos.x[atom] / kCellWidthAngstrom)),
          static_cast<std::int64_t>(std::floor(mol.pos.y[atom] / kCellWidthAngstrom)),
          static_cast<std::int64_t>(std::floor(mol.pos.z[atom] / kCellWidthAngstrom))};
}

bool finite_at(const MOL &mol, std::size_t atom) noexcept {
  return std::isfinite(mol.pos.x[atom]) && std::isfinite(mol.pos.y[atom]) &&
         std::isfinite(mol.pos.z[atom]);
}

// Covalent radius per atom; 0 for skipped atoms (unknown element, or the
// only atom of its residue: an ion).
std::vector<double> radii_of(const MOL &mol, const std::vector<mol_info> &info) {
  std::vector<double> radii(mol.size(), 0.0);
  if (info.size() != mol.size())
    return radii;
  std::map<std::tuple<std::string, std::size_t, std::string>, std::size_t> residue_size;
  for (const auto &atom : info)
    ++residue_size[{atom.get_chain(), atom.get_resnr(), atom.get_res_name()}];
  for (std::size_t i = 0; i < info.size(); ++i) {
    const auto key = std::make_tuple(info[i].get_chain(), info[i].get_resnr(),
                                     info[i].get_res_name());
    if (residue_size[key] > 1)
      radii[i] = covalent_radius_from_name(info[i].get_atom());
  }
  return radii;
}

std::string label(const mol_info &atom) {
  std::ostringstream out;
  out << atom.get_chain() << ':' << atom.get_resnr() << ':' << atom.get_res_name() << ' '
      << atom.get_atom();
  return out.str();
}

} // namespace

double covalent_radius_from_name(std::string_view atom_name) noexcept {
  std::size_t i = 0;
  while (i < atom_name.size() &&
         (std::isdigit(static_cast<unsigned char>(atom_name[i])) || atom_name[i] == ' '))
    ++i;
  if (i >= atom_name.size())
    return 0.0;
  const char first = static_cast<char>(std::toupper(static_cast<unsigned char>(atom_name[i])));
  const char second = i + 1 < atom_name.size()
                          ? static_cast<char>(std::toupper(static_cast<unsigned char>(atom_name[i + 1])))
                          : '\0';
  // CL/BR: chlorine/bromine (halogens and ions); not scanned.
  if ((first == 'C' && second == 'L') || (first == 'B' && second == 'R'))
    return 0.0;
  switch (first) {
  case 'H': return 0.31;
  case 'C': return 0.76;
  case 'N': return 0.71;
  case 'O': return 0.66;
  case 'S': return 1.05;
  case 'P': return 1.07;
  default: return 0.0;
  }
}

BondScan find_covalent_bonds(const MOL &mol1, const std::vector<mol_info> &info1,
                             const MOL &mol2, const std::vector<mol_info> &info2,
                             TitanExecutor &executor, std::size_t max_examples) {
  BondScan scan;
  if (!mol1.coordinates_consistent() || !mol2.coordinates_consistent() ||
      mol1.size() == 0 || mol2.size() == 0)
    return scan;
  const std::vector<double> r1 = radii_of(mol1, info1);
  const std::vector<double> r2 = radii_of(mol2, info2);

  // Index mol2, probe with mol1 (27 neighbouring cells): O(N + M).
  std::unordered_multimap<CellKey, std::size_t, CellKeyHash> cells;
  cells.reserve(mol2.size());
  for (std::size_t j = 0; j < mol2.size(); ++j)
    if (r2[j] > 0.0 && finite_at(mol2, j))
      cells.emplace(cell_of(mol2, j), j);

  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  std::vector<std::size_t> counts(workers, 0);
  std::vector<std::vector<BondedPair>> found(workers);
  executor.parallel_for(
      mol1.size(), executor.capacity(), 256,
      [&](std::size_t worker, std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          if (r1[i] <= 0.0 || !finite_at(mol1, i))
            continue;
          const CellKey c = cell_of(mol1, i);
          for (std::int64_t dz = -1; dz <= 1; ++dz)
            for (std::int64_t dy = -1; dy <= 1; ++dy)
              for (std::int64_t dx = -1; dx <= 1; ++dx) {
                const auto [first, last] = cells.equal_range({c.x + dx, c.y + dy, c.z + dz});
                for (auto it = first; it != last; ++it) {
                  const std::size_t j = it->second;
                  const double ddx = mol1.pos.x[i] - mol2.pos.x[j];
                  const double ddy = mol1.pos.y[i] - mol2.pos.y[j];
                  const double ddz = mol1.pos.z[i] - mol2.pos.z[j];
                  const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                  const double limit = kBondScale * (r1[i] + r2[j]);
                  if (d2 > limit * limit)
                    continue;
                  ++counts[worker];
                  if (found[worker].size() < max_examples)
                    found[worker].push_back({i, j, std::sqrt(d2)});
                }
              }
        }
      });

  for (std::size_t w = 0; w < workers; ++w) {
    scan.bonds += counts[w];
    scan.examples.insert(scan.examples.end(), found[w].begin(), found[w].end());
  }
  std::sort(scan.examples.begin(), scan.examples.end(),
            [](const BondedPair &a, const BondedPair &b) {
              return std::tie(a.mol1_atom, a.mol2_atom) < std::tie(b.mol1_atom, b.mol2_atom);
            });
  if (scan.examples.size() > max_examples)
    scan.examples.resize(max_examples);
  return scan;
}

std::string describe_examples(const BondScan &scan, const std::vector<mol_info> &info1,
                              const std::vector<mol_info> &info2) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  for (std::size_t k = 0; k < scan.examples.size(); ++k) {
    const BondedPair &pair = scan.examples[k];
    if (k)
      out << "; ";
    out << "mol1 "
        << (pair.mol1_atom < info1.size() ? label(info1[pair.mol1_atom]) : std::to_string(pair.mol1_atom + 1))
        << " - mol2 "
        << (pair.mol2_atom < info2.size() ? label(info2[pair.mol2_atom]) : std::to_string(pair.mol2_atom + 1))
        << ' ' << pair.distance_angstrom << " A";
  }
  return out.str();
}

std::string bonded_groups_error(const MOL &mol1, const std::vector<mol_info> &info1,
                                const MOL &mol2, const std::vector<mol_info> &info2,
                                TitanExecutor &executor) {
  const BondScan scan = find_covalent_bonds(mol1, info1, mol2, info2, executor);
  if (scan.bonds == 0)
    return {};
  std::ostringstream out;
  out << kBondedGroupsError << ": " << scan.bonds
      << " covalent bond(s) join the two groups (" << describe_examples(scan, info1, info2)
      << (scan.bonds > scan.examples.size() ? "; ..." : "")
      << "). Inter-group energies on this route have no bonded exclusions or 1-4 "
         "scaling, so the bonded pairs would dominate E_elec/E_vdW. Choose groups that "
         "are separate molecules, or give the full topology with -common_topo (the "
         "topology energies and MM/GBSA then apply its exclusions and listed pairs; "
         "with -gbsa its atom names must match the structure).";
  return out.str();
}

} // namespace titan_group_bonds

#include "titan_energy_validation.h"

#include "titan_executor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

namespace titan_energy_validation {
namespace {

constexpr std::size_t kParallelPairScanLimit = 4'000'000;

struct CellKey {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};

  bool operator==(const CellKey &) const noexcept = default;
};

struct CellKeyHash {
  std::size_t operator()(const CellKey &key) const noexcept {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    return seed;
  }
};

CellKey cell_for(const MOL &mol, std::size_t atom,
                 double cell_width_angstrom) noexcept {
  return {static_cast<std::int64_t>(
              std::floor(mol.pos.x[atom] / cell_width_angstrom)),
          static_cast<std::int64_t>(
              std::floor(mol.pos.y[atom] / cell_width_angstrom)),
          static_cast<std::int64_t>(
              std::floor(mol.pos.z[atom] / cell_width_angstrom))};
}

bool finite_coordinate(const MOL &mol, std::size_t atom) noexcept {
  return std::isfinite(mol.pos.x[atom]) && std::isfinite(mol.pos.y[atom]) &&
         std::isfinite(mol.pos.z[atom]);
}

} // namespace

std::optional<InvalidCoordinate>
find_nonfinite_coordinate(const MOL &mol1, const MOL &mol2) {
  const auto scan = [](const MOL &mol,
                       int molecule) -> std::optional<InvalidCoordinate> {
    if (!mol.coordinates_consistent())
      return InvalidCoordinate{molecule, 0, 's',
                               std::numeric_limits<double>::quiet_NaN()};
    for (std::size_t atom = 0; atom < mol.size(); ++atom) {
      if (!std::isfinite(mol.pos.x[atom]))
        return InvalidCoordinate{molecule, atom, 'x', mol.pos.x[atom]};
      if (!std::isfinite(mol.pos.y[atom]))
        return InvalidCoordinate{molecule, atom, 'y', mol.pos.y[atom]};
      if (!std::isfinite(mol.pos.z[atom]))
        return InvalidCoordinate{molecule, atom, 'z', mol.pos.z[atom]};
    }
    return std::nullopt;
  };
  if (auto invalid = scan(mol1, 1))
    return invalid;
  return scan(mol2, 2);
}

std::optional<OverlappingAtomPair>
find_overlapping_atoms(const MOL &mol1, const MOL &mol2,
                       bool identical_selections, TitanExecutor &executor,
                       double minimum_pair_distance_nm) {
  if (!mol1.coordinates_consistent() || !mol2.coordinates_consistent() ||
      mol1.size() == 0 || mol2.size() == 0 ||
      !std::isfinite(minimum_pair_distance_nm) ||
      minimum_pair_distance_nm <= 0.0)
    return std::nullopt;
  const double minimum_pair_distance_angstrom =
      10.0 * minimum_pair_distance_nm;
  const double minimum_pair_distance_squared_angstrom =
      minimum_pair_distance_angstrom * minimum_pair_distance_angstrom;

  // For the common protein-small-ligand shape, constructing a spatial hash is
  // more expensive than the coordinate comparisons themselves. Reuse TITAN's
  // executor for a bounded direct scan; retain the linear spatial-hash path
  // for large selections and self-interaction workloads.
  if (mol1.size() <= kParallelPairScanLimit / mol2.size()) {
    std::vector<std::optional<OverlappingAtomPair>> worker_hits(
        std::max<std::size_t>(1, executor.capacity()));
    executor.parallel_for(
        mol1.size(), executor.capacity(), 64,
        [&](std::size_t worker, std::size_t begin_atom,
            std::size_t end_atom) {
          std::optional<OverlappingAtomPair> local_hit;
          for (std::size_t mol1_atom = begin_atom;
               mol1_atom < end_atom && !local_hit; ++mol1_atom) {
            if (!finite_coordinate(mol1, mol1_atom))
              continue;
            for (std::size_t mol2_atom = 0; mol2_atom < mol2.size();
                 ++mol2_atom) {
              if ((identical_selections && mol1_atom == mol2_atom) ||
                  !finite_coordinate(mol2, mol2_atom))
                continue;
              const double delta_x =
                  mol1.pos.x[mol1_atom] - mol2.pos.x[mol2_atom];
              const double delta_y =
                  mol1.pos.y[mol1_atom] - mol2.pos.y[mol2_atom];
              const double delta_z =
                  mol1.pos.z[mol1_atom] - mol2.pos.z[mol2_atom];
              const double distance_squared = delta_x * delta_x +
                                              delta_y * delta_y +
                                              delta_z * delta_z;
              if (distance_squared <=
                  minimum_pair_distance_squared_angstrom) {
                local_hit = OverlappingAtomPair{
                    mol1_atom, mol2_atom,
                    0.1 * std::sqrt(distance_squared)};
                break;
              }
            }
          }
          worker_hits[worker] = local_hit;
        });

    std::optional<OverlappingAtomPair> best_hit;
    for (const auto &hit : worker_hits) {
      if (hit && (!best_hit || hit->mol1_atom < best_hit->mol1_atom ||
                  (hit->mol1_atom == best_hit->mol1_atom &&
                   hit->mol2_atom < best_hit->mol2_atom)))
        best_hit = hit;
    }
    return best_hit;
  }

  // Index the larger selection so the smaller selection performs the 27-cell
  // neighborhood probes. This keeps validation O(N + M) and cheap relative
  // to the O(N*M) nonbonded kernels it protects.
  const bool index_mol1 = mol1.size() >= mol2.size();
  const MOL &indexed = index_mol1 ? mol1 : mol2;
  const MOL &queried = index_mol1 ? mol2 : mol1;

  std::unordered_multimap<CellKey, std::size_t, CellKeyHash> cells;
  cells.max_load_factor(0.7F);
  cells.reserve(indexed.size());
  for (std::size_t atom = 0; atom < indexed.size(); ++atom) {
    if (finite_coordinate(indexed, atom))
      cells.emplace(cell_for(indexed, atom, minimum_pair_distance_angstrom),
                    atom);
  }

  // The index is immutable from here, so neighborhood probes can share it
  // safely. Each worker records its lowest query atom; the final reduction
  // preserves deterministic error reporting independent of thread count.
  std::vector<std::optional<OverlappingAtomPair>> worker_hits(
      std::max<std::size_t>(1, executor.capacity()));
  executor.parallel_for(
      queried.size(), executor.capacity(), 256,
      [&](std::size_t worker, std::size_t begin_query,
          std::size_t end_query) {
        std::optional<OverlappingAtomPair> local_hit;
        for (std::size_t query_atom = begin_query; query_atom < end_query;
             ++query_atom) {
          if (!finite_coordinate(queried, query_atom))
            continue;
          const CellKey query_cell =
              cell_for(queried, query_atom, minimum_pair_distance_angstrom);
          for (std::int64_t dz = -1; dz <= 1 && !local_hit; ++dz) {
            for (std::int64_t dy = -1; dy <= 1 && !local_hit; ++dy) {
              for (std::int64_t dx = -1; dx <= 1 && !local_hit; ++dx) {
                const CellKey neighbor{query_cell.x + dx, query_cell.y + dy,
                                       query_cell.z + dz};
                const auto [cell_begin, cell_end] =
                    cells.equal_range(neighbor);
                for (auto it = cell_begin; it != cell_end; ++it) {
                  const std::size_t indexed_atom = it->second;
                  const std::size_t mol1_atom =
                      index_mol1 ? indexed_atom : query_atom;
                  const std::size_t mol2_atom =
                      index_mol1 ? query_atom : indexed_atom;
                  if (identical_selections && mol1_atom == mol2_atom)
                    continue;

                  const double delta_x =
                      mol1.pos.x[mol1_atom] - mol2.pos.x[mol2_atom];
                  const double delta_y =
                      mol1.pos.y[mol1_atom] - mol2.pos.y[mol2_atom];
                  const double delta_z =
                      mol1.pos.z[mol1_atom] - mol2.pos.z[mol2_atom];
                  const double distance_squared = delta_x * delta_x +
                                                  delta_y * delta_y +
                                                  delta_z * delta_z;
                  if (distance_squared <=
                      minimum_pair_distance_squared_angstrom) {
                    local_hit = OverlappingAtomPair{
                        mol1_atom, mol2_atom,
                        0.1 * std::sqrt(distance_squared)};
                    break;
                  }
                }
              }
            }
          }
          if (local_hit)
            break;
        }
        worker_hits[worker] = local_hit;
      });

  std::optional<OverlappingAtomPair> best_hit;
  std::size_t best_query_atom = std::numeric_limits<std::size_t>::max();
  for (const auto &hit : worker_hits) {
    if (!hit)
      continue;
    const std::size_t query_atom =
        index_mol1 ? hit->mol2_atom : hit->mol1_atom;
    if (query_atom < best_query_atom) {
      best_query_atom = query_atom;
      best_hit = hit;
    }
  }
  return best_hit;
}

} // namespace titan_energy_validation

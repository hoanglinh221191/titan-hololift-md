#ifndef MOL_TOPOLOGY_H
#define MOL_TOPOLOGY_H

#include "MOL_info.h"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class Multi_MOL;

struct MolTopologyBond {
  std::uint32_t a = 0;
  std::uint32_t b = 0;
};

struct MolTopologyMoleculeRange {
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  std::string name;
  std::uint32_t copy_index = 0;
};

enum class MolTopologyVdwRadiusSource : std::uint8_t {
  None = 0,
  GromacsTopology,
  GromacsTopologyExplicitZeroLJ,
  MultiMolParameters,
  MultiMolParametersExplicitZeroLJ,
};

class MolTopology {
public:
  [[nodiscard]] bool available() const noexcept { return atom_count_ > 0; }
  [[nodiscard]] bool has_bonds() const noexcept { return !bonds_.empty(); }
  [[nodiscard]] std::uint32_t atom_count() const noexcept {
    return atom_count_;
  }
  [[nodiscard]] const std::vector<MolTopologyBond> &bonds() const noexcept {
    return bonds_;
  }
  [[nodiscard]] const std::vector<MolTopologyMoleculeRange> &
  molecule_ranges() const noexcept {
    return molecule_ranges_;
  }
  [[nodiscard]] const std::vector<std::uint32_t> &
  source_atom_ids() const noexcept {
    return source_atom_ids_;
  }
  [[nodiscard]] std::optional<double>
  vdw_radius(std::uint32_t atom) const noexcept;
  // Only source-identity-resolved topology masses; no parameter sentinel fallback.
  [[nodiscard]] std::optional<double>
  atomic_mass(std::uint32_t atom) const noexcept;
  void set_atomic_masses(std::vector<double> masses);
  [[nodiscard]] MolTopologyVdwRadiusSource
  vdw_radius_source(std::uint32_t atom) const noexcept;
  [[nodiscard]] std::size_t vdw_radius_count() const noexcept {
    return vdw_radius_count_;
  }

  [[nodiscard]] std::span<const std::uint32_t>
  neighbors(std::uint32_t atom) const noexcept;

  void clear() noexcept;
  void set_bonds(std::uint32_t atom_count, std::vector<MolTopologyBond> bonds,
                 std::vector<MolTopologyMoleculeRange> molecule_ranges = {},
                 std::vector<std::uint32_t> source_atom_ids = {},
                 std::vector<double> vdw_radii = {});
  void cache_missing_vdw_radii(
      std::span<const double> radii,
      std::span<const double> sigma = {},
      std::span<const double> epsilon = {},
      std::span<const std::int32_t> type_ids = {});
  void build_adjacency();

private:
  std::uint32_t atom_count_ = 0;
  std::vector<MolTopologyBond> bonds_;
  std::vector<MolTopologyMoleculeRange> molecule_ranges_;
  std::vector<std::uint32_t> row_offsets_;
  std::vector<std::uint32_t> col_indices_;
  std::vector<std::uint32_t> source_atom_ids_;
  std::vector<double> vdw_radii_;
  std::vector<double> atomic_masses_;
  std::vector<MolTopologyVdwRadiusSource> vdw_radius_sources_;
  std::size_t vdw_radius_count_ = 0;
};

std::expected<MolTopology, std::string>
build_gromacs_mol_topology(std::string_view topology_path,
                           std::span<const mol_info> selection_info);

inline std::expected<MolTopology, std::string>
build_gromacs_mol_topology(std::string_view topology_path,
                           const std::vector<mol_info> &selection_info) {
  return build_gromacs_mol_topology(
      topology_path,
      std::span<const mol_info>(selection_info.data(), selection_info.size()));
}

bool attach_gromacs_mol_topology_graph(Multi_MOL &mol,
                                       std::string_view topology_path);

#endif // MOL_TOPOLOGY_H

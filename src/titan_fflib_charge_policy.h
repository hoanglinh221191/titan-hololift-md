#ifndef TITAN_FFLIB_CHARGE_POLICY_H
#define TITAN_FFLIB_CHARGE_POLICY_H

namespace TitanFfLib {

struct ChargeSourcePolicy {
  bool molecule1_uses_topology{false};
  bool molecule2_uses_topology{false};
};

[[nodiscard]] constexpr ChargeSourcePolicy
resolve_charge_source_policy(bool common_topology, bool ligand_topology_available,
                             bool receptor_topology_available) noexcept {
  if (common_topology)
    return {true, true};
  // Legacy CLI: -topo/topo_name belongs to mol2 (ligand), whereas
  // -topo2/topo_name_2 belongs to mol1 (receptor). Keep the loader semantics.
  return {receptor_topology_available, ligand_topology_available};
}

} // namespace TitanFfLib

#endif

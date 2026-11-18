#include "gbsa/gb_topology_pairs.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

namespace titan_gbsa {
namespace {

constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();

std::vector<std::size_t> local_index(const titan_nb::Topology &topology,
                                     const std::vector<mol_info> &info) {
  std::vector<std::size_t> local(topology.atoms.size(), kAbsent);
  for (std::size_t i = 0; i < info.size(); ++i) {
    if (!info[i].has_original_atom_id())
      continue;
    const std::size_t global = info[i].get_original_atom_id();
    if (global < local.size())
      local[global] = i;
  }
  return local;
}

} // namespace

TopologyPairs build_topology_pairs(const titan_nb::Topology &topology,
                                   const std::vector<mol_info> &rec_info,
                                   const std::vector<mol_info> &lig_info) {
  TopologyPairs out;
  out.rec_size = rec_info.size();
  out.lig_size = lig_info.size();
  const std::vector<std::size_t> rec_local = local_index(topology, rec_info);
  const std::vector<std::size_t> lig_local = local_index(topology, lig_info);

  // (rec, lig) -> pair; a key can be both excluded and listed (1-4 pairs
  // with nrexcl >= 3).
  std::map<std::pair<std::size_t, std::size_t>, TopologyPair> found;
  const auto across = [&](std::uint64_t key) -> std::pair<std::size_t, std::size_t> {
    const std::size_t a = static_cast<std::size_t>(key >> 32);
    const std::size_t b = static_cast<std::size_t>(key & 0xffffffffULL);
    if (a >= rec_local.size() || b >= rec_local.size())
      return {kAbsent, kAbsent};
    if (rec_local[a] != kAbsent && lig_local[b] != kAbsent)
      return {rec_local[a], lig_local[b]};
    if (rec_local[b] != kAbsent && lig_local[a] != kAbsent)
      return {rec_local[b], lig_local[a]};
    return {kAbsent, kAbsent};
  };
  for (const std::uint64_t key : topology.exclusions) {
    const auto [r, l] = across(key);
    if (r == kAbsent)
      continue;
    TopologyPair &pair = found[{r, l}];
    pair.rec = r;
    pair.lig = l;
    pair.excluded = true;
  }
  for (const auto &[key, listed] : topology.listed_pairs) {
    const auto [r, l] = across(key);
    if (r == kAbsent)
      continue;
    TopologyPair &pair = found[{r, l}];
    pair.rec = r;
    pair.lig = l;
    pair.listed_c6 = listed.lj.c6;
    pair.listed_c12 = listed.lj.c12;
    pair.listed_qq = listed.charge_product;
  }
  out.pairs.reserve(found.size());
  for (const auto &[ids, pair] : found) {
    out.pairs.push_back(pair);
    out.excluded += pair.excluded ? 1 : 0;
    out.listed += (pair.listed_c6 != 0.0 || pair.listed_c12 != 0.0 ||
                   pair.listed_qq != 0.0)
                      ? 1
                      : 0;
  }
  return out;
}

} // namespace titan_gbsa

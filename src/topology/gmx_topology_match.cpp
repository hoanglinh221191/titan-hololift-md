#include "topology/gmx_topology_match.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace TitanTopo {
namespace {

std::string trimmed_upper(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos)
    return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  std::string out(text.substr(first, last - first + 1));
  for (char &ch : out)
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return out;
}

// Groups of residue names that are one residue in different protonation or
// naming conventions (AMBER, CHARMM, GROMOS, OPLS-AA).
int residue_family(const std::string &name) {
  static const std::unordered_map<std::string, int> families{
      {"HIS", 1},  {"HID", 1},  {"HIE", 1},  {"HIP", 1},  {"HSD", 1},
      {"HSE", 1},  {"HSP", 1},  {"HISA", 1}, {"HISB", 1}, {"HISH", 1},
      {"HISD", 1}, {"HISE", 1}, {"ASP", 2},  {"ASH", 2},  {"ASPP", 2},
      {"ASPH", 2}, {"GLU", 3},  {"GLH", 3},  {"GLUP", 3}, {"GLUH", 3},
      {"LYS", 4},  {"LYN", 4},  {"LSN", 4},  {"LYSH", 4}, {"CYS", 5},
      {"CYX", 5},  {"CYM", 5},  {"CYS2", 5}, {"CYSH", 5}};
  const auto it = families.find(name);
  return it == families.end() ? 0 : it->second;
}

bool terminal_prefix_of(const std::string &pdb, const std::string &topology) {
  return pdb.size() == topology.size() + 1 && (pdb[0] == 'N' || pdb[0] == 'C') &&
         pdb.compare(1, std::string::npos, topology) == 0;
}

} // namespace

std::string atom_name_key(std::string_view name) {
  std::string key = trimmed_upper(name);
  std::replace(key.begin(), key.end(), '*', '\'');
  if (key.size() > 1 && std::isdigit(static_cast<unsigned char>(key[0])))
    key = key.substr(1) + key.substr(0, 1);
  return key;
}

bool residue_names_compatible(std::string_view pdb_name, std::string_view topology_name) {
  std::string pdb = trimmed_upper(pdb_name);
  std::string top = trimmed_upper(topology_name);
  if (pdb == top)
    return true;
  if (terminal_prefix_of(pdb, top) || terminal_prefix_of(top, pdb))
    return true;
  // strip one terminal prefix on either side before the family test
  for (std::string *name : {&pdb, &top}) {
    if (name->size() == 4 && ((*name)[0] == 'N' || (*name)[0] == 'C') &&
        residue_family(*name) == 0 && residue_family(name->substr(1)) != 0)
      *name = name->substr(1);
  }
  const int family = residue_family(pdb);
  return family != 0 && family == residue_family(top);
}

NameCheck compare_positional_names(const std::vector<mol_info> &structure,
                                   const std::vector<topo_info_temp> &topology) {
  NameCheck out;
  out.atoms = std::min(structure.size(), topology.size());
  for (std::size_t i = 0; i < out.atoms; ++i) {
    const std::string pdb_atom = structure[i].get_atom();
    const bool atom_ok = atom_name_key(pdb_atom) == atom_name_key(topology[i].atom_name);
    const bool residue_ok =
        residue_names_compatible(structure[i].get_res_name(), topology[i].res_name);
    if (!residue_ok)
      ++out.residue_mismatches;
    if (atom_ok)
      continue;
    ++out.atom_mismatches;
    if (out.examples.size() < 10)
      out.examples.push_back("#" + std::to_string(i + 1) + " " +
                             structure[i].get_res_name() + std::to_string(structure[i].get_resnr()) +
                             "/" + pdb_atom + " vs " + topology[i].res_name +
                             std::to_string(topology[i].res_nr) + "/" + topology[i].atom_name);
  }
  return out;
}

std::vector<long long> match_by_names(const std::vector<mol_info> &structure,
                                      const std::vector<topo_info_temp> &topology) {
  // Same rule as the former linear search: the first topology atom, in file
  // order, not yet used, whose residue name equals the PDB one or is the PDB
  // name without an N/C terminal prefix, and whose atom name is identical.
  struct Bucket {
    std::vector<std::size_t> indices;
    std::size_t next{};
  };
  std::unordered_map<std::string, Bucket> buckets;
  buckets.reserve(topology.size());
  const auto key = [](const std::string &residue, const std::string &atom) {
    return residue + '\x1f' + atom;
  };
  for (std::size_t j = 0; j < topology.size(); ++j)
    buckets[key(topology[j].res_name, topology[j].atom_name)].indices.push_back(j);

  std::vector<long long> match(structure.size(), -1);
  for (std::size_t i = 0; i < structure.size(); ++i) {
    const std::string residue = structure[i].get_res_name();
    const std::string atom = structure[i].get_atom();
    std::array<Bucket *, 2> candidates{nullptr, nullptr};
    if (auto it = buckets.find(key(residue, atom)); it != buckets.end())
      candidates[0] = &it->second;
    if (residue.size() >= 2 && (residue[0] == 'N' || residue[0] == 'C')) {
      if (auto it = buckets.find(key(residue.substr(1), atom)); it != buckets.end())
        candidates[1] = &it->second;
    }
    Bucket *best = nullptr;
    for (Bucket *bucket : candidates) {
      if (bucket == nullptr || bucket->next >= bucket->indices.size())
        continue;
      if (best == nullptr || bucket->indices[bucket->next] < best->indices[best->next])
        best = bucket;
    }
    if (best != nullptr)
      match[i] = static_cast<long long>(best->indices[best->next++]);
  }
  return match;
}

} // namespace TitanTopo

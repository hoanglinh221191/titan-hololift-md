#include "topology/martini_topology.h"

#include "titan_ui.h"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace titan_martini {

void MartiniTopology::map_to_titan(std::vector<mol_info> &info,
                                   Multi_MOL_parameters &parameters,
                                   std::fstream &log_file,
                                   cache_sigma_epsilon *martini_cache) {
  if (info.size() != flat_system.size()) {
    TitanUI::print_info("\033[1;33m[WARNING]\033[0m MARTINI Parser: Topology "
                        "flattened atoms (" +
                        std::to_string(flat_system.size()) +
                        ") differs from PDB atoms (" +
                        std::to_string(info.size()) +
                        "). Engaging robust Name-Match fallback mapping.");
  }

  int successfully_mapped = 0;
  int charge_mapped = 0;
  double topology_charge_expected = 0.0;

  struct FallbackEntry {
    std::vector<std::pair<std::string, double>> candidates;
    std::size_t next_index{0};
    bool unique{true};

    void add(const AtomDef &atom) {
      const std::pair<std::string, double> candidate{atom.type, atom.charge};
      if (!candidates.empty() &&
          (candidates.front().first != candidate.first ||
           candidates.front().second != candidate.second)) {
        unique = false;
      }
      candidates.emplace_back(candidate);
    }

    bool resolve(std::string &atype, double &acharge) {
      if (candidates.empty())
        return false;
      if (unique) {
        atype = candidates.front().first;
        acharge = candidates.front().second;
        return true;
      }
      if (next_index >= candidates.size())
        return false;
      atype = candidates[next_index].first;
      acharge = candidates[next_index].second;
      ++next_index;
      return true;
    }

    bool ambiguous() const { return !unique; }
  };

  const auto relaxed_key = [](const std::string &resname,
                              const std::string &atomname) {
    return resname + ":" + atomname;
  };
  const auto exact_key = [&relaxed_key](const std::string &resname,
                                        size_t resnr,
                                        const std::string &atomname) {
    return relaxed_key(resname, std::to_string(resnr) + ":" + atomname);
  };

  // Keep the full candidate list per fallback key so subset mapping can
  // consume entries in topology order instead of silently keeping only the
  // last duplicate.
  std::unordered_map<std::string, FallbackEntry> fallback_exact_dict;
  std::unordered_map<std::string, FallbackEntry> fallback_relaxed_dict;
  for (const auto &atom : flat_system) {
    fallback_exact_dict[exact_key(atom.resname, static_cast<size_t>(atom.resnr),
                                  atom.atomname)]
        .add(atom);
    fallback_relaxed_dict[relaxed_key(atom.resname, atom.atomname)].add(atom);
  }

  const auto count_ambiguous_keys = [](const auto &dict) {
    size_t ambiguous = 0;
    for (const auto &[_, entry] : dict)
      ambiguous += static_cast<size_t>(entry.ambiguous());
    return ambiguous;
  };

  size_t fallback_exact_hits = 0;
  size_t fallback_relaxed_hits = 0;
  size_t fallback_ambiguous_hits = 0;
  size_t fallback_unmapped = 0;

  for (size_t i = 0; i < info.size(); ++i) {
    std::string atype = "";
    double acharge = 0.0;

    if (i < flat_system.size() && info.size() == flat_system.size()) {
      atype = flat_system[i].type;
      acharge = flat_system[i].charge;
    } else {
      const auto resolve_fallback = [&](auto &dict, const std::string &key,
                                        size_t &hit_counter) {
        auto it = dict.find(key);
        if (it == dict.end() || !it->second.resolve(atype, acharge))
          return false;
        ++hit_counter;
        if (it->second.ambiguous())
          ++fallback_ambiguous_hits;
        return true;
      };

      const std::string exact_lookup =
          exact_key(info[i].get_res_name(), info[i].get_resnr(),
                    info[i].get_atom());
      const std::string relaxed_lookup =
          relaxed_key(info[i].get_res_name(), info[i].get_atom());

      if (!resolve_fallback(fallback_exact_dict, exact_lookup,
                            fallback_exact_hits) &&
          !resolve_fallback(fallback_relaxed_dict, relaxed_lookup,
                            fallback_relaxed_hits)) {
        ++fallback_unmapped;
        continue; // Unmapped
      }
    }

    parameters[i].set_charge(acharge);
    topology_charge_expected += acharge;
    charge_mapped++;
    info[i].set_atom_type(atype);
    parameters[i].set_type_id(-1);

    double sigma = -1.0;
    double epsilon = -1.0;
    auto type_it = type_to_id.find(atype);
    if (type_it != type_to_id.end()) {
      const int tid = type_it->second;
      parameters[i].set_type_id(tid);
      const std::size_t diag =
          static_cast<std::size_t>(tid) * type_to_id.size() +
          static_cast<std::size_t>(tid);
      if (diag < pair_sigma_table.size()) {
        sigma = pair_sigma_table[diag];
        epsilon = pair_epsilon_table[diag];
      }
    }

    if ((sigma < 0.0 || epsilon < 0.0) && atomtype_params.contains(atype)) {
      sigma = atomtype_params.at(atype).sigma;
      epsilon = atomtype_params.at(atype).epsilon;
    }

    if (sigma >= 0.0 && epsilon >= 0.0) {
      // ENE-13 (21/10/2026): the radius slot is r_min/2 in Angstrom for every
      // force field (take_para: 5 * 2^(1/6) * sigma[nm]); the former
      // sigma * 2^(1/6) / 2 left it in nm, 10x too small, which disabled the
      // pbctopo steric test for CG beads and wrote 0.26 A radii to -pqr.
      const double radius = 5.0 * 1.122462048309372981 * sigma;
      parameters[i].set_epsilon_sigma_radius(epsilon, sigma, radius);
      successfully_mapped++;
    }

    if (info[i].get_atom().find("BB") != std::string::npos ||
        info[i].get_atom().find("PO4") != std::string::npos) {
      info[i].set_side_chain(false);
    } else {
      info[i].set_side_chain(true);
    }
  }

  TitanUI::log_kv(log_file, "VdW Types Loaded",
                  std::to_string(atomtype_params.size()));
  TitanUI::log_kv(log_file, "Martini Pair Types",
                  std::to_string(type_to_id.size()));
  TitanUI::log_kv(log_file, "Martini Explicit Pair Rules",
                  std::to_string(explicit_pair_rules));
  TitanUI::log_kv(log_file, "Martini Pair Fallbacks",
                  std::to_string(fallback_pair_rules));
  TitanUI::log_kv(log_file, "Templates Loaded",
                  std::to_string(molecule_types.size()));
  TitanUI::log_kv(log_file, "Flat Atoms Expanded",
                  std::to_string(flat_system.size()));
  TitanUI::log_kv(log_file, "Beads Mapped",
                  std::to_string(successfully_mapped) + " / " +
                      std::to_string(info.size()));
  {
    double mapped_charge = 0.0;
    for (size_t i = 0; i < parameters.size(); ++i)
      mapped_charge += parameters[i].get_charge();
    const double charge_delta = mapped_charge - topology_charge_expected;
    const bool charge_match = std::fabs(charge_delta) < 1e-8 &&
                              charge_mapped == static_cast<int>(info.size());
    std::ostringstream charge_msg;
    charge_msg << "expected=" << topology_charge_expected
               << " e, mapped=" << mapped_charge
               << " e, delta=" << charge_delta
               << " e, assigned=" << charge_mapped << "/" << info.size()
               << " [" << (charge_match ? "OK" : "MISMATCH") << "]";
    TitanUI::log_kv(log_file, "Topology Charge Cross-check",
                    charge_msg.str());
  }
  TitanUI::log_kv(log_file, "Martini Fallback Keys",
                  "exact=" + std::to_string(fallback_exact_dict.size()) +
                      " (" +
                      std::to_string(
                          count_ambiguous_keys(fallback_exact_dict)) +
                      " ambiguous), relaxed=" +
                      std::to_string(fallback_relaxed_dict.size()) + " (" +
                      std::to_string(
                          count_ambiguous_keys(fallback_relaxed_dict)) +
                      " ambiguous)");
  TitanUI::log_kv(log_file, "Martini Fallback Hits",
                  "exact=" + std::to_string(fallback_exact_hits) +
                      ", relaxed=" +
                      std::to_string(fallback_relaxed_hits) +
                      ", ambiguous=" +
                      std::to_string(fallback_ambiguous_hits) +
                      ", unmapped=" + std::to_string(fallback_unmapped));

  if (martini_cache != nullptr) {
    martini_cache->set_martini_pair_table(pair_sigma_table, pair_epsilon_table,
                                          static_cast<int>(type_to_id.size()));
  }

  assign_chain_numbers(info, log_file);

  TitanUI::print_info("MARTINI Parser: Extracted " +
                      std::to_string(explicit_pair_rules) +
                      " explicit pair rules, mapped " +
                      std::to_string(successfully_mapped) + " atoms.");
}

} // namespace titan_martini

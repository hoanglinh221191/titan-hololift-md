#include "Multi_MOL.h"
#include "topology_include_utils.h"
#include "titan_ui.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace {

bool resolve_topology_include_path(std::string_view current_file,
                                   std::string_view include_target,
                                   std::string &resolved_path) {
  namespace fs = std::filesystem;
  for (const auto &candidate :
       TitanTopo::build_include_candidates(current_file, include_target)) {
    std::error_code ec;
    if (fs::exists(candidate, ec)) {
      resolved_path = candidate.string();
      return true;
    }
  }
  return false;
}

} // namespace

struct LJParam {
  double sigma;
  double epsilon;
};
struct PairRule {
  std::string type_i;
  std::string type_j;
  LJParam lj;
};
struct AtomDef {
  std::string type;
  int resnr;
  std::string resname;
  std::string atomname;
  double charge;
};
struct MolTemplate {
  std::vector<AtomDef> atoms;
};

class MartiniTopology {
public:
  std::vector<std::pair<std::string, int>> system_molecules;

private:
  std::unordered_map<std::string, LJParam> atomtype_params;
  std::vector<PairRule> nonbond_rules;
  std::unordered_map<std::string, MolTemplate> molecule_types;
  std::vector<AtomDef> flat_system;
  std::unordered_map<std::string, int> type_to_id;
  std::vector<double> pair_sigma_table;
  std::vector<double> pair_epsilon_table;
  std::size_t explicit_pair_rules{0};
  std::size_t fallback_pair_rules{0};

  void finalize_pair_table() {
    type_to_id.clear();
    auto register_type = [&](const std::string &name) {
      if (!name.empty() && !type_to_id.contains(name))
        type_to_id.emplace(name, static_cast<int>(type_to_id.size()));
    };

    for (const auto &[name, _] : atomtype_params)
      register_type(name);
    for (const auto &rule : nonbond_rules) {
      register_type(rule.type_i);
      register_type(rule.type_j);
    }

    const std::size_t ntypes = type_to_id.size();
    pair_sigma_table.assign(ntypes * ntypes, -1.0);
    pair_epsilon_table.assign(ntypes * ntypes, -1.0);

    auto set_pair = [&](int ti, int tj, const LJParam &lj) {
      const std::size_t ij = static_cast<std::size_t>(ti) * ntypes +
                             static_cast<std::size_t>(tj);
      const std::size_t ji = static_cast<std::size_t>(tj) * ntypes +
                             static_cast<std::size_t>(ti);
      pair_sigma_table[ij] = lj.sigma;
      pair_sigma_table[ji] = lj.sigma;
      pair_epsilon_table[ij] = lj.epsilon;
      pair_epsilon_table[ji] = lj.epsilon;
    };

    for (const auto &[name, lj] : atomtype_params) {
      const int tid = type_to_id.at(name);
      set_pair(tid, tid, lj);
    }

    explicit_pair_rules = 0;
    for (const auto &rule : nonbond_rules) {
      const auto it_i = type_to_id.find(rule.type_i);
      const auto it_j = type_to_id.find(rule.type_j);
      if (it_i == type_to_id.end() || it_j == type_to_id.end())
        continue;
      set_pair(it_i->second, it_j->second, rule.lj);
      ++explicit_pair_rules;
    }

    fallback_pair_rules = 0;
    for (std::size_t i = 0; i < ntypes; ++i) {
      const double sigma_i = pair_sigma_table[i * ntypes + i];
      const double eps_i = pair_epsilon_table[i * ntypes + i];
      for (std::size_t j = i + 1; j < ntypes; ++j) {
        const std::size_t ij = i * ntypes + j;
        if (pair_sigma_table[ij] >= 0.0 && pair_epsilon_table[ij] >= 0.0)
          continue;
        const double sigma_j = pair_sigma_table[j * ntypes + j];
        const double eps_j = pair_epsilon_table[j * ntypes + j];
        if (sigma_i >= 0.0 && sigma_j >= 0.0 && eps_i >= 0.0 && eps_j >= 0.0) {
          const double sigma_ij = 0.5 * (sigma_i + sigma_j);
          const double epsilon_ij = std::sqrt(eps_i * eps_j);
          pair_sigma_table[ij] = pair_sigma_table[j * ntypes + i] = sigma_ij;
          pair_epsilon_table[ij] =
              pair_epsilon_table[j * ntypes + i] = epsilon_ij;
          ++fallback_pair_rules;
        }
      }
    }
  }

  std::string get_directive(const std::string &line) {
    size_t p1 = line.find('[');
    size_t p2 = line.find(']');
    if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1) {
      std::string dir = line.substr(p1 + 1, p2 - p1 - 1);
      size_t s = dir.find_first_not_of(" \t");
      size_t e = dir.find_last_not_of(" \t");
      if (s != std::string::npos)
        return dir.substr(s, e - s + 1);
    }
    return "";
  }

public:
  int files_read = 0;

  void parse_files(const std::string &top_file, std::fstream &log_file) {
    std::vector<std::string> lines;
    std::vector<std::string> include_stack;

    std::function<void(const std::string &)> read_file =
        [&](const std::string &filename) {
          const std::string normalized =
              std::filesystem::path(filename).lexically_normal().string();
          if (std::find(include_stack.begin(), include_stack.end(), normalized) !=
              include_stack.end()) {
            TitanUI::log_message(
                log_file,
                "[WARNING] MARTINI Parser: Skipping cyclic include: " +
                    normalized);
            return;
          }

          include_stack.emplace_back(normalized);
          std::fstream f(normalized, std::ios::in);
          if (!f.is_open()) {
            TitanUI::print_info(
                "\033[1;33m[WARNING]\033[0m MARTINI Parser: Cannot open "
                "included file: " +
                filename + " — check if file exists.");
            TitanUI::log_message(
                log_file,
                "[WARNING] Cannot open included file: " + normalized);
            include_stack.pop_back();
            return;
          }
          files_read++;
          std::string line;
          while (std::getline(f, line)) {
            size_t comment_pos = line.find(';');
            if (comment_pos != std::string::npos)
              line = line.substr(0, comment_pos); // Xóa comment

            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos)
              continue; // Bỏ qua dòng rỗng
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            // Phân giải #include
            if (const auto include_target =
                    TitanTopo::extract_include_target(line)) {
              std::string resolved_include;
              if (resolve_topology_include_path(normalized, *include_target,
                                                resolved_include)) {
                read_file(resolved_include);
              } else {
                TitanUI::print_info(
                    "\033[1;33m[WARNING]\033[0m MARTINI Parser: Cannot open "
                    "included file: " +
                    *include_target + " â€” check if file exists.");
                TitanUI::log_message(
                    log_file,
                    "[WARNING] Cannot open included file: " + *include_target +
                        " (from " + normalized + ")");
              }
              continue;
            }
            lines.emplace_back(line);
          }
          include_stack.pop_back();
        };

    read_file(top_file);

    std::string current_dir = "";
    std::string current_mol = "";

    for (const auto &line : lines) {
      std::string dir = get_directive(line);
      if (!dir.empty()) {
        current_dir = dir;
        if (dir == "moleculetype")
          current_mol = "";
        continue;
      }

      std::istringstream iss(line);
      std::vector<std::string> tokens;
      std::string token;
      while (iss >> token)
        tokens.emplace_back(token);
      if (tokens.empty())
        continue;

      if (current_dir == "nonbond_params") {
        if (tokens.size() >= 5) {
          try {
            double sigma = std::stod(tokens[3]);
            double epsilon = std::stod(tokens[4]);
            nonbond_rules.push_back({tokens[0], tokens[1], {sigma, epsilon}});
          } catch (...) {
          }
        }
      } else if (current_dir == "atomtypes") {
        if (tokens.size() >= 6) {
          try {
            double sigma = std::stod(tokens[tokens.size() - 2]);
            double epsilon = std::stod(tokens[tokens.size() - 1]);
            atomtype_params[tokens[0]] = {sigma, epsilon};
          } catch (...) {
          }
        }
      } else if (current_dir == "moleculetype") {
        if (current_mol.empty())
          current_mol = tokens[0];
      } else if (current_dir == "atoms") {
        if (!current_mol.empty() && tokens.size() >= 7) {
          try {
            std::string type = tokens[1];
            int resnr = std::stoi(tokens[2]);
            std::string resname = tokens[3];
            std::string atomname = tokens[4];
            double charge = std::stod(tokens[6]);
            molecule_types[current_mol].atoms.emplace_back(type, resnr, resname,
                                                           atomname, charge);
          } catch (...) {
          }
        }
      } else if (current_dir == "molecules") {
        if (tokens.size() >= 2) {
          try {
            system_molecules.emplace_back(tokens[0], std::stoi(tokens[1]));
          } catch (...) {
          }
        }
      }
    }

    finalize_pair_table();

    for (const auto &comp : system_molecules) {
      if (molecule_types.find(comp.first) == molecule_types.end()) {
        TitanUI::log_message(
            log_file,
            "[ERROR FATAL] Molecule name '" + comp.first +
                "' in [ molecules ] DOES NOT MATCH any [ moleculetype ]!");
      }
    }
  }

  void build_flat_system() {
    for (const auto &comp : system_molecules) {
      std::string m_name = comp.first;
      int count = comp.second;
      if (molecule_types.count(m_name)) {
        for (int c = 0; c < count; ++c) {
          for (const auto &atom : molecule_types[m_name].atoms) {
            flat_system.emplace_back(atom);
          }
        }
      }
    }
  }

  void map_to_titan(std::vector<mol_info> &info,
                    Multi_MOL_parameters &parameters, std::fstream &log_file,
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
        double radius = sigma * 1.122462048309372981 / 2.0; // r_min/2 approx
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

    // ---- Assign chainnum per bead ----------------------------------------
    // For a full-system MARTINI PDB, the flattened topology order matches the
    // coordinate file and we can stamp chain ids directly from [ molecules ].
    //
    // For filtered index groups (e.g. protein-only or RNA-only subsets), the
    // old "stamp from the beginning of the whole system" logic was wrong: it
    // could project chain ids from unrelated molecules onto the subset. That
    // is exactly how a single 800-nt RNA could be split into several fake
    // chains in LLPS DBSCAN output.
    //
    // To keep subset runs safe, we only overwrite chain ids when we can match
    // the current info sequence molecule-by-molecule against topology
    // templates. Otherwise we keep the native PDB/index-derived chain ids that
    // were already assigned by Multi_MOL::set_chainnum().
    {
      const auto atom_matches = [](const mol_info &mi, const AtomDef &atom) {
        return mi.get_res_name() == atom.resname && mi.get_atom() == atom.atomname;
      };

      std::vector<size_t> original_chainnums(info.size());
      for (size_t i = 0; i < info.size(); ++i)
        original_chainnums[i] = info[i].get_chainnum();

      std::vector<size_t> candidate_chainnums = original_chainnums;
      size_t assigned_atoms = 0;
      size_t assigned_chains = 0;
      std::string assignment_mode = "pdb-native (fallback)";

      if (info.size() == flat_system.size()) {
        size_t bead_offset = 0;
        for (const auto &[mol_name, mol_count] : system_molecules) {
          auto it = molecule_types.find(mol_name);
          if (it == molecule_types.end())
            continue;
          const size_t beads_per_mol = it->second.atoms.size();
          for (int c = 0; c < mol_count && bead_offset < candidate_chainnums.size();
               ++c) {
            const size_t span =
                std::min(beads_per_mol, candidate_chainnums.size() - bead_offset);
            for (size_t b = 0; b < span; ++b)
              candidate_chainnums[bead_offset + b] = assigned_chains;
            bead_offset += beads_per_mol;
            ++assigned_chains;
          }
        }
        assigned_atoms = info.size();
        assignment_mode = "topology-full-system";
      } else {
        size_t info_offset = 0;
        for (const auto &[mol_name, mol_count] : system_molecules) {
          auto it = molecule_types.find(mol_name);
          if (it == molecule_types.end())
            continue;
          const auto &tmpl_atoms = it->second.atoms;
          const size_t beads_per_mol = tmpl_atoms.size();
          for (int c = 0; c < mol_count && info_offset < info.size(); ++c) {
            if (info_offset + beads_per_mol > info.size())
              continue;

            bool matches = true;
            for (size_t b = 0; b < beads_per_mol; ++b) {
              if (!atom_matches(info[info_offset + b], tmpl_atoms[b])) {
                matches = false;
                break;
              }
            }
            if (!matches)
              continue;

            for (size_t b = 0; b < beads_per_mol; ++b)
              candidate_chainnums[info_offset + b] = assigned_chains;
            info_offset += beads_per_mol;
            assigned_chains++;
          }

          if (info_offset == info.size())
            break;
        }

        if (info_offset == info.size()) {
          assigned_atoms = info.size();
          assignment_mode = "topology-sequence-matched-subset";
        } else {
          assigned_atoms = 0;
          assigned_chains = 0;
        }
      }

      if (assigned_atoms == info.size()) {
        for (size_t i = 0; i < info.size(); ++i)
          info[i].set_chainnum(candidate_chainnums[i]);
      } else {
        TitanUI::log_message(
            log_file,
            "[WARNING] MARTINI chain assignment could not match this filtered "
            "group against topology templates exactly. Keeping native "
            "PDB/index-derived chain ids.");
        assignment_mode = "pdb-native (subset mismatch)";
        size_t unique_chains = 0;
        if (!original_chainnums.empty()) {
          std::vector<size_t> uniq = original_chainnums;
          std::sort(uniq.begin(), uniq.end());
          uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
          unique_chains = uniq.size();
        }
        assigned_chains = unique_chains;
      }

      TitanUI::log_kv(log_file, "Chain Assignment Mode", assignment_mode);
      TitanUI::log_kv(log_file, "Chains Assigned",
                      std::to_string(assigned_chains));
    }

    TitanUI::print_info("MARTINI Parser: Extracted " +
                        std::to_string(explicit_pair_rules) +
                        " explicit pair rules, mapped " +
                        std::to_string(successfully_mapped) + " atoms.");
  }

  size_t get_flat_size() const { return flat_system.size(); }

  int get_molecule_count(const std::string &target_mol) const {
    for (const auto &comp : system_molecules) {
      if (comp.first == target_mol) {
        return comp.second;
      }
    }
    return 1;
  }
};

int Multi_MOL::parse_martini_topology(const arguments &Arguments,
                                      std::fstream &log_file,
                                      cache_sigma_epsilon *martini_cache) {
  if (info.empty())
    return 0;

  TitanUI::print_info(
      "Initializing Object-Oriented MARTINI Topology Parser...");
  TitanUI::log_section(log_file, "MARTINI Topology Parser");

  MartiniTopology parser;

  // Step 2: Parse Topology Structure
  parser.parse_files(Arguments.topo_name, log_file);
  TitanUI::log_kv(log_file, "Files Parsed", std::to_string(parser.files_read));

  // Step 3: Flatten Templates to Linear Sequence
  parser.build_flat_system();

  // Step 4: Map Properties and Assign Respos/Chains
  parser.map_to_titan(info, parameters, log_file, martini_cache);

  // Auto-detect the primary molecule copies (e.g., molecule_0 or generic
  // Protein) Assuming the primary molecule type is the first one in the [
  // molecules ] section if not explicit
  if (!parser.system_molecules.empty()) {
    int auto_copies = parser.system_molecules.front().second;
    TitanUI::print_info("MARTINI Parser: Auto-detected " +
                        std::to_string(auto_copies) +
                        " copies of primary molecule '" +
                        parser.system_molecules.front().first + "'.");

    // We can't modify const arguments, but we can store this if needed
    // elsewhere, or we handle the copies generation right here based on
    // auto_copies if the design requires it. For now, Titan relies on
    // `Arguments.num_martini_copies` in `generate_martini_copies`. We will need
    // to update the caller or remove the const constraint if we must overwrite
    // it.
  }

  // Re-use `generate_index()` to build the system index using the PDB's native
  // chain/resnr
  // this->generate_index(); // REMOVED because PDB parser already does it!
  return 0;
}

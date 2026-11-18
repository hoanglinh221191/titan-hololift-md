#include "topology/martini_topology.h"

#include "titan_ui.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace titan_martini {

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
void MartiniTopology::assign_chain_numbers(std::vector<mol_info> &info,
                                           std::fstream &log_file) {
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

} // namespace titan_martini

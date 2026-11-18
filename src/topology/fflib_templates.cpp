// Residue templates of the lib_ff assigner: rtp atoms with their
// nonbonded parameters, and the lookup of an input atom in a template.
#include "topology/fflib_detail.h"

#include <cmath>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace TitanFfLib::detail {
namespace {

struct TemplateTablesCacheEntry {
  const amino_acid *amine{};
  const para *parameter{};
  std::size_t residue_count{};
  std::size_t atomtype_count{};
  std::shared_ptr<const TemplateTables> tables;
};

const AtomTemplate *
claim_atom_template_by_key(const ResidueTemplate &templ,
                           const std::string &atom_key,
                           std::unordered_set<std::size_t> &used_templates) {
  const auto it = templ.atom_to_index.find(atom_key);
  if (it == templ.atom_to_index.end() ||
      used_templates.find(it->second) != used_templates.end())
    return nullptr;
  used_templates.insert(it->second);
  return &templ.atoms[it->second];
}

std::string hydrogen_family(std::string atom_key) {
  atom_key = normalize_atom_key(std::move(atom_key));
  if (!is_hydrogen_name(atom_key))
    return atom_key;

  std::string family;
  for (char ch : atom_key) {
    if (std::isdigit(static_cast<unsigned char>(ch)))
      break;
    family.push_back(ch);
  }
  if (family.size() == 1 && atom_key.size() > 1 &&
      std::isalpha(static_cast<unsigned char>(atom_key[1])))
    family.push_back(atom_key[1]);
  return family.empty() ? atom_key : family;
}

TemplateTables build_tables(const amino_acid &amine, const para &parameter) {
  TemplateTables tables;
  tables.residues.reserve(amine.amino_acids.size());
  tables.vdw_by_type.reserve(parameter.base.size());
  for (std::size_t i = 0; i < parameter.base.size(); ++i) {
    if (!parameter.base[i].atom_type.empty())
      tables.vdw_by_type[parameter.base[i].atom_type] = i;
  }

  for (const auto &res_pair : amine.amino_acids) {
    ResidueTemplate templ;
    templ.name = normalize_residue_key(res_pair.first);
    templ.atoms.reserve(res_pair.second.size());
    templ.atom_to_index.reserve(res_pair.second.size());
    for (const auto &atom : res_pair.second) {
      AtomTemplate item;
      item.atom = trim_copy(atom.atom);
      item.atom_key = normalize_atom_key(atom.atom);
      item.atom_type = atom.atom_type;
      item.charge = atom.charge;
      item.mass = atom.atom_mass;
      item.sigma = atom.sigma;
      item.epsilon = atom.epsilon;
      item.radius = atom.radius;
      item.hydrogen = is_hydrogen_name(item.atom_key);

      const auto vdw_it = tables.vdw_by_type.find(item.atom_type);
      if (vdw_it != tables.vdw_by_type.end()) {
        const auto &base = parameter.base[vdw_it->second];
        if (std::fabs(item.sigma) <= 1e-16)
          item.sigma = base.c6_sigma;
        if (std::fabs(item.epsilon) <= 1e-16)
          item.epsilon = base.c12_epsilon;
        if (std::fabs(item.radius) <= 1e-16)
          item.radius = base.radius;
        if (item.mass <= 0.01)
          item.mass = base.mass;
      }

      templ.atom_to_index[item.atom_key] = templ.atoms.size();
      templ.atoms.emplace_back(std::move(item));
    }
    tables.residues.emplace(templ.name, std::move(templ));
  }
  return tables;
}

} // namespace

const AtomTemplate *
find_atom_template_for_input(const ResidueTemplate &templ,
                             const std::string &atom_key,
                             std::unordered_set<std::size_t> &used_templates) {
  if (const AtomTemplate *exact =
          claim_atom_template_by_key(templ, atom_key, used_templates))
    return exact;

  if (has_suffix(templ.name, "ILE") && atom_key == "CD1") {
    if (const AtomTemplate *alias =
            claim_atom_template_by_key(templ, "CD", used_templates))
      return alias;
  }
  // ENE-09 (21/10/2026): the C-terminal carboxylate oxygens are spelled
  // O/OXT (PDB, tleap), OT1/OT2 (CHARMM), O1/O2 (GROMOS/OPLS pdb2gmx) or
  // OC1/OC2 (the AMBER C-residue templates of GROMACS), and the N-terminal
  // ammonium hydrogens HT1-3 (CHARMM) or H1-3 (GROMACS). Map each spelling
  // onto whichever one the template uses.
  const auto try_aliases = [&](std::initializer_list<const char *> keys)
      -> const AtomTemplate * {
    for (const char *key : keys) {
      if (atom_key == key)
        continue;
      if (const AtomTemplate *alias =
              claim_atom_template_by_key(templ, key, used_templates))
        return alias;
    }
    return nullptr;
  };
  if (atom_key == "O" || atom_key == "O1" || atom_key == "OT1" ||
      atom_key == "OC1") {
    if (const AtomTemplate *alias = try_aliases({"O", "OC1", "O1", "OT1"}))
      return alias;
  }
  if (atom_key == "OXT" || atom_key == "O2" || atom_key == "OT2" ||
      atom_key == "OC2") {
    if (const AtomTemplate *alias =
            try_aliases({"OXT", "OC2", "O2", "OT2", "O"}))
      return alias;
  }
  if (atom_key == "HT1" || atom_key == "HT2" || atom_key == "HT3") {
    const std::string gromacs_name = "H" + atom_key.substr(2);
    if (const AtomTemplate *alias = try_aliases({gromacs_name.c_str()}))
      return alias;
  }
  if (atom_key == "HN" || atom_key == "H") {
    if (const AtomTemplate *alias = try_aliases({"H", "HN"}))
      return alias;
  }

  if (!is_hydrogen_name(atom_key))
    return nullptr;

  const std::string family = hydrogen_family(atom_key);
  for (std::size_t i = 0; i < templ.atoms.size(); ++i) {
    if (used_templates.find(i) != used_templates.end())
      continue;
    const AtomTemplate &candidate = templ.atoms[i];
    if (!candidate.hydrogen)
      continue;
    if (hydrogen_family(candidate.atom_key) != family)
      continue;
    used_templates.insert(i);
    return &candidate;
  }
  return nullptr;
}

bool template_is_protein(const ResidueTemplate &templ) {
  for (const char *key : {"N", "CA", "C", "O"}) {
    if (templ.atom_to_index.find(key) == templ.atom_to_index.end())
      return false;
  }
  return true;
}

std::shared_ptr<const TemplateTables>
cached_tables(const amino_acid &amine, const para &parameter) {
  static std::mutex cache_mutex;
  static std::vector<TemplateTablesCacheEntry> cache;

  std::lock_guard<std::mutex> lock(cache_mutex);
  for (const auto &entry : cache) {
    if (entry.amine == &amine && entry.parameter == &parameter &&
        entry.residue_count == amine.amino_acids.size() &&
        entry.atomtype_count == parameter.base.size())
      return entry.tables;
  }

  auto tables = std::make_shared<TemplateTables>(build_tables(amine, parameter));
  cache.push_back(TemplateTablesCacheEntry{
      &amine, &parameter, amine.amino_acids.size(), parameter.base.size(),
      tables});
  return tables;
}

} // namespace TitanFfLib::detail

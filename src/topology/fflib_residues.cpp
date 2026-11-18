// Residue spans of the input structure and the template candidates of
// each span (terminal, histidine, nucleic-acid and inferred names).
#include "topology/fflib_detail.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace TitanFfLib::detail {
namespace {

std::string residue_key(const std::string &chain, std::size_t resnr,
                        const std::string &resname) {
  return chain + '\x1f' + std::to_string(resnr) + '\x1f' + resname;
}

void add_unique_candidate(std::vector<std::string> &items,
                          const std::string &value) {
  if (value.empty())
    return;
  if (std::find(items.begin(), items.end(), value) == items.end())
    items.emplace_back(value);
}

// ENE-09 (21/10/2026). Histidine protonation names of each force-field family.
// take_coordinates.cpp already renames PDB "HIS" from its hydrogens for the
// main force fields; this is the fallback for names that reach the assigner
// unmapped (heavy-atom input, or a protonation name of another force field).
// Explicit HD1/HE2 hydrogens decide the state; without hydrogens the name's
// own state is kept and a plain HIS takes the family default (delta for
// CHARMM as in CHARMM-GUI, epsilon elsewhere as in tleap/pdb2gmx).
std::string histidine_candidate(const Multi_MOL &mol, const ResidueSpan &span,
                                const std::string &name,
                                std::string_view force_f) {
  static const std::unordered_map<std::string, char> state_of_name{
      {"HIS", '?'},  {"HID", 'D'},  {"HIE", 'E'},  {"HIP", 'P'},
      {"HSD", 'D'},  {"HSE", 'E'},  {"HSP", 'P'},  {"HISA", 'D'},
      {"HISB", 'E'}, {"HISH", 'P'}, {"HISD", 'D'}, {"HISE", 'E'}};
  const auto state_it = state_of_name.find(name);
  if (state_it == state_of_name.end())
    return {};

  bool hd1 = false, he2 = false, any_hydrogen = false;
  for (std::size_t i = span.begin; i < span.end && i < mol.info.size(); ++i) {
    const std::string key = normalize_atom_key(mol.info[i].get_atom());
    if (key == "HD1")
      hd1 = true;
    else if (key == "HE2")
      he2 = true;
    if (is_hydrogen_name(key))
      any_hydrogen = true;
  }
  char state = state_it->second;
  if (any_hydrogen && (hd1 || he2))
    state = (hd1 && he2) ? 'P' : (hd1 ? 'D' : 'E');
  if (state == '?')
    state = forcefield_is_charmm(force_f) ? 'D' : 'E';

  const char *const *names = nullptr;
  static const char *const amber[] = {"HID", "HIE", "HIP"};
  static const char *const charmm[] = {"HSD", "HSE", "HSP"};
  static const char *const gromos[] = {"HISA", "HISB", "HISH"};
  static const char *const opls[] = {"HISD", "HISE", "HISH"};
  if (forcefield_is_amber(force_f))
    names = amber;
  else if (forcefield_is_charmm(force_f))
    names = charmm;
  else if (forcefield_is_gromos(force_f))
    names = gromos;
  else if (forcefield_is_opls(force_f))
    names = opls;
  if (names == nullptr)
    return {};
  return names[state == 'D' ? 0 : (state == 'E' ? 1 : 2)];
}

bool residue_is_rna_like(const Multi_MOL &mol, const ResidueSpan &span,
                         const std::string &base_name) {
  if (base_name == "U" || base_name == "RU" || base_name == "RA" ||
      base_name == "RC" || base_name == "RG" || base_name == "ADE" ||
      base_name == "URA" || base_name == "CYT" || base_name == "GUA")
    return true;
  for (std::size_t i = span.begin; i < span.end; ++i) {
    if (normalize_atom_key(mol.info[i].get_atom()) == "O2'")
      return true;
  }
  return false;
}

std::string nucleic_base_code(const std::string &name) {
  std::string n = name;
  if (!n.empty() && (n.back() == '5' || n.back() == '3' || n.back() == 'N'))
    n.pop_back();
  if (n == "ADE" || n == "RA")
    return "A";
  if (n == "URA" || n == "RU")
    return "U";
  if (n == "CYT" || n == "RC")
    return "C";
  if (n == "GUA" || n == "RG")
    return "G";
  if (n == "THY" || n == "DT")
    return "T";
  if (n == "DA")
    return "A";
  if (n == "DC")
    return "C";
  if (n == "DG")
    return "G";
  if (n == "A" || n == "C" || n == "G" || n == "U" || n == "T")
    return n;
  return {};
}

std::string infer_protein_residue_from_atoms(const Multi_MOL &mol,
                                             const ResidueSpan &span) {
  std::unordered_set<std::string> atoms;
  atoms.reserve(span.end - span.begin);
  for (std::size_t i = span.begin; i < span.end; ++i)
    atoms.insert(normalize_atom_key(mol.info[i].get_atom()));

  auto has = [&](std::string_view atom) {
    return atoms.find(std::string(atom)) != atoms.end();
  };

  if (!has("N") || !has("CA") || !has("C") || !has("O"))
    return {};
  if (has("CG") && has("OD1") && has("ND2"))
    return "ASN";
  if (has("CD") && has("OE1") && has("NE2"))
    return "GLN";
  if (has("CG") && has("OD1") && has("OD2"))
    return "ASP";
  if (has("CD") && has("OE1") && has("OE2"))
    return "GLU";
  if (has("CG1") && has("CG2") && (has("CD") || has("CD1")))
    return "ILE";
  if (has("CG") && has("CD1") && has("CD2"))
    return "LEU";
  if (has("CG") && has("CD") && has("CE") && has("NZ"))
    return "LYS";
  if (has("CG") && has("CD") && has("NE") && has("CZ") && has("NH1") &&
      has("NH2"))
    return "ARG";
  if (has("CG") && has("SD") && has("CE"))
    return "MET";
  if (has("CB") && has("SG"))
    return "CYS";
  if (has("CB") && has("OG"))
    return "SER";
  if (has("CB") && has("OG1") && has("CG2"))
    return "THR";
  if (has("CB") && has("CG") && has("CD1") && has("CD2") && has("CE1") &&
      has("CE2") && has("CZ") && has("OH"))
    return "TYR";
  if (has("CB") && has("CG") && has("CD1") && has("CD2") && has("CE1") &&
      has("CE2") && has("CZ"))
    return "PHE";
  if (has("CB") && has("CG") && has("CD1") && has("NE1") && has("CE2"))
    return "TRP";
  if (has("CB") && has("CG") && has("CD2") && (has("ND1") || has("NE2")))
    return "HIS";
  if (has("CB") && has("CG") && has("CD"))
    return "PRO";
  if (has("CB") && has("CG1") && has("CG2"))
    return "VAL";
  if (has("CB"))
    return "ALA";
  if (has("HA2") || has("HA3"))
    return "GLY";
  return {};
}

std::vector<std::string>
residue_template_candidates(const Multi_MOL &mol, const ResidueSpan &span,
                            std::string_view force_f) {
  std::vector<std::string> out;
  const std::string current = normalize_residue_key(span.resname);
  std::string original = current;
  if (span.begin < mol.info.size())
    original = normalize_residue_key(mol.info[span.begin].get_original_res_name());

  add_unique_candidate(out, current);
  add_unique_candidate(out, original);
  add_unique_candidate(out, histidine_candidate(mol, span, current, force_f));
  add_unique_candidate(out, histidine_candidate(mol, span, original, force_f));

  const bool amber = forcefield_is_amber(force_f);
  const bool charmm = forcefield_is_charmm(force_f);
  auto add_terminal = [&](const std::string &base) {
    if (base.empty())
      return;
    if (amber) {
      if (span.first_in_chain)
        add_unique_candidate(out, "N" + base);
      if (span.last_in_chain)
        add_unique_candidate(out, "C" + base);
    }
    add_unique_candidate(out, base);
  };

  std::string protein_base = current;
  if (protein_base.size() == 4 &&
      (protein_base[0] == 'N' || protein_base[0] == 'C'))
    protein_base = protein_base.substr(1);
  add_terminal(protein_base);
  add_terminal(infer_protein_residue_from_atoms(mol, span));

  const std::string nuc = nucleic_base_code(current.empty() ? original : current);
  if (!nuc.empty()) {
    const bool rna = residue_is_rna_like(mol, span, current);
    if (amber) {
      const std::string prefix = rna ? "R" : "D";
      const std::string amber_base = prefix + nuc;
      if (span.first_in_chain)
        add_unique_candidate(out, amber_base + "5");
      if (span.last_in_chain)
        add_unique_candidate(out, amber_base + "3");
      if (span.first_in_chain && span.last_in_chain)
        add_unique_candidate(out, amber_base + "N");
      add_unique_candidate(out, amber_base);
    }
    if (charmm) {
      if (rna) {
        if (nuc == "A")
          add_unique_candidate(out, "ADE");
        else if (nuc == "U")
          add_unique_candidate(out, "URA");
        else if (nuc == "C")
          add_unique_candidate(out, "CYT");
        else if (nuc == "G")
          add_unique_candidate(out, "GUA");
      } else {
        add_unique_candidate(out, "D" + nuc);
      }
    }
  }

  return out;
}

} // namespace

const ResidueTemplate *
find_template(const TemplateTables &tables, const Multi_MOL &mol,
              const ResidueSpan &span, std::string_view force_f,
              std::string &matched_name) {
  for (const auto &candidate : residue_template_candidates(mol, span, force_f)) {
    const auto it = tables.residues.find(candidate);
    if (it != tables.residues.end()) {
      matched_name = candidate;
      return &it->second;
    }
  }
  matched_name.clear();
  return nullptr;
}

std::vector<ResidueSpan> build_residue_spans(const Multi_MOL &mol) {
  std::vector<ResidueSpan> spans;
  if (mol.info.empty())
    return spans;

  std::unordered_map<std::string, std::string> old_types;
  if (mol.Index.size() > 1) {
    old_types.reserve(mol.Index.size() * 4);
    for (std::size_t c = 0; c + 1 < mol.Index.size(); ++c) {
      for (std::size_t r = 0; r < mol.Index[c].get_size_v(); ++r) {
        old_types[residue_key(mol.Index[c].get_chain(),
                              mol.Index[c].get_res_num_e(r),
                              mol.Index[c].get_res_name_e(r))] =
            mol.Index[c].get_type_base(r);
      }
    }
  }

  std::size_t begin = 0;
  for (std::size_t i = 1; i <= mol.info.size(); ++i) {
    const bool end_span =
        i == mol.info.size() ||
        mol.info[i].get_resnr() != mol.info[begin].get_resnr() ||
        mol.info[i].get_chain() != mol.info[begin].get_chain() ||
        mol.info[i].get_res_name() != mol.info[begin].get_res_name();
    if (!end_span)
      continue;

    ResidueSpan span;
    span.begin = begin;
    span.end = i;
    span.chain = mol.info[begin].get_chain();
    span.resnr = mol.info[begin].get_resnr();
    span.resname = mol.info[begin].get_res_name();
    const auto type_it =
        old_types.find(residue_key(span.chain, span.resnr, span.resname));
    if (type_it != old_types.end())
      span.type = type_it->second;
    spans.emplace_back(std::move(span));
    begin = i;
  }

  for (std::size_t i = 0; i < spans.size(); ++i) {
    spans[i].first_in_chain = (i == 0 || spans[i - 1].chain != spans[i].chain);
    spans[i].last_in_chain =
        (i + 1 == spans.size() || spans[i + 1].chain != spans[i].chain);
  }
  return spans;
}

} // namespace TitanFfLib::detail

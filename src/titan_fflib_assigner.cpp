#include "titan_fflib_assigner.h"

#include "MOL.h"
#include "polymer_classification.h"
#include "titan_executor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace TitanFfLib {
namespace {

constexpr std::size_t kMaxAssignedReportRows = 2000;
constexpr double kHydrogenOffset = 0.10;

struct AtomTemplate {
  std::string atom;
  std::string atom_key;
  std::string atom_type;
  double charge{};
  double mass{1.0};
  double sigma{};
  double epsilon{};
  double radius{};
  bool hydrogen{};
};

struct ResidueTemplate {
  std::string name;
  std::vector<AtomTemplate> atoms;
  std::unordered_map<std::string, std::size_t> atom_to_index;
};

struct TemplateTables {
  std::unordered_map<std::string, ResidueTemplate> residues;
  std::unordered_map<std::string, std::size_t> vdw_by_type;
};

struct TemplateTablesCacheEntry {
  const amino_acid *amine{};
  const para *parameter{};
  std::size_t residue_count{};
  std::size_t atomtype_count{};
  std::shared_ptr<const TemplateTables> tables;
};

struct ResidueSpan {
  std::size_t begin{};
  std::size_t end{};
  std::string chain;
  std::size_t resnr{};
  std::string resname;
  std::string type{"unknown"};
  bool first_in_chain{};
  bool last_in_chain{};
};

struct HydrogenPlan {
  AtomTemplate atom;
  std::size_t anchor{};
  bool can_place{};
  std::string message;
};

struct ResiduePlan {
  const ResidueTemplate *templ{};
  std::string templ_name;
  std::vector<HydrogenPlan> hydrogens;
  std::size_t assigned{};
  std::size_t unmatched{};
  std::size_t missing_vdw{};
  std::size_t report_assigned{};
  std::vector<std::string> report_rows;
};

std::mutex &report_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string upper_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string normalize_atom_key(std::string atom) {
  atom = trim_copy(upper_copy(std::move(atom)));
  atom.erase(std::remove(atom.begin(), atom.end(), ' '), atom.end());
  std::replace(atom.begin(), atom.end(), '*', '\'');
  if (atom == "O1P")
    return "OP1";
  if (atom == "O2P")
    return "OP2";
  if (atom == "O3P")
    return "OP3";
  if (atom.size() > 1 && std::isdigit(static_cast<unsigned char>(atom[0])))
    atom = atom.substr(1) + atom.substr(0, 1);
  return atom;
}

std::string normalize_residue_key(std::string res) {
  res = trim_copy(upper_copy(std::move(res)));
  res.erase(std::remove(res.begin(), res.end(), ' '), res.end());
  return res;
}

bool is_hydrogen_name(const std::string &atom_key) {
  return !atom_key.empty() && atom_key[0] == 'H';
}

bool has_suffix(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

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
  if (atom_key == "O1" || atom_key == "OT1") {
    if (const AtomTemplate *alias =
            claim_atom_template_by_key(templ, "O", used_templates))
      return alias;
  }
  if (atom_key == "O2" || atom_key == "OT2") {
    if (const AtomTemplate *alias =
            claim_atom_template_by_key(templ, "OXT", used_templates))
      return alias;
    if (const AtomTemplate *alias =
            claim_atom_template_by_key(templ, "O", used_templates))
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

std::string report_escape(std::string value) {
  for (char &ch : value) {
    if (ch == '\t' || ch == '\n' || ch == '\r')
      ch = ' ';
  }
  return value;
}

std::string make_report_row(std::string_view molecule, const ResidueSpan &span,
                            std::string_view atom, std::string_view action,
                            std::string_view templ,
                            std::string_view atom_type, double charge,
                            double sigma, double epsilon,
                            std::string_view message) {
  std::ostringstream out;
  out << report_escape(std::string(molecule)) << '\t'
      << report_escape(span.chain) << '\t' << span.resnr << '\t'
      << report_escape(span.resname) << '\t'
      << report_escape(std::string(atom)) << '\t'
      << report_escape(std::string(action)) << '\t'
      << report_escape(std::string(templ)) << '\t'
      << report_escape(std::string(atom_type)) << '\t' << std::setprecision(10)
      << charge << '\t' << sigma << '\t' << epsilon << '\t'
      << report_escape(std::string(message)) << '\n';
  return out.str();
}

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

bool forcefield_is_amber(std::string_view force_f) {
  return std::string(force_f).find("AMBER") != std::string::npos;
}

bool forcefield_is_charmm(std::string_view force_f) {
  return std::string(force_f).find("CHARMM") != std::string::npos;
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

void set_atom_parameter(Multi_MOL &mol, std::size_t atom_index,
                        const AtomTemplate &templ) {
  mol.info[atom_index].set_atom_type(templ.atom_type);
  if (atom_index >= mol.parameters.size())
    mol.parameters.resize(mol.info.size());
  mol.parameters[atom_index].set_data(templ.charge, templ.radius, templ.sigma,
                                      templ.epsilon, templ.mass);
}

void set_unassigned(Multi_MOL &mol, std::size_t atom_index) {
  mol.info[atom_index].set_atom_type("UNASSIGNED");
  if (atom_index >= mol.parameters.size())
    mol.parameters.resize(mol.info.size());
  mol.parameters[atom_index].set_data(0.0, 0.0, 0.0, 0.0, 1.0);
  mol.parameters[atom_index].set_type_id(-1);
}

std::size_t find_anchor_atom(const Multi_MOL &mol, const ResidueSpan &span,
                             const std::string &hydrogen_key) {
  std::vector<std::string> candidates;
  if (hydrogen_key == "H" || hydrogen_key == "HN" ||
      hydrogen_key == "H1" || hydrogen_key == "H2" ||
      hydrogen_key == "H3") {
    candidates.emplace_back("N");
  }
  if (hydrogen_key.size() >= 2) {
    const char region = hydrogen_key[1];
    if (region == 'A')
      candidates.emplace_back("CA");
    else if (region == 'B')
      candidates.emplace_back("CB");
    else if (region == 'G')
      candidates.emplace_back("CG");
    else if (region == 'D')
      candidates.emplace_back("CD");
    else if (region == 'E')
      candidates.emplace_back("CE");
    else if (region == 'Z')
      candidates.emplace_back("CZ");
    else if (std::isdigit(static_cast<unsigned char>(region))) {
      candidates.emplace_back("C" + std::string(1, region));
      candidates.emplace_back("N" + std::string(1, region));
    }
  }
  if (hydrogen_key.find("'") != std::string::npos &&
      hydrogen_key.size() >= 2) {
    candidates.emplace_back("C" + std::string(1, hydrogen_key[1]) + "'");
    candidates.emplace_back("O" + std::string(1, hydrogen_key[1]) + "'");
  }
  candidates.emplace_back("N");
  candidates.emplace_back("CA");
  candidates.emplace_back("C");
  candidates.emplace_back("P");

  for (const auto &candidate : candidates) {
    for (std::size_t i = span.begin; i < span.end; ++i) {
      if (normalize_atom_key(mol.info[i].get_atom()) == candidate)
        return i;
    }
  }
  for (std::size_t i = span.begin; i < span.end; ++i) {
    if (!is_hydrogen_name(normalize_atom_key(mol.info[i].get_atom())))
      return i;
  }
  return span.begin;
}

std::tuple<double, double, double> hydrogen_position(const MOL &frame,
                                                     std::size_t anchor,
                                                     std::size_t salt) {
  const double x = anchor < frame.pos.x.size() ? frame.pos.x[anchor] : 0.0;
  const double y = anchor < frame.pos.y.size() ? frame.pos.y[anchor] : 0.0;
  const double z = anchor < frame.pos.z.size() ? frame.pos.z[anchor] : 0.0;
  const double sx = (salt % 3 == 0) ? 1.0 : (salt % 3 == 1 ? -0.5 : 0.5);
  const double sy = (salt % 5 == 0) ? 0.0 : (salt % 5 < 3 ? 0.866 : -0.866);
  const double sz = (salt % 7 == 0) ? 0.707 : -0.707;
  return {x + kHydrogenOffset * sx, y + kHydrogenOffset * sy,
          z + kHydrogenOffset * sz};
}

bool should_report_hydrogens(const arguments &Arguments) {
  return Arguments.fflib_hydrogens == "reference" && !Arguments.flag_xdr;
}

bool should_add_hydrogens(const arguments &Arguments, const Multi_MOL &mol) {
  if (!should_report_hydrogens(Arguments))
    return false;
  if (Arguments.flag_1_pdb &&
      (!Arguments.flag_nocal || Arguments.cmap.flag_cmap) &&
      mol.m_mol.size() <= 1)
    return false;
  return true;
}

std::string hydrogen_skip_policy_message(const arguments &Arguments,
                                         const Multi_MOL &mol) {
  if (Arguments.flag_xdr)
    return "hydrogen not added for XDR trajectory atom-count stability";
  if (Arguments.flag_1_pdb &&
      (!Arguments.flag_nocal || Arguments.cmap.flag_cmap) &&
      mol.m_mol.size() <= 1)
    return "hydrogen not added because PDB trajectory frames keep original atom count";
  return "no heavy atom anchor found";
}

void restore_index_types(Multi_MOL &mol, const std::vector<ResidueSpan> &spans) {
  if (mol.Index.size() <= 1)
    return;

  std::size_t span_pos = 0;
  for (std::size_t c = 0; c + 1 < mol.Index.size(); ++c) {
    std::unordered_set<std::string> chain_types;
    for (std::size_t r = 0; r < mol.Index[c].get_size_v(); ++r) {
      const std::string type =
          span_pos < spans.size() ? spans[span_pos].type : "unknown";
      mol.Index[c].set_type_base(r, type);
      chain_types.insert(type);
      ++span_pos;
    }
    if (chain_types.size() == 1)
      mol.Index[c].set_type(*chain_types.begin());
    else if (!chain_types.empty())
      mol.Index[c].set_type("complex");
  }
}

void append_report(const std::vector<ResiduePlan> &plans,
                   std::string_view molecule_label, std::size_t atom_count,
                   std::size_t final_count, bool strict_missing) {
  std::lock_guard<std::mutex> lock(report_mutex());
  const std::filesystem::path filename("fflib_assignment_report.tsv");
  static bool report_initialized = false;

  const std::ios::openmode mode =
      report_initialized ? (std::ios::out | std::ios::app)
                         : (std::ios::out | std::ios::trunc);
  std::fstream out(filename.string(), mode);
  if (!out.is_open())
    return;
  if (!report_initialized) {
    out << "molecule\tchain\tresnr\tresname\tatom\taction\ttemplate"
        << "\tatom_type\tcharge\tsigma\tepsilon\tmessage\n";
    report_initialized = true;
  }

  std::size_t assigned = 0;
  std::size_t unmatched = 0;
  std::size_t missing_vdw = 0;
  std::size_t added_h = 0;
  for (const auto &plan : plans) {
    assigned += plan.assigned;
    unmatched += plan.unmatched;
    missing_vdw += plan.missing_vdw;
    for (const auto &h : plan.hydrogens) {
      if (h.can_place)
        ++added_h;
    }
    for (const auto &row : plan.report_rows)
      out << row;
  }

  ResidueSpan summary;
  summary.chain = "*";
  summary.resname = "*";
  out << make_report_row(molecule_label, summary, "*",
                         strict_missing && unmatched > 0 ? "summary_failed"
                                                         : "summary",
                         "*", "*", 0.0, 0.0, 0.0,
                         "input_atoms=" + std::to_string(atom_count) +
                             "; final_atoms=" + std::to_string(final_count) +
                             "; assigned=" + std::to_string(assigned) +
                             "; unmatched=" + std::to_string(unmatched) +
                             "; missing_vdw=" + std::to_string(missing_vdw) +
                             "; added_h=" + std::to_string(added_h));
}

} // namespace

bool fill_vdw_from_atom_type(Multi_MOL_parameters &parameters,
                             std::size_t atom_index, const para &parameter,
                             std::string_view atom_type) {
  if (atom_index >= parameters.size())
    return false;

  const std::string atom_type_key(atom_type);
  for (std::size_t j = parameter.base.size(); j-- > 0;) {
    if (atom_type_key == parameter.base[j].atom_type) {
      parameters[atom_index].set_epsilon_sigma_radius(
          parameter.base[j].c12_epsilon, parameter.base[j].c6_sigma,
          parameter.base[j].radius);
      parameters[atom_index].set_type_id(static_cast<int>(j));
      return true;
    }
  }
  parameters[atom_index].set_type_id(-1);
  return false;
}

int assign_from_library(Multi_MOL &mol, const amino_acid &amine,
                        const para &parameter, const arguments &Arguments,
                        std::string_view force_f, TitanExecutor *executor,
                        std::string_view molecule_label) {
  if (mol.info.empty() || mol.m_mol.empty())
    return 0;

  if (mol.parameters.size() < mol.info.size())
    mol.parameters.resize(mol.info.size());

  const auto tables_holder = cached_tables(amine, parameter);
  const TemplateTables &tables = *tables_holder;
  const std::vector<ResidueSpan> spans = build_residue_spans(mol);
  std::vector<ResiduePlan> plans(spans.size());
  const bool report_h = should_report_hydrogens(Arguments);
  const bool add_h = should_add_hydrogens(Arguments, mol);

  auto assign_span = [&](std::size_t worker, std::size_t begin,
                         std::size_t end) {
    (void)worker;
    for (std::size_t s = begin; s < end; ++s) {
      auto &plan = plans[s];
      const auto &span = spans[s];
      plan.templ = find_template(tables, mol, span, force_f, plan.templ_name);
      if (plan.templ == nullptr) {
        for (std::size_t i = span.begin; i < span.end; ++i) {
          set_unassigned(mol, i);
          ++plan.unmatched;
          plan.report_rows.emplace_back(make_report_row(
              molecule_label, span, mol.info[i].get_atom(), "unmatched_residue",
              "", "UNASSIGNED", 0.0, 0.0, 0.0,
              "residue template not found in lib_ff"));
        }
        continue;
      }

      std::unordered_set<std::string> present_atoms;
      present_atoms.reserve(span.end - span.begin);
      std::unordered_set<std::size_t> used_template_atoms;
      used_template_atoms.reserve(span.end - span.begin);
      for (std::size_t i = span.begin; i < span.end; ++i) {
        const std::string atom_key = normalize_atom_key(mol.info[i].get_atom());
        present_atoms.insert(atom_key);
        const AtomTemplate *atom_template = find_atom_template_for_input(
            *plan.templ, atom_key, used_template_atoms);
        if (atom_template == nullptr) {
          set_unassigned(mol, i);
          ++plan.unmatched;
          plan.report_rows.emplace_back(make_report_row(
              molecule_label, span, mol.info[i].get_atom(), "unmatched_atom",
              plan.templ_name, "UNASSIGNED", 0.0, 0.0, 0.0,
              "atom not found in residue template"));
          continue;
        }

        set_atom_parameter(mol, i, *atom_template);
        if (!fill_vdw_from_atom_type(mol.parameters, i, parameter,
                                     atom_template->atom_type)) {
          ++plan.missing_vdw;
          plan.report_rows.emplace_back(make_report_row(
              molecule_label, span, mol.info[i].get_atom(), "missing_vdw",
              plan.templ_name, atom_template->atom_type, atom_template->charge,
              atom_template->sigma, atom_template->epsilon,
              "atom type not found in nonbonded table"));
        }
        ++plan.assigned;
        if (i < kMaxAssignedReportRows) {
          plan.report_rows.emplace_back(make_report_row(
              molecule_label, span, mol.info[i].get_atom(), "assigned",
              plan.templ_name, atom_template->atom_type, atom_template->charge,
              atom_template->sigma, atom_template->epsilon, ""));
        }
      }

      if (!report_h)
        continue;

      for (std::size_t atom_index = 0; atom_index < plan.templ->atoms.size();
           ++atom_index) {
        const auto &atom_template = plan.templ->atoms[atom_index];
        if (!atom_template.hydrogen ||
            used_template_atoms.find(atom_index) != used_template_atoms.end() ||
            present_atoms.find(atom_template.atom_key) != present_atoms.end())
          continue;
        HydrogenPlan h;
        h.atom = atom_template;
        h.anchor = find_anchor_atom(mol, span, atom_template.atom_key);
        h.can_place = add_h && h.anchor < span.end;
        h.message = h.can_place
                        ? "hydrogen placed from heavy atom anchor"
                        : (h.anchor >= span.end
                               ? "no heavy atom anchor found"
                               : hydrogen_skip_policy_message(Arguments, mol));
        plan.hydrogens.emplace_back(std::move(h));
        plan.report_rows.emplace_back(make_report_row(
            molecule_label, span, atom_template.atom,
            plan.hydrogens.back().can_place ? "added_hydrogen"
                                            : "skipped_hydrogen",
            plan.templ_name, atom_template.atom_type, atom_template.charge,
            atom_template.sigma, atom_template.epsilon,
            plan.hydrogens.back().message));
      }
    }
  };

  if (executor != nullptr && spans.size() > 1) {
    executor->parallel_for(spans.size(), Arguments.nt, 16, assign_span);
  } else {
    assign_span(0, 0, spans.size());
  }

  std::size_t extra_atoms = 0;
  for (const auto &plan : plans) {
    for (const auto &h : plan.hydrogens) {
      if (h.can_place)
        ++extra_atoms;
    }
  }

  const bool strict_missing = Arguments.fflib_missing == "strict";
  const std::size_t original_atom_count = mol.info.size();
  if (extra_atoms > 0) {
    std::vector<std::size_t> output_offsets(spans.size() + 1, 0);
    for (std::size_t s = 0; s < spans.size(); ++s) {
      std::size_t span_extra = 0;
      for (const auto &h : plans[s].hydrogens) {
        if (h.can_place)
          ++span_extra;
      }
      output_offsets[s + 1] =
          output_offsets[s] + (spans[s].end - spans[s].begin) + span_extra;
    }

    const std::size_t final_atoms = output_offsets.back();
    std::vector<mol_info> new_info(final_atoms);
    Multi_MOL_parameters new_parameters(final_atoms);
    new_parameters.ensure_type_ids();
    std::vector<AlignedVector3D> new_positions(mol.m_mol.size());
    for (auto &pos : new_positions)
      pos.resize(final_atoms);

    auto rebuild_span = [&](std::size_t worker, std::size_t begin,
                            std::size_t end) {
      (void)worker;
      for (std::size_t s = begin; s < end; ++s) {
        const auto &span = spans[s];
        std::size_t out = output_offsets[s];
        for (std::size_t atom = span.begin; atom < span.end; ++atom, ++out) {
          new_info[out] = mol.info[atom];
          new_parameters[out].set_data(mol.parameters[atom].get_charge(),
                                       mol.parameters[atom].get_radius(),
                                       mol.parameters[atom].get_sigma(),
                                       mol.parameters[atom].get_epsilon(),
                                       mol.parameters[atom].get_mass());
          new_parameters[out].set_type_id(mol.parameters[atom].get_type_id());
          for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
            if (atom < mol.m_mol[f].pos.x.size()) {
              new_positions[f].x[out] = mol.m_mol[f].pos.x[atom];
              new_positions[f].y[out] = mol.m_mol[f].pos.y[atom];
              new_positions[f].z[out] = mol.m_mol[f].pos.z[atom];
            }
          }
        }
        for (const auto &h : plans[s].hydrogens) {
          if (!h.can_place)
            continue;
          mol_info h_info(span.resnr, h.atom.atom_type, h.atom.atom, span.chain,
                          span.resname);
          h_info.set_side_chain(true);
          new_info[out] = std::move(h_info);
          new_parameters[out].set_data(h.atom.charge, h.atom.radius,
                                       h.atom.sigma, h.atom.epsilon,
                                       h.atom.mass);
          fill_vdw_from_atom_type(new_parameters, out, parameter,
                                  h.atom.atom_type);
          for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
            const auto [x, y, z] =
                hydrogen_position(mol.m_mol[f], h.anchor, out + f);
            new_positions[f].x[out] = x;
            new_positions[f].y[out] = y;
            new_positions[f].z[out] = z;
          }
          ++out;
        }
      }
    };

    if (executor != nullptr && spans.size() > 1) {
      executor->parallel_for(spans.size(), Arguments.nt, 16, rebuild_span);
    } else {
      rebuild_span(0, 0, spans.size());
    }

    mol.info = std::move(new_info);
    mol.parameters = std::move(new_parameters);
    for (std::size_t f = 0; f < mol.m_mol.size(); ++f) {
      mol.m_mol[f].pos = std::move(new_positions[f]);
      mol.m_mol[f].force_vec.reset();
      mol.m_mol[f].thread_local_forces.clear();
      mol.m_mol[f].Com.clear();
    }
    mol.Index.clear();
    mol.generate_index();
    restore_index_types(mol, spans);
    mol.set_chainnum();
    if (Arguments.flag_sidechain)
      mol.check_side_chain(Arguments);
  }

  for (std::size_t i = 0; i < mol.info.size() && i < mol.parameters.size(); ++i) {
    const auto type_it = parameter.atom_type_to_id.find(mol.info[i].get_atom_type());
    if (type_it != parameter.atom_type_to_id.end())
      mol.parameters[i].set_type_id(static_cast<int>(type_it->second));
  }

  append_report(plans, molecule_label, original_atom_count, mol.info.size(),
                strict_missing);

  std::size_t unmatched = 0;
  for (const auto &plan : plans)
    unmatched += plan.unmatched + plan.missing_vdw;
  if (unmatched > 0) {
    std::cout << "WARNING (" << molecule_label << "): " << unmatched
              << " atoms/VDW records were not fully assigned by lib_ff. See "
                 "fflib_assignment_report.tsv\n";
    if (strict_missing) {
      std::cerr << "ERROR (" << molecule_label
                << "): -fflib-missing strict requested.\n";
      std::exit(EXIT_FAILURE);
    }
  }

  return 0;
}

} // namespace TitanFfLib

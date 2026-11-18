#include "topology/fflib_detail.h"

#include <initializer_list>
#include <string>

namespace TitanFfLib::detail {

// ENE-09 (21/10/2026). Terminal patches. The CHARMM, GROMOS and OPLS-AA rtp
// files in lib_ff carry no N-/C-terminal residues: GROMACS applies the
// patches of aminoacids.n.tdb / aminoacids.c.tdb at pdb2gmx time, so a PDB
// from such a run has H1/H2/H3 on the first N and two carboxylate oxygens on
// the last C, with charges that no residue template reproduces. The tables
// below are the default zwitterion patches of those files (GROMACS 2026.3
// share/top: charmm27.ff NH3+/GLY-NH3+/PRO-NH2+/COO- = CHARMM36 NTER/GLYP/
// PROP/CTER; gromos43a1/53a6/54a7.ff NH3+/GLY-NH3+/PRO-NH2+/COO-;
// oplsaa.ff NH3+/GLY-NH3+/PRO-NH2+ and COO-/GLY-COO-/PRO-COO-). Each patch
// adds +1 e (N-terminus) or -1 e (C-terminus) to the template. AMBER needs
// none: its rtp files have the N- and C-prefixed terminal residues.

PatchFamily patch_family(std::string_view force_f) {
  const std::string ff(force_f);
  if (ff.find("CHARMM") != std::string::npos)
    return PatchFamily::Charmm;
  if (ff.find("GROMOS") != std::string::npos)
    return ff.find("43A") != std::string::npos ? PatchFamily::Gromos43
                                               : PatchFamily::Gromos;
  if (ff.find("OPLS") != std::string::npos)
    return PatchFamily::Opls;
  return PatchFamily::None;
}

const std::vector<PatchAtom> *nterm_patch(PatchFamily family,
                                          const std::string &templ_name) {
  const bool gly = templ_name == "GLY";
  const bool pro = templ_name == "PRO";
  switch (family) {
  case PatchFamily::Charmm: {
    static const std::vector<PatchAtom> standard{
        {"N", "NH3", -0.30}, {"CA", "CT1", 0.21}, {"HA", "HB1", 0.10},
        {"H1", "HC", 0.33},  {"H2", "HC", 0.33},  {"H3", "HC", 0.33}};
    static const std::vector<PatchAtom> glycine{
        {"N", "NH3", -0.30}, {"CA", "CT2", 0.13}, {"HA1", "HB2", 0.09},
        {"HA2", "HB2", 0.09}, {"H1", "HC", 0.33}, {"H2", "HC", 0.33},
        {"H3", "HC", 0.33}};
    static const std::vector<PatchAtom> proline{
        {"N", "NP", -0.07},   {"CD", "CP3", 0.16},  {"HD1", "HA2", 0.09},
        {"HD2", "HA2", 0.09}, {"CA", "CP1", 0.16},  {"HA", "HB1", 0.09},
        {"H1", "HC", 0.24},   {"H2", "HC", 0.24}};
    return gly ? &glycine : (pro ? &proline : &standard);
  }
  case PatchFamily::Gromos:
  case PatchFamily::Gromos43: {
    static const std::vector<PatchAtom> standard{
        {"N", "NL", 0.129}, {"CA", "CH1", 0.127}, {"H1", "H", 0.248},
        {"H2", "H", 0.248}, {"H3", "H", 0.248}};
    static const std::vector<PatchAtom> glycine{
        {"N", "NL", 0.129}, {"CA", "CH2", 0.127}, {"H1", "H", 0.248},
        {"H2", "H", 0.248}, {"H3", "H", 0.248}};
    // 53A6/54A7 PRO-NH2+ spreads the charge over CA and CD; 43A1 does not.
    static const std::vector<PatchAtom> proline{
        {"N", "NT", 0.128},   {"CA", "CH1", 0.188}, {"CD", "CH2r", 0.188},
        {"H1", "H", 0.248},   {"H2", "H", 0.248}};
    static const std::vector<PatchAtom> proline43{
        {"N", "NT", 0.2}, {"H1", "H", 0.4}, {"H2", "H", 0.4}};
    if (pro)
      return family == PatchFamily::Gromos43 ? &proline43 : &proline;
    return gly ? &glycine : &standard;
  }
  case PatchFamily::Opls: {
    static const std::vector<PatchAtom> standard{
        {"N", "opls_287", -0.3}, {"CA", "opls_293B", 0.25},
        {"H1", "opls_290", 0.33}, {"H2", "opls_290", 0.33},
        {"H3", "opls_290", 0.33}};
    static const std::vector<PatchAtom> glycine{
        {"N", "opls_287", -0.3}, {"CA", "opls_292B", 0.19},
        {"H1", "opls_290", 0.33}, {"H2", "opls_290", 0.33},
        {"H3", "opls_290", 0.33}};
    static const std::vector<PatchAtom> proline{
        {"N", "opls_309", -0.2},  {"CA", "opls_246", 0.23},
        {"CD", "opls_296", 0.17}, {"H1", "opls_310", 0.31},
        {"H2", "opls_310", 0.31}};
    return gly ? &glycine : (pro ? &proline : &standard);
  }
  case PatchFamily::None:
    break;
  }
  return nullptr;
}

const std::vector<PatchAtom> *cterm_patch(PatchFamily family,
                                          const std::string &templ_name) {
  switch (family) {
  case PatchFamily::Charmm: {
    static const std::vector<PatchAtom> patch{
        {"C", "CC", 0.34}, {"O", "OC", -0.67}, {"OXT", "OC", -0.67}};
    return &patch;
  }
  case PatchFamily::Gromos:
  case PatchFamily::Gromos43: {
    static const std::vector<PatchAtom> patch{
        {"C", "C", 0.27}, {"O", "OM", -0.635}, {"OXT", "OM", -0.635}};
    return &patch;
  }
  case PatchFamily::Opls: {
    static const std::vector<PatchAtom> standard{
        {"CA", "opls_283", 0.04}, {"C", "opls_271", 0.7},
        {"O", "opls_272", -0.8},  {"OXT", "opls_272", -0.8}};
    static const std::vector<PatchAtom> glycine{
        {"CA", "opls_284", -0.02}, {"C", "opls_271", 0.7},
        {"O", "opls_272", -0.8},   {"OXT", "opls_272", -0.8}};
    static const std::vector<PatchAtom> proline{
        {"CA", "opls_285", -0.09}, {"C", "opls_271", 0.7},
        {"O", "opls_272", -0.8},   {"OXT", "opls_272", -0.8}};
    if (templ_name == "GLY")
      return &glycine;
    return templ_name == "PRO" ? &proline : &standard;
  }
  case PatchFamily::None:
    break;
  }
  return nullptr;
}

// CHARMM (HT1..3, OT1/OT2), GROMACS (O1/O2, OC1/OC2) and PDB (OXT) spellings
// of the terminal atoms, mapped onto the patch names used in the tables above.
std::string terminal_patch_key(const std::string &atom_key) {
  if (atom_key == "HT1")
    return "H1";
  if (atom_key == "HT2")
    return "H2";
  if (atom_key == "HT3")
    return "H3";
  if (atom_key == "OT1" || atom_key == "O1" || atom_key == "OC1")
    return "O";
  if (atom_key == "OT2" || atom_key == "O2" || atom_key == "OC2")
    return "OXT";
  return atom_key;
}

const PatchAtom *find_patch_atom(const std::vector<PatchAtom> *patch,
                                 const std::string &key) {
  if (patch == nullptr)
    return nullptr;
  for (const auto &atom : *patch) {
    if (key == atom.key)
      return &atom;
  }
  return nullptr;
}

namespace {

bool span_has_any_atom(const Multi_MOL &mol, const ResidueSpan &span,
                       std::initializer_list<const char *> keys) {
  for (std::size_t i = span.begin; i < span.end && i < mol.info.size(); ++i) {
    const std::string key = normalize_atom_key(mol.info[i].get_atom());
    for (const char *wanted : keys) {
      if (key == wanted)
        return true;
    }
  }
  return false;
}

} // namespace

// Evidence that the structure carries the patched terminus: the three
// (two for PRO) ammonium hydrogens, or the second carboxylate oxygen. A
// capped or fragment chain has neither and keeps the plain template.
bool span_has_terminal_amine(const Multi_MOL &mol, const ResidueSpan &span) {
  return span_has_any_atom(mol, span, {"H1", "H2", "H3", "HT1", "HT2", "HT3"});
}

bool span_has_terminal_carboxylate(const Multi_MOL &mol,
                                   const ResidueSpan &span) {
  return span_has_any_atom(mol, span, {"OXT", "OT2", "O2", "OC2"});
}

void apply_terminal_patch(Multi_MOL &mol, std::size_t atom_index,
                          const PatchAtom &patch, const para &parameter) {
  mol.info[atom_index].set_atom_type(patch.type);
  if (atom_index >= mol.parameters.size())
    mol.parameters.resize(mol.info.size());
  double mass = mol.parameters[atom_index].get_mass();
  const auto type_it = parameter.atom_type_to_id.find(patch.type);
  if (type_it != parameter.atom_type_to_id.end() &&
      type_it->second < parameter.base.size() &&
      parameter.base[type_it->second].mass > 0.01)
    mass = parameter.base[type_it->second].mass;
  if (!(mass > 0.01))
    mass = 1.0;
  // sigma/epsilon/radius follow from the patched type (fill_vdw_from_atom_type).
  mol.parameters[atom_index].set_data(patch.charge, 0.0, 0.0, 0.0, mass);
}

} // namespace TitanFfLib::detail

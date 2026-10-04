#ifndef POLYMER_CLASSIFICATION_H
#define POLYMER_CLASSIFICATION_H

#include "MOL_info.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

namespace PolymerClassification {

enum class PolymerKind { Other, Protein, Rna, Dna };
enum class NucleicAtomRegion { Unknown, Base, Sugar, Phosphate };

inline std::string uppercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

inline bool is_rna_residue_name(const std::string &res_name_upper) {
  static const std::unordered_set<std::string> kRnaNames{
      "A",   "U",   "C",   "G",   "RA",  "RU",  "RC",  "RG",
      "ADE", "URA", "CYT", "GUA", "RIA", "RCY", "RUR", "RGU"};
  return kRnaNames.contains(res_name_upper);
}

inline bool is_dna_residue_name(const std::string &res_name_upper) {
  static const std::unordered_set<std::string> kDnaNames{
      "DA", "DT", "DC", "DG", "THY"};
  return kDnaNames.contains(res_name_upper);
}

inline bool is_nucleic_residue_name(const std::string &res_name_upper) {
  return is_rna_residue_name(res_name_upper) || is_dna_residue_name(res_name_upper);
}

inline bool is_protein_residue_name(const std::string &res_name_upper) {
  static const std::unordered_set<std::string> kProteinNames{
      "ALA", "ARG",  "ASN", "ASP", "ASH", "CYS", "CYX", "GLN", "GLU",
      "GLH", "GLY",  "HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
      "HISA", "HISB", "HISD", "HISE", "HISH",
      "ILE", "LEU",  "LYS", "LYN", "MET", "PHE", "PRO", "SER", "THR",
      "TRP", "TYR",  "VAL"};
  return kProteinNames.contains(res_name_upper);
}

inline PolymerKind classify_residue_polymer(const std::string &res_name) {
  const std::string upper = uppercase_copy(res_name);
  if (is_rna_residue_name(upper))
    return PolymerKind::Rna;
  if (is_dna_residue_name(upper))
    return PolymerKind::Dna;
  if (is_protein_residue_name(upper))
    return PolymerKind::Protein;
  return PolymerKind::Other;
}

inline bool is_nucleic_type(const std::string &mol_type) {
  const std::string upper = uppercase_copy(mol_type);
  return upper.find("RNA") != std::string::npos ||
         upper.find("DNA") != std::string::npos;
}

inline bool is_purine_residue_name(const std::string &res_name_upper) {
  return res_name_upper == "A" || res_name_upper == "G" ||
         res_name_upper == "RA" || res_name_upper == "RG" ||
         res_name_upper == "ADE" || res_name_upper == "GUA" ||
         res_name_upper == "RIA" || res_name_upper == "RGU" ||
         res_name_upper == "DA" || res_name_upper == "DG";
}

inline bool is_pyrimidine_residue_name(const std::string &res_name_upper) {
  return res_name_upper == "C" || res_name_upper == "U" ||
         res_name_upper == "RC" || res_name_upper == "RU" ||
         res_name_upper == "CYT" || res_name_upper == "URA" ||
         res_name_upper == "RCY" || res_name_upper == "RUR" ||
         res_name_upper == "DC" || res_name_upper == "DT" ||
         res_name_upper == "THY";
}

inline std::string normalize_nucleic_atom_name(std::string atom_name) {
  atom_name = uppercase_copy(atom_name);
  std::replace(atom_name.begin(), atom_name.end(), '*', '\'');
  return atom_name;
}

inline NucleicAtomRegion classify_nucleic_atom_region(const std::string &atom_name) {
  const std::string atom = normalize_nucleic_atom_name(atom_name);
  if (atom == "P" || atom == "OP1" || atom == "OP2" || atom == "OP3" ||
      atom == "O1P" || atom == "O2P" || atom == "O3P")
    return NucleicAtomRegion::Phosphate;

  if (atom == "O5'" || atom == "C5'" || atom == "C4'" || atom == "O4'" ||
      atom == "C3'" || atom == "O3'" || atom == "C2'" || atom == "C1'" ||
      atom == "O2'")
    return NucleicAtomRegion::Sugar;

  if (atom.empty())
    return NucleicAtomRegion::Unknown;
  if (atom[0] == 'N' || atom[0] == 'C' || atom[0] == 'O')
    return NucleicAtomRegion::Base;
  return NucleicAtomRegion::Unknown;
}

inline bool is_phosphate_oxygen(const std::string &atom_name) {
  const std::string atom = normalize_nucleic_atom_name(atom_name);
  return atom == "OP1" || atom == "OP2" || atom == "OP3" ||
         atom == "O1P" || atom == "O2P" || atom == "O3P";
}

inline bool is_nucleobase_carbon(const std::string &atom_name) {
  const std::string atom = normalize_nucleic_atom_name(atom_name);
  if (classify_nucleic_atom_region(atom) != NucleicAtomRegion::Base)
    return false;
  if (atom.empty() || atom[0] != 'C')
    return false;
  return atom == "C2" || atom == "C4" || atom == "C5" || atom == "C6" ||
         atom == "C7" || atom == "C8";
}

inline std::string detect_nucleic_polymer_type(const std::vector<mol_info> &info) {
  std::unordered_set<unsigned long long> seen_residues;
  size_t rna_hits = 0;
  size_t dna_hits = 0;

  for (const auto &atom : info) {
    const unsigned long long residue_key =
        (static_cast<unsigned long long>(atom.get_chainnum()) << 32) |
        static_cast<unsigned long long>(atom.get_resnr());
    if (!seen_residues.insert(residue_key).second)
      continue;

    const std::string res_name = uppercase_copy(atom.get_res_name());
    if (is_rna_residue_name(res_name)) {
      ++rna_hits;
    } else if (is_dna_residue_name(res_name)) {
      ++dna_hits;
    }
  }

  if (rna_hits == 0 && dna_hits == 0)
    return {};
  return (rna_hits >= dna_hits) ? "rna" : "dna";
}

} // namespace PolymerClassification

#endif // POLYMER_CLASSIFICATION_H

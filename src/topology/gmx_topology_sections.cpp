// [ moleculetype ], [ atomtypes ] and [ atoms ] readers of a GROMACS
// topology, shared by the protein and ligand paths.
#include "Multi_MOL.h"
#include "arguments.h"
#include "functions.h"
#include "info.h"
#include "miscellaneous.h"
#include "titan_ui.h"
#include "topology/gmx_topology_match.h"
#include "topology/include_utils.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
using std::string_view;
#include <vector>

using namespace std;
using index_type = class index;


int process_moleculetype(fstream &input_topo_protein, string &lig_name,
                         string &protein_1) {
  if (protein_1.find("moleculetype") != string::npos) {
    while (getline(input_topo_protein, protein_1)) {
      if (protein_1.length() > 1 && protein_1.find("[ ") == string::npos) {
        if (protein_1[0] != ';') {
          istringstream is(protein_1);
          int nrexcl{};
          is >> lig_name >> nrexcl;
          return 0;
        } else
          continue;
      } else
        return 1;
    }
    return 0;
  } else
    return 1;
}
namespace {
// The section a "[ name ]" line opens, lower case without brackets or spaces
// ("" for any other line). Comment text after ';' is ignored.
string topology_section_header(const string &line) {
  const string text = TitanTopo::trim_copy(string_view(line).substr(0, line.find(';')));
  if (text.empty() || text.front() != '[')
    return {};
  string name;
  for (const char c : text)
    if (c != '[' && c != ']' && c != ' ' && c != '\t')
      name.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return name;
}
} // namespace

int process_atomtypes(fstream &input_topo_protein, string &protein_1,
                      para &parameters, string force_f, string &lig_name) {
  // Only a [ atomtypes ] header opens the section: a comment or an include
  // that mentions the word does not, and preprocessor lines inside it are
  // skipped (Nov 2026; they were read as rows and reported as unreadable).
  if (topology_section_header(protein_1) == "atomtypes") {
    // ENE-16 (22/10/2026). Columns are taken from the end of the row, as
    // GROMACS does: ... mass charge ptype v w, with an optional bond type
    // and/or atomic number in front (6, 7 or 8 columns; OPLS-AA rows have 8
    // and were skipped silently). v/w are C6/C12 only for GROMOS; every
    // other force field, OPLS-AA included, has sigma/epsilon and the radius
    // r_min/2 = 5 * 2^(1/6) * sigma in Angstrom, as on the lib_ff path.
    const bool c6_c12 = force_f.find("GROMOS") != string::npos;
    std::size_t rows_skipped{};
    std::string skipped_example;
    while (getline(input_topo_protein, protein_1)) {
      if (protein_1.length() <= 1 || protein_1.find(";") == 0)
        continue;
      if (protein_1.find(";") != string::npos) {
        protein_1 = protein_1.substr(0, protein_1.find(";"));
      }
      const string row = TitanTopo::trim_copy(protein_1);
      if (row.empty() || row.front() == '#')
        continue;

      if (row.front() != '[' && protein_1.length() > 10 &&
          protein_1.find("[ ") == string::npos &&
          protein_1.find(" ]") == string::npos) {
        istringstream i_temp(protein_1);
        vector<string> cols;
        for (string token; i_temp >> token;)
          cols.push_back(token);
        const size_t n = cols.size();
        const bool ptype_ok =
            n >= 6 && (cols[n - 3] == "A" || cols[n - 3] == "V" ||
                       cols[n - 3] == "D" || cols[n - 3] == "S");
        double v{}, w{}, mass_column{};
        try {
          if (!ptype_ok)
            throw std::invalid_argument("layout");
          mass_column = stod(cols[n - 5]);
          v = stod(cols[n - 2]);
          w = stod(cols[n - 1]);
        } catch (const std::exception &) {
          if (rows_skipped++ == 0)
            skipped_example = protein_1;
          continue;
        }
        const double mass = mass_column > 0.1 ? mass_column : 1.0;
        // GROMOS C12 is always below 1e-4 (aromatic HC: C6 8.5e-5, C12
        // 1.5e-8, r_min/2 = 1.33 A), so only C6 = 0 means "no LJ" there, as
        // in take_para; sigma/epsilon types keep the former test.
        double radius = 0.0;
        if (c6_c12)
          radius = v < 1e-13 ? 0.0
                             : 10 * 0.5 * pow((w * 2 * pow(v, -1.0)), (1.0 / 6.0));
        else if (std::fabs(w) <= 0.0001 && std::fabs(v) <= 0.0001)
          radius = 0.0;
        else
          radius = 5.0 * pow(2.0, 1.0 / 6.0) * v;
        parameters.base.emplace_back(v, w, mass, radius, cols[0]);
        parameters.register_atom_type(cols[0]);
      } else if (protein_1.find("moleculetype") != string::npos) {
        process_moleculetype(input_topo_protein, lig_name, protein_1);
        break;
      } else
        break;
    }
    if (rows_skipped > 0)
      TitanUI::print_warning("[ atomtypes ]: " + std::to_string(rows_skipped) +
                             " row(s) could not be read (need ... mass charge "
                             "ptype v w) and were skipped, e.g. \"" +
                             skipped_example + "\"");
    return 0;
  }
  return 1;
}
int Multi_MOL::process_atom_from_topo(fstream &input_topo_protein,
                                      string &protein_1, string &lig_name,
                                      int &i,
                                      vector<topo_info_temp> &info_topo) {
  if (protein_1.find("[ atoms ]") != string::npos) {
    while (getline(input_topo_protein, protein_1)) {
      string line{protein_1};
      const size_t first_non_ws{line.find_first_not_of(" \t\r")};
      if (first_non_ws == string::npos)
        continue;

      line.erase(0, first_non_ws);

      if (line[0] == '[')
        break;

      if (line[0] == ';' || line[0] == '#')
        continue;

      if (const size_t comment_pos = line.find(';'); comment_pos != string::npos)
        line = line.substr(0, comment_pos);

      if (line.find_first_not_of(" \t\r") == string::npos)
        continue;

      // Every well-formed row inside [ atoms ] belongs to the active
      // moleculetype. Matching against lig_name breaks RNA/DNA because those
      // rows contain residue names (GUA/URA/...) instead of the parent name.
      istringstream topo_is(line);
      int i_temp{}, chrg{}, resnr{};
      string a_type{}, a_name{}, res_name{};
      double charge{}, mass{};
      if (!(topo_is >> i_temp >> a_type >> resnr >> res_name >> a_name >>
            chrg >> charge >> mass))
        continue;

      info_topo.emplace_back(a_name, a_type, res_name, charge, mass, resnr);
      ++i;
    }
  } else
    return 1;
  return 0;
}

int Multi_MOL::post_process_atom_from_topo(vector<topo_info_temp> &info_topo,
                                           bool stop_on_name_mismatch) {
  if (info_topo.empty()) {
    std::cerr << "[INFO] Standard topology parser: 0 atoms parsed from "
              << "topo file. If using MARTINI, this is expected.\n";
    return 1;
  }

  if (this->parameters.size() < info.size()) {
    this->parameters.resize(info.size());
  }

  // ASSIGN INTERNAL RESNUM to PDB atoms to allow unique identification
  // This separates "Display ResNr" (which can be duplicate/confusing) from
  // "Calculation ID". We do this purely based on PDB connectivity/changes.
  int internal_cnt = 0;
  if (!info.empty()) {
    info[0].set_internal_resnr(0);
  }
  for (size_t k = 1; k < info.size(); ++k) {
    if (info[k].get_resnr() != info[k - 1].get_resnr() ||
        info[k].get_chain() != info[k - 1].get_chain()) {
      internal_cnt++;
    }
    info[k].set_internal_resnr(internal_cnt);
  }

  // AUTO-SUBSET FOR COMMON TOPOLOGY SYSTEMS
  // If the parsed topology is larger than the extracted PDB group, we use the
  // original PDB atom IDs to pluck the exact corresponding topology atom,
  // bypassing string-matching.
  if (info_topo.size() > info.size() && !info.empty()) {
    const bool ids_were_set =
        std::all_of(info.begin(), info.end(), [](const mol_info &atom_info) {
          return atom_info.has_original_atom_id();
        });

    if (ids_were_set) {
      bool can_subset = true;
      for (size_t i = 0; i < info.size(); ++i) {
        if (info[i].get_original_atom_id() >= info_topo.size()) {
          can_subset = false;
          break;
        }
      }
      if (can_subset) {
        vector<topo_info_temp> subset_topo;
        subset_topo.reserve(info.size());
        for (size_t i = 0; i < info.size(); ++i) {
          subset_topo.push_back(info_topo[info[i].get_original_atom_id()]);
        }
        info_topo =
            std::move(subset_topo); // Now info.size() == info_topo.size()
      }
    }
  }

  // 1-1 MAPPING LOGIC (Common Topo)
  // ENE-12 (22/10/2026): the positional assignment is checked by name; the
  // report used to be commented out, so a structure in another atom order
  // silently took the charges of other atoms (topology/gmx_topology_match.h).
  if (info.size() == info_topo.size()) {
    const TitanTopo::NameCheck check =
        TitanTopo::compare_positional_names(info, info_topo);
    if (check.atom_mismatches > 0 || check.residue_mismatches > 0) {
      std::string message = "topology/structure names differ: atom names at " +
                            std::to_string(check.atom_mismatches) + " and residue names at " +
                            std::to_string(check.residue_mismatches) + " of " +
                            std::to_string(check.atoms) +
                            " atoms (topology parameters are assigned by position)";
      for (const auto &example : check.examples)
        message += "\n    " + example;
      if (!stop_on_name_mismatch &&
          static_cast<double>(check.atom_mismatches) >
              TitanTopo::kMaxNameMismatchFraction * static_cast<double>(check.atoms)) {
        TitanUI::print_warning(message +
                               "\n  MARTINI: these positional values are replaced by the MARTINI "
                               "topology reader (parse_martini_topology).");
      } else if (static_cast<double>(check.atom_mismatches) >
          TitanTopo::kMaxNameMismatchFraction * static_cast<double>(check.atoms)) {
        TitanUI::print_error(message +
                             "\n  More than 5% of the atom names differ: the structure is not in "
                             "the atom order of the topology. Write the structure from the run "
                             "input (e.g. gmx editconf -f topol.tpr) or select matching groups.");
        return 2;
      }
      TitanUI::print_warning(message);
    }
    for (size_t i = 0; i < info.size(); ++i) {
      info[i].set_atom_type(info_topo[i].atom_type);
      this->parameters[i].set_charge(info_topo[i].atom_charge);
      this->parameters[i].set_mass(info_topo[i].atom_mass);
      // Do NOT overwrite info[i].resnr - keep PDB numbering for output.
    }
  } else {
    // Different atom counts: match by residue and atom name.
    const std::vector<long long> match = TitanTopo::match_by_names(info, info_topo);
    std::size_t unmatched = 0;
    std::string examples;
    for (size_t i{}; i < info.size(); ++i) {
      if (match[i] >= 0) {
        const auto &topo_atom = info_topo[static_cast<size_t>(match[i])];
        info[i].set_atom_type(topo_atom.atom_type);
        this->parameters[i].set_charge(topo_atom.atom_charge);
        this->parameters[i].set_mass(topo_atom.atom_mass);
      } else {
        this->parameters[i].set_charge(0.0); // Reset to 0
        if (unmatched < 10)
          examples += "\n    " + info[i].get_res_name() + std::to_string(info[i].get_resnr()) +
                      "/" + info[i].get_atom();
        ++unmatched;
      }
    }
    if (unmatched > 0) {
      const std::string message =
          std::to_string(unmatched) + " of " + std::to_string(info.size()) +
          " structure atoms have no topology atom of the same residue and atom name "
          "(structure " + std::to_string(info.size()) + " atoms, topology " +
          std::to_string(info_topo.size()) + "); they keep charge 0:" + examples;
      if (static_cast<double>(unmatched) >
          TitanTopo::kMaxNameMismatchFraction * static_cast<double>(info.size())) {
        TitanUI::print_error(message + "\n  More than 5% of the atoms are unmatched.");
        return 2;
      }
      TitanUI::print_warning(message);
    }
  }

  return 0;
}

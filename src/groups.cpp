#include "groups.h"
#include "Multi_MOL.h"
#include "titan_charconv_compat.h"
#include "titan_ui.h"
#include "uti_func.h"
#include <algorithm>
#include <cctype>
#include <expected>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

using namespace std;

std::optional<size_t> groups_cal::find_group_index(std::string_view name) const {
  std::string wanted(name);
  std::transform(wanted.begin(), wanted.end(), wanted.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  for (size_t i = 0; i < groups.size(); ++i) {
    std::string candidate = groups[i].name;
    std::transform(candidate.begin(), candidate.end(), candidate.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    if (candidate == wanted)
      return i;
  }
  return std::nullopt;
}

namespace {

std::string trim_copy(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0)
    text.remove_prefix(1);
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0)
    text.remove_suffix(1);
  return std::string(text);
}

std::string uppercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

bool in_list(std::string_view value,
             std::initializer_list<std::string_view> values) {
  for (std::string_view candidate : values)
    if (value == candidate)
      return true;
  return false;
}

std::string normalize_group_type(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });

  if (value == "ion" || value == "ions")
    return "ions";
  if (value == "lig" || value == "ligand")
    return "ligand";
  if (value == "protein" || value == "rna" || value == "dna" ||
      value == "water" || value == "lipid")
    return value;
  return {};
}

std::string sanitize_group_suffix(std::string value, std::size_t fallback_id) {
  value = trim_copy(std::string_view{value.data(), value.size()});
  if (value.empty() || value == "NULL" || value == "NULL_S")
    value = std::to_string(fallback_id);

  for (char &c : value) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) == 0 && c != '_' && c != '-' && c != '.')
      c = '_';
  }
  return value.empty() ? std::to_string(fallback_id) : value;
}

std::expected<std::vector<group_cal>, std::string>
parse_ndx_fast(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open())
    return std::unexpected("index file is not found: " + filename);

  input.seekg(0, std::ios::end);
  const std::streamoff file_size = input.tellg();
  if (file_size < 0)
    return std::unexpected("cannot determine index file size: " + filename);
  input.seekg(0, std::ios::beg);

  std::string buffer(static_cast<std::size_t>(file_size), '\0');
  if (!buffer.empty())
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  if (!input && input.gcount() != static_cast<std::streamsize>(buffer.size()))
    return std::unexpected("cannot read complete index file: " + filename);

  std::vector<group_cal> parsed_groups;
  parsed_groups.reserve(64);

  group_cal *current_group = nullptr;
  const char *const data = buffer.data();
  const char *const end = data + buffer.size();
  const char *line_begin = data;
  std::size_t line_number = 1;

  auto parse_line = [&](const char *begin,
                        const char *line_end) -> std::expected<void, std::string> {
    const char *logical_end = line_end;
    for (const char *p = begin; p < line_end; ++p) {
      if (*p == ';') {
        logical_end = p;
        break;
      }
    }

    while (begin < logical_end &&
           std::isspace(static_cast<unsigned char>(*begin)) != 0)
      ++begin;
    while (logical_end > begin &&
           std::isspace(static_cast<unsigned char>(*(logical_end - 1))) != 0)
      --logical_end;
    if (begin == logical_end)
      return {};

    if (*begin == '[') {
      const char *close = std::find(begin + 1, logical_end, ']');
      if (close == logical_end) {
        return std::unexpected("invalid NDX header at line " +
                               std::to_string(line_number));
      }

      const std::string name =
          trim_copy(std::string_view(begin + 1,
                                     static_cast<std::size_t>(close - begin -
                                                              1)));
      if (name.empty()) {
        return std::unexpected("empty NDX group name at line " +
                               std::to_string(line_number));
      }

      parsed_groups.emplace_back(name);
      current_group = &parsed_groups.back();
      current_group->atom_index.reserve(4096);
      return {};
    }

    if (current_group == nullptr) {
      return std::unexpected("atom indices appear before the first NDX group at line " +
                             std::to_string(line_number));
    }

    const char *p = begin;
    while (p < logical_end) {
      while (p < logical_end &&
             std::isspace(static_cast<unsigned char>(*p)) != 0)
        ++p;
      if (p >= logical_end)
        break;

      std::size_t atom{};
      const auto parsed = titan::from_chars_compat(p, logical_end, atom);
      if (parsed.ec != std::errc{} || parsed.ptr == p || atom == 0) {
        return std::unexpected("invalid atom index at line " +
                               std::to_string(line_number));
      }
      if (parsed.ptr < logical_end &&
          std::isspace(static_cast<unsigned char>(*parsed.ptr)) == 0) {
        return std::unexpected("invalid atom-index token at line " +
                               std::to_string(line_number));
      }

      current_group->atom_index.emplace_back(atom);
      p = parsed.ptr;
    }

    return {};
  };

  for (const char *p = data; p < end; ++p) {
    if (*p != '\n')
      continue;
    auto parsed = parse_line(line_begin, p);
    if (!parsed)
      return std::unexpected(parsed.error());
    line_begin = p + 1;
    ++line_number;
  }

  if (line_begin < end) {
    auto parsed = parse_line(line_begin, end);
    if (!parsed)
      return std::unexpected(parsed.error());
  }

  if (parsed_groups.empty())
    return std::unexpected("index file contains no groups: " + filename);

  for (const group_cal &group : parsed_groups) {
    if (group.atom_index.empty()) {
      TitanUI::print_warning("Index group \"" + group.name +
                             "\" is empty.");
    }
  }

  return parsed_groups;
}

bool residue_has_o2_prime(const Multi_MOL &mol_in, std::size_t begin_atom,
                          std::size_t end_atom_exclusive) {
  if (begin_atom == 0 || end_atom_exclusive <= begin_atom)
    return false;
  const std::size_t begin = begin_atom - 1;
  const std::size_t end_atom =
      std::min(end_atom_exclusive - 1, mol_in.info.size());
  for (std::size_t atom = begin; atom < end_atom; ++atom) {
    if (mol_in.info[atom].get_atom() == "O2'" ||
        mol_in.info[atom].get_original_res_name() == "U")
      return true;
  }
  return false;
}

std::string classify_residue_type(const Multi_MOL &mol_in,
                                  const amino_acid &amine,
                                  std::size_t chain_index,
                                  std::size_t residue_index,
                                  std::size_t begin_atom,
                                  std::size_t end_atom_exclusive) {
  const auto &idx = mol_in.Index[chain_index];

  if (begin_atom == 0 || begin_atom - 1 >= mol_in.info.size()) {
    const std::string chain_type = normalize_group_type(idx.get_type());
    return chain_type.empty() ? "ligand" : chain_type;
  }

  std::string res_name = mol_in.info[begin_atom - 1].get_original_res_name();
  if (res_name.empty() || res_name == "UND")
    res_name = mol_in.info[begin_atom - 1].get_res_name();
  res_name = uppercase_copy(res_name);

  if (res_name.size() > 3 &&
      (res_name.front() == 'N' || res_name.front() == 'C'))
    res_name = res_name.substr(1);

  const std::string_view res_view{res_name.data(), res_name.size()};

  if (in_list(res_view, {"SOL", "WAT", "HOH", "TIP3", "TIP3P", "H2O",
                         "OH2"}))
    return "water";

  if (in_list(res_view, {"AG", "AG+", "AL", "AL3", "AL3+", "BA", "BA2",
                         "BA2+", "BR", "BR-", "CA", "CAL", "CA2", "CA2+",
                         "CD", "CD2", "CD2+", "CL", "CL-", "CLA", "CS",
                         "CS+", "CU", "CU+", "CU1", "CU1+", "CU2", "CU2+",
                         "F", "F-", "FE", "FE2", "FE2+", "FE3", "FE3+",
                         "HG", "HG2", "HG2+", "I", "I-", "IOD", "IOD-",
                         "K", "K+", "LI", "LI+", "MG", "MG2", "MG2+",
                         "MN", "MN2", "MN2+", "NA", "NA+", "POT", "RB",
                         "RB+", "SOD", "SR", "SR2", "SR2+", "ZN", "ZN2",
                         "ZN2+"}))
    return "ions";

  if (in_list(res_view, {"U", "RU", "URA"}))
    return "rna";
  if (in_list(res_view, {"DA", "DC", "DG", "DT", "THY"}))
    return "dna";
  if (in_list(res_view, {"RA", "RC", "RG", "ADE", "CYT", "GUA"}))
    return residue_has_o2_prime(mol_in, begin_atom, end_atom_exclusive) ? "rna"
                                                                        : "dna";
  if (in_list(res_view, {"A", "C", "G", "T"}))
    return residue_has_o2_prime(mol_in, begin_atom, end_atom_exclusive) ? "rna"
                                                                        : "dna";

  if (in_list(res_view, {"DOPA", "DOPC", "DOPE", "DOPG", "DOPS", "DMPA",
                         "DMPC", "DMPG", "DMPI", "DMPR", "DMPS", "DMPU",
                         "DDPC", "DPPA", "DPPC", "DPPE", "DPPG", "DPPS",
                         "DPRO", "CER1", "BGAL", "BGAM", "BGAN", "BGLC",
                         "ANE5", "POPC", "CHL1", "PSM"}))
    return "lipid";

  const std::string type_base =
      normalize_group_type(idx.get_type_base(residue_index));
  if (!type_base.empty())
    return type_base;

  if (amine.check_res_name(res_name) ||
      in_list(res_view, {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
                         "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU",
                         "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
                         "TYR", "VAL"}))
    return "protein";

  const std::string chain_type = normalize_group_type(idx.get_type());
  if (!chain_type.empty() && chain_type != "protein")
    return chain_type;

  return "ligand";
}

class AutoIndexBuilder {
  std::vector<group_cal> groups_;
  std::unordered_map<std::string, std::size_t> lookup_;

public:
  void add_atom(const std::string &name, std::size_t atom) {
    auto it = lookup_.find(name);
    if (it == lookup_.end()) {
      const std::size_t group_index = groups_.size();
      groups_.emplace_back(name);
      lookup_.emplace(name, group_index);
      it = lookup_.find(name);
    }
    groups_[it->second].atom_index.emplace_back(atom);
  }

  std::vector<group_cal> release() { return std::move(groups_); }
};

int write_index_file(const std::string &filename,
                     const std::vector<group_cal> &groups) {
  std::ofstream out_index_file(filename, std::ios::out | std::ios::trunc);
  if (!out_index_file.is_open()) {
    TitanUI::print_error("Cannot write index file: " + filename);
    return 1;
  }

  for (const group_cal &group : groups) {
    out_index_file << "[ " << group.name << " ]\n";
    for (std::size_t j{}; j < group.atom_index.size(); ++j) {
      out_index_file << std::setw(8) << group.atom_index[j];
      if ((j + 1) % 15 == 0)
        out_index_file << '\n';
    }
    out_index_file << "\n";
  }

  return 0;
}

} // namespace

int s_cal_energy(cache_sigma_epsilon &CachedData, const Multi_MOL &rec,
                 const Multi_MOL &lg, size_t begin_mol_1, size_t end_mol_1,
                 size_t begin_mol_2, size_t end_mol_2, size_t num_snap,
                 double &E_E, double &E_V);
int groups_cal::index(arguments &Arguments) {
  if (Arguments.flag_index == false)
    return 1;

  auto parsed_groups = parse_ndx_fast(Arguments.index_name);
  if (!parsed_groups) {
    TitanUI::print_warning(parsed_groups.error());
    return 1;
  }
  groups = std::move(parsed_groups.value());

  return finalize_index_groups(Arguments);
}

int groups_cal::finalize_index_groups(arguments &Arguments) {
  if (groups.empty()) {
    TitanUI::print_error("No groups are available in index data.");
    return 1;
  }

  if (Arguments.group_1 == 0 && Arguments.group_2 == 0 &&
      Arguments.flag_nocal == false)
    this->select_groups_to_cal_energy();
  else {
    this->group_1 = Arguments.group_1;
    this->group_2 = Arguments.group_2;

    if (this->group_1 >= this->group_num() ||
        this->group_2 >= this->group_num())
      this->select_groups_to_cal_energy();
  }
  this->groups[group_1].check_conti();
  this->groups[group_2].check_conti();
  if (Arguments.gist.flag_gist) {
    if (Arguments.n_wat < 0) {
      std::optional<size_t> candidate;
      for (size_t i=0;i<groups.size();++i) {
        const auto name=uppercase_copy(trim_copy(groups[i].name));
        if (name!="WATER" && name!="SOL" && name!="WAT" && name!="HOH") continue;
        if (candidate && groups[*candidate].atom_index!=groups[i].atom_index) {
          TitanUI::print_error("GIST water group is ambiguous; set -n_wat to one complete ordered O,H,H water group index.");
          return 1;
        }
        candidate=i;
      }
      if (!candidate) {
        TitanUI::print_error("GIST requires -n_wat <water group index> or a unique Water/SOL/WAT/HOH group.");
        return 1;
      }
      Arguments.n_wat=static_cast<int>(*candidate);
      TitanUI::print_info("[GIST] Selected water index group " + std::to_string(*candidate) + " (" + groups[*candidate].name + ") before coordinate loading.");
    }
    if (static_cast<size_t>(Arguments.n_wat)>=groups.size() ||
        groups[Arguments.n_wat].atom_index.empty() || groups[Arguments.n_wat].atom_index.size()%3) {
      TitanUI::print_error("GIST water group must contain complete ordered O,H,H triplets.");
      return 1;
    }
    std::set<size_t> solute(groups[group_1].atom_index.begin(),groups[group_1].atom_index.end());
    solute.insert(groups[group_2].atom_index.begin(),groups[group_2].atom_index.end());
    std::set<size_t> water;
    for (size_t atom:groups[Arguments.n_wat].atom_index) {
      if (!atom || solute.contains(atom) || !water.insert(atom).second) {
        TitanUI::print_error("GIST water group overlaps solute or contains duplicate/invalid atom indices.");
        return 1;
      }
    }
  }
  if (Arguments.entropy_quasi == true) {
    if (Arguments.group_entropy_com != -1) {
      this->group_entropy_com = Arguments.group_entropy_com;
      this->group_entropy_rec = Arguments.group_entropy_rec;
      this->group_entropy_lg = Arguments.group_entropy_lg;
    } else
      this->select_group_to_cal_entropy();
    this->groups[group_entropy_com].check_conti();
    this->groups[group_entropy_rec].check_conti();
    this->groups[group_entropy_lg].check_conti();
  }
  Arguments.flag_index = true;
  parse_index_flag = true;
  add_sub_groups(Arguments);

  this->groups[group_1].gen_setgroup();
  this->groups[group_2].gen_setgroup();
  if (add_group_pathway(Arguments) != 0)
    return 1;

  return 0;
}
void groups_cal::print() {
  fstream out("out_index_groups.dat", ios::out);
  for (size_t i = 0; i < groups.size(); ++i) {
    out << "[ " << groups[i].name << " ]\n";
    for (size_t j = 0; j < groups[i].atom_index.size(); ++j)
      out << groups[i].atom_index[j] << "  ";
    out << '\n';
  }
}
/*int groups_cal::calculate_energy_for_2_groups(const Multi_MOL &rec, const
Multi_MOL &lg, cache_sigma_epsilon& CachedData, int begin, int end, size_t
num_snap, int group_1, int group_2)
{
        double E_E = 0.0, E_V = 0.0;
        size_t begin_mol_1{}, begin_mol_2{}, end_mol_1{}, end_mol_2{};
        if
(this->groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1]
<= rec.m_mol[num_snap].size())
        {
                begin_mol_1 = groups[group_1].atom_index[0];
                end_mol_1 =
groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1];
        }
        if (this->groups[group_1].atom_index[0]> rec.m_mol[num_snap].size()
                &&
this->groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1] <=
(rec.m_mol[num_snap].size() + lg.m_mol[num_snap].size()))
        {
                begin_mol_2 = groups[group_1].atom_index[0];
                end_mol_2 =
groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1];
        }
        if
(this->groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1]
<= (rec.m_mol[num_snap].size() + lg.m_mol[num_snap].size())
                && this->groups[group_2].atom_index[0] >
rec.m_mol[num_snap].size())
        {
                begin_mol_2 = groups[group_2].atom_index[0];
                end_mol_2 =
groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1];
        }
        if
(this->groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1]
<= rec.m_mol[num_snap].size())
        {
                begin_mol_1 = groups[group_2].atom_index[0];
                end_mol_1 =
groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1];
        }
        if
(this->groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1]
<= rec.m_mol[num_snap].size()
                &&
this->groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1] <=
rec.m_mol[num_snap].size())
        {
                if
(groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1]<
groups[group_2].atom_index[0]) { begin_mol_1 = groups[group_1].atom_index[0];
                        end_mol_1 =
groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1];
                        begin_mol_2 = groups[group_2].atom_index[0];
                        end_mol_2 =
groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1];
                }
                else if (groups[group_1].atom_index[0] >
groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1])
                {
                        begin_mol_2 = groups[group_1].atom_index[0];
                        end_mol_2 =
groups[group_1].atom_index[this->groups[group_1].atom_index.size() - 1];
                        begin_mol_1 = groups[group_2].atom_index[0];
                        end_mol_1 =
groups[group_2].atom_index[this->groups[group_2].atom_index.size() - 1];
                }
                else
                        return 1;
        }
        if (end_mol_1 == 1 || end_mol_2 == 1)
                return 1;
        s_cal_energy(CachedData, rec, lg, begin_mol_1, end_mol_1, begin_mol_2,
end_mol_2, num_snap, E_E, E_V); this->E_elec.emplace_back(E_E);
        this->E_vdw.emplace_back(E_V);
        this->time_of_snap.emplace_back(rec.m_mol[num_snap].get_time());
        return 0;
}*/
int groups_cal::print_result_cal_to_file() {
  string output_name =
      "energy_groups_" + to_string(group_1) + "_" + to_string(group_2) + ".dat";
  fstream output_file(output_name, ios::out);
  output_file
      << "@    title \"Energy interaction between groups " << group_1 << " and "
      << group_2 << '\n'
      << "@    xaxis label \"Time (ps)\"\n"
      << "@    yaxis label \"Energy (kJ/mol)\""
      << "@ s0 legend \"Time\"\n"
      << "@ s1 legend \"electrostatic\"\n"
      << "@ s2 legend \"van der Waal\"\n"
      << "@   ps  Elec_e(kJ / mol)   VdW_e(kJ / mol) Total_e(kJ / mol)\n";
  for (size_t i = 0; i < E_elec.size(); ++i) {
    output_file << setw(6) << time_of_snap[i] << setw(17) << E_elec[i]
                << setw(17) << E_vdw[i] << setw(17) << E_elec[i] + E_vdw[i]
                << '\n';
  }
  return 0;
}
int groups_cal::select_groups_to_cal_energy() {
  cout << "Please choose 2 groups to calculate interaction energy:\n";
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < groups.size(); ++j)
      cout << setw(4) << j << setw(25) << groups[j].name << ": " << setw(7)
           << groups[j].atom_index.size() << " atoms\n";
    if (i == 0)
      cin >> group_1;
    else
      cin >> group_2;
  }
  return 0;
}
int groups_cal::select_group_to_cal_entropy() {
  cout << "Please choose 3 groups to calculate entropy:\n";
  bool test{true};
  int count{0}, in{};
  do {
    for (int j = 0; j < groups.size(); ++j)
      cout << setw(4) << j << setw(25) << groups[j].name << ": " << setw(7)
           << groups[j].atom_index.size() << " atoms\n";
    cin >> in;
    if (in < groups.size() && in > 0) {
      ++count;
      if (count == 1)
        group_entropy_com = in;
      else if (count == 2)
        group_entropy_rec = in;
      else
        group_entropy_lg = in;
    } else
      cout << "Your group is not in index file!\nPlease choose again!\n";
    if (count == 3)
      test = false;
  } while (test);
  return 0;
}

int s_cal_energy(cache_sigma_epsilon &CachedData, const Multi_MOL &rec,
                 const Multi_MOL &lg, size_t begin_mol_1, size_t end_mol_1,
                 size_t begin_mol_2, size_t end_mol_2, size_t num_snap,
                 double &E_E, double &E_V) {
  for (size_t j = begin_mol_1 - 1; j < end_mol_1 - 1; ++j)
    for (size_t i = begin_mol_2 - 1; i < end_mol_2 - 1; ++i) {
      double d{0.1 * sqrt(square(rec.m_mol[num_snap].pos.x[i] -
                                 lg.m_mol[num_snap].pos.x[j]) +
                          square(rec.m_mol[num_snap].pos.y[i] -
                                 lg.m_mol[num_snap].pos.y[j]) +
                          square(rec.m_mol[num_snap].pos.z[i] -
                                 lg.m_mol[num_snap].pos.z[j]))};
      if (rec.m_mol[num_snap].force_f.find("GROMOS") != string::npos) {
        CachedData.elec_ener[rec.m_mol[0].size() * j + i] =
            CachedData.charge[rec.m_mol[0].size() * j + i] / d;
        CachedData.vdw_ener[rec.m_mol[0].size() * j + i] =
            sqrt(rec.parameters[i].get_epsilon() *
                 lg.parameters[j].get_epsilon()) /
                c12(d) -
            sqrt(rec.parameters[i].get_sigma() * lg.parameters[j].get_sigma()) /
                c6(d);
      } else {
        CachedData.elec_ener[rec.m_mol[0].size() * j + i] =
            CachedData.charge[rec.m_mol[0].size() * j + i] / d;
        CachedData.vdw_ener[rec.m_mol[0].size() * j + i] =
            4.0 *
            sqrt(rec.parameters[i].get_epsilon() *
                 lg.parameters[j].get_epsilon()) *
            (c12(0.5 *
                 (rec.parameters[i].get_sigma() +
                  lg.parameters[j].get_sigma()) /
                 d) -
             c6(0.5 *
                (rec.parameters[i].get_sigma() + lg.parameters[j].get_sigma()) /
                d));
      }
      E_V += CachedData.vdw_ener[rec.m_mol[0].size() * j + i];
      E_E += CachedData.elec_ener[rec.m_mol[0].size() * j + i];
    }
  return 0;
}
int group_cal::check_conti() {
  for (size_t i{1}; i < atom_index.size(); ++i) {
    if (atom_index[i] != atom_index[i - 1] + 1) {
      this->is_continuous = false;
      return 1;
    }
  }
  this->is_continuous = true; // to clarify
  return 0;
}
bool group_cal::check_in(size_t ind_check) {
  if (group_set.empty()) {
    for (auto &atom : atom_index)
      if (atom == ind_check)
        return true;
    return false;
  } else {
    if (this->group_set.find(static_cast<size_t>(ind_check)) != group_set.end())
      return true;
    else
      return false;
  }
}
int groups_cal::create_group_entropy(const Multi_MOL &mol1,
                                     const Multi_MOL &mol2,
                                     const arguments &Arguments,
                                     string name_group) {
  group_cal group_temp{name_group}, group_temp_rec{"rec_" + name_group},
      group_temp_lg{"lg_" + name_group};
  vector<string> atoms_of_groups{};
  if (name_group == "calpha")
    atoms_of_groups.emplace_back("CA");
  if (name_group == "backbone") {
    atoms_of_groups.emplace_back("CA");
    atoms_of_groups.emplace_back("C");
    atoms_of_groups.emplace_back("N");
    atoms_of_groups.emplace_back("O");
  }
  for (size_t i{}; i < mol1.m_mol[0].size(); ++i) {
    if (Arguments.topo_name_2 == "no") {
      for (auto &atom : atoms_of_groups)
        if (mol1.info[i].get_atom() == atom) {
          group_temp.atom_index.emplace_back(i + 1);
          group_temp_rec.atom_index.emplace_back(i + 1);
        }
    } else {
      for (auto &atom : atoms_of_groups) {
        group_temp.atom_index.emplace_back(i + 1);
        group_temp_rec.atom_index.emplace_back(i + 1);
      }
    }
  }
  fstream test_topo(Arguments.topo_name, ios::in);
  bool test{test_topo.is_open()};
  test_topo.close();
  for (size_t i{}; i < mol2.m_mol[0].size(); ++i) {
    if (test == false) {
      for (auto &atom : atoms_of_groups)
        if (mol2.info[i].get_atom() == atom) {
          group_temp.atom_index.emplace_back(i + 1 + mol1.m_mol[0].size());
          group_temp_lg.atom_index.emplace_back(i + 1);
        }
    } else {
      for (auto &atom : atoms_of_groups) {
        group_temp.atom_index.emplace_back(i + 1 + mol1.m_mol[0].size());
        group_temp_lg.atom_index.emplace_back(i + 1);
      }
    }
  }
  group_temp.check_conti();
  group_temp_rec.check_conti();
  group_temp_lg.check_conti();
  groups.emplace_back(group_temp);
  groups.emplace_back(group_temp_rec);
  groups.emplace_back(group_temp_lg);
  group_entropy_com = groups.size() - 3;
  group_entropy_rec = groups.size() - 2;
  group_entropy_lg = groups.size() - 1;
  return 0;
}

groups_cal::groups_cal(const vector<class index> &Index)
    : entropy{false}, group_1{}, group_2{}, group_entropy_com{},
      group_entropy_rec{}, group_entropy_lg{} {}

int groups_cal::create_auto_index_file(arguments &Arguments, Multi_MOL &mol_in,
                                       const amino_acid &amine) {
  if (mol_in.Index.size() < 2 || mol_in.info.empty()) {
    TitanUI::print_error("Cannot auto-create index: reference topology is empty.");
    return 1;
  }

  const std::size_t atom_count =
      !mol_in.m_mol.empty() ? mol_in.m_mol[0].size() : mol_in.info.size();
  if (atom_count == 0) {
    TitanUI::print_error("Cannot auto-create index: reference has no atoms.");
    return 1;
  }

  AutoIndexBuilder builder;
  for (std::size_t atom = 1; atom <= atom_count; ++atom)
    builder.add_atom("System", atom);

  std::vector<std::string> chain_group_order;
  std::vector<std::string> type_chain_group_order;
  std::unordered_map<std::string, bool> seen_chain_group;
  std::unordered_map<std::string, bool> seen_type_chain_group;

  for (std::size_t chain_idx{}; chain_idx + 1 < mol_in.Index.size();
       ++chain_idx) {
    const auto &chain_index = mol_in.Index[chain_idx];
    const std::string chain_suffix =
        sanitize_group_suffix(chain_index.get_chain(), chain_idx + 1);
    const std::string chain_group = "chain_" + chain_suffix;
    if (!seen_chain_group[chain_group]) {
      seen_chain_group[chain_group] = true;
      chain_group_order.emplace_back(chain_group);
    }

    for (std::size_t res_idx{}; res_idx < chain_index.get_size_v();
         ++res_idx) {
      const std::size_t begin_atom = chain_index.get_res_pos_e(res_idx);
      std::size_t end_atom_exclusive =
          (res_idx + 1 < chain_index.get_size_v())
              ? chain_index.get_res_pos_e(res_idx + 1)
              : mol_in.Index[chain_idx + 1].get_res_pos_e(0);

      if (begin_atom == 0 || begin_atom > atom_count)
        continue;
      end_atom_exclusive =
          std::min(end_atom_exclusive, atom_count + static_cast<std::size_t>(1));
      if (end_atom_exclusive <= begin_atom)
        end_atom_exclusive = begin_atom + 1;

      const std::string type_group =
          classify_residue_type(mol_in, amine, chain_idx, res_idx, begin_atom,
                                end_atom_exclusive);
      const std::string type_chain_group = type_group + "_chain_" + chain_suffix;
      if (!seen_type_chain_group[type_chain_group]) {
        seen_type_chain_group[type_chain_group] = true;
        type_chain_group_order.emplace_back(type_chain_group);
      }

      for (std::size_t atom = begin_atom; atom < end_atom_exclusive; ++atom) {
        builder.add_atom(type_group, atom);
        builder.add_atom(chain_group, atom);
        builder.add_atom(type_chain_group, atom);
      }
    }
  }

  std::vector<group_cal> built_groups = builder.release();
  std::vector<group_cal> ordered_groups;
  ordered_groups.reserve(built_groups.size());
  std::unordered_map<std::string, bool> emitted;

  auto append_if_present = [&](const std::string &name) {
    if (emitted[name])
      return;
    auto it = std::find_if(
        built_groups.begin(), built_groups.end(),
        [&](const group_cal &group) { return group.name == name; });
    if (it == built_groups.end() || it->atom_index.empty())
      return;
    emitted[name] = true;
    ordered_groups.emplace_back(std::move(*it));
  };

  append_if_present("System");
  for (const std::string &name :
       {"protein", "rna", "dna", "ligand", "water", "ions", "lipid"})
    append_if_present(name);
  for (const std::string &name : chain_group_order)
    append_if_present(name);
  for (const std::string &name : type_chain_group_order)
    append_if_present(name);
  for (group_cal &group : built_groups)
    append_if_present(group.name);

  groups = std::move(ordered_groups);
  for (group_cal &group : groups)
    group.check_conti();

  if (write_index_file(Arguments.index_name, groups) != 0)
    return 1;

  Arguments.flag_index = true;
  TitanUI::print_info("Auto-created index file: " + Arguments.index_name);
  return 0;
}

int groups_cal::creat_index_file(arguments &Arguments, Multi_MOL &mol_in) {
  amino_acid empty_amine;
  return create_auto_index_file(Arguments, mol_in, empty_amine);
}

int groups_cal::add_sub_group_1(arguments &Arguments) {
  for (size_t i{}; i < Arguments.sub_group_1.size(); ++i) {
    sub_group_1.emplace_back(Arguments.sub_group_1[i], 0);
  }
  return 0;
}

int groups_cal::add_sub_group_2(arguments &Arguments) {
  for (size_t i{}; i < Arguments.sub_group_2.size(); ++i) {
    sub_group_2.emplace_back(Arguments.sub_group_2[i], 0);
  }
  return 0;
}

int groups_cal::add_sub_groups(arguments &Arguments) {
  if (Arguments.flag_sub_groups == false)
    return 1;
  add_sub_group_1(Arguments);
  add_sub_group_2(Arguments);
  check_sub_group();
  parse_subgroups();

  // Validate sub-group atoms fall within their parent group ranges
  auto validate = [&](vector<pair<int, size_t>> &subs, size_t parent_group,
                       const string &label) {
    if (groups[parent_group].atom_index.empty())
      return;
    size_t parent_lo = groups[parent_group].atom_index.front();
    size_t parent_hi = groups[parent_group].atom_index.back();
    for (size_t i{}; i < subs.size();) {
      size_t sg = subs[i].first;
      if (sg >= groups.size() || groups[sg].atom_index.empty()) {
        ++i;
        continue;
      }
      size_t sg_lo = groups[sg].atom_index.front();
      size_t sg_hi = groups[sg].atom_index.back();
      if (sg_lo < parent_lo || sg_hi > parent_hi) {
        cout << "[WARNING] -sg: Sub-group \"" << groups[sg].name
             << "\" (atoms " << sg_lo << "-" << sg_hi
             << ") is outside " << label << " group \""
             << groups[parent_group].name << "\" (atoms " << parent_lo
             << "-" << parent_hi << "). Removing.\n";
        subs.erase(subs.begin() + i);
      } else {
        ++i;
      }
    }
  };
  validate(sub_group_1, group_1, "mol1");
  validate(sub_group_2, group_2, "mol2");

  return 0;
}

int groups_cal::check_sub_group() {
  for (size_t i{}; i < sub_group_1.size(); ++i)
    if (sub_group_1[i].first < 0 || sub_group_1[i].first >= groups.size()) {
      cout << "Error occur! Sub group number " << sub_group_1[i].first
           << " is not contained in index file! (" << groups.size() << ")\n";
      sub_group_1.erase(sub_group_1.begin() + i);
      --i;
    }
  for (size_t i{}; i < sub_group_2.size(); ++i)
    if (sub_group_2[i].first < 0 || sub_group_2[i].first >= groups.size()) {
      cout << "Error occur! Sub group number " << sub_group_2[i].first
           << " is not contained in index file! (" << groups.size() << ")\n";
      sub_group_2.erase(sub_group_2.begin() + i);
      --i;
    }
  return 0;
}

int helper_parse_subgroups(vector<pair<int, size_t>> &subgroups_index,
                           groups_cal &groups, size_t group) {
  for (size_t i{}; i < subgroups_index.size(); ++i) {
    if (groups.atom_index(subgroups_index[i].first, 0) -
            groups.atom_index(group, 0) >=
        0) {
      subgroups_index[i].second =
          groups.atom_index(subgroups_index[i].first, 0) -
          groups.atom_index(group, 0);
    } else {
      cout << "Error occur! Sub group "
           << groups.group_name(subgroups_index[i].first)
           << " is not contained in group " << groups.group_name(group)
           << "!\n";
      subgroups_index.erase(subgroups_index.begin() + i);
    }
  }
  return 0;
}

int groups_cal::parse_subgroups() {
  helper_parse_subgroups(sub_group_1, *this, this->group_1_index());
  helper_parse_subgroups(sub_group_2, *this, this->group_2_index());
  return 0;
}

size_t groups_cal::add_group_pathway(arguments &Arguments) {
  if (Arguments.de.flag_pathway == false)
    return 0;
  if (!Arguments.de.pathway_file.empty())
    return 0;

  group_pathway = Arguments.de.group_pathway;
  if (group_pathway >= groups.size()) {
    cerr << "Pathway group " << group_pathway
         << " is not present in the index file.\n";
    return 1;
  }

  groups[group_pathway].gen_setgroup();
  if (groups[group_pathway].atom_index.empty()) {
    cerr << "Pathway group " << group_pathway << " is empty.\n";
    return 1;
  }

  for (size_t atom_index : groups[group_pathway].atom_index) {
    if (!groups[group_1].check_in(atom_index)) {
      cerr << "Pathway group must be a subset of reference group g1.\n";
      return 1;
    }
    if (groups[group_2].check_in(atom_index)) {
      cerr << "Pathway group must not overlap the pulled group g2.\n";
      return 1;
    }
  }
  return 0;
}

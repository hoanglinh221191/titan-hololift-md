#include "topology/fflib_rtp_parser.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace TitanFfLib {
namespace {

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

bool known_non_atom_section(std::string_view section) {
  static const std::unordered_set<std::string> sections{
      "bondedtypes", "bonds",        "angles",       "dihedrals",
      "impropers",   "cmap",         "exclusions",   "pairs",
      "constraints", "settles",      "virtual_sites2", "virtual_sites3",
      "virtual_sites4", "virtual_sitesn"};
  return sections.find(std::string(section)) != sections.end();
}

std::string line_error(std::size_t line_number, std::string_view message) {
  return "RTP line " + std::to_string(line_number) + ": " +
         std::string(message);
}

} // namespace

std::expected<RtpParseSummary, std::string>
parse_gromacs_rtp_atoms(std::istream &input, amino_acid &amine) {
  RtpParseSummary summary;
  std::string current_residue;
  bool in_atoms = false;
  std::string raw;
  std::size_t line_number = 0;

  while (std::getline(input, raw)) {
    ++line_number;
    const std::size_t comment = raw.find(';');
    if (comment != std::string::npos)
      raw.erase(comment);
    const std::size_t first_nonspace = raw.find_first_not_of(" \t\r\n");
    if (first_nonspace == std::string::npos)
      continue;
    const bool indented_header = first_nonspace != 0;
    std::string line = trim_copy(raw);
    if (line.empty() || line.front() == '#')
      continue;

    if (line.front() == '[') {
      const std::size_t close = line.find(']');
      if (close == std::string::npos)
        return std::unexpected(line_error(line_number, "unterminated section header"));
      const std::string section = trim_copy(line.substr(1, close - 1));
      if (section.empty())
        return std::unexpected(line_error(line_number, "empty section header"));
      const std::string lower = lower_copy(section);
      if (lower == "atoms") {
        if (current_residue.empty())
          return std::unexpected(
              line_error(line_number, "[ atoms ] appears before a residue"));
        in_atoms = true;
        continue;
      }
      if (indented_header || known_non_atom_section(lower)) {
        in_atoms = false;
        continue;
      }

      if (amine.amino_acids.find(section) != amine.amino_acids.end())
        return std::unexpected(
            line_error(line_number, "duplicate residue section " + section));
      amine.amino_acids.emplace(section, std::vector<info_aa>{});
      current_residue = section;
      in_atoms = false;
      ++summary.residue_count;
      continue;
    }

    if (!in_atoms)
      continue;
    std::istringstream fields(line);
    std::string atom;
    std::string atom_type;
    double charge = 0.0;
    int charge_group = 0;
    if (!(fields >> atom >> atom_type >> charge >> charge_group))
      return std::unexpected(
          line_error(line_number, "malformed [ atoms ] record"));
    amine.amino_acids[current_residue].emplace_back(
        charge_group, charge, 1.0, 0.0, 0.0, 0.0, atom, atom_type);
    ++summary.atom_count;
  }

  if (summary.residue_count == 0 || summary.atom_count == 0)
    return std::unexpected("RTP contains no residue atom records");
  return summary;
}

} // namespace TitanFfLib

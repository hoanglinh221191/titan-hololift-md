// Name keys of atoms and residues, and force-field families, used by
// the lib_ff assigner.
#include "topology/fflib_detail.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace TitanFfLib::detail {

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

bool forcefield_is_amber(std::string_view force_f) {
  return std::string(force_f).find("AMBER") != std::string::npos;
}

bool forcefield_is_charmm(std::string_view force_f) {
  return std::string(force_f).find("CHARMM") != std::string::npos;
}

bool forcefield_is_gromos(std::string_view force_f) {
  return std::string(force_f).find("GROMOS") != std::string::npos;
}

bool forcefield_is_opls(std::string_view force_f) {
  return std::string(force_f).find("OPLS") != std::string::npos;
}

} // namespace TitanFfLib::detail

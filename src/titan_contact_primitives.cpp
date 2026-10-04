#include "titan_contact_primitives.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace {

constexpr const char *kInvalidModeMessage =
    "CMap: -cmap-mode must be one of sidechain-com, sidechain-heavy, cb, ca, bb.";

bool residue_key_equal(const titan_contact::ResidueKey &lhs,
                       const titan_contact::ResidueKey &rhs) noexcept {
  return lhs == rhs;
}

void hash_combine(std::size_t &seed, std::size_t value) noexcept {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

bool residue_less_for_pair_order(const titan_contact::ResidueKey &lhs,
                                 const titan_contact::ResidueKey &rhs) {
  if (lhs.molecule != rhs.molecule)
    return lhs.molecule < rhs.molecule;
  if (lhs.chainnum != rhs.chainnum)
    return lhs.chainnum < rhs.chainnum;
  if (lhs.resnr != rhs.resnr)
    return lhs.resnr < rhs.resnr;
  return false;
}

bool residue_name_starts_gly(const std::string &resname) {
  return resname.starts_with("GLY");
}

} // namespace

namespace titan_contact {

bool ResiduePairKey::operator==(const ResiduePairKey &other) const noexcept {
  return residue_key_equal(first, other.first) &&
         residue_key_equal(second, other.second);
}

std::size_t ResiduePairKeyHash::operator()(
    const ResiduePairKey &pair) const noexcept {
  std::size_t seed = 0;
  hash_combine(seed, pair.first.molecule);
  hash_combine(seed, pair.first.chainnum);
  hash_combine(seed, pair.first.resnr);
  hash_combine(seed, pair.second.molecule);
  hash_combine(seed, pair.second.chainnum);
  hash_combine(seed, pair.second.resnr);
  return seed;
}

std::string uppercase_copy(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  return out;
}

std::string trim_copy(std::string_view value) {
  const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  while (!value.empty() && is_space(static_cast<unsigned char>(value.front())))
    value.remove_prefix(1);
  while (!value.empty() && is_space(static_cast<unsigned char>(value.back())))
    value.remove_suffix(1);
  return std::string(value);
}

ContactMode parse_contact_mode(std::string_view value) {
  const std::string mode = uppercase_copy(trim_copy(value));
  if (mode == "SIDECHAIN-COM" || mode == "SC-COM")
    return ContactMode::SidechainCOM;
  if (mode == "SIDECHAIN" || mode == "SIDECHAIN-HEAVY")
    return ContactMode::SidechainHeavy;
  if (mode == "CB")
    return ContactMode::CB;
  if (mode == "CA")
    return ContactMode::CA;
  if (mode == "BB" || mode == "BACKBONE")
    return ContactMode::Backbone;
  throw std::invalid_argument(kInvalidModeMessage);
}

const char *contact_mode_name(ContactMode mode) noexcept {
  switch (mode) {
  case ContactMode::SidechainCOM:
    return "sidechain-com";
  case ContactMode::SidechainHeavy:
    return "sidechain-heavy";
  case ContactMode::CB:
    return "cb";
  case ContactMode::CA:
    return "ca";
  case ContactMode::Backbone:
    return "bb";
  }
  return "unknown";
}

bool is_backbone_atom_name(std::string_view atom_name) {
  const std::string atom = uppercase_copy(trim_copy(atom_name));
  return atom == "N" || atom == "CA" || atom == "C" || atom == "O" ||
         atom == "OXT" || atom == "H" || atom == "H1" || atom == "H2" ||
         atom == "H3" || atom == "HA";
}

bool is_hydrogen_atom_name(std::string_view atom_name) {
  const std::string atom = uppercase_copy(trim_copy(atom_name));
  if (atom.empty())
    return false;
  std::size_t first_alpha = 0;
  while (first_alpha < atom.size() &&
         std::isdigit(static_cast<unsigned char>(atom[first_alpha]))) {
    ++first_alpha;
  }
  return first_alpha < atom.size() && atom[first_alpha] == 'H';
}

bool is_selected_contact_atom(const mol_info &info, ContactMode mode) {
  const std::string atom = uppercase_copy(trim_copy(info.get_atom()));
  const std::string resname = uppercase_copy(trim_copy(info.get_res_name()));
  const bool gly = residue_name_starts_gly(resname);

  switch (mode) {
  case ContactMode::SidechainCOM:
    return info.is_side_chain();
  case ContactMode::SidechainHeavy:
    if (gly)
      return atom == "CA";
    return !is_hydrogen_atom_name(atom) && !is_backbone_atom_name(atom);
  case ContactMode::CB:
    return gly ? atom == "CA" : atom == "CB";
  case ContactMode::CA:
    return atom == "CA";
  case ContactMode::Backbone:
    return !is_hydrogen_atom_name(atom) &&
           (atom == "N" || atom == "CA" || atom == "C" || atom == "O" ||
            atom == "OXT");
  }
  return false;
}

ResidueKey residue_key_from_atom(const mol_info &info, std::size_t ordinal,
                                 std::uint8_t molecule) {
  return ResidueKey{molecule, info.get_chainnum(), info.get_resnr(),
                    info.get_chain(), info.get_res_name(), ordinal};
}

ResiduePairKey make_ordered_pair(const ResidueKey &lhs,
                                 const ResidueKey &rhs) {
  if (residue_less_for_pair_order(rhs, lhs))
    return ResiduePairKey{rhs, lhs};
  return ResiduePairKey{lhs, rhs};
}

bool same_residue(const ResidueKey &lhs, const ResidueKey &rhs) noexcept {
  return lhs.molecule == rhs.molecule && lhs.chainnum == rhs.chainnum &&
         lhs.resnr == rhs.resnr;
}

bool same_chain(const ResidueKey &lhs, const ResidueKey &rhs) noexcept {
  return lhs.molecule == rhs.molecule && lhs.chainnum == rhs.chainnum;
}

} // namespace titan_contact

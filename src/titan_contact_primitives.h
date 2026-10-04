#ifndef TITAN_CONTACT_PRIMITIVES_H
#define TITAN_CONTACT_PRIMITIVES_H

#include "MOL_info.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace titan_contact {

enum class ContactMode : std::uint8_t {
  SidechainCOM,
  SidechainHeavy,
  CB,
  CA,
  Backbone
};

struct ResidueKey {
  std::uint8_t molecule{1};
  std::size_t chainnum{};
  std::size_t resnr{};
  std::string chain;
  std::string resname;
  std::size_t ordinal{};

  // Interface CMap identity includes molecule role as well as residue id.
  bool operator==(const ResidueKey &other) const noexcept {
    return molecule == other.molecule && chainnum == other.chainnum &&
           resnr == other.resnr;
  }
};

struct ResiduePairKey {
  ResidueKey first;
  ResidueKey second;

  bool operator==(const ResiduePairKey &other) const noexcept;
};

struct ResiduePairKeyHash {
  std::size_t operator()(const ResiduePairKey &pair) const noexcept;
};

std::string uppercase_copy(std::string_view value);
std::string trim_copy(std::string_view value);

ContactMode parse_contact_mode(std::string_view value);
const char *contact_mode_name(ContactMode mode) noexcept;

bool is_backbone_atom_name(std::string_view atom_name);
bool is_hydrogen_atom_name(std::string_view atom_name);
bool is_selected_contact_atom(const mol_info &info, ContactMode mode);

ResidueKey residue_key_from_atom(const mol_info &info, std::size_t ordinal,
                                 std::uint8_t molecule = 1);
ResiduePairKey make_ordered_pair(const ResidueKey &lhs,
                                 const ResidueKey &rhs);

bool same_residue(const ResidueKey &lhs, const ResidueKey &rhs) noexcept;
bool same_chain(const ResidueKey &lhs, const ResidueKey &rhs) noexcept;

} // namespace titan_contact

#endif // TITAN_CONTACT_PRIMITIVES_H

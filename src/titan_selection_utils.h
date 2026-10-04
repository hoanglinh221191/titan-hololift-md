#ifndef TITAN_SELECTION_UTILS_H
#define TITAN_SELECTION_UTILS_H

#include "Multi_MOL.h"

#include <cstdint>
#include <string>
#include <vector>

namespace titan_selection {

enum class TraceKind : std::uint8_t {
  None = 0,
  CA,
  C4Prime,
  MartiniBB
};

struct SelectedAtom {
  int atom_index{-1};
  std::size_t ordinal{0};
  std::size_t resnr{0};
  std::size_t chainnum{0};
  std::string chain{" "};
  std::string resname{"UNK"};
  std::string atom_name{"?"};
};

std::string uppercase_copy(std::string value);
std::string trim_copy(std::string value);
TraceKind select_trace_kind(const std::string &mol_type,
                            const std::string &force_field);
const char *trace_kind_name(TraceKind kind) noexcept;
int trace_priority(TraceKind kind, const std::string &atom_name) noexcept;
bool is_traceable_atom(const std::string &atom_name, TraceKind trace_kind) noexcept;
std::vector<std::string> parse_name_list(const std::string &raw_names);

void build_trace_atoms(const Multi_MOL &mol, const std::string &mol_type,
                       const std::string &force_field,
                       std::vector<SelectedAtom> &trace_atoms);
void build_all_atom_selection(const Multi_MOL &mol,
                              std::vector<SelectedAtom> &selected_atoms);
void build_named_atom_selection(const Multi_MOL &mol,
                                const std::vector<std::string> &atom_names,
                                std::vector<SelectedAtom> &selected_atoms);
void build_backbone_atoms(const Multi_MOL &mol,
                          std::vector<SelectedAtom> &selected_atoms);
void build_sticker_atoms(const Multi_MOL &mol, const std::string &sticker_names,
                         std::vector<SelectedAtom> &selected_atoms);
void build_channel_atoms(const Multi_MOL &mol, const std::string &mol_type,
                         const std::string &force_field,
                         const std::string &channel,
                         const std::string &sticker_names,
                         std::vector<SelectedAtom> &selected_atoms);

} // namespace titan_selection

#endif // TITAN_SELECTION_UTILS_H

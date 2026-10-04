#include "titan_selection_utils.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

namespace titan_selection {

namespace {

bool contains_name(const std::vector<std::string> &names,
                   const std::string &atom_name_upper) {
  return std::find(names.begin(), names.end(), atom_name_upper) != names.end();
}

SelectedAtom build_selected_atom(const mol_info &atom, std::size_t atom_index,
                                 std::size_t ordinal,
                                 std::string atom_name) {
  SelectedAtom selected{};
  selected.atom_index = static_cast<int>(atom_index);
  selected.ordinal = ordinal;
  selected.resnr = atom.get_resnr();
  selected.chainnum = atom.get_chainnum();
  selected.chain = atom.get_chain();
  selected.resname = atom.get_res_name();
  selected.atom_name = std::move(atom_name);
  return selected;
}

} // namespace

std::string uppercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

TraceKind select_trace_kind(const std::string &mol_type,
                            const std::string &force_field) {
  const std::string type = uppercase_copy(mol_type);
  const std::string ff = uppercase_copy(force_field);

  if (ff == "MARTINI") {
    if (type == "LIG" || type == "LIGAND")
      return TraceKind::None;
    return TraceKind::MartiniBB;
  }

  if (type == "PROTEIN" || type == "PROTEIN-USER")
    return TraceKind::CA;
  if (type == "RNA" || type == "DNA")
    return TraceKind::C4Prime;
  return TraceKind::None;
}

const char *trace_kind_name(TraceKind kind) noexcept {
  switch (kind) {
  case TraceKind::CA:
    return "CA";
  case TraceKind::C4Prime:
    return "C4'";
  case TraceKind::MartiniBB:
    return "BB";
  case TraceKind::None:
  default:
    return "none";
  }
}

int trace_priority(TraceKind kind, const std::string &atom_name) noexcept {
  switch (kind) {
  case TraceKind::CA:
    return atom_name == "CA" ? 2 : (atom_name == "CA " ? 1 : -1);
  case TraceKind::C4Prime:
    if (atom_name == "C4'")
      return 2;
    if (atom_name == "C4*")
      return 1;
    return -1;
  case TraceKind::MartiniBB:
    if (atom_name == "BB")
      return 2;
    if (atom_name == "BB2")
      return 1;
    return -1;
  case TraceKind::None:
  default:
    return -1;
  }
}

bool is_traceable_atom(const std::string &atom_name, TraceKind trace_kind) noexcept {
  return trace_priority(trace_kind, atom_name) >= 0;
}

std::vector<std::string> parse_name_list(const std::string &raw_names) {
  std::vector<std::string> names;
  std::istringstream input(raw_names);
  std::string token;
  while (std::getline(input, token, ',')) {
    token = uppercase_copy(trim_copy(token));
    if (!token.empty() &&
        std::find(names.begin(), names.end(), token) == names.end()) {
      names.push_back(token);
    }
  }
  return names;
}

void build_trace_atoms(const Multi_MOL &mol, const std::string &mol_type,
                       const std::string &force_field,
                       std::vector<SelectedAtom> &trace_atoms) {
  trace_atoms.clear();
  const TraceKind trace_kind = select_trace_kind(mol_type, force_field);
  if (trace_kind == TraceKind::None || mol.info.empty())
    return;

  std::size_t prev_chain = std::numeric_limits<std::size_t>::max();
  std::size_t prev_resnr = std::numeric_limits<std::size_t>::max();
  int best_priority = -1;
  SelectedAtom current{};
  bool has_residue = false;

  auto flush_current = [&]() {
    if (has_residue && current.atom_index >= 0) {
      current.ordinal = trace_atoms.size() + 1;
      trace_atoms.push_back(current);
    }
  };

  for (std::size_t atom_idx = 0; atom_idx < mol.info.size(); ++atom_idx) {
    const auto &atom = mol.info[atom_idx];
    const std::size_t chainnum = atom.get_chainnum();
    const std::size_t resnr = atom.get_resnr();
    const bool new_residue =
        !has_residue || chainnum != prev_chain || resnr != prev_resnr;

    if (new_residue) {
      flush_current();
      prev_chain = chainnum;
      prev_resnr = resnr;
      has_residue = true;
      best_priority = -1;
      current = SelectedAtom{};
      current.resnr = resnr;
      current.chainnum = chainnum;
      current.chain = atom.get_chain();
      current.resname = atom.get_res_name();
    }

    const std::string atom_name = uppercase_copy(trim_copy(atom.get_atom()));
    if (!is_traceable_atom(atom_name, trace_kind))
      continue;

    const int priority = trace_priority(trace_kind, atom_name);
    if (priority > best_priority) {
      best_priority = priority;
      current.atom_index = static_cast<int>(atom_idx);
      current.atom_name = trim_copy(atom.get_atom());
    }
  }

  flush_current();
}

void build_all_atom_selection(const Multi_MOL &mol,
                              std::vector<SelectedAtom> &selected_atoms) {
  selected_atoms.clear();
  selected_atoms.reserve(mol.info.size());
  for (std::size_t atom_idx = 0; atom_idx < mol.info.size(); ++atom_idx) {
    const auto &atom = mol.info[atom_idx];
    selected_atoms.push_back(build_selected_atom(
        atom, atom_idx, selected_atoms.size() + 1, trim_copy(atom.get_atom())));
  }
}

void build_named_atom_selection(const Multi_MOL &mol,
                                const std::vector<std::string> &atom_names,
                                std::vector<SelectedAtom> &selected_atoms) {
  selected_atoms.clear();
  if (mol.info.empty() || atom_names.empty())
    return;

  for (std::size_t atom_idx = 0; atom_idx < mol.info.size(); ++atom_idx) {
    const auto &atom = mol.info[atom_idx];
    const std::string atom_name = uppercase_copy(trim_copy(atom.get_atom()));
    if (!contains_name(atom_names, atom_name))
      continue;
    selected_atoms.push_back(build_selected_atom(
        atom, atom_idx, selected_atoms.size() + 1, trim_copy(atom.get_atom())));
  }
}

void build_backbone_atoms(const Multi_MOL &mol,
                          std::vector<SelectedAtom> &selected_atoms) {
  static const std::vector<std::string> kBackboneNames{
      "CA", "C4'", "C4*", "BB", "BB1", "BB2", "BB3"};
  build_named_atom_selection(mol, kBackboneNames, selected_atoms);
}

void build_sticker_atoms(const Multi_MOL &mol, const std::string &sticker_names,
                         std::vector<SelectedAtom> &selected_atoms) {
  build_named_atom_selection(mol, parse_name_list(sticker_names), selected_atoms);
}

void build_channel_atoms(const Multi_MOL &mol, const std::string &mol_type,
                         const std::string &force_field,
                         const std::string &channel,
                         const std::string &sticker_names,
                         std::vector<SelectedAtom> &selected_atoms) {
  const std::string normalized_channel = uppercase_copy(trim_copy(channel));
  if (normalized_channel == "ALL") {
    build_all_atom_selection(mol, selected_atoms);
    return;
  }
  if (normalized_channel == "TRACE") {
    build_trace_atoms(mol, mol_type, force_field, selected_atoms);
    return;
  }
  if (normalized_channel == "BACKBONE") {
    build_backbone_atoms(mol, selected_atoms);
    return;
  }
  if (normalized_channel == "STICKER") {
    build_sticker_atoms(mol, sticker_names, selected_atoms);
    return;
  }
  selected_atoms.clear();
}

} // namespace titan_selection

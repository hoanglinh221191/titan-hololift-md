#ifndef TITAN_TOPOLOGY_FFLIB_DETAIL_H
#define TITAN_TOPOLOGY_FFLIB_DETAIL_H

// Internal pieces of the lib_ff assigner (TitanFfLib::assign_from_library),
// shared by the fflib_*.cpp files of this folder. Not part of the public
// interface: include topology/fflib_assigner.h from outside.

#include "topology/fflib_assigner.h"

#include "MOL.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class TitanExecutor;

namespace TitanFfLib::detail {

constexpr std::size_t kMaxAssignedReportRows = 2000;
constexpr double kHydrogenOffset = 0.10;

struct AtomTemplate {
  std::string atom;
  std::string atom_key;
  std::string atom_type;
  double charge{};
  double mass{1.0};
  double sigma{};
  double epsilon{};
  double radius{};
  bool hydrogen{};
};

struct ResidueTemplate {
  std::string name;
  std::vector<AtomTemplate> atoms;
  std::unordered_map<std::string, std::size_t> atom_to_index;
};

struct TemplateTables {
  std::unordered_map<std::string, ResidueTemplate> residues;
  std::unordered_map<std::string, std::size_t> vdw_by_type;
};

struct ResidueSpan {
  std::size_t begin{};
  std::size_t end{};
  std::string chain;
  std::size_t resnr{};
  std::string resname;
  std::string type{"unknown"};
  bool first_in_chain{};
  bool last_in_chain{};
};

struct HydrogenPlan {
  AtomTemplate atom;
  std::size_t anchor{};
  bool can_place{};
  std::string message;
};

struct ResiduePlan {
  const ResidueTemplate *templ{};
  std::string templ_name;
  std::vector<HydrogenPlan> hydrogens;
  std::size_t assigned{};
  std::size_t unmatched{};
  std::size_t missing_vdw{};
  std::size_t report_assigned{};
  std::vector<std::string> report_rows;
};

// One atom of a terminal patch (fflib_terminal_patches.cpp).
struct PatchAtom {
  const char *key;
  const char *type;
  double charge;
};

enum class PatchFamily { None, Charmm, Gromos43, Gromos, Opls };

// fflib_names.cpp: atom/residue name keys and force-field families.
std::string upper_copy(std::string value);
std::string trim_copy(std::string value);
std::string normalize_atom_key(std::string atom);
std::string normalize_residue_key(std::string res);
bool is_hydrogen_name(const std::string &atom_key);
bool has_suffix(std::string_view value, std::string_view suffix);
bool forcefield_is_amber(std::string_view force_f);
bool forcefield_is_charmm(std::string_view force_f);
bool forcefield_is_gromos(std::string_view force_f);
bool forcefield_is_opls(std::string_view force_f);

// fflib_templates.cpp: residue templates built from the rtp + nonbonded table.
std::shared_ptr<const TemplateTables> cached_tables(const amino_acid &amine,
                                                    const para &parameter);
const AtomTemplate *
find_atom_template_for_input(const ResidueTemplate &templ,
                             const std::string &atom_key,
                             std::unordered_set<std::size_t> &used_templates);
bool template_is_protein(const ResidueTemplate &templ);

// fflib_residues.cpp: residue spans of the structure and their templates.
std::vector<ResidueSpan> build_residue_spans(const Multi_MOL &mol);
const ResidueTemplate *find_template(const TemplateTables &tables,
                                     const Multi_MOL &mol,
                                     const ResidueSpan &span,
                                     std::string_view force_f,
                                     std::string &matched_name);

// fflib_terminal_patches.cpp: N-/C-terminal patches (ENE-09).
PatchFamily patch_family(std::string_view force_f);
const std::vector<PatchAtom> *nterm_patch(PatchFamily family,
                                          const std::string &templ_name);
const std::vector<PatchAtom> *cterm_patch(PatchFamily family,
                                          const std::string &templ_name);
std::string terminal_patch_key(const std::string &atom_key);
const PatchAtom *find_patch_atom(const std::vector<PatchAtom> *patch,
                                 const std::string &key);
bool span_has_terminal_amine(const Multi_MOL &mol, const ResidueSpan &span);
bool span_has_terminal_carboxylate(const Multi_MOL &mol,
                                   const ResidueSpan &span);
void apply_terminal_patch(Multi_MOL &mol, std::size_t atom_index,
                          const PatchAtom &patch, const para &parameter);

// fflib_assign_residue.cpp: parameters of the atoms of one residue.
void assign_residue(Multi_MOL &mol, const TemplateTables &tables,
                    const ResidueSpan &span, ResiduePlan &plan,
                    const para &parameter, const arguments &Arguments,
                    std::string_view force_f, std::string_view molecule_label,
                    bool report_h, bool add_h);

// fflib_hydrogens.cpp: missing hydrogens of the templates.
std::size_t find_anchor_atom(const Multi_MOL &mol, const ResidueSpan &span,
                             const std::string &hydrogen_key);
bool should_report_hydrogens(const arguments &Arguments);
bool should_add_hydrogens(const arguments &Arguments, const Multi_MOL &mol);
std::string hydrogen_skip_policy_message(const arguments &Arguments,
                                         const Multi_MOL &mol);
void insert_planned_hydrogens(Multi_MOL &mol,
                              const std::vector<ResidueSpan> &spans,
                              const std::vector<ResiduePlan> &plans,
                              const para &parameter,
                              const arguments &Arguments,
                              TitanExecutor *executor);

// fflib_report.cpp: fflib_assignment_report.tsv and charge warnings.
std::string make_report_row(std::string_view molecule, const ResidueSpan &span,
                            std::string_view atom, std::string_view action,
                            std::string_view templ,
                            std::string_view atom_type, double charge,
                            double sigma, double epsilon,
                            std::string_view message);
void append_report(const std::vector<ResiduePlan> &plans,
                   std::string_view molecule_label, std::size_t atom_count,
                   std::size_t final_count, bool strict_missing);
void warn_fractional_charge(const Multi_MOL &mol,
                            const std::vector<ResiduePlan> &plans,
                            std::string_view molecule_label);

} // namespace TitanFfLib::detail

#endif // TITAN_TOPOLOGY_FFLIB_DETAIL_H

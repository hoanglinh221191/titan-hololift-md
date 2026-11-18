// Parameters of the atoms of one residue span: template atoms, terminal
// patches (ENE-09), LJ from the atom type, and the report rows.
#include "topology/fflib_detail.h"

#include <initializer_list>
#include <string>
#include <unordered_set>
#include <utility>

namespace TitanFfLib::detail {
namespace {

void set_atom_parameter(Multi_MOL &mol, std::size_t atom_index,
                        const AtomTemplate &templ) {
  mol.info[atom_index].set_atom_type(templ.atom_type);
  if (atom_index >= mol.parameters.size())
    mol.parameters.resize(mol.info.size());
  mol.parameters[atom_index].set_data(templ.charge, templ.radius, templ.sigma,
                                      templ.epsilon, templ.mass);
}

void set_unassigned(Multi_MOL &mol, std::size_t atom_index) {
  mol.info[atom_index].set_atom_type("UNASSIGNED");
  if (atom_index >= mol.parameters.size())
    mol.parameters.resize(mol.info.size());
  mol.parameters[atom_index].set_data(0.0, 0.0, 0.0, 0.0, 1.0);
  mol.parameters[atom_index].set_type_id(-1);
}

} // namespace

void assign_residue(Multi_MOL &mol, const TemplateTables &tables,
                    const ResidueSpan &span, ResiduePlan &plan,
                    const para &parameter, const arguments &Arguments,
                    std::string_view force_f, std::string_view molecule_label,
                    bool report_h, bool add_h) {
  plan.templ = find_template(tables, mol, span, force_f, plan.templ_name);
  if (plan.templ == nullptr) {
    for (std::size_t i = span.begin; i < span.end; ++i) {
      set_unassigned(mol, i);
      ++plan.unmatched;
      plan.report_rows.emplace_back(make_report_row(
          molecule_label, span, mol.info[i].get_atom(), "unmatched_residue",
          "", "UNASSIGNED", 0.0, 0.0, 0.0,
          "residue template not found in lib_ff"));
    }
    return;
  }

  // ENE-09: terminal patches for the force fields whose rtp has no
  // terminal residues (CHARMM, GROMOS, OPLS-AA).
  const std::vector<PatchAtom> *nterm_patch_atoms = nullptr;
  const std::vector<PatchAtom> *cterm_patch_atoms = nullptr;
  const PatchFamily family = patch_family(force_f);
  if (family != PatchFamily::None && template_is_protein(*plan.templ)) {
    if (span.first_in_chain && span_has_terminal_amine(mol, span))
      nterm_patch_atoms = nterm_patch(family, plan.templ_name);
    if (span.last_in_chain && span_has_terminal_carboxylate(mol, span))
      cterm_patch_atoms = cterm_patch(family, plan.templ_name);
  }

  std::unordered_set<std::string> present_atoms;
  present_atoms.reserve(span.end - span.begin);
  std::unordered_set<std::size_t> used_template_atoms;
  used_template_atoms.reserve(span.end - span.begin);
  for (std::size_t i = span.begin; i < span.end; ++i) {
    const std::string atom_key = normalize_atom_key(mol.info[i].get_atom());
    present_atoms.insert(atom_key);
    const std::string patch_key = terminal_patch_key(atom_key);
    const PatchAtom *patch = find_patch_atom(nterm_patch_atoms, patch_key);
    const char *patch_label = "NTER";
    if (patch == nullptr) {
      patch = find_patch_atom(cterm_patch_atoms, patch_key);
      patch_label = "CTER";
    }
    const AtomTemplate *atom_template = find_atom_template_for_input(
        *plan.templ, atom_key, used_template_atoms);
    if (atom_template == nullptr && patch == nullptr) {
      set_unassigned(mol, i);
      ++plan.unmatched;
      plan.report_rows.emplace_back(make_report_row(
          molecule_label, span, mol.info[i].get_atom(), "unmatched_atom",
          plan.templ_name, "UNASSIGNED", 0.0, 0.0, 0.0,
          "atom not found in residue template"));
      continue;
    }

    if (atom_template != nullptr)
      set_atom_parameter(mol, i, *atom_template);
    std::string atom_type =
        atom_template != nullptr ? atom_template->atom_type : std::string();
    double charge = atom_template != nullptr ? atom_template->charge : 0.0;
    std::string templ_label = plan.templ_name;
    if (patch != nullptr) {
      apply_terminal_patch(mol, i, *patch, parameter);
      atom_type = patch->type;
      charge = patch->charge;
      templ_label += std::string(":") + patch_label;
    }
    if (!fill_vdw_from_atom_type(mol.parameters, i, parameter, atom_type)) {
      ++plan.missing_vdw;
      plan.report_rows.emplace_back(make_report_row(
          molecule_label, span, mol.info[i].get_atom(), "missing_vdw",
          templ_label, atom_type, charge,
          mol.parameters[i].get_sigma(), mol.parameters[i].get_epsilon(),
          "atom type not found in nonbonded table"));
    }
    ++plan.assigned;
    if (i < kMaxAssignedReportRows) {
      plan.report_rows.emplace_back(make_report_row(
          molecule_label, span, mol.info[i].get_atom(),
          patch != nullptr ? "assigned_terminal_patch" : "assigned",
          templ_label, atom_type, charge, mol.parameters[i].get_sigma(),
          mol.parameters[i].get_epsilon(), ""));
    }
  }

  // The template amide hydrogen is replaced by the ammonium hydrogens of
  // the patch; it must not be reported or added as a missing hydrogen.
  if (nterm_patch_atoms != nullptr) {
    for (const char *key : {"HN", "H"}) {
      const auto it = plan.templ->atom_to_index.find(key);
      if (it != plan.templ->atom_to_index.end())
        used_template_atoms.insert(it->second);
    }
  }

  if (!report_h)
    return;

  for (std::size_t atom_index = 0; atom_index < plan.templ->atoms.size();
       ++atom_index) {
    const auto &atom_template = plan.templ->atoms[atom_index];
    if (!atom_template.hydrogen ||
        used_template_atoms.find(atom_index) != used_template_atoms.end() ||
        present_atoms.find(atom_template.atom_key) != present_atoms.end())
      continue;
    HydrogenPlan h;
    h.atom = atom_template;
    h.anchor = find_anchor_atom(mol, span, atom_template.atom_key);
    h.can_place = add_h && h.anchor < span.end;
    h.message = h.can_place
                    ? "hydrogen placed from heavy atom anchor"
                    : (h.anchor >= span.end
                           ? "no heavy atom anchor found"
                           : hydrogen_skip_policy_message(Arguments, mol));
    plan.hydrogens.emplace_back(std::move(h));
    plan.report_rows.emplace_back(make_report_row(
        molecule_label, span, atom_template.atom,
        plan.hydrogens.back().can_place ? "added_hydrogen"
                                        : "skipped_hydrogen",
        plan.templ_name, atom_template.atom_type, atom_template.charge,
        atom_template.sigma, atom_template.epsilon,
        plan.hydrogens.back().message));
  }
}

} // namespace TitanFfLib::detail

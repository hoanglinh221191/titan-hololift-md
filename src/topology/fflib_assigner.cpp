// lib_ff assigner: per-atom charges, types and LJ from the residue
// templates of the rtp files and the nonbonded table (no -common_topo).
//   fflib_templates.cpp         residue templates, atom aliases
//   fflib_residues.cpp          residue spans, template candidates, HIS
//   fflib_terminal_patches.cpp  N-/C-terminal patches (ENE-09)
//   fflib_assign_residue.cpp    parameters of one residue
//   fflib_hydrogens.cpp         missing template hydrogens
//   fflib_report.cpp            report file and charge warnings
//   fflib_nonbonded.cpp         atom-type LJ lookup, table check
#include "topology/fflib_detail.h"

#include "titan_executor.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace TitanFfLib {

using namespace detail;

int assign_from_library(Multi_MOL &mol, const amino_acid &amine,
                        const para &parameter, const arguments &Arguments,
                        std::string_view force_f, TitanExecutor *executor,
                        std::string_view molecule_label) {
  if (mol.info.empty() || mol.m_mol.empty())
    return 0;

  if (mol.parameters.size() < mol.info.size())
    mol.parameters.resize(mol.info.size());
  // CORE-12: allocate type_id before the parallel residue loop. set_type_id
  // allocates it on first use, so several workers reallocated the shared
  // vector at once (heap corruption, 0xC0000374, in ~2 % of two-molecule runs
  // with 16 threads).
  mol.parameters.ensure_type_ids();

  const auto tables_holder = cached_tables(amine, parameter);
  const TemplateTables &tables = *tables_holder;
  const std::vector<ResidueSpan> spans = build_residue_spans(mol);
  std::vector<ResiduePlan> plans(spans.size());
  const bool report_h = should_report_hydrogens(Arguments);
  const bool add_h = should_add_hydrogens(Arguments, mol);

  auto assign_span = [&](std::size_t worker, std::size_t begin,
                         std::size_t end) {
    (void)worker;
    for (std::size_t s = begin; s < end; ++s)
      assign_residue(mol, tables, spans[s], plans[s], parameter, Arguments,
                     force_f, molecule_label, report_h, add_h);
  };

  if (executor != nullptr && spans.size() > 1) {
    executor->parallel_for(spans.size(), Arguments.nt, 16, assign_span);
  } else {
    assign_span(0, 0, spans.size());
  }

  const bool strict_missing = Arguments.fflib_missing == "strict";
  const std::size_t original_atom_count = mol.info.size();
  insert_planned_hydrogens(mol, spans, plans, parameter, Arguments, executor);

  for (std::size_t i = 0; i < mol.info.size() && i < mol.parameters.size(); ++i) {
    const auto type_it = parameter.atom_type_to_id.find(mol.info[i].get_atom_type());
    if (type_it != parameter.atom_type_to_id.end())
      mol.parameters[i].set_type_id(static_cast<int>(type_it->second));
  }

  append_report(plans, molecule_label, original_atom_count, mol.info.size(),
                strict_missing);

  std::size_t unmatched = 0;
  for (const auto &plan : plans)
    unmatched += plan.unmatched + plan.missing_vdw;
  if (unmatched > 0) {
    std::cout << "WARNING (" << molecule_label << "): " << unmatched
              << " atoms/VDW records were not fully assigned by lib_ff. See "
                 "fflib_assignment_report.tsv\n";
    if (strict_missing) {
      std::cerr << "ERROR (" << molecule_label
                << "): -fflib-missing strict requested.\n";
      std::exit(EXIT_FAILURE);
    }
  }

  warn_fractional_charge(mol, plans, molecule_label);

  return 0;
}

} // namespace TitanFfLib

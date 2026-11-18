#ifndef TITAN_PARAMETER_AUDIT_H
#define TITAN_PARAMETER_AUDIT_H

#include "Multi_MOL.h"
#include "topology/fflib_charge_policy.h"
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <string>

namespace TitanFfLib {

// Opt-in diagnostic only: export the actual initialized SoA values, before
// energy evaluation. No parameter is changed and no topology is reparsed here.
inline const char *parameter_audit_path() noexcept {
  const char *path = std::getenv("TITAN_PARAMETER_AUDIT");
  return path != nullptr && *path != '\0' ? path : nullptr;
}

inline std::expected<void, std::string>
write_parameter_audit(const char *path, const Multi_MOL &mol1,
                      const Multi_MOL *mol2, ChargeSourcePolicy policy) {
  const auto consistent = [](const Multi_MOL &mol) {
    const auto n = mol.info.size();
    const auto &p = mol.parameters;
    return !mol.m_mol.empty() && mol.m_mol[0].size() == n &&
           p.charge.size() == n && p.mass.size() == n && p.sigma.size() == n &&
           p.epsilon.size() == n && p.radius.size() == n;
  };
  if (!consistent(mol1) || (mol2 != nullptr && !consistent(*mol2)))
    return std::unexpected("Parameter audit: atom/parameter size mismatch");
  std::error_code ec;
  if (std::filesystem::exists(path, ec) || ec)
    return std::unexpected("Parameter audit: refusing an existing/unreadable output");
  std::ofstream out(path);
  if (!out)
    return std::unexpected("Parameter audit: cannot create output");
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "molecule\tselection_index_0based\tsource_atom_id_0based\thas_source_id\t"
         "residue_number\tresidue\tatom\tatom_type\tcharge_e\tmass_u\t"
         "sigma_nm\tepsilon_kjmol\tradius_A\tassignment_policy\n";
  const auto write_molecule = [&](const Multi_MOL &mol, const char *label,
                                  bool topology) {
    const auto &p = mol.parameters;
    for (std::size_t i = 0; i < mol.info.size(); ++i) {
      const auto &atom = mol.info[i];
      out << label << '\t' << i << '\t' << atom.get_original_atom_id() << '\t'
          << atom.has_original_atom_id() << '\t' << atom.get_resnr() << '\t'
          << atom.get_res_name() << '\t' << atom.get_atom() << '\t'
          << atom.get_atom_type() << '\t' << p.charge[i] << '\t' << p.mass[i]
          << '\t' << p.sigma[i] << '\t' << p.epsilon[i] << '\t' << p.radius[i]
          << '\t' << (topology ? "topology" : "lib_ff") << '\n';
    }
  };
  write_molecule(mol1, "mol1", policy.molecule1_uses_topology);
  if (mol2 != nullptr)
    write_molecule(*mol2, "mol2", policy.molecule2_uses_topology);
  out.close();
  if (!out)
    return std::unexpected("Parameter audit: failed while writing output");
  return {};
}

} // namespace TitanFfLib
#endif

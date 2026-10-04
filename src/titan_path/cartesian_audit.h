#ifndef TITAN_PATH_CARTESIAN_AUDIT_H
#define TITAN_PATH_CARTESIAN_AUDIT_H

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>

namespace titan_path {
// Opt-in export of the exact Cartesian inputs AFTER PBC handling and
// immediately before pathway selection. Never reparses the topology or alters
// coordinates.
template <class Molecule>
void write_cartesian_audit(const std::filesystem::path &path,
                           const Molecule &receptor, const Molecule &ligand) {
  const auto validate = [](const Molecule &m) {
    const auto n = m.info.size();
    if (n == 0 || m.m_mol.empty() || m.parameters.mass.size() != n ||
        m.m_mol[0].pos.x.size() != n || m.m_mol[0].pos.y.size() != n ||
        m.m_mol[0].pos.z.size() != n)
      throw std::invalid_argument(
          "Cartesian audit atom/parameter mapping mismatch");
    std::set<std::size_t> ids;
    for (std::size_t i = 0; i < n; ++i) {
      if (!m.info[i].has_original_atom_id() ||
          !ids.insert(m.info[i].get_original_atom_id()).second ||
          !std::isfinite(m.parameters.mass[i]) || m.parameters.mass[i] <= 0 ||
          !std::isfinite(m.m_mol[0].pos.x[i]) ||
          !std::isfinite(m.m_mol[0].pos.y[i]) ||
          !std::isfinite(m.m_mol[0].pos.z[i]))
        throw std::invalid_argument(
            "Cartesian audit invalid atom identity/coordinate/mass");
    }
  };
  validate(receptor);
  validate(ligand);
  std::error_code ec;
  if (std::filesystem::exists(path, ec) || ec)
    throw std::invalid_argument(
        "Cartesian audit refuses existing/unreadable output");
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("Cannot create Cartesian audit");
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "molecule\tselection_index_0based\tsource_atom_id_0based\tx_A\ty_A\tz_"
         "A\tmass_u\n";
  const auto write = [&](const Molecule &m, const char *name) {
    for (std::size_t i = 0; i < m.info.size(); ++i)
      out << name << '\t' << i << '\t' << m.info[i].get_original_atom_id()
          << '\t' << m.m_mol[0].pos.x[i] << '\t' << m.m_mol[0].pos.y[i] << '\t'
          << m.m_mol[0].pos.z[i] << '\t' << m.parameters.mass[i] << '\n';
  };
  write(receptor, "mol1");
  write(ligand, "mol2");
  out.close();
  if (!out)
    throw std::runtime_error("Cartesian audit write failed");
}
} // namespace titan_path
#endif

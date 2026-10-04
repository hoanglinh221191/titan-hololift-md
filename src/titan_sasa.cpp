#include "titan_sasa.h"

#include "titan_surface.h"

#include <numeric>
#include <utility>

namespace TitanSASA {

std::vector<double>
compute_per_atom_sasa(std::span<const Atom> atoms, const Options &options,
                      std::span<const std::size_t> targets,
                      TitanExecutor *executor) {
  std::vector<TitanSurface::Atom> surface_atoms;
  surface_atoms.reserve(atoms.size());
  for (const Atom &atom : atoms)
    surface_atoms.push_back({atom.x, atom.y, atom.z, atom.radius});

  TitanSurface::Options surface_options;
  surface_options.probe_radius = options.probe_radius;
  surface_options.min_cell_size = options.min_cell_size;
  surface_options.point_count = options.point_count;
  surface_options.surface_mode = "sas";
  surface_options.emit_panels = false;
  surface_options.emit_neighbor_workspace = false;

  TitanSurface::Workspace workspace = TitanSurface::build_workspace(
      surface_atoms, surface_options, targets, executor);
  return std::move(workspace.area_by_atom);
}

double sum_sasa(std::span<const double> sasa) noexcept {
  return std::accumulate(sasa.begin(), sasa.end(), 0.0);
}

double nonpolar_energy_from_delta_sasa_kj(double delta_sasa_a2) noexcept {
  return kNonpolarGammaKJPerMolA2 * delta_sasa_a2;
}

} // namespace TitanSASA

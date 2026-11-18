#include "nocal/water_dipole_field.h"

#include "Multi_MOL.h"
#include "arguments.h"
#include "force.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace titan_nocal {

void WaterDipoleField::begin_trajectory(Multi_MOL &mol,
                                        const Trajectory &trajectory,
                                        const std::string &suffix) {
  trajectory_ = trajectory;
  suffix_ = suffix;
  waters_.clear();
  heavy_atoms_.clear();
  cosines_.clear();
  // The water and solute atoms come from the topology, the same every frame.
  mol.find_water(waters_, heavy_atoms_);
}

void WaterDipoleField::process_frame(const arguments &args,
                                     const Multi_MOL &mol, std::size_t snap) {
  if (waters_.empty() || snap >= mol.m_mol.size())
    return;
  const MOL &frame = mol.m_mol[snap];
  std::vector<std::size_t> selected;
  if (args.water_type == "surface")
    frame.find_surface_water(waters_, heavy_atoms_, selected);
  else
    frame.find_sphere_water(waters_, heavy_atoms_, selected,
                            args.d_crit_water);
  std::vector<force> fields(selected.size());
  frame.cal_electric_field(mol.info, args, mol.parameters, heavy_atoms_,
                           selected, fields);
  frame.cal_angle_dipoles_elecfield(mol.info, args, fields, mol.parameters,
                                    heavy_atoms_, selected, cosines_);
}

void WaterDipoleField::end_trajectory(const arguments &args) {
  if (suffix_.empty())
    return;
  const std::string path = trajectory_.file(suffix_);
  std::ofstream out(path, std::ios::out);
  if (!out)
    throw std::runtime_error("cannot write " + path);
  out << "#Cosine        Population (%)\n";
  suffix_.clear();
  if (cosines_.empty())
    return;
  const auto [low, high] = std::minmax_element(cosines_.begin(), cosines_.end());
  const double min = *low;
  const std::size_t bins =
      std::max<std::size_t>(1, args.number_bin_angle_dipole_efield);
  const double delta = (*high - min) / bins;
  std::vector<std::size_t> histogram(bins);
  for (const double c : cosines_) {
    const double position = delta > 0.0 ? (c - min) / delta : 0.0;
    const std::size_t bin = static_cast<std::size_t>(position);
    histogram[std::min(bin, bins - 1)] += 1;
  }
  for (std::size_t i = 0; i < bins; ++i)
    out << std::setw(15) << std::left << i * delta + min << " " << std::right
        << std::setw(15) << histogram[i] * 100. / cosines_.size() << '\n';
}

} // namespace titan_nocal

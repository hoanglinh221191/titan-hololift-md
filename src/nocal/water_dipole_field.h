#ifndef TITAN_NOCAL_WATER_DIPOLE_FIELD_H
#define TITAN_NOCAL_WATER_DIPOLE_FIELD_H
// -angle_dipole_efield: for the surface waters of molecule 1
// (-water-type=sphere: the waters within -dis-water of a solute heavy atom),
// the cosine between each water's dipole and the electric field of the
// solute heavy atoms at that water, collected over the frames of an input.
// When the input ends their distribution (percent per bin, bins spanning the
// data) is written to <stem>_distributionDipoleEfield.dat.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <string>
#include <vector>

class Multi_MOL;
struct arguments;

namespace titan_nocal {

class WaterDipoleField {
public:
  void begin_trajectory(Multi_MOL &mol, const Trajectory &trajectory,
                        const std::string &suffix);
  void process_frame(const arguments &args, const Multi_MOL &mol,
                     std::size_t snap);
  void end_trajectory(const arguments &args);

private:
  Trajectory trajectory_;
  std::string suffix_;
  std::vector<std::size_t> waters_, heavy_atoms_;
  std::vector<double> cosines_;
};

} // namespace titan_nocal

#endif

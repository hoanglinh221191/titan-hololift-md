#ifndef TITAN_NOCAL_DIPOLE_MOMENT_H
#define TITAN_NOCAL_DIPOLE_MOMENT_H
// -dipole-moment: molecule 1's dipole moment about its geometric centre
// (charge x A), one row per frame. <stem>_dipole_moment.dat holds the unit
// vector, <stem>_notnormalized__dipole_moment.dat the vector and its length.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <fstream>
#include <string>

class Multi_MOL;
struct arguments;

namespace titan_nocal {

class DipoleMoment {
public:
  void begin_trajectory(const arguments &args, const Trajectory &trajectory,
                        const std::string &suffix);
  void process_frame(const Multi_MOL &mol, std::size_t snap, double time);
  void end_trajectory();

private:
  std::ofstream unit_, raw_;
};

} // namespace titan_nocal

#endif

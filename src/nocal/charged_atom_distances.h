#ifndef TITAN_NOCAL_CHARGED_ATOM_DISTANCES_H
#define TITAN_NOCAL_CHARGED_ATOM_DISTANCES_H
// -charged_atom_dis: the distances from molecule 1's centre of mass to each
// of its charged atoms (|q| > 1e-7), collected over the frames of an input.
// When the input ends their distributions are written (percent per bin):
// <stem>_distributionChargedAtoms.dat for all charged atoms, and
// <stem>_positive_... and <stem>_negative_... for each sign.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <string>
#include <vector>

class Multi_MOL;
struct arguments;

namespace titan_nocal {

class ChargedAtomDistances {
public:
  void begin_trajectory(const Trajectory &trajectory,
                        const std::string &suffix);
  void process_frame(const Multi_MOL &mol, std::size_t snap);
  void end_trajectory(const arguments &args);

private:
  Trajectory trajectory_;
  std::string suffix_;
  std::vector<double> all_, positive_, negative_;
};

} // namespace titan_nocal

#endif

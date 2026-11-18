#ifndef TITAN_NOCAL_RESIDUE_COM_DISTANCE_H
#define TITAN_NOCAL_RESIDUE_COM_DISTANCE_H
// -resdis2com: the distance from each residue's centre of mass to molecule
// 1's centre of mass (both over all atoms). Mean and SD over the frames of
// an input go to <stem>_distances_res_2COM.dat when the input ends.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <string>
#include <vector>

class Multi_MOL;

namespace titan_nocal {

class ResidueComDistance {
public:
  void begin_trajectory(const Trajectory &trajectory,
                        const std::string &suffix);
  void process_frame(Multi_MOL &mol, std::size_t snap);
  void end_trajectory();

private:
  Trajectory trajectory_;
  std::string suffix_;
  std::vector<std::string> names_;
  std::vector<double> mean_, m2_; // running mean and sum of squared deviations
  std::size_t frames_{};
};

} // namespace titan_nocal

#endif

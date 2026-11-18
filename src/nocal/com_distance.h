#ifndef TITAN_NOCAL_COM_DISTANCE_H
#define TITAN_NOCAL_COM_DISTANCE_H
// -com-dis: per frame, the distance between the centres of mass of molecules
// 1 and 2 (its x, y or z component with -com-dis x|y|z), then the smallest
// |x|, |y| and |z| separation between molecule 1's centre and a molecule-2
// atom. One row per frame in <folder>/out_dis_com.dat, in both modes.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <fstream>
#include <string>

class Multi_MOL;
struct arguments;

namespace titan_nocal {

class ComDistance {
public:
  void begin_trajectory(const arguments &args, const Trajectory &trajectory,
                        const std::string &file_name);
  void process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                     std::size_t snap, double time);
  void end_trajectory();

private:
  std::ofstream out_;
  int mode_{0};
};

} // namespace titan_nocal

#endif

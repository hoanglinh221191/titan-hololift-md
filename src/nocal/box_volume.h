#ifndef TITAN_NOCAL_BOX_VOLUME_H
#define TITAN_NOCAL_BOX_VOLUME_H
// -cal-volume: the box volume of every frame. When the run ends each result
// folder gets volume.dat (one row per frame, inputs that share the folder in
// input order) and stat_volumes.dat lists mean, SD and 95% CI per folder.

#include "nocal/nocal_trajectory.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

class Multi_MOL;
struct arguments;

namespace titan_nocal {

class BoxVolume {
public:
  void begin_trajectory(const Trajectory &trajectory);
  void process_frame(const Multi_MOL &mol, std::size_t snap, double time);
  void finish_run(const arguments &args) const;

private:
  struct Input {
    std::size_t index{};
    std::string stem;
    std::vector<std::pair<double, double>> rows; // time, volume (A^3)
  };
  struct Folder {
    std::string path;
    std::vector<Input> inputs;
  };
  std::vector<Folder> folders_; // in the order of their first input
  std::size_t current_{};       // folder of the open input
};

} // namespace titan_nocal

#endif

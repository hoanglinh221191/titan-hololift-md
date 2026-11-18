#ifndef TITAN_NOCAL_ANALYSES_H
#define TITAN_NOCAL_ANALYSES_H
// The analyses of molecule 1 that -nocal runs, plus -com-dis, which the
// energy mode shares. Each works on one frame at a time; statistics over an
// input or over the run are written when that input or the run ends.
//
// The engine calls begin_trajectory() once an input's topology and charges
// are ready, process_frame() for every analysed frame (snap indexes the
// loaded chunk), end_trajectory() when the input ends and finish_run() once.
// Each throws std::runtime_error when it cannot write its output.

#include "nocal/box_volume.h"
#include "nocal/charged_atom_distances.h"
#include "nocal/com_distance.h"
#include "nocal/dipole_moment.h"
#include "nocal/nocal_trajectory.h"
#include "nocal/residue_com_distance.h"
#include "nocal/water_dipole_field.h"

#include <cstddef>

class Multi_MOL;
class info;
struct arguments;

namespace titan_nocal {

class Analyses {
public:
  void begin_trajectory(const arguments &args, const ::info &names,
                        Multi_MOL &mol1, const Trajectory &trajectory);
  void process_frame(const arguments &args, Multi_MOL &mol1,
                     const Multi_MOL &mol2, std::size_t snap);
  void end_trajectory(const arguments &args);
  void finish_run(const arguments &args) const;

private:
  bool com_{}, dipole_{}, charged_{}, water_{}, residues_{}, volume_{};
  ComDistance com_distance_;
  DipoleMoment dipole_moment_;
  ChargedAtomDistances charged_atom_distances_;
  WaterDipoleField water_dipole_field_;
  ResidueComDistance residue_com_distance_;
  BoxVolume box_volume_;
};

} // namespace titan_nocal

#endif

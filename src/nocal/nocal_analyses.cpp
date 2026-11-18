#include "nocal/nocal_analyses.h"

#include "Multi_MOL.h"
#include "arguments.h"
#include "info.h"

#include <stdexcept>
#include <string>

namespace titan_nocal {

void Analyses::begin_trajectory(const arguments &args, const ::info &names,
                                Multi_MOL &mol1, const Trajectory &trajectory) {
  // Only -com-dis runs in the energy mode as well.
  com_ = args.com_dis_2mol != 0;
  dipole_ = args.flag_nocal && args.flag_cal_dipole_moment;
  charged_ = args.flag_nocal && args.flag_charged_atom_distance_dis;
  water_ = args.flag_nocal && args.flag_angle_dipole_efield;
  residues_ = args.flag_nocal && args.flag_cal_dis_res_2com;
  volume_ = args.flag_nocal && args.flag_cal_vol;

  if (com_)
    com_distance_.begin_trajectory(args, trajectory, names.getDisCom2Mol());
  if (dipole_)
    dipole_moment_.begin_trajectory(args, trajectory,
                                    names.getDipoleFilename());
  if (charged_)
    charged_atom_distances_.begin_trajectory(
        trajectory, names.getDistributionDistanceCharged());
  if (water_)
    water_dipole_field_.begin_trajectory(
        mol1, trajectory, names.getDistributionAngleDipoleEfield());
  if (residues_)
    residue_com_distance_.begin_trajectory(trajectory,
                                           names.getOutputDistanceRes2COM());
  if (volume_)
    box_volume_.begin_trajectory(trajectory);
}

void Analyses::process_frame(const arguments &args, Multi_MOL &mol1,
                             const Multi_MOL &mol2, std::size_t snap) {
  if (snap >= mol1.m_mol.size())
    return;
  // These read a charge (and the topology) for every atom of the frame.
  if ((dipole_ || charged_ || water_) && !mol1.analysis_frame_ready(snap))
    throw std::runtime_error(
        "-nocal: molecule 1 has " + std::to_string(mol1.m_mol[snap].size()) +
        " atoms in this frame but charges for " +
        std::to_string(mol1.parameters.size()) + " and topology entries for " +
        std::to_string(mol1.info.size()) +
        "; -dipole-moment, -charged_atom_dis and -angle_dipole_efield need "
        "all of them");
  const double time = args.output_time(mol1.m_mol[snap].time_of_snap);
  if (com_)
    com_distance_.process_frame(mol1, mol2, snap, time);
  if (dipole_)
    dipole_moment_.process_frame(mol1, snap, time);
  if (charged_)
    charged_atom_distances_.process_frame(mol1, snap);
  if (water_)
    water_dipole_field_.process_frame(args, mol1, snap);
  if (residues_)
    residue_com_distance_.process_frame(mol1, snap);
  if (volume_)
    box_volume_.process_frame(mol1, snap, time);
}

void Analyses::end_trajectory(const arguments &args) {
  if (com_)
    com_distance_.end_trajectory();
  if (dipole_)
    dipole_moment_.end_trajectory();
  if (charged_)
    charged_atom_distances_.end_trajectory(args);
  if (water_)
    water_dipole_field_.end_trajectory(args);
  if (residues_)
    residue_com_distance_.end_trajectory();
}

void Analyses::finish_run(const arguments &args) const {
  if (args.flag_nocal && args.flag_cal_vol)
    box_volume_.finish_run(args);
}

} // namespace titan_nocal

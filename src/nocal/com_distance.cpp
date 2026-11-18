#include "nocal/com_distance.h"

#include "Multi_MOL.h"
#include "arguments.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace titan_nocal {

void ComDistance::begin_trajectory(const arguments &args,
                                   const Trajectory &trajectory,
                                   const std::string &file_name) {
  mode_ = args.com_dis_2mol;
  const std::string path = trajectory.folder + "/" + file_name;
  // Inputs without their own folder share one file, in input order.
  out_.open(path, trajectory.first_in_folder ? std::ios::out
                                             : std::ios::out | std::ios::app);
  if (!out_)
    throw std::runtime_error("cannot write " + path);
  if (trajectory.first_in_folder) {
    const char *component = mode_ == 1   ? "x"
                            : mode_ == 2 ? "y"
                            : mode_ == 3 ? "z"
                                         : "xyz";
    out_ << "# Time (" << args.tu << ")  d_COM (A, " << component
         << ")  min|x-x1|  min|y-y1|  min|z-z1| (A, molecule-2 atoms to the "
            "molecule-1 centre of mass)\n";
  }
  out_ << "# input " << trajectory.index + 1 << ": " << trajectory.stem
       << '\n';
}

void ComDistance::process_frame(const Multi_MOL &mol1, const Multi_MOL &mol2,
                                std::size_t snap, double time) {
  if (!out_.is_open() || snap >= mol2.m_mol.size())
    return;
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  double com1[3]{}, com2[3]{};
  const bool centres = mol1.com_mol(snap, com1) == 0 &&
                       mol2.com_mol(snap, com2) == 0;

  double distance = nan;
  if (centres) {
    const double dx = com1[0] - com2[0];
    const double dy = com1[1] - com2[1];
    const double dz = com1[2] - com2[2];
    distance = mode_ == 1   ? std::fabs(dx)
               : mode_ == 2 ? std::fabs(dy)
               : mode_ == 3 ? std::fabs(dz)
                            : std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  double nearest[3]{nan, nan, nan};
  const MOL &atoms = mol2.m_mol[snap];
  if (centres)
    for (std::size_t j = 0; j < atoms.pos.x.size(); ++j) {
      const double d[3]{std::fabs(atoms.pos.x[j] - com1[0]),
                        std::fabs(atoms.pos.y[j] - com1[1]),
                        std::fabs(atoms.pos.z[j] - com1[2])};
      for (int k = 0; k < 3; ++k)
        if (!(nearest[k] <= d[k]))
          nearest[k] = d[k];
    }

  out_ << std::setw(10) << std::left << time << std::setw(12) << std::right
       << distance << std::setw(12) << nearest[0] << std::setw(12)
       << nearest[1] << std::setw(12) << nearest[2] << '\n';
}

void ComDistance::end_trajectory() {
  if (out_.is_open())
    out_.close();
}

} // namespace titan_nocal

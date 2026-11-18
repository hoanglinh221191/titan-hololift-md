#include "nocal/dipole_moment.h"

#include "Multi_MOL.h"
#include "arguments.h"

#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace titan_nocal {

void DipoleMoment::begin_trajectory(const arguments &args,
                                    const Trajectory &trajectory,
                                    const std::string &suffix) {
  const std::string unit_path = trajectory.file(suffix);
  const std::string raw_path = trajectory.file("_notnormalized_" + suffix);
  unit_.open(unit_path, std::ios::out);
  raw_.open(raw_path, std::ios::out);
  if (!unit_ || !raw_)
    throw std::runtime_error("cannot write " + (unit_ ? raw_path : unit_path));
  unit_ << "# Time (" << args.tu << ")     M_x       M_y      M_z\n";
  raw_ << "# Time (" << args.tu << ")     M_x       M_y      M_z       modul\n";
}

void DipoleMoment::process_frame(const Multi_MOL &mol, std::size_t snap,
                                 double time) {
  if (!unit_.is_open() || snap >= mol.m_mol.size())
    return;
  const MOL &frame = mol.m_mol[snap];
  const std::size_t atoms = frame.size();
  double centre[3]{};
  for (std::size_t j = 0; j < atoms; ++j) {
    centre[0] += frame.pos.x[j];
    centre[1] += frame.pos.y[j];
    centre[2] += frame.pos.z[j];
  }
  for (double &c : centre)
    c /= 1. * atoms;
  double m[3]{};
  for (std::size_t j = 0; j < atoms; ++j) {
    const double q = mol.parameters[j].get_charge();
    m[0] += q * (frame.pos.x[j] - centre[0]);
    m[1] += q * (frame.pos.y[j] - centre[1]);
    m[2] += q * (frame.pos.z[j] - centre[2]);
  }
  const double length = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);

  unit_ << std::setw(15) << std::left << time << std::right;
  for (const double c : m)
    unit_ << std::setw(16) << (length < 10e-13 ? 0.0 : c / length);
  unit_ << '\n';
  raw_ << std::setw(15) << std::left << time << std::right << std::setw(16)
       << m[0] << std::setw(16) << m[1] << std::setw(16) << m[2]
       << std::setw(16) << length << '\n';
}

void DipoleMoment::end_trajectory() {
  if (unit_.is_open())
    unit_.close();
  if (raw_.is_open())
    raw_.close();
}

} // namespace titan_nocal

#include <string_view>
using std::string_view;
#include "Multi_MOL.h"
#include "info.h"
#include <fstream>
#include <iomanip>
#include <thread>
using namespace std;

int Multi_MOL::impl_cal_dipole_moment(const arguments &Arguments,
                                      vector<force> &dipoles, size_t begin,
                                      size_t end) {
  for (size_t i{begin}; i < end; ++i) {
    force dipole_temp, center;
    for (size_t j{}; j < this->m_mol[i].size(); ++j) {
      center.Fx += this->m_mol[i].pos.x[j];
      center.Fy += this->m_mol[i].pos.y[j];
      center.Fz += this->m_mol[i].pos.z[j];
    }
    center /= 1. * this->m_mol[i].size();
    for (size_t j{}; j < this->m_mol[i].size(); ++j) {
      dipole_temp.Fx += this->parameters[j].get_charge() *
                        (this->m_mol[i].pos.x[j] - center.Fx);
      dipole_temp.Fy += this->parameters[j].get_charge() *
                        (this->m_mol[i].pos.y[j] - center.Fy);
      dipole_temp.Fz += this->parameters[j].get_charge() *
                        (this->m_mol[i].pos.z[j] - center.Fz);
    }
    dipoles.emplace_back(dipole_temp);
  }
  return 0;
}

int Multi_MOL::cal_dipole_moment(const ::info &info_output,
                                 const arguments &Arguments,
                                 std::string_view prefix_name) {
  fstream output_dipole(
      std::string(prefix_name) + info_output.getDipoleFilename(), ios::out),
      output_dipole_notnormalized(std::string(prefix_name) + "_notnormalized_" +
                                      info_output.getDipoleFilename(),
                                  ios::out);
  output_dipole << "# Time (" << Arguments.tu
                << ")     M_x       M_y      M_z\n";
  output_dipole_notnormalized << "# Time (" << Arguments.tu
                              << ")     M_x       M_y      M_z       modul\n";

  // vector<force> dipole_m;
  // impl_cal_dipole_moment(Arguments, dipole_m, 0, m_mol.size());
  vector<vector<force>> dipoles{Arguments.nt};

  vector<thread> impl_dipole(Arguments.nt);
  for (size_t i{}; i < impl_dipole.size(); ++i)
    impl_dipole[i] =
        thread(&Multi_MOL::impl_cal_dipole_moment, &(*this), ref(Arguments),
               ref(dipoles[i]), i * m_mol.size() / Arguments.nt,
               (i + 1) * m_mol.size() / Arguments.nt);
  for (auto &t : impl_dipole)
    t.join();

  vector<force> dipole_m;
  for (auto &d : dipoles)
    for (auto &dd : d)
      dipole_m.emplace_back(dd);

  for (size_t i{}; i < dipole_m.size(); ++i) {
    if (dipole_m[i].modul() < 10e-13)
      output_dipole << setw(15) << left << this->m_mol[i].time_of_snap << right
                    << setw(16) << 0.0 << setw(16) << 0.0 << setw(16) << 0.0
                    << '\n';
    else
      output_dipole << setw(15) << left << this->m_mol[i].time_of_snap << right
                    << setw(16) << dipole_m[i].Fx / dipole_m[i].modul()
                    << setw(16) << dipole_m[i].Fy / dipole_m[i].modul()
                    << setw(16) << dipole_m[i].Fz / dipole_m[i].modul() << '\n';

    output_dipole_notnormalized
        << setw(15) << left << this->m_mol[i].time_of_snap << right << setw(16)
        << dipole_m[i].Fx << setw(16) << dipole_m[i].Fy << setw(16)
        << dipole_m[i].Fz << setw(16) << dipole_m[i].modul() << '\n';
  }
  return 0;
}

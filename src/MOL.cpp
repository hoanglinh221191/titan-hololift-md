#include "MOL.h"
#include "functions.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string_view>
using std::string_view;

using namespace std;

void MOL::get_centroid(std::span<double> com) const {
  com[0] = 0.;
  com[1] = 0.;
  com[2] = 0.;
  for (size_t i{}; i < pos.x.size(); ++i) {
    com[0] += pos.x[i];
    com[1] += pos.y[i];
    com[2] += pos.z[i];
  }
  com[0] /= pos.x.size();
  com[1] /= pos.y.size();
  com[2] /= pos.z.size();
}

int MOL::cal_dis_charged_atoms(vector<double> &distances,
                               vector<double> &distances_negative_charges,
                               vector<double> &distances_positive_charges,
                               const Multi_MOL_parameters &parameters,
                               std::span<const double> com) const {
  for (size_t i{}; i < pos.x.size(); ++i) {
    if (fabs(parameters[i].get_charge()) > 0.0000001) {
      double d_temp{sqrt(square(pos.x[i] - com[0]) + square(pos.y[i] - com[1]) +
                         square(pos.z[i] - com[2]))};

      distances.emplace_back(d_temp);

      if (parameters[i].get_charge() > 0.0000001)
        distances_positive_charges.emplace_back(d_temp);
      else
        distances_negative_charges.emplace_back(d_temp);
    }
  }
  return 0;
}

int MOL::cal_electric_field_1atom(const vector<mol_info> &info,
                                  const arguments &Arguments,
                                  const Multi_MOL_parameters &parameters,
                                  force &electric_field_vector,
                                  const vector<size_t> &list_heavy_atoms,
                                  size_t pos_o_water) const {
  for (size_t i{}; i < list_heavy_atoms.size(); ++i) {
    auto &atom = list_heavy_atoms[i];
    if (atom == pos_o_water)
      continue;
    const double temp_charge{138.935457839 * parameters[atom].get_charge()};
    const double d{0.1 * sqrt(square(pos.x[atom] - pos.x[pos_o_water]) +
                              square(pos.y[atom] - pos.y[pos_o_water]) +
                              square(pos.z[atom] - pos.z[pos_o_water]))};
    const double par_temp{temp_charge / pow(d, 3)};

    electric_field_vector +=
        force(0.1 * (pos.x[pos_o_water] - pos.x[atom]) * par_temp,
              0.1 * (pos.y[pos_o_water] - pos.y[atom]) * par_temp,
              0.1 * (pos.z[pos_o_water] - pos.z[atom]) * par_temp);
  }
  return 0;
}

int MOL::find_surface_water(const vector<size_t> &list_waters,
                            const vector<size_t> &list_heavy_atoms,
                            vector<size_t> &list_surface) const {
  for (size_t i{}; i < list_waters.size(); ++i) {
    bool flag_surface{false};
    for (size_t j{}; j < list_heavy_atoms.size(); ++j) {
      if (dis_between2_atoms(list_waters[i], list_heavy_atoms[j]) <= 3.5000) {
        flag_surface = true;
        break;
      }
    }
    if (flag_surface == true)
      list_surface.emplace_back(list_waters[i]);
  }
  return 0;
}

int MOL::find_sphere_water(const vector<size_t> &list_waters,
                           const vector<size_t> &list_heavy_atoms,
                           vector<size_t> &list_surface, double d_crit) const {
  double cx{}, cy{}, cz{};
  for (size_t i{}; i < list_heavy_atoms.size(); ++i) {
    cx += pos.x[list_heavy_atoms[i]];
    cy += pos.y[list_heavy_atoms[i]];
    cz += pos.z[list_heavy_atoms[i]];
  }
  cx /= list_heavy_atoms.size();
  cy /= list_heavy_atoms.size();
  cz /= list_heavy_atoms.size();

  for (size_t i{}; i < list_waters.size(); ++i) {
    if (dis_between_atom_and_point(list_waters[i], cx, cy, cz) <= d_crit)
      list_surface.emplace_back(list_waters[i]);
  }
  return 0;
}

int MOL::cal_dipoles_specific_atoms(const vector<size_t> &list_atoms,
                                    const Multi_MOL_parameters &parameters,
                                    const vector<mol_info> &info,
                                    vector<force> &dipoles_out,
                                    std::string_view atoms_type) const {
  if (atoms_type == "water")
    cal_dipoles_water_molecules(list_atoms, parameters, info, dipoles_out);
  else
    return 1;
  return 0;
}

int MOL::cal_dipoles_water_molecules(const vector<size_t> &list_atoms,
                                     const Multi_MOL_parameters &parameters,
                                     const vector<mol_info> &info,
                                     vector<force> &dipoles_out) const {
  for (size_t i{}; i < list_atoms.size(); ++i) {
    auto &atom = list_atoms[i];
    if (atom + 2 < pos.x.size() && info[atom + 1].get_atom_type()[0] == 'H' &&
        info[atom + 2].get_atom_type()[0] == 'H') {
      dipoles_out.emplace_back(
          force(pos.x[atom] * parameters[atom].get_charge() +
                    pos.x[atom + 1] * parameters[atom + 1].get_charge() +
                    pos.x[atom + 2] * parameters[atom + 2].get_charge(),
                pos.y[atom] * parameters[atom].get_charge() +
                    pos.y[atom + 1] * parameters[atom + 1].get_charge() +
                    pos.y[atom + 2] * parameters[atom + 2].get_charge(),
                pos.z[atom] * parameters[atom].get_charge() +
                    pos.z[atom + 1] * parameters[atom + 1].get_charge() +
                    pos.z[atom + 2] * parameters[atom + 2].get_charge()));
    } else
      continue;
  }
  return 0;
}

int MOL::helper_cal_electric_field(const vector<mol_info> &info,
                                   const arguments &Arguments,
                                   const Multi_MOL_parameters &parameters,
                                   vector<force> &elec_field,
                                   const vector<size_t> &list_heavy_atoms,
                                   const vector<size_t> &pos_o_waters,
                                   size_t begin, size_t end) const {
  for (size_t i{begin}; i < end; ++i) {
    if (i >= elec_field.size())
      elec_field.emplace_back(force());

    cal_electric_field_1atom(info, Arguments, parameters, elec_field[i],
                             list_heavy_atoms, pos_o_waters[i]);
  }
  return 0;
}

int MOL::cal_electric_field(const vector<mol_info> &info,
                            const arguments &Arguments,
                            const Multi_MOL_parameters &parameters,
                            const vector<size_t> &list_heavy_atoms,
                            const vector<size_t> &pos_o_waters,
                            vector<force> &elec_fields) const {
  vector<thread> threads_cal(Arguments.nt);

  for (size_t i{}; i < threads_cal.size(); ++i)
    threads_cal[i] =
        thread(&MOL::helper_cal_electric_field, this, ref(info), ref(Arguments),
               ref(parameters), ref(elec_fields), std::cref(list_heavy_atoms),
               std::cref(pos_o_waters), i * pos_o_waters.size() / Arguments.nt,
               (i + 1) * pos_o_waters.size() / Arguments.nt);
  for (auto &t : threads_cal)
    t.join();
  return 0;
}

int MOL::cal_angle_dipoles_elecfield(const vector<mol_info> &info,
                                     const arguments &Arguments,
                                     vector<force> &elec_fields,
                                     const Multi_MOL_parameters &parameters,
                                     const vector<size_t> &list_heavy_atoms,
                                     const vector<size_t> &pos_o_waters,
                                     vector<double> &cos_out) const {
  vector<force> dipoles;
  cal_dipoles_specific_atoms(pos_o_waters, parameters, info, dipoles, "water");

  size_t SIZE{dipoles.size() >= elec_fields.size() ? elec_fields.size()
                                                   : dipoles.size()};

  for (size_t i{}; i < SIZE; ++i) {
    cos_out.emplace_back((dipoles[i].Fx * elec_fields[i].Fx +
                          dipoles[i].Fy * elec_fields[i].Fy +
                          dipoles[i].Fz * elec_fields[i].Fz) /
                         (dipoles[i].modul() * elec_fields[i].modul()));
  }
  return 0;
}

int MOL::cal_min_max_mol(double &min_x, double &max_x, double &min_y,
                         double &max_y, double &min_z, double &max_z) const {
  min_x = 10e13;
  min_y = 10e13;
  min_z = 10e13;
  max_x = -10e13;
  max_y = -10e13;
  max_z = -10e13;
  for (size_t i{}; i < pos.x.size(); ++i) {
    if (max_x < pos.x[i])
      max_x = pos.x[i];
    if (min_x > pos.x[i])
      min_x = pos.x[i];
    if (max_y < pos.y[i])
      max_y = pos.y[i];
    if (min_y > pos.y[i])
      min_y = pos.y[i];
    if (max_z < pos.z[i])
      max_z = pos.z[i];
    if (min_z > pos.z[i])
      min_z = pos.z[i];
  }
  return 0;
}

double MOL::cal_minimum_distance_2mols(const MOL &other,
                                       const arguments &Arguments) const {
  double distance{10e12};
  double min_x_this{}, min_x_other{}, max_x_this{}, max_x_other{}, min_y_this{},
      min_y_other{}, max_y_this{}, max_y_other{}, min_z_this{}, min_z_other{},
      max_z_this{}, max_z_other{};

  // this->cal_min_max_mol(min_x_this, max_x_this, min_y_this, max_y_this,
  // min_z_this, max_z_this); other.cal_min_max_mol(min_x_other, max_x_other,
  // min_y_other, max_y_other, min_z_other, max_z_other);

  min_x_this -= Arguments.d_crit_minmax_mol;
  max_x_this += Arguments.d_crit_minmax_mol;
  min_y_this -= Arguments.d_crit_minmax_mol;
  max_y_this += Arguments.d_crit_minmax_mol;
  min_z_this -= Arguments.d_crit_minmax_mol;
  max_z_this += Arguments.d_crit_minmax_mol;

  min_x_other -= Arguments.d_crit_minmax_mol;
  max_x_other += Arguments.d_crit_minmax_mol;
  min_y_other -= Arguments.d_crit_minmax_mol;
  max_y_other += Arguments.d_crit_minmax_mol;
  min_z_other -= Arguments.d_crit_minmax_mol;
  max_z_other += Arguments.d_crit_minmax_mol;

  const double ddX{0.7 * (max_x_this - min_x_this)},
      ddY{0.7 * (max_y_this - min_y_this)},
      ddZ{0.7 * (max_z_this - min_z_this)};

  if (((min_x_other >= min_x_this && min_x_other <= max_x_this) ||
       (max_x_other >= min_x_this && max_x_other <= max_x_this)) &&
      ((min_y_other >= min_y_this && min_y_other <= max_y_this) ||
       (max_y_other >= min_y_this && max_y_other <= max_y_this)) &&
      ((min_z_other >= min_z_this && min_z_other <= max_z_this) ||
       (max_z_other >= min_z_this && max_z_other <= max_z_this))) {
    for (size_t i = 0; i < pos.x.size(); ++i)
      for (size_t j = 0; j < other.pos.x.size(); ++j) {
        if (fabs(pos.x[i] - other.pos.x[j]) > 0.7 * Box.get_a() ||
            fabs(pos.y[i] - other.pos.y[j]) > 0.7 * Box.get_b() ||
            fabs(pos.z[i] - other.pos.z[j]) > 0.7 * Box.get_c())
          continue;
        double dis{};
        if (Arguments.pbc.pbc_mem == false)
          dis = sqrt(square(pos.x[i] - other.pos.x[j]) +
                     square(pos.y[i] - other.pos.y[j]) +
                     square(pos.z[i] - other.pos.z[j]));
        else
          dis = fabs(pos.z[i] - other.pos.z[j]) +
                fabs(pos.x[i] - other.pos.x[j]) +
                fabs(pos.y[i] - other.pos.y[j]);
        if (dis < distance)
          distance = dis;
      }
  }

  return distance;
}

double MOL::cal_minimum_distance_mol_atom(const MOL &other,
                                          const arguments &Arguments,
                                          size_t &index) const {
  double distance{10e12};
  double atom_2_x{other.pos.x[index]}, atom_2_y{other.pos.y[index]},
      atom_2_z{other.pos.z[index]};
  double center[3]{};

  this->get_centroid(center);
  distance = sqrt(square(center[0] - atom_2_x) + square(center[1] - atom_2_y) +
                  square(center[2] - atom_2_z));
  return distance;
}

int MOL::translate(double dx, double dy, double dz) {
  for (size_t i = 0; i < pos.x.size(); ++i) {
    pos.x[i] += dx;
    pos.y[i] += dy;
    pos.z[i] += dz;
  }
  return 0;
}

bool MOL::mindist_mol(const MOL &lig, arguments &Arguments,
                      de_arg &DE_arg) const {
  if (Arguments.de.flag_de_auto_term == true && DE_arg.flag_old_vec == true) {
    double mindist{10000.};
    for (size_t i = 0; i < pos.x.size(); ++i)
      for (size_t j = 0; j < lig.pos.x.size(); ++j) {
        double dx = pos.x[i] - lig.pos.x[j];
        double dy = pos.y[i] - lig.pos.y[j];
        double dz = pos.z[i] - lig.pos.z[j];
        double d = sqrt(dx * dx + dy * dy + dz * dz);
        if (d < mindist)
          mindist = d;
      }
    cout << mindist << '\n';
    if (mindist > Arguments.d_standard)
      return true;
    else
      return false;
  } else
    return false;
}

void MOL::mutate_to_alanine(int res_id, const std::vector<mol_info> &info,
                            Multi_MOL_parameters &parameters) {
  backup_params.resize(parameters.size());
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i].get_resnr() == res_id) {
      std::string atom_name = info[i].get_atom();
      if (atom_name != "N" && atom_name != "H" && atom_name != "CA" &&
          atom_name != "HA" && atom_name != "C" && atom_name != "O" &&
          atom_name != "CB" && atom_name != "HB1" && atom_name != "HB2" &&
          atom_name != "HB3") {

        // Backup
        backup_params[i].charge = parameters[i].get_charge();
        backup_params[i].epsilon = parameters[i].get_epsilon();
        backup_params[i].sigma = parameters[i].get_sigma();
        backup_params[i].radius = parameters[i].get_radius();

        // Mutate
        parameters[i].set_charge(0.0);
        parameters[i].set_epsilon_sigma_radius(0.0, 0.0, 0.001);
      }
    }
  }
}

void MOL::restore_mutation(int res_id, const std::vector<mol_info> &info,
                           Multi_MOL_parameters &parameters) {
  for (size_t i = 0; i < info.size(); ++i) {
    if (info[i].get_resnr() == res_id) {
      std::string atom_name = info[i].get_atom();
      if (atom_name != "N" && atom_name != "H" && atom_name != "CA" &&
          atom_name != "HA" && atom_name != "C" && atom_name != "O" &&
          atom_name != "CB" && atom_name != "HB1" && atom_name != "HB2" &&
          atom_name != "HB3") {

        // Restore
        parameters[i].set_charge(backup_params[i].charge);
        parameters[i].set_epsilon_sigma_radius(backup_params[i].epsilon,
                                               backup_params[i].sigma,
                                               backup_params[i].radius);
      }
    }
  }
}

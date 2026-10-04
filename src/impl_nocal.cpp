#include "Multi_MOL.h"
#include "arguments.h"
#include "functions.h"
#include "groups.h"
#include "h_groups.h"
#include "titan_executor.h"
#include <fstream>
#include <iostream>
#include <thread>


int impl_nocal(Multi_MOL &pro_in, Multi_MOL &lig, fstream &log,
               arguments &Arguments, info &info_output, TitanExecutor &executor,
               amino_acid &amine, para &parameter, folders &paths,
               vector<string> &input_line, h_groups &groups_1,
               h_groups &groups_2, vector<vector<size_t>> &Hbonds_vector,
               int &input_name, vector<vector<double>> &Volumes) {
  log << "Nocal mode is turned on!\n";
  cout << "Nocal mode is turned on!\n";
  log << "Box type: " << pro_in.m_mol[0].Box.get_box_type() << '\n';

  pro_in.take_charge(amine, parameter, pro_in.m_mol[0].size(), 0ull,
                     Arguments.force_f, 0, &Arguments, &executor, "mol1");

  log << "Charge: " << pro_in.total_charge() << "e\n";

  if (Arguments.pbc.solv_pbctopo) {
    log << "Solving pbc...\n";
    log << "Will use pbctopo (VIBE: Virtual Intact Block Extraction)!\n";
    cout << "Will use pbctopo (VIBE: Virtual Intact Block Extraction)!\n";
    pro_in.solve_pbc_pbctopo(lig, Arguments, nullptr);
    pro_in.align_to_box();
  } else if (Arguments.pbc.solv_pbcn) {
    log << "Solving pbc...\n";
    log << "Will use the NEW strict branchless pbcn algorithm!\n";
    cout << "Will use the NEW strict branchless pbcn algorithm!\n";
    pro_in.solv_pbc_pbcn(lig, Arguments);
    pro_in.align_to_box();
  } else if (Arguments.pbc.solv_pbc_int && Arguments.pbc.chain_ref > -2) {
    log << "Solving pbc...\n";
    log << "Will use new pbc algorithm!\n";
    cout << "Will use new pbc algorithm!\n";
    if (Arguments.pbc.solv_pbc_int2 == 1)
      log << "pbc algorithm with minimum distance criteria\n";
    if (Arguments.pbc.solv_pbc_int3 == 1)
      log << "pbc algorithm with minimum total distance criteria\n";
    pro_in.solve_pbc_new(lig, Arguments, amine);
    pro_in.align_to_box();
  }

  if (Arguments.pbc.solv_pbc_clust == true && !Arguments.pbc.solv_pbc_int) {
    log << "Solving pbc ...\n";
    pro_in.solve_pbc_cluster(Arguments, info_output);
  }

  if (Arguments.com_dis_2mol != 0) {
    log << "Calculating distance between COMs and minimum x/y/z ...";
    fstream output_dis_2mol_file(
        paths.get_path(input_name - Arguments.b_input) +
            info_output.getDisCom2Mol(),
        ios::out);
    pro_in.com_dis_2mol(lig, Arguments, info_output, output_dis_2mol_file);
    log << " Done!\n";
  }

  if (Arguments.flag_cal_dipole_moment == true)
    pro_in.cal_dipole_moment(
        info_output, Arguments,
        Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4));

  if (Arguments.flag_charged_atom_distance_dis == true)
    pro_in.cal_distribution_distance_charged_atoms(Arguments, info_output);

  if (Arguments.flag_angle_dipole_efield == true)
    pro_in.cal_angle_dipole_water_elecField(Arguments, info_output);

  if (Arguments.flag_cal_dis_res_2com == true) {
    string prefix{paths.get_path(input_name - Arguments.b_input) == "."
                      ? info_output.getPdbFileName().substr(
                            0, info_output.getPdbFileName().length() - 4)
                      : paths.get_path(input_name - Arguments.b_input)};
    fstream output_dis_res2com_file(
        prefix + info_output.getOutputDistanceRes2COM(), ios::out);
    pro_in.cal_dis_res_2_com(Arguments, output_dis_res2com_file);
  }

  if (Arguments.flag_cal_vol == true) {
    vector<double> volumes_temp;
    pro_in.cal_volumes(volumes_temp);
    Volumes.emplace_back(volumes_temp);
  }

  log << "Number of snaps: " << pro_in.m_mol.size() << '\n';

  for (size_t num_snap{}; num_snap < pro_in.m_mol.size(); ++num_snap) {
    if (num_snap % 100 == 0)
      log << num_snap << "   ";
    if (num_snap % 2000 == 0)
      log << '\n';

    if (Arguments.flag_h_bond == 1) {
      MOL temp_mol(pro_in.m_mol[num_snap]);
      find_h_bonds(pro_in.m_mol[num_snap], pro_in.m_mol[0], temp_mol, temp_mol,
                   groups_1, groups_2, Arguments, num_snap, pro_in.info,
                   pro_in.info, Hbonds_vector);
    }
    if (Arguments.flag_pqr == 1 && !pro_in.m_mol.empty()) {
      write_pqr(pro_in, lig, input_line, input_name, num_snap, Arguments,
                paths);
    }
    write_pdb(pro_in, lig, Arguments, num_snap, paths, input_line, input_name);
  }

  if (Arguments.flag_h_bond == 1) {
    finalize_hbond_output_files();
  }
  if (Arguments.opdb) {
    finalize_pdb_output_stream();
  }

  return 0;
}

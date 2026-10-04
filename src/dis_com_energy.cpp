#include<iostream>
#include<fstream>
#include<algorithm>
#include<iomanip>
#include<thread>
#include"functions.h"
#include"uti_func.h"

int init_file(const folders& paths, fstream& output, fstream& output_elec, fstream& output_vdw, const size_t num_snap, int input_name,
	const info& out_info, arguments& Arguments);

inline double dis(std::span<double> com1, std::span<double> com2, arguments& Arguments)
{
	if (Arguments.com_dis_energy_2mol == 0)
		return sqrt(square(com1[0] - com2[0]) + square(com1[1] - com2[1]) + square(com1[2] - com2[2]));
	else if (Arguments.com_dis_energy_2mol == 1)
		return com1[0] - com2[0];
	else if (Arguments.com_dis_energy_2mol == 2)
		return com1[1] - com2[1];
	else if (Arguments.com_dis_energy_2mol == 3)
		return com1[2] - com2[2];
	else return 0.;
}

int dis_com_2mol(const Multi_MOL& mol1, const Multi_MOL& mol2, const size_t num_snap, const size_t NUM_snap, arguments& Arguments, int input_name
	, const folders& paths, double E_E, double E_V, const info& out_info)
{
	double com_mol1[3]{}, com_mol2[3]{};

	if (Arguments.nt > 1)
	{
		thread t1([&mol1, num_snap, &com_mol1](){ mol1.com_mol(num_snap, std::span<double>(com_mol1, 3)); });
		thread t2([&mol2, num_snap, &com_mol2](){ mol2.com_mol(num_snap, std::span<double>(com_mol2, 3)); });
		t1.join(); t2.join();
	}
	else
	{
		mol1.com_mol(num_snap, std::span<double>(com_mol1, 3));
		mol2.com_mol(num_snap, com_mol2);
	}

	fstream output_total, output_elec, output_vdw;
	init_file(paths, output_total, output_elec, output_vdw, num_snap, input_name, out_info, Arguments);


	const double distance{ dis(com_mol1,com_mol2,Arguments) };

	output_total << setw(17) << fixed << distance << " " << setw(17) << fixed << (E_E + E_V) / Arguments.eu_par<< '\n';
	output_elec << setw(17) << fixed << distance << " " << setw(17) << fixed << E_E / Arguments.eu_par << '\n';
	output_vdw << setw(17) << fixed << distance << " " << setw(17) << fixed << E_V / Arguments.eu_par << '\n';

	output_total.close();
	output_elec.close();
	output_vdw.close();
	return 0;
}



int init_file(const folders& paths, fstream& output, fstream& output_elec, fstream& output_vdw, const size_t num_snap, int input_name,
	const info& out_info, arguments& Arguments)
{
	output.open(paths.get_path(input_name - Arguments.b_input) + "/totalEnergy_" + out_info.getDisEner2Mol(), ios::out | ios::app);
	output_vdw.open(paths.get_path(input_name - Arguments.b_input) + "/vanderWaals_" + out_info.getDisEner2Mol(), ios::out | ios::app);
	output_elec.open(paths.get_path(input_name - Arguments.b_input) + "/electrostatic_" + out_info.getDisEner2Mol(), ios::out | ios::app);
	
	if (num_snap == 0ull)
	{
		output << "# Dis. of 2 coms (Angstrom) " << setw(25) << "Total Energy (" + Arguments.eu + ")\n";
		output_elec << "# Dis. of 2 coms (Angstrom) " << setw(30) << "Electrostatic Energy (" + Arguments.eu + ")\n";
		output_vdw << "# Dis. of 2 coms (Angstrom) " << setw(30) << "van der Waals Energy (" + Arguments.eu + ")\n";
	}

	return 0;
}
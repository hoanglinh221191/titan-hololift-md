#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include<iomanip>
#include"arguments.h"
#include"miscellaneous.h"
#include"folder.h"
#include"uti_func.h"
using namespace std;
int print_force_time(vector<force>& F_e, vector<force>& F_v, vector<double>& time_of_snap, vector<vector<force> >& F_e_vector, vector<vector<force> >& F_v_vector, arguments& Arguments, int input_name);
int out_force(vector<force>& F_e, vector<force>& F_v, vector<double>& time_of_snap, arguments& Arguments, int input_name, 
	vector<vector<force> >& F_e_vector, vector<vector<force> >& F_v_vector, vector<size_t>& size_snap)
{
	if (Arguments.fu == "pN")
	{
		for (size_t i{}; i < F_e.size(); ++i)
		{
			F_e[i] *= Arguments.fu_par;
			F_v[i] *= Arguments.fu_par;
		}
		for (size_t i{}; i< F_e_vector.size(); ++i)
			for (size_t j{}; j < F_e_vector[i].size(); ++j)
			{
				F_e_vector[i][j] *= Arguments.fu_par;
				F_v_vector[i][j] *= Arguments.fu_par;
			}
	}

	double elec_total{}, vdw_total{}, SD_e_temp{}, SD_v_temp{};
	for (int i{}; i < F_e.size(); i++)
	{
		elec_total = elec_total + F_e[i].modul() / size_snap[i];
		vdw_total = vdw_total + F_v[i].modul() / size_snap[i];
	}
	double elec_mean{ elec_total / F_e.size() };
	double vdw_mean{ vdw_total / F_v.size() };
	for (int i = 0; i < F_e.size(); i++)
	{
		SD_e_temp += square(F_e[i].modul() / size_snap[i] - elec_mean);
		SD_v_temp += square(F_v[i].modul() / size_snap[i] - vdw_mean);
	}
	SD_e_temp = sqrt(SD_e_temp / ((F_e.size())));
	SD_v_temp = sqrt(SD_v_temp / ((F_v.size())));

	fstream cal_out("cal_force.txt", ios::out);

	if ((Arguments.flag_pattern_input == true || Arguments.flag_listfile == true || Arguments.flag_all_folders == true) && input_name != Arguments.b_input)
	{
		for (size_t i{}; i < F_e.size(); ++i)
		{
			F_e[i] /= size_snap[i]*1.;
			F_v[i] /= size_snap[i]*1.;
		}
		cal_out << "Force is averaged from " << input_name - Arguments.b_input + 1 << " files.\n";
		//cout << "Force is averaged from " << input_name - Arguments.b_input + 1 << " files.\n";
		cal_out << "Average Electrostatic force: " << elec_mean << " " << Arguments.fu << ". Standard error: " << SD_e_temp / sqrt(F_e.size()) << " " << Arguments.fu << ". Confidence interval (95%): " << 1.96 * SD_e_temp / sqrt(F_e.size()) << " " << Arguments.fu << '\n';
		cal_out << "Average Van der waals force: " << vdw_mean << " " << Arguments.fu << ". Standard error: " << SD_v_temp / sqrt(F_v.size()) << " " << Arguments.fu << ". Confidence interval (95%): " << 1.96 * SD_v_temp / sqrt(F_v.size()) << " " << Arguments.fu << '\n';
	}
	else
	{
		cal_out << "Average Electrostatic force: " << elec_mean << " " << Arguments.fu << ". Standard deviation: " << SD_e_temp << " " << Arguments.fu << ". Confidence interval (95%): " << 1.96 * SD_e_temp / sqrt(F_e.size()) << " " << Arguments.fu << "\n";
		cal_out << "Average Van der waals force: " << vdw_mean << " " << Arguments.fu << ". Standard deviation: " << SD_v_temp << " " << Arguments.fu << ". Confidence interval (95%): " << 1.96 * SD_v_temp / sqrt(F_v.size()) << " " << Arguments.fu << "\n";
	}
	cal_out.close();
	//cout << "Average electrostatic force: " << elec_mean << " " << Arguments.fu << "\n";
	//cout << "Average Van der waals force: " << vdw_mean << " " << Arguments.fu << "\n";
	fstream out_xvg;
	out_xvg.open("cal_force.xvg", ios::out);
	out_xvg << "@    title \"Force\"\n";
	out_xvg << "@    xaxis label \"Time (" << Arguments.tu <<")\"\n";
	out_xvg << "@    yaxis label \"Force (" << Arguments.fu << ")\"\n";
	out_xvg << "@ s0 legend \"Time\"\n";
	out_xvg << "@ s1 legend \"Electrostatic\"\n";
	out_xvg << "@ s2 legend \"Van der Waal\"\n";
	out_xvg << "@ s3 legend \"Total Force\"\n";
	out_xvg << setw(6) << "#  " << Arguments.tu << " " << setw(9) << "Elec (" << Arguments.fu << ")" << setw(9) << "VdW (" << Arguments.fu << ")" << setw(9) << "Total (" << Arguments.fu << ")" << '\n';
	for (int i = 0; i < F_e.size(); ++i)
		out_xvg << setw(6) << time_of_snap[i] << setw(17) << F_e[i].modul() << setw(17) << F_v[i].modul() << setw(17) << norm3(F_e[i].Fx + F_v[i].Fx, F_e[i].Fy + F_v[i].Fy, F_e[i].Fz + F_v[i].Fz) << '\n';
	out_xvg.close();

	print_force_time(F_e, F_v, time_of_snap, F_e_vector, F_v_vector, Arguments, input_name);
	return 0;
}
int print_force_time(vector<force>& F_e, vector<force>& F_v, vector<double>& time_of_snap, vector<vector<force> >& F_e_vector, vector<vector<force> >& F_v_vector, arguments& Arguments, int input_name)
{
	fstream elec_out("electrostatic_force.dat", ios::out);
	fstream vdw_out("vanderwaals_force.dat", ios::out);
	fstream total_e("total_force.dat", ios::out);
	string error_u{ Arguments.flag_use_CI == true ? " CI(95 %) (" : "   SD (" };
	total_e << setw(6) << "#   " << Arguments.tu << setw(14) << "TotForce (" << Arguments.fu << ")" << setw(14) << error_u << Arguments.fu << ")\n";
	elec_out << setw(6) << "#   " << Arguments.tu << setw(14) << "ESForce (" << Arguments.fu << ")" << setw(14) << error_u << Arguments.fu << ")\n";
	vdw_out << setw(6) << "#   " << Arguments.tu << setw(14) << "VDWForce (" << Arguments.fu << ")" << setw(14) << error_u << Arguments.fu << ")\n";
	if ((Arguments.flag_pattern_input == true || Arguments.flag_listfile == true || Arguments.flag_all_folders == true) && input_name != Arguments.b_input)
	{
		vector<double> SD_e, SD_v, SD_t;
		for (size_t i{}; i < F_v_vector.size(); ++i)
		{
			double sde{}, sdv{}, sdt{};
			for (size_t j{}; j < F_v_vector[i].size(); ++j)
			{
				if (F_e_vector[j][i].modul() < 10e-9 || F_v_vector[j][i].modul() < 10e-11) { continue; }
				sde += square(F_e[i].modul() - F_e_vector[i][j].modul());
				sdv += square(F_v[i].modul() - F_v_vector[i][j].modul());
				sdt += square(norm3(F_e[i].Fx + F_v[i].Fx, F_e[i].Fy + F_v[i].Fy, F_e[i].Fz + F_v[i].Fz) - norm3(F_e_vector[i][j].Fx + F_v_vector[i][j].Fx, F_e_vector[i][j].Fy + F_v_vector[i][j].Fy, F_e_vector[i][j].Fz + F_v_vector[i][j].Fz));
			}
			SD_e.emplace_back(sqrt(sde / (F_e_vector[i].size())));
			SD_v.emplace_back(sqrt(sdv / (F_v_vector[i].size())));
			SD_t.emplace_back(sqrt(sdt / (F_v_vector[i].size())));
		}
		for (size_t i{}; i < F_e.size(); ++i)
		{
			double sde{ Arguments.flag_use_CI == true ? 1.96 * SD_e[i] / sqrt(F_e_vector[i].size()) : SD_e[i] },
				sdvdw{ Arguments.flag_use_CI == true ? 1.96 * SD_v[i] / sqrt(F_v_vector[i].size()) : SD_v[i] },
				sdtot{ Arguments.flag_use_CI == true ? 1.96 * SD_t[i] / sqrt(F_v_vector[i].size()) : SD_t[i] };

			elec_out << setw(6) << time_of_snap[i] << setw(17) << norm3(F_e[i].Fx, F_e[i].Fy, F_e[i].Fz)  
				<<setw(17)<<F_e[i].Fx<<setw(17)<<F_e[i].Fy<<setw(17)<<F_e[i].Fz << setw(17) << sde << '\n';
			vdw_out << setw(6) << time_of_snap[i] << setw(17) << norm3(F_v[i].Fx, F_v[i].Fy, F_v[i].Fz)
				<< setw(17) << F_v[i].Fx << setw(17) << F_v[i].Fy << setw(17) << F_v[i].Fz << setw(17) << sdvdw << '\n';
			total_e << setw(6) << time_of_snap[i] << setw(17) << norm3(F_e[i].Fx + F_v[i].Fx, F_e[i].Fy + F_v[i].Fy, F_e[i].Fz + F_v[i].Fz) 
				<< setw(17)<<F_e[i].Fx+F_v[i].Fx << setw(17)<<F_e[i].Fy+F_v[i].Fy<<setw(17)<<F_e[i].Fz+F_v[i].Fz<< setw(17) << sdtot << '\n';
		}
	}
	else
	{
		for (size_t i{}; i < F_e.size(); ++i)
		{
			elec_out << setw(6) << time_of_snap[i] << setw(17) << F_e[i].modul() 
				<< setw(17) << F_e[i].Fx << setw(17) << F_e[i].Fy << setw(17) << F_e[i].Fz << setw(17) << '\n';
			vdw_out << setw(6) << time_of_snap[i] << setw(17) << F_v[i].modul()
				<< setw(17) << F_v[i].Fx << setw(17) << F_v[i].Fy << setw(17) << F_v[i].Fz << setw(17) << '\n';
			total_e << setw(6) << time_of_snap[i] << setw(17) << norm3(F_e[i].Fx + F_v[i].Fx, F_e[i].Fy + F_v[i].Fy, F_e[i].Fz + F_v[i].Fz)
				<< setw(17) << F_e[i].Fx + F_v[i].Fx << setw(17) << F_e[i].Fy + F_v[i].Fy << setw(17) << F_e[i].Fz + F_v[i].Fz << '\n';
		}
	}
	elec_out.close();
	vdw_out.close();
	total_e.close();
	return 0;
}

int folders::out_force_to_folders(vector<force>& F_e, vector<force>& F_v, vector<double>& time_of_snap, vector<vector<force> >& F_e_vector, vector<vector<force> >& F_v_vector, arguments& Arguments)
{
	if (Arguments.flag_separation_results == false)
		return 1;
	fstream stat_out_elec("stat_force_elec.dat", ios::out), stat_out_vdw("stat_force_vdw.dat", ios::out), stat_out_total("stat_total_force.dat", ios::out),
		cal_out("cal.txt", ios::out | ios::app);
	stat_out_elec << setw(25) << "# folder name" << "   elec (" << Arguments.fu << ")   " << "SD (" << Arguments.fu << ") CI (95%) (" << Arguments.fu << ")\n";
	stat_out_vdw << setw(25) << "# folder name" << "   vdw (" << Arguments.fu << ")   SD (" << Arguments.fu << ") CI (95%) (" << Arguments.fu << ")\n";
	stat_out_total << setw(25) << "# folder name" << "   total force (" << Arguments.fu << ")   SD (" << Arguments.fu << ") CI (95%) (" << Arguments.fu << ")\n";
	cal_out << "----------------------------o0o----------------------------\n";
	cal_out << setw(10) << left << "No. traj" << right << setw(8) << "Elec (" << Arguments.fu << ")" << setw(8) << "vdW (" << Arguments.fu << ")\n";
	
	for (size_t i{}; i < paths.size(); ++i)
	{
		fstream out_elec(paths[i].string() + "/electrostatic_force.dat", ios::out);
		fstream out_vdw(paths[i].string() + "/vanderwaals_force.dat", ios::out);
		fstream total_force(paths[i].string() + "/total_force.dat", ios::out);
		out_elec << "# Time (" << Arguments.tu << ") " << setw(17) << "Electrostatic Force (" << Arguments.fu << ")\n";
		out_vdw << "# Time (" << Arguments.tu << ") " << setw(17) << "van der Waals Force (" << Arguments.fu << ")\n";
		total_force << "# Time (" << Arguments.tu << ") " << setw(17) << "Total Force (" << Arguments.fu << ")\n";

		double sd_e{}, sd_v{};
		double e_mean{}, v_mean{};
		size_t real_size{ F_e_vector.size() };
		for (size_t j{}; j < F_e_vector.size(); ++j)
		{
			if (F_e_vector[j].size() <= i)
				continue;
			if (F_e_vector[j][i].modul() < 10e-9 || F_v_vector[j][i].modul() < 10e-9) { --real_size; continue; }
			e_mean += F_e_vector[j][i].modul();
			v_mean += F_v_vector[j][i].modul();
		}
		e_mean /= real_size;
		v_mean /= real_size;
		for (size_t j{}; j < F_e_vector.size(); ++j)
		{
			if (F_e_vector[j].size() <= i) continue;
			out_elec << setw(6) << time_of_snap[j] << setw(17) << F_e_vector[j][i].modul() << '\n';
			out_vdw << setw(6) << time_of_snap[j] << setw(17) << F_v_vector[j][i].modul() << '\n';
			total_force << setw(6) << time_of_snap[j] << setw(17) << (F_e_vector[j][i] + F_v_vector[j][i]).modul() << '\n';
			sd_e += square(norm3(F_e_vector[j][i].Fx, F_e_vector[j][i].Fy, F_e_vector[j][i].Fz) - e_mean);
			sd_v += square(norm3(F_v_vector[j][i].Fx, F_v_vector[j][i].Fy, F_v_vector[j][i].Fz) - v_mean);
		}
		out_elec.close();
		out_vdw.close();
		total_force.close();
		fstream out_average(paths[i].string() + "/cal_force.dat", ios::out);
		out_average << "Average of Electrostatic force: " << e_mean << " (" << Arguments.fu << "). Standard deviation: " << sqrt(sd_e / real_size) << ". Confidence interval (95%): " << 1.96 * sqrt(sd_e) / real_size << "\n"
			<< "Average of van der Waals force: " << v_mean << " (" << Arguments.fu << "). Standard deviation: " << sqrt(sd_v / real_size) << ". Confidence interval (95%): " << 1.96 * sqrt(sd_v) / real_size << "\n";
		out_average.close();
		stat_out_elec << setw(25) << paths[i].string() << "  " << setw(17) << e_mean << setw(17) << sqrt(sd_e / real_size) << setw(17) << 1.96 * sqrt(sd_e) / real_size << '\n';
		stat_out_vdw << setw(25) << paths[i].string() << "  " << setw(17) << v_mean << setw(17) << sqrt(sd_v / real_size) << setw(17) << 1.96 * sqrt(sd_v) / real_size << '\n';
		stat_out_total << setw(25) << paths[i].string() << "  " << setw(17) << e_mean+v_mean << setw(17) << sqrt(sd_e / real_size) + sqrt(sd_v / real_size) << setw(17) << 1.96 * sqrt(sd_e) / real_size + 1.96 * sqrt(sd_v) / real_size << '\n';
		cal_out << setw(10) << left << Arguments.b_input + 1 + i << right << setw(16) << e_mean << setw(16) << v_mean << '\n';
	}
	stat_out_elec.close();
	stat_out_vdw.close();
	return 0;
}
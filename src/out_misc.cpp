#include<iostream>
#include<fstream>
#include<cmath>
#include<vector>
#include<iomanip>
#include"arguments.h"
#include"uti_func.h"
#include"folder.h"
using namespace std;

int folders::out_volume(vector<double>& time_of_snap, vector<vector<double> >& volumes, arguments& Arguments)
{
	if (Arguments.flag_cal_vol == false)
		return 1;

	size_t max_line_length{};
	for (size_t i{}; i < paths.size(); ++i)
		if (max_line_length < paths[i].string().length())
			max_line_length = paths[i].string().length();

	fstream stat_volumes("stat_volumes.dat", ios::out);
	stat_volumes << left << setw(max_line_length + 2) << right << "# folder name" << "   volume (A^3)   " << "SD (A^3)         CI (95%) (A^3)\n";

	for (size_t i{}; i < paths.size(); ++i)
	{
		fstream out_vol(paths[i].string() + "/volume.dat", ios::out);
		out_vol << "# Time (" << Arguments.tu << ") " << setw(17) << "box volume (A^3)\n";
		double ave_vol{}, sd_vol{};

		for (size_t j{}; j < volumes[i].size(); ++j)
		{
			ave_vol += volumes[i][j];
			out_vol << setw(6) << left << time_of_snap[j] << right << setw(17) << volumes[i][j] << '\n';
		}
		ave_vol /= volumes[i].size();
		
		for (size_t j{}; j < volumes[i].size(); ++j)
			sd_vol += square(volumes[i][j] - ave_vol);
		sd_vol = sqrt(sd_vol / volumes[i].size());

		stat_volumes << left << setw(25) << paths[i].string() << "  " << right << setw(17) << ave_vol << setw(17) << sd_vol << setw(17) << 1.96 * sd_vol / sqrt(volumes[i].size()) << '\n';
	}

	return 0;
}
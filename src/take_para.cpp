#include<vector>
#include<cmath>
#include<fstream>
#include<iostream>
#include<sstream>
#include"functions.h"
using namespace std;
size_t clear_string(string& str, const char* pattern)
{
	size_t find{ str.find(pattern) };
	if (find != string::npos)
		str = str.substr(0, find);
	return find;
}
void take_ff_para(amino_acid &amine_ff, para& par)
{
	for (auto& amino_acid : amine_ff.amino_acids)
		for (size_t k{}; k < amino_acid.second.size(); ++k)
		{
			for (size_t j{}; j < par.base.size(); ++j)
			{
				if (amino_acid.second[k].atom_type == par.base[j].atom_type)
				{
					amino_acid.second[k].epsilon = par.base[j].c12_epsilon;
					amino_acid.second[k].sigma = par.base[j].c6_sigma;
					amino_acid.second[k].atom_mass = par.base[j].mass;
					amino_acid.second[k].radius = par.base[j].radius;
					break;
				}
			}
		}
}
int take_para(para &parameters, string force_f, arguments& Arguments)
{
	string ff_name = Arguments.path_nonbonded+"ff" + force_f + "nonbonded.dat";
	fstream para_v;
	para_v.open(ff_name.c_str(), ios::in);
	if (!para_v.is_open())
	{
		if (Arguments.flag_use_common_topo == false)
		{
			cout << "Can not open file " << ff_name << "\n";
			exit(EXIT_FAILURE);
		}
		else
			return 0;
	}

	for (string str{}; getline(para_v, str);)
	{
		if (clear_string(str, ";") == 0 || clear_string(str, "#") == 0 || clear_string(str, "[") == 0)
			continue;

		istringstream is(str);
		string temp1{}, temp4{}, atom{};
		int atom_num{};
		double  mass{}, temp3{};
		double c6{}, c12{};
		if (force_f== "OPLSAA")
			is >> atom >> temp1 >> atom_num >> mass >> temp3 >> temp4 >> c6 >> c12;
		else
			is >> atom >> temp1 >> mass >> temp3 >> temp4 >> c6 >> c12;

		if (force_f == "AMBER99SB" || force_f == "AMBER99SB-ILDN" || force_f == "AMBER99SB-BSC1" || force_f == "CHARMM27" || force_f == "CHARMM36")
		{
			parameters.base.emplace_back(c6, c12, mass, 5.0*pow(2.0, 1.0 / 6.0)*c6, atom);
			parameters.register_atom_type(atom);
		}
		else if(force_f.find("GROMOS")!= string::npos)
		{
			if (c6 < 10e-14)
				parameters.base.emplace_back(c6, c12, mass, 0.0, atom);
			else
				parameters.base.emplace_back(c6, c12, mass, 5.*pow((c12 * 2 /c6), (1.0 / 6.0)), atom);
			parameters.register_atom_type(atom);
		}
		else
		{

			parameters.base.emplace_back(c6, c12, mass, 5.0*pow(2.0, 1.0 / 6.0)*c6, temp3, atom, temp1);
			parameters.register_atom_type(atom);
		}
	}
	para_v.close();
	return 0;
}

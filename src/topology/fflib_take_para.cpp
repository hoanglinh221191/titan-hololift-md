#include<vector>
#include<cctype>
#include<cmath>
#include<fstream>
#include<iostream>
#include<sstream>
#include<unordered_set>
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

namespace {
// Lower-cased section name of a "[ name ]" header line, or an empty string.
string section_name(const string& line)
{
	const size_t open{ line.find('[') };
	const size_t close{ line.find(']', open == string::npos ? 0 : open) };
	if (open == string::npos || close == string::npos || close <= open)
		return {};
	string name{ line.substr(open + 1, close - open - 1) };
	const size_t first{ name.find_first_not_of(" \t") };
	if (first == string::npos)
		return {};
	const size_t last{ name.find_last_not_of(" \t") };
	name = name.substr(first, last - first + 1);
	for (char& ch : name)
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	return name;
}
} // namespace

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

	// ENE-08 (21/10/2026): rows are atom types only while the current section
	// is [ atomtypes ] (or before any section header, for the tables that
	// carry none). A [ pairtypes ] or [ nonbond_params ] row read with the
	// atomtypes column layout yields sigma = epsilon = 0 for an existing type
	// name and, because the last registration wins, silently removed the LJ
	// parameters of 86 CHARMM36 types. Rows whose numeric columns do not parse
	// are rejected instead of being stored as zeros.
	string section{ "atomtypes" };
	size_t rows_other_sections{}, rows_malformed{}, rows_duplicate{};
	size_t rows_pair_overrides{}, rows_pair_malformed{};
	unordered_set<string> seen_types;
	seen_types.reserve(1024);

	for (string str{}; getline(para_v, str);)
	{
		clear_string(str, ";");
		clear_string(str, "#");
		const size_t first{ str.find_first_not_of(" \t\r\n") };
		if (first == string::npos)
			continue;
		str = str.substr(first);
		if (str.front() == '[')
		{
			const string name{ section_name(str) };
			section = name.empty() ? string("unknown") : name;
			continue;
		}
		if (section == "nonbond_params")
		{
			// ENE-11 (21/10/2026): pair-specific LJ (NBFIX/CUFIX; C6/C12 for
			// GROMOS) in the same units as the atom types of this table.
			istringstream pair_row(str);
			string type_i{}, type_j{};
			int funct{};
			double v1{}, v2{};
			if ((pair_row >> type_i >> type_j >> funct >> v1 >> v2) && funct == 1 &&
			    std::isfinite(v1) && std::isfinite(v2))
			{
				parameters.add_nonbond_override(type_i, type_j, v1, v2);
				++rows_pair_overrides;
			}
			else
				++rows_pair_malformed;
			continue;
		}
		if (section != "atomtypes")
		{
			++rows_other_sections;
			continue;
		}

		istringstream is(str);
		string temp1{}, temp4{}, atom{};
		int atom_num{};
		double  mass{}, temp3{};
		double c6{}, c12{};
		if (force_f== "OPLSAA")
			is >> atom >> temp1 >> atom_num >> mass >> temp3 >> temp4 >> c6 >> c12;
		else
			is >> atom >> temp1 >> mass >> temp3 >> temp4 >> c6 >> c12;
		if (is.fail() || atom.empty() || !std::isfinite(c6) || !std::isfinite(c12) ||
		    !std::isfinite(mass))
		{
			++rows_malformed;
			continue;
		}
		if (!seen_types.insert(atom).second)
			++rows_duplicate;

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

	if (rows_other_sections > 0 || rows_malformed > 0 || rows_duplicate > 0 ||
	    rows_pair_overrides > 0 || rows_pair_malformed > 0)
	{
		cout << "[lib_ff] " << ff_name << ": " << seen_types.size()
		     << " atom types read";
		if (rows_pair_overrides > 0)
			cout << "; " << rows_pair_overrides
			     << " [ nonbond_params ] pair overrides read";
		if (rows_pair_malformed > 0)
			cout << "; " << rows_pair_malformed
			     << " [ nonbond_params ] rows rejected (need: i j 1 v1 v2)";
		if (rows_other_sections > 0)
			cout << "; " << rows_other_sections
			     << " rows outside [ atomtypes ]/[ nonbond_params ] skipped "
			        "(1-4 [ pairtypes ] are not used for intermolecular energies)";
		if (rows_malformed > 0)
			cout << "; " << rows_malformed << " malformed rows rejected";
		if (rows_duplicate > 0)
			cout << "; " << rows_duplicate
			     << " duplicate type names (last definition wins)";
		cout << "\n";
	}
	// GROMOS defines most polar pairs explicitly (C12 classes I-III). A table
	// without them mixes the atom-type C6/C12 geometrically, which is the
	// class-I value for every pair.
	if (force_f.find("GROMOS") != string::npos && rows_pair_overrides == 0)
		cout << "[lib_ff] WARNING: " << ff_name
		     << " has no [ nonbond_params ]; GROMOS pair-specific C6/C12 "
		        "(C12 classes of polar pairs) are missing on the lib_ff path. "
		        "Use -common_topo with the GROMACS topology, or add the "
		        "[ nonbond_params ] block of ffnonbonded.itp to this table.\n";
	return 0;
}

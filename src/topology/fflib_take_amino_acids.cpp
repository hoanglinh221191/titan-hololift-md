#include "amino_a.h"
#include "functions.h"
#include "topology/fflib_rtp_parser.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace std;
int take_amino_acids(amino_acid &amine, int &num_amine, string &force_f,
                     const arguments &Arguments)
{
	/*string amine_name[36][2]{};
	amine_name[0][0] = "HD11"; amine_name[0][1] = "1HD1"; amine_name[1][0] = "HD12"; amine_name[1][1] = "2HD1"; amine_name[2][0] = "HD13"; amine_name[2][1] = "3HD1";
	amine_name[3][0] = "HD21"; amine_name[3][1] = "1HD2"; amine_name[4][0] = "HD22"; amine_name[4][1] = "2HD2"; amine_name[5][0] = "HD23"; amine_name[5][1] = "3HD2";
	amine_name[6][0] = "HE21"; amine_name[6][1] = "1HE2"; amine_name[7][0] = "HE22"; amine_name[7][1] = "2HE2"; amine_name[8][0] = "HG11"; amine_name[8][1] = "1HG1";
	amine_name[9][0] = "HG12"; amine_name[9][1] = "2HG1"; amine_name[10][0] = "HG13"; amine_name[10][1] = "3HG1"; amine_name[11][0] = "HG21"; amine_name[11][1] = "1HG2";
	amine_name[12][0] = "HG22"; amine_name[12][1] = "2HG2"; amine_name[13][0] = "HG23"; amine_name[13][1] = "3HG2"; amine_name[14][0] = "HH11"; amine_name[14][1] = "1HH1";
	amine_name[15][0] = "HH12"; amine_name[15][1] = "2HH1"; amine_name[16][0] = "HH21"; amine_name[16][1] = "1HH2"; amine_name[17][0] = "HH22"; amine_name[17][1] = "2HH2";
	amine_name[18][0] = "HH31"; amine_name[18][1] = "1HH3"; amine_name[19][0] = "HH32"; amine_name[19][1] = "2HH3"; amine_name[20][0] = "HH33"; amine_name[20][1] = "3HH3";
	amine_name[21][0] = "H5'1";amine_name[21][1] = "1H5'";amine_name[22][0] = "H5'2";amine_name[22][1] = "2H5'";amine_name[23][0] = "H2'1";amine_name[23][1] = "1H2'";
	amine_name[24][0] = "HO'2"; amine_name[24][1] = "2HO'"; amine_name[25][0] = "C_13"; amine_name[25][1] = "3C_1"; amine_name[26][0] = "C_10"; amine_name[26][1] = "0C_1";
	amine_name[27][0] = "C_11"; amine_name[27][1] = "1C_1"; amine_name[28][0] = "H_10"; amine_name[28][1] = "0H_1"; amine_name[29][0] = "C_12"; amine_name[29][1] = "2C_1";
	amine_name[30][0] = "C_15"; amine_name[30][1] = "5C_1"; amine_name[31][0] = "H_14"; amine_name[31][1] = "4H_1"; amine_name[32][0] = "H_15"; amine_name[32][1] = "5H_1";
	amine_name[33][0] = "H_16"; amine_name[33][1] = "6H_1"; amine_name[34][0] = "C_19"; amine_name[34][1] = "9C_1"; amine_name[35][0] = "H_27"; amine_name[35][1] = "7H_2";
	*/
	fstream amino(Arguments.path_amino_acid + force_f + "_aminoacids.rtp", ios::in);
	
	if (!amino.is_open())
	{
		if (Arguments.flag_use_common_topo == false)
		{
			cout << "Cannot open " << force_f << "_aminoacids.rtp" << '\n';
			exit(EXIT_FAILURE);
		}
		else
			return 0;
	}

	const auto parsed = TitanFfLib::parse_gromacs_rtp_atoms(amino, amine);
	if (!parsed) {
		cerr << "Cannot parse " << force_f
		     << "_aminoacids.rtp: " << parsed.error() << '\n';
		exit(EXIT_FAILURE);
	}
	num_amine += static_cast<int>(parsed->residue_count);
	amino.close();
	return 0;
}

#include"COM.h"
#include"MOL.h"
#include"distance_v2.h"
#include"functions.h"
#include<cmath>
size_t min_v(d_min& dmin_t);
//calculate distance between 2 residue of 2 protein or cal. the minimum distance from center of mass of protein's residue and one of atoms of ligand
void take_min(vector<com>& cen_cas, vector<com>& cen_ab, d_min &dmin, size_t begin, size_t end, size_t begin2, size_t end2, 
	const arguments& Arguments, int flag, double d_standard)
{
	if (flag==1)
	{
		for (size_t i{ begin }; i < end; ++i)
		{
			d_min dmin_test;
			for (size_t j{ begin2 }; j < end2; ++j)
			{
				//cout << cen_cas[i].get_res_num()<<" "<< cen_cas[i].get_chainid() << " "<<cen_ab[j].get_res_num()<<" " << cen_ab[j].get_chainid() << '\n';
				if (Arguments.self_int == true && cen_ab[j].get_chain() == cen_cas[i].get_chain() && cen_ab[j].get_res_num() == cen_cas[i].get_res_num())
					continue;

				double d_temp{ sqrt(square((cen_cas[i].get_x() - cen_ab[j].get_x())) + square((cen_cas[i].get_y() - cen_ab[j].get_y())) + square((cen_cas[i].get_z() - cen_ab[j].get_z()))) };
				if (d_temp <= d_standard)
				{
					dmin_test.base.emplace_back(d_temp, cen_ab[j].get_chain(), cen_cas[i].get_chain(), cen_ab[j].get_res_num(), 
						cen_cas[i].get_res_num(), cen_ab[j].get_atom_num(), cen_cas[i].get_atom_num(), cen_cas[i].get_res_name(), 
						cen_ab[j].get_res_name(), cen_cas[i].get_chainid(), cen_ab[j].get_chainid(), cen_cas[i].get_respos());
				}
			}
			if (!dmin_test.base.empty())
			{
				size_t pos{ min_v(dmin_test) };
				dmin.base.emplace_back(dmin_test.base[pos].d, dmin_test.base[pos].chain_ab, dmin_test.base[pos].chain_rec,
					dmin_test.base[pos].resnr_ab, dmin_test.base[pos].resnr_rec, dmin_test.base[pos].atom_num,
					dmin_test.base[pos].atom_num_rec, dmin_test.base[pos].res_name_rec, dmin_test.base[pos].res_name_ab
				,dmin_test.base[pos].chainid_rec,dmin_test.base[pos].chainid_ab,dmin_test.base[pos].respos_rec);
			}
		}
	}
	else
		for (size_t i{ begin }; i < end; ++i)
		{
			for (size_t j { begin2 }; j < end2; ++j)
			{
				if (Arguments.self_int == true && cen_ab[j].get_chain() == cen_cas[i].get_chain() && cen_ab[j].get_res_num() == cen_cas[i].get_res_num())
					continue;
				//cout << cen_cas[i].get_res_num() << " " << cen_cas[i].get_chainid() << " " << cen_ab[j].get_res_num() << " " << cen_ab[j].get_chainid() << '\n';

				double d_temp{ sqrt(square((cen_cas[i].get_x() - cen_ab[j].get_x())) + square((cen_cas[i].get_y() - cen_ab[j].get_y())) + square((cen_cas[i].get_z() - cen_ab[j].get_z()))) };
				if (d_temp <= d_standard)
				{
					dmin.base.emplace_back(d_temp, cen_ab[j].get_chain(), cen_cas[i].get_chain(), cen_ab[j].get_res_num(), 
						cen_cas[i].get_res_num(), cen_ab[j].get_atom_num(), cen_cas[i].get_atom_num(), cen_cas[i].get_res_name(),
						cen_ab[j].get_res_name(), cen_cas[i].get_chainid(), cen_ab[j].get_chainid(),cen_cas[i].get_respos());

				}
			}
		}
}
size_t min_v(d_min& dmin_t)
{
	size_t pos{};
	double min{ dmin_t.base[0].d };
	for (size_t i{ 1 }; i < dmin_t.base.size(); ++i)
	{
		if (dmin_t.base[i].d < min)
		{
			pos = i;
			min = dmin_t.base[i].d;
		}
	}
	return pos;
}
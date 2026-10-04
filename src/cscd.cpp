#include<iostream>
#include<fstream>
#include<algorithm>
#include<iomanip>
#include<thread>
#include"functions.h"
#include "titan_executor.h"
using namespace std;
int gen_files_for_cscd(Multi_MOL& pro_in, Multi_MOL& lig, const arguments& Arguments, fstream& log, size_t num_snap, int sc_flag, 
	int input_name, vector<string>& list_output_name, all_cscd& out_dis, const folders& paths, vector<fstream>& out_files)
{
	if (num_snap == 0)
	{
		if (!out_files.empty())
		{
			for (auto& file : out_files)file.close();
			out_files.clear();
		}
		if (!list_output_name.empty())list_output_name.clear();

		if (pro_in.get_mol_type() != "ligand")
		{
			for (size_t k{}; k < pro_in.Index.size() - 1; ++k)
				for (size_t l{}; l < pro_in.Index[k].get_size_v(); ++l)
				{
					string name_out_dis{ Arguments.name_out_cal_dis + pro_in.Index[k].get_chain() + to_string(pro_in.Index[k].get_res_num_e(l)) + ".dat" };
					list_output_name.emplace_back(name_out_dis);

					if (Arguments.flag_separation_results == true)
					{
						if (input_name - Arguments.b_input >= paths.get_size())return 1;
						out_files.emplace_back(fstream(paths.get_path(input_name - Arguments.b_input) + "/" + name_out_dis, ios::out));
					}
				}
		}
		else
		{
			for (size_t k{}; k < pro_in.Index.size() - 1; ++k)
			{
				string name_out_dis{ Arguments.name_out_cal_dis + pro_in.info[0].get_chain() + ".dat" };
				list_output_name.emplace_back(name_out_dis);

				if (Arguments.flag_separation_results == true)
				{
					if (input_name - Arguments.b_input >= paths.get_size())return 1;
					out_files.emplace_back(fstream(paths.get_path(input_name - Arguments.b_input) + "/" + name_out_dis, ios::out));
				}
			}
		}
	}
	else
	{
		for (size_t i{}; i < out_files.size(); ++i)
			if(!out_files[i].is_open())
			out_files[i].open(paths.get_path(input_name - Arguments.b_input) + "/" + list_output_name[i], ios::out|ios::app);
	}
	return 0;
}

int impl_cscd(Multi_MOL& pro_in, Multi_MOL& lig, const arguments& Arguments, fstream& log, TitanExecutor &executor, size_t num_snap, 
	int sc_flag, int input_name, vector<string>& list_output_name, all_cscd& out_dis, const folders& paths, vector<fstream>& out_files)
{
	vector<d_min> dis_sc(Arguments.nt);
	if (Arguments.flag_topo_1 == false)
	{
		executor.parallel_do(Arguments.nt, [&](size_t i) {
			take_min(ref(pro_in.m_mol[num_snap].Com), ref(lig.m_mol[num_snap].Com), ref(dis_sc[i]),
			         i * pro_in.m_mol[num_snap].Com.size() / Arguments.nt,
			         (i + 1ull) * pro_in.m_mol[num_snap].Com.size() / Arguments.nt,
			         0ull, lig.m_mol[num_snap].Com.size(), ref(Arguments), sc_flag, Arguments.limit_dis);
		});
	}
	else
	{
		executor.parallel_do(Arguments.nt, [&](size_t i) {
			find_d_rescom_com(ref(pro_in.m_mol[num_snap].Com), ref(lig), ref(dis_sc[i]),
			                  i * pro_in.m_mol[num_snap].Com.size() / Arguments.nt,
			                  (i + 1ull) * pro_in.m_mol[num_snap].Com.size() / Arguments.nt,
			                  num_snap, lig.m_mol[num_snap].Com.size(), Arguments.force_f,
			                  sc_flag, Arguments.limit_dis);
		});
	}

	if (gen_files_for_cscd(pro_in, lig, Arguments, log, num_snap, sc_flag, input_name, list_output_name, out_dis, paths, out_files) == 1) return 1;

	if (Arguments.flag_topo_1 == false && Arguments.flag_topo_2 == false)
	{
		size_t NumberofFile{};
		for (size_t k{};k < pro_in.Index.size() - 1;++k)
			for (size_t l{};l < pro_in.Index[k].get_size_v();++l)
			{
				if (Arguments.flag_separation_results == true)
				{
					out_files[NumberofFile].precision(3);
					out_files[NumberofFile] << setw(10) << fixed << pro_in.m_mol[num_snap].time_of_snap;
				}
				vector<double> dis_temp{};
				for (size_t kk{}; kk < lig.Index.size() - 1; ++kk)
					for (size_t ll{}; ll < lig.Index[kk].get_size_v(); ++ll)
					{
						for (size_t i{};i < Arguments.nt;++i)
							for (size_t j{};j < dis_sc[i].base.size();++j)
							{
								if (dis_sc[i].base[j].chain_rec == pro_in.Index[k].get_chain()
									&& dis_sc[i].base[j].resnr_rec == pro_in.Index[k].get_res_num_e(l)
									&& dis_sc[i].base[j].resnr_ab == lig.Index[kk].get_res_num_e(ll)
									&& dis_sc[i].base[j].chain_ab == lig.Index[kk].get_chain())
								{
									dis_temp.emplace_back(dis_sc[i].base[j].d);
									if (Arguments.flag_separation_results == true)
										out_files[NumberofFile] << setw(14) << fixed << dis_sc[i].base[j].d;
								}
							}
					}
				out_dis.emplace(list_output_name.size(), NumberofFile, num_snap, pro_in.m_mol[num_snap].time_of_snap, dis_temp);
				if (Arguments.flag_separation_results == true)
					out_files[NumberofFile] << '\n';
				out_files[NumberofFile].close();
				++NumberofFile;
			}
	}
	else if(Arguments.flag_topo_1 == true && Arguments.flag_topo_2 == false)
	{
		size_t NumberofFile{};
		for (size_t k{};k < pro_in.Index.size() - 1;++k)
			for (size_t l{};l < pro_in.Index[k].get_size_v();++l)
			{
					if (Arguments.flag_separation_results == true)
					{
						out_files[NumberofFile].precision(3);
						out_files[NumberofFile] << setw(10) << fixed << pro_in.m_mol[num_snap].time_of_snap;
					}
				vector<double> dis_temp{};
				for (size_t kk{};kk < lig.Index.size() - 1;++kk)
					{
						for (size_t i{};i < Arguments.nt;++i)
							for (size_t j{};j < dis_sc[i].base.size();++j)
							{
								if (dis_sc[i].base[j].chain_rec == pro_in.Index[k].get_chain()
									&& dis_sc[i].base[j].resnr_rec == pro_in.Index[k].get_res_num_e(l))
								{
									dis_temp.emplace_back(dis_sc[i].base[j].d);
									if (Arguments.flag_separation_results == true)
										out_files[NumberofFile] << setw(14) << fixed << dis_sc[i].base[j].d;
								}
							}
					}
				out_dis.emplace(list_output_name.size(), NumberofFile, num_snap, pro_in.m_mol[num_snap].time_of_snap, dis_temp);
				if (Arguments.flag_separation_results == true)
					out_files[NumberofFile] << '\n';
				out_files[NumberofFile].close();
				++NumberofFile;
			}
	}
	else if (Arguments.flag_topo_1 == true && Arguments.flag_topo_2 == true)
	{
		size_t NumberofFile{};
		for (size_t k{}; k < pro_in.Index.size() - 1; ++k)
			{
				if (Arguments.flag_separation_results == true)
				{
					out_files[NumberofFile].precision(3);
					out_files[NumberofFile] << setw(10) << fixed << pro_in.m_mol[num_snap].time_of_snap;
				}
				vector<double> dis_temp{};
				for (size_t kk{}; kk < lig.Index.size() - 1; ++kk)
				{
					for (size_t i{}; i < Arguments.nt; ++i)
						for (size_t j{}; j < dis_sc[i].base.size(); ++j)
						{
							if (dis_sc[i].base[j].chain_rec == pro_in.Index[k].get_chain())
							{
								dis_temp.emplace_back(dis_sc[i].base[j].d);
								if (Arguments.flag_separation_results == true)
									out_files[NumberofFile] << setw(14) << fixed << dis_sc[i].base[j].d;
							}
						}
				}
				out_dis.emplace(list_output_name.size(), NumberofFile, num_snap, pro_in.m_mol[num_snap].time_of_snap, dis_temp);
				if (Arguments.flag_separation_results == true)
					out_files[NumberofFile] << '\n';
				out_files[NumberofFile].close();
				++NumberofFile;
			}
	}
	for (auto& file : out_files)
		if (file.is_open())file.close();
	return 0;
}

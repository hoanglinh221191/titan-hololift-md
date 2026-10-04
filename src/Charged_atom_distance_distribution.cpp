#include"Multi_MOL.h"
#include"info.h"
#include<fstream>
#include<thread>
#include<iomanip>

int print_distribute_charge_distances(const vector<vector<double>>& input_data, fstream& outputfile, const arguments& Arguments);

int Multi_MOL::cal_distribution_distance_charged_atoms(const arguments& Arguments, const ::info& output_info)
{
	fstream output_total(Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4) + output_info.getDistributionDistanceCharged(), ios::out),
		output_positive(Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4) + "_positive" + output_info.getDistributionDistanceCharged(), ios::out),
		output_negative(Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4) + "_negative" + output_info.getDistributionDistanceCharged(), ios::out);

	vector<vector<double>> distances(m_mol.size()), distances_negative_charges(m_mol.size()), distances_positive_charges(m_mol.size());

	vector<thread> impl_threads(Arguments.nt);

	for (size_t i{}; i < impl_threads.size(); ++i)
		impl_threads[i] = thread(&Multi_MOL::impl_cal_distribution_distance_charged_atoms, this, ref(distances), ref(distances_negative_charges),
			ref(distances_positive_charges), i*this->m_mol.size() / impl_threads.size(), (i + 1)*this->m_mol.size() / impl_threads.size());
	for (auto& imp : impl_threads)imp.join();
	
	if (Arguments.nt < 2)
	{
		print_distribute_charge_distances(distances, output_total, Arguments);
		print_distribute_charge_distances(distances_positive_charges, output_positive, Arguments);
		print_distribute_charge_distances(distances_negative_charges, output_negative, Arguments);
	}
	else if (Arguments.nt == 2)
	{
		thread print1(print_distribute_charge_distances, ref(distances), ref(output_total), ref(Arguments));
		thread print2(print_distribute_charge_distances,ref(distances_positive_charges), ref(output_positive), ref(Arguments));
		print1.join();
		print2.join();
		print_distribute_charge_distances(distances_negative_charges, output_negative, Arguments);
	}
	else
	{
		thread print1(print_distribute_charge_distances, ref(distances), ref(output_total), ref(Arguments));
		thread print2(print_distribute_charge_distances, ref(distances_positive_charges), ref(output_positive), ref(Arguments));
		thread print3(print_distribute_charge_distances,ref(distances_negative_charges), ref(output_negative), ref(Arguments));
		print1.join();
		print2.join();
		print3.join();
	}

	return 0;
}

int Multi_MOL::impl_cal_distribution_distance_charged_atoms(vector<vector<double>>& distances, vector<vector<double>>& distances_negative_charges,
	vector<vector<double>>& distances_positive_charges, size_t begin, size_t end)
{
	for (size_t i{begin}; i < end; ++i)
	{
		double com[3]{};
		this->com_mol(i, com);

		this->m_mol[i].cal_dis_charged_atoms(distances[i], distances_negative_charges[i], distances_positive_charges[i], this->parameters, com);
	}

	return 0;
}

int print_distribute_charge_distances(const vector<vector<double>>& input_data, fstream& outputfile, const arguments& Arguments)
{
	double max{ -10000. }, min{ 1000000. };

	for (size_t i{};i<input_data.size();++i)
		for (size_t j{}; j < input_data[i].size(); ++j)
		{
			if (min > input_data[i][j])min = input_data[i][j];
			if (max < input_data[i][j])max = input_data[i][j];
		}

	double delta{ (max - min) / Arguments.number_charged_distances_bin };
	if (max < Arguments.charged_distances_max)
	{
		delta = (Arguments.charged_distances_max - Arguments.charged_distances_min) / Arguments.number_charged_distances_bin;
		min = Arguments.charged_distances_min;
	}

	vector<size_t> histo(Arguments.number_charged_distances_bin);
	size_t total_dis{};

	for (size_t i{}; i < input_data.size(); ++i)
	{
		total_dis += input_data[i].size();

		for (size_t j{}; j < input_data[i].size(); ++j)
		{
			size_t pos{ static_cast<size_t>((input_data[i][j] - min) / delta) };
			if (pos >= histo.size())
				pos = histo.size() - 1;
			++histo[pos];
		}
	}

	outputfile.precision(7);
	outputfile << "# Distance (A)                     Frequency\n";
	for (size_t i{}; i < histo.size(); ++i)
		outputfile << setw(8) << left << i * delta + min << setw(16) << right << histo[i] * 100. / total_dis  << '\n';
	outputfile.close();
	return 0;
}
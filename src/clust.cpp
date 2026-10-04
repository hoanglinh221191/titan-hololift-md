#include"clust.h"
#include"arguments.h"
#include"MOL.h"

int multi_cluster::emplace_clust(size_t image_1, size_t image_2, size_t atom_1, size_t atom_2)
{
	bool flag_in_1{ false }, flag_in_2{ false };
	size_t clust_pos_1{}, clust_pos_2{};
	for (size_t i{}; i < clusters.size(); ++i)
	{
		for (size_t j{}; j < clusters[i].clust.size(); ++j)
		{
			if (clusters[i].clust[j].first == image_1 && clusters[i].clust[j].second == atom_1)
			{
				flag_in_1 = true; clust_pos_1 = i;
			}
			if (clusters[i].clust[j].first == image_2 && clusters[i].clust[j].second == atom_2)
			{
				flag_in_2 = true; clust_pos_2 = i;
			}
		}
	}

	if (flag_in_1 == true && flag_in_2 == false)
	{
		clusters[clust_pos_1].clust.emplace_back(image_2, atom_2);
	}
	else if (flag_in_2 == true && flag_in_1 == false)
	{
		clusters[clust_pos_2].clust.emplace_back(image_1, atom_1);
	}
	else if (flag_in_1 == false && flag_in_2 == false)
	{
		cluster temp; temp.clust.emplace_back(image_1, atom_1); temp.clust.emplace_back(image_2, atom_2);
		clusters.emplace_back(temp);
	}
	else {}

	return 0;
}

int multi_cluster::find_candidate_clust(MOL& mol_in, arguments& Arguments)
{
	return 0;
}
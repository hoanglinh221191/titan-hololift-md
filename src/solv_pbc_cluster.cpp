#include"functions.h"
#include"clust.h"
#include<thread>

int init_rejected_images(vector<size_t>& rejected_images, arguments& Arguments)
{
	bool x{ false }, y{ false }, z{ false };
	for (size_t i{}; i < Arguments.pbc.pbc_clust_axis.size(); ++i)
	{
		cout << Arguments.pbc.pbc_clust_axis[i] << "  ";
		if (Arguments.pbc.pbc_clust_axis[i] == 'x')
		{
			x = true;
		}
		else if (Arguments.pbc.pbc_clust_axis[i] == 'y')
		{
			y = true;
		}
		else if (Arguments.pbc.pbc_clust_axis[i] == 'z')
		{
			z = true;
		}
	}

	if (x == false)
	{
		rejected_images.emplace_back(0); rejected_images.emplace_back(1);
		for (size_t j{ 18 }; j <= 25; ++j)
			rejected_images.emplace_back(j);
		for (size_t j{ 6 }; j <= 13; ++j)
			rejected_images.emplace_back(j);
	}
	if (y == false)
	{
		rejected_images.emplace_back(2); rejected_images.emplace_back(3);
		for (size_t j{ 6 }; j <= 9; ++j)
			rejected_images.emplace_back(j);
		for (size_t j{ 14 }; j <= 25; ++j)
			rejected_images.emplace_back(j);
	}
	if (z == false)
	{
		rejected_images.emplace_back(4); rejected_images.emplace_back(5);
		for (size_t j{ 10 }; j <= 25; ++j)
			rejected_images.emplace_back(j);
	}
	return 0;
}

bool check_in_rejected_images(vector<size_t>& rejected_images, size_t im)
{
	if (rejected_images.empty())return false;

	for (size_t i{};i<rejected_images.size();++i)
		if (im == rejected_images[i]) { return true; }
	return false;
}

int impl_cal_dis_clust(Multi_MOL& pbc_images, size_t i, multi_cluster& CLUST, vector<size_t>& rejected_images, arguments& Arguments)
{
	for (size_t j{}; j < pbc_images.m_mol[i].size(); ++j)
	{
		for (size_t k{}; k < pbc_images.m_mol.size(); ++k)
		{
			bool in_image{ false };
			for (auto& im : rejected_images)
				if (i == im || k == im)in_image = true;
			if (in_image == true) continue;

			for (size_t l{}; l < pbc_images.m_mol[k].size(); ++l)
			{
				if (i == k && j == l)continue;

				double dis{ sqrt(square(pbc_images.m_mol[k].pos.x[l] - pbc_images.m_mol[i].pos.x[j])
				+ square(pbc_images.m_mol[k].pos.y[l] - pbc_images.m_mol[i].pos.y[j])
				+ square(pbc_images.m_mol[k].pos.z[l] - pbc_images.m_mol[i].pos.z[j])) };

				if (dis <= Arguments.dis_clust)
					CLUST.emplace_clust(i, k, j, l);
			}
		}
	}
	return 0;
}

int cal_dis_clust(Multi_MOL& pbc_images, size_t begin, size_t end, multi_cluster& CLUST, vector<size_t>& rejected_images, arguments& Arguments)
{
	for (size_t i{ begin }; i < end; ++i)
		impl_cal_dis_clust(pbc_images, i, CLUST, rejected_images, Arguments);
	return 0;
}

int wrapper_cal_dis_clust(Multi_MOL& pbc_images, multi_cluster& CLUST, vector<size_t>& rejected_images, arguments& Arguments)
{
	vector<thread> cal(Arguments.nt);
	vector<multi_cluster> CLUST_m(Arguments.nt);

	for (size_t i{}; i < Arguments.nt; ++i)
		cal[i] = thread(cal_dis_clust, ref(pbc_images), i*pbc_images.m_mol.size() / Arguments.nt, (i + 1)*pbc_images.m_mol.size() / Arguments.nt
			, ref(CLUST_m[i]), ref(rejected_images), ref(Arguments));
	for (auto& t : cal)t.join();

	for (auto& C : CLUST_m)
		for (auto& cl : C.clusters)
			CLUST.clusters.emplace_back(cl);
	return 0;
}

int MOL::impl_solv_pbc_clust(arguments& Arguments, vector<class index>& Index)
{
	Multi_MOL pbc_images;
	pbc_images.m_mol.reserve(27);
	this->create_pbc_images(Arguments, pbc_images, Index);

	multi_cluster CLUST;

	vector<size_t> rejected_images;
	init_rejected_images(rejected_images, Arguments);
	cout << "done build rejected\n";

	wrapper_cal_dis_clust(pbc_images, CLUST, rejected_images, Arguments);

	cout << "SIZE: " << CLUST.clusters.size() << '\n';
	for (size_t i{}; i < CLUST.clusters.size(); ++i)
	{
		if (CLUST.clusters[i].clust.size() != this->size())
			CLUST.clusters.erase(CLUST.clusters.begin() + i);
	}
	
	if (CLUST.clusters.empty())return 1;
	cout << "fixing....\n";
	for (size_t i{}; i < CLUST.clusters[0].clust.size(); ++i)
	{
	}
	cout << "done\n";
	return 0;
}

int Multi_MOL::solve_pbc_cluster(arguments& Arguments, ::info& output_info)
{
	
	for (size_t i{}; i < this->m_mol.size(); ++i)
		this->m_mol[i].impl_solv_pbc_clust(Arguments, this->Index);
	
	return 0;
}

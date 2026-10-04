#include"MOL.h"
#include"matrix.h"
#include"groups.h"
#include"functions.h"
#include"uti_func.h"
#include<cmath>
#include<cfloat>
struct neighbor
{
	int neigbor_index;
	double distance;
	template<typename T>
	neighbor(int index, T&& d) :neigbor_index{ index }, distance{ forward<T>(d) } {}
	neighbor() :neigbor_index{}, distance{} {}
};
struct neighbors
{
	int nearest_num;
	vector<int> index;
	vector<neighbor> base;
	neighbors() :nearest_num{}, index(4,-1) {  base.reserve(100); }
	int find_nearest_neighbors(int num)
	{
		nearest_num = num;
		for (int i{};i < num;++i)
		{
			double dis_min{ 100000. };
			for (int j{};j < base.size();++j)
			{
				if (base[j].distance > 0. && base[j].distance < dis_min)
				{
					bool check{ false };
					for (int k{};k <= i;++k)
						if (index[k] == j) { check = true; break; }
					if (check == false)
					{
						index[i]= j;dis_min = base[j].distance;
					}
				}
			}
		}
		return 0;
	}
};
inline double euclidean_distance(matrix& vec1, int I, int J, int II)
{
	if (I == J)
		return 100000.;
	return fabs(vec1.mat[II][I] - vec1.mat[II][J]);
}
inline double euclidean_distance(matrix& vec1, int I, int J, int II, int JJ)
{
	return sqrt(square(vec1.mat[II][I] - vec1.mat[II][J]) + square(vec1.mat[JJ][I] - vec1.mat[JJ][J]));
}
double non_gaussian_entropy(matrix& matrix_in, int k_neighbor, int II)
{
	vector<neighbors > Neighbors;
	for (int i{};i < matrix_in.col;++i)
	{
		neighbors n_temp;
		for (int j{};j < matrix_in.col;++j)
			if (i != j)
			{
				n_temp.base.emplace_back(j, euclidean_distance(matrix_in, i, j, II));
			}
		Neighbors.emplace_back(n_temp);
	}
	for (auto& N : Neighbors)
		N.find_nearest_neighbors(k_neighbor);
	double entropy{};
	int gamma_n{ 1 };
	double factorial{ 0. };
	for (int i{ 2 };i <= gamma_n + 1;++i)
		factorial += log(i);
	double entropy_k{};
	for (auto& N : Neighbors)
	{
		entropy_k += log(N.base[N.index[k_neighbor - 1]].distance);
	}
	entropy_k *= 1. / static_cast<double>(matrix_in.col);
	entropy_k += 0.5772157;
	double L_k{ 1. };
	for (int j{ 1 };j <= k_neighbor;++j)
		L_k /= (1.0*j);
	entropy += entropy_k - L_k;
	return 1.38064*0.3*6.02*(entropy + (log(matrix_in.col) + 1./ 2.*log(3.14519) - factorial));
}

double non_gaussian_entropy_2nd(matrix& matrix_in, int k_neighbor, int II, int JJ)
{
	vector<neighbors > Neighbors;
	for (int i{};i < matrix_in.col;++i)
	{
		neighbors n_temp;
		for (int j{};j < matrix_in.col;++j)
			if (i != j)
			{
				n_temp.base.emplace_back(j, euclidean_distance(matrix_in, i, j, II,JJ));
			}
		Neighbors.emplace_back(n_temp);
	}
	for (auto& N : Neighbors)
		N.find_nearest_neighbors(k_neighbor);
	double entropy{};
	constexpr int gamma_n{ 2 / 2 };
	double factorial{ 0. };
	for (int i{ 2 };i <= gamma_n + 1;++i)
		factorial += log(i);
	double entropy_k{};
	for (auto& N : Neighbors)
	{
		entropy_k += log(N.base[N.index[k_neighbor - 1]].distance);
	}
	entropy_k *= 2. / static_cast<double>(matrix_in.col);
	entropy_k += 0.5772157;
	double L_k{ 1. };
	for (int j{ 1 };j <= k_neighbor;++j)
		L_k /= (1.0*j);
	entropy += entropy_k - L_k;
	return 1.38064*0.3*6.02*(entropy + (log(matrix_in.col) + 2. / 2.*log(3.14519) - factorial));
}
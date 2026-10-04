#include<vector>
#include<cmath>
#include<memory>
#include<thread>
#include"matrix.h"
#include"functions.h"
#include"uti_func.h"
using namespace std;
void mul(matrix& A, matrix& Q1, matrix& temp, int begin, int end, int I);
void mul_2(matrix& A, matrix& Q1, matrix& temp, int begin, int end, int I);
inline double delta(int i, int j)
{
	if (i == j)
		return 1.;
	else
		return 0.;
}
void reduce_h_form(matrix& A, int I, int K, size_t nt)
{
	size_t num_threads{ nt };
	double s{};
	for (auto i = I;i < A.size();++i)
		s += square(A.mat[i][K]);
	s = -1.0*sqrt(s) *A.mat[I][K] / fabs(A.mat[I][K]) ;
	double r{ 0.5*(square(s) - A.mat[I][K] * s) };
	vector<double> u{};
	unique_ptr<double[]> p_u_1{ new double[A.size()]{} };
	p_u_1[I] = (A.mat[I][K] - s);
	for (auto i = I + 1;i < A.size();++i)
		p_u_1[i] = A.mat[i][K];
	matrix Q1{ A.size() };
	for (auto i = I;i < Q1.row;++i)
		for (auto j = I;j < Q1.col;++j)
		{
			Q1.mat[i][j] -= 2.0*p_u_1[i] * p_u_1[j] / (4.0*r);
			if (std::isnan(Q1.mat[i][j])) {
				Q1.mat[i][j] = 0.0;
			}
		}
	matrix temp{ A.row - I, A.col };
	vector<thread> a( num_threads );
	for (auto i = 0;i < num_threads;++i)
		a[i] = thread(mul_2, ref(A), ref(Q1), ref(temp), static_cast<int>(i *(A.row - I) / num_threads), static_cast<int>((i + 1) *(A.row - I) / num_threads), I);
	for (auto& t : a)t.join();

	for (auto i = 0;i < temp.row;++i)
		for (auto j = 0;j < temp.col;++j)
			A.mat[i + I][j] = temp.mat[i][j];
	/******************************************/
	matrix temp_2{ A.row, A.col - I };
	for (auto i = 0;i < num_threads;++i)
		a[i] = thread(mul, ref(A), ref(Q1), ref(temp_2), static_cast<int>(i*A.row / num_threads), static_cast<int>((i + 1)*A.row / num_threads), I);
	for (auto& t : a)t.join();

	for (auto i = 0;i < temp_2.row;++i)
		for (auto j = 0;j < temp_2.col;++j)
			A.mat[i][j + I] = temp_2.mat[i][j];
}
void mul(matrix& A, matrix& Q1, matrix& temp, int begin, int end, int I)
{
	for (auto i = begin;i < end;++i)
		for (auto k = 0;k < A.row;++k)
		{
			double d{ A.mat[i][k] };
			for (auto j = 0;j < A.col - I;++j)
				temp.mat[i][j] += d * Q1.mat[k][j + I];
		}
}
void mul_2(matrix& A, matrix& Q1, matrix& temp, int begin, int end, int I)
{
	for (auto i = begin;i < end;++i)
		for (auto k = 0;k < A.row;++k)
		{
			double d{ Q1.mat[i + I][k] };
			for (auto j = 0;j < A.col;++j)
				temp.mat[i][j] += d * A.mat[k][j];
		}
}

int matrix::given_rotation(int& begin, int end, size_t nt)
{
	vector<matrix> G(nt, row);
	int new_row{ begin };
	for (int i{ 0 };i < begin - 1;++i)
	{
		if (fabs(mat[row - 1 - i][col - 2 - i]) < 10e-15)
			--new_row;
		else
			break;
	}
	vector<thread> threads_imp(nt);
	for (size_t i{};i < nt;++i)
		threads_imp[i] = thread(&matrix::impl_given_rotation, &(*this), ref(G[i]), static_cast<int>(i*new_row / nt), static_cast<int>((i + 1)*new_row / nt), new_row);
	for (auto& t : threads_imp)
		t.join();
	for (size_t i{}; i < nt;++i)
	{
		this->set_num_threads(nt);
		*this = (*this)*G[i];
	}
	return new_row;
}
int matrix::impl_given_rotation(matrix& G1, int begin, int end, int new_row)
{
	if (end = new_row) end = new_row - 1;
	for (auto i = begin;i < end;++i)
	{
		double r{ sqrt(square(mat[i][i]) + square(mat[i + 1][i])) };
		matrix G{ new_row };
		double c{ mat[i][i] / r }, s{ -1.0*mat[i + 1][i] / r };
		G.mat[i][i] = c;G.mat[i + 1][i + 1] = c;G.mat[i][i + 1] = -1.0*s;G.mat[i + 1][i] = s;
		G.special_mul(*this, i, 2);
		G.transpose();
		if (i == begin)
			G1 = G;
		else
			G1.special_mul_2(G, i, i + 2);
	}
	return 0;
}
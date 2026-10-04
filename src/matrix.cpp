#include<iostream>
#include<thread>
#include<functional>
#include"matrix.h"
#include<cmath>
#include<vector>
using namespace std;
matrix::matrix() :mat{ nullptr }, col{}, row{}, temp_p{ nullptr } {}
matrix::matrix(int size) : col{ size }, row{ size }, num_of_thread{1} {
	mat = new double*[size];
	temp_p = new double[size*size]{};
	for (auto i = 0;i < size;++i)
		mat[i] = &temp_p[i*size];
	for (auto i = 0;i < size;++i)
		for (auto j = 0;j < size;++j)
			if (i == j)
				mat[i][j] = 1.0;
			else
				mat[i][j] = 0.;
}
matrix::matrix(int _row, int _col) :row{ _row }, col{ _col }, num_of_thread{1} {
	mat = new double*[_row];
	temp_p = new double[_row*_col]{};
	for (auto i = 0;i < _row;++i)
		mat[i] = &temp_p[i*_col];
}
matrix::matrix(const matrix& rhs) :row{ rhs.row }, col{ rhs.col }, num_of_thread{ 1 } {
	mat = new double*[row];
	temp_p = new double[rhs.row*rhs.col];
	for (auto i = 0;i < row;++i)
		mat[i] = &temp_p[i*col];
	for (auto i = 0;i < rhs.row;++i)
		for (auto j = 0;j < rhs.col;++j)
			mat[i][j] = rhs.mat[i][j];
}
matrix::matrix(matrix&& rhs) :row{ rhs.row }, col{ rhs.col }, num_of_thread{1}
{
	mat = new double*[row];
	temp_p = rhs.temp_p;
	rhs.temp_p = nullptr;
	for (auto i = 0;i < row;++i)
	{
		mat[i] = rhs.mat[i]; rhs.mat[i] = nullptr; 
	}
	delete[] rhs.mat;
	rhs.mat = nullptr;
}
matrix::~matrix()
{
	if (mat != nullptr)
	{
		for (auto i = 0;i < row;++i)
			mat[i] = nullptr;
		delete[] temp_p;
		temp_p = nullptr;
		delete[] mat;
		mat = nullptr;
	}
}
matrix matrix::operator+(matrix& rhs)
{
	for (auto i = 0;i < row;++i)
		for (auto j = 0;j < col;++j)
			this->mat[i][j] += rhs.mat[i][j];
	return *this;
}
matrix matrix::operator+(matrix&& rhs)
{
	for (auto i = 0;i < row;++i)
		for (auto j = 0;j < col;++j)
			this->mat[i][j] += rhs.mat[i][j];
	return *this;
}

matrix matrix::operator*(matrix& rhs)
{
	matrix result{ row,rhs.col };
	if (rhs.col*row*col / num_of_thread > 10000)
	{
		vector<thread> mul_mat(num_of_thread);
		for (size_t i{};i < num_of_thread;++i)
			mul_mat[i] = thread(&matrix::impl_mul, &(*this), std::ref(rhs), std::ref(result), static_cast<int>(i*row / num_of_thread), static_cast<int>((i + 1)*row / num_of_thread));
		for (auto& t : mul_mat)t.join();
	}
	else
	{
		for (int i{};i < row;++i)
			for (int k{};k < col;++k)
				for (int j{};j < rhs.col;++j)
					result.mat[i][j] += this->mat[i][k] * rhs.mat[k][j];
	}
	return result;
}
matrix matrix::operator*(matrix&& rhs)
{
	matrix result{ row,rhs.col };
	if (rhs.col*row*col/num_of_thread > 10000)
	{
		vector<thread> mul_mat(num_of_thread);
		for (size_t i{};i < num_of_thread;++i)
			mul_mat[i] = thread(&matrix::impl_mul, &(*this), std::ref(rhs), std::ref(result), static_cast<int>(i*row / num_of_thread), static_cast<int>((i + 1)*row / num_of_thread));
		for (auto& t : mul_mat)t.join();
	}
	else
	{
		for (int i{};i < row;++i)
			for (int k{};k < col;++k)
				for (int j{};j < rhs.col;++j)
					result.mat[i][j] += this->mat[i][k] * rhs.mat[k][j];
	}
	return result;
}
void matrix::impl_mul(matrix& rhs, matrix& result, int begin, int end)
{
	for (auto i = begin;i < end;++i)
		for (auto k = 0;k < col;++k)
		{
			double temp{ this->mat[i][k] };
			for (auto j = 0;j < rhs.col;++j)
				result.mat[i][j] += temp * rhs.mat[k][j];
		}
}
matrix& matrix::operator=(matrix& rhs)
{
	if (rhs.col == this->col && rhs.row == this->row)
		for (auto i = 0;i < rhs.row;++i)
			for (auto j = 0;j < rhs.col;++j)
				this->mat[i][j] = rhs.mat[i][j];
	num_of_thread = rhs.num_of_thread;
	return *this;
}
matrix& matrix::operator=(matrix&& rhs)
{
	num_of_thread = rhs.num_of_thread;
	delete[] temp_p;
	temp_p = rhs.temp_p;
	rhs.temp_p = nullptr;
	for (auto i = 0;i < row;++i)
	{
		mat[i] = rhs.mat[i]; rhs.mat[i] = nullptr;  
	}
	delete[] rhs.mat;
	rhs.mat = nullptr;
	return *this;
}
matrix& matrix::move(matrix& rhs)
{
	delete[] temp_p;
	temp_p = rhs.temp_p;
	rhs.temp_p = nullptr;
	num_of_thread = rhs.num_of_thread;
	for (auto i = 0;i < row;++i)
	{
		mat[i] = rhs.mat[i]; rhs.mat[i] = nullptr;  
	}
	delete[] rhs.mat;
	rhs.mat = nullptr;
	return *this;
}
ostream& operator<<(ostream& os, const matrix& Obj)
{
	for (auto i = 0;i < Obj.row;++i)
	{
		for (auto j = 0;j < Obj.col;++j)
			os << Obj.mat[i][j] << "  ";
		os << '\n';
	}
	return os;
}
void matrix::transpose()
{
	for (auto i = 0;i < row;++i)
		for (auto j = i;j < col;++j)
		{
			if (i != j)
			{
				double temp = mat[j][i];
				mat[j][i] = mat[i][j];
				mat[i][j] = temp;
			}
		}
}
void matrix::special_mul(matrix& rhs, int begin, int range)
{
	matrix temp{ range,rhs.col };
	
	for (auto i = 0;i < range;++i)
		for (auto k = 0;k < row;++k)
		{
			double d{ this->mat[i + begin][k] };
			for (auto j = 0;j < rhs.col;++j)
				temp.mat[i][j] += d * rhs.mat[k][j];
		}
	for (auto i = begin; i < begin + range;++i)
		for (auto j = 0;j < rhs.col;++j)
			rhs.mat[i][j] = temp.mat[i - begin][j];
}
void matrix::special_mul_2(matrix& rhs, int begin, int end)
{
	matrix temp{ row, end - begin };
	for (auto i = 0;i < row;++i)
		for (auto j = 0;j < end - begin;++j)
		{
			double d{};
			for (auto k = 0;k < rhs.row; ++k)
				d += mat[i][k] * rhs.mat[k][j + begin];
			temp.mat[i][j] = d;
		}
	for (auto i = 0; i < col;++i)
		for (auto j = begin; j < end; ++j)
			this->mat[i][j] = temp.mat[i][j - begin];
}
int matrix::size() { if (row == col)return row; else return row > col ? col : row; }
int matrix::size() const { if (row == col)return row; else return row > col ? col : row; }
bool matrix::check_tol() const
{
	  double tol = 10e-16;
	for (auto i = 1;i < row;++i)
		for (auto j = 0;j < i;++j)
		{
			if (fabs(mat[i][j]) >= tol || std::isnan(mat[i][j]))
				return false;
		}
	return true;
}
void matrix::set_num_threads(size_t nt)
{
	num_of_thread = nt;
}
size_t matrix::get_num_threads() { return num_of_thread; }

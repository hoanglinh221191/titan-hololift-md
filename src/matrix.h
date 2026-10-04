#ifndef MATRIX_H
#define MATRIX_H
#include<iostream>
using namespace std;
class matrix
{
	double* temp_p;
	int impl_given_rotation(matrix& G1, int begin, int end, int new_row);
	void impl_mul(matrix& rhs, matrix& result, int begin, int end);
	size_t num_of_thread;
public:
	double** mat;
	int col;
	int row;
	matrix();
	matrix(int size);
	matrix(int _row, int _col);
	matrix(const matrix& rhs);
	matrix(matrix&& rhs);
	~matrix();
	matrix operator+(matrix& rhs);
	matrix operator+(matrix&& rhs);
	matrix operator*(matrix& rhs);
	matrix operator*(matrix&& rhs);
	matrix& operator=(matrix& rhs);
	matrix& operator=(matrix&& rhs);
	matrix& move(matrix& rhs);
	friend ostream& operator<<(ostream& os, const matrix& Obj);
	int given_rotation(int& begin, int end, size_t nt);
	void transpose();
	void special_mul(matrix& rhs, int begin, int end);
	void special_mul_2(matrix& rhs, int begin, int end);
	bool check_tol() const;
	int diagonalization(size_t nt);
	int size();
	int size() const;
	void set_num_threads(size_t nt);
	size_t get_num_threads();
};
#endif
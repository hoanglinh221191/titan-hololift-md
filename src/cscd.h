#ifndef CSCD_H
#define CSCD_H
#include<vector>
#include<fstream>
using namespace std;
struct element
{
	int size;
	double time;
	vector<double> distance;
	element() :size{ 1 }, time{} { distance.reserve(100); }
	element(double _time, vector<double>& _distance);
};
class res_cscd
{
	vector<element> dis_res;
public:
	res_cscd(){ dis_res.reserve(40); }

	res_cscd(double time, vector<double>& in);
	void emplace(double time, vector<double>& dis);

	double time(size_t i, size_t j) ;
	double distance(size_t i, size_t j) ;
	void print2file(fstream& out);

	void set(size_t i, double time, vector<double>& dis);
};
class all_cscd
{
	vector<res_cscd> CSCD;
public:
	all_cscd();
	all_cscd(size_t size);
	void set_size(size_t size);
	void emplace(size_t size, size_t pos, size_t pos_res_cscd, double time, vector<double>& dis);
	int printResult(vector<string>& list_file_name);
};
#endif

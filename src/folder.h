#ifndef FOLDER_H
#define FOLDER_H
#include <vector>
#include<string>
#include<filesystem>
#include"arguments.h"
#include"miscellaneous.h"
#include"force.h"
namespace fs = std::filesystem;
class folders
{
	std::vector<fs::path> paths;
	int add_input_paths(const arguments& Arguments, vector<string>& input_names);
public:
	int add_paths(const arguments& Arguments, vector<string>& input_names);
	template<typename T>
	void emplace_back(T&& p) { paths.emplace_back(p); }
	// 0 when done or not needed (no -opdbsep); 1 after printing the folder it could not create.
	int create_folders(arguments& Arguments);
	string get_path(size_t i) const;
	int out_to_folders(vector<double>& E_e, vector<double>& E_v, vector<double>& time_of_snap, vector<vector<double> >& E_e_vector, vector<vector<double> >& E_v_vector, arguments& Arguments);
	int out_force_to_folders(vector<force>& F_e, vector<force>& F_v, vector<double>& time_of_snap, vector<vector<force> >& F_e_vector, vector<vector<force> >& F_v_vector, arguments& Arguments);
	size_t get_size() const;
	fs::path path(size_t i);
	~folders() = default;
};
#endif // !FOLDER_H


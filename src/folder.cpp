#include<fstream>
#include<filesystem>
#include"folder.h"
#include<algorithm>
#include <stdexcept>
#include<iostream>
namespace fs = std::filesystem;
int folders::add_paths(const arguments& Arguments, vector<string>& input_lines)
{
	if (Arguments.flag_separation_results == false && Arguments.flag_all_folders == false)
	{
		paths.emplace_back(".");
		return 1;
	}
	if (Arguments.flag_all_folders == true)
	{
		if (!paths.empty()) return 1;
		fs::path here{ "." };
		folders temp_paths;
		for (auto& p : fs::recursive_directory_iterator(here))
		{
			if (p.path().extension() == ".pdb")
				temp_paths.emplace_back(p.path().parent_path().string() + "/" + p.path().stem().string());
		}
		for (size_t i{}; i < temp_paths.get_size(); ++i)
		{
			if (Arguments.flag_1_pdb == false && temp_paths.path(i).stem().string().substr(0, 3) == "lg_")
			{
				for (size_t j{}; j < temp_paths.get_size(); ++j)
					if (i != j && temp_paths.path(j).stem().string().substr(0, 4) == "rec_"
						&& temp_paths.path(i).parent_path().string() == temp_paths.path(j).parent_path().string()
						&& temp_paths.path(i).stem().string().substr(3) == temp_paths.path(j).stem().string().substr(4))
					{
						size_t find{ temp_paths.path(i).stem().string().substr(3).find(".pdb") != string::npos ? temp_paths.path(i).stem().string().substr(3).find(".pdb") : temp_paths.path(i).stem().string().length() - 1};
						paths.emplace_back(temp_paths.path(i).parent_path().string() + "/" + temp_paths.path(i).stem().string().substr(3, find - 2));
					}
			}
			else if (Arguments.flag_1_pdb == false && temp_paths.path(i).stem().string().substr(0, 4) == "rec_")
				continue;
			else { paths.emplace_back(temp_paths.path(i)); }
		}
		return 0;
	}
	else if(Arguments.flag_listfile==true)
	{
		fstream input(Arguments.list_filename, ios::in);
		for (string line_temp{}; getline(input, line_temp);)
		{
			if (line_temp.find("#") != string::npos)
				line_temp = line_temp.substr(0, line_temp.find("#"));

			line_temp.erase(remove(line_temp.begin(), line_temp.end(), ' '), line_temp.end());

			if (line_temp.length() >= 1)
			{
				if (line_temp.length()>4 && line_temp.substr(line_temp.length() - 4) == ".pdb")
					line_temp = line_temp.substr(0, line_temp.length() - 4);
				input_lines.emplace_back(line_temp);
				paths.emplace_back("./" + line_temp); 
			}
		}
		input.close();
	}
	else if (Arguments.flag_pattern_input == true && Arguments.flag_range_input == true)
	{
		for (int i{ Arguments.b_input }; i <= Arguments.e_input; i+=Arguments.step_input)
		{
			string name{ Arguments.input_pattern.substr(0,Arguments.input_pattern.find("*")) + to_string(i)
				+ Arguments.input_pattern.substr(Arguments.input_pattern.find("*") + 1) };

			if (name.length() > 4 && name.substr(name.length() - 4) == ".pdb")
				name = name.substr(0, name.length() - 4);

			paths.emplace_back("./"+name);
		}
	}
	else if (Arguments.flag_pattern_input == true && Arguments.flag_range_input == false)
	{
		for (int i{ Arguments.b_input };; i += Arguments.step_input)
		{
			string test_name{ Arguments.input_pattern.substr(0,Arguments.input_pattern.find("*")) + to_string(i)
				+ Arguments.input_pattern.substr(Arguments.input_pattern.find("*") + 1) };

			if (test_name.length() > 4 && test_name.substr(test_name.length() - 4) == ".pdb")
				test_name = test_name.substr(0, test_name.length() - 4);

			string open_name{ test_name };
			if (Arguments.flag_1_pdb == false) open_name = "rec_" + open_name;
			if (open_name.length() <= 4 || open_name.substr(open_name.length() - 4) != ".pdb")
				open_name += ".pdb";
			fstream test{ open_name,ios::in };
			if (test.is_open())
				paths.emplace_back("./" + test_name);
			else
				break;
		}
	}
	else 
	{
		string name{ Arguments.pdb_name };
		if (Arguments.pdb_name.length() > 4 && Arguments.pdb_name.substr(Arguments.pdb_name.length() - 4) == ".pdb")
			name = Arguments.pdb_name.substr(0, Arguments.pdb_name.length() - 4);
		paths.emplace_back("./"+name);
	}
	return 0;
}

int folders::create_folders(arguments& Arguments)
{
	if (Arguments.flag_separation_results == false) return 1;
	for (auto& p : paths)
	{
		fs::create_directory(p);
	}
	return 0;
}
string folders::get_path(size_t i) const
{
	if (i > paths.size() || paths.empty()) { cout << "Out of range (path)\n"; throw std::out_of_range("out of range (path)"); }
	return paths[i].string();
}
size_t folders::get_size() const { return paths.size(); }
fs::path folders::path(size_t i)
{
	if (i > paths.size() || paths.empty()) { cout << "Out of range(path)\n"; throw std::out_of_range("out of range (path)"); }
	return paths[i];
}
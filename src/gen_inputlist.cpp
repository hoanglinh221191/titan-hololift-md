#include "arguments.h"
#include "folder.h"
#include "titan_ui.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


using namespace std;
int gen_input_list(arguments &Arguments, vector<string> &input_line,
                   folders &paths, fstream &log, int &input_name) {
  if (Arguments.flag_listfile == true) {
    fstream input(Arguments.list_filename, ios::in);
    for (string line_temp{}; getline(input, line_temp);) {
      if (line_temp.find("#") != string::npos)
        line_temp = line_temp.substr(0, line_temp.find("#"));

      line_temp.erase(remove(line_temp.begin(), line_temp.end(), ' '),
                      line_temp.end());

      if (line_temp.length() >= 1) {
        if (line_temp.length() > 4 &&
            line_temp.substr(line_temp.length() - 4) ==
                Arguments.get_input_file_type())
          line_temp = line_temp.substr(0, line_temp.length() - 4);
        input_line.emplace_back(line_temp);
      }
    }
    input.close();
  } else if (Arguments.flag_pattern_input == true) {
    if (Arguments.flag_range_input == true)
      for (int i = Arguments.b_input; i <= Arguments.e_input;
           i += Arguments.step_input)
        input_line.emplace_back(to_string(i));
    else {
      for (int i = Arguments.b_input;; i += Arguments.step_input)
        if (i - Arguments.b_input < paths.get_size())
          input_line.emplace_back(to_string(i));
    }
  } else if (Arguments.flag_all_folders == true) {
    for (size_t i{}; i < paths.get_size(); ++i)
      input_line.emplace_back(paths.get_path(i));
  } else {
    std::cout
        << "\033[1;36m[TITAN]\033[0m \033[1;33m>> Auto-Detect Mode:\033[0m "
           "Scanning for input trajectories (no list file provided)...\n";
    TitanUI::log_kv(log, "Input Mode", "Auto-detect (No explicit list)");
  }

  input_name = Arguments.b_input;
  if (!input_line.empty())
    Arguments.set_number_of_files(input_line.size());
  return 0;
}
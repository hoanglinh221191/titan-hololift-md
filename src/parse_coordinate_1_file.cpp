#include "MOL.h"
#include "functions.h"
#include "titan_charconv_compat.h"
#include "titan_ui.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace std;

namespace {

std::string_view trim_ascii_ws(std::string_view text) {
  while (!text.empty() &&
         (text.front() == ' ' || text.front() == '\t' || text.front() == '\r')) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
    text.remove_suffix(1);
  }
  return text;
}

bool try_parse_title_time(std::string_view line, double &time_ps) {
  if (!(line.rfind("TITLE", 0) == 0))
    return false;

  size_t value_pos = line.find("t=");
  size_t value_offset = 2;
  if (value_pos == std::string_view::npos) {
    value_pos = line.find('=');
    value_offset = 1;
  }
  if (value_pos == std::string_view::npos)
    return false;

  size_t value_begin = value_pos + value_offset;
  size_t value_end = line.find("step", value_begin);
  if (value_end == std::string_view::npos)
    value_end = line.size();

  std::string_view value_sv = trim_ascii_ws(line.substr(value_begin, value_end - value_begin));
  if (value_sv.empty())
    return false;

  const char *begin = value_sv.data();
  const char *end = begin + value_sv.size();
  auto [ptr, ec] = titan::from_chars_compat(begin, end, time_ps);
  if (ec == std::errc() && ptr != begin)
    return true;

  try {
    time_ps = std::stod(std::string(value_sv));
    return true;
  } catch (...) {
    return false;
  }
}

template <typename MultiMolType>
void discard_uncommitted_frame(MultiMolType &mol, size_t frame_index) {
  if (frame_index < mol.m_mol.size())
    mol.m_mol.resize(frame_index);
}

} // namespace

int helper_parse_coor_not_copy(Multi_MOL &pro_in, Multi_MOL &lig,
                               Multi_MOL &water_traj, groups_cal &groups,
                               arguments &Arguments, box &Box_temp,
                               bool &flag_set_time, double &time_of_snap_temp,
                               fstream &input_pdb, size_t &I,
                               bool &flag_set_box, ::info &output_info,
                               string protein_1, size_t &atom_num,
                               string input_name) {
  if (!protein_1.empty() && protein_1.back() == '\r') {
    protein_1.pop_back();
  }

  if (flag_set_time == false) {
    time_of_snap_temp = static_cast<double>(I);
    if (try_parse_title_time(protein_1, time_of_snap_temp))
      flag_set_time = true;
  }

  if (protein_1.length() >= 55 && protein_1.substr(0, 6) == "CRYST1") {
    Box_temp =
        box(stod(protein_1.substr(6, 10)), stod(protein_1.substr(16, 9)),
            stod(protein_1.substr(25, 9)), stod(protein_1.substr(34, 7)),
            stod(protein_1.substr(41, 7)), stod(protein_1.substr(48, 7)));
    if (flag_set_box == false && I != 0)
      flag_set_box = true;
  }
  const bool is_atom_record =
      protein_1.length() >= 55 &&
      (protein_1.rfind("ATOM", 0) == 0 || protein_1.rfind("HETATM", 0) == 0);
  if (is_atom_record) {
    ++atom_num;
    if (Arguments.flag_end_set && time_of_snap_temp > Arguments.end_time)
      return 1;
    if (Arguments.flag_begin_set == false && I == 0)
      output_info.set_current_begin_time(time_of_snap_temp);
    if (check_in_time(Arguments, time_of_snap_temp) == true) {
      string atom{protein_1.substr(12, 4)};
      atom.erase(remove(atom.begin(), atom.end(), ' '), atom.end());
      string resname{protein_1.substr(17, 4)};
      resname.erase(remove(resname.begin(), resname.end(), ' '), resname.end());
      auto make_atom_info = [&]() {
        mol_info atom_info(stoull(protein_1.substr(22, 4)), string("NULL"), atom,
                           protein_1.substr(21, 1), resname);
        atom_info.set_original_atom_id(atom_num - 1);
        return atom_info;
      };
      if (!groups.empty()) {
        if (groups.get_group(groups.group_1_index())->check_in(atom_num))
          pro_in.emplace(I, Box_temp, time_of_snap_temp, Arguments.force_f,
                         make_atom_info(),
                         [&] {
                           MOL m;
                           m.pos.x.emplace_back(stod(protein_1.substr(31, 8)));
                           m.pos.y.emplace_back(stod(protein_1.substr(39, 8)));
                           m.pos.z.emplace_back(stod(protein_1.substr(47, 8)));
                           return m;
                         }());
        if (groups.get_group(groups.group_2_index())->check_in(atom_num))
          lig.emplace(I, Box_temp, time_of_snap_temp, Arguments.force_f,
                      make_atom_info(),
                      [&] {
                        MOL m;
                        m.pos.x.emplace_back(stod(protein_1.substr(31, 8)));
                        m.pos.y.emplace_back(stod(protein_1.substr(39, 8)));
                        m.pos.z.emplace_back(stod(protein_1.substr(47, 8)));
                        return m;
                      }());
        if (Arguments.n_wat >= 0 && Arguments.n_wat < groups.group_num() &&
            groups.get_group(Arguments.n_wat)->check_in(atom_num))
          water_traj.emplace(
              I, Box_temp, time_of_snap_temp, Arguments.force_f,
              make_atom_info(),
              [&] {
                MOL m;
                m.pos.x.emplace_back(stod(protein_1.substr(31, 8)));
                m.pos.y.emplace_back(stod(protein_1.substr(39, 8)));
                m.pos.z.emplace_back(stod(protein_1.substr(47, 8)));
                return m;
              }(),
              "water");
      } else {
        if (atom_num < Arguments.begin_mol_2)
          pro_in.emplace(I, Box_temp, time_of_snap_temp, Arguments.force_f,
                         make_atom_info(),
                         [&] {
                           MOL m;
                           m.pos.x.emplace_back(stod(protein_1.substr(31, 8)));
                           m.pos.y.emplace_back(stod(protein_1.substr(39, 8)));
                           m.pos.z.emplace_back(stod(protein_1.substr(47, 8)));
                           return m;
                         }());
        else
          lig.emplace(I, Box_temp, time_of_snap_temp, Arguments.force_f,
                      make_atom_info(),
                      [&] {
                        MOL m;
                        m.pos.x.emplace_back(stod(protein_1.substr(31, 8)));
                        m.pos.y.emplace_back(stod(protein_1.substr(39, 8)));
                        m.pos.z.emplace_back(stod(protein_1.substr(47, 8)));
                        return m;
                      }());
      }
    } else {
      flag_set_time = false;
      flag_set_box = false;
    }
  }
  const bool is_end_marker =
      (protein_1 == "ENDMDL" || protein_1 == "END" ||
       protein_1.rfind("ENDMDL", 0) == 0 || protein_1.rfind("END", 0) == 0);
  if (is_end_marker && check_in_time(Arguments, time_of_snap_temp) == true) {
    const bool pro_has_current_frame =
        I < pro_in.m_mol.size() && !pro_in.m_mol[I].pos.x.empty();
    const bool lig_has_current_frame =
        I < lig.m_mol.size() && !lig.m_mol[I].pos.x.empty();
    const bool frame_complete =
        pro_has_current_frame &&
        (!Arguments.needs_mol2_coordinates() || lig_has_current_frame);

    if (!frame_complete) {
      discard_uncommitted_frame(pro_in, I);
      discard_uncommitted_frame(lig, I);
      discard_uncommitted_frame(water_traj, I);
      atom_num = 0;
      flag_set_time = false;
      flag_set_box = false;
      return 0;
    }

    pro_in.check_and_pop_size_2last_snaps(I, " of molecule 1 ");
    if (Arguments.needs_mol2_coordinates())
      lig.check_and_pop_size_2last_snaps(I, " of molecule 2 ");

    if (flag_set_box == false && I != 0 && I < pro_in.m_mol.size()) {
      pro_in.m_mol[I].Box = pro_in.m_mol[I - 1].Box;
      if (Arguments.needs_mol2_coordinates() && I < lig.m_mol.size())
        lig.m_mol[I].Box = lig.m_mol[I - 1].Box;
    } else
      flag_set_box = false;

    ++I;
    atom_num = 0;
    flag_set_time = false;
    flag_set_box = false;
  }

  return 0;
}

void take_coordinate(Multi_MOL &pro_in, Multi_MOL &lig, Multi_MOL &water_traj,
                     groups_cal &groups, string input_name, string force_f,
                     arguments &Arguments, const amino_acid &amine,
                     ::info &output_info, int flag_raw, size_t max_frames) {
  auto try_direct_routed_parse = [&]() -> bool {
    auto route_plan = build_pdb_route_plan(groups, Arguments);
    if (!route_plan)
      return false;

    std::ifstream routed_input(input_name, std::ios::binary);
    if (!routed_input.is_open())
      return false;

    pro_in.m_mol.clear();
    lig.m_mol.clear();
    water_traj.m_mol.clear();

    double next_target_time_state = -1e18;
    const bool capture_topology =
        pro_in.info.empty() ||
        (Arguments.needs_mol2_coordinates() && lig.info.empty()) ||
        (Arguments.n_wat >= 0 && water_traj.info.empty());
    const size_t frame_limit =
        max_frames > 0 ? max_frames : std::numeric_limits<size_t>::max();

    Multi_MOL membrane_out;
    auto parsed = read_routed_pdb_chunk(
        routed_input, input_name, frame_limit, force_f, Arguments, output_info,
        pro_in, lig, water_traj, membrane_out, route_plan.value(),
        next_target_time_state, capture_topology);
    if (!parsed) {
      TitanUI::print_warning("Falling back to legacy 1-PDB parser: " +
                             parsed.error());
      pro_in.m_mol.clear();
      lig.m_mol.clear();
      water_traj.m_mol.clear();
      return false;
    }

    if (pro_in.m_mol.size() >= 2) {
      output_info.set_current_dt(
          pro_in.m_mol[pro_in.m_mol.size() - 1].time_of_snap -
          pro_in.m_mol[pro_in.m_mol.size() - 2].time_of_snap);
    }

    if (Arguments.flag_end_set == false && !pro_in.m_mol.empty()) {
      output_info.set_current_end_time(
          pro_in.m_mol[pro_in.m_mol.size() - 1].time_of_snap);
    }

    pro_in.gen_vector_parameters();
    lig.gen_vector_parameters();
    if (Arguments.n_wat >= 0)
      water_traj.gen_vector_parameters();

    return true;
  };

  if (try_direct_routed_parse())
    return;

  box Box_temp;
  double time_of_snap_temp{};
  bool flag_set_time{false}, flag_set_box{false};
  fstream input_pdb;
  input_pdb.open(input_name, ios::in);
  if (!input_pdb.is_open()) {
    cout << "Can not open " << input_name << " file!" << '\n';
    exit(EXIT_FAILURE);
  }

  size_t I{}, atom_num{};
  bool flag_end{true};
  const size_t flush_snap_count =
      max_frames > 0 ? std::min<size_t>(10, max_frames) : 10;
  while (flag_end == true) {
    if (max_frames > 0 && pro_in.m_mol.size() >= max_frames) {
      flag_end = false;
      break;
    }
    if (max_frames > 0 && lig.m_mol.size() >= max_frames) {
      flag_end = false;
      break;
    }

    vector<string> buffer_string;
    size_t count_snap{};
    for (string protein_1{}; getline(input_pdb, protein_1);) {
      if (!protein_1.empty() && protein_1.back() == '\r') {
        protein_1.pop_back();
      }
      buffer_string.emplace_back(protein_1);
      if (protein_1 == "ENDMDL" || protein_1 == "END") {
        ++count_snap;
        if (count_snap >= flush_snap_count) {
          for (size_t i{}; i < buffer_string.size(); ++i)
            if (1 == helper_parse_coor_not_copy(
                         pro_in, lig, water_traj, groups, Arguments, Box_temp,
                         flag_set_time, time_of_snap_temp, input_pdb, I,
                         flag_set_box, output_info, buffer_string[i], atom_num,
                         input_name)) {
              flag_end = false;
              break;
            }
          buffer_string.clear();
          count_snap = 0;
        }
      }
      if (flag_end == false)
        break;
    }
    if (!buffer_string.empty()) {
      if (buffer_string[buffer_string.size() - 1] != "ENDMDL" &&
          buffer_string[buffer_string.size() - 1] != "END")
        buffer_string.emplace_back("ENDMDL");
      for (size_t i{}; i < buffer_string.size(); ++i)
        if (1 == helper_parse_coor_not_copy(
                     pro_in, lig, water_traj, groups, Arguments, Box_temp,
                     flag_set_time, time_of_snap_temp, input_pdb, I,
                     flag_set_box, output_info, buffer_string[i], atom_num,
                     input_name)) {
          flag_end = false;
          buffer_string.clear();
          break;
        }
    }
    flag_end = false;
  }
  input_pdb.close();

  if (pro_in.m_mol.size() >= 2)
    output_info.set_current_dt(
        pro_in.m_mol[pro_in.m_mol.size() - 1].time_of_snap -
        pro_in.m_mol[pro_in.m_mol.size() - 2].time_of_snap);

  if (Arguments.flag_end_set == false && !pro_in.m_mol.empty())
    output_info.set_current_end_time(
        pro_in.m_mol[pro_in.m_mol.size() - 1].time_of_snap);

  if (Arguments.flag_begin_set == true && pro_in.m_mol.empty() &&
      time_of_snap_temp < Arguments.begin_time) {
    cout << "WARNING!!! Your begin time is " << Arguments.begin_time
         << " ps while time of the last snapshot is smaller than that!\n";
  }

  pro_in.gen_vector_parameters();
  lig.gen_vector_parameters();
  if (Arguments.n_wat >= 0)
    water_traj.gen_vector_parameters();
}

int check_and_settime(double &time_of_snap, string protein_1,
                      fstream &input_pdb, arguments &Arguments) {
  if (protein_1.length() >= 5 && protein_1.find("TITLE") != string::npos) {
    size_t found = protein_1.find("=");
    size_t found_2 = protein_1.find("step"),
           time_length{protein_1.length() < found_2 ? protein_1.length()
                                                    : found_2};

    if (found != string::npos && protein_1.length() > found + 2) {
      time_of_snap =
          (stod(protein_1.substr(found + 1, time_length - found - 1)));
      return 0;
    } else {
      getline(input_pdb, protein_1);
      found = protein_1.find("t=");
      found_2 = protein_1.find("step");
      time_length = protein_1.length() < found_2 ? protein_1.length() : found_2;
      if (found != string::npos) {
        time_of_snap =
            (stod(protein_1.substr(found + 2, time_length - found - 2)));
      }
      return 0;
    }
  } else
    return 1;
}

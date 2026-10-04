#include "Multi_MOL.h"
#include "functions.h"
#include "titan_charconv_compat.h"
#include "titan_ui.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

using std::string_view;

namespace {

constexpr size_t kPdbBufferSize = 1024 * 1024;
constexpr double kPdbTimeTolerancePs = 1e-4;

double fast_parse_double(std::string_view data) {
  const char *begin = data.data();
  const char *end = begin + data.size();
  while (begin < end && *begin == ' ')
    ++begin;

  double value = 0.0;
  auto [ptr, ec] = titan::from_chars_compat(begin, end, value);
  if (ec == std::errc())
    return value;
  return 0.0;
}

size_t fast_parse_u64(std::string_view data) {
  const char *begin = data.data();
  const char *end = begin + data.size();
  while (begin < end && *begin == ' ')
    ++begin;

  size_t value = 0;
  auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec == std::errc())
    return value;
  return 0;
}

constexpr std::string_view trim_spaces(std::string_view sv) {
  sv.remove_prefix(std::min(sv.find_first_not_of(" \t\r\n"), sv.size()));
  sv.remove_suffix(sv.empty() ? 0
                              : sv.size() - sv.find_last_not_of(" \t\r\n") - 1);
  return sv;
}

bool try_parse_title_time(double &time_of_snap, std::string_view protein_1,
                          std::string_view protein_2) {
  if (protein_1.length() < 5 ||
      protein_1.find("TITLE") == std::string_view::npos) {
    return false;
  }

  size_t found = protein_1.find('=');
  size_t found_2 = protein_1.find("step");
  size_t time_length = std::min(protein_1.length(), found_2);

  if (found != std::string_view::npos && protein_1.length() > found + 2) {
    time_of_snap =
        fast_parse_double(protein_1.substr(found + 1, time_length - found - 1));
    return true;
  }

  found = protein_2.find("t=");
  found_2 = protein_2.find("step");
  time_length = std::min(protein_2.length(), found_2);
  if (found != std::string_view::npos) {
    time_of_snap =
        fast_parse_double(protein_2.substr(found + 2, time_length - found - 2));
    return true;
  }

  return false;
}

bool is_atom_record(std::string_view line) {
  return line.length() >= 54 &&
         (line.starts_with("ATOM") || line.starts_with("HETATM"));
}

bool is_end_record(std::string_view line) {
  return line.starts_with("ENDMDL") || line.starts_with("END");
}

void parse_cryst1_box(box &box_temp, std::string_view line) {
  box_temp = box(fast_parse_double(line.substr(6, 9)),
                 fast_parse_double(line.substr(15, 9)),
                 fast_parse_double(line.substr(24, 9)),
                 fast_parse_double(line.substr(33, 7)),
                 fast_parse_double(line.substr(40, 7)),
                 fast_parse_double(line.substr(47, 7)));
}

mol_info make_atom_info(std::string_view line,
                        size_t original_atom_id = std::numeric_limits<size_t>::max()) {
  std::string_view atom = trim_spaces(line.substr(12, 4));
  std::string_view resname = trim_spaces(
      line.length() >= 21 ? line.substr(16, 5) : std::string_view{});
  if (resname.empty() && line.length() >= 21) {
    resname = trim_spaces(line.substr(17, 4));
  }

  mol_info atom_info(fast_parse_u64(line.substr(22, 4)), "NULL",
                     std::string(atom), std::string(line.substr(21, 1)),
                     std::string(resname));
  if (original_atom_id != std::numeric_limits<size_t>::max())
    atom_info.set_original_atom_id(original_atom_id);
  return atom_info;
}

double parse_atom_coord(std::string_view line, size_t offset) {
  return fast_parse_double(line.substr(offset, 8));
}

size_t expected_frame_atoms(const Multi_MOL &mol, size_t frame_index,
                            size_t hinted_count) {
  if (!mol.info.empty())
    return mol.info.size();
  if (frame_index > 0 && frame_index - 1 < mol.m_mol.size())
    return mol.m_mol[frame_index - 1].size();
  if (hinted_count > 0)
    return hinted_count;
  if (frame_index < mol.m_mol.size())
    return mol.m_mol[frame_index].size();
  return 0;
}

template <typename Sink>
std::expected<size_t, std::string> parse_pdb_stream_impl(
    std::istream &input_pdb, std::string_view input_name, size_t max_frames,
    std::string_view force_f, arguments &Arguments, ::info &output_info,
    double &next_target_time_state, Sink &sink) {
  if (!input_pdb.good() && !input_pdb.eof())
    return std::unexpected("Input PDB stream is not readable.");

  double time_of_snap_temp = 0.0;
  box box_temp;
  bool flag_set_time = false;
  bool flag_set_box = false;
  size_t committed_frames = 0;
  size_t source_atom_index = 0;

  std::vector<char> buffer(kPdbBufferSize);
  std::string remainder;
  remainder.reserve(512);

  bool stop_due_to_end = false;

  auto process_line = [&](std::string_view line,
                          std::string &error_message) -> bool {
    if (line.empty())
      return true;

    if (!flag_set_time) {
      time_of_snap_temp = static_cast<double>(committed_frames);
      if (try_parse_title_time(time_of_snap_temp, line, line))
        flag_set_time = true;
    }

    if (!flag_set_box && line.length() >= 54 && line.starts_with("CRYST1")) {
      parse_cryst1_box(box_temp, line);
      flag_set_box = true;
    }

    if (is_atom_record(line)) {
      ++source_atom_index;

      if (Arguments.flag_end_set && time_of_snap_temp > Arguments.end_time) {
        stop_due_to_end = true;
        return false;
      }

      if (!Arguments.flag_begin_set && committed_frames == 0)
        output_info.set_current_begin_time(time_of_snap_temp);

      if (check_in_time(Arguments, time_of_snap_temp)) {
        bool accept_dt = true;
        if (Arguments.dt > 0.0) {
          if (next_target_time_state < -1e17) {
            next_target_time_state = Arguments.flag_begin_set
                                         ? Arguments.begin_time
                                         : time_of_snap_temp;
          }
          if (time_of_snap_temp < next_target_time_state - kPdbTimeTolerancePs)
            accept_dt = false;
        }

        if (accept_dt) {
          if (!sink.append_atom(committed_frames, source_atom_index, line,
                                box_temp, time_of_snap_temp, force_f,
                                error_message)) {
            return false;
          }
        }
      }
    }

    if (is_end_record(line)) {
      std::string warning_message;
      const bool committed =
          sink.finalize_frame(committed_frames, flag_set_box, time_of_snap_temp,
                              warning_message);
      if (!warning_message.empty())
        TitanUI::print_warning(warning_message);

      if (committed) {
        ++committed_frames;
        if (Arguments.dt > 0.0) {
          while (next_target_time_state <=
                 time_of_snap_temp + kPdbTimeTolerancePs) {
            next_target_time_state += Arguments.dt;
          }
        }
      }

      flag_set_time = false;
      flag_set_box = false;
      source_atom_index = 0;
    }

    return true;
  };

  while (input_pdb) {
    input_pdb.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytes_read = input_pdb.gcount();
    if (bytes_read <= 0)
      break;

    std::string_view chunk(buffer.data(), static_cast<size_t>(bytes_read));
    size_t start = 0;
    size_t end = chunk.find('\n');
    bool hit_limit = false;

    while (end != std::string_view::npos) {
      std::string_view line_view = chunk.substr(start, end - start);
      if (!line_view.empty() && line_view.back() == '\r')
        line_view.remove_suffix(1);

      std::string materialized_line;
      if (!remainder.empty()) {
        remainder.append(line_view);
        materialized_line = remainder;
        remainder.clear();
      }

      std::string_view effective_line =
          materialized_line.empty() ? line_view : std::string_view(materialized_line);

      if (!effective_line.empty()) {
        std::string error_message;
        if (!process_line(effective_line, error_message)) {
          if (!error_message.empty())
            return std::unexpected(error_message);

          if (stop_due_to_end) {
            input_pdb.clear();
            input_pdb.seekg(0, std::ios_base::end);
            input_pdb.setstate(std::ios::eofbit);
            sink.discard_frame(committed_frames);
            return committed_frames;
          }
        }
      }

      start = end + 1;
      if (max_frames > 0 && committed_frames >= max_frames) {
        hit_limit = true;
        break;
      }
      end = chunk.find('\n', start);
    }

    if (hit_limit) {
      const std::streamoff rewind_bytes =
          static_cast<std::streamoff>(bytes_read) -
          static_cast<std::streamoff>(start);
      if (rewind_bytes > 0)
        input_pdb.seekg(-rewind_bytes, std::ios_base::cur);
      remainder.clear();
      break;
    }

    if (start < static_cast<size_t>(bytes_read)) {
      remainder.assign(chunk.substr(start));
    } else {
      remainder.clear();
    }
  }

  if (!remainder.empty() && !(max_frames > 0 && committed_frames >= max_frames)) {
    std::string error_message;
    if (!process_line(remainder, error_message) && !error_message.empty())
      return std::unexpected(error_message);
  }

  if (sink.has_pending_frame(committed_frames) &&
      !(max_frames > 0 && committed_frames >= max_frames)) {
    std::string warning_message;
    const bool committed = sink.finalize_frame(committed_frames, flag_set_box,
                                               time_of_snap_temp, warning_message);
    if (!warning_message.empty())
      TitanUI::print_warning(warning_message);
    if (committed) {
      ++committed_frames;
      if (Arguments.dt > 0.0) {
        while (next_target_time_state <= time_of_snap_temp + kPdbTimeTolerancePs)
          next_target_time_state += Arguments.dt;
      }
    }
  }

  return committed_frames;
}

struct SinglePdbSink {
  Multi_MOL &out;

  bool append_atom(size_t frame_index, size_t source_atom_index,
                   std::string_view line, const box &box_type, double time_ps,
                   std::string_view force_f, std::string &error_message) {
    const size_t reserve_hint =
        expected_frame_atoms(out, frame_index, frame_index == 0 ? out.info.size() : 0);
    if (frame_index >= out.m_mol.size())
      out.begin_frame(frame_index, box_type, time_ps, force_f, reserve_hint);

    if (frame_index == 0 && out.info.size() < out.m_mol[0].pos.x.size() + 1) {
      out.append_topology_if_frame0(make_atom_info(line, source_atom_index - 1));
    }

    out.append_xyz(frame_index, parse_atom_coord(line, 30),
                   parse_atom_coord(line, 38), parse_atom_coord(line, 46));
    (void)error_message;
    return true;
  }

  bool finalize_frame(size_t frame_index, bool had_box, double time_ps,
                      std::string &warning_message) {
    if (!has_pending_frame(frame_index))
      return false;

    const size_t current_atoms = out.m_mol[frame_index].size();
    const size_t expected_atoms = expected_frame_atoms(out, frame_index, 0);
    if (expected_atoms > 0 && current_atoms != expected_atoms) {
      warning_message = "Incomplete frame detected at " +
                        std::to_string(time_ps) + " ps. Skipping...";
      out.discard_pending_frame(frame_index);
      return false;
    }

    if (!had_box && frame_index != 0 && frame_index < out.m_mol.size()) {
      out.m_mol[frame_index].Box = out.m_mol[frame_index - 1].Box;
    }

    return true;
  }

  void discard_frame(size_t frame_index) { out.discard_pending_frame(frame_index); }

  bool has_pending_frame(size_t frame_index) const {
    return frame_index < out.m_mol.size() && !out.m_mol[frame_index].pos.x.empty();
  }
};

struct RoutedPdbSink {
  Multi_MOL &pro;
  Multi_MOL &lig;
  Multi_MOL &water;
  Multi_MOL &membrane;
  const PdbRoutePlan &route_plan;
  const bool require_lig;
  const bool require_water;
  const bool require_membrane;
  const bool capture_topology;

  bool append_atom(size_t frame_index, size_t source_atom_index,
                   std::string_view line, const box &box_type, double time_ps,
                   std::string_view force_f, std::string &error_message) {
    const PdbRouteEntry route = route_plan.route_for(source_atom_index);
    auto append_to = [&](Multi_MOL &target_mol, size_t reserve_hint,
                         std::string_view target_type) {
      if (frame_index >= target_mol.m_mol.size()) {
        target_mol.begin_frame(frame_index, box_type, time_ps, force_f,
                               expected_frame_atoms(target_mol, frame_index,
                                                    reserve_hint),
                               target_type);
      }
      if (capture_topology &&
          (target_mol.info.empty() ||
           target_mol.info.size() < target_mol.m_mol[0].pos.x.size() + 1)) {
        target_mol.append_topology_if_frame0(
            make_atom_info(line, source_atom_index - 1));
      }
      target_mol.append_xyz(frame_index, parse_atom_coord(line, 30),
                            parse_atom_coord(line, 38),
                            parse_atom_coord(line, 46));
    };

    switch (route.target) {
    case PdbRouteTarget::Pro:
      append_to(pro, std::max(route_plan.pro_atom_count, pro.info.size()),
                pro.get_mol_type());
      break;
    case PdbRouteTarget::Lig:
      append_to(lig, std::max(route_plan.lig_atom_count, lig.info.size()),
                lig.get_mol_type());
      break;
    case PdbRouteTarget::Wat:
      append_to(water, std::max(route_plan.wat_atom_count, water.info.size()),
                "water");
      break;
    case PdbRouteTarget::None:
      break;
    }
    if (route_plan.membrane_route_for(source_atom_index) !=
        std::numeric_limits<size_t>::max()) {
      append_to(membrane,
                std::max(route_plan.membrane_atom_count, membrane.info.size()),
                "lipid");
    }
    return true;
  }

  bool finalize_target(Multi_MOL &target_mol, size_t frame_index, bool had_box,
                       size_t hinted_count, bool required,
                       std::string_view label,
                       std::string &warning_message) const {
    if (frame_index >= target_mol.m_mol.size() ||
        target_mol.m_mol[frame_index].pos.x.empty()) {
      if (!required)
        return true;
      warning_message = std::string(label) + " frame is empty. Skipping...";
      return false;
    }

    const size_t current_atoms = target_mol.m_mol[frame_index].size();
    const size_t expected_atoms =
        expected_frame_atoms(target_mol, frame_index, hinted_count);
    if (expected_atoms > 0 && current_atoms != expected_atoms) {
      warning_message = std::string(label) + " frame has " +
                        std::to_string(current_atoms) + " atoms, expected " +
                        std::to_string(expected_atoms) + ". Skipping...";
      return false;
    }

    if (!had_box && frame_index != 0) {
      target_mol.m_mol[frame_index].Box = target_mol.m_mol[frame_index - 1].Box;
    }

    return true;
  }

  bool finalize_frame(size_t frame_index, bool had_box, double time_ps,
                      std::string &warning_message) {
    if (!has_pending_frame(frame_index))
      return false;

    std::string local_warning;
    if (!finalize_target(pro, frame_index, had_box, route_plan.pro_atom_count, true,
                         "Receptor", local_warning) ||
        !finalize_target(lig, frame_index, had_box, route_plan.lig_atom_count,
                         require_lig, "Ligand", local_warning) ||
        !finalize_target(water, frame_index, had_box, route_plan.wat_atom_count,
                         require_water, "Water", local_warning) ||
        !finalize_target(membrane, frame_index, had_box,
                         route_plan.membrane_atom_count, require_membrane,
                         "Membrane", local_warning)) {
      if (!local_warning.empty()) {
        warning_message = local_warning + " Time " + std::to_string(time_ps) +
                          " ps.";
      }
      discard_frame(frame_index);
      return false;
    }

    return true;
  }

  void discard_frame(size_t frame_index) {
    pro.discard_pending_frame(frame_index);
    lig.discard_pending_frame(frame_index);
    water.discard_pending_frame(frame_index);
    membrane.discard_pending_frame(frame_index);
  }

  bool has_pending_frame(size_t frame_index) const {
    const bool pro_pending =
        frame_index < pro.m_mol.size() && !pro.m_mol[frame_index].pos.x.empty();
    const bool lig_pending =
        frame_index < lig.m_mol.size() && !lig.m_mol[frame_index].pos.x.empty();
    const bool water_pending = frame_index < water.m_mol.size() &&
                               !water.m_mol[frame_index].pos.x.empty();
    const bool membrane_pending = frame_index < membrane.m_mol.size() &&
                                  !membrane.m_mol[frame_index].pos.x.empty();
    return pro_pending || lig_pending || water_pending || membrane_pending;
  }
};

} // namespace

std::expected<PdbRoutePlan, std::string>
build_pdb_route_plan(groups_cal &groups, const arguments &Arguments) {
  PdbRoutePlan plan;

  if (groups.empty()) {
    if (Arguments.begin_mol_2 > 0) {
      plan.split_begin_mol_2 = Arguments.begin_mol_2;
      return plan;
    }
    return std::unexpected("No PDB routing groups available.");
  }

  size_t max_source_atom_index = 0;
  auto update_max_index = [&](size_t group_index) {
    for (size_t i = 0; i < groups.groups_atom_number(group_index); ++i) {
      max_source_atom_index =
          std::max(max_source_atom_index, groups.atom_index(group_index, i));
    }
  };

  update_max_index(groups.group_1_index());
  if (Arguments.needs_mol2_coordinates())
    update_max_index(groups.group_2_index());
  if (Arguments.n_wat >= 0 && Arguments.n_wat < static_cast<int>(groups.group_num()))
    update_max_index(static_cast<size_t>(Arguments.n_wat));
  if (Arguments.needs_membrane_coordinates() &&
      Arguments.membrane.resolved_group >= 0)
    update_max_index(static_cast<size_t>(Arguments.membrane.resolved_group));

  plan.atom_routes.resize(max_source_atom_index + 1);
  plan.membrane_routes.assign(max_source_atom_index + 1,
                              std::numeric_limits<size_t>::max());

  auto assign_group = [&](size_t group_index, PdbRouteTarget target,
                          size_t &count_ref) -> std::expected<void, std::string> {
    for (size_t i = 0; i < groups.groups_atom_number(group_index); ++i) {
      const size_t source_atom_index = groups.atom_index(group_index, i);
      if (source_atom_index >= plan.atom_routes.size())
        plan.atom_routes.resize(source_atom_index + 1);

      if (plan.atom_routes[source_atom_index].target != PdbRouteTarget::None) {
        return std::unexpected("Overlapping PDB routing groups are not supported.");
      }

      plan.atom_routes[source_atom_index] = {target, count_ref++};
    }
    return {};
  };

  plan.pro_atom_count = 0;
  plan.lig_atom_count = 0;
  plan.wat_atom_count = 0;

  if (auto assigned =
          assign_group(groups.group_1_index(), PdbRouteTarget::Pro,
                       plan.pro_atom_count);
      !assigned) {
    return std::unexpected(assigned.error());
  }

  if (Arguments.needs_mol2_coordinates()) {
    if (auto assigned =
            assign_group(groups.group_2_index(), PdbRouteTarget::Lig,
                         plan.lig_atom_count);
        !assigned) {
      return std::unexpected(assigned.error());
    }
  }

  if (Arguments.n_wat >= 0 && Arguments.n_wat < static_cast<int>(groups.group_num())) {
    if (auto assigned =
            assign_group(static_cast<size_t>(Arguments.n_wat),
                         PdbRouteTarget::Wat, plan.wat_atom_count);
        !assigned) {
      return std::unexpected(assigned.error());
    }
  }

  if (Arguments.needs_membrane_coordinates()) {
    if (Arguments.membrane.resolved_group < 0)
      return std::unexpected("Membrane group has not been resolved.");
    const size_t membrane_group =
        static_cast<size_t>(Arguments.membrane.resolved_group);
    if (membrane_group >= groups.group_num())
      return std::unexpected("Membrane group index is out of range.");
    for (size_t i = 0; i < groups.groups_atom_number(membrane_group); ++i) {
      const size_t source_atom_index = groups.atom_index(membrane_group, i);
      if (source_atom_index >= plan.membrane_routes.size())
        plan.membrane_routes.resize(source_atom_index + 1,
                                    std::numeric_limits<size_t>::max());
      plan.membrane_routes[source_atom_index] = plan.membrane_atom_count++;
    }
  }

  return plan;
}

std::expected<size_t, std::string>
read_routed_pdb_chunk(std::istream &input_pdb, std::string_view input_name,
                      size_t chunk_size, std::string_view force_f,
                      arguments &Arguments, ::info &output_info,
                      Multi_MOL &pro_out, Multi_MOL &lig_out,
                      Multi_MOL &water_out, Multi_MOL &membrane_out,
                      const PdbRoutePlan &route_plan,
                      double &next_target_time_state, bool capture_topology) {
  RoutedPdbSink sink{pro_out,
                     lig_out,
                     water_out,
                     membrane_out,
                     route_plan,
                     Arguments.needs_mol2_coordinates(),
                     Arguments.n_wat >= 0,
                     Arguments.needs_membrane_coordinates(),
                     capture_topology};
  return parse_pdb_stream_impl(
      input_pdb, input_name,
      chunk_size == std::numeric_limits<size_t>::max() ? 0 : chunk_size, force_f,
      Arguments, output_info, next_target_time_state, sink);
}

int Multi_MOL::parse_pdb(std::string_view input_name, std::string_view force_f,
                         arguments &Arguments, ::info &output_info,
                         size_t max_frames) {
  std::ifstream input_pdb(std::string(input_name), std::ios::binary);
  if (!input_pdb.is_open()) {
    std::cerr << "Can not open " << input_name << " file!\n";
    return 1;
  }

  reset_parser_state();
  SinglePdbSink sink{*this};
  auto parsed = parse_pdb_stream_impl(
      input_pdb, input_name,
      max_frames == std::numeric_limits<size_t>::max() ? 0 : max_frames, force_f,
      Arguments, output_info, next_target_time, sink);
  if (!parsed)
    return 1;

  if (this->m_mol.size() >= 2) {
    output_info.set_current_dt(
        this->m_mol[this->m_mol.size() - 1].time_of_snap -
        this->m_mol[this->m_mol.size() - 2].time_of_snap);
  }

  if (!Arguments.flag_end_set && !this->m_mol.empty()) {
    output_info.set_current_end_time(
        this->m_mol[this->m_mol.size() - 1].time_of_snap);
  }

  if (Arguments.flag_begin_set && this->m_mol.empty() &&
      !input_pdb.fail() && !input_pdb.bad()) {
    std::cout << "WARNING!!! Your begin time is " << Arguments.begin_time
              << " ps while time of the last snapshot is smaller than that!\n";
  }

  return 0;
}

std::expected<size_t, std::string>
Multi_MOL::read_pdb_chunk(std::ifstream &input_pdb, size_t chunk_size,
                          arguments &Arguments, ::info &output_info,
                          std::string_view force_f) {
  if (!input_pdb.is_open())
    return std::unexpected("Input PDB stream is closed.");
  if (input_pdb.eof())
    return 0;

  SinglePdbSink sink{*this};
  auto parsed = parse_pdb_stream_impl(
      input_pdb, "stream",
      chunk_size == std::numeric_limits<size_t>::max() ? 0 : chunk_size, force_f,
      Arguments, output_info, next_target_time, sink);
  if (!parsed)
    return parsed;

  if (this->parameters.charge.empty() && !this->info.empty())
    this->gen_vector_parameters();

  return parsed.value();
}

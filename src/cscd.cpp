// Engine side of -cscd; the distances and files are in impl_cscd_class.cpp.
#include "functions.h"
#include "titan_executor.h"
#include "titan_ui.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// MOL::take_com uses the atoms marked side chain and falls back to the whole
// residue. The marks separate side chains from the backbone only where
// Multi_MOL::check_side_chain() classified amino-acid residues; with
// -side_chain_contact full-res, or without amino acids, every atom is marked
// or none is, and all centres are whole residues.
std::string centre_kind(const Multi_MOL &pro_in, const Multi_MOL &lig) {
  const auto marked = [](const Multi_MOL &mol) {
    const auto side = [](const mol_info &atom) { return atom.is_side_chain(); };
    return std::any_of(mol.info.begin(), mol.info.end(), side) &&
           !std::all_of(mol.info.begin(), mol.info.end(), side);
  };
  const bool mol1 = marked(pro_in);
  const bool mol2 = marked(lig);
  if (mol1 && mol2)
    return "side chain (whole residue for glycine and non-amino-acid "
           "residues)";
  if (!mol1 && !mol2)
    return "whole residue (no side chain is separated from the backbone)";
  return std::string("molecule 1 ") + (mol1 ? "side chain" : "whole residue") +
         ", molecule 2 " + (mol2 ? "side chain" : "whole residue");
}

// Internal residue number of each centre, through its first atom; empty when
// a centre points outside the topology.
std::vector<size_t> internal_residues(const Multi_MOL &mol,
                                      const std::vector<com> &centres) {
  std::vector<size_t> internal;
  internal.reserve(centres.size());
  for (const com &centre : centres) {
    const size_t atom = centre.get_respos();
    if (atom == 0 || atom > mol.info.size())
      return {};
    internal.push_back(mol.info[atom - 1].get_internal_resnr());
  }
  return internal;
}

} // namespace

int impl_cscd(Multi_MOL &pro_in, Multi_MOL &lig, const arguments &Arguments,
              TitanExecutor &executor, size_t threads, size_t num_snap,
              size_t frame, int sc_flag, all_cscd &out_dis) {
  try {
    if (num_snap >= pro_in.m_mol.size() || num_snap >= lig.m_mol.size())
      throw std::runtime_error("frame " + std::to_string(frame + 1) +
                               " is not loaded for both molecules");
    const MOL &mol1 = pro_in.m_mol[num_snap];
    const MOL &mol2 = lig.m_mol[num_snap];
    if (!out_dis.configured()) {
      all_cscd::Options options;
      // 1 = nearest partner only (default); -act sets 0, the prompt 2.
      options.nearest_only = sc_flag == 1;
      options.skip_same_residue = Arguments.self_int;
      options.file_prefix = Arguments.name_out_cal_dis;
      options.time_unit = Arguments.tu;
      options.centre = centre_kind(pro_in, lig);
      if (options.centre.starts_with("whole residue"))
        TitanUI::print_warning(
            "-cscd: no side chain is separated from the backbone "
            "(-side_chain_contact full-res, or no amino acids); the "
            "distances are between whole-residue centres of mass.");
      out_dis.configure(options, internal_residues(pro_in, mol1.Com),
                        internal_residues(lig, mol2.Com));
    }
    out_dis.add_frame(frame, Arguments.output_time(mol1.time_of_snap),
                      mol1.Com, mol2.Com, &executor, threads);
  } catch (const std::exception &e) {
    TitanUI::print_error(std::string("-cscd: ") + e.what());
    return 1;
  }
  return 0;
}

int end_cscd_trajectory(const arguments &Arguments, all_cscd &out_dis,
                        const folders &paths, int input_name) {
  if (!Arguments.cal_dis_sc_contact)
    return 0;
  try {
    std::string directory;
    if (Arguments.flag_separation_results) {
      const size_t trajectory =
          input_name >= Arguments.b_input
              ? static_cast<size_t>(input_name - Arguments.b_input)
              : 0;
      if (trajectory >= paths.get_size())
        throw std::runtime_error("no output folder for trajectory " +
                                 std::to_string(trajectory + 1));
      directory = paths.get_path(trajectory);
    }
    out_dis.end_trajectory(directory);
  } catch (const std::exception &e) {
    TitanUI::print_error(std::string("-cscd: ") + e.what());
    return 1;
  }
  return 0;
}

int print_cscd(const arguments &Arguments, all_cscd &out_dis, fstream &log) {
  if (!Arguments.cal_dis_sc_contact)
    return 1;
  if (!out_dis.configured()) {
    TitanUI::print_warning("-cscd: no frame reached the distance analysis; "
                           "no output written.");
    return 1;
  }
  size_t files = 0;
  try {
    files = out_dis.write("");
  } catch (const std::exception &e) {
    TitanUI::print_error(std::string("-cscd: ") + e.what());
    return 1;
  }
  TitanUI::log_kv(log, "CSCD Output",
                  std::to_string(files) + " files " +
                      Arguments.name_out_cal_dis + "*.dat, " +
                      std::to_string(out_dis.columns()) + " columns, " +
                      std::to_string(out_dis.merged_rows()) + " rows");
  if (out_dis.trajectories() > 1)
    TitanUI::log_kv(log, "CSCD Merged Trajectories",
                    std::to_string(out_dis.trajectories()) + " (" +
                        std::to_string(out_dis.averaged_rows()) +
                        " rows averaged at equal times)");
  return 0;
}

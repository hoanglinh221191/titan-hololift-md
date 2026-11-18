#include "engine/engine_common.h"

namespace engine_detail {

void keep_only_selected_frame(Multi_MOL &mol, std::size_t frame_index) {
  if (mol.m_mol.empty())
    return;
  if (frame_index >= mol.m_mol.size())
    frame_index = mol.m_mol.size() - 1;
  MOL selected_frame = std::move(mol.m_mol[frame_index]);
  mol.m_mol.clear();
  mol.m_mol.push_back(std::move(selected_frame));
}

std::string format_time_ps(double time_ps) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4) << time_ps << " ps";
  return oss.str();
}

std::string progress_source_label(const arguments &Arguments,
                                  const std::string &name_pro) {
  if (Arguments.flag_xdr && !Arguments.traj_file.empty()) {
    return TitanUI::compact_source_label(Arguments.traj_file, 18);
  } else if (Arguments.flag_1_pdb) {
    return TitanUI::compact_source_label(Arguments.pdb_name, 18);
  }
  return TitanUI::compact_source_label(name_pro, 18);
}

std::string pull_direct_selection_rule(const arguments &Arguments) {
  if (Arguments.flag_end_set) {
    return "last snapshot of first input with time <= " +
           format_time_ps(Arguments.end_time);
  }
  return "last available snapshot of first input";
}

void set_pull_snapshot_metadata(const PullDirectSnapshotSelection &selection,
                                de_arg &DE_arguments) {
  DE_arguments.snapshot_source_file = selection.source_file;
  DE_arguments.snapshot_input_mode = selection.input_mode;
  DE_arguments.snapshot_selection_rule = selection.selection_rule;
  DE_arguments.snapshot_time_ps = selection.time_ps;
  DE_arguments.snapshot_frame_index = selection.frame_index;
  DE_arguments.snapshot_metadata_valid = true;
}

std::expected<PullDirectSnapshotSelection, std::string>
prepare_pull_direct_snapshot(arguments &Arguments, de_arg &DE_arguments,
                             const std::string &force_f,
                             const std::string &name_pro,
                             const std::string &name_lig, Multi_MOL &pro_in,
                             Multi_MOL &lig, Multi_MOL &water_traj,
                             groups_cal &Groups_from_index_file,
                             const amino_acid &amine, info &output_info,
                             TitanExecutor &executor, std::size_t num_threads,
                             std::fstream &log) {
  arguments snapshot_args = Arguments;
  snapshot_args.flag_begin_set = false;
  snapshot_args.begin_time = 0.0;
  snapshot_args.dt = 0.0;

  PullDirectSnapshotSelection selection;
  selection.selection_rule = pull_direct_selection_rule(Arguments);

  if (snapshot_args.flag_xdr) {
    ChunkStreamer snapshot_streamer(snapshot_args, output_info,
                                    Groups_from_index_file, force_f, executor,
                                    std::max<std::size_t>(1, num_threads));
    Multi_MOL snapshot_membrane(force_f);
    auto stream_ok = snapshot_streamer.open_streams(
        snapshot_args.pdb_name, name_lig, pro_in, lig, snapshot_membrane);
    if (!stream_ok)
      return std::unexpected(stream_ok.error());

    const std::size_t chunk_size =
        std::max<std::size_t>(32, std::min<std::size_t>(256, num_threads * 32));

    Multi_MOL chunk_pro(force_f), chunk_lig(force_f), chunk_wat(force_f),
        chunk_membrane(force_f);
    MOL last_pro_frame, last_lig_frame;
    bool have_last_frame = false;
    bool have_last_lig_frame = snapshot_args.flag_nocal;
    std::size_t selected_frames = 0;

    while (true) {
      auto chunk_status = snapshot_streamer.read_next_chunk(
          chunk_size, chunk_pro, chunk_lig, chunk_wat, chunk_membrane);
      if (!chunk_status)
        return std::unexpected(chunk_status.error());
      if (chunk_status.value() == 0)
        break;

      selected_frames += chunk_status.value();
      if (!chunk_pro.m_mol.empty()) {
        last_pro_frame = std::move(chunk_pro.m_mol.back());
        have_last_frame = true;
      }
      if (!snapshot_args.flag_nocal && !chunk_lig.m_mol.empty()) {
        last_lig_frame = std::move(chunk_lig.m_mol.back());
        have_last_lig_frame = true;
      }
      chunk_pro.m_mol.clear();
      chunk_lig.m_mol.clear();
      chunk_wat.m_mol.clear();
      chunk_membrane.m_mol.clear();
    }

    if (!have_last_frame) {
      return std::unexpected(
          "Pull-direction snapshot selector: no frame in the first trajectory "
          "satisfied the requested -e window.");
    }
    if (!have_last_lig_frame) {
      return std::unexpected(
          "Pull-direction snapshot selector: ligand coordinates were not found "
          "in the selected trajectory frame.");
    }

    pro_in.m_mol.clear();
    pro_in.m_mol.push_back(std::move(last_pro_frame));
    if (!snapshot_args.flag_nocal) {
      lig.m_mol.clear();
      lig.m_mol.push_back(std::move(last_lig_frame));
    }
    water_traj.m_mol.clear();

    selection.input_mode = "xdr-first-trajectory";
    selection.source_file = snapshot_args.traj_file.empty() ? snapshot_args.pdb_name
                                                            : snapshot_args.traj_file;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames > 0 ? selected_frames - 1 : 0;
  } else if (snapshot_args.flag_1_pdb) {
    Multi_MOL snapshot_pro(force_f), snapshot_lig(force_f), snapshot_wat(force_f);
    take_coordinate(snapshot_pro, snapshot_lig, snapshot_wat, Groups_from_index_file,
                    snapshot_args.pdb_name, force_f, snapshot_args, amine,
                    output_info, 1, 0);
    if (snapshot_pro.m_mol.empty()) {
      return std::unexpected(
          "Pull-direction snapshot selector: no snapshot was found in the first "
          "PDB input.");
    }

    const std::size_t selected_frames = snapshot_pro.m_mol.size();
    keep_only_selected_frame(snapshot_pro, selected_frames - 1);
    if (!snapshot_args.flag_nocal && !snapshot_lig.m_mol.empty())
      keep_only_selected_frame(snapshot_lig, snapshot_lig.m_mol.size() - 1);
    pro_in.m_mol = std::move(snapshot_pro.m_mol);
    if (!snapshot_args.flag_nocal)
      lig.m_mol = std::move(snapshot_lig.m_mol);
    water_traj.m_mol.clear();

    selection.input_mode = "single-pdb-first-input";
    selection.source_file = snapshot_args.pdb_name;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames - 1;
  } else {
    Multi_MOL snapshot_pro(force_f), snapshot_lig(force_f);
    if (snapshot_pro.take_coordinate(name_pro, force_f, snapshot_args, amine,
                                     output_info, 1, 0) != 0) {
      return std::unexpected("Pull-direction snapshot selector: failed to read " +
                             name_pro);
    }
    if (!snapshot_args.flag_nocal &&
        snapshot_lig.take_coordinate(name_lig, force_f, snapshot_args, amine,
                                     output_info, 1, 0) != 0) {
      return std::unexpected("Pull-direction snapshot selector: failed to read " +
                             name_lig);
    }

    const std::size_t selected_frames =
        snapshot_args.flag_nocal
            ? snapshot_pro.m_mol.size()
            : std::min(snapshot_pro.m_mol.size(), snapshot_lig.m_mol.size());
    if (selected_frames == 0) {
      return std::unexpected(
          "Pull-direction snapshot selector: no aligned receptor/ligand frame "
          "was found in the first PDB pair.");
    }

    keep_only_selected_frame(snapshot_pro, selected_frames - 1);
    if (!snapshot_args.flag_nocal)
      keep_only_selected_frame(snapshot_lig, selected_frames - 1);
    pro_in.m_mol = std::move(snapshot_pro.m_mol);
    if (!snapshot_args.flag_nocal)
      lig.m_mol = std::move(snapshot_lig.m_mol);
    water_traj.m_mol.clear();

    selection.input_mode = "paired-pdb-first-input";
    selection.source_file = name_pro + " + " + name_lig;
    selection.time_ps = pro_in.m_mol[0].time_of_snap;
    selection.frame_index = selected_frames - 1;
  }

  TitanUI::log_section(log, "PULL-DIRECT SNAPSHOT SELECTION");
  TitanUI::log_kv(log, "Pull Snapshot Mode", selection.input_mode);
  TitanUI::log_kv(log, "Pull Snapshot Source", selection.source_file);
  TitanUI::log_kv(log, "Pull Snapshot Rule", selection.selection_rule);
  TitanUI::log_kv(log, "Pull Snapshot Time", format_time_ps(selection.time_ps));
  TitanUI::log_kv(log, "Pull Snapshot Frame",
                  std::to_string(selection.frame_index));
  TitanUI::log_message(
      log, "Pull-direct stops after the first input; remaining files are not read.");

  set_pull_snapshot_metadata(selection, DE_arguments);
  return selection;
}

} // namespace engine_detail

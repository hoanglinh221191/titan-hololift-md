#include "titan_sq.h"

#include "info.h"
#include "titan_sq_core.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::string box_type_string(const box &frame_box) {
  box box_copy = frame_box;
  return box_copy.get_box_type();
}

std::string join_labels(const std::vector<std::string> &labels) {
  std::string joined;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (i > 0)
      joined += ",";
    joined += labels[i];
  }
  return joined;
}

} // namespace

void TitanSqAnalyzer::configure(const SqArgs &args) {
  requested_ = args.flag_sq;
  rdf_bin_a_ = args.sq_rdf_bin;
  rdf_max_input_a_ = args.sq_rdf_max;
  rdf_max_auto_ = args.sq_rdf_max_auto;
  q_bin_inv_a_ = args.sq_q_bin;
  q_max_input_inv_a_ = args.sq_q_max;
  q_max_auto_ = args.sq_q_max_auto;
  enabled_ = requested_ && initialized_ && atom_count_ > 0;
}

void TitanSqAnalyzer::reset_trajectory_state() {
  layout_ready_ = false;
  warned_invalid_box_ = false;
  warned_empty_channels_ = false;
  fixed_rdf_max_a_ = 0.0;
  fixed_q_max_inv_a_ = 0.0;
  rdf_bin_count_ = 0;
  q_bin_count_ = 0;
  total_frames_seen_ = 0;
  valid_frames_ = 0;
  skipped_invalid_box_frames_ = 0;
  skipped_invalid_density_frames_ = 0;
  first_valid_volume_a3_ = 0.0;
  first_valid_box_type_.clear();
  for (auto &pair : pair_profiles_) {
    pair.rdf_hist.clear();
    pair.density_weight_sum_a3 = 0.0;
  }
}

void TitanSqAnalyzer::prepare_channels(const Multi_MOL &mol1,
                                       const arguments &args,
                                       std::fstream &log) {
  channels_.clear();
  pair_profiles_.clear();
  active_atom_mask_.assign(atom_count_, 0);

  std::vector<std::string> channel_labels;

  for (const std::string &channel_name : args.sq.sq_channels) {
    ChannelSelection channel{};
    channel.name = channel_name;
    channel.membership.assign(atom_count_, 0);
    titan_selection::build_channel_atoms(mol1, args.mol_1_type, args.force_f,
                                         channel_name, args.llps.llps_sticker_beads,
                                         channel.atoms);

    if (channel.atoms.empty()) {
      channel.status = "skipped_empty_selection";
      if (!warned_empty_channels_) {
        TitanUI::print_warning(
            "SQ analysis: one or more requested channels were empty and will "
            "be skipped.");
        warned_empty_channels_ = true;
      }
    } else {
      channel.active = true;
      channel.status = "active";
      channel.atom_indices.reserve(channel.atoms.size());
      for (const auto &atom : channel.atoms) {
        if (atom.atom_index < 0)
          continue;
        const std::size_t atom_idx = static_cast<std::size_t>(atom.atom_index);
        if (atom_idx >= atom_count_ || channel.membership[atom_idx] != 0)
          continue;
        channel.membership[atom_idx] = 1;
        channel.atom_indices.push_back(atom.atom_index);
        active_atom_mask_[atom_idx] = 1;
      }
      if (channel.atom_indices.empty()) {
        channel.active = false;
        channel.status = "skipped_empty_selection";
      } else {
        channel_labels.push_back(channel.name);
      }
    }

    channels_.push_back(std::move(channel));
  }

  for (std::size_t i = 0; i < channels_.size(); ++i) {
    if (!channels_[i].active)
      continue;
    for (std::size_t j = i; j < channels_.size(); ++j) {
      if (!channels_[j].active)
        continue;
      std::size_t overlap = 0;
      for (std::size_t atom_idx = 0; atom_idx < atom_count_; ++atom_idx) {
        if (channels_[i].membership[atom_idx] != 0 &&
            channels_[j].membership[atom_idx] != 0) {
          ++overlap;
        }
      }
      const std::size_t ordered_pair_count =
          channels_[i].atom_indices.size() * channels_[j].atom_indices.size() -
          overlap;
      if (ordered_pair_count == 0)
        continue;

      PairAccumulator pair{};
      pair.lhs_channel = i;
      pair.rhs_channel = j;
      pair.label = (channels_[i].name == channels_[j].name)
                       ? channels_[i].name
                       : (channels_[i].name + "__" + channels_[j].name);
      pair.ordered_pair_count = ordered_pair_count;
      pair.overlap_count = overlap;
      pair_profiles_.push_back(std::move(pair));
    }
  }

  if (log.is_open()) {
    TitanUI::log_kv(log, "SQ Requested Channels", join_labels(args.sq.sq_channels));
    TitanUI::log_kv(log, "SQ Active Channels",
                    channel_labels.empty() ? "none" : join_labels(channel_labels));
    for (const auto &channel : channels_) {
      TitanUI::log_kv(log, "SQ Channel " + channel.name,
                      std::to_string(channel.atom_indices.size()) + " (" +
                          channel.status + ")");
    }
  }
}

void TitanSqAnalyzer::init(const Multi_MOL &mol1, const arguments &args,
                           std::fstream &log) {
  mol_type_ = args.mol_1_type;
  force_field_ = args.force_f;
  sticker_names_ = args.llps.llps_sticker_beads;

  if (initialized_) {
    enabled_ = requested_ && atom_count_ > 0;
    reset_trajectory_state();
    return;
  }

  const std::size_t info_atoms = mol1.info.size();
  const std::size_t frame_atoms =
      mol1.m_mol.empty() ? 0 : mol1.m_mol.front().size();
  atom_count_ = (info_atoms > 0 && frame_atoms > 0)
                    ? std::min(info_atoms, frame_atoms)
                    : std::max(info_atoms, frame_atoms);

  initialized_ = true;
  enabled_ = requested_ && atom_count_ > 0;
  if (atom_count_ == 0) {
    TitanUI::print_warning(
        "SQ analysis: no atoms were found in molecule 1. Module is disabled.");
    if (log.is_open())
      TitanUI::log_kv(log, "SQ Atoms", "0");
    return;
  }

  prepare_channels(mol1, args, log);
  reset_trajectory_state();

  if (log.is_open()) {
    TitanUI::log_kv(log, "SQ Atoms", std::to_string(atom_count_));
    TitanUI::log_kv(log, "SQ RDF Bin", std::to_string(rdf_bin_a_) + " A");
    TitanUI::log_kv(log, "SQ RDF Max",
                    rdf_max_auto_ ? "auto"
                                  : (std::to_string(rdf_max_input_a_) + " A"));
    TitanUI::log_kv(log, "SQ q Bin",
                    std::to_string(q_bin_inv_a_) + " A^-1");
    TitanUI::log_kv(log, "SQ q Max",
                    q_max_auto_ ? "auto"
                                : (std::to_string(q_max_input_inv_a_) +
                                   " A^-1"));
  }
}

void TitanSqAnalyzer::ensure_layout_from_frame(const MOL &frame) {
  if (layout_ready_)
    return;

  const double min_extent =
      std::min({frame.Box.get_a(), frame.Box.get_b(), frame.Box.get_c()});
  fixed_rdf_max_a_ =
      rdf_max_auto_ ? (0.5 * min_extent) : rdf_max_input_a_;
  fixed_rdf_max_a_ = std::max(fixed_rdf_max_a_, rdf_bin_a_);
  rdf_bin_count_ = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(fixed_rdf_max_a_ / rdf_bin_a_)));

  if (q_max_auto_) {
    const double q_from_box =
        min_extent > 0.0 ? (2.0 * kPi / min_extent) : q_bin_inv_a_;
    const double q_nyquist = kPi / rdf_bin_a_;
    fixed_q_max_inv_a_ =
        std::max({q_bin_inv_a_, q_from_box, std::min(25.0, q_nyquist)});
  } else {
    fixed_q_max_inv_a_ = q_max_input_inv_a_;
  }
  fixed_q_max_inv_a_ = std::max(fixed_q_max_inv_a_, q_bin_inv_a_);
  q_bin_count_ = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(fixed_q_max_inv_a_ / q_bin_inv_a_)));

  for (auto &pair : pair_profiles_) {
    pair.rdf_hist.assign(rdf_bin_count_, 0.0);
  }

  first_valid_volume_a3_ = titan_sq_detail::frame_box_volume_a3(frame.Box);
  first_valid_box_type_ = box_type_string(frame.Box);
  layout_ready_ = true;
}

void TitanSqAnalyzer::accumulate_distance(std::size_t lhs, std::size_t rhs,
                                          double distance_a) {
  if (!layout_ready_ || !(distance_a >= 0.0) || distance_a >= fixed_rdf_max_a_)
    return;
  const std::size_t bin =
      std::min<std::size_t>(rdf_bin_count_ - 1,
                            static_cast<std::size_t>(distance_a / rdf_bin_a_));
  for (auto &pair : pair_profiles_) {
    const auto &channel_a = channels_[pair.lhs_channel];
    const auto &channel_b = channels_[pair.rhs_channel];
    double ordered_weight = 0.0;
    if (channel_a.membership[lhs] != 0 && channel_b.membership[rhs] != 0)
      ordered_weight += 1.0;
    if (channel_a.membership[rhs] != 0 && channel_b.membership[lhs] != 0)
      ordered_weight += 1.0;
    if (ordered_weight > 0.0)
      pair.rdf_hist[bin] += ordered_weight;
  }
}

void TitanSqAnalyzer::process_frame(const Multi_MOL &mol1, std::size_t snap_idx,
                                    TitanExecutor & /*executor*/) {
  if (!enabled_ || snap_idx >= mol1.m_mol.size())
    return;

  ++total_frames_seen_;
  const MOL &frame = mol1.m_mol[snap_idx];
  if (!titan_sq_detail::has_positive_box_lengths(frame.Box)) {
    ++skipped_invalid_box_frames_;
    if (!warned_invalid_box_) {
      TitanUI::print_warning(
          "SQ analysis: valid periodic box is required. Non-periodic / invalid "
          "frames will be skipped.");
      warned_invalid_box_ = true;
    }
    return;
  }

  const double volume_a3 = titan_sq_detail::frame_box_volume_a3(frame.Box);
  if (!(volume_a3 > 0.0)) {
    ++skipped_invalid_density_frames_;
    if (!warned_invalid_box_) {
      TitanUI::print_warning(
          "SQ analysis: could not derive a positive periodic box volume. "
          "Affected frames will be skipped.");
      warned_invalid_box_ = true;
    }
    return;
  }

  ensure_layout_from_frame(frame);
  ++valid_frames_;

  if (pair_profiles_.empty() || active_atom_mask_.size() < 2)
    return;

  for (auto &pair : pair_profiles_) {
    pair.density_weight_sum_a3 +=
        static_cast<double>(pair.ordered_pair_count) / volume_a3;
  }

  const auto xs = frame.get_pos_x();
  const auto ys = frame.get_pos_y();
  const auto zs = frame.get_pos_z();
  titan_sq_detail::for_each_active_pair_within_cutoff(
      xs, ys, zs, frame.Box, active_atom_mask_, fixed_rdf_max_a_, neighbor_grid_,
      [&](std::size_t lhs, std::size_t rhs, double distance_a) {
        accumulate_distance(lhs, rhs, distance_a);
      });
}

std::vector<double> TitanSqAnalyzer::build_g_profile(
    const PairAccumulator &pair) const {
  std::vector<double> g_profile(rdf_bin_count_, 0.0);
  if (!layout_ready_ || pair.density_weight_sum_a3 <= 0.0)
    return g_profile;

  for (std::size_t bin = 0; bin < rdf_bin_count_; ++bin) {
    const double r_inner = static_cast<double>(bin) * rdf_bin_a_;
    const double r_outer = r_inner + rdf_bin_a_;
    const double denominator =
        titan_sq_detail::shell_volume_a3(r_inner, r_outer) *
        pair.density_weight_sum_a3;
    if (denominator > 0.0)
      g_profile[bin] = pair.rdf_hist[bin] / denominator;
  }
  return g_profile;
}

void TitanSqAnalyzer::export_summary(const std::string &filename,
                                     std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("SQ analysis: failed to open " + filename);
    return;
  }

  out << "module: titan_sq\n";
  out << "frames_seen: " << total_frames_seen_ << "\n";
  out << "valid_periodic_frames: " << valid_frames_ << "\n";
  out << "skipped_invalid_box_frames: " << skipped_invalid_box_frames_ << "\n";
  out << "skipped_invalid_density_frames: " << skipped_invalid_density_frames_
      << "\n";
  out << "atom_count_mol1: " << atom_count_ << "\n";
  out << "rdf_bin_A: " << std::fixed << std::setprecision(6) << rdf_bin_a_
      << "\n";
  out << "rdf_max_A: "
      << (layout_ready_ ? fixed_rdf_max_a_
                        : (rdf_max_auto_ ? 0.0 : rdf_max_input_a_))
      << "\n";
  out << "q_bin_A^-1: " << q_bin_inv_a_ << "\n";
  out << "q_max_A^-1: "
      << (layout_ready_ ? fixed_q_max_inv_a_
                        : (q_max_auto_ ? 0.0 : q_max_input_inv_a_))
      << "\n";
  out << "rdf_method: streaming_pair_correlation\n";
  out << "sq_method: sq_from_rdf_lorch\n";
  if (!first_valid_box_type_.empty())
    out << "first_valid_box_type: " << first_valid_box_type_ << "\n";
  if (first_valid_volume_a3_ > 0.0)
    out << "first_valid_volume_A3: " << first_valid_volume_a3_ << "\n";

  for (const auto &channel : channels_) {
    out << "channel." << channel.name << ".status: " << channel.status << "\n";
    out << "channel." << channel.name
        << ".atoms: " << channel.atom_indices.size() << "\n";
  }
  for (const auto &pair : pair_profiles_) {
    out << "pair." << pair.label
        << ".ordered_pairs_per_frame: " << pair.ordered_pair_count << "\n";
    out << "pair." << pair.label
        << ".overlap_atoms: " << pair.overlap_count << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "SQ Summary", filename);
}

void TitanSqAnalyzer::export_rdf_profile(const PairAccumulator &pair,
                                         const std::string &filename,
                                         std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("SQ analysis: failed to open " + filename);
    return;
  }

  const auto &channel_a = channels_[pair.lhs_channel];
  const auto &channel_b = channels_[pair.rhs_channel];
  const std::vector<double> g_profile = build_g_profile(pair);
  const double mean_density_weight =
      valid_frames_ > 0
          ? pair.density_weight_sum_a3 / static_cast<double>(valid_frames_)
          : 0.0;

  out << "# TITAN SQ radial distribution function\n";
  out << "# distances in Angstrom; g_ab(r) dimensionless\n";
  out << "# channel_a " << channel_a.name << "\n";
  out << "# channel_b " << channel_b.name << "\n";
  out << "# frames_used " << valid_frames_ << "\n";
  out << "# frames_skipped_invalid_box "
      << (skipped_invalid_box_frames_ + skipped_invalid_density_frames_) << "\n";
  out << "# atoms_a " << channel_a.atom_indices.size() << "\n";
  out << "# atoms_b " << channel_b.atom_indices.size() << "\n";
  out << "# overlap_atoms " << pair.overlap_count << "\n";
  out << "# ordered_pairs_per_frame " << pair.ordered_pair_count << "\n";
  out << "# density_weight_mean_A^-3 " << std::fixed << std::setprecision(8)
      << mean_density_weight << "\n";
  out << "# rdf_bin_A " << rdf_bin_a_ << "\n";
  out << "# rdf_max_A " << fixed_rdf_max_a_ << "\n";
  out << "# method streaming_pair_correlation\n";
  out << "# columns = bin r_inner_A r_outer_A r_center_A g_ab "
         "ordered_pair_hits density_weight_sum_A^-3\n";

  out << std::fixed << std::setprecision(8);
  for (std::size_t bin = 0; bin < rdf_bin_count_; ++bin) {
    const double r_inner = static_cast<double>(bin) * rdf_bin_a_;
    const double r_outer = r_inner + rdf_bin_a_;
    const double r_center = 0.5 * (r_inner + r_outer);
    out << std::setw(6) << (bin + 1) << " " << std::setw(14) << r_inner << " "
        << std::setw(14) << r_outer << " " << std::setw(14) << r_center << " "
        << std::setw(14) << g_profile[bin] << " " << std::setw(18)
        << pair.rdf_hist[bin] << " " << std::setw(18)
        << pair.density_weight_sum_a3 << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "SQ RDF " + pair.label, filename);
}

void TitanSqAnalyzer::export_structure_factor(const PairAccumulator &pair,
                                              const std::string &filename,
                                              std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("SQ analysis: failed to open " + filename);
    return;
  }

  const auto &channel_a = channels_[pair.lhs_channel];
  const auto &channel_b = channels_[pair.rhs_channel];
  const std::vector<double> g_profile = build_g_profile(pair);
  const double atoms_a = static_cast<double>(channel_a.atom_indices.size());
  const double rho_partner_a3 =
      (valid_frames_ > 0 && atoms_a > 0.0)
          ? pair.density_weight_sum_a3 /
                (static_cast<double>(valid_frames_) * atoms_a)
          : 0.0;
  const std::vector<double> sq_profile =
      titan_sq_detail::build_structure_factor_from_gr(
          g_profile, rdf_bin_a_, fixed_rdf_max_a_, q_bin_inv_a_, q_bin_count_,
          rho_partner_a3, pair.lhs_channel == pair.rhs_channel);

  out << "# TITAN SQ structure factor from RDF\n";
  out << "# q in A^-1; S_ab(q) dimensionless\n";
  out << "# channel_a " << channel_a.name << "\n";
  out << "# channel_b " << channel_b.name << "\n";
  out << "# frames_used " << valid_frames_ << "\n";
  out << "# atoms_a " << channel_a.atom_indices.size() << "\n";
  out << "# atoms_b " << channel_b.atom_indices.size() << "\n";
  out << "# overlap_atoms " << pair.overlap_count << "\n";
  out << "# rho_partner_mean_A^-3 " << std::fixed << std::setprecision(8)
      << rho_partner_a3 << "\n";
  out << "# rdf_max_A " << fixed_rdf_max_a_ << "\n";
  out << "# q_bin_A^-1 " << q_bin_inv_a_ << "\n";
  out << "# q_max_A^-1 " << fixed_q_max_inv_a_ << "\n";
  out << "# method sq_from_rdf_lorch\n";
  out << "# columns = bin q_A^-1 S_ab\n";

  out << std::fixed << std::setprecision(8);
  for (std::size_t q_bin = 0; q_bin < q_bin_count_; ++q_bin) {
    const double q = static_cast<double>(q_bin + 1) * q_bin_inv_a_;
    out << std::setw(6) << (q_bin + 1) << " " << std::setw(14) << q << " "
        << std::setw(14) << sq_profile[q_bin] << "\n";
  }

  if (log.is_open())
    TitanUI::log_kv(log, "SQ Structure Factor " + pair.label, filename);
}

void TitanSqAnalyzer::finalize_trajectory(const std::string &base_dir,
                                          const info &info_output,
                                          const arguments &args,
                                          const std::vector<std::string> &input_line,
                                          int input_name, std::fstream &log) {
  if (!requested_ || !initialized_)
    return;

  const std::string summary_filename =
      base_dir + "/" +
      info_output.get_sq_summary_filename(args, input_line, input_name);
  export_summary(summary_filename, log);

  if (!layout_ready_ || valid_frames_ == 0 || pair_profiles_.empty()) {
    if (requested_) {
      TitanUI::print_warning(
          "SQ analysis: no valid periodic condensed-phase frames were available; "
          "only the summary file was written.");
    }
    reset_trajectory_state();
    return;
  }

  for (const auto &pair : pair_profiles_) {
    const std::string rdf_filename =
        base_dir + "/" +
        info_output.get_sq_rdf_filename(args, input_line, input_name,
                                        channels_[pair.lhs_channel].name,
                                        channels_[pair.rhs_channel].name);
    export_rdf_profile(pair, rdf_filename, log);

    if (pair.density_weight_sum_a3 > 0.0) {
      const std::string sq_filename =
          base_dir + "/" +
          info_output.get_sq_structure_factor_filename(
              args, input_line, input_name, channels_[pair.lhs_channel].name,
              channels_[pair.rhs_channel].name);
      export_structure_factor(pair, sq_filename, log);
    }
  }

  TitanUI::print_info("[SQ] Exported summary to " + summary_filename);
  reset_trajectory_state();
}

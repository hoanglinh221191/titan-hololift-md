#include "titan_contact_map.h"

#include "BoxMatrix.h"
#include "info.h"
#include "titan_contact_kernels.h"
#include "titan_executor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string normalize_choice(const std::string &value, const char *name,
                             const std::vector<std::string> &allowed) {
  const std::string normalized =
      lowercase_copy(titan_contact::trim_copy(value));
  if (std::find(allowed.begin(), allowed.end(), normalized) != allowed.end())
    return normalized;
  std::ostringstream message;
  message << "CMap: " << name << " has an invalid value.";
  throw std::invalid_argument(message.str());
}

std::string join_output_path(const std::string &base_dir,
                             const std::string &filename) {
  if (base_dir.empty() || base_dir == ".")
    return filename;
  return (std::filesystem::path(base_dir) / filename).string();
}

std::string molecule_role(std::uint8_t molecule) {
  return molecule == 2 ? "mol2" : "mol1";
}

bool residue_less(const titan_contact::ResidueKey &lhs,
                  const titan_contact::ResidueKey &rhs) {
  if (lhs.molecule != rhs.molecule)
    return lhs.molecule < rhs.molecule;
  if (lhs.chainnum != rhs.chainnum)
    return lhs.chainnum < rhs.chainnum;
  if (lhs.resnr != rhs.resnr)
    return lhs.resnr < rhs.resnr;
  return lhs.ordinal < rhs.ordinal;
}

void log_cmap_value(std::fstream &log, const std::string &key,
                    const std::string &value) {
  if (log.is_open())
    log << key << ": " << value << '\n';
}

} // namespace

void TitanContactMapAnalyzer::configure(const CMapArgs &args) {
  requested_ = args.flag_cmap;
  mode_ = titan_contact::parse_contact_mode(args.mode);
  scope_ = normalize_choice(args.scope, "-cmap-scope", {"mol1", "interface"});
  domain_ = normalize_choice(args.domain, "-cmap-domain",
                             {"intra", "inter", "both"});
  cutoff_a_ = args.cutoff_ang;
  if (!std::isfinite(cutoff_a_) || cutoff_a_ <= 0.0)
    throw std::invalid_argument("CMap: -cmap-cut must be finite and > 0.");
  min_seqsep_ = args.min_seqsep;
  enabled_ = requested_ && initialized_ && !row_residues_.empty() &&
             !col_residues_.empty();
}

void TitanContactMapAnalyzer::reset_trajectory_state() {
  warned_invalid_box_ = false;
  total_frames_seen_ = 0;
  valid_frames_ = 0;
  skipped_invalid_box_frames_ = 0;
  skipped_incomplete_frames_ = 0;
  row_occupancy_frames_.assign(row_residues_.size(), 0);
  col_occupancy_frames_.assign(col_residues_.size(), 0);
  contact_count_timeseries_.clear();
  records_.clear();
}

void TitanContactMapAnalyzer::accumulate_pooled_state() {
  pooled_total_frames_seen_ += total_frames_seen_;
  pooled_valid_frames_ += valid_frames_;
  pooled_skipped_invalid_box_frames_ += skipped_invalid_box_frames_;
  pooled_skipped_incomplete_frames_ += skipped_incomplete_frames_;
  if (pooled_row_occupancy_frames_.size() != row_occupancy_frames_.size())
    pooled_row_occupancy_frames_.assign(row_occupancy_frames_.size(), 0);
  if (pooled_col_occupancy_frames_.size() != col_occupancy_frames_.size())
    pooled_col_occupancy_frames_.assign(col_occupancy_frames_.size(), 0);
  for (std::size_t i = 0; i < row_occupancy_frames_.size(); ++i)
    pooled_row_occupancy_frames_[i] += row_occupancy_frames_[i];
  for (std::size_t i = 0; i < col_occupancy_frames_.size(); ++i)
    pooled_col_occupancy_frames_[i] += col_occupancy_frames_[i];
  for (const auto &[key, source] : records_) {
    auto [found, inserted] = pooled_records_.try_emplace(key);
    ContactRecord &target = found->second;
    if (inserted) {
      target = source;
      continue;
    }
    target.contact_frames += source.contact_frames;
    target.distance_sum_a += source.distance_sum_a;
    target.distance_samples += source.distance_samples;
    target.min_distance_a = std::min(target.min_distance_a, source.min_distance_a);
  }
}

void TitanContactMapAnalyzer::init(const Multi_MOL &mol1, const Multi_MOL *mol2,
                                   const arguments &, std::fstream &log) {
  if (!row_geometry_.build(mol1, 1, mode_))
    throw std::invalid_argument(
        "CMap: failed to build molecule 1 contact geometry.");
  row_residues_ = row_geometry_.residues();
  atom_count_ = mol1.info.size();
  if (scope_ == "interface") {
    if (mol2 == nullptr || mol2->info.empty())
      throw std::invalid_argument(
          "CMap: -cmap-scope=interface requires a valid molecule 2/group 2.");
    if (!col_geometry_.build(*mol2, 2, mode_))
      throw std::invalid_argument(
          "CMap: failed to build molecule 2 contact geometry.");
    col_residues_ = col_geometry_.residues();
  } else {
    col_geometry_ = row_geometry_;
    col_residues_ = row_residues_;
  }

  initialized_ = true;
  enabled_ = requested_ && !row_residues_.empty() && !col_residues_.empty();
  reset_trajectory_state();

  log_cmap_value(log, "CMap Mode", titan_contact::contact_mode_name(mode_));
  log_cmap_value(log, "CMap Scope", scope_);
  log_cmap_value(log, "CMap Cutoff", std::to_string(cutoff_a_) + " A");
  log_cmap_value(log, "CMap Domain", domain_);
  log_cmap_value(log, "CMap Rows", std::to_string(row_residues_.size()));
  log_cmap_value(log, "CMap Columns", std::to_string(col_residues_.size()));
}

bool TitanContactMapAnalyzer::has_valid_box(const box &frame_box) const
    noexcept {
  return frame_box.get_a() > 0.0 && frame_box.get_b() > 0.0 &&
         frame_box.get_c() > 0.0;
}

bool TitanContactMapAnalyzer::pair_allowed(
    const titan_contact::ResidueKey &lhs,
    const titan_contact::ResidueKey &rhs) const {
  if (scope_ == "interface")
    return true;
  if (titan_contact::same_residue(lhs, rhs))
    return false;
  const bool same_chain = titan_contact::same_chain(lhs, rhs);
  if (domain_ == "intra" && !same_chain)
    return false;
  if (domain_ == "inter" && same_chain)
    return false;
  if (same_chain && min_seqsep_ > 0) {
    const std::size_t diff = lhs.ordinal > rhs.ordinal
                                 ? lhs.ordinal - rhs.ordinal
                                 : rhs.ordinal - lhs.ordinal;
    if (diff < static_cast<std::size_t>(min_seqsep_))
      return false;
  }
  return true;
}

std::size_t TitanContactMapAnalyzer::feature_pair_count() const {
  std::size_t count = 0;
  for (std::size_t row = 0; row < row_residues_.size(); ++row) {
    for (std::size_t col = 0; col < col_residues_.size(); ++col) {
      if (scope_ == "mol1" && col <= row)
        continue;
      if (pair_allowed(row_residues_[row], col_residues_[col]))
        ++count;
    }
  }
  return count;
}

double TitanContactMapAnalyzer::latest_contact_count_normalized() const
    noexcept {
  if (contact_count_timeseries_.empty())
    return 0.0;
  const std::size_t possible = feature_pair_count();
  if (possible == 0)
    return 0.0;
  return static_cast<double>(contact_count_timeseries_.back().second) /
         static_cast<double>(possible);
}

std::size_t TitanContactMapAnalyzer::residue_index(
    const std::vector<titan_contact::ResidueKey> &residues,
    const titan_contact::ResidueKey &residue) const {
  const auto found = std::find(residues.begin(), residues.end(), residue);
  return found == residues.end()
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(std::distance(residues.begin(), found));
}

void TitanContactMapAnalyzer::process_frame(const Multi_MOL &mol1,
                                            const Multi_MOL *mol2,
                                            std::size_t snap_idx,
                                            TitanExecutor &executor) {
  if (!enabled_ || snap_idx >= mol1.m_mol.size())
    return;
  ++total_frames_seen_;
  const MOL &frame = mol1.m_mol[snap_idx];
  if (!has_valid_box(frame.Box)) {
    ++skipped_invalid_box_frames_;
    warned_invalid_box_ = true;
    return;
  }
  if (scope_ == "interface" &&
      (mol2 == nullptr || snap_idx >= mol2->m_mol.size())) {
    ++skipped_incomplete_frames_;
    return;
  }

  auto &row_points = row_points_scratch_;
  auto &col_points = col_points_scratch_;
  if (!row_geometry_.build_points(mol1, snap_idx, row_points) ||
      !(scope_ == "interface"
            ? col_geometry_.build_points(*mol2, snap_idx, col_points)
            : (col_points = row_points, true))) {
    ++skipped_incomplete_frames_;
    return;
  }

  ++valid_frames_;
  const BoxMatrix box_matrix(frame.Box);
  const double cutoff_sq = cutoff_a_ * cutoff_a_;
  std::unordered_map<titan_contact::ResiduePairKey, double,
                     titan_contact::ResiduePairKeyHash>
      frame_nearest;

  auto &row_soa = row_soa_scratch_;
  auto &col_soa = col_soa_scratch_;
  row_soa.x.clear();
  row_soa.y.clear();
  row_soa.z.clear();
  row_soa.x.reserve(row_points.size());
  row_soa.y.reserve(row_points.size());
  row_soa.z.reserve(row_points.size());
  for (const titan_contact_geometry::ContactPoint &point : row_points) {
    row_soa.x.push_back(point.x);
    row_soa.y.push_back(point.y);
    row_soa.z.push_back(point.z);
  }
  col_soa.x.clear();
  col_soa.y.clear();
  col_soa.z.clear();
  col_soa.x.reserve(col_points.size());
  col_soa.y.reserve(col_points.size());
  col_soa.z.reserve(col_points.size());
  for (const titan_contact_geometry::ContactPoint &point : col_points) {
    col_soa.x.push_back(point.x);
    col_soa.y.push_back(point.y);
    col_soa.z.push_back(point.z);
  }

  const std::size_t worker_count =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<std::unordered_map<titan_contact::ResiduePairKey, double,
                                 titan_contact::ResiduePairKeyHash>>
      worker_nearest(worker_count);

  auto consider_candidate =
      [&](std::size_t worker_id,
          const titan_contact_kernel::ContactCandidate &candidate) {
        const titan_contact_geometry::ContactPoint &lhs =
            row_points[candidate.row];
        const titan_contact_geometry::ContactPoint &rhs =
            col_points[candidate.col];
    if (!pair_allowed(lhs.residue, rhs.residue))
      return;
    const auto key = scope_ == "interface"
                         ? titan_contact::ResiduePairKey{lhs.residue,
                                                         rhs.residue}
                         : titan_contact::make_ordered_pair(lhs.residue,
                                                            rhs.residue);
        auto &nearest = worker_nearest[worker_id];
        auto found = nearest.find(key);
        if (found == nearest.end() || candidate.distance_a < found->second)
          nearest[key] = candidate.distance_a;
  };

  if (scope_ == "interface") {
    executor.parallel_do(worker_count, [&](std::size_t worker_id) {
      const std::size_t row_begin = row_points.size() * worker_id / worker_count;
      const std::size_t row_end =
          row_points.size() * (worker_id + 1) / worker_count;
      std::vector<titan_contact_kernel::ContactCandidate> candidates;
      titan_contact_kernel::scan_interface_contacts_optimized(
          row_soa, col_soa, box_matrix, cutoff_sq, row_begin, row_end,
          candidates);
      for (const auto &candidate : candidates)
        consider_candidate(worker_id, candidate);
    });
  } else {
    executor.parallel_do(worker_count, [&](std::size_t worker_id) {
      std::vector<titan_contact_kernel::ContactCandidate> candidates;
      for (std::size_t row = worker_id; row < row_points.size();
           row += worker_count) {
        titan_contact_kernel::scan_mol1_contacts_optimized(
            row_soa, box_matrix, cutoff_sq, row, row + 1, candidates);
      }
      for (const auto &candidate : candidates)
        consider_candidate(worker_id, candidate);
    });
  }

  for (const auto &nearest : worker_nearest) {
    for (const auto &[key, distance] : nearest) {
      auto found = frame_nearest.find(key);
      if (found == frame_nearest.end() || distance < found->second)
        frame_nearest[key] = distance;
    }
  }

  std::unordered_set<std::size_t> rows_contacted;
  std::unordered_set<std::size_t> cols_contacted;
  for (const auto &[key, distance] : frame_nearest) {
    auto [it, inserted] = records_.try_emplace(key);
    ContactRecord &record = it->second;
    if (inserted) {
      record.key = key;
      record.min_distance_a = distance;
    } else {
      record.min_distance_a = std::min(record.min_distance_a, distance);
    }
    ++record.contact_frames;
    record.distance_sum_a += distance;
    ++record.distance_samples;
    const std::size_t row_idx = residue_index(row_residues_, key.first);
    const std::size_t col_idx = residue_index(col_residues_, key.second);
    if (row_idx != std::numeric_limits<std::size_t>::max())
      rows_contacted.insert(row_idx);
    if (col_idx != std::numeric_limits<std::size_t>::max())
      cols_contacted.insert(col_idx);
    if (scope_ == "mol1") {
      const std::size_t second_row = residue_index(row_residues_, key.second);
      if (second_row != std::numeric_limits<std::size_t>::max())
        rows_contacted.insert(second_row);
    }
  }
  for (const std::size_t idx : rows_contacted)
    ++row_occupancy_frames_[idx];
  for (const std::size_t idx : cols_contacted)
    ++col_occupancy_frames_[idx];
  contact_count_timeseries_.emplace_back(frame.get_time(), frame_nearest.size());
}

std::string TitanContactMapAnalyzer::residue_label(
    const titan_contact::ResidueKey &residue) const {
  std::ostringstream out;
  out << molecule_role(residue.molecule) << '|' << residue.chain << ':'
      << residue.resnr << ':' << residue.resname;
  return out.str();
}

void TitanContactMapAnalyzer::export_summary(const std::string &filename,
                                             std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_cmap_value(log, "CMap Summary", "failed to open " + filename);
    return;
  }
  out << std::fixed << std::setprecision(6);
  out << "module: titan_cmap\n";
  out << "mode: " << titan_contact::contact_mode_name(mode_) << '\n';
  out << "scope: " << scope_ << '\n';
  out << "matrix_layout: " << (scope_ == "interface" ? "rectangular" : "square")
      << '\n';
  out << "cutoff_A: " << cutoff_a_ << '\n';
  if (scope_ == "mol1") {
    out << "domain: " << domain_ << '\n';
    out << "min_seqsep: " << min_seqsep_ << '\n';
  }
  out << "frames_seen: " << total_frames_seen_ << '\n';
  out << "valid_frames: " << valid_frames_ << '\n';
  out << "skipped_invalid_box_frames: " << skipped_invalid_box_frames_ << '\n';
  out << "skipped_incomplete_frames: " << skipped_incomplete_frames_ << '\n';
  out << "rows: " << row_residues_.size() << '\n';
  out << "columns: " << col_residues_.size() << '\n';
  out << "contact_edges: " << records_.size() << '\n';
  log_cmap_value(log, "CMap Summary", filename);
}

void TitanContactMapAnalyzer::export_edges(const std::string &filename,
                                           std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_cmap_value(log, "CMap Edges", "failed to open " + filename);
    return;
  }
  out << "molecule_i,molecule_j,chain_i,resnr_i,resname_i,chain_j,resnr_j,"
         "resname_j,probability,contact_frames,mean_distance_A,min_distance_A\n";
  std::vector<ContactRecord> records;
  records.reserve(records_.size());
  for (const auto &[_, record] : records_)
    records.push_back(record);
  std::sort(records.begin(), records.end(),
            [](const ContactRecord &lhs, const ContactRecord &rhs) {
              if (residue_less(lhs.key.first, rhs.key.first))
                return true;
              if (residue_less(rhs.key.first, lhs.key.first))
                return false;
              return residue_less(lhs.key.second, rhs.key.second);
            });
  out << std::fixed << std::setprecision(6);
  for (const ContactRecord &record : records) {
    const double probability =
        valid_frames_ > 0 ? static_cast<double>(record.contact_frames) /
                                static_cast<double>(valid_frames_)
                          : 0.0;
    const double mean_distance =
        record.distance_samples > 0
            ? record.distance_sum_a / static_cast<double>(record.distance_samples)
            : 0.0;
    out << molecule_role(record.key.first.molecule) << ','
        << molecule_role(record.key.second.molecule) << ','
        << record.key.first.chain << ',' << record.key.first.resnr << ','
        << record.key.first.resname << ','
        << record.key.second.chain << ',' << record.key.second.resnr << ','
        << record.key.second.resname << ',' << probability << ','
        << record.contact_frames << ',' << mean_distance << ','
        << record.min_distance_a << '\n';
  }
  log_cmap_value(log, "CMap Edges", filename);
}

void TitanContactMapAnalyzer::export_matrix(const std::string &filename,
                                            std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_cmap_value(log, "CMap Matrix", "failed to open " + filename);
    return;
  }
  out << "residue";
  for (const auto &residue : col_residues_)
    out << ',' << residue_label(residue);
  out << '\n';
  out << std::fixed << std::setprecision(6);
  for (const auto &row : row_residues_) {
    out << residue_label(row);
    for (const auto &col : col_residues_) {
      double probability = 0.0;
      if (!titan_contact::same_residue(row, col) || scope_ == "interface") {
        const auto key = scope_ == "interface"
                             ? titan_contact::ResiduePairKey{row, col}
                             : titan_contact::make_ordered_pair(row, col);
        const auto found = records_.find(key);
        if (found != records_.end() && valid_frames_ > 0)
          probability = static_cast<double>(found->second.contact_frames) /
                        static_cast<double>(valid_frames_);
      }
      out << ',' << probability;
    }
    out << '\n';
  }
  log_cmap_value(log, "CMap Matrix", filename);
}

void TitanContactMapAnalyzer::export_timeseries(const std::string &filename,
                                                std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_cmap_value(log, "CMap Timeseries", "failed to open " + filename);
    return;
  }
  out << "frame,time_ps,contact_pairs\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < contact_count_timeseries_.size(); ++i)
    out << i << ',' << contact_count_timeseries_[i].first << ','
        << contact_count_timeseries_[i].second << '\n';
  log_cmap_value(log, "CMap Timeseries", filename);
}

void TitanContactMapAnalyzer::export_occupancy(const std::string &filename,
                                               std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_cmap_value(log, "CMap Occupancy", "failed to open " + filename);
    return;
  }
  out << "molecule,residue,contact_frames,occupancy_probability\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < row_residues_.size(); ++i) {
    const double occupancy =
        valid_frames_ > 0 ? static_cast<double>(row_occupancy_frames_[i]) /
                                static_cast<double>(valid_frames_)
                          : 0.0;
    out << molecule_role(row_residues_[i].molecule) << ','
        << residue_label(row_residues_[i]) << ',' << row_occupancy_frames_[i]
        << ',' << occupancy << '\n';
  }
  if (scope_ == "interface") {
    for (std::size_t i = 0; i < col_residues_.size(); ++i) {
      const double occupancy =
          valid_frames_ > 0 ? static_cast<double>(col_occupancy_frames_[i]) /
                                  static_cast<double>(valid_frames_)
                            : 0.0;
      out << molecule_role(col_residues_[i].molecule) << ','
          << residue_label(col_residues_[i]) << ',' << col_occupancy_frames_[i]
          << ',' << occupancy << '\n';
    }
  }
  log_cmap_value(log, "CMap Occupancy", filename);
}

void TitanContactMapAnalyzer::finalize_trajectory(
    const std::string &base_dir, const info &info_output, const arguments &args,
    const std::vector<std::string> &input_line, int input_name,
    std::fstream &log) {
  if (!initialized_ || !requested_)
    return;
  accumulate_pooled_state();
  export_summary(join_output_path(base_dir, info_output.get_cmap_summary_filename(
                                                args, input_line, input_name)),
                 log);
  export_edges(join_output_path(base_dir, info_output.get_cmap_edges_filename(
                                              args, input_line, input_name)),
               log);
  export_matrix(join_output_path(base_dir, info_output.get_cmap_matrix_filename(
                                               args, input_line, input_name)),
                log);
  export_timeseries(
      join_output_path(base_dir, info_output.get_cmap_timeseries_filename(
                                     args, input_line, input_name)),
      log);
  export_occupancy(
      join_output_path(base_dir, info_output.get_cmap_occupancy_filename(
                                     args, input_line, input_name)),
      log);
  reset_trajectory_state();
}

void TitanContactMapAnalyzer::finalize_all(const std::string &base_dir,
                                           const info &info_output,
                                           std::fstream &log) {
  if (!requested_ || pooled_valid_frames_ == 0)
    return;
  const auto saved_valid_frames = valid_frames_;
  const auto saved_total_frames = total_frames_seen_;
  const auto saved_skipped_box = skipped_invalid_box_frames_;
  const auto saved_skipped_incomplete = skipped_incomplete_frames_;
  const auto saved_records = records_;
  const auto saved_row_occupancy = row_occupancy_frames_;
  const auto saved_col_occupancy = col_occupancy_frames_;
  total_frames_seen_ = pooled_total_frames_seen_;
  valid_frames_ = pooled_valid_frames_;
  skipped_invalid_box_frames_ = pooled_skipped_invalid_box_frames_;
  skipped_incomplete_frames_ = pooled_skipped_incomplete_frames_;
  records_ = pooled_records_;
  row_occupancy_frames_ = pooled_row_occupancy_frames_;
  col_occupancy_frames_ = pooled_col_occupancy_frames_;
  export_summary(join_output_path(base_dir,
                                  info_output.get_cmap_summary_average_filename()),
                 log);
  export_edges(join_output_path(base_dir,
                                info_output.get_cmap_edges_average_filename()),
               log);
  export_matrix(join_output_path(base_dir,
                                 info_output.get_cmap_matrix_average_filename()),
                log);
  export_occupancy(
      join_output_path(base_dir, info_output.get_cmap_occupancy_average_filename()),
      log);
  total_frames_seen_ = saved_total_frames;
  valid_frames_ = saved_valid_frames;
  skipped_invalid_box_frames_ = saved_skipped_box;
  skipped_incomplete_frames_ = saved_skipped_incomplete;
  records_ = saved_records;
  row_occupancy_frames_ = saved_row_occupancy;
  col_occupancy_frames_ = saved_col_occupancy;
}

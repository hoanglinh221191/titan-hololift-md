#include "titan_qnative.h"

#include "info.h"
#include "titan_executor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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
  message << "QNative: " << name << " has an invalid value.";
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

void log_qnative_value(std::fstream &log, const std::string &key,
                       const std::string &value) {
  if (log.is_open())
    log << key << ": " << value << '\n';
}

} // namespace

void TitanQNativeAnalyzer::configure(const QNativeArgs &args) {
  requested_ = args.flag_qnative;
  mode_ = titan_contact::parse_contact_mode(args.mode);
  scope_ = normalize_choice(args.scope, "-qnative-scope", {"mol1", "interface"});
  domain_ = normalize_choice(args.domain, "-qnative-domain",
                             {"intra", "inter", "both"});
  cutoff_a_ = args.cutoff_ang;
  if (!std::isfinite(cutoff_a_) || cutoff_a_ <= 0.0)
    throw std::invalid_argument("QNative: -qnative-cut must be finite and > 0.");
  min_seqsep_ = args.min_seqsep;
  soft_ = args.soft;
  beta_ = args.beta;
  if (!std::isfinite(beta_) || beta_ <= 0.0)
    throw std::invalid_argument("QNative: -qnative-beta must be finite and > 0.");
  reference_source_ =
      args.ref_explicit ? args.ref_file : std::string("loaded-reference");
  enabled_ = requested_ && initialized_;
}

void TitanQNativeAnalyzer::reset_trajectory_state() {
  warned_invalid_box_ = false;
  total_frames_seen_ = 0;
  valid_frames_ = 0;
  skipped_invalid_box_frames_ = 0;
  skipped_incomplete_frames_ = 0;
  occupancy_.clear();
  occupancy_.reserve(native_contacts_.size());
  for (const NativeContact &contact : native_contacts_)
    occupancy_.push_back(ContactOccupancy{contact.key, 0.0, 0});
  timeseries_.clear();
  q_values_.clear();
}

void TitanQNativeAnalyzer::accumulate_pooled_state() {
  pooled_total_frames_seen_ += total_frames_seen_;
  pooled_valid_frames_ += valid_frames_;
  pooled_skipped_invalid_box_frames_ += skipped_invalid_box_frames_;
  pooled_skipped_incomplete_frames_ += skipped_incomplete_frames_;
  if (pooled_occupancy_.size() != occupancy_.size())
    pooled_occupancy_.assign(occupancy_.size(), ContactOccupancy{});
  for (std::size_t i = 0; i < occupancy_.size(); ++i) {
    if (pooled_occupancy_[i].key.first.resnr == 0 &&
        pooled_occupancy_[i].key.second.resnr == 0) {
      pooled_occupancy_[i].key = occupancy_[i].key;
    }
    pooled_occupancy_[i].formed_sum += occupancy_[i].formed_sum;
    pooled_occupancy_[i].hard_frames += occupancy_[i].hard_frames;
  }
  pooled_timeseries_.insert(pooled_timeseries_.end(), timeseries_.begin(),
                            timeseries_.end());
  pooled_q_values_.insert(pooled_q_values_.end(), q_values_.begin(),
                          q_values_.end());
}

bool TitanQNativeAnalyzer::has_valid_box(const box &frame_box) const noexcept {
  return frame_box.get_a() > 0.0 && frame_box.get_b() > 0.0 &&
         frame_box.get_c() > 0.0;
}

bool TitanQNativeAnalyzer::pair_allowed(
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

double TitanQNativeAnalyzer::native_pair_distance(
    const std::vector<titan_contact_geometry::ContactPoint> &row_points,
    const std::vector<titan_contact_geometry::ContactPoint> &col_points,
    const titan_contact::ResiduePairKey &key,
    const BoxMatrix &box_matrix) const {
  double nearest_sq = std::numeric_limits<double>::infinity();
  for (const titan_contact_geometry::ContactPoint &lhs : row_points) {
    if (!(lhs.residue == key.first))
      continue;
    for (const titan_contact_geometry::ContactPoint &rhs : col_points) {
      if (!(rhs.residue == key.second))
        continue;
      double dx = lhs.x - rhs.x;
      double dy = lhs.y - rhs.y;
      double dz = lhs.z - rhs.z;
      box_matrix.apply_mic(dx, dy, dz);
      const double distance_sq = dx * dx + dy * dy + dz * dz;
      nearest_sq = std::min(nearest_sq, distance_sq);
    }
  }
  return std::isfinite(nearest_sq) ? std::sqrt(nearest_sq)
                                   : std::numeric_limits<double>::infinity();
}

void TitanQNativeAnalyzer::build_native_contacts_from_reference(
    const Multi_MOL &mol1, const Multi_MOL *mol2) {
  native_contacts_.clear();
  native_pair_indices_.clear();
  if (mol1.m_mol.empty())
    return;

  std::vector<titan_contact_geometry::ContactPoint> row_points;
  std::vector<titan_contact_geometry::ContactPoint> col_points;
  if (!row_geometry_.build_points(mol1, 0, row_points))
    return;
  if (scope_ == "interface") {
    if (mol2 == nullptr || mol2->m_mol.empty() ||
        !col_geometry_.build_points(*mol2, 0, col_points))
      return;
  } else {
    col_points = row_points;
  }

  std::optional<BoxMatrix> box_matrix;
  if (BoxMatrix::valid_box(mol1.m_mol.front().Box))
    box_matrix.emplace(mol1.m_mol.front().Box);
  std::unordered_map<titan_contact::ResiduePairKey, double,
                     titan_contact::ResiduePairKeyHash>
      nearest;

  auto consider_pair = [&](const titan_contact_geometry::ContactPoint &lhs,
                           const titan_contact_geometry::ContactPoint &rhs) {
    if (!pair_allowed(lhs.residue, rhs.residue))
      return;
    double dx = lhs.x - rhs.x;
    double dy = lhs.y - rhs.y;
    double dz = lhs.z - rhs.z;
    if (box_matrix)
      box_matrix->apply_mic(dx, dy, dz);
    const double distance_sq = dx * dx + dy * dy + dz * dz;
    if (distance_sq > cutoff_a_ * cutoff_a_)
      return;
    const auto key = scope_ == "interface"
                         ? titan_contact::ResiduePairKey{lhs.residue,
                                                         rhs.residue}
                         : titan_contact::make_ordered_pair(lhs.residue,
                                                            rhs.residue);
    const double distance = std::sqrt(distance_sq);
    auto found = nearest.find(key);
    if (found == nearest.end() || distance < found->second)
      nearest[key] = distance;
  };

  if (scope_ == "interface") {
    for (const titan_contact_geometry::ContactPoint &lhs : row_points)
      for (const titan_contact_geometry::ContactPoint &rhs : col_points)
        consider_pair(lhs, rhs);
  } else {
    for (std::size_t i = 0; i < row_points.size(); ++i)
      for (std::size_t j = i + 1; j < row_points.size(); ++j)
        consider_pair(row_points[i], row_points[j]);
  }

  native_contacts_.reserve(nearest.size());
  for (const auto &[key, distance] : nearest) {
    const std::size_t row_idx = residue_index(row_residues_, key.first);
    const std::size_t col_idx = residue_index(col_residues_, key.second);
    if (row_idx == std::numeric_limits<std::size_t>::max() ||
        col_idx == std::numeric_limits<std::size_t>::max())
      continue;
    native_contacts_.push_back(NativeContact{key, row_idx, col_idx, distance});
  }
  std::sort(native_contacts_.begin(), native_contacts_.end(),
            [](const NativeContact &lhs, const NativeContact &rhs) {
              if (residue_less(lhs.key.first, rhs.key.first))
                return true;
              if (residue_less(rhs.key.first, lhs.key.first))
                return false;
              return residue_less(lhs.key.second, rhs.key.second);
            });
  native_pair_indices_.reserve(native_contacts_.size());
  for (const NativeContact &contact : native_contacts_) {
    native_pair_indices_.push_back(
        titan_contact_kernel::NativePairIndex{contact.row_index,
                                              contact.col_index});
  }
}

void TitanQNativeAnalyzer::init(const Multi_MOL &mol1, const Multi_MOL *mol2,
                                const arguments &, std::fstream &log,
                                const Multi_MOL *trajectory_mol1,
                                const Multi_MOL *trajectory_mol2) {
  if (!row_geometry_.build(mol1, 1, mode_))
    throw std::invalid_argument(
        "QNative: failed to build molecule 1 contact geometry.");
  row_residues_ = row_geometry_.residues();
  if (scope_ == "interface") {
    if (mol2 == nullptr || mol2->info.empty())
      throw std::invalid_argument(
          "QNative: -qnative-scope=interface requires a valid molecule 2/group 2.");
    if (!col_geometry_.build(*mol2, 2, mode_))
      throw std::invalid_argument(
          "QNative: failed to build molecule 2 contact geometry.");
    col_residues_ = col_geometry_.residues();
  } else {
    col_geometry_ = row_geometry_;
    col_residues_ = row_residues_;
  }

  build_native_contacts_from_reference(mol1, mol2);
  if (!trajectory_row_geometry_.build(
          trajectory_mol1 != nullptr ? *trajectory_mol1 : mol1, 1, mode_))
    throw std::invalid_argument(
        "QNative: failed to build trajectory molecule 1 contact geometry.");
  if (scope_ == "interface") {
    const Multi_MOL *traj_mol2 =
        trajectory_mol2 != nullptr ? trajectory_mol2 : mol2;
    if (traj_mol2 != nullptr) {
      if (!trajectory_col_geometry_.build(*traj_mol2, 2, mode_))
        throw std::invalid_argument(
            "QNative: failed to build trajectory molecule 2 contact geometry.");
    }
  } else {
    trajectory_col_geometry_ = trajectory_row_geometry_;
  }
  initialized_ = true;
  enabled_ = requested_ && !row_residues_.empty() && !col_residues_.empty();
  reset_trajectory_state();

  log_qnative_value(log, "QNative Mode",
                    titan_contact::contact_mode_name(mode_));
  log_qnative_value(log, "QNative Scope", scope_);
  log_qnative_value(log, "QNative Cutoff", std::to_string(cutoff_a_) + " A");
  log_qnative_value(log, "QNative Native Contacts",
                    std::to_string(native_contacts_.size()));
  log_qnative_value(log, "QNative Reference", reference_source_);
}

std::size_t TitanQNativeAnalyzer::residue_index(
    const std::vector<titan_contact::ResidueKey> &residues,
    const titan_contact::ResidueKey &residue) const {
  const auto found = std::find(residues.begin(), residues.end(), residue);
  return found == residues.end()
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(std::distance(residues.begin(), found));
}

std::string TitanQNativeAnalyzer::residue_label(
    const titan_contact::ResidueKey &residue) const {
  std::ostringstream out;
  out << molecule_role(residue.molecule) << '|' << residue.chain << ':'
      << residue.resnr << ':' << residue.resname;
  return out.str();
}

double TitanQNativeAnalyzer::contact_contribution(double distance_a) const {
  if (!std::isfinite(distance_a))
    return 0.0;
  if (!soft_)
    return distance_a <= cutoff_a_ ? 1.0 : 0.0;
  const double exponent = beta_ * (distance_a - cutoff_a_);
  if (exponent > 60.0)
    return 0.0;
  if (exponent < -60.0)
    return 1.0;
  return 1.0 / (1.0 + std::exp(exponent));
}

void TitanQNativeAnalyzer::process_frame(const Multi_MOL &mol1,
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
  if (!trajectory_row_geometry_.build_points(mol1, snap_idx, row_points) ||
      !(scope_ == "interface"
            ? trajectory_col_geometry_.build_points(*mol2, snap_idx,
                                                     col_points)
            : (col_points = row_points, true))) {
    ++skipped_incomplete_frames_;
    return;
  }

  ++valid_frames_;
  const BoxMatrix box_matrix(frame.Box);
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

  double formed_sum = 0.0;
  std::vector<double> distances(native_contacts_.size(), 0.0);
  const std::size_t worker_count =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<double> worker_formed(worker_count, 0.0);
  executor.parallel_for(native_contacts_.size(), worker_count, 64,
                        [&](std::size_t worker_id, std::size_t begin,
                            std::size_t end) {
    titan_contact_kernel::native_distances_optimized(
        row_soa, col_soa, native_pair_indices_, begin, end, box_matrix,
        distances);
    double local_formed = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      const double contribution = contact_contribution(distances[i]);
      local_formed += contribution;
      occupancy_[i].formed_sum += contribution;
      if (distances[i] <= cutoff_a_)
        ++occupancy_[i].hard_frames;
    }
    worker_formed[worker_id] = local_formed;
  });
  for (const double value : worker_formed)
    formed_sum += value;

  const double total_native = static_cast<double>(native_contacts_.size());
  const double q = total_native > 0.0 ? formed_sum / total_native : 0.0;
  timeseries_.push_back(FrameRecord{frame.get_time(), q, formed_sum});
  q_values_.push_back(q);
  all_q_values_.push_back(q);
}

double TitanQNativeAnalyzer::mean_q() const {
  if (q_values_.empty())
    return 0.0;
  return std::accumulate(q_values_.begin(), q_values_.end(), 0.0) /
         static_cast<double>(q_values_.size());
}

double TitanQNativeAnalyzer::std_q() const {
  if (q_values_.size() < 2)
    return 0.0;
  const double mean = mean_q();
  double variance = 0.0;
  for (double q : q_values_)
    variance += (q - mean) * (q - mean);
  variance /= static_cast<double>(q_values_.size() - 1);
  return std::sqrt(variance);
}

double TitanQNativeAnalyzer::min_q() const {
  if (q_values_.empty())
    return 0.0;
  return *std::min_element(q_values_.begin(), q_values_.end());
}

double TitanQNativeAnalyzer::max_q() const {
  if (q_values_.empty())
    return 0.0;
  return *std::max_element(q_values_.begin(), q_values_.end());
}

void TitanQNativeAnalyzer::export_summary(const std::string &filename,
                                          std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_qnative_value(log, "QNative Summary", "failed to open " + filename);
    return;
  }
  out << std::fixed << std::setprecision(6);
  out << "module: titan_qnative\n";
  out << "mode: " << titan_contact::contact_mode_name(mode_) << '\n';
  out << "scope: " << scope_ << '\n';
  out << "reference_source: " << reference_source_ << '\n';
  out << "cutoff_A: " << cutoff_a_ << '\n';
  if (scope_ == "mol1") {
    out << "domain: " << domain_ << '\n';
    out << "min_seqsep: " << min_seqsep_ << '\n';
  }
  out << "soft: " << (soft_ ? "true" : "false") << '\n';
  out << "beta: " << beta_ << '\n';
  out << "frames_seen: " << total_frames_seen_ << '\n';
  out << "valid_frames: " << valid_frames_ << '\n';
  out << "skipped_invalid_box_frames: " << skipped_invalid_box_frames_ << '\n';
  out << "skipped_incomplete_frames: " << skipped_incomplete_frames_ << '\n';
  out << "total_native_contacts: " << native_contacts_.size() << '\n';
  out << "mean_Q: " << mean_q() << '\n';
  out << "std_Q: " << std_q() << '\n';
  out << "min_Q: " << min_q() << '\n';
  out << "max_Q: " << max_q() << '\n';
  log_qnative_value(log, "QNative Summary", filename);
}

void TitanQNativeAnalyzer::export_native_contacts(const std::string &filename,
                                                  std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_qnative_value(log, "QNative Native Contacts",
                      "failed to open " + filename);
    return;
  }
  out << "contact_id,molecule_i,molecule_j,residue_i,residue_j,reference_distance_A\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < native_contacts_.size(); ++i) {
    const NativeContact &contact = native_contacts_[i];
    out << (i + 1) << ',' << molecule_role(contact.key.first.molecule) << ','
        << molecule_role(contact.key.second.molecule) << ','
        << residue_label(contact.key.first) << ','
        << residue_label(contact.key.second) << ','
        << contact.reference_distance_a << '\n';
  }
  log_qnative_value(log, "QNative Native Contacts", filename);
}

void TitanQNativeAnalyzer::export_timeseries(const std::string &filename,
                                             std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_qnative_value(log, "QNative Timeseries", "failed to open " + filename);
    return;
  }
  out << "frame,time_ps,Q,formed_native_contacts,total_native_contacts\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < timeseries_.size(); ++i) {
    out << i << ',' << timeseries_[i].time_ps << ',' << timeseries_[i].q
        << ',' << timeseries_[i].formed << ',' << native_contacts_.size()
        << '\n';
  }
  log_qnative_value(log, "QNative Timeseries", filename);
}

void TitanQNativeAnalyzer::export_occupancy(const std::string &filename,
                                            std::fstream &log) const {
  std::ofstream out(filename);
  if (!out) {
    log_qnative_value(log, "QNative Occupancy", "failed to open " + filename);
    return;
  }
  out << "contact_id,residue_i,residue_j,occupancy_probability,formed_sum,valid_frames,hard_contact_frames\n";
  out << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < occupancy_.size(); ++i) {
    const ContactOccupancy &record = occupancy_[i];
    const double probability =
        valid_frames_ > 0 ? record.formed_sum / static_cast<double>(valid_frames_)
                          : 0.0;
    out << (i + 1) << ',' << residue_label(record.key.first) << ','
        << residue_label(record.key.second) << ',' << probability << ','
        << record.formed_sum << ',' << valid_frames_ << ','
        << record.hard_frames << '\n';
  }
  log_qnative_value(log, "QNative Occupancy", filename);
}

void TitanQNativeAnalyzer::finalize_trajectory(
    const std::string &base_dir, const info &info_output, const arguments &args,
    const std::vector<std::string> &input_line, int input_name,
    std::fstream &log) {
  if (!initialized_ || !requested_)
    return;
  accumulate_pooled_state();
  export_summary(join_output_path(base_dir, info_output.get_qnative_summary_filename(
                                                args, input_line, input_name)),
                 log);
  export_native_contacts(
      join_output_path(base_dir, info_output.get_qnative_native_contacts_filename(
                                     args, input_line, input_name)),
      log);
  export_timeseries(
      join_output_path(base_dir, info_output.get_qnative_timeseries_filename(
                                     args, input_line, input_name)),
      log);
  export_occupancy(
      join_output_path(base_dir, info_output.get_qnative_occupancy_filename(
                                     args, input_line, input_name)),
      log);
  reset_trajectory_state();
}

void TitanQNativeAnalyzer::finalize_all(const std::string &base_dir,
                                        const info &info_output,
                                        std::fstream &log) {
  if (!requested_ || pooled_valid_frames_ == 0)
    return;
  const auto saved_valid_frames = valid_frames_;
  const auto saved_total_frames = total_frames_seen_;
  const auto saved_skipped_box = skipped_invalid_box_frames_;
  const auto saved_skipped_incomplete = skipped_incomplete_frames_;
  const auto saved_occupancy = occupancy_;
  const auto saved_timeseries = timeseries_;
  const auto saved_q_values = q_values_;
  total_frames_seen_ = pooled_total_frames_seen_;
  valid_frames_ = pooled_valid_frames_;
  skipped_invalid_box_frames_ = pooled_skipped_invalid_box_frames_;
  skipped_incomplete_frames_ = pooled_skipped_incomplete_frames_;
  occupancy_ = pooled_occupancy_;
  timeseries_ = pooled_timeseries_;
  q_values_ = pooled_q_values_;
  export_summary(join_output_path(
                     base_dir, info_output.get_qnative_summary_average_filename()),
                 log);
  export_native_contacts(join_output_path(
                             base_dir,
                             info_output.get_qnative_native_contacts_average_filename()),
                         log);
  export_timeseries(join_output_path(
                        base_dir,
                        info_output.get_qnative_timeseries_average_filename()),
                    log);
  export_occupancy(join_output_path(
                       base_dir,
                       info_output.get_qnative_occupancy_average_filename()),
                   log);
  total_frames_seen_ = saved_total_frames;
  valid_frames_ = saved_valid_frames;
  skipped_invalid_box_frames_ = saved_skipped_box;
  skipped_incomplete_frames_ = saved_skipped_incomplete;
  occupancy_ = saved_occupancy;
  timeseries_ = saved_timeseries;
  q_values_ = saved_q_values;
}

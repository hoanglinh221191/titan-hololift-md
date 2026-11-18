#include "titan_msd.h"

#include "BoxMatrix.h"
#include "info.h"
#include "memory_utils.h"
#include "titan_executor.h"
#include "titan_ui.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {

std::string join_output_path(const std::string &base_dir,
                             const std::string &filename) {
  if (base_dir.empty() || base_dir == ".")
    return filename;
  return (std::filesystem::path(base_dir) / filename).string();
}

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<TitanMSDAnalyzer::ResidueRange>
parse_ranges(const std::string &raw_selector) {
  std::vector<TitanMSDAnalyzer::ResidueRange> ranges;
  std::stringstream ss(raw_selector);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token = trim_copy(token);
    if (token.empty())
      continue;

    TitanMSDAnalyzer::ResidueRange range;
    const auto colon = token.find(':');
    std::string span = token;
    if (colon != std::string::npos) {
      range.has_chain = true;
      range.chain = token.substr(0, colon);
      span = token.substr(colon + 1);
    }

    const auto dash = span.find('-');
    try {
      if (dash == std::string::npos) {
        range.begin = static_cast<std::size_t>(std::stoul(span));
        range.end = range.begin;
      } else {
        range.begin = static_cast<std::size_t>(std::stoul(span.substr(0, dash)));
        range.end = static_cast<std::size_t>(std::stoul(span.substr(dash + 1)));
      }
    } catch (...) {
      throw std::invalid_argument("MSD: invalid residue selector.");
    }
    if (range.begin == 0 || range.end == 0 || range.begin > range.end)
      throw std::invalid_argument("MSD: invalid residue selector.");
    ranges.push_back(std::move(range));
  }
  if (ranges.empty())
    throw std::invalid_argument("MSD: residue selector is empty.");
  return ranges;
}

void apply_mic_delta(const box &frame_box, double &dx, double &dy, double &dz) {
  BoxMatrix matrix(frame_box);
  matrix.apply_mic(dx, dy, dz);
}

} // namespace

void TitanMSDAnalyzer::configure(const MsdArgs &args) {
  requested_ = args.flag_msd;
  selection_ = args.selection;
  mode_ = args.mode;
  maxlag_ = args.maxlag;
  fit_window_ = args.fit_window;
  pbc_ = args.pbc;
  rotational_requested_ = args.rot;
  rotational_fit_window_ = args.rot_fit_window;
  memory_limit_ = args.memory_limit;
  rotational_analyzer_.configure(rotational_requested_, rotational_fit_window_,
                                 maxlag_, pbc_, selection_);
  use_mol2_ = selection_ == "mol2";
  enabled_ = requested_ && initialized_;
}

void TitanMSDAnalyzer::init(const Multi_MOL &mol1, const Multi_MOL *mol2,
                            const arguments &args, std::fstream &log) {
  if (!requested_)
    return;
  const Multi_MOL &target = target_molecule(mol1, mol2);
  build_selection(target, args);
  if (selected_atoms_.empty())
    throw std::runtime_error("MSD: selected atom set is empty.");
  point_count_ = mode_ == "atom" ? selected_atoms_.size() : 1;
  initialized_ = point_count_ > 0;
  enabled_ = requested_ && initialized_;
  rotational_analyzer_.init(selected_atoms_, selected_masses_, log);

  // STR-13. Bytes kept per frame: time and box, three doubles per point
  // (atom mode: per selected atom; com mode: raw and reconstructed COM), the
  // same again per atom for -msd-rot. Budget: -msd-memory-limit MB, or half
  // the free RAM at start (at least 1024 MB).
  constexpr std::size_t kMiB = 1024ULL * 1024ULL;
  bytes_per_frame_ = sizeof(double) + sizeof(box) +
                     3 * sizeof(double) * point_count_ * (mode_ == "atom" ? 1 : 2);
  if (rotational_requested_)
    bytes_per_frame_ += sizeof(double) + sizeof(box) +
                        3 * sizeof(double) * selected_atoms_.size();
  if (memory_limit_ == "auto") {
    memory_budget_bytes_ = std::max<std::size_t>(1024 * kMiB, get_available_ram() / 2);
  } else {
    memory_budget_bytes_ = static_cast<std::size_t>(
        std::max(1.0, std::stod(memory_limit_) * static_cast<double>(kMiB)));
  }
  const std::size_t frame_limit = memory_budget_bytes_ / std::max<std::size_t>(1, bytes_per_frame_);
  if (mode_ == "atom" || rotational_requested_) {
    std::ostringstream note;
    note << "MSD keeps every frame: " << bytes_per_frame_ << " B per frame ("
         << selected_atoms_.size() << " atoms), memory limit "
         << (memory_budget_bytes_ / kMiB) << " MB = " << frame_limit
         << " frames (-msd-memory-limit).";
    TitanUI::print_info(note.str());
  }
  if (log.is_open()) {
    TitanUI::log_kv(log, "MSD Selection", selection_);
    TitanUI::log_kv(log, "MSD Mode", mode_);
    TitanUI::log_kv(log, "MSD PBC", pbc_);
    TitanUI::log_kv(log, "MSD Selected Atoms",
                    std::to_string(selected_atoms_.size()));
    TitanUI::log_kv(log, "MSD Bytes Per Frame", std::to_string(bytes_per_frame_));
    TitanUI::log_kv(log, "MSD Memory Limit Frames", std::to_string(frame_limit));
  }
  reset_trajectory_state();
}

void TitanMSDAnalyzer::check_memory_budget() const {
  const std::size_t stored = time_ps_.size();
  if ((stored + 1) * bytes_per_frame_ <= memory_budget_bytes_)
    return;
  constexpr double kMiB = 1024.0 * 1024.0;
  std::ostringstream message;
  message << std::fixed << std::setprecision(3)
          << "MSD: frame " << (stored + 1) << " would raise the stored "
          << "coordinates to " << static_cast<double>((stored + 1) * bytes_per_frame_) / kMiB
          << " MB (" << bytes_per_frame_ << " B per frame for "
          << selected_atoms_.size() << " selected atoms in " << mode_
          << " mode), above the MSD memory limit of "
          << static_cast<double>(memory_budget_bytes_) / kMiB
          << " MB (-msd-memory-limit; auto = half the free RAM). Every frame is "
             "kept for the multiple-origin MSD and its time blocks: use "
             "-msd-mode=com, a smaller -msd-selection, fewer frames (-b/-e/-dt) "
             "or a larger -msd-memory-limit.";
  throw std::runtime_error(message.str());
}

void TitanMSDAnalyzer::reset_trajectory_state() {
  time_ps_.clear();
  boxes_.clear();
  box_valid_.clear();
  raw_x_.clear();
  raw_y_.clear();
  raw_z_.clear();
  reconstructed_com_x_.clear();
  reconstructed_com_y_.clear();
  reconstructed_com_z_.clear();
  rotational_analyzer_.reset_trajectory_state();
}

const Multi_MOL &
TitanMSDAnalyzer::target_molecule(const Multi_MOL &mol1,
                                  const Multi_MOL *mol2) const {
  if (!use_mol2_)
    return mol1;
  if (mol2 == nullptr || mol2->info.empty())
    throw std::runtime_error("MSD: -msd-selection=mol2 requires molecule 2.");
  return *mol2;
}

void TitanMSDAnalyzer::build_selection(const Multi_MOL &target,
                                       const arguments &args) {
  selected_atoms_.clear();
  selected_masses_.clear();

  if (selection_ == "mol1" || selection_ == "mol2") {
    titan_selection::build_all_atom_selection(target, selected_atoms_);
  } else if (selection_ == "trace") {
    titan_selection::build_trace_atoms(target, target.get_mol_type().empty()
                                                   ? args.mol_1_type
                                                   : std::string(target.get_mol_type()),
                                       target.force_f, selected_atoms_);
  } else if (selection_ == "backbone") {
    titan_selection::build_backbone_atoms(target, selected_atoms_);
  } else if (selection_.rfind("atom:", 0) == 0) {
    titan_selection::build_named_atom_selection(
        target, titan_selection::parse_name_list(selection_.substr(5)),
        selected_atoms_);
  } else if (selection_.rfind("residue:", 0) == 0) {
    build_residue_selection(target, selection_.substr(8));
  }

  selected_masses_.reserve(selected_atoms_.size());
  for (const auto &atom : selected_atoms_) {
    const std::size_t idx = static_cast<std::size_t>(atom.atom_index);
    selected_masses_.push_back(idx < target.parameters.mass.size()
                                   ? target.parameters.mass[idx]
                                   : 0.0);
  }
}

void TitanMSDAnalyzer::build_residue_selection(const Multi_MOL &target,
                                               const std::string &selector) {
  const std::vector<ResidueRange> ranges = parse_ranges(selector);
  for (std::size_t atom_idx = 0; atom_idx < target.info.size(); ++atom_idx) {
    const auto &atom = target.info[atom_idx];
    if (!residue_in_ranges(atom, ranges))
      continue;
    titan_selection::SelectedAtom selected;
    selected.atom_index = static_cast<int>(atom_idx);
    selected.ordinal = selected_atoms_.size() + 1;
    selected.resnr = atom.get_resnr();
    selected.chainnum = atom.get_chainnum();
    selected.chain = atom.get_chain();
    selected.resname = atom.get_res_name();
    selected.atom_name = titan_selection::trim_copy(atom.get_atom());
    selected_atoms_.push_back(std::move(selected));
  }
}

bool TitanMSDAnalyzer::residue_in_ranges(
    const mol_info &atom, const std::vector<ResidueRange> &ranges) const {
  for (const ResidueRange &range : ranges) {
    if (range.has_chain && range.chain != atom.get_chain())
      continue;
    if (atom.get_resnr() >= range.begin && atom.get_resnr() <= range.end)
      return true;
  }
  return false;
}

bool TitanMSDAnalyzer::valid_box(const box &frame_box) const noexcept {
  return BoxMatrix::valid_box(frame_box);
}

double TitanMSDAnalyzer::atom_mass(std::size_t atom_index) const noexcept {
  if (atom_index >= selected_masses_.size())
    return 0.0;
  return selected_masses_[atom_index] > 0.0 ? selected_masses_[atom_index] : 0.0;
}

void TitanMSDAnalyzer::process_frame(const Multi_MOL &mol1,
                                     const Multi_MOL *mol2,
                                     std::size_t snap_idx,
                                     TitanExecutor &) {
  if (!enabled_)
    return;
  const Multi_MOL &target = target_molecule(mol1, mol2);
  if (snap_idx >= target.m_mol.size())
    return;
  append_frame_points(target.m_mol[snap_idx]);
  rotational_analyzer_.append_frame(target.m_mol[snap_idx]);
}

void TitanMSDAnalyzer::append_frame_points(const MOL &frame) {
  if (!frame.coordinates_consistent())
    throw std::runtime_error("MSD: inconsistent coordinate arrays.");
  const double time_ps = frame.get_time();
  if (!std::isfinite(time_ps) ||
      (!time_ps_.empty() && !(time_ps > time_ps_.back())))
    throw std::runtime_error(
        "MSD: frame timestamps must be finite and strictly increasing.");
  check_memory_budget();
  time_ps_.push_back(time_ps);
  boxes_.push_back(frame.Box);
  box_valid_.push_back(valid_box(frame.Box));

  if (mode_ == "atom") {
    for (const auto &atom : selected_atoms_) {
      const std::size_t idx = static_cast<std::size_t>(atom.atom_index);
      if (idx >= frame.pos.x.size() || idx >= frame.pos.y.size() ||
          idx >= frame.pos.z.size())
        throw std::runtime_error("MSD: selected atom index exceeds frame size.");
      raw_x_.push_back(frame.pos.x[idx]);
      raw_y_.push_back(frame.pos.y[idx]);
      raw_z_.push_back(frame.pos.z[idx]);
    }
    return;
  }

  double wx = 0.0;
  double wy = 0.0;
  double wz = 0.0;
  double mass_sum = 0.0;
  double ux = 0.0;
  double uy = 0.0;
  double uz = 0.0;
  double reconstructed_wx = 0.0;
  double reconstructed_wy = 0.0;
  double reconstructed_wz = 0.0;
  double reconstructed_ux = 0.0;
  double reconstructed_uy = 0.0;
  double reconstructed_uz = 0.0;
  std::size_t valid_atoms = 0;
  const bool reconstruct_body = box_valid_.back();
  const std::optional<BoxMatrix> box_matrix =
      reconstruct_body ? std::optional<BoxMatrix>{frame.Box} : std::nullopt;
  // STR-28 (22/10/2026). The old reconstruction imaged every atom next to
  // atom 0 in each frame, which broke any body larger than half the box (the
  // H10 dimer in a 74 A box) and made its centre of mass jump between frames
  // (~120 A^2 at 50 ps for a molecule held in place). Now each atom is
  // unwrapped in time (minimum image of its step since the previous frame)
  // and the centre of mass is taken from those continuous positions, which
  // also follows a chain that changes image relative to the other. The first
  // frame (or one after a frame without a box) is made whole by walking the
  // selection in order, each atom nearest to the previous one (as MDA-07).
  const std::size_t n_sel = selected_atoms_.size();
  const bool previous_usable = reconstruct_body && box_valid_.size() >= 2 &&
                               box_valid_[box_valid_.size() - 2] &&
                               unwrapped_atom_x_.size() == n_sel;
  const std::optional<BoxMatrix> step_matrix =
      previous_usable ? std::optional<BoxMatrix>{boxes_[boxes_.size() - 2]}
                      : std::nullopt;
  if (!previous_usable) {
    unwrapped_atom_x_.assign(n_sel, 0.0);
    unwrapped_atom_y_.assign(n_sel, 0.0);
    unwrapped_atom_z_.assign(n_sel, 0.0);
    previous_atom_x_.assign(n_sel, 0.0);
    previous_atom_y_.assign(n_sel, 0.0);
    previous_atom_z_.assign(n_sel, 0.0);
  }
  for (std::size_t i = 0; i < n_sel; ++i) {
    const std::size_t idx = static_cast<std::size_t>(selected_atoms_[i].atom_index);
    if (idx >= frame.pos.x.size() || idx >= frame.pos.y.size() ||
        idx >= frame.pos.z.size())
      throw std::runtime_error("MSD: selected atom index exceeds frame size.");
    ++valid_atoms;
    const double x = frame.pos.x[idx];
    const double y = frame.pos.y[idx];
    const double z = frame.pos.z[idx];
    double reconstructed_x = x;
    double reconstructed_y = y;
    double reconstructed_z = z;
    if (previous_usable) {
      double dx = x - previous_atom_x_[i], dy = y - previous_atom_y_[i],
             dz = z - previous_atom_z_[i];
      step_matrix->apply_mic(dx, dy, dz);
      reconstructed_x = unwrapped_atom_x_[i] + dx;
      reconstructed_y = unwrapped_atom_y_[i] + dy;
      reconstructed_z = unwrapped_atom_z_[i] + dz;
    } else if (reconstruct_body && i > 0) {
      double dx = x - previous_atom_x_[i - 1], dy = y - previous_atom_y_[i - 1],
             dz = z - previous_atom_z_[i - 1];
      box_matrix->apply_mic(dx, dy, dz);
      reconstructed_x = unwrapped_atom_x_[i - 1] + dx;
      reconstructed_y = unwrapped_atom_y_[i - 1] + dy;
      reconstructed_z = unwrapped_atom_z_[i - 1] + dz;
    }
    previous_atom_x_[i] = x;
    previous_atom_y_[i] = y;
    previous_atom_z_[i] = z;
    unwrapped_atom_x_[i] = reconstructed_x;
    unwrapped_atom_y_[i] = reconstructed_y;
    unwrapped_atom_z_[i] = reconstructed_z;
    ux += x;
    uy += y;
    uz += z;
    reconstructed_ux += reconstructed_x;
    reconstructed_uy += reconstructed_y;
    reconstructed_uz += reconstructed_z;
    const double mass = atom_mass(i);
    if (mass > 0.0) {
      wx += x * mass;
      wy += y * mass;
      wz += z * mass;
      reconstructed_wx += reconstructed_x * mass;
      reconstructed_wy += reconstructed_y * mass;
      reconstructed_wz += reconstructed_z * mass;
      mass_sum += mass;
    }
  }
  if (valid_atoms == 0)
    throw std::runtime_error("MSD: no valid atoms in frame.");
  if (mass_sum > 0.0) {
    raw_x_.push_back(wx / mass_sum);
    raw_y_.push_back(wy / mass_sum);
    raw_z_.push_back(wz / mass_sum);
    reconstructed_com_x_.push_back(reconstructed_wx / mass_sum);
    reconstructed_com_y_.push_back(reconstructed_wy / mass_sum);
    reconstructed_com_z_.push_back(reconstructed_wz / mass_sum);
  } else {
    const double denom = static_cast<double>(valid_atoms);
    raw_x_.push_back(ux / denom);
    raw_y_.push_back(uy / denom);
    raw_z_.push_back(uz / denom);
    reconstructed_com_x_.push_back(reconstructed_ux / denom);
    reconstructed_com_y_.push_back(reconstructed_uy / denom);
    reconstructed_com_z_.push_back(reconstructed_uz / denom);
  }
}

std::size_t TitanMSDAnalyzer::resolved_maxlag(std::size_t frames) const {
  std::size_t maxlag = frames > 0 ? frames - 1 : 0;
  if (maxlag_ == "auto")
    return std::min<std::size_t>(maxlag, 1000);
  return std::min<std::size_t>(maxlag,
                               static_cast<std::size_t>(std::stoul(maxlag_)));
}

void TitanMSDAnalyzer::compute_lag_results(
    const AlignedDoubleVector &calc_x, const AlignedDoubleVector &calc_y,
    const AlignedDoubleVector &calc_z, const std::string &effective_pbc,
    TitanExecutor &executor, std::vector<LagResult> &results) const {
  results.clear();
  const std::size_t frames = time_ps_.size();
  if (frames < 2 || point_count_ == 0)
    return;
  lag_results_in_range(calc_x, calc_y, calc_z, effective_pbc, 0, frames,
                       resolved_maxlag(frames), executor, results);
}

void TitanMSDAnalyzer::prepare_coordinates(AlignedDoubleVector &calc_x,
                                           AlignedDoubleVector &calc_y,
                                           AlignedDoubleVector &calc_z,
                                           std::string &effective_pbc,
                                           std::string &pbc_warning) {
  const std::size_t frames = time_ps_.size();
  const bool all_boxes_valid =
      std::all_of(box_valid_.begin(), box_valid_.end(), [](bool v) { return v; });
  if ((pbc_ == "unwrap" || pbc_ == "mic") && !all_boxes_valid)
    throw std::runtime_error("MSD: -msd-pbc=" + pbc_ +
                             " requires valid periodic boxes for all frames.");

  effective_pbc = pbc_;
  if (pbc_ == "auto") {
    effective_pbc = all_boxes_valid ? "unwrap" : "raw";
    if (!all_boxes_valid)
      pbc_warning = "auto selected raw coordinates because at least one frame has no valid box";
  }

  // moved, not copied: the stored frames are not needed after this
  if (mode_ == "com" &&
      (effective_pbc == "unwrap" || effective_pbc == "mic")) {
    calc_x = std::move(reconstructed_com_x_);
    calc_y = std::move(reconstructed_com_y_);
    calc_z = std::move(reconstructed_com_z_);
  } else {
    calc_x = std::move(raw_x_);
    calc_y = std::move(raw_y_);
    calc_z = std::move(raw_z_);
  }
  raw_x_.clear();
  raw_y_.clear();
  raw_z_.clear();
  reconstructed_com_x_.clear();
  reconstructed_com_y_.clear();
  reconstructed_com_z_.clear();
  if (effective_pbc == "unwrap") {
    for (std::size_t frame = 1; frame < frames; ++frame) {
      for (std::size_t point = 0; point < point_count_; ++point) {
        const std::size_t cur = frame * point_count_ + point;
        const std::size_t prev = (frame - 1) * point_count_ + point;
        double dx = calc_x[cur] - calc_x[prev];
        double dy = calc_y[cur] - calc_y[prev];
        double dz = calc_z[cur] - calc_z[prev];
        apply_mic_delta(boxes_[frame - 1], dx, dy, dz);
        calc_x[cur] = calc_x[prev] + dx;
        calc_y[cur] = calc_y[prev] + dy;
        calc_z[cur] = calc_z[prev] + dz;
      }
    }
  }
}

// MSD over origins and targets inside frames [first_frame, end_frame).
void TitanMSDAnalyzer::lag_results_in_range(
    const AlignedDoubleVector &calc_x, const AlignedDoubleVector &calc_y,
    const AlignedDoubleVector &calc_z, const std::string &effective_pbc,
    std::size_t first_frame, std::size_t end_frame, std::size_t maxlag,
    TitanExecutor &executor, std::vector<LagResult> &results) const {
  results.clear();
  const std::size_t frames = end_frame - first_frame;
  if (frames < 2)
    return;
  maxlag = std::min(maxlag, frames - 1);
  results.resize(maxlag);

  const std::size_t workers = std::max<std::size_t>(1, executor.capacity());
  executor.parallel_for(maxlag, workers, 8,
                        [&](std::size_t, std::size_t begin, std::size_t end) {
    for (std::size_t lag0 = begin; lag0 < end; ++lag0) {
      const std::size_t lag = lag0 + 1;
      double sum_d2 = 0.0;
      double sum_dt = 0.0;
      std::uint64_t samples = 0;
      const std::size_t origins = frames - lag;
      for (std::size_t origin = first_frame; origin < first_frame + origins;
           ++origin) {
        const std::size_t target = origin + lag;
        sum_dt += time_ps_[target] - time_ps_[origin];
        for (std::size_t point = 0; point < point_count_; ++point) {
          const std::size_t a = origin * point_count_ + point;
          const std::size_t b = target * point_count_ + point;
          double dx = calc_x[b] - calc_x[a];
          double dy = calc_y[b] - calc_y[a];
          double dz = calc_z[b] - calc_z[a];
          if (effective_pbc == "mic")
            apply_mic_delta(boxes_[origin], dx, dy, dz);
          sum_d2 += dx * dx + dy * dy + dz * dz;
          ++samples;
        }
      }
      results[lag0] = LagResult{
          lag, origins > 0 ? sum_dt / static_cast<double>(origins) : 0.0,
          samples > 0 ? sum_d2 / static_cast<double>(samples) : 0.0, samples};
    }
  });
}

std::pair<double, double> TitanMSDAnalyzer::resolve_fit_window(
    const std::vector<LagResult> &results) const {
  std::vector<double> valid_times;
  for (const LagResult &row : results) {
    if (row.samples > 0 && row.lag_time_ps > 0.0)
      valid_times.push_back(row.lag_time_ps);
  }
  if (valid_times.empty())
    return {0.0, 0.0};
  if (fit_window_ == "auto") {
    // STR-13 (MDA-04). Only lags that keep at least half of the frames as
    // time origins (lag <= frames/2), and at least the first three lags; then
    // 20-80 % of that lag range, so the fit ends at <= 40 % of the trajectory.
    // The old window was 20-80 % of every lag up to min(frames - 1, 1000):
    // for a short run it reached 80 % of the trajectory, where only 20 % of
    // the frames are origins. gmx msd fits 10-90 % of its lag range (all
    // lags by default), which has the same problem.
    const std::size_t frames = time_ps_.size();
    std::vector<double> eligible;
    for (const LagResult &row : results)
      if (row.samples > 0 && row.lag_time_ps > 0.0 && 2 * row.lag_index <= frames)
        eligible.push_back(row.lag_time_ps);
    if (eligible.size() < 3)
      eligible.assign(valid_times.begin(),
                      valid_times.begin() +
                          static_cast<std::ptrdiff_t>(std::min<std::size_t>(3, valid_times.size())));
    valid_times = std::move(eligible);
    const std::size_t n = valid_times.size();
    const std::size_t lo = std::min(n - 1, static_cast<std::size_t>(std::floor(0.2 * static_cast<double>(n - 1))));
    const std::size_t hi = std::min(n - 1, static_cast<std::size_t>(std::ceil(0.8 * static_cast<double>(n - 1))));
    return {valid_times[lo], valid_times[hi]};
  }
  const auto comma = fit_window_.find(',');
  if (comma == std::string::npos)
    throw std::invalid_argument("MSD: -msd-fit-window must be auto or tmin,tmax.");
  const double tmin = std::stod(fit_window_.substr(0, comma));
  const double tmax = std::stod(fit_window_.substr(comma + 1));
  if (!(tmax > tmin))
    throw std::invalid_argument("MSD: -msd-fit-window requires tmax > tmin.");
  return {tmin, tmax};
}

TitanMSDAnalyzer::FitResult
TitanMSDAnalyzer::fit_linear(const std::vector<LagResult> &results,
                             bool loglog) const {
  const auto [tmin, tmax] = resolve_fit_window(results);
  return fit_linear_window(results, loglog, tmin, tmax);
}

TitanMSDAnalyzer::TimeBlocks TitanMSDAnalyzer::compute_time_blocks(
    const std::string &blocks_spec, const FitResult &diffusion,
    const FitResult &alpha, const AlignedDoubleVector &calc_x,
    const AlignedDoubleVector &calc_y, const AlignedDoubleVector &calc_z,
    const std::string &effective_pbc, TitanExecutor &executor) const {
  TimeBlocks blocks;
  const std::size_t frames = time_ps_.size();
  if (frames < 10 || point_count_ == 0 || !diffusion.ok) {
    blocks.status = diffusion.ok ? "too_few_frames" : "no_diffusion_fit";
    return blocks;
  }
  // block count as for the other time-series sidecars: auto = 5 below 50
  // frames, else floor(sqrt(frames)) up to 20
  std::size_t requested = 0;
  if (blocks_spec == "auto") {
    requested = frames < 50 ? 5
                            : std::min<std::size_t>(
                                  20, static_cast<std::size_t>(std::floor(
                                          std::sqrt(static_cast<double>(frames)))));
  } else {
    try {
      requested = static_cast<std::size_t>(std::max(0, std::stoi(blocks_spec)));
    } catch (...) {
      requested = 0;
    }
  }
  // a block must hold the whole fit window twice over: each fitted lag then
  // has at least half the block's frames as origins
  auto block_span = [&](std::size_t count, std::size_t index) {
    const std::size_t base = frames / count;
    const std::size_t rem = frames % count;
    const std::size_t begin = index * base + std::min(index, rem);
    const std::size_t len = base + (index < rem ? 1 : 0);
    return std::pair<std::size_t, std::size_t>{begin, begin + len};
  };
  std::size_t count = std::min(requested, frames / 2);
  for (; count >= 2; --count) {
    bool long_enough = true;
    for (std::size_t b = 0; b < count && long_enough; ++b) {
      const auto [begin, end] = block_span(count, b);
      long_enough = time_ps_[end - 1] - time_ps_[begin] >= 2.0 * diffusion.tmax_ps;
    }
    if (long_enough)
      break;
  }
  if (count < 2) {
    blocks.status = "fit_window_exceeds_half_block";
    return blocks;
  }

  const std::size_t maxlag = resolved_maxlag(frames);
  std::vector<LagResult> block_results;
  double length_sum = 0.0;
  for (std::size_t b = 0; b < count; ++b) {
    const auto [begin, end] = block_span(count, b);
    lag_results_in_range(calc_x, calc_y, calc_z, effective_pbc, begin, end,
                         maxlag, executor, block_results);
    const FitResult fit = fit_linear_window(block_results, false,
                                            diffusion.tmin_ps,
                                            diffusion.tmax_ps);
    const FitResult block_alpha =
        fit_linear_window(block_results, true, alpha.tmin_ps, alpha.tmax_ps);
    if (!fit.ok) {
      blocks.status = "block_fit_failed";
      blocks.slopes.clear();
      blocks.alphas.clear();
      return blocks;
    }
    blocks.slopes.push_back(fit.slope);
    blocks.alphas.push_back(block_alpha.ok
                                ? block_alpha.slope
                                : std::numeric_limits<double>::quiet_NaN());
    length_sum += time_ps_[end - 1] - time_ps_[begin];
  }
  blocks.count = count;
  blocks.length_ps = length_sum / static_cast<double>(count);
  blocks.status = "ok";
  return blocks;
}

TitanMSDAnalyzer::FitResult
TitanMSDAnalyzer::fit_linear_window(const std::vector<LagResult> &results,
                                    bool loglog, double tmin,
                                    double tmax) const {
  FitResult fit;
  fit.tmin_ps = tmin;
  fit.tmax_ps = tmax;

  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (const LagResult &row : results) {
    if (row.samples == 0 || !std::isfinite(row.lag_time_ps) ||
        !std::isfinite(row.msd_a2) || row.lag_time_ps < tmin ||
        row.lag_time_ps > tmax)
      continue;
    if (loglog && (row.lag_time_ps <= 0.0 || row.msd_a2 <= 0.0))
      continue;
    const double x = loglog ? std::log(row.lag_time_ps) : row.lag_time_ps;
    const double y = loglog ? std::log(row.msd_a2) : row.msd_a2;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
    ++fit.n;
  }
  if (fit.n < 3) {
    fit.status = "insufficient_points";
    return fit;
  }
  const double n = static_cast<double>(fit.n);
  const double denom = n * sxx - sx * sx;
  if (std::abs(denom) <= 1.0e-18) {
    fit.status = "singular_fit";
    return fit;
  }
  fit.slope = (n * sxy - sx * sy) / denom;
  fit.intercept = (sy - fit.slope * sx) / n;
  const double rden = (n * sxx - sx * sx) * (n * syy - sy * sy);
  fit.r2 = rden > 0.0 ? ((n * sxy - sx * sy) * (n * sxy - sx * sy)) / rden
                      : 0.0;
  fit.ok = true;
  fit.status = "ok";
  return fit;
}

void TitanMSDAnalyzer::export_timeseries(
    const std::string &filename, const std::vector<LagResult> &results,
    std::fstream &log) const {
  std::ofstream out(filename);
  out << "lag_index lag_time_ps msd_A2 samples\n";
  out << std::fixed << std::setprecision(6);
  for (const LagResult &row : results)
    out << row.lag_index << ' ' << row.lag_time_ps << ' ' << row.msd_a2 << ' '
        << row.samples << '\n';
  if (log.is_open())
    TitanUI::log_kv(log, "MSD Timeseries", filename);
}

void TitanMSDAnalyzer::export_diffusion_summary(
    const std::string &filename, const std::vector<LagResult> &,
    const FitResult &fit, const std::string &effective_pbc,
    const std::string &pbc_warning, const TimeBlocks &blocks,
    std::fstream &log) const {
  std::ofstream out(filename);
  const double d_a2_per_ps = fit.ok ? fit.slope / 6.0 : 0.0;
  out << std::fixed << std::setprecision(8);
  out << "frames: " << time_ps_.size() << '\n';
  out << "points: " << point_count_ << '\n';
  out << "selected_atoms: " << selected_atoms_.size() << '\n';
  out << "mode: " << mode_ << '\n';
  out << "selection: " << selection_ << '\n';
  out << "pbc_policy_requested: " << pbc_ << '\n';
  out << "pbc_policy_effective: " << effective_pbc << '\n';
  if (!pbc_warning.empty())
    out << "pbc_warning: " << pbc_warning << '\n';
  out << "fit_status: " << fit.status << '\n';
  out << "fit_window_policy: "
      << (fit_window_ == "auto"
              ? "auto (20-80 % of the lags with lag <= frames/2)"
              : "user")
      << '\n';
  out << "fit_window_ps: " << fit.tmin_ps << ',' << fit.tmax_ps << '\n';
  out << "fit_points: " << fit.n << '\n';
  out << "slope_A2_per_ps: " << fit.slope << '\n';
  out << "D_A2_per_ps: " << d_a2_per_ps << '\n';
  out << "D_cm2_per_s: " << d_a2_per_ps * 1.0e-4 << '\n';
  out << "R2: " << fit.r2 << '\n';
  if (blocks.status != "not_requested") {
    // per-block MSD slope and alpha over the same fit window, for the
    // time-block uncertainty of D (STR-25); full precision for the sidecar
    auto join = [](const std::vector<double> &values) {
      std::ostringstream text;
      text << std::setprecision(17);
      for (std::size_t i = 0; i < values.size(); ++i)
        text << (i ? "," : "") << values[i];
      return text.str();
    };
    out << "time_block_status: " << blocks.status << '\n';
    out << "time_blocks: " << blocks.count << '\n';
    out << "time_block_length_ps: " << blocks.length_ps << '\n';
    out << "time_block_slope_A2_per_ps: " << join(blocks.slopes) << '\n';
    out << "time_block_alpha: " << join(blocks.alphas) << '\n';
  }
  if (log.is_open())
    TitanUI::log_kv(log, "MSD Diffusion Summary", filename);
}

void TitanMSDAnalyzer::export_alpha_fit(const std::string &filename,
                                        const FitResult &fit,
                                        std::fstream &log) const {
  std::ofstream out(filename);
  out << std::fixed << std::setprecision(8);
  out << "fit_status: " << fit.status << '\n';
  out << "fit_window_ps: " << fit.tmin_ps << ',' << fit.tmax_ps << '\n';
  out << "valid_points: " << fit.n << '\n';
  out << "alpha: " << fit.slope << '\n';
  out << "prefactor_A2_per_ps_alpha: " << (fit.ok ? std::exp(fit.intercept) : 0.0)
      << '\n';
  out << "R2: " << fit.r2 << '\n';
  if (log.is_open())
    TitanUI::log_kv(log, "MSD Alpha Fit", filename);
}

void TitanMSDAnalyzer::finalize_trajectory(
    const std::string &base_dir, const info &info_output, const arguments &args,
    const std::vector<std::string> &input_line, int input_name,
    TitanExecutor &executor, std::fstream &log) {
  if (!enabled_)
    return;
  std::vector<LagResult> results;
  std::string effective_pbc = "not_evaluated";
  std::string pbc_warning;
  AlignedDoubleVector calc_x, calc_y, calc_z;
  if (time_ps_.size() >= 2 && point_count_ > 0) {
    prepare_coordinates(calc_x, calc_y, calc_z, effective_pbc, pbc_warning);
    compute_lag_results(calc_x, calc_y, calc_z, effective_pbc, executor,
                        results);
  }
  const FitResult diffusion_fit = fit_linear(results, false);
  const FitResult alpha_fit = fit_linear(results, true);
  const TimeBlocks time_blocks =
      args.uncertainty.mode == "off"
          ? TimeBlocks{}
          : compute_time_blocks(args.uncertainty.blocks, diffusion_fit,
                                alpha_fit, calc_x, calc_y, calc_z,
                                effective_pbc, executor);
  export_timeseries(
      join_output_path(base_dir, info_output.get_msd_timeseries_filename(
                                     args, input_line, input_name)),
      results, log);
  export_diffusion_summary(
      join_output_path(base_dir,
                       info_output.get_msd_diffusion_summary_filename(
                           args, input_line, input_name)),
      results, diffusion_fit, effective_pbc, pbc_warning, time_blocks, log);
  export_alpha_fit(
      join_output_path(base_dir, info_output.get_msd_alpha_fit_filename(
                                     args, input_line, input_name)),
      alpha_fit, log);
  if (rotational_analyzer_.enabled()) {
    rotational_analyzer_.finalize(
        join_output_path(base_dir, info_output.get_rotational_msd_filename(
                                       args, input_line, input_name)),
        join_output_path(
            base_dir,
            info_output.get_rotational_quaternion_timeseries_filename(
                args, input_line, input_name)),
        join_output_path(
            base_dir, info_output.get_rotational_covariance_fit_filename(
                          args, input_line, input_name)),
        join_output_path(
            base_dir, info_output.get_rotational_diffusion_summary_filename(
                          args, input_line, input_name)),
        executor, log);
  }
  reset_trajectory_state();
}

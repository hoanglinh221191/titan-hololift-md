#include "titan_s2.h"

#include "BoxMatrix.h"
#include "titan_alignment.h"
#include "titan_executor.h"
#include "titan_selection_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {

constexpr double kEpsilon = 1.0e-12;

std::string join_output_path(const std::string &base_dir,
                             const std::string &filename) {
  if (base_dir.empty() || base_dir == ".")
    return filename;
  return (std::filesystem::path(base_dir) / filename).string();
}

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

std::string normalized_atom_name(std::string value) {
  return titan_selection::uppercase_copy(titan_selection::trim_copy(value));
}

std::string normalized_option(std::string value) {
  return lowercase_copy(titan_selection::trim_copy(std::move(value)));
}

std::pair<double, double> parse_fit_window(const std::string &raw) {
  const auto comma = raw.find(',');
  if (comma == std::string::npos)
    throw std::invalid_argument("S2: -s2-fit-window must be auto or tmin,tmax.");
  double tmin = 0.0;
  double tmax = 0.0;
  try {
    tmin = std::stod(raw.substr(0, comma));
    tmax = std::stod(raw.substr(comma + 1));
  } catch (...) {
    throw std::invalid_argument("S2: -s2-fit-window must be auto or tmin,tmax.");
  }
  if (!std::isfinite(tmin) || !std::isfinite(tmax) || tmin < 0.0 ||
      tmax <= tmin)
    throw std::invalid_argument(
        "S2: -s2-fit-window requires finite 0 <= tmin < tmax.");
  return {tmin, tmax};
}

bool has_non_collinear_rows(titan::algebra::Matrix3ConstView rows,
                            double eps = 1.0e-12) noexcept {
  const double eps2 = eps * eps;
  titan::algebra::Vec3 base{};
  bool have_base = false;
  for (std::size_t r = 0; r < rows.rows; ++r) {
    const titan::algebra::Vec3 current{rows(r, 0), rows(r, 1), rows(r, 2)};
    if (titan::algebra::norm2(current) <= eps2)
      continue;
    if (!have_base) {
      base = current;
      have_base = true;
      continue;
    }
    if (titan::algebra::norm2(titan::algebra::cross(base, current)) > eps2)
      return true;
  }
  return false;
}

} // namespace

void TitanS2Analyzer::configure(const TitanS2Config &args) {
  args_ = args;
  requested_ = args.flag_s2;
  args_.vectors = normalized_option(args_.vectors);
  args_.align = normalized_option(args_.align);
  args_.maxlag = normalized_option(args_.maxlag);
  args_.fit_window = normalized_option(args_.fit_window);
  if (args_.vectors != "nh" && args_.vectors != "ca-c" &&
      args_.vectors != "both")
    throw std::invalid_argument("S2: -s2-vector must be nh, ca-c, or both.");
  if (args_.align != "rms" && args_.align != "none")
    throw std::invalid_argument("S2: -s2-align must be rms or none.");
  if (args_.maxlag != "auto") {
    try {
      if (std::stoull(args_.maxlag) == 0)
        throw std::invalid_argument("zero");
    } catch (...) {
      throw std::invalid_argument("S2: -s2-maxlag must be auto or >= 1.");
    }
  }
  if (args_.fit_window != "auto")
    (void)parse_fit_window(args_.fit_window);
  enabled_ = requested_ && initialized_;
}

void TitanS2Analyzer::init(const Multi_MOL &mol1, std::fstream &log) {
  if (!requested_)
    return;
  build_topology_cache(mol1);
  parse_residue_selection();
  plans_.clear();
  missing_nh_skipped_ = 0;
  pro_nh_skipped_ = 0;
  for (std::size_t residue_index = 0; residue_index < residues_.size();
       ++residue_index) {
    const Residue &residue = residues_[residue_index];
    if (!selected(residue))
      continue;
    if (wants_nh()) {
      if (titan_selection::uppercase_copy(residue.resname) == "PRO") {
        ++missing_nh_skipped_;
        ++pro_nh_skipped_;
      } else {
        const auto n = residue.atom_by_name.find("N");
        auto h = residue.atom_by_name.find("H");
        if (h == residue.atom_by_name.end())
          h = residue.atom_by_name.find("HN");
        if (n != residue.atom_by_name.end() &&
            h != residue.atom_by_name.end()) {
          plans_.push_back({residue_index, residue.ordinal, residue.resnr,
                            residue.chainnum, residue.chain, residue.resname,
                            "NH", n->second, h->second});
        } else {
          ++missing_nh_skipped_;
        }
      }
    }
    if (wants_ca_c()) {
      const auto ca = residue.atom_by_name.find("CA");
      const auto c = residue.atom_by_name.find("C");
      if (ca != residue.atom_by_name.end() && c != residue.atom_by_name.end()) {
        plans_.push_back({residue_index, residue.ordinal, residue.resnr,
                          residue.chainnum, residue.chain, residue.resname,
                          "CA-C", ca->second, c->second});
      }
    }
  }
  build_alignment_trace();
  vectors_.resize(plans_.size());
  const std::size_t frame_reserve = mol1.m_mol.size();
  for (auto &vectors : vectors_) {
    vectors.clear();
    vectors.reserve(frame_reserve);
  }
  scratch_from_x_.resize(plans_.size());
  scratch_from_y_.resize(plans_.size());
  scratch_from_z_.resize(plans_.size());
  scratch_to_x_.resize(plans_.size());
  scratch_to_y_.resize(plans_.size());
  scratch_to_z_.resize(plans_.size());
  scratch_frame_vectors_.reserve(plans_.size());
  time_ps_.reserve(frame_reserve);
  initialized_ = !plans_.empty();
  enabled_ = requested_ && initialized_;
  reset_trajectory_state();
  if (log.is_open()) {
    log << "S2 vectors: " << args_.vectors << '\n';
    log << "S2 residue selection: " << args_.residues << '\n';
    log << "S2 alignment: " << args_.align << '\n';
    log << "S2 planned vectors: " << plans_.size() << '\n';
    if (missing_nh_skipped_ > 0)
      log << "S2 warning: skipped NH vectors: " << missing_nh_skipped_ << '\n';
  }
}

void TitanS2Analyzer::reset_trajectory_state() {
  time_ps_.clear();
  for (auto &vector : vectors_)
    vector.clear();
  first_snapshot_ = MOL{};
  have_first_snapshot_ = false;
  have_alignment_reference_ = false;
  frames_seen_ = 0;
  frames_used_ = 0;
  raw_box_fallback_frames_ = 0;
}

void TitanS2Analyzer::build_topology_cache(const Multi_MOL &mol1) {
  residues_.clear();
  reference_info_ = mol1.info;
  std::size_t previous_chain = std::numeric_limits<std::size_t>::max();
  std::size_t previous_resnr = std::numeric_limits<std::size_t>::max();
  for (std::size_t atom_index = 0; atom_index < mol1.info.size();
       ++atom_index) {
    const mol_info &atom = mol1.info[atom_index];
    if (residues_.empty() || atom.get_chainnum() != previous_chain ||
        atom.get_resnr() != previous_resnr) {
      Residue residue;
      residue.ordinal = residues_.size() + 1;
      residue.resnr = atom.get_resnr();
      residue.chainnum = atom.get_chainnum();
      residue.chain = atom.get_chain();
      residue.resname = atom.get_res_name();
      residues_.push_back(std::move(residue));
      previous_chain = atom.get_chainnum();
      previous_resnr = atom.get_resnr();
    }
    residues_.back().atom_by_name.emplace(normalized_atom_name(atom.get_atom()),
                                          static_cast<int>(atom_index));
  }
}

void TitanS2Analyzer::parse_residue_selection() {
  ranges_.clear();
  const std::string raw = titan_selection::trim_copy(args_.residues);
  if (raw.empty() || normalized_option(raw) == "all")
    return;
  std::stringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = titan_selection::trim_copy(token);
    if (token.empty())
      continue;
    ResidueRange range;
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
        range.begin = static_cast<std::size_t>(std::stoull(span));
        range.end = range.begin;
      } else {
        range.begin =
            static_cast<std::size_t>(std::stoull(span.substr(0, dash)));
        range.end =
            static_cast<std::size_t>(std::stoull(span.substr(dash + 1)));
      }
    } catch (...) {
      throw std::invalid_argument("S2: invalid -s2-residues selector.");
    }
    if ((range.has_chain && range.chain.empty()) || range.begin == 0 ||
        range.end == 0 || range.begin > range.end)
      throw std::invalid_argument("S2: invalid -s2-residues selector.");
    ranges_.push_back(std::move(range));
  }
  if (ranges_.empty())
    throw std::invalid_argument("S2: invalid -s2-residues selector.");
}

bool TitanS2Analyzer::selected(const Residue &residue) const {
  if (ranges_.empty())
    return true;
  for (const ResidueRange &range : ranges_) {
    if (range.has_chain && range.chain != residue.chain)
      continue;
    if (residue.resnr >= range.begin && residue.resnr <= range.end)
      return true;
  }
  return false;
}

bool TitanS2Analyzer::wants_nh() const noexcept {
  return args_.vectors == "nh" || args_.vectors == "both";
}

bool TitanS2Analyzer::wants_ca_c() const noexcept {
  return args_.vectors == "ca-c" || args_.vectors == "both";
}

void TitanS2Analyzer::build_alignment_trace() {
  alignment_atoms_.clear();
  for (const Residue &residue : residues_) {
    const auto ca = residue.atom_by_name.find("CA");
    if (ca != residue.atom_by_name.end())
      alignment_atoms_.push_back(ca->second);
  }
  alignment_reference_matrix_.resize(alignment_atoms_.size());
  alignment_mobile_matrix_.resize(alignment_atoms_.size());
}

void TitanS2Analyzer::ensure_alignment_reference(const MOL &frame) {
  if (args_.align == "none" || have_alignment_reference_)
    return;
  if (alignment_atoms_.size() < 3)
    throw std::runtime_error(
        "S2: rms alignment requires at least three CA trace atoms.");
  for (std::size_t i = 0; i < alignment_atoms_.size(); ++i) {
    const std::size_t atom = static_cast<std::size_t>(alignment_atoms_[i]);
    if (atom >= frame.size())
      throw std::runtime_error("S2: CA alignment atom exceeds frame size.");
    alignment_reference_matrix_(i, 0) = frame.pos.x[atom];
    alignment_reference_matrix_(i, 1) = frame.pos.y[atom];
    alignment_reference_matrix_(i, 2) = frame.pos.z[atom];
  }
  titan::algebra::Vec3 centroid{};
  for (std::size_t i = 0; i < alignment_reference_matrix_.rows(); ++i) {
    centroid.x += alignment_reference_matrix_(i, 0);
    centroid.y += alignment_reference_matrix_(i, 1);
    centroid.z += alignment_reference_matrix_(i, 2);
  }
  centroid = centroid / static_cast<double>(alignment_reference_matrix_.rows());
  for (std::size_t i = 0; i < alignment_reference_matrix_.rows(); ++i) {
    alignment_reference_matrix_(i, 0) -= centroid.x;
    alignment_reference_matrix_(i, 1) -= centroid.y;
    alignment_reference_matrix_(i, 2) -= centroid.z;
  }
  if (!has_non_collinear_rows(alignment_reference_matrix_.const_view()))
    throw std::runtime_error(
        "S2: rms alignment CA trace must contain non-collinear atoms.");
  have_alignment_reference_ = true;
}

void TitanS2Analyzer::append_vectors(const MOL &frame) {
  alignment_rotation_ = titan::algebra::Mat3::identity();
  if (args_.align == "rms") {
    ensure_alignment_reference(frame);
    for (std::size_t i = 0; i < alignment_atoms_.size(); ++i) {
      const std::size_t atom = static_cast<std::size_t>(alignment_atoms_[i]);
      if (atom >= frame.size())
        throw std::runtime_error("S2: CA alignment atom exceeds frame size.");
      alignment_mobile_matrix_(i, 0) = frame.pos.x[atom];
      alignment_mobile_matrix_(i, 1) = frame.pos.y[atom];
      alignment_mobile_matrix_(i, 2) = frame.pos.z[atom];
    }
    TitanAlignment::align_frame_inplace(alignment_reference_matrix_.const_view(),
                                        alignment_mobile_matrix_.view(),
                                        alignment_rotation_);
  }

  for (std::size_t plan_index = 0; plan_index < plans_.size(); ++plan_index) {
    const VectorPlan &plan = plans_[plan_index];
    const std::size_t from = static_cast<std::size_t>(plan.from_atom);
    const std::size_t to = static_cast<std::size_t>(plan.to_atom);
    if (from >= frame.size() || to >= frame.size())
      throw std::runtime_error("S2: bond-vector atom exceeds frame size.");
    scratch_from_x_[plan_index] = frame.pos.x[from];
    scratch_from_y_[plan_index] = frame.pos.y[from];
    scratch_from_z_[plan_index] = frame.pos.z[from];
    scratch_to_x_[plan_index] = frame.pos.x[to];
    scratch_to_y_[plan_index] = frame.pos.y[to];
    scratch_to_z_[plan_index] = frame.pos.z[to];
  }
  const bool valid_box = BoxMatrix::valid_box(frame.Box);
  titan_s2_kernel::build_bond_vectors_optimized(
      scratch_from_x_, scratch_from_y_, scratch_from_z_, scratch_to_x_,
      scratch_to_y_, scratch_to_z_, valid_box ? &frame.Box : nullptr,
      scratch_frame_vectors_);
  if (!valid_box)
    ++raw_box_fallback_frames_;
  if (args_.align == "rms") {
    scratch_rotation_ = {
        alignment_rotation_(0, 0), alignment_rotation_(1, 0),
        alignment_rotation_(2, 0), alignment_rotation_(0, 1),
        alignment_rotation_(1, 1), alignment_rotation_(2, 1),
        alignment_rotation_(0, 2), alignment_rotation_(1, 2),
        alignment_rotation_(2, 2)};
    titan_s2_kernel::rotate_bond_vectors_inplace_optimized(
        scratch_frame_vectors_, scratch_rotation_);
  }
  for (std::size_t i = 0; i < plans_.size(); ++i) {
    vectors_[i].push(scratch_frame_vectors_.x[i],
                     scratch_frame_vectors_.y[i],
                     scratch_frame_vectors_.z[i],
                     scratch_frame_vectors_.valid[i] != 0);
  }
}

void TitanS2Analyzer::process_frame(const Multi_MOL &mol1,
                                    std::size_t snap_idx,
                                    TitanExecutor &) {
  if (!enabled_)
    return;
  ++frames_seen_;
  if (snap_idx >= mol1.m_mol.size())
    return;
  if (!mol1.topology_matches_frame(snap_idx))
    throw std::runtime_error("S2: mol1 topology does not match frame.");
  const MOL &frame = mol1.m_mol[snap_idx];
  if (!frame.coordinates_consistent())
    throw std::runtime_error("S2: inconsistent coordinate arrays.");
  const double time_ps = frame.get_time();
  if (!std::isfinite(time_ps) ||
      (!time_ps_.empty() && !(time_ps > time_ps_.back())))
    throw std::runtime_error(
        "S2: frame timestamps must be finite and strictly increasing.");
  if (!have_first_snapshot_) {
    first_snapshot_ = frame;
    have_first_snapshot_ = true;
  }
  append_vectors(frame);
  time_ps_.push_back(time_ps);
  ++frames_used_;
}

std::size_t TitanS2Analyzer::resolve_maxlag() const {
  if (time_ps_.size() < 2)
    return 0;
  const std::size_t bounded = time_ps_.size() - 1;
  if (args_.maxlag == "auto")
    return std::min<std::size_t>(bounded, 1000);
  return std::min<std::size_t>(
      bounded, static_cast<std::size_t>(std::stoull(args_.maxlag)));
}

std::vector<titan_s2_kernel::CorrelationRow> TitanS2Analyzer::fit_rows(
    const std::vector<titan_s2_kernel::CorrelationRow> &rows) const {
  if (args_.fit_window == "auto")
    return rows;
  const auto [tmin, tmax] = parse_fit_window(args_.fit_window);
  std::vector<titan_s2_kernel::CorrelationRow> filtered;
  for (const auto &row : rows)
    if (row.lag_time_ps >= tmin && row.lag_time_ps <= tmax)
      filtered.push_back(row);
  return filtered;
}

double TitanS2Analyzer::weighted_fit_r2(
    const std::vector<titan_s2_kernel::CorrelationRow> &rows,
    const titan_s2_kernel::FitResult &fit) const noexcept {
  if (!fit.ok || !(fit.tau_ps > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  double weight_sum = 0.0;
  double weighted_mean = 0.0;
  for (const auto &row : rows) {
    if (row.samples == 0 || !std::isfinite(row.correlation) ||
        !std::isfinite(row.lag_time_ps))
      continue;
    const double weight = static_cast<double>(row.samples);
    weight_sum += weight;
    weighted_mean += weight * row.correlation;
  }
  if (!(weight_sum > 0.0))
    return std::numeric_limits<double>::quiet_NaN();
  weighted_mean /= weight_sum;

  double residual = 0.0;
  double total = 0.0;
  for (const auto &row : rows) {
    if (row.samples == 0 || !std::isfinite(row.correlation) ||
        !std::isfinite(row.lag_time_ps))
      continue;
    const double weight = static_cast<double>(row.samples);
    const double decay = std::exp(-row.lag_time_ps / fit.tau_ps);
    const double model =
        fit.raw_plateau + (1.0 - fit.raw_plateau) * decay;
    const double delta = row.correlation - model;
    const double centered = row.correlation - weighted_mean;
    residual += weight * delta * delta;
    total += weight * centered * centered;
  }
  if (!(total > kEpsilon))
    return residual <= kEpsilon ? 1.0 : 0.0;
  return 1.0 - residual / total;
}

std::vector<TitanS2Analyzer::OrderParameterRow>
TitanS2Analyzer::compute_order_parameters(TitanExecutor &executor) const {
  std::vector<OrderParameterRow> results(plans_.size());
  const std::size_t maxlag = resolve_maxlag();
  for (std::size_t i = 0; i < plans_.size(); ++i) {
    const auto correlations = titan_s2_kernel::compute_correlation(
        vectors_[i], time_ps_, maxlag, executor);
    const auto selected_rows = fit_rows(correlations);
    const auto fit = titan_s2_kernel::fit_single_exp_plateau(selected_rows);
    double fit_window_start = std::numeric_limits<double>::quiet_NaN();
    double fit_window_end = std::numeric_limits<double>::quiet_NaN();
    std::uint64_t samples = 0;
    for (const auto &row : selected_rows) {
      if (row.samples == 0 || !std::isfinite(row.lag_time_ps))
        continue;
      if (!std::isfinite(fit_window_start))
        fit_window_start = row.lag_time_ps;
      fit_window_end = row.lag_time_ps;
      samples += row.samples;
    }
    const bool model_fit = fit.ok && !fit.used_tail_fallback;
    const double tau_ps =
        model_fit ? fit.tau_ps : std::numeric_limits<double>::quiet_NaN();
    const double weighted_residual =
        model_fit ? fit.weighted_residual
                  : std::numeric_limits<double>::quiet_NaN();
    const double fit_r2 =
        model_fit ? weighted_fit_r2(selected_rows, fit)
                  : std::numeric_limits<double>::quiet_NaN();
    results[i] = {i,
                  fit.s2,
                  fit.raw_plateau,
                  tau_ps,
                  weighted_residual,
                  fit_r2,
                  fit_window_start,
                  fit_window_end,
                  samples,
                  fit.ok,
                  fit.used_tail_fallback,
                  correlations.size(),
                  selected_rows.size()};
  }
  return results;
}

std::string TitanS2Analyzer::residue_label(const VectorPlan &plan) const {
  return plan.chain + ":" + std::to_string(plan.resnr) + ":" + plan.resname;
}

std::string TitanS2Analyzer::aggregate_key(const VectorPlan &plan) const {
  return residue_label(plan) + ":" + plan.kind;
}

void TitanS2Analyzer::export_order_csv(
    const std::string &filename,
    const std::vector<OrderParameterRow> &rows) const {
  std::ofstream out(filename);
  out << "residue,chain,resnr,resname,vector_type,plateau_C_raw,S2_clipped,"
         "tau_ps,fit_R2,fit_window_start_ps,fit_window_end_ps,samples,"
         "fit_status,weighted_residual,correlation_rows,fit_rows\n";
  out << std::fixed << std::setprecision(8);
  for (const auto &row : rows) {
    const VectorPlan &plan = plans_[row.plan_index];
    out << residue_label(plan) << ',' << plan.chain << ',' << plan.resnr << ','
        << plan.resname << ',' << plan.kind << ',' << row.raw_plateau << ','
        << row.s2 << ',' << row.tau_ps << ',' << row.fit_r2 << ','
        << row.fit_window_start_ps << ',' << row.fit_window_end_ps << ','
        << row.samples << ','
        << (row.fit_ok ? (row.used_tail_fallback ? "ok_tail_fallback"
                                                : "ok_single_exp")
                       : "failed")
        << ',' << row.weighted_residual << ',' << row.correlation_rows << ','
        << row.fit_rows << '\n';
  }
}

void TitanS2Analyzer::export_correlation_dat(const std::string &filename,
                                             TitanExecutor &executor) const {
  std::ofstream out(filename);
  out << "residue chain resnr resname vector_type lag_index lag_time_ps "
         "correlation samples\n";
  out << std::fixed << std::setprecision(8);
  const std::size_t maxlag = resolve_maxlag();
  for (std::size_t i = 0; i < plans_.size(); ++i) {
    const auto rows = titan_s2_kernel::compute_correlation(
        vectors_[i], time_ps_, maxlag, executor);
    for (const auto &row : rows)
      out << residue_label(plans_[i]) << ' ' << plans_[i].chain << ' '
          << plans_[i].resnr << ' ' << plans_[i].resname << ' '
          << plans_[i].kind << ' ' << row.lag_index << ' ' << row.lag_time_ps
          << ' ' << row.correlation << ' ' << row.samples << '\n';
  }
}

void TitanS2Analyzer::export_summary(
    const std::string &filename,
    const std::vector<OrderParameterRow> &rows) const {
  std::ofstream out(filename);
  out << "module: S2 bond-vector order parameters\n";
  out << "reference_source: first_analyzed_frame\n";
  out << "reference_note: Bond vectors are evaluated after Kabsch alignment "
         "to the first analyzed mol1 frame when s2_align=rms. Interpret S2 "
         "with fit diagnostics and trajectory length.\n";
  out << "bfactor_note: s2_bfactor.pdb stores clipped S2 in the B-factor field "
         "for visualization; NH is preferred and CA-C is used only when NH "
         "is unavailable.\n";
  out << "vectors: " << args_.vectors << '\n';
  out << "residue_selection: " << args_.residues << '\n';
  out << "alignment: " << args_.align << '\n';
  out << "maxlag: " << args_.maxlag << '\n';
  out << "effective_maxlag: " << resolve_maxlag() << '\n';
  out << "fit_window_ps: " << args_.fit_window << '\n';
  out << "pbc_policy: mic_when_box_valid_else_raw\n";
  out << "frames_seen: " << frames_seen_ << '\n';
  out << "frames_used: " << frames_used_ << '\n';
  out << "raw_box_fallback_frames: " << raw_box_fallback_frames_ << '\n';
  if (raw_box_fallback_frames_ > 0)
    out << "warning: raw bond-vector fallback used for "
        << raw_box_fallback_frames_
        << " analyzed frame(s) because the periodic box was missing or invalid.\n";
  out << "missing_nh_skipped: " << missing_nh_skipped_ << '\n';
  out << "pro_nh_skipped: " << pro_nh_skipped_ << '\n';
  if (missing_nh_skipped_ > 0)
    out << "warning: skipped " << missing_nh_skipped_
        << " NH vector(s) because PRO or required N/H atoms were unavailable.\n";
  out << "vectors_planned: " << plans_.size() << '\n';
  out << "fits_ok: "
      << std::count_if(rows.begin(), rows.end(),
                       [](const OrderParameterRow &row) { return row.fit_ok; })
      << '\n';
  out << "clipped_s2_count: "
      << std::count_if(rows.begin(), rows.end(),
                       [](const OrderParameterRow &row) {
                         return row.fit_ok &&
                                std::abs(row.raw_plateau - row.s2) > kEpsilon;
                       })
      << '\n';
}

void TitanS2Analyzer::export_pdb(const std::string &filename,
                                const std::map<std::string, double> &s2_by_key,
                                bool average) const {
  const MOL &snapshot = average ? aggregate_first_snapshot_ : first_snapshot_;
  const auto &info = average ? aggregate_reference_info_ : reference_info_;
  if (!snapshot.coordinates_consistent() || snapshot.size() != info.size())
    return;
  std::map<std::string, double> residue_s2;
  std::map<std::string, bool> residue_has_nh;
  for (const auto &[key, value] : s2_by_key) {
    const auto plan_it = aggregate_plans_.find(key);
    if (plan_it == aggregate_plans_.end())
      continue;
    const VectorPlan &plan = plan_it->second;
    const std::string residue = residue_label(plan);
    if (plan.kind == "NH" || !residue_has_nh[residue]) {
      residue_s2[residue] = value;
      residue_has_nh[residue] = plan.kind == "NH";
    }
  }
  std::ofstream out(filename);
  out << "REMARK " << (average ? "average " : "")
      << "S2 mapped to B-factor; NH preferred, otherwise CA-C\n";
  for (std::size_t atom = 0; atom < info.size(); ++atom) {
    const mol_info &meta = info[atom];
    const std::string residue =
        meta.get_chain() + ":" + std::to_string(meta.get_resnr()) + ":" +
        meta.get_res_name();
    const auto value = residue_s2.find(residue);
    const double bfactor = value == residue_s2.end() ? 0.0 : value->second;
    out << std::left << std::setw(6) << "ATOM" << std::right << std::setw(5)
        << atom + 1 << ' ' << std::setw(4) << meta.get_atom() << ' '
        << std::setw(3) << meta.get_res_name() << ' '
        << (meta.get_chain().empty() ? ' ' : meta.get_chain()[0])
        << std::setw(4) << meta.get_resnr() << "    " << std::fixed
        << std::setprecision(3) << std::setw(8) << snapshot.pos.x[atom]
        << std::setw(8) << snapshot.pos.y[atom] << std::setw(8)
        << snapshot.pos.z[atom] << std::setprecision(2) << std::setw(6) << 1.0
        << std::setw(6) << bfactor << '\n';
  }
  out << "END\n";
}

std::vector<std::string> TitanS2Analyzer::plan_signature() const {
  std::vector<std::string> signature;
  signature.reserve(plans_.size());
  for (const VectorPlan &plan : plans_) {
    signature.push_back(aggregate_key(plan) + ":" +
                        std::to_string(plan.from_atom) + ":" +
                        std::to_string(plan.to_atom));
  }
  return signature;
}

bool TitanS2Analyzer::accumulate_trajectory(
    const std::vector<OrderParameterRow> &rows) {
  if (frames_used_ == 0 || rows.empty())
    return false;
  const std::vector<std::string> signature = plan_signature();
  if (!aggregate_plan_signature_.empty() &&
      aggregate_plan_signature_ != signature)
    throw std::runtime_error(
        "S2: aggregate requires topology-identical vector plans across "
        "trajectories.");
  const bool has_valid_fit =
      std::any_of(rows.begin(), rows.end(),
                  [](const OrderParameterRow &row) { return row.fit_ok; });
  if (!has_valid_fit)
    return false;
  if (aggregate_plan_signature_.empty())
    aggregate_plan_signature_ = signature;
  if (!have_aggregate_snapshot_ && have_first_snapshot_) {
    aggregate_first_snapshot_ = first_snapshot_;
    aggregate_reference_info_ = reference_info_;
    have_aggregate_snapshot_ = true;
  }
  for (const auto &row : rows) {
    if (!row.fit_ok)
      continue;
    const VectorPlan &plan = plans_[row.plan_index];
    const std::string key = aggregate_key(plan);
    AggregateResult &aggregate = aggregate_results_[key];
    aggregate.raw_plateau.sum += row.raw_plateau;
    aggregate.raw_plateau.sum_sq += row.raw_plateau * row.raw_plateau;
    ++aggregate.raw_plateau.count;
    aggregate.s2.sum += row.s2;
    aggregate.s2.sum_sq += row.s2 * row.s2;
    ++aggregate.s2.count;
    aggregate_plans_.try_emplace(key, plan);
  }
  ++aggregate_trajectories_;
  return true;
}

void TitanS2Analyzer::export_average(const std::string &base_dir,
                                     const TitanS2OutputFiles &files) const {
  auto mean_sem = [](const AggregateCell &cell) {
    if (cell.count == 0)
      return std::pair{std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::quiet_NaN()};
    const double mean = cell.sum / static_cast<double>(cell.count);
    if (cell.count == 1)
      return std::pair{mean, std::numeric_limits<double>::quiet_NaN()};
    const double centered_sum =
        std::max(0.0, cell.sum_sq - cell.sum * cell.sum /
                                       static_cast<double>(cell.count));
    const double sem =
        std::sqrt(centered_sum /
                  (static_cast<double>(cell.count) *
                   static_cast<double>(cell.count - 1)));
    return std::pair{mean, sem};
  };
  std::map<std::string, double> averages;
  {
    std::ofstream out(join_output_path(base_dir, files.average_csv));
    out << "residue,chain,resnr,resname,vector_type,mean_plateau_C_raw,"
           "sem_plateau_C_raw,mean_S2_clipped,sem_S2_clipped,"
           "valid_replicates\n";
    out << std::fixed << std::setprecision(8);
    for (const auto &[key, aggregate] : aggregate_results_) {
      const auto plan = aggregate_plans_.find(key);
      if (plan == aggregate_plans_.end())
        continue;
      const auto [raw_mean, raw_sem] = mean_sem(aggregate.raw_plateau);
      const auto [s2_mean, s2_sem] = mean_sem(aggregate.s2);
      averages[key] = s2_mean;
      out << residue_label(plan->second) << ',' << plan->second.chain << ','
          << plan->second.resnr << ',' << plan->second.resname << ','
          << plan->second.kind << ',' << raw_mean << ',' << raw_sem << ','
          << s2_mean << ',' << s2_sem << ',' << aggregate.s2.count << '\n';
    }
  }
  {
    std::ofstream out(join_output_path(base_dir, files.average_summary_dat));
    out << "module: S2 replicate aggregate\n";
    out << "reference_note: Aggregate values are per-trajectory S2 means with "
           "SEM; trajectory correlation files remain separate.\n";
    out << "bfactor_note: Average PDB B-factor stores replicate mean S2 for NH "
           "when available, otherwise CA-C.\n";
    out << "replicates: " << aggregate_trajectories_ << '\n';
  }
  if (args_.average_pdb && have_aggregate_snapshot_)
    export_pdb(join_output_path(base_dir, files.average_pdb), averages, true);
}

void TitanS2Analyzer::finalize_trajectory(
    const std::string &base_dir, const TitanS2OutputFiles &files,
    bool export_average_files, TitanExecutor &executor, std::fstream &log) {
  if (!enabled_)
    return;
  const auto rows = compute_order_parameters(executor);
  export_order_csv(join_output_path(base_dir, files.order_csv), rows);
  export_correlation_dat(join_output_path(base_dir, files.correlation_dat),
                         executor);
  export_summary(join_output_path(base_dir, files.summary_dat), rows);
  std::map<std::string, double> trajectory_s2;
  for (const auto &row : rows) {
    if (!row.fit_ok)
      continue;
    const VectorPlan &plan = plans_[row.plan_index];
    const std::string key = aggregate_key(plan);
    trajectory_s2[key] = row.s2;
    aggregate_plans_.try_emplace(key, plan);
  }
  if (args_.pdb && have_first_snapshot_)
    export_pdb(join_output_path(base_dir, files.pdb), trajectory_s2, false);
  const bool accumulated = accumulate_trajectory(rows);
  if (export_average_files && accumulated)
    export_average(base_dir, files);
  if (log.is_open()) {
    log << "S2 analyzed frames: " << frames_used_ << '\n';
    if (raw_box_fallback_frames_ > 0)
      log << "S2 warning: raw bond-vector fallback frames: "
          << raw_box_fallback_frames_ << '\n';
    if (missing_nh_skipped_ > 0)
      log << "S2 warning: skipped NH vectors: " << missing_nh_skipped_ << '\n';
  }
  reset_trajectory_state();
}

void TitanS2Analyzer::finalize_all(const std::string &base_dir,
                                  const TitanS2OutputFiles &files) const {
  if (aggregate_trajectories_ == 0)
    return;
  export_average(base_dir, files);
}

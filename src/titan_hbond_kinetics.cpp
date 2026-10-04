#include "titan_hbond_kinetics.h"

#include "info.h"
#include "titan_executor.h"
#include "titan_interaction_primitives.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace {

using titan_hbond_kinetics::KeyAggregate;
using titan_hbond_kinetics::StreamingKinetics;

constexpr std::size_t kScopeCount = 3;
constexpr double kEpsilon = 1.0e-12;

std::size_t intersection_count(const std::vector<int> &lhs,
                               const std::vector<int> &rhs) {
  std::size_t count = 0;
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] < rhs[j]) {
      ++i;
    } else if (rhs[j] < lhs[i]) {
      ++j;
    } else {
      ++count;
      ++i;
      ++j;
    }
  }
  return count;
}

template <typename Event> void sort_unique_events(std::vector<Event> &events) {
  std::sort(events.begin(), events.end());
  events.erase(std::unique(events.begin(), events.end()), events.end());
}

double safe_divide(double num, double den) {
  return den > 0.0 ? num / den : 0.0;
}

double integrated_continuous_frames(const StreamingKinetics &kinetics) {
  double total = 0.0;
  const auto &origins = kinetics.origin_counts();
  const auto &hits = kinetics.continuous_hits();
  const std::size_t limit = std::min(origins.size(), hits.size());
  for (std::size_t lag = 0; lag < limit; ++lag) {
    if (origins[lag] == 0)
      continue;
    total += static_cast<double>(hits[lag]) /
             static_cast<double>(origins[lag]);
  }
  return total;
}

std::pair<double, std::uint64_t>
aggregate_lifetimes(const std::vector<KeyAggregate> &stats) {
  double total_frames = 0.0;
  std::uint64_t total_events = 0;
  for (const auto &stat : stats) {
    total_frames += static_cast<double>(stat.total_lifetime_frames);
    total_events += stat.event_count;
  }
  return {total_frames, total_events};
}

} // namespace

namespace titan_hbond_kinetics {

StreamingKinetics::StreamingKinetics(std::size_t maxlag) { reset(maxlag); }

void StreamingKinetics::reset(std::size_t maxlag) {
  maxlag_ = maxlag;
  frame_count_ = 0;
  origin_counts_.assign(maxlag_ + 1, 0);
  intermittent_hits_.assign(maxlag_ + 1, 0);
  continuous_hits_.assign(maxlag_ + 1, 0);
  key_stats_.clear();
  current_run_lengths_.clear();
  previous_active_ids_.clear();
  history_.clear();
  lifetime_histogram_.assign(1, 0);
}

void StreamingKinetics::ensure_id(std::size_t id) {
  if (key_stats_.size() <= id) {
    key_stats_.resize(id + 1);
    current_run_lengths_.resize(id + 1, 0);
  }
}

void StreamingKinetics::finalize_run(int id) {
  if (id < 0)
    return;
  const std::size_t idx = static_cast<std::size_t>(id);
  if (idx >= current_run_lengths_.size())
    return;
  const std::uint64_t run = current_run_lengths_[idx];
  if (run == 0)
    return;
  const auto lifetime = static_cast<std::size_t>(run);
  if (lifetime >= lifetime_histogram_.size())
    lifetime_histogram_.resize(lifetime + 1, 0);
  ++lifetime_histogram_[lifetime];
  key_stats_[idx].total_lifetime_frames += run;
  key_stats_[idx].max_lifetime_frames =
      std::max(key_stats_[idx].max_lifetime_frames, run);
  current_run_lengths_[idx] = 0;
}

void StreamingKinetics::process_active_ids(std::vector<int> active_ids) {
  std::sort(active_ids.begin(), active_ids.end());
  active_ids.erase(std::unique(active_ids.begin(), active_ids.end()),
                   active_ids.end());

  for (int id : active_ids) {
    if (id >= 0)
      ensure_id(static_cast<std::size_t>(id));
  }

  ++frame_count_;
  origin_counts_[0] += active_ids.size();
  intermittent_hits_[0] += active_ids.size();

  const std::size_t usable_history = std::min(maxlag_, history_.size());
  for (std::size_t lag = 1; lag <= usable_history; ++lag) {
    const auto &origin = history_[history_.size() - lag];
    origin_counts_[lag] += origin.size();
    intermittent_hits_[lag] += intersection_count(origin, active_ids);
  }

  std::size_t prev_pos = 0;
  std::size_t cur_pos = 0;
  while (prev_pos < previous_active_ids_.size() && cur_pos < active_ids.size()) {
    if (previous_active_ids_[prev_pos] < active_ids[cur_pos]) {
      finalize_run(previous_active_ids_[prev_pos]);
      ++prev_pos;
    } else if (active_ids[cur_pos] < previous_active_ids_[prev_pos]) {
      ++cur_pos;
    } else {
      ++prev_pos;
      ++cur_pos;
    }
  }
  while (prev_pos < previous_active_ids_.size()) {
    finalize_run(previous_active_ids_[prev_pos]);
    ++prev_pos;
  }

  for (int id : active_ids) {
    const std::size_t idx = static_cast<std::size_t>(id);
    if (current_run_lengths_[idx] == 0)
      key_stats_[idx].event_count++;
    ++current_run_lengths_[idx];
    key_stats_[idx].total_active_frames++;
    const std::size_t max_lag =
        std::min<std::size_t>(maxlag_, current_run_lengths_[idx] - 1);
    for (std::size_t lag = 0; lag <= max_lag; ++lag)
      continuous_hits_[lag]++;
  }

  history_.push_back(active_ids);
  if (history_.size() > maxlag_)
    history_.pop_front();
  previous_active_ids_ = std::move(active_ids);
}

void StreamingKinetics::finalize() {
  for (int id : previous_active_ids_)
    finalize_run(id);
  previous_active_ids_.clear();
  history_.clear();
}

} // namespace titan_hbond_kinetics

bool TitanHBondKineticsAnalyzer::HBondEvent::operator<(
    const HBondEvent &rhs) const noexcept {
  return std::tie(donor_mol, donor_idx, hydrogen_idx, acceptor_mol,
                  acceptor_idx) <
         std::tie(rhs.donor_mol, rhs.donor_idx, rhs.hydrogen_idx,
                  rhs.acceptor_mol, rhs.acceptor_idx);
}

bool TitanHBondKineticsAnalyzer::HBondEvent::operator==(
    const HBondEvent &rhs) const noexcept {
  return donor_mol == rhs.donor_mol && donor_idx == rhs.donor_idx &&
         hydrogen_idx == rhs.hydrogen_idx &&
         acceptor_mol == rhs.acceptor_mol &&
         acceptor_idx == rhs.acceptor_idx;
}

bool TitanHBondKineticsAnalyzer::SaltEvent::operator<(
    const SaltEvent &rhs) const noexcept {
  return std::tie(acidic_mol, acidic_idx, basic_mol, basic_idx) <
         std::tie(rhs.acidic_mol, rhs.acidic_idx, rhs.basic_mol,
                  rhs.basic_idx);
}

bool TitanHBondKineticsAnalyzer::SaltEvent::operator==(
    const SaltEvent &rhs) const noexcept {
  return acidic_mol == rhs.acidic_mol && acidic_idx == rhs.acidic_idx &&
         basic_mol == rhs.basic_mol && basic_idx == rhs.basic_idx;
}

bool TitanHBondKineticsAnalyzer::WaterEvent::operator<(
    const WaterEvent &rhs) const noexcept {
  return std::tie(water_index, atom_a_mol, atom_a_idx, atom_b_mol, atom_b_idx) <
         std::tie(rhs.water_index, rhs.atom_a_mol, rhs.atom_a_idx,
                  rhs.atom_b_mol, rhs.atom_b_idx);
}

bool TitanHBondKineticsAnalyzer::WaterEvent::operator==(
    const WaterEvent &rhs) const noexcept {
  return water_index == rhs.water_index && atom_a_mol == rhs.atom_a_mol &&
         atom_a_idx == rhs.atom_a_idx && atom_b_mol == rhs.atom_b_mol &&
         atom_b_idx == rhs.atom_b_idx;
}

void TitanHBondKineticsAnalyzer::configure(const HBondKineticsArgs &args) {
  requested_ = args.flag_hbk;
  maxlag_ = args.hbk_maxlag > 0 ? static_cast<std::size_t>(args.hbk_maxlag) : 0;
  hbond_cutoff_ang_ = args.hbk_hbond_cutoff;
  hbond_cutoff_sq_ = hbond_cutoff_ang_ * hbond_cutoff_ang_;
  angle_cutoff_deg_ = args.hbk_angle_deg;
  angle_cos_cutoff_ =
      std::cos(angle_cutoff_deg_ * 3.14159265358979323846 / 180.0);
  salt_cutoff_ang_ = args.hbk_salt_cutoff;
  salt_cutoff_sq_ = salt_cutoff_ang_ * salt_cutoff_ang_;
  water_cutoff_ang_ = args.hbk_water_cutoff;
  water_cutoff_sq_ = water_cutoff_ang_ * water_cutoff_ang_;
  configure_requested_kinds(args);
  enabled_ = requested_;
}

void TitanHBondKineticsAnalyzer::configure_requested_kinds(
    const HBondKineticsArgs &args) {
  request_hbond_ = false;
  request_salt_ = false;
  request_water_ = false;
  for (const std::string &kind : args.hbk_kinds) {
    if (kind == "all") {
      request_hbond_ = true;
      request_salt_ = true;
      request_water_ = true;
      break;
    }
    if (kind == "hbond")
      request_hbond_ = true;
    else if (kind == "salt")
      request_salt_ = true;
    else if (kind == "water")
      request_water_ = true;
  }
  if (!request_hbond_ && !request_salt_ && !request_water_) {
    request_hbond_ = true;
    request_salt_ = true;
    request_water_ = true;
  }
}

void TitanHBondKineticsAnalyzer::reset_trajectory_state() {
  frames_seen_ = 0;
  water_missing_frames_ = 0;
  warned_missing_water_ = false;
  warned_missing_mol2_ = false;
  previous_time_ps_ = 0.0;
  has_previous_time_ = false;
  observed_dt_sum_ps_ = 0.0;
  observed_dt_count_ = 0;

  for (auto &series : hbond_series_)
    series.reset(maxlag_);
  for (auto &series : salt_series_)
    series.reset(maxlag_);
  for (auto &series : water_series_)
    series.reset(maxlag_);
}

void TitanHBondKineticsAnalyzer::prepare_molecule_cache(const Multi_MOL &mol,
                                                        const std::string &label,
                                                        MoleculeCache &cache) {
  cache = {};
  cache.label = label;
  if (mol.info.empty() || mol.m_mol.empty())
    return;

  const std::size_t atom_count =
      std::min<std::size_t>(mol.info.size(), mol.m_mol.front().size());
  if (atom_count == 0)
    return;

  cache.available = true;
  cache.atoms.reserve(atom_count);
  cache.donors.reserve(atom_count / 8 + 1);
  cache.acceptors.reserve(atom_count / 8 + 1);
  cache.acidic_atoms.reserve(atom_count / 16 + 1);
  cache.basic_atoms.reserve(atom_count / 16 + 1);
  cache.heavy_atoms.reserve(atom_count);

  const MOL &frame0 = mol.m_mol.front();
  for (std::size_t atom_idx = 0; atom_idx < atom_count; ++atom_idx) {
    const auto &atom_info = mol.info[atom_idx];
    AtomLabel label_info;
    label_info.atom_index = atom_idx;
    label_info.chain_id = atom_info.get_chainnum();
    label_info.resnr = atom_info.get_resnr();
    label_info.chain = atom_info.get_chain();
    label_info.res_name = atom_info.get_original_res_name();
    label_info.atom_name = atom_info.get_atom();
    cache.atoms.push_back(std::move(label_info));

    if (!titan_interaction::is_hydrogen_atom(atom_info))
      cache.heavy_atoms.push_back(atom_idx);

    const std::string atom_name = atom_info.get_atom();
    const std::string res_name = atom_info.get_res_name();
    if (titan_interaction::is_hbond_acceptor(atom_name, res_name))
      cache.acceptors.push_back(atom_idx);
    if (titan_interaction::is_acidic(atom_name, res_name))
      cache.acidic_atoms.push_back(atom_idx);
    if (titan_interaction::is_basic(atom_name, res_name))
      cache.basic_atoms.push_back(atom_idx);
    if (titan_interaction::is_hbond_donor(atom_name, res_name)) {
      const auto span =
          titan_interaction::find_residue_span_by_atom(mol.info, atom_idx);
      const std::size_t hydrogen_idx = titan_interaction::find_bonded_hydrogen(
          frame0, mol.info, atom_idx, span.first, span.second);
      if (hydrogen_idx != SIZE_MAX && hydrogen_idx < atom_count) {
        cache.donors.push_back(
            {atom_idx, static_cast<std::size_t>(hydrogen_idx)});
      }
    }
  }
}

void TitanHBondKineticsAnalyzer::configure_scope_states(const arguments &args,
                                                        std::fstream &log) {
  hbond_series_.resize(kScopeCount);
  salt_series_.resize(kScopeCount);
  water_series_.resize(kScopeCount);

  allow_inter_ = !args.flag_nocal && mol2_cache_.available;
  allow_intra_mol1_ = mol1_cache_.available;
  allow_intra_mol2_ = !args.flag_nocal && mol2_cache_.available;

  if (args.hbk.hbk_domain == "inter") {
    allow_intra_mol1_ = false;
    allow_intra_mol2_ = false;
  } else if (args.hbk.hbk_domain == "intra") {
    allow_inter_ = false;
  }

  auto setup_scope_vector = [&](auto &series_vec, bool requested_kind,
                                bool require_water) {
    for (std::size_t idx = 0; idx < series_vec.size(); ++idx) {
      const ScopeKind scope = static_cast<ScopeKind>(idx);
      auto &series = series_vec[idx];
      series.reset(maxlag_);
      series.scope_label = scope_label(scope);
      series.requested = requested_kind;
      series.enabled = false;
      if (!requested_kind) {
        series.status = "not_requested";
        continue;
      }

      bool scope_allowed = false;
      std::string blocked_reason;
      if (scope == ScopeKind::Inter) {
        scope_allowed = allow_inter_;
        blocked_reason =
            args.flag_nocal ? "disabled_by_-nocal" : "missing_mol2";
      } else if (scope == ScopeKind::IntraMol1) {
        scope_allowed = allow_intra_mol1_;
        blocked_reason = "missing_mol1";
      } else {
        scope_allowed = allow_intra_mol2_;
        blocked_reason =
            args.flag_nocal ? "disabled_by_-nocal" : "missing_mol2";
      }

      if (!scope_allowed) {
        series.status = blocked_reason;
        continue;
      }
      if (require_water && !water_stream_available_) {
        series.status = "skipped_no_water_stream";
        continue;
      }

      series.enabled = true;
      series.status = "active";
    }
  };

  setup_scope_vector(hbond_series_, request_hbond_, false);
  setup_scope_vector(salt_series_, request_salt_, false);
  setup_scope_vector(water_series_, request_water_, true);

  if ((request_hbond_ || request_salt_ || request_water_) && !allow_inter_ &&
      args.hbk.hbk_domain != "intra" && !warned_missing_mol2_) {
    if (args.flag_nocal) {
      TitanUI::print_warning(
          "HBond kinetics: inter and mol2-intra scopes are disabled by -nocal.");
    } else if (!mol2_cache_.available) {
      TitanUI::print_warning(
          "HBond kinetics: mol2 was unavailable; inter and intra_mol2 scopes "
          "will be skipped.");
    }
    warned_missing_mol2_ = true;
  }
  if (request_water_ && !water_stream_available_) {
    TitanUI::print_warning(
        "HBond kinetics: water stream was not available. Water-bridge "
        "kinetics will be exported as header-only files.");
  }

  if (log.is_open()) {
    std::string kind_labels;
    if (request_hbond_)
      kind_labels += (kind_labels.empty() ? "" : ",") + std::string("hbond");
    if (request_salt_)
      kind_labels += (kind_labels.empty() ? "" : ",") + std::string("salt");
    if (request_water_)
      kind_labels += (kind_labels.empty() ? "" : ",") + std::string("water");
    TitanUI::log_kv(log, "HBK Requested Kinds", kind_labels);
    TitanUI::log_kv(log, "HBK Domain", args.hbk.hbk_domain);
    TitanUI::log_kv(log, "HBK Max Lag", std::to_string(maxlag_) + " frames");
    TitanUI::log_kv(log, "HBK Hbond Cutoff",
                    std::to_string(hbond_cutoff_ang_) + " A");
    TitanUI::log_kv(log, "HBK Hbond Angle",
                    std::to_string(angle_cutoff_deg_) + " deg");
    TitanUI::log_kv(log, "HBK Salt Cutoff",
                    std::to_string(salt_cutoff_ang_) + " A");
    TitanUI::log_kv(log, "HBK Water Cutoff",
                    std::to_string(water_cutoff_ang_) + " A");
    TitanUI::log_kv(log, "HBK Water Stream",
                    water_stream_available_ ? "available" : "missing");
  }
}

void TitanHBondKineticsAnalyzer::init(const Multi_MOL &mol1,
                                      const Multi_MOL *mol2,
                                      bool water_stream_available,
                                      const arguments &args,
                                      std::fstream &log) {
  fallback_dt_ps_ = args.dt > 0.0 ? args.dt : 0.0;
  water_stream_available_ = water_stream_available;
  prepare_molecule_cache(mol1, "mol1", mol1_cache_);
  if (mol2 != nullptr)
    prepare_molecule_cache(*mol2, "mol2", mol2_cache_);
  else
    mol2_cache_ = {};

  initialized_ = true;
  enabled_ = requested_;
  reset_trajectory_state();
  configure_scope_states(args, log);
}

void TitanHBondKineticsAnalyzer::update_frame_timing(double time_ps) {
  if (has_previous_time_ && time_ps > previous_time_ps_ + kEpsilon) {
    observed_dt_sum_ps_ += (time_ps - previous_time_ps_);
    ++observed_dt_count_;
  }
  previous_time_ps_ = time_ps;
  has_previous_time_ = true;
}

const char *TitanHBondKineticsAnalyzer::scope_label(ScopeKind scope) const {
  switch (scope) {
  case ScopeKind::Inter:
    return "inter";
  case ScopeKind::IntraMol1:
    return "intra_mol1";
  case ScopeKind::IntraMol2:
    return "intra_mol2";
  }
  return "unknown";
}

std::size_t TitanHBondKineticsAnalyzer::scope_index(ScopeKind scope) const {
  return static_cast<std::size_t>(scope);
}

double TitanHBondKineticsAnalyzer::frame_dt_ps() const {
  if (observed_dt_count_ > 0)
    return observed_dt_sum_ps_ / static_cast<double>(observed_dt_count_);
  return fallback_dt_ps_;
}

const TitanHBondKineticsAnalyzer::MoleculeCache &
TitanHBondKineticsAnalyzer::molecule_cache(int mol_id) const {
  return mol_id == 0 ? mol1_cache_ : mol2_cache_;
}

void TitanHBondKineticsAnalyzer::detect_hbonds_between(
    const MoleculeCache &donor_cache, const MOL &donor_frame, int donor_mol,
    const MoleculeCache &acceptor_cache, const MOL &acceptor_frame,
    int acceptor_mol, bool exclude_same_residue, TitanExecutor &executor,
    std::vector<HBondEvent> &out_events) const {
  out_events.clear();
  if (donor_cache.donors.empty() || acceptor_cache.acceptors.empty())
    return;

  const std::size_t worker_slots =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<std::vector<HBondEvent>> worker_events(worker_slots);
  executor.parallel_for(
      donor_cache.donors.size(), worker_slots, 16,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &local = worker_events[worker_id];
        const double *xs = acceptor_frame.pos.x.data();
        const double *ys = acceptor_frame.pos.y.data();
        const double *zs = acceptor_frame.pos.z.data();
        const auto &targets = acceptor_cache.acceptors;

        for (std::size_t donor_pos = begin; donor_pos < end; ++donor_pos) {
          const DonorSite &donor_site = donor_cache.donors[donor_pos];
          if (donor_site.donor_index >= donor_frame.size() ||
              donor_site.hydrogen_index >= donor_frame.size() ||
              donor_site.donor_index >= donor_cache.atoms.size()) {
            continue;
          }

          const double px = donor_frame.pos.x[donor_site.donor_index];
          const double py = donor_frame.pos.y[donor_site.donor_index];
          const double pz = donor_frame.pos.z[donor_site.donor_index];

          std::size_t target_pos = 0;
          for (; target_pos + 3 < targets.size(); target_pos += 4) {
            const std::int64_t gather_idx[4] = {
                static_cast<std::int64_t>(targets[target_pos]),
                static_cast<std::int64_t>(targets[target_pos + 1]),
                static_cast<std::int64_t>(targets[target_pos + 2]),
                static_cast<std::int64_t>(targets[target_pos + 3])};
            unsigned mask = titan_simd_dispatch::point_vs_gathered4_cutoff_mask(
                px, py, pz, xs, ys, zs, gather_idx, hbond_cutoff_sq_, 0.0, false);
            while (mask != 0U) {
              const unsigned lane = static_cast<unsigned>(std::countr_zero(mask));
              const std::size_t acceptor_idx =
                  targets[target_pos + static_cast<std::size_t>(lane)];
              mask &= (mask - 1U);
              if (acceptor_idx >= acceptor_frame.size() ||
                  acceptor_idx >= acceptor_cache.atoms.size())
                continue;
              if (donor_mol == acceptor_mol &&
                  acceptor_idx == donor_site.donor_index)
                continue;
              if (exclude_same_residue && donor_mol == acceptor_mol &&
                  donor_cache.atoms[donor_site.donor_index].chain_id ==
                      acceptor_cache.atoms[acceptor_idx].chain_id &&
                  donor_cache.atoms[donor_site.donor_index].resnr ==
                      acceptor_cache.atoms[acceptor_idx].resnr) {
                continue;
              }

              double angle_deg = 0.0;
              if (!titan_interaction::try_calculate_angle_deg(
                      donor_frame, donor_site.donor_index,
                      donor_site.hydrogen_index, acceptor_frame, acceptor_idx,
                      angle_cos_cutoff_, angle_deg)) {
                continue;
              }
              local.push_back({donor_mol,
                               static_cast<int>(donor_site.donor_index),
                               static_cast<int>(donor_site.hydrogen_index),
                               acceptor_mol,
                               static_cast<int>(acceptor_idx)});
            }
          }

          for (; target_pos < targets.size(); ++target_pos) {
            const std::size_t acceptor_idx = targets[target_pos];
            if (acceptor_idx >= acceptor_frame.size() ||
                acceptor_idx >= acceptor_cache.atoms.size())
              continue;
            if (donor_mol == acceptor_mol &&
                acceptor_idx == donor_site.donor_index)
              continue;
            if (exclude_same_residue && donor_mol == acceptor_mol &&
                donor_cache.atoms[donor_site.donor_index].chain_id ==
                    acceptor_cache.atoms[acceptor_idx].chain_id &&
                donor_cache.atoms[donor_site.donor_index].resnr ==
                    acceptor_cache.atoms[acceptor_idx].resnr) {
              continue;
            }

            const double dx = px - acceptor_frame.pos.x[acceptor_idx];
            const double dy = py - acceptor_frame.pos.y[acceptor_idx];
            const double dz = pz - acceptor_frame.pos.z[acceptor_idx];
            const double dist_sq = dx * dx + dy * dy + dz * dz;
            if (!(dist_sq < hbond_cutoff_sq_))
              continue;

            double angle_deg = 0.0;
            if (!titan_interaction::try_calculate_angle_deg(
                    donor_frame, donor_site.donor_index,
                    donor_site.hydrogen_index, acceptor_frame, acceptor_idx,
                    angle_cos_cutoff_, angle_deg)) {
              continue;
            }
            local.push_back({donor_mol,
                             static_cast<int>(donor_site.donor_index),
                             static_cast<int>(donor_site.hydrogen_index),
                             acceptor_mol, static_cast<int>(acceptor_idx)});
          }
        }
      });

  for (auto &local : worker_events) {
    out_events.insert(out_events.end(), local.begin(), local.end());
  }
  sort_unique_events(out_events);
}

void TitanHBondKineticsAnalyzer::detect_salts_between(
    const MoleculeCache &acid_cache, const MOL &acid_frame, int acid_mol,
    const MoleculeCache &base_cache, const MOL &base_frame, int base_mol,
    bool exclude_same_residue, TitanExecutor &executor,
    std::vector<SaltEvent> &out_events) const {
  out_events.clear();
  if (acid_cache.acidic_atoms.empty() || base_cache.basic_atoms.empty())
    return;

  const std::size_t worker_slots =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<std::vector<SaltEvent>> worker_events(worker_slots);
  executor.parallel_for(
      acid_cache.acidic_atoms.size(), worker_slots, 16,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &local = worker_events[worker_id];
        const double *xs = base_frame.pos.x.data();
        const double *ys = base_frame.pos.y.data();
        const double *zs = base_frame.pos.z.data();
        const auto &targets = base_cache.basic_atoms;

        for (std::size_t acid_pos = begin; acid_pos < end; ++acid_pos) {
          const std::size_t acid_idx = acid_cache.acidic_atoms[acid_pos];
          if (acid_idx >= acid_frame.size() || acid_idx >= acid_cache.atoms.size())
            continue;

          const double px = acid_frame.pos.x[acid_idx];
          const double py = acid_frame.pos.y[acid_idx];
          const double pz = acid_frame.pos.z[acid_idx];

          std::size_t target_pos = 0;
          for (; target_pos + 3 < targets.size(); target_pos += 4) {
            const std::int64_t gather_idx[4] = {
                static_cast<std::int64_t>(targets[target_pos]),
                static_cast<std::int64_t>(targets[target_pos + 1]),
                static_cast<std::int64_t>(targets[target_pos + 2]),
                static_cast<std::int64_t>(targets[target_pos + 3])};
            unsigned mask = titan_simd_dispatch::point_vs_gathered4_cutoff_mask(
                px, py, pz, xs, ys, zs, gather_idx, salt_cutoff_sq_, 0.0, false);
            while (mask != 0U) {
              const unsigned lane = static_cast<unsigned>(std::countr_zero(mask));
              const std::size_t base_idx =
                  targets[target_pos + static_cast<std::size_t>(lane)];
              mask &= (mask - 1U);
              if (base_idx >= base_frame.size() || base_idx >= base_cache.atoms.size())
                continue;
              if (acid_mol == base_mol && base_idx == acid_idx)
                continue;
              if (exclude_same_residue && acid_mol == base_mol &&
                  acid_cache.atoms[acid_idx].chain_id ==
                      base_cache.atoms[base_idx].chain_id &&
                  acid_cache.atoms[acid_idx].resnr ==
                      base_cache.atoms[base_idx].resnr) {
                continue;
              }
              local.push_back({acid_mol, static_cast<int>(acid_idx), base_mol,
                               static_cast<int>(base_idx)});
            }
          }

          for (; target_pos < targets.size(); ++target_pos) {
            const std::size_t base_idx = targets[target_pos];
            if (base_idx >= base_frame.size() || base_idx >= base_cache.atoms.size())
              continue;
            if (acid_mol == base_mol && base_idx == acid_idx)
              continue;
            if (exclude_same_residue && acid_mol == base_mol &&
                acid_cache.atoms[acid_idx].chain_id ==
                    base_cache.atoms[base_idx].chain_id &&
                acid_cache.atoms[acid_idx].resnr ==
                    base_cache.atoms[base_idx].resnr) {
              continue;
            }
            const double dx = px - base_frame.pos.x[base_idx];
            const double dy = py - base_frame.pos.y[base_idx];
            const double dz = pz - base_frame.pos.z[base_idx];
            const double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq >= salt_cutoff_sq_)
              continue;
            local.push_back({acid_mol, static_cast<int>(acid_idx), base_mol,
                             static_cast<int>(base_idx)});
          }
        }
      });

  for (auto &local : worker_events)
    out_events.insert(out_events.end(), local.begin(), local.end());
  sort_unique_events(out_events);
}

void TitanHBondKineticsAnalyzer::collect_nearby_heavy_atoms(
    const MOL &water_frame, std::size_t oxygen_idx, const MoleculeCache &cache,
    const MOL &frame, double cutoff_sq, std::vector<std::size_t> &hits) const {
  hits.clear();
  if (oxygen_idx >= water_frame.size() || cache.heavy_atoms.empty())
    return;

  const double px = water_frame.pos.x[oxygen_idx];
  const double py = water_frame.pos.y[oxygen_idx];
  const double pz = water_frame.pos.z[oxygen_idx];
  const double *xs = frame.pos.x.data();
  const double *ys = frame.pos.y.data();
  const double *zs = frame.pos.z.data();

  std::size_t pos = 0;
  for (; pos + 3 < cache.heavy_atoms.size(); pos += 4) {
    const std::int64_t gather_idx[4] = {
        static_cast<std::int64_t>(cache.heavy_atoms[pos]),
        static_cast<std::int64_t>(cache.heavy_atoms[pos + 1]),
        static_cast<std::int64_t>(cache.heavy_atoms[pos + 2]),
        static_cast<std::int64_t>(cache.heavy_atoms[pos + 3])};
    unsigned mask = titan_simd_dispatch::point_vs_gathered4_cutoff_mask(
        px, py, pz, xs, ys, zs, gather_idx, cutoff_sq, 0.0, true);
    while (mask != 0U) {
      const unsigned lane = static_cast<unsigned>(std::countr_zero(mask));
      hits.push_back(cache.heavy_atoms[pos + static_cast<std::size_t>(lane)]);
      mask &= (mask - 1U);
    }
  }

  for (; pos < cache.heavy_atoms.size(); ++pos) {
    const std::size_t atom_idx = cache.heavy_atoms[pos];
    if (atom_idx >= frame.size())
      continue;
    const double dx = px - frame.pos.x[atom_idx];
    const double dy = py - frame.pos.y[atom_idx];
    const double dz = pz - frame.pos.z[atom_idx];
    if ((dx * dx + dy * dy + dz * dz) <= cutoff_sq)
      hits.push_back(atom_idx);
  }
}

void TitanHBondKineticsAnalyzer::detect_water_inter(
    const MOL &water_frame, const MoleculeCache &mol1_cache, const MOL &mol1_frame,
    const MoleculeCache &mol2_cache, const MOL &mol2_frame,
    TitanExecutor &executor, std::vector<WaterEvent> &out_events) const {
  out_events.clear();
  const std::size_t water_count = water_frame.size() / 3;
  if (water_count == 0 || mol1_cache.heavy_atoms.empty() ||
      mol2_cache.heavy_atoms.empty()) {
    return;
  }

  const std::size_t worker_slots =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<std::vector<WaterEvent>> worker_events(worker_slots);
  executor.parallel_for(
      water_count, worker_slots, 8,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &local = worker_events[worker_id];
        std::vector<std::size_t> hits_a;
        std::vector<std::size_t> hits_b;

        for (std::size_t water_idx = begin; water_idx < end; ++water_idx) {
          const std::size_t oxygen_idx = water_idx * 3;
          collect_nearby_heavy_atoms(water_frame, oxygen_idx, mol1_cache, mol1_frame,
                                     water_cutoff_sq_, hits_a);
          collect_nearby_heavy_atoms(water_frame, oxygen_idx, mol2_cache, mol2_frame,
                                     water_cutoff_sq_, hits_b);
          if (hits_a.empty() || hits_b.empty())
            continue;

          for (std::size_t atom_a : hits_a) {
            for (std::size_t atom_b : hits_b) {
              local.push_back({static_cast<int>(water_idx), 0,
                               static_cast<int>(atom_a), 1,
                               static_cast<int>(atom_b)});
            }
          }
        }
      });

  for (auto &local : worker_events)
    out_events.insert(out_events.end(), local.begin(), local.end());
  sort_unique_events(out_events);
}

void TitanHBondKineticsAnalyzer::detect_water_intra(
    const MOL &water_frame, const MoleculeCache &cache, const MOL &frame,
    int mol_id, TitanExecutor &executor, std::vector<WaterEvent> &out_events) const {
  out_events.clear();
  const std::size_t water_count = water_frame.size() / 3;
  if (water_count == 0 || cache.heavy_atoms.size() < 2)
    return;

  const std::size_t worker_slots =
      std::max<std::size_t>(1, executor.capacity());
  std::vector<std::vector<WaterEvent>> worker_events(worker_slots);
  executor.parallel_for(
      water_count, worker_slots, 8,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        auto &local = worker_events[worker_id];
        std::vector<std::size_t> hits;

        for (std::size_t water_idx = begin; water_idx < end; ++water_idx) {
          const std::size_t oxygen_idx = water_idx * 3;
          collect_nearby_heavy_atoms(water_frame, oxygen_idx, cache, frame,
                                     water_cutoff_sq_, hits);
          if (hits.size() < 2)
            continue;

          for (std::size_t i = 0; i + 1 < hits.size(); ++i) {
            for (std::size_t j = i + 1; j < hits.size(); ++j) {
              const std::size_t atom_a = hits[i];
              const std::size_t atom_b = hits[j];
              if (cache.atoms[atom_a].chain_id == cache.atoms[atom_b].chain_id &&
                  cache.atoms[atom_a].resnr == cache.atoms[atom_b].resnr) {
                continue;
              }
              const int first = static_cast<int>(std::min(atom_a, atom_b));
              const int second = static_cast<int>(std::max(atom_a, atom_b));
              local.push_back({static_cast<int>(water_idx), mol_id, first, mol_id,
                               second});
            }
          }
        }
      });

  for (auto &local : worker_events)
    out_events.insert(out_events.end(), local.begin(), local.end());
  sort_unique_events(out_events);
}

int TitanHBondKineticsAnalyzer::intern_hbond_event(ScopeKind scope,
                                                   const HBondEvent &event) {
  auto &series = hbond_series_[scope_index(scope)];
  const std::string key =
      std::to_string(event.donor_mol) + "|" + std::to_string(event.donor_idx) +
      "|" + std::to_string(event.hydrogen_idx) + "|" +
      std::to_string(event.acceptor_mol) + "|" +
      std::to_string(event.acceptor_idx);
  const auto it = series.id_by_key.find(key);
  if (it != series.id_by_key.end())
    return it->second;

  const int id = static_cast<int>(series.descriptors.size());
  series.id_by_key.emplace(key, id);
  const auto &donor_cache = molecule_cache(event.donor_mol);
  const auto &acceptor_cache = molecule_cache(event.acceptor_mol);

  HBondDescriptor descriptor;
  descriptor.scope = scope_label(scope);
  descriptor.donor_mol = donor_cache.label;
  descriptor.hydrogen_mol = donor_cache.label;
  descriptor.acceptor_mol = acceptor_cache.label;
  descriptor.donor = donor_cache.atoms[static_cast<std::size_t>(event.donor_idx)];
  descriptor.hydrogen =
      donor_cache.atoms[static_cast<std::size_t>(event.hydrogen_idx)];
  descriptor.acceptor =
      acceptor_cache.atoms[static_cast<std::size_t>(event.acceptor_idx)];
  series.descriptors.push_back(std::move(descriptor));
  series.kinetics.ensure_id(static_cast<std::size_t>(id));
  return id;
}

int TitanHBondKineticsAnalyzer::intern_salt_event(ScopeKind scope,
                                                  const SaltEvent &event) {
  auto &series = salt_series_[scope_index(scope)];
  const std::string key =
      std::to_string(event.acidic_mol) + "|" +
      std::to_string(event.acidic_idx) + "|" +
      std::to_string(event.basic_mol) + "|" +
      std::to_string(event.basic_idx);
  const auto it = series.id_by_key.find(key);
  if (it != series.id_by_key.end())
    return it->second;

  const int id = static_cast<int>(series.descriptors.size());
  series.id_by_key.emplace(key, id);
  const auto &acid_cache = molecule_cache(event.acidic_mol);
  const auto &base_cache = molecule_cache(event.basic_mol);

  SaltDescriptor descriptor;
  descriptor.scope = scope_label(scope);
  descriptor.acidic_mol = acid_cache.label;
  descriptor.basic_mol = base_cache.label;
  descriptor.acidic =
      acid_cache.atoms[static_cast<std::size_t>(event.acidic_idx)];
  descriptor.basic = base_cache.atoms[static_cast<std::size_t>(event.basic_idx)];
  series.descriptors.push_back(std::move(descriptor));
  series.kinetics.ensure_id(static_cast<std::size_t>(id));
  return id;
}

int TitanHBondKineticsAnalyzer::intern_water_event(ScopeKind scope,
                                                   const WaterEvent &event) {
  auto &series = water_series_[scope_index(scope)];
  const std::string key =
      std::to_string(event.water_index) + "|" +
      std::to_string(event.atom_a_mol) + "|" +
      std::to_string(event.atom_a_idx) + "|" +
      std::to_string(event.atom_b_mol) + "|" +
      std::to_string(event.atom_b_idx);
  const auto it = series.id_by_key.find(key);
  if (it != series.id_by_key.end())
    return it->second;

  const int id = static_cast<int>(series.descriptors.size());
  series.id_by_key.emplace(key, id);
  const auto &cache_a = molecule_cache(event.atom_a_mol);
  const auto &cache_b = molecule_cache(event.atom_b_mol);

  WaterDescriptor descriptor;
  descriptor.scope = scope_label(scope);
  descriptor.water_index = event.water_index;
  descriptor.atom_a_mol = cache_a.label;
  descriptor.atom_b_mol = cache_b.label;
  descriptor.atom_a = cache_a.atoms[static_cast<std::size_t>(event.atom_a_idx)];
  descriptor.atom_b = cache_b.atoms[static_cast<std::size_t>(event.atom_b_idx)];
  series.descriptors.push_back(std::move(descriptor));
  series.kinetics.ensure_id(static_cast<std::size_t>(id));
  return id;
}

void TitanHBondKineticsAnalyzer::process_hbond_scope(
    ScopeKind scope, std::vector<HBondEvent> events) {
  auto &series = hbond_series_[scope_index(scope)];
  if (!series.enabled)
    return;
  sort_unique_events(events);
  std::vector<int> ids;
  ids.reserve(events.size());
  for (const auto &event : events)
    ids.push_back(intern_hbond_event(scope, event));
  series.kinetics.process_active_ids(std::move(ids));
}

void TitanHBondKineticsAnalyzer::process_salt_scope(
    ScopeKind scope, std::vector<SaltEvent> events) {
  auto &series = salt_series_[scope_index(scope)];
  if (!series.enabled)
    return;
  sort_unique_events(events);
  std::vector<int> ids;
  ids.reserve(events.size());
  for (const auto &event : events)
    ids.push_back(intern_salt_event(scope, event));
  series.kinetics.process_active_ids(std::move(ids));
}

void TitanHBondKineticsAnalyzer::process_water_scope(
    ScopeKind scope, std::vector<WaterEvent> events) {
  auto &series = water_series_[scope_index(scope)];
  if (!series.enabled)
    return;
  sort_unique_events(events);
  std::vector<int> ids;
  ids.reserve(events.size());
  for (const auto &event : events)
    ids.push_back(intern_water_event(scope, event));
  series.kinetics.process_active_ids(std::move(ids));
}

void TitanHBondKineticsAnalyzer::process_frame(const MOL &mol1_frame,
                                               const MOL *mol2_frame,
                                               const MOL *water_frame,
                                               TitanExecutor &executor) {
  if (!requested_ || !enabled_)
    return;

  ++frames_seen_;
  update_frame_timing(mol1_frame.time_of_snap);

  if (hbond_series_[scope_index(ScopeKind::Inter)].enabled) {
    std::vector<HBondEvent> events;
    if (mol2_frame != nullptr) {
      std::vector<HBondEvent> forward;
      std::vector<HBondEvent> reverse;
      detect_hbonds_between(mol1_cache_, mol1_frame, 0, mol2_cache_, *mol2_frame,
                            1, false, executor, forward);
      detect_hbonds_between(mol2_cache_, *mol2_frame, 1, mol1_cache_, mol1_frame,
                            0, false, executor, reverse);
      events.reserve(forward.size() + reverse.size());
      events.insert(events.end(), forward.begin(), forward.end());
      events.insert(events.end(), reverse.begin(), reverse.end());
    }
    process_hbond_scope(ScopeKind::Inter, std::move(events));
  }
  if (hbond_series_[scope_index(ScopeKind::IntraMol1)].enabled) {
    std::vector<HBondEvent> events;
    detect_hbonds_between(mol1_cache_, mol1_frame, 0, mol1_cache_, mol1_frame, 0,
                          true, executor, events);
    process_hbond_scope(ScopeKind::IntraMol1, std::move(events));
  }
  if (hbond_series_[scope_index(ScopeKind::IntraMol2)].enabled) {
    std::vector<HBondEvent> events;
    if (mol2_frame != nullptr) {
      detect_hbonds_between(mol2_cache_, *mol2_frame, 1, mol2_cache_, *mol2_frame,
                            1, true, executor, events);
    }
    process_hbond_scope(ScopeKind::IntraMol2, std::move(events));
  }

  if (salt_series_[scope_index(ScopeKind::Inter)].enabled) {
    std::vector<SaltEvent> events;
    if (mol2_frame != nullptr) {
      std::vector<SaltEvent> forward;
      std::vector<SaltEvent> reverse;
      detect_salts_between(mol1_cache_, mol1_frame, 0, mol2_cache_, *mol2_frame,
                           1, false, executor, forward);
      detect_salts_between(mol2_cache_, *mol2_frame, 1, mol1_cache_, mol1_frame,
                           0, false, executor, reverse);
      events.reserve(forward.size() + reverse.size());
      events.insert(events.end(), forward.begin(), forward.end());
      events.insert(events.end(), reverse.begin(), reverse.end());
    }
    process_salt_scope(ScopeKind::Inter, std::move(events));
  }
  if (salt_series_[scope_index(ScopeKind::IntraMol1)].enabled) {
    std::vector<SaltEvent> events;
    detect_salts_between(mol1_cache_, mol1_frame, 0, mol1_cache_, mol1_frame, 0,
                         true, executor, events);
    process_salt_scope(ScopeKind::IntraMol1, std::move(events));
  }
  if (salt_series_[scope_index(ScopeKind::IntraMol2)].enabled) {
    std::vector<SaltEvent> events;
    if (mol2_frame != nullptr) {
      detect_salts_between(mol2_cache_, *mol2_frame, 1, mol2_cache_, *mol2_frame,
                           1, true, executor, events);
    }
    process_salt_scope(ScopeKind::IntraMol2, std::move(events));
  }

  const bool any_water_enabled =
      water_series_[0].enabled || water_series_[1].enabled || water_series_[2].enabled;
  if (any_water_enabled) {
    if (water_frame == nullptr) {
      ++water_missing_frames_;
      if (!warned_missing_water_) {
        TitanUI::print_warning(
            "HBond kinetics: water trajectory was missing for one or more frames. "
            "Water-bridge kinetics will treat those frames as empty.");
        warned_missing_water_ = true;
      }
      for (auto &series : water_series_) {
        if (series.enabled)
          series.kinetics.process_active_ids({});
      }
    } else {
      if (water_series_[scope_index(ScopeKind::Inter)].enabled) {
        std::vector<WaterEvent> events;
        if (mol2_frame != nullptr) {
          detect_water_inter(*water_frame, mol1_cache_, mol1_frame, mol2_cache_,
                             *mol2_frame, executor, events);
        }
        process_water_scope(ScopeKind::Inter, std::move(events));
      }
      if (water_series_[scope_index(ScopeKind::IntraMol1)].enabled) {
        std::vector<WaterEvent> events;
        detect_water_intra(*water_frame, mol1_cache_, mol1_frame, 0, executor,
                           events);
        process_water_scope(ScopeKind::IntraMol1, std::move(events));
      }
      if (water_series_[scope_index(ScopeKind::IntraMol2)].enabled) {
        std::vector<WaterEvent> events;
        if (mol2_frame != nullptr) {
          detect_water_intra(*water_frame, mol2_cache_, *mol2_frame, 1, executor,
                             events);
        }
        process_water_scope(ScopeKind::IntraMol2, std::move(events));
      }
    }
  }
}

void TitanHBondKineticsAnalyzer::export_summary(const std::string &filename,
                                                std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "module: titan_hbond_kinetics\n";
  out << "frames_seen: " << frames_seen_ << "\n";
  out << "requested_hbond: " << (request_hbond_ ? "yes" : "no") << "\n";
  out << "requested_salt: " << (request_salt_ ? "yes" : "no") << "\n";
  out << "requested_water: " << (request_water_ ? "yes" : "no") << "\n";
  out << "water_stream_available: "
      << (water_stream_available_ ? "yes" : "no") << "\n";
  out << "water_missing_frames: " << water_missing_frames_ << "\n";
  out << "maxlag_frames: " << maxlag_ << "\n";
  out << "frame_dt_ps: " << std::fixed << std::setprecision(6) << frame_dt_ps()
      << "\n";
  out << "hbond_cutoff_A: " << hbond_cutoff_ang_ << "\n";
  out << "hbond_angle_deg: " << angle_cutoff_deg_ << "\n";
  out << "salt_cutoff_A: " << salt_cutoff_ang_ << "\n";
  out << "water_cutoff_A: " << water_cutoff_ang_ << "\n";

  auto write_series_summary = [&](const char *kind_name, const auto &series_vec) {
    for (const auto &series : series_vec) {
      const auto lifetime = aggregate_lifetimes(series.kinetics.key_stats());
      const double mean_frames = safe_divide(
          lifetime.first, static_cast<double>(lifetime.second));
      const double tau_frames =
          integrated_continuous_frames(series.kinetics);
      out << kind_name << "." << series.scope_label
          << ".status: " << series.status << "\n";
      out << kind_name << "." << series.scope_label
          << ".frames: " << series.kinetics.frame_count() << "\n";
      out << kind_name << "." << series.scope_label
          << ".keys: " << series.descriptors.size() << "\n";
      out << kind_name << "." << series.scope_label
          << ".events: " << lifetime.second << "\n";
      out << kind_name << "." << series.scope_label
          << ".mean_lifetime_frames: " << mean_frames << "\n";
      out << kind_name << "." << series.scope_label
          << ".mean_lifetime_ps: " << mean_frames * frame_dt_ps() << "\n";
      out << kind_name << "." << series.scope_label
          << ".integrated_ct_frames: " << tau_frames << "\n";
      out << kind_name << "." << series.scope_label
          << ".integrated_ct_ps: " << tau_frames * frame_dt_ps() << "\n";
    }
  };

  write_series_summary("hbond", hbond_series_);
  write_series_summary("salt", salt_series_);
  write_series_summary("water", water_series_);

  if (log.is_open())
    TitanUI::log_kv(log, "HBK Summary", filename);
}

void TitanHBondKineticsAnalyzer::export_hbond_pairs(
    const std::string &filename, std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "Scope,DonorMol,DonorChain,DonorResID,DonorResName,DonorAtom,"
         "HydrogenMol,HydrogenChain,HydrogenResID,HydrogenResName,HydrogenAtom,"
         "AcceptorMol,AcceptorChain,AcceptorResID,AcceptorResName,AcceptorAtom,"
         "Occupancy_%,Events,MeanLifetime_frames,MeanLifetime_ps,"
         "MaxLifetime_frames,MaxLifetime_ps\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &series : hbond_series_) {
    const auto &stats = series.kinetics.key_stats();
    const double frames = static_cast<double>(series.kinetics.frame_count());
    for (std::size_t id = 0; id < series.descriptors.size(); ++id) {
      const auto &descriptor = series.descriptors[id];
      const auto &stat = stats[id];
      const double mean_frames = safe_divide(
          static_cast<double>(stat.total_lifetime_frames),
          static_cast<double>(stat.event_count));
      const double occupancy = safe_divide(
          100.0 * static_cast<double>(stat.total_active_frames), frames);
      out << descriptor.scope << "," << descriptor.donor_mol << ","
          << descriptor.donor.chain << "," << descriptor.donor.resnr << ","
          << descriptor.donor.res_name << "," << descriptor.donor.atom_name << ","
          << descriptor.hydrogen_mol << "," << descriptor.hydrogen.chain << ","
          << descriptor.hydrogen.resnr << "," << descriptor.hydrogen.res_name
          << "," << descriptor.hydrogen.atom_name << ","
          << descriptor.acceptor_mol << "," << descriptor.acceptor.chain << ","
          << descriptor.acceptor.resnr << "," << descriptor.acceptor.res_name
          << "," << descriptor.acceptor.atom_name << "," << occupancy << ","
          << stat.event_count << "," << mean_frames << ","
          << mean_frames * frame_dt_ps() << "," << stat.max_lifetime_frames
          << "," << stat.max_lifetime_frames * frame_dt_ps() << "\n";
    }
  }

  if (log.is_open())
    TitanUI::log_kv(log, "HBK HBond Pairs", filename);
}

void TitanHBondKineticsAnalyzer::export_salt_pairs(
    const std::string &filename, std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "Scope,AcidicMol,AcidicChain,AcidicResID,AcidicResName,AcidicAtom,"
         "BasicMol,BasicChain,BasicResID,BasicResName,BasicAtom,"
         "Occupancy_%,Events,MeanLifetime_frames,MeanLifetime_ps,"
         "MaxLifetime_frames,MaxLifetime_ps\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &series : salt_series_) {
    const auto &stats = series.kinetics.key_stats();
    const double frames = static_cast<double>(series.kinetics.frame_count());
    for (std::size_t id = 0; id < series.descriptors.size(); ++id) {
      const auto &descriptor = series.descriptors[id];
      const auto &stat = stats[id];
      const double mean_frames = safe_divide(
          static_cast<double>(stat.total_lifetime_frames),
          static_cast<double>(stat.event_count));
      const double occupancy = safe_divide(
          100.0 * static_cast<double>(stat.total_active_frames), frames);
      out << descriptor.scope << "," << descriptor.acidic_mol << ","
          << descriptor.acidic.chain << "," << descriptor.acidic.resnr << ","
          << descriptor.acidic.res_name << "," << descriptor.acidic.atom_name
          << "," << descriptor.basic_mol << "," << descriptor.basic.chain << ","
          << descriptor.basic.resnr << "," << descriptor.basic.res_name << ","
          << descriptor.basic.atom_name << "," << occupancy << ","
          << stat.event_count << "," << mean_frames << ","
          << mean_frames * frame_dt_ps() << "," << stat.max_lifetime_frames
          << "," << stat.max_lifetime_frames * frame_dt_ps() << "\n";
    }
  }

  if (log.is_open())
    TitanUI::log_kv(log, "HBK Salt Pairs", filename);
}

void TitanHBondKineticsAnalyzer::export_water_pairs(
    const std::string &filename, std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "Scope,WaterID,AtomAMol,AtomAChain,AtomAResID,AtomAResName,AtomAAtom,"
         "AtomBMol,AtomBChain,AtomBResID,AtomBResName,AtomBAtom,"
         "Occupancy_%,Events,MeanLifetime_frames,MeanLifetime_ps,"
         "MaxLifetime_frames,MaxLifetime_ps\n";
  out << std::fixed << std::setprecision(6);
  for (const auto &series : water_series_) {
    const auto &stats = series.kinetics.key_stats();
    const double frames = static_cast<double>(series.kinetics.frame_count());
    for (std::size_t id = 0; id < series.descriptors.size(); ++id) {
      const auto &descriptor = series.descriptors[id];
      const auto &stat = stats[id];
      const double mean_frames = safe_divide(
          static_cast<double>(stat.total_lifetime_frames),
          static_cast<double>(stat.event_count));
      const double occupancy = safe_divide(
          100.0 * static_cast<double>(stat.total_active_frames), frames);
      out << descriptor.scope << "," << descriptor.water_index << ","
          << descriptor.atom_a_mol << "," << descriptor.atom_a.chain << ","
          << descriptor.atom_a.resnr << "," << descriptor.atom_a.res_name << ","
          << descriptor.atom_a.atom_name << "," << descriptor.atom_b_mol << ","
          << descriptor.atom_b.chain << "," << descriptor.atom_b.resnr << ","
          << descriptor.atom_b.res_name << "," << descriptor.atom_b.atom_name
          << "," << occupancy << "," << stat.event_count << "," << mean_frames
          << "," << mean_frames * frame_dt_ps() << ","
          << stat.max_lifetime_frames << ","
          << stat.max_lifetime_frames * frame_dt_ps() << "\n";
    }
  }

  if (log.is_open())
    TitanUI::log_kv(log, "HBK Water Pairs", filename);
}

void TitanHBondKineticsAnalyzer::export_ct_file(const std::string &filename,
                                                InteractionKind kind,
                                                std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "# TITAN HBond kinetics correlation profile\n";
  out << "# columns = Scope lag_frame lag_ps C_continuous C_intermittent "
         "origin_count\n";
  out << std::fixed << std::setprecision(8);

  auto write_series = [&](const auto &series_vec) {
    for (const auto &series : series_vec) {
      if (!series.enabled)
        continue;
      const auto &origins = series.kinetics.origin_counts();
      const auto &cont = series.kinetics.continuous_hits();
      const auto &inter = series.kinetics.intermittent_hits();
      const std::size_t limit =
          std::min(origins.size(), std::min(cont.size(), inter.size()));
      for (std::size_t lag = 0; lag < limit; ++lag) {
        const double origin = static_cast<double>(origins[lag]);
        out << series.scope_label << " " << lag << " "
            << lag * frame_dt_ps() << " "
            << safe_divide(static_cast<double>(cont[lag]), origin) << " "
            << safe_divide(static_cast<double>(inter[lag]), origin) << " "
            << origins[lag] << "\n";
      }
    }
  };

  switch (kind) {
  case InteractionKind::HBond:
    write_series(hbond_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK HBond C(t)", filename);
    break;
  case InteractionKind::Salt:
    write_series(salt_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK Salt C(t)", filename);
    break;
  case InteractionKind::Water:
    write_series(water_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK Water C(t)", filename);
    break;
  }
}

void TitanHBondKineticsAnalyzer::export_lifetime_file(
    const std::string &filename, InteractionKind kind, std::fstream &log) const {
  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("HBond kinetics: failed to open " + filename);
    return;
  }

  out << "# TITAN HBond kinetics lifetime histogram\n";
  out << "# columns = Scope lifetime_frames lifetime_ps count\n";
  out << std::fixed << std::setprecision(8);

  auto write_series = [&](const auto &series_vec) {
    for (const auto &series : series_vec) {
      if (!series.enabled)
        continue;
      const auto &histogram = series.kinetics.lifetime_histogram();
      for (std::size_t lifetime = 1; lifetime < histogram.size(); ++lifetime) {
        const std::uint64_t count = histogram[lifetime];
        if (count == 0)
          continue;
        out << series.scope_label << " " << lifetime << " "
            << lifetime * frame_dt_ps() << " " << count << "\n";
      }
    }
  };

  switch (kind) {
  case InteractionKind::HBond:
    write_series(hbond_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK HBond Lifetimes", filename);
    break;
  case InteractionKind::Salt:
    write_series(salt_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK Salt Lifetimes", filename);
    break;
  case InteractionKind::Water:
    write_series(water_series_);
    if (log.is_open())
      TitanUI::log_kv(log, "HBK Water Lifetimes", filename);
    break;
  }
}

void TitanHBondKineticsAnalyzer::finalize_trajectory(
    const std::string &base_dir, const info &info_output, const arguments &args,
    const std::vector<std::string> &input_line, int input_name,
    std::fstream &log) {
  if (!requested_)
    return;

  for (auto &series : hbond_series_)
    if (series.enabled)
      series.kinetics.finalize();
  for (auto &series : salt_series_)
    if (series.enabled)
      series.kinetics.finalize();
  for (auto &series : water_series_)
    if (series.enabled)
      series.kinetics.finalize();

  const std::string summary_file =
      base_dir + "/" +
      info_output.get_hbond_kinetics_summary_filename(args, input_line,
                                                      input_name);
  export_summary(summary_file, log);

  export_hbond_pairs(base_dir + "/" +
                         info_output.get_hbond_kinetics_pairs_filename(
                             args, input_line, input_name, "hbond"),
                     log);
  export_salt_pairs(base_dir + "/" +
                        info_output.get_hbond_kinetics_pairs_filename(
                            args, input_line, input_name, "salt"),
                    log);
  export_water_pairs(base_dir + "/" +
                         info_output.get_hbond_kinetics_pairs_filename(
                             args, input_line, input_name, "water"),
                     log);

  export_ct_file(base_dir + "/" +
                     info_output.get_hbond_kinetics_ct_filename(
                         args, input_line, input_name, "hbond"),
                 InteractionKind::HBond, log);
  export_ct_file(base_dir + "/" +
                     info_output.get_hbond_kinetics_ct_filename(
                         args, input_line, input_name, "salt"),
                 InteractionKind::Salt, log);
  export_ct_file(base_dir + "/" +
                     info_output.get_hbond_kinetics_ct_filename(
                         args, input_line, input_name, "water"),
                 InteractionKind::Water, log);

  export_lifetime_file(base_dir + "/" +
                           info_output.get_hbond_kinetics_lifetimes_filename(
                               args, input_line, input_name, "hbond"),
                       InteractionKind::HBond, log);
  export_lifetime_file(base_dir + "/" +
                           info_output.get_hbond_kinetics_lifetimes_filename(
                               args, input_line, input_name, "salt"),
                       InteractionKind::Salt, log);
  export_lifetime_file(base_dir + "/" +
                           info_output.get_hbond_kinetics_lifetimes_filename(
                               args, input_line, input_name, "water"),
                       InteractionKind::Water, log);

  TitanUI::print_info("[HBK] Exported kinetics summary to " + summary_file);
  reset_trajectory_state();
}

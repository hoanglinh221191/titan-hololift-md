#include "titan_dssp.h"
#include "titan_executor.h"
#include "titan_simd_dispatch.h"

#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinimalDistance = 0.5;
constexpr double kMinimalDistance2 = kMinimalDistance * kMinimalDistance;
constexpr double kMinimalCADistance = 9.0;
constexpr double kMinimalCADistance2 = kMinimalCADistance * kMinimalCADistance;
constexpr double kMinHBondEnergy = -9.9;
constexpr double kCouplingConstant = -27.888;

std::string uppercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::toupper(c));
                 });
  return value;
}

std::string trim_copy(std::string value) {
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string::npos)
    return {};
  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1);
}

std::string normalized_name(std::string value) {
  return uppercase_copy(trim_copy(std::move(value)));
}

bool is_proline_like(const std::string &resname) {
  const std::string upper = normalized_name(resname);
  return upper.starts_with("PRO") || upper == "HYP";
}

bool is_primary_oxygen_name(const std::string &atom_name) {
  return atom_name == "O" || atom_name == "O1" || atom_name == "OT1" ||
         atom_name == "OC1";
}

bool is_backbone_oxygen_name(const std::string &atom_name) {
  return is_primary_oxygen_name(atom_name) || atom_name == "OXT" ||
         atom_name == "OT2" || atom_name == "OC2";
}

bool is_backbone_hydrogen_name(const std::string &atom_name) {
  return atom_name == "H" || atom_name == "HN" || atom_name == "H1" ||
         atom_name == "H2" || atom_name == "H3" || atom_name == "HT1" ||
         atom_name == "HT2" || atom_name == "HT3" || atom_name == "1H" ||
         atom_name == "2H" || atom_name == "3H";
}

double dot3(double ax, double ay, double az, double bx, double by, double bz) {
  return ax * bx + ay * by + az * bz;
}

double norm3(double x, double y, double z) {
  return std::sqrt(x * x + y * y + z * z);
}

double clamp_cos(double value) {
  return std::max(-1.0, std::min(1.0, value));
}

} // namespace

bool TitanDSSPAnalyzer::supports_moltype(const std::string &mol_type) {
  return mol_type == "protein" || mol_type == "protein-user";
}

std::size_t TitanDSSPAnalyzer::compatible_backbone_residue_count(
    const std::vector<mol_info> &info) {
  if (info.empty())
    return 0;

  std::size_t prev_chain = std::numeric_limits<std::size_t>::max();
  std::size_t prev_resnr = std::numeric_limits<std::size_t>::max();
  bool has_n = false;
  bool has_ca = false;
  bool has_c = false;
  bool has_o = false;
  std::size_t compatible_residues = 0;

  auto flush_residue = [&]() {
    if (has_n && has_ca && has_c && has_o)
      ++compatible_residues;
    has_n = false;
    has_ca = false;
    has_c = false;
    has_o = false;
  };

  for (const auto &atom_info : info) {
    const std::size_t chainnum = atom_info.get_chainnum();
    const std::size_t resnr = atom_info.get_resnr();
    if (chainnum != prev_chain || resnr != prev_resnr) {
      if (prev_chain != std::numeric_limits<std::size_t>::max()) {
        flush_residue();
      }
      prev_chain = chainnum;
      prev_resnr = resnr;
    }

    const std::string atom_name = normalized_name(atom_info.get_atom());
    if (atom_name == "N") {
      has_n = true;
    } else if (atom_name == "CA") {
      has_ca = true;
    } else if (atom_name == "C") {
      has_c = true;
    } else if (is_backbone_oxygen_name(atom_name)) {
      has_o = true;
    }
  }

  flush_residue();
  return compatible_residues;
}

bool TitanDSSPAnalyzer::has_protein_backbone_signature(
    const std::vector<mol_info> &info) {
  const std::size_t compatible_residues =
      TitanDSSPAnalyzer::compatible_backbone_residue_count(info);
  return compatible_residues >= 3;
}

char TitanDSSPAnalyzer::state_to_char(State state) {
  switch (state) {
  case State::H:
    return 'H';
  case State::B:
    return 'B';
  case State::E:
    return 'E';
  case State::G:
    return 'G';
  case State::I:
    return 'I';
  case State::T:
    return 'T';
  case State::S:
    return 'S';
  case State::C:
  default:
    return 'C';
  }
}

const char *TitanDSSPAnalyzer::state_name(State state) {
  switch (state) {
  case State::H:
    return "alpha_helix";
  case State::B:
    return "beta_bridge";
  case State::E:
    return "beta_strand";
  case State::G:
    return "310_helix";
  case State::I:
    return "pi_helix";
  case State::T:
    return "turn";
  case State::S:
    return "bend";
  case State::C:
  default:
    return "coil";
  }
}

void TitanDSSPAnalyzer::configure(const DsspArgs &args) {
  requested_ = args.flag_dssp;
  hbond_cutoff_ = args.hbond_energy_cutoff;
  enabled_ = requested_ && initialized_;
}

void TitanDSSPAnalyzer::configure_segments(const SegmentArgs &args,
                                           std::uint8_t molecule) {
  segment_config_file_ = args.config_file;
  molecule_id_ = molecule;
}

void TitanDSSPAnalyzer::init(const Multi_MOL &mol, const std::string &label) {
  label_ = label;
  if (initialized_)
    return;

  build_residue_map(mol.info);
  load_segment_annotations();
  initialized_ = !residues_.empty();
  enabled_ = requested_ && initialized_;
  if (!initialized_)
    return;

  ensure_scratch_capacity(residues_.size());
  current_traj_counts_.assign(residues_.size(), {});
}

void TitanDSSPAnalyzer::build_residue_map(const std::vector<mol_info> &info) {
  residues_.clear();
  if (info.empty())
    return;

  std::size_t prev_chain = std::numeric_limits<std::size_t>::max();
  std::size_t prev_resnr = std::numeric_limits<std::size_t>::max();
  std::size_t local_index = 0;

  for (std::size_t atom_idx = 0; atom_idx < info.size(); ++atom_idx) {
    const auto &atom_info = info[atom_idx];
    const std::size_t chainnum = atom_info.get_chainnum();
    const std::size_t resnr = atom_info.get_resnr();

    const bool new_residue =
        residues_.empty() || chainnum != prev_chain || resnr != prev_resnr;
    if (new_residue) {
      if (residues_.empty() || chainnum != prev_chain)
        local_index = 0;
      else
        ++local_index;

      residues_.push_back(ResidueBackbone{
          residues_.size() + 1, resnr, atom_info.get_internal_resnr(),
          chainnum, local_index,
          atom_info.get_chain(), atom_info.get_res_name(), -1, -1, -1, -1, -1});

      prev_chain = chainnum;
      prev_resnr = resnr;
    }

    ResidueBackbone &residue = residues_.back();
    const std::string atom_name = normalized_name(atom_info.get_atom());

    if (atom_name == "N" && residue.atom_n < 0) {
      residue.atom_n = static_cast<int>(atom_idx);
      continue;
    }
    if (atom_name == "CA" && residue.atom_ca < 0) {
      residue.atom_ca = static_cast<int>(atom_idx);
      continue;
    }
    if (atom_name == "C" && residue.atom_c < 0) {
      residue.atom_c = static_cast<int>(atom_idx);
      continue;
    }
    if (is_backbone_hydrogen_name(atom_name) &&
        (residue.atom_h < 0 || atom_name == "H" || atom_name == "HN")) {
      residue.atom_h = static_cast<int>(atom_idx);
      continue;
    }
    if (is_backbone_oxygen_name(atom_name) &&
        (residue.atom_o < 0 || is_primary_oxygen_name(atom_name))) {
      residue.atom_o = static_cast<int>(atom_idx);
      continue;
    }
  }
}

void TitanDSSPAnalyzer::load_segment_annotations() {
  residue_annotations_.assign(residues_.size(), {});
  resolved_segments_.clear();
  segment_config_loaded_ = false;
  if (segment_config_file_.empty() || residues_.empty())
    return;

  segment_config_ = titan_segments::load_config(segment_config_file_);
  segment_config_loaded_ = true;

  std::map<std::pair<std::string, std::string>, std::size_t>
      resolved_lookup;
  for (const titan_segments::Entity &entity : segment_config_.entities) {
    if (entity.molecule != molecule_id_)
      continue;
    const std::size_t all_index = resolved_segments_.size();
    resolved_segments_.push_back(
        ResolvedSegment{entity.name, "all", "all", 0, 0, {}});
    resolved_lookup.emplace(std::make_pair(entity.name, "all"), all_index);
    for (const titan_segments::Segment *segment :
         segment_config_.segments_for_entity(entity.name)) {
      const std::size_t index = resolved_segments_.size();
      resolved_segments_.push_back(
          ResolvedSegment{entity.name, segment->name,
                          titan_segments::coordinate_name(segment->coordinate),
                          segment->begin, segment->end, {}});
      resolved_lookup.emplace(std::make_pair(entity.name, segment->name),
                              index);
    }
  }

  for (std::size_t residue_index = 0; residue_index < residues_.size();
       ++residue_index) {
    const ResidueBackbone &residue = residues_[residue_index];
    titan_segments::ResidueDescriptor descriptor;
    descriptor.molecule = molecule_id_;
    descriptor.chain_number = residue.chainnum;
    descriptor.chain = residue.chain;
    descriptor.residue_number = residue.resnr;
    descriptor.internal_residue_number = residue.internal_resnr;
    descriptor.ordinal = residue.local_index + 1;
    descriptor.residue_name = residue.resname;

    const titan_segments::Entity *matched_entity = nullptr;
    for (const titan_segments::Entity &entity : segment_config_.entities) {
      if (!titan_segments::entity_matches(entity, descriptor))
        continue;
      if (matched_entity != nullptr) {
        throw std::runtime_error(
            "DSSP segment entities overlap at molecule " +
            std::to_string(static_cast<int>(molecule_id_)) + ", chain '" +
            residue.chain + "', residue " + std::to_string(residue.resnr) +
            ".");
      }
      matched_entity = &entity;
    }
    if (matched_entity == nullptr)
      continue;

    const auto all_it =
        resolved_lookup.find({matched_entity->name, "all"});
    if (all_it != resolved_lookup.end())
      resolved_segments_[all_it->second].residue_indices.push_back(
          residue_index);

    const auto segments =
        segment_config_.segments_for_entity(matched_entity->name);
    const titan_segments::Segment *matched_segment = nullptr;
    for (const titan_segments::Segment *segment : segments) {
      if (!titan_segments::segment_matches(*segment, descriptor))
        continue;
      if (matched_segment != nullptr) {
        throw std::runtime_error(
            "DSSP segments overlap for entity '" + matched_entity->name +
            "' at ordinal " + std::to_string(descriptor.ordinal) + ".");
      }
      matched_segment = segment;
    }

    residue_annotations_[residue_index].entity = matched_entity->name;
    residue_annotations_[residue_index].segment =
        matched_segment != nullptr
            ? matched_segment->name
            : (segments.empty() ? "all" : "unassigned");
    if (matched_segment != nullptr) {
      const auto segment_it =
          resolved_lookup.find({matched_entity->name, matched_segment->name});
      if (segment_it != resolved_lookup.end())
        resolved_segments_[segment_it->second].residue_indices.push_back(
            residue_index);
    }
  }

  std::map<std::string, std::vector<std::size_t>> unassigned;
  for (std::size_t i = 0; i < residue_annotations_.size(); ++i) {
    if (!residue_annotations_[i].entity.empty() &&
        residue_annotations_[i].segment == "unassigned") {
      unassigned[residue_annotations_[i].entity].push_back(i);
    }
  }
  for (auto &[entity, indices] : unassigned) {
    resolved_segments_.push_back(ResolvedSegment{
        entity, "unassigned", "resolved", 0, 0, std::move(indices)});
  }
}

const char *
TitanDSSPAnalyzer::bridge_type_name(BridgeType type) noexcept {
  switch (type) {
  case BridgeType::Parallel:
    return "parallel";
  case BridgeType::AntiParallel:
    return "antiparallel";
  case BridgeType::None:
    return "none";
  }
  return "none";
}

void TitanDSSPAnalyzer::ensure_scratch_capacity(std::size_t n_res) {
  auto resize_aligned = [n_res](AlignedDoubleVector &vec) {
    vec.assign(n_res, 0.0);
  };

  resize_aligned(n_x_);
  resize_aligned(n_y_);
  resize_aligned(n_z_);
  resize_aligned(h_x_);
  resize_aligned(h_y_);
  resize_aligned(h_z_);
  resize_aligned(ca_x_);
  resize_aligned(ca_y_);
  resize_aligned(ca_z_);
  resize_aligned(c_x_);
  resize_aligned(c_y_);
  resize_aligned(c_z_);
  resize_aligned(o_x_);
  resize_aligned(o_y_);
  resize_aligned(o_z_);
  resize_aligned(kappa_deg_);

  has_n_.assign(n_res, 0);
  has_h_.assign(n_res, 0);
  has_ca_.assign(n_res, 0);
  has_c_.assign(n_res, 0);
  has_o_.assign(n_res, 0);
  donor_valid_.assign(n_res, 0);
  acceptor_valid_.assign(n_res, 0);
  has_kappa_.assign(n_res, 0);
  frame_states_.assign(n_res, State::C);
  best_acceptors_.assign(n_res, {});
  helix_flags_.assign(n_res, {});
}

void TitanDSSPAnalyzer::prepare_backbone_coordinates(const MOL &frame) {
  const std::size_t n_res = residues_.size();
  if (n_res == 0)
    return;

  std::fill(n_x_.begin(), n_x_.end(), 0.0);
  std::fill(n_y_.begin(), n_y_.end(), 0.0);
  std::fill(n_z_.begin(), n_z_.end(), 0.0);
  std::fill(h_x_.begin(), h_x_.end(), 0.0);
  std::fill(h_y_.begin(), h_y_.end(), 0.0);
  std::fill(h_z_.begin(), h_z_.end(), 0.0);
  std::fill(ca_x_.begin(), ca_x_.end(), 0.0);
  std::fill(ca_y_.begin(), ca_y_.end(), 0.0);
  std::fill(ca_z_.begin(), ca_z_.end(), 0.0);
  std::fill(c_x_.begin(), c_x_.end(), 0.0);
  std::fill(c_y_.begin(), c_y_.end(), 0.0);
  std::fill(c_z_.begin(), c_z_.end(), 0.0);
  std::fill(o_x_.begin(), o_x_.end(), 0.0);
  std::fill(o_y_.begin(), o_y_.end(), 0.0);
  std::fill(o_z_.begin(), o_z_.end(), 0.0);
  std::fill(kappa_deg_.begin(), kappa_deg_.end(), 0.0);
  std::fill(has_n_.begin(), has_n_.end(), 0);
  std::fill(has_h_.begin(), has_h_.end(), 0);
  std::fill(has_ca_.begin(), has_ca_.end(), 0);
  std::fill(has_c_.begin(), has_c_.end(), 0);
  std::fill(has_o_.begin(), has_o_.end(), 0);
  std::fill(has_kappa_.begin(), has_kappa_.end(), 0);

  auto copy_atom = [&frame](int atom_idx, std::size_t residue_idx,
                            AlignedDoubleVector &x, AlignedDoubleVector &y,
                            AlignedDoubleVector &z,
                            std::vector<std::uint8_t> &mask) {
    if (atom_idx < 0)
      return;
    const std::size_t ai = static_cast<std::size_t>(atom_idx);
    if (ai >= frame.pos.x.size() || ai >= frame.pos.y.size() ||
        ai >= frame.pos.z.size()) {
      return;
    }
    x[residue_idx] = frame.pos.x[ai];
    y[residue_idx] = frame.pos.y[ai];
    z[residue_idx] = frame.pos.z[ai];
    mask[residue_idx] = 1;
  };

  for (std::size_t i = 0; i < n_res; ++i) {
    const ResidueBackbone &residue = residues_[i];
    copy_atom(residue.atom_n, i, n_x_, n_y_, n_z_, has_n_);
    copy_atom(residue.atom_h, i, h_x_, h_y_, h_z_, has_h_);
    copy_atom(residue.atom_ca, i, ca_x_, ca_y_, ca_z_, has_ca_);
    copy_atom(residue.atom_c, i, c_x_, c_y_, c_z_, has_c_);
    copy_atom(residue.atom_o, i, o_x_, o_y_, o_z_, has_o_);
  }

  for (std::size_t i = 1; i < n_res; ++i) {
    if (!has_n_[i] || has_h_[i] || is_proline_like(residues_[i].resname))
      continue;
    if (!has_c_[i - 1] || !has_o_[i - 1])
      continue;

    h_x_[i] = n_x_[i] + (c_x_[i - 1] - o_x_[i - 1]);
    h_y_[i] = n_y_[i] + (c_y_[i - 1] - o_y_[i - 1]);
    h_z_[i] = n_z_[i] + (c_z_[i - 1] - o_z_[i - 1]);
    has_h_[i] = 1;
  }

  for (std::size_t i = 0; i < n_res; ++i) {
    donor_valid_[i] = static_cast<std::uint8_t>(has_n_[i] && has_h_[i]);
    acceptor_valid_[i] = static_cast<std::uint8_t>(has_c_[i] && has_o_[i]);
  }

  for (std::size_t i = 2; i + 2 < n_res; ++i) {
    if (!has_ca_[i - 2] || !has_ca_[i] || !has_ca_[i + 2])
      continue;
    if (!no_chain_break_between(i - 2, i + 2))
      continue;

    const double v1x = ca_x_[i] - ca_x_[i - 2];
    const double v1y = ca_y_[i] - ca_y_[i - 2];
    const double v1z = ca_z_[i] - ca_z_[i - 2];
    const double v2x = ca_x_[i + 2] - ca_x_[i];
    const double v2y = ca_y_[i + 2] - ca_y_[i];
    const double v2z = ca_z_[i + 2] - ca_z_[i];
    const double denom = norm3(v1x, v1y, v1z) * norm3(v2x, v2y, v2z);
    if (denom <= 0.0)
      continue;

    const double cos_theta =
        clamp_cos(dot3(v1x, v1y, v1z, v2x, v2y, v2z) / denom);
    const double sin_theta =
        std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    const double kappa = std::atan2(sin_theta, cos_theta) * 180.0 / kPi;
    if (!std::isnan(kappa)) {
      kappa_deg_[i] = kappa;
      has_kappa_[i] = 1;
    }
  }
}

bool TitanDSSPAnalyzer::no_chain_break_between(std::size_t begin,
                                               std::size_t end) const noexcept {
  if (begin >= residues_.size() || end >= residues_.size())
    return false;
  if (begin > end)
    std::swap(begin, end);
  const auto &lhs = residues_[begin];
  const auto &rhs = residues_[end];
  return lhs.chainnum == rhs.chainnum &&
         rhs.local_index == lhs.local_index + (end - begin);
}

bool TitanDSSPAnalyzer::donor_acceptor_excluded(std::size_t donor,
                                                std::size_t acceptor) const
    noexcept {
  return donor == acceptor || acceptor + 1 == donor;
}

double TitanDSSPAnalyzer::hbond_energy_scalar(std::size_t donor,
                                              std::size_t acceptor) const
    noexcept {
  if (donor >= residues_.size() || acceptor >= residues_.size())
    return 0.0;
  if (is_proline_like(residues_[donor].resname))
    return 0.0;

  const double dx_ho = h_x_[donor] - o_x_[acceptor];
  const double dy_ho = h_y_[donor] - o_y_[acceptor];
  const double dz_ho = h_z_[donor] - o_z_[acceptor];
  const double dx_hc = h_x_[donor] - c_x_[acceptor];
  const double dy_hc = h_y_[donor] - c_y_[acceptor];
  const double dz_hc = h_z_[donor] - c_z_[acceptor];
  const double dx_nc = n_x_[donor] - c_x_[acceptor];
  const double dy_nc = n_y_[donor] - c_y_[acceptor];
  const double dz_nc = n_z_[donor] - c_z_[acceptor];
  const double dx_no = n_x_[donor] - o_x_[acceptor];
  const double dy_no = n_y_[donor] - o_y_[acceptor];
  const double dz_no = n_z_[donor] - o_z_[acceptor];

  const double d2_ho = dx_ho * dx_ho + dy_ho * dy_ho + dz_ho * dz_ho;
  const double d2_hc = dx_hc * dx_hc + dy_hc * dy_hc + dz_hc * dz_hc;
  const double d2_nc = dx_nc * dx_nc + dy_nc * dy_nc + dz_nc * dz_nc;
  const double d2_no = dx_no * dx_no + dy_no * dy_no + dz_no * dz_no;

  double result = 0.0;
  if (d2_ho < kMinimalDistance2 || d2_hc < kMinimalDistance2 ||
      d2_nc < kMinimalDistance2 || d2_no < kMinimalDistance2) {
    result = kMinHBondEnergy;
  } else {
    const double d_ho = std::sqrt(d2_ho);
    const double d_hc = std::sqrt(d2_hc);
    const double d_nc = std::sqrt(d2_nc);
    const double d_no = std::sqrt(d2_no);
    result = kCouplingConstant / d_ho - kCouplingConstant / d_hc +
             kCouplingConstant / d_nc - kCouplingConstant / d_no;
    result = std::round(result * 1000.0) / 1000.0;
    if (result < kMinHBondEnergy)
      result = kMinHBondEnergy;
  }
  return result;
}

void TitanDSSPAnalyzer::update_best_bond(std::array<BondRecord, 2> &best,
                                         std::size_t acceptor,
                                         double energy) const noexcept {
  if (energy < best[0].energy) {
    best[1] = best[0];
    best[0] = BondRecord{static_cast<int>(acceptor), energy};
  } else if (energy < best[1].energy) {
    best[1] = BondRecord{static_cast<int>(acceptor), energy};
  }
}

void TitanDSSPAnalyzer::compute_hbond_topology(TitanExecutor &executor) {
  const std::size_t n_res = residues_.size();
  if (n_res == 0)
    return;

  for (auto &best : best_acceptors_) {
    best[0] = BondRecord{};
    best[1] = BondRecord{};
  }
  for (auto &flags : helix_flags_) {
    flags[0] = HelixFlag::None;
    flags[1] = HelixFlag::None;
    flags[2] = HelixFlag::None;
  }

  const std::size_t worker_count =
      std::max<std::size_t>(1, std::min<std::size_t>(executor.capacity(), n_res));

  auto worker = [this, n_res](std::size_t begin, std::size_t end) {
    for (std::size_t donor = begin; donor < end; ++donor) {
      if (!donor_valid_[donor] || !has_ca_[donor])
        continue;

      std::array<BondRecord, 2> local_best{};
      const double donor_ca_x = ca_x_[donor];
      const double donor_ca_y = ca_y_[donor];
      const double donor_ca_z = ca_z_[donor];

      std::size_t acceptor = 0;
      for (; acceptor + 4 <= n_res; acceptor += 4) {
        const unsigned within_cutoff =
            titan_simd_dispatch::point_vs_four_cutoff_mask(
                donor_ca_x, donor_ca_y, donor_ca_z, ca_x_.data() + acceptor,
                ca_y_.data() + acceptor, ca_z_.data() + acceptor,
                kMinimalCADistance2, -1.0, true);

        for (std::size_t lane = 0; lane < 4; ++lane) {
          const std::size_t idx = acceptor + lane;
          if (!has_ca_[idx] || !acceptor_valid_[idx] ||
              donor_acceptor_excluded(donor, idx))
            continue;
          if ((within_cutoff & (1u << lane)) == 0u)
            continue;

          update_best_bond(local_best, idx, hbond_energy_scalar(donor, idx));
        }
      }

      for (; acceptor < n_res; ++acceptor) {
        if (!has_ca_[acceptor] || !acceptor_valid_[acceptor] ||
            donor_acceptor_excluded(donor, acceptor))
          continue;
        const double dx = donor_ca_x - ca_x_[acceptor];
        const double dy = donor_ca_y - ca_y_[acceptor];
        const double dz = donor_ca_z - ca_z_[acceptor];
        if (dx * dx + dy * dy + dz * dz > kMinimalCADistance2)
          continue;

        update_best_bond(local_best, acceptor, hbond_energy_scalar(donor, acceptor));
      }

      best_acceptors_[donor] = local_best;
    }
  };

  executor.parallel_for(
      n_res, worker_count, 1,
      [&](std::size_t /*worker_id*/, std::size_t begin, std::size_t end) {
        worker(begin, end);
      });
}

bool TitanDSSPAnalyzer::test_bond(std::size_t donor,
                                  std::size_t acceptor) const noexcept {
  if (donor >= best_acceptors_.size())
    return false;
  for (const auto &bond : best_acceptors_[donor]) {
    if (bond.residue == static_cast<int>(acceptor) &&
        bond.energy < hbond_cutoff_) {
      return true;
    }
  }
  return false;
}

TitanDSSPAnalyzer::BridgeType
TitanDSSPAnalyzer::test_bridge(std::size_t r1, std::size_t r2) const noexcept {
  if (r1 == 0 || r2 == 0 || r1 + 1 >= residues_.size() ||
      r2 + 1 >= residues_.size()) {
    return BridgeType::None;
  }
  if (!no_chain_break_between(r1 - 1, r1 + 1) ||
      !no_chain_break_between(r2 - 1, r2 + 1)) {
    return BridgeType::None;
  }

  const bool parallel =
      (test_bond(r1 + 1, r2) && test_bond(r2, r1 - 1)) ||
      (test_bond(r2 + 1, r1) && test_bond(r1, r2 - 1));
  if (parallel)
    return BridgeType::Parallel;

  const bool antiparallel =
      (test_bond(r1 + 1, r2 - 1) && test_bond(r2 + 1, r1 - 1)) ||
      (test_bond(r2, r1) && test_bond(r1, r2));
  return antiparallel ? BridgeType::AntiParallel : BridgeType::None;
}

void TitanDSSPAnalyzer::classify_current_frame() {
  const std::size_t n_res = residues_.size();
  if (n_res == 0)
    return;

  std::fill(frame_states_.begin(), frame_states_.end(), State::C);
  current_frame_bridge_pairs_.clear();
  current_frame_ladders_.clear();
  for (auto &flags : helix_flags_) {
    flags[0] = HelixFlag::None;
    flags[1] = HelixFlag::None;
    flags[2] = HelixFlag::None;
  }

  std::vector<Bridge> bridges;
  bridges.reserve(n_res / 2 + 1);

  for (std::size_t i = 1; i + 1 < n_res; ++i) {
    if (!has_ca_[i])
      continue;
    for (std::size_t j = i + 1; j + 1 < n_res; ++j) {
      if (!has_ca_[j])
        continue;
      const double dx = ca_x_[i] - ca_x_[j];
      const double dy = ca_y_[i] - ca_y_[j];
      const double dz = ca_z_[i] - ca_z_[j];
      if (dx * dx + dy * dy + dz * dz > kMinimalCADistance2)
        continue;

      const BridgeType type = test_bridge(i, j);
      if (type == BridgeType::None)
        continue;

      const long long ordinal_i =
          static_cast<long long>(residues_[i].local_index + 1);
      const long long ordinal_j =
          static_cast<long long>(residues_[j].local_index + 1);
      const long long registry =
          type == BridgeType::Parallel ? ordinal_j - ordinal_i
                                       : ordinal_i + ordinal_j;
      current_frame_bridge_pairs_.push_back(
          FrameBridgePair{i, j, type, registry});

      bool found = false;
      for (auto &bridge : bridges) {
        if (bridge.type != type || bridge.i.empty() ||
            i != bridge.i.back() + 1) {
          continue;
        }

        if (type == BridgeType::Parallel) {
          if (!bridge.j.empty() && bridge.j.back() + 1 == j) {
            bridge.i.push_back(i);
            bridge.j.push_back(j);
            found = true;
            break;
          }
        } else {
          if (!bridge.j.empty() && bridge.j.front() == j + 1) {
            bridge.i.push_back(i);
            bridge.j.push_front(j);
            found = true;
            break;
          }
        }
      }

      if (!found) {
        Bridge bridge;
        bridge.type = type;
        bridge.i.push_back(i);
        bridge.j.push_back(j);
        bridges.push_back(std::move(bridge));
      }
    }
  }

  std::sort(bridges.begin(), bridges.end(),
            [](const Bridge &lhs, const Bridge &rhs) {
              if (lhs.i.empty() || rhs.i.empty())
                return lhs.i.size() < rhs.i.size();
              return lhs.i.front() < rhs.i.front();
            });

  for (std::size_t i = 0; i < bridges.size(); ++i) {
    for (std::size_t j = i + 1; j < bridges.size(); ++j) {
      const long long ibi = static_cast<long long>(bridges[i].i.front());
      const long long iei = static_cast<long long>(bridges[i].i.back());
      const long long jbi = static_cast<long long>(bridges[i].j.front());
      const long long jei = static_cast<long long>(bridges[i].j.back());
      const long long ibj = static_cast<long long>(bridges[j].i.front());
      const long long iej = static_cast<long long>(bridges[j].i.back());
      const long long jbj = static_cast<long long>(bridges[j].j.front());
      const long long jej = static_cast<long long>(bridges[j].j.back());

      if (bridges[i].type != bridges[j].type)
        continue;
      if (!no_chain_break_between(
              static_cast<std::size_t>(std::min(ibi, ibj)),
              static_cast<std::size_t>(std::max(iei, iej))) ||
          !no_chain_break_between(
              static_cast<std::size_t>(std::min(jbi, jbj)),
              static_cast<std::size_t>(std::max(jei, jej)))) {
        continue;
      }
      if (ibj - iei >= 6 || (iei >= ibj && ibi <= iej))
        continue;

      bool bulge = false;
      if (bridges[i].type == BridgeType::Parallel) {
        bulge = (((jbj - jei) < 6 && (ibj - iei) < 3) || ((jbj - jei) < 3));
      } else {
        bulge = (((jbi - jej) < 6 && (ibj - iei) < 3) || ((jbi - jej) < 3));
      }
      if (!bulge)
        continue;

      bridges[i].i.insert(bridges[i].i.end(), bridges[j].i.begin(),
                          bridges[j].i.end());
      if (bridges[i].type == BridgeType::Parallel) {
        bridges[i].j.insert(bridges[i].j.end(), bridges[j].j.begin(),
                            bridges[j].j.end());
      } else {
        bridges[i].j.insert(bridges[i].j.begin(), bridges[j].j.begin(),
                            bridges[j].j.end());
      }
      bridges.erase(bridges.begin() + static_cast<std::ptrdiff_t>(j));
      --j;
    }
  }

  for (const auto &bridge : bridges) {
    if (bridge.i.empty() || bridge.j.empty())
      continue;
    const bool interchain =
        residues_[bridge.i.front()].chainnum !=
        residues_[bridge.j.front()].chainnum;
    current_frame_ladders_.push_back(
        FrameLadder{bridge.type,
                    std::min(bridge.i.size(), bridge.j.size()), interchain});
  }

  for (const auto &bridge : bridges) {
    const State ss = (bridge.i.size() > 1) ? State::E : State::B;
    for (std::size_t idx = bridge.i.front(); idx <= bridge.i.back(); ++idx) {
      if (frame_states_[idx] != State::E)
        frame_states_[idx] = ss;
    }
    for (std::size_t idx = bridge.j.front(); idx <= bridge.j.back(); ++idx) {
      if (frame_states_[idx] != State::E)
        frame_states_[idx] = ss;
    }
  }

  const std::array<std::size_t, 3> strides{3, 4, 5};
  for (std::size_t helix_idx = 0; helix_idx < strides.size(); ++helix_idx) {
    const std::size_t stride = strides[helix_idx];
    for (std::size_t i = 0; i + stride < n_res; ++i) {
      if (!no_chain_break_between(i, i + stride))
        continue;
      if (!test_bond(i + stride, i))
        continue;

      helix_flags_[i + stride][helix_idx] = HelixFlag::End;
      for (std::size_t j = i + 1; j < i + stride; ++j) {
        if (helix_flags_[j][helix_idx] == HelixFlag::None)
          helix_flags_[j][helix_idx] = HelixFlag::Middle;
      }

      if (helix_flags_[i][helix_idx] == HelixFlag::End)
        helix_flags_[i][helix_idx] = HelixFlag::StartAndEnd;
      else
        helix_flags_[i][helix_idx] = HelixFlag::Start;
    }
  }

  auto is_start = [](HelixFlag flag) {
    return flag == HelixFlag::Start || flag == HelixFlag::StartAndEnd;
  };

  for (std::size_t i = 1; i + 4 < n_res; ++i) {
    if (is_start(helix_flags_[i][1]) && is_start(helix_flags_[i - 1][1])) {
      for (std::size_t j = i; j <= i + 3; ++j)
        frame_states_[j] = State::H;
    }
  }

  for (std::size_t i = 1; i + 3 < n_res; ++i) {
    if (is_start(helix_flags_[i][0]) && is_start(helix_flags_[i - 1][0])) {
      bool empty = true;
      for (std::size_t j = i; empty && j <= i + 2; ++j)
        empty = frame_states_[j] == State::C || frame_states_[j] == State::G;
      if (empty) {
        for (std::size_t j = i; j <= i + 2; ++j)
          frame_states_[j] = State::G;
      }
    }
  }

  for (std::size_t i = 1; i + 5 < n_res; ++i) {
    if (is_start(helix_flags_[i][2]) && is_start(helix_flags_[i - 1][2])) {
      bool empty = true;
      for (std::size_t j = i; empty && j <= i + 4; ++j)
        empty = frame_states_[j] == State::C || frame_states_[j] == State::I ||
                frame_states_[j] == State::H;
      if (empty) {
        for (std::size_t j = i; j <= i + 4; ++j)
          frame_states_[j] = State::I;
      }
    }
  }

  for (std::size_t i = 1; i + 1 < n_res; ++i) {
    if (frame_states_[i] != State::C)
      continue;

    bool is_turn = false;
    for (std::size_t helix_idx = 0; helix_idx < strides.size() && !is_turn;
         ++helix_idx) {
      const std::size_t stride = strides[helix_idx];
      for (std::size_t k = 1; k < stride && !is_turn; ++k) {
        if (i >= k)
          is_turn = is_start(helix_flags_[i - k][helix_idx]);
      }
    }

    if (is_turn)
      frame_states_[i] = State::T;
    else if (has_kappa_[i] && kappa_deg_[i] > 70.0)
      frame_states_[i] = State::S;
  }
}

void TitanDSSPAnalyzer::process_frame(const Multi_MOL &mol, std::size_t snap_idx,
                                      TitanExecutor &executor) {
  if (!enabled_ || !initialized_ || snap_idx >= mol.m_mol.size())
    return;

  prepare_backbone_coordinates(mol.m_mol[snap_idx]);
  compute_hbond_topology(executor);
  classify_current_frame();
  bridge_frames_.push_back(current_frame_bridge_pairs_);
  ladder_frames_.push_back(current_frame_ladders_);

  FrameProfile frame_profile{};
  frame_profile.fill(0.0);

  for (std::size_t residue_idx = 0; residue_idx < frame_states_.size();
       ++residue_idx) {
    const std::size_t idx = state_index(frame_states_[residue_idx]);
    current_traj_counts_[residue_idx][idx]++;
    frame_profile[idx] += 1.0;
  }
  if (!frame_states_.empty()) {
    const double inv_res = 1.0 / static_cast<double>(frame_states_.size());
    for (double &value : frame_profile)
      value *= inv_res;
  }
  current_traj_frame_profiles_.push_back(frame_profile);
  ++current_traj_frames_;
  ++total_frames_;
}

void TitanDSSPAnalyzer::finalize_trajectory() {
  if (!enabled_ || current_traj_frames_ == 0 || residues_.empty())
    return;

  std::vector<double> fractions(residues_.size() * kStateCount, 0.0);
  const double inv_frames = 1.0 / static_cast<double>(current_traj_frames_);

  for (std::size_t residue_idx = 0; residue_idx < residues_.size();
       ++residue_idx) {
    for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
      fractions[flat_index(residue_idx, state_idx)] =
          static_cast<double>(current_traj_counts_[residue_idx][state_idx]) *
          inv_frames;
    }
    current_traj_counts_[residue_idx].fill(0);
  }

  trajectory_fractions_.push_back(std::move(fractions));
  trajectory_frame_profiles_.push_back(current_traj_frame_profiles_);
  current_traj_frame_profiles_.clear();
  trajectory_frame_counts_.push_back(current_traj_frames_);
  current_traj_frames_ = 0;
}

void TitanDSSPAnalyzer::compute_mean_sem(std::vector<double> &mean,
                                         std::vector<double> &sem) const {
  const std::size_t n_traj = trajectory_fractions_.size();
  const std::size_t n_values = residues_.size() * kStateCount;

  mean.assign(n_values, 0.0);
  sem.assign(n_values, 0.0);
  if (n_traj == 0 || n_values == 0)
    return;

  for (const auto &traj : trajectory_fractions_) {
    for (std::size_t idx = 0; idx < traj.size(); ++idx)
      mean[idx] += traj[idx];
  }
  for (double &value : mean)
    value /= static_cast<double>(n_traj);

  if (n_traj <= 1)
    return;

  for (const auto &traj : trajectory_fractions_) {
    for (std::size_t idx = 0; idx < traj.size(); ++idx) {
      const double diff = traj[idx] - mean[idx];
      sem[idx] += diff * diff;
    }
  }
  for (double &value : sem) {
    value = std::sqrt(value / static_cast<double>(n_traj - 1));
    value /= std::sqrt(static_cast<double>(n_traj));
  }
}

std::string
TitanDSSPAnalyzer::build_consensus(const std::vector<double> &mean) const {
  std::string consensus;
  consensus.reserve(residues_.size());

  for (std::size_t residue_idx = 0; residue_idx < residues_.size();
       ++residue_idx) {
    std::size_t best_state = 0;
    double best_fraction = mean[flat_index(residue_idx, 0)];
    for (std::size_t state_idx = 1; state_idx < kStateCount; ++state_idx) {
      const double candidate = mean[flat_index(residue_idx, state_idx)];
      if (candidate > best_fraction) {
        best_fraction = candidate;
        best_state = state_idx;
      }
    }
    consensus.push_back(state_to_char(static_cast<State>(best_state)));
  }

  return consensus;
}

void TitanDSSPAnalyzer::export_results(const std::string &filename,
                                       std::fstream &log) const {
  if (!enabled_ || trajectory_fractions_.empty() || residues_.empty())
    return;

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("[DSSP] Could not open output file: " + filename);
    return;
  }

  const std::size_t n_traj = trajectory_fractions_.size();
  const std::size_t n_res = residues_.size();

  std::vector<double> mean, sem;
  compute_mean_sem(mean, sem);
  const std::string consensus = build_consensus(mean);

  out << "# TITAN DSSP-like secondary-structure summary\n";
  out << "# molecule = " << label_ << "\n";
  out << "# residues = " << n_res << "\n";
  out << "# trajectories = " << n_traj << "\n";
  out << "# total_frames = " << total_frames_ << "\n";
  out << "# algorithm = DSSP-inspired (CA cutoff 9A, Kabsch-Sander bond "
         "energy, TestBridge/TestBond helix-sheet rules)\n";
  out << "# error = sem_across_input_trajectories\n";
  out << "# states = H(alpha) B(bridge) E(strand) G(310) I(pi) T(turn) "
         "S(bend) C(coil)\n";
  out << "# martinize2_dssp = " << consensus << "\n";
  out << "# columns = res_idx chain resnr resname dominant dominant_mean "
         "dominant_sem H_mean H_sem B_mean B_sem E_mean E_sem G_mean G_sem "
         "I_mean I_sem T_mean T_sem S_mean S_sem C_mean C_sem\n";
  out << std::fixed << std::setprecision(4);

  for (std::size_t residue_idx = 0; residue_idx < n_res; ++residue_idx) {
    std::size_t best_state = 0;
    double best_fraction = mean[flat_index(residue_idx, 0)];
    for (std::size_t state_idx = 1; state_idx < kStateCount; ++state_idx) {
      const double candidate = mean[flat_index(residue_idx, state_idx)];
      if (candidate > best_fraction) {
        best_fraction = candidate;
        best_state = state_idx;
      }
    }

    const auto &residue = residues_[residue_idx];
    out << residue.ordinal << ' ' << residue.chain << ' ' << residue.resnr
        << ' ' << residue.resname << ' '
        << state_to_char(static_cast<State>(best_state)) << ' '
        << mean[flat_index(residue_idx, best_state)] << ' '
        << sem[flat_index(residue_idx, best_state)];

    for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
      out << ' ' << mean[flat_index(residue_idx, state_idx)] << ' '
          << sem[flat_index(residue_idx, state_idx)];
    }
    out << '\n';
  }

  if (log.is_open()) {
    TitanUI::log_kv(log, "DSSP " + label_, filename);
    TitanUI::log_kv(log, "DSSP " + label_ + " trajectories",
                    std::to_string(n_traj));
    TitanUI::log_kv(log, "DSSP " + label_ + " martinize2", consensus);
  }
}

void TitanDSSPAnalyzer::export_content_profile(const std::string &filename,
                                               std::fstream &log) const {
  if (!enabled_ || trajectory_fractions_.empty() || residues_.empty())
    return;

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("[DSSP] Could not open output file: " + filename);
    return;
  }

  const std::size_t n_traj = trajectory_fractions_.size();
  const std::size_t n_res = residues_.size();
  std::vector<double> mean, sem;
  compute_mean_sem(mean, sem);

  std::vector<double> protein_mean(kStateCount, 0.0);
  std::vector<double> protein_sem(kStateCount, 0.0);
  std::vector<double> traj_protein_content(n_traj * kStateCount, 0.0);

  const double inv_res = 1.0 / static_cast<double>(n_res);
  for (std::size_t traj_idx = 0; traj_idx < n_traj; ++traj_idx) {
    const auto &traj = trajectory_fractions_[traj_idx];
    for (std::size_t residue_idx = 0; residue_idx < n_res; ++residue_idx) {
      for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
        traj_protein_content[traj_idx * kStateCount + state_idx] +=
            traj[flat_index(residue_idx, state_idx)];
      }
    }
    for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx)
      traj_protein_content[traj_idx * kStateCount + state_idx] *= inv_res;
  }

  for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
    for (std::size_t traj_idx = 0; traj_idx < n_traj; ++traj_idx)
      protein_mean[state_idx] +=
          traj_protein_content[traj_idx * kStateCount + state_idx];
    protein_mean[state_idx] /= static_cast<double>(n_traj);
  }

  if (n_traj > 1) {
    for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
      for (std::size_t traj_idx = 0; traj_idx < n_traj; ++traj_idx) {
        const double diff =
            traj_protein_content[traj_idx * kStateCount + state_idx] -
            protein_mean[state_idx];
        protein_sem[state_idx] += diff * diff;
      }
      protein_sem[state_idx] =
          std::sqrt(protein_sem[state_idx] / static_cast<double>(n_traj - 1));
      protein_sem[state_idx] /= std::sqrt(static_cast<double>(n_traj));
    }
  }

  out << "# TITAN DSSP secondary-structure content profile\n";
  out << "# molecule = " << label_ << "\n";
  out << "# residues = " << n_res << "\n";
  out << "# trajectories = " << n_traj << "\n";
  out << "# total_frames = " << total_frames_ << "\n";
  out << "# meaning: protein_mean/protein_sem = average residue occupancy of a\n";
  out << "# state across the whole molecule, aggregated over trajectories\n";
  out << "# residue_mean/residue_sem = occupancy of a state at one residue\n";
  out << "# error = sem_across_input_trajectories\n";
  out << "# states = H(alpha) B(bridge) E(strand) G(310) I(pi) T(turn) "
         "S(bend) C(coil)\n";
  out << "# state_summary columns = state state_name protein_mean protein_sem\n";
  out << std::fixed << std::setprecision(4);

  for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
    const State state = static_cast<State>(state_idx);
    out << state_to_char(state) << ' ' << state_name(state) << ' '
        << protein_mean[state_idx] << ' ' << protein_sem[state_idx] << '\n';
  }

  for (std::size_t state_idx = 0; state_idx < kStateCount; ++state_idx) {
    const State state = static_cast<State>(state_idx);
    out << "\n# residue_content state = " << state_to_char(state) << ' '
        << state_name(state) << '\n';
    out << "# columns = res_idx chain resnr resname residue_mean residue_sem\n";
    for (std::size_t residue_idx = 0; residue_idx < n_res; ++residue_idx) {
      const auto &residue = residues_[residue_idx];
      out << residue.ordinal << ' ' << residue.chain << ' ' << residue.resnr
          << ' ' << residue.resname << ' '
          << mean[flat_index(residue_idx, state_idx)] << ' '
          << sem[flat_index(residue_idx, state_idx)] << '\n';
    }
  }

  if (log.is_open()) {
    TitanUI::log_kv(log, "DSSP " + label_ + " content", filename);
  }
}

void TitanDSSPAnalyzer::export_segment_content(const std::string &filename,
                                               std::fstream &log) const {
  if (!enabled_ || !segment_config_loaded_ || trajectory_fractions_.empty() ||
      resolved_segments_.empty())
    return;

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("[DSSP] Could not open output file: " + filename);
    return;
  }
  out << "# Segment definitions are read from " << segment_config_file_ << "\n";
  out << "# beta_total=B+E; helix_total=H+G+I\n";
  out << "molecule,entity,segment,coordinate,begin,end,residue_count,state,"
         "mean,sem,trajectories\n";
  out << std::setprecision(12);

  struct StateGroup {
    std::string name;
    std::vector<std::size_t> states;
  };
  std::vector<StateGroup> groups;
  for (std::size_t state = 0; state < kStateCount; ++state) {
    groups.push_back(
        StateGroup{std::string(1, state_to_char(static_cast<State>(state))),
                   {state}});
  }
  groups.push_back(StateGroup{"beta_total",
                              {state_index(State::B), state_index(State::E)}});
  groups.push_back(StateGroup{
      "helix_total", {state_index(State::H), state_index(State::G),
                      state_index(State::I)}});

  for (const ResolvedSegment &segment : resolved_segments_) {
    for (const StateGroup &group : groups) {
      std::vector<double> trajectory_values;
      trajectory_values.reserve(trajectory_fractions_.size());
      if (!segment.residue_indices.empty()) {
        for (const auto &trajectory : trajectory_fractions_) {
          double value = 0.0;
          for (const std::size_t residue : segment.residue_indices) {
            for (const std::size_t state : group.states)
              value += trajectory[flat_index(residue, state)];
          }
          value /= static_cast<double>(segment.residue_indices.size());
          trajectory_values.push_back(value);
        }
      }

      double mean = std::numeric_limits<double>::quiet_NaN();
      double sem = std::numeric_limits<double>::quiet_NaN();
      if (!trajectory_values.empty()) {
        mean = std::accumulate(trajectory_values.begin(),
                               trajectory_values.end(), 0.0) /
               static_cast<double>(trajectory_values.size());
        if (trajectory_values.size() > 1) {
          double ss = 0.0;
          for (const double value : trajectory_values) {
            const double delta = value - mean;
            ss += delta * delta;
          }
          sem = std::sqrt(
                    ss / static_cast<double>(trajectory_values.size() - 1)) /
                std::sqrt(static_cast<double>(trajectory_values.size()));
        }
      }
      out << label_ << ',' << segment.entity << ',' << segment.segment << ','
          << segment.coordinate << ',' << segment.begin << ',' << segment.end
          << ',' << segment.residue_indices.size() << ',' << group.name << ','
          << mean << ',' << sem << ',' << trajectory_values.size() << '\n';
    }
  }

  if (log.is_open())
    TitanUI::log_kv(log, "DSSP " + label_ + " segments", filename);
}

void TitanDSSPAnalyzer::export_bridge_registry(const std::string &filename,
                                               std::fstream &log) const {
  if (!enabled_ || bridge_frames_.empty() || residues_.empty())
    return;

  using Key = std::tuple<std::size_t, std::size_t, int, long long>;
  std::map<Key, std::size_t> counts;
  for (const auto &frame : bridge_frames_) {
    for (const FrameBridgePair &pair : frame) {
      counts[{pair.residue_i, pair.residue_j,
              static_cast<int>(pair.type), pair.registry}]++;
    }
  }

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("[DSSP] Could not open output file: " + filename);
    return;
  }
  out << "# Parallel registry=ordinal_j-ordinal_i; antiparallel "
         "registry=ordinal_i+ordinal_j.\n";
  out << "orientation,registry,interchain,entity_i,segment_i,chain_i,"
         "ordinal_i,resnr_i,resname_i,entity_j,segment_j,chain_j,ordinal_j,"
         "resnr_j,resname_j,contact_frames,frame_occupancy\n";
  out << std::setprecision(12);
  for (const auto &[key, count] : counts) {
    const auto [i, j, type_value, registry] = key;
    const ResidueBackbone &first = residues_[i];
    const ResidueBackbone &second = residues_[j];
    const ResidueAnnotation empty_annotation{};
    const ResidueAnnotation &annotation_i =
        i < residue_annotations_.size() ? residue_annotations_[i]
                                        : empty_annotation;
    const ResidueAnnotation &annotation_j =
        j < residue_annotations_.size() ? residue_annotations_[j]
                                        : empty_annotation;
    const BridgeType type = static_cast<BridgeType>(type_value);
    out << bridge_type_name(type) << ',' << registry << ','
        << (first.chainnum != second.chainnum ? "true" : "false") << ','
        << annotation_i.entity << ',' << annotation_i.segment << ','
        << first.chain << ',' << first.local_index + 1 << ',' << first.resnr
        << ',' << first.resname << ',' << annotation_j.entity << ','
        << annotation_j.segment << ',' << second.chain << ','
        << second.local_index + 1 << ',' << second.resnr << ','
        << second.resname << ',' << count << ','
        << static_cast<double>(count) /
               static_cast<double>(bridge_frames_.size())
        << '\n';
  }

  if (log.is_open())
    TitanUI::log_kv(log, "DSSP " + label_ + " bridges", filename);
}

void TitanDSSPAnalyzer::export_ladder_distribution(
    const std::string &filename, std::fstream &log) const {
  if (!enabled_ || ladder_frames_.empty())
    return;

  using Key = std::tuple<int, std::size_t, bool>;
  std::map<Key, std::size_t> counts;
  std::size_t total_ladders = 0;
  for (const auto &frame : ladder_frames_) {
    for (const FrameLadder &ladder : frame) {
      counts[{static_cast<int>(ladder.type), ladder.length,
              ladder.interchain}]++;
      ++total_ladders;
    }
  }

  std::ofstream out(filename);
  if (!out.is_open()) {
    TitanUI::print_warning("[DSSP] Could not open output file: " + filename);
    return;
  }
  out << "orientation,ladder_length,interchain,count,count_per_frame,"
         "fraction_of_ladders\n";
  out << std::setprecision(12);
  for (const auto &[key, count] : counts) {
    const auto [type_value, length, interchain] = key;
    out << bridge_type_name(static_cast<BridgeType>(type_value)) << ','
        << length << ',' << (interchain ? "true" : "false") << ',' << count
        << ','
        << static_cast<double>(count) /
               static_cast<double>(ladder_frames_.size())
        << ','
        << (total_ladders > 0
                ? static_cast<double>(count) /
                      static_cast<double>(total_ladders)
                : 0.0)
        << '\n';
  }

  if (log.is_open())
    TitanUI::log_kv(log, "DSSP " + label_ + " ladders", filename);
}

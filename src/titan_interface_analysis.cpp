#include "titan_interface_analysis.h"

#include "BoxMatrix.h"
#include "info.h"
#include "titan_executor.h"
#include "titan_sasa.h"
#include "titan_ui.h"
#include "titan_uncertainty.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

namespace {

double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

double sasa_radius(const Multi_MOL &mol, std::size_t atom_index) {
  if (const auto topology_radius =
          mol.topology.vdw_radius(static_cast<std::uint32_t>(atom_index));
      topology_radius.has_value() && std::isfinite(*topology_radius) &&
      *topology_radius > 0.0) {
    return *topology_radius;
  }
  if (atom_index < mol.parameters.radius.size() &&
      std::isfinite(mol.parameters.radius[atom_index]) &&
      mol.parameters.radius[atom_index] > 0.0) {
    return mol.parameters.radius[atom_index];
  }
  return quiet_nan();
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos)
    return value;
  std::string escaped{"\""};
  for (const char c : value) {
    if (c == '"')
      escaped += "\"\"";
    else
      escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}

double mean_value(const std::vector<double> &values) {
  if (values.empty())
    return quiet_nan();
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

std::vector<double>
block_means(const std::vector<double> &values,
            const std::vector<titan_uncertainty::Block> &blocks) {
  std::vector<double> means;
  means.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block.end <= block.begin || block.end > values.size())
      continue;
    double sum = 0.0;
    for (std::size_t i = block.begin; i < block.end; ++i)
      sum += values[i];
    means.push_back(sum / static_cast<double>(block.end - block.begin));
  }
  return means;
}

} // namespace

std::size_t TitanInterfaceAnalyzer::ResidueIdentityHash::operator()(
    const ResidueIdentity &key) const noexcept {
  std::size_t seed = static_cast<std::size_t>(key.molecule);
  seed ^= key.chain_number + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  seed ^= key.residue_number + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  return seed;
}

void TitanInterfaceAnalyzer::configure(const InterfaceAnalysisArgs &args,
                                       const SegmentArgs &segments) {
  requested_ = args.flag_interface_analysis;
  config_file_ = segments.config_file;
  cutoff_a_ = args.cutoff_ang;
  state_min_contacts_ = args.state_min_contacts;
  sasa_stride_ = args.sasa_stride;
  sasa_points_ = args.sasa_points;
  contact_mode_ = titan_contact::parse_contact_mode(args.contact_mode);
  enabled_ = requested_ && initialized_;
}

void TitanInterfaceAnalyzer::init(const Multi_MOL &mol1,
                                  const Multi_MOL *mol2,
                                  const arguments &args,
                                  std::fstream &log) {
  if (!requested_)
    return;
  if (initialized_)
    return;
  if (config_file_.empty())
    throw std::runtime_error(
        "Interface analysis requires a segment configuration file.");

  config_ = titan_segments::load_config(config_file_);
  build_runtime(mol1, mol2, log);
  initialized_ = true;
  enabled_ = true;

  TitanUI::log_kv(log, "Interface config", config_file_);
  TitanUI::log_kv(log, "Interface contact mode",
                  titan_contact::contact_mode_name(contact_mode_));
  TitanUI::log_kv(log, "Interface cutoff",
                  std::to_string(cutoff_a_) + " A");
  TitanUI::log_kv(log, "Interface uncertainty", args.uncertainty.mode);
}

void TitanInterfaceAnalyzer::build_runtime(const Multi_MOL &mol1,
                                           const Multi_MOL *mol2,
                                           std::fstream &log) {
  entities_.clear();
  entities_.reserve(config_.entities.size());
  for (const titan_segments::Entity &definition : config_.entities)
    entities_.push_back(EntityRuntime{definition, {}, {}});

  auto entity_index = [&](const titan_segments::Entity *entity) -> int {
    if (entity == nullptr)
      return -1;
    for (std::size_t i = 0; i < entities_.size(); ++i) {
      if (entities_[i].definition.name == entity->name)
        return static_cast<int>(i);
    }
    return -1;
  };

  core_a_index_ = entity_index(config_.entity_for_role("core_a"));
  core_b_index_ = entity_index(config_.entity_for_role("core_b"));
  partner_index_ = entity_index(config_.entity_for_role("partner"));
  if (core_a_index_ < 0 || core_b_index_ < 0 ||
      core_a_index_ == core_b_index_) {
    throw std::runtime_error(
        "Interface analysis requires distinct core_a and core_b roles.");
  }

  const bool needs_mol1 =
      std::any_of(entities_.begin(), entities_.end(),
                  [](const EntityRuntime &entity) {
                    return entity.definition.molecule == 1;
                  });
  const bool needs_mol2 =
      std::any_of(entities_.begin(), entities_.end(),
                  [](const EntityRuntime &entity) {
                    return entity.definition.molecule == 2;
                  });
  if (needs_mol1) {
    geometry_mol1_active_ = geometry_mol1_.build(mol1, 1, contact_mode_);
    if (!geometry_mol1_active_)
      throw std::runtime_error(
          "Interface analysis could not build mol1 contact geometry.");
    build_entities_for_molecule(mol1, 1);
    build_atom_membership(mol1, 1);
  }
  if (needs_mol2) {
    if (mol2 == nullptr)
      throw std::runtime_error(
          "Interface config references mol2, but molecule 2 is unavailable.");
    geometry_mol2_active_ = geometry_mol2_.build(*mol2, 2, contact_mode_);
    if (!geometry_mol2_active_)
      throw std::runtime_error(
          "Interface analysis could not build mol2 contact geometry.");
    build_entities_for_molecule(*mol2, 2);
    build_atom_membership(*mol2, 2);
  }

  for (const int role_index :
       {core_a_index_, core_b_index_, partner_index_}) {
    if (role_index < 0)
      continue;
    const EntityRuntime &entity =
        entities_[static_cast<std::size_t>(role_index)];
    if (entity.residues.empty()) {
      throw std::runtime_error("Interface entity '" +
                               entity.definition.name +
                               "' matched no contact residues.");
    }
    if (entity.atoms.empty()) {
      throw std::runtime_error("Interface entity '" +
                               entity.definition.name + "' matched no atoms.");
    }
  }

  runtime_points_.resize(entities_.size());
  build_segment_channels();

  sasa_enabled_ = sasa_stride_ > 0;
  if (sasa_enabled_) {
    std::size_t invalid_radius_count = 0;
    std::string invalid_radius_example;
    auto radii_valid = [&](const EntityRuntime &entity) {
      for (const AtomRef &ref : entity.atoms) {
        const Multi_MOL &mol = ref.molecule == 1 ? mol1 : *mol2;
        if (!std::isfinite(sasa_radius(mol, ref.atom_index))) {
          ++invalid_radius_count;
          if (invalid_radius_example.empty() &&
              ref.atom_index < mol.info.size()) {
            const mol_info &atom = mol.info[ref.atom_index];
            invalid_radius_example =
                "mol" + std::to_string(ref.molecule) + " atom " +
                std::to_string(ref.atom_index + 1) + " " + atom.get_atom() +
                " " + atom.get_res_name() +
                std::to_string(atom.get_resnr());
          }
          return false;
        }
      }
      return true;
    };
    for (const int role_index :
         {core_a_index_, core_b_index_, partner_index_}) {
      if (role_index >= 0 &&
          !radii_valid(entities_[static_cast<std::size_t>(role_index)])) {
        sasa_enabled_ = false;
        TitanUI::print_warning(
            "[Interface] SASA/BSA disabled because at least one selected atom "
            "has no positive topology or parameter radius.");
        TitanUI::log_kv(
            log, "Interface SASA",
            "disabled: invalid radii=" +
                std::to_string(invalid_radius_count) +
                (invalid_radius_example.empty()
                     ? std::string{}
                     : ", first=" + invalid_radius_example));
        break;
      }
    }
  }

  TitanUI::log_kv(log, "Interface core_a",
                  entities_[static_cast<std::size_t>(core_a_index_)]
                      .definition.name);
  TitanUI::log_kv(log, "Interface core_b",
                  entities_[static_cast<std::size_t>(core_b_index_)]
                      .definition.name);
  TitanUI::log_kv(
      log, "Interface partner",
      partner_index_ >= 0
          ? entities_[static_cast<std::size_t>(partner_index_)].definition.name
          : "none");
  TitanUI::log_kv(log, "Interface SASA",
                  sasa_enabled_
                      ? "enabled every " + std::to_string(sasa_stride_) +
                            " valid frames"
                      : "disabled");
}

void TitanInterfaceAnalyzer::build_entities_for_molecule(
    const Multi_MOL &mol, std::uint8_t molecule) {
  const auto &geometry =
      molecule == 1 ? geometry_mol1_ : geometry_mol2_;
  for (const auto &selection : geometry.selections()) {
    if (selection.atoms.empty())
      continue;
    const std::size_t atom_index = selection.atoms.front();
    if (atom_index >= mol.info.size())
      throw std::runtime_error(
          "Interface contact geometry contains an invalid atom index.");

    titan_segments::ResidueDescriptor descriptor;
    descriptor.molecule = molecule;
    descriptor.chain_number = selection.residue.chainnum;
    descriptor.chain = selection.residue.chain;
    descriptor.residue_number = selection.residue.resnr;
    descriptor.internal_residue_number =
        mol.info[atom_index].get_internal_resnr();
    descriptor.ordinal = selection.residue.ordinal;
    descriptor.residue_name = selection.residue.resname;

    std::vector<std::size_t> matches;
    for (std::size_t i = 0; i < entities_.size(); ++i) {
      if (titan_segments::entity_matches(entities_[i].definition, descriptor))
        matches.push_back(i);
    }
    if (matches.empty())
      continue;
    if (matches.size() > 1) {
      throw std::runtime_error(
          "Interface entities overlap at molecule " +
          std::to_string(static_cast<int>(molecule)) + ", chain '" +
          descriptor.chain + "', residue " +
          std::to_string(descriptor.residue_number) + ".");
    }

    EntityRuntime &runtime = entities_[matches.front()];
    const auto segments =
        config_.segments_for_entity(runtime.definition.name);
    std::vector<const titan_segments::Segment *> segment_matches;
    for (const titan_segments::Segment *segment : segments) {
      if (titan_segments::segment_matches(*segment, descriptor))
        segment_matches.push_back(segment);
    }
    if (segment_matches.size() > 1) {
      throw std::runtime_error(
          "Segments overlap for entity '" + runtime.definition.name +
          "' at ordinal " + std::to_string(descriptor.ordinal) + ".");
    }

    EntityResidue residue;
    residue.descriptor = descriptor;
    residue.segment =
        segment_matches.empty()
            ? (segments.empty() ? "all" : "unassigned")
            : segment_matches.front()->name;
    const std::size_t residue_index = runtime.residues.size();
    runtime.residues.push_back(std::move(residue));

    const ResidueIdentity identity{molecule, descriptor.chain_number,
                                   descriptor.residue_number};
    if (!residue_locations_
             .emplace(identity,
                      ResidueLocation{matches.front(), residue_index})
             .second) {
      throw std::runtime_error(
          "Interface residue identity is not unique in the selected topology.");
    }
  }
}

void TitanInterfaceAnalyzer::build_atom_membership(const Multi_MOL &mol,
                                                   std::uint8_t molecule) {
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> ordinal_by_key;
  std::map<std::size_t, std::size_t> next_ordinal_by_chain;
  for (std::size_t atom_index = 0; atom_index < mol.info.size(); ++atom_index) {
    const mol_info &atom = mol.info[atom_index];
    const auto key = std::make_pair(atom.get_chainnum(), atom.get_resnr());
    auto ordinal_it = ordinal_by_key.find(key);
    if (ordinal_it == ordinal_by_key.end()) {
      const std::size_t ordinal = ++next_ordinal_by_chain[key.first];
      ordinal_it = ordinal_by_key.emplace(key, ordinal).first;
    }

    titan_segments::ResidueDescriptor descriptor;
    descriptor.molecule = molecule;
    descriptor.chain_number = atom.get_chainnum();
    descriptor.chain = atom.get_chain();
    descriptor.residue_number = atom.get_resnr();
    descriptor.internal_residue_number = atom.get_internal_resnr();
    descriptor.ordinal = ordinal_it->second;
    descriptor.residue_name = atom.get_res_name();

    int match = -1;
    for (std::size_t i = 0; i < entities_.size(); ++i) {
      if (!titan_segments::entity_matches(entities_[i].definition, descriptor))
        continue;
      if (match >= 0) {
        throw std::runtime_error(
            "Interface entities overlap while assigning atoms.");
      }
      match = static_cast<int>(i);
    }
    if (match >= 0) {
      entities_[static_cast<std::size_t>(match)].atoms.push_back(
          AtomRef{molecule, atom_index});
    }
  }
}

void TitanInterfaceAnalyzer::build_segment_channels() {
  segment_channels_.clear();
  auto add_interface = [&](const std::string &name, int first, int second) {
    if (first < 0 || second < 0)
      return;
    const EntityRuntime &entity_a =
        entities_[static_cast<std::size_t>(first)];
    const EntityRuntime &entity_b =
        entities_[static_cast<std::size_t>(second)];
    std::set<std::string> segments_a;
    std::set<std::string> segments_b;
    for (const EntityResidue &residue : entity_a.residues)
      segments_a.insert(residue.segment);
    for (const EntityResidue &residue : entity_b.residues)
      segments_b.insert(residue.segment);
    for (const std::string &segment_a : segments_a) {
      for (const std::string &segment_b : segments_b) {
        segment_channels_.push_back(
            SegmentChannel{name, entity_a.definition.name, segment_a,
                           entity_b.definition.name, segment_b});
      }
    }
  };

  add_interface("core_a_core_b", core_a_index_, core_b_index_);
  add_interface("core_a_partner", core_a_index_, partner_index_);
  add_interface("core_b_partner", core_b_index_, partner_index_);
}

void TitanInterfaceAnalyzer::populate_runtime_points(
    const Multi_MOL &mol, std::size_t snap_idx,
    const titan_contact_geometry::ContactGeometryCache &geometry,
    std::vector<titan_contact_geometry::ContactPoint> &point_scratch) {
  if (!geometry.build_points(mol, snap_idx, point_scratch))
    throw std::runtime_error(
        "Interface analysis could not build contact points for a frame.");
  for (const auto &point : point_scratch) {
    const ResidueIdentity identity{point.residue.molecule,
                                   point.residue.chainnum,
                                   point.residue.resnr};
    const auto found = residue_locations_.find(identity);
    if (found == residue_locations_.end())
      continue;
    runtime_points_[found->second.entity_index].push_back(
        RuntimePoint{point.x, point.y, point.z,
                     found->second.residue_index});
  }
}

std::vector<std::pair<std::size_t, std::size_t>>
TitanInterfaceAnalyzer::contact_pairs(std::size_t entity_a,
                                      std::size_t entity_b,
                                      const box &frame_box) const {
  const BoxMatrix box_matrix(frame_box);
  const double cutoff2 = cutoff_a_ * cutoff_a_;
  std::set<std::pair<std::size_t, std::size_t>> unique_pairs;
  for (const RuntimePoint &a : runtime_points_[entity_a]) {
    for (const RuntimePoint &b : runtime_points_[entity_b]) {
      double dx = a.x - b.x;
      double dy = a.y - b.y;
      double dz = a.z - b.z;
      box_matrix.apply_mic(dx, dy, dz);
      if (dx * dx + dy * dy + dz * dz <= cutoff2)
        unique_pairs.emplace(a.residue_index, b.residue_index);
    }
  }
  return {unique_pairs.begin(), unique_pairs.end()};
}

void TitanInterfaceAnalyzer::add_segment_counts(
    const std::string &interface_name, std::size_t entity_a,
    std::size_t entity_b,
    const std::vector<std::pair<std::size_t, std::size_t>> &pairs,
    std::vector<std::uint32_t> &counts) const {
  for (const auto &[residue_a, residue_b] : pairs) {
    const std::string &segment_a =
        entities_[entity_a].residues[residue_a].segment;
    const std::string &segment_b =
        entities_[entity_b].residues[residue_b].segment;
    for (std::size_t channel = 0; channel < segment_channels_.size();
         ++channel) {
      const SegmentChannel &candidate = segment_channels_[channel];
      if (candidate.interface_name == interface_name &&
          candidate.entity_a == entities_[entity_a].definition.name &&
          candidate.entity_b == entities_[entity_b].definition.name &&
          candidate.segment_a == segment_a &&
          candidate.segment_b == segment_b) {
        ++counts[channel];
        break;
      }
    }
  }
}

void TitanInterfaceAnalyzer::process_frame(
    const Multi_MOL &mol1, const Multi_MOL *mol2, std::size_t snap_idx,
    std::size_t global_frame_idx, TitanExecutor &executor) {
  if (!enabled_)
    return;
  if (snap_idx >= mol1.m_mol.size())
    return;

  const box *frame_box = nullptr;
  if (BoxMatrix::valid_box(mol1.m_mol[snap_idx].Box))
    frame_box = &mol1.m_mol[snap_idx].Box;
  else if (mol2 != nullptr && snap_idx < mol2->m_mol.size() &&
           BoxMatrix::valid_box(mol2->m_mol[snap_idx].Box))
    frame_box = &mol2->m_mol[snap_idx].Box;
  if (frame_box == nullptr) {
    if (!warned_invalid_box_) {
      TitanUI::print_warning(
          "[Interface] Skipping frames with invalid periodic boxes.");
      warned_invalid_box_ = true;
    }
    return;
  }

  for (auto &points : runtime_points_)
    points.clear();
  if (geometry_mol1_active_)
    populate_runtime_points(mol1, snap_idx, geometry_mol1_,
                            point_scratch_mol1_);
  if (geometry_mol2_active_) {
    if (mol2 == nullptr || snap_idx >= mol2->m_mol.size())
      return;
    populate_runtime_points(*mol2, snap_idx, geometry_mol2_,
                            point_scratch_mol2_);
  }

  const std::size_t core_a = static_cast<std::size_t>(core_a_index_);
  const std::size_t core_b = static_cast<std::size_t>(core_b_index_);
  const auto core_pairs = contact_pairs(core_a, core_b, *frame_box);
  std::vector<std::pair<std::size_t, std::size_t>> core_a_partner_pairs;
  std::vector<std::pair<std::size_t, std::size_t>> core_b_partner_pairs;
  if (partner_index_ >= 0) {
    const std::size_t partner = static_cast<std::size_t>(partner_index_);
    core_a_partner_pairs =
        contact_pairs(core_a, partner, *frame_box);
    core_b_partner_pairs =
        contact_pairs(core_b, partner, *frame_box);
  }

  FrameRecord record;
  record.frame_index = global_frame_idx;
  record.time_ps = mol1.m_mol[snap_idx].time_of_snap;
  record.core_contacts =
      static_cast<std::uint32_t>(core_pairs.size());
  record.core_a_partner_contacts =
      static_cast<std::uint32_t>(core_a_partner_pairs.size());
  record.core_b_partner_contacts =
      static_cast<std::uint32_t>(core_b_partner_pairs.size());
  record.segment_contacts.assign(segment_channels_.size(), 0);
  add_segment_counts("core_a_core_b", core_a, core_b, core_pairs,
                     record.segment_contacts);

  if (partner_index_ < 0) {
    record.state = "two_body";
    record.proximal_entity = "none";
  } else {
    const std::size_t partner = static_cast<std::size_t>(partner_index_);
    add_segment_counts("core_a_partner", core_a, partner,
                       core_a_partner_pairs, record.segment_contacts);
    add_segment_counts("core_b_partner", core_b, partner,
                       core_b_partner_pairs, record.segment_contacts);

    const bool touches_a =
        record.core_a_partner_contacts >=
        static_cast<std::uint32_t>(state_min_contacts_);
    const bool touches_b =
        record.core_b_partner_contacts >=
        static_cast<std::uint32_t>(state_min_contacts_);
    if (touches_a && touches_b)
      record.state = "bridge";
    else if (touches_a)
      record.state = "cap_core_a";
    else if (touches_b)
      record.state = "cap_core_b";
    else
      record.state = "detached";

    const std::string &name_a = entities_[core_a].definition.name;
    const std::string &name_b = entities_[core_b].definition.name;
    if (record.core_a_partner_contacts > record.core_b_partner_contacts)
      record.proximal_entity = name_a;
    else if (record.core_b_partner_contacts > record.core_a_partner_contacts)
      record.proximal_entity = name_b;
    else
      record.proximal_entity =
          record.core_a_partner_contacts == 0 ? "none" : "tie";
  }

  current_records_.push_back(std::move(record));
  ++valid_frame_counter_;
  if (sasa_enabled_ &&
      (valid_frame_counter_ - 1) %
              static_cast<std::size_t>(sasa_stride_) ==
          0) {
    sample_sasa(mol1, mol2, snap_idx, global_frame_idx, executor);
  }
}

void TitanInterfaceAnalyzer::sample_sasa(
    const Multi_MOL &mol1, const Multi_MOL *mol2, std::size_t snap_idx,
    std::size_t global_frame_idx, TitanExecutor &executor) {
  auto atoms_for = [&](int entity_index) {
    std::vector<TitanSASA::Atom> atoms;
    if (entity_index < 0)
      return atoms;
    const EntityRuntime &entity =
        entities_[static_cast<std::size_t>(entity_index)];
    atoms.reserve(entity.atoms.size());
    for (const AtomRef &ref : entity.atoms) {
      const Multi_MOL *mol = ref.molecule == 1 ? &mol1 : mol2;
      if (mol == nullptr || snap_idx >= mol->m_mol.size() ||
          ref.atom_index >= mol->m_mol[snap_idx].size() ||
          !std::isfinite(sasa_radius(*mol, ref.atom_index))) {
        throw std::runtime_error(
            "Interface SASA encountered incomplete coordinates or radii.");
      }
      const MOL &frame = mol->m_mol[snap_idx];
      atoms.push_back(TitanSASA::Atom{
          frame.pos.x[ref.atom_index], frame.pos.y[ref.atom_index],
          frame.pos.z[ref.atom_index], sasa_radius(*mol, ref.atom_index)});
    }
    return atoms;
  };
  auto concatenate = [](const std::vector<TitanSASA::Atom> &first,
                        const std::vector<TitanSASA::Atom> &second) {
    std::vector<TitanSASA::Atom> result;
    result.reserve(first.size() + second.size());
    result.insert(result.end(), first.begin(), first.end());
    result.insert(result.end(), second.begin(), second.end());
    return result;
  };
  auto sasa = [&](const std::vector<TitanSASA::Atom> &atoms) {
    if (atoms.empty())
      return 0.0;
    TitanSASA::Options options;
    options.point_count = sasa_points_;
    return TitanSASA::sum_sasa(
        TitanSASA::compute_per_atom_sasa(atoms, options, {}, &executor));
  };

  const auto atoms_a = atoms_for(core_a_index_);
  const auto atoms_b = atoms_for(core_b_index_);
  const auto atoms_partner = atoms_for(partner_index_);
  const auto atoms_ab = concatenate(atoms_a, atoms_b);
  const auto atoms_complex = concatenate(atoms_ab, atoms_partner);

  SasaRecord record;
  record.frame_index = global_frame_idx;
  record.time_ps = mol1.m_mol[snap_idx].time_of_snap;
  record.sasa_core_a = sasa(atoms_a);
  record.sasa_core_b = sasa(atoms_b);
  record.sasa_partner = sasa(atoms_partner);
  record.sasa_core_pair = sasa(atoms_ab);
  record.sasa_complex =
      partner_index_ >= 0 ? sasa(atoms_complex) : record.sasa_core_pair;
  record.dimer_buried_total =
      record.sasa_core_a + record.sasa_core_b - record.sasa_core_pair;
  record.partner_buried_total =
      partner_index_ >= 0
          ? record.sasa_core_pair + record.sasa_partner -
                record.sasa_complex
          : 0.0;
  current_sasa_records_.push_back(record);
}

TitanInterfaceAnalyzer::MetricSummary TitanInterfaceAnalyzer::summarize_metric(
    const std::vector<double> &values, const UncertaintyArgs &uncertainty,
    unsigned int seed_offset) const {
  MetricSummary summary;
  summary.samples = values.size();
  summary.mean = mean_value(values);
  summary.sem = summary.ci95_low = summary.ci95_high =
      summary.bootstrap_low = summary.bootstrap_high = quiet_nan();
  if (values.empty()) {
    summary.status = "no_samples";
    return summary;
  }
  if (uncertainty.mode == "off") {
    summary.status = "point_estimate_only";
    return summary;
  }

  const auto blocks =
      titan_uncertainty::partition_blocks(values.size(), uncertainty.blocks);
  const std::vector<double> means = block_means(values, blocks);
  summary.blocks = means.size();
  if (means.size() < 2) {
    summary.status = "insufficient_blocks";
    return summary;
  }
  const auto stats = titan_uncertainty::summarize_values(means);
  summary.sem = stats.sem;
  summary.ci95_low = summary.mean - 1.96 * summary.sem;
  summary.ci95_high = summary.mean + 1.96 * summary.sem;
  if (uncertainty.mode == "bootstrap") {
    const auto ci = titan_uncertainty::bootstrap_ci95(
        means, uncertainty.bootstrap_samples,
        uncertainty.seed + seed_offset);
    summary.bootstrap_low = ci.first;
    summary.bootstrap_high = ci.second;
  }
  summary.status = "ok";
  return summary;
}

void TitanInterfaceAnalyzer::export_records(
    const std::string &timeseries_filename,
    const std::string &population_filename,
    const std::string &segment_filename,
    const std::string &segment_timeseries_filename,
    const std::string &metrics_filename, const std::string &sasa_filename,
    const std::vector<FrameRecord> &records,
    const std::vector<SasaRecord> &sasa_records, const arguments &args,
    std::fstream &log) const {
  const auto ensure_parent = [](const std::string &filename) {
    const fs::path parent = fs::path(filename).parent_path();
    if (!parent.empty())
      fs::create_directories(parent);
  };
  ensure_parent(timeseries_filename);

  std::ofstream timeseries(timeseries_filename);
  if (!timeseries.is_open())
    throw std::runtime_error("Could not open " + timeseries_filename);
  timeseries << "# contact_definition="
             << titan_contact::contact_mode_name(contact_mode_)
             << ", cutoff_A=" << cutoff_a_ << "\n";
  timeseries << "# state_threshold_contact_pairs=" << state_min_contacts_
             << "\n";
  timeseries
      << "frame,time_ps,state,proximal_entity,core_a_core_b_contacts,"
         "core_a_partner_contacts,core_b_partner_contacts\n";
  timeseries << std::setprecision(12);
  for (const FrameRecord &record : records) {
    timeseries << record.frame_index << ',' << record.time_ps << ','
               << csv_escape(record.state) << ','
               << csv_escape(record.proximal_entity) << ','
               << record.core_contacts << ','
               << record.core_a_partner_contacts << ','
               << record.core_b_partner_contacts << '\n';
  }

  const std::vector<std::string> states =
      partner_index_ >= 0
          ? std::vector<std::string>{"detached", "cap_core_a", "cap_core_b",
                                     "bridge"}
          : std::vector<std::string>{"two_body"};
  std::ofstream populations(population_filename);
  if (!populations.is_open())
    throw std::runtime_error("Could not open " + population_filename);
  populations
      << "state,n_frames,n_blocks,population,block_sem,ci95_low,ci95_high,"
         "bootstrap_ci95_low,bootstrap_ci95_high,status\n";
  populations << std::setprecision(12);
  for (std::size_t state_index = 0; state_index < states.size();
       ++state_index) {
    std::vector<double> indicator;
    indicator.reserve(records.size());
    for (const FrameRecord &record : records)
      indicator.push_back(record.state == states[state_index] ? 1.0 : 0.0);
    const MetricSummary summary =
        summarize_metric(indicator, args.uncertainty,
                         static_cast<unsigned int>(state_index));
    populations << states[state_index] << ',' << summary.samples << ','
                << summary.blocks << ',' << summary.mean << ',' << summary.sem
                << ',' << summary.ci95_low << ',' << summary.ci95_high << ','
                << summary.bootstrap_low << ',' << summary.bootstrap_high
                << ',' << summary.status << '\n';
  }

  std::ofstream segments(segment_filename);
  if (!segments.is_open())
    throw std::runtime_error("Could not open " + segment_filename);
  segments
      << "interface,entity_a,segment_a,entity_b,segment_b,n_frames,n_blocks,"
         "mean_contact_pairs,contact_probability,block_sem,ci95_low,"
         "ci95_high,bootstrap_ci95_low,bootstrap_ci95_high,status\n";
  segments << std::setprecision(12);
  for (std::size_t channel = 0; channel < segment_channels_.size(); ++channel) {
    std::vector<double> values;
    values.reserve(records.size());
    std::size_t occupied = 0;
    for (const FrameRecord &record : records) {
      const double value =
          channel < record.segment_contacts.size()
              ? static_cast<double>(record.segment_contacts[channel])
              : 0.0;
      values.push_back(value);
      occupied += value > 0.0 ? 1U : 0U;
    }
    const MetricSummary summary = summarize_metric(
        values, args.uncertainty, static_cast<unsigned int>(100 + channel));
    const double probability =
        records.empty()
            ? quiet_nan()
            : static_cast<double>(occupied) /
                  static_cast<double>(records.size());
    const SegmentChannel &definition = segment_channels_[channel];
    segments << definition.interface_name << ','
             << csv_escape(definition.entity_a) << ','
             << csv_escape(definition.segment_a) << ','
             << csv_escape(definition.entity_b) << ','
             << csv_escape(definition.segment_b) << ',' << summary.samples
             << ',' << summary.blocks << ',' << summary.mean << ','
             << probability << ',' << summary.sem << ',' << summary.ci95_low
             << ',' << summary.ci95_high << ',' << summary.bootstrap_low << ','
              << summary.bootstrap_high << ',' << summary.status << '\n';
  }

  if (!segment_timeseries_filename.empty()) {
    ensure_parent(segment_timeseries_filename);
    std::ofstream segment_timeseries(segment_timeseries_filename);
    if (!segment_timeseries.is_open())
      throw std::runtime_error("Could not open " +
                               segment_timeseries_filename);
    segment_timeseries << "# contact_definition="
                       << titan_contact::contact_mode_name(contact_mode_)
                       << ", cutoff_A=" << cutoff_a_ << "\n";
    segment_timeseries
        << "frame,time_ps,interface,entity_a,segment_a,entity_b,segment_b,"
           "contact_pairs,occupied\n";
    segment_timeseries << std::setprecision(12);
    for (const FrameRecord &record : records) {
      for (std::size_t channel = 0; channel < segment_channels_.size();
           ++channel) {
        const SegmentChannel &definition = segment_channels_[channel];
        const std::uint32_t contacts =
            channel < record.segment_contacts.size()
                ? record.segment_contacts[channel]
                : 0U;
        segment_timeseries
            << record.frame_index << ',' << record.time_ps << ','
            << csv_escape(definition.interface_name) << ','
            << csv_escape(definition.entity_a) << ','
            << csv_escape(definition.segment_a) << ','
            << csv_escape(definition.entity_b) << ','
            << csv_escape(definition.segment_b) << ',' << contacts << ','
            << (contacts > 0U ? 1 : 0) << '\n';
      }
    }
  }

  std::ofstream metrics(metrics_filename);
  if (!metrics.is_open())
    throw std::runtime_error("Could not open " + metrics_filename);
  metrics
      << "metric,units,n_samples,n_blocks,mean,block_sem,ci95_low,ci95_high,"
         "bootstrap_ci95_low,bootstrap_ci95_high,status\n";
  metrics << std::setprecision(12);
  auto write_metric = [&](const std::string &name, const std::string &units,
                          const std::vector<double> &values,
                          unsigned int seed_offset) {
    const MetricSummary summary =
        summarize_metric(values, args.uncertainty, seed_offset);
    metrics << name << ',' << units << ',' << summary.samples << ','
            << summary.blocks << ',' << summary.mean << ',' << summary.sem
            << ',' << summary.ci95_low << ',' << summary.ci95_high << ','
            << summary.bootstrap_low << ',' << summary.bootstrap_high << ','
            << summary.status << '\n';
  };

  std::vector<double> core_contacts;
  std::vector<double> contacts_a_partner;
  std::vector<double> contacts_b_partner;
  core_contacts.reserve(records.size());
  contacts_a_partner.reserve(records.size());
  contacts_b_partner.reserve(records.size());
  for (const FrameRecord &record : records) {
    core_contacts.push_back(record.core_contacts);
    contacts_a_partner.push_back(record.core_a_partner_contacts);
    contacts_b_partner.push_back(record.core_b_partner_contacts);
  }
  write_metric("core_a_core_b_contacts", "residue_pairs", core_contacts, 500);
  if (partner_index_ >= 0) {
    write_metric("core_a_partner_contacts", "residue_pairs",
                 contacts_a_partner, 501);
    write_metric("core_b_partner_contacts", "residue_pairs",
                 contacts_b_partner, 502);
  }

  std::vector<double> dimer_buried;
  std::vector<double> partner_buried;
  dimer_buried.reserve(sasa_records.size());
  partner_buried.reserve(sasa_records.size());
  for (const SasaRecord &record : sasa_records) {
    dimer_buried.push_back(record.dimer_buried_total);
    partner_buried.push_back(record.partner_buried_total);
  }
  write_metric("core_pair_buried_surface_total", "A^2", dimer_buried, 510);
  if (partner_index_ >= 0) {
    write_metric("partner_buried_surface_total", "A^2", partner_buried, 511);
  }

  std::ofstream sasa_out(sasa_filename);
  if (!sasa_out.is_open())
    throw std::runtime_error("Could not open " + sasa_filename);
  sasa_out
      << "# Buried surface total = SASA(body1) + SASA(body2) - SASA(complex).\n";
  sasa_out
      << "# Per-side convention is one half of the reported total.\n";
  sasa_out
      << "# Coordinates must contain whole entities; PBCTOPO is recommended.\n";
  sasa_out << "# Radius source priority: GROMACS topology, then TITAN "
              "force-field parameters; nonpositive radii disable SASA.\n";
  sasa_out
      << "frame,time_ps,sasa_core_a_A2,sasa_core_b_A2,sasa_partner_A2,"
         "sasa_core_pair_A2,sasa_complex_A2,core_pair_buried_total_A2,"
         "core_pair_buried_per_side_A2,partner_buried_total_A2,"
         "partner_buried_per_side_A2\n";
  sasa_out << std::setprecision(12);
  for (const SasaRecord &record : sasa_records) {
    sasa_out << record.frame_index << ',' << record.time_ps << ','
             << record.sasa_core_a << ',' << record.sasa_core_b << ','
             << record.sasa_partner << ',' << record.sasa_core_pair << ','
             << record.sasa_complex << ',' << record.dimer_buried_total << ','
             << 0.5 * record.dimer_buried_total << ','
             << record.partner_buried_total << ','
             << 0.5 * record.partner_buried_total << '\n';
  }

  if (log.is_open()) {
    TitanUI::log_kv(log, "Interface states", population_filename);
    TitanUI::log_kv(log, "Interface segment contacts", segment_filename);
    if (!segment_timeseries_filename.empty())
      TitanUI::log_kv(log, "Interface segment time series",
                      segment_timeseries_filename);
    TitanUI::log_kv(log, "Interface metrics", metrics_filename);
    TitanUI::log_kv(log, "Interface SASA", sasa_filename);
  }
}

void TitanInterfaceAnalyzer::finalize_trajectory(
    const std::string &base_dir, const info &info_output,
    const arguments &args, const std::vector<std::string> &input_line,
    int input_name, std::fstream &log) {
  if (!enabled_ || current_records_.empty())
    return;
  const auto path = [&](const std::string &filename) {
    return (fs::path(base_dir) / filename).string();
  };
  export_records(
      path(info_output.get_interface_timeseries_filename(args, input_line,
                                                         input_name)),
      path(info_output.get_interface_populations_filename(args, input_line,
                                                          input_name)),
      path(info_output.get_interface_segments_filename(args, input_line,
                                                       input_name)),
      path(info_output.get_interface_segment_timeseries_filename(
          args, input_line, input_name)),
      path(info_output.get_interface_metrics_filename(args, input_line,
                                                      input_name)),
      path(info_output.get_interface_sasa_filename(args, input_line,
                                                   input_name)),
      current_records_, current_sasa_records_, args, log);

  pooled_records_.insert(pooled_records_.end(), current_records_.begin(),
                         current_records_.end());
  pooled_sasa_records_.insert(pooled_sasa_records_.end(),
                              current_sasa_records_.begin(),
                              current_sasa_records_.end());
  ++trajectory_count_;
  reset_trajectory();
}

void TitanInterfaceAnalyzer::finalize_all(const std::string &base_dir,
                                          const info &info_output,
                                          const arguments &args,
                                          std::fstream &log) {
  if (!enabled_ || trajectory_count_ < 2 || pooled_records_.empty())
    return;
  const auto path = [&](const std::string &filename) {
    return (fs::path(base_dir) / filename).string();
  };
  export_records(path(info_output.get_interface_timeseries_average_filename()),
                 path(info_output.get_interface_populations_average_filename()),
                 path(info_output.get_interface_segments_average_filename()),
                 std::string(),
                 path(info_output.get_interface_metrics_average_filename()),
                 path(info_output.get_interface_sasa_average_filename()),
                 pooled_records_, pooled_sasa_records_, args, log);
}

void TitanInterfaceAnalyzer::reset_trajectory() {
  current_records_.clear();
  current_sasa_records_.clear();
  valid_frame_counter_ = 0;
}

#include "engine/engine_common.h"

namespace engine_detail {

bool edecomp_mm_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp &&
         (Arguments.edecomp.source == "mm" ||
          Arguments.edecomp.source == "all");
}

bool edecomp_gbsa_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp && Arguments.flag_gbsa &&
         (Arguments.edecomp.source == "gbsa" ||
          Arguments.edecomp.source == "all");
}

bool edecomp_pbsa_requested(const arguments &Arguments) {
  return Arguments.edecomp.flag_edecomp && Arguments.pbe.pbsa &&
         (Arguments.edecomp.source == "pbsa" ||
          Arguments.edecomp.source == "all");
}

titan_energy_decomp::RankingMetric
edecomp_rank_metric(const std::string &rank_by) {
  if (rank_by == "mean-abs")
    return titan_energy_decomp::RankingMetric::MeanAbs;
  return titan_energy_decomp::RankingMetric::AbsMean;
}

titan_energy_decomp::EngineConfig make_edecomp_config(
    const arguments &Arguments) {
  titan_energy_decomp::EngineConfig config;
  if (Arguments.edecomp.edge_top == "all") {
    config.top_edge_count = std::numeric_limits<std::size_t>::max();
  } else {
    config.top_edge_count =
        static_cast<std::size_t>(std::stoull(Arguments.edecomp.edge_top));
  }
  config.edge_metric = edecomp_rank_metric(Arguments.edecomp.rank_by);
  config.ranking_metric = edecomp_rank_metric(Arguments.edecomp.rank_by);
  config.sort_direction = titan_energy_decomp::SortDirection::Descending;
  return config;
}

std::string edecomp_output_suffix(const arguments &Arguments,
                                  const std::vector<std::string> &input_line,
                                  int input_name) {
  if (Arguments.flag_separation_results)
    return {};
  const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                           Arguments.flag_pattern_input || Arguments.flag_xdr;
  if (!multi_input)
    return {};
  const int index =
      input_name >= Arguments.b_input ? input_name - Arguments.b_input : 0;
  return "_" + std::to_string(index);
}

// EDecomp files follow -eu like every other energy output.
titan_energy_decomp::OutputUnit edecomp_output_unit(const arguments &Arguments) {
  return {Arguments.eu, Arguments.eu_par};
}

std::optional<std::size_t>
find_edecomp_entity_index(const titan_energy_decomp::EntityAxis &axis,
                          const std::string &chain, int resnr) {
  for (std::size_t i = 0; i < axis.entities.size(); ++i) {
    const auto &entity = axis.entities[i];
    if (entity.chain == chain &&
        static_cast<int>(entity.residue_number) == resnr) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t>
find_edecomp_atom_entity_index(const titan_energy_decomp::EntityAxis &axis,
                               std::size_t one_based_atom_index,
                               const std::string &chain, int resnr,
                               const std::string &atom_name) {
  if (one_based_atom_index > 0 &&
      one_based_atom_index <= axis.entities.size()) {
    return one_based_atom_index - 1;
  }
  for (std::size_t i = 0; i < axis.entities.size(); ++i) {
    const auto &entity = axis.entities[i];
    if (entity.chain == chain &&
        static_cast<int>(entity.residue_number) == resnr &&
        (atom_name.empty() || entity.atom_name == atom_name)) {
      return i;
    }
  }
  return std::nullopt;
}

std::vector<titan_energy_decomp::EndpointEntityRow>
make_gbsa_endpoint_rows(const GBSA_Aggregator &frame_decomp,
                        const titan_energy_decomp::EndpointEntityDecompositionEngine
                            &engine) {
  std::vector<titan_energy_decomp::EndpointEntityRow> rows;
  rows.reserve(frame_decomp.get_decomp().size());
  const auto &mol1_residues = engine.mol1_axes().residues;
  const auto &mol2_residues = engine.mol2_axes().residues;
  for (const auto &entry : frame_decomp.get_decomp()) {
    const std::string role = entry.is_ligand ? "mol2" : "mol1";
    const auto &axis = entry.is_ligand ? mol2_residues : mol1_residues;
    const auto entity_index =
        find_edecomp_entity_index(axis, entry.chain_str, entry.resnr);
    if (!entity_index)
      continue;
    rows.push_back({titan_energy_decomp::EntityKind::Residue, role,
                    *entity_index, 0.5 * entry.E_elec, 0.5 * entry.E_vdw,
                    entry.G_polar, entry.G_nonpolar});
  }
  return rows;
}

std::vector<titan_energy_decomp::EndpointEntityRow> make_pbsa_endpoint_rows(
    const std::vector<TitanPBE::EDecompRow> &frame_rows,
    const titan_energy_decomp::EndpointEntityDecompositionEngine &engine) {
  std::vector<titan_energy_decomp::EndpointEntityRow> rows;
  rows.reserve(frame_rows.size());
  for (const TitanPBE::EDecompRow &entry : frame_rows) {
    const std::string role =
        (entry.molecule == "ligand" || entry.molecule == "mol2") ? "mol2"
                                                                  : "mol1";
    const bool atom_level = entry.unit_type == "atom";
    const auto &axes = role == "mol2" ? engine.mol2_axes()
                                      : engine.mol1_axes();
    const auto entity_index =
        atom_level ? find_edecomp_atom_entity_index(
                         axes.atoms, entry.atom_index, entry.chain,
                         static_cast<int>(entry.resnr), entry.atom_name)
                   : find_edecomp_entity_index(
                         axes.residues, entry.chain,
                         static_cast<int>(entry.resnr));
    if (!entity_index)
      continue;
    rows.push_back({atom_level ? titan_energy_decomp::EntityKind::Atom
                               : titan_energy_decomp::EntityKind::Residue,
                    role, *entity_index, entry.elec_kj_mol, entry.vdw_kj_mol,
                    entry.polar_kj_mol, entry.nonpolar_kj_mol});
  }
  return rows;
}

void auto_adjust_polymer_type(arguments &Arguments, Multi_MOL &mol,
                              std::fstream &log, bool is_mol1) {
  const std::string detected_type =
      PolymerClassification::detect_nucleic_polymer_type(mol.info);
  if (detected_type.empty())
    return;

  std::string &mol_type =
      is_mol1 ? Arguments.mol_1_type : Arguments.mol_2_type;
  const bool explicit_type =
      is_mol1 ? Arguments.mol_1_type_explicit : Arguments.mol_2_type_explicit;
  const std::string label = is_mol1 ? "Molecule 1 Type" : "Molecule 2 Type";
  const std::string molecule_name = is_mol1 ? "molecule 1" : "molecule 2";

  if (!explicit_type && is_generic_autotype_candidate(mol_type)) {
    mol_type = detected_type;
    mol.set_mol_type(detected_type);
    for (auto &frame : mol.m_mol)
      frame.mol_type = detected_type;

    const std::string message =
        "Auto-detected " + molecule_name + " as " + detected_type +
        " from topology.";
    TitanUI::print_warning(message);
    TitanUI::log_kv(log, label, mol_type);
    TitanUI::log_message(log, "[Auto-Type] " + message);
    return;
  }

  const std::string resolved_upper =
      PolymerClassification::uppercase_copy(mol_type);
  const std::string detected_upper =
      PolymerClassification::uppercase_copy(detected_type);
  const bool resolved_matches_detected =
      resolved_upper.find(detected_upper) != std::string::npos;
  if (!resolved_matches_detected) {
    const std::string message =
        PolymerClassification::uppercase_copy(molecule_name) +
        " topology looks like " + detected_type +
        " but resolved type is " + mol_type +
        ". RNA/DNA-aware modules will fall back to residue/atom detection.";
    TitanUI::print_warning(message);
    TitanUI::log_message(log, "[Auto-Type] " + message);
  }
}

} // namespace engine_detail

#include "titan_output_manifest.h"
#include "titan_ui.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace titan_output_manifest {
namespace {

void csv_string(std::ostream &out, const std::string &value) {
  const bool quote = value.find_first_of(",\"\n\r") != std::string::npos;
  if (!quote) {
    out << value;
    return;
  }
  out << '"';
  for (const char c : value) {
    if (c == '"')
      out << "\"\"";
    else
      out << c;
  }
  out << '"';
}

std::string bool_text(bool value) { return value ? "true" : "false"; }

std::string indexed(const std::string &single_name) {
  const auto dot = single_name.find_last_of('.');
  if (dot == std::string::npos)
    return single_name + "[_N]";
  return single_name.substr(0, dot) + "[_N]" + single_name.substr(dot);
}

bool likely_multi_input(const arguments &args) {
  return args.flag_listfile || args.flag_pattern_input || args.flag_xdr;
}

void add(std::vector<OutputManifestRow> &rows, std::string module,
         std::string filename, std::string units, std::string averaging,
         std::string note, std::string trajectory = "all") {
  rows.push_back(OutputManifestRow{std::move(module), std::move(filename),
                                   std::move(trajectory), std::move(units),
                                   std::move(averaging), std::move(note)});
}

void add_indexed(std::vector<OutputManifestRow> &rows, const arguments &args,
                 std::string module, std::string filename, std::string units,
                 std::string averaging, std::string note) {
  if (likely_multi_input(args))
    filename = indexed(filename);
  add(rows, std::move(module), std::move(filename), std::move(units),
      std::move(averaging), std::move(note),
      likely_multi_input(args) ? "per_trajectory" : "all");
}

std::string active_modules_text(const arguments &args) {
  std::vector<std::string> modules;
  if (!args.flag_nocal)
    modules.push_back("MM");
  if (args.flag_gbsa)
    modules.push_back("GBSA");
  if (args.pbe.pbsa)
    modules.push_back("MM/PBSA");
  else if (args.pbe.flag_pbe)
    modules.push_back("PBE");
  if (args.edecomp.flag_edecomp)
    modules.push_back("EDecomp");
  if (args.flag_cas)
    modules.push_back("CAS");
  if (args.cmap.flag_cmap)
    modules.push_back("CMap");
  if (args.interface_analysis.flag_interface_analysis)
    modules.push_back("Interface");
  if (args.dssp.flag_dssp)
    modules.push_back("DSSP");
  if (args.flag_network)
    modules.push_back("DCCM");
  if (args.qnative.flag_qnative)
    modules.push_back("QNative");
  if (args.dihedral.flag_dihedral)
    modules.push_back("Dihedral");
  if (args.msd.flag_msd)
    modules.push_back("MSD");
  if (args.membrane.flag_membrane)
    modules.push_back("Membrane");
  if (args.s2.flag_s2)
    modules.push_back("S2");
  if (args.feature_export.export_features)
    modules.push_back("FeatureExport");
  if (args.perf_report.mode != "off")
    modules.push_back("Performance");
  if (args.uncertainty.mode != "off")
    modules.push_back("Uncertainty");
  if (modules.empty())
    return "none";
  std::ostringstream out;
  for (std::size_t i = 0; i < modules.size(); ++i) {
    if (i)
      out << ',';
    out << modules[i];
  }
  return out.str();
}

} // namespace

std::vector<OutputManifestRow> build_output_manifest(const arguments &args,
                                                     const info &out_info) {
  std::vector<OutputManifestRow> rows;
  // Energy files are written in the -eu unit (normalised to "kJ/mol" or
  // "kcal/mol" by post_process); the MM/PBSA energy CSV instead carries both
  // units side by side.
  const std::string &energy_unit = args.eu;
  add(rows, "Core", "log.txt", "text", "single run",
      "Main runtime log and warnings.");
  add(rows, "Core", out_info.get_titan_run_config_summary_filename(), "text",
      "single run", "Resolved run configuration and active module summary.");
  add(rows, "Core", out_info.get_titan_outputs_manifest_filename(), "table",
      "single run", "Expected output manifest for this resolved configuration.");
  if (!args.flag_nocal) {
    add(rows, "MM", "cal.txt", energy_unit, "all analyzed frames",
        "Electrostatic plus van der Waals energy summary.");
    add(rows, "MM", "cal.xvg", energy_unit, "time series",
        "Electrostatic plus van der Waals energy by frame.");
  }
  if (!args.out_pdb_name.empty())
    add(rows, "PDB", out_info.get_pdboutname(args), "coordinates",
        "trajectory", "Optional coordinate output from -opdb/-opdbsep.");

  if (args.cmap.flag_cmap) {
    add_indexed(rows, args, "CMap", "contact_probability_matrix.csv",
                "probability", "per trajectory",
                "Residue contact probability matrix.");
    add_indexed(rows, args, "CMap", "contact_probability_edges.csv",
                "probability", "per trajectory",
                "Sparse contact edges with probabilities.");
    add_indexed(rows, args, "CMap", "contact_count_timeseries.csv", "count",
                "time series", "Per-frame contact count.");
    add_indexed(rows, args, "CMap", "contact_residue_occupancy.csv",
                "probability", "per trajectory",
                "Residue occupancy in observed contacts.");
    add_indexed(rows, args, "CMap", "contact_probability_summary.dat", "text",
                "per trajectory", "Contact-map settings and summary.");
  }
  if (args.interface_analysis.flag_interface_analysis) {
    add_indexed(rows, args, "Interface", "interface_state_timeseries.csv",
                "contact pairs", "time series",
                "Three-entity contact state and pair counts by frame.");
    add_indexed(rows, args, "Interface", "interface_state_populations.csv",
                "probability", "block estimate",
                "State populations with block uncertainty.");
    add_indexed(rows, args, "Interface", "interface_segment_contacts.csv",
                "contact pairs", "block estimate",
                "Segment-pair contact means and occupancies.");
    add_indexed(rows, args, "Interface", "interface_segment_timeseries.csv",
                "contact pairs", "time series",
                "Per-frame segment-pair contact counts and occupancy.");
    add_indexed(rows, args, "Interface", "interface_metrics.csv", "mixed",
                "block estimate",
                "Contact and buried-surface observables.");
    add_indexed(rows, args, "Interface", "interface_sasa_timeseries.csv",
                "A^2", "strided time series",
                "SASA and total/per-side buried surface.");
  }
  if (args.dssp.flag_dssp) {
    add(rows, "DSSP", out_info.get_dssp_mol1_filename(), "occupancy",
        "all trajectories", "Per-residue secondary-state occupancy.");
    add(rows, "DSSP", out_info.get_dssp_mol1_content_filename(), "occupancy",
        "all trajectories", "Whole-object secondary-state content.");
    if (!args.segments.config_file.empty()) {
      add(rows, "DSSP", out_info.get_dssp_mol1_segments_filename(),
          "occupancy", "all trajectories",
          "Entity and segment secondary-state content.");
    }
    add(rows, "DSSP", out_info.get_dssp_mol1_bridges_filename(),
        "occupancy", "all frames",
        "Bridge orientation and registry distribution.");
    add(rows, "DSSP", out_info.get_dssp_mol1_ladders_filename(), "count",
        "all frames", "Ladder orientation and length distribution.");
  }
  if (args.flag_network) {
    add(rows, "DCCM", "dccm_matrix.csv", "correlation", "all frames",
        "Complete signed displacement-correlation matrix.");
    if (args.uncertainty.mode != "off") {
      add(rows, "DCCM", out_info.get_dccm_block_uncertainty_filename(),
          "correlation", "block estimate",
          "Pairwise DCCM uncertainty and sign stability.");
      add(rows, "DCCM", out_info.get_dccm_block_values_filename(),
          "correlation", "per block",
          "Per-block pair correlations for between-ensemble contrasts.");
    }
  }
  if (args.qnative.flag_qnative) {
    add_indexed(rows, args, "QNative", "qnative_timeseries.csv", "fraction",
                "time series", "Fraction of native contacts by frame.");
    add_indexed(rows, args, "QNative", "qnative_native_contacts.csv", "list",
                "reference", "Native contact list from the reference.");
    add_indexed(rows, args, "QNative", "qnative_contact_occupancy.csv",
                "probability", "per trajectory",
                "Native-contact occupancy over trajectory.");
    add_indexed(rows, args, "QNative", "qnative_summary.dat", "text",
                "per trajectory", "Q-native settings and summary.");
  }
  if (args.dihedral.flag_dihedral) {
    const bool backbone = args.dihedral.kind == "backbone" ||
                          args.dihedral.kind == "both";
    const bool sidechain = args.dihedral.kind == "sidechain" ||
                           args.dihedral.kind == "both";
    if (backbone) {
      add_indexed(rows, args, "Dihedral", "dihedral_backbone_timeseries.csv",
                  "degrees", "time series", "Phi/psi/omega by residue.");
      add_indexed(rows, args, "Dihedral", "dihedral_ramachandran.csv",
                  "count", "histogram", "Occupied phi/psi bins.");
    }
    if (sidechain) {
      add_indexed(rows, args, "Dihedral", "dihedral_sidechain_timeseries.csv",
                  "degrees", "time series", "Chi1/chi2 by residue.");
      add_indexed(rows, args, "Dihedral", "dihedral_rotamer_populations.csv",
                  "occupancy", "per trajectory", "Rotamer populations.");
      add_indexed(rows, args, "Dihedral", "dihedral_rotamer_transitions.csv",
                  "count", "per trajectory", "Rotamer transition counts.");
    }
    if (args.dihedral.export_tica)
      add_indexed(rows, args, "Dihedral", "dihedral_feature_matrix.csv",
                  "sin/cos", "time series", "TICA-ready torsion features.");
    add_indexed(rows, args, "Dihedral", "dihedral_summary.dat", "text",
                "per trajectory", "Dihedral settings and summary.");
  }
  if (args.msd.flag_msd) {
    add_indexed(rows, args, "MSD", "msd_timeseries.dat", "A^2",
                "per trajectory", "Translational MSD by lag.");
    add_indexed(rows, args, "MSD", "msd_diffusion_summary.dat", "text",
                "per trajectory", "Diffusion fit summary.");
    add_indexed(rows, args, "MSD", "msd_alpha_fit.dat", "text",
                "per trajectory", "Anomalous exponent fit.");
    if (args.msd.rot)
      add_indexed(rows, args, "MSD", "rotational_diffusion_summary.dat",
                  "1/ps", "per trajectory",
                  "Rigid-body rotational diffusion tensor summary.");
  }
  if (args.membrane.flag_membrane) {
    add_indexed(rows, args, "Membrane", "membrane_leaflets.csv", "label",
                "time series", "Leaflet assignment by lipid.");
    add_indexed(rows, args, "Membrane", "membrane_apl_timeseries.csv", "A^2",
                "time series", "Area per lipid by frame.");
    add_indexed(rows, args, "Membrane", "membrane_thickness_map.csv", "A",
                "grid average", "Local membrane thickness map.");
    add_indexed(rows, args, "Membrane", "membrane_summary.dat", "text",
                "per trajectory", "Membrane analysis summary.");
    if (args.membrane.protein_contacts || args.membrane.enrichment)
      add_indexed(rows, args, "Membrane",
                  "membrane_protein_lipid_contacts.csv", "A",
                  "per trajectory",
                  "Mol1 residue side-chain COM to lipid bead contacts.");
    if (args.membrane.enrichment)
      add_indexed(rows, args, "Membrane", "membrane_lipid_enrichment.csv",
                  "ratio", "per trajectory",
                  "Lipid resname enrichment near protein versus bulk leaflet.");
    if (args.membrane.curvature == "grid")
      add_indexed(rows, args, "Membrane", "membrane_curvature_map.csv",
                  "1/A", "grid average",
                  "Headgroup-grid curvature proxy.");
    if (args.membrane.protein_contacts || args.membrane.enrichment)
      add_indexed(rows, args, "Membrane",
                  "membrane_proximity_coupling.csv", "mixed",
                  "per trajectory",
                  "Near-vs-far protein-proximity deltas for APL, thickness, "
                  "order, curvature, contact probability, and composition.");
    if (args.membrane.protein_contacts || args.membrane.enrichment ||
        args.membrane.curvature == "grid")
      add_indexed(rows, args, "Membrane", "membrane_coupling_summary.dat",
                  "text", "per trajectory",
                  "Protein-lipid coupling settings and top contacts.");
  }
  if (args.s2.flag_s2) {
    add_indexed(rows, args, "S2", "s2_order_parameters.csv", "unitless",
                "per trajectory", "Lipari-Szabo order parameters.");
    add_indexed(rows, args, "S2", "s2_correlation_functions.dat", "unitless",
                "correlation", "Per-vector correlation functions.");
    add_indexed(rows, args, "S2", "s2_summary.dat", "text",
                "per trajectory", "S2 fit summary.");
    if (args.s2.bfactor_pdb)
      add_indexed(rows, args, "S2", "s2_bfactor.pdb", "PDB B-factor",
                  "visualization", "Clipped S2 mapped to B-factor.");
  }
  if (args.feature_export.export_features) {
    add_indexed(rows, args, "FeatureExport", "features_by_frame.csv",
                "mixed", "time series", "Observable feature matrix.");
    add_indexed(rows, args, "FeatureExport", "features_metadata.csv",
                "metadata", "single table", "Feature column metadata.");
    add_indexed(rows, args, "FeatureExport", "feature_export_summary.dat",
                "text", "per trajectory", "Feature source resolution.");
  }
  if (args.flag_gbsa) {
    add(rows, "GBSA", out_info.get_gbsa_snapshots_filename(), energy_unit,
        "snapshots", "GBSA per-snapshot components.");
    add(rows, "GBSA", out_info.get_gbsa_summary_filename(), "text",
        "all analyzed frames", "GBSA summary.");
  }
  if (args.pbe.pbsa) {
    add(rows, "MM/PBSA", "mm_pbsa_summary.dat", "text",
        "all analyzed frames", "MM/PBSA solver and component summary.");
    add(rows, "MM/PBSA", "mm_pbsa_energy.csv", "kJ/mol and kcal/mol",
        "time series", "MM/PBSA binding components by frame.");
  }
  if (args.edecomp.flag_edecomp) {
    const auto add_edecomp_mm = [&]() {
      add(rows, "EDecomp-MM", out_info.get_edecomp_mm_residue_matrix_total_filename(),
          energy_unit, "mean over frames",
          "Direct pairwise MM elec+vdW residue matrix.");
      add(rows, "EDecomp-MM", out_info.get_edecomp_mm_residue_edges_filename(),
          energy_unit, "mean over frames", "Pairwise MM residue edges.");
      add(rows, "EDecomp-MM", out_info.get_edecomp_mm_residue_ranking_filename(),
          energy_unit, "mean over frames", "Residue interaction ranking.");
    };
    if (args.edecomp.source == "mm" || args.edecomp.source == "all")
      add_edecomp_mm();
    if (args.edecomp.source == "gbsa" || args.edecomp.source == "all") {
      add(rows, "EDecomp-MMGBSA",
          out_info.get_edecomp_mmgbsa_residue_map_filename(), energy_unit,
          "mean over frames",
          "MM/GBSA endpoint entity decomposition, not a pair matrix.");
      add(rows, "EDecomp-MMGBSA",
          out_info.get_edecomp_mmgbsa_residue_ranking_filename(), energy_unit,
          "mean over frames", "MM/GBSA entity ranking.");
    }
    if (args.edecomp.source == "pbsa" || args.edecomp.source == "all") {
      add(rows, "EDecomp-MMPBSA",
          out_info.get_edecomp_mmpbsa_residue_map_filename(), energy_unit,
          "mean over frames",
          "MM/PBSA endpoint entity decomposition, not a pair matrix.");
      add(rows, "EDecomp-MMPBSA",
          out_info.get_edecomp_mmpbsa_residue_ranking_filename(), energy_unit,
          "mean over frames", "MM/PBSA entity ranking.");
    }
    add(rows, "EDecomp", out_info.get_edecomp_summary_filename(), "text",
        "all analyzed frames", "Unified energy decomposition summary.");
  }
  if (args.flag_cas) {
    add(rows, "CAS", out_info.get_cas_receptor_filename(), energy_unit,
        "mean over frames", "GBSA alanine-like receptor scan.");
    add(rows, "CAS", out_info.get_cas_summary_filename(), "text",
        "all analyzed frames", "CAS method and limitation summary.");
    add(rows, "CAS", out_info.get_cas_performance_filename(), "timing",
        "all analyzed frames", "CAS performance counters.");
  }
  if (args.hotspot.flag_hotspot_report) {
    add(rows, "Hotspot", out_info.get_hotspot_residue_ranking_filename(),
        "dimensionless", "single run",
        "CAS/EDecomp residue hotspot consensus ranking.");
    add(rows, "Hotspot", out_info.get_hotspot_summary_filename(), "text",
        "single run", "Hotspot source coverage and method notes.");
    add(rows, "Hotspot", out_info.get_hotspot_bfactor_pdb_filename(),
        "PDB B-factor", "visualization",
        "Dimensionless hotspot score mapped to B-factor when a template is available.");
  }
  if (args.perf_report.mode != "off") {
    add(rows, "Performance", out_info.get_titan_performance_summary_filename(),
        "timing", "all analyzed frames", "Shared performance summary.");
    add(rows, "Performance", out_info.get_titan_performance_modules_filename(),
        "timing", "module aggregate", "Shared module timing table.");
    if (args.perf_report.mode == "detail")
      add(rows, "Performance", out_info.get_titan_performance_frames_filename(),
          "timing", "frame/module detail", "Shared frame timing table.");
  }
  if (args.uncertainty.mode != "off") {
    add(rows, "Uncertainty", out_info.get_uncertainty_summary_filename(),
        "diagnostics", "single run",
        "Cross-module uncertainty summary; point estimates unchanged.");
    if (args.cmap.flag_cmap)
      add(rows, "Uncertainty", out_info.get_cmap_uncertainty_filename(),
          "diagnostics", "per trajectory", "CMap uncertainty sidecar.");
    if (args.qnative.flag_qnative)
      add(rows, "Uncertainty", out_info.get_qnative_uncertainty_filename(),
          "diagnostics", "per trajectory", "QNative uncertainty sidecar.");
    if (args.dihedral.flag_dihedral)
      add(rows, "Uncertainty", out_info.get_dihedral_uncertainty_filename(),
          "diagnostics", "per trajectory", "Dihedral uncertainty sidecar.");
    if (args.msd.flag_msd)
      add(rows, "Uncertainty", out_info.get_msd_uncertainty_filename(),
          "diagnostics", "per trajectory", "MSD uncertainty sidecar.");
    if (args.s2.flag_s2)
      add(rows, "Uncertainty", out_info.get_s2_uncertainty_filename(),
          "diagnostics", "per trajectory", "S2 uncertainty sidecar.");
    if (args.membrane.flag_membrane)
      add(rows, "Uncertainty", out_info.get_membrane_uncertainty_filename(),
          "diagnostics", "per trajectory", "Membrane uncertainty sidecar.");
    if (args.edecomp.flag_edecomp)
      add(rows, "Uncertainty", out_info.get_edecomp_uncertainty_filename(),
          "diagnostics", "per trajectory", "EDecomp uncertainty sidecar.");
  }
  if (args.output_manifest.schema_export == "json")
    add(rows, "Schema", out_info.get_titan_schema_filename(), "json",
        "single run", "Resolved option schema export.");

  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    if (a.module != b.module)
      return a.module < b.module;
    return a.filename < b.filename;
  });
  return rows;
}

void export_output_manifest(const std::vector<OutputManifestRow> &rows,
                            const std::string &filename, std::fstream *log) {
  std::ofstream out(filename);
  out << "module,filename,trajectory,units,averaging_policy,semantic_note\n";
  for (const auto &row : rows) {
    csv_string(out, row.module);
    out << ',';
    csv_string(out, row.filename);
    out << ',';
    csv_string(out, row.trajectory);
    out << ',';
    csv_string(out, row.units);
    out << ',';
    csv_string(out, row.averaging_policy);
    out << ',';
    csv_string(out, row.semantic_note);
    out << '\n';
  }
  if (log)
    TitanUI::log_kv(*log, "Output Manifest", filename);
}

void export_run_config_summary(const arguments &args,
                               const std::vector<OutputManifestRow> &rows,
                               const std::string &filename,
                               std::fstream *log) {
  std::ofstream out(filename);
  out << "module: titan_run_config\n";
  out << "dry_run: " << bool_text(args.output_manifest.dry_run) << '\n';
  out << "output_manifest: "
      << bool_text(args.output_manifest.output_manifest) << '\n';
  out << "schema_export: " << args.output_manifest.schema_export << '\n';
  out << "input: " << args.pdb_name << '\n';
  out << "trajectory: " << args.traj_file << '\n';
  out << "nocal: " << bool_text(args.flag_nocal) << '\n';
  out << "force_field: " << args.force_f << '\n';
  out << "active_modules: " << active_modules_text(args) << '\n';
  out << "expected_output_rows: " << rows.size() << '\n';
  out << "dry_run_note: dry-run exits before trajectory processing and writes "
         "only manifest/schema/config files.\n";
  out << "manifest_note: manifest rows describe expected primary outputs; "
         "some debug or legacy compatibility sidecars may be omitted.\n";
  if (log)
    TitanUI::log_kv(*log, "Run Config Summary", filename);
}

} // namespace titan_output_manifest

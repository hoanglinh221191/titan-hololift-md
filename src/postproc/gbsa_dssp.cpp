#include "postproc/postproc_common.h"

using namespace postproc_detail;

// ============================================================================
// 1. GBSA + ML + CAS + Nucleic Acid post-processing
// ============================================================================
void run_gbsa_postproc(const arguments &args, size_t tot_snap_allfiles,
                       MLFeatureExtractor &ml_extractor,
                       GBSA_Aggregator &global_gbsa_agg,
                       CASAnalyzer &cas_analyzer, NucleicProfiler &na_profiler,
                       bool global_has_nucleic_acid, info &info_output,
                       std::fstream &log) {
  if (args.flag_gbsa) {
    if (args.flag_ml) {
      if (tot_snap_allfiles > 0) {
        TitanUI::print_module_header("Machine Learning Feature Extraction");
        std::string ml_out_file = info_output.get_ml_features_filename();
        ml_extractor.export_features(ml_out_file);
        log << "Exported ML features to " << ml_out_file << "\n";
        TitanUI::print_info("Successfully exported 5 ML features to " +
                            ml_out_file);

        // KIN-10: the bundled random forest is closed. Its training set,
        // feature units, cross-validation score and applicability domain are
        // unknown, and feature 0 is an absolute interaction energy, so the
        // prediction is not transferable between systems, force fields or
        // cutoff conventions. Feature export stays available because the five
        // descriptors are reproducible on their own; only the pKd number is
        // withheld. Re-open it by restoring this branch once a model card
        // exists (see KIN-10 in docs/TITAN_CODE_REVIEW_2026-09-08.md).
        const std::string ml_closed_notice =
            "ML features exported. Affinity prediction is closed: the bundled "
            "model has no training/calibration provenance and its leading "
            "feature is an absolute energy, so the pKd is not transferable. No "
            "prediction file is written.";
        TitanUI::print_info(ml_closed_notice);
        log << "[Titan ML] " << ml_closed_notice << "\n";
        if (args.ml_predict_experimental) {
          const std::string ml_optin_notice =
              "-ml-predict-experimental no longer enables the bundled model; "
              "the option is retained only to report that it has no effect.";
          TitanUI::print_info(ml_optin_notice);
          log << "[Titan ML] " << ml_optin_notice << "\n";
        }
      }
    }
    try {
      if (tot_snap_allfiles > 0) {
        log << "Printing GBSA statistics & decompositions...\n";
        TitanUI::print_module_header("GBSA Statistics & Decompositions");
        std::string out_dir = ".";
        global_gbsa_agg.print_all_stats(out_dir, &log);
        TitanUI::print_info(
            "GBSA Decomposition files successfully written to " + out_dir);
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception in GBSA: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception in GBSA!\n";
    }

    if (args.flag_cas) {
      try {
        if (tot_snap_allfiles > 0) {
          cas_analyzer.export_results(info_output.get_cas_receptor_filename(),
                                      info_output.get_cas_ligand_filename(),
                                      args.eu_par);
          cas_analyzer.export_hotspot_pdb(
              info_output.get_cas_hotspot_pdb_filename(), args.eu_par);
          cas_analyzer.export_summary(info_output.get_cas_summary_filename(),
                                      args);
          cas_analyzer.export_performance(
              info_output.get_cas_performance_filename());
        }
      } catch (const std::exception &e) {
        std::cerr << "[ERROR] Exception in CAS: " << e.what() << "\n";
      } catch (...) {
        std::cerr << "[ERROR] Unknown exception in CAS!\n";
      }
    }
  }

  if (global_has_nucleic_acid) {
    try {
      if (tot_snap_allfiles > 0) {
        na_profiler.export_results(
            info_output.get_nucleic_acid_interactions_filename());
      }
    } catch (const std::exception &e) {
      std::cerr << "[ERROR] Exception in NA Profiler: " << e.what() << "\n";
    } catch (...) {
      std::cerr << "[ERROR] Unknown exception in NA Profiler!\n";
    }
  }
}

void run_dssp_postproc(const arguments &args, TitanDSSPAnalyzer &mol1_dssp,
                       TitanDSSPAnalyzer &mol2_dssp, info &info_output,
                       std::fstream &log) {
  if (!args.dssp.flag_dssp)
    return;

  bool exported = false;
  if (mol1_dssp.has_results()) {
    mol1_dssp.export_results(info_output.get_dssp_mol1_filename(), log);
    mol1_dssp.export_content_profile(
        info_output.get_dssp_mol1_content_filename(), log);
    mol1_dssp.export_segment_content(
        info_output.get_dssp_mol1_segments_filename(), log);
    mol1_dssp.export_bridge_registry(
        info_output.get_dssp_mol1_bridges_filename(), log);
    mol1_dssp.export_ladder_distribution(
        info_output.get_dssp_mol1_ladders_filename(), log);
    TitanUI::print_info("[DSSP] Exported mol1 summary to " +
                        info_output.get_dssp_mol1_filename());
    TitanUI::print_info("[DSSP] Exported mol1 content to " +
                        info_output.get_dssp_mol1_content_filename());
    exported = true;
  }
  if (mol2_dssp.has_results()) {
    mol2_dssp.export_results(info_output.get_dssp_mol2_filename(), log);
    mol2_dssp.export_content_profile(
        info_output.get_dssp_mol2_content_filename(), log);
    mol2_dssp.export_segment_content(
        info_output.get_dssp_mol2_segments_filename(), log);
    mol2_dssp.export_bridge_registry(
        info_output.get_dssp_mol2_bridges_filename(), log);
    mol2_dssp.export_ladder_distribution(
        info_output.get_dssp_mol2_ladders_filename(), log);
    TitanUI::print_info("[DSSP] Exported mol2 summary to " +
                        info_output.get_dssp_mol2_filename());
    TitanUI::print_info("[DSSP] Exported mol2 content to " +
                        info_output.get_dssp_mol2_content_filename());
    exported = true;
  }

  if (!exported && log.is_open()) {
    TitanUI::log_kv(log, "DSSP",
                    "No protein/protein-user molecule produced output");
  }
}

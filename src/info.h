#ifndef INFO_H
#define INFO_H
#include "arguments.h"
#include "groups.h"
#include <string>

using std::string;
class info {
  string Build, PdbFileName, TotalEnergyFileName, TotalEnergyTimeFileName,
      LogFile, EntropyFile, EntropyQuasiFile, group_1, group_2, stat_hbonds,
      stat_average_HB, stat_maxforce_res, dis_energy_2mol, res_list_hb_rec,
      res_list_hb_lg, dis_com_2mol, dipole_filename;
  double current_begin_time, current_end_time;
  double current_dt;

public:
  info(const arguments &Arguments)
      : Build{Arguments.get_build_number()}, PdbFileName{Arguments.pdb_name},
        TotalEnergyFileName{"cal.txt"}, TotalEnergyTimeFileName{"cal.xvg"},
        LogFile{"log.txt"}, EntropyFile{"out_entropy.dat"},
        EntropyQuasiFile{"out_entropy_quasiHarmonic.dat"}, group_1{"mol1"},
        group_2{"mol2"}, stat_hbonds{"stat_hydrogenbonds.dat"},
        stat_average_HB{"stat_average_HB.dat"},
        stat_maxforce_res{"_out_stat_maxforce_res.dat"},
        dis_energy_2mol{"out_dis_com_energy.dat"}, res_list_hb_rec{},
        res_list_hb_lg{}, dis_com_2mol{"out_dis_com.dat"},
        current_begin_time{1000000000000.}, current_end_time{}, current_dt{} {}
  string getBuild() const { return Build; }
  int get_info_group(const groups_cal &groups) {
    if (groups.get_parse_index_flag()) {
      group_1 = groups.group_name(groups.group_1_index());
      group_2 = groups.group_name(groups.group_2_index());
    } else {
    }
    return 1;
  }

  string get_pca_fel_filename() const { return "pca_fel.csv"; }
  string get_pca_fes_2d_filename() const { return "pca_fes_2d.dat"; }
  string get_tica_fes_2d_filename() const { return "tica_fes_2d.dat"; }
  string get_mbar_tica_fes_2d_filename() const { return "mbar_tica_fes_2d.dat"; }
  string get_mbar_pca_fes_2d_filename() const { return "mbar_pca_fes_2d.dat"; }
  string get_pca_zts_path_filename() const { return "pca_zts_path.csv"; }
  string get_tica_zts_path_filename() const { return "tica_zts_path.csv"; }
  string get_fes_2d_filename() const { return get_pca_fes_2d_filename(); }
  string get_cluster_representatives_filename() const {
    return "cluster_representatives.dat";
  }
  string get_tica_cluster_representatives_filename() const {
    return "tica_cluster_representatives.dat";
  }
  string get_mbar_tica_cluster_representatives_filename() const {
    return "mbar_tica_cluster_representatives.csv";
  }
  string get_mbar_pca_cluster_representatives_filename() const {
    return "mbar_pca_cluster_representatives.csv";
  }

  void setPdbName(const arguments &Arguments) {
    PdbFileName = Arguments.pdb_name;
  }
  string get_group_1_name() const { return group_1; }
  string get_group_2_name() const { return group_2; }
  string get_stathb_name() const { return stat_hbonds; }
  string get_stat_ave_hb_name() const { return stat_average_HB; }
  string get_stat_maxforce_name() const { return stat_maxforce_res; }
  string getPdbFileName() const { return PdbFileName; }
  string get_pdboutname(const arguments &Arguments) const {
    return Arguments.get_pdbOutput();
  }
  string getEntropyQuasi() const { return EntropyQuasiFile; }
  string get_res_list_hb_rec_filename() {
    return "out_list_res_HB_" + group_1 + ".dat";
  }
  string get_res_list_hb_lg_filename() {
    return "out_list_res_HB_" + group_2 + ".dat";
  }
  string getDisCom2Mol() const { return dis_com_2mol; }
  string getDisEner2Mol() const { return dis_energy_2mol; }
  string getDipoleFilename() const { return "_dipole_moment.dat"; }
  string getDistributionDistanceCharged() const {
    return "_distributionChargedAtoms.dat";
  }
  string getDistributionAngleDipoleEfield() const {
    return "_distributionDipoleEfield.dat";
  }
  string getOutputDistanceRes2COM() const { return "_distances_res_2COM.dat"; }
  string get_gbsa_snapshots_filename() const { return "gbsa_snapshots.dat"; }
  string get_gbsa_timeseries_avg_filename() const {
    return "gbsa_timeseries_avg.dat";
  }
  string get_gbsa_summary_filename() const { return "gbsa_summary.dat"; }
  string get_gbsa_decomposition_filename() const {
    return "gbsa_decomposition.dat";
  }
  string get_edecomp_mm_residue_matrix_elec_filename() const {
    return "edecomp_mm_residue_matrix_elec.csv";
  }
  string get_edecomp_mm_residue_matrix_vdw_filename() const {
    return "edecomp_mm_residue_matrix_vdw.csv";
  }
  string get_edecomp_mm_residue_matrix_total_filename() const {
    return "edecomp_mm_residue_matrix_total.csv";
  }
  string get_edecomp_mm_residue_edges_filename() const {
    return "edecomp_mm_residue_edges.csv";
  }
  string get_edecomp_mm_residue_ranking_filename() const {
    return "edecomp_mm_residue_ranking.csv";
  }
  string get_edecomp_mm_atom_edges_filename() const {
    return "edecomp_mm_atom_edges.csv";
  }
  string get_edecomp_mm_atom_ranking_filename() const {
    return "edecomp_mm_atom_ranking.csv";
  }
  string get_edecomp_mmgbsa_residue_map_filename() const {
    return "edecomp_mmgbsa_residue_map.csv";
  }
  string get_edecomp_mmgbsa_residue_ranking_filename() const {
    return "edecomp_mmgbsa_residue_ranking.csv";
  }
  string get_edecomp_mmgbsa_atom_map_filename() const {
    return "edecomp_mmgbsa_atom_map.csv";
  }
  string get_edecomp_mmgbsa_atom_ranking_filename() const {
    return "edecomp_mmgbsa_atom_ranking.csv";
  }
  string get_edecomp_mmpbsa_residue_map_filename() const {
    return "edecomp_mmpbsa_residue_map.csv";
  }
  string get_edecomp_mmpbsa_residue_ranking_filename() const {
    return "edecomp_mmpbsa_residue_ranking.csv";
  }
  string get_edecomp_mmpbsa_atom_map_filename() const {
    return "edecomp_mmpbsa_atom_map.csv";
  }
  string get_edecomp_mmpbsa_atom_ranking_filename() const {
    return "edecomp_mmpbsa_atom_ranking.csv";
  }
  string get_edecomp_summary_filename() const { return "edecomp_summary.dat"; }

  string get_DE_output_filename() { return "DE_out_vector.dat"; }
  string get_DE_log_filename() { return "DE_log.dat"; }
  string get_DE_pdb_vector_filename() { return "DE_vector.pdb"; }
  string get_pull_direction_aligned_pdb_filename() const {
    return "pull_direction_aligned_snapshot.pdb";
  }
  string get_pull_direction_system_info_filename() const {
    return "pull_direction_system_info.txt";
  }
  string get_pull_direction_status_filename() const {
    return "pull_direction_status.txt";
  }

  string get_cas_receptor_filename() const { return "cas_receptor.dat"; }
  string get_cas_ligand_filename() const { return "cas_ligand.dat"; }
  string get_cas_hotspot_pdb_filename() const {
    return "cas_hotspots_b_factor.pdb";
  }
  string get_cas_summary_filename() const { return "cas_summary.dat"; }
  string get_cas_performance_filename() const { return "cas_performance.dat"; }
  string get_hotspot_residue_ranking_filename() const {
    return "hotspot_residue_ranking.csv";
  }
  string get_hotspot_summary_filename() const { return "hotspot_summary.dat"; }
  string get_hotspot_bfactor_pdb_filename() const {
    return "hotspot_bfactor.pdb";
  }
  string get_titan_performance_summary_filename() const {
    return "titan_performance_summary.dat";
  }
  string get_titan_performance_modules_filename() const {
    return "titan_performance_modules.csv";
  }
  string get_titan_performance_frames_filename() const {
    return "titan_performance_frames.csv";
  }
  string get_uncertainty_summary_filename() const {
    return "uncertainty_summary.dat";
  }
  string get_cmap_uncertainty_filename() const { return "cmap_uncertainty.csv"; }
  string get_qnative_uncertainty_filename() const {
    return "qnative_uncertainty.csv";
  }
  string get_dihedral_uncertainty_filename() const {
    return "dihedral_uncertainty.csv";
  }
  string get_msd_uncertainty_filename() const { return "msd_uncertainty.csv"; }
  string get_s2_uncertainty_filename() const { return "s2_uncertainty.csv"; }
  string get_membrane_uncertainty_filename() const {
    return "membrane_uncertainty.csv";
  }
  string get_edecomp_uncertainty_filename() const {
    return "edecomp_uncertainty.csv";
  }
  string get_titan_outputs_manifest_filename() const {
    return "titan_outputs_manifest.csv";
  }
  string get_titan_run_config_summary_filename() const {
    return "titan_run_config_summary.dat";
  }
  string get_titan_schema_filename() const { return "titan_schema.json"; }
  string get_dssp_mol1_filename() const { return "dssp_mol1.dat"; }
  string get_dssp_mol2_filename() const { return "dssp_mol2.dat"; }
  string get_dssp_mol1_content_filename() const {
    return "dssp_mol1_content.dat";
  }
  string get_dssp_mol2_content_filename() const {
    return "dssp_mol2_content.dat";
  }
  string get_dssp_mol1_segments_filename() const {
    return "dssp_mol1_segments.csv";
  }
  string get_dssp_mol2_segments_filename() const {
    return "dssp_mol2_segments.csv";
  }
  string get_dssp_mol1_bridges_filename() const {
    return "dssp_mol1_bridges.csv";
  }
  string get_dssp_mol2_bridges_filename() const {
    return "dssp_mol2_bridges.csv";
  }
  string get_dssp_mol1_ladders_filename() const {
    return "dssp_mol1_ladders.csv";
  }
  string get_dssp_mol2_ladders_filename() const {
    return "dssp_mol2_ladders.csv";
  }
  string get_dccm_block_uncertainty_filename() const {
    return "dccm_block_uncertainty.csv";
  }
  string get_dccm_block_values_filename() const {
    return "dccm_block_values.csv";
  }
  string get_network_communities_filename() const {
    return "network_communities.csv";
  }
  string get_network_gc_edges_filename() const {
    return "network_gc_edges.csv";
  }
  string get_network_mi_edges_filename() const {
    return "raw_mi_edges.csv";
  }
  string get_network_te_edges_filename() const {
    return "network_te_edges.csv";
  }
  string get_graph_summary_filename() const { return "graph_summary.dat"; }
  string get_graph_edges_filename(const std::string &source) const {
    return "graph_" + source + "_edges.csv";
  }
  string get_graph_nodes_filename(const std::string &source) const {
    return "graph_" + source + "_nodes.csv";
  }
  string get_graph_communities_filename(const std::string &source) const {
    return "graph_" + source + "_communities.csv";
  }
  string get_graph_community_summary_filename(const std::string &source) const {
    return "graph_" + source + "_community_summary.csv";
  }
  string get_graph_supergraph_filename(const std::string &source) const {
    return "graph_" + source + "_supergraph.csv";
  }
  string get_graph_bottleneck_edges_filename(const std::string &source) const {
    return "graph_" + source + "_bottleneck_edges.csv";
  }
  string get_graph_bottleneck_nodes_filename(const std::string &source) const {
    return "graph_" + source + "_bottleneck_nodes.csv";
  }
  string get_corr_knn_summary_filename() const {
    return "corr_knn_summary.dat";
  }
  string get_corr_knn_mi_pairs_filename() const {
    return "corr_knn_mi_pairs.csv";
  }
  string get_corr_knn_gc_pairs_filename() const {
    return "corr_knn_gc_pairs.csv";
  }
  string get_corr_knn_te_pairs_filename() const {
    return "corr_knn_te_pairs.csv";
  }
  string get_cmap_matrix_filename(const arguments &Arguments,
                                  const std::vector<string> &input_line,
                                  int input_name) const {
    if (Arguments.flag_separation_results)
      return "contact_probability_matrix.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "contact_probability_matrix_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "contact_probability_matrix.csv";
  }
  string get_cmap_edges_filename(const arguments &Arguments,
                                 const std::vector<string> &input_line,
                                 int input_name) const {
    if (Arguments.flag_separation_results)
      return "contact_probability_edges.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "contact_probability_edges_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "contact_probability_edges.csv";
  }
  string get_cmap_summary_filename(const arguments &Arguments,
                                   const std::vector<string> &input_line,
                                   int input_name) const {
    if (Arguments.flag_separation_results)
      return "contact_probability_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "contact_probability_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "contact_probability_summary.dat";
  }
  string get_cmap_timeseries_filename(const arguments &Arguments,
                                      const std::vector<string> &input_line,
                                      int input_name) const {
    if (Arguments.flag_separation_results)
      return "contact_count_timeseries.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "contact_count_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "contact_count_timeseries.csv";
  }
  string get_cmap_occupancy_filename(const arguments &Arguments,
                                     const std::vector<string> &input_line,
                                     int input_name) const {
    if (Arguments.flag_separation_results)
      return "contact_residue_occupancy.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "contact_residue_occupancy_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "contact_residue_occupancy.csv";
  }
  string get_cmap_matrix_average_filename() const {
    return "contact_probability_matrix_average.csv";
  }
  string get_cmap_edges_average_filename() const {
    return "contact_probability_edges_average.csv";
  }
  string get_cmap_summary_average_filename() const {
    return "contact_probability_summary_average.dat";
  }
  string get_cmap_occupancy_average_filename() const {
    return "contact_residue_occupancy_average.csv";
  }
  string get_interface_filename(const std::string &stem,
                                const arguments &Arguments,
                                const std::vector<string> &input_line,
                                int input_name) const {
    if (Arguments.flag_separation_results)
      return stem + ".csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input) {
      return stem + "_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    }
    return stem + ".csv";
  }
  string get_interface_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_state_timeseries", Arguments,
                                  input_line, input_name);
  }
  string get_interface_populations_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_state_populations", Arguments,
                                  input_line, input_name);
  }
  string get_interface_segments_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_segment_contacts", Arguments,
                                  input_line, input_name);
  }
  string get_interface_segment_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_segment_timeseries", Arguments,
                                  input_line, input_name);
  }
  string get_interface_metrics_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_metrics", Arguments, input_line,
                                  input_name);
  }
  string get_interface_sasa_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return get_interface_filename("interface_sasa_timeseries", Arguments,
                                  input_line, input_name);
  }
  string get_interface_timeseries_average_filename() const {
    return "interface_state_timeseries_average.csv";
  }
  string get_interface_populations_average_filename() const {
    return "interface_state_populations_average.csv";
  }
  string get_interface_segments_average_filename() const {
    return "interface_segment_contacts_average.csv";
  }
  string get_interface_metrics_average_filename() const {
    return "interface_metrics_average.csv";
  }
  string get_interface_sasa_average_filename() const {
    return "interface_sasa_timeseries_average.csv";
  }
  string get_qnative_timeseries_filename(const arguments &Arguments,
                                         const std::vector<string> &input_line,
                                         int input_name) const {
    if (Arguments.flag_separation_results)
      return "qnative_timeseries.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "qnative_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "qnative_timeseries.csv";
  }
  string get_qnative_native_contacts_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "qnative_native_contacts.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "qnative_native_contacts_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "qnative_native_contacts.csv";
  }
  string get_qnative_occupancy_filename(const arguments &Arguments,
                                        const std::vector<string> &input_line,
                                        int input_name) const {
    if (Arguments.flag_separation_results)
      return "qnative_contact_occupancy.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "qnative_contact_occupancy_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "qnative_contact_occupancy.csv";
  }
  string get_qnative_summary_filename(const arguments &Arguments,
                                      const std::vector<string> &input_line,
                                      int input_name) const {
    if (Arguments.flag_separation_results)
      return "qnative_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "qnative_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "qnative_summary.dat";
  }
  string get_qnative_timeseries_average_filename() const {
    return "qnative_timeseries_average.csv";
  }
  string get_qnative_native_contacts_average_filename() const {
    return "qnative_native_contacts_average.csv";
  }
  string get_qnative_occupancy_average_filename() const {
    return "qnative_contact_occupancy_average.csv";
  }
  string get_qnative_summary_average_filename() const {
    return "qnative_summary_average.dat";
  }
  string get_dihedral_timeseries_filename(const arguments &Arguments,
                                          const std::vector<string> &input_line,
                                          int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_backbone_timeseries.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_backbone_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_backbone_timeseries.csv";
  }
  string get_dihedral_ramachandran_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_ramachandran.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_ramachandran_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_ramachandran.csv";
  }
  string get_dihedral_summary_filename(const arguments &Arguments,
                                       const std::vector<string> &input_line,
                                       int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "dihedral_summary.dat";
  }
  string get_dihedral_feature_matrix_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_feature_matrix.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_feature_matrix_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_feature_matrix.csv";
  }
  string get_features_by_frame_filename(const arguments &Arguments,
                                        const std::vector<string> &input_line,
                                        int input_name) const {
    if (Arguments.flag_separation_results)
      return "features_by_frame.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "features_by_frame_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "features_by_frame.csv";
  }
  string get_features_metadata_filename(const arguments &Arguments,
                                        const std::vector<string> &input_line,
                                        int input_name) const {
    if (Arguments.flag_separation_results)
      return "features_metadata.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "features_metadata_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "features_metadata.csv";
  }
  string get_feature_export_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "feature_export_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "feature_export_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "feature_export_summary.dat";
  }
  string get_dihedral_sidechain_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_sidechain_timeseries.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_sidechain_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_sidechain_timeseries.csv";
  }
  string get_dihedral_rotamer_populations_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_rotamer_populations.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_rotamer_populations_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_rotamer_populations.csv";
  }
  string get_dihedral_rotamer_transitions_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "dihedral_rotamer_transitions.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "dihedral_rotamer_transitions_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "dihedral_rotamer_transitions.csv";
  }
  string get_msd_timeseries_filename(const arguments &Arguments,
                                     const std::vector<string> &input_line,
                                     int input_name) const {
    if (Arguments.flag_separation_results)
      return "msd_timeseries.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "msd_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "msd_timeseries.dat";
  }
  string get_msd_diffusion_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "msd_diffusion_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "msd_diffusion_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "msd_diffusion_summary.dat";
  }
  string get_msd_alpha_fit_filename(const arguments &Arguments,
                                    const std::vector<string> &input_line,
                                    int input_name) const {
    if (Arguments.flag_separation_results)
      return "msd_alpha_fit.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "msd_alpha_fit_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "msd_alpha_fit.dat";
  }
  string indexed_analysis_filename(const arguments &Arguments,
                                   const std::vector<string> &input_line,
                                   int input_name, const string &stem,
                                   const string &extension) const {
    if (Arguments.flag_separation_results)
      return stem + extension;
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return stem + "_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             extension;
    return stem + extension;
  }
  string get_rotational_msd_filename(const arguments &Arguments,
                                     const std::vector<string> &input_line,
                                     int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "rotational_msd", ".dat");
  }
  string get_rotational_quaternion_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "rotational_quaternion_timeseries",
                                     ".csv");
  }
  string get_rotational_covariance_fit_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "rotational_covariance_fit", ".csv");
  }
  string get_rotational_diffusion_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "rotational_diffusion_summary", ".dat");
  }
  string get_membrane_leaflets_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_leaflets", ".csv");
  }
  string get_membrane_apl_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_apl_timeseries", ".csv");
  }
  string get_membrane_apl_per_lipid_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_apl_per_lipid", ".csv");
  }
  string get_membrane_thickness_timeseries_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_thickness_timeseries", ".csv");
  }
  string get_membrane_thickness_map_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_thickness_map", ".csv");
  }
  string get_membrane_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_summary", ".dat");
  }
  string get_membrane_order_parameters_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_order_parameters", ".csv");
  }
  string get_membrane_interdigitation_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_interdigitation", ".dat");
  }
  string get_membrane_flipflop_events_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_flipflop_events", ".csv");
  }
  string get_membrane_protein_lipid_contacts_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_protein_lipid_contacts", ".csv");
  }
  string get_membrane_lipid_enrichment_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_lipid_enrichment", ".csv");
  }
  string get_membrane_curvature_map_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_curvature_map", ".csv");
  }
  string get_membrane_proximity_coupling_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_proximity_coupling", ".csv");
  }
  string get_membrane_coupling_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "membrane_coupling_summary", ".dat");
  }
  string get_membrane_summary_average_filename() const {
    return "membrane_summary_average.dat";
  }
  string get_membrane_thickness_map_average_filename() const {
    return "membrane_thickness_map_average.csv";
  }
  string get_membrane_order_parameters_average_filename() const {
    return "membrane_order_parameters_average.csv";
  }
  string get_membrane_lipid_enrichment_average_filename() const {
    return "membrane_lipid_enrichment_average.csv";
  }
  string get_membrane_curvature_map_average_filename() const {
    return "membrane_curvature_map_average.csv";
  }
  string get_membrane_proximity_coupling_average_filename() const {
    return "membrane_proximity_coupling_average.csv";
  }
  string get_membrane_coupling_summary_average_filename() const {
    return "membrane_coupling_summary_average.dat";
  }
  string get_s2_order_parameters_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "s2_order_parameters", ".csv");
  }
  string get_s2_correlation_functions_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "s2_correlation_functions", ".dat");
  }
  string get_s2_summary_filename(const arguments &Arguments,
                                 const std::vector<string> &input_line,
                                 int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "s2_summary", ".dat");
  }
  string get_s2_bfactor_filename(const arguments &Arguments,
                                 const std::vector<string> &input_line,
                                 int input_name) const {
    return indexed_analysis_filename(Arguments, input_line, input_name,
                                     "s2_bfactor", ".pdb");
  }
  string get_s2_order_parameters_average_filename() const {
    return "s2_order_parameters_average.csv";
  }
  string get_s2_summary_average_filename() const {
    return "s2_summary_average.dat";
  }
  string get_s2_bfactor_average_filename() const {
    return "s2_bfactor_average.pdb";
  }
  string get_network_community_summary_filename() const {
    return "network_community_summary.csv";
  }
  string get_memory_report_filename() const { return "memory_report.dat"; }
  string get_network_supergraph_filename() const {
    return "network_supergraph.csv";
  }
  string get_network_community_id_pdb_filename() const {
    return "network_community_id_bfactor.pdb";
  }
  string get_network_participation_pdb_filename() const {
    return "network_participation_coeff_bfactor.pdb";
  }
  string get_rmsd_filename(const arguments &Arguments,
                           const std::vector<string> &input_line,
                           int input_name) const {
    if (Arguments.flag_separation_results)
      return "rmsd.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "rmsd_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "rmsd.dat";
  }
  string get_rmsf_filename(const arguments &Arguments,
                           const std::vector<string> &input_line,
                           int input_name) const {
    if (Arguments.flag_separation_results)
      return "rmsf.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "rmsf_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "rmsf.dat";
  }
  string get_rg2_filename(const arguments &Arguments,
                          const std::vector<string> &input_line,
                          int input_name) const {
    if (Arguments.flag_separation_results)
      return "rg2.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "rg2_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "rg2.dat";
  }
  string get_rg2_summary_filename(const arguments &Arguments,
                                  const std::vector<string> &input_line,
                                  int input_name) const {
    if (Arguments.flag_separation_results)
      return "rg2_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "rg2_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "rg2_summary.dat";
  }
  string get_rg2_hist_filename(const arguments &Arguments,
                               const std::vector<string> &input_line,
                               int input_name) const {
    if (Arguments.flag_separation_results)
      return "rg2_hist.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "rg2_hist_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "rg2_hist.dat";
  }
  string get_rg2_all_filename() const { return "rg2_all.dat"; }
  string get_rg2_summary_all_filename() const { return "rg2_summary_all.dat"; }
  string get_rg2_hist_all_filename() const { return "rg2_hist_all.dat"; }
  string get_lj_timeseries_filename(const arguments &Arguments,
                                    const std::vector<string> &input_line,
                                    int input_name) const {
    if (Arguments.flag_separation_results)
      return "lj_timeseries.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "lj_timeseries_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "lj_timeseries.dat";
  }
  string get_lj_rho_r_filename(const arguments &Arguments,
                               const std::vector<string> &input_line,
                               int input_name) const {
    if (Arguments.flag_separation_results)
      return "lj_cluster_rho_r.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "lj_cluster_rho_r_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "lj_cluster_rho_r.dat";
  }
  string get_lj_features_filename(const arguments &Arguments,
                                  const std::vector<string> &input_line,
                                  int input_name) const {
    if (Arguments.flag_separation_results)
      return "lj_features.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "lj_features_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : input_name) +
             ".dat";
    return "lj_features.dat";
  }
  string get_lj_summary_filename(const arguments &Arguments,
                                 const std::vector<string> &input_line,
                                 int input_name) const {
    if (Arguments.flag_separation_results)
      return "lj_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "lj_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "lj_summary.dat";
  }

  string get_sq_summary_filename(const arguments &Arguments,
                                 const std::vector<string> &input_line,
                                 int input_name) const {
    if (Arguments.flag_separation_results)
      return "sq_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "sq_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "sq_summary.dat";
  }
  string get_sq_rdf_filename(const arguments &Arguments,
                             const std::vector<string> &input_line,
                             int input_name, const std::string &channel_a,
                             const std::string &channel_b) const {
    const std::string label =
        (channel_a == channel_b) ? channel_a : (channel_a + "__" + channel_b);
    if (Arguments.flag_separation_results)
      return "sq_rdf_" + label + ".dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "sq_rdf_" + label + "_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "sq_rdf_" + label + ".dat";
  }
  string get_sq_structure_factor_filename(const arguments &Arguments,
                                          const std::vector<string> &input_line,
                                          int input_name,
                                          const std::string &channel_a,
                                          const std::string &channel_b) const {
    const std::string label =
        (channel_a == channel_b) ? channel_a : (channel_a + "__" + channel_b);
    if (Arguments.flag_separation_results)
      return "sq_structure_factor_" + label + ".dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "sq_structure_factor_" + label + "_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "sq_structure_factor_" + label + ".dat";
  }

  string get_hbond_kinetics_summary_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name) const {
    if (Arguments.flag_separation_results)
      return "hbond_kinetics_summary.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "hbond_kinetics_summary_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "hbond_kinetics_summary.dat";
  }
  string get_hbond_kinetics_pairs_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name, const std::string &kind) const {
    if (Arguments.flag_separation_results)
      return "hbond_kinetics_" + kind + "_pairs.csv";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "hbond_kinetics_" + kind + "_pairs_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".csv";
    return "hbond_kinetics_" + kind + "_pairs.csv";
  }
  string get_hbond_kinetics_ct_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name, const std::string &kind) const {
    if (Arguments.flag_separation_results)
      return "hbond_kinetics_" + kind + "_ct.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "hbond_kinetics_" + kind + "_ct_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "hbond_kinetics_" + kind + "_ct.dat";
  }
  string get_hbond_kinetics_lifetimes_filename(
      const arguments &Arguments, const std::vector<string> &input_line,
      int input_name, const std::string &kind) const {
    if (Arguments.flag_separation_results)
      return "hbond_kinetics_" + kind + "_lifetimes.dat";
    const bool multi_input = input_line.size() > 1 || Arguments.flag_listfile ||
                             Arguments.flag_pattern_input || Arguments.flag_xdr;
    if (multi_input)
      return "hbond_kinetics_" + kind + "_lifetimes_" +
             std::to_string(input_name >= Arguments.b_input
                                ? input_name - Arguments.b_input
                                : 0) +
             ".dat";
    return "hbond_kinetics_" + kind + "_lifetimes.dat";
  }

  string get_global_water_bridges_timeseries() const {
    return "global_water_bridges_timeseries.csv";
  }
  string get_global_water_bridges_summary() const {
    return "global_water_bridges_summary.txt";
  }
  string get_global_water_survival() const {
    return "global_water_survival.csv";
  }
  string get_water_pdb_filename(const arguments &Arguments,
                                const std::vector<string> &input_line,
                                int input_name) const {
    if (Arguments.flag_sep_opdb == true && !input_line.empty()) {
      size_t posfile = input_name - Arguments.b_input;
      return Arguments.get_pdbOutput().substr(
                 0, Arguments.get_pdbOutput().length() - 4) +
             "_waters_" + std::to_string(posfile) + ".pdb";
    }
    return "waters_out" +
           (input_line.empty()
                ? ""
                : "_" + input_line[input_name - Arguments.b_input]) +
           ".pdb";
  }

  string get_pca_filename(const arguments &Arguments,
                          const std::vector<string> &input_line,
                          int input_name) const {
    if (Arguments.flag_sep_opdb == true && !input_line.empty()) {
      size_t posfile = input_name - Arguments.b_input;
      return "pca_fel_" + std::to_string(posfile) + ".csv";
    }
    return "pca_fel" +
           (input_line.empty()
                ? ""
                : "_" + input_line[input_name - Arguments.b_input]) +
           ".csv";
  }

  void set_current_begin_time(double t) { current_begin_time = t; }
  double get_current_begin_time() const { return current_begin_time; }
  void set_current_end_time(double t) { current_end_time = t; }
  double get_current_end_time() const { return current_end_time; }
  void set_current_dt(double dt) { current_dt = dt; }
  double get_current_dt() const { return current_dt; }
  string get_ml_features_filename() const { return "titan_ml_features.csv"; }
  string get_ml_prediction_filename() const {
    return "titan_rf_prediction.txt";
  }
  string get_tica_eigenvalues_filename() const {
    return "tica_eigenvalues.dat";
  }
  string get_tica_eigenvectors_filename() const {
    return "tica_eigenvectors.dat";
  }
  string get_tica_projections_filename() const {
    return "tica_projections.csv";
  }
  string get_umap_embedding_filename() const {
    return "titan_umap_embedding.csv";
  }
  string get_umap_fes_2d_filename() const { return "umap_fes_2d.dat"; }
  string get_umap_cluster_representatives_filename() const {
    return "umap_cluster_representatives.csv";
  }
  string get_umap_equilibrium_summary_filename() const {
    return "umap_equilibrium_summary.dat";
  }
  // --- MSM Phase 2 output files ---
  string get_msm_tpm_filename() const { return "msm_tpm.dat"; }
  string get_msm_discrete_traj_filename() const {
    return "msm_discrete_traj.dat";
  }
  string get_msm_timescales_filename() const { return "msm_timescales.dat"; }
  string get_msm_centroids_filename() const { return "msm_centroids.dat"; }
  string get_msm_its_filename() const { return "msm_its_scan.dat"; }
  string get_msm_macrostates_filename() const { return "msm_macrostates.dat"; }
  // Python-friendly extended outputs
  string get_msm_macrostate_traj_filename() const {
    return "msm_macrostate_traj.dat";
  }
  string get_msm_count_matrix_filename() const {
    return "msm_count_matrix.dat";
  }
  string get_msm_stationary_filename() const { return "msm_stationary.dat"; }
  string get_msm_equilibrium_summary_filename() const {
    return "msm_equilibrium_summary.dat";
  }
  string get_dccm_equilibrium_context_filename() const {
    return "dccm_equilibrium_context.dat";
  }
  string get_allostery_equilibrium_ranking_filename() const {
    return "allostery_equilibrium_ranking.csv";
  }
  string get_corr_knn_equilibrium_ranking_filename() const {
    return "corr_knn_equilibrium_ranking.csv";
  }
  string get_lspmd_equilibrium_ranking_filename() const {
    return "lspmd_equilibrium_ranking.csv";
  }
  string get_msm_its_tpm_dir() const { return "msm_its_tpm"; }
  string get_out_mindis_subgroups() { return "mininum_distance_"; }
  string get_out_histogram_mindis() { return "histo_mindistance_"; }
  // LLPS output files
  string get_llps_timeseries_filename() const { return "llps_timeseries.dat"; }
  string get_llps_ct_filename() const { return "llps_autocorr.dat"; }
  string get_llps_degree_dist_filename() const {
    return "llps_degree_dist.dat";
  }
  string get_llps_contact_map_filename() const {
    return "llps_contact_map.dat";
  }
  string get_llps_chain_residue_contact_map_filename() const {
    return "llps_contact_map_chain_residue.csv";
  }
  string get_llps_performance_report_filename() const {
    return "llps_performance_report.dat";
  }
  string get_llps_summary_filename() const { return "llps_summary.dat"; }
  string get_llps_density_pro_filename() const {
    return "llps_density_protein.dx";
  }
  string get_llps_density_rna_filename() const { return "llps_density_rna.dx"; }
  string get_llps_polymer_scaling_filename() const {
    return "polymer_scaling.dat";
  }
  string get_llps_sticker_hotspots_filename() const {
    return "llps_sticker_hotspots.csv";
  }
  // GIST output prefix
  string get_gist_prefix() const { return "gist"; }
  // LLPS DBSCAN output files
  string get_llps_dbscan_droplets_filename() const { return "llps_dbscan_droplets.pdb"; }
  string get_llps_dbscan_timeseries_filename() const { return "llps_dbscan_timeseries.csv"; }
  string get_llps_membership_autocorr_protein_filename() const {
    return "llps_membership_autocorr_protein.dat";
  }
  string get_llps_membership_autocorr_rna_filename() const {
    return "llps_membership_autocorr_rna.dat";
  }
  string get_llps_residence_times_filename() const {
    return "llps_residence_times.csv";
  }
  string get_llps_entry_exit_rates_filename() const {
    return "llps_entry_exit_rates.dat";
  }
  string get_llps_cluster_lineage_filename() const {
    return "llps_cluster_lineage.csv";
  }
  string get_llps_fusion_fission_filename() const {
    return "llps_fusion_fission.dat";
  }
  string get_llps_droplet_genealogy_summary_filename() const {
    return "llps_droplet_genealogy_summary.dat";
  }
  string get_llps_frap_recovery_proxy_filename() const {
    return "llps_frap_recovery_proxy.dat";
  }
  string get_llps_droplet_shape_timeseries_filename() const {
    return "llps_droplet_shape_timeseries.csv";
  }
  string get_llps_sticker_contact_lifetimes_filename() const {
    return "llps_sticker_contact_lifetimes.dat";
  }
  string get_llps_prna_contact_lifetimes_filename() const {
    return "llps_protein_rna_contact_lifetimes.dat";
  }
  string get_llps_neighbor_rewiring_filename() const {
    return "llps_neighbor_rewiring.dat";
  }
  string get_llps_interface_turnover_filename() const {
    return "llps_interface_turnover.dat";
  }
  string get_llps_dense_msd_protein_filename() const {
    return "llps_dense_msd_protein.dat";
  }
  string get_llps_dense_msd_rna_filename() const {
    return "llps_dense_msd_rna.dat";
  }
  string get_llps_liquidity_summary_filename() const {
    return "llps_liquidity_summary.dat";
  }
  string get_llps_rna_compaction_binding_summary_filename() const {
    return "llps_rna_compaction_binding_summary.dat";
  }
  string get_llps_rna_compaction_binding_binned_filename() const {
    return "llps_rna_compaction_binding_binned.csv";
  }
  string get_llps_rna_chain_compaction_summary_filename() const {
    return "llps_rna_chain_compaction_summary.csv";
  }
  string get_llps_rna_intrachain_contacts_summary_filename() const {
    return "llps_rna_intrachain_contacts_summary.csv";
  }
  string get_llps_rna_compaction_trace_filename() const {
    return "llps_rna_compaction_trace.csv";
  }
  string get_llps_rna_energy_compaction_summary_filename() const {
    return "llps_rna_energy_compaction_summary.dat";
  }
  string get_llps_rna_energy_compaction_binned_filename() const {
    return "llps_rna_energy_compaction_binned.csv";
  }
  string get_llps_rna_binding_energy_trace_filename() const {
    return "llps_rna_binding_energy_trace.csv";
  }
  string get_llps_prna_pair_energy_trace_filename() const {
    return "llps_prna_pair_energy_trace.csv";
  }
  string get_llps_rna_information_summary_filename() const {
    return "llps_rna_information_summary.dat";
  }
  string get_llps_rna_information_per_chain_filename() const {
    return "llps_rna_information_per_chain.csv";
  }
  string get_llps_rna_compaction_state_histogram_filename() const {
    return "llps_rna_compaction_state_histogram.csv";
  }
  string get_llps_rna_partner_state_histogram_filename() const {
    return "llps_rna_partner_state_histogram.csv";
  }
  string get_llps_rna_transition_information_filename() const {
    return "llps_rna_transition_information.csv";
  }
  string get_llps_rna_binding_compaction_information_filename() const {
    return "llps_rna_binding_compaction_information.csv";
  }
  string get_llps_rna_partner_state_trace_filename() const {
    return "llps_rna_partner_state_trace.csv";
  }
  string get_llps_rna_topology_summary_filename() const {
    return "llps_rna_topology_summary.dat";
  }
  string get_llps_rna_topology_per_chain_filename() const {
    return "llps_rna_topology_per_chain.csv";
  }
  string get_llps_rna_topology_binding_binned_filename() const {
    return "llps_rna_topology_binding_binned.csv";
  }
  string get_llps_rna_topology_state_histogram_filename() const {
    return "llps_rna_topology_state_histogram.csv";
  }
  string get_llps_rna_bridge_lifetimes_filename() const {
    return "llps_rna_bridge_lifetimes.dat";
  }
  string get_llps_rna_clamp_lifetimes_filename() const {
    return "llps_rna_clamp_lifetimes.dat";
  }
  string get_llps_rna_loop_closure_efficiency_filename() const {
    return "llps_rna_loop_closure_efficiency.dat";
  }
  string get_llps_rna_local_competition_summary_filename() const {
    return "llps_rna_local_competition_summary.dat";
  }
  string get_llps_rna_topology_trace_filename() const {
    return "llps_rna_topology_trace.csv";
  }
  string get_llps_rna_bridge_trace_filename() const {
    return "llps_rna_bridge_trace.csv";
  }
  string get_llps_rna_clamp_trace_filename() const {
    return "llps_rna_clamp_trace.csv";
  }
  string get_llps_chain_membership_trace_filename() const {
    return "llps_chain_membership_trace.csv";
  }
  string get_llps_chain_contact_trace_filename() const {
    return "llps_chain_contact_trace.csv";
  }
  string get_llps_phase_partition_summary_filename() const {
    return "llps_phase_partition_summary.dat";
  }
  string get_llps_radial_profiles_filename() const {
    return "llps_radial_profiles.csv";
  }
  string get_llps_slab_profiles_filename() const {
    return "llps_slab_profiles.csv";
  }
  string get_llps_composition_timeseries_filename() const {
    return "llps_composition_timeseries.csv";
  }
  string get_llps_contact_taxonomy_filename() const {
    return "llps_contact_taxonomy.csv";
  }
  string get_llps_contact_taxonomy_summary_filename() const {
    return "llps_contact_taxonomy_summary.dat";
  }
  string get_llps_convergence_report_filename() const {
    return "llps_convergence_report.dat";
  }
  string get_llps_paper_summary_md_filename() const {
    return "llps_paper_summary.md";
  }
  string get_llps_paper_summary_json_filename() const {
    return "llps_paper_summary.json";
  }
  string get_llps_paper_fractal_timeseries_filename() const {
    return "llps_paper_fractal_timeseries.csv";
  }
  string get_llps_paper_fractal_scales_filename() const {
    return "llps_paper_fractal_scales.csv";
  }
  string get_llps_paper_fractal_summary_filename() const {
    return "llps_paper_fractal_summary.dat";
  }
  string get_llps_paper_fractal_readme_filename() const {
    return "llps_paper_fractal_readme.md";
  }
  // IFP output file
  string get_ifp_filename() const { return "ifp_occupancy.csv"; }
  string get_ifp_rna_annotations_filename() const {
    return "ifp_rna_annotations.csv";
  }
  string get_ifp_pbctopo_frame_filter_filename() const {
    return "ifp_pbctopo_frame_filter.csv";
  }
  string get_ifp_pbctopo_filter_summary_filename() const {
    return "ifp_pbctopo_filter_summary.csv";
  }
  string get_nucleic_acid_interactions_filename() const {
    return "nucleic_acid_interactions.csv";
  }
  string get_awh_rc_summary_filename() const {
    return "awh_rc_bin_summary.dat";
  }
  string get_awh_ifp_by_bin_filename() const {
    return "awh_ifp_by_bin.csv";
  }
  string get_awh_ifp_heatmap_filename() const {
    return "awh_ifp_heatmap.dat";
  }
  string get_awh_bin_structural_summary_filename() const {
    return "awh_bin_structural_summary.dat";
  }
  string get_awh_bin_top_contacts_filename() const {
    return "awh_bin_top_contacts.csv";
  }
  string get_awh_walker_stability_by_bin_filename() const {
    return "awh_walker_stability_by_bin.dat";
  }
  string get_awh_walker_pair_overlap_filename() const {
    return "awh_walker_pair_overlap.dat";
  }
  string get_awh_rc_frame_map_filename() const {
    return "awh_rc_frame_map.dat";
  }
  string get_awh_frame_assignments_filename() const {
    return "awh_frame_assignments.dat";
  }
  string get_awh_walker_diagnostics_filename() const {
    return "awh_walker_diagnostics.dat";
  }
  string get_awh_rc_overlap_filename() const {
    return "awh_rc_overlap_diagnostics.dat";
  }
  string get_awh_exploratory_context_filename() const {
    return "awh_exploratory_context.dat";
  }
  string get_awh_rep_pdb_prefix() const {
    return "awh_rep_bin";
  }
  // DHAM output files
  string get_dham_pi_filename() const { return "dham_stationary_distribution.dat"; }
  string get_dham_fes_filename() const { return "dham_free_energy_profile.dat"; }
  string get_dham_tpm_filename() const { return "dham_unbiased_tpm.dat"; }
  string get_dham_window_audit_filename() const { return "dham_window_audit.dat"; }
  string get_mbar_frame_weights_filename() const { return "mbar_frame_weights.dat"; }
  string get_mbar_summary_filename() const { return "mbar_summary.dat"; }
  string get_mbar_window_audit_filename() const { return "mbar_window_audit.dat"; }
  string get_mbar_cv_filename() const { return "mbar_cv.dat"; }
  // TPT output files
  string get_tpt_summary_filename() const { return "tpt_summary.dat"; }
  string get_tpt_committor_filename() const { return "tpt_committor.dat"; }
  string get_tpt_flux_edges_filename() const { return "tpt_flux_edges.dat"; }
  string get_tpt_paths_filename() const { return "tpt_paths.dat"; }
  string get_tpt_state_labels_filename() const { return "tpt_state_labels.dat"; }
  string get_tpt_kinetics_filename() const { return "tpt_kinetics.dat"; }
  string get_tpt_reactive_fes_2d_filename() const { return "tpt_reactive_fes_2d.dat"; }
  string get_tpt_flux_map_2d_filename() const { return "tpt_flux_map_2d.dat"; }
  string get_tpt_bootstrap_summary_filename() const { return "tpt_bootstrap_summary.dat"; }
  string get_tpt_bootstrap_states_filename() const { return "tpt_bootstrap_states.dat"; }
  string get_tpt_all_pairs_kinetics_filename() const { return "tpt_all_pairs_kinetics.dat"; }
  string get_tpt_path_clusters_filename() const { return "tpt_path_clusters.dat"; }
  string get_tpt_bottleneck_edges_filename() const { return "tpt_bottleneck_edges.dat"; }
  string get_tpt_bottleneck_states_filename() const { return "tpt_bottleneck_states.dat"; }
  string get_tpt_equilibrium_overlay_filename() const {
    return "tpt_equilibrium_overlay.dat";
  }
  string get_validation_summary_filename() const { return "validation_summary.dat"; }
  string get_validation_ck_summary_filename() const {
    return "validation_ck_summary.dat";
  }
  string get_validation_ck_states_filename() const {
    return "validation_ck_states.dat";
  }
  string get_validation_vamp2_ranking_filename() const {
    return "validation_vamp2_ranking.csv";
  }
  string get_validation_selected_model_filename() const {
    return "validation_selected_model.dat";
  }
  string get_validation_msm_bootstrap_summary_filename() const {
    return "validation_msm_bootstrap_summary.dat";
  }
  string get_validation_msm_bootstrap_states_filename() const {
    return "validation_msm_bootstrap_states.dat";
  }
  string get_validation_tpt_bootstrap_summary_filename() const {
    return "validation_tpt_bootstrap_summary.dat";
  }
  string get_validation_tpt_bootstrap_states_filename() const {
    return "validation_tpt_bootstrap_states.dat";
  }
};
#endif

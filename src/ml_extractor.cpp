#include "ml_extractor.h"
#include "titan_rf_model.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>

MLFeatureExtractor::MLFeatureExtractor() {
  // Kyte-Doolittle Hydrophobicity Scale
  hydrophobicity_scale = {
      {"ALA", 1.8},  {"ARG", -4.5}, {"ASN", -3.5}, {"ASP", -3.5}, {"CYS", 2.5},
      {"GLN", -3.5}, {"GLU", -3.5}, {"GLY", -0.4}, {"HIS", -3.2}, {"ILE", 4.5},
      {"LEU", 3.8},  {"LYS", -3.9}, {"MET", 1.9},  {"PHE", 2.8},  {"PRO", -1.6},
      {"SER", -0.8}, {"THR", -0.7}, {"TRP", -0.9}, {"TYR", -1.3}, {"VAL", 4.2}};

  // User's 16 Rare & Strong Motifs (Score = 1.0)
  motif_dictionary = {
      {"ALA-PRO-SER", 1.0}, {"ASN-TYR-GLY", 1.0}, {"GLY-ASP-ASP", 1.0},
      {"GLY-GLN-GLY", 1.0}, {"GLY-GLY-GLY", 1.0}, {"GLY-TRP-GLY", 1.0},
      {"HIS-THR-PHE", 1.0}, {"MET-SER-GLY", 1.0}, {"PRO-SER-SER", 1.0},
      {"SER-GLY-MET", 1.0}, {"SER-LYS-SER", 1.0}, {"SER-SER-LYS", 1.0},
      {"THR-ALA-ALA", 1.0}, {"THR-PHE-PRO", 1.0}, {"TRP-GLY-ASP", 1.0},
      {"TYR-GLY-TRP", 1.0}};
  processed_frames = 0;
}

void MLFeatureExtractor::process_frame(const MOL &rec, const MOL &lig,
                                       const std::vector<mol_info> &rec_info,
                                       const std::vector<mol_info> &lig_info,
                                       const Multi_MOL_parameters &lig_params,
                                       const std::vector<d_min> &dis_min,
                                       double frame_amber_energy,
                                       double distance_cutoff) {

  accumulated_features.amber_energy += frame_amber_energy;

  // 1. Identify Contact Residues & Average Distance
  std::unordered_set<int> contact_lig_res_ids;
  double dist_sum = 0.0;
  int dist_count = 0;

  for (const auto &d_thread : dis_min) {
    for (const auto &contact : d_thread.base) {
      if (contact.d <= distance_cutoff) { // Target: 6.0 A
        contact_lig_res_ids.insert(contact.resnr_ab);
        dist_sum += contact.d;
        dist_count++;
      }
    }
  }

  if (dist_count > 0) {
    accumulated_features.avg_contact_distance += (dist_sum / dist_count);
  }

  // 2. Pre-compute full Ligand Residue map (Name & Total Net Charge)
  std::unordered_map<int, std::string> res_names;
  std::unordered_map<int, double> res_charges;

  for (size_t a = 0; a < lig.size(); ++a) {
    int r_id = lig_info[a].get_resnr();
    if (res_names.find(r_id) == res_names.end()) {
      res_names[r_id] = lig_info[a].get_res_name();
      res_charges[r_id] = 0.0;
    }
    res_charges[r_id] +=
        lig_params.charge[a]; // Summing actual FF partial charges
  }

  // 3. Process Contact Properties
  std::vector<int> sorted_contacts(contact_lig_res_ids.begin(),
                                   contact_lig_res_ids.end());
  std::sort(sorted_contacts.begin(), sorted_contacts.end());

  double frame_hydro = 0.0;
  double frame_charge = 0.0;
  double frame_motif_score = 0.0;

  for (int res_id : sorted_contacts) {
    if (res_names.count(res_id)) {
      std::string r_name = res_names[res_id];
      if (hydrophobicity_scale.count(r_name)) {
        frame_hydro += hydrophobicity_scale[r_name];
      }
      frame_charge +=
          res_charges[res_id]; // Adds the exact Net Charge of the residue
    }
  }

  // 4. Extract 3-mer Binding Motifs
  for (size_t i = 0; i < sorted_contacts.size(); ++i) {
    if (i + 2 < sorted_contacts.size()) {
      int r1 = sorted_contacts[i];
      int r2 = sorted_contacts[i + 1];
      int r3 = sorted_contacts[i + 2];

      // Strict sequence continuity
      if ((r2 == r1 + 1) && (r3 == r2 + 1)) {
        if (res_names.count(r1) && res_names.count(r2) && res_names.count(r3)) {
          std::string motif =
              res_names[r1] + "-" + res_names[r2] + "-" + res_names[r3];
          if (motif_dictionary.count(motif)) {
            frame_motif_score += motif_dictionary[motif];
          }
        }
      }
    }
  }

  accumulated_features.hydrophobicity_ab_contact += frame_hydro;
  accumulated_features.charge_ab_contact += frame_charge;
  accumulated_features.motif_score += frame_motif_score;

  processed_frames++;
}

void MLFeatureExtractor::export_features(const std::string &filename) {
  if (processed_frames == 0)
    return;

  std::ofstream out(filename);
  out << "Amber_Energy,Avg_Contact_Distance,Motif_Score,charge_ab_contact,"
         "hydrophobicity_ab_contact\n";

  out << std::fixed << std::setprecision(4)
      << (accumulated_features.amber_energy / processed_frames) << ","
      << (accumulated_features.avg_contact_distance / processed_frames) << ","
      << (accumulated_features.motif_score / processed_frames) << ","
      << (accumulated_features.charge_ab_contact / processed_frames) << ","
      << (accumulated_features.hydrophobicity_ab_contact / processed_frames)
      << "\n";

  out.close();
}

double MLFeatureExtractor::predict_affinity() const {
  if (processed_frames == 0)
    return 0.0;

  double rf_input[5];
  rf_input[0] = accumulated_features.amber_energy / processed_frames;
  rf_input[1] = accumulated_features.avg_contact_distance / processed_frames;
  rf_input[2] = accumulated_features.motif_score / processed_frames;
  rf_input[3] = accumulated_features.charge_ab_contact / processed_frames;
  rf_input[4] =
      accumulated_features.hydrophobicity_ab_contact / processed_frames;

  return predict_rf_pKD(rf_input);
}

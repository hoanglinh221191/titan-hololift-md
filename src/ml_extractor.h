#ifndef ML_EXTRACTOR_H
#define ML_EXTRACTOR_H

#include "MOL.h"
#include "MOL_info.h"
#include "MOL_para.h"
#include "distance_v2.h"
#include <string>
#include <unordered_map>
#include <vector>

struct MLFeatures {
  double amber_energy = 0.0;
  double avg_contact_distance = 0.0;
  double motif_score = 0.0;
  double charge_ab_contact = 0.0;
  double hydrophobicity_ab_contact = 0.0;
};

class MLFeatureExtractor {
public:
  MLFeatureExtractor();

  void process_frame(const MOL &rec, const MOL &lig,
                     const std::vector<mol_info> &rec_info,
                     const std::vector<mol_info> &lig_info,
                     const Multi_MOL_parameters &lig_params,
                     const std::vector<d_min> &dis_min,
                     double frame_amber_energy, double distance_cutoff);

  void export_features(const std::string &filename);
  double predict_affinity() const;

private:
  std::unordered_map<std::string, double> hydrophobicity_scale;
  std::unordered_map<std::string, double> motif_dictionary;
  MLFeatures accumulated_features;
  int processed_frames = 0;
};

#endif

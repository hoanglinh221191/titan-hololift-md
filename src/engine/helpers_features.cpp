#include "engine/engine_common.h"

namespace engine_detail {

double determinant3x3(const BoxMatrix &box_matrix) {
  return box_matrix.m[0][0] *
             (box_matrix.m[1][1] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][1]) -
         box_matrix.m[0][1] *
             (box_matrix.m[1][0] * box_matrix.m[2][2] -
              box_matrix.m[1][2] * box_matrix.m[2][0]) +
         box_matrix.m[0][2] *
             (box_matrix.m[1][0] * box_matrix.m[2][1] -
              box_matrix.m[1][1] * box_matrix.m[2][0]);
}

double frame_box_volume_nm3(const box &frame_box) {
  if (frame_box.get_a() <= 0.0 || frame_box.get_b() <= 0.0 ||
      frame_box.get_c() <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const BoxMatrix box_matrix(frame_box);
  const double volume_a3 = std::abs(determinant3x3(box_matrix));
  if (!(volume_a3 > 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return volume_a3 / 1000.0;
}

bool is_generic_autotype_candidate(const std::string &mol_type) {
  return mol_type == "lig" || mol_type == "ligand" ||
         mol_type == "protein" || mol_type == "protein-user" ||
         mol_type == "complex";
}

void append_module_label(std::string &modules, std::string_view label) {
  if (!modules.empty())
    modules += ' ';
  modules += label;
}

bool feature_source_is_observable_tica(const arguments &Arguments) {
  const std::string &source = Arguments.feature_export.source;
  return Arguments.feature_export.source_explicit && source != "dccm" &&
         source != "lj";
}

bool feature_source_needs_dccm(const arguments &Arguments,
                               bool legacy_tica_mode) {
  if (legacy_tica_mode)
    return true;
  const std::string &source = Arguments.feature_export.source;
  return source == "auto" || source == "dccm" || source == "mixed";
}

void add_dccm_feature_columns(std::vector<FeatureColumn> &columns, int n_ca) {
  for (int a = 0; a < n_ca; ++a) {
    for (int b = a + 1; b < n_ca; ++b) {
      columns.push_back(FeatureColumn{
          "dccm", "ca_distance_" + std::to_string(a) + "_" + std::to_string(b),
          "A", false, "", "mol1_mol2_trace"});
    }
  }
}

bool append_dccm_pairwise_features(std::vector<double> &values,
                                   DCCMNetwork &dccm_analyzer, int n_ca,
                                   Eigen::MatrixXd &mobile,
                                   Eigen::MatrixXd &ref_centered,
                                   Eigen::Matrix3d &rotation,
                                   bool &ref_initialized) {
  if (n_ca <= 0 || dccm_analyzer.get_total_frames() == 0)
    return false;
  const auto &tx = dccm_analyzer.get_latest_traj_x();
  const auto &ty = dccm_analyzer.get_latest_traj_y();
  const auto &tz = dccm_analyzer.get_latest_traj_z();
  if (tx.size() < static_cast<std::size_t>(n_ca) ||
      ty.size() < static_cast<std::size_t>(n_ca) ||
      tz.size() < static_cast<std::size_t>(n_ca))
    return false;
  for (int a = 0; a < n_ca; ++a) {
    mobile(a, 0) = tx[static_cast<std::size_t>(a)];
    mobile(a, 1) = ty[static_cast<std::size_t>(a)];
    mobile(a, 2) = tz[static_cast<std::size_t>(a)];
  }
  if (!ref_initialized) {
    ref_centered = mobile;
    Eigen::RowVectorXd col_mean = ref_centered.colwise().mean();
    ref_centered.rowwise() -= col_mean;
    mobile = ref_centered;
    ref_initialized = true;
  } else {
    TitanAlignment::align_frame_inplace(ref_centered, mobile, rotation);
  }
  for (int a = 0; a < n_ca; ++a) {
    for (int b = a + 1; b < n_ca; ++b) {
      const double dx = mobile(a, 0) - mobile(b, 0);
      const double dy = mobile(a, 1) - mobile(b, 1);
      const double dz = mobile(a, 2) - mobile(b, 2);
      values.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  }
  return true;
}

std::vector<FeatureColumn> build_observable_feature_columns(
    const arguments &Arguments, int n_ca, const TitanQNativeAnalyzer &qnative,
    const TitanDihedralAnalyzer &dihedral, const TitanContactMapAnalyzer &cmap,
    const TitanLJAnalyzer &lj_analyzer, bool tica_use_lj_features) {
  const std::string &source = Arguments.feature_export.source;
  std::vector<FeatureColumn> columns;
  const bool include_dccm =
      source == "dccm" || source == "mixed" ||
      (source == "auto" && Arguments.msm.flag_tica && !tica_use_lj_features);
  const bool include_lj =
      source == "lj" || (source == "auto" && tica_use_lj_features) ||
      (source == "auto" && !Arguments.msm.flag_tica && lj_analyzer.enabled());
  const bool include_qnative =
      source == "qnative" || source == "mixed" ||
      (source == "auto" && qnative.enabled());
  const bool include_dihedral =
      source == "dihedral" || source == "mixed" ||
      (source == "auto" && dihedral.enabled());
  const bool include_cmap =
      source == "cmap" || (source == "mixed" && cmap.enabled()) ||
      (source == "auto" && cmap.enabled());

  if (include_dccm)
    add_dccm_feature_columns(columns, n_ca);
  if (include_lj) {
    for (int i = 0; i < lj_analyzer.feature_dim(); ++i) {
      columns.push_back(
          FeatureColumn{"lj", "lj_phase2_" + std::to_string(i),
                        "dimensionless", false, "", "mol1"});
    }
  }
  if (include_qnative) {
    columns.push_back(
        FeatureColumn{"qnative", "Q", "fraction", false, "", "mol1"});
  }
  if (include_dihedral) {
    for (const auto &[name, residue] : dihedral.tica_feature_columns()) {
      columns.push_back(FeatureColumn{"dihedral", name, "dimensionless",
                                      false, residue, "mol1"});
    }
  }
  if (include_cmap) {
    columns.push_back(
        FeatureColumn{"cmap", "contact_count", "count", false, "", "mol1"});
    if (source == "cmap") {
      columns.push_back(FeatureColumn{"cmap", "contact_count_normalized",
                                      "fraction", false, "", "mol1"});
    }
  }
  return columns;
}

bool build_observable_feature_vector(
    const arguments &Arguments, DCCMNetwork &dccm_analyzer, int n_ca,
    Eigen::MatrixXd &mobile, Eigen::MatrixXd &ref_centered,
    Eigen::Matrix3d &rotation, bool &ref_initialized,
    const TitanQNativeAnalyzer &qnative, const TitanDihedralAnalyzer &dihedral,
    const TitanContactMapAnalyzer &cmap, const TitanLJAnalyzer &lj_analyzer,
    Eigen::VectorXd &out_values) {
  const std::string &source = Arguments.feature_export.source;
  std::vector<double> values;
  const bool include_dccm =
      source == "dccm" || source == "mixed" ||
      (source == "auto" && Arguments.msm.flag_tica && n_ca > 0);
  const bool include_lj =
      source == "lj" ||
      (source == "auto" && n_ca == 0 && lj_analyzer.has_feature_vector());
  const bool include_qnative =
      source == "qnative" || source == "mixed" ||
      (source == "auto" && qnative.enabled());
  const bool include_dihedral =
      source == "dihedral" || source == "mixed" ||
      (source == "auto" && dihedral.enabled());
  const bool include_cmap = source == "cmap" ||
                            (source == "mixed" && cmap.enabled()) ||
                            (source == "auto" && cmap.enabled());

  if (include_dccm &&
      !append_dccm_pairwise_features(values, dccm_analyzer, n_ca, mobile,
                                     ref_centered, rotation, ref_initialized))
    return false;
  if (include_lj) {
    if (!lj_analyzer.has_feature_vector())
      return false;
    const Eigen::VectorXd &lj = lj_analyzer.latest_feature_vector();
    for (Eigen::Index i = 0; i < lj.size(); ++i)
      values.push_back(lj(i));
  }
  if (include_qnative) {
    if (!qnative.latest_q_valid())
      return false;
    values.push_back(qnative.latest_q());
  }
  if (include_dihedral) {
    if (!dihedral.append_latest_tica_features(values))
      return false;
  }
  if (include_cmap) {
    if (!cmap.latest_contact_count_valid())
      return false;
    values.push_back(cmap.latest_contact_count());
    if (source == "cmap")
      values.push_back(cmap.latest_contact_count_normalized());
  }
  out_values.resize(static_cast<Eigen::Index>(values.size()));
  for (Eigen::Index i = 0; i < out_values.size(); ++i)
    out_values(i) = values[static_cast<std::size_t>(i)];
  return true;
}

std::string format_binary_bytes(long double bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  constexpr long double kib = 1024.0L;
  constexpr long double mib = kib * 1024.0L;
  constexpr long double gib = mib * 1024.0L;
  if (bytes >= gib) {
    oss << (bytes / gib) << " GiB";
  } else if (bytes >= mib) {
    oss << (bytes / mib) << " MiB";
  } else if (bytes >= kib) {
    oss << (bytes / kib) << " KiB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

bool should_enable_dense_pair_cache(const arguments &Arguments,
                                    size_t mol1_atoms, size_t mol2_atoms,
                                    size_t available_ram,
                                    std::string &reason) {
  if (Arguments.force_f == "MARTINI") {
    reason = "disabled for MARTINI";
    return false;
  }

  const bool sg_cache_requested =
      Arguments.flag_sub_groups &&
      (Arguments.sg_mode == "cache" || Arguments.sg_cache == "on");
  if (!(Arguments.flag_cache == 1 || Arguments.force_map == 1 ||
        Arguments.force_map == 3 ||
        (Arguments.edecomp.flag_edecomp &&
         (Arguments.edecomp.source == "mm" ||
          Arguments.edecomp.source == "all")) ||
        sg_cache_requested)) {
    reason = "not requested by active modules";
    return false;
  }

  const long double pair_count =
      static_cast<long double>(mol1_atoms) * static_cast<long double>(mol2_atoms);
  const long double bytes_per_pair =
      static_cast<long double>(sizeof(double)) *
      static_cast<long double>(Arguments.flag_cal_force ? 8 : 2);
  const long double estimated_bytes = pair_count * bytes_per_pair;

  if (Arguments.flag_gbsa && Arguments.flag_xdr) {
    reason = "disabled for XDR + GBSA to avoid multi-GB pair cache (" +
             format_binary_bytes(estimated_bytes) + ")";
    return false;
  }

  constexpr long double hard_cap_bytes = 1.5L * 1024.0L * 1024.0L * 1024.0L;
  if (estimated_bytes > hard_cap_bytes) {
    reason = "estimated cache " + format_binary_bytes(estimated_bytes) +
             " exceeds hard cap";
    return false;
  }

  if (available_ram > 0) {
    const long double ram_limit =
        static_cast<long double>(available_ram) * 0.20L;
    if (estimated_bytes > ram_limit) {
      reason = "estimated cache " + format_binary_bytes(estimated_bytes) +
               " exceeds 20% of available RAM";
      return false;
    }
  }

  reason = "enabled (" + format_binary_bytes(estimated_bytes) + ")";
  return true;
}

} // namespace engine_detail

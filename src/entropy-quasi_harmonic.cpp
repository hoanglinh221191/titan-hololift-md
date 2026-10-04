#include "MOL.h"
#include "MOL_info.h"
#include "Multi_MOL.h"
#include "folder.h"
#include "functions.h"
#include "groups.h"
#include "info.h"
#include "titan_qha_model.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace std;

// Analytical Gaussian Entropy formulation based on Quantum Harmonic Oscillator
double gaussian_entropy(const std::vector<double> &valid_evals, double temperature) {
  return titan_qha_ts_kj_mol(valid_evals, temperature);
}

// Kabsch algorithm to find optimal rotation matrix aligning Q to P.
inline Eigen::Matrix3d get_kabsch_rotation(const Eigen::MatrixXd &P,
                                           const Eigen::MatrixXd &Q) {
  Eigen::Matrix3d H = P * Q.transpose();
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  double d = (U * V.transpose()).determinant() > 0 ? 1.0 : -1.0;
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  D(2, 2) = d;
  return U * D * V.transpose();
}

Eigen::MatrixXd extract_frame(size_t frame_index,
                              const std::vector<size_t> &filtered_indices,
                              const Multi_MOL &mol1, const Multi_MOL &mol2) {
  size_t N = filtered_indices.size();
  Eigen::MatrixXd mat(3, N);
  size_t mol1size = mol1.m_mol[0].size();
  for (size_t i = 0; i < N; ++i) {
    size_t atom_index = filtered_indices[i];
    if (atom_index < mol1size) {
      mat(0, i) = mol1.m_mol[frame_index].pos.x[atom_index];
      mat(1, i) = mol1.m_mol[frame_index].pos.y[atom_index];
      mat(2, i) = mol1.m_mol[frame_index].pos.z[atom_index];
    } else {
      mat(0, i) = mol2.m_mol[frame_index].pos.x[atom_index - mol1size];
      mat(1, i) = mol2.m_mol[frame_index].pos.y[atom_index - mol1size];
      mat(2, i) = mol2.m_mol[frame_index].pos.z[atom_index - mol1size];
    }
  }
  return mat;
}

double imp_cal_entropy(groups_cal &groups, const Multi_MOL &mol1,
                       const Multi_MOL &mol2, arguments &Arguments,
                       size_t group_index, const folders &paths,
                       size_t file_pos, const info &info_output) {
  std::vector<size_t> filtered_indices;
  std::vector<double> atom_masses;
  size_t num_atoms_in_group = groups.groups_atom_number(group_index);

  for (size_t i = 0; i < num_atoms_in_group; ++i) {
    size_t atom_index = groups.atom_index(group_index, i) - 1;
    std::string atom_name;
    double mass = 0.0;
    bool is_ligand = false;

    if (atom_index < mol1.m_mol[0].size()) {
      atom_name = mol1.info[atom_index].get_atom();
      mass = mol1.parameters[atom_index].get_mass();
      is_ligand =
          (mol1.get_mol_type() == "ligand" || mol1.get_mol_type() == "lig");
    } else {
      size_t mol1size = mol1.m_mol[0].size();
      atom_name = mol2.info[atom_index - mol1size].get_atom();
      mass = mol2.parameters[atom_index - mol1size].get_mass();
      is_ligand =
          (mol2.get_mol_type() == "ligand" || mol2.get_mol_type() == "lig");
    }

    if (is_ligand) {
      // For ligand, extract heavy atoms (not starting with H/h)
      if (!atom_name.empty() && atom_name[0] != 'H' && atom_name[0] != 'h') {
        filtered_indices.emplace_back(atom_index);
        atom_masses.emplace_back(mass);
      }
    } else {
      // For protein/nucleic acid, extract backbone representatives
      if (atom_name == "CA" || atom_name == "BB" || atom_name == "C4'" ||
          atom_name == "C4*" || atom_name == "BB2") {
        filtered_indices.emplace_back(atom_index);
        atom_masses.emplace_back(mass);
      }
    }
  }

  size_t N_filtered = filtered_indices.size();
  if (N_filtered == 0) {
    cout << "No backbone atoms found for QHA in "
         << groups.group_name(group_index) << "\n";
    return 0.0;
  }

  size_t M_frames = mol1.m_mol.size();
  if (M_frames < 2)
    return 0.0;

  Eigen::MatrixXd P_raw = extract_frame(0, filtered_indices, mol1, mol2);
  Eigen::Vector3d P_centroid = P_raw.rowwise().mean();
  Eigen::MatrixXd P = P_raw.colwise() - P_centroid;

  Eigen::MatrixXd X(3 * N_filtered, M_frames);

  size_t nt = std::min(std::max<size_t>(1, Arguments.nt), M_frames);
  std::vector<std::thread> threads(nt);
  size_t chunk_size = M_frames / nt;

  for (size_t t = 0; t < nt; ++t) {
    size_t start_frame = t * chunk_size;
    size_t end_frame = (t == nt - 1) ? M_frames : (t + 1) * chunk_size;

    threads[t] = std::thread([&, start_frame, end_frame]() {
      for (size_t j = start_frame; j < end_frame; ++j) {
        Eigen::MatrixXd Q_raw = extract_frame(j, filtered_indices, mol1, mol2);
        Eigen::Vector3d Q_centroid = Q_raw.rowwise().mean();
        Eigen::MatrixXd Q = Q_raw.colwise() - Q_centroid;

        Eigen::Matrix3d R = get_kabsch_rotation(P, Q);
        Eigen::MatrixXd Q_aligned = R * Q;

        for (size_t i = 0; i < N_filtered; ++i) {
          double sqrt_m = std::sqrt(atom_masses[i]);
          X(i * 3 + 0, j) = sqrt_m * (Q_aligned(0, i) - P(0, i));
          X(i * 3 + 1, j) = sqrt_m * (Q_aligned(1, i) - P(1, i));
          X(i * 3 + 2, j) = sqrt_m * (Q_aligned(2, i) - P(2, i));
        }
      }
    });
  }

  for (auto &t : threads) {
    if (t.joinable())
      t.join();
  }

  // Mean center X across columns just to be completely rigorous
  Eigen::VectorXd X_mean = X.rowwise().mean();
  X.colwise() -= X_mean;

  Eigen::MatrixXd cov = (X * X.transpose()) / static_cast<double>(M_frames - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("QHA covariance eigensolver did not converge");
  Eigen::VectorXd evals = solver.eigenvalues();

  std::string base_dir =
      (paths.get_size() > file_pos) ? paths.get_path(file_pos) : ".";
  std::string eigen_filename =
      base_dir + "/qha_eigenvalues_" + groups.group_name(group_index) + ".dat";

  std::fstream out_eigen(eigen_filename, std::ios::out);
  out_eigen << "# Mode_Index \t Eigenvalue(amu*Angstrom^2)\n";
  out_eigen << "# Temperature_K " << Arguments.qha_temperature << '\n';

  int total_modes = evals.size();
  std::vector<double> valid_evals;

  int mode_index = 1;
  for (int i = total_modes - 1; i >= 6; --i) {
    if (evals(i) > 1e-10) {
      valid_evals.emplace_back(evals(i));
      out_eigen << mode_index << " \t " << std::fixed << std::setprecision(6)
                << evals(i) << "\n";
      mode_index++;
    }
  }
  out_eigen.close();

  return gaussian_entropy(valid_evals, Arguments.qha_temperature);
}

int cal_entropy(const Multi_MOL &mol1, const Multi_MOL &mol2,
                groups_cal &groups, arguments &Arguments, const folders &paths,
                size_t file_pos, const info &info_output) {
  if (Arguments.entropy_quasi == true && Arguments.group_entropy_com == -1)
    groups.create_group_entropy(mol1, mol2, Arguments, "calpha");

  double com_entropy = imp_cal_entropy(groups, mol1, mol2, Arguments,
                                       groups.group_entropy_com_index(), paths,
                                       file_pos, info_output);
  cout << "com entropy done, entropy= " << com_entropy
       << "\n----------------------\n";
  double rec_entropy = imp_cal_entropy(groups, mol1, mol2, Arguments,
                                       groups.group_entropy_rec_index(), paths,
                                       file_pos, info_output);
  cout << "rec entropy done, entropy= " << rec_entropy
       << "\n----------------------\n";
  double lg_entropy = imp_cal_entropy(groups, mol1, mol2, Arguments,
                                      groups.group_entropy_lg_index(), paths,
                                      file_pos, info_output);
  cout << "lg entropy done, entropy= " << lg_entropy
       << "\n-----------------------\n";

  string entropy_name{info_output.getEntropyQuasi()};
  if (paths.get_size() > file_pos)
    entropy_name = paths.get_path(file_pos) + "/" + entropy_name;

  fstream entropy_result(entropy_name, ios::out);
  entropy_result << "Entropy (TS) of group "
                 << groups.group_name(groups.group_entropy_com_index()) << ": "
                 << com_entropy << " kJ/mol\n"
                 << "Entropy (TS) of group "
                 << groups.group_name(groups.group_entropy_rec_index()) << ": "
                 << rec_entropy << " kJ/mol\n"
                 << "Entropy (TS) of group "
                 << groups.group_name(groups.group_entropy_lg_index()) << ": "
                 << lg_entropy << " kJ/mol\n"
                 << "----------------------------------------------------------"
                    "---------\n";

  entropy_result << "Delta Entropy (T*(S_com-S_rec-S_lg)): "
                 << com_entropy - rec_entropy - lg_entropy
                 << " kJ/mol = "
                 << (com_entropy - rec_entropy - lg_entropy) /
                        4.184
                 << " kcal/mol\n";
  entropy_result.close();
  return 0;
}

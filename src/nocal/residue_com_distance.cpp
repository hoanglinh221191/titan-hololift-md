#include "nocal/residue_com_distance.h"

#include "Multi_MOL.h"
#include "miscellaneous.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace titan_nocal {

void ResidueComDistance::begin_trajectory(const Trajectory &trajectory,
                                          const std::string &suffix) {
  trajectory_ = trajectory;
  suffix_ = suffix;
  names_.clear();
  mean_.clear();
  m2_.clear();
  frames_ = 0;
}

void ResidueComDistance::process_frame(Multi_MOL &mol, std::size_t snap) {
  if (mol.Index.size() < 2 || snap >= mol.m_mol.size())
    return;
  // Residue centres chain by chain; the molecule's centre is their
  // mass-weighted mean.
  std::vector<multi_points> residues(mol.Index.size() - 1);
  for (std::size_t c = 0; c + 1 < mol.Index.size(); ++c)
    mol.m_mol[snap].take_com_all_atoms(mol.parameters, c,
                                       mol.Index[c + 1].get_res_pos_e(0),
                                       mol.Index, mol.info, residues[c]);
  double mass = 0.0;
  point centre{0.0, 0.0, 0.0};
  for (const multi_points &chain : residues)
    for (std::size_t j = 0; j < chain.points.size(); ++j) {
      const double m = chain.mass[j];
      mass += m;
      centre.x += chain.points[j].x * m;
      centre.y += chain.points[j].y * m;
      centre.z += chain.points[j].z * m;
    }
  centre.x /= mass;
  centre.y /= mass;
  centre.z /= mass;

  std::vector<double> distances;
  std::vector<std::string> names;
  for (std::size_t c = 0; c < residues.size(); ++c)
    for (std::size_t j = 0; j < residues[c].points.size(); ++j) {
      const point &p = residues[c].points[j];
      distances.push_back(std::sqrt((p.x - centre.x) * (p.x - centre.x) +
                                    (p.y - centre.y) * (p.y - centre.y) +
                                    (p.z - centre.z) * (p.z - centre.z)));
      std::string chain = mol.Index[c].get_chain();
      chain.erase(0, chain.find_first_not_of(' '));
      names.push_back(chain + residues[c].name[j]);
    }

  if (frames_ == 0) {
    names_ = std::move(names);
    mean_.assign(distances.size(), 0.0);
    m2_.assign(distances.size(), 0.0);
  } else if (distances.size() != mean_.size()) {
    throw std::runtime_error("-resdis2com: the residue count changed from " +
                             std::to_string(mean_.size()) + " to " +
                             std::to_string(distances.size()));
  }
  ++frames_;
  for (std::size_t i = 0; i < distances.size(); ++i) {
    const double delta = distances[i] - mean_[i];
    mean_[i] += delta / frames_;
    m2_[i] += delta * (distances[i] - mean_[i]);
  }
}

void ResidueComDistance::end_trajectory() {
  if (suffix_.empty())
    return;
  const std::string path = trajectory_.file(suffix_);
  std::ofstream out(path, std::ios::out);
  if (!out)
    throw std::runtime_error("cannot write " + path);
  out << "# Residue    Mean (A)      SD (A), over " << frames_ << " frames\n";
  for (std::size_t i = 0; i < names_.size(); ++i)
    out << std::setw(10) << std::left << names_[i] << std::right
        << std::setw(12) << mean_[i] << std::setw(12)
        << std::sqrt(m2_[i] / frames_) << '\n';
  suffix_.clear();
}

} // namespace titan_nocal

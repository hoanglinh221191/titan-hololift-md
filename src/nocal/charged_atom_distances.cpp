#include "nocal/charged_atom_distances.h"

#include "Multi_MOL.h"
#include "arguments.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace titan_nocal {

namespace {

// Percent of the distances per bin. The bins run from the
// -charged_distances_min value to the -charged_distances_max value unless a
// distance reaches that maximum; then they span the data.
void write_distribution(const std::vector<double> &data,
                        const std::string &path, const arguments &args) {
  std::ofstream out(path, std::ios::out);
  if (!out)
    throw std::runtime_error("cannot write " + path);
  const std::size_t bins = std::max<std::size_t>(1, args.number_charged_distances_bin);
  double min = 1000000., max = -10000.;
  for (const double d : data) {
    min = std::min(min, d);
    max = std::max(max, d);
  }
  double delta = (max - min) / bins;
  if (max < args.charged_distances_max) {
    delta = (args.charged_distances_max - args.charged_distances_min) / bins;
    min = args.charged_distances_min;
  }
  std::vector<std::size_t> histogram(bins);
  for (const double d : data) {
    const double position = (d - min) / delta;
    std::size_t bin = position > 0.0 ? static_cast<std::size_t>(position) : 0;
    histogram[std::min(bin, bins - 1)] += 1;
  }

  out.precision(7);
  out << "# Distance (A)                     Frequency\n";
  for (std::size_t i = 0; i < bins; ++i)
    out << std::setw(8) << std::left << i * delta + min << std::setw(16)
        << std::right
        << (data.empty() ? 0.0 : histogram[i] * 100. / data.size()) << '\n';
}

} // namespace

void ChargedAtomDistances::begin_trajectory(const Trajectory &trajectory,
                                            const std::string &suffix) {
  trajectory_ = trajectory;
  suffix_ = suffix;
  all_.clear();
  positive_.clear();
  negative_.clear();
}

void ChargedAtomDistances::process_frame(const Multi_MOL &mol,
                                         std::size_t snap) {
  double com[3]{};
  if (snap >= mol.m_mol.size() || mol.com_mol(snap, com) != 0)
    return;
  mol.m_mol[snap].cal_dis_charged_atoms(all_, negative_, positive_,
                                        mol.parameters, com);
}

void ChargedAtomDistances::end_trajectory(const arguments &args) {
  if (suffix_.empty())
    return;
  write_distribution(all_, trajectory_.file(suffix_), args);
  write_distribution(positive_, trajectory_.file("_positive" + suffix_), args);
  write_distribution(negative_, trajectory_.file("_negative" + suffix_), args);
  suffix_.clear();
}

} // namespace titan_nocal

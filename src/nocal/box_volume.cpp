#include "nocal/box_volume.h"

#include "Multi_MOL.h"
#include "arguments.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace titan_nocal {

void BoxVolume::begin_trajectory(const Trajectory &trajectory) {
  current_ = static_cast<std::size_t>(
      std::find_if(folders_.begin(), folders_.end(),
                   [&](const Folder &f) { return f.path == trajectory.folder; }) -
      folders_.begin());
  if (current_ == folders_.size())
    folders_.push_back(Folder{trajectory.folder, {}});
  folders_[current_].inputs.push_back(
      Input{trajectory.index, trajectory.stem, {}});
}

void BoxVolume::process_frame(const Multi_MOL &mol, std::size_t snap,
                              double time) {
  if (current_ >= folders_.size() || snap >= mol.m_mol.size())
    return;
  folders_[current_].inputs.back().rows.emplace_back(
      time, mol.m_mol[snap].Box.volume());
}

void BoxVolume::finish_run(const arguments &args) const {
  if (folders_.empty())
    return;
  std::size_t width = 0;
  for (const Folder &f : folders_)
    width = std::max(width, f.path.size());
  std::ofstream stats("stat_volumes.dat", std::ios::out);
  if (!stats)
    throw std::runtime_error("cannot write stat_volumes.dat");
  stats << std::setw(static_cast<int>(width) + 2) << std::right
        << "# folder name" << "   volume (A^3)   "
        << "SD (A^3)         CI (95%) (A^3)\n";

  for (const Folder &folder : folders_) {
    const std::string path = folder.path + "/volume.dat";
    std::ofstream out(path, std::ios::out);
    if (!out)
      throw std::runtime_error("cannot write " + path);
    out << "# Time (" << args.tu << ") " << std::setw(17)
        << "box volume (A^3)\n";
    std::size_t n = 0;
    double mean = 0.0, m2 = 0.0;
    for (const Input &input : folder.inputs) {
      if (folder.inputs.size() > 1)
        out << "# input " << input.index + 1 << ": " << input.stem << '\n';
      for (const auto &[time, volume] : input.rows) {
        out << std::setw(6) << std::left << time << std::right
            << std::setw(17) << volume << '\n';
        ++n;
        const double delta = volume - mean;
        mean += delta / n;
        m2 += delta * (volume - mean);
      }
    }
    const double sd = n > 0 ? std::sqrt(m2 / n) : 0.0;
    stats << std::left << std::setw(25) << folder.path << "  " << std::right
          << std::setw(17) << mean << std::setw(17) << sd << std::setw(17)
          << (n > 0 ? 1.96 * sd / std::sqrt(1. * n) : 0.0) << '\n';
  }
}

} // namespace titan_nocal

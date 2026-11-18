#include "cscd.h"
#include "titan_executor.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();

std::string trimmed(const std::string &text) {
  const auto first = text.find_first_not_of(' ');
  if (first == std::string::npos)
    return {};
  return text.substr(first, text.find_last_not_of(' ') - first + 1);
}

std::string residue_label(const com &centre) {
  return trimmed(centre.get_chain()) + std::to_string(centre.get_res_num());
}

// Fixed-point with `decimals` digits, right-aligned in `width` characters:
// the same digits as iostream's fixed/setprecision, without its per-value cost.
void append_fixed(std::string &line, double value, int width, int decimals) {
  char buffer[352]; // DBL_MAX in fixed notation has 309 integer digits

  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                    std::chars_format::fixed, decimals);
  const std::size_t length =
      result.ec == std::errc{} ? static_cast<std::size_t>(result.ptr - buffer)
                               : 0;
  if (length < static_cast<std::size_t>(width))
    line.append(static_cast<std::size_t>(width) - length, ' ');
  line.append(buffer, length);
}

void append_right(std::string &line, const std::string &text, int width) {
  if (text.size() < static_cast<std::size_t>(width))
    line.append(static_cast<std::size_t>(width) - text.size(), ' ');
  line += text;
}

double min_squared(double x, double y, double z, const double *x2,
                   const double *y2, const double *z2, std::size_t begin,
                   std::size_t end) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t j = begin; j < end; ++j) {
    const double dx = x - x2[j];
    const double dy = y - y2[j];
    const double dz = z - z2[j];
    const double d2 = dx * dx + dy * dy + dz * dz;
    best = d2 < best ? d2 : best;
  }
  return best;
}

void all_distances(double x, double y, double z, const double *x2,
                   const double *y2, const double *z2, std::size_t begin,
                   std::size_t end, double *out) {
  for (std::size_t j = begin; j < end; ++j) {
    const double dx = x - x2[j];
    const double dy = y - y2[j];
    const double dz = z - z2[j];
    out[j - begin] = std::sqrt(dx * dx + dy * dy + dz * dz);
  }
}

} // namespace

void all_cscd::configure(const Options &options,
                         std::vector<std::size_t> mol1_internal,
                         std::vector<std::size_t> mol2_internal) {
  if (layout_ready_)
    throw std::logic_error("cscd: configure() after the first frame");
  options_ = options;
  mol1_internal_ = std::move(mol1_internal);
  mol2_internal_ = std::move(mol2_internal);
  configured_ = true;
}

namespace {

// chain+number, unique within the molecule (labels name files and -act
// columns). A chain+number that repeats (blank chain IDs, insertion codes, a
// glycan numbered like the protein) adds the internal residue number, and the
// centre's first atom where even that repeats (per-atom ligand centres).
std::vector<std::string> unique_labels(const std::vector<com> &centres,
                                       const std::vector<std::size_t> &internal) {
  std::vector<std::string> labels(centres.size());
  for (std::size_t i = 0; i < centres.size(); ++i)
    labels[i] = residue_label(centres[i]);
  const auto add_where_repeated = [&](const auto &suffix) {
    std::unordered_map<std::string, std::size_t> count;
    for (const std::string &label : labels)
      ++count[label];
    for (std::size_t i = 0; i < labels.size(); ++i)
      if (count[labels[i]] > 1)
        labels[i] += suffix(i);
  };
  if (internal.size() == centres.size())
    add_where_repeated([&](std::size_t i) {
      return "_r" + std::to_string(internal[i] + 1);
    });
  add_where_repeated([&](std::size_t i) {
    return "_a" + std::to_string(centres[i].get_respos());
  });
  return labels;
}

std::vector<std::size_t> first_atoms(const std::vector<com> &centres,
                                     const char *molecule) {
  std::vector<std::size_t> atoms(centres.size());
  for (std::size_t i = 0; i < centres.size(); ++i)
    atoms[i] = centres[i].get_respos();
  std::vector<std::size_t> sorted = atoms;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
    throw std::runtime_error(std::string(molecule) +
                             " has two centres starting at the same atom");
  return atoms;
}

} // namespace

void all_cscd::build_layout(const std::vector<com> &mol1,
                            const std::vector<com> &mol2) {
  if (mol1.empty())
    throw std::runtime_error("molecule 1 has no residue centres");
  if (mol2.empty())
    throw std::runtime_error("molecule 2 has no residue centres");

  const std::size_t n1 = mol1.size();
  const std::size_t n2 = mol2.size();
  mol1_atom_ = first_atoms(mol1, "molecule 1");
  mol2_atom_ = first_atoms(mol2, "molecule 2");
  labels_ = unique_labels(mol1, mol1_internal_);
  partner_labels_ = unique_labels(mol2, mol2_internal_);

  // -g=x-x: both molecules are the same group, so a residue's partner copy
  // starts at the same atom.
  std::unordered_map<std::size_t, std::size_t> mol2_index;
  if (options_.skip_same_residue)
    for (std::size_t j = 0; j < n2; ++j)
      mol2_index.emplace(mol2_atom_[j], j);

  self_partner_.assign(n1, kNone);
  offset_.resize(n1);
  total_columns_ = 0;
  for (std::size_t r = 0; r < n1; ++r) {
    if (options_.skip_same_residue) {
      const auto found = mol2_index.find(mol1_atom_[r]);
      if (found != mol2_index.end())
        self_partner_[r] = found->second;
    }
    offset_[r] = total_columns_;
    total_columns_ += options_.nearest_only
                          ? 1
                          : n2 - (self_partner_[r] == kNone ? 0 : 1);
  }
  layout_ready_ = true;
}

void all_cscd::check_layout(const std::vector<com> &mol1,
                            const std::vector<com> &mol2) const {
  if (mol1.size() != mol1_atom_.size() || mol2.size() != mol2_atom_.size())
    throw std::runtime_error(
        "the residue lists changed between frames (" +
        std::to_string(mol1_atom_.size()) + "/" +
        std::to_string(mol2_atom_.size()) + " residues, now " +
        std::to_string(mol1.size()) + "/" + std::to_string(mol2.size()) + ")");
  for (std::size_t r = 0; r < mol1.size(); ++r)
    if (mol1[r].get_respos() != mol1_atom_[r])
      throw std::runtime_error("molecule-1 residue " + labels_[r] +
                               " changed between frames");
  for (std::size_t j = 0; j < mol2.size(); ++j)
    if (mol2[j].get_respos() != mol2_atom_[j])
      throw std::runtime_error("molecule-2 residue " + partner_labels_[j] +
                               " changed between frames");
}

void all_cscd::add_frame(std::size_t frame, double time,
                         const std::vector<com> &mol1,
                         const std::vector<com> &mol2, TitanExecutor *executor,
                         std::size_t threads) {
  if (!configured_)
    throw std::logic_error("cscd: add_frame() before configure()");
  if (layout_ready_)
    check_layout(mol1, mol2);
  else
    build_layout(mol1, mol2);
  if (!current_.frame.empty() && frame <= current_.frame.back())
    throw std::runtime_error("frame " + std::to_string(frame + 1) +
                             " arrived after frame " +
                             std::to_string(current_.frame.back() + 1));

  const std::size_t n1 = mol1.size();
  const std::size_t n2 = mol2.size();
  x2_.resize(n2);
  y2_.resize(n2);
  z2_.resize(n2);
  for (std::size_t j = 0; j < n2; ++j) {
    x2_[j] = mol2[j].get_x();
    y2_[j] = mol2[j].get_y();
    z2_[j] = mol2[j].get_z();
  }

  current_.frame.push_back(frame);
  current_.time.push_back(time);
  const std::size_t row_begin = current_.value.size();
  current_.value.resize(row_begin + total_columns_);
  double *const row = current_.value.data() + row_begin;
  const double *const x2 = x2_.data();
  const double *const y2 = y2_.data();
  const double *const z2 = z2_.data();

  const auto residues = [&](std::size_t, std::size_t begin, std::size_t end) {
    for (std::size_t r = begin; r < end; ++r) {
      const double x = mol1[r].get_x();
      const double y = mol1[r].get_y();
      const double z = mol1[r].get_z();
      // The skipped partner splits the scan into two contiguous ranges.
      const std::size_t skip = self_partner_[r];
      const std::size_t cut = skip == kNone ? n2 : skip;
      const std::size_t resume = skip == kNone ? n2 : skip + 1;
      double *const out = row + offset_[r];
      if (options_.nearest_only) {
        const double best =
            std::min(min_squared(x, y, z, x2, y2, z2, 0, cut),
                     min_squared(x, y, z, x2, y2, z2, resume, n2));
        out[0] = std::isinf(best) ? std::numeric_limits<double>::quiet_NaN()
                                  : std::sqrt(best);
      } else {
        all_distances(x, y, z, x2, y2, z2, 0, cut, out);
        all_distances(x, y, z, x2, y2, z2, resume, n2, out + cut);
      }
    }
  };

  // About 32k pair distances per worker before splitting pays off.
  const std::size_t grain = std::max<std::size_t>(1, 32768 / n2);
  if (executor != nullptr && threads > 1 && n1 > grain)
    executor->parallel_for(n1, threads, grain, residues);
  else
    residues(0, 0, n1);
}

void all_cscd::end_trajectory(const std::string &directory) {
  if (current_.time.empty())
    return;
  if (!directory.empty())
    write_files(directory, current_, nullptr);

  // Two frames are at the same time when they differ by less than about two
  // float32 ulps (XTC stores time as float); saved frames are never closer.
  const auto same_time = [](double a, double b) {
    return std::abs(a - b) <= std::max(1e-6, 2.5e-7 * std::abs(b));
  };
  const std::size_t rows = current_.time.size();
  const std::size_t old_rows = count_.size();
  std::vector<std::size_t> unmatched;
  for (std::size_t i = 0; i < rows; ++i) {
    const double t = current_.time[i];
    // sum_.time[0, old_rows) is sorted; rows added below are not searched,
    // so a trajectory never averages with itself.
    const auto first = sum_.time.begin();
    const auto last = first + static_cast<std::ptrdiff_t>(old_rows);
    auto it = std::lower_bound(first, last, t);
    std::size_t match = kNone;
    if (it != last && same_time(*it, t))
      match = static_cast<std::size_t>(it - first);
    else if (it != first && same_time(*(it - 1), t))
      match = static_cast<std::size_t>(it - first) - 1;
    if (match == kNone) {
      unmatched.push_back(i);
      continue;
    }
    const double *src = current_.value.data() + i * total_columns_;
    double *dst = sum_.value.data() + match * total_columns_;
    for (std::size_t c = 0; c < total_columns_; ++c)
      dst[c] += src[c];
    ++count_[match];
  }

  if (old_rows == 0) {
    sum_.time = std::move(current_.time);
    sum_.value = std::move(current_.value);
    count_.assign(rows, 1);
  } else {
    for (const std::size_t i : unmatched) {
      sum_.time.push_back(current_.time[i]);
      const double *src = current_.value.data() + i * total_columns_;
      sum_.value.insert(sum_.value.end(), src, src + total_columns_);
      count_.push_back(1);
    }
  }
  sort_merged_by_time();
  current_ = Rows{};
  ++trajectories_;
}

void all_cscd::sort_merged_by_time() {
  if (std::is_sorted(sum_.time.begin(), sum_.time.end()))
    return;
  std::vector<std::size_t> order(sum_.time.size());
  for (std::size_t k = 0; k < order.size(); ++k)
    order[k] = k;
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) {
                     return sum_.time[a] < sum_.time[b];
                   });
  Rows sorted;
  std::vector<std::size_t> counts(order.size());
  sorted.time.resize(order.size());
  sorted.value.resize(sum_.value.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    sorted.time[k] = sum_.time[order[k]];
    counts[k] = count_[order[k]];
    std::copy_n(sum_.value.data() + order[k] * total_columns_, total_columns_,
                sorted.value.data() + k * total_columns_);
  }
  sum_ = std::move(sorted);
  count_ = std::move(counts);
}

std::size_t all_cscd::averaged_rows() const {
  return static_cast<std::size_t>(
      std::count_if(count_.begin(), count_.end(),
                    [](std::size_t count) { return count > 1; }));
}

std::size_t all_cscd::write(const std::string &directory) {
  end_trajectory({});
  if (count_.empty())
    return 0;
  write_files(directory, sum_, &count_);
  return labels_.size();
}

void all_cscd::write_files(const std::string &directory, const Rows &rows,
                           const std::vector<std::size_t> *counts) const {
  constexpr int kTimeWidth = 10;
  constexpr int kValueWidth = 14;
  constexpr int kDecimals = 4;
  const std::size_t n2 = partner_labels_.size();

  for (std::size_t r = 0; r < labels_.size(); ++r) {
    const std::string path = (directory.empty() ? std::string{}
                                                : directory + "/") +
                             options_.file_prefix + labels_[r] + ".dat";
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out)
      throw std::runtime_error("cannot write " + path);

    const std::size_t skip = self_partner_[r];
    const std::size_t columns =
        options_.nearest_only ? 1 : n2 - (skip == kNone ? 0 : 1);
    std::string text;
    text += "# TITAN -cscd: centre-of-mass distance (" + options_.length_unit +
            ") from molecule-1 residue " + labels_[r] + "\n";
    text += "# centres: " + options_.centre + "\n";
    if (options_.nearest_only)
      text += "# column 2: nearest molecule-2 residue (-act gives one "
              "column per residue)\n";
    else
      text += "# columns 2-" + std::to_string(columns + 1) +
              ": molecule-2 residues, labelled below\n";
    if (counts != nullptr && trajectories_ > 1)
      text += "# rows: " + std::to_string(trajectories_) +
              " trajectories merged by time; " +
              std::to_string(averaged_rows()) + " of " +
              std::to_string(count_.size()) +
              " rows average frames of several trajectories\n";
    text += '#';
    append_right(text, "time(" + options_.time_unit + ")", kTimeWidth - 1);
    if (options_.nearest_only) {
      append_right(text, "nearest", kValueWidth);
    } else {
      for (std::size_t j = 0; j < n2; ++j)
        if (j != skip)
          append_right(text, partner_labels_[j], kValueWidth);
    }
    text += '\n';

    for (std::size_t k = 0; k < rows.time.size(); ++k) {
      const std::size_t count = counts != nullptr ? (*counts)[k] : 1;
      if (count == 0)
        continue;
      append_fixed(text, rows.time[k], kTimeWidth, kDecimals);
      const double *values =
          rows.value.data() + k * total_columns_ + offset_[r];
      for (std::size_t c = 0; c < columns; ++c)
        append_fixed(text, count == 1 ? values[c] : values[c] / count,
                     kValueWidth, kDecimals);
      text += '\n';
      if (text.size() > (1u << 20)) {
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        text.clear();
      }
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out)
      throw std::runtime_error("failed while writing " + path);
  }
}

// Split from titan_llps.cpp: macro contact kernels and geometry.
#include "titan_llps.h"
#include "titan_gyration.h"
#include "titan_simd_dispatch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

void TitanLLPS::UF::init(int n) {
  parent.resize(n);
  rank_.resize(n, 0);
  size_.resize(n, 1);
  std::iota(parent.begin(), parent.end(), 0);
}

void TitanLLPS::UF::reset(int n) {
  std::iota(parent.begin(), parent.begin() + n, 0);
  std::fill(rank_.begin(), rank_.begin() + n, 0);
  std::fill(size_.begin(), size_.begin() + n, 1);
}

int TitanLLPS::UF::find(int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void TitanLLPS::UF::merge(int a, int b) {
  a = find(a);
  b = find(b);
  if (a == b)
    return;
  if (rank_[a] < rank_[b])
    std::swap(a, b);
  parent[b] = a;
  size_[a] += size_[b];
  if (rank_[a] == rank_[b])
    ++rank_[a];
}

int TitanLLPS::UF::largest_component() const {
  return static_cast<int>(*std::max_element(size_.begin(), size_.end()));
}

uint64_t TitanLLPS::pack_ordered_pair(int a, int b) {
  if (a > b)
    std::swap(a, b);
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}

uint64_t TitanLLPS::pack_prna_pair(int protein_chain, int rna_chain_local) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(protein_chain)) << 32) |
         static_cast<uint32_t>(rna_chain_local);
}

std::pair<int, int> TitanLLPS::unpack_pair(uint64_t key) {
  const int a = static_cast<int>(key >> 32);
  const int b = static_cast<int>(key & 0xffffffffu);
  return {a, b};
}

float TitanLLPS::pbc_dist2(float dx, float dy, float dz, float bx, float by,
                           float bz) {
  if (bx > 0) dx -= bx * std::round(dx / bx);
  if (by > 0) dy -= by * std::round(dy / by);
  if (bz > 0) dz -= bz * std::round(dz / bz);
  return dx * dx + dy * dy + dz * dz;
}

int TitanLLPS::pair_index(int i, int j) const {
  return i * (2 * n_macro_ - i - 1) / 2 + (j - i - 1);
}

std::size_t TitanLLPS::contact_upper_index(int i, int j) const {
  if (i > j)
    std::swap(i, j);
  return static_cast<std::size_t>(i) *
             static_cast<std::size_t>(2 * n_stickers_ - i - 1) / 2 +
         static_cast<std::size_t>(j - i - 1);
}

void TitanLLPS::set_adj_bit(std::vector<uint64_t> &words, int i,
                                   int j) const {
  if (i > j)
    std::swap(i, j);
  const int idx = pair_index(i, j);
  words[static_cast<std::size_t>(idx >> 6)] |= (uint64_t(1) << (idx & 63));
}

bool TitanLLPS::kernel_allows_avx2() const noexcept {
  if (kernel_mode_ == LlpsKernelMode::Scalar)
    return false;
#if TITAN_SIMD_X86
  return titan_simd_dispatch::runtime_has_avx2();
#else
  return false;
#endif
}

void TitanLLPS::build_contacts(
    const std::vector<float> &pos_nm, float bx, float by, float bz,
    std::vector<uint64_t> &macro_contact_keys,
    std::vector<uint64_t> *sticker_pair_keys,
    std::vector<uint64_t> *protein_chain_pair_keys) {
  macro_contact_keys.clear();
  if (sticker_pair_keys)
    sticker_pair_keys->clear();
  if (protein_chain_pair_keys)
    protein_chain_pair_keys->clear();
  detailed_pp_frame_keys_buf_.clear();
  const int n_atoms = static_cast<int>(pos_nm.size()) / 3;

  float Lx = bx;
  float Ly = by;
  float Lz = bz;
  float shift_x = 0.0f;
  float shift_y = 0.0f;
  float shift_z = 0.0f;
  const bool periodic = cell_.metric.has_value() || (bx > 0.0f && by > 0.0f && bz > 0.0f);
  if(cell_.metric) {Lx=cell_.heights[0];Ly=cell_.heights[1];Lz=cell_.heights[2];}
  if (!periodic) {
    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float zmin = std::numeric_limits<float>::max();
    float xmax = -std::numeric_limits<float>::max();
    float ymax = -std::numeric_limits<float>::max();
    float zmax = -std::numeric_limits<float>::max();
    for (int i = 0; i < n_atoms; ++i) {
      xmin = std::min(xmin, pos_nm[3 * i + 0]);
      ymin = std::min(ymin, pos_nm[3 * i + 1]);
      zmin = std::min(zmin, pos_nm[3 * i + 2]);
      xmax = std::max(xmax, pos_nm[3 * i + 0]);
      ymax = std::max(ymax, pos_nm[3 * i + 1]);
      zmax = std::max(zmax, pos_nm[3 * i + 2]);
    }
    Lx = xmax - xmin + 2.0f * r_cut_;
    Ly = ymax - ymin + 2.0f * r_cut_;
    Lz = zmax - zmin + 2.0f * r_cut_;
    shift_x = -xmin + r_cut_;
    shift_y = -ymin + r_cut_;
    shift_z = -zmin + r_cut_;
  }

  titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff = r_cut_;
  spec.len_x = Lx;
  spec.len_y = Ly;
  spec.len_z = Lz;
  spec.shift_x = shift_x;
  spec.shift_y = shift_y;
  spec.shift_z = shift_z;
  spec.periodic = periodic;

  const auto active_fn = [&](std::size_t atom) noexcept {
    return atom < atom_macro_.size() && atom_macro_[atom] >= 0;
  };
  const auto coord_fn = [&](std::size_t atom, double &x, double &y,
                            double &z) noexcept {
    x = static_cast<double>(pos_nm[3 * atom + 0]);
    y = static_cast<double>(pos_nm[3 * atom + 1]);
    z = static_cast<double>(pos_nm[3 * atom + 2]);
    cell_.grid_coordinate(x,y,z);
  };

  ++cur_gen_;
  ++sticker_cur_gen_;
  contact_grid_.for_each_pair(
      static_cast<std::size_t>(n_atoms), spec, coord_fn, active_fn,
      [&](std::size_t lhs, std::size_t rhs) {
        const int a = static_cast<int>(lhs);
        const int b = static_cast<int>(rhs);
        const int mi = atom_macro_[lhs];
        const int mj = atom_macro_[rhs];
        if (mi < 0 || mj < 0)
          return;

        const double d2 = frame_dist2(
            pos_nm[3 * a + 0] - pos_nm[3 * b + 0],
            pos_nm[3 * a + 1] - pos_nm[3 * b + 1],
            pos_nm[3 * a + 2] - pos_nm[3 * b + 2], periodic ? bx : 0.0f,
            periodic ? by : 0.0f, periodic ? bz : 0.0f);
        if (d2 >= r_cut2_)
          return;

        if (mi != mj) {
          const int key = std::min(mi, mj) * n_macro_ + std::max(mi, mj);
          if (seen_gen_[static_cast<std::size_t>(key)] != cur_gen_) {
            seen_gen_[static_cast<std::size_t>(key)] = cur_gen_;
            macro_contact_keys.push_back(
                pack_ordered_pair(std::min(mi, mj), std::max(mi, mj)));
          }
        }

        if (lhs >= sticker_local_idx_.size() || rhs >= sticker_local_idx_.size())
          return;
        const int sticker_i = sticker_local_idx_[lhs];
        const int sticker_j = sticker_local_idx_[rhs];
        if (sticker_i < 0 || sticker_j < 0)
          return;

        const int chain_i =
            (a < static_cast<int>(atom_chain_.size())) ? atom_chain_[a] : -1;
        const int chain_j =
            (b < static_cast<int>(atom_chain_.size())) ? atom_chain_[b] : -1;
        if (chain_i < 0 || chain_j < 0)
          return;

        if (contacts_on_ &&
            lhs < protein_atom_contact_node_.size() &&
            rhs < protein_atom_contact_node_.size()) {
          const int residue_i = protein_atom_contact_node_[lhs];
          const int residue_j = protein_atom_contact_node_[rhs];
          if (residue_i >= 0 && residue_j >= 0 && residue_i != residue_j) {
            detailed_pp_frame_keys_buf_.push_back(
                pack_ordered_pair(residue_i, residue_j));
          }
        }

        // Preserve legacy map/lifetime semantics: only contacts between
        // distinct dimer-sized macro-nodes contribute to those outputs.
        if (mi == mj || chain_i == chain_j)
          return;

        if (sticker_pair_keys)
          sticker_pair_keys->push_back(pack_ordered_pair(a, b));
        if (protein_chain_pair_keys)
          protein_chain_pair_keys->push_back(pack_ordered_pair(chain_i, chain_j));

        if (!contacts_on_ || n_stickers_ <= 0)
          return;
        int si = sticker_i;
        int sj = sticker_j;
        if (si == sj)
          return;
        if (si > sj)
          std::swap(si, sj);
        const std::size_t sticker_key = contact_upper_index(si, sj);
        if (sticker_key < sticker_seen_gen_.size() &&
            sticker_seen_gen_[sticker_key] == sticker_cur_gen_) {
          return;
        }
        if (sticker_key < sticker_seen_gen_.size())
          sticker_seen_gen_[sticker_key] = sticker_cur_gen_;
        contact_accum_upper_[sticker_key] += 1.0f;
      });

  if (sticker_pair_keys) {
    std::sort(sticker_pair_keys->begin(), sticker_pair_keys->end());
    sticker_pair_keys->erase(
        std::unique(sticker_pair_keys->begin(), sticker_pair_keys->end()),
        sticker_pair_keys->end());
  }
  if (protein_chain_pair_keys) {
    std::sort(protein_chain_pair_keys->begin(), protein_chain_pair_keys->end());
    protein_chain_pair_keys->erase(
        std::unique(protein_chain_pair_keys->begin(),
                    protein_chain_pair_keys->end()),
        protein_chain_pair_keys->end());
  }
  if (contacts_on_ && !detailed_pp_frame_keys_buf_.empty()) {
    std::sort(detailed_pp_frame_keys_buf_.begin(),
              detailed_pp_frame_keys_buf_.end());
    detailed_pp_frame_keys_buf_.erase(
        std::unique(detailed_pp_frame_keys_buf_.begin(),
                    detailed_pp_frame_keys_buf_.end()),
        detailed_pp_frame_keys_buf_.end());
    for (const uint64_t key : detailed_pp_frame_keys_buf_)
      ++detailed_contact_frame_counts_[key];
  }
}

void TitanLLPS::build_protein_rna_contacts(
    const std::vector<float> &pos_nm, const float *rna_nm_data, int n_rna,
    float bx, float by, float bz, std::vector<uint64_t> &protein_rna_pairs) {
  protein_rna_pairs.clear();
  detailed_prna_frame_keys_buf_.clear();
  if ((!dynamics_enabled_ && !contacts_on_) || prna_cutoff_nm_ <= 0.0f ||
      !rna_nm_data ||
      n_rna <= 0 || sticker_atom_idx_.empty() || rna_atom_chain_.empty())
    return;

  const int usable_rna_atoms =
      std::min(n_rna, static_cast<int>(rna_atom_chain_.size()));
  if (usable_rna_atoms <= 0)
    return;

  float Lx = bx;
  float Ly = by;
  float Lz = bz;
  float shift_x = 0.0f;
  float shift_y = 0.0f;
  float shift_z = 0.0f;
  const bool periodic = cell_.metric.has_value() || (bx > 0.0f && by > 0.0f && bz > 0.0f);
  if(cell_.metric) {Lx=cell_.heights[0];Ly=cell_.heights[1];Lz=cell_.heights[2];}
  if (!periodic) {
    float xmin = std::numeric_limits<float>::max();
    float ymin = std::numeric_limits<float>::max();
    float zmin = std::numeric_limits<float>::max();
    float xmax = -std::numeric_limits<float>::max();
    float ymax = -std::numeric_limits<float>::max();
    float zmax = -std::numeric_limits<float>::max();
    for (int atom : sticker_atom_idx_) {
      xmin = std::min(xmin, pos_nm[3 * atom + 0]);
      ymin = std::min(ymin, pos_nm[3 * atom + 1]);
      zmin = std::min(zmin, pos_nm[3 * atom + 2]);
      xmax = std::max(xmax, pos_nm[3 * atom + 0]);
      ymax = std::max(ymax, pos_nm[3 * atom + 1]);
      zmax = std::max(zmax, pos_nm[3 * atom + 2]);
    }
    for (int i = 0; i < usable_rna_atoms; ++i) {
      xmin = std::min(xmin, rna_nm_data[3 * i + 0]);
      ymin = std::min(ymin, rna_nm_data[3 * i + 1]);
      zmin = std::min(zmin, rna_nm_data[3 * i + 2]);
      xmax = std::max(xmax, rna_nm_data[3 * i + 0]);
      ymax = std::max(ymax, rna_nm_data[3 * i + 1]);
      zmax = std::max(zmax, rna_nm_data[3 * i + 2]);
    }
    Lx = xmax - xmin + 2.0f * prna_cutoff_nm_;
    Ly = ymax - ymin + 2.0f * prna_cutoff_nm_;
    Lz = zmax - zmin + 2.0f * prna_cutoff_nm_;
    shift_x = -xmin + prna_cutoff_nm_;
    shift_y = -ymin + prna_cutoff_nm_;
    shift_z = -zmin + prna_cutoff_nm_;
  }

  titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff = prna_cutoff_nm_;
  spec.len_x = Lx;
  spec.len_y = Ly;
  spec.len_z = Lz;
  spec.shift_x = shift_x;
  spec.shift_y = shift_y;
  spec.shift_z = shift_z;
  spec.periodic = periodic;

  const std::size_t sticker_count = sticker_atom_idx_.size();
  const std::size_t point_count = sticker_count + usable_rna_atoms;
  const auto active_fn = [&](std::size_t) noexcept { return true; };
  const auto coord_fn = [&](std::size_t idx, double &x, double &y,
                            double &z) noexcept {
    if (idx < sticker_count) {
      const int atom = sticker_atom_idx_[idx];
      x = static_cast<double>(pos_nm[3 * atom + 0]);
      y = static_cast<double>(pos_nm[3 * atom + 1]);
      z = static_cast<double>(pos_nm[3 * atom + 2]);
      cell_.grid_coordinate(x,y,z);
      return;
    }
    const int rna_atom = static_cast<int>(idx - sticker_count);
    x = static_cast<double>(rna_nm_data[3 * rna_atom + 0]);
    y = static_cast<double>(rna_nm_data[3 * rna_atom + 1]);
    z = static_cast<double>(rna_nm_data[3 * rna_atom + 2]);
    cell_.grid_coordinate(x,y,z);
  };

  prna_grid_.for_each_pair(
      point_count, spec, coord_fn, active_fn,
      [&](std::size_t lhs, std::size_t rhs) {
        const bool lhs_is_sticker = lhs < sticker_count;
        const bool rhs_is_sticker = rhs < sticker_count;
        if (lhs_is_sticker == rhs_is_sticker)
          return;

        int protein_atom = -1;
        int rna_atom = -1;
        if (lhs_is_sticker) {
          protein_atom = sticker_atom_idx_[lhs];
          rna_atom = static_cast<int>(rhs - sticker_count);
        } else {
          protein_atom = sticker_atom_idx_[rhs];
          rna_atom = static_cast<int>(lhs - sticker_count);
        }

        const float px = pos_nm[3 * protein_atom + 0];
        const float py = pos_nm[3 * protein_atom + 1];
        const float pz = pos_nm[3 * protein_atom + 2];
        const float rx = rna_nm_data[3 * rna_atom + 0];
        const float ry = rna_nm_data[3 * rna_atom + 1];
        const float rz = rna_nm_data[3 * rna_atom + 2];
        const double d2 = frame_dist2(px - rx, py - ry, pz - rz,
                                   periodic ? bx : 0.0f, periodic ? by : 0.0f,
                                   periodic ? bz : 0.0f);
        if (d2 >= prna_cutoff2_nm_)
          return;

        const int protein_chain =
            (protein_atom < static_cast<int>(atom_chain_.size()))
                ? atom_chain_[protein_atom]
                : -1;
        const int rna_chain_local =
            (rna_atom < static_cast<int>(rna_atom_chain_.size()))
                ? rna_atom_chain_[rna_atom]
                : -1;
        if (protein_chain < 0 || rna_chain_local < 0)
          return;
        protein_rna_pairs.push_back(
            pack_prna_pair(protein_chain, rna_chain_local));

        if (contacts_on_ &&
            protein_atom < static_cast<int>(protein_atom_contact_node_.size()) &&
            rna_atom < static_cast<int>(rna_atom_contact_node_.size())) {
          const int protein_residue =
              protein_atom_contact_node_[static_cast<std::size_t>(protein_atom)];
          const int rna_residue =
              rna_atom_contact_node_[static_cast<std::size_t>(rna_atom)];
          if (protein_residue >= 0 && rna_residue >= 0) {
            detailed_prna_frame_keys_buf_.push_back(
                pack_ordered_pair(protein_residue, rna_residue));
          }
        }
      });

  std::sort(protein_rna_pairs.begin(), protein_rna_pairs.end());
  protein_rna_pairs.erase(
      std::unique(protein_rna_pairs.begin(), protein_rna_pairs.end()),
      protein_rna_pairs.end());
  if (contacts_on_ && !detailed_prna_frame_keys_buf_.empty()) {
    std::sort(detailed_prna_frame_keys_buf_.begin(),
              detailed_prna_frame_keys_buf_.end());
    detailed_prna_frame_keys_buf_.erase(
        std::unique(detailed_prna_frame_keys_buf_.begin(),
                    detailed_prna_frame_keys_buf_.end()),
        detailed_prna_frame_keys_buf_.end());
    for (const uint64_t key : detailed_prna_frame_keys_buf_)
      ++detailed_contact_frame_counts_[key];
  }
}

float TitanLLPS::compute_gcc_rg(const std::vector<float> &pos_nm, float bx,
                                float by, float bz,
                                const std::vector<int> &gcc_macros) const {
  (void)bx; (void)by; (void)bz;
  using titan_llps_geometry::Point;
  gcc_center_.reset();
  const auto nan=std::numeric_limits<float>::quiet_NaN();
  if(gcc_macros.empty()) return nan;
  std::vector<Point> lifted(n_macro_);
  std::vector<bool> visited(n_macro_,false),member(n_macro_,false);
  std::vector<std::vector<int>> edges(n_macro_);
  for(int m:gcc_macros) {if(bb_atom_[m]<0) return nan;member[m]=true;}
  for(auto key:macro_contact_keys_buf_) {
    auto [a,b]=unpack_pair(key);
    if(member[a] && member[b]) {edges[a].push_back(b);edges[b].push_back(a);}
  }
  const int root=gcc_macros.front(), atom=bb_atom_[root];
  lifted[root]={pos_nm[3*atom],pos_nm[3*atom+1],pos_nm[3*atom+2]};
  std::vector<int> queue{root};visited[root]=true;
  for(std::size_t q=0;q<queue.size();++q) {
    int a=queue[q];
    for(int b:edges[a]) {
      Point d{};for(int k=0;k<3;++k)d[k]=double(pos_nm[3*bb_atom_[b]+k])-pos_nm[3*bb_atom_[a]+k];
      bool ambiguous=false;d=cell_.delta(d,&ambiguous);
      if(ambiguous) return nan;
      Point candidate{};for(int k=0;k<3;++k) candidate[k]=lifted[a][k]+d[k];
      if(!visited[b]) {lifted[b]=candidate;visited[b]=true;queue.push_back(b);}
      else for(int k=0;k<3;++k) if(std::abs(lifted[b][k]-candidate[k])>1e-5) return nan; // winding representative graph
    }
  }
  if(queue.size()!=gcc_macros.size()) return nan;
  Point center{};for(int m:gcc_macros)for(int k=0;k<3;++k)center[k]+=lifted[m][k]/gcc_macros.size();
  double r2=0;for(int m:gcc_macros)for(int k=0;k<3;++k)r2+=std::pow(lifted[m][k]-center[k],2);
  gcc_center_=center;
  return std::sqrt(r2/gcc_macros.size());
}

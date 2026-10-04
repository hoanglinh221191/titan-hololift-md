// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "titan_exact_periodic_box.h"
#include "MOL.h"
#include "MOL_info.h"
#include "box_type.h"
#include "info.h"
#include "titan_executor.h"
#include "titan_energy_decomposition.h"
#include "titan_llps_stats.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace titan_llps_stats;

void TitanLLPSCluster::refresh_rna_compaction_topology_cache(
    const std::vector<mol_info> &pro_info,
    const std::vector<mol_info> &lig_info) {
  std::vector<std::string> sticker_names(rna_compaction_sticker_names_.begin(),
                                         rna_compaction_sticker_names_.end());
  std::sort(sticker_names.begin(), sticker_names.end());
  std::ostringstream sig;
  for (std::size_t i = 0; i < sticker_names.size(); ++i) {
    if (i > 0)
      sig << ';';
    sig << sticker_names[i];
  }
  const std::string sticker_signature = sig.str();
  if (rna_compaction_topology_cache_.ready &&
      rna_compaction_topology_cache_.pro_info_size == pro_info.size() &&
      rna_compaction_topology_cache_.lig_info_size == lig_info.size() &&
      rna_compaction_topology_cache_.sticker_signature == sticker_signature) {
    return;
  }

  rna_compaction_topology_cache_ = {};
  auto &cache = rna_compaction_topology_cache_;
  cache.pro_info_size = pro_info.size();
  cache.lig_info_size = lig_info.size();
  cache.sticker_signature = sticker_signature;

  if (!rna_compaction_sticker_names_.empty()) {
    cache.sticker_atom_indices.reserve(pro_info.size() / 4 + 1);
    cache.sticker_chain_ids.reserve(pro_info.size() / 4 + 1);
    for (std::size_t i = 0; i < pro_info.size(); ++i) {
      if (rna_compaction_sticker_names_.find(pro_info[i].get_atom()) ==
          rna_compaction_sticker_names_.end())
        continue;
      const int chain_id = static_cast<int>(pro_info[i].get_chainnum());
      const int local_index =
          static_cast<int>(cache.sticker_atom_indices.size());
      cache.sticker_atom_indices.push_back(i);
      cache.sticker_chain_ids.push_back(chain_id);
      cache.sticker_indices_by_chain[chain_id].push_back(local_index);
    }
  }

  cache.rna_atom_chain_ids.reserve(lig_info.size());
  cache.rna_atom_residue_ids.reserve(lig_info.size());
  cache.rna_chain_ids.reserve(lig_info.size());
  for (const auto &atom : lig_info) {
    const int chain_id = static_cast<int>(atom.get_chainnum());
    cache.rna_atom_chain_ids.push_back(chain_id);
    cache.rna_atom_residue_ids.push_back(static_cast<int>(atom.get_resnr()));
    cache.rna_chain_ids.push_back(chain_id);
  }
  std::sort(cache.rna_chain_ids.begin(), cache.rna_chain_ids.end());
  cache.rna_chain_ids.erase(
      std::unique(cache.rna_chain_ids.begin(), cache.rna_chain_ids.end()),
      cache.rna_chain_ids.end());
  cache.ready = true;
}

void TitanLLPSCluster::update_rna_compaction(
    const MOL &pro, const std::vector<mol_info> &pro_info, const MOL &lig,
    const std::vector<mol_info> &lig_info, const box &Box, double time_ns,
    int frame_idx, const std::vector<ChainFrameSummary> &frame_chains) {
  const std::size_t n_rna =
      std::min({lig.pos.x.size(), lig.pos.y.size(), lig.pos.z.size(),
                lig_info.size()});
  if (!rna_compaction_enabled_ || n_rna == 0)
    return;
  refresh_rna_compaction_topology_cache(pro_info, lig_info);

  struct RNABead {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    int residue{0};
    std::size_t atom_index{0};
  };
  struct ProteinSticker {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    int chain{0};
    std::size_t atom_index{0};
  };
  struct ProteinRNAEnergyAccum {
    int contact_count{0};
    double elec{0.0};
    double vdw{0.0};
  };
  struct ContactObservation {
    int residue{0};
    std::size_t local_bead_index{0};
    double ux{0.0};
    double uy{0.0};
    double uz{0.0};
    double elec{0.0};
    double vdw{0.0};
  };
  struct ContactPatch {
    int min_res{0};
    int max_res{0};
    double sum_x{0.0};
    double sum_y{0.0};
    double sum_z{0.0};
    int point_count{0};
    int contact_count{0};
    double elec{0.0};
    double vdw{0.0};

    double com_x() const {
      return (point_count > 0) ? (sum_x / static_cast<double>(point_count)) : 0.0;
    }
    double com_y() const {
      return (point_count > 0) ? (sum_y / static_cast<double>(point_count)) : 0.0;
    }
    double com_z() const {
      return (point_count > 0) ? (sum_z / static_cast<double>(point_count)) : 0.0;
    }
  };
  struct BridgeRecord {
    int protein_chain_id{0};
    int patch_count{0};
    int bridge_count{0};
    int bridge_residue_span_max{0};
    double bridge_geom_span_max_ang{0.0};
    int bridge_contact_count{0};
    double bridge_attraction_kjmol{0.0};
    bool supports_longrange_contact{false};
  };
  struct ClampRecord {
    int protein_chain_id_a{0};
    int protein_chain_id_b{0};
    int clamp_count{0};
    int clamp_residue_span_max{0};
    double clamp_geom_span_max_ang{0.0};
    int clamp_supported_longrange_contacts{0};
    double clamp_attraction_kjmol{0.0};
  };

  std::unordered_map<int, std::pair<bool, int>> dense_lookup;
  dense_lookup.reserve(frame_chains.size());
  for (const auto &chain : frame_chains) {
    if (chain.species == 1) {
      dense_lookup[chain.raw_chain_id] =
          std::make_pair(chain.inside_dense, chain.cluster_id);
    }
  }

  std::unordered_map<int, std::vector<RNABead>> rna_chains;
  rna_chains.reserve(rna_compaction_topology_cache_.rna_chain_ids.size() + 1);
  for (int chain_id : rna_compaction_topology_cache_.rna_chain_ids)
    rna_chains.emplace(chain_id, std::vector<RNABead>{});
  for (std::size_t i = 0; i < n_rna &&
                          i < rna_compaction_topology_cache_
                                  .rna_atom_chain_ids.size();
       ++i) {
    const int chain_id =
        rna_compaction_topology_cache_.rna_atom_chain_ids[i];
    rna_chains[chain_id].push_back(
        {lig.pos.x[i], lig.pos.y[i], lig.pos.z[i],
         rna_compaction_topology_cache_.rna_atom_residue_ids[i], i});
  }
  if (rna_chains.empty())
    return;

  const TitanExactPeriodicBox bmat(Box);
  const double prna_cut = rna_compaction_prna_cutoff_angstrom_;
  const double prna_cut_sq = prna_cut * prna_cut;
  const double intra_cut = rna_compaction_intra_cutoff_angstrom_;
  const double intra_cut_sq = intra_cut * intra_cut;

  std::vector<ProteinSticker> stickers;
  const std::size_t n_pro =
      std::min({pro.pos.x.size(), pro.pos.y.size(), pro.pos.z.size(),
                pro_info.size()});
  stickers.reserve(rna_compaction_topology_cache_.sticker_atom_indices.size());
  for (std::size_t local = 0;
       local < rna_compaction_topology_cache_.sticker_atom_indices.size();
       ++local) {
    const std::size_t i =
        rna_compaction_topology_cache_.sticker_atom_indices[local];
    if (i < n_pro) {
      stickers.push_back({pro.pos.x[i], pro.pos.y[i], pro.pos.z[i],
                          rna_compaction_topology_cache_.sticker_chain_ids[local],
                          i});
    }
  }
  std::unordered_map<int, std::vector<int>> fallback_sticker_indices_by_chain;
  const auto *sticker_indices_by_chain =
      &rna_compaction_topology_cache_.sticker_indices_by_chain;
  if (stickers.size() !=
      rna_compaction_topology_cache_.sticker_atom_indices.size()) {
    fallback_sticker_indices_by_chain.reserve(stickers.size() + 1);
    for (std::size_t i = 0; i < stickers.size(); ++i) {
      fallback_sticker_indices_by_chain[stickers[i].chain].push_back(
          static_cast<int>(i));
    }
    sticker_indices_by_chain = &fallback_sticker_indices_by_chain;
  }

  const double bx = bmat.heights[0];
  const double by = bmat.heights[1];
  const double bz = bmat.heights[2];
  const bool have_periodic_box = bx > 0.0 && by > 0.0 && bz > 0.0;
  int nx = 1;
  int ny = 1;
  int nz = 1;
  double cx = std::max(prna_cut, 1.0);
  double cy = std::max(prna_cut, 1.0);
  double cz = std::max(prna_cut, 1.0);
  std::unordered_map<uint64_t, std::vector<int>> sticker_grid;
  if (have_periodic_box && !stickers.empty()) {
    nx = std::max(1, static_cast<int>(bx / std::max(prna_cut, 1e-6)));
    ny = std::max(1, static_cast<int>(by / std::max(prna_cut, 1e-6)));
    nz = std::max(1, static_cast<int>(bz / std::max(prna_cut, 1e-6)));
    cx = bx / static_cast<double>(nx);
    cy = by / static_cast<double>(ny);
    cz = bz / static_cast<double>(nz);
    sticker_grid.reserve(stickers.size() * 2 + 1);
    auto periodic_cell = [&](double x, double y, double z) {
      bmat.grid_coordinate(x,y,z);
      int ix = static_cast<int>(std::floor(x / cx));
      int iy = static_cast<int>(std::floor(y / cy));
      int iz = static_cast<int>(std::floor(z / cz));
      ix = ((ix % nx) + nx) % nx;
      iy = ((iy % ny) + ny) % ny;
      iz = ((iz % nz) + nz) % nz;
      return pack_periodic_cell(ix, iy, iz, ny, nz);
    };
    for (std::size_t i = 0; i < stickers.size(); ++i)
      sticker_grid[periodic_cell(stickers[i].x, stickers[i].y, stickers[i].z)]
          .push_back(static_cast<int>(i));
  }

  auto count_protein_binding =
      [&](const std::vector<RNABead> &beads,
          const std::vector<std::array<double, 3>> &xyz, int &bound_chains,
          int &bead_contacts, EnergyPairDecomposition &chain_energy,
          std::vector<int> &partner_chain_ids,
          std::unordered_map<int, ProteinRNAEnergyAccum> &pair_energy,
          std::unordered_map<int, std::vector<ContactObservation>>
              &observations_by_protein) {
    bound_chains = 0;
    bead_contacts = 0;
    chain_energy = {};
    partner_chain_ids.clear();
    pair_energy.clear();
    observations_by_protein.clear();
    if (!have_periodic_box || stickers.empty())
      return;
    std::vector<int> chain_hits;
    for (std::size_t bead_idx = 0; bead_idx < beads.size() && bead_idx < xyz.size();
         ++bead_idx) {
      const auto &rna = beads[bead_idx];
      double gx=rna.x,gy=rna.y,gz=rna.z;bmat.grid_coordinate(gx,gy,gz);
      int ix = static_cast<int>(std::floor(gx / cx));
      int iy = static_cast<int>(std::floor(gy / cy));
      int iz = static_cast<int>(std::floor(gz / cz));
      ix = ((ix % nx) + nx) % nx;
      iy = ((iy % ny) + ny) % ny;
      iz = ((iz % nz) + nz) % nz;
      std::array<uint64_t, 27> seen_cell_keys{};
      int seen_count = 0;
      for (int dx_cell = -1; dx_cell <= 1; ++dx_cell) {
        const int jx = ((ix + dx_cell) % nx + nx) % nx;
        for (int dy_cell = -1; dy_cell <= 1; ++dy_cell) {
          const int jy = ((iy + dy_cell) % ny + ny) % ny;
          for (int dz_cell = -1; dz_cell <= 1; ++dz_cell) {
            const int jz = ((iz + dz_cell) % nz + nz) % nz;
            const auto key = pack_periodic_cell(jx, jy, jz, ny, nz);
            bool duplicate_cell = false;
            for (int s = 0; s < seen_count; ++s) {
              if (seen_cell_keys[static_cast<std::size_t>(s)] == key) {
                duplicate_cell = true;
                break;
              }
            }
            if (duplicate_cell)
              continue;
            seen_cell_keys[static_cast<std::size_t>(seen_count++)] = key;
            const auto hit = sticker_grid.find(key);
            if (hit == sticker_grid.end())
              continue;
            for (int sticker_index : hit->second) {
              const auto &sticker =
                  stickers[static_cast<std::size_t>(sticker_index)];
              double dx = rna.x - sticker.x;
              double dy = rna.y - sticker.y;
              double dz = rna.z - sticker.z;
              bmat.apply_mic(dx, dy, dz);
              if (dx * dx + dy * dy + dz * dz <= prna_cut_sq) {
                ++bead_contacts;
                chain_hits.push_back(sticker.chain);
                EnergyPairDecomposition e{};
                if (energy_provider_ != nullptr && energy_provider_->ready()) {
                  e = energy_provider_->pair_energy(
                      pro, lig, sticker.atom_index, rna.atom_index);
                  chain_energy += e;
                  auto &pair_acc = pair_energy[sticker.chain];
                  ++pair_acc.contact_count;
                  pair_acc.elec += e.elec;
                  pair_acc.vdw += e.vdw;
                }
                auto &observations = observations_by_protein[sticker.chain];
                observations.push_back({rna.residue,
                                        bead_idx,
                                        xyz[bead_idx][0],
                                        xyz[bead_idx][1],
                                        xyz[bead_idx][2],
                                        e.elec,
                                        e.vdw});
              }
            }
          }
        }
      }
    }
    std::sort(chain_hits.begin(), chain_hits.end());
    chain_hits.erase(std::unique(chain_hits.begin(), chain_hits.end()),
                     chain_hits.end());
    partner_chain_ids = chain_hits;
    bound_chains = static_cast<int>(chain_hits.size());
  };

  auto count_intrachain_contacts =
      [&](const std::vector<std::array<double, 3>> &xyz,
          const std::vector<RNABead> &beads, int &contacts,
          int &longrange_contacts,
          std::vector<std::pair<int, int>> *longrange_pairs) {
        contacts = 0;
        longrange_contacts = 0;
        if (xyz.size() < 2 || intra_cut <= 0.0)
          return;
        std::unordered_map<uint64_t, std::vector<int>> grid;
        grid.reserve(xyz.size() * 2 + 1);
        std::unordered_set<uint64_t> unique_longrange_pairs;
        if (longrange_pairs != nullptr)
          longrange_pairs->clear();
        auto cell_of_unwrapped = [&](const std::array<double, 3> &p) {
          const int ix = static_cast<int>(std::floor(p[0] / intra_cut));
          const int iy = static_cast<int>(std::floor(p[1] / intra_cut));
          const int iz = static_cast<int>(std::floor(p[2] / intra_cut));
          return std::array<int, 3>{ix, iy, iz};
        };
        for (std::size_t i = 0; i < xyz.size(); ++i) {
          const auto cell = cell_of_unwrapped(xyz[i]);
          for (int dx_cell = -1; dx_cell <= 1; ++dx_cell) {
            for (int dy_cell = -1; dy_cell <= 1; ++dy_cell) {
              for (int dz_cell = -1; dz_cell <= 1; ++dz_cell) {
                const uint64_t key =
                    pack_signed_cell(cell[0] + dx_cell, cell[1] + dy_cell,
                                     cell[2] + dz_cell);
                const auto hit = grid.find(key);
                if (hit == grid.end())
                  continue;
                for (int j : hit->second) {
                  const auto jj = static_cast<std::size_t>(j);
                  if (beads[i].residue == beads[jj].residue)
                    continue;
                  const double ddx = xyz[i][0] - xyz[jj][0];
                  const double ddy = xyz[i][1] - xyz[jj][1];
                  const double ddz = xyz[i][2] - xyz[jj][2];
                  if (ddx * ddx + ddy * ddy + ddz * ddz > intra_cut_sq)
                    continue;
                  ++contacts;
                  if (std::abs(beads[i].residue - beads[jj].residue) >=
                      rna_compaction_longrange_min_) {
                    ++longrange_contacts;
                    if (longrange_pairs != nullptr) {
                      const int lo_res =
                          std::min(beads[i].residue, beads[jj].residue);
                      const int hi_res =
                          std::max(beads[i].residue, beads[jj].residue);
                      const uint64_t key = pack_u32_pair(lo_res, hi_res);
                      if (unique_longrange_pairs.insert(key).second) {
                        longrange_pairs->push_back({lo_res, hi_res});
                      }
                    }
                  }
                }
              }
            }
          }
          grid[pack_signed_cell(cell[0], cell[1], cell[2])].push_back(
              static_cast<int>(i));
        }
      };

  auto build_contact_patches =
      [&](const std::vector<ContactObservation> &observations) {
        std::vector<ContactObservation> sorted = observations;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ContactObservation &a, const ContactObservation &b) {
                    if (a.residue != b.residue)
                      return a.residue < b.residue;
                    return a.local_bead_index < b.local_bead_index;
                  });
        std::vector<ContactPatch> patches;
        for (const auto &obs : sorted) {
          const bool need_new_patch =
              patches.empty() ||
              obs.residue > patches.back().max_res + rna_topology_patch_gap_max_ + 1;
          if (need_new_patch) {
            ContactPatch patch;
            patch.min_res = obs.residue;
            patch.max_res = obs.residue;
            patches.push_back(patch);
          }
          auto &patch = patches.back();
          patch.min_res = std::min(patch.min_res, obs.residue);
          patch.max_res = std::max(patch.max_res, obs.residue);
          patch.sum_x += obs.ux;
          patch.sum_y += obs.uy;
          patch.sum_z += obs.uz;
          patch.point_count += 1;
          patch.contact_count += 1;
          patch.elec += obs.elec;
          patch.vdw += obs.vdw;
        }
        return patches;
      };

  auto patch_residue_span = [](const ContactPatch &a, const ContactPatch &b) {
    return std::max(std::abs(a.min_res - b.max_res),
                    std::abs(a.max_res - b.min_res));
  };

  auto patch_geom_span_angstrom = [](const ContactPatch &a,
                                     const ContactPatch &b) {
    const double dx = a.com_x() - b.com_x();
    const double dy = a.com_y() - b.com_y();
    const double dz = a.com_z() - b.com_z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  auto count_supported_longrange_pairs =
      [&](const ContactPatch &a, const ContactPatch &b,
          const std::vector<std::pair<int, int>> &longrange_pairs) {
        int supported = 0;
        for (const auto &pair : longrange_pairs) {
          const bool direct = pair.first >= a.min_res && pair.first <= a.max_res &&
                              pair.second >= b.min_res &&
                              pair.second <= b.max_res;
          const bool reverse = pair.first >= b.min_res && pair.first <= b.max_res &&
                               pair.second >= a.min_res &&
                               pair.second <= a.max_res;
          if (direct || reverse)
            ++supported;
        }
        return supported;
      };

  auto attraction_strength = [](double total_kjmol) {
    return std::max(0.0, -total_kjmol);
  };

  auto count_local_pp_contacts =
      [&](const std::vector<int> &bound_chain_ids, int &count,
          EnergyPairDecomposition &local_pp_energy) {
    count = 0;
    local_pp_energy = {};
    if (!rna_topology_enabled_ || bound_chain_ids.size() < 2 ||
        rna_topology_pp_cutoff_angstrom_ <= 0.0 || !have_periodic_box)
      return;
    const double pp_cut_sq =
        rna_topology_pp_cutoff_angstrom_ * rna_topology_pp_cutoff_angstrom_;
    for (std::size_t a = 0; a < bound_chain_ids.size(); ++a) {
      const auto ita = sticker_indices_by_chain->find(bound_chain_ids[a]);
      if (ita == sticker_indices_by_chain->end())
        continue;
      for (std::size_t b = a + 1; b < bound_chain_ids.size(); ++b) {
        const auto itb = sticker_indices_by_chain->find(bound_chain_ids[b]);
        if (itb == sticker_indices_by_chain->end())
          continue;
        for (int idx_a : ita->second) {
          const auto &sa = stickers[static_cast<std::size_t>(idx_a)];
          for (int idx_b : itb->second) {
            const auto &sb = stickers[static_cast<std::size_t>(idx_b)];
            double dx = sa.x - sb.x;
            double dy = sa.y - sb.y;
            double dz = sa.z - sb.z;
            bmat.apply_mic(dx, dy, dz);
            if (dx * dx + dy * dy + dz * dz <= pp_cut_sq) {
              ++count;
              if (energy_provider_ != nullptr && energy_provider_->ready()) {
                local_pp_energy +=
                    energy_provider_->rec_pair_energy(pro, sa.atom_index,
                                                      sb.atom_index);
              }
            }
          }
        }
      }
    }
  };

  const auto &rna_chain_ids = rna_compaction_topology_cache_.rna_chain_ids;

  for (int rna_chain_id : rna_chain_ids) {
    const auto &beads = rna_chains.at(rna_chain_id);
    if (beads.empty())
      continue;

    std::vector<std::array<double, 3>> xyz(beads.size());
    xyz.front() = {beads.front().x, beads.front().y, beads.front().z};
    for (std::size_t i = 1; i < beads.size(); ++i) {
      double dx = beads[i].x - beads[i - 1].x;
      double dy = beads[i].y - beads[i - 1].y;
      double dz = beads[i].z - beads[i - 1].z;
      if (have_periodic_box)
        bmat.apply_mic(dx, dy, dz);
      xyz[i] = {xyz[i - 1][0] + dx, xyz[i - 1][1] + dy,
                xyz[i - 1][2] + dz};
    }

    double cx_chain = 0.0;
    double cy_chain = 0.0;
    double cz_chain = 0.0;
    for (const auto &p : xyz) {
      cx_chain += p[0];
      cy_chain += p[1];
      cz_chain += p[2];
    }
    const double inv_n = 1.0 / static_cast<double>(xyz.size());
    cx_chain *= inv_n;
    cy_chain *= inv_n;
    cz_chain *= inv_n;

    double xx = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yy = 0.0;
    double yz = 0.0;
    double zz = 0.0;
    for (const auto &p : xyz) {
      const double dx = p[0] - cx_chain;
      const double dy = p[1] - cy_chain;
      const double dz = p[2] - cz_chain;
      xx += dx * dx;
      xy += dx * dy;
      xz += dx * dz;
      yy += dy * dy;
      yz += dy * dz;
      zz += dz * dz;
    }
    xx *= inv_n;
    xy *= inv_n;
    xz *= inv_n;
    yy *= inv_n;
    yz *= inv_n;
    zz *= inv_n;

    const auto eig_a2 = eigenvalues_symmetric_3x3(xx, xy, xz, yy, yz, zz);
    const double rg2_a2 = std::max(0.0, eig_a2[0] + eig_a2[1] + eig_a2[2]);
    const double rg_nm = std::sqrt(rg2_a2) * 0.1;
    double end_to_end_nm = 0.0;
    if (xyz.size() > 1) {
      const double dx = xyz.back()[0] - xyz.front()[0];
      const double dy = xyz.back()[1] - xyz.front()[1];
      const double dz = xyz.back()[2] - xyz.front()[2];
      end_to_end_nm = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.1;
    }

    const double l1 = std::max(0.0, eig_a2[0]) * 0.01;
    const double l2 = std::max(0.0, eig_a2[1]) * 0.01;
    const double l3 = std::max(0.0, eig_a2[2]) * 0.01;
    const double rg2_nm2 = std::max(0.0, l1 + l2 + l3);
    const double asphericity_nm2 = l1 - 0.5 * (l2 + l3);
    const double normalized_asphericity =
        (rg2_nm2 > 0.0) ? (asphericity_nm2 / rg2_nm2) : 0.0;
    const double acylindricity_nm2 = l2 - l3;
    const double mean_l = rg2_nm2 / 3.0;
    const double relative_shape_anisotropy =
        (rg2_nm2 > 0.0)
            ? (1.5 * ((l1 - mean_l) * (l1 - mean_l) +
                      (l2 - mean_l) * (l2 - mean_l) +
                      (l3 - mean_l) * (l3 - mean_l)) /
               (rg2_nm2 * rg2_nm2))
            : 0.0;
    const double aspect_ratio = (l3 > 1e-12) ? std::sqrt(l1 / l3) : 0.0;

    int intrachain_contacts = 0;
    int longrange_intrachain_contacts = 0;
    std::vector<std::pair<int, int>> longrange_pairs;
    count_intrachain_contacts(xyz, beads, intrachain_contacts,
                              longrange_intrachain_contacts, &longrange_pairs);
    const double longrange_fraction =
        (intrachain_contacts > 0)
            ? static_cast<double>(longrange_intrachain_contacts) /
                  static_cast<double>(intrachain_contacts)
            : 0.0;

    int bound_protein_chains = 0;
    int protein_rna_bead_contacts = 0;
    EnergyPairDecomposition prna_energy;
    std::vector<int> partner_chain_ids;
    std::unordered_map<int, ProteinRNAEnergyAccum> pair_energy_by_protein;
    std::unordered_map<int, std::vector<ContactObservation>>
        observations_by_protein;
    count_protein_binding(beads, xyz, bound_protein_chains,
                          protein_rna_bead_contacts, prna_energy,
                          partner_chain_ids, pair_energy_by_protein,
                          observations_by_protein);
    const bool have_prna_energy =
        energy_provider_ != nullptr && energy_provider_->ready();

    int topology_state = 0;
    bool any_bridge = false;
    int num_bridge_proteins = 0;
    bool any_clamp = false;
    int num_clamp_pairs = 0;
    int bridge_residue_span_max = 0;
    double bridge_geom_span_max_ang = 0.0;
    int clamp_supported_longrange_contacts = 0;
    double bridge_total_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    double nonbridge_total_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    double bridge_attraction_strength_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    double nonbridge_attraction_strength_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    double prna_attraction_strength_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    int local_pp_sticker_contacts = 0;
    EnergyPairDecomposition local_pp_energy;
    double local_pp_total_kjmol = std::numeric_limits<double>::quiet_NaN();
    double local_pp_attraction_strength_kjmol =
        std::numeric_limits<double>::quiet_NaN();
    double local_pp_to_prna_contact_ratio =
        std::numeric_limits<double>::quiet_NaN();
    double local_pp_to_prna_attraction_ratio =
        std::numeric_limits<double>::quiet_NaN();
    double local_pp_to_bridge_attraction_ratio =
        std::numeric_limits<double>::quiet_NaN();

    std::vector<BridgeRecord> bridge_records;
    std::vector<ClampRecord> clamp_records;
    if (rna_topology_enabled_) {
      std::unordered_map<int, std::vector<ContactPatch>> patches_by_protein;
      patches_by_protein.reserve(observations_by_protein.size() + 1);
      for (int protein_chain_id : partner_chain_ids) {
        const auto obs_it = observations_by_protein.find(protein_chain_id);
        if (obs_it == observations_by_protein.end())
          continue;
        auto patches = build_contact_patches(obs_it->second);
        if (patches.empty())
          continue;
        patches_by_protein[protein_chain_id] = patches;
      }

      bridge_records.reserve(patches_by_protein.size());
      for (int protein_chain_id : partner_chain_ids) {
        const auto patch_it = patches_by_protein.find(protein_chain_id);
        if (patch_it == patches_by_protein.end())
          continue;
        const auto &patches = patch_it->second;
        if (patches.size() < 2)
          continue;
        BridgeRecord record;
        record.protein_chain_id = protein_chain_id;
        record.patch_count = static_cast<int>(patches.size());
        std::vector<uint8_t> patch_used(patches.size(), 0);
        for (std::size_t a = 0; a < patches.size(); ++a) {
          for (std::size_t b = a + 1; b < patches.size(); ++b) {
            const int residue_span = patch_residue_span(patches[a], patches[b]);
            if (residue_span < rna_compaction_longrange_min_)
              continue;
            const double geom_span_ang =
                patch_geom_span_angstrom(patches[a], patches[b]);
            if (geom_span_ang < rna_topology_bridge_geom_min_angstrom_)
              continue;
            ++record.bridge_count;
            patch_used[a] = 1;
            patch_used[b] = 1;
            record.bridge_residue_span_max =
                std::max(record.bridge_residue_span_max, residue_span);
            record.bridge_geom_span_max_ang =
                std::max(record.bridge_geom_span_max_ang, geom_span_ang);
            if (count_supported_longrange_pairs(patches[a], patches[b],
                                                longrange_pairs) > 0) {
              record.supports_longrange_contact = true;
            }
          }
        }
        if (record.bridge_count <= 0)
          continue;
        for (std::size_t idx = 0; idx < patches.size(); ++idx) {
          if (patch_used[idx] != 0)
            record.bridge_contact_count += patches[idx].contact_count;
        }
        if (have_prna_energy) {
          const auto energy_it = pair_energy_by_protein.find(protein_chain_id);
          if (energy_it != pair_energy_by_protein.end()) {
            record.bridge_attraction_kjmol =
                energy_it->second.elec + energy_it->second.vdw;
          }
        } else {
          record.bridge_attraction_kjmol =
              std::numeric_limits<double>::quiet_NaN();
        }
        bridge_records.push_back(record);
        bridge_residue_span_max =
            std::max(bridge_residue_span_max, record.bridge_residue_span_max);
        bridge_geom_span_max_ang =
            std::max(bridge_geom_span_max_ang, record.bridge_geom_span_max_ang);
      }

      clamp_records.reserve(partner_chain_ids.size());
      for (std::size_t a = 0; a < partner_chain_ids.size(); ++a) {
        const auto patch_it_a = patches_by_protein.find(partner_chain_ids[a]);
        if (patch_it_a == patches_by_protein.end())
          continue;
        for (std::size_t b = a + 1; b < partner_chain_ids.size(); ++b) {
          const auto patch_it_b = patches_by_protein.find(partner_chain_ids[b]);
          if (patch_it_b == patches_by_protein.end())
            continue;
          const auto &patches_a = patch_it_a->second;
          const auto &patches_b = patch_it_b->second;
          ClampRecord record;
          record.protein_chain_id_a = partner_chain_ids[a];
          record.protein_chain_id_b = partner_chain_ids[b];
          std::unordered_set<uint64_t> supported_pairs;
          for (const auto &patch_a : patches_a) {
            for (const auto &patch_b : patches_b) {
              const int residue_span = patch_residue_span(patch_a, patch_b);
              if (residue_span < rna_compaction_longrange_min_)
                continue;
              const double geom_span_ang =
                  patch_geom_span_angstrom(patch_a, patch_b);
              if (geom_span_ang < rna_topology_bridge_geom_min_angstrom_)
                continue;
              const int support_count =
                  count_supported_longrange_pairs(patch_a, patch_b, longrange_pairs);
              if (rna_topology_clamp_require_loop_ && support_count <= 0)
                continue;
              ++record.clamp_count;
              record.clamp_residue_span_max =
                  std::max(record.clamp_residue_span_max, residue_span);
              record.clamp_geom_span_max_ang =
                  std::max(record.clamp_geom_span_max_ang, geom_span_ang);
              for (const auto &pair : longrange_pairs) {
                const bool direct =
                    pair.first >= patch_a.min_res && pair.first <= patch_a.max_res &&
                    pair.second >= patch_b.min_res &&
                    pair.second <= patch_b.max_res;
                const bool reverse =
                    pair.first >= patch_b.min_res && pair.first <= patch_b.max_res &&
                    pair.second >= patch_a.min_res &&
                    pair.second <= patch_a.max_res;
                if (direct || reverse)
                  supported_pairs.insert(pack_u32_pair(pair.first, pair.second));
              }
            }
          }
          if (record.clamp_count <= 0)
            continue;
          record.clamp_supported_longrange_contacts =
              static_cast<int>(supported_pairs.size());
          if (have_prna_energy) {
            const auto energy_a = pair_energy_by_protein.find(record.protein_chain_id_a);
            const auto energy_b = pair_energy_by_protein.find(record.protein_chain_id_b);
            if (energy_a != pair_energy_by_protein.end())
              record.clamp_attraction_kjmol +=
                  energy_a->second.elec + energy_a->second.vdw;
            if (energy_b != pair_energy_by_protein.end())
              record.clamp_attraction_kjmol +=
                  energy_b->second.elec + energy_b->second.vdw;
          } else {
            record.clamp_attraction_kjmol =
                std::numeric_limits<double>::quiet_NaN();
          }
          clamp_supported_longrange_contacts +=
              record.clamp_supported_longrange_contacts;
          clamp_records.push_back(record);
        }
      }

      any_bridge = !bridge_records.empty();
      num_bridge_proteins = static_cast<int>(bridge_records.size());
      any_clamp = !clamp_records.empty();
      num_clamp_pairs = static_cast<int>(clamp_records.size());
      topology_state = (bound_protein_chains <= 0)
                           ? 0
                           : (any_clamp ? 3 : (any_bridge ? 2 : 1));
      if (have_prna_energy) {
        prna_attraction_strength_kjmol = attraction_strength(prna_energy.total());
        bridge_total_kjmol = 0.0;
        for (const auto &record : bridge_records)
          if (std::isfinite(record.bridge_attraction_kjmol))
            bridge_total_kjmol += record.bridge_attraction_kjmol;
        nonbridge_total_kjmol = prna_energy.total() - bridge_total_kjmol;
        bridge_attraction_strength_kjmol =
            attraction_strength(bridge_total_kjmol);
        nonbridge_attraction_strength_kjmol =
            attraction_strength(nonbridge_total_kjmol);
      }
      count_local_pp_contacts(partner_chain_ids, local_pp_sticker_contacts,
                              local_pp_energy);
      if (energy_provider_ != nullptr && energy_provider_->ready()) {
        local_pp_total_kjmol = local_pp_energy.total();
        local_pp_attraction_strength_kjmol =
            attraction_strength(local_pp_total_kjmol);
      }
      if (protein_rna_bead_contacts > 0) {
        local_pp_to_prna_contact_ratio =
            static_cast<double>(local_pp_sticker_contacts) /
            static_cast<double>(protein_rna_bead_contacts);
      }
      if (prna_attraction_strength_kjmol > 1e-12 &&
          std::isfinite(local_pp_attraction_strength_kjmol)) {
        local_pp_to_prna_attraction_ratio =
            local_pp_attraction_strength_kjmol / prna_attraction_strength_kjmol;
      }
      if (bridge_attraction_strength_kjmol > 1e-12 &&
          std::isfinite(local_pp_attraction_strength_kjmol)) {
        local_pp_to_bridge_attraction_ratio =
            local_pp_attraction_strength_kjmol /
            bridge_attraction_strength_kjmol;
      }
    }

    bool inside_dense = false;
    int cluster_id = -1;
    const auto dense_hit = dense_lookup.find(rna_chain_id);
    if (dense_hit != dense_lookup.end()) {
      inside_dense = dense_hit->second.first;
      cluster_id = dense_hit->second.second;
    }

    RNACompactionSample sample;
    sample.time_ns = time_ns;
    sample.frame = frame_idx;
    sample.rna_chain_id = rna_chain_id;
    sample.cluster_id = cluster_id;
    sample.partner_state_id = register_rna_partner_state(partner_chain_ids);
    sample.bound_protein_chains = static_cast<double>(bound_protein_chains);
    sample.protein_rna_bead_contacts =
        static_cast<double>(protein_rna_bead_contacts);
    sample.rg_nm = rg_nm;
    sample.end_to_end_nm = end_to_end_nm;
    sample.relative_shape_anisotropy = relative_shape_anisotropy;
    sample.longrange_contact_fraction = longrange_fraction;
    sample.intrachain_contacts = static_cast<double>(intrachain_contacts);
    sample.longrange_intrachain_contacts =
        static_cast<double>(longrange_intrachain_contacts);
    sample.energy_elec_kjmol =
        have_prna_energy ? prna_energy.elec
                         : std::numeric_limits<double>::quiet_NaN();
    sample.energy_vdw_kjmol =
        have_prna_energy ? prna_energy.vdw
                         : std::numeric_limits<double>::quiet_NaN();
    sample.energy_total_kjmol =
        have_prna_energy ? prna_energy.total()
                         : std::numeric_limits<double>::quiet_NaN();
    sample.topology_state = topology_state;
    sample.any_bridge = any_bridge;
    sample.num_bridge_proteins = num_bridge_proteins;
    sample.any_clamp = any_clamp;
    sample.num_clamp_pairs = num_clamp_pairs;
    sample.bridge_residue_span_max = static_cast<double>(bridge_residue_span_max);
    sample.bridge_geom_span_max_nm = 0.1 * bridge_geom_span_max_ang;
    sample.clamp_supported_longrange_contacts =
        static_cast<double>(clamp_supported_longrange_contacts);
    sample.bridge_attraction_kjmol = bridge_total_kjmol;
    sample.nonbridge_attraction_kjmol = nonbridge_total_kjmol;
    sample.local_pp_elec_kjmol =
        std::isfinite(local_pp_total_kjmol) ? local_pp_energy.elec
                                            : std::numeric_limits<double>::quiet_NaN();
    sample.local_pp_vdw_kjmol =
        std::isfinite(local_pp_total_kjmol) ? local_pp_energy.vdw
                                            : std::numeric_limits<double>::quiet_NaN();
    sample.local_pp_total_kjmol = local_pp_total_kjmol;
    sample.local_pp_attraction_strength_kjmol =
        local_pp_attraction_strength_kjmol;
    sample.prna_attraction_strength_kjmol = prna_attraction_strength_kjmol;
    sample.bridge_attraction_strength_kjmol =
        bridge_attraction_strength_kjmol;
    sample.nonbridge_attraction_strength_kjmol =
        nonbridge_attraction_strength_kjmol;
    sample.local_pp_sticker_contacts =
        static_cast<double>(local_pp_sticker_contacts);
    sample.local_pp_to_prna_contact_ratio = local_pp_to_prna_contact_ratio;
    sample.local_pp_to_prna_attraction_ratio =
        local_pp_to_prna_attraction_ratio;
    sample.local_pp_to_bridge_attraction_ratio =
        local_pp_to_bridge_attraction_ratio;
    sample.inside_dense = inside_dense;
    rna_compaction_samples_.push_back(sample);

    auto &chain_stats = rna_chain_compaction_stats_[rna_chain_id];
    chain_stats.inside_dense.add(inside_dense ? 1.0 : 0.0);
    chain_stats.bound_protein_chains.add(sample.bound_protein_chains);
    chain_stats.protein_rna_bead_contacts.add(sample.protein_rna_bead_contacts);
    chain_stats.rg_nm.add(rg_nm);
    chain_stats.end_to_end_nm.add(end_to_end_nm);
    chain_stats.relative_shape_anisotropy.add(relative_shape_anisotropy);
    chain_stats.longrange_contact_fraction.add(longrange_fraction);
    chain_stats.intrachain_contacts.add(static_cast<double>(intrachain_contacts));
    chain_stats.longrange_intrachain_contacts.add(
        static_cast<double>(longrange_intrachain_contacts));

    auto &bin_stats = rna_binding_bin_stats_[bound_protein_chains];
    bin_stats.rg_nm.add(rg_nm);
    bin_stats.end_to_end_nm.add(end_to_end_nm);
    bin_stats.asphericity_nm2.add(asphericity_nm2);
    bin_stats.longrange_contact_fraction.add(longrange_fraction);

    if (energy_correlation_enabled_) {
      auto &energy_bin = rna_energy_bin_stats_[bound_protein_chains];
      energy_bin.energy_elec_kjmol.add(prna_energy.elec);
      energy_bin.energy_vdw_kjmol.add(prna_energy.vdw);
      energy_bin.energy_total_kjmol.add(prna_energy.total());
      energy_bin.attraction_strength_kjmol.add(-prna_energy.total());
      energy_bin.rg_nm.add(rg_nm);
      energy_bin.end_to_end_nm.add(end_to_end_nm);
      energy_bin.relative_shape_anisotropy.add(relative_shape_anisotropy);
      energy_bin.longrange_contact_fraction.add(longrange_fraction);

      if (rna_binding_energy_trace_out_.is_open()) {
        rna_binding_energy_trace_out_ << std::fixed << std::setprecision(6)
                                      << time_ns << "," << frame_idx << ","
                                      << rna_chain_id << ","
                                      << (inside_dense ? 1 : 0) << ","
                                      << cluster_id << ","
                                      << bound_protein_chains << ","
                                      << protein_rna_bead_contacts << ","
                                      << rg_nm << "," << end_to_end_nm << ","
                                      << relative_shape_anisotropy << ","
                                      << longrange_fraction << ","
                                      << prna_energy.elec << ","
                                      << prna_energy.vdw << ","
                                      << prna_energy.total() << "\n";
      }

      if (energy_correlation_detailed_ &&
          prna_pair_energy_trace_out_.is_open()) {
        std::vector<int> protein_ids;
        protein_ids.reserve(pair_energy_by_protein.size());
        for (const auto &entry : pair_energy_by_protein)
          protein_ids.push_back(entry.first);
        std::sort(protein_ids.begin(), protein_ids.end());
        for (int protein_chain_id : protein_ids) {
          const auto &pair_acc = pair_energy_by_protein.at(protein_chain_id);
          if (pair_acc.contact_count <= 0)
            continue;
          prna_pair_energy_trace_out_
              << std::fixed << std::setprecision(6) << time_ns << ","
              << frame_idx << "," << protein_chain_id << "," << rna_chain_id
              << "," << pair_acc.contact_count << "," << pair_acc.elec << ","
              << pair_acc.vdw << "," << (pair_acc.elec + pair_acc.vdw) << ","
              << rg_nm << "," << longrange_fraction << "\n";
        }
      }
    }

    if (rna_topology_enabled_ && rna_topology_detailed_) {
      if (rna_topology_trace_out_.is_open()) {
        rna_topology_trace_out_
            << std::fixed << std::setprecision(6) << time_ns << ","
            << frame_idx << "," << rna_chain_id << ","
            << (inside_dense ? 1 : 0) << "," << cluster_id << ","
            << bound_protein_chains << "," << topology_state << ","
            << (any_bridge ? 1 : 0) << "," << num_bridge_proteins << ","
            << (any_clamp ? 1 : 0) << "," << num_clamp_pairs << ","
            << bridge_residue_span_max << "," << (0.1 * bridge_geom_span_max_ang)
            << "," << longrange_intrachain_contacts << ","
            << clamp_supported_longrange_contacts << ","
            << format_optional_csv(bridge_total_kjmol) << ","
            << format_optional_csv(nonbridge_total_kjmol) << ","
            << format_optional_csv(bridge_attraction_strength_kjmol) << ","
            << format_optional_csv(nonbridge_attraction_strength_kjmol) << ","
            << format_optional_csv(prna_attraction_strength_kjmol) << ","
            << local_pp_sticker_contacts << ","
            << format_optional_csv(sample.local_pp_elec_kjmol) << ","
            << format_optional_csv(sample.local_pp_vdw_kjmol) << ","
            << format_optional_csv(local_pp_total_kjmol) << ","
            << format_optional_csv(local_pp_attraction_strength_kjmol) << ","
            << format_optional_csv(local_pp_to_prna_contact_ratio) << ","
            << format_optional_csv(local_pp_to_prna_attraction_ratio) << ","
            << format_optional_csv(local_pp_to_bridge_attraction_ratio) << "\n";
      }
      if (rna_bridge_trace_out_.is_open()) {
        for (const auto &record : bridge_records) {
          rna_bridge_trace_out_
              << std::fixed << std::setprecision(6) << time_ns << ","
              << frame_idx << "," << rna_chain_id << ","
              << record.protein_chain_id << "," << record.patch_count << ","
              << record.bridge_count << "," << record.bridge_residue_span_max
              << "," << (0.1 * record.bridge_geom_span_max_ang) << ","
              << record.bridge_contact_count << ","
              << format_optional_csv(record.bridge_attraction_kjmol) << ","
              << format_optional_csv(
                     attraction_strength(record.bridge_attraction_kjmol))
              << ","
              << (record.supports_longrange_contact ? 1 : 0) << "\n";
        }
      }
      if (rna_clamp_trace_out_.is_open()) {
        for (const auto &record : clamp_records) {
          rna_clamp_trace_out_
              << std::fixed << std::setprecision(6) << time_ns << ","
              << frame_idx << "," << rna_chain_id << ","
              << record.protein_chain_id_a << ","
              << record.protein_chain_id_b << "," << record.clamp_count << ","
              << record.clamp_residue_span_max << ","
              << (0.1 * record.clamp_geom_span_max_ang) << ","
              << record.clamp_supported_longrange_contacts << ","
              << format_optional_csv(record.clamp_attraction_kjmol) << ","
              << format_optional_csv(
                     attraction_strength(record.clamp_attraction_kjmol))
              << "\n";
        }
      }
    }

    if (rna_compaction_detailed_) {
      if (rna_compaction_trace_streaming_ &&
          rna_compaction_trace_out_.is_open()) {
        rna_compaction_trace_out_
            << time_ns << "," << frame_idx << "," << rna_chain_id << ","
            << (inside_dense ? 1 : 0) << "," << cluster_id << ","
            << bound_protein_chains << "," << protein_rna_bead_contacts << ","
            << rg_nm << "," << end_to_end_nm << "," << asphericity_nm2 << ","
            << normalized_asphericity << "," << acylindricity_nm2 << ","
            << relative_shape_anisotropy << "," << aspect_ratio << ","
            << intrachain_contacts << "," << longrange_intrachain_contacts
            << "," << longrange_fraction << "\n";
      } else {
        rna_compaction_trace_rows_.push_back(
            {time_ns,
             frame_idx,
             rna_chain_id,
             inside_dense ? 1 : 0,
             cluster_id,
             bound_protein_chains,
             protein_rna_bead_contacts,
             rg_nm,
             end_to_end_nm,
             asphericity_nm2,
             normalized_asphericity,
             acylindricity_nm2,
             relative_shape_anisotropy,
             aspect_ratio,
             intrachain_contacts,
             longrange_intrachain_contacts,
             longrange_fraction});
      }
    }
  }
}

void TitanLLPSCluster::export_rna_compaction_outputs(
    const info &info_output) const {
  if (!rna_compaction_enabled_)
    return;

  auto format_value = [](double value) {
    if (!std::isfinite(value))
      return std::string("NA");
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
  };

  {
    std::ofstream out(
        info_output.get_llps_rna_compaction_binding_summary_filename());
    if (out.is_open()) {
      out << "# CG RNA chain compaction/conformation vs protein binding.\n";
      out << "# This is not secondary-structure folding; metrics are computed "
             "from coarse-grained chain geometry.\n";
      out << "samples_all: " << rna_compaction_samples_.size() << "\n";
      uint64_t inside_count = 0;
      for (const auto &sample : rna_compaction_samples_)
        if (sample.inside_dense)
          ++inside_count;
      out << "samples_inside_dense: " << inside_count << "\n";

      auto emit_correlations = [&](const std::string &prefix,
                                   bool inside_dense_only) {
        auto emit_one = [&](const std::string &x_name,
                            const std::string &y_name, auto x_get,
                            auto y_get) {
          std::vector<double> x;
          std::vector<double> y;
          x.reserve(rna_compaction_samples_.size());
          y.reserve(rna_compaction_samples_.size());
          for (const auto &sample : rna_compaction_samples_) {
            if (inside_dense_only && !sample.inside_dense)
              continue;
            x.push_back(x_get(sample));
            y.push_back(y_get(sample));
          }
          out << prefix << "_pearson_" << x_name << "_vs_" << y_name << ": "
              << format_value(pearson_corr(x, y)) << "\n";
          out << prefix << "_spearman_" << x_name << "_vs_" << y_name << ": "
              << format_value(spearman_corr(x, y)) << "\n";
        };

        auto bound = [](const RNACompactionSample &s) {
          return s.bound_protein_chains;
        };
        auto bead_contacts = [](const RNACompactionSample &s) {
          return s.protein_rna_bead_contacts;
        };
        auto rg = [](const RNACompactionSample &s) { return s.rg_nm; };
        auto e2e = [](const RNACompactionSample &s) {
          return s.end_to_end_nm;
        };
        auto anisotropy = [](const RNACompactionSample &s) {
          return s.relative_shape_anisotropy;
        };
        auto longrange_fraction = [](const RNACompactionSample &s) {
          return s.longrange_contact_fraction;
        };

        emit_one("BoundProteinChains", "Rg_nm", bound, rg);
        emit_one("BoundProteinChains", "EndToEnd_nm", bound, e2e);
        emit_one("BoundProteinChains", "RelativeShapeAnisotropy", bound,
                 anisotropy);
        emit_one("BoundProteinChains", "LongRangeContactFraction", bound,
                 longrange_fraction);
        emit_one("ProteinRNABeadContacts", "Rg_nm", bead_contacts, rg);
        emit_one("ProteinRNABeadContacts", "EndToEnd_nm", bead_contacts, e2e);
        emit_one("ProteinRNABeadContacts", "RelativeShapeAnisotropy",
                 bead_contacts, anisotropy);
        emit_one("ProteinRNABeadContacts", "LongRangeContactFraction",
                 bead_contacts, longrange_fraction);
      };

      emit_correlations("all_frames", false);
      emit_correlations("inside_dense_only", true);
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_compaction_binding_summary_filename());
    }
  }

  if (energy_correlation_enabled_) {
    {
      std::ofstream out(
          info_output.get_llps_rna_energy_compaction_summary_filename());
      if (out.is_open()) {
        out << "# Protein-RNA interaction energy decomposition vs CG RNA "
               "compaction.\n";
        out << "# Raw sign is preserved: Etotal < 0 means a more favorable "
               "interaction energy. This is not binding free energy.\n";
        out << "samples_all: " << rna_compaction_samples_.size() << "\n";
        uint64_t inside_count = 0;
        uint64_t contacting_count = 0;
        for (const auto &sample : rna_compaction_samples_) {
          if (sample.inside_dense)
            ++inside_count;
          if (sample.bound_protein_chains > 0.0)
            ++contacting_count;
        }
        out << "samples_inside_dense: " << inside_count << "\n";
        out << "samples_contacting: " << contacting_count << "\n";

        auto emit_scope = [&](const std::string &prefix, auto keep_sample) {
          auto emit_one = [&](const std::string &energy_name,
                              const std::string &metric_name, auto energy_get,
                              auto metric_get) {
            std::vector<double> x;
            std::vector<double> y;
            x.reserve(rna_compaction_samples_.size());
            y.reserve(rna_compaction_samples_.size());
            for (const auto &sample : rna_compaction_samples_) {
              if (!keep_sample(sample))
                continue;
              x.push_back(energy_get(sample));
              y.push_back(metric_get(sample));
            }
            out << prefix << "_pearson_" << energy_name << "_vs_"
                << metric_name << ": " << format_value(pearson_corr(x, y))
                << "\n";
            out << prefix << "_spearman_" << energy_name << "_vs_"
                << metric_name << ": " << format_value(spearman_corr(x, y))
                << "\n";
          };

          auto e_elec = [](const RNACompactionSample &s) {
            return s.energy_elec_kjmol;
          };
          auto e_vdw = [](const RNACompactionSample &s) {
            return s.energy_vdw_kjmol;
          };
          auto e_total = [](const RNACompactionSample &s) {
            return s.energy_total_kjmol;
          };
          auto attraction = [](const RNACompactionSample &s) {
            return -s.energy_total_kjmol;
          };
          auto rg = [](const RNACompactionSample &s) { return s.rg_nm; };
          auto e2e = [](const RNACompactionSample &s) {
            return s.end_to_end_nm;
          };
          auto anisotropy = [](const RNACompactionSample &s) {
            return s.relative_shape_anisotropy;
          };
          auto longrange_fraction = [](const RNACompactionSample &s) {
            return s.longrange_contact_fraction;
          };

          emit_one("Eelec_kJmol", "Rg_nm", e_elec, rg);
          emit_one("Eelec_kJmol", "EndToEnd_nm", e_elec, e2e);
          emit_one("Eelec_kJmol", "RelativeShapeAnisotropy", e_elec,
                   anisotropy);
          emit_one("Eelec_kJmol", "LongRangeContactFraction", e_elec,
                   longrange_fraction);
          emit_one("Evdw_kJmol", "Rg_nm", e_vdw, rg);
          emit_one("Evdw_kJmol", "EndToEnd_nm", e_vdw, e2e);
          emit_one("Evdw_kJmol", "RelativeShapeAnisotropy", e_vdw,
                   anisotropy);
          emit_one("Evdw_kJmol", "LongRangeContactFraction", e_vdw,
                   longrange_fraction);
          emit_one("Etotal_kJmol", "Rg_nm", e_total, rg);
          emit_one("Etotal_kJmol", "EndToEnd_nm", e_total, e2e);
          emit_one("Etotal_kJmol", "RelativeShapeAnisotropy", e_total,
                   anisotropy);
          emit_one("Etotal_kJmol", "LongRangeContactFraction", e_total,
                   longrange_fraction);
          emit_one("AttractionStrength_kJmol", "Rg_nm", attraction, rg);
          emit_one("AttractionStrength_kJmol", "EndToEnd_nm", attraction,
                   e2e);
          emit_one("AttractionStrength_kJmol", "RelativeShapeAnisotropy",
                   attraction, anisotropy);
          emit_one("AttractionStrength_kJmol", "LongRangeContactFraction",
                   attraction, longrange_fraction);
        };

        emit_scope("all_frames",
                   [](const RNACompactionSample &) { return true; });
        emit_scope("inside_dense_only", [](const RNACompactionSample &s) {
          return s.inside_dense;
        });
        emit_scope("contacting_only", [](const RNACompactionSample &s) {
          return s.bound_protein_chains > 0.0;
        });
        TitanUI::print_info(
            "  Written: " +
            info_output.get_llps_rna_energy_compaction_summary_filename());
      }
    }

    {
      std::ofstream out(
          info_output.get_llps_rna_energy_compaction_binned_filename());
      if (out.is_open()) {
        out << "BoundProteinChainsBin,FrameCount,EelecMean_kJmol,"
               "EelecStd_kJmol,EvdwMean_kJmol,EvdwStd_kJmol,"
               "EtotalMean_kJmol,EtotalStd_kJmol,"
               "AttractionStrengthMean_kJmol,AttractionStrengthStd_kJmol,"
               "RgMean_nm,RgStd_nm,EndToEndMean_nm,EndToEndStd_nm,"
               "RelativeShapeAnisotropyMean,RelativeShapeAnisotropyStd,"
               "LongRangeContactFractionMean,LongRangeContactFractionStd\n";
        std::vector<int> bins;
        bins.reserve(rna_energy_bin_stats_.size());
        for (const auto &entry : rna_energy_bin_stats_)
          bins.push_back(entry.first);
        std::sort(bins.begin(), bins.end());
        out << std::fixed << std::setprecision(6);
        for (int bin : bins) {
          const auto &stats = rna_energy_bin_stats_.at(bin);
          out << bin << "," << stats.energy_total_kjmol.n << ","
              << stats.energy_elec_kjmol.mean() << ","
              << stats.energy_elec_kjmol.stddev() << ","
              << stats.energy_vdw_kjmol.mean() << ","
              << stats.energy_vdw_kjmol.stddev() << ","
              << stats.energy_total_kjmol.mean() << ","
              << stats.energy_total_kjmol.stddev() << ","
              << stats.attraction_strength_kjmol.mean() << ","
              << stats.attraction_strength_kjmol.stddev() << ","
              << stats.rg_nm.mean() << "," << stats.rg_nm.stddev() << ","
              << stats.end_to_end_nm.mean() << ","
              << stats.end_to_end_nm.stddev() << ","
              << stats.relative_shape_anisotropy.mean() << ","
              << stats.relative_shape_anisotropy.stddev() << ","
              << stats.longrange_contact_fraction.mean() << ","
              << stats.longrange_contact_fraction.stddev() << "\n";
        }
        TitanUI::print_info(
            "  Written: " +
            info_output.get_llps_rna_energy_compaction_binned_filename());
      }
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_compaction_binding_binned_filename());
    if (out.is_open()) {
      out << "BoundProteinChainsBin,FrameCount,RgMean_nm,RgStd_nm,"
             "EndToEndMean_nm,EndToEndStd_nm,AsphericityMean,"
             "AsphericityStd,LongRangeContactFractionMean,"
             "LongRangeContactFractionStd\n";
      std::vector<int> bins;
      bins.reserve(rna_binding_bin_stats_.size());
      for (const auto &entry : rna_binding_bin_stats_)
        bins.push_back(entry.first);
      std::sort(bins.begin(), bins.end());
      out << std::fixed << std::setprecision(6);
      for (int bin : bins) {
        const auto &stats = rna_binding_bin_stats_.at(bin);
        out << bin << "," << stats.rg_nm.n << "," << stats.rg_nm.mean() << ","
            << stats.rg_nm.stddev() << "," << stats.end_to_end_nm.mean()
            << "," << stats.end_to_end_nm.stddev() << ","
            << stats.asphericity_nm2.mean() << ","
            << stats.asphericity_nm2.stddev() << ","
            << stats.longrange_contact_fraction.mean() << ","
            << stats.longrange_contact_fraction.stddev() << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_compaction_binding_binned_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_chain_compaction_summary_filename());
    if (out.is_open()) {
      out << "RNAChainID,FrameCount,InsideDenseFraction,"
             "MeanBoundProteinChains,MeanProteinRNABeadContacts,MeanRg_nm,"
             "StdRg_nm,MeanEndToEnd_nm,StdEndToEnd_nm,"
             "MeanRelativeShapeAnisotropy,MeanLongRangeContactFraction\n";
      std::vector<int> chains;
      chains.reserve(rna_chain_compaction_stats_.size());
      for (const auto &entry : rna_chain_compaction_stats_)
        chains.push_back(entry.first);
      std::sort(chains.begin(), chains.end());
      out << std::fixed << std::setprecision(6);
      for (int chain : chains) {
        const auto &stats = rna_chain_compaction_stats_.at(chain);
        out << chain << "," << stats.rg_nm.n << ","
            << stats.inside_dense.mean() << ","
            << stats.bound_protein_chains.mean() << ","
            << stats.protein_rna_bead_contacts.mean() << ","
            << stats.rg_nm.mean() << "," << stats.rg_nm.stddev() << ","
            << stats.end_to_end_nm.mean() << ","
            << stats.end_to_end_nm.stddev() << ","
            << stats.relative_shape_anisotropy.mean() << ","
            << stats.longrange_contact_fraction.mean() << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_chain_compaction_summary_filename());
    }
  }

  {
    std::ofstream out(
        info_output.get_llps_rna_intrachain_contacts_summary_filename());
    if (out.is_open()) {
      out << "RNAChainID,FrameCount,MeanIntrachainContacts,"
             "StdIntrachainContacts,MeanLongRangeIntrachainContacts,"
             "StdLongRangeIntrachainContacts,MeanLongRangeContactFraction,"
             "StdLongRangeContactFraction\n";
      std::vector<int> chains;
      chains.reserve(rna_chain_compaction_stats_.size());
      for (const auto &entry : rna_chain_compaction_stats_)
        chains.push_back(entry.first);
      std::sort(chains.begin(), chains.end());
      out << std::fixed << std::setprecision(6);
      for (int chain : chains) {
        const auto &stats = rna_chain_compaction_stats_.at(chain);
        out << chain << "," << stats.intrachain_contacts.n << ","
            << stats.intrachain_contacts.mean() << ","
            << stats.intrachain_contacts.stddev() << ","
            << stats.longrange_intrachain_contacts.mean() << ","
            << stats.longrange_intrachain_contacts.stddev() << ","
            << stats.longrange_contact_fraction.mean() << ","
            << stats.longrange_contact_fraction.stddev() << "\n";
      }
      TitanUI::print_info(
          "  Written: " +
          info_output.get_llps_rna_intrachain_contacts_summary_filename());
    }
  }

  export_rna_information_outputs(info_output);
  export_rna_topology_outputs(info_output);

  if (rna_compaction_detailed_) {
    if (rna_compaction_trace_streaming_) {
      if (rna_compaction_trace_out_.is_open())
        const_cast<std::ofstream &>(rna_compaction_trace_out_).flush();
      TitanUI::print_info(
          "  Written: " + info_output.get_llps_rna_compaction_trace_filename());
    } else {
    std::ofstream out(info_output.get_llps_rna_compaction_trace_filename());
    if (out.is_open()) {
      out << "Time_ns,Frame,RNAChainID,InsideDense,ClusterID,"
             "BoundProteinChains,ProteinRNABeadContacts,Rg_nm,EndToEnd_nm,"
             "Asphericity_nm2,NormalizedAsphericity,Acylindricity_nm2,"
             "RelativeShapeAnisotropy,AspectRatio,IntrachainContacts,"
             "LongRangeIntrachainContacts,LongRangeContactFraction\n";
      out << std::fixed << std::setprecision(6);
      for (const auto &row : rna_compaction_trace_rows_) {
        out << row.time_ns << "," << row.frame << "," << row.rna_chain_id
            << "," << row.inside_dense << "," << row.cluster_id << ","
            << row.bound_protein_chains << ","
            << row.protein_rna_bead_contacts << "," << row.rg_nm << ","
            << row.end_to_end_nm << "," << row.asphericity_nm2 << ","
            << row.normalized_asphericity << "," << row.acylindricity_nm2
            << "," << row.relative_shape_anisotropy << ","
            << row.aspect_ratio << "," << row.intrachain_contacts << ","
            << row.longrange_intrachain_contacts << ","
            << row.longrange_contact_fraction << "\n";
      }
      TitanUI::print_info(
          "  Written: " + info_output.get_llps_rna_compaction_trace_filename());
    }
    }
  }
}

// Split from titan_llps_cluster.cpp. Keep TITAN_LLPS public behavior unchanged.
#include "titan_llps_cluster.h"
#include "titan_exact_periodic_box.h"
#include "MOL.h"
#include "MOL_info.h"
#include "box_type.h"
#include "titan_executor.h"
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

std::vector<LLPSNode> TitanLLPSCluster::extract_nodes(
    const MOL &pro, const std::vector<mol_info> &pro_info, const MOL &lig,
    const std::vector<mol_info> &lig_info) const {
  std::vector<LLPSNode> nodes;
  nodes.reserve(pro.pos.x.size() + lig.pos.x.size());
  int nid = 1;

  auto add_mol = [&](const MOL &mol, const std::vector<mol_info> &info,
                     int mol_type) {
    if (mol.pos.x.empty())
      return;
    const std::size_t n = std::min(
        {mol.pos.x.size(), mol.pos.y.size(), mol.pos.z.size(), info.size()});
    for (std::size_t i = 0; i < n; ++i) {
      const std::string &atom = info[i].get_atom();
      // Liquidity v3: "P" is a nucleic-acid backbone name only. In the
      // Martini phosphoserine used here the side-chain phosphate bead of
      // SER176/180/184/188 is also called "P"; accepting it for the protein
      // made every phos chain carry 4 side-chain beads as extra "backbone"
      // nodes (423 vs 419), i.e. a different unit definition for WT and phos.
      const bool protein_backbone = (mol_type == 1) && (atom == "BB" || atom == "CA");
      const bool nucleic_backbone =
          (mol_type != 1) && (atom == "BB" || atom == "BB1" || atom == "BB2" ||
                              atom == "BB3" || atom == "P");
      if (protein_backbone || nucleic_backbone) {
        nodes.push_back({nid++, mol_type,
                         static_cast<int>(info[i].get_chainnum()), mol.pos.x[i],
                         mol.pos.y[i], mol.pos.z[i], 0, false});
      }
    }
  };

  add_mol(pro, pro_info, 1);
  add_mol(lig, lig_info, 2);
  return nodes;
}

void TitanLLPSCluster::run_dbscan(TitanExecutor &executor,
                                  std::vector<LLPSNode> &nodes,
                                  const box &Box) {
  if (nodes.empty())
    return;
  if(!std::isfinite(eps_) || eps_<=0 || min_pts_<1)
    throw std::invalid_argument("DBSCAN requires a positive finite radius and min_pts >= 1");
  for(const auto& node:nodes)if(!std::isfinite(node.x)||!std::isfinite(node.y)||!std::isfinite(node.z))
    throw std::invalid_argument("DBSCAN coordinates must be finite");
  const TitanExactPeriodicBox bmat(Box);
  const double bx=bmat.heights[0],by=bmat.heights[1],bz=bmat.heights[2];
  // Liquidity v3 (performance only): for an orthorhombic cell the nearest
  // image is exactly the per-axis rounding, so skip the general lattice
  // solver, which made this O(pairs) loop cost ~2.4 s per frame for 1.1e4
  // nodes. Triclinic cells still use the exact solver.
  const bool ortho_cell =
      std::abs(bmat.m[1][0]) <= 1e-9 * bmat.m[0][0] &&
      std::abs(bmat.m[2][0]) <= 1e-9 * bmat.m[0][0] &&
      std::abs(bmat.m[2][1]) <= 1e-9 * bmat.m[1][1];
  const double lx_o = bmat.m[0][0], ly_o = bmat.m[1][1], lz_o = bmat.m[2][2];
  const auto fast_mic = [&](double &dx, double &dy, double &dz) {
    if (ortho_cell) {
      dx -= lx_o * std::nearbyint(dx / lx_o);
      dy -= ly_o * std::nearbyint(dy / ly_o);
      dz -= lz_o * std::nearbyint(dz / lz_o);
    } else {
      bmat.apply_mic(dx, dy, dz);
    }
  };
  // Larger candidate cells are conservative. Bound the dense grid independently
  // of a nearly empty but very large simulation cell.
  const int limit=std::clamp(int(std::ceil(std::cbrt(double(nodes.size())*8))),1,256);
  const auto count=[&](double height){return int(std::clamp(std::floor(height/eps_),1.0,double(limit)));};
  const int nx=count(bx),ny=count(by),nz=count(bz);
  const double cx = bx / nx;
  const double cy = by / ny;
  const double cz = bz / nz;
  const int ncells = nx * ny * nz;

  head_.assign(static_cast<std::size_t>(ncells), -1);
  nxt_.assign(nodes.size(), -1);
  cell_id_.assign(nodes.size(), -1);
  neighbor_count_.assign(nodes.size(), 0);

  auto cell_of = [&](double x, double y, double z) {
    bmat.grid_coordinate(x,y,z);
    int ix = static_cast<int>(std::floor(x / cx));
    int iy = static_cast<int>(std::floor(y / cy));
    int iz = static_cast<int>(std::floor(z / cz));
    ix = ((ix % nx) + nx) % nx;
    iy = ((iy % ny) + ny) % ny;
    iz = ((iz % nz) + nz) % nz;
    return ix * ny * nz + iy * nz + iz;
  };

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const int cell = cell_of(nodes[i].x, nodes[i].y, nodes[i].z);
    cell_id_[i] = cell;
    nxt_[i] = head_[static_cast<std::size_t>(cell)];
    head_[static_cast<std::size_t>(cell)] = static_cast<int>(i);
    nodes[i].cluster_id = 0;
    nodes[i].is_core = false;
  }

  const auto enumerate_neighbors = [&](int node_index, auto &&fn) {
    const int cell = cell_id_[static_cast<std::size_t>(node_index)];
    const int ix = cell / (ny * nz);
    const int rem = cell % (ny * nz);
    const int iy = rem / nz;
    const int iz = rem % nz;
    int checked_cells[27];
    int checked = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      const int cx2 = ((ix + dx) % nx + nx) % nx;
      for (int dy = -1; dy <= 1; ++dy) {
        const int cy2 = ((iy + dy) % ny + ny) % ny;
        for (int dz = -1; dz <= 1; ++dz) {
          const int cz2 = ((iz + dz) % nz + nz) % nz;
          const int cell = cx2 * ny * nz + cy2 * nz + cz2;
          bool dup = false;
          for (int k = 0; k < checked; ++k) {
            if (checked_cells[k] == cell) {
              dup = true;
              break;
            }
          }
          if (!dup)
            checked_cells[checked++] = cell;
        }
      }
    }

    for (int k = 0; k < checked; ++k) {
      for (int j = head_[static_cast<std::size_t>(checked_cells[k])]; j != -1;
            j = nxt_[static_cast<std::size_t>(j)]) {
        if (j == node_index)
          continue;
        double dx = nodes[static_cast<std::size_t>(node_index)].x -
                    nodes[static_cast<std::size_t>(j)].x;
        double dy = nodes[static_cast<std::size_t>(node_index)].y -
                    nodes[static_cast<std::size_t>(j)].y;
        double dz = nodes[static_cast<std::size_t>(node_index)].z -
                    nodes[static_cast<std::size_t>(j)].z;
        fast_mic(dx, dy, dz);
        if (dx * dx + dy * dy + dz * dz <= eps_sq_)
          fn(j);
      }
    }
  };

  const std::size_t min_grain = 64;
  const std::size_t workers = std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), nodes.size()));
  executor.parallel_for(nodes.size(), workers, min_grain,
                        [&](std::size_t /*worker*/, std::size_t begin,
                            std::size_t end) {
                          for (std::size_t i = begin; i < end; ++i) {
                            int count = 0;
                            enumerate_neighbors(static_cast<int>(i),
                                                [&](int) { ++count; });
                            neighbor_count_[i] = count;
                            nodes[i].is_core =
                                (count + 1 >= static_cast<int>(min_pts_));
                          }
                        });
  perf_.dbscan_neighbor_checks += static_cast<uint64_t>(
      std::accumulate(neighbor_count_.begin(), neighbor_count_.end(), 0ULL));
  perf_.process_executor_workers =
      std::max<std::size_t>(perf_.process_executor_workers, workers);

  int cur_cluster = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].cluster_id != 0 || !nodes[i].is_core)
      continue;
    ++cur_cluster;
    nodes[i].cluster_id = cur_cluster;
    bfs_q_.clear();
    bfs_q_.push_back(static_cast<int>(i));
    for (std::size_t qi = 0; qi < bfs_q_.size(); ++qi) {
      const int curr = bfs_q_[qi];
      ++perf_.dbscan_cluster_expansions;
      enumerate_neighbors(curr, [&](int nb) {
        if (nodes[static_cast<std::size_t>(nb)].cluster_id == 0) {
          nodes[static_cast<std::size_t>(nb)].cluster_id = cur_cluster;
          if (nodes[static_cast<std::size_t>(nb)].is_core)
            bfs_q_.push_back(nb);
        }
      });
    }
  }

  for (auto &node : nodes)
    if (node.cluster_id == 0)
      node.cluster_id = -1;
}

void TitanLLPSCluster::fuzzy_refine(TitanExecutor &executor,
                                    const std::vector<LLPSNode> &nodes,
                                    const box &Box, int giant_cluster_id,
                                    std::vector<bool> &is_dense,
                                    std::vector<double> &bfactor,
                                    std::vector<double> &occupancy) {
  const int N = static_cast<int>(nodes.size());
  is_dense.assign(static_cast<std::size_t>(N), false);
  bfactor.assign(static_cast<std::size_t>(N), 0.0);
  occupancy.assign(static_cast<std::size_t>(N), 0.0);
  if (N == 0)
    return;

  const TitanExactPeriodicBox bmat(Box);
  int max_cid = 0;
  for (const auto &n : nodes)
    if (n.cluster_id > max_cid)
      max_cid = n.cluster_id;
  if (max_cid == 0)
    return;

  std::vector<int> cid_list;
  cid_list.reserve(static_cast<std::size_t>(max_cid));
  for (int c = 1; c <= max_cid; ++c) {
    bool has = false;
    for (const auto &n : nodes) {
      if (n.cluster_id == c) {
        has = true;
        break;
      }
    }
    if (has)
      cid_list.push_back(c);
  }
  const int K_dense = static_cast<int>(cid_list.size());
  if (K_dense == 0)
    return;
  const int K_total = K_dense + 1;
  const int dilute_k = K_dense;

  std::vector<int> cid_to_idx(static_cast<std::size_t>(max_cid + 1), -1);
  for (int k = 0; k < K_dense; ++k)
    cid_to_idx[static_cast<std::size_t>(cid_list[static_cast<std::size_t>(k)])] =
        k;

  const int giant_k =
      (giant_cluster_id > 0 && giant_cluster_id <= max_cid)
          ? cid_to_idx[static_cast<std::size_t>(giant_cluster_id)]
          : -1;

  std::vector<std::array<double, 3>> X_k(static_cast<std::size_t>(K_total));
  int ref_idx = -1;
  for (int i = 0; i < N; ++i) {
    if (nodes[static_cast<std::size_t>(i)].cluster_id == giant_cluster_id) {
      ref_idx = i;
      break;
    }
  }
  if (ref_idx < 0)
    return;

  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  int cnt = 0;
  for (int i = 0; i < N; ++i) {
    if (nodes[static_cast<std::size_t>(i)].cluster_id != giant_cluster_id)
      continue;
    double dx = nodes[static_cast<std::size_t>(i)].x - nodes[static_cast<std::size_t>(ref_idx)].x;
    double dy = nodes[static_cast<std::size_t>(i)].y - nodes[static_cast<std::size_t>(ref_idx)].y;
    double dz = nodes[static_cast<std::size_t>(i)].z - nodes[static_cast<std::size_t>(ref_idx)].z;
    bmat.apply_mic(dx, dy, dz);
    sx += nodes[static_cast<std::size_t>(ref_idx)].x + dx;
    sy += nodes[static_cast<std::size_t>(ref_idx)].y + dy;
    sz += nodes[static_cast<std::size_t>(ref_idx)].z + dz;
    ++cnt;
  }
  const double gx = sx / cnt;
  const double gy = sy / cnt;
  const double gz = sz / cnt;

  double sum_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    if (nodes[static_cast<std::size_t>(i)].cluster_id != giant_cluster_id)
      continue;
    double dx = nodes[static_cast<std::size_t>(i)].x - gx;
    double dy = nodes[static_cast<std::size_t>(i)].y - gy;
    double dz = nodes[static_cast<std::size_t>(i)].z - gz;
    bmat.apply_mic(dx, dy, dz);
    sum_sq += dx * dx + dy * dy + dz * dz;
  }
  double r_bar = std::sqrt(sum_sq / cnt);
  if (r_bar < 1.0)
    r_bar = 1.0;

  const double r_cutoff = 2.0 * r_bar;
  const double sigma_pos = r_bar / 3.0;
  const double lambda = 15.0 * sigma_pos * r_bar;
  const double r_cutoff_sq = r_cutoff * r_cutoff;

  for (int k = 0; k < K_dense; ++k) {
    const int cid = cid_list[static_cast<std::size_t>(k)];
    int ref = -1;
    for (int i = 0; i < N; ++i) {
      if (nodes[static_cast<std::size_t>(i)].cluster_id == cid) {
        ref = i;
        break;
      }
    }
    double wsx = 0.0;
    double wsy = 0.0;
    double wsz = 0.0;
    int wcnt = 0;
    for (int i = 0; i < N; ++i) {
      if (nodes[static_cast<std::size_t>(i)].cluster_id != cid)
        continue;
      double dx = nodes[static_cast<std::size_t>(i)].x - nodes[static_cast<std::size_t>(ref)].x;
      double dy = nodes[static_cast<std::size_t>(i)].y - nodes[static_cast<std::size_t>(ref)].y;
      double dz = nodes[static_cast<std::size_t>(i)].z - nodes[static_cast<std::size_t>(ref)].z;
      bmat.apply_mic(dx, dy, dz);
      wsx += nodes[static_cast<std::size_t>(ref)].x + dx;
      wsy += nodes[static_cast<std::size_t>(ref)].y + dy;
      wsz += nodes[static_cast<std::size_t>(ref)].z + dz;
      ++wcnt;
    }
    X_k[static_cast<std::size_t>(k)] = {wsx / wcnt, wsy / wcnt, wsz / wcnt};
  }
  X_k[static_cast<std::size_t>(dilute_k)] = {0.0, 0.0, 0.0};

  std::vector<double> u(static_cast<std::size_t>(N) * K_total, 0.0);
  auto U = [&](int i, int k) -> double & {
    return u[static_cast<std::size_t>(i * K_total + k)];
  };
  constexpr int max_iter = 50;
  constexpr double conv_thr = 0.01;
  constexpr double dilute_bg = 1e-4;
  const std::size_t fuzzy_workers = std::max<std::size_t>(
      1, std::min<std::size_t>(executor.capacity(), static_cast<std::size_t>(N)));
  perf_.process_executor_workers =
      std::max<std::size_t>(perf_.process_executor_workers, fuzzy_workers);

  for (int iter = 0; iter < max_iter; ++iter) {
    executor.parallel_for(
        static_cast<std::size_t>(N), fuzzy_workers, 96,
        [&](std::size_t /*worker*/, std::size_t begin, std::size_t end) {
          for (std::size_t ii = begin; ii < end; ++ii) {
            const int i = static_cast<int>(ii);
            double sum_raw = 0.0;
            for (int k = 0; k < K_dense; ++k) {
              double dx = nodes[static_cast<std::size_t>(i)].x -
                          X_k[static_cast<std::size_t>(k)][0];
              double dy = nodes[static_cast<std::size_t>(i)].y -
                          X_k[static_cast<std::size_t>(k)][1];
              double dz = nodes[static_cast<std::size_t>(i)].z -
                          X_k[static_cast<std::size_t>(k)][2];
              bmat.apply_mic(dx, dy, dz);
              const double d_sq = dx * dx + dy * dy + dz * dz;
              if (d_sq < r_cutoff_sq) {
                const double t = std::sqrt(d_sq) / r_cutoff;
                const double C = 1.0 - 3.0 * t * t + 2.0 * t * t * t;
                U(i, k) = std::exp(-d_sq / lambda) * C;
              } else {
                U(i, k) = 0.0;
              }
              sum_raw += U(i, k);
            }
            U(i, dilute_k) = dilute_bg;
            sum_raw += dilute_bg;
            if (sum_raw > 0.0) {
              for (int k = 0; k < K_total; ++k)
                U(i, k) /= sum_raw;
            } else {
              for (int k = 0; k < K_total; ++k)
                U(i, k) = 1.0 / K_total;
            }
          }
        });

    double max_shift = 0.0;
    for (int k = 0; k < K_dense; ++k) {
      int best_i = 0;
      double best_u = U(0, k);
      for (int i = 1; i < N; ++i) {
        if (U(i, k) > best_u) {
          best_u = U(i, k);
          best_i = i;
        }
      }
      double wsx = 0.0;
      double wsy = 0.0;
      double wsz = 0.0;
      double wsum = 0.0;
      for (int i = 0; i < N; ++i) {
        if (U(i, k) < 1e-8)
          continue;
        double dx = nodes[static_cast<std::size_t>(i)].x - nodes[static_cast<std::size_t>(best_i)].x;
        double dy = nodes[static_cast<std::size_t>(i)].y - nodes[static_cast<std::size_t>(best_i)].y;
        double dz = nodes[static_cast<std::size_t>(i)].z - nodes[static_cast<std::size_t>(best_i)].z;
        bmat.apply_mic(dx, dy, dz);
        wsx += U(i, k) * (nodes[static_cast<std::size_t>(best_i)].x + dx);
        wsy += U(i, k) * (nodes[static_cast<std::size_t>(best_i)].y + dy);
        wsz += U(i, k) * (nodes[static_cast<std::size_t>(best_i)].z + dz);
        wsum += U(i, k);
      }
      if (wsum > 0.0) {
        const double nx = wsx / wsum;
        const double ny = wsy / wsum;
        const double nz = wsz / wsum;
        double sdx = nx - X_k[static_cast<std::size_t>(k)][0];
        double sdy = ny - X_k[static_cast<std::size_t>(k)][1];
        double sdz = nz - X_k[static_cast<std::size_t>(k)][2];
        bmat.apply_mic(sdx, sdy, sdz);
        const double shift = sdx * sdx + sdy * sdy + sdz * sdz;
        max_shift = std::max(max_shift, shift);
        X_k[static_cast<std::size_t>(k)] = {nx, ny, nz};
      }
    }
    if (std::sqrt(max_shift) < conv_thr)
      break;
  }

  for (int i = 0; i < N; ++i) {
    int dom_k = 0;
    for (int k = 1; k < K_total; ++k)
      if (U(i, k) > U(i, dom_k))
        dom_k = k;
    const bool dense = (dom_k != dilute_k) && (U(i, dom_k) > 0.4);
    is_dense[static_cast<std::size_t>(i)] = dense;
    const double u_giant = (giant_k >= 0) ? U(i, giant_k) : 0.0;
    bfactor[static_cast<std::size_t>(i)] = std::clamp(u_giant, 0.0, 1.0);
    if (u_giant < 0.2)
      occupancy[static_cast<std::size_t>(i)] = 0.0;
    else if (u_giant < 0.8)
      occupancy[static_cast<std::size_t>(i)] = 0.5;
    else
      occupancy[static_cast<std::size_t>(i)] = 1.0;
  }
}

void TitanLLPSCluster::write_frame_to_pdb(
    const std::vector<LLPSNode> &nodes, int frame_idx,
    const std::vector<double> &bfactor, const std::vector<double> &occupancy) {
  if (!vmd_out_.is_open())
    return;
  char buf[128];
  std::snprintf(buf, sizeof(buf), "MODEL     %4d\n", frame_idx);
  vmd_out_ << buf;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto &n = nodes[i];
    const char *resname = (n.type == 1) ? "PRO" : "RNA";
    const char chain_id = (n.type == 1) ? 'A' : 'B';
    const double bf = (i < bfactor.size()) ? bfactor[i] : 0.0;
    const double occ = (i < occupancy.size()) ? occupancy[i] : 1.0;
    const int atom_serial = (n.id - 1) % 99999 + 1;
    const std::string res_str = encode_hybrid36(n.chain_idx);
    std::snprintf(buf, sizeof(buf),
                  "ATOM  %5d  CA  %3s %c%4s    %8.3f%8.3f%8.3f%6.2f%6.2f\n",
                  atom_serial, resname, chain_id, res_str.c_str(), n.x, n.y,
                  n.z, occ, bf);
    vmd_out_ << buf;
  }
  vmd_out_ << "ENDMDL\n";
}

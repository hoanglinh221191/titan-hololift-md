#include "h_bond.h"
#include "titan_simd_dispatch.h"
#include "titan_thread_compat.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "MOL.h"
#include "functions.h"
#include "miscellaneous.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinDistanceSq = 1.0e-4;
constexpr std::size_t kHBondOutputBufferSize = 1U << 20;
constexpr std::string_view kHBondAllFile = "out_hydrogen_bonds.dat";
constexpr std::string_view kHBondReceptorFile =
    "out_hydrogen_bonds_receptor.dat";
constexpr std::string_view kHBondLigandFile = "out_hydrogen_bonds_ligand.dat";

inline bool is_hydrogen_atom(const mol_info &info) {
  const auto &atom_type = info.get_atom_type();
  const std::string atom_name = info.get_atom();
  return (!atom_type.empty() && atom_type[0] == 'H') ||
         (!atom_name.empty() && atom_name[0] == 'H') ||
         (atom_name.size() >= 2 && atom_name[1] == 'H');
}

inline bool same_residue(const std::vector<mol_info> &info, std::size_t lhs,
                         std::size_t rhs) {
  return info[lhs].get_chainnum() == info[rhs].get_chainnum() &&
         info[lhs].get_resnr() == info[rhs].get_resnr();
}

inline double squared_distance(const MOL &lhs, std::size_t lhs_idx,
                               const MOL &rhs, std::size_t rhs_idx) noexcept {
  const double dx = lhs.pos.x[lhs_idx] - rhs.pos.x[rhs_idx];
  const double dy = lhs.pos.y[lhs_idx] - rhs.pos.y[rhs_idx];
  const double dz = lhs.pos.z[lhs_idx] - rhs.pos.z[rhs_idx];
  return dx * dx + dy * dy + dz * dz;
}

inline double clamp_cosine(double value) noexcept {
  return std::clamp(value, -1.0, 1.0);
}

bool try_calculate_angle_deg(const MOL &donor_mol, std::size_t donor_idx,
                             std::size_t hydrogen_idx, const MOL &acceptor_mol,
                             std::size_t acceptor_idx, double cos_cutoff,
                             double &angle_deg) noexcept {
  const double dh_x = donor_mol.pos.x[donor_idx] - donor_mol.pos.x[hydrogen_idx];
  const double dh_y = donor_mol.pos.y[donor_idx] - donor_mol.pos.y[hydrogen_idx];
  const double dh_z = donor_mol.pos.z[donor_idx] - donor_mol.pos.z[hydrogen_idx];

  const double ah_x =
      acceptor_mol.pos.x[acceptor_idx] - donor_mol.pos.x[hydrogen_idx];
  const double ah_y =
      acceptor_mol.pos.y[acceptor_idx] - donor_mol.pos.y[hydrogen_idx];
  const double ah_z =
      acceptor_mol.pos.z[acceptor_idx] - donor_mol.pos.z[hydrogen_idx];

  const double dh_norm_sq = dh_x * dh_x + dh_y * dh_y + dh_z * dh_z;
  const double ah_norm_sq = ah_x * ah_x + ah_y * ah_y + ah_z * ah_z;
  if (dh_norm_sq <= std::numeric_limits<double>::epsilon() ||
      ah_norm_sq <= std::numeric_limits<double>::epsilon()) {
    return false;
  }

  const double dot = dh_x * ah_x + dh_y * ah_y + dh_z * ah_z;
  const double cosine =
      clamp_cosine(dot / std::sqrt(dh_norm_sq * ah_norm_sq));
  if (cosine > cos_cutoff) {
    return false;
  }

  angle_deg = std::acos(cosine) * 180.0 / kPi;
  return true;
}

void process_group_pair(const MOL &pro_in, const MOL &lig, const h_group &group_1,
                        const h_group &group_2, double pair_dis_sq,
                        double cos_cutoff, h_bonds_wrapper &dat_pro_in,
                        h_bonds_wrapper &dat_lig) {
  const bool group_1_has_h = group_1.has_hydrogen();
  const bool group_2_has_h = group_2.has_hydrogen();
  if ((!group_1_has_h && !group_2_has_h) || pair_dis_sq <= kMinDistanceSq) {
    return;
  }

  const double pair_dis = std::sqrt(pair_dis_sq);

  if (group_1_has_h) {
    for (const std::size_t hydrogen_idx : group_1.h_index) {
      double angle_deg = 0.0;
      if (try_calculate_angle_deg(pro_in, group_1.index, hydrogen_idx, lig,
                                  group_2.index, cos_cutoff, angle_deg)) {
        dat_lig.dat.emplace_back(group_1.index, group_2.index, hydrogen_idx,
                                 angle_deg, pair_dis);
      }
    }
  }

  if (group_2_has_h) {
    for (const std::size_t hydrogen_idx : group_2.h_index) {
      double angle_deg = 0.0;
      if (try_calculate_angle_deg(lig, group_2.index, hydrogen_idx, pro_in,
                                  group_1.index, cos_cutoff, angle_deg)) {
        dat_pro_in.dat.emplace_back(group_2.index, group_1.index, hydrogen_idx,
                                    angle_deg, pair_dis);
      }
    }
  }
}

unsigned distance_mask_dispatch(const MOL &lhs, std::size_t lhs_idx,
                                const h_groups &rhs_groups,
                                std::size_t rhs_begin, const MOL &rhs,
                                double cutoff_sq) noexcept {
  const std::int64_t gather_idx[4] = {
      static_cast<std::int64_t>(rhs_groups.groups[rhs_begin].index),
      static_cast<std::int64_t>(rhs_groups.groups[rhs_begin + 1].index),
      static_cast<std::int64_t>(rhs_groups.groups[rhs_begin + 2].index),
      static_cast<std::int64_t>(rhs_groups.groups[rhs_begin + 3].index)};
  return titan_simd_dispatch::point_vs_gathered4_cutoff_mask(
      lhs.pos.x[lhs_idx], lhs.pos.y[lhs_idx], lhs.pos.z[lhs_idx],
      rhs.pos.x.data(), rhs.pos.y.data(), rhs.pos.z.data(), gather_idx,
      cutoff_sq, kMinDistanceSq, true);
}

bool valid_hbond_record(const h_bonds &bond,
                        const std::vector<mol_info> &acceptor_info,
                        const std::vector<mol_info> &donor_info) {
  return bond.get_index_acceptor() < acceptor_info.size() &&
         bond.get_index_h() < donor_info.size() &&
         bond.get_index_donor() < donor_info.size();
}

void append_hbond_row(std::ostringstream &out, const mol_info &acceptor,
                      const mol_info &hydrogen, const mol_info &donor,
                      const h_bonds &bond, std::size_t frame_index,
                      double time_value) {
  out << acceptor.get_atom() << '\t' << acceptor.get_chain() << '\t'
      << acceptor.get_resnr() << '\t' << hydrogen.get_atom() << '\t'
      << hydrogen.get_resnr() << '\t' << donor.get_atom() << '\t'
      << donor.get_chain() << '\t' << donor.get_resnr() << '\t'
      << acceptor.get_original_res_name() << '\t'
      << donor.get_original_res_name() << '\t' << std::fixed
      << std::setprecision(3) << bond.get_dis() << '\t' << bond.get_angle()
      << '\t' << frame_index << '\t' << time_value << '\n';
}

void write_hbond_header(std::ostream &out, std::string_view description,
                        std::string_view time_unit) {
  out << "# " << description << '\n';
  out << "#AcceptorAtom\tAcceptorChain\tAcceptorResNum\tHydrogenAtom\t"
         "HydrogenResNum\tDonorAtom\tDonorChain\tDonorResNum\t"
         "AcceptorResName\tDonorResName\tDistance_A\tAngle_DHA_deg\tFrame\t"
       << "Time_" << time_unit << '\n';
}

struct hbond_output_manager {
  std::ofstream all_stream;
  std::ofstream rec_stream;
  std::ofstream lig_stream;
  std::array<char, kHBondOutputBufferSize> all_buffer{};
  std::array<char, kHBondOutputBufferSize> rec_buffer{};
  std::array<char, kHBondOutputBufferSize> lig_buffer{};

  static void prepare_stream(std::ofstream &stream,
                             std::array<char, kHBondOutputBufferSize> &buffer) {
    stream.close();
    stream.clear();
    stream.rdbuf()->pubsetbuf(buffer.data(),
                              static_cast<std::streamsize>(buffer.size()));
  }

  static void close_stream(std::ofstream &stream) {
    if (stream.is_open()) {
      stream.flush();
      stream.close();
    }
    stream.clear();
  }

  void close() {
    close_stream(all_stream);
    close_stream(rec_stream);
    close_stream(lig_stream);
  }

  int init(const arguments &Arguments) {
    close();

    prepare_stream(all_stream, all_buffer);
    all_stream.open(std::string(kHBondAllFile), std::ios::out | std::ios::trunc);
    write_hbond_header(all_stream, "All intermolecular hydrogen bonds",
                       Arguments.tu);

    prepare_stream(rec_stream, rec_buffer);
    rec_stream.open(std::string(kHBondReceptorFile),
                    std::ios::out | std::ios::trunc);
    write_hbond_header(rec_stream,
                       "Rows where the receptor-side atom is the acceptor",
                       Arguments.tu);

    prepare_stream(lig_stream, lig_buffer);
    lig_stream.open(std::string(kHBondLigandFile),
                    std::ios::out | std::ios::trunc);
    write_hbond_header(lig_stream,
                       "Rows where the ligand-side atom is the acceptor",
                       Arguments.tu);
    return 0;
  }

  void ensure_open_for_append() {
    if (!all_stream.is_open()) {
      prepare_stream(all_stream, all_buffer);
      all_stream.open(std::string(kHBondAllFile), std::ios::out | std::ios::app);
    }
    if (!rec_stream.is_open()) {
      prepare_stream(rec_stream, rec_buffer);
      rec_stream.open(std::string(kHBondReceptorFile),
                      std::ios::out | std::ios::app);
    }
    if (!lig_stream.is_open()) {
      prepare_stream(lig_stream, lig_buffer);
      lig_stream.open(std::string(kHBondLigandFile),
                      std::ios::out | std::ios::app);
    }
  }

  void append(const std::string &all_rows, const std::string &rec_rows,
              const std::string &lig_rows) {
    if (all_rows.empty() && rec_rows.empty() && lig_rows.empty()) {
      return;
    }

    ensure_open_for_append();
    if (!all_rows.empty() && all_stream.is_open()) {
      all_stream << all_rows;
    }
    if (!rec_rows.empty() && rec_stream.is_open()) {
      rec_stream << rec_rows;
    }
    if (!lig_rows.empty() && lig_stream.is_open()) {
      lig_stream << lig_rows;
    }
  }
};

hbond_output_manager &get_hbond_output_manager() {
  static hbond_output_manager manager;
  return manager;
}

struct hbond_cell_key {
  int x{};
  int y{};
  int z{};

  bool operator==(const hbond_cell_key &rhs) const noexcept = default;
};

struct hbond_cell_key_hash {
  std::size_t operator()(const hbond_cell_key &key) const noexcept {
    const std::size_t h1 = std::hash<int>()(key.x);
    const std::size_t h2 = std::hash<int>()(key.y);
    const std::size_t h3 = std::hash<int>()(key.z);
    return h1 ^ (h2 << 1U) ^ (h3 << 2U);
  }
};

struct hbond_spatial_index {
  double cell_size{};
  std::unordered_map<hbond_cell_key, std::vector<std::size_t>,
                     hbond_cell_key_hash>
      buckets;
};

inline hbond_cell_key make_hbond_cell_key(double x, double y, double z,
                                          double cell_size) noexcept {
  const double inv_cell_size = 1.0 / cell_size;
  return {static_cast<int>(std::floor(x * inv_cell_size)),
          static_cast<int>(std::floor(y * inv_cell_size)),
          static_cast<int>(std::floor(z * inv_cell_size))};
}

hbond_spatial_index build_hbond_spatial_index(const MOL &mol,
                                              const h_groups &groups,
                                              double cell_size) {
  hbond_spatial_index spatial_index;
  spatial_index.cell_size = cell_size;
  spatial_index.buckets.reserve(groups.groups.size());

  for (std::size_t group_idx = 0; group_idx < groups.groups.size(); ++group_idx) {
    const std::size_t atom_idx = groups.groups[group_idx].index;
    const hbond_cell_key key = make_hbond_cell_key(
        mol.pos.x[atom_idx], mol.pos.y[atom_idx], mol.pos.z[atom_idx],
        cell_size);
    spatial_index.buckets[key].emplace_back(group_idx);
  }

  return spatial_index;
}

inline void evaluate_group_pair(const MOL &pro_in, const MOL &lig,
                                const h_group &group_1, const h_group &group_2,
                                bool group_1_has_h, double dis_h_sq,
                                double cos_cutoff,
                                h_bonds_wrapper &dat_pro_in,
                                h_bonds_wrapper &dat_lig) {
  if (!group_1_has_h && !group_2.has_hydrogen()) {
    return;
  }

  const double pair_dis_sq =
      squared_distance(pro_in, group_1.index, lig, group_2.index);
  if (pair_dis_sq > dis_h_sq || pair_dis_sq <= kMinDistanceSq) {
    return;
  }

  process_group_pair(pro_in, lig, group_1, group_2, pair_dis_sq, cos_cutoff,
                     dat_pro_in, dat_lig);
}

} // namespace

int print_hbond_per_snap(const MOL &pro_in, const MOL &lig,
                         const std::vector<mol_info> &info_rec,
                         const std::vector<mol_info> &info_lg,
                         const arguments &Arguments, std::size_t frame_index);

inline bool check_sidechain_hbond(const arguments &Arguments,
                                  std::string_view atom) {
  return (!Arguments.flag_sidechain_hbond ||
          (Arguments.flag_sidechain_hbond && atom != "N" && atom != "O"));
}

int init_hbond_output_files(const arguments &Arguments) {
  get_hbond_output_manager().init(Arguments);
  return 0;
}

int finalize_hbond_output_files() {
  get_hbond_output_manager().close();
  return 0;
}

int h_group::add_groups(const MOL &mol_in, std::size_t i, char _name,
                        double bond_length, const std::vector<mol_info> &info) {
  name = _name;
  index = i;
  h_index.clear();

  if (i >= mol_in.size() || i >= info.size()) {
    return 1;
  }

  const double bond_length_sq = bond_length * bond_length;
  std::size_t begin = i;
  while (begin > 0 && same_residue(info, begin - 1, i)) {
    --begin;
  }

  std::size_t end = i + 1;
  while (end < mol_in.size() && end < info.size() && same_residue(info, end, i)) {
    ++end;
  }

  for (std::size_t j = begin; j < end; ++j) {
    if (j == i || !is_hydrogen_atom(info[j])) {
      continue;
    }
    if (squared_distance(mol_in, i, mol_in, j) <= bond_length_sq) {
      h_index.emplace_back(j);
    }
  }

  return 0;
}

int h_groups::find_groups(const MOL &mol_in, const std::vector<mol_info> &info,
                          const arguments &Arguments) {
  groups.clear();
  groups.reserve(mol_in.size() / 3 + 1);

  for (std::size_t i = 0; i < mol_in.size() && i < info.size(); ++i) {
    h_group group_temp;
    const auto &atom_type = info[i].get_atom_type();
    const std::string atom_name = info[i].get_atom();

    if ((!atom_type.empty() && atom_type[0] == 'O') ||
        (!atom_name.empty() && atom_name[0] == 'O')) {
      if (!check_sidechain_hbond(Arguments, atom_name)) {
        continue;
      }
      group_temp.add_groups(mol_in, i, 'O', 1.1, info);
      groups.emplace_back(std::move(group_temp));
    } else if ((!atom_type.empty() && atom_type[0] == 'N') ||
               (!atom_name.empty() && atom_name[0] == 'N')) {
      if (!check_sidechain_hbond(Arguments, atom_name)) {
        continue;
      }
      group_temp.add_groups(mol_in, i, 'N', 1.5, info);
      groups.emplace_back(std::move(group_temp));
    } else if ((!atom_type.empty() && atom_type[0] == 'F') ||
               (!atom_name.empty() && atom_name[0] == 'F')) {
      group_temp.add_groups(mol_in, i, 'F', 1.1, info);
      groups.emplace_back(std::move(group_temp));
    }
  }

  return 0;
}

void cal_h_bonds(MOL &pro_in, MOL &lig, std::size_t begin, std::size_t end,
                 const arguments &Arguments, h_groups &groups_1,
                 h_groups &groups_2,
                 const hbond_spatial_index *groups_2_spatial,
                 h_bonds_wrapper &dat_pro_in, h_bonds_wrapper &dat_lig) {
  const double dis_h_sq = Arguments.dis_h * Arguments.dis_h;
  const double cos_cutoff = std::cos(Arguments.angle_h * kPi / 180.0);
  const std::size_t groups_2_size = groups_2.groups.size();

  for (std::size_t i = begin; i < end; ++i) {
    const h_group &group_1 = groups_1.groups[i];
    const bool group_1_has_h = group_1.has_hydrogen();

    if (groups_2_spatial != nullptr) {
      const hbond_cell_key center = make_hbond_cell_key(
          pro_in.pos.x[group_1.index], pro_in.pos.y[group_1.index],
          pro_in.pos.z[group_1.index], groups_2_spatial->cell_size);

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dz = -1; dz <= 1; ++dz) {
            const hbond_cell_key neighbor{center.x + dx, center.y + dy,
                                          center.z + dz};
            const auto bucket_it = groups_2_spatial->buckets.find(neighbor);
            if (bucket_it == groups_2_spatial->buckets.end()) {
              continue;
            }

            for (const std::size_t group_2_idx : bucket_it->second) {
              evaluate_group_pair(pro_in, lig, group_1,
                                  groups_2.groups[group_2_idx], group_1_has_h,
                                  dis_h_sq, cos_cutoff, dat_pro_in, dat_lig);
            }
          }
        }
      }
      continue;
    }

    std::size_t j = 0;

    for (; j + 3 < groups_2_size; j += 4) {
      unsigned mask = distance_mask_dispatch(pro_in, group_1.index, groups_2, j,
                                             lig, dis_h_sq);
      while (mask != 0U) {
        const unsigned lane = static_cast<unsigned>(std::countr_zero(mask));
        const h_group &group_2 = groups_2.groups[j + lane];
        evaluate_group_pair(pro_in, lig, group_1, group_2, group_1_has_h,
                            dis_h_sq, cos_cutoff, dat_pro_in, dat_lig);
        mask &= (mask - 1U);
      }
    }

    for (; j < groups_2_size; ++j) {
      evaluate_group_pair(pro_in, lig, group_1, groups_2.groups[j],
                          group_1_has_h, dis_h_sq, cos_cutoff, dat_pro_in,
                          dat_lig);
    }
  }
}

struct bounded_hbond_selection {
  std::array<std::size_t, 3> indices{};
  std::size_t size{0};
};

inline bool hbond_distance_precedes(const std::vector<h_bonds> &records,
                                    std::size_t lhs,
                                    std::size_t rhs) noexcept {
  const double lhs_distance = records[lhs].get_dis();
  const double rhs_distance = records[rhs].get_dis();
  return lhs_distance < rhs_distance ||
         (lhs_distance == rhs_distance && lhs < rhs);
}

inline void insert_hbond_candidate(bounded_hbond_selection &selection,
                                   std::size_t candidate,
                                   std::size_t limit,
                                   const std::vector<h_bonds> &records) {
  std::size_t insert_pos = selection.size;
  for (std::size_t i = 0; i < selection.size; ++i) {
    if (hbond_distance_precedes(records, candidate, selection.indices[i])) {
      insert_pos = i;
      break;
    }
  }

  if (selection.size < limit) {
    for (std::size_t i = selection.size; i > insert_pos; --i) {
      selection.indices[i] = selection.indices[i - 1];
    }
    selection.indices[insert_pos] = candidate;
    ++selection.size;
    return;
  }

  if (insert_pos >= limit) {
    return;
  }

  for (std::size_t i = limit - 1; i > insert_pos; --i) {
    selection.indices[i] = selection.indices[i - 1];
  }
  selection.indices[insert_pos] = candidate;
}

int find_h_bonds(MOL &pro_in, MOL &pro_in_ref, MOL &lig, MOL &lig_ref,
                 h_groups &groups_1, h_groups &groups_2,
                 const arguments &Arguments, std::size_t num_snap,
                 std::vector<mol_info> &info_rec,
                 std::vector<mol_info> &info_lg,
                 std::vector<std::vector<std::size_t>> &Hbonds_vector) {
  (void)pro_in_ref;
  (void)lig_ref;

  if (num_snap == 0) {
    if (Arguments.nt >= 2) {
      titan::jthread t1(&h_groups::find_groups, &groups_1, std::cref(pro_in),
                      std::cref(info_rec), std::cref(Arguments));
      titan::jthread t2(&h_groups::find_groups, &groups_2, std::cref(lig),
                      std::cref(info_lg), std::cref(Arguments));
    } else {
      groups_1.find_groups(pro_in, info_rec, Arguments);
      groups_2.find_groups(lig, info_lg, Arguments);
    }
  }

  pro_in.list_h_bonds.dat.clear();
  lig.list_h_bonds.dat.clear();

  if (groups_1.groups.empty() || groups_2.groups.empty()) {
    if (num_snap >= Hbonds_vector.size()) {
      Hbonds_vector.resize(num_snap + 1);
    }
    Hbonds_vector[num_snap].emplace_back(0);
    return 0;
  }

  const std::size_t worker_count =
      std::min<std::size_t>(std::max<std::size_t>(1, Arguments.nt),
                            groups_1.groups.size());
  std::vector<h_bonds_wrapper> h_bonds_pro(worker_count);
  std::vector<h_bonds_wrapper> h_bonds_lig(worker_count);
  const std::size_t pair_count = groups_1.groups.size() * groups_2.groups.size();
  const bool use_spatial_index = pair_count >= 8192;
  const hbond_spatial_index groups_2_spatial =
      use_spatial_index
          ? build_hbond_spatial_index(
                lig, groups_2, std::max(Arguments.dis_h, 1.0))
          : hbond_spatial_index{};
  const hbond_spatial_index *groups_2_spatial_ptr =
      use_spatial_index ? &groups_2_spatial : nullptr;

  if (worker_count == 1) {
    cal_h_bonds(pro_in, lig, 0, groups_1.groups.size(), Arguments, groups_1,
                groups_2, groups_2_spatial_ptr, h_bonds_pro[0], h_bonds_lig[0]);
  } else {
    std::vector<titan::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      const std::size_t begin = i * groups_1.groups.size() / worker_count;
      const std::size_t end = (i + 1) * groups_1.groups.size() / worker_count;
      workers.emplace_back(cal_h_bonds, std::ref(pro_in), std::ref(lig), begin,
                           end, std::cref(Arguments), std::ref(groups_1),
                           std::ref(groups_2), groups_2_spatial_ptr,
                           std::ref(h_bonds_pro[i]), std::ref(h_bonds_lig[i]));
    }
  }

  std::size_t total_pro = 0;
  std::size_t total_lig = 0;
  for (std::size_t i = 0; i < worker_count; ++i) {
    total_pro += h_bonds_pro[i].dat.size();
    total_lig += h_bonds_lig[i].dat.size();
  }
  pro_in.list_h_bonds.dat.reserve(total_pro);
  lig.list_h_bonds.dat.reserve(total_lig);

  for (std::size_t i = 0; i < worker_count; ++i) {
    pro_in.list_h_bonds.dat.insert(pro_in.list_h_bonds.dat.end(),
                                   h_bonds_pro[i].dat.begin(),
                                   h_bonds_pro[i].dat.end());
    lig.list_h_bonds.dat.insert(lig.list_h_bonds.dat.end(),
                                h_bonds_lig[i].dat.begin(),
                                h_bonds_lig[i].dat.end());
  }

  lig.post_proc_hbonds(lig, info_lg, info_rec);
  pro_in.post_proc_hbonds(pro_in, info_rec, info_lg);
  print_hbond_per_snap(pro_in, lig, info_rec, info_lg, Arguments, num_snap);

  if (num_snap >= Hbonds_vector.size()) {
    Hbonds_vector.resize(num_snap + 1);
  }
  Hbonds_vector[num_snap].emplace_back(pro_in.list_h_bonds.dat.size() +
                                       lig.list_h_bonds.dat.size());
  return 0;
}

int MOL::post_proc_hbonds(const MOL &mol, std::vector<mol_info> &info,
                          std::vector<mol_info> &counter_info) {
  (void)mol;
  const std::array<std::pair<char, std::size_t>, 4> max_hbonds = {
      std::pair<char, std::size_t>{'O', 2},
      std::pair<char, std::size_t>{'N', 3},
      std::pair<char, std::size_t>{'H', 1},
      std::pair<char, std::size_t>{'F', 1}};

  for (const auto &atom_dat : max_hbonds) {
    imp_post_proc(atom_dat, mol, info, counter_info);
  }
  return 0;
}

int MOL::imp_post_proc(const std::pair<char, std::size_t> &atom_dat,
                       const MOL &mol, std::vector<mol_info> &info,
                       std::vector<mol_info> &counter_info) {
  (void)mol;
  if (list_h_bonds.dat.empty()) {
    return 0;
  }

  std::unordered_map<std::size_t, bounded_hbond_selection> grouped_records;
  grouped_records.reserve(list_h_bonds.dat.size());
  std::vector<unsigned char> matched_record(list_h_bonds.dat.size(), 0U);

  for (std::size_t record_idx = 0; record_idx < list_h_bonds.dat.size();
       ++record_idx) {
    std::size_t tracked_index = 0;
    const std::string *atom_name = nullptr;

    if (atom_dat.first == 'H') {
      tracked_index = list_h_bonds.dat[record_idx].get_index_h();
      if (tracked_index >= counter_info.size()) {
        continue;
      }
      atom_name = &counter_info[tracked_index].get_atom_type();
    } else {
      tracked_index = list_h_bonds.dat[record_idx].get_index_acceptor();
      if (tracked_index >= info.size()) {
        continue;
      }
      atom_name = &info[tracked_index].get_atom_type();
    }

    if (atom_name != nullptr &&
        atom_name->find(atom_dat.first) != std::string::npos) {
      matched_record[record_idx] = 1U;
      auto &selection = grouped_records[tracked_index];
      insert_hbond_candidate(selection, record_idx, atom_dat.second,
                             list_h_bonds.dat);
    }
  }

  if (std::find(matched_record.begin(), matched_record.end(), 1U) ==
      matched_record.end()) {
    return 0;
  }

  std::vector<unsigned char> keep_record(list_h_bonds.dat.size(), 1U);
  for (std::size_t i = 0; i < matched_record.size(); ++i) {
    if (matched_record[i] != 0U) {
      keep_record[i] = 0U;
    }
  }

  for (auto &[tracked_index, selection] : grouped_records) {
    (void)tracked_index;
    for (std::size_t i = 0; i < selection.size; ++i) {
      keep_record[selection.indices[i]] = 1U;
    }
  }

  if (std::find(keep_record.begin(), keep_record.end(), 0U) ==
      keep_record.end()) {
    return 0;
  }

  std::vector<h_bonds> filtered_records;
  filtered_records.reserve(list_h_bonds.dat.size());
  for (std::size_t i = 0; i < list_h_bonds.dat.size(); ++i) {
    if (keep_record[i] != 0U) {
      filtered_records.emplace_back(std::move(list_h_bonds.dat[i]));
    }
  }
  list_h_bonds.dat.swap(filtered_records);
  return 0;
}

void h_groups::print_group(int m) {
  std::fstream groups_out("out_hbonds_groups_" + std::to_string(m) + ".dat",
                          std::ios::out);
  for (std::size_t i = 0; i < groups.size(); ++i) {
    groups_out << groups[i].name << " " << groups[i].index << '\n';
    for (std::size_t j = 0; j < groups[i].h_index.size(); ++j) {
      groups_out << groups[i].h_index[j] << "   ";
    }
    groups_out << '\n';
  }
}

int stat_hbonds(std::vector<std::vector<std::size_t>> &Hbonds_vector,
                const info &infos, arguments &Arguments,
                std::vector<double> &time_of_snaps,
                std::vector<std::string> &list_name) {
  if (Arguments.flag_h_bond == 0) {
    return 1;
  }

  std::ofstream stat_hb(infos.get_stathb_name(), std::ios::out);
  stat_hb << "#Time (" << Arguments.tu
          << ")\tMean_HB\tSD_HB\tMin_HB\tMax_HB\tSampleCount\n";

  for (std::size_t i = 0; i < Hbonds_vector.size(); ++i) {
    if (Hbonds_vector[i].empty()) {
      continue;
    }

    double mean = 0.0;
    double sd = 0.0;
    std::size_t min_hb = Hbonds_vector[i].front();
    std::size_t max_hb = Hbonds_vector[i].front();

    for (const std::size_t value : Hbonds_vector[i]) {
      mean += static_cast<double>(value);
      min_hb = std::min(min_hb, value);
      max_hb = std::max(max_hb, value);
    }
    mean /= static_cast<double>(Hbonds_vector[i].size());

    for (const std::size_t value : Hbonds_vector[i]) {
      sd += square(static_cast<double>(value) - mean);
    }
    sd = std::sqrt(sd / static_cast<double>(Hbonds_vector[i].size()));

    stat_hb << std::fixed << std::setprecision(3) << time_of_snaps[i] << '\t'
            << mean << '\t' << sd << '\t' << min_hb << '\t' << max_hb << '\t'
            << Hbonds_vector[i].size() << '\n';
  }
  stat_hb.close();

  std::ofstream stat_ave_hb(infos.get_stat_ave_hb_name(), std::ios::out);
  stat_ave_hb << "#Trajectory\tMean_HB\tSD_HB\tFrameCount\n";

  const auto write_summary_line =
      [&stat_ave_hb](const std::string &traj_name,
                     const std::vector<std::size_t> &samples) {
        if (samples.empty()) {
          return;
        }

        double mean = 0.0;
        double variance_sum = 0.0;
        for (const std::size_t sample : samples) {
          mean += static_cast<double>(sample);
        }
        mean /= static_cast<double>(samples.size());

        for (const std::size_t sample : samples) {
          variance_sum += square(static_cast<double>(sample) - mean);
        }

        stat_ave_hb << traj_name << '\t' << std::fixed << std::setprecision(3)
                    << mean << '\t'
                    << std::sqrt(variance_sum /
                                 static_cast<double>(samples.size()))
                    << '\t' << samples.size() << '\n';
      };

  const bool is_multi_trajectory_average_mode =
      (Arguments.flag_pattern_input || Arguments.flag_listfile ||
       Arguments.flag_all_folders) &&
      !Arguments.flag_cat_result_multi_infile;

  if (!Hbonds_vector.empty()) {
    if (!is_multi_trajectory_average_mode) {
      std::vector<std::size_t> all_samples;
      for (const auto &frame_values : Hbonds_vector) {
        all_samples.insert(all_samples.end(), frame_values.begin(),
                           frame_values.end());
      }

      const std::string traj_name =
          !list_name.empty() ? list_name.front() : "Traj_1";
      write_summary_line(traj_name, all_samples);
    } else {
      std::size_t trajectory_count = 0;
      for (const auto &frame_values : Hbonds_vector) {
        trajectory_count = std::max(trajectory_count, frame_values.size());
      }

      if (!list_name.empty()) {
        trajectory_count = std::min(trajectory_count, list_name.size());
      }

      for (std::size_t trajectory_idx = 0; trajectory_idx < trajectory_count;
           ++trajectory_idx) {
        std::vector<std::size_t> trajectory_samples;
        trajectory_samples.reserve(Hbonds_vector.size());

        for (const auto &frame_values : Hbonds_vector) {
          if (trajectory_idx < frame_values.size()) {
            trajectory_samples.emplace_back(frame_values[trajectory_idx]);
          }
        }

        const std::string traj_name =
            (list_name.size() > trajectory_idx)
                ? list_name[trajectory_idx]
                : ("Traj_" + std::to_string(trajectory_idx + 1));
        write_summary_line(traj_name, trajectory_samples);
      }
    }
  }

  stat_ave_hb.close();
  return 0;
}

int print_hbond_per_snap(const MOL &pro_in, const MOL &lig,
                         const std::vector<mol_info> &info_rec,
                         const std::vector<mol_info> &info_lg,
                         const arguments &Arguments, std::size_t frame_index) {
  const double time_value = pro_in.get_time();
  std::ostringstream out_all;
  std::ostringstream out_rec;
  std::ostringstream out_lig;

  std::size_t skipped_rec = 0;
  std::size_t skipped_lig = 0;

  for (const h_bonds &bond : pro_in.list_h_bonds.dat) {
    if (!valid_hbond_record(bond, info_rec, info_lg)) {
      ++skipped_rec;
      continue;
    }

    const mol_info &acceptor = info_rec[bond.get_index_acceptor()];
    const mol_info &hydrogen = info_lg[bond.get_index_h()];
    const mol_info &donor = info_lg[bond.get_index_donor()];
    append_hbond_row(out_all, acceptor, hydrogen, donor, bond, frame_index,
                     time_value);
    append_hbond_row(out_rec, acceptor, hydrogen, donor, bond, frame_index,
                     time_value);
  }

  for (const h_bonds &bond : lig.list_h_bonds.dat) {
    if (!valid_hbond_record(bond, info_lg, info_rec)) {
      ++skipped_lig;
      continue;
    }

    const mol_info &acceptor = info_lg[bond.get_index_acceptor()];
    const mol_info &hydrogen = info_rec[bond.get_index_h()];
    const mol_info &donor = info_rec[bond.get_index_donor()];
    append_hbond_row(out_all, acceptor, hydrogen, donor, bond, frame_index,
                     time_value);
    append_hbond_row(out_lig, acceptor, hydrogen, donor, bond, frame_index,
                     time_value);
  }

  if (skipped_rec + skipped_lig > 0) {
    std::cerr << "[WARNING] Skipped " << (skipped_rec + skipped_lig)
              << " malformed H-bond records at frame " << frame_index << ".\n";
  }

  const std::string all_rows = out_all.str();
  const std::string rec_rows = out_rec.str();
  const std::string lig_rows = out_lig.str();
  get_hbond_output_manager().append(all_rows, rec_rows, lig_rows);

  (void)Arguments;
  return 0;
}

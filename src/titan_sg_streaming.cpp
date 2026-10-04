#include "titan_sg_streaming.h"

#include "c6c12.h"
#include "charmm_nonbonded.h"
#include "martini_nonbonded.h"
#include "titan_simd_dispatch.h"
#include "titan_ui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

#if TITAN_SIMD_X86
#include <immintrin.h>
#endif

namespace {

struct KernelEnergy {
  double elec{0.0};
  double vdw{0.0};
  KernelEnergy &operator+=(const KernelEnergy &rhs) {
    elec += rhs.elec;
    vdw += rhs.vdw;
    return *this;
  }
};

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

bool contains_ff(const std::string &ff, const char *needle) {
  return lowercase_copy(ff).find(lowercase_copy(needle)) != std::string::npos;
}

std::string format_bytes(long double bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  constexpr long double kib = 1024.0L;
  constexpr long double mib = kib * 1024.0L;
  constexpr long double gib = mib * 1024.0L;
  if (bytes >= gib)
    oss << bytes / gib << " GiB";
  else if (bytes >= mib)
    oss << bytes / mib << " MiB";
  else if (bytes >= kib)
    oss << bytes / kib << " KiB";
  else
    oss << bytes << " B";
  return oss.str();
}

std::string safe_filename(std::string name) {
  for (char &c : name) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
        c == '"' || c == '<' || c == '>' || c == '|')
      c = '_';
  }
  return name;
}

double resolved_cutoff_nm(const arguments &args) {
  return titan_resolve_sg_cutoff_nm(args);
}

martini::ElectrostaticConfig martini_config_for_sg(const arguments &args,
                                                   double cutoff_nm,
                                                   bool cutoff_active) {
  auto cfg = martini::electrostatic_config(args);
  if (cutoff_active && cutoff_nm > 0.0) {
    cfg.cutoff_nm = cutoff_nm;
    cfg.k_rf = 0.5 / cube(cfg.cutoff_nm);
    cfg.c_rf = 1.5 / cfg.cutoff_nm;
  }
  return cfg;
}

struct CellList {
  bool enabled{false};
  int nx{1}, ny{1}, nz{1};
  double lx{0.0}, ly{0.0}, lz{0.0};
  double cell_x{0.0}, cell_y{0.0}, cell_z{0.0};
  std::vector<int> head;
  std::vector<int> next;
};

int wrap_index(int idx, int dim) {
  int r = idx % dim;
  if (r < 0)
    r += dim;
  return r;
}

int cell_id_from_coord(double x, double y, double z, const CellList &grid) {
  int ix = static_cast<int>(std::floor(x / grid.cell_x));
  int iy = static_cast<int>(std::floor(y / grid.cell_y));
  int iz = static_cast<int>(std::floor(z / grid.cell_z));
  ix = wrap_index(ix, grid.nx);
  iy = wrap_index(iy, grid.ny);
  iz = wrap_index(iz, grid.nz);
  return ix * grid.ny * grid.nz + iy * grid.nz + iz;
}

CellList build_ligand_cell_list(const std::vector<std::size_t> &lig_atoms,
                                const MOL &lig_frame, double cutoff_nm) {
  CellList grid;
  const double cutoff_a = cutoff_nm * 10.0;
  grid.lx = lig_frame.Box.get_a();
  grid.ly = lig_frame.Box.get_b();
  grid.lz = lig_frame.Box.get_c();
  if (lig_atoms.empty() || cutoff_a <= 0.0 || grid.lx <= 0.0 ||
      grid.ly <= 0.0 || grid.lz <= 0.0)
    return grid;

  grid.nx = std::max(1, static_cast<int>(grid.lx / cutoff_a));
  grid.ny = std::max(1, static_cast<int>(grid.ly / cutoff_a));
  grid.nz = std::max(1, static_cast<int>(grid.lz / cutoff_a));
  grid.cell_x = grid.lx / static_cast<double>(grid.nx);
  grid.cell_y = grid.ly / static_cast<double>(grid.ny);
  grid.cell_z = grid.lz / static_cast<double>(grid.nz);
  grid.head.assign(static_cast<std::size_t>(grid.nx * grid.ny * grid.nz), -1);
  grid.next.assign(lig_atoms.size(), -1);

  for (std::size_t pos = 0; pos < lig_atoms.size(); ++pos) {
    const std::size_t atom = lig_atoms[pos];
    if (atom >= lig_frame.size())
      continue;
    const int cell = cell_id_from_coord(lig_frame.pos.x[atom],
                                        lig_frame.pos.y[atom],
                                        lig_frame.pos.z[atom], grid);
    grid.next[pos] = grid.head[static_cast<std::size_t>(cell)];
    grid.head[static_cast<std::size_t>(cell)] = static_cast<int>(pos);
  }
  grid.enabled = true;
  return grid;
}

#if TITAN_SIMD_X86
TITAN_TARGET_AVX2_FMA
KernelEnergy martini_exact_range_avx2(
    const std::vector<std::size_t> &rec_atoms,
    const std::vector<std::size_t> &lig_atoms, const MOL &rec_frame,
    const MOL &lig_frame, const Multi_MOL &rec, const Multi_MOL &lig,
    const cache_sigma_epsilon &cached_data, const arguments &args,
    std::size_t begin_rec, std::size_t end_rec, double cutoff_nm,
    bool cutoff_active, const martini::ElectrostaticConfig &mcfg) {
  KernelEnergy out;
  const __m256d v_one = _mm256_set1_pd(1.0);
  const __m256d v_four = _mm256_set1_pd(4.0);
  const __m256d v_scale = _mm256_set1_pd(0.01);
  const __m256d v_min_d = _mm256_set1_pd(1e-10);
  const __m256d v_coul = _mm256_set1_pd(martini::kCoulomb / mcfg.epsilon_r);
  const __m256d v_k_rf = _mm256_set1_pd(mcfg.k_rf);
  const __m256d v_c_rf = _mm256_set1_pd(mcfg.c_rf);
  const __m256d v_lx = _mm256_set1_pd(rec_frame.Box.get_a());
  const __m256d v_ly = _mm256_set1_pd(rec_frame.Box.get_b());
  const __m256d v_lz = _mm256_set1_pd(rec_frame.Box.get_c());
  const bool pbc_x = rec_frame.Box.get_a() > 0.0;
  const bool pbc_y = rec_frame.Box.get_b() > 0.0;
  const bool pbc_z = rec_frame.Box.get_c() > 0.0;
  const bool has_pair_table = cached_data.has_martini_pair_table();
  const int type_count = cached_data.martini_types();

  for (std::size_t lig_atom : lig_atoms) {
    if (lig_atom >= lig_frame.size() || lig_atom >= lig.parameters.size())
      continue;
    const __m256d v_lig_x = _mm256_set1_pd(lig_frame.pos.x[lig_atom]);
    const __m256d v_lig_y = _mm256_set1_pd(lig_frame.pos.y[lig_atom]);
    const __m256d v_lig_z = _mm256_set1_pd(lig_frame.pos.z[lig_atom]);
    const __m256d v_lig_charge = _mm256_set1_pd(lig.parameters.charge[lig_atom]);
    const int lig_type = lig.parameters.get_type_id(lig_atom);

    for (std::size_t rp = begin_rec; rp < end_rec; rp += 4) {
      alignas(32) double rx[4]{}, ry[4]{}, rz[4]{}, rq[4]{}, rs[4]{}, re[4]{};
      alignas(32) double sigma_lane[4]{}, epsilon_lane[4]{};
      alignas(32) double active_lane[4]{};
      std::array<std::size_t, 4> rec_lane{};
      for (int k = 0; k < 4; ++k) {
        const bool active = rp + static_cast<std::size_t>(k) < end_rec;
        const std::size_t atom =
            active ? rec_atoms[rp + static_cast<std::size_t>(k)] : 0;
        rec_lane[static_cast<std::size_t>(k)] = atom;
        active_lane[k] = active && atom < rec_frame.size() &&
                                 atom < rec.parameters.size()
                             ? 1.0
                             : 0.0;
        if (active_lane[k] == 0.0)
          continue;
        rx[k] = rec_frame.pos.x[atom];
        ry[k] = rec_frame.pos.y[atom];
        rz[k] = rec_frame.pos.z[atom];
        rq[k] = rec.parameters.charge[atom];
        rs[k] = rec.parameters.sigma[atom];
        re[k] = rec.parameters.epsilon[atom];
        const int rec_type = rec.parameters.get_type_id(atom);
        if (has_pair_table && lig_type >= 0 && rec_type >= 0) {
          const std::size_t idx = static_cast<std::size_t>(lig_type) *
                                      static_cast<std::size_t>(type_count) +
                                  static_cast<std::size_t>(rec_type);
          sigma_lane[k] = cached_data.martini_pair_sigma[idx];
          epsilon_lane[k] = cached_data.martini_pair_epsilon[idx];
        } else {
          sigma_lane[k] = 0.5 * (lig.parameters.sigma[lig_atom] +
                                 rec.parameters.sigma[atom]);
          epsilon_lane[k] = std::sqrt(std::max(
              0.0, lig.parameters.epsilon[lig_atom] *
                       rec.parameters.epsilon[atom]));
        }
      }

      __m256d dx = _mm256_sub_pd(_mm256_load_pd(rx), v_lig_x);
      __m256d dy = _mm256_sub_pd(_mm256_load_pd(ry), v_lig_y);
      __m256d dz = _mm256_sub_pd(_mm256_load_pd(rz), v_lig_z);
      if (pbc_x)
        dx = _mm256_sub_pd(
            dx, _mm256_mul_pd(v_lx, _mm256_round_pd(_mm256_div_pd(dx, v_lx),
                                                    _MM_FROUND_TO_NEAREST_INT |
                                                        _MM_FROUND_NO_EXC)));
      if (pbc_y)
        dy = _mm256_sub_pd(
            dy, _mm256_mul_pd(v_ly, _mm256_round_pd(_mm256_div_pd(dy, v_ly),
                                                    _MM_FROUND_TO_NEAREST_INT |
                                                        _MM_FROUND_NO_EXC)));
      if (pbc_z)
        dz = _mm256_sub_pd(
            dz, _mm256_mul_pd(v_lz, _mm256_round_pd(_mm256_div_pd(dz, v_lz),
                                                    _MM_FROUND_TO_NEAREST_INT |
                                                        _MM_FROUND_NO_EXC)));

      __m256d r2 = _mm256_add_pd(_mm256_mul_pd(dx, dx),
                                 _mm256_add_pd(_mm256_mul_pd(dy, dy),
                                               _mm256_mul_pd(dz, dz)));
      r2 = _mm256_mul_pd(r2, v_scale);
      const __m256d d = _mm256_sqrt_pd(r2);
      const __m256d safe_d =
          _mm256_blendv_pd(v_one, d, _mm256_cmp_pd(d, v_min_d, _CMP_GT_OQ));
      const __m256d inv_d = _mm256_div_pd(v_one, safe_d);

      const __m256d sigma = _mm256_load_pd(sigma_lane);
      const __m256d epsilon = _mm256_load_pd(epsilon_lane);
      const __m256d sr = _mm256_div_pd(sigma, safe_d);
      const __m256d sr2 = _mm256_mul_pd(sr, sr);
      const __m256d sr6 = _mm256_mul_pd(_mm256_mul_pd(sr2, sr2), sr2);
      const __m256d vdw = _mm256_mul_pd(
          _mm256_mul_pd(v_four, epsilon),
          _mm256_sub_pd(_mm256_mul_pd(sr6, sr6), sr6));
      const __m256d rf =
          _mm256_sub_pd(_mm256_add_pd(inv_d, _mm256_mul_pd(v_k_rf, r2)), v_c_rf);
      const __m256d elec = _mm256_mul_pd(
          _mm256_mul_pd(v_coul, _mm256_mul_pd(v_lig_charge, _mm256_load_pd(rq))),
          rf);

      alignas(32) double d_lane[4]{}, e_lane[4]{}, v_lane[4]{};
      _mm256_store_pd(d_lane, d);
      _mm256_store_pd(e_lane, elec);
      _mm256_store_pd(v_lane, vdw);
      for (int k = 0; k < 4; ++k) {
        if (active_lane[k] == 0.0 || d_lane[k] <= 1e-10)
          continue;
        if (cutoff_active && cutoff_nm > 0.0 && d_lane[k] > cutoff_nm)
          continue;
        const std::size_t rec_atom = rec_lane[static_cast<std::size_t>(k)];
        if (martini::excluded_pair(args, rec.info, lig.info, rec_atom, lig_atom,
                                   d_lane[k], sigma_lane[k]))
          continue;
        out.elec += e_lane[k];
        out.vdw += v_lane[k];
      }
    }
  }
  return out;
}
#endif

} // namespace

void SgStreamingAnalyzer::RunningStats::add(double elec, double vdw) {
  const double total = elec + vdw;
  ++n;
  const double dn = static_cast<double>(n);
  const double de = elec - mean_elec;
  mean_elec += de / dn;
  m2_elec += de * (elec - mean_elec);
  const double dv = vdw - mean_vdw;
  mean_vdw += dv / dn;
  m2_vdw += dv * (vdw - mean_vdw);
  const double dt = total - mean_total;
  mean_total += dt / dn;
  m2_total += dt * (total - mean_total);
}

double SgStreamingAnalyzer::RunningStats::sd_elec() const {
  return n > 1 ? std::sqrt(m2_elec / static_cast<double>(n - 1)) : 0.0;
}

double SgStreamingAnalyzer::RunningStats::sd_vdw() const {
  return n > 1 ? std::sqrt(m2_vdw / static_cast<double>(n - 1)) : 0.0;
}

double SgStreamingAnalyzer::RunningStats::sd_total() const {
  return n > 1 ? std::sqrt(m2_total / static_cast<double>(n - 1)) : 0.0;
}

bool SgStreamingAnalyzer::should_use_stream_mode(
    const cache_sigma_epsilon &cached_data, const arguments &args,
    std::size_t rec_atoms, std::size_t lig_atoms,
    std::size_t available_ram_bytes, std::fstream *log) {
  const long double dense_bytes =
      static_cast<long double>(rec_atoms) * static_cast<long double>(lig_atoms) *
      static_cast<long double>(sizeof(double)) *
      static_cast<long double>(args.flag_cal_force ? 8 : 2);
  const double fallback_budget =
      available_ram_bytes > 0
          ? 0.65 * static_cast<double>(available_ram_bytes)
          : 10.0 * 1024.0 * 1024.0 * 1024.0;
  memory_budget_bytes_ =
      args.sg_memory_gb > 0.0 ? args.sg_memory_gb * 1024.0 * 1024.0 * 1024.0
                              : fallback_budget;

  const bool cache_available = !cached_data.elec_ener.empty();
  if (args.sg_mode == "stream") {
    cache_policy_note_ = "forced stream by -sg-mode=stream";
    return true;
  }
  if (args.sg_mode == "cache") {
    if (cache_available) {
      cache_policy_note_ = "forced cache by -sg-mode=cache";
      return false;
    }
    cache_policy_note_ =
        "requested cache, but dense pair cache is unavailable; using stream";
    if (log)
      TitanUI::log_message(*log, "[SG] " + cache_policy_note_);
    return true;
  }
  if (args.sg_cache == "off" || args.low_memory) {
    cache_policy_note_ = args.sg_cache == "off"
                             ? "cache disabled by -sg-cache=off"
                             : "low-memory mode";
    return true;
  }
  if (!cache_available) {
    cache_policy_note_ = "dense pair cache unavailable";
    return true;
  }
  if (args.sg_cache == "on" && dense_bytes <= memory_budget_bytes_) {
    cache_policy_note_ = "cache requested and within budget";
    return false;
  }
  if (dense_bytes > memory_budget_bytes_) {
    cache_policy_note_ = "estimated dense cache " + format_bytes(dense_bytes) +
                         " exceeds -sg budget " +
                         format_bytes(memory_budget_bytes_);
    return true;
  }
  cache_policy_note_ = "auto selected cache";
  return false;
}

SgStreamingAnalyzer::AtomGroup
SgStreamingAnalyzer::build_group(groups_cal &groups, int subgroup_index,
                                 std::size_t parent_first,
                                 std::size_t mol_size) const {
  AtomGroup group;
  group.group_index = subgroup_index;
  group.name = groups.group_name(static_cast<std::size_t>(subgroup_index));
  const std::size_t atom_count =
      groups.groups_atom_number(static_cast<std::size_t>(subgroup_index));
  group.atoms.reserve(atom_count);
  for (std::size_t k = 0; k < atom_count; ++k) {
    const std::size_t global_atom =
        groups.atom_index(static_cast<std::size_t>(subgroup_index), k);
    if (global_atom < parent_first)
      continue;
    const std::size_t local_atom = global_atom - parent_first;
    if (local_atom < mol_size)
      group.atoms.push_back(local_atom);
  }
  return group;
}

bool SgStreamingAnalyzer::init(groups_cal &groups, const Multi_MOL &rec,
                               const Multi_MOL &lig,
                               const cache_sigma_epsilon &cached_data,
                               const arguments &args,
                               std::size_t available_ram_bytes,
                               std::fstream *log) {
  active_ = false;
  pairs_.clear();
  if (!args.flag_sub_groups)
    return false;

  const std::size_t rec_atoms =
      !rec.m_mol.empty() ? rec.m_mol.front().size() : rec.info.size();
  const std::size_t lig_atoms =
      !lig.m_mol.empty() ? lig.m_mol.front().size() : lig.info.size();
  const bool stream = should_use_stream_mode(cached_data, args, rec_atoms,
                                             lig_atoms, available_ram_bytes, log);
  resolved_mode_ = stream ? "stream" : "cache";
  cutoff_nm_ = resolved_cutoff_nm(args);
  cutoff_active_ = cutoff_nm_ > 0.0;
  resolved_kernel_ = lowercase_copy(args.sg_kernel);
  if (resolved_kernel_ == "auto") {
    resolved_kernel_ = titan_simd_dispatch::runtime_has_avx2() ? "avx2"
                                                               : "scalar";
  }
  if (resolved_kernel_ == "avx2" &&
      !titan_simd_dispatch::runtime_has_avx2()) {
    resolved_kernel_ = "scalar";
    cache_policy_note_ += "; AVX2 unavailable, using scalar";
  }
  use_cell_list_ = cutoff_active_;
  energy_provider_.configure(rec, lig, cached_data, args, cutoff_nm_,
                             cutoff_active_);

  write_memory_report(cached_data, args, rec_atoms, lig_atoms,
                      available_ram_bytes);
  if (!stream)
    return false;

  const std::size_t parent1_first =
      groups.groups_atom_number(groups.group_1_index()) > 0
          ? groups.atom_index(groups.group_1_index(), 0)
          : 0;
  const std::size_t parent2_first =
      groups.groups_atom_number(groups.group_2_index()) > 0
          ? groups.atom_index(groups.group_2_index(), 0)
          : 0;

  std::vector<AtomGroup> rec_groups;
  std::vector<AtomGroup> lig_groups;
  rec_groups.reserve(groups.subgroup1_size());
  lig_groups.reserve(groups.subgroup2_size());
  for (std::size_t i = 0; i < groups.subgroup1_size(); ++i)
    rec_groups.push_back(build_group(groups, groups.subgroup_1_index(i),
                                     parent1_first, rec_atoms));
  for (std::size_t i = 0; i < groups.subgroup2_size(); ++i)
    lig_groups.push_back(build_group(groups, groups.subgroup_2_index(i),
                                     parent2_first, lig_atoms));

  pairs_.reserve(rec_groups.size() * lig_groups.size());
  for (const auto &rg : rec_groups) {
    for (const auto &lg : lig_groups) {
      if (rg.atoms.empty() || lg.atoms.empty())
        continue;
      PairState pair;
      pair.name = safe_filename(rg.name + "-" + lg.name);
      pair.rec_group = rg;
      pair.lig_group = lg;
      pairs_.push_back(std::move(pair));
    }
  }

  if (args.sg_timeseries == "summary-only")
    timeseries_full_ = false;
  else if (args.sg_timeseries == "full")
    timeseries_full_ = true;
  else
    timeseries_full_ = pairs_.size() <= 256;

  active_ = !pairs_.empty();
  if (!active_) {
    TitanUI::print_warning("[SG] No valid subgroup pairs for streaming mode.");
    return false;
  }
  open_timeseries_files(args);
  if (log) {
    TitanUI::log_kv(*log, "SG Mode", resolved_mode_);
    TitanUI::log_kv(*log, "SG Kernel", resolved_kernel_);
    TitanUI::log_kv(*log, "SG Cutoff",
                    cutoff_active_ ? std::to_string(cutoff_nm_) + " nm"
                                   : "none");
    TitanUI::log_kv(*log, "SG Pairs", std::to_string(pairs_.size()));
    TitanUI::log_message(*log, "[SG] " + cache_policy_note_);
  }
  return true;
}

void SgStreamingAnalyzer::write_memory_report(
    const cache_sigma_epsilon &cached_data, const arguments &args,
    std::size_t rec_atoms, std::size_t lig_atoms,
    std::size_t available_ram_bytes) const {
  std::ofstream out("sg_memory_report.dat");
  const long double dense_bytes =
      static_cast<long double>(rec_atoms) * static_cast<long double>(lig_atoms) *
      static_cast<long double>(sizeof(double)) *
      static_cast<long double>(args.flag_cal_force ? 8 : 2);
  out << "# TITAN -sg memory report\n";
  out << "requested_mode " << args.sg_mode << "\n";
  out << "resolved_mode " << resolved_mode_ << "\n";
  out << "cache_policy " << args.sg_cache << "\n";
  out << "cache_available " << (!cached_data.elec_ener.empty() ? "true" : "false")
      << "\n";
  out << "rec_atoms " << rec_atoms << "\n";
  out << "lig_atoms " << lig_atoms << "\n";
  out << "estimated_dense_cache_bytes " << std::fixed << std::setprecision(0)
      << dense_bytes << "\n";
  out << "estimated_dense_cache_human " << format_bytes(dense_bytes) << "\n";
  out << "available_ram_bytes " << available_ram_bytes << "\n";
  out << "sg_memory_budget_bytes " << std::fixed << std::setprecision(0)
      << memory_budget_bytes_ << "\n";
  out << "note " << cache_policy_note_ << "\n";
}

void SgStreamingAnalyzer::open_timeseries_files(const arguments &args) {
  for (auto &pair : pairs_) {
    pair.elec_ts.open("energy_electrostatic_time_" + pair.name + ".xvg",
                      std::ios::out);
    pair.vdw_ts.open("energy_vdw_time_" + pair.name + ".xvg", std::ios::out);
    pair.total_ts.open("energy_total_time_" + pair.name + ".xvg",
                       std::ios::out);
    auto write_header = [&](std::ofstream &out, const char *series) {
      out << "# TITAN -sg streaming time series\n";
      out << "# mode=" << resolved_mode_ << " kernel=" << resolved_kernel_
          << " cache=off";
      if (!timeseries_full_)
        out << " timeseries=summary-only";
      out << "\n";
      out << "@    title \"Subgroup " << series << " energy\"\n";
      out << "@    xaxis label \"Time (" << args.tu << ")\"\n";
      out << "@    yaxis label \"Energy (" << args.eu << ")\"\n";
      out << "@ s0 legend \"Time\"\n";
      out << "@ s1 legend \"" << series << "\"\n";
      out << "@ s2 legend \"Streaming SD placeholder\"\n";
    };
    write_header(pair.elec_ts, "Electrostatic");
    write_header(pair.vdw_ts, "van der Waals");
    write_header(pair.total_ts, "Total");
    if (!timeseries_full_) {
      pair.elec_ts << "# Data omitted by -sg-timeseries=summary-only\n";
      pair.vdw_ts << "# Data omitted by -sg-timeseries=summary-only\n";
      pair.total_ts << "# Data omitted by -sg-timeseries=summary-only\n";
    }
  }
}

SgStreamingAnalyzer::PairEnergy SgStreamingAnalyzer::compute_pair_energy(
    const PairState &pair, const MOL &rec_frame, const MOL &lig_frame,
    const Multi_MOL &rec, const Multi_MOL &lig,
    const cache_sigma_epsilon &cached_data, const arguments &args,
    TitanExecutor &executor) {
  PairEnergy result;
  if (pair.rec_group.atoms.empty() || pair.lig_group.atoms.empty())
    return result;

  const auto mcfg = martini_config_for_sg(args, cutoff_nm_, cutoff_active_);
  const bool martini = contains_ff(cached_data.force_field_name.empty()
                                       ? args.force_f
                                       : cached_data.force_field_name,
                                   "martini");
  const bool avx2_exact =
      resolved_kernel_ == "avx2" && martini && !use_cell_list_;
  const CellList grid =
      use_cell_list_ ? build_ligand_cell_list(pair.lig_group.atoms, lig_frame,
                                              cutoff_nm_)
                     : CellList{};
  if (grid.enabled)
    ++cell_list_frames_;
  else
    ++exact_frames_;

  const std::size_t worker_count = std::max<std::size_t>(1, args.nt);
  std::vector<KernelEnergy> local(worker_count);
  executor.parallel_for(
      pair.rec_group.atoms.size(), worker_count, 16,
      [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
        KernelEnergy acc;
        if (avx2_exact) {
#if TITAN_SIMD_X86
          acc += martini_exact_range_avx2(
              pair.rec_group.atoms, pair.lig_group.atoms, rec_frame, lig_frame,
              rec, lig, cached_data, args, begin, end, cutoff_nm_,
              cutoff_active_, mcfg);
#endif
        } else if (grid.enabled) {
          std::array<int, 27> seen_cells{};
          for (std::size_t rp = begin; rp < end; ++rp) {
            const std::size_t rec_atom = pair.rec_group.atoms[rp];
            if (rec_atom >= rec_frame.size())
              continue;
            const int center =
                cell_id_from_coord(rec_frame.pos.x[rec_atom],
                                   rec_frame.pos.y[rec_atom],
                                   rec_frame.pos.z[rec_atom], grid);
            const int cx = center / (grid.ny * grid.nz);
            const int rem = center - cx * grid.ny * grid.nz;
            const int cy = rem / grid.nz;
            const int cz = rem - cy * grid.nz;
            int seen_count = 0;
            for (int ox = -1; ox <= 1; ++ox)
              for (int oy = -1; oy <= 1; ++oy)
                for (int oz = -1; oz <= 1; ++oz) {
                  const int nx = wrap_index(cx + ox, grid.nx);
                  const int ny = wrap_index(cy + oy, grid.ny);
                  const int nz = wrap_index(cz + oz, grid.nz);
                  const int cell = nx * grid.ny * grid.nz + ny * grid.nz + nz;
                  bool duplicate = false;
                  for (int s = 0; s < seen_count; ++s) {
                    if (seen_cells[static_cast<std::size_t>(s)] == cell) {
                      duplicate = true;
                      break;
                    }
                  }
                  if (duplicate)
                    continue;
                  seen_cells[static_cast<std::size_t>(seen_count++)] = cell;
                  for (int pos = grid.head[static_cast<std::size_t>(cell)];
                       pos != -1; pos = grid.next[static_cast<std::size_t>(pos)]) {
                    const std::size_t lig_atom =
                        pair.lig_group.atoms[static_cast<std::size_t>(pos)];
                    const auto energy = energy_provider_.pair_energy(
                        rec_frame, lig_frame, rec_atom, lig_atom);
                    acc.elec += energy.elec;
                    acc.vdw += energy.vdw;
                  }
                }
          }
        } else {
          for (std::size_t rp = begin; rp < end; ++rp) {
            const std::size_t rec_atom = pair.rec_group.atoms[rp];
            for (std::size_t lig_atom : pair.lig_group.atoms) {
              const auto energy = energy_provider_.pair_energy(
                  rec_frame, lig_frame, rec_atom, lig_atom);
              acc.elec += energy.elec;
              acc.vdw += energy.vdw;
            }
          }
        }
        if (worker_id < local.size())
          local[worker_id] += acc;
      });
  for (const auto &v : local) {
    result.elec += v.elec;
    result.vdw += v.vdw;
  }
  if (avx2_exact)
    ++avx2_frames_;
  return result;
}

void SgStreamingAnalyzer::process_frame(
    const MOL &rec_frame, const MOL &lig_frame, const Multi_MOL &rec,
    const Multi_MOL &lig, const cache_sigma_epsilon &cached_data,
    const arguments &args, TitanExecutor &executor,
    std::size_t trajectory_index) {
  if (!active_)
    return;
  const auto t0 = std::chrono::steady_clock::now();
  double time = rec_frame.time_of_snap;
  if (args.tu == "ns")
    time /= 1000.0;

  for (auto &pair : pairs_) {
    const PairEnergy energy =
        compute_pair_energy(pair, rec_frame, lig_frame, rec, lig, cached_data,
                            args, executor);
    pair.all.add(energy.elec, energy.vdw);
    if (trajectory_index >= pair.per_traj.size())
      pair.per_traj.resize(trajectory_index + 1);
    pair.per_traj[trajectory_index].add(energy.elec, energy.vdw);
    if (timeseries_full_) {
      pair.elec_ts << std::setw(12) << std::left << time << std::right
                   << std::setw(17) << energy.elec / args.eu_par
                   << std::setw(17) << 0.0 << '\n';
      pair.vdw_ts << std::setw(12) << std::left << time << std::right
                  << std::setw(17) << energy.vdw / args.eu_par
                  << std::setw(17) << 0.0 << '\n';
      pair.total_ts << std::setw(12) << std::left << time << std::right
                    << std::setw(17)
                    << (energy.elec + energy.vdw) / args.eu_par
                    << std::setw(17) << 0.0 << '\n';
    }
  }
  ++frames_processed_;
  process_time_ += std::chrono::steady_clock::now() - t0;
}

void SgStreamingAnalyzer::finalize(const arguments &args) {
  if (!active_)
    return;
  for (const auto &pair : pairs_) {
    std::ofstream out("average_energy_force_" + pair.name + ".dat");
    out << "# TITAN -sg streaming summary\n";
    out << "# mode=" << resolved_mode_ << " kernel=" << resolved_kernel_
        << " cache=off\n";
    out << "Average Electrostatic energy: "
        << pair.all.mean_elec / args.eu_par << " (" << args.eu
        << "). Standard deviation: " << pair.all.sd_elec() / args.eu_par
        << " (" << args.eu << ")\n";
    out << "Average van der Waals energy: " << pair.all.mean_vdw / args.eu_par
        << " (" << args.eu
        << "). Standard deviation: " << pair.all.sd_vdw() / args.eu_par
        << " (" << args.eu << ")\n";
    out << "Average total energy: " << pair.all.mean_total / args.eu_par
        << " (" << args.eu
        << "). Standard deviation: " << pair.all.sd_total() / args.eu_par
        << " (" << args.eu << ")\n";
    out << "Frames: " << pair.all.n << "\n";
  }

  std::ofstream per("sg_per_trajectory_summary.csv");
  per << "GroupPair,Trajectory,Frames,MeanElec,StdElec,MeanVdw,StdVdw,"
         "MeanTotal,StdTotal,Unit\n";
  for (const auto &pair : pairs_) {
    for (std::size_t traj = 0; traj < pair.per_traj.size(); ++traj) {
      const auto &st = pair.per_traj[traj];
      if (st.n == 0)
        continue;
      per << pair.name << ',' << traj << ',' << st.n << ','
          << st.mean_elec / args.eu_par << ',' << st.sd_elec() / args.eu_par
          << ',' << st.mean_vdw / args.eu_par << ','
          << st.sd_vdw() / args.eu_par << ','
          << st.mean_total / args.eu_par << ','
          << st.sd_total() / args.eu_par << ',' << args.eu << '\n';
    }
  }

  std::ofstream perf("sg_performance_report.dat");
  const double ms =
      std::chrono::duration<double, std::milli>(process_time_).count();
  perf << "# TITAN -sg streaming performance report\n";
  perf << "resolved_mode " << resolved_mode_ << "\n";
  perf << "resolved_kernel " << resolved_kernel_ << "\n";
  perf << "frames_processed " << frames_processed_ << "\n";
  perf << "subgroup_pairs " << pairs_.size() << "\n";
  perf << "cutoff_active " << (cutoff_active_ ? "true" : "false") << "\n";
  perf << "cutoff_nm " << cutoff_nm_ << "\n";
  perf << "timeseries_full " << (timeseries_full_ ? "true" : "false") << "\n";
  perf << "cell_list_pair_frames " << cell_list_frames_ << "\n";
  perf << "exact_pair_frames " << exact_frames_ << "\n";
  perf << "avx2_pair_frames " << avx2_frames_ << "\n";
  perf << "wall_ms " << ms << "\n";
  if (frames_processed_ > 0)
    perf << "ms_per_frame " << ms / static_cast<double>(frames_processed_)
         << "\n";
}

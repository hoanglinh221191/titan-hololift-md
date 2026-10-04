#include "titan_energy_decomposition.h"
#include "titan_energy_validation.h"
#include "titan_electrostatics.h"

#include "arguments.h"
#include "c6c12.h"
#include "charmm_nonbonded.h"
#include "gromos_nonbonded.h"
#include "titan_topology_energy.h"
#include "martini_nonbonded.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace {

std::string lowercase_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

martini::ElectrostaticConfig
martini_config_for_cutoff(const arguments &args, double cutoff_nm,
                          bool cutoff_active) {
  auto cfg = martini::electrostatic_config(args);
  if (cutoff_active && cutoff_nm > 0.0) {
    cfg.cutoff_nm = cutoff_nm;
    cfg.k_rf = 0.5 / (cfg.cutoff_nm * cfg.cutoff_nm * cfg.cutoff_nm);
    cfg.c_rf = 1.5 / cfg.cutoff_nm;
  }
  return cfg;
}

} // namespace

bool titan_energy_force_field_contains(const std::string &ff,
                                       const char *needle) {
  return lowercase_copy(ff).find(lowercase_copy(needle)) != std::string::npos;
}

double titan_energy_mic_delta(double d, double box_len) {
  if (box_len > 0.0 && std::isfinite(box_len))
    d -= std::nearbyint(d / box_len) * box_len;
  return d;
}

double titan_resolve_sg_cutoff_nm(const arguments &args) {
  if (args.sg_cutoff == "none")
    return -1.0;
  if (args.sg_cutoff == "value" && args.sg_cutoff_value_nm > 0.0)
    return args.sg_cutoff_value_nm;
  if (args.distance_energy_cutoff > 0.0)
    return args.distance_energy_cutoff;
  if (titan_energy_force_field_contains(args.force_f, "martini"))
    return martini::kDefaultCutoffNm;
  return -1.0;
}

void EnergyDecompositionProvider::configure(
    const Multi_MOL &rec, const Multi_MOL &lig,
    const cache_sigma_epsilon &cached_data, const arguments &args,
    double cutoff_nm, bool cutoff_active) {
  rec_ = &rec;
  lig_ = &lig;
  cached_data_ = &cached_data;
  args_ = &args;
  cutoff_nm_ = cutoff_nm;
  cutoff_active_ = cutoff_active && cutoff_nm > 0.0;
  force_field_ = cached_data.force_field_name.empty() ? args.force_f
                                                      : cached_data.force_field_name;
}

bool EnergyDecompositionProvider::ready() const noexcept {
  return rec_ != nullptr && lig_ != nullptr && cached_data_ != nullptr &&
         args_ != nullptr && has_force_parameters();
}

bool EnergyDecompositionProvider::has_force_parameters() const noexcept {
  return rec_ != nullptr && lig_ != nullptr && !rec_->parameters.empty() &&
         !lig_->parameters.empty();
}

EnergyPairDecomposition EnergyDecompositionProvider::pair_energy(
    const MOL &rec_frame, const MOL &lig_frame, std::size_t rec_pos,
    std::size_t lig_pos) const {
  if (!ready() || rec_ == nullptr || lig_ == nullptr)
    return {};
  return pair_energy_generic(rec_frame, *rec_, rec_pos, lig_frame, *lig_,
                             lig_pos);
}

bool EnergyDecompositionProvider::inter_pair_energies_batch(
    const MOL &rec_frame, const MOL &lig_frame, std::size_t rec_begin,
    std::size_t rec_end, std::size_t lig_begin, std::size_t lig_end,
    std::span<double> elec_out, std::span<double> vdw_out) const {
  if (!ready() || rec_ == nullptr || lig_ == nullptr ||
      !rec_frame.coordinates_consistent() ||
      !lig_frame.coordinates_consistent() || rec_begin > rec_end ||
      lig_begin > lig_end || rec_end > rec_frame.size() ||
      lig_end > lig_frame.size() || rec_end > rec_->parameters.size() ||
      lig_end > lig_->parameters.size())
    return false;

  const std::size_t rec_count = rec_end - rec_begin;
  const std::size_t lig_count = lig_end - lig_begin;
  if (lig_count != 0 &&
      rec_count > std::numeric_limits<std::size_t>::max() / lig_count)
    return false;
  const std::size_t output_size = rec_count * lig_count;
  if (elec_out.size() < output_size || vdw_out.size() < output_size)
    return false;

  for (std::size_t lig_pos = lig_begin; lig_pos < lig_end; ++lig_pos) {
    const std::size_t row = lig_pos - lig_begin;
    std::size_t rec_pos = rec_begin;
#ifdef __AVX2__
    const __m256d lig_x = _mm256_set1_pd(lig_frame.pos.x[lig_pos]);
    const __m256d lig_y = _mm256_set1_pd(lig_frame.pos.y[lig_pos]);
    const __m256d lig_z = _mm256_set1_pd(lig_frame.pos.z[lig_pos]);
    for (; rec_pos + 4 <= rec_end; rec_pos += 4) {
      __m256d dx = _mm256_sub_pd(
          _mm256_loadu_pd(rec_frame.pos.x.data() + rec_pos), lig_x);
      __m256d dy = _mm256_sub_pd(
          _mm256_loadu_pd(rec_frame.pos.y.data() + rec_pos), lig_y);
      __m256d dz = _mm256_sub_pd(
          _mm256_loadu_pd(rec_frame.pos.z.data() + rec_pos), lig_z);
      const __m256d squared_distance = _mm256_add_pd(
          _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
          _mm256_mul_pd(dz, dz));
      alignas(32) double squared_distance_lane[4];
      _mm256_store_pd(squared_distance_lane, squared_distance);
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const auto energy = pair_energy_generic_at_distance(
            rec_frame, *rec_, rec_pos + lane, lig_frame, *lig_, lig_pos,
            0.1 * std::sqrt(squared_distance_lane[lane]));
        const std::size_t index =
            titan_energy_cache_index(row, rec_count, rec_pos - rec_begin + lane);
        elec_out[index] = energy.elec;
        vdw_out[index] = energy.vdw;
      }
    }
#endif
    for (; rec_pos < rec_end; ++rec_pos) {
      const auto energy =
          pair_energy_generic(rec_frame, *rec_, rec_pos, lig_frame, *lig_,
                              lig_pos);
      const std::size_t index =
          titan_energy_cache_index(row, rec_count, rec_pos - rec_begin);
      elec_out[index] = energy.elec;
      vdw_out[index] = energy.vdw;
    }
  }
  return true;
}

EnergyPairDecomposition EnergyDecompositionProvider::rec_pair_energy(
    const MOL &rec_frame, std::size_t rec_pos_a, std::size_t rec_pos_b) const {
  if (cached_data_ == nullptr || args_ == nullptr || rec_ == nullptr)
    return {};
  return pair_energy_generic(rec_frame, *rec_, rec_pos_a, rec_frame, *rec_,
                             rec_pos_b);
}

EnergyPairDecomposition EnergyDecompositionProvider::lig_pair_energy(
    const MOL &lig_frame, std::size_t lig_pos_a, std::size_t lig_pos_b) const {
  if (cached_data_ == nullptr || args_ == nullptr || lig_ == nullptr)
    return {};
  return pair_energy_generic(lig_frame, *lig_, lig_pos_a, lig_frame, *lig_,
                             lig_pos_b);
}

const mol_info *EnergyDecompositionProvider::rec_info(
    std::size_t pos) const noexcept {
  if (rec_ == nullptr || pos >= rec_->info.size())
    return nullptr;
  return &rec_->info[pos];
}

const mol_info *EnergyDecompositionProvider::lig_info(
    std::size_t pos) const noexcept {
  if (lig_ == nullptr || pos >= lig_->info.size())
    return nullptr;
  return &lig_->info[pos];
}

EnergyPairDecomposition EnergyDecompositionProvider::pair_energy_generic(
    const MOL &frame_a, const Multi_MOL &mol_a, std::size_t pos_a,
    const MOL &frame_b, const Multi_MOL &mol_b, std::size_t pos_b) const {
  if (cached_data_ == nullptr || args_ == nullptr || pos_a >= frame_a.size() ||
      pos_b >= frame_b.size() || pos_a >= mol_a.parameters.size() ||
      pos_b >= mol_b.parameters.size() || !frame_a.coordinates_consistent() ||
      !frame_b.coordinates_consistent())
    return {};

  double dx = frame_a.pos.x[pos_a] - frame_b.pos.x[pos_b];
  double dy = frame_a.pos.y[pos_a] - frame_b.pos.y[pos_b];
  double dz = frame_a.pos.z[pos_a] - frame_b.pos.z[pos_b];
  // Match the standard energy kernels exactly: frames are consumed in their
  // preprocessed/reconstructed coordinate image and no implicit MIC is added.
  const double d_nm = 0.1 * std::sqrt(dx * dx + dy * dy + dz * dz);
  return pair_energy_generic_at_distance(frame_a, mol_a, pos_a, frame_b, mol_b,
                                         pos_b, d_nm);
}

EnergyPairDecomposition
EnergyDecompositionProvider::pair_energy_generic_at_distance(
    const MOL &frame_a, const Multi_MOL &mol_a, std::size_t pos_a,
    const MOL &frame_b, const Multi_MOL &mol_b, std::size_t pos_b,
    double d_nm) const {
  EnergyPairDecomposition out;
  if (cached_data_ == nullptr || args_ == nullptr || pos_a >= frame_a.size() ||
      pos_b >= frame_b.size() || pos_a >= mol_a.parameters.size() ||
      pos_b >= mol_b.parameters.size())
    return out;
  if (titan_nb::topology_energy_required(*cached_data_, *args_)) {
    const auto pair = titan_nb::evaluate_pair(*cached_data_, *args_, mol_a.info.at(pos_a),
                                            mol_b.info.at(pos_b), d_nm,
                                            cutoff_active_ ? cutoff_nm_ : -1.0);
    const double factor = args_->self_int && &mol_a != &mol_b ? 0.5 : 1.0;
    return {factor * pair.elec, factor * pair.vdw};
  }
  if (d_nm <= titan_energy_validation::kMinimumPairDistanceNm)
    return out;
  if (cutoff_active_ && cutoff_nm_ > 0.0 && d_nm > cutoff_nm_)
    return out;

  if (titan_energy_force_field_contains(force_field_, "martini")) {
    const auto cfg =
        martini_config_for_cutoff(*args_, cutoff_nm_, cutoff_active_);
    if (!martini::within_cutoff(cfg, d_nm))
      return out;
    double sigma_ij{}, epsilon_ij{};
    martini::pair_params(*cached_data_, mol_a.parameters, mol_b.parameters, pos_a,
                         pos_b, sigma_ij, epsilon_ij);
    if (martini::excluded_pair(*args_, mol_a.info, mol_b.info, pos_a, pos_b,
                               d_nm, sigma_ij))
      return out;
    out.elec = martini::electro_energy(cfg, mol_a.parameters.charge[pos_a],
                                       mol_b.parameters.charge[pos_b], d_nm);
    out.vdw = martini::lj_energy(sigma_ij, epsilon_ij, d_nm);
    return out;
  }

  const bool is_amber =
      titan_energy_force_field_contains(force_field_, "amber");
  const bool is_charmm =
      titan_energy_force_field_contains(force_field_, "charmm");
  const bool is_opls = titan_energy_force_field_contains(force_field_, "opls");
  const bool is_gromos =
      titan_energy_force_field_contains(force_field_, "gromos");
  const double qi = mol_a.parameters.charge[pos_a];
  const double qj = mol_b.parameters.charge[pos_b];
  double screening = 1.0;
  if (args_->flag_elec_screening && (is_amber || is_charmm) &&
      args_->salt_concentration > 0.0) {
    const double lambda_debye = titan_electrostatics::debye_length_nm(
        args_->salt_concentration);
    screening = std::exp(-d_nm / lambda_debye);
  }
  out.elec = 138.935457839 * qi * qj * screening /
             (std::max(1e-12, args_->epsilon_in) * d_nm);

  if (is_charmm) {
    double sigma_ij{}, epsilon_ij{};
    charmm::pair_params(*cached_data_, mol_a.parameters, mol_b.parameters, pos_a,
                        pos_b, sigma_ij, epsilon_ij);
    if (args_->self_int &&
        ((sigma_ij > 0.0 && d_nm < sigma_ij) ||
         (args_->flag_interchain_energy &&
          mol_b.info[pos_b].get_chain() == mol_a.info[pos_a].get_chain())))
      return {};
    out.vdw = charmm::lj_energy(sigma_ij, epsilon_ij, d_nm);
  } else if (is_gromos) {
    if (args_->self_int &&
        ((&mol_a == &mol_b && pos_a == pos_b) ||
         (args_->flag_interchain_energy &&
          mol_b.info[pos_b].get_chain() == mol_a.info[pos_a].get_chain())))
      return {};
    double c6_ij{}, c12_ij{};
    gromos::pair_params(*cached_data_, mol_a.parameters, mol_b.parameters,
                        pos_a, pos_b, c6_ij, c12_ij);
    const double inv_r2 = 1.0 / (d_nm * d_nm);
    const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
    out.vdw = c12_ij * inv_r6 * inv_r6 - c6_ij * inv_r6;
  } else {
    double sigma_ij =
        0.5 * (mol_b.parameters.sigma[pos_b] + mol_a.parameters.sigma[pos_a]);
    const double epsilon_ij = std::sqrt(std::max(
        0.0, mol_b.parameters.epsilon[pos_b] * mol_a.parameters.epsilon[pos_a]));
    if (is_opls)
      sigma_ij = std::sqrt(std::max(
          0.0, mol_b.parameters.sigma[pos_b] * mol_a.parameters.sigma[pos_a]));
    const bool repulsive_only =
        is_amber && (mol_b.parameters.sigma[pos_b] < 0.0 ||
                     mol_a.parameters.sigma[pos_a] < 0.0);
    const double sr6 = c6(sigma_ij / d_nm);
    if (args_->self_int &&
        (sr6 > 1.0 ||
         (args_->flag_interchain_energy &&
          mol_b.info[pos_b].get_chain() == mol_a.info[pos_a].get_chain())))
      return {};
    // AMBER and OPLS-AA store the Lennard-Jones well depth as epsilon.
    // Match the production pair kernels' 4*epsilon convention.
    out.vdw = 4.0 * epsilon_ij *
              (square(sr6) - (repulsive_only ? 0.0 : sr6));
  }
  return out;
}

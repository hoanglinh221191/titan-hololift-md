#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace TitanEntropy {

struct InteractionEntropyOptions {
  double rt_energy{1.0};
  double filter_sigma{2.0};
  std::size_t min_samples{1};
  std::size_t block_size{0};
};

struct InteractionEntropyResult {
  std::size_t samples{0};
  std::size_t filtered_samples{0};
  double ess{0.0};
  double dimensionless{0.0};
  double minus_tds_energy{0.0};
  std::size_t block_size{0};
  std::size_t block_count{0};
  double block_mean_energy{0.0};
  double block_std_energy{0.0};
  double block_drift_energy{0.0};
  std::string status{"disabled"};
};

InteractionEntropyResult
compute_interaction_entropy(std::span<const double> total_energy,
                            const InteractionEntropyOptions &options);

InteractionEntropyResult
compute_interaction_entropy(std::span<const double> elec_energy,
                            std::span<const double> vdw_energy,
                            const InteractionEntropyOptions &options);

} // namespace TitanEntropy

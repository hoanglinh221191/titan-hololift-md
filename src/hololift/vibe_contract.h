#ifndef TITAN_HOLOLIFT_VIBE_CONTRACT_H
#define TITAN_HOLOLIFT_VIBE_CONTRACT_H

#include "types.h"

#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace titan_hololift {

HoloLiftSourceContract
make_hololift_vibe_contract(std::string producer_version,
                            std::string package_id,
                            std::string trajectory_id);

std::expected<void, std::string>
validate_hololift_vibe_contract(const HoloLiftSourceContract &contract);

inline HoloLiftSourceContract
make_hololift_vibe_v1_contract(std::string producer_version,
                               std::string package_id,
                               std::string trajectory_id) {
  return make_hololift_vibe_contract(
      std::move(producer_version), std::move(package_id),
      std::move(trajectory_id));
}

inline std::expected<void, std::string>
validate_hololift_vibe_v1_contract(
    const HoloLiftSourceContract &contract) {
  return validate_hololift_vibe_contract(contract);
}

std::expected<HoloLiftObservationClass, std::string>
parse_hololift_vibe_observation_class(std::string_view value);

std::string_view
hololift_vibe_observation_class_name(HoloLiftObservationClass value) noexcept;

} // namespace titan_hololift

#endif

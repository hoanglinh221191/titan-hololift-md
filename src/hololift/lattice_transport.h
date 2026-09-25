#ifndef TITAN_HOLOLIFT_LATTICE_TRANSPORT_H
#define TITAN_HOLOLIFT_LATTICE_TRANSPORT_H

#include "types.h"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>

namespace titan_hololift {

enum class HoloLiftLatticeBasisPolicy : std::uint8_t {
  RequireBasisContinuous,
  TransportUnimodularBasis,
};

enum class HoloLiftLatticeBasisTransportFailure : std::uint8_t {
  InvalidPolicy,
  InvalidInputBox,
  TransformOutOfRange,
  NoUnimodularCandidate,
  Ambiguous,
  BasisDiscontinuity,
  ContinuityMismatch,
  AutomaticResidualGate,
  AutomaticImprovementGate,
  InvalidSelectedTransport,
};

struct HoloLiftLatticeBasisTransportError {
  HoloLiftLatticeBasisTransportFailure failure =
      HoloLiftLatticeBasisTransportFailure::InvalidPolicy;
  std::string message;
};

[[nodiscard]] constexpr bool hololift_lattice_basis_policy_is_experimental(
    HoloLiftLatticeBasisPolicy policy) noexcept {
  return policy == HoloLiftLatticeBasisPolicy::TransportUnimodularBasis;
}

struct HoloLiftLatticeBasisTransport {
  // Row-basis convention: transported_box = to_reference * input_box.
  std::array<std::int64_t, 9> to_reference{
      1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<std::int64_t, 9> from_reference{
      1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 9> transported_box{};
  double relative_mismatch = 0.0;
  double identity_relative_mismatch = 0.0;
  double improvement_ratio = 1.0;
  bool applied = false;
  bool ambiguous = false;
  bool identity_fast_path = false;
  bool local_search_completed = false;
  bool accepted_under_policy = false;
  bool explicit_remap_provenance_available = false;
  bool provenance_sufficient_for_certification = false;
};

std::expected<HoloLiftLatticeBasisTransport, std::string>
find_hololift_lattice_basis_transport(
    std::span<const double, 9> previous_reference_box,
    std::span<const double, 9> current_input_box,
    HoloLiftLatticeBasisPolicy policy, double maximum_relative_mismatch,
    double ambiguity_tolerance,
    double automatic_nonidentity_max_relative_mismatch = 0.02,
    double automatic_minimum_improvement_ratio = 4.0);

std::expected<HoloLiftLatticeBasisTransport,
              HoloLiftLatticeBasisTransportError>
evaluate_hololift_lattice_basis_transport(
    std::span<const double, 9> previous_reference_box,
    std::span<const double, 9> current_input_box,
    HoloLiftLatticeBasisPolicy policy, double maximum_relative_mismatch,
    double ambiguity_tolerance,
    double automatic_nonidentity_max_relative_mismatch = 0.02,
    double automatic_minimum_improvement_ratio = 4.0);

std::expected<HoloLiftLatticeBasisTransport, std::string>
make_hololift_lattice_basis_transport(
    const std::array<std::int64_t, 9> &to_reference,
    std::span<const double, 9> input_box);

std::expected<HoloLiftLatticeImage, std::string>
hololift_lattice_image_to_reference(
    const HoloLiftLatticeImage &input_image,
    const HoloLiftLatticeBasisTransport &transport);

std::expected<HoloLiftLatticeImage, std::string>
hololift_lattice_image_from_reference(
    const HoloLiftLatticeImage &reference_image,
    const HoloLiftLatticeBasisTransport &transport);

std::array<double, 3> hololift_fractional_to_reference(
    const std::array<double, 3> &input_fractional,
    const HoloLiftLatticeBasisTransport &transport) noexcept;

} // namespace titan_hololift

#endif

#include "lattice_transport.h"

#include "../pbctopo/lattice_math.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace titan_hololift {
namespace {

constexpr std::array<std::int64_t, 9> kIdentity{
    1, 0, 0, 0, 1, 0, 0, 0, 1};

std::array<double, 9>
multiply(const std::array<std::int64_t, 9> &lhs,
         std::span<const double, 9> rhs) noexcept {
  std::array<double, 9> result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        result[3 * row + column] +=
            static_cast<double>(lhs[3 * row + inner]) *
            rhs[3 * inner + column];
      }
    }
  }
  return result;
}

std::int64_t determinant(const std::array<std::int64_t, 9> &matrix) noexcept {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

std::array<std::int64_t, 9>
unimodular_inverse(const std::array<std::int64_t, 9> &matrix,
                   std::int64_t determinant_value) noexcept {
  return {
      (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant_value,
      (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant_value,
      (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant_value,
      (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant_value,
      (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant_value,
      (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant_value,
      (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant_value,
      (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant_value,
      (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant_value};
}

double relative_mismatch(std::span<const double, 9> reference,
                         const std::array<double, 9> &candidate) noexcept {
  double difference2 = 0.0;
  double reference2 = 0.0;
  for (std::size_t index = 0; index < 9; ++index) {
    const double difference = candidate[index] - reference[index];
    difference2 += difference * difference;
    reference2 += reference[index] * reference[index];
  }
  if (!std::isfinite(difference2) || !std::isfinite(reference2) ||
      reference2 <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(difference2 / reference2);
}

double transport_improvement_ratio(double identity_mismatch,
                                   double selected_mismatch) noexcept {
  if (!std::isfinite(identity_mismatch) || identity_mismatch < 0.0 ||
      !std::isfinite(selected_mismatch) || selected_mismatch < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (identity_mismatch == 0.0 && selected_mismatch == 0.0)
    return 1.0;
  return identity_mismatch /
         std::max(selected_mismatch, std::numeric_limits<double>::epsilon());
}

struct BasisCandidate {
  std::array<std::int64_t, 9> matrix = kIdentity;
  std::array<double, 9> transported_box{};
  double mismatch = std::numeric_limits<double>::infinity();
  bool valid = false;
};

void consider_candidate(std::span<const double, 9> previous_reference_box,
                        std::span<const double, 9> current_input_box,
                        const std::array<std::int64_t, 9> &matrix,
                        BasisCandidate &best, BasisCandidate &second) noexcept {
  const std::int64_t det = determinant(matrix);
  if (det != 1 && det != -1)
    return;
  BasisCandidate candidate;
  candidate.matrix = matrix;
  candidate.transported_box = multiply(matrix, current_input_box);
  candidate.mismatch =
      relative_mismatch(previous_reference_box, candidate.transported_box);
  candidate.valid = std::isfinite(candidate.mismatch);
  if (!candidate.valid)
    return;
  if (!best.valid || candidate.mismatch < best.mismatch ||
      (candidate.mismatch == best.mismatch && candidate.matrix < best.matrix)) {
    if (best.valid && best.matrix != candidate.matrix)
      second = best;
    best = candidate;
  } else if (candidate.matrix != best.matrix &&
             (!second.valid || candidate.mismatch < second.mismatch ||
              (candidate.mismatch == second.mismatch &&
               candidate.matrix < second.matrix))) {
    second = candidate;
  }
}

bool checked_add(std::int64_t lhs, std::int64_t rhs,
                 std::int64_t &result) noexcept {
  constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
  if ((rhs > 0 && lhs > maximum - rhs) ||
      (rhs < 0 && lhs < minimum - rhs)) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool checked_multiply(std::int64_t lhs, std::int64_t rhs,
                      std::int64_t &result) noexcept {
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
  if ((lhs == -1 && rhs == std::numeric_limits<std::int64_t>::min()) ||
      (rhs == -1 && lhs == std::numeric_limits<std::int64_t>::min())) {
    return false;
  }
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  const auto minimum = std::numeric_limits<std::int64_t>::min();
  if (lhs > 0) {
    if ((rhs > 0 && lhs > maximum / rhs) ||
        (rhs < 0 && rhs < minimum / lhs))
      return false;
  } else if ((rhs > 0 && lhs < minimum / rhs) ||
             (rhs < 0 && lhs < maximum / rhs)) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

std::expected<HoloLiftLatticeImage, std::string>
multiply_transpose_checked(const std::array<std::int64_t, 9> &matrix,
                           const HoloLiftLatticeImage &image) {
  const std::array<std::int64_t, 3> source{image.x, image.y, image.z};
  std::array<std::int64_t, 3> result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t inner = 0; inner < 3; ++inner) {
      std::int64_t product = 0;
      if (!checked_multiply(matrix[3 * inner + row], source[inner], product) ||
          !checked_add(result[row], product, result[row])) {
        return std::unexpected("HoloLift lattice-basis image transform overflows");
      }
    }
  }
  return HoloLiftLatticeImage{result[0], result[1], result[2]};
}

} // namespace

std::expected<HoloLiftLatticeBasisTransport,
              HoloLiftLatticeBasisTransportError>
evaluate_hololift_lattice_basis_transport(
    std::span<const double, 9> previous_reference_box,
    std::span<const double, 9> current_input_box,
    HoloLiftLatticeBasisPolicy policy, double maximum_relative_mismatch,
    double ambiguity_tolerance,
    double automatic_nonidentity_max_relative_mismatch,
    double automatic_minimum_improvement_ratio) {
  if (!std::isfinite(maximum_relative_mismatch) ||
      maximum_relative_mismatch <= 0.0 ||
      !std::isfinite(ambiguity_tolerance) || ambiguity_tolerance <= 0.0 ||
      !std::isfinite(automatic_nonidentity_max_relative_mismatch) ||
      automatic_nonidentity_max_relative_mismatch <= 0.0 ||
      !std::isfinite(automatic_minimum_improvement_ratio) ||
      automatic_minimum_improvement_ratio <= 1.0) {
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::InvalidPolicy,
        "HoloLift lattice-basis transport policy is invalid"});
  }

  const titan_pbctopo::PbctopoLatticeMetric previous_metric(
      previous_reference_box);
  const titan_pbctopo::PbctopoLatticeMetric current_metric(current_input_box);
  if (!previous_metric.valid() || !current_metric.valid())
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::InvalidInputBox,
        "HoloLift lattice-basis input box failed the shared lattice audit"});
  std::array<double, 9> real_transform{};
  for (std::size_t row = 0; row < 3; ++row) {
    const auto scaled = current_metric.cartesian_to_scaled(
        previous_reference_box[3 * row],
        previous_reference_box[3 * row + 1],
        previous_reference_box[3 * row + 2]);
    std::copy(scaled.begin(), scaled.end(),
              real_transform.begin() + static_cast<std::ptrdiff_t>(3 * row));
  }
  const auto identity_box = multiply(kIdentity, current_input_box);
  const double identity_mismatch =
      relative_mismatch(previous_reference_box, identity_box);

  double maximum_identity_transform_delta = 0.0;
  for (std::size_t index = 0; index < real_transform.size(); ++index) {
    maximum_identity_transform_delta = std::max(
        maximum_identity_transform_delta,
        std::fabs(real_transform[index] -
                  static_cast<double>(kIdentity[index])));
  }
  constexpr double kIdentityVoronoiInterior = 0.25;
  if (std::isfinite(identity_mismatch) &&
      identity_mismatch <= maximum_relative_mismatch &&
      maximum_identity_transform_delta < kIdentityVoronoiInterior) {
    HoloLiftLatticeBasisTransport result;
    result.transported_box = identity_box;
    result.relative_mismatch = identity_mismatch;
    result.identity_relative_mismatch = identity_mismatch;
    result.improvement_ratio = 1.0;
    result.identity_fast_path = true;
    result.accepted_under_policy = true;
    result.provenance_sufficient_for_certification = true;
    return result;
  }

  std::array<std::int64_t, 9> rounded{};
  for (std::size_t index = 0; index < rounded.size(); ++index) {
    if (!std::isfinite(real_transform[index]) ||
        std::fabs(real_transform[index]) > 1023.0) {
      return std::unexpected(HoloLiftLatticeBasisTransportError{
          HoloLiftLatticeBasisTransportFailure::TransformOutOfRange,
          "HoloLift lattice-basis transform exceeds the configured local "
          "search range"});
    }
    rounded[index] = static_cast<std::int64_t>(std::llround(real_transform[index]));
  }

  BasisCandidate best;
  BasisCandidate second;
  consider_candidate(previous_reference_box, current_input_box, kIdentity,
                     best, second);
  std::array<std::int64_t, 9> candidate{};
  constexpr std::size_t combination_count = 19683; // 3^9
  for (std::size_t code = 0; code < combination_count; ++code) {
    std::size_t digits = code;
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      candidate[index] = rounded[index] +
                         static_cast<std::int64_t>(digits % 3) - 1;
      digits /= 3;
    }
    if (candidate == kIdentity)
      continue;
    consider_candidate(previous_reference_box, current_input_box, candidate,
                       best, second);
  }
  if (!best.valid)
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::NoUnimodularCandidate,
        "HoloLift found no unimodular basis transport"});

  const double tie_scale =
      std::max({1.0, std::fabs(best.mismatch),
                second.valid ? std::fabs(second.mismatch) : 0.0});
  const bool ambiguous =
      second.valid &&
      std::fabs(second.mismatch - best.mismatch) <=
          ambiguity_tolerance * tie_scale;
  BasisCandidate selected = best;
  if (ambiguous)
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::Ambiguous,
        "HoloLift lattice-basis transport is ambiguous"});
  if (policy == HoloLiftLatticeBasisPolicy::RequireBasisContinuous) {
    if (best.matrix != kIdentity &&
        best.mismatch + ambiguity_tolerance * tie_scale < identity_mismatch) {
      return std::unexpected(HoloLiftLatticeBasisTransportError{
          HoloLiftLatticeBasisTransportFailure::BasisDiscontinuity,
          "HoloLift detected a frame-dependent lattice-basis remapping"});
    }
    selected.matrix = kIdentity;
    selected.transported_box = identity_box;
    selected.mismatch = identity_mismatch;
    selected.valid = std::isfinite(identity_mismatch);
  } else if (best.matrix != kIdentity) {
    const double permitted_nonidentity_mismatch =
        std::min(maximum_relative_mismatch,
                 automatic_nonidentity_max_relative_mismatch);
    const double improvement_ratio =
        transport_improvement_ratio(identity_mismatch, best.mismatch);
    if (best.mismatch > permitted_nonidentity_mismatch) {
      return std::unexpected(HoloLiftLatticeBasisTransportError{
          HoloLiftLatticeBasisTransportFailure::AutomaticResidualGate,
          "HoloLift automatic basis transport residual exceeds the "
          "nonidentity safety gate"});
    }
    if (!std::isfinite(improvement_ratio)) {
      return std::unexpected(HoloLiftLatticeBasisTransportError{
          HoloLiftLatticeBasisTransportFailure::AutomaticImprovementGate,
          "HoloLift automatic basis transport improvement is invalid"});
    }
    if (improvement_ratio < automatic_minimum_improvement_ratio) {
      return std::unexpected(HoloLiftLatticeBasisTransportError{
          HoloLiftLatticeBasisTransportFailure::AutomaticImprovementGate,
          "HoloLift automatic basis transport lacks sufficient improvement "
          "over identity"});
    }
  }
  if (!selected.valid ||
      selected.mismatch > maximum_relative_mismatch) {
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::ContinuityMismatch,
        "HoloLift lattice-basis continuity exceeds the configured mismatch"});
  }

  const std::int64_t det = determinant(selected.matrix);
  if (det != 1 && det != -1)
    return std::unexpected(HoloLiftLatticeBasisTransportError{
        HoloLiftLatticeBasisTransportFailure::InvalidSelectedTransport,
        "HoloLift selected a non-unimodular basis transport"});
  HoloLiftLatticeBasisTransport result;
  result.to_reference = selected.matrix;
  result.from_reference = unimodular_inverse(selected.matrix, det);
  result.transported_box = selected.transported_box;
  result.relative_mismatch = selected.mismatch;
  result.identity_relative_mismatch = identity_mismatch;
  result.improvement_ratio =
      transport_improvement_ratio(identity_mismatch, selected.mismatch);
  result.applied = selected.matrix != kIdentity;
  result.ambiguous = ambiguous;
  result.local_search_completed = true;
  result.accepted_under_policy = true;
  // No current Phase 0 producer supplies an exact remap matrix. Therefore an
  // automatically inferred nonidentity transform remains experimental and
  // cannot support a strong local temporal-lift certificate by itself.
  result.provenance_sufficient_for_certification = !result.applied;
  return result;
}

std::expected<HoloLiftLatticeBasisTransport, std::string>
find_hololift_lattice_basis_transport(
    std::span<const double, 9> previous_reference_box,
    std::span<const double, 9> current_input_box,
    HoloLiftLatticeBasisPolicy policy, double maximum_relative_mismatch,
    double ambiguity_tolerance,
    double automatic_nonidentity_max_relative_mismatch,
    double automatic_minimum_improvement_ratio) {
  auto result = evaluate_hololift_lattice_basis_transport(
      previous_reference_box, current_input_box, policy,
      maximum_relative_mismatch, ambiguity_tolerance,
      automatic_nonidentity_max_relative_mismatch,
      automatic_minimum_improvement_ratio);
  if (!result)
    return std::unexpected(result.error().message);
  return *result;
}

std::expected<HoloLiftLatticeBasisTransport, std::string>
make_hololift_lattice_basis_transport(
    const std::array<std::int64_t, 9> &to_reference,
    std::span<const double, 9> input_box) {
  const titan_pbctopo::PbctopoLatticeMetric input_metric(input_box);
  if (!input_metric.valid()) {
    return std::unexpected(
        "HoloLift input box failed the shared lattice audit");
  }
  if (std::any_of(to_reference.begin(), to_reference.end(),
                  [](std::int64_t value) {
                    return value < -1024 || value > 1024;
                  })) {
    return std::unexpected(
        "HoloLift lattice-basis matrix exceeds the verified integer range");
  }
  const std::int64_t det = determinant(to_reference);
  if (det != 1 && det != -1)
    return std::unexpected("HoloLift lattice-basis matrix is not unimodular");
  HoloLiftLatticeBasisTransport result;
  result.to_reference = to_reference;
  result.from_reference = unimodular_inverse(to_reference, det);
  result.transported_box = multiply(to_reference, input_box);
  if (!std::all_of(result.transported_box.begin(),
                   result.transported_box.end(),
                   [](double value) { return std::isfinite(value); })) {
    return std::unexpected("HoloLift transported box is not finite");
  }
  const titan_pbctopo::PbctopoLatticeMetric transported_metric(
      result.transported_box);
  if (!transported_metric.valid()) {
    return std::unexpected(
        "HoloLift transported box failed the shared lattice audit");
  }
  result.applied = to_reference != kIdentity;
  return result;
}

std::expected<HoloLiftLatticeImage, std::string>
hololift_lattice_image_to_reference(
    const HoloLiftLatticeImage &input_image,
    const HoloLiftLatticeBasisTransport &transport) {
  return multiply_transpose_checked(transport.from_reference, input_image);
}

std::expected<HoloLiftLatticeImage, std::string>
hololift_lattice_image_from_reference(
    const HoloLiftLatticeImage &reference_image,
    const HoloLiftLatticeBasisTransport &transport) {
  return multiply_transpose_checked(transport.to_reference, reference_image);
}

std::array<double, 3> hololift_fractional_to_reference(
    const std::array<double, 3> &input_fractional,
    const HoloLiftLatticeBasisTransport &transport) noexcept {
  std::array<double, 3> result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t inner = 0; inner < 3; ++inner) {
      result[row] +=
          static_cast<double>(transport.from_reference[3 * inner + row]) *
          input_fractional[inner];
    }
  }
  return result;
}

} // namespace titan_hololift

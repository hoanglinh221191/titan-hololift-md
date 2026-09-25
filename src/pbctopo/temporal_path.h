#ifndef TITAN_PBCTOPO_TEMPORAL_PATH_H
#define TITAN_PBCTOPO_TEMPORAL_PATH_H

#include "types.h"

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace titan_pbctopo {

struct PbctopoTemporalCandidate {
  std::array<double, 3> center{1.5, 1.5, 1.5};
  double emission_cost = 0.0;
  std::array<double, 6> normalized_gram{1.0, 0.0, 0.0, 1.0, 0.0, 1.0};
  std::vector<Int3> component_offsets;
  std::vector<std::uint64_t> component_ids;
  std::vector<double> component_weights;
  std::uint64_t assignment_signature = 0;
  bool metric_available = false;
  bool certified = false;
};

struct PbctopoTemporalPathResult {
  bool valid = false;
  std::vector<std::size_t> selected;
  double total_cost = 0.0;
  std::size_t transition_count = 0;
};

double pbctopo_temporal_center_jump2(const std::array<double, 3> &lhs,
                                     const std::array<double, 3> &rhs);
std::array<double, 6> pbctopo_temporal_normalized_gram(
    const std::array<double, 9> &row_major_basis);
double pbctopo_temporal_assignment_jump2(
    const PbctopoTemporalCandidate &lhs,
    const PbctopoTemporalCandidate &rhs);

PbctopoTemporalPathResult solve_pbctopo_temporal_viterbi(
    const std::vector<std::vector<PbctopoTemporalCandidate>> &frames,
    double transition_penalty);

} // namespace titan_pbctopo

#endif

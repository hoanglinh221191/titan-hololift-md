#pragma once
// Histogram entropy estimators for the information-theoretic network measures
// (mutual information, transfer entropy) in dccm_network.cpp.
//
// KIN-14: probabilities are the plug-in frequencies c_i / N and every entropy
// term carries the Miller-Madow correction (m - 1) / (2 N) (m occupied cells,
// N samples; Miller 1955), the first-order bias of the plug-in estimator.
// The former 0.5 pseudocount per cell added 0.5 B, 0.5 B^2 and 0.5 B^3 fake
// counts to the 1-D, 2-D and 3-D histograms, a bias that depended on N, B and
// tau and did not cancel between the terms of TE.
#include <cmath>
#include <cstddef>
#include <vector>

namespace titan_info {

// Plug-in Shannon entropy (nats) of a histogram, p_i = c_i / sum(c).
inline double plugin_entropy(const double *counts, std::size_t cells) {
  double total = 0.0;
  for (std::size_t i = 0; i < cells; ++i)
    total += counts[i];
  if (!(total > 0.0))
    return 0.0;
  double h = 0.0;
  for (std::size_t i = 0; i < cells; ++i) {
    if (counts[i] > 0.0) {
      const double p = counts[i] / total;
      h -= p * std::log(p);
    }
  }
  return h;
}

inline double plugin_entropy(const std::vector<double> &counts) {
  return plugin_entropy(counts.data(), counts.size());
}

// Plug-in entropy plus the Miller-Madow correction (m - 1) / (2 N) of a count
// histogram (m = occupied cells, N = total count).
inline double miller_madow_entropy(const double *counts, std::size_t cells) {
  double total = 0.0;
  std::size_t occupied = 0;
  for (std::size_t i = 0; i < cells; ++i) {
    total += counts[i];
    occupied += counts[i] > 0.0 ? 1u : 0u;
  }
  if (!(total > 0.0))
    return 0.0;
  return plugin_entropy(counts, cells) +
         static_cast<double>(occupied - 1) / (2.0 * total);
}

inline double miller_madow_entropy(const std::vector<double> &counts) {
  return miller_madow_entropy(counts.data(), counts.size());
}

// Transfer entropy TE(src -> dst) in nats of two binned series (bins in
// [0, bins)), lag tau frames:
//   H(D_t, D_t-tau) + H(D_t-tau, S_t-tau) - H(D_t-tau) - H(D_t, D_t-tau, S_t-tau)
// over the frames - tau pairs, every term a Miller-Madow entropy. The same
// terms, histogram layouts and estimator as the network path in
// dccm_network.cpp.
template <typename BinType>
double transfer_entropy(const BinType *src, const BinType *dst,
                        std::size_t frames, std::size_t bins, std::size_t tau) {
  if (tau == 0 || frames <= tau || bins == 0)
    return 0.0;
  std::vector<double> h_auto(bins * bins, 0.0), h_cross(bins * bins, 0.0),
      h_past(bins, 0.0), h3(bins * bins * bins, 0.0);
  for (std::size_t f = tau; f < frames; ++f) {
    const std::size_t now = static_cast<std::size_t>(dst[f]);
    const std::size_t past = static_cast<std::size_t>(dst[f - tau]);
    const std::size_t src_past = static_cast<std::size_t>(src[f - tau]);
    h_auto[now * bins + past] += 1.0;
    h_cross[past * bins + src_past] += 1.0;
    h_past[past] += 1.0;
    h3[(now * bins + past) * bins + src_past] += 1.0;
  }
  return miller_madow_entropy(h_auto) + miller_madow_entropy(h_cross) -
         miller_madow_entropy(h_past) - miller_madow_entropy(h3);
}

} // namespace titan_info

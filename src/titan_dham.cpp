// ============================================================================
// TitanDHAM - helpers of the -dham input path (see titan_dham.h)
// ============================================================================
#include "titan_dham.h"
#include "titan_thread_compat.h"
#include <algorithm>
#include <stdexcept>
#include <thread>

// ============================================================================
// compute_state_energies (static)
// ============================================================================
Eigen::MatrixXd TitanDHAM::compute_state_energies(
    const std::unordered_map<int, std::vector<int>>&    per_replica_dtrajs,
    const std::unordered_map<int, std::vector<double>>& per_replica_energies,
    int n_states, int n_replicas, int num_threads)
{
    if(n_states<=0 || n_replicas<=0 || num_threads<=0) throw std::invalid_argument("DHAM state-energy dimensions and thread count must be positive");
    for(const auto& [k,traj]:per_replica_dtrajs) {
        if(k<0 || k>=n_replicas) throw std::out_of_range("DHAM trajectory replica id out of range");
        for(int state:traj) if(state < -1 || state>=n_states) throw std::out_of_range("DHAM state id out of range (-1 is the unsampled sentinel)");
        auto it=per_replica_energies.find(k);
        if(it!=per_replica_energies.end() && it->second.size()!=traj.size())
            throw std::invalid_argument("DHAM trajectory/energy lengths differ; align by frame time before computing state means");
    }
    for(const auto& [k,values]:per_replica_energies) {
        if(k<0 || k>=n_replicas) throw std::out_of_range("DHAM energy replica id out of range");
        for(double value:values) if(std::isinf(value)) throw std::invalid_argument("DHAM energy contains infinity");
    }
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_states, n_replicas);
    Eigen::MatrixXi counts  = Eigen::MatrixXi::Zero(n_states, n_replicas);

    // Opt 4: Each thread processes disjoint columns — zero locks needed
    auto worker = [&](int k_start, int k_end) {
        for (int k = k_start; k < k_end; ++k) {
            auto it_d = per_replica_dtrajs.find(k);
            auto it_e = per_replica_energies.find(k);
            if (it_d == per_replica_dtrajs.end() ||
                it_e == per_replica_energies.end()) continue;

            const auto& dtraj   = it_d->second;
            const auto& energies = it_e->second;

            size_t n_frames = dtraj.size();
            for (size_t t = 0; t < n_frames; ++t) {
                double e = energies[t];
                if (std::isnan(e)) continue;
                int s = dtraj[t];
                if (s < 0 || s >= n_states) continue;
                result(s, k) += e;
                counts(s, k) += 1;
            }
        }
    };

    int n_use = std::min(num_threads, n_replicas);
    if (n_use <= 1) {
        worker(0, n_replicas);
    } else {
    std::vector<titan::jthread> threads;
        threads.reserve(static_cast<size_t>(n_use));
        int chunk = (n_replicas + n_use - 1) / n_use;
        for (int t = 0; t < n_use; ++t) {
            int ks = t * chunk;
            int ke = std::min(ks + chunk, n_replicas);
            if (ks >= ke) continue;
            threads.emplace_back(worker, ks, ke);
        }
    }

    for (int i = 0; i < n_states; ++i) {
        for (int k = 0; k < n_replicas; ++k) {
            if (counts(i, k) > 0)
                result(i, k) /= static_cast<double>(counts(i, k));
            else
                result(i, k) = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return result;
}

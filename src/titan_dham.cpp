// ============================================================================
// TitanDHAM — Dynamic Histogram Analysis Method (Implementation)
// ============================================================================
#include "titan_dham.h"
#include "titan_markov_estimator.h"
#include "titan_thread_compat.h"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace {

std::uint64_t pack_transition_key(int state_i, int state_j, int n_states) noexcept {
    return static_cast<std::uint64_t>(state_i) *
               static_cast<std::uint64_t>(n_states) +
           static_cast<std::uint64_t>(state_j);
}

int unpack_transition_source(std::uint64_t key, int n_states) noexcept {
    return static_cast<int>(key / static_cast<std::uint64_t>(n_states));
}

int unpack_transition_dest(std::uint64_t key, int n_states) noexcept {
    return static_cast<int>(key % static_cast<std::uint64_t>(n_states));
}

} // namespace

// ============================================================================
// Constructor
// ============================================================================
TitanDHAM::TitanDHAM(int n_states, int n_replicas,
                       const std::vector<double>& temps,
                       double target_temp)
    : n_states_(n_states), n_replicas_(n_replicas)
{
    if(n_states<=0 || n_replicas<=0 || !std::isfinite(target_temp) || target_temp<=0)
        throw std::invalid_argument("DHAM dimensions and target temperature must be positive");
    for(double temp:temps) if(!std::isfinite(temp) || temp<=0)
        throw std::invalid_argument("DHAM replica temperatures must be finite and positive");
    if (static_cast<int>(temps.size()) != n_replicas) {
        TitanUI::print_error("TitanDHAM: temps.size() (" +
            std::to_string(temps.size()) + ") != n_replicas (" +
            std::to_string(n_replicas) + ").");
        throw std::invalid_argument("TitanDHAM: temps/n_replicas mismatch.");
    }

    beta_target_ = 1.0 / (DHAM_kB_kJ * target_temp);

    betas_.resize(n_replicas);
    for (int k = 0; k < n_replicas; ++k)
        betas_[k] = 1.0 / (DHAM_kB_kJ * temps[k]);

    C_k_sparse_.resize(n_replicas);

    c_out_  = MatrixXdRM::Zero(n_states, n_replicas);
    u_ik_   = MatrixXdRM::Constant(
        n_states, n_replicas, std::numeric_limits<double>::infinity());

    p_i_ = Eigen::VectorXd::Constant(n_states, 1.0 / n_states);
}

// ============================================================================
// accumulate_counts
// ============================================================================
void TitanDHAM::accumulate_counts(int replica_k, int state_i, int state_j) {
    if (replica_k < 0 || replica_k >= n_replicas_ ||
        state_i  < 0 || state_i  >= n_states_   ||
        state_j  < 0 || state_j  >= n_states_) throw std::out_of_range("DHAM transition replica/state index out of range");

    const std::uint64_t key =
        pack_transition_key(state_i, state_j, n_states_);
    auto& counts = C_k_sparse_[static_cast<size_t>(replica_k)];
    counts[key] += 1.0;
    c_out_(state_i, replica_k) += 1.0;
    solved_ = false;
}

// ============================================================================
// set_state_energy
// ============================================================================
void TitanDHAM::set_state_energy(int replica_k, int state_i,
                                   double mean_energy_kJ) {
    if (replica_k < 0 || replica_k >= n_replicas_ ||
        state_i  < 0 || state_i  >= n_states_) throw std::out_of_range("DHAM energy replica/state index out of range");
    if(!std::isfinite(mean_energy_kJ)) throw std::invalid_argument("DHAM state energy must be finite");
    // u_ik(i,k) = beta_k * U_k(i)  (dimensionless reduced potential)
    u_ik_(state_i, static_cast<int>(replica_k)) =
        betas_[static_cast<size_t>(replica_k)] * mean_energy_kJ;
    solved_ = false;
}

// ============================================================================
// rebuild_c_out — row sums of sparse count maps
// ============================================================================
void TitanDHAM::rebuild_c_out() {
    c_out_.setZero();
    for (int k = 0; k < n_replicas_; ++k) {
        for (const auto& entry : C_k_sparse_[static_cast<size_t>(k)]) {
            const int state_i = unpack_transition_source(entry.first, n_states_);
            if (state_i >= 0 && state_i < n_states_)
                c_out_(state_i, k) += entry.second;
        }
    }
}

// Explicit small-bias DHAM approximation (Rosta & Hummer, 2015):
// M_ij ~ sum_k C^k_ij / sum_k n^k_i exp[-(b^k_j-b^k_i)/2], followed
// by row normalization. b^k_i=(beta_k-beta_target)*U^k_i uses the available
// state-mean energy approximation; it is not a general H-REMD cross-energy map.
bool TitanDHAM::solve(int max_iter, double tol) {
    solved_ = false;
    target_tpm_.resize(0,0);
    if (max_iter<=0 || !std::isfinite(tol) || tol<=0) return false;
    rebuild_c_out();
    Eigen::MatrixXd total=Eigen::MatrixXd::Zero(n_states_,n_states_);
    for(const auto &replica:C_k_sparse_) for(const auto &[key,count]:replica)
        total(unpack_transition_source(key,n_states_),unpack_transition_dest(key,n_states_))+=count;
    if(!titan_markov::connected(total) || total.sum()<=0) {
        TitanUI::print_error("DHAM requires lag counts on one strongly connected visited set.");
        return false;
    }
    target_tpm_=Eigen::MatrixXd::Zero(n_states_,n_states_);
    for(int i=0;i<n_states_;++i) {
        Eigen::VectorXd row_log=Eigen::VectorXd::Constant(n_states_,-std::numeric_limits<double>::infinity());
        for(int j=0;j<n_states_;++j) {
            if(total(i,j)<=0) continue;
            double max_log=-std::numeric_limits<double>::infinity();
            std::vector<double> terms;
            for(int k=0;k<n_replicas_;++k) {
                if(c_out_(i,k)<=0) continue;
                const double factor=1.0-beta_target_/betas_[k];
                double bias_difference=0.0;
                if(std::abs(factor)>1e-14) {
                    if(!std::isfinite(u_ik_(i,k)) || !std::isfinite(u_ik_(j,k))) {
                        TitanUI::print_error("DHAM lacks state energies needed for a sampled transition bias denominator.");
                        target_tpm_.resize(0,0);return false;
                    }
                    bias_difference=factor*(u_ik_(j,k)-u_ik_(i,k));
                }
                const double term=std::log(c_out_(i,k))-0.5*bias_difference;
                if(!std::isfinite(term)) {target_tpm_.resize(0,0);return false;}
                terms.push_back(term);max_log=std::max(max_log,term);
            }
            if(terms.empty()) {target_tpm_.resize(0,0);return false;}
            double sum=0;for(double term:terms)sum+=std::exp(term-max_log);
            row_log[j]=std::log(total(i,j))-max_log-std::log(sum);
        }
        const double maximum=row_log.maxCoeff();
        if(!std::isfinite(maximum)) {target_tpm_.resize(0,0);return false;}
        target_tpm_.row(i)=(row_log.array()-maximum).exp().matrix().transpose();
        target_tpm_.row(i)/=target_tpm_.row(i).sum();
    }
    std::string error;
    if(!titan_markov::stationary(target_tpm_,p_i_,error)) {
        TitanUI::print_error("DHAM: "+error);target_tpm_.resize(0,0);return false;
    }
    if((target_tpm_.transpose()*p_i_-p_i_).cwiseAbs().maxCoeff()>tol)
        return false;
    f_k_=Eigen::VectorXd::Constant(n_replicas_,std::numeric_limits<double>::quiet_NaN());
    for(int k=0;k<n_replicas_;++k) {
        const double factor=1.0-beta_target_/betas_[k];
        Eigen::VectorXd terms(n_states_);
        bool complete=true;
        for(int i=0;i<n_states_;++i) {
            if(std::abs(factor)>1e-14 && !std::isfinite(u_ik_(i,k))) {complete=false;break;}
            terms[i]=std::log(p_i_[i])-(std::abs(factor)>1e-14 ? factor*u_ik_(i,k) : 0.0);
        }
        if(complete) {const double m=terms.maxCoeff();f_k_[k]=-m-std::log((terms.array()-m).exp().sum());}
    }
    if(std::isfinite(f_k_[0])) {const double anchor=f_k_[0];f_k_.array()-=anchor;}
    solved_=true;
    TitanUI::print_info("DHAM: explicit state-energy bias approximation; stationary residual verified.");
    return true;
}

Eigen::MatrixXd TitanDHAM::get_unbiased_transition_matrix() const {
    if(!solved_) throw std::runtime_error("DHAM transition matrix requires an accepted solve");
    return target_tpm_;
}

// ============================================================================
// get_free_energy_profile
// ============================================================================
Eigen::VectorXd TitanDHAM::get_free_energy_profile() const {
    if (!solved_) return Eigen::VectorXd::Zero(n_states_);

    const double kT = 1.0 / beta_target_;  // kJ/mol
    Eigen::VectorXd G(n_states_);

    double p_max = p_i_.maxCoeff();
    if (p_max < 1e-300) return Eigen::VectorXd::Zero(n_states_);

    for (int i = 0; i < n_states_; ++i) {
        double p = p_i_(i);
        if (p < 1e-300)
            G(i) = std::numeric_limits<double>::infinity();
        else
            G(i) = -kT * std::log(p / p_max);
    }
    return G;
}

// ============================================================================
// get_frame_weights
// ============================================================================
std::unordered_map<int, std::vector<double>>
TitanDHAM::get_frame_weights(
    const std::unordered_map<int, std::vector<int>>& per_replica_dtrajs) const {
    if (!solved_) {
        TitanUI::print_error(
            "TitanDHAM::get_frame_weights: solve() must run first.");
        return {};
    }

    std::vector<double> N_i(static_cast<size_t>(n_states_), 0.0);
    for (const auto& entry : per_replica_dtrajs) {
        for (int s : entry.second) {
            if (s >= 0 && s < n_states_)
                N_i[static_cast<size_t>(s)] += 1.0;
        }
    }

    std::unordered_map<int, std::vector<double>> result;
    result.reserve(per_replica_dtrajs.size());
    for (const auto& entry : per_replica_dtrajs) {
        std::vector<double> weights(entry.second.size(), 0.0);
        for (size_t t = 0; t < entry.second.size(); ++t) {
            const int s = entry.second[t];
            if (s < 0 || s >= n_states_)
                continue;
            const double Ni = N_i[static_cast<size_t>(s)];
            if (Ni > 0.0)
                weights[t] = p_i_(s) / Ni;
        }
        result.emplace(entry.first, std::move(weights));
    }
    return result;
}

// ============================================================================
// get_flattened_frame_weights
// ============================================================================
std::vector<double> TitanDHAM::get_flattened_frame_weights(
    const std::unordered_map<int, std::vector<int>>& per_replica_dtrajs,
    int n_replicas) const {
    auto per_replica_weights = get_frame_weights(per_replica_dtrajs);

    size_t total = 0;
    for (int k = 0; k < n_replicas; ++k) {
        auto it = per_replica_dtrajs.find(k);
        if (it != per_replica_dtrajs.end())
            total += it->second.size();
    }

    std::vector<double> flat;
    flat.reserve(total);
    for (int k = 0; k < n_replicas; ++k) {
        auto it = per_replica_weights.find(k);
        if (it == per_replica_weights.end())
            continue;
        flat.insert(flat.end(), it->second.begin(), it->second.end());
    }
    return flat;
}

// ============================================================================
// write_results
// ============================================================================
void TitanDHAM::write_results(const std::string& pi_file,
                                 const std::string& fes_file,
                                const std::string& tpm_file,
                                std::fstream& log,
                                double eu_par) const {
    if (!solved_) {
        TitanUI::print_warning("TitanDHAM::write_results: solve() was not called.");
        return;
    }

    // Stationary distribution
    {
        std::ofstream f(pi_file);
        if (f.is_open()) {
            f << "# DHAM Reweighted Stationary Distribution at Target Temperature\n";
            f << "# state_id  p_i\n";
            for (int i = 0; i < n_states_; ++i)
                f << i << "  " << std::fixed << std::setprecision(8) << p_i_(i) << "\n";
            TitanUI::print_info("DHAM: stationary distribution written to " + pi_file);
            TitanUI::log_kv(log, "DHAM Pi File", pi_file);
        }
    }

    // Free energy profile
    {
        Eigen::VectorXd G = get_free_energy_profile();
        std::string unit_label = (eu_par > 1.1) ? "kcal/mol" : "kJ/mol";
        std::ofstream f(fes_file);
        if (f.is_open()) {
            f << "# DHAM Reweighted Free Energy Profile (" << unit_label << ", min=0)\n";
            f << "# state_id  G_" << unit_label << "\n";
            for (int i = 0; i < n_states_; ++i) {
                f << i << "  " << std::fixed << std::setprecision(4);
                if (std::isinf(G(i)))
                    f << "inf";
                else
                    f << G(i) / eu_par;
                f << "\n";
            }
            TitanUI::print_info("DHAM: free energy profile (" + unit_label +
                                ") written to " + fes_file);
            TitanUI::log_kv(log, "DHAM FES File", fes_file);
        }
    }

    // Unbiased transition matrix
    {
        Eigen::MatrixXd T = get_unbiased_transition_matrix();
        std::ofstream f(tpm_file);
        if (f.is_open()) {
            f << "# DHAM state-energy small-bias approximation at target temperature\n";
            f << "# " << n_states_ << "x" << n_states_ << "\n";
            f << std::fixed << std::setprecision(8);
            for (int i = 0; i < n_states_; ++i) {
                for (int j = 0; j < n_states_; ++j) {
                    if (j > 0) f << " ";
                    f << T(i, j);
                }
                f << "\n";
            }
            TitanUI::print_info("DHAM: unbiased TPM written to " + tpm_file);
            TitanUI::log_kv(log, "DHAM TPM File", tpm_file);
        }
    }
}

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

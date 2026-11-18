#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace titan_markov {

inline bool connected(const Eigen::MatrixXd &matrix) {
  const int n = static_cast<int>(matrix.rows());
  if (n == 0 || matrix.cols() != n) return false;
  for (bool reverse : {false, true}) {
    std::vector<unsigned char> seen(n, 0);
    std::vector<int> stack{0}; seen[0] = 1;
    while (!stack.empty()) {
      const int i=stack.back();stack.pop_back();
      for(int j=0;j<n;++j) if(!seen[j] && (reverse ? matrix(j,i) : matrix(i,j))>0) {
        seen[j]=1;stack.push_back(j);
      }
    }
    if (std::find(seen.begin(),seen.end(),0)!=seen.end()) return false;
  }
  return true;
}

inline bool stationary(const Eigen::MatrixXd &tpm, Eigen::VectorXd &pi,
                       std::string &error) {
  pi.resize(0);
  const int n=static_cast<int>(tpm.rows());
  if(n==0 || tpm.cols()!=n || !tpm.allFinite() || tpm.minCoeff()<0 ||
     (tpm.rowwise().sum().array()-1).abs().maxCoeff()>1e-10 || !connected(tpm)) {
    error="stationary distribution requires a finite stochastic matrix on one strongly connected active set";
    return false;
  }
  Eigen::MatrixXd system=tpm.transpose()-Eigen::MatrixXd::Identity(n,n);
  system.row(n-1).setOnes();
  Eigen::VectorXd rhs=Eigen::VectorXd::Zero(n);rhs[n-1]=1;
  Eigen::FullPivLU<Eigen::MatrixXd> lu(system);
  if(lu.rank()!=n) {error="stationary system is numerically rank deficient";return false;}
  pi=lu.solve(rhs);
  if(!pi.allFinite() || pi.minCoeff() < -1e-10 || pi.sum()<=0) {
    error="invalid stationary solution";pi.resize(0);return false;
  }
  pi=pi.cwiseMax(0);pi/=pi.sum();
  if((tpm.transpose()*pi-pi).cwiseAbs().maxCoeff()>1e-10) {
    error="stationary residual exceeds tolerance";pi.resize(0);return false;
  }
  return true;
}

// Reversible maximum-likelihood flux fixed point:
// x_ij=(c_ij+c_ji)/(c_i/pi_i+c_j/pi_j), pi_i=sum_j x_ij.
// Reject unvisited/disconnected sets rather than inventing transition rows.
inline bool reversible_mle(const Eigen::MatrixXd &input, Eigen::MatrixXd &tpm,
                           Eigen::VectorXd &pi, std::string &error,
                           int max_iterations=100000, double tol=1e-11) {
  tpm.resize(0,0);pi.resize(0);
  const int n=static_cast<int>(input.rows());
  if(n==0 || input.cols()!=n || !input.allFinite() || input.minCoeff()<0 ||
     input.sum()<=0 || !connected(input)) {
    error="reversible MSM requires counts on one strongly connected visited set; reduce clusters or select that set";
    return false;
  }
  const Eigen::MatrixXd counts=input/input.sum();
  const Eigen::VectorXd row=counts.rowwise().sum();
  if(row.minCoeff()<=0) {error="unvisited MSM count row";return false;}
  const Eigen::MatrixXd symmetric=(counts+counts.transpose()).eval();
  pi=symmetric.rowwise().sum();pi/=pi.sum();
  Eigen::MatrixXd flux(n,n);
  for(int it=0;it<max_iterations;++it) {
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)
      flux(i,j)=symmetric(i,j)/(row[i]/pi[i]+row[j]/pi[j]);
    Eigen::VectorXd next=flux.rowwise().sum();
    const double total=next.sum();
    if(!next.allFinite() || next.minCoeff()<=0 || total<=0) break;
    next/=total;
    const double delta=((next-pi).array().abs()/pi.array().max(1e-15)).maxCoeff();
    pi=next;
    if(delta<tol) {
      const Eigen::VectorXd sums=flux.rowwise().sum();
      tpm=flux.array().colwise()/sums.array();
      pi=sums/sums.sum();
      if((tpm.transpose()*pi-pi).cwiseAbs().maxCoeff()<=1e-10) return true;
      break;
    }
  }
  error="reversible MSM maximum-likelihood iteration did not converge";
  tpm.resize(0,0);pi.resize(0);return false;
}

// KIN-09: spectrum of a transition matrix in detailed balance with pi.
// S = D^{1/2} T D^{-1/2} (D = diag pi) is symmetric, so its eigenvalues are
// real and its eigenvectors u orthonormal; the right eigenvectors of T are
// D^{-1/2} u (pi-orthogonal, as PCCA+ needs), returned at unit 2-norm like
// those of a general eigensolver. Returns false (caller falls back to the
// general solver) when pi is not positive or detailed balance does not hold
// to rel_tol.
inline bool reversible_spectrum(const Eigen::MatrixXd &tpm,
                                const Eigen::VectorXd &pi,
                                Eigen::VectorXd &eigenvalues,
                                Eigen::MatrixXd *right_eigenvectors,
                                double rel_tol=1e-10) {
  const int n=static_cast<int>(tpm.rows());
  if(n==0 || tpm.cols()!=n || pi.size()!=n || !tpm.allFinite() ||
     !pi.allFinite() || pi.minCoeff()<=0) return false;
  double flux_max=0, asymmetry=0;
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
    flux_max=std::max(flux_max,std::abs(pi[i]*tpm(i,j)));
    asymmetry=std::max(asymmetry,std::abs(pi[i]*tpm(i,j)-pi[j]*tpm(j,i)));
  }
  if(flux_max<=0 || asymmetry>rel_tol*flux_max) return false;
  const Eigen::VectorXd root=pi.cwiseSqrt();
  Eigen::MatrixXd s=root.asDiagonal()*tpm*root.cwiseInverse().asDiagonal();
  s=(0.5*(s+s.transpose())).eval();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      s, right_eigenvectors!=nullptr ? Eigen::ComputeEigenvectors
                                     : Eigen::EigenvaluesOnly);
  if(solver.info()!=Eigen::Success) return false;
  eigenvalues=solver.eigenvalues();
  if(right_eigenvectors!=nullptr) {
    *right_eigenvectors=root.cwiseInverse().asDiagonal()*solver.eigenvectors();
    for(int k=0;k<n;++k) {
      const double norm=right_eigenvectors->col(k).norm();
      if(!(norm>0)) return false;
      right_eigenvectors->col(k)/=norm;
    }
  }
  return true;
}
} // namespace titan_markov

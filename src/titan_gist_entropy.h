#pragma once
// First-order orientational NN entropy on SO(3), relative to uniform Haar
// measure. Labeled O,H1,H2 convention matches the usual GIST estimator.
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace titan_gist {
using Quaternion = std::array<double,4>; // w,x,y,z
inline Quaternion orientation(const std::array<double,3>& o,const std::array<double,3>& h1,const std::array<double,3>& h2) {
 Eigen::Vector3d x(h1[0]-o[0],h1[1]-o[1],h1[2]-o[2]);
 Eigen::Vector3d y(h2[0]-o[0],h2[1]-o[1],h2[2]-o[2]);
 if(!x.allFinite()||!y.allFinite()||x.norm()<1e-8||y.norm()<1e-8)throw std::invalid_argument("GIST invalid water orientation");
 x.normalize();y-=x*x.dot(y);if(y.norm()<1e-8)throw std::invalid_argument("GIST collinear water orientation");y.normalize();
 Eigen::Matrix3d r;r.col(0)=x;r.col(1)=y;r.col(2)=x.cross(y);Eigen::Quaterniond q(r);q.normalize();
 return {q.w(),q.x(),q.y(),q.z()};
}
// Haar probability of an SO(3) angular ball, radius theta in [0,pi].
inline double ball_probability(double theta) {
 if(!std::isfinite(theta)||theta<0||theta>std::acos(-1.)+1e-12)throw std::invalid_argument("Invalid SO(3) angular radius");
 const double t2=theta*theta;
 const double volume=theta<1e-3 ? theta*t2*(1./6.-t2/120.+t2*t2/5040.) : theta-std::sin(theta);
 return volume/std::acos(-1.);
}
// Exact balanced KD search in R4, querying q and -q for antipodal equivalence.
// No distance approximation or random subsampling is used.
class QuaternionTree {
 const std::vector<Quaternion>& q_;std::vector<std::size_t> indices_;
 void build(std::size_t lo,std::size_t hi,int depth){if(lo>=hi)return;auto mid=(lo+hi)/2;int axis=depth%4;std::nth_element(indices_.begin()+lo,indices_.begin()+mid,indices_.begin()+hi,[&](auto i,auto j){return q_[i][axis]<q_[j][axis];});build(lo,mid,depth+1);build(mid+1,hi,depth+1);}
 void query(std::size_t lo,std::size_t hi,int depth,const Quaternion& p,std::size_t omit,double& best) const {
  if(lo>=hi||best==0.)return;auto mid=(lo+hi)/2,i=indices_[mid];double d=0;for(int k=0;k<4;++k)d+=(p[k]-q_[i][k])*(p[k]-q_[i][k]);if(i!=omit)best=std::min(best,d);
  double delta=p[depth%4]-q_[i][depth%4];if(delta<0){query(lo,mid,depth+1,p,omit,best);if(delta*delta<=best)query(mid+1,hi,depth+1,p,omit,best);}else{query(mid+1,hi,depth+1,p,omit,best);if(delta*delta<=best)query(lo,mid,depth+1,p,omit,best);}
 }
public:
 explicit QuaternionTree(const std::vector<Quaternion>& q):q_(q),indices_(q.size()) {
  for(const auto& p:q){double norm=0;for(double a:p){if(!std::isfinite(a))throw std::invalid_argument("Nonfinite GIST quaternion");norm+=a*a;}if(std::abs(norm-1)>1e-8)throw std::invalid_argument("Non-unit GIST quaternion");}
  std::iota(indices_.begin(),indices_.end(),0);build(0,q.size(),0);
 }
 double nearest_angle(std::size_t i) const {double best=std::numeric_limits<double>::infinity();query(0,q_.size(),0,q_[i],i,best);Quaternion minus=q_[i];for(auto& a:minus)a=-a;query(0,q_.size(),0,minus,i,best);return 4*std::asin(std::clamp(std::sqrt(best)/2,0.,std::sqrt(.5)));}
};
struct EntropyResult {double s_over_R=std::numeric_limits<double>::quiet_NaN();std::size_t samples=0,zero_neighbors=0;};
inline EntropyResult orientational_entropy(const std::vector<Quaternion>& q) {
 EntropyResult out;out.samples=q.size();if(q.size()<2)return out;QuaternionTree tree(q);double sum=0;
 for(std::size_t i=0;i<q.size();++i){double probability=ball_probability(tree.nearest_angle(i));if(!(probability>1e-30)){++out.zero_neighbors;continue;}sum+=std::log(q.size()*probability);}
 // Repeated/quantized orientations do not define a continuous NN entropy.
 // Preserve NaN rather than silently ignoring them or imposing an angle floor.
 if(!out.zero_neighbors)out.s_over_R=sum/q.size()+0.5772156649015328606;
 return out;
}
}

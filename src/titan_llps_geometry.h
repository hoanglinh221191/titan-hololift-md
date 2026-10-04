#pragma once
#include "pbctopo/lattice_math.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace titan_llps_geometry {
using Point = std::array<double, 3>;
struct Cell {
  std::array<double, 9> matrix{}; // row major, columns are lattice vectors, nm
  Point heights{};
  double volume = 0;
  std::optional<titan_pbctopo::PbctopoLatticeMetric> metric;
  Cell() = default;
  explicit Cell(const std::array<double, 9>& b) : matrix(b), metric(std::in_place, std::span<const double,9>(matrix)) {
    if (!metric->valid()) throw std::invalid_argument("LLPS: invalid periodic lattice");
    volume = std::abs(b[0]*(b[4]*b[8]-b[5]*b[7])-b[1]*(b[3]*b[8]-b[5]*b[6])+b[2]*(b[3]*b[7]-b[4]*b[6]));
    auto x=metric->cartesian_to_scaled(1,0,0), y=metric->cartesian_to_scaled(0,1,0), z=metric->cartesian_to_scaled(0,0,1);
    for (int i=0;i<3;++i) heights[i]=1/std::sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
  }
  static Cell from_box(box b, double x, double y, double z) {
    if (b.get_a()==0 && b.get_b()==0 && b.get_c()==0 && b.is_orthorhombic()) {
      if (x==0 && y==0 && z==0) return {};
      if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x>0 && y>0 && z>0))
        throw std::invalid_argument("LLPS: fallback cell lengths must be finite and positive");
      return Cell({x,0,0,0,y,0,0,0,z});
    }
    double a[3],c[3],d[3]; b.get_v1(a); b.get_v2(c); b.get_v3(d);
    return Cell({a[0]*.1,c[0]*.1,d[0]*.1,a[1]*.1,c[1]*.1,d[1]*.1,a[2]*.1,c[2]*.1,d[2]*.1});
  }
  Point fractional(Point p) const {return metric ? metric->cartesian_to_scaled(p[0],p[1],p[2]) : p;}
  Point cartesian(Point p) const {return metric ? metric->scaled_to_cartesian(p[0],p[1],p[2]) : p;}
  Point delta(Point p, bool* ambiguous=nullptr) const {
    for(double v:p) if(!std::isfinite(v)) throw std::invalid_argument("LLPS: non-finite coordinate");
    if (!metric) return p;
    const auto q=metric->nearest_image_cartesian(p[0],p[1],p[2]);
    if (!q.valid || q.node_budget_exhausted) throw std::runtime_error("LLPS: nearest-image search failed");
    if(ambiguous) *ambiguous=q.ambiguous;
    return {q.x,q.y,q.z};
  }
  void grid_coordinate(double& x,double& y,double& z) const {
    if (!metric) return;
    auto s=fractional({x,y,z});
    x=(s[0]-std::floor(s[0]))*heights[0]; y=(s[1]-std::floor(s[1]))*heights[1]; z=(s[2]-std::floor(s[2]))*heights[2];
  }
};
struct PhaseConfig {
  std::string geometry="none";
  int axis=2; // normal to fractional a/b/c planes, not Cartesian x/y/z
  double dense=0, dilute_inner=0, dilute_outer=0;
  void validate() const {
    if(geometry!="none" && geometry!="droplet" && geometry!="slab") throw std::invalid_argument("LLPS: phase geometry must be none, droplet or slab");
    if(geometry=="none") return;
    if(axis<0 || axis>2 || !std::isfinite(dense) || !std::isfinite(dilute_inner) || !std::isfinite(dilute_outer) || !(dense>0 && dilute_inner>dense && dilute_outer>dilute_inner))
      throw std::invalid_argument("LLPS phase windows require 0 < dense radius/half-width < dilute inner < dilute outer (nm)");
  }
};
struct PhaseResult {
  double dense=std::numeric_limits<double>::quiet_NaN(), dilute=dense;
  double dense_volume=0, dilute_volume=0;
  int dense_count=0, dilute_count=0;
  std::string status="not_requested";
};
inline std::optional<Point> circular_center(const Cell& cell,const std::vector<Point>& points) {
  if(points.empty() || !cell.metric) return {};
  Point center{}; constexpr double twopi=6.2831853071795864769;
  for(int k=0;k<3;++k) {
    double c=0,s=0; for(auto p:points) {double t=twopi*cell.fractional(p)[k]; c+=std::cos(t); s+=std::sin(t);}
    if(std::hypot(c,s)<1e-7*points.size()) return {};
    center[k]=std::atan2(s,c)/twopi;
  }
  return cell.cartesian(center);
}
inline PhaseResult measure(const Cell& cell,const PhaseConfig& cfg,const std::vector<Point>& all,
                           const std::vector<Point>& gcc,const std::optional<Point>& embedded_center) {
  cfg.validate(); PhaseResult r;
  if(cfg.geometry=="none") return r;
  if(!cell.metric) {r.status="no_periodic_cell"; return r;}
  const double limit=.5*(cfg.geometry=="slab" ? cell.heights[cfg.axis] : *std::min_element(cell.heights.begin(),cell.heights.end()));
  if(!(cfg.dilute_outer<limit)) {r.status="window_exceeds_cell"; return r;}
  Point center{};
  if(cfg.geometry=="droplet") {
    if(!embedded_center) {r.status="gcc_embedding_ambiguous"; return r;}
    center=*embedded_center;
    constexpr double factor=4.1887902047863909846;
    r.dense_volume=factor*std::pow(cfg.dense,3);
    r.dilute_volume=factor*(std::pow(cfg.dilute_outer,3)-std::pow(cfg.dilute_inner,3));
  } else {
    if(gcc.empty()) {r.status="empty_gcc"; return r;}
    double c=0,s=0; constexpr double twopi=6.2831853071795864769;
    for(auto p:gcc) {double t=twopi*cell.fractional(p)[cfg.axis]; c+=std::cos(t);s+=std::sin(t);}
    if(std::hypot(c,s)<1e-7*gcc.size()) {r.status="slab_center_ambiguous";return r;}
    center[cfg.axis]=std::atan2(s,c)/twopi;
    r.dense_volume=cell.volume/cell.heights[cfg.axis]*2*cfg.dense;
    r.dilute_volume=cell.volume/cell.heights[cfg.axis]*2*(cfg.dilute_outer-cfg.dilute_inner);
  }
  for(auto p:all) {
    double d;
    if(cfg.geometry=="droplet") {auto q=cell.delta({p[0]-center[0],p[1]-center[1],p[2]-center[2]});d=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);}
    else {double f=cell.fractional(p)[cfg.axis]-center[cfg.axis];d=std::abs(f-std::round(f))*cell.heights[cfg.axis];}
    if(d<cfg.dense) ++r.dense_count;
    if(d>=cfg.dilute_inner && d<cfg.dilute_outer) ++r.dilute_count;
  }
  r.dense=r.dense_count/r.dense_volume;r.dilute=r.dilute_count/r.dilute_volume;r.status="ok";return r;
}
} // namespace titan_llps_geometry

#pragma once
#include "BoxMatrix.h"
#include "pbctopo/lattice_math.h"
#include <array>
#include <stdexcept>
#include <algorithm>
#include "titan_neighbor_grid.h"

// Angstrom adapter for analyzers that historically used BoxMatrix. Keep the
// legacy matrix storage convention; use the checked lattice solver for MIC.
struct TitanExactPeriodicBox : BoxMatrix {
  titan_pbctopo::PbctopoLatticeMetric metric;
  std::array<double,3> heights{};
  explicit TitanExactPeriodicBox(const box& b) : BoxMatrix(b), metric(b) {
    if(!metric.valid()) throw std::invalid_argument("Analyzer requires a valid periodic lattice");
    for(int k=0;k<3;++k) {
      double sum=0;for(int j=0;j<3;++j)sum+=inv[j][k]*inv[j][k];
      heights[k]=1/std::sqrt(sum);
    }
  }
  void apply_mic(double& x,double& y,double& z) const {
    const auto q=metric.nearest_image_cartesian(x,y,z);
    if(!q.valid || q.node_budget_exhausted) throw std::runtime_error("Analyzer nearest lattice image failed");
    x=q.x;y=q.y;z=q.z;
  }
  void grid_coordinate(double& x,double& y,double& z) const {
    auto f=metric.cartesian_to_scaled(x,y,z);
    for(double value:f)if(!std::isfinite(value))throw std::invalid_argument("Analyzer non-finite fractional coordinate");
    x=(f[0]-std::floor(f[0]))*heights[0];
    y=(f[1]-std::floor(f[1]))*heights[1];
    z=(f[2]-std::floor(f[2]))*heights[2];
  }
  double contained_radius() const {return .5**std::min_element(heights.begin(),heights.end());}
};

// Finite embedding of a connected radius graph. Nonzero winding or a nearest
// image tie has no unique finite-cluster shape under this observation model.
inline bool titan_lift_radius_graph(const TitanExactPeriodicBox& cell,
    const std::vector<std::array<double,3>>& points,double cutoff,
    std::vector<std::array<double,3>>& lifted) {
  if(points.empty()) {lifted.clear();return true;}
  std::vector<std::vector<std::size_t>> edges(points.size());
  titan_neighbor::HalfShellGrid grid;titan_neighbor::OrthorhombicGridSpec spec;
  spec.cutoff=cutoff;spec.len_x=cell.heights[0];spec.len_y=cell.heights[1];spec.len_z=cell.heights[2];spec.periodic=true;
  grid.for_each_pair(points.size(),spec,[&](std::size_t i,double& x,double& y,double& z){x=points[i][0];y=points[i][1];z=points[i][2];cell.grid_coordinate(x,y,z);},
    [](std::size_t){return true;},[&](std::size_t i,std::size_t j){double x=points[j][0]-points[i][0],y=points[j][1]-points[i][1],z=points[j][2]-points[i][2];cell.apply_mic(x,y,z);if(x*x+y*y+z*z<=cutoff*cutoff){edges[i].push_back(j);edges[j].push_back(i);}});
  lifted.resize(points.size());lifted[0]=points[0];std::vector<bool> seen(points.size());seen[0]=true;std::vector<std::size_t> queue{0};
  for(std::size_t n=0;n<queue.size();++n) {
    auto i=queue[n];for(auto j:edges[i]) {
      auto q=cell.metric.nearest_image_cartesian(points[j][0]-points[i][0],points[j][1]-points[i][1],points[j][2]-points[i][2]);
      if(!q.valid || q.ambiguous || q.node_budget_exhausted)return false;
      std::array<double,3> next{lifted[i][0]+q.x,lifted[i][1]+q.y,lifted[i][2]+q.z};
      if(!seen[j]){seen[j]=true;lifted[j]=next;queue.push_back(j);}
      else for(int k=0;k<3;++k)if(std::abs(lifted[j][k]-next[k])>1e-7*std::max(1.0,cutoff))return false;
    }
  }
  return queue.size()==points.size();
}

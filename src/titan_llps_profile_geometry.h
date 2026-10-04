#pragma once
#include "titan_exact_periodic_box.h"
#include <string>
#include <numbers>

struct TitanLLPSProfileGeometry {
 bool valid=false;
 std::string geometry,reason;
 double limit_nm=0,radius_nm=0,cross_section_nm2=0,dense_envelope_nm3=0;
 std::vector<double> distance_nm;
 static TitanLLPSProfileGeometry build(const box& Box,
     const std::vector<std::array<double,3>>& points,const std::vector<bool>& dense,
     std::string geometry,int axis,double cutoff_a,double bin_nm) {
  TitanLLPSProfileGeometry out;out.geometry=geometry;
  if(points.size()!=dense.size()||axis<0||axis>2||!std::isfinite(bin_nm)||bin_nm<=0) {out.reason="invalid_profile_input";return out;}
  TitanExactPeriodicBox cell(Box);std::vector<std::array<double,3>> selected;
  for(std::size_t i=0;i<points.size();++i) {
   for(double x:points[i])if(!std::isfinite(x)){out.reason="nonfinite_coordinate";return out;}
   if(dense[i])selected.push_back(points[i]);
  }
  if(selected.empty()){out.reason="no_dense_nodes";return out;}
  if(geometry=="auto"&&!Box.is_orthorhombic()){out.reason="triclinic_requires_explicit_geometry";return out;}
  std::vector<std::array<double,3>> lifted;
  bool finite=false;
  if(geometry!="slab")finite=titan_lift_radius_graph(cell,selected,cutoff_a,lifted);
  if(geometry=="auto") {
   if(!finite)geometry="slab";
   else {
    auto lo=lifted.front(),hi=lo;
    for(const auto& p:lifted)for(int d=0;d<3;++d){lo[d]=std::min(lo[d],p[d]);hi[d]=std::max(hi[d],p[d]);}
    double largest=0,smallest=std::numeric_limits<double>::infinity();
    for(int d=0;d<3;++d){largest=std::max(largest,hi[d]-lo[d]);smallest=std::min(smallest,hi[d]-lo[d]);}
    geometry=(hi[axis]-lo[axis]>.55*cell.heights[axis]||largest/std::max(.01,smallest)>2.2)?"slab":"droplet";
   }
  }
  out.geometry=geometry;std::array<double,3> center{};double center_fraction=0;
  if(geometry=="droplet") {
   if(!finite){out.reason="nonfinite_or_ambiguous_dense_embedding";return out;}
   for(const auto& p:lifted)for(int d=0;d<3;++d)center[d]+=p[d]/lifted.size();
   out.limit_nm=cell.contained_radius()/10.;
  } else if(geometry=="slab") {
   double cosine=0,sine=0;
   for(const auto& p:selected){double f=cell.metric.cartesian_to_scaled(p[0],p[1],p[2])[axis];f-=std::floor(f);const double angle=2*std::numbers::pi*f;cosine+=std::cos(angle);sine+=std::sin(angle);}
   if(std::hypot(cosine,sine)<1e-10*selected.size()){out.reason="ambiguous_slab_center";return out;}
   center_fraction=std::atan2(sine,cosine)/(2*std::numbers::pi);
   out.limit_nm=cell.heights[axis]/20.;
   out.cross_section_nm2=Box.volume()/1000./(cell.heights[axis]/10.);
  } else {out.reason="unknown_geometry";return out;}
  if(bin_nm>=out.limit_nm){out.reason="bin_not_smaller_than_profile_domain";return out;}
  out.distance_nm.resize(points.size());
  for(std::size_t i=0;i<points.size();++i) {
   const auto& p=points[i];double dist;
   if(geometry=="slab") {double f=cell.metric.cartesian_to_scaled(p[0],p[1],p[2])[axis];dist=std::abs(std::remainder(f-center_fraction,1.))*cell.heights[axis]/10.;}
   else {double x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];cell.apply_mic(x,y,z);dist=std::sqrt(x*x+y*y+z*z)/10.;}
   out.distance_nm[i]=dist;if(dense[i])out.radius_nm=std::max(out.radius_nm,dist);
  }
  out.radius_nm=std::max(out.radius_nm,bin_nm);
  if(out.radius_nm>=out.limit_nm-1e-10){out.reason="dense_envelope_reaches_profile_domain";return out;}
  out.dense_envelope_nm3=geometry=="slab" ? 2*out.radius_nm*out.cross_section_nm2 : 4*std::numbers::pi*std::pow(out.radius_nm,3)/3;
  out.valid=true;return out;
 }
 double shell_volume(double begin_nm,double end_nm) const {
  if(!valid||begin_nm<0||end_nm<=begin_nm)return 0;
  const double end=std::min(end_nm,limit_nm);if(end<=begin_nm)return 0;
  return geometry=="slab"?2*(end-begin_nm)*cross_section_nm2:4*std::numbers::pi*(end*end*end-begin_nm*begin_nm*begin_nm)/3;
 }
};

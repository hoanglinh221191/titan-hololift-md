#include "tabi_charge_resolution.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace TitanPBE::Tabi {
namespace {
using V=std::array<double,3>;
constexpr auto none=std::numeric_limits<std::size_t>::max();
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double norm(V a){return std::sqrt(dot(a,a));}
V xyz(const Atom&a){return {a.x,a.y,a.z};}
std::array<V,3> vertices(const Panel&p){return {{{p.v0x,p.v0y,p.v0z},{p.v1x,p.v1y,p.v1z},{p.v2x,p.v2y,p.v2z}}};}
std::array<V,3> unit_vertices(const Panel&p,const Atom&s){
  auto v=vertices(p);
  for(auto &x:v){x=sub(x,xyz(s));x=mul(x,1/norm(x));}
  return v;
}

// Maximize direction.u over a convex minor-arc spherical triangle. Candidates
// are the interior radial projection, endpoints, and great-circle arc maxima.
V extremum(const std::array<V,3>&v,V direction){
  const double length=norm(direction);
  if(length==0)return v[0];
  const V u=mul(direction,1/length);
  bool inside=true;
  for(int i=0;i<3;++i){
    const V n=cross(v[i],v[(i+1)%3]);
    const double sign=dot(n,v[(i+2)%3])>=0?1:-1;
    if(sign*dot(n,u)<-1e-14*norm(n))inside=false;
  }
  if(inside)return u;
  V best=v[0];double score=dot(u,best);
  for(int i=0;i<3;++i){
    const V a=v[i],b=v[(i+1)%3],n=cross(a,b);
    if(dot(u,b)>score){best=b;score=dot(u,b);}
    const double n2=dot(n,n);
    if(n2<=1e-28)continue;
    V candidate=sub(u,mul(n,dot(u,n)/n2));
    const double c=norm(candidate);
    if(c<=1e-14)continue;
    candidate=mul(candidate,1/c);
    if(dot(cross(a,candidate),n)>=-1e-14*n2 &&
       dot(cross(candidate,b),n)>=-1e-14*n2 && dot(u,candidate)>score){
      best=candidate;score=dot(u,candidate);
    }
  }
  return best;
}
struct Box {
  V lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
  void include(const Box &b){for(int d=0;d<3;++d){lo[d]=std::min(lo[d],b.lo[d]);hi[d]=std::max(hi[d],b.hi[d]);}}
  double distance(V p)const{double s=0;for(int d=0;d<3;++d){double v=std::max({lo[d]-p[d],0.,p[d]-hi[d]});s+=v*v;}return std::sqrt(s);}
};
struct Primitive {Box box;double h;std::size_t panel;};
struct Node {Box box;double max_h{0};std::size_t owner{none},representative{0},begin{0},end{0},left{none},right{none};};
struct Tree {
  const std::vector<Atom>&atoms;
  const std::vector<Panel>&panels;
  std::vector<Primitive> primitives;
  std::vector<Node> nodes;
  std::size_t build(std::size_t begin,std::size_t end){
    Node node;node.begin=begin;node.end=end;node.owner=panels[primitives[begin].panel].atom_index;
    for(std::size_t i=begin;i<end;++i){
      node.box.include(primitives[i].box);
      if(primitives[i].h>node.max_h){node.max_h=primitives[i].h;node.representative=primitives[i].panel;}
      if(panels[primitives[i].panel].atom_index!=node.owner)node.owner=none;
    }
    const auto index=nodes.size();nodes.push_back(node);
    if(end-begin>16){
      int axis=0;for(int d=1;d<3;++d)if(node.box.hi[d]-node.box.lo[d]>node.box.hi[axis]-node.box.lo[axis])axis=d;
      const auto middle=begin+(end-begin)/2;
      std::nth_element(primitives.begin()+begin,primitives.begin()+middle,primitives.begin()+end,[axis](const Primitive&a,const Primitive&b){
        const double ac=a.box.lo[axis]+a.box.hi[axis],bc=b.box.lo[axis]+b.box.hi[axis];
        return ac==bc?a.panel<b.panel:ac<bc;
      });
      node.left=build(begin,middle);node.right=build(middle,end);
    }
    nodes[index]=node;return index;
  }
  double bound(const Node&node,const Atom&charge)const{
    double d=node.box.distance(xyz(charge));
    if(node.owner!=none){
      const auto &s=atoms[node.owner];
      d=std::max(d,std::abs(norm(sub(xyz(charge),xyz(s)))-s.radius));
    }
    return d;
  }
};
double edge_size(const Panel&p){auto v=vertices(p);return std::max({norm(sub(v[0],v[1])),norm(sub(v[1],v[2])),norm(sub(v[2],v[0]))});}
}

double charge_to_spherical_panel_distance(const Atom&charge,const Atom&sphere,const Panel&panel){
  const V relative=sub(xyz(charge),xyz(sphere));
  if(norm(relative)==0)return sphere.radius;
  const V nearest=extremum(unit_vertices(panel,sphere),relative);
  return norm(sub(relative,mul(nearest,sphere.radius)));
}

ChargeResolutionDiagnostic diagnose_charge_resolution(const std::vector<Atom>&atoms,
    const std::vector<Panel>&panels,const Options&options){
  ChargeResolutionDiagnostic result;
  if(options.primary_surface!="sphere-patch" || options.surface_mode!="vdw" || options.surface_geometry!="triangle"){
    result.status="unsupported_surface";return result;
  }
  const auto begin=std::chrono::steady_clock::now();
  auto finish=[&](){result.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return result;};
  for(const auto &a:atoms){
    if(!std::isfinite(a.q) || (a.q!=0 && (!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(a.z)))){
      result.status="invalid_charge";return finish();
    }
    if(a.q!=0)++result.charged_atoms;
  }
  if(result.charged_atoms==0){result.status="no_charges";return finish();}
  if(panels.empty()){result.status="empty_surface";return finish();}
  Tree tree{atoms,panels,{}, {}};tree.primitives.reserve(panels.size());tree.nodes.reserve(panels.size()/8+2);
  for(std::size_t j=0;j<panels.size();++j){
    const auto &p=panels[j];
    if(!p.has_triangle_geometry || p.atom_index>=atoms.size() || !(atoms[p.atom_index].radius>0)){
      result.status="invalid_surface";return finish();
    }
    const auto &sphere=atoms[p.atom_index];
    bool valid=std::isfinite(sphere.radius);
    for(double x:xyz(sphere))valid=valid&&std::isfinite(x);
    for(auto v:vertices(p)){
      for(double x:v)valid=valid&&std::isfinite(x);
      valid=valid&&norm(sub(v,xyz(sphere)))>0;
    }
    if(!valid){result.status="invalid_surface";return finish();}
    const auto unit=unit_vertices(p,sphere);
    Primitive primitive;primitive.panel=j;primitive.h=edge_size(p);
    if(!(primitive.h>0) || !std::isfinite(primitive.h)) {result.status="invalid_surface";return finish();}
    for(int axis=0;axis<3;++axis){
      V direction{0,0,0};direction[axis]=1;const auto hi=extremum(unit,direction);
      direction[axis]=-1;const auto lo=extremum(unit,direction);
      const double center=xyz(sphere)[axis];const double pad=1e-12*std::max({1.,std::abs(center),sphere.radius});
      primitive.box.lo[axis]=center+sphere.radius*lo[axis]-pad;
      primitive.box.hi[axis]=center+sphere.radius*hi[axis]+pad;
    }
    tree.primitives.push_back(primitive);
  }
  tree.build(0,tree.primitives.size());
  std::vector<std::size_t> stack;stack.reserve(64);
  for(std::size_t i=0;i<atoms.size();++i){
    const auto &charge=atoms[i];if(charge.q==0)continue;
    double best_ratio=INFINITY,best_distance=INFINITY,ratio_distance=0,ratio_h=0;std::size_t best_panel=0;
    auto evaluate=[&](std::size_t j,double h){
      ++result.patch_evaluations;
      double d=charge_to_spherical_panel_distance(charge,atoms[panels[j].atom_index],panels[j]);
      best_distance=std::min(best_distance,d);
      if(d/h<best_ratio){best_ratio=d/h;ratio_distance=d;ratio_h=h;best_panel=j;}
    };
    evaluate(tree.nodes[0].representative,tree.nodes[0].max_h);
    stack.clear();stack.push_back(0);
    while(!stack.empty()){
      const auto node_index=stack.back();stack.pop_back();const auto &node=tree.nodes[node_index];
      const double lower=tree.bound(node,charge);
      if(lower/node.max_h>=best_ratio && lower>=best_distance)continue;
      if(node.left==none){for(auto k=node.begin;k<node.end;++k)evaluate(tree.primitives[k].panel,tree.primitives[k].h);}
      else {
        const auto &l=tree.nodes[node.left],&r=tree.nodes[node.right];
        if(tree.bound(l,charge)/l.max_h < tree.bound(r,charge)/r.max_h){stack.push_back(node.right);stack.push_back(node.left);}
        else {stack.push_back(node.left);stack.push_back(node.right);}
      }
    }
    if(result.min_ratio<0 || best_ratio<result.min_ratio){
      result.min_ratio=best_ratio;result.distance_a=ratio_distance;result.local_edge_a=ratio_h;
      result.charge_index=i;result.panel_index=best_panel;
    }
    if(charge.radius>0 && std::isfinite(charge.radius)){
      const double ratio=best_distance/charge.radius;
      if(result.min_distance_over_atom_radius<0 || ratio<result.min_distance_over_atom_radius)result.min_distance_over_atom_radius=ratio;
      if(best_distance+1e-10*std::max(1.,charge.radius)<charge.radius)++result.surface_inside_atom_count;
    }
  }
  result.warning=result.min_ratio<1.5;
  result.status=result.warning?"underresolved":"no_flag";
  return finish();
}
}

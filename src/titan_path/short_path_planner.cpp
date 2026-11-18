#include "titan_path/short_path_planner.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace titan_path {
ShortPathRanking rank_short_path_frontier(
    std::span<const SelectionCandidate> candidates,std::size_t width,
    bool terminal,double energy_epsilon,double navigation_epsilon) {
  if(width==0||width>64||candidates.size()>20000)
    throw std::invalid_argument("SHORT_PATH_RANKING_BUDGET");
  std::vector<SelectionCandidate> pool(candidates.begin(),candidates.end());
  for(auto &c:pool)
    if(!std::isfinite(c.path_distance)||c.path_distance<0||
       c.minimum_clearance<1.5||c.angle_deg>20.+1e-6||c.angle_deg<0)
      c.valid=false;
  auto pick=[&](){
    return terminal?select_epsilon_progress_candidate(pool,RouteMode::Advance,
        energy_epsilon,false,false,navigation_epsilon)
      :select_epsilon_nonnegative_prefix_candidate(pool,energy_epsilon,navigation_epsilon);
  };
  const auto first=pick();
  ShortPathRanking out{{},first.minimum_energy_key,first.minimum_navigation_cost,false};
  if(!first.found)return out;
  for(auto &c:pool)
    if(c.energy_key>first.minimum_energy_key+energy_epsilon+1e-12||
       c.navigation_cost>first.minimum_navigation_cost+navigation_epsilon+1e-12)
      c.valid=false;
  while(out.indices.size()<width){
    const auto chosen=pick();if(!chosen.found)break;
    out.indices.push_back(chosen.index);pool[chosen.index].valid=false;
  }
  return out;
}
}

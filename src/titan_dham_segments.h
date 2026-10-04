#pragma once
// Audited segment/exchange metadata. Counts are conservative candidates, not
// an estimator corrected for state-dependent exchange censoring.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace titan_dham_segments {
inline constexpr double time_tolerance_ps=1e-6;
struct Frame { std::size_t source_index; double time; int segment; double local, energy, volume; };
struct Segment { double begin,end; std::vector<double> exchanges; };
struct Replica { double temperature{},pressure{}; bool declared=false; std::map<int,Segment> segments; std::vector<Frame> frames; };
enum class Decision { eligible, segment_boundary, time_gap, exchange, missing_energy };
inline const char* name(Decision d) {
 switch(d) {
 case Decision::eligible:return "eligible";
 case Decision::segment_boundary:return "segment_boundary";
 case Decision::time_gap:return "time_gap";
 case Decision::exchange:return "exchange";
 case Decision::missing_energy:return "missing_exact_energy";
 }
 throw std::logic_error("Invalid segment decision");
}
template<class T> inline T number(const std::string& token) {
 T v{};auto r=std::from_chars(token.data(),token.data()+token.size(),v);
 if(r.ec!=std::errc{} || r.ptr!=token.data()+token.size())throw std::invalid_argument("Invalid segment metadata number: "+token);
 if constexpr(std::is_floating_point_v<T>)if(!std::isfinite(v))throw std::invalid_argument("Non-finite segment metadata number");
 return v;
}
class Metadata {
public:
 std::vector<Replica> replicas;
 static Metadata read(const std::string& path,int expected_replicas) {
  if(expected_replicas<=0)throw std::invalid_argument("Invalid metadata replica count");
  std::ifstream f(path);std::string line;
  if(!f || !std::getline(f,line))throw std::invalid_argument("Cannot read DHAM segment metadata");
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line!="TITAN_DHAM_SEGMENTS_V1")throw std::invalid_argument("Unknown DHAM segment metadata version");
  Metadata out;out.replicas.resize(expected_replicas);std::size_t line_no=1;
  while(std::getline(f,line)) {
   ++line_no;std::istringstream input(line);std::vector<std::string> t;std::string token;
   while(input>>token)t.push_back(token);
   if(t.empty()||t[0].starts_with('#'))continue;
   try {
    if(t.size()<2)throw std::invalid_argument("Missing replica");
    int k=number<int>(t[1]);if(k<0||k>=expected_replicas)throw std::invalid_argument("Replica outside metadata domain");
    auto& rep=out.replicas[k];
    if(t[0]=="replica" && t.size()==4) {
     if(rep.declared)throw std::invalid_argument("Duplicate replica declaration");
     rep.temperature=number<double>(t[2]);rep.pressure=number<double>(t[3]);rep.declared=true;
     if(rep.temperature<=0||rep.pressure<0)throw std::invalid_argument("Invalid ensemble metadata");
    } else if(!rep.declared)throw std::invalid_argument("Replica must be declared first");
    else if(t[0]=="segment" && t.size()==5) {
     int id=number<int>(t[2]);double begin=number<double>(t[3]),end=number<double>(t[4]);
     if(id<1||begin<0||end<begin||!rep.segments.emplace(id,Segment{begin,end,{}}).second)throw std::invalid_argument("Invalid or duplicate segment coverage");
    } else if(t[0]=="frame" && t.size()==9) {
     Frame row{number<std::size_t>(t[2]),number<double>(t[3]),number<int>(t[4]),number<double>(t[5]),t[6]=="NA"?std::numeric_limits<double>::quiet_NaN():number<double>(t[6]),number<double>(t[7])};
     // Explicit exact/absent status prevents a nearest sample being relabelled.
     if((t[8]!="exact"&&t[8]!="absent") || ((t[8]=="exact")!=std::isfinite(row.energy)) || row.volume<=0)throw std::invalid_argument("Invalid energy status or volume");
     const auto it=rep.segments.find(row.segment);
     if(it==rep.segments.end()||row.local<it->second.begin-time_tolerance_ps||row.local>it->second.end+time_tolerance_ps)throw std::invalid_argument("Frame outside declared exchange coverage");
     if(!rep.frames.empty()) {
      const auto& prev=rep.frames.back();
      if(row.time<=prev.time+time_tolerance_ps||row.source_index<=prev.source_index)throw std::invalid_argument("Frame identity must increase within replica");
      if(row.segment<prev.segment)throw std::invalid_argument("Segment reentry");
      if(row.segment==prev.segment && (row.local<=prev.local || std::abs((row.time-row.local)-(prev.time-prev.local))>time_tolerance_ps))throw std::invalid_argument("Inconsistent segment clock");
     }
     rep.frames.push_back(row);
    } else if(t[0]=="exchange" && t.size()==4) {
     int id=number<int>(t[2]);double time=number<double>(t[3]);auto it=rep.segments.find(id);
     if(it==rep.segments.end()||time<it->second.begin||time>it->second.end)throw std::invalid_argument("Exchange outside coverage");
     it->second.exchanges.push_back(time);
    } else throw std::invalid_argument("Unknown record or field count");
   } catch(const std::exception& error) {throw std::invalid_argument("DHAM metadata line "+std::to_string(line_no)+": "+error.what());}
  }
  if(!f.eof())throw std::invalid_argument("DHAM metadata read failure");
  for(auto& rep:out.replicas) {
   if(!rep.declared||rep.frames.empty())throw std::invalid_argument("Missing replica frames");
   for(auto& [id,segment]:rep.segments) {
    std::sort(segment.exchanges.begin(),segment.exchanges.end());
    if(std::adjacent_find(segment.exchanges.begin(),segment.exchanges.end())!=segment.exchanges.end())throw std::invalid_argument("Duplicate exchange event");
   }
  }
  return out;
 }
 const Frame& match(int replica,double time) const {
  if(replica<0||static_cast<std::size_t>(replica)>=replicas.size()||!std::isfinite(time))throw std::invalid_argument("Invalid frame identity lookup");
  const auto& frames=replicas[replica].frames;
  auto it=std::lower_bound(frames.begin(),frames.end(),time-time_tolerance_ps,[](const Frame& frame,double t){return frame.time<t;});
  if(it==frames.end()||std::abs(it->time-time)>time_tolerance_ps)throw std::invalid_argument("No exact metadata frame at requested timestamp");
  auto next=it+1;if(next!=frames.end()&&std::abs(next->time-time)<=time_tolerance_ps)throw std::invalid_argument("Ambiguous metadata timestamp");
  return *it;
 }
 Decision transition(int replica,const Frame& first,const Frame& last,double lag_ps) const {
  if(first.segment!=last.segment)return Decision::segment_boundary;
  if(!std::isfinite(lag_ps)||lag_ps<=0||std::abs(last.local-first.local-lag_ps)>time_tolerance_ps)return Decision::time_gap;
  const auto& events=replicas.at(replica).segments.at(first.segment).exchanges;
  auto event=std::lower_bound(events.begin(),events.end(),first.local-time_tolerance_ps);
  if(event!=events.end()&&*event<=last.local+time_tolerance_ps)return Decision::exchange;
  if(!std::isfinite(first.energy)||!std::isfinite(last.energy))return Decision::missing_energy;
  return Decision::eligible;
 }
};
struct CountAudit {std::map<Decision,std::size_t> decisions;std::map<std::pair<int,int>,std::size_t> counts;};
inline CountAudit audit_counts(const Metadata& metadata,int replica,const std::vector<int>& states,const std::vector<double>& times,int lag_frames,int n_states) {
 if(states.size()!=times.size()||lag_frames<=0||n_states<=0)throw std::invalid_argument("Invalid segment count input dimensions");
 std::vector<const Frame*> frames;frames.reserve(times.size());
 for(std::size_t i=0;i<times.size();++i) {
  if(states[i]<0||states[i]>=n_states||(i&&times[i]<=times[i-1]))throw std::invalid_argument("Invalid state or time order");
  frames.push_back(&metadata.match(replica,times[i]));
 }
 CountAudit out;if(times.size()<2)return out;
 const double lag_ps=(times[1]-times[0])*lag_frames;
 for(std::size_t i=0;i+static_cast<std::size_t>(lag_frames)<times.size();++i) {
  const auto j=i+lag_frames;const auto decision=metadata.transition(replica,*frames[i],*frames[j],lag_ps);
  ++out.decisions[decision];if(decision==Decision::eligible)++out.counts[{states[i],states[j]}];
 }
 return out;
}
} // namespace titan_dham_segments

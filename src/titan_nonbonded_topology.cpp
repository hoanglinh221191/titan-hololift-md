#include "titan_nonbonded_topology.h"
#include "topology_include_utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace titan_nb {
std::uint64_t pair_key(std::size_t a, std::size_t b) {
  if (a > UINT32_MAX || b > UINT32_MAX) throw std::runtime_error("topology atom index overflow");
  return (std::uint64_t(std::min(a,b)) << 32) | std::max(a,b);
}
PairTerms Topology::pair(std::size_t a, std::size_t b) const {
  if (a>=atoms.size() || b>=atoms.size()) throw std::runtime_error("topology atom identity out of range");
  PairTerms result;
  if (a==b) return result;
  const auto key=pair_key(a,b);
  if (!exclusions.contains(key)) {
    result.normal=nonbond_matrix[std::size_t(atoms[a].type)*type_count+atoms[b].type];
    result.normal_charge=atoms[a].charge*atoms[b].charge;
  }
  if (auto it=listed_pairs.find(key);it!=listed_pairs.end()) {
    result.listed=it->second.lj;result.listed_charge=it->second.charge_product;
    result.has_listed=true;
  }
  return result;
}

namespace {
struct Line { std::string text, where; };
using Fields=std::vector<std::string>;
Fields fields(const std::string &s) {
  std::istringstream in(s);Fields result;for(std::string v;in>>v;)result.push_back(v);return result;
}
[[noreturn]] void fail(const Line &line,const std::string &message) {
  throw std::runtime_error("unsupported_topology_self_energy: "+line.where+": "+message);
}
double number(const std::string &s,const Line &line) {
  std::size_t used=0;double value;
  try {value=std::stod(s,&used);}catch(...){fail(line,"invalid numeric field '"+s+"'");}
  if(used!=s.size() || !std::isfinite(value))fail(line,"nonfinite or malformed number '"+s+"'");
  return value;
}
std::uint32_t integer(const std::string &s,const Line &line,bool zero=false) {
  double value=number(s,line);
  if(value<(zero?0:1) || value>UINT32_MAX || std::floor(value)!=value)fail(line,"invalid integer '"+s+"'");
  return static_cast<std::uint32_t>(value);
}

struct Reader {
  std::unordered_map<std::string,std::string> macros;
  std::vector<std::filesystem::path> stack,inputs;
  std::vector<Line> lines;
  void read(std::filesystem::path path) {
    path=std::filesystem::absolute(path).lexically_normal();
    if(stack.size()>=64 || std::find(stack.begin(),stack.end(),path)!=stack.end())
      throw std::runtime_error("unsupported_topology_self_energy: recursive/deep topology include: "+path.string());
    std::ifstream file(path);
    if(!file)throw std::runtime_error("unsupported_topology_self_energy: cannot open "+path.string());
    stack.push_back(path);inputs.push_back(path);
    struct Branch {bool parent,condition,in_else;};std::vector<Branch> branches;
    bool active=true;std::size_t line_number=0;
    for(std::string text;std::getline(file,text);) {
      ++line_number;text=TitanTopo::trim_copy(text.substr(0,text.find(';')));
      Line line{text,path.string()+":"+std::to_string(line_number)};
      if(text.empty())continue;
      if(text.front()=='#') {
        Fields f=fields(TitanTopo::trim_copy(text.substr(1)));
        if(f.empty())continue;
        const auto &command=f[0];
        if(command=="ifdef" || command=="ifndef") {
          if(f.size()!=2)fail(line,"malformed conditional");
          bool condition=macros.contains(f[1]);if(command=="ifndef")condition=!condition;
          branches.push_back({active,condition,false});active=active&&condition;
        } else if(command=="else") {
          if(branches.empty() || branches.back().in_else)fail(line,"unmatched/duplicate #else");
          branches.back().in_else=true;active=branches.back().parent&&!branches.back().condition;
        } else if(command=="endif") {
          if(branches.empty())fail(line,"unmatched #endif");
          active=branches.back().parent;branches.pop_back();
        } else if(command=="if" || command=="elif") {
          fail(line,"#if/#elif expressions require a preprocessed gmx grompp -pp topology");
        } else if(!active)continue;
        else if(command=="include") {
          auto target=TitanTopo::extract_include_target(text);
          if(!target)fail(line,"malformed #include");
          bool found=false;
          for(auto candidate:TitanTopo::build_include_candidates(path.string(),*target)) {
            if(std::filesystem::is_regular_file(candidate)){read(candidate);found=true;break;}
          }
          if(!found)fail(line,"unresolved include "+*target);
        } else if(command=="define") {
          if(f.size()<2 || f[1].find('(')!=std::string::npos)fail(line,"unsupported macro definition");
          std::string value;for(std::size_t i=2;i<f.size();++i){if(i>2)value+=' ';value+=f[i];}
          macros[f[1]]=value;
        } else if(command=="undef") {
          if(f.size()!=2)fail(line,"malformed #undef");macros.erase(f[1]);
        } else fail(line,"unsupported preprocessor directive "+command);
        continue;
      }
      if(!active)continue;
      // Expand object-like numeric parameter macros without executing code.
      if(text.front()!='[') {
        for(int pass=0;pass<16;++pass) {
          auto f=fields(text);bool changed=false;std::string expanded;
          for(auto &token:f) {
            if(auto it=macros.find(token);it!=macros.end()){token=it->second;changed=true;}
            if(!expanded.empty())expanded+=' ';expanded+=token;
          }
          text=expanded;if(!changed)break;if(pass==15)fail(line,"recursive macro expansion");
        }
      }
      line.text=text;lines.push_back(std::move(line));
    }
    if(!branches.empty())fail({{},path.string()},"unterminated conditional");
    stack.pop_back();
  }
};
struct AtomType {std::string name;double v,w;};
struct RawPair {std::uint32_t a,b,function;Fields parameters;Line line;bool nb;};
struct Molecule {
  std::string name;std::uint32_t nrexcl{};
  std::vector<Atom> atoms;
  std::vector<std::pair<std::uint32_t,std::uint32_t>> bonds,excluded;
  std::vector<RawPair> pairs;
};
Coefficients coefficients(double v,double w,int rule,const Line &line) {
  if(v<0 || w<0)fail(line,"negative LJ parameters require an explicitly supported convention");
  if(rule==1)return {v,w};
  const double v6=std::pow(v,6);Coefficients c{4*w*v6,4*w*v6*v6};
  if(!std::isfinite(c.c6)||!std::isfinite(c.c12))fail(line,"LJ parameter overflow");
  return c;
}
Coefficients mix(const AtomType&a,const AtomType&b,int rule,const Line&line) {
  if(rule==1)return coefficients(std::sqrt(a.v)*std::sqrt(b.v),std::sqrt(a.w)*std::sqrt(b.w),rule,line);
  return coefficients(rule==2 ? 0.5*(a.v+b.v) : std::sqrt(a.v)*std::sqrt(b.v),
                      std::sqrt(a.w)*std::sqrt(b.w),rule,line);
}
}

std::shared_ptr<const Topology> read_topology(const std::filesystem::path &path) {
  Reader reader;reader.read(path);
  auto result=std::make_shared<Topology>();result->input_files=reader.inputs;
  std::vector<AtomType> types;std::unordered_map<std::string,std::uint32_t> type_ids;
  std::unordered_map<std::uint64_t,Coefficients> overrides,pairtypes;
  std::vector<Molecule> molecules;std::unordered_map<std::string,std::size_t> molecule_ids;
  std::vector<std::pair<std::string,std::uint32_t>> instances;
  std::string section;Molecule *mol=nullptr;bool defaults=false,generate_pairs=false;
  for(const Line &line:reader.lines) {
    if(line.text.front()=='[') {
      const auto end=line.text.find(']');if(end==std::string::npos)fail(line,"unclosed section");
      section=TitanTopo::trim_copy(line.text.substr(1,end-1));
      if(section.starts_with("virtual_sites") || section=="polarization" || section=="thole_polarization" ||
         section=="intermolecular_interactions")fail(line,"unsupported interaction section "+section);
      continue;
    }
    const Fields f=fields(line.text);if(f.empty())continue;
    if(section=="defaults") {
      if(defaults || f.size()<2 || integer(f[0],line)!=1)fail(line,"one LJ [defaults] definition is required");
      result->combination_rule=integer(f[1],line);
      if(result->combination_rule>3)fail(line,"unsupported combination rule");
      if(f.size()>2 && f[2]!="yes" && f[2]!="no")fail(line,"invalid gen-pairs");
      generate_pairs=f.size()>2 && f[2]=="yes";
      if(f.size()>3)result->fudge_lj=number(f[3],line);
      if(f.size()>4)result->fudge_qq=number(f[4],line);
      if(result->fudge_lj<0 || result->fudge_qq<0 || f.size()>6 ||
         (f.size()==6 && number(f[5],line)!=12))fail(line,"unsupported LJ exponent/scaling");
      defaults=true;continue;
    }
    if(!defaults)fail(line,"missing [defaults]; supply the full system topology");
    if(section=="atomtypes") {
      if(f.size()<6 || f[f.size()-3]!="A")fail(line,"unsupported atom type format/particle type");
      if(type_ids.contains(f[0]))fail(line,"duplicate atomtype "+f[0]);
      const double v=number(f[f.size()-2],line),w=number(f.back(),line);
      coefficients(v,w,result->combination_rule,line);
      type_ids[f[0]]=static_cast<std::uint32_t>(types.size());types.push_back({f[0],v,w});
    } else if(section=="nonbond_params" || section=="pairtypes") {
      if(f.size()!=5 || integer(f[2],line)!=1 || !type_ids.contains(f[0]) || !type_ids.contains(f[1]))
        fail(line,"unsupported/unknown pair type parameters");
      auto &table=section=="pairtypes"?pairtypes:overrides;
      table[pair_key(type_ids.at(f[0]),type_ids.at(f[1]))]=coefficients(number(f[3],line),number(f[4],line),result->combination_rule,line);
    } else if(section=="moleculetype") {
      if(f.size()!=2 || molecule_ids.contains(f[0]))fail(line,"invalid/duplicate moleculetype");
      molecules.push_back({});mol=&molecules.back();mol->name=f[0];mol->nrexcl=integer(f[1],line,true);
      if(mol->nrexcl>32)fail(line,"unreasonably large nrexcl");
      molecule_ids[f[0]]=molecules.size()-1;
    } else if(section=="molecules") {
      if(f.size()!=2 || !molecule_ids.contains(f[0]))fail(line,"unknown molecule instance");
      instances.emplace_back(f[0],integer(f[1],line,true));
    } else if(section=="atoms") {
      if(!mol || f.size()<7 || f.size()>8 || integer(f[0],line)!=mol->atoms.size()+1 || !type_ids.contains(f[1]))
        fail(line,"atoms must be consecutive, resolved A-state entries (B-state not supported)");
      mol->atoms.push_back({type_ids.at(f[1]),number(f[6],line),f[4]});
    } else if(section=="bonds" || section=="constraints") {
      if(!mol || f.size()<3)fail(line,"invalid bond/constraint");
      const auto a=integer(f[0],line)-1,b=integer(f[1],line)-1,fn=integer(f[2],line);
      if(section=="constraints" && fn>2)fail(line,"unsupported constraint type");
      if(section=="bonds" && fn>10)fail(line,"unsupported bond type");
      if((section=="bonds" && (fn<=5 || fn==7 || fn==8)) || (section=="constraints" && fn==1))
        mol->bonds.emplace_back(a,b);
    } else if(section=="settles") {
      if(!mol || f.size()!=4 || integer(f[1],line)!=1 || mol->nrexcl==0)fail(line,"unsupported SETTLE definition");
      const auto a=integer(f[0],line)-1;
      mol->bonds.insert(mol->bonds.end(),{{a,a+1},{a,a+2},{a+1,a+2}});
    } else if(section=="exclusions") {
      if(!mol || f.size()<2)fail(line,"invalid exclusions");
      const auto a=integer(f[0],line)-1;
      for(std::size_t i=1;i<f.size();++i)mol->excluded.emplace_back(a,integer(f[i],line)-1);
    } else if(section=="pairs" || section=="pairs_nb") {
      if(!mol || f.size()<3)fail(line,"invalid listed pair");
      RawPair pair{integer(f[0],line)-1,integer(f[1],line)-1,integer(f[2],line),Fields(f.begin()+3,f.end()),line,section=="pairs_nb"};
      if(pair.function>2 || (pair.nb&&pair.function!=1))fail(line,"unsupported listed pair function");
      mol->pairs.push_back(std::move(pair));
    }
  }
  if(!defaults || types.empty() || instances.empty())throw std::runtime_error("unsupported_topology_self_energy: full defaults/atomtypes/molecules topology required");
  result->type_count=static_cast<std::uint32_t>(types.size());
  if(types.size()>8192)throw std::runtime_error("topology type matrix exceeds bounded memory limit");
  result->nonbond_matrix.resize(types.size()*types.size());
  Line location{{},path.string()};
  for(std::size_t a=0;a<types.size();++a)for(std::size_t b=0;b<types.size();++b) {
    auto c=mix(types[a],types[b],result->combination_rule,location);
    if(auto it=overrides.find(pair_key(a,b));it!=overrides.end())c=it->second;
    result->nonbond_matrix[a*types.size()+b]=c;
  }
  for(const auto &[name,copies]:instances) {
    const Molecule &m=molecules[molecule_ids.at(name)];const auto n=m.atoms.size();
    if(n==0 || std::uint64_t(result->atoms.size())+std::uint64_t(copies)*n>UINT32_MAX)fail(location,"empty/oversized molecule expansion");
    std::vector<std::vector<std::uint32_t>> adjacent(n);
    auto check_pair=[&](std::size_t a,std::size_t b){if(a>=n||b>=n||a==b)fail(location,"invalid local atom pair in "+name);};
    for(auto [a,b]:m.bonds){check_pair(a,b);adjacent[a].push_back(b);adjacent[b].push_back(a);}
    std::unordered_set<std::uint64_t> excluded;
    std::vector<int> distance(n,-1);std::vector<std::uint32_t> queue;
    for(std::uint32_t a=0;a<n;++a) {
      queue.clear();queue.push_back(a);distance[a]=0;
      for(std::size_t head=0;head<queue.size();++head) {
        const auto b=queue[head];if(distance[b]>=static_cast<int>(m.nrexcl))continue;
        for(auto c:adjacent[b])if(distance[c]<0){distance[c]=distance[b]+1;queue.push_back(c);excluded.insert(pair_key(a,c));}
      }
      for(auto b:queue)distance[b]=-1;
    }
    for(auto [a,b]:m.excluded){check_pair(a,b);excluded.insert(pair_key(a,b));}
    std::unordered_map<std::uint64_t,ListedPair> listed;
    for(const auto &p:m.pairs) {
      check_pair(p.a,p.b);const auto key=pair_key(p.a,p.b);
      if(listed.contains(key))fail(p.line,"duplicate listed pair");
      const Atom &a=m.atoms[p.a],&b=m.atoms[p.b];ListedPair value;
      const auto &f=p.parameters;
      if(p.nb || p.function==2) {
        const std::size_t start=p.nb?0:1;
        if(f.size()!=start+4)fail(p.line,"explicit charge/LJ fields required for pairs_nb or pairs function 2");
        value.charge_product=(p.nb?1:number(f[0],p.line))*number(f[start],p.line)*number(f[start+1],p.line);
        value.lj=coefficients(number(f[start+2],p.line),number(f[start+3],p.line),result->combination_rule,p.line);
      } else {
        value.charge_product=result->fudge_qq*a.charge*b.charge;
        if(f.size()==2)value.lj=coefficients(number(f[0],p.line),number(f[1],p.line),result->combination_rule,p.line);
        else if(!f.empty())fail(p.line,"unsupported listed pair parameters/B-state");
        else if(auto it=pairtypes.find(pair_key(a.type,b.type));it!=pairtypes.end())value.lj=it->second;
        else if(generate_pairs) {
          // grompp generates LJ-14 from the resolved nonbonded matrix,
          // including nonbond_params, before applying fudgeLJ.
          value.lj=result->nonbond_matrix[std::size_t(a.type)*result->type_count+b.type];
          value.lj.c6*=result->fudge_lj;value.lj.c12*=result->fudge_lj;
        } else fail(p.line,"no pairtypes and gen-pairs=no; cannot infer 1-4 LJ parameters");
      }
      listed[key]=value;
    }
    for(std::uint32_t copy=0;copy<copies;++copy) {
      const std::size_t offset=result->atoms.size();
      result->atoms.insert(result->atoms.end(),m.atoms.begin(),m.atoms.end());
      for(auto key:excluded)result->exclusions.insert(pair_key(offset+(key>>32),offset+(key&UINT32_MAX)));
      for(const auto &[key,value]:listed)result->listed_pairs[pair_key(offset+(key>>32),offset+(key&UINT32_MAX))]=value;
    }
  }
  return result;
}
} // namespace titan_nb

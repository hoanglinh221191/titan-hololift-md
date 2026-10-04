#ifndef GROUPS_H
#define GROUPS_H
#include "MOL.h"
#include "cache_sigma_epsilon.h"
#include "miscellaneous.h"
#include <set>
#include <string>
#include <optional>

struct group_cal {
  bool is_continuous;
  std::string name;
  std::vector<size_t> atom_index;
  set<size_t> group_set;
  group_cal() : is_continuous{true}, name{"NULL"} {}
  group_cal(string _name) : is_continuous{false}, name{_name} {
    atom_index.reserve(1000);
  }
  group_cal(string _name, size_t begin, size_t end)
      : is_continuous{true}, name{_name} {
    for (size_t i{begin}; i <= end; ++i)
      atom_index.emplace_back(i);
  }
  int check_conti();
  bool check_in(size_t ind_check);
  int gen_setgroup() {
    for (auto &a : atom_index)
      group_set.emplace(a);
    return 0;
  }
};
class groups_cal {
  std::vector<group_cal> groups;
  bool entropy;
  bool parse_index_flag;
  size_t group_1, group_2, group_entropy_com, group_entropy_rec,
      group_entropy_lg, group_pathway;
  vector<pair<int, size_t>> sub_group_1, sub_group_2;
  std::vector<double> E_elec, E_vdw, time_of_snap;

public:
  groups_cal()
      : entropy{false}, parse_index_flag{false}, group_1{}, group_2{},
        group_entropy_com{}, group_entropy_rec{}, group_entropy_lg{},
        group_pathway{} {
    groups.reserve(25);
  }
  groups_cal(const vector<class index> &Index);
  int index(arguments &Arguments);
  void print();
  group_cal *get_group(size_t group_index) { return &groups[group_index]; }
  // int calculate_energy_for_2_groups(  Multi_MOL &rec,   Multi_MOL &lg,
  // cache_sigma_epsilon& CachedData, int begin, int end, size_t num_snap, int
  // group_1, int group_2);
  int print_result_cal_to_file();
  int select_groups_to_cal_energy();
  int select_group_to_cal_entropy();
  bool empty() { return groups.empty(); }
  size_t group_num() { return groups.size(); }
  bool get_parse_index_flag() const { return parse_index_flag; }
  size_t group_1_index() const { return group_1; }
  size_t group_2_index() const { return group_2; }
  size_t group_pathway_index() const { return group_pathway; }
  const size_t group_entropy_com_index() { return group_entropy_com; }
  size_t group_entropy_rec_index() { return group_entropy_rec; }
  size_t group_entropy_lg_index() { return group_entropy_lg; }
  int create_group_entropy(const Multi_MOL &mol1, const Multi_MOL &mol2,
                           const arguments &Arguments, string name_group);
  int finalize_index_groups(arguments &Arguments);
  int create_auto_index_file(arguments &Arguments, Multi_MOL &mol_in,
                             const amino_acid &amine);
  int creat_index_file(arguments &Arguments, Multi_MOL &mol_in);

  size_t atom_index(size_t group, size_t index) {
    if (group < groups.size() && group >= 0 &&
        index < groups[group].atom_index.size() && index >= 0)
      return groups[group].atom_index[index];
    else
      return 0;
  }
  size_t groups_atom_number(size_t group) {
    if (group < groups.size() && group >= 0)
      return groups[group].atom_index.size();
    else
      return 0;
  }
  std::string group_name(size_t group) const { return groups[group].name; }
  std::optional<size_t> find_group_index(std::string_view name) const;
  bool is_cont(size_t group_index) { return groups[group_index].is_continuous; }
  int add_group_from_range(string group_name, size_t begin, size_t end) {
    if (groups.empty())
      group_1 = 0;
    else if (groups.size() == 1)
      group_2 = 1;
    else {
    }
    groups.emplace_back(group_name, begin, end);
    return 0;
  }
  int add_empty_group(string group_name) {
    if (groups.empty())
      group_1 = 0;
    else if (groups.size() == 1)
      group_2 = 1;
    groups.emplace_back(group_name);
    return 0;
  }
  int add_sub_group_1(arguments &Arguments);
  int add_sub_group_2(arguments &Arguments);
  int check_sub_group();
  int add_sub_groups(arguments &Arguments);
  int parse_subgroups();
  size_t subgroup1_size() const { return sub_group_1.size(); }
  size_t subgroup2_size() const { return sub_group_2.size(); }
  int subgroup_1_index(size_t i) const { return sub_group_1[i].first; }
  int subgroup_2_index(size_t i) { return sub_group_2[i].first; }
  size_t subgroup_1_offset(size_t i) { return sub_group_1[i].second; }
  size_t subgroup_2_offset(size_t i) { return sub_group_2[i].second; }
  size_t add_group_pathway(arguments &Arguments);
};
#endif

#ifndef H_GROUPS_H
#define H_GROUPS_H
#include"MOL.h"
using namespace std;

struct h_group
{
	char name;
	size_t index;
	vector<size_t> h_index;
	h_group() :name{ '-' }, index{} {
		h_index.reserve(100);
	}
	[[nodiscard]] bool has_hydrogen() const noexcept { return !h_index.empty(); }
	int add_groups(const MOL& mol_in, size_t i, char _name, double bond_length,
	               const vector<mol_info>& info);
};

struct h_groups
{
	std::vector<h_group> groups;
	int find_groups(const MOL& mol_in, const vector<mol_info>& info,
	                const arguments& Arguments);
	void print_group(int m);
};

#endif // !H_GROUPS_H

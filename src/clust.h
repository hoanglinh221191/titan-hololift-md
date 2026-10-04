#ifndef CLUST_H
#define CLUST_H
#include<vector>
#include"MOL.h"
using namespace std;
struct cluster
{
	vector<pair<size_t,size_t> > clust;
};

struct multi_cluster
{
	vector<cluster> clusters;
	int emplace_clust(size_t image_1, size_t image_2, size_t atom_1, size_t atom_2);
	int find_candidate_clust(MOL& mol_in, arguments& Arguments);
};
#endif

#include<fstream>
#include<string>
#include<vector>
#include<iostream>
#include<cmath>
using namespace std;

int gen_chain_list_cell(size_t chain, vector<vector<size_t>>& data)
{

	size_t num = static_cast<size_t>(pow(27, chain));
	data =  vector<vector<size_t>>{num, vector<size_t>(chain)};

	vector<size_t> pos(chain, { 1 }), pos2(chain, { 1 });
	for (size_t i{  }; i < num; ++i)
	{
		for (size_t j{ 1 }; j <= chain; ++j)
		{
			data[i][j-1] = pos[j-1]-1;
			if ((i + 1) % (static_cast<size_t>(pow(27, j - 1))) == 0)
				pos[j - 1] += 1;
			if (pos[j - 1] == 28)pos[j - 1] = 1;
		}
	}
	return 0;
}
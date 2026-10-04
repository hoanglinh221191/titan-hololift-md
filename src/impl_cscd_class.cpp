#include "cscd.h"
#include <iomanip>
#include <iostream>

element::element(double _time, vector<double> &_distance)
    : size{1}, time{_time}, distance(_distance) {}
res_cscd::res_cscd(double _time, vector<double> &in) {
  dis_res.emplace_back(_time, in);
}

void res_cscd::emplace(double time, vector<double> &dis) {
  dis_res.emplace_back(time, dis);
}

void res_cscd::print2file(fstream &out) {
  out.precision(4);
  for (auto &dat : dis_res) {
    out << setw(10) << fixed << dat.time;
    for (auto &dis : dat.distance)
      out << setw(14) << fixed << dis / dat.size;
    out << '\n';
  }
}

double res_cscd::distance(size_t i, size_t j) { return dis_res[i].distance[j]; }

double res_cscd::time(size_t i, size_t j) { return dis_res[i].time; }

void res_cscd::set(size_t i, double time, vector<double> &dis) {
  if (i >= dis_res.size())
    dis_res.emplace_back(time, dis);
  else {
    ++(dis_res[i].size);
    for (size_t j{}; j < dis.size(); ++j)
      dis_res[i].distance[j] += dis[j];
  }
}

all_cscd::all_cscd() { CSCD.reserve(100); }
all_cscd::all_cscd(size_t size) { CSCD = vector<res_cscd>(size); }

void all_cscd::set_size(size_t size) {
  if (!CSCD.empty())
    CSCD.clear();
  CSCD = vector<res_cscd>(size);
}

void all_cscd::emplace(size_t size, size_t pos, size_t pos_res_cscd,
                       double time, vector<double> &dis) {
  if (CSCD.size() < size)
    CSCD.emplace_back(time, dis);
  else
    CSCD[pos].set(pos_res_cscd, time, dis);
}

int all_cscd::printResult(vector<string> &list_file_name) {
  for (size_t i{}; i < list_file_name.size(); ++i) {
    if (i >= CSCD.size())
      continue;
    fstream out(list_file_name[i], ios::out);
    CSCD[i].print2file(out);
    out.close();
  }
  return 0;
}
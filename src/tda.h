#ifndef TITAN_TDA_H
#define TITAN_TDA_H

#include "MOL.h"
#include "Multi_MOL.h"
#include "arguments.h"
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class TDAAnalyzer {
public:
  ~TDAAnalyzer();

  void export_TDA_pointcloud(const Multi_MOL &pro_in, const Multi_MOL &lig,
                             int traj_id, int num_snap,
                             const std::vector<int> &target_idx_1,
                             const std::vector<int> &target_idx_2,
                             const arguments &Arguments,
                             const std::string &output_csv);

  void flush_TDA_pointcloud();

private:
  struct PointRow {
    int traj_id{0};
    int frame{0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
  };

  void add_point_row(const PointRow &row, std::size_t max_points,
                     const std::string &output_csv);

  std::vector<PointRow> reservoir_rows_;
  std::size_t reservoir_seen_{0};
  std::size_t reservoir_max_points_{0};
  std::string reservoir_output_csv_;
  std::mt19937_64 reservoir_rng_{0x5444415f4d454dULL};
};

#endif // TITAN_TDA_H

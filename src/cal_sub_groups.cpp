#include "Multi_MOL.h"
#include "energy_force.h"
#include "folder.h"
#include "groups.h"
#include "miscellaneous.h"
#include "titan_executor.h"
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <thread>

using namespace std;
using namespace energy_force_val;

int force_energy_add_intrin(const cache_sigma_epsilon &CachedData, size_t *i,
                            double &electro_energy,
                            double &van_der_waals_energy, force &force_elec,
                            force &force_vdw,
                            const Multi_MOL_parameters &rec_parameters,
                            const Multi_MOL_parameters &lg_parameters,
                            const Multi_MOL &rec, const Multi_MOL &lg,
                            size_t snap, const arguments &Arguments,
                            size_t rec_pos = 0, size_t lg_pos = 0);

int force_energy_add(const cache_sigma_epsilon &CachedData, size_t i,
                     double &electro_energy, double &van_der_waals_energy,
                     force &force_elec, force &force_vdw,
                     const Multi_MOL_parameters &rec_parameters,
                     const Multi_MOL_parameters &lg_parameters,
                     const Multi_MOL &rec, const Multi_MOL &lg, size_t snap,
                     const arguments &Arguments, size_t rec_pos = 0,
                     size_t lg_pos = 0);

int sup_cal_one_subgroup_optimized(groups_cal &groups, int group_rec_index,
                                   size_t sub1_pos, int group_lg_index,
                                   size_t sub2_pos, const Multi_MOL &rec,
                                   const Multi_MOL &lg,
                                   const cache_sigma_epsilon &CachedData,
                                   energy_res_base &energy_force_out,
                                   const arguments &Arguments, size_t num_snap,
                                   size_t begin, size_t end) {
  // 1. Tính toán các Offset cơ sở
  size_t offset1{groups.subgroup_1_offset(sub1_pos) -
                 groups.atom_index(group_rec_index, 0)};
  size_t offset2{groups.subgroup_2_offset(sub2_pos) -
                 groups.atom_index(group_lg_index, 0)};

  // Số lượng nguyên tử trong nhóm Ligand
  size_t size_lg_group = groups.groups_atom_number(group_lg_index);

  // Stride (Bước nhảy) của mảng Global: Kích thước của 1 phân tử (số lượng cặp
  // tương tác/hàng) Dựa vào code cũ: index = Stride * Ligand + Receptor
  size_t mol_size_stride = rec.m_mol[num_snap].size();

  // Lấy con trỏ thô tới dữ liệu (SoA arrays)
  const double *raw_elec = CachedData.elec_ener.data();
  const double *raw_vdw = CachedData.vdw_ener.data();

  // Thanh ghi tích lũy tổng AVX: Chứa [Elec1, Vdw1, Elec0, Vdw0]
  __m256d v_total = _mm256_setzero_pd();

  // Biến tích lũy cho phần dư (Scalar)
  double scalar_e = 0.0;
  double scalar_v = 0.0;

  // --- TỐI ƯU HÓA: ĐẢO VÒNG LẶP (Loop Interchange) ---
  // Đưa vòng lặp Ligand (j) ra ngoài, Receptor (i) vào trong.
  // Lý do: Index bộ nhớ thay đổi theo i (Receptor). Khi i chạy liên tục, ta
  // truy cập bộ nhớ tuần tự -> Rất nhanh.
  for (size_t j = 0; j < size_lg_group; ++j) {
    // Tính Base Index cho nguyên tử Ligand j hiện tại
    // Công thức gốc: Global_Index = Stride * (Lig_Index) + Rec_Index
    size_t lg_atom_idx = groups.atom_index(group_lg_index, j) + offset2;
    size_t base_idx_for_ligand = mol_size_stride * lg_atom_idx;

    size_t i = begin;

    // --- VÒNG LẶP TRONG: RECEPTOR (Xử lý 2 nguyên tử cùng lúc) ---
    // Mỗi nguyên tử có 2 giá trị (Elec, Vdw) = 128 bit.
    // 1 thanh ghi AVX 256-bit chứa được 2 nguyên tử (4 giá trị double).
    // Điều kiện dừng: i <= end - 2
    for (; i + 2 <= end; i += 2) {
      // Lấy index thực của 2 nguyên tử Receptor liên tiếp
      size_t rec_atom_idx_0 = groups.atom_index(group_rec_index, i) + offset1;
      size_t rec_atom_idx_1 =
          groups.atom_index(group_rec_index, i + 1) + offset1;

      // Tính vị trí trong mảng dữ liệu CachedData
      size_t idx0 = base_idx_for_ligand + rec_atom_idx_0;
      size_t idx1 = base_idx_for_ligand + rec_atom_idx_1;

      // Load dữ liệu từ 2 mảng riêng biệt (SoA)
      // raw_elec và raw_vdw
      __m128d p_elec =
          _mm_set_pd(raw_elec[idx1], raw_elec[idx0]); // [Elec1, Elec0]
      __m128d p_vdw = _mm_set_pd(raw_vdw[idx1], raw_vdw[idx0]); // [Vdw1, Vdw0]

      // Ghép 2 cặp vào 1 thanh ghi 256-bit: [Elec1, Vdw1, Elec0, Vdw0]
      // unpacklo(A, B) lấy element thấp của A và B -> [A0, B0] = [Elec0, Vdw0]
      // unpackhi(A, B) lấy element cao của A và B -> [A1, B1] = [Elec1, Vdw1]
      __m128d p0 = _mm_unpacklo_pd(p_vdw, p_elec); // [Elec0, Vdw0]
      __m128d p1 = _mm_unpackhi_pd(p_vdw, p_elec); // [Elec1, Vdw1]
      __m256d v_pair = _mm256_set_m128d(p1, p0);

      // Cộng dồn vào tổng
      v_total = _mm256_add_pd(v_total, v_pair);
    }

    // --- XỬ LÝ PHẦN DƯ CỦA RECEPTOR (Nếu số lượng lẻ hoặc index cuối) ---
    for (; i < end; ++i) {
      size_t rec_atom_idx = groups.atom_index(group_rec_index, i) + offset1;
      size_t idx = base_idx_for_ligand + rec_atom_idx;

      // Cộng thủ công
      scalar_e += CachedData.elec_ener[idx];
      scalar_v += CachedData.vdw_ener[idx];
    }
  }

  // --- TỔNG HỢP KẾT QUẢ TỪ AVX ---
  // Store thanh ghi 256-bit ra mảng tạm
  // Thứ tự trong memory sẽ là: [Elec0, Vdw0, Elec1, Vdw1]
  double temp[4];
  _mm256_storeu_pd(temp, v_total);

  // temp[0] = Elec atom 0, temp[1] = Vdw atom 0
  // temp[2] = Elec atom 1, temp[3] = Vdw atom 1
  double sum_avx_e = temp[0] + temp[2];
  double sum_avx_v = temp[1] + temp[3];

  // Cộng tất cả vào biến output của hàm
  energy_force_out.e_e += (sum_avx_e + scalar_e);
  energy_force_out.e_v += (sum_avx_v + scalar_v);

  return 0;
}

int sup_cal_one_subgroup(groups_cal &groups, int group_rec_index,
                         size_t sub1_pos, int group_lg_index, size_t sub2_pos,
                         const Multi_MOL &rec, const Multi_MOL &lg,
                         const cache_sigma_epsilon &CachedData,
                         energy_res_base &energy_force_out,
                         const arguments &Arguments, size_t num_snap,
                         size_t begin, size_t end) {
  size_t offset1{groups.subgroup_1_offset(sub1_pos) -
                 groups.atom_index(group_rec_index, 0)},
      offset2{groups.subgroup_2_offset(sub2_pos) -
              groups.atom_index(group_lg_index, 0)};
  // size_t size_lg_cal{ groups.groups_atom_number(group_lg_index) / 4 };
  size_t size_lg_group{groups.groups_atom_number(group_lg_index)};
  size_t mol_size_stride{rec.m_mol[num_snap].size()};

  const double *elec_ptr = CachedData.elec_ener.data();
  const double *vdw_ptr = CachedData.vdw_ener.data();
  __m256d v_total_sum = _mm256_setzero_pd();

  for (size_t i{begin}; i < end; ++i) {
    size_t atomIndex_i{groups.atom_index(group_rec_index, i)},
        offset{atomIndex_i + offset1};

    size_t j{};
    for (; j <= (size_lg_group >= 4 ? size_lg_group - 4 : 0); j += 4) {
      /*size_t pos[4]{rec.m_mol[num_snap].size() *
         (groups.atom_index(group_lg_index, j) + offset2)
              + offset ,rec.m_mol[num_snap].size() *
         (groups.atom_index(group_lg_index, j+1) + offset2)
              + offset ,rec.m_mol[num_snap].size() *
         (groups.atom_index(group_lg_index, j+2) + offset2)
              + offset ,rec.m_mol[num_snap].size() *
         (groups.atom_index(group_lg_index, j+3) + offset2)
              + offset };*/
      size_t idx0 =
          mol_size_stride * (groups.atom_index(group_lg_index, j) + offset2) +
          offset;
      size_t idx1 = mol_size_stride *
                        (groups.atom_index(group_lg_index, j + 1) + offset2) +
                    offset;
      size_t idx2 = mol_size_stride *
                        (groups.atom_index(group_lg_index, j + 2) + offset2) +
                    offset;
      size_t idx3 = mol_size_stride *
                        (groups.atom_index(group_lg_index, j + 3) + offset2) +
                    offset;

      __m128d p_elec01 = _mm_set_pd(elec_ptr[idx1], elec_ptr[idx0]);
      __m128d p_vdw01 = _mm_set_pd(vdw_ptr[idx1], vdw_ptr[idx0]);
      __m128d p0 = _mm_unpacklo_pd(p_vdw01, p_elec01); // {elec0, vdw0}
      __m128d p1 = _mm_unpackhi_pd(p_vdw01, p_elec01); // {elec1, vdw1}

      __m128d p_elec23 = _mm_set_pd(elec_ptr[idx3], elec_ptr[idx2]);
      __m128d p_vdw23 = _mm_set_pd(vdw_ptr[idx3], vdw_ptr[idx2]);
      __m128d p2 = _mm_unpacklo_pd(p_vdw23, p_elec23); // {elec2, vdw2}
      __m128d p3 = _mm_unpackhi_pd(p_vdw23, p_elec23); // {elec3, vdw3}

      __m256d v01 = _mm256_set_m128d(p1, p0);
      __m256d v23 = _mm256_set_m128d(p3, p2);
      v_total_sum = _mm256_add_pd(v_total_sum, v01);
      v_total_sum = _mm256_add_pd(v_total_sum, v23);

      /*force_energy_add_intrin(CachedData, pos, energy_force_out.e_e,
         energy_force_out.e_v, energy_force_out.F_elec, energy_force_out.F_vdw,
         rec.parameters, lg.parameters, rec, lg, num_snap, Arguments, offset,
         groups.atom_index(group_lg_index, j) + offset2);*/
    }
    // if (size_lg_cal * 4 < groups.groups_atom_number(group_lg_index))
    //	for (size_t j{ size_lg_cal * 4 }; j <
    // groups.groups_atom_number(group_lg_index); ++j)
    for (; j < size_lg_group; ++j) {
      force_energy_add(
          CachedData,
          rec.m_mol[num_snap].size() *
                  (groups.atom_index(group_lg_index, j) + offset2) +
              offset,
          energy_force_out.e_e, energy_force_out.e_v, energy_force_out.F_elec,
          energy_force_out.F_vdw, rec.parameters, lg.parameters, rec, lg,
          num_snap, Arguments, offset,
          groups.atom_index(group_lg_index, j) + offset2);
    }
  }
  double temp[4];
  _mm256_storeu_pd(temp, v_total_sum);
  energy_force_out.e_e += (temp[0] + temp[2]);

  energy_force_out.e_v += (temp[1] + temp[3]);
  return 0;
}

int cal_one_sub_group_multi_threads(
    groups_cal &groups, int group_rec_index, size_t sub1_pos,
    int group_lg_index, size_t sub2_pos, const Multi_MOL &rec,
    const Multi_MOL &lg, const cache_sigma_epsilon &CachedData,
    energy_res_base &energy_force_out, const arguments &Arguments,
    TitanExecutor &executor, size_t num_snap, size_t num_thread) {
  vector<energy_res_base> energy_force_out_mthreads(num_thread);
  energy_force_out.time = rec.m_mol[num_snap].time_of_snap;

  executor.parallel_do(num_thread, [&](size_t i) {
    sup_cal_one_subgroup_optimized(
        groups, group_rec_index, sub1_pos, group_lg_index, sub2_pos, rec, lg,
        CachedData, energy_force_out_mthreads[i], Arguments, num_snap,
        i * groups.groups_atom_number(group_rec_index) / num_thread,
        (i + 1) * groups.groups_atom_number(group_rec_index) / num_thread);
  });

  for (auto &ef_out : energy_force_out_mthreads) {
    energy_force_out += ef_out;
  }
  return 0;
}

int cal_one_sub_group_single_thread(groups_cal &groups, int group_rec_index,
                                    size_t sub1_pos, int group_lg_index,
                                    size_t sub2_pos, const Multi_MOL &rec,
                                    const Multi_MOL &lg,
                                    const cache_sigma_epsilon &CachedData,
                                    energy_res_base &energy_force_out,
                                    const arguments &Arguments,
                                    size_t num_snap) {
  energy_force_out.time = rec.m_mol[num_snap].time_of_snap;

  sup_cal_one_subgroup_optimized(groups, group_rec_index, sub1_pos,
                                 group_lg_index, sub2_pos, rec, lg, CachedData,
                                 energy_force_out, Arguments, num_snap, 0,
                                 groups.groups_atom_number(group_rec_index));

  return 0;
}

int cal_sub_groups_1_file(groups_cal &groups, const Multi_MOL &rec,
                          const Multi_MOL &lg,
                          const cache_sigma_epsilon &CachedData,
                          vector<energy_groups> &Energy_groups_out,
                          size_t num_snap, arguments &Arguments,
                          TitanExecutor &executor) {
  for (size_t i{}; i < groups.subgroup1_size(); ++i)
    for (size_t j{}; j < groups.subgroup2_size(); ++j) {
      energy_res_base Egroup_temp;

      if (groups.groups_atom_number(groups.subgroup_1_index(i)) < 1000 &&
          groups.groups_atom_number(groups.subgroup_2_index(j)) < 1000)
        cal_one_sub_group_single_thread(
            groups, groups.subgroup_1_index(i), i, groups.subgroup_2_index(j),
            j, rec, lg, CachedData, Egroup_temp, Arguments, num_snap);
      else
        cal_one_sub_group_multi_threads(groups, groups.subgroup_1_index(i), i,
                                        groups.subgroup_2_index(j), j, rec, lg,
                                        CachedData, Egroup_temp, Arguments,
                                        executor, num_snap, Arguments.nt);

      bool check_in{false};
      for (size_t k{}; k < Energy_groups_out.size(); ++k)
        if (groups.group_name(groups.subgroup_1_index(i)) + "-" +
                groups.group_name(groups.subgroup_2_index(j)) ==
            Energy_groups_out[k].group_name) {
          check_in = true;
          break;
        }

      if (check_in == false) {
        Energy_groups_out.emplace_back(
            Egroup_temp, groups.group_name(groups.subgroup_1_index(i)),
            groups.group_name(groups.subgroup_2_index(j)));
      } else
        Energy_groups_out[i * groups.subgroup2_size() + j].base.emplace_back(
            Egroup_temp);
    }
  return 0;
}

int Energy_force_subgroups(groups_cal &groups, const Multi_MOL &rec,
                           const Multi_MOL &lg,
                           const cache_sigma_epsilon &CachedData,
                           vector<vector<energy_groups>> &EF_subgroups,
                           vector<energy_groups> &Energy_groups_out,
                           size_t num_snap, size_t MAX_NUM,
                           arguments &Arguments, TitanExecutor &executor) {
  cal_sub_groups_1_file(groups, rec, lg, CachedData, Energy_groups_out,
                        num_snap, Arguments, executor);

  if (num_snap == MAX_NUM - 1)
    EF_subgroups.emplace_back(Energy_groups_out);
  return 0;
}

int average_EFgroups(vector<vector<energy_groups>> &EF_groups,
                     multi_value_energy_force &average_EF_groups_alltrajs,
                     multi_value_energy_force &SD_EF_groups_alltrajs,
                     multi_value_energy_force &average_EF_groups_pertraj,
                     multi_value_energy_force &SD_EF_groups_pertraj,
                     vector<multi_value_energy_force> &average_time_EF_groups,
                     vector<multi_value_energy_force> &SD_time_EF_groups,
                     arguments &Arguments) {
  if (EF_groups.empty())
    return 1;

  average_time_EF_groups =
      vector<multi_value_energy_force>(EF_groups[0].size());
  SD_time_EF_groups = vector<multi_value_energy_force>(EF_groups[0].size());

  average_EF_groups_alltrajs.values =
      vector<value_energy_force>(EF_groups[0].size());
  SD_EF_groups_alltrajs.values =
      vector<value_energy_force>(EF_groups[0].size());

  for (size_t k{}; k < EF_groups.size(); ++k) {
    for (size_t i{}; i < EF_groups[k].size(); ++i) {
      value_energy_force pertraj_ave{}, pertraj_sd{};
      for (size_t j{}; j < EF_groups[k][i].base.size(); ++j) {
        if (j >= average_time_EF_groups[i].values.size()) {
          average_time_EF_groups[i].values.emplace_back(
              EF_groups[k][i].base[j]);
          SD_time_EF_groups[i].values.emplace_back(value_energy_force());
        } else {
          average_time_EF_groups[i].values[j] += EF_groups[k][i].base[j];
        }

        pertraj_ave += EF_groups[k][i].base[j];
      }

      pertraj_ave /= pertraj_ave.sz;

      for (size_t j{}; j < EF_groups[k][i].base.size(); ++j) {
        value_energy_force temp{pertraj_ave - EF_groups[k][i].base[j]};
        pertraj_sd += temp.multiply(temp);
      }

      pertraj_sd /= pertraj_ave.sz;
      if (Arguments.flag_use_CI == true) {
        pertraj_sd /= pertraj_ave.sz;
        pertraj_sd = pertraj_sd.sqrt().multiply(1.96);
      } else
        pertraj_sd = pertraj_sd.sqrt();

      average_EF_groups_pertraj.values.emplace_back(pertraj_ave);
      SD_EF_groups_pertraj.values.emplace_back(pertraj_sd);
    }
  }

  for (size_t i{}; i < EF_groups[0].size(); ++i) {
    for (size_t k{}; k < EF_groups.size(); ++k) {
      average_EF_groups_alltrajs.values[i] +=
          average_EF_groups_pertraj.values[k * EF_groups[0].size() + i];
    }
    average_EF_groups_alltrajs.values[i] /=
        average_EF_groups_alltrajs.values[i].sz;
  }

  for (size_t i{}; i < EF_groups[0].size(); ++i) {
    for (size_t k{}; k < EF_groups.size(); ++k) {
      value_energy_force temp{
          average_EF_groups_pertraj.values[k * EF_groups[0].size() + i] -
          average_EF_groups_alltrajs.values[i]};
      SD_EF_groups_alltrajs.values[i] += temp.multiply(temp);
    }
    SD_EF_groups_alltrajs.values[i] /= average_EF_groups_alltrajs.values[i].sz;
    if (Arguments.flag_use_CI == true) {
      SD_EF_groups_alltrajs.values[i] /=
          average_EF_groups_alltrajs.values[i].sz;
      SD_EF_groups_alltrajs.values[i] =
          SD_EF_groups_alltrajs.values[i].sqrt().multiply(1.96);
    } else
      SD_EF_groups_alltrajs.values[i] = SD_EF_groups_alltrajs.values[i].sqrt();

    for (size_t j{}; j < EF_groups[0][i].base.size(); ++j) {
      average_time_EF_groups[i].values[j] /=
          average_time_EF_groups[i].values[j].sz;
      for (size_t k{}; k < EF_groups.size(); ++k) {
        if (j >= EF_groups[k][i].base.size())
          break;
        value_energy_force temp{average_time_EF_groups[i].values[j] -
                                EF_groups[k][i].base[j]};
        SD_time_EF_groups[i].values[j] += temp.multiply(temp);
      }
      SD_time_EF_groups[i].values[j] /= SD_time_EF_groups[i].values[j].sz;
      if (Arguments.flag_use_CI == true) {
        SD_time_EF_groups[i].values[j] /= SD_time_EF_groups[i].values[j].sz;
        SD_time_EF_groups[i].values[j] =
            SD_time_EF_groups[i].values[j].sqrt().multiply(1.96);
      } else
        SD_time_EF_groups[i].values[j] = SD_time_EF_groups[i].values[j].sqrt();
    }
  }
  return 0;
}

int print_value_energy_force(arguments &Arguments, fstream &out_all,
                             string error_type, size_t i,
                             multi_value_energy_force &average,
                             multi_value_energy_force &SD) {
  if (average.values.size() <= i)
    return 1;

  out_all << "Average Electrostatic energy: "
          << average.values[i].E_elec / Arguments.eu_par << " (" << Arguments.eu
          << "). " << error_type << ": "
          << SD.values[i].E_elec / Arguments.eu_par << " (" << Arguments.eu
          << ")\n"
          << "Average van der Waals energy: "
          << average.values[i].E_vdw / Arguments.eu_par << " (" << Arguments.eu
          << "). " << error_type << ": "
          << SD.values[i].E_vdw / Arguments.eu_par << " (" << Arguments.eu
          << ")\n"
          << "Average total energy: "
          << (average.values[i].E_elec + average.values[i].E_vdw) /
                 Arguments.eu_par
          << " (" << Arguments.eu << "). " << error_type << ": "
          << (SD.values[i].E_elec + SD.values[i].E_vdw) / Arguments.eu_par
          << " (" << Arguments.eu << ")\n";
  if (Arguments.flag_cal_force == true)
    out_all << "Average Electrostatic force: "
            << average.values[i].Modul_F_elec * Arguments.fu_par << " ("
            << Arguments.fu << "). " << error_type << ": "
            << SD.values[i].Modul_F_elec * Arguments.fu_par << " ("
            << Arguments.fu << ")\n"
            << "Average van der Waals force: "
            << average.values[i].Modul_F_vdw * Arguments.fu_par << " ("
            << Arguments.fu << "). " << error_type << ": "
            << SD.values[i].Modul_F_vdw * Arguments.fu_par << " ("
            << Arguments.fu << ")\n"
            << "Average total non-bonded force: "
            << average.values[i].Modul_F_total * Arguments.fu_par << " ("
            << Arguments.fu << "). " << error_type << ": "
            << SD.values[i].Modul_F_total * Arguments.fu_par << " ("
            << Arguments.fu << ")\n";
  return 0;
}

int post_process_Energy_force_subgroups(
    vector<double> &time, vector<vector<energy_groups>> &EF_groups,
    folders &paths, arguments &Arguments, info &info_output) {
  if (EF_groups.empty())
    return 1;

  multi_value_energy_force average_EF_groups_alltrajs, SD_EF_groups_alltrajs,
      average_EF_groups_pertraj, SD_EF_groups_pertraj;
  vector<multi_value_energy_force> average_time_EF_groups, SD_time_EF_groups;

  average_EFgroups(EF_groups, average_EF_groups_alltrajs, SD_EF_groups_alltrajs,
                   average_EF_groups_pertraj, SD_EF_groups_pertraj,
                   average_time_EF_groups, SD_time_EF_groups, Arguments);

  string error_type{Arguments.flag_use_CI == true ? "95% CI"
                                                  : "Standard deviation"};

  for (size_t i{}; i < average_EF_groups_alltrajs.values.size(); ++i) {
    string name{"average_energy_force_" + EF_groups[0][i].group_name + ".dat"};
    fstream out_all(name, ios::out);
    print_value_energy_force(Arguments, out_all, error_type, i,
                             average_EF_groups_alltrajs, SD_EF_groups_alltrajs);
  }

  for (size_t i{}; i < paths.get_size(); ++i) {
    // if (i >= average_EF_groups_pertraj.values.size())break;

    for (size_t j{}; j < EF_groups[0].size(); ++j) {
      if (j >= EF_groups[0].size())
        break;
      fstream out_average_EF(paths.get_path(i) + "/average_energy_force_" +
                                 EF_groups[0][j].group_name + ".dat",
                             ios::out);
      print_value_energy_force(Arguments, out_average_EF, error_type,
                               i * EF_groups[0].size() + j,
                               average_EF_groups_pertraj, SD_EF_groups_pertraj);
    }
  }

  for (size_t i{}; i < average_time_EF_groups.size(); ++i) {
    fstream out_ener_elec_xvg("energy_electrostatic_time_" +
                                  EF_groups[0][i].group_name + ".xvg",
                              ios::out),
        out_ener_vdw_xvg(
            "energy_vdw_time_" + EF_groups[0][i].group_name + ".xvg", ios::out),
        out_ener_total_xvg("energy_total_time_" + EF_groups[0][i].group_name +
                               ".xvg",
                           ios::out),
        out_force_elec_xvg, out_force_vdw_xvg, out_force_total_xvg;

    out_ener_elec_xvg << "@    title \"Energy\"\n";
    out_ener_elec_xvg << "@    xaxis label \"Time (" << Arguments.tu << ")\"\n";
    out_ener_elec_xvg << "@    yaxis label \"Energy (" << Arguments.eu
                      << ")\"\n";
    out_ener_elec_xvg << "@ s0 legend \"Time\"\n";
    out_ener_elec_xvg << "@ s1 legend \"Electrostatic\"\n";
    out_ener_elec_xvg << "@ s2 legend \"" << error_type << "\"\n";

    out_ener_vdw_xvg << "@    title \"Energy\"\n";
    out_ener_vdw_xvg << "@    xaxis label \"Time (" << Arguments.tu << ")\"\n";
    out_ener_vdw_xvg << "@    yaxis label \"Energy (" << Arguments.eu
                     << ")\"\n";
    out_ener_vdw_xvg << "@ s0 legend \"Time\"\n";
    out_ener_vdw_xvg << "@ s1 legend \"van der Waals\"\n";
    out_ener_vdw_xvg << "@ s2 legend \"" << error_type << "\"\n";

    out_ener_total_xvg << "@    title \"Energy\"\n";
    out_ener_total_xvg << "@    xaxis label \"Time (" << Arguments.tu
                       << ")\"\n";
    out_ener_total_xvg << "@    yaxis label \"Energy (" << Arguments.eu
                       << ")\"\n";
    out_ener_total_xvg << "@ s0 legend \"Time\"\n";
    out_ener_total_xvg << "@ s1 legend \"Total energy\"\n";
    out_ener_total_xvg << "@ s2 legend \"" << error_type << "\"\n";

    if (Arguments.flag_cal_force == true) {
      out_force_elec_xvg.open("force_electrostatic_time_" +
                                  EF_groups[0][i].group_name + ".xvg",
                              ios::out);
      out_force_vdw_xvg.open(
          "force_vdw_time_" + EF_groups[0][i].group_name + ".xvg", ios::out);
      out_force_total_xvg.open(
          "force_total_time" + EF_groups[0][i].group_name + ".xvg", ios::out);
    }

    for (size_t j{}; j < average_time_EF_groups[i].values.size(); ++j) {
      out_ener_elec_xvg << setw(6) << left << time[j] << right << setw(17)
                        << average_time_EF_groups[i].values[j].E_elec /
                               Arguments.eu_par
                        << setw(17)
                        << SD_time_EF_groups[i].values[j].E_elec /
                               Arguments.eu_par
                        << '\n';
      out_ener_vdw_xvg << setw(6) << left << time[j] << right << setw(17)
                       << average_time_EF_groups[i].values[j].E_vdw /
                              Arguments.eu_par
                       << setw(17)
                       << SD_time_EF_groups[i].values[j].E_vdw /
                              Arguments.eu_par
                       << '\n';
      out_ener_total_xvg << setw(6) << left << time[j] << right << setw(17)
                         << (average_time_EF_groups[i].values[j].E_elec +
                             average_time_EF_groups[i].values[j].E_vdw) /
                                Arguments.eu_par
                         << setw(17)
                         << (SD_time_EF_groups[i].values[j].E_elec +
                             SD_time_EF_groups[i].values[j].E_vdw) /
                                Arguments.eu_par
                         << '\n';
      if (Arguments.flag_cal_force == true) {
        out_force_elec_xvg << setw(6) << left << time[j] << right << setw(17)
                           << average_time_EF_groups[i].values[j].Modul_F_elec *
                                  Arguments.fu_par
                           << setw(17)
                           << SD_time_EF_groups[i].values[j].Modul_F_elec *
                                  Arguments.fu_par
                           << '\n';
        out_force_vdw_xvg << setw(6) << left << time[j] << right << setw(17)
                          << average_time_EF_groups[i].values[j].Modul_F_vdw *
                                 Arguments.fu_par
                          << setw(17)
                          << SD_time_EF_groups[i].values[j].Modul_F_vdw *
                                 Arguments.fu_par
                          << '\n';
        out_force_total_xvg
            << setw(6) << left << time[j] << right << setw(17)
            << average_time_EF_groups[i].values[j].Modul_F_total *
                   Arguments.fu_par
            << setw(17)
            << SD_time_EF_groups[i].values[j].Modul_F_total * Arguments.fu_par
            << '\n';
      }
    }
  }
  return 0;
}

int sup_distance_one_subgroup(groups_cal &groups, int group_rec_index,
                              size_t sub1_pos, int group_lg_index,
                              size_t sub2_pos, const Multi_MOL &rec,
                              const Multi_MOL &lg, const arguments &Arguments,
                              size_t num_snap, size_t begin, size_t end,
                              double &min) {
  size_t offset1{groups.subgroup_1_offset(sub1_pos) -
                 groups.atom_index(group_rec_index, 0)},
      offset2{groups.subgroup_2_offset(sub2_pos) -
              groups.atom_index(group_lg_index, 0)};
  size_t size_lg_cal{groups.groups_atom_number(group_lg_index) / 4};

  min = 10e12;
  for (size_t i{begin}; i < end; ++i) {
    size_t atomIndex_i{groups.atom_index(group_rec_index, i)},
        offset{atomIndex_i + offset1};
    for (size_t j{}; j < groups.groups_atom_number(group_lg_index); ++j) {
      double dis{
          square(rec.m_mol[num_snap].pos.x[offset] -
                 lg.m_mol[num_snap]
                     .pos.x[groups.atom_index(group_lg_index, j) + offset2]) +
          square(rec.m_mol[num_snap].pos.y[offset] -
                 lg.m_mol[num_snap]
                     .pos.y[groups.atom_index(group_lg_index, j) + offset2]) +
          square(rec.m_mol[num_snap].pos.z[offset] -
                 lg.m_mol[num_snap]
                     .pos.z[groups.atom_index(group_lg_index, j) + offset2])};
      if (dis <= min)
        min = dis;
    }
  }
  min = sqrt(min);
  return 0;
}

int dis_sub_group_multi_threads(groups_cal &groups, int group_rec_index,
                                size_t sub1_pos, int group_lg_index,
                                size_t sub2_pos, const Multi_MOL &rec,
                                const Multi_MOL &lg, const arguments &Arguments,
                                TitanExecutor &executor, size_t num_snap,
                                size_t num_thread,
                                vector<pair<double, double>> &mindis_vec) {
  vector<double> mdis_v(num_thread);
  executor.parallel_do(num_thread, [&](size_t i) {
    sup_distance_one_subgroup(
        groups, group_rec_index, sub1_pos, group_lg_index, sub2_pos, rec, lg,
        Arguments, num_snap,
        i * groups.groups_atom_number(group_rec_index) / num_thread,
        (i + 1) * groups.groups_atom_number(group_rec_index) / num_thread,
        mdis_v[i]);
  });

  double min{mdis_v[0]};
  for (size_t i{}; i < num_thread; ++i)
    if (mdis_v[i] < min)
      min = mdis_v[i];
  mindis_vec.emplace_back(rec.m_mol[num_snap].time_of_snap, min);
  return 0;
}

int cal_mindis_sub_groups_1_file(groups_cal &groups, const Multi_MOL &rec,
                                 const Multi_MOL &lg,
                                 vector<mindistance_group> &MIN_DIS,
                                 size_t num_snap, arguments &Arguments,
                                 TitanExecutor &executor) {
  for (size_t i{}; i < groups.subgroup1_size(); ++i)
    for (size_t j{}; j < groups.subgroup2_size(); ++j) {
      vector<pair<double, double>> mindis_vec;
      dis_sub_group_multi_threads(groups, groups.subgroup_1_index(i), i,
                                  groups.subgroup_2_index(j), j, rec, lg,
                                  Arguments, executor, num_snap,
                                  Arguments.nt, mindis_vec);
      string group1name = groups.group_name(groups.subgroup_1_index(i));
      string group2name = groups.group_name(groups.subgroup_2_index(j));

      bool checkin{false};
      for (size_t k{}; k < MIN_DIS.size(); ++k)
        if (MIN_DIS[k].group1name == group1name &&
            MIN_DIS[k].group2name == group2name) {
          MIN_DIS[k].data.emplace_back(mindis_vec);
          checkin = true;
          break;
        }
      if (checkin == false)
        MIN_DIS.emplace_back(group1name, group2name, mindis_vec);
    }
  return 0;
}

int mindis_subgroups(groups_cal &groups, const Multi_MOL &rec,
                     const Multi_MOL &lg, size_t num_snap, size_t MAX_NUM,
                     arguments &Arguments, TitanExecutor &executor,
                     vector<mindistance_group> &MIN_DIS) {
  cal_mindis_sub_groups_1_file(groups, rec, lg, MIN_DIS, num_snap, Arguments,
                               executor);
  return 0;
}

int post_proc_mindis(groups_cal &groups, arguments &Arguments,
                     info &info_output, vector<mindistance_group> &MIN_DIS) {
  double min_val{10e14}, max_val{-1.};
  for (size_t i{}; i < MIN_DIS.size(); ++i) {
    for (size_t j{}; j < MIN_DIS[i].data.size(); ++j)
      for (size_t k{}; k < MIN_DIS[i].data[j].size(); ++k) {
        if (min_val > MIN_DIS[i].data[j][k].second)
          min_val = MIN_DIS[i].data[j][k].second;
        if (max_val < MIN_DIS[i].data[j][k].second)
          max_val = MIN_DIS[i].data[j][k].second;
      }
  }
  for (size_t i{}; i < MIN_DIS.size(); ++i) {
    fstream output(info_output.get_out_histogram_mindis() +
                       MIN_DIS[i].group1name + "-" + MIN_DIS[i].group2name +
                       ".dat",
                   ios::out);
    double delta{(max_val - min_val) / Arguments.number_mindis_bin};
    size_t total_snap{};
    vector<double> histo(Arguments.number_mindis_bin, {0.});
    for (size_t j{}; j < MIN_DIS[i].data.size(); ++j)
      for (size_t k{}; k < MIN_DIS[i].data[j].size(); ++k) {
        ++total_snap;
        size_t pos{static_cast<size_t>(
            (MIN_DIS[i].data[j][k].second - min_val) / delta)};
        if (pos >= Arguments.number_mindis_bin)
          pos = Arguments.number_mindis_bin - 1;

        histo[pos] += 1.;
      }
    for (size_t j{}; j < histo.size(); ++j) {
      output << setw(12) << left << j * delta + 0.5 * delta + min_val << right
             << setw(15) << histo[j] * 100. / total_snap << '\n';
    }
  }
  return 0;
}

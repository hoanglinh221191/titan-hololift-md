#ifndef TITAN_PBE_H
#define TITAN_PBE_H

#include "MOL.h"
#include "MOL_para.h"
#include "arguments.h"
#include "tabi/types.h"

#include <cstddef>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class TitanExecutor;
class EnergyDecompositionProvider;

namespace TitanPBE {

class AnalyzerImpl;

struct EDecompRow {
  std::string molecule;
  std::string unit_type;
  std::string chain;
  std::size_t resnr{0};
  std::size_t respos{0};
  std::string resname;
  std::size_t atom_index{0};
  std::string atom_name;
  double elec_kj_mol{0.0};
  double vdw_kj_mol{0.0};
  double polar_kj_mol{0.0};
  double nonpolar_kj_mol{0.0};
};

using EDecompSink = std::function<void(const std::vector<EDecompRow> &)>;

class Analyzer {
public:
  Analyzer();
  ~Analyzer();

  Analyzer(Analyzer &&) noexcept;
  Analyzer &operator=(Analyzer &&) noexcept;
  Analyzer(const Analyzer &) = delete;
  Analyzer &operator=(const Analyzer &) = delete;

  void configure(const arguments &args);
  [[nodiscard]] bool enabled() const noexcept;
  void set_edecomp_sink(EDecompSink sink);

  void begin_trajectory(std::size_t trajectory_id,
                        const std::string &source_label);
  void end_trajectory(std::fstream &log);

  FrameResult consume_frame(const MOL &mol1, const Multi_MOL_parameters &params1,
                            const MOL *mol2,
                            const Multi_MOL_parameters *params2,
                            std::size_t trajectory_id,
                            std::size_t frame_index,
                            std::size_t global_index,
                            TitanExecutor &executor, std::fstream &log,
                            const EnergyDecompositionProvider *energy_provider =
                                nullptr);

  void finalize(std::fstream &log);
  // Non-empty when a BIE frame did not succeed (status other than ok); the
  // caller exits non-zero after every module has written its outputs.
  [[nodiscard]] std::string failure_summary() const;

private:
  std::unique_ptr<AnalyzerImpl> impl_;
};

} // namespace TitanPBE

#endif

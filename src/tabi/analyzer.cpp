#include "../titan_pbe.h"

#include "analyzer_impl.h"

#include <memory>
#include <utility>

namespace TitanPBE {

Analyzer::Analyzer() : impl_(std::make_unique<AnalyzerImpl>()) {}

Analyzer::~Analyzer() = default;

Analyzer::Analyzer(Analyzer &&) noexcept = default;

Analyzer &Analyzer::operator=(Analyzer &&) noexcept = default;

void Analyzer::configure(const arguments &args) { impl_->configure(args); }

bool Analyzer::enabled() const noexcept { return impl_ && impl_->enabled(); }

void Analyzer::set_edecomp_sink(EDecompSink sink) {
  impl_->set_edecomp_sink(std::move(sink));
}

void Analyzer::begin_trajectory(std::size_t trajectory_id,
                                const std::string &source_label) {
  impl_->begin_trajectory(trajectory_id, source_label);
}

void Analyzer::end_trajectory(std::fstream &log) { impl_->end_trajectory(log); }

FrameResult Analyzer::consume_frame(
    const MOL &mol1, const Multi_MOL_parameters &params1, const MOL *mol2,
    const Multi_MOL_parameters *params2, std::size_t trajectory_id,
    std::size_t frame_index, std::size_t global_index, TitanExecutor &executor,
    std::fstream &log, const EnergyDecompositionProvider *energy_provider) {
  return impl_->consume_frame(mol1, params1, mol2, params2, trajectory_id,
                              frame_index, global_index, executor, log,
                              energy_provider);
}

void Analyzer::finalize(std::fstream &log) { impl_->finalize(log); }

} // namespace TitanPBE

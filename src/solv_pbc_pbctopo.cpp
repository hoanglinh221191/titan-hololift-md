#include "Multi_MOL.h"
#include "pbctopo/solve/solver.h"

// PBCTOPO image reconstruction. The solver lives in src/pbctopo/solve/:
// PbctopoSolveRun (whole-run state, setup, temporal pass, reports) and
// PbctopoFrameSolver (one frame, stage by stage).
int Multi_MOL::solve_pbc_pbctopo(Multi_MOL &mol2, arguments &Arguments,
                                 TitanExecutor *executor,
                                 size_t frame_report_offset) {
  return pbctopo_solve::PbctopoSolveRun(*this, mol2, Arguments, executor,
                                        frame_report_offset)
      .run();
}

#ifndef TITAN_TABI_FMM_KIFMM_CHANNEL_HYBRID_OPERATOR_H
#define TITAN_TABI_FMM_KIFMM_CHANNEL_HYBRID_OPERATOR_H

#include "tabi_fmm_kifmm_hierarchy_real_panel.h"

namespace titan::tabi::fmm_kifmm {

// E46 facade for the frozen dual-worklist channel-hybrid prepared operator.
// The public signatures remain in tabi_fmm_kifmm_hierarchy_real_panel.h for
// source compatibility; solver code includes this header to depend on the
// operator layer rather than the monolithic real-panel implementation.
// DualWorklistChannelHybridPreparedOperator
// prepare_dual_worklist_channel_hybrid_operator
// apply_prepared_dual_worklist_channel_hybrid_operator
// assemble_dual_worklist_channel_hybrid_tabi_system_matvec

} // namespace titan::tabi::fmm_kifmm

#endif // TITAN_TABI_FMM_KIFMM_CHANNEL_HYBRID_OPERATOR_H

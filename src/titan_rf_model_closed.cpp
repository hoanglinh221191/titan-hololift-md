// Closed stand-in for the bundled random-forest affinity model (KIN-10).
//
// The generated model in titan_rf_model.cpp is 199,809 lines of nested
// branches. It is withheld because it has no training set, feature units,
// cross-validation score or applicability domain on record, and because its
// leading feature is an absolute interaction energy, which makes the
// prediction depend on system size, force field and cutoff conventions rather
// than on binding alone. The engine therefore exports the five descriptors and
// emits no pKd.
//
// Linking this stub instead of the generated source keeps the symbol available
// so no caller needs an #if, while removing the real cost of carrying a model
// nothing is allowed to use: on GCC 13.3 the generated file takes about 6 s and
// 843 MB of peak compiler memory, and it contributes a 3.3 MB (clang) to 4.9 MB
// (GCC) object to the binary.
//
// Build both ways with TITAN_ENABLE_RF_MODEL:
//   make                          -> this stub (default, model closed)
//   make TITAN_ENABLE_RF_MODEL=1  -> the generated model
//
// NaN is deliberate. It is not a neutral-looking number that could be mistaken
// for a prediction, and it propagates visibly if the path is ever re-enabled
// without restoring the real model.

#include "titan_rf_model.h"

#include <limits>

double predict_rf_pKD(double * /*input*/) {
  return std::numeric_limits<double>::quiet_NaN();
}

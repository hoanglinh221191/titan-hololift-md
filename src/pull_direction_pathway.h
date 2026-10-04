#ifndef PULL_DIRECTION_PATHWAY_H
#define PULL_DIRECTION_PATHWAY_H

#include "Multi_MOL.h"
#include "cache_sigma_epsilon.h"
#include "differential_evolution.h"
#include "groups.h"
#include "info.h"
#include "titan_executor.h"

int run_pathway_pull_direction(Multi_MOL &rec, Multi_MOL &lg,
                               cache_sigma_epsilon &CachedData,
                               de_arg &DE_arguments, arguments &Arguments,
                               info &output_info, TitanExecutor &executor,
                               groups_cal &Groups_from_index_file);

#endif

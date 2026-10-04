#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <cstddef>

// Returns the available physical memory in bytes.
// Returns 0 if unable to determine.
size_t get_available_ram();

// Returns the total physical memory in bytes.
// Returns 0 if unable to determine.
size_t get_total_ram();

#endif // MEMORY_UTILS_H

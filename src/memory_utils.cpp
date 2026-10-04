#include "memory_utils.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <cstdlib> // For fallback includes
#include <iostream>


size_t get_available_ram() {
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return static_cast<size_t>(status.ullAvailPhys);
  }
#else
  // Linux/Unix implementation (simplified)
  long pages = sysconf(_SC_AVPHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
  }
#endif
  return 0;
}

size_t get_total_ram() {
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    return static_cast<size_t>(status.ullTotalPhys);
  }
#else
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
  }
#endif
  return 0;
}

#ifndef TITAN_PERFORMANCE_OBSERVER_H
#define TITAN_PERFORMANCE_OBSERVER_H

#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

class TitanPerformanceObserver {
public:
  enum class Mode { Off, Summary, Detail };

  struct ModuleRecord {
    std::string module;
    std::uint64_t frames_seen{0};
    std::uint64_t frames_analyzed{0};
    double wall_ms{0.0};
    double export_ms{0.0};
    std::uint64_t warning_count{0};
    std::uint64_t skipped_frame_count{0};
    std::map<std::string, double> counters;
  };

  struct FrameRecord {
    std::uint64_t frame{0};
    double time_ps{0.0};
    std::string module;
    double wall_ms{0.0};
    std::string phase{"compute"};
  };

  TitanPerformanceObserver() = default;

  void configure(const std::string &mode);
  [[nodiscard]] bool enabled() const noexcept { return mode_ != Mode::Off; }
  [[nodiscard]] bool detail_enabled() const noexcept {
    return mode_ == Mode::Detail;
  }
  [[nodiscard]] Mode mode() const noexcept { return mode_; }

  void begin_module_frame(const std::string &module, std::uint64_t frame,
                          double time_ps);
  void end_module_frame(const std::string &module, bool analyzed = true);
  void add_frame_elapsed(const std::string &module, std::uint64_t frame,
                         double time_ps, double wall_ms,
                         const std::string &phase = "compute",
                         bool analyzed = true);
  void add_export_elapsed(const std::string &module, double wall_ms);
  void add_counter(const std::string &module, const std::string &name,
                   double value);
  void add_warning(const std::string &module, std::uint64_t count = 1);
  void add_skipped_frame(const std::string &module,
                         std::uint64_t count = 1);

  void export_outputs(const std::string &summary_filename,
                      const std::string &modules_filename,
                      const std::string &frames_filename,
                      std::fstream &log) const;

private:
  struct ActiveTimer {
    std::uint64_t frame{0};
    double time_ps{0.0};
    std::chrono::steady_clock::time_point start;
  };

  ModuleRecord &module_record(const std::string &module);
  [[nodiscard]] const ModuleRecord *
  find_module_record(const std::string &module) const;

  Mode mode_{Mode::Off};
  std::map<std::string, ModuleRecord> modules_;
  std::vector<FrameRecord> frames_;
  std::unordered_map<std::string, ActiveTimer> active_;
};

#endif // TITAN_PERFORMANCE_OBSERVER_H

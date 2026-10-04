#include "titan_performance_observer.h"
#include "titan_ui.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

void write_csv_string(std::ostream &out, const std::string &value) {
  const bool needs_quotes = value.find_first_of(",\"\n\r") != std::string::npos;
  if (!needs_quotes) {
    out << value;
    return;
  }
  out << '"';
  for (const char c : value) {
    if (c == '"')
      out << "\"\"";
    else
      out << c;
  }
  out << '"';
}

} // namespace

void TitanPerformanceObserver::configure(const std::string &mode) {
  const std::string normalized = lower_copy(mode);
  if (normalized == "summary")
    mode_ = Mode::Summary;
  else if (normalized == "detail")
    mode_ = Mode::Detail;
  else
    mode_ = Mode::Off;
}

TitanPerformanceObserver::ModuleRecord &
TitanPerformanceObserver::module_record(const std::string &module) {
  ModuleRecord &record = modules_[module];
  if (record.module.empty())
    record.module = module;
  return record;
}

const TitanPerformanceObserver::ModuleRecord *
TitanPerformanceObserver::find_module_record(const std::string &module) const {
  const auto it = modules_.find(module);
  return it == modules_.end() ? nullptr : &it->second;
}

void TitanPerformanceObserver::begin_module_frame(const std::string &module,
                                                  std::uint64_t frame,
                                                  double time_ps) {
  if (!enabled())
    return;
  active_[module] = ActiveTimer{frame, time_ps,
                                std::chrono::steady_clock::now()};
  (void)module_record(module);
}

void TitanPerformanceObserver::end_module_frame(const std::string &module,
                                                bool analyzed) {
  if (!enabled())
    return;
  const auto it = active_.find(module);
  if (it == active_.end())
    return;
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - it->second.start);
  add_frame_elapsed(module, it->second.frame, it->second.time_ps,
                    elapsed.count(), "compute", analyzed);
  active_.erase(it);
}

void TitanPerformanceObserver::add_frame_elapsed(const std::string &module,
                                                 std::uint64_t frame,
                                                 double time_ps,
                                                 double wall_ms,
                                                 const std::string &phase,
                                                 bool analyzed) {
  if (!enabled())
    return;
  ModuleRecord &record = module_record(module);
  record.frames_seen += 1;
  if (analyzed)
    record.frames_analyzed += 1;
  else
    record.skipped_frame_count += 1;
  record.wall_ms += wall_ms;
  if (detail_enabled())
    frames_.push_back(FrameRecord{frame, time_ps, module, wall_ms, phase});
}

void TitanPerformanceObserver::add_export_elapsed(const std::string &module,
                                                  double wall_ms) {
  if (!enabled())
    return;
  ModuleRecord &record = module_record(module);
  record.export_ms += wall_ms;
  record.wall_ms += wall_ms;
  if (detail_enabled())
    frames_.push_back(FrameRecord{0, std::numeric_limits<double>::quiet_NaN(),
                                  module, wall_ms, "export"});
}

void TitanPerformanceObserver::add_counter(const std::string &module,
                                           const std::string &name,
                                           double value) {
  if (!enabled())
    return;
  module_record(module).counters[name] += value;
}

void TitanPerformanceObserver::add_warning(const std::string &module,
                                           std::uint64_t count) {
  if (!enabled())
    return;
  module_record(module).warning_count += count;
}

void TitanPerformanceObserver::add_skipped_frame(const std::string &module,
                                                 std::uint64_t count) {
  if (!enabled())
    return;
  module_record(module).skipped_frame_count += count;
}

void TitanPerformanceObserver::export_outputs(
    const std::string &summary_filename, const std::string &modules_filename,
    const std::string &frames_filename, std::fstream &log) const {
  if (!enabled())
    return;

  {
    std::ofstream out(summary_filename);
    if (!out) {
      TitanUI::log_kv(log, "Performance Summary",
                      "failed to open " + summary_filename);
      return;
    }
    out << "module: titan_performance_observer\n";
    out << "mode: " << (detail_enabled() ? "detail" : "summary") << '\n';
    out << "module_count: " << modules_.size() << '\n';
    out << "detail_rows: " << frames_.size() << '\n';
    out << "memory_note: peak scratch capacity is module-reported when "
           "available; otherwise unknown.\n";
    out << "compatibility_note: module-specific performance files are kept; "
           "this report is a shared cross-module observability layer.\n";
  }
  TitanUI::log_kv(log, "Performance Summary", summary_filename);

  {
    std::ofstream out(modules_filename);
    if (!out) {
      TitanUI::log_kv(log, "Performance Modules",
                      "failed to open " + modules_filename);
      return;
    }
    out << "module,frames_seen,frames_analyzed,wall_ms,avg_frame_ms,export_ms,"
           "warning_count,skipped_frame_count,counters\n";
    out << std::fixed << std::setprecision(6);
    for (const auto &[name, record] : modules_) {
      const double avg_frame_ms =
          record.frames_analyzed == 0
              ? std::numeric_limits<double>::quiet_NaN()
              : record.wall_ms / static_cast<double>(record.frames_analyzed);
      write_csv_string(out, name);
      out << ',' << record.frames_seen << ',' << record.frames_analyzed << ','
          << record.wall_ms << ',' << avg_frame_ms << ',' << record.export_ms
          << ',' << record.warning_count << ',' << record.skipped_frame_count
          << ',';
      std::ostringstream counters;
      bool first = true;
      counters << std::fixed << std::setprecision(6);
      for (const auto &[counter_name, value] : record.counters) {
        if (!first)
          counters << ';';
        first = false;
        counters << counter_name << '=' << value;
      }
      write_csv_string(out, counters.str());
      out << '\n';
    }
  }
  TitanUI::log_kv(log, "Performance Modules", modules_filename);

  if (detail_enabled()) {
    std::vector<FrameRecord> sorted_frames = frames_;
    std::sort(sorted_frames.begin(), sorted_frames.end(),
              [](const FrameRecord &a, const FrameRecord &b) {
                if (a.module != b.module)
                  return a.module < b.module;
                if (a.frame != b.frame)
                  return a.frame < b.frame;
                return a.phase < b.phase;
              });
    std::ofstream out(frames_filename);
    if (!out) {
      TitanUI::log_kv(log, "Performance Frames",
                      "failed to open " + frames_filename);
      return;
    }
    out << "module,frame,time_ps,phase,wall_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const FrameRecord &frame : sorted_frames) {
      write_csv_string(out, frame.module);
      out << ',' << frame.frame << ',' << frame.time_ps << ',';
      write_csv_string(out, frame.phase);
      out << ',' << frame.wall_ms << '\n';
    }
    TitanUI::log_kv(log, "Performance Frames", frames_filename);
  }
}

#ifndef TITAN_UI_H
#define TITAN_UI_H

#include <fstream>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

namespace TitanUI {

inline bool &progress_line_active() {
  static bool active = false;
  return active;
}

inline std::chrono::steady_clock::time_point &last_progress_render_time() {
  static auto last = std::chrono::steady_clock::time_point{};
  return last;
}

inline bool should_render_progress(bool force = false) {
  using clock = std::chrono::steady_clock;
  constexpr auto kProgressRefresh = std::chrono::milliseconds(500);

  auto &last = last_progress_render_time();
  const auto now = clock::now();
  if (force || last.time_since_epoch().count() == 0 ||
      now - last >= kProgressRefresh) {
    last = now;
    return true;
  }
  return false;
}

inline void prepare_message_line(std::ostream &out) {
  if (!progress_line_active())
    return;
  out << "\n\r\033[K";
  progress_line_active() = false;
}

inline std::string clamp_display_text(const std::string &text,
                                      std::size_t max_width) {
  if (text.size() <= max_width)
    return text;
  if (max_width <= 3)
    return text.substr(0, max_width);
  return text.substr(0, max_width - 3) + "...";
}

inline std::string compact_source_label(const std::string &path_text,
                                        std::size_t max_width = 18) {
  const std::filesystem::path path(path_text);
  std::string label;

  const auto filename = path.filename().string();
  const auto parent = path.parent_path().filename().string();
  if (!parent.empty() && parent != "." && parent != ".." && !filename.empty()) {
    label = (std::filesystem::path(parent) / filename).generic_string();
  } else if (!filename.empty()) {
    label = filename;
  } else {
    label = path.generic_string();
  }

  return clamp_display_text(label, max_width);
}

inline void print_runtime_module_summary(const std::string &requested,
                                         const std::string &active,
                                         const std::string &skipped = {},
                                         const std::string &streaming = {}) {
  prepare_message_line(std::cout);
  if (requested.empty() && active.empty() && skipped.empty() &&
      streaming.empty())
    return;

  std::cout << "\n\033[1;36m[TITAN Runtime]\033[0m\n";
  if (!requested.empty())
    std::cout << "  \033[1;33mRequested\033[0m : " << requested << "\n";
  if (!active.empty())
    std::cout << "  \033[1;32mActive   \033[0m : " << active << "\n";
  if (!streaming.empty())
    std::cout << "  \033[1;36mStreaming\033[0m : " << streaming << "\n";
  if (!skipped.empty())
    std::cout << "  \033[1;90mSkipped  \033[0m : " << skipped << "\n";
  std::cout << "\n";
}

inline void write_logo_art(std::ostream &out) {
    // Gradient Lửa (Vàng -> Đỏ sáng -> Đỏ sậm) cho TITAN, Cyan cho 2026
    out << "\033[1;33m" // Yellow (Đỉnh chữ TITAN)
        << "  ████████╗██╗████████╗ █████╗ ███╗   ██╗        \033[1;36m██████╗  ██████╗ ██████╗  ██████╗\n"
        << "\033[1;33m" // Khôi phục lại màu Vàng cho dòng 2 của TITAN
        << "  ╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║        \033[1;36m╚════██╗██╔═████╗╚════██╗██╔════╝\n"
        << "\033[1;31m" // Bright Red (Giữa chữ TITAN)
        << "     ██║   ██║   ██║   ███████║██╔██╗ ██║        \033[1;36m █████╔╝██║██╔██║ █████╔╝███████╗\n"
        << "\033[1;31m" 
        << "     ██║   ██║   ██║   ██╔══██║██║╚██╗██║        \033[1;36m██╔═══╝ ████╔╝██║██╔═══╝ ██╔═══██╗\n"
        << "\033[0;31m" // Dark Red (Chân chữ TITAN)
        << "     ██║   ██║   ██║   ██║  ██║██║ ╚████║        \033[1;36m███████╗╚██████╔╝███████╗╚██████╔╝\n"
        << "\033[0;31m" 
        << "     ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝        \033[1;36m╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝\n"
        << "\033[0m";  // Reset màu về mặc định
}

inline void write_logo_art_log(std::ostream &out) {
    // Standard ASCII (Font "Big") đảm bảo monospace tuyệt đối trong file text/log
    out << "  _______  _____  _______          _   _             ___   ___   ___   __ \n";
    out << " |__   __||_   _||__   __|   /\\   | \\ | |           |__ \\ / _ \\ |__ \\ / / \n";
    out << "    | |     | |     | |     /  \\  |  \\| |              ) | | | |   ) / /_ \n";
    out << "    | |     | |     | |    / /\\ \\ | . ` |             / /| | | |  / / '_ \\\n";
    out << "    | |    _| |_    | |   / ____ \\| |\\  |            / /_| |_| | / /| (_) |\n";
    out << "    |_|   |_____|   |_|  /_/    \\_\\_| \\_|           |____|\\___/ |____|\\___/\n";
}


inline void show_progress(const std::string &module_name, size_t current,
                          size_t total, const std::string &status_text = "",
                          const std::string &module_text = "") {
  if (total == 0)
    return;

  const bool is_final = (current + 1 >= total);
  if (!should_render_progress(is_final))
    return;

  progress_line_active() = true;
  float progress = static_cast<float>(current + 1) / total;
  int barWidth = 28;
  const std::string display_name = clamp_display_text(module_name, 18);
  const std::string display_status = clamp_display_text(status_text, 36);
  const std::string display_modules = clamp_display_text(module_text, 42);
  std::cout << "\r\033[1;36m[TITAN]\033[0m \033[1;33m" << std::setw(18)
            << std::left << display_name << "\033[0m [";

  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "\033[1;32m=\033[0m";
    else if (i == pos)
      std::cout << "\033[1;32m>\033[0m";
    else
      std::cout << " ";
  }

  std::cout << "] " << std::setw(3) << std::right << int(progress * 100.0)
            << "%";
  if (!display_status.empty())
    std::cout << " | " << display_status;
  if (!display_modules.empty())
    std::cout << " | Active: " << display_modules;
  std::cout << "\033[K" << std::flush;

  if (is_final) {
    std::cout << std::endl;
    progress_line_active() = false;
  }
}

inline void show_progress_exact(const std::string &module_name, size_t completed,
                                size_t total,
                                const std::string &status_text = "",
                                const std::string &module_text = "") {
  if (total == 0)
    return;

  const bool is_final = (completed >= total);
  if (!should_render_progress(is_final))
    return;

  progress_line_active() = true;
  const float progress =
      static_cast<float>(std::min(completed, total)) / static_cast<float>(total);
  const int barWidth = 28;
  const std::string display_name = clamp_display_text(module_name, 18);
  const std::string display_status = clamp_display_text(status_text, 36);
  const std::string display_modules = clamp_display_text(module_text, 42);

  std::cout << "\r\033[1;36m[TITAN]\033[0m \033[1;33m" << std::setw(18)
            << std::left << display_name << "\033[0m [";

  const int pos = static_cast<int>(barWidth * progress);
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "\033[1;32m=\033[0m";
    else if (i == pos && completed < total)
      std::cout << "\033[1;32m>\033[0m";
    else
      std::cout << " ";
  }

  std::cout << "] " << std::setw(3) << std::right
            << static_cast<int>(progress * 100.0f) << "%";
  if (!display_status.empty())
    std::cout << " | " << display_status;
  if (!display_modules.empty())
    std::cout << " | Active: " << display_modules;
  std::cout << "\033[K" << std::flush;

  if (is_final) {
    std::cout << std::endl;
    progress_line_active() = false;
  }
}

inline void show_continuous_progress(const std::string &module_name,
                                     size_t current_frame,
                                     const std::string &status_text = "",
                                     const std::string &module_text = "") {
  if (!should_render_progress())
    return;

  progress_line_active() = true;
  const std::string display_name = clamp_display_text(module_name, 18);
  const std::string display_status = clamp_display_text(status_text, 44);
  const std::string display_modules = clamp_display_text(module_text, 40);
  std::ostringstream oss;
  oss << "Frame " << current_frame;
  if (!display_status.empty())
    oss << " | " << display_status;

  std::cout << "\r\033[1;36m[TITAN]\033[0m \033[1;33m" << std::setw(18)
            << std::left << display_name << "\033[0m "
            << "\033[1;32m" << oss.str() << "\033[0m";
  if (!display_modules.empty())
    std::cout << " | Active: " << display_modules;
  std::cout << "\033[K" << std::flush;
}

inline void print_module_header(const std::string &header_name) {
  prepare_message_line(std::cout);
  std::cout << "\n\033[1;35m>>> " << header_name << " <<<\033[0m\n";
}

inline void print_logo() {
  prepare_message_line(std::cout);
  std::cout << "\n";
  write_logo_art(std::cout);
  std::cout
      << "  \033[1;33mThermodynamic & Interaction Trajectory ANalyzer\033[0m\n";
  std::cout
      << "  "
         "\033[1;32m===============================================\033[0m\n\n";
}

inline void log_logo(std::fstream &log_file) {
  if (!log_file.is_open())
    return;
  log_file << "\n";
  write_logo_art_log(log_file);
  log_file << "  Thermodynamic & Interaction Trajectory ANalyzer\n";
  log_file << "  ===============================================\n\n";
}

inline void log_section(std::fstream &log_file, const std::string &title) {
  if (!log_file.is_open())
    return;
  log_file << "\n=============================================================="
              "==================\n";
  log_file << "[ " << title << " ]\n";
  log_file << "================================================================"
              "================\n";
}

inline void log_kv(std::fstream &log_file, const std::string &key,
                   const std::string &value) {
  if (!log_file.is_open())
    return;
  log_file << std::left << std::setw(30) << key << " : " << value << "\n";
}

inline void log_message(std::fstream &log_file, const std::string &message) {
  if (!log_file.is_open())
    return;
  log_file << message << "\n";
}

inline void log_time(std::fstream &log_file, const std::string &task,
                     double time_sec) {
  if (!log_file.is_open())
    return;
  log_file << "  - " << std::left << std::setw(40) << task << " : "
           << std::fixed << std::setprecision(3) << std::right << std::setw(10)
           << time_sec << " s\n";
}

inline void print_error(const std::string &error_msg) {
  prepare_message_line(std::cerr);
  std::cerr << "\n\033[1;31m[ERROR]\033[0m " << error_msg << "\n";
}

inline void print_warning(const std::string &warning_msg) {
  prepare_message_line(std::cout);
  std::cout << "\033[1;33m[WARNING]\033[0m " << warning_msg << "\n";
}

inline void print_info(const std::string &info_msg) {
  prepare_message_line(std::cout);
  std::cout << info_msg << "\n";
}

} // namespace TitanUI

#endif // TITAN_UI_H

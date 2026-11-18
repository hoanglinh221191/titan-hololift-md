// fflib_assignment_report.tsv and the charge warnings of the assigner.
#include "topology/fflib_detail.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace TitanFfLib::detail {
namespace {

std::mutex &report_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::string report_escape(std::string value) {
  for (char &ch : value) {
    if (ch == '\t' || ch == '\n' || ch == '\r')
      ch = ' ';
  }
  return value;
}

} // namespace

std::string make_report_row(std::string_view molecule, const ResidueSpan &span,
                            std::string_view atom, std::string_view action,
                            std::string_view templ,
                            std::string_view atom_type, double charge,
                            double sigma, double epsilon,
                            std::string_view message) {
  std::ostringstream out;
  out << report_escape(std::string(molecule)) << '\t'
      << report_escape(span.chain) << '\t' << span.resnr << '\t'
      << report_escape(span.resname) << '\t'
      << report_escape(std::string(atom)) << '\t'
      << report_escape(std::string(action)) << '\t'
      << report_escape(std::string(templ)) << '\t'
      << report_escape(std::string(atom_type)) << '\t' << std::setprecision(10)
      << charge << '\t' << sigma << '\t' << epsilon << '\t'
      << report_escape(std::string(message)) << '\n';
  return out.str();
}

void append_report(const std::vector<ResiduePlan> &plans,
                   std::string_view molecule_label, std::size_t atom_count,
                   std::size_t final_count, bool strict_missing) {
  std::lock_guard<std::mutex> lock(report_mutex());
  const std::filesystem::path filename("fflib_assignment_report.tsv");
  static bool report_initialized = false;

  const std::ios::openmode mode =
      report_initialized ? (std::ios::out | std::ios::app)
                         : (std::ios::out | std::ios::trunc);
  std::fstream out(filename.string(), mode);
  if (!out.is_open())
    return;
  if (!report_initialized) {
    out << "molecule\tchain\tresnr\tresname\tatom\taction\ttemplate"
        << "\tatom_type\tcharge\tsigma\tepsilon\tmessage\n";
    report_initialized = true;
  }

  std::size_t assigned = 0;
  std::size_t unmatched = 0;
  std::size_t missing_vdw = 0;
  std::size_t added_h = 0;
  for (const auto &plan : plans) {
    assigned += plan.assigned;
    unmatched += plan.unmatched;
    missing_vdw += plan.missing_vdw;
    for (const auto &h : plan.hydrogens) {
      if (h.can_place)
        ++added_h;
    }
    for (const auto &row : plan.report_rows)
      out << row;
  }

  ResidueSpan summary;
  summary.chain = "*";
  summary.resname = "*";
  out << make_report_row(molecule_label, summary, "*",
                         strict_missing && unmatched > 0 ? "summary_failed"
                                                         : "summary",
                         "*", "*", 0.0, 0.0, 0.0,
                         "input_atoms=" + std::to_string(atom_count) +
                             "; final_atoms=" + std::to_string(final_count) +
                             "; assigned=" + std::to_string(assigned) +
                             "; unmatched=" + std::to_string(unmatched) +
                             "; missing_vdw=" + std::to_string(missing_vdw) +
                             "; added_h=" + std::to_string(added_h));
}

void warn_fractional_charge(const Multi_MOL &mol,
                            const std::vector<ResiduePlan> &plans,
                            std::string_view molecule_label) {
  // ENE-09: a complete molecule carries an integer charge. A fractional
  // total points at unassigned atoms, a terminus left without its patch, or a
  // residue template whose own charges do not add up (for example the HISE
  // of an old OPLS-AA rtp, CE1 0.182 instead of 0.295); name those templates.
  double total_charge = 0.0;
  for (std::size_t i = 0; i < mol.info.size() && i < mol.parameters.size(); ++i)
    total_charge += mol.parameters[i].get_charge();
  if (std::isfinite(total_charge) &&
      std::fabs(total_charge - std::round(total_charge)) > 1.0e-3) {
    std::vector<std::string> fractional_templates;
    for (const auto &plan : plans) {
      if (plan.templ == nullptr)
        continue;
      double template_charge = 0.0;
      for (const auto &atom : plan.templ->atoms)
        template_charge += atom.charge;
      if (std::fabs(template_charge - std::round(template_charge)) > 1.0e-3) {
        std::ostringstream label;
        label << plan.templ_name << " (" << std::setprecision(4)
              << template_charge << " e)";
        if (std::find(fractional_templates.begin(), fractional_templates.end(),
                      label.str()) == fractional_templates.end())
          fractional_templates.push_back(label.str());
      }
    }
    std::ostringstream message;
    message << "WARNING (" << molecule_label << "): lib_ff charges add up to "
            << std::setprecision(6) << total_charge
            << " e, not an integer";
    if (!fractional_templates.empty()) {
      message << "; residue templates whose charges do not add up to an "
                 "integer:";
      for (const auto &label : fractional_templates)
        message << ' ' << label;
    }
    message << ". See fflib_assignment_report.tsv\n";
    std::cout << message.str();
  }
}

} // namespace TitanFfLib::detail

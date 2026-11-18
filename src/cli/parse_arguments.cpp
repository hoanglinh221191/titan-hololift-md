#include "MOL.h"
#include "arguments.h"
#include "cli/option_validation.h"
#include "functions.h"
#include "titan_build_info.h"
#include "titan_ui.h"
#include "titan_version_info.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
int arguments::parse_arguments(int argc, char **argv) {
  if (!parsing_arg_file_tokens) {
    const titan_cli::ValidationResult validation =
        titan_cli::validate_terminal_tokens(argc, argv);
    titan_cli::emit_diagnostics(validation);
    if (!validation.ok())
      return titan_cli::kInvalidArguments;
  }
  // The option helpers (parse_helpers.cpp) read the tokens of this call;
  // parse_arg_file() re-enters parse_arguments() with its own tokens.
  struct TokenScope {
    arguments &self;
    int saved_argc;
    char **saved_argv;
    TokenScope(arguments &owner, int count, char **tokens)
        : self(owner), saved_argc(owner.cli_argc_), saved_argv(owner.cli_argv_) {
      owner.cli_argc_ = count;
      owner.cli_argv_ = tokens;
    }
    ~TokenScope() {
      self.cli_argc_ = saved_argc;
      self.cli_argv_ = saved_argv;
    }
  } token_scope(*this, argc, argv);

  // The option families (src/cli/parse_options_*.cpp) in the order of the
  // former single loop body: the first that consumes argv[i] ends the token.
  using OptionFamily = OptionStep (arguments::*)(int, char **, int &,
                                                 const std::string &,
                                                 std::string &);
  static constexpr OptionFamily option_families[] = {
      &arguments::parse_options_general,
      &arguments::parse_options_pbe,
      &arguments::parse_options_bie,
      &arguments::parse_options_pbe_solver,
      &arguments::parse_options_analysis,
      &arguments::parse_options_llps_topology,
      &arguments::parse_options_inputs,
      &arguments::parse_options_kinetics,
  };

  for (int i = 1; i < argc; ++i) {
    // Remember what is being parsed so a std::sto* failure can name the option
    // instead of surfacing a bare "stoi"/"stod".
    titan_cli::active_option_token() = argv[i];
    const std::string arg = normalize_pbe_cli_alias(string(argv[i]));
    std::string common_value;

    bool consumed = false;
    for (const OptionFamily family : option_families) {
      const OptionStep step = (this->*family)(argc, argv, i, arg, common_value);
      if (step.kind == OptionStep::stop)
        return step.code;
      if (step.kind == OptionStep::consumed) {
        consumed = true;
        break;
      }
    }
    if (consumed)
      continue;
    command += argv[i];
    command += " ";
  }
  // Parsing finished without throwing, so a later exception from the engine
  // must not be attributed to the last option inspected.
  titan_cli::active_option_token().clear();

  if (flag_topo_1 && flag_topo_2 && topo_name == topo_name_2) {
    flag_use_common_topo = true;
  }

  if (hotspot.flag_hotspot_report) {
    hotspot.source = lowercase_copy(hotspot.source);
    hotspot.rank_by = lowercase_copy(hotspot.rank_by);
    if (!hotspot.rank_by_explicit) {
      if (hotspot.source == "cas")
        hotspot.rank_by = "cas-ddg";
      else if (hotspot.source == "edecomp")
        hotspot.rank_by = "edecomp-abs-mean";
      else
        hotspot.rank_by = "combined";
    }
    if (const std::string validation_error = hotspot.validation_error();
        !validation_error.empty()) {
      TitanUI::print_error("Hotspot report: " + validation_error);
      exit(EXIT_FAILURE);
    }
    if (hotspot.source == "cas" || hotspot.source == "both")
      flag_cas = true;
    if (hotspot.source == "edecomp" || hotspot.source == "both") {
      edecomp.flag_edecomp = true;
      if (edecomp.level == "atom") {
        TitanUI::print_error("Hotspot report: EDecomp hotspot ranking requires "
                             "-edecomp-level=residue or -edecomp-level=both.");
        exit(EXIT_FAILURE);
      }
    }
  }

  if (edecomp.flag_edecomp) {
    const std::string validation_error = edecomp.validation_error();
    if (!validation_error.empty()) {
      TitanUI::print_error("Energy decomposition: " + validation_error);
      exit(EXIT_FAILURE);
    }
    if (edecomp.source == "gbsa" && !flag_gbsa) {
      TitanUI::print_error(
          "Energy decomposition: -edecomp-source=gbsa requires -gbsa.");
      exit(EXIT_FAILURE);
    }
    if (edecomp.source == "pbsa" && !pbe.pbsa) {
      TitanUI::print_error(
          "Energy decomposition: -edecomp-source=pbsa requires -pbsa=true.");
      exit(EXIT_FAILURE);
    }
  }

  if (pbc.solv_pbctopo) {
    const bool conflict_with_reference =
        pbc.solv_pbc == 1 || pbc.solv_pbc_int == 1 || pbc.solv_pbcn ||
        pbc.pbc_aggressive || pbc.pbc_mem || pbc.flag_pbc_ref || pbc.pbc_envi ||
        pbc.solv_pbc_clust;
    if (conflict_with_reference) {
      TitanUI::print_error(
          "The option -pbctopo cannot be combined with reference/legacy PBC "
          "modes.");
      return titan_cli::kInvalidArguments;
    }
  }

  return 0;
}
int arguments::parse_arguments_interactive(int argc_in) {
  int flag;
  string force_f;
  do {
    cout << "Please choose the force field that you want to use:" << '\n';
    cout << "1. AMBER99SB" << '\n';
    cout << "2. AMBER99SB-ILDN" << '\n';
    cout << "3. AMBER99SB-BSC1" << '\n';
    cout << "4. AMBER99SB-DISP" << '\n';
    cout << "5. AMBER14SB" << '\n';
    cout << "6. AMBER19SB" << '\n';
    cout << "7. GROMOS43A1" << '\n';
    cout << "8. GROMOS53A6" << '\n';
    cout << "9. GROMOS54A7" << '\n';
    cout << "10. CHARMM27" << '\n';
    cout << "11. CHARMM36" << '\n';
    cout << "12. OPLSAA" << '\n';
    getline(cin, force_f);
    istringstream st(force_f);
    st >> flag;
  } while (flag > 12 || flag < 1);
  if (flag == 1) {
    force_f = "AMBER99SB";
  } else if (flag == 2) {
    force_f = "AMBER99SB-ILDN";
  } else if (flag == 3) {
    force_f = "AMBER99SB-BSC1";
  } else if (flag == 4) {
    force_f = "AMBER99SB-DISP";
  } else if (flag == 5) {
    force_f = "AMBER14SB";
  } else if (flag == 6) {
    force_f = "AMBER19SB";
  } else if (flag == 7) {
    force_f = "GROMOS43A1";
  } else if (flag == 8) {
    force_f = "GROMOS53A6";
  } else if (flag == 9) {
    force_f = "GROMOS54A7";
  } else if (flag == 10) {
    force_f = "CHARMM27";
  } else if (flag == 11) {
    force_f = "CHARMM36";
  } else
    force_f = "OPLSAA";

  do {
    cout << "Do you want to take only minimun sc-ct distance? (Y/N=1/2)";
    if (!(cin >> flag_all_side_chain_ct)) {
      cin.clear();
      cin.ignore(10000, '\n');
    }
  } while (flag_all_side_chain_ct != 1 && flag_all_side_chain_ct != 2);
  do {
    cout << "Do you want to creat .PQR file? (Y/N=1/2)";
    cin >> flag_pqr;
  } while (flag_pqr != 1 && flag_pqr != 2);

  cout << "Do you want to use topo file (no if you don't want to use topo): ";
  cin >> topo_name;
  if (check_topo(topo_name, 1) == true)
    flag_topo_1 = true;

  cout << "Number of threads: ";
  cin >> nt;
  return 0;
}
bool arguments::check_topo(string NameofTopoFile, int Topo) {
  if (NameofTopoFile == "no")
    return false;
  else {
    auto assign_topo_name = [&](const std::string &name) {
      if (Topo == 2)
        topo_name_2 = name;
      else
        topo_name = name;
    };

    fstream input_topo_protein(NameofTopoFile, ios::in);
    bool topo_is_open{input_topo_protein.is_open()};
    input_topo_protein.close();
    if (topo_is_open == true)
      return true;
    else {
      int attempts = 0;
      constexpr int MAX_ATTEMPTS = 5;
      while (!topo_is_open) {
        cout << "Cannot open " << NameofTopoFile
             << ". Please type your lig topo file (no if you don't want to use "
                "topo): ";
        cin >> NameofTopoFile;
        cout << NameofTopoFile << "\n";
        assign_topo_name(NameofTopoFile);
        if (NameofTopoFile == "no")
          return false;
        else {
          input_topo_protein.open(NameofTopoFile, ios::in);
          if (input_topo_protein.is_open()) {
            input_topo_protein.close();
            return true;
          } else {
            input_topo_protein.close();
            ++attempts;
            if (attempts >= MAX_ATTEMPTS) {
              TitanUI::print_error("Too many invalid attempts (" +
                                   std::to_string(MAX_ATTEMPTS) +
                                   "). Exiting.");
              return false;
            }
            cout << "File not found. Attempts remaining: "
                 << (MAX_ATTEMPTS - attempts) << '\n';
            continue;
          }
        }
      }
      return false;
    }
  }
}

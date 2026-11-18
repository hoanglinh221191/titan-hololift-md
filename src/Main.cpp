//                                              VdW_elec_calculator
// V9
/*********************************************************************************************************************/
// Program calculating the Electrostatic, Van der Waals energy and side-chain
// contacts, interaction energy for every atom/residue of interacting molecule
// Also creat .PQR files for APBS
// Author: Nguyen Hoang Linh
// Modified: 19-12-2013 (dd-mm-yyyy)
// Reason: Test PPL, C++AMP, cal. time execution
// Modified: 20-12-2013
// Reason: Calculate the side-chain contacts. Incompleted.................
// Modified: 25-12-2013
// Reason: Edit number of threads of engines (2 to 4 threads). Fixed concurrency
// issues Modified: 15-1-2014 Reason: Rewrite program from class of vectors to
// vector of class, protected member and user defined  ructor, destructor and
// accessors Modified: 22->24-1-2014 Reason: Fix some functions and use prefix
// ++ instead of postfix Modified: 30-3-2014 Reason: Expand for GROMOS43A1 force
// field Modified: 25-5-2014 Reason: Fix some bugs, VdW_elec_cal no longer need
// input.txt and *_topo files to search input Modified: 29-5-2014 Reason: You
// can use input.txt or not Modified: 4-6-2014->6-6-2014 Reason: More flexible
// in using threads->optimize multithreaded programming Modified: 8-6-2014
// Reason: Use index class to increase performance
// Modified: 20-6-2014
// Reason: Calculate interaction energy per atom of ligand and per residue of
// receptor (Gromos force field), take_com is rewritten Modified: 12-7-2014
// Reason: Fix some bugs and increase performance
// Modified: 20-10-2014
// Reason: User can use .pdb files which have multi snaps, optimize performance
// Modified: 18-12-2014
// Reason: Fully support force fields gromos 43a1 and 53a6
// Modified: 2-5-2015
// Reason: User can use only 1 file .pdb for AMBER99SB force field
// Modified: 22-7-2015
// Reason: User can use only 1 file .pdb for gromos force fields
// Modifield: 8-10-2015
// Reason: User can use index file to choose what atom groups that he/she wants
// to calculate Modified: 9-10-2015 Reason: User use command arguments instead
// of interactive options Modified: 15-10-2015 Reason: Fix some bugs Modified:
// 30-11-2015 Reason: modify some classes, support more forcefields, add pbc
// solver for 1 molecule Modified: 2-3-2016 Reason: Support more force fields,
// hydrogen bond finding, support RNA and DNA for amber99sb-ildn force field,
// parameter file, com-com distance, intensive pbc solver, enhance performance
// Modified: 17-3-2016
// Reason: Fix bugs in post process coordinates
// Modified: 21-3-2016
// Reason: use unorderd map for amino acids, add some checks of arguments
// Modified: 10-4-2016
// Reason: entropy calculation, quasi harmonic with non gaussian contribution
// Modified: 22-4-2016
// Reason: entropy calculation, duan-zhang method, doi:10.1021/jacs.6b02682
// Modified: 27-4-2016
// Reason: complete new pbc solver algorithm
// Modified: 30-4-2016
// Reason: expand new pbc solver algorithm to residues of chain and molecule 2
// Modified: 29-6-2016
// Reason: support OPLSAA forcefield
// Modified: 14-7-2016
// Reason: add more arguments: -pattern-input, -topo2, enhance speed
// Modified: x-10-2016
// Reason: multi files .pdb support
// Modified: 1-1-2017
// Reason: split data struct, reduce ram consumption about 5 times, support
// folder for input natively, change tool name from elec_vdw_calculator to
// universal calculator Modified: 26-5-2017 Reason: improve performance,
// optimize cache data, improve pbc solver, reduce warnings, calculate force,
// new emap Modified: 2-6-2017 Reason: improve sefl-interaction, calculate
// energy interaction as a function of different coordinate of coms
/**********************************************************************************************************************/
#include "functions.h"
#include "cli/option_validation.h"
#ifndef TITAN_WITHOUT_HOLOLIFT
#include "hololift/temporal_runner.h"
#endif
#include "titan_ui.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

using namespace std;

// -pbctopo-temporal=hololift: once VIBE has written its package into the working directory, HoloLift-MD selects the
// relative periodic images for the whole trajectory (default settings: strict admission, evidence charge 1000) and
// writes hololift_*.csv beside the package, with the selected path as hololift_coordinates.{xtc,pdb,csv}.  The
// coordinates keep HoloLift's relative images and unwrap the common translation with the displacement-preserving
// (toroidal) scheme, so they serve transport under constant pressure directly.
static int run_engine(arguments &Arguments) {
  const int status = engine(Arguments);
  if (status != 0 || !Arguments.pbc.solv_pbctopo || Arguments.pbc.pbctopo_temporal != "hololift")
    return status;
#ifdef TITAN_WITHOUT_HOLOLIFT
  // Built with HOLOLIFT=0; arguments.cpp refuses -pbctopo-temporal=hololift.
  return 1;
#else
  titan_hololift::HoloLiftRunOptions options;
  options.system_label = "hololift";
  options.package_directory = std::filesystem::current_path();
  options.trajectory_path = Arguments.traj_file;
  options.output_directory = std::filesystem::current_path();
  const std::size_t threads =
      Arguments.nt > 0 ? Arguments.nt : std::max<std::size_t>(1, std::thread::hardware_concurrency());
  options.dp_workers = std::min<std::size_t>(threads, 64);
  options.trajectory_workers = *options.dp_workers;
  options.coordinates = titan_hololift::HoloLiftCoordinateOutput::Toroidal;
  options.pdb = titan_hololift::HoloLiftPdbOutput::First;
  options.reference_pdb = Arguments.pdb_name;
  TitanUI::print_info("HoloLift-MD: temporal selection of periodic images on the VIBE package");
  return titan_hololift::run_hololift_temporal_value(options);
#endif
}

int main(int argc, char **argv) {
  try {
    arguments Arguments{};
    if (argc != 1) {
      const int parse_status = Arguments.parse_arguments(argc, argv);
      if (parse_status == titan_cli::kHandledAndExit)
        return 0;
      if (parse_status != titan_cli::kParseSuccess)
        return parse_status;
      return run_engine(Arguments);
    } else {
      const int parse_status = Arguments.parse_arg_file();
      if (parse_status == titan_cli::kParseSuccess)
        return run_engine(Arguments);
      if (parse_status == titan_cli::kHandledAndExit)
        return 0;
      if (parse_status != titan_cli::kNoArgumentFile)
        return parse_status;
      Arguments.parse_arguments_interactive(argc);
      return run_engine(Arguments);
    }
  } catch (const std::exception &e) {
    const std::string &option = titan_cli::active_option_token();
    if (!option.empty()) {
      // A std::sto* failure while reading an option value reports only the
      // function name, which is useless on a long command line.
      std::cerr << "Exception caught in main while handling '" << option
                << "': " << e.what()
                << "\n  Check the value supplied for that option." << std::endl;
    } else {
      std::cerr << "Exception caught in main: " << e.what() << std::endl;
    }
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught in main!" << std::endl;
    return 1;
  }
  return 0;
}

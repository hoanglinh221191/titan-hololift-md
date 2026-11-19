# HoloLift

HoloLift reconstructs molecular assemblies in periodic molecular-dynamics trajectories so that they stay continuous
in time. In each frame every molecule has a copy in every cell. When the molecules touch, the contacts say which
copies belong together, but a frame does not decide this when the molecules are apart or when one touches two
copies of another. A tool that rebuilds each frame on its own then lets molecules jump from one copy of a partner to
another between frames. HoloLift chooses the copies for the whole trajectory at once: it keeps the contacts of every
frame and otherwise keeps the motion continuous. It writes the result as a trajectory.

HoloLift runs inside TITAN, after VIBE, the per-frame reconstruction of TITAN
([TITAN-VIBE](https://github.com/hoanglinh221191/titan-vibe)). One command goes from an `.xtc` file to the
reconstructed trajectory. (Up to version 1.1.0 the method was called HoloLift-MD.)

## Quick start (Windows)

1. Download `HoloLift-1.2.1-windows-x64.zip` from the
   [release page](https://github.com/hoanglinh221191/titan-hololift-md/releases) and unzip it. It needs Windows on
   x86-64 with AVX2 and FMA, and nothing else.
2. Run the example in PowerShell:

   ```powershell
   cd HoloLift-1.2.1\examples\01_separation
   ..\..\titan.exe -a arg.in
   python check_example.py
   ```

   The example is a designed trajectory of two peptides that part and meet again, with a known answer. The last
   line of the run starts with `PASS`; `check_example.py` compares the result with the design.
3. Read [docs/TUTORIAL.md](docs/TUTORIAL.md). It explains the output with this example and then shows how to run
   HoloLift on your own trajectory.

`titan.exe -hololift-help` prints a one-page reference.

## Run on your own trajectory

Write the options into a file, one per line, for example `arg.in`, and run `titan.exe -a arg.in` in the folder that
should receive the output:

```text
-s=reference.pdb
-traj=traj.xtc
-n=index.ndx
-g=1-1
-common_topo=topol.top
-ff=charmm36
-path-a=C:/HoloLift-1.2.1/forcefields
-path-n=C:/HoloLift-1.2.1/forcefields
-pbctopo=true
-pbctopo-temporal=hololift
-nocal=true
-no-dssp=true
-noprint-side_chain_contact=true
-nt=8
```

| Option | Meaning |
| --- | --- |
| `-s` | A structure in the atom order of the trajectory; its atom records name the atoms of the output PDB |
| `-traj` | The trajectory, `.xtc` |
| `-n`, `-g` | The index file, and the group of molecules to follow (`-g=1-1`: group 1, counted from 0) |
| `-common_topo` | The GROMACS topology; its bonds define the molecules |
| `-ff`, `-path-a`, `-path-n` | The force field of the topology and the folder of its tables (`forcefields/`); `-ff=martini` needs no table |
| `-pbctopo=true` | VIBE, the reconstruction of each frame |
| `-pbctopo-temporal=hololift` | HoloLift after VIBE |
| `-nocal=true`, `-no-dssp=true`, `-noprint-side_chain_contact=true` | Skip TITAN's energy and structure analyses |
| `-nt` | Threads; no thread count changes a result |
| `-b`, `-e`, `-dt` | Optional: first and last time and the stride (ps) |

The tutorial (section 7) says how to make each input with GROMACS and what to do when a run fails.

## Output

| File | Content |
| --- | --- |
| `hololift_coordinates.xtc` | The reconstructed trajectory: the selected atoms in their input order, nm |
| `hololift_coordinates.pdb` | Its first frame, with the atom records of `-s` |
| `hololift_coordinates.csv` | Per frame: the centre of the anchor molecule (the largest one) in the cell and on the written path, and the translation between them |
| `hololift_frame_results.csv` | One row per frame: VIBE's choice (`framewise_*`) and HoloLift's (`temporal_*`), with the cell shift of each molecule, the costs, the certificate counters of the transition into the frame and `selected_evidence_unsupported` |
| `hololift_segment_results.csv` | One row per stretch without a gap: objective, second-best objective and gap, number of optimal paths (capped at 2), certificate flags |
| `hololift_system_summary.csv` | The settings used, the objective, `changed_from_framewise_frames`, `evidence_unsupported_selections`, gaps and `audit_digest_sha256` |
| `hololift_policy_admissions.csv` | Candidates admitted by the `carry` or `restore` settings (empty by default) |
| `pbctopo_*.csv` | VIBE's results: per-frame reports, candidate lists, component layout and hard edges (`titan.exe -vibe-help`) |

In the written coordinates the relative copies are HoloLift's choice. The centre of the anchor molecule advances by
the nearest-image displacement between its wrapped centres in consecutive cells (Bullerjahn et al., J. Chem. Theory
Comput. 2023), so a change of box size under constant pressure does not enter the motion. A frame after a gap
restarts the anchor at its position in the cell, and gap frames are not written. Where HoloLift's choice itself
changes, one molecule moves by a cell vector in absolute coordinates; which one depends on the anchor.

The trajectory is read in blocks, never held whole: once to prepare every frame, and once after the solve to audit
the all-atom reconstruction of the selected path and write it out. Neither the block size nor the thread count
changes any output.

## What is in this repository

| Path | Content |
| --- | --- |
| `src/` | The complete TITAN source: HoloLift in `src/hololift`, VIBE in `src/pbctopo`, and TITAN's other analyses |
| `Makefile`, `VERSION`, `tools/generate_*.py` | TITAN's build |
| `examples/` | The examples; `examples/README.md` lists them |
| `forcefields/` | TITAN's force-field tables, read through `-path-a` and `-path-n` |
| `docs/TUTORIAL.md` | The tutorial |
| `docs/TITAN_MANUAL.txt` | TITAN's full manual (`titan.exe -h`) |
| `tools/hololift_temporal_value_runner.cpp` | The standalone runner (below) |
| `test/` | Two smoke tests |
| `RELEASE_NOTES_v1.2.1.md` | What changed in this version |
| `RELEASE_NOTES_v1.2.0.md` | What changed in version 1.2.0 |

The release zip holds `titan.exe`, `examples/`, `forcefields/`, `docs/`, this README, the release notes, the
licenses and `SHA256SUMS.txt`.

## Build from source

Tested on Windows 11 with MSYS2 UCRT64 (Clang 22.1.4, GNU make, Python 3). From a shell in the repository root:

```powershell
C:\msys64\ucrt64\bin\mingw32-make.exe all -j8
```

This writes `titan.exe`. The default `PROFILE=portable` needs a CPU with AVX2 and FMA; `PROFILE=native` builds for
the CPU at hand. `make all` on Linux with Clang 20 or later, or g++ 13 or later, follows the same Makefile; that route
has not been tested for this release.

## The standalone runner

The runner starts from an existing VIBE package (the `pbctopo_*.csv` files of a run) and takes HoloLift's settings
from environment variables.

```powershell
.\tools\build_hololift_temporal_value_runner.ps1 -Output build\hololift_temporal_value_runner.exe `
    -OptimizationFlags '-O3','-mavx2','-mfma','-mtune=generic'
```

Every frame is bound to the VIBE package by digests of its wrapped coordinates, compared bit for bit, so build the
runner with the compiler and flags of the `titan.exe` that wrote the package: the line above matches the portable
build and the release binary. A mismatch shows as `TRAJECTORY_ERROR wrapped XTC coordinates do not match VIBE
binding` on a triclinic cell.

```text
hololift_temporal_value_runner <system-label> <vibe-package-dir> <trajectory.xtc> <output-dir> [--transport-radius <N>]
```

| Argument | Meaning |
| --- | --- |
| `system-label` | Free text written into the `system` column of every output |
| `vibe-package-dir` | Output folder of a VIBE run: `pbctopo_frame_report.csv`, `pbctopo_assignment_candidates.csv`, `pbctopo_component_layout.csv`, `pbctopo_hard_edges.csv`; `pbctopo_rejected_candidates.csv` and `pbctopo_temporal_path.csv` are read when present |
| `trajectory.xtc` | The trajectory VIBE read |
| `output-dir` | Where the results are written |
| `--transport-radius <N>` | Optional: widen each frame's candidate list by relative-image translations up to radius `N` |

| Variable | Default | Meaning |
| --- | --- | --- |
| `HOLOLIFT_EVIDENCE_SET_WEIGHT` | `1000` | w_s, the charge on a selection without contact evidence when another candidate has it |
| `HOLOLIFT_RANK_WEIGHT` | `1` | w_r, the ordinal rank weight |
| `HOLOLIFT_TRANSITION_WEIGHT` | `1` | w_p, the transition weight |
| `HOLOLIFT_RANK_COST` | `linear` | `linear` or `saturated` rank cost |
| `HOLOLIFT_EVIDENCE_ABSENT` | `strict` | Candidate admission: `strict` (provider-certified candidates only), `carry` or `restore` |
| `HOLOLIFT_EVIDENCE_WEIGHT` | `0` | w_e, an optional charge on every candidate other than a unique supported one |
| `HOLOLIFT_DP_WORKERS` | `1` | Threads for the dynamic program (1-64) |
| `HOLOLIFT_TRAJECTORY_WORKERS` | `HOLOLIFT_DP_WORKERS`, else all hardware threads | Threads for binding, preparation, audit and coordinates (1-64) |
| `HOLOLIFT_TRAJECTORY_BLOCK` | automatic | Frames per trajectory block |
| `HOLOLIFT_COORDINATES` | `none` | `lattice` or `toroidal`: also write `hololift_coordinates.{xtc,pdb,csv}` (`titan.exe` writes `toroidal`) |
| `HOLOLIFT_PDB` | `first` when coordinates are written | `none`, `first` or `all` frames as PDB models |
| `HOLOLIFT_REFERENCE_PDB` | none | Structure whose atom records name the PDB atoms |
| `HOLOLIFT_TIMING` | unset | Any value: wall time of each stage on standard error |

`titan.exe` runs HoloLift with these defaults. The runner prints one summary line and exits with 0 on success; any
other exit code comes with an error class (`TRAJECTORY_ERROR`, `PREPARATION_ERROR`, `SOLVE_ERROR`, ...) on
standard error.

## Tests

```powershell
.\test\hololift_phase0_smoke.ps1 -Compiler clang++
.\test\hololift_phase1_smoke.ps1 -Compiler clang++
```

Each script compiles a self-contained test against the sources and runs it; a nonzero exit code is a failure.

## Citation

If you use HoloLift, please cite the paper (see `CITATION.cff`):

Na O My, Thi Chinh Ngo, Tam V.-T. Mai, Quang Linh Huynh, Thi Minh Thu Tran, Thi Thu Ha Pham, Mai Suan Li and
Hoang Linh Nguyen. *HoloLift: Temporal Selection of Periodic Images for Molecular Assemblies in Simulation
Trajectories.* Submitted.

## License

MIT (see `LICENSE`). The XTC reader in `src/gmxtraj`, the Eigen library in `src/eigen`, the GBn2 tables in
`src/titan_gbn2_tables.h` and the force-field tables in `forcefields/` have their own terms; see
`THIRD_PARTY_NOTICES.md`. `SOURCE_PROVENANCE.json` names the TITAN commit this source tree was exported from.

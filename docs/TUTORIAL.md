# HoloLift tutorial

This tutorial takes about fifteen minutes. You run HoloLift on a small designed trajectory whose right answer is
known, look at what it writes, and then set up a run on your own trajectory.

## 1. What HoloLift is for

In a simulation with periodic boundary conditions every molecule has a copy in every cell. To analyse an assembly,
for example the distance between two chains, their relative orientation, or how the whole assembly moves, you need
one choice of copies in every frame.

Usually the contacts decide. If chain B touches chain A, the copy of B that touches A is the right one. There are two
situations where one frame does not decide:

- **The molecules are apart.** No copy of B touches A, so no copy is singled out.
- **B touches two copies of A at once.** Two placements are equally supported.

A tool that rebuilds each frame on its own still writes a correct frame: every molecule is whole and every contact
is kept. But in these frames its choice can jump from one copy of A to another between consecutive frames, although
nothing like that happened in the simulation. HoloLift makes the choice for all frames together. It keeps the
contacts that each frame shows, and otherwise it keeps the molecules moving continuously.

HoloLift runs inside TITAN, after VIBE. VIBE rebuilds each frame and lists the placements that frame allows,
ranked; its first choice is the per-frame answer. HoloLift reads these lists and selects one placement per frame.

## 2. Get HoloLift

**Windows (x86-64, a processor with AVX2 and FMA: Intel from 2013, AMD from 2015).** Download
`HoloLift-1.2.1-windows-x64.zip` from the release page and unzip it. It contains `titan.exe`, the `forcefields`
tables, the `examples` and these documents. Nothing needs to be installed.

**Other systems.** Build `titan.exe` from the source, as described in `README.md`.

Check that it runs. In PowerShell, in the unzipped folder:

```powershell
.\titan.exe -v
.\titan.exe -hololift-help
```

`-hololift-help` prints a one-page reference. `titan.exe -h` prints the full TITAN manual, which you do not need for
HoloLift.

Python 3 is optional; it runs the checking script of the example. With MDAnalysis installed
(`pip install MDAnalysis`), the script also checks the written coordinates.

## 3. The example

`examples/01_separation` holds a trajectory of two amyloid-beta(1-42) peptides, A and B, with water removed. The
starting structure is a frame of the peptide-pair example of TITAN-VIBE. The motion is designed: both peptides move
as rigid bodies, so the right answer is known in every frame. `make_frames.py` builds the frames and documents the
design.

The cell is a rhombic dodecahedron with 110 Å between neighbouring copies. There are 250 frames, 100 ps apart:

| Frames | What B does |
| --- | --- |
| 0-39 | B is bound to A |
| 40-69 | B leaves A, towards the point halfway between A and one neighbouring copy of A |
| 70-179 | B drifts back and forth across that halfway point, at least 18 Å from every copy of A |
| 180-209 | B comes back to the copy of A it left |
| 210-249 | B is bound to A again |

The pair also drifts through the cell, so the peptides cross the cell faces. As in a trajectory written by an MD
engine, every atom is wrapped into the cell, so the peptides are often cut into pieces.

B never changes partner. The right reconstruction keeps B next to the same copy of A in all 250 frames. But while B
is past the halfway point, it is nearer to the other copy of A. A per-frame choice follows it there and back, which
puts B on the other side of A for a few frames at a time. `truth.csv` lists the designed answer frame by frame,
including which copy of A is nearer.

## 4. Run it

```powershell
cd examples\01_separation
..\..\titan.exe -a arg.in
```

On Linux or macOS, use `../../titan` with your own build. The run takes about 20 s on a laptop, with 4 threads or
with 1. The screen shows TITAN's progress. The last line HoloLift prints is

```text
PASS system=hololift frames=250 policy=strict carried=0 evidence_unsupported=0 candidates=31250 transport_radius=0
synthetic_rank_inversion_v1=0 synthetic_domain_miss_v1=0 synthetic_domain_miss_v2=0 changed=60 gaps=0
elapsed_ms=249.5 audit=71544ea777fc2d9d482ccf2ea32e686ad8e8439967dcaf514b969c1ebee9b5d1 coordinates=toroidal
written=250
```

(one line on the screen; `elapsed_ms` varies from run to run).

`arg.in` holds the options, one per line:

| Option | Meaning |
| --- | --- |
| `-ff=charmm36` | The force field of the topology. VIBE takes the atom radii from it. |
| `-path-a=../../forcefields`, `-path-n=../../forcefields` | Where the force-field tables are |
| `-s=reference.pdb` | A structure in the atom order of the trajectory. Its atom names are used in the output PDB. |
| `-traj=frames.xtc` | The trajectory |
| `-n=index.ndx`, `-g=0-0` | The index file, and the group to follow: group 0, both peptides |
| `-common_topo=topol.top` | The GROMACS topology. Its bonds define the molecules. |
| `-pbctopo=true` | Run VIBE, the reconstruction of each frame |
| `-pbctopo-temporal=hololift` | Run HoloLift after VIBE |
| `-nocal=true`, `-no-dssp=true`, `-noprint-side_chain_contact=true` | Skip TITAN's energy and structure analyses |
| `-opdb=placed.pdb` | Also write VIBE's per-frame result, one model per frame, to compare with |
| `-nt=4` | Threads. The thread count never changes a result. |

## 5. Read the result

### The PASS line

| Field | Meaning | Here |
| --- | --- | --- |
| `frames` | Frames read | 250 |
| `candidates` | Candidate placements VIBE listed, over all frames: 125 per frame here | 31,250 |
| `changed` | Frames where HoloLift chose differently from VIBE's first choice for that frame | 60 |
| `evidence_unsupported` | Frames where HoloLift kept a placement without contact evidence although another placement had it. 0 means that every frame keeps the contacts it shows. | 0 |
| `gaps` | Frames VIBE could not rebuild. HoloLift solves the path separately on each side of a gap. | 0 |
| `audit` | A checksum of the result: the same input gives the same value, with any thread count | |
| `written` | Frames written to `hololift_coordinates.xtc` | 250 |

`policy`, `carried`, `transport_radius` and `synthetic_*` are settings of the standalone runner; `titan.exe`
always runs with `policy=strict` and the others at 0.

Any other ending, or an exit code other than 0, is an error. Its class (`TRAJECTORY_ERROR`, `SOLVE_ERROR`, ...) is
printed on standard error.

### Compare with the designed answer

```powershell
python check_example.py
```

prints

```text
frames                            250 (designed: 250)
frames VIBE could not reconstruct 0
HoloLift selections without contact evidence although another candidate had it: 0
B nearer to the other copy of A   62 frames: 70-79, 89-96, 106-115, 125-133, 142-151, 160-168, 179-184   (truth.csv)
HoloLift differs from VIBE        60 frames: 70-79, 89-96, 106-114, 125-132, 142-151, 160-168, 179-184
HoloLift coordinates: B relative to A as designed in 250 of 250 frames (largest deviation 0.002 A)
VIBE per-frame output (placed.pdb): B next to another copy of A in 60 frames: 70-79, 89-96, 106-114, 125-132, 142-151, 160-168, 179-184
PASS
```

(the last two lines need MDAnalysis). Read it from the bottom up:

- HoloLift's trajectory has B where the design put it in all 250 frames; the deviation of 0.002 Å is the rounding of
  the XTC format.
- VIBE's per-frame output puts B next to the other copy of A in 60 frames, in seven stretches: each time B drifts
  past the halfway point, and back when it returns.
- `truth.csv` says that B is nearer to the other copy of A in 62 frames. In frames 115 and 133 the two copies are
  less than 0.3 Å from being equally near, and VIBE, which ranks the copies by its own score rather than by the
  smallest atom distance, still puts its partner first.
- HoloLift differs from VIBE in exactly the 60 frames where VIBE moved B.

### The frame table

`hololift_frame_results.csv` has one row per frame. The columns you need first:

| Column | Meaning |
| --- | --- |
| `frame_index`, `time_ps` | The frame |
| `framewise_component_images` | VIBE's first choice: the cell shift `a:b:c` applied to each molecule, molecules separated by `|` |
| `temporal_component_images` | HoloLift's choice, in the same form |
| `changed_from_framewise` | 1 where the two differ |
| `framewise_candidate_rank`, `temporal_candidate_rank` | The rank VIBE gave to each choice (1 = its first choice) |
| `transition_cost` | HoloLift's cost of the move from the previous frame |
| `selected_evidence_unsupported` | 1 if HoloLift's choice gives up a contact that another placement keeps |

Only the difference between the shifts of two molecules matters; a shift applied to all molecules together moves
the whole assembly by one cell. Around the first stretch where the two differ, the table reads (some columns):

```text
frame_index,time_ps,framewise_component_images,temporal_component_images,changed_from_framewise,framewise_candidate_rank,temporal_candidate_rank,transition_cost
69,6900,0:0:0|0:0:0,0:0:0|0:0:0,0,1,1,0.0229
70,7000,0:0:0|0:1:-1,0:0:0|0:0:0,1,1,2,0.0094
71,7100,0:0:0|0:1:-1,0:0:0|0:0:0,1,1,2,0.0403
79,7900,0:0:0|0:1:-1,0:0:0|0:0:0,1,1,2,0.0386
80,8000,0:0:0|0:0:0,0:0:0|0:0:0,0,1,1,0.0438
```

At frame 70 VIBE's first choice shifts B by b - c, one cell vector, to the other copy of A; HoloLift takes VIBE's
second candidate, which keeps B where it was, at a small transition cost. At frame 80 B is back on the partner's side
of the halfway point and both agree again.

VIBE's own report, `pbctopo_frame_report.csv`, labels the frames. While B is bound, `evidence_state` is
`supported_connected` (`status` `certified`): the contacts fix the placement. While B is apart, from frame 55 in
the leaving stretch to frame 193 in the returning one, it is `weak_disconnected` (`status` `weak_observation`): the
frame does not fix where B sits relative to A. These are the frames where HoloLift's choice matters.
`titan.exe -vibe-help` explains all labels.

### All output files

| File | Content |
| --- | --- |
| `hololift_coordinates.xtc` | The reconstructed trajectory of the group, in nm |
| `hololift_coordinates.pdb` | Its first frame, with the atom names of `-s` |
| `hololift_coordinates.csv` | Per frame: where the largest molecule's centre is in the cell and on the written path |
| `hololift_frame_results.csv` | One row per frame (above) |
| `hololift_segment_results.csv` | One row per stretch of frames without a gap: the total cost and how far the second-best path is from it |
| `hololift_system_summary.csv` | The totals of the run and the settings used |
| `hololift_policy_admissions.csv` | Placements added by the `carry` or `restore` settings of the standalone runner; empty by default |
| `pbctopo_*.csv` | VIBE's results, which HoloLift read |
| `placed.pdb` | VIBE's per-frame result (from `-opdb`) |

## 6. Look at the trajectories

Load the reconstructed trajectory as its first frame plus the XTC:

- **VMD:** `vmd hololift_coordinates.pdb hololift_coordinates.xtc`
- **PyMOL:** `load hololift_coordinates.pdb, hl` and then `load_traj hololift_coordinates.xtc, hl`
- **ChimeraX:** `open hololift_coordinates.pdb` and then `open hololift_coordinates.xtc structureModel #1`

Then open `placed.pdb` (VIBE, one model per frame) and `frames.xtc` with `reference.pdb` (the raw input) beside it.
Play frames 60-100:

- In `frames.xtc` the peptides are cut by the cell faces.
- In `placed.pdb` both peptides are whole in every frame, but at frame 70 B jumps to the other side of A, and at
  frame 80 it jumps back.
- In `hololift_coordinates.xtc` B drifts away from A and comes back, and the pair moves smoothly across the cell
  faces.

The written coordinates do not wrap the assembly back into the cell. They follow its motion, also when the box
changes size under constant pressure. Where HoloLift's choice itself changes, in a real trajectory, one molecule
moves by a cell vector, as it must.

## 7. Your own trajectory

### What you need

| Input | How to make it with GROMACS |
| --- | --- |
| The trajectory, `.xtc` | Your MD output. Frames must be close enough in time that no molecule moves more than a small part of the cell between two frames, because HoloLift compares consecutive frames. |
| A structure in the same atom order, `.pdb` | `gmx trjconv -s topol.tpr -f traj.xtc -o reference.pdb -dump 0`, group System |
| The topology, `.top` with its `.itp` files | The topology of the simulation. TITAN reads its `[ molecules ]` and bonds. |
| An index file with the molecules to follow | `gmx make_ndx -f topol.tpr -o index.ndx`, then make a group with exactly those molecules, for example the protein |

If the trajectory holds the solvent, leave it there; the index group selects the molecules. Or, to read less, write
a trajectory of the molecules alone (`gmx trjconv -n index.ndx`) with a structure and a topology of the same
molecules, as in the example.

### arg.in

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
-b=0
-e=100000
-dt=100
```

- `-g=1-1` follows group 1 of the index file; groups are counted from 0, in the order of the file.
- `-ff` names a table in `forcefields/` (`charmm36`, `amber99sb-ildn`, `amber14sb`, `gromos54a7`, `oplsaa`, ...;
  case does not matter). A MARTINI system uses `-ff=martini`, which needs no table.
- `-b`, `-e` and `-dt` choose the first and last time and the stride, in ps. Leave them out to use every frame.
- Run `titan.exe -a arg.in` in an empty folder; the outputs are written there.

Start with a short window (`-e`) to see how long a frame takes. Most of the time goes into VIBE's reconstruction of
each frame, and it grows with the number of molecules.

### When something goes wrong

| Symptom | What to do |
| --- | --- |
| `Cannot open CHARMM36_aminoacids.rtp` | `-path-a` and `-path-n` must point at the `forcefields` folder |
| TITAN stops while reading the inputs | The structure, the trajectory and the topology must hold the same atoms in the same order |
| `gaps` > 0 | VIBE could not rebuild these frames; `pbctopo_frame_report.csv` gives the reason (`status`, `fallback_reason`) |
| `evidence_unsupported` > 0 | In these frames HoloLift gave up a contact to keep the motion continuous; look at them (`selected_evidence_unsupported` = 1) |
| A run ends without `PASS` | The error class on standard error names the stage: 2 inputs or settings, 3 trajectory, 4-5 preparation, 6 solve, 7-9 audit, 10 output, 14 coordinates |

## 8. How HoloLift chooses

For every frame VIBE supplies a short list of candidate placements, ranked, each marked as supported by a contact
or not. HoloLift gives every path through these lists a cost:

- the rank of each chosen candidate (weight 1);
- the squared displacement of the molecules from one frame to the next, after removing any common translation of
  the whole assembly by a cell vector (weight 1);
- a large charge (1000) on a chosen candidate without contact evidence when another candidate of that frame has it.

It then finds the cheapest path by dynamic programming, which is exact on the supplied lists. The paper gives the
definitions and the validation. `titan.exe` uses these default settings; the standalone runner in `tools/` takes
other values (see `README.md`).

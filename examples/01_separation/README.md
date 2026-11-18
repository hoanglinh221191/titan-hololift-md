# Example 01: two peptides that part and meet again

Two amyloid-beta(1-42) peptides, A and B (1,254 atoms, CHARMM36m, water removed), in a rhombic dodecahedron with
110 Å between neighbouring copies; 250 frames, 100 ps apart. B leaves A, drifts back and forth across the point
halfway between A and one neighbouring copy of A, and returns to the copy it left. Both peptides move as rigid
bodies, so the right answer is known in every frame: B keeps its partner.

```powershell
..\..\titan.exe -a arg.in
python check_example.py
```

| File | Content |
| --- | --- |
| `arg.in` | The options of the run |
| `frames.xtc` | The trajectory, every atom wrapped into the cell |
| `reference.pdb` | Frame 0 with both peptides whole, in the atom order of the trajectory (chain IDs A and B) |
| `index.ndx` | Group 0: both peptides |
| `topol.top`, `topol_Protein_chain_A.itp`, `topol_Protein_chain_A2.itp` | The topology |
| `peptides.pdb` | The starting structure: frame 26 of the peptide-pair example of TITAN-VIBE 1.1.0, both chains whole |
| `make_frames.py` | Builds `frames.xtc`, `reference.pdb`, `index.ndx` and `truth.csv` from `peptides.pdb` (numpy and MDAnalysis; fixed seed) |
| `truth.csv` | The design, frame by frame: B's centre relative to A's, its distance to its partner and to the other copies of A, and which copy is nearer |
| `check_example.py` | Compares the result with `truth.csv` and `EXPECTED.json` (Python 3; MDAnalysis for the coordinate check) |
| `EXPECTED.json` | The result of this release: the frames where HoloLift differs from VIBE, and the PASS line |

[docs/TUTORIAL.md](../../docs/TUTORIAL.md) walks through the run and its output.

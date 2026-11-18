# Examples

| Folder | What it shows |
| --- | --- |
| `01_separation` | Two peptides that part, drift around the point halfway between two copies of each other, and meet again. The answer is known: B keeps its partner. VIBE's per-frame choice follows B to the nearer copy of A; HoloLift keeps the partner in every frame. |

Run an example from its folder, so that its outputs are written there:

```powershell
cd examples\01_separation
..\..\titan.exe -a arg.in
python check_example.py
```

[docs/TUTORIAL.md](../docs/TUTORIAL.md) walks through the example and its output.

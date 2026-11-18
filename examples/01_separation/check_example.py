#!/usr/bin/env python3
"""Compare the result of example 01 with its designed answer.

    python check_example.py

Run it in this folder after  ..\\..\\titan.exe -a arg.in  (Windows) or  ../../titan.exe -a arg.in.
It reads hololift_system_summary.csv, hololift_frame_results.csv, truth.csv and EXPECTED.json and needs only
Python 3.  If MDAnalysis is installed it also measures, frame by frame, where B sits relative to A in the
reconstructed trajectory (hololift_coordinates.xtc) and in VIBE's per-frame output (placed.pdb).
It prints what it found and ends with PASS or FAIL (exit code 0 or 1).
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    with open(HERE / name, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def runs(flags):
    """Frame ranges where flags is true, as text: 70-79, 89-96, ..."""
    out, start = [], None
    for t, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            out.append(f'{start}-{t - 1}' if t - 1 > start else f'{start}')
            start = None
    return ', '.join(out) if out else 'none'


def main():
    problems = []
    for name in ('hololift_system_summary.csv', 'hololift_frame_results.csv', 'truth.csv', 'EXPECTED.json'):
        if not (HERE / name).is_file():
            print(f'{name} not found: run  ..\\..\\titan.exe -a arg.in  in this folder first')
            return 1
    summary = rows('hololift_system_summary.csv')[0]
    frames = rows('hololift_frame_results.csv')
    truth = rows('truth.csv')
    expected = json.loads((HERE / 'EXPECTED.json').read_text())

    n = len(frames)
    changed = [r['changed_from_framewise'] == '1' for r in frames]
    other = [r['nearer_copy'] == 'other' for r in truth]
    print(f'frames                            {n} (designed: {len(truth)})')
    print(f'frames VIBE could not reconstruct {summary["observation_gap_frames"]}')
    print(f'HoloLift selections without contact evidence although another candidate had it: '
          f'{summary["evidence_unsupported_selections"]}')
    print(f'B nearer to the other copy of A   {sum(other)} frames: {runs(other)}   (truth.csv)')
    print(f'HoloLift differs from VIBE        {sum(changed)} frames: {runs(changed)}')
    if n != len(truth):
        problems.append('frame count')
    if summary['observation_gap_frames'] != '0' or summary['evidence_unsupported_selections'] != '0':
        problems.append('gaps or unsupported selections')
    if [t for t, c in enumerate(changed) if c] != expected['changed_frames']:
        problems.append('the frames that differ from VIBE are not those of EXPECTED.json')

    try:
        import numpy as np
        import MDAnalysis as mda
    except ImportError:
        print('MDAnalysis is not installed: the coordinate check is skipped')
    else:
        import warnings
        warnings.filterwarnings('ignore')
        design = np.array([[float(r[f'b_minus_a_{c}_A']) for c in 'xyz'] for r in truth])
        chain = np.array([line[21] for line in (HERE / 'reference.pdb').read_text().splitlines()
                          if line.startswith(('ATOM', 'HETATM'))])
        ia, ib = np.flatnonzero(chain == 'A'), np.flatnonzero(chain == 'B')

        def offsets(universe):
            return np.array([universe.atoms.positions[ib].mean(axis=0) - universe.atoms.positions[ia].mean(axis=0)
                             for _ in universe.trajectory])

        hololift = offsets(mda.Universe(str(HERE / 'hololift_coordinates.pdb'), str(HERE / 'hololift_coordinates.xtc')))
        error = np.linalg.norm(hololift - design, axis=1)
        print(f'HoloLift coordinates: B relative to A as designed in {int((error < 0.05).sum())} of {n} frames '
              f'(largest deviation {error.max():.3f} A)')
        if (error >= 0.05).any():
            problems.append('HoloLift coordinates differ from the design')
        if (HERE / 'placed.pdb').is_file():
            vibe = offsets(mda.Universe(str(HERE / 'placed.pdb')))
            jump = np.linalg.norm(vibe - design, axis=1) > 10.0
            print(f'VIBE per-frame output (placed.pdb): B next to another copy of A in {int(jump.sum())} frames: '
                  f'{runs(list(jump))}')

    print('PASS' if not problems else 'FAIL: ' + '; '.join(problems))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""
daw.py  --  Grid sheet -> instrumental. Each row is a track: its (repeated) handwritten
label is a text query into OpenCrate, and its filled columns are the trigger steps.

    python daw.py grid.json --out song.wav        # from a v2 transcription
    python daw.py photo.jpg --out song.wav        # runs v2 extraction first (loads Qwen)

Per row:
  label   = the majority handwritten text in that row  (assumed the same across the row)
  steps   = which columns are filled  (the trigger positions)
  sample  = OpenCrate search(label) -> softmax-pick one match (higher score = likelier)
Then the row's sample is placed at each trigger step, all rows are mixed, and the master
is peak-normalised so stacking tracks never clips. Tempo fixed at 90 BPM (16th-note grid).
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np

import opencrate
import sequencer as seq

V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2")


def load_grid(source):
    """Return (rows, cols, grid). `source` is a v2 JSON transcription or an image."""
    if source.lower().endswith(".json"):
        d = json.load(open(source))
        return d["rows"], d["cols"], d["grid"]
    # image -> run the v2 vision pipeline (loads Qwen2.5-VL)
    sys.path.insert(0, V2)
    import extract_grid as eg
    import ocr

    rows, cols, flat = eg.rectify(source)
    cells, _ = eg.slice_cells(flat, rows, cols, -1)
    to_read = [None if ocr.is_blank(c) else c for c in cells]
    print(
        f"reading {sum(c is not None for c in to_read)} filled cells with Qwen2.5-VL..."
    )
    text = ocr.recognize(to_read)
    grid = [[text[r * cols + c] for c in range(cols)] for r in range(rows)]
    return rows, cols, grid


def row_label_and_steps(cells):
    """(majority text, [trigger columns]) for a row, or (None, []) if empty."""
    filled = [(c, t.strip()) for c, t in enumerate(cells) if t.strip()]
    if not filled:
        return None, []
    label = Counter(t for _, t in filled).most_common(1)[0][0]
    return label, [c for c, _ in filled]


def build(
    source, out, bpm, loops, steps_per_beat, k, temperature, seed, server, stems_dir
):
    rows, cols, grid = load_grid(source)
    print(f"grid {rows}x{cols} @ {bpm} BPM, {loops} loop(s)\n")

    step = seq.step_duration(bpm, steps_per_beat)
    total = loops * cols * step
    rng = np.random.default_rng(seed)

    stems, tracks = [], []
    for r in range(rows):
        label, cols_hit = row_label_and_steps(grid[r])
        if not label:
            continue
        label = f"{label} drum one-shot"  # for now - improves results
        hits = opencrate.search(label, k=k, server=server)
        hit, prob = opencrate.pick(hits, temperature=temperature, rng=rng)
        if hit is None:
            print(f"  row {r}: '{label}' -> no matches, skipped")
            continue
        try:
            sample = seq.load_sample(hit["path"])
        except Exception as e:
            print(f"  row {r}: '{label}' -> failed to load {hit['path']}: {e}")
            continue
        onsets = [(L * cols + c) * step for L in range(loops) for c in cols_hit]
        stem = seq.build_stem(sample, onsets, total)
        stems.append(stem)
        tracks.append((r, label, hit, prob, cols_hit))
        print(
            f"  row {r}: '{label}'  ->  {os.path.basename(hit['path'])}  "
            f"(score {hit['score']:.3f}, p {prob:.0%})  steps {cols_hit}"
        )
        if stems_dir:
            os.makedirs(stems_dir, exist_ok=True)
            # normalise the exported stem so it's individually listenable (the mix still
            # uses the true-level stem); mix([stem]) peak-normalises a single track
            seq.write_wav(
                os.path.join(stems_dir, f"row{r}_{label.replace(' ', '_')}.wav"),
                seq.mix([stem]),
            )

    if not stems:
        raise SystemExit("no playable rows found")
    master = seq.mix(stems)
    seq.write_wav(out, master)
    print(f"\nwrote {out}  ({len(stems)} tracks, {len(master)/seq.SR:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="v2 JSON transcription or a photo of the sheet")
    p.add_argument("--out", default="song.wav")
    p.add_argument("--bpm", type=float, default=90.0)
    p.add_argument(
        "--loops", type=int, default=4, help="how many times to repeat the pattern"
    )
    p.add_argument(
        "--steps-per-beat", type=int, default=4, help="4 = columns are 16th notes"
    )
    p.add_argument("--k", type=int, default=8, help="candidates fetched per row")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="pick softness (smaller = greedier)",
    )
    p.add_argument("--seed", type=int, default=None, help="reproducible sample picks")
    p.add_argument("--server", default=opencrate.DEFAULT_SERVER)
    p.add_argument("--stems", default=None, help="also write per-row stems to this dir")
    a = p.parse_args()
    build(
        a.source,
        a.out,
        a.bpm,
        a.loops,
        a.steps_per_beat,
        a.k,
        a.temperature,
        a.seed,
        a.server,
        a.stems,
    )


if __name__ == "__main__":
    main()

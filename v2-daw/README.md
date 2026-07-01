# paperton v2-daw

Turn a photographed grid sheet into an instrumental. It's a step sequencer: each **row**
is a track, each **column** is a 16th-note step.

```
sheet photo ─► v2 extraction ─► grid of text
            ─► per row: label = the row's handwritten word, steps = the filled columns
            ─► OpenCrate search(label) ─► softmax-pick a sample (higher score = likelier)
            ─► place the sample at each step ─► build stems ─► mix (volume-safe) ─► WAV
```

## How a row becomes a track

Each row is assumed to hold **one** repeated word (e.g. `ride`, `vinyl clap`, `long 808`).

- **label** — the majority text in the row — is sent to OpenCrate as a text query.
- **steps** — the columns that are filled — are where that sample triggers.
- The sample is chosen **probabilistically**: OpenCrate's scores are turned into a
  softmax distribution, so the best match is most likely but not guaranteed — different
  runs give different (still on-theme) sounds. Fix it with `--seed`.

## Volume / mixing (why it doesn't blow out)

Audio clips the moment a sample exceeds ±1.0. To stack tracks safely:

1. each sample is **peak-normalised** on load (tracks sit at comparable levels);
2. a track is built by **adding** its sample at each step (overlapping tails sum — correct);
3. all tracks are summed, then the master is **peak-normalised to −1 dBFS** — one gain
   that scales the loudest peak just under full-scale, so nothing ever clips while every
   track keeps its relative level.

Stems are built first, then mixed, so you can export/inspect each track (`--stems DIR`).

## Requires

- The **OpenCrate** API running with an index built (`uvicorn server:app --port 8000` in
  the opencrate2 repo). `daw.py` calls its `/search` endpoint.
- `pip install requests librosa soundfile numpy` (audio) — plus the v2 deps if you feed a
  photo (Qwen2.5-VL) rather than a transcription JSON.

## Use

```bash
# from a v2 transcription (fast; make it with: v2/extract_grid.py photo.jpg --json grid.json)
python daw.py grid.json --out song.wav --seed 0 --stems stems

# or straight from a photo (runs v2 extraction first — loads Qwen2.5-VL)
python daw.py ../input/IMG_8468.JPG --out song.wav
```

Tempo is fixed at **90 BPM**. Useful flags: `--loops N` (repeat the pattern, default 4),
`--k` (candidates per row), `--temperature` (pick softness), `--server URL`.

## Files

- `daw.py` — orchestrator: transcription → per-row query/pick → stems → mix → WAV.
- `sequencer.py` — audio engine: sample loading, stem building, timing, volume-safe mix.
- `opencrate.py` — `/search` client + softmax probabilistic pick.

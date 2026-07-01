# paperton v2

Generate a printable grid sheet, fill cells in by hand, photograph it, and get back a
transcription of every cell — including **multi-word** ("vinyl clap", "long 808") and
**multi-line** cells, which v1 could not read.

The split:

- **Geometry (carried from v1, it works):** four corner ArUco markers self-encode the
  grid's `rows×cols`; a homography flattens the photo at any angle/zoom; cell boundaries
  are snapped onto the actual printed lines (pixel-perfect, robust to paper curl).
- **Recognition (new):** each non-blank cell is read by **Qwen2.5-VL-3B-Instruct**, a
  local vision-language model. A VLM reads multi-line / multi-word handwriting natively,
  where v1's line-level TrOCR failed.

```
photo ─► detect markers ─► decode rows×cols ─► homography ─► snap to printed lines
      ─► slice cells ─► is_blank? skip : Qwen2.5-VL ─► text ─► CSV / JSON
```

## Install

```
pip install -r requirements.txt
```

First run downloads the Qwen2.5-VL-3B weights (~8 GB) to the Hugging Face cache. It runs
in bf16 on Apple MPS / CUDA, falling back to fp16 then CPU.

## Use

Generate a sheet (always US-letter landscape, square cells, no in-cell labels):

```
python make_grid.py --rows 8 --cols 16        # -> grid.png (print it)
```

Transcribe a photo of the filled-in sheet (blind — no grid size needed):

```
python extract_grid.py photo.jpg --csv out.csv --json out.json
   detected: rows=8 cols=16
   reading 11/128 filled cells with Qwen2.5-VL (~4s each)...
   wrote out.csv (8x16, 11 filled)
```

Useful flags: `--save-cells DIR` dumps the rectified sheet and every cell crop;
`--no-snap` disables grid-line snapping; `--inset N` overrides the per-edge trim.

## Files

- `gridlib.py` — geometry + self-describing marker encoding (shared by generator/extractor).
- `make_grid.py` — printable sheet generator.
- `extract_grid.py` — detect → rectify → snap → slice → recognize → CSV/JSON.
- `ocr.py` — blank detection + Qwen2.5-VL per-cell recognizer.

## Notes

- Per-cell calls are ~4s each on an M2; blank cells are skipped, so sparse sheets cost
  only a handful of calls. Dense sheets (every cell filled) are correspondingly slower.
- Multi-line cells are returned as a single space-joined string (e.g. `vinyl clap`).

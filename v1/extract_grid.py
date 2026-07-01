#!/usr/bin/env python
"""
extract_grid.py  --  Read a photo/scan of a grid sheet and OCR every cell.

Self-describing AND scale-free: the corner markers encode rows/cols, and the grid
is reconstructed RELATIVE TO THE MARKERS (the quad of their inner corners), so you
never pass the grid size or any geometry, and it works at any print zoom.

    python extract_grid.py photo.jpg                 # blind -> prints text grid + cells.csv
    python extract_grid.py photo.jpg --csv out.csv
    python extract_grid.py photo.jpg --save-cells cells --debug

Pipeline: detect 4 markers -> decode rows/cols from their IDs -> map the markers'
inner corners to a canonical rectangle (homography undoes perspective + scale) ->
warp -> slice into an even rows x cols grid -> OCR each non-blank cell (TrOCR).
"""
import argparse, csv, os
import numpy as np
import cv2
import gridlib as gl
import ocr


def rectify(photo_path):
    """Detect markers, decode size, return (rows, cols, warped_image)."""
    img = cv2.imread(photo_path)
    if img is None:
        raise SystemExit(f"could not read {photo_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dic = cv2.aruco.getPredefinedDictionary(gl.ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise SystemExit("no markers detected")
    found = {int(i): corners[k][0] for k, i in enumerate(ids.flatten())}

    decoded = gl.decode_ids(found.keys())
    if decoded is None:
        raise SystemExit(f"could not decode a valid grid from markers {sorted(found)}")
    rows, cols = decoded
    role_to_id = gl.encode_ids(rows, cols)
    missing = [r for r in (0, 1, 2, 3) if role_to_id[r] not in found]
    if missing:
        raise SystemExit(f"missing corner markers for roles {missing}")

    C = gl.canonical(rows, cols)
    src = np.array(
        [found[role_to_id[r]][gl.INNER[r]] for r in (0, 1, 2, 3)], np.float32
    )
    dst = np.array([C["dst_inner"][r] for r in (0, 1, 2, 3)], np.float32)
    flat = cv2.warpPerspective(
        img, cv2.getPerspectiveTransform(src, dst), (C["W"], C["H"])
    )
    return rows, cols, flat


def _snap(profile, expected, win):
    """Move each expected boundary to the darkest column/row within +-win, but only
    if there's a real line there (clear peak); otherwise keep the even position."""
    med = np.median(profile)
    mad = np.median(np.abs(profile - med)) + 1e-6
    out = []
    for e in expected:
        a, b = max(0, e - win), min(len(profile) - 1, e + win)
        j = a + int(np.argmax(profile[a : b + 1]))
        out.append(j if profile[j] - med > 4 * mad else e)
    for i in range(1, len(out)):  # keep strictly increasing
        out[i] = max(out[i], out[i - 1] + 1)
    return out


def snap_grid(flat, xs, ys):
    """Snap even cell boundaries onto the actual printed grid lines, correcting the
    residual drift left by paper curl / lens distortion after the flat-plane warp."""
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    dark = 255.0 - cv2.divide(g, bg, scale=255)  # ink, with lighting flattened
    win = max(4, int(0.18 * gl.CELL))
    return _snap(dark.sum(0), xs, win), _snap(dark.sum(1), ys, win)


def slice_cells(flat, rows, cols, inset, snap=True):
    """Return a list of cell crops (row-major), with the inset trimmed off."""
    C = gl.canonical(rows, cols)
    xs, ys = C["xs"], C["ys"]
    if snap:
        xs, ys = snap_grid(flat, xs, ys)
    if inset < 0:
        # just enough to clear the printed grid line; more than this clips ink that
        # overflows toward the cell edges (handwriting is often larger than the box)
        inset = max(1, int(0.04 * gl.CELL))
    inset = max(0, min(inset, (gl.CELL - 2) // 2))
    cells = []
    for r in range(rows):
        for c in range(cols):
            cells.append(
                flat[
                    ys[r] + inset : ys[r + 1] - inset, xs[c] + inset : xs[c + 1] - inset
                ]
            )
    return cells, inset


def print_grid(text, rows, cols):
    """Pretty-print the OCR'd grid as an aligned table."""
    w = max(3, max((len(t) for t in text), default=1))
    for r in range(rows):
        line = " | ".join(text[r * cols + c].center(w) for c in range(cols))
        print(f"  {line}")


def extract(photo_path, csv_path, inset, save_cells, debug, batch, snap, model_name):
    rows, cols, flat = rectify(photo_path)
    print(f"detected: rows={rows} cols={cols}")

    cells, inset = slice_cells(flat, rows, cols, inset, snap)

    if save_cells:
        os.makedirs(save_cells, exist_ok=True)
        if debug:
            cv2.imwrite(os.path.join(save_cells, "_rectified.png"), flat)
        for i, cell in enumerate(cells):
            cv2.imwrite(
                os.path.join(save_cells, f"cell_{i // cols}_{i % cols}.png"), cell
            )

    # blank cells -> None so OCR skips them (no wasted inference, no hallucination)
    to_read = [None if ocr.is_blank(cell) else cell for cell in cells]
    n_filled = sum(c is not None for c in to_read)
    print(
        f"OCR: {n_filled}/{len(cells)} non-blank cells (TrOCR, this may take a moment)..."
    )
    text = ocr.ocr_cells(to_read, batch_size=batch, model_name=model_name)

    print_grid(text, rows, cols)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for r in range(rows):
            writer.writerow([text[r * cols + c] for c in range(cols)])
    print(f"wrote {csv_path}  ({rows}x{cols}, {n_filled} non-blank)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("photo")
    p.add_argument("--csv", default="cells.csv", help="output CSV path")
    p.add_argument(
        "--inset", type=int, default=-1, help="px trimmed per edge (-1 auto)"
    )
    p.add_argument("--save-cells", default=None, help="also dump cell PNGs to this dir")
    p.add_argument("--batch", type=int, default=16, help="OCR batch size")
    p.add_argument(
        "--model",
        choices=list(ocr.MODELS),
        default=ocr.DEFAULT_MODEL,
        help="TrOCR model: large (words/phrases) or base (single chars, faster)",
    )
    p.add_argument(
        "--no-snap",
        action="store_true",
        help="skip snapping cell boundaries onto the printed grid lines",
    )
    p.add_argument(
        "--debug", action="store_true", help="with --save-cells, also save rectified"
    )
    a = p.parse_args()
    extract(
        a.photo, a.csv, a.inset, a.save_cells, a.debug, a.batch, not a.no_snap, a.model
    )


if __name__ == "__main__":
    main()

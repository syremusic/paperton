#!/usr/bin/env python3
"""
extract_grid.py  --  Read a photo/scan of a grid sheet and slice out every cell.

Self-describing AND scale-free: the corner markers encode rows/cols, and the grid
is reconstructed RELATIVE TO THE MARKERS (the quad of their inner corners), so you
never pass the grid size or any geometry, and it works at any print zoom.

    python3 extract_grid.py photo.jpg                 # blind, auto-detects size
    python3 extract_grid.py photo.jpg --out cells --debug

Pipeline: detect 4 markers -> decode rows/cols from their IDs -> map the markers'
inner corners to a canonical rectangle (homography undoes perspective + scale) ->
warp -> slice into an even rows x cols grid.
"""
import argparse, os
import numpy as np
import cv2
import gridlib as gl


def extract(photo_path, out_dir, inset, debug):
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
    print(f"detected: rows={rows} cols={cols}")

    missing = [r for r in (0, 1, 2, 3) if role_to_id[r] not in found]
    if missing:
        raise SystemExit(f"missing corner markers for roles {missing}")

    C = gl.canonical(rows, cols)
    src = np.array(
        [found[role_to_id[r]][gl.INNER[r]] for r in (0, 1, 2, 3)], np.float32
    )
    dst = np.array([C["dst_inner"][r] for r in (0, 1, 2, 3)], np.float32)

    Hmat = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(img, Hmat, (C["W"], C["H"]))

    os.makedirs(out_dir, exist_ok=True)
    if debug:
        cv2.imwrite(os.path.join(out_dir, "_rectified.png"), flat)

    xs, ys = C["xs"], C["ys"]
    if inset < 0:
        inset = max(1, int(0.06 * gl.CELL))
    inset = max(0, min(inset, (gl.CELL - 2) // 2))

    n = 0
    for r in range(rows):
        for c in range(cols):
            crop = flat[
                ys[r] + inset : ys[r + 1] - inset, xs[c] + inset : xs[c + 1] - inset
            ]
            cv2.imwrite(os.path.join(out_dir, f"cell_{r}_{c}.png"), crop)
            n += 1
    print(f"extracted {n} cells -> {out_dir}/ (inset {inset}px, cell {gl.CELL}px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("photo")
    p.add_argument("--out", default="cells")
    p.add_argument(
        "--inset", type=int, default=-1, help="px trimmed per edge (-1 auto)"
    )
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    extract(a.photo, a.out, a.inset, a.debug)


if __name__ == "__main__":
    main()

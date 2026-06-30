#!/usr/bin/env python3
"""
make_grid.py  --  Generate a printable, SELF-DESCRIBING grid sheet.

Always US-letter landscape. The four corner ArUco markers encode the sheet's own
rows/cols, and the grid is drawn to exactly fill the markers' inner-corner quad, so
extract_grid.py can read any sheet with no sidecar -- at any print scale.

Examples:
    python3 make_grid.py --rows 6  --cols 4
    python3 make_grid.py --rows 8  --cols 64
    python3 make_grid.py --rows 12 --cols 20 --out sheet
"""
import argparse
import numpy as np
import cv2
import gridlib as gl


def build(rows, cols, over):
    L = gl.print_layout(rows, cols, **over)
    img = np.full((L["H"], L["W"]), 255, np.uint8)
    ids = gl.encode_ids(rows, cols)

    dic = cv2.aruco.getPredefinedDictionary(gl.ARUCO_DICT)
    for role, (x, y) in L["mpos"].items():
        m = cv2.aruco.generateImageMarker(dic, ids[role], L["marker"])
        img[y : y + L["marker"], x : x + L["marker"]] = m

    xs, ys = L["xs"], L["ys"]
    for x in xs:
        cv2.line(img, (x, ys[0]), (x, ys[-1]), 0, L["line_px"])
    for y in ys:
        cv2.line(img, (xs[0], y), (xs[-1], y), 0, L["line_px"])

    return img, L, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--dpi", type=int, default=gl.PRINT["dpi"])
    p.add_argument("--margin", type=float, default=gl.PRINT["margin_in"])
    p.add_argument("--marker", type=float, default=gl.PRINT["marker_in"])
    p.add_argument("--out", default="grid")
    a = p.parse_args()

    over = dict(dpi=a.dpi, margin_in=a.margin, marker_in=a.marker)
    img, L, ids = build(a.rows, a.cols, over)
    cv2.imwrite(f"{a.out}.png", img)

    cwp, chp = L["xs"][1] - L["xs"][0], L["ys"][1] - L["ys"][0]
    print(f"wrote {a.out}.png ({L['W']}x{L['H']} px @ {a.dpi}dpi, letter-landscape)")
    print(f"  encoded ids {ids}  ->  rows={a.rows} cols={a.cols}")
    print(f"  cell ~{cwp}x{chp}px | print at any scale; extraction is marker-relative")
    if min(cwp, chp) < 24:
        print("  NOTE: cells small; raise --dpi for more detail per cell.")


if __name__ == "__main__":
    main()

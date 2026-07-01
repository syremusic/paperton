#!/usr/bin/env python
"""
explainer.py  --  One-figure visual of how v2 turns a photo into text.

Three steps: detect markers -> flatten & slice -> Qwen2.5-VL reads each filled cell.
Geometry is computed live from the real photo; the Qwen readings are hardcoded from an
actual run, so this renders instantly without loading the 8GB model. Output: explainer.png

    python explainer.py
"""
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

import gridlib as gl
from extract_grid import rectify, slice_cells

PHOTO = "../input/IMG_8468.JPG"
BG, FG, DIM, ACC = "#0d1117", "#e6edf3", "#8b949e", "#58a6ff"

# Actual Qwen2.5-VL readings (hardcoded). The first three are the multi-line / multi-word
# cells that v1 could not read at all -- the whole point of v2.
READINGS = [((0, 0), "vinyl clap"), ((1, 1), "long 808"), ((3, 4), "ride")]


def rgb(b):
    return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def main():
    img = cv2.imread(PHOTO)
    if img is None:
        raise SystemExit(f"could not read {PHOTO}")
    rows, cols, flat = rectify(PHOTO)
    cells, _ = slice_cells(flat, rows, cols, -1)
    C = gl.canonical(rows, cols)

    # detect markers for the step-1 overlay
    dic = cv2.aruco.getPredefinedDictionary(gl.ARUCO_DICT)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    found = {int(i): corners[k][0] for k, i in enumerate(ids.flatten())}
    role_to_id = gl.encode_ids(rows, cols)

    fig = plt.figure(figsize=(16, 7), facecolor=BG, dpi=140)
    fig.text(
        0.5,
        0.95,
        "paperton v2 — photo to text, one cell at a time",
        ha="center",
        color=FG,
        fontsize=24,
        fontweight="bold",
    )

    # ---- step 1: detect markers --------------------------------------------------
    ax1 = fig.add_axes([0.02, 0.12, 0.27, 0.74])
    ax1.imshow(rgb(img))
    ax1.axis("off")
    for r in (0, 1, 2, 3):
        ax1.add_patch(
            Polygon(
                found[role_to_id[r]], closed=True, fill=False, edgecolor=ACC, lw=2.5
            )
        )
    ax1.set_title("1. detect the 4 markers", color=FG, fontsize=15, pad=8)
    fig.text(
        0.155,
        0.10,
        f"they self-encode the size  →  {rows} × {cols}",
        ha="center",
        color=DIM,
        fontsize=12,
    )

    # ---- step 2: flatten + slice -------------------------------------------------
    ax2 = fig.add_axes([0.36, 0.12, 0.27, 0.74])
    ax2.imshow(rgb(flat))
    for x in C["xs"]:
        ax2.axvline(x, color=ACC, lw=0.6, alpha=0.7)
    for y in C["ys"]:
        ax2.axhline(y, color=ACC, lw=0.6, alpha=0.7)
    ax2.set_xlim(0, C["W"])
    ax2.set_ylim(C["H"], 0)
    ax2.axis("off")
    ax2.set_title("2. flatten & slice into cells", color=FG, fontsize=15, pad=8)
    fig.text(
        0.495,
        0.10,
        "homography undoes the angle; lines snap to the print",
        ha="center",
        color=DIM,
        fontsize=12,
    )

    # ---- step 3: Qwen reads each filled cell -------------------------------------
    ax3 = fig.add_axes([0.70, 0.12, 0.28, 0.74])
    ax3.axis("off")
    ax3.set_title("3. Qwen2.5-VL reads each cell", color=FG, fontsize=15, pad=8)
    for k, ((r, c), txt) in enumerate(READINGS):
        y = 0.62 - k * 0.20
        axc = fig.add_axes([0.71, y, 0.11, 0.16])
        axc.imshow(rgb(cells[r * cols + c]))
        axc.axis("off")
        for s in axc.spines.values():
            s.set_visible(False)
        fig.add_artist(
            FancyArrowPatch(
                (0.825, y + 0.08),
                (0.865, y + 0.08),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=18,
                color=ACC,
                lw=2,
            )
        )
        fig.text(
            0.875,
            y + 0.08,
            f"“{txt}”",
            va="center",
            color=FG,
            fontsize=17,
            fontweight="bold",
        )
    fig.text(
        0.84,
        0.10,
        "multi-word & multi-line cells, which v1 could not read",
        ha="center",
        color=DIM,
        fontsize=12,
    )

    # arrows between the three steps
    for x0, x1 in [(0.295, 0.355), (0.635, 0.695)]:
        fig.add_artist(
            FancyArrowPatch(
                (x0, 0.49),
                (x1, 0.49),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=26,
                color=DIM,
                lw=2.5,
            )
        )

    fig.savefig("explainer.png", facecolor=BG)
    print("wrote explainer.png")


if __name__ == "__main__":
    main()

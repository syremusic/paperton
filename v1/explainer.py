#!/usr/bin/env python3
"""
explainer.py  --  Render a step-by-step visual of the v1 grid engine for video use.

Runs the real scan (IMG_8466.JPG) through every stage and saves a polished panel
per stage into explainer_out/, plus a combined poster explainer.png. The OCR text
is hardcoded (the actual result from our run) so this renders instantly without
loading the TrOCR model -- it's a presentation asset, not the pipeline itself.

    python3 explainer.py
"""
import os
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch

import gridlib as gl
import ocr
from extract_grid import rectify, slice_cells

PHOTO = "IMG_8466.JPG"
OUT = "explainer_out"

# palette (GitHub-dark, reads well on a YouTube timeline)
BG, FG, DIM = "#0d1117", "#e6edf3", "#8b949e"
ACC, GOOD, BAD = "#58a6ff", "#3fb950", "#f85149"
ROLE_COLOR = {0: "#58a6ff", 1: "#d29922", 2: "#bc8cff", 3: "#3fb950"}
ROLE_NAME = {0: "rows", 1: "cols", 2: "checksum", 3: "anchor"}

# Hardcoded actual OCR output (row-major) + ground truth for the two filled rows.
OCR = {
    0: list("aBcDeF6HIsKLmn0P"),
    1: [
        "a",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "1",
        "12",
        "13",
        "14",
        "1st",
        "16",
    ],
}
GT = {0: list("ABCDEFGHIJKLMNOP"), 1: [str(i) for i in range(1, 17)]}
SAMPLE = 1  # cell index used for the preprocessing close-up ('B')


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def newfig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG, dpi=150)
    return fig


def header(fig, step, title, subtitle):
    fig.text(
        0.063,
        0.945,
        f"STEP {step}/7",
        ha="center",
        va="center",
        color=BG,
        fontsize=15,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc=ACC, ec="none"),
    )
    fig.text(
        0.12,
        0.95,
        title,
        ha="left",
        va="center",
        color=FG,
        fontsize=27,
        fontweight="bold",
    )
    fig.text(0.12, 0.905, subtitle, ha="left", va="center", color=DIM, fontsize=15)
    fig.text(
        0.985,
        0.025,
        "paperton · v1 engine",
        ha="right",
        va="center",
        color=DIM,
        fontsize=12,
    )


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return path


def detect_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dic = cv2.aruco.getPredefinedDictionary(gl.ARUCO_DICT)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    return {int(i): corners[k][0] for k, i in enumerate(ids.flatten())}


# ----------------------------------------------------------------------------- steps
def step1_input(img, found, role_to_id):
    fig = newfig(14, 8.2)
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.80])
    ax.imshow(rgb(img))
    ax.axis("off")
    for role in (0, 1, 2, 3):
        quad = found[role_to_id[role]]
        col = ROLE_COLOR[role]
        ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor=col, lw=3))
        cx, cy = quad.mean(0)
        ax.annotate(
            f"id {role_to_id[role]}  →  {ROLE_NAME[role]}",
            (cx, cy),
            color=BG,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none"),
        )
    header(
        fig,
        1,
        "Find the four corner markers",
        "ArUco markers are detected in the raw photo — no alignment or cropping needed.",
    )
    return save(fig, "01_input_markers.png")


def step2_decode(found, role_to_id, rows, cols):
    fig = newfig(14, 8.2)
    header(
        fig,
        2,
        "Markers describe the sheet",
        "Each marker's ID encodes one number. The sheet tells the engine its own size.",
    )
    rules = [
        (0, role_to_id[0], f"band 0  →  rows = {rows}"),
        (1, role_to_id[1], f"band 1  →  cols = {role_to_id[1]}−250 = {cols}"),
        (
            2,
            role_to_id[2],
            f"band 2  →  checksum {role_to_id[2]}−500 = {role_to_id[2]-500}  ✓ valid",
        ),
        (3, role_to_id[3], "band 3  →  orientation anchor"),
    ]
    for k, (role, mid, text) in enumerate(rules):
        y = 0.74 - k * 0.165
        ax = fig.add_axes([0.12, y, 0.06, 0.12])
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0.02",
                fc=ROLE_COLOR[role],
                ec="none",
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.5,
            0.5,
            str(mid),
            ha="center",
            va="center",
            color=BG,
            fontsize=24,
            fontweight="bold",
            transform=ax.transAxes,
        )
        fig.text(0.22, y + 0.06, text, ha="left", va="center", color=FG, fontsize=20)
    fig.text(
        0.12,
        0.085,
        f"decoded:  {rows} rows × {cols} cols",
        ha="left",
        va="center",
        color=GOOD,
        fontsize=22,
        fontweight="bold",
    )
    return save(fig, "02_decode.png")


def step3_rectify(flat):
    fig = newfig(14, 8.2)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.80])
    ax.imshow(rgb(flat))
    ax.axis("off")
    header(
        fig,
        3,
        "Flatten the perspective",
        "A homography from the four markers warps the photo to a perfect rectangle — "
        "at any angle or print zoom.",
    )
    return save(fig, "03_rectify.png")


def step4_slice(flat, rows, cols):
    C = gl.canonical(rows, cols)
    fig = newfig(14, 8.2)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.80])
    ax.imshow(rgb(flat))
    for x in C["xs"]:
        ax.axvline(x, color=ACC, lw=1.2, alpha=0.9)
    for y in C["ys"]:
        ax.axhline(y, color=ACC, lw=1.2, alpha=0.9)
    ax.set_xlim(0, C["W"])
    ax.set_ylim(C["H"], 0)
    ax.axis("off")
    header(
        fig,
        4,
        "Slice into cells",
        f"The decoded {rows}×{cols} layout cuts the rectangle into {rows*cols} even cells.",
    )
    return save(fig, "04_slice.png")


def step5_blank(cells, rows, cols):
    filled = np.array([0.0 if ocr.is_blank(c) else 1.0 for c in cells]).reshape(
        rows, cols
    )
    n = int(filled.sum())
    fig = newfig(14, 8.2)
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.68])
    ax.imshow(
        filled,
        cmap=matplotlib.colors.ListedColormap(["#161b22", ACC]),
        aspect="equal",
        vmin=0,
        vmax=1,
        extent=(0, cols, rows, 0),
    )
    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color=BG, lw=2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(DIM)
    header(
        fig,
        5,
        "Skip the blank cells",
        f"A cheap ink test finds {n} inked cells of {rows*cols}; only those run OCR.",
    )
    fig.text(0.06, 0.05, "■ inked → OCR", color=ACC, fontsize=15, fontweight="bold")
    fig.text(0.30, 0.05, "■ blank → skipped", color=DIM, fontsize=15, fontweight="bold")
    return save(fig, "05_blank.png")


def _prep_stages(cell):
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), 15)
    norm = cv2.divide(g, bg, scale=255)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(
        (bw < 128).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    ys, xs = np.where(ink > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    char = bw[y0 : y1 + 1, x0 : x1 + 1]
    h, w = char.shape
    pad = int(0.4 * max(h, w))
    canvas = np.full((h + 2 * pad, w + 2 * pad), 255, np.uint8)
    canvas[pad : pad + h, pad : pad + w] = char
    final = cv2.resize(canvas, (128, 128), interpolation=cv2.INTER_CUBIC)
    return [
        ("raw cell", g),
        ("flatten lighting", norm),
        ("binarize", bw),
        ("crop + margin", canvas),
        ("upscale → TrOCR", final),
    ]


def step6_preprocess(cell, result):
    stages = _prep_stages(cell)
    fig = newfig(14, 6.4)
    header(
        fig,
        6,
        "Clean up each glyph",
        "TrOCR expects a centred, high-contrast character. This is the single biggest "
        "accuracy win (19% → 81%).",
    )
    n = len(stages)
    w = 0.128
    gap = (0.80 - n * w) / (n - 1)  # tiles span 0.04..0.84; result sits to the right
    for k, (label, im) in enumerate(stages):
        x = 0.04 + k * (w + gap)
        ax = fig.add_axes([x, 0.30, w, 0.42])
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(DIM)
        fig.text(x + w / 2, 0.24, label, ha="center", va="top", color=FG, fontsize=13)
        if k < n - 1:
            fig.add_artist(
                FancyArrowPatch(
                    (x + w + gap * 0.15, 0.51),
                    (x + w + gap * 0.85, 0.51),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=22,
                    color=ACC,
                    lw=2,
                )
            )
    fig.add_artist(
        FancyArrowPatch(
            (0.855, 0.51),
            (0.90, 0.51),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=24,
            color=GOOD,
            lw=2.5,
        )
    )
    fig.text(
        0.945,
        0.51,
        f"“{result}”",
        ha="center",
        va="center",
        color=GOOD,
        fontsize=44,
        fontweight="bold",
    )
    fig.text(0.945, 0.24, "OCR reads", ha="center", va="top", color=DIM, fontsize=13)
    return save(fig, "06_preprocess.png")


def step7_ocr(flat, rows, cols):
    C = gl.canonical(rows, cols)
    fig = newfig(14, 8.2)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.80])
    ax.imshow(rgb(flat), alpha=0.25)
    for x in C["xs"]:
        ax.axvline(x, color=DIM, lw=0.6, alpha=0.5)
    for y in C["ys"]:
        ax.axhline(y, color=DIM, lw=0.6, alpha=0.5)
    correct = total = 0
    for r in (0, 1):
        for c in range(cols):
            pred, truth = OCR[r][c], GT[r][c]
            ok = pred.lower() == truth.lower()
            correct += ok
            total += 1
            cx = (C["xs"][c] + C["xs"][c + 1]) / 2
            cy = (C["ys"][r] + C["ys"][r + 1]) / 2
            ax.text(
                cx,
                cy,
                pred,
                ha="center",
                va="center",
                color=GOOD if ok else BAD,
                fontsize=22,
                fontweight="bold",
            )
            if not ok:
                ax.text(
                    cx,
                    cy + C["ys"][1] * 0.34,
                    truth,
                    ha="center",
                    va="center",
                    color=DIM,
                    fontsize=11,
                )
    ax.set_xlim(0, C["W"])
    ax.set_ylim(C["H"], 0)
    ax.axis("off")
    header(
        fig,
        7,
        "Read every cell with TrOCR",
        f"Local handwriting OCR — {correct}/{total} filled cells correct ({correct/total:.0%}).  "
        "Green = match, red = miss (truth shown small).",
    )
    return save(fig, "07_ocr.png")


def poster(paths):
    """Stack the stage PNGs into one tall poster."""
    imgs = [cv2.imread(p) for p in paths]
    w = min(im.shape[1] for im in imgs)
    sep = np.full((6, w, 3), 0x30, np.uint8)
    rowsimg = []
    for im in imgs:
        s = w / im.shape[1]
        rowsimg.append(cv2.resize(im, (w, int(im.shape[0] * s))))
        rowsimg.append(sep)
    cv2.imwrite("explainer.png", cv2.vconcat(rowsimg[:-1]))


def main():
    img = cv2.imread(PHOTO)
    if img is None:
        raise SystemExit(f"could not read {PHOTO}")
    found = detect_markers(img)
    rows, cols, flat = rectify(PHOTO)
    role_to_id = gl.encode_ids(rows, cols)
    cells, _ = slice_cells(flat, rows, cols, -1)

    paths = [
        step1_input(img, found, role_to_id),
        step2_decode(found, role_to_id, rows, cols),
        step3_rectify(flat),
        step4_slice(flat, rows, cols),
        step5_blank(cells, rows, cols),
        step6_preprocess(cells[SAMPLE], OCR[SAMPLE // cols][SAMPLE % cols]),
        step7_ocr(flat, rows, cols),
    ]
    poster(paths)
    print(f"wrote {len(paths)} stage panels to {OUT}/ and combined explainer.png")


if __name__ == "__main__":
    main()

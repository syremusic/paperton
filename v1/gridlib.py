"""
gridlib.py  --  Shared geometry + self-describing marker encoding.

Sheets are ALWAYS US-letter landscape. The four corner ArUco markers encode the
grid's rows/cols in their IDs, and -- the key design point -- the grid is defined
RELATIVE TO THE MARKERS, never to absolute paper coordinates. The grid occupies
exactly the quadrilateral spanned by the four markers' inner corners.

That makes extraction independent of print scale: print at 50% or 90% zoom, on any
printer, and it still rectifies and slices correctly, because the homography is fit
to the detected markers and everything downstream is pure ratios. Paper size,
margins and DPI never enter the scan path -- they only affect how a sheet prints.

Each corner lives in its own ID band, which also fixes orientation:

    band 0  (ids   0-249)  TopLeft     -> rows
    band 1  (ids 250-499)  TopRight    -> cols
    band 2  (ids 500-749)  BottomRight -> checksum (validates the read)
    band 3  (ids 750-999)  BottomLeft  -> anchor (constant)

Both generator and extractor import this one module, so they cannot disagree about
where cells are.
"""

import cv2

ARUCO_DICT = cv2.aruco.DICT_5X5_1000
BAND = 250
CELL = 100  # canonical px per cell in the rectified grid

# Physical print layout -- US-letter landscape only. These affect ONLY how a sheet
# is printed; extraction never reads them (it works purely off the markers).
PRINT = dict(
    dpi=300, page_w_in=11.0, page_h_in=8.5, margin_in=0.25, marker_in=0.6, line_px=2
)

# ArUco reports each marker's 4 corners as TL,TR,BR,BL in the marker's own frame.
# For each page corner, the corner facing the grid centre is a different index:
INNER = {0: 2, 1: 3, 2: 0, 3: 1}


def checksum(rows, cols):
    return (rows * 7 + cols * 13) % BAND


def encode_ids(rows, cols):
    if not (1 <= rows < BAND and 1 <= cols < BAND):
        raise ValueError(f"rows/cols must each be 1..{BAND - 1}")
    return {0: rows, 1: BAND + cols, 2: 2 * BAND + checksum(rows, cols), 3: 3 * BAND}


def decode_ids(detected_ids):
    """Return (rows, cols) or None if the markers don't form a valid sheet."""
    by_role = {}
    for i in detected_ids:
        role = i // BAND
        if 0 <= role <= 3:
            by_role.setdefault(role, []).append(i)
    if set(by_role) != {0, 1, 2, 3} or any(len(v) != 1 for v in by_role.values()):
        return None
    rows = by_role[0][0]
    cols = by_role[1][0] - BAND
    cs = by_role[2][0] - 2 * BAND
    if rows < 1 or cols < 1 or checksum(rows, cols) != cs:
        return None
    return rows, cols


def canonical(rows, cols):
    """Rectified grid in canonical pixels. The markers' inner corners map to the
    rectangle corners; cells are an even subdivision of that rectangle."""
    W, H = cols * CELL, rows * CELL
    xs = [c * CELL for c in range(cols + 1)]
    ys = [r * CELL for r in range(rows + 1)]
    dst_inner = {0: (0, 0), 1: (W, 0), 2: (W, H), 3: (0, H)}
    return dict(W=W, H=H, xs=xs, ys=ys, dst_inner=dst_inner)


def print_layout(rows, cols, **override):
    """Pixel layout for PRINTING one letter-landscape sheet. Cells are SQUARE: the
    grid uses the largest square cell that fits, is anchored at the top-left, and
    the markers hug its corners (so a short grid leaves the lower page blank rather
    than stretching). The markers' inner corners are the grid corners -> matches
    canonical()."""
    p = {**PRINT, **override}
    dpi = p["dpi"]
    px = lambda inches: int(round(inches * dpi))
    W, H = px(p["page_w_in"]), px(p["page_h_in"])
    margin, marker = px(p["margin_in"]), px(p["marker_in"])
    gx0 = gy0 = margin + marker  # top-left grid corner = TL marker's inner corner
    avail_w = W - 2 * (margin + marker)
    avail_h = H - 2 * (margin + marker)
    if avail_w <= 0 or avail_h <= 0:
        raise ValueError("page too small for these margins/markers")
    cell = min(avail_w / cols, avail_h / rows)  # largest square cell that fits
    gx1, gy1 = round(gx0 + cell * cols), round(gy0 + cell * rows)
    mpos = {  # each marker's inner corner sits exactly on a grid corner
        0: (gx0 - marker, gy0 - marker),
        1: (gx1, gy0 - marker),
        2: (gx1, gy1),
        3: (gx0 - marker, gy1),
    }
    xs = [round(gx0 + c * cell) for c in range(cols + 1)]
    ys = [round(gy0 + r * cell) for r in range(rows + 1)]
    return dict(
        W=W,
        H=H,
        marker=marker,
        mpos=mpos,
        xs=xs,
        ys=ys,
        line_px=p["line_px"],
        dpi=dpi,
        cell_w=cell,
        cell_h=cell,
    )

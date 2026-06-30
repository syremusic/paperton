"""
ocr.py  --  Handwriting OCR for grid cells via Microsoft TrOCR (local model).

Runs entirely offline after the first run (which downloads ~1.3GB of weights).
Uses Apple MPS / CUDA if available, else CPU.

TrOCR is a line/word recogniser, so isolated cell glyphs need help. Two cheap steps
roughly double accuracy on handwriting (~19% -> ~81% on our test sheet):

  * is_blank()  skips empty boxes (no wasted inference, no hallucinated text).
  * prep()      turns each cell into a clean, centred black-on-white glyph with
                margin -- the kind of image TrOCR was trained on.
  * clean()     strips the stray punctuation / doubled tokens TrOCR emits for
                single characters ("3 3" -> "3", "F." -> "F").
"""

import re
import cv2
import numpy as np
from functools import lru_cache

# "large" handles multi-letter words/phrases best (it's a line recogniser) and is
# the right default for open handwriting like "vinyl kick". "base" is faster and a
# bit better on ISOLATED single characters. Both share TrOCR's quirk of nudging
# block capitals toward common words (HAT -> that); large just does it less.
MODELS = {
    "large": "microsoft/trocr-large-handwritten",
    "base": "microsoft/trocr-base-handwritten",
}
DEFAULT_MODEL = "large"


@lru_cache(maxsize=2)
def _model(name=DEFAULT_MODEL):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    repo = MODELS.get(name, name)
    proc = TrOCRProcessor.from_pretrained(repo)
    model = VisionEncoderDecoderModel.from_pretrained(repo)
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    model.to(dev).eval()
    return proc, model, dev


def is_blank(cell_bgr, thresh=0.006):
    """True if the cell has essentially no ink (empty box).

    Measures the fraction of pixels clearly darker than the paper background. We do
    NOT use Otsu here: Otsu forces a foreground/background split even on blank paper
    and would flag every empty cell as inked. A handful of speckle pixels stay well
    under `thresh`; real strokes (even faint pencil) sit far above it.
    """
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    m = int(0.12 * max(h, w))  # drop any residual grid-line border
    if m:
        g = g[m : h - m, m : w - m]
    bg = np.percentile(g, 80)  # paper level
    return float((g < bg - 55).mean()) < thresh


def prep(cell_bgr):
    """Normalise a cell into a centred black-on-white glyph TrOCR can read.

    Flatten illumination, binarise, crop to the ink, pad with a white margin, and
    upscale. Returns an RGB image, or None if no ink is found.
    """
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), 15)
    norm = cv2.divide(g, bg, scale=255)  # flatten uneven lighting
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(
        (bw < 128).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    ys, xs = np.where(ink > 0)
    if len(xs) < 5:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    char = bw[y0 : y1 + 1, x0 : x1 + 1]
    h, w = char.shape
    pad = int(0.4 * max(h, w))
    canvas = np.full((h + 2 * pad, w + 2 * pad), 255, np.uint8)
    canvas[pad : pad + h, pad : pad + w] = char
    s = 128 / canvas.shape[0]
    canvas = cv2.resize(canvas, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


def clean(s):
    """Tidy TrOCR output while preserving multi-word cells ("vinyl kick").

    Strip stray punctuation from each token and collapse consecutive duplicates,
    which is how TrOCR doubles a lone glyph ("3 3" -> "3", "HAT HAT" -> "HAT"),
    without dropping real extra words.
    """
    toks = [re.sub(r"[^A-Za-z0-9]", "", t) for t in s.split()]
    toks = [t for t in toks if t]
    out = []
    for t in toks:
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return " ".join(out)


def ocr_cells(cells_bgr, batch_size=16, model_name=DEFAULT_MODEL):
    """OCR a list of BGR cell crops. Entries that are None (blank) yield ''.

    Returns a list[str] aligned with the input.
    """
    import torch
    from PIL import Image

    proc, model, dev = _model(model_name)
    results = [""] * len(cells_bgr)
    imgs, owners = [], []
    for i, cell in enumerate(cells_bgr):
        if cell is None:
            continue
        p = prep(cell)
        rgb = p if p is not None else cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        imgs.append(Image.fromarray(rgb))
        owners.append(i)

    for b in range(0, len(imgs), batch_size):
        pix = proc(
            images=imgs[b : b + batch_size], return_tensors="pt"
        ).pixel_values.to(dev)
        with torch.no_grad():
            out = model.generate(pix, max_new_tokens=16)  # room for short phrases
        for i, t in zip(
            owners[b : b + batch_size], proc.batch_decode(out, skip_special_tokens=True)
        ):
            results[i] = clean(t)
    return results

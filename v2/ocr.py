"""
ocr.py  --  Per-cell handwriting recognition via Qwen2.5-VL-3B (local, transformers).

A vision-language model, unlike v1's line-level TrOCR, natively reads multi-word and
multi-line cells ("vinyl clap", "long 808", "E / sub") -- the content v1 could not.
Runs locally on Apple MPS in bf16 (falls back to fp16, then CPU). Blank cells are
detected cheaply and skipped, so sparse sheets only cost a handful of model calls.

Pipeline per cell:  is_blank? -> skip ;  else  prep_vl (upscale, RGB) -> Qwen -> text
"""

import cv2
import numpy as np
from functools import lru_cache

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "Transcribe the handwritten text in this image exactly. It may contain multiple "
    "words or lines; read top to bottom and separate words with a single space. "
    "Reply with only the transcribed text, nothing else. If the image is blank, reply "
    "with nothing."
)


def is_blank(cell_bgr, thresh=0.006):
    """True if the cell has essentially no ink (empty box).

    Fraction of pixels clearly darker than the paper background. Not Otsu -- Otsu forces
    a split on blank paper and would flag every empty cell as inked.
    """
    g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    m = int(0.12 * max(h, w))  # drop any residual grid-line border
    if m:
        g = g[m : h - m, m : w - m]
    bg = np.percentile(g, 80)  # paper level
    return float((g < bg - 55).mean()) < thresh


def prep_vl(cell_bgr, target=240):
    """Cell crop -> RGB PIL image sized for the VLM. Just an upscale (cubic) + colour
    convert -- no binarisation, which throws away the grayscale cues a VLM uses."""
    from PIL import Image

    h, w = cell_bgr.shape[:2]
    s = target / max(h, w)
    if s > 1:
        cell_bgr = cv2.resize(
            cell_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC
        )
    return Image.fromarray(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB))


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if torch.backends.mps.is_available():
        dev, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        dev, dtype = "cuda", torch.bfloat16
    else:
        dev, dtype = "cpu", torch.float32
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=dtype)
    except Exception:  # some MPS builds lack bf16 kernels
        dtype = torch.float16 if dev != "cpu" else torch.float32
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, dtype=dtype)
    model = model.to(dev).eval()
    # bound the visual-token count so cell images stay cheap and RAM-safe
    proc = AutoProcessor.from_pretrained(
        MODEL, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )
    return model, proc, dev


def _read(pil):
    import torch

    model, proc, dev = _model()
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    ans = proc.batch_decode(
        [out[0][inputs.input_ids.shape[-1] :]], skip_special_tokens=True
    )[0]
    return " ".join(ans.split())  # collapse stray whitespace/newlines


def recognize(cells_bgr):
    """Recognize a list of BGR cell crops. Entries that are None (blank) yield ''.

    One model call per filled cell; returns a list[str] aligned with the input.
    """
    results = [""] * len(cells_bgr)
    for i, cell in enumerate(cells_bgr):
        if cell is None:
            continue
        results[i] = _read(prep_vl(cell))
    return results

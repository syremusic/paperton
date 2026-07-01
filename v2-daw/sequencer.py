"""
sequencer.py  --  Turn (sample, trigger-steps) tracks into one mixed instrumental.

The audio model, and why volumes are handled the way they are:

  * Everything is float32 in [-1, 1]. Digital audio clips ("blows out the speakers")
    the instant a sample exceeds 1.0, so we never clip until the very end.
  * Each one-shot sample is peak-normalised on load, so a naturally-quiet and a
    naturally-loud sample sit at a comparable level (rough balance between tracks).
  * A track/stem is built by *adding* its sample into a silent buffer at each trigger
    onset. One-shots ring out and overlap later steps -- that's musically correct, and
    summing is exactly how overlapping sound combines.
  * The master is the sum of all stems. Summing N tracks can easily exceed 1.0, so the
    final step peak-normalises the whole mix to -1 dBFS: one scalar gain that scales the
    loudest peak to just under full-scale, preserving every track's relative level while
    guaranteeing no clipping.

Build stems first, then mix -- so a stem can be inspected/exported on its own.
"""

import numpy as np
import soundfile as sf
import librosa

SR = 44100


def step_duration(bpm, steps_per_beat=4):
    """Seconds per grid column. steps_per_beat=4 => each column is a 16th note."""
    return 60.0 / bpm / steps_per_beat


def load_sample(path, sr=SR, peak=0.9):
    """Load an audio file as mono float32, peak-normalised to `peak`."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y.astype(np.float32)
    m = float(np.abs(y).max())
    if m > 0:
        y *= peak / m
    return y


def build_stem(sample, onsets_sec, total_sec, sr=SR):
    """A single track: `sample` overlaid at every onset. The buffer is long enough for
    the last hit to ring out fully (no abrupt cut)."""
    n = int(round(total_sec * sr)) + len(sample) + 1
    buf = np.zeros(n, np.float32)
    for t in onsets_sec:
        i = int(round(t * sr))
        buf[i : i + len(sample)] += sample  # add = overlay overlapping tails
    return buf


def mix(stems, target_peak=0.89):
    """Sum stems and peak-normalise the master to `target_peak` (~-1 dBFS). This is the
    single gain that keeps the mix from clipping while preserving track balance."""
    n = max((len(s) for s in stems), default=0)
    master = np.zeros(n, np.float32)
    for s in stems:
        master[: len(s)] += s
    peak = float(np.abs(master).max())
    if peak > 0:
        master *= target_peak / peak
    return np.clip(master, -1.0, 1.0)


def write_wav(path, audio, sr=SR):
    sf.write(path, audio, sr, subtype="PCM_16")

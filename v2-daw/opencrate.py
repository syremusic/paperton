"""
opencrate.py  --  Client for the OpenCrate /search API + probabilistic sample choice.

`search(query, k)` hits the local server (text -> ranked [{path, score}] via CLAP).
`pick(hits, temperature)` turns the scores into a probability distribution and samples
from it, so higher-scoring matches are more likely but not guaranteed -- this is what
gives the sequencer variety instead of always grabbing the single top hit.
"""

import numpy as np
import requests

DEFAULT_SERVER = "http://localhost:8000"


def search(query, k=8, server=DEFAULT_SERVER, timeout=30):
    """Text query -> list of {path, score}, ranked. Empty list on empty query."""
    if not query.strip():
        return []
    r = requests.post(
        f"{server}/search", json={"query": query, "k": k}, timeout=timeout
    )
    r.raise_for_status()
    return r.json()


def pick(hits, temperature=0.05, rng=None):
    """Sample one hit with probability ~ softmax(score / temperature).

    CLAP scores sit in a narrow band (~0.45-0.51), so a small temperature is needed to
    turn small score gaps into a real preference. Returns (hit, probability).
    """
    if not hits:
        return None, 0.0
    rng = rng or np.random.default_rng()
    scores = np.array([h["score"] for h in hits], dtype=np.float64)
    logits = (scores - scores.max()) / max(temperature, 1e-6)
    p = np.exp(logits)
    p /= p.sum()
    idx = int(rng.choice(len(hits), p=p))
    return hits[idx], float(p[idx])

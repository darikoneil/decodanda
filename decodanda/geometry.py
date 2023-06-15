from __future__ import annotations
import numpy as np


def compute_centroids(conditioned_rasters: dict) -> dict:
    return {w: np.hstack([np.nanmean(r, 0) for r in conditioned_rasters[w]]) for w in conditioned_rasters.keys()}

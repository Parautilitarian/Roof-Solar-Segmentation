"""Shared helpers for the Roof-Solar-Segmentation data pipeline.

Used by the SpaceNet and North Carolina preparation scripts.  Everything here is
deliberately dependency-light: numpy, Pillow, rasterio, shapely, pyproj.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
#  Image scaling (proposal Sec. 3.1: 16-bit -> 8-bit, one fixed stretch)
# --------------------------------------------------------------------------- #
def compute_percentile_stretch(sample_arrays, lo=2.0, hi=98.0):
    """Percentiles over a sample of images, computed once (train partition only
    in the full pipeline) and then applied identically to every chip."""
    pixels = np.concatenate(
        [np.asarray(a, dtype=np.float64).reshape(a.shape[0], -1) for a in sample_arrays],
        axis=1,
    ) if False else None
    # Simpler, memory-safe path: per-band percentiles averaged over samples.
    los, his = [], []
    for a in sample_arrays:
        arr = np.asarray(a, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None, ...]
        los.append(np.percentile(arr, lo, axis=(1, 2)))
        his.append(np.percentile(arr, hi, axis=(1, 2)))
    return np.mean(los, axis=0), np.mean(his, axis=0)


def apply_stretch(array, lo_vals, hi_vals):
    """(bands,H,W) numeric array -> (H,W,3) uint8 using fixed per-band lo/hi."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3)
    arr = arr[:3]
    out = np.empty((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
    for b in range(3):
        lo, hi = float(lo_vals[b]), float(hi_vals[b])
        scaled = np.clip((arr[b] - lo) / max(hi - lo, 1e-6), 0, 1)
        out[..., b] = (scaled * 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
#  Geographic block split (proposal Sec. 3.1: ~1 km blocks, one-chip buffer)
# --------------------------------------------------------------------------- #
def block_id_for_point(x, y, block_m=1000.0):
    return (int(np.floor(x / block_m)), int(np.floor(y / block_m)))


def assign_block_splits(block_ids, fractions=(0.70, 0.15, 0.15), seed=661):
    """Deterministically assign unique block ids -> train/val/test."""
    uniq = sorted(set(block_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    mapping = {}
    for i, b in enumerate(uniq):
        mapping[b] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return mapping


def buffered_blocks(block_ids):
    """All neighbouring blocks (incl. diagonals) -- used to enforce the
    one-chip buffer: a chip is dropped if its block touches another split."""
    out = set()
    for bx, by in block_ids:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.add((bx + dx, by + dy))
    return out


# --------------------------------------------------------------------------- #
#  Small IO helpers
# --------------------------------------------------------------------------- #
def save_png(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_overlay(image_rgb, mask, path, alpha=0.45):
    """Red overlay of the (binary) mask on the RGB chip, for visual QA."""
    img = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    overlay = Image.new("RGB", img.size, (255, 0, 0))
    m = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * int(255 * alpha))
    img.paste(overlay, (0, 0), m)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], outline=(255, 255, 0))
    save_png(np.asarray(img), path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def stable_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

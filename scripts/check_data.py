#!/usr/bin/env python3
"""Verify a prepared dataset against the proposal's midterm data criteria.

Checks (proposal Table 6, midterm):
  1. No chip appears in more than one split (split.csv consistency + on-disk).
  2. Masks are binary and aligned 1:1 with images; chip sizes are uniform.
  3. SpaceNet only: rasterized area within 2% of polygon area (from stats.json).
  4. Overlay QA images exist (inspect >=20 visually before training).
  5. area_capacity module is exact on synthetic masks.

    python scripts/check_data.py --dataset data/processed/spacenet_vegas
    python scripts/check_data.py --dataset data/processed/nc
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = Path(args.dataset)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and cond

    print(f"Checking {ds} ...")
    if not (ds / "split.csv").exists():
        print("  [FAIL] split.csv missing -- run a prepare_*.py script first")
        return 1
    rows = list(csv.DictReader(open(ds / "split.csv")))
    names = [r["chip"] for r in rows]
    check(len(names) == len(set(names)), f"no duplicate chips in split.csv ({len(names)} chips)")
    splits = Counter(r["split"] for r in rows)
    print(f"        splits: {dict(splits)}")

    # on-disk agreement + binary masks + uniform size
    sizes, nonbinary, missing = Counter(), 0, 0
    for r in rows:
        ip, mp = ds / "images" / r["split"] / r["chip"], ds / "masks" / r["split"] / r["chip"]
        if not ip.exists() or not mp.exists():
            missing += 1
            continue
        img, msk = Image.open(ip), np.asarray(Image.open(mp))
        sizes[(img.size, msk.shape[::-1])] += 1
        if not set(np.unique(msk)).issubset({0, 255}):
            nonbinary += 1
    check(missing == 0, f"every split.csv chip exists under images/ and masks/ (missing={missing})")
    check(len(sizes) == 1, f"uniform image/mask size: {dict(sizes)}")
    check(nonbinary == 0, f"all masks binary 0/255 (non-binary={nonbinary})")

    if (ds / "stats.json").exists():
        stats = json.loads((ds / "stats.json").read_text())
        print(f"        stats: chips={stats.get('chips')} "
              f"roof_fraction={stats.get('roof_pixel_fraction_mean')}")
        area = stats.get("area_check") or {}
        if area:
            check(abs(area["median_raster_over_polygon_area"] - 1) <= 0.02,
                  f"median raster/polygon area ratio = {area['median_raster_over_polygon_area']} "
                  f"(target within 2%)")
            print(f"        per-chip within 2%: {area.get('frac_chips_within_2pct')}")

    overlays = list((ds / "overlays").glob("*.png")) if (ds / "overlays").exists() else []
    check(len(overlays) >= 1, f"{len(overlays)} QA overlays in overlays/ -- VISUALLY INSPECT "
                              f">=20 before training (proposal midterm criterion)")

    rc = subprocess.run([sys.executable, str(ROOT / "scripts/area_capacity.py"),
                         "--self-test"], capture_output=True, text=True)
    check(rc.returncode == 0, "area_capacity.py synthetic self-test")
    if rc.returncode:
        print(rc.stdout, rc.stderr)

    print("RESULT:", "ALL CHECKS PASSED" if ok else "FAILURES ABOVE -- do not train yet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

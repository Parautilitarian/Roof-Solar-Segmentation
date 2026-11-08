#!/usr/bin/env python3
"""Masks -> roof area (m^2) -> solar capacity (kWp).  Proposal Sec. 3.5.

    A_k   = sum over predicted-roof pixels of pixel area
    C_k   = u * rho * A_k          (u = 0.70 usable, rho = 0.16 kWp/m^2)

Because C is linear in A, the *relative* capacity error equals the relative
area error; changing u or rho rescales every model equally and is never
evidence that one model is better (proposal Sec. 3.5).  These are estimates
under stated assumptions -- not structural-suitability or generation claims.

Usage
-----
    python scripts/area_capacity.py --self-test
    python scripts/area_capacity.py --pred-dir results/preds --mask-dir data/processed/nc/masks/test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

U_DEFAULT, RHO_DEFAULT = 0.70, 0.16  # after Gagnon et al. (NREL, 2016)


def area_m2(mask, pixel_area_m2):
    return float((np.asarray(mask) > 0).sum()) * float(pixel_area_m2)


def capacity_kwp(area, u=U_DEFAULT, rho=RHO_DEFAULT):
    return float(u) * float(rho) * float(area)


def compare(pred_mask, true_mask, pixel_area_m2, u=U_DEFAULT, rho=RHO_DEFAULT):
    a_hat, a = area_m2(pred_mask, pixel_area_m2), area_m2(true_mask, pixel_area_m2)
    c_hat, c = capacity_kwp(a_hat, u, rho), capacity_kwp(a, u, rho)
    return dict(area_pred_m2=a_hat, area_true_m2=a, area_error_m2=a_hat - a,
                area_rel_error=(a_hat - a) / a if a else float("nan"),
                capacity_pred_kwp=c_hat, capacity_true_kwp=c,
                capacity_error_kwp=c_hat - c)


def load_mask(path):
    return (np.asarray(Image.open(path)) > 0).astype(np.uint8)


def self_test():
    rng = np.random.default_rng(661)
    failures = []
    # 1. exactness on synthetic masks (midterm criterion)
    m = np.zeros((100, 100), np.uint8); m[10:30, 10:50] = 1  # 800 px
    if area_m2(m, 0.09) != 72.0:
        failures.append("synthetic area != 72.0 m2")
    r = compare(m, m, 0.09)
    if r["area_error_m2"] != 0 or r["capacity_error_kwp"] != 0:
        failures.append("identical masks must give zero error")
    # 2. relative area error == relative capacity error (linearity)
    pred = (rng.random((200, 200)) > 0.7).astype(np.uint8)
    true = (rng.random((200, 200)) > 0.6).astype(np.uint8)
    r = compare(pred, true, 0.09)
    cap_rel = (r["capacity_pred_kwp"] - r["capacity_true_kwp"]) / r["capacity_true_kwp"]
    if not np.isclose(cap_rel, r["area_rel_error"]):
        failures.append("relative capacity error != relative area error")
    # 3. empty mask
    if area_m2(np.zeros((5, 5), np.uint8), 0.09) != 0.0:
        failures.append("empty mask area != 0")
    if failures:
        print("SELF-TEST FAILED:", failures); return 1
    print("area_capacity self-test: OK (exact on synthetic masks; "
          "relative capacity error == relative area error)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pred-dir", help="directory of predicted mask PNGs")
    ap.add_argument("--mask-dir", help="directory of reference mask PNGs (same filenames)")
    ap.add_argument("--pixel-area-m2", type=float, default=0.09, help="0.30 m GSD -> 0.09")
    ap.add_argument("--u", type=float, default=U_DEFAULT)
    ap.add_argument("--rho", type=float, default=RHO_DEFAULT)
    args = ap.parse_args()

    if args.self_test or not (args.pred_dir and args.mask_dir):
        return self_test()

    rows = []
    for pred_path in sorted(Path(args.pred_dir).glob("*.png")):
        true_path = Path(args.mask_dir) / pred_path.name
        if not true_path.exists():
            continue
        rows.append((pred_path.name, compare(load_mask(pred_path), load_mask(true_path),
                                             args.pixel_area_m2, args.u, args.rho)))
    if not rows:
        print("No matching pred/true mask pairs found."); return 1
    tot_hat = sum(r[1]["area_pred_m2"] for r in rows)
    tot_true = sum(r[1]["area_true_m2"] for r in rows)
    print(f"{'chip':45s} {'area err m2':>12s} {'rel %':>8s} {'cap err kWp':>12s}")
    for name, r in rows:
        print(f"{name:45s} {r['area_error_m2']:12.1f} {100*r['area_rel_error']:8.2f} "
              f"{r['capacity_error_kwp']:12.2f}")
    print(f"\nAggregate: pred {tot_hat:,.0f} m2 vs true {tot_true:,.0f} m2 "
          f"({100*(tot_hat-tot_true)/tot_true:+.2f}%), "
          f"capacity {capacity_kwp(tot_hat, args.u, args.rho):,.1f} vs "
          f"{capacity_kwp(tot_true, args.u, args.rho):,.1f} kWp")
    print("Note: aggregate signed error can cancel per-area errors (hypothesis H5) -- "
          "report both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

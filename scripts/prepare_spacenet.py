#!/usr/bin/env python3
"""Prepare SpaceNet 2 chips for MMSegmentation-style training.

For every downloaded PS-RGB chip + building GeoJSON:
  * rasterize footprint polygons -> binary mask (1 = roof proxy)
  * convert 16-bit imagery -> 8-bit with ONE percentile stretch computed over
    all downloaded chips (in the full run: the training partition only --
    see proposal Sec. 3.1; with a development subset the difference is noted
    in the stats file)
  * assign the chip to train/val/test by ~1 km geographic block of its centre
    (no random chip-level split -> no leakage between adjacent chips)
  * verify rasterized area vs. polygon area (proposal midterm check: <=2%)

Output
------
    data/processed/spacenet_vegas/
        images/{train,val,test}/*.png      masks/{train,val,test}/*.png
        split.csv                          chip, split, block, centre coords
        stats.json                         counts, roof fraction, area check

    python scripts/prepare_spacenet.py [--city AOI_2_Vegas] [--block-m 1000]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).parent))
from common import (apply_stretch, assign_block_splits, block_id_for_point,
                    compute_percentile_stretch, save_overlay, save_png, write_json)

ROOT = Path(__file__).resolve().parents[1]


def load_polygons(geojson_path, dst_crs):
    data = json.loads(Path(geojson_path).read_text())
    geoms_ll, geoms_dst = [], []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        geoms_ll.append(shape(geom))
        geoms_dst.append(shape(transform_geom("EPSG:4326", dst_crs, geom)))
    return geoms_ll, geoms_dst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="AOI_2_Vegas")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--block-m", type=float, default=1000.0)
    ap.add_argument("--overlays", type=int, default=20,
                    help="number of QA overlay PNGs to write (proposal: >=20)")
    args = ap.parse_args()

    raw = Path(args.raw) if args.raw else ROOT / "data/raw/spacenet" / args.city
    out = Path(args.out) if args.out else ROOT / "data/processed" / (
        "spacenet_vegas" if args.city == "AOI_2_Vegas" else f"spacenet_{args.city.lower()}")
    tifs = sorted((raw / "PS-RGB").glob("*.tif"))
    if not tifs:
        print(f"No chips in {raw}/PS-RGB -- run download_spacenet.py first", file=sys.stderr)
        return 1
    print(f"{len(tifs)} chips found in {raw}")

    # ---- read all chips, collect centres (in the raster CRS, metres) ------- #
    chips = []
    for tif in tifs:
        img_id = tif.stem.split("_img")[-1]
        gj = raw / "geojson_buildings" / (
            f"SN2_buildings_train_{args.city}_geojson_buildings_img{img_id}.geojson")
        if not gj.exists():
            print(f"  skip img{img_id}: no geojson")
            continue
        with rasterio.open(tif) as ds:
            array = ds.read()
            transform, crs, (h, w) = ds.transform, ds.crs, ds.shape
            cx, cy = ds.transform * (w / 2, h / 2)
            bounds = ds.bounds
        _, polys = load_polygons(gj, crs)
        # SpaceNet PS-RGB chips are EPSG:4326.  The proposal requires areas and
        # the geographic split in a metric CRS, so derive the local UTM zone
        # from the chip centre and do all area/block maths there.
        utm_epsg = 32600 + int((cx + 180) // 6) + 1 if crs.to_epsg() == 4326 else None
        if utm_epsg:
            to_utm = Transformer.from_crs(crs, f"EPSG:{utm_epsg}", always_xy=True).transform
            centre_utm = to_utm(cx, cy)
            polys_utm = [shp_transform(to_utm, g) for g in polys]
            xs, ys = to_utm([bounds.left, bounds.right], [bounds.bottom, bounds.top])
            pixel_area = abs((xs[1] - xs[0]) * (ys[1] - ys[0])) / (w * h)
        else:
            centre_utm, polys_utm = (cx, cy), polys
            pixel_area = abs(transform.a * transform.e)
        chips.append(dict(id=f"img{img_id}", tif=tif, array=array, transform=transform,
                          crs=crs, shape=(h, w), centre=centre_utm, polys=polys,
                          polys_utm=polys_utm, pixel_area=pixel_area,
                          utm_epsg=utm_epsg))
    print(f"{len(chips)} chips with matching geojsons; CRS: {chips[0]['crs']}, "
          f"area/block CRS: EPSG:{chips[0]['utm_epsg']}, "
          f"pixel area {chips[0]['pixel_area']:.3f} m^2")

    # ---- one fixed stretch for everyone ------------------------------------ #
    lo, hi = compute_percentile_stretch([c["array"] for c in chips])
    print(f"percentile stretch lo={np.round(lo,1)} hi={np.round(hi,1)}")

    # ---- geographic block split -------------------------------------------- #
    blocks = [block_id_for_point(*c["centre"], args.block_m) for c in chips]
    split_of = assign_block_splits(blocks)
    print("block split counts:", {s: list(split_of[b] for b in blocks).count(s)
                                  for s in ("train", "val", "test")})

    rows, area_ratios, roof_fracs = [], [], []
    n_overlay = 0
    for chip, block in zip(chips, blocks):
        split = split_of[block]
        h, w = chip["shape"]
        mask = rasterize(((g, 1) for g in chip["polys"]), out_shape=(h, w),
                         transform=chip["transform"], fill=0, dtype="uint8")
        image = apply_stretch(chip["array"], lo, hi)
        name = f"{args.city}_{chip['id']}.png"
        save_png(image, out / "images" / split / name)
        save_png((mask * 255).astype(np.uint8), out / "masks" / split / name)

        # area verification: raster area vs polygon (vector) area, both in UTM m^2
        px_area = chip["pixel_area"]
        raster_area = float(mask.sum()) * px_area
        vector_area = float(sum(g.area for g in chip["polys_utm"]))
        if vector_area > 0:
            area_ratios.append(raster_area / vector_area)
        roof_fracs.append(float(mask.mean()))
        rows.append(dict(chip=name, split=split, block=f"{block[0]}_{block[1]}",
                         centre_x=chip["centre"][0], centre_y=chip["centre"][1],
                         roof_px=int(mask.sum()), roof_fraction=round(float(mask.mean()), 5),
                         raster_area_m2=round(raster_area, 1),
                         polygon_area_m2=round(vector_area, 1)))
        if n_overlay < args.overlays:
            save_overlay(image, mask, out / "overlays" / name)
            n_overlay += 1

    with open(out / "split.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = dict(
        dataset=f"SpaceNet 2 {args.city} (PS-RGB, downloaded subset)",
        chips=len(rows),
        splits={s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        chip_shape=chips[0]["shape"], crs=str(chips[0]["crs"]),
        area_crs=f"EPSG:{chips[0]['utm_epsg']}",
        pixel_area_m2=round(float(chips[0]["pixel_area"]), 4),
        roof_pixel_fraction_mean=round(float(np.mean(roof_fracs)), 5),
        stretch_lo=lo.tolist(), stretch_hi=hi.tolist(),
        area_check=dict(
            n_chips_with_buildings=len(area_ratios),
            median_raster_over_polygon_area=round(float(np.median(area_ratios)), 4),
            frac_chips_within_2pct=round(float(np.mean(
                [abs(r - 1) <= 0.02 for r in area_ratios])), 3) if area_ratios else None),
        note="Development subset: stretch computed over all downloaded chips. "
             "For the full run, compute it on the training partition only (proposal Sec. 3.1).",
    )
    write_json(stats, out / "stats.json")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out}/  (images/, masks/, split.csv, stats.json, overlays/ x{n_overlay})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

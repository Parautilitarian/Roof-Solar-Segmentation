#!/usr/bin/env python3
"""Prepare the North Carolina dataset: NC OneMap imagery + NC footprints.

Pipeline per AOI (configs/nc_aois.json):
  * reproject the exported tile from NC State Plane (EPSG:6543, ftUS) to
    UTM 17N (EPSG:32617, metres) at 0.30 m -- the same ground sample distance
    as SpaceNet, in a metric CRS so pixel area is exactly 0.09 m^2
  * load the county footprint shapefile, keep polygons intersecting the tile,
    transform them onto the tile grid and rasterize -> binary roof mask
  * cut the tile into non-overlapping 650x650 chips (195 m, the SpaceNet chip
    footprint) and write MMSegmentation-style images/masks
  * split by CITY, not by chip: Raleigh (Wake) = train, Durham = val,
    Charlotte (Mecklenburg) = test.  Charlotte is therefore a fully held-out
    city -- the NC version of the proposal's geographic split, and the honest
    test for "labels are scarce in a new city".

Output
------
    data/processed/nc/
        images/{train,val,test}/*.png   masks/{train,val,test}/*.png
        split.csv    stats.json    overlays/*.png

    python scripts/prepare_nc.py [--min-roof-frac 0.005]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box
from shapely.ops import transform as shp_transform

sys.path.insert(0, str(Path(__file__).parent))
from common import save_overlay, save_png, write_json

ROOT = Path(__file__).resolve().parents[1]
TARGET_CRS = "EPSG:32617"  # UTM 17N (metres) -- covers all of NC in this project
CHIP = 650                 # px, at 0.30 m -> 195 m chips (SpaceNet: 650px/200 m)


def find_footprint_source(county_dir):
    """NC county downloads are Esri File Geodatabases (.gdb, layer
    S_BUILDING_FP); older/other counties may be shapefiles.  Return
    (path, layer_or_None)."""
    gdbs = sorted(Path(county_dir).rglob("*.gdb"))
    if gdbs:
        return gdbs[0], "S_BUILDING_FP"
    shps = sorted(Path(county_dir).rglob("*.shp"))
    if shps:
        return shps[0], None
    return None, None


def read_footprints(source, layer, bbox_utm_bounds):
    """Read only footprints intersecting the tile.  bbox must be supplied in
    the *source* CRS, so the UTM tile bounds are transformed first.
    Returns shapely geometries already in TARGET_CRS."""
    info = pyogrio.read_info(source, layer=layer) if layer else pyogrio.read_info(source)
    src_crs = info["crs"]
    to_src = Transformer.from_crs(TARGET_CRS, src_crs, always_xy=True).transform
    minx, miny, maxx, maxy = bbox_utm_bounds
    xs, ys = to_src([minx, maxx], [miny, maxy])
    gdf = pyogrio.read_dataframe(source, layer=layer,
                                 bbox=(min(xs), min(ys), max(xs), max(ys)))
    to_utm = Transformer.from_crs(src_crs, TARGET_CRS, always_xy=True).transform
    geoms = [shp_transform(to_utm, g) for g in gdf.geometry if g is not None]
    return geoms, src_crs, len(gdf)


def ensure_georef(src_path):
    """ArcGIS exportImage TIFFs usually embed CRS/transform; if not, restore
    them from the <name>.export.json sidecar written by download_nc.py."""
    with rasterio.open(src_path) as ds:
        if ds.crs is not None and not ds.transform.is_identity:
            return
    sidecar = Path(str(src_path).replace(".tif", ".export.json"))
    if not sidecar.exists():
        raise RuntimeError(f"{src_path} has no georeferencing and no export.json sidecar")
    meta = json.loads(sidecar.read_text())
    ext, w, h = meta["extent"], meta["width"], meta["height"]
    from rasterio.transform import from_bounds
    transform = from_bounds(ext["xmin"], ext["ymin"], ext["xmax"], ext["ymax"], w, h)
    with rasterio.open(src_path, "r+") as ds:
        ds.crs = rasterio.crs.CRS.from_epsg(6543)
        ds.transform = transform
    print(f"  restored georeferencing for {Path(src_path).name} from export.json")


def reproject_tile(src_path, dst_path, gsd=0.30):
    ensure_georef(src_path)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=gsd)
        profile = dict(src.profile, crs=TARGET_CRS, transform=transform,
                       width=width, height=height, compress="lzw")
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b), destination=rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs=TARGET_CRS,
                          resampling=Resampling.bilinear)
    return dst_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/nc_aois.json"))
    ap.add_argument("--raw", default=str(ROOT / "data/raw/nc"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/nc"))
    ap.add_argument("--min-roof-frac", type=float, default=0.0,
                    help="drop chips with less roof than this (default: keep all)")
    ap.add_argument("--overlays", type=int, default=20)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    raw, out = Path(args.raw), Path(args.out)
    rows, roof_fracs, area_ratios, n_overlay = [], [], [], 0
    county_source = {}

    for aoi in cfg["aois"]:
        tif = raw / "imagery" / f"{aoi['name']}.tif"
        if not tif.exists():
            print(f"skip {aoi['name']}: no imagery (run download_nc.py)")
            continue
        county = aoi["county"]
        if county not in county_source:
            source, layer = find_footprint_source(raw / "footprints" / county)
            if source is None:
                print(f"skip {aoi['name']}: no {county} footprints")
                continue
            county_source[county] = (source, layer)
            print(f"{county}: footprint source {source.name} (layer={layer})")
        source, layer = county_source[county]

        utm_tif = raw / "imagery" / f"{aoi['name']}_utm.tif"
        if not utm_tif.exists():
            reproject_tile(tif, utm_tif, gsd=cfg["gsd_m"])

        with rasterio.open(utm_tif) as ds:
            image = np.moveaxis(ds.read(), 0, -1)  # H,W,bands
            transform, (H, W) = ds.transform, ds.shape
            bounds = ds.bounds
        polys, src_crs, n_read = read_footprints(source, layer, bounds)
        full_mask = rasterize(((g, 1) for g in polys), out_shape=(H, W),
                              transform=transform, fill=0, dtype="uint8")
        px_area = abs(transform.a * transform.e)
        print(f"{aoi['name']}: {W}x{H}px, {len(polys)} footprints in tile, "
              f"roof fraction {full_mask.mean():.3f}")

        split = aoi["split_role"]
        n_kept = 0
        for row in range(0, H - CHIP + 1, CHIP):
            for col in range(0, W - CHIP + 1, CHIP):
                m = full_mask[row:row + CHIP, col:col + CHIP]
                frac = float(m.mean())
                if frac < args.min_roof_frac:
                    continue
                img = image[row:row + CHIP, col:col + CHIP, :3].astype(np.uint8)
                if img.shape != (CHIP, CHIP, 3):
                    continue
                name = f"{aoi['name']}_r{row}_c{col}.png"
                save_png(img, out / "images" / split / name)
                save_png((m * 255).astype(np.uint8), out / "masks" / split / name)
                roof_fracs.append(frac)
                rows.append(dict(chip=name, split=split, aoi=aoi["name"], city=aoi["city"],
                                 county=county, row=row, col=col,
                                 roof_fraction=round(frac, 5),
                                 roof_area_m2=round(float(m.sum()) * px_area, 1)))
                n_kept += 1
                if n_overlay < args.overlays:
                    save_overlay(img, m, out / "overlays" / name)
                    n_overlay += 1
        print(f"  -> {n_kept} chips to {split}/")

    if rows:
        with open(out / "split.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    stats = dict(
        dataset="North Carolina pilot: NC OneMap imagery (0.30 m) + NC statewide footprints",
        target_crs=TARGET_CRS, chip_px=CHIP, gsd_m=cfg["gsd_m"],
        pixel_area_m2=cfg["gsd_m"] ** 2,
        chips=len(rows),
        splits={s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        cities={s: sorted({r["city"] for r in rows if r["split"] == s})
                for s in ("train", "val", "test")},
        roof_pixel_fraction_mean=round(float(np.mean(roof_fracs)), 5) if roof_fracs else None,
        split_design="by city: Raleigh=train, Durham=val, Charlotte=test (held-out city)",
        caveats=[
            "Footprint vintages (Wake/Durham 2021, Mecklenburg 2023) can predate the imagery; inspect overlays/.",
            "Footprints are a roof-area proxy, not usable roof area (proposal Sec. 3.1).",
            "Pilot AOIs are 1.2 km tiles; scale out per county once the pipeline is trusted.",
        ],
    )
    write_json(stats, out / "stats.json")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out}/  (images/, masks/, split.csv, stats.json, overlays/ x{n_overlay})")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download the North Carolina data: building footprints + aerial imagery.

Footprints (labels)
    NC statewide building-footprint shapefiles (NC Emergency Management via
    the NC Spatial Data Download site), one zip per county:
    https://sdd.nc.gov/downloads/public/BuildingFootprintsbyCounty/Current/

Imagery
    NC OneMap Orthoimagery Program image service (6-inch native, flown
    leaf-off on a 4-year cycle).  Tiles are exported with ArcGIS
    ``exportImage`` at 0.30 m so they are directly comparable with
    SpaceNet / LoveDA, centred on the AOIs in ``configs/nc_aois.json``.

Examples
--------
    python scripts/download_nc.py                 # footprints + all AOI tiles
    python scripts/download_nc.py --footprints-only
    python scripts/download_nc.py --aoi raleigh_downtown
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}  [{dest.stat().st_size/1e6:.1f} MB]")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300,
                      headers={"User-Agent": "Mozilla/5.0 (ECE661 research)"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def fetch_footprints(cfg, out_dir):
    print("== NC building footprints ==")
    needed = sorted({a["county"] for a in cfg["aois"]})
    for county in needed:
        url = cfg["counties"][county]
        zpath = download(url, out_dir / "footprints" / Path(url).name)
        extract_dir = out_dir / "footprints" / county
        # NC county files are Esri File Geodatabases (.gdb), sometimes shapefiles.
        if not list(extract_dir.rglob("*.gdb")) and not list(extract_dir.rglob("*.shp")):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(extract_dir)
        found = list(extract_dir.rglob("*.gdb")) + list(extract_dir.rglob("*.shp"))
        print(f"  {county}: {len(found)} geospatial source(s) -> "
              f"{found[0] if found else 'NONE FOUND'}")


def export_imagery(cfg, aoi, out_dir):
    """ArcGIS exportImage for one AOI. bbox is given in lon/lat (4326); the
    raster is requested in NC State Plane (EPSG:6543) at the config GSD."""
    size_m, gsd = cfg["aoi_size_m"], cfg["gsd_m"]
    px = int(round(size_m / gsd))
    # ~ size_m box around the centre, in degrees (good enough for a request bbox;
    # the returned raster's own CRS/transform is authoritative thereafter).
    dlat = (size_m / 2) / 111_320.0
    import math
    dlon = (size_m / 2) / (111_320.0 * math.cos(math.radians(aoi["center_lat"])))
    bbox = (f"{aoi['center_lon']-dlon},{aoi['center_lat']-dlat},"
            f"{aoi['center_lon']+dlon},{aoi['center_lat']+dlat}")
    params = {
        "bbox": bbox, "bboxSR": "4326", "imageSR": "6543",
        "size": f"{px},{px}", "format": "tiff", "pixelType": "U8",
        "interpolation": "RSP_BilinearInterpolation",
        "compression": "LZW", "f": "json",
    }
    dest = out_dir / "imagery" / f"{aoi['name']}.tif"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}  [{dest.stat().st_size/1e6:.1f} MB]")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  exporting {aoi['name']} ({px}x{px}px @ {gsd} m) ...")
    # NOTE: f=image returns an HTML error page from this service; f=json
    # returns {"href": <temp file URL>, "extent": {...}} -- download that.
    meta = requests.get(f"{cfg['imagery_service']}/exportImage", params=params,
                        timeout=600,
                        headers={"User-Agent": "Mozilla/5.0 (ECE661 research)"}).json()
    if "href" not in meta:
        raise RuntimeError(f"exportImage failed for {aoi['name']}: {str(meta)[:300]}")
    (dest.parent / f"{aoi['name']}.export.json").write_text(json.dumps(meta, indent=2))
    with requests.get(meta["href"], stream=True, timeout=600,
                      headers={"User-Agent": "Mozilla/5.0 (ECE661 research)"}) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tif.part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    print(f"    -> {dest}  [{dest.stat().st_size/1e6:.1f} MB]")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/nc_aois.json"))
    ap.add_argument("--out", default=str(ROOT / "data/raw/nc"))
    ap.add_argument("--footprints-only", action="store_true")
    ap.add_argument("--imagery-only", action="store_true")
    ap.add_argument("--aoi", help="only this AOI name (imagery)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)

    if not args.imagery_only:
        fetch_footprints(cfg, out)
    if not args.footprints_only:
        print("== NC OneMap imagery (0.30 m exports) ==")
        aois = [a for a in cfg["aois"] if not args.aoi or a["name"] == args.aoi]
        if not aois:
            print(f"No AOI named {args.aoi!r}", file=sys.stderr)
            return 1
        for aoi in aois:
            export_imagery(cfg, aoi, out)
    print("\nNext: python scripts/prepare_nc.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Download SpaceNet 2 (SN2 buildings) data from the public AWS S3 bucket.

No AWS account / credentials needed -- the bucket is public over HTTPS.

The full Las Vegas training tarball is 25.6 GB, which does not fit comfortably
on a laptop that also has to hold the extracted chips.  The default mode
therefore downloads individual PS-RGB chips + their building GeoJSONs (each
chip is 2.5 MB), which is enough to develop / validate the whole pipeline and
to run the 5% label-budget regime.  Use --full-tarball on Colab / a machine
with >=60 GB free disk for the complete experiment grid.

Examples
--------
    # Pipeline-development subset: 120 Vegas chips (~300 MB)
    python scripts/download_spacenet.py --city AOI_2_Vegas --num-chips 120

    # Full city tarball (Colab / big disk only, 25.6 GB download)
    python scripts/download_spacenet.py --city AOI_2_Vegas --full-tarball
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tqdm import tqdm

BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{BUCKET}/?{urllib.parse.urlencode(q)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        keys += [e.text for e in root.findall("s3:Contents/s3:Key", NS)]
        truncated = root.findtext("s3:IsTruncated", namespaces=NS) == "true"
        if not truncated:
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def download(url, dest, desc=""):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                        desc=desc or dest.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def img_id(key):
    m = re.search(r"_img(\d+)\.", key)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="AOI_2_Vegas",
                    choices=["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"])
    ap.add_argument("--num-chips", type=int, default=120,
                    help="individual PS-RGB chips to download (ignored with --full-tarball)")
    ap.add_argument("--full-tarball", action="store_true",
                    help="download the full city tarball instead (25.6 GB for Vegas)")
    ap.add_argument("--out", default="data/raw/spacenet")
    args = ap.parse_args()

    out = Path(args.out) / args.city
    out.mkdir(parents=True, exist_ok=True)
    base = f"spacenet/SN2_buildings/train/{args.city}"

    if args.full_tarball:
        key = f"spacenet/SN2_buildings/train/tarballs/SN2_buildings_train_{args.city}.tar.gz"
        download(f"{BUCKET}/{key}", out.parent / f"SN2_{args.city}.tar.gz", "full tarball")
        print("Extract with:  tar -xzf", out.parent / f"SN2_{args.city}.tar.gz", "-C", out)
        return

    print(f"Listing PS-RGB chips for {args.city} ...")
    rgb_keys = sorted([k for k in list_keys(f"{base}/PS-RGB/") if k.endswith(".tif")],
                      key=img_id)
    gj_keys = {img_id(k): k for k in list_keys(f"{base}/geojson_buildings/")
               if k.endswith(".geojson")}
    ids = [img_id(k) for k in rgb_keys if img_id(k) in gj_keys][: args.num_chips]
    print(f"{len(rgb_keys)} chips available; downloading {len(ids)} chips + geojsons")

    rgb_by_id = {img_id(k): k for k in rgb_keys}
    for i in ids:
        download(f"{BUCKET}/{rgb_by_id[i]}", out / "PS-RGB" / Path(rgb_by_id[i]).name)
        download(f"{BUCKET}/{gj_keys[i]}", out / "geojson_buildings" / Path(gj_keys[i]).name)
    print(f"\nDone: {len(ids)} chips in {out}/  (PS-RGB/*.tif, geojson_buildings/*.geojson)")
    print("Next: python scripts/prepare_spacenet.py")


if __name__ == "__main__":
    sys.exit(main())

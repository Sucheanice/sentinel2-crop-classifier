# -*- coding: utf-8 -*-
import os, sys, time, argparse
import requests
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
OUT_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据6"
BBOX = [105.0, 30.1, 105.8, 31.2]
BANDS = ["B02", "B03", "B04", "B08"]
BAND_ASSET_MAP = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}
TILE = "48RWU"

SCENE_DATES = [
    "2025-05-20",
    "2025-06-26",
    "2025-07-16",
    "2025-08-03",
]

def search_scene(catalog, target_date):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=target_date,
        max_items=20,
    )
    for item in list(search.items()):
        if TILE in item.id:
            return item
    return None

def download_band(item, band, out_path, session):
    if os.path.exists(out_path):
        print(f"    {band}.tif exists, skip")
        return True

    asset_key = BAND_ASSET_MAP[band]
    href = item.assets[asset_key].href
    resp = session.get(href, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(out_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
    size_mb = downloaded / (1024 * 1024)
    print(f"    {band}.tif  {size_mb:.1f} MB  OK")
    return True

def main():
    t0 = time.time()
    session = requests.Session()

    print("Connecting to Earth Search STAC...")
    catalog = Client.open(STAC_URL)

    os.makedirs(OUT_DIR, exist_ok=True)

    for target_date in SCENE_DATES:
        print(f"\nSearching {target_date}...")
        item = search_scene(catalog, target_date)
        if item is None:
            print(f"  ERROR: No {TILE} scene found for {target_date}")
            continue

        cloud = item.properties.get("eo:cloud_cover", 0)
        scene_name = f"{target_date}_{TILE}_cloud{cloud:.1f}"
        scene_dir = os.path.join(OUT_DIR, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        print(f"  {item.id}  cloud={cloud:.2f}%")

        for band in BANDS:
            out_path = os.path.join(scene_dir, band + ".tif")
            try:
                download_band(item, band, out_path, session)
            except Exception as e:
                print(f"    {band}.tif FAILED: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {OUT_DIR}")

if __name__ == "__main__":
    main()

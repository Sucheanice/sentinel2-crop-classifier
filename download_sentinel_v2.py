# -*- coding: utf-8 -*-
import os, sys, time, argparse
import requests
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]  # v2: 增加红边波段
BANDS_ALL = BANDS_10M + BANDS_20M
BAND_ASSET_MAP = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B8A": "nir08", "B11": "swir16", "B12": "swir22",
}

PHENO_WINDOWS = [
    ("2025-05", "2025-05-01", "2025-06-01", "返青/拔节"),
    ("2025-06", "2025-06-01", "2025-07-01", "分蘖/抽穗"),
    ("2025-07", "2025-07-01", "2025-08-01", "抽穗/灌浆"),
    ("2025-08", "2025-08-01", "2025-09-01", "灌浆/收获"),
]

TILE_BBOX = {
    "48RWU": [105.0, 29.7, 106.2, 30.8],
    "48RWV": [105.0, 30.6, 106.2, 31.7],
}
DEFAULT_BBOX = [105.0, 29.7, 106.2, 31.7]


def search_best_in_window(catalog, start, end, tile, bbox):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="%s/%s" % (start, end),
        max_items=30,
    )
    best = None
    best_cloud = 999
    for item in search.items():
        if tile not in item.id:
            continue
        cloud = item.properties.get("eo:cloud_cover", 100)
        if cloud < best_cloud:
            best_cloud = cloud
            best = item
    return best, best_cloud


def download_band(item, band, out_path, session, max_retries=3):
    if os.path.exists(out_path):
        print(f"    {band}.tif exists, skip")
        return True
    asset_key = BAND_ASSET_MAP[band]
    href = item.assets[asset_key].href
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(href, stream=True, timeout=600)
            resp.raise_for_status()
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
            os.rename(tmp_path, out_path)
            size_mb = downloaded / (1024 * 1024)
            print(f"    {band}.tif  {size_mb:.1f} MB  OK")
            return True
        except Exception as e:
            print(f"    {band}.tif attempt {attempt}/{max_retries} FAILED: {e}")
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
            if attempt < max_retries:
                time.sleep(5 * attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Sentinel-2 L2A scenes")
    parser.add_argument("--tile", default="48RWV", help="MGRS tile (default: 48RWV)")
    parser.add_argument("--proxy", default=None, help="HTTPS proxy, e.g. http://127.0.0.1:33210")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: 待训练数据6)")
    parser.add_argument("--dates", default=None,
                        help="Comma-separated fixed dates (e.g. 2025-05-20,2025-06-26). Overrides window search.")
    parser.add_argument("--regions", default=None,
                        help="Comma-separated region names (anju,shehong). Uses tile(s) from TILE_BBOX.")
    args = parser.parse_args()

    t0 = time.time()
    session = requests.Session()
    if args.proxy:
        session.proxies = {"http": args.proxy, "https": args.proxy}

    if args.regions:
        region_tile_map = {"anju": "48RWU", "shehong": "48RWV"}
        tiles = [region_tile_map.get(r.strip(), r.strip()) for r in args.regions.split(",")]
    else:
        tiles = [args.tile]

    out_base = args.out_dir or os.path.join(BASE_DIR, "待训练数据6")

    print("Connecting to Earth Search STAC...")
    catalog = Client.open(STAC_URL)

    for tile in tiles:
        bbox = TILE_BBOX.get(tile, DEFAULT_BBOX)
        out_dir = out_base if len(tiles) == 1 else os.path.join(out_base, "s2_%s" % tile)
        os.makedirs(out_dir, exist_ok=True)
        print("\n=== Tile: %s, dir: %s ===" % (tile, out_dir))

        if args.dates:
            target_dates = [d.strip() for d in args.dates.split(",")]
            for target_date in target_dates:
                print("\nSearching %s..." % target_date)
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=target_date,
                    max_items=20,
                )
                item = None
                for it in list(search.items()):
                    if tile in it.id:
                        item = it
                        break
                if item is None:
                    print("  ERROR: No %s scene found for %s" % (tile, target_date))
                    continue
                cloud = item.properties.get("eo:cloud_cover", 0)
                scene_name = "%s_%s_cloud%.1f" % (target_date, tile, cloud)
                scene_dir = os.path.join(out_dir, scene_name)
                os.makedirs(scene_dir, exist_ok=True)
                print("  %s  cloud=%.2f%%" % (item.id, cloud))
                for band in BANDS_ALL:
                    out_path = os.path.join(scene_dir, band + ".tif")
                    try:
                        download_band(item, band, out_path, session)
                    except Exception as e:
                        print("    %s.tif FAILED: %s" % (band, e))
        else:
            for wname, wstart, wend, wdesc in PHENO_WINDOWS:
                print("\nSearching window %s (%s) ..." % (wname, wdesc))
                item, best_cloud = search_best_in_window(catalog, wstart, wend, tile, bbox)
                if item is None:
                    print("  NO SCENE FOUND in %s for %s" % (wname, tile))
                    continue
                scene_date = item.datetime.strftime("%Y-%m-%d")
                scene_name = "%s_%s_cloud%.1f" % (scene_date, tile, best_cloud)
                scene_dir = os.path.join(out_dir, scene_name)
                os.makedirs(scene_dir, exist_ok=True)
                print("  %s  date=%s  cloud=%.2f%%" % (item.id, scene_date, best_cloud))
                for band in BANDS_ALL:
                    out_path = os.path.join(scene_dir, band + ".tif")
                    try:
                        download_band(item, band, out_path, session)
                    except Exception as e:
                        print("    %s.tif FAILED: %s" % (band, e))

    elapsed = time.time() - t0
    print("\nDone in %.1fs" % elapsed)
    print("Output: %s" % out_dir)


if __name__ == "__main__":
    main()

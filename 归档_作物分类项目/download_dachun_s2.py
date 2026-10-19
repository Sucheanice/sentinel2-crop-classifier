# -*- coding: utf-8 -*-
"""Planetary Computer 版 — 江油大春 S2 并行下载 (Azure, 中国快40x)。

下载 2025-06/07/08 三期，每期 48SWA + 48SVA 两个 tile。
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import planetary_computer
import pystac_client
import requests

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "江油_s2")
os.makedirs(OUT_DIR, exist_ok=True)

DACHUN_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]
TILES = ["48SWA", "48SVA"]

BBOX = [104.9, 31.9, 105.2, 32.2]

WINDOWS = [
    ("2025-06-26", "2025-06-20", "2025-07-01"),
    ("2025-07-16", "2025-07-10", "2025-07-21"),
    ("2025-08-03", "2025-07-28", "2025-08-08"),
]


def download_one_band(args):
    href, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return (True, os.path.basename(out_path))
    tmp = out_path + ".tmp"
    for attempt in range(3):
        try:
            r = requests.get(href, timeout=600, stream=True)
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=262144):
                    f.write(chunk)
            os.rename(tmp, out_path)
            sz = os.path.getsize(out_path) / 1e6
            return (True, f"{os.path.basename(out_path)} ({sz:.0f}MB)")
        except Exception as e:
            time.sleep(3)
    return (False, f"{os.path.basename(out_path)}: FAIL")


def main():
    t0 = time.time()
    print("="*60)
    print("江油 大春 S2 下载 (Planetary Computer)")
    print(f"目标: 3期 × 2tile = 6场景")
    print("="*60)

    cat = pystac_client.Client.open(STAC_URL)
    total_ok, total_fail = 0, 0

    for target, start, end in WINDOWS:
        print(f"\n[{target}] 搜索...")
        search = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX, datetime=f"{start}/{end}", max_items=20,
        )

        # 按 tile 分组，找每 tile 云量最低的
        best_per_tile = {}
        for item in search.items():
            item_id = item.id
            parts = item_id.split('_')
            tile_parts = [p for p in parts if p.startswith('T') and len(p) == 6]
            tile_id = tile_parts[0][1:] if tile_parts else ''
            if tile_id not in TILES:
                continue
            cloud = item.properties.get("eo:cloud_cover", 100)
            if tile_id not in best_per_tile or cloud < best_per_tile[tile_id][0]:
                best_per_tile[tile_id] = (cloud, item)

        if not best_per_tile:
            print("  未找到任何场景!"); continue

        # 打印候选
        for ti in TILES:
            if ti in best_per_tile:
                c, _ = best_per_tile[ti]
                print(f"    {ti} cloud={c:.1f}%")
            else:
                print(f"    {ti} 未找到")

        # 逐个 tile 下载
        for ti in TILES:
            if ti not in best_per_tile:
                continue

            cloud, item = best_per_tile[ti]
            signed = planetary_computer.sign(item)
            dt = item.properties['datetime'][:10]
            scene_name = f"{dt}_{ti}"
            scene_dir = os.path.join(OUT_DIR, scene_name)

            existing = all(os.path.exists(os.path.join(scene_dir, f"{b}.tif"))
                           for b in DACHUN_BANDS)
            if existing:
                print(f"  -> {scene_name} 已完整, 跳过"); continue

            os.makedirs(scene_dir, exist_ok=True)
            print(f"  下载: {scene_name} cloud={cloud:.1f}%")

            tasks = []
            for band in DACHUN_BANDS:
                if band in signed.assets:
                    href = signed.assets[band].href
                    out = os.path.join(scene_dir, f"{band}.tif")
                    tasks.append((href, out))

            ok, fail = 0, 0
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {pool.submit(download_one_band, t): t for t in tasks}
                for f in as_completed(futures):
                    success, info = f.result()
                    if success:
                        ok += 1
                    else:
                        fail += 1
                    print(f"  [{ok+fail}/{len(tasks)}] {info}", flush=True)

            print(f"  -> {scene_name}: {ok}/{len(tasks)} OK")
            total_ok += ok
            total_fail += fail

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"完成: {elapsed:.0f}s, OK={total_ok}, FAIL={total_fail}")
    print(f"输出: {OUT_DIR}")


if __name__ == "__main__":
    main()

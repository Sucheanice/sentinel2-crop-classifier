# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 - v3: pystac_client搜索 + 手动SAS签名(加超时)"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pystac_client

WORK_DIR = r"E:\工作相关\2026年\0624 待测试数据"
OUT_DIR = os.path.join(WORK_DIR, "小春_s2_48RWU_summer")
os.makedirs(OUT_DIR, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]
TILE_BBOX = [105.0, 30.2, 106.2, 31.1]
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

TARGET_DATES = [
    ("2025-06-26", "S2A_MSIL2A_20250626T034201_R061_T48RWU_20250626T073921"),
    ("2025-07-26", "S2B_MSIL2A_20250726T032519_R018_T48RWU_20250726T055341"),
    ("2025-08-28", "S2B_MSIL2A_20250828T033539_R061_T48RWU_20250828T061956"),
]

sas_session = requests.Session()

def get_sas_token(href):
    """获取单个 URL 的 SAS token (带超时)"""
    try:
        # planetary_computer sign API
        r = sas_session.post(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a",
            json={"href": href},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()
        return result.get("href", result.get("token", ""))
    except Exception as e:
        print(f"  SAS获取失败: {e}", flush=True)
        return ""

def get_signed_hrefs(item):
    """为STAC item的所有波段获取signed URLs"""
    hrefs = {}
    for band in S2_BANDS:
        if band in item.assets:
            raw_href = item.assets[band].href
            signed = get_sas_token(raw_href)
            hrefs[band] = signed if signed else raw_href
    return hrefs

def download_one_band(args):
    href, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        sz = os.path.getsize(out_path) / 1e6
        return (True, f"{os.path.basename(out_path)} ({sz:.0f}MB) skip")
    tmp = out_path + ".tmp"
    for attempt in range(3):
        try:
            r = sas_session.get(href, timeout=600, stream=True)
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=262144):
                    f.write(chunk)
            os.rename(tmp, out_path)
            sz = os.path.getsize(out_path) / 1e6
            return (True, f"{os.path.basename(out_path)} ({sz:.0f}MB)")
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
    return (False, f"{os.path.basename(out_path)}: FAIL")

def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("遂宁 48RWU 夏季 S2 下载 v3", flush=True)
    print("=" * 60, flush=True)

    print("连接 STAC...", flush=True)
    cat = pystac_client.Client.open(STAC_URL)

    for date_str, scene_id in TARGET_DATES:
        scene_dir = os.path.join(OUT_DIR, f"{date_str}_48RWU")

        all_exist = all(
            os.path.exists(os.path.join(scene_dir, f"{b}.tif"))
            for b in S2_BANDS
        )
        if all_exist:
            print(f"\n[{date_str}] 已存在", flush=True)
            continue

        os.makedirs(scene_dir, exist_ok=True)

        print(f"\n[{date_str}] 搜索...", flush=True)
        search = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=TILE_BBOX,
            datetime=date_str,
            max_items=20,
        )
        item = None
        for it in search.items():
            if "T48RWU" in it.id:
                item = it
                break

        if item is None:
            print(f"  未找到 T48RWU 场景!", flush=True)
            continue

        cloud = item.properties.get("eo:cloud_cover", 0)
        print(f"  云量: {cloud:.1f}% ({item.id})", flush=True)

        # 获取 signed URLs
        print(f"  获取 SAS tokens...", flush=True)
        hrefs = get_signed_hrefs(item)
        if not hrefs:
            print(f"  签名失败!", flush=True)
            continue

        # 并行下载
        tasks = []
        for band in S2_BANDS:
            if band in hrefs:
                out = os.path.join(scene_dir, f"{band}.tif")
                tasks.append((hrefs[band], out))

        ok = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(download_one_band, t): t for t in tasks}
            for f in as_completed(futures):
                success, info = f.result()
                if success: ok += 1
                print(f"  [{ok}/{len(tasks)}] {info}", flush=True)

        print(f"  -> {date_str}: 完成", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"完成: {elapsed:.0f}s", flush=True)
    print(f"输出: {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()

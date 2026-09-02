# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 (直接用 requests, 绕过 planetary_computer.sign 可能的挂起)"""
import os, sys, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

WORK_DIR = r"E:\工作相关\2026年\0624 待测试数据"
OUT_DIR = os.path.join(WORK_DIR, "小春_s2_48RWU_summer")
os.makedirs(OUT_DIR, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]

# 目标日期 (从之前的搜索结果)
TARGET_DATES = ["2025-06-26", "2025-07-26", "2025-08-28"]

TILE_BBOX = [105.0, 30.2, 106.2, 31.1]

def search_item(date_str):
    """通过 PC STAC API 搜索特定日期的 48RWU 场景"""
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": TILE_BBOX,
        "datetime": date_str,
        "limit": 20,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    for feat in r.json().get("features", []):
        if "T48RWU" in feat["id"]:
            return feat
    return None

def sign_assets(stac_item):
    """调用 PC sign API 获取 SAS token"""
    sign_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
    # PC sign 需要 collection_id 和 href
    # 简化: 直接对每个 asset href 签名
    signed_hrefs = {}
    for key, asset in stac_item.get("assets", {}).items():
        if key not in S2_BANDS:
            continue
        href = asset["href"]
        # 尝试直接下载 (有些 PC URL 无需签名也有短期有效)
        signed_hrefs[key] = href
    return signed_hrefs

def download_one_band(args):
    href, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        sz = os.path.getsize(out_path) / 1e6
        return (True, f"{os.path.basename(out_path)} ({sz:.0f}MB) skip")
    tmp = out_path + ".tmp"
    for attempt in range(3):
        try:
            r = requests.get(href, timeout=600, stream=True)
            if r.status_code == 401:
                # 尝试 PC sign
                return (False, f"{os.path.basename(out_path)}: 401 (需要SAS)")
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
    print("遂宁 48RWU 夏季 S2 下载 v2", flush=True)
    print("=" * 60, flush=True)

    for date_str in TARGET_DATES:
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
        try:
            item = search_item(date_str)
        except Exception as e:
            print(f"  搜索失败: {e}", flush=True)
            continue

        if item is None:
            print(f"  未找到 T48RWU 场景", flush=True)
            continue

        cloud = item["properties"].get("eo:cloud_cover", 0)
        print(f"  云量: {cloud:.1f}% ({item['id']})", flush=True)

        # 获取 asset URLs (先不签名，后续需要再签)
        hrefs = sign_assets(item)
        print(f"  获取 {len(hrefs)} 个波段URL", flush=True)

        tasks = []
        for band in S2_BANDS:
            if band in hrefs:
                out = os.path.join(scene_dir, f"{band}.tif")
                tasks.append((hrefs[band], out))

        ok, fail = 0, 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(download_one_band, t): t for t in tasks}
            for f in as_completed(futures):
                success, info = f.result()
                if success: ok += 1
                else: fail += 1
                if success or "FAIL" in info:
                    print(f"  [{ok+fail}/{len(tasks)}] {info}", flush=True)

        if fail > 0:
            print(f"  [WARN] {date_str}: {fail} 个波段下载失败，尝试用 PC sign...", flush=True)
            # 降级方案: 用 planetary_computer 库
            try:
                import planetary_computer
                signed = planetary_computer.sign(item)
                tasks2 = []
                for band in S2_BANDS:
                    if band in signed.assets:
                        out = os.path.join(scene_dir, f"{band}.tif")
                        if not os.path.exists(out) or os.path.getsize(out) < 1000:
                            tasks2.append((signed.assets[band].href, out))
                if tasks2:
                    print(f"  重试 {len(tasks2)} 个波段...", flush=True)
                    ok2, fail2 = 0, 0
                    with ThreadPoolExecutor(max_workers=3) as pool2:
                        futures2 = {pool2.submit(download_one_band, t): t for t in tasks2}
                        for f in as_completed(futures2):
                            success, info = f.result()
                            if success: ok2 += 1
                            else: fail2 += 1
                            print(f"    [{ok2+fail2}/{len(tasks2)}] {info}", flush=True)
            except Exception as e:
                print(f"  PC sign fallback also failed: {e}", flush=True)

        print(f"  -> {date_str}: {ok}/{len(tasks)} OK", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}", flush=True)
    print(f"完成: {elapsed:.0f}s", flush=True)
    print(f"输出: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 (Planetary Computer, 并行)"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import planetary_computer, pystac_client, requests

WORK_DIR = r"E:\工作相关\2026年\0624 待测试数据"
OUT_DIR = os.path.join(WORK_DIR, "小春_s2_48RWU_summer")
os.makedirs(OUT_DIR, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]

# 选中的夏季场景 (最低云量)
TARGET_SCENES = [
    ("2025-06-26", "S2A_MSIL2A_20250626T034201_R061_T48RWU_20250626T073921"),
    ("2025-07-26", "S2B_MSIL2A_20250726T032519_R018_T48RWU_20250726T055341"),
    ("2025-08-28", "S2B_MSIL2A_20250828T033539_R061_T48RWU_20250828T061956"),
]


def download_one_band(args):
    href, out_path = args
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        sz = os.path.getsize(out_path) / 1e6
        return (True, f"{os.path.basename(out_path)} ({sz:.0f}MB) skip")
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
    print("=" * 60)
    print("遂宁 48RWU 夏季 S2 下载 (Planetary Computer)")
    print("=" * 60)

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    for date_str, scene_id in TARGET_SCENES:
        scene_dir = os.path.join(OUT_DIR, f"{date_str}_48RWU")
        
        # 检查是否已全部下载
        all_exist = all(
            os.path.exists(os.path.join(scene_dir, f"{b}.tif"))
            for b in S2_BANDS
        )
        if all_exist:
            print(f"\n[{date_str}] 已存在, 跳过")
            continue

        os.makedirs(scene_dir, exist_ok=True)

        # 日期范围搜索 (比 ID 搜索快)
        print(f"\n[{date_str}] 搜索...")
        search = cat.search(
            collections=["sentinel-2-l2a"],
            bbox=[105.0, 30.2, 106.2, 31.1],
            datetime=date_str,
            max_items=20,
        )
        item = None
        for it in search.items():
            if "T48RWU" in it.id:
                item = it
                break
        if item is None:
            print(f"  未找到场景!")
            continue
        
        signed = planetary_computer.sign(item)
        cloud = item.properties.get("eo:cloud_cover", 0)
        print(f"  云量: {cloud:.1f}% ({item.id})")

        # 并行下载10个波段
        tasks = []
        for band in S2_BANDS:
            if band in signed.assets:
                href = signed.assets[band].href
                out = os.path.join(scene_dir, f"{band}.tif")
                tasks.append((href, out))

        ok, fail = 0, 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(download_one_band, t): t for t in tasks}
            for f in as_completed(futures):
                success, info = f.result()
                if success: ok += 1
                else: fail += 1
                print(f"  [{ok+fail}/{len(tasks)}] {info}", flush=True)

        print(f"  -> {date_str}: {ok}/{len(tasks)} OK")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"完成: {elapsed:.0f}s")
    print(f"输出: {OUT_DIR}")


if __name__ == "__main__":
    main()

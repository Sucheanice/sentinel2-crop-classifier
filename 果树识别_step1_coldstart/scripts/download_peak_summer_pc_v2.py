# -*- coding: utf-8 -*-
"""从 Planetary Computer 下载遂宁 48RWU 夏季全瓦片 - v2: requests直接下载, 不用rasterio读远程"""
import os, sys, time
from pathlib import Path
import planetary_computer, pystac_client, requests

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s2_48RWU_summer" / "2025-06-26_48RWU"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("遂宁 48RWU 夏季全瓦片下载 (PC + requests直连)")
    log("=" * 60)

    # 搜索场景
    log("搜索 2025-06-26 48RWU...")
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = cat.search(
        collections=["sentinel-2-l2a"],
        bbox=[105.0, 30.2, 106.2, 31.1],
        datetime="2025-06-26",
        max_items=20,
    )
    item = None
    for it in search.items():
        if "T48RWU" in it.id:
            item = it
            break

    if item is None:
        log("[FATAL] 未找到场景")
        return

    cloud = item.properties.get("eo:cloud_cover", "?")
    log(f"找到: {item.id}, 云量: {cloud}%")

    # sign
    log("获取 SAS token...")
    signed = planetary_computer.sign(item)

    # 逐个波段下载
    total_mb = 0
    for band in S2_BANDS:
        out_path = OUT_DIR / f"{band}.tif"
        if out_path.exists():
            sz_mb = out_path.stat().st_size / 1e6
            log(f"  {band}.tif 已存在 ({sz_mb:.0f}MB)")
            total_mb += sz_mb
            continue

        if band not in signed.assets:
            log(f"  [WARN] {band} 不在assets中")
            continue

        href = signed.assets[band].href
        log(f"  下载 {band}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = requests.get(href, timeout=600, stream=True)
                r.raise_for_status()
                tmp = str(out_path) + ".tmp"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=262144):
                        f.write(chunk)
                os.rename(tmp, str(out_path))
                sz_mb = out_path.stat().st_size / 1e6
                total_mb += sz_mb
                log(f"  {band}.tif ({sz_mb:.0f}MB) OK")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    log(f"  {band} 失败, {wait_s}s后重试: {str(e)[:50]}")
                    time.sleep(wait_s)
                else:
                    log(f"  [WARN] {band} 最终失败: {str(e)[:60]}")

    elapsed = time.time() - t0
    log(f"\n完成: {elapsed/60:.1f}分, 总 {total_mb:.0f}MB")
    log(f"输出: {OUT_DIR}")

if __name__ == "__main__":
    main()

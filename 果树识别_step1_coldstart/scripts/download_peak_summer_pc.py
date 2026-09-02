# -*- coding: utf-8 -*-
"""从 Planetary Computer 下载遂宁 48RWU 夏季全瓦片 (COG格式, 不裁剪)"""
import os, sys, time
from pathlib import Path
import planetary_computer, pystac_client, requests, rasterio

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s2_48RWU_summer" / "2025-06-26_48RWU"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("遂宁 48RWU 夏季全瓦片下载 (Planetary Computer)")
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
        log("[FATAL] 未找到 48RWU 场景")
        return

    log(f"找到: {item.id}, 云量: {item.properties.get('eo:cloud_cover', '?')}%")

    # sign
    log("获取 SAS token...")
    signed = planetary_computer.sign(item)

    # 下载每个波段 (不裁剪, 保留原始COG)
    for band in S2_BANDS:
        out_path = OUT_DIR / f"{band}.tif"
        if out_path.exists():
            sz = out_path.stat().st_size / 1e6
            log(f"  {band}.tif 已存在 ({sz:.0f}MB)")
            continue

        if band not in signed.assets:
            log(f"  [WARN] {band} 不在 assets 中")
            continue

        href = signed.assets[band].href
        log(f"  下载 {band}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                                  GDAL_HTTP_TIMEOUT="300"):
                    with rasterio.open(href) as src:
                        data = src.read(1).astype("float32")
                        profile = src.profile.copy()
                        profile.update(driver="GTiff", dtype="float32",
                                       compress="deflate", tiled=True,
                                       blockxsize=256, blockysize=256)
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(data, 1)

                # 验证
                with rasterio.open(out_path) as chk:
                    d = chk.read(1)
                    valid_pct = (d > 0).sum() / d.size * 100
                sz = out_path.stat().st_size / 1e6
                log(f"  {band}.tif ({sz:.0f}MB, valid={valid_pct:.1f}%) OK")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    log(f"  {band} 失败, {wait_s}s后重试: {str(e)[:50]}")
                    time.sleep(wait_s)
                else:
                    log(f"  [WARN] {band} 最终失败: {str(e)[:60]}")

    elapsed = time.time() - t0
    log(f"\n完成: {elapsed/60:.1f}分")

if __name__ == "__main__":
    main()

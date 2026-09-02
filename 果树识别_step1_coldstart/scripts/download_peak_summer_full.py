# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 全瓦片 (不裁剪, 仅 peak_summer)"""
import os, sys, time, shutil, tempfile
from pathlib import Path
import rasterio

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s2_48RWU_summer" / "2025-06-26_48RWU"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
S2_RES10 = {"B02","B03","B04","B08"}

# peak_summer 场景
SCENE_ID = "S2A_MSIL2A_20250626T034201_R061_T48RWU_20250626T073921"
DATE = "2025-06-26"

def build_s3_urls(scene_id):
    parts = scene_id.split("_")
    utm_zone = parts[4][1:3]    # 48
    lat_band = parts[4][3]      # R
    square = parts[4][4:6]      # WU
    date_part = parts[2][:8]    # 20250626
    year = date_part[:4]
    month = str(int(date_part[4:6]))
    day = str(int(date_part[6:8]))
    seq = "0"
    base = f"https://sentinel-s2-l2a.s3.eu-central-1.amazonaws.com/tiles/{utm_zone}/{lat_band}/{square}/{year}/{month}/{day}/{seq}"
    urls = {}
    for band in S2_BANDS:
        if band in S2_RES10:
            urls[band] = f"{base}/R10m/{band}.jp2"
        else:
            urls[band] = f"{base}/R20m/{band}.jp2"
    return urls

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log(f"遂宁 48RWU 全瓦片下载: {DATE}")
    log("=" * 60)

    # 检查是否已全部存在
    all_exist = all((OUT_DIR / f"{b}.tif").exists() for b in S2_BANDS)
    if all_exist:
        log("全部波段已存在, 跳过")
        return

    band_urls = build_s3_urls(SCENE_ID)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sn_full_"))
    
    import requests
    session = requests.Session()

    for band in S2_BANDS:
        url = band_urls.get(band)
        out_path = OUT_DIR / f"{band}.tif"
        if out_path.exists():
            log(f"  {band}.tif 已存在")
            continue

        local_jp2 = tmp_dir / f"{band}.jp2"
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if not local_jp2.exists() or local_jp2.stat().st_size < 1000:
                    r = session.get(url, timeout=300, stream=True)
                    r.raise_for_status()
                    with open(local_jp2, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                    sz_mb = local_jp2.stat().st_size / 1e6
                else:
                    sz_mb = local_jp2.stat().st_size / 1e6

                # 读取 JP2, 转写为 GeoTIFF (保持原始投影和范围, 不裁剪!)
                with rasterio.open(local_jp2) as src:
                    data = src.read(1).astype("float32")
                    profile = src.profile.copy()
                    profile.update(driver="GTiff", dtype="float32",
                                   compress="deflate", tiled=True,
                                   blockxsize=256, blockysize=256)
                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(data, 1)

                log(f"  {band}.tif ({sz_mb:.1f}MB) OK")
                if local_jp2.exists():
                    local_jp2.unlink()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    log(f"  {band} 失败, {wait_s}s后重试...")
                    time.sleep(wait_s)
                    if local_jp2.exists():
                        local_jp2.unlink()
                else:
                    log(f"  [WARN] {band} 最终失败: {str(e)[:60]}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    log(f"\n完成: {elapsed/60:.1f}分, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()

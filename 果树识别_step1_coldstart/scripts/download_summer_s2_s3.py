# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 - v4: AWS S3 直连, 无需 SAS token
较慢(~14KB/s) 但可靠, 每场景约20-30分钟
"""
import os, sys, time, math, shutil, tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_DIR = WORK_DIR / "小春_s2_48RWU_summer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
S2_RES10 = {"B02","B03","B04","B08"}
S2_RES20 = {"B05","B06","B07","B8A","B11","B12"}

SUINING_BBOX_WGS = [105.00, 30.15, 106.05, 31.10]  # 与重建脚本一致

# 夏季场景 (预定义, 从之前的PC搜索获取)
SUMMER_SCENES = {
    "peak_summer": {
        "date": "2025-06-26",
        "id": "S2A_MSIL2A_20250626T034201_R061_T48RWU_20250626T073921",
    },
    "mid_summer": {
        "date": "2025-07-26",
        "id": "S2B_MSIL2A_20250726T032519_R018_T48RWU_20250726T055341",
    },
    "late_summer": {
        "date": "2025-08-28",
        "id": "S2B_MSIL2A_20250828T033539_R061_T48RWU_20250828T061956",
    },
}

def build_s3_urls(scene_id, date_str):
    """构建 AWS S3 公开 URL (与 run_transfer_learning.py 一致)"""
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

def download_and_clip_season(season, info):
    """下载一季的10波段, 裁剪到遂宁区域, 写出多波段GeoTIFF"""
    scene_id = info["id"]
    date_str = info["date"]

    # 输出裁剪后的多波段GeoTIFF
    out_path = OUT_DIR / f"{season}_10m_10band.tif"
    if out_path.exists():
        log(f"  [{season}] 已缓存")
        return out_path

    # 计算目标网格
    bbox_wgs = list(SUINING_BBOX_WGS)
    dst_crs = "EPSG:32648"
    bbox_utm = transform_bounds("EPSG:4326", dst_crs, *bbox_wgs)
    res = 10.0
    xmin = math.floor(bbox_utm[0] / res) * res
    ymax = math.ceil(bbox_utm[3] / res) * res
    xmax = math.ceil(bbox_utm[2] / res) * res
    ymin = math.floor(bbox_utm[1] / res) * res
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)
    dst_transform = rasterio.transform.from_origin(xmin, ymax, res, res)

    log(f"  [{season}] {date_str} 下载10波段 (AWS S3)...")
    band_urls = build_s3_urls(scene_id, date_str)
    tmp_dir = Path(tempfile.mkdtemp(prefix="sn_summer_"))
    bands_data = []

    import requests
    session = requests.Session()

    for band in S2_BANDS:
        url = band_urls.get(band)
        if not url:
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

                # 本地读取+重投影裁剪
                with rasterio.open(local_jp2) as src:
                    band_arr = np.zeros((height, width), dtype="float32")
                    reproject(
                        source=src.read(1),
                        destination=band_arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
                bands_data.append(band_arr)
                log(f"    {band}.jp2 {sz_mb:.1f}MB OK")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    log(f"    {band} 失败, {wait_s}s后重试...")
                    time.sleep(wait_s)
                    if local_jp2.exists():
                        local_jp2.unlink()
                else:
                    log(f"    [WARN] {band} 最终失败: {str(e)[:60]}")

    # 清理
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(bands_data) != len(S2_BANDS):
        log(f"  [WARN] {season} 仅获取 {len(bands_data)}/{len(S2_BANDS)} 波段")
        if len(bands_data) < 5:
            return None

    # 写出多波段GeoTIFF
    raster = np.stack(bands_data, axis=0).astype("float32")
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": len(bands_data), "dtype": "float32",
        "crs": dst_crs, "transform": dst_transform,
        "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster)
        for i, b in enumerate(S2_BANDS[:len(bands_data)]):
            dst.set_band_description(i + 1, b)
    log(f"  -> {out_path.name} ({raster.nbytes/1e6:.0f} MB)")
    return out_path


def main():
    t0 = time.time()
    log("=" * 60)
    log("遂宁 48RWU 夏季 S2 下载 (AWS S3)")
    log("=" * 60)

    results = {}
    for season, info in SUMMER_SCENES.items():
        log(f"\n--- {season} ---")
        out = download_and_clip_season(season, info)
        if out:
            results[season] = str(out)

    elapsed = time.time() - t0
    log(f"\n{'=' * 60}")
    log(f"完成: {elapsed/60:.1f}分")
    log(f"获取 {len(results)}/3 季")
    for s, p in results.items():
        log(f"  {s}: {p}")


if __name__ == "__main__":
    main()

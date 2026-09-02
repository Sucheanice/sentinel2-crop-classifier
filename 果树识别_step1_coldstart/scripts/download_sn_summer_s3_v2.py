# -*- coding: utf-8 -*-
"""下载遂宁 48RWU 夏季 S2 - 与叙永下载逻辑一致 (S3 JP2 + reproject到目标网格)"""
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

SUINING_BBOX_WGS = [105.00, 30.15, 106.05, 31.10]  # 与 rebuild_suining_cube.py 一致

# 用 mid_summer (7月, 云2.2%) 替代 peak_summer, JP2 解码有效像素多50倍
SCENE = {
    "scene_id": "S2B_MSIL2A_20250726T032519_R018_T48RWU_20250726T055341",
    "date": "2025-07-26",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_s3_urls(scene_id, date_str):
    """与 run_transfer_learning.py 的 _build_s3_urls 完全一致"""
    parts = scene_id.split("_")
    utm_zone = parts[4][1:3]    # 48
    lat_band = parts[4][3]      # R
    square = parts[4][4:6]      # WU
    date_part = parts[2][:8]
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


def compute_target_grid():
    """与 rebuild_suining_cube.py 的 compute_target_grid 完全一致"""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    bbox = SUINING_BBOX_WGS
    xmin, ymin = t.transform(bbox[0], bbox[1])
    xmax, ymax = t.transform(bbox[2], bbox[3])
    res = 10.0
    xmin = math.floor(xmin / res) * res
    ymax = math.ceil(ymax / res) * res
    xmax = math.ceil(xmax / res) * res
    ymin = math.floor(ymin / res) * res
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)
    return width, height, transform


def main():
    t0 = time.time()
    log("=" * 60)
    log("遂宁 48RWU 夏季 S2 下载 (S3 JP2, 与叙永逻辑一致)")
    log("=" * 60)

    scene_id = SCENE["scene_id"]
    date_str = SCENE["date"]
    out_path = OUT_DIR / "summer_10m_10band.tif"

    if out_path.exists():
        log(f"已存在: {out_path.name}")
        return

    # 计算目标网格 (与 rebuild_suining_cube.py 一致)
    width, height, dst_transform = compute_target_grid()
    dst_crs = "EPSG:32648"
    log(f"目标网格: {width}x{height} @ 10m")

    # 构建 S3 URL
    band_urls = build_s3_urls(scene_id, date_str)
    log(f"场景: {scene_id}")
    log(f"日期: {date_str}")

    # 下载并重投影
    tmp_dir = Path(tempfile.mkdtemp(prefix="sn_peak_"))
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
                valid_pct = (band_arr > 0).sum() / band_arr.size * 100
                bands_data.append(band_arr)
                log(f"  {band}.jp2 ({sz_mb:.1f}MB) -> valid={valid_pct:.1f}% OK")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_s = 2 ** attempt
                    log(f"  {band} 重试 {attempt+1}/{max_retries}...")
                    time.sleep(wait_s)
                    if local_jp2.exists():
                        local_jp2.unlink()
                else:
                    log(f"  [WARN] {band} 失败: {str(e)[:60]}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(bands_data) != len(S2_BANDS):
        log(f"[ERROR] 仅获取 {len(bands_data)}/{len(S2_BANDS)} 波段")
        return

    # 写出多波段 GeoTIFF
    raster = np.stack(bands_data, axis=0).astype("float32")
    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": len(S2_BANDS), "dtype": "float32",
        "crs": dst_crs, "transform": dst_transform,
        "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster)
        for i, b in enumerate(S2_BANDS):
            dst.set_band_description(i + 1, b)

    elapsed = time.time() - t0
    log(f"\n-> {out_path.name} ({raster.nbytes/1e6:.0f}MB), {elapsed/60:.1f}分")
    log("完成!")


if __name__ == "__main__":
    main()

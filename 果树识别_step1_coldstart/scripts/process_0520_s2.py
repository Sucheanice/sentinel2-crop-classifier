# -*- coding: utf-8 -*-
"""处理 5/20 遂宁 S2：合并本地 4 个 10m 波段 + 迅雷下载的 6 个 20m 波段 → 多波段 GeoTIFF"""
import time, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

# 5/20 已有的 4 个 10m 波段 (完整瓦片, 大春识别下载)
S2_10M_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据\待训练数据6\2025-05-20_48RWU_cloud1.4")
# 5/20 迅雷下载的 6 个 20m 波段
S2_20M_DIR = Path(r"E:\迅雷下载\08131420")

OUT_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据\小春_s2_48RWU_summer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

S2_BANDS_10M = ["B02", "B03", "B04", "B08"]
S2_BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
S2_BANDS_ALL = S2_BANDS_10M + S2_BANDS_20M

RES = 10.0
SUINING_BBOX_WGS = [105.00, 30.15, 106.05, 31.10]


def compute_target_grid():
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    xmin, ymin = t.transform(SUINING_BBOX_WGS[0], SUINING_BBOX_WGS[1])
    xmax, ymax = t.transform(SUINING_BBOX_WGS[2], SUINING_BBOX_WGS[3])
    xmin = math.floor(xmin / RES) * RES
    ymax = math.ceil(ymax / RES) * RES
    xmax = math.ceil(xmax / RES) * RES
    ymin = math.floor(ymin / RES) * RES
    width = int((xmax - xmin) / RES)
    height = int((ymax - ymin) / RES)
    return width, height, from_origin(xmin, ymax, RES, RES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("=" * 60)
    log("处理 5/20 遂宁 S2 → 多波段 GeoTIFF")
    log("=" * 60)

    width, height, dst_transform = compute_target_grid()
    log(f"目标网格: {width}x{height} @ 10m")

    bands_data = []

    # 10m 波段 (从 tif 读)
    for band in S2_BANDS_10M:
        p = S2_10M_DIR / f"{band}.tif"
        if not p.exists():
            log(f"[FATAL] {p} 不存在")
            return
        log(f"处理 10m {band}.tif...")
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            data[data == 0] = np.nan
            dst = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=data, destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs="EPSG:32648",
                resampling=Resampling.bilinear,
            )
        valid_pct = (~np.isnan(dst)).sum() / dst.size * 100
        log(f"  -> valid={valid_pct:.1f}%")
        bands_data.append(dst)

    # 20m 波段 (从 jp2 读, 重采样到 10m)
    for band in S2_BANDS_20M:
        p = S2_20M_DIR / f"{band}.jp2"
        if not p.exists():
            log(f"[FATAL] {p} 不存在")
            return
        log(f"处理 20m {band}.jp2...")
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            data[data == 0] = np.nan
            dst = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=data, destination=dst,
                src_transform=src.transform, src_crs="EPSG:32648",
                dst_transform=dst_transform, dst_crs="EPSG:32648",
                resampling=Resampling.bilinear,
            )
        valid_pct = (~np.isnan(dst)).sum() / dst.size * 100
        log(f"  -> valid={valid_pct:.1f}%")
        bands_data.append(dst)

    valid_all = np.all([~np.isnan(b) for b in bands_data], axis=0).sum()
    log(f"全波段共同有效: {valid_all}/{width*height} ({valid_all/(width*height)*100:.1f}%)")

    # 写出多波段 GeoTIFF
    raster = np.stack(bands_data, axis=0).astype("float32")
    raster = np.nan_to_num(raster, nan=-9999.0)
    out_path = OUT_DIR / "late_spring_10m_10band.tif"

    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": len(S2_BANDS_ALL), "dtype": "float32",
        "crs": "EPSG:32648", "transform": dst_transform,
        "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "nodata": -9999.0,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster)
        for i, b in enumerate(S2_BANDS_ALL):
            dst.set_band_description(i + 1, b)

    log(f"\n-> {out_path.name} ({raster.nbytes/1e6:.0f}MB), {(time.time()-t0)/60:.1f}分")
    log("完成!")


if __name__ == "__main__":
    main()

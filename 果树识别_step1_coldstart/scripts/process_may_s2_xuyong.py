# -*- coding: utf-8 -*-
"""处理叙永 5月 S2 (2025-05-12) → 10m 10波段 GeoTIFF，对齐叙永缓存网格

与 data/xuyong_s2/*_10m_10band.tif 完全一致:
  - 网格 2963x1232 @10m, EPSG:32648, 原点 (536440, 3075230)
  - 波段顺序 B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (与 S2_BANDS 一致)
"""
import os
import time
from pathlib import Path

# PROJ_LIB 必须在 CRS 操作前指向 rasterio 自带 proj 数据，否则命中 PostgreSQL 的旧 proj.db 报错
os.environ["PROJ_LIB"] = r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data"

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

JP2_DIR = Path(r"E:\迅雷下载\08131435-叙永3")
OUT_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据\果树识别_step1_coldstart\data\xuyong_s2")

S2_BANDS_ALL = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

RES = 10.0
XMIN, YMAX = 536440.0, 3075230.0
WIDTH, HEIGHT = 2963, 1232
NODATA = -9999.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("=" * 60)
    log("处理叙永 5月 S2 (2025-05-12) → late_spring 10波段")
    log("=" * 60)

    dst_transform = from_origin(XMIN, YMAX, RES, RES)
    log(f"目标网格: {WIDTH}x{HEIGHT} @10m, 原点 ({XMIN}, {YMAX})")

    bands_data = []
    for band in S2_BANDS_ALL:
        p = JP2_DIR / f"{band}.jp2"
        if not p.exists():
            log(f"[FATAL] {p} 不存在")
            return
        log(f"处理 {band}.jp2 ...")
        with rasterio.open(p) as src:
            data = src.read(1).astype("float32")
            data[data == 0] = np.nan
            dst = np.full((HEIGHT, WIDTH), np.nan, dtype="float32")
            reproject(
                source=data, destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs="EPSG:32648",
                resampling=Resampling.bilinear,
            )
        valid_pct = (~np.isnan(dst)).sum() / dst.size * 100
        log(f"  -> valid={valid_pct:.1f}%")
        bands_data.append(dst)

    valid_all = np.all([~np.isnan(b) for b in bands_data], axis=0).sum()
    log(f"全波段共同有效: {valid_all}/{WIDTH * HEIGHT} ({valid_all / (WIDTH * HEIGHT) * 100:.1f}%)")

    raster = np.stack(bands_data, axis=0).astype("float32")
    raster = np.nan_to_num(raster, nan=NODATA)
    out_path = OUT_DIR / "late_spring_10m_10band.tif"
    profile = {
        "driver": "GTiff", "height": HEIGHT, "width": WIDTH,
        "count": len(S2_BANDS_ALL), "dtype": "float32",
        "crs": "EPSG:32648", "transform": dst_transform,
        "nodata": NODATA, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster)
        for i, b in enumerate(S2_BANDS_ALL):
            dst.set_band_description(i + 1, b)
    log(f"输出: {out_path}  ({raster.nbytes / 1e6:.0f} MB)")
    log(f"完成, 耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""从 Planetary Computer 下载遂宁 48RWU mid_summer COG - 超长超时版"""
import os, sys, time
from pathlib import Path
import planetary_computer
import pystac_client
import requests

WORK_DIR = Path(r"E:\工作相关\2026年\0624 待测试数据")
OUT_PATH = WORK_DIR / "小春_s2_48RWU_summer" / "summer_10m_10band.tif"

S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 60)
log("遂宁 48RWU mid_summer COG下载 (PC, 超长超时)")
log("=" * 60)

# 搜索场景
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
items = list(cat.search(
    collections=["sentinel-2-l2a"],
    bbox=[105, 30.2, 106.2, 31.1],
    datetime="2025-07-26",
    max_items=20,
).items())

item = None
for it in items:
    if "T48RWU" in it.id:
        item = it
        break
if item is None:
    log("[FATAL] 未找到场景")
    sys.exit(1)

cloud = item.properties.get("eo:cloud_cover", "?")
log(f"场景: {item.id}, 云量: {cloud}%")

# Sign
signed = planetary_computer.sign(item)
log("SAS token 获取成功")

# 逐个波段下载
import rasterio, numpy as np
bands_data = []

for band in S2_BANDS:
    if band not in signed.assets:
        log(f"  [{band}] 不在assets中")
        continue

    href = signed.assets[band].href
    log(f"  [{band}] 下载...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="600",
                GDAL_HTTP_MAX_RETRY="3",
                CPL_VSIL_CURL_USE_HEAD="NO",
            ):
                with rasterio.open(href) as src:
                    data = src.read(1).astype("float32")
                    valid_pct = (data > 0).sum() / data.size * 100
                    elapsed = time.time() - t0
                    log(f"    {elapsed:.0f}s, {data.shape}, valid={valid_pct:.1f}%")
                    bands_data.append(data)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"    重试 {attempt+1}/{max_retries}: {str(e)[:50]}")
                time.sleep(5)
            else:
                log(f"    [FAIL] {str(e)[:60]}")

if len(bands_data) < 5:
    log(f"[FATAL] 仅 {len(bands_data)} 波段")
    sys.exit(1)

# 写出多波段 GeoTIFF
raster = np.stack(bands_data, axis=0).astype("float32")
first = bands_data[0]
H, W = first.shape

profile = {
    "driver": "GTiff", "height": H, "width": W,
    "count": len(bands_data), "dtype": "float32",
    "crs": "EPSG:32648",
    "transform": rasterio.open(href).transform if 'href' in dir() else None,
    "compress": "deflate", "tiled": True,
    "blockxsize": 256, "blockysize": 256,
}

# 从第一个波段获取transform
first_href = signed.assets[S2_BANDS[0]].href
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="600"):
    with rasterio.open(first_href) as src:
        profile["transform"] = src.transform
        profile["crs"] = src.crs

with rasterio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(raster)

log(f"\n完成 -> {OUT_PATH.name} ({raster.nbytes/1e6:.0f}MB)")

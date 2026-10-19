# -*- coding: utf-8 -*-
"""gee_download_pingwu_rgb_local.py — 分块 getDownloadURL 直下本地干净 RGB

绕开 Google Drive：把 bbox 切成小块，逐块 getDownloadURL 下载，
本地 rasterio merge 拼接，并裁剪到乡镇边界，直接落到本地目录。
"""
import ee
import time
import os
import math
import requests
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask

ee.Initialize()

BASE = os.path.dirname(os.path.abspath(__file__))
# 官方边界（CGCS2000/EPSG:4523），GEE 调用前需 to_crs(4326)
BOUNDARY = os.path.join(BASE, '平武_水晶镇_土城藏族乡_界.shp')
OUT_DIR = os.path.join(BASE, '江油_composites')
TMP = os.path.join(BASE, '_tmp_pingwu_rgb')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

TILE_DEG = 0.05  # 每块约 0.05°（~4.7km），保证单块 median 不超内存
SCALE = 10


def mask_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def rgb_median(start, end, roi):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(roi).filterDate(start, end).map(mask_clouds))
    med = col.select(['B4', 'B3', 'B2']).median().rename(['red', 'green', 'blue'])
    rgb = (med.divide(10000).multiply(3.0).clamp(0, 1)
           .multiply(255).round().uint8())
    return rgb


def download_tile(img, roi, path):
    params = {
        'scale': SCALE,
        'crs': 'EPSG:4326',
        'region': roi,
        'format': 'GEO_TIFF',
        'bands': ['red', 'green', 'blue'],
    }
    url = img.getDownloadURL(params)
    last_err = None
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=600)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return True
            last_err = f'HTTP {r.status_code} size={len(r.content)}'
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(5)
    print(f'    [FAIL] {os.path.basename(path)}: {last_err}')
    return False


def download_period(tag, start, end, roi, tiles):
    tile_files = []
    total = len(tiles)
    for i, t in enumerate(tiles):
        path = os.path.join(TMP, f'{tag}_{i:03d}.tif')
        if os.path.exists(path):
            tile_files.append(path)
            continue
        img = rgb_median(start, end, t)
        ok = download_tile(img, t, path)
        if ok:
            tile_files.append(path)
        print(f'  [{tag}] {i + 1}/{total} ok={ok}')
    return tile_files


def mosaic_and_clip(tag, tile_files):
    srcs = [rasterio.open(p) for p in tile_files]
    mosaic, out_trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans,
        'count': 3,
        'dtype': 'uint8',
        'nodata': 0,
        'driver': 'GTiff',
        'compress': 'lzw',
    })
    full_path = os.path.join(TMP, f'{tag}_full.tif')
    with rasterio.open(full_path, 'w', **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()

    # 裁剪到乡镇边界
    boundary = gpd.read_file(BOUNDARY).to_crs('EPSG:4326')
    with rasterio.open(full_path) as src:
        out_img, out_trans2 = mask(src, boundary.geometry, crop=True, nodata=0)
        meta2 = src.meta.copy()
        meta2.update({
            'height': out_img.shape[1],
            'width': out_img.shape[2],
            'transform': out_trans2,
            'nodata': 0,
        })
    out_path = os.path.join(OUT_DIR, f'平武_水晶镇_土城藏族乡_{tag}_RGB.tif')
    with rasterio.open(out_path, 'w', **meta2) as dst:
        dst.write(out_img)
    print(f'[done] {out_path} {out_img.shape[2]}x{out_img.shape[1]}')
    return out_path


def main():
    gdf = gpd.read_file(BOUNDARY)
    minx, miny, maxx, maxy = gdf.to_crs(4326).total_bounds

    # 生成网格
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append(ee.Geometry.Rectangle(
                [x, y, min(x + TILE_DEG, maxx), min(y + TILE_DEG, maxy)]))
            y += TILE_DEG
        x += TILE_DEG
    print(f'[grid] {len(tiles)} tiles, bbox=({minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f})')

    for tag, start, end in [
        ('summer', '2025-06-01', '2025-09-01'),
        ('winter', '2025-12-01', '2026-03-01'),
    ]:
        print(f'[download] {tag} ({start} ~ {end})')
        files = download_period(tag, start, end, None, tiles)
        if not files:
            print(f'[error] {tag} 没有可用 tile')
            continue
        mosaic_and_clip(tag, files)

    print('\n全部完成，文件在 ' + OUT_DIR)


if __name__ == '__main__':
    main()

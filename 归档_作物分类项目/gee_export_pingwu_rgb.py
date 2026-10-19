# -*- coding: utf-8 -*-
"""gee_export_pingwu_rgb.py — 导出水晶镇/土城藏族乡 云掩膜+median 合成的干净 RGB

getDownloadURL 对大区域 median 会报「User memory limit exceeded」，
改用 Export.image.toDrive（批处理，分 tile 计算，内存限制宽松），
导出到 Google Drive 的 pingwu_rgb 文件夹，再手动下载。
"""
import ee
import time
import geopandas as gpd
import os

ee.Initialize()

BASE = os.path.dirname(os.path.abspath(__file__))
# 官方边界（CGCS2000/EPSG:4523），GEE 调用前需 to_crs(4326)
BOUNDARY = os.path.join(BASE, '平武_水晶镇_土城藏族乡_界.shp')


def mask_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def rgb_median(start, end, roi):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(roi).filterDate(start, end).map(mask_clouds))
    med = col.select(['B4', 'B3', 'B2']).median()
    rgb = (med.divide(10000).multiply(3.0).clamp(0, 1)
           .multiply(255).round().uint8())
    return rgb


def main():
    gdf = gpd.read_file(BOUNDARY)
    minx, miny, maxx, maxy = gdf.to_crs(4326).total_bounds
    roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    tasks = [
        ('summer', '2025-06-01', '2025-09-01'),
        ('winter', '2025-12-01', '2026-03-01'),
    ]
    started = []
    for tag, start, end in tasks:
        img = rgb_median(start, end, roi)
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f'pingwu_rgb_{tag}',
            folder='pingwu_rgb',
            fileNamePrefix=f'平武_水晶镇_土城藏族乡_{tag}_RGB',
            scale=10,
            region=roi,
            crs='EPSG:4326',
            maxPixels=1e9,
        )
        task.start()
        started.append((tag, task))
        print(f'[started] {tag} -> Drive/pingwu_rgb/平武_水晶镇_土城藏族乡_{tag}_RGB.tif')

    # 轮询直到全部完成
    done = set()
    while len(done) < len(started):
        for tag, task in started:
            if tag in done:
                continue
            st = task.status()
            state = st.get('state')
            if state in ('COMPLETED', 'FAILED', 'CANCELED'):
                print(f'[{tag}] {state}')
                if state == 'FAILED':
                    print('  error:', st.get('error_message'))
                done.add(tag)
        time.sleep(15)

    print('\n全部导出完成，请到 Google Drive 的 pingwu_rgb 文件夹下载两个 tif。')


if __name__ == '__main__':
    main()

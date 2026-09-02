# -*- coding: utf-8 -*-
"""experiment_snic.py — 前进村范围 SNIC 分割粒度实验（确定 size 参数）

目标：让 SNIC 超像素尺度接近训练地块（训练库中位 17.5 像素 ≈ 1749 m²），
     避免过碎（噪声大）或过粗（混块）。测试 size=3/4/5/6/8。
"""
import ee
import geopandas as gpd
import numpy as np

ee.Initialize()

# 前进村 ROI（转 WGS84）
g = gpd.read_file(r'待测试数据前进0806\前进0806.gdb', layer='dltb')
g = g[g['ZLDWMC'] == '前进村']
minx, miny, maxx, maxy = g.to_crs(4326).total_bounds
roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)

# D11（7月中，作物旺期）median 做分割底图：真彩色 + NDVI
d11 = s2.filterDate('2025-07-11', '2025-07-20').map(mask_s2_clouds).median()
ndvi = d11.normalizedDifference(['B8', 'B4']).rename('NDVI')
seg_base = d11.select(['B4', 'B3', 'B2']).addBands(ndvi)

print('== 前进村 SNIC 分割粒度实验 ==')
print(f'参考：训练地块中位 17.5 像素(1749m²)，前进村耕地中位 13 像素(1252m²)')

for size in [3, 4, 5, 6, 8]:
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=seg_base, size=size, compactness=5, connectivity=8,
        neighborhoodSize=256)
    clusters = snic.select('clusters')
    hist = clusters.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=roi, scale=10, maxPixels=1e9)
    h = ee.Dictionary(hist.get('clusters')).getInfo()
    counts = np.array(list(h.values()), dtype=float)
    areas = counts * 100  # 10m 像素 -> m²
    print(f'size={size}: 超像素数={len(counts):>6}, '
          f'面积m² 中位={np.median(areas):>6.0f} 均值={areas.mean():>7.0f} '
          f'p25={np.percentile(areas, 25):>6.0f} p75={np.percentile(areas, 75):>6.0f} '
          f'像素中位={np.median(counts):>5.1f}')

print('[done]')

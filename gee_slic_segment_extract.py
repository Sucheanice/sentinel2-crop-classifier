# -*- coding: utf-8 -*-
"""gee_slic_segment_extract.py — SNIC 分割 + 超像素对象关键3旬特征提取（GEE 端）

面向对象预测第一步：在哨兵影像上做 SNIC 超像素分割（默认 size=4），
矢量化每个超像素，提取关键3旬（D04/D11/D14）×10波段 zonal mean + 有效像元数。

输入：范围 shp/gpkg（任意 CRS），或 --qianjincun 直接读前进村 dltb 做测试
输出：slic_objects.gpkg（label + 30波段特征 + buf_pixel + geometry，WGS84）

用法：
  python gee_slic_segment_extract.py --qianjincun --limit 500           # 测试前 500 对象
  python gee_slic_segment_extract.py --input 村界.shp --output out.gpkg  # 正式
"""
import ee
import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

ee.Initialize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOWS = [
    ('D04', '2025-05-01', '2025-05-10'),
    ('D11', '2025-07-11', '2025-07-20'),
    ('D14', '2025-08-11', '2025-08-20'),
]
PHASES = [p for p, _, _ in WINDOWS]
BANDS_GEE = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

SNIC_SIZE = 4
SNIC_COMPACTNESS = 5
SNIC_CONNECTIVITY = 8
SNIC_NEIGHBORHOOD = 256
BATCH = 3000


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_composite(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS_GEE).median().rename([f'{prefix}_{b}' for b in BANDS_GEE])
        composite = med if composite is None else composite.addBands(med)
    return composite


def build_seg_base(roi):
    """分割底图：D11（7月中，作物旺期）median 真彩色 + NDVI。"""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    d11 = s2.filterDate('2025-07-11', '2025-07-20').map(mask_s2_clouds).median()
    ndvi = d11.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return d11.select(['B4', 'B3', 'B2']).addBands(ndvi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', help='范围 shp/gpkg（任意 CRS）')
    ap.add_argument('--qianjincun', action='store_true', help='测试：读前进村 dltb 范围')
    ap.add_argument('--output', default=os.path.join(BASE_DIR, 'slic_objects.gpkg'))
    ap.add_argument('--size', type=int, default=SNIC_SIZE)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个对象（0=全量）')
    args = ap.parse_args()

    if args.qianjincun:
        g = gpd.read_file(os.path.join(BASE_DIR, '待测试数据前进0806', '前进0806.gdb'), layer='dltb')
        g = g[g['ZLDWMC'] == '前进村']
    else:
        g = gpd.read_file(args.input)
    minx, miny, maxx, maxy = g.to_crs(4326).total_bounds
    roi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

    # 1) SNIC 分割
    seg_base = build_seg_base(roi)
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=seg_base, size=args.size, compactness=SNIC_COMPACTNESS,
        connectivity=SNIC_CONNECTIVITY, neighborhoodSize=SNIC_NEIGHBORHOOD)
    clusters = snic.select('clusters')

    # 2) 矢量化超像素
    vec = clusters.reduceToVectors(
        scale=10, geometryType='polygon', eightConnected=True, bestEffort=True,
        geometry=roi)
    n_obj = vec.size().getInfo()
    print(f'[snic] size={args.size} 超像素对象数={n_obj}')

    n_use = n_obj if args.limit <= 0 else min(args.limit, n_obj)
    vec_list = vec.toList(n_use)
    print(f'[extract] 处理 {n_use} 个对象')

    # 3) 关键3旬 composite + 有效像元掩膜
    composite = build_composite(roi)
    band_names = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE]
    valid_any = (composite.select('D04_B8').mask()
                 .Or(composite.select('D11_B8').mask())
                 .Or(composite.select('D14_B8').mask()))

    rows, geoms = [], []
    nb = (n_use + BATCH - 1) // BATCH
    for bi in range(nb):
        sub_list = vec_list.slice(bi * BATCH, (bi + 1) * BATCH)
        sub_fc = ee.FeatureCollection(sub_list)
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
        cnt = valid_any.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.sum().unweighted(), scale=10, tileScale=4)

        feats = reduced.getInfo()['features']
        cnts = {f['properties']['label']: f['properties'].get('sum', 0)
                for f in cnt.getInfo()['features']}
        for f in feats:
            props = f['properties']
            label = props['label']
            row = {k: props.get(k) for k in ['label'] + band_names}
            row['buf_pixel'] = cnts.get(label, 0)
            rows.append(row)
            geoms.append(shape(f['geometry']))
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n_use})')

    df = pd.DataFrame(rows)
    if not df.empty and geoms:
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')
        gdf.to_file(args.output, driver='GPKG')
        print(f'[done] {args.output}: {len(gdf)} 对象, 列={list(df.columns)}')
        print(f'  NaN 行数: {df[band_names].isna().any(axis=1).sum()}/{len(df)}')
    else:
        print('!! 无结果')


if __name__ == '__main__':
    main()

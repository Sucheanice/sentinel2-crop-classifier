# -*- coding: utf-8 -*-
"""gee_extract_dachun.py — 大春(水稻/玉米/高粱) GEE 云端特征提取（对比本地影像）

与本地 extract_dachun_23counties.py 对齐：4 时相(5/6/7/8月) x 10 波段。
差异：GEE 用 SCL 云掩膜 + median 合成（去云更干净），本地是单景。
范围：射洪/泸县/龙马潭/船山/纳溪 5 县（含高粱，用于 GEE vs 本地对比）。
输出：gee_dachun_5counties.csv
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

WINDOWS = [
    ('P1', '2025-05-01', '2025-05-31'),
    ('P2', '2025-06-01', '2025-06-30'),
    ('P3', '2025-07-01', '2025-07-31'),
    ('P4', '2025-08-01', '2025-08-31'),
]

GPKG = r'e:\工作相关\2026年\0624 待测试数据\大春训练标注库_完整字段.gpkg'
OUT_CSV = r'e:\工作相关\2026年\0624 待测试数据\gee_dachun_5counties.csv'

COUNTIES = ['射洪市', '泸县', '龙马潭区', '船山区', '纳溪区']
CROPS = ['水稻', '玉米', '高粱']
BATCH_SIZE = 3000


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_composite(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])
        composite = med if composite is None else composite.addBands(med)
    return composite


def main():
    gdf = gpd.read_file(GPKG)
    gdf = gdf[gdf['ZWMC'].isin(CROPS) & gdf['QXMC'].isin(COUNTIES)].copy()
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    gdf['fid'] = range(len(gdf))
    n = len(gdf)
    print(f'[input] {n} 块')
    print(gdf['ZWMC'].value_counts().to_string())

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    keep_cols = ['fid', 'ZWMC', 'QXMC', 'geometry']

    rows = []
    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(nb):
        sub = gdf.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sub_fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])
        composite = build_composite(sub_fc.geometry())
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid', 'ZWMC', 'QXMC'] + band_names)
        feats = reduced.getInfo()['features']
        for f in feats:
            props = f['properties']
            rows.append({k: props.get(k) for k in ['fid', 'ZWMC', 'QXMC'] + band_names})
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n})')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')


if __name__ == '__main__':
    main()

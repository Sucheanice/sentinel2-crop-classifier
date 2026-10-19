# -*- coding: utf-8 -*-
"""gee_extract_dachun_23counties_3dekad.py — 23县库关键3旬 GEE 提取（水稻/玉米二分类）

与训练口径完全一致：S2_SR_HARMONIZED + SCL 云掩膜 + median 合成 + reduceRegions(mean, scale=10)。

只提取关键 3 旬（数据驱动时相自选结论）：
  D04 = 5月上 (05-01 ~ 05-10)
  D11 = 7月中 (07-11 ~ 07-20)
  D14 = 8月中 (08-11 ~ 08-20)

输入：待训练数据大春/大春标注_水稻玉米.gpkg（类别 in 水稻/玉米，全部 22 县）
输出：gee_dachun_23counties_3dekad.csv（3旬 x 10波段 = 30 列 + fid/类别/QXMC）
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# 关键 3 旬（与 13.10 结论一致）
WINDOWS = [
    ('D04', '2025-05-01', '2025-05-10'),
    ('D11', '2025-07-11', '2025-07-20'),
    ('D14', '2025-08-11', '2025-08-20'),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(BASE_DIR, '待训练数据大春', '大春标注_水稻玉米.gpkg')
OUT_CSV = os.path.join(BASE_DIR, 'gee_dachun_23counties_3dekad.csv')

CROPS = ['水稻', '玉米']
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
    gdf = gdf[gdf['类别'].isin(CROPS)].copy()
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    gdf['fid'] = range(len(gdf))
    n = len(gdf)
    print(f'[input] {n} 块')
    print(f'  类别分布: {dict(gdf["类别"].value_counts())}')
    print(f'  县分布: {dict(gdf["QXMC"].value_counts())}')

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    keep_cols = ['fid', '类别', 'QXMC', 'geometry']

    rows = []
    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'[plan] 共 {nb} 批，每批 {BATCH_SIZE}')
    for bi in range(nb):
        sub = gdf.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sub_fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])
        composite = build_composite(sub_fc.geometry())
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid', '类别', 'QXMC'] + band_names)
        feats = reduced.getInfo()['features']
        for f in feats:
            props = f['properties']
            rows.append({k: props.get(k) for k in ['fid', '类别', 'QXMC'] + band_names})
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n})')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')


if __name__ == '__main__':
    main()

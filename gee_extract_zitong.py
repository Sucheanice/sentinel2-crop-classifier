# -*- coding: utf-8 -*-
"""
梓潼县小春特征提取（无真值推理用）
================================
梓潼县.shp 只有水稻/玉米（大春标注），无小麦/油菜。
本脚本不过滤 ZWMC，保留全部地块边界，提取 4 期小春特征（12/1/3/5月 × 10波段），
供 89.89% 小春模型推理「小麦/油菜」预测图。

时相与训练数据完全一致（gee_extract_features.py 的 4 期窗口）。
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
WINDOWS = [
    ('P1', '2024-12-01', '2025-01-05'),
    ('P2', '2025-01-06', '2025-02-10'),
    ('P3', '2025-03-01', '2025-04-05'),
    ('P4', '2025-04-25', '2025-05-31'),
]
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
    shp = r'e:\工作相关\2026年\0624 待测试数据\待训练数据绵阳市\绵阳市\梓潼县\矢量\梓潼县.shp'
    out_csv = r'e:\工作相关\2026年\0624 待测试数据\gee_梓潼县_小春特征.csv'

    gdf = gpd.read_file(shp)
    print(f'[input] {shp}: {len(gdf)} 块')
    print(f'  字段: {list(gdf.columns)}')
    print(f'  ZWMC 分布: {dict(gdf["ZWMC"].astype(str).value_counts())}')

    # 不过滤 ZWMC，保留全部地块边界
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    gdf['fid'] = range(len(gdf))
    n = len(gdf)

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    # 保留 QXMC + 原始 ZWMC(大春标注) 作参考
    keep_cols = ['fid', 'ZWMC', 'QXMC', 'geometry']

    rows = []
    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'[plan] 共 {nb} 批，每批 {BATCH_SIZE}')
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
    df.to_csv(out_csv, index=False)
    print(f'[done] {out_csv}: {df.shape}')


if __name__ == '__main__':
    main()

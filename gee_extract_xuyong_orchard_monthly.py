# -*- coding: utf-8 -*-
"""gee_extract_xuyong_orchard_monthly.py — 叙永 276 块果园 12 个月月级特征提取

目标：提取李子(190)/柑橘(86) 的 12 个月月级光谱+指数特征，
用于探索「落叶果园 vs 常绿果园」的物候曲线差异（12 个月时间序列）。

输入：待训练数据水果/叙永水果成果数据/result.shp（276 块，ZWMC=李子/柑橘）
输出：xuyong_orchard_monthly.csv（276 块 × [fid/ZWMC/XZMC + 12月×(10波段+NDVI+EVI)]）
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# 2025 年 12 个自然月
MONTHS = [
    ('M01', '2025-01-01', '2025-02-01'),
    ('M02', '2025-02-01', '2025-03-01'),
    ('M03', '2025-03-01', '2025-04-01'),
    ('M04', '2025-04-01', '2025-05-01'),
    ('M05', '2025-05-01', '2025-06-01'),
    ('M06', '2025-06-01', '2025-07-01'),
    ('M07', '2025-07-01', '2025-08-01'),
    ('M08', '2025-08-01', '2025-09-01'),
    ('M09', '2025-09-01', '2025-10-01'),
    ('M10', '2025-10-01', '2025-11-01'),
    ('M11', '2025-11-01', '2025-12-01'),
    ('M12', '2025-12-01', '2026-01-01'),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(BASE_DIR, '待训练数据水果', '叙永水果成果数据', 'result.shp')
OUT_CSV = os.path.join(BASE_DIR, 'xuyong_orchard_monthly.csv')


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_monthly(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in MONTHS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])
        ndvi = med.normalizedDifference([f'{prefix}_B8', f'{prefix}_B4']).rename(f'{prefix}_NDVI')
        evi = med.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': med.select(f'{prefix}_B8'),
             'RED': med.select(f'{prefix}_B4'),
             'BLUE': med.select(f'{prefix}_B2')},
        ).rename(f'{prefix}_EVI')
        med = med.addBands(ndvi).addBands(evi)
        composite = med if composite is None else composite.addBands(med)
    return composite


def main():
    gdf = gpd.read_file(GPKG)
    gdf['fid'] = range(len(gdf))
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    print(f'[input] {len(gdf)} 块')
    print(f'  树种分布: {dict(gdf["ZWMC"].value_counts())}')

    keep_cols = ['fid', 'ZWMC', 'XZMC', 'geometry']
    band_names = []
    for p, _, _ in MONTHS:
        band_names += [f'{p}_{b}' for b in BANDS]
        band_names += [f'{p}_NDVI', f'{p}_EVI']

    fc = ee.FeatureCollection(gdf[keep_cols].__geo_interface__['features'])
    composite = build_monthly(fc.geometry())
    reduced = composite.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=2,
    ).select(['fid', 'ZWMC', 'XZMC'] + band_names)

    feats = reduced.getInfo()['features']
    rows = [{k: f['properties'].get(k) for k in ['fid', 'ZWMC', 'XZMC'] + band_names}
            for f in feats]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')


if __name__ == '__main__':
    main()

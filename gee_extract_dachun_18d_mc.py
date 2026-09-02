# -*- coding: utf-8 -*-
"""gee_extract_dachun_18d_mc.py — 18 旬「多景去云合成」重提

对雨季/季末高云旬（缺失率>50%）把合成窗口从 10 天扩到 30 天，
聚合更多无云像元做 median，减少 NaN；低云旬保持 10 天窗口不牺牲时间分辨率。
输出：gee_dachun_5counties_18d_mc.csv
"""
import ee
import geopandas as gpd
import pandas as pd

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# (prefix, start, end) —— 高云旬已扩窗到 30 天
WINDOWS = [
    ('D01', '2025-04-01', '2025-04-10'),
    ('D02', '2025-04-01', '2025-04-30'),
    ('D03', '2025-04-11', '2025-05-10'),
    ('D04', '2025-05-01', '2025-05-10'),
    ('D05', '2025-05-01', '2025-05-31'),
    ('D06', '2025-05-11', '2025-06-10'),
    ('D07', '2025-05-22', '2025-06-20'),
    ('D08', '2025-06-01', '2025-06-30'),
    ('D09', '2025-06-21', '2025-06-30'),
    ('D10', '2025-07-01', '2025-07-10'),
    ('D11', '2025-07-11', '2025-07-20'),
    ('D12', '2025-07-21', '2025-07-31'),
    ('D13', '2025-08-01', '2025-08-10'),
    ('D14', '2025-08-11', '2025-08-20'),
    ('D15', '2025-08-21', '2025-08-31'),
    ('D16', '2025-08-22', '2025-09-20'),
    ('D17', '2025-09-01', '2025-09-30'),
    ('D18', '2025-09-11', '2025-09-30'),
]

GPKG = r'e:\工作相关\2026年\0624 待测试数据\大春训练标注库_完整字段.gpkg'
OUT_CSV = r'e:\工作相关\2026年\0624 待测试数据\gee_dachun_5counties_18d_mc.csv'

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

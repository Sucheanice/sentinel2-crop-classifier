# -*- coding: utf-8 -*-
"""gee_extract_pingwu_town_monthly.py — 平武 水晶镇/土城藏族乡 12 个月月级特征（粗略观察）

目标：没有中草药地块标注，按乡镇行政边界（官方平武县.shp，CGCS2000/EPSG:4523）做区域级粗略物候观察。
提取两个乡镇 12 个月 NDVI/EVI + 关键波段，输出：
  1. CSV（2 行 × 乡镇名 + 12 月特征）
  2. shp（乡镇边界 + 12 月 NDVI 等属性字段，CGCS2000，可直接套合到影像）

输入：平武_水晶镇_土城藏族乡_界.shp（官方行政边界，EPSG:4523）
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 官方边界（CGCS2000/EPSG:4523），GEE 调用前转 WGS84；输出 shp 保持 4523
BOUNDARY = os.path.join(BASE_DIR, '平武_水晶镇_土城藏族乡_界.shp')
OUT_CSV = os.path.join(BASE_DIR, '平武_中草药乡镇_monthly.csv')
OUT_SHP = os.path.join(BASE_DIR, '平武_水晶镇_土城藏族乡_特征.shp')

BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
MONTHS = [
    ('M01', '2025-01-01', '2025-02-01'), ('M02', '2025-02-01', '2025-03-01'),
    ('M03', '2025-03-01', '2025-04-01'), ('M04', '2025-04-01', '2025-05-01'),
    ('M05', '2025-05-01', '2025-06-01'), ('M06', '2025-06-01', '2025-07-01'),
    ('M07', '2025-07-01', '2025-08-01'), ('M08', '2025-08-01', '2025-09-01'),
    ('M09', '2025-09-01', '2025-10-01'), ('M10', '2025-10-01', '2025-11-01'),
    ('M11', '2025-11-01', '2025-12-01'), ('M12', '2025-12-01', '2026-01-01'),
]


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
    gdf = gpd.read_file(BOUNDARY)
    # OSM relation 名可能是「水晶镇/土城藏族乡」，补一个乡镇名字段
    if 'name' not in gdf.columns:
        gdf['name'] = '未知'
    gdf = gdf.reset_index(drop=True)
    gdf['fid'] = range(len(gdf))
    print(f'[input] {len(gdf)} 个乡镇, CRS={gdf.crs}')
    print(gdf[['name']].to_string())

    band_names = []
    for p, _, _ in MONTHS:
        band_names += [f'{p}_{b}' for b in BANDS]
        band_names += [f'{p}_NDVI', f'{p}_EVI']

    # GEE 内部用 WGS84，传入前转换；本地输出 shp 仍保留 CGCS2000(4523)
    gdf_gee = gdf.to_crs('EPSG:4326')
    fc = ee.FeatureCollection(gdf_gee[['fid', 'name', 'geometry']].__geo_interface__['features'])
    composite = build_monthly(fc.geometry())
    reduced = composite.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=20, tileScale=4,
    ).select(['fid', 'name'] + band_names)

    feats = reduced.getInfo()['features']
    rows = [{k: f['properties'].get(k) for k in ['fid', 'name'] + band_names}
            for f in feats]
    df = pd.DataFrame(rows)

    # 1) CSV
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done csv] {OUT_CSV}: {df.shape}')

    # 2) shp：边界 + 特征字段
    ndvi_cols = [f'{p}_NDVI' for p, _, _ in MONTHS]
    gdf_out = gdf.merge(df[['fid'] + band_names], on='fid', how='left')
    gdf_out = gdf_out.rename(columns={c: c.lower() for c in gdf_out.columns})
    gdf_out.to_file(OUT_SHP, encoding='utf-8')
    print(f'[done shp] {OUT_SHP}')

    # 3) 打印 NDVI 时序（物候观察）
    print('\n=== 12 个月 NDVI 时序 ===')
    for _, r in df.iterrows():
        print(f"\n{r['name']}:")
        for p, _, _ in MONTHS:
            v = r[f'{p}_NDVI']
            print(f"  {p}  NDVI={v:.3f}" if v == v else f"  {p}  NDVI=NaN")


if __name__ == '__main__':
    main()

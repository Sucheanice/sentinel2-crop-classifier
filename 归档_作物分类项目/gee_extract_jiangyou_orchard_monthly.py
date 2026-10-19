# -*- coding: utf-8 -*-
"""gee_extract_jiangyou_orchard_monthly.py — 江油 4 镇 244 块果园 12 个月月级特征提取（v2 升级）

输入：训练与标注数据/江油市资源调查地块_按镇/江油市果园地块.shp（244 块，crop=果园/果树）
输出：jiangyou_orchard_monthly_2025_v2.csv
  标识字段：fid/town/village/crop/area_mu
  月度波段：12 月 × 10 波段
  月度指数：12 月 × 8 指数（NDVI/EVI/NDWI/SAVI/LSWI/NDRE/RVI/MNDWI）
  时序统计：每波段/每指数 跨 12 月 min/max/mean/std/range
  物候指标：NDVI amplitude/growth_rate/senescence_rate/peak_index/integral
  首尾差值：NDVI/EVI/NDWI/SAVI

实现：按月逐月 reduceRegions（每次仅 18 波段），避免一次性超多波段导致 GEE 内存超限；
      物候/时序统计/首尾差值在 pandas 内按镇计算。
"""
import os

# 修复 PROJ 库版本不匹配：系统 PROJ_LIB 指向 PostgreSQL/PostGIS 旧 proj.db，强制覆盖。
os.environ['PROJ_LIB'] = r'D:\miniconda3\envs\agri_local\Library\share\proj'

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
VI = ['NDVI', 'EVI', 'NDWI', 'SAVI', 'LSWI', 'NDRE', 'RVI', 'MNDWI']

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

GPKG = r"E:/工作相关/2026年/0624 待测试数据/训练与标注数据/江油市资源调查地块_按镇/江油市果园地块.shp"
OUT_CSV = r"E:/工作相关/2026年/0624 待测试数据/归档_作物分类项目/jiangyou_orchard_monthly_2025_v2.csv"


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_month(s2, prefix, start, end):
    col = s2.filterDate(start, end).map(mask_s2_clouds)
    med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])

    b2 = med.select(f'{prefix}_B2')
    b3 = med.select(f'{prefix}_B3')
    b4 = med.select(f'{prefix}_B4')
    b5 = med.select(f'{prefix}_B5')
    b8 = med.select(f'{prefix}_B8')
    b11 = med.select(f'{prefix}_B11')

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename(f'{prefix}_NDVI')
    evi = med.expression(
        '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': b8, 'RED': b4, 'BLUE': b2}).rename(f'{prefix}_EVI')
    ndwi = b3.subtract(b8).divide(b3.add(b8)).rename(f'{prefix}_NDWI')
    savi = med.expression(
        '1.5 * (NIR - RED) / (NIR + RED + 0.5)',
        {'NIR': b8, 'RED': b4}).rename(f'{prefix}_SAVI')
    lswi = b8.subtract(b11).divide(b8.add(b11)).rename(f'{prefix}_LSWI')
    mndwi = b3.subtract(b11).divide(b3.add(b11)).rename(f'{prefix}_MNDWI')
    ndre = b8.subtract(b5).divide(b8.add(b5)).rename(f'{prefix}_NDRE')
    rvi = b8.divide(b4.add(1e-6)).rename(f'{prefix}_RVI')

    return med.addBands([ndvi, evi, ndwi, savi, lswi, mndwi, ndre, rvi])


def add_derived_features(df):
    """在 12 个月月值基础上追加时序统计、物候指标、首尾差值（对齐 common.py 特征体系）。"""
    month_prefixes = [m[0] for m in MONTHS]

    # Layer 3: 时序统计（每波段/每指数跨 12 月 min/max/mean/std/range）
    for b in BANDS:
        arr = df[[f'{p}_{b}' for p in month_prefixes]].to_numpy(dtype=float)
        df[f'TSTAT_{b}_min'] = np.nanmin(arr, axis=1)
        df[f'TSTAT_{b}_max'] = np.nanmax(arr, axis=1)
        df[f'TSTAT_{b}_mean'] = np.nanmean(arr, axis=1)
        df[f'TSTAT_{b}_std'] = np.nanstd(arr, axis=1)
        df[f'TSTAT_{b}_range'] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)

    for vi in VI:
        arr = df[[f'{p}_{vi}' for p in month_prefixes]].to_numpy(dtype=float)
        df[f'TSTAT_{vi}_min'] = np.nanmin(arr, axis=1)
        df[f'TSTAT_{vi}_max'] = np.nanmax(arr, axis=1)
        df[f'TSTAT_{vi}_mean'] = np.nanmean(arr, axis=1)
        df[f'TSTAT_{vi}_std'] = np.nanstd(arr, axis=1)
        df[f'TSTAT_{vi}_range'] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)

    # Layer 4: 物候指标（基于 NDVI 12 月序列）
    ndvi = df[[f'{p}_NDVI' for p in month_prefixes]].to_numpy(dtype=float)
    n = ndvi.shape[1]
    half = max(2, n // 2)
    df['PHENO_amplitude'] = np.nanmax(ndvi, axis=1) - np.nanmin(ndvi, axis=1)
    df['PHENO_growth_rate'] = (np.nanmean(ndvi[:, :half], axis=1) - ndvi[:, 0]) / (half - 1 + 1e-8)
    df['PHENO_senescence_rate'] = (ndvi[:, -1] - np.nanmean(ndvi[:, half:], axis=1)) / (n - half + 1e-8)
    ndvi_filled = np.where(np.isnan(ndvi), -np.inf, ndvi)
    peak = np.argmax(ndvi_filled, axis=1).astype(float)
    peak[np.all(np.isnan(ndvi), axis=1)] = np.nan
    df['PHENO_peak_index'] = peak
    df['PHENO_integral'] = np.nansum((ndvi[:, :-1] + ndvi[:, 1:]) * 0.5, axis=1)

    # Layer 5: 首尾差值（变化方向）
    for vi in ['NDVI', 'EVI', 'NDWI', 'SAVI']:
        df[f'DELTA_{vi}'] = df[f'M12_{vi}'] - df[f'M01_{vi}']

    return df


def main():
    gdf = gpd.read_file(GPKG)
    gdf = gdf.to_crs(epsg=4326)  # GEE 需要经纬度坐标（源为 CGCS2000 3度带投影）
    gdf['fid'] = range(len(gdf))
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    print(f'[input] {len(gdf)} 块')
    print(f'  镇分布: {dict(gdf["town"].value_counts())}')

    keep_cols = ['fid', 'town', 'village', 'crop', 'area_mu', 'geometry']
    out_props = ['fid', 'town', 'village', 'crop', 'area_mu']

    all_rows = []
    for town, sub in gdf.groupby('town'):
        fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(fc.geometry())
        town_rows = None
        for prefix, start, end in MONTHS:
            month_bands = [f'{prefix}_{b}' for b in BANDS] + [f'{prefix}_{vi}' for vi in VI]
            med = build_month(s2, prefix, start, end)
            reduced = med.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=20, tileScale=8,
            ).select(out_props + month_bands)
            feats = reduced.getInfo()['features']
            rows = [{k: f['properties'].get(k) for k in out_props + month_bands} for f in feats]
            mdf = pd.DataFrame(rows)
            town_rows = mdf if town_rows is None else town_rows.merge(mdf, on=out_props, how='outer')

        town_rows = add_derived_features(town_rows)
        all_rows += town_rows.to_dict('records')
        print(f'  [town] {town}: {len(sub)} 块 × 12 月 → {len(town_rows.columns)} 特征完成')

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_CSV}: {df.shape}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""gee_extract_jiangyou_orchard_sar_glcm.py — 江油 244 块果园 SAR + GLCM 纹理特征

SAR：Sentinel-1 GRD 按月 VV/VH(dB) + VV-VH，12 月 + 时序统计（min/max/mean/std/range）
GLCM：Sentinel-2 生长季(7-8月) B8 与 NDVI 各 7 个纹理统计量（contrast/corr/asm/ent/var/idm/diss）

输出：
  jiangyou_orchard_sar_glcm_2025.csv  —— fid + SAR/GLCM 特征
  jiangyou_orchard_features_v3.csv    —— v2 光谱特征 + SAR/GLCM 合并全量表
"""
import os

# 修复 PROJ 库版本不匹配：系统 PROJ_LIB 指向 PostgreSQL/PostGIS 旧 proj.db，强制覆盖。
os.environ['PROJ_LIB'] = r'D:\miniconda3\envs\agri_local\Library\share\proj'

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

ee.Initialize()

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
V2_CSV = r"E:/工作相关/2026年/0624 待测试数据/归档_作物分类项目/jiangyou_orchard_monthly_2025_v2.csv"
OUT_SAR_GLCM = r"E:/工作相关/2026年/0624 待测试数据/归档_作物分类项目/jiangyou_orchard_sar_glcm_2025.csv"
OUT_FULL = r"E:/工作相关/2026年/0624 待测试数据/归档_作物分类项目/jiangyou_orchard_features_v3.csv"

GLCM_METRICS = ['contrast', 'corr', 'asm', 'ent', 'var', 'idm', 'diss']
SAR_FEATS = ['SAR_VV', 'SAR_VH', 'SAR_VV_VH']


def mask_s2_clouds(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_sar_month(s1, prefix, start, end):
    # COPERNICUS/S1_GRD 的 VV/VH 波段在 GEE 中已是 dB 值（负值），直接使用，不做 log10 转换。
    col = s1.filterDate(start, end)
    vv = col.select('VV').median().rename(f'{prefix}_SAR_VV')
    vh = col.select('VH').median().rename(f'{prefix}_SAR_VH')
    diff = vv.subtract(vh).rename(f'{prefix}_SAR_VV_VH')
    return vv.addBands(vh).addBands(diff)


def build_glcm(s2):
    col = s2.filterDate('2025-07-01', '2025-09-01').map(mask_s2_clouds)
    med = col.select(['B4', 'B8']).median()
    b8 = med.select('B8')
    ndvi = med.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # 量化到 8bit（0-255）以满足 glcmTexture 输入要求
    b8_q = b8.unitScale(0, 10000).multiply(255).toByte()
    ndvi_q = ndvi.unitScale(-1, 1).multiply(255).toByte()

    glcm_b8 = b8_q.glcmTexture(1)
    glcm_ndvi = ndvi_q.glcmTexture(1)

    bands = []
    for src, name in [(glcm_b8, 'B8'), (glcm_ndvi, 'NDVI')]:
        for m in GLCM_METRICS:
            bands.append(src.select(f'{name}_{m}').rename(f'GLCM_{name}_{m}'))
    return ee.Image.cat(bands)


def add_sar_tstats(df):
    """对 12 个月 SAR_VV/SAR_VH/SAR_VV_VH 计算时序统计（对齐 common.py 口径）。"""
    month_prefixes = [m[0] for m in MONTHS]
    for feat in SAR_FEATS:
        arr = df[[f'{p}_{feat}' for p in month_prefixes]].to_numpy(dtype=float)
        df[f'TSTAT_{feat}_min'] = np.nanmin(arr, axis=1)
        df[f'TSTAT_{feat}_max'] = np.nanmax(arr, axis=1)
        df[f'TSTAT_{feat}_mean'] = np.nanmean(arr, axis=1)
        df[f'TSTAT_{feat}_std'] = np.nanstd(arr, axis=1)
        df[f'TSTAT_{feat}_range'] = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
    return df


def main():
    gdf = gpd.read_file(GPKG)
    gdf = gdf.to_crs(epsg=4326)  # GEE 需要经纬度坐标（源为 CGCS2000 3度带投影）
    gdf['fid'] = range(len(gdf))
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    print(f'[input] {len(gdf)} 块')

    keep_cols = ['fid', 'geometry']
    out_props = ['fid']

    all_rows = []
    for town, sub in gdf.groupby('town'):
        fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])

        # === Sentinel-1 SAR 按月 ===
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(fc.geometry()) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))

        town_rows = None
        for prefix, start, end in MONTHS:
            bands = [f'{prefix}_{feat}' for feat in SAR_FEATS]
            img = build_sar_month(s1, prefix, start, end)
            reduced = img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=20, tileScale=8,
            ).select(out_props + bands)
            feats = reduced.getInfo()['features']
            rows = [{k: f['properties'].get(k) for k in out_props + bands} for f in feats]
            mdf = pd.DataFrame(rows)
            town_rows = mdf if town_rows is None else town_rows.merge(mdf, on=out_props, how='outer')

        # === GLCM 生长季纹理 ===
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(fc.geometry())
        glcm_img = build_glcm(s2)
        glcm_bands = [f'GLCM_{name}_{m}' for name in ['B8', 'NDVI'] for m in GLCM_METRICS]
        reduced_glcm = glcm_img.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=20, tileScale=4,
        ).select(out_props + glcm_bands)
        feats = reduced_glcm.getInfo()['features']
        rows = [{k: f['properties'].get(k) for k in out_props + glcm_bands} for f in feats]
        glcm_df = pd.DataFrame(rows)
        town_rows = town_rows.merge(glcm_df, on=out_props, how='outer')

        town_rows = add_sar_tstats(town_rows)
        all_rows += town_rows.to_dict('records')
        print(f'  [town] {town}: {len(sub)} 块 → {len(town_rows.columns)} SAR/GLCM 特征完成')

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_SAR_GLCM, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_SAR_GLCM}: {df.shape}')

    # 合并 v2 光谱特征 → v3 全量表
    df_v2 = pd.read_csv(V2_CSV)
    df_full = df_v2.merge(df, on='fid', how='inner')
    df_full.to_csv(OUT_FULL, index=False, encoding='utf-8-sig')
    print(f'[done] {OUT_FULL}: {df_full.shape} (v2 + SAR/GLCM)')


if __name__ == '__main__':
    main()

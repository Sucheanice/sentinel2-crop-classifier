# -*- coding: utf-8 -*-
"""
GEE 小春特征提取（云端，免下载影像）
流程：读本地 shp -> S2 L2A 按 4 时相窗口合成(SCL云掩膜 + median) -> reduceRegions 提取 zonal mean
输出：每地块 4 期 x 10 波段 = 40 列特征 CSV

时相（小春 2024-2025 生长季）：
  P1 越冬  2024-12
  P2 休眠  2025-01
  P3 返青/油菜开花 2025-03
  P4 收获前 2025-05
"""
import ee
import geopandas as gpd
import pandas as pd
import sys
import os

ee.Initialize()  # 项目已用 earthengine set_project 固化

# Sentinel-2 L2A 10 波段（与本地 extract_features 一致）
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# 4 个时相窗口
WINDOWS = [
    ('P1', '2024-12-01', '2025-01-05'),
    ('P2', '2025-01-06', '2025-02-10'),
    ('P3', '2025-03-01', '2025-04-05'),
    ('P4', '2025-04-25', '2025-05-31'),
]


def mask_s2_clouds(img):
    """SCL 云掩膜：去除云影(3)/中云(8)/高云(9)/薄卷云(10)"""
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def build_composite(roi):
    """合成 4 期 40 波段影像"""
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS).median().rename([f'{prefix}_{b}' for b in BANDS])
        composite = med if composite is None else composite.addBands(med)
    return composite


def extract(shp_path, out_csv=None, sample_n=None, batch_size=None):
    gdf = gpd.read_file(shp_path)
    # 只留小麦/油菜
    gdf = gdf[gdf['ZWMC'].isin(['小麦', '油菜'])].copy()
    if sample_n:
        gdf = gdf.head(sample_n)
    # 简化几何（容差约 10m），大幅减小 payload；zonal mean 对边界精度不敏感
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    gdf['fid'] = range(len(gdf))
    n = len(gdf)
    print(f'[input] {shp_path}: {n} 块 (小麦/油菜)')
    # 大县自动分批（单次 payload 超 10MB 风险），小县一次性
    if batch_size is None and n > 5000:
        batch_size = 3000

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    keep_cols = ['fid', 'ZWMC', 'QXMC', 'geometry']

    if batch_size is None:
        # 一次性 reduceRegions + getInfo
        fc = ee.FeatureCollection(gdf[keep_cols].__geo_interface__['features'])
        composite = build_composite(fc.geometry())
        reduced = composite.reduceRegions(
            collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid', 'ZWMC', 'QXMC'] + band_names)
        feats = reduced.getInfo()['features']
        rows = []
        for f in feats:
            props = f['properties']
            rows.append({k: props.get(k) for k in ['fid', 'ZWMC', 'QXMC'] + band_names})
    else:
        # 分批：每批用「子集范围」独立合成，避免每批重算整个县的 median
        rows = []
        nb = (n + batch_size - 1) // batch_size
        for bi in range(nb):
            sub = gdf.iloc[bi * batch_size:(bi + 1) * batch_size]
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
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f'[done] {out_csv}: {df.shape}')
    else:
        print(f'[done] {shp_path}: {df.shape}')
    return df


def extract_dir(dir_path, out_csv, sample_n=None):
    """循环提取某目录下所有 shp，合并保存"""
    import os
    fs = sorted(f for f in os.listdir(dir_path) if f.endswith('.shp'))
    parts = []
    for f in fs:
        df = extract(os.path.join(dir_path, f), None, sample_n=sample_n)
        parts.append(df)
    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(out_csv, index=False)
    print(f'[merge] {out_csv}: {all_df.shape}')
    return all_df


if __name__ == '__main__':
    base = r'e:\工作相关\2026年\0624 待测试数据\2024小春'
    # 广安 5 区
    extract_dir(
        os.path.join(base, '广安市'),
        r'e:\工作相关\2026年\0624 待测试数据\gee_广安5区_特征.csv',
    )

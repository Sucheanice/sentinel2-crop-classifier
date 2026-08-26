# -*- coding: utf-8 -*-
"""
GEE 提取 6 村图斑的哨兵 NIR/红边/SWIR 波段（用地分类融合用）
================================================================
无人机影像只有 RGB 三波段（0715~0721 拍摄，2025年），缺乏区分植被/非植被的
NIR/红边信息。本脚本从 Sentinel-2 提取同期（2025-07）8 个波段 zonal mean，
与无人机 RGB 特征拼接后重训，重点提升「农用地 vs 建设用地」大类。

fid 顺序与 landuse_train_multi.load_all() 完全一致（前进/印坪/大岳/沉水/马阁寺/龙宫）。
"""
import ee
import geopandas as gpd
import pandas as pd

ee.Initialize()

BASE = r'F:\0421给yxx\提交成果'
VILLAGES = ['前进村', '印坪村', '大岳村', '沉水村', '马阁寺村', '龙宫村']
FILES = ['2前进.shp', '2印坪.shp', '2大岳.shp', '2沉水.shp', '2马阁寺.shp', '2龙宫.shp']

# 补充无人机 RGB 所缺的波段：红边(5/6/7)+NIR(8/8A)+SWIR(11/12)，另带 B4 算 NDVI
BANDS = ['B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
WINDOWS = [('JUL', '2025-07-01', '2025-07-31')]
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


def load_all():
    import os
    parts = []
    for v, f in zip(VILLAGES, FILES):
        shp = os.path.join(BASE, v, '矢量', f)
        g = gpd.read_file(shp)
        g['村'] = v
        parts.append(g)
    gdf = pd.concat(parts, ignore_index=True)
    gdf['fid'] = range(len(gdf))
    return gdf


def main():
    out_csv = r'e:\工作相关\2026年\0624 待测试数据\gee_6村_哨兵波段_202507.csv'
    gdf = load_all()
    print(f'[input] 6村图斑: {len(gdf)}')
    print(f'  原始 CRS: {gdf.crs}')

    # 转 WGS84（GEE 需要经纬度），简化几何约 10m
    gdf = gdf.to_crs('EPSG:4326')
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.0001, preserve_topology=True)

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    keep_cols = ['fid', '村', 'geometry']
    n = len(gdf)

    rows = []
    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'[plan] 共 {nb} 批，每批 {BATCH_SIZE}')
    for bi in range(nb):
        sub = gdf.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sub_fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])
        composite = build_composite(sub_fc.geometry())
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid', '村'] + band_names)
        feats = reduced.getInfo()['features']
        for f in feats:
            props = f['properties']
            rows.append({k: props.get(k) for k in ['fid', '村'] + band_names})
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n})')

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'[done] {out_csv}: {df.shape}')
    print(f'  缺失 fid 数: {n - df["fid"].nunique()}')


if __name__ == '__main__':
    main()

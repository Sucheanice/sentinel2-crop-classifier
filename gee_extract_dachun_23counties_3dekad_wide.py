# -*- coding: utf-8 -*-
"""gee_extract_dachun_23counties_3dekad_wide.py — 23县库关键3旬 GEE 提取（20天宽窗口版）

与 gee_extract_dachun_23counties_3dekad.py 完全一致，仅把窗口从 10 天加宽到 20 天，
用于评估 20 天窗口重训时训练样本的缺值率/保留率变化。

关键 3 旬（20 天宽窗口，与 gee_slic_extract_wide.py 一致）：
  D04 = 5月上 (04-25 ~ 05-15)
  D11 = 7月中 (07-05 ~ 07-25)
  D14 = 8月中 (08-05 ~ 08-25)

输入：待训练数据大春/大春标注_水稻玉米.gpkg（类别 in 水稻/玉米，全部 22 县）
输出：gee_dachun_23counties_3dekad_wide.csv
"""
import ee
import geopandas as gpd
import pandas as pd
import os

ee.Initialize()

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# 20 天宽窗口
WINDOWS = [
    ('D04', '2025-04-25', '2025-05-15'),
    ('D11', '2025-07-05', '2025-07-25'),
    ('D14', '2025-08-05', '2025-08-25'),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPKG = os.path.join(BASE_DIR, '待训练数据大春', '大春标注_水稻玉米.gpkg')
OUT_CSV = os.path.join(BASE_DIR, 'gee_dachun_23counties_3dekad_wide.csv')

CROPS = ['水稻', '玉米']
BATCH_SIZE = 1000  # 20天窗口单批计算量较大，3000 会触发 GEE 超时，降为 1000


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

    band_names = [f'{p}_{b}' for p, _, _ in WINDOWS for b in BANDS]
    keep_cols = ['fid', '类别', 'QXMC', 'geometry']

    # 断点续传：已有 CSV 则跳过已完成的 fid
    done_fids = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV, encoding='utf-8-sig', usecols=['fid'])
        done_fids = set(prev['fid'].astype(int))
        print(f'[resume] 已发现 {len(done_fids)} 块已提取，跳过')
        gdf = gdf[~gdf['fid'].isin(done_fids)].copy()

    n = len(gdf)
    print(f'[input] 待提取 {n} 块 (总 87521)')
    if n == 0:
        print('[done] 全部已完成')
        return

    nb = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'[plan] 共 {nb} 批，每批 {BATCH_SIZE}')

    header = not os.path.exists(OUT_CSV)
    for bi in range(nb):
        sub = gdf.iloc[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        sub_fc = ee.FeatureCollection(sub[keep_cols].__geo_interface__['features'])
        composite = build_composite(sub_fc.geometry())
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4,
        ).select(['fid', '类别', 'QXMC'] + band_names)
        feats = reduced.getInfo()['features']
        rows = [{k: f['properties'].get(k) for k in ['fid', '类别', 'QXMC'] + band_names}
                for f in feats]
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, mode='a', index=False, encoding='utf-8-sig', header=header)
        header = False
        print(f'  batch {bi + 1}/{nb} done (本批 {len(df)} 块)')

    total = len(done_fids) + n
    print(f'[done] {OUT_CSV}: 累计 {total} 块')


if __name__ == '__main__':
    main()

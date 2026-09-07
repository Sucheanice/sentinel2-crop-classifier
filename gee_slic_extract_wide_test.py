# -*- coding: utf-8 -*-
"""gee_slic_extract_wide_test.py — 文通镇加宽时相窗口补提取实验

针对云覆盖导致的特征 NaN 缺失，把关键 3 旬窗口从 10 天放宽到 20 天（前后各扩 5 天，
不重叠），其余逻辑与 yanting 版一致。只重提文通镇，对比 NaN 比例与缺失率。
"""
import ee
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping

ee.Initialize()

from gee_slic_segment_extract import (
    PHASES, BANDS_GEE, mask_s2_clouds, build_seg_base,
    SNIC_SIZE, SNIC_COMPACTNESS, SNIC_CONNECTIVITY, SNIC_NEIGHBORHOOD, BATCH,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YANTING_SHP = os.path.join(BASE_DIR, r'待训练数据绵阳市\绵阳市\盐亭县\矢量\盐亭县.shp')

# 加宽窗口：每旬 10 天 -> 20 天
WIDE_WINDOWS = [
    ('D04', '2025-04-25', '2025-05-15'),
    ('D11', '2025-07-05', '2025-07-25'),
    ('D14', '2025-08-05', '2025-08-25'),
]


def build_composite_wide(roi):
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    composite = None
    for prefix, start, end in WIDE_WINDOWS:
        col = s2.filterDate(start, end).map(mask_s2_clouds)
        med = col.select(BANDS_GEE).median().rename([f'{prefix}_{b}' for b in BANDS_GEE])
        composite = med if composite is None else composite.addBands(med)
    return composite


def shp_to_ee_geometry(gdf):
    union = gdf.to_crs(4326).geometry.unary_union
    return ee.Geometry(mapping(union))


def main():
    name = '文通镇'
    gdf = gpd.read_file(YANTING_SHP)
    sub = gdf[gdf['XZMC'] == name]
    roi = shp_to_ee_geometry(sub)

    # 分割底图改用加宽 D11（07-05~07-25），消除云覆盖导致的 SNIC 分割空洞
    seg_base = build_seg_base(roi, wide=True)
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=seg_base, size=SNIC_SIZE, compactness=SNIC_COMPACTNESS,
        connectivity=SNIC_CONNECTIVITY, neighborhoodSize=SNIC_NEIGHBORHOOD)
    clusters = snic.select('clusters')
    vec = clusters.reduceToVectors(
        scale=10, geometryType='polygon', eightConnected=True, bestEffort=True,
        geometry=roi)
    n_obj = vec.size().getInfo()
    print(f'[snic] 文通镇 超像素对象数={n_obj}')

    composite = build_composite_wide(roi)
    band_names = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE]
    valid_any = (composite.select('D04_B8').mask()
                 .Or(composite.select('D11_B8').mask())
                 .Or(composite.select('D14_B8').mask()))

    rows, geoms = [], []
    vec_list = vec.toList(n_obj)
    nb = (n_obj + BATCH - 1) // BATCH
    for bi in range(nb):
        sub_list = vec_list.slice(bi * BATCH, (bi + 1) * BATCH)
        sub_fc = ee.FeatureCollection(sub_list)
        reduced = composite.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4)
        cnt = valid_any.reduceRegions(
            collection=sub_fc, reducer=ee.Reducer.sum().unweighted(), scale=10, tileScale=4)
        feats = reduced.getInfo()['features']
        cnts = {f['properties']['label']: f['properties'].get('sum', 0)
                for f in cnt.getInfo()['features']}
        for f in feats:
            props = f['properties']
            label = props['label']
            row = {k: props.get(k) for k in ['label'] + band_names}
            row['buf_pixel'] = cnts.get(label, 0)
            rows.append(row)
            geoms.append(shape(f['geometry']))
        print(f'  batch {bi + 1}/{nb} done (累计 {len(rows)}/{n_obj})')

    df = pd.DataFrame(rows)
    out_path = os.path.join(BASE_DIR, 'slic_文通镇_wide.gpkg')
    if not df.empty and geoms:
        gdf_out = gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')
        gdf_out.to_file(out_path, driver='GPKG')
        print(f'[done] {out_path}: {len(gdf_out)} 对象')
        print(f'  NaN 行数: {df[band_names].isna().any(axis=1).sum()}/{len(df)}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""gee_slic_segment_extract_yanting.py — 盐亭 4 乡镇大春 SNIC 分割 + 特征提取

从盐亭县.shp 按乡镇(XZMC)筛选出第一批 4 个目标乡镇，逐乡镇做：
  - 乡镇界精确 ROI（unary_union，替代外接矩形 bbox）
  - SNIC 超像素分割（size=4）
  - 关键3旬(D04/D11/D14) × 10波段 zonal mean + 有效像元数

输出：slic_{乡镇名}.gpkg（label + 30波段特征 + buf_pixel + geometry, WGS84）

用法：
  python gee_slic_segment_extract_yanting.py                 # 全量 4 乡镇
  python gee_slic_segment_extract_yanting.py --only 文通镇   # 只跑单个乡镇
"""
import ee
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping

ee.Initialize()

from gee_slic_segment_extract import (
    WINDOWS, PHASES, BANDS_GEE,
    build_composite, build_seg_base,
    SNIC_SIZE, SNIC_COMPACTNESS, SNIC_CONNECTIVITY, SNIC_NEIGHBORHOOD, BATCH,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YANTING_SHP = os.path.join(BASE_DIR, r'待训练数据绵阳市\绵阳市\盐亭县\矢量\盐亭县.shp')
TOWNS = ['大兴回族乡', '富驿镇', '文通镇', '莲花湖乡']


def shp_to_ee_geometry(gdf):
    """乡镇地块 shp -> ee.Geometry 精确多边形（合并所有面）。"""
    union = gdf.to_crs(4326).geometry.unary_union
    return ee.Geometry(mapping(union))


def extract_town(name, gdf_sub, size=SNIC_SIZE):
    out_path = os.path.join(BASE_DIR, f'slic_{name}.gpkg')
    print(f'\n===== {name}（{len(gdf_sub)} 块投保地块）=====')
    roi = shp_to_ee_geometry(gdf_sub)

    # 1) SNIC 分割（乡镇界内）
    seg_base = build_seg_base(roi)
    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image=seg_base, size=size, compactness=SNIC_COMPACTNESS,
        connectivity=SNIC_CONNECTIVITY, neighborhoodSize=SNIC_NEIGHBORHOOD)
    clusters = snic.select('clusters')
    vec = clusters.reduceToVectors(
        scale=10, geometryType='polygon', eightConnected=True, bestEffort=True,
        geometry=roi)
    n_obj = vec.size().getInfo()
    print(f'[snic] size={size} 超像素对象数={n_obj}')

    # 2) 关键3旬 composite + 有效像元掩膜
    composite = build_composite(roi)
    band_names = [f'{p}_{b}' for p in PHASES for b in BANDS_GEE]
    valid_any = (composite.select('D04_B8').mask()
                 .Or(composite.select('D11_B8').mask())
                 .Or(composite.select('D14_B8').mask()))

    # 3) 分批 reduceRegions 提取
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
    if not df.empty and geoms:
        gdf_out = gpd.GeoDataFrame(df, geometry=geoms, crs='EPSG:4326')
        gdf_out.to_file(out_path, driver='GPKG')
        print(f'[done] {out_path}: {len(gdf_out)} 对象')
        print(f'  NaN 行数: {df[band_names].isna().any(axis=1).sum()}/{len(df)}')
    return out_path


def main():
    only = None
    if '--only' in sys.argv:
        only = sys.argv[sys.argv.index('--only') + 1]
    gdf = gpd.read_file(YANTING_SHP)
    for name in TOWNS:
        if only and name != only:
            continue
        sub = gdf[gdf['XZMC'] == name]
        if len(sub) == 0:
            print(f'!! {name}: XZMC 无匹配')
            continue
        try:
            extract_town(name, sub)
        except Exception as e:
            print(f'!! {name} 失败: {e}')


if __name__ == '__main__':
    main()

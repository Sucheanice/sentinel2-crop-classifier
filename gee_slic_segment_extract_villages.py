# -*- coding: utf-8 -*-
"""
gee_slic_segment_extract_villages.py — 江油 9 村村界精确 ROI 批量提取
================================================================
与 gee_slic_segment_extract.py 的区别：
  - ROI 用「村界 shp 精确多边形」而不是外接矩形 bbox（避免村界外无标注区域）
  - 批量循环 9 村，每村输出独立 gpkg

输出：slic_{村名}.gpkg（label + 30波段特征 + buf_pixel + geometry, WGS84）
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
    mask_s2_clouds, build_composite, build_seg_base,
    SNIC_SIZE, SNIC_COMPACTNESS, SNIC_CONNECTIVITY, SNIC_NEIGHBORHOOD, BATCH,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 9 村：村名 -> 村界 shp
VILLAGES = [
    ('前进村', r'F:\0421给yxx\提交成果\前进村\矢量\前进村村界.shp'),
    ('印坪村', r'F:\0421给yxx\提交成果\印坪村\矢量\村边界.shp'),
    ('大岳村', r'F:\0421给yxx\提交成果\大岳村\矢量\1.shp'),
    ('沉水村', r'F:\0421给yxx\提交成果\沉水村\矢量\沉水村村界.shp'),
    ('马阁寺村', r'F:\0421给yxx\提交成果\马阁寺村\矢量\马阁寺村村界.shp'),
    ('龙宫村', r'F:\0421给yxx\提交成果\龙宫村\矢量\龙宫村村界.shp'),
    ('宝珠村', r'E:\工作相关\2026年\0624 待测试数据\20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\宝珠村村界.shp'),
    ('朝天村', r'E:\工作相关\2026年\0624 待测试数据\20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\朝天村村界.shp'),
    ('统一村', r'E:\工作相关\2026年\0624 待测试数据\20260902绵阳(项目任务区)\20260901绵阳对应影像和矢量\统一村村界.shp'),
]


def shp_to_ee_geometry(gdf):
    """村界 shp -> ee.Geometry 精确多边形（合并所有面）。"""
    union = gdf.to_crs(4326).geometry.unary_union
    return ee.Geometry(mapping(union))


def extract_village(name, boundary_shp, size=SNIC_SIZE):
    out_path = os.path.join(BASE_DIR, f'slic_{name}.gpkg')
    print(f'\n===== {name} =====')
    gdf = gpd.read_file(boundary_shp)
    roi = shp_to_ee_geometry(gdf)

    # 1) SNIC 分割（村界内）
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
    for name, bnd in VILLAGES:
        if only and name != only:
            continue
        try:
            extract_village(name, bnd)
        except Exception as e:
            print(f'!! {name} 失败: {e}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""临时诊断：盐亭/文通镇 D11(7月) S1 可用性 + 23县库水稻/玉米 S1 区分度。

目的：在决定「补 SAR 特征重训模型」前，先验证 S1(VV/VH) 在关键旬是否
  1) 覆盖文通镇（雷达不受云影响，应接近全有效）；
  2) 对水稻/玉米有区分度（水稻水面/灌水 VV 弱，玉米旱地 VV 强）。
"""
import ee
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping

ee.Initialize()

BASE_DIR = r'e:\工作相关\2026年\0624 待测试数据'

D11_START, D11_END = '2025-07-05', '2025-07-25'


def to_db(img):
    return ee.Image(10.0).multiply(img.log10())


def s1_col(roi, start, end):
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(roi).filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    vv = s1.select('VV').map(to_db).median().rename('VV')
    vh = s1.select('VH').map(to_db).median().rename('VH')
    return vv.addBands(vh)


def main():
    # 1) 文通镇 S1 可用性
    g = gpd.read_file(os.path.join(BASE_DIR, '盐亭_文通镇_地块.shp'))
    roi = ee.Geometry(mapping(g.to_crs(4326).geometry.union_all()))
    n = (ee.ImageCollection('COPERNICUS/S1_GRD')
         .filterBounds(roi).filterDate(D11_START, D11_END)
         .filter(ee.Filter.eq('instrumentMode', 'IW')).size().getInfo())
    print(f'[文通镇] D11 S1 景数={n}')
    s1img = s1_col(roi, D11_START, D11_END)
    frac = ee.Number(s1img.select('VV').mask()
                     .reduceRegion(ee.Reducer.mean(), roi, 10, tileScale=4).get('VV'))
    print(f'[文通镇] VV 有效像元占比={round(frac.getInfo(), 3)}')

    # 2) 23县库 水稻/玉米 S1 区分度（抽样各 300）
    train = gpd.read_file(os.path.join(BASE_DIR, '待训练数据大春', '大春标注_水稻玉米.gpkg'))
    train = train[train['类别'].isin(['水稻', '玉米'])].copy()
    sample = (train.groupby('类别', group_keys=False)
              .apply(lambda d: d.sample(n=min(300, len(d)), random_state=42)))
    sample = sample.reset_index(drop=True)
    sample['fid'] = range(len(sample))
    sample['geometry'] = sample['geometry'].simplify(tolerance=0.0001, preserve_topology=True)
    print(f'[样本] 水稻/玉米 抽样共 {len(sample)} 块：{dict(sample["类别"].value_counts())}')

    fc = ee.FeatureCollection(
        sample[['fid', '类别', 'geometry']].__geo_interface__['features'])
    s1 = s1_col(fc.geometry(), D11_START, D11_END)
    red = s1.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                           scale=10, tileScale=4).select(['fid', '类别', 'VV', 'VH'])
    feats = red.getInfo()['features']
    rows = [{'fid': f['properties']['fid'], '类别': f['properties']['类别'],
             'VV': f['properties'].get('VV'), 'VH': f['properties'].get('VH')}
            for f in feats]
    df = pd.DataFrame(rows)
    df = df[df['VV'].notna()].copy()
    print(f'[样本] 有效 S1 提取 {len(df)}/{len(sample)} 块\n')
    for c in ['VV', 'VH']:
        print(f'--- {c} (dB) 分作物 ---')
        print(df.groupby('类别')[c].agg(['count', 'mean', 'median', 'std']).round(2).to_string())


if __name__ == '__main__':
    main()

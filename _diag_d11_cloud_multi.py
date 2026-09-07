# -*- coding: utf-8 -*-
"""临时诊断：盐亭4乡镇 各旬无云占比对比（判断富驿/莲花湖/大兴是否需两旬模型兜底）。"""
import ee
import geopandas as gpd
from shapely.geometry import mapping

ee.Initialize()

files = {
    '文通镇': r'e:\工作相关\2026年\0624 待测试数据\盐亭_文通镇_地块.shp',
    '富驿镇': r'e:\工作相关\2026年\0624 待测试数据\盐亭_富驿镇_地块.shp',
    '莲花湖乡': r'e:\工作相关\2026年\0624 待测试数据\盐亭_莲花湖乡_地块.shp',
    '大兴回族乡': r'e:\工作相关\2026年\0624 待测试数据\盐亭_大兴回族乡_地块.shp',
}

s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')


def msk(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


wins = [
    ('D04 10天', '2025-05-01', '2025-05-10'),
    ('D11 10天', '2025-07-11', '2025-07-20'),
    ('D11 20天', '2025-07-05', '2025-07-25'),
    ('D14 10天', '2025-08-11', '2025-08-20'),
]

for name, path in files.items():
    g = gpd.read_file(path)
    roi = ee.Geometry(mapping(g.to_crs(4326).geometry.union_all()))
    col = s2.filterBounds(roi)
    print(name)
    for wname, s, e in wins:
        img = col.filterDate(s, e).map(msk).median().select('B8')
        f = img.mask().reduceRegion(ee.Reducer.mean(), roi, 10, tileScale=4).get('B8')
        print('   ', wname, round(ee.Number(f).getInfo(), 3))

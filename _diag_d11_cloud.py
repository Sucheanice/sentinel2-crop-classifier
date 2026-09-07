# -*- coding: utf-8 -*-
"""临时诊断：文通镇 D11(7月中) 分割底图云覆盖占比 + SNIC 覆盖占比。"""
import ee
import geopandas as gpd
from shapely.geometry import mapping

ee.Initialize()

g = gpd.read_file(r'e:\工作相关\2026年\0624 待测试数据\盐亭_文通镇_地块.shp')
roi = ee.Geometry(mapping(g.to_crs(4326).geometry.union_all()))

s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)


def msk(img):
    scl = img.select('SCL')
    cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
    return img.updateMask(cloud.Not())


def m(img, s, e):
    return s2.filterDate(s, e).map(msk).median().select('B8')


def frac(img):
    return ee.Number(img.mask().reduceRegion(ee.Reducer.mean(), roi, 10, tileScale=4).get('B8'))


wins = [
    ('D04 10天 05-01~05-10', '2025-05-01', '2025-05-10'),
    ('D04 20天 04-25~05-15', '2025-04-25', '2025-05-15'),
    ('D11 10天 07-11~07-20', '2025-07-11', '2025-07-20'),
    ('D11 20天 07-05~07-25', '2025-07-05', '2025-07-25'),
    ('D14 10天 08-11~08-20', '2025-08-11', '2025-08-20'),
    ('D14 20天 08-05~08-25', '2025-08-05', '2025-08-25'),
]
for name, s, e in wins:
    print(name, '无云占比:', round(frac(m(None, s, e)).getInfo(), 3))

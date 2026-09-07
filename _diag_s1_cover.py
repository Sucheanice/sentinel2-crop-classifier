# -*- coding: utf-8 -*-
"""临时诊断：文通镇(盐亭) 生长季 S1 覆盖（按月份 + 升降轨统计景数与有效占比）。"""
import ee
import geopandas as gpd
from shapely.geometry import mapping

ee.Initialize()

BASE = r'e:\工作相关\2026年\0624 待测试数据'
g = gpd.read_file(BASE + r'\盐亭_文通镇_地块.shp')
roi = ee.Geometry(mapping(g.to_crs(4326).geometry.union_all()))

MONTHS = [
    ('04月', '2025-04-01', '2025-04-30'),
    ('05月', '2025-05-01', '2025-05-31'),
    ('06月', '2025-06-01', '2025-06-30'),
    ('07月', '2025-07-01', '2025-07-31'),
    ('08月', '2025-08-01', '2025-08-31'),
]

for name, s, e in MONTHS:
    col = (ee.ImageCollection('COPERNICUS/S1_GRD')
           .filterBounds(roi).filterDate(s, e)
           .filter(ee.Filter.eq('instrumentMode', 'IW')))
    n = col.size().getInfo()
    asc = (col.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))).size().getInfo()
    desc = (col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))).size().getInfo()
    print(f'{name} 景数={n} (升轨={asc} 降轨={desc})')

# 打印 7 月所有景的日期 + 轨道 + 极化（确认覆盖细节）
print('\n--- 7月 S1 景明细 ---')
jul = (ee.ImageCollection('COPERNICUS/S1_GRD')
       .filterBounds(roi).filterDate('2025-07-01', '2025-07-31')
       .filter(ee.Filter.eq('instrumentMode', 'IW')))
feats = jul.getInfo()['features']
for f in feats:
    p = f['properties']
    print(f"  {p.get('system:index')[:50]}  pass={p.get('orbitProperties_pass')}  "
          f"pol={p.get('transmitterReceiverPolarisation')}")

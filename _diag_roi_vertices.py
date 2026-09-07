# -*- coding: utf-8 -*-
"""诊断各乡镇地块 union 几何顶点数，确定 GEE payload 是否超限及 simplify 效果"""
import geopandas as gpd
import os
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection

BASE = r"e:\工作相关\2026年\0624 待测试数据"
TOWNS = [
    '梓潼_黎雅镇', '梓潼_石牛镇', '梓潼_金龙镇', '梓潼_玛瑙镇',
    '三台_金石镇', '三台_新鲁镇', '三台_三元镇',
    '江油_新安镇', '江油_太平镇', '江油_马角镇', '江油_河口镇',
]


def count_vertices(geom):
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords) + sum(len(i.coords) for i in geom.interiors)
    if geom.geom_type == 'MultiPolygon':
        return sum(count_vertices(p) for p in geom.geoms)
    if geom.geom_type == 'GeometryCollection':
        return sum(count_vertices(g) for g in geom.geoms)
    return 0


print(f"{'乡镇':<14}{'块数':>7}{'原顶点':>9}{'s0.0001':>9}{'s0.0002':>9}{'s0.0005':>9}")
for t in TOWNS:
    path = os.path.join(BASE, f'{t}_地块.shp')
    if not os.path.exists(path):
        continue
    g = gpd.read_file(path)
    u = g.to_crs(4326).geometry.union_all()
    v0 = count_vertices(u)
    v1 = count_vertices(u.simplify(0.0001, preserve_topology=True))
    v2 = count_vertices(u.simplify(0.0002, preserve_topology=True))
    v5 = count_vertices(u.simplify(0.0005, preserve_topology=True))
    print(f"{t:<14}{len(g):>7}{v0:>9}{v1:>9}{v2:>9}{v5:>9}")

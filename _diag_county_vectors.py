# -*- coding: utf-8 -*-
"""诊断绵阳 5 县县级矢量内容，判断是地块边界还是行政边界"""
import geopandas as gpd
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据\待训练数据绵阳市\绵阳市"
COUNTIES = {
    '梓潼县': os.path.join(BASE, '梓潼县', '矢量', '梓潼县.shp'),
    '盐亭县': os.path.join(BASE, '盐亭县', '矢量', '盐亭县.shp'),
    '三台县': os.path.join(BASE, '三台县', '矢量', '三台县.shp'),
    '平武县': os.path.join(BASE, '平武县', '矢量', '平武县.shp'),
    '江油市': os.path.join(BASE, '江油市', '矢量', '江油市.shp'),
}

for name, path in COUNTIES.items():
    print(f"\n{'='*60}\n{name}: {path}")
    if not os.path.exists(path):
        print("  [缺失] 文件不存在")
        continue
    g = gpd.read_file(path)
    print(f"  要素数: {len(g)}")
    print(f"  CRS: {g.crs}")
    print(f"  几何类型: {g.geom_type.value_counts().to_dict()}")
    print(f"  字段: {list(g.columns)}")
    # 关键字段取值分布
    for c in g.columns:
        if c.lower() in ('geometry', 'fid', 'id', 'objectid', 'shape_leng', 'shape_area'):
            continue
        vals = g[c].astype(str).value_counts().head(6)
        if len(vals) <= 30:
            print(f"    {c}: {dict(vals)}")
        else:
            print(f"    {c}: {len(vals)} 个唯一值，例: {list(vals.index[:4])}")

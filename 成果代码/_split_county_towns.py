# -*- coding: utf-8 -*-
"""按乡镇拆分梓潼/三台/江油三县的地块 shp"""
import geopandas as gpd
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
MBASE = os.path.join(BASE, '待训练数据绵阳市', '绵阳市')
COUNTIES = {
    '梓潼': os.path.join(MBASE, '梓潼县', '矢量', '梓潼县.shp'),
    '三台': os.path.join(MBASE, '三台县', '矢量', '三台县.shp'),
    '江油': os.path.join(MBASE, '江油市', '矢量', '江油市.shp'),
}

for county, path in COUNTIES.items():
    g = gpd.read_file(path)
    print(f'\n{county}: {len(g)} 块, 乡镇={g["XZMC"].value_counts().to_dict()}')
    for xz, sub in g.groupby('XZMC'):
        name = xz.replace(' ', '').replace('\u3000', '')
        out = os.path.join(BASE, f'{county}_{name}_地块.shp')
        sub.to_file(out, encoding='utf-8')
        print(f'  -> {county}_{name}_地块.shp: {len(sub)} 块')

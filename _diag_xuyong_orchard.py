# -*- coding: utf-8 -*-
"""诊断叙永果园标注 + 大春标注库类别，为 12个月特征探索做准备"""
import geopandas as gpd
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
RESULT = os.path.join(BASE, "待训练数据水果", "叙永水果成果数据", "result.shp")
DAHCUN = os.path.join(BASE, "待训练数据大春", "大春训练标注库.gpkg")

print("=== 叙永果园 result.shp ===")
g = gpd.read_file(RESULT)
print(f"块数: {len(g)}")
print(f"CRS: {g.crs}")
print(f"字段: {list(g.columns)}")
for c in g.columns:
    if c != 'geometry':
        vals = g[c].astype(str).value_counts().head(10)
        print(f"  {c}: {dict(vals)}")

print("\n=== 大春训练标注库.gpkg ===")
d = gpd.read_file(DAHCUN)
print(f"块数: {len(d)}")
print(f"字段: {list(d.columns)}")
if '类别' in d.columns:
    print(f"类别分布: {dict(d['类别'].value_counts())}")

# -*- coding: utf-8 -*-
"""清洗绵阳 5 县承保标注 → 水稻/玉米二分类 gpkg（制种水稻归入水稻）。

输出: 待训练数据绵阳市/绵阳标注_水稻玉米.gpkg
字段与 23 县库对齐（保留 ZWMC/QXMC/SZMC/TBMJ 等 + 新增 类别）。
"""
import os
import pandas as pd
import geopandas as gpd

BASE = r"E:\工作相关\2026年\0624 待测试数据\待训练数据绵阳市\绵阳市"
OUT_DIR = r"E:\工作相关\2026年\0624 待测试数据\待训练数据绵阳市"
OUT = os.path.join(OUT_DIR, "绵阳标注_水稻玉米.gpkg")

counties = ["三台县", "平武县", "梓潼县", "江油市", "盐亭县"]
KEEP = {"水稻", "玉米"}

# 5 县都同名存在的核心字段（丢弃 SHAPE_*/OBJECTID/ID 等易冲突冗余字段）
CORE_COLS = ["SJDM", "SJMC", "SZDM", "SZMC", "QXDM", "QXMC", "XZDM", "XZMC",
             "CJDM", "CJMC", "TFBH", "DKBH", "TBMJ", "ZTMC", "ZTDM", "ZWMC",
             "BZ", "JGMC", "JGDM", "BDDM", "TBDDM", "TBJJ", "YEAR_"]

frames = []
for c in counties:
    shp = os.path.join(BASE, c, "矢量", c + ".shp")
    gdf = gpd.read_file(shp)
    # 制种水稻 → 水稻
    gdf['ZWMC'] = gdf['ZWMC'].astype(str).str.strip().replace({"制种水稻": "水稻"})
    n_before = len(gdf)
    gdf = gdf[gdf['ZWMC'].isin(KEEP)].copy()
    gdf['类别'] = gdf['ZWMC']
    gdf = gdf[CORE_COLS + ["类别", "geometry"]].rename(columns={"YEAR_": "YEAR"})
    print(f"{c}: 原始{n_before} → 保留{len(gdf)}")
    frames.append(gdf)

merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
# 与 23 县库对齐的核心字段（保留全部原始字段 + 类别）
merged.to_file(OUT, driver="GPKG", index=False)

print(f"\n合计: {len(merged)} 块")
print(f"类别分布: {dict(merged['ZWMC'].value_counts())}")
print(f"区县分布: {dict(merged['QXMC'].value_counts())}")
print(f"已写出: {OUT}")

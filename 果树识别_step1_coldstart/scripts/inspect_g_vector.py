# -*- coding: utf-8 -*-
"""批量勘察 G 盘汇交矢量：统计每县 作物名称(ZWMC) 与 投保季节(TBJJ) 分布"""
import os, glob
import geopandas as gpd

root = r"G:\20260810保险项目成果数据汇交"
shps = sorted(glob.glob(os.path.join(root, "**", "*.shp"), recursive=True))

for s in shps:
    gdf = None
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            gdf = gpd.read_file(s, encoding=enc)
            break
        except Exception:
            continue
    if gdf is None:
        try:
            gdf = gpd.read_file(s)
        except Exception as e:
            print("READ ERROR:", s, e)
            continue

    zw = gdf["ZWMC"].astype(str).value_counts() if "ZWMC" in gdf.columns else None
    jj = gdf["TBJJ"].astype(str).value_counts() if "TBJJ" in gdf.columns else None
    rel = os.path.relpath(s, root)
    print("=" * 70)
    print(f"{rel}  |  记录数={len(gdf)}")
    if zw is not None:
        print("  作物(ZWMC):", dict(zw))
    if jj is not None:
        print("  季节(TBJJ):", dict(jj))

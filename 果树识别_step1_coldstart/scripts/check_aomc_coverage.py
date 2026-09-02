# -*- coding: utf-8 -*-
"""检查 AOMC 苹果园标签在遂宁/安居区的覆盖情况。"""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

AOMC = r"E:\工作相关\2026年\0624 待测试数据\果树识别_step1_coldstart\data\aomc_sichuan_10m.tif"
ANJU = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\地图属性数据补齐\遂宁市\安居区.shp"

with rasterio.open(AOMC) as src:
    aomc = src.read(1)
    print(f"AOMC缓存: shape={src.shape}, CRS={src.crs}")
    print(f"苹果园像素(>0)总数={int(np.sum(aomc>0)):,}")
    # 分布在哪些类别值
    vals, cnts = np.unique(aomc, return_counts=True)
    for v, c in zip(vals, cnts):
        if v != 0:
            print(f"  值{v}: {c:,} 像素")

    gdf = gpd.read_file(ANJU).to_crs(src.crs)
    mask = rasterize([(g, 1) for g in gdf.geometry],
                     out_shape=src.shape, transform=src.transform,
                     fill=0, dtype="uint8")
    in_anju = mask == 1
    print(f"安居区内像素={int(in_anju.sum()):,}")
    print(f"安居区内苹果园像素={int((in_anju & (aomc>0)).sum()):,}")

# -*- coding: utf-8 -*-
"""检查 S1 dB 数据对安居区的实际覆盖情况。"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol

ANJU = r"E:\工作相关\2026年\0624 待测试数据\待训练数据\地图属性数据补齐\遂宁市\安居区.shp"
S1_VV = r"E:\工作相关\2026年\0624 待测试数据\小春_s1_48RWU\2024-11-21_S1_asc\vv_db.tif"

gdf = gpd.read_file(ANJU)
print(f"安居区 CRS={gdf.crs}, 数量={len(gdf)}")
print(f"安居区 total_bounds (lon/lat): {gdf.total_bounds}")

with rasterio.open(S1_VV) as src:
    print(f"\nS1 CRS={src.crs}, shape={src.shape}")
    gdf_utm = gdf.to_crs(src.crs)
    # 栅格化安居区
    mask = rasterize([(g, 1) for g in gdf_utm.geometry],
                     out_shape=(src.height, src.width),
                     transform=src.transform, fill=0, dtype="uint8")
    data = src.read(1)
    valid = data != src.nodata
    # 安居区内有效像素占比
    in_anju = mask == 1
    n_anju = in_anju.sum()
    n_covered = (in_anju & valid).sum()
    print(f"安居区内总像素={n_anju}, 其中S1有效={n_covered} ({n_covered/n_anju*100:.1f}%)")

    # 按安居区纬度带细分 (用安居区质心纬度)
    gdf_utm["centroid"] = gdf_utm.geometry.centroid
    # 简单: 打印安居区边界经纬度
    gdf_wgs = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf_wgs.total_bounds
    print(f"安居区范围: lon [{minx:.3f},{maxx:.3f}] lat [{miny:.3f},{maxy:.3f}]")

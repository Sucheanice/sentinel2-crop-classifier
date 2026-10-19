# -*- coding: utf-8 -*-
"""vectorize_anju_cropmap.py — 把安居区全图逐像素分类栅格矢量化成 shp（套合 DOM 坐标系）。

输入: dachun_prediction/anju_cropmap_48RWU.tif (UTM 48N, 0=水稻 1=玉米 2=非植被 255=nodata)
输出: dachun_prediction/anju_cropmap.shp (CGCS2000 GK35, 与 人保-安居区DOM.img 同坐标系)
处理: 矢量化水稻/玉米 -> 面积过滤小碎斑 -> 简化边界 -> 转 DOM CRS
"""
import os
import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd
from pyproj import CRS

for _p in [
    r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python312\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python311\site-packages\rasterio\proj_data",
]:
    if os.path.isdir(_p):
        os.environ["PROJ_LIB"] = _p
        break

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_TIF = os.path.join(BASE, "dachun_prediction", "anju_cropmap_48RWU.tif")
DOM_IMG = os.path.join(BASE, "待训练数据", "DOM", "人保-安居区DOM.img")
OUT_SHP = os.path.join(BASE, "dachun_prediction", "anju_cropmap.shp")

MIN_AREA_M2 = 1000.0   # 过滤 < 1000 m² 的小碎斑（10m 栅格 1 像素=100m²）
SIMPLIFY_M = 5.0       # 简化容差 5m


def main():
    # 1) DOM CRS（作为目标坐标系）
    with rasterio.open(DOM_IMG) as dom:
        dom_crs = CRS.from_wkt(dom.crs.to_wkt())
    print(f"DOM CRS: {dom_crs.name}")

    # 2) 读分类栅格
    with rasterio.open(SRC_TIF) as src:
        a = src.read(1)
        transform = src.transform
        src_crs = src.crs
    print(f"栅格: {a.shape}, CRS={src_crs}")

    # 2.1) 形态学平滑：水稻/玉米掩膜做开闭运算，去掉椒盐噪声与碎斑
    from scipy.ndimage import binary_opening, binary_closing
    struct = np.ones((3, 3), dtype=bool)
    rice_mask = binary_closing(binary_opening(a == 0, struct), struct)
    maize_mask = binary_closing(binary_opening(a == 1, struct), struct)
    # 合并：水稻优先，其次玉米，其余为非植被
    smooth = np.full(a.shape, 2, dtype=np.uint8)
    smooth[rice_mask] = 0
    smooth[maize_mask & ~rice_mask] = 1
    a = smooth
    print(f"平滑后: 水稻 {int((a == 0).sum())}px, 玉米 {int((a == 1).sum())}px, "
          f"非植被 {int((a == 2).sum())}px")

    gdfs = []
    for cls_val, cls_name in [(0, "水稻"), (1, "玉米")]:
        mask = (a == cls_val)
        n_px = int(mask.sum())
        print(f"{cls_name}: {n_px} 像素")
        if n_px == 0:
            continue
        polys = []
        for geom, val in rasterio_shapes(a, mask=mask, transform=transform):
            g = shape(geom)
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            polys.append(g)
        # 合并成 GeoSeries
        gs = gpd.GeoSeries(polys, crs=src_crs)
        # 面积过滤（在源 CRS UTM 下，单位米）
        area = gs.area
        keep = area >= MIN_AREA_M2
        gs = gs[keep]
        # 简化
        gs = gs.simplify(SIMPLIFY_M)
        gdf = gpd.GeoDataFrame(geometry=gs, crs=src_crs)
        gdf["crop"] = cls_name
        gdf["cls"] = cls_val
        gdfs.append(gdf)
        print(f"  {cls_name} 矢量化: {len(gdf)} 个多边形")

    if not gdfs:
        print("无有效多边形")
        return

    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=src_crs)
    print(f"合并后: {len(gdf)} 个多边形")

    # 3) 转 DOM CRS
    gdf = gdf.to_crs(dom_crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    # 4) 保存
    gdf.to_file(OUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
    print(f"保存: {OUT_SHP}")
    print("类别分布:")
    print(gdf.groupby("crop").agg(n=("crop", "size"), area_m2=("area_m2", "sum")).to_string())


import pandas as pd

if __name__ == "__main__":
    main()

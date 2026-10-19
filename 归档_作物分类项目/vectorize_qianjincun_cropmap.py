# -*- coding: utf-8 -*-
"""vectorize_qianjincun_cropmap.py — 前进村盲预测栅格矢量化成 shp（套合前进村.img DOM）。"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape
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
SRC_TIF = os.path.join(BASE, "dachun_prediction", "qianjincun_cropmap_48SWA.tif")
DOM_IMG = os.path.join(BASE, "待测试数据前进0806", "20260805", "前进村", "前进村.img")
OUT_SHP = os.path.join(BASE, "dachun_prediction", "qianjincun_cropmap.shp")

MIN_AREA_M2 = 500.0    # 前进村地块小，过滤阈值放低
SIMPLIFY_M = 5.0


def main():
    with rasterio.open(DOM_IMG) as dom:
        dom_crs = CRS.from_wkt(dom.crs.to_wkt())
    print(f"DOM CRS: {dom_crs.name}")

    with rasterio.open(SRC_TIF) as src:
        a = src.read(1)
        transform = src.transform
        src_crs = src.crs
    print(f"栅格: {a.shape}, CRS={src_crs}")

    # 形态学平滑
    from scipy.ndimage import binary_opening, binary_closing
    struct = np.ones((3, 3), dtype=bool)
    rice_mask = binary_closing(binary_opening(a == 0, struct), struct)
    maize_mask = binary_closing(binary_opening(a == 1, struct), struct)
    smooth = np.full(a.shape, 2, dtype=np.uint8)
    smooth[rice_mask] = 0
    smooth[maize_mask & ~rice_mask] = 1
    a = smooth
    print(f"平滑后: 水稻 {int((a==0).sum())}px, 玉米 {int((a==1).sum())}px")

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
        gs = gpd.GeoSeries(polys, crs=src_crs)
        gs = gs[gs.area >= MIN_AREA_M2]
        gs = gs.simplify(SIMPLIFY_M)
        gdf = gpd.GeoDataFrame(geometry=gs, crs=src_crs)
        gdf["crop"] = cls_name
        gdf["cls"] = cls_val
        gdfs.append(gdf)
        print(f"  {cls_name} 矢量化: {len(gdf)} 个多边形")

    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=src_crs)
    gdf = gdf.to_crs(dom_crs)
    gdf["area_m2"] = gdf.geometry.area.round().astype("int64")
    gdf = gdf[gdf["area_m2"] <= 1e7]
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    gdf.to_file(OUT_SHP, driver="ESRI Shapefile", encoding="utf-8")
    print(f"保存: {OUT_SHP}")
    print(dict(gdf["crop"].value_counts()))


if __name__ == "__main__":
    main()

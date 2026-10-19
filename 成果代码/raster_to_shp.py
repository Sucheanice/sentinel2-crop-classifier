# -*- coding: utf-8 -*-
"""raster_to_shp.py — 将预测栅格转SHP地块 (连通域合并版 v2)。
策略: 先用 ndimage.label 做连通域标记，再用 rasterio.features.shapes 一次性
矢量化整张标记图，最后按 label 合并像素→地块。比逐连通域调用 shapes() 快两个数量级。
"""
import os, sys, time
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RASTER_PATH = os.path.join(BASE_DIR, "江油_prediction", "qianjincun_dachun_pixel_pred.tif")
SHP_OUT = os.path.join(BASE_DIR, "江油_prediction", "qianjincun_dachun_parcels.shp")

CLASS_NAMES = {0: '非耕地', 1: '水稻', 2: '玉米', 3: '无法判定', 4: '非植被'}
MIN_PIXELS = 4  # 低于此像素数的连通域丢弃

def main():
    t0 = time.time()
    print("加载栅格...", flush=True)

    with rasterio.open(RASTER_PATH) as src:
        data = src.read(1)
        crs = src.crs
        tf = src.transform

    records = []

    for cls_val, cls_name in CLASS_NAMES.items():
        mask = (data == cls_val)
        n_total = mask.sum()
        if n_total == 0:
            continue

        # 跳过巨大的非耕地背景 (class 0) — 全图幅 12,000 km² 没必要矢量化
        if cls_val == 0:
            print(f"  跳过 {cls_name} ({n_total} px, 背景)", flush=True)
            continue

        print(f"  {cls_name}: {n_total} px → 连通域标记...", flush=True)
        t1 = time.time()

        # Step 1: ndimage.label 连通域标记
        labeled, num_features = ndimage.label(mask)
        print(f"    连通域: {num_features}, {time.time()-t1:.1f}s", flush=True)

        if num_features == 0:
            continue

        # Step 2: 一次性矢量化整张 labeled 图 (每个连通域生成一个多边形)
        # rasterio.features.shapes 会在连通域边界自动切分，高效得多
        t2 = time.time()
        geo_map = {}  # region_id → list of geometries
        for geom_dict, region_id in shapes(labeled.astype(np.int32), mask=labeled > 0, transform=tf):
            rid = int(region_id)
            if rid == 0:
                continue
            g = shape(geom_dict)
            if g.is_empty:
                continue
            g = g.simplify(10, preserve_topology=True)
            if g.is_empty:
                continue
            if rid not in geo_map:
                geo_map[rid] = []
            geo_map[rid].append(g)
        print(f"    矢量化: {len(geo_map)} 个区域, {time.time()-t2:.1f}s", flush=True)

        # Step 3: 合并每个连通域内的碎片并过滤
        for region_id, geoms in geo_map.items():
            from shapely.ops import unary_union
            merged = unary_union(geoms)
            if merged.is_empty:
                continue
            # 处理 MultiPolygon: 拆成多个独立地块
            if merged.geom_type == 'MultiPolygon':
                polys = list(merged.geoms)
            elif merged.geom_type == 'Polygon':
                polys = [merged]
            else:
                continue

            for poly in polys:
                if poly.is_empty:
                    continue
                if poly.area < MIN_PIXELS * 100:  # 100m² ≈ 1个S2像素
                    continue
                records.append({
                    'geometry': poly,
                    'class_id': cls_val,
                    'class_name': cls_name,
                    'area_m2': poly.area,
                })

        print(f"    有效地块: {len(records)} (累计)", flush=True)

    if not records:
        print("ERROR: 无有效地块!", flush=True)
        return

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf = gdf.sort_values('area_m2', ascending=False).reset_index(drop=True)
    gdf.insert(0, 'parcel_id', range(1, len(gdf) + 1))

    gdf_wgs = gdf.to_crs('EPSG:4326')
    gdf_wgs.to_file(SHP_OUT, driver='ESRI Shapefile', encoding='utf-8')

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"完成: {elapsed:.0f}s, {len(gdf)} 个地块")
    for name in ['玉米', '水稻', '无法判定', '非植被']:
        sub = gdf[gdf['class_name'] == name]
        if len(sub):
            print(f"  {name}: {len(sub)} 块, {sub['area_m2'].sum()/666.67:.1f} 亩, 中位{sub['area_m2'].median():.0f}m²")
    print(f"输出: {SHP_OUT}")


if __name__ == '__main__':
    main()

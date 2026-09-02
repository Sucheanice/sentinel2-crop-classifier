# -*- coding: utf-8 -*-
"""将遂宁全境果树SHP裁剪并重投影到安居区DOM坐标系"""
from pathlib import Path
import geopandas as gpd
import rasterio
from shapely.geometry import box
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHP_IN = ROOT / "outputs" / "shapefile_xuyong_model" / "orchard_patches.shp"
DOM_PATH = Path(r"E:\工作相关\2026年\0624 待测试数据\待训练数据\DOM\人保-安居区DOM.img")
OUT_DIR = ROOT / "outputs" / "anju"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 读取DOM的CRS和范围
with rasterio.open(DOM_PATH) as src:
    dom_crs = src.crs
    dom_bounds = src.bounds
print(f"DOM CRS: {dom_crs}")
print(f"DOM bounds: {dom_bounds}")

# 2. 加载全境果树SHP (UTM 48N)
gdf = gpd.read_file(SHP_IN)
print(f"全境果树斑块: {len(gdf)} (CRS: {gdf.crs})")

# 3. 重投影到DOM坐标系
gdf_dom = gdf.to_crs(dom_crs)
print(f"重投影完成 (CRS: {gdf_dom.crs})")

# 4. 裁剪到DOM范围
dom_box = box(dom_bounds.left, dom_bounds.bottom, dom_bounds.right, dom_bounds.top)
gdf_clip = gdf_dom[gdf_dom.intersects(dom_box)].copy()
print(f"与DOM有交集: {len(gdf_clip)} 个斑块")

# 裁剪到DOM精确边界
gdf_clip["geometry"] = gdf_clip.intersection(dom_box)
gdf_clip = gdf_clip[~gdf_clip.is_empty]

# 重新计算面积 (m²)
gdf_clip["area_m2"] = gdf_clip.area
gdf_clip["area_mu"] = gdf_clip["area_m2"] / 666.667

print(f"裁剪后有效斑块: {len(gdf_clip)}")

# 5. 输出
out_shp = OUT_DIR / "orchard_anju_dom_v4.shp"
gdf_clip.to_file(out_shp, encoding="utf-8")
print(f"输出: {out_shp}")

# 统计
total_ha = gdf_clip["area_m2"].sum() / 10000
total_mu = gdf_clip["area_mu"].sum()
avg_ha = gdf_clip["area_m2"].mean() / 10000
print(f"\n=== 安居区果树识别统计 ===")
print(f"斑块数: {len(gdf_clip)}")
print(f"总面积: {total_ha:.0f} 公顷 ({total_mu:.0f} 亩)")
print(f"平均斑块: {avg_ha:.2f} 公顷")
print(f"最大斑块: {gdf_clip['area_m2'].max()/10000:.1f} 公顷")
print(f"最小斑块: {gdf_clip['area_m2'].min():.0f} m²")

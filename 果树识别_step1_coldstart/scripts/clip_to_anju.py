# -*- coding: utf-8 -*-
"""将新版 orchard_patches.shp (LightGBM 4季) 裁切到安居区范围"""
import geopandas as gpd
from shapely.ops import unary_union
from pathlib import Path

ROOT = Path(r"E:\工作相关\2026年\0624 待测试数据")
SHP_IN = ROOT / "果树识别_step1_coldstart" / "outputs" / "shapefile_xuyong_model" / "orchard_patches.shp"
ANJU = ROOT / "待训练数据" / "地图属性数据补齐" / "遂宁市" / "安居区.shp"
OUT_DIR = ROOT / "果树识别_step1_coldstart" / "outputs" / "anju"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. 安居区边界 (WGS84)
gdf_anju = gpd.read_file(ANJU)
anju_union = unary_union(gdf_anju.geometry.tolist())
print(f"安居区: {len(gdf_anju)} 地块, CRS={gdf_anju.crs}")

# 2. 果园斑块 (UTM 48N)
gdf_orchard = gpd.read_file(SHP_IN)
print(f"果园斑块(全遂宁): {len(gdf_orchard)}, CRS={gdf_orchard.crs}")

# 3. 安居区边界转 UTM 48N 再裁切
anju_utm = gpd.GeoDataFrame(geometry=[anju_union], crs=gdf_anju.crs).to_crs(gdf_orchard.crs)
anju_boundary = anju_utm.geometry.iloc[0]

# 用边界做空间裁剪（保留完整斑块，不做 intersect 切碎）
gdf_clip = gdf_orchard[gdf_orchard.intersects(anju_boundary)].copy()
print(f"与安居区相交: {len(gdf_clip)} 个斑块")

# 统计
gdf_clip["area_m2"] = gdf_clip.area
gdf_clip["area_mu"] = gdf_clip["area_m2"] / 666.667
total_ha = gdf_clip["area_m2"].sum() / 10000
print(f"总面积: {total_ha:.0f} 公顷 ({gdf_clip['area_mu'].sum():.0f} 亩)")

# 4. 输出 (保持 UTM 48N, 也可转 WGS84)
out_shp = OUT_DIR / "orchard_patches_anju.shp"
gdf_clip.to_file(out_shp, encoding="utf-8")
print(f"输出: {out_shp}")
print(f"斑块数: {len(gdf_clip)}")

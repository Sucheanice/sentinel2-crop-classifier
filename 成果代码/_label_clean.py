# -*- coding: utf-8 -*-
"""标签清洗：从大春训练标注库中只保留「水稻 + 玉米」二分类地块。"""
import os

for _p in [
    r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python312\site-packages\rasterio\proj_data",
    r"C:\Users\lenovo\AppData\Roaming\Python\Python311\site-packages\rasterio\proj_data",
]:
    if os.path.isdir(_p):
        os.environ.setdefault("PROJ_LIB", _p)
        break

import geopandas as gpd

SRC = r"e:\工作相关\2026年\0624 待测试数据\大春训练标注库_完整字段.gpkg"
DST = r"e:\工作相关\2026年\0624 待测试数据\待训练数据大春\大春标注_水稻玉米.gpkg"

gdf = gpd.read_file(SRC)
print("CRS:", gdf.crs)
print("行数:", len(gdf))
print("列名:", list(gdf.columns))
print()

# 找出类别字段：唯一值严格落在 {玉米,水稻,高粱,果树,经济林,蔬菜} 内、且至少含玉米+水稻
CATEGORIES = {"玉米", "水稻", "高粱", "果树", "经济林", "蔬菜"}
cat_col = None
for c in gdf.columns:
    if gdf[c].dtype == object:
        vals = set(str(v).strip() for v in gdf[c].unique())
        if {"玉米", "水稻"} <= vals and vals <= CATEGORIES:
            cat_col = c
            break
print("类别字段判定为:", cat_col)
if cat_col:
    print(gdf[cat_col].value_counts().to_string())
    print()

    # 只保留 水稻 + 玉米
    keep_set = {"水稻", "玉米"}
    mask = gdf[cat_col].astype(str).str.strip().isin(keep_set)
    gdf_clean = gdf[mask].copy()
    print(f"清洗后（水稻+玉米）: {len(gdf_clean)} 块（原 {len(gdf)} 块）")
    print(gdf_clean[cat_col].value_counts().to_string())

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    gdf_clean.to_file(DST, driver="GPKG")
    print("\n已保存:", DST)
else:
    print("未自动识别到类别字段，请人工确认。")

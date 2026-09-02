# -*- coding: utf-8 -*-
"""合并 G 盘汇交的大春矢量 → 统一大春训练标注库（按作物类型归类 + 精简字段）"""
import os, glob
import geopandas as gpd
import pandas as pd

root = r"G:\20260810保险项目成果数据汇交"
out_dir = r"E:\工作相关\2026年\0624 待测试数据\待训练数据大春"
os.makedirs(out_dir, exist_ok=True)

# 作物名称 -> 标准化类别
CATEGORY_MAP = {
    "玉米": "玉米", "水稻": "水稻", "高粱": "高粱",
    "李子": "果树", "柑橘": "果树", "桂圆": "果树", "梨": "果树",
    "柠檬": "果树", "荔枝": "果树", "桃子": "果树", "橘子": "果树",
    "柑": "果树", "枇杷": "果树", "苹果": "果树",
    "花椒": "经济林", "核桃": "经济林",
    "生姜": "蔬菜", "蔬菜": "蔬菜",
}

shps = sorted(glob.glob(os.path.join(root, "**", "*.shp"), recursive=True))
frames = []

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
            print("SKIP", s, e)
            continue

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    def col(name):
        if name in gdf.columns:
            return gdf[name].astype(str).str.strip()
        return pd.Series([""] * len(gdf), index=gdf.index)

    zw = col("ZWMC")
    cat = zw.map(CATEGORY_MAP).fillna("其他")
    area = pd.to_numeric(col("TBMJ"), errors="coerce")

    out = gpd.GeoDataFrame({
        "类别": cat,
        "作物": zw,
        "市": col("SZMC"),
        "区县": col("QXMC"),
        "季节": col("TBJJ"),
        "面积": area,
        "geometry": gdf.geometry.values,
    }, crs="EPSG:4326")
    frames.append(out)
    print(f"{os.path.relpath(s, root)}: {len(out)} 块")

merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
print("=" * 60)
print("总计:", len(merged), "块")
print("类别分布:", dict(merged["类别"].value_counts()))

# 主文件：GeoPackage（中文字段名，无编码问题）
gpkg = os.path.join(out_dir, "大春训练标注库.gpkg")
merged.to_file(gpkg, driver="GPKG")
print("已写出:", gpkg)

# 兼容版：Shapefile（英文字段名，dbf 字段名<=10字符）
merged_en = merged.rename(columns={
    "类别": "category", "作物": "crop", "市": "city",
    "区县": "county", "季节": "season", "面积": "area",
})
shp = os.path.join(out_dir, "大春训练标注库.shp")
merged_en.to_file(shp, encoding="utf-8")
print("已写出:", shp)

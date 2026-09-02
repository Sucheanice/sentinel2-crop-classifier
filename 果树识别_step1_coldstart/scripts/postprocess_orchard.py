# -*- coding: utf-8 -*-
"""
果园识别后处理 (v3.1):
  对已有的 LightGBM 概率栅格做三道后处理, 解决 v3 过度识别问题:
    ① NDVI 校验: 夏季 NDVI < 阈值 → 强制非果园 (果树常绿 vs 大春已收获)
    ② 连通域去噪: binary_opening 去孤立点 + 移除 < 最小面积碎斑
    ③ 置信度分级: prob ≥ 高阈值 → 高置信 / 其余 → 低置信(不确定)
  矢量化输出带字段 SHP, 并裁切安居区。

不重新训练、不重新全图推理, 只读已有概率栅格 + 夏季 NDVI。
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(r"E:\工作相关\2026年\0624 待测试数据")
PROJ = ROOT / "果树识别_step1_coldstart"
DATA = PROJ / "data" / "suining_features"

PROB_PATH = PROJ / "outputs" / "orchard_xuyong_model_suining_prob.tif"
CUBE_PATH = DATA / "feature_cube.npy"
BAND_NAMES_PATH = DATA / "band_names.json"
TRANSFORM_PATH = DATA / "transform.json"

ANJU_PATH = ROOT / "待训练数据" / "地图属性数据补齐" / "遂宁市" / "安居区.shp"

OUT_FULL = PROJ / "outputs" / "shapefile_xuyong_model" / "orchard_patches_post.shp"
OUT_ANJU = PROJ / "outputs" / "anju" / "orchard_patches_anju_post.shp"

# ============ 可调参数 ============
NDVI_THRESHOLD = 0.45      # 夏季 NDVI 低于此值 → 强制非果园
PROB_THRESHOLD = 0.5       # 二值化概率阈值
HIGH_CONF = 0.7            # 高置信度概率阈值
MIN_AREA_HA = 0.3          # 最小斑块面积 (公顷)
MIN_AREA_PX = int(MIN_AREA_HA * 10000 / 100)  # 10m 像素 = 100 m², 0.3ha = 30 px
# =================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("果园识别后处理 (NDVI校验 + 连通域去噪 + 置信度分级)")
    log("=" * 60)

    # 1. 加载概率栅格
    with rasterio.open(PROB_PATH) as src:
        prob = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
    H, W = prob.shape
    log(f"概率栅格: {H}x{W}, CRS={crs}")

    # 2. 加载夏季 NDVI (索引 48)
    with open(BAND_NAMES_PATH) as f:
        band_names = json.load(f)
    ndvi_idx = band_names.index("vi_summer_ndvi")
    log(f"夏季 NDVI 波段索引: {ndvi_idx} ({band_names[ndvi_idx]})")
    cube = np.load(CUBE_PATH, mmap_mode="r")
    summer_ndvi = np.array(cube[ndvi_idx]).astype("float32")
    del cube

    # 3. NDVI 校验
    prob_corr = prob.copy()
    valid_ndvi = ~np.isnan(summer_ndvi)
    low_ndvi = valid_ndvi & (summer_ndvi < NDVI_THRESHOLD)
    n_low = int(low_ndvi.sum())
    prob_corr[low_ndvi] = 0.0
    log(f"① NDVI校验: {n_low:,} 像素 (夏季NDVI<{NDVI_THRESHOLD}) 被强制为非果园 "
        f"({n_low/prob.size*100:.1f}%)")

    # 4. 二值化
    binary = (prob_corr >= PROB_THRESHOLD).astype("uint8")
    n_before = int(binary.sum())
    log(f"② 二值化 (prob≥{PROB_THRESHOLD}): {n_before:,} 像素 ({n_before/prob.size*100:.2f}%)")

    # 5. 连通域去噪
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3))).astype("uint8")
    labels, n_lbl = ndimage.label(binary, structure=np.ones((3, 3)))
    log(f"   binary_opening + label: {n_lbl} 个连通域")

    counts = np.bincount(labels.ravel())
    small_labels = np.where(counts < MIN_AREA_PX)[0]
    small_mask = np.isin(labels, small_labels)
    binary[small_mask] = 0
    n_after = int(binary.sum())
    log(f"   移除 <{MIN_AREA_HA}ha 碎斑: {n_before - n_after:,} 像素被去除, 剩 {n_after:,} 像素 "
        f"({n_after/prob.size*100:.3f}%)")

    # 重新 label
    labels, n_lbl = ndimage.label(binary, structure=np.ones((3, 3)))
    log(f"   最终连通域: {n_lbl} 个")

    if n_lbl == 0:
        log("[FATAL] 无果园斑块")
        return

    # 6. zonal 统计 (numpy bincount 快速聚合)
    flat_labels = labels.ravel()
    cnt = np.bincount(flat_labels, minlength=n_lbl + 1)
    sum_prob = np.bincount(flat_labels, weights=prob_corr.ravel(), minlength=n_lbl + 1)
    sum_ndvi = np.bincount(flat_labels, weights=np.nan_to_num(summer_ndvi, nan=0.0).ravel(), minlength=n_lbl + 1)
    max_prob = np.zeros(n_lbl + 1, dtype="float32")
    np.maximum.at(max_prob, flat_labels, prob_corr.ravel())
    log("   zonal 统计完成")

    # 7. 矢量化
    geoms = []
    for geom, value in shapes(labels.astype("int32"), transform=transform):
        if value == 0:
            continue
        lbl = int(value)
        c = cnt[lbl]
        if c == 0:
            continue
        area_m2 = c * 100.0
        mean_prob = float(sum_prob[lbl] / c)
        mean_ndvi = float(sum_ndvi[lbl] / c)
        max_p = float(max_prob[lbl])
        conf = "高" if max_p >= HIGH_CONF else "低"
        geoms.append({
            "geometry": shape(geom),
            "area_ha": area_m2 / 10000.0,
            "area_m2": area_m2,
            "area_mu": area_m2 / 666.667,
            "max_prob": max_p,
            "mean_prob": mean_prob,
            "ndvi_summer": mean_ndvi,
            "conf_flag": conf,
            "prdct_lbl": "果园",
        })
    log(f"   矢量化: {len(geoms)} 个斑块")

    if not geoms:
        log("[FATAL] 无斑块")
        return

    gdf = gpd.GeoDataFrame(geoms, crs=crs)
    total_ha = gdf["area_ha"].sum()
    n_high = int((gdf["conf_flag"] == "高").sum())
    log(f"   总面积: {total_ha:.0f} ha, 高置信 {n_high} 个, 低置信 {len(gdf) - n_high} 个")

    # 8. 输出全遂宁
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT_FULL, encoding="utf-8")
    log(f"   全遂宁 SHP -> {OUT_FULL}")

    # 9. 裁切安居区
    gdf_anju = gpd.read_file(ANJU_PATH)
    anju_union = unary_union(gdf_anju.geometry.tolist())
    anju_utm = gpd.GeoDataFrame(geometry=[anju_union], crs=gdf_anju.crs).to_crs(crs)
    boundary = anju_utm.geometry.iloc[0]
    gdf_anju_clip = gdf[gdf.intersects(boundary)].copy()
    anju_ha = gdf_anju_clip["area_ha"].sum()
    n_high_a = int((gdf_anju_clip["conf_flag"] == "高").sum())
    gdf_anju_clip.to_file(OUT_ANJU, encoding="utf-8")
    log(f"   安居区 SHP -> {OUT_ANJU} ({len(gdf_anju_clip)} 斑块, {anju_ha:.0f} ha, 高置信 {n_high_a})")

    # 10. 汇总
    log("=" * 60)
    log(f"完成: {time.time()-t0:.0f}s")
    log(f"全遂宁: {len(gdf)} 斑块, {total_ha:.0f} ha, 高置信 {n_high} ({n_high/len(gdf)*100:.1f}%)")
    log(f"安居区: {len(gdf_anju_clip)} 斑块, {anju_ha:.0f} ha, 高置信 {n_high_a} ({n_high_a/max(len(gdf_anju_clip),1)*100:.1f}%)")

if __name__ == "__main__":
    main()

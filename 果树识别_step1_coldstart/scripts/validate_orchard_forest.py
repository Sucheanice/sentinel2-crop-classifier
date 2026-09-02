# -*- coding: utf-8 -*-
"""
果园 vs 林地 验证脚本
=====================
对新模型推理结果做交叉验证:
  1. 果园面积对比 (旧 350625 ha vs 新)
  2. WorldCover 类别交叉: 果园结果落在 Tree cover(10) / Cropland(40) / 其他 的分布
     (用于判断"林地误报"是否被抑制)
"""
from __future__ import annotations

import os
os.environ["PROJ_LIB"] = r"C:\Users\lenovo\AppData\Roaming\Python\Python313\site-packages\rasterio\proj_data"

import json
import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds as window_from_bounds

PROJ_DIR = r"E:\工作相关\2026年\0624 待测试数据\果树识别_step1_coldstart"
OUTPUT_DIR = PROJ_DIR + r"\outputs"
SN_WC = r"E:\工作相关\2026年\0624 待测试数据\ESA_WorldCover_10m_2021_v200_N30E105_Map.tif"
NEW_PRED = OUTPUT_DIR + r"\orchard_vs_forest_suining.tif"

WC_NAMES = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop",
            50: "Built", 60: "Bare", 70: "Snow", 80: "Water", 90: "Wetland"}


def main():
    # 1. 读新模型推理结果
    with rasterio.open(NEW_PRED) as src:
        pred = src.read(1)
        transform = src.transform
        crs = src.crs
    H, W = pred.shape
    print(f"推理栅格: {H}x{W}, CRS={crs}")
    orchard_px = int((pred == 1).sum())
    orchard_ha = orchard_px * 100 / 10000
    print(f"新模型果园: {orchard_px:,} px = {orchard_ha:,.0f} ha")
    old_ha = 350625.4
    print(f"旧模型果园: {old_ha:,.0f} ha")
    print(f"面积比: {orchard_ha/old_ha:.4f} (新/旧)")

    # 2. WorldCover 交叉
    # 遂宁推理范围的 WGS84 bounds
    bbox_wgs = transform_bounds(crs, "EPSG:4326",
                                transform.c, transform.f + transform.e * H,
                                transform.c + transform.a * W, transform.f)
    print(f"推理范围 WGS84: {bbox_wgs}")

    wc_out = np.zeros((H, W), dtype="uint8")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(SN_WC) as src:
            win = window_from_bounds(*bbox_wgs, transform=src.transform)
            win = win.round_lengths().round_offsets()
            wc_sub = src.read(1, window=win)
            win_transform = src.window_transform(win)
            reproject(source=wc_sub, destination=wc_out,
                      src_transform=win_transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest)
    print(f"WorldCover 已重投影到推理网格")

    # 3. 果园像元在 WorldCover 各类别的分布
    orchard_mask = pred == 1
    wc_in_orchard = wc_out[orchard_mask]
    total = orchard_mask.sum()
    print(f"\n果园结果落在 WorldCover 各类别的分布 (共 {total:,} px):")
    from collections import Counter
    cnt = Counter(wc_in_orchard.tolist())
    for cls in sorted(cnt, key=lambda x: -cnt[x]):
        name = WC_NAMES.get(cls, f"cls{cls}")
        print(f"  {cls:3d} {name:8s}: {cnt[cls]:>10,} px ({cnt[cls]/total*100:.1f}%)")

    # 4. 全图 WorldCover 各类别被"误判为果园"的比例 (林地误报率)
    print(f"\n全图 WorldCover 各类别被判成果园的比例:")
    for cls in [10, 40, 30, 20, 50]:
        cls_mask = wc_out == cls
        cls_total = int(cls_mask.sum())
        if cls_total == 0:
            continue
        cls_orchard = int((cls_mask & orchard_mask).sum())
        name = WC_NAMES.get(cls, f"cls{cls}")
        print(f"  {cls:3d} {name:8s}: 共 {cls_total:>10,} px, 其中 {cls_orchard:>9,} 判成果园 ({cls_orchard/cls_total*100:.2f}%)")

    report = {
        "old_orchard_ha": old_ha,
        "new_orchard_ha": round(orchard_ha, 1),
        "ratio": round(orchard_ha / old_ha, 4),
        "orchard_by_worldcover_class": {str(k): v for k, v in cnt.items()},
    }
    with open(OUTPUT_DIR + r"\validate_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n验证报告: validate_report.json")


if __name__ == "__main__":
    main()

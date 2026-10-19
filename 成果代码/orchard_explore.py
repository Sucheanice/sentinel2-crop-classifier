#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""orchard_explore.py — 果园识别·无标注探索

对每个产区采样若干 10m 像元，提取 **2025 全年 36 旬** 的 NDVI / EVI 时序，
用于观察"落叶果树 vs 常绿果树"的物候差异（RGB 单期影像做不到的事）。

输出：
  - 各产区 NDVI/EVI 旬序列 CSV（逐点）
  - 各产区均值曲线（含 P25~P75 带）
  - 6 产区对照曲线图
  - 对逐点曲线做 KMeans 聚类，看自然分群是否对应物候类型

用法：
  python orchard_explore.py --out 输出目录 [--pixels 1500] [--year 2025]
"""
import argparse
import calendar
import json
import os
import sys

import ee
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ee.Initialize()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
# SCL: 3=云影 8=中概率云 9=高概率云 10=薄卷云 11=雪
CLOUD_SCL = [3, 8, 9, 10, 11]
BANDS = ["B2", "B3", "B4", "B8", "B11"]

# 产区定义（lon, lat 中心；半边长 ~0.011° ≈ 1.1km，即约 2.2×2.4 km 的小样区）
AREAS = [
    ("桃_龙泉驿山泉镇", 104.320, 30.540, "落叶"),
    ("柑橘_蒲江西来镇", 103.453, 30.237, "常绿"),
    ("石榴_会理彰冠镇", 102.290, 26.720, "落叶"),
    ("苹果_盐源卫城镇", 101.550, 27.470, "落叶"),
    ("芒果_米易丙谷镇", 102.110, 26.890, "热带常绿"),
    ("柚_广安龙安乡", 106.640, 30.470, "常绿"),
]
HALF = 0.011


def dekad_list(year):
    out = []
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        for k, (d0, d1) in enumerate([(1, 10), (11, 20), (21, last)], 1):
            out.append((f"{year}{m:02d}{k}", f"{year}-{m:02d}-{d0:02d}",
                        f"{year}-{m:02d}-{d1:02d}T23:59:59"))
    return out


def mask_s2(img):
    scl = img.select("SCL")
    m = None
    for c in CLOUD_SCL:
        m = scl.eq(c) if m is None else m.Or(scl.eq(c))
    return img.updateMask(m.Not())


def build_named_stack(aoi, year):
    """返回 (36 旬 × [NDVI, EVI, NDWI] 的堆叠影像, 旬标签列表)"""
    dek = dekad_list(year)
    s2 = ee.ImageCollection(S2).filterBounds(aoi)
    imgs, names = [], []
    for name, st, en in dek:
        col = s2.filterDate(st, en).map(mask_s2).select(BANDS)
        comp = col.median()
        ndvi = comp.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{name}")
        evi = comp.expression(
            "2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": comp.select("B8"),
                                          "R": comp.select("B4"),
                                          "B": comp.select("B2")}).rename(f"EVI_{name}")
        ndwi = comp.normalizedDifference(["B3", "B8"]).rename(f"NDWI_{name}")
        imgs += [ndvi, evi, ndwi]
        names.append(name)
    return ee.Image.cat(imgs), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pixels", type=int, default=1500)
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    all_frames, mats = [], {}
    for label, lon, lat, kind in AREAS:
        aoi = ee.Geometry.Rectangle([lon - HALF, lat - HALF, lon + HALF, lat + HALF],
                                    proj="EPSG:4326", geodesic=False)
        stack, names = build_named_stack(aoi, a.year)
        print(f"[{label}] 采样 {a.pixels} 点 …", flush=True)
        fc = stack.sample(region=aoi, scale=10, numPixels=a.pixels, seed=42,
                          geometries=True, tileScale=4, dropNulls=False)
        feats = fc.getInfo()["features"]
        rows = []
        for f in feats:
            p = f["properties"]
            g = f.get("geometry")
            r = {"area": label, "kind": kind}
            if g:
                r["lon"], r["lat"] = g["coordinates"][0], g["coordinates"][1]
            for pre in ["NDVI", "EVI", "NDWI"]:
                for nm in names:
                    r[f"{pre}_{nm}"] = p.get(f"{pre}_{nm}")
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, f"旬序列_{label}.csv"), index=False,
                  encoding="utf-8-sig")
        print(f"    得到 {len(df)} 点；NDVI 有效值 {int(df.filter(like='NDVI_').notna().any(axis=1).sum())} 点")
        all_frames.append(df)
        mats[label] = df

    allf = pd.concat(all_frames, ignore_index=True)
    allf.to_csv(os.path.join(a.out, "旬序列_全部产区.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out, "旬标签.json"), "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)
    print(f"\n总 {len(allf)} 点，写出 {a.out}")


if __name__ == "__main__":
    main()

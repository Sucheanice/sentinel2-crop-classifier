#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""orchard_supervised.py — 果园识别·有监督验证（江油 244 块调查果园 vs 同镇耕地）

用**真实果园边界**（江油市资源调查地块）作正样本，同镇交付耕地作负样本，
提 2025 全年 36 旬 NDVI（地块级 zonal mean），检验：

  1. 果园与「判为玉米的耕地」在物候上能否分开；
  2. 「冬季 NDVI 绝对值」判据对果园的有效性；
  3. 果园的落叶 / 常绿构成。

用法：
  python orchard_supervised.py --out 输出目录 [--year 2025] [--per-group 300]
"""
import argparse
import calendar
import os
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ee.Initialize()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCL = [3, 8, 9, 10, 11]
BANDS = ["B2", "B3", "B4", "B8", "B11"]
CRS = "EPSG:4523"
MU = 0.0015
BATCH = 400

ROOT = r"e:\工作相关\2026年\0624 待测试数据"
ORCH = os.path.join(ROOT, r"训练与标注数据\江油市资源调查地块_按镇\江油市果园地块.shp")
OUTR = os.path.join(ROOT, r"成果数据\1019 第二版前序数据3\我们的赋值\JY_OUT")
# 1019 江油镇编码：太平 510781101 / 新安 510781109 / 河口 510781114 / 马角 510781118
AOIS = ["510781101", "510781109", "510781114", "510781118"]


def dekad_list(year):
    out = []
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        for k, (d0, d1) in enumerate([(1, 10), (11, 20), (21, last)], 1):
            out.append((f"{m:02d}{k}", f"{year}-{m:02d}-{d0:02d}", f"{year}-{m:02d}-{d1:02d}T23:59:59"))
    return out


def mask_s2(img):
    scl = img.select("SCL")
    m = None
    for c in CLOUD_SCL:
        m = scl.eq(c) if m is None else m.Or(scl.eq(c))
    return img.updateMask(m.Not())


def build_ndvi_stack(aoi, year):
    s2 = ee.ImageCollection(S2).filterBounds(aoi)
    imgs, names = [], []
    for name, st, en in dekad_list(year):
        comp = s2.filterDate(st, en).map(mask_s2).select(BANDS).median()
        imgs.append(comp.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{name}"))
        names.append(name)
    return ee.Image.cat(imgs), names


def extract(gdf_wgs84, names, tag, year):
    """分批 reduceRegions 取地块级 NDVI 旬均值（含有效像元计数）"""
    rows = []
    n = len(gdf_wgs84)
    nb = (n + BATCH - 1) // BATCH
    for bi in range(nb):
        sub = gdf_wgs84.iloc[bi * BATCH:(bi + 1) * BATCH]
        fc = ee.FeatureCollection(sub[["fid", "geometry"]].__geo_interface__["features"])
        stack, _ = build_ndvi_stack(fc.geometry(), year)
        bands = [f"NDVI_{x}" for x in names]
        red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                  scale=10, tileScale=4).select(["fid"] + bands)
        # 覆盖像元数：36 旬掩膜的并集（任一旬有效即算）
        cnt = stack.mask().reduce(ee.Reducer.max()).reduceRegions(
            collection=fc, reducer=ee.Reducer.sum().unweighted(),
            scale=10, tileScale=4)
        feats = red.getInfo()["features"]
        cnts = {f["properties"]["fid"]: f["properties"].get("sum", 0)
                for f in cnt.getInfo()["features"]}
        for f in feats:
            p = f["properties"]
            r = {"fid": p["fid"], "n_px": cnts.get(p["fid"], 0)}
            r.update({f"NDVI_{x}": p.get(f"NDVI_{x}") for x in names})
            rows.append(r)
        print(f"    [{tag}] batch {bi+1}/{nb} → {len(rows)}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--per-group", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---- 正样本：调查果园 ----
    orch = gpd.read_file(ORCH).to_crs(CRS).reset_index(drop=True)
    orch["fid"] = ["OR%03d" % i for i in range(len(orch))]
    orch["group"] = "果园(调查)"
    print(f"果园 {len(orch)} 块 / {orch.geometry.area.sum()*MU:.1f} 亩")

    # ---- 负样本：同四镇交付耕地，按作物分层 ----
    parts = []
    for x in AOIS:
        f = [y for y in os.listdir(os.path.join(OUTR, x, "交付")) if y.endswith("_交付.shp")][0]
        parts.append(gpd.read_file(os.path.join(OUTR, x, "交付", f)).to_crs(CRS))
    fa = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=CRS)
    fa = fa[fa.AG_BZBM == "A01"].copy()
    fa["group"] = np.where(fa.CROP_BZMC == "玉米", "耕地_玉米",
                           np.where(fa.CROP_BZMC == "水稻", "耕地_水稻", "耕地_未判作物"))
    print("对照候选:", dict(fa.group.value_counts()))

    picks = []
    for g, sub in fa.groupby("group"):
        k = min(a.per_group, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), k, replace=False)])
    neg = pd.concat(picks, ignore_index=True)
    neg["fid"] = ["FA%04d" % i for i in range(len(neg))]
    print("抽取对照:", dict(neg.group.value_counts()))

    allp = pd.concat([orch[["fid", "group", "town", "crop", "geometry"]],
                      neg[["fid", "group", "geometry"]]], ignore_index=True)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=CRS).to_crs("EPSG:4326")
    allp["geometry"] = allp.geometry.simplify(tolerance=0.0001, preserve_topology=True)

    names = [d[0] for d in dekad_list(a.year)]
    print(f"\n[GEE] 提取 {len(allp)} 块 × {len(names)} 旬 NDVI …")
    df = extract(allp, names, "all", a.year)

    meta = allp.drop(columns="geometry")[["fid", "group"]].copy()
    if "town" in allp.columns:
        meta = meta.join(allp[["town", "crop"]])
    df = df.merge(meta, on="fid", how="left")

    out_csv = os.path.join(a.out, f"果园监督对照_{a.year}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[输出] {out_csv}")

    # ---- 季节汇总 ----
    nd = [f"NDVI_{x}" for x in names]
    m2c = {x: int(x[:2]) for x in names}
    seas = {"冬季(12,1,2)": [12, 1, 2], "春季(3,4,5)": [3, 4, 5],
            "夏季(6,7,8)": [6, 7, 8], "秋季(9,10,11)": [9, 10, 11]}
    for lab, ms in seas.items():
        cols = [f"NDVI_{x}" for x in names if m2c[x] in ms]
        df[lab] = df[cols].mean(axis=1, skipna=True)
    df["振幅"] = df["夏季(6,7,8)"] - df["冬季(12,1,2)"]
    df["有效旬数"] = df[nd].notna().sum(axis=1)

    print("\n=== 各组分季节 NDVI（中位）===")
    t = df.groupby("group").agg(
        块数=("fid", "size"),
        有效旬数中位=("有效旬数", "median"),
        冬季=("冬季(12,1,2)", "median"), 春季=("春季(3,4,5)", "median"),
        夏季=("夏季(6,7,8)", "median"), 秋季=("秋季(9,10,11)", "median"),
        振幅=("振幅", "median")).round(3)
    print(t.to_string())

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[重写含季节列] {out_csv}")


if __name__ == "__main__":
    main()

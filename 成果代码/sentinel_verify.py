#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sentinel_verify.py — 用哨兵（S2 10m）给 SAM 地块做「是不是耕地」判定与判水。

思路（对象级，不逐像元）：
  地块 shp → GEE 取 S2 关键3旬 × 10 波段 zonal mean（S2_SR_HARMONIZED + SCL 云掩膜 + median）
           → compute_feature_matrix（155 维，含 NDVI/EVI/NDWI/MNDWI/LSWI 时序 + 物候）
           → cropland_filter_general.pkl 判「作物 / 非作物」
           → 同时算 MNDWI/NDWI 判水
云覆盖/过小地块 → 用加宽 20 天窗口补提一次（WIDE_WINDOWS）

用法：
  python sentinel_verify.py --input 地块.shp --out 输出目录 \
      [--model cropland_filter_general.pkl] [--wide-fallback 1]
"""
import argparse
import os
import pickle
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

BASE = r"e:\工作相关\2026年\0624 待测试数据\归档_作物分类项目"
sys.path.insert(0, BASE)
sys.stdout.reconfigure(encoding="utf-8")

from common import compute_feature_matrix          # noqa: E402
import predict_dachun_gee as P                      # noqa: E402  (内含 ee.Initialize)

PX_MIN = 3


def extract(gdf_wgs84, windows, tag):
    print(f"[GEE:{tag}] 提取 {len(gdf_wgs84)} 块 …", flush=True)
    df = P.extract_features(gdf_wgs84, windows=windows)
    df = df.rename(columns={c: f"{c.rsplit('_',1)[0]}_{P.BAND_MAP[c.rsplit('_',1)[1]]}"
                            for c in df.columns
                            if "_" in c and c.rsplit("_", 1)[1] in P.BAND_MAP})
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(
        r"e:\工作相关\2026年\0624 待测试数据\模型", "cropland_filter_general.pkl"))
    ap.add_argument("--wide-fallback", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.model, "rb") as f:
        bundle = pickle.load(f)
    print(f"模型 {os.path.basename(a.model)}: {bundle['class_names']}")

    g = gpd.read_file(a.input)
    g = g.to_crs(4326).reset_index(drop=True)
    g["fid"] = np.arange(len(g))
    print(f"地块 {len(g)} 块")

    feat = extract(g, None, "标准10天窗")
    feat.to_csv(os.path.join(a.out, "sentinel_bands_standard.csv"),
                index=False, encoding="utf-8-sig")

    if a.wide_fallback:
        b08 = [c for c in feat.columns if c.endswith("_B08")]
        any_missing = feat[b08].isna().any(axis=1).values if b08 else np.zeros(len(feat), bool)
        need = feat.loc[(feat["buf_pixel"].fillna(0) < PX_MIN) | any_missing, ["fid"]]
        if len(need):
            sub = g[g["fid"].isin(need["fid"])].copy()
            print(f"[补提] {len(sub)} 块像元不足（<{PX_MIN}），改用 20 天宽窗")
            w = extract(sub, P.WIDE_WINDOWS, "宽窗补提")
            feat = feat.set_index("fid")
            w = w.set_index("fid")
            for c in w.columns:
                feat.loc[w.index, c] = w[c]
            feat = feat.reset_index()
            feat.to_csv(os.path.join(a.out, "sentinel_bands_wide.csv"),
                        index=False, encoding="utf-8-sig")

    # ── 特征矩阵 + 推理 ──
    df = feat.copy()
    phases = bundle["phases"]
    bv = {}
    for p in phases:
        for b in P.BANDS_FEAT:
            c = f"{p}_{b}"
            if c in df.columns:
                bv[c] = pd.to_numeric(df[c], errors="coerce").astype(float).values
    X, fnames = compute_feature_matrix(bv, phases, available_bands=P.BANDS_FEAT)
    X = pd.DataFrame(X, columns=fnames)
    model = bundle["model"]
    for m in [c for c in bundle["feature_names"] if c not in X.columns]:
        X[m] = 0.0
    Xm = X[bundle["feature_names"]].fillna(0).values.astype(np.float32)
    prob = model.predict_proba(Xm)[:, 1]
    is_crop = (prob >= 0.5).astype(int)

    # ── 指数（逐旬）──
    def band(p, b):
        c = f"{p}_{b}"
        return pd.to_numeric(df[c], errors="coerce").astype(float).values if c in df.columns \
            else np.full(len(df), np.nan)
    ndvi, mndwi, ndwi = [], [], []
    for p in phases:
        b3, b4, b8, b11 = band(p, "B03"), band(p, "B04"), band(p, "B08"), band(p, "B11")
        ndvi.append((b8 - b4) / np.where(np.abs(b8 + b4) < 1e-10, np.nan, b8 + b4))
        mndwi.append((b3 - b11) / np.where(np.abs(b3 + b11) < 1e-10, np.nan, b3 + b11))
        ndwi.append((b3 - b8) / np.where(np.abs(b3 + b8) < 1e-10, np.nan, b3 + b8))
    with np.errstate(invalid="ignore"):
        ndvi_max = np.nanmax(np.vstack(ndvi), axis=0)
        mndwi_max = np.nanmax(np.vstack(mndwi), axis=0)
        ndwi_max = np.nanmax(np.vstack(ndwi), axis=0)
    dekad_missing = df[[f"{p}_B08" for p in phases]].isna().any(axis=1).values
    buf = pd.to_numeric(df["buf_pixel"], errors="coerce").fillna(0).values

    # ── 综合判定 ──
    verdict = np.where(
        buf < 1, "无覆盖",
        np.where(buf < PX_MIN, "过小(<3px)",
                 np.where(mndwi_max > 0.30, "水体",
                          np.where(dekad_missing, "云覆盖缺失",
                                   np.where(is_crop == 1, "作物", "非作物(林/建/草)")))))
    out = g.copy()
    out["crop_prob"] = np.round(prob, 3)
    out["is_crop"] = is_crop
    out["ndvi_max"] = np.round(ndvi_max, 3)
    out["mndwi_max"] = np.round(mndwi_max, 3)
    out["ndwi_max"] = np.round(ndwi_max, 3)
    out["buf_pixel"] = buf.astype(int)
    out["dekad_miss"] = dekad_missing.astype(int)
    out["s2_verdict"] = verdict
    out = out.to_crs(4523)
    out["area_mu"] = (out.geometry.area * 0.0015).round(3)

    shp = os.path.join(a.out, "exp10_sentinel.shp")
    out.to_file(shp, driver="ESRI Shapefile", encoding="UTF-8")
    out.to_file(os.path.join(a.out, "exp10_sentinel.gpkg"), driver="GPKG", encoding="utf-8")
    out.drop(columns="geometry").to_csv(os.path.join(a.out, "exp10_sentinel.csv"),
                                       index=False, encoding="utf-8-sig")

    print("\n=== 哨兵判定分布 ===")
    t = out.groupby("s2_verdict").agg(块数=("fid", "size"), 面积亩=("area_mu", "sum"))
    t["块占比%"] = (t["块数"] / len(out) * 100).round(1)
    t["面积占比%"] = (t["面积亩"] / out.area_mu.sum() * 100).round(1)
    print(t.to_string())
    print(f"\n写出 {shp}")


if __name__ == "__main__":
    main()

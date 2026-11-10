# -*- coding: utf-8 -*-
"""validate_crop_library.py — 作物特征库跨区验证（绵阳地块）

思路：用绵阳交付件里「同事前序数据」的作物标签当作真值（CROP_BZMC = 水稻/玉米），
把地块的 98 维指纹分别与水稻库、玉米库比对打分，谁的分高就判谁，
看和真值是否一致 —— 这是检验「遂宁/泸州/雅安样本训的库能不能迁到绵阳」的直接证据。

用法：
    python validate_crop_library.py [--per-class 300] [--towns 黎雅镇,水晶镇]
输出：
    成果数据\作物特征库\验证\验证明细_{镇}.csv
    成果数据\作物特征库\验证\验证汇总.csv
"""
import argparse
import os
import sys

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import citrus_pomelo_extract as EX
from citrus_pomelo_score import fingerprints, score_one
from merge_fruit_into_delivery import block_fit_dist

ROOT = r"e:\工作相关\2026年\0624 待测试数据"
DELIV = os.path.join(ROOT, r"成果数据\1102 交付赋值数据")
LIB = os.path.join(ROOT, r"成果数据\作物特征库\自净")
OUT = os.path.join(ROOT, r"成果数据\作物特征库\验证")

TOWNS = [("ZT_OUT", "510725103", "黎雅镇"),
         ("PW_OUT", "510727108", "水晶镇"),
         ("JY_OUT", "510781109", "新安镇"),
         ("YT_OUT", "510723102", "富驿镇")]

LABEL2KEY = {"水稻": "rice", "玉米": "corn"}
NAME = {"rice": "水稻", "corn": "玉米"}
BATCH = 400


def load_town(out, code, town, per_class=300, seed=11):
    p = os.path.join(DELIV, out, code, "交付", f"{town}_交付.shp")
    g = gpd.read_file(p)
    g = g[g["AG_BZBM"] == "A01"]
    g = g[g["CROP_BZMC"].isin(list(LABEL2KEY))].copy()
    if "FRT_BZMC" in g.columns:      # 抛掉已判为水果的块，避免标签冲突
        g = g[g["FRT_BZMC"].isna() | (g["FRT_BZMC"].astype(str).str.strip() == "")]
    g = g[~g.geometry.is_empty & g.geometry.notna()].copy()
    g["geometry"] = [x if x.is_valid else x.buffer(0) for x in g.geometry]
    picks = []
    for lab in LABEL2KEY:
        sub = g[g["CROP_BZMC"] == lab]
        if len(sub) > per_class:
            sub = sub.sample(per_class, random_state=seed)
        picks.append(sub)
    g = pd.concat(picks, ignore_index=True)
    g["fid"] = range(len(g))
    return g.to_crs("EPSG:4326"), g


def extract(g, year=EX.YEAR):
    b = g.total_bounds
    aoi = ee.Geometry.Rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                proj="EPSG:4326", geodesic=False)
    stack, names = EX.build_stack(aoi, year)
    cols = EX.band_names(names)
    gw = g.copy()
    gw["geometry"] = gw.geometry.simplify(tolerance=0.0001, preserve_topology=True)
    rows = []
    nb = (len(gw) + BATCH - 1) // BATCH
    for bi in range(nb):
        sub = gw.iloc[bi * BATCH:(bi + 1) * BATCH]
        fc = ee.FeatureCollection(sub[["fid", "geometry"]].__geo_interface__["features"])
        red = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(),
                                 scale=10, tileScale=4).select(["fid"] + cols)
        for f in EX.getinfo(red)["features"]:
            pr = f["properties"]
            rows.append({"fid": pr["fid"], **{c: pr.get(c) for c in cols}})
        print(f"      batch {bi+1}/{nb} → {len(rows)} 块", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--towns", help="逗号分隔的镇名，默认 4 个")
    ap.add_argument("--reuse", action="store_true", help="复用已提取的 特征_{镇}.csv，不重跑 GEE")
    a = ap.parse_args()
    towns = TOWNS
    if a.towns:
        keep = {t.strip() for t in a.towns.split(",")}
        towns = [t for t in TOWNS if t[2] in keep]
    os.makedirs(OUT, exist_ok=True)
    EX.ee_init()

    lib_fp = {}
    for k in ("rice", "corn"):
        src = pd.read_csv(os.path.join(LIB, f"来源产区_{k}_2025_自净.csv"))
        lib_fp[k] = fingerprints(src)
    common = [c for c in lib_fp["rice"].columns if c in lib_fp["corn"].columns]

    summary, allrec = [], []
    for out, code, town in towns:
        g, gsrc = load_town(out, code, town, a.per_class)
        cnt = gsrc["CROP_BZMC"].value_counts().to_dict()
        fpath = os.path.join(OUT, f"特征_{town}.csv")
        if a.reuse and os.path.exists(fpath):
            df = pd.read_csv(fpath)
            print(f"[{town}] 复用特征 {fpath}（{len(df)} 块）", flush=True)
        else:
            print(f"[{town}] 抽样 {len(g)} 块 {cnt}，提取 540 维 …", flush=True)
            df = extract(g)
            df.to_csv(fpath, index=False, encoding="utf-8-sig")
        tfp = fingerprints(df)

        # 口径 A：单库区间打分（P10~P90 区间内 1 分，区间外高斯衰减）→ 取高分
        sc = {k: score_one(lib_fp[k], tfp, common)[0] for k in ("rice", "corn")}
        pred = np.where(sc["rice"].values >= sc["corn"].values, "rice", "corn")
        # 口径 B：对称距离（|x−库中位|/库MAD 的均值）→ 取小者，不受库宽窄影响
        dist = {k: block_fit_dist(lib_fp[k], tfp, common) for k in ("rice", "corn")}
        pred_b = np.where(dist["rice"].values <= dist["corn"].values, "rice", "corn")

        truth = g["CROP_BZMC"].map(LABEL2KEY).values
        rec = pd.DataFrame({"fid": df["fid"].values, "真值": truth, "预测_区间": pred,
                            "预测_距离": pred_b,
                            "分_水稻": sc["rice"].values.round(4),
                            "分_玉米": sc["corn"].values.round(4),
                            "距_水稻": dist["rice"].values.round(4),
                            "距_玉米": dist["corn"].values.round(4)})
        conf = np.abs(dist["rice"].values - dist["corn"].values)
        rec["距离边际"] = conf.round(4)
        rec = rec.merge(gsrc[["fid", "CJBM", "SJMJ"]], on="fid", how="left")
        rec.to_csv(os.path.join(OUT, f"验证明细_{town}.csv"), index=False, encoding="utf-8-sig")
        allrec.append(rec.assign(镇=town))

        acc = (rec["真值"] == rec["预测_区间"]).mean()
        acc_b = (rec["真值"] == rec["预测_距离"]).mean()
        cm = pd.crosstab(rec["真值"], rec["预测_距离"])
        hi = rec[conf >= 0.3]
        summary.append({"镇": town, "样本": len(rec),
                        "水稻": int(cnt.get("水稻", 0)), "玉米": int(cnt.get("玉米", 0)),
                        "一致率_区间": round(float(acc), 3),
                        "一致率_距离": round(float(acc_b), 3),
                        "水稻查准_距离": round(float((rec[rec["真值"] == "rice"]["预测_距离"] == "rice").mean()), 3),
                        "玉米查准_距离": round(float((rec[rec["真值"] == "corn"]["预测_距离"] == "corn").mean()), 3),
                        "高置信样本": len(hi),
                        "高置信一致率": round(float((hi["真值"] == hi["预测_距离"]).mean()), 3) if len(hi) else np.nan})
        print(f"  → 一致率(区间) {acc:.3f} / (距离) {acc_b:.3f}\n{cm.to_string()}\n")

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT, "验证汇总.csv"), index=False, encoding="utf-8-sig")
    allr = pd.concat(allrec, ignore_index=True)
    print("=== 汇总 ===")
    print(s.to_string(index=False))
    print(f"\n合并 {len(allr)} 块：一致率(区间) {(allr['真值'] == allr['预测_区间']).mean():.3f}"
          f" / (距离) {(allr['真值'] == allr['预测_距离']).mean():.3f}")
    print(pd.crosstab(allr["真值"], allr["预测_距离"]).to_string())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""citrus_pomelo_score.py — 柑橘 / 蜜柚：指数指纹打分 + 三级嫌疑

输入（citrus_pomelo_extract.py 的产出）：
    来源产区_{crop}_{year}.csv      正样本（产区点）
    目标地块_{crop}_{镇}_{year}.csv 待赋值地块
输出：
    {镇}_柑橘赋值.shp / .xlsx（或蜜柚），逐块 分值 + 高/中/低 嫌疑

指纹口径（从 9 指数 × 36 旬 派生）：
    季节均值（冬 12/1/2、春 3/4/5、夏 6/7/8、秋 9/10/11）× 关键指数
    全年振幅、冬季稳定性、成熟期值

用法：
    python citrus_pomelo_score.py --crop citrus --src-dir DIR --out-dir DIR [--hi 0.70] [--mid 0.45]
"""
import argparse
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MU = 0.0015
TARGET_CRS = "EPSG:4523"
DELIV = r"e:\工作相关\2026年\0624 待测试数据\成果数据\0922 前序环节数据\MY第一批_我们的赋值"

# 目标镇（与 extract 的 TARGETS 对应）
TARGETS = {
    "citrus": [("PW_OUT", "510727108", "水晶镇"), ("PW_OUT", "510727216", "豆叩羌族乡"),
               ("PW_OUT", "510727107", "大桥镇"), ("PW_OUT", "510727208", "旧堡羌族乡"),
               ("JY_OUT", "33", "新安镇"), ("JY_OUT", "38", "河口镇"),
               ("JY_OUT", "26", "太平镇"), ("JY_OUT", "42", "马角镇")],
    "pomelo": [("ZT_OUT", "510725103", "黎雅镇"), ("ZT_OUT", "510725112", "金龙镇")],
}
# 水果标的：名称 + 编码（8 位，与交付「大春作物」口径一致 = 附表三 6 位 + "00"）
NAME = {"citrus": ("柑橘", "AD040000"), "pomelo": ("蜜柚", "AD040200")}

# 参与打分的指数（EVI 云残留噪声大，不进指纹；其余 14 个全用）
USE_IDX = ["NDVI", "SAVI", "RVI", "NDWI", "MNDWI", "LSWI", "NDII", "NDRE", "NDI45",
           "CVI", "TCARI", "IRECI", "MTCI", "GCVI"]
SEASON = {"冬": [12, 1, 2], "春": [3, 4, 5], "夏": [6, 7, 8], "秋": [9, 10, 11]}


def dekad_labels(year=2025):
    import calendar
    out = []
    for m in range(1, 13):
        for k in (1, 2, 3):
            out.append(f"{m:02d}{k}")
    return out


def fingerprints(df, year=2025):
    """从 9×36 的旬列派生指纹指标，返回 DataFrame（保留原索引）"""
    labs = dekad_labels(year)
    mm = {l: int(l[:2]) for l in labs}
    feats = {}
    for p in USE_IDX:
        cols = [f"{p}_{l}" for l in labs if f"{p}_{l}" in df.columns]
        if not cols:
            continue
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        for s, ms in SEASON.items():
            sc = [c for c in cols if mm[c.split("_")[1]] in ms]
            feats[f"{s}_{p}"] = sub[sc].mean(axis=1, skipna=True)
        feats[f"振幅_{p}"] = sub[[c for c in cols if mm[c.split("_")[1]] in (7, 8)]].mean(
            axis=1, skipna=True) - sub[[c for c in cols if mm[c.split("_")[1]] in (12, 1, 2)]].mean(
            axis=1, skipna=True)
        feats[f"冬稳定_{p}"] = sub[[c for c in cols if mm[c.split("_")[1]] == 12]].std(axis=1)
        feats[f"成熟期_{p}"] = sub[[c for c in cols if mm[c.split("_")[1]] in (10, 11)]].mean(
            axis=1, skipna=True)
    return pd.DataFrame(feats, index=df.index)


def score_one(src_fp, tgt_fp, cols):
    """对每个指纹指标：来源产区 P10~P90 为典型区间，区间内 1 分，否则按 MAD 距离衰减"""
    sc = pd.DataFrame(index=tgt_fp.index, dtype=float)
    w = []
    for c in cols:
        s = src_fp[c].dropna()
        if len(s) < 20:
            continue
        lo, hi = s.quantile(.10), s.quantile(.90)
        med = s.median()
        mad = max((s - med).abs().median() * 1.4826, 1e-3)
        v = tgt_fp[c]
        z = np.where((v >= lo) & (v <= hi), 0.0,
                     np.where(v < lo, (lo - v) / mad, (v - hi) / mad))
        sc[c] = np.exp(-0.5 * (np.asarray(z, float) / 1.5) ** 2)   # 距离衰减
        w.append(c)
    sc = sc[w]
    total = sc.mean(axis=1, skipna=True)
    return total, sc


def build_fingerprint(src_csv, year=2025):
    """从（自净后的）来源样本建指纹：每个指标的 P10/P25/P50/P75/P90/MAD/mean/std。
    这个表就是「特征库」本体 —— 打分时用它给出的典型区间与离散度。"""
    src = pd.read_csv(src_csv)
    fp = fingerprints(src, year)
    med = fp.median()
    tab = pd.DataFrame({
        "P10": fp.quantile(.10), "P25": fp.quantile(.25), "P50": med,
        "P75": fp.quantile(.75), "P90": fp.quantile(.90),
        "mean": fp.mean(), "std": fp.std(),
        "MAD": (fp - med).abs().median() * 1.4826,
        "N": fp.notna().sum().astype(int),
    }).round(5)
    return tab, fp, src


def compare(a_fp, b_fp, na, nb):
    """两个库的分辨力：逐指标算 Cohen's d 与 P10~P90 区间重叠度。
    Cohen's d 越小 / 重叠度越高 → 该指标分不开这两个作物。"""
    rows = []
    for c in a_fp.columns:
        if c not in b_fp.columns:
            continue
        a, b = a_fp[c].dropna(), b_fp[c].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
        alo, ahi = a.quantile(.10), a.quantile(.90)
        blo, bhi = b.quantile(.10), b.quantile(.90)
        ov = max(0.0, min(ahi, bhi) - max(alo, blo)) / max(
            1e-9, max(ahi, bhi) - min(alo, blo))
        rows.append({"指标": c, na: round(a.mean(), 4), nb: round(b.mean(), 4),
                     "CohenD": round(float(d), 3), "区间重叠度": round(float(ov), 3),
                     "可分辨": "否" if abs(d) < 0.2 or ov > 0.8 else "是"})
    t = pd.DataFrame(rows)
    return t.reindex(t["CohenD"].abs().sort_values(ascending=False).index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", choices=["citrus", "pomelo"], help="不打分时可不传（建指纹会两个都做）")
    ap.add_argument("--src-dir", required=True, help="特征 CSV 目录")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--hi", type=float, default=0.70)
    ap.add_argument("--mid", type=float, default=0.45)
    ap.add_argument("--build-fingerprint", action="store_true",
                    help="只从「自净」样本建指纹库并导出，不打分目标镇")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if a.build_fingerprint:
        fps = {}
        for crop in (["citrus", "pomelo"] if not a.crop else [a.crop]):
            p = os.path.join(a.src_dir, f"来源产区_{crop}_{a.year}_自净.csv")
            if not os.path.exists(p):
                print(f"  !! 缺自净样本，跳过 {crop}: {p}")
                continue
            tab, fp, src = build_fingerprint(p, a.year)
            fps[crop] = fp
            out = os.path.join(a.out_dir, f"指纹_{crop}_{a.year}.csv")
            tab.to_csv(out, encoding="utf-8-sig")
            print(f"[指纹] {crop}: {len(src)} 个常绿像元 → {fp.shape[1]} 维指纹 -> {out}")
        if len(fps) == 2:
            t = compare(fps["citrus"], fps["pomelo"], "柑橘", "蜜柚")
            op = os.path.join(a.out_dir, "指纹对比_柑橘vs蜜柚.csv")
            t.to_csv(op, index=False, encoding="utf-8-sig")
            print(f"\n=== 柑橘 vs 蜜柚 可分性（Cohen's d 越大越好分）===")
            print(t.head(12).to_string(index=False))
            n_ok = (t["可分辨"] == "是").sum()
            print(f"\n共 {len(t)} 个指标，可分辨 {n_ok} 个 -> {op}")
        return

    src_f = os.path.join(a.src_dir, f"来源产区_{a.crop}_{a.year}_自净.csv")
    if not os.path.exists(src_f):      # 没有自净版就退回原始版
        src_f = os.path.join(a.src_dir, f"来源产区_{a.crop}_{a.year}.csv")
    src = pd.read_csv(src_f)
    print(f"[来源] {src_f}: {len(src)} 点，产区 {src['region'].nunique()} 个")
    src_fp = fingerprints(src, a.year)
    print(f"       指纹 {src_fp.shape[1]} 维")

    label, code = NAME[a.crop]
    summary = []
    for aoi, aid, town in TARGETS[a.crop]:
        tf = os.path.join(a.src_dir, f"目标地块_{a.crop}_{town}_{a.year}.csv")
        if not os.path.exists(tf):
            print(f"  !! 缺目标特征，跳过 {town}")
            continue
        tgt = pd.read_csv(tf)
        tfp = fingerprints(tgt, a.year)
        common = [c for c in src_fp.columns if c in tfp.columns]
        total, detail = score_one(src_fp, tfp, common)
        lv = np.where(total >= a.hi, "高", np.where(total >= a.mid, "中", "低"))

        # 挂回几何（交付件在 DELIV/{OUT}/{AOI编码}/交付/）
        d = os.path.join(DELIV, aoi, aid, "交付")
        f = glob.glob(os.path.join(d, "*_交付.shp"))[0]
        g = gpd.read_file(f).to_crs(TARGET_CRS)
        g = g[g.AG_BZBM == "A01"].copy().reset_index(drop=True)
        g["_score"] = np.nan
        g["_level"] = ""
        g.loc[tgt["fid"].values, "_score"] = total.round(4).values
        g.loc[tgt["fid"].values, "_level"] = lv
        g["_crop_code"] = np.where(g["_level"].isin(["高", "中"]), code, "")
        g["_crop_name"] = np.where(g["_level"].isin(["高", "中"]), label, "")

        ddir = os.path.join(a.out_dir, town)
        os.makedirs(ddir, exist_ok=True)
        out = g.rename(columns={"XH": "XH", "SJMJ": "SJMJ"})
        out.to_file(os.path.join(ddir, f"{town}_{label}赋值.shp"),
                    driver="ESRI Shapefile", encoding="UTF-8")
        g.drop(columns="geometry").to_csv(
            os.path.join(ddir, f"{town}_{label}赋值.csv"), index=False, encoding="utf-8-sig")
        cnt = pd.Series(lv).value_counts().to_dict()
        print(f"  [{town}] {len(g)} 块 → 高 {cnt.get('高',0)} / 中 {cnt.get('中',0)} / 低 {cnt.get('低',0)}"
              f"  （分值中位 {np.nanmedian(total):.3f}）")
        summary.append({"镇": town, "块数": len(g), **{f"{k}嫌疑": v for k, v in cnt.items()},
                        "分值中位": round(float(np.nanmedian(total)), 4)})

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(os.path.join(a.out_dir, f"{a.crop}_嫌疑汇总.csv"), index=False, encoding="utf-8-sig")
        print("\n=== 汇总 ===")
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()

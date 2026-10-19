# -*- coding: utf-8 -*-
"""orchard_supervised_report.py — 果园有监督对照的统计与出图

输入：orchard_supervised.py 产出的 `果园监督对照_{year}.csv`
输出：物候曲线对照图、冬季 NDVI 分布图、判据分离曲线、统计表
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GROUPS = ["果园(调查)", "耕地_玉米", "耕地_水稻", "耕地_未判作物"]
COLORS = {"果园(调查)": "#d62728", "耕地_玉米": "#1f77b4",
          "耕地_水稻": "#2ca02c", "耕地_未判作物": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    nd_cols = [c for c in df.columns if c.startswith("NDVI_") and len(c) <= 9]
    nd_cols = sorted(nd_cols, key=lambda c: c.split("_")[1])
    labels = [c.split("_")[1] for c in nd_cols]
    print(f"{len(df)} 块 × {len(nd_cols)} 旬")
    print(dict(df.group.value_counts()))

    # ---------- 1) 物候曲线对照 ----------
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(nd_cols))
    xt = [i for i, l in enumerate(labels) if l.endswith("1")]
    for g in GROUPS:
        s = df[df.group == g][nd_cols]
        if s.empty:
            continue
        med = s.median(axis=0, skipna=True).values
        q1 = s.quantile(.25).values
        q3 = s.quantile(.75).values
        ax.plot(x, med, color=COLORS[g], lw=2, label=f"{g} (n={len(s)})")
        ax.fill_between(x, q1, q3, color=COLORS[g], alpha=.13)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(l[:2])}月" for l in np.array(labels)[xt]])
    ax.set_ylabel("NDVI（地块级旬均值，中位 + P25~P75）")
    ax.set_title("江油：调查果园 vs 交付耕地 · 2025 全年 36 旬 NDVI")
    ax.grid(alpha=.3)
    ax.legend()
    p1 = os.path.join(a.out, "果园监督_物候曲线对照.png")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"  -> {p1}")

    # ---------- 2) 季节统计 ----------
    seas = {"冬季(12,1,2)": [12, 1, 2], "春季(3,4,5)": [3, 4, 5],
            "夏季(6,7,8)": [6, 7, 8], "秋季(9,10,11)": [9, 10, 11]}
    m2c = {c: int(l[:2]) for c, l in zip(nd_cols, labels)}
    for lab, ms in seas.items():
        cols = [c for c in nd_cols if m2c[c] in ms]
        df[lab] = df[cols].mean(axis=1, skipna=True)
    df["振幅"] = df["夏季(6,7,8)"] - df["冬季(12,1,2)"]
    df["有效旬数"] = df[nd_cols].notna().sum(axis=1)

    def q(s):
        return pd.Series({"P10": s.quantile(.10), "P25": s.quantile(.25),
                          "P50": s.quantile(.50), "P75": s.quantile(.75),
                          "P90": s.quantile(.90)})
    t = []
    for g in GROUPS:
        s = df[df.group == g]
        if s.empty:
            continue
        r = {"组": g, "块数": len(s), "有效旬数中位": s["有效旬数"].median()}
        for lab in list(seas) + ["振幅"]:
            r[lab + "_P50"] = round(s[lab].median(), 3)
        r["冬季NDVI_P10"] = round(s["冬季(12,1,2)"].quantile(.10), 3)
        r["冬季NDVI_P90"] = round(s["冬季(12,1,2)"].quantile(.90), 3)
        t.append(r)
    tab = pd.DataFrame(t)
    print("\n=== 季节统计 ===")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(a.out, "果园监督_季节统计.csv"), index=False, encoding="utf-8-sig")

    # ---------- 3) 冬季 NDVI 分布 ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for g in GROUPS:
        s = df[df.group == g]["冬季(12,1,2)"].dropna()
        if len(s) < 5:
            continue
        ax.hist(s, bins=np.arange(-0.1, 0.85, 0.05), alpha=.5,
                label=f"{g} (n={len(s)}, 中位 {s.median():.2f})", color=COLORS[g])
    ax.axvline(0.20, color="k", ls="--", lw=1.2)
    ax.text(0.205, ax.get_ylim()[1]*.9, "落叶果园上限 0.20", fontsize=9)
    ax.axvline(0.40, color="k", ls=":", lw=1.2)
    ax.text(0.405, ax.get_ylim()[1]*.9, "常绿果园下限 0.40", fontsize=9)
    ax.set_xlabel("冬季（12/1/2 月）NDVI")
    ax.set_ylabel("地块数")
    ax.set_title("冬季 NDVI 分布：果园 vs 耕地")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    p2 = os.path.join(a.out, "果园监督_冬季NDVI分布.png")
    fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"  -> {p2}")

    # ---------- 4) 判据分离 ----------
    o = df[df.group == "果园(调查)"]["冬季(12,1,2)"].dropna()
    rows = []
    for g in ["耕地_玉米", "耕地_水稻", "耕地_未判作物"]:
        n = df[df.group == g]["冬季(12,1,2)"].dropna()
        if len(n) < 5:
            continue
        for th in np.arange(0.10, 0.55, 0.05):
            rows.append({"对照": g, "阈值": round(th, 2),
                         "果园判为常绿(>阈)": round((o > th).mean() * 100, 1),
                         "对照误报为常绿(>阈)": round((n > th).mean() * 100, 1)})
    sep = pd.DataFrame(rows)
    print("\n=== 阈值分离（冬季 NDVI > 阈值 判为常绿/果园）===")
    print(sep.pivot(index="阈值", columns="对照",
                    values=["果园判为常绿(>阈)", "对照误报为常绿(>阈)"]).to_string())
    sep.to_csv(os.path.join(a.out, "果园监督_阈值分离.csv"), index=False, encoding="utf-8-sig")

    # ---------- 5) 逐月有效旬数 + 月 NDVI ----------
    rows = []
    for mth in range(1, 13):
        cs = [c for c in nd_cols if m2c[c] == mth]
        r = {"月": mth, "旬数": len(cs)}
        for g in GROUPS:
            s = df[df.group == g][cs]
            r[f"{g}_有效旬"] = round(s.notna().sum(axis=1).median(), 1)
            r[f"{g}_NDVI"] = round(s.median(axis=1, skipna=True).median(), 3)
        r["果园-玉米"] = round(r["果园(调查)_NDVI"] - r["耕地_玉米_NDVI"], 3)
        r["果园-水稻"] = round(r["果园(调查)_NDVI"] - r["耕地_水稻_NDVI"], 3)
        rows.append(r)
    mon = pd.DataFrame(rows)
    print("\n=== 逐月 NDVI（中位）与有效旬数 ===")
    print(mon.to_string(index=False))
    mon.to_csv(os.path.join(a.out, "果园监督_逐月.csv"), index=False, encoding="utf-8-sig")

    # ---------- 6) 单特征 AUC ----------
    def auc(pos, neg):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
        if len(pos) < 3 or len(neg) < 3:
            return np.nan
        r = pd.Series(np.concatenate([pos, neg])).rank().values
        return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    POS = df.group == "果园(调查)"
    NEG = {"玉米": df.group == "耕地_玉米", "水稻": df.group == "耕地_水稻",
           "未判作物": df.group == "耕地_未判作物", "耕地合计": df.group.str.startswith("耕地")}
    feats = {}
    for mth in range(1, 13):
        cs = [c for c in nd_cols if m2c[c] == mth]
        feats[f"{mth}月均"] = df[cs].mean(axis=1, skipna=True)
    for nm, ms in [("5-6月均", (5, 6)), ("10-11月均", (10, 11)),
                   ("夏(7,8)均", (7, 8)), ("冬(12,1,2)均", (12, 1, 2))]:
        feats[nm] = df[[c for c in nd_cols if m2c[c] in ms]].mean(axis=1, skipna=True)

    arow = []
    for fn, v in feats.items():
        r = {"特征": fn, "果园中位": round(np.nanmedian(v[POS]), 3)}
        for gn, mask in NEG.items():
            r[f"AUC_{gn}"] = round(auc(v[POS], v[mask]), 3)
        arow.append(r)
    au = pd.DataFrame(arow)
    print("\n=== 单特征 AUC（果园为正类，取值高者判为果园；0.5=无判别力）===")
    print(au.to_string(index=False))
    au.to_csv(os.path.join(a.out, "果园监督_AUC.csv"), index=False, encoding="utf-8-sig")

    df.to_csv(a.csv, index=False, encoding="utf-8-sig")
    print(f"\n[输出目录] {a.out}")


if __name__ == "__main__":
    main()

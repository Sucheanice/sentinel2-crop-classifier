# -*- coding: utf-8 -*-
"""citrus_pomelo_clean.py — 产区样本「常绿像元」筛分 + 目视核验（v2）

回答的问题：**我们选的产区，里面真的种着目标作物吗？**

判据（v2，改用「全年最低月 NDVI」，比 v1 的聚类取最绿群稳定、且能排除冬小麦）：

    常绿像元 = 全年 12 个月**每个月**的 NDVI 都 >= --min-month-ndvi（默认 0.40）

原理：柑橘 / 柚是常绿，全年每个月都绿；而水稻、玉米、小麦、油菜这类一年生作物，
在泡田 / 收割 / 换茬期必然掉到 0.2 附近 —— 只要有一个月低，就不算常绿。
（v1 用「冬季 NDVI 高 + 振幅小」，会把「冬小麦+夏玉米」两熟田误收进来，已废弃。）

输出：
  1. 表：各产区 四季 NDVI / 全年最低月 / 最高月 / **常绿像元占比** / 判定
  2. 逐月 NDVI 表（12 列）→ 可直接画物候曲线
  3. 自净样本 CSV（只留常绿像元，供后续建指纹库）
  4. KML：红 = 常绿像元，灰 = 其余 → 拖进 Google Earth 即可目视核对

用法：
    python citrus_pomelo_clean.py --csv 来源产区_citrus_2025_园.csv --out 输出目录
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SEASON = {"冬": [12, 1, 2], "春": [3, 4, 5], "夏": [6, 7, 8], "秋": [9, 10, 11]}
MONTHS = list(range(1, 13))


def ndvi_matrix(df, year=2025):
    """返回 (36 旬 NDVI 矩阵, 旬标签)"""
    labs = [f"{m:02d}{k}" for m in MONTHS for k in (1, 2, 3)]
    cols = [f"NDVI_{l}" for l in labs if f"NDVI_{l}" in df.columns]
    labs = [c.split("_", 1)[1] for c in cols]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    # 沿旬轴线性插值补缺；有效旬 <3 的点整体置 NaN
    idx = np.arange(len(cols))
    out = np.full(x.shape, np.nan)
    for i in range(len(x)):
        v = x.values[i]
        ok = ~np.isnan(v)
        if ok.sum() >= 6:
            out[i] = np.interp(idx, idx[ok], v[ok])
    return out, labs


def monthly(m, labs):
    """36 旬 → 12 个月均（每 3 旬一月）"""
    mm = np.full((m.shape[0], 12), np.nan)
    for j, mo in enumerate(MONTHS):
        c = [i for i, l in enumerate(labs) if int(l[:2]) == mo]
        if c:
            mm[:, j] = np.nanmean(m[:, c], axis=1)
    return mm


def write_kml(lon, lat, flag, names, path, title):
    p = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         f'<name>{title}</name>',
         '<Style id="a"><IconStyle><color>ff0000ff</color><scale>0.9</scale></IconStyle></Style>',
         '<Style id="b"><IconStyle><color>ff9c9c9c</color><scale>0.6</scale></IconStyle></Style>']
    for lo, la, f, nm in zip(lon, lat, flag, names):
        if pd.isna(lo) or pd.isna(la):
            continue
        st = "a" if f else "b"
        p.append(f'<Placemark><name>{nm}</name><styleUrl>#{st}</styleUrl>'
                 f'<Point><coordinates>{lo},{la},0</coordinates></Point></Placemark>')
    p.append("</Document></kml>")
    open(path, "w", encoding="utf-8").write("\n".join(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--min-month-ndvi", type=float, default=0.40,
                    help="常绿判据：全年每个月 NDVI 都需 >= 该值（默认 0.40）")
    ap.add_argument("--min-months", type=int, default=12,
                    help="至少要有几个月的有效数据才参与判定（默认 12）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    crop = df["crop"].iloc[0] if "crop" in df.columns else "crop"
    print(f"[输入] {a.csv}: {len(df)} 点，产区 {df['region'].nunique()} 个，"
          f"判据=全年每月NDVI>={a.min_month_ndvi}")

    X, labs = ndvi_matrix(df, a.year)
    MM = monthly(X, labs)
    valid_m = (~np.isnan(MM)).sum(axis=1)
    mn = np.nanmin(MM, axis=1)          # 全年最低月
    mx = np.nanmax(MM, axis=1)
    flag = (mn >= a.min_month_ndvi) & (valid_m >= a.min_months)

    df = df.reset_index(drop=True)
    df["_minmonth"] = mn
    df["_maxmonth"] = mx
    df["_valid_m"] = valid_m
    df["_常绿"] = flag
    df = pd.concat([df, pd.DataFrame(MM, columns=[f"月{i:02d}" for i in MONTHS])], axis=1)

    rows = []
    for reg, s in df.groupby("region"):
        mm = {k: float(np.nanmean(s[[f"月{m:02d}" for m in MS]].values))
              for k, MS in SEASON.items()}
        n_ok = int(s["_常绿"].sum())
        rows.append({
            "产区": reg, "点数": len(s),
            "冬": round(mm["冬"], 3), "春": round(mm["春"], 3),
            "夏": round(mm["夏"], 3), "秋": round(mm["秋"], 3),
            "最低月": round(float(s["_minmonth"].median()), 3),
            "最高月": round(float(s["_maxmonth"].median()), 3),
            "常绿占比%": round(n_ok / len(s) * 100, 1),
            "判定": "✅" if n_ok / len(s) >= 0.30 else "⚠️常绿像元过少(可能选错区)",
        })
    tab = pd.DataFrame(rows).sort_values("常绿占比%", ascending=False)
    print("\n=== 各产区「全年常绿像元」占比与物候 ===")
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(a.out, f"产区自净_{crop}_{a.year}.csv"),
               index=False, encoding="utf-8-sig")

    # 逐月 NDVI（每产区一行，便于画曲线）
    mo = df.groupby("region")[[f"月{m:02d}" for m in MONTHS]].median().round(3)
    mo.to_csv(os.path.join(a.out, f"逐月NDVI_{crop}_{a.year}.csv"), encoding="utf-8-sig")
    print(f"\n[逐月物候] {os.path.join(a.out, f'逐月NDVI_{crop}_{a.year}.csv')}")

    clean = df[df["_常绿"]].copy()
    cp = os.path.join(a.out, f"来源产区_{crop}_{a.year}_自净.csv")
    clean.drop(columns=["_minmonth", "_maxmonth", "_valid_m", "_常绿"],
               errors="ignore").to_csv(cp, index=False, encoding="utf-8-sig")
    print(f"[自净样本] {len(clean)} 点（原 {len(df)}，剔掉 {(1-len(clean)/len(df))*100:.1f}% 非常绿像元）-> {cp}")

    kml = os.path.join(a.out, f"产区抽样点_分群_{crop}_{a.year}.kml")
    nm = [f'{r["region"]} 最低月{r["_minmonth"]:.2f}' for _, r in df.iterrows()]
    write_kml(df["lon"], df["lat"], df["_常绿"], nm, kml,
              f'{crop} 产区抽样点（红=全年常绿 {len(clean)} 点，灰=其余）')
    print(f"[目视] {kml}（红=全年常绿像元，灰=其余）")


if __name__ == "__main__":
    main()

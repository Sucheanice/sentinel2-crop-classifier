# -*- coding: utf-8 -*-
"""fruit_vs_cropland.py — 两个问题的量化核查

Q1 复核：用户怀疑的地块（「是橘子」「可能是橘子」等）到底像不像柑橘？
        做法：把每块真值地的指纹分别去比「柑橘库」「耕地代理库」，看它倒向哪边。

Q2 可分性：这套水果指纹能不能和耕地作物（水稻/玉米）分开？
        做法：用「产区镇级样本里非常绿的点」当耕地代理（这些镇以水稻/玉米/油菜为主），
              与柑橘库 / 蜜柚库比 Cohen's d 与区间重叠度。

注意：耕地代理是从产区镇级混合样本里筛出来的，没有作物标签，只能看"是不是一年生作物"；
      要精确到"水稻还是玉米"必须另提有标签的样本。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from citrus_pomelo_score import fingerprints, score_one, dekad_labels  # noqa: E402

D = r"e:\工作相关\2026年\0624 待测试数据\成果数据\柑橘蜜柚识别"
CLEAN = os.path.join(D, "自净")


def min_month_ndvi(df, year=2025):
    labs = dekad_labels(year)
    cols = [f"NDVI_{l}" for l in labs if f"NDVI_{l}" in df.columns]
    x = df[cols].apply(pd.to_numeric, errors="coerce").values
    idx = np.arange(len(cols))
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[i]
        ok = ~np.isnan(v)
        if ok.sum() >= 6:
            full = np.interp(idx, idx[ok], v[ok])
            out[i] = min(full[j * 3:j * 3 + 3].mean() for j in range(12))
    return out


def load_clean(crop):
    return pd.read_csv(os.path.join(CLEAN, f"来源产区_{crop}_2025_自净.csv"))


def load_cropland_proxy():
    """产区镇级混合样本里「不是全年常绿」的点 → 水稻/玉米/油菜/林地 代理"""
    out = []
    for f in ["来源产区_pomelo_2025.csv", "来源产区_citrus_2025_s0of3.csv"]:
        p = os.path.join(D, f)
        if not os.path.exists(p):
            print(f"  !! 缺 {f}")
            continue
        d = pd.read_csv(p)
        mm = min_month_ndvi(d)
        d = d.assign(_mm=mm)
        sub = d[d["_mm"] < 0.35]
        print(f"   {f}: {len(d)} 点 → 非常绿（耕地代理）{len(sub)} 点")
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def fit_dist(lib_fp, tgt_fp):
    """目标指纹到「库」的归一化偏离：|目标中位 − 库中位| / 库MAD，逐指标算再取中位/均值。
    越小 = 越像这个库。（不能用区间包含度：库越宽越容易拿高分，比不出高下。）"""
    zs = {}
    for c in lib_fp.columns:
        if c not in tgt_fp.columns:
            continue
        s = lib_fp[c].dropna()
        if len(s) < 20:
            continue
        m = s.median()
        mad = max((s - m).abs().median() * 1.4826, 1e-3)
        zs[c] = abs(float(tgt_fp[c].median()) - m) / mad
    z = pd.Series(zs)
    return float(z.median()), float(z.mean()), len(z)


def main():
    print("=" * 78)
    print("Q2  水果库 vs 耕地代理  可分性")
    print("=" * 78)
    cit, pom = load_clean("citrus"), load_clean("pomelo")
    crop_proxy = load_cropland_proxy()
    print(f"\n柑橘库 {len(cit)} 点 | 蜜柚库 {len(pom)} 点 | 耕地代理 {len(crop_proxy)} 点")

    f_cit, f_pom, f_crop = (fingerprints(cit), fingerprints(pom),
                            fingerprints(crop_proxy))
    rows = []
    for c in f_cit.columns:
        if c not in f_crop.columns:
            continue
        a, b = f_cit[c].dropna(), f_crop[c].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
        alo, ahi = a.quantile(.10), a.quantile(.90)
        blo, bhi = b.quantile(.10), b.quantile(.90)
        ov = max(0.0, min(ahi, bhi) - max(alo, blo)) / max(1e-9, max(ahi, bhi) - min(alo, blo))
        rows.append({"指标": c, "柑橘": round(a.mean(), 4), "耕地代理": round(b.mean(), 4),
                     "CohenD": round(float(d), 2), "区间重叠度": round(float(ov), 3)})
    t = pd.DataFrame(rows)
    t = t.reindex(t["CohenD"].abs().sort_values(ascending=False).index)
    ok = (t["CohenD"].abs() >= 0.5).sum()
    print(f"\n柑橘 vs 耕地代理：{len(t)} 个指标，|d|>=0.5 的 {ok} 个")
    print(t.head(12).to_string(index=False))

    # 用柑橘库对耕地代理打分 —— 耕地应当拿低分
    common = [c for c in f_cit.columns if c in f_crop.columns]
    sc_crop, _ = score_one(f_cit, f_crop, common)
    sc_cit, _ = score_one(f_cit, f_cit, common)
    print(f"\n【关键检验】用柑橘库打分：")
    print(f"   柑橘库自己        : 中位 {np.nanmedian(sc_cit):.3f}")
    print(f"   耕地代理（水稻玉米）: 中位 {np.nanmedian(sc_crop):.3f}")
    print(f"   → 若耕地代理显著更低，说明能分开")

    print("\n" + "=" * 78)
    print("Q1  复核：每块真值地的归属")
    print("=" * 78)
    truth = pd.read_csv(os.path.join(D, "来源产区_citrus_2025_真值合集.csv"))
    res = []
    for reg, s in truth.groupby("region"):
        f = fingerprints(s)
        z_cit, _, n = fit_dist(f_cit, f)
        z_pom, _, _ = fit_dist(f_pom, f)
        z_crop, _, _ = fit_dist(f_crop, f)
        mm = min_month_ndvi(s)
        best = min([("柑橘", z_cit), ("蜜柚", z_pom), ("耕地/其他", z_crop)], key=lambda x: x[1])
        res.append({"地块": reg, "点数": len(s),
                    "最低月NDVI": round(float(np.nanmedian(mm)), 3),
                    "偏离柑橘库": round(z_cit, 2), "偏离蜜柚库": round(z_pom, 2),
                    "偏离耕地代理": round(z_crop, 2), "最像": best[0]})
    r = pd.DataFrame(res).sort_values("偏离柑橘库")
    print(r.to_string(index=False))
    r.to_csv(os.path.join(CLEAN, "真值地块归属复核.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[输出] {os.path.join(CLEAN, '真值地块归属复核.csv')}")


if __name__ == "__main__":
    main()

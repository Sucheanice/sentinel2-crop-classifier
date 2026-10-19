# -*- coding: utf-8 -*-
"""分析叙永果园 12 个月物候曲线：李子(落叶) vs 柑橘(常绿)"""
import pandas as pd
import os
import numpy as np

BASE = r"e:\工作相关\2026年\0624 待测试数据"
CSV = os.path.join(BASE, "xuyong_orchard_monthly.csv")

df = pd.read_csv(CSV, encoding="utf-8-sig")
MONTHS = [f"M{i:02d}" for i in range(1, 13)]
lizi = df[df["ZWMC"] == "李子"]
ganju = df[df["ZWMC"] == "柑橘"]
print(f"李子 {len(lizi)} 块, 柑橘 {len(ganju)} 块\n")

print("=== 每月 NDVI 均值（李子 vs 柑橘）===")
print(f"{'月':<6}{'李子NDVI':>10}{'柑橘NDVI':>10}{'差值(柑-李)':>12}{'李缺值':>8}{'柑缺值':>8}")
for m in MONTHS:
    c = f"{m}_NDVI"
    lv = lizi[c].mean()
    gv = ganju[c].mean()
    ln = lizi[c].isna().sum()
    gn = ganju[c].isna().sum()
    print(f"{m:<6}{lv:>10.3f}{gv:>10.3f}{gv-lv:>12.3f}{ln:>8}{gn:>8}")

print("\n=== 每月 EVI 均值（李子 vs 柑橘）===")
print(f"{'月':<6}{'李子EVI':>10}{'柑橘EVI':>10}{'差值(柑-李)':>12}")
for m in MONTHS:
    c = f"{m}_EVI"
    lv = lizi[c].mean()
    gv = ganju[c].mean()
    print(f"{m:<6}{lv:>10.3f}{gv:>10.3f}{gv-lv:>12.3f}")

# 差异最大月份（标准化均值差 |mean差|/pooled std）
print("\n=== 单月 NDVI 区分能力排序（|均值差|/合并标准差）===")
rows = []
for m in MONTHS:
    c = f"{m}_NDVI"
    lv, gv = lizi[c].mean(), ganju[c].mean()
    pooled = np.sqrt((lizi[c].var() + ganju[c].var()) / 2)
    d = abs(gv - lv) / pooled if pooled > 0 else 0
    rows.append((m, d, gv - lv))
rows.sort(key=lambda x: -x[1])
for m, d, diff in rows:
    print(f"{m}: d'={d:.2f}  (柑-李={diff:+.3f})")

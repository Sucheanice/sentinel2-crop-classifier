# -*- coding: utf-8 -*-
"""对比果园 vs 周边植被(林地/草地/农田) 12 个月物候曲线"""
import pandas as pd
import os

BASE = r"e:\工作相关\2026年\0624 待测试数据"
ORCH = os.path.join(BASE, "xuyong_orchard_monthly.csv")
CTRL = os.path.join(BASE, "xuyong_control_monthly.csv")
MONTHS = [f"M{i:02d}" for i in range(1, 13)]

orc = pd.read_csv(ORCH, encoding="utf-8-sig")
orc["group"] = orc["ZWMC"]
ctrl = pd.read_csv(CTRL, encoding="utf-8-sig")
ctrl["group"] = ctrl["landtype"]

groups = ["李子", "柑橘", "林地", "草地", "农田"]
print("各类样本量:", {g: len(orc[orc.group == g]) + len(ctrl[ctrl.group == g]) for g in groups} if False else
      "李子=%d 柑橘=%d 林地=%d 草地=%d 农田=%d" % (
          len(orc[orc.group == '李子']), len(orc[orc.group == '柑橘']),
          len(ctrl[ctrl.group == '林地']), len(ctrl[ctrl.group == '草地']), len(ctrl[ctrl.group == '农田'])))

print("\n=== 12 个月 NDVI 均值（五类）===")
header = f"{'月':<6}" + "".join(f"{g:>10}" for g in groups)
print(header)
for m in MONTHS:
    c = f"{m}_NDVI"
    vals = []
    for g in groups:
        sub = orc[orc.group == g] if g in ('李子', '柑橘') else ctrl[ctrl.group == g]
        vals.append(sub[c].mean())
    print(f"{m:<6}" + "".join(f"{v:>10.3f}" for v in vals))

# 果园 vs 林地 差值（用户最初问题核心）
print("\n=== 关键对比：果园 vs 林地 / 果园 vs 农田（NDVI 差值，全年各月）===")
print(f"{'月':<6}{'李-林':>8}{'柑-林':>8}{'李-农田':>8}{'柑-农田':>8}")
lin = ctrl[ctrl.group == '林地']
nong = ctrl[ctrl.group == '农田']
for m in MONTHS:
    c = f"{m}_NDVI"
    lizi = orc[orc.group == '李子'][c].mean()
    ganju = orc[orc.group == '柑橘'][c].mean()
    print(f"{m:<6}{lizi - lin[c].mean():>8.3f}{ganju - lin[c].mean():>8.3f}"
          f"{lizi - nong[c].mean():>8.3f}{ganju - nong[c].mean():>8.3f}")

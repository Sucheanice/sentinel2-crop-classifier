#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""① 验证用户给的 3 个坐标的冬夏 NDVI；② 自动扫描石榴/苹果候选区（冬季NDVI<0.2 且 夏季>0.5）。"""
import json
import os
import sys

import ee
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ee.Initialize()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCL = [3, 8, 9, 10, 11]
OUT = os.path.join(r"e:\工作相关\2026年\0624 待测试数据", "成果数据", "果园识别探索")
os.makedirs(OUT, exist_ok=True)

# 冬季 = 2024-12 ~ 2025-02（前一个冬天，与 2025 生长季配套）；夏季 = 2025-06 ~ 2025-08
WIN = ("2024-12-01", "2025-03-01")
SUM = ("2025-06-01", "2025-09-01")


def mask_s2(img):
    m = None
    scl = img.select("SCL")
    for c in CLOUD_SCL:
        m = scl.eq(c) if m is None else m.Or(scl.eq(c))
    return img.updateMask(m.Not()).divide(10000)


def ndvi_comp(aoi, rng):
    col = ee.ImageCollection(S2).filterDate(*rng).filterBounds(aoi).map(mask_s2)
    med = col.median()
    return med.normalizedDifference(["B8", "B4"]).rename("ndvi"), col.size()


# ── ① 验证 3 个坐标 ──
PTS = [("石榴基地(你给的)", 101.77543, 26.30509),
       ("待确认1(是否苹果)", 101.77236, 26.32038),
       ("待确认2(盐源·可能苹果)", 101.53235, 27.47198)]
print("=== ① 你给的 3 个坐标：冬夏 NDVI（缓冲区 200 m 均值）===")
print(f"{'点位':<24} {'冬季NDVI':>9} {'夏季NDVI':>9} {'夏-冬':>7}  {'判读'}")
for nm, lon, lat in PTS:
    aoi = ee.Geometry.Point([lon, lat]).buffer(200)
    w, _ = ndvi_comp(aoi, WIN)
    s, _ = ndvi_comp(aoi, SUM)
    st = w.addBands(s).reduceRegion(ee.Reducer.mean(), aoi, scale=10,
                                    maxPixels=1e9, bestEffort=True).getInfo()
    wv, sv = st.get("ndvi"), st.get("ndvi_1")
    if wv is None or sv is None:
        verdict = "数据不足"
    elif wv < 0.20 and sv > 0.50:
        verdict = "★ 落叶果园特征（很可能是果园）"
    elif wv < 0.20 and sv <= 0.50:
        verdict = "冬季裸地但夏季不旺 → 可能休耕/稀疏"
    elif 0.20 <= wv <= 0.35:
        verdict = "★ 冬小麦/油菜田特征"
    elif wv > 0.45:
        verdict = "★ 常绿植被（非落叶果园）"
    else:
        verdict = "中间态"
    print(f"{nm:<24} {wv if wv is None else round(wv,3):>9} "
          f"{sv if sv is None else round(sv,3):>9} "
          f"{'—' if (wv is None or sv is None) else round(sv-wv,3):>7}  {verdict}")

# ── ② 自动扫描候选区 ──
REGIONS = [
    ("石榴_会理_鹿厂彰冠城南", [101.95, 26.45, 102.40, 26.85]),
    ("石榴_攀枝花仁和_大田", [101.65, 26.20, 101.95, 26.45]),
    ("苹果_盐源_卫城润盐", [101.40, 27.35, 101.70, 27.60]),
    ("苹果_小金_沃日", [102.30, 30.90, 102.55, 31.10]),
]
CRIT = "冬季NDVI<0.20 且 夏季NDVI>0.50"

print(f"\n=== ② 自动扫描（判据：{CRIT}）===")
print(f"{'区域':<26} {'冬有效%':>8} {'候选km²':>9} {'占比%':>7}")
summary = []
for nm, bb in REGIONS:
    aoi = ee.Geometry.Rectangle(bb, proj="EPSG:4326", geodesic=False)
    w, nw = ndvi_comp(aoi, WIN)
    s, ns = ndvi_comp(aoi, SUM)
    both = w.mask().And(s.mask())
    cand = w.lt(0.20).And(s.gt(0.50)).And(both)
    area = ee.Image.pixelArea().divide(1e6)
    st = ee.Image.cat([cand.rename("cand").multiply(area),
                       both.rename("both").multiply(area)]).reduceRegion(
        ee.Reducer.sum(), aoi, scale=30, maxPixels=1e13, bestEffort=True).getInfo()
    tot = ee.Image.pixelArea().divide(1e6).reduceRegion(
        ee.Reducer.sum(), aoi, scale=100, maxPixels=1e13, bestEffort=True).getInfo()
    ca = st.get("cand") or 0.0
    ba = st.get("both") or 0.0
    ta = tot.get("area") or 1.0
    print(f"{nm:<26} {ba/ta*100:>7.1f}% {ca:>9.1f} {ca/ta*100:>6.1f}%")
    summary.append({"区域": nm, "影像数(冬/夏)": f"{nw.getInfo()}/{ns.getInfo()}",
                    "候选面积km²": round(ca, 1), "区域面积km²": round(ta, 1),
                    "候选占比%": round(ca / ta * 100, 2)})

    # 导出候选多边形（100m 聚合，简化，限制数量）
    vec = cand.selfMask().reduceToVectors(
        geometry=aoi, scale=100, geometryType="polygon", eightConnected=False,
        maxPixels=1e13, bestEffort=True, labelProperty="c").map(
        lambda f: f.simplify(100)).limit(4000)
    feats = vec.getInfo()["features"]
    kml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f'<name>{nm} 落叶果园候选</name>',
           '<Style id="c"><LineStyle><color>ff0000ff</color></LineStyle>'
           '<PolyStyle><color>400000ff</color></PolyStyle></Style>']
    for f in feats:
        g = f.get("geometry")
        if not g:
            continue
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for p in polys:
            rings = " ".join(
                " ".join(f"{c[0]:.6f},{c[1]:.6f},0" for c in ring) for ring in p)
            kml.append(f'<Placemark><styleUrl>#c</styleUrl><Polygon><outerBoundaryIs>'
                       f'<LinearRing><coordinates>{rings}</coordinates></LinearRing>'
                       f'</outerBoundaryIs></Polygon></Placemark>')
    kml.append('</Document></kml>')
    open(os.path.join(OUT, f"落叶果园候选_{nm}.kml"), "w", encoding="utf-8").write("\n".join(kml))
    print(f"    候选多边形 {len(feats)} 个 → 落叶果园候选_{nm}.kml")

import pandas as pd
pd.DataFrame(summary).to_csv(os.path.join(OUT, "自动扫描_候选区统计.csv"),
                             index=False, encoding="utf-8-sig")
print(f"\n完成，输出目录：{OUT}")

# -*- coding: utf-8 -*-
"""kml_truth.py — 把 Google Earth 里手绘标注的 KML 多边形转成真值地块矢量

用途：把「人工在 Google Earth 上圈出、确知是某作物」的地块转成 gpkg，
供 citrus_pomelo_extract.py --truth-shp 直接作为样本来源。

用法：
    python kml_truth.py --kml 标注.kml [--out 输出.gpkg] [--keep 柑橘]

  --keep 只保留名称里含该词的标注（如“柑橘”）；不传则全部保留。
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import geopandas as gpd
from shapely.geometry import Polygon

sys.stdout.reconfigure(encoding="utf-8")
NS = "{http://www.opengis.net/kml/2.2}"
CRS_M = "EPSG:4523"


def parse_kml(path, keep=None):
    rows = []
    for pm in ET.parse(path).getroot().iter(NS + "Placemark"):
        nm = (pm.findtext(NS + "name", default="") or "").strip()
        if keep and keep not in nm:
            continue
        for poly in pm.iter(NS + "Polygon"):
            pts = []
            for c in poly.iter(NS + "coordinates"):
                for tok in (c.text or "").strip().split():
                    p = tok.split(",")
                    if len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
            if len(pts) >= 4:
                try:
                    g = Polygon(pts)
                    if not g.is_valid:
                        g = g.buffer(0)
                    if not g.is_empty:
                        rows.append({"名称": nm or "未命名", "geometry": g})
                except Exception:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml", required=True)
    ap.add_argument("--out")
    ap.add_argument("--keep", help="只保留名称含该词的标注")
    ap.add_argument("--ids", help="只保留指定序号的多边形，如 15,16,17（序号同打印列表）")
    a = ap.parse_args()

    rows = parse_kml(a.kml, a.keep)
    if not rows:
        print("!! 没解析到任何多边形")
        return
    if a.ids:
        want = {int(x) for x in a.ids.split(",") if x.strip()}
        rows = [r for i, r in enumerate(rows) if i in want]
        print(f"[筛选] 只保留序号 {sorted(want)} → {len(rows)} 个")
    g = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gm = g.to_crs(CRS_M)
    g["面积_亩"] = (gm.area / 666.667).round(2)
    inner = gm.geometry.apply(lambda p: p.buffer(-5).area if p.is_valid else 0)
    g["像元数"] = (inner / 100).round(0).astype(int)
    g["_id"] = range(len(g))

    print(f"[解析] {a.kml}")
    print(f"多边形 {len(g)} 个，来自 {g['名称'].nunique()} 种标注名")
    print("\n按标注名汇总:")
    print(g.groupby("名称")[["面积_亩", "像元数"]].agg(["count", "sum"]).round(1).to_string())
    print(f"\n合计 {g['面积_亩'].sum():.1f} 亩 | 可用 10m 像元 {g['像元数'].sum()} 个")
    print(f"单块 {g['面积_亩'].min():.1f} ~ {g['面积_亩'].max():.1f} 亩")
    print(f"范围(WGS84) {[round(x, 5) for x in g.total_bounds]}")
    print(f"\n最大的 10 块:")
    print(g.nlargest(10, "面积_亩")[["_id", "名称", "面积_亩", "像元数"]].to_string(index=False))

    out = a.out or os.path.splitext(a.kml)[0] + ".gpkg"
    g.to_file(out, driver="GPKG", encoding="utf-8")
    print(f"\n[输出] {out}")


if __name__ == "__main__":
    main()

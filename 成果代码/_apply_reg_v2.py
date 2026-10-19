# -*- coding: utf-8 -*-
"""对 fix2（未规则化）输出应用新版规则化，产出新 shp，并统计验证。

复用 geo_infer_v10.py 的生产函数，保证与服务器端一致。
用法:
  python _apply_reg_v2.py <in.shp> <out.shp>
"""
import sys
import numpy as np
import geopandas as gpd

sys.path.insert(0, r"e:\工作相关\2026年\0624 待测试数据\sam3_win\sam3_service")
import geo_infer_v10 as M

in_shp = sys.argv[1]
out_shp = sys.argv[2]

g = gpd.read_file(in_shp)
attr_cols = [c for c in g.columns if c != "geometry"]
items = []
for _, r in g.iterrows():
    props = {c: r[c] for c in attr_cols}
    items.append((r.geometry, props))

print(f"输入 {len(items)} 个 polygon", flush=True)
reg = M._regularize_items(items, 2.5)
print(f"规则化后 {len(reg)} 个 polygon", flush=True)

# 写回 shp（字段与输入一致）
from shapely.geometry import mapping
import fiona

schema = {"geometry": "Polygon", "properties": {}}
for c in attr_cols:
    dt = g[c].dtype
    if c == "id":
        schema["properties"][c] = "int"
    elif dt == "float64" or dt == "float32":
        schema["properties"][c] = "float"
    else:
        schema["properties"][c] = "str"

out_rows = []
for poly, props in reg:
    geoms = []
    if poly.geom_type == "MultiPolygon":
        geoms.extend([p for p in poly.geoms if not p.is_empty])
    elif poly.geom_type == "Polygon" and not poly.is_empty:
        geoms.append(poly)
    for gm in geoms:
        # 与生产一致：写 shp 前按 min_area_m2 过滤规则化产生的碎片
        if gm.area < 67.0:
            continue
        pp = dict(props)
        out_rows.append((gm, pp))

with fiona.open(out_shp, "w", driver="ESRI Shapefile", crs=g.crs,
                schema=schema, encoding="UTF-8") as sink:
    for i, (gm, pp) in enumerate(out_rows, 1):
        pp["id"] = i
        sink.write({"geometry": mapping(gm), "properties": pp})

print(f"写出 {len(out_rows)} 个多边形 -> {out_shp}", flush=True)

# 验证
def nv(p):
    if p is None or p.is_empty:
        return 0
    if p.geom_type == "Polygon":
        return len(p.exterior.coords) - 1
    return sum(len(x.exterior.coords) - 1 for x in p.geoms)

vs = np.array([nv(p) for p, _ in out_rows])
print(f"顶点: min={vs.min()} p50={int(np.median(vs))} p90={int(np.percentile(vs, 90))} max={vs.max()}")

from shapely.strtree import STRtree
polys = [p for p, _ in out_rows]
tree = STRtree(polys)
ov = 0
for i, p in enumerate(polys):
    for j in tree.query(p):
        j = int(j)
        if j <= i:
            continue
        try:
            if p.intersects(polys[j]) and p.intersection(polys[j]).area > 0.5:
                ov += 1
        except Exception:
            continue
print(f"相交对(>0.5m²): {ov}")

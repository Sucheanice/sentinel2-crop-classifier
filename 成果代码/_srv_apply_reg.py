# -*- coding: utf-8 -*-
"""服务器端：对 fix2（未规则化）输出应用新版规则化，产出最终 shp（仅 fiona，不依赖 geopandas）。

用法:
  python _srv_apply_reg.py <in.shp> <out.shp> [min_area_m2]
"""
import sys
import numpy as np
import fiona
from shapely.geometry import shape, mapping
from shapely.strtree import STRtree

sys.path.insert(0, "/data4/sam3_project/sam3_train/Code/sam3-main/sam3_service")
import geo_infer_v10 as M

in_shp = sys.argv[1]
out_shp = sys.argv[2]
min_area = float(sys.argv[3]) if len(sys.argv) > 3 else 67.0

items = []
with fiona.open(in_shp) as src:
    schema = src.schema
    crs = src.crs
    for feat in src:
        props = dict(feat["properties"])
        geom = shape(feat["geometry"])
        if geom is None or geom.is_empty:
            continue
        items.append((geom, props))

print(f"输入 {len(items)} 个 polygon", flush=True)
reg = M._regularize_items(items, 2.5)
print(f"规则化后 {len(reg)} 个 polygon", flush=True)

out_rows = []
for poly, props in reg:
    geoms = []
    if poly.geom_type == "MultiPolygon":
        geoms.extend([p for p in poly.geoms if not p.is_empty])
    elif poly.geom_type == "Polygon" and not poly.is_empty:
        geoms.append(poly)
    for gm in geoms:
        if gm.area < min_area:
            continue
        out_rows.append((gm, dict(props)))

with fiona.open(out_shp, "w", driver="ESRI Shapefile", crs=crs,
                schema=schema, encoding="UTF-8") as sink:
    for i, (gm, pp) in enumerate(out_rows, 1):
        pp["id"] = i
        sink.write({"geometry": mapping(gm), "properties": pp})

print(f"写出 {len(out_rows)} 个多边形 -> {out_shp}", flush=True)


def nv(p):
    if p is None or p.is_empty:
        return 0
    if p.geom_type == "Polygon":
        return len(p.exterior.coords) - 1
    return sum(len(x.exterior.coords) - 1 for x in p.geoms)

vs = np.array([nv(p) for p, _ in out_rows])
print(f"顶点: min={vs.min()} p50={int(np.median(vs))} p90={int(np.percentile(vs, 90))} max={vs.max()}", flush=True)

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
print(f"相交对(>0.5m²): {ov}", flush=True)

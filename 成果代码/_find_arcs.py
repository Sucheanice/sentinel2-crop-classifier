# -*- coding: utf-8 -*-
"""找弧线地块：凸度>0.92 且顶点多，或含长段同向短边。"""
import sys
import numpy as np
import geopandas as gpd

shp = sys.argv[1]
g = gpd.read_file(shp)

def edge_angles(p):
    c = np.array(p.exterior.coords[:-1], dtype=float)
    e = np.diff(np.vstack([c, c[0]]), axis=0)
    L = np.hypot(e[:, 0], e[:, 1])
    a = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    return e, L, a

def arc_run_len(p, max_turn=15.0, min_len=0.5):
    """最长连续同向且每步转角<max_turn的边数（弧线特征）。"""
    e, L, a = edge_angles(p)
    n = len(a)
    turns = (np.diff(a) + 180.0) % 360.0 - 180.0  # [-180,180)
    best = 0; run = 1; sign = 0
    for i in range(n - 1):
        t = turns[i]
        s = 1 if t > 0 else (-1 if t < 0 else 0)
        if s != 0 and (sign == 0 or s == sign) and abs(t) <= max_turn:
            sign = s
            run += 1
        else:
            run = 1; sign = s
        best = max(best, run)
    return best

def nv(p):
    return len(p.exterior.coords) - 1

rows = []
for _, r in g.iterrows():
    p = r.geometry
    if p is None or p.is_empty or p.geom_type != "Polygon":
        continue
    n = nv(p)
    if n < 10:
        continue
    ch = p.convex_hull.area
    conv = p.area / ch if ch > 0 else 0
    run = arc_run_len(p)
    if conv >= 0.90 and run >= 5:
        rows.append((round(conv, 3), run, n, int(r["id"]), round(p.area, 1)))

rows.sort(reverse=True)
print("凸度 | 弧线边长 | 顶点 | id | 面积")
for x in rows[:40]:
    print(x)

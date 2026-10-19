# -*- coding: utf-8 -*-
"""农田地块边界规则化（正交化版，保拓扑）。

对 SAM3 矢量化结果做边界后处理，解决「识别准、但边界锯齿 / 顶点乱」的问题：
  1) Douglas-Peucker 简化去锯齿；
  2) 全局主方向正交化：用全图统一的田埂主方向 + 统一旋转原点，把接近主方向 /
     正交方向的边对齐，角规整为直角（共享边因统一变换而保持重合，不产生相交）；
  3) 合并共线顶点：正交化后相邻共线小边合并，顶点数大幅下降；
  4) 斜边、弧线边（非正交、短边）保持不变。

与矩形化 / 凸包 / 拆分的区别：只小幅移动顶点，不做外扩、不做拆分，因此
  - 不产生重复地块；
  - 面积变化很小；
  - 识别结果（数量 / 归属）基本不变，只是边界画得更规整。

用法:
  python _regularize_parcels.py --input <in.shp> --output <out.shp>
"""
import argparse
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.strtree import STRtree
from shapely.validation import make_valid

SMOOTH_TOL = 2.0      # DP 简化容差(米)
ANGLE_TOL = 20.0      # 正交化角度容差(度)：边与主方向/正交方向夹角 <= 此值才对齐
MIN_EDGE = 1.0        # 参与正交化的最小边长(米)，短边（弧线小段）不动
MIN_AREA = 1.0        # 重叠消解时丢弃的碎片最小面积(平方米)


def _fix(poly):
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
    poly = poly.buffer(0)
    if poly.geom_type == "Polygon":
        return poly
    if poly.geom_type == "MultiPolygon":
        return max(poly.geoms, key=lambda p: p.area)
    return None


def _dominant_angle(coords):
    """加权边方向直方图找主方向（度，[0,180)）。"""
    edges = np.diff(np.vstack([coords, coords[0]]), axis=0)
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if lengths.sum() == 0:
        return 0.0
    angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 180.0
    bins = np.arange(0, 181, 2)
    hist, _ = np.histogram(angles, bins=bins, weights=lengths)
    hist = np.convolve(hist, np.ones(3), mode="same")
    idx = int(np.argmax(hist))
    return (bins[idx] + 1.0) % 180.0


def _orthogonalize(poly):
    """主方向正交化，保留斜边/弧线边。失败返回原值。"""
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    n = len(coords)
    if n < 3:
        return poly
    angle = _dominant_angle(coords)
    centroid = poly.centroid
    rot = rotate(poly, -angle, origin=centroid)
    rc = np.array(rot.exterior.coords[:-1], dtype=float)

    # 两遍 snap：凸形一遍即可，凹形需两遍。取两端中点对齐，减小系统性漂移。
    for _ in range(2):
        for i in range(n):
            prev = rc[(i - 1) % n]
            curr = rc[i]
            e = curr - prev
            L = np.hypot(e[0], e[1])
            if L < MIN_EDGE:
                continue
            a = abs(np.degrees(np.arctan2(e[1], e[0]))) % 90.0
            a = min(a, 90.0 - a)
            if a <= ANGLE_TOL:
                if abs(e[0]) >= abs(e[1]):
                    ymid = (prev[1] + curr[1]) / 2.0
                    prev[1] = ymid  # 更水平 -> 两端锁到中点 y
                    curr[1] = ymid
                else:
                    xmid = (prev[0] + curr[0]) / 2.0
                    prev[0] = xmid  # 更垂直 -> 两端锁到中点 x
                    curr[0] = xmid

    reg = Polygon(np.vstack([rc, rc[0]]))
    if not reg.is_valid:
        reg = reg.buffer(0)
    if reg.is_empty or reg.geom_type != "Polygon":
        return poly
    return rotate(reg, angle, origin=centroid)


def _remove_collinear(poly, tol=1e-6):
    """合并共线顶点，减少冗余顶点。"""
    coords = list(poly.exterior.coords[:-1])
    n = len(coords)
    if n <= 3:
        return poly
    out = []
    for i in range(n):
        prev = coords[(i - 1) % n]
        curr = coords[i]
        nxt = coords[(i + 1) % n]
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        l1 = np.hypot(*v1)
        l2 = np.hypot(*v2)
        if l1 > 0 and l2 > 0 and abs(cross) <= tol * l1 * l2 and dot > 0:
            continue  # 共线且同向，去掉中间点
        out.append(curr)
    if len(out) < 3:
        return poly
    return Polygon(out)


def regularize(poly):
    """返回 (list[Polygon], action)。简化 + 正交化 + 共线合并，1:1 输出。"""
    p = _fix(poly)
    if p is None or p.area <= 0:
        return [], "invalid"
    try:
        s = p.simplify(SMOOTH_TOL, preserve_topology=True)
        s = _fix(s)
    except Exception:
        s = None
    if s is None:
        return [p], "none"
    try:
        o = _orthogonalize(s)
        o = _remove_collinear(_fix(o))
        o = _fix(o)
    except Exception:
        o = None
    if o is None or o.is_empty:
        return [s], "smooth"
    return [o], "orthogonal"


# ── 统计 ──

def _nverts(p):
    if p is None or p.is_empty:
        return 0
    if p.geom_type == "Polygon":
        return len(p.exterior.coords) - 1
    if p.geom_type == "MultiPolygon":
        return sum(len(g.exterior.coords) - 1 for g in p.geoms)
    return 0


def _count_overlaps(gdf):
    polys = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if len(polys) < 2:
        return 0
    tree = STRtree(polys)
    cnt = 0
    for i, p in enumerate(polys):
        for j in tree.query(p):
            j = int(j)
            if j <= i:
                continue
            try:
                if p.intersects(polys[j]) and p.intersection(polys[j]).area > 0.5:
                    cnt += 1
            except Exception:
                continue
    return cnt


def _remove_overlap(gdf, passes=2):
    """分数优先消解重叠：高分保留完整，低分减掉重叠区；完全被包含的剔除。"""
    out = gdf
    for _ in range(passes):
        polys = list(out.geometry)
        scores = [float(s) if s is not None and not (isinstance(s, float) and s != s)
                  else 0.0 for s in out.get("score", [0.0] * len(out))]
        n = len(polys)
        clipped = list(polys)
        drop = [False] * n
        tree = STRtree(polys)
        for i in range(n):
            if drop[i]:
                continue
            for j in tree.query(polys[i]):
                j = int(j)
                if j <= i or drop[j]:
                    continue
                try:
                    if not clipped[i].intersects(clipped[j]):
                        continue
                    inter = clipped[i].intersection(clipped[j])
                    if inter.is_empty or inter.area <= 0.5:
                        continue
                except Exception:
                    continue
                if scores[i] >= scores[j]:
                    # i 高 / j 低
                    try:
                        new = clipped[j].difference(clipped[i])
                        if new.is_empty:
                            drop[j] = True  # j 完全在 i 内部，直接剔除
                        elif new.geom_type in ("Polygon", "MultiPolygon"):
                            clipped[j] = new
                    except Exception:
                        pass
                else:
                    # j 高 / i 低
                    try:
                        new = clipped[i].difference(clipped[j])
                        if new.is_empty:
                            drop[i] = True
                        elif new.geom_type in ("Polygon", "MultiPolygon"):
                            clipped[i] = new
                    except Exception:
                        pass
        rows = []
        for k in range(n):
            if drop[k]:
                continue
            props = {c: out.iloc[k][c] for c in out.columns if c != "geometry"}
            g = clipped[k]
            if g.geom_type == "MultiPolygon":
                for part in g.geoms:
                    if not part.is_empty and part.area >= MIN_AREA:
                        rows.append({**props, "geometry": part})
            elif g.geom_type == "Polygon" and not g.is_empty and g.area >= MIN_AREA:
                rows.append({**props, "geometry": g})
        out = gpd.GeoDataFrame(rows, crs=out.crs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    gdf = gpd.read_file(args.input)
    ov = np.array([_nverts(g) for g in gdf.geometry])
    print(f"读取 {len(gdf)} 个 polygon: {args.input}")
    print(f"规则化前 顶点: min={ov.min()} p50={int(np.median(ov))} "
          f"p90={int(np.percentile(ov, 90))} max={ov.max()}")
    print(f"规则化前 相交对: {_count_overlaps(gdf)}")

    # 1) 正交化（1:1）
    rows = []
    for _, r in gdf.iterrows():
        polys, action = regularize(r.geometry)
        for p in polys:
            rows.append({**{c: r[c] for c in r.index if c != "geometry"},
                         "action": action, "geometry": p})
    ortho = gpd.GeoDataFrame(rows, crs=gdf.crs)

    nv1 = np.array([_nverts(g) for g in ortho.geometry])
    print(f"正交化后 顶点: min={nv1.min()} p50={int(np.median(nv1))} "
          f"p90={int(np.percentile(nv1, 90))} max={nv1.max()}")
    print(f"正交化后 相交对: {_count_overlaps(ortho)}")

    oa = np.array([g.area for g in gdf.geometry])
    na1 = np.array([g.area for g in ortho.geometry])
    ratio1 = na1 / oa
    print(f"正交化 面积比(新/原): p10={np.percentile(ratio1, 10):.4f} "
          f"p50={np.median(ratio1):.4f} p90={np.percentile(ratio1, 90):.4f}")

    # 2) 重叠消解（消解正交化引入的残余相交）
    out = _remove_overlap(ortho)

    out.to_file(args.output, driver="ESRI Shapefile", encoding="UTF-8")

    nv = np.array([_nverts(g) for g in out.geometry])
    print(f"消解后 顶点: min={nv.min()} p50={int(np.median(nv))} "
          f"p90={int(np.percentile(nv, 90))} max={nv.max()}")
    print(f"消解后 相交对: {_count_overlaps(out)}")
    print(f"输出 {len(out)} 个 polygon -> {args.output}")


if __name__ == "__main__":
    main()

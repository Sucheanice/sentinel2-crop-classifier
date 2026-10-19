#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parcel_post.py — 地块后处理（替代引擎里"无约束连通合并"）

背景
----
geo_infer_v10 的 `_merge_touching_and_fill` 是**按连通组取并集**：地里相邻地块本就连在一起，
于是整片田被合成巨块；耕地挨着水塘、水塘挨着林子时还会**串成一块**（同一图斑里既有水体又有耕地）。
本脚本把这件事改成**受控**的两步：

  1) 水体交叉剔除：耕地块被水体覆盖 ≥ `--water-ratio` → 剔除（水体有自己的层，不该算耕地）
  2) 小碎块并入邻居：仅对面积 < `--merge-small-mu` 亩的碎块，并给**共享边界最长**的邻居，
     且共享边长度须 ≥ `--min-shared-edge` 米（只点接触不并）。不跨大块合并、不做连通传播。

用法
----
  python parcel_post.py --parcels w03_fix.shp --water w03_water.shp --out w03_post.shp \
      --water-ratio 0.5 --merge-small-mu 0.3
"""
import argparse
import sys

import fiona
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.stdout.reconfigure(encoding="utf-8")
T = "EPSG:4523"
MU = 0.0015


def main():
    ap = argparse.ArgumentParser(description="地块后处理：水体剔除 + 小碎块并入邻居")
    ap.add_argument("--parcels", required=True)
    ap.add_argument("--water", help="水体层（可选；不给则跳过水体剔除）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--water-ratio", type=float, default=0.5)
    ap.add_argument("--merge-small-mu", type=float, default=0.3,
                    help="面积小于该值(亩)的碎块并入邻居；0=不并")
    ap.add_argument("--min-shared-edge", type=float, default=1.0, help="共享边界最短长度(米)")
    # ── 特征规则（需先跑 parcel_features.py，输入带 lum_p10/grain/cmpt/n_vert 等列）──
    ap.add_argument("--drop-dark-forest", type=int, default=0,
                    help="1=剔除林地误检（默认 lum_p10 + grain 双条件）")
    ap.add_argument("--forest-p10", type=float, default=35.0)
    ap.add_argument("--forest-grain", type=float, default=3.5)
    ap.add_argument("--forest-grain-only", type=int, default=0,
                    help="1=林地只用 grain 判（去掉 lum_p10 条件）。"
                         "实测：保护暗耕地靠 grain，亮度条件只会漏掉『亮而有纹理』的林地/种植园")
    ap.add_argument("--drop-mixed-shape", type=int, default=0,
                    help="1=剔除『又碎又毛糙』的混类图斑（cmpt + n_vert）")
    ap.add_argument("--mixed-cmpt", type=float, default=0.15)
    ap.add_argument("--mixed-verts", type=int, default=100)
    ap.add_argument("--flag-water", type=int, default=0,
                    help="1=加 flag_water 列（暗且平滑=水体候选），只标记不删")
    ap.add_argument("--water-p10", type=float, default=30.0,
                    help="实测：30 时误标 4/64；放宽到 40 会误标 37/64")
    ap.add_argument("--water-grain", type=float, default=2.0)
    ap.add_argument("--id-offset", type=int, default=0,
                    help="id 重编号基数：输出 id = id-offset + 1..N（多块实验时保证全局唯一）")
    ap.add_argument("--keep-id", type=int, default=0,
                    help="1=保留输入的 id 不变（推荐）：这样交付件的 id 跨版本稳定，"
                         "人工反馈的 id 永远对得上")
    ap.add_argument("--chunk", type=int, default=0, help="写入/覆盖 chunk 字段（块号）")
    args = ap.parse_args()

    g = gpd.read_file(args.parcels).to_crs(T).reset_index(drop=True)
    g["area_mu"] = g.geometry.area * MU
    n0 = len(g)
    print(f"[输入] 耕地 {n0} 块 / {g['area_mu'].sum():.1f} 亩")

    # ── 0) 特征规则剔除（林地/混类）与水体标记 ──
    if args.drop_mixed_shape and "cmpt" in g.columns:
        m = (g["cmpt"].astype(float) < args.mixed_cmpt) & \
            (g["n_vert"].astype(float) > args.mixed_verts)
        print(f"[规则·混类] cmpt<{args.mixed_cmpt} 且 n_vert>{args.mixed_verts}: "
              f"{int(m.sum())} 块 / {g.geometry.area[m].sum()*MU:.1f} 亩 → 删")
        g = g[~m].reset_index(drop=True)
    if args.drop_dark_forest and "grain" in g.columns:
        if args.forest_grain_only:
            m = g["grain"].astype(float) > args.forest_grain
            print(f"[规则·林地] 仅 grain>{args.forest_grain}（不带亮度条件）: "
                  f"{int(m.sum())} 块 / {g.geometry.area[m].sum()*MU:.1f} 亩 → 删")
        else:
            m = (g["lum_p10"].astype(float) < args.forest_p10) & \
                (g["grain"].astype(float) > args.forest_grain)
            print(f"[规则·林地] lum_p10<{args.forest_p10} 且 grain>{args.forest_grain}: "
                  f"{int(m.sum())} 块 / {g.geometry.area[m].sum()*MU:.1f} 亩 → 删")
        g = g[~m].reset_index(drop=True)
    if args.flag_water and "grain" in g.columns:
        g["flag_water"] = ((g["lum_p10"].astype(float) < args.water_p10) &
                           (g["grain"].astype(float) < args.water_grain)).astype(int)
        print(f"[规则·水体候选] 标记 {int(g['flag_water'].sum())} 块（只标记不删）")


    # ── 1) 水体交叉剔除 ──
    wratio = np.zeros(len(g))
    if args.water:
        w = gpd.read_file(args.water).to_crs(T)
        wt = list(w.geometry)
        tree = w.sindex
        for i, geom in enumerate(g.geometry):
            a = geom.area
            if a <= 0:
                continue
            inter = 0.0
            for j in tree.query(geom):
                it = geom.intersection(wt[j]).area
                if it > 1.0:
                    inter += it
            wratio[i] = min(inter / a, 1.0)
        g["wratio"] = wratio.round(4)
        drop = g["wratio"] >= args.water_ratio
        print(f"[水体剔除] 覆盖≥{args.water_ratio:.0%} 的耕地块: {int(drop.sum())} 块 "
              f"({g.geometry.area[drop].sum()*MU:.1f} 亩) → 删")
        g = g[~drop].reset_index(drop=True)
    else:
        g["wratio"] = 0.0

    # ── 2) 小碎块并入共享边最长的邻居 ──
    nmerged = np.zeros(len(g), dtype=int)
    if args.merge_small_mu > 0 and len(g) > 1:
        polys = list(g.geometry)
        props = g.drop(columns="geometry").to_dict("records")
        alive = [True] * len(polys)
        order = sorted(range(len(polys)), key=lambda i: polys[i].area)  # 从小到大
        n_merge = 0
        for i in order:
            if not alive[i]:
                continue
            if polys[i].area * MU >= args.merge_small_mu:
                break  # 后面都更大
            # 找共享边界最长的存活邻居
            best_j, best_len = -1, 0.0
            for j in range(len(polys)):
                if j == i or not alive[j]:
                    continue
                pj = polys[j]
                if not polys[i].intersects(pj):
                    continue
                try:
                    L = polys[i].boundary.intersection(pj.boundary).length
                except Exception:
                    continue
                if L > best_len:
                    best_len, best_j = L, j
            if best_j < 0 or best_len < args.min_shared_edge:
                continue  # 没有像样的邻居，保留
            try:
                polys[best_j] = unary_union([polys[best_j], polys[i]])
            except Exception:
                continue
            alive[i] = False
            nmerged[best_j] += 1 + nmerged[i]
            n_merge += 1
        print(f"[碎块并入] 并掉 <{args.merge_small_mu} 亩的碎块 {n_merge} 块 → 剩余 "
              f"{sum(alive)} 块")
        keep = [k for k in range(len(polys)) if alive[k]]
        out = gpd.GeoDataFrame(
            {**{c: [props[k][c] for k in keep] for c in g.columns if c != "geometry"},
             "nmerged": [nmerged[k] for k in keep]},
            geometry=[polys[k] for k in keep], crs=T)
    else:
        out = g.copy()
        out["nmerged"] = 0

    out = out.reset_index(drop=True)
    if "area_m2" not in out.columns:
        out["area_m2"] = out.geometry.area.round(2)
    if not args.keep_id:
        out["id"] = args.id_offset + np.arange(1, len(out) + 1)
    if args.chunk:
        out["chunk"] = int(args.chunk)
    out.to_file(args.out, driver="ESRI Shapefile", encoding="UTF-8")
    mu = out.geometry.area * MU
    print(f"[输出] {len(out)} 块 / {mu.sum():.1f} 亩  最大 {mu.max():.1f} 亩  "
          f"中位 {mu.median():.2f} 亩  id {int(out['id'].min())}~{int(out['id'].max())}"
          f"  → {args.out}")


if __name__ == "__main__":
    main()

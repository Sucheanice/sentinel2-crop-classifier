# -*- coding: utf-8 -*-
"""前序数据检查与处理：耕地层与水体层的交叉检查 + 水体细分为鱼塘/水体

为什么需要这一步
----------------
前序按影像出两层：耕地地块（farmland）、水体（pond）。两者会互相"串味"：

1. **耕地层里混进了水体块**。SAM3 用 farmland 提示也会在水面（尤其带水草/藻类的塘）上响应，
   这些块被当耕地打了作物标签（实测 26 号 AOI：86 块被水体覆盖 ≥50%，其中 68 块被判成水稻/玉米）。
   水体面已由水体层表达，因此这些块应从耕地类里剔除，也不参与作物赋值。
2. **水体层需要分「鱼塘」和「水体」**。交付口径里鱼塘属养殖地块（农业生产分类 B03 /
   用地分类 A1104 养殖水面），河流、水库等属其他水面（用地分类 B1199），两者不是一回事。
   判据（按业务口径）：**形状方正**（偏矩形/正方形）且 **面积较小** 的判为鱼塘。

用法
----
    python check_frontend_layers.py --farmland 绵阳_farmland_merged.shp --pond 绵阳_pond_merged.shp \
        --out-dir 前序检查 [--plot]

输出
----
    <out>_farmland_clean.gpkg   剔除水体块后的耕地层（多一列 water_ratio 记录被水体覆盖比例）
    <out>_pond_kind.gpkg        水体层（多两列 squareness 方正度、kind 鱼塘/水体）
    <out>_water_preview.png     可选：形状判据散点图 + 分布图，便于核定阈值
"""
import argparse
import os
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

sys.stdout.reconfigure(encoding="utf-8")

TARGET_CRS = "EPSG:4523"
MU_FACTOR = 0.0015  # 亩 = m² × 0.0015


def squareness(geom):
    """方正度 = 自身面积 / 最小外接旋转矩形面积；越接近 1 越方正（矩形/正方形）"""
    mrr = geom.minimum_rotated_rectangle
    if mrr.is_empty or mrr.area <= 0:
        return 0.0
    return min(geom.area / mrr.area, 1.0)


def water_overlap_ratio(farmland, pond):
    """每个耕地块被水体覆盖的比例"""
    tree = pond.sindex
    pg = list(pond.geometry)
    out = np.zeros(len(farmland))
    for i, g in enumerate(farmland.geometry):
        if g.area <= 0:
            continue
        a = 0.0
        for j in tree.query(g):
            inter = g.intersection(pg[j]).area
            if inter > 1.0:
                a += inter
        out[i] = min(a / g.area, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="前序数据检查与处理（水体/耕地交叉 + 鱼塘/水体细分）")
    ap.add_argument("--farmland", required=True, help="耕地层（拼好的）")
    ap.add_argument("--pond", required=True, help="水体层（拼好的）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="成果")
    ap.add_argument("--water-ratio", type=float, default=0.5,
                    help="耕地块被水体覆盖超过该比例即判为水体并剔除，默认 0.5")
    ap.add_argument("--square-min", type=float, default=0.6,
                    help="判鱼塘的方正度下限（面积/最小外接矩形），默认 0.6")
    ap.add_argument("--area-max-mu", type=float, default=5.0,
                    help="判鱼塘的面积上限（亩），默认 5")
    ap.add_argument("--plot", action="store_true", help="输出判据散点图与分布图")
    args = ap.parse_args()

    fa = gpd.read_file(args.farmland).to_crs(TARGET_CRS).reset_index(drop=True)
    po = gpd.read_file(args.pond).to_crs(TARGET_CRS).reset_index(drop=True)
    print(f"[输入] 耕地层 {len(fa)} 块 / {fa.geometry.area.sum()*MU_FACTOR:.1f} 亩")
    print(f"[输入] 水体层 {len(po)} 块 / {po.geometry.area.sum()*MU_FACTOR:.1f} 亩")

    # 1) 耕地层里的水体块
    fa["water_ratio"] = water_overlap_ratio(fa, po).round(4)
    drop = fa["water_ratio"] >= args.water_ratio
    print(f"\n[耕地层检查] 被水体覆盖 >= {args.water_ratio:.0%} 的块 = {int(drop.sum())} 块 "
          f"({fa.geometry.area[drop].sum()*MU_FACTOR:.1f} 亩) -> 从耕地类剔除")
    clean = fa[~drop].copy().reset_index(drop=True)

    # 2) 水体层细分 鱼塘 / 水体
    po["面积/亩"] = (po.geometry.area * MU_FACTOR).round(4)
    po["squareness"] = [round(squareness(g), 4) for g in po.geometry]
    is_pond = (po["squareness"] >= args.square_min) & (po["面积/亩"] <= args.area_max_mu)
    po["kind"] = np.where(is_pond, "鱼塘", "水体")
    print(f"\n[水体层细分] 方正度 >= {args.square_min} 且 面积 <= {args.area_max_mu} 亩 -> 鱼塘")
    for k, sub in po.groupby("kind"):
        print(f"   {k}: {len(sub)} 块 / {sub['面积/亩'].sum():.1f} 亩 "
              f"(方正度中位 {sub['squareness'].median():.2f}, 面积中位 {sub['面积/亩'].median():.2f} 亩)")

    os.makedirs(args.out_dir, exist_ok=True)
    fpath = os.path.join(args.out_dir, f"{args.prefix}_farmland_clean.gpkg")
    ppath = os.path.join(args.out_dir, f"{args.prefix}_pond_kind.gpkg")
    clean.to_file(fpath, driver="GPKG", encoding="utf-8")
    po.to_file(ppath, driver="GPKG", encoding="utf-8")
    print(f"\n[输出] {fpath}  ({len(clean)} 块)")
    print(f"[输出] {ppath}  ({len(po)} 块)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
        handles = []
        for k, c in [("鱼塘", "#1f77b4"), ("水体", "#d62728")]:
            sub = po[po.kind == k]
            if len(sub):
                sub.plot(ax=axes[0], color=c, edgecolor="none")
                handles.append(mpatches.Patch(color=c, label=f"{k} ({len(sub)}块 / {sub['面积/亩'].sum():.0f}亩)"))
        axes[0].set_axis_off()
        axes[0].legend(handles=handles, loc="lower right", fontsize=11)
        axes[0].set_title(f"水体层细分（方正度>={args.square_min} 且 面积<={args.area_max_mu}亩 判鱼塘）")
        for k, c in [("鱼塘", "#1f77b4"), ("水体", "#d62728")]:
            sub = po[po.kind == k]
            axes[1].scatter(sub["面积/亩"], sub["squareness"], s=22, c=c, label=k, alpha=0.8)
        axes[1].axhline(args.square_min, ls="--", c="gray")
        axes[1].axvline(args.area_max_mu, ls="--", c="gray")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("面积（亩，对数轴）")
        axes[1].set_ylabel("方正度 = 面积 / 最小外接矩形面积")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.savefig(os.path.join(args.out_dir, f"{args.prefix}_water_preview.png"),
                    bbox_inches="tight", facecolor="white")
        print(f"[输出] {os.path.join(args.out_dir, args.prefix + '_water_preview.png')}")


if __name__ == "__main__":
    main()

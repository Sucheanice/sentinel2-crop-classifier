# -*- coding: utf-8 -*-
"""耕地 shp 跨影像拼接：重叠带按"谁的分割更完整就用谁"归属

背景
----
一个村落可能横跨多幅影像，前序会分别得到「绵阳1_farmland.shp」「绵阳2_farmland.shp」等。
这些 shp 的外包框基本等于各自影像的覆盖范围，而影像被自身边界切断时，边界附近的地块只有
半块（不完整）。因此重叠带不能简单"以某一幅为主"，而要按区域归属。

归属判据（核心）
----------------
对任意位置，比较"该位置到各幅影像自身边界的最近距离"：
    距离越大 → 该影像在此处越靠内部 → 分割越完整 → 采用该幅
形式化为（影像 i 的外包框为 [x0,y0,x1,y1]，点 p）：
    d_i(p) = min(p.x - x0, x1 - p.x, p.y - y0, y1 - p.y)
    归属 = argmax_i d_i(p)，逐地块按质心判定

该判据天然同时覆盖三种叠置情形（无需人工选方向）：
  * 南北单交（上下两幅，x 范围相同）→ 分界退化为一条水平中线
  * 东西单交（左右两幅，y 范围相同）→ 分界退化为一条垂直中线
  * 双交（既南北错开、又东西错开，如本项目）→ 分界是一条阶梯折线，四个方向的边界都被考虑

拼合方式（贪心 + 裁剪）
----------------------
按「归属侧优先 → score 高优先 → 面积大优先」的顺序逐块并入，每块先裁掉与已并入内容重叠的
部分，再判断是否纳入：
  * 裁完剩余面积 < KEEP_RATIO × 原面积 → 判为重复地块，丢弃
  * 非归属侧候选裁完剩余 < MIN_FILL_AREA → 判为细碎条带，丢弃（避免碎渣）
结果天然满足：无重叠、无缝补（归属侧没识别到的部分由对侧补齐）、跨分界线的地块不会被切成半块。

面积一律按几何重算：area_m2 = 几何面积，area_px = area_m2 / --pixel-area

用法
----
    # 1) 最简：给目录，自动取其中全部 *_farmland.shp（必须恰好 2 个）
    python merge_farmland_shp.py --dir "E:\\...\\26\\farmland" --out "E:\\...\\农田_farmland.shp"

    # 2) 显式指定两幅输入
    python merge_farmland_shp.py --a 绵阳1_farmland.shp --b 绵阳2_farmland.shp --out merged.shp

    # 3) 手动指定方向（当外包框不能代表影像范围时用；默认 auto 不用填）
    python merge_farmland_shp.py --dir ... --out ... --mode y --split-ratio 0.5

    # 4) 输出附加来源字段（便于核对接缝，默认不加）
    python merge_farmland_shp.py --dir ... --out ... --add-src

    # 5) 同一套逻辑用于其他地类层（如水塘），改文件名匹配即可
    python merge_farmland_shp.py --dir "E:\\...\\26\\pond" --pattern "*_pond.shp" --out "...\\绵阳_pond_merged.shp"

输出字段：id, area_px, area_m2, score, class, code, [原有其余属性字段原样透传], geometry
（--add-src 时再多一列 src 记录地块来源）
"""
import argparse
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.stdout.reconfigure(encoding="utf-8")

MIN_OVERLAP_AREA = 1.0   # 判定"两地块重叠"的面积门槛（m²）
KEEP_RATIO = 0.3         # 与已有内容重叠比例超过 1-KEEP_RATIO 且残留很小时判为重复地块
SLIVER_MAX = 5000.0      # 残留小于此面积（m²）才可能被判为重复地块的"壳"
MIN_FILL_AREA = 300.0    # 对侧补块的最小面积（m²），低于此视为细碎条带丢弃
MIN_PIECE_AREA = 100.0   # 裁剪产生的碎片面积下限（m²），低于此丢弃
TREE_REBUILD_STEP = 500  # 空间索引重建步长（越小越准越慢）


# --------------------------------------------------------------------------- #
# 归属判定
# --------------------------------------------------------------------------- #
def _edge_dist(bounds, xs, ys, use_x=True, use_y=True):
    """点到外包框"发生交接方向"上边界的最近距离，越小说明在此处越可能被切断

    影像范围完全对齐的那个方向两幅都是同样的边界（不存在谁更完整），因此不参与比较，
    否则共享边距离会打平并干扰归属判定。
    """
    x0, y0, x1, y1 = bounds
    terms = []
    if use_x:
        terms += [xs - x0, x1 - xs]
    if use_y:
        terms += [ys - y0, y1 - ys]
    if not terms:  # 两个方向都没有交接 → 退回四边距离
        terms = [xs - x0, x1 - xs, ys - y0, y1 - ys]
    return np.minimum.reduce(terms)


def _handover(bounds_a, bounds_b, k):
    """沿第 k 轴（0=x,1=y）是否形成交接：一幅更靠小值侧、另一幅更靠大值侧"""
    a0, a1 = bounds_a[k], bounds_a[k + 2]
    b0, b1 = bounds_b[k], bounds_b[k + 2]
    return (a0 < b0 and a1 < b1) or (b0 < a0 and b1 < a1)


def pair_axes(bxA, bxB):
    """返回 (use_x, use_y)：两个方向是否参与归属比较"""
    return _handover(bxA, bxB, 0), _handover(bxA, bxB, 1)


def ownership(gA, gB, mode="auto", axis="y", split_ratio=0.5):
    """返回 (keepA, keepB)：两幅各自"归属给自己"的地块布尔数组"""
    bxA, bxB = gA.total_bounds, gB.total_bounds
    cxA, cyA = gA.geometry.centroid.x.values, gA.geometry.centroid.y.values
    cxB, cyB = gB.geometry.centroid.x.values, gB.geometry.centroid.y.values

    if mode == "auto":
        use_x, use_y = pair_axes(bxA, bxB)
        # 用 > 与 >= 错开，保证同一位置不会两幅都算归属
        keepA = _edge_dist(bxA, cxA, cyA, use_x, use_y) >= _edge_dist(bxB, cxA, cyA, use_x, use_y)
        keepB = _edge_dist(bxB, cxB, cyB, use_x, use_y) > _edge_dist(bxA, cxB, cyB, use_x, use_y)
        return keepA, keepB

    # 手动模式：沿指定轴在重叠带内切一条线
    k = 1 if axis == "y" else 0
    lo = max(bxA[k], bxB[k])
    hi = min(bxA[k + 2], bxB[k + 2])
    if lo >= hi:
        return np.ones(len(gA), bool), np.ones(len(gB), bool)
    split = lo + split_ratio * (hi - lo)
    pa = cyA if axis == "y" else cxA
    pb = cyB if axis == "y" else cxB
    low_is_a = bxA[k] <= bxB[k]  # 低侧影像=沿轴更靠小值、延伸更远的一幅
    if low_is_a:
        return pa < split, pb >= split
    return pa >= split, pb < split


# --------------------------------------------------------------------------- #
# 拼合
# --------------------------------------------------------------------------- #
def _extra_cols(gdf):
    """除几何与本脚本会重算的字段外，其余属性字段原样透传"""
    fixed = {"geometry", "id", "area_px", "area_m2", "score", "class", "code"}
    return [c for c in gdf.columns if c not in fixed]


def _item(gdf, i, src, owner, extra=()):
    r = gdf.iloc[i]
    return {
        "score": float(r["score"]) if "score" in gdf.columns else 0.0,
        "class": r["class"] if "class" in gdf.columns else "farmland",
        "code": r["code"] if "code" in gdf.columns else "A01",
        "src": src,
        "owner": owner,
        **{c: r[c] for c in extra},
        "geometry": r.geometry,
    }


def _polys(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return [p for p in geom.geoms if p.geom_type == "Polygon" and not p.is_empty]
    return []


def _assemble(items, min_ov, keep_ratio, min_fill, rebuild_step,
              sliver_max=SLIVER_MAX, min_piece=MIN_PIECE_AREA):
    """贪心并入：归属侧优先，逐块裁掉与已并入内容重叠的部分，再判断是否保留"""
    order = sorted(range(len(items)),
                   key=lambda i: (0 if items[i]["owner"] else 1,
                                  -items[i]["score"],
                                  -items[i]["geometry"].area))
    occupied, out, tail = [], [], []  # tail = 上次重建索引之后新增的部分 (bounds, geom)
    tree, last_build = None, 0
    stat = {"owner": 0, "fill": 0, "drop_dup": 0, "drop_sliver": 0}

    for i in order:
        it = items[i]
        g = it["geometry"]
        gb = g.bounds
        rem, cut = g, []
        if occupied:
            if tree is None or len(occupied) - last_build >= rebuild_step:
                tree, last_build, tail = STRtree(occupied), len(occupied), []
            # 索引里的部分用空间索引查，索引之后新增的部分（tail）直接查，避免漏判造成重叠
            near = [occupied[j] for j in tree.query(g)]
            near += [o for bb, o in tail
                     if bb[2] > gb[0] and bb[0] < gb[2] and bb[3] > gb[1] and bb[1] < gb[3]]
            cut = [o for o in near if g.intersection(o).area > min_ov]
            if cut:
                rem = g.difference(unary_union(cut))
        parts = _polys(rem)
        # 裁剪过的按碎片下限过滤，未裁剪的（源地块）只剔除空几何以保持原样
        parts = [p for p in parts if p.area >= min_piece] if cut else parts
        total = sum(p.area for p in parts)
        # 残留既占比很小、又绝对面积很小 → 判为重复地块的"壳"，丢弃
        if total < keep_ratio * g.area and total < sliver_max:
            stat["drop_dup"] += 1
            continue
        if not it["owner"] and total < min_fill:  # 对侧补块太碎
            stat["drop_sliver"] += 1
            continue
        if it["owner"]:
            stat["owner"] += 1
        else:
            stat["fill"] += 1
        for p in parts:
            out.append({**{k: v for k, v in it.items() if k != "geometry"}, "geometry": p})
            occupied.append(p)
            tail.append((p.bounds, p))
    return out, stat


def _to_gdf(rows, crs, pixel_area):
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).reset_index(drop=True)
    if out.empty:
        return out
    out["area_m2"] = out.geometry.area.round(2)
    out["area_px"] = (out["area_m2"] / pixel_area).round(0).astype("int64")
    out.insert(0, "id", np.arange(1, len(out) + 1))
    fixed = ["id", "area_px", "area_m2", "score", "class", "code"]
    extra = [c for c in out.columns if c not in fixed + ["src", "owner", "geometry"]]
    return out[fixed + extra + ["src", "geometry"]]


def merge_two(gA, gB, name_a, name_b, mode="auto", axis="y", split_ratio=0.5,
              min_ov=MIN_OVERLAP_AREA, pixel_area=0.64,
              keep_ratio=KEEP_RATIO, min_fill=MIN_FILL_AREA):
    """两幅 shp 拼接，返回 (结果 GeoDataFrame, 统计 dict)"""
    if str(gA.crs) != str(gB.crs):
        raise SystemExit(f"[错误] 两幅坐标系不一致: {gA.crs} / {gB.crs}")

    bxA, bxB = gA.total_bounds, gB.total_bounds
    ov_x = min(bxA[2], bxB[2]) - max(bxA[0], bxB[0])
    ov_y = min(bxA[3], bxB[3]) - max(bxA[1], bxB[1])
    stats = {
        "ov_x": ov_x, "ov_y": ov_y,
        "hand_x": _handover(bxA, bxB, 0), "hand_y": _handover(bxA, bxB, 1),
        "nA": len(gA), "nB": len(gB), "overlap": ov_x > 0 and ov_y > 0,
    }

    keepA, keepB = ownership(gA, gB, mode=mode, axis=axis, split_ratio=split_ratio)
    ex_a, ex_b = _extra_cols(gA), _extra_cols(gB)
    items = [_item(gA, i, name_a, bool(keepA[i]), ex_a) for i in range(len(gA))]
    items += [_item(gB, i, name_b, bool(keepB[i]), ex_b) for i in range(len(gB))]
    if not stats["overlap"]:  # 两幅不重叠，全部并入即可
        items = [{**it, "owner": True} for it in items]
    stats["extra_cols"] = sorted(set(ex_a) | set(ex_b))

    rows, stat = _assemble(items, min_ov, keep_ratio, min_fill, TREE_REBUILD_STEP)
    stats.update(stat)
    src_count = {}
    for r in rows:
        src_count[r["src"]] = src_count.get(r["src"], 0) + 1
    stats["src"] = src_count
    return _to_gdf(rows, gA.crs, pixel_area), stats


# --------------------------------------------------------------------------- #
# 诊断
# --------------------------------------------------------------------------- #
def _count_overlaps(gdf, min_ov=MIN_OVERLAP_AREA):
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    n = 0
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            if j > i and g.intersection(geoms[j]).area > min_ov:
                n += 1
    return n


def _short_tag(name):
    """取一个有区分度的单字符标签，如 绵阳1_farmland.shp -> 1、绵阳2_pond.shp -> 2"""
    stem = os.path.splitext(os.path.basename(name))[0]
    for suf in ("_farmland", "_pond", "_merged"):
        stem = stem.replace(suf, "")
    digits = re.findall(r"\d", stem)
    if digits:
        return digits[-1]
    return stem[:1] if stem else "?"


def ownership_map(gA, gB, name_a, name_b, cols=32, rows=12):
    """打印重叠区归属示意：网格内标出该位置归属哪一幅（北在上）"""
    bxA, bxB = gA.total_bounds, gB.total_bounds
    x0, y0 = max(bxA[0], bxB[0]), max(bxA[1], bxB[1])
    x1, y1 = min(bxA[2], bxB[2]), min(bxA[3], bxB[3])
    dx, dy = (x1 - x0) / cols, (y1 - y0) / rows
    tag_a, tag_b = _short_tag(name_a), _short_tag(name_b)
    use_x, use_y = pair_axes(bxA, bxB)
    print(f"重叠区归属示意（北在上；{tag_a}={name_a}, {tag_b}={name_b}）")
    for r in range(rows - 1, -1, -1):
        line = []
        for c in range(cols):
            px, py = x0 + (c + 0.5) * dx, y0 + (r + 0.5) * dy
            da = _edge_dist(bxA, px, py, use_x, use_y)
            db = _edge_dist(bxB, px, py, use_x, use_y)
            line.append(tag_a if da >= db else tag_b)
        print("  " + "".join(line))


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def _read(path):
    if not os.path.isfile(path):
        raise SystemExit(f"[错误] 输入文件不存在: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        raise SystemExit(f"[错误] 输入文件无要素: {path}")
    return gdf.reset_index(drop=True)


def _find_inputs(folder, pattern="*_farmland.shp"):
    """目录中自动找出两幅输入 shp（排除已拼接的产物）"""
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    files = [f for f in files if "merged" not in os.path.basename(f).lower()]
    if len(files) != 2:
        raise SystemExit(
            f"[错误] 目录中需恰好 2 个待拼接 shp（{pattern}），实际找到 {len(files)} 个:\n  "
            + "\n  ".join(files)
        )
    return files


def main():
    ap = argparse.ArgumentParser(description="耕地 shp 跨影像拼接（重叠带按谁的分割更完整就用谁）")
    ap.add_argument("--dir", help="输入目录（自动取其中恰好 2 个 shp）")
    ap.add_argument("--pattern", default="*_farmland.shp",
                    help="--dir 模式下的文件名匹配，默认 *_farmland.shp（水体用 *_pond.shp）")
    ap.add_argument("--a", help="输入 shp 1")
    ap.add_argument("--b", help="输入 shp 2")
    ap.add_argument("--out", required=True, help="输出 shp 路径")
    ap.add_argument("--mode", choices=["auto", "y", "x"], default="auto",
                    help="归属方式：auto=按到自身影像边界的距离（默认，自动覆盖单交/双交）；"
                         "y/x=沿该轴手动切一条线")
    ap.add_argument("--split-ratio", type=float, default=0.5,
                    help="--mode y/x 时，分界线在重叠带内的位置比例，0.5=中线（默认）")
    ap.add_argument("--pixel-area", type=float, default=0.64, help="单像元面积 m²，默认 0.64")
    ap.add_argument("--add-src", action="store_true", help="输出附加 src 字段记录地块来源")
    args = ap.parse_args()

    if args.dir:
        pa, pb = _find_inputs(args.dir, args.pattern)
    elif args.a and args.b:
        pa, pb = args.a, args.b
    else:
        raise SystemExit("[错误] 需指定 --dir，或同时指定 --a 与 --b")

    gA, gB = _read(pa), _read(pb)
    na, nb = os.path.basename(pa), os.path.basename(pb)
    print(f"输入 A: {na}  {len(gA)} 块")
    print(f"输入 B: {nb}  {len(gB)} 块")
    print(f"坐标系: {gA.crs}")

    merged, s = merge_two(gA, gB, na, nb, mode=args.mode,
                          axis=args.mode if args.mode != "auto" else "y",
                          split_ratio=args.split_ratio, pixel_area=args.pixel_area)

    if s["overlap"]:
        kind = ("东西+南北双交" if s["hand_x"] and s["hand_y"] else
                "南北单交" if s["hand_y"] else
                "东西单交" if s["hand_x"] else "包含/并列（无交接）")
        print(f"叠置方式: {kind}   （交接方向: x={s['hand_x']} y={s['hand_y']}）")
        print(f"重叠范围: x {s['ov_x']:.1f} m, y {s['ov_y']:.1f} m")
        print(f"归属方式: {args.mode}")
        print(f"归属侧保留: {s['owner']} 块   对侧补块(裁剪后): {s['fill']} 块")
        print(f"因重叠过多丢弃(判为重复): {s['drop_dup']} 块   因过碎丢弃: {s['drop_sliver']} 块")
        if args.mode == "auto":
            ownership_map(gA, gB, na, nb)
    else:
        print("[提示] 两幅无重叠，按并列合并")

    print(f"来源分布: {s['src']}")
    if s.get("extra_cols"):
        print(f"原样透传的额外属性字段: {s['extra_cols']}")
    print(f"输出 {len(merged)} 块   面积合计 {merged.area_m2.sum() / 666.667:.1f} 亩   "
          f"最大地块 {merged.geometry.area.max() / 666.667:.1f} 亩")
    print(f"残留面积重叠对: {_count_overlaps(merged)}")

    cols = [c for c in merged.columns if c != "src" or args.add_src]
    out_path = args.out
    try:
        merged[cols].to_file(out_path, driver="ESRI Shapefile", encoding="UTF-8")
    except PermissionError:
        out_path = os.path.splitext(args.out)[0] + "_new.shp"
        print(f"[警告] 输出文件被占用（请关闭 GIS 中打开的文件）: {args.out}")
        merged[cols].to_file(out_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"完成 -> {out_path}")


if __name__ == "__main__":
    main()

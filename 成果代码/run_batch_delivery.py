# -*- coding: utf-8 -*-
"""run_batch_delivery.py — 通用批次驱动：任意前序数据批次 → 我们的赋值 → 单一交付件

**以后来新批次不需要再写新代码**，只要给 `--base`（批次根目录）和若干 `--src`（源 shp）。

复用 `run_my1_delivery.py` 的核心三阶段，口径完全一致：
    GEE 大春作物赋值（水稻/玉米，含年份回退 + missing_rule=two）
    → 「每块必判」交付导出（1 shp + 1 xlsx，非植被→用地分类 C1299 其他未利用地）

源的约定（与本项目同事交付的前序成果一致）：
    * 平铺 shp，`class` 全为 farmland，**无水体层**
    * CRS 为未命名 TM（CM 105°E / FE 35,500,000，等价 EPSG:4523）或任意可转 4523 的 CRS
    * 允许存在少量无效几何（prep 阶段 `buffer(0)` 修复）

镇 → 输出 OUT 按**县区编码前 6 位自动判定**，不需要手工指定：
    510722 三台 → ST_OUT   510723 盐亭 → YT_OUT   510725 梓潼 → ZT_OUT
    510727 平武 → PW_OUT   510781 江油 → JY_OUT

输出：`{base}\\我们的赋值\\{OUT}\\{镇编码}\\交付\\{镇名}_交付.shp / .xlsx / .gpkg`

用法（以 1019 江油批次为例）：
    python run_batch_delivery.py --base "…\\成果数据\\1019 第二版前序数据3" \
        --src "jy\\jy_my9\\my9.shp" --src "jy\\jy_my45\\jy45_merge.shp" \
        --src "jy\\jy_my12\\jy12_merge.shp" --src "jy\\jy_my10\\my10_best.shp"

    python run_batch_delivery.py --base "…\\1016 第二版前序数据2" --src "pw\\pw678_merge.shp"
    python run_batch_delivery.py --base "…\\1014 第二版前序数据" \
        --src "已有成果\\st\\my3.shp" --src "已有成果\\yt\\my3.shp" \
        --src "已有成果\\zt\\zt_my10\\my10_best.shp" --src "已有成果\\zt\\zt_my3\\my3.shp"

    --only 510781101,太平镇     只跑指定镇
    --dry-run                   只读源、报镇与 OUT 归属，不调 GEE
    --force                     忽略已有产物重跑
"""
import argparse
import os
import sys
import traceback

import geopandas as gpd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, BASE)

from run_my1_delivery import (  # noqa: E402
    TARGET_CRS, MU, log, load_villages, stage_gee, stage_deliver)
from sam_pipeline import assign_villages  # noqa: E402

# 县区编码前 6 位 → 输出 OUT 目录（第一批次任务区 5 县）
COUNTY_OUT = {
    "510722": "ST_OUT",   # 三台
    "510723": "YT_OUT",   # 盐亭
    "510725": "ZT_OUT",   # 梓潼
    "510727": "PW_OUT",   # 平武
    "510781": "JY_OUT",   # 江油
}


def town_out(town_code, fallback=""):
    return COUNTY_OUT.get(str(town_code)[:6], fallback)


def load_sources(args):
    """把 --src 展开成绝对路径列表（相对路径按 --base 解析）。"""
    out = []
    for s in args.src:
        p = s if os.path.isabs(s) else os.path.join(args.base, s)
        out.append(p)
    return out


def run_source(src, villages, tmap, args, ok, fail):
    print(f"\n{'='*74}\n{src}\n{'='*74}")
    if not os.path.exists(src):
        log(f"  !! 源不存在，跳过")
        fail.append(f"{src}: 缺失")
        return
    fa = gpd.read_file(src)
    print(f"  源 {len(fa)} 块, 字段 {list(fa.columns)}")
    fa = fa.to_crs(TARGET_CRS).reset_index(drop=True)

    # 无效几何（自相交等）→ buffer(0) 修复；修复后为空则剔除
    bad = ~fa.geometry.is_valid
    if bad.any():
        nbad = int(bad.sum())
        fa.loc[bad, "geometry"] = fa.loc[bad, "geometry"].buffer(0)
        empty = fa.geometry.is_empty
        if empty.any():
            log(f"  [修复] 无效几何 {nbad} → buffer(0) 后空几何 {int(empty.sum())} 块已剔除")
            fa = fa[~empty].reset_index(drop=True)
        else:
            log(f"  [修复] 无效几何 {nbad} 块已 buffer(0) 修复")
    print(f"  投影后 {len(fa)} 块 / {fa.geometry.area.sum()*MU:.1f} 亩")

    fa = assign_villages(fa, villages)
    miss = int(fa["village_code"].astype(str).str.strip().eq("").sum())
    print(f"  村界归属后 无归属 {miss} 块")
    fa["_town"] = fa["village_code"].astype(str).str[:9]

    for town, sub in fa.groupby("_town"):
        name = tmap.get(town, town)
        out_name = town_out(town, args.out_name)
        if not out_name:
            log(f"  !! {town} {name} 的县区编码不在映射表，跳过（可用 --out-name 指定）")
            fail.append(f"{town}/{name}: 未知县区编码")
            continue
        if args.only and not any(k in (town, str(name)) for k in args.keys):
            print(f"  [跳过] {town} {name}（{len(sub)} 块，不在本次范围）")
            continue
        print(f"\n  [{town} {name} → {out_name}] {len(sub)} 块 / {sub.geometry.area.sum()*MU:.1f} 亩")
        if args.dry_run:
            ok.append(f"{out_name}/{town}/{name}")
            continue
        out_dir = os.path.join(args.out_root, out_name, town)
        chk = os.path.join(out_dir, "前序检查")
        os.makedirs(chk, exist_ok=True)
        vill_p = os.path.join(chk, f"{name}_耕地_带村界.gpkg")
        try:
            sub.reset_index(drop=True).to_file(vill_p, driver="GPKG", encoding="utf-8")
            b_ck = os.path.join(out_dir, "作物赋值", f"{name}_B_质检.gpkg")
            if args.force or not os.path.exists(b_ck):
                stage_gee(out_dir, name, vill_p, force=args.force, wide=args.wide,
                          missing_rule=args.missing_rule,
                          fallback_years=args.fallback_years)
            if not os.path.exists(b_ck):
                raise RuntimeError("缺 B_质检 gpkg")
            stage_deliver(out_dir, name, b_ck, None, force=args.force)
            ok.append(f"{out_name}/{town}/{name}")
        except Exception as e:
            log(f"    !! 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail.append(f"{out_name}/{town}/{name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="批次根目录（源与默认输出根的父目录）")
    ap.add_argument("--src", action="append", required=True,
                    help="源 shp（可重复；相对路径按 --base 解析）")
    ap.add_argument("--out-root", help="输出根（默认 {base}\\我们的赋值）")
    ap.add_argument("--out-name", default="", help="县区编码未命中映射时的兜底 OUT 名")
    ap.add_argument("--only", help="只跑匹配的镇编码/镇名（逗号分隔）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wide", action="store_true", help="GEE 用加宽窗口")
    ap.add_argument("--missing-rule", choices=["any", "two"], default="two")
    ap.add_argument("--fallback-years", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="只读源、报镇与 OUT 归属，不调 GEE")
    args = ap.parse_args()
    args.src = load_sources(args)
    args.out_root = args.out_root or os.path.join(args.base, "我们的赋值")
    args.keys = [k.strip() for k in args.only.split(",")] if args.only else []

    print(f"[批次根] {args.base}")
    print(f"[输出根] {args.out_root}")
    print(f"[源] {len(args.src)} 个")
    villages = load_villages()
    print(f"[村界] {len(villages)} 个村")
    tmap = (villages.assign(_t=villages["village_code"].astype(str).str[:9])
            .drop_duplicates("_t").set_index("_t")["乡镇级"].to_dict())

    ok, fail = [], []
    for src in args.src:
        run_source(src, villages, tmap, args, ok, fail)

    if args.dry_run:
        log("\n[dry-run] 未调用 GEE，未产出交付件")
    log(f"\n=== 完成 {len(ok)} / 失败 {len(fail)} ===")
    for f in fail:
        log("  失败 " + f)
    log(f"[输出根] {args.out_root}")


if __name__ == "__main__":
    main()

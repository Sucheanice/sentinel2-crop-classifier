# -*- coding: utf-8 -*-
"""run_1009_delivery.py — 1009 批次前序成果（my3.shp）→ 我们的赋值 → 单一交付件

与 `run_my1_delivery.py` 用同一套 stage（GEE 赋值 + 交付导出）与同一套参数默认值，
差异只在「源数据形态」：

  * 源是**单个平铺 shp** `成果数据\\1009前序环节数据\\my3.shp`，`class` 全为 farmland，
    **没有对应的水体层**（故交付件里不含 鱼塘 / 养殖水面 / 其他水面）
  * 一个文件跨两个镇（石牛镇 510725108 / 玛瑙镇 510725107），按村界归属后**按镇拆分**
  * 输出到 `成果数据\\1009前序环节数据\\我们的赋值\\ZT_OUT\\{镇编码}\\`（不动 0922 已有成果）

用法：
    python run_1009_delivery.py                      # 两个镇都跑
    python run_1009_delivery.py --only 510725107     # 只跑玛瑙镇（也可写镇名）
    python run_1009_delivery.py --force              # 忽略已有产物重跑
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

SRC = os.path.join(ROOT, r"成果数据\1009前序环节数据\my3.shp")
OUT_ROOT = os.path.join(ROOT, r"成果数据\1009前序环节数据\我们的赋值")
OUT_NAME = "ZT_OUT"     # 梓潼县（510725）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑匹配的镇编码/镇名（逗号分隔）")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重跑")
    ap.add_argument("--wide", action="store_true", help="GEE 用加宽窗口")
    ap.add_argument("--missing-rule", choices=["any", "two"], default="two")
    ap.add_argument("--fallback-years", type=int, default=1)
    args = ap.parse_args()

    fa = gpd.read_file(SRC)
    print(f"[源] {SRC}: {len(fa)} 块, CRS={fa.crs}")
    print(f"    字段 {list(fa.columns)}")
    fa = fa.to_crs(TARGET_CRS).reset_index(drop=True)
    print(f"    投影后 {len(fa)} 块 / {fa.geometry.area.sum()*MU:.1f} 亩")

    villages = load_villages()
    print(f"[村界] {len(villages)} 个村（绵阳片区）")
    fa = assign_villages(fa, villages)
    miss = int(fa["village_code"].astype(str).str.strip().eq("").sum())
    print(f"[村界归属] 无归属 {miss} 块")

    # 镇编码 = 村级码前 9 位；镇名从村界库取
    fa["_town"] = fa["village_code"].astype(str).str[:9]
    tmap = (villages.assign(_t=villages["village_code"].astype(str).str[:9])
            .drop_duplicates("_t").set_index("_t")["乡镇级"].to_dict())

    keys = [k.strip() for k in args.only.split(",")] if args.only else []
    ok, fail = [], []
    for town, sub in fa.groupby("_town"):
        name = tmap.get(town, town)
        if keys and not any(k in (town, str(name)) for k in keys):
            continue
        print(f"\n[{town} {name}] {len(sub)} 块 / {sub.geometry.area.sum()*MU:.1f} 亩")
        out_dir = os.path.join(OUT_ROOT, OUT_NAME, town)
        chk = os.path.join(out_dir, "前序检查")
        os.makedirs(chk, exist_ok=True)
        vill_p = os.path.join(chk, f"{name}_耕地_带村界.gpkg")
        try:
            sub = sub.reset_index(drop=True)
            sub.to_file(vill_p, driver="GPKG", encoding="utf-8")
            print(f"    [prep] {vill_p}")

            if args.force or not os.path.exists(
                    os.path.join(out_dir, "作物赋值", f"{name}_B_质检.gpkg")):
                stage_gee(out_dir, name, vill_p, force=args.force, wide=args.wide,
                          missing_rule=args.missing_rule,
                          fallback_years=args.fallback_years)
            b_ck = os.path.join(out_dir, "作物赋值", f"{name}_B_质检.gpkg")
            if not os.path.exists(b_ck):
                raise RuntimeError("缺 B_质检 gpkg")
            stage_deliver(out_dir, name, b_ck, None, force=args.force)
            ok.append(f"{town}/{name}")
        except Exception as e:
            log(f"    !! 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail.append(f"{town}/{name}: {e}")

    log(f"\n=== 完成 {len(ok)} / 失败 {len(fail)} ===")
    for f in fail:
        log("  失败 " + f)
    log(f"[输出根] {os.path.join(OUT_ROOT, OUT_NAME)}")


if __name__ == "__main__":
    main()

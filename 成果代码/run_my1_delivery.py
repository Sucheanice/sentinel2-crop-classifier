# -*- coding: utf-8 -*-
"""run_my1_delivery.py — MY 第一批前序成果 → 我们的赋值 → 交付三类

流水线（与 0917 太平镇口径一致）：
    源耕地/水体层 → 前序检查（剔耕地层里的水体块 + 水体细分鱼塘/水体）
    → 村界归属（最大重叠面积） → GEE 作物赋值（水稻/玉米）
    → 交付单一成果（地块 + 农业生产分类/大春作物/用地分类三类编码并排），shp + xlsx + gpkg

范围：MY第一批成果 下 5 个 OUT（JY/ST/PW/YT/ZT）全部 AOI，按 AOI 分别产出。

用法：
    python run_my1_delivery.py --stage prep            # 阶段1：前序检查 + 村界归属（不连 GEE）
    python run_my1_delivery.py --stage gee             # 阶段2：GEE 作物赋值
    python run_my1_delivery.py --stage deliver         # 阶段3：三类交付导出
    python run_my1_delivery.py --stage all             # 一把跑完
    python run_my1_delivery.py --stage prep --only 510722133   # 只跑匹配的 AOI

可断点续跑：目标文件已存在则跳过（--force 重跑）。
"""
import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
ARCH = os.path.join(ROOT, "归档_作物分类项目")
sys.path.insert(0, ARCH)
sys.path.insert(0, BASE)

from check_frontend_layers import squareness, water_overlap_ratio  # noqa: E402
from sam_pipeline import assign_villages  # noqa: E402

TARGET_CRS = "EPSG:4523"
MU = 0.0015
PRIMARY_YEAR = 2025      # GEE 特征主用年份（模型窗口口径）
WATER_RATIO = 0.50      # 与水体层重叠 >= 该比例 → 候选水体块（不再单独剔除，见 stage_gee）
SQUARE_MIN = 0.60       # 鱼塘判据：方正度下限
AREA_MAX_MU = 5.0       # 鱼塘判据：面积上限（亩）
CROP_LABELS = ("水稻", "玉米")   # 模型判为作物的标签（水体剔除以外的保留项）

# ⚠️ 本文件的 stage_gee / stage_deliver 是通用能力（run_batch_delivery.py 也调用它们）；
#    而 main() / list_aois() / stage_prep() 是 0922 那套「{OUT}/{AOI}/farmland」布局的**旧驱动**，
#    已被 run_batch_delivery.py 取代。0922 数据已删，下面常量指向最新批次，仅避免悬空路径。
#    新批次请用：python run_batch_delivery.py --base "…" --src "…"
MY = os.path.join(ROOT, r"成果数据\1016 第二版前序数据2")
OUT_ROOT = os.path.join(ROOT, r"成果数据\1016 第二版前序数据2\我们的赋值")
VILL = os.path.join(ROOT, r"训练与标注数据\0907村界等数据\绵阳\四川省村界\四川省村界\四川省_村界.shp")
MODEL = os.path.join(ROOT, r"模型\dachun_binary_23counties_3dekad_wide.pkl")
EXPORT = os.path.join(BASE, "export_delivery_xlsx.py")

OUTS = ["JY_OUT", "ST_OUT", "PW_OUT", "YT_OUT", "ZT_OUT"]

# 交付 shp 的字段名（10 字节上限，中文不可用 → 英文缩写）
SHORT = {
    "序号": "XH", "村编码": "CJBM", "村名称": "CJMC", "期数（时间）": "QSSJ",
    "地块唯一编码": "DKYWBM", "实际面积/亩": "SJMJ",
    "农业生产分类编码": "AG_BZBM", "农业生产分类名称": "AG_BZMC",
    "大春作物编码": "CROP_BZBM", "大春作物名称": "CROP_BZMC",
    "用地分类编码": "LAND_BZBM", "用地分类名称": "LAND_BZMC",
}

# 旧的三类交付件（合并前）——新的单一「交付」件产出后清理，避免混淆
LEGACY_TAGS = ["地块边界", "大春作物", "用地分类"]
LEGACY_EXT = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".shp.xml", ".xlsx", ".gpkg"]


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
# AOI 枚举与源解析
# --------------------------------------------------------------------------- #
def list_aois(only=None):
    res = []
    keys = [k.strip() for k in only.split(",")] if only else []
    for o in OUTS:
        od = os.path.join(MY, o)
        if not os.path.isdir(od):
            continue
        for a in sorted(os.listdir(od)):
            ap = os.path.join(od, a)
            if not os.path.isdir(ap) or not os.path.isdir(os.path.join(ap, "farmland")):
                continue
            if keys and not any(k == f"{o}/{a}" or k == o or k == a or k in a or k in o
                                for k in keys):
                continue
            res.append((o, a, ap))
    return res


def _resolve(kind_dir, tag):
    """返回 (源路径或列表, 说明)。优先 多影像拼接 产物，其次单图，最后多图待拼接。"""
    if not os.path.isdir(kind_dir):
        return None, "缺失"
    m = sorted(glob.glob(os.path.join(kind_dir, "*_多影像拼接", f"*_{tag}_拼接.shp")))
    if m:
        return m[0], "拼接"
    raw = sorted(glob.glob(os.path.join(kind_dir, f"*_{tag}.shp")))
    if len(raw) == 0:
        return None, "缺失"
    if len(raw) == 1:
        return raw[0], "单图"
    return raw, f"多图待拼接({len(raw)})"


def merge_multi(paths, out_path, tag):
    """多图无拼接产物时，按 merge_farmland_shp 的两两拼接逻辑顺序合并。"""
    from merge_farmland_shp import merge_two
    cur = gpd.read_file(paths[0]).to_crs(TARGET_CRS)
    cur_name = os.path.basename(paths[0])
    for p in paths[1:]:
        nxt = gpd.read_file(p).to_crs(TARGET_CRS)
        nxt_name = os.path.basename(p)
        cur, _ = merge_two(cur, nxt, cur_name, nxt_name)
        cur_name = out_path
    cur.to_file(out_path, driver="GPKG", encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
# 村界与命名
# --------------------------------------------------------------------------- #
def load_villages():
    v = gpd.read_file(VILL, bbox=(103.8, 31.0, 105.8, 32.7))
    v = v.to_crs(TARGET_CRS).rename(
        columns={"村级码": "village_code", "村级名": "village_name"})
    return v.reset_index(drop=True)


def dominant_town(villages, gdf):
    pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), crs=gdf.crs)
    j = gpd.sjoin(pts, villages[["乡镇级", "geometry"]], predicate="within", how="left")
    s = j["乡镇级"].dropna()
    return s.mode().iloc[0] if len(s) else None


# --------------------------------------------------------------------------- #
# 阶段 1：前序检查 + 村界归属
# --------------------------------------------------------------------------- #
def stage_prep(out_dir, aoi, ap, villages, prefix, force=False):
    chk = os.path.join(out_dir, "前序检查")
    os.makedirs(chk, exist_ok=True)
    clean_p = os.path.join(chk, f"{prefix}_farmland_clean.gpkg")
    kind_p = os.path.join(chk, f"{prefix}_pond_kind.gpkg")
    vill_p = os.path.join(chk, f"{prefix}_耕地_带村界.gpkg")
    if not force and os.path.exists(vill_p) and os.path.exists(kind_p):
        log(f"    [prep] 已存在，跳过")
        return clean_p, kind_p, vill_p

    fsrc, fhow = _resolve(os.path.join(ap, "farmland"), "farmland")
    psrc, phow = _resolve(os.path.join(ap, "pond"), "pond")
    log(f"    耕地源={fhow} 水体源={phow}")
    if isinstance(fsrc, list):
        fsrc = merge_multi(fsrc, os.path.join(chk, f"{prefix}_farmland_merge.gpkg"), "farmland")
    if isinstance(psrc, list):
        psrc = merge_multi(psrc, os.path.join(chk, f"{prefix}_pond_merge.gpkg"), "pond")

    fa = gpd.read_file(fsrc).to_crs(TARGET_CRS).reset_index(drop=True)
    po = gpd.read_file(psrc).to_crs(TARGET_CRS).reset_index(drop=True)
    log(f"    输入 耕地 {len(fa)} 块 / {fa.geometry.area.sum()*MU:.1f} 亩，"
        f"水体 {len(po)} 块 / {po.geometry.area.sum()*MU:.1f} 亩")

    # 1) 标注（不在此剔除）耕地层里与水体层重叠高的块
    #    原口径「重叠 ≥50% 即剔」会把泡田期水田（此时与水体层几乎 100% 重叠）整片误删。
    #    实测新安镇被删的 2,128 块送 GEE 赋值后有 83.6% 判为水稻 → 改为候选标注，
    #    是否剔交由 stage_gee 用模型结果联合判定。
    fa["water_ratio"] = water_overlap_ratio(fa, po).round(4)
    cand = fa["water_ratio"] >= WATER_RATIO
    clean = fa.copy().reset_index(drop=True)
    log(f"    [水体块标注] 与水体层重叠 ≥{WATER_RATIO} 的 {int(cand.sum())} 块 / "
        f"{fa.geometry.area[cand].sum()*MU:.1f} 亩（不在此剔除，改由赋值后联合判定）")

    # 2) 水体细分 鱼塘/水体
    po["面积/亩"] = (po.geometry.area * MU).round(4)
    po["squareness"] = [round(squareness(g), 4) for g in po.geometry]
    is_p = (po["squareness"] >= SQUARE_MIN) & (po["面积/亩"] <= AREA_MAX_MU)
    po["kind"] = np.where(is_p, "鱼塘", "水体")
    n1 = int((po["kind"] == "鱼塘").sum())
    log(f"    [水体细分] 鱼塘 {n1} / 水体 {len(po)-n1}")

    # 3) 村界归属
    clean = assign_villages(clean, villages)
    cleaned_kind = assign_villages(po, villages)
    miss = int((clean["village_code"].astype(str) == "").sum())
    log(f"    [村界归属] 耕地无归属 {miss} 块")

    clean.to_file(clean_p, driver="GPKG", encoding="utf-8")
    cleaned_kind.to_file(kind_p, driver="GPKG", encoding="utf-8")
    clean.to_file(vill_p, driver="GPKG", encoding="utf-8")
    return clean_p, kind_p, vill_p


# --------------------------------------------------------------------------- #
# 阶段 2：GEE 作物赋值
# --------------------------------------------------------------------------- #
def _cloud_missing_fids(feat, phases, k=2):
    """返回仍缺 ≥k 旬有效像元的 fid 集合（判定口径与 run_predict 的 missing_rule 一致）。"""
    cols = [f"{p}_B8" for p in phases if f"{p}_B8" in feat.columns]
    if not cols:
        return set()
    miss = feat[cols].isna().sum(axis=1)
    return set(feat.loc[miss >= k, "fid"].tolist())


def _merge_feat(base, patch):
    """用 patch 覆盖 base 中同 fid 的行（列对齐），返回新 DataFrame。"""
    a = base.set_index("fid")
    b = patch.set_index("fid").reindex(columns=a.columns)
    a.loc[b.index] = b
    return a.reset_index()


def stage_gee(out_dir, prefix, vill_p, force=False, wide=False,
              missing_rule="two", fallback_years=1):
    from predict_dachun_gee import WIDE_WINDOWS, WINDOWS, extract_features, run_predict
    from make_abc_sam3 import build_c

    gdir = os.path.join(out_dir, "作物赋值")
    os.makedirs(gdir, exist_ok=True)
    base = os.path.join(gdir, prefix)
    if not force and os.path.exists(base + "_B_质检.gpkg"):
        log(f"    [gee] 已存在，跳过")
        return base

    with open(MODEL, "rb") as f:
        bundle = pickle.load(f)
    phases = bundle.get("phases", ["D04", "D11", "D14"])
    gdf = gpd.read_file(vill_p).reset_index(drop=True)
    gdf["fid"] = range(len(gdf))
    gw = gdf.to_crs("EPSG:4326")
    gw["geometry"] = gw.geometry.simplify(tolerance=0.0001, preserve_topology=True)
    windows = WIDE_WINDOWS if wide else WINDOWS
    log(f"    [gee] {len(gdf)} 块，窗口={'WIDE' if wide else 'KEY'}，"
        f"missing_rule={missing_rule}，回退年数={fallback_years}")

    # 1) 主年份提取
    feat = extract_features(gw, windows=windows, year=PRIMARY_YEAR)
    if feat.empty:
        raise RuntimeError("GEE 无有效特征")
    feat["src_year"] = PRIMARY_YEAR

    # 2) 云缺失兜底：仍缺 ≥2 旬的块，回退上一年同旬窗口重提
    for k in range(1, fallback_years + 1):
        y = PRIMARY_YEAR - k
        need = _cloud_missing_fids(feat, phases, k=2)
        if not need:
            break
        log(f"    [gee] 云缺失 {len(need)} 块 → 回退 {y} 年同旬重提")
        f2 = extract_features(gw[gw["fid"].isin(need)], windows=windows, year=y)
        if f2.empty:
            continue
        f2["src_year"] = y
        feat = _merge_feat(feat, f2)
        log(f"    [gee] {y} 年后仍缺 ≥2 旬 {len(_cloud_missing_fids(feat, phases, k=2))} 块")

    out = run_predict(gdf, feat, bundle, missing_rule=missing_rule)

    # 3) 候选水体块联合判定：重叠高 且 模型未判为作物 → 才剔
    #    避免把泡田期（5月上旬）水田整体误删；重叠高但判为水稻/玉米的块保留。
    if "water_ratio" in out.columns:
        wr = pd.to_numeric(out["water_ratio"], errors="coerce").fillna(0)
        cand = wr >= WATER_RATIO
        is_crop = out["prdct_cf"].isin(CROP_LABELS)
        is_water = cand & ~is_crop
        keep_bak = int((cand & is_crop).sum())
        log(f"    [剔水体块] 重叠候选 {int(cand.sum())} 块 → 模型判作物保留 {keep_bak} 块，"
            f"最终剔除 {int(is_water.sum())} 块")
        out = out[~is_water].reset_index(drop=True)

    A = out[["id", "class", "area_m2", "prdct_cf", "conf_flag", "max_prob",
             "px_statu", "geometry"]].copy() if "id" in out.columns else \
        out[["class", "area_m2", "prdct_cf", "conf_flag", "max_prob", "px_statu", "geometry"]].copy()
    A.to_file(base + "_A_赋值.gpkg", driver="GPKG", encoding="utf-8")
    out.to_file(base + "_B_质检.gpkg", driver="GPKG", encoding="utf-8")
    build_c(out).to_file(base + "_C_交付.gpkg", driver="GPKG", encoding="utf-8")
    log("    [gee] 作物分布:\n" + out["prdct_cf"].value_counts().to_string())
    log("    [gee] 影像年份:\n" + out["src_year"].value_counts().to_string())
    return base


# --------------------------------------------------------------------------- #
# 阶段 3：交付（xlsx + gpkg + shp）
# --------------------------------------------------------------------------- #
def _drop_unassigned(path, out_path, tag):
    """交付要求每行都有 12 位村编码；无村归属的退化碎片（如三元镇 1 块 0.13 m²）
    无法编码，直接从交付输入剔除并记录。"""
    g = gpd.read_file(path)
    if "village_code" not in g.columns:
        return path
    bad = g["village_code"].astype(str).str.strip() == ""
    if not bad.any():
        return path
    a = g.geometry.area[bad].sum()
    log(f"    [deliver] {tag} 剔除无村归属 {int(bad.sum())} 块 / {a:.2f} m²")
    g[~bad].reset_index(drop=True).to_file(out_path, driver="GPKG", encoding="utf-8")
    return out_path


def stage_deliver(out_dir, prefix, b_ck, kind_p=None, force=False):
    """kind_p 为 None 时不并入水体层（源数据无水体层时用）。"""
    ddir = os.path.join(out_dir, "交付")
    os.makedirs(ddir, exist_ok=True)
    shp_p = os.path.join(ddir, f"{prefix}_交付.shp")
    if not force and os.path.exists(shp_p):
        log(f"    [deliver] 已存在，跳过")
        return ddir

    b_ck = _drop_unassigned(b_ck, b_ck.replace(".gpkg", "_deliv.gpkg"), "耕地")
    if kind_p:
        kind_p = _drop_unassigned(kind_p, kind_p.replace(".gpkg", "_deliv.gpkg"), "水体")

    cmd = [sys.executable, EXPORT, "--farmland", b_ck]
    if kind_p:
        cmd += ["--pond", kind_p]
    cmd += ["--village-lib", VILL, "--out-dir", ddir, "--prefix", prefix, "--also-gpkg"]
    subprocess.run(cmd, check=True)

    # gpkg 镜像 → 交付 shp（字段名转英文缩写，WKT 文本列省去，几何即边界）
    g = gpd.read_file(os.path.join(ddir, f"{prefix}_交付.gpkg")).to_crs(TARGET_CRS)
    g = g.rename(columns=SHORT)
    g = g[[c for c in SHORT.values() if c in g.columns] + ["geometry"]]
    g.to_file(shp_p, driver="ESRI Shapefile", encoding="UTF-8")
    log(f"    [deliver] 交付 shp -> {shp_p}")

    # 旧的三类交付件已被合并替代，产出成功后清理，避免混淆
    removed, locked = 0, []
    for t in LEGACY_TAGS:
        for ext in LEGACY_EXT:
            p = os.path.join(ddir, f"{prefix}_{t}{ext}")
            if not os.path.exists(p):
                continue
            try:
                os.remove(p)
                removed += 1
            except OSError:
                locked.append(os.path.basename(p))
    msg = f"    [deliver] 清理旧三类交付件 {removed} 个"
    if locked:
        msg += f"；{len(locked)} 个被占用未删: {locked}"
    log(msg)
    return ddir


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "gee", "deliver", "all"], default="all")
    ap.add_argument("--only", help="只跑匹配的 OUT/AOI 子串")
    ap.add_argument("--force", action="store_true", help="忽略已有产物重跑")
    ap.add_argument("--wide", action="store_true", help="GEE 用加宽窗口")
    ap.add_argument("--missing-rule", choices=["any", "two"], default="two",
                    help="云缺失口径：any=缺任一旬即降级(旧)；two=缺 ≥2 旬才降级(默认)")
    ap.add_argument("--fallback-years", type=int, default=1,
                    help="仍缺 ≥2 旬的块回退上一年同旬重提的年数（默认 1）")
    ap.add_argument("--dry-run", action="store_true", help="只列 AOI 与源，不作业")
    args = ap.parse_args()

    aois = list_aois(args.only)
    log(f"[范围] {len(aois)} 个 AOI")

    if args.dry_run:
        for o, a, ap_ in aois:
            f, fh = _resolve(os.path.join(ap_, "farmland"), "farmland")
            p, ph = _resolve(os.path.join(ap_, "pond"), "pond")
            log(f"  {o}/{a}: farmland={fh}  pond={ph}")
        return

    villages = load_villages()
    log(f"[村界] {len(villages)} 个村（绵阳片区）")

    ok, fail = [], []
    for i, (o, a, ap_) in enumerate(aois, 1):
        log(f"\n[{i}/{len(aois)}] {o}/{a}")
        out_dir = os.path.join(OUT_ROOT, o, a)
        os.makedirs(out_dir, exist_ok=True)
        try:
            fsrc, _ = _resolve(os.path.join(ap_, "farmland"), "farmland")
            if fsrc is None:
                raise RuntimeError("无耕地层")
            if isinstance(fsrc, list):
                fsrc = fsrc[0]
            g0 = gpd.read_file(fsrc, rows=2000).to_crs(TARGET_CRS)
            town = dominant_town(villages, g0) if villages is not None else None
            prefix = town or a
            log(f"    prefix={prefix}")

            vill_p = os.path.join(out_dir, "前序检查", f"{prefix}_耕地_带村界.gpkg")
            clean_p = os.path.join(out_dir, "前序检查", f"{prefix}_farmland_clean.gpkg")
            kind_p = os.path.join(out_dir, "前序检查", f"{prefix}_pond_kind.gpkg")

            if args.stage in ("prep", "all"):
                clean_p, kind_p, vill_p = stage_prep(
                    out_dir, a, ap_, villages, prefix, force=args.force)
            if args.stage in ("gee", "all"):
                if not os.path.exists(vill_p):
                    raise RuntimeError("缺 带村界 gpkg，先跑 prep")
                stage_gee(out_dir, prefix, vill_p, force=args.force, wide=args.wide,
                          missing_rule=args.missing_rule,
                          fallback_years=args.fallback_years)
            if args.stage in ("deliver", "all"):
                b_ck = os.path.join(out_dir, "作物赋值", f"{prefix}_B_质检.gpkg")
                if not os.path.exists(b_ck):
                    raise RuntimeError("缺 B_质检 gpkg，先跑 gee")
                stage_deliver(out_dir, prefix, b_ck, kind_p, force=args.force)
            ok.append(f"{o}/{a}")
        except Exception as e:
            log(f"    !! 失败: {type(e).__name__}: {e}")
            traceback.print_exc()
            fail.append(f"{o}/{a}: {e}")

    log(f"\n=== 完成 {len(ok)} / 失败 {len(fail)} ===")
    for f in fail:
        log("  失败 " + f)
    log(f"[输出根] {OUT_ROOT}")


if __name__ == "__main__":
    main()
